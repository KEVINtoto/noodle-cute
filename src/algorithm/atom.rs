//! Pure-safe, device-independent CuTe Copy & MMA atoms, thread–value tilings, fragment
//! retiling, explicit copy policies, and register-reuse MMA GEMM (`#![no_std]`, zero `unsafe`).
//!
//! An atom trait ([`CopyTraits`] / [`MmaTraits`]) describes a single instruction's static
//! `(thread, value) -> coord` layouts and register-level execution contract. Architecture-specific
//! instruction sets implement these same traits in downstream backend crates while reusing all
//! layout partitioning, TV-map derivation, and fragment retiling defined here.
//!
//! # Main Interfaces
//!
//! - **Register encoding**: [`SIZEOF_BITS`] and [`RegisterCodec`] (zero-`unsafe` 1-to-1 and
//!   native-endian multi-element packing into wider integer registers).
//! - **Copy atoms & tilings** ([`mod@copy`]): [`CopyTraits`], [`UniversalCopy`], [`CopyAtom`],
//!   [`AtomValLayouts`], [`TiledCopy`], [`make_tiled_copy`], [`ThrValMap`], [`ThrCopy`],
//!   [`PredicatedCopy`], [`CopyPolicy`], [`copy_with`], and [`copy_if_with`].
//! - **MMA atoms, tilings & retiling** ([`mod@mma`]): [`MmaTraits`], [`UniversalFma`],
//!   [`MmaAtom`], [`TiledMma`], [`make_tiled_mma`], [`OperandA`], [`OperandB`], [`OperandC`],
//!   [`MmaOperand`], [`ThrFrgOf`], [`ThrMma`], [`OperandCoord`], [`PartitionOperand`],
//!   [`MmaCopyLayout`], [`make_tiled_copy_a`], [`make_tiled_copy_b`], [`make_tiled_copy_c`],
//!   [`Retile`], [`Rank`], [`RankOf`], [`GemmOperand`], [`GemmOperandMut`], [`gemm_mma`], and
//!   [`gemm_fma`].
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algorithm::{
//!     atom::{
//!         CopyAtom, MmaAtom, UniversalCopy, UniversalFma, make_tiled_copy_c, make_tiled_mma,
//!     },
//!     copy_with,
//! };
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::tensor::make_tensor;
//! use noodle_cute::{int_tuple, tuple};
//!
//! let mma = make_tiled_mma(
//!     MmaAtom::<UniversalFma<f32>>::new(),
//!     Layout::from_shape(int_tuple![Int::<2>, Int::<2>]),
//!     tuple![..],
//! );
//! let tiled_copy = make_tiled_copy_c(CopyAtom::<UniversalCopy<f32>, f32>::new(), &mma);
//! let data = [7.0f32; 16];
//! let tensor = make_tensor(&data, int_tuple![Int::<4>, Int::<4>]);
//! let mut fragment = mma.get_slice(1i32).partition_fragment_c(tensor);
//! let thr_copy = tiled_copy.get_slice(1i32);
//! let src = thr_copy.partition_s(tensor);
//! let mut dst = thr_copy.retile_d(fragment.view_mut());
//! copy_with(&tiled_copy, &src, &mut dst);
//! assert_eq!(fragment(0i32), 7.0);
//! ```
//!
//! # Notes
//!
//! - All atom and tiling layouts are strictly static; partitioned tensors may carry dynamic
//!   extents/strides, coordinate-basis strides ([`identity_tensor`](crate::tensor::identity_tensor)),
//!   or `F2`-swizzled strides.
//! - Every operation is pure `#![no_std]` with zero `unsafe`; aliased accumulator accumulation
//!   (`D` aliasing `C`) is supported safely through [`CellAccessor`](crate::tensor::CellAccessor)
//!   views or [`gemm_fma`].

pub mod copy;
pub mod mma;

use crate::NoodleCuteError;
use crate::integer::{Int, int};
use crate::layout::{Layout, LayoutBase};
use crate::stride::PrefixProduct;
use crate::tuple::{GetOutput, TupleProduct};

pub use copy::{
    AtomValLayouts, CopyAtom, CopyPolicy, CopyTraits, PredicatedCopy, ThrCopy, ThrPartition,
    ThrValMap, Tidfrg, TiledCopy, UniversalCopy, copy_if_with, copy_with, make_tiled_copy,
};
pub use mma::{
    GemmOperand, GemmOperandMut, MmaAtom, MmaCopyLayout, MmaCopyTile, MmaOperand, MmaPartition,
    MmaTraits, OperandA, OperandB, OperandC, OperandCoord, PartitionOperand, PermMode, Rank,
    RankOf, Retile, ThrFrg, ThrFrgOf, ThrMma, TiledMma, UniversalFma, gemm_fma, gemm_mma,
    make_tiled_copy_a, make_tiled_copy_b, make_tiled_copy_c, make_tiled_mma,
};

/// CuTe `sizeof_bits_v<T>` as signed [`int`].
///
/// Rejects zero-sized types or types whose bit width overflows [`int`] at compile time.
pub const SIZEOF_BITS<T>: int = {
    let bytes = size_of::<T>();
    NoodleCuteError::InvalidShape.check_with(
        bytes > 0 && bytes <= (int::MAX as usize) / 8,
        "atom element type must be non-zero-sized and fit in int bits",
    );
    (bytes * 8) as int
};

/// Static total size of top-level mode `I` of layout `L` (`size<I>(L)`).
pub type ModeSize<L, const I: int> =
    <GetOutput<<L as LayoutBase>::Shape, I> as TupleProduct>::Output;

/// Column-major compact layout type for shape `S`.
#[doc(hidden)]
pub type ColMajorLayout<S> = Layout<S, <S as PrefixProduct<Int<1>>>::Stride>;

/// Pure-safe encoding between logical tensor elements (`Self`) and instruction registers (`Reg`).
///
/// - Any `T: Copy` implements `RegisterCodec<T>` with `ELEMS_PER_REG = 1` (identity).
/// - Standard scalar element types also implement `RegisterCodec<Reg>` for wider unsigned
///   integer registers (`u16`, `u32`, `u64`, `u128`) using native-endian byte packing with
///   zero `unsafe` and no alignment requirements.
pub trait RegisterCodec<Reg: Copy>: Copy {
    /// Number of `Self` elements packed into one `Reg` register.
    const ELEMS_PER_REG: int;

    /// Packs `Self::ELEMS_PER_REG` consecutive elements read via `load(0..ELEMS_PER_REG)` into one `Reg`.
    fn pack_reg(load: impl FnMut(int) -> Self) -> Reg;

    /// Unpacks `reg` into `Self::ELEMS_PER_REG` consecutive elements written via `store(i, val)`.
    fn unpack_reg(reg: Reg, store: impl FnMut(int, Self));
}

impl<T: Copy> RegisterCodec<T> for T {
    const ELEMS_PER_REG: int = 1;

    fn pack_reg(mut load: impl FnMut(int) -> Self) -> T {
        load(0)
    }

    fn unpack_reg(reg: T, mut store: impl FnMut(int, Self)) {
        store(0, reg);
    }
}

macro_rules! impl_register_pack {
    ($elem:ty => $($reg:ty),+ $(,)?) => {
        $(
            impl RegisterCodec<$reg> for $elem {
                const ELEMS_PER_REG: int = (size_of::<$reg>() / size_of::<$elem>()) as int;

                fn pack_reg(mut load: impl FnMut(int) -> Self) -> $reg {
                    const REG_BYTES: usize = size_of::<$reg>();
                    const ELEM_BYTES: usize = size_of::<$elem>();
                    const COUNT: usize = REG_BYTES / ELEM_BYTES;
                    let mut bytes = [0u8; REG_BYTES];
                    let mut i = 0;
                    while i < COUNT {
                        let chunk = load(i as int).to_ne_bytes();
                        let start = i * ELEM_BYTES;
                        bytes[start..start + ELEM_BYTES].copy_from_slice(&chunk);
                        i += 1;
                    }
                    <$reg>::from_ne_bytes(bytes)
                }

                fn unpack_reg(reg: $reg, mut store: impl FnMut(int, Self)) {
                    const REG_BYTES: usize = size_of::<$reg>();
                    const ELEM_BYTES: usize = size_of::<$elem>();
                    const COUNT: usize = REG_BYTES / ELEM_BYTES;
                    let bytes = reg.to_ne_bytes();
                    let mut i = 0;
                    while i < COUNT {
                        let start = i * ELEM_BYTES;
                        let mut chunk = [0u8; ELEM_BYTES];
                        chunk.copy_from_slice(&bytes[start..start + ELEM_BYTES]);
                        store(i as int, <$elem>::from_ne_bytes(chunk));
                        i += 1;
                    }
                }
            }
        )+
    };
}

impl_register_pack!(u8 => u16, u32, u64, u128);
impl_register_pack!(i8 => u16, u32, u64, u128);
impl_register_pack!(u16 => u32, u64, u128);
impl_register_pack!(i16 => u32, u64, u128);
impl_register_pack!(u32 => u64, u128);
impl_register_pack!(i32 => u64, u128);
impl_register_pack!(f32 => u32, u64, u128);
impl_register_pack!(u64 => u128);
impl_register_pack!(i64 => u128);
impl_register_pack!(f64 => u64, u128);
