//! MMA atoms, thread–value MMA tilings, MMA-compatible copy layouts, fragment retiling,
//! and 5-rank register-reuse MMA GEMM (`MmaTraits`, `UniversalFma`, `MmaAtom`, `TiledMma`,
//! `ThrMma`, `make_tiled_copy_{a,b,c}`, `Retile`, `gemm_mma`, `gemm_fma`).
//!
//! # Main Interfaces
//!
//! - [`MmaTraits`], [`UniversalFma`], and [`MmaAtom`]: Pure-safe MMA instruction layout
//!   contracts and register-closure FMA execution (zero `unsafe`).
//! - [`TiledMma`], [`make_tiled_mma`], [`PermMode`], [`OperandA`]/[`OperandB`]/[`OperandC`],
//!   [`MmaOperand`], [`ThrFrgOf`], [`ThrMma`], and [`PartitionOperand`]: Thread–fragment
//!   partitioning (`partition_a`/`b`/`c`) and owning register fragment allocation
//!   (`partition_fragment_a`/`b`/`c`).
//! - [`MmaCopyLayout`], [`make_tiled_copy_a`], [`make_tiled_copy_b`], [`make_tiled_copy_c`],
//!   and [`Retile`] (`ThrCopy::retile_s`/`retile_d`): Derives MMA-matched [`TiledCopy`] TV
//!   maps and retiles register fragments without moving memory.
//! - [`Rank`], [`RankOf`], [`GemmOperand`], [`GemmOperandMut`], [`gemm_mma`], and
//!   [`gemm_fma`]: 5-rank fragment GEMM dispatch with CuTe's register-reuse serpentine order.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algorithm::atom::{MmaAtom, UniversalFma, make_tiled_mma};
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::tensor::make_tensor;
//! use noodle_cute::{int_tuple, tuple};
//!
//! let mma = make_tiled_mma(
//!     MmaAtom::<UniversalFma<f32>>::new(),
//!     Layout::from_shape(int_tuple![Int::<4>, Int::<2>]),
//!     tuple![..],
//! );
//! let c = [0f32; 8 * 4];
//! let tile = make_tensor(&c, int_tuple![Int::<8>, Int::<4>]);
//! // Thread 5 is (m, n) = (1, 1): rows 1 and 5 of columns 1 and 3.
//! let mine = mma.get_slice(5i32).partition_c(tile);
//! assert_eq!(mine.shape(), int_tuple![Int::<1>, Int::<2>, Int::<2>]);
//! assert_eq!(mine.accessor.offset_index(), 1 + 8);
//! ```

use core::marker::PhantomData;
use core::ops::{Add, Mul, RangeFull};

use crate::NoodleCuteError;
use crate::StaticValue;
use crate::algebra::common::FragmentLike;
use crate::algebra::{
    CoalesceProfile, ComposeTile, Composition, LogicalDivide, LogicalProduct, RightInverse,
    TiledProduct, ZippedDivide, ZippedDivideAt, right_inverse,
};
use crate::basis::{Mode, ModeEnd};
use crate::integer::{ArithMul, Int, Integer, Product, add, int, mul, rem};
use crate::layout::{GetFlatCoord, Layout, LayoutBase, OffsetAndSlice, Recast};
use crate::shape::{Shape, ShapeScalar};
use crate::stride::{PrefixProduct, Stride};
use crate::tensor::{
    AccessorOffset, OwnedStorage, Tensor, TensorIndex, TensorSet, make_fragment_like,
    make_owned_tensor,
};
use crate::tuple::{
    Get, GetOutput, LeafProfile, Nil, NilProfile, PadModes, ProductEach, RepeatModes, Til, Tuple,
    TupleProduct, TupleProfile, Wrap, Zip2, product_each,
};

use super::copy::{AtomValLayouts, CopyAtom, CopyTraits, ThrCopy, TiledCopy, ValRef};
use super::{ColMajorLayout, ModeSize, RegisterCodec};

// ------------------------------------------------------------------
// MmaTraits, UniversalFma, MmaAtom
// ------------------------------------------------------------------

/// One MMA instruction: CuTe `MMA_Traits<MMAOperation>` with pure-safe register execution.
///
/// `ALayout`, `BLayout`, and `CLayout` map `(thread, value)` to `(m, k)`, `(n, k)`, and
/// `(m, n)` within the static `ShapeMnk`.
pub trait MmaTraits {
    /// Logical element type of operand A.
    type ValA;
    /// Logical element type of operand B.
    type ValB;
    /// Logical element type of accumulator C/D.
    type ValC;
    /// Static `(M, N, K)` shape of one instruction.
    type ShapeMnk;
    /// Logical thread ID to physical thread index layout.
    type ThrId;
    /// `(thread, value) -> (m, k)` layout.
    type ALayout;
    /// `(thread, value) -> (n, k)` layout.
    type BLayout;
    /// `(thread, value) -> (m, n)` layout.
    type CLayout;
    /// Register type of output D.
    type DRegister: Copy;
    /// Register type of input A.
    type ARegister: Copy;
    /// Register type of input B.
    type BRegister: Copy;
    /// Register type of accumulator C.
    type CRegister: Copy;
    /// Number of D registers per instruction.
    const D_REGISTERS: int;
    /// Number of A registers per instruction.
    const A_REGISTERS: int;
    /// Number of B registers per instruction.
    const B_REGISTERS: int;
    /// Number of C registers per instruction.
    const C_REGISTERS: int;

    /// Computes `D = A * B + C` on the instruction's registers.
    fn fma(
        d: impl FnMut(int, Self::DRegister),
        a: impl FnMut(int) -> Self::ARegister,
        b: impl FnMut(int) -> Self::BRegister,
        c: impl FnMut(int) -> Self::CRegister,
    );
}

/// Scalar fused multiply-add on a single thread: CuTe `UniversalFMA<D, A, B, C>`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UniversalFma<D, A = D, B = A, C = D>(PhantomData<(D, A, B, C)>);

type OneThread = Layout<Int<1>, Int<0>>;
type Unit2 = Layout<Til<Int<1>, Til<Int<1>, Nil>>, Til<Int<0>, Til<Int<0>, Nil>>>;

impl<D: Copy, A: Copy, B: Copy, C: Copy> MmaTraits for UniversalFma<D, A, B, C>
where
    A: Mul<B>,
    A::Output: Add<C, Output = D>,
{
    type ValA = A;
    type ValB = B;
    type ValC = C;
    type ShapeMnk = Til<Int<1>, Til<Int<1>, Til<Int<1>, Nil>>>;
    type ThrId = OneThread;
    type ALayout = Unit2;
    type BLayout = Unit2;
    type CLayout = Unit2;
    type DRegister = D;
    type ARegister = A;
    type BRegister = B;
    type CRegister = C;
    const D_REGISTERS: int = 1;
    const A_REGISTERS: int = 1;
    const B_REGISTERS: int = 1;
    const C_REGISTERS: int = 1;

    fn fma(
        mut d: impl FnMut(int, D),
        mut a: impl FnMut(int) -> A,
        mut b: impl FnMut(int) -> B,
        mut c: impl FnMut(int) -> C,
    ) {
        d(0, a(0) * b(0) + c(0));
    }
}

/// An MMA instruction wrapper: CuTe `MMA_Atom<MMAOperation>`.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct MmaAtom<Op>(PhantomData<Op>);

impl<Op> Clone for MmaAtom<Op> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Op> Copy for MmaAtom<Op> {}

impl<Op> MmaAtom<Op> {
    /// Constructs the zero-sized MMA atom for `Op`.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Op: MmaTraits> MmaAtom<Op> {
    /// Executes one MMA instruction `D = A * B + C` over register fragments
    /// (CuTe `MMA_Atom::call`).
    pub fn call<AD, SD, DD, AA, SA, DA, AB, SB, DB, AC, SC, DC, TD, TA, TB, TC>(
        &self,
        d: &mut Tensor<AD, Layout<SD, DD>>,
        a: &Tensor<AA, Layout<SA, DA>>,
        b: &Tensor<AB, Layout<SB, DB>>,
        c: &Tensor<AC, Layout<SC, DC>>,
    ) where
        SD: Shape,
        DD: Stride<SD>,
        SA: Shape,
        DA: Stride<SA>,
        SB: Shape,
        DB: Stride<SB>,
        SC: Shape,
        DC: Stride<SC>,
        Tensor<AD, Layout<SD, DD>>: TensorSet<int, Item = TD>,
        Tensor<AA, Layout<SA, DA>>: TensorIndex<int, Output = TA>,
        Tensor<AB, Layout<SB, DB>>: TensorIndex<int, Output = TB>,
        Tensor<AC, Layout<SC, DC>>: TensorIndex<int, Output = TC>,
        TD: RegisterCodec<Op::DRegister>,
        TA: RegisterCodec<Op::ARegister>,
        TB: RegisterCodec<Op::BRegister>,
        TC: RegisterCodec<Op::CRegister>,
    {
        let pd = TD::ELEMS_PER_REG;
        let pa = TA::ELEMS_PER_REG;
        let pb = TB::ELEMS_PER_REG;
        let pc = TC::ELEMS_PER_REG;
        Op::fma(
            |r, reg| {
                let base = mul(r, pd);
                TD::unpack_reg(reg, |e, val| d.set(add(base, e), val));
            },
            |r| {
                let base = mul(r, pa);
                TA::pack_reg(|e| a.at(add(base, e)))
            },
            |r| {
                let base = mul(r, pb);
                TB::pack_reg(|e| b.at(add(base, e)))
            },
            |r| {
                let base = mul(r, pc);
                TC::pack_reg(|e| c.at(add(base, e)))
            },
        );
    }
}

// ------------------------------------------------------------------
// TiledMma, PermMode, make_tiled_mma
// ------------------------------------------------------------------

type Pair<A, B> = Til<A, Til<B, Nil>>;
type Mnk<P0, P1, P2> = Til<P0, Til<P1, Til<P2, Nil>>>;
type Vmnk4<V, M, N, K> = Til<V, Mnk<M, N, K>>;
type Three = Mnk<Int<1>, Int<1>, Int<1>>;

/// MMA atoms tiled over threads: CuTe `TiledMMA<MMA_Atom, AtomLayoutMNK, PermutationMNK>`.
///
/// `thr_layout_vmnk` is `tiled_product(ThrId, padded(thr_layout, 1:0))`, mapping
/// `(atom thread, m, n, k)` to a thread index. `permutation_mnk` holds the resolved M, N, K
/// tilers (`..` entries already expanded to `size(atom shape) * size(thread mode)`).
#[derive(Debug, PartialEq, Eq)]
pub struct TiledMma<Op, Vmnk, Perm> {
    /// Underlying MMA atom.
    pub atom: MmaAtom<Op>,
    /// `(V, M, N, K) -> thread index`.
    pub thr_layout_vmnk: Vmnk,
    /// Resolved `(PermM, PermN, PermK)` tilers.
    pub permutation_mnk: Perm,
}

impl<Op, Vmnk: Clone, Perm: Clone> Clone for TiledMma<Op, Vmnk, Perm> {
    fn clone(&self) -> Self {
        Self {
            atom: self.atom,
            thr_layout_vmnk: self.thr_layout_vmnk.clone(),
            permutation_mnk: self.permutation_mnk.clone(),
        }
    }
}

impl<Op, Vmnk: Copy, Perm: Copy> Copy for TiledMma<Op, Vmnk, Perm> {}

type PadThrShape<TS> = <<TS as Wrap>::Wrapped as PadModes<Three, Int<1>>>::Output;
type PadThrStride<TD> = <<TD as Wrap>::Wrapped as PadModes<Three, Int<0>>>::Output;
type PadThr<TS, TD> = Layout<PadThrShape<TS>, PadThrStride<TD>>;
type Vmnk<Op, TS, TD> = <<Op as MmaTraits>::ThrId as TiledProduct<PadThr<TS, TD>>>::Output;
type VmnkShape<V> = <V as LayoutBase>::Shape;
type ThrSize<V, const I: int> = <GetOutput<VmnkShape<V>, I> as TupleProduct>::Output;
type AtomSize<Op, const I: int> =
    <GetOutput<<Op as MmaTraits>::ShapeMnk, I> as TupleProduct>::Output;
type PaddedPerm<P> = <<P as Wrap>::Wrapped as PadModes<Three, RangeFull>>::Output;

/// Resolves a permutation mode (`..` becomes `A * T`, while static `Int<N>` and `Layout` stay
/// unchanged) for [`make_tiled_mma`].
pub const trait PermMode<A, T>: Copy {
    /// Resolved permutation tiler type.
    type Output: Copy;

    /// Resolves `self`.
    fn resolve(self) -> Self::Output;
}

const impl<A, T> PermMode<A, T> for RangeFull
where
    A: [const] Integer + StaticValue + [const] ArithMul<A, T>,
    T: [const] Integer + StaticValue,
    Product<A, T>: Copy,
{
    type Output = Product<A, T>;

    fn resolve(self) -> Self::Output {
        mul(A::VALUE, T::VALUE)
    }
}

const impl<A, T, const N: int> PermMode<A, T> for Int<N> {
    type Output = Self;

    fn resolve(self) -> Self {
        self
    }
}

const impl<A, T, S: Copy, D: Copy> PermMode<A, T> for Layout<S, D> {
    type Output = Self;

    fn resolve(self) -> Self {
        self
    }
}

type PermOut<Op, V, P, const I: int, const J: int> =
    <GetOutput<PaddedPerm<P>, I> as PermMode<AtomSize<Op, I>, ThrSize<V, J>>>::Output;
type Perms<Op, V, P> =
    Mnk<PermOut<Op, V, P, 0, 1>, PermOut<Op, V, P, 1, 2>, PermOut<Op, V, P, 2, 3>>;

/// Resulting [`TiledMma`] type returned by [`make_tiled_mma`].
pub type MadeMma<Op, TS, TD, P> = TiledMma<Op, Vmnk<Op, TS, TD>, Perms<Op, Vmnk<Op, TS, TD>, P>>;

/// Constructs a [`TiledMma`] arranging `atom` across `thr_layout` and `permutations`
/// (CuTe `make_tiled_mma(atom, thr_layout, permutations)`).
///
/// `thr_layout` is padded with `1:0` modes to rank 3 `(M, N, K)`. `permutations` accepts
/// `..`, `Nil`, or a 1–3 mode tuple of `..` / `Int<N>` / `Layout` entries, padded with `..`
/// to rank 3 where each `..` resolves to `atom_extent * thr_extent`.
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::algorithm::atom::{MmaAtom, UniversalFma, make_tiled_mma};
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
/// use noodle_cute::{int_tuple, tuple};
///
/// let atom = MmaAtom::<UniversalFma<f32>>::new();
/// let mma = make_tiled_mma(atom, Layout::from_shape(int_tuple![Int::<16>, Int::<16>]), tuple![..]);
/// assert_eq!(mma.thr_layout_vmnk.shape, int_tuple![Int::<1>, Int::<16>, Int::<16>, Int::<1>]);
/// assert_eq!(mma.permutation_mnk, int_tuple![Int::<16>, Int::<16>, Int::<1>]);
/// ```
pub const fn make_tiled_mma<Op, TS, TD, P>(
    atom: MmaAtom<Op>,
    thr_layout: Layout<TS, TD>,
    permutations: P,
) -> MadeMma<Op, TS, TD, P>
where
    Op: MmaTraits,
    Op::ThrId: StaticValue + [const] TiledProduct<PadThr<TS, TD>>,
    Op::ShapeMnk: Get<0> + Get<1> + Get<2>,
    GetOutput<Op::ShapeMnk, 0>: TupleProduct,
    GetOutput<Op::ShapeMnk, 1>: TupleProduct,
    GetOutput<Op::ShapeMnk, 2>: TupleProduct,
    TS: Shape + [const] Wrap,
    TD: Stride<TS> + [const] Wrap,
    TS::Wrapped: [const] PadModes<Three, Int<1>>,
    TD::Wrapped: [const] PadModes<Three, Int<0>>,
    PadThrShape<TS>: Shape,
    PadThrStride<TD>: Stride<PadThrShape<TS>>,
    Vmnk<Op, TS, TD>: LayoutBase,
    VmnkShape<Vmnk<Op, TS, TD>>: Get<1> + Get<2> + Get<3>,
    GetOutput<VmnkShape<Vmnk<Op, TS, TD>>, 1>: TupleProduct,
    GetOutput<VmnkShape<Vmnk<Op, TS, TD>>, 2>: TupleProduct,
    GetOutput<VmnkShape<Vmnk<Op, TS, TD>>, 3>: TupleProduct,
    P: [const] Wrap,
    P::Wrapped: [const] PadModes<Three, RangeFull>,
    PaddedPerm<P>: [const] Get<0> + [const] Get<1> + [const] Get<2>,
    GetOutput<PaddedPerm<P>, 0>: [const] PermMode<AtomSize<Op, 0>, ThrSize<Vmnk<Op, TS, TD>, 1>>,
    GetOutput<PaddedPerm<P>, 1>: [const] PermMode<AtomSize<Op, 1>, ThrSize<Vmnk<Op, TS, TD>, 2>>,
    GetOutput<PaddedPerm<P>, 2>: [const] PermMode<AtomSize<Op, 2>, ThrSize<Vmnk<Op, TS, TD>, 3>>,
{
    let thr_mnk = Layout::new(
        thr_layout.shape.wrap().pad_modes(Int::<1>),
        thr_layout.stride.wrap().pad_modes(Int::<0>),
    );
    let thr_layout_vmnk = Op::ThrId::VALUE.tiled_product(thr_mnk);
    let perms = permutations.wrap().pad_modes(..);
    let p0 = Get::<0>::get(&perms).resolve();
    let p1 = Get::<1>::get(&perms).resolve();
    let p2 = Get::<2>::get(&perms).resolve();
    TiledMma {
        atom,
        thr_layout_vmnk,
        permutation_mnk: Til(p0, Til(p1, Til(p2, Nil))),
    }
}

// ------------------------------------------------------------------
// Operand markers, MmaOperand, ThrFrgOf, ThrMma, PartitionOperand
// ------------------------------------------------------------------

/// Marker selecting the `(M, K)` operand A of an MMA.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OperandA;

/// Marker selecting the `(N, K)` operand B of an MMA.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OperandB;

/// Marker selecting the `(M, N)` accumulator C/D of an MMA.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OperandC;

/// Extracts `(perm, atom_tile, tv, thr_tile)` for operand `W` (`OperandA`/`B`/`C`) of a [`TiledMma`].
pub const trait MmaOperand<W>: Copy {
    /// Two-mode permutation tiler for operand `W`.
    type Perm: Copy;
    /// Atom extents in operand `W`'s two modes.
    type AtomTile: Copy;
    /// Atom `(thread, value) -> coord` layout for operand `W`.
    type Tv: Copy;
    /// Thread atom counts in operand `W`'s two modes.
    type ThrTile: Copy;

    /// Returns `(perm, atom_tile, tv, thr_tile)`.
    fn operand(&self) -> (Self::Perm, Self::AtomTile, Self::Tv, Self::ThrTile);
}

macro_rules! impl_mma_operand {
    ($w:ident: perms $p0:literal $p1:literal, atom $m0:literal $m1:literal, $tv:ident, threads $t0:literal $t1:literal) => {
        const impl<Op, V, P> MmaOperand<$w> for TiledMma<Op, V, P>
        where
            Op: MmaTraits,
            V: Copy + LayoutBase,
            P: Copy + [const] Get<$p0> + [const] Get<$p1>,
            Op::ShapeMnk: Get<$m0> + Get<$m1>,
            GetOutput<Op::ShapeMnk, $m0>: TupleProduct,
            GetOutput<Op::ShapeMnk, $m1>: TupleProduct,
            AtomSize<Op, $m0>: StaticValue,
            AtomSize<Op, $m1>: StaticValue,
            Op::$tv: StaticValue,
            VmnkShape<V>: Get<$t0> + Get<$t1>,
            GetOutput<VmnkShape<V>, $t0>: TupleProduct,
            GetOutput<VmnkShape<V>, $t1>: TupleProduct,
            ThrSize<V, $t0>: StaticValue,
            ThrSize<V, $t1>: StaticValue,
        {
            type Perm = Pair<GetOutput<P, $p0>, GetOutput<P, $p1>>;
            type AtomTile = Pair<AtomSize<Op, $m0>, AtomSize<Op, $m1>>;
            type Tv = Op::$tv;
            type ThrTile = Pair<ThrSize<V, $t0>, ThrSize<V, $t1>>;

            fn operand(&self) -> (Self::Perm, Self::AtomTile, Self::Tv, Self::ThrTile) {
                let perm = Til(
                    Get::<$p0>::get(&self.permutation_mnk),
                    Til(Get::<$p1>::get(&self.permutation_mnk), Nil),
                );
                (
                    perm,
                    StaticValue::VALUE,
                    StaticValue::VALUE,
                    StaticValue::VALUE,
                )
            }
        }
    };
}

impl_mma_operand!(OperandA: perms 0 2, atom 0 2, ALayout, threads 1 3);
impl_mma_operand!(OperandB: perms 1 2, atom 1 2, BLayout, threads 2 3);
impl_mma_operand!(OperandC: perms 0 1, atom 0 1, CLayout, threads 1 2);

type Mode1 = Mode<1, ModeEnd>;
type Keep<M> = Til<M, Til<RangeFull, Nil>>;
type Permuted<L, M, W> = <L as LogicalDivide<<M as MmaOperand<W>>::Perm>>::Output;
type AtomTiled<L, M, W> =
    <Permuted<L, M, W> as ZippedDivide<<M as MmaOperand<W>>::AtomTile>>::Output;
type OperandTv<L, M, W> =
    <AtomTiled<L, M, W> as ComposeTile<Keep<<M as MmaOperand<W>>::Tv>>>::Output;
type ThrDivided<L, M, W> =
    <OperandTv<L, M, W> as ZippedDivideAt<<M as MmaOperand<W>>::ThrTile, Mode1>>::Output;

/// `((ThrV, (Thr0, Thr1)), (FrgV, (Rest0, Rest1, ...)))` layout produced by [`ThrFrgOf::thrfrg`].
pub type ThrFrg<L, M, W> = <ThrDivided<L, M, W> as Zip2>::Output;

/// CuTe `TiledMMA::thrfrg_A`, `thrfrg_B`, and `thrfrg_C` for operand `W`.
///
/// Divides `layout` by the operand permutation, zipped-divides by the atom tile, composes
/// mode 0 with the atom TV layout, zipped-divides mode 1 by the thread atom counts, and
/// transposes via [`Zip2`] into `((ThrV, (Thr0, Thr1)), (FrgV, (Rest0, Rest1, ...)))`.
pub const trait ThrFrgOf<L, W>: Copy {
    /// Resulting thread–fragment layout.
    type Output: Copy;

    /// Computes the thread–fragment layout of `layout`.
    fn thrfrg(&self, layout: &L) -> Self::Output;
}

const impl<Op, V, P, L, W> ThrFrgOf<L, W> for TiledMma<Op, V, P>
where
    Self: [const] MmaOperand<W>,
    L: Copy + [const] LogicalDivide<<Self as MmaOperand<W>>::Perm>,
    Permuted<L, Self, W>: [const] ZippedDivide<<Self as MmaOperand<W>>::AtomTile>,
    AtomTiled<L, Self, W>: [const] ComposeTile<Keep<<Self as MmaOperand<W>>::Tv>>,
    OperandTv<L, Self, W>: [const] ZippedDivideAt<<Self as MmaOperand<W>>::ThrTile, Mode1>,
    ThrDivided<L, Self, W>: [const] Zip2,
{
    type Output = ThrFrg<L, Self, W>;

    fn thrfrg(&self, layout: &L) -> Self::Output {
        let (perm, atom_tile, tv, thr_tile) = self.operand();
        let tv_layout = (*layout)
            .logical_divide(perm)
            .zipped_divide(atom_tile)
            .compose_tile(Til(tv, Til(.., Nil)));
        ZippedDivideAt::<_, Mode1>::zipped_divide_at(tv_layout, thr_tile).zip2()
    }
}

impl<Op, V: Copy, P> TiledMma<Op, V, P> {
    /// Returns the per-thread partitioner for `thr_idx` (CuTe `TiledMMA::get_slice`).
    pub const fn get_slice<X>(&self, thr_idx: X) -> ThrMma<'_, Op, V, P, <V as GetFlatCoord<X>>::Output>
    where
        V: [const] GetFlatCoord<X>,
    {
        ThrMma {
            tiled: self,
            thr_vmnk: self.thr_layout_vmnk.get_flat_coord(thr_idx),
        }
    }
}

/// One thread's view of a [`TiledMma`]: CuTe `ThrMMA`.
#[derive(Debug)]
pub struct ThrMma<'a, Op, V, P, F> {
    tiled: &'a TiledMma<Op, V, P>,
    /// Thread's flat `(v, m, n, k)` coordinate in `thr_layout_vmnk`.
    pub thr_vmnk: F,
}

impl<Op, V, P, F: Copy> Clone for ThrMma<'_, Op, V, P, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Op, V, P, F: Copy> Copy for ThrMma<'_, Op, V, P, F> {}

/// Projects a thread's `(v, m, n, k)` coordinate to operand `W`'s `(v, (t0, t1))` coordinate.
pub const trait OperandCoord<W>: Copy {
    /// Projected coordinate type.
    type Output: Copy;

    /// Returns `(v, (t0, t1))` for operand `W`.
    fn operand_coord(&self) -> Self::Output;
}

const impl<V: Copy, M: Copy, N: Copy, K: Copy> OperandCoord<OperandA> for Vmnk4<V, M, N, K> {
    type Output = Pair<V, Pair<M, K>>;

    fn operand_coord(&self) -> Self::Output {
        let Til(v, Til(m, Til(_, Til(k, Nil)))) = *self;
        Til(v, Til(Til(m, Til(k, Nil)), Nil))
    }
}

const impl<V: Copy, M: Copy, N: Copy, K: Copy> OperandCoord<OperandB> for Vmnk4<V, M, N, K> {
    type Output = Pair<V, Pair<N, K>>;

    fn operand_coord(&self) -> Self::Output {
        let Til(v, Til(_, Til(n, Til(k, Nil)))) = *self;
        Til(v, Til(Til(n, Til(k, Nil)), Nil))
    }
}

const impl<V: Copy, M: Copy, N: Copy, K: Copy> OperandCoord<OperandC> for Vmnk4<V, M, N, K> {
    type Output = Pair<V, Pair<M, N>>;

    fn operand_coord(&self) -> Self::Output {
        let Til(v, Til(m, Til(n, Til(_, Nil)))) = *self;
        Til(v, Til(Til(m, Til(n, Nil)), Nil))
    }
}

type RestOf<F> =
    <GetOutput<GetOutput<<F as LayoutBase>::Shape, 1>, 1> as RepeatModes<RangeFull>>::Output;
type OperandSlice<F, C> = Pair<C, Pair<RangeFull, RestOf<F>>>;
type OperandFrg<L, Op, V, P, W> = <TiledMma<Op, V, P> as ThrFrgOf<L, W>>::Output;
type OperandAt<Th, W> = <Th as OperandCoord<W>>::Output;

/// One thread's `(FrgV, Rest0, Rest1, ...)` sub-[`Tensor`] for MMA operand `W`.
pub type MmaPartition<A, L, Op, V, P, Th, W> = Tensor<
    <A as AccessorOffset<
        <OperandFrg<L, Op, V, P, W> as OffsetAndSlice<
            OperandSlice<OperandFrg<L, Op, V, P, W>, OperandAt<Th, W>>,
        >>::Offset,
    >>::Output,
    Layout<
        <OperandFrg<L, Op, V, P, W> as OffsetAndSlice<
            OperandSlice<OperandFrg<L, Op, V, P, W>, OperandAt<Th, W>>,
        >>::SlicedShape,
        <OperandFrg<L, Op, V, P, W> as OffsetAndSlice<
            OperandSlice<OperandFrg<L, Op, V, P, W>, OperandAt<Th, W>>,
        >>::SlicedStride,
    >,
>;

/// CuTe `ThrMMA::partition_A`, `partition_B`, and `partition_C` for operand `W`.
pub const trait PartitionOperand<T, W> {
    /// Resulting per-thread partitioned tensor type.
    type Output;

    /// Partitions `tensor` for this thread and operand `W`.
    fn partition_operand(&self, tensor: T) -> Self::Output;
}

const impl<Op, V, P, Th, A, S, D, W> PartitionOperand<Tensor<A, Layout<S, D>>, W>
    for ThrMma<'_, Op, V, P, Th>
where
    S: Shape,
    D: Stride<S>,
    Th: [const] OperandCoord<W>,
    TiledMma<Op, V, P>: [const] ThrFrgOf<Layout<S, D>, W>,
    OperandFrg<Layout<S, D>, Op, V, P, W>: LayoutBase
        + [const] OffsetAndSlice<
            OperandSlice<OperandFrg<Layout<S, D>, Op, V, P, W>, OperandAt<Th, W>>,
        >,
    <OperandFrg<Layout<S, D>, Op, V, P, W> as LayoutBase>::Shape: Get<1>,
    GetOutput<<OperandFrg<Layout<S, D>, Op, V, P, W> as LayoutBase>::Shape, 1>: Get<1>,
    GetOutput<GetOutput<<OperandFrg<Layout<S, D>, Op, V, P, W> as LayoutBase>::Shape, 1>, 1>:
        [const] RepeatModes<RangeFull>,
    A: [const] AccessorOffset<
        <OperandFrg<Layout<S, D>, Op, V, P, W> as OffsetAndSlice<
            OperandSlice<OperandFrg<Layout<S, D>, Op, V, P, W>, OperandAt<Th, W>>,
        >>::Offset,
    >,
{
    type Output = MmaPartition<A, Layout<S, D>, Op, V, P, Th, W>;

    fn partition_operand(&self, tensor: Tensor<A, Layout<S, D>>) -> Self::Output {
        let frg = self.tiled.thrfrg(&tensor.layout);
        let rest = <GetOutput<
            GetOutput<<OperandFrg<Layout<S, D>, Op, V, P, W> as LayoutBase>::Shape, 1>,
            1,
        > as RepeatModes<RangeFull>>::repeat_modes(..);
        let coord = Til(
            self.thr_vmnk.operand_coord(),
            Til(Til(.., Til(rest, Nil)), Nil),
        );
        let (off, sliced) = frg.offset_and_slice(coord);
        Tensor {
            accessor: tensor.accessor.offset(off),
            layout: sliced,
        }
    }
}

type OwnedFrag<T, L> = Tensor<<L as OwnedStorage<T>>::Engine, L>;

impl<Op: MmaTraits, V, P, Th> ThrMma<'_, Op, V, P, Th> {
    /// CuTe `ThrMMA::partition_A`: this thread's `(FrgV, RestM, RestK, ...)` of an `(M, K, ...)` tensor.
    pub const fn partition_a<T>(&self, tensor: T) -> <Self as PartitionOperand<T, OperandA>>::Output
    where
        Self: [const] PartitionOperand<T, OperandA>,
    {
        self.partition_operand(tensor)
    }

    /// CuTe `ThrMMA::partition_B`: this thread's `(FrgV, RestN, RestK, ...)` of an `(N, K, ...)` tensor.
    pub const fn partition_b<T>(&self, tensor: T) -> <Self as PartitionOperand<T, OperandB>>::Output
    where
        Self: [const] PartitionOperand<T, OperandB>,
    {
        self.partition_operand(tensor)
    }

    /// CuTe `ThrMMA::partition_C`: this thread's `(FrgV, RestM, RestN, ...)` of an `(M, N, ...)` tensor.
    pub const fn partition_c<T>(&self, tensor: T) -> <Self as PartitionOperand<T, OperandC>>::Output
    where
        Self: [const] PartitionOperand<T, OperandC>,
    {
        self.partition_operand(tensor)
    }

    /// CuTe `ThrMMA::partition_fragment_A`: allocates an owning register fragment shaped like
    /// [`partition_a`](Self::partition_a) with element type `Op::ValA`.
    pub fn partition_fragment_a<T, A, L>(
        &self,
        tensor: T,
    ) -> OwnedFrag<Op::ValA, <L as FragmentLike>::Output>
    where
        Self: PartitionOperand<T, OperandA, Output = Tensor<A, L>>,
        L: FragmentLike,
        L::Output: OwnedStorage<Op::ValA>,
    {
        make_fragment_like::<Op::ValA, L>(&self.partition_a(tensor).layout)
    }

    /// CuTe `ThrMMA::partition_fragment_B`: allocates an owning register fragment shaped like
    /// [`partition_b`](Self::partition_b) with element type `Op::ValB`.
    pub fn partition_fragment_b<T, A, L>(
        &self,
        tensor: T,
    ) -> OwnedFrag<Op::ValB, <L as FragmentLike>::Output>
    where
        Self: PartitionOperand<T, OperandB, Output = Tensor<A, L>>,
        L: FragmentLike,
        L::Output: OwnedStorage<Op::ValB>,
    {
        make_fragment_like::<Op::ValB, L>(&self.partition_b(tensor).layout)
    }

    /// CuTe `ThrMMA::partition_fragment_C`: allocates a compact column-major owning accumulator
    /// fragment of `Op::ValC` with the shape of [`partition_c`](Self::partition_c).
    pub fn partition_fragment_c<T, A, L>(
        &self,
        tensor: T,
    ) -> OwnedFrag<Op::ValC, ColMajorLayout<L::Shape>>
    where
        Self: PartitionOperand<T, OperandC, Output = Tensor<A, L>>,
        L: LayoutBase,
        L::Shape: Shape + PrefixProduct<Int<1>>,
        <L::Shape as PrefixProduct<Int<1>>>::Stride: Stride<L::Shape>,
        ColMajorLayout<L::Shape>: StaticValue + OwnedStorage<Op::ValC>,
    {
        let _ = self.partition_c(tensor);
        make_owned_tensor(<ColMajorLayout<L::Shape> as StaticValue>::VALUE)
    }
}

// ------------------------------------------------------------------
// MMA-compatible copy layouts (make_tiled_copy_{a,b,c}) & Retile
// ------------------------------------------------------------------

/// Static extent of one resolved MMA permutation mode (`Int<N>` or `Layout<S, D>`).
#[doc(hidden)]
pub trait TileExtent {
    type Size;
}

impl<const N: int> TileExtent for Int<N> {
    type Size = Self;
}

impl<S: TupleProduct, D> TileExtent for Layout<S, D> {
    type Size = S::Output;
}

type PermOf<M, W> = <M as MmaOperand<W>>::Perm;
type ExtentOf<M, W, const I: int> = <GetOutput<PermOf<M, W>, I> as TileExtent>::Size;

/// Two-dimensional static tile extents `(Extent0, Extent1)` of MMA operand `W`.
pub type MmaCopyTile<M, W> = Pair<ExtentOf<M, W, 0>, ExtentOf<M, W, 1>>;

type RefLayout<M, W> = ColMajorLayout<MmaCopyTile<M, W>>;
type RefFrg<M, W> = <M as ThrFrgOf<RefLayout<M, W>, W>>::Output;
type Sizes4<V> = Vmnk4<ThrSize<V, 0>, ThrSize<V, 1>, ThrSize<V, 2>, ThrSize<V, 3>>;

/// Projects a `(V, M, N, K)` thread coordinate to operand `W`'s linearized thread coordinate.
#[doc(hidden)]
pub trait CopyProjection<V> {
    type Layout;
}

macro_rules! impl_copy_projection {
    ($w:ty, $d1:ty, $d2:ty, $d3:ty) => {
        impl<V: LayoutBase> CopyProjection<V> for $w
        where
            VmnkShape<V>: Get<0> + Get<1> + Get<2> + Get<3>,
            GetOutput<VmnkShape<V>, 0>: TupleProduct,
            GetOutput<VmnkShape<V>, 1>: TupleProduct,
            GetOutput<VmnkShape<V>, 2>: TupleProduct,
            GetOutput<VmnkShape<V>, 3>: TupleProduct,
            ThrSize<V, 0>: ArithMul<ThrSize<V, 0>, ThrSize<V, 1>>
                + ArithMul<ThrSize<V, 0>, ThrSize<V, 2>>,
        {
            type Layout = Layout<Sizes4<V>, Vmnk4<Int<1>, $d1, $d2, $d3>>;
        }
    };
}

impl_copy_projection!(
    OperandA,
    ThrSize<V, 0>,
    Int<0>,
    Product<ThrSize<V, 0>, ThrSize<V, 1>>
);
impl_copy_projection!(
    OperandB,
    Int<0>,
    ThrSize<V, 0>,
    Product<ThrSize<V, 0>, ThrSize<V, 2>>
);
impl_copy_projection!(
    OperandC,
    ThrSize<V, 0>,
    Product<ThrSize<V, 0>, ThrSize<V, 1>>,
    Int<0>
);

type ThreadMap<V, W> =
    <<W as CopyProjection<V>>::Layout as Composition<<V as RightInverse>::Output>>::Output;
type MappedFrg<M, V, W> = <RefFrg<M, W> as ComposeTile<Keep<ThreadMap<V, W>>>>::Output;
type TvProfile = TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>;

/// CuTe `get_layoutA_TV`, `get_layoutB_TV`, and `get_layoutC_TV` for operand `W`.
pub const trait MmaCopyLayout<W>: Copy {
    /// `(thread index, value index) -> operand tile index` layout.
    type Tv: Copy;
    /// Two-dimensional operand tile extents.
    type Tile: Copy;

    /// Returns `(layout_tv, tile)`.
    fn copy_layout(&self) -> (Self::Tv, Self::Tile);
}

const impl<Op, V, P, W> MmaCopyLayout<W> for TiledMma<Op, V, P>
where
    Self: [const] MmaOperand<W> + [const] ThrFrgOf<RefLayout<Self, W>, W>,
    PermOf<Self, W>: Get<0> + Get<1>,
    GetOutput<PermOf<Self, W>, 0>: TileExtent,
    GetOutput<PermOf<Self, W>, 1>: TileExtent,
    MmaCopyTile<Self, W>: Shape + PrefixProduct<Int<1>> + StaticValue,
    <MmaCopyTile<Self, W> as PrefixProduct<Int<1>>>::Stride: Stride<MmaCopyTile<Self, W>>,
    RefLayout<Self, W>: StaticValue,
    V: RightInverse,
    W: CopyProjection<V>,
    <W as CopyProjection<V>>::Layout: Composition<<V as RightInverse>::Output>,
    ThreadMap<V, W>: StaticValue,
    RefFrg<Self, W>: [const] ComposeTile<Keep<ThreadMap<V, W>>>,
    MappedFrg<Self, V, W>: [const] CoalesceProfile<TvProfile>,
{
    type Tv = <MappedFrg<Self, V, W> as CoalesceProfile<TvProfile>>::Output;
    type Tile = MmaCopyTile<Self, W>;

    fn copy_layout(&self) -> (Self::Tv, Self::Tile) {
        let frg = self.thrfrg(&RefLayout::<Self, W>::VALUE);
        let tv = frg
            .compose_tile(Til(ThreadMap::<V, W>::VALUE, Til(.., Nil)))
            .coalesce_with_profile();
        (tv, Self::Tile::VALUE)
    }
}

/// Compile-time divisibility check ensuring an atom's thread/value counts divide the MMA TV layout.
#[doc(hidden)]
pub const COPY_TILE_DIVISIBLE<T: Integer, V: Integer, AT: Integer, AV: Integer>: bool = match (
    T::STATIC_VALUE,
    V::STATIC_VALUE,
    AT::STATIC_VALUE,
    AV::STATIC_VALUE,
) {
    (Some(t), Some(v), Some(at), Some(av)) => at > 0 && av > 0 && rem(t, at) == 0 && rem(v, av) == 0,
    _ => false,
};

/// Compile-time guard rejecting copy atoms whose thread/value counts do not divide the MMA TV layout.
#[diagnostic::on_unimplemented(
    message = "copy atom thread/value counts do not evenly divide the MMA TV layout",
    label = "incompatible copy atom for this TiledMma operand",
    note = "ensure atom thread count divides TV threads and atom value count divides TV values"
)]
#[doc(hidden)]
pub const trait ValidCopyTile<const OK: bool> {}
const impl ValidCopyTile<true> for () {}

macro_rules! impl_mma_copy {
    ($function:ident, $method:ident, $w:ty, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Atoms whose thread or value count does not divide the operand TV layout fail at
        /// compile time:
        ///
        /// ```compile_fail,E0271
        /// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
        /// # #![allow(incomplete_features)]
        /// use noodle_cute::algorithm::atom::{
        ///     CopyAtom, MmaAtom, UniversalCopy, UniversalFma, make_tiled_copy_c, make_tiled_mma,
        /// };
        /// use noodle_cute::integer::Int;
        /// use noodle_cute::layout::Layout;
        /// use noodle_cute::{int_tuple, tuple};
        ///
        /// let mma = make_tiled_mma(
        ///     MmaAtom::<UniversalFma<f32>>::new(),
        ///     Layout::from_shape(int_tuple![Int::<2>, Int::<2>]),
        ///     tuple![..],
        /// );
        /// // 128-bit copy needs 4 f32 values per thread, but the MMA only has 1 per thread.
        /// let _ = make_tiled_copy_c(CopyAtom::<UniversalCopy<u128>, f32>::new(), &mma);
        /// ```
        pub const fn $function<Op, V, M>(
            atom: CopyAtom<Op, V>,
            mma: &M,
        ) -> TiledCopy<Op, V, M::Tv, M::Tile>
        where
            Op: CopyTraits,
            M: [const] MmaCopyLayout<$w>,
            CopyAtom<Op, V>: AtomValLayouts,
            M::Tv: LayoutBase,
            <M::Tv as LayoutBase>::Shape: Get<0> + Get<1>,
            GetOutput<<M::Tv as LayoutBase>::Shape, 0>: TupleProduct,
            GetOutput<<M::Tv as LayoutBase>::Shape, 1>: TupleProduct,
            ModeSize<M::Tv, 0>: Integer,
            ModeSize<M::Tv, 1>: Integer,
            ValRef<Op, V>: LayoutBase,
            <ValRef<Op, V> as LayoutBase>::Shape: Get<0> + Get<1>,
            GetOutput<<ValRef<Op, V> as LayoutBase>::Shape, 0>: TupleProduct,
            GetOutput<<ValRef<Op, V> as LayoutBase>::Shape, 1>: TupleProduct,
            ModeSize<ValRef<Op, V>, 0>: Integer,
            ModeSize<ValRef<Op, V>, 1>: Integer,
            (): [const] ValidCopyTile<
                {
                    COPY_TILE_DIVISIBLE::<
                        ModeSize<M::Tv, 0>,
                        ModeSize<M::Tv, 1>,
                        ModeSize<ValRef<Op, V>, 0>,
                        ModeSize<ValRef<Op, V>, 1>,
                    >
                },
            >,
        {
            let (layout_tv, tiler) = mma.copy_layout();
            TiledCopy {
                atom,
                layout_tv,
                tiler,
            }
        }

        impl<Op, V, P> TiledMma<Op, V, P> {
            /// Returns the MMA operand's `(thread index, value index) -> tile index` layout.
            pub const fn $method(&self) -> <Self as MmaCopyLayout<$w>>::Tv
            where
                Self: [const] MmaCopyLayout<$w>,
            {
                <Self as MmaCopyLayout<$w>>::copy_layout(self).0
            }
        }
    };
}

impl_mma_copy!(
    make_tiled_copy_a,
    get_layout_a_tv,
    OperandA,
    "CuTe `make_tiled_copy_A`: copy tiling matching the MMA A thread–value map."
);
impl_mma_copy!(
    make_tiled_copy_b,
    get_layout_b_tv,
    OperandB,
    "CuTe `make_tiled_copy_B`: copy tiling matching the MMA B thread–value map."
);
impl_mma_copy!(
    make_tiled_copy_c,
    get_layout_c_tv,
    OperandC,
    "CuTe `make_tiled_copy_C`: copy tiling matching the MMA C thread–value map."
);

// --- Retile ---

/// Extracts the tail modes `1..` of a non-empty [`Til`] tuple.
#[doc(hidden)]
pub trait TailModes {
    type Rest: Tuple;
}

impl<H, T: Tuple> TailModes for Til<H, T> {
    type Rest = T;
}

type FrgValues<L> = GetOutput<<L as LayoutBase>::Shape, 0>;
type VCount<L> = <FrgValues<L> as TupleProduct>::Output;

/// Compile-time upcast factor `threads * fragment_mode0_size` used by [`Retile`].
#[doc(hidden)]
pub const RETILE_FACTOR<T: Integer, V: Integer>: int = match (T::STATIC_VALUE, V::STATIC_VALUE) {
    (Some(t), Some(v)) => mul(t, v),
    _ => NoodleCuteError::Unsupported.panic_with("retile requires a static reference fragment"),
};

type InverseTile<TV, T> =
    <<TV as RightInverse>::Output as Composition<ColMajorLayout<T>>>::Output;
type FrgMn<TV, T, L> =
    <InverseTile<TV, T> as Recast<Int<{ RETILE_FACTOR::<ModeSize<TV, 0>, VCount<L>> }>>>::Output;
type VLayout<L> = Layout<VCount<L>, Int<1>>;
type FrgProduct<TV, T, L> =
    <VLayout<L> as LogicalProduct<<FrgMn<TV, T, L> as RightInverse>::Output>>::Output;
type RetileV<TV, T, L, Op, V> =
    <FrgProduct<TV, T, L> as ZippedDivide<Layout<ModeSize<ValRef<Op, V>, 1>, Int<1>>>>::Output;
type RetileTile<TV, T, L> =
    Til<VCount<L>, <<FrgMn<TV, T, L> as LayoutBase>::Shape as ProductEach>::Output>;
type RetileT<TV, T, L> = <L as ZippedDivide<RetileTile<TV, T, L>>>::Output;
type RetileMapped<TV, T, L, Op, V> =
    <RetileT<TV, T, L> as ComposeTile<Keep<RetileV<TV, T, L, Op, V>>>>::Output;
type RestShape<L> = <<L as LayoutBase>::Shape as TailModes>::Rest;
type RetileRest<L> = <RestShape<L> as RepeatModes<RangeFull>>::Output;
type RetileCoord<L> = Pair<RangeFull, Til<Int<0>, RetileRest<L>>>;

/// CuTe `TiledCopy::retile`: re-views a register fragment layout in copy-instruction order
/// without moving any data.
pub const trait Retile<L>: Copy {
    /// Retiled fragment [`Layout`] type.
    type Output: Copy;

    /// Computes the retiled layout of `layout`.
    fn retile(&self, layout: &L) -> Self::Output;
}

const impl<Op, V, TV, T, L> Retile<L> for TiledCopy<Op, V, TV, T>
where
    Op: CopyTraits,
    CopyAtom<Op, V>: AtomValLayouts,
    ValRef<Op, V>: LayoutBase,
    <ValRef<Op, V> as LayoutBase>::Shape: Get<1>,
    GetOutput<<ValRef<Op, V> as LayoutBase>::Shape, 1>: TupleProduct,
    ModeSize<ValRef<Op, V>, 1>: ShapeScalar + StaticValue,
    TV: Copy + LayoutBase + [const] RightInverse,
    TV::Shape: Get<0>,
    GetOutput<TV::Shape, 0>: TupleProduct,
    ModeSize<TV, 0>: Integer,
    T: Copy + Shape + PrefixProduct<Int<1>>,
    <T as PrefixProduct<Int<1>>>::Stride: Stride<T>,
    ColMajorLayout<T>: StaticValue,
    <TV as RightInverse>::Output: [const] Composition<ColMajorLayout<T>>,
    L: Copy + LayoutBase + [const] ZippedDivide<RetileTile<TV, T, L>>,
    L::Shape: Get<0> + TailModes,
    FrgValues<L>: TupleProduct,
    VCount<L>: ShapeScalar + StaticValue,
    InverseTile<TV, T>: [const] Recast<Int<{ RETILE_FACTOR::<ModeSize<TV, 0>, VCount<L>> }>>,
    FrgMn<TV, T, L>: [const] LayoutBase + [const] RightInverse,
    <FrgMn<TV, T, L> as LayoutBase>::Shape: [const] ProductEach,
    <<FrgMn<TV, T, L> as LayoutBase>::Shape as ProductEach>::Output: Copy,
    VLayout<L>: StaticValue + [const] LogicalProduct<<FrgMn<TV, T, L> as RightInverse>::Output>,
    FrgProduct<TV, T, L>: [const] ZippedDivide<Layout<ModeSize<ValRef<Op, V>, 1>, Int<1>>>,
    RetileT<TV, T, L>: [const] ComposeTile<Keep<RetileV<TV, T, L, Op, V>>>,
    RestShape<L>: [const] RepeatModes<RangeFull>,
    RetileMapped<TV, T, L, Op, V>: [const] OffsetAndSlice<RetileCoord<L>>,
{
    type Output = Layout<
        <RetileMapped<TV, T, L, Op, V> as OffsetAndSlice<RetileCoord<L>>>::SlicedShape,
        <RetileMapped<TV, T, L, Op, V> as OffsetAndSlice<RetileCoord<L>>>::SlicedStride,
    >;

    fn retile(&self, layout: &L) -> Self::Output {
        let inverse = right_inverse(self.layout_tv);
        let frg_mn = inverse
            .composition(ColMajorLayout::<T>::VALUE)
            .recast(Int::<{ RETILE_FACTOR::<ModeSize<TV, 0>, VCount<L>> }>);
        let tile = Til(VCount::<L>::VALUE, product_each(&frg_mn.shape()));
        let product = VLayout::<L>::VALUE.logical_product(right_inverse(frg_mn));
        let frg_v = product.zipped_divide(Layout::new(
            ModeSize::<ValRef<Op, V>, 1>::VALUE,
            Int::<1>,
        ));
        let mapped = (*layout)
            .zipped_divide(tile)
            .compose_tile(Til(frg_v, Til(.., Nil)));
        let coord = Til(
            ..,
            Til(
                Til(Int::<0>, <RestShape<L> as RepeatModes<RangeFull>>::repeat_modes(..)),
                Nil,
            ),
        );
        let (_, sliced) = mapped.offset_and_slice(coord);
        sliced
    }
}

type RetiledTensor<A, L, Op, V, TV, T> = Tensor<A, <TiledCopy<Op, V, TV, T> as Retile<L>>::Output>;

impl<Op, V, TV, T, X> ThrCopy<'_, Op, V, TV, T, X> {
    /// CuTe `ThrCopy::retile_S`: re-views a register fragment in copy-instruction order
    /// without moving data.
    pub const fn retile_s<A: Copy, L: Copy>(
        &self,
        tensor: Tensor<A, L>,
    ) -> RetiledTensor<A, L, Op, V, TV, T>
    where
        TiledCopy<Op, V, TV, T>: [const] Retile<L>,
    {
        Tensor {
            accessor: tensor.accessor,
            layout: self.tiled.retile(&tensor.layout),
        }
    }

    /// CuTe `ThrCopy::retile_D`: the same reference-fragment transformation as [`retile_s`](Self::retile_s).
    pub const fn retile_d<A: Copy, L: Copy>(
        &self,
        tensor: Tensor<A, L>,
    ) -> RetiledTensor<A, L, Op, V, TV, T>
    where
        TiledCopy<Op, V, TV, T>: [const] Retile<L>,
    {
        self.retile_s(tensor)
    }
}

// ------------------------------------------------------------------
// 5-rank register-reuse GEMM: Rank, RankOf, GemmOperand, gemm_mma, gemm_fma
// ------------------------------------------------------------------

/// Rank marker (`1`, `2`, or `3`) selecting the CuTe GEMM dispatch mode.
pub struct Rank<const N: usize>;

/// Associates a shape or [`Tensor`] with its [`Rank`] marker for [`gemm_mma`] / [`gemm_fma`].
pub trait RankOf {
    /// Rank marker (`Rank<1>`, `Rank<2>`, or `Rank<3>`).
    type Rank;
}

impl RankOf for int {
    type Rank = Rank<1>;
}

impl<const N: int> RankOf for Int<N> {
    type Rank = Rank<1>;
}

impl<A> RankOf for Til<A, Nil> {
    type Rank = Rank<1>;
}

impl<A, B> RankOf for Til<A, Til<B, Nil>> {
    type Rank = Rank<2>;
}

impl<A, B, C> RankOf for Til<A, Til<B, Til<C, Nil>>> {
    type Rank = Rank<3>;
}

impl<Acc, S: RankOf, D> RankOf for Tensor<Acc, Layout<S, D>> {
    type Rank = S::Rank;
}

type Coord2 = Til<int, Til<int, Nil>>;
type Coord3 = Til<int, Til<int, Til<int, Nil>>>;

/// Internal rank-dispatched element read/write helper keyed by operand rank and accumulator rank `RC`.
#[doc(hidden)]
pub trait OperandKindRead<RC, Acc, S, D> {
    type Elem: Copy;
    fn extents(layout: &Layout<S, D>) -> [int; 3];
    fn load_vxy(tensor: &Tensor<Acc, Layout<S, D>>, v: int, x: int, y: int) -> Self::Elem;
}

/// Internal rank-dispatched mutable store helper keyed by operand rank and accumulator rank `RC`.
#[doc(hidden)]
pub trait OperandKindWrite<RC, Acc, S, D> {
    type Elem: Copy;
    fn extents_mut(layout: &Layout<S, D>) -> [int; 3];
    fn store_vxy(tensor: &mut Tensor<Acc, Layout<S, D>>, v: int, x: int, y: int, val: Self::Elem);
}

// [1] (V) with rank-1 accumulator: (V) x (V) => (V).
impl<Acc, S, D, E> OperandKindRead<Rank<1>, Acc, S, D> for Rank<1>
where
    S: Shape,
    D: Stride<S>,
    E: Copy,
    Tensor<Acc, Layout<S, D>>: TensorIndex<int, Output = E>,
{
    type Elem = E;
    fn extents(layout: &Layout<S, D>) -> [int; 3] {
        [layout.shape.size(), 1, 1]
    }
    fn load_vxy(tensor: &Tensor<Acc, Layout<S, D>>, v: int, _: int, _: int) -> E {
        tensor.at(v)
    }
}

impl<Acc, S, D, E> OperandKindWrite<Rank<1>, Acc, S, D> for Rank<1>
where
    S: Shape,
    D: Stride<S>,
    E: Copy,
    Tensor<Acc, Layout<S, D>>: TensorSet<int, Item = E>,
{
    type Elem = E;
    fn extents_mut(layout: &Layout<S, D>) -> [int; 3] {
        [layout.shape.size(), 1, 1]
    }
    fn store_vxy(tensor: &mut Tensor<Acc, Layout<S, D>>, v: int, _: int, _: int, val: E) {
        tensor.set(v, val);
    }
}

// [2] (M) or (N) with rank-2 accumulator: (M) x (N) => (M, N).
impl<Acc, S, D, E> OperandKindRead<Rank<2>, Acc, S, D> for Rank<1>
where
    S: Shape,
    D: Stride<S>,
    E: Copy,
    Tensor<Acc, Layout<S, D>>: TensorIndex<int, Output = E>,
{
    type Elem = E;
    fn extents(layout: &Layout<S, D>) -> [int; 3] {
        [1, layout.shape.size(), 1]
    }
    fn load_vxy(tensor: &Tensor<Acc, Layout<S, D>>, _: int, x: int, _: int) -> E {
        tensor.at(x)
    }
}

// [2]/[3] (M, K), (N, K), or (M, N) with rank-2 accumulator.
impl<Acc, S0, S1, D, E> OperandKindRead<Rank<2>, Acc, Til<S0, Til<S1, Nil>>, D> for Rank<2>
where
    S0: Shape,
    S1: Shape,
    D: Stride<Til<S0, Til<S1, Nil>>>,
    E: Copy,
    Tensor<Acc, Layout<Til<S0, Til<S1, Nil>>, D>>: TensorIndex<Coord2, Output = E>,
{
    type Elem = E;
    fn extents(layout: &Layout<Til<S0, Til<S1, Nil>>, D>) -> [int; 3] {
        [1, layout.shape.0.size(), layout.shape.1.0.size()]
    }
    fn load_vxy(tensor: &Tensor<Acc, Layout<Til<S0, Til<S1, Nil>>, D>>, _: int, x: int, y: int) -> E {
        tensor.at(Til(x, Til(y, Nil)))
    }
}

impl<Acc, S0, S1, D, E> OperandKindWrite<Rank<2>, Acc, Til<S0, Til<S1, Nil>>, D> for Rank<2>
where
    S0: Shape,
    S1: Shape,
    D: Stride<Til<S0, Til<S1, Nil>>>,
    E: Copy,
    Tensor<Acc, Layout<Til<S0, Til<S1, Nil>>, D>>: TensorSet<Coord2, Item = E>,
{
    type Elem = E;
    fn extents_mut(layout: &Layout<Til<S0, Til<S1, Nil>>, D>) -> [int; 3] {
        [1, layout.shape.0.size(), layout.shape.1.0.size()]
    }
    fn store_vxy(
        tensor: &mut Tensor<Acc, Layout<Til<S0, Til<S1, Nil>>, D>>,
        _: int,
        x: int,
        y: int,
        val: E,
    ) {
        tensor.set(Til(x, Til(y, Nil)), val);
    }
}

// [4] (V, M) or (V, N) with rank-3 accumulator: (V, M) x (V, N) => (V, M, N).
impl<Acc, S0, S1, D, E> OperandKindRead<Rank<3>, Acc, Til<S0, Til<S1, Nil>>, D> for Rank<2>
where
    S0: Shape,
    S1: Shape,
    D: Stride<Til<S0, Til<S1, Nil>>>,
    E: Copy,
    Tensor<Acc, Layout<Til<S0, Til<S1, Nil>>, D>>: TensorIndex<Coord2, Output = E>,
{
    type Elem = E;
    fn extents(layout: &Layout<Til<S0, Til<S1, Nil>>, D>) -> [int; 3] {
        [layout.shape.0.size(), layout.shape.1.0.size(), 1]
    }
    fn load_vxy(tensor: &Tensor<Acc, Layout<Til<S0, Til<S1, Nil>>, D>>, v: int, x: int, _: int) -> E {
        tensor.at(Til(v, Til(x, Nil)))
    }
}

// [4]/[5] (V, M, K), (V, N, K), or (V, M, N) with rank-3 accumulator.
impl<Acc, S0, S1, S2, D, E> OperandKindRead<Rank<3>, Acc, Til<S0, Til<S1, Til<S2, Nil>>>, D>
    for Rank<3>
where
    S0: Shape,
    S1: Shape,
    S2: Shape,
    D: Stride<Til<S0, Til<S1, Til<S2, Nil>>>>,
    E: Copy,
    Tensor<Acc, Layout<Til<S0, Til<S1, Til<S2, Nil>>>, D>>: TensorIndex<Coord3, Output = E>,
{
    type Elem = E;
    fn extents(layout: &Layout<Til<S0, Til<S1, Til<S2, Nil>>>, D>) -> [int; 3] {
        [
            layout.shape.0.size(),
            layout.shape.1.0.size(),
            layout.shape.1.1.0.size(),
        ]
    }
    fn load_vxy(
        tensor: &Tensor<Acc, Layout<Til<S0, Til<S1, Til<S2, Nil>>>, D>>,
        v: int,
        x: int,
        y: int,
    ) -> E {
        tensor.at(Til(v, Til(x, Til(y, Nil))))
    }
}

impl<Acc, S0, S1, S2, D, E> OperandKindWrite<Rank<3>, Acc, Til<S0, Til<S1, Til<S2, Nil>>>, D>
    for Rank<3>
where
    S0: Shape,
    S1: Shape,
    S2: Shape,
    D: Stride<Til<S0, Til<S1, Til<S2, Nil>>>>,
    E: Copy,
    Tensor<Acc, Layout<Til<S0, Til<S1, Til<S2, Nil>>>, D>>: TensorSet<Coord3, Item = E>,
{
    type Elem = E;
    fn extents_mut(layout: &Layout<Til<S0, Til<S1, Til<S2, Nil>>>, D>) -> [int; 3] {
        [
            layout.shape.0.size(),
            layout.shape.1.0.size(),
            layout.shape.1.1.0.size(),
        ]
    }
    fn store_vxy(
        tensor: &mut Tensor<Acc, Layout<Til<S0, Til<S1, Til<S2, Nil>>>, D>>,
        v: int,
        x: int,
        y: int,
        val: E,
    ) {
        tensor.set(Til(v, Til(x, Til(y, Nil))), val);
    }
}

/// Read-only [`gemm_mma`] / [`gemm_fma`] operand addressed as `(v, x, y)` given accumulator rank `RC`.
pub trait GemmOperand<RC> {
    /// Element value type.
    type Elem: Copy;

    /// Returns `[V, X, Y]` extents (absent modes have extent `1`).
    fn extents(&self) -> [int; 3];

    /// Reads element `(v, x, y)`.
    fn load_vxy(&self, v: int, x: int, y: int) -> Self::Elem;
}

impl<Acc, S: RankOf, D, RC> GemmOperand<RC> for Tensor<Acc, Layout<S, D>>
where
    S::Rank: OperandKindRead<RC, Acc, S, D>,
{
    type Elem = <S::Rank as OperandKindRead<RC, Acc, S, D>>::Elem;

    fn extents(&self) -> [int; 3] {
        <S::Rank as OperandKindRead<RC, Acc, S, D>>::extents(&self.layout)
    }

    fn load_vxy(&self, v: int, x: int, y: int) -> Self::Elem {
        <S::Rank as OperandKindRead<RC, Acc, S, D>>::load_vxy(self, v, x, y)
    }
}

/// Mutable [`gemm_mma`] / [`gemm_fma`] destination operand addressed as `(v, x, y)` given
/// accumulator rank `RC`.
pub trait GemmOperandMut<RC> {
    /// Element value type written to storage.
    type Elem: Copy;

    /// Returns `[V, X, Y]` extents (absent modes have extent `1`).
    fn extents_mut(&self) -> [int; 3];

    /// Writes `val` at `(v, x, y)`.
    fn store_vxy(&mut self, v: int, x: int, y: int, val: Self::Elem);
}

impl<Acc, S: RankOf, D, RC> GemmOperandMut<RC> for Tensor<Acc, Layout<S, D>>
where
    S::Rank: OperandKindWrite<RC, Acc, S, D>,
{
    type Elem = <S::Rank as OperandKindWrite<RC, Acc, S, D>>::Elem;

    fn extents_mut(&self) -> [int; 3] {
        <S::Rank as OperandKindWrite<RC, Acc, S, D>>::extents_mut(&self.layout)
    }

    fn store_vxy(&mut self, v: int, x: int, y: int, val: Self::Elem) {
        <S::Rank as OperandKindWrite<RC, Acc, S, D>>::store_vxy(self, v, x, y, val);
    }
}

/// CuTe's register-reuse serpentine traversal orders over `(M, N)` chosen by operand byte widths.
fn traverse(
    m_count: int,
    n_count: int,
    a_bytes: int,
    b_bytes: int,
    mut f: impl FnMut(int, int),
) {
    match (a_bytes, b_bytes) {
        (8, 8) | (8, 4) => {
            let mut m: int = 0;
            while m < m_count {
                let mut n: int = 0;
                while n < n_count {
                    let ns = if m & 1 != 0 { n_count - 1 - n } else { n };
                    f(m, ns);
                    n = add(n, 1);
                }
                m = add(m, 1);
            }
        }
        (4, 4) => {
            let mut m: int = 0;
            while m < m_count {
                let mut n: int = 0;
                while n < n_count {
                    let ns = if m & 2 != 0 { n_count - 1 - n } else { n };
                    f(m, ns);
                    if m + 1 < m_count {
                        f(add(m, 1), ns);
                    }
                    n = add(n, 1);
                }
                m = add(m, 2);
            }
        }
        _ => {
            let mut n: int = 0;
            while n < n_count {
                let mut m: int = 0;
                while m < m_count {
                    let ms = if n & 1 != 0 { m_count - 1 - m } else { m };
                    f(ms, n);
                    m = add(m, 1);
                }
                n = add(n, 1);
            }
        }
    }
}

/// CuTe `gemm(mma, D, A, B, C)`: pure-safe 5-rank register-fragment MMA GEMM (`D = A * B + C`).
///
/// Dispatches by the ranks of `A`, `B`, `C`, and `D`:
/// - `[1]` `(V) x (V) => (V)`
/// - `[2]` `(M) x (N) => (M, N)`
/// - `[3]` `(M, K) x (N, K) => (M, N)`
/// - `[4]` `(V, M) x (V, N) => (V, M, N)`
/// - `[5]` `(V, M, K) x (V, N, K) => (V, M, N)`
///
/// For each `k`, `(m, n)` instructions execute in CuTe's register-reuse serpentine order.
/// To accumulate in place (`D` aliasing `C`) with zero `unsafe`, copy a
/// [`CellAccessor`](crate::tensor::CellAccessor) view (`let mut d = c; gemm_mma(&mma, &mut d, &a, &b, &c)`)
/// or use [`gemm_fma`] for scalar FMA.
pub fn gemm_mma<Op, D, A, B, C>(_: &MmaAtom<Op>, d: &mut D, a: &A, b: &B, c: &C)
where
    Op: MmaTraits,
    C: RankOf + GemmOperand<C::Rank>,
    D: GemmOperandMut<C::Rank>,
    A: GemmOperand<C::Rank>,
    B: GemmOperand<C::Rank>,
    D::Elem: RegisterCodec<Op::DRegister>,
    A::Elem: RegisterCodec<Op::ARegister>,
    B::Elem: RegisterCodec<Op::BRegister>,
    C::Elem: RegisterCodec<Op::CRegister>,
{
    let pd = D::Elem::ELEMS_PER_REG;
    let pa = A::Elem::ELEMS_PER_REG;
    let pb = B::Elem::ELEMS_PER_REG;
    let pc = C::Elem::ELEMS_PER_REG;
    let [va, m_count, k_count] = a.extents();
    let [vb, n_count, bk_count] = b.extents();
    let [_, cm_count, cn_count] = c.extents();
    let [_, dm_count, dn_count] = d.extents_mut();
    NoodleCuteError::SizeMismatch.check_with(
        m_count == cm_count
            && m_count == dm_count
            && n_count == cn_count
            && n_count == dn_count
            && k_count == bk_count,
        "gemm_mma: operand extent mismatch",
    );
    let a_bytes = mul(va, size_of::<A::Elem>() as int);
    let b_bytes = mul(vb, size_of::<B::Elem>() as int);
    let mut k: int = 0;
    while k < k_count {
        traverse(m_count, n_count, a_bytes, b_bytes, |m, n| {
            Op::fma(
                |r, reg| {
                    let base = mul(r, pd);
                    D::Elem::unpack_reg(reg, |e, val| d.store_vxy(add(base, e), m, n, val));
                },
                |r| {
                    let base = mul(r, pa);
                    A::Elem::pack_reg(|e| a.load_vxy(add(base, e), m, k))
                },
                |r| {
                    let base = mul(r, pb);
                    B::Elem::pack_reg(|e| b.load_vxy(add(base, e), n, k))
                },
                |r| {
                    let base = mul(r, pc);
                    C::Elem::pack_reg(|e| c.load_vxy(add(base, e), m, n))
                },
            );
        });
        k = add(k, 1);
    }
}

/// CuTe `gemm(A, B, C)`: in-place `C += A * B` using [`UniversalFma`] across all 5 fragment
/// dispatch ranks (works directly on both owning [`ArrayEngine`](crate::tensor::ArrayEngine)
/// fragments and mutable views with zero `unsafe`).
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::algorithm::gemm_fma;
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::tensor::make_owned_tensor;
///
/// let mut a = make_owned_tensor::<f32, _>(int_tuple![Int::<2>, Int::<4>]);
/// let mut b = make_owned_tensor::<f32, _>(int_tuple![Int::<3>, Int::<4>]);
/// let mut c = make_owned_tensor::<f32, _>(int_tuple![Int::<2>, Int::<3>]);
/// a.as_mut_slice().iter_mut().enumerate().for_each(|(i, x)| *x = i as f32);
/// b.as_mut_slice().fill(1.0);
/// gemm_fma(&a, &b, &mut c);
/// // c(1, n) = sum_k a(1, k) = 1 + 3 + 5 + 7 = 16.
/// assert_eq!(c(1i32, 2i32), 16.0);
/// ```
pub fn gemm_fma<A, B, C>(a: &A, b: &B, c: &mut C)
where
    C: RankOf + GemmOperand<C::Rank> + GemmOperandMut<C::Rank, Elem = <C as GemmOperand<C::Rank>>::Elem>,
    A: GemmOperand<C::Rank>,
    B: GemmOperand<C::Rank>,
    A::Elem: Mul<B::Elem>,
    <A::Elem as Mul<B::Elem>>::Output: Add<<C as GemmOperand<C::Rank>>::Elem, Output = <C as GemmOperand<C::Rank>>::Elem>,
{
    let [va, m_count, k_count] = a.extents();
    let [vb, n_count, bk_count] = b.extents();
    let [vc, cm_count, cn_count] = c.extents();
    NoodleCuteError::SizeMismatch.check_with(
        va == 1
            && vb == 1
            && vc == 1
            && m_count == cm_count
            && n_count == cn_count
            && k_count == bk_count,
        "gemm_fma: operand extent mismatch",
    );
    let a_bytes = size_of::<A::Elem>() as int;
    let b_bytes = size_of::<B::Elem>() as int;
    let mut k: int = 0;
    while k < k_count {
        traverse(m_count, n_count, a_bytes, b_bytes, |m, n| {
            let cv = c.load_vxy(0, m, n);
            UniversalFma::<
                <C as GemmOperand<C::Rank>>::Elem,
                A::Elem,
                B::Elem,
                <C as GemmOperand<C::Rank>>::Elem,
            >::fma(
                |_, val| c.store_vxy(0, m, n, val),
                |_| a.load_vxy(0, m, k),
                |_| b.load_vxy(0, n, k),
                |_| cv,
            );
        });
        k = add(k, 1);
    }
}
