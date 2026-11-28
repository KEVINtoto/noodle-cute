//! Reference element-wise algorithms over CuTe [`Tensor`]s (Whitepaper §2.6),
//! built on top of [`crate::algorithm::atom`].
//!
//! These algorithms define the exact mathematical semantics of COPY and batched GEMM
//! by delegating scalar execution to [`CopyAtom`] ([`UniversalCopy`] via [`copy_with`])
//! and [`UniversalFma`](super::atom::UniversalFma) (via [`gemm_fma`]), serving both as
//! executable implementations and as the oracle for layout-analyzed optimizations such
//! as [`crate::algorithm::copy()`].
//!
//! # Main Interfaces
//!
//! - [`copy`]: Rank-1 element-wise transfer `dst[i] = src[i]` over flat coordinates
//!   `0..size(dst)` via [`CopyAtom`] and [`copy_with`]. Realizes memcpy, gather, scatter,
//!   broadcast, constant fill, and transpose through layout choice alone.
//! - [`gemm`] / [`Gemm`]: Rank-3 batched matrix multiplication
//!   `C[m, n, l] += A[m, k, l] * B[n, k, l]` and rank-2 unbatched
//!   `(M, K) x (N, K) -> (M, N)` multiplication via [`gemm_fma`].
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algorithm::reference::{copy, gemm};
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::tensor::make_tensor;
//!
//! // Reference COPY (row-major -> col-major transpose):
//! let src_buf = [0i32, 1, 2, 3, 4, 5];
//! let mut dst_buf = [0i32; 6];
//! let src = make_tensor(
//!     &src_buf,
//!     Layout::new(int_tuple![Int::<2>, Int::<3>], int_tuple![Int::<3>, Int::<1>]),
//! );
//! let mut dst = make_tensor(
//!     &mut dst_buf,
//!     Layout::new(int_tuple![Int::<2>, Int::<3>], int_tuple![Int::<1>, Int::<2>]),
//! );
//! copy(src, &mut dst);
//! assert_eq!(dst_buf, [0, 3, 1, 4, 2, 5]);
//!
//! // Reference GEMM (rank-2 unbatched):
//! let a_buf = [1i32, 2, 3, 4];
//! let b_buf = [5i32, 6, 7, 8];
//! let mut c_buf = [0i32; 4];
//! let a = make_tensor(&a_buf, int_tuple![Int::<2>, Int::<2>]);
//! let b = make_tensor(&b_buf, int_tuple![Int::<2>, Int::<2>]);
//! let mut c = make_tensor(&mut c_buf, int_tuple![Int::<2>, Int::<2>]);
//! gemm(a, b, &mut c);
//! assert_eq!(c(0i32, 0i32), 1 * 5 + 3 * 7);
//! ```
//!
//! # Compile-Time Rejections
//!
//! Statically mismatched copy sizes fail at compile time:
//!
//! ```compile_fail,E0271
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algorithm::reference::copy;
//! use noodle_cute::integer::Int;
//! use noodle_cute::tensor::make_tensor;
//!
//! let src_buf = [0i32; 4];
//! let mut dst_buf = [0i32; 8];
//! let src = make_tensor(&src_buf, Int::<4>);
//! let mut dst = make_tensor(&mut dst_buf, Int::<8>);
//! copy(src, &mut dst);
//! ```
//!
//! Mixing rank-2 and rank-3 operands or passing rank-1 operands to [`gemm`] fails at
//! compile time:
//!
//! ```compile_fail,E0308
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algorithm::reference::gemm;
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::tensor::make_tensor;
//!
//! let buf = [0i32; 4];
//! let mut out = [0i32; 4];
//! let a = make_tensor(&buf, int_tuple![Int::<2>, Int::<2>]);
//! let b = make_tensor(&buf, int_tuple![Int::<2>, Int::<2>, Int::<1>]);
//! let mut c = make_tensor(&mut out, int_tuple![Int::<2>, Int::<2>]);
//! gemm(a, b, &mut c);
//! ```
//!
//! Statically mismatched GEMM extents fail at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algorithm::reference::gemm;
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::tensor::make_tensor;
//!
//! let a_buf = [0i32; 6];
//! let b_buf = [0i32; 8];
//! let mut c_buf = [0i32; 6];
//! let a = make_tensor(&a_buf, int_tuple![Int::<2>, Int::<3>]);
//! let b = make_tensor(&b_buf, int_tuple![Int::<2>, Int::<4>]);
//! let mut c = make_tensor(&mut c_buf, int_tuple![Int::<2>, Int::<2>]);
//! gemm(a, b, &mut c);
//! ```
//!
//! # Notes
//!
//! - Dynamic size mismatches panic at runtime in both debug and release builds.
//! - [`copy`] does not detect write-after-write conflicts when `dst` aliases multiple
//!   coordinates to the same address; use [`crate::algorithm::copy()`] to reject stride-0
//!   write-after-write races and drop redundant writes.

use core::ops::{Add, Deref, DerefMut, Mul};

use crate::NoodleCuteError;
use crate::integer::{add, int};
use crate::layout::Layout;
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tensor::{Tensor, TensorIndex, TensorSet};
use crate::tuple::{Nil, Til};

use super::atom::{
    CopyAtom, GemmOperand, GemmOperandMut, Rank, RankOf, UniversalCopy, copy_with, gemm_fma,
};

/// Returns `true` unless `L` and `R` both have statically known, unequal total sizes.
#[doc(hidden)]
pub const STATIC_SIZE_COMPAT<L: Shape, R: Shape>: bool = match (L::STATIC_SIZE, R::STATIC_SIZE) {
    (Some(l), Some(r)) => l == r,
    _ => true,
};

/// Compile-time guard rejecting statically mismatched operand sizes (`COMPAT == true`).
#[doc(hidden)]
pub const trait ValidSizeMatch<const COMPAT: bool> {}
const impl ValidSizeMatch<true> for () {}

/// Reference element-wise COPY: `dst[i] = src[i]` for every flat index `i` in `0..size(dst)`.
///
/// Delegates execution to [`CopyAtom`] with [`UniversalCopy`] via [`copy_with`]. Statically
/// mismatched total sizes are rejected at compile time; dynamic mismatches panic at runtime.
pub fn copy<AS, SS, DS, AD, SD, DD, T>(
    src: Tensor<AS, Layout<SS, DS>>,
    dst: &mut Tensor<AD, Layout<SD, DD>>,
) where
    SS: Shape,
    DS: Stride<SS>,
    SD: Shape,
    DD: Stride<SD>,
    T: Copy,
    (): ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SS, SD> }>,
    Tensor<AS, Layout<SS, DS>>: TensorIndex<int, Output = T>,
    Tensor<AD, Layout<SD, DD>>: TensorSet<int, Item = T>,
{
    let atom = CopyAtom::<UniversalCopy<T>, T>::new();
    copy_with(&atom, &src, dst);
}

/// 2-D flat coordinate `(x, y)` used by rank-2 [`Gemm`].
type Coord2 = Til<int, Til<int, Nil>>;

/// 3-D flat coordinate `(x, y, l)` used by rank-3 batched [`Gemm`].
type Coord3 = Til<int, Til<int, Til<int, Nil>>>;

/// Rank-2 tensor type `(S0, S1) : (D0, D1)`.
type Tensor2<A, S0, S1, D0, D1> = Tensor<A, Layout<Til<S0, Til<S1, Nil>>, Til<D0, Til<D1, Nil>>>>;

/// Rank-3 tensor type `(S0, S1, S2) : (D0, D1, D2)`.
type Tensor3<A, S0, S1, S2, D0, D1, D2> =
    Tensor<A, Layout<Til<S0, Til<S1, Til<S2, Nil>>>, Til<D0, Til<D1, Til<D2, Nil>>>>>;

/// Adapts one fixed batch index `batch` of a rank-3 `(X, Y, L)` tensor `R` as a rank-2
/// [`GemmOperand`] / [`GemmOperandMut`] so batched [`Gemm`] delegates directly to [`gemm_fma`].
struct BatchSlice<R> {
    tensor: R,
    batch: int,
}

impl<R> RankOf for BatchSlice<R> {
    type Rank = Rank<2>;
}

impl<R, Acc, S0, S1, SL, D0, D1, DL, E> GemmOperand<Rank<2>> for BatchSlice<R>
where
    R: Deref<Target = Tensor3<Acc, S0, S1, SL, D0, D1, DL>>,
    S0: Shape,
    S1: Shape,
    SL: Shape,
    D0: Stride<S0>,
    D1: Stride<S1>,
    DL: Stride<SL>,
    E: Copy,
    R::Target: TensorIndex<Coord3, Output = E>,
{
    type Elem = E;

    fn extents(&self) -> [int; 3] {
        [
            1,
            self.tensor.layout.shape.0.size(),
            self.tensor.layout.shape.1.0.size(),
        ]
    }

    fn load_vxy(&self, _: int, x: int, y: int) -> E {
        self.tensor.at(Til(x, Til(y, Til(self.batch, Nil))))
    }
}

impl<R, Acc, S0, S1, SL, D0, D1, DL, E> GemmOperandMut<Rank<2>> for BatchSlice<R>
where
    R: DerefMut<Target = Tensor3<Acc, S0, S1, SL, D0, D1, DL>>,
    S0: Shape,
    S1: Shape,
    SL: Shape,
    D0: Stride<S0>,
    D1: Stride<S1>,
    DL: Stride<SL>,
    E: Copy,
    R::Target: TensorSet<Coord3, Item = E>,
{
    type Elem = E;

    fn extents_mut(&self) -> [int; 3] {
        [
            1,
            self.tensor.layout.shape.0.size(),
            self.tensor.layout.shape.1.0.size(),
        ]
    }

    fn store_vxy(&mut self, _: int, x: int, y: int, val: E) {
        self.tensor.set(Til(x, Til(y, Til(self.batch, Nil))), val);
    }
}

/// Reference batched GEMM trait backing [`gemm`].
///
/// Implemented via [`gemm_fma`] for:
/// - Rank-3 operands `A:(M, K, L)`, `B:(N, K, L)`, `C:(M, N, L)`.
/// - Rank-2 operands `A:(M, K)`, `B:(N, K)`, `C:(M, N)`.
///
/// Each mode `M`, `N`, `K`, `L` may itself be a hierarchical multi-mode.
pub trait Gemm<B, C> {
    /// Accumulates `C[m, n, l] += A[m, k, l] * B[n, k, l]` in place.
    fn gemm(self, b: B, c: &mut C);
}

impl<
    AA,
    SMA,
    SKA,
    SLA,
    DMA,
    DKA,
    DLA,
    AB,
    SNB,
    SKB,
    SLB,
    DNB,
    DKB,
    DLB,
    AC,
    SMC,
    SNC,
    SLC,
    DMC,
    DNC,
    DLC,
    TA,
    TB,
    TC,
> Gemm<Tensor3<AB, SNB, SKB, SLB, DNB, DKB, DLB>, Tensor3<AC, SMC, SNC, SLC, DMC, DNC, DLC>>
    for Tensor3<AA, SMA, SKA, SLA, DMA, DKA, DLA>
where
    SMA: Shape,
    SKA: Shape,
    SLA: Shape,
    DMA: Stride<SMA>,
    DKA: Stride<SKA>,
    DLA: Stride<SLA>,
    SNB: Shape,
    SKB: Shape,
    SLB: Shape,
    DNB: Stride<SNB>,
    DKB: Stride<SKB>,
    DLB: Stride<SLB>,
    SMC: Shape,
    SNC: Shape,
    SLC: Shape,
    DMC: Stride<SMC>,
    DNC: Stride<SNC>,
    DLC: Stride<SLC>,
    (): ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SMA, SMC> }>
        + ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SNB, SNC> }>
        + ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SKA, SKB> }>
        + ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SLA, SLC> }>
        + ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SLB, SLC> }>
        + ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SLA, SLB> }>,
    Self: TensorIndex<Coord3, Output = TA>,
    Tensor3<AB, SNB, SKB, SLB, DNB, DKB, DLB>: TensorIndex<Coord3, Output = TB>,
    Tensor3<AC, SMC, SNC, SLC, DMC, DNC, DLC>:
        TensorIndex<Coord3, Output = TC> + TensorSet<Coord3, Item = TC>,
    TA: Copy + Mul<TB>,
    TB: Copy,
    TC: Copy,
    <TA as Mul<TB>>::Output: Add<TC, Output = TC>,
{
    fn gemm(
        self,
        b: Tensor3<AB, SNB, SKB, SLB, DNB, DKB, DLB>,
        c: &mut Tensor3<AC, SMC, SNC, SLC, DMC, DNC, DLC>,
    ) {
        let size_m = c.layout.shape.0.size();
        let size_n = c.layout.shape.1.0.size();
        let size_k = self.layout.shape.1.0.size();
        let size_l = c.layout.shape.1.1.0.size();

        NoodleCuteError::SizeMismatch.check_with(
            self.layout.shape.0.size() == size_m,
            "gemm: row size mismatch",
        );
        NoodleCuteError::SizeMismatch.check_with(
            b.layout.shape.0.size() == size_n,
            "gemm: column size mismatch",
        );
        NoodleCuteError::SizeMismatch.check_with(
            b.layout.shape.1.0.size() == size_k,
            "gemm: reduction size mismatch",
        );
        NoodleCuteError::SizeMismatch.check_with(
            self.layout.shape.1.1.0.size() == size_l && b.layout.shape.1.1.0.size() == size_l,
            "gemm: batch size mismatch",
        );

        let mut l: int = 0;
        while l < size_l {
            let a_l = BatchSlice {
                tensor: &self,
                batch: l,
            };
            let b_l = BatchSlice {
                tensor: &b,
                batch: l,
            };
            let mut c_l = BatchSlice {
                tensor: &mut *c,
                batch: l,
            };
            gemm_fma(&a_l, &b_l, &mut c_l);
            l = add(l, 1);
        }
    }
}

impl<AA, SMA, SKA, DMA, DKA, AB, SNB, SKB, DNB, DKB, AC, SMC, SNC, DMC, DNC, TA, TB, TC>
    Gemm<Tensor2<AB, SNB, SKB, DNB, DKB>, Tensor2<AC, SMC, SNC, DMC, DNC>>
    for Tensor2<AA, SMA, SKA, DMA, DKA>
where
    SMA: Shape,
    SKA: Shape,
    DMA: Stride<SMA>,
    DKA: Stride<SKA>,
    SNB: Shape,
    SKB: Shape,
    DNB: Stride<SNB>,
    DKB: Stride<SKB>,
    SMC: Shape,
    SNC: Shape,
    DMC: Stride<SMC>,
    DNC: Stride<SNC>,
    (): ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SMA, SMC> }>
        + ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SNB, SNC> }>
        + ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SKA, SKB> }>,
    Self: TensorIndex<Coord2, Output = TA>,
    Tensor2<AB, SNB, SKB, DNB, DKB>: TensorIndex<Coord2, Output = TB>,
    Tensor2<AC, SMC, SNC, DMC, DNC>:
        TensorIndex<Coord2, Output = TC> + TensorSet<Coord2, Item = TC>,
    TA: Copy + Mul<TB>,
    TB: Copy,
    TC: Copy,
    <TA as Mul<TB>>::Output: Add<TC, Output = TC>,
{
    fn gemm(
        self,
        b: Tensor2<AB, SNB, SKB, DNB, DKB>,
        c: &mut Tensor2<AC, SMC, SNC, DMC, DNC>,
    ) {
        gemm_fma(&self, &b, c);
    }
}

/// Reference batched GEMM: `C[m, n, l] += A[m, k, l] * B[n, k, l]`.
///
/// Accepts either all rank-3 operands `(M, K, L)`, `(N, K, L)`, `(M, N, L)` or all
/// rank-2 unbatched operands `(M, K)`, `(N, K)`, `(M, N)`, delegating execution to
/// [`gemm_fma`]. Any mode may be a hierarchical multi-mode.
pub fn gemm<A, B, C>(a: A, b: B, c: &mut C)
where
    A: Gemm<B, C>,
{
    a.gemm(b, c);
}
