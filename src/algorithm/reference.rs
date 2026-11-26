//! Reference (naive) element-wise algorithms over CuTe [`Tensor`]s (Whitepaper §2.6).
//!
//! These straightforward loops define the exact mathematical semantics of COPY and
//! batched GEMM, serving both as executable implementations and as the oracle for
//! layout-analyzed optimizations such as [`crate::algorithm::copy()`].
//!
//! # Main Interfaces
//!
//! - [`copy`]: Rank-1 element-wise transfer `dst[i] = src[i]` over flat coordinates
//!   `0..size(dst)`. Realizes memcpy, gather, scatter, broadcast, constant fill, and
//!   transpose through layout choice alone.
//! - [`gemm`] / [`Gemm`]: Rank-3 batched matrix multiplication
//!   `C[m, n, l] += A[m, k, l] * B[n, k, l]`, with rank-2 `(M, K) x (N, K) -> (M, N)`
//!   operands automatically promoted with a zero-copy size-1 batch mode `1:0`.
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

use core::ops::{Add, Mul};

use crate::integer::{Int, add, int};
use crate::layout::Layout;
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tensor::{AccessorBase, Tensor, TensorIndex, TensorSet};
use crate::tuple::{Nil, Til};

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
/// Statically mismatched total sizes are rejected at compile time; dynamic mismatches
/// panic at runtime.
pub const fn copy<AS, SS, DS, AD, SD, DD, T>(
    src: Tensor<AS, Layout<SS, DS>>,
    dst: &mut Tensor<AD, Layout<SD, DD>>,
) where
    AS: Copy,
    SS: [const] Shape,
    DS: [const] Stride<SS>,
    SD: [const] Shape,
    DD: [const] Stride<SD>,
    (): ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SS, SD> }>,
    Tensor<AS, Layout<SS, DS>>: [const] TensorIndex<int, Output = T>,
    Tensor<AD, Layout<SD, DD>>: [const] TensorSet<int, Item = T>,
{
    let n = dst.size();
    assert!(src.size() == n, "copy: size mismatch");
    let mut i: int = 0;
    while i < n {
        dst.set(i, src.at(i));
        i = add(i, 1);
    }
}

/// 3-D flat coordinate `(m, n, l)` used by [`Gemm`].
type Coord3 = Til<int, Til<int, Til<int, Nil>>>;

/// Reference batched GEMM trait backing [`gemm`].
///
/// Implemented for:
/// - Rank-3 operands `A:(M, K, L)`, `B:(N, K, L)`, `C:(M, N, L)`.
/// - Rank-2 operands `A:(M, K)`, `B:(N, K)`, `C:(M, N)`, promoted with a zero-copy
///   size-1 batch mode `Layout(Int<1>, Int<0>)`.
///
/// Each mode `M`, `N`, `K`, `L` may itself be a hierarchical multi-mode.
pub const trait Gemm<B, C> {
    /// Accumulates `C[m, n, l] += A[m, k, l] * B[n, k, l]` in place.
    fn gemm(self, b: B, c: &mut C);
}

const impl<
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
>
    Gemm<
        Tensor<AB, Layout<Til<SNB, Til<SKB, Til<SLB, Nil>>>, Til<DNB, Til<DKB, Til<DLB, Nil>>>>>,
        Tensor<AC, Layout<Til<SMC, Til<SNC, Til<SLC, Nil>>>, Til<DMC, Til<DNC, Til<DLC, Nil>>>>>,
    > for Tensor<AA, Layout<Til<SMA, Til<SKA, Til<SLA, Nil>>>, Til<DMA, Til<DKA, Til<DLA, Nil>>>>>
where
    AA: Copy,
    AB: Copy,
    SMA: [const] Shape,
    SKA: [const] Shape,
    SLA: [const] Shape,
    DMA: [const] Stride<SMA>,
    DKA: [const] Stride<SKA>,
    DLA: [const] Stride<SLA>,
    SNB: [const] Shape,
    SKB: [const] Shape,
    SLB: [const] Shape,
    DNB: [const] Stride<SNB>,
    DKB: [const] Stride<SKB>,
    DLB: [const] Stride<SLB>,
    SMC: [const] Shape,
    SNC: [const] Shape,
    SLC: [const] Shape,
    DMC: [const] Stride<SMC>,
    DNC: [const] Stride<SNC>,
    DLC: [const] Stride<SLC>,
    (): ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SMA, SMC> }>
        + ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SNB, SNC> }>
        + ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SKA, SKB> }>
        + ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SLA, SLC> }>
        + ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SLB, SLC> }>
        + ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SLA, SLB> }>,
    Self: [const] TensorIndex<Coord3, Output = TA>,
    Tensor<AB, Layout<Til<SNB, Til<SKB, Til<SLB, Nil>>>, Til<DNB, Til<DKB, Til<DLB, Nil>>>>>:
        [const] TensorIndex<Coord3, Output = TB>,
    Tensor<AC, Layout<Til<SMC, Til<SNC, Til<SLC, Nil>>>, Til<DMC, Til<DNC, Til<DLC, Nil>>>>>:
        [const] TensorIndex<Coord3, Output = TC> + [const] TensorSet<Coord3, Item = TC>,
    TA: [const] Mul<TB>,
    TC: [const] Add<<TA as Mul<TB>>::Output, Output = TC>,
{
    fn gemm(
        self,
        b: Tensor<AB, Layout<Til<SNB, Til<SKB, Til<SLB, Nil>>>, Til<DNB, Til<DKB, Til<DLB, Nil>>>>>,
        c: &mut Tensor<
            AC,
            Layout<Til<SMC, Til<SNC, Til<SLC, Nil>>>, Til<DMC, Til<DNC, Til<DLC, Nil>>>>,
        >,
    ) {
        let size_m = c.layout.shape.0.size();
        let size_n = c.layout.shape.1.0.size();
        let size_k = self.layout.shape.1.0.size();
        let size_l = c.layout.shape.1.1.0.size();

        assert!(
            self.layout.shape.0.size() == size_m,
            "gemm: row size mismatch"
        );
        assert!(
            b.layout.shape.0.size() == size_n,
            "gemm: column size mismatch"
        );
        assert!(
            b.layout.shape.1.0.size() == size_k,
            "gemm: reduction size mismatch"
        );
        assert!(
            self.layout.shape.1.1.0.size() == size_l && b.layout.shape.1.1.0.size() == size_l,
            "gemm: batch size mismatch"
        );

        let mut l: int = 0;
        while l < size_l {
            let mut k: int = 0;
            while k < size_k {
                let mut n: int = 0;
                while n < size_n {
                    let mut m: int = 0;
                    while m < size_m {
                        let a_val = self.at(Til(m, Til(k, Til(l, Nil))));
                        let b_val = b.at(Til(n, Til(k, Til(l, Nil))));
                        let c_coord = Til(m, Til(n, Til(l, Nil)));
                        let acc = c.at(c_coord) + a_val * b_val;
                        c.set(c_coord, acc);
                        m = add(m, 1);
                    }
                    n = add(n, 1);
                }
                k = add(k, 1);
            }
            l = add(l, 1);
        }
    }
}

const impl<AA, SMA, SKA, DMA, DKA, AB, SNB, SKB, DNB, DKB, AC, SMC, SNC, DMC, DNC>
    Gemm<
        Tensor<AB, Layout<Til<SNB, Til<SKB, Nil>>, Til<DNB, Til<DKB, Nil>>>>,
        Tensor<AC, Layout<Til<SMC, Til<SNC, Nil>>, Til<DMC, Til<DNC, Nil>>>>,
    > for Tensor<AA, Layout<Til<SMA, Til<SKA, Nil>>, Til<DMA, Til<DKA, Nil>>>>
where
    AA: AccessorBase,
    AB: AccessorBase,
    AC: AccessorBase,
    SMA: [const] Shape,
    SKA: [const] Shape,
    DMA: [const] Stride<SMA>,
    DKA: [const] Stride<SKA>,
    SNB: [const] Shape,
    SKB: [const] Shape,
    DNB: [const] Stride<SNB>,
    DKB: [const] Stride<SKB>,
    SMC: [const] Shape,
    SNC: [const] Shape,
    DMC: [const] Stride<SMC>,
    DNC: [const] Stride<SNC>,
    Tensor<AA, Layout<Til<SMA, Til<SKA, Til<Int<1>, Nil>>>, Til<DMA, Til<DKA, Til<Int<0>, Nil>>>>>:
        [const] Gemm<
                Tensor<
                    AB,
                    Layout<
                        Til<SNB, Til<SKB, Til<Int<1>, Nil>>>,
                        Til<DNB, Til<DKB, Til<Int<0>, Nil>>>,
                    >,
                >,
                Tensor<
                    AC,
                    Layout<
                        Til<SMC, Til<SNC, Til<Int<1>, Nil>>>,
                        Til<DMC, Til<DNC, Til<Int<0>, Nil>>>,
                    >,
                >,
            >,
{
    fn gemm(
        self,
        b: Tensor<AB, Layout<Til<SNB, Til<SKB, Nil>>, Til<DNB, Til<DKB, Nil>>>>,
        c: &mut Tensor<AC, Layout<Til<SMC, Til<SNC, Nil>>, Til<DMC, Til<DNC, Nil>>>>,
    ) {
        let a3 = Tensor::new(
            self.accessor,
            Layout::new(
                Til(
                    self.layout.shape.0,
                    Til(self.layout.shape.1.0, Til(Int::<1>, Nil)),
                ),
                Til(
                    self.layout.stride.0,
                    Til(self.layout.stride.1.0, Til(Int::<0>, Nil)),
                ),
            ),
        );
        let b3 = Tensor::new(
            b.accessor,
            Layout::new(
                Til(
                    b.layout.shape.0,
                    Til(b.layout.shape.1.0, Til(Int::<1>, Nil)),
                ),
                Til(
                    b.layout.stride.0,
                    Til(b.layout.stride.1.0, Til(Int::<0>, Nil)),
                ),
            ),
        );
        let mut c3 = Tensor::new(
            c.accessor,
            Layout::new(
                Til(
                    c.layout.shape.0,
                    Til(c.layout.shape.1.0, Til(Int::<1>, Nil)),
                ),
                Til(
                    c.layout.stride.0,
                    Til(c.layout.stride.1.0, Til(Int::<0>, Nil)),
                ),
            ),
        );
        a3.gemm(b3, &mut c3);
    }
}

/// Reference batched GEMM: `C[m, n, l] += A[m, k, l] * B[n, k, l]`.
///
/// Accepts either all rank-3 operands `(M, K, L)`, `(N, K, L)`, `(M, N, L)` or all
/// rank-2 unbatched operands `(M, K)`, `(N, K)`, `(M, N)` (which append a size-1 batch
/// mode `1:0` without copying). Any mode may be a hierarchical multi-mode.
pub const fn gemm<A, B, C>(a: A, b: B, c: &mut C)
where
    A: [const] Gemm<B, C>,
{
    a.gemm(b, c);
}
