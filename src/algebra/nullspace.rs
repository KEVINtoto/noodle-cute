//! Layout nullspace algebra: coordinates mapped to zero by a [`Layout`].
//!
//! For a layout $L : \mathbb{Z}_{|L|} \to D$ with flattened shapes $(s_0, \dots, s_{k-1})$,
//! flattened strides $(d_0, \dots, d_{k-1})$, and exclusive shape prefix products
//! $(p_0, \dots, p_{k-1})$, [`nullspace`] collects every mode whose stride is statically
//! known to be zero (`Int<0>`, `BasisCombination<Nil>`, or `BasisTerm` with a statically-zero
//! coefficient) and returns the compact layout enumerating their 1D domain coordinates:
//!
//! $$\mathrm{nullspace}(L) = (s_i)_{i \in Z} : (p_i)_{i \in Z}, \qquad Z = \{i \mid d_i \equiv_{\text{static}} 0\}$$
//!
//! If $Z$ is empty (no statically-zero stride modes), the trivial singleton layout
//! `Layout(Int<1>, Int<0>)` is returned. Every output coordinate satisfies:
//!
//! $$L(\mathrm{nullspace}(L)(i)) = 0 \qquad \forall i \in [0, |\mathrm{nullspace}(L)|)$$
//!
//! # Main Interfaces
//!
//! - [`Nullspace`] trait (`nullspace`)
//! - [`Layout::nullspace`]
//! - [`nullspace`] free function (supporting [`Layout`] and [`Tiler`] inputs)
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::nullspace::nullspace;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::{e, int_tuple, tuple};
//!
//! // Injective coordinate layout -> trivial 1:0 nullspace
//! let inj = Layout::new(int_tuple![Int::<4>, Int::<5>], tuple![e![0], e![1]]);
//! assert_eq!(nullspace(inj), Layout::new(Int::<1>, Int::<0>));
//!
//! // Single broadcast mode at index 2 -> 6:8
//! let l = Layout::new(
//!     int_tuple![Int::<2>, Int::<4>, Int::<6>],
//!     int_tuple![Int::<1>, Int::<2>, Int::<0>],
//! );
//! assert_eq!(nullspace(l), Layout::new(Int::<6>, Int::<8>));
//! ```
//!
//! # Notes
//!
//! - Only statically-zero strides are collected; a dynamic stride equal to `0` at runtime
//!   is treated as non-zero statically and excluded.
//! - Modes occurring strictly after the last statically-zero stride mode are never multiplied
//!   into the prefix product, avoiding spurious overflow.
//! - Static prefix-product overflow before a zero-stride mode fails at compile time (`E0080`):
//!
//! ```compile_fail,E0080
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::nullspace::nullspace;
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let _ = nullspace(Layout::new(
//!     int_tuple![Int::<2_000_000_000>, Int::<2>, Int::<4>],
//!     int_tuple![Int::<1>, Int::<1>, Int::<0>],
//! ));
//! ```

use crate::basis::{BasisCoeff, BasisCombination, BasisPath, BasisTerm, BasisTuple};
use crate::integer::{ArithMul, Int, Integer, Product, int, mul};
use crate::layout::{Layout, Tiler, tiler_to_layout};
use crate::shape::{Shape, ShapeScalar};
use crate::stride::{Stride, StrideScalar};
use crate::tuple::{FlatTuple, Flatten, Item, Leaf, Nil, Til, Tuple};

/// Compile-time check for whether every term in a [`BasisTuple`] is statically zero.
#[doc(hidden)]
pub const trait NullspaceBasisTuple: BasisTuple {
    const ALL_STATIC_ZERO: bool;
}

const impl NullspaceBasisTuple for Nil {
    const ALL_STATIC_ZERO: bool = true;
}

const impl<V, P, T> NullspaceBasisTuple for Til<BasisTerm<V, P>, T>
where
    V: BasisCoeff,
    P: BasisPath,
    T: NullspaceBasisTuple,
    BasisTerm<V, P>: Leaf,
{
    const ALL_STATIC_ZERO: bool = V::IS_STATIC_ZERO && T::ALL_STATIC_ZERO;
}

/// Compile-time predicate classifying whether a stride leaf is statically known to be zero.
#[doc(hidden)]
pub const trait NullspaceStrideLeaf: Leaf {
    const IS_STATIC_ZERO: bool;
}

const impl<I: Integer> NullspaceStrideLeaf for I {
    const IS_STATIC_ZERO: bool = matches!(I::STATIC_VALUE, Some(0));
}

const impl<V: BasisCoeff, P: BasisPath> NullspaceStrideLeaf for BasisTerm<V, P> {
    const IS_STATIC_ZERO: bool = V::IS_STATIC_ZERO;
}

const impl<T: NullspaceBasisTuple> NullspaceStrideLeaf for BasisCombination<T> {
    const IS_STATIC_ZERO: bool = T::ALL_STATIC_ZERO;
}

/// Compile-time predicate testing whether a flat stride tuple contains any statically-zero mode.
#[doc(hidden)]
pub const trait HasZeroStride: FlatTuple {
    const HAS_ZERO: bool;
}

const impl HasZeroStride for Nil {
    const HAS_ZERO: bool = false;
}

const impl<H: NullspaceStrideLeaf, T: HasZeroStride> HasZeroStride for Til<H, T> {
    const HAS_ZERO: bool = H::IS_STATIC_ZERO || T::HAS_ZERO;
}

const HAS_ZERO_STRIDE<FD: HasZeroStride>: bool = FD::HAS_ZERO;
const STRIDE_IS_STATIC_ZERO<D: NullspaceStrideLeaf>: bool = D::IS_STATIC_ZERO;

/// Prepends one nullspace `(S, P)` mode onto a non-empty flat nullspace suffix layout.
#[doc(hidden)]
pub const trait PrependNullspaceMode<S: ShapeScalar, P: StrideScalar>: Copy {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn prepend_nullspace_mode(self, s: S, p: P) -> Layout<Self::Shape, Self::Stride>;
}

const impl<ST, PT, S, P> PrependNullspaceMode<S, P> for Layout<ST, PT>
where
    ST: ShapeScalar,
    PT: StrideScalar + Stride<ST>,
    S: ShapeScalar,
    P: StrideScalar + Stride<S>,
{
    type Shape = Til<S, Til<ST, Nil>>;
    type Stride = Til<P, Til<PT, Nil>>;

    fn prepend_nullspace_mode(self, s: S, p: P) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(Til(s, Til(self.shape, Nil)), Til(p, Til(self.stride, Nil)))
    }
}

const impl<HS, TS, HP, TP, S, P> PrependNullspaceMode<S, P> for Layout<Til<HS, TS>, Til<HP, TP>>
where
    HS: ShapeScalar,
    TS: Tuple + Shape,
    HP: StrideScalar + Stride<HS>,
    TP: Tuple + Stride<TS>,
    S: ShapeScalar,
    P: StrideScalar + Stride<S>,
{
    type Shape = Til<S, Til<HS, TS>>;
    type Stride = Til<P, Til<HP, TP>>;

    fn prepend_nullspace_mode(self, s: S, p: P) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(Til(s, self.shape), Til(p, self.stride))
    }
}

/// Walks flattened `(shape, stride)` pairs carrying exclusive prefix product `P`
/// and collects all statically-zero stride modes.
///
/// Only invoked when `FD` contains at least one statically-zero stride mode.
#[doc(hidden)]
pub const trait NullspaceWalk<FD: HasZeroStride, P: ShapeScalar>: FlatTuple {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn nullspace_walk(self, fd: FD, p: P) -> Layout<Self::Shape, Self::Stride>;
}

/// Dispatches one step of [`NullspaceWalk`] based on whether the head stride is statically
/// zero (`KEEP`) and whether any later stride in the tail is statically zero (`MORE`).
#[doc(hidden)]
pub const trait NullspaceWalkStep<
    FD: HasZeroStride,
    P: ShapeScalar,
    const KEEP: bool,
    const MORE: bool,
>: FlatTuple
{
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn nullspace_step(self, fd: FD, p: P) -> Layout<Self::Shape, Self::Stride>;
}

// Last statically-zero mode: emit unwrapped `(SH, P)` immediately without computing `p * sh`.
const impl<SH, ST, DH, DT, P> NullspaceWalkStep<Til<DH, DT>, P, true, false> for Til<SH, ST>
where
    SH: [const] ShapeScalar,
    ST: FlatTuple,
    DH: NullspaceStrideLeaf,
    DT: HasZeroStride,
    P: ShapeScalar + StrideScalar + Stride<SH>,
{
    type Shape = SH;
    type Stride = P;

    fn nullspace_step(self, _: Til<DH, DT>, p: P) -> Layout<SH, P> {
        let _ = self.0.size();
        Layout::new(self.0, p)
    }
}

// Non-zero head with more statically-zero modes in tail: advance `p * sh` and recurse.
const impl<SH, ST, DH, DT, P> NullspaceWalkStep<Til<DH, DT>, P, false, true> for Til<SH, ST>
where
    SH: [const] ShapeScalar,
    ST: FlatTuple + [const] NullspaceWalk<DT, Product<P, SH>>,
    DH: NullspaceStrideLeaf,
    DT: HasZeroStride,
    P: [const] ShapeScalar + [const] ArithMul<P, SH>,
    Product<P, SH>: [const] ShapeScalar,
{
    type Shape = <ST as NullspaceWalk<DT, Product<P, SH>>>::Shape;
    type Stride = <ST as NullspaceWalk<DT, Product<P, SH>>>::Stride;

    fn nullspace_step(self, fd: Til<DH, DT>, p: P) -> Layout<Self::Shape, Self::Stride> {
        let _ = self.0.size();
        let next_p = mul(p, self.0);
        self.1.nullspace_walk(fd.1, next_p)
    }
}

// Statically-zero head with more statically-zero modes in tail: advance, recurse, and prepend.
const impl<SH, ST, DH, DT, P> NullspaceWalkStep<Til<DH, DT>, P, true, true> for Til<SH, ST>
where
    SH: [const] ShapeScalar,
    ST: FlatTuple + [const] NullspaceWalk<DT, Product<P, SH>>,
    DH: NullspaceStrideLeaf,
    DT: HasZeroStride,
    P: [const] ShapeScalar + StrideScalar + Stride<SH> + [const] ArithMul<P, SH>,
    Product<P, SH>: [const] ShapeScalar,
    Layout<
        <ST as NullspaceWalk<DT, Product<P, SH>>>::Shape,
        <ST as NullspaceWalk<DT, Product<P, SH>>>::Stride,
    >: [const] PrependNullspaceMode<SH, P>,
{
    type Shape = <Layout<
        <ST as NullspaceWalk<DT, Product<P, SH>>>::Shape,
        <ST as NullspaceWalk<DT, Product<P, SH>>>::Stride,
    > as PrependNullspaceMode<SH, P>>::Shape;
    type Stride = <Layout<
        <ST as NullspaceWalk<DT, Product<P, SH>>>::Shape,
        <ST as NullspaceWalk<DT, Product<P, SH>>>::Stride,
    > as PrependNullspaceMode<SH, P>>::Stride;

    fn nullspace_step(self, fd: Til<DH, DT>, p: P) -> Layout<Self::Shape, Self::Stride> {
        let _ = self.0.size();
        let next_p = mul(p, self.0);
        let tail = self.1.nullspace_walk(fd.1, next_p);
        tail.prepend_nullspace_mode(self.0, p)
    }
}

const impl<SH, ST, DH, DT, P> NullspaceWalk<Til<DH, DT>, P> for Til<SH, ST>
where
    SH: ShapeScalar,
    ST: FlatTuple,
    DH: NullspaceStrideLeaf,
    DT: HasZeroStride,
    P: ShapeScalar,
    Self: [const] NullspaceWalkStep<
            Til<DH, DT>,
            P,
            { STRIDE_IS_STATIC_ZERO::<DH> },
            { HAS_ZERO_STRIDE::<DT> },
        >,
{
    type Shape = <Self as NullspaceWalkStep<
        Til<DH, DT>,
        P,
        { STRIDE_IS_STATIC_ZERO::<DH> },
        { HAS_ZERO_STRIDE::<DT> },
    >>::Shape;
    type Stride = <Self as NullspaceWalkStep<
        Til<DH, DT>,
        P,
        { STRIDE_IS_STATIC_ZERO::<DH> },
        { HAS_ZERO_STRIDE::<DT> },
    >>::Stride;

    fn nullspace_walk(self, fd: Til<DH, DT>, p: P) -> Layout<Self::Shape, Self::Stride> {
        self.nullspace_step(fd, p)
    }
}

/// Dispatches flattened `(shape, stride)` pairs to either the trivial `1:0` layout
/// (`HAS_ZERO == false`) or the prefix-product [`NullspaceWalk`] (`HAS_ZERO == true`).
#[doc(hidden)]
pub const trait NullspaceDispatch<FD: HasZeroStride, const HAS_ZERO: bool>:
    FlatTuple
{
    type Output: Copy;
    fn nullspace_dispatch(self, fd: FD) -> Self::Output;
}

const impl<FS: FlatTuple, FD: HasZeroStride> NullspaceDispatch<FD, false> for FS {
    type Output = Layout<Int<1>, Int<0>>;

    fn nullspace_dispatch(self, _: FD) -> Self::Output {
        Layout::new(Int::<1>, Int::<0>)
    }
}

const impl<FS, FD> NullspaceDispatch<FD, true> for FS
where
    FS: FlatTuple + [const] NullspaceWalk<FD, Int<1>>,
    FD: HasZeroStride,
{
    type Output =
        Layout<<FS as NullspaceWalk<FD, Int<1>>>::Shape, <FS as NullspaceWalk<FD, Int<1>>>::Stride>;

    fn nullspace_dispatch(self, fd: FD) -> Self::Output {
        self.nullspace_walk(fd, Int::<1>)
    }
}

/// Computes the nullspace layout enumerating coordinates that `self` maps to zero.
pub const trait Nullspace: Copy {
    /// Resulting nullspace [`Layout`].
    type Output: Copy;

    /// Computes the nullspace layout, panicking on negative dynamic extents or [`int`] overflow.
    fn nullspace(self) -> Self::Output;
}

const impl<S, D> Nullspace for Layout<S, D>
where
    S: Shape + [const] Flatten,
    D: Stride<S> + [const] Flatten,
    D::Output: HasZeroStride,
    S::Output: [const] NullspaceDispatch<D::Output, { HAS_ZERO_STRIDE::<D::Output> }>,
{
    type Output =
        <S::Output as NullspaceDispatch<D::Output, { HAS_ZERO_STRIDE::<D::Output> }>>::Output;

    fn nullspace(self) -> Self::Output {
        self.shape
            .flatten()
            .nullspace_dispatch(self.stride.flatten())
    }
}

macro_rules! impl_tiler_nullspace {
    ([$($g:tt)*] $T:ty) => {
        const impl<$($g)*> Nullspace for $T
        where
            Self: [const] Tiler,
            <Self as Tiler>::OutputLayout: [const] Nullspace,
        {
            type Output = <<Self as Tiler>::OutputLayout as Nullspace>::Output;

            fn nullspace(self) -> Self::Output {
                tiler_to_layout(self).nullspace()
            }
        }
    };
}

impl_tiler_nullspace!([] Nil);
impl_tiler_nullspace!([] int);
impl_tiler_nullspace!([const N: int,] Int<N>);
impl_tiler_nullspace!([H: Item, T: Tuple,] Til<H, T>);

impl<S, D> Layout<S, D> {
    /// Computes the nullspace layout of coordinates that `self` maps to zero.
    pub const fn nullspace(self) -> <Self as Nullspace>::Output
    where
        Self: [const] Nullspace,
    {
        Nullspace::nullspace(self)
    }
}

/// Computes the nullspace layout of a [`Layout`] or [`Tiler`].
pub const fn nullspace<A: [const] Nullspace>(a: A) -> A::Output {
    a.nullspace()
}
