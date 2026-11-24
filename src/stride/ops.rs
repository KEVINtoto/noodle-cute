//! Leaf stride multiplication, accumulation, and stride scaling.
//!
//! # Main Interfaces
//!
//! - [`StrideMul`]: Multiplies a stride leaf (integer, [`ScaledBasis`], or
//!   [`BasisCombination`]) by a scalar coordinate or extent.
//! - [`Accumulate`]: Combines intermediate products during [`inner_product`](super::inner_product),
//!   treating [`Nil`] and static scalar zeros as additive identities.
//! - [`ScaleBasisFunctor`] / [`ScaleStride`]: Scales all leaves in a stride by an integer
//!   or basis factor via [`TransformLeaf`].
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::basis::{Mode, ScaledBasis};
//! use noodle_cute::integer::Int;
//! use noodle_cute::stride::{Accumulate, ScaleStride, StrideMul};
//! use noodle_cute::tuple::{Nil, TupleContainer};
//!
//! assert_eq!(Int::<3>.stride_mul(Int::<4>), Int::<12>);
//! let b0 = ScaledBasis::<Int<1>, Mode<0>>::new(Int::<1>);
//! assert_eq!(Int::<0>.accumulate(b0), b0);
//! assert_eq!(
//!     TupleContainer::new(Int::<2>, TupleContainer::new(Int::<3>, Nil)).scale_stride(b0),
//!     TupleContainer::new(
//!         ScaledBasis::<Int<2>, Mode<0>>::new(Int::<2>),
//!         TupleContainer::new(ScaledBasis::<Int<3>, Mode<0>>::new(Int::<3>), Nil),
//!     )
//! );
//! ```
//!
//! # Notes
//!
//! - [`Nil`] and static scalar `Int<0>` act as additive identities under [`Accumulate`];
//!   accumulating a dynamic scalar with a basis term or prefix-conflicting basis paths
//!   fails at compile time.
//! - Multiplying a basis stride by static `Int<0>` collapses to [`BasisCombination<Nil>`].

use core::ops::Mul;

use crate::basis::ops::BasisAdd;
use crate::basis::{
    BasisCoeff, BasisCombination, BasisPath, BasisTuple, COEFF_IS_STATIC_ZERO, ScaledBasis,
};
use crate::integer::{ArithAdd, ArithMul, Int, Integer, Product, Sum, add, mul};
use crate::tuple::{Item, Leaf, Nil, Til, TransformLeaf, TransformLeafFunctor};

/// Integer multiplication between a stride leaf and a scalar coordinate or extent.
///
/// - `Integer × Integer -> Integer` ($d \cdot s$).
/// - `ScaledBasis<V, P> × Integer -> ScaledBasis<V · S, P>` (or `BasisCombination<Nil>` for static zero `S`).
/// - `BasisCombination<T> × Integer -> BasisCombination<...>` (collapsing static zero `S` to `BasisCombination<Nil>`).
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::basis::{Mode, ScaledBasis};
/// use noodle_cute::integer::Int;
/// use noodle_cute::stride::StrideMul;
///
/// // Integer multiplication: 3 * 4 = 12
/// assert_eq!(Int::<3>.stride_mul(Int::<4>), Int::<12>);
///
/// // Unit basis scaled by extent 4 -> 4 * E(0)
/// let unit0 = ScaledBasis::<Int<1>, Mode<0>>::new(Int::<1>);
/// let sb = unit0.stride_mul(Int::<4>);
/// assert_eq!(sb, ScaledBasis::<Int<4>, Mode<0>>::new(Int::<4>));
///
/// // ScaledBasis scaled by extent 2 -> (4 * 2) * E(0) = 8 * E(0)
/// let sb2 = sb.stride_mul(Int::<2>);
/// assert_eq!(sb2, ScaledBasis::<Int<8>, Mode<0>>::new(Int::<8>));
/// ```
pub const trait StrideMul<S>: Copy {
    /// The resulting stride type after multiplication.
    type Output: Copy;

    /// Multiplies this stride leaf by the scalar coordinate or extent.
    fn stride_mul(self, shape: S) -> Self::Output;
}

const impl<A, S> StrideMul<S> for A
where
    A: [const] Integer + [const] ArithMul<A, S>,
    S: [const] Integer,
{
    type Output = Product<A, S>;
    fn stride_mul(self, shape: S) -> Self::Output {
        mul(self, shape)
    }
}

const impl<V, P, S> StrideMul<S> for ScaledBasis<V, P>
where
    V: BasisCoeff,
    P: BasisPath,
    S: [const] BasisCoeff,
    Self: [const] Mul<S>,
    <Self as Mul<S>>::Output: Copy,
{
    type Output = <Self as Mul<S>>::Output;
    fn stride_mul(self, shape: S) -> Self::Output {
        self * shape
    }
}

const impl<T: BasisTuple, S: [const] BasisCoeff> StrideMul<S> for BasisCombination<T>
where
    Self: [const] Mul<S>,
    <Self as Mul<S>>::Output: Copy,
{
    type Output = <Self as Mul<S>>::Output;
    fn stride_mul(self, shape: S) -> Self::Output {
        self * shape
    }
}

/// Accumulation operation in the stride integer module: `self.accumulate(rhs)`.
///
/// Combines intermediate products during an [`inner_product`](super::inner_product). For ordinary
/// integers, this is standard addition. For basis elements, [`Nil`] and static scalar zeros act as
/// additive identities, while non-zero basis terms combine via [`BasisAdd`].
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::basis::{Mode, ScaledBasis};
/// use noodle_cute::integer::Int;
/// use noodle_cute::stride::Accumulate;
/// use noodle_cute::tuple::Nil;
///
/// let b0 = ScaledBasis::<Int<1>, Mode<0>>::new(Int::<1>);
///
/// // Identity with static zero and Nil
/// assert_eq!(Int::<0>.accumulate(b0), b0);
/// assert_eq!(b0.accumulate(Nil), b0);
///
/// // Integer addition
/// assert_eq!(Int::<5>.accumulate(Int::<7>), Int::<12>);
/// ```
pub const trait Accumulate<R>: Copy {
    /// Accumulated output type.
    type Output: Copy;
    /// Combines `self` with `rhs`.
    fn accumulate(self, rhs: R) -> Self::Output;
}

const impl Accumulate<Nil> for Nil {
    type Output = Nil;
    fn accumulate(self, _: Nil) -> Self::Output {
        Nil
    }
}

const impl<B: Integer> Accumulate<B> for Nil {
    type Output = B;
    fn accumulate(self, rhs: B) -> Self::Output {
        rhs
    }
}

const impl<V: BasisCoeff, P: Copy> Accumulate<ScaledBasis<V, P>> for Nil {
    type Output = ScaledBasis<V, P>;
    fn accumulate(self, rhs: ScaledBasis<V, P>) -> Self::Output {
        rhs
    }
}

const impl<T: BasisTuple> Accumulate<BasisCombination<T>> for Nil {
    type Output = BasisCombination<T>;
    fn accumulate(self, rhs: BasisCombination<T>) -> Self::Output {
        rhs
    }
}

const impl<A: Integer> Accumulate<Nil> for A {
    type Output = A;
    fn accumulate(self, _: Nil) -> Self::Output {
        self
    }
}

const impl<A, B> Accumulate<B> for A
where
    A: [const] Integer + [const] ArithAdd<A, B>,
    B: [const] Integer,
{
    type Output = Sum<A, B>;
    fn accumulate(self, rhs: B) -> Self::Output {
        add(self, rhs)
    }
}

const impl<V: BasisCoeff, P: Copy> Accumulate<ScaledBasis<V, P>> for Int<0> {
    type Output = ScaledBasis<V, P>;
    fn accumulate(self, rhs: ScaledBasis<V, P>) -> Self::Output {
        rhs
    }
}

/// Dispatches `Integer` accumulated with `BasisCombination<Nil>`:
/// `IS_ZERO = true` yields `BasisCombination<Nil>`, `false` keeps `Self`.
#[doc(hidden)]
pub const trait ScalarAccumulateEmptyComb<const IS_ZERO: bool>: Integer {
    type Output: Copy;
    fn accumulate_empty_comb(self, rhs: BasisCombination<Nil>) -> Self::Output;
}

const impl<A: Integer> ScalarAccumulateEmptyComb<true> for A {
    type Output = BasisCombination<Nil>;
    fn accumulate_empty_comb(self, rhs: BasisCombination<Nil>) -> Self::Output {
        rhs
    }
}

const impl<A: Integer> ScalarAccumulateEmptyComb<false> for A {
    type Output = A;
    fn accumulate_empty_comb(self, _: BasisCombination<Nil>) -> Self::Output {
        self
    }
}

/// Dispatches `BasisCombination<Nil>` accumulated with scalar `B`:
/// `IS_ZERO = true` keeps `BasisCombination<Nil>`, `false` yields `B`.
#[doc(hidden)]
pub const trait EmptyCombAccumulateScalar<B: Integer, const IS_ZERO: bool>: Copy {
    type Output: Copy;
    fn empty_comb_accumulate(self, rhs: B) -> Self::Output;
}

const impl<B: Integer> EmptyCombAccumulateScalar<B, true> for BasisCombination<Nil> {
    type Output = BasisCombination<Nil>;
    fn empty_comb_accumulate(self, _: B) -> Self::Output {
        self
    }
}

const impl<B: Integer> EmptyCombAccumulateScalar<B, false> for BasisCombination<Nil> {
    type Output = B;
    fn empty_comb_accumulate(self, rhs: B) -> Self::Output {
        rhs
    }
}

const impl<A> Accumulate<BasisCombination<Nil>> for A
where
    A: [const] Integer
        + [const] BasisCoeff
        + [const] ScalarAccumulateEmptyComb<{ COEFF_IS_STATIC_ZERO::<A> }>,
{
    type Output = <A as ScalarAccumulateEmptyComb<{ COEFF_IS_STATIC_ZERO::<A> }>>::Output;
    fn accumulate(self, rhs: BasisCombination<Nil>) -> Self::Output {
        self.accumulate_empty_comb(rhs)
    }
}

const impl<H, T> Accumulate<BasisCombination<Til<H, T>>> for Int<0>
where
    H: Copy,
    T: BasisTuple,
    Til<H, T>: BasisTuple,
{
    type Output = BasisCombination<Til<H, T>>;
    fn accumulate(self, rhs: BasisCombination<Til<H, T>>) -> Self::Output {
        rhs
    }
}

const impl<V: BasisCoeff, P: Copy> Accumulate<Nil> for ScaledBasis<V, P> {
    type Output = Self;
    fn accumulate(self, _: Nil) -> Self::Output {
        self
    }
}

const impl<V: BasisCoeff, P: Copy> Accumulate<Int<0>> for ScaledBasis<V, P> {
    type Output = Self;
    fn accumulate(self, _: Int<0>) -> Self::Output {
        self
    }
}

const impl<VL, PL, VR, PR> Accumulate<ScaledBasis<VR, PR>> for ScaledBasis<VL, PL>
where
    VL: BasisCoeff,
    PL: Copy,
    Self: [const] BasisAdd<ScaledBasis<VR, PR>>,
{
    type Output = <Self as BasisAdd<ScaledBasis<VR, PR>>>::Output;
    fn accumulate(self, rhs: ScaledBasis<VR, PR>) -> Self::Output {
        self.basis_add(rhs)
    }
}

const impl<VL, PL, T> Accumulate<BasisCombination<T>> for ScaledBasis<VL, PL>
where
    VL: BasisCoeff,
    PL: Copy,
    T: BasisTuple,
    Self: [const] BasisAdd<BasisCombination<T>>,
{
    type Output = <Self as BasisAdd<BasisCombination<T>>>::Output;
    fn accumulate(self, rhs: BasisCombination<T>) -> Self::Output {
        self.basis_add(rhs)
    }
}

const impl Accumulate<Nil> for BasisCombination<Nil> {
    type Output = Self;
    fn accumulate(self, _: Nil) -> Self::Output {
        self
    }
}

const impl<B> Accumulate<B> for BasisCombination<Nil>
where
    B: [const] Integer + [const] BasisCoeff,
    Self: [const] EmptyCombAccumulateScalar<B, { COEFF_IS_STATIC_ZERO::<B> }>,
{
    type Output = <Self as EmptyCombAccumulateScalar<B, { COEFF_IS_STATIC_ZERO::<B> }>>::Output;
    fn accumulate(self, rhs: B) -> Self::Output {
        self.empty_comb_accumulate(rhs)
    }
}

const impl<V: BasisCoeff, P: Copy> Accumulate<ScaledBasis<V, P>> for BasisCombination<Nil> {
    type Output = ScaledBasis<V, P>;
    fn accumulate(self, rhs: ScaledBasis<V, P>) -> Self::Output {
        rhs
    }
}

const impl<TR: BasisTuple> Accumulate<BasisCombination<TR>> for BasisCombination<Nil> {
    type Output = BasisCombination<TR>;
    fn accumulate(self, rhs: BasisCombination<TR>) -> Self::Output {
        rhs
    }
}

const impl<H, TL> Accumulate<Nil> for BasisCombination<Til<H, TL>>
where
    H: Copy,
    TL: BasisTuple,
    Til<H, TL>: BasisTuple,
{
    type Output = Self;
    fn accumulate(self, _: Nil) -> Self::Output {
        self
    }
}

const impl<H, TL> Accumulate<Int<0>> for BasisCombination<Til<H, TL>>
where
    H: Copy,
    TL: BasisTuple,
    Til<H, TL>: BasisTuple,
{
    type Output = Self;
    fn accumulate(self, _: Int<0>) -> Self::Output {
        self
    }
}

const impl<H, TL, VR, PR> Accumulate<ScaledBasis<VR, PR>> for BasisCombination<Til<H, TL>>
where
    H: Copy,
    TL: BasisTuple,
    Til<H, TL>: BasisTuple,
    Self: [const] BasisAdd<ScaledBasis<VR, PR>>,
{
    type Output = <Self as BasisAdd<ScaledBasis<VR, PR>>>::Output;
    fn accumulate(self, rhs: ScaledBasis<VR, PR>) -> Self::Output {
        self.basis_add(rhs)
    }
}

const impl<H, TL, TR> Accumulate<BasisCombination<TR>> for BasisCombination<Til<H, TL>>
where
    H: Copy,
    TL: BasisTuple,
    Til<H, TL>: BasisTuple,
    TR: BasisTuple,
    Self: [const] BasisAdd<BasisCombination<TR>>,
{
    type Output = <Self as BasisAdd<BasisCombination<TR>>>::Output;
    fn accumulate(self, rhs: BasisCombination<TR>) -> Self::Output {
        self.basis_add(rhs)
    }
}

/// Functor used by [`ScaleStride`] to scale scalar or basis leaves within hierarchical tuples.
///
/// - When applied to an [`Integer`] leaf with `Factor: Integer`, multiplies the scalars.
/// - When applied to an [`Integer`] leaf with `Factor: ScaledBasis` or `BasisCombination`, scales the basis by the scalar leaf.
/// - When applied to a [`ScaledBasis`] or [`BasisCombination`] leaf with `Factor: Integer`, scales the basis by `Factor`.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::basis::{Mode, ScaledBasis};
/// use noodle_cute::integer::Int;
/// use noodle_cute::stride::ScaleBasisFunctor;
/// use noodle_cute::tuple::TransformLeafFunctor;
///
/// // ScaleBasisFunctor with unit basis: turns integer into ScaledBasis
/// let f_basis = ScaleBasisFunctor(ScaledBasis::<Int<1>, Mode<0>>::new(Int::<1>));
/// assert_eq!(f_basis.apply(Int::<3>), ScaledBasis::<Int<3>, Mode<0>>::new(Int::<3>));
///
/// // ScaleBasisFunctor with Integer: scales existing ScaledBasis
/// let f_factor = ScaleBasisFunctor(Int::<4>);
/// let sb = ScaledBasis::<Int<3>, Mode<0>>::new(Int::<3>);
/// assert_eq!(f_factor.apply(sb), ScaledBasis::<Int<12>, Mode<0>>::new(Int::<12>));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScaleBasisFunctor<Factor>(pub Factor);

const impl<V, Factor> TransformLeafFunctor<V> for ScaleBasisFunctor<Factor>
where
    V: [const] Integer + [const] Leaf + [const] ArithMul<V, Factor>,
    Factor: [const] Integer,
    Product<V, Factor>: [const] Item,
{
    type Output = Product<V, Factor>;
    fn apply(&self, leaf: V) -> Self::Output {
        mul(leaf, self.0)
    }
}

const impl<V, VF, PF> TransformLeafFunctor<V> for ScaleBasisFunctor<ScaledBasis<VF, PF>>
where
    V: [const] Integer + [const] Leaf,
    VF: BasisCoeff,
    PF: BasisPath,
    ScaledBasis<VF, PF>: [const] StrideMul<V>,
    <ScaledBasis<VF, PF> as StrideMul<V>>::Output: [const] Item,
{
    type Output = <ScaledBasis<VF, PF> as StrideMul<V>>::Output;
    fn apply(&self, leaf: V) -> Self::Output {
        self.0.stride_mul(leaf)
    }
}

const impl<V, T> TransformLeafFunctor<V> for ScaleBasisFunctor<BasisCombination<T>>
where
    V: [const] Integer + [const] Leaf,
    T: BasisTuple,
    BasisCombination<T>: [const] StrideMul<V>,
    <BasisCombination<T> as StrideMul<V>>::Output: [const] Item,
{
    type Output = <BasisCombination<T> as StrideMul<V>>::Output;
    fn apply(&self, leaf: V) -> Self::Output {
        self.0.stride_mul(leaf)
    }
}

const impl<V, P, Factor> TransformLeafFunctor<ScaledBasis<V, P>> for ScaleBasisFunctor<Factor>
where
    V: BasisCoeff,
    P: BasisPath,
    ScaledBasis<V, P>: [const] Leaf,
    Factor: [const] BasisCoeff,
    ScaledBasis<V, P>: [const] StrideMul<Factor>,
    <ScaledBasis<V, P> as StrideMul<Factor>>::Output: [const] Item,
{
    type Output = <ScaledBasis<V, P> as StrideMul<Factor>>::Output;
    fn apply(&self, leaf: ScaledBasis<V, P>) -> Self::Output {
        leaf.stride_mul(self.0)
    }
}

const impl<T, Factor> TransformLeafFunctor<BasisCombination<T>> for ScaleBasisFunctor<Factor>
where
    T: BasisTuple,
    BasisCombination<T>: [const] Leaf,
    Factor: [const] BasisCoeff,
    BasisCombination<T>: [const] StrideMul<Factor>,
    <BasisCombination<T> as StrideMul<Factor>>::Output: [const] Item,
{
    type Output = <BasisCombination<T> as StrideMul<Factor>>::Output;
    fn apply(&self, leaf: BasisCombination<T>) -> Self::Output {
        leaf.stride_mul(self.0)
    }
}

/// Scales the stride leaves within a hierarchical tuple or scalar stride.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::basis::{Mode, ScaledBasis};
/// use noodle_cute::integer::Int;
/// use noodle_cute::stride::ScaleStride;
/// use noodle_cute::tuple::{Nil, TupleContainer};
///
/// let s = TupleContainer::new(
///     ScaledBasis::<Int<2>, Mode<0>>::new(Int::<2>),
///     TupleContainer::new(ScaledBasis::<Int<3>, Mode<1>>::new(Int::<3>), Nil),
/// );
/// let scaled = s.scale_stride(Int::<4>);
/// assert_eq!(
///     scaled,
///     TupleContainer::new(
///         ScaledBasis::<Int<8>, Mode<0>>::new(Int::<8>),
///         TupleContainer::new(ScaledBasis::<Int<12>, Mode<1>>::new(Int::<12>), Nil),
///     )
/// );
/// ```
pub const trait ScaleStride<Factor>: Copy {
    /// Scaled stride output type.
    type Output;
    /// Scales every stride leaf in `self` by `factor`.
    fn scale_stride(self, factor: Factor) -> Self::Output;
}

const impl<T, Factor> ScaleStride<Factor> for T
where
    Factor: Copy,
    T: [const] TransformLeaf<ScaleBasisFunctor<Factor>>,
{
    type Output = <T as TransformLeaf<ScaleBasisFunctor<Factor>>>::Output;

    fn scale_stride(self, factor: Factor) -> Self::Output {
        self.transform(&ScaleBasisFunctor(factor))
    }
}
