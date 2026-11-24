//! Linear arithmetic (`+`, `-`, unary `-`, scalar `*`) for basis terms and combinations.
//!
//! Merges path-sorted sparse basis chains at compile time or runtime, automatically
//! pruning static zeros and normalizing outputs via [`CanonicalBasis`].
//!
//! # Main Interfaces
//!
//! - **Coefficient algebra traits**: [`BasisCoeffAdd`], [`BasisCoeffSub`],
//!   [`BasisCoeffNeg`], [`BasisCoeffMul`] (implemented for all [`Integer`] types and
//!   extensible to custom rings).
//! - **Basis operation traits**: [`BasisAdd`], [`BasisSub`], [`BasisNeg`], [`BasisMul`].
//! - **Operators**: `const` `core::ops::{Add, Sub, Neg, Mul}` overloads on [`BasisTerm`],
//!   [`BasisCombination`], [`Int<N>`], and [`int`].
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{basis::BasisCombination, e, integer::Int};
//!
//! assert_eq!(Int::<2> * e![0] + Int::<3> * e![1] - Int::<2> * e![0], Int::<3> * e![1]);
//! assert_eq!(Int::<0> * (5i32 * e![1]), BasisCombination::zero());
//! assert_eq!(-(Int::<4> * e![0]), Int::<-4> * e![0]);
//! ```
//!
//! # Notes
//!
//! - Static `Int<0>` acts as an additive identity and an absorbing multiplicative zero;
//!   adding a non-static-zero scalar or multiplying two basis elements fails at compile time.
//! - Prefix-conflicting paths (e.g. `e![0] + e![0, 1]`) are rejected at compile time.
//! - Left and right scalar multiplication preserve operand order so non-commutative
//!   coefficient algebras behave accurately.
//! - Integer overflow follows [`crate::integer`] rules (compile-time error for static
//!   operands, runtime panic for dynamic operands).

use core::ops::{Add, Mul, Neg, Sub};

use crate::integer::{
    ArithAdd, ArithMul, ArithNeg, ArithSub, Difference, Int, Integer, Product, Sum, add, int, mul,
    neg, sub,
};
use crate::tuple::{Nil, Til};

use super::{
    ANY_STATIC_ZERO, BasisCoeff, BasisCombination, BasisPath, BasisTerm, BasisTuple,
    COEFF_IS_STATIC_ZERO, CanonicalBasis, ComparePath, EmitBasisTerm, IntoBasisCombination,
    PATH_ORDER, PathOrdering,
};

/// Coefficient addition for [`BasisCoeff`].
pub const trait BasisCoeffAdd<R>: BasisCoeff {
    /// Resulting coefficient type.
    type Output: [const] BasisCoeff;
    /// Evaluates `self + rhs`.
    fn coeff_add(self, rhs: R) -> Self::Output;
}

/// Coefficient negation for [`BasisCoeff`].
pub const trait BasisCoeffNeg: BasisCoeff {
    /// Resulting negated coefficient type.
    type Output: [const] BasisCoeff;
    /// Evaluates `-self`.
    fn coeff_neg(self) -> Self::Output;
}

/// Coefficient subtraction for [`BasisCoeff`].
pub const trait BasisCoeffSub<R>: BasisCoeff {
    /// Resulting coefficient type.
    type Output: [const] BasisCoeff;
    /// Evaluates `self - rhs`.
    fn coeff_sub(self, rhs: R) -> Self::Output;
}

/// Coefficient multiplication for [`BasisCoeff`].
pub const trait BasisCoeffMul<R>: BasisCoeff {
    /// Resulting coefficient type.
    type Output: [const] BasisCoeff;
    /// Evaluates `self * rhs`.
    fn coeff_mul(self, rhs: R) -> Self::Output;
}

const impl<L, R> BasisCoeffAdd<R> for L
where
    L: [const] Integer + [const] ArithAdd<L, R>,
    R: [const] Integer,
    Sum<L, R>: [const] BasisCoeff,
{
    type Output = Sum<L, R>;
    fn coeff_add(self, rhs: R) -> Self::Output {
        add(self, rhs)
    }
}

const impl<V> BasisCoeffNeg for V
where
    V: [const] ArithNeg,
    <V as ArithNeg>::Output: [const] BasisCoeff,
{
    type Output = <V as ArithNeg>::Output;
    fn coeff_neg(self) -> Self::Output {
        neg(self)
    }
}

const impl<L, R> BasisCoeffSub<R> for L
where
    L: [const] Integer + [const] ArithSub<L, R>,
    R: [const] Integer,
    Difference<L, R>: [const] BasisCoeff,
{
    type Output = Difference<L, R>;
    fn coeff_sub(self, rhs: R) -> Self::Output {
        sub(self, rhs)
    }
}

const impl<L, R> BasisCoeffMul<R> for L
where
    L: [const] Integer + [const] ArithMul<L, R>,
    R: [const] Integer,
    Product<L, R>: [const] BasisCoeff,
{
    type Output = Product<L, R>;
    fn coeff_mul(self, rhs: R) -> Self::Output {
        mul(self, rhs)
    }
}

/// Merge-adds two path-sorted `BasisTuple` chains (`Self` and `R`), pruning static zeros.
/// `ORDER` dispatches head comparison (`Dispatch` -> `Less`/`Equal`/`Greater`); omitting
/// `PrefixConflict` rejects incompatible path depths at compile time.
#[doc(hidden)]
pub const trait MergeAdd<
    R: [const] BasisTuple,
    const ORDER: PathOrdering = { PathOrdering::Dispatch },
>: [const] BasisTuple
{
    type Output: [const] BasisTuple;
    fn merge_add(self, rhs: R) -> Self::Output;
}

const impl MergeAdd<Nil> for Nil {
    type Output = Nil;
    fn merge_add(self, _: Nil) -> Self::Output {
        Nil
    }
}

const impl<V, P, T> MergeAdd<Nil> for Til<BasisTerm<V, P>, T>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    T: [const] MergeAdd<Nil>,
    <T as MergeAdd<Nil>>::Output: [const] EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>,
{
    type Output = <<T as MergeAdd<Nil>>::Output as EmitBasisTerm<
        V,
        P,
        { COEFF_IS_STATIC_ZERO::<V> },
    >>::Output;
    fn merge_add(self, _: Nil) -> Self::Output {
        self.1.merge_add(Nil).emit(self.0.value)
    }
}

const impl<V, P, T> MergeAdd<Til<BasisTerm<V, P>, T>> for Nil
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    T: [const] BasisTuple,
    Nil: [const] MergeAdd<T>,
    <Nil as MergeAdd<T>>::Output: [const] EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>,
{
    type Output = <<Nil as MergeAdd<T>>::Output as EmitBasisTerm<
        V,
        P,
        { COEFF_IS_STATIC_ZERO::<V> },
    >>::Output;
    fn merge_add(self, rhs: Til<BasisTerm<V, P>, T>) -> Self::Output {
        Nil.merge_add(rhs.1).emit(rhs.0.value)
    }
}

const impl<VL, PL, TL, VR, PR, TR> MergeAdd<Til<BasisTerm<VR, PR>, TR>>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: ComparePath<PR>,
    TL: [const] BasisTuple,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    Self: [const] MergeAdd<Til<BasisTerm<VR, PR>, TR>, { PATH_ORDER::<PL, PR> }>,
{
    type Output = <Self as MergeAdd<Til<BasisTerm<VR, PR>, TR>, { PATH_ORDER::<PL, PR> }>>::Output;
    fn merge_add(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Self::Output {
        <Self as MergeAdd<Til<BasisTerm<VR, PR>, TR>, { PATH_ORDER::<PL, PR> }>>::merge_add(
            self, rhs,
        )
    }
}

const impl<VL, PL, TL, VR, PR, TR> MergeAdd<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::Equal }>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff + [const] BasisCoeffAdd<VR>,
    PL: [const] BasisPath,
    TL: [const] BasisTuple + [const] MergeAdd<TR>,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    <TL as MergeAdd<TR>>::Output:
        [const] EmitBasisTerm<VL::Output, PL, { COEFF_IS_STATIC_ZERO::<VL::Output> }>,
{
    type Output = <<TL as MergeAdd<TR>>::Output as EmitBasisTerm<
        VL::Output,
        PL,
        { COEFF_IS_STATIC_ZERO::<VL::Output> },
    >>::Output;
    fn merge_add(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Self::Output {
        self.1
            .merge_add(rhs.1)
            .emit(self.0.value.coeff_add(rhs.0.value))
    }
}

const impl<VL, PL, TL, VR, PR, TR> MergeAdd<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::Less }>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: [const] BasisPath,
    TL: [const] BasisTuple + [const] MergeAdd<Til<BasisTerm<VR, PR>, TR>>,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    <TL as MergeAdd<Til<BasisTerm<VR, PR>, TR>>>::Output:
        [const] EmitBasisTerm<VL, PL, { COEFF_IS_STATIC_ZERO::<VL> }>,
{
    type Output = <<TL as MergeAdd<Til<BasisTerm<VR, PR>, TR>>>::Output as EmitBasisTerm<
        VL,
        PL,
        { COEFF_IS_STATIC_ZERO::<VL> },
    >>::Output;
    fn merge_add(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Self::Output {
        self.1.merge_add(rhs).emit(self.0.value)
    }
}

const impl<VL, PL, TL, VR, PR, TR> MergeAdd<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::Greater }>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: [const] BasisPath,
    TL: [const] BasisTuple,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    Self: [const] MergeAdd<TR>,
    <Self as MergeAdd<TR>>::Output: [const] EmitBasisTerm<VR, PR, { COEFF_IS_STATIC_ZERO::<VR> }>,
{
    type Output = <<Self as MergeAdd<TR>>::Output as EmitBasisTerm<
        VR,
        PR,
        { COEFF_IS_STATIC_ZERO::<VR> },
    >>::Output;
    fn merge_add(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Self::Output {
        self.merge_add(rhs.1).emit(rhs.0.value)
    }
}

/// Addition on basis elements, returning a [`CanonicalBasis`] result.
///
/// Prefix conflicts are rejected at compile time:
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::e;
/// let _ = e![0] + e![0, 1];
/// ```
///
/// Adding a non-static-zero scalar is also rejected at compile time:
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::e;
/// let _ = e![0] + 1i32;
/// ```
pub const trait BasisAdd<R>: [const] IntoBasisCombination {
    /// Canonical basis output type.
    type Output: Copy;
    /// Evaluates `self + rhs`.
    fn basis_add(self, rhs: R) -> Self::Output;
}

const impl<L, R> BasisAdd<R> for L
where
    L: [const] IntoBasisCombination,
    R: [const] IntoBasisCombination,
    L::Terms: [const] MergeAdd<R::Terms>,
    <L::Terms as MergeAdd<R::Terms>>::Output: [const] CanonicalBasis,
{
    type Output = <<L::Terms as MergeAdd<R::Terms>>::Output as CanonicalBasis>::Output;
    fn basis_add(self, rhs: R) -> Self::Output {
        self.into_basis_terms()
            .merge_add(rhs.into_basis_terms())
            .canonical_basis()
    }
}

/// Negates every term's coefficient in a `BasisTuple`, pruning static zeros.
#[doc(hidden)]
pub const trait NegBasisTuple: [const] BasisTuple {
    type Output: [const] BasisTuple;
    fn neg_tuple(self) -> Self::Output;
}

const impl NegBasisTuple for Nil {
    type Output = Nil;
    fn neg_tuple(self) -> Self::Output {
        Nil
    }
}

const impl<V, P, T> NegBasisTuple for Til<BasisTerm<V, P>, T>
where
    V: [const] BasisCoeff + [const] BasisCoeffNeg,
    P: [const] BasisPath,
    T: [const] NegBasisTuple,
    <T as NegBasisTuple>::Output:
        [const] EmitBasisTerm<V::Output, P, { ANY_STATIC_ZERO::<V, V::Output> }>,
{
    type Output = <<T as NegBasisTuple>::Output as EmitBasisTerm<
        V::Output,
        P,
        { ANY_STATIC_ZERO::<V, V::Output> },
    >>::Output;
    fn neg_tuple(self) -> Self::Output {
        self.1.neg_tuple().emit(self.0.value.coeff_neg())
    }
}

/// Unary negation on basis elements, returning a [`CanonicalBasis`] result.
///
/// Static negation of `Int<{ int::MIN }>` fails at compile time:
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{e, integer::Int};
/// let _ = -(Int::<-2_147_483_648> * e![0]);
/// ```
pub const trait BasisNeg: [const] IntoBasisCombination {
    /// Canonical basis output type.
    type Output: Copy;
    /// Evaluates `-self`.
    fn basis_neg(self) -> Self::Output;
}

const impl<T> BasisNeg for T
where
    T: [const] IntoBasisCombination,
    T::Terms: [const] NegBasisTuple,
    <T::Terms as NegBasisTuple>::Output: [const] CanonicalBasis,
{
    type Output = <<T::Terms as NegBasisTuple>::Output as CanonicalBasis>::Output;
    fn basis_neg(self) -> Self::Output {
        self.into_basis_terms().neg_tuple().canonical_basis()
    }
}

/// Merge-subtracts two path-sorted `BasisTuple` chains (`Self - R`), pruning static zeros.
/// `ORDER` dispatches head comparison; omitting `PrefixConflict` rejects incompatible depths.
#[doc(hidden)]
pub const trait MergeSub<
    R: [const] BasisTuple,
    const ORDER: PathOrdering = { PathOrdering::Dispatch },
>: [const] BasisTuple
{
    type Output: [const] BasisTuple;
    fn merge_sub(self, rhs: R) -> Self::Output;
}

const impl MergeSub<Nil> for Nil {
    type Output = Nil;
    fn merge_sub(self, _: Nil) -> Self::Output {
        Nil
    }
}

const impl<V, P, T> MergeSub<Nil> for Til<BasisTerm<V, P>, T>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    T: [const] MergeSub<Nil>,
    <T as MergeSub<Nil>>::Output: [const] EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>,
{
    type Output = <<T as MergeSub<Nil>>::Output as EmitBasisTerm<
        V,
        P,
        { COEFF_IS_STATIC_ZERO::<V> },
    >>::Output;
    fn merge_sub(self, _: Nil) -> Self::Output {
        self.1.merge_sub(Nil).emit(self.0.value)
    }
}

const impl<V, P, T> MergeSub<Til<BasisTerm<V, P>, T>> for Nil
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    T: [const] BasisTuple,
    Til<BasisTerm<V, P>, T>: [const] NegBasisTuple,
{
    type Output = <Til<BasisTerm<V, P>, T> as NegBasisTuple>::Output;
    fn merge_sub(self, rhs: Til<BasisTerm<V, P>, T>) -> Self::Output {
        rhs.neg_tuple()
    }
}

const impl<VL, PL, TL, VR, PR, TR> MergeSub<Til<BasisTerm<VR, PR>, TR>>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: ComparePath<PR>,
    TL: [const] BasisTuple,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    Self: [const] MergeSub<Til<BasisTerm<VR, PR>, TR>, { PATH_ORDER::<PL, PR> }>,
{
    type Output = <Self as MergeSub<Til<BasisTerm<VR, PR>, TR>, { PATH_ORDER::<PL, PR> }>>::Output;
    fn merge_sub(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Self::Output {
        <Self as MergeSub<Til<BasisTerm<VR, PR>, TR>, { PATH_ORDER::<PL, PR> }>>::merge_sub(
            self, rhs,
        )
    }
}

const impl<VL, PL, TL, VR, PR, TR> MergeSub<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::Equal }>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff + [const] BasisCoeffSub<VR>,
    PL: [const] BasisPath,
    TL: [const] BasisTuple + [const] MergeSub<TR>,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    <TL as MergeSub<TR>>::Output:
        [const] EmitBasisTerm<VL::Output, PL, { COEFF_IS_STATIC_ZERO::<VL::Output> }>,
{
    type Output = <<TL as MergeSub<TR>>::Output as EmitBasisTerm<
        VL::Output,
        PL,
        { COEFF_IS_STATIC_ZERO::<VL::Output> },
    >>::Output;
    fn merge_sub(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Self::Output {
        self.1
            .merge_sub(rhs.1)
            .emit(self.0.value.coeff_sub(rhs.0.value))
    }
}

const impl<VL, PL, TL, VR, PR, TR> MergeSub<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::Less }>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: [const] BasisPath,
    TL: [const] BasisTuple + [const] MergeSub<Til<BasisTerm<VR, PR>, TR>>,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    <TL as MergeSub<Til<BasisTerm<VR, PR>, TR>>>::Output:
        [const] EmitBasisTerm<VL, PL, { COEFF_IS_STATIC_ZERO::<VL> }>,
{
    type Output = <<TL as MergeSub<Til<BasisTerm<VR, PR>, TR>>>::Output as EmitBasisTerm<
        VL,
        PL,
        { COEFF_IS_STATIC_ZERO::<VL> },
    >>::Output;
    fn merge_sub(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Self::Output {
        self.1.merge_sub(rhs).emit(self.0.value)
    }
}

const impl<VL, PL, TL, VR, PR, TR> MergeSub<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::Greater }>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: [const] BasisPath,
    TL: [const] BasisTuple,
    VR: [const] BasisCoeff + [const] BasisCoeffNeg,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    Self: [const] MergeSub<TR>,
    <Self as MergeSub<TR>>::Output:
        [const] EmitBasisTerm<VR::Output, PR, { ANY_STATIC_ZERO::<VR, VR::Output> }>,
{
    type Output = <<Self as MergeSub<TR>>::Output as EmitBasisTerm<
        VR::Output,
        PR,
        { ANY_STATIC_ZERO::<VR, VR::Output> },
    >>::Output;
    fn merge_sub(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Self::Output {
        self.merge_sub(rhs.1).emit(rhs.0.value.coeff_neg())
    }
}

/// Subtraction on basis elements, returning a [`CanonicalBasis`] result.
pub const trait BasisSub<R>: [const] IntoBasisCombination {
    /// Canonical basis output type.
    type Output: Copy;
    /// Evaluates `self - rhs`.
    fn basis_sub(self, rhs: R) -> Self::Output;
}

const impl<L, R> BasisSub<R> for L
where
    L: [const] IntoBasisCombination,
    R: [const] IntoBasisCombination,
    L::Terms: [const] MergeSub<R::Terms>,
    <L::Terms as MergeSub<R::Terms>>::Output: [const] CanonicalBasis,
{
    type Output = <<L::Terms as MergeSub<R::Terms>>::Output as CanonicalBasis>::Output;
    fn basis_sub(self, rhs: R) -> Self::Output {
        self.into_basis_terms()
            .merge_sub(rhs.into_basis_terms())
            .canonical_basis()
    }
}

/// Scales every term of a `BasisTuple` by scalar `S`.
/// `ZERO = true` short-circuits static-zero scalars to `Nil`;
/// `LEFT = true` evaluates `S * coeff`, `LEFT = false` evaluates `coeff * S`.
#[doc(hidden)]
pub const trait ScaleBasisTerms<S: [const] BasisCoeff, const ZERO: bool, const LEFT: bool>:
    [const] BasisTuple
{
    type Output: [const] BasisTuple;
    fn scale(self, scalar: S) -> Self::Output;
}

const impl<T: [const] BasisTuple, S: [const] BasisCoeff, const LEFT: bool>
    ScaleBasisTerms<S, true, LEFT> for T
{
    type Output = Nil;
    fn scale(self, _: S) -> Self::Output {
        Nil
    }
}

const impl<S: [const] BasisCoeff, const LEFT: bool> ScaleBasisTerms<S, false, LEFT> for Nil {
    type Output = Nil;
    fn scale(self, _: S) -> Self::Output {
        Nil
    }
}

const impl<V, P, T, S> ScaleBasisTerms<S, false, false> for Til<BasisTerm<V, P>, T>
where
    V: [const] BasisCoeff + [const] BasisCoeffMul<S>,
    P: [const] BasisPath,
    T: [const] ScaleBasisTerms<S, false, false>,
    S: [const] BasisCoeff,
    <T as ScaleBasisTerms<S, false, false>>::Output:
        [const] EmitBasisTerm<V::Output, P, { ANY_STATIC_ZERO::<V, V::Output> }>,
{
    type Output = <<T as ScaleBasisTerms<S, false, false>>::Output as EmitBasisTerm<
        V::Output,
        P,
        { ANY_STATIC_ZERO::<V, V::Output> },
    >>::Output;
    fn scale(self, scalar: S) -> Self::Output {
        self.1.scale(scalar).emit(self.0.value.coeff_mul(scalar))
    }
}

const impl<V, P, T, S> ScaleBasisTerms<S, false, true> for Til<BasisTerm<V, P>, T>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    T: [const] ScaleBasisTerms<S, false, true>,
    S: [const] BasisCoeff + [const] BasisCoeffMul<V>,
    <T as ScaleBasisTerms<S, false, true>>::Output:
        [const] EmitBasisTerm<S::Output, P, { ANY_STATIC_ZERO::<V, S::Output> }>,
{
    type Output = <<T as ScaleBasisTerms<S, false, true>>::Output as EmitBasisTerm<
        S::Output,
        P,
        { ANY_STATIC_ZERO::<V, S::Output> },
    >>::Output;
    fn scale(self, scalar: S) -> Self::Output {
        self.1.scale(scalar).emit(scalar.coeff_mul(self.0.value))
    }
}

/// Heterogeneous scalar-basis multiplication preserving operand order (`self * rhs`).
///
/// Multiplying two basis elements is rejected at compile time:
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::e;
/// let _ = e![0] * e![1];
/// ```
pub const trait BasisMul<R: Copy>: Copy {
    /// Canonical basis output type.
    type Output: Copy;
    /// Evaluates `self * rhs`.
    fn basis_mul(self, rhs: R) -> Self::Output;
}

const impl<V, P, R> BasisMul<R> for BasisTerm<V, P>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    R: [const] BasisCoeff,
    Til<BasisTerm<V, P>, Nil>: [const] ScaleBasisTerms<R, { COEFF_IS_STATIC_ZERO::<R> }, false>,
    <Til<BasisTerm<V, P>, Nil> as ScaleBasisTerms<R, { COEFF_IS_STATIC_ZERO::<R> }, false>>::Output:
        [const] CanonicalBasis,
{
    type Output = <<Til<BasisTerm<V, P>, Nil> as ScaleBasisTerms<
        R,
        { COEFF_IS_STATIC_ZERO::<R> },
        false,
    >>::Output as CanonicalBasis>::Output;
    fn basis_mul(self, rhs: R) -> Self::Output {
        Til(self, Nil).scale(rhs).canonical_basis()
    }
}

const impl<T, R> BasisMul<R> for BasisCombination<T>
where
    T: [const] BasisTuple + [const] ScaleBasisTerms<R, { COEFF_IS_STATIC_ZERO::<R> }, false>,
    R: [const] BasisCoeff,
    <T as ScaleBasisTerms<R, { COEFF_IS_STATIC_ZERO::<R> }, false>>::Output: [const] CanonicalBasis,
{
    type Output = <<T as ScaleBasisTerms<R, { COEFF_IS_STATIC_ZERO::<R> }, false>>::Output as CanonicalBasis>::Output;
    fn basis_mul(self, rhs: R) -> Self::Output {
        self.0.scale(rhs).canonical_basis()
    }
}

const impl<L, V, P> BasisMul<BasisTerm<V, P>> for L
where
    L: [const] BasisCoeff,
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    Til<BasisTerm<V, P>, Nil>: [const] ScaleBasisTerms<L, { COEFF_IS_STATIC_ZERO::<L> }, true>,
    <Til<BasisTerm<V, P>, Nil> as ScaleBasisTerms<L, { COEFF_IS_STATIC_ZERO::<L> }, true>>::Output:
        [const] CanonicalBasis,
{
    type Output = <<Til<BasisTerm<V, P>, Nil> as ScaleBasisTerms<
        L,
        { COEFF_IS_STATIC_ZERO::<L> },
        true,
    >>::Output as CanonicalBasis>::Output;
    fn basis_mul(self, rhs: BasisTerm<V, P>) -> Self::Output {
        Til(rhs, Nil).scale(self).canonical_basis()
    }
}

const impl<L, T> BasisMul<BasisCombination<T>> for L
where
    L: [const] BasisCoeff,
    T: [const] BasisTuple + [const] ScaleBasisTerms<L, { COEFF_IS_STATIC_ZERO::<L> }, true>,
    <T as ScaleBasisTerms<L, { COEFF_IS_STATIC_ZERO::<L> }, true>>::Output: [const] CanonicalBasis,
{
    type Output = <<T as ScaleBasisTerms<L, { COEFF_IS_STATIC_ZERO::<L> }, true>>::Output as CanonicalBasis>::Output;
    fn basis_mul(self, rhs: BasisCombination<T>) -> Self::Output {
        rhs.0.scale(self).canonical_basis()
    }
}

macro_rules! impl_op {
    ($Trait:ident $method:ident $BasisTrait:ident $basis_method:ident; [$($g:tt)*]; $lhs:ty; $rhs:ty) => {
        const impl<$($g)*> $Trait<$rhs> for $lhs
        where
            Self: [const] $BasisTrait<$rhs>,
        {
            type Output = <Self as $BasisTrait<$rhs>>::Output;
            fn $method(self, rhs: $rhs) -> Self::Output {
                self.$basis_method(rhs)
            }
        }
    };
}

macro_rules! impl_add_sub_op {
    ([$($g:tt)*]; $lhs:ty; $rhs:ty) => {
        impl_op!(Add add BasisAdd basis_add; [$($g)*]; $lhs; $rhs);
        impl_op!(Sub sub BasisSub basis_sub; [$($g)*]; $lhs; $rhs);
    };
}

impl_add_sub_op!([V: [const] BasisCoeff, P: [const] BasisPath, R]; BasisTerm<V, P>; R);
impl_add_sub_op!([T: [const] BasisTuple, R]; BasisCombination<T>; R);
impl_add_sub_op!([V: [const] BasisCoeff, P: [const] BasisPath]; Int<0>; BasisTerm<V, P>);
impl_add_sub_op!([T: [const] BasisTuple]; Int<0>; BasisCombination<T>);

const impl<V: [const] BasisCoeff, P: [const] BasisPath> Neg for BasisTerm<V, P>
where
    Self: [const] BasisNeg,
{
    type Output = <Self as BasisNeg>::Output;
    fn neg(self) -> Self::Output {
        self.basis_neg()
    }
}

const impl<T: [const] BasisTuple> Neg for BasisCombination<T>
where
    Self: [const] BasisNeg,
{
    type Output = <Self as BasisNeg>::Output;
    fn neg(self) -> Self::Output {
        self.basis_neg()
    }
}

impl_op!(Mul mul BasisMul basis_mul; [V: [const] BasisCoeff, P: [const] BasisPath, R: Copy]; BasisTerm<V, P>; R);
impl_op!(Mul mul BasisMul basis_mul; [T: [const] BasisTuple, R: Copy]; BasisCombination<T>; R);

macro_rules! impl_left_mul {
    ([$($g:tt)+]; $lhs:ty) => {
        impl_op!(Mul mul BasisMul basis_mul; [$($g)+, V: [const] BasisCoeff, P: [const] BasisPath]; $lhs; BasisTerm<V, P>);
        impl_op!(Mul mul BasisMul basis_mul; [$($g)+, T: [const] BasisTuple]; $lhs; BasisCombination<T>);
    };
    ([]; $lhs:ty) => {
        impl_op!(Mul mul BasisMul basis_mul; [V: [const] BasisCoeff, P: [const] BasisPath]; $lhs; BasisTerm<V, P>);
        impl_op!(Mul mul BasisMul basis_mul; [T: [const] BasisTuple]; $lhs; BasisCombination<T>);
    };
}

impl_left_mul!([const N: int]; Int<N>);
impl_left_mul!([]; int);
