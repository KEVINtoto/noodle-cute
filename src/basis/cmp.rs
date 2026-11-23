//! Zero-padded equality and colexicographical ordering for basis elements and scalars.
//!
//! Compares sparse [`BasisTerm`], [`BasisCombination`], and scalar values mathematically
//! by treating missing modes as zero, with both runtime/`const` value comparisons and
//! static type-level queries.
//!
//! # Main Interfaces
//!
//! - **Coefficient comparison traits**: [`BasisCoeffEq`] and [`BasisCoeffOrd`].
//! - **Zero-padded equality**: [`basis_eq`] and [`static_basis_eq`].
//! - **Colexicographical ordering**: [`colex_cmp`], [`static_colex_cmp`], [`colex_lt`],
//!   [`colex_le`], [`colex_gt`], and [`colex_ge`].
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use core::cmp::Ordering;
//! use noodle_cute::{basis::{basis_eq, colex_cmp, static_colex_cmp}, basis_tuple, e, integer::Int};
//!
//! assert!(basis_eq(e![0] + 0i32 * e![1], e![0]));
//! assert_eq!(colex_cmp(basis_tuple![100i32, 2i32], basis_tuple![1i32, 3i32]), Ordering::Less);
//! assert_eq!(
//!     static_colex_cmp(basis_tuple![100i32, Int::<2>], basis_tuple![1i32, Int::<3>]),
//!     Some(Ordering::Less)
//! );
//! ```
//!
//! # Notes
//!
//! - [`basis_eq`] treats omitted modes and dynamic zeros as equal (even across
//!   prefix-conflicting zero subtrees), whereas `PartialEq` compares exact term chains.
//! - [`colex_cmp`] orders by the highest differing coordinate path first and rejects
//!   [`PathOrdering::PrefixConflict`] (including dynamic scalars vs basis terms) at compile time.
//! - [`static_basis_eq`] and [`static_colex_cmp`] return `None` when dynamic coefficients
//!   prevent a static decision.

use core::cmp::Ordering;

use crate::integer::{Integer, cmp, eq, static_and, static_cmp, static_cmp_of, static_eq};
use crate::tuple::{Nil, Til};

use super::{
    BasisCoeff, BasisCombination, BasisPath, BasisTerm, BasisTuple, COEFF_IS_STATIC_ZERO,
    ComparePath, EmitBasisTerm, IS_LEFT_PREFIX, ModeEnd, PATH_ORDER, PathOrdering,
};

/// Coefficient equality (value-level and static three-valued) for [`BasisCoeff`].
pub const trait BasisCoeffEq<R: BasisCoeff>: BasisCoeff {
    /// Returns `true` iff `self` and `rhs` have equal values.
    fn coeff_eq(&self, rhs: &R) -> bool;
    /// Compile-time three-valued equality check (`None` if dynamic).
    fn coeff_static_eq(&self, rhs: &R) -> Option<bool>;
}

/// Coefficient total ordering (value-level and static) for [`BasisCoeff`].
pub const trait BasisCoeffOrd<R: BasisCoeff>: BasisCoeff {
    /// Compile-time ordering between `Self` and `R` (`None` if dynamic).
    const STATIC_CMP: Option<Ordering> = None;
    /// Compares `self` and `rhs`.
    fn coeff_cmp(&self, rhs: &R) -> Ordering;
    /// Compile-time comparison between `self` and `rhs` (`None` if dynamic).
    fn coeff_static_cmp(&self, rhs: &R) -> Option<Ordering>;
}

const impl<L: [const] Integer, R: [const] Integer> BasisCoeffEq<R> for L {
    fn coeff_eq(&self, rhs: &R) -> bool {
        eq(*self, *rhs)
    }
    fn coeff_static_eq(&self, rhs: &R) -> Option<bool> {
        static_eq(*self, *rhs)
    }
}

const impl<L: [const] Integer, R: [const] Integer> BasisCoeffOrd<R> for L {
    const STATIC_CMP: Option<Ordering> = static_cmp_of::<L, R>();
    fn coeff_cmp(&self, rhs: &R) -> Ordering {
        cmp(*self, *rhs)
    }
    fn coeff_static_cmp(&self, rhs: &R) -> Option<Ordering> {
        static_cmp(*self, *rhs)
    }
}

const fn static_reverse(ord: Option<Ordering>) -> Option<Ordering> {
    match ord {
        Some(o) => Some(o.reverse()),
        None => None,
    }
}

// Normalizes scalars, `BasisTerm`, and `BasisCombination` into a zero-stripped `BasisTuple`
// for `basis_eq` and `colex_cmp`.
#[doc(hidden)]
pub const trait IntoColexTerms: Copy {
    type Terms: [const] BasisTuple;
    fn into_colex_terms(self) -> Self::Terms;
}

const impl<V: [const] BasisCoeff> IntoColexTerms for V
where
    Nil: [const] EmitBasisTerm<V, ModeEnd, { COEFF_IS_STATIC_ZERO::<V> }>,
{
    type Terms = <Nil as EmitBasisTerm<V, ModeEnd, { COEFF_IS_STATIC_ZERO::<V> }>>::Output;
    fn into_colex_terms(self) -> Self::Terms {
        Nil.emit(self)
    }
}

const impl<V: [const] BasisCoeff, P: [const] BasisPath> IntoColexTerms for BasisTerm<V, P>
where
    Nil: [const] EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>,
{
    type Terms = <Nil as EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>>::Output;
    fn into_colex_terms(self) -> Self::Terms {
        Nil.emit(self.value)
    }
}

const impl<T: [const] BasisTuple> IntoColexTerms for BasisCombination<T> {
    type Terms = T;
    fn into_colex_terms(self) -> Self::Terms {
        self.0
    }
}

// Zero-padded structural equality over path-sorted `BasisTuple` chains (`Self` and `R`).
// `ORDER` dispatches head comparison; `LEFT_PREFIX` chooses which side to zero-check and
// advance on `PathOrdering::PrefixConflict`.
#[doc(hidden)]
pub const trait MergeEq<
    R: [const] BasisTuple,
    const ORDER: PathOrdering = { PathOrdering::Dispatch },
    const LEFT_PREFIX: bool = false,
>: [const] BasisTuple
{
    fn merge_eq(self, rhs: R) -> bool;
    fn static_merge_eq(self, rhs: R) -> Option<bool>;
}

const impl MergeEq<Nil> for Nil {
    fn merge_eq(self, _: Nil) -> bool {
        true
    }
    fn static_merge_eq(self, _: Nil) -> Option<bool> {
        Some(true)
    }
}

const impl<V, P, T> MergeEq<Nil> for Til<BasisTerm<V, P>, T>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    T: [const] MergeEq<Nil>,
{
    fn merge_eq(self, _: Nil) -> bool {
        self.0.value.is_zero() && self.1.merge_eq(Nil)
    }
    fn static_merge_eq(self, _: Nil) -> Option<bool> {
        static_and(self.0.value.static_is_zero(), self.1.static_merge_eq(Nil))
    }
}

const impl<V, P, T> MergeEq<Til<BasisTerm<V, P>, T>> for Nil
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    T: [const] BasisTuple,
    Nil: [const] MergeEq<T>,
{
    fn merge_eq(self, rhs: Til<BasisTerm<V, P>, T>) -> bool {
        rhs.0.value.is_zero() && Nil.merge_eq(rhs.1)
    }
    fn static_merge_eq(self, rhs: Til<BasisTerm<V, P>, T>) -> Option<bool> {
        static_and(rhs.0.value.static_is_zero(), Nil.static_merge_eq(rhs.1))
    }
}

const impl<VL, PL, TL, VR, PR, TR> MergeEq<Til<BasisTerm<VR, PR>, TR>>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: ComparePath<PR>,
    TL: [const] BasisTuple,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    Self: [const] MergeEq<
            Til<BasisTerm<VR, PR>, TR>,
            { PATH_ORDER::<PL, PR> },
            { IS_LEFT_PREFIX::<PL, PR> },
        >,
{
    fn merge_eq(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> bool {
        <Self as MergeEq<
            Til<BasisTerm<VR, PR>, TR>,
            { PATH_ORDER::<PL, PR> },
            { IS_LEFT_PREFIX::<PL, PR> },
        >>::merge_eq(self, rhs)
    }
    fn static_merge_eq(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Option<bool> {
        <Self as MergeEq<
            Til<BasisTerm<VR, PR>, TR>,
            { PATH_ORDER::<PL, PR> },
            { IS_LEFT_PREFIX::<PL, PR> },
        >>::static_merge_eq(self, rhs)
    }
}

const impl<VL, PL, TL, VR, PR, TR, const LP: bool>
    MergeEq<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::Equal }, LP> for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff + [const] BasisCoeffEq<VR>,
    PL: [const] BasisPath,
    TL: [const] MergeEq<TR>,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
{
    fn merge_eq(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> bool {
        self.0.value.coeff_eq(&rhs.0.value) && self.1.merge_eq(rhs.1)
    }
    fn static_merge_eq(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Option<bool> {
        static_and(
            self.0.value.coeff_static_eq(&rhs.0.value),
            self.1.static_merge_eq(rhs.1),
        )
    }
}

const impl<VL, PL, TL, VR, PR, TR, const LP: bool>
    MergeEq<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::Less }, LP> for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: [const] BasisPath,
    TL: [const] MergeEq<Til<BasisTerm<VR, PR>, TR>>,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
{
    fn merge_eq(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> bool {
        self.0.value.is_zero() && self.1.merge_eq(rhs)
    }
    fn static_merge_eq(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Option<bool> {
        static_and(self.0.value.static_is_zero(), self.1.static_merge_eq(rhs))
    }
}

const impl<VL, PL, TL, VR, PR, TR, const LP: bool>
    MergeEq<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::Greater }, LP>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: [const] BasisPath,
    TL: [const] BasisTuple,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    Self: [const] MergeEq<TR>,
{
    fn merge_eq(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> bool {
        rhs.0.value.is_zero() && self.merge_eq(rhs.1)
    }
    fn static_merge_eq(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Option<bool> {
        static_and(rhs.0.value.static_is_zero(), self.static_merge_eq(rhs.1))
    }
}

const impl<VL, PL, TL, VR, PR, TR>
    MergeEq<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::PrefixConflict }, true>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: [const] BasisPath,
    TL: [const] MergeEq<Til<BasisTerm<VR, PR>, TR>>,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
{
    fn merge_eq(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> bool {
        self.0.value.is_zero() && self.1.merge_eq(rhs)
    }
    fn static_merge_eq(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Option<bool> {
        static_and(self.0.value.static_is_zero(), self.1.static_merge_eq(rhs))
    }
}

const impl<VL, PL, TL, VR, PR, TR>
    MergeEq<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::PrefixConflict }, false>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: [const] BasisPath,
    TL: [const] BasisTuple,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    Self: [const] MergeEq<TR>,
{
    fn merge_eq(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> bool {
        rhs.0.value.is_zero() && self.merge_eq(rhs.1)
    }
    fn static_merge_eq(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Option<bool> {
        static_and(rhs.0.value.static_is_zero(), self.static_merge_eq(rhs.1))
    }
}

/// Zero-padded mathematical equality between two basis elements or scalars.
pub const fn basis_eq<L, R>(lhs: L, rhs: R) -> bool
where
    L: [const] IntoColexTerms,
    R: [const] IntoColexTerms,
    L::Terms: [const] MergeEq<R::Terms>,
{
    lhs.into_colex_terms().merge_eq(rhs.into_colex_terms())
}

/// Compile-time three-valued zero-padded equality (`None` if statically undecided).
pub const fn static_basis_eq<L, R>(lhs: L, rhs: R) -> Option<bool>
where
    L: [const] IntoColexTerms,
    R: [const] IntoColexTerms,
    L::Terms: [const] MergeEq<R::Terms>,
{
    lhs.into_colex_terms()
        .static_merge_eq(rhs.into_colex_terms())
}

// Single-pass colexicographical comparison over path-sorted `BasisTuple` chains (`Self` vs `R`),
// comparing tails (higher paths) before heads. `ORDER` dispatches head comparison; omitting
// `PrefixConflict` rejects incompatible depths at compile time.
#[doc(hidden)]
pub const trait MergeColex<
    R: [const] BasisTuple,
    const ORDER: PathOrdering = { PathOrdering::Dispatch },
>: [const] BasisTuple
{
    const STATIC_ORDER: Option<Ordering>;
    fn merge_colex(self, rhs: R) -> Ordering;
    fn static_merge_colex(self, rhs: R) -> Option<Ordering>;
}

const impl MergeColex<Nil> for Nil {
    const STATIC_ORDER: Option<Ordering> = Some(Ordering::Equal);
    fn merge_colex(self, _: Nil) -> Ordering {
        Ordering::Equal
    }
    fn static_merge_colex(self, _: Nil) -> Option<Ordering> {
        Some(Ordering::Equal)
    }
}

const impl<V, P, T> MergeColex<Nil> for Til<BasisTerm<V, P>, T>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    T: [const] MergeColex<Nil>,
{
    const STATIC_ORDER: Option<Ordering> = match T::STATIC_ORDER {
        Some(Ordering::Equal) => V::STATIC_CMP_ZERO,
        other => other,
    };
    fn merge_colex(self, _: Nil) -> Ordering {
        let tail = self.1.merge_colex(Nil);
        if matches!(tail, Ordering::Equal) {
            self.0.value.cmp_zero()
        } else {
            tail
        }
    }
    fn static_merge_colex(self, _: Nil) -> Option<Ordering> {
        match self.1.static_merge_colex(Nil) {
            Some(Ordering::Equal) => self.0.value.static_cmp_zero(),
            other => other,
        }
    }
}

const impl<V, P, T> MergeColex<Til<BasisTerm<V, P>, T>> for Nil
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    T: [const] BasisTuple,
    Nil: [const] MergeColex<T>,
{
    const STATIC_ORDER: Option<Ordering> = match <Nil as MergeColex<T>>::STATIC_ORDER {
        Some(Ordering::Equal) => static_reverse(V::STATIC_CMP_ZERO),
        other => other,
    };
    fn merge_colex(self, rhs: Til<BasisTerm<V, P>, T>) -> Ordering {
        let tail = Nil.merge_colex(rhs.1);
        if matches!(tail, Ordering::Equal) {
            rhs.0.value.cmp_zero().reverse()
        } else {
            tail
        }
    }
    fn static_merge_colex(self, rhs: Til<BasisTerm<V, P>, T>) -> Option<Ordering> {
        match Nil.static_merge_colex(rhs.1) {
            Some(Ordering::Equal) => static_reverse(rhs.0.value.static_cmp_zero()),
            other => other,
        }
    }
}

const impl<VL, PL, TL, VR, PR, TR> MergeColex<Til<BasisTerm<VR, PR>, TR>>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: ComparePath<PR>,
    TL: [const] BasisTuple,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    Self: [const] MergeColex<Til<BasisTerm<VR, PR>, TR>, { PATH_ORDER::<PL, PR> }>,
{
    const STATIC_ORDER: Option<Ordering> =
        <Self as MergeColex<Til<BasisTerm<VR, PR>, TR>, { PATH_ORDER::<PL, PR> }>>::STATIC_ORDER;
    fn merge_colex(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Ordering {
        <Self as MergeColex<Til<BasisTerm<VR, PR>, TR>, { PATH_ORDER::<PL, PR> }>>::merge_colex(
            self, rhs,
        )
    }
    fn static_merge_colex(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Option<Ordering> {
        <Self as MergeColex<Til<BasisTerm<VR, PR>, TR>, { PATH_ORDER::<PL, PR> }>>::static_merge_colex(
            self, rhs,
        )
    }
}

const impl<VL, PL, TL, VR, PR, TR> MergeColex<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::Equal }>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff + [const] BasisCoeffOrd<VR>,
    PL: [const] BasisPath,
    TL: [const] MergeColex<TR>,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
{
    const STATIC_ORDER: Option<Ordering> = match <TL as MergeColex<TR>>::STATIC_ORDER {
        Some(Ordering::Equal) => <VL as BasisCoeffOrd<VR>>::STATIC_CMP,
        other => other,
    };
    fn merge_colex(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Ordering {
        let tail = self.1.merge_colex(rhs.1);
        if matches!(tail, Ordering::Equal) {
            self.0.value.coeff_cmp(&rhs.0.value)
        } else {
            tail
        }
    }
    fn static_merge_colex(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Option<Ordering> {
        match self.1.static_merge_colex(rhs.1) {
            Some(Ordering::Equal) => self.0.value.coeff_static_cmp(&rhs.0.value),
            other => other,
        }
    }
}

const impl<VL, PL, TL, VR, PR, TR> MergeColex<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::Less }>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: [const] BasisPath,
    TL: [const] MergeColex<Til<BasisTerm<VR, PR>, TR>>,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
{
    const STATIC_ORDER: Option<Ordering> =
        match <TL as MergeColex<Til<BasisTerm<VR, PR>, TR>>>::STATIC_ORDER {
            Some(Ordering::Equal) => VL::STATIC_CMP_ZERO,
            other => other,
        };
    fn merge_colex(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Ordering {
        let tail = self.1.merge_colex(rhs);
        if matches!(tail, Ordering::Equal) {
            self.0.value.cmp_zero()
        } else {
            tail
        }
    }
    fn static_merge_colex(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Option<Ordering> {
        match self.1.static_merge_colex(rhs) {
            Some(Ordering::Equal) => self.0.value.static_cmp_zero(),
            other => other,
        }
    }
}

const impl<VL, PL, TL, VR, PR, TR> MergeColex<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::Greater }>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: [const] BasisPath,
    TL: [const] BasisTuple,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    Self: [const] MergeColex<TR>,
{
    const STATIC_ORDER: Option<Ordering> = match <Self as MergeColex<TR>>::STATIC_ORDER {
        Some(Ordering::Equal) => static_reverse(VR::STATIC_CMP_ZERO),
        other => other,
    };
    fn merge_colex(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Ordering {
        let tail = self.merge_colex(rhs.1);
        if matches!(tail, Ordering::Equal) {
            rhs.0.value.cmp_zero().reverse()
        } else {
            tail
        }
    }
    fn static_merge_colex(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Option<Ordering> {
        match self.static_merge_colex(rhs.1) {
            Some(Ordering::Equal) => static_reverse(rhs.0.value.static_cmp_zero()),
            other => other,
        }
    }
}

/// Colexicographical ordering between two basis elements or scalars.
///
/// Incompatible nesting depths (`PrefixConflict`) are rejected at compile time:
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{basis::colex_cmp, e};
/// let _ = colex_cmp(e![0], e![0, 1]);
/// ```
///
/// Comparing a dynamic (non-static-zero) scalar against a basis element also fails at compile time:
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{basis::colex_cmp, e};
/// let _ = colex_cmp(0i32, e![0]);
/// ```
pub const fn colex_cmp<L, R>(lhs: L, rhs: R) -> Ordering
where
    L: [const] IntoColexTerms,
    R: [const] IntoColexTerms,
    L::Terms: [const] MergeColex<R::Terms>,
{
    lhs.into_colex_terms().merge_colex(rhs.into_colex_terms())
}

/// Compile-time static colexicographical ordering (`None` if statically undecided).
pub const fn static_colex_cmp<L, R>(lhs: L, rhs: R) -> Option<Ordering>
where
    L: [const] IntoColexTerms,
    R: [const] IntoColexTerms,
    L::Terms: [const] MergeColex<R::Terms>,
{
    lhs.into_colex_terms()
        .static_merge_colex(rhs.into_colex_terms())
}

/// Returns `true` iff `lhs < rhs` in colexicographical order.
pub const fn colex_lt<L, R>(lhs: L, rhs: R) -> bool
where
    L: [const] IntoColexTerms,
    R: [const] IntoColexTerms,
    L::Terms: [const] MergeColex<R::Terms>,
{
    matches!(colex_cmp(lhs, rhs), Ordering::Less)
}

/// Returns `true` iff `lhs <= rhs` in colexicographical order.
pub const fn colex_le<L, R>(lhs: L, rhs: R) -> bool
where
    L: [const] IntoColexTerms,
    R: [const] IntoColexTerms,
    L::Terms: [const] MergeColex<R::Terms>,
{
    !matches!(colex_cmp(lhs, rhs), Ordering::Greater)
}

/// Returns `true` iff `lhs > rhs` in colexicographical order.
pub const fn colex_gt<L, R>(lhs: L, rhs: R) -> bool
where
    L: [const] IntoColexTerms,
    R: [const] IntoColexTerms,
    L::Terms: [const] MergeColex<R::Terms>,
{
    matches!(colex_cmp(lhs, rhs), Ordering::Greater)
}

/// Returns `true` iff `lhs >= rhs` in colexicographical order.
pub const fn colex_ge<L, R>(lhs: L, rhs: R) -> bool
where
    L: [const] IntoColexTerms,
    R: [const] IntoColexTerms,
    L::Terms: [const] MergeColex<R::Terms>,
{
    !matches!(colex_cmp(lhs, rhs), Ordering::Less)
}
