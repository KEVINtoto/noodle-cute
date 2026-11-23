//! Algebraic basis decomposition (`basis_repr`, `is_basis`) and single-basis
//! accessors (`proj`, `unit`).
//!
//! Exposes a uniform non-empty term decomposition for scalars, [`BasisTerm`], and
//! [`BasisCombination`], along with path projection and unit-basis extraction for
//! single-term basis elements.
//!
//! # Main Interfaces
//!
//! - [`BasisRepr`] / [`basis_repr`]: Decomposes any scalar or basis element into a
//!   non-empty [`BasisTuple`] of `(value, path)` terms.
//! - [`is_basis`]: Checks whether an element has exactly one term (including scalars and zero).
//! - [`SingleBasis`]: Trait implemented only by single-term basis elements.
//! - [`proj`]: Projects a tuple or basis element along a single basis element's path.
//! - [`unit()`]: Returns the unit basis element matching a single basis element's algebra and path.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{basis::{is_basis, proj, unit}, e, int_tuple};
//!
//! assert!(is_basis(5i32 * e![1, 0]));
//! assert!(!is_basis(e![0] + e![1]));
//! assert_eq!(proj(int_tuple![7, (8, 9)], 2i32 * e![1, 0]), 8);
//! assert_eq!(unit(2i32 * e![1]), e![1]);
//! ```
//!
//! # Notes
//!
//! - All-zero inputs (`Int<0>`, `BasisCombination::zero()`, `BasisTerm<Int<0>, P>`)
//!   normalize to `[(Int<0>, ModeEnd)]` in [`basis_repr`] and have path [`ModeEnd`]
//!   under [`SingleBasis`].
//! - Passing a multi-term [`BasisCombination`] to [`proj`] or [`unit()`] fails at compile time.

use crate::integer::{Int, eq as scalar_eq, int};
use crate::tuple::{Get, GetOutput, Item, Mode, ModeEnd, Nil, Til};

use super::{
    BasisCoeff, BasisCombination, BasisPath, BasisTerm, BasisTuple, COEFF_IS_STATIC_ZERO,
    EmitBasisTerm,
};

// Ensures a `BasisTuple` is non-empty by mapping `Nil` to a single `(Int<0>, ModeEnd)` term.
#[doc(hidden)]
pub const trait NonEmptyBasisRepr: [const] BasisTuple {
    type Output: [const] BasisTuple;
    fn non_empty_repr(self) -> Self::Output;
}

const impl NonEmptyBasisRepr for Nil {
    type Output = Til<BasisTerm<Int<0>, ModeEnd>, Nil>;
    fn non_empty_repr(self) -> Self::Output {
        Til(BasisTerm::new(Int), Nil)
    }
}

const impl<V: [const] BasisCoeff, P: [const] BasisPath, T: [const] BasisTuple> NonEmptyBasisRepr
    for Til<BasisTerm<V, P>, T>
{
    type Output = Self;
    fn non_empty_repr(self) -> Self::Output {
        self
    }
}

/// Decomposes a scalar, basis term, or combination into a non-empty [`FlatTuple`](crate::tuple::FlatTuple) of
/// `(value, path)` terms, collapsing all-zero inputs to `[(Int<0>, ModeEnd)]`.
pub const trait BasisRepr: Copy {
    /// Non-empty [`BasisTuple`] chain representing `Self`.
    type Output: [const] BasisTuple;
    /// Returns the non-empty `(value, path)` term list.
    fn basis_repr(self) -> Self::Output;
}

const impl<V: [const] BasisCoeff> BasisRepr for V
where
    Nil: [const] EmitBasisTerm<V, ModeEnd, { COEFF_IS_STATIC_ZERO::<V> }>,
    <Nil as EmitBasisTerm<V, ModeEnd, { COEFF_IS_STATIC_ZERO::<V> }>>::Output:
        [const] NonEmptyBasisRepr,
{
    type Output = <<Nil as EmitBasisTerm<V, ModeEnd, { COEFF_IS_STATIC_ZERO::<V> }>>::Output as NonEmptyBasisRepr>::Output;
    fn basis_repr(self) -> Self::Output {
        Nil.emit(self).non_empty_repr()
    }
}

const impl<V: [const] BasisCoeff, P: [const] BasisPath> BasisRepr for BasisTerm<V, P>
where
    Nil: [const] EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>,
    <Nil as EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>>::Output: [const] NonEmptyBasisRepr,
{
    type Output = <<Nil as EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>>::Output as NonEmptyBasisRepr>::Output;
    fn basis_repr(self) -> Self::Output {
        Nil.emit(self.value).non_empty_repr()
    }
}

const impl<T: [const] NonEmptyBasisRepr> BasisRepr for BasisCombination<T> {
    type Output = <T as NonEmptyBasisRepr>::Output;
    fn basis_repr(self) -> Self::Output {
        self.0.non_empty_repr()
    }
}

/// Decomposes `x` into a non-empty [`FlatTuple`](crate::tuple::FlatTuple) of [`BasisTerm`] entries.
pub const fn basis_repr<T: [const] BasisRepr>(x: T) -> T::Output {
    x.basis_repr()
}

/// Returns `true` iff `x` represents a single scaled basis element (including scalars and zero).
pub const fn is_basis<T: BasisRepr>(_: T) -> bool {
    scalar_eq(T::Output::RANK, 1)
}

// Resolves path and unit basis for `BasisTerm<V, P>`; `ZERO = true` collapses static zeros to `ModeEnd`.
#[doc(hidden)]
pub const trait SingleBasisTerm<V: [const] BasisCoeff, P: [const] BasisPath, const ZERO: bool> {
    type Path: [const] BasisPath;
    type Unit: Copy;
    fn unit_basis_term(self) -> Self::Unit;
}

const impl<V: [const] BasisCoeff, P: [const] BasisPath> SingleBasisTerm<V, P, true>
    for BasisTerm<V, P>
{
    type Path = ModeEnd;
    type Unit = V::Unit;
    fn unit_basis_term(self) -> Self::Unit {
        V::algebra_unit()
    }
}

const impl<V: [const] BasisCoeff, P: [const] BasisPath> SingleBasisTerm<V, P, false>
    for BasisTerm<V, P>
{
    type Path = P;
    type Unit = BasisTerm<V::Unit, P>;
    fn unit_basis_term(self) -> Self::Unit {
        BasisTerm::new(V::algebra_unit())
    }
}

/// Trait implemented only for single-term basis elements (scalars, [`BasisTerm`], and
/// [`BasisCombination<Nil>`]), excluding multi-term combinations.
pub const trait SingleBasis: Copy {
    /// Coordinate path of this basis element (`ModeEnd` for scalars and zero).
    type Path: [const] BasisPath;
    /// Multiplicative identity basis type in this element's algebra and path.
    type Unit: Copy;
    /// Returns the unit basis element along `Self::Path`.
    fn unit_basis(self) -> Self::Unit;
}

const impl<V: [const] BasisCoeff> SingleBasis for V {
    type Path = ModeEnd;
    type Unit = V::Unit;
    fn unit_basis(self) -> Self::Unit {
        V::algebra_unit()
    }
}

const impl SingleBasis for BasisCombination<Nil> {
    type Path = ModeEnd;
    type Unit = Int<1>;
    fn unit_basis(self) -> Self::Unit {
        Int
    }
}

const impl<V: [const] BasisCoeff, P: [const] BasisPath> SingleBasis for BasisTerm<V, P>
where
    Self: [const] SingleBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>,
{
    type Path = <Self as SingleBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>>::Path;
    type Unit = <Self as SingleBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>>::Unit;
    fn unit_basis(self) -> Self::Unit {
        self.unit_basis_term()
    }
}

// Recursively indexes an `Item` along type-level `BasisPath` `P` using `Get`.
#[doc(hidden)]
pub const trait ProjectByPath<P: [const] BasisPath>: [const] Item {
    type Output: [const] Item;
    fn project_path(&self) -> Self::Output;
}

const impl<X: [const] Item> ProjectByPath<ModeEnd> for X {
    type Output = X;
    fn project_path(&self) -> Self::Output {
        *self
    }
}

const impl<X, const M: int, Next> ProjectByPath<Mode<M, Next>> for X
where
    Mode<M, Next>: [const] BasisPath,
    X: [const] Get<M>,
    Next: [const] BasisPath,
    GetOutput<X, M>: [const] ProjectByPath<Next>,
{
    type Output = <GetOutput<X, M> as ProjectByPath<Next>>::Output;
    fn project_path(&self) -> Self::Output {
        Get::<M>::get(self).project_path()
    }
}

/// Extracts the component of `x` along the coordinate path of single basis element `basis`.
///
/// Multi-term basis combinations fail at compile time:
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{basis::proj, e, tuple};
/// let _ = proj(tuple![1i32, 2i32], e![0] + e![1]);
/// ```
pub const fn proj<X, B>(x: X, _: B) -> <X as ProjectByPath<B::Path>>::Output
where
    B: [const] SingleBasis,
    X: [const] ProjectByPath<B::Path>,
{
    x.project_path()
}

/// Returns the multiplicative identity in `basis`'s algebra along `basis`'s coordinate path.
///
/// Multi-term basis combinations fail at compile time:
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{basis::unit, e};
/// let _ = unit(e![0] + e![1]);
/// ```
pub const fn unit<B: [const] SingleBasis>(basis: B) -> B::Unit {
    basis.unit_basis()
}
