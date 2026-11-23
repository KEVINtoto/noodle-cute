//! Sparse, path-sorted coordinate tuples and basis vectors.
//!
//! [`BasisCombination`] stores a [`FlatTuple`] chain of [`BasisTerm<V, P>`] leaves
//! sorted in strict lexicographical order of their type-level coordinate paths
//! `P` ([`ModeEnd`] and [`Mode<M, Next>`]), omitting static zeros.
//!
//! # Core Types & Submodules
//!
//! - **Core types**: [`BasisCoeff`], [`BasisTerm<V, P>`], [`ScaledBasis<V, P>`],
//!   [`Basis<P>`], [`BasisTuple`], [`BasisCombination<T>`], and canonicalization
//!   ([`CanonicalBasis`], [`CanonicalBasisTuple`], [`IntoBasisCombination`]).
//! - [`mod@path`]: Type-level non-negative paths ([`BasisPath`], [`PathOrdering`], [`ComparePath`]).
//! - [`mod@ops`]: Arithmetic (`+`, `-`, unary `-`, scalar `*`) merging sorted chains and
//!   rejecting prefix conflicts at compile time.
//! - [`mod@cmp`]: Zero-padded equality ([`basis_eq`], [`static_basis_eq`]) and
//!   colexicographical ordering ([`colex_cmp`], [`static_colex_cmp`]).
//! - [`mod@index`]: Signed mode indexing ([`Get`], [`BASIS_NORMALIZED_INDEX`]).
//! - [`mod@repr`]: Basis decomposition and single-basis accessors ([`basis_repr`],
//!   [`is_basis`], [`SingleBasis`], [`proj`], [`unit()`]).
//! - [`mod@convert`]: Conversions between hierarchical tuples and basis forms
//!   ([`make_basis_like`], [`from_tuple`], [`as_tuple`]).
//! - **Macros**: [`e!`](crate::e!) and [`basis_tuple!`](crate::basis_tuple!).
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{basis::as_tuple, basis_tuple, e, int_tuple, integer::Int, walk_mode};
//!
//! let v = Int::<2> * e![0] + 3i32 * e![1, 1];
//! assert_eq!(v.get::<0>(), Int::<2>);
//! assert_eq!(walk_mode!(v, 1, 1), 3i32);
//! assert_eq!(as_tuple(v), int_tuple![Int::<2>, (Int::<0>, 3i32)]);
//! assert_eq!(basis_tuple![Int::<2>, (Int::<0>, 3i32)], v);
//! ```
//!
//! # Notes
//!
//! - Coordinate paths require non-negative mode indices; negative indices do not
//!   implement [`BasisPath`].
//! - Operations normalize results via [`CanonicalBasis`]: empty chains collapse to
//!   [`BasisCombination<Nil>`], single terms unwrap to [`BasisTerm`], and multi-term
//!   chains remain [`BasisCombination`].
//! - [`BasisCombination`]'s tuple-struct constructor is private so its invariants
//!   (strictly sorted unique paths, no prefix conflicts, no static zeros) are enforced
//!   by construction.
//! - Prefix-conflicting paths (such as `e![0]` and `e![0, 1]`) are rejected at compile
//!   time by arithmetic and colexicographical comparison.

use core::cmp::Ordering;
use core::fmt::{Display, Formatter};
use core::marker::PhantomData;

use crate::integer::{
    Int, Integer, cmp as scalar_cmp, eq as scalar_eq, int, static_cmp, static_cmp_of, static_eq_of,
    static_true,
};
use crate::tuple::{FlatTuple, Get, GetOutput, Item, LeafProfile, Nil, Til};
pub use crate::tuple::{Mode, ModeEnd};

pub mod cmp;
pub mod convert;
pub mod index;
pub mod ops;
pub mod path;
pub mod repr;

pub use cmp::*;
pub use convert::*;
pub use index::*;
pub use ops::*;
pub use path::*;
pub use repr::*;

/// Coefficient abstraction for basis terms, implemented for all [`Integer`] types
/// and extensible to algebraic rings such as `F2`.
pub const trait BasisCoeff: Copy {
    /// `true` iff this coefficient is statically guaranteed to be zero.
    const IS_STATIC_ZERO: bool = false;
    /// `true` iff this coefficient's value is known at compile time.
    const IS_STATIC: bool = false;
    /// Compile-time ordering against zero, or `None` if dynamic.
    const STATIC_CMP_ZERO: Option<Ordering> = None;
    /// Compile-time signed integer value when the coefficient is a static [`Integer`].
    const STATIC_VALUE: Option<int> = None;
    /// Multiplicative identity type in this coefficient's algebra.
    type Unit: [const] BasisCoeff;

    /// Returns `true` iff `self` equals zero.
    fn is_zero(&self) -> bool;

    /// Three-valued static zero check (`None` if dynamic).
    fn static_is_zero(&self) -> Option<bool> {
        if Self::IS_STATIC_ZERO {
            Some(true)
        } else if Self::IS_STATIC {
            Some(false)
        } else {
            None
        }
    }

    /// Compares `self` against zero.
    fn cmp_zero(&self) -> Ordering;
    /// Compile-time comparison of `self` against zero (`None` if dynamic).
    fn static_cmp_zero(&self) -> Option<Ordering>;
    /// Returns the multiplicative identity element of this coefficient's algebra.
    fn algebra_unit() -> Self::Unit;
}

const impl<I: [const] Integer> BasisCoeff for I {
    const IS_STATIC_ZERO: bool = static_true(static_eq_of::<I, Int<0>>());
    const IS_STATIC: bool = I::IS_STATIC;
    const STATIC_CMP_ZERO: Option<Ordering> = static_cmp_of::<I, Int<0>>();
    const STATIC_VALUE: Option<int> = I::STATIC_VALUE;
    type Unit = Int<1>;

    fn is_zero(&self) -> bool {
        scalar_eq(*self, Int::<0>)
    }

    fn cmp_zero(&self) -> Ordering {
        scalar_cmp(*self, Int::<0>)
    }

    fn static_cmp_zero(&self) -> Option<Ordering> {
        static_cmp(*self, Int::<0>)
    }

    fn algebra_unit() -> Self::Unit {
        Int
    }
}

/// A single `(value, P)` term stored inside a [`BasisCombination`] or standalone as [`ScaledBasis`].
#[derive(Clone, Copy, Default, Debug)]
#[repr(transparent)]
pub struct BasisTerm<V, P> {
    /// Coefficient attached to basis path `P`.
    pub value: V,
    path: PhantomData<P>,
}

const impl<VL, VR, P> PartialEq<BasisTerm<VR, P>> for BasisTerm<VL, P>
where
    VL: [const] PartialEq<VR>,
{
    fn eq(&self, rhs: &BasisTerm<VR, P>) -> bool {
        self.value == rhs.value
    }
}

const impl<V: [const] Eq, P> Eq for BasisTerm<V, P> {}

/// Type alias for a single scaled basis vector `V * E(P)`.
pub type ScaledBasis<V, P> = BasisTerm<V, P>;

const impl<V: BasisCoeff, P: BasisPath> Item for BasisTerm<V, P> {
    const RANK: int = 1;
    const DEPTH: int = 0;
    const LEAVES: int = 1;
    const IS_STATIC: bool = V::IS_STATIC;
    type Profile = LeafProfile;
}

/// Unit basis vector `Int<1> * E(P)`.
pub type Basis<P> = BasisTerm<Int<1>, P>;

impl<P: BasisPath> Basis<P> {
    /// Compile-time constant unit basis along path `P`.
    pub const UNIT: Self = BasisTerm::new(Int);

    /// Constructs the unit basis along path `P`.
    pub const fn unit() -> Self {
        Self::UNIT
    }
}

/// Constructs the unit basis along path `P`.
pub const fn e<P: BasisPath>() -> Basis<P> {
    Basis::<P>::UNIT
}

impl<V, P: BasisPath> BasisTerm<V, P> {
    /// Creates a basis term carrying `value` at path `P`.
    pub const fn new(value: V) -> Self {
        Self {
            value,
            path: PhantomData,
        }
    }

    /// Prepends outer mode index `M` onto this basis term's path.
    pub const fn at<const M: int>(self) -> BasisTerm<V, Mode<M, P>>
    where
        V: Copy,
        Mode<M, P>: BasisPath,
    {
        BasisTerm::new(self.value)
    }
}

impl<V, const M: int, Next> BasisTerm<V, Mode<M, Next>> {
    /// Copies the mode at index `I` from this single basis term (returning [`Int<0>`] on miss).
    pub const fn get<const I: int>(&self) -> GetOutput<Self, I>
    where
        Self: [const] Get<I>,
    {
        Get::<I>::get(self)
    }
}

/// Marker trait for flat tuples of [`BasisTerm<V, P>`].
pub const trait BasisTuple: [const] FlatTuple {}
const impl BasisTuple for Nil {}
const impl<V: [const] BasisCoeff, P: [const] BasisPath, T: [const] BasisTuple> BasisTuple
    for Til<BasisTerm<V, P>, T>
{
}

/// Sparse linear combination of path-sorted [`BasisTerm`]s.
///
/// Direct tuple-struct construction is private and [`Default`] is restricted to `Nil`:
///
/// ```compile_fail,E0423
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{basis::BasisCombination, e, tuple};
/// let _ = BasisCombination(tuple![e![1], e![0]]);
/// ```
///
/// ```compile_fail,E0599
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{basis::{Basis, BasisCombination}, mode, tuple::{Nil, Til}};
/// let _ = BasisCombination::<Til<Basis<mode![1]>, Til<Basis<mode![0]>, Nil>>>::default();
/// ```
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct BasisCombination<T: BasisTuple>(T);

const impl<L: BasisTuple, R: BasisTuple> PartialEq<BasisCombination<R>> for BasisCombination<L>
where
    L: [const] PartialEq<R>,
{
    fn eq(&self, rhs: &BasisCombination<R>) -> bool {
        self.0 == rhs.0
    }
}

const impl<T: BasisTuple + [const] Eq> Eq for BasisCombination<T> {}

const impl<T: BasisTuple> Item for BasisCombination<T> {
    const RANK: int = 1;
    const DEPTH: int = 0;
    const LEAVES: int = 1;
    const IS_STATIC: bool = T::IS_STATIC;
    type Profile = LeafProfile;
}

impl Default for BasisCombination<Nil> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<T: BasisTuple> BasisCombination<T> {
    /// Returns a copy of the underlying path-sorted [`BasisTuple`] chain.
    pub const fn terms(&self) -> T {
        self.0
    }

    /// Copies the mode at index `I` (zero-padded with [`Int<0>`] for positive indices past the largest path).
    ///
    /// Negative indices below `-(MAX_MODE + 1)` fail at compile time:
    ///
    /// ```compile_fail,E0080
    /// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops)]
    /// # #![allow(incomplete_features)]
    /// use noodle_cute::e;
    /// let _ = (e![0] + e![1]).get::<-3>();
    /// ```
    pub const fn get<const I: int>(&self) -> GetOutput<Self, I>
    where
        Self: [const] Get<I>,
    {
        Get::<I>::get(self)
    }
}

impl BasisCombination<Nil> {
    /// Returns the canonical zero basis combination `BasisCombination(Nil)`.
    pub const fn zero() -> Self {
        BasisCombination(Nil)
    }

    /// Builds a canonical basis element or combination from a hierarchical tuple.
    #[allow(clippy::new_ret_no_self)]
    pub const fn new<T: [const] CollectBasisTerms<0>>(t: T) -> <T::Terms as CanonicalBasis>::Output
    where
        T::Terms: [const] CanonicalBasis,
    {
        t.collect_terms().canonical_basis()
    }
}

// Formats trailing mode indices inside a multi-index `@(m0, m1, ...)` suffix.
trait BasisPathInnerDisplay {
    fn fmt_path_tail(f: &mut Formatter<'_>) -> core::fmt::Result;
}

impl BasisPathInnerDisplay for ModeEnd {
    fn fmt_path_tail(_: &mut Formatter<'_>) -> core::fmt::Result {
        Ok(())
    }
}

impl<const M: int, Next: BasisPathInnerDisplay> BasisPathInnerDisplay for Mode<M, Next> {
    fn fmt_path_tail(f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, ", {M}")?;
        Next::fmt_path_tail(f)
    }
}

// Formats the `@m` or `@(m0, m1, ...)` path suffix of a `BasisTerm` (`ModeEnd` prints nothing).
trait BasisPathDisplay {
    fn fmt_path_suffix(f: &mut Formatter<'_>) -> core::fmt::Result;
}

impl BasisPathDisplay for ModeEnd {
    fn fmt_path_suffix(_: &mut Formatter<'_>) -> core::fmt::Result {
        Ok(())
    }
}

impl<const M: int> BasisPathDisplay for Mode<M, ModeEnd> {
    fn fmt_path_suffix(f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "@{M}")
    }
}

impl<const M: int, const M2: int, Next: BasisPathInnerDisplay> BasisPathDisplay
    for Mode<M, Mode<M2, Next>>
{
    fn fmt_path_suffix(f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "@({M}, {M2}")?;
        Next::fmt_path_tail(f)?;
        f.write_str(")")
    }
}

impl<V: Display, P: BasisPathDisplay> Display for BasisTerm<V, P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        Display::fmt(&self.value, f)?;
        P::fmt_path_suffix(f)
    }
}

// Formats remaining terms of a `BasisCombination` joined by `" + "`.
trait BasisCombinationTailDisplay {
    fn fmt_terms_tail(&self, f: &mut Formatter<'_>) -> core::fmt::Result;
}

impl BasisCombinationTailDisplay for Nil {
    fn fmt_terms_tail(&self, _: &mut Formatter<'_>) -> core::fmt::Result {
        Ok(())
    }
}

impl<H: Display, T: BasisCombinationTailDisplay> BasisCombinationTailDisplay for Til<H, T> {
    fn fmt_terms_tail(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.write_str(" + ")?;
        Display::fmt(&self.0, f)?;
        self.1.fmt_terms_tail(f)
    }
}

impl Display for BasisCombination<Nil> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.write_str("0")
    }
}

impl<H: Display, T: BasisTuple + BasisCombinationTailDisplay> Display
    for BasisCombination<Til<H, T>>
where
    Til<H, T>: BasisTuple,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        Display::fmt(&self.0.0, f)?;
        self.0.1.fmt_terms_tail(f)
    }
}

pub(crate) const COEFF_IS_STATIC_ZERO<V: BasisCoeff>: bool = V::IS_STATIC_ZERO;
pub(crate) const ANY_STATIC_ZERO<A: BasisCoeff, B: BasisCoeff>: bool =
    A::IS_STATIC_ZERO || B::IS_STATIC_ZERO;

// Compile-time guard rejecting static-zero terms (`IS_ZERO == false`).
#[doc(hidden)]
pub const trait ValidBasisTerm<const IS_ZERO: bool> {}
const impl ValidBasisTerm<false> for () {}

// Compile-time guard requiring strict ascending path order (`ORDER == Less`) and non-static-zero coefficient.
#[doc(hidden)]
pub const trait ValidBasisOrder<const ORDER: PathOrdering, const IS_ZERO: bool> {}
const impl ValidBasisOrder<{ PathOrdering::Less }, false> for () {}

// Verifies that every term after `PrevPath` is strictly greater in path order and non-static-zero.
#[doc(hidden)]
pub const trait SortedBasisTail<PrevPath: BasisPath>: [const] BasisTuple {}

const impl<PrevPath: BasisPath> SortedBasisTail<PrevPath> for Nil {}

const impl<PrevPath, V, P, T> SortedBasisTail<PrevPath> for Til<BasisTerm<V, P>, T>
where
    PrevPath: [const] ComparePath<P> + [const] BasisPath,
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    T: [const] SortedBasisTail<P>,
    (): [const] ValidBasisOrder<{ PATH_ORDER::<PrevPath, P> }, { COEFF_IS_STATIC_ZERO::<V> }>,
{
}

/// Marker trait for canonical [`BasisTuple`] chains: strictly sorted unique paths,
/// no prefix conflicts, and no static zero coefficients.
pub const trait CanonicalBasisTuple: [const] BasisTuple {}

const impl CanonicalBasisTuple for Nil {}

const impl<V, P, T> CanonicalBasisTuple for Til<BasisTerm<V, P>, T>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    T: [const] SortedBasisTail<P>,
    (): [const] ValidBasisTerm<{ COEFF_IS_STATIC_ZERO::<V> }>,
{
}

// Prepends `BasisTerm<V, P>` onto `Self` when `OMIT = false`, or drops static zeros when `OMIT = true`.
#[doc(hidden)]
pub const trait EmitBasisTerm<V: [const] BasisCoeff, P: [const] BasisPath, const OMIT: bool>:
    [const] BasisTuple
{
    type Output: [const] BasisTuple;
    fn emit(self, value: V) -> Self::Output;
}

const impl<T: [const] BasisTuple, V: [const] BasisCoeff, P: [const] BasisPath>
    EmitBasisTerm<V, P, true> for T
{
    type Output = T;
    fn emit(self, _: V) -> Self::Output {
        self
    }
}

const impl<T: [const] BasisTuple, V: [const] BasisCoeff, P: [const] BasisPath>
    EmitBasisTerm<V, P, false> for T
{
    type Output = Til<BasisTerm<V, P>, T>;
    fn emit(self, value: V) -> Self::Output {
        Til(BasisTerm::new(value), self)
    }
}

/// Converts basis terms, combinations, or static zero [`Int<0>`] into a normalized [`BasisTuple`].
pub const trait IntoBasisCombination: Copy {
    /// Zero-stripped, path-sorted [`BasisTuple`] chain.
    type Terms: [const] BasisTuple;
    /// Extracts the normalized [`BasisTuple`] terms.
    fn into_basis_terms(self) -> Self::Terms;
}

const impl<V: [const] BasisCoeff, P: [const] BasisPath> IntoBasisCombination for BasisTerm<V, P>
where
    Nil: [const] EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>,
{
    type Terms = <Nil as EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>>::Output;
    fn into_basis_terms(self) -> Self::Terms {
        Nil.emit(self.value)
    }
}

const impl<T: [const] BasisTuple> IntoBasisCombination for BasisCombination<T> {
    type Terms = T;
    fn into_basis_terms(self) -> Self::Terms {
        self.0
    }
}

const impl IntoBasisCombination for Int<0> {
    type Terms = Nil;
    fn into_basis_terms(self) -> Self::Terms {
        Nil
    }
}

/// Normalizes a [`BasisTuple`] chain into its canonical user-facing representation:
/// - `Nil` -> `BasisCombination<Nil>`
/// - single term -> `BasisTerm<V, P>` ([`ScaledBasis<V, P>`])
/// - two or more terms -> `BasisCombination<Self>`
///
/// Unsorted, duplicate, prefix-conflicting, or static-zero terms are rejected at compile time:
///
/// ```compile_fail,E0599
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{basis::CanonicalBasis, e, tuple};
/// let _ = tuple![e![1], e![0]].canonical_basis();
/// ```
///
/// ```compile_fail,E0599
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{basis::CanonicalBasis, e, tuple};
/// let _ = tuple![e![0], e![0]].canonical_basis();
/// ```
pub const trait CanonicalBasis: [const] BasisTuple {
    /// Canonical basis type (`BasisCombination<Nil>`, `BasisTerm<V, P>`, or `BasisCombination<Self>`).
    type Output: Copy;
    /// Wraps or unwraps `self` into its canonical representation.
    fn canonical_basis(self) -> Self::Output;
}

const impl CanonicalBasis for Nil {
    type Output = BasisCombination<Nil>;
    fn canonical_basis(self) -> Self::Output {
        BasisCombination(Nil)
    }
}

const impl<V: [const] BasisCoeff, P: [const] BasisPath> CanonicalBasis for Til<BasisTerm<V, P>, Nil>
where
    Self: [const] CanonicalBasisTuple,
{
    type Output = BasisTerm<V, P>;
    fn canonical_basis(self) -> Self::Output {
        self.0
    }
}

const impl<V0, P0, V1, P1, T> CanonicalBasis for Til<BasisTerm<V0, P0>, Til<BasisTerm<V1, P1>, T>>
where
    V0: [const] BasisCoeff,
    P0: [const] BasisPath,
    V1: [const] BasisCoeff,
    P1: [const] BasisPath,
    T: [const] BasisTuple,
    Self: [const] CanonicalBasisTuple,
{
    type Output = BasisCombination<Self>;
    fn canonical_basis(self) -> Self::Output {
        BasisCombination(self)
    }
}

/// Constructs a unit basis vector along the given non-negative mode path.
///
/// - `e!()` produces [`Int<1>`] (the rank-zero unit basis).
/// - `e!(i, ...)` produces [`Basis<mode!(i, ...)>`].
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{basis::Basis, e, integer::Int, mode};
///
/// assert_eq!(e![], Int::<1>);
/// assert_eq!(e![0], Basis::<mode![0]>::UNIT);
/// assert_eq!(e![1, 0], e![0].at::<1>());
/// ```
///
/// Negative mode indices are rejected at compile time:
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::e;
/// let _ = e![-1];
/// ```
#[macro_export]
macro_rules! e {
    () => {
        $crate::integer::Int::<1>
    };
    ($($idx:expr),+ $(,)?) => {
        $crate::basis::e::<$crate::mode!($($idx),+)>()
    };
}

/// Constructs a canonical [`BasisTerm`] or [`BasisCombination`] from a hierarchical tuple literal.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{basis::BasisCombination, basis_tuple, e, integer::Int};
///
/// assert_eq!(basis_tuple![], BasisCombination::zero());
/// assert_eq!(basis_tuple![Int::<0>, Int::<1>], e![1]);
/// assert_eq!(basis_tuple![1i32, (2i32, 3i32)], 1i32 * e![0] + 2i32 * e![1, 0] + 3i32 * e![1, 1]);
/// ```
#[macro_export]
macro_rules! basis_tuple {
    ($($input:tt)*) => {
        $crate::basis::from_tuple($crate::tuple!($($input)*))
    };
}
