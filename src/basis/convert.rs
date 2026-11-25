//! Conversions between hierarchical tuples and sparse basis representations
//! (`make_basis_like`, `from_tuple`, `as_tuple`).
//!
//! Bridges dense hierarchical [`Tuple`] trees and sparse path-sorted [`BasisTerm`] /
//! [`BasisCombination`] values in both `const` and runtime contexts.
//!
//! # Main Interfaces
//!
//! - [`make_basis_like`] / [`MakeBasisLikeOutput`]: Builds a tuple of unit basis vectors
//!   congruent to a guide item's structural profile.
//! - [`make_top_basis_like`] / [`MakeTopBasisLikeOutput`]: Builds a flat tuple with one
//!   top-level unit basis vector (`e![i]`) per top-level mode of a guide item.
//! - [`from_tuple`] (also [`BasisCombination::new`] and [`basis_tuple!`](crate::basis_tuple!)):
//!   Converts a hierarchical tuple into a canonical sparse basis element.
//! - [`IntoBasisCoord`] / [`into_basis_coord`]: Normalizes a scalar, basis element, or
//!   hierarchical tuple coordinate into its sparse basis / scalar form.
//! - [`AsTuple`] / [`as_tuple`]: Materializes a sparse basis element (or nested tuple
//!   containing basis elements) back into dense nested tuples.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{basis::{as_tuple, from_tuple, make_basis_like}, e, int_tuple, integer::Int, tuple};
//!
//! assert_eq!(
//!     make_basis_like(int_tuple![10, (20, 30)]),
//!     tuple![e![0], (e![1, 0], e![1, 1])]
//! );
//! let b = from_tuple(int_tuple![Int::<0>, (2i32, 3i32)]);
//! assert_eq!(as_tuple(b), int_tuple![Int::<0>, (2i32, 3i32)]);
//! ```
//!
//! # Notes
//!
//! - [`from_tuple`] drops static `Int<0>` entries and canonicalizes to
//!   [`BasisCombination<Nil>`] (0 terms), [`BasisTerm`] (1 term), or
//!   [`BasisCombination`] (2+ terms).
//! - [`as_tuple`] zero-pads omitted lower modes with [`Int<0>`] and collapses empty
//!   combinations to [`Int<0>`], while leaving plain scalars and [`Nil`] unchanged.

use crate::integer::{INCREASE, Int, Integer, int};
use crate::tuple::{
    Append, FlatTuple, Item, LeafProfile, Mode, ModeEnd, Nil, NilProfile, Til, Tuple, TupleProfile,
};

use super::{
    AppendMode, Basis, BasisCoeff, BasisCombination, BasisPath, BasisTerm, BasisTuple,
    COEFF_IS_STATIC_ZERO, CanonicalBasis, EQ_MODE, EmitBasisTerm,
};

/// Maps one structural profile node at accumulated path prefix `P` to `Nil`, `Basis<P>`, or a nested tuple.
#[doc(hidden)]
pub const trait MakeBasisLikeHead<P: [const] BasisPath> {
    type Output: [const] Item;
    fn make_basis_head() -> Self::Output;
}

const impl<P: [const] BasisPath> MakeBasisLikeHead<P> for NilProfile {
    type Output = Nil;
    fn make_basis_head() -> Self::Output {
        Nil
    }
}

const impl<P: [const] BasisPath> MakeBasisLikeHead<P> for LeafProfile {
    type Output = Basis<P>;
    fn make_basis_head() -> Self::Output {
        Basis::<P>::UNIT
    }
}

const impl<P: [const] BasisPath, H, T> MakeBasisLikeHead<P> for TupleProfile<H, T>
where
    Self: [const] MakeBasisLikeTail<P, 0>,
{
    type Output = <Self as MakeBasisLikeTail<P, 0>>::Output;
    fn make_basis_head() -> Self::Output {
        Self::make_basis_tail()
    }
}

/// Walks sibling profile nodes under parent path `P`, assigning mode index `IDX` to each head.
#[doc(hidden)]
pub const trait MakeBasisLikeTail<P: [const] BasisPath, const IDX: int> {
    type Output: [const] Tuple;
    fn make_basis_tail() -> Self::Output;
}

const impl<P: [const] BasisPath, const IDX: int> MakeBasisLikeTail<P, IDX> for NilProfile {
    type Output = Nil;
    fn make_basis_tail() -> Self::Output {
        Nil
    }
}

const impl<P, const IDX: int, H, T> MakeBasisLikeTail<P, IDX> for TupleProfile<H, T>
where
    P: [const] AppendMode<IDX>,
    H: [const] MakeBasisLikeHead<P::Output>,
    T: [const] MakeBasisLikeTail<P, { INCREASE::<IDX> }>,
{
    type Output = Til<H::Output, T::Output>;
    fn make_basis_tail() -> Self::Output {
        Til(H::make_basis_head(), T::make_basis_tail())
    }
}

/// Top-level profile dispatcher for `make_basis_like` (`LeafProfile` -> `Int<1>`, `NilProfile` -> `Nil`).
#[doc(hidden)]
pub const trait MakeBasisLikeTopProfile {
    type Output: [const] Item;
    fn make_basis_top() -> Self::Output;
}

const impl MakeBasisLikeTopProfile for NilProfile {
    type Output = Nil;
    fn make_basis_top() -> Self::Output {
        Nil
    }
}

const impl MakeBasisLikeTopProfile for LeafProfile {
    type Output = Int<1>;
    fn make_basis_top() -> Self::Output {
        Int
    }
}

const impl<H, T> MakeBasisLikeTopProfile for TupleProfile<H, T>
where
    Self: [const] MakeBasisLikeTail<ModeEnd, 0>,
{
    type Output = <Self as MakeBasisLikeTail<ModeEnd, 0>>::Output;
    fn make_basis_top() -> Self::Output {
        Self::make_basis_tail()
    }
}

/// Output type produced by [`make_basis_like`] for item `T`.
pub type MakeBasisLikeOutput<T> = <<T as Item>::Profile as MakeBasisLikeTopProfile>::Output;

/// Builds a hierarchical tuple of unit basis vectors matching the structure of `profile`.
///
/// Integer leaves map to `Int<1>` (`e![]`), [`Nil`] maps to `Nil`, and nested tuples
/// assign `e![i, ...]` at each leaf path.
pub const fn make_basis_like<T: Item>(_: T) -> MakeBasisLikeOutput<T>
where
    T::Profile: [const] MakeBasisLikeTopProfile,
{
    <T::Profile as MakeBasisLikeTopProfile>::make_basis_top()
}

/// Walks top-level profile modes starting at index `IDX`, assigning one flat `Basis<Mode<IDX>>`
/// per mode without descending into nested heads.
#[doc(hidden)]
pub const trait MakeTopBasisTail<const IDX: int> {
    type Output: [const] Tuple;
    fn make_top_basis_tail() -> Self::Output;
}

const impl<const IDX: int> MakeTopBasisTail<IDX> for NilProfile {
    type Output = Nil;
    fn make_top_basis_tail() -> Self::Output {
        Nil
    }
}

const impl<const IDX: int, H, T> MakeTopBasisTail<IDX> for TupleProfile<H, T>
where
    Mode<IDX, ModeEnd>: [const] BasisPath,
    T: [const] MakeTopBasisTail<{ INCREASE::<IDX> }>,
{
    type Output = Til<Basis<Mode<IDX, ModeEnd>>, T::Output>;
    fn make_top_basis_tail() -> Self::Output {
        Til(Basis::<Mode<IDX, ModeEnd>>::UNIT, T::make_top_basis_tail())
    }
}

/// Top-level profile dispatcher for [`make_top_basis_like`]: `NilProfile` -> `Nil`,
/// `LeafProfile` -> `Int<1>`, and `TupleProfile` -> flat `(e![0], e![1], ...)`.
#[doc(hidden)]
pub const trait MakeTopBasisLikeProfile {
    type Output: [const] Item;
    fn make_top_basis() -> Self::Output;
}

const impl MakeTopBasisLikeProfile for NilProfile {
    type Output = Nil;
    fn make_top_basis() -> Self::Output {
        Nil
    }
}

const impl MakeTopBasisLikeProfile for LeafProfile {
    type Output = Int<1>;
    fn make_top_basis() -> Self::Output {
        Int
    }
}

const impl<H, T> MakeTopBasisLikeProfile for TupleProfile<H, T>
where
    Self: [const] MakeTopBasisTail<0>,
{
    type Output = <Self as MakeTopBasisTail<0>>::Output;
    fn make_top_basis() -> Self::Output {
        Self::make_top_basis_tail()
    }
}

/// Output type produced by [`make_top_basis_like`] for item `T`.
pub type MakeTopBasisLikeOutput<T> = <<T as Item>::Profile as MakeTopBasisLikeProfile>::Output;

/// Builds a flat tuple assigning one top-level unit basis vector `e![i]` to each
/// top-level mode of `guide` (without descending into nested sub-modes).
///
/// Paired with [`Layout::from_shape_with`](crate::layout::Layout::from_shape_with), this
/// distributes each top-level basis across nested sub-modes via exclusive prefix products,
/// matching PyCuTe's `Layout(shape, tuple(E(i) for i in range(rank(shape))))`.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{basis::make_top_basis_like, e, int_tuple, integer::Int, tuple};
///
/// assert_eq!(
///     make_top_basis_like(int_tuple![Int::<1>, Int::<4>, (Int::<2>, Int::<2>)]),
///     tuple![e![0], e![1], e![2]]
/// );
/// ```
pub const fn make_top_basis_like<T: Item>(_: T) -> MakeTopBasisLikeOutput<T>
where
    T::Profile: [const] MakeTopBasisLikeProfile,
{
    <T::Profile as MakeTopBasisLikeProfile>::make_top_basis()
}

/// Prepends outer mode index `M` onto every term's path in a `BasisTuple`.
#[doc(hidden)]
pub const trait PrependModeToTerms<const M: int>: [const] BasisTuple {
    type Output: [const] BasisTuple;
    fn prepend_mode(self) -> Self::Output;
}

const impl<const M: int> PrependModeToTerms<M> for Nil {
    type Output = Nil;
    fn prepend_mode(self) -> Self::Output {
        Nil
    }
}

const impl<const M: int, V, P, T> PrependModeToTerms<M> for Til<BasisTerm<V, P>, T>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    Mode<M, P>: [const] BasisPath,
    T: [const] PrependModeToTerms<M>,
{
    type Output = Til<BasisTerm<V, Mode<M, P>>, T::Output>;
    fn prepend_mode(self) -> Self::Output {
        Til(self.0.at::<M>(), self.1.prepend_mode())
    }
}

/// Extracts sparse `BasisTuple` terms from one node (`Nil`, scalar, `BasisTerm`, `BasisCombination`, or nested tuple).
#[doc(hidden)]
pub const trait CollectBasisNode {
    type Terms: [const] BasisTuple;
    fn collect_node(self) -> Self::Terms;
}

const impl CollectBasisNode for Nil {
    type Terms = Nil;
    fn collect_node(self) -> Self::Terms {
        Nil
    }
}

const impl<V: [const] BasisCoeff> CollectBasisNode for V
where
    Nil: [const] EmitBasisTerm<V, ModeEnd, { COEFF_IS_STATIC_ZERO::<V> }>,
{
    type Terms = <Nil as EmitBasisTerm<V, ModeEnd, { COEFF_IS_STATIC_ZERO::<V> }>>::Output;
    fn collect_node(self) -> Self::Terms {
        Nil.emit(self)
    }
}

const impl<V: [const] BasisCoeff, P: [const] BasisPath> CollectBasisNode for BasisTerm<V, P>
where
    Nil: [const] EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>,
{
    type Terms = <Nil as EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>>::Output;
    fn collect_node(self) -> Self::Terms {
        Nil.emit(self.value)
    }
}

const impl<T: [const] BasisTuple> CollectBasisNode for BasisCombination<T> {
    type Terms = T;
    fn collect_node(self) -> Self::Terms {
        self.0
    }
}

const impl<H, T> CollectBasisNode for Til<H, T>
where
    Self: [const] CollectBasisTerms<0>,
{
    type Terms = <Self as CollectBasisTerms<0>>::Terms;
    fn collect_node(self) -> Self::Terms {
        self.collect_terms()
    }
}

/// Collects sparse, path-sorted `BasisTerm`s from a hierarchical tuple starting at mode index `M`.
#[doc(hidden)]
pub const trait CollectBasisTerms<const M: int>: [const] Tuple {
    type Terms: [const] BasisTuple;
    fn collect_terms(self) -> Self::Terms;
}

const impl<const M: int> CollectBasisTerms<M> for Nil {
    type Terms = Nil;
    fn collect_terms(self) -> Self::Terms {
        Nil
    }
}

const impl<const M: int, H, T> CollectBasisTerms<M> for Til<H, T>
where
    H: [const] Item + [const] CollectBasisNode,
    H::Terms: [const] PrependModeToTerms<M>,
    T: [const] Tuple + [const] CollectBasisTerms<{ INCREASE::<M> }>,
    <H::Terms as PrependModeToTerms<M>>::Output: [const] Append<T::Terms>,
    <<H::Terms as PrependModeToTerms<M>>::Output as Append<T::Terms>>::Output: [const] BasisTuple,
{
    type Terms = <<H::Terms as PrependModeToTerms<M>>::Output as Append<T::Terms>>::Output;
    fn collect_terms(self) -> Self::Terms {
        self.0
            .collect_node()
            .prepend_mode()
            .append(self.1.collect_terms())
    }
}

/// Converts a hierarchical tuple into a canonical [`BasisTerm`] or [`BasisCombination`].
pub const fn from_tuple<T: [const] CollectBasisTerms<0>>(
    t: T,
) -> <T::Terms as CanonicalBasis>::Output
where
    T::Terms: [const] CanonicalBasis,
{
    BasisCombination::<Nil>::new(t)
}

/// Normalizes a scalar, basis element, or hierarchical tuple coordinate into its
/// sparse basis / scalar representation (mirroring PyCuTe's `ArithTuple(coord)`).
pub const trait IntoBasisCoord: Copy {
    /// Normalized sparse basis or scalar coordinate type.
    type Output: Copy;

    /// Converts `self` into its sparse basis / scalar form.
    fn into_basis_coord(self) -> Self::Output;
}

const impl IntoBasisCoord for Nil {
    type Output = Nil;

    fn into_basis_coord(self) -> Nil {
        Nil
    }
}

const impl<I: [const] Integer> IntoBasisCoord for I {
    type Output = I;

    fn into_basis_coord(self) -> I {
        self
    }
}

const impl<V: [const] BasisCoeff, P: [const] BasisPath> IntoBasisCoord for BasisTerm<V, P> {
    type Output = Self;

    fn into_basis_coord(self) -> Self {
        self
    }
}

const impl<T: [const] BasisTuple> IntoBasisCoord for BasisCombination<T> {
    type Output = Self;

    fn into_basis_coord(self) -> Self {
        self
    }
}

const impl<H, T> IntoBasisCoord for Til<H, T>
where
    Self: [const] CollectBasisTerms<0>,
    <Self as CollectBasisTerms<0>>::Terms: [const] CanonicalBasis,
{
    type Output = <<Self as CollectBasisTerms<0>>::Terms as CanonicalBasis>::Output;

    fn into_basis_coord(self) -> Self::Output {
        from_tuple(self)
    }
}

/// Normalizes a scalar, basis element, or hierarchical tuple coordinate into its
/// sparse basis / scalar form.
pub const fn into_basis_coord<C: [const] IntoBasisCoord>(coord: C) -> C::Output {
    coord.into_basis_coord()
}

/// Splits the contiguous prefix of terms with leading mode `MODE` (stripping that outer mode),
/// returning `(Head, Tail)` for dense tuple materialization.
#[doc(hidden)]
pub const trait SplitModePrefix<const MODE: int>: [const] FlatTuple {
    type Head: [const] FlatTuple;
    type Tail: [const] FlatTuple;
    fn split_mode_prefix(self) -> (Self::Head, Self::Tail);
}

/// Single-step dispatch for `SplitModePrefix`: `MATCH = true` moves the head term into `Head`.
#[doc(hidden)]
pub const trait SplitModePrefixStep<const MODE: int, const MATCH: bool>:
    [const] FlatTuple
{
    type Head: [const] FlatTuple;
    type Tail: [const] FlatTuple;
    fn split_mode_prefix_step(self) -> (Self::Head, Self::Tail);
}

const impl<const MODE: int> SplitModePrefix<MODE> for Nil {
    type Head = Nil;
    type Tail = Nil;
    fn split_mode_prefix(self) -> (Nil, Nil) {
        (Nil, Nil)
    }
}

const impl<const MODE: int, V, const M: int, Next, T> SplitModePrefix<MODE>
    for Til<BasisTerm<V, Mode<M, Next>>, T>
where
    V: [const] BasisCoeff,
    Next: [const] BasisPath,
    Mode<M, Next>: [const] BasisPath,
    T: [const] FlatTuple,
    Self: [const] SplitModePrefixStep<MODE, { EQ_MODE::<M, MODE> }>,
{
    type Head = <Self as SplitModePrefixStep<MODE, { EQ_MODE::<M, MODE> }>>::Head;
    type Tail = <Self as SplitModePrefixStep<MODE, { EQ_MODE::<M, MODE> }>>::Tail;
    fn split_mode_prefix(self) -> (Self::Head, Self::Tail) {
        self.split_mode_prefix_step()
    }
}

const impl<const MODE: int, V, const M: int, Next, T> SplitModePrefixStep<MODE, false>
    for Til<BasisTerm<V, Mode<M, Next>>, T>
where
    V: [const] BasisCoeff,
    Next: [const] BasisPath,
    Mode<M, Next>: [const] BasisPath,
    T: [const] FlatTuple,
{
    type Head = Nil;
    type Tail = Self;
    fn split_mode_prefix_step(self) -> (Nil, Self) {
        (Nil, self)
    }
}

const impl<const MODE: int, V, const M: int, Next, T> SplitModePrefixStep<MODE, true>
    for Til<BasisTerm<V, Mode<M, Next>>, T>
where
    V: [const] BasisCoeff,
    Next: [const] BasisPath,
    Mode<M, Next>: [const] BasisPath,
    T: [const] FlatTuple + [const] SplitModePrefix<MODE>,
{
    type Head = Til<BasisTerm<V, Next>, <T as SplitModePrefix<MODE>>::Head>;
    type Tail = <T as SplitModePrefix<MODE>>::Tail;
    fn split_mode_prefix_step(self) -> (Self::Head, Self::Tail) {
        let (head, tail) = self.1.split_mode_prefix();
        (Til(BasisTerm::new(self.0.value), head), tail)
    }
}

/// Reconstructs a dense tuple chain from a sorted `BasisTuple` starting at leading mode `MODE`,
/// padding absent lower modes with `Int<0>`.
#[doc(hidden)]
pub const trait CombModeChain<const MODE: int>: [const] FlatTuple {
    type Output: [const] Tuple;
    fn comb_mode_chain(self) -> Self::Output;
}

/// Single-step dispatch for `CombModeChain`: `MATCH = false` emits `Int<0>` for gap `MODE`,
/// `MATCH = true` splits and recursively materializes all terms under `MODE`.
#[doc(hidden)]
pub const trait CombModeChainStep<const MODE: int, const MATCH: bool>:
    [const] FlatTuple
{
    type Output: [const] Tuple;
    fn comb_mode_chain_step(self) -> Self::Output;
}

/// Materializes a path-sorted `BasisTuple` into `Int<0>` (empty), scalar `V` (`ModeEnd`), or a dense tuple.
#[doc(hidden)]
pub const trait CombAsTuple: [const] FlatTuple {
    type Output: [const] Item;
    fn comb_as_tuple(self) -> Self::Output;
}

const impl<const MODE: int> CombModeChain<MODE> for Nil {
    type Output = Nil;
    fn comb_mode_chain(self) -> Nil {
        Nil
    }
}

const impl<const MODE: int, V, const M: int, Next, T> CombModeChain<MODE>
    for Til<BasisTerm<V, Mode<M, Next>>, T>
where
    V: [const] BasisCoeff,
    Next: [const] BasisPath,
    Mode<M, Next>: [const] BasisPath,
    T: [const] FlatTuple,
    Self: [const] CombModeChainStep<MODE, { EQ_MODE::<M, MODE> }>,
{
    type Output = <Self as CombModeChainStep<MODE, { EQ_MODE::<M, MODE> }>>::Output;
    fn comb_mode_chain(self) -> Self::Output {
        self.comb_mode_chain_step()
    }
}

const impl<const MODE: int, V, const M: int, Next, T> CombModeChainStep<MODE, false>
    for Til<BasisTerm<V, Mode<M, Next>>, T>
where
    V: [const] BasisCoeff,
    Next: [const] BasisPath,
    Mode<M, Next>: [const] BasisPath,
    T: [const] FlatTuple,
    Self: [const] CombModeChain<{ INCREASE::<MODE> }>,
{
    type Output = Til<Int<0>, <Self as CombModeChain<{ INCREASE::<MODE> }>>::Output>;
    fn comb_mode_chain_step(self) -> Self::Output {
        Til(Int, self.comb_mode_chain())
    }
}

const impl<const MODE: int, V, const M: int, Next, T> CombModeChainStep<MODE, true>
    for Til<BasisTerm<V, Mode<M, Next>>, T>
where
    V: [const] BasisCoeff,
    Next: [const] BasisPath,
    Mode<M, Next>: [const] BasisPath,
    T: [const] FlatTuple,
    Self: [const] SplitModePrefix<MODE>,
    <Self as SplitModePrefix<MODE>>::Head: [const] CombAsTuple,
    <Self as SplitModePrefix<MODE>>::Tail: [const] CombModeChain<{ INCREASE::<MODE> }>,
{
    type Output = Til<
        <<Self as SplitModePrefix<MODE>>::Head as CombAsTuple>::Output,
        <<Self as SplitModePrefix<MODE>>::Tail as CombModeChain<{ INCREASE::<MODE> }>>::Output,
    >;
    fn comb_mode_chain_step(self) -> Self::Output {
        let (head, tail) = self.split_mode_prefix();
        Til(head.comb_as_tuple(), tail.comb_mode_chain())
    }
}

const impl CombAsTuple for Nil {
    type Output = Int<0>;
    fn comb_as_tuple(self) -> Int<0> {
        Int
    }
}

const impl<V: [const] BasisCoeff + [const] Item> CombAsTuple for Til<BasisTerm<V, ModeEnd>, Nil> {
    type Output = V;
    fn comb_as_tuple(self) -> V {
        self.0.value
    }
}

const impl<V, const M: int, Next, T> CombAsTuple for Til<BasisTerm<V, Mode<M, Next>>, T>
where
    V: [const] BasisCoeff,
    Next: [const] BasisPath,
    Mode<M, Next>: [const] BasisPath,
    T: [const] FlatTuple,
    Self: [const] CombModeChain<0>,
{
    type Output = <Self as CombModeChain<0>>::Output;
    fn comb_as_tuple(self) -> Self::Output {
        self.comb_mode_chain()
    }
}

/// Materializes a [`BasisTerm`], [`BasisCombination`], scalar, or nested tuple into plain nested tuples.
///
/// Omitted lower modes in sparse basis terms are materialized as [`Int<0>`], and empty
/// combinations collapse to [`Int<0>`].
pub const trait AsTuple: [const] Item {
    /// Materialized plain nested tuple or scalar type.
    type Output: [const] Item;
    /// Converts `self` into plain nested tuples.
    fn as_tuple(self) -> Self::Output;
}

const impl AsTuple for Nil {
    type Output = Nil;
    fn as_tuple(self) -> Nil {
        Nil
    }
}

const impl<I: [const] Integer> AsTuple for I {
    type Output = I;
    fn as_tuple(self) -> I {
        self
    }
}

const impl<V: [const] BasisCoeff + [const] Item, P: [const] BasisPath> AsTuple for BasisTerm<V, P>
where
    Nil: [const] EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>,
    <Nil as EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>>::Output: [const] CombAsTuple,
{
    type Output = <<Nil as EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>>::Output as CombAsTuple>::Output;
    fn as_tuple(self) -> Self::Output {
        Nil.emit(self.value).comb_as_tuple()
    }
}

const impl<T: [const] BasisTuple + [const] CombAsTuple> AsTuple for BasisCombination<T> {
    type Output = <T as CombAsTuple>::Output;
    fn as_tuple(self) -> Self::Output {
        self.0.comb_as_tuple()
    }
}

const impl<H, T> AsTuple for Til<H, T>
where
    H: [const] AsTuple,
    T: [const] Tuple + [const] AsTuple,
    T::Output: [const] Tuple,
{
    type Output = Til<H::Output, T::Output>;
    fn as_tuple(self) -> Self::Output {
        Til(self.0.as_tuple(), self.1.as_tuple())
    }
}

/// Converts `x` (a [`BasisCombination`], [`BasisTerm`], scalar, or nested tuple) into plain nested tuples.
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{basis::as_tuple, basis_tuple, e, int_tuple, integer::Int};
///
/// assert_eq!(as_tuple(42i32), 42i32);
/// assert_eq!(as_tuple(e![1]), int_tuple![Int::<0>, Int::<1>]);
/// assert_eq!(
///     as_tuple(basis_tuple![1i32, 2i32, (3i32, 4i32)]),
///     int_tuple![1i32, 2i32, (3i32, 4i32)]
/// );
/// ```
pub const fn as_tuple<T: [const] AsTuple>(x: T) -> T::Output {
    x.as_tuple()
}
