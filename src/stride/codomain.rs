//! Codomain structural profile (`CodomainProfile`) and codomain shape (`Coshape`, `coshape`).
//!
//! # Main Interfaces
//!
//! - [`CodomainProfile`]: Reflects the hierarchical [`Item::Profile`] of a stride's codomain
//!   (scalar offsets yield [`LeafProfile`](crate::tuple::LeafProfile); basis strides merge
//!   their active paths into a nested [`TupleProfile`](crate::tuple::TupleProfile)).
//! - [`Coshape`] / [`coshape`]: Computes the bounding codomain [`Shape`] of a congruent
//!   `(shape, stride)` pair by evaluating the maximum offset and incrementing each leaf by 1.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::stride::coshape;
//! use noodle_cute::{e, int_tuple, tuple};
//!
//! let shape = int_tuple![Int::<4>, Int::<8>];
//! assert_eq!(coshape(shape, int_tuple![Int::<1>, Int::<4>]), Int::<32>);
//! assert_eq!(coshape(shape, tuple![e![0], e![2]]), int_tuple![Int::<4>, Int::<1>, Int::<8>]);
//! ```
//!
//! # Notes
//!
//! - [`coshape`] requires positive shape extents (`> 0`) and non-negative maximum offsets.
//! - Sparse gaps in active basis paths are padded with `Int<1>` in the resulting coshape.

use crate::basis::{
    BasisCoeff, BasisCombination, BasisPath, BasisTerm, BasisTuple, COEFF_IS_STATIC_ZERO,
    CombAsTuple, EQ_MODE, EmitBasisTerm, IntoBasisCombination, MergeAdd, Mode, ModeEnd,
    SplitModePrefix,
};
use crate::integer::{INCREASE, IS_NON_NEG, Int, Integer, add, int, sub};
use crate::shape::{Shape, ValidShapeExtent};
use crate::tuple::{Item, Leaf, Nil, Til, Tuple};

use super::{InnerProduct, Stride};

/// Tracks the hierarchical profile of a stride's codomain (output coordinate structure).
///
/// Derived from the stride's active basis paths, matching [`coshape`] and
/// [`Layout::coprofile`](crate::layout::Layout::coprofile):
/// - Integer strides (including multi-mode tuples of integer strides and [`Nil`]) map to 1D
///   offsets ([`LeafProfile`](crate::tuple::LeafProfile)).
/// - Coordinate basis strides ([`ScaledBasis`](crate::basis::ScaledBasis), [`BasisCombination`],
///   and tuples containing them) merge their active basis paths into the multi-dimensional codomain
///   [`TupleProfile`](crate::tuple::TupleProfile).
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use core::any::TypeId;
/// use noodle_cute::basis::{Basis, Mode, ScaledBasis};
/// use noodle_cute::integer::Int;
/// use noodle_cute::stride::CodomainProfile;
/// use noodle_cute::tuple::{LeafProfile, Nil, NilProfile, TupleContainer, TupleProfile};
///
/// fn profile_of<T: CodomainProfile>() -> TypeId {
///     TypeId::of::<T::Profile>()
/// }
///
/// assert_eq!(profile_of::<Nil>(), TypeId::of::<LeafProfile>());
/// assert_eq!(profile_of::<Int<1>>(), TypeId::of::<LeafProfile>());
/// assert_eq!(
///     profile_of::<ScaledBasis<Int<1>, Mode<0>>>(),
///     TypeId::of::<TupleProfile<LeafProfile, NilProfile>>()
/// );
///
/// // Multi-mode integer strides produce a 1D scalar offset codomain
/// type Scalar2D = TupleContainer<Int<1>, TupleContainer<Int<2>, Nil>>;
/// assert_eq!(profile_of::<Scalar2D>(), TypeId::of::<LeafProfile>());
///
/// // Multi-mode basis strides merge their active axes into the codomain profile
/// type Basis2D = TupleContainer<Basis<Mode<0>>, TupleContainer<Basis<Mode<1>>, Nil>>;
/// assert_eq!(
///     profile_of::<Basis2D>(),
///     TypeId::of::<TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>>()
/// );
/// ```
pub const trait CodomainProfile {
    /// The structural profile of values produced in the stride's codomain.
    type Profile: 'static;
}

const impl<D> CodomainProfile for D
where
    D: StrideSkeleton,
    <D as StrideSkeleton>::Terms: CombAsTuple,
    <<<D as StrideSkeleton>::Terms as CombAsTuple>::Output as Item>::Profile: 'static,
{
    type Profile = <<<D as StrideSkeleton>::Terms as CombAsTuple>::Output as Item>::Profile;
}

const SUB_ONE_EXTENT<const N: int>: int = {
    assert!(N > 0, "coshape requires positive shape extents");
    sub(N, 1)
};

const ADD_ONE_OFFSET<const N: int>: int = {
    assert!(N >= 0, "coshape requires non-negative maximum offset");
    add(N, 1)
};

/// Maps a non-negative offset leaf scalar to its coshape extent (`offset + 1`).
#[doc(hidden)]
pub const trait OffsetExtent: [const] Integer {
    type Output: [const] Shape;
    fn offset_extent(self) -> Self::Output;
}

const impl OffsetExtent for int {
    type Output = int;
    fn offset_extent(self) -> int {
        assert!(self >= 0, "coshape requires non-negative maximum offset");
        add(self, 1)
    }
}

const impl<const N: int> OffsetExtent for Int<N>
where
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ ADD_ONE_OFFSET::<N> }> }>,
{
    type Output = Int<{ ADD_ONE_OFFSET::<N> }>;
    fn offset_extent(self) -> Self::Output {
        Int
    }
}

/// Decrements every leaf extent of a shape by `1` to form its maximum coordinate.
#[doc(hidden)]
pub const trait MaxCoord: [const] Shape {
    type Output: Item;
    fn max_coord(self) -> Self::Output;
}

const impl MaxCoord for Nil {
    type Output = Nil;
    fn max_coord(self) -> Nil {
        Nil
    }
}

const impl MaxCoord for int {
    type Output = int;
    fn max_coord(self) -> int {
        assert!(self > 0, "coshape requires positive shape extents");
        sub(self, 1)
    }
}

const impl<const N: int> MaxCoord for Int<N>
where
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<N> }>,
{
    type Output = Int<{ SUB_ONE_EXTENT::<N> }>;
    fn max_coord(self) -> Self::Output {
        Int
    }
}

const impl<H: [const] MaxCoord, T: [const] Tuple + [const] MaxCoord> MaxCoord for Til<H, T>
where
    <T as MaxCoord>::Output: [const] Tuple,
{
    type Output = Til<<H as MaxCoord>::Output, <T as MaxCoord>::Output>;
    fn max_coord(self) -> Self::Output {
        Til(self.0.max_coord(), self.1.max_coord())
    }
}

/// Replaces non-static-zero basis term coefficients with `Int<1>` while keeping all active paths.
#[doc(hidden)]
pub const trait MapUnitTerms: [const] BasisTuple {
    type Output: [const] BasisTuple;
    fn map_unit_terms(self) -> Self::Output;
}

const impl MapUnitTerms for Nil {
    type Output = Nil;
    fn map_unit_terms(self) -> Nil {
        Nil
    }
}

const impl<V, P, Rest> MapUnitTerms for Til<BasisTerm<V, P>, Rest>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    Rest: [const] MapUnitTerms,
    BasisTerm<V, P>: Leaf,
    BasisTerm<Int<1>, P>: Leaf,
    <Rest as MapUnitTerms>::Output: [const] EmitBasisTerm<Int<1>, P, { COEFF_IS_STATIC_ZERO::<V> }>,
{
    type Output = <<Rest as MapUnitTerms>::Output as EmitBasisTerm<
        Int<1>,
        P,
        { COEFF_IS_STATIC_ZERO::<V> },
    >>::Output;

    fn map_unit_terms(self) -> Self::Output {
        self.1.map_unit_terms().emit(Int::<1>)
    }
}

/// Collects the path-sorted skeleton of active basis paths present in a stride.
#[doc(hidden)]
pub const trait StrideSkeleton: Item {
    type Terms: [const] BasisTuple;
    fn skeleton_terms(self) -> Self::Terms;
}

const impl StrideSkeleton for Nil {
    type Terms = Nil;
    fn skeleton_terms(self) -> Nil {
        Nil
    }
}

const impl<I: Integer> StrideSkeleton for I {
    type Terms = Nil;
    fn skeleton_terms(self) -> Nil {
        Nil
    }
}

const impl<V, P> StrideSkeleton for BasisTerm<V, P>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    Self: Leaf,
    BasisTerm<Int<1>, P>: Leaf,
    Nil: [const] EmitBasisTerm<Int<1>, P, { COEFF_IS_STATIC_ZERO::<V> }>,
{
    type Terms = <Nil as EmitBasisTerm<Int<1>, P, { COEFF_IS_STATIC_ZERO::<V> }>>::Output;
    fn skeleton_terms(self) -> Self::Terms {
        Nil.emit(Int::<1>)
    }
}

const impl<T: [const] MapUnitTerms> StrideSkeleton for BasisCombination<T>
where
    Self: Leaf,
{
    type Terms = T::Output;
    fn skeleton_terms(self) -> Self::Terms {
        self.terms().map_unit_terms()
    }
}

const impl<H: [const] StrideSkeleton, T: Tuple + [const] StrideSkeleton> StrideSkeleton
    for Til<H, T>
where
    H::Terms: [const] MergeAdd<T::Terms>,
{
    type Terms = <H::Terms as MergeAdd<T::Terms>>::Output;
    fn skeleton_terms(self) -> Self::Terms {
        self.0.skeleton_terms().merge_add(self.1.skeleton_terms())
    }
}

/// Conditionally splits the leading `MODE` prefix of a sorted `BasisTuple` when `MATCH == true`.
#[doc(hidden)]
pub const trait CondSplitModePrefix<const MODE: int, const MATCH: bool>:
    [const] BasisTuple
{
    type Head: [const] BasisTuple;
    type Tail: [const] BasisTuple;
    fn cond_split_mode_prefix(self) -> (Self::Head, Self::Tail);
}

const impl<const MODE: int, T: [const] BasisTuple> CondSplitModePrefix<MODE, false> for T {
    type Head = Nil;
    type Tail = Self;
    fn cond_split_mode_prefix(self) -> (Nil, Self) {
        (Nil, self)
    }
}

const impl<const MODE: int, T: [const] SplitModePrefix<MODE> + [const] BasisTuple>
    CondSplitModePrefix<MODE, true> for T
where
    <T as SplitModePrefix<MODE>>::Head: [const] BasisTuple,
    <T as SplitModePrefix<MODE>>::Tail: [const] BasisTuple,
{
    type Head = <T as SplitModePrefix<MODE>>::Head;
    type Tail = <T as SplitModePrefix<MODE>>::Tail;
    fn cond_split_mode_prefix(self) -> (Self::Head, Self::Tail) {
        self.split_mode_prefix()
    }
}

/// Recursively constructs the coshape from a maximum-offset chain (`Self`) guided by active path skeleton `Skel`.
#[doc(hidden)]
pub const trait CombCoshapeValue<Skel: [const] BasisTuple>: [const] BasisTuple {
    type Output: [const] Shape;
    fn comb_coshape_value(self, skel: Skel) -> Self::Output;
}

/// Walks mode indices starting at `MODE` across offset (`Self`) and skeleton (`Skel`) chains,
/// padding absent modes with `Int<1>`.
#[doc(hidden)]
pub const trait CombCoshapeChain<const MODE: int, Skel: [const] BasisTuple>:
    [const] BasisTuple
{
    type Output: [const] Tuple + [const] Shape;
    fn comb_coshape_chain(self, skel: Skel) -> Self::Output;
}

/// Single-step dispatch for `CombCoshapeChain` based on whether the offset (`OFF_MATCH`)
/// or skeleton (`SKEL_MATCH`) head matches `MODE`.
#[doc(hidden)]
pub const trait CombCoshapeChainStep<
    const MODE: int,
    Skel: [const] BasisTuple,
    const OFF_MATCH: bool,
    const SKEL_MATCH: bool,
>: [const] BasisTuple
{
    type Output: [const] Tuple + [const] Shape;
    fn comb_coshape_chain_step(self, skel: Skel) -> Self::Output;
}

const impl CombCoshapeValue<Nil> for Nil {
    type Output = Int<1>;
    fn comb_coshape_value(self, _: Nil) -> Int<1> {
        Int
    }
}

const impl<SV: [const] BasisCoeff> CombCoshapeValue<Til<BasisTerm<SV, ModeEnd>, Nil>> for Nil
where
    BasisTerm<SV, ModeEnd>: Leaf,
{
    type Output = Int<1>;
    fn comb_coshape_value(self, _: Til<BasisTerm<SV, ModeEnd>, Nil>) -> Int<1> {
        Int
    }
}

const impl<SV, const M: int, SP, SRest> CombCoshapeValue<Til<BasisTerm<SV, Mode<M, SP>>, SRest>>
    for Nil
where
    SV: [const] BasisCoeff,
    SP: [const] BasisPath,
    Mode<M, SP>: [const] BasisPath,
    SRest: [const] BasisTuple,
    BasisTerm<SV, Mode<M, SP>>: Leaf,
    Self: [const] CombCoshapeChain<0, Til<BasisTerm<SV, Mode<M, SP>>, SRest>>,
{
    type Output = <Self as CombCoshapeChain<0, Til<BasisTerm<SV, Mode<M, SP>>, SRest>>>::Output;
    fn comb_coshape_value(self, skel: Til<BasisTerm<SV, Mode<M, SP>>, SRest>) -> Self::Output {
        self.comb_coshape_chain(skel)
    }
}

const impl<V: [const] OffsetExtent + [const] BasisCoeff> CombCoshapeValue<Nil>
    for Til<BasisTerm<V, ModeEnd>, Nil>
where
    BasisTerm<V, ModeEnd>: Leaf,
{
    type Output = <V as OffsetExtent>::Output;
    fn comb_coshape_value(self, _: Nil) -> Self::Output {
        self.0.value.offset_extent()
    }
}

const impl<V: [const] OffsetExtent + [const] BasisCoeff, SV: [const] BasisCoeff>
    CombCoshapeValue<Til<BasisTerm<SV, ModeEnd>, Nil>> for Til<BasisTerm<V, ModeEnd>, Nil>
where
    BasisTerm<V, ModeEnd>: Leaf,
    BasisTerm<SV, ModeEnd>: Leaf,
{
    type Output = <V as OffsetExtent>::Output;
    fn comb_coshape_value(self, _: Til<BasisTerm<SV, ModeEnd>, Nil>) -> Self::Output {
        self.0.value.offset_extent()
    }
}

const impl<V, const M: int, P, Rest, Skel> CombCoshapeValue<Skel>
    for Til<BasisTerm<V, Mode<M, P>>, Rest>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    Mode<M, P>: [const] BasisPath,
    Rest: [const] BasisTuple,
    Skel: [const] BasisTuple,
    BasisTerm<V, Mode<M, P>>: Leaf,
    Self: [const] CombCoshapeChain<0, Skel>,
{
    type Output = <Self as CombCoshapeChain<0, Skel>>::Output;
    fn comb_coshape_value(self, skel: Skel) -> Self::Output {
        self.comb_coshape_chain(skel)
    }
}

const impl<const MODE: int> CombCoshapeChain<MODE, Nil> for Nil {
    type Output = Nil;
    fn comb_coshape_chain(self, _: Nil) -> Nil {
        Nil
    }
}

const impl<const MODE: int, V, const M: int, P, Rest> CombCoshapeChain<MODE, Nil>
    for Til<BasisTerm<V, Mode<M, P>>, Rest>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    Mode<M, P>: [const] BasisPath,
    Rest: [const] BasisTuple,
    BasisTerm<V, Mode<M, P>>: Leaf,
    Self: [const] CombCoshapeChainStep<MODE, Nil, { EQ_MODE::<M, MODE> }, false>,
{
    type Output = <Self as CombCoshapeChainStep<MODE, Nil, { EQ_MODE::<M, MODE> }, false>>::Output;
    fn comb_coshape_chain(self, skel: Nil) -> Self::Output {
        self.comb_coshape_chain_step(skel)
    }
}

const impl<const MODE: int, SV, const SM: int, SP, SRest>
    CombCoshapeChain<MODE, Til<BasisTerm<SV, Mode<SM, SP>>, SRest>> for Nil
where
    SV: [const] BasisCoeff,
    SP: [const] BasisPath,
    Mode<SM, SP>: [const] BasisPath,
    SRest: [const] BasisTuple,
    BasisTerm<SV, Mode<SM, SP>>: Leaf,
    Self: [const] CombCoshapeChainStep<
            MODE,
            Til<BasisTerm<SV, Mode<SM, SP>>, SRest>,
            false,
            { EQ_MODE::<SM, MODE> },
        >,
{
    type Output = <Self as CombCoshapeChainStep<
        MODE,
        Til<BasisTerm<SV, Mode<SM, SP>>, SRest>,
        false,
        { EQ_MODE::<SM, MODE> },
    >>::Output;
    fn comb_coshape_chain(self, skel: Til<BasisTerm<SV, Mode<SM, SP>>, SRest>) -> Self::Output {
        self.comb_coshape_chain_step(skel)
    }
}

const impl<const MODE: int, V, const M: int, P, Rest, SV, const SM: int, SP, SRest>
    CombCoshapeChain<MODE, Til<BasisTerm<SV, Mode<SM, SP>>, SRest>>
    for Til<BasisTerm<V, Mode<M, P>>, Rest>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    Mode<M, P>: [const] BasisPath,
    Rest: [const] BasisTuple,
    SV: [const] BasisCoeff,
    SP: [const] BasisPath,
    Mode<SM, SP>: [const] BasisPath,
    SRest: [const] BasisTuple,
    BasisTerm<V, Mode<M, P>>: Leaf,
    BasisTerm<SV, Mode<SM, SP>>: Leaf,
    Self: [const] CombCoshapeChainStep<
            MODE,
            Til<BasisTerm<SV, Mode<SM, SP>>, SRest>,
            { EQ_MODE::<M, MODE> },
            { EQ_MODE::<SM, MODE> },
        >,
{
    type Output = <Self as CombCoshapeChainStep<
        MODE,
        Til<BasisTerm<SV, Mode<SM, SP>>, SRest>,
        { EQ_MODE::<M, MODE> },
        { EQ_MODE::<SM, MODE> },
    >>::Output;
    fn comb_coshape_chain(self, skel: Til<BasisTerm<SV, Mode<SM, SP>>, SRest>) -> Self::Output {
        self.comb_coshape_chain_step(skel)
    }
}

const impl<
    const MODE: int,
    Off: [const] CondSplitModePrefix<MODE, OFF_MATCH>,
    Skel: [const] CondSplitModePrefix<MODE, SKEL_MATCH>,
    const OFF_MATCH: bool,
    const SKEL_MATCH: bool,
> CombCoshapeChainStep<MODE, Skel, OFF_MATCH, SKEL_MATCH> for Off
where
    <Off as CondSplitModePrefix<MODE, OFF_MATCH>>::Head:
        [const] CombCoshapeValue<<Skel as CondSplitModePrefix<MODE, SKEL_MATCH>>::Head>,
    <Off as CondSplitModePrefix<MODE, OFF_MATCH>>::Tail: [const] CombCoshapeChain<
            { INCREASE::<MODE> },
            <Skel as CondSplitModePrefix<MODE, SKEL_MATCH>>::Tail,
        >,
{
    type Output = Til<
        <<Off as CondSplitModePrefix<MODE, OFF_MATCH>>::Head as CombCoshapeValue<
            <Skel as CondSplitModePrefix<MODE, SKEL_MATCH>>::Head,
        >>::Output,
        <<Off as CondSplitModePrefix<MODE, OFF_MATCH>>::Tail as CombCoshapeChain<
            { INCREASE::<MODE> },
            <Skel as CondSplitModePrefix<MODE, SKEL_MATCH>>::Tail,
        >>::Output,
    >;

    fn comb_coshape_chain_step(self, skel: Skel) -> Self::Output {
        let (oh, ot) = self.cond_split_mode_prefix();
        let (sh, st) = skel.cond_split_mode_prefix();
        Til(oh.comb_coshape_value(sh), ot.comb_coshape_chain(st))
    }
}

/// Converts a scalar or basis maximum offset (`Self`) into a bounding codomain `Shape`
/// guided by active path skeleton `Skel`.
#[doc(hidden)]
pub const trait MaxOffsetToCoshape<Skel: [const] BasisTuple>: Copy {
    type Output: [const] Shape;
    fn max_offset_to_coshape(self, skeleton: Skel) -> Self::Output;
}

const impl<I: [const] OffsetExtent> MaxOffsetToCoshape<Nil> for I {
    type Output = <I as OffsetExtent>::Output;
    fn max_offset_to_coshape(self, _: Nil) -> Self::Output {
        self.offset_extent()
    }
}

const impl<V, P, Skel> MaxOffsetToCoshape<Skel> for BasisTerm<V, P>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    Skel: [const] BasisTuple,
    Self: [const] IntoBasisCombination + Leaf,
    <Self as IntoBasisCombination>::Terms: [const] CombCoshapeValue<Skel>,
{
    type Output = <<Self as IntoBasisCombination>::Terms as CombCoshapeValue<Skel>>::Output;
    fn max_offset_to_coshape(self, skeleton: Skel) -> Self::Output {
        self.into_basis_terms().comb_coshape_value(skeleton)
    }
}

const impl<T, Skel> MaxOffsetToCoshape<Skel> for BasisCombination<T>
where
    T: [const] CombCoshapeValue<Skel>,
    Skel: [const] BasisTuple,
    Self: Leaf,
{
    type Output = <T as CombCoshapeValue<Skel>>::Output;
    fn max_offset_to_coshape(self, skeleton: Skel) -> Self::Output {
        self.terms().comb_coshape_value(skeleton)
    }
}

/// Computes the codomain shape of a layout defined by `(shape, stride)`.
///
/// Requires positive leaf extents in `shape` and non-negative maximum offsets
/// (e.g., non-negative strides); panics on zero/negative extents, negative offsets, or [`int`] overflow.
pub const trait Coshape<D: Stride<Self>>: [const] Shape {
    /// Bounding codomain shape type.
    type Output: [const] Shape;
    /// Evaluates the codomain shape for `self` paired with `stride`.
    fn coshape(self, stride: D) -> Self::Output;
}

const impl<S, D> Coshape<D> for S
where
    S: [const] MaxCoord + [const] Shape,
    D: Stride<S> + [const] StrideSkeleton,
    <S as MaxCoord>::Output: [const] InnerProduct<D>,
    <<S as MaxCoord>::Output as InnerProduct<D>>::Output:
        [const] MaxOffsetToCoshape<<D as StrideSkeleton>::Terms>,
{
    type Output = <<<S as MaxCoord>::Output as InnerProduct<D>>::Output as MaxOffsetToCoshape<
        <D as StrideSkeleton>::Terms,
    >>::Output;

    fn coshape(self, stride: D) -> Self::Output {
        let max_coord = self.max_coord();
        let skeleton = stride.skeleton_terms();
        let max_offset = super::inner_product(max_coord, stride);
        max_offset.max_offset_to_coshape(skeleton)
    }
}

/// Computes the codomain shape of a layout defined by a congruent `(shape, stride)` pair.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::stride::coshape;
/// use noodle_cute::{e, int_tuple, tuple};
///
/// let shape = int_tuple![Int::<4>, Int::<8>];
/// let scalar_stride = int_tuple![Int::<1>, Int::<4>];
/// assert_eq!(coshape(shape, scalar_stride), Int::<32>);
///
/// let basis_stride = tuple![e![0], e![2]];
/// assert_eq!(
///     coshape(shape, basis_stride),
///     int_tuple![Int::<4>, Int::<1>, Int::<8>]
/// );
/// ```
pub const fn coshape<S: [const] Coshape<D>, D: Stride<S>>(shape: S, stride: D) -> S::Output {
    shape.coshape(stride)
}
