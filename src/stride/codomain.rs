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
//! - [`coshape`] bounds a zero-origin codomain. Every mode must have non-negative
//!   contributions. Use [`linear_bounds`] for signed linear address ranges.
//! - Sparse gaps in active basis paths are padded with `Int<1>` in the resulting coshape.

use crate::NoodleCuteError;
use crate::basis::{
    BasisCoeff, BasisCombination, BasisPath, BasisTerm, BasisTuple, COEFF_IS_STATIC_ZERO,
    CanonicalBasis, CombAsTuple, ComparePath, EQ_MODE, EmitBasisTerm, IntoBasisCombination,
    MergeAdd, Mode, ModeEnd, PATH_ORDER, PathOrdering, SplitModePrefix,
};
use crate::f2::{F2, F2OrOutput, F2OrScalar, F2Scalar, ValidF2Value, clmul};
use crate::integer::{
    ArithAdd, BOTH_STATIC, INCREASE, IS_NON_NEG, Int, Integer, Sum, add, bit_length_non_neg, bitor,
    int, mul, shiftl, sub,
};
use crate::shape::{Shape, ShapeScalar, ValidShapeExtent};
use crate::tuple::{Item, Leaf, Nil, Til, Tuple};

use super::{Stride, StrideMul};

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
/// use noodle_cute::tuple::{LeafProfile, Nil, NilProfile, Til, TupleProfile};
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
/// type Scalar2D = Til<Int<1>, Til<Int<2>, Nil>>;
/// assert_eq!(profile_of::<Scalar2D>(), TypeId::of::<LeafProfile>());
///
/// // Multi-mode basis strides merge their active axes into the codomain profile
/// type Basis2D = Til<Basis<Mode<0>>, Til<Basis<Mode<1>>, Nil>>;
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

/// Maps a non-negative offset leaf scalar to its coshape extent
/// (`offset + 1` for integers, `1 << bit_length(offset)` for `F2`).
#[doc(hidden)]
pub const trait OffsetExtent: Copy {
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

const F2_COSHAPE_EXTENT<const N: int>: int = shiftl(1i32, bit_length_non_neg(N));

const impl OffsetExtent for F2<int> {
    type Output = int;
    fn offset_extent(self) -> int {
        shiftl(1i32, bit_length_non_neg(self.value()))
    }
}

const impl<const N: int> OffsetExtent for F2<Int<N>>
where
    (): [const] ValidF2Value<{ IS_NON_NEG::<N> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ F2_COSHAPE_EXTENT::<N> }> }>,
{
    type Output = Int<{ F2_COSHAPE_EXTENT::<N> }>;
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

const impl<V: [const] F2Scalar> StrideSkeleton for F2<V> {
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

/// Combines two scalar mode-maximum contributions for codomain bounding:
/// ordinary addition for monotone [`Integer`] codomains, or bitwise OR for non-monotone `F2`
/// codomains so bits never cancel.
#[doc(hidden)]
pub const trait CoshapeCoeffAccum<R>: BasisCoeff {
    type Output: [const] BasisCoeff;
    fn coshape_coeff_accum(self, rhs: R) -> Self::Output;
}

const impl<L, R> CoshapeCoeffAccum<R> for L
where
    L: [const] Integer + [const] ArithAdd<L, R>,
    R: [const] Integer,
    Sum<L, R>: [const] BasisCoeff,
{
    type Output = Sum<L, R>;
    fn coshape_coeff_accum(self, rhs: R) -> Self::Output {
        add(self, rhs)
    }
}

const impl<L, R> CoshapeCoeffAccum<F2<R>> for F2<L>
where
    L: [const] F2Scalar + [const] F2OrScalar<R, { BOTH_STATIC::<L, R> }>,
    R: [const] F2Scalar,
{
    type Output = F2<F2OrOutput<L, R>>;
    fn coshape_coeff_accum(self, rhs: F2<R>) -> Self::Output {
        crate::f2::f2_or(self, rhs)
    }
}

const impl<L: [const] F2Scalar> CoshapeCoeffAccum<Int<0>> for F2<L> {
    type Output = Self;
    fn coshape_coeff_accum(self, _: Int<0>) -> Self {
        self
    }
}

const impl<R: [const] F2Scalar> CoshapeCoeffAccum<F2<R>> for Int<0> {
    type Output = F2<R>;
    fn coshape_coeff_accum(self, rhs: F2<R>) -> F2<R> {
        rhs
    }
}

/// Merge-combines two path-sorted `BasisTuple` chains using [`CoshapeCoeffAccum`].
#[doc(hidden)]
pub const trait MergeCoshapeAdd<
    R: [const] BasisTuple,
    const ORDER: PathOrdering = { PathOrdering::Dispatch },
>: [const] BasisTuple
{
    type Output: [const] BasisTuple;
    fn merge_coshape_add(self, rhs: R) -> Self::Output;
}

const impl MergeCoshapeAdd<Nil> for Nil {
    type Output = Nil;
    fn merge_coshape_add(self, _: Nil) -> Nil {
        Nil
    }
}

const impl<V, P, T> MergeCoshapeAdd<Nil> for Til<BasisTerm<V, P>, T>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    T: [const] MergeCoshapeAdd<Nil>,
    <T as MergeCoshapeAdd<Nil>>::Output: [const] EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>,
{
    type Output = <<T as MergeCoshapeAdd<Nil>>::Output as EmitBasisTerm<
        V,
        P,
        { COEFF_IS_STATIC_ZERO::<V> },
    >>::Output;
    fn merge_coshape_add(self, _: Nil) -> Self::Output {
        self.1.merge_coshape_add(Nil).emit(self.0.value)
    }
}

const impl<V, P, T> MergeCoshapeAdd<Til<BasisTerm<V, P>, T>> for Nil
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    T: [const] BasisTuple,
    Nil: [const] MergeCoshapeAdd<T>,
    <Nil as MergeCoshapeAdd<T>>::Output: [const] EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>,
{
    type Output = <<Nil as MergeCoshapeAdd<T>>::Output as EmitBasisTerm<
        V,
        P,
        { COEFF_IS_STATIC_ZERO::<V> },
    >>::Output;
    fn merge_coshape_add(self, rhs: Til<BasisTerm<V, P>, T>) -> Self::Output {
        Nil.merge_coshape_add(rhs.1).emit(rhs.0.value)
    }
}

const impl<VL, PL, TL, VR, PR, TR> MergeCoshapeAdd<Til<BasisTerm<VR, PR>, TR>>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: ComparePath<PR>,
    TL: [const] BasisTuple,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    Self: [const] MergeCoshapeAdd<Til<BasisTerm<VR, PR>, TR>, { PATH_ORDER::<PL, PR> }>,
{
    type Output =
        <Self as MergeCoshapeAdd<Til<BasisTerm<VR, PR>, TR>, { PATH_ORDER::<PL, PR> }>>::Output;
    fn merge_coshape_add(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Self::Output {
        <Self as MergeCoshapeAdd<Til<BasisTerm<VR, PR>, TR>, { PATH_ORDER::<PL, PR> }>>::merge_coshape_add(
            self, rhs,
        )
    }
}

const impl<VL, PL, TL, VR, PR, TR>
    MergeCoshapeAdd<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::Equal }>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff + [const] CoshapeCoeffAccum<VR>,
    PL: [const] BasisPath,
    TL: [const] BasisTuple + [const] MergeCoshapeAdd<TR>,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    <TL as MergeCoshapeAdd<TR>>::Output:
        [const] EmitBasisTerm<VL::Output, PL, { COEFF_IS_STATIC_ZERO::<VL::Output> }>,
{
    type Output = <<TL as MergeCoshapeAdd<TR>>::Output as EmitBasisTerm<
        VL::Output,
        PL,
        { COEFF_IS_STATIC_ZERO::<VL::Output> },
    >>::Output;
    fn merge_coshape_add(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Self::Output {
        self.1
            .merge_coshape_add(rhs.1)
            .emit(self.0.value.coshape_coeff_accum(rhs.0.value))
    }
}

const impl<VL, PL, TL, VR, PR, TR>
    MergeCoshapeAdd<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::Less }>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: [const] BasisPath,
    TL: [const] BasisTuple + [const] MergeCoshapeAdd<Til<BasisTerm<VR, PR>, TR>>,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    <TL as MergeCoshapeAdd<Til<BasisTerm<VR, PR>, TR>>>::Output:
        [const] EmitBasisTerm<VL, PL, { COEFF_IS_STATIC_ZERO::<VL> }>,
{
    type Output = <<TL as MergeCoshapeAdd<Til<BasisTerm<VR, PR>, TR>>>::Output as EmitBasisTerm<
        VL,
        PL,
        { COEFF_IS_STATIC_ZERO::<VL> },
    >>::Output;
    fn merge_coshape_add(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Self::Output {
        self.1.merge_coshape_add(rhs).emit(self.0.value)
    }
}

const impl<VL, PL, TL, VR, PR, TR>
    MergeCoshapeAdd<Til<BasisTerm<VR, PR>, TR>, { PathOrdering::Greater }>
    for Til<BasisTerm<VL, PL>, TL>
where
    VL: [const] BasisCoeff,
    PL: [const] BasisPath,
    TL: [const] BasisTuple,
    VR: [const] BasisCoeff,
    PR: [const] BasisPath,
    TR: [const] BasisTuple,
    Self: [const] MergeCoshapeAdd<TR>,
    <Self as MergeCoshapeAdd<TR>>::Output:
        [const] EmitBasisTerm<VR, PR, { COEFF_IS_STATIC_ZERO::<VR> }>,
{
    type Output = <<Self as MergeCoshapeAdd<TR>>::Output as EmitBasisTerm<
        VR,
        PR,
        { COEFF_IS_STATIC_ZERO::<VR> },
    >>::Output;
    fn merge_coshape_add(self, rhs: Til<BasisTerm<VR, PR>, TR>) -> Self::Output {
        self.merge_coshape_add(rhs.1).emit(rhs.0.value)
    }
}

/// Combines two basis contributions for codomain bounding via [`MergeCoshapeAdd`].
#[doc(hidden)]
pub const trait CoshapeBasisAdd<R>: [const] IntoBasisCombination {
    type Output: Copy;
    fn coshape_basis_add(self, rhs: R) -> Self::Output;
}

const impl<L, R> CoshapeBasisAdd<R> for L
where
    L: [const] IntoBasisCombination,
    R: [const] IntoBasisCombination,
    L::Terms: [const] MergeCoshapeAdd<R::Terms>,
    <L::Terms as MergeCoshapeAdd<R::Terms>>::Output: [const] CanonicalBasis,
{
    type Output = <<L::Terms as MergeCoshapeAdd<R::Terms>>::Output as CanonicalBasis>::Output;
    fn coshape_basis_add(self, rhs: R) -> Self::Output {
        self.into_basis_terms()
            .merge_coshape_add(rhs.into_basis_terms())
            .canonical_basis()
    }
}

/// Dispatches `Integer` accumulated with `BasisCombination<Nil>` during `coshape`.
#[doc(hidden)]
pub const trait CoshapeScalarEmptyComb<const IS_ZERO: bool>: Integer {
    type Output: Copy;
    fn coshape_scalar_empty_comb(self, rhs: BasisCombination<Nil>) -> Self::Output;
}

const impl<A: Integer> CoshapeScalarEmptyComb<true> for A {
    type Output = BasisCombination<Nil>;
    fn coshape_scalar_empty_comb(self, rhs: BasisCombination<Nil>) -> Self::Output {
        rhs
    }
}

const impl<A: Integer> CoshapeScalarEmptyComb<false> for A {
    type Output = A;
    fn coshape_scalar_empty_comb(self, _: BasisCombination<Nil>) -> Self::Output {
        self
    }
}

/// Dispatches `BasisCombination<Nil>` accumulated with scalar `B` during `coshape`.
#[doc(hidden)]
pub const trait CoshapeEmptyCombScalar<B: Integer, const IS_ZERO: bool>: Copy {
    type Output: Copy;
    fn coshape_empty_comb_scalar(self, rhs: B) -> Self::Output;
}

const impl<B: Integer> CoshapeEmptyCombScalar<B, true> for BasisCombination<Nil> {
    type Output = BasisCombination<Nil>;
    fn coshape_empty_comb_scalar(self, _: B) -> Self::Output {
        self
    }
}

const impl<B: Integer> CoshapeEmptyCombScalar<B, false> for BasisCombination<Nil> {
    type Output = B;
    fn coshape_empty_comb_scalar(self, rhs: B) -> Self::Output {
        rhs
    }
}

/// Accumulates maximal mode contributions `(s_i - 1) * d_i` across leaves for [`Coshape`].
#[doc(hidden)]
pub const trait CoshapeAccumulate<R>: Copy {
    type Output: Copy;
    fn coshape_accumulate(self, rhs: R) -> Self::Output;
}

const impl CoshapeAccumulate<Nil> for Nil {
    type Output = Nil;
    fn coshape_accumulate(self, _: Nil) -> Nil {
        Nil
    }
}

const impl<B: Integer> CoshapeAccumulate<B> for Nil {
    type Output = B;
    fn coshape_accumulate(self, rhs: B) -> B {
        rhs
    }
}

const impl<V: BasisCoeff, P: Copy> CoshapeAccumulate<BasisTerm<V, P>> for Nil {
    type Output = BasisTerm<V, P>;
    fn coshape_accumulate(self, rhs: BasisTerm<V, P>) -> Self::Output {
        rhs
    }
}

const impl<T: BasisTuple> CoshapeAccumulate<BasisCombination<T>> for Nil {
    type Output = BasisCombination<T>;
    fn coshape_accumulate(self, rhs: BasisCombination<T>) -> Self::Output {
        rhs
    }
}

const impl<A: Integer> CoshapeAccumulate<Nil> for A {
    type Output = A;
    fn coshape_accumulate(self, _: Nil) -> A {
        self
    }
}

const impl<A, B> CoshapeAccumulate<B> for A
where
    A: [const] Integer + [const] ArithAdd<A, B>,
    B: [const] Integer,
{
    type Output = Sum<A, B>;
    fn coshape_accumulate(self, rhs: B) -> Self::Output {
        add(self, rhs)
    }
}

const impl<V: BasisCoeff, P: Copy> CoshapeAccumulate<BasisTerm<V, P>> for Int<0> {
    type Output = BasisTerm<V, P>;
    fn coshape_accumulate(self, rhs: BasisTerm<V, P>) -> Self::Output {
        rhs
    }
}

const impl<A> CoshapeAccumulate<BasisCombination<Nil>> for A
where
    A: [const] Integer
        + [const] BasisCoeff
        + [const] CoshapeScalarEmptyComb<{ COEFF_IS_STATIC_ZERO::<A> }>,
{
    type Output = <A as CoshapeScalarEmptyComb<{ COEFF_IS_STATIC_ZERO::<A> }>>::Output;
    fn coshape_accumulate(self, rhs: BasisCombination<Nil>) -> Self::Output {
        self.coshape_scalar_empty_comb(rhs)
    }
}

const impl<H, T> CoshapeAccumulate<BasisCombination<Til<H, T>>> for Int<0>
where
    H: Copy,
    T: BasisTuple,
    Til<H, T>: BasisTuple,
{
    type Output = BasisCombination<Til<H, T>>;
    fn coshape_accumulate(self, rhs: BasisCombination<Til<H, T>>) -> Self::Output {
        rhs
    }
}

const impl<V: BasisCoeff, P: Copy> CoshapeAccumulate<Nil> for BasisTerm<V, P> {
    type Output = Self;
    fn coshape_accumulate(self, _: Nil) -> Self::Output {
        self
    }
}

const impl<V: BasisCoeff, P: Copy> CoshapeAccumulate<Int<0>> for BasisTerm<V, P> {
    type Output = Self;
    fn coshape_accumulate(self, _: Int<0>) -> Self::Output {
        self
    }
}

const impl<VL, PL, VR, PR> CoshapeAccumulate<BasisTerm<VR, PR>> for BasisTerm<VL, PL>
where
    VL: BasisCoeff,
    PL: Copy,
    Self: [const] CoshapeBasisAdd<BasisTerm<VR, PR>>,
{
    type Output = <Self as CoshapeBasisAdd<BasisTerm<VR, PR>>>::Output;
    fn coshape_accumulate(self, rhs: BasisTerm<VR, PR>) -> Self::Output {
        self.coshape_basis_add(rhs)
    }
}

const impl<VL, PL, T> CoshapeAccumulate<BasisCombination<T>> for BasisTerm<VL, PL>
where
    VL: BasisCoeff,
    PL: Copy,
    T: BasisTuple,
    Self: [const] CoshapeBasisAdd<BasisCombination<T>>,
{
    type Output = <Self as CoshapeBasisAdd<BasisCombination<T>>>::Output;
    fn coshape_accumulate(self, rhs: BasisCombination<T>) -> Self::Output {
        self.coshape_basis_add(rhs)
    }
}

const impl CoshapeAccumulate<Nil> for BasisCombination<Nil> {
    type Output = Self;
    fn coshape_accumulate(self, _: Nil) -> Self::Output {
        self
    }
}

const impl<B> CoshapeAccumulate<B> for BasisCombination<Nil>
where
    B: [const] Integer + [const] BasisCoeff,
    Self: [const] CoshapeEmptyCombScalar<B, { COEFF_IS_STATIC_ZERO::<B> }>,
{
    type Output = <Self as CoshapeEmptyCombScalar<B, { COEFF_IS_STATIC_ZERO::<B> }>>::Output;
    fn coshape_accumulate(self, rhs: B) -> Self::Output {
        self.coshape_empty_comb_scalar(rhs)
    }
}

const impl<V: BasisCoeff, P: Copy> CoshapeAccumulate<BasisTerm<V, P>> for BasisCombination<Nil> {
    type Output = BasisTerm<V, P>;
    fn coshape_accumulate(self, rhs: BasisTerm<V, P>) -> Self::Output {
        rhs
    }
}

const impl<TR: BasisTuple> CoshapeAccumulate<BasisCombination<TR>> for BasisCombination<Nil> {
    type Output = BasisCombination<TR>;
    fn coshape_accumulate(self, rhs: BasisCombination<TR>) -> Self::Output {
        rhs
    }
}

const impl<H, TL> CoshapeAccumulate<Nil> for BasisCombination<Til<H, TL>>
where
    H: Copy,
    TL: BasisTuple,
    Til<H, TL>: BasisTuple,
{
    type Output = Self;
    fn coshape_accumulate(self, _: Nil) -> Self::Output {
        self
    }
}

const impl<H, TL> CoshapeAccumulate<Int<0>> for BasisCombination<Til<H, TL>>
where
    H: Copy,
    TL: BasisTuple,
    Til<H, TL>: BasisTuple,
{
    type Output = Self;
    fn coshape_accumulate(self, _: Int<0>) -> Self::Output {
        self
    }
}

const impl<H, TL, VR, PR> CoshapeAccumulate<BasisTerm<VR, PR>> for BasisCombination<Til<H, TL>>
where
    H: Copy,
    TL: BasisTuple,
    Til<H, TL>: BasisTuple,
    Self: [const] CoshapeBasisAdd<BasisTerm<VR, PR>>,
{
    type Output = <Self as CoshapeBasisAdd<BasisTerm<VR, PR>>>::Output;
    fn coshape_accumulate(self, rhs: BasisTerm<VR, PR>) -> Self::Output {
        self.coshape_basis_add(rhs)
    }
}

const impl<H, TL, TR> CoshapeAccumulate<BasisCombination<TR>> for BasisCombination<Til<H, TL>>
where
    H: Copy,
    TL: BasisTuple,
    Til<H, TL>: BasisTuple,
    TR: BasisTuple,
    Self: [const] CoshapeBasisAdd<BasisCombination<TR>>,
{
    type Output = <Self as CoshapeBasisAdd<BasisCombination<TR>>>::Output;
    fn coshape_accumulate(self, rhs: BasisCombination<TR>) -> Self::Output {
        self.coshape_basis_add(rhs)
    }
}

const impl<V: [const] F2Scalar> CoshapeAccumulate<F2<V>> for Nil {
    type Output = F2<V>;
    fn coshape_accumulate(self, rhs: F2<V>) -> F2<V> {
        rhs
    }
}

const impl<V: [const] F2Scalar> CoshapeAccumulate<Nil> for F2<V> {
    type Output = Self;
    fn coshape_accumulate(self, _: Nil) -> Self {
        self
    }
}

const impl<V: [const] F2Scalar> CoshapeAccumulate<Int<0>> for F2<V> {
    type Output = Self;
    fn coshape_accumulate(self, _: Int<0>) -> Self {
        self
    }
}

const impl<V: [const] F2Scalar> CoshapeAccumulate<F2<V>> for Int<0> {
    type Output = F2<V>;
    fn coshape_accumulate(self, rhs: F2<V>) -> F2<V> {
        rhs
    }
}

const impl<L, R> CoshapeAccumulate<F2<R>> for F2<L>
where
    L: [const] F2Scalar + [const] F2OrScalar<R, { BOTH_STATIC::<L, R> }>,
    R: [const] F2Scalar,
{
    type Output = F2<F2OrOutput<L, R>>;
    fn coshape_accumulate(self, rhs: F2<R>) -> Self::Output {
        self.coshape_coeff_accum(rhs)
    }
}

const impl<V: [const] F2Scalar> CoshapeAccumulate<BasisCombination<Nil>> for F2<V> {
    type Output = Self;
    fn coshape_accumulate(self, _: BasisCombination<Nil>) -> Self {
        self
    }
}

const impl<V: [const] F2Scalar> CoshapeAccumulate<F2<V>> for BasisCombination<Nil> {
    type Output = F2<V>;
    fn coshape_accumulate(self, rhs: F2<V>) -> F2<V> {
        rhs
    }
}

const impl<V: BasisCoeff, P: Copy> CoshapeAccumulate<BasisTerm<V, P>> for F2<Int<0>> {
    type Output = BasisTerm<V, P>;
    fn coshape_accumulate(self, rhs: BasisTerm<V, P>) -> Self::Output {
        rhs
    }
}

const impl<V: BasisCoeff, P: Copy> CoshapeAccumulate<F2<Int<0>>> for BasisTerm<V, P> {
    type Output = Self;
    fn coshape_accumulate(self, _: F2<Int<0>>) -> Self::Output {
        self
    }
}

const impl<H, T> CoshapeAccumulate<BasisCombination<Til<H, T>>> for F2<Int<0>>
where
    H: Copy,
    T: BasisTuple,
    Til<H, T>: BasisTuple,
{
    type Output = BasisCombination<Til<H, T>>;
    fn coshape_accumulate(self, rhs: BasisCombination<Til<H, T>>) -> Self::Output {
        rhs
    }
}

const impl<H, T> CoshapeAccumulate<F2<Int<0>>> for BasisCombination<Til<H, T>>
where
    H: Copy,
    T: BasisTuple,
    Til<H, T>: BasisTuple,
{
    type Output = Self;
    fn coshape_accumulate(self, _: F2<Int<0>>) -> Self::Output {
        self
    }
}

/// Verifies each coordinate component of a mode contribution fits a zero-origin codomain.
#[doc(hidden)]
pub const trait CoshapeContribution: Copy {
    /// Rejects contributions reaching below the origin before modes are combined.
    fn validate(self);
}

const NONNEGATIVE_CONTRIBUTION<V: BasisCoeff>: bool =
    !matches!(V::STATIC_CMP_ZERO, Some(core::cmp::Ordering::Less));

const impl<V: [const] BasisCoeff> CoshapeContribution for V
where
    (): [const] ValidShapeExtent<{ NONNEGATIVE_CONTRIBUTION::<V> }>,
{
    fn validate(self) {
        NoodleCuteError::InvalidShape.check_with(
            !matches!(self.cmp_zero(), core::cmp::Ordering::Less),
            "coshape requires non-negative mode contributions",
        );
    }
}

const impl<V: [const] CoshapeContribution, P: Copy> CoshapeContribution for BasisTerm<V, P> {
    fn validate(self) {
        self.value.validate();
    }
}

const impl CoshapeContribution for Nil {
    fn validate(self) {}
}

const impl<H: [const] CoshapeContribution, T: [const] CoshapeContribution> CoshapeContribution
    for Til<H, T>
{
    fn validate(self) {
        self.0.validate();
        self.1.validate();
    }
}

const impl<T: BasisTuple + [const] CoshapeContribution> CoshapeContribution
    for BasisCombination<T>
{
    fn validate(self) {
        self.terms().validate();
    }
}

/// Folds maximal mode contributions `(s_i - 1) * d_i` via [`CoshapeAccumulate`].
#[doc(hidden)]
pub const trait CoshapeInnerProduct<D, Acc = Int<0>>: Copy {
    type Output: Copy;
    fn coshape_inner_product(self, stride: D, acc: Acc) -> Self::Output;
}

const impl<Acc: Copy> CoshapeInnerProduct<Nil, Acc> for Nil {
    type Output = Acc;
    fn coshape_inner_product(self, _: Nil, acc: Acc) -> Self::Output {
        acc
    }
}

const impl<S, D, Acc> CoshapeInnerProduct<D, Acc> for S
where
    S: [const] Integer,
    D: [const] StrideMul<S>,
    D::Output: [const] CoshapeContribution,
    Acc: [const] CoshapeAccumulate<D::Output>,
{
    type Output = Acc::Output;
    fn coshape_inner_product(self, stride: D, acc: Acc) -> Self::Output {
        let contribution = stride.stride_mul(self);
        contribution.validate();
        acc.coshape_accumulate(contribution)
    }
}

const impl<H, T, RH, RT, Acc> CoshapeInnerProduct<Til<RH, RT>, Acc> for Til<H, T>
where
    Acc: Copy,
    H: [const] CoshapeInnerProduct<RH, Acc>,
    T: Tuple + [const] CoshapeInnerProduct<RT, H::Output>,
    RH: Copy,
    RT: Copy,
{
    type Output = T::Output;
    fn coshape_inner_product(self, stride: Til<RH, RT>, acc: Acc) -> Self::Output {
        let acc1 = self.0.coshape_inner_product(stride.0, acc);
        self.1.coshape_inner_product(stride.1, acc1)
    }
}

/// Computes the codomain shape of a layout defined by `(shape, stride)`.
///
/// Requires positive leaf extents in `shape` and non-negative per-mode offsets
/// along each codomain axis. Use [`linear_bounds`] for signed linear address ranges.
/// Static negative mode contributions are rejected during compilation.
///
/// ```compile_fail
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{integer::Int, int_tuple, stride::coshape};
/// let _ = coshape(int_tuple![Int::<2>, Int::<2>], int_tuple![Int::<-1>, Int::<2>]);
/// ```
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
    <S as MaxCoord>::Output: [const] CoshapeInnerProduct<D>,
    <<S as MaxCoord>::Output as CoshapeInnerProduct<D>>::Output:
        [const] MaxOffsetToCoshape<<D as StrideSkeleton>::Terms>,
{
    type Output =
        <<<S as MaxCoord>::Output as CoshapeInnerProduct<D>>::Output as MaxOffsetToCoshape<
            <D as StrideSkeleton>::Terms,
        >>::Output;

    fn coshape(self, stride: D) -> Self::Output {
        let max_coord = self.max_coord();
        let skeleton = stride.skeleton_terms();
        let max_offset = max_coord.coshape_inner_product(stride, Int::<0>);
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
///
/// A static codomain bit span of `1 << 31` exceeds [`int::MAX`] and fails at compile time.
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::f2::f2;
/// use noodle_cute::integer::Int;
/// use noodle_cute::stride::coshape;
///
/// let _ = coshape(Int::<2>, f2(Int::<1_073_741_824>));
/// ```
pub const fn coshape<S: [const] Coshape<D>, D: Stride<S>>(shape: S, stride: D) -> S::Output {
    shape.coshape(stride)
}

/// Accumulator tracking reachable linear offsets across leaves of a non-empty layout.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearBoundsAcc {
    /// No non-zero leaf spans accumulated yet (`[0, 0]`).
    Zero,
    /// Integer-strided closed offset interval `[min, max]`.
    Integer(int, int),
    /// Binary-field (`F2`) accumulated bitwise OR mask of maximal leaf spans.
    F2Mask(int),
}

impl LinearBoundsAcc {
    /// Combines an integer leaf span `(s - 1) * d` into the running bounds.
    #[doc(hidden)]
    pub const fn push_integer_span(self, span: int) -> Self {
        let (lo_d, hi_d) = if span < 0 { (span, 0) } else { (0, span) };
        match self {
            Self::Zero => Self::Integer(lo_d, hi_d),
            Self::Integer(lo, hi) => Self::Integer(add(lo, lo_d), add(hi, hi_d)),
            Self::F2Mask(mask) => {
                assert!(
                    span == 0,
                    "cannot mix nonzero integer and F2 strides in linear bounds"
                );
                Self::F2Mask(mask)
            }
        }
    }

    /// Combines a carry-less `F2` leaf span `(s - 1) * d` into the running bit-span mask.
    #[doc(hidden)]
    pub const fn push_f2_span(self, span: int) -> Self {
        match self {
            Self::Zero => Self::F2Mask(span),
            Self::F2Mask(mask) => Self::F2Mask(bitor(mask, span)),
            Self::Integer(0, 0) => Self::F2Mask(span),
            Self::Integer(_, _) => {
                assert!(
                    span == 0,
                    "cannot mix nonzero integer and F2 strides in linear bounds"
                );
                self
            }
        }
    }

    /// Finalizes the accumulated bounds into a closed `[min_offset, max_offset]` interval.
    #[doc(hidden)]
    pub const fn finish(self) -> (int, int) {
        match self {
            Self::Zero => (0, 0),
            Self::Integer(lo, hi) => (lo, hi),
            Self::F2Mask(mask) => {
                let bits = bit_length_non_neg(mask);
                let hi = if bits == 31 {
                    int::MAX
                } else {
                    sub(shiftl(1i32, bits), 1)
                };
                (0, hi)
            }
        }
    }
}

/// Walks a congruent `(shape, stride)` pair to fold reachable linear offset bounds.
///
/// Implemented only for linear strides (integers, `F2`, `BasisCombination<Nil>`, `Nil`),
/// rejecting coordinate-basis strides at compile time.
#[doc(hidden)]
pub const trait LinearBoundsWalk<S: Shape>: Copy {
    fn walk_linear_bounds(self, shape: S, acc: LinearBoundsAcc) -> LinearBoundsAcc;
}

const impl LinearBoundsWalk<Nil> for Nil {
    fn walk_linear_bounds(self, _: Nil, acc: LinearBoundsAcc) -> LinearBoundsAcc {
        acc
    }
}

const impl<S: [const] ShapeScalar, I: [const] Integer> LinearBoundsWalk<S> for I {
    fn walk_linear_bounds(self, shape: S, acc: LinearBoundsAcc) -> LinearBoundsAcc {
        let max_c = sub(shape.value(), 1);
        let span = mul(max_c, self.value());
        acc.push_integer_span(span)
    }
}

const impl<S: [const] ShapeScalar, V: [const] F2Scalar> LinearBoundsWalk<S> for F2<V> {
    fn walk_linear_bounds(self, shape: S, acc: LinearBoundsAcc) -> LinearBoundsAcc {
        let max_c = sub(shape.value(), 1);
        let span = clmul(max_c, self.value());
        acc.push_f2_span(span)
    }
}

const impl<S: [const] ShapeScalar> LinearBoundsWalk<S> for BasisCombination<Nil> {
    fn walk_linear_bounds(self, _: S, acc: LinearBoundsAcc) -> LinearBoundsAcc {
        acc
    }
}

const impl<SH, ST, DH, DT> LinearBoundsWalk<Til<SH, ST>> for Til<DH, DT>
where
    SH: [const] Shape,
    ST: Tuple + [const] Shape,
    DH: [const] LinearBoundsWalk<SH>,
    DT: Tuple + [const] LinearBoundsWalk<ST>,
{
    fn walk_linear_bounds(self, shape: Til<SH, ST>, acc: LinearBoundsAcc) -> LinearBoundsAcc {
        let acc1 = self.0.walk_linear_bounds(shape.0, acc);
        self.1.walk_linear_bounds(shape.1, acc1)
    }
}

/// Computes the exact closed `[min_offset, max_offset]` interval reachable by all in-bounds
/// coordinates of a linear-strided `(shape, stride)` pair, or `None` when the domain is empty.
///
/// Unlike [`Coshape`], `LinearBounds`:
/// - handles negative strides accurately (returning negative `min_offset` when reached);
/// - handles zero-extent shapes cleanly (`None` when `shape.has_zero() == true`), while
///   preserving `Some((0, 0))` for rank-0 [`Nil`] shapes (which contain 1 coordinate);
/// - rejects non-linear coordinate-basis strides at compile time.
pub const trait LinearBounds<D: Stride<Self>>: [const] Shape {
    /// Returns `None` if `self` has any zero extent, or `Some((min_offset, max_offset))`.
    fn linear_bounds(self, stride: D) -> Option<(int, int)>;
}

const impl<S, D> LinearBounds<D> for S
where
    S: [const] Shape,
    D: Stride<S> + [const] LinearBoundsWalk<S>,
{
    fn linear_bounds(self, stride: D) -> Option<(int, int)> {
        if self.has_zero() {
            None
        } else {
            Some(
                stride
                    .walk_linear_bounds(self, LinearBoundsAcc::Zero)
                    .finish(),
            )
        }
    }
}

/// Returns the reachable linear offset interval `Some((min_offset, max_offset))` for a
/// congruent linear `(shape, stride)` pair, or `None` when `shape` has a zero extent.
pub const fn linear_bounds<S: [const] LinearBounds<D>, D: Stride<S>>(
    shape: S,
    stride: D,
) -> Option<(int, int)> {
    shape.linear_bounds(stride)
}
