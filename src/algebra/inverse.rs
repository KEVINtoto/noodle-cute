//! Layout right-inverse ($L^{\ddagger}$) and left-inverse ($L^{\dagger}$) algebra.
//!
//! Both operations invert a [`Layout`] $L : \mathbb{Z}_{|L|} \to D$ independently along
//! each codomain axis determined by [`Coprofile`], mapping coordinates in $L$'s codomain
//! back to 1D domain indices in $\mathbb{Z}_{|L|}$.
//!
//! - **Right inverse ([`right_inverse`])**:
//!   Returns the largest injective layout $R = L^{\ddagger}$ that undoes $L$ on the contiguous
//!   prefix of each codomain axis:
//!
//!   $$\mathrm{coprofile}(L) \preceq_{\text{weak}} \mathrm{shape}(R), \qquad R(L(R(i))) = R(i) \quad \forall i \in [0, |R|)$$
//!
//!   When $L$'s codomain is $\mathbb{Z}$, this strengthens to the canonical right inverse
//!   $L(R(i)) = i$ for all $i \in [0, |R|)$. Non-injective, negative, or gapped strides simply
//!   do not continue the chain and are skipped.
//!
//! - **Left inverse ([`left_inverse`])**:
//!   Returns a layout $R = L^{\dagger}$ that undoes $L$ on its entire domain (extending
//!   its codomain with padded hole extents where $L$'s image has gaps):
//!
//!   $$\mathrm{coprofile}(L) \preceq_{\text{weak}} \mathrm{shape}(R), \qquad L(R(L(i))) = L(i) \quad \forall i \in [0, |L|)$$
//!
//!   When $L$ is injective, $R(L(i)) = i$ for all $i \in [0, |L|)$. Along each codomain axis,
//!   $L$'s active positive strides ordered as $d_0 \le d_1 \le \dots$ (with extents $s_k$) must
//!   form an ordered divisible non-overlapping chain:
//!
//!   $$d_{k-1} \mid d_k \quad \text{and} \quad d_k / d_{k-1} \ge s_{k-1}$$
//!
//!   Violations are rejected at compile time for static inputs and panic in [`left_inverse`]
//!   for dynamic inputs.
//!
//! For binary-polynomial strides, inversion works on independently addressable
//! coordinate bits. Canonical two-mode polynomial radices use the divisor's
//! degree to bound the low digit. Static binary modes may split into smaller
//! modes when successive inverse columns have different strides. Runtime axes
//! with three or more active modes use binary elimination and return a fixed
//! hierarchy of binary or unit extents. A partial binary digit contributes its
//! independently variable low bits. The same basis handles non-binary static
//! axes of three or more modes. Mixed integer and binary axes
//! require binary domain extents so their index fields can be combined by XOR.
//! The stride walk may return a partial inverse when a mode cannot extend the
//! covered prefix.
//!
//! # Main Interfaces
//!
//! - [`RightInverse`] / [`right_inverse`] / [`Layout::right_inverse`]
//! - [`LeftInverse`] / [`left_inverse`] / [`Layout::left_inverse`]
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::inverse::{left_inverse, right_inverse};
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! // Contiguous bijection: right_inverse == left_inverse
//! let col = Layout::new(int_tuple![Int::<4>, Int::<8>], int_tuple![Int::<1>, Int::<4>]);
//! assert_eq!(right_inverse(col), Layout::new(Int::<32>, Int::<1>));
//! assert_eq!(left_inverse(col), Layout::new(Int::<32>, Int::<1>));
//!
//! // Gapped layout: right_inverse stops at the hole, left_inverse pads across it
//! let gap = Layout::new(int_tuple![Int::<4>, Int::<8>], int_tuple![Int::<1>, Int::<5>]);
//! assert_eq!(right_inverse(gap), Layout::new(Int::<4>, Int::<1>));
//! assert_eq!(
//!     left_inverse(gap),
//!     Layout::new(int_tuple![Int::<5>, Int::<8>], int_tuple![Int::<1>, Int::<4>])
//! );
//! ```
//!
//! # Notes
//!
//! Non-injective static layouts are rejected by `left_inverse` at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::inverse::left_inverse;
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let _ = left_inverse(Layout::new(
//!     int_tuple![Int::<63>, Int::<2>],
//!     int_tuple![Int::<1>, Int::<1>],
//! ));
//! ```
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::inverse::left_inverse;
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let _ = left_inverse(Layout::new(
//!     int_tuple![Int::<2>, Int::<3>],
//!     int_tuple![Int::<2>, Int::<1>],
//! ));
//! ```
//!
//! Coprime (non-chainable) static strides are rejected by `left_inverse` at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::inverse::left_inverse;
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let _ = left_inverse(Layout::new(
//!     int_tuple![Int::<2>, Int::<2>],
//!     int_tuple![Int::<2>, Int::<3>],
//! ));
//! ```
//!
//! Negative static strides are rejected by `left_inverse` at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::inverse::left_inverse;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let _ = left_inverse(Layout::new(Int::<4>, Int::<-2>));
//! ```
//!
//! Zero static extents are rejected by both `right_inverse` and `left_inverse` at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::inverse::right_inverse;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let _ = right_inverse(Layout::new(Int::<0>, Int::<1>));
//! ```
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::inverse::left_inverse;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let _ = left_inverse(Layout::new(Int::<0>, Int::<0>));
//! ```
//!
//! Multi-term basis combinations and incompatible stride depths fail at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::inverse::right_inverse;
//! use noodle_cute::e;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let _ = right_inverse(Layout::new(Int::<4>, e![0] + e![1]));
//! ```
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::inverse::left_inverse;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::{e, int_tuple, tuple};
//!
//! let _ = left_inverse(Layout::new(
//!     int_tuple![Int::<2>, Int::<3>],
//!     tuple![e![1], Int::<5>],
//! ));
//! ```

use core::marker::ConstParamTy;

use crate::basis::{
    Basis, BasisCoeff, BasisPath, CoeffAlgebra, ComparePath, MakeBasisLikeOutput,
    MakeBasisLikeTopProfile, ModeEnd, PathOrdering, make_basis_like,
};
use crate::f2::{
    BinaryPreimages, F2, F2Scalar, F2XorOutput, F2XorScalar, ValidF2Value, clmul_maybe_overflows,
    f2, f2_add,
};
use crate::integer::{
    BOTH_STATIC, IS_NON_NEG, Int, Integer, bit_floor, bit_length_non_neg, bitxor, carry_free_mask,
    div, eq, int, is_pow2, mul, mul_maybe_overflows, rem, sub, uint,
};
use crate::layout::{Coprofile, Layout, LayoutBase, ScaleIfNotOne, Tiler, tiler_to_layout};
use crate::shape::{
    AssertPositiveShape, BinaryExtents, IndexToCoord, Shape, ShapeScalar, ValidShapeExtent,
};
use crate::stride::{
    ContainsF2, InnerProduct, PrefixProduct, Stride, StrideScalar, prefix_product_with,
};
use crate::tuple::{Append, FlatTuple, Flatten, Item, Leaf, LeafProfile, Nil, Til, Tuple};

use super::coalesce::{Coalesce, DropTrailingUnit};

/// Scalar `(shape, stride, pps)` mode triple collected along a single codomain axis.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ModeTriple<S, D, P> {
    pub shape: S,
    pub stride: D,
    pub pps: P,
}

const impl<S: Item, D: BasisCoeff, P: Item> Item for ModeTriple<S, D, P> {
    const RANK: int = 1;
    const DEPTH: int = 0;
    const LEAVES: int = 1;
    const IS_STATIC: bool = S::IS_STATIC && D::IS_STATIC && P::IS_STATIC;
    type Profile = LeafProfile;
}

/// Compile-time decision for filtering one flattened `(S, D, P)` triple against `TargetP`.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum InverseAxisFilterAction {
    /// Drop static-zero strides, static size-1 modes, or modes belonging to another axis.
    Skip,
    /// Retain `(S, D::Value, P)` for this axis.
    Keep,
    /// Incompatible stride depth (prefix conflict between `D::Path` and `TargetP`).
    PrefixConflict,
}

const fn inverse_axis_filter_action<S, D, TargetP>() -> InverseAxisFilterAction
where
    S: ShapeScalar,
    D: StrideScalar,
    TargetP: BasisPath,
    D::Path: ComparePath<TargetP>,
{
    if D::Value::IS_STATIC_ZERO {
        return InverseAxisFilterAction::Skip;
    }
    match D::Path::ORDER {
        PathOrdering::Equal => {
            if matches!(S::STATIC_VALUE, Some(1)) {
                InverseAxisFilterAction::Skip
            } else {
                InverseAxisFilterAction::Keep
            }
        }
        PathOrdering::Less | PathOrdering::Greater => InverseAxisFilterAction::Skip,
        PathOrdering::PrefixConflict | PathOrdering::Dispatch => {
            InverseAxisFilterAction::PrefixConflict
        }
    }
}

const INVERSE_AXIS_FILTER_ACTION<S: ShapeScalar, D: StrideScalar, TargetP: BasisPath>:
    InverseAxisFilterAction = inverse_axis_filter_action::<S, D, TargetP>()
where
    D::Path: ComparePath<TargetP>;

/// Applies one [`InverseAxisFilterAction`] to prepend `(S, D::Value, P)` onto suffix `Self`.
#[doc(hidden)]
pub const trait FilterInverseStep<S, D, P, const ACTION: InverseAxisFilterAction>:
    FlatTuple
{
    type Output: FlatTuple;
    fn filter_inverse_step(self, s: S, d: D, p: P) -> Self::Output;
}

const impl<S: Copy, D: Copy, P: Copy, Rest: FlatTuple>
    FilterInverseStep<S, D, P, { InverseAxisFilterAction::Skip }> for Rest
{
    type Output = Rest;
    fn filter_inverse_step(self, _: S, _: D, _: P) -> Self::Output {
        self
    }
}

const impl<S, D, P, Rest> FilterInverseStep<S, D, P, { InverseAxisFilterAction::Keep }> for Rest
where
    S: ShapeScalar,
    D: [const] StrideScalar,
    P: ShapeScalar,
    Rest: FlatTuple,
{
    type Output = Til<ModeTriple<S, D::Value, P>, Rest>;
    fn filter_inverse_step(self, s: S, d: D, p: P) -> Self::Output {
        Til(
            ModeTriple {
                shape: s,
                stride: d.value(),
                pps: p,
            },
            self,
        )
    }
}

/// Filters flattened `(shape, stride, pps)` triples to those active on codomain axis `TargetP`.
#[doc(hidden)]
pub const trait FilterInverseModes<TargetP: BasisPath, FD: FlatTuple, FP: FlatTuple>:
    FlatTuple
{
    type Output: FlatTuple;
    fn filter_inverse_modes(self, strides: FD, pps: FP) -> Self::Output;
}

const impl<TargetP: BasisPath> FilterInverseModes<TargetP, Nil, Nil> for Nil {
    type Output = Nil;
    fn filter_inverse_modes(self, _: Nil, _: Nil) -> Nil {
        Nil
    }
}

const impl<TargetP, S, TS, D, TD, P, TP> FilterInverseModes<TargetP, Til<D, TD>, Til<P, TP>>
    for Til<S, TS>
where
    TargetP: BasisPath,
    S: ShapeScalar,
    TS: FlatTuple + [const] FilterInverseModes<TargetP, TD, TP>,
    D: [const] StrideScalar,
    TD: FlatTuple,
    P: ShapeScalar,
    TP: FlatTuple,
    D::Path: ComparePath<TargetP>,
    <TS as FilterInverseModes<TargetP, TD, TP>>::Output:
        [const] FilterInverseStep<S, D, P, { INVERSE_AXIS_FILTER_ACTION::<S, D, TargetP> }>,
{
    type Output = <<TS as FilterInverseModes<TargetP, TD, TP>>::Output as FilterInverseStep<
        S,
        D,
        P,
        { INVERSE_AXIS_FILTER_ACTION::<S, D, TargetP> },
    >>::Output;

    fn filter_inverse_modes(self, strides: Til<D, TD>, pps: Til<P, TP>) -> Self::Output {
        let rest = self.1.filter_inverse_modes(strides.1, pps.1);
        rest.filter_inverse_step(self.0, strides.0, pps.0)
    }
}

const fn inverse_mode_precedes<DX: BasisCoeff, DY: BasisCoeff>() -> bool {
    match (DX::STATIC_VALUE, DY::STATIC_VALUE) {
        (Some(x), Some(y)) => x <= y,
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (None, None) => true,
    }
}

const INVERSE_MODE_PRECEDES<DX: BasisCoeff, DY: BasisCoeff>: bool =
    inverse_mode_precedes::<DX, DY>();

/// Single insertion-sort step placing `ModeTriple<SX, DX, PX>` before (`PRECEDES == true`)
/// or after (`false`) the head of sorted suffix `Self`.
#[doc(hidden)]
pub const trait InsertSortedTripleStep<SX, DX, PX, const PRECEDES: bool>: FlatTuple {
    type Output: FlatTuple;
    fn insert_sorted_triple_step(self, mode: ModeTriple<SX, DX, PX>) -> Self::Output;
}

/// Inserts `ModeTriple<SX, DX, PX>` into sorted [`ModeTriple`] tuple `Self` by ascending static stride.
#[doc(hidden)]
pub const trait InsertSortedTriple<SX: ShapeScalar, DX: BasisCoeff, PX: ShapeScalar>:
    FlatTuple
{
    type Output: FlatTuple;
    fn insert_sorted_triple(self, mode: ModeTriple<SX, DX, PX>) -> Self::Output;
}

const impl<SX: ShapeScalar, DX: BasisCoeff, PX: ShapeScalar> InsertSortedTriple<SX, DX, PX>
    for Nil
{
    type Output = Til<ModeTriple<SX, DX, PX>, Nil>;
    fn insert_sorted_triple(self, mode: ModeTriple<SX, DX, PX>) -> Self::Output {
        Til(mode, Nil)
    }
}

const impl<SX, DX, PX, SY, DY, PY, Rest> InsertSortedTripleStep<SX, DX, PX, true>
    for Til<ModeTriple<SY, DY, PY>, Rest>
where
    SX: ShapeScalar,
    DX: BasisCoeff,
    PX: ShapeScalar,
    SY: ShapeScalar,
    DY: BasisCoeff,
    PY: ShapeScalar,
    Rest: FlatTuple,
{
    type Output = Til<ModeTriple<SX, DX, PX>, Self>;
    fn insert_sorted_triple_step(self, mode: ModeTriple<SX, DX, PX>) -> Self::Output {
        Til(mode, self)
    }
}

const impl<SX, DX, PX, SY, DY, PY, Rest> InsertSortedTripleStep<SX, DX, PX, false>
    for Til<ModeTriple<SY, DY, PY>, Rest>
where
    SX: ShapeScalar,
    DX: BasisCoeff,
    PX: ShapeScalar,
    SY: ShapeScalar,
    DY: BasisCoeff,
    PY: ShapeScalar,
    Rest: FlatTuple + [const] InsertSortedTriple<SX, DX, PX>,
{
    type Output = Til<ModeTriple<SY, DY, PY>, <Rest as InsertSortedTriple<SX, DX, PX>>::Output>;
    fn insert_sorted_triple_step(self, mode: ModeTriple<SX, DX, PX>) -> Self::Output {
        Til(self.0, self.1.insert_sorted_triple(mode))
    }
}

const impl<SX, DX, PX, SY, DY, PY, Rest> InsertSortedTriple<SX, DX, PX>
    for Til<ModeTriple<SY, DY, PY>, Rest>
where
    SX: ShapeScalar,
    DX: BasisCoeff,
    PX: ShapeScalar,
    SY: ShapeScalar,
    DY: BasisCoeff,
    PY: ShapeScalar,
    Rest: FlatTuple,
    Self: [const] InsertSortedTripleStep<SX, DX, PX, { INVERSE_MODE_PRECEDES::<DX, DY> }>,
{
    type Output =
        <Self as InsertSortedTripleStep<SX, DX, PX, { INVERSE_MODE_PRECEDES::<DX, DY> }>>::Output;
    fn insert_sorted_triple(self, mode: ModeTriple<SX, DX, PX>) -> Self::Output {
        self.insert_sorted_triple_step(mode)
    }
}

/// Stably sorts a flat list of [`ModeTriple`]s by ascending static stride,
/// placing dynamic strides after static ones in their original left-to-right order.
#[doc(hidden)]
pub const trait SortInverseModes: FlatTuple {
    type Output: FlatTuple;
    fn sort_inverse_modes(self) -> Self::Output;
}

const impl SortInverseModes for Nil {
    type Output = Nil;
    fn sort_inverse_modes(self) -> Nil {
        Nil
    }
}

const impl<SH, DH, PH, T> SortInverseModes for Til<ModeTriple<SH, DH, PH>, T>
where
    SH: ShapeScalar,
    DH: BasisCoeff,
    PH: ShapeScalar,
    T: FlatTuple + [const] SortInverseModes,
    T::Output: [const] InsertSortedTriple<SH, DH, PH>,
{
    type Output = <T::Output as InsertSortedTriple<SH, DH, PH>>::Output;
    fn sort_inverse_modes(self) -> Self::Output {
        self.1.sort_inverse_modes().insert_sorted_triple(self.0)
    }
}

/// Algebraic classification of active strides collected along a single codomain axis.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum AxisAlgebra {
    /// Empty axis or all active strides are integers.
    Integer,
    /// All active strides on this axis are `F2`.
    F2,
    /// Mixed `Integer`/`F2` strides on the same axis or unsupported custom ring.
    Invalid,
}

const fn algebra_of_coeff(alg: CoeffAlgebra) -> AxisAlgebra {
    match alg {
        CoeffAlgebra::Integer => AxisAlgebra::Integer,
        CoeffAlgebra::F2 => AxisAlgebra::F2,
        CoeffAlgebra::Other => AxisAlgebra::Invalid,
    }
}

const fn combine_axis_algebra(head: AxisAlgebra, tail: AxisAlgebra) -> AxisAlgebra {
    match (head, tail) {
        (AxisAlgebra::Integer, AxisAlgebra::Integer) => AxisAlgebra::Integer,
        (AxisAlgebra::F2, AxisAlgebra::F2) => AxisAlgebra::F2,
        _ => AxisAlgebra::Invalid,
    }
}

/// Determines the uniform [`AxisAlgebra`] of a sorted flat list of [`ModeTriple`]s.
#[doc(hidden)]
pub const trait AxisAlgebraOf: FlatTuple {
    const ALGEBRA: AxisAlgebra;
}

const impl AxisAlgebraOf for Nil {
    const ALGEBRA: AxisAlgebra = AxisAlgebra::Integer;
}

const impl<S: Item, D: BasisCoeff, P: Item> AxisAlgebraOf for Til<ModeTriple<S, D, P>, Nil> {
    const ALGEBRA: AxisAlgebra = algebra_of_coeff(D::ALGEBRA);
}

const impl<S: Item, D: BasisCoeff, P: Item, H2, T2> AxisAlgebraOf
    for Til<ModeTriple<S, D, P>, Til<H2, T2>>
where
    Til<H2, T2>: AxisAlgebraOf,
{
    const ALGEBRA: AxisAlgebra = combine_axis_algebra(
        algebra_of_coeff(D::ALGEBRA),
        <Til<H2, T2> as AxisAlgebraOf>::ALGEBRA,
    );
}

const AXIS_ALGEBRA_OF<M: AxisAlgebraOf>: AxisAlgebra = M::ALGEBRA;

/// Prepends one inverse `(S, P)` mode onto an empty, scalar, or tuple flat suffix layout.
#[doc(hidden)]
pub const trait PrependInverseMode<S: ShapeScalar, P: StrideScalar>: Copy {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn prepend_inverse_mode(self, s: S, p: P) -> Layout<Self::Shape, Self::Stride>;
}

const impl<S, P> PrependInverseMode<S, P> for Layout<Nil, Nil>
where
    S: ShapeScalar,
    P: StrideScalar + Stride<S>,
{
    type Shape = S;
    type Stride = P;

    fn prepend_inverse_mode(self, s: S, p: P) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(s, p)
    }
}

const impl<ST, DT, S, P> PrependInverseMode<S, P> for Layout<ST, DT>
where
    ST: ShapeScalar,
    DT: StrideScalar + Stride<ST>,
    S: ShapeScalar,
    P: StrideScalar + Stride<S>,
{
    type Shape = Til<S, Til<ST, Nil>>;
    type Stride = Til<P, Til<DT, Nil>>;

    fn prepend_inverse_mode(self, s: S, p: P) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(Til(s, Til(self.shape, Nil)), Til(p, Til(self.stride, Nil)))
    }
}

const impl<HS, TS, HD, TD, S, P> PrependInverseMode<S, P> for Layout<Til<HS, TS>, Til<HD, TD>>
where
    HS: ShapeScalar,
    TS: Tuple + Shape,
    HD: StrideScalar + Stride<HS>,
    TD: Tuple + Stride<TS>,
    S: ShapeScalar,
    P: StrideScalar + Stride<S>,
{
    type Shape = Til<S, Til<HS, TS>>;
    type Stride = Til<P, Til<HD, TD>>;

    fn prepend_inverse_mode(self, s: S, p: P) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(Til(s, self.shape), Til(p, self.stride))
    }
}

/// Running covered extent state for [`RightInverseFold`].
///
/// Tracks either a compile-time `Int<N>`, a runtime `DynamicExtent(Option<int>)`,
/// or `ExhaustedExtent` (when a prior static mode's product exceeded `int::MAX`, so no
/// subsequent `int` stride can match).
#[doc(hidden)]
pub const trait RightRunningExtent: Copy {
    const STATIC_EXTENT: Option<int>;
    const IS_EXHAUSTED: bool;
    fn dynamic_extent(self) -> Option<int>;
}

/// Static marker indicating covered extent exceeded `int::MAX`.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExhaustedExtent;

/// Runtime running covered extent (`None` once `curr_d * s` exceeds `int::MAX`).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynamicExtent(pub Option<int>);

impl DynamicExtent {
    const fn advance(current: Option<int>, extent: int) -> Self {
        match current {
            Some(c) => Self(mul_maybe_overflows(c, extent)),
            None => Self(None),
        }
    }

    const fn advance_f2(current: Option<int>, extent: int) -> Self {
        match current {
            Some(c) => Self(clmul_maybe_overflows(c, extent)),
            None => Self(None),
        }
    }
}

const impl<const N: int> RightRunningExtent for Int<N> {
    const STATIC_EXTENT: Option<int> = Some(N);
    const IS_EXHAUSTED: bool = false;
    fn dynamic_extent(self) -> Option<int> {
        Some(N)
    }
}

const impl RightRunningExtent for ExhaustedExtent {
    const STATIC_EXTENT: Option<int> = None;
    const IS_EXHAUSTED: bool = true;
    fn dynamic_extent(self) -> Option<int> {
        None
    }
}

const impl RightRunningExtent for DynamicExtent {
    const STATIC_EXTENT: Option<int> = None;
    const IS_EXHAUSTED: bool = false;
    fn dynamic_extent(self) -> Option<int> {
        self.0
    }
}

/// Compile-time action for one mode in the sorted 1D right-inverse chain walk.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum RightFoldAction {
    /// Mode statically cannot continue the chain (mismatched/negative static stride or exhausted).
    StaticSkip,
    /// Static `D == CurrD`, static `S`, and `CurrD * S` fits in `int`.
    StaticMatchStaticNext,
    /// Static `D == CurrD`, static `S`, and `CurrD * S` exceeds `int::MAX`.
    StaticMatchExhausted,
    /// Static `D == CurrD` with dynamic `S`.
    StaticMatchDynamicNext,
    /// At least one of `D` or `CurrD` is dynamic: match `d == curr_d` at runtime.
    DynamicStep,
}

const fn right_fold_action<S: ShapeScalar, D: Integer, CurrD: RightRunningExtent>()
-> RightFoldAction {
    if CurrD::IS_EXHAUSTED {
        return RightFoldAction::StaticSkip;
    }
    if let Some(d) = D::STATIC_VALUE
        && d < 0
    {
        return RightFoldAction::StaticSkip;
    }
    match (D::STATIC_VALUE, CurrD::STATIC_EXTENT) {
        (Some(d), Some(c)) => {
            if d != c {
                return RightFoldAction::StaticSkip;
            }
            match S::STATIC_VALUE {
                Some(s) => {
                    if mul_maybe_overflows(c, s).is_some() {
                        RightFoldAction::StaticMatchStaticNext
                    } else {
                        RightFoldAction::StaticMatchExhausted
                    }
                }
                None => RightFoldAction::StaticMatchDynamicNext,
            }
        }
        _ => RightFoldAction::DynamicStep,
    }
}

const RIGHT_FOLD_ACTION<S: ShapeScalar, D: Integer, CurrD: RightRunningExtent>: RightFoldAction =
    right_fold_action::<S, D, CurrD>();

const STATIC_RIGHT_NEXT_D<CurrD: RightRunningExtent, S: ShapeScalar>: int = mul(
    CurrD::STATIC_EXTENT.expect("expected static running extent in right_inverse"),
    S::STATIC_VALUE.expect("expected static shape in right_inverse"),
);

/// Executes one [`RightFoldAction`] step of [`RightInverseFold`] against running extent `CurrD`.
#[doc(hidden)]
pub const trait RightInverseFoldStep<CurrD: RightRunningExtent, const ACTION: RightFoldAction>:
    FlatTuple
{
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn right_inverse_fold_step(self, curr_d: CurrD) -> Layout<Self::Shape, Self::Stride>;
}

/// Folds a sorted flat list of [`ModeTriple`]s along one codomain axis into the
/// pre-coalesced 1D right inverse layout (or `Layout<Nil, Nil>` if no mode matched).
#[doc(hidden)]
pub const trait RightInverseFold<CurrD: RightRunningExtent>: FlatTuple {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn right_inverse_fold(self, curr_d: CurrD) -> Layout<Self::Shape, Self::Stride>;
}

const impl<CurrD: RightRunningExtent> RightInverseFold<CurrD> for Nil {
    type Shape = Nil;
    type Stride = Nil;

    fn right_inverse_fold(self, _: CurrD) -> Layout<Nil, Nil> {
        Layout::new(Nil, Nil)
    }
}

const impl<S, D, P, Rest, CurrD> RightInverseFold<CurrD> for Til<ModeTriple<S, D, P>, Rest>
where
    S: ShapeScalar,
    D: Integer,
    P: ShapeScalar,
    Rest: FlatTuple,
    CurrD: RightRunningExtent,
    Self: [const] RightInverseFoldStep<CurrD, { RIGHT_FOLD_ACTION::<S, D, CurrD> }>,
{
    type Shape = <Self as RightInverseFoldStep<CurrD, { RIGHT_FOLD_ACTION::<S, D, CurrD> }>>::Shape;
    type Stride =
        <Self as RightInverseFoldStep<CurrD, { RIGHT_FOLD_ACTION::<S, D, CurrD> }>>::Stride;

    fn right_inverse_fold(self, curr_d: CurrD) -> Layout<Self::Shape, Self::Stride> {
        self.right_inverse_fold_step(curr_d)
    }
}

const impl<S, D, P, Rest, CurrD> RightInverseFoldStep<CurrD, { RightFoldAction::StaticSkip }>
    for Til<ModeTriple<S, D, P>, Rest>
where
    S: ShapeScalar,
    D: Integer,
    P: ShapeScalar,
    CurrD: RightRunningExtent,
    Rest: FlatTuple + [const] RightInverseFold<CurrD>,
{
    type Shape = <Rest as RightInverseFold<CurrD>>::Shape;
    type Stride = <Rest as RightInverseFold<CurrD>>::Stride;

    fn right_inverse_fold_step(self, curr_d: CurrD) -> Layout<Self::Shape, Self::Stride> {
        self.1.right_inverse_fold(curr_d)
    }
}

const impl<S, D, P, Rest, CurrD>
    RightInverseFoldStep<CurrD, { RightFoldAction::StaticMatchStaticNext }>
    for Til<ModeTriple<S, D, P>, Rest>
where
    S: ShapeScalar,
    D: Integer,
    P: ShapeScalar + StrideScalar + Stride<S>,
    CurrD: RightRunningExtent,
    Rest: FlatTuple + [const] RightInverseFold<Int<{ STATIC_RIGHT_NEXT_D::<CurrD, S> }>>,
    Layout<
        <Rest as RightInverseFold<Int<{ STATIC_RIGHT_NEXT_D::<CurrD, S> }>>>::Shape,
        <Rest as RightInverseFold<Int<{ STATIC_RIGHT_NEXT_D::<CurrD, S> }>>>::Stride,
    >: [const] PrependInverseMode<S, P>,
{
    type Shape = <Layout<
        <Rest as RightInverseFold<Int<{ STATIC_RIGHT_NEXT_D::<CurrD, S> }>>>::Shape,
        <Rest as RightInverseFold<Int<{ STATIC_RIGHT_NEXT_D::<CurrD, S> }>>>::Stride,
    > as PrependInverseMode<S, P>>::Shape;
    type Stride = <Layout<
        <Rest as RightInverseFold<Int<{ STATIC_RIGHT_NEXT_D::<CurrD, S> }>>>::Shape,
        <Rest as RightInverseFold<Int<{ STATIC_RIGHT_NEXT_D::<CurrD, S> }>>>::Stride,
    > as PrependInverseMode<S, P>>::Stride;

    fn right_inverse_fold_step(self, _: CurrD) -> Layout<Self::Shape, Self::Stride> {
        let tail = self.1.right_inverse_fold(Int);
        tail.prepend_inverse_mode(self.0.shape, self.0.pps)
    }
}

const impl<S, D, P, Rest, CurrD>
    RightInverseFoldStep<CurrD, { RightFoldAction::StaticMatchExhausted }>
    for Til<ModeTriple<S, D, P>, Rest>
where
    S: ShapeScalar,
    D: Integer,
    P: ShapeScalar + StrideScalar + Stride<S>,
    CurrD: RightRunningExtent,
    Rest: FlatTuple + [const] RightInverseFold<ExhaustedExtent>,
    Layout<
        <Rest as RightInverseFold<ExhaustedExtent>>::Shape,
        <Rest as RightInverseFold<ExhaustedExtent>>::Stride,
    >: [const] PrependInverseMode<S, P>,
{
    type Shape = <Layout<
        <Rest as RightInverseFold<ExhaustedExtent>>::Shape,
        <Rest as RightInverseFold<ExhaustedExtent>>::Stride,
    > as PrependInverseMode<S, P>>::Shape;
    type Stride = <Layout<
        <Rest as RightInverseFold<ExhaustedExtent>>::Shape,
        <Rest as RightInverseFold<ExhaustedExtent>>::Stride,
    > as PrependInverseMode<S, P>>::Stride;

    fn right_inverse_fold_step(self, _: CurrD) -> Layout<Self::Shape, Self::Stride> {
        let tail = self.1.right_inverse_fold(ExhaustedExtent);
        tail.prepend_inverse_mode(self.0.shape, self.0.pps)
    }
}

const impl<S, D, P, Rest, CurrD>
    RightInverseFoldStep<CurrD, { RightFoldAction::StaticMatchDynamicNext }>
    for Til<ModeTriple<S, D, P>, Rest>
where
    S: [const] ShapeScalar,
    D: Integer,
    P: ShapeScalar + StrideScalar + Stride<S>,
    CurrD: [const] RightRunningExtent,
    Rest: FlatTuple + [const] RightInverseFold<DynamicExtent>,
    Layout<
        <Rest as RightInverseFold<DynamicExtent>>::Shape,
        <Rest as RightInverseFold<DynamicExtent>>::Stride,
    >: [const] PrependInverseMode<S, P>,
{
    type Shape = <Layout<
        <Rest as RightInverseFold<DynamicExtent>>::Shape,
        <Rest as RightInverseFold<DynamicExtent>>::Stride,
    > as PrependInverseMode<S, P>>::Shape;
    type Stride = <Layout<
        <Rest as RightInverseFold<DynamicExtent>>::Shape,
        <Rest as RightInverseFold<DynamicExtent>>::Stride,
    > as PrependInverseMode<S, P>>::Stride;

    fn right_inverse_fold_step(self, curr_d: CurrD) -> Layout<Self::Shape, Self::Stride> {
        let next_curr = DynamicExtent::advance(curr_d.dynamic_extent(), self.0.shape.value());
        let tail = self.1.right_inverse_fold(next_curr);
        tail.prepend_inverse_mode(self.0.shape, self.0.pps)
    }
}

const impl<S, D, P, Rest, CurrD> RightInverseFoldStep<CurrD, { RightFoldAction::DynamicStep }>
    for Til<ModeTriple<S, D, P>, Rest>
where
    S: [const] ShapeScalar,
    D: [const] Integer,
    P: [const] ShapeScalar,
    CurrD: [const] RightRunningExtent,
    Rest: FlatTuple + [const] RightInverseFold<DynamicExtent>,
    Layout<
        <Rest as RightInverseFold<DynamicExtent>>::Shape,
        <Rest as RightInverseFold<DynamicExtent>>::Stride,
    >: [const] PrependInverseMode<int, int>,
{
    type Shape = <Layout<
        <Rest as RightInverseFold<DynamicExtent>>::Shape,
        <Rest as RightInverseFold<DynamicExtent>>::Stride,
    > as PrependInverseMode<int, int>>::Shape;
    type Stride = <Layout<
        <Rest as RightInverseFold<DynamicExtent>>::Shape,
        <Rest as RightInverseFold<DynamicExtent>>::Stride,
    > as PrependInverseMode<int, int>>::Stride;

    fn right_inverse_fold_step(self, curr_d: CurrD) -> Layout<Self::Shape, Self::Stride> {
        let s = self.0.shape.value();
        let curr_opt = curr_d.dynamic_extent();
        let matched = match curr_opt {
            Some(c) => eq(self.0.stride, c) && s > 1,
            None => false,
        };
        if matched {
            let next_curr = DynamicExtent::advance(curr_opt, s);
            let tail = self.1.right_inverse_fold(next_curr);
            tail.prepend_inverse_mode(s, self.0.pps.value())
        } else {
            let tail = self.1.right_inverse_fold(DynamicExtent(curr_opt));
            tail.prepend_inverse_mode(1, 0)
        }
    }
}

/// Normalizes a 1D right-inverse axis: empty becomes `1:0`, and multi-mode layouts coalesce
/// with trailing unit-mode elimination (matching PyCuTe's `Layout._coalesce`).
#[doc(hidden)]
pub const trait FinalizeRightAxis: Copy {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn finalize_right_axis(self) -> Layout<Self::Shape, Self::Stride>;
}

const impl FinalizeRightAxis for Layout<Nil, Nil> {
    type Shape = Int<1>;
    type Stride = Int<0>;

    fn finalize_right_axis(self) -> Layout<Int<1>, Int<0>> {
        Layout::new(Int, Int)
    }
}

const impl<S, D> FinalizeRightAxis for Layout<S, D>
where
    S: ShapeScalar,
    D: StrideScalar + Stride<S>,
{
    type Shape = S;
    type Stride = D;

    fn finalize_right_axis(self) -> Layout<S, D> {
        self
    }
}

const impl<HS, TS, HD, TD> FinalizeRightAxis for Layout<Til<HS, TS>, Til<HD, TD>>
where
    HS: ShapeScalar,
    TS: Tuple + Shape,
    HD: StrideScalar + Stride<HS>,
    TD: Tuple + Stride<TS>,
    Self: [const] Coalesce<true>,
    <Self as Coalesce<true>>::Output: [const] DropTrailingUnit,
{
    type Shape = <<Self as Coalesce<true>>::Output as DropTrailingUnit>::Shape;
    type Stride = <<Self as Coalesce<true>>::Output as DropTrailingUnit>::Stride;

    fn finalize_right_axis(self) -> Layout<Self::Shape, Self::Stride> {
        let coalesced = Coalesce::<true>::coalesce(self);
        coalesced.drop_trailing_unit()
    }
}

/// Compile-time action for one mode in the sorted 1D left-inverse chain fold.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum LeftFoldAction {
    /// Static `D`, `PrevD`, `PrevS`, valid ordered chain with `gap == 1` (omit leading `1:0`).
    StaticSkipGap,
    /// Static `D`, `PrevD`, `PrevS`, valid ordered chain with `gap > 1` (emit `(Int<gap>, PrevP)`).
    StaticEmitGap,
    /// Static positive `D` and static `S >= 2`, dynamic `PrevD` or `PrevS`: validate at runtime,
    /// recovering static `(D, S, P)`.
    DynamicNoSkip,
    /// Dynamic `D` or `S`: skip runtime `d == 0 || s == 1` via a neutral size-1 mode, else validate.
    DynamicMaySkip,
    /// Statically invalid (`D <= 0`, `PrevD` does not divide `D`, or `D / PrevD < PrevS`).
    Invalid,
}

const fn left_fold_action<S: ShapeScalar, D: Integer, PrevD: ShapeScalar, PrevS: ShapeScalar>()
-> LeftFoldAction {
    if matches!(S::STATIC_VALUE, Some(0))
        || matches!(PrevD::STATIC_VALUE, Some(0))
        || matches!(PrevS::STATIC_VALUE, Some(0))
    {
        return LeftFoldAction::Invalid;
    }
    if let Some(d) = D::STATIC_VALUE
        && d <= 0
    {
        return LeftFoldAction::Invalid;
    }

    if let (Some(d), Some(pd)) = (D::STATIC_VALUE, PrevD::STATIC_VALUE) {
        if rem(d, pd) != 0 {
            return LeftFoldAction::Invalid;
        }
        let gap = div(d, pd);
        if let Some(ps) = PrevS::STATIC_VALUE {
            if gap < ps {
                return LeftFoldAction::Invalid;
            }
            return if gap == 1 {
                LeftFoldAction::StaticSkipGap
            } else {
                LeftFoldAction::StaticEmitGap
            };
        }
        return if S::STATIC_VALUE.is_some() {
            LeftFoldAction::DynamicNoSkip
        } else {
            LeftFoldAction::DynamicMaySkip
        };
    }

    if D::STATIC_VALUE.is_some() && S::STATIC_VALUE.is_some() {
        LeftFoldAction::DynamicNoSkip
    } else {
        LeftFoldAction::DynamicMaySkip
    }
}

const LEFT_FOLD_ACTION<
    S: ShapeScalar,
    D: Integer,
    PrevD: ShapeScalar,
    PrevS: ShapeScalar,
>: LeftFoldAction = left_fold_action::<S, D, PrevD, PrevS>();

const STATIC_LEFT_GAP<D: Integer, PrevD: ShapeScalar>: int = div(
    D::STATIC_VALUE.expect("expected static stride in left_inverse"),
    PrevD::STATIC_VALUE.expect("expected static previous stride in left_inverse"),
);

const STATIC_LEFT_NEXT_D<D: Integer>: int =
    D::STATIC_VALUE.expect("expected static stride in left_inverse");

/// Executes one [`LeftFoldAction`] step of [`LeftInverseFold`] carrying pending `(PrevD, PrevS, PrevP)`.
#[doc(hidden)]
pub const trait LeftInverseFoldStep<
    PrevD: ShapeScalar,
    PrevS: ShapeScalar,
    PrevP: ShapeScalar,
    const ACTION: LeftFoldAction,
>: FlatTuple
{
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn left_inverse_fold_step(
        self,
        prev_d: PrevD,
        prev_s: PrevS,
        prev_p: PrevP,
    ) -> Layout<Self::Shape, Self::Stride>;
}

/// Folds a sorted flat list of [`ModeTriple`]s along one codomain axis into the
/// pre-coalesced 1D left inverse layout, carrying the previous pending mode `(prev_d, prev_s, prev_p)`.
#[doc(hidden)]
pub const trait LeftInverseFold<PrevD: ShapeScalar, PrevS: ShapeScalar, PrevP: ShapeScalar>:
    FlatTuple
{
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn left_inverse_fold(
        self,
        prev_d: PrevD,
        prev_s: PrevS,
        prev_p: PrevP,
    ) -> Layout<Self::Shape, Self::Stride>;
}

const impl<PrevD, PrevS, PrevP> LeftInverseFold<PrevD, PrevS, PrevP> for Nil
where
    PrevD: ShapeScalar,
    PrevS: ShapeScalar,
    PrevP: ShapeScalar + StrideScalar + Stride<PrevS>,
{
    type Shape = PrevS;
    type Stride = PrevP;

    fn left_inverse_fold(self, _: PrevD, prev_s: PrevS, prev_p: PrevP) -> Layout<PrevS, PrevP> {
        Layout::new(prev_s, prev_p)
    }
}

const impl<S, D, P, Rest, PrevD, PrevS, PrevP> LeftInverseFold<PrevD, PrevS, PrevP>
    for Til<ModeTriple<S, D, P>, Rest>
where
    S: ShapeScalar,
    D: Integer,
    P: ShapeScalar,
    Rest: FlatTuple,
    PrevD: ShapeScalar,
    PrevS: ShapeScalar,
    PrevP: ShapeScalar,
    Self: [const] LeftInverseFoldStep<PrevD, PrevS, PrevP, { LEFT_FOLD_ACTION::<S, D, PrevD, PrevS> }>,
{
    type Shape = <Self as LeftInverseFoldStep<
        PrevD,
        PrevS,
        PrevP,
        { LEFT_FOLD_ACTION::<S, D, PrevD, PrevS> },
    >>::Shape;
    type Stride = <Self as LeftInverseFoldStep<
        PrevD,
        PrevS,
        PrevP,
        { LEFT_FOLD_ACTION::<S, D, PrevD, PrevS> },
    >>::Stride;

    fn left_inverse_fold(
        self,
        prev_d: PrevD,
        prev_s: PrevS,
        prev_p: PrevP,
    ) -> Layout<Self::Shape, Self::Stride> {
        self.left_inverse_fold_step(prev_d, prev_s, prev_p)
    }
}

const impl<S, D, P, Rest, PrevD, PrevS, PrevP>
    LeftInverseFoldStep<PrevD, PrevS, PrevP, { LeftFoldAction::StaticSkipGap }>
    for Til<ModeTriple<S, D, P>, Rest>
where
    S: ShapeScalar,
    D: Integer,
    P: ShapeScalar,
    PrevD: ShapeScalar,
    PrevS: ShapeScalar,
    PrevP: ShapeScalar,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_LEFT_NEXT_D::<D> }> }>,
    Rest: FlatTuple + [const] LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>,
{
    type Shape = <Rest as LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>>::Shape;
    type Stride = <Rest as LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>>::Stride;

    fn left_inverse_fold_step(
        self,
        _: PrevD,
        _: PrevS,
        _: PrevP,
    ) -> Layout<Self::Shape, Self::Stride> {
        self.1.left_inverse_fold(Int, self.0.shape, self.0.pps)
    }
}

const impl<S, D, P, Rest, PrevD, PrevS, PrevP>
    LeftInverseFoldStep<PrevD, PrevS, PrevP, { LeftFoldAction::StaticEmitGap }>
    for Til<ModeTriple<S, D, P>, Rest>
where
    S: ShapeScalar,
    D: Integer,
    P: ShapeScalar,
    PrevD: ShapeScalar,
    PrevS: ShapeScalar,
    PrevP: ShapeScalar + StrideScalar,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_LEFT_NEXT_D::<D> }> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_LEFT_GAP::<D, PrevD> }> }>,
    Rest: FlatTuple + [const] LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>,
    Layout<
        <Rest as LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>>::Shape,
        <Rest as LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>>::Stride,
    >: [const] PrependInverseMode<Int<{ STATIC_LEFT_GAP::<D, PrevD> }>, PrevP>,
{
    type Shape = <Layout<
        <Rest as LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>>::Shape,
        <Rest as LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>>::Stride,
    > as PrependInverseMode<Int<{ STATIC_LEFT_GAP::<D, PrevD> }>, PrevP>>::Shape;
    type Stride = <Layout<
        <Rest as LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>>::Shape,
        <Rest as LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>>::Stride,
    > as PrependInverseMode<Int<{ STATIC_LEFT_GAP::<D, PrevD> }>, PrevP>>::Stride;

    fn left_inverse_fold_step(
        self,
        _: PrevD,
        _: PrevS,
        prev_p: PrevP,
    ) -> Layout<Self::Shape, Self::Stride> {
        let tail = self.1.left_inverse_fold(Int, self.0.shape, self.0.pps);
        tail.prepend_inverse_mode(Int, prev_p)
    }
}

const impl<S, D, P, Rest, PrevD, PrevS, PrevP>
    LeftInverseFoldStep<PrevD, PrevS, PrevP, { LeftFoldAction::DynamicNoSkip }>
    for Til<ModeTriple<S, D, P>, Rest>
where
    S: ShapeScalar,
    D: [const] Integer,
    P: ShapeScalar,
    PrevD: [const] ShapeScalar,
    PrevS: [const] ShapeScalar,
    PrevP: ShapeScalar + StrideScalar,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_LEFT_NEXT_D::<D> }> }>,
    Rest: FlatTuple + [const] LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>,
    Layout<
        <Rest as LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>>::Shape,
        <Rest as LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>>::Stride,
    >: [const] PrependInverseMode<int, PrevP>,
{
    type Shape = <Layout<
        <Rest as LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>>::Shape,
        <Rest as LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>>::Stride,
    > as PrependInverseMode<int, PrevP>>::Shape;
    type Stride = <Layout<
        <Rest as LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>>::Shape,
        <Rest as LeftInverseFold<Int<{ STATIC_LEFT_NEXT_D::<D> }>, S, P>>::Stride,
    > as PrependInverseMode<int, PrevP>>::Stride;

    fn left_inverse_fold_step(
        self,
        prev_d: PrevD,
        prev_s: PrevS,
        prev_p: PrevP,
    ) -> Layout<Self::Shape, Self::Stride> {
        let d = self.0.stride.value();
        let pd = prev_d.value();
        assert!(rem(d, pd) == 0, "left_inverse requires divisible strides");
        let gap = div(d, pd);
        assert!(
            gap >= prev_s.value(),
            "left_inverse requires non-overlapping modes"
        );
        let tail = self.1.left_inverse_fold(Int, self.0.shape, self.0.pps);
        tail.prepend_inverse_mode(gap, prev_p)
    }
}

const impl<S, D, P, Rest, PrevD, PrevS, PrevP>
    LeftInverseFoldStep<PrevD, PrevS, PrevP, { LeftFoldAction::DynamicMaySkip }>
    for Til<ModeTriple<S, D, P>, Rest>
where
    S: [const] ShapeScalar,
    D: [const] Integer,
    P: [const] ShapeScalar,
    PrevD: [const] ShapeScalar,
    PrevS: [const] ShapeScalar,
    PrevP: [const] ShapeScalar + StrideScalar,
    Rest: FlatTuple + [const] LeftInverseFold<int, int, int>,
    Layout<
        <Rest as LeftInverseFold<int, int, int>>::Shape,
        <Rest as LeftInverseFold<int, int, int>>::Stride,
    >: [const] PrependInverseMode<int, PrevP>,
{
    type Shape = <Layout<
        <Rest as LeftInverseFold<int, int, int>>::Shape,
        <Rest as LeftInverseFold<int, int, int>>::Stride,
    > as PrependInverseMode<int, PrevP>>::Shape;
    type Stride = <Layout<
        <Rest as LeftInverseFold<int, int, int>>::Shape,
        <Rest as LeftInverseFold<int, int, int>>::Stride,
    > as PrependInverseMode<int, PrevP>>::Stride;

    fn left_inverse_fold_step(
        self,
        prev_d: PrevD,
        prev_s: PrevS,
        prev_p: PrevP,
    ) -> Layout<Self::Shape, Self::Stride> {
        let s = self.0.shape.value();
        let d = self.0.stride.value();
        let pd = prev_d.value();
        let ps = prev_s.value();
        let pp = Integer::value(prev_p);

        // Runtime size-1 or stride-0 modes carry no positional information and are skipped
        // without altering the running chain state.
        if s == 1 || d == 0 {
            let tail = self.1.left_inverse_fold(pd, ps, pp);
            return tail.prepend_inverse_mode(1, prev_p);
        }

        assert!(d > 0, "left_inverse requires non-negative strides");
        assert!(rem(d, pd) == 0, "left_inverse requires divisible strides");
        let gap = div(d, pd);
        assert!(gap >= ps, "left_inverse requires non-overlapping modes");

        let tail = self.1.left_inverse_fold(d, s, self.0.pps.value());
        tail.prepend_inverse_mode(gap, prev_p)
    }
}

/// Coalesces a 1D left-inverse axis with extended-domain semantics (`Coalesce<true>`,
/// matching PyCuTe's `._coalesce_z(coprof)`).
#[doc(hidden)]
pub const trait FinalizeLeftAxis: Copy {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn finalize_left_axis(self) -> Layout<Self::Shape, Self::Stride>;
}

const impl<S, D> FinalizeLeftAxis for Layout<S, D>
where
    S: ShapeScalar,
    D: StrideScalar + Stride<S>,
{
    type Shape = S;
    type Stride = D;

    fn finalize_left_axis(self) -> Layout<S, D> {
        self
    }
}

const impl<HS, TS, HD, TD> FinalizeLeftAxis for Layout<Til<HS, TS>, Til<HD, TD>>
where
    HS: ShapeScalar,
    TS: Tuple + Shape,
    HD: StrideScalar + Stride<HS>,
    TD: Tuple + Stride<TS>,
    Self: [const] Coalesce<true>,
    <Self as Coalesce<true>>::Output: [const] LayoutBase,
{
    type Shape = <<Self as Coalesce<true>>::Output as LayoutBase>::Shape;
    type Stride = <<Self as Coalesce<true>>::Output as LayoutBase>::Stride;

    fn finalize_left_axis(self) -> Layout<Self::Shape, Self::Stride> {
        let coalesced = Coalesce::<true>::coalesce(self);
        Layout::new(coalesced.shape(), coalesced.stride())
    }
}

/// Selects the per-axis fold & finalization (`RIGHT = true` for right inverse, `false` for left)
/// dispatched by [`AxisAlgebra`].
#[doc(hidden)]
pub const trait InvertSortedAxis<
    const RIGHT: bool,
    const ALGEBRA: AxisAlgebra = { AxisAlgebra::Integer },
>: FlatTuple
{
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn invert_sorted_axis(self, domain_tail_size: Option<int>)
    -> Layout<Self::Shape, Self::Stride>;
}

const impl<M> InvertSortedAxis<true, { AxisAlgebra::Integer }> for M
where
    M: FlatTuple + [const] RightInverseFold<Int<1>>,
    Layout<<M as RightInverseFold<Int<1>>>::Shape, <M as RightInverseFold<Int<1>>>::Stride>:
        [const] FinalizeRightAxis,
{
    type Shape = <Layout<
        <M as RightInverseFold<Int<1>>>::Shape,
        <M as RightInverseFold<Int<1>>>::Stride,
    > as FinalizeRightAxis>::Shape;
    type Stride = <Layout<
        <M as RightInverseFold<Int<1>>>::Shape,
        <M as RightInverseFold<Int<1>>>::Stride,
    > as FinalizeRightAxis>::Stride;

    fn invert_sorted_axis(self, _: Option<int>) -> Layout<Self::Shape, Self::Stride> {
        self.right_inverse_fold(Int).finalize_right_axis()
    }
}

const impl<M> InvertSortedAxis<false, { AxisAlgebra::Integer }> for M
where
    M: FlatTuple + [const] LeftInverseFold<Int<1>, Int<1>, Int<0>>,
    Layout<
        <M as LeftInverseFold<Int<1>, Int<1>, Int<0>>>::Shape,
        <M as LeftInverseFold<Int<1>, Int<1>, Int<0>>>::Stride,
    >: [const] FinalizeLeftAxis,
{
    type Shape = <Layout<
        <M as LeftInverseFold<Int<1>, Int<1>, Int<0>>>::Shape,
        <M as LeftInverseFold<Int<1>, Int<1>, Int<0>>>::Stride,
    > as FinalizeLeftAxis>::Shape;
    type Stride = <Layout<
        <M as LeftInverseFold<Int<1>, Int<1>, Int<0>>>::Shape,
        <M as LeftInverseFold<Int<1>, Int<1>, Int<0>>>::Stride,
    > as FinalizeLeftAxis>::Stride;

    fn invert_sorted_axis(self, _: Option<int>) -> Layout<Self::Shape, Self::Stride> {
        let raw = self.left_inverse_fold(Int, Int, Int);
        raw.finalize_left_axis()
    }
}

// ------------------------------------------------------------------
// F2 axis inversion (characteristic-2 back-substitution for right_inverse,
// unit-gap validation for left_inverse)
// ------------------------------------------------------------------

/// Appends one inverse `(S, D)` mode onto an empty, scalar, or tuple flat prefix layout.
#[doc(hidden)]
pub const trait AppendInverseMode<S: ShapeScalar, D: StrideScalar>: Copy {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn append_inverse_mode(self, s: S, d: D) -> Layout<Self::Shape, Self::Stride>;
}

const impl<S, D> AppendInverseMode<S, D> for Layout<Nil, Nil>
where
    S: ShapeScalar,
    D: StrideScalar + Stride<S>,
{
    type Shape = S;
    type Stride = D;

    fn append_inverse_mode(self, s: S, d: D) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(s, d)
    }
}

const impl<ST, DT, S, D> AppendInverseMode<S, D> for Layout<ST, DT>
where
    ST: ShapeScalar,
    DT: StrideScalar + Stride<ST>,
    S: ShapeScalar,
    D: StrideScalar + Stride<S>,
{
    type Shape = Til<ST, Til<S, Nil>>;
    type Stride = Til<DT, Til<D, Nil>>;

    fn append_inverse_mode(self, s: S, d: D) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(Til(self.shape, Til(s, Nil)), Til(self.stride, Til(d, Nil)))
    }
}

const impl<HS, TS, HD, TD, S, D> AppendInverseMode<S, D> for Layout<Til<HS, TS>, Til<HD, TD>>
where
    HS: [const] ShapeScalar,
    TS: [const] Tuple + Shape,
    HD: [const] StrideScalar + Stride<HS>,
    TD: [const] Tuple + Stride<TS>,
    S: [const] ShapeScalar,
    D: [const] StrideScalar + Stride<S>,
    Til<HS, TS>: [const] Append<Til<S, Nil>>,
    Til<HD, TD>: [const] Append<Til<D, Nil>>,
    <Til<HS, TS> as Append<Til<S, Nil>>>::Output: Shape,
    <Til<HD, TD> as Append<Til<D, Nil>>>::Output:
        Stride<<Til<HS, TS> as Append<Til<S, Nil>>>::Output>,
{
    type Shape = <Til<HS, TS> as Append<Til<S, Nil>>>::Output;
    type Stride = <Til<HD, TD> as Append<Til<D, Nil>>>::Output;

    fn append_inverse_mode(self, s: S, d: D) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(
            self.shape.append(Til(s, Nil)),
            self.stride.append(Til(d, Nil)),
        )
    }
}

/// Evaluates an accumulated `F2` inverse prefix layout at a dynamic `F2<int>` residue,
/// returning `F2(0)` for the empty prefix `Layout<Nil, Nil>`.
#[doc(hidden)]
pub const trait F2BackSubDynEval: Copy {
    fn eval_f2_residue(&self, residue: F2<int>) -> F2<int>;
}

const impl F2BackSubDynEval for Layout<Nil, Nil> {
    fn eval_f2_residue(&self, _: F2<int>) -> F2<int> {
        f2(0i32)
    }
}

const impl<S, D, OutV> F2BackSubDynEval for Layout<S, D>
where
    S: ShapeScalar,
    D: StrideScalar + Stride<S>,
    F2<int>: [const] IndexToCoord<S>,
    <F2<int> as IndexToCoord<S>>::Output: [const] InnerProduct<D, Output = F2<OutV>>,
    OutV: [const] F2Scalar,
{
    fn eval_f2_residue(&self, residue: F2<int>) -> F2<int> {
        f2(self.call(residue).value())
    }
}

const impl<HS, TS, HD, TD, OutV> F2BackSubDynEval for Layout<Til<HS, TS>, Til<HD, TD>>
where
    HS: ShapeScalar,
    TS: Tuple + Shape,
    HD: StrideScalar + Stride<HS>,
    TD: Tuple + Stride<TS>,
    F2<int>: [const] IndexToCoord<Til<HS, TS>>,
    <F2<int> as IndexToCoord<Til<HS, TS>>>::Output:
        [const] InnerProduct<Til<HD, TD>, Output = F2<OutV>>,
    OutV: [const] F2Scalar,
{
    fn eval_f2_residue(&self, residue: F2<int>) -> F2<int> {
        f2(self.call(residue).value())
    }
}

/// Compile-time action for one mode in the sorted 1D `F2` right-inverse chain walk.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum F2RightFoldAction {
    /// Mode statically cannot continue the chain (out-of-range reach or exhausted).
    StaticSkip,
    /// Static `D == CurrD` (zero residue), static `S`, and `clmul(CurrD, S)` fits in `int`.
    StaticZeroResStaticNext,
    /// Static `D == CurrD` (zero residue), static `S`, and `clmul(CurrD, S)` overflows `int`.
    StaticZeroResExhausted,
    /// Static `D == CurrD` (zero residue) with dynamic `S`.
    StaticZeroResDynamicNext,
    /// Static non-zero residue with `clmul(S - 1, residue) < CurrD` and `clmul(CurrD, S)` fits.
    StaticBackSubStaticNext,
    /// Resolve successive input bits separately across a multi-mode inverse prefix.
    SplitBits,
    /// Static non-zero residue with `clmul(S - 1, residue) < CurrD` and `clmul(CurrD, S)` overflows.
    StaticBackSubExhausted,
    /// Dynamic `D`, `CurrD`, or dynamic `S` with non-zero residue: evaluate at runtime.
    DynamicStep,
}

const fn f2_residue_fits(s: int, residue: int, c: int) -> bool {
    let max_coord = sub(s, 1);
    matches!(clmul_maybe_overflows(max_coord, residue), Some(product) if product < c)
}

const fn f2_right_fold_action<S: ShapeScalar, D: F2Scalar, CurrD: RightRunningExtent>()
-> F2RightFoldAction {
    if CurrD::IS_EXHAUSTED {
        return F2RightFoldAction::StaticSkip;
    }
    match (D::STATIC_VALUE, CurrD::STATIC_EXTENT) {
        (Some(d), Some(c)) => {
            let residue = bitxor(c, d);
            if residue == 0 {
                match S::STATIC_VALUE {
                    Some(s) => {
                        if clmul_maybe_overflows(c, s).is_some() {
                            F2RightFoldAction::StaticZeroResStaticNext
                        } else {
                            F2RightFoldAction::StaticZeroResExhausted
                        }
                    }
                    None => F2RightFoldAction::StaticZeroResDynamicNext,
                }
            } else {
                match S::STATIC_VALUE {
                    Some(s) => {
                        if s <= 1 || !f2_residue_fits(s, residue, c) {
                            return F2RightFoldAction::StaticSkip;
                        }
                        if clmul_maybe_overflows(c, s).is_none() {
                            F2RightFoldAction::StaticBackSubExhausted
                        } else {
                            F2RightFoldAction::StaticBackSubStaticNext
                        }
                    }
                    None => F2RightFoldAction::DynamicStep,
                }
            }
        }
        _ => F2RightFoldAction::DynamicStep,
    }
}

const F2_RIGHT_FOLD_ACTION<S: ShapeScalar, D: F2Scalar, CurrD: RightRunningExtent, Acc: LayoutBase>:
    F2RightFoldAction = {
    let action = f2_right_fold_action::<S, D, CurrD>();
    if matches!(action,F2RightFoldAction::StaticBackSubStaticNext | F2RightFoldAction::StaticBackSubExhausted)
        && Acc::Shape::LEAVES > 1 && matches!(S::STATIC_VALUE,Some(s) if s > 2 && is_pow2(s)) {
        F2RightFoldAction::SplitBits
    } else { action }
};

const STATIC_F2_RESIDUE<CurrD: RightRunningExtent, D: F2Scalar>: int =
    match (CurrD::STATIC_EXTENT, D::STATIC_VALUE) {
        (Some(c), Some(d)) => bitxor(c, d),
        _ => 0,
    };

const STATIC_F2_NEXT_D<CurrD: RightRunningExtent, S: ShapeScalar>: int =
    match (CurrD::STATIC_EXTENT, S::STATIC_VALUE) {
        (Some(c), Some(s)) => clmul_maybe_overflows(c, s).unwrap_or(1),
        _ => 1,
    };

/// Converts a non-negative `ShapeScalar` (`Int<N>` or `int`) into `F2<Self>`.
#[doc(hidden)]
pub const trait ShapeScalarToF2: [const] ShapeScalar {
    type Scalar: [const] F2Scalar;
    fn to_f2(self) -> F2<Self::Scalar>;
}

const impl ShapeScalarToF2 for int {
    type Scalar = int;
    fn to_f2(self) -> F2<int> {
        F2::new(self)
    }
}

const impl<const N: int> ShapeScalarToF2 for Int<N>
where
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<N> }>,
    (): [const] ValidF2Value<{ IS_NON_NEG::<N> }>,
{
    type Scalar = Int<N>;
    fn to_f2(self) -> F2<Int<N>> {
        f2(Int::<N>)
    }
}

/// Executes one [`F2RightFoldAction`] step threading `(CurrD, Acc)` forward across sorted `F2` modes.
#[doc(hidden)]
pub const trait F2RightInverseFoldStep<
    CurrD: RightRunningExtent,
    Acc: Copy,
    const ACTION: F2RightFoldAction,
>: FlatTuple
{
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn f2_right_inverse_fold_step(
        self,
        curr_d: CurrD,
        acc: Acc,
    ) -> Layout<Self::Shape, Self::Stride>;
}

/// Forward-accumulating fold over sorted `F2` [`ModeTriple`]s carrying running covered
/// extent `CurrD` and accepted prefix layout `Acc` for characteristic-2 back-substitution.
#[doc(hidden)]
pub const trait F2RightInverseFold<CurrD: RightRunningExtent, Acc: Copy>: FlatTuple {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn f2_right_inverse_fold(self, curr_d: CurrD, acc: Acc) -> Layout<Self::Shape, Self::Stride>;
}

const impl<CurrD: RightRunningExtent, AccS: Shape, AccD: Stride<AccS>>
    F2RightInverseFold<CurrD, Layout<AccS, AccD>> for Nil
{
    type Shape = AccS;
    type Stride = AccD;

    fn f2_right_inverse_fold(
        self,
        _: CurrD,
        acc: Layout<AccS, AccD>,
    ) -> Layout<Self::Shape, Self::Stride> {
        acc
    }
}

const impl<S, D, P, Rest, CurrD, Acc> F2RightInverseFold<CurrD, Acc>
    for Til<ModeTriple<S, F2<D>, P>, Rest>
where
    S: ShapeScalar,
    D: F2Scalar,
    P: ShapeScalar,
    Rest: FlatTuple,
    CurrD: RightRunningExtent,
    Acc: [const] LayoutBase,
    Self: [const] F2RightInverseFoldStep<CurrD, Acc, { F2_RIGHT_FOLD_ACTION::<S, D, CurrD, Acc> }>,
{
    type Shape = <Self as F2RightInverseFoldStep<
        CurrD,
        Acc,
        { F2_RIGHT_FOLD_ACTION::<S, D, CurrD, Acc> },
    >>::Shape;
    type Stride = <Self as F2RightInverseFoldStep<
        CurrD,
        Acc,
        { F2_RIGHT_FOLD_ACTION::<S, D, CurrD, Acc> },
    >>::Stride;

    fn f2_right_inverse_fold(self, curr_d: CurrD, acc: Acc) -> Layout<Self::Shape, Self::Stride> {
        self.f2_right_inverse_fold_step(curr_d, acc)
    }
}

const impl<S, D, P, Rest, CurrD, Acc>
    F2RightInverseFoldStep<CurrD, Acc, { F2RightFoldAction::StaticSkip }>
    for Til<ModeTriple<S, F2<D>, P>, Rest>
where
    S: ShapeScalar,
    D: F2Scalar,
    P: ShapeScalar,
    CurrD: RightRunningExtent,
    Acc: Copy,
    Rest: FlatTuple + [const] F2RightInverseFold<CurrD, Acc>,
{
    type Shape = <Rest as F2RightInverseFold<CurrD, Acc>>::Shape;
    type Stride = <Rest as F2RightInverseFold<CurrD, Acc>>::Stride;

    fn f2_right_inverse_fold_step(
        self,
        curr_d: CurrD,
        acc: Acc,
    ) -> Layout<Self::Shape, Self::Stride> {
        self.1.f2_right_inverse_fold(curr_d, acc)
    }
}

type AppendedLayout<Acc, S, D> =
    Layout<<Acc as AppendInverseMode<S, D>>::Shape, <Acc as AppendInverseMode<S, D>>::Stride>;

const impl<S, D, P, Rest, CurrD, Acc>
    F2RightInverseFoldStep<CurrD, Acc, { F2RightFoldAction::StaticZeroResStaticNext }>
    for Til<ModeTriple<S, F2<D>, P>, Rest>
where
    S: ShapeScalar,
    D: F2Scalar,
    P: [const] ShapeScalarToF2,
    F2<P::Scalar>: StrideScalar + Stride<S>,
    CurrD: RightRunningExtent,
    Acc: [const] AppendInverseMode<S, F2<P::Scalar>>,
    Rest: FlatTuple
        + [const] F2RightInverseFold<
            Int<{ STATIC_F2_NEXT_D::<CurrD, S> }>,
            AppendedLayout<Acc, S, F2<P::Scalar>>,
        >,
{
    type Shape = <Rest as F2RightInverseFold<
        Int<{ STATIC_F2_NEXT_D::<CurrD, S> }>,
        AppendedLayout<Acc, S, F2<P::Scalar>>,
    >>::Shape;
    type Stride = <Rest as F2RightInverseFold<
        Int<{ STATIC_F2_NEXT_D::<CurrD, S> }>,
        AppendedLayout<Acc, S, F2<P::Scalar>>,
    >>::Stride;

    fn f2_right_inverse_fold_step(self, _: CurrD, acc: Acc) -> Layout<Self::Shape, Self::Stride> {
        let next_acc = acc.append_inverse_mode(self.0.shape, self.0.pps.to_f2());
        self.1.f2_right_inverse_fold(Int, next_acc)
    }
}

const impl<S, D, P, Rest, CurrD, Acc>
    F2RightInverseFoldStep<CurrD, Acc, { F2RightFoldAction::StaticZeroResExhausted }>
    for Til<ModeTriple<S, F2<D>, P>, Rest>
where
    S: ShapeScalar,
    D: F2Scalar,
    P: [const] ShapeScalarToF2,
    F2<P::Scalar>: StrideScalar + Stride<S>,
    CurrD: RightRunningExtent,
    Acc: [const] AppendInverseMode<S, F2<P::Scalar>>,
    Rest: FlatTuple
        + [const] F2RightInverseFold<ExhaustedExtent, AppendedLayout<Acc, S, F2<P::Scalar>>>,
{
    type Shape =
        <Rest as F2RightInverseFold<ExhaustedExtent, AppendedLayout<Acc, S, F2<P::Scalar>>>>::Shape;
    type Stride = <Rest as F2RightInverseFold<
        ExhaustedExtent,
        AppendedLayout<Acc, S, F2<P::Scalar>>,
    >>::Stride;

    fn f2_right_inverse_fold_step(self, _: CurrD, acc: Acc) -> Layout<Self::Shape, Self::Stride> {
        let next_acc = acc.append_inverse_mode(self.0.shape, self.0.pps.to_f2());
        self.1.f2_right_inverse_fold(ExhaustedExtent, next_acc)
    }
}

const impl<S, D, P, Rest, CurrD, Acc>
    F2RightInverseFoldStep<CurrD, Acc, { F2RightFoldAction::StaticZeroResDynamicNext }>
    for Til<ModeTriple<S, F2<D>, P>, Rest>
where
    S: [const] ShapeScalar,
    D: F2Scalar,
    P: [const] ShapeScalarToF2,
    F2<P::Scalar>: StrideScalar + Stride<S>,
    CurrD: [const] RightRunningExtent,
    Acc: [const] AppendInverseMode<S, F2<P::Scalar>>,
    Rest: FlatTuple
        + [const] F2RightInverseFold<DynamicExtent, AppendedLayout<Acc, S, F2<P::Scalar>>>,
{
    type Shape =
        <Rest as F2RightInverseFold<DynamicExtent, AppendedLayout<Acc, S, F2<P::Scalar>>>>::Shape;
    type Stride =
        <Rest as F2RightInverseFold<DynamicExtent, AppendedLayout<Acc, S, F2<P::Scalar>>>>::Stride;

    fn f2_right_inverse_fold_step(
        self,
        curr_d: CurrD,
        acc: Acc,
    ) -> Layout<Self::Shape, Self::Stride> {
        let s = self.0.shape.value();
        let next_curr = DynamicExtent::advance_f2(curr_d.dynamic_extent(), s);
        let next_acc = acc.append_inverse_mode(self.0.shape, self.0.pps.to_f2());
        self.1.f2_right_inverse_fold(next_curr, next_acc)
    }
}

const HALF_EXTENT<S: Integer>: int = div(S::STATIC_VALUE.unwrap(),2);
type SplitF2Modes<S, D, P, Rest> = Til<
    ModeTriple<Int<2>, F2<D>, P>,
    Til<
        ModeTriple<
            Int<{ HALF_EXTENT::<S> }>,
            F2<crate::integer::Product<D, Int<2>>>,
            crate::integer::Product<P, Int<2>>,
        >,
        Rest,
    >,
>;

const impl<S, D, P, Rest, CurrD, Acc>
    F2RightInverseFoldStep<CurrD, Acc, { F2RightFoldAction::SplitBits }>
    for Til<ModeTriple<S, F2<D>, P>, Rest>
where
    S: ShapeScalar,
    D: [const] F2Scalar + [const] crate::integer::ArithMul<D, Int<2>>,
    P: [const] ShapeScalar + [const] crate::integer::ArithMul<P, Int<2>>,
    crate::integer::Product<D, Int<2>>: [const] F2Scalar,
    Rest: FlatTuple,
    CurrD: RightRunningExtent,
    Acc: Copy,
    SplitF2Modes<S, D, P, Rest>: [const] F2RightInverseFold<CurrD, Acc>,
{
    type Shape = <SplitF2Modes<S, D, P, Rest> as F2RightInverseFold<CurrD, Acc>>::Shape;
    type Stride = <SplitF2Modes<S, D, P, Rest> as F2RightInverseFold<CurrD, Acc>>::Stride;
    fn f2_right_inverse_fold_step(self, c: CurrD, acc: Acc) -> Layout<Self::Shape, Self::Stride> {
        let mode = self.0;
        let low = ModeTriple {
            shape: Int::<2>,
            stride: mode.stride,
            pps: mode.pps,
        };
        let high = ModeTriple {
            shape: Int::<{ HALF_EXTENT::<S> }>,
            stride: F2::new(mul(mode.stride.inner(), Int::<2>)),
            pps: mul(mode.pps, Int::<2>),
        };
        Til(low, Til(high, self.1)).f2_right_inverse_fold(c, acc)
    }
}

type StaticBackSubStride<P, CorrV> = F2<F2XorOutput<<P as ShapeScalarToF2>::Scalar, CorrV>>;

const impl<S, D, P, Rest, CurrD, AccS, AccD, CorrV>
    F2RightInverseFoldStep<
        CurrD,
        Layout<AccS, AccD>,
        { F2RightFoldAction::StaticBackSubStaticNext },
    > for Til<ModeTriple<S, F2<D>, P>, Rest>
where
    S: ShapeScalar,
    D: F2Scalar,
    P: [const] ShapeScalarToF2,
    CurrD: RightRunningExtent,
    AccS: Shape,
    AccD: Stride<AccS>,
    (): [const] ValidF2Value<{ IS_NON_NEG::<{ STATIC_F2_RESIDUE::<CurrD, D> }> }>,
    F2<Int<{ STATIC_F2_RESIDUE::<CurrD, D> }>>: [const] IndexToCoord<AccS>,
    <F2<Int<{ STATIC_F2_RESIDUE::<CurrD, D> }>> as IndexToCoord<AccS>>::Output:
        [const] InnerProduct<AccD, Output = F2<CorrV>>,
    CorrV: [const] F2Scalar,
    P::Scalar: [const] F2XorScalar<CorrV, { BOTH_STATIC::<P::Scalar, CorrV> }>,
    StaticBackSubStride<P, CorrV>: StrideScalar + Stride<S>,
    Layout<AccS, AccD>: [const] AppendInverseMode<S, StaticBackSubStride<P, CorrV>>,
    Rest: FlatTuple
        + [const] F2RightInverseFold<
            Int<{ STATIC_F2_NEXT_D::<CurrD, S> }>,
            AppendedLayout<Layout<AccS, AccD>, S, StaticBackSubStride<P, CorrV>>,
        >,
{
    type Shape = <Rest as F2RightInverseFold<
        Int<{ STATIC_F2_NEXT_D::<CurrD, S> }>,
        AppendedLayout<Layout<AccS, AccD>, S, StaticBackSubStride<P, CorrV>>,
    >>::Shape;
    type Stride = <Rest as F2RightInverseFold<
        Int<{ STATIC_F2_NEXT_D::<CurrD, S> }>,
        AppendedLayout<Layout<AccS, AccD>, S, StaticBackSubStride<P, CorrV>>,
    >>::Stride;

    fn f2_right_inverse_fold_step(
        self,
        _: CurrD,
        acc: Layout<AccS, AccD>,
    ) -> Layout<Self::Shape, Self::Stride> {
        let corr = acc.call(f2(Int::<{ STATIC_F2_RESIDUE::<CurrD, D> }>));
        let stride = f2_add(self.0.pps.to_f2(), corr);
        let next_acc = acc.append_inverse_mode(self.0.shape, stride);
        self.1.f2_right_inverse_fold(Int, next_acc)
    }
}

const impl<S, D, P, Rest, CurrD, AccS, AccD, CorrV>
    F2RightInverseFoldStep<CurrD, Layout<AccS, AccD>, { F2RightFoldAction::StaticBackSubExhausted }>
    for Til<ModeTriple<S, F2<D>, P>, Rest>
where
    S: ShapeScalar,
    D: F2Scalar,
    P: [const] ShapeScalarToF2,
    CurrD: RightRunningExtent,
    AccS: Shape,
    AccD: Stride<AccS>,
    (): [const] ValidF2Value<{ IS_NON_NEG::<{ STATIC_F2_RESIDUE::<CurrD, D> }> }>,
    F2<Int<{ STATIC_F2_RESIDUE::<CurrD, D> }>>: [const] IndexToCoord<AccS>,
    <F2<Int<{ STATIC_F2_RESIDUE::<CurrD, D> }>> as IndexToCoord<AccS>>::Output:
        [const] InnerProduct<AccD, Output = F2<CorrV>>,
    CorrV: [const] F2Scalar,
    P::Scalar: [const] F2XorScalar<CorrV, { BOTH_STATIC::<P::Scalar, CorrV> }>,
    StaticBackSubStride<P, CorrV>: StrideScalar + Stride<S>,
    Layout<AccS, AccD>: [const] AppendInverseMode<S, StaticBackSubStride<P, CorrV>>,
    Rest: FlatTuple
        + [const] F2RightInverseFold<
            ExhaustedExtent,
            AppendedLayout<Layout<AccS, AccD>, S, StaticBackSubStride<P, CorrV>>,
        >,
{
    type Shape = <Rest as F2RightInverseFold<
        ExhaustedExtent,
        AppendedLayout<Layout<AccS, AccD>, S, StaticBackSubStride<P, CorrV>>,
    >>::Shape;
    type Stride = <Rest as F2RightInverseFold<
        ExhaustedExtent,
        AppendedLayout<Layout<AccS, AccD>, S, StaticBackSubStride<P, CorrV>>,
    >>::Stride;

    fn f2_right_inverse_fold_step(
        self,
        _: CurrD,
        acc: Layout<AccS, AccD>,
    ) -> Layout<Self::Shape, Self::Stride> {
        let corr = acc.call(f2(Int::<{ STATIC_F2_RESIDUE::<CurrD, D> }>));
        let stride = f2_add(self.0.pps.to_f2(), corr);
        let next_acc = acc.append_inverse_mode(self.0.shape, stride);
        self.1.f2_right_inverse_fold(ExhaustedExtent, next_acc)
    }
}

const impl<S, D, P, Rest, CurrD, Acc>
    F2RightInverseFoldStep<CurrD, Acc, { F2RightFoldAction::DynamicStep }>
    for Til<ModeTriple<S, F2<D>, P>, Rest>
where
    S: [const] ShapeScalar,
    D: [const] F2Scalar,
    P: [const] ShapeScalar,
    CurrD: [const] RightRunningExtent,
    Acc: [const] F2BackSubDynEval + [const] AppendInverseMode<int, F2<int>>,
    Rest: FlatTuple + [const] F2RightInverseFold<DynamicExtent, AppendedLayout<Acc, int, F2<int>>>,
{
    type Shape =
        <Rest as F2RightInverseFold<DynamicExtent, AppendedLayout<Acc, int, F2<int>>>>::Shape;
    type Stride =
        <Rest as F2RightInverseFold<DynamicExtent, AppendedLayout<Acc, int, F2<int>>>>::Stride;

    fn f2_right_inverse_fold_step(
        self,
        curr_d: CurrD,
        acc: Acc,
    ) -> Layout<Self::Shape, Self::Stride> {
        let s = self.0.shape.value();
        let d = self.0.stride.value();
        let curr_opt = curr_d.dynamic_extent();
        let matched = if s <= 1 || d == 0 {
            false
        } else if let Some(c) = curr_opt {
            let residue = bitxor(c, d);
            residue == 0 || f2_residue_fits(s, residue, c)
        } else {
            false
        };
        if matched {
            let c = curr_opt.unwrap();
            let residue = bitxor(c, d);
            let base_stride = f2(Integer::value(self.0.pps));
            let stride = if residue == 0 {
                base_stride
            } else {
                f2_add(base_stride, acc.eval_f2_residue(f2(residue)))
            };
            let next_curr = DynamicExtent::advance_f2(curr_opt, s);
            let next_acc = acc.append_inverse_mode(s, stride);
            self.1.f2_right_inverse_fold(next_curr, next_acc)
        } else {
            let next_acc = acc.append_inverse_mode(1, f2(0i32));
            self.1
                .f2_right_inverse_fold(DynamicExtent(curr_opt), next_acc)
        }
    }
}

/// Polynomial division by a non-binary radix s produces remainders below
/// 2^degree(s). Keep the original polynomial positional weight for the quotient.
/// When both the forward and inverse strides encode the divisor, the mapping
/// is the identity on the resulting contiguous binary space.
const fn polynomial_radices(s: int, d: int, p: int, t: int, e: int, q: int) -> [int; 4] {
    let mut modes = [s, t, e, q];
    if p == 1 && q == s && s > 1 && !is_pow2(s) && t > 1 {
        let h = bit_floor(s);
        modes[0] = h;
        if d == 1 && e == s && is_pow2(t) {
            modes[2] = h;
            modes[3] = h;
        }
    } else if q == 1 && p == t && t > 1 && !is_pow2(t) && s > 1 {
        modes[1] = bit_floor(t);
    }
    modes
}
/// Normalizes a canonical polynomial radix before the inverse stride walk.
#[doc(hidden)]
pub const trait InverseRadix: FlatTuple {
    type Modes: FlatTuple;
    fn inverse_radix(self) -> Self::Modes;
}

type PairModes<S, D, P, T, E, Q> = Til<ModeTriple<S, F2<D>, P>, Til<ModeTriple<T, F2<E>, Q>, Nil>>;
const PAIR_VALUES<S:Integer,D:Integer,P:Integer,T:Integer,E:Integer,Q:Integer>:[int;4] =
    match (S::STATIC_VALUE,D::STATIC_VALUE,P::STATIC_VALUE,T::STATIC_VALUE,E::STATIC_VALUE,Q::STATIC_VALUE) {
        (Some(s),Some(d),Some(p),Some(t),Some(e),Some(q)) => polynomial_radices(s,d,p,t,e,q),
        _=>[0;4],
    };
const PAIR_CASE<S:Integer,D:Integer,P:Integer,T:Integer,E:Integer,Q:Integer>:int =
    if matches!((S::STATIC_VALUE,T::STATIC_VALUE),(Some(s),Some(t)) if is_pow2(s) && is_pow2(t)) {0}
    else if S::IS_STATIC && D::IS_STATIC && P::IS_STATIC && T::IS_STATIC && E::IS_STATIC && Q::IS_STATIC {1}
    else {-1};
const PAIR_S<S:Integer,D:Integer,P:Integer,T:Integer,E:Integer,Q:Integer>:int=PAIR_VALUES::<S,D,P,T,E,Q>[0];
const PAIR_T<S:Integer,D:Integer,P:Integer,T:Integer,E:Integer,Q:Integer>:int=PAIR_VALUES::<S,D,P,T,E,Q>[1];
const PAIR_E<S:Integer,D:Integer,P:Integer,T:Integer,E:Integer,Q:Integer>:int=PAIR_VALUES::<S,D,P,T,E,Q>[2];
const PAIR_Q<S:Integer,D:Integer,P:Integer,T:Integer,E:Integer,Q:Integer>:int=PAIR_VALUES::<S,D,P,T,E,Q>[3];
/// Selects unchanged, static, or runtime normalization for two sorted modes.
#[doc(hidden)]
pub const trait NormalizeRadix<const CASE: int>: FlatTuple {
    type Modes: FlatTuple;
    fn normalize_radix(self) -> Self::Modes;
}
const impl<M: FlatTuple> NormalizeRadix<0> for M {
    type Modes = M;
    fn normalize_radix(self) -> M {
        self
    }
}
const impl<S: ShapeScalar, D: F2Scalar, P: ShapeScalar, T: ShapeScalar, E: F2Scalar, Q: ShapeScalar>
    NormalizeRadix<1> for PairModes<S, D, P, T, E, Q>
where
    (): [const] ValidF2Value<{ IS_NON_NEG::<{ PAIR_E::<S, D, P, T, E, Q> }> }>,
{
    type Modes = PairModes<
        Int<{ PAIR_S::<S, D, P, T, E, Q> }>,
        D,
        P,
        Int<{ PAIR_T::<S, D, P, T, E, Q> }>,
        Int<{ PAIR_E::<S, D, P, T, E, Q> }>,
        Int<{ PAIR_Q::<S, D, P, T, E, Q> }>,
    >;
    fn normalize_radix(self) -> Self::Modes {
        Til(
            ModeTriple {
                shape: Int,
                stride: self.0.stride,
                pps: self.0.pps,
            },
            Til(
                ModeTriple {
                    shape: Int,
                    stride: f2(Int),
                    pps: Int,
                },
                Nil,
            ),
        )
    }
}
const impl<
    S: [const] ShapeScalar,
    D: [const] F2Scalar,
    P: [const] ShapeScalar,
    T: [const] ShapeScalar,
    E: [const] F2Scalar,
    Q: [const] ShapeScalar,
> NormalizeRadix<-1> for PairModes<S, D, P, T, E, Q>
{
    type Modes = PairModes<int, D, P, int, int, int>;
    fn normalize_radix(self) -> Self::Modes {
        let a = self.0;
        let b = self.1.0;
        let [s, t, e, q] = polynomial_radices(
            a.shape.value(),
            a.stride.value(),
            a.pps.value(),
            b.shape.value(),
            b.stride.value(),
            b.pps.value(),
        );
        Til(
            ModeTriple {
                shape: s,
                stride: a.stride,
                pps: a.pps,
            },
            Til(
                ModeTriple {
                    shape: t,
                    stride: f2(e),
                    pps: q,
                },
                Nil,
            ),
        )
    }
}
const impl<S: ShapeScalar, D: F2Scalar, P: ShapeScalar, T: ShapeScalar, E: F2Scalar, Q: ShapeScalar>
    InverseRadix for PairModes<S, D, P, T, E, Q>
where
    Self: [const] NormalizeRadix<{ PAIR_CASE::<S, D, P, T, E, Q> }>,
{
    type Modes = <Self as NormalizeRadix<{ PAIR_CASE::<S, D, P, T, E, Q> }>>::Modes;
    fn inverse_radix(self) -> Self::Modes {
        self.normalize_radix()
    }
}
const impl<H: Leaf> InverseRadix for Til<H, Nil> {
    type Modes = Self;
    fn inverse_radix(self) -> Self {
        self
    }
}
const impl<H: Leaf, J: Leaf, K: Leaf, T: FlatTuple> InverseRadix for Til<H, Til<J, Til<K, T>>> {
    type Modes = Self;
    fn inverse_radix(self) -> Self {
        self
    }
}
/// Candidate trailing mode `(shape, stride, pps)` on a sorted F2 axis.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct F2TailMode {
    shape: int,
    stride: int,
    pps: int,
}

/// Adds each independent coordinate bit of a sorted mode to the binary basis
/// and identifies the highest-weight active mode on the axis.
#[doc(hidden)]
pub const trait InsertInverseBits: FlatTuple {
    const STATIC_BINARY: bool;
    /// Inserts independent coordinate bits for every active mode whose positional weight differs from `skip_p`.
    fn insert_bits(self, basis: &mut BinaryPreimages, skip_p: int);
    /// Returns the active mode (`shape > 1, stride != 0`) with the largest positional weight `pps`.
    fn max_pps_mode(self, best: Option<F2TailMode>) -> Option<F2TailMode>;
}
const impl InsertInverseBits for Nil {
    const STATIC_BINARY: bool = true;
    fn insert_bits(self, _: &mut BinaryPreimages, _: int) {}
    fn max_pps_mode(self, best: Option<F2TailMode>) -> Option<F2TailMode> {
        best
    }
}
const impl<
    S: [const] ShapeScalar,
    D: [const] F2Scalar,
    P: [const] ShapeScalar,
    T: [const] InsertInverseBits,
> InsertInverseBits for Til<ModeTriple<S, F2<D>, P>, T>
{
    const STATIC_BINARY: bool = matches!(S::STATIC_VALUE, Some(s) if is_pow2(s))
        && matches!(P::STATIC_VALUE, Some(p) if is_pow2(p))
        && T::STATIC_BINARY;

    fn insert_bits(self, basis: &mut BinaryPreimages, skip_p: int) {
        let s = self.0.shape.value();
        let d = self.0.stride.value();
        let p = self.0.pps.value();
        if d != 0 && s > 1 && p != skip_p {
            // Only the complete low binary field can be varied independently.
            let mut k = 1;
            while k <= s / 2 {
                basis.insert(mul(d, k), mul(p, k));
                k = mul(k, 2);
            }
        }
        self.1.insert_bits(basis, skip_p);
    }

    fn max_pps_mode(self, best: Option<F2TailMode>) -> Option<F2TailMode> {
        let s = self.0.shape.value();
        let d = self.0.stride.value();
        let p = self.0.pps.value();
        let next = if d != 0 && s > 1 {
            match best {
                Some(curr) if curr.pps >= p => best,
                _ => Some(F2TailMode {
                    shape: s,
                    stride: d,
                    pps: p,
                }),
            }
        } else {
            best
        };
        self.1.max_pps_mode(next)
    }
}

/// Selects the static stride fold or the runtime binary basis for a multi-mode axis.
#[doc(hidden)]
pub const trait InvertF2Axis<const BINARY: bool>: FlatTuple {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn invert_f2_axis(self, domain_tail_size: Option<int>) -> Layout<Self::Shape, Self::Stride>;
}
const impl<M> InvertF2Axis<false> for M
where
    M: [const] InverseRadix,
    M::Modes: FlatTuple + [const] F2RightInverseFold<Int<1>, Layout<Nil, Nil>>,
    Layout<
        <M::Modes as F2RightInverseFold<Int<1>, Layout<Nil, Nil>>>::Shape,
        <M::Modes as F2RightInverseFold<Int<1>, Layout<Nil, Nil>>>::Stride,
    >: [const] FinalizeRightAxis,
{
    type Shape = <Layout<
        <M::Modes as F2RightInverseFold<Int<1>, Layout<Nil, Nil>>>::Shape,
        <M::Modes as F2RightInverseFold<Int<1>, Layout<Nil, Nil>>>::Stride,
    > as FinalizeRightAxis>::Shape;
    type Stride = <Layout<
        <M::Modes as F2RightInverseFold<Int<1>, Layout<Nil, Nil>>>::Shape,
        <M::Modes as F2RightInverseFold<Int<1>, Layout<Nil, Nil>>>::Stride,
    > as FinalizeRightAxis>::Stride;

    fn invert_f2_axis(self, _: Option<int>) -> Layout<Self::Shape, Self::Stride> {
        self.inverse_radix()
            .f2_right_inverse_fold(Int, Layout::new(Nil, Nil))
            .finalize_right_axis()
    }
}

// A fixed tuple keeps the return type independent of the runtime matrix rank.
// Unused high bits contribute unit extents and zero strides.
type Bits4<V> = Til<V, Til<V, Til<V, Til<V, Nil>>>>;
type Bits8<V> = Til<Bits4<V>, Til<Bits4<V>, Nil>>;
type Bits32<V> = Til<Bits8<V>, Til<Bits8<V>, Til<Bits8<V>, Til<Bits8<V>, Nil>>>>;
const fn four<V: [const] Item>(a: &[V; 32], i: usize) -> Bits4<V> {
    crate::tuple![a[i], a[i + 1], a[i + 2], a[i + 3]]
}
const fn eight<V: [const] Item>(a: &[V; 32], i: usize) -> Bits8<V> {
    crate::tuple![four(a, i), four(a, i + 4)]
}
const fn binary_tuple<V: [const] Item>(a: &[V; 32]) -> Bits32<V> {
    crate::tuple![eight(a, 0), eight(a, 8), eight(a, 16), eight(a, 24)]
}

/// Extends a solved binary prefix `0..C` (`C = 2^b`) with the full trailing non-power-of-two mode
/// when its stride is `F2(C)` and its preimage bit field is disjoint from the prefix preimages.
const fn trailing_tail_inverse<M: [const] InsertInverseBits>(
    modes: M,
    domain_tail_size: Option<int>,
    binary_bits: usize,
) -> Option<Layout<Bits32<int>, Bits32<F2<int>>>> {
    let Some(total_size) = domain_tail_size else {
        return None;
    };
    let Some(tail) = modes.max_pps_mode(None) else {
        return None;
    };
    if tail.shape <= 1 || is_pow2(tail.shape) {
        return None;
    }
    if tail.stride <= 0 || !is_pow2(tail.stride) || tail.pps <= 0 || !is_pow2(tail.pps) {
        return None;
    }
    if !matches!(mul_maybe_overflows(tail.pps, tail.shape), Some(end) if end == total_size) {
        return None;
    }
    let Some(tail_p_mask) = carry_free_mask(tail.shape, tail.pps) else {
        return None;
    };
    let Some(extended_size) = mul_maybe_overflows(tail.stride, tail.shape) else {
        return None;
    };
    if binary_bits >= 31 || extended_size <= (1i32 << binary_bits) {
        return None;
    }

    let prefix_bits = (bit_length_non_neg(tail.stride) - 1) as usize;
    let mut prefix_basis = BinaryPreimages::new();
    modes.insert_bits(&mut prefix_basis, tail.pps);
    let mut shape = [1; 32];
    let mut stride = [f2(0i32); 32];
    let mut prefix_mask: uint = 0;
    let mut j = 0;
    while j < prefix_bits {
        let Some(x) = prefix_basis.solve(1 << j) else {
            return None;
        };
        shape[j] = 2;
        stride[j] = f2(x);
        prefix_mask |= x as uint;
        j += 1;
    }
    if prefix_mask >= tail.pps as uint || (prefix_mask & tail_p_mask) != 0 {
        return None;
    }
    shape[prefix_bits] = tail.shape;
    stride[prefix_bits] = f2(tail.pps);
    Some(Layout::new(binary_tuple(&shape), binary_tuple(&stride)))
}

const impl<M: [const] InsertInverseBits> InvertF2Axis<true> for M {
    type Shape = Bits32<int>;
    type Stride = Bits32<F2<int>>;
    fn invert_f2_axis(self, domain_tail_size: Option<int>) -> Layout<Self::Shape, Self::Stride> {
        let mut basis = BinaryPreimages::new();
        self.insert_bits(&mut basis, 0);
        let mut shape = [1; 32];
        let mut stride = [f2(0i32); 32];
        let mut k = 0;
        while k < 31 {
            let Some(x) = basis.solve(1 << k) else {
                break;
            };
            shape[k] = 2;
            stride[k] = f2(x);
            k += 1;
        }
        if let Some(extended) = trailing_tail_inverse(self, domain_tail_size, k) {
            return extended;
        }
        Layout::new(binary_tuple(&shape), binary_tuple(&stride))
    }
}
const USE_BINARY_BASIS<M: InsertInverseBits>: bool = M::RANK > 2 && (!M::IS_STATIC || !M::STATIC_BINARY);
const impl<M> InvertSortedAxis<true, { AxisAlgebra::F2 }> for M
where
    M: InsertInverseBits + [const] InvertF2Axis<{ USE_BINARY_BASIS::<M> }>,
{
    type Shape = <M as InvertF2Axis<{ USE_BINARY_BASIS::<M> }>>::Shape;
    type Stride = <M as InvertF2Axis<{ USE_BINARY_BASIS::<M> }>>::Stride;
    fn invert_sorted_axis(self, domain_tail_size: Option<int>) -> Layout<Self::Shape, Self::Stride> {
        self.invert_f2_axis(domain_tail_size)
    }
}
/// Validates a single-mode `F2` axis for `left_inverse` (`gap == 1`, i.e. stride == `F2(1)`).
const F2_LEFT_UNIT_OK<D: F2Scalar>: bool = match D::STATIC_VALUE {
    Some(d) => d == 1,
    None => true,
};

/// Compile-time guard restricting `F2` `left_inverse` to unit-gap (`F2(1)`) chains.
#[doc(hidden)]
pub const trait ValidF2LeftInverse<const OK: bool> {}
const impl ValidF2LeftInverse<true> for () {}

const impl<S, D, P> InvertSortedAxis<false, { AxisAlgebra::F2 }>
    for Til<ModeTriple<S, F2<D>, P>, Nil>
where
    S: [const] ShapeScalar,
    D: [const] F2Scalar,
    P: [const] ShapeScalar + StrideScalar + Stride<S>,
    (): [const] ValidF2LeftInverse<{ F2_LEFT_UNIT_OK::<D> }>,
{
    type Shape = S;
    type Stride = P;

    fn invert_sorted_axis(self, _: Option<int>) -> Layout<S, P> {
        let d = self.0.stride.value();
        assert!(
            d == 1,
            "left_inverse on F2-strided layout requires every gap to be 1"
        );
        Layout::new(self.0.shape, self.0.pps)
    }
}

type SortedAxisModes<TargetP, FS, FD, FP> =
    <<FS as FilterInverseModes<TargetP, FD, FP>>::Output as SortInverseModes>::Output;

/// Walks the codomain basis tree (`make_basis_like(coprofile)`) and assembles the
/// per-axis inverses into a single [`Layout`].
#[doc(hidden)]
pub const trait BuildInverse<const RIGHT: bool, FS: FlatTuple, FD: FlatTuple, FP: FlatTuple>:
    Item
{
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn build_inverse(
        self,
        flat_s: FS,
        flat_d: FD,
        flat_p: FP,
        domain_tail_size: Option<int>,
    ) -> Layout<Self::Shape, Self::Stride>;
}

const impl<const RIGHT: bool, FS: FlatTuple, FD: FlatTuple, FP: FlatTuple>
    BuildInverse<RIGHT, FS, FD, FP> for Nil
{
    type Shape = Nil;
    type Stride = Nil;

    fn build_inverse(self, _: FS, _: FD, _: FP, _: Option<int>) -> Layout<Nil, Nil> {
        Layout::new(Nil, Nil)
    }
}

const impl<const RIGHT: bool, FS, FD, FP> BuildInverse<RIGHT, FS, FD, FP> for Int<1>
where
    FS: FlatTuple + [const] FilterInverseModes<ModeEnd, FD, FP>,
    FD: FlatTuple,
    FP: FlatTuple,
    <FS as FilterInverseModes<ModeEnd, FD, FP>>::Output: [const] SortInverseModes,
    SortedAxisModes<ModeEnd, FS, FD, FP>: AxisAlgebraOf
        + [const] InvertSortedAxis<RIGHT, { AXIS_ALGEBRA_OF::<SortedAxisModes<ModeEnd, FS, FD, FP>> }>,
{
    type Shape = <SortedAxisModes<ModeEnd, FS, FD, FP> as InvertSortedAxis<
        RIGHT,
        { AXIS_ALGEBRA_OF::<SortedAxisModes<ModeEnd, FS, FD, FP>> },
    >>::Shape;
    type Stride = <SortedAxisModes<ModeEnd, FS, FD, FP> as InvertSortedAxis<
        RIGHT,
        { AXIS_ALGEBRA_OF::<SortedAxisModes<ModeEnd, FS, FD, FP>> },
    >>::Stride;

    fn build_inverse(
        self,
        flat_s: FS,
        flat_d: FD,
        flat_p: FP,
        domain_tail_size: Option<int>,
    ) -> Layout<Self::Shape, Self::Stride> {
        flat_s
            .filter_inverse_modes(flat_d, flat_p)
            .sort_inverse_modes()
            .invert_sorted_axis(domain_tail_size)
    }
}

const impl<const RIGHT: bool, P, FS, FD, FP> BuildInverse<RIGHT, FS, FD, FP> for Basis<P>
where
    P: [const] BasisPath,
    Self: Item,
    FS: FlatTuple + [const] FilterInverseModes<P, FD, FP>,
    FD: FlatTuple,
    FP: FlatTuple,
    <FS as FilterInverseModes<P, FD, FP>>::Output: [const] SortInverseModes,
    SortedAxisModes<P, FS, FD, FP>: AxisAlgebraOf
        + [const] InvertSortedAxis<RIGHT, { AXIS_ALGEBRA_OF::<SortedAxisModes<P, FS, FD, FP>> }>,
{
    type Shape = <SortedAxisModes<P, FS, FD, FP> as InvertSortedAxis<
        RIGHT,
        { AXIS_ALGEBRA_OF::<SortedAxisModes<P, FS, FD, FP>> },
    >>::Shape;
    type Stride = <SortedAxisModes<P, FS, FD, FP> as InvertSortedAxis<
        RIGHT,
        { AXIS_ALGEBRA_OF::<SortedAxisModes<P, FS, FD, FP>> },
    >>::Stride;

    fn build_inverse(
        self,
        flat_s: FS,
        flat_d: FD,
        flat_p: FP,
        domain_tail_size: Option<int>,
    ) -> Layout<Self::Shape, Self::Stride> {
        flat_s
            .filter_inverse_modes(flat_d, flat_p)
            .sort_inverse_modes()
            .invert_sorted_axis(domain_tail_size)
    }
}

const impl<const RIGHT: bool, H, T, FS, FD, FP> BuildInverse<RIGHT, FS, FD, FP> for Til<H, T>
where
    FS: FlatTuple,
    FD: FlatTuple,
    FP: FlatTuple,
    H: [const] BuildInverse<RIGHT, FS, FD, FP>,
    T: Tuple + [const] BuildInverse<RIGHT, FS, FD, FP>,
    T::Shape: Tuple + Shape,
    T::Stride: Tuple + Stride<T::Shape>,
{
    type Shape = Til<H::Shape, T::Shape>;
    type Stride = Til<H::Stride, T::Stride>;

    fn build_inverse(
        self,
        flat_s: FS,
        flat_d: FD,
        flat_p: FP,
        domain_tail_size: Option<int>,
    ) -> Layout<Self::Shape, Self::Stride> {
        let head = self.0.build_inverse(flat_s, flat_d, flat_p, domain_tail_size);
        let tail = self.1.build_inverse(flat_s, flat_d, flat_p, domain_tail_size);
        Layout::new(Til(head.shape, tail.shape), Til(head.stride, tail.stride))
    }
}

/// Computes the largest right inverse layout $L^{\ddagger}$.
pub const trait RightInverse: Copy {
    /// Resulting right-inverse [`Layout`] type.
    type Output: Copy;

    /// Computes the right inverse, panicking if a dynamic precondition or overflow fails.
    fn right_inverse(self) -> Self::Output;
}

/// Computes the left inverse layout $L^{\dagger}$.
pub const trait LeftInverse: Copy {
    /// Resulting left-inverse [`Layout`] type.
    type Output: Copy;

    /// Computes the left inverse, panicking if a dynamic precondition or overflow fails.
    fn left_inverse(self) -> Self::Output;
}

type LayoutBases<S, D> = MakeBasisLikeOutput<<Layout<S, D> as Coprofile>::Output>;
type CoalescedLayout<S, D> =
    <Layout<<S as Flatten>::Output, <D as Flatten>::Output> as Coalesce<true>>::Output;
type CoalescedFlatS<S, D> = <<CoalescedLayout<S, D> as LayoutBase>::Shape as Flatten>::Output;
type CoalescedFlatD<S, D> = <<CoalescedLayout<S, D> as LayoutBase>::Stride as Flatten>::Output;
type CoalescedFlatP<S, D> = <CoalescedFlatS<S, D> as PrefixProduct<Int<1>>>::Stride;
type BuiltInverseLayout<const RIGHT: bool, S, D> = Layout<
    <LayoutBases<S, D> as BuildInverse<
        RIGHT,
        CoalescedFlatS<S, D>,
        CoalescedFlatD<S, D>,
        CoalescedFlatP<S, D>,
    >>::Shape,
    <LayoutBases<S, D> as BuildInverse<
        RIGHT,
        CoalescedFlatS<S, D>,
        CoalescedFlatD<S, D>,
        CoalescedFlatP<S, D>,
    >>::Stride,
>;

const fn invert_layout<const RIGHT: bool, S, D>(
    layout: Layout<S, D>,
    domain_tail_size: Option<int>,
) -> BuiltInverseLayout<RIGHT, S, D>
where
    S: Shape + [const] Flatten,
    D: Stride<S> + [const] Flatten,
    S::Output: [const] AssertPositiveShape,
    Layout<S::Output, D::Output>: [const] Coalesce<true>,
    CoalescedLayout<S, D>: [const] LayoutBase,
    <CoalescedLayout<S, D> as LayoutBase>::Shape: [const] Flatten,
    <CoalescedLayout<S, D> as LayoutBase>::Stride: [const] Flatten,
    CoalescedFlatS<S, D>: [const] PrefixProduct<Int<1>>,
    CoalescedFlatP<S, D>: FlatTuple,
    Layout<S, D>: [const] Coprofile,
    <<Layout<S, D> as Coprofile>::Output as Item>::Profile: [const] MakeBasisLikeTopProfile,
    LayoutBases<S, D>: [const] BuildInverse<
            RIGHT,
            CoalescedFlatS<S, D>,
            CoalescedFlatD<S, D>,
            CoalescedFlatP<S, D>,
        >,
{
    layout.shape.flatten().assert_positive_shape();
    let bases = make_basis_like(layout.coprofile());
    let coalesced = layout.coalesce_extended();
    let flat_s = coalesced.shape().flatten();
    let flat_d = coalesced.stride().flatten();
    let flat_p = prefix_product_with(flat_s, Int::<1>);
    bases.build_inverse(flat_s, flat_d, flat_p, domain_tail_size)
}

/// Verifies that an F2 preimage can be decoded by the original domain shape.
#[doc(hidden)]
pub const trait InverseCoordinates<const F2_INDEX: bool>: Shape {
    fn validate_inverse_coordinates(self);
}
const impl<S: Shape> InverseCoordinates<false> for S {
    fn validate_inverse_coordinates(self) {}
}
const impl<S: [const] BinaryExtents> InverseCoordinates<true> for S
where
    F2<int>: [const] IndexToCoord<S>,
{
    fn validate_inverse_coordinates(self) {
        if !self.binary_extents() {
            let _ = f2(0i32).index_to_coord(&self);
        }
    }
}
const RETURNS_F2<L: LayoutBase>: bool = L::Stride::HAS_F2 where L::Stride: ContainsF2;

const LIFT_INVERSE<L:LayoutBase,S:BinaryExtents>:bool = {
    let mixed=L::Stride::HAS_F2 && L::Stride::HAS_INTEGER;
    assert!(!mixed || !matches!(S::STATIC_BINARY,Some(false)), "mixed inverse needs disjoint binary domain fields");
    mixed
} where L::Stride: ContainsF2;

const KEEP_INVERSE<L:LayoutBase,S:BinaryExtents>:bool = !LIFT_INVERSE::<L,S> where L::Stride:ContainsF2;

const impl<S, D> RightInverse for Layout<S, D>
where
    S: [const] BinaryExtents
        + Shape
        + [const] Flatten
        + [const] InverseCoordinates<{ RETURNS_F2::<BuiltInverseLayout<true, S, D>> }>,
    D: Stride<S> + [const] Flatten,
    S::Output: [const] AssertPositiveShape,
    Layout<S::Output, D::Output>: [const] Coalesce<true>,
    CoalescedLayout<S, D>: [const] LayoutBase,
    <CoalescedLayout<S, D> as LayoutBase>::Shape: [const] Flatten,
    <CoalescedLayout<S, D> as LayoutBase>::Stride: [const] Flatten,
    CoalescedFlatS<S, D>: [const] PrefixProduct<Int<1>>,
    CoalescedFlatP<S, D>: FlatTuple,
    Self: [const] Coprofile,
    <<Self as Coprofile>::Output as Item>::Profile: [const] MakeBasisLikeTopProfile,
    LayoutBases<S, D>: [const] BuildInverse<true, CoalescedFlatS<S, D>, CoalescedFlatD<S, D>, CoalescedFlatP<S, D>>,
    BuiltInverseLayout<true, S, D>: LayoutBase
        + [const] ScaleIfNotOne<F2<Int<1>>, { KEEP_INVERSE::<BuiltInverseLayout<true, S, D>, S> }>,
    <BuiltInverseLayout<true, S, D> as LayoutBase>::Stride: ContainsF2,
{
    type Output = <BuiltInverseLayout<true, S, D> as ScaleIfNotOne<
        F2<Int<1>>,
        { KEEP_INVERSE::<BuiltInverseLayout<true, S, D>, S> },
    >>::Output;

    fn right_inverse(self) -> Self::Output {
        <S as InverseCoordinates<{RETURNS_F2::<BuiltInverseLayout<true,S,D>>}>>::validate_inverse_coordinates(self.shape);
        if LIFT_INVERSE::<BuiltInverseLayout<true, S, D>, S> {
            assert!(
                self.shape.binary_extents(),
                "mixed inverse needs disjoint binary domain fields"
            );
        }
        let domain_tail_size = self.shape.leading_binary_size();
        <BuiltInverseLayout<true, S, D> as ScaleIfNotOne<
            F2<Int<1>>,
            { KEEP_INVERSE::<BuiltInverseLayout<true, S, D>, S> },
        >>::scale_if_not_one(invert_layout::<true, S, D>(self, domain_tail_size), f2(Int::<1>))
    }
}

const impl<S, D> LeftInverse for Layout<S, D>
where
    S: Shape + [const] Flatten,
    D: Stride<S> + [const] Flatten,
    S::Output: [const] AssertPositiveShape,
    Layout<S::Output, D::Output>: [const] Coalesce<true>,
    CoalescedLayout<S, D>: [const] LayoutBase,
    <CoalescedLayout<S, D> as LayoutBase>::Shape: [const] Flatten,
    <CoalescedLayout<S, D> as LayoutBase>::Stride: [const] Flatten,
    CoalescedFlatS<S, D>: [const] PrefixProduct<Int<1>>,
    CoalescedFlatP<S, D>: FlatTuple,
    Self: [const] Coprofile,
    <<Self as Coprofile>::Output as Item>::Profile: [const] MakeBasisLikeTopProfile,
    LayoutBases<S, D>: [const] BuildInverse<
            false,
            CoalescedFlatS<S, D>,
            CoalescedFlatD<S, D>,
            CoalescedFlatP<S, D>,
        >,
{
    type Output = BuiltInverseLayout<false, S, D>;

    fn left_inverse(self) -> Self::Output {
        invert_layout::<false, S, D>(self, None)
    }
}

macro_rules! impl_tiler_inverse {
    ([$($g:tt)*] $T:ty) => {
        const impl<$($g)*> RightInverse for $T
        where
            Self: [const] Tiler,
            <Self as Tiler>::OutputLayout: [const] RightInverse,
        {
            type Output = <<Self as Tiler>::OutputLayout as RightInverse>::Output;

            fn right_inverse(self) -> Self::Output {
                tiler_to_layout(self).right_inverse()
            }
        }

        const impl<$($g)*> LeftInverse for $T
        where
            Self: [const] Tiler,
            <Self as Tiler>::OutputLayout: [const] LeftInverse,
        {
            type Output = <<Self as Tiler>::OutputLayout as LeftInverse>::Output;

            fn left_inverse(self) -> Self::Output {
                tiler_to_layout(self).left_inverse()
            }
        }
    };
}

impl_tiler_inverse!([] Nil);
impl_tiler_inverse!([] int);
impl_tiler_inverse!([const N: int,] Int<N>);
impl_tiler_inverse!([H: Item, T: Tuple,] Til<H, T>);

impl<S, D> Layout<S, D> {
    /// Computes the largest right inverse layout $L^{\ddagger}$.
    pub const fn right_inverse(self) -> <Self as RightInverse>::Output
    where
        Self: [const] RightInverse,
    {
        RightInverse::right_inverse(self)
    }

    /// Computes the left inverse layout $L^{\dagger}$.
    pub const fn left_inverse(self) -> <Self as LeftInverse>::Output
    where
        Self: [const] LeftInverse,
    {
        LeftInverse::left_inverse(self)
    }
}

/// Computes the largest right inverse layout of a [`Layout`] or [`Tiler`].
///
/// Every codomain axis must use one stride algebra. Combining integer and [`F2`]
/// strides on the same axis is a compile-time error.
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::algebra::right_inverse;
/// use noodle_cute::f2::f2;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
/// use noodle_cute::{int_tuple, tuple};
///
/// let _ = right_inverse(Layout::new(
///     int_tuple![Int::<4>, Int::<4>],
///     tuple![Int::<1>, f2(Int::<4>)],
/// ));
/// ```
pub const fn right_inverse<A: [const] RightInverse>(a: A) -> A::Output {
    a.right_inverse()
}

/// Computes the left inverse layout of a [`Layout`] or [`Tiler`].
///
/// Each F2 codomain axis supports one active mode with unit stride. A larger
/// static stride fails at compile time.
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::algebra::left_inverse;
/// use noodle_cute::f2::f2;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
///
/// let _ = left_inverse(Layout::new(Int::<8>, f2(Int::<2>)));
/// ```
pub const fn left_inverse<A: [const] LeftInverse>(a: A) -> A::Output {
    a.left_inverse()
}
