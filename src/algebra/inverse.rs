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
    Basis, BasisPath, ComparePath, MakeBasisLikeOutput, MakeBasisLikeTopProfile, ModeEnd,
    PathOrdering, make_basis_like,
};
use crate::integer::{IS_NON_NEG, Int, Integer, div, eq, int, mul, rem};
use crate::layout::{Coprofile, Layout, LayoutBase, Tiler, tiler_to_layout};
use crate::shape::{AssertPositiveShape, Shape, ShapeScalar, ValidShapeExtent};
use crate::stride::{PrefixProduct, Stride, StrideScalar, prefix_product_with};
use crate::tuple::{FlatTuple, Flatten, Item, LeafProfile, Nil, Til, Tuple};

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

const impl<S: Item, D: Item, P: Item> Item for ModeTriple<S, D, P> {
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
    D::Value: Integer,
    TargetP: BasisPath,
    D::Path: ComparePath<TargetP>,
{
    if matches!(D::Value::STATIC_VALUE, Some(0)) {
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
    D::Value: Integer,
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
    D::Value: [const] Integer,
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
    D::Value: [const] Integer,
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

const fn inverse_mode_precedes<DX: Integer, DY: Integer>() -> bool {
    match (DX::STATIC_VALUE, DY::STATIC_VALUE) {
        (Some(x), Some(y)) => x <= y,
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (None, None) => true,
    }
}

const INVERSE_MODE_PRECEDES<DX: Integer, DY: Integer>: bool = inverse_mode_precedes::<DX, DY>();

/// Single insertion-sort step placing `ModeTriple<SX, DX, PX>` before (`PRECEDES == true`)
/// or after (`false`) the head of sorted suffix `Self`.
#[doc(hidden)]
pub const trait InsertSortedTripleStep<SX, DX, PX, const PRECEDES: bool>: FlatTuple {
    type Output: FlatTuple;
    fn insert_sorted_triple_step(self, mode: ModeTriple<SX, DX, PX>) -> Self::Output;
}

/// Inserts `ModeTriple<SX, DX, PX>` into sorted [`ModeTriple`] tuple `Self` by ascending static stride.
#[doc(hidden)]
pub const trait InsertSortedTriple<SX: ShapeScalar, DX: Integer, PX: ShapeScalar>:
    FlatTuple
{
    type Output: FlatTuple;
    fn insert_sorted_triple(self, mode: ModeTriple<SX, DX, PX>) -> Self::Output;
}

const impl<SX: ShapeScalar, DX: Integer, PX: ShapeScalar> InsertSortedTriple<SX, DX, PX> for Nil {
    type Output = Til<ModeTriple<SX, DX, PX>, Nil>;
    fn insert_sorted_triple(self, mode: ModeTriple<SX, DX, PX>) -> Self::Output {
        Til(mode, Nil)
    }
}

const impl<SX, DX, PX, SY, DY, PY, Rest> InsertSortedTripleStep<SX, DX, PX, true>
    for Til<ModeTriple<SY, DY, PY>, Rest>
where
    SX: ShapeScalar,
    DX: Integer,
    PX: ShapeScalar,
    SY: ShapeScalar,
    DY: Integer,
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
    DX: Integer,
    PX: ShapeScalar,
    SY: ShapeScalar,
    DY: Integer,
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
    DX: Integer,
    PX: ShapeScalar,
    SY: ShapeScalar,
    DY: Integer,
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
    DH: Integer,
    PH: ShapeScalar,
    T: FlatTuple + [const] SortInverseModes,
    T::Output: [const] InsertSortedTriple<SH, DH, PH>,
{
    type Output = <T::Output as InsertSortedTriple<SH, DH, PH>>::Output;
    fn sort_inverse_modes(self) -> Self::Output {
        self.1.sort_inverse_modes().insert_sorted_triple(self.0)
    }
}

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
    -> RightFoldAction
{
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
                    if c.checked_mul(s).is_some() {
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
    type Shape =
        <Self as RightInverseFoldStep<CurrD, { RIGHT_FOLD_ACTION::<S, D, CurrD> }>>::Shape;
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
        let next_curr = match curr_d.dynamic_extent() {
            Some(c) => DynamicExtent(c.checked_mul(self.0.shape.value())),
            None => DynamicExtent(None),
        };
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
            let next_curr = match curr_opt {
                Some(c) => DynamicExtent(c.checked_mul(s)),
                None => DynamicExtent(None),
            };
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

const fn left_fold_action<
    S: ShapeScalar,
    D: Integer,
    PrevD: ShapeScalar,
    PrevS: ShapeScalar,
>() -> LeftFoldAction {
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
    Self: [const] LeftInverseFoldStep<
            PrevD,
            PrevS,
            PrevP,
            { LEFT_FOLD_ACTION::<S, D, PrevD, PrevS> },
        >,
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

/// Selects the per-axis fold & finalization (`RIGHT = true` for right inverse, `false` for left).
#[doc(hidden)]
pub const trait InvertSortedAxis<const RIGHT: bool>: FlatTuple {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn invert_sorted_axis(self) -> Layout<Self::Shape, Self::Stride>;
}

const impl<M> InvertSortedAxis<true> for M
where
    M: FlatTuple + [const] RightInverseFold<Int<1>>,
    Layout<
        <M as RightInverseFold<Int<1>>>::Shape,
        <M as RightInverseFold<Int<1>>>::Stride,
    >: [const] FinalizeRightAxis,
{
    type Shape = <Layout<
        <M as RightInverseFold<Int<1>>>::Shape,
        <M as RightInverseFold<Int<1>>>::Stride,
    > as FinalizeRightAxis>::Shape;
    type Stride = <Layout<
        <M as RightInverseFold<Int<1>>>::Shape,
        <M as RightInverseFold<Int<1>>>::Stride,
    > as FinalizeRightAxis>::Stride;

    fn invert_sorted_axis(self) -> Layout<Self::Shape, Self::Stride> {
        self.right_inverse_fold(Int).finalize_right_axis()
    }
}

const impl<M> InvertSortedAxis<false> for M
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

    fn invert_sorted_axis(self) -> Layout<Self::Shape, Self::Stride> {
        let raw = self.left_inverse_fold(Int, Int, Int);
        raw.finalize_left_axis()
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
    ) -> Layout<Self::Shape, Self::Stride>;
}

const impl<const RIGHT: bool, FS: FlatTuple, FD: FlatTuple, FP: FlatTuple>
    BuildInverse<RIGHT, FS, FD, FP> for Nil
{
    type Shape = Nil;
    type Stride = Nil;

    fn build_inverse(self, _: FS, _: FD, _: FP) -> Layout<Nil, Nil> {
        Layout::new(Nil, Nil)
    }
}

const impl<const RIGHT: bool, FS, FD, FP> BuildInverse<RIGHT, FS, FD, FP> for Int<1>
where
    FS: FlatTuple + [const] FilterInverseModes<ModeEnd, FD, FP>,
    FD: FlatTuple,
    FP: FlatTuple,
    <FS as FilterInverseModes<ModeEnd, FD, FP>>::Output: [const] SortInverseModes,
    SortedAxisModes<ModeEnd, FS, FD, FP>: [const] InvertSortedAxis<RIGHT>,
{
    type Shape = <SortedAxisModes<ModeEnd, FS, FD, FP> as InvertSortedAxis<RIGHT>>::Shape;
    type Stride = <SortedAxisModes<ModeEnd, FS, FD, FP> as InvertSortedAxis<RIGHT>>::Stride;

    fn build_inverse(
        self,
        flat_s: FS,
        flat_d: FD,
        flat_p: FP,
    ) -> Layout<Self::Shape, Self::Stride> {
        flat_s
            .filter_inverse_modes(flat_d, flat_p)
            .sort_inverse_modes()
            .invert_sorted_axis()
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
    SortedAxisModes<P, FS, FD, FP>: [const] InvertSortedAxis<RIGHT>,
{
    type Shape = <SortedAxisModes<P, FS, FD, FP> as InvertSortedAxis<RIGHT>>::Shape;
    type Stride = <SortedAxisModes<P, FS, FD, FP> as InvertSortedAxis<RIGHT>>::Stride;

    fn build_inverse(
        self,
        flat_s: FS,
        flat_d: FD,
        flat_p: FP,
    ) -> Layout<Self::Shape, Self::Stride> {
        flat_s
            .filter_inverse_modes(flat_d, flat_p)
            .sort_inverse_modes()
            .invert_sorted_axis()
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
    ) -> Layout<Self::Shape, Self::Stride> {
        let head = self.0.build_inverse(flat_s, flat_d, flat_p);
        let tail = self.1.build_inverse(flat_s, flat_d, flat_p);
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
    bases.build_inverse(flat_s, flat_d, flat_p)
}

const impl<S, D> RightInverse for Layout<S, D>
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
            true,
            CoalescedFlatS<S, D>,
            CoalescedFlatD<S, D>,
            CoalescedFlatP<S, D>,
        >,
{
    type Output = BuiltInverseLayout<true, S, D>;

    fn right_inverse(self) -> Self::Output {
        invert_layout::<true, S, D>(self)
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
        invert_layout::<false, S, D>(self)
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
pub const fn right_inverse<A: [const] RightInverse>(a: A) -> A::Output {
    a.right_inverse()
}

/// Computes the left inverse layout of a [`Layout`] or [`Tiler`].
pub const fn left_inverse<A: [const] LeftInverse>(a: A) -> A::Output {
    a.left_inverse()
}
