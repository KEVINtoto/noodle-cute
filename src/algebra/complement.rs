//! Layout complement algebra ($A^*$ and extended $A^* \text{ to } M$).
//!
//! The complement of a [`Layout`] $A$ produces a layout $R$ whose image fills the
//! codomain holes of $A$:
//!
//! 1. **Weak congruence with codomain**: $\mathrm{coprofile}(A) \preceq_{\text{weak}} \mathrm{shape}(R)$.
//! 2. **Strictly ordered (including extended domain)**: $R(i - 1) < R(i)$ for all $i \ge 1$.
//! 3. **Disjoint from $A$**: $R(i) \neq A(j)$ for all $i \ge 1$ and $j \in [0, |A|)$.
//!
//! When $A$'s sorted non-degenerate strides form a divisible chain ($d_{k-1} s_{k-1} \mid d_k$),
//! `make_layout!(A, R)` tiles the codomain bijectively (up to stride-0 modes).
//!
//! # Main Interfaces
//!
//! - [`Complement`] / [`complement`] / [`Layout::complement`]:
//!   Minimal (free) complement preserving the trailing `1 : c_m` extension marker.
//! - [`ComplementWith`] / [`complement_with`] / [`Layout::complement_with`]:
//!   Complement extended to cover a target `extend: Shape`.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::complement::{complement, complement_with};
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let l = Layout::new(Int::<4>, Int::<2>);
//! assert_eq!(
//!     complement(l),
//!     Layout::new(int_tuple![Int::<2>, Int::<1>], int_tuple![Int::<1>, Int::<8>])
//! );
//! assert_eq!(
//!     complement_with(l, Int::<20>),
//!     Layout::new(int_tuple![Int::<2>, Int::<3>], int_tuple![Int::<1>, Int::<8>])
//! );
//! ```
//!
//! # Notes
//!
//! Non-injective static layouts are rejected at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::complement::complement;
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let _ = complement(Layout::new(
//!     int_tuple![Int::<4>, Int::<2>],
//!     int_tuple![Int::<1>, Int::<2>],
//! ));
//! ```
//!
//! Negative static strides are rejected at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::complement::complement;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let _ = complement(Layout::new(Int::<4>, Int::<-2>));
//! ```
//!
//! Zero static extent with non-zero stride is rejected at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::complement::complement;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let _ = complement(Layout::new(Int::<0>, Int::<2>));
//! ```
//!
//! Static running-stride overflow is rejected at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::complement::complement;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let _ = complement(Layout::new(Int::<2>, Int::<2_000_000_000>));
//! ```
//!
//! Multi-term basis combinations are rejected at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::complement::complement;
//! use noodle_cute::e;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let _ = complement(Layout::new(Int::<4>, e![0] + e![1]));
//! ```
//!
//! Incompatible stride depths (mixing non-zero scalar with basis stride) fail at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::complement::complement;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::{e, int_tuple, tuple};
//!
//! let _ = complement(Layout::new(
//!     int_tuple![Int::<2>, Int::<3>],
//!     tuple![e![1], Int::<5>],
//! ));
//! ```
//!
//! Extending with a shape shallower than `coprofile(A)` fails at compile time:
//!
//! ```compile_fail,E0308
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::complement::complement_with;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::{e, int_tuple, tuple};
//!
//! let l = Layout::new(int_tuple![Int::<2>, Int::<3>], tuple![e![0], e![1]]);
//! let _ = complement_with(l, Int::<16>);
//! ```

use core::marker::ConstParamTy;

use crate::basis::{
    Basis, BasisPath, ComparePath, MakeBasisLikeOutput, MakeBasisLikeTopProfile, ModeEnd,
    PathOrdering, make_basis_like,
};
use crate::integer::{IS_NON_NEG, Int, Integer, ceil_div, div, int, mul, mul_maybe_overflows};
use crate::layout::{Coprofile, Layout, Tiler, tiler_to_layout};
use crate::shape::{Shape, ShapeScalar, ValidShapeExtent};
use crate::stride::{PrefixProduct, ScaleStride, Stride, StrideScalar, prefix_product_with};
use crate::tuple::{
    Append, FlatTuple, Flatten, Item, Leaf, LeafProfile, Nil, NilProfile, Til, Tuple, TupleProfile,
};

use super::coalesce::{Coalesce, DropTrailingUnit};

/// Scalar `(shape, stride)` mode pair collected along a single codomain axis.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ModePair<S, D> {
    pub shape: S,
    pub stride: D,
}

const impl<S: Item, D: Item> Item for ModePair<S, D> {
    const RANK: int = 1;
    const DEPTH: int = 0;
    const LEAVES: int = 1;
    const IS_STATIC: bool = S::IS_STATIC && D::IS_STATIC;
    type Profile = LeafProfile;
}

/// Compile-time decision for filtering one flattened `(S, D)` pair against `TargetP`.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum AxisFilterAction {
    /// Drop static-zero strides, static size-1 modes on this axis, or modes on another axis.
    Skip,
    /// Retain `(S, D::Value)` for this axis.
    Keep,
    /// Incompatible stride depth (prefix conflict between `D::Path` and `TargetP`).
    PrefixConflict,
}

const fn axis_filter_action<S, D, TargetP>() -> AxisFilterAction
where
    S: ShapeScalar,
    D: StrideScalar,
    D::Value: Integer,
    TargetP: BasisPath,
    D::Path: ComparePath<TargetP>,
{
    if matches!(D::Value::STATIC_VALUE, Some(0)) {
        return AxisFilterAction::Skip;
    }
    match D::Path::ORDER {
        PathOrdering::Equal => {
            if matches!(S::STATIC_VALUE, Some(1)) {
                AxisFilterAction::Skip
            } else {
                AxisFilterAction::Keep
            }
        }
        PathOrdering::Less | PathOrdering::Greater => AxisFilterAction::Skip,
        PathOrdering::PrefixConflict | PathOrdering::Dispatch => AxisFilterAction::PrefixConflict,
    }
}

const AXIS_FILTER_ACTION<S: ShapeScalar, D: StrideScalar, TargetP: BasisPath>: AxisFilterAction =
    axis_filter_action::<S, D, TargetP>()
where
    D::Value: Integer,
    D::Path: ComparePath<TargetP>;

/// Applies one [`AxisFilterAction`] to prepend `(S, D::Value)` onto filtered suffix `Self`.
#[doc(hidden)]
pub const trait FilterAxisStep<S, D, const ACTION: AxisFilterAction>: FlatTuple {
    type Output: FlatTuple;
    fn filter_axis_step(self, s: S, d: D) -> Self::Output;
}

const impl<S: Copy, D: Copy, Rest: FlatTuple> FilterAxisStep<S, D, { AxisFilterAction::Skip }>
    for Rest
{
    type Output = Rest;
    fn filter_axis_step(self, _: S, _: D) -> Self::Output {
        self
    }
}

const impl<S, D, Rest> FilterAxisStep<S, D, { AxisFilterAction::Keep }> for Rest
where
    S: ShapeScalar,
    D: [const] StrideScalar,
    D::Value: [const] Integer,
    Rest: FlatTuple,
{
    type Output = Til<ModePair<S, D::Value>, Rest>;
    fn filter_axis_step(self, s: S, d: D) -> Self::Output {
        Til(
            ModePair {
                shape: s,
                stride: d.value(),
            },
            self,
        )
    }
}

/// Filters flattened `(shape, stride)` pairs to those active on codomain axis `TargetP`.
#[doc(hidden)]
pub const trait FilterAxisModes<TargetP: BasisPath, FD: FlatTuple>: FlatTuple {
    type Output: FlatTuple;
    fn filter_axis_modes(self, strides: FD) -> Self::Output;
}

const impl<TargetP: BasisPath> FilterAxisModes<TargetP, Nil> for Nil {
    type Output = Nil;
    fn filter_axis_modes(self, _: Nil) -> Nil {
        Nil
    }
}

const impl<TargetP, S, TS, D, TD> FilterAxisModes<TargetP, Til<D, TD>> for Til<S, TS>
where
    TargetP: BasisPath,
    S: ShapeScalar,
    TS: FlatTuple + [const] FilterAxisModes<TargetP, TD>,
    D: [const] StrideScalar,
    D::Value: [const] Integer,
    TD: FlatTuple,
    D::Path: ComparePath<TargetP>,
    <TS as FilterAxisModes<TargetP, TD>>::Output:
        [const] FilterAxisStep<S, D, { AXIS_FILTER_ACTION::<S, D, TargetP> }>,
{
    type Output = <<TS as FilterAxisModes<TargetP, TD>>::Output as FilterAxisStep<
        S,
        D,
        { AXIS_FILTER_ACTION::<S, D, TargetP> },
    >>::Output;

    fn filter_axis_modes(self, strides: Til<D, TD>) -> Self::Output {
        let rest = self.1.filter_axis_modes(strides.1);
        rest.filter_axis_step(self.0, strides.0)
    }
}

const fn mode_precedes<DX: Integer, DY: Integer>() -> bool {
    match (DX::STATIC_VALUE, DY::STATIC_VALUE) {
        (Some(x), Some(y)) => x <= y,
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (None, None) => true,
    }
}

const MODE_PRECEDES<DX: Integer, DY: Integer>: bool = mode_precedes::<DX, DY>();

/// Single insertion-sort comparison step inserting `ModePair<SX, DX>` before (`PRECEDES == true`)
/// or after (`false`) the head of sorted suffix `Self`.
#[doc(hidden)]
pub const trait InsertSortedModeStep<SX, DX, const PRECEDES: bool>: FlatTuple {
    type Output: FlatTuple;
    fn insert_sorted_step(self, mode: ModePair<SX, DX>) -> Self::Output;
}

/// Inserts `ModePair<SX, DX>` into sorted [`ModePair`] tuple `Self` by ascending static stride.
#[doc(hidden)]
pub const trait InsertSortedMode<SX: ShapeScalar, DX: Integer>: FlatTuple {
    type Output: FlatTuple;
    fn insert_sorted(self, mode: ModePair<SX, DX>) -> Self::Output;
}

const impl<SX: ShapeScalar, DX: Integer> InsertSortedMode<SX, DX> for Nil {
    type Output = Til<ModePair<SX, DX>, Nil>;
    fn insert_sorted(self, mode: ModePair<SX, DX>) -> Self::Output {
        Til(mode, Nil)
    }
}

const impl<SX, DX, SY, DY, Rest> InsertSortedModeStep<SX, DX, true> for Til<ModePair<SY, DY>, Rest>
where
    SX: ShapeScalar,
    DX: Integer,
    SY: ShapeScalar,
    DY: Integer,
    Rest: FlatTuple,
{
    type Output = Til<ModePair<SX, DX>, Self>;
    fn insert_sorted_step(self, mode: ModePair<SX, DX>) -> Self::Output {
        Til(mode, self)
    }
}

const impl<SX, DX, SY, DY, Rest> InsertSortedModeStep<SX, DX, false> for Til<ModePair<SY, DY>, Rest>
where
    SX: ShapeScalar,
    DX: Integer,
    SY: ShapeScalar,
    DY: Integer,
    Rest: FlatTuple + [const] InsertSortedMode<SX, DX>,
{
    type Output = Til<ModePair<SY, DY>, <Rest as InsertSortedMode<SX, DX>>::Output>;
    fn insert_sorted_step(self, mode: ModePair<SX, DX>) -> Self::Output {
        Til(self.0, self.1.insert_sorted(mode))
    }
}

const impl<SX, DX, SY, DY, Rest> InsertSortedMode<SX, DX> for Til<ModePair<SY, DY>, Rest>
where
    SX: ShapeScalar,
    DX: Integer,
    SY: ShapeScalar,
    DY: Integer,
    Rest: FlatTuple,
    Self: [const] InsertSortedModeStep<SX, DX, { MODE_PRECEDES::<DX, DY> }>,
{
    type Output = <Self as InsertSortedModeStep<SX, DX, { MODE_PRECEDES::<DX, DY> }>>::Output;
    fn insert_sorted(self, mode: ModePair<SX, DX>) -> Self::Output {
        self.insert_sorted_step(mode)
    }
}

/// Stably sorts a flat list of [`ModePair`]s by ascending static stride,
/// placing dynamic strides after static ones in their original left-to-right order.
#[doc(hidden)]
pub const trait SortModes: FlatTuple {
    type Output: FlatTuple;
    fn sort_modes(self) -> Self::Output;
}

const impl SortModes for Nil {
    type Output = Nil;
    fn sort_modes(self) -> Nil {
        Nil
    }
}

const impl<SH, DH, T> SortModes for Til<ModePair<SH, DH>, T>
where
    SH: ShapeScalar,
    DH: Integer,
    T: FlatTuple + [const] SortModes,
    T::Output: [const] InsertSortedMode<SH, DH>,
{
    type Output = <T::Output as InsertSortedMode<SH, DH>>::Output;
    fn sort_modes(self) -> Self::Output {
        self.1.sort_modes().insert_sorted(self.0)
    }
}

/// Prepends one complement `(G, C)` mode onto a non-empty flat complement suffix layout.
#[doc(hidden)]
pub const trait PrependComplementMode<G: ShapeScalar, C: StrideScalar>: Copy {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn prepend_complement_mode(self, gap: G, stride: C) -> Layout<Self::Shape, Self::Stride>;
}

const impl<ST, DT, G, C> PrependComplementMode<G, C> for Layout<ST, DT>
where
    ST: ShapeScalar,
    DT: StrideScalar + Stride<ST>,
    G: ShapeScalar,
    C: StrideScalar + Stride<G>,
{
    type Shape = Til<G, Til<ST, Nil>>;
    type Stride = Til<C, Til<DT, Nil>>;

    fn prepend_complement_mode(self, gap: G, stride: C) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(
            Til(gap, Til(self.shape, Nil)),
            Til(stride, Til(self.stride, Nil)),
        )
    }
}

const impl<HS, TS, HD, TD, G, C> PrependComplementMode<G, C> for Layout<Til<HS, TS>, Til<HD, TD>>
where
    HS: ShapeScalar,
    TS: Tuple + Shape,
    HD: StrideScalar + Stride<HS>,
    TD: Tuple + Stride<TS>,
    G: ShapeScalar,
    C: StrideScalar + Stride<G>,
{
    type Shape = Til<G, Til<HS, TS>>;
    type Stride = Til<C, Til<HD, TD>>;

    fn prepend_complement_mode(self, gap: G, stride: C) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(Til(gap, self.shape), Til(stride, self.stride))
    }
}

/// Compile-time action for one mode in the sorted 1D complement fold.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum FoldAction {
    /// Static `D >= CurrD`, static `S >= 2`, `D / CurrD == 1`, static `D * S`.
    StaticSkipGapStaticNext,
    /// Static `D >= CurrD`, static `S >= 2`, `D / CurrD > 1`, static `D * S`.
    StaticEmitGapStaticNext,
    /// Static `D >= CurrD`, dynamic `S`, `D / CurrD == 1`, dynamic `D * S`.
    StaticSkipGapDynamicNext,
    /// Static `D >= CurrD`, dynamic `S`, `D / CurrD > 1`, dynamic `D * S`.
    StaticEmitGapDynamicNext,
    /// At least one of `D` or `CurrD` is dynamic: validate and compute gap/next at runtime.
    DynamicStep,
    /// Statically invalid (`D < CurrD`, `D <= 0`, `S == 0`, or `D * S` overflows `int`).
    Invalid,
}

const fn fold_action<S: ShapeScalar, D: Integer, CurrD: ShapeScalar>() -> FoldAction {
    match (D::STATIC_VALUE, CurrD::STATIC_VALUE) {
        (Some(d), Some(c)) => {
            if c <= 0 || d < c {
                return FoldAction::Invalid;
            }
            let gap = div(d, c);
            match S::STATIC_VALUE {
                Some(s) => {
                    if s <= 0 || mul_maybe_overflows(d, s).is_none() {
                        return FoldAction::Invalid;
                    }
                    if gap == 1 {
                        FoldAction::StaticSkipGapStaticNext
                    } else {
                        FoldAction::StaticEmitGapStaticNext
                    }
                }
                None => {
                    if gap == 1 {
                        FoldAction::StaticSkipGapDynamicNext
                    } else {
                        FoldAction::StaticEmitGapDynamicNext
                    }
                }
            }
        }
        _ => {
            if let Some(d) = D::STATIC_VALUE
                && d < 0
            {
                return FoldAction::Invalid;
            }
            if let (Some(d), Some(s)) = (D::STATIC_VALUE, S::STATIC_VALUE)
                && (s <= 0 || mul_maybe_overflows(d, s).is_none())
            {
                return FoldAction::Invalid;
            }
            FoldAction::DynamicStep
        }
    }
}

const FOLD_ACTION<S: ShapeScalar, D: Integer, CurrD: ShapeScalar>: FoldAction =
    fold_action::<S, D, CurrD>();

const STATIC_GAP<D: Integer, CurrD: ShapeScalar>: int = div(
    D::STATIC_VALUE.expect("expected static stride in complement"),
    CurrD::STATIC_VALUE.expect("expected static running stride in complement"),
);

const STATIC_NEXT_D<D: Integer, S: ShapeScalar>: int = mul(
    D::STATIC_VALUE.expect("expected static stride in complement"),
    S::STATIC_VALUE.expect("expected static shape in complement"),
);

/// Executes one [`FoldAction`] step of [`ComplementFold`] against running stride `CurrD`.
#[doc(hidden)]
pub const trait ComplementFoldStep<CurrD: ShapeScalar, const ACTION: FoldAction>:
    FlatTuple
{
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn fold_step(self, curr_d: CurrD) -> Layout<Self::Shape, Self::Stride>;
}

/// Folds a sorted flat list of scalar [`ModePair`]s into the minimal 1D complement layout,
/// eagerly omitting statically trivial `gap == 1` modes while retaining trailing `1 : c_m`.
#[doc(hidden)]
pub const trait ComplementFold<CurrD: ShapeScalar>: FlatTuple {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn complement_fold(self, curr_d: CurrD) -> Layout<Self::Shape, Self::Stride>;
}

const impl<CurrD: [const] ShapeScalar> ComplementFold<CurrD> for Nil {
    type Shape = Int<1>;
    type Stride = CurrD;

    fn complement_fold(self, curr_d: CurrD) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(Int, curr_d)
    }
}

const impl<S, D, Rest, CurrD> ComplementFold<CurrD> for Til<ModePair<S, D>, Rest>
where
    S: ShapeScalar,
    D: Integer,
    Rest: FlatTuple,
    CurrD: ShapeScalar,
    Self: [const] ComplementFoldStep<CurrD, { FOLD_ACTION::<S, D, CurrD> }>,
{
    type Shape = <Self as ComplementFoldStep<CurrD, { FOLD_ACTION::<S, D, CurrD> }>>::Shape;
    type Stride = <Self as ComplementFoldStep<CurrD, { FOLD_ACTION::<S, D, CurrD> }>>::Stride;

    fn complement_fold(self, curr_d: CurrD) -> Layout<Self::Shape, Self::Stride> {
        self.fold_step(curr_d)
    }
}

const impl<S, D, Rest, CurrD> ComplementFoldStep<CurrD, { FoldAction::StaticSkipGapStaticNext }>
    for Til<ModePair<S, D>, Rest>
where
    S: ShapeScalar,
    D: Integer,
    CurrD: ShapeScalar,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_NEXT_D::<D, S> }> }>,
    Rest: FlatTuple + [const] ComplementFold<Int<{ STATIC_NEXT_D::<D, S> }>>,
{
    type Shape = <Rest as ComplementFold<Int<{ STATIC_NEXT_D::<D, S> }>>>::Shape;
    type Stride = <Rest as ComplementFold<Int<{ STATIC_NEXT_D::<D, S> }>>>::Stride;

    fn fold_step(self, _: CurrD) -> Layout<Self::Shape, Self::Stride> {
        self.1.complement_fold(Int)
    }
}

const impl<S, D, Rest, CurrD> ComplementFoldStep<CurrD, { FoldAction::StaticEmitGapStaticNext }>
    for Til<ModePair<S, D>, Rest>
where
    S: ShapeScalar,
    D: Integer,
    CurrD: [const] ShapeScalar,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_NEXT_D::<D, S> }> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_GAP::<D, CurrD> }> }>,
    Rest: FlatTuple + [const] ComplementFold<Int<{ STATIC_NEXT_D::<D, S> }>>,
    Layout<
        <Rest as ComplementFold<Int<{ STATIC_NEXT_D::<D, S> }>>>::Shape,
        <Rest as ComplementFold<Int<{ STATIC_NEXT_D::<D, S> }>>>::Stride,
    >: [const] PrependComplementMode<Int<{ STATIC_GAP::<D, CurrD> }>, CurrD>,
{
    type Shape = <Layout<
        <Rest as ComplementFold<Int<{ STATIC_NEXT_D::<D, S> }>>>::Shape,
        <Rest as ComplementFold<Int<{ STATIC_NEXT_D::<D, S> }>>>::Stride,
    > as PrependComplementMode<Int<{ STATIC_GAP::<D, CurrD> }>, CurrD>>::Shape;
    type Stride = <Layout<
        <Rest as ComplementFold<Int<{ STATIC_NEXT_D::<D, S> }>>>::Shape,
        <Rest as ComplementFold<Int<{ STATIC_NEXT_D::<D, S> }>>>::Stride,
    > as PrependComplementMode<Int<{ STATIC_GAP::<D, CurrD> }>, CurrD>>::Stride;

    fn fold_step(self, curr_d: CurrD) -> Layout<Self::Shape, Self::Stride> {
        let tail = self.1.complement_fold(Int);
        tail.prepend_complement_mode(Int, curr_d)
    }
}

const impl<S, D, Rest, CurrD> ComplementFoldStep<CurrD, { FoldAction::StaticSkipGapDynamicNext }>
    for Til<ModePair<S, D>, Rest>
where
    S: [const] ShapeScalar,
    D: [const] Integer,
    CurrD: ShapeScalar,
    Rest: FlatTuple + [const] ComplementFold<int>,
{
    type Shape = <Rest as ComplementFold<int>>::Shape;
    type Stride = <Rest as ComplementFold<int>>::Stride;

    fn fold_step(self, _: CurrD) -> Layout<Self::Shape, Self::Stride> {
        let s = self.0.shape.value();
        assert!(s > 0, "complement extent must be positive");
        let next_d = mul(self.0.stride.value(), s);
        self.1.complement_fold(next_d)
    }
}

const impl<S, D, Rest, CurrD> ComplementFoldStep<CurrD, { FoldAction::StaticEmitGapDynamicNext }>
    for Til<ModePair<S, D>, Rest>
where
    S: [const] ShapeScalar,
    D: [const] Integer,
    CurrD: [const] ShapeScalar,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_GAP::<D, CurrD> }> }>,
    Rest: FlatTuple + [const] ComplementFold<int>,
    Layout<<Rest as ComplementFold<int>>::Shape, <Rest as ComplementFold<int>>::Stride>:
        [const] PrependComplementMode<Int<{ STATIC_GAP::<D, CurrD> }>, CurrD>,
{
    type Shape = <Layout<
        <Rest as ComplementFold<int>>::Shape,
        <Rest as ComplementFold<int>>::Stride,
    > as PrependComplementMode<Int<{ STATIC_GAP::<D, CurrD> }>, CurrD>>::Shape;
    type Stride = <Layout<
        <Rest as ComplementFold<int>>::Shape,
        <Rest as ComplementFold<int>>::Stride,
    > as PrependComplementMode<Int<{ STATIC_GAP::<D, CurrD> }>, CurrD>>::Stride;

    fn fold_step(self, curr_d: CurrD) -> Layout<Self::Shape, Self::Stride> {
        let s = self.0.shape.value();
        assert!(s > 0, "complement extent must be positive");
        let next_d = mul(self.0.stride.value(), s);
        let tail = self.1.complement_fold(next_d);
        tail.prepend_complement_mode(Int, curr_d)
    }
}

const impl<S, D, Rest, CurrD> ComplementFoldStep<CurrD, { FoldAction::DynamicStep }>
    for Til<ModePair<S, D>, Rest>
where
    S: [const] ShapeScalar,
    D: [const] Integer,
    CurrD: [const] ShapeScalar,
    Rest: FlatTuple + [const] ComplementFold<int>,
    Layout<<Rest as ComplementFold<int>>::Shape, <Rest as ComplementFold<int>>::Stride>:
        [const] PrependComplementMode<int, CurrD>,
{
    type Shape = <Layout<
        <Rest as ComplementFold<int>>::Shape,
        <Rest as ComplementFold<int>>::Stride,
    > as PrependComplementMode<int, CurrD>>::Shape;
    type Stride = <Layout<
        <Rest as ComplementFold<int>>::Shape,
        <Rest as ComplementFold<int>>::Stride,
    > as PrependComplementMode<int, CurrD>>::Stride;

    fn fold_step(self, curr_d: CurrD) -> Layout<Self::Shape, Self::Stride> {
        let s = self.0.shape.value();
        let c = curr_d.value();
        let d = self.0.stride.value();
        assert!(s >= 0, "shape extent must be non-negative");
        if d == 0 || s == 1 {
            let tail = self.1.complement_fold(c);
            return tail.prepend_complement_mode(1, curr_d);
        }
        assert!(s > 0, "complement extent must be positive");
        assert!(d > 0, "complement stride must be positive");
        assert!(d >= c, "complement requires non-overlapping sorted strides");
        let gap = div(d, c);
        let next_d = mul(d, s);
        let tail = self.1.complement_fold(next_d);
        tail.prepend_complement_mode(gap, curr_d)
    }
}

/// Walks the codomain basis tree (`make_basis_like(coprofile)`) and assembles the
/// per-axis minimal complements into a single [`Layout`].
#[doc(hidden)]
pub const trait BuildComplement<FS: FlatTuple, FD: FlatTuple>: Item {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn build_complement(self, flat_s: FS, flat_d: FD) -> Layout<Self::Shape, Self::Stride>;
}

const impl<FS: FlatTuple, FD: FlatTuple> BuildComplement<FS, FD> for Nil {
    type Shape = Nil;
    type Stride = Nil;

    fn build_complement(self, _: FS, _: FD) -> Layout<Nil, Nil> {
        Layout::new(Nil, Nil)
    }
}

const impl<FS, FD> BuildComplement<FS, FD> for Int<1>
where
    FS: FlatTuple + [const] FilterAxisModes<ModeEnd, FD>,
    FD: FlatTuple,
    <FS as FilterAxisModes<ModeEnd, FD>>::Output: [const] SortModes,
    <<FS as FilterAxisModes<ModeEnd, FD>>::Output as SortModes>::Output:
        [const] ComplementFold<Int<1>>,
{
    type Shape =
        <<<FS as FilterAxisModes<ModeEnd, FD>>::Output as SortModes>::Output as ComplementFold<
            Int<1>,
        >>::Shape;
    type Stride =
        <<<FS as FilterAxisModes<ModeEnd, FD>>::Output as SortModes>::Output as ComplementFold<
            Int<1>,
        >>::Stride;

    fn build_complement(self, flat_s: FS, flat_d: FD) -> Layout<Self::Shape, Self::Stride> {
        flat_s
            .filter_axis_modes(flat_d)
            .sort_modes()
            .complement_fold(Int)
    }
}

const impl<P, FS, FD> BuildComplement<FS, FD> for Basis<P>
where
    P: [const] BasisPath,
    Self: Leaf,
    FS: FlatTuple + [const] FilterAxisModes<P, FD>,
    FD: FlatTuple,
    <FS as FilterAxisModes<P, FD>>::Output: [const] SortModes,
    <<FS as FilterAxisModes<P, FD>>::Output as SortModes>::Output: [const] ComplementFold<Int<1>>,
    <<<FS as FilterAxisModes<P, FD>>::Output as SortModes>::Output as ComplementFold<Int<1>>>::Stride:
        [const] ScaleStride<Self>,
    <<<<FS as FilterAxisModes<P, FD>>::Output as SortModes>::Output as ComplementFold<Int<1>>>::Stride as ScaleStride<
        Self,
    >>::Output: Stride<
        <<<FS as FilterAxisModes<P, FD>>::Output as SortModes>::Output as ComplementFold<Int<1>>>::Shape,
    >,
{
    type Shape = <<<FS as FilterAxisModes<P, FD>>::Output as SortModes>::Output as ComplementFold<
        Int<1>,
    >>::Shape;
    type Stride = <<<<FS as FilterAxisModes<P, FD>>::Output as SortModes>::Output as ComplementFold<
        Int<1>,
    >>::Stride as ScaleStride<Self>>::Output;

    fn build_complement(self, flat_s: FS, flat_d: FD) -> Layout<Self::Shape, Self::Stride> {
        let axis = flat_s
            .filter_axis_modes(flat_d)
            .sort_modes()
            .complement_fold(Int);
        Layout::new(axis.shape, axis.stride.scale_stride(self))
    }
}

const impl<H, T, FS, FD> BuildComplement<FS, FD> for Til<H, T>
where
    FS: FlatTuple,
    FD: FlatTuple,
    H: [const] BuildComplement<FS, FD>,
    T: Tuple + [const] BuildComplement<FS, FD>,
    T::Shape: Tuple + Shape,
    T::Stride: Tuple + Stride<T::Shape>,
{
    type Shape = Til<H::Shape, T::Shape>;
    type Stride = Til<H::Stride, T::Stride>;

    fn build_complement(self, flat_s: FS, flat_d: FD) -> Layout<Self::Shape, Self::Stride> {
        let head = self.0.build_complement(flat_s, flat_d);
        let tail = self.1.build_complement(flat_s, flat_d);
        Layout::new(Til(head.shape, tail.shape), Til(head.stride, tail.stride))
    }
}

/// Splits a 1D minimal complement layout into `(prefix_shape, prefix_stride, last_stride)`.
#[doc(hidden)]
pub const trait SplitLastMode: Copy {
    type PrefixShape: Tuple + Shape;
    type PrefixStride: Tuple + Stride<Self::PrefixShape>;
    type LastStride: [const] StrideScalar;

    fn split_last_mode(self) -> (Self::PrefixShape, Self::PrefixStride, Self::LastStride);
}

const impl<S, D> SplitLastMode for Layout<S, D>
where
    S: ShapeScalar,
    D: [const] StrideScalar + Stride<S>,
{
    type PrefixShape = Nil;
    type PrefixStride = Nil;
    type LastStride = D;

    fn split_last_mode(self) -> (Nil, Nil, D) {
        (Nil, Nil, self.stride)
    }
}

const impl<S, D> SplitLastMode for Layout<Til<S, Nil>, Til<D, Nil>>
where
    S: ShapeScalar,
    D: [const] StrideScalar + Stride<S>,
{
    type PrefixShape = Nil;
    type PrefixStride = Nil;
    type LastStride = D;

    fn split_last_mode(self) -> (Nil, Nil, D) {
        (Nil, Nil, self.stride.0)
    }
}

const impl<S0, S1, TS, D0, D1, TD> SplitLastMode
    for Layout<Til<S0, Til<S1, TS>>, Til<D0, Til<D1, TD>>>
where
    S0: ShapeScalar,
    S1: ShapeScalar,
    TS: Tuple + Shape,
    D0: StrideScalar + Stride<S0>,
    D1: StrideScalar + Stride<S1>,
    TD: Tuple + Stride<TS>,
    Layout<Til<S1, TS>, Til<D1, TD>>: [const] SplitLastMode,
{
    type PrefixShape = Til<S0, <Layout<Til<S1, TS>, Til<D1, TD>> as SplitLastMode>::PrefixShape>;
    type PrefixStride = Til<D0, <Layout<Til<S1, TS>, Til<D1, TD>> as SplitLastMode>::PrefixStride>;
    type LastStride = <Layout<Til<S1, TS>, Til<D1, TD>> as SplitLastMode>::LastStride;

    fn split_last_mode(self) -> (Self::PrefixShape, Self::PrefixStride, Self::LastStride) {
        let (tail_s, tail_d, last_d) = Layout::new(self.shape.1, self.stride.1).split_last_mode();
        (
            Til(self.shape.0, tail_s),
            Til(self.stride.0, tail_d),
            last_d,
        )
    }
}

/// Ensures a flattened `extend` shape has at least one extent leaf (`Nil -> (1,)`).
#[doc(hidden)]
pub const trait NonEmptyFlatShape: FlatTuple {
    type Output: FlatTuple + Shape;
    fn non_empty_flat_shape(self) -> Self::Output;
}

const impl NonEmptyFlatShape for Nil {
    type Output = Til<Int<1>, Nil>;
    fn non_empty_flat_shape(self) -> Self::Output {
        Til(Int, Nil)
    }
}

const impl<H: ShapeScalar, T: FlatTuple + Shape> NonEmptyFlatShape for Til<H, T> {
    type Output = Self;
    fn non_empty_flat_shape(self) -> Self::Output {
        self
    }
}

/// Compile-time action for one leaf of the sequential ceiling-division extension loop.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum CeilStepAction {
    /// Both `S` and `SizeC` are static and positive.
    StaticBoth,
    /// `SizeC` is static `Int<1>`, `S` is dynamic `int`: `r = s`, `next_size_c = Int<1>`.
    UnitSizeC,
    /// `S` is static `Int<1>`, `SizeC` is dynamic `int`: `r = Int<1>`, `next_size_c = size_c`.
    UnitShape,
    /// General dynamic ceiling division for both `r` and `next_size_c`.
    DynamicBoth,
    /// Static zero in `S` or `SizeC`.
    Invalid,
}

const fn ceil_step_action<S: ShapeScalar, SizeC: ShapeScalar>() -> CeilStepAction {
    match (S::STATIC_VALUE, SizeC::STATIC_VALUE) {
        (Some(0), _) | (_, Some(0)) => CeilStepAction::Invalid,
        (Some(_), Some(_)) => CeilStepAction::StaticBoth,
        (None, Some(1)) => CeilStepAction::UnitSizeC,
        (Some(1), None) => CeilStepAction::UnitShape,
        _ => CeilStepAction::DynamicBoth,
    }
}

const CEIL_STEP_ACTION<S: ShapeScalar, SizeC: ShapeScalar>: CeilStepAction =
    ceil_step_action::<S, SizeC>();

const STATIC_CEIL_DIV<A: ShapeScalar, B: ShapeScalar>: int = ceil_div(
    A::STATIC_VALUE.expect("expected static numerator"),
    B::STATIC_VALUE.expect("expected static denominator"),
);

/// Executes one [`CeilStepAction`] step of [`ExtendShapeLoop`] against remaining cover `SizeC`.
#[doc(hidden)]
pub const trait ExtendShapeStep<SizeC: ShapeScalar, const ACTION: CeilStepAction>:
    FlatTuple
{
    type Output: [const] Tuple + FlatTuple + Shape;
    fn extend_shape_step(self, size_c: SizeC) -> Self::Output;
}

/// Sequential ceiling-division loop over flattened leaves of `extend`:
/// `r_i = ceil_div(s_i, size_c)`, `size_c = ceil_div(size_c, s_i)`.
#[doc(hidden)]
pub const trait ExtendShapeLoop<SizeC: ShapeScalar>: FlatTuple {
    type Output: [const] Tuple + FlatTuple + Shape;
    fn extend_shape_loop(self, size_c: SizeC) -> Self::Output;
}

const impl<SizeC: ShapeScalar> ExtendShapeLoop<SizeC> for Nil {
    type Output = Nil;
    fn extend_shape_loop(self, _: SizeC) -> Nil {
        Nil
    }
}

const impl<S, TS, SizeC> ExtendShapeLoop<SizeC> for Til<S, TS>
where
    S: ShapeScalar,
    TS: FlatTuple,
    SizeC: ShapeScalar,
    Self: [const] ExtendShapeStep<SizeC, { CEIL_STEP_ACTION::<S, SizeC> }>,
{
    type Output = <Self as ExtendShapeStep<SizeC, { CEIL_STEP_ACTION::<S, SizeC> }>>::Output;
    fn extend_shape_loop(self, size_c: SizeC) -> Self::Output {
        self.extend_shape_step(size_c)
    }
}

const impl<S, TS, SizeC> ExtendShapeStep<SizeC, { CeilStepAction::StaticBoth }> for Til<S, TS>
where
    S: ShapeScalar,
    SizeC: ShapeScalar,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_CEIL_DIV::<S, SizeC> }> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_CEIL_DIV::<SizeC, S> }> }>,
    TS: FlatTuple + [const] ExtendShapeLoop<Int<{ STATIC_CEIL_DIV::<SizeC, S> }>>,
{
    type Output = Til<
        Int<{ STATIC_CEIL_DIV::<S, SizeC> }>,
        <TS as ExtendShapeLoop<Int<{ STATIC_CEIL_DIV::<SizeC, S> }>>>::Output,
    >;

    fn extend_shape_step(self, _: SizeC) -> Self::Output {
        let tail = self.1.extend_shape_loop(Int);
        Til(Int, tail)
    }
}

const impl<S, TS, SizeC> ExtendShapeStep<SizeC, { CeilStepAction::UnitSizeC }> for Til<S, TS>
where
    S: [const] ShapeScalar,
    SizeC: ShapeScalar,
    TS: FlatTuple + [const] ExtendShapeLoop<Int<1>>,
{
    type Output = Til<S, <TS as ExtendShapeLoop<Int<1>>>::Output>;

    fn extend_shape_step(self, _: SizeC) -> Self::Output {
        assert!(
            self.0.value() > 0,
            "complement_with extend extent must be positive"
        );
        let tail = self.1.extend_shape_loop(Int);
        Til(self.0, tail)
    }
}

const impl<S, TS, SizeC> ExtendShapeStep<SizeC, { CeilStepAction::UnitShape }> for Til<S, TS>
where
    S: ShapeScalar,
    SizeC: [const] ShapeScalar,
    TS: FlatTuple + [const] ExtendShapeLoop<SizeC>,
{
    type Output = Til<Int<1>, <TS as ExtendShapeLoop<SizeC>>::Output>;

    fn extend_shape_step(self, size_c: SizeC) -> Self::Output {
        assert!(
            size_c.value() > 0,
            "complement_with running stride must be positive"
        );
        let tail = self.1.extend_shape_loop(size_c);
        Til(Int, tail)
    }
}

const impl<S, TS, SizeC> ExtendShapeStep<SizeC, { CeilStepAction::DynamicBoth }> for Til<S, TS>
where
    S: [const] ShapeScalar,
    SizeC: [const] ShapeScalar,
    TS: FlatTuple + [const] ExtendShapeLoop<int>,
{
    type Output = Til<int, <TS as ExtendShapeLoop<int>>::Output>;

    fn extend_shape_step(self, size_c: SizeC) -> Self::Output {
        let s = self.0.value();
        let c = size_c.value();
        assert!(s > 0, "complement_with extend extent must be positive");
        assert!(c > 0, "complement_with running stride must be positive");
        let r = ceil_div(s, c);
        let next_c = ceil_div(c, s);
        let tail = self.1.extend_shape_loop(next_c);
        Til(r, tail)
    }
}

/// Extends a single codomain axis's free complement layout to cover `Extend`.
#[doc(hidden)]
pub const trait ExtendAxis<Extend: Shape>: Copy {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn extend_axis(self, extend: Extend) -> Layout<Self::Shape, Self::Stride>;
}

type FlatNonEmptyExtend<E> = <<E as Flatten>::Output as NonEmptyFlatShape>::Output;
type ExtendedTailShape<L, E> = <FlatNonEmptyExtend<E> as ExtendShapeLoop<
    <<L as SplitLastMode>::LastStride as StrideScalar>::Value,
>>::Output;
type ExtendedTailStride<L, E> =
    <ExtendedTailShape<L, E> as PrefixProduct<<L as SplitLastMode>::LastStride>>::Stride;
type CombinedAxisShape<L, E> =
    <<L as SplitLastMode>::PrefixShape as Append<ExtendedTailShape<L, E>>>::Output;
type CombinedAxisStride<L, E> =
    <<L as SplitLastMode>::PrefixStride as Append<ExtendedTailStride<L, E>>>::Output;
type ExtendedAxisCoalesced<L, E> =
    <Layout<CombinedAxisShape<L, E>, CombinedAxisStride<L, E>> as Coalesce<true>>::Output;

const impl<SC, DC, Extend> ExtendAxis<Extend> for Layout<SC, DC>
where
    Self: [const] SplitLastMode,
    <<Self as SplitLastMode>::LastStride as StrideScalar>::Value: [const] ShapeScalar,
    Extend: [const] Shape + [const] Flatten,
    <Extend as Flatten>::Output: [const] NonEmptyFlatShape,
    FlatNonEmptyExtend<Extend>:
        [const] ExtendShapeLoop<<<Self as SplitLastMode>::LastStride as StrideScalar>::Value>,
    ExtendedTailShape<Self, Extend>: [const] PrefixProduct<<Self as SplitLastMode>::LastStride>,
    ExtendedTailStride<Self, Extend>: [const] Tuple + Stride<ExtendedTailShape<Self, Extend>>,
    <Self as SplitLastMode>::PrefixShape: [const] Append<ExtendedTailShape<Self, Extend>>,
    <Self as SplitLastMode>::PrefixStride: [const] Append<ExtendedTailStride<Self, Extend>>,
    CombinedAxisShape<Self, Extend>: Shape,
    CombinedAxisStride<Self, Extend>: Stride<CombinedAxisShape<Self, Extend>>,
    Layout<CombinedAxisShape<Self, Extend>, CombinedAxisStride<Self, Extend>>:
        [const] Coalesce<true>,
    ExtendedAxisCoalesced<Self, Extend>: [const] DropTrailingUnit,
{
    type Shape = <ExtendedAxisCoalesced<Self, Extend> as DropTrailingUnit>::Shape;
    type Stride = <ExtendedAxisCoalesced<Self, Extend> as DropTrailingUnit>::Stride;

    fn extend_axis(self, extend: Extend) -> Layout<Self::Shape, Self::Stride> {
        let (prefix_s, prefix_d, last_d) = self.split_last_mode();
        let flat_extend = extend.flatten().non_empty_flat_shape();
        let shape_r = flat_extend.extend_shape_loop(last_d.value());
        let tail_d = prefix_product_with(shape_r, last_d);
        let full_s = prefix_s.append(shape_r);
        let full_d = prefix_d.append(tail_d);
        let coalesced = Coalesce::<true>::coalesce(Layout::new(full_s, full_d));
        coalesced.drop_trailing_unit()
    }
}

/// Walks the codomain profile alongside `extend` and `make_basis_like(extend)`,
/// extending covered axes and preserving trailing untouched modes of `extend`.
#[doc(hidden)]
pub const trait ExtendWalk<CoprofProfile, Extend, ExtendBases>: Copy {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn extend_walk(
        self,
        extend: Extend,
        extend_bases: ExtendBases,
    ) -> Layout<Self::Shape, Self::Stride>;
}

const impl<SC, DC, Extend, ExtendBases> ExtendWalk<LeafProfile, Extend, ExtendBases>
    for Layout<SC, DC>
where
    SC: Shape,
    DC: Stride<SC>,
    Extend: [const] Shape,
    ExtendBases: Copy,
    Self: [const] ExtendAxis<Extend>,
{
    type Shape = <Self as ExtendAxis<Extend>>::Shape;
    type Stride = <Self as ExtendAxis<Extend>>::Stride;

    fn extend_walk(self, extend: Extend, _: ExtendBases) -> Layout<Self::Shape, Self::Stride> {
        self.extend_axis(extend)
    }
}

const impl ExtendWalk<NilProfile, Nil, Nil> for Layout<Nil, Nil> {
    type Shape = Nil;
    type Stride = Nil;

    fn extend_walk(self, _: Nil, _: Nil) -> Layout<Nil, Nil> {
        Layout::new(Nil, Nil)
    }
}

const impl<EH, ET, BH, BT> ExtendWalk<NilProfile, Til<EH, ET>, Til<BH, BT>> for Layout<Nil, Nil>
where
    EH: Shape,
    ET: Tuple + Shape,
    BH: Stride<EH>,
    BT: Tuple + Stride<ET>,
{
    type Shape = Til<EH, ET>;
    type Stride = Til<BH, BT>;

    fn extend_walk(
        self,
        extend: Til<EH, ET>,
        extend_bases: Til<BH, BT>,
    ) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(extend, extend_bases)
    }
}

const impl<CH, CT, SCH, SCT, DCH, DCT, EH, ET, BH, BT>
    ExtendWalk<TupleProfile<CH, CT>, Til<EH, ET>, Til<BH, BT>>
    for Layout<Til<SCH, SCT>, Til<DCH, DCT>>
where
    SCH: Shape,
    SCT: Tuple + Shape,
    DCH: Stride<SCH>,
    DCT: Tuple + Stride<SCT>,
    EH: Shape,
    ET: Tuple + Shape,
    BH: Item,
    BT: Tuple,
    Layout<SCH, DCH>: [const] ExtendWalk<CH, EH, BH>,
    Layout<SCT, DCT>: [const] ExtendWalk<CT, ET, BT>,
    <Layout<SCT, DCT> as ExtendWalk<CT, ET, BT>>::Shape: Tuple + Shape,
    <Layout<SCT, DCT> as ExtendWalk<CT, ET, BT>>::Stride:
        Tuple + Stride<<Layout<SCT, DCT> as ExtendWalk<CT, ET, BT>>::Shape>,
{
    type Shape = Til<
        <Layout<SCH, DCH> as ExtendWalk<CH, EH, BH>>::Shape,
        <Layout<SCT, DCT> as ExtendWalk<CT, ET, BT>>::Shape,
    >;
    type Stride = Til<
        <Layout<SCH, DCH> as ExtendWalk<CH, EH, BH>>::Stride,
        <Layout<SCT, DCT> as ExtendWalk<CT, ET, BT>>::Stride,
    >;

    fn extend_walk(
        self,
        extend: Til<EH, ET>,
        extend_bases: Til<BH, BT>,
    ) -> Layout<Self::Shape, Self::Stride> {
        let head = Layout::new(self.shape.0, self.stride.0).extend_walk(extend.0, extend_bases.0);
        let tail = Layout::new(self.shape.1, self.stride.1).extend_walk(extend.1, extend_bases.1);
        Layout::new(Til(head.shape, tail.shape), Til(head.stride, tail.stride))
    }
}

/// Computes the minimal (free) complement layout $A^*$.
pub const trait Complement: Copy {
    /// Resulting complement [`Layout`] type.
    type Output: Copy;

    /// Computes the minimal complement, panicking if a dynamic precondition or overflow fails.
    fn complement(self) -> Self::Output;
}

/// Computes the complement layout $A^*$ extended to cover target shape `Extend`.
pub const trait ComplementWith<Extend: Shape>: Copy {
    /// Resulting extended complement [`Layout`] type.
    type Output: Copy;

    /// Computes the extended complement, panicking if a dynamic precondition or overflow fails.
    fn complement_with(self, extend: Extend) -> Self::Output;
}

type LayoutBases<S, D> = MakeBasisLikeOutput<<Layout<S, D> as Coprofile>::Output>;

const impl<S, D> Complement for Layout<S, D>
where
    S: Shape + [const] Flatten,
    D: Stride<S> + [const] Flatten,
    Self: [const] Coprofile,
    <<Self as Coprofile>::Output as Item>::Profile: [const] MakeBasisLikeTopProfile,
    LayoutBases<S, D>: [const] BuildComplement<S::Output, D::Output>,
{
    type Output = Layout<
        <LayoutBases<S, D> as BuildComplement<S::Output, D::Output>>::Shape,
        <LayoutBases<S, D> as BuildComplement<S::Output, D::Output>>::Stride,
    >;

    fn complement(self) -> Self::Output {
        let flat_s = self.shape.flatten();
        let flat_d = self.stride.flatten();
        let bases = make_basis_like(self.coprofile());
        bases.build_complement(flat_s, flat_d)
    }
}

type FreeComplementLayout<S, D> = Layout<
    <LayoutBases<S, D> as BuildComplement<<S as Flatten>::Output, <D as Flatten>::Output>>::Shape,
    <LayoutBases<S, D> as BuildComplement<<S as Flatten>::Output, <D as Flatten>::Output>>::Stride,
>;

const impl<S, D, Extend> ComplementWith<Extend> for Layout<S, D>
where
    S: Shape + [const] Flatten,
    D: Stride<S> + [const] Flatten,
    Self: [const] Coprofile,
    <<Self as Coprofile>::Output as Item>::Profile: [const] MakeBasisLikeTopProfile,
    LayoutBases<S, D>: [const] BuildComplement<S::Output, D::Output>,
    Extend: [const] Shape,
    Extend::Profile: [const] MakeBasisLikeTopProfile,
    FreeComplementLayout<S, D>: [const] ExtendWalk<
            <<Self as Coprofile>::Output as Item>::Profile,
            Extend,
            MakeBasisLikeOutput<Extend>,
        >,
{
    type Output = Layout<
        <FreeComplementLayout<S, D> as ExtendWalk<
            <<Self as Coprofile>::Output as Item>::Profile,
            Extend,
            MakeBasisLikeOutput<Extend>,
        >>::Shape,
        <FreeComplementLayout<S, D> as ExtendWalk<
            <<Self as Coprofile>::Output as Item>::Profile,
            Extend,
            MakeBasisLikeOutput<Extend>,
        >>::Stride,
    >;

    fn complement_with(self, extend: Extend) -> Self::Output {
        let free = self.complement();
        let extend_bases = make_basis_like(extend);
        free.extend_walk(extend, extend_bases)
    }
}

macro_rules! impl_tiler_complement {
    ([$($g:tt)*] $T:ty) => {
        const impl<$($g)*> Complement for $T
        where
            Self: [const] Tiler,
            <Self as Tiler>::OutputLayout: [const] Complement,
        {
            type Output = <<Self as Tiler>::OutputLayout as Complement>::Output;

            fn complement(self) -> Self::Output {
                tiler_to_layout(self).complement()
            }
        }

        const impl<$($g)* Extend: [const] Shape> ComplementWith<Extend> for $T
        where
            Self: [const] Tiler,
            <Self as Tiler>::OutputLayout: [const] ComplementWith<Extend>,
        {
            type Output = <<Self as Tiler>::OutputLayout as ComplementWith<Extend>>::Output;

            fn complement_with(self, extend: Extend) -> Self::Output {
                tiler_to_layout(self).complement_with(extend)
            }
        }
    };
}

impl_tiler_complement!([] Nil);
impl_tiler_complement!([] int);
impl_tiler_complement!([const N: int,] Int<N>);
impl_tiler_complement!([H: Item, T: Tuple,] Til<H, T>);

impl<S, D> Layout<S, D> {
    /// Computes the minimal complement layout $A^*$.
    pub const fn complement(self) -> <Self as Complement>::Output
    where
        Self: [const] Complement,
    {
        Complement::complement(self)
    }

    /// Computes the complement layout $A^*$ extended to cover `extend`.
    pub const fn complement_with<Extend: [const] Shape>(
        self,
        extend: Extend,
    ) -> <Self as ComplementWith<Extend>>::Output
    where
        Self: [const] ComplementWith<Extend>,
    {
        ComplementWith::complement_with(self, extend)
    }
}

/// Computes the minimal complement layout of a [`Layout`] or [`Tiler`].
///
/// Strides must use integer arithmetic. [`F2`](crate::f2::F2) strides fail at compile time.
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::algebra::complement;
/// use noodle_cute::f2::f2;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
///
/// let _ = complement(Layout::new(Int::<4>, f2(Int::<2>)));
/// ```
pub const fn complement<A: [const] Complement>(a: A) -> A::Output {
    a.complement()
}

/// Computes the complement layout of `a`, extended to cover `extend`.
pub const fn complement_with<A: [const] ComplementWith<Extend>, Extend: [const] Shape>(
    a: A,
    extend: Extend,
) -> A::Output {
    a.complement_with(extend)
}
