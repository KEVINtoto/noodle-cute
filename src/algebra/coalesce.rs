//! Coalesce for typed and runtime layouts, and the `coalesce_x` used by composition.
//!
//! Follows CuTe `layout.hpp` `coalesce` and `bw_coalesce`. Typed layouts change structure only
//! from facts known to the type, and both type-level paths share the backward fold and `Prepend`.
//! Runtime layouts simplify by their current values; an empty input is normalized to `1:0` by
//! this crate's scalar convention.

use core::marker::ConstParamTy;
use core::ops::Mul;

use crate::basis::ScaledBasis;
use crate::int::{Int, Integer, mul};
use crate::layout::{DynLayout, Layout, LayoutDesc, LayoutMap, MAX_RANK, Shape, StrideOf};
use crate::tuple::{Flatten, Item, Nil};
use crate::{Error, IntTuple, Result};

/// Selects ordinary coalesce or CuTe's extended-domain `coalesce_x`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum CoalesceMode {
    /// Preserves the logical-domain mapping and removes static unit modes.
    Standard,
    /// Keeps the trailing stride used by extended indexing.
    Extended,
}

/// Coalescing action selected from static shape and stride facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum CoalesceAction {
    Discard,
    Retain,
    Merge,
}

// Decision<CoalesceAction> was replaced by direct ApplyPrepend trait dispatch.

/// Selects whether a leaf is discarded or retained when the folded tail is empty.
pub const fn unit_decision(s: Option<isize>) -> CoalesceAction {
    if matches!(s, Some(1)) {
        CoalesceAction::Discard
    } else {
        CoalesceAction::Retain
    }
}

/// Selects whether to discard, retain, or merge a leaf with the folded tail's head.
pub const fn merge_decision(
    s: Option<isize>,
    d: Option<isize>,
    tail_shape: Option<isize>,
    next: Option<isize>,
) -> CoalesceAction {
    if matches!(s, Some(1)) {
        return CoalesceAction::Discard;
    }
    // None means unknown at the type level; the runtime path passes every value as known.
    // CuTe's `_0 * x` is a static `_0` even when `x` is dynamic.
    let product = match (s, d) {
        (Some(0), _) | (_, Some(0)) => Some(0),
        (Some(a), Some(b)) => a.checked_mul(b),
        _ => None,
    };
    match (product, tail_shape, next) {
        (Some(p), Some(_), Some(c)) if p == c => CoalesceAction::Merge,
        _ => CoalesceAction::Retain,
    }
}

// Reuse these constants in bounds and associated-type projections so generic
// const expressions refer to identical definitions.
pub const UNIT_DECISION<S: Integer>: CoalesceAction = unit_decision(S::STATIC);
pub trait CoalesceStride {
    const STATIC: Option<isize>;
    const BASIS: Option<usize>;
}
impl<I: Integer> CoalesceStride for I {
    const STATIC: Option<isize> = I::STATIC;
    const BASIS: Option<usize> = None;
}
impl<V: Integer, const M: usize> CoalesceStride for ScaledBasis<V, M> {
    const STATIC: Option<isize> = V::STATIC;
    const BASIS: Option<usize> = Some(M);
}
pub const MERGE_DECISION<S: Integer, D: CoalesceStride, TS: Integer, TD: CoalesceStride>:
    CoalesceAction = if matches!(S::STATIC, Some(1)) {
    CoalesceAction::Discard
} else if match (D::BASIS, TD::BASIS) {
    (None, None) => true,
    (Some(a), Some(b)) => a == b,
    _ => false,
} {
    merge_decision(S::STATIC, D::STATIC, TS::STATIC, TD::STATIC)
} else {
    CoalesceAction::Retain
};

/// Applies the prepend action to an empty or non-empty tail layout.
pub trait ApplyPrepend<
    const ACTION: CoalesceAction,
    S,
    D,
    const MODE: CoalesceMode = { CoalesceMode::Standard },
>
{
    type Output;
    fn apply_prepend(self, s: S, d: D) -> Self::Output;
}

impl<S, D> ApplyPrepend<{ CoalesceAction::Discard }, S, D, { CoalesceMode::Standard }>
    for Layout<Nil, Nil>
{
    type Output = Layout<Nil, Nil>;
    fn apply_prepend(self, _: S, _: D) -> Self::Output {
        self
    }
}

impl<S, D> ApplyPrepend<{ CoalesceAction::Discard }, S, D, { CoalesceMode::Extended }>
    for Layout<Nil, Nil>
{
    type Output = Layout<IntTuple<Int<2>, Nil>, IntTuple<D, Nil>>;
    fn apply_prepend(self, _: S, d: D) -> Self::Output {
        Layout {
            shape: IntTuple(Int, Nil),
            stride: IntTuple(d, Nil),
        }
    }
}

impl<S, D, const MODE: CoalesceMode> ApplyPrepend<{ CoalesceAction::Retain }, S, D, MODE>
    for Layout<Nil, Nil>
{
    type Output = Layout<IntTuple<S, Nil>, IntTuple<D, Nil>>;
    fn apply_prepend(self, s: S, d: D) -> Self::Output {
        Layout {
            shape: IntTuple(s, Nil),
            stride: IntTuple(d, Nil),
        }
    }
}

impl<S, D, TS, TSR, TD, TDR, const MODE: CoalesceMode>
    ApplyPrepend<{ CoalesceAction::Discard }, S, D, MODE>
    for Layout<IntTuple<TS, TSR>, IntTuple<TD, TDR>>
{
    type Output = Self;
    fn apply_prepend(self, _: S, _: D) -> Self::Output {
        self
    }
}

impl<S, D, TS, TSR, TD, TDR, const MODE: CoalesceMode>
    ApplyPrepend<{ CoalesceAction::Retain }, S, D, MODE>
    for Layout<IntTuple<TS, TSR>, IntTuple<TD, TDR>>
{
    type Output = Layout<IntTuple<S, IntTuple<TS, TSR>>, IntTuple<D, IntTuple<TD, TDR>>>;
    fn apply_prepend(self, s: S, d: D) -> Self::Output {
        Layout {
            shape: IntTuple(s, self.shape),
            stride: IntTuple(d, self.stride),
        }
    }
}

impl<S: Mul<TS>, D, TS, TSR, TD, TDR, const MODE: CoalesceMode>
    ApplyPrepend<{ CoalesceAction::Merge }, S, D, MODE>
    for Layout<IntTuple<TS, TSR>, IntTuple<TD, TDR>>
{
    type Output = Layout<IntTuple<S::Output, TSR>, IntTuple<D, TDR>>;
    fn apply_prepend(self, s: S, d: D) -> Self::Output {
        Layout {
            shape: IntTuple(s * self.shape.0, self.shape.1),
            stride: IntTuple(d, self.stride.1),
        }
    }
}

/// Dispatches the prepend action using the static extent and stride metadata.
pub trait Prepend<S, D> {
    type Output;
    fn prepend(self, s: S, d: D) -> Self::Output;
}

impl<S: Integer, D> Prepend<S, D> for Layout<Nil, Nil>
where
    Self: ApplyPrepend<{ UNIT_DECISION::<S> }, S, D>,
{
    type Output = <Self as ApplyPrepend<{ UNIT_DECISION::<S> }, S, D>>::Output;
    fn prepend(self, s: S, d: D) -> Self::Output {
        self.apply_prepend(s, d)
    }
}

impl<S: Integer, D: CoalesceStride, TS: Integer, TSR, TD: CoalesceStride, TDR> Prepend<S, D>
    for Layout<IntTuple<TS, TSR>, IntTuple<TD, TDR>>
where
    Self: ApplyPrepend<{ MERGE_DECISION::<S, D, TS, TD> }, S, D>,
{
    type Output = <Self as ApplyPrepend<{ MERGE_DECISION::<S, D, TS, TD> }, S, D>>::Output;
    fn prepend(self, s: S, d: D) -> Self::Output {
        self.apply_prepend(s, d)
    }
}

/// Coalesces a flattened layout at the type level.
///
/// The result can still be empty when the input is empty or every extent is a
/// static one; [`FinishCoalesce`] normalizes that case.
/// `CoalesceMode::Extended` selects the `coalesce_x` seed, which keeps the trailing stride.
pub trait CoalesceFold<const MODE: CoalesceMode = { CoalesceMode::Standard }> {
    type Output;
    fn fold(self) -> Self::Output;
}

impl CoalesceFold for Layout<Nil, Nil> {
    type Output = Self;
    fn fold(self) -> Self {
        self
    }
}

impl<S: Integer, D> CoalesceFold for Layout<IntTuple<S, Nil>, IntTuple<D, Nil>>
where
    Layout<Nil, Nil>: Prepend<S, D>,
{
    type Output = <Layout<Nil, Nil> as Prepend<S, D>>::Output;
    fn fold(self) -> Self::Output {
        Layout {
            shape: Nil,
            stride: Nil,
        }
        .prepend(self.shape.0, self.stride.0)
    }
}

impl CoalesceFold<{ CoalesceMode::Extended }> for Layout<Nil, Nil> {
    // Upstream requires a nonempty input; an empty layout follows this crate's scalar convention.
    type Output = Layout<IntTuple<Int<1>, Nil>, IntTuple<Int<0>, Nil>>;
    fn fold(self) -> Self::Output {
        Layout {
            shape: IntTuple(Int, Nil),
            stride: IntTuple(Int, Nil),
        }
    }
}

impl<S: Integer, D> CoalesceFold<{ CoalesceMode::Extended }>
    for Layout<IntTuple<S, Nil>, IntTuple<D, Nil>>
where
    Layout<Nil, Nil>: ApplyPrepend<{ UNIT_DECISION::<S> }, S, D, { CoalesceMode::Extended }>,
{
    type Output = <Layout<Nil, Nil> as ApplyPrepend<
        { UNIT_DECISION::<S> },
        S,
        D,
        { CoalesceMode::Extended },
    >>::Output;
    fn fold(self) -> Self::Output {
        Layout {
            shape: Nil,
            stride: Nil,
        }
        .apply_prepend(self.shape.0, self.stride.0)
    }
}

impl<S, D, SH, ST, DH, DT, const MODE: CoalesceMode> CoalesceFold<MODE>
    for Layout<IntTuple<S, IntTuple<SH, ST>>, IntTuple<D, IntTuple<DH, DT>>>
where
    Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>: CoalesceFold<MODE>,
    <Layout<IntTuple<SH, ST>, IntTuple<DH, DT>> as CoalesceFold<MODE>>::Output: Prepend<S, D>,
{
    type Output =
        <<Layout<IntTuple<SH, ST>, IntTuple<DH, DT>> as CoalesceFold<MODE>>::Output as Prepend<
            S,
            D,
        >>::Output;
    fn fold(self) -> Self::Output {
        Layout {
            shape: self.shape.1,
            stride: self.stride.1,
        }
        .fold()
        .prepend(self.shape.0, self.stride.0)
    }
}

/// Validates the folded layout and normalizes an empty result to `1:0`.
pub trait FinishCoalesce {
    type Output: LayoutMap;
    fn finish(self) -> Result<Self::Output>;
}

impl FinishCoalesce for Layout<Nil, Nil> {
    type Output = Layout<Int<1>, Int<0>>;
    // An empty input or removal of every static unit extent yields a single
    // logical element at address zero.
    fn finish(self) -> Result<Self::Output> {
        Layout::new(Int, Int)
    }
}

// `S`/`D` are the first shape/stride leaf types; `TSR`/`TDR` the remaining tuples.
impl<S, TSR, D, TDR> FinishCoalesce for Layout<IntTuple<S, TSR>, IntTuple<D, TDR>>
where
    IntTuple<S, TSR>: Shape,
    IntTuple<D, TDR>: Item<Profile = <IntTuple<S, TSR> as Item>::Profile>,
{
    type Output = Self;
    fn finish(self) -> Result<Self> {
        Layout::new(self.shape, self.stride)
    }
}

/// Result type of typed coalesce; [`CoalesceMode::Extended`] selects CuTe `coalesce_x`,
/// which keeps the extended mapping. The default is [`CoalesceMode::Standard`].
///
/// For fully static inputs, `<CoalescedLayout<S, D> as StaticValue>::VALUE` is the result as
/// a constant: static unit modes are removed, contiguous modes merged, and an empty result is
/// normalized to `1:0`. The extended version replaces a trailing static unit extent with 2.
/// Static merge products must be representable as `isize`.
///
/// ```
/// use noodle_cute::{Int, IntTuple, Nil, StaticValue, algebra::{CoalesceMode, CoalescedLayout}, layout::Layout};
/// type Pair<const A: isize, const B: isize> = IntTuple<Int<A>, IntTuple<Int<B>, Nil>>;
/// type Single<const S: isize, const D: isize> = Layout<IntTuple<Int<S>, Nil>, IntTuple<Int<D>, Nil>>;
/// const C: Single<6, 1> = <CoalescedLayout<Pair<2, 3>, Pair<1, 2>> as StaticValue>::VALUE;
/// const X: Single<4, 1> = <CoalescedLayout<Pair<2, 1>, Pair<1, 2>, { CoalesceMode::Extended }> as StaticValue>::VALUE;
/// ```
///
/// Runtime leaves have no constant value; use [`Layout::coalesce`] instead:
/// ```compile_fail
/// use noodle_cute::{DynInt, Int, StaticValue, algebra::CoalescedLayout};
/// const C: CoalescedLayout<DynInt, Int<1>> = StaticValue::VALUE;
/// ```
/// Static merge products must fit in `isize`:
/// ```compile_fail
/// use noodle_cute::{Int, IntTuple, Nil, StaticValue, algebra::CoalescedLayout};
/// type Shape = IntTuple<Int<{ isize::MAX }>, IntTuple<Int<2>, Nil>>;
/// type Stride = IntTuple<Int<0>, IntTuple<Int<0>, Nil>>;
/// const C: CoalescedLayout<Shape, Stride> = StaticValue::VALUE;
/// ```
pub type CoalescedLayout<S, D, const MODE: CoalesceMode = { CoalesceMode::Standard }> = <<Layout<
    <S as Flatten>::Output,
    <D as Flatten>::Output,
> as CoalesceFold<MODE>>::Output as FinishCoalesce>::Output;

impl<S: Shape + Flatten, D: StrideOf<S> + Flatten> Layout<S, D>
where
    Layout<S::Output, D::Output>: CoalesceFold,
    <Layout<S::Output, D::Output> as CoalesceFold>::Output: FinishCoalesce,
{
    /// Flattens and coalesces while preserving the logical linear address map.
    ///
    /// Removes static unit extents and merges adjacent leaves only when their
    /// static values prove `shape * stride == next_stride`. Static result leaves
    /// remain encoded in their types; runtime values never alter the tuple
    /// profile. An empty result is normalized to the scalar layout `1:0`.
    /// As in CuTe's `bw_coalesce`, merging also requires the tail's leading shape to be static.
    /// The input shape and output layout are validated at runtime; a static merge product
    /// that overflows fails at instantiation.
    pub fn coalesce(&self) -> Result<CoalescedLayout<S, D>> {
        self.size()?;
        self.clone().flatten().fold().finish()
    }
}

impl<S: Shape + Flatten, D: StrideOf<S> + Flatten> Layout<S, D>
where
    Layout<S::Output, D::Output>: CoalesceFold<{ CoalesceMode::Extended }>,
    <Layout<S::Output, D::Output> as CoalesceFold<{ CoalesceMode::Extended }>>::Output:
        FinishCoalesce,
{
    /// Typed CuTe `coalesce_x`: simplifies while keeping the extended-domain mapping, for
    /// static and mixed leaves.
    ///
    /// Like CuTe, a trailing static unit extent becomes 2 while a dynamic one keeps its value,
    /// so the logical size may grow. For a nonempty input with positive leading extents the
    /// extended mapping is unchanged. Merging uses only static facts; an empty input becomes
    /// `1:0`. The input shape and the output size and address range are validated.
    ///
    /// ```
    /// use noodle_cute::{DynInt, Int, shape, stride};
    /// use noodle_cute::layout::{Layout, LayoutMap};
    /// let a = Layout::new(shape![Int::<2>, Int::<1>], stride![Int::<1>, DynInt(8)])?;
    /// let b = a.coalesce_extended()?;
    /// assert_eq!(b.size()?, 4);
    /// for i in 0..12 { assert_eq!(a.index_extended(i)?, b.index_extended(i)?); }
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn coalesce_extended(&self) -> Result<CoalescedLayout<S, D, { CoalesceMode::Extended }>> {
        self.size()?;
        self.clone().flatten().fold().finish()
    }
}

impl DynLayout {
    /// CuTe `coalesce(layout)`: drops size-1 leaves and merges adjacent leaves
    /// when `s0 * d0 == d1`. A scalar or all-size-1 layout becomes `1:0`.
    ///
    /// The logical size and mapping on `[0, size)` are preserved. The extended
    /// mapping can change when a trailing size-1 leaf is removed.
    ///
    /// ```
    /// use noodle_cute::layout::{DynLayout, LayoutMap};
    ///
    /// let layout = DynLayout::new(&[2, 3, 1], &[1, 2, 99])?;
    /// let compact = layout.coalesce()?;
    /// assert_eq!(compact.shapes(), &[6]);
    /// assert_eq!(compact.strides(), &[1]);
    /// assert_eq!(layout.index(5)?, compact.index(5)?);
    /// // Removing the trailing unit axis changes extension beyond the domain.
    /// assert_eq!(layout.index_extended(6)?, 99);
    /// assert_eq!(compact.index_extended(6)?, 6);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn coalesce(&self) -> Result<Self> {
        self.coalesce_impl(CoalesceMode::Standard)
    }

    /// CuTe `coalesce_x`: keeps the trailing stride that determines the extended direction.
    ///
    /// A trailing unit extent keeps its actual value unless it merges with the previous leaf, so
    /// `(2,1):(2,8)` is unchanged and `(2,1):(1,2)` becomes `2:1`. The logical size is unchanged,
    /// and the extended mapping is preserved for a nonempty input with positive leading extents.
    /// Zero extents still simplify by CuTe's algebra, which need not keep the division-by-zero
    /// errors of the original extended decoding. An empty input becomes `1:0`, which also accepts
    /// positive extended indices and maps them to address 0.
    pub fn coalesce_extended(&self) -> Result<Self> {
        self.coalesce_impl(CoalesceMode::Extended)
    }

    fn coalesce_impl(&self, mode: CoalesceMode) -> Result<Self> {
        let desc = LayoutDesc::from_dyn_layout(self);
        let coalesced = coalesce_desc(&desc, mode)?;
        Ok(coalesced.to_dyn_layout())
    }
}

/// Coalesces a [`LayoutDesc`] according to CuTe coalescing rules.
pub const fn coalesce_desc(desc: &LayoutDesc, mode: CoalesceMode) -> Result<LayoutDesc> {
    let mut out = LayoutDesc::empty();
    let mut a = desc.rank;
    while a > 0 {
        a -= 1;
        let s = desc.shapes[a];
        let d = desc.strides[a];
        if s == 1 && !(matches!(mode, CoalesceMode::Extended) && a + 1 == desc.rank) {
            continue;
        }
        if out.rank > 0 {
            let j = out.rank - 1;
            if matches!(
                merge_decision(Some(s), Some(d), Some(out.shapes[j]), Some(out.strides[j])),
                CoalesceAction::Merge
            ) {
                let merged_s = match mul(s, out.shapes[j]) {
                    Ok(v) => v,
                    Err(e) => return Err(e),
                };
                out.shapes[j] = merged_s;
                out.strides[j] = d;
                continue;
            }
        }
        if out.rank >= MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        out.shapes[out.rank] = s;
        out.strides[out.rank] = d;
        out.rank += 1;
    }
    if out.rank == 0 {
        out.shapes[0] = 1;
        out.strides[0] = 0;
        out.rank = 1;
    }
    let mut left = 0;
    let mut right = out.rank - 1;
    while left < right {
        out.shapes.swap(left, right);
        out.strides.swap(left, right);
        left += 1;
        right -= 1;
    }
    Ok(out)
}
