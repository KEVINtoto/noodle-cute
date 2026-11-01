//! Coalesce for typed and runtime layouts, and the `coalesce_x` used by composition.
//!
//! Follows CuTe `layout.hpp` `coalesce` and `bw_coalesce`. Typed layouts change structure only
//! from facts known to the type, and both type-level paths share the backward fold and `Prepend`.
//! Runtime layouts simplify by their current values; an empty input is normalized to `1:0` by
//! this crate's scalar convention.

use core::ops::Mul;

use crate::int::{Int, Integer, mul};
use crate::layout::{DynLayout, Layout, LayoutMap, Shape};
use crate::tuple::{Flatten, Item, Nil};
use crate::{IntTuple, Result};

/// Selects a coalescing action from facts known at compile time.
///
/// The fold changes types only using static facts: a runtime extent equal to one
/// remains in the output tuple. Action codes are [`DISCARD`] for a static unit
/// extent, [`RETAIN`] to keep a leaf, and [`MERGE`] to merge it with the folded
/// tail's head.
pub struct Decision<const N: u8>;

/// Drops a static unit extent.
pub const DISCARD: u8 = 0;
/// Keeps the leaf as a separate mode.
pub const RETAIN: u8 = 1;
/// Merges the leaf with the folded tail's head.
pub const MERGE: u8 = 2;

/// Selects whether a leaf is discarded or retained when the folded tail is empty.
pub const fn unit_decision(s: Option<isize>) -> u8 {
    if matches!(s, Some(1)) {
        DISCARD
    } else {
        RETAIN
    }
}

/// Selects whether to discard, retain, or merge a leaf with the folded tail's head.
pub const fn merge_decision(
    s: Option<isize>,
    d: Option<isize>,
    tail_shape: Option<isize>,
    next: Option<isize>,
) -> u8 {
    if matches!(s, Some(1)) {
        return DISCARD;
    }
    // None means unknown at the type level; the runtime path passes every value as known.
    // CuTe's `_0 * x` is a static `_0` even when `x` is dynamic.
    let product = match (s, d) {
        (Some(0), _) | (_, Some(0)) => Some(0),
        (Some(a), Some(b)) => a.checked_mul(b),
        _ => None,
    };
    match (product, tail_shape, next) {
        (Some(p), Some(_), Some(c)) if p == c => MERGE,
        _ => RETAIN,
    }
}

// Reuse these constants in bounds and associated-type projections so generic
// const expressions refer to identical definitions.
pub const UNIT_DECISION<S: Integer>: u8 = unit_decision(S::STATIC);
pub const MERGE_DECISION<S: Integer, D: Integer, TS: Integer, TD: Integer>: u8 =
    merge_decision(S::STATIC, D::STATIC, TS::STATIC, TD::STATIC);

/// Adds a leaf to an empty tail; with `EXTENDED`, a trailing static unit extent becomes 2 to keep its stride.
pub trait PrependToEmpty<S, D, const EXTENDED: bool = false> {
    type Output;
    fn apply(s: S, d: D) -> Self::Output;
}

impl<S, D> PrependToEmpty<S, D, false> for Decision<DISCARD> {
    type Output = Layout<Nil, Nil>;
    fn apply(_: S, _: D) -> Self::Output {
        Layout {
            shape: Nil,
            stride: Nil,
        }
    }
}

impl<S, D, const EXTENDED: bool> PrependToEmpty<S, D, EXTENDED> for Decision<RETAIN> {
    type Output = Layout<IntTuple<S, Nil>, IntTuple<D, Nil>>;
    fn apply(s: S, d: D) -> Self::Output {
        Layout {
            shape: IntTuple(s, Nil),
            stride: IntTuple(d, Nil),
        }
    }
}

/// Prepends `(s, d)` to a folded tail. Type parameters:
/// - `S`, `D`: types of the new shape and stride leaves being prepended.
/// - `TS`, `TD`: types of the tail's first shape and stride leaves.
/// - `TSR`, `TDR`: types of the remaining shape and stride tuples of the tail
///   after its first leaf.
pub trait PrependToNonempty<S, D, TS, TSR, TD, TDR> {
    type Output;
    fn apply(s: S, d: D, tail: Layout<IntTuple<TS, TSR>, IntTuple<TD, TDR>>) -> Self::Output;
}

impl<S, D, TS, TSR, TD, TDR> PrependToNonempty<S, D, TS, TSR, TD, TDR> for Decision<DISCARD> {
    type Output = Layout<IntTuple<TS, TSR>, IntTuple<TD, TDR>>;
    fn apply(_: S, _: D, tail: Self::Output) -> Self::Output {
        tail
    }
}

impl<S, D, TS, TSR, TD, TDR> PrependToNonempty<S, D, TS, TSR, TD, TDR> for Decision<RETAIN> {
    type Output = Layout<IntTuple<S, IntTuple<TS, TSR>>, IntTuple<D, IntTuple<TD, TDR>>>;
    fn apply(s: S, d: D, tail: Layout<IntTuple<TS, TSR>, IntTuple<TD, TDR>>) -> Self::Output {
        Layout {
            shape: IntTuple(s, tail.shape),
            stride: IntTuple(d, tail.stride),
        }
    }
}

impl<S: Mul<TS>, D, TS, TSR, TD, TDR> PrependToNonempty<S, D, TS, TSR, TD, TDR>
    for Decision<MERGE>
{
    type Output = Layout<IntTuple<S::Output, TSR>, IntTuple<D, TDR>>;
    fn apply(s: S, d: D, tail: Layout<IntTuple<TS, TSR>, IntTuple<TD, TDR>>) -> Self::Output {
        Layout {
            shape: IntTuple(s * tail.shape.0, tail.shape.1),
            stride: IntTuple(d, tail.stride.1),
        }
    }
}

/// Dispatches the prepend action using the static extent and stride metadata.
///
/// `S`/`D` are the new shape/stride leaf types. For a nonempty tail, `TS`/`TD`
/// are its first shape/stride leaf types and `TSR`/`TDR` the remaining tuples.
pub trait Prepend<S, D> {
    type Output;
    fn prepend(self, s: S, d: D) -> Self::Output;
}

impl<S: Integer, D> Prepend<S, D> for Layout<Nil, Nil>
where
    Decision<{ UNIT_DECISION::<S> }>: PrependToEmpty<S, D>,
{
    type Output = <Decision<{ UNIT_DECISION::<S> }> as PrependToEmpty<S, D>>::Output;
    fn prepend(self, s: S, d: D) -> Self::Output {
        <Decision<{ UNIT_DECISION::<S> }> as PrependToEmpty<S, D>>::apply(s, d)
    }
}

impl<S: Integer, D: Integer, TS: Integer, TSR, TD: Integer, TDR> Prepend<S, D>
    for Layout<IntTuple<TS, TSR>, IntTuple<TD, TDR>>
where
    Decision<{ MERGE_DECISION::<S, D, TS, TD> }>: PrependToNonempty<S, D, TS, TSR, TD, TDR>,
{
    type Output = <Decision<{ MERGE_DECISION::<S, D, TS, TD> }> as PrependToNonempty<
        S,
        D,
        TS,
        TSR,
        TD,
        TDR,
    >>::Output;
    fn prepend(self, s: S, d: D) -> Self::Output {
        <Decision<{ MERGE_DECISION::<S, D, TS, TD> }> as PrependToNonempty<
            S,
            D,
            TS,
            TSR,
            TD,
            TDR,
        >>::apply(s, d, self)
    }
}

/// Coalesces a flattened layout at the type level.
///
/// The result can still be empty when the input is empty or every extent is a
/// static one; [`FinishCoalesce`] normalizes that case.
/// `EXTENDED` selects the `coalesce_x` seed, which keeps the trailing stride.
pub trait CoalesceFold<const EXTENDED: bool = false> {
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

impl CoalesceFold<true> for Layout<Nil, Nil> {
    // Upstream requires a nonempty input; an empty layout follows this crate's scalar convention.
    type Output = Layout<IntTuple<Int<1>, Nil>, IntTuple<Int<0>, Nil>>;
    fn fold(self) -> Self::Output {
        Layout {
            shape: IntTuple(Int, Nil),
            stride: IntTuple(Int, Nil),
        }
    }
}

impl<S, D> PrependToEmpty<S, D, true> for Decision<DISCARD> {
    type Output = Layout<IntTuple<Int<2>, Nil>, IntTuple<D, Nil>>;
    fn apply(_: S, d: D) -> Self::Output {
        Layout {
            shape: IntTuple(Int, Nil),
            stride: IntTuple(d, Nil),
        }
    }
}
impl<S: Integer, D> CoalesceFold<true> for Layout<IntTuple<S, Nil>, IntTuple<D, Nil>>
where
    Decision<{ UNIT_DECISION::<S> }>: PrependToEmpty<S, D, true>,
{
    type Output = <Decision<{ UNIT_DECISION::<S> }> as PrependToEmpty<S, D, true>>::Output;
    fn fold(self) -> Self::Output {
        <Decision<{ UNIT_DECISION::<S> }> as PrependToEmpty<S, D, true>>::apply(
            self.shape.0,
            self.stride.0,
        )
    }
}

impl<S, D, SH, ST, DH, DT, const EXTENDED: bool> CoalesceFold<EXTENDED>
    for Layout<IntTuple<S, IntTuple<SH, ST>>, IntTuple<D, IntTuple<DH, DT>>>
where
    Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>: CoalesceFold<EXTENDED>,
    <Layout<IntTuple<SH, ST>, IntTuple<DH, DT>> as CoalesceFold<EXTENDED>>::Output: Prepend<S, D>,
{
    type Output = <<Layout<IntTuple<SH, ST>, IntTuple<DH, DT>> as CoalesceFold<EXTENDED>>::Output as Prepend<S, D>>::Output;
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

/// Result type of the typed coalesce; `EXTENDED` selects CuTe `coalesce_x`, which keeps the
/// extended mapping.
///
/// For fully static inputs, `<CoalescedLayout<S, D> as StaticValue>::VALUE` is the result as
/// a constant: static unit modes are removed, contiguous modes merged, and an empty result is
/// normalized to `1:0`. The extended version replaces a trailing static unit extent with 2.
/// Static merge products must be representable as `isize`.
///
/// ```
/// use noodle_cute::{Int, IntTuple, Nil, StaticValue, algebra::CoalescedLayout, layout::Layout};
/// type Pair<const A: isize, const B: isize> = IntTuple<Int<A>, IntTuple<Int<B>, Nil>>;
/// type Single<const S: isize, const D: isize> = Layout<IntTuple<Int<S>, Nil>, IntTuple<Int<D>, Nil>>;
/// const C: Single<6, 1> = <CoalescedLayout<Pair<2, 3>, Pair<1, 2>> as StaticValue>::VALUE;
/// const X: Single<4, 1> = <CoalescedLayout<Pair<2, 1>, Pair<1, 2>, true> as StaticValue>::VALUE;
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
pub type CoalescedLayout<S, D, const EXTENDED: bool = false> = <<Layout<
    <S as Flatten>::Output,
    <D as Flatten>::Output,
> as CoalesceFold<EXTENDED>>::Output as FinishCoalesce>::Output;

impl<S: Shape + Flatten, D: Item<Profile = S::Profile> + Flatten> Layout<S, D>
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

impl<S: Shape + Flatten, D: Item<Profile = S::Profile> + Flatten> Layout<S, D>
where
    Layout<S::Output, D::Output>: CoalesceFold<true>,
    <Layout<S::Output, D::Output> as CoalesceFold<true>>::Output: FinishCoalesce,
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
    pub fn coalesce_extended(&self) -> Result<CoalescedLayout<S, D, true>> {
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
        self.coalesce_impl(false)
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
        self.coalesce_impl(true)
    }

    // At runtime the actual trailing extent keeps the extended stride, avoiding the overflow a
    // placeholder extent of 2 could add.
    fn coalesce_impl(&self, extended: bool) -> Result<Self> {
        let mut out = Self::scalar();
        for a in (0..self.rank).rev() {
            let (s, d) = (self.shape[a], self.stride[a]);
            if s == 1 && !(extended && a + 1 == self.rank) {
                continue;
            }
            if out.rank > 0 {
                let j = out.rank - 1;
                if merge_decision(Some(s), Some(d), Some(out.shape[j]), Some(out.stride[j]))
                    == MERGE
                {
                    out.shape[j] = mul(s, out.shape[j])?;
                    out.stride[j] = d;
                    continue;
                }
            }
            out.shape[out.rank] = s;
            out.stride[out.rank] = d;
            out.rank += 1;
        }
        if out.rank == 0 {
            out.rank = 1;
            out.shape[0] = 1;
            out.stride[0] = 0;
        }
        out.shape[..out.rank].reverse();
        out.stride[..out.rank].reverse();
        Ok(out)
    }
}
