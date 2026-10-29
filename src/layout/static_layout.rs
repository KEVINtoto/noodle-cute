//! Typed affine layouts with static or mixed static/runtime leaves.
//!
//! Type-level coalescing uses only facts encoded in integer types. Intermediate
//! fold results use [`Layout`] directly and are validated when the fold finishes.

use core::fmt::Debug;
use core::ops::Mul;

use crate::int::{Int, Integer, mul};
use crate::tuple::{Flatten, Item, Nil, Tuple};
use crate::{Error, IntTuple, Result};

use super::{AffineLayoutMap, DynLayout, GroupedLayout, LayoutMap, MAX_RANK, Shape};

/// An affine map defined by a shape and structurally congruent strides.
///
/// The shape and stride profiles must match at the type level, including nested
/// modes. Each leaf contributes `coordinate * stride` to the address. Static
/// leaves occupy no storage; runtime leaves retain their values in the layout.
/// Negative and zero strides are allowed; [`LayoutMap::is_injective`] checks
/// whether distinct logical coordinates can share an address.
///
/// Public fields allow construction without validation. Use [`Layout::new`] to
/// check the shape and address arithmetic. The coalescing fold also builds
/// unchecked `Layout { shape, stride }` intermediates before validating its result.
///
/// ```
/// use noodle_cute::{DynInt, Int, shape, stride};
/// use noodle_cute::layout::{Layout, LayoutMap};
///
/// let layout = Layout::new(
///     shape![Int::<2>, DynInt(3)],
///     stride![Int::<1>, Int::<2>],
/// )?;
/// assert_eq!(layout.size()?, 6);
/// assert_eq!(layout.index_of_coord(&[1, 2])?, 5);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout<S, D> {
    /// Nonnegative extents arranged as an integer leaf or nested tuple.
    pub shape: S,
    /// Signed address increments with the same nested profile as `shape`.
    pub stride: D,
}

impl<S: Shape, D: Item<Profile = S::Profile>> Layout<S, D> {
    /// Constructs a layout after checking nonnegative extents, logical size,
    /// and reachable address endpoints with checked arithmetic.
    ///
    /// Negative extents return [`Error::InvalidShape`]; unrepresentable size or
    /// address calculations return [`Error::Overflow`]. Addresses may be negative
    /// or repeated, and an empty domain has no address endpoints to check.
    pub fn new(shape: S, stride: D) -> Result<Self> {
        let layout = Self { shape, stride };
        layout.size()?;
        layout.address_range()?;
        Ok(layout)
    }

    /// Flattens nested modes in leaf order without changing the linear address map.
    ///
    /// Static leaf values remain encoded in the result type; flattening neither
    /// merges contiguous leaves nor removes unit extents.
    pub fn flatten(self) -> Layout<<S as Flatten>::Output, <D as Flatten>::Output>
    where
        S: Flatten,
        D: Flatten,
    {
        Layout {
            shape: self.shape.flatten(),
            stride: self.stride.flatten(),
        }
    }
}

impl<S: Shape, D: Item<Profile = S::Profile>> AffineLayoutMap for Layout<S, D> {
    fn affine_rank(&self) -> usize {
        S::RANK
    }

    fn shape(&self, leaf: usize) -> Result<isize> {
        self.shape.leaf(leaf)
    }

    fn stride(&self, leaf: usize) -> Result<isize> {
        self.stride.leaf(leaf)
    }

    fn leaves(&self) -> core::ops::Range<usize> {
        0..S::LEAVES
    }

    fn mode_span(&self, axis: usize) -> Result<(usize, usize)> {
        self.shape.mode_span(axis)
    }

    /// Exports flattened runtime metadata for reflection or backend parameters.
    ///
    /// The result has one axis per leaf and discards the original mode grouping.
    /// Returns [`Error::CapacityExceeded`] if the leaf count exceeds [`MAX_RANK`].
    /// The exported values are validated by [`DynLayout::new`].
    fn as_flat_layout(&self) -> Result<DynLayout> {
        if S::LEAVES > MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        let mut shape = [0; MAX_RANK];
        let mut stride = [0; MAX_RANK];
        for i in 0..S::LEAVES {
            shape[i] = self.shape.leaf(i)?;
            stride[i] = self.stride.leaf(i)?;
        }
        DynLayout::new(&shape[..S::LEAVES], &stride[..S::LEAVES])
    }

    fn as_grouped_layout(&self) -> Result<GroupedLayout> {
        if S::RANK > MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        let mut groups = [0usize; MAX_RANK];
        for (axis, g) in groups[..S::RANK].iter_mut().enumerate() {
            *g = self.shape.mode_span(axis)?.1;
        }
        let mut grouped = GroupedLayout::new(self.as_flat_layout()?, &groups[..S::RANK])?;
        // A leaf type (depth zero) represents a CuTe integral shape, distinct
        // from a one-element tuple even though both have rank one.
        grouped.integral = S::DEPTH == 0;
        Ok(grouped)
    }
}

//-----------------------------------------------------------------------------
// Compact stride construction.

/// Constructs typed compact layouts with the leftmost leaf varying fastest.
/// Also serves as the direction marker for [`Compact`].
#[derive(Clone, Copy, Debug)]
pub struct LayoutLeft;

/// Constructs typed compact layouts with the rightmost leaf varying fastest.
/// Also serves as the direction marker for [`Compact`].
#[derive(Clone, Copy, Debug)]
pub struct LayoutRight;

/// Builds compact strides while preserving tuple structure and static arithmetic.
///
/// `Dir` is [`LayoutLeft`] or [`LayoutRight`] and determines the order in which
/// leaf extents multiply the running stride. The result is
/// `(stride_tree, start * product_of_extents)`. This low-level operation uses
/// ordinary integer multiplication; use the direction marker's `make` method
/// for checked layout construction.
pub trait Compact<Dir, Start: Integer>: Item {
    /// Stride tree with the same nested structure as the shape.
    type Stride: Item<Profile = Self::Profile>;
    /// Running stride after all extents have been multiplied into `Start`.
    type End: Integer;
    /// Assigns strides beginning at `start` and returns the final running stride.
    fn compact(self, start: Start) -> (Self::Stride, Self::End);
}

// Leaves and empty tuples are direction-independent; only `IntTuple` changes
// its recursive evaluation order with the direction marker.
impl<I, Dir, Start> Compact<Dir, Start> for I
where
    I: Integer + Debug + Mul<Start>,
    <I as Mul<Start>>::Output: Integer,
    Start: Integer + Debug,
{
    type Stride = Start;
    type End = <I as Mul<Start>>::Output;
    fn compact(self, start: Start) -> (Start, Self::End) {
        (start, self * start)
    }
}

impl<Dir, Start: Integer> Compact<Dir, Start> for Nil {
    type Stride = Nil;
    type End = Start;
    fn compact(self, start: Start) -> (Nil, Start) {
        (Nil, start)
    }
}

impl<H, T, Start> Compact<LayoutLeft, Start> for IntTuple<H, T>
where
    Start: Integer,
    H: Compact<LayoutLeft, Start>,
    T: Tuple + Compact<LayoutLeft, H::End>,
    T::Stride: Tuple,
{
    type Stride = IntTuple<H::Stride, T::Stride>;
    type End = T::End;
    fn compact(self, start: Start) -> (Self::Stride, Self::End) {
        let (h, end) = self.0.compact(start);
        let (t, end) = self.1.compact(end);
        (IntTuple(h, t), end)
    }
}

impl<H, T, Start> Compact<LayoutRight, Start> for IntTuple<H, T>
where
    Start: Integer,
    T: Tuple + Compact<LayoutRight, Start>,
    H: Compact<LayoutRight, T::End>,
    T::Stride: Tuple,
{
    type Stride = IntTuple<H::Stride, T::Stride>;
    type End = H::End;
    fn compact(self, start: Start) -> (Self::Stride, Self::End) {
        let (t, end) = self.1.compact(start);
        let (h, end) = self.0.compact(end);
        (IntTuple(h, t), end)
    }
}

impl LayoutLeft {
    /// Builds a compact layout with left-to-right cumulative strides.
    ///
    /// Validates extents and partial products before constructing strides,
    /// retaining static values in their types. The first leaf has stride one.
    ///
    /// ```
    /// use noodle_cute::{Int, shape};
    /// use noodle_cute::layout::{AffineLayoutMap, LayoutLeft, LayoutMap};
    ///
    /// let layout = LayoutLeft::make(shape![Int::<2>, (Int::<3>, Int::<4>)])?;
    /// assert_eq!(layout.rank(), 2);
    /// assert_eq!(layout.extent(1)?, 12);
    /// assert_eq!(layout.as_flat_layout()?.strides(), &[1, 2, 6]);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn make<S: Shape + Compact<Self, Int<1>>>(shape: S) -> Result<Layout<S, S::Stride>> {
        make_compact(shape, 0..S::LEAVES)
    }
}

impl LayoutRight {
    /// Builds a compact layout with right-to-left cumulative strides.
    ///
    /// Checks extents and partial products for overflow. The last leaf has
    /// stride one; logical linear indices still use left-major order.
    pub fn make<S: Shape + Compact<Self, Int<1>>>(shape: S) -> Result<Layout<S, S::Stride>> {
        make_compact(shape, (0..S::LEAVES).rev())
    }
}

// Typed stride construction uses ordinary multiplication. Reject negative
// extents first, then preflight partial products in the same leaf order so
// overflow before a zero extent returns an error rather than panicking, even
// when the final logical size is zero.
fn make_compact<Dir, S: Shape + Compact<Dir, Int<1>>>(
    shape: S,
    order: impl Iterator<Item = usize>,
) -> Result<Layout<S, S::Stride>> {
    for i in 0..S::LEAVES {
        if shape.leaf(i)? < 0 {
            return Err(Error::InvalidShape);
        }
    }
    let mut scale = 1;
    for i in order {
        scale = mul(scale, shape.leaf(i)?)?;
    }
    let (stride, _) = shape.clone().compact(Int);
    Layout::new(shape, stride)
}

//-----------------------------------------------------------------------------
// Type-level coalescing.

/// Selects a coalescing action from facts known at compile time.
///
/// The fold changes types only using static facts: a runtime extent equal to one
/// remains in the output tuple. Action codes are [`DISCARD`] for a static unit
/// extent, [`RETAIN`] to keep a leaf, and [`MERGE`] to merge it with the folded
/// tail's head.
#[doc(hidden)]
pub struct Decision<const N: u8>;

/// Drops a static unit extent.
#[doc(hidden)]
pub const DISCARD: u8 = 0;
/// Keeps the leaf as a separate mode.
#[doc(hidden)]
pub const RETAIN: u8 = 1;
/// Merges the leaf with the folded tail's head.
#[doc(hidden)]
pub const MERGE: u8 = 2;

/// Selects whether a leaf is discarded or retained when the folded tail is empty.
#[doc(hidden)]
pub const fn unit_decision(s: Option<isize>) -> u8 {
    if matches!(s, Some(1)) {
        DISCARD
    } else {
        RETAIN
    }
}

/// Selects whether to discard, retain, or merge a leaf with the folded tail's head.
#[doc(hidden)]
pub const fn merge_decision(s: Option<isize>, d: Option<isize>, next: Option<isize>) -> u8 {
    if matches!(s, Some(1)) {
        return DISCARD;
    }
    // Merge only when static values prove `s * d == next_stride` without
    // overflow. Coincidental equality between runtime values cannot change types.
    match (s, d, next) {
        (Some(a), Some(b), Some(c)) => match a.checked_mul(b) {
            Some(v) if v == c => MERGE,
            _ => RETAIN,
        },
        _ => RETAIN,
    }
}

// Reuse these constants in bounds and associated-type projections so generic
// const expressions refer to identical definitions.
#[doc(hidden)]
pub const UNIT_DECISION<S: Integer>: u8 = unit_decision(S::STATIC);
#[doc(hidden)]
pub const MERGE_DECISION<S: Integer, D: Integer, TD: Integer>: u8 =
    merge_decision(S::STATIC, D::STATIC, TD::STATIC);

/// Prepends a shape/stride leaf `(s, d)` to an empty folded tail.
#[doc(hidden)]
pub trait PrependToEmpty<S, D> {
    type Output;
    fn apply(s: S, d: D) -> Self::Output;
}

impl<S, D> PrependToEmpty<S, D> for Decision<DISCARD> {
    type Output = Layout<Nil, Nil>;
    fn apply(_: S, _: D) -> Self::Output {
        Layout {
            shape: Nil,
            stride: Nil,
        }
    }
}

impl<S, D> PrependToEmpty<S, D> for Decision<RETAIN> {
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
#[doc(hidden)]
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
#[doc(hidden)]
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

impl<S: Integer, D: Integer, TS, TSR, TD: Integer, TDR> Prepend<S, D>
    for Layout<IntTuple<TS, TSR>, IntTuple<TD, TDR>>
where
    Decision<{ MERGE_DECISION::<S, D, TD> }>: PrependToNonempty<S, D, TS, TSR, TD, TDR>,
{
    type Output = <Decision<{ MERGE_DECISION::<S, D, TD> }> as PrependToNonempty<
        S,
        D,
        TS,
        TSR,
        TD,
        TDR,
    >>::Output;
    fn prepend(self, s: S, d: D) -> Self::Output {
        <Decision<{ MERGE_DECISION::<S, D, TD> }> as PrependToNonempty<
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
#[doc(hidden)]
pub trait CoalesceFold {
    type Output;
    fn fold(self) -> Self::Output;
}

impl CoalesceFold for Layout<Nil, Nil> {
    type Output = Self;
    fn fold(self) -> Self {
        self
    }
}

// `S`/`D` are the first shape/stride leaf types; `TSR`/`TDR` are the remaining
// shape/stride tuples, folded first and then prepended with `(S, D)`.
impl<S, D, TSR, TDR> CoalesceFold for Layout<IntTuple<S, TSR>, IntTuple<D, TDR>>
where
    Layout<TSR, TDR>: CoalesceFold,
    <Layout<TSR, TDR> as CoalesceFold>::Output: Prepend<S, D>,
{
    type Output = <<Layout<TSR, TDR> as CoalesceFold>::Output as Prepend<S, D>>::Output;
    // Fold right to left so each leaf is compared with the head of the
    // already-coalesced tail.
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
#[doc(hidden)]
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

/// The exact typed layout produced by static coalescing of a shape/stride pair.
pub type CoalescedLayout<S, D> = <<Layout<
    <S as Flatten>::Output,
    <D as Flatten>::Output,
> as CoalesceFold>::Output as FinishCoalesce>::Output;

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
    /// Shape and merged-extent arithmetic is checked before the fold, and the
    /// resulting layout is validated before it is returned.
    pub fn coalesce(&self) -> Result<CoalescedLayout<S, D>> {
        self.size()?;
        // The type fold uses ordinary integer operators. Preflight the exact
        // dynamic products it can create so even giant empty shapes return a
        // checked error rather than panicking during a proven static merge.
        let mut next_shape = None;
        let mut next_stride = None;
        for i in (0..S::LEAVES).rev() {
            let shape_static = self.shape.leaf_static(i)?;
            if shape_static == Some(1) {
                continue;
            }
            let stride_static = self.stride.leaf_static(i)?;
            let value = self.shape.leaf(i)?;
            let merge = match (shape_static, stride_static, next_stride) {
                (Some(a), Some(b), Some(c)) => a.checked_mul(b) == Some(c),
                _ => false,
            };
            next_shape = Some(if merge {
                mul(value, next_shape.unwrap_or(1))?
            } else {
                value
            });
            next_stride = stride_static;
        }
        self.clone().flatten().fold().finish()
    }
}
