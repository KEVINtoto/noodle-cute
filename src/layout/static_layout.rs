//! Typed affine layouts with static or mixed static/runtime leaves.
//!
//! Type-level coalescing uses only facts encoded in integer types. Intermediate
//! fold results use [`Layout`] directly and are validated when the fold finishes.

use core::ops::Mul;

use crate::int::{Int, Integer, add, mul};
use crate::tuple::{Flatten, Item, Nil};
use crate::{Error, IntTuple, Result};

use super::{
    DynLayout, GroupedLayout, LayoutMap, MAX_RANK, Shape, affine_address_range, affine_index,
    affine_index_extended, affine_injective, checked_size, injective_by_enumeration,
};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout<S, D> {
    /// Nonnegative extents arranged as an integer leaf or nested tuple.
    pub shape: S,
    /// Signed address increments with the same nested profile as `shape`.
    pub stride: D,
}

// Validate every extent before taking the product: a zero extent makes the
// domain empty, but must not hide a negative extent elsewhere in the shape.
pub(super) fn shape_size<S: Item>(shape: &S) -> Result<isize> {
    checked_size(0..S::LEAVES, |i| shape.leaf(i))
}

impl<S: Shape, D: Item<Profile = S::Profile>> Layout<S, D> {
    /// Constructs a layout after checking nonnegative extents, logical size,
    /// and reachable address endpoints with checked arithmetic.
    ///
    /// Negative extents return [`Error::InvalidShape`]; unrepresentable size or
    /// address calculations return [`Error::Overflow`]. Addresses may be negative
    /// or repeated, and an empty domain has no address endpoints to check.
    pub fn new(shape: S, stride: D) -> Result<Self> {
        shape_size(&shape)?;
        let layout = Self { shape, stride };
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

    /// Exports flattened runtime metadata for reflection or backend parameters.
    ///
    /// The result has one axis per leaf and discards the original mode grouping.
    /// Returns [`Error::CapacityExceeded`] if the leaf count exceeds [`MAX_RANK`].
    /// The exported values are validated by [`DynLayout::new`].
    pub fn to_dyn(&self) -> Result<DynLayout> {
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
}

impl<S: Shape, D: Item<Profile = S::Profile>> LayoutMap for Layout<S, D> {
    fn size(&self) -> Result<isize> {
        shape_size(&self.shape)
    }

    fn rank(&self) -> usize {
        S::RANK
    }

    fn extent(&self, axis: usize) -> Result<isize> {
        let (start, len) = self.shape.mode_span(axis)?;
        checked_size(start..start + len, |i| self.shape.leaf(i))
    }

    fn index(&self, linear: isize) -> Result<isize> {
        affine_index(
            0..S::LEAVES,
            |i| self.shape.leaf(i),
            |i| self.stride.leaf(i),
            linear,
        )
    }

    fn index_extended(&self, linear: isize) -> Result<isize> {
        self.size()?;
        affine_index_extended(
            0..S::LEAVES,
            |i| self.shape.leaf(i),
            |i| self.stride.leaf(i),
            linear,
        )
    }

    // Evaluate each mode directly over its leaf span. Avoiding `as_affine`
    // keeps this operation independent of the runtime metadata capacity.
    fn index_extended_of_coord(&self, coord: &[isize]) -> Result<isize> {
        self.size()?;
        if coord.len() != S::RANK {
            return Err(Error::StructureMismatch);
        }
        let mut result = 0;
        for (axis, &c) in coord.iter().enumerate() {
            let (start, len) = self.shape.mode_span(axis)?;
            let part = affine_index_extended(
                start..start + len,
                |i| self.shape.leaf(i),
                |i| self.stride.leaf(i),
                c,
            )?;
            result = add(result, part)?;
        }
        Ok(result)
    }

    // Both the flattened leaf count and the top-level mode count must fit
    // within `MAX_RANK`; `to_dyn` checks the leaf count below.
    fn as_affine(&self) -> Result<Option<GroupedLayout>> {
        if S::RANK > MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        let mut groups = [0usize; MAX_RANK];
        for (axis, g) in groups[..S::RANK].iter_mut().enumerate() {
            *g = self.shape.mode_span(axis)?.1;
        }
        let mut grouped = GroupedLayout::new(self.to_dyn()?, &groups[..S::RANK])?;
        // A leaf type (depth zero) represents a CuTe integral shape, distinct
        // from a one-element tuple even though both have rank one.
        grouped.integral = S::DEPTH == 0;
        Ok(Some(grouped))
    }

    fn address_range(&self) -> Result<Option<(isize, isize)>> {
        affine_address_range(
            0..S::LEAVES,
            |i| self.shape.leaf(i),
            |i| self.stride.leaf(i),
        )
    }

    fn is_injective(&self) -> Result<bool> {
        let proof = affine_injective(
            0..S::LEAVES,
            |i| self.shape.leaf(i),
            |i| self.stride.leaf(i),
        )?;
        // The fast proof uses a sufficient separation condition. If it cannot
        // decide, enumerate exactly so irregular strides are not misclassified.
        match proof {
            Some(injective) => Ok(injective),
            None => injective_by_enumeration(self),
        }
    }
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
