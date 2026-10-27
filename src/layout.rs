//! Affine layouts with typed or runtime shapes and signed strides.
//!
//! Logical linear indices use CuTe's left-major order: the leftmost leaf varies
//! fastest, even when strides describe right-major storage. Nested tuples retain
//! their top-level modes, each of which can also be addressed by a linear
//! coordinate within that mode. Empty tuples represent scalars of size one;
//! any zero extent makes the logical domain empty.
//!
//! [`Layout`] preserves static integer types and tuple structure. [`DynLayout`]
//! stores flat runtime leaves, while [`GroupedLayout`] also records top-level
//! mode boundaries. Only the runtime metadata is bounded by [`MAX_RANK`].

use core::fmt::Debug;
use core::ops::Mul;

use crate::int::{DynInt, Int, Integer, NON_NEGATIVE, add, mul};
use crate::tuple::{Item, Nil, Tuple};
use crate::{Assert, Error, IntTuple, Result, True};

/// Maximum number of leaves or top-level modes in fixed runtime metadata.
///
/// Typed layouts and tuples have no rank limit imposed by this constant, but
/// exporting them through [`Layout::to_dyn`] or [`LayoutMap::as_affine`] can
/// return [`Error::CapacityExceeded`].
pub const MAX_RANK: usize = 32;

/// Selects an entire top-level mode or fixes it at one bounded coordinate.
/// Used by [`DynLayout::slice`]; strided ranges are not supported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slice {
    /// Retains the mode, preserving its order relative to other retained modes.
    All,
    /// Fixes the mode at this coordinate and removes it from the output shape.
    Index(isize),
}

/// Builds a recursive shape tuple from static or runtime integer leaves.
///
/// Use `Int::<N>` for static extents and `DynInt(n)` or `isize` for runtime
/// extents. Parentheses introduce nested modes. This macro only constructs the
/// tuple; layout constructors validate runtime extents and arithmetic.
#[macro_export]
macro_rules! shape { ($($xs:tt)*) => { $crate::int_tuple!($($xs)*) }; }

/// Builds a stride tuple whose nesting must match the corresponding shape.
///
/// Strides may be positive, zero (broadcast), or negative (reversed traversal).
/// The macro itself does not validate the resulting address range.
#[macro_export]
macro_rules! stride { ($($xs:tt)*) => { $crate::int_tuple!($($xs)*) }; }

/// Builds a coordinate tuple, preserving any nested modes in its type.
///
/// This constructs tuple data without checking coordinate bounds.
/// [`LayoutMap::index_of_coord`] instead accepts a flat slice containing one
/// linear coordinate per top-level mode.
#[macro_export]
macro_rules! coord { ($($xs:tt)*) => { $crate::int_tuple!($($xs)*) }; }

/// An integer leaf or nested tuple that can serve as a layout shape.
///
/// Negative [`Int`] extents are rejected by trait bounds. Runtime extents are
/// checked by layout constructors, since this marker alone cannot validate
/// their values. [`Nil`] denotes a scalar with no axes and size one.
pub trait Shape: Item {}
impl<const N: isize> Shape for Int<N> where Assert<{ NON_NEGATIVE::<N> }>: True {}
impl Shape for DynInt {}
impl Shape for isize {}
impl Shape for Nil {}
impl<H: Shape, T: Tuple + Shape> Shape for IntTuple<H, T> {}

/// A checked map from CuTe left-major logical coordinates to signed addresses.
///
/// Logical order is independent of stride order: the leftmost leaf varies
/// fastest. Addresses may be negative, repeated, or separated by holes.
/// A scalar has size one; a zero extent makes the domain empty. Arithmetic
/// that cannot be represented as an `isize` returns [`Error::Overflow`].
pub trait LayoutMap: Clone {
    /// Returns the number of elements in the logical domain.
    ///
    /// Scalars have size one and layouts with a zero extent have size zero.
    /// Invalid extents or an overflowing product return an error.
    fn size(&self) -> Result<isize>;

    /// Returns the number of top-level modes, counting each nested tuple as one.
    /// This can differ from the total number of integer leaves.
    fn rank(&self) -> usize;

    /// Returns the size of top-level mode `axis`.
    ///
    /// A nested mode's extent is the product of its leaf extents; an empty mode
    /// has extent one. An invalid axis returns [`Error::AxisOutOfBounds`].
    fn extent(&self, axis: usize) -> Result<isize>;

    /// Maps a left-major linear index in `[0, size)` to a signed address.
    ///
    /// Negative indices and indices at or beyond the logical size return
    /// [`Error::CoordinateOutOfBounds`]. Strides determine the address, so
    /// consecutive indices need not produce consecutive or distinct addresses.
    fn index(&self, linear: isize) -> Result<isize>;

    /// Maps one bounded coordinate per top-level mode to a signed address.
    ///
    /// A nested mode is represented by a single left-major linear coordinate.
    /// A slice of the wrong length returns [`Error::StructureMismatch`]; each
    /// coordinate must be in `[0, extent)` or the result is
    /// [`Error::CoordinateOutOfBounds`]. The default implementation encodes the
    /// coordinates in left-major order and calls [`LayoutMap::index`].
    fn index_of_coord(&self, coord: &[isize]) -> Result<isize> {
        self.index(coord_to_linear(self, coord)?)
    }

    /// Maps a nonnegative linear index using CuTe's extended domain.
    ///
    /// For affine layouts, every leaf except the last is reduced modulo its
    /// extent, and the last leaf absorbs the remaining quotient. Thus indices
    /// at or beyond `size` continue along the last leaf's stride. Within the
    /// logical domain this agrees with [`LayoutMap::index`]. Negative indices,
    /// or extensions a map cannot represent, return
    /// [`Error::CoordinateOutOfBounds`].
    fn index_extended(&self, linear: isize) -> Result<isize>;

    /// Maps one nonnegative extended coordinate per top-level mode.
    ///
    /// Affine layouts decompose each coordinate within its own mode, leaving
    /// that mode's last leaf unbounded. Unlike [`LayoutMap::index_extended`],
    /// extension never carries into the next top-level mode. A wrong coordinate
    /// count returns [`Error::StructureMismatch`]. An empty mode accepts only
    /// coordinate zero.
    fn index_extended_of_coord(&self, coord: &[isize]) -> Result<isize>;

    /// Exports flat runtime affine leaves and their top-level grouping.
    ///
    /// Returns `None` when no affine representation is available. Exporting a
    /// typed layout can return [`Error::CapacityExceeded`] even when its direct
    /// indexing operations succeed, because runtime metadata uses fixed arrays.
    fn as_affine(&self) -> Result<Option<GroupedLayout>>;

    /// Returns the inclusive bounds `(min, max)` of reachable addresses.
    ///
    /// Empty domains return `None`. The interval may contain holes and does not
    /// imply injectivity. Affine layouts compute the bounds from the signed
    /// contribution of each leaf without enumerating the domain.
    fn address_range(&self) -> Result<Option<(isize, isize)>>;

    /// Returns the address span `max - min + 1`, including padding holes.
    ///
    /// The empty domain yields zero. For a nonempty affine layout the span is
    /// `sum((extent - 1) * |stride|) + 1`, so negative strides contribute by
    /// magnitude. For nonnegative layouts starting at zero, this also equals
    /// the usual CuTe `cosize` value.
    fn cosize(&self) -> Result<isize> {
        match self.address_range()? {
            Some((min, max)) => add(max.checked_sub(min).ok_or(Error::Overflow)?, 1),
            None => Ok(0),
        }
    }

    /// Returns `max_address + 1`, the storage length required from address zero.
    ///
    /// Returns zero for an empty domain and [`Error::Unsupported`] if any
    /// reachable address is negative. Unlike [`LayoutMap::cosize`], this includes
    /// the unused prefix `[0, min_address)` when the first address is positive.
    fn address_end(&self) -> Result<isize> {
        match self.address_range()? {
            Some((min, max)) if min >= 0 => add(max, 1),
            Some(_) => Err(Error::Unsupported),
            None => Ok(0),
        }
    }

    /// Returns whether all logical elements have distinct addresses.
    ///
    /// Empty and single-element domains are injective. Affine layouts first try
    /// a stride-separation proof, then compare address pairs if the proof is
    /// inconclusive. The fallback takes `O(size^2)` time.
    fn is_injective(&self) -> Result<bool>;
}

/// Checks every pair of logical indices for equal addresses in `O(size^2)` time.
/// Used when the affine stride-separation proof is inconclusive.
pub(crate) fn injective_by_enumeration<L: LayoutMap>(layout: &L) -> Result<bool> {
    let n = layout.size()?;
    for i in 0..n {
        for j in 0..i {
            if layout.index(i)? == layout.index(j)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Encodes bounded top-level coordinates in left-major logical order.
///
/// Uses extents, not address strides:
/// `linear = c0 + extent0 * c1 + extent0 * extent1 * c2 + ...`.
/// A coordinate-count mismatch, an empty domain, out-of-bounds coordinates, or
/// overflowing arithmetic returns an error before the map is evaluated.
pub(crate) fn coord_to_linear<L: LayoutMap>(layout: &L, coord: &[isize]) -> Result<isize> {
    if coord.len() != layout.rank() {
        return Err(Error::StructureMismatch);
    }
    if layout.size()? == 0 {
        return Err(Error::CoordinateOutOfBounds);
    }
    let mut index = 0;
    let mut scale = 1;
    for (axis, &c) in coord.iter().enumerate() {
        let extent = layout.extent(axis)?;
        if c < 0 || c >= extent {
            return Err(Error::CoordinateOutOfBounds);
        }
        index = add(index, mul(c, scale)?)?;
        scale = mul(scale, extent)?;
    }
    Ok(index)
}

//-----------------------------------------------------------------------------
// These affine algorithms access shape and stride by leaf index, so both
// `Layout` and `DynLayout` can use them. Typed layouts call them directly,
// without converting to fixed-capacity runtime metadata.

/// Validates all extents, then computes their product with checked arithmetic.
///
/// A zero extent makes the size zero, but must not hide another negative extent.
/// Check for zero before multiplying so an empty domain cannot overflow its
/// logical size. The product of an empty range is one.
pub(crate) fn checked_size(
    leaves: core::ops::Range<usize>,
    shape: impl Fn(usize) -> Result<isize>,
) -> Result<isize> {
    let mut zero = false;
    for i in leaves.clone() {
        let s = shape(i)?;
        if s < 0 {
            return Err(Error::InvalidShape);
        }
        zero |= s == 0;
    }
    if zero {
        return Ok(0);
    }
    let mut n = 1;
    for i in leaves {
        n = mul(n, shape(i)?)?;
    }
    Ok(n)
}

/// Decodes a bounded linear index as left-major mixed-radix leaf coordinates.
///
/// Validates `linear` against the checked size before dividing by extents, so
/// empty domains are rejected without division by zero. Flattening nested modes
/// in leaf order preserves the same logical traversal.
pub(crate) fn affine_index(
    leaves: core::ops::Range<usize>,
    shape: impl Fn(usize) -> Result<isize>,
    stride: impl Fn(usize) -> Result<isize>,
    mut linear: isize,
) -> Result<isize> {
    if linear < 0 || linear >= checked_size(leaves.clone(), &shape)? {
        return Err(Error::CoordinateOutOfBounds);
    }
    let mut result = 0;
    for i in leaves {
        let s = shape(i)?;
        result = add(result, mul(linear % s, stride(i)?)?)?;
        linear /= s;
    }
    Ok(result)
}

/// Evaluates CuTe's extended `crd2idx` on the given range of affine leaves.
///
/// Leading leaves take `linear % extent`; the last takes the entire remaining
/// quotient, without consulting its extent. Negative indices and nonpositive
/// leading extents are rejected. A mode without leaves accepts only index zero.
/// Callers validate the shape separately; all address arithmetic is checked.
pub(crate) fn affine_index_extended(
    leaves: core::ops::Range<usize>,
    shape: impl Fn(usize) -> Result<isize>,
    stride: impl Fn(usize) -> Result<isize>,
    mut linear: isize,
) -> Result<isize> {
    if linear < 0 {
        return Err(Error::CoordinateOutOfBounds);
    }
    if leaves.is_empty() {
        return if linear == 0 {
            Ok(0)
        } else {
            Err(Error::CoordinateOutOfBounds)
        };
    }
    let last = leaves.end - 1;
    let mut result = 0;
    for i in leaves.start..last {
        let s = shape(i)?;
        if s <= 0 {
            return Err(Error::CoordinateOutOfBounds);
        }
        result = add(result, mul(linear % s, stride(i)?)?)?;
        linear /= s;
    }
    add(result, mul(linear, stride(last)?)?)
}

/// Computes inclusive address bounds by summing each leaf's signed endpoint.
/// Negative strides contribute to the minimum and positive strides to the
/// maximum. An empty domain has no reachable addresses and returns `None`.
pub(crate) fn affine_address_range(
    leaves: core::ops::Range<usize>,
    shape: impl Fn(usize) -> Result<isize>,
    stride: impl Fn(usize) -> Result<isize>,
) -> Result<Option<(isize, isize)>> {
    if checked_size(leaves.clone(), &shape)? == 0 {
        return Ok(None);
    }
    let mut min = 0;
    let mut max = 0;
    for i in leaves {
        let delta = mul(shape(i)? - 1, stride(i)?)?;
        min = add(min, delta.min(0))?;
        max = add(max, delta.max(0))?;
    }
    Ok(Some((min, max)))
}

/// Tries to prove injectivity by separating the address spans of affine leaves.
///
/// Processes leaves of extent greater than one in ascending
/// `(|stride|, leaf_index)` order. Each stride must exceed the span covered by
/// preceding leaves, ensuring that translated blocks cannot overlap. Repeated
/// scans select the next leaf in `O(leaves^2)` time without a sorting buffer or
/// a [`MAX_RANK`] limit.
///
/// `Some(true)` is a successful proof; `Some(false)` identifies a nontrivial
/// zero-stride leaf. `None` is inconclusive and requires exact enumeration.
pub(crate) fn affine_injective(
    leaves: core::ops::Range<usize>,
    shape: impl Fn(usize) -> Result<isize>,
    stride: impl Fn(usize) -> Result<isize>,
) -> Result<Option<bool>> {
    if checked_size(leaves.clone(), &shape)? <= 1 {
        return Ok(Some(true));
    }
    let mut span = 0isize;
    let mut prev: Option<(usize, usize)> = None;
    loop {
        let mut next = None;
        for i in leaves.clone() {
            // Unit extents cannot introduce overlap.
            if shape(i)? <= 1 {
                continue;
            }
            let d = stride(i)?;
            if d == 0 {
                return Ok(Some(false));
            }
            let key = (d.unsigned_abs(), i);
            if prev.is_none_or(|p| key > p) && next.is_none_or(|n| key < n) {
                next = Some(key);
            }
        }
        let Some((abs, i)) = next else {
            return Ok(Some(true));
        };
        // Magnitudes cover both forward and reversed axes. The magnitude of
        // `isize::MIN` cannot be represented as an `isize`.
        let d = isize::try_from(abs).map_err(|_| Error::Overflow)?;
        if d <= span {
            return Ok(None);
        }
        span = add(span, mul(shape(i)? - 1, d)?)?;
        prev = Some((abs, i));
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

// Typed stride construction uses ordinary multiplication. Preflight partial
// products in the same leaf order so overflow before a zero extent returns an
// error rather than panicking, even when the final logical size is zero.
fn make_compact<Dir, S: Shape + Compact<Dir, Int<1>>>(
    shape: S,
    order: impl Iterator<Item = usize>,
) -> Result<Layout<S, S::Stride>> {
    static_layout::shape_size(&shape)?;
    let mut scale = 1;
    for i in order {
        scale = mul(scale, shape.leaf(i)?)?;
    }
    let (stride, _) = shape.clone().compact(Int);
    Layout::new(shape, stride)
}

mod dynamic_layout;
mod static_layout;

pub use dynamic_layout::*;
pub use static_layout::*;
