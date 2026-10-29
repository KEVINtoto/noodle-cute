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

use core::ops::Range;

use crate::int::{DynInt, Int, NON_NEGATIVE, add, mul};
use crate::tuple::{Item, Nil, Tuple};
use crate::{Assert, Error, IntTuple, Result, True};

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
    /// Returns the number of top-level modes, counting each nested tuple as one.
    /// This can differ from the total number of integer leaves.
    fn rank(&self) -> usize;

    /// Returns the size of top-level mode `axis`.
    ///
    /// A nested mode's extent is the product of its leaf extents; an empty mode
    /// has extent one. An invalid axis returns [`Error::AxisOutOfBounds`].
    fn extent(&self, axis: usize) -> Result<isize>;

    /// Returns the number of elements in the logical domain.
    ///
    /// Scalars have size one and layouts with a zero extent have size zero.
    /// Invalid extents or an overflowing product return an error.
    fn size(&self) -> Result<isize>;

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
    ///
    /// Default implementation uses the affine layout if linear index is out of
    /// the logical domain.
    fn index_extended(&self, linear: isize) -> Result<isize> {
        if linear >= 0 && linear < self.size()? {
            return self.index(linear);
        }
        match self.as_affine()? {
            Some(layout) => layout.index_extended(linear),
            None => Err(Error::CoordinateOutOfBounds),
        }
    }

    /// Maps one nonnegative extended coordinate per top-level mode.
    ///
    /// Affine layouts decompose each coordinate within its own mode, leaving
    /// that mode's last leaf unbounded. Unlike [`LayoutMap::index_extended`],
    /// extension never carries into the next top-level mode. A wrong coordinate
    /// count returns [`Error::StructureMismatch`]. An empty mode accepts only
    /// coordinate zero.
    ///
    /// Default implementation checks for nonnegative coordinates within extents
    /// and falls back to the affine layout if available.
    fn index_extended_of_coord(&self, coord: &[isize]) -> Result<isize> {
        if coord.len() != self.rank() {
            return Err(Error::StructureMismatch);
        }
        if coord.len() == self.rank()
            && coord
                .iter()
                .enumerate()
                .all(|(a, &c)| c >= 0 && self.extent(a).is_ok_and(|e| c < e))
        {
            return self.index_of_coord(coord);
        }
        match self.as_affine()? {
            Some(layout) if layout.rank() == self.rank() => layout.index_extended_of_coord(coord),
            Some(_) => Err(Error::StructureMismatch),
            None => Err(Error::CoordinateOutOfBounds),
        }
    }

    /// Returns the inclusive bounds `(min, max)` of reachable addresses.
    ///
    /// Empty domains return `None`. The interval may contain holes and does not
    /// imply injectivity. Affine layouts compute the bounds from the signed
    /// contribution of each leaf without enumerating the domain.
    ///
    /// Default implementation uses linear scanning.
    fn address_range(&self) -> Result<Option<(isize, isize)>> {
        let size = self.size()?;
        if size == 0 {
            return Ok(None);
        }
        let mut min = self.index(0)?;
        let mut max = min;
        for i in 1..size {
            let address = self.index(i)?;
            min = min.min(address);
            max = max.max(address);
        }
        Ok(Some((min, max)))
    }

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

    /// Exports flat runtime affine leaves and their top-level grouping.
    ///
    /// Returns `None` when no affine representation is available. Exporting a
    /// typed layout can return [`Error::CapacityExceeded`] even when its direct
    /// indexing operations succeed, because runtime metadata uses fixed arrays.
    fn as_affine(&self) -> Result<Option<GroupedLayout>>;
}

/// Layouts admitting a flat affine representation (hierarchy may be erased).
/// Implementations must export metadata that preserves the mapping from logical
/// linear indices to addresses.
pub trait AffineLayoutMap: LayoutMap {
    /// Same as [`LayoutMap::rank`].
    fn affine_rank(&self) -> usize;

    /// Returns the extent of flat leaf `leaf`, numbered head-first and
    /// depth-first across all top-level modes.
    ///
    /// Unlike [`LayoutMap::extent`], nested modes are not multiplied together.
    /// An invalid leaf returns [`Error::AxisOutOfBounds`].
    fn shape(&self, leaf: usize) -> Result<isize>;

    /// Returns the signed address stride of flat leaf `leaf`.
    ///
    /// Uses the same leaf numbering and errors as [`AffineLayoutMap::shape`].
    fn stride(&self, leaf: usize) -> Result<isize>;

    /// Returns the range of flat leaf indices for this layout.
    ///
    /// The range is half-open, starting at zero and ending at the number of leaves.
    fn leaves(&self) -> Range<usize>;

    /// Returns the start index and length of the flat leaf span for top-level mode `axis`.
    ///
    /// An invalid axis returns [`Error::AxisOutOfBounds`].
    fn mode_span(&self, axis: usize) -> Result<(usize, usize)>;

    /// Returns the flat shape/stride metadata; conversion fails when a static
    /// layout's leaf count exceeds `MAX_RANK`.
    fn as_flat_layout(&self) -> Result<DynLayout>;

    /// Returns flat leaves with their top-level grouping, used when CuTe
    /// algebra distributes over top-level modes (right-distributivity).
    fn as_grouped_layout(&self) -> Result<GroupedLayout>;
}

impl<T: AffineLayoutMap> LayoutMap for T {
    fn rank(&self) -> usize {
        self.affine_rank()
    }

    fn extent(&self, axis: usize) -> Result<isize> {
        let (start, len) = self.mode_span(axis)?;
        checked_size(self, start..start + len)
    }

    fn size(&self) -> Result<isize> {
        checked_size(self, self.leaves())
    }

    fn index(&self, linear: isize) -> Result<isize> {
        affine_index(self, self.leaves(), linear)
    }

    fn index_extended(&self, linear: isize) -> Result<isize> {
        self.size()?;
        affine_index_extended(self, self.leaves(), linear)
    }

    fn index_extended_of_coord(&self, coord: &[isize]) -> Result<isize> {
        if coord.len() != self.affine_rank() {
            return Err(Error::StructureMismatch);
        }
        self.size()?;
        let mut result = 0;
        for (axis, &c) in coord.iter().enumerate() {
            let (start, len) = self.mode_span(axis)?;
            let part = affine_index_extended(self, start..start + len, c)?;
            result = add(result, part)?;
        }
        Ok(result)
    }

    fn address_range(&self) -> Result<Option<(isize, isize)>> {
        affine_address_range(self, self.leaves())
    }

    fn is_injective(&self) -> Result<bool> {
        let proof = affine_injective(self, self.leaves())?;
        // The fast proof uses a sufficient condition. Fall back to exact
        // enumeration when it cannot decide an irregular stride pattern.
        match proof {
            Some(injective) => Ok(injective),
            None => is_injective_by_enumeration(self),
        }
    }

    fn as_affine(&self) -> Result<Option<GroupedLayout>> {
        self.as_grouped_layout().map(Some)
    }
}

/// Checks every pair of logical indices for equal addresses in `O(size^2)` time.
/// Used when the affine stride-separation proof is inconclusive.
pub(crate) fn is_injective_by_enumeration<L: LayoutMap>(layout: &L) -> Result<bool> {
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

#[allow(dead_code)]
/// Convert one logical linear coordinate to flat top-level coordinates.
/// The leftmost axis varies fastest; `out` must have length `rank`, and `linear`
/// must lie in `[0, size)`. Each nested mode produces one top-level linear
/// coordinate rather than separate coordinates for its leaves.
pub(crate) fn linear_to_coord<L: LayoutMap>(
    layout: &L,
    mut linear: isize,
    coord: &mut [isize],
) -> Result<()> {
    if coord.len() < layout.rank() {
        return Err(Error::StructureMismatch);
    }
    if linear < 0 || linear >= layout.size()? {
        return Err(Error::CoordinateOutOfBounds);
    }
    for (a, c) in coord.iter_mut().enumerate() {
        let s = layout.extent(a)?;
        *c = linear % s;
        linear /= s;
    }
    Ok(())
}

/// Creates a left-major compact identity map whose logical indices equal addresses.
/// The same construction constraints as [`DynLayout::left`] apply.
/// Note: this is the *linearized* identity. CuTe `make_identity_layout(shape)`
/// maps a coordinate to the coordinate tuple itself (ArithmeticTuple strides
/// `E<i>`); the coordinate-valued counterpart here is `identity_tensor`.
pub fn identity_layout(shape: &[isize]) -> Result<DynLayout> {
    DynLayout::left(shape)
}

/// CuTe `Layout::get_flat_coord(index)`: computes each leaf coordinate as
/// `(index / stride) % shape`, then combines leaves within each top-level mode
/// in left-major order and writes the coordinates to `out[..rank]`.
///
/// This inverts compact layouts. Unit extents have coordinate zero. `index`
/// must be nonnegative; non-unit leaves with nonpositive strides return
/// [`Error::Unsupported`]. This is not a general inverse for padded or overlapping
/// layouts, and it does not check whether `index` is a reachable address.
///
/// ```
/// use noodle_cute::layout::{DynLayout, get_flat_coord};
///
/// let layout = DynLayout::right(&[2, 3])?;
/// let mut coord = [0; 2];
/// get_flat_coord(&layout, 4, &mut coord)?;
/// assert_eq!(coord, [1, 1]);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
pub fn get_flat_coord<L: AffineLayoutMap>(
    layout: &L,
    index: isize,
    coord: &mut [isize],
) -> Result<()> {
    layout.size()?;
    let rank = layout.affine_rank();
    if coord.len() < rank {
        return Err(Error::StructureMismatch);
    }
    if index < 0 {
        return Err(Error::CoordinateOutOfBounds);
    }
    for (m, o) in coord[..rank].iter_mut().enumerate() {
        let (start, len) = layout.mode_span(m)?;
        let (mut c, mut scale) = (0, 1);
        for i in start..start + len {
            let (s, d) = (layout.shape(i)?, layout.stride(i)?);
            if s != 1 {
                if s <= 0 || d <= 0 {
                    return Err(Error::Unsupported);
                }
                c = add(c, mul((index / d) % s, scale)?)?;
            }
            scale = mul(scale, s)?;
        }
        *o = c;
    }
    Ok(())
}

//-----------------------------------------------------------------------------
// These affine algorithms access shape and stride by leaf index through
// `AffineLayoutMap::shape` and `AffineLayoutMap::stride`, so both `Layout` and
// `DynLayout` can use them. Typed layouts call them directly, without converting
// to fixed-capacity runtime metadata.

/// Validates all extents, then computes their product with checked arithmetic.
///
/// A zero extent makes the size zero, but must not hide another negative extent.
/// Check for zero before multiplying so an empty domain cannot overflow its
/// logical size. The product of an empty range is one.
pub(crate) fn checked_size<L: AffineLayoutMap>(layout: &L, leaves: Range<usize>) -> Result<isize> {
    let mut zero = false;
    for i in leaves.clone() {
        let s = layout.shape(i)?;
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
        n = mul(n, layout.shape(i)?)?;
    }
    Ok(n)
}

/// Decodes a bounded linear index as left-major mixed-radix leaf coordinates.
///
/// Validates `linear` against the checked size before dividing by extents, so
/// empty domains are rejected without division by zero. Flattening nested modes
/// in leaf order preserves the same logical traversal.
pub(crate) fn affine_index<L: AffineLayoutMap>(
    layout: &L,
    leaves: Range<usize>,
    mut linear: isize,
) -> Result<isize> {
    if linear < 0 || linear >= checked_size(layout, leaves.clone())? {
        return Err(Error::CoordinateOutOfBounds);
    }
    let mut result = 0;
    for i in leaves {
        let s = layout.shape(i)?;
        result = add(result, mul(linear % s, layout.stride(i)?)?)?;
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
pub(crate) fn affine_index_extended<L: AffineLayoutMap>(
    layout: &L,
    leaves: core::ops::Range<usize>,
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
        let s = layout.shape(i)?;
        if s <= 0 {
            return Err(Error::CoordinateOutOfBounds);
        }
        result = add(result, mul(linear % s, layout.stride(i)?)?)?;
        linear /= s;
    }
    add(result, mul(linear, layout.stride(last)?)?)
}

/// Computes inclusive address bounds by summing each leaf's signed endpoint.
/// Negative strides contribute to the minimum and positive strides to the
/// maximum. An empty domain has no reachable addresses and returns `None`.
pub(crate) fn affine_address_range<L: AffineLayoutMap>(
    layout: &L,
    leaves: core::ops::Range<usize>,
) -> Result<Option<(isize, isize)>> {
    if checked_size(layout, leaves.clone())? == 0 {
        return Ok(None);
    }
    let mut min = 0;
    let mut max = 0;
    for i in leaves {
        let delta = mul(layout.shape(i)? - 1, layout.stride(i)?)?;
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
pub(crate) fn affine_injective<L: AffineLayoutMap>(
    layout: &L,
    leaves: core::ops::Range<usize>,
) -> Result<Option<bool>> {
    if checked_size(layout, leaves.clone())? <= 1 {
        return Ok(Some(true));
    }
    let mut span = 0isize;
    let mut prev: Option<(usize, usize)> = None;
    loop {
        let mut next = None;
        for i in leaves.clone() {
            // Unit extents cannot introduce overlap.
            if layout.shape(i)? <= 1 {
                continue;
            }
            let d = layout.stride(i)?;
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
        span = add(span, mul(layout.shape(i)? - 1, d)?)?;
        prev = Some((abs, i));
    }
}

mod dynamic_layout;
mod static_layout;

pub use dynamic_layout::*;
pub use static_layout::*;
