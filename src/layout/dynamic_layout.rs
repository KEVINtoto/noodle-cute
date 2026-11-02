use crate::{Error, Result, Slice};

use super::{AffineLayoutMap, LayoutMap, add, mul};

/// Maximum number of leaves or top-level modes in fixed runtime metadata.
///
/// Typed layouts and tuples have no rank limit imposed by this constant, but
/// exporting them through [`AffineLayoutMap::as_flat_layout`] or [`LayoutMap::as_affine`] can
/// return [`Error::CapacityExceeded`].
pub const MAX_RANK: usize = 32;

/// An allocation-free affine layout with runtime extents and strides.
///
/// Every axis is a leaf (i.e., flattened, not hierarchical), and at most
/// [`MAX_RANK`] leaves can be stored. An empty shape describes a scalar
/// with size 1 and address 0; any zero extent makes the logical domain empty.
/// Only the first `rank` entries of each array are active.
#[derive(Clone, Copy, Debug, Eq)]
pub struct DynLayout {
    pub(crate) shape: [isize; MAX_RANK],
    pub(crate) stride: [isize; MAX_RANK],
    pub(crate) rank: usize,
}

// Entries past `rank` are storage, not layout data.
impl PartialEq for DynLayout {
    fn eq(&self, other: &Self) -> bool {
        self.shapes() == other.shapes() && self.strides() == other.strides()
    }
}

impl DynLayout {
    /// Creates the rank-0 scalar used as the starting point for layout builders.
    pub(crate) const fn scalar() -> Self {
        Self {
            shape: [0; MAX_RANK],
            stride: [0; MAX_RANK],
            rank: 0,
        }
    }

    /// Creates a layout from equally sized shape and stride slices.
    ///
    /// Extents must be nonnegative, and the rank must not exceed `MAX_RANK`.
    /// Construction checks for overflow in the logical size and reachable
    /// address bounds. Zero and negative strides are allowed; overlapping
    /// addresses are accepted and can be checked with `LayoutMap::is_injective`.
    ///
    /// ```
    /// use noodle_cute::layout::{DynLayout, LayoutMap};
    ///
    /// let reversed = DynLayout::new(&[3], &[-2])?;
    /// assert_eq!(reversed.index(2)?, -4);
    /// assert_eq!(reversed.address_range()?, Some((-4, 0)));
    /// let empty = DynLayout::new(&[0], &[1])?;
    /// assert_eq!(empty.address_range()?, None);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn new(shape: &[isize], stride: &[isize]) -> Result<Self> {
        if shape.len() != stride.len() {
            return Err(Error::StructureMismatch);
        }
        if shape.len() > MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        let mut out = Self::scalar();
        out.rank = shape.len();
        out.shape[..shape.len()].copy_from_slice(shape);
        out.stride[..stride.len()].copy_from_slice(stride);
        out.size()?;
        out.address_range()?;
        Ok(out)
    }

    /// Creates a compact layout whose leftmost axis has stride 1.
    /// Each subsequent stride is the product of the extents to its left.
    pub fn left(shape: &[isize]) -> Result<Self> {
        Self::compact(shape, false)
    }

    /// Creates a compact layout whose rightmost axis has stride 1.
    /// Each preceding stride is the product of the extents to its right.
    /// `LayoutMap::index` still decodes logical indices with the leftmost axis
    /// varying fastest, regardless of this physical storage order.
    pub fn right(shape: &[isize]) -> Result<Self> {
        Self::compact(shape, true)
    }

    fn compact(shape: &[isize], right: bool) -> Result<Self> {
        if shape.len() > MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        // Report a negative extent before any partial-product overflow.
        if shape.iter().any(|&s| s < 0) {
            return Err(Error::InvalidShape);
        }
        let mut strides = [0; MAX_RANK];
        let mut scale = 1;
        for j in 0..shape.len() {
            let i = if right { shape.len() - 1 - j } else { j };
            strides[i] = scale;
            scale = mul(scale, shape[i])?;
        }
        Self::new(shape, &strides[..shape.len()])
    }

    /// Returns the active leaf extents in logical axis order.
    pub fn shapes(&self) -> &[isize] {
        &self.shape[..self.rank]
    }

    /// Returns the signed address strides corresponding to [`Self::shapes`].
    pub fn strides(&self) -> &[isize] {
        &self.stride[..self.rank]
    }

    /// Selects a sublayout using exactly one selector per axis.
    ///
    /// Fixed axes are removed; `Slice::All` axes retain their extents and
    /// strides. The returned offset is the address contribution of the fixed
    /// coordinates and must be added when accessing the original storage.
    /// Fixed coordinates are checked and their offset is computed even when a
    /// retained axis is empty. Out-of-bounds coordinates and offset overflow
    /// return errors.
    ///
    /// ```
    /// use noodle_cute::{Slice, layout::{DynLayout, LayoutMap}};
    ///
    /// let layout = DynLayout::left(&[2, 3])?;
    /// let (column, offset) = layout.slice(&[Slice::All, Slice::Index(2)])?;
    /// assert_eq!(offset, 4);
    /// assert_eq!(offset + column.index(1)?, layout.index_of_coord(&[1, 2])?);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn slice(&self, selectors: &[Slice]) -> Result<(Self, isize)> {
        if selectors.len() != self.rank {
            return Err(Error::StructureMismatch);
        }
        let mut result = LayoutBuilder::new();
        let mut offset = 0;
        for (i, s) in selectors.iter().enumerate() {
            match *s {
                Slice::All => result.push(self.shape[i], self.stride[i])?,
                Slice::Index(c) => {
                    if c < 0 || c >= self.shape[i] {
                        return Err(Error::CoordinateOutOfBounds);
                    }
                    offset = add(offset, mul(c, self.stride[i])?)?;
                }
            }
        }
        Ok((result.finish_flat()?, offset))
    }

    /// CuTe `filter(layout)`: treats stride-0 leaves as size 1, then coalesces.
    ///
    /// For a nonempty domain, this preserves the set of reachable addresses,
    /// but may change the logical size and coordinate mapping. Replacing a
    /// zero extent on a stride-0 leaf can also make an empty domain nonempty.
    /// The resulting size and address bounds must remain representable.
    pub fn filter(&self) -> Result<Self> {
        let mut zeros = *self;
        for a in 0..zeros.rank {
            if zeros.stride[a] == 0 {
                zeros.shape[a] = 1;
            }
        }
        let out = zeros.coalesce()?;
        out.size()?;
        out.address_range()?;
        Ok(out)
    }
}

impl AffineLayoutMap for DynLayout {
    fn affine_rank(&self) -> usize {
        self.rank
    }

    // Each axis is a single leaf, so leaf and axis indices coincide.
    fn leaf_shape(&self, leaf: usize) -> Result<isize> {
        self.shapes()
            .get(leaf)
            .copied()
            .ok_or(Error::AxisOutOfBounds)
    }

    fn leaf_stride(&self, leaf: usize) -> Result<isize> {
        self.strides()
            .get(leaf)
            .copied()
            .ok_or(Error::AxisOutOfBounds)
    }

    fn leaf_count(&self) -> usize {
        self.rank
    }

    fn mode_span(&self, axis: usize) -> Result<(usize, usize)> {
        if axis >= self.rank {
            return Err(Error::AxisOutOfBounds);
        }
        Ok((axis, 1))
    }

    fn as_flat_layout(&self) -> Result<DynLayout> {
        Ok(*self)
    }

    fn as_grouped_layout(&self) -> Result<GroupedLayout> {
        GroupedLayout::from_flat(*self)
    }
}

/// Runtime affine leaves grouped into top-level modes, as used by CuTe layout
/// algebra operations such as composition, complement, divide, and product.
///
/// Each mode is linearized with its leftmost leaf varying fastest, matching
/// `super::Layout`'s nested modes; deeper nesting is flattened. Both the number
/// of leaves and the number of top-level modes are bounded by `MAX_RANK`.
/// The `integral` flag distinguishes a CuTe integer shape from a tuple shape;
/// an integral shape has exactly one mode containing one leaf.
#[derive(Clone, Copy, Debug, Eq)]
pub struct GroupedLayout {
    pub(crate) flat: DynLayout,
    pub(crate) groups: [u8; MAX_RANK],
    pub(crate) rank: usize,
    pub(crate) integral: bool,
}

impl PartialEq for GroupedLayout {
    fn eq(&self, other: &Self) -> bool {
        self.flat == other.flat
            && self.groups().eq(other.groups())
            && self.integral == other.integral
    }
}

impl GroupedLayout {
    /// Groups consecutive leaves of `flat` into top-level tuple modes.
    ///
    /// `groups[a]` gives the leaf count of mode `a`; the counts must sum to
    /// `flat.rank()`, and there may be at most `MAX_RANK` modes. A mode with no
    /// leaves represents an empty tuple with extent 1. This constructor always
    /// records a tuple shape, including when there is just one mode.
    ///
    /// ```
    /// use noodle_cute::layout::{DynLayout, GroupedLayout, LayoutMap};
    ///
    /// let flat = DynLayout::left(&[2, 3, 4])?;
    /// let grouped = GroupedLayout::new(flat, &[2, 1])?;
    /// assert_eq!(grouped.rank(), 2);
    /// assert_eq!(grouped.extent(0)?, 6);
    /// assert_eq!(grouped.index_of_coord(&[5, 2])?, 17);
    /// assert_eq!(grouped.flat().index(17)?, 17);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn new(flat: DynLayout, groups: &[usize]) -> Result<Self> {
        if groups.len() > MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        let total = groups.iter().try_fold(0usize, |sum, &n| sum.checked_add(n));
        if total != Some(flat.rank) {
            return Err(Error::StructureMismatch);
        }
        let mut out = Self {
            flat,
            groups: [0; MAX_RANK],
            rank: groups.len(),
            integral: false,
        };
        for (g, &n) in out.groups.iter_mut().zip(groups) {
            *g = n as u8;
        }
        Ok(out)
    }

    /// Makes each flat leaf a separate top-level mode.
    /// Rank 1 is recorded as a CuTe integer shape, matching `make_layout(n)`.
    pub fn from_flat(flat: DynLayout) -> Result<Self> {
        let groups = [1usize; MAX_RANK];
        let mut out = Self::new(flat, &groups[..flat.rank])?;
        out.integral = flat.rank == 1;
        Ok(out)
    }

    /// Returns the flat affine metadata for all leaves.
    pub fn flat(&self) -> &DynLayout {
        &self.flat
    }

    /// Returns leaf extents; use `LayoutMap::extent` for top-level mode sizes.
    pub fn shapes(&self) -> &[isize] {
        self.flat.shapes()
    }

    /// Returns the signed address stride of each flat leaf.
    pub fn strides(&self) -> &[isize] {
        self.flat.strides()
    }

    /// Iterates over the leaf count of each top-level mode.
    pub fn groups(&self) -> impl Iterator<Item = usize> + '_ {
        self.groups[..self.rank].iter().map(|&g| g as usize)
    }

    /// Reports whether the shape is a CuTe integer rather than a tuple.
    pub fn is_integral(&self) -> bool {
        self.integral
    }

    /// Returns the flat sublayout of top-level mode `axis`.
    /// An empty mode becomes a rank-0 scalar; an invalid axis returns an error.
    pub fn mode(&self, axis: usize) -> Result<DynLayout> {
        let (start, len) = self.mode_span(axis)?;
        DynLayout::new(
            &self.flat.shapes()[start..start + len],
            &self.flat.strides()[start..start + len],
        )
    }
}

impl AffineLayoutMap for GroupedLayout {
    fn affine_rank(&self) -> usize {
        self.rank
    }

    fn leaf_shape(&self, leaf: usize) -> Result<isize> {
        self.flat.leaf_shape(leaf)
    }

    fn leaf_stride(&self, leaf: usize) -> Result<isize> {
        self.flat.leaf_stride(leaf)
    }

    fn leaf_count(&self) -> usize {
        self.flat.leaf_count()
    }

    fn mode_span(&self, axis: usize) -> Result<(usize, usize)> {
        if axis >= self.rank {
            return Err(Error::AxisOutOfBounds);
        }
        let start = self.groups[..axis].iter().map(|&g| g as usize).sum();
        Ok((start, self.groups[axis] as usize))
    }

    fn as_flat_layout(&self) -> Result<DynLayout> {
        Ok(self.flat)
    }

    fn as_grouped_layout(&self) -> Result<GroupedLayout> {
        Ok(*self)
    }
}

/// Accumulates runtime leaves into top-level modes for algebra results.
///
/// `push`/`push_leaves` append leaves to the open mode and `close_mode` ends it, so
/// consecutive pushes before one `close_mode` merge into a single mode.
#[derive(Clone, Copy)]
pub(crate) struct LayoutBuilder {
    shape: [isize; MAX_RANK],
    stride: [isize; MAX_RANK],
    leaves: usize,
    groups: [usize; MAX_RANK],
    modes: usize,
    closed: usize,
}

impl LayoutBuilder {
    pub(crate) const fn new() -> Self {
        Self {
            shape: [0; MAX_RANK],
            stride: [0; MAX_RANK],
            leaves: 0,
            groups: [0; MAX_RANK],
            modes: 0,
            closed: 0,
        }
    }

    pub(crate) fn push(&mut self, s: isize, d: isize) -> Result<()> {
        if self.leaves == MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        self.shape[self.leaves] = s;
        self.stride[self.leaves] = d;
        self.leaves += 1;
        Ok(())
    }

    pub(crate) fn push_leaves(&mut self, layout: &DynLayout) -> Result<()> {
        for (&s, &d) in layout.shapes().iter().zip(layout.strides()) {
            self.push(s, d)?;
        }
        Ok(())
    }

    /// Ends the open mode; a mode may have no leaves.
    pub(crate) fn close_mode(&mut self) -> Result<()> {
        if self.modes == MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        self.groups[self.modes] = self.leaves - self.closed;
        self.modes += 1;
        self.closed = self.leaves;
        Ok(())
    }

    /// Appends every top-level mode of `layout` as its own mode.
    pub(crate) fn push_modes(&mut self, layout: &GroupedLayout) -> Result<()> {
        for axis in 0..layout.rank() {
            self.push_leaves(&layout.mode(axis)?)?;
            self.close_mode()?;
        }
        Ok(())
    }

    /// Validates the leaves as a flat layout, ignoring mode boundaries.
    pub(crate) fn finish_flat(&self) -> Result<DynLayout> {
        DynLayout::new(&self.shape[..self.leaves], &self.stride[..self.leaves])
    }

    /// Validates the leaves and groups them by the closed modes.
    pub(crate) fn finish(&self) -> Result<GroupedLayout> {
        GroupedLayout::new(self.finish_flat()?, &self.groups[..self.modes])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_groups_closed_modes_and_enforces_capacity() {
        let mut builder = LayoutBuilder::new();
        builder.close_mode().unwrap();
        builder
            .push_leaves(&DynLayout::new(&[2, 3], &[1, 10]).unwrap())
            .unwrap();
        builder.push(4, 60).unwrap();
        builder.close_mode().unwrap();
        let grouped = builder.finish().unwrap();
        assert_eq!(grouped.shapes(), &[2, 3, 4]);
        assert!(grouped.groups().eq([0, 3]));
        assert_eq!(
            builder.finish_flat(),
            DynLayout::new(&[2, 3, 4], &[1, 10, 60])
        );

        let mut leaves = LayoutBuilder::new();
        for _ in 0..MAX_RANK {
            leaves.push(1, 0).unwrap();
        }
        assert_eq!(leaves.push(1, 0), Err(Error::CapacityExceeded));

        let mut modes = LayoutBuilder::new();
        for _ in 0..MAX_RANK {
            modes.close_mode().unwrap();
        }
        assert_eq!(modes.close_mode(), Err(Error::CapacityExceeded));
    }
}
