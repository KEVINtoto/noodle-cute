//! Algebraic algorithms for affine layouts.
//!
//! Provides fundamental layout algebra operations matching upstream CUTLASS/CuTe semantics:
//! - Layout composition ([`composition`])
//! - Layout complement ([`complement`])
//! - Layout division ([`divide`])
//! - Layout inversion ([`inverse`])
//! - Layout product ([`product`])

use crate::int::{add, mul};
use crate::layout::{DynLayout, GroupedLayout, LayoutMap, MAX_RANK, is_injective_by_enumeration};
use crate::{Error, Result, Slice};

pub mod complement;
pub mod composition;
pub mod divide;
pub mod inverse;
pub mod product;

/// Fixed-capacity leaf buffer for assembling flat layout results in CuTe algebra.
#[derive(Clone, Copy)]
struct Leaves {
    shape: [isize; MAX_RANK],
    stride: [isize; MAX_RANK],
    len: usize,
}

impl Leaves {
    const fn new() -> Self {
        Self {
            shape: [0; MAX_RANK],
            stride: [0; MAX_RANK],
            len: 0,
        }
    }

    /// Pushes a new leaf shape and stride into the buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CapacityExceeded`] if `self.len == MAX_RANK`.
    fn push(&mut self, s: isize, d: isize) -> Result<()> {
        if self.len == MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        self.shape[self.len] = s;
        self.stride[self.len] = d;
        self.len += 1;
        Ok(())
    }

    /// Constructs a [`DynLayout`] from the accumulated leaves.
    ///
    /// # Errors
    ///
    /// Returns an error if the accumulated shapes and strides form an invalid layout.
    fn layout(&self) -> Result<DynLayout> {
        DynLayout::new(&self.shape[..self.len], &self.stride[..self.len])
    }
}

/// Leaf buffer with top-level grouping for assembling per-mode results of tuple-tiler partitioning and division.
#[derive(Clone, Copy)]
pub(crate) struct Modes {
    leaves: Leaves,
    groups: [usize; MAX_RANK],
    pub(crate) rank: usize,
}

impl Modes {
    pub(crate) const fn new() -> Self {
        Self {
            leaves: Leaves::new(),
            groups: [0; MAX_RANK],
            rank: 0,
        }
    }

    /// Appends a mode layout into the buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CapacityExceeded`] if appending would exceed [`MAX_RANK`] modes or leaves.
    pub(crate) fn push(&mut self, mode: &DynLayout) -> Result<()> {
        if self.rank == MAX_RANK || self.leaves.len + mode.rank() > MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        for (&s, &d) in mode.shapes().iter().zip(mode.strides()) {
            self.leaves.push(s, d)?;
        }
        self.groups[self.rank] = mode.rank();
        self.rank += 1;
        Ok(())
    }

    /// Retrieves the `m`-th mode as a [`DynLayout`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::AxisOutOfBounds`] if `m >= self.rank`.
    pub(crate) fn mode(&self, m: usize) -> Result<DynLayout> {
        if m >= self.rank {
            return Err(Error::AxisOutOfBounds);
        }
        let start = self.groups[..m].iter().sum::<usize>();
        let end = start + self.groups[m];
        DynLayout::new(
            &self.leaves.shape[start..end],
            &self.leaves.stride[start..end],
        )
    }

    /// Merges modes within `range` into a single flat mode (e.g. for zipped/tiled tile or rest groups).
    ///
    /// # Errors
    ///
    /// Returns [`Error::AxisOutOfBounds`] if `range.start > range.end` or `range.end > self.rank`.
    pub(crate) fn merged(&self, range: core::ops::Range<usize>) -> Result<DynLayout> {
        if range.start > range.end || range.end > self.rank {
            return Err(Error::AxisOutOfBounds);
        }
        let start = self.groups[..range.start].iter().sum::<usize>();
        let end = start + self.groups[range].iter().sum::<usize>();
        DynLayout::new(
            &self.leaves.shape[start..end],
            &self.leaves.stride[start..end],
        )
    }

    /// Builds a [`GroupedLayout`] from the accumulated modes.
    ///
    /// # Errors
    ///
    /// Returns an error if constructing the layout or grouped layout fails.
    pub(crate) fn layout(&self) -> Result<GroupedLayout> {
        GroupedLayout::new(self.leaves.layout()?, &self.groups[..self.rank])
    }
}

/// Slices `modes` by `coord`: modes selected by [`Slice::Index`] contribute to the address offset
/// and are removed, while [`Slice::All`] and omitted trailing modes are retained.
///
/// If an error occurs while appending modes to `out`, `out` may be partially updated without rollback.
/// Callers should discard `out` if an error is returned.
///
/// # Errors
///
/// - Returns [`Error::StructureMismatch`] if `coord.len() > modes.rank`.
/// - Returns [`Error::CoordinateOutOfBounds`] if a coordinate index is negative or out of bounds for its mode.
/// - Returns [`Error::Overflow`] if accumulating the offset overflows.
/// - Returns [`Error::CapacityExceeded`] if appending to `out` exceeds capacity limits.
#[allow(dead_code)]
pub(crate) fn slice_modes(modes: &Modes, coord: &[Slice], out: &mut Modes) -> Result<isize> {
    if coord.len() > modes.rank {
        return Err(Error::StructureMismatch);
    }
    let mut offset = 0;
    for m in 0..modes.rank {
        let mode = modes.mode(m)?;
        match coord.get(m) {
            Some(Slice::Index(c)) => {
                if *c < 0 || *c >= mode.size()? {
                    return Err(Error::CoordinateOutOfBounds);
                }
                offset = add(offset, mode.index(*c)?)?;
            }
            Some(Slice::All) | None => out.push(&mode)?,
        }
    }
    Ok(offset)
}

/// Coordinate hierarchy selected by the divide/product variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grouping {
    /// Retains two top-level groups: `(tile, rest)`. For per-axis division, retains the nested
    /// `(tile, rest)` hierarchy within each original axis.
    Logical,
    /// Groups the tiles of all axes into one mode and the rests into another: `((tiles...), (rests...))`.
    Zipped,
    /// Groups all tiles into a single top-level mode, leaving rest axes exposed individually: `((tiles...), rest0, rest1, ...)`.
    Tiled,
    /// Flattens all axes with tiles first followed by rests: `(tile0, tile1, ..., rest0, rest1, ...)`.
    Flat,
}

/// Composite layout combining a tile layout and a rest layout according to a [`Grouping`].
///
/// Under [`Grouping::Logical`] and [`Grouping::Zipped`], the layout retains two groups.
/// Under [`Grouping::Tiled`], rest axes are unpacked individually.
/// Under [`Grouping::Flat`], all axes of both tile and rest are flattened.
/// Mapped addresses are the sum of tile and rest addresses; tile coordinates vary fastest in linear order.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::{Grouping, TileRest};
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let tile = DynLayout::new(&[2], &[1])?;
/// let rest = DynLayout::new(&[3], &[2])?;
/// let tr = TileRest::new(tile, rest, Grouping::Logical);
/// assert_eq!(tr.size()?, 6);
/// assert_eq!(tr.rank(), 2);
/// assert_eq!(tr.index(5)?, 5);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct TileRest<T, R> {
    tile: T,
    rest: R,
    grouping: Grouping,
}

impl<T, R> TileRest<T, R> {
    /// Creates a new `TileRest` composite layout with the given grouping.
    pub fn new(tile: T, rest: R, grouping: Grouping) -> Self {
        Self {
            tile,
            rest,
            grouping,
        }
    }

    /// Returns a reference to the tile layout.
    pub fn tile(&self) -> &T {
        &self.tile
    }

    /// Returns a reference to the rest layout.
    pub fn rest(&self) -> &R {
        &self.rest
    }

    /// Returns the coordinate grouping strategy.
    pub fn grouping(&self) -> Grouping {
        self.grouping
    }
}

impl<T: LayoutMap, R: LayoutMap> LayoutMap for TileRest<T, R> {
    fn size(&self) -> Result<isize> {
        mul(self.tile.size()?, self.rest.size()?)
    }

    fn rank(&self) -> usize {
        match self.grouping {
            Grouping::Logical | Grouping::Zipped => 2,
            Grouping::Tiled => 1 + self.rest.rank(),
            Grouping::Flat => self.tile.rank() + self.rest.rank(),
        }
    }

    fn extent(&self, a: usize) -> Result<isize> {
        match self.grouping {
            Grouping::Logical | Grouping::Zipped => match a {
                0 => self.tile.size(),
                1 => self.rest.size(),
                _ => Err(Error::AxisOutOfBounds),
            },
            Grouping::Tiled => {
                if a == 0 {
                    self.tile.size()
                } else {
                    self.rest.extent(a - 1)
                }
            }
            Grouping::Flat => {
                if a < self.tile.rank() {
                    self.tile.extent(a)
                } else {
                    self.rest.extent(a - self.tile.rank())
                }
            }
        }
    }

    fn index(&self, i: isize) -> Result<isize> {
        if i < 0 || i >= self.size()? {
            return Err(Error::CoordinateOutOfBounds);
        }
        let tile_size = self.tile.size()?;
        // A valid linear index implies non-zero total size, so tile_size is guaranteed non-zero.
        add(
            self.tile.index(i % tile_size)?,
            self.rest.index(i / tile_size)?,
        )
    }

    fn index_extended(&self, i: isize) -> Result<isize> {
        if i >= 0 && i < self.size()? {
            return self.index(i);
        }
        if let Some(affine) = self.as_affine()? {
            return affine.index_extended(i);
        }
        // Extended domain only extends the slowest rest group; the tile group continues to wrap modulo.
        let tile_size = self.tile.size()?;
        if i < 0 || tile_size <= 0 {
            return Err(Error::CoordinateOutOfBounds);
        }
        add(
            self.tile.index(i % tile_size)?,
            self.rest.index_extended(i / tile_size)?,
        )
    }

    fn address_range(&self) -> Result<Option<(isize, isize)>> {
        // Domain is the Cartesian product tile × rest; address is their sum, so endpoints add coordinate-wise.
        match (self.tile.address_range()?, self.rest.address_range()?) {
            (Some((a0, a1)), Some((b0, b1))) => Ok(Some((add(a0, b0)?, add(a1, b1)?))),
            _ => Ok(None),
        }
    }

    fn as_affine(&self) -> Result<Option<GroupedLayout>> {
        let (Some(tile), Some(rest)) = (self.tile.as_affine()?, self.rest.as_affine()?) else {
            return Ok(None);
        };
        let mut leaves = Leaves::new();
        for layout in [&tile, &rest] {
            for (&s, &d) in layout.shapes().iter().zip(layout.strides()) {
                leaves.push(s, d)?;
            }
        }
        let mut groups = [0usize; MAX_RANK];
        let mut rank = 0;
        let mut put = |n: usize| -> Result<()> {
            *groups.get_mut(rank).ok_or(Error::CapacityExceeded)? = n;
            rank += 1;
            Ok(())
        };
        match self.grouping {
            Grouping::Logical | Grouping::Zipped => {
                put(tile.shapes().len())?;
                put(rest.shapes().len())?;
            }
            Grouping::Tiled => {
                put(tile.shapes().len())?;
                rest.groups().try_for_each(&mut put)?;
            }
            Grouping::Flat => {
                tile.groups().try_for_each(&mut put)?;
                rest.groups().try_for_each(&mut put)?;
            }
        }
        GroupedLayout::new(leaves.layout()?, &groups[..rank]).map(Some)
    }

    fn is_injective(&self) -> Result<bool> {
        match self.as_affine() {
            Ok(Some(a)) => a.is_injective(), // Fast path using stride-separation proof for affine layouts.
            Ok(None) | Err(Error::CapacityExceeded) => is_injective_by_enumeration(self),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaves_buffer_builds_flat_layouts_and_enforces_max_rank() {
        let mut leaves = Leaves::new();
        let empty = leaves.layout().unwrap();
        assert_eq!(empty.rank(), 0);
        assert_eq!(empty.size(), Ok(1));

        for i in 0..MAX_RANK {
            assert_eq!(leaves.push(2, (i == 0) as isize), Ok(()));
        }
        assert_eq!(leaves.push(2, 0), Err(Error::CapacityExceeded));
        let full = leaves.layout().unwrap();
        assert_eq!(full.rank(), MAX_RANK);
        assert_eq!(full.shapes(), &[2; MAX_RANK]);
    }

    #[test]
    fn modes_buffer_preserves_mode_spans_and_merges_ranges() {
        let mut modes = Modes::new();
        let scalar = DynLayout::new(&[], &[]).unwrap();
        let nested = DynLayout::new(&[2, 3], &[1, 10]).unwrap();
        let tail = DynLayout::new(&[4], &[60]).unwrap();
        modes.push(&scalar).unwrap();
        modes.push(&nested).unwrap();
        modes.push(&tail).unwrap();
        assert_eq!(modes.rank, 3);
        assert_eq!(modes.mode(0), Ok(scalar));
        assert_eq!(modes.mode(1), Ok(nested));
        assert_eq!(modes.mode(2), Ok(tail));
        assert_eq!(modes.mode(3), Err(Error::AxisOutOfBounds));
        assert_eq!(modes.merged(0..0), Ok(scalar));
        assert_eq!(modes.merged(1..1), Ok(scalar));
        assert_eq!(modes.merged(3..3), Ok(scalar));
        assert_eq!(modes.merged(1..3), DynLayout::new(&[2, 3, 4], &[1, 10, 60]));
        let reversed = core::ops::Range { start: 2, end: 1 };
        assert_eq!(modes.merged(reversed), Err(Error::AxisOutOfBounds));
        assert_eq!(modes.merged(0..4), Err(Error::AxisOutOfBounds));

        let grouped = modes.layout().unwrap();
        assert_eq!(grouped.rank(), 3);
        assert_eq!(grouped.extent(0), Ok(1));
        assert_eq!(grouped.extent(1), Ok(6));
        assert_eq!(grouped.extent(2), Ok(4));

        // Leaf capacity overflow must not partially mutate `Modes`.
        let mut partial = Modes::new();
        let leaves_30 = DynLayout::new(&[1; 30], &[0; 30]).unwrap();
        partial.push(&leaves_30).unwrap();
        assert_eq!(
            partial.push(&DynLayout::new(&[1; 3], &[0; 3]).unwrap()),
            Err(Error::CapacityExceeded)
        );
        assert_eq!(partial.rank, 1);
        assert_eq!(partial.leaves.len, 30);

        // Zero-leaf scalar modes exercise the mode-count guard independently of leaf capacity.
        let mut max_modes = Modes::new();
        for _ in 0..MAX_RANK {
            assert_eq!(max_modes.push(&scalar), Ok(()));
        }
        assert_eq!(max_modes.push(&scalar), Err(Error::CapacityExceeded));
    }

    #[test]
    fn slice_modes_accumulates_offsets_retains_unindexed_modes_and_checks_bounds() {
        let mut modes = Modes::new();
        modes
            .push(&DynLayout::new(&[2, 2], &[1, 6]).unwrap())
            .unwrap();
        modes.push(&DynLayout::new(&[3], &[2]).unwrap()).unwrap();
        modes.push(&DynLayout::new(&[2], &[24]).unwrap()).unwrap();
        let full = modes.layout().unwrap();

        // Mode 0 is fixed at linear coordinate 3 (1 + 6 = 7), mode 1 uses Slice::All,
        // and mode 2 is retained via an omitted trailing selector.
        let mut sliced = Modes::new();
        let offset = slice_modes(&modes, &[Slice::Index(3), Slice::All], &mut sliced).unwrap();
        assert_eq!(offset, 7);
        let sub = sliced.layout().unwrap();
        assert_eq!(sub.rank(), 2);
        assert_eq!(sub.shapes(), &[3, 2]);
        assert_eq!(sub.strides(), &[2, 24]);
        for c2 in 0..2 {
            for c1 in 0..3 {
                assert_eq!(
                    offset + sub.index_of_coord(&[c1, c2]).unwrap(),
                    full.index_of_coord(&[3, c1, c2]).unwrap()
                );
            }
        }

        // Fixing every mode yields a rank-0 scalar and the exact point address.
        let mut point = Modes::new();
        let point_offset = slice_modes(
            &modes,
            &[Slice::Index(2), Slice::Index(1), Slice::Index(1)],
            &mut point,
        )
        .unwrap();
        assert_eq!(point_offset, full.index_of_coord(&[2, 1, 1]).unwrap());
        assert_eq!(point.rank, 0);
        assert_eq!(point.layout().unwrap().size(), Ok(1));

        // Validate selector length, coordinate bounds (including empty modes), and overflow.
        let mut err_out = Modes::new();
        assert_eq!(
            slice_modes(&modes, &[Slice::All; 4], &mut err_out),
            Err(Error::StructureMismatch)
        );
        assert_eq!(
            slice_modes(&modes, &[Slice::Index(-1)], &mut err_out),
            Err(Error::CoordinateOutOfBounds)
        );
        assert_eq!(
            slice_modes(&modes, &[Slice::Index(4)], &mut err_out),
            Err(Error::CoordinateOutOfBounds)
        );

        let mut with_empty = Modes::new();
        with_empty
            .push(&DynLayout::new(&[0], &[1]).unwrap())
            .unwrap();
        assert_eq!(
            slice_modes(&with_empty, &[Slice::Index(0)], &mut err_out),
            Err(Error::CoordinateOutOfBounds)
        );

        let mut overflowing = Modes::new();
        overflowing
            .push(&DynLayout::new(&[2], &[isize::MAX]).unwrap())
            .unwrap();
        overflowing
            .push(&DynLayout::new(&[2], &[1]).unwrap())
            .unwrap();
        assert_eq!(
            slice_modes(
                &overflowing,
                &[Slice::Index(1), Slice::Index(1)],
                &mut err_out
            ),
            Err(Error::Overflow)
        );
    }
}
