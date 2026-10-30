//! Divide algorithms for affine layouts.

use crate::int::{add, mul, product_slice};
use crate::layout::{AffineLayoutMap, DynLayout, GroupedLayout, LayoutMap, MAX_RANK};
use crate::{Error, Result};

use super::complement::{complement, complement_shape};
use super::composition::{Composition, composition, composition_layout};
use super::{Grouping, Modes, TileRest};

/// Composed layout dividing an original layout via a tile and its complement layout.
pub type Divided<A, B> = Composition<A, TileRest<B, DynLayout>>;

// Divides a layout into a tile and its complement, returning a composed layout.
fn divide<A: LayoutMap, B: AffineLayoutMap>(
    layout: A,
    tile: B,
    grouping: Grouping,
) -> Result<Divided<A, B>> {
    if tile.size()? <= 0 {
        return Err(Error::InvalidShape);
    }

    let rest = match layout.as_affine()? {
        Some(affine) => complement_shape(tile.clone(), affine.flat().coalesce()?.shapes())?,
        None => complement(tile.clone(), layout.size()?)?,
    };
    let inner = TileRest::new(tile, rest, grouping);
    composition(layout, inner)
}

/// CuTe `logical_divide(layout, tile)`: divides a layout into two top-level groups `(tile, rest)`.
///
/// The tile must have a positive size. When `tile` does not evenly divide `layout`, the remainder is
/// padded (so the result size may exceed `layout.size()`), requiring `layout` to support an extended
/// domain (supported by all affine layouts), returning [`Error::CoordinateOutOfBounds`] otherwise.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::divide::logical_divide;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[8], &[1])?;
/// let tile = DynLayout::new(&[2], &[1])?;
/// let div = logical_divide(layout, tile)?;
/// assert_eq!(div.size()?, 8);
/// assert_eq!(div.rank(), 2);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// - Returns [`Error::InvalidShape`] if `tile.size()? <= 0`.
/// - Returns [`Error::CoordinateOutOfBounds`] if padded coordinates exceed the layout's extended domain.
/// - Returns other layout errors if complement or composition fails.
pub fn logical_divide<A: LayoutMap, B: AffineLayoutMap>(
    layout: A,
    tile: B,
) -> Result<Divided<A, B>> {
    divide(layout, tile, Grouping::Logical)
}

/// CuTe `zipped_divide(layout, tile)` for a single layout tiler.
///
/// For a single layout tiler, zipped and logical divide have identical groups.
/// For a tuple of axis tilers use [`zipped_divide_axes`] or [`zipped_divide_modes`] to transpose the hierarchy.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::divide::zipped_divide;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[8], &[1])?;
/// let tile = DynLayout::new(&[2], &[1])?;
/// let div = zipped_divide(layout, tile)?;
/// assert_eq!(div.size()?, 8);
/// assert_eq!(div.rank(), 2);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`logical_divide`].
pub fn zipped_divide<A: LayoutMap, B: AffineLayoutMap>(
    layout: A,
    tile: B,
) -> Result<Divided<A, B>> {
    divide(layout, tile, Grouping::Zipped)
}

/// CuTe `tiled_divide(layout, tile)`: divides `layout` using `tile`, exposing a single tile group
/// followed by each unpacked rest axis.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::divide::tiled_divide;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[8], &[1])?;
/// let tile = DynLayout::new(&[2], &[1])?;
/// let div = tiled_divide(layout, tile)?;
/// assert_eq!(div.size()?, 8);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`logical_divide`].
pub fn tiled_divide<A: LayoutMap, B: AffineLayoutMap>(layout: A, tile: B) -> Result<Divided<A, B>> {
    divide(layout, tile, Grouping::Tiled)
}

/// CuTe `flat_divide(layout, tile)`: divides `layout` using `tile`, flattening all tile axes followed
/// by all rest axes.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::divide::flat_divide;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[8], &[1])?;
/// let tile = DynLayout::new(&[2], &[1])?;
/// let div = flat_divide(layout, tile)?;
/// assert_eq!(div.size()?, 8);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`logical_divide`].
pub fn flat_divide<A: LayoutMap, B: AffineLayoutMap>(layout: A, tile: B) -> Result<Divided<A, B>> {
    divide(layout, tile, Grouping::Flat)
}

/// Per-mode tile/rest decomposition layout.
///
/// Each original axis satisfies `extent = tile * rest`. Reorders logical coordinates while
/// preserving reachable addresses and injectivity.
/// - `Logical` retains one `(tile, rest)` group for each original axis.
/// - `Zipped` transposes these groups into `(tiles, rests)`.
/// - `Tiled` groups all tiles into mode 0 and exposes rest axes individually.
/// - `Flat` exposes all tile axes followed by all rest axes.
#[derive(Clone, Debug)]
pub struct DividedAxes<L> {
    pub(crate) layout: L,
    pub(crate) tile: [isize; MAX_RANK],
    pub(crate) rest: [isize; MAX_RANK],
    pub(crate) grouping: Grouping,
}

impl<L: LayoutMap> DividedAxes<L> {
    /// Returns a reference to the underlying layout.
    pub fn layout(&self) -> &L {
        &self.layout
    }

    /// Returns the tile shape slice across all axes.
    pub fn tile_shape(&self) -> &[isize] {
        &self.tile[..self.layout.rank()]
    }

    /// Returns the rest shape slice across all axes.
    pub fn rest_shape(&self) -> &[isize] {
        &self.rest[..self.layout.rank()]
    }

    /// Returns the coordinate grouping strategy.
    pub fn grouping(&self) -> Grouping {
        self.grouping
    }
}

fn divide_axes<L: LayoutMap>(
    layout: L,
    tile: &[isize],
    grouping: Grouping,
) -> Result<DividedAxes<L>> {
    if tile.len() != layout.rank() {
        return Err(Error::StructureMismatch);
    }
    if tile.len() > MAX_RANK {
        return Err(Error::CapacityExceeded);
    }
    let mut out = DividedAxes {
        layout,
        tile: [0; MAX_RANK],
        rest: [0; MAX_RANK],
        grouping,
    };
    for (a, &t) in tile.iter().enumerate() {
        if t <= 0 {
            return Err(Error::InvalidShape);
        }
        let s = out.layout.extent(a)?;
        if s % t != 0 {
            return Err(Error::NotDivisible);
        }
        out.tile[a] = t;
        out.rest[a] = s / t;
    }
    Ok(out)
}

impl<L: LayoutMap> LayoutMap for DividedAxes<L> {
    fn size(&self) -> Result<isize> {
        self.layout.size()
    }

    fn rank(&self) -> usize {
        let r = self.layout.rank();
        match self.grouping {
            Grouping::Logical => r,
            Grouping::Zipped => 2,
            Grouping::Tiled => 1 + r,
            Grouping::Flat => 2 * r,
        }
    }

    fn extent(&self, a: usize) -> Result<isize> {
        let r = self.layout.rank();
        match self.grouping {
            Grouping::Logical => self.layout.extent(a),
            Grouping::Zipped => match a {
                0 => product_slice(&self.tile[..r]),
                1 => product_slice(&self.rest[..r]),
                _ => Err(Error::AxisOutOfBounds),
            },
            Grouping::Tiled => {
                if a == 0 {
                    product_slice(&self.tile[..r])
                } else if a <= r {
                    Ok(self.rest[a - 1])
                } else {
                    Err(Error::AxisOutOfBounds)
                }
            }
            Grouping::Flat => {
                if a < r {
                    Ok(self.tile[a])
                } else if a < 2 * r {
                    Ok(self.rest[a - r])
                } else {
                    Err(Error::AxisOutOfBounds)
                }
            }
        }
    }

    fn index(&self, mut i: isize) -> Result<isize> {
        if i < 0 || i >= self.size()? {
            return Err(Error::CoordinateOutOfBounds);
        }
        if self.grouping == Grouping::Logical {
            return self.layout.index(i);
        }
        let r = self.layout.rank();
        let tile_size = product_slice(&self.tile[..r])?;
        let mut tile = i % tile_size;
        i /= tile_size;
        let mut coord = [0; MAX_RANK];
        // Decode all tile and rest coordinates separately first, then reconstruct each original axis coordinate.
        for (a, c) in coord[..r].iter_mut().enumerate() {
            let t = tile % self.tile[a];
            tile /= self.tile[a];
            let rest = i % self.rest[a];
            i /= self.rest[a];
            *c = add(t, mul(rest, self.tile[a])?)?;
        }
        self.layout.index_of_coord(&coord[..r])
    }

    fn address_range(&self) -> Result<Option<(isize, isize)>> {
        self.layout.address_range()
    }

    fn is_injective(&self) -> Result<bool> {
        self.layout.is_injective()
    }

    fn as_affine(&self) -> Result<Option<GroupedLayout>> {
        Ok(None)
    }
}

/// Divides each top-level axis individually with a per-axis tile shape.
///
/// The `tile` slice length must equal the layout rank. Each tile extent must be positive and evenly divide the axis extent.
/// Each axis remains a top-level mode interpreted internally as `(tile, rest)`; linear index order is preserved.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::divide::logical_divide_axes;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[4, 6], &[1, 4])?;
/// let div = logical_divide_axes(layout, &[2, 3])?;
/// assert_eq!(div.size()?, 24);
/// assert_eq!(div.rank(), 2);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// - Returns [`Error::StructureMismatch`] if `tile.len() != layout.rank()`.
/// - Returns [`Error::CapacityExceeded`] if `tile.len() > MAX_RANK`.
/// - Returns [`Error::InvalidShape`] if any tile extent is `<= 0`.
/// - Returns [`Error::NotDivisible`] if an axis extent is not evenly divisible by its tile extent.
pub fn logical_divide_axes<L: LayoutMap>(layout: L, tile: &[isize]) -> Result<DividedAxes<L>> {
    divide_axes(layout, tile, Grouping::Logical)
}

/// Divides axes identically to [`logical_divide_axes`], grouping all tile coordinates before all rest coordinates.
///
/// Produces two top-level axes: product of tiles and product of rests.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::divide::zipped_divide_axes;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[4, 6], &[1, 4])?;
/// let div = zipped_divide_axes(layout, &[2, 3])?;
/// assert_eq!(div.size()?, 24);
/// assert_eq!(div.rank(), 2);
/// assert_eq!(div.extent(0)?, 6);
/// assert_eq!(div.extent(1)?, 4);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`logical_divide_axes`].
pub fn zipped_divide_axes<L: LayoutMap>(layout: L, tile: &[isize]) -> Result<DividedAxes<L>> {
    divide_axes(layout, tile, Grouping::Zipped)
}

/// Divides axes identically to [`logical_divide_axes`], grouping all tiles into the first axis, followed by each individual rest axis.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::divide::tiled_divide_axes;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[4, 6], &[1, 4])?;
/// let div = tiled_divide_axes(layout, &[2, 3])?;
/// assert_eq!(div.size()?, 24);
/// assert_eq!(div.rank(), 3);
/// assert_eq!(div.extent(0)?, 6);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`logical_divide_axes`].
pub fn tiled_divide_axes<L: LayoutMap>(layout: L, tile: &[isize]) -> Result<DividedAxes<L>> {
    divide_axes(layout, tile, Grouping::Tiled)
}

/// Divides axes identically to [`logical_divide_axes`], flattening all axes in the order `[tile0, tile1, ..., rest0, rest1, ...]`.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::divide::flat_divide_axes;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[4, 6], &[1, 4])?;
/// let div = flat_divide_axes(layout, &[2, 3])?;
/// assert_eq!(div.size()?, 24);
/// assert_eq!(div.rank(), 4);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`logical_divide_axes`].
pub fn flat_divide_axes<L: LayoutMap>(layout: L, tile: &[isize]) -> Result<DividedAxes<L>> {
    divide_axes(layout, tile, Grouping::Flat)
}

// CuTe `logical_divide` for tuple tilers: mode m is divided by tiler[m],
// returning tile modes and rest modes; rest modes are followed by uncovered trailing modes.
pub(crate) fn divide_modes<L: AffineLayoutMap>(
    layout: &L,
    tiler: &[DynLayout],
) -> Result<(Modes, Modes)> {
    let layout = layout.as_grouped_layout()?;
    if tiler.len() > layout.rank() {
        return Err(Error::StructureMismatch);
    }
    let (mut tiles, mut rests) = (Modes::new(), Modes::new());
    for m in 0..layout.rank() {
        let mode = layout.mode(m)?;
        let Some(t) = tiler.get(m) else {
            rests.push(&mode)?;
            continue;
        };
        if t.size()? <= 0 {
            return Err(Error::InvalidShape);
        }
        // Same as single-tiler logical_divide: rest is the complement of the tiler relative to the coalesced mode shape.
        let rest = complement_shape(*t, mode.coalesce()?.shapes())?;
        tiles.push(composition_layout(&mode, t)?.flat())?;
        rests.push(composition_layout(&mode, &rest)?.flat())?;
    }
    Ok((tiles, rests))
}

/// CuTe `logical_divide(layout, tiler)` with a per-mode tuple tiler (an integer `n` corresponds to `n:1`).
///
/// The `m`-th top-level mode becomes `(tile_m, rest_m)`. Any trailing modes not covered by `tiler`
/// remain unchanged. If a mode is not evenly divisible, it is padded consistently with the single-tiler
/// version, extending into the layout's extended domain.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::divide::logical_divide_modes;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[4, 6], &[1, 4])?;
/// let tiler = [DynLayout::new(&[2], &[1])?, DynLayout::new(&[3], &[1])?];
/// let div = logical_divide_modes(&layout, &tiler)?;
/// assert_eq!(div.rank(), 2);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// - Returns [`Error::StructureMismatch`] if `tiler.len() > layout.rank()`.
/// - Returns [`Error::InvalidShape`] if any tiler mode has size `<= 0`.
/// - Returns [`Error::NotDivisible`] if composition divisibility conditions fail.
/// - Returns [`Error::CapacityExceeded`] if leaf or group storage exceeds [`MAX_RANK`].
pub fn logical_divide_modes<L: AffineLayoutMap>(
    layout: &L,
    tiler: &[DynLayout],
) -> Result<GroupedLayout> {
    let (tiles, rests) = divide_modes(layout, tiler)?;
    let mut out = Modes::new();
    for m in 0..rests.rank {
        if m < tiles.rank {
            // Leaves of tile_m and rest_m are combined into a single top-level mode.
            let mut both = Modes::new();
            both.push(&tiles.mode(m)?)?;
            both.push(&rests.mode(m)?)?;
            out.push(&both.merged(0..2)?)?;
        } else {
            out.push(&rests.mode(m)?)?;
        }
    }
    out.layout()
}

/// CuTe `zipped_divide(layout, tiler)`: groups modes into
/// `((tile_0, tile_1, ...), (rest_0, rest_1, ..., trailing_modes...))`.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::divide::zipped_divide_modes;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[4, 6], &[1, 4])?;
/// let tiler = [DynLayout::new(&[2], &[1])?, DynLayout::new(&[3], &[1])?];
/// let div = zipped_divide_modes(&layout, &tiler)?;
/// assert_eq!(div.rank(), 2);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`logical_divide_modes`].
pub fn zipped_divide_modes<L: AffineLayoutMap>(
    layout: &L,
    tiler: &[DynLayout],
) -> Result<GroupedLayout> {
    let (tiles, rests) = divide_modes(layout, tiler)?;
    let mut out = Modes::new();
    out.push(&tiles.merged(0..tiles.rank)?)?;
    out.push(&rests.merged(0..rests.rank)?)?;
    out.layout()
}

/// CuTe `tiled_divide(layout, tiler)`: groups modes into `((tile_0, tile_1, ...), rest_0, rest_1, ...)`.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::divide::tiled_divide_modes;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[4, 6], &[1, 4])?;
/// let tiler = [DynLayout::new(&[2], &[1])?, DynLayout::new(&[3], &[1])?];
/// let div = tiled_divide_modes(&layout, &tiler)?;
/// assert_eq!(div.rank(), 3);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`logical_divide_modes`].
pub fn tiled_divide_modes<L: AffineLayoutMap>(
    layout: &L,
    tiler: &[DynLayout],
) -> Result<GroupedLayout> {
    let (tiles, rests) = divide_modes(layout, tiler)?;
    let mut out = Modes::new();
    out.push(&tiles.merged(0..tiles.rank)?)?;
    for m in 0..rests.rank {
        out.push(&rests.mode(m)?)?;
    }
    out.layout()
}

/// CuTe `flat_divide(layout, tiler)`: flattens modes into `(tile_0, tile_1, ..., rest_0, rest_1, ...)`.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::divide::flat_divide_modes;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[4, 6], &[1, 4])?;
/// let tiler = [DynLayout::new(&[2], &[1])?, DynLayout::new(&[3], &[1])?];
/// let div = flat_divide_modes(&layout, &tiler)?;
/// assert_eq!(div.rank(), 4);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`logical_divide_modes`].
pub fn flat_divide_modes<L: AffineLayoutMap>(
    layout: &L,
    tiler: &[DynLayout],
) -> Result<GroupedLayout> {
    let (tiles, rests) = divide_modes(layout, tiler)?;
    let mut out = Modes::new();
    for m in 0..tiles.rank {
        out.push(&tiles.mode(m)?)?;
    }
    for m in 0..rests.rank {
        out.push(&rests.mode(m)?)?;
    }
    out.layout()
}
