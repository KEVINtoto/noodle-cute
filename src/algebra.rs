//! Layout algebra matching upstream CUTLASS/CuTe semantics: coalesce, composition,
//! complement, divide, product, and inverse.
//!
//! Each operation is available for typed [`Layout`](crate::layout::Layout)s, which keep static
//! leaves and nested profiles, and for runtime affine maps through free functions returning
//! [`DynLayout`](crate::layout::DynLayout) or [`GroupedLayout`] (or lazy views for non-affine maps). Typed results are
//! named by aliases such as [`TypedComposition`]; for fully static inputs
//! `<Alias as StaticValue>::VALUE` is the result as a constant (see [`crate::StaticValue`]).

use crate::int::{add, mul};
use crate::layout::{GroupedLayout, LayoutBuilder, LayoutMap, is_injective_by_enumeration};
use crate::{Error, Result};

mod coalesce;
mod complement;
mod composition;
mod divide;
mod inverse;
mod product;
mod typed;

pub use coalesce::CoalescedLayout;
pub use complement::{
    Complement, TypedComplement, complement, complement_extended, complement_shape,
};
pub use composition::{
    Composition, TypedCompose, TypedComposition, composition, composition_layout,
};
pub use divide::{
    Divided, TypedDivide, TypedDivided, flat_divide, flat_divide_modes, logical_divide,
    logical_divide_modes, tiled_divide, tiled_divide_modes, zipped_divide, zipped_divide_modes,
};
pub use inverse::{
    Inverse, TypedInverse, TypedInvert, dense_left_inverse, dense_right_inverse, left_inverse,
    right_inverse,
};
pub use product::{
    Produced, TypedProduct, TypedProductLayout, blocked_product, flat_product, flat_product_modes,
    logical_product, logical_product_modes, raked_product, tiled_product, tiled_product_modes,
    zipped_product, zipped_product_modes,
};

/// `KIND` of [`TypedDivided`] and [`TypedProductLayout`]: two groups `(tile, rest)` (CuTe `logical_*`).
pub const LOGICAL: u8 = 0;
/// `KIND` for CuTe `zipped_*`: `((tiles...), (rests...))`.
pub const ZIPPED: u8 = 1;
/// `KIND` for CuTe `tiled_*`: `((tiles...), rest0, rest1, ...)`.
pub const TILED: u8 = 2;
/// `KIND` for CuTe `flat_*`: `(tile0, tile1, ..., rest0, rest1, ...)`.
pub const FLAT: u8 = 3;
/// `KIND` for CuTe `blocked_product`: mode `m` is `(block_m, rest_m)`.
pub const BLOCKED: u8 = 4;
/// `KIND` for CuTe `raked_product`: mode `m` is `(rest_m, block_m)`.
pub const RAKED: u8 = 5;

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
        match self.as_affine() {
            Ok(Some(affine)) => return affine.index_extended(i),
            Ok(None) | Err(Error::CapacityExceeded) => {}
            Err(e) => return Err(e),
        }
        // When `rest` is a 0-leaf scalar `():()`, the last leaf of the concatenated layout belongs to `tile`.
        if matches!(self.rest.as_affine(), Ok(Some(r)) if r.shapes().is_empty()) {
            return self.tile.index_extended(i);
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
        group_tile_rest(&tile, &rest, self.grouping).map(Some)
    }

    fn is_injective(&self) -> Result<bool> {
        match self.as_affine() {
            Ok(Some(a)) => a.is_injective(), // Fast path using stride-separation proof for affine layouts.
            Ok(None) | Err(Error::CapacityExceeded) => is_injective_by_enumeration(self),
            Err(e) => Err(e),
        }
    }
}

/// Groups per-mode tile and rest results: logical pairs `(tile_m, rest_m)` in each mode,
/// zipped/tiled/flat follow CuTe's `tile_unzip` and slicing; rest modes without a tile are kept.
pub(crate) fn group_modes(
    tiles: &GroupedLayout,
    rests: &GroupedLayout,
    grouping: Grouping,
) -> Result<GroupedLayout> {
    let mut out = LayoutBuilder::new();
    match grouping {
        Grouping::Logical => {
            for m in 0..rests.rank() {
                if m < tiles.rank() {
                    out.push_leaves(&tiles.mode(m)?)?;
                }
                out.push_leaves(&rests.mode(m)?)?;
                out.close_mode()?;
            }
        }
        Grouping::Zipped => {
            for part in [tiles, rests] {
                out.push_leaves(part.flat())?;
                out.close_mode()?;
            }
        }
        Grouping::Tiled => {
            out.push_leaves(tiles.flat())?;
            out.close_mode()?;
            out.push_modes(rests)?;
        }
        Grouping::Flat => {
            out.push_modes(tiles)?;
            out.push_modes(rests)?;
        }
    }
    out.finish()
}

/// Groups a single layout tiler's `(tile, rest)`; for one layout tiler CuTe's `tile_unzip`
/// is the identity, so logical matches zipped.
pub(crate) fn group_tile_rest(
    tile: &GroupedLayout,
    rest: &GroupedLayout,
    grouping: Grouping,
) -> Result<GroupedLayout> {
    let grouping = match grouping {
        Grouping::Logical => Grouping::Zipped,
        other => other,
    };
    group_modes(tile, rest, grouping)
}
