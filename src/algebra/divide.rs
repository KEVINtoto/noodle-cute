//! Divide for static, mixed, and runtime layouts.
//!
//! Typed methods accept integer, layout, and nested tuple tilers; an empty tuple keeps the
//! original layout and a partial tuple keeps the trailing modes. Zipped transposes by the
//! tiler profile, while tiled and flat unpack one level of modes. Typed composition failures
//! are reported directly; the generic runtime functions keep the lazy pointwise fallback.
//!
//! ```
//! use noodle_cute::{DynInt, Int, IntTuple, Nil, StaticValue};
//! use noodle_cute::algebra::TypedDivided;
//! use noodle_cute::layout::{Layout, LayoutMap};
//! type Pair<A, B> = IntTuple<A, IntTuple<B, Nil>>;
//! const C: Layout<Pair<Int<3>, Int<4>>, Pair<Int<2>, Int<6>>> =
//!     <TypedDivided<Layout<Int<12>, Int<2>>, Int<3>> as StaticValue>::VALUE;
//! let input = Layout::new(DynInt(10), Int::<2>)?;
//! let result = input.logical_divide(Int::<4>)?;
//! assert_eq!(result.size()?, 12);
//! assert_eq!(result.index_of_coord(&[3, 2])?, 22);
//! # Ok::<(), noodle_cute::Error>(())
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, StaticValue, algebra::TypedDivided, layout::Layout};
//! const C: TypedDivided<Layout<Int<8>, Int<1>>, Int<0>> = StaticValue::VALUE;
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, int_tuple, layout::Layout};
//! let input = Layout::new(Int::<8>, Int::<1>).unwrap();
//! let _ = input.logical_divide(int_tuple![Int::<2>, Int::<2>]);
//! ```
//! ```compile_fail
//! use noodle_cute::{DynInt, Int, StaticValue, algebra::TypedDivided, layout::Layout};
//! const C: TypedDivided<Layout<DynInt, Int<1>>, Int<2>> = StaticValue::VALUE;
//! ```

use crate::layout::{AffineLayoutMap, DynLayout, GroupedLayout, Layout, LayoutMap};
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
    let mut divided = composition(layout, TileRest::new(tile, rest, grouping))?;
    // CuTe unpacks tiled/flat groups after composing, so group the composed tile and rest.
    if divided.normalized.is_some()
        && let Some(affine) = divided.outer().as_affine()?
    {
        let inner = divided.inner();
        let tiles = top_modes(&composition_layout(&affine, inner.tile())?)?;
        let rests = top_modes(&composition_layout(&affine, inner.rest())?)?;
        // A layout tiler's tile_unzip is the identity, so logical matches zipped.
        let grouping = match grouping {
            Grouping::Logical => Grouping::Zipped,
            other => other,
        };
        divided.normalized = Some(super::group_modes(&tiles, &rests, grouping)?);
    }
    Ok(divided)
}

fn top_modes(layout: &GroupedLayout) -> Result<Modes> {
    let mut modes = Modes::new();
    for axis in 0..layout.rank() {
        modes.push(&layout.mode(axis)?)?;
    }
    Ok(modes)
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
/// use noodle_cute::algebra::logical_divide;
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
/// For a tuple of mode tilers use [`zipped_divide_modes`] to transpose the hierarchy.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::zipped_divide;
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
/// use noodle_cute::algebra::tiled_divide;
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
/// use noodle_cute::algebra::flat_divide;
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
/// use noodle_cute::algebra::logical_divide_modes;
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
/// - Returns [`Error::CapacityExceeded`] if leaf or group storage exceeds [`MAX_RANK`](crate::layout::MAX_RANK).
pub fn logical_divide_modes<L: AffineLayoutMap>(
    layout: &L,
    tiler: &[DynLayout],
) -> Result<GroupedLayout> {
    let (tiles, rests) = divide_modes(layout, tiler)?;
    super::group_modes(&tiles, &rests, Grouping::Logical)
}

/// CuTe `zipped_divide(layout, tiler)`: groups modes into
/// `((tile_0, tile_1, ...), (rest_0, rest_1, ..., trailing_modes...))`.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::zipped_divide_modes;
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
    super::group_modes(&tiles, &rests, Grouping::Zipped)
}

/// CuTe `tiled_divide(layout, tiler)`: groups modes into `((tile_0, tile_1, ...), rest_0, rest_1, ...)`.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::tiled_divide_modes;
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
    super::group_modes(&tiles, &rests, Grouping::Tiled)
}

/// CuTe `flat_divide(layout, tiler)`: flattens modes into `(tile_0, tile_1, ..., rest_0, rest_1, ...)`.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::flat_divide_modes;
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
    super::group_modes(&tiles, &rests, Grouping::Flat)
}

pub struct DivideOp;

pub trait CoalesceTarget {
    type Shape;
    fn cotarget(&self) -> Result<Self::Shape>;
}
impl<S, D> CoalesceTarget for Layout<S, D>
where
    S: crate::layout::Shape + crate::tuple::Flatten,
    D: crate::tuple::Item<Profile = S::Profile> + crate::tuple::Flatten,
    Layout<S::Output, D::Output>: super::coalesce::CoalesceFold,
    <Layout<S::Output, D::Output> as super::coalesce::CoalesceFold>::Output:
        super::coalesce::FinishCoalesce,
    super::CoalescedLayout<S, D>: crate::layout::LayoutParts,
{
    type Shape = <super::CoalescedLayout<S, D> as crate::layout::LayoutParts>::Shape;
    fn cotarget(&self) -> Result<Self::Shape> {
        use crate::layout::LayoutParts;
        Ok(self.coalesce()?.into_layout().shape)
    }
}
type DivideInner<L, T> =
    super::typed::Joined<T, super::TypedComplement<T, <L as CoalesceTarget>::Shape>>;
impl<L, T> super::typed::Binary<L, T> for DivideOp
where
    L: CoalesceTarget + super::TypedCompose<DivideInner<L, T>>,
    T: LayoutMap + Clone + crate::layout::LayoutParts + super::Complement<L::Shape>,
    super::TypedComplement<T, L::Shape>: crate::layout::LayoutParts,
    super::TypedComposition<L, DivideInner<L, T>>: crate::layout::LayoutParts,
{
    type Output = super::TypedComposition<L, DivideInner<L, T>>;
    fn run(layout: L, tiler: T) -> Result<Self::Output> {
        if tiler.size()? == 0 {
            return Err(Error::InvalidShape);
        }
        let rest = tiler.clone().complement_typed(layout.cotarget()?)?;
        layout.compose(super::typed::join(tiler, rest))
    }
}

super::typed::operation_api!(TypedDivide, TypedDivided, DivideOp, divide_typed;
    (LOGICAL, logical_divide),
    (ZIPPED, zipped_divide),
    (TILED, tiled_divide),
    (FLAT, flat_divide),
);
