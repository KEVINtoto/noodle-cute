//! 全静态、混合类型与运行时布局的 divide。
//!
//! 类型化方法支持整数、布局和嵌套 tuple tiler；空 tuple 保留原布局，部分 tuple
//! 保留尾部模式。zipped 按 tiler profile 转置，tiled/flat 只解开一层模式。
//! `*_divide_axes` 是本库的严格整除接口，要求每个顶层模式恰好对应一个整数 tile。
//! 类型化组合失败直接报错；通用自由函数保留原有惰性组合回退。
//!
//! ```
//! use noodle_cute::{Int, IntTuple, Nil, DynInt, layout::{Layout, LayoutMap}};
//! use noodle_cute::algebra::static_logical_divide;
//! type Pair<A, B> = IntTuple<A, IntTuple<B, Nil>>;
//! const C: Layout<Pair<Int<3>, Int<4>>, Pair<Int<2>, Int<6>>> = static_logical_divide(
//!     Layout { shape: Int::<12>, stride: Int::<2> }, Int::<3>,
//! );
//! let input = Layout::new(DynInt(10), Int::<2>)?;
//! let result = input.logical_divide(Int::<4>)?;
//! assert_eq!(result.size()?, 12);
//! assert_eq!(result.index_of_coord(&[3, 2])?, 22);
//! # Ok::<(), noodle_cute::Error>(())
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, layout::Layout, algebra::static_logical_divide};
//! let _ = static_logical_divide(Layout { shape: Int::<8>, stride: Int::<1> }, Int::<0>);
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, int_tuple, layout::Layout};
//! let input = Layout::new(Int::<8>, Int::<1>).unwrap();
//! let _ = input.logical_divide(int_tuple![Int::<2>, Int::<2>]);
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, layout::Layout, algebra::static_logical_divide_axes};
//! let _ = static_logical_divide_axes(Layout { shape: Int::<10>, stride: Int::<1> }, Int::<3>);
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, shape, stride, layout::Layout};
//! let input = Layout::new(shape![Int::<8>, Int::<6>], stride![Int::<1>, Int::<8>]).unwrap();
//! let _ = input.logical_divide_axes(shape![Int::<2>]);
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, DynInt, layout::Layout, algebra::static_logical_divide};
//! let _ = static_logical_divide(Layout { shape: DynInt(8), stride: Int::<1> }, Int::<2>);
//! ```

use core::ops::Div;

use crate::int::{Integer, add, mul, product_slice};
use crate::layout::{AffineLayoutMap, DynLayout, GroupedLayout, Layout, LayoutMap, MAX_RANK};
use crate::tuple::{Item, Product as TupleProduct};
use crate::{Assert, Error, Int, IntTuple, Nil, Result, True};

use super::complement::{complement, complement_shape};
use super::composition::{Composition, composition, composition_layout};
use super::typed::{IntoModes, ValidateTiler, Wrap};
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
    super::group_modes(&tiles, &rests, Grouping::Logical)
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
    super::group_modes(&tiles, &rests, Grouping::Zipped)
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
    super::group_modes(&tiles, &rests, Grouping::Tiled)
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
    super::group_modes(&tiles, &rests, Grouping::Flat)
}

#[doc(hidden)]
pub struct DivideOp;

#[doc(hidden)]
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

super::typed::operation_api!(TypedDivide, TypedDivided, StaticDivide, DivideOp, divide_typed, ();
    (0, logical_divide, static_logical_divide),
    (1, zipped_divide, static_zipped_divide),
    (2, tiled_divide, static_tiled_divide),
    (3, flat_divide, static_flat_divide),
);

macro_rules! dynamic_divide_methods {
    ($($name:ident),*) => {$ (
        impl DynLayout {
            /// 运行时分块入口，沿用自由函数的补布局、扩展域与错误规则。
            pub fn $name<T: AffineLayoutMap>(&self, tiler: T) -> Result<Divided<Self, T>> {
                $name(*self, tiler)
            }
        }
    )*};
}
dynamic_divide_methods!(logical_divide, zipped_divide, tiled_divide, flat_divide);

#[doc(hidden)]
pub struct AxesOp;
#[doc(hidden)]
pub const AXIS_VALID<S: TupleProduct, T: Integer>: bool = match (S::Output::STATIC, T::STATIC) {
    (_, Some(t)) if t <= 0 => false,
    (Some(s), Some(t)) => s % t == 0,
    _ => true,
} where S::Output: Integer;

#[doc(hidden)]
pub trait StrictAxes<T> {
    fn strict_axes(&self, tiler: &T) -> Result<()>;
}
impl StrictAxes<Nil> for Layout<Nil, Nil> {
    fn strict_axes(&self, _: &Nil) -> Result<()> {
        Ok(())
    }
}
impl<S, ST, D, DT, T: Integer, TT> StrictAxes<IntTuple<T, TT>>
    for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    S: crate::layout::Shape + TupleProduct,
    D: Item<Profile = S::Profile>,
    <S as TupleProduct>::Output: Integer,
    Assert<{ AXIS_VALID::<S, T> }>: True,
    ST: Clone,
    DT: Clone,
    Layout<ST, DT>: StrictAxes<TT>,
{
    fn strict_axes(&self, tiler: &IntTuple<T, TT>) -> Result<()> {
        let t = tiler.0.value();
        if t <= 0 {
            return Err(Error::InvalidShape);
        }
        let s = Layout {
            shape: self.shape.0.clone(),
            stride: self.stride.0.clone(),
        }
        .size()?;
        if s % t != 0 {
            return Err(Error::NotDivisible);
        }
        Layout {
            shape: self.shape.1.clone(),
            stride: self.stride.1.clone(),
        }
        .strict_axes(&tiler.1)
    }
}
#[doc(hidden)]
pub trait AxesCheck<T> {
    fn axes_check(&self, tiler: &T) -> Result<()>;
}
impl<S: Wrap + Clone, D: Wrap + Clone, T: Wrap + Clone> AxesCheck<T> for Layout<S, D>
where
    Layout<S::Output, D::Output>: StrictAxes<T::Output>,
{
    fn axes_check(&self, tiler: &T) -> Result<()> {
        self.clone().into_modes().strict_axes(&tiler.clone().wrap())
    }
}
impl<L: AxesCheck<T>, T> ValidateTiler<L, T> for AxesOp {
    fn validate(layout: &L, tiler: &T) -> Result<()> {
        layout.axes_check(tiler)
    }
}
impl<S: Integer + Div<T>, D: Integer, T: Integer>
    super::typed::Binary<Layout<S, D>, Layout<T, Int<1>>> for AxesOp
where
    (): super::composition::ComposeProduct<T, D>,
{
    type Output = Layout<
        super::typed::Pair<T, S::Output>,
        super::typed::Pair<D, <() as super::composition::ComposeProduct<T, D>>::Output>,
    >;
    fn run(layout: Layout<S, D>, tiler: Layout<T, Int<1>>) -> Result<Self::Output> {
        let next =
            <() as super::composition::ComposeProduct<T, D>>::product(tiler.shape, layout.stride)?;
        Ok(super::typed::join(
            Layout {
                shape: tiler.shape,
                stride: layout.stride,
            },
            Layout {
                shape: layout.shape / tiler.shape,
                stride: next,
            },
        ))
    }
}
impl<H, T, D, I> super::typed::Binary<Layout<IntTuple<H, T>, D>, Layout<I, Int<1>>> for AxesOp
where
    DivideOp: super::typed::Binary<Layout<IntTuple<H, T>, D>, Layout<I, Int<1>>>,
{
    type Output =
        <DivideOp as super::typed::Binary<Layout<IntTuple<H, T>, D>, Layout<I, Int<1>>>>::Output;
    fn run(layout: Layout<IntTuple<H, T>, D>, tiler: Layout<I, Int<1>>) -> Result<Self::Output> {
        DivideOp::run(layout, tiler)
    }
}
impl<D, I> super::typed::Binary<Layout<Nil, D>, Layout<I, Int<1>>> for AxesOp
where
    DivideOp: super::typed::Binary<Layout<Nil, D>, Layout<I, Int<1>>>,
{
    type Output = <DivideOp as super::typed::Binary<Layout<Nil, D>, Layout<I, Int<1>>>>::Output;
    fn run(layout: Layout<Nil, D>, tiler: Layout<I, Int<1>>) -> Result<Self::Output> {
        DivideOp::run(layout, tiler)
    }
}

super::typed::operation_api!(TypedDivideAxes, TypedDividedAxes, StaticDivideAxes, AxesOp, divide_axes_typed, AxesOp;
    (0, logical_divide_axes, static_logical_divide_axes),
    (1, zipped_divide_axes, static_zipped_divide_axes),
    (2, tiled_divide_axes, static_tiled_divide_axes),
    (3, flat_divide_axes, static_flat_divide_axes),
);

macro_rules! dynamic_divide_axes_methods {
    ($($name:ident),*) => {$ (
        impl DynLayout {
            /// 严格按轴整除；tile 长度须等于 rank，错误条件与同名自由函数一致。
            pub fn $name(&self, tiler: &[isize]) -> Result<DividedAxes<Self>> { $name(*self, tiler) }
        }
    )*};
}
dynamic_divide_axes_methods!(
    logical_divide_axes,
    zipped_divide_axes,
    tiled_divide_axes,
    flat_divide_axes
);
macro_rules! dynamic_divide_modes_methods {
    ($($name:ident),*) => {$ (
        impl DynLayout {
            /// 运行时 tuple tiler 分块，保留未被 tiler 覆盖的尾部模式。
            pub fn $name(&self, tiler: &[DynLayout]) -> Result<GroupedLayout> { $name(self, tiler) }
        }
    )*};
}
dynamic_divide_modes_methods!(
    logical_divide_modes,
    zipped_divide_modes,
    tiled_divide_modes,
    flat_divide_modes
);
