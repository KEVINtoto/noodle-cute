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

use crate::basis::StrideKind;
use crate::layout::{
    AffineLayoutMap, ComposedLayout, DynLayout, GroupedLayout, Layout, LayoutBuilder, LayoutMap,
    LayoutParts, Shape, StrideOf,
};
use crate::tuple::Product;
use crate::tuple::{Flatten, Item};
use crate::{Bool, Error, Result};

use super::coalesce::{CoalesceFold, FinishCoalesce};
use super::complement::{complement, complement_shape};
use super::composition::{Composition, composition, composition_layout};
use super::typed::{Arrange, Arranged, Binary, Joined, Kind, Transform, Transformed, join};
use super::{
    CoalescedLayout, Complement, Grouping, TileRest, TypedComplement, TypedCompose,
    TypedComposition,
};

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
        let tile = composition_layout(&affine, inner.tile())?;
        let rest = composition_layout(&affine, inner.rest())?;
        divided.normalized = Some(super::group_tile_rest(&tile, &rest, grouping)?);
    }
    Ok(divided)
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
fn divide_modes<L: AffineLayoutMap>(
    layout: &L,
    tiler: &[DynLayout],
) -> Result<(GroupedLayout, GroupedLayout)> {
    let layout = layout.as_grouped_layout()?;
    if tiler.len() > layout.rank() {
        return Err(Error::StructureMismatch);
    }
    let (mut tiles, mut rests) = (LayoutBuilder::new(), LayoutBuilder::new());
    for m in 0..layout.rank() {
        let mode = layout.mode(m)?;
        let Some(t) = tiler.get(m) else {
            rests.push_leaves(&mode)?;
            rests.close_mode()?;
            continue;
        };
        if t.size()? <= 0 {
            return Err(Error::InvalidShape);
        }
        // Same as single-tiler logical_divide: rest is the complement of the tiler relative to the coalesced mode shape.
        let rest = complement_shape(*t, mode.coalesce()?.shapes())?;
        tiles.push_leaves(composition_layout(&mode, t)?.flat())?;
        tiles.close_mode()?;
        rests.push_leaves(composition_layout(&mode, &rest)?.flat())?;
        rests.close_mode()?;
    }
    Ok((tiles.finish()?, rests.finish()?))
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
    S: Shape + Flatten,
    D: StrideOf<S> + Flatten,
    Layout<S::Output, D::Output>: CoalesceFold,
    <Layout<S::Output, D::Output> as CoalesceFold>::Output: FinishCoalesce,
    CoalescedLayout<S, D>: LayoutParts,
{
    type Shape = <CoalescedLayout<S, D> as LayoutParts>::Shape;
    fn cotarget(&self) -> Result<Self::Shape> {
        Ok(self.coalesce()?.into_layout().shape)
    }
}
type DivideInner<L, T> = Joined<T, TypedComplement<T, <L as CoalesceTarget>::Shape>>;
impl<L, T> Binary<L, T> for DivideOp
where
    L: CoalesceTarget + TypedCompose<DivideInner<L, T>>,
    T: LayoutMap + Clone + LayoutParts + Complement<L::Shape>,
    TypedComplement<T, L::Shape>: LayoutParts,
    TypedComposition<L, DivideInner<L, T>>: LayoutParts,
{
    type Output = TypedComposition<L, DivideInner<L, T>>;
    fn run(layout: L, tiler: T) -> Result<Self::Output> {
        if tiler.size()? == 0 {
            return Err(Error::InvalidShape);
        }
        let rest = tiler.clone().complement_typed(layout.cotarget()?)?;
        layout.compose(join(tiler, rest))
    }
}

super::typed::operation_api!(
    #[diagnostic::on_unimplemented(
        message = "layout `{Self}` cannot be divided by `{T}` with grouping {GROUPING}",
        label = "unsupported typed division",
        note = "use a tiler compatible with the layout's modes and choose a Grouping variant"
    )]
    TypedDivide, TypedDivided, Output, const GROUPING: Grouping = Grouping::Logical,
    DivideOp, Kind<GROUPING>, divide_typed;
    (Grouping::Logical, logical_divide),
    (Grouping::Zipped, zipped_divide),
    (Grouping::Tiled, tiled_divide),
    (Grouping::Flat, flat_divide),
);

//-----------------------------------------------------------------------------
// Identity layouts: single-leaf modes with basis strides.

/// Divides basis-strided modes through typed composition and an integer complement.
pub struct BasisDivideOp;

type BasisInner<T, S> = Joined<T, TypedComplement<T, S>>;

impl<S, D, T> Binary<Layout<S, D>, T> for BasisDivideOp
where
    S: Product + Clone,
    T: LayoutMap + Clone + LayoutParts + Complement<<S as Product>::Output>,
    TypedComplement<T, <S as Product>::Output>: LayoutParts,
    Layout<S, D>: TypedCompose<BasisInner<T, <S as Product>::Output>>,
    TypedComposition<Layout<S, D>, BasisInner<T, <S as Product>::Output>>: LayoutParts,
{
    type Output = TypedComposition<Layout<S, D>, BasisInner<T, <S as Product>::Output>>;
    fn run(layout: Layout<S, D>, tiler: T) -> Result<Self::Output> {
        if tiler.size()? == 0 {
            return Err(Error::InvalidShape);
        }
        let rest = tiler
            .clone()
            .complement_typed(layout.shape.clone().product_typed())?;
        layout.compose(join(tiler, rest))
    }
}

const BASIS<D: StrideKind>: bool = D::BASIS;

/// CuTe `zipped_divide(layout, tiler)` for every layout a tensor can hold.
///
/// Integer-strided layouts use [`TypedDivide`] with [`Grouping::Zipped`] and
/// validate the result. Basis layouts use the same tiler/complement construction
/// with coordinate-valued composition; they have no integer address range.
///
/// ```
/// use noodle_cute::algebra::ZippedDivide;
/// use noodle_cute::basis::make_identity_layout;
/// use noodle_cute::{DynInt, Int, shape};
///
/// let identity = make_identity_layout(shape![DynInt(10), DynInt(8)]);
/// let tiled = ZippedDivide::zipped_divide(&identity, shape![Int::<4>, Int::<4>])?;
/// // ((4, 4), (3, 2)): the tail tile of the first mode reaches row 11, not column 1.
/// assert_eq!(tiled.crd2idx(shape![(3i32, 0i32), (2i32, 0i32)]).0, 11);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
pub trait ZippedDivide<T> {
    /// The divided layout `((tile...), (rest...))`, of the same kind as `Self`.
    type Output;
    /// Divides by `tiler` and zips the tile and rest modes.
    fn zipped_divide(&self, tiler: T) -> Result<Self::Output>;
}

pub trait ZipKind<S, D, T, const G: Grouping = { Grouping::Zipped }> {
    type Output: LayoutParts;
    fn run(layout: Layout<S, D>, tiler: T) -> Result<Self::Output>;
}

impl<S, D, T, const G: Grouping> ZipKind<S, D, T, G> for Bool<false>
where
    Layout<S, D>: TypedDivide<T, G>,
    TypedDivided<Layout<S, D>, T, G>: LayoutParts,
{
    type Output = TypedDivided<Layout<S, D>, T, G>;
    fn run(layout: Layout<S, D>, tiler: T) -> Result<Self::Output> {
        layout.divide_typed(tiler)
    }
}

impl<S, D, T, const G: Grouping> ZipKind<S, D, T, G> for Bool<true>
where
    S: Item,
    Layout<S, D>: Transform<T, BasisDivideOp>,
    Kind<G>: Arrange<Transformed<Layout<S, D>, T, BasisDivideOp>, T>,
{
    type Output = Arranged<Layout<S, D>, T, BasisDivideOp, Kind<G>>;
    fn run(layout: Layout<S, D>, tiler: T) -> Result<Self::Output> {
        layout.shape.product()?;
        Ok(<Kind<G> as Arrange<_, T>>::arrange(
            layout.transform(tiler)?,
        ))
    }
}

type Zipped<S, D, T> = <Bool<{ BASIS::<D> }> as ZipKind<S, D, T>>::Output;

impl<S: Clone, D: Clone + StrideKind, T> ZippedDivide<T> for Layout<S, D>
where
    Bool<{ BASIS::<D> }>: ZipKind<S, D, T>,
{
    type Output =
        Layout<<Zipped<S, D, T> as LayoutParts>::Shape, <Zipped<S, D, T> as LayoutParts>::Stride>;
    fn zipped_divide(&self, tiler: T) -> Result<Self::Output> {
        Ok(<Bool<{ BASIS::<D> }> as ZipKind<S, D, T>>::run(self.clone(), tiler)?.into_layout())
    }
}

// layout_composed.hpp: zipped_divide(ComposedLayout) divides `layout_b`.
impl<A: Clone, O: Clone, B: ZippedDivide<T>, T> ZippedDivide<T> for ComposedLayout<A, O, B> {
    type Output = ComposedLayout<A, O, B::Output>;
    fn zipped_divide(&self, tiler: T) -> Result<Self::Output> {
        self.map_b(|b| b.zipped_divide(tiler))
    }
}

/// CuTe `logical_divide(layout, tiler)` for plain and composed layouts.
///
/// A plain layout uses [`TypedDivide`] with [`Grouping::Logical`]; a composed
/// layout divides `layout_b`, as `layout_composed.hpp` does.
pub trait LogicalDivide<T> {
    /// The divided layout.
    type Output;
    /// Divides by `tiler`, keeping `(tile, rest)` within each mode.
    fn logical_divide_by(&self, tiler: T) -> Result<Self::Output>;
}

impl<S: Clone, D: Clone + StrideKind, T> LogicalDivide<T> for Layout<S, D>
where
    Bool<{ BASIS::<D> }>: ZipKind<S, D, T, { Grouping::Logical }>,
{
    type Output = <Bool<{ BASIS::<D> }> as ZipKind<S, D, T, { Grouping::Logical }>>::Output;
    fn logical_divide_by(&self, tiler: T) -> Result<Self::Output> {
        <Bool<{ BASIS::<D> }> as ZipKind<S, D, T, { Grouping::Logical }>>::run(self.clone(), tiler)
    }
}

impl<A: Clone, O: Clone, B: LogicalDivide<T>, T> LogicalDivide<T> for ComposedLayout<A, O, B> {
    type Output = ComposedLayout<A, O, B::Output>;
    fn logical_divide_by(&self, tiler: T) -> Result<Self::Output> {
        self.map_b(|b| b.logical_divide_by(tiler))
    }
}
