//! Product for static, mixed, and runtime layouts.
//!
//! Typed methods accept integer, layout, and nested tuple tilers and reuse
//! `complement_extended` and composition. Blocked and raked products pad the tiler with
//! `1:0` modes before composing, then zip the block and rest mode by mode. The runtime
//! functions keep the lazy composition; `*_product_modes` accept partial tuple tilers.
//!
//! ```
//! use noodle_cute::{DynInt, Int, IntTuple, Nil, StaticValue};
//! use noodle_cute::algebra::TypedProductLayout;
//! use noodle_cute::layout::{Layout, LayoutMap};
//! type Pair<A, B> = IntTuple<A, IntTuple<B, Nil>>;
//! const C: Layout<Pair<Int<2>, Int<3>>, Pair<Int<1>, Int<2>>> =
//!     <TypedProductLayout<Layout<Int<2>, Int<1>>, Int<3>> as StaticValue>::VALUE;
//! let block = Layout::new(DynInt(2), Int::<1>)?;
//! let result = block.logical_product(Int::<3>)?;
//! assert_eq!(result.index(5)?, 5);
//! # Ok::<(), noodle_cute::Error>(())
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, StaticValue, algebra::TypedProductLayout, layout::Layout};
//! const C: TypedProductLayout<Layout<Int<0>, Int<1>>, Int<3>> = StaticValue::VALUE;
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, StaticValue, algebra::TypedProductLayout, layout::Layout};
//! const C: TypedProductLayout<Layout<Int<2>, Int<2>>, Layout<Int<2>, Int<3>>> = StaticValue::VALUE;
//! ```
//! ```compile_fail
//! use noodle_cute::{DynInt, Int, StaticValue, algebra::TypedProductLayout, layout::Layout};
//! const C: TypedProductLayout<Layout<Int<2>, Int<1>>, DynInt> = StaticValue::VALUE;
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, int_tuple, layout::Layout};
//! let block = Layout::new(Int::<2>, Int::<1>).unwrap();
//! let _ = block.blocked_product(int_tuple![Int::<3>]);
//! ```

use core::marker::ConstParamTy;

use crate::layout::{
    AffineLayoutMap, DynLayout, GroupedLayout, Layout, LayoutBuilder, LayoutMap, LayoutParts,
    Shape, StrideOf,
};
use crate::{Error, Int, IntTuple, Nil, Result};

use super::complement::complement_extended;
use super::composition::{Composition, composition, composition_layout};
use super::typed::{Arrange, Binary, IntoModes, Joined, Kind, Pair, Split, join};
use super::{
    Complement, ComplementMode, Grouping, ProductKind, TileRest, TypedComplement, TypedCompose,
    TypedComposition,
};

/// Product layout formed by adding a tile address to a rest address mapped through the complement of the tile.
pub type Produced<A, B> = TileRest<A, Composition<DynLayout, B>>;

// Construct a product layout from a block and a tiler layout, returning a composed layout.
fn product<A: AffineLayoutMap, B: LayoutMap>(
    block: A,
    tiler: B,
    grouping: Grouping,
) -> Result<Produced<A, B>> {
    // CuTe: logical_product(block, tiler) = (block, complement(block) o tiler), with complement
    // having no cotarget; rest values extend through its terminal 1:(s_last*d_last) mode.
    let complement = complement_extended(block.clone())?;
    let rest = composition(complement, tiler)?;
    let result = TileRest::new(block, rest, grouping);
    result.size()?;
    Ok(result)
}

/// CuTe `logical_product(block, tiler)`: produces two groups `(block, complement(block) ∘ tiler)`.
///
/// `block` must satisfy complement constraints (nonnegative strides, non-overlapping).
/// The resulting size is `block.size() * tiler.size()`. A non-affine `tiler` is evaluated
/// pointwise and must have nonnegative addresses.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::logical_product;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let block = DynLayout::new(&[2], &[1])?;
/// let tiler = DynLayout::new(&[3], &[1])?;
/// let prod = logical_product(block, tiler)?;
/// assert_eq!(prod.size()?, 6);
/// assert_eq!(prod.rank(), 2);
/// assert_eq!(prod.index(5)?, 5);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// - Returns [`Error::Unsupported`] if `block` has negative strides or size 0.
/// - Returns [`Error::NonInjective`] if `block` strides overlap.
/// - Returns [`Error::CoordinateOutOfBounds`] if a non-affine `tiler` produces negative addresses.
/// - Returns [`Error::Overflow`] if size computation overflows.
pub fn logical_product<A: AffineLayoutMap, B: LayoutMap>(
    block: A,
    tiler: B,
) -> Result<Produced<A, B>> {
    product(block, tiler, Grouping::Logical)
}

/// CuTe `zipped_product(block, tiler)`: for a single block layout, zipped product produces the same two groups as logical product.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::zipped_product;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let block = DynLayout::new(&[2], &[1])?;
/// let tiler = DynLayout::new(&[3], &[1])?;
/// let prod = zipped_product(block, tiler)?;
/// assert_eq!(prod.size()?, 6);
/// assert_eq!(prod.rank(), 2);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`logical_product`].
pub fn zipped_product<A: AffineLayoutMap, B: LayoutMap>(
    block: A,
    tiler: B,
) -> Result<Produced<A, B>> {
    product(block, tiler, Grouping::Zipped)
}

/// CuTe `tiled_product(block, tiler)`: groups `block` into mode 0 and exposes each axis of `tiler` individually.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::tiled_product;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let block = DynLayout::new(&[2], &[1])?;
/// let tiler = DynLayout::new(&[3], &[1])?;
/// let prod = tiled_product(block, tiler)?;
/// assert_eq!(prod.size()?, 6);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`logical_product`].
pub fn tiled_product<A: AffineLayoutMap, B: LayoutMap>(
    block: A,
    tiler: B,
) -> Result<Produced<A, B>> {
    product(block, tiler, Grouping::Tiled)
}

/// CuTe `flat_product(block, tiler)`: flattens all axes of `block` followed by all axes of `tiler`.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::flat_product;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let block = DynLayout::new(&[2], &[1])?;
/// let tiler = DynLayout::new(&[3], &[1])?;
/// let prod = flat_product(block, tiler)?;
/// assert_eq!(prod.size()?, 6);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`logical_product`].
pub fn flat_product<A: AffineLayoutMap, B: LayoutMap>(
    block: A,
    tiler: B,
) -> Result<Produced<A, B>> {
    product(block, tiler, Grouping::Flat)
}

// CuTe blocked/raked product: pads both sides to max(rank) with unit 1:0 modes,
// then zips the two groups of logical_product mode-by-mode; raked places rest before block in each mode.
fn rank_product<A: AffineLayoutMap, B: AffineLayoutMap>(
    block: &A,
    tiler: &B,
    raked: bool,
) -> Result<GroupedLayout> {
    let block = block.as_grouped_layout()?;
    let tiler = tiler.as_grouped_layout()?;
    let rank = block.rank().max(tiler.rank());
    // Padded 1:0 modes are filtered out by complement, not affecting the complement layout.
    let complement = complement_extended(block)?;
    let unit = DynLayout::new(&[1], &[0])?;
    let mut out = LayoutBuilder::new();
    for m in 0..rank {
        let b = if m < block.rank() {
            block.mode(m)?
        } else {
            unit
        };
        // CuTe composition on a stride-0 rhs produces s:0 directly; padding modes evaluate to 1:0.
        let r = if m < tiler.rank() {
            let mode = GroupedLayout::from_flat(tiler.mode(m)?)?;
            *composition_layout(&complement, &mode)?.flat()
        } else {
            unit
        };
        let (first, second) = if raked { (r, b) } else { (b, r) };
        out.push_leaves(&first)?;
        out.push_leaves(&second)?;
        out.close_mode()?;
    }
    out.finish()
}

/// CuTe `blocked_product(block, tiler)`: rank is `max(block.rank(), tiler.rank())`, where the `m`-th mode is
/// `(block_m, (complement(block) ∘ tiler)_m)`, repeating the entire block along each mode.
///
/// Missing modes are padded with `1:0`; these shape-1 leaves are retained to match CuTe's mode structure.
/// Each mode must satisfy the divisibility requirements of [`composition_layout`].
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::blocked_product;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let block = DynLayout::new(&[2], &[1])?;
/// let tiler = DynLayout::new(&[3], &[1])?;
/// let bp = blocked_product(block, tiler)?;
/// assert_eq!(bp.rank(), 1);
/// assert_eq!(bp.shapes(), &[2, 3]);
/// assert_eq!(bp.strides(), &[1, 2]);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// - Returns [`Error::NotDivisible`] if mode composition divisibility conditions fail.
/// - Returns [`Error::CapacityExceeded`] if leaf or group storage exceeds [`MAX_RANK`](crate::layout::MAX_RANK).
pub fn blocked_product<A: AffineLayoutMap, B: AffineLayoutMap>(
    block: A,
    tiler: B,
) -> Result<GroupedLayout> {
    rank_product(&block, &tiler, false)
}

/// CuTe `raked_product(block, tiler)`: has the same leaves as [`blocked_product`], but each mode is
/// ordered as `(rest_m, block_m)`, interleaving (raking) elements of `block` across modes.
///
/// Each mode must satisfy the divisibility requirements of [`composition_layout`].
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::raked_product;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let block = DynLayout::new(&[2], &[1])?;
/// let tiler = DynLayout::new(&[3], &[1])?;
/// let rp = raked_product(block, tiler)?;
/// assert_eq!(rp.rank(), 1);
/// assert_eq!(rp.shapes(), &[3, 2]);
/// assert_eq!(rp.strides(), &[2, 1]);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`blocked_product`].
pub fn raked_product<A: AffineLayoutMap, B: AffineLayoutMap>(
    block: A,
    tiler: B,
) -> Result<GroupedLayout> {
    rank_product(&block, &tiler, true)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum ProductPreparation {
    Logical,
    Ranked,
}

pub struct ProductOp<const MODE: ProductPreparation>;
pub const PREPARATION<const KIND: ProductKind>: ProductPreparation = match KIND {
    ProductKind::Blocked | ProductKind::Raked => ProductPreparation::Ranked,
    _ => ProductPreparation::Logical,
};
impl<S, D, TS, TD> Binary<Layout<S, D>, Layout<TS, TD>>
    for ProductOp<{ ProductPreparation::Logical }>
where
    S: Shape,
    D: StrideOf<S>,
    TS: Shape,
    TD: StrideOf<TS>,
    Layout<S, D>: Complement<Nil, { ComplementMode::Extended }>,
    TypedComplement<Layout<S, D>, Nil, { ComplementMode::Extended }>: TypedCompose<Layout<TS, TD>>,
    ProductRest<S, D, TS, TD>: LayoutParts,
{
    type Output = Joined<Layout<S, D>, ProductRest<S, D, TS, TD>>;
    fn run(layout: Layout<S, D>, tiler: Layout<TS, TD>) -> Result<Self::Output> {
        let complement = layout.clone().complement_typed(Nil)?;
        let rest = complement.compose(tiler)?;
        Ok(join(layout, rest))
    }
}
type PaddedTiler<B, T> = <<T as IntoModes>::Output as PadModes<<B as IntoModes>::Output>>::Output;
// CuTe blocked/raked_product pad both sides to the larger rank before logical_product, so each
// tiler mode composes into exactly one rest mode, even when it yields a multi-mode layout.
impl<B, T> Binary<B, T> for ProductOp<{ ProductPreparation::Ranked }>
where
    B: IntoModes,
    T: IntoModes,
    T::Output: PadModes<B::Output>,
    ProductOp<{ ProductPreparation::Logical }>: Binary<B, PaddedTiler<B, T>>,
{
    type Output =
        <ProductOp<{ ProductPreparation::Logical }> as Binary<B, PaddedTiler<B, T>>>::Output;
    fn run(block: B, tiler: T) -> Result<Self::Output> {
        <ProductOp<{ ProductPreparation::Logical }> as Binary<B, PaddedTiler<B, T>>>::run(
            block,
            tiler.into_modes().pad_modes(),
        )
    }
}

pub trait PadModes<B> {
    type Output;
    fn pad_modes(self) -> Self::Output;
}
impl PadModes<Layout<Nil, Nil>> for Layout<Nil, Nil> {
    type Output = Self;
    fn pad_modes(self) -> Self {
        self
    }
}
impl<S, ST, D, DT> PadModes<Layout<Nil, Nil>> for Layout<IntTuple<S, ST>, IntTuple<D, DT>> {
    type Output = Self;
    fn pad_modes(self) -> Self {
        self
    }
}
impl<S, ST, D, DT, B, BT, E, ET> PadModes<Layout<IntTuple<B, BT>, IntTuple<E, ET>>>
    for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    Layout<ST, DT>: PadModes<Layout<BT, ET>>,
    <Layout<ST, DT> as PadModes<Layout<BT, ET>>>::Output: LayoutParts,
{
    type Output = Layout<
        IntTuple<S, <<Layout<ST, DT> as PadModes<Layout<BT, ET>>>::Output as LayoutParts>::Shape>,
        IntTuple<D, <<Layout<ST, DT> as PadModes<Layout<BT, ET>>>::Output as LayoutParts>::Stride>,
    >;
    fn pad_modes(self) -> Self::Output {
        let tail = Layout {
            shape: self.shape.1,
            stride: self.stride.1,
        }
        .pad_modes()
        .into_layout();
        Layout {
            shape: IntTuple(self.shape.0, tail.shape),
            stride: IntTuple(self.stride.0, tail.stride),
        }
    }
}
impl<B, BT, E, ET> PadModes<Layout<IntTuple<B, BT>, IntTuple<E, ET>>> for Layout<Nil, Nil>
where
    Layout<Nil, Nil>: PadModes<Layout<BT, ET>>,
    <Layout<Nil, Nil> as PadModes<Layout<BT, ET>>>::Output: LayoutParts,
{
    type Output = Layout<
        IntTuple<
            Int<1>,
            <<Layout<Nil, Nil> as PadModes<Layout<BT, ET>>>::Output as LayoutParts>::Shape,
        >,
        IntTuple<
            Int<0>,
            <<Layout<Nil, Nil> as PadModes<Layout<BT, ET>>>::Output as LayoutParts>::Stride,
        >,
    >;
    fn pad_modes(self) -> Self::Output {
        let tail = self.pad_modes().into_layout();
        Layout {
            shape: IntTuple(Int, tail.shape),
            stride: IntTuple(Int, tail.stride),
        }
    }
}

type ProductRest<S, D, TS, TD> = TypedComposition<
    TypedComplement<Layout<S, D>, Nil, { ComplementMode::Extended }>,
    Layout<TS, TD>,
>;

pub struct ProductArrangement<const KIND: ProductKind>;

macro_rules! grouped_arrangement {
    ($($kind:ident),*) => { $(
        impl<L, T> Arrange<L, T> for ProductArrangement<{ ProductKind::$kind }>
        where Kind<{ Grouping::$kind }>: Arrange<L, T>,
        {
            type Output = <Kind<{ Grouping::$kind }> as Arrange<L, T>>::Output;
            fn arrange(layout: L) -> Self::Output {
                <Kind<{ Grouping::$kind }> as Arrange<L, T>>::arrange(layout)
            }
        }
    )* };
}
grouped_arrangement!(Logical, Zipped, Tiled, Flat);

super::typed::operation_api!(
    #[diagnostic::on_unimplemented(
        message = "layout `{Self}` cannot form a typed product with `{T}` and product kind {KIND}",
        label = "unsupported typed product",
        note = "choose a ProductKind compatible with the tiler; blocked and raked products require layout tilers"
    )]
    TypedProduct, TypedProductLayout, Output, const KIND: ProductKind = ProductKind::Logical,
    ProductOp<{ PREPARATION::<KIND> }>, ProductArrangement<KIND>, product_typed;
    (ProductKind::Logical, logical_product),
    (ProductKind::Zipped, zipped_product),
    (ProductKind::Tiled, tiled_product),
    (ProductKind::Flat, flat_product),
    (ProductKind::Blocked, blocked_product),
    (ProductKind::Raked, raked_product),
);

pub trait ZipModes<R> {
    type Output: LayoutParts;
    fn zip_modes(self, rhs: R) -> Self::Output;
}
impl ZipModes<Layout<Nil, Nil>> for Layout<Nil, Nil> {
    type Output = Self;
    fn zip_modes(self, _: Self) -> Self {
        self
    }
}
impl<S, ST, D, DT, R, RT, E, ET> ZipModes<Layout<IntTuple<R, RT>, IntTuple<E, ET>>>
    for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    Layout<ST, DT>: ZipModes<Layout<RT, ET>>,
{
    type Output = Layout<
        IntTuple<
            Pair<S, R>,
            <<Layout<ST, DT> as ZipModes<Layout<RT, ET>>>::Output as LayoutParts>::Shape,
        >,
        IntTuple<
            Pair<D, E>,
            <<Layout<ST, DT> as ZipModes<Layout<RT, ET>>>::Output as LayoutParts>::Stride,
        >,
    >;
    fn zip_modes(self, rhs: Layout<IntTuple<R, RT>, IntTuple<E, ET>>) -> Self::Output {
        let head = join(
            Layout {
                shape: self.shape.0,
                stride: self.stride.0,
            },
            Layout {
                shape: rhs.shape.0,
                stride: rhs.stride.0,
            },
        );
        let tail = Layout {
            shape: self.shape.1,
            stride: self.stride.1,
        }
        .zip_modes(Layout {
            shape: rhs.shape.1,
            stride: rhs.stride.1,
        })
        .into_layout();
        Layout {
            shape: IntTuple(head.shape, tail.shape),
            stride: IntTuple(head.stride, tail.stride),
        }
    }
}
type UnitMode = Layout<IntTuple<Int<1>, Nil>, IntTuple<Int<0>, Nil>>;
fn unit_mode() -> UnitMode {
    Layout {
        shape: IntTuple(Int, Nil),
        stride: IntTuple(Int, Nil),
    }
}
impl<S, ST, D, DT> ZipModes<Layout<Nil, Nil>> for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    Self: ZipModes<UnitMode>,
{
    type Output = <Self as ZipModes<UnitMode>>::Output;
    fn zip_modes(self, _: Layout<Nil, Nil>) -> Self::Output {
        self.zip_modes(unit_mode())
    }
}
impl<S, ST, D, DT> ZipModes<Layout<IntTuple<S, ST>, IntTuple<D, DT>>> for Layout<Nil, Nil>
where
    UnitMode: ZipModes<Layout<IntTuple<S, ST>, IntTuple<D, DT>>>,
{
    type Output = <UnitMode as ZipModes<Layout<IntTuple<S, ST>, IntTuple<D, DT>>>>::Output;
    fn zip_modes(self, rhs: Layout<IntTuple<S, ST>, IntTuple<D, DT>>) -> Self::Output {
        unit_mode().zip_modes(rhs)
    }
}
type BlockModes<L> = <<L as Split>::Tile as IntoModes>::Output;
type RestModes<L> = <<L as Split>::Rest as IntoModes>::Output;
impl<L: Split, S, D> Arrange<L, Layout<S, D>> for ProductArrangement<{ ProductKind::Blocked }>
where
    L::Tile: IntoModes,
    L::Rest: IntoModes,
    BlockModes<L>: ZipModes<RestModes<L>>,
{
    type Output = <BlockModes<L> as ZipModes<RestModes<L>>>::Output;
    fn arrange(layout: L) -> Self::Output {
        let (block, rest) = layout.split();
        block.into_modes().zip_modes(rest.into_modes())
    }
}
impl<L: Split, S, D> Arrange<L, Layout<S, D>> for ProductArrangement<{ ProductKind::Raked }>
where
    L::Tile: IntoModes,
    L::Rest: IntoModes,
    RestModes<L>: ZipModes<BlockModes<L>>,
{
    type Output = <RestModes<L> as ZipModes<BlockModes<L>>>::Output;
    fn arrange(layout: L) -> Self::Output {
        let (block, rest) = layout.split();
        rest.into_modes().zip_modes(block.into_modes())
    }
}
fn product_modes<L: AffineLayoutMap>(
    block: &L,
    tiler: &[DynLayout],
    grouping: Grouping,
) -> Result<GroupedLayout> {
    let block = block.as_grouped_layout()?;
    if tiler.len() > block.rank() {
        return Err(Error::StructureMismatch);
    }
    let (mut blocks, mut rests) = (LayoutBuilder::new(), LayoutBuilder::new());
    for m in 0..block.rank() {
        let mode = block.mode(m)?;
        if let Some(t) = tiler.get(m) {
            let comp = complement_extended(mode)?;
            blocks.push_leaves(&mode)?;
            blocks.close_mode()?;
            rests.push_leaves(composition_layout(&comp, t)?.flat())?;
        } else {
            rests.push_leaves(&mode)?;
        }
        rests.close_mode()?;
    }
    super::group_modes(&blocks.finish()?, &rests.finish()?, grouping)
}
/// CuTe `logical_product(block, tiler)` with a per-mode tuple tiler: mode `m` becomes
/// `(block_m, complement(block_m) ∘ tiler[m])`; trailing modes not covered by `tiler` are kept.
///
/// # Errors
///
/// - Returns [`Error::StructureMismatch`] if `tiler.len() > block.rank()`.
/// - Returns complement or composition errors of the individual modes.
pub fn logical_product_modes<L: AffineLayoutMap>(
    block: &L,
    tiler: &[DynLayout],
) -> Result<GroupedLayout> {
    product_modes(block, tiler, Grouping::Logical)
}

/// CuTe `zipped_product(block, tiler)` with a per-mode tuple tiler:
/// `((block_0, block_1, ...), (rest_0, rest_1, ..., trailing_modes...))`.
///
/// # Errors
///
/// See [`logical_product_modes`].
pub fn zipped_product_modes<L: AffineLayoutMap>(
    block: &L,
    tiler: &[DynLayout],
) -> Result<GroupedLayout> {
    product_modes(block, tiler, Grouping::Zipped)
}

/// CuTe `tiled_product(block, tiler)` with a per-mode tuple tiler:
/// `((block_0, block_1, ...), rest_0, rest_1, ...)`.
///
/// # Errors
///
/// See [`logical_product_modes`].
pub fn tiled_product_modes<L: AffineLayoutMap>(
    block: &L,
    tiler: &[DynLayout],
) -> Result<GroupedLayout> {
    product_modes(block, tiler, Grouping::Tiled)
}

/// CuTe `flat_product(block, tiler)` with a per-mode tuple tiler:
/// `(block_0, block_1, ..., rest_0, rest_1, ...)`.
///
/// # Errors
///
/// See [`logical_product_modes`].
pub fn flat_product_modes<L: AffineLayoutMap>(
    block: &L,
    tiler: &[DynLayout],
) -> Result<GroupedLayout> {
    product_modes(block, tiler, Grouping::Flat)
}
