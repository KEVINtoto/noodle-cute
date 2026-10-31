//! 全静态、混合类型与运行时布局的 product。
//!
//! 类型化方法支持整数、布局和嵌套 tuple tiler，复用 complement_extended 和
//! composition。blocked/raked 先把 tiler 按顶层模式补齐 1:0 再组合，最后逐模式 zip。
//! 运行时自由函数保留惰性组合；`*_product_modes` 支持部分 tuple tiler。
//!
//! ```
//! use noodle_cute::{Int, IntTuple, Nil, DynInt, layout::{Layout, LayoutMap}};
//! use noodle_cute::algebra::static_logical_product;
//! type Pair<A, B> = IntTuple<A, IntTuple<B, Nil>>;
//! const C: Layout<Pair<Int<2>, Int<3>>, Pair<Int<1>, Int<2>>> = static_logical_product(
//!     Layout { shape: Int::<2>, stride: Int::<1> }, Int::<3>,
//! );
//! let block = Layout::new(DynInt(2), Int::<1>)?;
//! let result = block.logical_product(Int::<3>)?;
//! assert_eq!(result.index(5)?, 5);
//! # Ok::<(), noodle_cute::Error>(())
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, layout::Layout, algebra::static_logical_product};
//! let _ = static_logical_product(Layout { shape: Int::<0>, stride: Int::<1> }, Int::<3>);
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, layout::Layout, algebra::static_logical_product};
//! let _ = static_logical_product(
//!     Layout { shape: Int::<2>, stride: Int::<2> },
//!     Layout { shape: Int::<2>, stride: Int::<3> },
//! );
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, DynInt, layout::Layout, algebra::static_logical_product};
//! let _ = static_logical_product(Layout { shape: Int::<2>, stride: Int::<1> }, DynInt(3));
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, int_tuple, layout::Layout};
//! let block = Layout::new(Int::<2>, Int::<1>).unwrap();
//! let _ = block.blocked_product(int_tuple![Int::<3>]);
//! ```

use crate::layout::{
    AffineLayoutMap, DynLayout, GroupedLayout, Layout, LayoutMap, LayoutParts, MAX_RANK,
};
use crate::{Error, Int, IntTuple, Nil, Result};

use super::coalesce::Decision;
use super::complement::complement_extended;
use super::composition::{Composition, composition, composition_layout};
use super::typed::{Arrange, IntoModes, Split};
use super::{Grouping, Leaves, TileRest};

/// Product layout formed by adding a tile address to a rest address mapped through the complement of the tile.
pub type Product<A, B> = TileRest<A, Composition<DynLayout, B>>;

// Construct a product layout from a block and a tiler layout, returning a composed layout.
fn product<A: AffineLayoutMap, B: LayoutMap>(
    block: A,
    tiler: B,
    grouping: Grouping,
) -> Result<Product<A, B>> {
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
/// use noodle_cute::algebra::product::logical_product;
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
) -> Result<Product<A, B>> {
    product(block, tiler, Grouping::Logical)
}

/// CuTe `zipped_product(block, tiler)`: for a single block layout, zipped product produces the same two groups as logical product.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::product::zipped_product;
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
) -> Result<Product<A, B>> {
    product(block, tiler, Grouping::Zipped)
}

/// CuTe `tiled_product(block, tiler)`: groups `block` into mode 0 and exposes each axis of `tiler` individually.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::product::tiled_product;
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
) -> Result<Product<A, B>> {
    product(block, tiler, Grouping::Tiled)
}

/// CuTe `flat_product(block, tiler)`: flattens all axes of `block` followed by all axes of `tiler`.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::product::flat_product;
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
pub fn flat_product<A: AffineLayoutMap, B: LayoutMap>(block: A, tiler: B) -> Result<Product<A, B>> {
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
    let mut leaves = Leaves::new();
    let mut groups = [0usize; MAX_RANK];
    for (m, group) in groups.iter_mut().enumerate().take(rank) {
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
        for part in [first, second] {
            for (&s, &d) in part.shapes().iter().zip(part.strides()) {
                leaves.push(s, d)?;
            }
        }
        *group = first.rank() + second.rank();
    }
    GroupedLayout::new(leaves.layout()?, &groups[..rank])
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
/// use noodle_cute::algebra::product::blocked_product;
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
/// - Returns [`Error::CapacityExceeded`] if leaf or group storage exceeds [`MAX_RANK`].
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
/// use noodle_cute::algebra::product::raked_product;
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

#[doc(hidden)]
pub struct ProductOp<const RANKED: bool>;
#[doc(hidden)]
pub const RANKED<const KIND: u8>: bool = KIND >= 4;
impl<S, D, TS, TD> super::typed::Binary<Layout<S, D>, Layout<TS, TD>> for ProductOp<false>
where
    S: crate::layout::Shape,
    D: crate::tuple::Item<Profile = S::Profile>,
    TS: crate::layout::Shape,
    TD: crate::tuple::Item<Profile = TS::Profile>,
    Layout<S, D>: super::Complement<crate::Nil, true>,
    super::TypedComplement<Layout<S, D>, crate::Nil, true>: super::TypedCompose<Layout<TS, TD>>,
    ProductRest<S, D, TS, TD>: crate::layout::LayoutParts,
{
    type Output = super::typed::Joined<Layout<S, D>, ProductRest<S, D, TS, TD>>;
    fn run(layout: Layout<S, D>, tiler: Layout<TS, TD>) -> Result<Self::Output> {
        use super::{Complement, TypedCompose};
        let complement = layout.clone().complement_typed(crate::Nil)?;
        let rest = complement.compose(tiler)?;
        Ok(super::typed::join(layout, rest))
    }
}
type PaddedTiler<B, T> = <<T as IntoModes>::Output as PadModes<<B as IntoModes>::Output>>::Output;
// CuTe blocked/raked_product pad both sides to the larger rank before logical_product, so each
// tiler mode composes into exactly one rest mode, even when it yields a multi-mode layout.
impl<B, T> super::typed::Binary<B, T> for ProductOp<true>
where
    B: IntoModes,
    T: IntoModes,
    T::Output: PadModes<B::Output>,
    ProductOp<false>: super::typed::Binary<B, PaddedTiler<B, T>>,
{
    type Output = <ProductOp<false> as super::typed::Binary<B, PaddedTiler<B, T>>>::Output;
    fn run(block: B, tiler: T) -> Result<Self::Output> {
        <ProductOp<false> as super::typed::Binary<B, PaddedTiler<B, T>>>::run(
            block,
            tiler.into_modes().pad_modes(),
        )
    }
}

#[doc(hidden)]
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

type ProductRest<S, D, TS, TD> =
    super::TypedComposition<super::TypedComplement<Layout<S, D>, crate::Nil, true>, Layout<TS, TD>>;

super::typed::operation_api!(TypedProduct, TypedProductLayout, StaticProduct, ProductOp<{ RANKED::<KIND> }>, product_typed, ();
    (0, logical_product, static_logical_product),
    (1, zipped_product, static_zipped_product),
    (2, tiled_product, static_tiled_product),
    (3, flat_product, static_flat_product),
    (4, blocked_product, static_blocked_product),
    (5, raked_product, static_raked_product),
);

macro_rules! dynamic_product_methods {
    ($($name:ident),*) => {$ (
        impl DynLayout {
            /// 运行时乘积入口，沿用自由函数的补布局、扩展域与错误规则。
            pub fn $name<T: LayoutMap>(&self, tiler: T) -> Result<Product<Self, T>> {
                $name(*self, tiler)
            }
        }
    )*};
}
dynamic_product_methods!(logical_product, zipped_product, tiled_product, flat_product);

#[doc(hidden)]
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
            super::typed::Pair<S, R>,
            <<Layout<ST, DT> as ZipModes<Layout<RT, ET>>>::Output as LayoutParts>::Shape,
        >,
        IntTuple<
            super::typed::Pair<D, E>,
            <<Layout<ST, DT> as ZipModes<Layout<RT, ET>>>::Output as LayoutParts>::Stride,
        >,
    >;
    fn zip_modes(self, rhs: Layout<IntTuple<R, RT>, IntTuple<E, ET>>) -> Self::Output {
        let head = super::typed::join(
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
impl<L: Split, S, D> Arrange<L, Layout<S, D>> for Decision<4>
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
impl<L: Split, S, D> Arrange<L, Layout<S, D>> for Decision<5>
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
impl DynLayout {
    /// 运行时 blocked_product，缺失模式以 1:0 补齐。
    pub fn blocked_product<T: AffineLayoutMap>(&self, tiler: T) -> Result<GroupedLayout> {
        blocked_product(*self, tiler)
    }
    /// 运行时 raked_product，各模式中 rest 位于 block 之前。
    pub fn raked_product<T: AffineLayoutMap>(&self, tiler: T) -> Result<GroupedLayout> {
        raked_product(*self, tiler)
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
    let (mut blocks, mut rests) = (super::Modes::new(), super::Modes::new());
    for m in 0..block.rank() {
        let mode = block.mode(m)?;
        if let Some(t) = tiler.get(m) {
            let comp = complement_extended(mode)?;
            blocks.push(&mode)?;
            rests.push(composition_layout(&comp, t)?.flat())?;
        } else {
            rests.push(&mode)?;
        }
    }
    super::group_modes(&blocks, &rests, grouping)
}
macro_rules! runtime_product_modes {
    ($(($name:ident, $group:ident)),*) => {$ (
        /// 按各顶层模式独立求 product；未覆盖的尾部模式保留原值。
        ///
        /// tiler 超过 rank 返回 StructureMismatch；补布局或组合失败沿用相应错误。
        pub fn $name<L: AffineLayoutMap>(block: &L, tiler: &[DynLayout]) -> Result<GroupedLayout> {
            product_modes(block, tiler, Grouping::$group)
        }
        impl DynLayout {
            /// 运行时 tuple tiler 乘积，与同名自由函数共用算法。
            pub fn $name(&self, tiler: &[DynLayout]) -> Result<GroupedLayout> { $name(self, tiler) }
        }
    )*};
}
runtime_product_modes!(
    (logical_product_modes, Logical),
    (zipped_product_modes, Zipped),
    (tiled_product_modes, Tiled),
    (flat_product_modes, Flat)
);
