//! Vectorization analysis, tiling to a shape, and the swizzle-free part of a
//! layout: CuTe `max_common_vector`, `max_common_layout`, `tile_to_shape`, and
//! `get_nonswizzle_portion` (`layout.hpp`, `swizzle_layout.hpp`,
//! `layout_composed.hpp`).

use crate::int::{Arith, CeilQuot, Int, Op, ceil_quot};
use crate::layout::{
    ColMajorLayout, CompactLeft, ComposedLayout, Layout, LayoutParts, MAX_RANK, Shape,
    StaticLeaves, StrideOf,
};
use crate::swizzle::Swizzle;
use crate::tuple::{Flatten, Get, IntTuple, Nil, ProductEach, Wrap};
use crate::{Bool, Result, StaticValue};

use super::coalesce::{CoalesceFold, FinishCoalesce};
use super::{
    CoalescedLayout, ProductKind, TypedCompose, TypedComposition, TypedInverse, TypedInvert,
    TypedProduct, TypedProductLayout,
};

type Composed<A, B> = TypedComposition<A, TypedInverse<B>>;
type CShape<A, B> = <Composed<A, B> as LayoutParts>::Shape;
type CStride<A, B> = <Composed<A, B> as LayoutParts>::Stride;
type Common<A, B> = CoalescedLayout<CShape<A, B>, CStride<A, B>>;
type CommonShape<A, B> = <Common<A, B> as LayoutParts>::Shape;
type CommonStride<A, B> = <Common<A, B> as LayoutParts>::Stride;
type FlatComposed<A, B> =
    Layout<<CShape<A, B> as Flatten>::Output, <CStride<A, B> as Flatten>::Output>;

/// Whether the leading mode of a coalesced layout is static with stride one.
pub const LEADS<S: StaticLeaves, D: StaticLeaves>: bool =
    matches!((S::STATICS[0], D::STATICS[0]), (Some(_), Some(1)));
/// The extent of that leading mode, or one.
pub const LEADING<S: StaticLeaves, D: StaticLeaves>: isize = match (S::STATICS[0], D::STATICS[0]) {
    (Some(s), Some(1)) => s,
    _ => 1,
};

const fn min(a: isize, b: isize) -> isize {
    if a < b { a } else { b }
}

/// CuTe `max_common_vector(a, b)`: the number of leading elements that are
/// contiguous in both layouts and correspond logically.
///
/// For plain layouts this is the static extent of the leading stride-1 mode of
/// `coalesce(composition(a, right_inverse(b)))`, or one when that mode is
/// dynamic or not unit-strided; like upstream it is known from the types
/// alone. A swizzled layout caps the vector at `2^M`, its unswizzled low bits,
/// unless both sides use the same swizzle. Alignment and dynamic divisibility
/// are not considered.
///
/// ```
/// use noodle_cute::algebra::max_common_vector;
/// use noodle_cute::{Int, Layout, shape, stride};
///
/// let row_major = Layout::new(shape![Int::<4>, Int::<8>], stride![Int::<8>, Int::<1>])?;
/// let col_major = Layout::new(shape![Int::<4>, Int::<8>], stride![Int::<1>, Int::<4>])?;
/// assert_eq!(max_common_vector(&col_major, &col_major), 32);
/// assert_eq!(max_common_vector(&row_major, &col_major), 1);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
pub trait MaxCommonVector<B> {
    /// The common vector length.
    const VALUE: isize;
}

impl<S, D, S2, D2> MaxCommonVector<Layout<S2, D2>> for Layout<S, D>
where
    Layout<S2, D2>: TypedInvert,
    Self: TypedCompose<TypedInverse<Layout<S2, D2>>>,
    Composed<Self, Layout<S2, D2>>: LayoutParts,
    CShape<Self, Layout<S2, D2>>: Flatten,
    CStride<Self, Layout<S2, D2>>: Flatten,
    FlatComposed<Self, Layout<S2, D2>>: CoalesceFold,
    <FlatComposed<Self, Layout<S2, D2>> as CoalesceFold>::Output: FinishCoalesce,
    Common<Self, Layout<S2, D2>>: LayoutParts,
    CommonShape<Self, Layout<S2, D2>>: StaticLeaves,
    CommonStride<Self, Layout<S2, D2>>: StaticLeaves,
{
    const VALUE: isize =
        LEADING::<CommonShape<Self, Layout<S2, D2>>, CommonStride<Self, Layout<S2, D2>>>;
}

impl<const B: u32, const M: u32, const S: i32, O, L, S2, D2> MaxCommonVector<Layout<S2, D2>>
    for ComposedLayout<Swizzle<B, M, S>, O, L>
where
    L: MaxCommonVector<Layout<S2, D2>>,
{
    const VALUE: isize = min(L::VALUE, 1 << M);
}

impl<S, D, const B: u32, const M: u32, const SH: i32, O, L>
    MaxCommonVector<ComposedLayout<Swizzle<B, M, SH>, O, L>> for Layout<S, D>
where
    L: MaxCommonVector<Layout<S, D>>,
{
    const VALUE: isize = min(L::VALUE, 1 << M);
}

impl<
    const B0: u32,
    const M0: u32,
    const S0: i32,
    O0,
    L0,
    const B1: u32,
    const M1: u32,
    const S1: i32,
    O1,
    L1,
> MaxCommonVector<ComposedLayout<Swizzle<B1, M1, S1>, O1, L1>>
    for ComposedLayout<Swizzle<B0, M0, S0>, O0, L0>
where
    L0: MaxCommonVector<L1>,
{
    // The same swizzle on both sides cancels; otherwise both caps apply.
    const VALUE: isize = if B0 == B1 && M0 == M1 && S0 == S1 {
        L0::VALUE
    } else {
        min(min(L0::VALUE, 1 << M0), 1 << M1)
    };
}

/// Free-function form of [`MaxCommonVector`].
pub fn max_common_vector<A: MaxCommonVector<B>, B>(_: &A, _: &B) -> isize {
    A::VALUE
}

mod detail {
    use crate::Result;

    pub trait CommonKind<I, C> {
        type Output;
        fn common(inverse: I, common: C) -> Result<Self::Output>;
    }
}
use detail::CommonKind;

impl<I, C> CommonKind<I, C> for Bool<false> {
    type Output = Layout<Int<1>, Int<0>>;
    fn common(_: I, _: C) -> Result<Self::Output> {
        Ok(Layout {
            shape: Int,
            stride: Int,
        })
    }
}

type Leading<C> = Layout<
    <<C as LayoutParts>::Shape as Get<0>>::Output,
    <<C as LayoutParts>::Stride as Get<0>>::Output,
>;

impl<I, C> CommonKind<I, C> for Bool<true>
where
    C: LayoutParts,
    C::Shape: Get<0>,
    C::Stride: Get<0>,
    <C::Shape as Get<0>>::Output: Clone,
    <C::Stride as Get<0>>::Output: Clone,
    I: TypedCompose<Leading<C>>,
{
    type Output = TypedComposition<I, Leading<C>>;
    fn common(inverse: I, common: C) -> Result<Self::Output> {
        inverse.compose(common.into_layout().get::<0>())
    }
}

type LeadKind<A, B> = Bool<{ LEADS::<CommonShape<A, B>, CommonStride<A, B>> }>;

/// CuTe `max_common_layout(a, b)`: the layout `r` of the common contiguous
/// vector, so that `composition(a, r)` and `composition(b, r)` are identities.
///
/// It is `composition(right_inverse(b), layout<0>(common))` when the leading
/// mode of `common = coalesce(composition(a, right_inverse(b)))` is static and
/// unit-strided, and `1:0` otherwise. The typed algebra steps are checked.
pub trait MaxCommonLayout<B> {
    /// The common layout.
    type Output;
    /// Returns the common layout.
    fn max_common_layout(&self, b: &B) -> Result<Self::Output>;
}

impl<S, D, S2, D2> MaxCommonLayout<Layout<S2, D2>> for Layout<S, D>
where
    S: Shape,
    D: StrideOf<S>,
    S2: Shape,
    D2: StrideOf<S2>,
    Layout<S2, D2>: TypedInvert,
    TypedInverse<Layout<S2, D2>>: Clone,
    Self: TypedCompose<TypedInverse<Layout<S2, D2>>>,
    Composed<Self, Layout<S2, D2>>: LayoutParts,
    CShape<Self, Layout<S2, D2>>: Shape + Flatten,
    CStride<Self, Layout<S2, D2>>: StrideOf<CShape<Self, Layout<S2, D2>>> + Flatten,
    FlatComposed<Self, Layout<S2, D2>>: CoalesceFold,
    <FlatComposed<Self, Layout<S2, D2>> as CoalesceFold>::Output: FinishCoalesce,
    Common<Self, Layout<S2, D2>>: LayoutParts,
    CommonShape<Self, Layout<S2, D2>>: StaticLeaves,
    CommonStride<Self, Layout<S2, D2>>: StaticLeaves,
    LeadKind<Self, Layout<S2, D2>>:
        CommonKind<TypedInverse<Layout<S2, D2>>, Common<Self, Layout<S2, D2>>>,
{
    type Output = <LeadKind<Self, Layout<S2, D2>> as CommonKind<
        TypedInverse<Layout<S2, D2>>,
        Common<Self, Layout<S2, D2>>,
    >>::Output;
    fn max_common_layout(&self, b: &Layout<S2, D2>) -> Result<Self::Output> {
        let inverse = b.right_inverse()?;
        let common = self
            .clone()
            .compose(inverse.clone())?
            .into_layout()
            .coalesce()?;
        <LeadKind<Self, Layout<S2, D2>> as CommonKind<_, _>>::common(inverse, common)
    }
}

/// The underlying layout of a swizzled layout: CuTe `get_nonswizzle_portion`.
///
/// A plain layout is returned unchanged.
pub trait NonSwizzlePortion {
    /// The layout without its swizzle.
    type Output;
    /// Returns the layout without its swizzle.
    fn get_nonswizzle_portion(&self) -> Self::Output;
}

impl<S: Clone, D: Clone> NonSwizzlePortion for Layout<S, D> {
    type Output = Self;
    fn get_nonswizzle_portion(&self) -> Self {
        self.clone()
    }
}

impl<const B: u32, const M: u32, const S: i32, O, L: Clone> NonSwizzlePortion
    for ComposedLayout<Swizzle<B, M, S>, O, L>
{
    type Output = L;
    fn get_nonswizzle_portion(&self) -> L {
        self.layout_b.clone()
    }
}

mod tiling {
    /// `append<R>(block)`: pads a layout with `1:0` modes to the rank of a target shape.
    pub trait PadTo<T> {
        type Shape;
        type Stride;
        fn pad(self) -> crate::layout::Layout<Self::Shape, Self::Stride>;
    }

    /// `ceil_div` mode by mode.
    pub trait CeilDivEach<B> {
        type Output;
        fn ceil_div_each(self, b: B) -> Self::Output;
    }
}
use tiling::CeilDivEach;
pub(crate) use tiling::PadTo;

impl PadTo<Nil> for Layout<Nil, Nil> {
    type Shape = Nil;
    type Stride = Nil;
    fn pad(self) -> Self {
        self
    }
}

impl<TH, TT> PadTo<IntTuple<TH, TT>> for Layout<Nil, Nil>
where
    Layout<Nil, Nil>: PadTo<TT>,
{
    type Shape = IntTuple<Int<1>, <Layout<Nil, Nil> as PadTo<TT>>::Shape>;
    type Stride = IntTuple<Int<0>, <Layout<Nil, Nil> as PadTo<TT>>::Stride>;
    fn pad(self) -> Layout<Self::Shape, Self::Stride> {
        let tail = PadTo::<TT>::pad(self);
        Layout {
            shape: IntTuple(Int, tail.shape),
            stride: IntTuple(Int, tail.stride),
        }
    }
}

impl<SH, ST, DH, DT, TH, TT> PadTo<IntTuple<TH, TT>> for Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>
where
    Layout<ST, DT>: PadTo<TT>,
{
    type Shape = IntTuple<SH, <Layout<ST, DT> as PadTo<TT>>::Shape>;
    type Stride = IntTuple<DH, <Layout<ST, DT> as PadTo<TT>>::Stride>;
    fn pad(self) -> Layout<Self::Shape, Self::Stride> {
        let tail = PadTo::<TT>::pad(Layout {
            shape: self.shape.1,
            stride: self.stride.1,
        });
        Layout {
            shape: IntTuple(self.shape.0, tail.shape),
            stride: IntTuple(self.stride.0, tail.stride),
        }
    }
}

impl CeilDivEach<Nil> for Nil {
    type Output = Nil;
    fn ceil_div_each(self, _: Nil) -> Nil {
        Nil
    }
}

impl<AH, AT, BH, BT> CeilDivEach<IntTuple<BH, BT>> for IntTuple<AH, AT>
where
    AH: Arith<BH, { Op::CeilDiv }>,
    AT: CeilDivEach<BT>,
{
    type Output = IntTuple<CeilQuot<AH, BH>, AT::Output>;
    fn ceil_div_each(self, b: IntTuple<BH, BT>) -> Self::Output {
        IntTuple(ceil_quot(self.0, b.0), self.1.ceil_div_each(b.1))
    }
}

const fn evenly_divides(
    target: [Option<isize>; MAX_RANK],
    block: [Option<isize>; MAX_RANK],
    n: usize,
) -> bool {
    let mut i = 0;
    while i < n {
        if let (Some(t), Some(b)) = (target[i], block[i])
            && t % b != 0
        {
            return false;
        }
        i += 1;
    }
    true
}

type Wrapped<T> = <T as Wrap>::Output;
type PadShape<S, D, T> = <Layout<Wrapped<S>, Wrapped<D>> as PadTo<Wrapped<T>>>::Shape;
type PadStride<S, D, T> = <Layout<Wrapped<S>, Wrapped<D>> as PadTo<Wrapped<T>>>::Stride;
type Repeats<PS, TT> =
    <<TT as ProductEach>::Output as CeilDivEach<<PS as ProductEach>::Output>>::Output;

/// The blocked product of a padded block and its repeat counts.
pub trait TileBlocks<TT> {
    type Output;
    fn tile_blocks(self) -> Result<Self::Output>;
}

impl<PS, PD, TT> TileBlocks<TT> for Layout<PS, PD>
where
    PS: ProductEach + Clone,
    PS::Output: StaticLeaves,
    TT: ProductEach + StaticLeaves,
    TT::Output: CeilDivEach<PS::Output>,
    Repeats<PS, TT>: CompactLeft,
    Self: TypedProduct<ColMajorLayout<Repeats<PS, TT>>, { ProductKind::Blocked }>,
{
    type Output =
        TypedProductLayout<Self, ColMajorLayout<Repeats<PS, TT>>, { ProductKind::Blocked }>;
    fn tile_blocks(self) -> Result<Self::Output> {
        const {
            assert!(
                evenly_divides(
                    TT::STATICS,
                    <PS::Output as StaticLeaves>::STATICS,
                    TT::COUNT
                ),
                "tile_to_shape: block shape does not divide the target shape"
            )
        };
        self.product_typed(<ColMajorLayout<Repeats<PS, TT>> as StaticValue>::VALUE)
    }
}

/// CuTe `tile_to_shape(block, trg_shape)`: repeats `block` to fill the target shape.
///
/// The block is padded with `1:0` modes to the rank of the target, each target
/// mode is divided by the block's mode size, and the block is repeated with
/// `blocked_product` over a column-major layout of the repeat counts, upstream's
/// default `GenColMajor` order. The target and the repeat counts must be static,
/// and the target must be divisible by the block. A swizzled block keeps its
/// swizzle.
///
/// ```
/// use noodle_cute::algebra::TileToShape;
/// use noodle_cute::layout::LayoutRight;
/// use noodle_cute::{Int, shape, stride};
///
/// // An 8x8 row-major atom repeated over a 16x32 tile, column-major over atoms.
/// let atom = LayoutRight::make(shape![Int::<8>, Int::<8>])?;
/// let tile = atom.tile_to_shape(shape![Int::<16>, Int::<32>])?;
/// assert_eq!(tile.shape, shape![(Int::<8>, Int::<2>), (Int::<8>, Int::<4>)]);
/// assert_eq!(tile.stride, stride![(Int::<8>, Int::<64>), (Int::<1>, Int::<128>)]);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
pub trait TileToShape<T> {
    /// The tiled layout.
    type Output;
    /// Tiles `self` over `target`.
    fn tile_to_shape(&self, target: T) -> Result<Self::Output>;
}

impl<S, D, T> TileToShape<T> for Layout<S, D>
where
    S: Wrap + Clone,
    D: Wrap + Clone,
    T: Wrap,
    Layout<Wrapped<S>, Wrapped<D>>: PadTo<Wrapped<T>>,
    Layout<PadShape<S, D, T>, PadStride<S, D, T>>: TileBlocks<Wrapped<T>>,
{
    type Output = <Layout<PadShape<S, D, T>, PadStride<S, D, T>> as TileBlocks<Wrapped<T>>>::Output;
    fn tile_to_shape(&self, _: T) -> Result<Self::Output> {
        PadTo::<Wrapped<T>>::pad(Layout {
            shape: self.shape.clone().wrap(),
            stride: self.stride.clone().wrap(),
        })
        .tile_blocks()
    }
}

// layout_composed.hpp: tile_to_shape(ComposedLayout) tiles `layout_b`.
impl<A: Clone, O: Clone, B: TileToShape<T>, T> TileToShape<T> for ComposedLayout<A, O, B> {
    type Output = ComposedLayout<A, O, B::Output>;
    fn tile_to_shape(&self, target: T) -> Result<Self::Output> {
        self.map_b(|b| b.tile_to_shape(target))
    }
}
