//! Hierarchical CuTe [`Layout`] definitions, constructors, indexing, and non-algebra operations.
//!
//! A [`Layout`] maps coordinates in a shape domain to offsets in a stride codomain via:
//!
//! $$L(c) = \langle \mathrm{idx2crd}(c, S), D \rangle$$
//!
//! where `shape: S` implements [`Shape`] and `stride: D` implements [`Stride<S>`]
//! (enforcing structural congruence at compile time).
//!
//! # Core Types & Submodules
//!
//! - **Core types**: [`LayoutBase`], [`IsLayout`], [`is_layout`], [`Layout<S, D>`], and
//!   structural equality [`LayoutItemEq`].
//! - [`mod@tiler`]: Layout concatenation ([`Concatenation`], [`MakeLayout`], [`make_layout`]),
//!   tiler-to-layout conversion ([`Tiler`], [`tiler_to_layout`], [`tiler_to_layout_with`]),
//!   and codomain queries ([`coshape`], [`Coprofile`], [`coprofile`]).
//! - [`mod@order`]: Order-induced compact layouts ([`OrderKey`], [`MakeOrderedLayout`],
//!   [`make_ordered_layout`], [`MakeLayoutLike`], [`make_layout_like`]).
//! - [`mod@recast`]: Rational element-unit rescaling ([`Fraction`], [`RecastScale`],
//!   [`Recast`], [`recast()`], [`recast_leaf_values`]).
//! - **Macro**: [`make_layout!`](crate::make_layout!).
//!
//! # Examples
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let a = Layout::new(
//!     int_tuple![Int::<3>, (Int::<2>, Int::<4>)],
//!     int_tuple![Int::<2>, (Int::<1>, Int::<6>)],
//! );
//! assert_eq!(a(17i32), 17);
//! assert_eq!(a(2i32, 5i32), 17);
//! assert_eq!(a(2i32, int_tuple![1i32, 2i32]), 17);
//! ```
//!
//! # Notes
//!
//! - [`Layout<S, D>`] implements [`Item`] as an atomic [`Leaf`](crate::tuple::Leaf)
//!   (`RANK = 1, DEPTH = 0, LEAVES = 1`) so tuples of layouts (such as tilers) treat each
//!   layout as one leaf, whereas inherent [`Layout::rank`], [`Layout::depth`], and
//!   [`Layout::length`] query `self.shape`.
//! - A [`Layout`] is directly callable with 1 to 6 coordinate arguments via `Fn`/`FnMut`/`FnOnce`,
//!   delegating to [`Layout::call`].
//! - `PartialEq` checks structural shape and stride equality via [`LayoutItemEq`] (supporting
//!   mixed static/dynamic integers) rather than functional equivalence.

use crate::StaticValue;
use crate::basis::{
    BasisCoeff, BasisCombination, BasisPath, BasisTerm, BasisTuple, MakeBasisLikeOutput,
    MakeBasisLikeTopProfile, make_basis_like,
};
use crate::f2::{F2, F2Scalar};
use crate::integer::{Int, eq, int};
use crate::shape::{Crd2Crd, IndexToCoord, Shape, crd2crd};
use crate::slice::Slice;
use crate::stride::{
    Coshape, Idx2CrdWithStride, InnerProduct, LinearBounds, PrefixProduct, Stride, SuffixProduct,
    idx2crd_with_stride, inner_product, prefix_product, prefix_product_with, same_path,
    suffix_product, suffix_product_with,
};
use crate::tuple::{
    Append, DiceBy, Flatten, Get, GetOutput, Group, Item, LeafProfile, Nil, RepeatModes, Select,
    SliceBy, SliceRange, Til, Wrap, Zip2, slice_by,
};

pub(crate) mod alias;
pub mod order;
pub mod recast;
pub mod tiler;

pub use order::*;
pub use recast::*;
pub use tiler::*;

/// Marker/interface trait for layout types exposing a [`Shape`] and congruent [`Stride`].
pub const trait LayoutBase: Copy {
    /// Coordinate domain shape type.
    type Shape: Shape;
    /// Congruent stride type.
    type Stride: Stride<Self::Shape>;

    /// Returns a copy of the layout's shape.
    fn shape(&self) -> Self::Shape;

    /// Returns a copy of the layout's stride.
    fn stride(&self) -> Self::Stride;
}

/// Compile-time layout type predicate backing [`is_layout`].
pub const trait IsLayout: Copy {
    /// `true` iff `Self` is a CuTe layout.
    const IS_LAYOUT: bool;
}

/// Returns `true` iff `x` is a [`Layout`].
///
/// ```
/// # #![feature(const_trait_impl)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::{Layout, is_layout};
///
/// let layout = Layout::from_shape(int_tuple![Int::<4>, Int::<8>]);
/// assert!(is_layout(&layout));
/// assert!(!is_layout(&int_tuple![Int::<4>, Int::<8>]));
/// assert!(!is_layout(&42i32));
/// ```
pub const fn is_layout<T: IsLayout>(_: &T) -> bool {
    T::IS_LAYOUT
}

/// Resulting layout type of [`Layout::append`].
pub type AppendedLayout<S, D, S2, D2> = Layout<
    <<S as Wrap>::Wrapped as Append<Til<S2, Nil>>>::Output,
    <<D as Wrap>::Wrapped as Append<Til<D2, Nil>>>::Output,
>;

/// A CuTe layout mapping coordinates in `shape` to offsets via `stride`.
///
/// Note on [`Item`] vs inherent queries: `Layout<S, D>` implements [`Item`] as an atomic
/// [`Leaf`](crate::tuple::Leaf) (`RANK = 1, DEPTH = 0, LEAVES = 1`) so tuples of layouts
/// (such as tilers) treat each `Layout` as a single leaf, whereas the inherent methods
/// [`Layout::rank`], [`Layout::depth`], and [`Layout::length`] query `self.shape` directly.
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
///
/// let a = Layout::new(
///     int_tuple![Int::<3>, (Int::<2>, Int::<4>)],
///     int_tuple![Int::<2>, (Int::<1>, Int::<6>)],
/// );
/// assert_eq!(a(17i32), 17);
/// assert_eq!(a(2i32, 5i32), 17);
/// assert_eq!(a(2i32, int_tuple![1i32, 2i32]), 17);
/// ```
#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct Layout<S, D> {
    /// Non-negative extents arranged as an integer leaf, [`Nil`], or nested tuple.
    pub shape: S,
    /// Strides with the same nested profile as `shape`.
    pub stride: D,
}

const impl<S: Item, D: Item> Item for Layout<S, D> {
    const RANK: int = 1;
    const DEPTH: int = 0;
    const LEAVES: int = 1;
    const IS_STATIC: bool = S::IS_STATIC && D::IS_STATIC;
    type Profile = LeafProfile;
}

const impl<S: Shape + StaticValue, D: Stride<S> + StaticValue> StaticValue for Layout<S, D> {
    const VALUE: Self = Layout {
        shape: S::VALUE,
        stride: D::VALUE,
    };
}

const impl<S, D> Zip2 for Layout<S, D>
where
    S: [const] Zip2,
    D: [const] Zip2,
    S::Output: Shape,
    D::Output: Stride<S::Output>,
{
    type Output = Layout<S::Output, D::Output>;

    fn zip2(self) -> Self::Output {
        Layout::new(self.shape.zip2(), self.stride.zip2())
    }
}

/// Unpacks a linear index into a flat (top-level rank) coordinate of a compact layout
/// (CuTe `layout.get_flat_coord(idx)`).
pub const trait GetFlatCoord<I>: Copy {
    /// Flat coordinate type (one integer per top-level mode of `Self`).
    type Output: Copy;
    /// Evaluates `crd2crd(idx2crd_with_stride(index, shape, stride), shape, repeat_modes(1, shape))`.
    fn get_flat_coord(&self, index: I) -> Self::Output;
}

const impl<S, D, I> GetFlatCoord<I> for Layout<S, D>
where
    S: [const] Shape + [const] RepeatModes<Int<1>>,
    D: Stride<S>,
    <S as RepeatModes<Int<1>>>::Output: Shape,
    I: [const] Idx2CrdWithStride<S, D>,
    <I as Idx2CrdWithStride<S, D>>::Output: [const] Crd2Crd<S, <S as RepeatModes<Int<1>>>::Output>,
{
    type Output = <<I as Idx2CrdWithStride<S, D>>::Output as Crd2Crd<
        S,
        <S as RepeatModes<Int<1>>>::Output,
    >>::Output;

    fn get_flat_coord(&self, index: I) -> Self::Output {
        let crd = idx2crd_with_stride(index, &self.shape, &self.stride);
        let target = S::repeat_modes(Int::<1>);
        crd2crd(crd, &self.shape, &target)
    }
}

/// Unpacks `index` into a flat top-level coordinate for compact `layout` (CuTe `get_flat_coord`).
pub const fn get_flat_coord<L: [const] GetFlatCoord<I>, I>(layout: &L, index: I) -> L::Output {
    layout.get_flat_coord(index)
}

/// Constructs the identity layout mapping every coordinate in `shape` to itself via basis strides.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::basis::as_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::make_identity_layout;
/// use noodle_cute::int_tuple;
///
/// let id = make_identity_layout(int_tuple![Int::<3>, Int::<4>]);
/// assert_eq!(as_tuple(id(1i32, 2i32)), int_tuple![1i32, 2i32]);
/// ```
pub const fn make_identity_layout<S>(shape: S) -> Layout<S, MakeBasisLikeOutput<S>>
where
    S: [const] Shape,
    S::Profile: [const] MakeBasisLikeTopProfile,
    MakeBasisLikeOutput<S>: Stride<S>,
{
    Layout::new(shape, make_basis_like(shape))
}

const impl<S: Shape, D: Stride<S>> LayoutBase for Layout<S, D> {
    type Shape = S;
    type Stride = D;

    fn shape(&self) -> S {
        self.shape
    }

    fn stride(&self) -> D {
        self.stride
    }
}

const impl<S: Copy, D: Copy> IsLayout for Layout<S, D> {
    const IS_LAYOUT: bool = true;
}

const impl IsLayout for Nil {
    const IS_LAYOUT: bool = false;
}

const impl IsLayout for int {
    const IS_LAYOUT: bool = false;
}

const impl<const N: int> IsLayout for Int<N> {
    const IS_LAYOUT: bool = false;
}

const impl<V: Copy> IsLayout for F2<V> {
    const IS_LAYOUT: bool = false;
}

const impl<V: Copy, P: Copy> IsLayout for BasisTerm<V, P> {
    const IS_LAYOUT: bool = false;
}

const impl<T: BasisTuple> IsLayout for BasisCombination<T> {
    const IS_LAYOUT: bool = false;
}

const impl<H: Copy, T: Copy> IsLayout for Til<H, T> {
    const IS_LAYOUT: bool = false;
}

const impl IsLayout for core::ops::RangeFull {
    const IS_LAYOUT: bool = false;
}

/// Evaluates a coordinate (possibly containing `..` wildcards) to a codomain offset
/// and slices the layout to the modes left unnamed by `..`.
///
/// Mirrors PyCuTe's `Layout._offset_and_slice(crd)`: when `C` contains no `..`
/// wildcards, `SlicedShape` and `SlicedStride` are both [`Nil`] (rank 0).
pub const trait OffsetAndSlice<C> {
    /// Evaluated codomain offset type.
    type Offset: Copy;
    /// Shape of the residual sliced layout ([`Nil`] for complete coordinates).
    type SlicedShape: [const] Shape;
    /// Stride of the residual sliced layout ([`Nil`] for complete coordinates).
    type SlicedStride: Stride<Self::SlicedShape>;

    /// Evaluates `coord` and returns `(offset, sliced_layout)`.
    fn offset_and_slice(
        &self,
        coord: C,
    ) -> (Self::Offset, Layout<Self::SlicedShape, Self::SlicedStride>);
}

/// Output `(offset, sliced_layout)` pair produced by [`OffsetAndSlice`].
pub type OffsetAndSliceOutput<L, C> = (
    <L as OffsetAndSlice<C>>::Offset,
    Layout<<L as OffsetAndSlice<C>>::SlicedShape, <L as OffsetAndSlice<C>>::SlicedStride>,
);

const impl<S, D, C> OffsetAndSlice<C> for Layout<S, D>
where
    S: [const] Shape,
    D: [const] Stride<S>,
    C: Copy + [const] IndexToCoord<S> + [const] SliceBy<S> + [const] SliceBy<D>,
    <C as IndexToCoord<S>>::Output: [const] InnerProduct<D>,
    <C as SliceBy<S>>::Output: [const] Shape,
    <C as SliceBy<D>>::Output: Stride<<C as SliceBy<S>>::Output>,
{
    type Offset = <<C as IndexToCoord<S>>::Output as InnerProduct<D>>::Output;
    type SlicedShape = <C as SliceBy<S>>::Output;
    type SlicedStride = <C as SliceBy<D>>::Output;

    fn offset_and_slice(
        &self,
        coord: C,
    ) -> (Self::Offset, Layout<Self::SlicedShape, Self::SlicedStride>) {
        let offset = self.call(coord);
        let sliced_shape = slice_by(&coord, &self.shape);
        let sliced_stride = slice_by(&coord, &self.stride);
        (offset, Layout::new(sliced_shape, sliced_stride))
    }
}

impl<S, D> Layout<S, D> {
    /// Creates a new layout directly from a congruent `(shape, stride)` pair.
    pub const fn new(shape: S, stride: D) -> Self
    where
        S: Shape,
        D: Stride<S>,
    {
        Self { shape, stride }
    }

    /// Returns a copy of this layout's shape.
    pub const fn shape(&self) -> S
    where
        S: Copy,
    {
        self.shape
    }

    /// Returns a copy of this layout's stride.
    pub const fn stride(&self) -> D
    where
        D: Copy,
    {
        self.stride
    }

    /// Returns the total number of integral coordinates in the layout's domain.
    pub const fn size(&self) -> int
    where
        S: [const] Shape,
    {
        self.shape.size()
    }

    /// Returns `true` iff any shape leaf equals zero.
    pub const fn has_zero(&self) -> bool
    where
        S: [const] Shape,
    {
        self.shape.has_zero()
    }

    /// Returns the number of top-level modes of `self.shape`.
    pub const fn rank(&self) -> int
    where
        S: Item,
    {
        S::RANK
    }

    /// Returns the nesting depth of `self.shape`.
    pub const fn depth(&self) -> int
    where
        S: Item,
    {
        S::DEPTH
    }

    /// Returns the total number of leaves in `self.shape`.
    pub const fn length(&self) -> int
    where
        S: Item,
    {
        S::LEAVES
    }

    /// Returns `true` iff both `self.shape` and `self.stride` are compile-time static.
    pub const fn is_static(&self) -> bool
    where
        S: Item,
        D: Item,
    {
        S::IS_STATIC && D::IS_STATIC
    }

    /// Flattens nested modes in both `shape` and `stride` without merging leaves.
    pub const fn flatten(&self) -> Layout<S::Output, D::Output>
    where
        S: [const] Flatten,
        D: [const] Flatten,
    {
        Layout {
            shape: self.shape.flatten(),
            stride: self.stride.flatten(),
        }
    }

    /// Computes the codomain shape bounding all offsets produced by this layout.
    pub const fn coshape(&self) -> <S as Coshape<D>>::Output
    where
        S: [const] Coshape<D>,
        D: Stride<S>,
    {
        self.shape.coshape(self.stride)
    }

    /// Computes the exact closed `[min_offset, max_offset]` interval reachable by all
    /// in-bounds coordinates of a linear-strided layout, or `None` when the domain is empty.
    pub const fn linear_bounds(&self) -> Option<(int, int)>
    where
        S: [const] LinearBounds<D>,
        D: Stride<S>,
    {
        self.shape.linear_bounds(self.stride)
    }

    /// Computes the total size of the codomain (`size(coshape(self))`).
    pub const fn cosize(&self) -> int
    where
        S: [const] Coshape<D>,
        D: Stride<S>,
        <S as Coshape<D>>::Output: [const] Shape,
    {
        self.coshape().size()
    }

    /// Computes the hierarchical profile of the layout's codomain.
    pub const fn coprofile(&self) -> <Self as Coprofile>::Output
    where
        Self: [const] Coprofile,
    {
        Coprofile::coprofile(self)
    }

    /// Evaluates the layout at coordinate `coord` (integral, flat, or natural).
    pub const fn call<C>(
        &self,
        coord: C,
    ) -> <<C as IndexToCoord<S>>::Output as InnerProduct<D>>::Output
    where
        S: Shape,
        D: Copy,
        C: [const] IndexToCoord<S>,
        <C as IndexToCoord<S>>::Output: [const] InnerProduct<D>,
    {
        inner_product(crate::shape::idx2crd(coord, &self.shape), self.stride)
    }

    /// Evaluates `coord` (which may contain `..` wildcards) to a codomain offset and
    /// returns `(offset, sliced_layout)` retaining the modes marked with `..`.
    pub const fn offset_and_slice<C>(&self, coord: C) -> OffsetAndSliceOutput<Self, C>
    where
        Self: [const] OffsetAndSlice<C>,
    {
        OffsetAndSlice::offset_and_slice(self, coord)
    }

    /// Copies the sublayout at top-level mode `I` (supporting negative indices).
    pub const fn get<const I: int>(&self) -> GetOutput<Self, I>
    where
        Self: [const] Get<I>,
    {
        Get::<I>::get(self)
    }

    /// Extracts a contiguous range of top-level modes according to `SLICE`.
    pub const fn slice<const SLICE: Slice>(
        &self,
    ) -> Layout<<S as SliceRange<SLICE>>::Output, <D as SliceRange<SLICE>>::Output>
    where
        S: [const] SliceRange<SLICE>,
        D: [const] SliceRange<SLICE>,
    {
        Layout {
            shape: self.shape.slice_range(),
            stride: self.stride.slice_range(),
        }
    }

    /// Selects top-level modes according to type-level index path `Indices` (`mode![...]`).
    pub const fn select<Indices>(
        &self,
    ) -> Layout<<S as Select<Indices>>::Output, <D as Select<Indices>>::Output>
    where
        S: [const] Select<Indices>,
        D: [const] Select<Indices>,
    {
        Layout {
            shape: self.shape.select(),
            stride: self.stride.select(),
        }
    }

    /// Groups top-level modes in `[B, E)` into a single nested mode (wrapping scalar layouts first).
    pub const fn group<const B: int, const E: int>(
        &self,
    ) -> Layout<<S::Wrapped as Group<B, E>>::Output, <D::Wrapped as Group<B, E>>::Output>
    where
        S: [const] Wrap,
        D: [const] Wrap,
        S::Wrapped: [const] Group<B, E>,
        D::Wrapped: [const] Group<B, E>,
    {
        Layout {
            shape: self.shape.wrap().group(),
            stride: self.stride.wrap().group(),
        }
    }

    /// Appends `mode` as a new last top-level mode (CuTe `append(layout, mode)`).
    pub const fn append<S2, D2>(
        &self,
        mode: Layout<S2, D2>,
    ) -> AppendedLayout<S, D, S2, D2>
    where
        S: [const] Wrap,
        D: [const] Wrap,
        S2: [const] Item,
        D2: [const] Item,
        S::Wrapped: [const] Append<Til<S2, Nil>>,
        D::Wrapped: [const] Append<Til<D2, Nil>>,
    {
        Layout {
            shape: self.shape.wrap().append(Til(mode.shape, Nil)),
            stride: self.stride.wrap().append(Til(mode.stride, Nil)),
        }
    }

    /// Prepends `mode` as a new first top-level mode (CuTe `prepend(layout, mode)`).
    pub const fn prepend<S2, D2>(
        &self,
        mode: Layout<S2, D2>,
    ) -> Layout<Til<S2, S::Wrapped>, Til<D2, D::Wrapped>>
    where
        S: [const] Wrap,
        D: [const] Wrap,
        S2: Copy,
        D2: Copy,
    {
        Layout {
            shape: Til(mode.shape, self.shape.wrap()),
            stride: Til(mode.stride, self.stride.wrap()),
        }
    }

    /// Keeps the modes of `self` whose counterpart in `proj` is not `..` (CuTe `dice(proj, layout)`).
    pub const fn dice<P>(
        &self,
        proj: &P,
    ) -> Layout<<P as DiceBy<S>>::Output, <P as DiceBy<D>>::Output>
    where
        S: [const] Item,
        D: [const] Item,
        P: [const] DiceBy<S> + [const] DiceBy<D>,
    {
        Layout {
            shape: proj.dice_by(&self.shape),
            stride: proj.dice_by(&self.stride),
        }
    }

    /// Transposes a layout of paired modes `((a_0, b_0), ...)` into `((a_0, ...), (b_0, ...))`.
    pub const fn zip2(self) -> <Self as Zip2>::Output
    where
        Self: [const] Zip2,
    {
        Zip2::zip2(self)
    }

    /// Unpacks `index` into a flat top-level coordinate for a compact integer layout.
    pub const fn get_flat_coord<I>(&self, index: I) -> <Self as GetFlatCoord<I>>::Output
    where
        Self: [const] GetFlatCoord<I>,
    {
        GetFlatCoord::get_flat_coord(self, index)
    }

    /// Builds a compact layout with the same shape whose strides follow `self.stride`'s order.
    pub const fn make_layout_like(&self) -> <Self as MakeLayoutLike>::Output
    where
        Self: [const] MakeLayoutLike,
    {
        MakeLayoutLike::make_layout_like(self)
    }

    /// Rescales this layout's element unit by `scale`.
    pub const fn recast<Scale: [const] RecastScale>(
        &self,
        scale: Scale,
    ) -> <Self as Recast<Scale>>::Output
    where
        Self: [const] Recast<Scale>,
    {
        Recast::recast(self, scale)
    }

    /// Rescales this layout from `Old` element units to `New` element units (CuTe `recast_layout<Old, New>`).
    pub const fn recast_layout<Old, New>(
        &self,
    ) -> <Self as Recast<RecastTypeScale<Old, New>>>::Output
    where
        Self: [const] Recast<RecastTypeScale<Old, New>>,
    {
        recast_layout::<Old, New, Self>(self)
    }
}

impl<S: Shape> Layout<S, <S as PrefixProduct>::Stride>
where
    S: PrefixProduct,
    <S as PrefixProduct>::Stride: Stride<S>,
{
    /// Creates a compact column-major layout from `shape` using `prefix_product(shape)`.
    pub const fn from_shape(shape: S) -> Self
    where
        S: [const] PrefixProduct,
    {
        let stride = prefix_product(shape);
        Self::new(shape, stride)
    }
}

impl<S: Shape> Layout<S, <S as SuffixProduct>::Stride>
where
    S: SuffixProduct,
    <S as SuffixProduct>::Stride: Stride<S>,
{
    /// Creates a compact row-major layout from `shape` using `suffix_product(shape)` (CuTe `LayoutRight`).
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
    /// # #![allow(incomplete_features)]
    /// use noodle_cute::int_tuple;
    /// use noodle_cute::integer::Int;
    /// use noodle_cute::layout::Layout;
    ///
    /// let l = Layout::from_shape_right(int_tuple![Int::<3>, (Int::<2>, Int::<4>)]);
    /// assert_eq!(l.stride, int_tuple![Int::<8>, (Int::<4>, Int::<1>)]);
    /// ```
    pub const fn from_shape_right(shape: S) -> Self
    where
        S: [const] SuffixProduct,
    {
        let stride = suffix_product(shape);
        Self::new(shape, stride)
    }
}

impl<S> Layout<S, S> {
    /// Creates a column-major layout from `shape` seeded by `init` (`prefix_product_with`).
    ///
    /// Call unqualified as `Layout::from_shape_with(shape, init)` so `S` is inferred from `shape`
    /// without specifying a stride type parameter.
    pub const fn from_shape_with<Init>(
        shape: S,
        init: Init,
    ) -> Layout<S, <S as PrefixProduct<Init>>::Stride>
    where
        S: Shape + [const] PrefixProduct<Init>,
        <S as PrefixProduct<Init>>::Stride: Stride<S>,
    {
        let stride = prefix_product_with(shape, init);
        Layout::new(shape, stride)
    }

    /// Creates a row-major layout from `shape` seeded by `init` (`suffix_product_with`).
    pub const fn from_shape_right_with<Init>(
        shape: S,
        init: Init,
    ) -> Layout<S, <S as SuffixProduct<Init>>::Stride>
    where
        S: Shape + [const] SuffixProduct<Init>,
        <S as SuffixProduct<Init>>::Stride: Stride<S>,
    {
        let stride = suffix_product_with(shape, init);
        Layout::new(shape, stride)
    }
}

const impl<S, D, C> FnOnce<(C,)> for Layout<S, D>
where
    S: Shape,
    D: Copy,
    C: Copy + [const] IndexToCoord<S>,
    <C as IndexToCoord<S>>::Output: [const] InnerProduct<D>,
{
    type Output = <<C as IndexToCoord<S>>::Output as InnerProduct<D>>::Output;

    extern "rust-call" fn call_once(self, args: (C,)) -> Self::Output {
        let (c0,) = args;
        self.call(c0)
    }
}

const impl<S, D, C> FnMut<(C,)> for Layout<S, D>
where
    S: Shape,
    D: Copy,
    C: Copy + [const] IndexToCoord<S>,
    <C as IndexToCoord<S>>::Output: [const] InnerProduct<D>,
{
    extern "rust-call" fn call_mut(&mut self, args: (C,)) -> Self::Output {
        let (c0,) = args;
        self.call(c0)
    }
}

const impl<S, D, C> Fn<(C,)> for Layout<S, D>
where
    S: Shape,
    D: Copy,
    C: Copy + [const] IndexToCoord<S>,
    <C as IndexToCoord<S>>::Output: [const] InnerProduct<D>,
{
    extern "rust-call" fn call(&self, args: (C,)) -> Self::Output {
        let (c0,) = args;
        self.call(c0)
    }
}

macro_rules! impl_layout_fn_multi {
    ($($param:ident : $idx:tt),+) => {
        impl_layout_fn_multi!(@build [$($param : $idx),+] []);
    };
    (@build [$param:ident : $idx:tt] [$($rev:ident : $ridx:tt),*]) => {
        impl_layout_fn_multi!(@emit [$param : $idx $(, $rev : $ridx)*]);
    };
    (@build [$head:ident : $hidx:tt, $($tail:ident : $tidx:tt),+] [$($rev:ident : $ridx:tt),*]) => {
        impl_layout_fn_multi!(@build [$($tail : $tidx),+] [$head : $hidx $(, $rev : $ridx)*]);
    };
    (@emit [$($rparam:ident : $ridx:tt),+]) => {
        impl_layout_fn_multi!(@nested_ty [$($rparam),+] Nil => @final [$($rparam : $ridx),+]);
    };
    (@nested_ty [$head:ident] $acc:ty => @final [$($rparam:ident : $ridx:tt),+]) => {
        impl_layout_fn_multi!(@impl_with Til<$head, $acc>; [$($rparam : $ridx),+]);
    };
    (@nested_ty [$head:ident, $($tail:ident),+] $acc:ty => @final [$($rparam:ident : $ridx:tt),+]) => {
        impl_layout_fn_multi!(@nested_ty [$($tail),+] Til<$head, $acc> => @final [$($rparam : $ridx),+]);
    };
    (@impl_with $CoordTy:ty; [$($rparam:ident : $ridx:tt),+]) => {
        impl_layout_fn_multi!(@reverse_back [$($rparam : $ridx),+] [] $CoordTy);
    };
    (@reverse_back [$last:ident : $lidx:tt] [$($fwd:ident : $fidx:tt),*] $CoordTy:ty) => {
        impl_layout_fn_multi!(@emit_impl [$last : $lidx $(, $fwd : $fidx)*] $CoordTy);
    };
    (@reverse_back [$head:ident : $hidx:tt, $($tail:ident : $tidx:tt),+] [$($fwd:ident : $fidx:tt),*] $CoordTy:ty) => {
        impl_layout_fn_multi!(@reverse_back [$($tail : $tidx),+] [$head : $hidx $(, $fwd : $fidx)*] $CoordTy);
    };
    (@emit_impl [$($param:ident : $idx:tt),+] $CoordTy:ty) => {
        const impl<S, D, $($param),+> FnOnce<($($param,)+)> for Layout<S, D>
        where
            S: Shape,
            D: Copy,
            $($param: Item,)+
            $CoordTy: [const] IndexToCoord<S>,
            <$CoordTy as IndexToCoord<S>>::Output: [const] InnerProduct<D>,
        {
            type Output = <<$CoordTy as IndexToCoord<S>>::Output as InnerProduct<D>>::Output;

            extern "rust-call" fn call_once(self, args: ($($param,)+)) -> Self::Output {
                let crd: $CoordTy = impl_layout_fn_multi!(@pack args; $($idx),+);
                self.call(crd)
            }
        }

        const impl<S, D, $($param),+> FnMut<($($param,)+)> for Layout<S, D>
        where
            S: Shape,
            D: Copy,
            $($param: Item,)+
            $CoordTy: [const] IndexToCoord<S>,
            <$CoordTy as IndexToCoord<S>>::Output: [const] InnerProduct<D>,
        {
            extern "rust-call" fn call_mut(&mut self, args: ($($param,)+)) -> Self::Output {
                let crd: $CoordTy = impl_layout_fn_multi!(@pack args; $($idx),+);
                self.call(crd)
            }
        }

        const impl<S, D, $($param),+> Fn<($($param,)+)> for Layout<S, D>
        where
            S: Shape,
            D: Copy,
            $($param: Item,)+
            $CoordTy: [const] IndexToCoord<S>,
            <$CoordTy as IndexToCoord<S>>::Output: [const] InnerProduct<D>,
        {
            extern "rust-call" fn call(&self, args: ($($param,)+)) -> Self::Output {
                let crd: $CoordTy = impl_layout_fn_multi!(@pack args; $($idx),+);
                self.call(crd)
            }
        }
    };
    (@pack $args:ident; $head:tt) => {
        Til($args.$head, Nil)
    };
    (@pack $args:ident; $head:tt, $($tail:tt),+) => {
        Til($args.$head, impl_layout_fn_multi!(@pack $args; $($tail),+))
    };
}

impl_layout_fn_multi!(C0: 0, C1: 1);
impl_layout_fn_multi!(C0: 0, C1: 1, C2: 2);
impl_layout_fn_multi!(C0: 0, C1: 1, C2: 2, C3: 3);
impl_layout_fn_multi!(C0: 0, C1: 1, C2: 2, C3: 3, C4: 4);
impl_layout_fn_multi!(C0: 0, C1: 1, C2: 2, C3: 3, C4: 4, C5: 5);

const impl<S, D, const I: int> Get<I> for Layout<S, D>
where
    S: Shape + [const] Wrap,
    D: Stride<S> + [const] Wrap,
    S::Wrapped: [const] Get<I>,
    D::Wrapped: [const] Get<I>,
    Layout<GetOutput<S::Wrapped, I>, GetOutput<D::Wrapped, I>>: [const] Item,
{
    type Output = Layout<GetOutput<S::Wrapped, I>, GetOutput<D::Wrapped, I>>;

    fn get(&self) -> Self::Output {
        Layout {
            shape: self.shape.wrap().get(),
            stride: self.stride.wrap().get(),
        }
    }
}

/// Structural equality across hierarchical tuple shapes and strides, supporting
/// heterogeneous scalar types and differing tuple structures. Two scalar leaves of
/// integer or [`F2`] type compare by their numeric values.
pub const trait LayoutItemEq<Rhs> {
    /// Returns `true` iff `self` and `rhs` are structurally identical with equal leaves.
    fn layout_item_eq(&self, rhs: &Rhs) -> bool;
}

const impl LayoutItemEq<Nil> for Nil {
    fn layout_item_eq(&self, _: &Nil) -> bool {
        true
    }
}

const impl<H1, T1, H2, T2> LayoutItemEq<Til<H2, T2>> for Til<H1, T1>
where
    H1: [const] LayoutItemEq<H2>,
    T1: [const] LayoutItemEq<T2>,
{
    fn layout_item_eq(&self, rhs: &Til<H2, T2>) -> bool {
        self.0.layout_item_eq(&rhs.0) && self.1.layout_item_eq(&rhs.1)
    }
}

const impl<V1, P1, V2, P2> LayoutItemEq<BasisTerm<V2, P2>> for BasisTerm<V1, P1>
where
    V1: [const] crate::basis::BasisCoeffEq<V2>,
    V2: BasisCoeff,
    P1: [const] crate::basis::ComparePath<P2>,
    P2: BasisPath,
{
    fn layout_item_eq(&self, rhs: &BasisTerm<V2, P2>) -> bool {
        same_path::<P1, P2>() && self.value.coeff_eq(&rhs.value)
    }
}

const impl<T1, T2> LayoutItemEq<BasisCombination<T2>> for BasisCombination<T1>
where
    T1: BasisTuple + [const] LayoutItemEq<T2>,
    T2: BasisTuple,
{
    fn layout_item_eq(&self, rhs: &BasisCombination<T2>) -> bool {
        self.terms().layout_item_eq(&rhs.terms())
    }
}

macro_rules! impl_scalar_eq_pair {
    ([$($lg:tt)*] $L:ty, [$($rg:tt)*] $R:ty) => {
        const impl<$($lg)* $($rg)*> LayoutItemEq<$R> for $L {
            fn layout_item_eq(&self, rhs: &$R) -> bool {
                eq(*self, *rhs)
            }
        }
    };
}

macro_rules! impl_scalar_eq_matrix {
    ($L:ty, [$($lg:tt)*]) => {
        impl_scalar_eq_pair!([$($lg)*] $L, [] int);
        impl_scalar_eq_pair!([$($lg)*] $L, [const R: int,] Int<R>);
    };
}

impl_scalar_eq_matrix!(int, []);
impl_scalar_eq_matrix!(Int<L>, [const L: int,]);

const impl<L: [const] F2Scalar, R: [const] F2Scalar> LayoutItemEq<F2<R>> for F2<L> {
    fn layout_item_eq(&self, rhs: &F2<R>) -> bool {
        eq(self.inner(), rhs.inner())
    }
}

const impl<V: [const] F2Scalar, const N: int> LayoutItemEq<Int<N>> for F2<V> {
    fn layout_item_eq(&self, rhs: &Int<N>) -> bool {
        eq(self.inner(), *rhs)
    }
}

const impl<const N: int, V: [const] F2Scalar> LayoutItemEq<F2<V>> for Int<N> {
    fn layout_item_eq(&self, rhs: &F2<V>) -> bool {
        eq(*self, rhs.inner())
    }
}

const impl<V: [const] F2Scalar> LayoutItemEq<int> for F2<V> {
    fn layout_item_eq(&self, rhs: &int) -> bool {
        eq(self.inner(), *rhs)
    }
}

const impl<V: [const] F2Scalar> LayoutItemEq<F2<V>> for int {
    fn layout_item_eq(&self, rhs: &F2<V>) -> bool {
        eq(*self, rhs.inner())
    }
}

macro_rules! impl_false_eq {
    ([$($g:tt)*] $L:ty, $R:ty) => {
        const impl<$($g)*> LayoutItemEq<$R> for $L {
            fn layout_item_eq(&self, _: &$R) -> bool {
                false
            }
        }
    };
}

macro_rules! impl_false_both {
    ([$($g:tt)*] $A:ty, $B:ty) => {
        impl_false_eq!([$($g)*] $A, $B);
        impl_false_eq!([$($g)*] $B, $A);
    };
}

impl_false_both!([H, T] Nil, Til<H, T>);
impl_false_both!([V, P] Nil, BasisTerm<V, P>);
impl_false_both!([T: BasisTuple] Nil, BasisCombination<T>);
impl_false_both!([H, T, V, P] Til<H, T>, BasisTerm<V, P>);
impl_false_both!([H, T, B: BasisTuple] Til<H, T>, BasisCombination<B>);
impl_false_both!([V, P, B: BasisTuple] BasisTerm<V, P>, BasisCombination<B>);

macro_rules! impl_false_with_scalars {
    ([$($g:tt)*] $Other:ty) => {
        impl_false_both!([$($g)*] $Other, int);
        impl_false_both!([$($g)* const N: int] $Other, Int<N>);
        impl_false_both!([$($g)* F: [const] F2Scalar] $Other, F2<F>);
    };
}

impl_false_with_scalars!([] Nil);
impl_false_with_scalars!([H, T,] Til<H, T>);
impl_false_with_scalars!([V, P,] BasisTerm<V, P>);

macro_rules! impl_non_empty_comb_false_with_scalar {
    ([$($sg:tt)*] $S:ty) => {
        const impl<H, T, $($sg)*> LayoutItemEq<$S> for BasisCombination<Til<H, T>>
        where
            Til<H, T>: BasisTuple,
        {
            fn layout_item_eq(&self, _: &$S) -> bool {
                false
            }
        }
        const impl<H, T, $($sg)*> LayoutItemEq<BasisCombination<Til<H, T>>> for $S
        where
            Til<H, T>: BasisTuple,
        {
            fn layout_item_eq(&self, _: &BasisCombination<Til<H, T>>) -> bool {
                false
            }
        }
    };
}

impl_non_empty_comb_false_with_scalar!([] int);
impl_non_empty_comb_false_with_scalar!([const N: int] Int<N>);
impl_non_empty_comb_false_with_scalar!([V: [const] F2Scalar] F2<V>);

macro_rules! impl_empty_comb_eq_scalar {
    ([$($sg:tt)*] $S:ty) => {
        const impl<$($sg)*> LayoutItemEq<$S> for BasisCombination<Nil> {
            fn layout_item_eq(&self, rhs: &$S) -> bool {
                crate::basis::BasisCoeffEq::coeff_eq(rhs, &Int::<0>)
            }
        }
        const impl<$($sg)*> LayoutItemEq<BasisCombination<Nil>> for $S {
            fn layout_item_eq(&self, _: &BasisCombination<Nil>) -> bool {
                crate::basis::BasisCoeffEq::coeff_eq(self, &Int::<0>)
            }
        }
    };
}

impl_empty_comb_eq_scalar!([V: [const] F2Scalar] F2<V>);
impl_empty_comb_eq_scalar!([] int);
impl_empty_comb_eq_scalar!([const N: int] Int<N>);

const impl<S1, D1, S2, D2> PartialEq<Layout<S2, D2>> for Layout<S1, D1>
where
    S1: [const] LayoutItemEq<S2>,
    D1: [const] LayoutItemEq<D2>,
{
    fn eq(&self, other: &Layout<S2, D2>) -> bool {
        self.shape.layout_item_eq(&other.shape) && self.stride.layout_item_eq(&other.stride)
    }
}

const impl<S, D> Eq for Layout<S, D>
where
    S: [const] LayoutItemEq<S>,
    D: [const] LayoutItemEq<D>,
{
}

impl<S: core::fmt::Display, D: core::fmt::Display> core::fmt::Display for Layout<S, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}:{}", self.shape, self.stride)
    }
}

/// Concatenates zero or more [`Layout`] arguments so each becomes one top-level mode.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
/// use noodle_cute::{int_tuple, make_layout};
///
/// let a = Layout::new(Int::<3>, Int::<1>);
/// let b = Layout::new(Int::<4>, Int::<3>);
/// assert_eq!(
///     make_layout!(a, b),
///     Layout::new(int_tuple![Int::<3>, Int::<4>], int_tuple![Int::<1>, Int::<3>])
/// );
/// ```
#[macro_export]
macro_rules! make_layout {
    () => {
        $crate::layout::make_layout($crate::tuple::Nil)
    };
    ($($item:expr),+ $(,)?) => {
        $crate::layout::make_layout($crate::tuple!($($item),+))
    };
}
