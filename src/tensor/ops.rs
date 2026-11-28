//! Tensor indexing, slicing, layout-algebra pass-through, [`domain_offset`], and partitioning.
//!
//! # Main Interfaces
//!
//! - **Indexing & slicing**: [`TensorIndex`] (`Tensor::at` and `const` `Fn` call syntax for
//!   1–6 arguments) and [`TensorSet`] (`Tensor::set`).
//! - **Layout-algebra pass-through**: `Tensor` implementations of
//!   [`CoalesceTarget`], [`CoalesceExtended`], [`Composition`], [`ComposeTile`],
//!   [`LogicalDivide`], [`LogicalDivideAt`], and [`IntoGcdShape`] (with [`ZippedDivide`],
//!   [`ZippedDivideAt`], [`TiledDivide`], and
//!   [`FlatDivide`] inherited automatically via blanket impls).
//! - **Origin shift**: [`domain_offset`] / [`Tensor::domain_offset`].
//! - **Partitioning & tiling**:
//!   - [`local_tile`] / [`Tensor::local_tile`] (`zipped_divide(tensor, tiler).at(tuple!(.., coord))`,
//!     keeping mode 0 grouped as `Tile` per PyCuTe);
//!   - [`inner_partition`] / [`Tensor::inner_partition`] (unpacks `Tile` modes and pads shorter
//!     tuple `coord` with `..` per CuTe `tensor_impl.hpp`);
//!   - [`outer_partition`] / [`Tensor::outer_partition`] (slices `Tile` at `coord` and unpacks
//!     `Rest` modes);
//!   - [`local_tile_proj`] / [`Tensor::local_tile_proj`] (`dice_by` projection + [`inner_partition`]);
//!   - [`local_partition`] / [`Tensor::local_partition`] and [`local_partition_proj`] /
//!     [`Tensor::local_partition_proj`] (thread-layout partitioning via [`outer_partition`]).
//!
//! # Notes
//!
//! - Complete coordinates (no `..` wildcards) slice the layout to `Layout<Nil, Nil>` (rank 0)
//!   and dereference the accessor; coordinates containing `..` produce a rank-`> 0` sliced
//!   layout and return a sub-[`Tensor`] sharing the underlying storage.
//! - Calling [`Tensor::set`] with a partial coordinate (containing `..`) or on a read-only
//!   accessor fails at compile time.

use core::ops::RangeFull;

use crate::algebra::coalesce::{CoalesceExtended, CoalesceProfile, CoalesceTarget};
use crate::algebra::composition::{ComposeTile, Composition};
use crate::algebra::greatest_common_domain::IntoGcdShape;
use crate::algebra::logical_divide::{
    FlatDivide, LogicalDivide, LogicalDivideAt, RegroupFlat, RegroupTiled, TiledDivide,
    ZippedDivide, ZippedDivideAt,
};
use crate::layout::{GetFlatCoord, Layout, OffsetAndSlice};
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tuple::{
    DiceBy, Get, GetOutput, Item, Nil, PadCoord, ProductEach, RepeatModes, Til, Tuple,
    WeaklyCongruent, Wrap, dice_by, product_each,
};

use super::Tensor;
use super::accessor::{Accessor, AccessorOffset, MutableAccessor};

/// Compile-time dispatch on the residual sliced layout from [`OffsetAndSlice`]:
/// - `Layout<Nil, Nil>` (rank 0, complete coordinate) dereferences the accessor.
/// - `Layout<Til<SH, ST>, Til<DH, DT>>` (rank > 0, partial coordinate)
///   shifts the accessor and returns a sub-[`Tensor`].
#[doc(hidden)]
pub const trait TensorIndexDispatch<A, Off> {
    type Output;

    fn index_dispatch(self, accessor: &A, offset: Off) -> Self::Output;
}

const impl<A, Off> TensorIndexDispatch<A, Off> for Layout<Nil, Nil>
where
    A: [const] Accessor<Off>,
{
    type Output = A::Item;

    fn index_dispatch(self, accessor: &A, offset: Off) -> Self::Output {
        accessor.load(offset)
    }
}

const impl<A, Off, SH, ST, DH, DT> TensorIndexDispatch<A, Off> for Layout<Til<SH, ST>, Til<DH, DT>>
where
    A: [const] AccessorOffset<Off>,
    SH: Shape,
    ST: Tuple + Shape,
    DH: Stride<SH>,
    DT: Tuple + Stride<ST>,
{
    type Output = Tensor<A::Output, Layout<Til<SH, ST>, Til<DH, DT>>>;

    fn index_dispatch(self, accessor: &A, offset: Off) -> Self::Output {
        Tensor {
            accessor: accessor.offset(offset),
            layout: self,
        }
    }
}

/// Unified read/slice indexing on a [`Tensor`] with coordinate `C`.
///
/// Mirrors PyCuTe's `Tensor.__getitem__`: a complete coordinate evaluates to the element
/// value, while a coordinate containing one or more `..` wildcards evaluates to a
/// sub-[`Tensor`] over the unnamed modes.
pub const trait TensorIndex<C> {
    /// Element value (for complete coordinates) or sub-[`Tensor`] (for slicing coordinates).
    type Output;

    /// Evaluates or slices `self` at `coord`.
    fn tensor_at(&self, coord: C) -> Self::Output;
}

const impl<A, S, D, C> TensorIndex<C> for Tensor<A, Layout<S, D>>
where
    S: [const] Shape,
    D: [const] Stride<S>,
    Layout<S, D>: [const] OffsetAndSlice<C>,
    Layout<
        <Layout<S, D> as OffsetAndSlice<C>>::SlicedShape,
        <Layout<S, D> as OffsetAndSlice<C>>::SlicedStride,
    >: [const] TensorIndexDispatch<A, <Layout<S, D> as OffsetAndSlice<C>>::Offset>,
{
    type Output = <Layout<
        <Layout<S, D> as OffsetAndSlice<C>>::SlicedShape,
        <Layout<S, D> as OffsetAndSlice<C>>::SlicedStride,
    > as TensorIndexDispatch<A, <Layout<S, D> as OffsetAndSlice<C>>::Offset>>::Output;

    fn tensor_at(&self, coord: C) -> Self::Output {
        let (offset, sliced) = self.layout.offset_and_slice(coord);
        sliced.index_dispatch(&self.accessor, offset)
    }
}

/// Element write on a mutable [`Tensor`] requiring a complete coordinate `C`.
///
/// Mirrors PyCuTe's `Tensor.__setitem__`: incomplete coordinates (containing `..`) fail at
/// compile time because their residual sliced layout is not `Layout<Nil, Nil>`.
pub const trait TensorSet<C> {
    /// Element type written to the backing accessor.
    type Item;

    /// Writes `value` at complete coordinate `coord`.
    fn tensor_set(&mut self, coord: C, value: Self::Item);
}

const impl<A, S, D, C> TensorSet<C> for Tensor<A, Layout<S, D>>
where
    S: [const] Shape,
    D: [const] Stride<S>,
    Layout<S, D>: [const] OffsetAndSlice<C, SlicedShape = Nil, SlicedStride = Nil>,
    A: [const] MutableAccessor<<Layout<S, D> as OffsetAndSlice<C>>::Offset>,
{
    type Item = A::Item;

    fn tensor_set(&mut self, coord: C, value: Self::Item) {
        let (offset, _) = self.layout.offset_and_slice(coord);
        self.accessor.store(offset, value);
    }
}

// ------------------------------------------------------------------
// Callable const Fn/FnMut/FnOnce on Tensor (1 to 6 coordinate arguments)
// ------------------------------------------------------------------

const impl<A, S, D, C> FnOnce<(C,)> for Tensor<A, Layout<S, D>>
where
    A: Copy,
    S: Copy,
    D: Copy,
    C: Copy,
    Self: [const] TensorIndex<C>,
{
    type Output = <Self as TensorIndex<C>>::Output;

    extern "rust-call" fn call_once(self, args: (C,)) -> Self::Output {
        let (c0,) = args;
        self.tensor_at(c0)
    }
}

const impl<A, S, D, C> FnMut<(C,)> for Tensor<A, Layout<S, D>>
where
    A: Copy,
    S: Copy,
    D: Copy,
    C: Copy,
    Self: [const] TensorIndex<C>,
{
    extern "rust-call" fn call_mut(&mut self, args: (C,)) -> Self::Output {
        let (c0,) = args;
        self.tensor_at(c0)
    }
}

const impl<A, S, D, C> Fn<(C,)> for Tensor<A, Layout<S, D>>
where
    A: Copy,
    S: Copy,
    D: Copy,
    C: Copy,
    Self: [const] TensorIndex<C>,
{
    extern "rust-call" fn call(&self, args: (C,)) -> Self::Output {
        let (c0,) = args;
        self.tensor_at(c0)
    }
}

macro_rules! impl_tensor_fn_multi {
    ($($param:ident : $idx:tt),+) => {
        impl_tensor_fn_multi!(@build [$($param : $idx),+] []);
    };
    (@build [$param:ident : $idx:tt] [$($rev:ident : $ridx:tt),*]) => {
        impl_tensor_fn_multi!(@emit [$param : $idx $(, $rev : $ridx)*]);
    };
    (@build [$head:ident : $hidx:tt, $($tail:ident : $tidx:tt),+] [$($rev:ident : $ridx:tt),*]) => {
        impl_tensor_fn_multi!(@build [$($tail : $tidx),+] [$head : $hidx $(, $rev : $ridx)*]);
    };
    (@emit [$($rparam:ident : $ridx:tt),+]) => {
        impl_tensor_fn_multi!(@nested_ty [$($rparam),+] Nil => @final [$($rparam : $ridx),+]);
    };
    (@nested_ty [$head:ident] $acc:ty => @final [$($rparam:ident : $ridx:tt),+]) => {
        impl_tensor_fn_multi!(@impl_with Til<$head, $acc>; [$($rparam : $ridx),+]);
    };
    (@nested_ty [$head:ident, $($tail:ident),+] $acc:ty => @final [$($rparam:ident : $ridx:tt),+]) => {
        impl_tensor_fn_multi!(@nested_ty [$($tail),+] Til<$head, $acc> => @final [$($rparam : $ridx),+]);
    };
    (@impl_with $CoordTy:ty; [$($rparam:ident : $ridx:tt),+]) => {
        impl_tensor_fn_multi!(@reverse_back [$($rparam : $ridx),+] [] $CoordTy);
    };
    (@reverse_back [$last:ident : $lidx:tt] [$($fwd:ident : $fidx:tt),*] $CoordTy:ty) => {
        impl_tensor_fn_multi!(@emit_impl [$last : $lidx $(, $fwd : $fidx)*] $CoordTy);
    };
    (@reverse_back [$head:ident : $hidx:tt, $($tail:ident : $tidx:tt),+] [$($fwd:ident : $fidx:tt),*] $CoordTy:ty) => {
        impl_tensor_fn_multi!(@reverse_back [$($tail : $tidx),+] [$head : $hidx $(, $fwd : $fidx)*] $CoordTy);
    };
    (@emit_impl [$($param:ident : $idx:tt),+] $CoordTy:ty) => {
        const impl<A, S, D, $($param),+> FnOnce<($($param,)+)> for Tensor<A, Layout<S, D>>
        where
            A: Copy,
            S: Copy,
            D: Copy,
            $($param: Item,)+
            Self: [const] TensorIndex<$CoordTy>,
        {
            type Output = <Self as TensorIndex<$CoordTy>>::Output;

            extern "rust-call" fn call_once(self, args: ($($param,)+)) -> Self::Output {
                let crd: $CoordTy = impl_tensor_fn_multi!(@pack args; $($idx),+);
                self.tensor_at(crd)
            }
        }

        const impl<A, S, D, $($param),+> FnMut<($($param,)+)> for Tensor<A, Layout<S, D>>
        where
            A: Copy,
            S: Copy,
            D: Copy,
            $($param: Item,)+
            Self: [const] TensorIndex<$CoordTy>,
        {
            extern "rust-call" fn call_mut(&mut self, args: ($($param,)+)) -> Self::Output {
                let crd: $CoordTy = impl_tensor_fn_multi!(@pack args; $($idx),+);
                self.tensor_at(crd)
            }
        }

        const impl<A, S, D, $($param),+> Fn<($($param,)+)> for Tensor<A, Layout<S, D>>
        where
            A: Copy,
            S: Copy,
            D: Copy,
            $($param: Item,)+
            Self: [const] TensorIndex<$CoordTy>,
        {
            extern "rust-call" fn call(&self, args: ($($param,)+)) -> Self::Output {
                let crd: $CoordTy = impl_tensor_fn_multi!(@pack args; $($idx),+);
                self.tensor_at(crd)
            }
        }
    };
    (@pack $args:ident; $head:tt) => {
        Til($args.$head, Nil)
    };
    (@pack $args:ident; $head:tt, $($tail:tt),+) => {
        Til($args.$head, impl_tensor_fn_multi!(@pack $args; $($tail),+))
    };
}

impl_tensor_fn_multi!(C0: 0, C1: 1);
impl_tensor_fn_multi!(C0: 0, C1: 1, C2: 2);
impl_tensor_fn_multi!(C0: 0, C1: 1, C2: 2, C3: 3);
impl_tensor_fn_multi!(C0: 0, C1: 1, C2: 2, C3: 3, C4: 4);
impl_tensor_fn_multi!(C0: 0, C1: 1, C2: 2, C3: 3, C4: 4, C5: 5);

// ------------------------------------------------------------------
// Layout-algebra pass-through for Tensor
// ------------------------------------------------------------------

const impl<A: Copy, S: Copy, D: Copy, const EXTENDED: bool> CoalesceTarget<EXTENDED>
    for Tensor<A, Layout<S, D>>
where
    Layout<S, D>: [const] CoalesceTarget<EXTENDED>,
{
    type Output = Tensor<A, <Layout<S, D> as CoalesceTarget<EXTENDED>>::Output>;

    fn coalesce_target(self) -> Self::Output {
        Tensor {
            accessor: self.accessor,
            layout: self.layout.coalesce_target(),
        }
    }
}

const impl<A: Copy, S: Copy, D: Copy, P> CoalesceProfile<P> for Tensor<A, Layout<S, D>>
where
    Layout<S, D>: [const] CoalesceProfile<P>,
{
    type Output = Tensor<A, <Layout<S, D> as CoalesceProfile<P>>::Output>;

    fn coalesce_with_profile(self) -> Self::Output {
        Tensor {
            accessor: self.accessor,
            layout: self.layout.coalesce_with_profile(),
        }
    }
}

const impl<A: Copy, S: Copy, D: Copy, P> CoalesceExtended<P> for Tensor<A, Layout<S, D>>
where
    Layout<S, D>: [const] CoalesceExtended<P>,
{
    type Output = Tensor<A, <Layout<S, D> as CoalesceExtended<P>>::Output>;

    fn coalesce_z(self) -> Self::Output {
        Tensor {
            accessor: self.accessor,
            layout: self.layout.coalesce_z(),
        }
    }
}

const impl<A: Copy, S: Copy, D: Copy, B> Composition<B> for Tensor<A, Layout<S, D>>
where
    Layout<S, D>: [const] Composition<B>,
{
    type Output = Tensor<A, <Layout<S, D> as Composition<B>>::Output>;

    fn composition(self, b: B) -> Self::Output {
        Tensor {
            accessor: self.accessor,
            layout: self.layout.composition(b),
        }
    }
}

const impl<A: Copy, S: Copy, D: Copy, T> ComposeTile<T> for Tensor<A, Layout<S, D>>
where
    Layout<S, D>: [const] ComposeTile<T>,
{
    type Output = Tensor<A, <Layout<S, D> as ComposeTile<T>>::Output>;

    fn compose_tile(self, tile: T) -> Self::Output {
        Tensor {
            accessor: self.accessor,
            layout: self.layout.compose_tile(tile),
        }
    }
}

const impl<A: Copy, S: Copy, D: Copy> RegroupTiled for Tensor<A, Layout<S, D>>
where
    Layout<S, D>: [const] RegroupTiled,
{
    type Output = Tensor<A, <Layout<S, D> as RegroupTiled>::Output>;

    fn regroup_tiled(self) -> Self::Output {
        Tensor {
            accessor: self.accessor,
            layout: self.layout.regroup_tiled(),
        }
    }
}

const impl<A: Copy, S: Copy, D: Copy> RegroupFlat for Tensor<A, Layout<S, D>>
where
    Layout<S, D>: [const] RegroupFlat,
{
    type Output = Tensor<A, <Layout<S, D> as RegroupFlat>::Output>;

    fn regroup_flat(self) -> Self::Output {
        Tensor {
            accessor: self.accessor,
            layout: self.layout.regroup_flat(),
        }
    }
}

const impl<A: Copy, S: Copy, D: Copy, B> LogicalDivide<B> for Tensor<A, Layout<S, D>>
where
    Layout<S, D>: [const] LogicalDivide<B>,
{
    type Output = Tensor<A, <Layout<S, D> as LogicalDivide<B>>::Output>;

    fn logical_divide(self, b: B) -> Self::Output {
        Tensor {
            accessor: self.accessor,
            layout: self.layout.logical_divide(b),
        }
    }
}

const impl<A: Copy, S: Copy, D: Copy, B, Path> LogicalDivideAt<B, Path> for Tensor<A, Layout<S, D>>
where
    Layout<S, D>: [const] LogicalDivideAt<B, Path>,
{
    type Output = Tensor<A, <Layout<S, D> as LogicalDivideAt<B, Path>>::Output>;

    fn logical_divide_at(self, b: B) -> Self::Output {
        Tensor {
            accessor: self.accessor,
            layout: self.layout.logical_divide_at(b),
        }
    }
}

const impl<A: Copy, S: [const] Shape, D: Stride<S>> IntoGcdShape for Tensor<A, Layout<S, D>> {
    type Shape = S;

    fn into_gcd_shape(self) -> S {
        self.layout.shape
    }
}

const impl<A, S: [const] Shape, D: Stride<S>> IntoGcdShape for &Tensor<A, Layout<S, D>> {
    type Shape = S;

    fn into_gcd_shape(self) -> S {
        self.layout.shape
    }
}

// ------------------------------------------------------------------
// Inherent Tensor indexing and algebra methods
// ------------------------------------------------------------------

impl<A, L> Tensor<A, L> {
    /// Reads the element at complete coordinate `coord`, or returns the sub-[`Tensor`]
    /// over the modes marked with `..` in `coord`.
    pub const fn at<C>(&self, coord: C) -> <Self as TensorIndex<C>>::Output
    where
        Self: [const] TensorIndex<C>,
    {
        self.tensor_at(coord)
    }

    /// Writes `value` at complete coordinate `coord`.
    ///
    /// Incomplete coordinates containing `..` and read-only accessors are rejected at compile time.
    pub const fn set<C>(&mut self, coord: C, value: <Self as TensorSet<C>>::Item)
    where
        Self: [const] TensorSet<C>,
    {
        self.tensor_set(coord, value);
    }

    /// Coalesces the tensor's layout while keeping its accessor unchanged.
    pub const fn coalesce(self) -> <Self as CoalesceTarget<false>>::Output
    where
        Self: [const] CoalesceTarget<false>,
    {
        self.coalesce_target()
    }

    /// Coalesces the tensor's layout over its extended domain while keeping its accessor unchanged.
    pub const fn coalesce_extended(self) -> <Self as CoalesceTarget<true>>::Output
    where
        Self: [const] CoalesceTarget<true>,
    {
        self.coalesce_target()
    }

    /// Coalesces the tensor's layout mode-by-mode according to profile `P` (dropping
    /// trailing unit modes inside each `LeafProfile` subtree).
    pub const fn coalesce_with_profile<P>(self) -> <Self as CoalesceProfile<P>>::Output
    where
        Self: [const] CoalesceProfile<P>,
    {
        CoalesceProfile::<P>::coalesce_with_profile(self)
    }

    /// Profile-guided extended-domain coalescing (`coalesce_z`) of the tensor's layout.
    pub const fn coalesce_z<P>(self) -> <Self as CoalesceExtended<P>>::Output
    where
        Self: [const] CoalesceExtended<P>,
    {
        CoalesceExtended::<P>::coalesce_z(self)
    }

    /// Composes the tensor's layout with `b` while keeping its accessor unchanged.
    pub const fn composition<B>(self, b: B) -> <Self as Composition<B>>::Output
    where
        Self: [const] Composition<B>,
    {
        Composition::composition(self, b)
    }

    /// Composes the tensor's layout mode by mode with tuple tile `tile`, keeping `..` modes.
    pub const fn compose_tile<T>(self, tile: T) -> <Self as ComposeTile<T>>::Output
    where
        Self: [const] ComposeTile<T>,
    {
        ComposeTile::compose_tile(self, tile)
    }

    /// Logically divides the tensor's layout by `b` while keeping its accessor unchanged.
    pub const fn logical_divide<B>(self, b: B) -> <Self as LogicalDivide<B>>::Output
    where
        Self: [const] LogicalDivide<B>,
    {
        LogicalDivide::logical_divide(self, b)
    }

    /// Logically divides only the sub-mode at `Path` (`mode![...]`) by `b`.
    pub const fn logical_divide_at<Path, B>(
        self,
        b: B,
    ) -> <Self as LogicalDivideAt<B, Path>>::Output
    where
        Self: [const] LogicalDivideAt<B, Path>,
    {
        LogicalDivideAt::logical_divide_at(self, b)
    }

    /// Promotes `b` via [`tiler_to_layout`](crate::layout::tiler_to_layout) and divides the
    /// tensor into `(Tile, Grid)` modes while keeping its accessor unchanged.
    pub const fn zipped_divide<B>(self, b: B) -> <Self as ZippedDivide<B>>::Output
    where
        Self: [const] ZippedDivide<B>,
    {
        ZippedDivide::zipped_divide(self, b)
    }

    /// Divides the tensor into `(Tile, Rest_0, ...)` while keeping its accessor unchanged.
    pub const fn tiled_divide<B>(self, b: B) -> <Self as TiledDivide<B>>::Output
    where
        Self: [const] TiledDivide<B>,
    {
        TiledDivide::tiled_divide(self, b)
    }

    /// Divides the tensor into `(Tile_0, ..., Rest_0, ...)` while keeping its accessor unchanged.
    pub const fn flat_divide<B>(self, b: B) -> <Self as FlatDivide<B>>::Output
    where
        Self: [const] FlatDivide<B>,
    {
        FlatDivide::flat_divide(self, b)
    }

    /// Promotes `b` and zipped-divides only the sub-mode at `Path` (`mode![...]`).
    pub const fn zipped_divide_at<Path, B>(self, b: B) -> <Self as ZippedDivideAt<B, Path>>::Output
    where
        Self: [const] ZippedDivideAt<B, Path>,
    {
        ZippedDivideAt::zipped_divide_at(self, b)
    }
}

impl<A, S, D> Tensor<A, Layout<S, D>> {
    /// Moves the tensor's origin to complete coordinate `coord`, keeping `self.layout` unchanged
    /// (CuTe `domain_offset`).
    pub const fn domain_offset<C>(self, coord: C) -> DomainOffsetOutput<A, S, D, C>
    where
        S: [const] Shape,
        D: [const] Stride<S>,
        Layout<S, D>: [const] OffsetAndSlice<C, SlicedShape = Nil, SlicedStride = Nil>,
        A: [const] AccessorOffset<<Layout<S, D> as OffsetAndSlice<C>>::Offset>,
    {
        domain_offset(self, coord)
    }

    /// Returns the tile of `self` at `coord` (forwarding to [`local_tile`]).
    pub const fn local_tile<B, C, TS, TD>(self, tiler: B, coord: C) -> LocalTileOutput<A, TS, TD, C>
    where
        A: Copy,
        S: Shape,
        D: Stride<S>,
        Self: [const] ZippedDivide<B, Output = Tensor<A, Layout<TS, TD>>>,
        TS: Shape + [const] Wrap,
        TD: Stride<TS>,
        TS::Wrapped: [const] Get<1>,
        C: Item + WeaklyCongruent<GetOutput<TS::Wrapped, 1>>,
        Tensor<A, Layout<TS, TD>>: [const] TensorIndex<LocalTileCoord<C>>,
    {
        local_tile(self, tiler, coord)
    }

    /// CuTe `inner_partition(self, tiler, coord)`: selects tile `coord` with unpacked tile modes
    /// and `..`-padded trailing rest modes.
    pub const fn inner_partition<B, C, TS, TD>(
        self,
        tiler: B,
        coord: C,
    ) -> InnerPartitionOutput<A, TS, TD, C>
    where
        A: Copy,
        S: Shape,
        D: Stride<S>,
        Self: [const] ZippedDivide<B, Output = Tensor<A, Layout<TS, TD>>>,
        TS: Shape + [const] InnerCoord<C>,
        TD: Stride<TS>,
        Tensor<A, Layout<TS, TD>>: [const] TensorIndex<<TS as InnerCoord<C>>::Output>,
    {
        inner_partition(self, tiler, coord)
    }

    /// CuTe `outer_partition(self, tiler, coord)`: selects element `coord` of every tile with
    /// unpacked rest modes.
    pub const fn outer_partition<B, C, TS, TD>(
        self,
        tiler: B,
        coord: C,
    ) -> OuterPartitionOutput<A, TS, TD, C>
    where
        A: Copy,
        S: Shape,
        D: Stride<S>,
        Self: [const] ZippedDivide<B, Output = Tensor<A, Layout<TS, TD>>>,
        TS: Shape + [const] OuterCoord<C>,
        TD: Stride<TS>,
        Tensor<A, Layout<TS, TD>>: [const] TensorIndex<<TS as OuterCoord<C>>::Output>,
    {
        outer_partition(self, tiler, coord)
    }

    /// CuTe `local_tile(self, tiler, coord, proj)`: dices `tiler` and `coord` with `proj`, then
    /// delegates to [`inner_partition`].
    pub const fn local_tile_proj<B, C, P, TS, TD>(
        self,
        tiler: B,
        coord: C,
        proj: P,
    ) -> InnerPartitionOutput<A, TS, TD, <P as DiceBy<C>>::Output>
    where
        A: Copy,
        S: Shape,
        D: Stride<S>,
        B: [const] Item,
        C: [const] Item,
        P: [const] DiceBy<B> + [const] DiceBy<C>,
        Self: [const] ZippedDivide<<P as DiceBy<B>>::Output, Output = Tensor<A, Layout<TS, TD>>>,
        TS: Shape + [const] InnerCoord<<P as DiceBy<C>>::Output>,
        TD: Stride<TS>,
        Tensor<A, Layout<TS, TD>>:
            [const] TensorIndex<<TS as InnerCoord<<P as DiceBy<C>>::Output>>::Output>,
    {
        local_tile_proj(self, tiler, coord, proj)
    }

    /// CuTe `local_partition(self, thr_layout, index)`: returns the strided share of thread `index`.
    pub const fn local_partition<LS, LD, X, TS, TD>(
        self,
        thr_layout: &Layout<LS, LD>,
        index: X,
    ) -> LocalPartitionOutput<A, LS, LD, X, TS, TD>
    where
        A: Copy,
        S: Shape,
        D: Stride<S>,
        LS: Shape + [const] ProductEach,
        LD: Stride<LS>,
        Layout<LS, LD>: [const] GetFlatCoord<X>,
        Self: [const] ZippedDivide<
                <LS as ProductEach>::Output,
                Output = Tensor<A, Layout<TS, TD>>,
            >,
        TS: Shape + [const] OuterCoord<<Layout<LS, LD> as GetFlatCoord<X>>::Output>,
        TD: Stride<TS>,
        Tensor<A, Layout<TS, TD>>:
            [const] TensorIndex<<TS as OuterCoord<<Layout<LS, LD> as GetFlatCoord<X>>::Output>>::Output>,
    {
        local_partition(self, thr_layout, index)
    }

    /// CuTe `local_partition(self, thr_layout, index, proj)`: dices `thr_layout` with `proj`, then
    /// delegates to [`local_partition`].
    pub const fn local_partition_proj<LS, LD, X, P, TS, TD>(
        self,
        thr_layout: &Layout<LS, LD>,
        index: X,
        proj: P,
    ) -> LocalPartitionOutput<A, <P as DiceBy<LS>>::Output, <P as DiceBy<LD>>::Output, X, TS, TD>
    where
        A: Copy,
        S: Shape,
        D: Stride<S>,
        LS: [const] Item,
        LD: [const] Item,
        P: [const] DiceBy<LS> + [const] DiceBy<LD>,
        <P as DiceBy<LS>>::Output: Shape + [const] ProductEach,
        <P as DiceBy<LD>>::Output: Stride<<P as DiceBy<LS>>::Output>,
        Layout<<P as DiceBy<LS>>::Output, <P as DiceBy<LD>>::Output>: [const] GetFlatCoord<X>,
        Self: [const] ZippedDivide<
                <<P as DiceBy<LS>>::Output as ProductEach>::Output,
                Output = Tensor<A, Layout<TS, TD>>,
            >,
        TS: Shape
            + [const] OuterCoord<
                <Layout<<P as DiceBy<LS>>::Output, <P as DiceBy<LD>>::Output> as GetFlatCoord<X>>::Output,
            >,
        TD: Stride<TS>,
        Tensor<A, Layout<TS, TD>>: [const] TensorIndex<
                <TS as OuterCoord<
                    <Layout<<P as DiceBy<LS>>::Output, <P as DiceBy<LD>>::Output> as GetFlatCoord<X>>::Output,
                >>::Output,
            >,
    {
        local_partition_proj(self, thr_layout, index, proj)
    }
}

// ------------------------------------------------------------------
// domain_offset
// ------------------------------------------------------------------

/// Resulting shifted [`Tensor`] type of [`domain_offset`].
pub type DomainOffsetOutput<A, S, D, C> =
    Tensor<<A as AccessorOffset<<Layout<S, D> as OffsetAndSlice<C>>::Offset>>::Output, Layout<S, D>>;

/// Shifts `tensor`'s accessor origin by `tensor.layout(coord)` while keeping `tensor.layout`
/// unchanged (CuTe `domain_offset`).
///
/// Requires a complete coordinate `coord` (coordinates containing `..` fail at compile time).
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::tensor::{domain_offset, make_tensor};
///
/// let data: [i32; 12] = core::array::from_fn(|i| i as i32);
/// let t = make_tensor(&data, int_tuple![Int::<3>, Int::<4>]);
/// let shifted = domain_offset(t, int_tuple![1i32, 1i32]);
/// assert_eq!(shifted.layout, t.layout);
/// assert_eq!(shifted(0i32, 0i32), 4);
/// assert_eq!(shifted(1i32, 2i32), 11);
/// ```
///
/// Partial slicing coordinates containing `..` fail at compile time:
///
/// ```compile_fail,E0271
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::tensor::{domain_offset, make_tensor};
/// use noodle_cute::{int_tuple, tuple};
///
/// let data = [0i32; 12];
/// let t = make_tensor(&data, int_tuple![Int::<3>, Int::<4>]);
/// let _ = domain_offset(t, tuple![1i32, ..]);
/// ```
pub const fn domain_offset<A, S, D, C>(
    tensor: Tensor<A, Layout<S, D>>,
    coord: C,
) -> DomainOffsetOutput<A, S, D, C>
where
    S: [const] Shape,
    D: [const] Stride<S>,
    Layout<S, D>: [const] OffsetAndSlice<C, SlicedShape = Nil, SlicedStride = Nil>,
    A: [const] AccessorOffset<<Layout<S, D> as OffsetAndSlice<C>>::Offset>,
{
    let (offset, _) = tensor.layout.offset_and_slice(coord);
    Tensor {
        accessor: tensor.accessor.offset(offset),
        layout: tensor.layout,
    }
}

// ------------------------------------------------------------------
// Partitioning: inner_partition, outer_partition, local_tile, local_tile_proj,
//               local_partition, local_partition_proj
// ------------------------------------------------------------------

/// Constructs the slicing coordinate `(repeat_modes(.., Tile), pad_coord(coord, Rest))` for
/// [`inner_partition`].
#[doc(hidden)]
pub const trait InnerCoord<C>: Copy {
    type Output: Copy;
    fn inner_coord(coord: C) -> Self::Output;
}

const impl<S0, S1, C> InnerCoord<C> for Til<S0, Til<S1, Nil>>
where
    S0: [const] RepeatModes<RangeFull>,
    S1: [const] Item,
    C: [const] PadCoord<S1>,
{
    type Output = Til<S0::Output, Til<C::Output, Nil>>;

    fn inner_coord(coord: C) -> Self::Output {
        Til(S0::repeat_modes(..), Til(coord.pad_coord(), Nil))
    }
}

/// Constructs the slicing coordinate `(pad_coord(coord, Tile), repeat_modes(.., Rest))` for
/// [`outer_partition`].
#[doc(hidden)]
pub const trait OuterCoord<C>: Copy {
    type Output: Copy;
    fn outer_coord(coord: C) -> Self::Output;
}

const impl<S0, S1, C> OuterCoord<C> for Til<S0, Til<S1, Nil>>
where
    S0: [const] Item,
    S1: [const] RepeatModes<RangeFull>,
    C: [const] PadCoord<S0>,
{
    type Output = Til<C::Output, Til<S1::Output, Nil>>;

    fn outer_coord(coord: C) -> Self::Output {
        Til(coord.pad_coord(), Til(S1::repeat_modes(..), Nil))
    }
}

/// Output sub-[`Tensor`] type returned by [`inner_partition`] and [`local_tile_proj`].
pub type InnerPartitionOutput<A, TS, TD, C> =
    <Tensor<A, Layout<TS, TD>> as TensorIndex<<TS as InnerCoord<C>>::Output>>::Output;

/// Output sub-[`Tensor`] type returned by [`outer_partition`].
pub type OuterPartitionOutput<A, TS, TD, C> =
    <Tensor<A, Layout<TS, TD>> as TensorIndex<<TS as OuterCoord<C>>::Output>>::Output;

/// Output sub-[`Tensor`] type returned by [`local_partition`] and [`local_partition_proj`].
pub type LocalPartitionOutput<A, LS, LD, X, TS, TD> =
    OuterPartitionOutput<A, TS, TD, <Layout<LS, LD> as GetFlatCoord<X>>::Output>;

/// CuTe `inner_partition(tensor, tiler, coord)`: selects tile `coord` with unpacked tile modes.
///
/// Splits `tensor` via [`zipped_divide`](crate::algebra::zipped_divide) into `(Tile, Rest)`
/// and slices with `(repeat_modes(.., Tile), pad_coord(coord, Rest))`:
/// - `Tile`'s top-level modes are unpacked directly into the output's leading modes;
/// - a shorter tuple `coord` is padded with `..` so unspecified trailing rest modes are kept;
/// - a scalar integer `coord` indexes the entire `Rest` mode colexicographically.
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::tensor::{inner_partition, make_tensor};
///
/// let data: [i32; 96] = core::array::from_fn(|i| i as i32);
/// let t = make_tensor(&data, int_tuple![12i32, 8i32]);
/// let tile = inner_partition(t, int_tuple![Int::<4>, Int::<2>], int_tuple![2i32, 3i32]);
/// assert_eq!(tile.shape(), int_tuple![Int::<4>, Int::<2>]);
/// assert_eq!(tile(1i32, 1i32), 9 + 7 * 12);
/// ```
pub const fn inner_partition<A, S, D, B, C, TS, TD>(
    tensor: Tensor<A, Layout<S, D>>,
    tiler: B,
    coord: C,
) -> InnerPartitionOutput<A, TS, TD, C>
where
    A: Copy,
    S: Shape,
    D: Stride<S>,
    Tensor<A, Layout<S, D>>: [const] ZippedDivide<B, Output = Tensor<A, Layout<TS, TD>>>,
    TS: Shape + [const] InnerCoord<C>,
    TD: Stride<TS>,
    Tensor<A, Layout<TS, TD>>: [const] TensorIndex<<TS as InnerCoord<C>>::Output>,
{
    let tiled = tensor.zipped_divide(tiler);
    tiled.at(<TS as InnerCoord<C>>::inner_coord(coord))
}

/// CuTe `outer_partition(tensor, tiler, coord)`: selects element `coord` of every tile with
/// unpacked rest modes.
///
/// Splits `tensor` via [`zipped_divide`](crate::algebra::zipped_divide) into `(Tile, Rest)`
/// and slices with `(pad_coord(coord, Tile), repeat_modes(.., Rest))`.
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::tensor::{make_tensor, outer_partition};
///
/// let data: [i32; 96] = core::array::from_fn(|i| i as i32);
/// let t = make_tensor(&data, int_tuple![12i32, 8i32]);
/// let spread = outer_partition(t, int_tuple![Int::<4>, Int::<2>], int_tuple![1i32, 1i32]);
/// assert_eq!(spread.shape(), int_tuple![3i32, 4i32]);
/// assert_eq!(spread(2i32, 3i32), 9 + 7 * 12);
/// ```
pub const fn outer_partition<A, S, D, B, C, TS, TD>(
    tensor: Tensor<A, Layout<S, D>>,
    tiler: B,
    coord: C,
) -> OuterPartitionOutput<A, TS, TD, C>
where
    A: Copy,
    S: Shape,
    D: Stride<S>,
    Tensor<A, Layout<S, D>>: [const] ZippedDivide<B, Output = Tensor<A, Layout<TS, TD>>>,
    TS: Shape + [const] OuterCoord<C>,
    TD: Stride<TS>,
    Tensor<A, Layout<TS, TD>>: [const] TensorIndex<<TS as OuterCoord<C>>::Output>,
{
    let tiled = tensor.zipped_divide(tiler);
    tiled.at(<TS as OuterCoord<C>>::outer_coord(coord))
}

/// CuTe `local_tile(tensor, tiler, coord, proj)`: projects `tiler` and `coord` with
/// [`dice_by`] (`..` drops a mode, any non-wildcard keeps it) and delegates to [`inner_partition`].
///
/// Allows a single `(BLK_M, BLK_N, BLK_K)` tiler and `(bx, by, ..)` block coordinate to serve
/// `A`, `B`, and `C` tensors uniformly.
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::tensor::{local_tile_proj, make_tensor};
/// use noodle_cute::{int_tuple, tuple};
///
/// let a_data = [0f32; 64 * 32];
/// let a = make_tensor(&a_data, int_tuple![64i32, 32i32]);
/// let cta_tiler = int_tuple![Int::<16>, Int::<32>, Int::<8>];
/// let cta_coord = tuple![1i32, 0i32, ..];
/// let g_a = local_tile_proj(a, cta_tiler, cta_coord, tuple![Int::<1>, .., Int::<1>]);
/// assert_eq!(g_a.shape(), int_tuple![Int::<16>, Int::<8>, 4i32]);
/// ```
pub const fn local_tile_proj<A, S, D, B, C, P, TS, TD>(
    tensor: Tensor<A, Layout<S, D>>,
    tiler: B,
    coord: C,
    proj: P,
) -> InnerPartitionOutput<A, TS, TD, <P as DiceBy<C>>::Output>
where
    A: Copy,
    S: Shape,
    D: Stride<S>,
    B: [const] Item,
    C: [const] Item,
    P: [const] DiceBy<B> + [const] DiceBy<C>,
    Tensor<A, Layout<S, D>>:
        [const] ZippedDivide<<P as DiceBy<B>>::Output, Output = Tensor<A, Layout<TS, TD>>>,
    TS: Shape + [const] InnerCoord<<P as DiceBy<C>>::Output>,
    TD: Stride<TS>,
    Tensor<A, Layout<TS, TD>>:
        [const] TensorIndex<<TS as InnerCoord<<P as DiceBy<C>>::Output>>::Output>,
{
    inner_partition(tensor, dice_by(&proj, &tiler), dice_by(&proj, &coord))
}

/// CuTe `local_partition(tensor, thr_layout, index)`: returns the strided elements owned by
/// thread `index`.
///
/// Divides `tensor` by `product_each(&thr_layout.shape)` and slices the tile mode at
/// `thr_layout.get_flat_coord(index)` via [`outer_partition`].
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
/// use noodle_cute::tensor::{local_partition, make_tensor};
///
/// let data: [i32; 16 * 8] = core::array::from_fn(|i| i as i32);
/// let tile = make_tensor(&data, int_tuple![Int::<16>, Int::<8>]);
/// let threads = Layout::from_shape(int_tuple![Int::<4>, Int::<2>]);
/// let mine = local_partition(tile, &threads, 5i32);
/// assert_eq!(mine.shape(), int_tuple![Int::<4>, Int::<4>]);
/// assert_eq!(mine(2i32, 3i32), 9 + 7 * 16);
/// ```
pub const fn local_partition<A, S, D, LS, LD, X, TS, TD>(
    tensor: Tensor<A, Layout<S, D>>,
    thr_layout: &Layout<LS, LD>,
    index: X,
) -> LocalPartitionOutput<A, LS, LD, X, TS, TD>
where
    A: Copy,
    S: Shape,
    D: Stride<S>,
    LS: Shape + [const] ProductEach,
    LD: Stride<LS>,
    Layout<LS, LD>: [const] GetFlatCoord<X>,
    Tensor<A, Layout<S, D>>:
        [const] ZippedDivide<<LS as ProductEach>::Output, Output = Tensor<A, Layout<TS, TD>>>,
    TS: Shape + [const] OuterCoord<<Layout<LS, LD> as GetFlatCoord<X>>::Output>,
    TD: Stride<TS>,
    Tensor<A, Layout<TS, TD>>:
        [const] TensorIndex<<TS as OuterCoord<<Layout<LS, LD> as GetFlatCoord<X>>::Output>>::Output>,
{
    outer_partition(
        tensor,
        product_each(&thr_layout.shape),
        thr_layout.get_flat_coord(index),
    )
}

/// CuTe `local_partition(tensor, thr_layout, index, proj)`: projects `thr_layout` with
/// [`Layout::dice`] and delegates to [`local_partition`].
pub const fn local_partition_proj<A, S, D, LS, LD, X, P, TS, TD>(
    tensor: Tensor<A, Layout<S, D>>,
    thr_layout: &Layout<LS, LD>,
    index: X,
    proj: P,
) -> LocalPartitionOutput<A, <P as DiceBy<LS>>::Output, <P as DiceBy<LD>>::Output, X, TS, TD>
where
    A: Copy,
    S: Shape,
    D: Stride<S>,
    LS: [const] Item,
    LD: [const] Item,
    P: [const] DiceBy<LS> + [const] DiceBy<LD>,
    <P as DiceBy<LS>>::Output: Shape + [const] ProductEach,
    <P as DiceBy<LD>>::Output: Stride<<P as DiceBy<LS>>::Output>,
    Layout<<P as DiceBy<LS>>::Output, <P as DiceBy<LD>>::Output>: [const] GetFlatCoord<X>,
    Tensor<A, Layout<S, D>>: [const] ZippedDivide<
            <<P as DiceBy<LS>>::Output as ProductEach>::Output,
            Output = Tensor<A, Layout<TS, TD>>,
        >,
    TS: Shape
        + [const] OuterCoord<
            <Layout<<P as DiceBy<LS>>::Output, <P as DiceBy<LD>>::Output> as GetFlatCoord<X>>::Output,
        >,
    TD: Stride<TS>,
    Tensor<A, Layout<TS, TD>>: [const] TensorIndex<
            <TS as OuterCoord<
                <Layout<<P as DiceBy<LS>>::Output, <P as DiceBy<LD>>::Output> as GetFlatCoord<X>>::Output,
            >>::Output,
        >,
{
    local_partition(tensor, &thr_layout.dice(&proj), index)
}

/// Slicing coordinate `(.., coord)` used internally by [`local_tile`].
#[doc(hidden)]
pub type LocalTileCoord<C> = Til<RangeFull, Til<C, Nil>>;

/// Output sub-[`Tensor`] type returned by [`local_tile`].
pub type LocalTileOutput<A, TS, TD, C> =
    <Tensor<A, Layout<TS, TD>> as TensorIndex<LocalTileCoord<C>>>::Output;

/// Returns the tile of `tensor` at `coord` in the grid of `tiler`-shaped tiles.
///
/// Splits `tensor` via [`zipped_divide`](crate::algebra::zipped_divide) into `(Tile, Grid)`
/// and slices `Grid` with `coord` (`tiled.at(tuple!(.., coord))`), so mode 0 of the
/// result is `Tile` followed by any grid modes marked `..` in `coord`.
///
/// # Compile-Time Checks
///
/// - `tensor` must be a [`Tensor`] (passing a [`Layout`] fails to type-check).
/// - `coord` must be [`WeaklyCongruent`] with `Grid` (mode 1 of `zipped_divide(tensor, tiler)`).
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::basis::as_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::tensor::{identity_tensor, local_tile};
/// use noodle_cute::{int_tuple, tuple};
///
/// let id = identity_tensor(int_tuple![Int::<8>, Int::<12>]);
/// let tile = local_tile(id, int_tuple![Int::<4>, Int::<6>], int_tuple![1i32, 1i32]);
/// assert_eq!(tile.shape(), int_tuple![(Int::<4>, Int::<6>)]);
/// assert_eq!(as_tuple(tile.at(tuple!((3i32, 5i32)))), int_tuple![7i32, 11i32]);
/// ```
pub const fn local_tile<A, S, D, B, C, TS, TD>(
    tensor: Tensor<A, Layout<S, D>>,
    tiler: B,
    coord: C,
) -> LocalTileOutput<A, TS, TD, C>
where
    A: Copy,
    S: Shape,
    D: Stride<S>,
    Tensor<A, Layout<S, D>>: [const] ZippedDivide<B, Output = Tensor<A, Layout<TS, TD>>>,
    TS: Shape + [const] Wrap,
    TD: Stride<TS>,
    TS::Wrapped: [const] Get<1>,
    C: Item + WeaklyCongruent<GetOutput<TS::Wrapped, 1>>,
    Tensor<A, Layout<TS, TD>>: [const] TensorIndex<LocalTileCoord<C>>,
{
    let tiled = tensor.zipped_divide(tiler);
    tiled.at(Til(.., Til(coord, Nil)))
}
