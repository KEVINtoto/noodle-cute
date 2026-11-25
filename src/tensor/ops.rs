//! Tensor indexing, slicing, layout-algebra pass-through, and [`local_tile`].
//!
//! # Main Interfaces
//!
//! - **Indexing & slicing**: [`TensorIndex`] (`Tensor::at` and `const` `Fn` call syntax for
//!   1–6 arguments) and [`TensorSet`] (`Tensor::set`).
//! - **Layout-algebra pass-through**: `Tensor` implementations of
//!   [`CoalesceTarget`], [`CoalesceExtended`], [`Composition`], [`LogicalDivide`],
//!   [`LogicalDivideAt`], and [`IntoGcdShape`] (with [`ZippedDivide`] and [`ZippedDivideAt`]
//!   inherited automatically via blanket implementations).
//! - **Block tiling**: [`local_tile`] (`zipped_divide(tensor, tiler).at(tuple!(.., coord))`
//!   with compile-time [`WeaklyCongruent`] grid validation).
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
use crate::algebra::composition::Composition;
use crate::algebra::greatest_common_domain::IntoGcdShape;
use crate::algebra::logical_divide::{
    LogicalDivide, LogicalDivideAt, ZippedDivide, ZippedDivideAt,
};
use crate::layout::{Layout, OffsetAndSlice};
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tuple::{Get, GetOutput, Item, Nil, Til, Tuple, WeaklyCongruent, Wrap};

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

    /// Promotes `b` and zipped-divides only the sub-mode at `Path` (`mode![...]`).
    pub const fn zipped_divide_at<Path, B>(self, b: B) -> <Self as ZippedDivideAt<B, Path>>::Output
    where
        Self: [const] ZippedDivideAt<B, Path>,
    {
        ZippedDivideAt::zipped_divide_at(self, b)
    }
}

// ------------------------------------------------------------------
// local_tile
// ------------------------------------------------------------------

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
