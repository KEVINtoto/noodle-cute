//! Tensors: a layout composed with an engine that holds or points to elements
//! (`tensor_impl.hpp`).
//!
//! A [`ViewEngine`] holds an iterator: a raw or memory-space tagged pointer, or a
//! [`CoordIter`] for identity tensors. An [`ArrayEngine`] owns register-like
//! storage sized by the `cosize` of a static layout. As in `docs/device-roadmap.md`,
//! layouts are validated when they are built and partitioned; evaluating a
//! tensor is unchecked address arithmetic.
//!
//! Views are `Copy`, and slicing or partitioning consumes and returns views.
//! Element access through pointers is `unsafe` ([`Tensor::read`],
//! [`Tensor::write`]) because a layout cannot prove that an address is valid or
//! free of races between threads. Owned tensors are accessed safely with bounds
//! checks, and [`Tensor::view`] / [`Tensor::view_mut`] turn them into views
//! for slicing, where upstream decays them to pointer views implicitly.
//!
//! ```
//! use noodle_cute::coord::X;
//! use noodle_cute::layout::LayoutLeft;
//! use noodle_cute::tensor::make_tensor;
//! use noodle_cute::{Int, shape};
//!
//! let data: [i32; 12] = core::array::from_fn(|i| i as i32);
//! let a = make_tensor(data.as_ptr(), LayoutLeft::make(shape![Int::<3>, Int::<4>])?);
//! let column = a.slice(shape![X, 2i32]);
//! assert_eq!(unsafe { column.read(shape![1i32]) }, 7);
//! # Ok::<(), noodle_cute::Error>(())
//! ```

use crate::algebra::ZippedDivide;
use crate::basis::{IdentityShape, OffsetAdd, make_identity_layout};
use crate::coord::{Crd2Crd, Crd2Idx, DiceOf, Idx2Crd, InnerCoord, OuterCoord, RepeatModes, dice};
use crate::int::{DynInt, Int, Integer};
use crate::layout::{
    DomainOffset, FragmentLike, Layout, LayoutLike, LayoutShape, MapCoord, RecastView, SliceAt,
    StaticDesc,
};
use crate::pointer::{Advance, CoordIter, Iter, Read, RecastPtr, Write};
use crate::tuple::{Product, ProductEach};
use crate::{Result, StaticValue};

/// A non-owning engine holding an iterator: CuTe `ViewEngine`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct ViewEngine<I>(pub I);

/// Owned storage for `N` elements: CuTe `ArrayEngine`.
///
/// Aligned to 16 bytes like upstream `array_aligned`, so 128-bit vector copies
/// of a fragment stay aligned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct ArrayEngine<T, const N: usize>(pub [T; N]);

impl<T: Copy + Default, const N: usize> Default for ArrayEngine<T, N> {
    fn default() -> Self {
        Self([T::default(); N])
    }
}

/// A layout composed with an engine: CuTe `Tensor<Engine, Layout>`.
///
/// `repr(C)` keeps the layout-then-engine field order of upstream's `rep_`, so a
/// view built on the host can be passed by value as a kernel parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Tensor<E, L> {
    /// Maps coordinates to element offsets.
    pub layout: L,
    /// Holds or points to the elements.
    pub engine: E,
}

/// `cosize_v<Layout>` of a static layout.
pub const COSIZE<L: StaticDesc>: usize = match L::DESC.cosize() {
    Ok(n) => n as usize,
    Err(_) => panic!("the static layout has no representable cosize"),
};

/// Owned storage sized by a static layout: the engine of CuTe `make_tensor<T>(layout)`.
pub trait OwnedStorage<T> {
    /// The owned engine.
    type Engine;
    /// Returns default-initialized storage.
    fn storage() -> Self::Engine;
}

impl<T: Copy + Default, L: StaticDesc> OwnedStorage<T> for L {
    type Engine = ArrayEngine<T, { COSIZE::<L> }>;
    fn storage() -> Self::Engine {
        ArrayEngine::default()
    }
}

/// A non-owning tensor: CuTe `make_tensor(iter, layout)`.
pub fn make_tensor<I: Iter, L>(iter: I, layout: L) -> Tensor<ViewEngine<I>, L> {
    Tensor {
        layout,
        engine: ViewEngine(iter),
    }
}

/// An owning tensor of default-initialized elements: CuTe `make_tensor<T>(layout)`.
///
/// As upstream, the layout must be static; its `cosize` sizes the storage.
///
/// ```
/// use noodle_cute::tensor::make_owned_tensor;
/// use noodle_cute::layout::LayoutLeft;
/// use noodle_cute::{Int, shape};
///
/// let mut frag = make_owned_tensor::<f32, _>(LayoutLeft::make(shape![Int::<2>, Int::<4>])?);
/// *frag.get_mut(shape![1i32, 3i32]) = 2.5;
/// assert_eq!(frag.engine.0[7], 2.5);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
pub fn make_owned_tensor<T, L: OwnedStorage<T>>(layout: L) -> Tensor<L::Engine, L> {
    Tensor {
        layout,
        engine: L::storage(),
    }
}

type Like<S, D> = <Layout<S, D> as LayoutLike>::Output;
type Fragment<L> = <L as FragmentLike>::Output;

/// An owning tensor with the shape and stride order of `layout`: CuTe `make_tensor_like<T>`.
pub fn make_tensor_like<T, S, D>(
    layout: &Layout<S, D>,
) -> Tensor<<Like<S, D> as OwnedStorage<T>>::Engine, Like<S, D>>
where
    Layout<S, D>: LayoutLike,
    Like<S, D>: OwnedStorage<T>,
{
    make_owned_tensor(layout.make_layout_like())
}

/// An owning register fragment shaped like `layout`: CuTe `make_fragment_like<T>`.
///
/// A swizzled layout gives the fragment of its `layout_b`, as upstream.
///
/// ```
/// use noodle_cute::tensor::make_fragment_like;
/// use noodle_cute::{DynInt, Int, Layout, shape, stride};
///
/// // A thread's partition: (values, M, N). Mode 0 stays contiguous, and the
/// // static strides order N (8) before M (64).
/// let partition = Layout::new(
///     shape![Int::<2>, Int::<4>, Int::<2>],
///     stride![Int::<1>, Int::<64>, Int::<8>],
/// )?;
/// let frag = make_fragment_like::<f32, _>(&partition);
/// assert_eq!(frag.layout.stride, stride![Int::<1>, Int::<4>, Int::<2>]);
/// assert_eq!(frag.engine.0.len(), 16);
///
/// // Runtime strides are ignored and ordered from left to right, as upstream.
/// let dynamic = Layout::new(partition.shape, stride![Int::<1>, DynInt(64), DynInt(8)])?;
/// let frag = make_fragment_like::<f32, _>(&dynamic);
/// assert_eq!(frag.layout.stride, stride![Int::<1>, Int::<2>, Int::<8>]);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
pub fn make_fragment_like<T, L>(
    _: &L,
) -> Tensor<<Fragment<L> as OwnedStorage<T>>::Engine, Fragment<L>>
where
    L: FragmentLike,
    Fragment<L>: OwnedStorage<T>,
{
    make_owned_tensor(<Fragment<L> as StaticValue>::VALUE)
}

/// A tensor whose element at each coordinate is the coordinate itself:
/// CuTe `make_identity_tensor(shape)`.
///
/// Partitioning it like a data tensor yields the global coordinate of every
/// element, which [`elem_less`](crate::coord::elem_less) compares with the shape
/// to predicate tail tiles.
///
/// ```
/// use noodle_cute::coord::elem_less;
/// use noodle_cute::tensor::make_identity_tensor;
/// use noodle_cute::{DynInt, Int, shape};
///
/// let extent = shape![DynInt(10), DynInt(8)];
/// let identity = make_identity_tensor(extent);
/// let tail = identity.local_tile(shape![Int::<4>, Int::<4>], shape![2i32, 1i32])?;
/// let c = tail.get(shape![3i32, 0i32]);
/// assert_eq!(c, shape![11i32, 4i32]);
/// assert!(!elem_less(&c, &extent));
/// # Ok::<(), noodle_cute::Error>(())
/// ```
pub fn make_identity_tensor<S: IdentityShape>(
    shape: S,
) -> View<CoordIter<S::Zero>, Layout<S, S::Stride>> {
    make_tensor(CoordIter(S::zero()), make_identity_layout(shape))
}

impl<T: Copy, const N: usize, S, D> Tensor<ArrayEngine<T, N>, Layout<S, D>> {
    fn storage_index<C>(&self, coord: C) -> usize
    where
        C: Crd2Idx<S, D>,
        C::Output: Integer,
    {
        usize::try_from(self.layout.crd2idx(coord).value())
            .ok()
            .filter(|&i| i < N)
            .expect("coordinate maps outside the owned storage")
    }

    /// Returns the element at `coord`.
    ///
    /// # Panics
    ///
    /// Panics if the coordinate maps outside the storage, which can only happen
    /// for coordinates outside the layout's domain.
    pub fn get<C>(&self, coord: C) -> T
    where
        C: Crd2Idx<S, D>,
        C::Output: Integer,
    {
        self.engine.0[self.storage_index(coord)]
    }

    /// Returns a mutable reference to the element at `coord`; panics like [`Tensor::get`].
    pub fn get_mut<C>(&mut self, coord: C) -> &mut T
    where
        C: Crd2Idx<S, D>,
        C::Output: Integer,
    {
        let i = self.storage_index(coord);
        &mut self.engine.0[i]
    }

    /// A read-only view of the storage, for slicing and partitioning.
    pub fn view(&self) -> Tensor<ViewEngine<*const T>, Layout<S, D>>
    where
        Layout<S, D>: Clone,
    {
        make_tensor(self.engine.0.as_ptr(), self.layout.clone())
    }

    /// A writable view of the storage, valid while `self` is borrowed and not moved.
    pub fn view_mut(&mut self) -> Tensor<ViewEngine<*mut T>, Layout<S, D>>
    where
        Layout<S, D>: Clone,
    {
        make_tensor(self.engine.0.as_mut_ptr(), self.layout.clone())
    }
}

impl<Z: Copy, L> Tensor<ViewEngine<CoordIter<Z>>, L> {
    /// Returns the coordinate stored at `coord` of an identity tensor.
    pub fn get<C>(&self, coord: C) -> <Z as OffsetAdd<L::Output>>::Output
    where
        L: MapCoord<C>,
        Z: OffsetAdd<L::Output>,
    {
        self.engine.0.0.offset_add(self.layout.map_coord(coord))
    }
}

type View<I, L> = Tensor<ViewEngine<I>, L>;
type Adv<I, O> = <I as Advance<O>>::Output;
type Mapped<I, L, C> = Adv<I, <L as MapCoord<C>>::Output>;
type SlicedView<I, L, C> = View<Adv<I, <L as SliceAt<C>>::Offset>, <L as SliceAt<C>>::Sliced>;
type Zipped<L, T> = <L as ZippedDivide<T>>::Output;
type ZShape<L, T> = <Zipped<L, T> as LayoutShape>::Shape;
type Inner<L, T, C> = <ZShape<L, T> as InnerCoord<C>>::Output;
type Outer<L, T, C> = <ZShape<L, T> as OuterCoord<C>>::Output;
type InnerTile<I, L, T, C> = SlicedView<I, Zipped<L, T>, Inner<L, T, C>>;
type OuterTile<I, L, T, C> = SlicedView<I, Zipped<L, T>, Outer<L, T, C>>;
type FlatCoord<X, S, D> =
    <<X as Idx2Crd<S, D>>::Output as Crd2Crd<S, <S as RepeatModes<Int<1>>>::Output>>::Output;
type Diced<P, B> = <P as DiceOf<B>>::Output;
type Each<S> = <S as ProductEach>::Output;
type ProjTile<I, L, T, C, P> = InnerTile<I, L, Diced<P, T>, Diced<P, C>>;
type Partition<I, L, TS, TD, X> = OuterTile<I, L, Each<TS>, FlatCoord<X, TS, TD>>;
type ProjPartition<I, L, TS, TD, X, P> = Partition<I, L, Diced<P, TS>, Diced<P, TD>, X>;
type Recast<I, N, L> = View<
    <Adv<I, DynInt> as RecastPtr<N>>::Output,
    <L as RecastView<<I as Iter>::Value, N>>::Output,
>;

impl<I: Iter, L> Tensor<ViewEngine<I>, L> {
    /// The iterator at the tensor's origin: CuTe `tensor.data()`.
    pub fn data(&self) -> I {
        self.engine.0
    }

    /// The number of elements: CuTe `size(tensor)`.
    pub fn size(&self) -> isize
    where
        L: LayoutShape,
        L::Shape: Product<Output: Integer>,
    {
        self.layout.shape().product_typed().value()
    }

    /// Reads the element at `coord`: CuTe `tensor(coord)`.
    ///
    /// # Safety
    ///
    /// The address `data() + layout(coord)` must be valid for reads, and no other
    /// thread may write it concurrently.
    pub unsafe fn read<C>(&self, coord: C) -> <Mapped<I, L, C> as Iter>::Value
    where
        L: MapCoord<C>,
        I: Advance<L::Output>,
        Mapped<I, L, C>: Read,
    {
        // SAFETY: forwarded to the caller.
        unsafe { self.engine.0.advance(self.layout.map_coord(coord)).read() }
    }

    /// Writes the element at `coord`: CuTe `tensor(coord) = value`.
    ///
    /// # Safety
    ///
    /// The address `data() + layout(coord)` must be valid for writes, and no
    /// other thread may access it concurrently.
    pub unsafe fn write<C>(&self, coord: C, value: <Mapped<I, L, C> as Iter>::Value)
    where
        L: MapCoord<C>,
        I: Advance<L::Output>,
        Mapped<I, L, C>: Write,
    {
        // SAFETY: forwarded to the caller.
        unsafe {
            self.engine
                .0
                .advance(self.layout.map_coord(coord))
                .write(value)
        }
    }

    /// The sub-tensor of the modes marked by [`Underscore`](crate::coord::Underscore):
    /// CuTe `tensor(coord)` with `_`. A plain layout moves the origin to the
    /// slice's first element; a composed layout keeps the offset itself.
    pub fn slice<C>(self, coord: C) -> SlicedView<I, L, C>
    where
        L: SliceAt<C>,
        I: Advance<L::Offset>,
    {
        let (layout, offset) = self.layout.slice_and_offset(coord);
        make_tensor(self.engine.0.advance(offset), layout)
    }

    /// Moves the origin to `coord`, keeping the shape: CuTe `domain_offset(coord, tensor)`.
    pub fn domain_offset<C>(self, coord: C) -> View<Adv<I, L::Offset>, L::Moved>
    where
        L: DomainOffset<C>,
        I: Advance<L::Offset>,
    {
        let (layout, offset) = self.layout.domain_offset(coord);
        make_tensor(self.engine.0.advance(offset), layout)
    }

    /// CuTe `zipped_divide(tensor, tiler)`: `((tile...), (rest...))` over the same data.
    pub fn zipped_divide<T>(self, tiler: T) -> Result<View<I, Zipped<L, T>>>
    where
        L: ZippedDivide<T>,
    {
        Ok(make_tensor(
            self.engine.0,
            self.layout.zipped_divide(tiler)?,
        ))
    }

    /// CuTe `inner_partition(tensor, tiler, coord)`: the tile selected by `coord`.
    ///
    /// The tensor is zipped-divided by `tiler` and the rest mode is sliced at
    /// `coord`; a tuple `coord` is padded with `_` to the rank of the rest mode,
    /// so its unspecified trailing modes are kept, while an integer indexes the
    /// whole rest mode. Tail tiles extend past the shape; predicate them with an
    /// identity tensor.
    pub fn inner_partition<T, C>(self, tiler: T, coord: C) -> Result<InnerTile<I, L, T, C>>
    where
        L: ZippedDivide<T>,
        Zipped<L, T>: LayoutShape + SliceAt<Inner<L, T, C>>,
        ZShape<L, T>: InnerCoord<C>,
        I: Advance<<Zipped<L, T> as SliceAt<Inner<L, T, C>>>::Offset>,
    {
        let tiled = self.zipped_divide(tiler)?;
        Ok(tiled.slice(<ZShape<L, T> as InnerCoord<C>>::inner_coord(coord)))
    }

    /// CuTe `outer_partition(tensor, tiler, coord)`: element `coord` of every tile.
    ///
    /// The tensor is zipped-divided by `tiler` and the tile mode is sliced at
    /// `coord`, padded like [`Tensor::inner_partition`], keeping the rest modes.
    pub fn outer_partition<T, C>(self, tiler: T, coord: C) -> Result<OuterTile<I, L, T, C>>
    where
        L: ZippedDivide<T>,
        Zipped<L, T>: LayoutShape + SliceAt<Outer<L, T, C>>,
        ZShape<L, T>: OuterCoord<C>,
        I: Advance<<Zipped<L, T> as SliceAt<Outer<L, T, C>>>::Offset>,
    {
        let tiled = self.zipped_divide(tiler)?;
        Ok(tiled.slice(<ZShape<L, T> as OuterCoord<C>>::outer_coord(coord)))
    }

    /// CuTe `local_tile(tensor, tiler, coord)`: the CTA-level tile at `coord`,
    /// the same as [`Tensor::inner_partition`].
    ///
    /// ```
    /// use noodle_cute::coord::X;
    /// use noodle_cute::layout::LayoutLeft;
    /// use noodle_cute::tensor::make_tensor;
    /// use noodle_cute::{DynInt, Int, shape};
    ///
    /// let data: [f32; 96] = core::array::from_fn(|i| i as f32);
    /// let a = make_tensor(data.as_ptr(), LayoutLeft::make(shape![DynInt(12), DynInt(8)])?);
    /// // Tile (4, 8) at block (2, _): rows 8..12, every column tile.
    /// let tile = a.local_tile(shape![Int::<4>, Int::<8>], shape![2i32, X])?;
    /// assert_eq!(unsafe { tile.read(shape![1i32, 3i32, 0i32]) }, (9 + 3 * 12) as f32);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn local_tile<T, C>(self, tiler: T, coord: C) -> Result<InnerTile<I, L, T, C>>
    where
        L: ZippedDivide<T>,
        Zipped<L, T>: LayoutShape + SliceAt<Inner<L, T, C>>,
        ZShape<L, T>: InnerCoord<C>,
        I: Advance<<Zipped<L, T> as SliceAt<Inner<L, T, C>>>::Offset>,
    {
        self.inner_partition(tiler, coord)
    }

    /// CuTe `local_tile(tensor, tiler, coord, proj)`: projects the tiler and the
    /// coordinate with `dice(proj, ...)` first, so one CTA tiler serves A, B, and C.
    ///
    /// ```
    /// use noodle_cute::coord::X;
    /// use noodle_cute::layout::LayoutLeft;
    /// use noodle_cute::tensor::make_tensor;
    /// use noodle_cute::{DynInt, Int, shape};
    ///
    /// let a_data = [0f32; 64 * 32];
    /// let a = make_tensor(a_data.as_ptr(), LayoutLeft::make(shape![DynInt(64), DynInt(32)])?);
    /// let cta_tiler = shape![Int::<16>, Int::<32>, Int::<8>];
    /// let cta_coord = shape![1i32, 0i32, X];
    /// // A uses the (M, K) modes: (BLK_M, BLK_K, k).
    /// let g_a = a.local_tile_proj(cta_tiler, cta_coord, shape![Int::<1>, X, Int::<1>])?;
    /// assert_eq!(g_a.layout.shape, shape![Int::<16>, Int::<8>, DynInt(4)]);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn local_tile_proj<T, C, P>(
        self,
        tiler: T,
        coord: C,
        proj: P,
    ) -> Result<ProjTile<I, L, T, C, P>>
    where
        P: DiceOf<T> + DiceOf<C> + Clone,
        L: ZippedDivide<Diced<P, T>>,
        Zipped<L, Diced<P, T>>: LayoutShape + SliceAt<Inner<L, Diced<P, T>, Diced<P, C>>>,
        ZShape<L, Diced<P, T>>: InnerCoord<Diced<P, C>>,
        I: Advance<<Zipped<L, Diced<P, T>> as SliceAt<Inner<L, Diced<P, T>, Diced<P, C>>>>::Offset>,
    {
        self.local_tile(dice(proj.clone(), &tiler), dice(proj, &coord))
    }

    /// CuTe `local_partition(tensor, thr_layout, index)`: the elements of thread `index`.
    ///
    /// The tensor is divided by the per-mode sizes of `thr_layout`
    /// (`product_each(shape)`) and the tile mode is sliced at
    /// `thr_layout.get_flat_coord(index)`, so thread `index` takes every element
    /// at its position within each tile of threads.
    ///
    /// ```
    /// use noodle_cute::layout::LayoutLeft;
    /// use noodle_cute::tensor::make_tensor;
    /// use noodle_cute::{Int, shape};
    ///
    /// let data: [i32; 16 * 8] = core::array::from_fn(|i| i as i32);
    /// let tile = make_tensor(data.as_ptr(), LayoutLeft::make(shape![Int::<16>, Int::<8>])?);
    /// let threads = LayoutLeft::make(shape![Int::<4>, Int::<2>])?;
    /// // Thread 5 is (1, 1) in the 4x2 thread grid: rows 1, 5, 9, 13 and columns 1, 3, 5, 7.
    /// let mine = tile.local_partition(&threads, 5i32)?;
    /// assert_eq!(mine.layout.shape, shape![Int::<4>, Int::<4>]);
    /// assert_eq!(unsafe { mine.read(shape![2i32, 3i32]) }, 9 + 7 * 16);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn local_partition<TS, TD, X>(
        self,
        thr_layout: &Layout<TS, TD>,
        index: X,
    ) -> Result<Partition<I, L, TS, TD, X>>
    where
        X: Integer + Idx2Crd<TS, TD>,
        TS: ProductEach + RepeatModes<Int<1>> + Clone,
        X::Output: Crd2Crd<TS, <TS as RepeatModes<Int<1>>>::Output>,
        L: ZippedDivide<Each<TS>>,
        Zipped<L, Each<TS>>: LayoutShape + SliceAt<Outer<L, Each<TS>, FlatCoord<X, TS, TD>>>,
        ZShape<L, Each<TS>>: OuterCoord<FlatCoord<X, TS, TD>>,
        I: Advance<
            <Zipped<L, Each<TS>> as SliceAt<Outer<L, Each<TS>, FlatCoord<X, TS, TD>>>>::Offset,
        >,
    {
        self.outer_partition(
            thr_layout.shape.clone().product_each(),
            thr_layout.get_flat_coord(index),
        )
    }

    /// CuTe `local_partition(tensor, thr_layout, index, proj)`: partitions with
    /// `dice(proj, thr_layout)`, so one thread layout serves A, B, and C.
    pub fn local_partition_proj<TS, TD, X, P>(
        self,
        thr_layout: &Layout<TS, TD>,
        index: X,
        proj: P,
    ) -> Result<ProjPartition<I, L, TS, TD, X, P>>
    where
        P: DiceOf<TS> + DiceOf<TD> + Clone,
        X: Integer + Idx2Crd<Diced<P, TS>, Diced<P, TD>>,
        Diced<P, TS>: ProductEach + RepeatModes<Int<1>> + Clone,
        X::Output: Crd2Crd<Diced<P, TS>, <Diced<P, TS> as RepeatModes<Int<1>>>::Output>,
        L: ZippedDivide<Each<Diced<P, TS>>>,
        Zipped<L, Each<Diced<P, TS>>>: LayoutShape + SliceAt<ProjOuter<L, X, P, TS, TD>>,
        ZShape<L, Each<Diced<P, TS>>>: OuterCoord<FlatCoord<X, Diced<P, TS>, Diced<P, TD>>>,
        I: Advance<<Zipped<L, Each<Diced<P, TS>>> as SliceAt<ProjOuter<L, X, P, TS, TD>>>::Offset>,
    {
        self.local_partition(&thr_layout.dice(proj), index)
    }

    /// Reinterprets the elements as `N`: CuTe `recast<N>(tensor)`.
    ///
    /// The layout is rescaled by [`RecastView`]. When `N` is wider and the layout
    /// is plain, the pointer first moves by
    /// `sum(min((old_extent - new_extent) * old_stride, 0))` old elements, as
    /// upstream does. Like upstream this checks neither alignment nor the
    /// divisibility of dynamic strides.
    ///
    /// ```
    /// use noodle_cute::layout::LayoutLeft;
    /// use noodle_cute::tensor::make_tensor;
    /// use noodle_cute::{Int, shape};
    ///
    /// let bytes: [u8; 8] = [1, 0, 0, 0, 2, 0, 0, 0];
    /// let t = make_tensor(bytes.as_ptr(), LayoutLeft::make(shape![Int::<8>])?);
    /// let words = t.recast::<u32>();
    /// assert_eq!(words.layout.shape, shape![Int::<2>]);
    /// assert_eq!(unsafe { words.read(1i32) }, u32::from_le_bytes([2, 0, 0, 0]));
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn recast<N>(self) -> Recast<I, N, L>
    where
        L: RecastView<I::Value, N>,
        I: Advance<DynInt>,
        Adv<I, DynInt>: RecastPtr<N>,
    {
        let (layout, offset) = self.layout.recast_view();
        make_tensor(self.engine.0.advance(DynInt(offset)).recast_ptr(), layout)
    }
}

type ProjOuter<L, X, P, TS, TD> =
    Outer<L, Each<Diced<P, TS>>, FlatCoord<X, Diced<P, TS>, Diced<P, TD>>>;
