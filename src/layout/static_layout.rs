//! Typed affine layouts with static or mixed static/runtime leaves.

use core::fmt::Debug;

use crate::int::{Arith, Int, Integer, Op, Prod};
use crate::tuple::{
    Append, Flatten, Get, Group, Item, LeafProfile, Nil, Select, Take, Tuple, Wrap,
};
use crate::{Bool, Error, IntTuple, Result, StaticValue};

use super::{
    AffineLayoutMap, DynLayout, GroupedLayout, LayoutDesc, LayoutMap, MAX_RANK, Shape, StaticDesc,
    StrideOf, TypedLeaves, address_span, affine_address_range, affine_index, checked_size,
};

/// An affine map defined by a shape and structurally congruent strides.
///
/// The shape and stride profiles must match at the type level, including nested
/// modes. Each leaf contributes `coordinate * stride` to the address. Static
/// leaves occupy no storage; runtime leaves retain their values in the layout.
/// Negative and zero strides are allowed; [`LayoutMap::is_injective`] checks
/// whether distinct logical coordinates can share an address.
///
/// Public fields allow construction without validation, for example in `const`
/// items. Use [`Layout::new`] to check the shape and address arithmetic.
///
/// `repr(C)` fixes the field order across host and device targets so a layout
/// can be passed by value as a kernel parameter; fully static layouts stay
/// zero-sized.
///
/// ```
/// use noodle_cute::{DynInt, Int, shape, stride};
/// use noodle_cute::layout::{Layout, LayoutMap};
///
/// let layout = Layout::new(
///     shape![Int::<2>, DynInt(3)],
///     stride![Int::<1>, Int::<2>],
/// )?;
/// assert_eq!(layout.size()?, 6);
/// assert_eq!(layout.index_of_coord(&[1, 2])?, 5);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Layout<S, D> {
    /// Nonnegative extents arranged as an integer leaf or nested tuple.
    pub shape: S,
    /// Signed address increments with the same nested profile as `shape`.
    pub stride: D,
}

/// Shape/stride projection of typed algebra results, including unvalidated intermediates.
pub trait LayoutParts {
    type Shape;
    type Stride;
    /// Returns the layout value without validation or normalization.
    fn into_layout(self) -> Layout<Self::Shape, Self::Stride>;
}
impl<S, D> LayoutParts for Layout<S, D> {
    type Shape = S;
    type Stride = D;
    fn into_layout(self) -> Self {
        self
    }
}

impl<S: Shape, D: StrideOf<S>> Layout<S, D> {
    /// Returns the checked logical size, including during constant evaluation.
    ///
    /// This agrees with [`LayoutMap::size`]. Constant calls require const `Item`
    /// implementations for the leaves and the caller's `const_trait_impl` feature.
    pub const fn size(&self) -> Result<isize>
    where
        S: [const] Item,
        D: [const] StrideOf<S>,
    {
        checked_size(&TypedLeaves(&self.shape, &self.stride), 0..S::LEAVES)
    }

    /// Returns the checked address span, including during constant evaluation.
    ///
    /// Semantics match [`LayoutMap::cosize`], including negative strides and empty domains.
    ///
    /// ```
    /// #![feature(const_trait_impl)]
    /// use noodle_cute::{DynInt, Int, Layout};
    ///
    /// const L: Layout<DynInt, Int<2>> = Layout {
    ///     shape: DynInt(4),
    ///     stride: Int,
    /// };
    /// const STORAGE: usize = match L.cosize() {
    ///     Ok(n) => n as usize,
    ///     Err(_) => panic!("invalid layout storage span"),
    /// };
    /// let buffer = [0u8; STORAGE];
    /// assert_eq!(buffer.len(), 7);
    /// ```
    pub const fn cosize(&self) -> Result<isize>
    where
        S: [const] Item,
        D: [const] StrideOf<S>,
    {
        address_span(affine_address_range(
            &TypedLeaves(&self.shape, &self.stride),
            0..S::LEAVES,
        )?)
    }

    /// Maps a bounded left-major index, including during constant evaluation.
    ///
    /// Semantics and errors match [`LayoutMap::index`]. Runtime metadata exports
    /// and `is_injective` remain available through the ordinary `LayoutMap` trait.
    pub const fn index(&self, linear: isize) -> Result<isize>
    where
        S: [const] Item,
        D: [const] StrideOf<S>,
    {
        affine_index(
            &TypedLeaves(&self.shape, &self.stride),
            0..S::LEAVES,
            linear,
        )
    }

    /// Converts this layout into a [`LayoutDesc`].
    pub const fn to_desc(&self) -> Result<LayoutDesc>
    where
        S: [const] Item,
        D: [const] StrideOf<S>,
    {
        if S::LEAVES > MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        let mut desc = LayoutDesc::empty();
        let mut i = 0;
        while i < S::LEAVES {
            let s = match self.shape.leaf(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            let d = match self.stride.leaf(i) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            if s < 0 {
                return Err(Error::InvalidShape);
            }
            desc.shapes[i] = s;
            desc.strides[i] = d;
            i += 1;
        }
        desc.rank = S::LEAVES;
        Ok(desc)
    }

    /// Evaluates CuTe composition `self ∘ inner` on layout descriptors at compile time.
    pub const fn compose_desc<S2, D2>(&self, inner: &Layout<S2, D2>) -> Result<LayoutDesc>
    where
        S: [const] Item,
        D: [const] StrideOf<S>,
        S2: Shape + [const] Item,
        D2: [const] StrideOf<S2>,
    {
        let outer_desc = match self.to_desc() {
            Ok(d) => d,
            Err(e) => return Err(e),
        };
        let inner_desc = match inner.to_desc() {
            Ok(d) => d,
            Err(e) => return Err(e),
        };
        crate::algebra::compose_desc(&outer_desc, &inner_desc)
    }

    /// Evaluates CuTe complement on layout descriptors at compile time.
    pub const fn complement_desc<const C: isize>(&self) -> Result<LayoutDesc>
    where
        S: [const] Item,
        D: [const] StrideOf<S>,
    {
        let desc = match self.to_desc() {
            Ok(d) => d,
            Err(e) => return Err(e),
        };
        crate::algebra::complement_desc(&desc, &[C], crate::algebra::ComplementMode::Standard)
    }

    /// Evaluates CuTe extended complement on layout descriptors at compile time.
    pub const fn complement_extended_desc(&self) -> Result<LayoutDesc>
    where
        S: [const] Item,
        D: [const] StrideOf<S>,
    {
        let desc = match self.to_desc() {
            Ok(d) => d,
            Err(e) => return Err(e),
        };
        crate::algebra::complement_desc(&desc, &[], crate::algebra::ComplementMode::Extended)
    }

    /// Constructs a layout after checking nonnegative extents, logical size,
    /// and reachable address endpoints with checked arithmetic.
    ///
    /// Negative extents return [`Error::InvalidShape`]; unrepresentable size or
    /// address calculations return [`Error::Overflow`]. Addresses may be negative
    /// or repeated, and an empty domain has no address endpoints to check.
    pub fn new(shape: S, stride: D) -> Result<Self> {
        let layout = Self { shape, stride };
        layout.size()?;
        layout.address_range()?;
        Ok(layout)
    }

    /// Flattens nested modes in leaf order without changing the linear address map.
    ///
    /// Static leaf values remain encoded in the result type; flattening neither
    /// merges contiguous leaves nor removes unit extents. An integral layout such as
    /// `4:1` becomes the single-mode tuple `(4):(1)`, whereas CuTe's `flatten`
    /// keeps it integral; see [`Flatten`].
    pub fn flatten(self) -> Layout<<S as Flatten>::Output, <D as Flatten>::Output>
    where
        S: Flatten,
        D: Flatten,
    {
        Layout {
            shape: self.shape.flatten(),
            stride: self.stride.flatten(),
        }
    }
}

pub(super) type Wrapped<T> = <T as Wrap>::Output;
pub(super) type Taken<T, const B: usize, const E: usize> = <Wrapped<T> as Take<B, E>>::Output;
pub(super) type Grouped<T, const B: usize, const E: usize> = <Wrapped<T> as Group<B, E>>::Output;
pub(super) type Appended<T, X> = <Wrapped<T> as Append<IntTuple<X, Nil>>>::Output;

/// Mode operations from CuTe `layout.hpp`. They rearrange shape and stride
/// together without validating the result; use [`Layout::new`] when a checked
/// layout is needed. An integral layout acts as a rank-1 tuple, as upstream.
impl<S, D> Layout<S, D> {
    /// Returns top-level mode `I` as a layout: CuTe `get<I>(layout)`.
    ///
    /// Chain calls to reach nested modes. Per-mode queries such as `size<I>` are
    /// the existing queries on this result, for example `layout.get::<1>().size()`.
    ///
    /// ```
    /// use noodle_cute::{Int, Layout, shape, stride};
    ///
    /// let layout = Layout::new(
    ///     shape![Int::<2>, (Int::<3>, Int::<4>)],
    ///     stride![Int::<12>, (Int::<1>, Int::<3>)],
    /// )?;
    /// let inner = layout.get::<1>().get::<1>();
    /// assert_eq!(inner, Layout { shape: Int::<4>, stride: Int::<3> });
    /// assert_eq!(layout.get::<1>().size()?, 12);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn get<const I: usize>(&self) -> Layout<<S as Get<I>>::Output, <D as Get<I>>::Output>
    where
        S: Get<I>,
        D: Get<I>,
        <S as Get<I>>::Output: Clone,
        <D as Get<I>>::Output: Clone,
    {
        Layout {
            shape: Get::<I>::get(&self.shape).clone(),
            stride: Get::<I>::get(&self.stride).clone(),
        }
    }

    /// Keeps the top-level modes in `[B, E)`: CuTe `take<B, E>(layout)`.
    ///
    /// As upstream, the range must be nonempty (`B < E`) and within the rank;
    /// otherwise compilation fails:
    ///
    /// ```compile_fail
    /// use noodle_cute::{Int, Layout, shape, stride};
    /// let layout = Layout::new(shape![Int::<2>, Int::<3>], stride![Int::<1>, Int::<2>]).unwrap();
    /// let _ = layout.take::<1, 1>();
    /// ```
    pub fn take<const B: usize, const E: usize>(&self) -> Layout<Taken<S, B, E>, Taken<D, B, E>>
    where
        S: Wrap + Clone,
        D: Wrap + Clone,
        Wrapped<S>: Take<B, E>,
        Wrapped<D>: Take<B, E>,
    {
        const { assert!(B < E, "take: empty range") };
        Layout {
            shape: self.shape.clone().wrap().take(),
            stride: self.stride.clone().wrap().take(),
        }
    }

    /// Keeps the top-level modes named by a tuple of `Int<N>`: CuTe `select<I...>(layout)`.
    ///
    /// ```
    /// use noodle_cute::{Int, Layout, int_tuple, shape, stride};
    ///
    /// let layout = Layout::new(shape![Int::<2>, Int::<3>, Int::<4>], stride![Int::<1>, Int::<2>, Int::<6>])?;
    /// let picked = layout.select(int_tuple![Int::<2>, Int::<0>]);
    /// assert_eq!(picked, Layout { shape: shape![Int::<4>, Int::<2>], stride: stride![Int::<6>, Int::<1>] });
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn select<I: Clone>(
        &self,
        indices: I,
    ) -> Layout<<S as Select<I>>::Output, <D as Select<I>>::Output>
    where
        S: Select<I>,
        D: Select<I>,
    {
        Layout {
            shape: self.shape.select(indices.clone()),
            stride: self.stride.select(indices),
        }
    }

    /// Groups the top-level modes in `[B, E)` into one mode: CuTe `group<B, E>(layout)`.
    pub fn group<const B: usize, const E: usize>(
        &self,
    ) -> Layout<Grouped<S, B, E>, Grouped<D, B, E>>
    where
        S: Wrap + Clone,
        D: Wrap + Clone,
        Wrapped<S>: Group<B, E>,
        Wrapped<D>: Group<B, E>,
    {
        Layout {
            shape: self.shape.clone().wrap().group(),
            stride: self.stride.clone().wrap().group(),
        }
    }

    /// Adds `mode` as a new last top-level mode: CuTe `append(layout, x)`.
    ///
    /// ```
    /// use noodle_cute::{Int, Layout, shape, stride};
    ///
    /// let rows = Layout::new(Int::<4>, Int::<1>)?;
    /// let tile = rows.append(Layout::new(Int::<8>, Int::<4>)?);
    /// assert_eq!(tile, Layout { shape: shape![Int::<4>, Int::<8>], stride: stride![Int::<1>, Int::<4>] });
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn append<S2, D2>(&self, mode: Layout<S2, D2>) -> Layout<Appended<S, S2>, Appended<D, D2>>
    where
        S: Wrap + Clone,
        D: Wrap + Clone,
        Wrapped<S>: Append<IntTuple<S2, Nil>>,
        Wrapped<D>: Append<IntTuple<D2, Nil>>,
    {
        Layout {
            shape: self.shape.clone().wrap().append(IntTuple(mode.shape, Nil)),
            stride: self
                .stride
                .clone()
                .wrap()
                .append(IntTuple(mode.stride, Nil)),
        }
    }

    /// Adds `mode` as a new first top-level mode: CuTe `prepend(layout, x)`.
    pub fn prepend<S2, D2>(
        &self,
        mode: Layout<S2, D2>,
    ) -> Layout<IntTuple<S2, Wrapped<S>>, IntTuple<D2, Wrapped<D>>>
    where
        S: Wrap + Clone,
        D: Wrap + Clone,
    {
        Layout {
            shape: IntTuple(mode.shape, self.shape.clone().wrap()),
            stride: IntTuple(mode.stride, self.stride.clone().wrap()),
        }
    }
}

/// Transposes a tuple of layouts into a layout of tuples; used by [`make_layout!`](crate::make_layout).
#[doc(hidden)]
pub trait Concat {
    type Shape;
    type Stride;
    fn concat(self) -> Layout<Self::Shape, Self::Stride>;
}

impl Concat for Nil {
    type Shape = Nil;
    type Stride = Nil;
    fn concat(self) -> Layout<Nil, Nil> {
        Layout {
            shape: Nil,
            stride: Nil,
        }
    }
}

impl<S, D, T: Concat> Concat for IntTuple<Layout<S, D>, T> {
    type Shape = IntTuple<S, T::Shape>;
    type Stride = IntTuple<D, T::Stride>;
    fn concat(self) -> Layout<Self::Shape, Self::Stride> {
        let tail = self.1.concat();
        Layout {
            shape: IntTuple(self.0.shape, tail.shape),
            stride: IntTuple(self.0.stride, tail.stride),
        }
    }
}

impl<S: Shape, D: StrideOf<S>> AffineLayoutMap for Layout<S, D> {
    fn affine_rank(&self) -> usize {
        S::RANK
    }

    fn leaf_shape(&self, leaf: usize) -> Result<isize> {
        self.shape.leaf(leaf)
    }

    fn leaf_stride(&self, leaf: usize) -> Result<isize> {
        self.stride.leaf(leaf)
    }

    fn leaf_count(&self) -> usize {
        S::LEAVES
    }

    fn mode_span(&self, axis: usize) -> Result<(usize, usize)> {
        self.shape.mode_span(axis)
    }

    /// Exports flattened runtime metadata for reflection or backend parameters.
    ///
    /// The result has one axis per leaf and discards the original mode grouping.
    /// Returns [`Error::CapacityExceeded`] if the leaf count exceeds [`MAX_RANK`].
    /// The exported values are validated by [`DynLayout::new`].
    fn as_flat_layout(&self) -> Result<DynLayout> {
        if S::LEAVES > MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        let mut shape = [0; MAX_RANK];
        let mut stride = [0; MAX_RANK];
        for i in 0..S::LEAVES {
            shape[i] = self.shape.leaf(i)?;
            stride[i] = self.stride.leaf(i)?;
        }
        DynLayout::new(&shape[..S::LEAVES], &stride[..S::LEAVES])
    }

    fn as_grouped_layout(&self) -> Result<GroupedLayout> {
        if S::RANK > MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        let mut groups = [0usize; MAX_RANK];
        for (axis, g) in groups[..S::RANK].iter_mut().enumerate() {
            *g = self.shape.mode_span(axis)?.1;
        }
        let mut grouped = GroupedLayout::new(self.as_flat_layout()?, &groups[..S::RANK])?;
        // A leaf type (depth zero) represents a CuTe integral shape, distinct
        // from a one-element tuple even though both have rank one.
        grouped.integral = S::DEPTH == 0;
        Ok(grouped)
    }
}

//-----------------------------------------------------------------------------
// Compact stride construction.

/// Constructs typed compact layouts with the leftmost leaf varying fastest.
/// Also serves as the direction marker for [`Compact`].
#[derive(Clone, Copy, Debug)]
pub struct LayoutLeft;

/// Constructs typed compact layouts with the rightmost leaf varying fastest.
/// Also serves as the direction marker for [`Compact`].
#[derive(Clone, Copy, Debug)]
pub struct LayoutRight;

/// Builds compact strides while preserving tuple structure and static arithmetic.
///
/// `Dir` is [`LayoutLeft`] or [`LayoutRight`] and determines the order in which
/// leaf extents multiply the running stride. The result is
/// `(stride_tree, start * product_of_extents)`. Leaves combine with the rules of
/// [`Integer`], so a static zero extent keeps the following strides static, and a
/// static unit extent takes stride `Int<0>`, as in CuTe's `compact_major`. Each
/// product is checked in its result type and
/// returns [`Error::Overflow`] when it does not fit; this also covers `i32`
/// strides. Use the direction marker's `make` method for validated layouts.
pub trait Compact<Dir, Start: Integer>: Item {
    /// Stride tree with the same nested structure as the shape.
    type Stride: Item<Profile = Self::Profile>;
    /// Running stride after all extents have been multiplied into `Start`.
    type End: Integer;
    /// Assigns strides beginning at `start` and returns the final running stride.
    fn compact(self, start: Start) -> Result<(Self::Stride, Self::End)>;
}

const STATIC_UNIT<I: Integer>: bool = matches!(I::STATIC, Some(1));

mod detail {
    use super::{Arith, Error, Int, Integer, Item, LeafProfile, Op, Prod, Result};
    use crate::Bool;

    /// One leaf of `compact`; a static unit extent takes stride `_0`, as upstream.
    pub trait CompactLeaf<I, Start> {
        type Stride: Item<Profile = LeafProfile>;
        type End: Integer;
        fn compact_leaf(extent: I, start: Start) -> Result<(Self::Stride, Self::End)>;
    }

    impl<I, Start: Integer + core::fmt::Debug> CompactLeaf<I, Start> for Bool<true> {
        type Stride = Int<0>;
        type End = Start;
        fn compact_leaf(_: I, start: Start) -> Result<(Int<0>, Start)> {
            Ok((Int, start))
        }
    }

    impl<I, Start> CompactLeaf<I, Start> for Bool<false>
    where
        I: Arith<Start, { Op::Mul }>,
        Start: Integer + core::fmt::Debug,
    {
        type Stride = Start;
        type End = Prod<I, Start>;
        fn compact_leaf(extent: I, start: Start) -> Result<(Start, Self::End)> {
            Ok((start, extent.checked(start).ok_or(Error::Overflow)?))
        }
    }
}
use detail::CompactLeaf;

// Leaves and empty tuples are direction-independent; only `IntTuple` changes
// its recursive evaluation order with the direction marker.
impl<I, Dir, Start> Compact<Dir, Start> for I
where
    I: Integer + Debug,
    Start: Integer + Debug,
    Bool<{ STATIC_UNIT::<I> }>: CompactLeaf<I, Start>,
{
    type Stride = <Bool<{ STATIC_UNIT::<I> }> as CompactLeaf<I, Start>>::Stride;
    type End = <Bool<{ STATIC_UNIT::<I> }> as CompactLeaf<I, Start>>::End;
    fn compact(self, start: Start) -> Result<(Self::Stride, Self::End)> {
        <Bool<{ STATIC_UNIT::<I> }> as CompactLeaf<I, Start>>::compact_leaf(self, start)
    }
}

impl<Dir, Start: Integer> Compact<Dir, Start> for Nil {
    type Stride = Nil;
    type End = Start;
    fn compact(self, start: Start) -> Result<(Nil, Start)> {
        Ok((Nil, start))
    }
}

impl<H, T, Start> Compact<LayoutLeft, Start> for IntTuple<H, T>
where
    Start: Integer,
    H: Compact<LayoutLeft, Start>,
    T: Tuple + Compact<LayoutLeft, H::End>,
    T::Stride: Tuple,
{
    type Stride = IntTuple<H::Stride, T::Stride>;
    type End = T::End;
    fn compact(self, start: Start) -> Result<(Self::Stride, Self::End)> {
        let (h, end) = self.0.compact(start)?;
        let (t, end) = self.1.compact(end)?;
        Ok((IntTuple(h, t), end))
    }
}

impl<H, T, Start> Compact<LayoutRight, Start> for IntTuple<H, T>
where
    Start: Integer,
    T: Tuple + Compact<LayoutRight, Start>,
    H: Compact<LayoutRight, T::End>,
    T::Stride: Tuple,
{
    type Stride = IntTuple<H::Stride, T::Stride>;
    type End = H::End;
    fn compact(self, start: Start) -> Result<(Self::Stride, Self::End)> {
        let (t, end) = self.1.compact(start)?;
        let (h, end) = self.0.compact(end)?;
        Ok((IntTuple(h, t), end))
    }
}

impl LayoutLeft {
    /// Builds a compact layout with left-to-right cumulative strides.
    ///
    /// Validates extents and partial products before constructing strides,
    /// retaining static values in their types. The first leaf has stride one.
    ///
    /// ```
    /// use noodle_cute::{Int, shape};
    /// use noodle_cute::layout::{AffineLayoutMap, LayoutLeft, LayoutMap};
    ///
    /// let layout = LayoutLeft::make(shape![Int::<2>, (Int::<3>, Int::<4>)])?;
    /// assert_eq!(layout.rank(), 2);
    /// assert_eq!(layout.extent(1)?, 12);
    /// assert_eq!(layout.as_flat_layout()?.strides(), &[1, 2, 6]);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn make<S: Shape + Compact<Self, Int<1>>>(shape: S) -> Result<Layout<S, S::Stride>> {
        make_compact(shape)
    }
}

impl LayoutRight {
    /// Builds a compact layout with right-to-left cumulative strides.
    ///
    /// Checks extents and partial products for overflow. The last leaf has
    /// stride one; logical linear indices still use left-major order.
    pub fn make<S: Shape + Compact<Self, Int<1>>>(shape: S) -> Result<Layout<S, S::Stride>> {
        make_compact(shape)
    }
}

// Reject negative extents first; `compact` then checks the partial products in
// leaf order, so overflow before a zero extent returns an error even when the
// final logical size is zero.
fn make_compact<Dir, S: Shape + Compact<Dir, Int<1>>>(shape: S) -> Result<Layout<S, S::Stride>> {
    for i in 0..S::LEAVES {
        if shape.leaf(i)? < 0 {
            return Err(Error::InvalidShape);
        }
    }
    let (stride, _) = shape.clone().compact(Int)?;
    Layout::new(shape, stride)
}

impl<S, D> StaticDesc for Layout<S, D>
where
    Self: StaticValue,
    S: Shape + const Item,
    D: const StrideOf<S>,
{
    const DESC: LayoutDesc = match Self::VALUE.to_desc() {
        Ok(desc) => desc,
        Err(_) => panic!("invalid static layout descriptor"),
    };
}
