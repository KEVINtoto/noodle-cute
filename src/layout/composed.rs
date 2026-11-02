//! A function composed after an offset and a layout: CuTe `ComposedLayout<A, O, B>`
//! (`layout_composed.hpp`), and the layout operations tensors use for both kinds.
//!
//! `ComposedLayout { layout_a, offset, layout_b }` maps a coordinate `c` to
//! `layout_a(offset + layout_b(c))`. With a [`Swizzle`](crate::swizzle::Swizzle)
//! as `layout_a` it is a swizzled shared-memory layout. Queries, mode
//! operations, slicing, and the layout algebra act on `layout_b`; slicing and
//! domain offsets accumulate into
//! `offset` and leave the tensor's pointer unchanged, as upstream does for
//! composed layouts.
//!
//! ```
//! use noodle_cute::layout::{ComposedLayout, MapCoord, LayoutLeft};
//! use noodle_cute::swizzle::Swizzle;
//! use noodle_cute::{Int, shape};
//!
//! let smem = Swizzle::<3, 3, 3>.compose(LayoutLeft::make(shape![Int::<64>, Int::<8>])?);
//! // Row 0 of column 1 is address 64; its bit 6 flips bit 3.
//! assert_eq!(smem.map_coord(shape![0i32, 1i32]), 72);
//! # Ok::<(), noodle_cute::Error>(())
//! ```

use crate::basis::OffsetAdd;
use crate::coord::{Crd2Idx, SliceOf};
use crate::int::Int;
use crate::swizzle::{MERGED_Y, MERGED_Z, MakeSwizzle};
use crate::swizzle::{Swizzle, Swizzled};
use crate::{Bool, Result};

use crate::tuple::{Append, Flatten, Get, Group, Item, Take, Wrap};

use super::static_layout::{Appended, Grouped, Taken, Wrapped};
use super::{Layout, Shape, StrideOf};

/// `layout_a ∘ offset ∘ layout_b`: CuTe `ComposedLayout<A, O, B>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ComposedLayout<A, O, B> {
    /// The outer function, such as a [`Swizzle`].
    pub layout_a: A,
    /// The offset added before `layout_a`.
    pub offset: O,
    /// The inner layout, which defines the shape.
    pub layout_b: B,
}

/// CuTe `make_composed_layout(a, o, b)` and the three-argument `composition`.
pub fn make_composed_layout<A, O, B>(
    layout_a: A,
    offset: O,
    layout_b: B,
) -> ComposedLayout<A, O, B> {
    ComposedLayout {
        layout_a,
        offset,
        layout_b,
    }
}

// Implementation traits that appear in public bounds but are not part of the API.
mod detail {
    /// The right-hand side of `composition(swizzle, rhs)`.
    pub trait ComposeSwizzle<R> {
        type Output;
        fn compose_swizzle(self, rhs: R) -> Self::Output;
    }

    /// `composition(swizzle, 0, layout)`, which drops a zero-bit swizzle.
    pub trait IdentityKind<Sw, L> {
        type Output;
        fn compose(swizzle: Sw, layout: L) -> Self::Output;
    }
}
pub(crate) use detail::ComposeSwizzle;
use detail::IdentityKind;

const IS_IDENTITY<const B: u32>: bool = B == 0;

impl<Sw, L> IdentityKind<Sw, L> for Bool<true> {
    type Output = L;
    fn compose(_: Sw, layout: L) -> L {
        layout
    }
}

impl<Sw, L> IdentityKind<Sw, L> for Bool<false> {
    type Output = ComposedLayout<Sw, Int<0>, L>;
    fn compose(swizzle: Sw, layout: L) -> Self::Output {
        make_composed_layout(swizzle, Int, layout)
    }
}

impl<const B: u32, const M: u32, const S: i32, LS, LD> ComposeSwizzle<Layout<LS, LD>>
    for Swizzle<B, M, S>
where
    Bool<{ IS_IDENTITY::<B> }>: IdentityKind<Self, Layout<LS, LD>>,
{
    type Output = <Bool<{ IS_IDENTITY::<B> }> as IdentityKind<Self, Layout<LS, LD>>>::Output;
    fn compose_swizzle(self, layout: Layout<LS, LD>) -> Self::Output {
        <Bool<{ IS_IDENTITY::<B> }> as IdentityKind<_, _>>::compose(self, layout)
    }
}

type Merged<
    const B0: u32,
    const M0: u32,
    const S0: i32,
    const B1: u32,
    const M1: u32,
    const S1: i32,
> = MakeSwizzle<{ MERGED_Y::<B0, M0, S0, B1, M1, S1> }, { MERGED_Z::<B0, M0, S0, B1, M1, S1> }>;

impl<const B0: u32, const M0: u32, const S0: i32, const B1: u32, const M1: u32, const S1: i32>
    ComposeSwizzle<Swizzle<B1, M1, S1>> for Swizzle<B0, M0, S0>
{
    type Output = Merged<B0, M0, S0, B1, M1, S1>;
    fn compose_swizzle(self, _: Swizzle<B1, M1, S1>) -> Self::Output {
        Swizzle
    }
}

impl<const B: u32, const M: u32, const S: i32> Swizzle<B, M, S> {
    /// CuTe `composition(swizzle, rhs)`.
    ///
    /// With a layout this is `swizzle ∘ 0 ∘ layout`, or the layout itself for a
    /// zero-bit swizzle. With another swizzle of the same shift `S` the two
    /// merge into one swizzle whose masks are the XOR of theirs, as
    /// `make_swizzle`; a different shift fails to compile.
    ///
    /// ```
    /// use noodle_cute::layout::LayoutLeft;
    /// use noodle_cute::swizzle::Swizzle;
    /// use noodle_cute::{Int, shape};
    ///
    /// let layout = LayoutLeft::make(shape![Int::<8>, Int::<8>])?;
    /// assert_eq!(Swizzle::<0, 3, 3>.compose(layout), layout);
    /// assert_eq!(Swizzle::<1, 3, 3>.compose(Swizzle::<1, 4, 3>), Swizzle::<2, 3, 3>);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn compose<R>(self, rhs: R) -> <Self as ComposeSwizzle<R>>::Output
    where
        Self: ComposeSwizzle<R>,
    {
        self.compose_swizzle(rhs)
    }
}

/// The outer function of a composed layout applied to an offset.
pub trait OuterFn<X> {
    /// The resulting address.
    type Output;
    /// Applies the function.
    fn apply(&self, offset: X) -> Self::Output;
}

impl<const B: u32, const M: u32, const S: i32, X: Swizzled<B, M, S>> OuterFn<X>
    for Swizzle<B, M, S>
{
    type Output = X::Output;
    fn apply(&self, offset: X) -> X::Output {
        offset.swizzled()
    }
}

impl<S, D, X: Crd2Idx<S, D>> OuterFn<X> for Layout<S, D> {
    type Output = X::Output;
    fn apply(&self, offset: X) -> X::Output {
        self.crd2idx(offset)
    }
}

/// Maps a coordinate to an offset: `layout(coord)` for either layout kind.
pub trait MapCoord<C> {
    /// The offset.
    type Output;
    /// Evaluates the layout at `coord` without checks.
    fn map_coord(&self, coord: C) -> Self::Output;
}

impl<S, D, C: Crd2Idx<S, D>> MapCoord<C> for Layout<S, D> {
    type Output = C::Output;
    fn map_coord(&self, coord: C) -> C::Output {
        self.crd2idx(coord)
    }
}

type Inner<B, C> = <B as MapCoord<C>>::Output;
type Shifted<O, B, C> = <O as OffsetAdd<Inner<B, C>>>::Output;

impl<A, O, B, C> MapCoord<C> for ComposedLayout<A, O, B>
where
    B: MapCoord<C>,
    O: Copy + OffsetAdd<Inner<B, C>>,
    A: OuterFn<Shifted<O, B, C>>,
{
    type Output = A::Output;
    fn map_coord(&self, coord: C) -> A::Output {
        self.layout_a
            .apply(self.offset.offset_add(self.layout_b.map_coord(coord)))
    }
}

/// The shape of either layout kind; a composed layout has the shape of `layout_b`.
pub trait LayoutShape {
    /// The shape.
    type Shape;
    /// Returns the shape.
    fn shape(&self) -> Self::Shape;
}

impl<S: Clone, D> LayoutShape for Layout<S, D> {
    type Shape = S;
    fn shape(&self) -> S {
        self.shape.clone()
    }
}

impl<A, O, B: LayoutShape> LayoutShape for ComposedLayout<A, O, B> {
    type Shape = B::Shape;
    fn shape(&self) -> B::Shape {
        self.layout_b.shape()
    }
}

/// CuTe `slice_and_offset(coord, layout)` for either layout kind.
///
/// A plain layout returns the sliced layout and the origin's address. A
/// composed layout slices `layout_b`, adds that address to its own `offset`,
/// and returns `Int<0>`, as upstream's generic `ComposedLayout` overload does.
/// Upstream additionally specializes swizzles to decay slices into plain
/// layouts where possible; that optimization is not implemented, and the
/// addresses are the same.
pub trait SliceAt<C> {
    /// The kept modes.
    type Sliced;
    /// The offset to add to the tensor's iterator.
    type Offset;
    /// Slices the layout.
    fn slice_and_offset(&self, coord: C) -> (Self::Sliced, Self::Offset);
}

impl<S, D, C> SliceAt<C> for Layout<S, D>
where
    C: SliceOf<S> + SliceOf<D> + Crd2Idx<S, D> + Clone,
    S: Clone,
    D: Clone,
{
    type Sliced = Layout<<C as SliceOf<S>>::Output, <C as SliceOf<D>>::Output>;
    type Offset = <C as Crd2Idx<S, D>>::Output;
    fn slice_and_offset(&self, coord: C) -> (Self::Sliced, Self::Offset) {
        Layout::slice_and_offset(self, coord)
    }
}

impl<A: Clone, O, B, C> SliceAt<C> for ComposedLayout<A, O, B>
where
    B: SliceAt<C>,
    O: Copy + OffsetAdd<B::Offset>,
{
    type Sliced = ComposedLayout<A, O::Output, B::Sliced>;
    type Offset = Int<0>;
    fn slice_and_offset(&self, coord: C) -> (Self::Sliced, Int<0>) {
        let (sliced, offset) = self.layout_b.slice_and_offset(coord);
        (
            make_composed_layout(
                self.layout_a.clone(),
                self.offset.offset_add(offset),
                sliced,
            ),
            Int,
        )
    }
}

/// CuTe `domain_offset(coord, layout)` for either layout kind.
///
/// A plain layout is unchanged and returns `layout(coord)` for the iterator; a
/// composed layout adds `layout_b(coord)` to its own offset and returns `Int<0>`.
pub trait DomainOffset<C> {
    /// The layout after the move.
    type Moved;
    /// The offset to add to the tensor's iterator.
    type Offset;
    /// Moves the origin to `coord`.
    fn domain_offset(&self, coord: C) -> (Self::Moved, Self::Offset);
}

impl<S: Clone, D: Clone, C: Crd2Idx<S, D>> DomainOffset<C> for Layout<S, D> {
    type Moved = Self;
    type Offset = C::Output;
    fn domain_offset(&self, coord: C) -> (Self, C::Output) {
        (self.clone(), self.crd2idx(coord))
    }
}

impl<A: Clone, O, B: Clone, C> DomainOffset<C> for ComposedLayout<A, O, B>
where
    B: MapCoord<C>,
    O: Copy + OffsetAdd<Inner<B, C>>,
{
    type Moved = ComposedLayout<A, Shifted<O, B, C>, B>;
    type Offset = Int<0>;
    fn domain_offset(&self, coord: C) -> (Self::Moved, Int<0>) {
        let offset = self.offset.offset_add(self.layout_b.map_coord(coord));
        (
            make_composed_layout(self.layout_a.clone(), offset, self.layout_b.clone()),
            Int,
        )
    }
}

/// Shape-preserving algebra on `layout_b`, CuTe's `composition(a, o, op(b))` overloads.
impl<A: Clone, O: Clone, B> ComposedLayout<A, O, B> {
    /// Applies `op` to `layout_b`, keeping the function and the offset.
    pub fn map_b<B2>(&self, op: impl FnOnce(&B) -> Result<B2>) -> Result<ComposedLayout<A, O, B2>> {
        Ok(make_composed_layout(
            self.layout_a.clone(),
            self.offset.clone(),
            op(&self.layout_b)?,
        ))
    }
}

type Mode<T, const I: usize> = <T as Get<I>>::Output;
type AppendedLayout<S, D, S2, D2> = Layout<Appended<S, S2>, Appended<D, D2>>;

/// Queries and mode operations on `layout_b`, keeping the function and the
/// offset: CuTe's `ComposedLayout` overloads in `layout_composed.hpp`.
impl<A: Clone, O: Clone, S, D> ComposedLayout<A, O, Layout<S, D>> {
    /// CuTe `size(layout)`: the size of `layout_b`.
    pub fn size(&self) -> Result<isize>
    where
        S: Shape,
        D: StrideOf<S>,
    {
        self.layout_b.size()
    }

    /// CuTe `cosize(layout)`: the address span of `layout_b`, as upstream,
    /// which for a swizzle is also the span of the swizzled addresses.
    pub fn cosize(&self) -> Result<isize>
    where
        S: Shape,
        D: StrideOf<S>,
    {
        self.layout_b.cosize()
    }

    /// CuTe `rank(layout)`: the number of top-level modes of `layout_b`.
    pub fn rank(&self) -> usize
    where
        S: Item,
    {
        S::RANK
    }

    /// CuTe `depth(layout)`: the nesting depth of `layout_b`.
    pub fn depth(&self) -> usize
    where
        S: Item,
    {
        S::DEPTH
    }

    fn with_b<S2, D2>(&self, layout_b: Layout<S2, D2>) -> ComposedLayout<A, O, Layout<S2, D2>> {
        make_composed_layout(self.layout_a.clone(), self.offset.clone(), layout_b)
    }

    /// CuTe `get<I>(layout)`: mode `I` of `layout_b` under the same function and offset.
    pub fn get<const I: usize>(&self) -> ComposedLayout<A, O, Layout<Mode<S, I>, Mode<D, I>>>
    where
        S: Get<I>,
        D: Get<I>,
        Mode<S, I>: Clone,
        Mode<D, I>: Clone,
    {
        self.with_b(self.layout_b.get::<I>())
    }

    /// CuTe `take<B, E>(layout)`; see [`Layout::take`].
    pub fn take<const B: usize, const E: usize>(
        &self,
    ) -> ComposedLayout<A, O, Layout<Taken<S, B, E>, Taken<D, B, E>>>
    where
        S: Wrap + Clone,
        D: Wrap + Clone,
        Wrapped<S>: Take<B, E>,
        Wrapped<D>: Take<B, E>,
    {
        self.with_b(self.layout_b.take::<B, E>())
    }

    /// CuTe `flatten(layout)`; see [`Layout::flatten`].
    pub fn flatten(
        &self,
    ) -> ComposedLayout<A, O, Layout<<S as Flatten>::Output, <D as Flatten>::Output>>
    where
        S: Shape + Flatten + Clone,
        D: StrideOf<S> + Flatten + Clone,
    {
        self.with_b(self.layout_b.clone().flatten())
    }

    /// CuTe `append(layout, mode)`; see [`Layout::append`].
    pub fn append<S2, D2>(
        &self,
        mode: Layout<S2, D2>,
    ) -> ComposedLayout<A, O, AppendedLayout<S, D, S2, D2>>
    where
        S: Wrap + Clone,
        D: Wrap + Clone,
        Wrapped<S>: Append<crate::IntTuple<S2, crate::Nil>>,
        Wrapped<D>: Append<crate::IntTuple<D2, crate::Nil>>,
    {
        self.with_b(self.layout_b.append(mode))
    }

    /// CuTe `group<B, E>(layout)`; see [`Layout::group`].
    pub fn group<const B: usize, const E: usize>(
        &self,
    ) -> ComposedLayout<A, O, Layout<Grouped<S, B, E>, Grouped<D, B, E>>>
    where
        S: Wrap + Clone,
        D: Wrap + Clone,
        Wrapped<S>: Group<B, E>,
        Wrapped<D>: Group<B, E>,
    {
        self.with_b(self.layout_b.group::<B, E>())
    }
}
