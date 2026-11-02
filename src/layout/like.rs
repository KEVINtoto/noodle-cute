//! Compact layouts ordered like a reference layout: CuTe `compact_order`,
//! `make_ordered_layout`, `make_layout_like`, and `make_fragment_like`
//! (`stride.hpp`, `layout.hpp`).
//!
//! These build the static layouts of owning tensors, so every shape leaf must be
//! static; a dynamic extent fails to compile. Reference strides may be dynamic:
//! as upstream, dynamic order values are treated as larger than every static one
//! and ordered from left to right. Upstream also accepts dynamic shapes here;
//! this crate does not, because the results only size register storage.

use core::marker::ConstParamTy;

use crate::algebra::{ProductKind, TypedProduct, TypedProductLayout};
use crate::int::{Int, Integer};
use crate::tuple::{IntTuple, Item, Nil};
use crate::{Bool, StaticValue};

use crate::swizzle::Swizzle;

use super::{ComposedLayout, Layout, MAX_RANK};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum Order {
    /// `compact_order(filter_zeros(stride, shape), stride)`.
    LikeStride,
    /// `compact_order(shape, stride)`.
    ByStride,
    /// `compact_major<LayoutLeft>(filter_zeros(stride, shape))`.
    LeftFiltered,
    /// `compact_major<LayoutLeft>(shape)`.
    Left,
}

// Implementation traits that appear in public bounds but are not part of the API.
mod detail {
    use super::{MAX_RANK, Order};

    /// The static values of a flattened tree, `None` for runtime leaves.
    pub trait StaticLeaves {
        const STATICS: [Option<isize>; MAX_RANK];
        const COUNT: usize;
    }

    /// Strides of a shape subtree whose first leaf is flat leaf `POS` of `RS`.
    pub trait Ordered<RS, RD, const K: Order, const POS: usize> {
        type Output;
        fn ordered() -> Self::Output;
    }

    pub trait FragmentKind<S, D> {
        type Output;
    }
}
pub(crate) use detail::StaticLeaves;
use detail::{FragmentKind, Ordered};

const fn concat(
    head: [Option<isize>; MAX_RANK],
    head_count: usize,
    tail: [Option<isize>; MAX_RANK],
    tail_count: usize,
) -> [Option<isize>; MAX_RANK] {
    assert!(
        head_count + tail_count <= MAX_RANK,
        "owning layouts are limited to MAX_RANK leaves"
    );
    let mut out = head;
    let mut i = 0;
    while i < tail_count {
        out[head_count + i] = tail[i];
        i += 1;
    }
    out
}

impl<I: Integer> StaticLeaves for I {
    const STATICS: [Option<isize>; MAX_RANK] = {
        let mut out = [None; MAX_RANK];
        out[0] = I::STATIC;
        out
    };
    const COUNT: usize = 1;
}

impl StaticLeaves for Nil {
    const STATICS: [Option<isize>; MAX_RANK] = [None; MAX_RANK];
    const COUNT: usize = 0;
}

impl<H: StaticLeaves, T: StaticLeaves> StaticLeaves for IntTuple<H, T> {
    const STATICS: [Option<isize>; MAX_RANK] = concat(H::STATICS, H::COUNT, T::STATICS, T::COUNT);
    const COUNT: usize = H::COUNT + T::COUNT;
}

const fn static_extent(shape: Option<isize>) -> isize {
    match shape {
        Some(s) => s,
        None => panic!("owning tensor layouts need static extents"),
    }
}

// Flat leaf `i`'s stride: the product of the (filtered) extents ordered before it,
// or `_0` for a unit extent, as in `compact_order` and `compact`.
const fn ordered_stride(
    shapes: [Option<isize>; MAX_RANK],
    orders: [Option<isize>; MAX_RANK],
    count: usize,
    kind: Order,
    i: usize,
) -> isize {
    let filter = matches!(kind, Order::LikeStride | Order::LeftFiltered);
    let by_stride = matches!(kind, Order::LikeStride | Order::ByStride);
    let mut extents = [0; MAX_RANK];
    let mut ranks = [0; MAX_RANK];
    // Dynamic orders count as larger than the largest static one, left to right.
    let mut max = 0;
    let mut j = 0;
    while j < count {
        if let Some(o) = orders[j]
            && o > max
        {
            max = o;
        }
        j += 1;
    }
    j = 0;
    while j < count {
        extents[j] = if filter && matches!(orders[j], Some(0)) {
            1
        } else {
            static_extent(shapes[j])
        };
        ranks[j] = if !by_stride {
            j as isize
        } else {
            match orders[j] {
                Some(o) => o,
                None => max + 1 + j as isize,
            }
        };
        j += 1;
    }
    if extents[i] == 1 {
        return 0;
    }
    let mut start = 1isize;
    j = 0;
    while j < count {
        if ranks[j] < ranks[i] {
            start = match start.checked_mul(extents[j]) {
                Some(v) => v,
                None => panic!("owning tensor layout overflows"),
            };
        }
        j += 1;
    }
    start
}

pub const ORDERED<RS: StaticLeaves, RD: StaticLeaves, const K: Order, const POS: usize>: isize =
    ordered_stride(RS::STATICS, RD::STATICS, RS::COUNT, K, POS);
pub const SHIFT<H: Item, const POS: usize>: usize = POS + H::LEAVES;

impl<I, RS, RD, const K: Order, const POS: usize> Ordered<RS, RD, K, POS> for I
where
    I: Integer,
    RS: StaticLeaves,
    RD: StaticLeaves,
{
    type Output = Int<{ ORDERED::<RS, RD, K, POS> }>;
    fn ordered() -> Self::Output {
        Int
    }
}

impl<RS, RD, const K: Order, const POS: usize> Ordered<RS, RD, K, POS> for Nil {
    type Output = Nil;
    fn ordered() -> Nil {
        Nil
    }
}

impl<H, T, RS, RD, const K: Order, const POS: usize> Ordered<RS, RD, K, POS> for IntTuple<H, T>
where
    H: Item + Ordered<RS, RD, K, POS>,
    T: Ordered<RS, RD, K, { SHIFT::<H, POS> }>,
{
    type Output = IntTuple<H::Output, T::Output>;
    fn ordered() -> Self::Output {
        IntTuple(H::ordered(), T::ordered())
    }
}

type Strides<S, D, const K: Order> = <S as Ordered<S, D, K, 0>>::Output;

/// CuTe `make_layout_like(layout)`: a compact layout with the stride order of `layout`.
///
/// Static zero strides stay zero and their extents do not contribute to the
/// other strides. Equal static strides receive equal compact strides, as upstream.
pub trait LayoutLike {
    /// The compact layout.
    type Output: StaticValue;
}

impl<S: StaticLeaves, D: StaticLeaves> LayoutLike for Layout<S, D>
where
    S: Ordered<S, D, { Order::LikeStride }, 0>,
    Layout<S, Strides<S, D, { Order::LikeStride }>>: StaticValue,
{
    type Output = Layout<S, Strides<S, D, { Order::LikeStride }>>;
}

/// CuTe `make_ordered_layout(shape, order)`: a compact layout whose strides
/// increase in the order of `order`.
pub type OrderedLayout<S, O> = Layout<S, Strides<S, O, { Order::ByStride }>>;

/// CuTe `make_layout(shape)` for a static shape: compact left-major, with `Int<0>` for unit extents.
pub type ColMajorLayout<S> = <S as CompactLeft>::Layout;

/// Static tuple shapes with a compact left-major layout; see [`ColMajorLayout`].
pub trait CompactLeft {
    /// The compact layout.
    type Layout: StaticValue;
}

// Implemented for tuples only: rustdoc probes blanket impls on concrete types
// such as `DynInt`, which would evaluate the strides of a dynamic shape.
impl<H, T> CompactLeft for IntTuple<H, T>
where
    Self: StaticLeaves + Ordered<Self, Self, { Order::Left }, 0>,
    Layout<Self, Strides<Self, Self, { Order::Left }>>: StaticValue,
{
    type Layout = Layout<Self, Strides<Self, Self, { Order::Left }>>;
}

const MULTI_MODE<S: Item>: bool = S::RANK > 1;

type Mode0Layout<S0, D0> = Layout<S0, Strides<S0, D0, { Order::LeftFiltered }>>;

// `take<1, R>` of a multi-mode tuple is its tail chain.
impl<S0, SR, D0, DR> FragmentKind<IntTuple<S0, SR>, IntTuple<D0, DR>> for Bool<true>
where
    S0: StaticLeaves + Ordered<S0, D0, { Order::LeftFiltered }, 0>,
    D0: StaticLeaves,
    SR: StaticLeaves + Ordered<SR, DR, { Order::ByStride }, 0>,
    DR: StaticLeaves,
    Mode0Layout<S0, D0>: TypedProduct<OrderedLayout<SR, DR>, { ProductKind::Tiled }>,
{
    type Output =
        TypedProductLayout<Mode0Layout<S0, D0>, OrderedLayout<SR, DR>, { ProductKind::Tiled }>;
}

impl<S, D> FragmentKind<S, D> for Bool<false>
where
    S: StaticLeaves + Ordered<S, S, { Order::Left }, 0>,
{
    type Output = Layout<S, Strides<S, S, { Order::Left }>>;
}

/// CuTe `make_fragment_like(layout)`: a register layout with the shape of `layout`.
///
/// For more than one mode, mode 0 (the atom's values) is compact left-major with
/// static zero strides kept, and the remaining modes follow the stride order of
/// `layout`, joined by `tiled_product`. A layout with one mode is compact left-major.
pub trait FragmentLike {
    /// The fragment layout.
    type Output: StaticValue;
}

impl<S: Item, D> FragmentLike for Layout<S, D>
where
    Bool<{ MULTI_MODE::<S> }>: FragmentKind<S, D>,
    <Bool<{ MULTI_MODE::<S> }> as FragmentKind<S, D>>::Output: StaticValue,
{
    type Output = <Bool<{ MULTI_MODE::<S> }> as FragmentKind<S, D>>::Output;
}

// swizzle_layout.hpp: the fragment of a swizzled layout is that of `layout_b`.
impl<const B: u32, const M: u32, const SH: i32, O, L: FragmentLike> FragmentLike
    for ComposedLayout<Swizzle<B, M, SH>, O, L>
{
    type Output = L::Output;
}

impl<S, D> Layout<S, D> {
    /// CuTe `make_layout_like(layout)`; see [`LayoutLike`].
    ///
    /// ```
    /// use noodle_cute::{DynInt, Int, Layout, shape, stride};
    ///
    /// // Strides (0, 42, 1, 0) order the modes 2, 1; the zero strides stay zero.
    /// let reference = Layout::new(
    ///     shape![Int::<2>, Int::<3>, Int::<4>, Int::<5>],
    ///     stride![Int::<0>, DynInt(42), Int::<1>, Int::<0>],
    /// )?;
    /// let like = reference.make_layout_like();
    /// assert_eq!(like.stride, stride![Int::<0>, Int::<4>, Int::<1>, Int::<0>]);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn make_layout_like(&self) -> <Self as LayoutLike>::Output
    where
        Self: LayoutLike,
    {
        StaticValue::VALUE
    }

    /// CuTe `make_fragment_like(layout)`; see [`FragmentLike`].
    pub fn make_fragment_like(&self) -> <Self as FragmentLike>::Output
    where
        Self: FragmentLike,
    {
        StaticValue::VALUE
    }
}
