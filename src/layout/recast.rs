//! Element-width changes of a layout: CuTe `upcast`, `downcast`, and
//! `recast_layout` (`layout.hpp`).
//!
//! Like upstream these are unchecked: static strides must satisfy the
//! divisibility condition at compile time, and dynamic strides are assumed to be
//! multiples of the factor.

use core::marker::ConstParamTy;

use crate::Bool;
use crate::int::{Arith, CeilQuot, Int, Integer, Op, Prod, Quot, ceil_div, ceil_quot, prod, quot};
use crate::swizzle::{Swizzle, downcast_swizzle, upcast_swizzle};
use crate::tuple::{IntTuple, Item, Nil};

use super::{ComposedLayout, Layout, LayoutParts, make_composed_layout};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum StrideCase {
    Zero,
    Static,
    Dynamic,
}

const fn stride_case(d: Option<isize>) -> StrideCase {
    match d {
        Some(0) => StrideCase::Zero,
        Some(_) => StrideCase::Static,
        None => StrideCase::Dynamic,
    }
}

const CASE<D: Integer>: StrideCase = stride_case(D::STATIC);
const UNIT_STRIDE<D: Integer>: bool = matches!(D::STATIC, Some(1 | -1));

const fn static_stride<D: Integer>() -> isize {
    match D::STATIC {
        Some(d) => d,
        None => panic!("expected a static stride"),
    }
}

// The number of old elements merged into one new element along this stride.
const UP_SHAPE<D: Integer, const N: isize>: isize = {
    let d = static_stride::<D>();
    assert!(N > 0, "the upcast factor must be positive");
    assert!(
        d % N == 0 || N % d.abs() == 0,
        "upcast stride divisibility condition"
    );
    ceil_div(N, d.abs())
};
const UP_STRIDE<D: Integer, const N: isize>: isize = {
    let d = static_stride::<D>();
    d.signum() * ceil_div(d.abs(), N)
};

pub struct Case<const C: StrideCase>;

// Implementation traits that appear in public bounds but are not part of the API.
mod detail {
    pub trait UpcastLeaf<S, D, const N: isize> {
        type Shape;
        type Stride;
        fn upcast_leaf(shape: S, stride: D) -> (Self::Shape, Self::Stride);
    }

    pub trait DowncastLeaf<S, D, const N: isize> {
        type Shape;
        type Stride;
        fn downcast_leaf(shape: S, stride: D) -> (Self::Shape, Self::Stride);
    }

    /// Whether a stride tree has a static unit stride (`has_int1`).
    pub trait UnitStrides {
        const ANY: bool;
    }

    pub trait RecastKind<L> {
        type Output;
        fn recast(layout: L) -> Self::Output;
    }
}
use detail::{DowncastLeaf, RecastKind, UnitStrides, UpcastLeaf};

impl<S, D, const N: isize> UpcastLeaf<S, D, N> for Case<{ StrideCase::Zero }> {
    type Shape = S;
    type Stride = D;
    fn upcast_leaf(shape: S, stride: D) -> (S, D) {
        (shape, stride)
    }
}

impl<S, D, const N: isize> UpcastLeaf<S, D, N> for Case<{ StrideCase::Static }>
where
    D: Integer,
    S: Arith<Int<{ UP_SHAPE::<D, N> }>, { Op::CeilDiv }>,
{
    type Shape = CeilQuot<S, Int<{ UP_SHAPE::<D, N> }>>;
    type Stride = Int<{ UP_STRIDE::<D, N> }>;
    fn upcast_leaf(shape: S, _: D) -> (Self::Shape, Self::Stride) {
        (ceil_quot(shape, Int), Int)
    }
}

impl<S, D, const N: isize> UpcastLeaf<S, D, N> for Case<{ StrideCase::Dynamic }>
where
    D: Arith<Int<N>, { Op::Div }>,
{
    type Shape = S;
    type Stride = Quot<D, Int<N>>;
    fn upcast_leaf(shape: S, stride: D) -> (S, Self::Stride) {
        (shape, quot(stride, Int))
    }
}

impl<S, D, const N: isize> DowncastLeaf<S, D, N> for Bool<true>
where
    S: Arith<Int<N>, { Op::Mul }>,
{
    type Shape = Prod<S, Int<N>>;
    type Stride = D;
    fn downcast_leaf(shape: S, stride: D) -> (Self::Shape, D) {
        (prod(shape, Int), stride)
    }
}

impl<S, D, const N: isize> DowncastLeaf<S, D, N> for Bool<false>
where
    D: Arith<Int<N>, { Op::Mul }>,
{
    type Shape = S;
    type Stride = Prod<D, Int<N>>;
    fn downcast_leaf(shape: S, stride: D) -> (S, Self::Stride) {
        (shape, prod(stride, Int))
    }
}

impl<I: Integer> UnitStrides for I {
    const ANY: bool = UNIT_STRIDE::<I>;
}

impl UnitStrides for Nil {
    const ANY: bool = false;
}

impl<H: UnitStrides, T: UnitStrides> UnitStrides for IntTuple<H, T> {
    const ANY: bool = H::ANY || T::ANY;
}

/// CuTe `upcast<N>(layout)`: views `N` old elements as one new element.
///
/// A static unit stride divides its extent by `N` (rounding up); other static
/// strides `d` become `signum(d) * ceil(|d| / N)` and their extents shrink by
/// `ceil(N / |d|)`, failing to compile unless `d % N == 0 || N % |d| == 0`.
/// Static zero strides are kept, and dynamic strides are divided by `N`.
pub trait Upcast<const N: isize> {
    /// The upcast shape.
    type Shape;
    /// The upcast strides.
    type Stride;
    /// Returns the upcast layout.
    fn upcast(self) -> Layout<Self::Shape, Self::Stride>;
}

/// CuTe `downcast<N>(shape, stride)`: views one old element as `N` new elements.
///
/// A static stride of `±1` multiplies its extent by `N`; every other stride is
/// multiplied by `N`.
pub trait Downcast<const N: isize> {
    /// The downcast shape.
    type Shape;
    /// The downcast strides.
    type Stride;
    /// Returns the downcast layout.
    fn downcast(self) -> Layout<Self::Shape, Self::Stride>;
}

impl<S: Integer, D: Integer, const N: isize> Upcast<N> for Layout<S, D>
where
    Case<{ CASE::<D> }>: UpcastLeaf<S, D, N>,
{
    type Shape = <Case<{ CASE::<D> }> as UpcastLeaf<S, D, N>>::Shape;
    type Stride = <Case<{ CASE::<D> }> as UpcastLeaf<S, D, N>>::Stride;
    fn upcast(self) -> Layout<Self::Shape, Self::Stride> {
        let (shape, stride) =
            <Case<{ CASE::<D> }> as UpcastLeaf<S, D, N>>::upcast_leaf(self.shape, self.stride);
        Layout { shape, stride }
    }
}

impl<S: Integer, D: Integer, const N: isize> Downcast<N> for Layout<S, D>
where
    Bool<{ UNIT_STRIDE::<D> }>: DowncastLeaf<S, D, N>,
{
    type Shape = <Bool<{ UNIT_STRIDE::<D> }> as DowncastLeaf<S, D, N>>::Shape;
    type Stride = <Bool<{ UNIT_STRIDE::<D> }> as DowncastLeaf<S, D, N>>::Stride;
    fn downcast(self) -> Layout<Self::Shape, Self::Stride> {
        let (shape, stride) = <Bool<{ UNIT_STRIDE::<D> }> as DowncastLeaf<S, D, N>>::downcast_leaf(
            self.shape,
            self.stride,
        );
        Layout { shape, stride }
    }
}

macro_rules! modes {
    ($trait:ident, $method:ident) => {
        impl<const N: isize> $trait<N> for Layout<Nil, Nil> {
            type Shape = Nil;
            type Stride = Nil;
            fn $method(self) -> Self {
                self
            }
        }

        impl<SH, ST, DH, DT, const N: isize> $trait<N>
            for Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>
        where
            Layout<SH, DH>: $trait<N>,
            Layout<ST, DT>: $trait<N>,
        {
            type Shape = IntTuple<
                <Layout<SH, DH> as $trait<N>>::Shape,
                <Layout<ST, DT> as $trait<N>>::Shape,
            >;
            type Stride = IntTuple<
                <Layout<SH, DH> as $trait<N>>::Stride,
                <Layout<ST, DT> as $trait<N>>::Stride,
            >;
            fn $method(self) -> Layout<Self::Shape, Self::Stride> {
                let h = $trait::<N>::$method(Layout {
                    shape: self.shape.0,
                    stride: self.stride.0,
                });
                let t = $trait::<N>::$method(Layout {
                    shape: self.shape.1,
                    stride: self.stride.1,
                });
                Layout {
                    shape: IntTuple(h.shape, t.shape),
                    stride: IntTuple(h.stride, t.stride),
                }
            }
        }
    };
}
modes!(Upcast, upcast);
modes!(Downcast, downcast);

const fn gcd(mut a: isize, mut b: isize) -> isize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

const fn bits<T>() -> isize {
    let bits = (core::mem::size_of::<T>() * 8) as isize;
    assert!(
        bits > 0,
        "recast requires sized, non-zero-sized element types"
    );
    bits
}

// sizeof_bits<New> / sizeof_bits<Old> as a reduced fraction.
const RECAST_NUM<Old, New>: isize = bits::<New>() / gcd(bits::<New>(), bits::<Old>());
const RECAST_DEN<Old, New>: isize = bits::<Old>() / gcd(bits::<New>(), bits::<Old>());

#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum RecastCase {
    Same,
    Down,
    Up,
    Both,
}

const fn recast_case(num: isize, den: isize) -> RecastCase {
    match (num, den) {
        (1, 1) => RecastCase::Same,
        (1, _) => RecastCase::Down,
        (_, 1) => RecastCase::Up,
        _ => RecastCase::Both,
    }
}

const RECAST_CASE<Old, New>: RecastCase =
    recast_case(RECAST_NUM::<Old, New>, RECAST_DEN::<Old, New>);

pub struct Recast<Old, New, const C: RecastCase>(core::marker::PhantomData<(Old, New)>);

impl<Old, New, S, D> RecastKind<Layout<S, D>> for Recast<Old, New, { RecastCase::Same }> {
    type Output = Layout<S, D>;
    fn recast(layout: Layout<S, D>) -> Layout<S, D> {
        layout
    }
}

impl<Old, New, S, D> RecastKind<Layout<S, D>> for Recast<Old, New, { RecastCase::Down }>
where
    Layout<S, D>: Downcast<{ RECAST_DEN::<Old, New> }>,
    D: UnitStrides,
{
    type Output = Layout<
        <Layout<S, D> as Downcast<{ RECAST_DEN::<Old, New> }>>::Shape,
        <Layout<S, D> as Downcast<{ RECAST_DEN::<Old, New> }>>::Stride,
    >;
    fn recast(layout: Layout<S, D>) -> Self::Output {
        const { assert!(D::ANY, "downcast requires adjacent elements") };
        Downcast::downcast(layout)
    }
}

impl<Old, New, S, D> RecastKind<Layout<S, D>> for Recast<Old, New, { RecastCase::Up }>
where
    Layout<S, D>: Upcast<{ RECAST_NUM::<Old, New> }>,
{
    type Output = Layout<
        <Layout<S, D> as Upcast<{ RECAST_NUM::<Old, New> }>>::Shape,
        <Layout<S, D> as Upcast<{ RECAST_NUM::<Old, New> }>>::Stride,
    >;
    fn recast(layout: Layout<S, D>) -> Self::Output {
        Upcast::upcast(layout)
    }
}

type UpcastNum<Old, New, S, D> = Layout<
    <Layout<S, D> as Upcast<{ RECAST_NUM::<Old, New> }>>::Shape,
    <Layout<S, D> as Upcast<{ RECAST_NUM::<Old, New> }>>::Stride,
>;

impl<Old, New, S, D> RecastKind<Layout<S, D>> for Recast<Old, New, { RecastCase::Both }>
where
    Layout<S, D>: Upcast<{ RECAST_NUM::<Old, New> }>,
    UpcastNum<Old, New, S, D>: Downcast<{ RECAST_DEN::<Old, New> }>,
    <Layout<S, D> as Upcast<{ RECAST_NUM::<Old, New> }>>::Stride: UnitStrides,
{
    type Output = Layout<
        <UpcastNum<Old, New, S, D> as Downcast<{ RECAST_DEN::<Old, New> }>>::Shape,
        <UpcastNum<Old, New, S, D> as Downcast<{ RECAST_DEN::<Old, New> }>>::Stride,
    >;
    fn recast(layout: Layout<S, D>) -> Self::Output {
        const {
            assert!(
                <<Layout<S, D> as Upcast<{ RECAST_NUM::<Old, New> }>>::Stride as UnitStrides>::ANY,
                "downcast requires adjacent elements"
            )
        };
        Downcast::downcast(Upcast::upcast(layout))
    }
}

/// CuTe `recast_layout<Old, New>(layout)`; see [`Layout::recast_layout`].
pub trait RecastLayout<Old, New> {
    /// The shape counted in `New` elements.
    type Shape;
    /// The strides counted in `New` elements.
    type Stride;
    /// Returns the layout counted in `New` elements.
    fn recast_layout(self) -> Layout<Self::Shape, Self::Stride>;
}

type Recasted<Old, New, L> =
    <Recast<Old, New, { RECAST_CASE::<Old, New> }> as RecastKind<L>>::Output;

impl<Old, New, S, D> RecastLayout<Old, New> for Layout<S, D>
where
    Recast<Old, New, { RECAST_CASE::<Old, New> }>: RecastKind<Self>,
    Recasted<Old, New, Self>: LayoutParts,
{
    type Shape = <Recasted<Old, New, Self> as LayoutParts>::Shape;
    type Stride = <Recasted<Old, New, Self> as LayoutParts>::Stride;
    fn recast_layout(self) -> Layout<Self::Shape, Self::Stride> {
        <Recast<Old, New, { RECAST_CASE::<Old, New> }> as RecastKind<Self>>::recast(self)
            .into_layout()
    }
}

impl<S, D> Layout<S, D> {
    /// CuTe `upcast<N>(layout)`; see [`Upcast`].
    pub fn upcast<const N: isize>(
        self,
    ) -> Layout<<Self as Upcast<N>>::Shape, <Self as Upcast<N>>::Stride>
    where
        Self: Upcast<N>,
    {
        Upcast::upcast(self)
    }

    /// CuTe `downcast<N>(layout)`; see [`Downcast`]. As upstream, the strides
    /// must contain a static unit stride, or compilation fails.
    pub fn downcast<const N: isize>(
        self,
    ) -> Layout<<Self as Downcast<N>>::Shape, <Self as Downcast<N>>::Stride>
    where
        Self: Downcast<N>,
        D: UnitStrides,
    {
        const { assert!(D::ANY, "downcast requires adjacent elements") };
        Downcast::downcast(self)
    }

    /// CuTe `recast_layout<Old, New>(layout)`: rescales by `size_of::<New>() / size_of::<Old>()`.
    ///
    /// The ratio is reduced to `num / den`; the layout is upcast by `num` and then
    /// downcast by `den`, skipping a factor of one.
    ///
    /// ```
    /// use noodle_cute::{Int, Layout, shape, stride};
    ///
    /// // 16 bytes along the contiguous mode become 4 `u32`s; the row stride shrinks too.
    /// let bytes = Layout::new(shape![Int::<16>, Int::<8>], stride![Int::<1>, Int::<16>])?;
    /// let words = bytes.recast_layout::<u8, u32>();
    /// assert_eq!(words, Layout { shape: shape![Int::<4>, Int::<8>], stride: stride![Int::<1>, Int::<4>] });
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn recast_layout<Old, New>(
        self,
    ) -> Layout<<Self as RecastLayout<Old, New>>::Shape, <Self as RecastLayout<Old, New>>::Stride>
    where
        Self: RecastLayout<Old, New>,
    {
        RecastLayout::<Old, New>::recast_layout(self)
    }
}

/// The layout of a tensor reinterpreted from `Old` to `New` elements, with the
/// pointer adjustment in `Old` elements: the layout half of CuTe `recast<New>(tensor)`.
pub trait RecastView<Old, New> {
    /// The recast layout.
    type Output;
    /// Returns the recast layout and the pointer offset in `Old` elements.
    fn recast_view(self) -> (Self::Output, isize);
}

impl<Old, New, S: Item, D: Item> RecastView<Old, New> for Layout<S, D>
where
    Self: RecastLayout<Old, New> + Clone,
    <Self as RecastLayout<Old, New>>::Shape: Item,
{
    type Output =
        Layout<<Self as RecastLayout<Old, New>>::Shape, <Self as RecastLayout<Old, New>>::Stride>;
    fn recast_view(self) -> (Self::Output, isize) {
        let new = self.clone().recast_layout::<Old, New>();
        let mut offset = 0;
        // tensor_impl.hpp: an upcast of a plain layout moves to the lowest covered address.
        if size_of::<Old>() < size_of::<New>() {
            for i in 0..S::LEAVES {
                if let (Ok(s), Ok(n), Ok(d)) =
                    (self.shape.leaf(i), new.shape.leaf(i), self.stride.leaf(i))
                {
                    offset += ((s - n) * d).min(0);
                }
            }
        }
        (new, offset)
    }
}

pub const RECAST_SWIZZLE_B<const B: u32, const M: u32, Old, New>: u32 =
    upcast_swizzle(B, M, RECAST_NUM::<Old, New> as u32).0;
pub const RECAST_SWIZZLE_M<const B: u32, const M: u32, Old, New>: u32 = downcast_swizzle(
    upcast_swizzle(B, M, RECAST_NUM::<Old, New> as u32).1,
    RECAST_DEN::<Old, New> as u32,
);

type Upcast1<O, Old, New> = Quot<O, Int<{ RECAST_NUM::<Old, New> }>>;

// layout_composed.hpp / swizzle_layout.hpp: recast each of the swizzle, the offset, and layout_b.
impl<Old, New, const B: u32, const M: u32, const S: i32, O, L> RecastView<Old, New>
    for ComposedLayout<Swizzle<B, M, S>, O, L>
where
    O: Arith<Int<{ RECAST_NUM::<Old, New> }>, { Op::Div }>,
    Upcast1<O, Old, New>: Arith<Int<{ RECAST_DEN::<Old, New> }>, { Op::Mul }>,
    L: RecastLayout<Old, New>,
{
    type Output = ComposedLayout<
        Swizzle<{ RECAST_SWIZZLE_B::<B, M, Old, New> }, { RECAST_SWIZZLE_M::<B, M, Old, New> }, S>,
        Prod<Upcast1<O, Old, New>, Int<{ RECAST_DEN::<Old, New> }>>,
        Layout<L::Shape, L::Stride>,
    >;
    fn recast_view(self) -> (Self::Output, isize) {
        let offset = prod(quot(self.offset, Int), Int);
        (
            make_composed_layout(Swizzle, offset, self.layout_b.recast_layout()),
            0,
        )
    }
}

/// `B` of the swizzle in CuTe `upcast<N>(Swizzle<B, M, S>)`.
pub const UPCAST_SWIZZLE_B<const B: u32, const M: u32, const N: isize>: u32 =
    upcast_swizzle(B, M, N as u32).0;
/// `M` of the swizzle in CuTe `upcast<N>(Swizzle<B, M, S>)`.
pub const UPCAST_SWIZZLE_M<const B: u32, const M: u32, const N: isize>: u32 =
    upcast_swizzle(B, M, N as u32).1;
/// `M` of the swizzle in CuTe `downcast<N>(Swizzle<B, M, S>)`.
pub const DOWNCAST_SWIZZLE_M<const M: u32, const N: isize>: u32 = downcast_swizzle(M, N as u32);

/// CuTe `upcast<N>` of a swizzled layout: the swizzle, the offset (divided by
/// `N`), and `layout_b` are upcast together. `N` must be a power of two.
pub trait SwizzleUpcast<const N: isize> {
    /// The upcast layout.
    type Output;
    /// Returns the upcast layout.
    fn upcast(self) -> Self::Output;
}

impl<const B: u32, const M: u32, const S: i32, O, LS, LD, const N: isize> SwizzleUpcast<N>
    for ComposedLayout<Swizzle<B, M, S>, O, Layout<LS, LD>>
where
    O: Arith<Int<N>, { Op::Div }>,
    Layout<LS, LD>: Upcast<N>,
{
    type Output = ComposedLayout<
        Swizzle<{ UPCAST_SWIZZLE_B::<B, M, N> }, { UPCAST_SWIZZLE_M::<B, M, N> }, S>,
        Quot<O, Int<N>>,
        Layout<<Layout<LS, LD> as Upcast<N>>::Shape, <Layout<LS, LD> as Upcast<N>>::Stride>,
    >;
    fn upcast(self) -> Self::Output {
        make_composed_layout(
            Swizzle,
            quot(self.offset, Int),
            Upcast::upcast(self.layout_b),
        )
    }
}

/// CuTe `downcast<N>` of a swizzled layout: the swizzle, the offset
/// (multiplied by `N`), and `layout_b` are downcast together. As for
/// [`Layout::downcast`], `layout_b` needs a static unit stride.
pub trait SwizzleDowncast<const N: isize> {
    /// The downcast layout.
    type Output;
    /// Returns the downcast layout.
    fn downcast(self) -> Self::Output;
}

impl<const B: u32, const M: u32, const S: i32, O, LS, LD, const N: isize> SwizzleDowncast<N>
    for ComposedLayout<Swizzle<B, M, S>, O, Layout<LS, LD>>
where
    O: Arith<Int<N>, { Op::Mul }>,
    Layout<LS, LD>: Downcast<N>,
    LD: UnitStrides,
{
    type Output = ComposedLayout<
        Swizzle<B, { DOWNCAST_SWIZZLE_M::<M, N> }, S>,
        Prod<O, Int<N>>,
        Layout<<Layout<LS, LD> as Downcast<N>>::Shape, <Layout<LS, LD> as Downcast<N>>::Stride>,
    >;
    fn downcast(self) -> Self::Output {
        make_composed_layout(
            Swizzle,
            prod(self.offset, Int),
            self.layout_b.downcast::<N>(),
        )
    }
}

impl<A, O, B> ComposedLayout<A, O, B> {
    /// CuTe `upcast<N>(layout)` of a swizzled layout; see [`SwizzleUpcast`].
    pub fn upcast<const N: isize>(self) -> <Self as SwizzleUpcast<N>>::Output
    where
        Self: SwizzleUpcast<N>,
    {
        SwizzleUpcast::upcast(self)
    }

    /// CuTe `downcast<N>(layout)` of a swizzled layout; see [`SwizzleDowncast`].
    pub fn downcast<const N: isize>(self) -> <Self as SwizzleDowncast<N>>::Output
    where
        Self: SwizzleDowncast<N>,
    {
        SwizzleDowncast::downcast(self)
    }

    /// CuTe `recast_layout<Old, New>(layout)` of a swizzled layout: the layout
    /// half of [`RecastView`], which never moves the pointer of a composed layout.
    ///
    /// ```
    /// use noodle_cute::layout::LayoutRight;
    /// use noodle_cute::swizzle::Swizzle;
    /// use noodle_cute::{Int, shape, stride};
    ///
    /// // 8 rows of 64 halves; as 128-bit vectors each row has 8 vectors.
    /// let smem = Swizzle::<3, 3, 3>.compose(LayoutRight::make(shape![Int::<8>, Int::<64>])?);
    /// let vectors = smem.recast_layout::<u16, u128>();
    /// assert_eq!(vectors.layout_a, Swizzle::<3, 0, 3>);
    /// assert_eq!(vectors.layout_b.shape, shape![Int::<8>, Int::<8>]);
    /// assert_eq!(vectors.layout_b.stride, stride![Int::<8>, Int::<1>]);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn recast_layout<Old, New>(self) -> <Self as RecastView<Old, New>>::Output
    where
        Self: RecastView<Old, New>,
    {
        self.recast_view().0
    }
}
