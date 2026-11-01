//! Typed tiler traversal, `tile_unzip`, and grouping shared by divide and product.

use crate::int::Integer;
use crate::layout::{Layout, LayoutMap, LayoutParts, Shape};
use crate::tuple::{Append, Item};
use crate::{Int, IntTuple, Nil, Result};
use core::fmt::Debug;

use super::{FLAT, LOGICAL, TILED, ZIPPED};

pub type Pair<A, B> = IntTuple<A, IntTuple<B, Nil>>;
pub type Joined<A, B> = Layout<
    Pair<<A as LayoutParts>::Shape, <B as LayoutParts>::Shape>,
    Pair<<A as LayoutParts>::Stride, <B as LayoutParts>::Stride>,
>;
pub fn join<A: LayoutParts, B: LayoutParts>(a: A, b: B) -> Joined<A, B> {
    let (a, b) = (a.into_layout(), b.into_layout());
    Layout {
        shape: IntTuple(a.shape, IntTuple(b.shape, Nil)),
        stride: IntTuple(a.stride, IntTuple(b.stride, Nil)),
    }
}

pub trait Checked: LayoutParts {
    type Output: LayoutMap;
    fn checked(self) -> Result<Self::Output>;
}
impl<S: Shape, D: Item<Profile = S::Profile>> Checked for Layout<S, D> {
    type Output = Self;
    fn checked(self) -> Result<Self> {
        Layout::new(self.shape, self.stride)
    }
}

pub trait Wrap {
    type Output;
    fn wrap(self) -> Self::Output;
}
impl<I: Integer> Wrap for I {
    type Output = IntTuple<I, Nil>;
    fn wrap(self) -> Self::Output {
        IntTuple(self, Nil)
    }
}
impl Wrap for Nil {
    type Output = Self;
    fn wrap(self) -> Self {
        self
    }
}
impl<H, T> Wrap for IntTuple<H, T> {
    type Output = Self;
    fn wrap(self) -> Self {
        self
    }
}
pub trait IntoModes {
    type Output: LayoutParts;
    fn into_modes(self) -> Self::Output;
}
impl<S: Wrap, D: Wrap> IntoModes for Layout<S, D> {
    type Output = Layout<S::Output, D::Output>;
    fn into_modes(self) -> Self::Output {
        Layout {
            shape: self.shape.wrap(),
            stride: self.stride.wrap(),
        }
    }
}

pub trait Split: LayoutParts {
    type Tile: LayoutParts;
    type Rest: LayoutParts;
    fn split(self) -> (Self::Tile, Self::Rest);
}
impl<TS, RS, TD, RD> Split for Layout<Pair<TS, RS>, Pair<TD, RD>> {
    type Tile = Layout<TS, TD>;
    type Rest = Layout<RS, RD>;
    fn split(self) -> (Self::Tile, Self::Rest) {
        (
            Layout {
                shape: self.shape.0,
                stride: self.stride.0,
            },
            Layout {
                shape: self.shape.1.0,
                stride: self.stride.1.0,
            },
        )
    }
}

pub trait Binary<L, T> {
    type Output: LayoutParts;
    fn run(layout: L, tiler: T) -> Result<Self::Output>;
}
pub trait Transform<T, Op> {
    type Output: LayoutParts;
    fn transform(self, tiler: T) -> Result<Self::Output>;
}
impl<S, D, TS, TD, Op> Transform<Layout<TS, TD>, Op> for Layout<S, D>
where
    Op: Binary<Self, Layout<TS, TD>>,
{
    type Output = Op::Output;
    fn transform(self, tiler: Layout<TS, TD>) -> Result<Self::Output> {
        Op::run(self, tiler)
    }
}
impl<S, D, I: Integer + Debug, Op> Transform<I, Op> for Layout<S, D>
where
    Op: Binary<Self, Layout<I, Int<1>>>,
{
    type Output = Op::Output;
    fn transform(self, tiler: I) -> Result<Self::Output> {
        Op::run(
            self,
            Layout {
                shape: tiler,
                stride: Int,
            },
        )
    }
}
impl<S, D, Op> Transform<Nil, Op> for Layout<S, D> {
    type Output = Self;
    fn transform(self, _: Nil) -> Result<Self> {
        Ok(self)
    }
}
impl<S: Wrap, D: Wrap, H, T, Op> Transform<IntTuple<H, T>, Op> for Layout<S, D>
where
    Layout<S::Output, D::Output>: TransformModes<IntTuple<H, T>, Op>,
{
    type Output = <Layout<S::Output, D::Output> as TransformModes<IntTuple<H, T>, Op>>::Output;
    fn transform(self, tiler: IntTuple<H, T>) -> Result<Self::Output> {
        self.into_modes().transform_modes(tiler)
    }
}
pub trait TransformModes<T, Op> {
    type Output: LayoutParts;
    fn transform_modes(self, tiler: T) -> Result<Self::Output>;
}
impl<S, D, Op> TransformModes<Nil, Op> for Layout<S, D> {
    type Output = Self;
    fn transform_modes(self, _: Nil) -> Result<Self> {
        Ok(self)
    }
}
impl<S, ST, D, DT, H, T, Op> TransformModes<IntTuple<H, T>, Op>
    for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    Layout<S, D>: Transform<H, Op>,
    Layout<ST, DT>: TransformModes<T, Op>,
{
    type Output = Layout<
        IntTuple<
            <<Layout<S, D> as Transform<H, Op>>::Output as LayoutParts>::Shape,
            <<Layout<ST, DT> as TransformModes<T, Op>>::Output as LayoutParts>::Shape,
        >,
        IntTuple<
            <<Layout<S, D> as Transform<H, Op>>::Output as LayoutParts>::Stride,
            <<Layout<ST, DT> as TransformModes<T, Op>>::Output as LayoutParts>::Stride,
        >,
    >;
    fn transform_modes(self, tiler: IntTuple<H, T>) -> Result<Self::Output> {
        let h = Layout {
            shape: self.shape.0,
            stride: self.stride.0,
        }
        .transform(tiler.0)?
        .into_layout();
        let t = Layout {
            shape: self.shape.1,
            stride: self.stride.1,
        }
        .transform_modes(tiler.1)?
        .into_layout();
        Ok(Layout {
            shape: IntTuple(h.shape, t.shape),
            stride: IntTuple(h.stride, t.stride),
        })
    }
}

pub trait Unzip<T> {
    type Output: Split;
    fn unzip(self) -> Self::Output;
}
impl<L: Split, S, D> Unzip<Layout<S, D>> for L {
    type Output = Self;
    fn unzip(self) -> Self {
        self
    }
}
impl<L: Split, I: Integer> Unzip<I> for L {
    type Output = Self;
    fn unzip(self) -> Self {
        self
    }
}
impl<S, D> Unzip<Nil> for Layout<S, D> {
    type Output = Joined<Layout<Nil, Nil>, Self>;
    fn unzip(self) -> Self::Output {
        join(
            Layout {
                shape: Nil,
                stride: Nil,
            },
            self,
        )
    }
}
type Unzipped<S, D, T> = <Layout<S, D> as Unzip<T>>::Output;
type Tile<L> = <L as Split>::Tile;
type Rest<L> = <L as Split>::Rest;
type Cons<H, T> = Layout<
    IntTuple<<H as LayoutParts>::Shape, <T as LayoutParts>::Shape>,
    IntTuple<<H as LayoutParts>::Stride, <T as LayoutParts>::Stride>,
>;
impl<S, ST, D, DT, H, T> Unzip<IntTuple<H, T>> for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    Layout<S, D>: Unzip<H>,
    Layout<ST, DT>: Unzip<T>,
{
    type Output = Joined<
        Cons<Tile<Unzipped<S, D, H>>, Tile<Unzipped<ST, DT, T>>>,
        Cons<Rest<Unzipped<S, D, H>>, Rest<Unzipped<ST, DT, T>>>,
    >;
    fn unzip(self) -> Self::Output {
        let (ht, hr) = Layout {
            shape: self.shape.0,
            stride: self.stride.0,
        }
        .unzip()
        .split();
        let (tt, tr) = Layout {
            shape: self.shape.1,
            stride: self.stride.1,
        }
        .unzip()
        .split();
        let (ht, hr, tt, tr) = (
            ht.into_layout(),
            hr.into_layout(),
            tt.into_layout(),
            tr.into_layout(),
        );
        join(
            Layout {
                shape: IntTuple(ht.shape, tt.shape),
                stride: IntTuple(ht.stride, tt.stride),
            },
            Layout {
                shape: IntTuple(hr.shape, tr.shape),
                stride: IntTuple(hr.stride, tr.stride),
            },
        )
    }
}

pub trait Regroup<const KIND: u8> {
    type Output: LayoutParts;
    fn regroup(self) -> Self::Output;
}
impl<L: Split> Regroup<{ ZIPPED }> for L {
    type Output = Self;
    fn regroup(self) -> Self {
        self
    }
}
impl<TS, RS: Wrap, TD, RD: Wrap> Regroup<{ TILED }> for Layout<Pair<TS, RS>, Pair<TD, RD>> {
    type Output = Layout<IntTuple<TS, RS::Output>, IntTuple<TD, RD::Output>>;
    fn regroup(self) -> Self::Output {
        Layout {
            shape: IntTuple(self.shape.0, self.shape.1.0.wrap()),
            stride: IntTuple(self.stride.0, self.stride.1.0.wrap()),
        }
    }
}
impl<TS: Wrap, RS: Wrap, TD: Wrap, RD: Wrap> Regroup<{ FLAT }>
    for Layout<Pair<TS, RS>, Pair<TD, RD>>
where
    TS::Output: Append<RS::Output>,
    TD::Output: Append<RD::Output>,
{
    type Output = Layout<
        <TS::Output as Append<RS::Output>>::Output,
        <TD::Output as Append<RD::Output>>::Output,
    >;
    fn regroup(self) -> Self::Output {
        Layout {
            shape: self.shape.0.wrap().append(self.shape.1.0.wrap()),
            stride: self.stride.0.wrap().append(self.stride.1.0.wrap()),
        }
    }
}

/// Selects how a typed divide or product arranges its `(tile, rest)` result.
pub struct Kind<const KIND: u8>;
pub trait Arrange<L, T> {
    type Output: LayoutParts;
    fn arrange(layout: L) -> Self::Output;
}
impl<L: LayoutParts, T> Arrange<L, T> for Kind<{ LOGICAL }> {
    type Output = L;
    fn arrange(layout: L) -> L {
        layout
    }
}
macro_rules! arrange {
    ($($kind:ident),*) => {$ (
        impl<L: Unzip<T>, T> Arrange<L, T> for Kind<{ $kind }>
        where L::Output: Regroup<{ $kind }>,
        {
            type Output = <L::Output as Regroup<{ $kind }>>::Output;
            fn arrange(layout: L) -> Self::Output { layout.unzip().regroup() }
        }
    )*};
}
arrange!(ZIPPED, TILED, FLAT);

pub type Transformed<L, T, Op> = <L as Transform<T, Op>>::Output;
pub type Arranged<L, T, Op, const KIND: u8> =
    <Kind<KIND> as Arrange<Transformed<L, T, Op>, T>>::Output;

macro_rules! operation_api {
    ($trait:ident, $alias:ident, $op:ty, $apply:ident; $(($kind:ident, $method:ident)),+ $(,)?) => {
        /// Typed layout transform; `KIND` is one of the grouping constants such as [`LOGICAL`](super::LOGICAL).
        pub trait $trait<T, const KIND: u8 = { super::LOGICAL }> {
            type Output: crate::layout::LayoutMap;
            fn $apply(self, tiler: T) -> crate::Result<Self::Output>;
        }
        /// Result type of the typed transform, keeping static integers and the full hierarchy.
        ///
        /// For fully static inputs, `<Alias as StaticValue>::VALUE` is the result as a constant.
        pub type $alias<L, T, const KIND: u8 = { super::LOGICAL }> = <L as $trait<T, KIND>>::Output;
        impl<S: crate::layout::Shape, D: crate::tuple::Item<Profile = S::Profile>, T, const KIND: u8>
            $trait<T, KIND> for crate::layout::Layout<S, D>
        where
            Self: super::typed::Transform<T, $op>,
            super::typed::Kind<KIND>: super::typed::Arrange<super::typed::Transformed<Self, T, $op>, T>,
            super::typed::Arranged<Self, T, $op, KIND>: super::typed::Checked,
        {
            type Output = <super::typed::Arranged<Self, T, $op, KIND> as super::typed::Checked>::Output;
            fn $apply(self, tiler: T) -> crate::Result<Self::Output> {
                use super::typed::{Arrange, Checked, Transform};
                self.size()?;
                let logical = self.transform(tiler)?;
                <super::typed::Kind<KIND> as Arrange<_, T>>::arrange(logical).checked()
            }
        }
        impl<S: crate::layout::Shape, D: crate::tuple::Item<Profile = S::Profile>> crate::layout::Layout<S, D> {
            $(
                /// Checked typed transform with the tiler and grouping rules of this module,
                /// keeping static integers and nested profiles.
                ///
                /// Runtime errors are returned through `Result`; only static facts change the
                /// structure, and typed results are not limited by `MAX_RANK`.
                pub fn $method<T>(&self, tiler: T) -> crate::Result<$alias<Self, T, { super::$kind }>>
                where Self: $trait<T, { super::$kind }>,
                { self.clone().$apply(tiler) }
            )+
        }
    };
}
pub(super) use operation_api;
