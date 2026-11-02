//! Typed tiler traversal, `tile_unzip`, and grouping shared by divide and product.

use crate::coord::Underscore;
use crate::int::Integer;
use crate::layout::{Layout, LayoutMap, LayoutParts, Shape, StrideOf};
use crate::tuple::Append;
use crate::{Int, IntTuple, Nil, Result};
use core::fmt::Debug;

use super::Grouping;

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
impl<S: Shape, D: StrideOf<S>> Checked for Layout<S, D> {
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
// layout.hpp: an `_` tiler mode leaves the mode unchanged.
impl<S, D, Op> Transform<Underscore, Op> for Layout<S, D> {
    type Output = Self;
    fn transform(self, _: Underscore) -> Result<Self> {
        Ok(self)
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
// tuple_algorithms.hpp zip2_by: a mode kept by `_` is already a (tile, rest) pair.
impl<L: Split> Unzip<Underscore> for L {
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

pub trait Regroup<const GROUPING: Grouping> {
    type Output: LayoutParts;
    fn regroup(self) -> Self::Output;
}
impl<L: Split> Regroup<{ Grouping::Zipped }> for L {
    type Output = Self;
    fn regroup(self) -> Self {
        self
    }
}
impl<TS, RS: Wrap, TD, RD: Wrap> Regroup<{ Grouping::Tiled }>
    for Layout<Pair<TS, RS>, Pair<TD, RD>>
{
    type Output = Layout<IntTuple<TS, RS::Output>, IntTuple<TD, RD::Output>>;
    fn regroup(self) -> Self::Output {
        Layout {
            shape: IntTuple(self.shape.0, self.shape.1.0.wrap()),
            stride: IntTuple(self.stride.0, self.stride.1.0.wrap()),
        }
    }
}
impl<TS: Wrap, RS: Wrap, TD: Wrap, RD: Wrap> Regroup<{ Grouping::Flat }>
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
pub struct Kind<const GROUPING: Grouping>;
pub trait Arrange<L, T> {
    type Output: LayoutParts;
    fn arrange(layout: L) -> Self::Output;
}
impl<L: LayoutParts, T> Arrange<L, T> for Kind<{ Grouping::Logical }> {
    type Output = L;
    fn arrange(layout: L) -> L {
        layout
    }
}
macro_rules! arrange {
    ($($kind:ident),*) => {$ (
        impl<L: Unzip<T>, T> Arrange<L, T> for Kind<{ Grouping::$kind }>
        where L::Output: Regroup<{ Grouping::$kind }>,
        {
            type Output = <L::Output as Regroup<{ Grouping::$kind }>>::Output;
            fn arrange(layout: L) -> Self::Output { layout.unzip().regroup() }
        }
    )*};
}
arrange!(Zipped, Tiled, Flat);

pub type Transformed<L, T, Op> = <L as Transform<T, Op>>::Output;
pub type Arranged<L, T, Op, A> = <A as Arrange<Transformed<L, T, Op>, T>>::Output;

pub(super) macro operation_api($(#[$attr:meta])* $trait:ident, $alias:ident, $output:ident,
     const $selector:ident: $selector_ty:ty = $default:expr,
     $op:ty, $arranger:ty, $apply:ident; $(($choice:expr, $method:ident)),+ $(,)?) {
        /// Typed layout transform selected by an enum-valued mode.
        $(#[$attr])*
        pub trait $trait<T, const $selector: $selector_ty = { $default }> {
            type $output: crate::layout::LayoutMap;
            fn $apply(self, tiler: T) -> crate::Result<Self::$output>;
        }
        /// Result type of the typed transform, keeping static integers and the full hierarchy.
        ///
        /// For fully static inputs, `<Alias as StaticValue>::VALUE` is the result as a constant.
        pub type $alias<L, T, const $selector: $selector_ty = { $default }> = <L as $trait<T, $selector>>::$output;
        impl<S: crate::layout::Shape, D: crate::layout::StrideOf<S>, T, const $selector: $selector_ty>
            $trait<T, $selector> for crate::layout::Layout<S, D>
        where
            Self: super::typed::Transform<T, $op>,
            $arranger: super::typed::Arrange<super::typed::Transformed<Self, T, $op>, T>,
            super::typed::Arranged<Self, T, $op, $arranger>: super::typed::Checked,
        {
            type $output = <super::typed::Arranged<Self, T, $op, $arranger> as super::typed::Checked>::Output;
            fn $apply(self, tiler: T) -> crate::Result<Self::$output> {
                use super::typed::{Arrange, Checked, Transform};
                self.size()?;
                let logical = self.transform(tiler)?;
                <$arranger as Arrange<_, T>>::arrange(logical).checked()
            }
        }
        impl<S: crate::layout::Shape, D: crate::layout::StrideOf<S>> crate::layout::Layout<S, D> {
            $(
                /// Checked typed transform with the tiler and grouping rules of this module,
                /// keeping static integers and nested profiles.
                ///
                /// Runtime errors are returned through `Result`; only static facts change the
                /// structure, and typed results are not limited by `MAX_RANK`.
                pub fn $method<T>(&self, tiler: T) -> crate::Result<$alias<Self, T, { $choice }>>
                where Self: $trait<T, { $choice }>,
                { self.clone().$apply(tiler) }
            )+
        }
    }
