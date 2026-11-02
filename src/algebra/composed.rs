//! The layout algebra of a composed layout, applied to `layout_b` while the
//! function and the offset are kept: CuTe's `ComposedLayout` overloads of
//! `composition`, `complement`, the divides and products, and `coalesce`
//! (`layout_composed.hpp`).

use crate::Result;
use crate::layout::{ComposedLayout, Layout, Shape, StrideOf};
use crate::tuple::Flatten;

use super::coalesce::{CoalesceFold, FinishCoalesce};
use super::{
    CoalescedLayout, Complement, Grouping, ProductKind, TypedComplement, TypedCompose,
    TypedComposition, TypedDivide, TypedDivided, TypedProduct, TypedProductLayout,
};

type Composed<A, O, S, D, R> = ComposedLayout<A, O, TypedComposition<Layout<S, D>, R>>;

macro_rules! forward_tiler {
    ($($method:ident: $trait:ident, $alias:ident, $choice:expr;)*) => {
        impl<A: Clone, O: Clone, S: Shape, D: StrideOf<S>> ComposedLayout<A, O, Layout<S, D>> {
            $(
                #[doc = concat!("CuTe `", stringify!($method), "` of a composed layout: [`Layout::", stringify!($method), "`] of `layout_b`.")]
                pub fn $method<T>(
                    &self,
                    tiler: T,
                ) -> Result<ComposedLayout<A, O, $alias<Layout<S, D>, T, { $choice }>>>
                where
                    Layout<S, D>: $trait<T, { $choice }>,
                {
                    self.map_b(|b| b.$method(tiler))
                }
            )*
        }
    };
}

forward_tiler! {
    logical_divide: TypedDivide, TypedDivided, Grouping::Logical;
    tiled_divide: TypedDivide, TypedDivided, Grouping::Tiled;
    flat_divide: TypedDivide, TypedDivided, Grouping::Flat;
    logical_product: TypedProduct, TypedProductLayout, ProductKind::Logical;
    zipped_product: TypedProduct, TypedProductLayout, ProductKind::Zipped;
    tiled_product: TypedProduct, TypedProductLayout, ProductKind::Tiled;
    flat_product: TypedProduct, TypedProductLayout, ProductKind::Flat;
    blocked_product: TypedProduct, TypedProductLayout, ProductKind::Blocked;
    raked_product: TypedProduct, TypedProductLayout, ProductKind::Raked;
}

impl<A: Clone, O: Clone, S: Shape, D: StrideOf<S>> ComposedLayout<A, O, Layout<S, D>> {
    /// CuTe `composition(layout, tiler)` of a composed layout: `layout_a ∘ offset ∘ (layout_b ∘ rhs)`.
    pub fn composition<R>(&self, rhs: R) -> Result<Composed<A, O, S, D, R>>
    where
        Layout<S, D>: TypedCompose<R>,
    {
        self.map_b(|b| b.clone().compose(rhs))
    }

    /// CuTe `complement(layout, cotarget)` of a composed layout: the complement
    /// of `layout_b`, a plain layout.
    pub fn complement<C>(&self, cotarget: C) -> Result<TypedComplement<Layout<S, D>, C>>
    where
        Layout<S, D>: Complement<C>,
    {
        self.layout_b.complement(cotarget)
    }
}

impl<A: Clone, O: Clone, S: Shape + Flatten, D: StrideOf<S> + Flatten>
    ComposedLayout<A, O, Layout<S, D>>
where
    Layout<S::Output, D::Output>: CoalesceFold,
    <Layout<S::Output, D::Output> as CoalesceFold>::Output: FinishCoalesce,
{
    /// CuTe `coalesce(layout)` of a composed layout: [`Layout::coalesce`] of `layout_b`.
    pub fn coalesce(&self) -> Result<ComposedLayout<A, O, CoalescedLayout<S, D>>> {
        self.map_b(Layout::coalesce)
    }
}
