//! Layout logical product ($A \otimes B$) and rank-sensitive blocked / raked products.
//!
//! Logical product reproduces a tile layout $A$ across the layout of tiles $B$:
//! - Mode 0 (the **tile**): $A$ itself;
//! - Mode 1 (the **grid**): $A$'s minimal (free) complement composed with $B$, i.e. $A^* \circ B$.
//!
//! Formally, for single-layout operands:
//!
//! $$A \otimes B \;=\; \left(A,\; \mathrm{complement}(A) \circ B\right)$$
//!
//! When $B$ is a tuple tiler, [`logical_product`] distributes **by-mode** across the
//! modes of $A$ (leaving trailing modes untouched and skipping `Nil` placeholders).
//! A non-`Layout` tiler $A$ (scalar or tuple) is promoted via [`tiler_to_layout`]
//! before $B$ is applied. Unlike [`logical_divide`](super::logical_divide()),
//! `Nil` is not a valid left-hand operand because there is no identity tile to reproduce.
//!
//! For equal-rank layouts $A$ and $B$, the rank-sensitive variants pair mode $i$ of $A$
//! with mode $i$ of $A^* \circ B$:
//! - [`blocked_product`]: puts each tile mode before its grid mode so tiles remain contiguous;
//! - [`raked_product`]: puts each grid mode before its tile mode for cyclic interleaving.
//!
//! # Main Interfaces
//!
//! - [`LogicalProduct`] / [`logical_product`] / [`Layout::logical_product`]
//! - [`LogicalProductAt`] / [`logical_product_at`] / [`Layout::logical_product_at`]
//! - [`BlockedProduct`] / [`blocked_product`] / [`Layout::blocked_product`]
//! - [`RakedProduct`] / [`raked_product`] / [`Layout::raked_product`]
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::logical_product::{blocked_product, logical_product, raked_product};
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::int_tuple;
//!
//! let tile = Layout::new(int_tuple![Int::<2>, Int::<5>], int_tuple![Int::<5>, Int::<1>]);
//! let grid = Layout::new(int_tuple![Int::<3>, Int::<4>], int_tuple![Int::<1>, Int::<3>]);
//!
//! assert_eq!(
//!     blocked_product(tile, grid),
//!     Layout::new(
//!         int_tuple![(Int::<2>, Int::<3>), (Int::<5>, Int::<4>)],
//!         int_tuple![(Int::<5>, Int::<10>), (Int::<1>, Int::<30>)],
//!     )
//! );
//! assert_eq!(
//!     raked_product(tile, grid),
//!     Layout::new(
//!         int_tuple![(Int::<3>, Int::<2>), (Int::<4>, Int::<5>)],
//!         int_tuple![(Int::<10>, Int::<5>), (Int::<30>, Int::<1>)],
//!     )
//! );
//! assert_eq!(
//!     logical_product(Layout::new(Int::<3>, Int::<1>), Layout::new(Int::<4>, Int::<1>)),
//!     Layout::new(int_tuple![Int::<3>, Int::<4>], int_tuple![Int::<1>, Int::<3>])
//! );
//! ```
//!
//! # Notes
//!
//! - `Nil` is rejected as a left-hand operand (both directly and with a mode path).
//! - A by-mode tiler `B` may not out-rank `A`, and `blocked_product` / `raked_product` require
//!   equal ranks.
//! - Invalid complements, negative dynamic extents, out-of-bounds mode paths, or stride
//!   overflows fail at compile time for static operands and panic at runtime for dynamic
//!   operands.
//!
//! # Compile-Time Rejections
//!
//! `Nil` is rejected as a left-hand operand (both directly and with a mode path):
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::logical_product::logical_product;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::tuple::Nil;
//!
//! let _ = logical_product(Nil, Layout::new(Int::<2>, Int::<1>));
//! ```
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::logical_product::logical_product_at;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::mode;
//! use noodle_cute::tuple::Nil;
//!
//! let _ = logical_product_at::<mode![1], _, _>(Nil, Layout::new(Int::<2>, Int::<1>));
//! ```
//!
//! A by-mode tiler `B` may not out-rank `A`:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::logical_product::logical_product;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::tuple;
//!
//! let _ = logical_product(
//!     Int::<6>,
//!     tuple![Layout::new(Int::<2>, Int::<1>), Layout::new(Int::<2>, Int::<1>)],
//! );
//! ```
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::logical_product::logical_product;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::{int_tuple, tuple};
//!
//! let a = Layout::new(int_tuple![Int::<6>, Int::<4>], int_tuple![Int::<4>, Int::<1>]);
//! let _ = logical_product(
//!     a,
//!     tuple![
//!         Layout::new(Int::<2>, Int::<1>),
//!         Layout::new(Int::<2>, Int::<1>),
//!         Layout::new(Int::<2>, Int::<1>),
//!     ],
//! );
//! ```
//!
//! Out-of-bounds mode paths are rejected at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::logical_product::logical_product_at;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::{int_tuple, mode};
//!
//! let a = Layout::new(int_tuple![Int::<3>, Int::<5>], int_tuple![Int::<1>, Int::<20>]);
//! let _ = logical_product_at::<mode![2], _, _>(a, Layout::new(Int::<4>, Int::<1>));
//! ```
//!
//! ```compile_fail,E0080
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::logical_product::logical_product_at;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::{int_tuple, mode};
//!
//! let a = Layout::new(int_tuple![Int::<3>, Int::<5>], int_tuple![Int::<1>, Int::<20>]);
//! let _ = logical_product_at::<mode![-3], _, _>(a, Layout::new(Int::<4>, Int::<1>));
//! ```
//!
//! `blocked_product` and `raked_product` reject mismatched ranks and non-layout tuple tilers
//! at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::logical_product::blocked_product;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::int_tuple;
//!
//! let _ = blocked_product(
//!     Layout::from_shape(Int::<4>),
//!     Layout::from_shape(int_tuple![Int::<2>, Int::<3>]),
//! );
//! ```
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::logical_product::raked_product;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::int_tuple;
//!
//! let _ = raked_product(
//!     Layout::from_shape(Int::<4>),
//!     Layout::from_shape(int_tuple![Int::<2>, Int::<3>]),
//! );
//! ```
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::logical_product::blocked_product;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::{int_tuple, tuple};
//!
//! let tile = Layout::new(int_tuple![Int::<2>, Int::<5>], int_tuple![Int::<5>, Int::<1>]);
//! let _ = blocked_product(tile, tuple![Int::<3>, Int::<4>]);
//! ```
//!
//! Non-injective static `A` is rejected at compile time (via `complement`):
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::logical_product::logical_product;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::int_tuple;
//!
//! let _ = logical_product(
//!     Layout::new(int_tuple![Int::<4>, Int::<2>], int_tuple![Int::<1>, Int::<2>]),
//!     Layout::new(Int::<2>, Int::<1>),
//! );
//! ```

use crate::basis::{Mode, ModeEnd};
use crate::integer::{Int, int};
use crate::layout::{Layout, Tiler, tiler_to_layout};
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tuple::{Get, Item, Nil, ReplaceAt, Til, Tuple, Wrap};

use super::complement::Complement;
use super::composition::Composition;

/// Reproduces layout `self` over the tile layout `B`: $A \otimes B = (A, A^* \circ B)$.
pub const trait LogicalProduct<B>: Copy {
    /// Resulting product [`Layout`].
    type Output: Copy;

    /// Computes logical product, panicking on dynamic failure or overflow.
    fn logical_product(self, b: B) -> Self::Output;
}

/// Applies [`LogicalProduct`] only to the sub-mode selected by type-level `Path` ([`Mode`] / [`ModeEnd`]).
pub const trait LogicalProductAt<B, Path>: Copy {
    /// Resulting layout with only `Path` reproduced and all sibling modes preserved.
    type Output: Copy;

    /// Mode-targeted logical product.
    fn logical_product_at(self, b: B) -> Self::Output;
}

/// Rank-sensitive blocked product pairing each mode of `self` before the corresponding mode of $A^* \circ B$.
pub const trait BlockedProduct<B>: Copy {
    /// Resulting blocked layout with `rank(Output) == rank(Self) == rank(B)`.
    type Output: Copy;

    /// Computes blocked product, panicking on dynamic failure or overflow.
    fn blocked_product(self, b: B) -> Self::Output;
}

/// Rank-sensitive raked (cyclic) product pairing each mode of $A^* \circ B$ before the corresponding mode of `self`.
pub const trait RakedProduct<B>: Copy {
    /// Resulting raked layout with `rank(Output) == rank(Self) == rank(B)`.
    type Output: Copy;

    /// Computes raked product, panicking on dynamic failure or overflow.
    fn raked_product(self, b: B) -> Self::Output;
}

/// By-mode tuple product helper zipping wrapped `A` modes against tuple tiler `B`.
#[doc(hidden)]
pub const trait LogicalProductTuple<B>: Copy {
    type Shape: Tuple + Shape;
    type Stride: Tuple + Stride<Self::Shape>;

    fn logical_product_tuple(self, b: B) -> Layout<Self::Shape, Self::Stride>;
}

const impl<SA: Shape + Tuple, DA: Stride<SA> + Tuple> LogicalProductTuple<Nil> for Layout<SA, DA> {
    type Shape = SA;
    type Stride = DA;

    fn logical_product_tuple(self, _: Nil) -> Layout<SA, DA> {
        self
    }
}

const impl<ASH, AST, ADH, ADT, BH, BT, HS, HD> LogicalProductTuple<Til<BH, BT>>
    for Layout<Til<ASH, AST>, Til<ADH, ADT>>
where
    ASH: Shape,
    AST: Tuple + Shape,
    ADH: Stride<ASH>,
    ADT: Tuple + Stride<AST>,
    BH: Copy,
    BT: Copy,
    Layout<ASH, ADH>: [const] LogicalProduct<BH, Output = Layout<HS, HD>>,
    HS: Shape,
    HD: Stride<HS>,
    Layout<AST, ADT>: [const] LogicalProductTuple<BT>,
{
    type Shape = Til<HS, <Layout<AST, ADT> as LogicalProductTuple<BT>>::Shape>;
    type Stride = Til<HD, <Layout<AST, ADT> as LogicalProductTuple<BT>>::Stride>;

    fn logical_product_tuple(self, b: Til<BH, BT>) -> Layout<Self::Shape, Self::Stride> {
        let head = Layout::new(self.shape.0, self.stride.0).logical_product(b.0);
        let tail = Layout::new(self.shape.1, self.stride.1).logical_product_tuple(b.1);
        Layout::new(Til(head.shape, tail.shape), Til(head.stride, tail.stride))
    }
}

/// Pairwise mode zipping of two equal-rank tuple layouts into `((L_0, R_0), (L_1, R_1), ...)`.
///
/// Shared by [`BlockedProduct`] (`zip_modes(tile, grid)`) and [`RakedProduct`] (`zip_modes(grid, tile)`).
#[doc(hidden)]
pub const trait ZipModes<R>: Copy {
    type Shape: Tuple + Shape;
    type Stride: Tuple + Stride<Self::Shape>;

    fn zip_modes(self, rhs: R) -> Layout<Self::Shape, Self::Stride>;
}

const impl ZipModes<Layout<Nil, Nil>> for Layout<Nil, Nil> {
    type Shape = Nil;
    type Stride = Nil;

    fn zip_modes(self, _: Layout<Nil, Nil>) -> Layout<Nil, Nil> {
        Layout::new(Nil, Nil)
    }
}

const impl<LSH, LST, LDH, LDT, RSH, RST, RDH, RDT> ZipModes<Layout<Til<RSH, RST>, Til<RDH, RDT>>>
    for Layout<Til<LSH, LST>, Til<LDH, LDT>>
where
    LSH: Shape,
    LST: Tuple + Shape,
    LDH: Stride<LSH>,
    LDT: Tuple + Stride<LST>,
    RSH: Shape,
    RST: Tuple + Shape,
    RDH: Stride<RSH>,
    RDT: Tuple + Stride<RST>,
    Layout<LST, LDT>: [const] ZipModes<Layout<RST, RDT>>,
{
    type Shape = Til<
        Til<LSH, Til<RSH, Nil>>,
        <Layout<LST, LDT> as ZipModes<Layout<RST, RDT>>>::Shape,
    >;
    type Stride = Til<
        Til<LDH, Til<RDH, Nil>>,
        <Layout<LST, LDT> as ZipModes<Layout<RST, RDT>>>::Stride,
    >;

    fn zip_modes(
        self,
        rhs: Layout<Til<RSH, RST>, Til<RDH, RDT>>,
    ) -> Layout<Self::Shape, Self::Stride> {
        let tail = Layout::new(self.shape.1, self.stride.1)
            .zip_modes(Layout::new(rhs.shape.1, rhs.stride.1));
        Layout::new(
            Til(Til(self.shape.0, Til(rhs.shape.0, Nil)), tail.shape),
            Til(Til(self.stride.0, Til(rhs.stride.0, Nil)), tail.stride),
        )
    }
}

const impl<SA: Shape, DA: Stride<SA>> LogicalProduct<Nil> for Layout<SA, DA> {
    type Output = Self;

    fn logical_product(self, _: Nil) -> Self {
        self
    }
}

macro_rules! impl_layout_logical_product_scalar {
    ([$($g:tt)*] $S:ty) => {
        const impl<SA, DA, $($g)*> LogicalProduct<$S> for Layout<SA, DA>
        where
            SA: Shape,
            DA: Stride<SA>,
            $S: Shape,
            Int<1>: Stride<$S>,
            Self: [const] LogicalProduct<Layout<$S, Int<1>>>,
        {
            type Output = <Self as LogicalProduct<Layout<$S, Int<1>>>>::Output;

            fn logical_product(self, b: $S) -> Self::Output {
                self.logical_product(Layout::new(b, Int::<1>))
            }
        }
    };
}

impl_layout_logical_product_scalar!([] int);
impl_layout_logical_product_scalar!([const N: int] Int<N>);

const impl<SA, DA, SB, DB, RS, RD> LogicalProduct<Layout<SB, DB>> for Layout<SA, DA>
where
    SA: Shape,
    DA: Stride<SA>,
    SB: Shape,
    DB: Stride<SB>,
    Self: [const] Complement,
    <Self as Complement>::Output: [const] Composition<Layout<SB, DB>, Output = Layout<RS, RD>>,
    RS: Shape,
    RD: Stride<RS>,
{
    type Output = Layout<Til<SA, Til<RS, Nil>>, Til<DA, Til<RD, Nil>>>;

    fn logical_product(self, b: Layout<SB, DB>) -> Self::Output {
        let comp = self.complement();
        let rhs = comp.composition(b);
        Layout::new(
            Til(self.shape, Til(rhs.shape, Nil)),
            Til(self.stride, Til(rhs.stride, Nil)),
        )
    }
}

const impl<SA, DA, BH: Copy, BT: Copy> LogicalProduct<Til<BH, BT>> for Layout<SA, DA>
where
    SA: Shape + [const] Wrap,
    DA: Stride<SA> + [const] Wrap,
    SA::Wrapped: Tuple + Shape,
    DA::Wrapped: Tuple + Stride<SA::Wrapped>,
    Layout<SA::Wrapped, DA::Wrapped>: [const] LogicalProductTuple<Til<BH, BT>>,
{
    type Output = Layout<
        <Layout<SA::Wrapped, DA::Wrapped> as LogicalProductTuple<Til<BH, BT>>>::Shape,
        <Layout<SA::Wrapped, DA::Wrapped> as LogicalProductTuple<Til<BH, BT>>>::Stride,
    >;

    fn logical_product(self, b: Til<BH, BT>) -> Self::Output {
        Layout::new(self.shape.wrap(), self.stride.wrap()).logical_product_tuple(b)
    }
}

macro_rules! impl_tiler_logical_product {
    ([$($g:tt)*] $T:ty) => {
        const impl<$($g)* B> LogicalProduct<B> for $T
        where
            Self: [const] Tiler,
            <Self as Tiler>::OutputLayout: [const] LogicalProduct<B>,
        {
            type Output = <<Self as Tiler>::OutputLayout as LogicalProduct<B>>::Output;

            fn logical_product(self, b: B) -> Self::Output {
                tiler_to_layout(self).logical_product(b)
            }
        }
    };
}

impl_tiler_logical_product!([] int);
impl_tiler_logical_product!([const N: int,] Int<N>);
impl_tiler_logical_product!([H: Item, T: Tuple,] Til<H, T>);

const impl<SA, DA, B> LogicalProductAt<B, ModeEnd> for Layout<SA, DA>
where
    SA: Shape,
    DA: Stride<SA>,
    Self: [const] LogicalProduct<B>,
{
    type Output = <Self as LogicalProduct<B>>::Output;

    fn logical_product_at(self, b: B) -> Self::Output {
        self.logical_product(b)
    }
}

const impl<SA, DA, B, const I: int, Next, SubS, SubD, SubOutS, SubOutD>
    LogicalProductAt<B, Mode<I, Next>> for Layout<SA, DA>
where
    SA: Shape + [const] Wrap,
    DA: Stride<SA> + [const] Wrap,
    SA::Wrapped: [const] Get<I, Output = SubS>,
    DA::Wrapped: [const] Get<I, Output = SubD>,
    SubS: Shape,
    SubD: Stride<SubS>,
    Layout<SubS, SubD>: [const] LogicalProductAt<B, Next, Output = Layout<SubOutS, SubOutD>>,
    SubOutS: [const] Item + Shape,
    SubOutD: [const] Item + Stride<SubOutS>,
    SA::Wrapped: [const] ReplaceAt<SubOutS, I>,
    DA::Wrapped: [const] ReplaceAt<SubOutD, I>,
    <SA::Wrapped as ReplaceAt<SubOutS, I>>::Output: Shape,
    <DA::Wrapped as ReplaceAt<SubOutD, I>>::Output:
        Stride<<SA::Wrapped as ReplaceAt<SubOutS, I>>::Output>,
{
    type Output = Layout<
        <SA::Wrapped as ReplaceAt<SubOutS, I>>::Output,
        <DA::Wrapped as ReplaceAt<SubOutD, I>>::Output,
    >;

    fn logical_product_at(self, b: B) -> Self::Output {
        let s_wrap = self.shape.wrap();
        let d_wrap = self.stride.wrap();
        let sub = Layout::new(
            <SA::Wrapped as Get<I>>::get(&s_wrap),
            <DA::Wrapped as Get<I>>::get(&d_wrap),
        );
        let prod_sub =
            <Layout<SubS, SubD> as LogicalProductAt<B, Next>>::logical_product_at(sub, b);
        Layout::new(
            <SA::Wrapped as ReplaceAt<SubOutS, I>>::replace_at(&s_wrap, prod_sub.shape),
            <DA::Wrapped as ReplaceAt<SubOutD, I>>::replace_at(&d_wrap, prod_sub.stride),
        )
    }
}

macro_rules! impl_tiler_logical_product_at {
    ([$($g:tt)*] $T:ty) => {
        const impl<$($g)* B, Path> LogicalProductAt<B, Path> for $T
        where
            Self: [const] Tiler,
            <Self as Tiler>::OutputLayout: [const] LogicalProductAt<B, Path>,
        {
            type Output = <<Self as Tiler>::OutputLayout as LogicalProductAt<B, Path>>::Output;

            fn logical_product_at(self, b: B) -> Self::Output {
                tiler_to_layout(self).logical_product_at(b)
            }
        }
    };
}

impl_tiler_logical_product_at!([] int);
impl_tiler_logical_product_at!([const N: int,] Int<N>);
impl_tiler_logical_product_at!([H: Item, T: Tuple,] Til<H, T>);

type WrappedLayout<S, D> = Layout<<S as Wrap>::Wrapped, <D as Wrap>::Wrapped>;

macro_rules! impl_layout_rank_product_scalar {
    ($Trait:ident, $method:ident, [$($g:tt)*] $S:ty) => {
        const impl<SA, DA, $($g)*> $Trait<$S> for Layout<SA, DA>
        where
            SA: Shape,
            DA: Stride<SA>,
            $S: Shape,
            Int<1>: Stride<$S>,
            Self: [const] $Trait<Layout<$S, Int<1>>>,
        {
            type Output = <Self as $Trait<Layout<$S, Int<1>>>>::Output;

            fn $method(self, b: $S) -> Self::Output {
                self.$method(Layout::new(b, Int::<1>))
            }
        }
    };
}

impl_layout_rank_product_scalar!(BlockedProduct, blocked_product, [] int);
impl_layout_rank_product_scalar!(BlockedProduct, blocked_product, [const N: int] Int<N>);
impl_layout_rank_product_scalar!(RakedProduct, raked_product, [] int);
impl_layout_rank_product_scalar!(RakedProduct, raked_product, [const N: int] Int<N>);

const impl<SA, DA, SB, DB, RS, RD> BlockedProduct<Layout<SB, DB>> for Layout<SA, DA>
where
    SA: Shape + [const] Wrap,
    DA: Stride<SA> + [const] Wrap,
    SA::Wrapped: Tuple + Shape,
    DA::Wrapped: Tuple + Stride<SA::Wrapped>,
    SB: Shape + [const] Wrap,
    DB: Stride<SB> + [const] Wrap,
    SB::Wrapped: Tuple + Shape,
    DB::Wrapped: Tuple + Stride<SB::Wrapped>,
    Self: [const] Complement,
    <Self as Complement>::Output:
        [const] Composition<WrappedLayout<SB, DB>, Output = Layout<RS, RD>>,
    RS: Tuple + Shape,
    RD: Tuple + Stride<RS>,
    WrappedLayout<SA, DA>: [const] ZipModes<Layout<RS, RD>>,
{
    type Output = Layout<
        <WrappedLayout<SA, DA> as ZipModes<Layout<RS, RD>>>::Shape,
        <WrappedLayout<SA, DA> as ZipModes<Layout<RS, RD>>>::Stride,
    >;

    fn blocked_product(self, b: Layout<SB, DB>) -> Self::Output {
        let comp = self.complement();
        let b_wrap = Layout::new(b.shape.wrap(), b.stride.wrap());
        let rhs = comp.composition(b_wrap);
        Layout::new(self.shape.wrap(), self.stride.wrap()).zip_modes(rhs)
    }
}

const impl<SA, DA, SB, DB, RS, RD> RakedProduct<Layout<SB, DB>> for Layout<SA, DA>
where
    SA: Shape + [const] Wrap,
    DA: Stride<SA> + [const] Wrap,
    SA::Wrapped: Tuple + Shape,
    DA::Wrapped: Tuple + Stride<SA::Wrapped>,
    SB: Shape + [const] Wrap,
    DB: Stride<SB> + [const] Wrap,
    SB::Wrapped: Tuple + Shape,
    DB::Wrapped: Tuple + Stride<SB::Wrapped>,
    Self: [const] Complement,
    <Self as Complement>::Output:
        [const] Composition<WrappedLayout<SB, DB>, Output = Layout<RS, RD>>,
    RS: Tuple + Shape,
    RD: Tuple + Stride<RS>,
    Layout<RS, RD>: [const] ZipModes<WrappedLayout<SA, DA>>,
{
    type Output = Layout<
        <Layout<RS, RD> as ZipModes<WrappedLayout<SA, DA>>>::Shape,
        <Layout<RS, RD> as ZipModes<WrappedLayout<SA, DA>>>::Stride,
    >;

    fn raked_product(self, b: Layout<SB, DB>) -> Self::Output {
        let comp = self.complement();
        let b_wrap = Layout::new(b.shape.wrap(), b.stride.wrap());
        let rhs = comp.composition(b_wrap);
        rhs.zip_modes(Layout::new(self.shape.wrap(), self.stride.wrap()))
    }
}

macro_rules! impl_tiler_rank_product {
    ($Trait:ident, $method:ident, [$($g:tt)*] $T:ty) => {
        const impl<$($g)* B> $Trait<B> for $T
        where
            Self: [const] Tiler,
            <Self as Tiler>::OutputLayout: [const] $Trait<B>,
        {
            type Output = <<Self as Tiler>::OutputLayout as $Trait<B>>::Output;

            fn $method(self, b: B) -> Self::Output {
                tiler_to_layout(self).$method(b)
            }
        }
    };
}

impl_tiler_rank_product!(BlockedProduct, blocked_product, [] int);
impl_tiler_rank_product!(BlockedProduct, blocked_product, [const N: int,] Int<N>);
impl_tiler_rank_product!(BlockedProduct, blocked_product, [H: Item, T: Tuple,] Til<H, T>);
impl_tiler_rank_product!(RakedProduct, raked_product, [] int);
impl_tiler_rank_product!(RakedProduct, raked_product, [const N: int,] Int<N>);
impl_tiler_rank_product!(RakedProduct, raked_product, [H: Item, T: Tuple,] Til<H, T>);

impl<S, D> Layout<S, D> {
    /// Reproduces `self` across tiler `b`: $A \otimes B = (A, A^* \circ B)$.
    pub const fn logical_product<B>(self, b: B) -> <Self as LogicalProduct<B>>::Output
    where
        Self: [const] LogicalProduct<B>,
    {
        LogicalProduct::logical_product(self, b)
    }

    /// Reproduces only the sub-mode selected by `Path` (`mode![...]`) across tiler `b`.
    pub const fn logical_product_at<Path, B>(
        self,
        b: B,
    ) -> <Self as LogicalProductAt<B, Path>>::Output
    where
        Self: [const] LogicalProductAt<B, Path>,
    {
        LogicalProductAt::logical_product_at(self, b)
    }

    /// Rank-sensitive blocked product placing each tile mode before its grid mode.
    pub const fn blocked_product<B>(self, b: B) -> <Self as BlockedProduct<B>>::Output
    where
        Self: [const] BlockedProduct<B>,
    {
        BlockedProduct::blocked_product(self, b)
    }

    /// Rank-sensitive raked product placing each grid mode before its tile mode.
    pub const fn raked_product<B>(self, b: B) -> <Self as RakedProduct<B>>::Output
    where
        Self: [const] RakedProduct<B>,
    {
        RakedProduct::raked_product(self, b)
    }
}

/// Computes the logical product `a x b` (`(a, complement(a) o b)`).
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::algebra::logical_product::logical_product;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
/// use noodle_cute::int_tuple;
///
/// let res = logical_product(
///     Layout::new(int_tuple![Int::<2>, Int::<2>], int_tuple![Int::<4>, Int::<1>]),
///     Layout::new(Int::<6>, Int::<1>),
/// );
/// assert_eq!(
///     res,
///     Layout::new(
///         int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<3>)],
///         int_tuple![(Int::<4>, Int::<1>), (Int::<2>, Int::<8>)],
///     )
/// );
/// ```
pub const fn logical_product<A, B>(a: A, b: B) -> <A as LogicalProduct<B>>::Output
where
    A: [const] LogicalProduct<B>,
{
    a.logical_product(b)
}

/// Computes [`logical_product`] at the sub-mode selected by `Path` (`mode![...]`).
pub const fn logical_product_at<Path, A, B>(a: A, b: B) -> <A as LogicalProductAt<B, Path>>::Output
where
    A: [const] LogicalProductAt<B, Path>,
{
    a.logical_product_at(b)
}

/// Computes the rank-sensitive blocked product of tile `a` arranged over grid `b`.
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::algebra::logical_product::blocked_product;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
/// use noodle_cute::int_tuple;
///
/// let tile = Layout::new(int_tuple![Int::<2>, Int::<5>], int_tuple![Int::<5>, Int::<1>]);
/// let grid = Layout::new(int_tuple![Int::<3>, Int::<4>], int_tuple![Int::<1>, Int::<3>]);
/// assert_eq!(
///     blocked_product(tile, grid),
///     Layout::new(
///         int_tuple![(Int::<2>, Int::<3>), (Int::<5>, Int::<4>)],
///         int_tuple![(Int::<5>, Int::<10>), (Int::<1>, Int::<30>)],
///     )
/// );
/// ```
pub const fn blocked_product<A, B>(a: A, b: B) -> <A as BlockedProduct<B>>::Output
where
    A: [const] BlockedProduct<B>,
{
    a.blocked_product(b)
}

/// Computes the rank-sensitive raked (cyclic) product of tile `a` distributed through grid `b`.
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::algebra::logical_product::raked_product;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
/// use noodle_cute::int_tuple;
///
/// let tile = Layout::new(int_tuple![Int::<2>, Int::<5>], int_tuple![Int::<5>, Int::<1>]);
/// let grid = Layout::new(int_tuple![Int::<3>, Int::<4>], int_tuple![Int::<1>, Int::<3>]);
/// assert_eq!(
///     raked_product(tile, grid),
///     Layout::new(
///         int_tuple![(Int::<3>, Int::<2>), (Int::<4>, Int::<5>)],
///         int_tuple![(Int::<10>, Int::<5>), (Int::<30>, Int::<1>)],
///     )
/// );
/// ```
pub const fn raked_product<A, B>(a: A, b: B) -> <A as RakedProduct<B>>::Output
where
    A: [const] RakedProduct<B>,
{
    a.raked_product(b)
}
