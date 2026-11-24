//! Layout logical division ($A \oslash B$) and zipped division algebra.
//!
//! Logical division splits a layout $A$ into two levels with respect to a tiler $B$:
//! - Mode 0 (the **tile**): elements selected by $B$, i.e. $A \circ B$;
//! - Mode 1 (the **grid**): layout of those tiles across $A$'s domain, i.e. $A \circ B^*_{\mathrm{ext}}$
//!   where $B^*_{\mathrm{ext}} = \mathrm{complement}(B, \mathrm{shape}(A))$.
//!
//! Formally, for single-layout operands:
//!
//! $$A \oslash B \;=\; A \circ \left(B,\; \mathrm{complement}(B, \mathrm{shape}(A))\right)$$
//!
//! When $B$ is a tuple tiler, [`logical_divide`] distributes **by-mode** across the
//! modes of $A$ (leaving trailing modes untouched and skipping `Nil` placeholders),
//! whereas [`zipped_divide`] first promotes $B$ via [`tiler_to_layout`] and collects
//! all tile modes into mode 0 and all grid modes into mode 1.
//!
//! # Main Interfaces
//!
//! - [`LogicalDivide`] / [`logical_divide`] / [`Layout::logical_divide`]
//! - [`ZippedDivide`] / [`zipped_divide`] / [`Layout::zipped_divide`]
//! - [`LogicalDivideAt`] / [`logical_divide_at`] / [`Layout::logical_divide_at`]
//! - [`ZippedDivideAt`] / [`zipped_divide_at`] / [`Layout::zipped_divide_at`]
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::logical_divide::{logical_divide, zipped_divide};
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::{int_tuple, tuple};
//!
//! let a = Layout::new(int_tuple![Int::<6>, Int::<4>], int_tuple![Int::<4>, Int::<1>]);
//! let tiler = tuple![Layout::new(Int::<2>, Int::<1>), Layout::new(Int::<2>, Int::<1>)];
//!
//! // By-mode division keeps each mode paired with its own grid: ((2, 3), (2, 2))
//! let by_mode = logical_divide(a, tiler);
//! assert_eq!(
//!     by_mode,
//!     Layout::new(
//!         int_tuple![(Int::<2>, Int::<3>), (Int::<2>, Int::<2>)],
//!         int_tuple![(Int::<4>, Int::<8>), (Int::<1>, Int::<2>)],
//!     )
//! );
//!
//! // Zipped division groups all tile modes in mode 0 and all grid modes in mode 1
//! let zipped = zipped_divide(a, tiler);
//! assert_eq!(
//!     zipped,
//!     Layout::new(
//!         int_tuple![(Int::<2>, Int::<2>), (Int::<3>, Int::<2>)],
//!         int_tuple![(Int::<4>, Int::<1>), (Int::<8>, Int::<2>)],
//!     )
//! );
//! ```
//!
//! # Notes
//!
//! - Invalid complements, negative dynamic extents, out-of-bounds mode paths, or stride
//!   overflows fail at compile time for static operands and panic at runtime for dynamic
//!   operands.

use crate::basis::{BasisPath, Mode, ModeEnd};
use crate::integer::{Int, int};
use crate::layout::{Layout, LayoutBase, MakeLayout, Tiler, tiler_to_layout};
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tuple::{Get, Item, Lift, Nil, ReplaceAt, Til, Tuple, Wrap, lift};

use super::complement::{Complement, ComplementWith};
use super::composition::Composition;

/// Splits layout `self` by tiler `B`: $A \oslash B = A \circ (B, B^*)$.
pub const trait LogicalDivide<B>: Copy {
    /// Resulting divided [`Layout`] (or [`Nil`] when both `self` and `B` are [`Nil`]).
    type Output: Copy;

    /// Computes logical division, panicking on dynamic failure or overflow.
    fn logical_divide(self, b: B) -> Self::Output;
}

/// Promotes tiler `B` via [`tiler_to_layout`] before dividing, regrouping modes as `(Tile, Grid)`.
pub const trait ZippedDivide<B>: Copy {
    /// Resulting `(Tile, Grid)` layout (or [`Nil`]).
    type Output: Copy;

    /// Computes zipped division, panicking on dynamic failure or overflow.
    fn zipped_divide(self, b: B) -> Self::Output;
}

/// Applies [`LogicalDivide`] only to the sub-mode selected by type-level `Path` ([`Mode`] / [`ModeEnd`]).
pub const trait LogicalDivideAt<B, Path>: Copy {
    /// Resulting layout with only `Path` divided and all sibling modes preserved.
    type Output: Copy;

    /// Mode-targeted logical division.
    fn logical_divide_at(self, b: B) -> Self::Output;
}

/// Applies [`ZippedDivide`] only to the sub-mode selected by type-level `Path` ([`Mode`] / [`ModeEnd`]).
pub const trait ZippedDivideAt<B, Path>: Copy {
    /// Resulting layout with `Path` zipped-divided and all sibling modes preserved.
    type Output: Copy;

    /// Mode-targeted zipped division.
    fn zipped_divide_at(self, b: B) -> Self::Output;
}

/// Transforms each leaf of a non-`Nil` tiler into `make_layout!(leaf, complement(leaf))`
/// for `logical_divide(Nil, B)`.
#[doc(hidden)]
pub const trait FreeDivideTilerHead: Item {
    type Output: Item;
    fn free_divide_head(self) -> Self::Output;
}

/// Tail helper for [`FreeDivideTilerHead`], preserving terminating [`Nil`].
#[doc(hidden)]
pub const trait FreeDivideTilerTail: Tuple {
    type Output: Tuple;
    fn free_divide_tail(self) -> Self::Output;
}

const impl FreeDivideTilerHead for Nil {
    type Output = Layout<Til<Int<1>, Til<Int<1>, Nil>>, Til<Int<1>, Til<Int<1>, Nil>>>;

    fn free_divide_head(self) -> Self::Output {
        Layout::new(
            Til(Int::<1>, Til(Int::<1>, Nil)),
            Til(Int::<1>, Til(Int::<1>, Nil)),
        )
    }
}

macro_rules! impl_free_divide_scalar {
    ([$($g:tt)*] $S:ty) => {
        const impl<$($g)*> FreeDivideTilerHead for $S
        where
            $S: Shape,
            Int<1>: Stride<$S>,
            Layout<$S, Int<1>>: [const] Complement,
            <Layout<$S, Int<1>> as Complement>::Output: [const] LayoutBase,
        {
            type Output = Layout<
                Til<$S, Til<<<Layout<$S, Int<1>> as Complement>::Output as LayoutBase>::Shape, Nil>>,
                Til<
                    Int<1>,
                    Til<<<Layout<$S, Int<1>> as Complement>::Output as LayoutBase>::Stride, Nil>,
                >,
            >;

            fn free_divide_head(self) -> Self::Output {
                let bl = Layout::new(self, Int::<1>);
                let comp = bl.complement();
                Layout::new(
                    Til(bl.shape, Til(comp.shape(), Nil)),
                    Til(bl.stride, Til(comp.stride(), Nil)),
                )
            }
        }
    };
}

impl_free_divide_scalar!([] int);
impl_free_divide_scalar!([const N: int,] Int<N>);

const impl<SB, DB> FreeDivideTilerHead for Layout<SB, DB>
where
    SB: Shape,
    DB: Stride<SB>,
    Self: [const] Complement,
    <Self as Complement>::Output: [const] LayoutBase,
{
    type Output = Layout<
        Til<SB, Til<<<Self as Complement>::Output as LayoutBase>::Shape, Nil>>,
        Til<DB, Til<<<Self as Complement>::Output as LayoutBase>::Stride, Nil>>,
    >;

    fn free_divide_head(self) -> Self::Output {
        let comp = self.complement();
        Layout::new(
            Til(self.shape, Til(comp.shape(), Nil)),
            Til(self.stride, Til(comp.stride(), Nil)),
        )
    }
}

const impl<BH, BT> FreeDivideTilerHead for Til<BH, BT>
where
    BH: [const] FreeDivideTilerHead,
    BT: Tuple + [const] FreeDivideTilerTail,
{
    type Output = Til<BH::Output, BT::Output>;

    fn free_divide_head(self) -> Self::Output {
        Til(self.0.free_divide_head(), self.1.free_divide_tail())
    }
}

const impl FreeDivideTilerTail for Nil {
    type Output = Nil;

    fn free_divide_tail(self) -> Nil {
        Nil
    }
}

const impl<BH, BT> FreeDivideTilerTail for Til<BH, BT>
where
    BH: [const] FreeDivideTilerHead,
    BT: Tuple + [const] FreeDivideTilerTail,
{
    type Output = Til<BH::Output, BT::Output>;

    fn free_divide_tail(self) -> Self::Output {
        Til(self.0.free_divide_head(), self.1.free_divide_tail())
    }
}

const impl LogicalDivide<Nil> for Nil {
    type Output = Nil;

    fn logical_divide(self, _: Nil) -> Nil {
        Nil
    }
}

macro_rules! impl_nil_logical_divide_leaf {
    ([$($g:tt)*] $B:ty) => {
        const impl<$($g)*> LogicalDivide<$B> for Nil
        where
            $B: [const] FreeDivideTilerHead,
        {
            type Output = <$B as FreeDivideTilerHead>::Output;

            fn logical_divide(self, b: $B) -> Self::Output {
                b.free_divide_head()
            }
        }
    };
}

impl_nil_logical_divide_leaf!([] int);
impl_nil_logical_divide_leaf!([const N: int,] Int<N>);
impl_nil_logical_divide_leaf!([SB: Shape, DB: Stride<SB>,] Layout<SB, DB>);

const impl<BH, BT> LogicalDivide<Til<BH, BT>> for Nil
where
    Til<BH, BT>: [const] FreeDivideTilerTail,
    <Til<BH, BT> as FreeDivideTilerTail>::Output: [const] Tiler,
    <<Til<BH, BT> as FreeDivideTilerTail>::Output as Tiler>::OutputLayout: Copy,
{
    type Output = <<Til<BH, BT> as FreeDivideTilerTail>::Output as Tiler>::OutputLayout;

    fn logical_divide(self, b: Til<BH, BT>) -> Self::Output {
        tiler_to_layout(b.free_divide_tail())
    }
}

/// By-mode tuple division helper zipping wrapped `A` modes against tuple tiler `B`.
#[doc(hidden)]
pub const trait LogicalDivideTuple<B>: Copy {
    type Shape: Tuple + Shape;
    type Stride: Tuple + Stride<Self::Shape>;

    fn logical_divide_tuple(self, b: B) -> Layout<Self::Shape, Self::Stride>;
}

const impl<SA: Shape + Tuple, DA: Stride<SA> + Tuple> LogicalDivideTuple<Nil> for Layout<SA, DA> {
    type Shape = SA;
    type Stride = DA;

    fn logical_divide_tuple(self, _: Nil) -> Layout<SA, DA> {
        self
    }
}

const impl<ASH, AST, ADH, ADT, BH, BT, HS, HD> LogicalDivideTuple<Til<BH, BT>>
    for Layout<Til<ASH, AST>, Til<ADH, ADT>>
where
    ASH: Shape,
    AST: Tuple + Shape,
    ADH: Stride<ASH>,
    ADT: Tuple + Stride<AST>,
    BH: Copy,
    BT: Copy,
    Layout<ASH, ADH>: [const] LogicalDivide<BH, Output = Layout<HS, HD>>,
    HS: Shape,
    HD: Stride<HS>,
    Layout<AST, ADT>: [const] LogicalDivideTuple<BT>,
{
    type Shape = Til<HS, <Layout<AST, ADT> as LogicalDivideTuple<BT>>::Shape>;
    type Stride = Til<HD, <Layout<AST, ADT> as LogicalDivideTuple<BT>>::Stride>;

    fn logical_divide_tuple(self, b: Til<BH, BT>) -> Layout<Self::Shape, Self::Stride> {
        let head = Layout::new(self.shape.0, self.stride.0).logical_divide(b.0);
        let tail = Layout::new(self.shape.1, self.stride.1).logical_divide_tuple(b.1);
        Layout::new(Til(head.shape, tail.shape), Til(head.stride, tail.stride))
    }
}

const impl<SA: Shape, DA: Stride<SA>> LogicalDivide<Nil> for Layout<SA, DA> {
    type Output = Self;

    fn logical_divide(self, _: Nil) -> Self {
        self
    }
}

macro_rules! impl_layout_logical_divide_scalar {
    ([$($g:tt)*] $S:ty) => {
        const impl<SA, DA, $($g)*> LogicalDivide<$S> for Layout<SA, DA>
        where
            SA: Shape,
            DA: Stride<SA>,
            $S: Shape,
            Int<1>: Stride<$S>,
            Self: [const] LogicalDivide<Layout<$S, Int<1>>>,
        {
            type Output = <Self as LogicalDivide<Layout<$S, Int<1>>>>::Output;

            fn logical_divide(self, b: $S) -> Self::Output {
                self.logical_divide(Layout::new(b, Int::<1>))
            }
        }
    };
}

impl_layout_logical_divide_scalar!([] int);
impl_layout_logical_divide_scalar!([const N: int] Int<N>);

type LogicalDivideRhs<SB, DB, SA> = Layout<
    Til<SB, Til<<<Layout<SB, DB> as ComplementWith<SA>>::Output as LayoutBase>::Shape, Nil>>,
    Til<DB, Til<<<Layout<SB, DB> as ComplementWith<SA>>::Output as LayoutBase>::Stride, Nil>>,
>;

const impl<SA, DA, SB, DB> LogicalDivide<Layout<SB, DB>> for Layout<SA, DA>
where
    SA: [const] Shape,
    DA: Stride<SA>,
    SB: Shape,
    DB: Stride<SB>,
    Layout<SB, DB>: [const] ComplementWith<SA>,
    <Layout<SB, DB> as ComplementWith<SA>>::Output: [const] LayoutBase,
    Self: [const] Composition<LogicalDivideRhs<SB, DB, SA>>,
{
    type Output = <Self as Composition<LogicalDivideRhs<SB, DB, SA>>>::Output;

    fn logical_divide(self, b: Layout<SB, DB>) -> Self::Output {
        let comp = b.complement_with(self.shape);
        let rhs = Layout::new(
            Til(b.shape, Til(comp.shape(), Nil)),
            Til(b.stride, Til(comp.stride(), Nil)),
        );
        self.composition(rhs)
    }
}

const impl<SA, DA, BH: Copy, BT: Copy> LogicalDivide<Til<BH, BT>> for Layout<SA, DA>
where
    SA: Shape + [const] Wrap,
    DA: Stride<SA> + [const] Wrap,
    SA::Wrapped: Tuple + Shape,
    DA::Wrapped: Tuple + Stride<SA::Wrapped>,
    Layout<SA::Wrapped, DA::Wrapped>: [const] LogicalDivideTuple<Til<BH, BT>>,
{
    type Output = Layout<
        <Layout<SA::Wrapped, DA::Wrapped> as LogicalDivideTuple<Til<BH, BT>>>::Shape,
        <Layout<SA::Wrapped, DA::Wrapped> as LogicalDivideTuple<Til<BH, BT>>>::Stride,
    >;

    fn logical_divide(self, b: Til<BH, BT>) -> Self::Output {
        Layout::new(self.shape.wrap(), self.stride.wrap()).logical_divide_tuple(b)
    }
}

macro_rules! impl_tiler_logical_divide {
    ([$($g:tt)*] $T:ty) => {
        const impl<$($g)* B> LogicalDivide<B> for $T
        where
            Self: [const] Tiler,
            <Self as Tiler>::OutputLayout: [const] LogicalDivide<B>,
        {
            type Output = <<Self as Tiler>::OutputLayout as LogicalDivide<B>>::Output;

            fn logical_divide(self, b: B) -> Self::Output {
                tiler_to_layout(self).logical_divide(b)
            }
        }
    };
}

impl_tiler_logical_divide!([] int);
impl_tiler_logical_divide!([const N: int,] Int<N>);
impl_tiler_logical_divide!([H: Item, T: Tuple,] Til<H, T>);

const impl<A, B> ZippedDivide<B> for A
where
    B: [const] Tiler,
    A: [const] LogicalDivide<B::OutputLayout>,
{
    type Output = <A as LogicalDivide<B::OutputLayout>>::Output;

    fn zipped_divide(self, b: B) -> Self::Output {
        self.logical_divide(tiler_to_layout(b))
    }
}

const impl<B> LogicalDivideAt<B, ModeEnd> for Nil
where
    Nil: [const] LogicalDivide<B>,
{
    type Output = <Nil as LogicalDivide<B>>::Output;

    fn logical_divide_at(self, b: B) -> Self::Output {
        self.logical_divide(b)
    }
}

const impl<const I: int, Next> LogicalDivideAt<Nil, Mode<I, Next>> for Nil {
    type Output = Nil;

    fn logical_divide_at(self, _: Nil) -> Nil {
        Nil
    }
}

macro_rules! impl_nil_logical_divide_at_mode {
    ([$($g:tt)*] $B:ty) => {
        const impl<$($g)* const I: int, Next, InnerS, InnerD>
            LogicalDivideAt<$B, Mode<I, Next>> for Nil
        where
            Mode<I, Next>: [const] BasisPath,
            Nil: [const] LogicalDivide<$B, Output = Layout<InnerS, InnerD>>,
            InnerS: Shape,
            InnerD: Stride<InnerS>,
            Layout<InnerS, InnerD>:
                [const] Lift<MakeLayout, Layout<Int<1>, Int<0>>, Mode<I, Next>>,
            <Layout<InnerS, InnerD> as Lift<
                MakeLayout,
                Layout<Int<1>, Int<0>>,
                Mode<I, Next>,
            >>::Output: Copy,
        {
            type Output = <Layout<InnerS, InnerD> as Lift<
                MakeLayout,
                Layout<Int<1>, Int<0>>,
                Mode<I, Next>,
            >>::Output;

            fn logical_divide_at(self, b: $B) -> Self::Output {
                let inner = self.logical_divide(b);
                lift::<Mode<I, Next>, _, _, _>(
                    &inner,
                    &MakeLayout,
                    Layout::new(Int::<1>, Int::<0>),
                )
            }
        }
    };
}

impl_nil_logical_divide_at_mode!([] int);
impl_nil_logical_divide_at_mode!([const N: int,] Int<N>);
impl_nil_logical_divide_at_mode!([SB: Shape, DB: Stride<SB>,] Layout<SB, DB>);
impl_nil_logical_divide_at_mode!([BH, BT,] Til<BH, BT>);

const impl<SA, DA, B> LogicalDivideAt<B, ModeEnd> for Layout<SA, DA>
where
    SA: Shape,
    DA: Stride<SA>,
    Self: [const] LogicalDivide<B>,
{
    type Output = <Self as LogicalDivide<B>>::Output;

    fn logical_divide_at(self, b: B) -> Self::Output {
        self.logical_divide(b)
    }
}

const impl<SA, DA, B, const I: int, Next, SubS, SubD, SubOutS, SubOutD>
    LogicalDivideAt<B, Mode<I, Next>> for Layout<SA, DA>
where
    SA: Shape + [const] Wrap,
    DA: Stride<SA> + [const] Wrap,
    SA::Wrapped: [const] Get<I, Output = SubS>,
    DA::Wrapped: [const] Get<I, Output = SubD>,
    SubS: Shape,
    SubD: Stride<SubS>,
    Layout<SubS, SubD>: [const] LogicalDivideAt<B, Next, Output = Layout<SubOutS, SubOutD>>,
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

    fn logical_divide_at(self, b: B) -> Self::Output {
        let s_wrap = self.shape.wrap();
        let d_wrap = self.stride.wrap();
        let sub = Layout::new(
            <SA::Wrapped as Get<I>>::get(&s_wrap),
            <DA::Wrapped as Get<I>>::get(&d_wrap),
        );
        let div_sub = <Layout<SubS, SubD> as LogicalDivideAt<B, Next>>::logical_divide_at(sub, b);
        Layout::new(
            <SA::Wrapped as ReplaceAt<SubOutS, I>>::replace_at(&s_wrap, div_sub.shape),
            <DA::Wrapped as ReplaceAt<SubOutD, I>>::replace_at(&d_wrap, div_sub.stride),
        )
    }
}

macro_rules! impl_tiler_logical_divide_at {
    ([$($g:tt)*] $T:ty) => {
        const impl<$($g)* B, Path> LogicalDivideAt<B, Path> for $T
        where
            Self: [const] Tiler,
            <Self as Tiler>::OutputLayout: [const] LogicalDivideAt<B, Path>,
        {
            type Output = <<Self as Tiler>::OutputLayout as LogicalDivideAt<B, Path>>::Output;

            fn logical_divide_at(self, b: B) -> Self::Output {
                tiler_to_layout(self).logical_divide_at(b)
            }
        }
    };
}

impl_tiler_logical_divide_at!([] int);
impl_tiler_logical_divide_at!([const N: int,] Int<N>);
impl_tiler_logical_divide_at!([H: Item, T: Tuple,] Til<H, T>);

const impl<A, B, Path> ZippedDivideAt<B, Path> for A
where
    B: [const] Tiler,
    A: [const] LogicalDivideAt<B::OutputLayout, Path>,
{
    type Output = <A as LogicalDivideAt<B::OutputLayout, Path>>::Output;

    fn zipped_divide_at(self, b: B) -> Self::Output {
        self.logical_divide_at(tiler_to_layout(b))
    }
}

impl<S, D> Layout<S, D> {
    /// Splits `self` by tiler `b`: $A \oslash B = A \circ (B, B^*)$.
    pub const fn logical_divide<B>(self, b: B) -> <Self as LogicalDivide<B>>::Output
    where
        Self: [const] LogicalDivide<B>,
    {
        LogicalDivide::logical_divide(self, b)
    }

    /// Splits only the sub-mode selected by `Path` (`mode![...]`) using tiler `b`.
    pub const fn logical_divide_at<Path, B>(
        self,
        b: B,
    ) -> <Self as LogicalDivideAt<B, Path>>::Output
    where
        Self: [const] LogicalDivideAt<B, Path>,
    {
        LogicalDivideAt::logical_divide_at(self, b)
    }

    /// Promotes `b` via [`tiler_to_layout`] and divides `self`, returning `(Tile, Grid)`.
    pub const fn zipped_divide<B>(self, b: B) -> <Self as ZippedDivide<B>>::Output
    where
        Self: [const] ZippedDivide<B>,
    {
        ZippedDivide::zipped_divide(self, b)
    }

    /// Promotes `b` via [`tiler_to_layout`] and divides only the sub-mode selected by `Path`.
    pub const fn zipped_divide_at<Path, B>(self, b: B) -> <Self as ZippedDivideAt<B, Path>>::Output
    where
        Self: [const] ZippedDivideAt<B, Path>,
    {
        ZippedDivideAt::zipped_divide_at(self, b)
    }
}

/// Computes the logical division `a / b` (`a o (b, complement(b, shape(a)))`).
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::algebra::logical_divide::logical_divide;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
/// use noodle_cute::int_tuple;
///
/// let res = logical_divide(Layout::new(Int::<24>, Int::<1>), Layout::new(Int::<4>, Int::<2>));
/// assert_eq!(
///     res,
///     Layout::new(
///         int_tuple![Int::<4>, (Int::<2>, Int::<3>)],
///         int_tuple![Int::<2>, (Int::<1>, Int::<8>)],
///     )
/// );
/// ```
pub const fn logical_divide<A, B>(a: A, b: B) -> <A as LogicalDivide<B>>::Output
where
    A: [const] LogicalDivide<B>,
{
    a.logical_divide(b)
}

/// Computes [`logical_divide`] at the sub-mode selected by `Path` (`mode![...]`).
pub const fn logical_divide_at<Path, A, B>(a: A, b: B) -> <A as LogicalDivideAt<B, Path>>::Output
where
    A: [const] LogicalDivideAt<B, Path>,
{
    a.logical_divide_at(b)
}

/// Computes `logical_divide(a, tiler_to_layout(b))`, grouping tile modes in mode 0 and grid modes in mode 1.
pub const fn zipped_divide<A, B>(a: A, b: B) -> <A as ZippedDivide<B>>::Output
where
    A: [const] ZippedDivide<B>,
{
    a.zipped_divide(b)
}

/// Computes [`zipped_divide`] at the sub-mode selected by `Path` (`mode![...]`).
pub const fn zipped_divide_at<Path, A, B>(a: A, b: B) -> <A as ZippedDivideAt<B, Path>>::Output
where
    A: [const] ZippedDivideAt<B, Path>,
{
    a.zipped_divide_at(b)
}
