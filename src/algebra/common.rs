//! Vectorization analysis, shape-targeted tiling, and register fragment layouts.
//!
//! This module provides higher-level CuTe layout algebra utilities built on
//! [`coalesce`](mod@super::coalesce), [`composition`](mod@super::composition),
//! [`inverse`](mod@super::inverse), and [`logical_product`](mod@super::logical_product):
//!
//! - [`MaxCommonVector`] / [`max_common_vector`] / [`Layout::max_common_vector`]:
//!   Maximal static leading contiguous vector width shared by two layouts
//!   (extracted from `coalesce(composition(a, right_inverse(b)))`).
//! - [`MaxCommonLayout`] / [`max_common_layout`] / [`Layout::max_common_layout`]:
//!   Common contiguous layout $R$ such that $A \circ R$ and $B \circ R$ are identities
//!   (or `1:0` when no static unit-strided common prefix exists).
//! - [`TileToShape`] / [`tile_to_shape`] / [`Layout::tile_to_shape`]:
//!   Pads a block layout with `1:0` modes to a target shape's rank and repeats it via
//!   [`blocked_product`](super::blocked_product) in column-major block order.
//! - [`FragmentLike`] / [`make_fragment_like`] / [`Layout::make_fragment_like`]:
//!   Constructs a compact register fragment layout matching `layout`'s shape: compact
//!   left-major (preserving static zero strides) on mode 0, and ordered like `layout`'s
//!   remaining strides on modes $1 \dots R - 1$, joined by [`tiled_product`](super::tiled_product).
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::{
//!     make_fragment_like, max_common_layout, max_common_vector, tile_to_shape,
//! };
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let col = Layout::from_shape(int_tuple![Int::<4>, Int::<8>]);
//! let row = Layout::from_shape_right(int_tuple![Int::<4>, Int::<8>]);
//! assert_eq!(max_common_vector(col, col), Int::<32>);
//! assert_eq!(max_common_vector(row, col), Int::<1>);
//! assert_eq!(max_common_layout(col, col), Layout::new(Int::<32>, Int::<1>));
//! assert_eq!(max_common_layout(row, col), Layout::new(Int::<1>, Int::<0>));
//!
//! let atom = Layout::from_shape_right(int_tuple![Int::<8>, Int::<8>]);
//! let tiled = tile_to_shape(atom, int_tuple![Int::<16>, Int::<32>]);
//! assert_eq!(
//!     tiled,
//!     Layout::new(
//!         int_tuple![(Int::<8>, Int::<2>), (Int::<8>, Int::<4>)],
//!         int_tuple![(Int::<8>, Int::<64>), (Int::<1>, Int::<128>)],
//!     )
//! );
//!
//! let part = Layout::new(
//!     int_tuple![Int::<2>, Int::<4>, Int::<2>],
//!     int_tuple![Int::<1>, Int::<64>, Int::<8>],
//! );
//! assert_eq!(
//!     make_fragment_like(&part),
//!     Layout::new(
//!         int_tuple![Int::<2>, Int::<4>, Int::<2>],
//!         int_tuple![Int::<1>, Int::<4>, Int::<2>],
//!     )
//! );
//! ```
//!
//! # Notes
//!
//! - [`tile_to_shape`] requires `rank(block) <= rank(target)` and exact per-mode
//!   divisibility (`target_mode_size % block_mode_size == 0`, with positive block
//!   mode sizes). Violations fail at compile time for static operands and panic with
//!   [`NoodleCuteError::NotDivisible`] or [`NoodleCuteError::InvalidShape`] for
//!   dynamic operands.
//! - [`max_common_vector`] works uniformly on plain integer layouts and `F2`-swizzled
//!   layouts via [`right_inverse`](super::right_inverse) + [`composition`](super::composition())
//!   + [`coalesce`](super::coalesce()).

use crate::NoodleCuteError;
use crate::basis::{BasisCombination, BasisTerm, BasisTuple};
use crate::f2::F2;
use crate::integer::{IS_NON_NEG, Int, Integer, StaticInteger, int};
use crate::layout::{
    Layout, MakeLayoutLeftFiltered, MakeOrderedLayout, make_ordered_layout,
};
use crate::shape::{Shape, ShapeScalar};
use crate::stride::{PrefixProduct, Stride};
use crate::tuple::{Item, Nil, PadModes, ProductEach, Til, Tuple, Wrap};

use super::coalesce::Coalesce;
use super::composition::Composition;
use super::inverse::RightInverse;
use super::logical_product::{BlockedProduct, TiledProduct};

// ------------------------------------------------------------------
// Shared leading-vector analysis helpers (also reused by algorithm::copy)
// ------------------------------------------------------------------

/// Classifies whether a stride leaf is statically a unit stride (`Int<1>`, or `F2<Int<1>>`
/// when `ALLOW_F2 == true`), excluding dynamic `int`, non-unit constants, and basis strides.
#[doc(hidden)]
pub const trait StrideIsStaticUnit<const ALLOW_F2: bool = false>: Copy {
    const IS_UNIT: bool = false;
}

const impl<I: Integer, const ALLOW_F2: bool> StrideIsStaticUnit<ALLOW_F2> for I {
    const IS_UNIT: bool = matches!(I::STATIC_VALUE, Some(1));
}

const impl<V: Integer, const ALLOW_F2: bool> StrideIsStaticUnit<ALLOW_F2> for F2<V> {
    const IS_UNIT: bool = ALLOW_F2 && matches!(V::STATIC_VALUE, Some(1));
}
const impl<V: Copy, P: Copy, const ALLOW_F2: bool> StrideIsStaticUnit<ALLOW_F2> for BasisTerm<V, P> {}
const impl<T: BasisTuple, const ALLOW_F2: bool> StrideIsStaticUnit<ALLOW_F2>
    for BasisCombination<T>
{
}

/// Compile-time flag: `true` iff extent `S` is static `>= 2` and stride `D` is static unit.
#[doc(hidden)]
pub const CAN_VECTORIZE<S: ShapeScalar, D: StrideIsStaticUnit<ALLOW_F2>, const ALLOW_F2: bool>:
    bool = D::IS_UNIT && matches!(S::STATIC_VALUE, Some(n) if n > 1);

/// Selects leading static extent `Int<N>` when `CAN_VEC == true`, or falls back to `Int<1>`.
#[doc(hidden)]
pub const trait SelectVecWidth<S: [const] ShapeScalar, const CAN_VEC: bool> {
    type Output: [const] ShapeScalar + [const] StaticInteger;
    fn select_vec_width(shape: S) -> Self::Output;
}

const impl<const N: int> SelectVecWidth<Int<N>, true> for ()
where
    Int<N>: [const] ShapeScalar,
{
    type Output = Int<N>;

    fn select_vec_width(shape: Int<N>) -> Int<N> {
        shape
    }
}

const impl<S: [const] ShapeScalar> SelectVecWidth<S, false> for () {
    type Output = Int<1>;

    fn select_vec_width(_: S) -> Int<1> {
        Int::<1>
    }
}

/// Extracts the maximal leading static unit-strided vector width `Int<V>` (`V >= 1`)
/// from a coalesced layout.
#[doc(hidden)]
pub const trait LeadingVecSize<const ALLOW_F2: bool = false>: Copy {
    type Output: [const] ShapeScalar + [const] StaticInteger;
    fn leading_vec_size(self) -> Self::Output;
}

const impl<const ALLOW_F2: bool> LeadingVecSize<ALLOW_F2> for Layout<Nil, Nil> {
    type Output = Int<1>;

    fn leading_vec_size(self) -> Int<1> {
        Int::<1>
    }
}

const impl<S, D, const ALLOW_F2: bool> LeadingVecSize<ALLOW_F2> for Layout<S, D>
where
    S: [const] ShapeScalar,
    D: StrideIsStaticUnit<ALLOW_F2>,
    (): [const] SelectVecWidth<S, { CAN_VECTORIZE::<S, D, ALLOW_F2> }>,
{
    type Output = <() as SelectVecWidth<S, { CAN_VECTORIZE::<S, D, ALLOW_F2> }>>::Output;

    fn leading_vec_size(self) -> Self::Output {
        <() as SelectVecWidth<S, { CAN_VECTORIZE::<S, D, ALLOW_F2> }>>::select_vec_width(self.shape)
    }
}

const impl<S0, ST, D0, DT, const ALLOW_F2: bool> LeadingVecSize<ALLOW_F2>
    for Layout<Til<S0, ST>, Til<D0, DT>>
where
    S0: [const] ShapeScalar,
    ST: Copy,
    D0: StrideIsStaticUnit<ALLOW_F2>,
    DT: Copy,
    (): [const] SelectVecWidth<S0, { CAN_VECTORIZE::<S0, D0, ALLOW_F2> }>,
{
    type Output = <() as SelectVecWidth<S0, { CAN_VECTORIZE::<S0, D0, ALLOW_F2> }>>::Output;

    fn leading_vec_size(self) -> Self::Output {
        <() as SelectVecWidth<S0, { CAN_VECTORIZE::<S0, D0, ALLOW_F2> }>>::select_vec_width(
            self.shape.0,
        )
    }
}

// ------------------------------------------------------------------
// MaxCommonVector / MaxCommonLayout
// ------------------------------------------------------------------

/// CuTe `max_common_vector(a, b)`: maximal static number of leading elements that are
/// contiguous in both layouts and correspond logically.
pub const trait MaxCommonVector<B>: Copy {
    /// Static vector length `Int<V>` (`V >= 1`).
    type Output: [const] ShapeScalar + [const] StaticInteger;
    /// Compile-time integer value of [`Self::Output`].
    const VALUE: int = Self::Output::VALUE;

    /// Returns the maximal common contiguous vector width.
    fn max_common_vector(self, b: B) -> Self::Output;
}

const impl<SA, DA, SB, DB> MaxCommonVector<Layout<SB, DB>> for Layout<SA, DA>
where
    SA: Shape,
    DA: Stride<SA>,
    SB: Shape,
    DB: Stride<SB>,
    Layout<SB, DB>: [const] RightInverse,
    Self: [const] Composition<<Layout<SB, DB> as RightInverse>::Output>,
    <Self as Composition<<Layout<SB, DB> as RightInverse>::Output>>::Output: [const] Coalesce,
    <<Self as Composition<<Layout<SB, DB> as RightInverse>::Output>>::Output as Coalesce>::Output:
        [const] LeadingVecSize<true>,
{
    type Output = <<<Self as Composition<
        <Layout<SB, DB> as RightInverse>::Output,
    >>::Output as Coalesce>::Output as LeadingVecSize<true>>::Output;

    fn max_common_vector(self, b: Layout<SB, DB>) -> Self::Output {
        let inv_b = b.right_inverse();
        let common = self.composition(inv_b).coalesce();
        LeadingVecSize::<true>::leading_vec_size(common)
    }
}

/// Selects `composition(inv_b, S:1)` when `CAN_VEC == true`, or falls back to `1:0`.
#[doc(hidden)]
pub const trait SelectCommonLayout<InvB: Copy, S: Shape, const CAN_VEC: bool> {
    type Output: Copy;
    fn select_common_layout(inv_b: InvB, extent: S) -> Self::Output;
}

const impl<InvB: Copy, S: Shape> SelectCommonLayout<InvB, S, false> for () {
    type Output = Layout<Int<1>, Int<0>>;

    fn select_common_layout(_: InvB, _: S) -> Layout<Int<1>, Int<0>> {
        Layout::new(Int::<1>, Int::<0>)
    }
}

const impl<InvB, S> SelectCommonLayout<InvB, S, true> for ()
where
    S: Shape,
    Int<1>: Stride<S>,
    InvB: [const] Composition<Layout<S, Int<1>>>,
{
    type Output = <InvB as Composition<Layout<S, Int<1>>>>::Output;

    fn select_common_layout(inv_b: InvB, extent: S) -> Self::Output {
        inv_b.composition(Layout::new(extent, Int::<1>))
    }
}

/// Extracts `composition(inv_b, extent_0(common):1)` when `common`'s leading mode is static and
/// unit-strided (`V > 1`), or returns `1:0`.
#[doc(hidden)]
pub const trait LeadingCommonLayout<InvB: Copy>: Copy {
    type Output: Copy;
    fn leading_common_layout(self, inv_b: InvB) -> Self::Output;
}

const impl<InvB: Copy> LeadingCommonLayout<InvB> for Layout<Nil, Nil> {
    type Output = Layout<Int<1>, Int<0>>;

    fn leading_common_layout(self, _: InvB) -> Layout<Int<1>, Int<0>> {
        Layout::new(Int::<1>, Int::<0>)
    }
}

const impl<InvB, S, D> LeadingCommonLayout<InvB> for Layout<S, D>
where
    InvB: Copy,
    S: [const] ShapeScalar,
    D: Stride<S> + StrideIsStaticUnit<true>,
    (): [const] SelectCommonLayout<InvB, S, { CAN_VECTORIZE::<S, D, true> }>,
{
    type Output = <() as SelectCommonLayout<InvB, S, { CAN_VECTORIZE::<S, D, true> }>>::Output;

    fn leading_common_layout(self, inv_b: InvB) -> Self::Output {
        <() as SelectCommonLayout<InvB, S, { CAN_VECTORIZE::<S, D, true> }>>::select_common_layout(
            inv_b, self.shape,
        )
    }
}

const impl<InvB, S0, ST, D0, DT> LeadingCommonLayout<InvB> for Layout<Til<S0, ST>, Til<D0, DT>>
where
    InvB: Copy,
    S0: [const] ShapeScalar,
    ST: Copy,
    D0: Stride<S0> + StrideIsStaticUnit<true>,
    DT: Copy,
    (): [const] SelectCommonLayout<InvB, S0, { CAN_VECTORIZE::<S0, D0, true> }>,
{
    type Output = <() as SelectCommonLayout<InvB, S0, { CAN_VECTORIZE::<S0, D0, true> }>>::Output;

    fn leading_common_layout(self, inv_b: InvB) -> Self::Output {
        <() as SelectCommonLayout<InvB, S0, { CAN_VECTORIZE::<S0, D0, true> }>>::select_common_layout(
            inv_b,
            self.shape.0,
        )
    }
}

/// CuTe `max_common_layout(a, b)`: returns the common contiguous layout `R` such that
/// `composition(a, R)` and `composition(b, R)` are identities (or `1:0` if none).
pub const trait MaxCommonLayout<B>: Copy {
    /// Resulting common contiguous [`Layout`].
    type Output: Copy;

    /// Returns the maximal common contiguous layout shared by `self` and `b`.
    fn max_common_layout(self, b: B) -> Self::Output;
}

const impl<SA, DA, SB, DB> MaxCommonLayout<Layout<SB, DB>> for Layout<SA, DA>
where
    SA: Shape,
    DA: Stride<SA>,
    SB: Shape,
    DB: Stride<SB>,
    Layout<SB, DB>: [const] RightInverse,
    Self: [const] Composition<<Layout<SB, DB> as RightInverse>::Output>,
    <Self as Composition<<Layout<SB, DB> as RightInverse>::Output>>::Output: [const] Coalesce,
    <<Self as Composition<<Layout<SB, DB> as RightInverse>::Output>>::Output as Coalesce>::Output:
        [const] LeadingCommonLayout<<Layout<SB, DB> as RightInverse>::Output>,
{
    type Output = <<<Self as Composition<
        <Layout<SB, DB> as RightInverse>::Output,
    >>::Output as Coalesce>::Output as LeadingCommonLayout<
        <Layout<SB, DB> as RightInverse>::Output,
    >>::Output;

    fn max_common_layout(self, b: Layout<SB, DB>) -> Self::Output {
        let inv_b = b.right_inverse();
        let common = self.composition(inv_b).coalesce();
        common.leading_common_layout(inv_b)
    }
}

// ------------------------------------------------------------------
// TileToShape
// ------------------------------------------------------------------

/// Compile-time guard verifying static exact divisibility (`OK == true`).
#[doc(hidden)]
pub const trait ValidExactDiv<const OK: bool> {}
const impl ValidExactDiv<true> for () {}

#[doc(hidden)]
pub const IS_EXACT_DIVISIBLE<const T: int, const B: int>: bool = B > 0 && T >= 0 && T % B == 0;

#[doc(hidden)]
pub const EXACT_QUOTIENT<const T: int, const B: int>: int = if B > 0 { T / B } else { 0 };

#[doc(hidden)]
pub const IS_POS_DIVISOR<const B: int>: bool = B > 0;

/// Exact non-negative integer division `self / B` for [`TileToShape`].
///
/// Static operands validate `B > 0`, `T >= 0`, and `T % B == 0` at compile time;
/// dynamic operands validate at runtime with [`NoodleCuteError::InvalidShape`] /
/// [`NoodleCuteError::NotDivisible`].
#[doc(hidden)]
pub const trait ExactDiv<B>: Copy {
    type Output: [const] ShapeScalar;
    fn exact_div(self, rhs: B) -> Self::Output;
}

const impl<const T: int, const B: int> ExactDiv<Int<B>> for Int<T>
where
    (): [const] ValidExactDiv<{ IS_EXACT_DIVISIBLE::<T, B> }>,
    Int<{ EXACT_QUOTIENT::<T, B> }>: [const] ShapeScalar,
{
    type Output = Int<{ EXACT_QUOTIENT::<T, B> }>;

    fn exact_div(self, _: Int<B>) -> Self::Output {
        Int
    }
}

const impl<const B: int> ExactDiv<Int<B>> for int
where
    (): [const] ValidExactDiv<{ IS_POS_DIVISOR::<B> }>,
{
    type Output = int;

    fn exact_div(self, _: Int<B>) -> int {
        NoodleCuteError::InvalidShape.check_with(
            self >= 0,
            "tile_to_shape target mode extent must be non-negative",
        );
        NoodleCuteError::NotDivisible.check_with(
            self % B == 0,
            "tile_to_shape target mode extent is not divisible by block mode size",
        );
        self / B
    }
}

const impl<const T: int> ExactDiv<int> for Int<T>
where
    (): [const] ValidExactDiv<{ IS_NON_NEG::<T> }>,
{
    type Output = int;

    fn exact_div(self, rhs: int) -> int {
        NoodleCuteError::InvalidShape.check_with(
            rhs > 0,
            "tile_to_shape block mode size must be positive",
        );
        NoodleCuteError::NotDivisible.check_with(
            T % rhs == 0,
            "tile_to_shape target mode extent is not divisible by block mode size",
        );
        T / rhs
    }
}

const impl ExactDiv<int> for int {
    type Output = int;

    fn exact_div(self, rhs: int) -> int {
        NoodleCuteError::InvalidShape.check_with(
            self >= 0 && rhs > 0,
            "tile_to_shape requires non-negative target and positive block extents",
        );
        NoodleCuteError::NotDivisible.check_with(
            self % rhs == 0,
            "tile_to_shape target mode extent is not divisible by block mode size",
        );
        self / rhs
    }
}

/// Element-wise [`ExactDiv`] across two equal-rank flat tuples of mode sizes.
///
/// Rejects rank mismatches (`rank(block) > rank(target)`) at compile time.
#[doc(hidden)]
pub const trait ExactDivEach<B: Tuple>: Tuple {
    type Output: Tuple + Shape;
    fn exact_div_each(self, rhs: B) -> Self::Output;
}

const impl ExactDivEach<Nil> for Nil {
    type Output = Nil;

    fn exact_div_each(self, _: Nil) -> Nil {
        Nil
    }
}

const impl<TH, TT, BH, BT> ExactDivEach<Til<BH, BT>> for Til<TH, TT>
where
    TH: [const] Item + [const] ExactDiv<BH>,
    TT: Tuple + [const] ExactDivEach<BT>,
    BH: [const] Item,
    BT: Tuple,
{
    type Output = Til<TH::Output, TT::Output>;

    fn exact_div_each(self, rhs: Til<BH, BT>) -> Self::Output {
        Til(self.0.exact_div(rhs.0), self.1.exact_div_each(rhs.1))
    }
}

type PadBlockShape<S, T> =
    <<S as Wrap>::Wrapped as PadModes<<T as Wrap>::Wrapped, Int<1>>>::Output;
type PadBlockStride<D, T> =
    <<D as Wrap>::Wrapped as PadModes<<T as Wrap>::Wrapped, Int<0>>>::Output;
type TileRepeats<S, T> = <<<T as Wrap>::Wrapped as ProductEach>::Output as ExactDivEach<
    <PadBlockShape<S, T> as ProductEach>::Output,
>>::Output;
type RepeatLayout<S, T> =
    Layout<TileRepeats<S, T>, <TileRepeats<S, T> as PrefixProduct<Int<1>>>::Stride>;

/// CuTe `tile_to_shape(block, target)`: pads `self` with `1:0` modes to `target`'s rank
/// and repeats it via [`blocked_product`](super::blocked_product) in column-major order.
///
/// Static non-divisible targets or blocks out-ranking `target` fail at compile time:
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::algebra::tile_to_shape;
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
///
/// let atom = Layout::from_shape(int_tuple![Int::<4>, Int::<4>]);
/// let _ = tile_to_shape(atom, int_tuple![Int::<10>, Int::<8>]);
/// ```
pub const trait TileToShape<T>: Copy {
    /// Resulting tiled [`Layout`].
    type Output: Copy;

    /// Repeats `self` to cover `target`.
    fn tile_to_shape(self, target: T) -> Self::Output;
}

const impl<S, D, T> TileToShape<T> for Layout<S, D>
where
    S: Shape + [const] Wrap,
    D: Stride<S> + [const] Wrap,
    T: Shape + [const] Wrap,
    S::Wrapped: [const] Tuple + Shape + [const] PadModes<T::Wrapped, Int<1>>,
    D::Wrapped: [const] Tuple + Stride<S::Wrapped> + [const] PadModes<T::Wrapped, Int<0>>,
    T::Wrapped: [const] Tuple + Shape + [const] ProductEach,
    PadBlockShape<S, T>: Tuple + Shape + [const] ProductEach,
    PadBlockStride<D, T>: Tuple + Stride<PadBlockShape<S, T>>,
    <T::Wrapped as ProductEach>::Output:
        Tuple + [const] ExactDivEach<<PadBlockShape<S, T> as ProductEach>::Output>,
    <PadBlockShape<S, T> as ProductEach>::Output: Tuple,
    TileRepeats<S, T>: [const] PrefixProduct<Int<1>>,
    <TileRepeats<S, T> as PrefixProduct<Int<1>>>::Stride: Stride<TileRepeats<S, T>>,
    Layout<PadBlockShape<S, T>, PadBlockStride<D, T>>: [const] BlockedProduct<RepeatLayout<S, T>>,
{
    type Output = <Layout<PadBlockShape<S, T>, PadBlockStride<D, T>> as BlockedProduct<
        RepeatLayout<S, T>,
    >>::Output;

    fn tile_to_shape(self, target: T) -> Self::Output {
        let t_wrap = target.wrap();
        let s_pad = self.shape.wrap().pad_modes(Int::<1>);
        let d_pad = self.stride.wrap().pad_modes(Int::<0>);
        let target_sizes = t_wrap.product_each();
        let block_sizes = s_pad.product_each();
        let repeats = target_sizes.exact_div_each(block_sizes);
        let repeat_layout = Layout::from_shape(repeats);
        Layout::new(s_pad, d_pad).blocked_product(repeat_layout)
    }
}

// ------------------------------------------------------------------
// FragmentLike / make_fragment_like
// ------------------------------------------------------------------

/// Compile-time flag: `true` iff `S` has more than one top-level mode (`RANK > 1`).
#[doc(hidden)]
pub const IS_MULTI_MODE<S: Item>: bool = S::RANK > 1;

/// Dispatches [`FragmentLike`] between single-mode (`MULTI == false`, compact left-major)
/// and multi-mode (`MULTI == true`, `tiled_product(left_filtered(mode 0), ordered(rest))`).
#[doc(hidden)]
pub const trait FragmentLikeStep<S, D, const MULTI: bool> {
    type Output;
    fn make_fragment(layout: &Layout<S, D>) -> Self::Output;
}

const impl<S, D> FragmentLikeStep<S, D, false> for ()
where
    S: Shape + [const] PrefixProduct<Int<1>>,
    D: Stride<S>,
    <S as PrefixProduct<Int<1>>>::Stride: Stride<S>,
{
    type Output = Layout<S, <S as PrefixProduct<Int<1>>>::Stride>;

    fn make_fragment(layout: &Layout<S, D>) -> Self::Output {
        Layout::from_shape(layout.shape)
    }
}

const impl<S0, SR, D0, DR> FragmentLikeStep<Til<S0, SR>, Til<D0, DR>, true> for ()
where
    S0: Shape,
    SR: Tuple + Shape,
    D0: Stride<S0>,
    DR: Tuple + Stride<SR>,
    Layout<S0, D0>: [const] MakeLayoutLeftFiltered,
    SR: [const] MakeOrderedLayout<DR>,
    <Layout<S0, D0> as MakeLayoutLeftFiltered>::Output:
        [const] TiledProduct<<SR as MakeOrderedLayout<DR>>::Output>,
{
    type Output = <<Layout<S0, D0> as MakeLayoutLeftFiltered>::Output as TiledProduct<
        <SR as MakeOrderedLayout<DR>>::Output,
    >>::Output;

    fn make_fragment(layout: &Layout<Til<S0, SR>, Til<D0, DR>>) -> Self::Output {
        let mode0 = Layout::new(layout.shape.0, layout.stride.0).make_layout_left_filtered();
        let rest = make_ordered_layout(&layout.shape.1, &layout.stride.1);
        mode0.tiled_product(rest)
    }
}

/// CuTe `make_fragment_like(layout)`: constructs a compact register fragment layout
/// with the same shape as `self`.
///
/// For multi-mode layouts (`RANK > 1`), mode 0 is compact left-major preserving static
/// stride-0 leaves, and modes `1..R` follow the relative stride order of `self`, combined
/// with [`tiled_product`](super::tiled_product). Single-mode layouts (`RANK <= 1`) use
/// compact left-major strides ([`Layout::from_shape`]).
pub const trait FragmentLike: Copy {
    /// Resulting compact register fragment [`Layout`].
    type Output;

    /// Constructs the fragment layout matching `self`.
    fn make_fragment_like(&self) -> Self::Output;
}

const impl<S, D> FragmentLike for Layout<S, D>
where
    S: Shape,
    D: Stride<S>,
    (): [const] FragmentLikeStep<S, D, { IS_MULTI_MODE::<S> }>,
{
    type Output = <() as FragmentLikeStep<S, D, { IS_MULTI_MODE::<S> }>>::Output;

    fn make_fragment_like(&self) -> Self::Output {
        <() as FragmentLikeStep<S, D, { IS_MULTI_MODE::<S> }>>::make_fragment(self)
    }
}

// ------------------------------------------------------------------
// Layout forwarding methods and top-level free functions
// ------------------------------------------------------------------

impl<S, D> Layout<S, D> {
    /// Maximal static leading contiguous vector width shared by `self` and `b`.
    pub const fn max_common_vector<B>(self, b: B) -> <Self as MaxCommonVector<B>>::Output
    where
        Self: [const] MaxCommonVector<B>,
    {
        MaxCommonVector::max_common_vector(self, b)
    }

    /// Maximal common contiguous layout shared by `self` and `b` (or `1:0`).
    pub const fn max_common_layout<B>(self, b: B) -> <Self as MaxCommonLayout<B>>::Output
    where
        Self: [const] MaxCommonLayout<B>,
    {
        MaxCommonLayout::max_common_layout(self, b)
    }

    /// Pads `self` with `1:0` modes to `target`'s rank and repeats it to fill `target`.
    pub const fn tile_to_shape<T>(self, target: T) -> <Self as TileToShape<T>>::Output
    where
        Self: [const] TileToShape<T>,
    {
        TileToShape::tile_to_shape(self, target)
    }

    /// Constructs a compact register fragment layout shaped like `self`.
    pub const fn make_fragment_like(&self) -> <Self as FragmentLike>::Output
    where
        Self: [const] FragmentLike,
    {
        FragmentLike::make_fragment_like(self)
    }
}

/// Returns the maximal static leading contiguous vector width shared by layouts `a` and `b`.
pub const fn max_common_vector<A, B>(a: A, b: B) -> <A as MaxCommonVector<B>>::Output
where
    A: [const] MaxCommonVector<B>,
{
    a.max_common_vector(b)
}

/// Returns the maximal common contiguous layout `R` shared by layouts `a` and `b` (or `1:0`).
pub const fn max_common_layout<A, B>(a: A, b: B) -> <A as MaxCommonLayout<B>>::Output
where
    A: [const] MaxCommonLayout<B>,
{
    a.max_common_layout(b)
}

/// Pads `block` with `1:0` modes to `target`'s rank and repeats it via [`blocked_product`](super::blocked_product).
pub const fn tile_to_shape<A, T>(block: A, target: T) -> <A as TileToShape<T>>::Output
where
    A: [const] TileToShape<T>,
{
    block.tile_to_shape(target)
}

/// Constructs a compact register fragment [`Layout`] shaped like `layout`.
pub const fn make_fragment_like<L: [const] FragmentLike>(layout: &L) -> L::Output {
    layout.make_fragment_like()
}
