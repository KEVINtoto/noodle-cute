//! Compile-time layout coalescing: mode merging and hierarchy simplification.
//!
//! Coalescing simplifies a [`Layout`] into an algebraically equivalent layout with
//! minimal rank and depth, preserving its 1-D coordinate evaluation:
//!
//! $$R(i) == A(i) \quad \text{for all } i \in [0, |A|)$$
//!
//! # Main Interfaces
//!
//! - [`Coalesce`] / [`coalesce()`] / [`Layout::coalesce`]: Flattens and folds adjacent
//!   contiguous modes, dropping trailing size-1 modes (`EXTENDED = false`).
//! - [`coalesce_extended`] / [`Layout::coalesce_extended`]: Extended-domain coalescing
//!   (`EXTENDED = true`, matching CuTe's `coalesce_z`), preserving the rightmost unit extent
//!   so $R(i) == A(i)$ holds for every integer $i \in \mathbb{Z}$.
//! - [`CoalesceExtended`]: Profile-guided extended-domain coalescing preserving outer mode
//!   boundaries dictated by a structural profile (`LeafProfile`, `NilProfile`, `TupleProfile`).
//!
//! # Core Algebraic Rules
//!
//! Coalescing folds modes from right to left using static facts encoded in their types:
//!
//! 1. **Contiguous Mode Merging**:
//!    Two adjacent modes $s_a : d_a$ and $s_b : d_b$ merge into $(s_a \cdot s_b) : d_a$
//!    when $d_a \cdot s_a = d_b$.
//! 2. **Size-1 Mode Elimination**:
//!    Modes with static extent 1 contribute nothing to index calculation ($0 \cdot d = 0$).
//!    Intermediate unit extents are discarded, allowing their surrounding neighbors to merge:
//!    $$\text{Layout}((2, 1, 6), (1, 7, 2)) \longrightarrow \text{Layout}(12, 1)$$
//! 3. **Trailing Size-1 Extents (`EXTENDED`)**:
//!    - Standard [`Coalesce<false>`] drops trailing size-1 modes.
//!    - Extended [`Coalesce<true>`] preserves the rightmost unit extent.
//! 4. **Coordinate Basis Orthogonality**:
//!    [`ScaledBasis<V, P>`](type@crate::basis::ScaledBasis) and [`BasisCombination`] strides
//!    merge only when their paths match term-for-term.
//! 5. **Dynamic Fallback**:
//!    Runtime dynamic extents or strides are safely retained when static proof of linearity
//!    is unavailable; static strides can still merge across a dynamic suffix extent.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::coalesce::{Coalesce, coalesce, coalesce_extended};
//! use noodle_cute::basis::{Mode, ScaledBasis};
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::{int_tuple, tuple};
//!
//! // 1. Merging contiguous modes: (2, 4):(1, 2) -> 8:1
//! let l = Layout::new(
//!     int_tuple![Int::<2>, Int::<4>],
//!     int_tuple![Int::<1>, Int::<2>],
//! );
//! assert_eq!(coalesce(l), Layout::new(Int::<8>, Int::<1>));
//!
//! // 2. Eliminating intermediate size-1 mode: (2, 1, 6):(1, 7, 2) -> 12:1
//! let l_unit = Layout::new(
//!     int_tuple![Int::<2>, Int::<1>, Int::<6>],
//!     int_tuple![Int::<1>, Int::<7>, Int::<2>],
//! );
//! assert_eq!(l_unit.coalesce(), Layout::new(Int::<12>, Int::<1>));
//!
//! // 3. Trailing size-1 dropped in coalesce vs preserved in coalesce_extended:
//! let l_trail = Layout::new(
//!     int_tuple![Int::<2>, Int::<1>],
//!     int_tuple![Int::<1>, Int::<0>],
//! );
//! assert_eq!(l_trail.coalesce(), Layout::new(Int::<2>, Int::<1>));
//! assert_eq!(coalesce_extended(l_trail), l_trail);
//!
//! // 4. ScaledBasis mode merging along the same axis:
//! let l_basis = Layout::new(
//!     int_tuple![Int::<2>, Int::<4>],
//!     tuple![
//!         ScaledBasis::<Int<1>, Mode<0>>::new(Int::<1>),
//!         ScaledBasis::<Int<2>, Mode<0>>::new(Int::<2>),
//!     ],
//! );
//! assert_eq!(
//!     l_basis.coalesce(),
//!     Layout::new(Int::<8>, ScaledBasis::<Int<1>, Mode<0>>::new(Int::<1>))
//! );
//! ```
//!
//! # Notes
//!
//! - Merged static extents that overflow [`int`] fail at compile time (`E0080`);
//!   dynamic extent merges or negative dynamic extents panic at runtime.

use core::marker::ConstParamTy;

use crate::basis::{
    BasisCoeff, BasisCombination, BasisPath, BasisTerm, BasisTuple, CoeffAlgebra, ComparePath,
    ModeEnd,
};
use crate::f2::clmul_maybe_overflows;
use crate::integer::{ArithMul, Int, Product, int, integer_product_eq, is_pow2, mul};
use crate::layout::Layout;
use crate::shape::{Shape, ShapeScalar};
use crate::stride::{Stride, same_path};
use crate::tuple::{Flatten, Leaf, LeafProfile, Nil, NilProfile, Til, Tuple, TupleProfile};

/// Action to apply when prepending an outer mode `s:d` onto an already-coalesced suffix.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum PrependAction {
    /// Retain the prepended mode as a distinct head mode in the layout tuple.
    Retain,
    /// Discard the prepended mode (used when its extent is statically 1).
    Discard,
    /// Merge the prepended mode with the leading mode of the suffix, multiplying extents.
    Merge,
}

const fn static_coeff_product_eq<S: ShapeScalar, L: BasisCoeff, R: BasisCoeff>() -> bool {
    let (Some(s), Some(l), Some(r)) = (S::STATIC_VALUE, L::STATIC_VALUE, R::STATIC_VALUE) else {
        return false;
    };
    match (L::ALGEBRA, R::ALGEBRA) {
        (CoeffAlgebra::Integer, CoeffAlgebra::Integer) => integer_product_eq(l, s, r),
        // Over F2, linearity at (0, 1) and (s - 1, 1) holds iff l == r == 0, or s is a power
        // of two and clmul(l, s) == r (avoiding 2 * s - 1 overflow).
        (CoeffAlgebra::F2, CoeffAlgebra::F2) => {
            (l == 0 && r == 0)
                || (is_pow2(s)
                    && matches!(clmul_maybe_overflows(l, s), Some(product) if product == r))
        }
        (CoeffAlgebra::Integer, CoeffAlgebra::F2) | (CoeffAlgebra::F2, CoeffAlgebra::Integer) => {
            l == 0 && r == 0
        }
        _ => false,
    }
}

/// Term-by-term static check for whether `Self * S == Rhs` across path-sorted [`BasisTuple`]s.
#[doc(hidden)]
pub const trait BasisTupleProductEq<S: ShapeScalar, Rhs: BasisTuple>: BasisTuple {
    const PRODUCT_EQ: bool;
}

const impl<S: ShapeScalar> BasisTupleProductEq<S, Nil> for Nil {
    const PRODUCT_EQ: bool = true;
}

const impl<S: ShapeScalar, H, T> BasisTupleProductEq<S, Til<H, T>> for Nil
where
    Til<H, T>: BasisTuple,
{
    const PRODUCT_EQ: bool = false;
}

const impl<S: ShapeScalar, H, T> BasisTupleProductEq<S, Nil> for Til<H, T>
where
    Til<H, T>: BasisTuple,
{
    const PRODUCT_EQ: bool = matches!(S::STATIC_VALUE, Some(0));
}

const impl<S, VL, PL, TL, VR, PR, TR> BasisTupleProductEq<S, Til<BasisTerm<VR, PR>, TR>>
    for Til<BasisTerm<VL, PL>, TL>
where
    S: ShapeScalar,
    VL: BasisCoeff,
    PL: BasisPath + ComparePath<PR>,
    TL: BasisTuple + BasisTupleProductEq<S, TR>,
    VR: BasisCoeff,
    PR: BasisPath,
    TR: BasisTuple,
    BasisTerm<VL, PL>: Leaf,
    BasisTerm<VR, PR>: Leaf,
{
    const PRODUCT_EQ: bool = same_path::<PL, PR>()
        && static_coeff_product_eq::<S, VL, VR>()
        && <TL as BasisTupleProductEq<S, TR>>::PRODUCT_EQ;
}

/// Compile-time check for whether stride leaf `Self` scaled by `S` equals `NextD`.
#[doc(hidden)]
pub const trait CoalesceStrideProductEq<S: ShapeScalar, NextD: Leaf>: Leaf {
    const PRODUCT_EQ: bool;
}

const impl<S: ShapeScalar, L: BasisCoeff + Leaf, R: BasisCoeff + Leaf> CoalesceStrideProductEq<S, R>
    for L
{
    const PRODUCT_EQ: bool = static_coeff_product_eq::<S, L, R>();
}

const impl<S: ShapeScalar, L: BasisCoeff + Leaf, VR: BasisCoeff, PR: BasisPath>
    CoalesceStrideProductEq<S, BasisTerm<VR, PR>> for L
where
    ModeEnd: ComparePath<PR>,
{
    const PRODUCT_EQ: bool = same_path::<ModeEnd, PR>() && static_coeff_product_eq::<S, L, VR>();
}

const impl<S: ShapeScalar, L: BasisCoeff + Leaf> CoalesceStrideProductEq<S, BasisCombination<Nil>>
    for L
{
    const PRODUCT_EQ: bool = static_coeff_product_eq::<S, L, Int<0>>();
}

const impl<S: ShapeScalar, L: BasisCoeff + Leaf, HR, TR>
    CoalesceStrideProductEq<S, BasisCombination<Til<HR, TR>>> for L
where
    Til<HR, TR>: BasisTuple,
{
    const PRODUCT_EQ: bool = false;
}

const impl<S: ShapeScalar, VL: BasisCoeff, PL: BasisPath, R: BasisCoeff + Leaf>
    CoalesceStrideProductEq<S, R> for BasisTerm<VL, PL>
where
    PL: ComparePath<ModeEnd>,
{
    const PRODUCT_EQ: bool = same_path::<PL, ModeEnd>() && static_coeff_product_eq::<S, VL, R>();
}

const impl<S: ShapeScalar, VL: BasisCoeff, PL: BasisPath, VR: BasisCoeff, PR: BasisPath>
    CoalesceStrideProductEq<S, BasisTerm<VR, PR>> for BasisTerm<VL, PL>
where
    PL: ComparePath<PR>,
{
    const PRODUCT_EQ: bool = same_path::<PL, PR>() && static_coeff_product_eq::<S, VL, VR>();
}

const impl<S: ShapeScalar, VL: BasisCoeff, PL: BasisPath>
    CoalesceStrideProductEq<S, BasisCombination<Nil>> for BasisTerm<VL, PL>
where
    PL: ComparePath<ModeEnd>,
{
    const PRODUCT_EQ: bool =
        same_path::<PL, ModeEnd>() && static_coeff_product_eq::<S, VL, Int<0>>();
}

const impl<S: ShapeScalar, VL: BasisCoeff, PL: BasisPath, HR, TR>
    CoalesceStrideProductEq<S, BasisCombination<Til<HR, TR>>> for BasisTerm<VL, PL>
where
    Til<HR, TR>: BasisTuple,
{
    const PRODUCT_EQ: bool = false;
}

const impl<S: ShapeScalar, R: BasisCoeff + Leaf> CoalesceStrideProductEq<S, R>
    for BasisCombination<Nil>
{
    const PRODUCT_EQ: bool = static_coeff_product_eq::<S, Int<0>, R>();
}

const impl<S: ShapeScalar, HL, TL, R: BasisCoeff + Leaf> CoalesceStrideProductEq<S, R>
    for BasisCombination<Til<HL, TL>>
where
    Til<HL, TL>: BasisTuple,
{
    const PRODUCT_EQ: bool = matches!(S::STATIC_VALUE, Some(0)) && R::IS_STATIC_ZERO;
}

const impl<S: ShapeScalar, VR: BasisCoeff, PR: BasisPath>
    CoalesceStrideProductEq<S, BasisTerm<VR, PR>> for BasisCombination<Nil>
where
    ModeEnd: ComparePath<PR>,
{
    const PRODUCT_EQ: bool =
        same_path::<ModeEnd, PR>() && static_coeff_product_eq::<S, Int<0>, VR>();
}

const impl<S: ShapeScalar, HL, TL, VR: BasisCoeff, PR: BasisPath>
    CoalesceStrideProductEq<S, BasisTerm<VR, PR>> for BasisCombination<Til<HL, TL>>
where
    Til<HL, TL>: BasisTuple,
{
    const PRODUCT_EQ: bool = matches!(S::STATIC_VALUE, Some(0)) && VR::IS_STATIC_ZERO;
}

const impl<S: ShapeScalar, TL: BasisTuple, TR: BasisTuple>
    CoalesceStrideProductEq<S, BasisCombination<TR>> for BasisCombination<TL>
where
    TL: BasisTupleProductEq<S, TR>,
{
    const PRODUCT_EQ: bool = <TL as BasisTupleProductEq<S, TR>>::PRODUCT_EQ;
}

const fn prepend_action<S, D, NextD>() -> PrependAction
where
    S: ShapeScalar,
    D: CoalesceStrideProductEq<S, NextD>,
    NextD: Leaf,
{
    let Some(s) = S::STATIC_VALUE else {
        return PrependAction::Retain;
    };
    if s == 1 {
        return PrependAction::Discard;
    }
    if D::PRODUCT_EQ {
        PrependAction::Merge
    } else {
        PrependAction::Retain
    }
}

const PREPEND_ACTION<S: ShapeScalar, D: CoalesceStrideProductEq<S, NextD>, NextD: Leaf>:
    PrependAction = prepend_action::<S, D, NextD>();

const NIL_DECISION<S: ShapeScalar, const EXTENDED: bool>: PrependAction =
    if matches!(S::STATIC_VALUE, Some(1)) && !EXTENDED {
        PrependAction::Discard
    } else {
        PrependAction::Retain
    };

/// Applies one [`PrependAction`] to an already-coalesced suffix layout, folding mode `S:D`
/// and exposing the leading stride [`FirstStride`](CoalescePrepend::FirstStride).
#[doc(hidden)]
pub const trait CoalescePrepend<S, D, const ACTION: PrependAction>: Copy {
    /// The resulting layout type after prepending mode `S:D`.
    type Output: Copy;
    /// The stride of the leading mode in [`Self::Output`].
    type FirstStride: Leaf;
    /// Prepends mode `s:d` onto `self`.
    fn prepend(self, s: S, d: D) -> Self::Output;
}

const impl<S: Copy, D: Copy> CoalescePrepend<S, D, { PrependAction::Discard }>
    for Layout<Nil, Nil>
{
    type Output = Self;
    type FirstStride = Int<0>;
    fn prepend(self, _: S, _: D) -> Self {
        self
    }
}

const impl<S, D> CoalescePrepend<S, D, { PrependAction::Retain }> for Layout<Nil, Nil>
where
    S: [const] ShapeScalar,
    D: Leaf + Stride<S>,
{
    type Output = Layout<S, D>;
    type FirstStride = D;
    fn prepend(self, s: S, d: D) -> Self::Output {
        let _ = s.size();
        Layout::new(s, d)
    }
}

// When prepending to a suffix discarded to Nil, a zero-stride mode can match
// FirstStride (Int<0>); treat Merge identically to Retain.
const impl<S, D> CoalescePrepend<S, D, { PrependAction::Merge }> for Layout<Nil, Nil>
where
    S: [const] ShapeScalar,
    D: Leaf + Stride<S>,
{
    type Output = Layout<S, D>;
    type FirstStride = D;
    fn prepend(self, s: S, d: D) -> Self::Output {
        let _ = s.size();
        Layout::new(s, d)
    }
}

const impl<AccS, AccD, S: Copy, D: Copy> CoalescePrepend<S, D, { PrependAction::Discard }>
    for Layout<AccS, AccD>
where
    AccS: ShapeScalar,
    AccD: Leaf + Stride<AccS>,
{
    type Output = Self;
    type FirstStride = AccD;
    fn prepend(self, _: S, _: D) -> Self {
        self
    }
}

const impl<AccS, AccD, S, D> CoalescePrepend<S, D, { PrependAction::Merge }> for Layout<AccS, AccD>
where
    AccD: Leaf + Stride<AccS>,
    S: [const] ShapeScalar + [const] ArithMul<S, AccS>,
    AccS: [const] ShapeScalar,
    Product<S, AccS>: ShapeScalar,
    D: Leaf + Stride<Product<S, AccS>>,
{
    type Output = Layout<Product<S, AccS>, D>;
    type FirstStride = D;
    fn prepend(self, s: S, d: D) -> Self::Output {
        let _ = s.size();
        let _ = self.shape.size();
        Layout::new(mul(s, self.shape), d)
    }
}

const impl<AccS, AccD, S, D> CoalescePrepend<S, D, { PrependAction::Retain }> for Layout<AccS, AccD>
where
    AccS: ShapeScalar,
    AccD: Leaf + Stride<AccS>,
    S: [const] ShapeScalar,
    D: Leaf + Stride<S>,
{
    type Output = Layout<Til<S, Til<AccS, Nil>>, Til<D, Til<AccD, Nil>>>;
    type FirstStride = D;
    fn prepend(self, s: S, d: D) -> Self::Output {
        let _ = s.size();
        Layout::new(Til(s, Til(self.shape, Nil)), Til(d, Til(self.stride, Nil)))
    }
}

const impl<HS, TS, HD, TD, S: Copy, D: Copy> CoalescePrepend<S, D, { PrependAction::Discard }>
    for Layout<Til<HS, TS>, Til<HD, TD>>
where
    HS: ShapeScalar,
    HD: Leaf + Stride<HS>,
    TS: Tuple + Shape,
    TD: Tuple + Stride<TS>,
{
    type Output = Self;
    type FirstStride = HD;
    fn prepend(self, _: S, _: D) -> Self {
        self
    }
}

const impl<HS, TS, HD, TD, S, D> CoalescePrepend<S, D, { PrependAction::Merge }>
    for Layout<Til<HS, TS>, Til<HD, TD>>
where
    HS: [const] ShapeScalar,
    HD: Leaf + Stride<HS>,
    TS: Tuple + Shape,
    TD: Tuple + Stride<TS>,
    S: [const] ShapeScalar + [const] ArithMul<S, HS>,
    Product<S, HS>: ShapeScalar,
    D: Leaf + Stride<Product<S, HS>>,
{
    type Output = Layout<Til<Product<S, HS>, TS>, Til<D, TD>>;
    type FirstStride = D;
    fn prepend(self, s: S, d: D) -> Self::Output {
        let _ = s.size();
        let _ = self.shape.0.size();
        Layout::new(
            Til(mul(s, self.shape.0), self.shape.1),
            Til(d, self.stride.1),
        )
    }
}

const impl<HS, TS, HD, TD, S, D> CoalescePrepend<S, D, { PrependAction::Retain }>
    for Layout<Til<HS, TS>, Til<HD, TD>>
where
    HS: ShapeScalar,
    TS: Tuple,
    HD: Leaf,
    TD: Tuple,
    S: [const] ShapeScalar,
    D: Leaf,
    Til<S, Til<HS, TS>>: Shape,
    Til<D, Til<HD, TD>>: Stride<Til<S, Til<HS, TS>>>,
{
    type Output = Layout<Til<S, Til<HS, TS>>, Til<D, Til<HD, TD>>>;
    type FirstStride = D;
    fn prepend(self, s: S, d: D) -> Self::Output {
        let _ = s.size();
        Layout::new(Til(s, self.shape), Til(d, self.stride))
    }
}

/// Coalesces a flat layout from right to left, selecting output types statically.
///
/// [`Layout::coalesce`] and [`Layout::coalesce_extended`] flatten the input
/// hierarchy before dispatching to this trait.
///
/// # Const Generic Parameter
///
/// - `EXTENDED = false` (default): Drops trailing unit extents, matching standard
///   CuTe `coalesce`.
/// - `EXTENDED = true`: Preserves the rightmost unit extent, matching CuTe
///   `coalesce_z` (extended domain).
///
/// # Examples
///
/// Standard compile-time evaluation:
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::algebra::coalesce::Coalesce;
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
/// use noodle_cute::tuple::{Nil, Til};
///
/// const L: Layout<
///     Til<Int<2>, Til<Int<4>, Nil>>,
///     Til<Int<1>, Til<Int<2>, Nil>>,
/// > = Layout::new(
///     int_tuple![Int::<2>, Int::<4>],
///     int_tuple![Int::<1>, Int::<2>],
/// );
///
/// const RESULT: Layout<Int<8>, Int<1>> = L.coalesce();
/// assert_eq!(RESULT, Layout::new(Int::<8>, Int::<1>));
/// ```
///
/// Extended domain preserving trailing unit mode:
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
///
/// let l = Layout::new(
///     int_tuple![Int::<2>, Int::<1>],
///     int_tuple![Int::<1>, Int::<0>],
/// );
///
/// assert_eq!(l.coalesce(), Layout::new(Int::<2>, Int::<1>));
/// assert_eq!(l.coalesce_extended(), l);
/// ```
///
/// Merged static extents must remain representable as an [`int`]:
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
///
/// let _ = Layout::new(
///     int_tuple![Int::<2>, Int::<2_147_483_647>],
///     int_tuple![Int::<1>, Int::<2>],
/// )
/// .coalesce();
/// ```
pub const trait Coalesce<const EXTENDED: bool = false>: Copy {
    /// The coalesced layout type.
    type Output: Copy;
    /// The stride of the leading mode in [`Self::Output`].
    type FirstStride: Leaf;
    /// Coalesces `self`.
    fn coalesce(self) -> Self::Output;
}

type Coalesced<S, D, const EXTENDED: bool> = <Layout<S, D> as Coalesce<EXTENDED>>::Output;

type CoalescedFirstStride<S, D, const EXTENDED: bool> =
    <Layout<S, D> as Coalesce<EXTENDED>>::FirstStride;

type NilPrepended<S, D, const EXTENDED: bool> =
    <Layout<Nil, Nil> as CoalescePrepend<S, D, { NIL_DECISION::<S, EXTENDED> }>>::Output;

type NilPrependedFirstStride<S, D, const EXTENDED: bool> =
    <Layout<Nil, Nil> as CoalescePrepend<S, D, { NIL_DECISION::<S, EXTENDED> }>>::FirstStride;

type LayoutPrepended<S, D, TS, TD, const EXTENDED: bool> =
    <Coalesced<TS, TD, EXTENDED> as CoalescePrepend<
        S,
        D,
        { PREPEND_ACTION::<S, D, CoalescedFirstStride<TS, TD, EXTENDED>> },
    >>::Output;

type LayoutPrependedFirstStride<S, D, TS, TD, const EXTENDED: bool> =
    <Coalesced<TS, TD, EXTENDED> as CoalescePrepend<
        S,
        D,
        { PREPEND_ACTION::<S, D, CoalescedFirstStride<TS, TD, EXTENDED>> },
    >>::FirstStride;

const impl<const EXTENDED: bool> Coalesce<EXTENDED> for Layout<Nil, Nil> {
    type Output = Self;
    type FirstStride = Int<0>;
    fn coalesce(self) -> Self::Output {
        self
    }
}

const impl<S, D, const EXTENDED: bool> Coalesce<EXTENDED> for Layout<S, D>
where
    S: [const] ShapeScalar,
    D: Leaf + Stride<S>,
    Layout<Nil, Nil>: [const] CoalescePrepend<S, D, { NIL_DECISION::<S, EXTENDED> }>,
{
    type Output = NilPrepended<S, D, EXTENDED>;
    type FirstStride = NilPrependedFirstStride<S, D, EXTENDED>;
    fn coalesce(self) -> Self::Output {
        Layout::new(Nil, Nil).prepend(self.shape, self.stride)
    }
}

const impl<S, D, const EXTENDED: bool> Coalesce<EXTENDED> for Layout<Til<S, Nil>, Til<D, Nil>>
where
    S: [const] ShapeScalar,
    D: Leaf + Stride<S>,
    Layout<Nil, Nil>: [const] CoalescePrepend<S, D, { NIL_DECISION::<S, EXTENDED> }>,
{
    type Output = NilPrepended<S, D, EXTENDED>;
    type FirstStride = NilPrependedFirstStride<S, D, EXTENDED>;
    fn coalesce(self) -> Self::Output {
        Layout::new(Nil, Nil).prepend(self.shape.0, self.stride.0)
    }
}

const impl<S0, D0, S1, D1, TS, TD, const EXTENDED: bool> Coalesce<EXTENDED>
    for Layout<Til<S0, Til<S1, TS>>, Til<D0, Til<D1, TD>>>
where
    S0: [const] ShapeScalar,
    D0: Leaf + Stride<S0>,
    S1: [const] ShapeScalar,
    D1: Leaf + Stride<S1>,
    TS: Tuple + Shape,
    TD: Tuple + Stride<TS>,
    Layout<Til<S1, TS>, Til<D1, TD>>: [const] Coalesce<EXTENDED>,
    D0: CoalesceStrideProductEq<S0, CoalescedFirstStride<Til<S1, TS>, Til<D1, TD>, EXTENDED>>,
    Coalesced<Til<S1, TS>, Til<D1, TD>, EXTENDED>: [const] CoalescePrepend<
            S0,
            D0,
            { PREPEND_ACTION::<S0, D0, CoalescedFirstStride<Til<S1, TS>, Til<D1, TD>, EXTENDED>> },
        >,
{
    type Output = LayoutPrepended<S0, D0, Til<S1, TS>, Til<D1, TD>, EXTENDED>;
    type FirstStride = LayoutPrependedFirstStride<S0, D0, Til<S1, TS>, Til<D1, TD>, EXTENDED>;
    fn coalesce(self) -> Self::Output {
        Coalesce::<EXTENDED>::coalesce(Layout::new(self.shape.1, self.stride.1))
            .prepend(self.shape.0, self.stride.0)
    }
}

impl<S, D> Layout<S, D> {
    /// Flattens the layout and coalesces adjacent modes over its coordinate domain.
    pub const fn coalesce(self) -> Coalesced<S::Output, D::Output, false>
    where
        S: [const] Flatten,
        D: [const] Flatten,
        Layout<S::Output, D::Output>: [const] Coalesce,
    {
        Coalesce::<false>::coalesce(Layout {
            shape: self.shape.flatten(),
            stride: self.stride.flatten(),
        })
    }

    /// Flattens and coalesces while preserving the trailing unit mode.
    pub const fn coalesce_extended(self) -> Coalesced<S::Output, D::Output, true>
    where
        S: [const] Flatten,
        D: [const] Flatten,
        Layout<S::Output, D::Output>: [const] Coalesce<true>,
    {
        Coalesce::<true>::coalesce(Layout {
            shape: self.shape.flatten(),
            stride: self.stride.flatten(),
        })
    }
}

/// High-level coalescing interface implemented for [`Layout`] (flattening before folding)
/// and [`Tensor`](crate::tensor::Tensor) (coalescing the tensor's layout while keeping its accessor).
pub const trait CoalesceTarget<const EXTENDED: bool = false>: Copy {
    /// Resulting coalesced layout or tensor type.
    type Output: Copy;
    /// Flattens and coalesces `self`.
    fn coalesce_target(self) -> Self::Output;
}

const impl<S, D, const EXTENDED: bool> CoalesceTarget<EXTENDED> for Layout<S, D>
where
    S: [const] Flatten,
    D: [const] Flatten,
    Layout<S::Output, D::Output>: [const] Coalesce<EXTENDED>,
{
    type Output = Coalesced<S::Output, D::Output, EXTENDED>;
    fn coalesce_target(self) -> Self::Output {
        Coalesce::<EXTENDED>::coalesce(Layout {
            shape: self.shape.flatten(),
            stride: self.stride.flatten(),
        })
    }
}

/// Flattens and coalesces `value` (a [`Layout`] or [`Tensor`](crate::tensor::Tensor))
/// over its coordinate domain (dropping trailing unit modes).
///
/// Note: accepts a single type parameter `T` so both [`Layout`] and
/// [`Tensor`](crate::tensor::Tensor) share one free function (matching [`composition`](super::composition())
/// and [`logical_divide`](super::logical_divide())); callers previously writing
/// `coalesce::<S, D>(layout)` can write `coalesce(layout)` or `coalesce::<Layout<S, D>>(layout)`.
pub const fn coalesce<T>(value: T) -> T::Output
where
    T: [const] CoalesceTarget<false>,
{
    value.coalesce_target()
}

/// Flattens and coalesces `value` (a [`Layout`] or [`Tensor`](crate::tensor::Tensor))
/// over its extended domain (preserving the rightmost unit mode).
///
/// See [`coalesce()`] for the single type-parameter convention.
pub const fn coalesce_extended<T>(value: T) -> T::Output
where
    T: [const] CoalesceTarget<true>,
{
    value.coalesce_target()
}

/// Profile-guided coalescing (`coalesce(obj, profile)`) over the coordinate domain
/// (dropping trailing unit modes within each `LeafProfile` subtree).
pub const fn coalesce_with_profile<P, T>(value: T) -> T::Output
where
    T: [const] CoalesceProfile<P>,
{
    value.coalesce_with_profile()
}

/// Profile-guided extended-domain coalescing (`coalesce_z`) for a [`Layout`] or
/// [`Tensor`](crate::tensor::Tensor).
pub const fn coalesce_z<P, T>(value: T) -> T::Output
where
    T: [const] CoalesceExtended<P>,
{
    value.coalesce_z()
}

const SHAPE_IS_STATIC_ONE<S: ShapeScalar>: bool = matches!(S::STATIC_VALUE, Some(1));

/// Drops the second mode of a 2-mode flat layout when `DROP == true`, unwrapping to scalar `S0:D0`.
#[doc(hidden)]
pub const trait DropLastPair<const DROP: bool>: Copy {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn drop_last_pair(self) -> Layout<Self::Shape, Self::Stride>;
}

const impl<S0, S1, D0, D1> DropLastPair<true>
    for Layout<Til<S0, Til<S1, Nil>>, Til<D0, Til<D1, Nil>>>
where
    S0: ShapeScalar,
    S1: ShapeScalar,
    D0: Leaf + Stride<S0>,
    D1: Leaf + Stride<S1>,
{
    type Shape = S0;
    type Stride = D0;

    fn drop_last_pair(self) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(self.shape.0, self.stride.0)
    }
}

const impl<S0, S1, D0, D1> DropLastPair<false>
    for Layout<Til<S0, Til<S1, Nil>>, Til<D0, Til<D1, Nil>>>
where
    S0: ShapeScalar,
    S1: ShapeScalar,
    D0: Leaf + Stride<S0>,
    D1: Leaf + Stride<S1>,
{
    type Shape = Til<S0, Til<S1, Nil>>;
    type Stride = Til<D0, Til<D1, Nil>>;

    fn drop_last_pair(self) -> Layout<Self::Shape, Self::Stride> {
        self
    }
}

/// Drops the second mode of a 2-mode tail layout when `DROP == true`, keeping a tuple `(S0,):(D0,)`.
#[doc(hidden)]
pub const trait DropLastPairTuple<const DROP: bool>: Copy {
    type Shape: Tuple + Shape;
    type Stride: Tuple + Stride<Self::Shape>;
    fn drop_last_pair_tuple(self) -> Layout<Self::Shape, Self::Stride>;
}

const impl<S0, S1, D0, D1> DropLastPairTuple<true>
    for Layout<Til<S0, Til<S1, Nil>>, Til<D0, Til<D1, Nil>>>
where
    S0: ShapeScalar,
    S1: ShapeScalar,
    D0: Leaf + Stride<S0>,
    D1: Leaf + Stride<S1>,
{
    type Shape = Til<S0, Nil>;
    type Stride = Til<D0, Nil>;

    fn drop_last_pair_tuple(self) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(Til(self.shape.0, Nil), Til(self.stride.0, Nil))
    }
}

const impl<S0, S1, D0, D1> DropLastPairTuple<false>
    for Layout<Til<S0, Til<S1, Nil>>, Til<D0, Til<D1, Nil>>>
where
    S0: ShapeScalar,
    S1: ShapeScalar,
    D0: Leaf + Stride<S0>,
    D1: Leaf + Stride<S1>,
{
    type Shape = Til<S0, Til<S1, Nil>>;
    type Stride = Til<D0, Til<D1, Nil>>;

    fn drop_last_pair_tuple(self) -> Layout<Self::Shape, Self::Stride> {
        self
    }
}

/// Recursively drops a trailing static size-1 mode from a flat tail layout of rank >= 2.
#[doc(hidden)]
pub const trait DropLastFromTail: Copy {
    type Shape: Tuple + Shape;
    type Stride: Tuple + Stride<Self::Shape>;
    fn drop_last_from_tail(self) -> Layout<Self::Shape, Self::Stride>;
}

const impl<S0, S1, D0, D1> DropLastFromTail for Layout<Til<S0, Til<S1, Nil>>, Til<D0, Til<D1, Nil>>>
where
    S0: ShapeScalar,
    S1: ShapeScalar,
    D0: Leaf + Stride<S0>,
    D1: Leaf + Stride<S1>,
    Self: [const] DropLastPairTuple<{ SHAPE_IS_STATIC_ONE::<S1> }>,
{
    type Shape = <Self as DropLastPairTuple<{ SHAPE_IS_STATIC_ONE::<S1> }>>::Shape;
    type Stride = <Self as DropLastPairTuple<{ SHAPE_IS_STATIC_ONE::<S1> }>>::Stride;

    fn drop_last_from_tail(self) -> Layout<Self::Shape, Self::Stride> {
        self.drop_last_pair_tuple()
    }
}

const impl<S0, S1, S2, TS, D0, D1, D2, TD> DropLastFromTail
    for Layout<Til<S0, Til<S1, Til<S2, TS>>>, Til<D0, Til<D1, Til<D2, TD>>>>
where
    S0: ShapeScalar,
    S1: ShapeScalar,
    S2: ShapeScalar,
    TS: Tuple + Shape,
    D0: Leaf + Stride<S0>,
    D1: Leaf + Stride<S1>,
    D2: Leaf + Stride<S2>,
    TD: Tuple + Stride<TS>,
    Layout<Til<S1, Til<S2, TS>>, Til<D1, Til<D2, TD>>>: [const] DropLastFromTail,
{
    type Shape =
        Til<S0, <Layout<Til<S1, Til<S2, TS>>, Til<D1, Til<D2, TD>>> as DropLastFromTail>::Shape>;
    type Stride =
        Til<D0, <Layout<Til<S1, Til<S2, TS>>, Til<D1, Til<D2, TD>>> as DropLastFromTail>::Stride>;

    fn drop_last_from_tail(self) -> Layout<Self::Shape, Self::Stride> {
        let tail = Layout::new(self.shape.1, self.stride.1).drop_last_from_tail();
        Layout::new(
            Til(self.shape.0, tail.shape),
            Til(self.stride.0, tail.stride),
        )
    }
}

/// Drops a trailing static size-1 mode when a flat layout has rank >= 2,
/// preserving a standalone `1:d` layout (matching PyCuTe's `Layout._coalesce`).
#[doc(hidden)]
pub const trait DropTrailingUnit: Copy {
    /// Resulting shape after optional trailing unit elimination.
    type Shape: Shape;
    /// Resulting congruent stride type.
    type Stride: Stride<Self::Shape>;
    /// Drops a trailing static size-1 mode if rank >= 2.
    fn drop_trailing_unit(self) -> Layout<Self::Shape, Self::Stride>;
}

const impl DropTrailingUnit for Layout<Nil, Nil> {
    type Shape = Nil;
    type Stride = Nil;
    fn drop_trailing_unit(self) -> Layout<Self::Shape, Self::Stride> {
        self
    }
}

const impl<S, D> DropTrailingUnit for Layout<S, D>
where
    S: ShapeScalar,
    D: Leaf + Stride<S>,
{
    type Shape = S;
    type Stride = D;
    fn drop_trailing_unit(self) -> Layout<Self::Shape, Self::Stride> {
        self
    }
}

const impl<S, D> DropTrailingUnit for Layout<Til<S, Nil>, Til<D, Nil>>
where
    S: ShapeScalar,
    D: Leaf + Stride<S>,
{
    type Shape = S;
    type Stride = D;
    fn drop_trailing_unit(self) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(self.shape.0, self.stride.0)
    }
}

const impl<S0, S1, D0, D1> DropTrailingUnit for Layout<Til<S0, Til<S1, Nil>>, Til<D0, Til<D1, Nil>>>
where
    S0: ShapeScalar,
    S1: ShapeScalar,
    D0: Leaf + Stride<S0>,
    D1: Leaf + Stride<S1>,
    Self: [const] DropLastPair<{ SHAPE_IS_STATIC_ONE::<S1> }>,
{
    type Shape = <Self as DropLastPair<{ SHAPE_IS_STATIC_ONE::<S1> }>>::Shape;
    type Stride = <Self as DropLastPair<{ SHAPE_IS_STATIC_ONE::<S1> }>>::Stride;

    fn drop_trailing_unit(self) -> Layout<Self::Shape, Self::Stride> {
        self.drop_last_pair()
    }
}

const impl<S0, S1, S2, TS, D0, D1, D2, TD> DropTrailingUnit
    for Layout<Til<S0, Til<S1, Til<S2, TS>>>, Til<D0, Til<D1, Til<D2, TD>>>>
where
    S0: ShapeScalar,
    S1: ShapeScalar,
    S2: ShapeScalar,
    TS: Tuple + Shape,
    D0: Leaf + Stride<S0>,
    D1: Leaf + Stride<S1>,
    D2: Leaf + Stride<S2>,
    TD: Tuple + Stride<TS>,
    Layout<Til<S1, Til<S2, TS>>, Til<D1, Til<D2, TD>>>: [const] DropLastFromTail,
{
    type Shape =
        Til<S0, <Layout<Til<S1, Til<S2, TS>>, Til<D1, Til<D2, TD>>> as DropLastFromTail>::Shape>;
    type Stride =
        Til<D0, <Layout<Til<S1, Til<S2, TS>>, Til<D1, Til<D2, TD>>> as DropLastFromTail>::Stride>;

    fn drop_trailing_unit(self) -> Layout<Self::Shape, Self::Stride> {
        let tail = Layout::new(self.shape.1, self.stride.1).drop_last_from_tail();
        Layout::new(
            Til(self.shape.0, tail.shape),
            Til(self.stride.0, tail.stride),
        )
    }
}

/// Profile-guided extended-domain coalescing (`coalesce_z`).
///
/// Coalesces each leaf subtree under `P` using [`Coalesce<true>`] while preserving outer
/// mode boundaries defined by [`TupleProfile`].
pub const trait CoalesceExtended<P = LeafProfile> {
    /// Resulting profile-preserving coalesced [`Layout`] type.
    type Output;
    /// Coalesces `self` with extended-domain semantics under profile `P`.
    fn coalesce_z(self) -> Self::Output;
}

const impl<S: Shape, D: Stride<S>> CoalesceExtended<NilProfile> for Layout<S, D> {
    type Output = Self;
    fn coalesce_z(self) -> Self::Output {
        self
    }
}

const impl<S, D> CoalesceExtended for Layout<S, D>
where
    Self: Copy,
    S: [const] Flatten,
    D: [const] Flatten,
    Layout<S::Output, D::Output>: [const] Coalesce<true>,
{
    type Output = <Layout<S::Output, D::Output> as Coalesce<true>>::Output;
    fn coalesce_z(self) -> Self::Output {
        self.coalesce_extended()
    }
}

macro_rules! impl_scalar_coalesce_extended_tuple {
    ([$($g:tt)*] $S:ty) => {
        const impl<$($g)* D, PH, NewS, NewD> CoalesceExtended<TupleProfile<PH, NilProfile>>
            for Layout<$S, D>
        where
            $S: Shape,
            D: Stride<$S>,
            Layout<$S, D>: [const] CoalesceExtended<PH, Output = Layout<NewS, NewD>>,
            NewS: Shape,
            NewD: Stride<NewS>,
        {
            type Output = Layout<Til<NewS, Nil>, Til<NewD, Nil>>;
            fn coalesce_z(self) -> Self::Output {
                let head = <Self as CoalesceExtended<PH>>::coalesce_z(self);
                Layout::new(Til(head.shape, Nil), Til(head.stride, Nil))
            }
        }
    };
}

impl_scalar_coalesce_extended_tuple!([] int);
impl_scalar_coalesce_extended_tuple!([const N: int,] Int<N>);

const impl<HS, TS, HD, TD, PH, PT, NewHS, NewTS, NewHD, NewTD>
    CoalesceExtended<TupleProfile<PH, PT>> for Layout<Til<HS, TS>, Til<HD, TD>>
where
    HS: Shape,
    TS: Tuple + Shape,
    HD: Stride<HS>,
    TD: Tuple + Stride<TS>,
    Layout<HS, HD>: [const] CoalesceExtended<PH, Output = Layout<NewHS, NewHD>>,
    Layout<TS, TD>: [const] CoalesceExtended<PT, Output = Layout<NewTS, NewTD>>,
    NewHS: Shape,
    NewTS: Tuple + Shape,
    NewHD: Stride<NewHS>,
    NewTD: Tuple + Stride<NewTS>,
{
    type Output = Layout<Til<NewHS, NewTS>, Til<NewHD, NewTD>>;
    fn coalesce_z(self) -> Self::Output {
        let head = Layout::new(self.shape.0, self.stride.0).coalesce_z();
        let tail = Layout::new(self.shape.1, self.stride.1).coalesce_z();
        Layout::new(Til(head.shape, tail.shape), Til(head.stride, tail.stride))
    }
}

/// Profile-guided standard coalescing (`coalesce(obj, profile)`).
///
/// Coalesces each leaf subtree under `P` using [`Coalesce<false>`] (dropping trailing unit modes)
/// while preserving outer mode boundaries defined by [`TupleProfile`] and leaving [`NilProfile`]
/// subtrees untouched.
pub const trait CoalesceProfile<P = LeafProfile>: Copy {
    /// Resulting profile-preserving coalesced layout or tensor type.
    type Output: Copy;
    /// Coalesces `self` under profile `P`.
    fn coalesce_with_profile(self) -> Self::Output;
}

const impl<S: Shape, D: Stride<S>> CoalesceProfile<NilProfile> for Layout<S, D> {
    type Output = Self;
    fn coalesce_with_profile(self) -> Self::Output {
        self
    }
}

const impl<S, D> CoalesceProfile for Layout<S, D>
where
    Self: Copy,
    S: [const] Flatten,
    D: [const] Flatten,
    Layout<S::Output, D::Output>: [const] Coalesce<true>,
    <Layout<S::Output, D::Output> as Coalesce<true>>::Output: [const] DropTrailingUnit,
{
    type Output = Layout<
        <<Layout<S::Output, D::Output> as Coalesce<true>>::Output as DropTrailingUnit>::Shape,
        <<Layout<S::Output, D::Output> as Coalesce<true>>::Output as DropTrailingUnit>::Stride,
    >;
    fn coalesce_with_profile(self) -> Self::Output {
        self.coalesce_extended().drop_trailing_unit()
    }
}

macro_rules! impl_scalar_coalesce_profile_tuple {
    ([$($g:tt)*] $S:ty) => {
        const impl<$($g)* D, PH, NewS, NewD> CoalesceProfile<TupleProfile<PH, NilProfile>>
            for Layout<$S, D>
        where
            $S: Shape,
            D: Stride<$S>,
            Layout<$S, D>: [const] CoalesceProfile<PH, Output = Layout<NewS, NewD>>,
            NewS: Shape,
            NewD: Stride<NewS>,
        {
            type Output = Layout<Til<NewS, Nil>, Til<NewD, Nil>>;
            fn coalesce_with_profile(self) -> Self::Output {
                let head = <Self as CoalesceProfile<PH>>::coalesce_with_profile(self);
                Layout::new(Til(head.shape, Nil), Til(head.stride, Nil))
            }
        }
    };
}

impl_scalar_coalesce_profile_tuple!([] int);
impl_scalar_coalesce_profile_tuple!([const N: int,] Int<N>);

const impl<HS, TS, HD, TD, PH, PT, NewHS, NewTS, NewHD, NewTD> CoalesceProfile<TupleProfile<PH, PT>>
    for Layout<Til<HS, TS>, Til<HD, TD>>
where
    HS: Shape,
    TS: Tuple + Shape,
    HD: Stride<HS>,
    TD: Tuple + Stride<TS>,
    Layout<HS, HD>: [const] CoalesceProfile<PH, Output = Layout<NewHS, NewHD>>,
    Layout<TS, TD>: [const] CoalesceProfile<PT, Output = Layout<NewTS, NewTD>>,
    NewHS: Shape,
    NewTS: Tuple + Shape,
    NewHD: Stride<NewHS>,
    NewTD: Tuple + Stride<NewTS>,
{
    type Output = Layout<Til<NewHS, NewTS>, Til<NewHD, NewTD>>;
    fn coalesce_with_profile(self) -> Self::Output {
        let head = Layout::new(self.shape.0, self.stride.0).coalesce_with_profile();
        let tail = Layout::new(self.shape.1, self.stride.1).coalesce_with_profile();
        Layout::new(Til(head.shape, tail.shape), Til(head.stride, tail.stride))
    }
}

impl<S, D> Layout<S, D> {
    /// Coalesces `self` mode-by-mode according to profile `P` (dropping trailing unit modes
    /// inside each `LeafProfile` subtree).
    pub const fn coalesce_with_profile<P>(self) -> <Self as CoalesceProfile<P>>::Output
    where
        Self: [const] CoalesceProfile<P>,
    {
        CoalesceProfile::<P>::coalesce_with_profile(self)
    }
}
