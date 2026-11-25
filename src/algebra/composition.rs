//! Layout composition algebra ($A \circ B$).
//!
//! Group composition produces a layout whose domain is $B$ and whose values
//! are $A$ evaluated at $B$'s values:
//!
//! $$R(i) = A(B(i)) \quad \text{for all } i \in [0, |B|)$$
//!
//! # Main Interfaces
//!
//! - [`Composition`] trait (`composition`)
//! - [`Layout::composition`]
//! - [`composition`] free function (supporting [`Layout`], [`Tiler`], and [`Nil`] operands)
//!
//! # Multi-Stage Pipeline
//!
//! 1. **Entry Point & Structural Decomposition** ([`Composition`]):
//!    - `Nil` identity on either side (with `Nil` placeholders in a tuple RHS promoted to `1`).
//!    - RHS distributive over multi-mode layouts: $A \circ (B_0, B_1, \dots) \longrightarrow (A \circ B_0, A \circ B_1, \dots)$.
//!    - By-mode composition over tuple tilers: $(A_0, A_1) \circ (B_0, B_1) \longrightarrow (A_0 \circ B_0, A_1 \circ B_1)$.
//! 2. **Extended-Domain Normalization** ([`CoalesceExtended`]):
//!    Pre-coalesces $A$ per `coprofile(B)` and projects along `B`'s basis path via [`ProjectByPath`].
//! 3. **Atomic Composition**:
//!    - Rank-1 base case: $s:d \circ N:M \longrightarrow N : (d \cdot M)$.
//!    - Multi-mode case: divides out stride $M$, truncates to target extent $N$, coalesces with
//!      extended-domain semantics, and drops a trailing static unit mode.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::composition::composition;
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let a = Layout::new(int_tuple![Int::<6>, Int::<2>], int_tuple![Int::<8>, Int::<2>]);
//! let b = Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<3>, Int::<1>]);
//! assert_eq!(
//!     composition(a, b),
//!     Layout::new(
//!         int_tuple![(Int::<2>, Int::<2>), Int::<3>],
//!         int_tuple![(Int::<24>, Int::<2>), Int::<8>],
//!     )
//! );
//! ```
//!
//! # Notes
//!
//! - When `A` still has multiple modes after coalescing, each RHS mode requires a static
//!   stride and static extents for compile-time divisibility checking.
//! - Stride overflow follows [`crate::integer`] rules (compile-time error for static operands,
//!   runtime panic for dynamic operands).
//!
//! Integer RHS modes entering binary-polynomial strides must combine without
//! carrying, both within each coordinate and between modes on the same axis.
//!
//! ```compile_fail
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{algebra::composition, integer::Int, layout::Layout, swizzle::f2};
//! const BAD: Layout<Int<4>, noodle_cute::swizzle::StaticF2<3>> = composition(
//!     Layout::new(Int::<16>, f2(Int::<1>)), Layout::new(Int::<4>, Int::<3>));
//! ```

use core::marker::ConstParamTy;

use crate::basis::ComparePath;
use crate::basis::{BasisCoeff, ProjectByPath, SingleBasis};
use crate::integer::ops::carry_free_mask;
use crate::integer::uint;
use crate::integer::{
    ArithFloorDiv, FloorQuotient, IS_NON_NEG, Int, Integer, div, floor_div, floor_rem, int, sub,
};
use crate::layout::{Coprofile, Layout, Tiler, tiler_to_layout};
use crate::shape::{Shape, ShapeScalar, ValidShapeExtent};
use crate::stride::{ContainsF2, same_path};
use crate::stride::{Stride, StrideMul, StrideScalar};
use crate::swizzle::F2;
use crate::tuple::{Item, LeafProfile, Nil, Til, Tuple, Wrap};

use super::coalesce::{Coalesce, CoalesceExtended, DropTrailingUnit};

/// Tracks integer bit contributions before distributing composition across RHS modes.
#[doc(hidden)]
pub const trait CompositionMask<S: Copy>: Copy {
    const STATIC_MASK: Option<uint>;
    fn mask(self, shape: S) -> uint;
}
const impl<S: [const] ShapeScalar, I: [const] Integer> CompositionMask<S> for I {
    const STATIC_MASK: Option<uint> = match (S::STATIC_VALUE, I::STATIC_VALUE) {
        (Some(s), Some(d)) => Some(carry_free_mask(s, d).unwrap_or(uint::MAX)),
        _ => None,
    };
    fn mask(self, s: S) -> uint {
        carry_free_mask(s.value(), self.value()).expect("integer stride carries in F2 composition")
    }
}
const impl<S: Copy, V: Copy> CompositionMask<S> for F2<V> {
    const STATIC_MASK: Option<uint> = Some(0);
    fn mask(self, _: S) -> uint {
        0
    }
}
/// Checks cross-mode carries on matching codomain paths in two RHS subtrees.
#[doc(hidden)]
pub const trait CrossModeCarry<S: Copy, R: Copy, T: Copy>: Copy {
    const STATIC_OK: Option<bool>;
    fn disjoint(self, s: S, rhs: R, t: T) -> bool;
}
const impl<L, R, S, T> CrossModeCarry<S, R, T> for L
where
    L: [const] StrideScalar,
    R: [const] StrideScalar,
    S: ShapeScalar,
    T: ShapeScalar,
    L::Path: ComparePath<R::Path>,
    L::Value: [const] CompositionMask<S>,
    R::Value: [const] CompositionMask<T>,
{
    const STATIC_OK: Option<bool> = if !same_path::<L::Path, R::Path>() {
        Some(true)
    } else {
        match (L::Value::STATIC_MASK, R::Value::STATIC_MASK) {
            (Some(l), Some(r)) => Some(l != uint::MAX && r != uint::MAX && l & r == 0),
            _ => None,
        }
    };
    fn disjoint(self, s: S, rhs: R, t: T) -> bool {
        !same_path::<L::Path, R::Path>() || self.value().mask(s) & rhs.value().mask(t) == 0
    }
}
const fn both(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}
const impl<L: StrideScalar, S: ShapeScalar> CrossModeCarry<S, Nil, Nil> for L {
    const STATIC_OK: Option<bool> = Some(true);
    fn disjoint(self, _: S, _: Nil, _: Nil) -> bool {
        true
    }
}
const impl<L, S, RH, RT, TH, TT> CrossModeCarry<S, Til<RH, RT>, Til<TH, TT>> for L
where
    L: StrideScalar + [const] CrossModeCarry<S, RH, TH> + [const] CrossModeCarry<S, RT, TT>,
    S: ShapeScalar,
    RH: Copy,
    RT: Copy,
    TH: Copy,
    TT: Copy,
{
    const STATIC_OK: Option<bool> = both(
        <L as CrossModeCarry<S, RH, TH>>::STATIC_OK,
        <L as CrossModeCarry<S, RT, TT>>::STATIC_OK,
    );
    fn disjoint(self, s: S, r: Til<RH, RT>, t: Til<TH, TT>) -> bool {
        self.disjoint(s, r.0, t.0) && self.disjoint(s, r.1, t.1)
    }
}
const impl<R: Copy, T: Copy> CrossModeCarry<Nil, R, T> for Nil {
    const STATIC_OK: Option<bool> = Some(true);
    fn disjoint(self, _: Nil, _: R, _: T) -> bool {
        true
    }
}
const impl<LH, LT, SH, ST, R: Copy, T: Copy> CrossModeCarry<Til<SH, ST>, R, T> for Til<LH, LT>
where
    LH: [const] CrossModeCarry<SH, R, T>,
    LT: [const] CrossModeCarry<ST, R, T>,
    SH: Copy,
    ST: Copy,
{
    const STATIC_OK: Option<bool> = both(LH::STATIC_OK, LT::STATIC_OK);
    fn disjoint(self, s: Til<SH, ST>, r: R, t: T) -> bool {
        self.0.disjoint(s.0, r, t) && self.1.disjoint(s.1, r, t)
    }
}
const CHECK_OVERLAP<L:ContainsF2,H:CrossModeCarry<S,R,T>,S:Copy,R:Copy,T:Copy>:bool={
    assert!(!L::HAS_F2 || !matches!(H::STATIC_OK,Some(false)), "integer modes carry in F2 composition");
    L::HAS_F2 && H::STATIC_OK.is_none()
};

const NEEDS_MASK_CHECK<L: ContainsF2, D: CompositionMask<S>, S: Copy>: bool =
    {
        assert!(!L::HAS_F2 || !matches!(D::STATIC_MASK, Some(uint::MAX)),
                "integer carries in F2 composition");
        L::HAS_F2 && D::STATIC_MASK.is_none()
    };

/// Compile-time action when dividing out stride `M` from head mode `S0:D0`.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum DivideOutAction {
    /// Zero stride broadcasts a single position across the whole target extent.
    Broadcast,
    /// Stride is a multiple of `S0`: discard head mode and recurse on tail with `floor_div(M, S0)`.
    SkipWholeMode,
    /// Positive stride divides `S0` cleanly: trim extent to `S0 / M` and scale stride by `M`.
    ScaleAndTrim,
    /// Stride does not divide `S0`, but `floor_div(S0, M) >= N - 1` keeps access within mode 0.
    ScaleAndKeep,
    /// Stride cannot be linearly evaluated across modes.
    Invalid,
}

const fn divide_out_action<S: ShapeScalar, M: Integer>() -> DivideOutAction {
    let Some(m) = M::STATIC_VALUE else {
        return DivideOutAction::Invalid;
    };
    if m == 0 {
        return DivideOutAction::Broadcast;
    }
    let Some(s) = S::STATIC_VALUE else {
        return DivideOutAction::Invalid;
    };
    if s <= 0 {
        return DivideOutAction::Invalid;
    }

    if floor_rem(m, s) == 0 {
        return DivideOutAction::SkipWholeMode;
    }
    if m > 0 && floor_rem(s, m) == 0 {
        DivideOutAction::ScaleAndTrim
    } else {
        DivideOutAction::ScaleAndKeep
    }
}

const fn divide_out_keep_ok<S: ShapeScalar, M: Integer, N: ShapeScalar>() -> bool {
    let (Some(s), Some(m), Some(n)) = (S::STATIC_VALUE, M::STATIC_VALUE, N::STATIC_VALUE) else {
        return false;
    };
    if s <= 0 || m == 0 {
        return false;
    }
    if n <= 1 {
        return true;
    }
    if m < 0 {
        return false;
    }
    floor_div(s, m) >= sub(n, 1)
}

/// Compile-time action when truncating mode `S` against remaining target extent `Rem`.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum KeepModeAction {
    /// `rem < s` or `rem == 0`: truncate head mode to `rem` and discard tail modes.
    Truncate,
    /// `rem % s == 0`: keep head mode and recurse on tail with `rem / s`.
    Keep,
    /// `rem` does not divide cleanly across mode boundaries.
    Invalid,
}

const fn keep_mode_action<S: ShapeScalar, Rem: ShapeScalar>() -> KeepModeAction {
    let (Some(s), Some(rem_v)) = (S::STATIC_VALUE, Rem::STATIC_VALUE) else {
        return KeepModeAction::Invalid;
    };
    if rem_v < s || rem_v == 0 {
        KeepModeAction::Truncate
    } else if s > 0 && floor_rem(rem_v, s) == 0 {
        KeepModeAction::Keep
    } else {
        KeepModeAction::Invalid
    }
}

const DIVIDE_OUT_ACTION<S: ShapeScalar, M: Integer>: DivideOutAction =
    divide_out_action::<S, M>();

const DIVIDE_OUT_KEEP_OK<S: ShapeScalar, M: Integer, N: ShapeScalar>: bool =
    divide_out_keep_ok::<S, M, N>();

const DIVIDE_OUT_TRIM_S<S: ShapeScalar, M: Integer>: int = div(
    S::STATIC_VALUE.expect("expected a static extent"),
    M::STATIC_VALUE.expect("expected a static stride"),
);

const KEEP_MODE_ACTION<S: ShapeScalar, Rem: ShapeScalar>: KeepModeAction =
    keep_mode_action::<S, Rem>();

const KEEP_MODE_QUOTIENT<S: ShapeScalar, Rem: ShapeScalar>: int = div(
    Rem::STATIC_VALUE.expect("expected a static target extent"),
    S::STATIC_VALUE.expect("expected a static extent"),
);

/// Divides stride factor `M` out of a coalesced LHS layout for target extent `N`.
#[doc(hidden)]
pub const trait DivideOut<M: Integer, N: ShapeScalar>: Copy {
    type Output: Copy;
    fn divide_out(self, factor: M) -> Self::Output;
}

/// Dispatches one [`DivideOut`] step on head `(S0, D0)` and tail `Layout<TS, TD>` per `ACTION`.
#[doc(hidden)]
pub const trait DivideOutStep<
    S0,
    D0,
    TS,
    TD,
    M: Integer,
    N: ShapeScalar,
    const ACTION: DivideOutAction,
>: Copy
{
    type Output: Copy;
    fn divide_out_step(head_s: S0, head_d: D0, tail: Layout<TS, TD>, factor: M) -> Self::Output;
}

const impl<S, D, M: Integer, N: ShapeScalar> DivideOut<M, N> for Layout<S, D>
where
    S: ShapeScalar,
    D: Stride<S> + [const] StrideMul<M>,
    <D as StrideMul<M>>::Output: Stride<S>,
{
    type Output = Layout<S, <D as StrideMul<M>>::Output>;
    fn divide_out(self, factor: M) -> Self::Output {
        Layout::new(self.shape, self.stride.stride_mul(factor))
    }
}

const impl<S, D, M: Integer, N: ShapeScalar> DivideOut<M, N> for Layout<Til<S, Nil>, Til<D, Nil>>
where
    S: ShapeScalar,
    D: Stride<S> + [const] StrideMul<M>,
    <D as StrideMul<M>>::Output: Stride<S>,
{
    type Output = Layout<Til<S, Nil>, Til<<D as StrideMul<M>>::Output, Nil>>;
    fn divide_out(self, factor: M) -> Self::Output {
        Layout::new(
            Til(self.shape.0, Nil),
            Til(self.stride.0.stride_mul(factor), Nil),
        )
    }
}

const impl<S0, D0, S1, D1, TS, TD, M: Integer, N: ShapeScalar> DivideOut<M, N>
    for Layout<Til<S0, Til<S1, TS>>, Til<D0, Til<D1, TD>>>
where
    S0: ShapeScalar,
    D0: StrideScalar + Stride<S0>,
    S1: ShapeScalar,
    D1: StrideScalar + Stride<S1>,
    TS: Tuple + Shape,
    TD: Tuple + Stride<TS>,
    Layout<Til<S0, Til<S1, TS>>, Til<D0, Til<D1, TD>>>: [const] DivideOutStep<
            S0,
            D0,
            Til<S1, TS>,
            Til<D1, TD>,
            M,
            N,
            { DIVIDE_OUT_ACTION::<S0, M> },
        >,
{
    type Output = <Layout<Til<S0, Til<S1, TS>>, Til<D0, Til<D1, TD>>> as DivideOutStep<
        S0,
        D0,
        Til<S1, TS>,
        Til<D1, TD>,
        M,
        N,
        { DIVIDE_OUT_ACTION::<S0, M> },
    >>::Output;

    fn divide_out(self, factor: M) -> Self::Output {
        <Layout<Til<S0, Til<S1, TS>>, Til<D0, Til<D1, TD>>> as DivideOutStep<
            S0,
            D0,
            Til<S1, TS>,
            Til<D1, TD>,
            M,
            N,
            { DIVIDE_OUT_ACTION::<S0, M> },
        >>::divide_out_step(
            self.shape.0,
            self.stride.0,
            Layout::new(self.shape.1, self.stride.1),
            factor,
        )
    }
}

const impl<S0, D0, TS, TD, M, N> DivideOutStep<S0, D0, TS, TD, M, N, { DivideOutAction::Broadcast }>
    for Layout<Til<S0, TS>, Til<D0, TD>>
where
    S0: ShapeScalar,
    D0: Stride<S0>,
    TS: Tuple + Shape,
    TD: Tuple + Stride<TS>,
    M: Integer + Stride<Int<1>>,
    N: ShapeScalar,
{
    type Output = Layout<Int<1>, M>;

    fn divide_out_step(_: S0, _: D0, _: Layout<TS, TD>, factor: M) -> Self::Output {
        Layout::new(Int, factor)
    }
}

const impl<S0, D0, TS, TD, M: Integer, N: ShapeScalar>
    DivideOutStep<S0, D0, TS, TD, M, N, { DivideOutAction::SkipWholeMode }>
    for Layout<Til<S0, TS>, Til<D0, TD>>
where
    S0: [const] ShapeScalar,
    D0: StrideScalar + Stride<S0>,
    TS: Tuple + Shape + Copy,
    TD: Tuple + Stride<TS> + Copy,
    M: [const] Integer + [const] ArithFloorDiv<M, S0>,
    Layout<TS, TD>: [const] DivideOut<FloorQuotient<M, S0>, N>,
{
    type Output = <Layout<TS, TD> as DivideOut<FloorQuotient<M, S0>, N>>::Output;

    fn divide_out_step(head_s: S0, _: D0, tail: Layout<TS, TD>, factor: M) -> Self::Output {
        tail.divide_out(floor_div(factor, head_s))
    }
}

const impl<S0, D0, TS, TD, M: Integer, N: ShapeScalar>
    DivideOutStep<S0, D0, TS, TD, M, N, { DivideOutAction::ScaleAndTrim }>
    for Layout<Til<S0, TS>, Til<D0, TD>>
where
    S0: ShapeScalar,
    D0: Stride<S0> + [const] StrideMul<M>,
    TS: Tuple + Shape + Copy,
    TD: Tuple + Stride<TS> + Copy,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ DIVIDE_OUT_TRIM_S::<S0, M> }> }>,
    <D0 as StrideMul<M>>::Output: Stride<Int<{ DIVIDE_OUT_TRIM_S::<S0, M> }>>,
{
    type Output =
        Layout<Til<Int<{ DIVIDE_OUT_TRIM_S::<S0, M> }>, TS>, Til<<D0 as StrideMul<M>>::Output, TD>>;

    fn divide_out_step(_: S0, head_d: D0, tail: Layout<TS, TD>, factor: M) -> Self::Output {
        Layout::new(
            Til(Int::<{ DIVIDE_OUT_TRIM_S::<S0, M> }>, tail.shape),
            Til(head_d.stride_mul(factor), tail.stride),
        )
    }
}

/// Compile-time guard validating that `ScaleAndKeep` stays within head mode extent (`OK == true`).
#[doc(hidden)]
pub const trait DivideOutKeepAllowed<const OK: bool> {}

const impl DivideOutKeepAllowed<true> for () {}

const impl<S0, D0, TS, TD, M: Integer, N: ShapeScalar>
    DivideOutStep<S0, D0, TS, TD, M, N, { DivideOutAction::ScaleAndKeep }>
    for Layout<Til<S0, TS>, Til<D0, TD>>
where
    S0: ShapeScalar,
    D0: Stride<S0> + [const] StrideMul<M>,
    TS: Tuple + Shape + Copy,
    TD: Tuple + Stride<TS> + Copy,
    (): [const] DivideOutKeepAllowed<{ DIVIDE_OUT_KEEP_OK::<S0, M, N> }>,
    <D0 as StrideMul<M>>::Output: Stride<S0>,
{
    type Output = Layout<Til<S0, TS>, Til<<D0 as StrideMul<M>>::Output, TD>>;

    fn divide_out_step(head_s: S0, head_d: D0, tail: Layout<TS, TD>, factor: M) -> Self::Output {
        Layout::new(
            Til(head_s, tail.shape),
            Til(head_d.stride_mul(factor), tail.stride),
        )
    }
}

/// Truncates a post-[`DivideOut`] layout to cover exactly target extent `Rem`.
#[doc(hidden)]
pub const trait KeepShape<Rem: ShapeScalar>: Copy {
    type Output: Copy;
    fn keep_shape(self, rem: Rem) -> Self::Output;
}

/// Dispatches one [`KeepShape`] step on head `(S0, D0)` and tail `Layout<TS, TD>` per `ACTION`.
#[doc(hidden)]
pub const trait KeepModeStep<S0, D0, TS, TD, Rem: ShapeScalar, const ACTION: KeepModeAction>:
    Copy
{
    type Output: Copy;
    fn keep_mode_step(head_s: S0, head_d: D0, tail: Layout<TS, TD>, rem: Rem) -> Self::Output;
}

const impl<S, D, Rem: [const] ShapeScalar> KeepShape<Rem> for Layout<S, D>
where
    S: ShapeScalar,
    D: StrideScalar + Stride<Rem>,
{
    type Output = Layout<Rem, D>;
    fn keep_shape(self, rem: Rem) -> Self::Output {
        let _ = rem.size();
        Layout::new(rem, self.stride)
    }
}

const impl<S, D, Rem: [const] ShapeScalar> KeepShape<Rem> for Layout<Til<S, Nil>, Til<D, Nil>>
where
    S: ShapeScalar,
    D: StrideScalar + Stride<S> + Stride<Rem>,
{
    type Output = Layout<Til<Rem, Nil>, Til<D, Nil>>;
    fn keep_shape(self, rem: Rem) -> Self::Output {
        let _ = rem.size();
        Layout::new(Til(rem, Nil), Til(self.stride.0, Nil))
    }
}

const impl<S0, D0, S1, D1, TS, TD, Rem: ShapeScalar> KeepShape<Rem>
    for Layout<Til<S0, Til<S1, TS>>, Til<D0, Til<D1, TD>>>
where
    S0: ShapeScalar,
    D0: StrideScalar + Stride<S0>,
    S1: ShapeScalar,
    D1: StrideScalar + Stride<S1>,
    TS: Tuple + Shape,
    TD: Tuple + Stride<TS>,
    Layout<Til<S0, Til<S1, TS>>, Til<D0, Til<D1, TD>>>: [const] KeepModeStep<S0, D0, Til<S1, TS>, Til<D1, TD>, Rem, { KEEP_MODE_ACTION::<S0, Rem> }>,
{
    type Output = <Layout<Til<S0, Til<S1, TS>>, Til<D0, Til<D1, TD>>> as KeepModeStep<
        S0,
        D0,
        Til<S1, TS>,
        Til<D1, TD>,
        Rem,
        { KEEP_MODE_ACTION::<S0, Rem> },
    >>::Output;

    fn keep_shape(self, rem: Rem) -> Self::Output {
        <Layout<Til<S0, Til<S1, TS>>, Til<D0, Til<D1, TD>>> as KeepModeStep<
            S0,
            D0,
            Til<S1, TS>,
            Til<D1, TD>,
            Rem,
            { KEEP_MODE_ACTION::<S0, Rem> },
        >>::keep_mode_step(
            self.shape.0,
            self.stride.0,
            Layout::new(self.shape.1, self.stride.1),
            rem,
        )
    }
}

const impl<S0, D0, TS, TD, Rem: [const] ShapeScalar>
    KeepModeStep<S0, D0, TS, TD, Rem, { KeepModeAction::Truncate }>
    for Layout<Til<S0, TS>, Til<D0, TD>>
where
    S0: ShapeScalar,
    D0: StrideScalar + Stride<S0> + Stride<Rem>,
    TS: Tuple + Shape + Copy,
    TD: Tuple + Stride<TS> + Copy,
{
    type Output = Layout<Til<Rem, Nil>, Til<D0, Nil>>;

    fn keep_mode_step(_: S0, head_d: D0, _: Layout<TS, TD>, rem: Rem) -> Self::Output {
        let _ = rem.size();
        Layout::new(Til(rem, Nil), Til(head_d, Nil))
    }
}

const impl<S0, D0, TS, TD, Rem: ShapeScalar, TailS, TailD>
    KeepModeStep<S0, D0, TS, TD, Rem, { KeepModeAction::Keep }> for Layout<Til<S0, TS>, Til<D0, TD>>
where
    S0: ShapeScalar,
    D0: StrideScalar + Stride<S0>,
    TS: Tuple + Shape + Copy,
    TD: Tuple + Stride<TS> + Copy,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ KEEP_MODE_QUOTIENT::<S0, Rem> }> }>,
    Layout<TS, TD>:
        [const] KeepShape<Int<{ KEEP_MODE_QUOTIENT::<S0, Rem> }>, Output = Layout<TailS, TailD>>,
    TailS: Tuple + Shape,
    TailD: Tuple + Stride<TailS>,
    Til<S0, TailS>: Shape,
    Til<D0, TailD>: Stride<Til<S0, TailS>>,
{
    type Output = Layout<Til<S0, TailS>, Til<D0, TailD>>;

    fn keep_mode_step(head_s: S0, head_d: D0, tail: Layout<TS, TD>, _: Rem) -> Self::Output {
        let tail_kept = tail.keep_shape(Int);
        Layout::new(Til(head_s, tail_kept.shape), Til(head_d, tail_kept.stride))
    }
}

/// Composes an extended-coalesced LHS layout with a single scalar-strided RHS mode `SB:DB`.
#[doc(hidden)]
pub const trait AtomCompose<B>: Copy {
    type Output: Copy;
    fn atom_compose(self, b: B) -> Self::Output;
}

const impl<B: Copy> AtomCompose<B> for Layout<Nil, Nil> {
    type Output = Layout<Nil, Nil>;
    fn atom_compose(self, _: B) -> Self::Output {
        Layout::new(Nil, Nil)
    }
}

const impl<SA, DA, SB, DB> AtomCompose<Layout<SB, DB>> for Layout<SA, DA>
where
    SA: [const] ShapeScalar,
    DA: ContainsF2 + StrideScalar + Stride<SA> + [const] StrideMul<DB>,
    SB: [const] ShapeScalar,
    DB: [const] CompositionMask<SB> + BasisCoeff + Stride<SB>,
    <DA as StrideMul<DB>>::Output: Stride<SB>,
{
    type Output = Layout<SB, <DA as StrideMul<DB>>::Output>;
    fn atom_compose(self, b: Layout<SB, DB>) -> Self::Output {
        let _ = self.shape.size();
        let _ = b.shape.size();
        if NEEDS_MASK_CHECK::<DA, DB, SB> {
            b.stride.mask(b.shape);
        }
        Layout::new(b.shape, self.stride.stride_mul(b.stride))
    }
}

const impl<S0, TS, D0, TD, SB, DB, DivOut, KeptOut, CoalOut> AtomCompose<Layout<SB, DB>>
    for Layout<Til<S0, TS>, Til<D0, TD>>
where
    S0: ShapeScalar,
    D0: StrideScalar + Stride<S0>,
    TS: Tuple + Shape,
    TD: Tuple + Stride<TS>,
    SB: [const] ShapeScalar,
    DB: Integer + Stride<SB>,
    Self: [const] DivideOut<DB, SB, Output = DivOut>,
    DivOut: [const] KeepShape<SB, Output = KeptOut>,
    KeptOut: [const] Coalesce<true, Output = CoalOut>,
    CoalOut: [const] DropTrailingUnit,
{
    type Output = Layout<CoalOut::Shape, CoalOut::Stride>;

    fn atom_compose(self, b: Layout<SB, DB>) -> Self::Output {
        let _ = b.shape.size();
        let div = self.divide_out(b.stride);
        let kept = div.keep_shape(b.shape);
        Coalesce::<true>::coalesce(kept).drop_trailing_unit()
    }
}

/// Replaces `Nil` leaf slots in a tiler tree with `Int<1>` (preserving terminating `Nil`).
#[doc(hidden)]
pub const trait ReplaceNilHead: Item {
    type Output: Item;
    fn replace_nil_head(self) -> Self::Output;
}

/// Replaces `Nil` head slots along a tiler tuple chain while preserving the terminal `Nil`.
#[doc(hidden)]
pub const trait ReplaceNilTail: Tuple {
    type Output: Tuple;
    fn replace_nil_tail(self) -> Self::Output;
}

const impl ReplaceNilHead for Nil {
    type Output = Int<1>;
    fn replace_nil_head(self) -> Int<1> {
        Int
    }
}

const impl ReplaceNilHead for int {
    type Output = int;
    fn replace_nil_head(self) -> int {
        self
    }
}

const impl<const N: int> ReplaceNilHead for Int<N> {
    type Output = Int<N>;
    fn replace_nil_head(self) -> Int<N> {
        self
    }
}

const impl<S: Shape, D: Stride<S>> ReplaceNilHead for Layout<S, D> {
    type Output = Self;
    fn replace_nil_head(self) -> Self {
        self
    }
}

const impl<H: [const] ReplaceNilHead, T: Tuple + [const] ReplaceNilTail> ReplaceNilHead
    for Til<H, T>
{
    type Output = Til<H::Output, T::Output>;
    fn replace_nil_head(self) -> Self::Output {
        Til(self.0.replace_nil_head(), self.1.replace_nil_tail())
    }
}

const impl ReplaceNilTail for Nil {
    type Output = Nil;
    fn replace_nil_tail(self) -> Nil {
        Nil
    }
}

const impl<H: [const] ReplaceNilHead, T: Tuple + [const] ReplaceNilTail> ReplaceNilTail
    for Til<H, T>
{
    type Output = Til<H::Output, T::Output>;
    fn replace_nil_tail(self) -> Self::Output {
        Til(self.0.replace_nil_head(), self.1.replace_nil_tail())
    }
}

/// By-mode tuple composition helper zipping modes of wrapped `A` with tuple tiler `B`.
#[doc(hidden)]
pub const trait CompositionTuple<B>: Copy {
    type Shape: Tuple + Shape;
    type Stride: Tuple + Stride<Self::Shape>;
    fn composition_tuple(self, b: B) -> Layout<Self::Shape, Self::Stride>;
}

const impl<SA: Shape, DA: Stride<SA>> CompositionTuple<Nil> for Layout<SA, DA> {
    type Shape = Nil;
    type Stride = Nil;
    fn composition_tuple(self, _: Nil) -> Layout<Nil, Nil> {
        Layout::new(Nil, Nil)
    }
}

const impl<ASH, AST, ADH, ADT, BH, BT, HS, HD> CompositionTuple<Til<BH, BT>>
    for Layout<Til<ASH, AST>, Til<ADH, ADT>>
where
    ASH: Shape,
    AST: Tuple + Shape,
    ADH: Stride<ASH>,
    ADT: Tuple + Stride<AST>,
    BH: Copy,
    BT: Copy,
    Layout<ASH, ADH>: [const] Composition<BH, Output = Layout<HS, HD>>,
    HS: Shape,
    HD: Stride<HS>,
    Layout<AST, ADT>: [const] CompositionTuple<BT>,
{
    type Shape = Til<HS, <Layout<AST, ADT> as CompositionTuple<BT>>::Shape>;
    type Stride = Til<HD, <Layout<AST, ADT> as CompositionTuple<BT>>::Stride>;

    fn composition_tuple(self, b: Til<BH, BT>) -> Layout<Self::Shape, Self::Stride> {
        let head = Layout::new(self.shape.0, self.stride.0).composition(b.0);
        let tail = Layout::new(self.shape.1, self.stride.1).composition_tuple(b.1);
        Layout::new(Til(head.shape, tail.shape), Til(head.stride, tail.stride))
    }
}

/// Layout (or tensor) composition algebra $A \circ B$.
pub const trait Composition<B>: Copy {
    /// Resulting composed [`Layout`], [`Tensor`](crate::tensor::Tensor), or [`Nil`].
    type Output: Copy;
    /// Computes `self o b`, panicking if a dynamic precondition or overflow check fails.
    fn composition(self, b: B) -> Self::Output;
}

const impl Composition<Nil> for Nil {
    type Output = Nil;
    fn composition(self, _: Nil) -> Nil {
        Nil
    }
}

const impl Composition<int> for Nil {
    type Output = Layout<int, Int<1>>;
    fn composition(self, b: int) -> Self::Output {
        let _ = b.size();
        Layout::new(b, Int::<1>)
    }
}

const impl<const N: int> Composition<Int<N>> for Nil
where
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<N> }>,
{
    type Output = Layout<Int<N>, Int<1>>;
    fn composition(self, b: Int<N>) -> Self::Output {
        Layout::new(b, Int::<1>)
    }
}

const impl<SB: Shape, DB: Stride<SB>> Composition<Layout<SB, DB>> for Nil {
    type Output = Layout<SB, DB>;
    fn composition(self, b: Layout<SB, DB>) -> Self::Output {
        b
    }
}

const impl<BH, BT> Composition<Til<BH, BT>> for Nil
where
    Til<BH, BT>: [const] ReplaceNilTail,
    <Til<BH, BT> as ReplaceNilTail>::Output: [const] Tiler,
    <<Til<BH, BT> as ReplaceNilTail>::Output as Tiler>::OutputLayout: Copy,
{
    type Output = <<Til<BH, BT> as ReplaceNilTail>::Output as Tiler>::OutputLayout;
    fn composition(self, b: Til<BH, BT>) -> Self::Output {
        tiler_to_layout(b.replace_nil_tail())
    }
}

macro_rules! impl_tiler_composition {
    ([$($g:tt)*] $T:ty) => {
        const impl<$($g)* B> Composition<B> for $T
        where
            Self: [const] Tiler,
            <Self as Tiler>::OutputLayout: [const] Composition<B>,
        {
            type Output = <<Self as Tiler>::OutputLayout as Composition<B>>::Output;

            fn composition(self, b: B) -> Self::Output {
                tiler_to_layout(self).composition(b)
            }
        }
    };
}

impl_tiler_composition!([] int);
impl_tiler_composition!([const N: int,] Int<N>);
impl_tiler_composition!([H: Item, T: Tuple,] Til<H, T>);

const impl<SA: Shape, DA: Stride<SA>> Composition<Nil> for Layout<SA, DA> {
    type Output = Self;
    fn composition(self, _: Nil) -> Self {
        self
    }
}

macro_rules! impl_layout_compose_scalar_tiler {
    ([$($g:tt)*] $S:ty) => {
        const impl<SA, DA, $($g)*> Composition<$S> for Layout<SA, DA>
        where
            SA: Shape,
            DA: Stride<SA>,
            $S: Shape,
            Int<1>: Stride<$S>,
            Self: [const] Composition<Layout<$S, Int<1>>>,
        {
            type Output = <Self as Composition<Layout<$S, Int<1>>>>::Output;

            fn composition(self, b: $S) -> Self::Output {
                self.composition(Layout::new(b, Int::<1>))
            }
        }
    };
}

impl_layout_compose_scalar_tiler!([] int);
impl_layout_compose_scalar_tiler!([const N: int] Int<N>);

const impl<SA, DA, BH: Copy, BT: Copy> Composition<Til<BH, BT>> for Layout<SA, DA>
where
    SA: Shape + [const] Wrap,
    DA: Stride<SA> + [const] Wrap,
    SA::Wrapped: Shape,
    DA::Wrapped: Stride<SA::Wrapped>,
    Layout<SA::Wrapped, DA::Wrapped>: [const] CompositionTuple<Til<BH, BT>>,
{
    type Output = Layout<
        <Layout<SA::Wrapped, DA::Wrapped> as CompositionTuple<Til<BH, BT>>>::Shape,
        <Layout<SA::Wrapped, DA::Wrapped> as CompositionTuple<Til<BH, BT>>>::Stride,
    >;

    fn composition(self, b: Til<BH, BT>) -> Self::Output {
        Layout::new(self.shape.wrap(), self.stride.wrap()).composition_tuple(b)
    }
}

const impl<SA: Shape, DA: Stride<SA>> Composition<Layout<Nil, Nil>> for Layout<SA, DA> {
    type Output = Layout<Nil, Nil>;
    fn composition(self, _: Layout<Nil, Nil>) -> Self::Output {
        Layout::new(Nil, Nil)
    }
}

// RHS distributive over multi-mode layouts: (BS0, (BS1, BST))
const impl<SA, DA, BS0, BS1, BST, BD0, BD1, BDT, HeadOutS, HeadOutD, TailOutS, TailOutD>
    Composition<Layout<Til<BS0, Til<BS1, BST>>, Til<BD0, Til<BD1, BDT>>>> for Layout<SA, DA>
where
    SA: Shape,
    DA: ContainsF2 + Stride<SA>,
    BD0: [const] CrossModeCarry<BS0, Til<BD1, BDT>, Til<BS1, BST>>,
    BS0: Shape,
    BD0: Stride<BS0>,
    BS1: Shape,
    BD1: Stride<BS1>,
    BST: Tuple + Shape,
    BDT: Tuple + Stride<BST>,
    Self: [const] Composition<Layout<BS0, BD0>, Output = Layout<HeadOutS, HeadOutD>>,
    Self: [const] Composition<
            Layout<Til<BS1, BST>, Til<BD1, BDT>>,
            Output = Layout<TailOutS, TailOutD>,
        >,
    HeadOutS: Shape,
    HeadOutD: Stride<HeadOutS>,
    TailOutS: Tuple + Shape,
    TailOutD: Tuple + Stride<TailOutS>,
{
    type Output = Layout<Til<HeadOutS, TailOutS>, Til<HeadOutD, TailOutD>>;

    fn composition(
        self,
        b: Layout<Til<BS0, Til<BS1, BST>>, Til<BD0, Til<BD1, BDT>>>,
    ) -> Self::Output {
        if CHECK_OVERLAP::<DA, BD0, BS0, Til<BD1, BDT>, Til<BS1, BST>> {
            assert!(
                b.stride.0.disjoint(b.shape.0, b.stride.1, b.shape.1),
                "integer modes carry in F2 composition"
            );
        }
        let head_res = self.composition(Layout::new(b.shape.0, b.stride.0));
        let tail_res = self.composition(Layout::new(b.shape.1, b.stride.1));
        Layout::new(
            Til(head_res.shape, tail_res.shape),
            Til(head_res.stride, tail_res.stride),
        )
    }
}

// RHS distributive over singleton tuple layout: (BS0,)
const impl<SA, DA, BS0, BD0, HeadOutS, HeadOutD> Composition<Layout<Til<BS0, Nil>, Til<BD0, Nil>>>
    for Layout<SA, DA>
where
    SA: Shape,
    DA: Stride<SA>,
    BS0: Shape,
    BD0: Stride<BS0>,
    Self: [const] Composition<Layout<BS0, BD0>, Output = Layout<HeadOutS, HeadOutD>>,
    HeadOutS: Shape,
    HeadOutD: Stride<HeadOutS>,
{
    type Output = Layout<Til<HeadOutS, Nil>, Til<HeadOutD, Nil>>;

    fn composition(self, b: Layout<Til<BS0, Nil>, Til<BD0, Nil>>) -> Self::Output {
        let head_res = self.composition(Layout::new(b.shape.0, b.stride.0));
        Layout::new(Til(head_res.shape, Nil), Til(head_res.stride, Nil))
    }
}

// RHS single mode (integer or ScaledBasis stride):
// 1. Coalesce A per coprofile(B)
// 2. Project sublayout along B's basis path via ProjectByPath
// 3. Coalesce selected sublayout with LeafProfile
// 4. AtomCompose with scalar stride value
const impl<SA, DA, SB, DB, CoS, CoD, SubS, SubD, NormS, NormD> Composition<Layout<SB, DB>>
    for Layout<SA, DA>
where
    SA: Shape,
    DA: Stride<SA>,
    SB: [const] ShapeScalar,
    DB: [const] StrideScalar + [const] SingleBasis + Stride<SB>,
    DB::Value: BasisCoeff + Stride<SB>,
    Layout<SB, DB>: [const] Coprofile,
    Self: [const] CoalesceExtended<
            <<Layout<SB, DB> as Coprofile>::Output as Item>::Profile,
            Output = Layout<CoS, CoD>,
        >,
    CoS: Shape,
    CoD: Stride<CoS>,
    Layout<CoS, CoD>: [const] ProjectByPath<<DB as SingleBasis>::Path, Output = Layout<SubS, SubD>>,
    SubS: Shape,
    SubD: Stride<SubS>,
    Layout<SubS, SubD>: [const] CoalesceExtended<LeafProfile, Output = Layout<NormS, NormD>>,
    NormS: Shape,
    NormD: Stride<NormS>,
    Layout<NormS, NormD>: [const] AtomCompose<Layout<SB, DB::Value>>,
{
    type Output = <Layout<NormS, NormD> as AtomCompose<Layout<SB, DB::Value>>>::Output;

    fn composition(self, b: Layout<SB, DB>) -> Self::Output {
        assert!(
            b.shape.value() >= 0,
            "composition RHS extent must be non-negative"
        );
        let a_coprof = self.coalesce_z();
        let sub_a = a_coprof.project_path();
        let norm_a = sub_a.coalesce_z();
        norm_a.atom_compose(Layout::new(b.shape, b.stride.value()))
    }
}

impl<S, D> Layout<S, D> {
    /// Computes the layout composition `self o b`.
    pub const fn composition<B>(self, b: B) -> <Self as Composition<B>>::Output
    where
        Self: [const] Composition<B>,
    {
        Composition::composition(self, b)
    }
}

/// Computes the layout composition `a o b`.
///
/// When `a` still has multiple modes after coalescing, each right-hand mode
/// uses a static integer stride and the layout extents used for division must be static.
pub const fn composition<A, B>(a: A, b: B) -> <A as Composition<B>>::Output
where
    A: [const] Composition<B>,
{
    a.composition(b)
}
