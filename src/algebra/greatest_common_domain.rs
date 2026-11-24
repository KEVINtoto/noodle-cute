//! Greatest common domain of two CuTe shapes (or layouts).
//!
//! [`greatest_common_domain`] computes a depth-1 [`Layout`] whose shape is an ordered
//! factorization of the common divisor that `shape(A)` and `shape(B)` share in colexicographical
//! order, and whose stride records the exact linear offset at which each common factor appears.
//!
//! Because admissibility of composition $S \circ (s : d)$ for arbitrary strides on shape $S$
//! requires every factor to start and end cleanly within single leaves of both shapes, the
//! algorithm walks the flattened positive leaf sequences of `shape(A)` and `shape(B)` while
//! maintaining their running prefix products $(p_A, p_B)$:
//!
//! 1. Skip unit (`1`) leaves.
//! 2. Compute the smallest jointly aligned offset $L = \mathrm{lcm}(p_A, p_B)$ and residues
//!    $r_A = L / p_A$, $r_B = L / p_B$.
//! 3. If $r_A \mid a_i$ and $r_B \mid b_j$ and $g = \gcd(a_i / r_A, b_j / r_B) > 1$:
//!    emit mode $g : L$, divide $a_i$ by $r_A g$ and $b_j$ by $r_B g$, and set
//!    $p_A = p_B = L \cdot g$.
//! 4. Otherwise compare leaf ends $e_A = p_A a_i$ and $e_B = p_B b_j$: advance $A$ if
//!    $e_A \mid e_B$ or $e_B \nmid e_A$, and symmetrically advance $B$ if $e_B \mid e_A$ or
//!    $e_A \nmid e_B$.
//! 5. If no common factor exists, return the trivial singleton `Layout((1,), (0,))`.
//!
//! # Key Post-Conditions
//!
//! - `depth(R) == 1`
//! - `size(R)` divides `gcd(size(A), size(B))`
//! - Symmetric: `greatest_common_domain(A, B) == greatest_common_domain(B, A)`
//! - `composition(shape(A), R)` and `composition(shape(B), R)` are both admissible
//! - `size(greatest_common_domain(logical_divide(shape(A), R)[1], logical_divide(shape(B), R)[1])) == 1`
//!
//! # Main Interfaces
//!
//! - [`IntoGcdShape`] (implemented for [`Layout`] and every [`Shape`])
//! - [`GreatestCommonDomain`] trait (`greatest_common_domain`)
//! - [`Layout::greatest_common_domain`]
//! - [`greatest_common_domain`] free function
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::greatest_common_domain::greatest_common_domain;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::int_tuple;
//!
//! let g = greatest_common_domain(
//!     int_tuple![Int::<5>, Int::<3>, Int::<4>],
//!     int_tuple![Int::<10>, Int::<6>],
//! );
//! assert_eq!(
//!     g,
//!     Layout::new(
//!         int_tuple![Int::<5>, Int::<2>],
//!         int_tuple![Int::<1>, Int::<30>],
//!     )
//! );
//! ```
//!
//! # Notes
//!
//! - All input shape leaves must be strictly positive: static zero extents fail at compile time,
//!   and dynamic zero or negative extents panic at runtime.
//! - Running-offset overflow past `int::MAX` is rejected at compile time for static operands and
//!   panics at runtime for dynamic operands.
//!
//! Static zero extents are rejected at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::greatest_common_domain::greatest_common_domain;
//! use noodle_cute::integer::Int;
//!
//! let _ = greatest_common_domain(Int::<0>, Int::<4>);
//! ```
//!
//! Static running-offset overflow past `int::MAX` is rejected at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::greatest_common_domain::greatest_common_domain;
//! use noodle_cute::integer::Int;
//! use noodle_cute::int_tuple;
//!
//! let _ = greatest_common_domain(
//!     int_tuple![Int::<1_500_000_000>, Int::<2>, Int::<5>],
//!     int_tuple![Int::<1_500_000_000>, Int::<2>, Int::<5>],
//! );
//! ```

use core::marker::ConstParamTy;

use crate::integer::ops::{checked_lcm, divides_product};
use crate::integer::{IS_NON_NEG, Int, div, gcd, int, lcm, mul, rem};
use crate::layout::Layout;
use crate::shape::{AssertPositiveShape, Shape, ShapeScalar, ValidShapeExtent};
use crate::stride::Stride;
use crate::tuple::{FlatTuple, Flatten, Item, Nil, Til, Tuple};

/// Extracts the [`Shape`] from a [`Layout`] or any [`Shape`] value.
pub const trait IntoGcdShape: Copy {
    /// Underlying shape type.
    type Shape: [const] Shape;

    /// Returns the shape of `self` (ignoring strides for layouts).
    fn into_gcd_shape(self) -> Self::Shape;
}

const impl<S: [const] Shape, D: Stride<S>> IntoGcdShape for Layout<S, D> {
    type Shape = S;

    fn into_gcd_shape(self) -> S {
        self.shape
    }
}

const impl IntoGcdShape for Nil {
    type Shape = Nil;

    fn into_gcd_shape(self) -> Nil {
        Nil
    }
}

const impl IntoGcdShape for int {
    type Shape = int;

    fn into_gcd_shape(self) -> int {
        self
    }
}

const impl<const N: int> IntoGcdShape for Int<N>
where
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<N> }>,
{
    type Shape = Int<N>;

    fn into_gcd_shape(self) -> Int<N> {
        self
    }
}

const impl<H: [const] Shape, T: [const] Tuple + [const] Shape> IntoGcdShape for Til<H, T> {
    type Shape = Self;

    fn into_gcd_shape(self) -> Self {
        self
    }
}

const IS_STATIC_ONE<S: ShapeScalar>: bool = matches!(S::STATIC_VALUE, Some(1));

/// Conditionally prepends leaf `H` to stripped tail `Self` when `SKIP` is `false`.
#[doc(hidden)]
pub const trait StripStaticOneStep<H: ShapeScalar, const SKIP: bool>: FlatTuple {
    type Output: FlatTuple;
    fn strip_static_one_step(self, head: H) -> Self::Output;
}

const impl<H: ShapeScalar, Rest: FlatTuple> StripStaticOneStep<H, true> for Rest {
    type Output = Rest;
    fn strip_static_one_step(self, _: H) -> Self::Output {
        self
    }
}

const impl<H: ShapeScalar, Rest: FlatTuple> StripStaticOneStep<H, false> for Rest {
    type Output = Til<H, Rest>;
    fn strip_static_one_step(self, head: H) -> Self::Output {
        Til(head, self)
    }
}

/// Removes compile-time `Int<1>` leaves from a flat shape tuple, preserving dynamic leaves.
#[doc(hidden)]
pub const trait StripStaticOnes: FlatTuple {
    type Output: FlatTuple;
    fn strip_static_ones(self) -> Self::Output;
}

const impl StripStaticOnes for Nil {
    type Output = Nil;
    fn strip_static_ones(self) -> Nil {
        Nil
    }
}

const impl<H, T> StripStaticOnes for Til<H, T>
where
    H: ShapeScalar,
    T: [const] StripStaticOnes,
    T::Output: [const] StripStaticOneStep<H, { IS_STATIC_ONE::<H> }>,
{
    type Output = <T::Output as StripStaticOneStep<H, { IS_STATIC_ONE::<H> }>>::Output;

    fn strip_static_ones(self) -> Self::Output {
        self.1.strip_static_ones().strip_static_one_step(self.0)
    }
}

/// Copies leaves of a flat shape tuple into a fixed-length `[int; LEN]` array starting at `idx`.
#[doc(hidden)]
pub const trait FillIntArray<const LEN: usize>: FlatTuple {
    fn fill_int_array(self, out: &mut [int; LEN], idx: usize);
}

const impl<const LEN: usize> FillIntArray<LEN> for Nil {
    fn fill_int_array(self, _: &mut [int; LEN], _: usize) {}
}

const impl<H, T, const LEN: usize> FillIntArray<LEN> for Til<H, T>
where
    H: [const] ShapeScalar,
    T: [const] FillIntArray<LEN>,
{
    fn fill_int_array(self, out: &mut [int; LEN], idx: usize) {
        out[idx] = self.0.value();
        self.1.fill_int_array(out, idx + 1);
    }
}

const fn flat_to_int_array<F: [const] FillIntArray<LEN>, const LEN: usize>(flat: F) -> [int; LEN] {
    let mut arr = [1i32; LEN];
    flat.fill_int_array(&mut arr, 0);
    arr
}

const IS_ZERO_USIZE<const N: usize>: bool = N == 0;
const DEC_USIZE<const N: usize>: usize = N.saturating_sub(1);
const FLAT_LEAVES_USIZE<T: Item>: usize = T::LEAVES as usize;
const GCD_DYNAMIC_SLOTS<FA: Item, FB: Item>: usize = (FA::LEAVES + FB::LEAVES - 1) as usize;

/// Builds a flat `int` tuple of length `LEN` from a fixed-size array slice starting at `idx`.
#[doc(hidden)]
pub const trait IntTupleOfLen<const IS_ZERO: bool, const LEN: usize> {
    type Tuple: [const] Tuple + FlatTuple + [const] Shape + Stride<Self::Tuple>;
    fn from_array_at<const K: usize>(arr: &[int; K], idx: usize) -> Self::Tuple;
}

const impl<const LEN: usize> IntTupleOfLen<true, LEN> for () {
    type Tuple = Nil;

    fn from_array_at<const K: usize>(_: &[int; K], _: usize) -> Nil {
        Nil
    }
}

const impl<const LEN: usize> IntTupleOfLen<false, LEN> for ()
where
    (): [const] IntTupleOfLen<{ IS_ZERO_USIZE::<{ DEC_USIZE::<LEN> }> }, { DEC_USIZE::<LEN> }>,
{
    type Tuple =
        Til<
            int,
            <() as IntTupleOfLen<
                { IS_ZERO_USIZE::<{ DEC_USIZE::<LEN> }> },
                { DEC_USIZE::<LEN> },
            >>::Tuple,
        >;

    fn from_array_at<const K: usize>(arr: &[int; K], idx: usize) -> Self::Tuple {
        Til(
            arr[idx],
            <() as IntTupleOfLen<
                { IS_ZERO_USIZE::<{ DEC_USIZE::<LEN> }> },
                { DEC_USIZE::<LEN> },
            >>::from_array_at(arr, idx + 1),
        )
    }
}

type DynamicIntTuple<const LEN: usize> =
    <() as IntTupleOfLen<{ IS_ZERO_USIZE::<LEN> }, LEN>>::Tuple;

/// Exact `#![no_std]` dynamic GCD walk over remaining positive shape leaf arrays.
///
/// Unused trailing slots (when fewer than `K = M + N - 1` factors are emitted) are padded with
/// `(1, 0)`, preserving `depth == 1`, `size(R)`, symmetry, and pointwise map equivalence.
const fn gcd_dynamic<const M: usize, const N: usize, const K: usize>(
    mut a: [int; M],
    mut b: [int; N],
    mut p_a: int,
    mut p_b: int,
) -> ([int; K], [int; K]) {
    let mut out_s = [1i32; K];
    let mut out_d = [0i32; K];
    let mut out_len = 0usize;

    let mut i = 0usize;
    let mut j = 0usize;

    while i < M && j < N {
        if a[i] == 1 {
            i += 1;
            continue;
        }
        if b[j] == 1 {
            j += 1;
            continue;
        }

        let lcm_pab = lcm(p_a, p_b);
        let r_a = div(lcm_pab, p_a);
        let r_b = div(lcm_pab, p_b);

        if rem(a[i], r_a) == 0 && rem(b[j], r_b) == 0 {
            let qa = div(a[i], r_a);
            let qb = div(b[j], r_b);
            let g = gcd(qa, qb);
            if g != 1 {
                out_s[out_len] = g;
                out_d[out_len] = lcm_pab;
                out_len += 1;
                a[i] = div(qa, g);
                b[j] = div(qb, g);

                let a_last = i + 1 == M;
                let b_last = j + 1 == N;
                if (a[i] == 1 && a_last)
                    || (b[j] == 1 && b_last)
                    || (a[i] > 1 && b[j] > 1 && (a_last || b_last))
                {
                    break;
                }

                let next_p = mul(lcm_pab, g);
                p_a = next_p;
                p_b = next_p;
                continue;
            }
        }

        let a_last = i + 1 == M;
        let b_last = j + 1 == N;
        if a_last && b_last {
            break;
        }

        match (p_a.checked_mul(a[i]), p_b.checked_mul(b[j])) {
            (Some(ea), Some(eb)) => {
                let q_a = rem(eb, ea) == 0;
                let q_b = rem(ea, eb) == 0;
                if q_a || !q_b {
                    p_a = ea;
                    i += 1;
                }
                if q_b || !q_a {
                    p_b = eb;
                    j += 1;
                }
            }
            (Some(ea), None) => {
                if divides_product(ea, p_b, b[j]) {
                    p_a = ea;
                    i += 1;
                } else if a_last || b_last {
                    break;
                } else {
                    panic!("greatest_common_domain running offset overflows int");
                }
            }
            (None, Some(eb)) => {
                if divides_product(eb, p_a, a[i]) {
                    p_b = eb;
                    j += 1;
                } else if a_last || b_last {
                    break;
                } else {
                    panic!("greatest_common_domain running offset overflows int");
                }
            }
            (None, None) => panic!("greatest_common_domain running offset overflows int"),
        }
    }

    (out_s, out_d)
}

/// Compile-time action for one step of the greatest-common-domain walk.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum GcdStepAction {
    /// Emit `(g, lcm_pab)` and terminate (at least one sequence is exhausted).
    EmitAndStop,
    /// Emit `(g, lcm_pab)`, consume both heads (`rem_a == 1 && rem_b == 1`), set `pA = pB = lcm * g`.
    EmitBothDone,
    /// Emit `(g, lcm_pab)`, consume head `A` (`rem_a == 1`), replace head `B` with `rem_b`.
    EmitADone,
    /// Emit `(g, lcm_pab)`, replace head `A` with `rem_a`, consume head `B` (`rem_b == 1`).
    EmitBDone,
    /// Emit `(g, lcm_pab)` with `rem_a > 1 && rem_b > 1` (so next step advances both heads).
    EmitAdvanceBoth,
    /// No match; advance head `A` only (`eA | eB` strictly).
    AdvanceA,
    /// No match; advance head `B` only (`eB | eA` strictly).
    AdvanceB,
    /// No match; advance both heads (`eA == eB` or mutually non-divisible).
    AdvanceBoth,
    /// No match and advancing exhausts at least one sequence; terminate immediately.
    Stop,
    /// At least one head leaf is dynamic; finish the remaining suffix via fixed-array solver.
    DynamicFallback,
    /// Statically overflows `int` when computing a required offset.
    Invalid,
}

const fn gcd_step_action<
    PA: ShapeScalar,
    PB: ShapeScalar,
    AH: ShapeScalar,
    BH: ShapeScalar,
    const AT_EMPTY: bool,
    const BT_EMPTY: bool,
>() -> GcdStepAction {
    let (Some(p_a), Some(p_b), Some(a), Some(b)) = (
        PA::STATIC_VALUE,
        PB::STATIC_VALUE,
        AH::STATIC_VALUE,
        BH::STATIC_VALUE,
    ) else {
        return GcdStepAction::DynamicFallback;
    };
    if p_a <= 0 || p_b <= 0 || a <= 0 || b <= 0 {
        return GcdStepAction::Invalid;
    }

    let Ok(lcm_pab) = checked_lcm(p_a, p_b) else {
        return GcdStepAction::Invalid;
    };
    let r_a = div(lcm_pab, p_a);
    let r_b = div(lcm_pab, p_b);

    if rem(a, r_a) == 0 && rem(b, r_b) == 0 {
        let qa = div(a, r_a);
        let qb = div(b, r_b);
        let g = gcd(qa, qb);
        if g != 1 {
            let rem_a = div(qa, g);
            let rem_b = div(qb, g);
            if (rem_a == 1 && AT_EMPTY)
                || (rem_b == 1 && BT_EMPTY)
                || (rem_a > 1 && rem_b > 1 && (AT_EMPTY || BT_EMPTY))
            {
                return GcdStepAction::EmitAndStop;
            }
            if rem_a == 1 && rem_b == 1 {
                return if lcm_pab.checked_mul(g).is_some() {
                    GcdStepAction::EmitBothDone
                } else {
                    GcdStepAction::Invalid
                };
            }
            if rem_a == 1 {
                return if lcm_pab.checked_mul(g).is_some() {
                    GcdStepAction::EmitADone
                } else {
                    GcdStepAction::Invalid
                };
            }
            if rem_b == 1 {
                return if lcm_pab.checked_mul(g).is_some() {
                    GcdStepAction::EmitBDone
                } else {
                    GcdStepAction::Invalid
                };
            }
            return if p_a.checked_mul(a).is_some() && p_b.checked_mul(b).is_some() {
                GcdStepAction::EmitAdvanceBoth
            } else {
                GcdStepAction::Invalid
            };
        }
    }

    if AT_EMPTY && BT_EMPTY {
        return GcdStepAction::Stop;
    }

    match (p_a.checked_mul(a), p_b.checked_mul(b)) {
        (Some(ea), Some(eb)) => {
            let q_a = rem(eb, ea) == 0;
            let q_b = rem(ea, eb) == 0;
            match (q_a, q_b) {
                (true, false) => {
                    if AT_EMPTY {
                        GcdStepAction::Stop
                    } else {
                        GcdStepAction::AdvanceA
                    }
                }
                (false, true) => {
                    if BT_EMPTY {
                        GcdStepAction::Stop
                    } else {
                        GcdStepAction::AdvanceB
                    }
                }
                _ => {
                    if AT_EMPTY || BT_EMPTY {
                        GcdStepAction::Stop
                    } else {
                        GcdStepAction::AdvanceBoth
                    }
                }
            }
        }
        (Some(ea), None) => {
            if divides_product(ea, p_b, b) {
                if AT_EMPTY {
                    GcdStepAction::Stop
                } else {
                    GcdStepAction::AdvanceA
                }
            } else if AT_EMPTY || BT_EMPTY {
                GcdStepAction::Stop
            } else {
                GcdStepAction::Invalid
            }
        }
        (None, Some(eb)) => {
            if divides_product(eb, p_a, a) {
                if BT_EMPTY {
                    GcdStepAction::Stop
                } else {
                    GcdStepAction::AdvanceB
                }
            } else if AT_EMPTY || BT_EMPTY {
                GcdStepAction::Stop
            } else {
                GcdStepAction::Invalid
            }
        }
        (None, None) => GcdStepAction::Invalid,
    }
}

const TAIL_IS_EMPTY<T: Item>: bool = T::LEAVES == 0;

const GCD_STEP_ACTION<
    PA: ShapeScalar,
    PB: ShapeScalar,
    AH: ShapeScalar,
    BH: ShapeScalar,
    AT: Item,
    BT: Item,
>: GcdStepAction = gcd_step_action::<PA, PB, AH, BH, { TAIL_IS_EMPTY::<AT> }, { TAIL_IS_EMPTY::<BT> }>();

const fn static_lcm_or_one<PA: ShapeScalar, PB: ShapeScalar>() -> int {
    let (Some(p_a), Some(p_b)) = (PA::STATIC_VALUE, PB::STATIC_VALUE) else {
        return 1;
    };
    match checked_lcm(p_a, p_b) {
        Ok(l) if l > 0 => l,
        _ => 1,
    }
}

const fn static_gcd_parts<PA: ShapeScalar, PB: ShapeScalar, AH: ShapeScalar, BH: ShapeScalar>()
-> (int, int, int, int) {
    let (Some(p_a), Some(p_b), Some(a), Some(b)) = (
        PA::STATIC_VALUE,
        PB::STATIC_VALUE,
        AH::STATIC_VALUE,
        BH::STATIC_VALUE,
    ) else {
        return (1, 1, 1, 1);
    };
    if p_a <= 0 || p_b <= 0 || a <= 0 || b <= 0 {
        return (1, 1, 1, 1);
    }
    let Ok(l) = checked_lcm(p_a, p_b) else {
        return (1, 1, 1, 1);
    };
    let r_a = div(l, p_a);
    let r_b = div(l, p_b);
    if rem(a, r_a) != 0 || rem(b, r_b) != 0 {
        return (1, 1, 1, 1);
    }
    let qa = div(a, r_a);
    let qb = div(b, r_b);
    let g = gcd(qa, qb);
    if g <= 0 {
        return (1, 1, 1, 1);
    }
    let rem_a = div(qa, g);
    let rem_b = div(qb, g);
    let next_p = match l.checked_mul(g) {
        Some(v) => v,
        None => 1,
    };
    (g, rem_a, rem_b, next_p)
}

const fn static_end_or_one<P: ShapeScalar, S: ShapeScalar>() -> int {
    let (Some(p), Some(s)) = (P::STATIC_VALUE, S::STATIC_VALUE) else {
        return 1;
    };
    if p <= 0 || s <= 0 {
        return 1;
    }
    match p.checked_mul(s) {
        Some(v) => v,
        None => 1,
    }
}

const STATIC_LCM<PA: ShapeScalar, PB: ShapeScalar>: int = static_lcm_or_one::<PA, PB>();

const STATIC_GCD_FACTOR<
    PA: ShapeScalar,
    PB: ShapeScalar,
    AH: ShapeScalar,
    BH: ShapeScalar,
>: int = static_gcd_parts::<PA, PB, AH, BH>().0;

const STATIC_REM_A<
    PA: ShapeScalar,
    PB: ShapeScalar,
    AH: ShapeScalar,
    BH: ShapeScalar,
>: int = static_gcd_parts::<PA, PB, AH, BH>().1;

const STATIC_REM_B<
    PA: ShapeScalar,
    PB: ShapeScalar,
    AH: ShapeScalar,
    BH: ShapeScalar,
>: int = static_gcd_parts::<PA, PB, AH, BH>().2;

const STATIC_NEXT_P<
    PA: ShapeScalar,
    PB: ShapeScalar,
    AH: ShapeScalar,
    BH: ShapeScalar,
>: int = static_gcd_parts::<PA, PB, AH, BH>().3;

const STATIC_END<P: ShapeScalar, S: ShapeScalar>: int = static_end_or_one::<P, S>();

/// Walks two unit-stripped flat positive shape tuples collecting common `(factor, offset)` pairs.
#[doc(hidden)]
pub const trait GcdWalk<PA: ShapeScalar, PB: ShapeScalar, FB: FlatTuple>: FlatTuple {
    type Shape: [const] Tuple + FlatTuple + [const] Shape;
    type Stride: FlatTuple + Stride<Self::Shape>;
    fn gcd_walk(self, fb: FB, p_a: PA, p_b: PB) -> Layout<Self::Shape, Self::Stride>;
}

/// Executes one compile-time dispatched [`GcdStepAction`] step of [`GcdWalk`].
#[doc(hidden)]
pub const trait GcdWalkStep<
    PA: ShapeScalar,
    PB: ShapeScalar,
    FB: FlatTuple,
    const ACTION: GcdStepAction,
>: FlatTuple
{
    type Shape: [const] Tuple + FlatTuple + [const] Shape;
    type Stride: FlatTuple + Stride<Self::Shape>;
    fn gcd_step(self, fb: FB, p_a: PA, p_b: PB) -> Layout<Self::Shape, Self::Stride>;
}

// Base cases: either flat tuple exhausted
const impl<PA: ShapeScalar, PB: ShapeScalar, FB: FlatTuple> GcdWalk<PA, PB, FB> for Nil {
    type Shape = Nil;
    type Stride = Nil;

    fn gcd_walk(self, _: FB, _: PA, _: PB) -> Layout<Nil, Nil> {
        Layout::new(Nil, Nil)
    }
}

const impl<PA: ShapeScalar, PB: ShapeScalar, AH: ShapeScalar, AT: FlatTuple> GcdWalk<PA, PB, Nil>
    for Til<AH, AT>
{
    type Shape = Nil;
    type Stride = Nil;

    fn gcd_walk(self, _: Nil, _: PA, _: PB) -> Layout<Nil, Nil> {
        Layout::new(Nil, Nil)
    }
}

// Recursive dispatch on two non-empty flat tuples
const impl<PA, PB, AH, AT, BH, BT> GcdWalk<PA, PB, Til<BH, BT>> for Til<AH, AT>
where
    PA: ShapeScalar,
    PB: ShapeScalar,
    AH: ShapeScalar,
    AT: FlatTuple,
    BH: ShapeScalar,
    BT: FlatTuple,
    Self: [const] GcdWalkStep<PA, PB, Til<BH, BT>, { GCD_STEP_ACTION::<PA, PB, AH, BH, AT, BT> }>,
{
    type Shape = <Self as GcdWalkStep<
        PA,
        PB,
        Til<BH, BT>,
        { GCD_STEP_ACTION::<PA, PB, AH, BH, AT, BT> },
    >>::Shape;
    type Stride = <Self as GcdWalkStep<
        PA,
        PB,
        Til<BH, BT>,
        { GCD_STEP_ACTION::<PA, PB, AH, BH, AT, BT> },
    >>::Stride;

    fn gcd_walk(self, fb: Til<BH, BT>, p_a: PA, p_b: PB) -> Layout<Self::Shape, Self::Stride> {
        self.gcd_step(fb, p_a, p_b)
    }
}

// Step: Stop
const impl<PA, PB, AH, AT, BH, BT> GcdWalkStep<PA, PB, Til<BH, BT>, { GcdStepAction::Stop }>
    for Til<AH, AT>
where
    PA: ShapeScalar,
    PB: ShapeScalar,
    AH: ShapeScalar,
    AT: FlatTuple,
    BH: ShapeScalar,
    BT: FlatTuple,
{
    type Shape = Nil;
    type Stride = Nil;

    fn gcd_step(self, _: Til<BH, BT>, _: PA, _: PB) -> Layout<Nil, Nil> {
        Layout::new(Nil, Nil)
    }
}

// Step: EmitAndStop
const impl<PA, PB, AH, AT, BH, BT> GcdWalkStep<PA, PB, Til<BH, BT>, { GcdStepAction::EmitAndStop }>
    for Til<AH, AT>
where
    PA: ShapeScalar,
    PB: ShapeScalar,
    AH: ShapeScalar,
    AT: FlatTuple,
    BH: ShapeScalar,
    BT: FlatTuple,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_GCD_FACTOR::<PA, PB, AH, BH> }> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_LCM::<PA, PB> }> }>,
{
    type Shape = Til<Int<{ STATIC_GCD_FACTOR::<PA, PB, AH, BH> }>, Nil>;
    type Stride = Til<Int<{ STATIC_LCM::<PA, PB> }>, Nil>;

    fn gcd_step(self, _: Til<BH, BT>, _: PA, _: PB) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(Til(Int, Nil), Til(Int, Nil))
    }
}

// Step: EmitBothDone
const impl<PA, PB, AH, AT, BH, BT> GcdWalkStep<PA, PB, Til<BH, BT>, { GcdStepAction::EmitBothDone }>
    for Til<AH, AT>
where
    PA: ShapeScalar,
    PB: ShapeScalar,
    AH: ShapeScalar,
    BH: ShapeScalar,
    BT: FlatTuple,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_GCD_FACTOR::<PA, PB, AH, BH> }> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_LCM::<PA, PB> }> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_NEXT_P::<PA, PB, AH, BH> }> }>,
    AT: FlatTuple
        + [const] GcdWalk<
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            BT,
        >,
{
    type Shape = Til<
        Int<{ STATIC_GCD_FACTOR::<PA, PB, AH, BH> }>,
        <AT as GcdWalk<
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            BT,
        >>::Shape,
    >;
    type Stride = Til<
        Int<{ STATIC_LCM::<PA, PB> }>,
        <AT as GcdWalk<
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            BT,
        >>::Stride,
    >;

    fn gcd_step(self, fb: Til<BH, BT>, _: PA, _: PB) -> Layout<Self::Shape, Self::Stride> {
        let tail = self.1.gcd_walk(fb.1, Int, Int);
        Layout::new(Til(Int, tail.shape), Til(Int, tail.stride))
    }
}

// Step: EmitADone
const impl<PA, PB, AH, AT, BH, BT> GcdWalkStep<PA, PB, Til<BH, BT>, { GcdStepAction::EmitADone }>
    for Til<AH, AT>
where
    PA: ShapeScalar,
    PB: ShapeScalar,
    AH: ShapeScalar,
    BH: ShapeScalar,
    BT: FlatTuple,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_GCD_FACTOR::<PA, PB, AH, BH> }> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_LCM::<PA, PB> }> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_NEXT_P::<PA, PB, AH, BH> }> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_REM_B::<PA, PB, AH, BH> }> }>,
    AT: FlatTuple
        + [const] GcdWalk<
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            Til<Int<{ STATIC_REM_B::<PA, PB, AH, BH> }>, BT>,
        >,
{
    type Shape = Til<
        Int<{ STATIC_GCD_FACTOR::<PA, PB, AH, BH> }>,
        <AT as GcdWalk<
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            Til<Int<{ STATIC_REM_B::<PA, PB, AH, BH> }>, BT>,
        >>::Shape,
    >;
    type Stride = Til<
        Int<{ STATIC_LCM::<PA, PB> }>,
        <AT as GcdWalk<
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            Til<Int<{ STATIC_REM_B::<PA, PB, AH, BH> }>, BT>,
        >>::Stride,
    >;

    fn gcd_step(self, fb: Til<BH, BT>, _: PA, _: PB) -> Layout<Self::Shape, Self::Stride> {
        let tail = self.1.gcd_walk(Til(Int, fb.1), Int, Int);
        Layout::new(Til(Int, tail.shape), Til(Int, tail.stride))
    }
}

// Step: EmitBDone
const impl<PA, PB, AH, AT, BH, BT> GcdWalkStep<PA, PB, Til<BH, BT>, { GcdStepAction::EmitBDone }>
    for Til<AH, AT>
where
    PA: ShapeScalar,
    PB: ShapeScalar,
    AH: ShapeScalar,
    AT: FlatTuple,
    BH: ShapeScalar,
    BT: FlatTuple,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_GCD_FACTOR::<PA, PB, AH, BH> }> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_LCM::<PA, PB> }> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_NEXT_P::<PA, PB, AH, BH> }> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_REM_A::<PA, PB, AH, BH> }> }>,
    Til<Int<{ STATIC_REM_A::<PA, PB, AH, BH> }>, AT>: [const] GcdWalk<
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            BT,
        >,
{
    type Shape = Til<
        Int<{ STATIC_GCD_FACTOR::<PA, PB, AH, BH> }>,
        <Til<Int<{ STATIC_REM_A::<PA, PB, AH, BH> }>, AT> as GcdWalk<
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            BT,
        >>::Shape,
    >;
    type Stride = Til<
        Int<{ STATIC_LCM::<PA, PB> }>,
        <Til<Int<{ STATIC_REM_A::<PA, PB, AH, BH> }>, AT> as GcdWalk<
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            Int<{ STATIC_NEXT_P::<PA, PB, AH, BH> }>,
            BT,
        >>::Stride,
    >;

    fn gcd_step(self, fb: Til<BH, BT>, _: PA, _: PB) -> Layout<Self::Shape, Self::Stride> {
        let tail = Til(Int, self.1).gcd_walk(fb.1, Int, Int);
        Layout::new(Til(Int, tail.shape), Til(Int, tail.stride))
    }
}

// Step: EmitAdvanceBoth (`rem_a > 1 && rem_b > 1` -> next iteration always advances both)
const impl<PA, PB, AH, AT, BH, BT>
    GcdWalkStep<PA, PB, Til<BH, BT>, { GcdStepAction::EmitAdvanceBoth }> for Til<AH, AT>
where
    PA: ShapeScalar,
    PB: ShapeScalar,
    AH: ShapeScalar,
    BH: ShapeScalar,
    BT: FlatTuple,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_GCD_FACTOR::<PA, PB, AH, BH> }> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_LCM::<PA, PB> }> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_END::<PA, AH> }> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_END::<PB, BH> }> }>,
    AT: FlatTuple
        + [const] GcdWalk<Int<{ STATIC_END::<PA, AH> }>, Int<{ STATIC_END::<PB, BH> }>, BT>,
{
    type Shape = Til<
        Int<{ STATIC_GCD_FACTOR::<PA, PB, AH, BH> }>,
        <AT as GcdWalk<Int<{ STATIC_END::<PA, AH> }>, Int<{ STATIC_END::<PB, BH> }>, BT>>::Shape,
    >;
    type Stride = Til<
        Int<{ STATIC_LCM::<PA, PB> }>,
        <AT as GcdWalk<Int<{ STATIC_END::<PA, AH> }>, Int<{ STATIC_END::<PB, BH> }>, BT>>::Stride,
    >;

    fn gcd_step(self, fb: Til<BH, BT>, _: PA, _: PB) -> Layout<Self::Shape, Self::Stride> {
        let tail = self.1.gcd_walk(fb.1, Int, Int);
        Layout::new(Til(Int, tail.shape), Til(Int, tail.stride))
    }
}

// Step: AdvanceA
const impl<PA, PB, AH, AT, BH, BT> GcdWalkStep<PA, PB, Til<BH, BT>, { GcdStepAction::AdvanceA }>
    for Til<AH, AT>
where
    PA: ShapeScalar,
    PB: ShapeScalar,
    AH: ShapeScalar,
    BH: ShapeScalar,
    BT: FlatTuple,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_END::<PA, AH> }> }>,
    AT: FlatTuple + [const] GcdWalk<Int<{ STATIC_END::<PA, AH> }>, PB, Til<BH, BT>>,
{
    type Shape = <AT as GcdWalk<Int<{ STATIC_END::<PA, AH> }>, PB, Til<BH, BT>>>::Shape;
    type Stride = <AT as GcdWalk<Int<{ STATIC_END::<PA, AH> }>, PB, Til<BH, BT>>>::Stride;

    fn gcd_step(self, fb: Til<BH, BT>, _: PA, p_b: PB) -> Layout<Self::Shape, Self::Stride> {
        self.1.gcd_walk(fb, Int, p_b)
    }
}

// Step: AdvanceB
const impl<PA, PB, AH, AT, BH, BT> GcdWalkStep<PA, PB, Til<BH, BT>, { GcdStepAction::AdvanceB }>
    for Til<AH, AT>
where
    PA: ShapeScalar,
    PB: ShapeScalar,
    AH: ShapeScalar,
    AT: FlatTuple,
    BH: ShapeScalar,
    BT: FlatTuple,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_END::<PB, BH> }> }>,
    Self: [const] GcdWalk<PA, Int<{ STATIC_END::<PB, BH> }>, BT>,
{
    type Shape = <Self as GcdWalk<PA, Int<{ STATIC_END::<PB, BH> }>, BT>>::Shape;
    type Stride = <Self as GcdWalk<PA, Int<{ STATIC_END::<PB, BH> }>, BT>>::Stride;

    fn gcd_step(self, fb: Til<BH, BT>, p_a: PA, _: PB) -> Layout<Self::Shape, Self::Stride> {
        self.gcd_walk(fb.1, p_a, Int)
    }
}

// Step: AdvanceBoth
const impl<PA, PB, AH, AT, BH, BT> GcdWalkStep<PA, PB, Til<BH, BT>, { GcdStepAction::AdvanceBoth }>
    for Til<AH, AT>
where
    PA: ShapeScalar,
    PB: ShapeScalar,
    AH: ShapeScalar,
    BH: ShapeScalar,
    BT: FlatTuple,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_END::<PA, AH> }> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_END::<PB, BH> }> }>,
    AT: FlatTuple
        + [const] GcdWalk<Int<{ STATIC_END::<PA, AH> }>, Int<{ STATIC_END::<PB, BH> }>, BT>,
{
    type Shape =
        <AT as GcdWalk<Int<{ STATIC_END::<PA, AH> }>, Int<{ STATIC_END::<PB, BH> }>, BT>>::Shape;
    type Stride =
        <AT as GcdWalk<Int<{ STATIC_END::<PA, AH> }>, Int<{ STATIC_END::<PB, BH> }>, BT>>::Stride;

    fn gcd_step(self, fb: Til<BH, BT>, _: PA, _: PB) -> Layout<Self::Shape, Self::Stride> {
        self.1.gcd_walk(fb.1, Int, Int)
    }
}

// Step: DynamicFallback (fixed-size array solver for remaining suffix)
const impl<PA, PB, AH, AT, BH, BT>
    GcdWalkStep<PA, PB, Til<BH, BT>, { GcdStepAction::DynamicFallback }> for Til<AH, AT>
where
    PA: [const] ShapeScalar,
    PB: [const] ShapeScalar,
    AH: [const] ShapeScalar,
    AT: FlatTuple,
    BH: [const] ShapeScalar,
    BT: FlatTuple,
    Self: [const] FillIntArray<{ FLAT_LEAVES_USIZE::<Self> }>,
    Til<BH, BT>: [const] FillIntArray<{ FLAT_LEAVES_USIZE::<Til<BH, BT>> }>,
    (): [const] IntTupleOfLen<
            { IS_ZERO_USIZE::<{ GCD_DYNAMIC_SLOTS::<Self, Til<BH, BT>> }> },
            { GCD_DYNAMIC_SLOTS::<Self, Til<BH, BT>> },
        >,
{
    type Shape = DynamicIntTuple<{ GCD_DYNAMIC_SLOTS::<Self, Til<BH, BT>> }>;
    type Stride = DynamicIntTuple<{ GCD_DYNAMIC_SLOTS::<Self, Til<BH, BT>> }>;

    fn gcd_step(self, fb: Til<BH, BT>, p_a: PA, p_b: PB) -> Layout<Self::Shape, Self::Stride> {
        let a_arr = flat_to_int_array::<Self, { FLAT_LEAVES_USIZE::<Self> }>(self);
        let b_arr = flat_to_int_array::<Til<BH, BT>, { FLAT_LEAVES_USIZE::<Til<BH, BT>> }>(fb);
        let (out_s, out_d) = gcd_dynamic::<
            { FLAT_LEAVES_USIZE::<Self> },
            { FLAT_LEAVES_USIZE::<Til<BH, BT>> },
            { GCD_DYNAMIC_SLOTS::<Self, Til<BH, BT>> },
        >(a_arr, b_arr, p_a.value(), p_b.value());
        let shape = <() as IntTupleOfLen<
            { IS_ZERO_USIZE::<{ GCD_DYNAMIC_SLOTS::<Self, Til<BH, BT>> }> },
            { GCD_DYNAMIC_SLOTS::<Self, Til<BH, BT>> },
        >>::from_array_at(&out_s, 0);
        let stride = <() as IntTupleOfLen<
            { IS_ZERO_USIZE::<{ GCD_DYNAMIC_SLOTS::<Self, Til<BH, BT>> }> },
            { GCD_DYNAMIC_SLOTS::<Self, Til<BH, BT>> },
        >>::from_array_at(&out_d, 0);
        Layout::new(shape, stride)
    }
}

/// Converts an empty factor list `Layout<Nil, Nil>` into the canonical depth-1 singleton `Layout((1,), (0,))`.
#[doc(hidden)]
pub const trait FinalizeGcd: Copy {
    type Shape: [const] Tuple + FlatTuple + [const] Shape;
    type Stride: FlatTuple + Stride<Self::Shape>;
    fn finalize_gcd(self) -> Layout<Self::Shape, Self::Stride>;
}

const impl FinalizeGcd for Layout<Nil, Nil> {
    type Shape = Til<Int<1>, Nil>;
    type Stride = Til<Int<0>, Nil>;

    fn finalize_gcd(self) -> Layout<Self::Shape, Self::Stride> {
        Layout::new(Til(Int::<1>, Nil), Til(Int::<0>, Nil))
    }
}

const impl<SH, ST, DH, DT> FinalizeGcd for Layout<Til<SH, ST>, Til<DH, DT>>
where
    SH: [const] ShapeScalar,
    ST: [const] Tuple + FlatTuple + [const] Shape,
    DH: [const] ShapeScalar + Stride<SH>,
    DT: FlatTuple + Stride<ST>,
{
    type Shape = Til<SH, ST>;
    type Stride = Til<DH, DT>;

    fn finalize_gcd(self) -> Self {
        self
    }
}

/// Computes the greatest common domain [`Layout`] of two shapes or layouts.
pub const trait GreatestCommonDomain<B>: Copy {
    /// Depth-1 output [`Layout`] shape type.
    type Shape: [const] Tuple + FlatTuple + [const] Shape;
    /// Depth-1 output [`Layout`] stride type.
    type Stride: FlatTuple + Stride<Self::Shape>;

    /// Computes the greatest common domain, panicking on dynamic non-positive extent or `int` overflow.
    fn greatest_common_domain(self, b: B) -> Layout<Self::Shape, Self::Stride>;
}

type StrippedFlatShape<X> =
    <<<X as IntoGcdShape>::Shape as Flatten>::Output as StripStaticOnes>::Output;

type WalkedGcdLayout<A, B> = Layout<
    <StrippedFlatShape<A> as GcdWalk<Int<1>, Int<1>, StrippedFlatShape<B>>>::Shape,
    <StrippedFlatShape<A> as GcdWalk<Int<1>, Int<1>, StrippedFlatShape<B>>>::Stride,
>;

const impl<A, B> GreatestCommonDomain<B> for A
where
    A: [const] IntoGcdShape,
    B: [const] IntoGcdShape,
    A::Shape: [const] Flatten,
    B::Shape: [const] Flatten,
    <A::Shape as Flatten>::Output: [const] AssertPositiveShape + [const] StripStaticOnes,
    <B::Shape as Flatten>::Output: [const] AssertPositiveShape + [const] StripStaticOnes,
    StrippedFlatShape<A>: [const] GcdWalk<Int<1>, Int<1>, StrippedFlatShape<B>>,
    WalkedGcdLayout<A, B>: [const] FinalizeGcd,
{
    type Shape = <WalkedGcdLayout<A, B> as FinalizeGcd>::Shape;
    type Stride = <WalkedGcdLayout<A, B> as FinalizeGcd>::Stride;

    fn greatest_common_domain(self, b: B) -> Layout<Self::Shape, Self::Stride> {
        let flat_a = self.into_gcd_shape().flatten();
        let flat_b = b.into_gcd_shape().flatten();
        flat_a.assert_positive_shape();
        flat_b.assert_positive_shape();
        let stripped_a = flat_a.strip_static_ones();
        let stripped_b = flat_b.strip_static_ones();
        let walked = stripped_a.gcd_walk(stripped_b, Int::<1>, Int::<1>);
        walked.finalize_gcd()
    }
}

impl<S, D> Layout<S, D> {
    /// Computes the greatest common domain [`Layout`] shared by `self.shape` and `other`'s shape.
    pub const fn greatest_common_domain<B>(
        self,
        other: B,
    ) -> Layout<<Self as GreatestCommonDomain<B>>::Shape, <Self as GreatestCommonDomain<B>>::Stride>
    where
        Self: [const] GreatestCommonDomain<B>,
    {
        GreatestCommonDomain::greatest_common_domain(self, other)
    }
}

/// Computes the greatest common domain [`Layout`] of two shapes or layouts.
pub const fn greatest_common_domain<A, B>(
    a: A,
    b: B,
) -> Layout<<A as GreatestCommonDomain<B>>::Shape, <A as GreatestCommonDomain<B>>::Stride>
where
    A: [const] GreatestCommonDomain<B>,
{
    a.greatest_common_domain(b)
}
