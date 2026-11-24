//! Unary and binary arithmetic for static ([`Int<N>`]) and dynamic ([`int`]) integers.
//!
//! Implements checked arithmetic in `int`/`uint` width without wider intermediate types,
//! preserving compile-time static types when operands are static or satisfy absorbing
//! algebraic identities.
//!
//! # Main Interfaces
//!
//! - **Unary operations**: [`neg`] (also unary `-` for [`Int<N>`]), [`abs`], and traits
//!   [`ArithNeg`], [`ArithAbs`].
//! - **Binary functions**: [`add`], [`sub`], [`mul`], [`div`], [`rem`], [`ceil_div`],
//!   [`floor_div`], [`floor_rem`], [`floor_divmod`], [`gcd`], [`lcm`], plus `const`
//!   `core::ops` operator overloads (`+`, `-`, `*`, `/`, `%`) whenever at least one
//!   operand is [`Int<N>`].
//! - **Result type aliases**: [`Sum`], [`Difference`], [`Product`], [`Quotient`],
//!   [`Remainder`], [`CeilQuotient`], [`FloorQuotient`], [`FloorRemainder`], [`Gcd`],
//!   [`Lcm`], [`ArithOpResult`], and operation tag [`ArithOp`].
//! - **Generic trait bounds**: [`ArithAdd`], [`ArithSub`], [`ArithMul`], [`ArithDiv`],
//!   [`ArithRem`], [`ArithCeilDiv`], [`ArithFloorDiv`], [`ArithFloorRem`],
//!   [`ArithFloorDivMod`], [`ArithGcd`], [`ArithLcm`].
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::{Int, ceil_div, floor_divmod, gcd, lcm};
//!
//! const P: Int<6> = Int::<2> * Int::<3>;
//! assert_eq!(P, Int::<6>);
//! assert_eq!(ceil_div(Int::<7>, Int::<3>), Int::<3>);
//! assert_eq!(floor_divmod(-7i32, Int::<3>), (-3i32, 2i32));
//! assert_eq!(gcd(Int::<-12>, Int::<18>), Int::<6>);
//! assert_eq!(lcm(Int::<0>, i32::MIN), Int::<0>);
//! ```
//!
//! # Notes
//!
//! - Invalid static results (overflow, division by zero, or `neg`/`abs` of `int::MIN`)
//!   fail at compile time (`E0080`); invalid dynamic results panic in both debug and
//!   release builds.
//! - Division/remainder by static `Int<0>` fails at compile time even for dynamic dividends.
//! - Absorbing identities (`0 * x`, `x * 0`, `x % 1`, `floor_rem(x, 1)`, `lcm(0, x)`,
//!   `gcd(±1, x)`) preserve static `Int<0>` or `Int<1>`; dividing static `Int<0>` by a
//!   dynamic divisor preserves `Int<0>` while checking `b != 0` at runtime.

use core::marker::ConstParamTy;

use crate::{NoodleCuteError, NoodleCuteResult, Result, integer::uint};

use super::{Int, Integer, int, uint_to_int};

const fn checked_neg(a: int) -> Result<int> {
    a.checked_neg()
        .ok_or(NoodleCuteError::SignedMinValueOverflow)
}

const fn checked_abs(a: int) -> Result<int> {
    a.checked_abs()
        .ok_or(NoodleCuteError::SignedMinValueOverflow)
}

const STATIC_NEG_RESULT<const N: int>: int = checked_neg(N).v();

const STATIC_ABS_RESULT<const N: int>: int = checked_abs(N).v();

/// Unary negation for CuTe integers (`-x == Int<0> - x`).
pub const trait ArithNeg: [const] Integer {
    /// Negated output type ([`Int<R>`] for static operands, [`int`] for dynamic operands).
    type Output: [const] Integer;
    /// Evaluates `-self`, panicking on `int::MIN`.
    fn neg(self) -> Self::Output;
}

const impl<const N: int> ArithNeg for Int<N> {
    type Output = Int<{ STATIC_NEG_RESULT::<N> }>;
    fn neg(self) -> Self::Output {
        Int
    }
}

const impl ArithNeg for int {
    type Output = int;
    fn neg(self) -> Self::Output {
        checked_neg(self).v()
    }
}

const impl<const N: int> core::ops::Neg for Int<N> {
    type Output = Int<{ STATIC_NEG_RESULT::<N> }>;
    fn neg(self) -> Self::Output {
        Int
    }
}

/// Negates a CuTe integer (`-i`), preserving static vs dynamic representation.
///
/// Fails at compile time for `Int<{ int::MIN }>` and panics at runtime for `int::MIN`.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, neg};
///
/// assert_eq!(neg(Int::<5>), Int::<-5>);
/// assert_eq!(-Int::<-5>, Int::<5>);
/// assert_eq!(neg(12i32), -12i32);
/// ```
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, neg};
/// let _ = neg(Int::<-2_147_483_648>);
/// ```
pub const fn neg<I: [const] ArithNeg>(i: I) -> <I as ArithNeg>::Output {
    i.neg()
}

/// Absolute value preserving static vs dynamic representation.
pub const trait ArithAbs: [const] Integer {
    /// Output type ([`Int<R>`] for static operands, [`int`] for dynamic operands).
    type Output: [const] Integer;
    /// Evaluates `|self|`, panicking on `int::MIN`.
    fn abs(self) -> Self::Output;
}

const impl<const N: int> ArithAbs for Int<N> {
    type Output = Int<{ STATIC_ABS_RESULT::<N> }>;
    fn abs(self) -> Self::Output {
        Int
    }
}

const impl ArithAbs for int {
    type Output = int;
    fn abs(self) -> Self::Output {
        checked_abs(self).v()
    }
}

/// Returns the absolute value of a CuTe integer.
///
/// Static inputs produce [`Int<R>`]; dynamic inputs produce [`int`].
/// Fails at compile time for `Int<{ int::MIN }>` and panics at runtime for `int::MIN`.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, abs};
///
/// assert_eq!(abs(Int::<-12>), Int::<12>);
/// assert_eq!(abs(-12i32), 12i32);
/// ```
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, abs};
/// let _ = abs(Int::<-2_147_483_648>);
/// ```
pub const fn abs<I: [const] ArithAbs>(i: I) -> I::Output {
    <I as ArithAbs>::abs(i)
}

/// Binary integer operations supported by [`ArithOpResult`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum ArithOp {
    /// Addition (`+`).
    Add,
    /// Subtraction (`-`).
    Sub,
    /// Multiplication (`*`).
    Mul,
    /// Truncating integer division (`/`).
    Div,
    /// Truncating integer remainder (`%`).
    Rem,
    /// Ceiling integer division (`ceil(a / b)`).
    CeilDiv,
    /// Floor integer division (`floor(a / b)`).
    FloorDiv,
    /// Floor integer remainder (`a - floor(a / b) * b`).
    FloorRem,
    /// Non-negative greatest common divisor (`gcd(|a|, |b|)`).
    Gcd,
    /// Non-negative least common multiple (`lcm(|a|, |b|)`).
    Lcm,
}

/// Dispatch strategy selected from operand staticness and algebraic absorbing rules:
/// - `Static`: compile-time `Int<R>` (both static, div/rem by static 0, or absorbing identity).
/// - `StaticCheckedDiv`: static `Int<0>` with a runtime `b != 0` check (0 / dyn, etc.).
/// - `Dynamic`: checked evaluation yielding primitive `int`.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ArithDispatch {
    Static = 0,
    StaticCheckedDiv = 1,
    Dynamic = 2,
}

const fn checked_ceil_div(a: int, b: int) -> Result<int> {
    if b == 0 {
        return Err(NoodleCuteError::DivisionByZero);
    }
    match (a.checked_div(b), a.checked_rem(b)) {
        (Some(q), Some(r)) => {
            if r != 0 && ((r < 0) == (b < 0)) {
                Ok(q + 1)
            } else {
                Ok(q)
            }
        }
        _ => Err(NoodleCuteError::ArithmeticOverflow),
    }
}

const fn checked_floor_divmod(a: int, b: int) -> Result<(int, int)> {
    if b == 0 {
        return Err(NoodleCuteError::DivisionByZero);
    }
    match (a.checked_div(b), a.checked_rem(b)) {
        (Some(q), Some(r)) => {
            if r != 0 && ((r < 0) != (b < 0)) {
                Ok((q - 1, r + b))
            } else {
                Ok((q, r))
            }
        }
        _ => Err(NoodleCuteError::ArithmeticOverflow),
    }
}

const fn checked_gcd(a: int, b: int) -> Result<int> {
    let mut ua = a.unsigned_abs();
    let mut ub = b.unsigned_abs();
    while ub != 0 {
        let r = ua % ub;
        ua = ub;
        ub = r;
    }
    uint_to_int(ua)
}

pub(crate) const fn checked_lcm(a: int, b: int) -> Result<int> {
    if a == 0 || b == 0 {
        return Ok(0);
    }
    let g = checked_gcd(a, b)?;
    let q = a.unsigned_abs() / (g as uint);
    let Some(prod) = q.checked_mul(b.unsigned_abs()) else {
        return Err(NoodleCuteError::ArithmeticOverflow);
    };
    uint_to_int(prod)
}

/// Returns `true` iff `l * s == r` mathematically without overflowing `int`.
pub(crate) const fn integer_product_eq(l: int, s: int, r: int) -> bool {
    matches!(l.checked_mul(s), Some(p) if p == r)
}

/// Returns `true` iff `divisor` divides `p * factor` (for positive inputs) without wider types.
pub(crate) const fn divides_product(divisor: int, p: int, factor: int) -> bool {
    let g = gcd(divisor, p);
    rem(factor, div(divisor, g)) == 0
}

const fn checked_op(a: int, b: int, op: ArithOp) -> Result<int> {
    match op {
        ArithOp::Add => a.checked_add(b).ok_or(NoodleCuteError::ArithmeticOverflow),
        ArithOp::Sub => a.checked_sub(b).ok_or(NoodleCuteError::ArithmeticOverflow),
        ArithOp::Mul => a.checked_mul(b).ok_or(NoodleCuteError::ArithmeticOverflow),
        ArithOp::Div => {
            if b == 0 {
                Err(NoodleCuteError::DivisionByZero)
            } else {
                a.checked_div(b).ok_or(NoodleCuteError::ArithmeticOverflow)
            }
        }
        ArithOp::Rem => {
            if b == 0 {
                Err(NoodleCuteError::DivisionByZero)
            } else {
                a.checked_rem(b).ok_or(NoodleCuteError::ArithmeticOverflow)
            }
        }
        ArithOp::CeilDiv => checked_ceil_div(a, b),
        ArithOp::FloorDiv => match checked_floor_divmod(a, b) {
            Ok((q, _)) => Ok(q),
            Err(e) => Err(e),
        },
        ArithOp::FloorRem => match checked_floor_divmod(a, b) {
            Ok((_, r)) => Ok(r),
            Err(e) => Err(e),
        },
        ArithOp::Gcd => checked_gcd(a, b),
        ArithOp::Lcm => checked_lcm(a, b),
    }
}

const fn arith_dispatch(a: Option<int>, b: Option<int>, op: ArithOp) -> ArithDispatch {
    use ArithOp::*;
    match (a, b, op) {
        (Some(_), Some(_), _) => ArithDispatch::Static,
        // Division/remainder by static 0 always fails at compile time.
        (None, Some(0), Div | Rem | CeilDiv | FloorDiv | FloorRem) => ArithDispatch::Static,
        // Unconditional absorbing identities.
        (Some(0), None, Mul | Lcm)
        | (None, Some(0), Mul | Lcm)
        | (None, Some(1), Rem | FloorRem)
        | (Some(1 | -1), None, Gcd)
        | (None, Some(1 | -1), Gcd) => ArithDispatch::Static,
        // Static 0 numerator with dynamic divisor: static Int<0> after checking b != 0.
        (Some(0), None, Div | Rem | CeilDiv | FloorDiv | FloorRem) => {
            ArithDispatch::StaticCheckedDiv
        }
        _ => ArithDispatch::Dynamic,
    }
}

const fn static_op_result(a: Option<int>, b: Option<int>, op: ArithOp) -> int {
    use ArithOp::*;
    match (a, b, op) {
        (Some(a), Some(b), _) => checked_op(a, b, op).v(),
        (None, Some(0), Div | Rem | CeilDiv | FloorDiv | FloorRem) => {
            Result::<int>::Err(NoodleCuteError::DivisionByZero).v()
        }
        (Some(0), None, Mul | Lcm)
        | (None, Some(0), Mul | Lcm)
        | (None, Some(1), Rem | FloorRem) => 0,
        (Some(1 | -1), None, Gcd) | (None, Some(1 | -1), Gcd) => 1,
        _ => unreachable!(),
    }
}

/// Selects the `ArithDispatch` discriminant for binary `OP` on operand types `A` and `B`.
#[doc(hidden)]
pub const ARITH_DISPATCH<A: Integer, B: Integer, const OP: ArithOp>: u8 =
    arith_dispatch(A::STATIC_VALUE, B::STATIC_VALUE, OP) as u8;

const STATIC_OP_RESULT<A: Integer, B: Integer, const OP: ArithOp>: int =
    static_op_result(A::STATIC_VALUE, B::STATIC_VALUE, OP);

/// Unified binary dispatcher for `Self` and `B` under operation `OP` and `DISPATCH` strategy.
#[doc(hidden)]
pub const trait Arithmetic<B: [const] Integer, const OP: ArithOp, const DISPATCH: u8>:
    [const] Integer
{
    type Output: [const] Integer;
    fn apply(self, b: B) -> Self::Output;
}

const impl<A: [const] Integer, B: [const] Integer, const OP: ArithOp>
    Arithmetic<B, OP, { ArithDispatch::Static as u8 }> for A
{
    type Output = Int<{ STATIC_OP_RESULT::<A, B, OP> }>;

    fn apply(self, _: B) -> Self::Output {
        Int
    }
}

const impl<A: [const] Integer, B: [const] Integer, const OP: ArithOp>
    Arithmetic<B, OP, { ArithDispatch::StaticCheckedDiv as u8 }> for A
{
    type Output = Int<0>;

    fn apply(self, b: B) -> Self::Output {
        if b.value() == 0 {
            Result::<()>::Err(NoodleCuteError::DivisionByZero).v();
        }
        Int
    }
}

const impl<A: [const] Integer, B: [const] Integer, const OP: ArithOp>
    Arithmetic<B, OP, { ArithDispatch::Dynamic as u8 }> for A
{
    type Output = int;

    fn apply(self, b: B) -> Self::Output {
        checked_op(self.value(), b.value(), OP).v()
    }
}

/// Binds `Arithmetic` to the computed `ARITH_DISPATCH` for `A`, `B`, and `OP`.
#[doc(hidden)]
pub const trait ArithmeticAlias<A: Integer, B: Integer, const OP: ArithOp> =
    [const] Arithmetic<B, OP, { ARITH_DISPATCH::<A, B, OP> }>;

#[inline(always)]
const fn apply_op<A, B, const OP: ArithOp>(a: A, b: B) -> ArithOpResult<A, B, OP>
where
    A: [const] Integer + [const] ArithmeticAlias<A, B, OP>,
    B: [const] Integer,
{
    <A as Arithmetic<B, OP, { ARITH_DISPATCH::<A, B, OP> }>>::apply(a, b)
}

/// Trait alias for addition of two CuTe integer types.
pub const trait ArithAdd<A: Integer, B: Integer> = [const] ArithmeticAlias<A, B, { ArithOp::Add }>;
/// Trait alias for subtraction of two CuTe integer types.
pub const trait ArithSub<A: Integer, B: Integer> = [const] ArithmeticAlias<A, B, { ArithOp::Sub }>;
/// Trait alias for multiplication of two CuTe integer types.
pub const trait ArithMul<A: Integer, B: Integer> = [const] ArithmeticAlias<A, B, { ArithOp::Mul }>;
/// Trait alias for truncating division of two CuTe integer types.
pub const trait ArithDiv<A: Integer, B: Integer> = [const] ArithmeticAlias<A, B, { ArithOp::Div }>;
/// Trait alias for truncating remainder of two CuTe integer types.
pub const trait ArithRem<A: Integer, B: Integer> = [const] ArithmeticAlias<A, B, { ArithOp::Rem }>;
/// Trait alias for ceiling division of two CuTe integer types.
pub const trait ArithCeilDiv<A: Integer, B: Integer> =
    [const] ArithmeticAlias<A, B, { ArithOp::CeilDiv }>;
/// Trait alias for floor division of two CuTe integer types.
pub const trait ArithFloorDiv<A: Integer, B: Integer> =
    [const] ArithmeticAlias<A, B, { ArithOp::FloorDiv }>;
/// Trait alias for floor remainder of two CuTe integer types.
pub const trait ArithFloorRem<A: Integer, B: Integer> =
    [const] ArithmeticAlias<A, B, { ArithOp::FloorRem }>;
/// Combined trait for both floor division and floor remainder of two CuTe integer types.
pub const trait ArithFloorDivMod<A: Integer, B: Integer>:
    [const] ArithFloorDiv<A, B> + [const] ArithFloorRem<A, B>
{
}
const impl<A, B> ArithFloorDivMod<A, B> for A
where
    A: [const] Integer + [const] ArithFloorDiv<A, B> + [const] ArithFloorRem<A, B>,
    B: [const] Integer,
{
}
/// Trait alias for greatest common divisor of two CuTe integer types.
pub const trait ArithGcd<A: Integer, B: Integer> = [const] ArithmeticAlias<A, B, { ArithOp::Gcd }>;

/// Trait alias for least common multiple of two CuTe integer types.
pub const trait ArithLcm<A: Integer, B: Integer> = [const] ArithmeticAlias<A, B, { ArithOp::Lcm }>;

/// Output type selected by the operation and static/dynamic status of `A` and `B`.
pub type ArithOpResult<A, B, const OP: ArithOp> =
    <A as Arithmetic<B, OP, { ARITH_DISPATCH::<A, B, OP> }>>::Output;

/// Output type of addition between two CuTe integer types `A` and `B`.
pub type Sum<A, B> = ArithOpResult<A, B, { ArithOp::Add }>;

/// Output type of subtraction between two CuTe integer types `A` and `B`.
pub type Difference<A, B> = ArithOpResult<A, B, { ArithOp::Sub }>;

/// Output type of multiplication between two CuTe integer types `A` and `B`.
pub type Product<A, B> = ArithOpResult<A, B, { ArithOp::Mul }>;

/// Output type of truncating division between two CuTe integer types `A` and `B`.
pub type Quotient<A, B> = ArithOpResult<A, B, { ArithOp::Div }>;

/// Output type of truncating remainder between two CuTe integer types `A` and `B`.
pub type Remainder<A, B> = ArithOpResult<A, B, { ArithOp::Rem }>;

/// Output type of ceiling division between two CuTe integer types `A` and `B`.
pub type CeilQuotient<A, B> = ArithOpResult<A, B, { ArithOp::CeilDiv }>;

/// Quotient type of floor division ([`floor_div`] / [`floor_divmod`]).
pub type FloorQuotient<A, B> = ArithOpResult<A, B, { ArithOp::FloorDiv }>;

/// Remainder type of floor division ([`floor_rem`] / [`floor_divmod`]).
pub type FloorRemainder<A, B> = ArithOpResult<A, B, { ArithOp::FloorRem }>;

/// Output type of `gcd(A, B)`.
pub type Gcd<A, B> = ArithOpResult<A, B, { ArithOp::Gcd }>;

/// Output type of `lcm(A, B)`.
pub type Lcm<A, B> = ArithOpResult<A, B, { ArithOp::Lcm }>;

/// Adds two CuTe integers.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, add};
///
/// assert_eq!(add(Int::<2>, Int::<3>), Int::<5>);
/// assert_eq!(add(Int::<2>, 3i32), 5i32);
/// ```
///
/// Static overflow is rejected at compile time:
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, add};
/// let _ = add(Int::<2_147_483_647>, Int::<1>);
/// ```
pub const fn add<A, B>(a: A, b: B) -> Sum<A, B>
where
    A: [const] Integer + [const] ArithAdd<A, B>,
    B: [const] Integer,
{
    apply_op::<A, B, { ArithOp::Add }>(a, b)
}

/// Subtracts `b` from `a`.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, sub};
///
/// assert_eq!(sub(Int::<2>, Int::<5>), Int::<-3>);
/// assert_eq!(sub(2i32, Int::<5>), -3i32);
/// ```
///
/// Static underflow/overflow is rejected at compile time:
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, sub};
/// let _ = sub(Int::<-2_147_483_648>, Int::<1>);
/// ```
pub const fn sub<A, B>(a: A, b: B) -> Difference<A, B>
where
    A: [const] Integer + [const] ArithSub<A, B>,
    B: [const] Integer,
{
    apply_op::<A, B, { ArithOp::Sub }>(a, b)
}

/// Multiplies two CuTe integers.
///
/// Multiplying by static `Int<0>` always preserves static `Int<0>`, even when the other
/// operand is dynamic.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, mul};
///
/// assert_eq!(mul(Int::<2>, Int::<3>), Int::<6>);
/// let z: Int<0> = mul(Int::<0>, i32::MIN);
/// assert_eq!(z, Int::<0>);
/// ```
pub const fn mul<A, B>(a: A, b: B) -> Product<A, B>
where
    A: [const] Integer + [const] ArithMul<A, B>,
    B: [const] Integer,
{
    apply_op::<A, B, { ArithOp::Mul }>(a, b)
}

/// Truncating integer division `a / b`.
///
/// Dividing static `Int<0>` by a dynamic divisor preserves static `Int<0>` while
/// still checking `b != 0` at runtime.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, div};
///
/// assert_eq!(div(Int::<-7>, Int::<3>), Int::<-2>);
/// assert_eq!(div(-7i32, Int::<3>), -2i32);
/// let z: Int<0> = div(Int::<0>, -3i32);
/// assert_eq!(z, Int::<0>);
/// ```
///
/// Dividing by static `Int<0>` is rejected at compile time even for dynamic dividends
/// (for both [`div`] and the `/` operator):
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, div};
/// let _ = div(10i32, Int::<0>);
/// ```
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// let _ = 10i32 / Int::<0>;
/// ```
pub const fn div<A, B>(a: A, b: B) -> Quotient<A, B>
where
    A: [const] Integer + [const] ArithDiv<A, B>,
    B: [const] Integer,
{
    apply_op::<A, B, { ArithOp::Div }>(a, b)
}

/// Truncating integer remainder `a % b`.
///
/// Remainder modulo static `Int<1>` always yields static `Int<0>`.
/// Dividing static `Int<0>` by a dynamic divisor preserves static `Int<0>` while
/// still checking `b != 0` at runtime.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, rem};
///
/// assert_eq!(rem(Int::<-7>, Int::<3>), Int::<-1>);
/// assert_eq!(rem(-7i32, Int::<3>), -1i32);
/// let z1: Int<0> = rem(i32::MIN, Int::<1>);
/// let z2: Int<0> = rem(Int::<0>, -3i32);
/// assert_eq!((z1, z2), (Int::<0>, Int::<0>));
/// ```
///
/// Remainder modulo static `Int<0>` is rejected at compile time even for dynamic dividends
/// (for both [`rem`] and the `%` operator):
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, rem};
/// let _ = rem(10i32, Int::<0>);
/// ```
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// let _ = 10i32 % Int::<0>;
/// ```
pub const fn rem<A, B>(a: A, b: B) -> Remainder<A, B>
where
    A: [const] Integer + [const] ArithRem<A, B>,
    B: [const] Integer,
{
    apply_op::<A, B, { ArithOp::Rem }>(a, b)
}

/// Ceiling integer division `ceil(a / b)`.
///
/// Computes `a / b` rounded toward `+∞` using checked division and remainder so `a + b` cannot overflow.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, ceil_div};
///
/// assert_eq!(ceil_div(Int::<7>, Int::<3>), Int::<3>);
/// assert_eq!(ceil_div(Int::<-7>, Int::<3>), Int::<-2>);
/// assert_eq!(ceil_div(7i32, Int::<-3>), -2i32);
/// ```
///
/// Division by static `Int<0>` is rejected at compile time (for both static and dynamic dividends):
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, ceil_div};
/// let _ = ceil_div(5i32, Int::<0>);
/// ```
pub const fn ceil_div<A, B>(a: A, b: B) -> CeilQuotient<A, B>
where
    A: [const] Integer + [const] ArithCeilDiv<A, B>,
    B: [const] Integer,
{
    apply_op::<A, B, { ArithOp::CeilDiv }>(a, b)
}

/// Floor quotient `floor(a / b)`.
///
/// Dividing static `Int<0>` by a dynamic divisor preserves static `Int<0>` while
/// still checking `b != 0` at runtime.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, floor_div};
///
/// assert_eq!(floor_div(Int::<-7>, Int::<3>), Int::<-3>);
/// assert_eq!(floor_div(-7i32, Int::<-3>), 2i32);
/// let z: Int<0> = floor_div(Int::<0>, -3i32);
/// assert_eq!(z, Int::<0>);
/// ```
///
/// Division by static `Int<0>` is rejected at compile time (even for dynamic dividends):
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, floor_div};
/// let _ = floor_div(5i32, Int::<0>);
/// ```
pub const fn floor_div<A, B>(a: A, b: B) -> FloorQuotient<A, B>
where
    A: [const] Integer + [const] ArithFloorDiv<A, B>,
    B: [const] Integer,
{
    apply_op::<A, B, { ArithOp::FloorDiv }>(a, b)
}

/// Floor remainder `a - floor(a / b) * b` (takes the sign of divisor `b`).
///
/// Floor remainder modulo static `Int<1>` always yields static `Int<0>`.
/// Floor remainder of static `Int<0>` by a dynamic divisor preserves static `Int<0>`
/// while checking `b != 0` at runtime.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, floor_rem};
///
/// assert_eq!(floor_rem(Int::<-7>, Int::<3>), Int::<2>);
/// assert_eq!(floor_rem(-7i32, Int::<-3>), -1i32);
/// let z1: Int<0> = floor_rem(i32::MIN, Int::<1>);
/// let z2: Int<0> = floor_rem(Int::<0>, -3i32);
/// assert_eq!((z1, z2), (Int::<0>, Int::<0>));
/// ```
///
/// Division by static `Int<0>` is rejected at compile time (even for dynamic dividends):
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, floor_rem};
/// let _ = floor_rem(5i32, Int::<0>);
/// ```
pub const fn floor_rem<A, B>(a: A, b: B) -> FloorRemainder<A, B>
where
    A: [const] Integer + [const] ArithFloorRem<A, B>,
    B: [const] Integer,
{
    apply_op::<A, B, { ArithOp::FloorRem }>(a, b)
}

/// Computes floor quotient and remainder matching Python `divmod(a, b)`.
///
/// Dividing a dynamic integer by static `Int<1>` returns `(int, Int<0>)`, preserving
/// a statically-zero remainder. Dividing static `Int<0>` by a dynamic divisor returns
/// `(Int<0>, Int<0>)` after checking `b != 0`.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, floor_divmod};
///
/// assert_eq!(floor_divmod(Int::<-7>, Int::<3>), (Int::<-3>, Int::<2>));
/// assert_eq!(floor_divmod(-7i32, Int::<-3>), (2i32, -1i32));
/// let (q, r): (i32, Int<0>) = floor_divmod(-7i32, Int::<1>);
/// assert_eq!((q, r), (-7, Int::<0>));
/// ```
///
/// Division by static `Int<0>` is rejected at compile time (even for dynamic dividends):
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, floor_divmod};
/// let _ = floor_divmod(5i32, Int::<0>);
/// ```
pub const fn floor_divmod<A, B>(a: A, b: B) -> (FloorQuotient<A, B>, FloorRemainder<A, B>)
where
    A: [const] Integer + [const] ArithFloorDivMod<A, B>,
    B: [const] Integer,
{
    (floor_div(a, b), floor_rem(a, b))
}

/// Computes the non-negative greatest common divisor of `a` and `b` (`gcd(0, 0) == 0`).
///
/// When either operand is static `Int<1>` or `Int<-1>`, the result is statically `Int<1>`.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, gcd};
///
/// assert_eq!(gcd(Int::<-12>, Int::<18>), Int::<6>);
/// assert_eq!(gcd(12i32, Int::<-18>), 6i32);
/// let one: Int<1> = gcd(Int::<1>, i32::MIN);
/// assert_eq!(one, Int::<1>);
/// ```
///
/// When `gcd(|a|, |b|)` exceeds `int::MAX` (e.g. `gcd(int::MIN, 0)`), static evaluation fails at compile time:
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, gcd};
/// let _ = gcd(Int::<-2_147_483_648>, Int::<0>);
/// ```
pub const fn gcd<A, B>(a: A, b: B) -> Gcd<A, B>
where
    A: [const] Integer + [const] ArithGcd<A, B>,
    B: [const] Integer,
{
    apply_op::<A, B, { ArithOp::Gcd }>(a, b)
}

/// Computes the non-negative least common multiple of `a` and `b` (`lcm(0, x) == 0`).
///
/// When either operand is static `Int<0>`, the result is statically `Int<0>`.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, lcm};
///
/// assert_eq!(lcm(Int::<4>, Int::<-6>), Int::<12>);
/// assert_eq!(lcm(-4i32, Int::<6>), 12i32);
/// let zero: Int<0> = lcm(Int::<0>, i32::MIN);
/// assert_eq!(zero, Int::<0>);
/// ```
///
/// Static overflow is rejected at compile time:
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, lcm};
/// let _ = lcm(Int::<65536>, Int::<65537>);
/// ```
pub const fn lcm<A, B>(a: A, b: B) -> Lcm<A, B>
where
    A: [const] Integer + [const] ArithLcm<A, B>,
    B: [const] Integer,
{
    apply_op::<A, B, { ArithOp::Lcm }>(a, b)
}

macro_rules! impl_op {
    ($Trait:ident $method:ident $Op:ident; [$($g:tt)*] $lhs:ty, $rhs:ty) => {
        const impl<$($g)*> core::ops::$Trait<$rhs> for $lhs
        where
            $lhs: [const] Integer + [const] ArithmeticAlias<$lhs, $rhs, { ArithOp::$Op }>,
            $rhs: [const] Integer,
        {
            type Output = ArithOpResult<$lhs, $rhs, { ArithOp::$Op }>;
            fn $method(self, rhs: $rhs) -> Self::Output {
                apply_op::<$lhs, $rhs, { ArithOp::$Op }>(self, rhs)
            }
        }
    };
}
macro_rules! impl_ops {
    ($($Trait:ident $method:ident $Op:ident),+) => {$(
        impl_op!($Trait $method $Op; [const N: int, B: [const] Integer] Int<N>, B);
        impl_op!($Trait $method $Op; [const N: int] int, Int<N>);
    )+};
}
impl_ops!(Add add Add, Sub sub Sub, Mul mul Mul, Div div Div, Rem rem Rem);
