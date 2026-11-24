//! Binary polynomial stride scalar [`F2`] over $\mathbb{F}_2\[x\]$.
//!
//! Represents non-negative integers whose bits are coefficients of polynomials over
//! the two-element field:
//! - Addition and subtraction are bitwise XOR (`+` == `-`, unary `-x == x`);
//! - Multiplication is carry-less polynomial multiplication (`clmul`);
//! - Euclidean division ([`f2_divmod`]) is polynomial long division with remainder degree
//!   strictly less than divisor degree.
//!
//! # Main Interfaces
//!
//! - [`F2<V>`] / [`StaticF2<N>`] / [`F2Scalar`]: Static (`F2<Int<N>>`, zero-sized) and
//!   dynamic (`F2<int>`) binary-field stride scalars.
//! - [`f2`] / [`IntoF2`]: Idempotent constructor (`f2(f2(x)) == f2(x)`).
//! - [`f2_add`], [`f2_sub`], [`f2_mul`], [`f2_divmod`]: `const`-evaluable binary-field
//!   operations, plus `const` `core::ops::{Add, Sub, Neg, Mul}` overloads.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::swizzle::{F2, f2, f2_divmod};
//!
//! const SUM: F2<Int<0b0110>> = f2(Int::<0b1010>) + f2(Int::<0b1100>);
//! const PROD: F2<Int<0b101>> = f2(Int::<0b11>) * Int::<0b11>;
//! assert_eq!(SUM, Int::<0b0110>);
//! assert_eq!(PROD, f2(Int::<0b101>));
//! assert_eq!(f2_divmod(f2(Int::<0b1011>), Int::<0b11>), (f2(Int::<0b110>), f2(Int::<0b1>)));
//! ```
//!
//! # Notes
//!
//! - Negative static `F2` values (`F2<Int<-1>>`), adding nonzero integers (`f2(Int::<1>) + Int::<1>`),
//!   multiplying by negative integers, carry-less overflow past `int::MAX`, and dividing by `0`
//!   are rejected at compile time for static operands and panic at runtime for dynamic operands.

use core::cmp::Ordering;
use core::fmt::{Display, Formatter};
use core::ops::{Add, Mul, Neg, Sub};

use crate::integer::ops::{checked_cldivmod, checked_clmul};
use crate::integer::{BOTH_STATIC, IS_NON_NEG, Int, Integer, cmp, eq, int};
use crate::{NoodleCuteError, NoodleCuteResult, Result};

/// Compile-time guard restricting static [`F2`] values to non-negative integers (`VALID == true`).
#[doc(hidden)]
pub const trait ValidF2Value<const VALID: bool> {}
const impl ValidF2Value<true> for () {}

mod sealed {
    use crate::integer::{IS_NON_NEG, Int, int};

    use super::{F2, ValidF2Value};

    /// Seals [`F2Scalar`](super::F2Scalar) to `int` and non-negative `Int<N>`.
    #[doc(hidden)]
    pub trait SealedF2Scalar {}
    impl SealedF2Scalar for int {}
    impl<const N: int> SealedF2Scalar for Int<N> where (): ValidF2Value<{ IS_NON_NEG::<N> }> {}

    /// Seals [`F2Operand`](super::F2Operand) to `int`, `Int<N>`, and `F2<V>`.
    #[doc(hidden)]
    pub trait SealedF2Operand {}
    impl SealedF2Operand for int {}
    impl<const N: int> SealedF2Operand for Int<N> {}
    impl<V: SealedF2Scalar> SealedF2Operand for F2<V> {}
}

/// Sealed trait for non-negative scalar integers stored inside [`F2<V>`]
/// ([`int`] and non-negative [`Int<N>`]).
pub const trait F2Scalar: sealed::SealedF2Scalar + [const] Integer {}
const impl<V: sealed::SealedF2Scalar + [const] Integer> F2Scalar for V {}

/// A stride scalar over the binary polynomial ring $\mathbb{F}_2\[x\]$:
/// `+` / `-` is bitwise XOR and `*` is carry-less multiplication.
///
/// Negative static values are rejected at compile time:
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::swizzle::F2;
/// let _ = F2::new(Int::<-1>);
/// ```
///
/// Adding a nonzero integer to an [`F2`] value is rejected at compile time:
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::swizzle::f2;
/// let _ = f2(Int::<1>) + Int::<1>;
/// ```
///
/// ```compile_fail
/// use noodle_cute::swizzle::StaticF2;
/// let _ = StaticF2::<-1>::default();
/// ```
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F2<V = int>(pub(crate) V);

impl<V: F2Scalar + Default> Default for F2<V> {
    fn default() -> Self {
        Self::new(V::default())
    }
}

/// Type alias for a compile-time static [`F2`] value (`F2<Int<N>>`).
pub type StaticF2<const N: int> = F2<Int<N>>;

impl<V> F2<V> {
    /// Constructs a validated [`F2`] value, panicking if a dynamic [`int`] is negative.
    pub const fn new(value: V) -> Self
    where
        V: [const] F2Scalar,
    {
        assert!(value.value() >= 0, "F2 value must be non-negative");
        Self(value)
    }

    /// Returns the underlying static [`Int<N>`] or dynamic [`int`] value.
    pub const fn inner(self) -> V
    where
        V: Copy,
    {
        self.0
    }

    /// Returns the underlying native [`int`] value.
    pub const fn value(self) -> int
    where
        V: [const] Integer,
    {
        self.0.value()
    }
}

impl<V: Display> Display for F2<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "F{}", self.0)
    }
}

/// Idempotent conversion into [`F2`]: wraps [`int`] or non-negative [`Int<N>`] and passes
/// existing [`F2<V>`] through unchanged.
pub const trait IntoF2: Copy {
    /// Resulting [`F2`] type.
    type Output: Copy;

    /// Converts `self` into an [`F2`] value.
    fn into_f2(self) -> Self::Output;
}

const impl IntoF2 for int {
    type Output = F2<int>;

    fn into_f2(self) -> F2<int> {
        F2::new(self)
    }
}

const impl<const N: int> IntoF2 for Int<N>
where
    (): [const] ValidF2Value<{ IS_NON_NEG::<N> }>,
{
    type Output = F2<Int<N>>;

    fn into_f2(self) -> F2<Int<N>> {
        F2(Int)
    }
}

const impl<V: [const] F2Scalar> IntoF2 for F2<V> {
    type Output = Self;

    fn into_f2(self) -> Self {
        self
    }
}

/// Idempotent [`F2`] constructor matching PyCuTe's `F2(value)`.
pub const fn f2<T: [const] IntoF2>(value: T) -> T::Output {
    value.into_f2()
}

// ------------------------------------------------------------------
// Equality and ordering
// ------------------------------------------------------------------

const impl<L: [const] F2Scalar, R: [const] F2Scalar> PartialEq<F2<R>> for F2<L> {
    fn eq(&self, rhs: &F2<R>) -> bool {
        eq(self.0, rhs.0)
    }
}

const impl<V: [const] F2Scalar> Eq for F2<V> {}

const impl<V: [const] F2Scalar, const N: int> PartialEq<Int<N>> for F2<V> {
    fn eq(&self, rhs: &Int<N>) -> bool {
        eq(self.0, *rhs)
    }
}

const impl<V: [const] F2Scalar> PartialEq<int> for F2<V> {
    fn eq(&self, rhs: &int) -> bool {
        eq(self.0, *rhs)
    }
}

const impl<const N: int, V: [const] F2Scalar> PartialEq<F2<V>> for Int<N> {
    fn eq(&self, rhs: &F2<V>) -> bool {
        eq(*self, rhs.0)
    }
}

const impl<V: [const] F2Scalar> PartialEq<F2<V>> for int {
    fn eq(&self, rhs: &F2<V>) -> bool {
        eq(*self, rhs.0)
    }
}

const impl<L: [const] F2Scalar, R: [const] F2Scalar> PartialOrd<F2<R>> for F2<L> {
    fn partial_cmp(&self, rhs: &F2<R>) -> Option<Ordering> {
        Some(cmp(self.0, rhs.0))
    }
}

const impl<V: [const] F2Scalar> Ord for F2<V> {
    fn cmp(&self, rhs: &Self) -> Ordering {
        cmp(self.0, rhs.0)
    }
}

const impl<V: [const] F2Scalar, const N: int> PartialOrd<Int<N>> for F2<V> {
    fn partial_cmp(&self, rhs: &Int<N>) -> Option<Ordering> {
        Some(cmp(self.0, *rhs))
    }
}

const impl<V: [const] F2Scalar> PartialOrd<int> for F2<V> {
    fn partial_cmp(&self, rhs: &int) -> Option<Ordering> {
        Some(cmp(self.0, *rhs))
    }
}

const impl<const N: int, V: [const] F2Scalar> PartialOrd<F2<V>> for Int<N> {
    fn partial_cmp(&self, rhs: &F2<V>) -> Option<Ordering> {
        Some(cmp(*self, rhs.0))
    }
}

const impl<V: [const] F2Scalar> PartialOrd<F2<V>> for int {
    fn partial_cmp(&self, rhs: &F2<V>) -> Option<Ordering> {
        Some(cmp(*self, rhs.0))
    }
}

// ------------------------------------------------------------------
// Shared operand abstraction for F2 multiplication and Euclidean division
// ------------------------------------------------------------------

/// Sealed trait unifying integer and [`F2`] operands for [`f2_mul`] and [`f2_divmod`].
#[doc(hidden)]
pub const trait F2Operand: sealed::SealedF2Operand + Copy {
    const STATIC_VALUE: Option<int>;
    const GUARANTEED_NON_NEG: bool;
    fn operand_value(self) -> int;
}

const impl F2Operand for int {
    const STATIC_VALUE: Option<int> = None;
    const GUARANTEED_NON_NEG: bool = false;
    fn operand_value(self) -> int {
        self
    }
}

const impl<const N: int> F2Operand for Int<N> {
    const STATIC_VALUE: Option<int> = Some(N);
    const GUARANTEED_NON_NEG: bool = N >= 0;
    fn operand_value(self) -> int {
        N
    }
}

const impl<V: [const] F2Scalar> F2Operand for F2<V> {
    const STATIC_VALUE: Option<int> = V::STATIC_VALUE;
    const GUARANTEED_NON_NEG: bool = true;
    fn operand_value(self) -> int {
        self.0.value()
    }
}

// ------------------------------------------------------------------
// Addition / Subtraction (bitwise XOR) and Unary Negation
// ------------------------------------------------------------------

const STATIC_F2_XOR<L: F2Scalar, R: F2Scalar>: int = match (L::STATIC_VALUE, R::STATIC_VALUE) {
    (Some(l), Some(r)) => l ^ r,
    _ => 0,
};

/// Dispatches `F2<Self> ^ F2<R>` between compile-time static (`BOTH_STATIC = true`) and dynamic.
#[doc(hidden)]
pub const trait F2XorScalar<R: [const] F2Scalar, const BOTH_STATIC: bool>:
    [const] F2Scalar
{
    type Output: [const] F2Scalar;
    fn xor_scalar(self, rhs: R) -> Self::Output;
}

const impl<L: [const] F2Scalar, R: [const] F2Scalar> F2XorScalar<R, true> for L
where
    (): [const] ValidF2Value<{ IS_NON_NEG::<{ STATIC_F2_XOR::<L, R> }> }>,
{
    type Output = Int<{ STATIC_F2_XOR::<L, R> }>;
    fn xor_scalar(self, _: R) -> Self::Output {
        Int
    }
}

const impl<L: [const] F2Scalar, R: [const] F2Scalar> F2XorScalar<R, false> for L {
    type Output = int;
    fn xor_scalar(self, rhs: R) -> Self::Output {
        self.value() ^ rhs.value()
    }
}

/// Output scalar type of `F2<L> + F2<R>`.
#[doc(hidden)]
pub type F2XorOutput<L, R> = <L as F2XorScalar<R, { BOTH_STATIC::<L, R> }>>::Output;

/// Bitwise-OR scalar dispatcher used by codomain bounding (`coshape`) so bits never cancel.
const STATIC_F2_OR<L: F2Scalar, R: F2Scalar>: int = match (L::STATIC_VALUE, R::STATIC_VALUE) {
    (Some(l), Some(r)) => l | r,
    _ => 0,
};

#[doc(hidden)]
pub const trait F2OrScalar<R: [const] F2Scalar, const BOTH_STATIC: bool>:
    [const] F2Scalar
{
    type Output: [const] F2Scalar;
    fn or_scalar(self, rhs: R) -> Self::Output;
}

const impl<L: [const] F2Scalar, R: [const] F2Scalar> F2OrScalar<R, true> for L
where
    (): [const] ValidF2Value<{ IS_NON_NEG::<{ STATIC_F2_OR::<L, R> }> }>,
{
    type Output = Int<{ STATIC_F2_OR::<L, R> }>;
    fn or_scalar(self, _: R) -> Self::Output {
        Int
    }
}

const impl<L: [const] F2Scalar, R: [const] F2Scalar> F2OrScalar<R, false> for L {
    type Output = int;
    fn or_scalar(self, rhs: R) -> Self::Output {
        self.value() | rhs.value()
    }
}

/// Output scalar of bitwise OR between two `F2` scalars `L` and `R`.
#[doc(hidden)]
pub type F2OrOutput<L, R> = <L as F2OrScalar<R, { BOTH_STATIC::<L, R> }>>::Output;

/// Evaluates `lhs + rhs` (bitwise XOR) in $\mathbb{F}_2\[x\]$.
pub const fn f2_add<L, R>(lhs: F2<L>, rhs: F2<R>) -> F2<F2XorOutput<L, R>>
where
    L: [const] F2Scalar + [const] F2XorScalar<R, { BOTH_STATIC::<L, R> }>,
    R: [const] F2Scalar,
{
    F2(lhs.0.xor_scalar(rhs.0))
}

/// Evaluates `lhs - rhs` in $\mathbb{F}_2\[x\]$ (identical to [`f2_add`], since XOR is self-inverse).
pub const fn f2_sub<L, R>(lhs: F2<L>, rhs: F2<R>) -> F2<F2XorOutput<L, R>>
where
    L: [const] F2Scalar + [const] F2XorScalar<R, { BOTH_STATIC::<L, R> }>,
    R: [const] F2Scalar,
{
    f2_add(lhs, rhs)
}

const impl<L, R> Add<F2<R>> for F2<L>
where
    L: [const] F2Scalar + [const] F2XorScalar<R, { BOTH_STATIC::<L, R> }>,
    R: [const] F2Scalar,
{
    type Output = F2<F2XorOutput<L, R>>;
    fn add(self, rhs: F2<R>) -> Self::Output {
        f2_add(self, rhs)
    }
}

const impl<L, R> Sub<F2<R>> for F2<L>
where
    L: [const] F2Scalar + [const] F2XorScalar<R, { BOTH_STATIC::<L, R> }>,
    R: [const] F2Scalar,
{
    type Output = F2<F2XorOutput<L, R>>;
    fn sub(self, rhs: F2<R>) -> Self::Output {
        f2_sub(self, rhs)
    }
}

const impl<V: [const] F2Scalar> Add<Int<0>> for F2<V> {
    type Output = Self;
    fn add(self, _: Int<0>) -> Self {
        self
    }
}

const impl<V: [const] F2Scalar> Sub<Int<0>> for F2<V> {
    type Output = Self;
    fn sub(self, _: Int<0>) -> Self {
        self
    }
}

const impl<V: [const] F2Scalar> Add<F2<V>> for Int<0> {
    type Output = F2<V>;
    fn add(self, rhs: F2<V>) -> F2<V> {
        rhs
    }
}

const impl<V: [const] F2Scalar> Sub<F2<V>> for Int<0> {
    type Output = F2<V>;
    fn sub(self, rhs: F2<V>) -> F2<V> {
        rhs
    }
}

const impl<V: [const] F2Scalar> Neg for F2<V> {
    type Output = Self;
    fn neg(self) -> Self {
        self
    }
}

// ------------------------------------------------------------------
// Carry-less Multiplication (clmul)
// ------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum F2MulDispatch {
    Static = 0,
    StaticZero = 1,
    StaticZeroCheckRhs = 2,
    StaticZeroCheckLhs = 3,
    Dynamic = 4,
}

const fn f2_mul_dispatch(
    a: Option<int>,
    b: Option<int>,
    a_non_neg: bool,
    b_non_neg: bool,
) -> F2MulDispatch {
    if matches!(a, Some(v) if v < 0) || matches!(b, Some(v) if v < 0) {
        return F2MulDispatch::Static;
    }
    match (a, b) {
        (Some(_), Some(_)) => F2MulDispatch::Static,
        (Some(0), None) => {
            if b_non_neg {
                F2MulDispatch::StaticZero
            } else {
                F2MulDispatch::StaticZeroCheckRhs
            }
        }
        (None, Some(0)) => {
            if a_non_neg {
                F2MulDispatch::StaticZero
            } else {
                F2MulDispatch::StaticZeroCheckLhs
            }
        }
        _ => F2MulDispatch::Dynamic,
    }
}

const fn static_f2_mul_val(a: Option<int>, b: Option<int>) -> int {
    match (a, b) {
        (Some(va), Some(vb)) => checked_clmul(va, vb).v(),
        (Some(va), None) => {
            assert!(va >= 0, "F2 multiplication requires non-negative operands");
            0
        }
        (None, Some(vb)) => {
            assert!(vb >= 0, "F2 multiplication requires non-negative operands");
            0
        }
        (None, None) => 0,
    }
}

/// Dispatch discriminant for carry-less multiplication of [`F2Operand`]s `A` and `B`.
#[doc(hidden)]
pub const F2_MUL_DISPATCH<A: F2Operand, B: F2Operand>: u8 =
    f2_mul_dispatch(A::STATIC_VALUE, B::STATIC_VALUE, A::GUARANTEED_NON_NEG, B::GUARANTEED_NON_NEG)
        as u8;

const STATIC_F2_MUL_RESULT<A: F2Operand, B: F2Operand>: int =
    static_f2_mul_val(A::STATIC_VALUE, B::STATIC_VALUE);

/// Dispatches carry-less multiplication between `Self` and `B` per `DISPATCH`.
#[doc(hidden)]
pub const trait F2MulOp<B: [const] F2Operand, const DISPATCH: u8>: [const] F2Operand {
    type OutputScalar: [const] F2Scalar;
    fn apply_f2_mul(self, rhs: B) -> F2<Self::OutputScalar>;
}

const impl<A: [const] F2Operand, B: [const] F2Operand> F2MulOp<B, { F2MulDispatch::Static as u8 }>
    for A
where
    (): [const] ValidF2Value<{ IS_NON_NEG::<{ STATIC_F2_MUL_RESULT::<A, B> }> }>,
{
    type OutputScalar = Int<{ STATIC_F2_MUL_RESULT::<A, B> }>;
    fn apply_f2_mul(self, _: B) -> F2<Self::OutputScalar> {
        F2(Int)
    }
}

const impl<A: [const] F2Operand, B: [const] F2Operand>
    F2MulOp<B, { F2MulDispatch::StaticZero as u8 }> for A
{
    type OutputScalar = Int<0>;
    fn apply_f2_mul(self, _: B) -> F2<Int<0>> {
        F2(Int)
    }
}

const impl<A: [const] F2Operand, B: [const] F2Operand>
    F2MulOp<B, { F2MulDispatch::StaticZeroCheckRhs as u8 }> for A
{
    type OutputScalar = Int<0>;
    fn apply_f2_mul(self, rhs: B) -> F2<Int<0>> {
        assert!(
            rhs.operand_value() >= 0,
            "F2 multiplication requires non-negative operands"
        );
        F2(Int)
    }
}

const impl<A: [const] F2Operand, B: [const] F2Operand>
    F2MulOp<B, { F2MulDispatch::StaticZeroCheckLhs as u8 }> for A
{
    type OutputScalar = Int<0>;
    fn apply_f2_mul(self, _: B) -> F2<Int<0>> {
        assert!(
            self.operand_value() >= 0,
            "F2 multiplication requires non-negative operands"
        );
        F2(Int)
    }
}

const impl<A: [const] F2Operand, B: [const] F2Operand> F2MulOp<B, { F2MulDispatch::Dynamic as u8 }>
    for A
{
    type OutputScalar = int;
    fn apply_f2_mul(self, rhs: B) -> F2<int> {
        F2(checked_clmul(self.operand_value(), rhs.operand_value()).v())
    }
}

/// Output type of carry-less multiplication `f2_mul(A, B)`.
pub type F2Product<A, B> = F2<<A as F2MulOp<B, { F2_MUL_DISPATCH::<A, B> }>>::OutputScalar>;

/// Evaluates carry-less multiplication `a * b` in $\mathbb{F}_2\[x\]$.
///
/// Negative static operands or static carry-less overflow fail at compile time:
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::swizzle::f2;
/// let _ = f2(Int::<1>) * Int::<-1>;
/// ```
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::swizzle::f2;
/// let _ = f2(Int::<2>) * f2(Int::<1_073_741_824>);
/// ```
pub const fn f2_mul<A, B>(a: A, b: B) -> F2Product<A, B>
where
    A: [const] F2Operand + [const] F2MulOp<B, { F2_MUL_DISPATCH::<A, B> }>,
    B: [const] F2Operand,
{
    <A as F2MulOp<B, { F2_MUL_DISPATCH::<A, B> }>>::apply_f2_mul(a, b)
}

const impl<L, R> Mul<F2<R>> for F2<L>
where
    L: [const] F2Scalar,
    R: [const] F2Scalar,
    F2<L>: [const] F2MulOp<F2<R>, { F2_MUL_DISPATCH::<F2<L>, F2<R>> }>,
{
    type Output = F2Product<F2<L>, F2<R>>;
    fn mul(self, rhs: F2<R>) -> Self::Output {
        f2_mul(self, rhs)
    }
}

const impl<L, const N: int> Mul<Int<N>> for F2<L>
where
    L: [const] F2Scalar,
    F2<L>: [const] F2MulOp<Int<N>, { F2_MUL_DISPATCH::<F2<L>, Int<N>> }>,
{
    type Output = F2Product<F2<L>, Int<N>>;
    fn mul(self, rhs: Int<N>) -> Self::Output {
        f2_mul(self, rhs)
    }
}

const impl<L> Mul<int> for F2<L>
where
    L: [const] F2Scalar,
    F2<L>: [const] F2MulOp<int, { F2_MUL_DISPATCH::<F2<L>, int> }>,
{
    type Output = F2Product<F2<L>, int>;
    fn mul(self, rhs: int) -> Self::Output {
        f2_mul(self, rhs)
    }
}

const impl<const N: int, R> Mul<F2<R>> for Int<N>
where
    R: [const] F2Scalar,
    Int<N>: [const] F2MulOp<F2<R>, { F2_MUL_DISPATCH::<Int<N>, F2<R>> }>,
{
    type Output = F2Product<Int<N>, F2<R>>;
    fn mul(self, rhs: F2<R>) -> Self::Output {
        f2_mul(self, rhs)
    }
}

const impl<R> Mul<F2<R>> for int
where
    R: [const] F2Scalar,
    int: [const] F2MulOp<F2<R>, { F2_MUL_DISPATCH::<int, F2<R>> }>,
{
    type Output = F2Product<int, F2<R>>;
    fn mul(self, rhs: F2<R>) -> Self::Output {
        f2_mul(self, rhs)
    }
}

// ------------------------------------------------------------------
// Carry-less Polynomial Euclidean Division (f2_divmod)
// ------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum F2DivDispatch {
    Static = 0,
    StaticZeroDividend = 1,
    StaticOneDivisor = 2,
    Dynamic = 3,
}

const fn f2_div_dispatch(a: Option<int>, d: Option<int>) -> F2DivDispatch {
    if matches!(d, Some(v) if v <= 0) {
        return F2DivDispatch::Static;
    }
    match (a, d) {
        (Some(_), Some(_)) => F2DivDispatch::Static,
        (Some(0), None) => F2DivDispatch::StaticZeroDividend,
        (None, Some(1)) => F2DivDispatch::StaticOneDivisor,
        _ => F2DivDispatch::Dynamic,
    }
}

const fn static_f2_divmod_val(a: Option<int>, d: Option<int>) -> (int, int) {
    match (a, d) {
        (Some(va), Some(vd)) => checked_cldivmod(va, vd).v(),
        (None, Some(0)) => Result::<(int, int)>::Err(NoodleCuteError::DivisionByZero).v(),
        (None, Some(vd)) => {
            assert!(vd > 0, "F2 division requires positive divisor");
            (0, 0)
        }
        _ => (0, 0),
    }
}

/// Dispatch discriminant for polynomial division `f2_divmod(F2<A>, D)`.
#[doc(hidden)]
pub const F2_DIV_DISPATCH<A: F2Scalar, D: F2Operand>: u8 =
    f2_div_dispatch(A::STATIC_VALUE, D::STATIC_VALUE) as u8;

const STATIC_F2_QUOT<A: F2Scalar, D: F2Operand>: int =
    static_f2_divmod_val(A::STATIC_VALUE, D::STATIC_VALUE).0;

const STATIC_F2_REM<A: F2Scalar, D: F2Operand>: int =
    static_f2_divmod_val(A::STATIC_VALUE, D::STATIC_VALUE).1;

/// Dispatches polynomial Euclidean division `f2_divmod(self, divisor)` per `DISPATCH`.
#[doc(hidden)]
pub const trait F2DivModOp<D: [const] F2Operand, const DISPATCH: u8>: Copy {
    type QuotScalar: [const] F2Scalar;
    type RemScalar: [const] F2Scalar;
    fn apply_f2_divmod(self, divisor: D) -> (F2<Self::QuotScalar>, F2<Self::RemScalar>);
}

const impl<A: [const] F2Scalar, D: [const] F2Operand> F2DivModOp<D, { F2DivDispatch::Static as u8 }>
    for F2<A>
where
    (): [const] ValidF2Value<{ IS_NON_NEG::<{ STATIC_F2_QUOT::<A, D> }> }>,
    (): [const] ValidF2Value<{ IS_NON_NEG::<{ STATIC_F2_REM::<A, D> }> }>,
{
    type QuotScalar = Int<{ STATIC_F2_QUOT::<A, D> }>;
    type RemScalar = Int<{ STATIC_F2_REM::<A, D> }>;

    fn apply_f2_divmod(self, _: D) -> (F2<Self::QuotScalar>, F2<Self::RemScalar>) {
        (F2(Int), F2(Int))
    }
}

const impl<A: [const] F2Scalar, D: [const] F2Operand>
    F2DivModOp<D, { F2DivDispatch::StaticZeroDividend as u8 }> for F2<A>
{
    type QuotScalar = Int<0>;
    type RemScalar = Int<0>;

    fn apply_f2_divmod(self, divisor: D) -> (F2<Int<0>>, F2<Int<0>>) {
        let d = divisor.operand_value();
        if d == 0 {
            Result::<()>::Err(NoodleCuteError::DivisionByZero).v();
        }
        assert!(d > 0, "F2 division requires positive divisor");
        (F2(Int), F2(Int))
    }
}

const impl<A: [const] F2Scalar, D: [const] F2Operand>
    F2DivModOp<D, { F2DivDispatch::StaticOneDivisor as u8 }> for F2<A>
{
    type QuotScalar = A;
    type RemScalar = Int<0>;

    fn apply_f2_divmod(self, _: D) -> (F2<A>, F2<Int<0>>) {
        (self, F2(Int))
    }
}

const impl<A: [const] F2Scalar, D: [const] F2Operand>
    F2DivModOp<D, { F2DivDispatch::Dynamic as u8 }> for F2<A>
{
    type QuotScalar = int;
    type RemScalar = int;

    fn apply_f2_divmod(self, divisor: D) -> (F2<int>, F2<int>) {
        let (q, r) = checked_cldivmod(self.value(), divisor.operand_value()).v();
        (F2(q), F2(r))
    }
}

/// Quotient type of [`f2_divmod`].
pub type F2Quotient<A, D> = F2<<F2<A> as F2DivModOp<D, { F2_DIV_DISPATCH::<A, D> }>>::QuotScalar>;

/// Remainder type of [`f2_divmod`].
pub type F2Remainder<A, D> = F2<<F2<A> as F2DivModOp<D, { F2_DIV_DISPATCH::<A, D> }>>::RemScalar>;

/// Euclidean polynomial division in $\mathbb{F}_2\[x\]$: returns the unique `(q, r)` satisfying
/// `a == q * d + r` with `deg(r) < deg(d)`.
///
/// Dividing by static `0` or a negative integer fails at compile time:
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::swizzle::{f2, f2_divmod};
/// let _ = f2_divmod(f2(10i32), Int::<0>);
/// ```
pub const fn f2_divmod<A, D>(a: F2<A>, d: D) -> (F2Quotient<A, D>, F2Remainder<A, D>)
where
    A: [const] F2Scalar,
    D: [const] F2Operand,
    F2<A>: [const] F2DivModOp<D, { F2_DIV_DISPATCH::<A, D> }>,
{
    <F2<A> as F2DivModOp<D, { F2_DIV_DISPATCH::<A, D> }>>::apply_f2_divmod(a, d)
}
