//! Static and dynamic integer types and arithmetic for CuTe.
//!
//! CuTe coordinates, shapes, strides, and layouts require an algebra that seamlessly
//! bridges compile-time constants ([`Int<N>`]) and run-time values ([`int`]).
//!
//! # Core Concepts
//!
//! - **Static Integers ([`Int<N>`])**: Zero-sized types representing compile-time `int`
//!   constants. Operations between static integers are evaluated at compile time into
//!   precise `Int<R>` result types.
//! - **Dynamic Integers ([`int`])**: Native 32-bit signed primitive integer (`i32`).
//! - **Unified Abstraction ([`Integer`], [`StaticInteger`])**: Sealed traits unifying
//!   value access, compile-time inspection (`STATIC_VALUE`, `IS_STATIC`, `VALUE`),
//!   and `const` comparisons.
//! - **Submodules**:
//!   - [`mod@ops`]: Binary arithmetic (`+`, `-`, `*`, `/`, `%`, `&`, `|`, `^`),
//!     `ceil_div`, unary negation (`neg`, `-`), `abs`, `floor_divmod` / `floor_div` /
//!     `floor_rem`, signed shifts (`shiftl`, `shiftr`), and number-theoretic helpers
//!     (`gcd`, `lcm`).
//!   - [`mod@cmp`]: `const` comparisons, three-valued static predicates, and `min` / `max`.
//!
//! # Arithmetic & Comparison Rules
//!
//! Binary arithmetic ([`add`], [`sub`], [`mul`], [`div`], [`rem`], [`ceil_div`],
//! [`floor_divmod`], [`min`], [`max`], [`gcd`], [`lcm`]) and unary operations
//! ([`neg`], [`abs`]) follow consistent representation and overflow rules:
//!
//! 1. **Static × Static**: Evaluated at compile time into [`Int<R>`]. Invalid static
//!    results (signed `int` overflow, division by zero, or negating/`abs` of `int::MIN`)
//!    are compile-time errors.
//! 2. **Static absorbing / error rules with a dynamic operand**:
//!    - Dividing or taking the remainder by static `Int<0>` fails at compile time (`E0080`),
//!      even when the dividend is dynamic.
//!    - Unconditional absorbing identities (`0 * x`, `x * 0`, `x % 1`, `floor_rem(x, 1)`,
//!      `lcm(0, x)`, `lcm(x, 0)`, `gcd(±1, x)`, `gcd(x, ±1)`) preserve static `Int<0>` or `Int<1>`.
//!    - Static zero dividend with a dynamic divisor (`0 / x`, `0 % x`, `ceil_div(0, x)`,
//!      `floor_div(0, x)`, `floor_rem(0, x)`) preserves static `Int<0>` while still checking
//!      `x != 0` at runtime.
//! 3. **General dynamic / mixed operands**: Evaluated into primitive [`int`]. Operations panic
//!    on overflow, division by zero, or negating/`abs` of `int::MIN`, in both debug and
//!    release builds.
//! 4. **Operators**: `+`, `-`, `*`, `/`, `%`, unary `-` (for [`Int<N>`]), and
//!    `==`, `!=`, `<`, `<=`, `>`, `>=` (under `#![feature(const_cmp)]`) are overloaded whenever
//!    at least one operand is [`Int<N>`].
//!
//! # Examples
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::{
//!     Int, Integer, add, ceil_div, floor_divmod, gcd, lcm, static_eq, static_true,
//! };
//!
//! // Static arithmetic produces static types and evaluates in const contexts:
//! const SUM: Int<5> = Int::<2> + Int::<3>;
//! assert_eq!(SUM, Int::<5>);
//! assert_eq!(-Int::<4>, Int::<-4>);
//! assert!(Int::<3> == 3i32);
//! assert!(Int::<-1> < Int::<0>);
//!
//! // Floor and ceiling divisions, plus number-theoretic helpers:
//! assert_eq!(floor_divmod(Int::<-7>, Int::<3>), (Int::<-3>, Int::<2>));
//! assert_eq!(ceil_div(Int::<7>, Int::<3>), Int::<3>);
//! assert_eq!(gcd(Int::<-12>, Int::<18>), Int::<6>);
//! assert_eq!(lcm(Int::<4>, Int::<-6>), Int::<12>);
//!
//! // Three-valued static predicates:
//! assert!(static_true(static_eq(add(Int::<2>, Int::<3>), Int::<5>)));
//! assert!(!static_true(static_eq(Int::<5>, 5i32)));
//! ```

use core::fmt::{Display, Formatter};

use crate::{NoodleCuteError, Result};

pub mod cmp;
pub mod ops;

pub use cmp::*;
pub use ops::*;

/// Signed runtime integer type used across CuTe.
#[allow(non_camel_case_types)]
pub type int = i32;

#[allow(non_camel_case_types)]
pub(crate) type uint = u32;

/// A signed integer whose value is known at compile time.
///
/// `Int<N>` is a zero-sized type (ZST) encoding its constant [`int`] value in the type system.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, Integer, StaticInteger};
///
/// let x = Int::<42>;
/// assert_eq!(x.value(), 42);
/// assert!(Int::<42>::IS_STATIC);
/// assert!(!i32::IS_STATIC);
/// assert_eq!(Int::<42>::STATIC_VALUE, Some(42));
/// assert_eq!(Int::<42>::VALUE, 42);
/// assert_eq!(Int::<2> * Int::<3>, Int::<6>);
/// ```
#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct Int<const N: int>;

impl<const N: int> Display for Int<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        Display::fmt(&N, f)
    }
}

mod sealed {
    use super::{Int, int};

    /// Seals [`Integer`](super::Integer) and [`StaticInteger`](super::StaticInteger) to `int` and `Int<N>`.
    #[doc(hidden)]
    pub trait SealedInteger {}
    impl SealedInteger for int {}
    impl<const N: int> SealedInteger for Int<N> {}
}

/// A sealed trait representing integers that can be either compile-time constants or run-time values.
///
/// Implementors are [`int`] (`i32`) and [`Int<N>`].
pub const trait Integer: sealed::SealedInteger + Copy + [const] Ord {
    /// The compile-time constant value if `Self` is static (`Some(N)` for [`Int<N>`], `None` for [`int`]).
    const STATIC_VALUE: Option<int>;

    /// `true` iff `Self` is a compile-time static integer (`STATIC_VALUE.is_some()`).
    const IS_STATIC: bool = Self::STATIC_VALUE.is_some();

    /// Returns the underlying native [`int`] value.
    fn value(self) -> int;
}

const impl Integer for int {
    const STATIC_VALUE: Option<int> = None;
    fn value(self) -> int {
        self
    }
}

const impl<const N: int> Integer for Int<N> {
    const STATIC_VALUE: Option<int> = Some(N);
    fn value(self) -> int {
        N
    }
}

/// A sealed trait implemented only by compile-time constant integers ([`Int<N>`]).
pub const trait StaticInteger: [const] Integer {
    /// The compile-time constant [`int`] value encoded by `Self`.
    const VALUE: int;
}
const impl<const N: int> StaticInteger for Int<N> {
    const VALUE: int = N;
}

/// Compile-time dispatch flag: `true` when both `A` and `B` are static `Int<N>`.
#[doc(hidden)]
pub const BOTH_STATIC<A: Integer, B: Integer>: bool = A::IS_STATIC && B::IS_STATIC;

/// Returns `true` when compile-time signed integer `N >= 0`.
#[doc(hidden)]
pub const IS_NON_NEG<const N: int>: bool = N >= 0;

pub(crate) const IS_ZERO<const N: int>: bool = N == 0;
pub(crate) const NOT_ZERO<const N: int>: bool = N != 0;
pub(crate) const IS_ONE<const N: int>: bool = N == 1;

// Saturating step counters for compile-time recursive trait bounds.
pub(crate) const DECREASE<const N: int>: int = N.saturating_sub(1);
pub(crate) const INCREASE<const N: int>: int = N.saturating_add(1);

pub(crate) const fn uint_to_int(n: uint) -> Result<int> {
    if n <= int::MAX as uint {
        Ok(n as int)
    } else {
        Err(NoodleCuteError::ArithmeticOverflow)
    }
}
