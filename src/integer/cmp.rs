//! Integer comparisons, three-valued static predicates, and extrema (`min` / `max`).
//!
//! Provides heterogeneous `const` comparisons across [`Int<N>`] and [`int`],
//! type- and value-level static predicates returning `Option<bool>` / `Option<Ordering>`,
//! and staticness-preserving [`min`] / [`max`].
//!
//! # Main Interfaces
//!
//! - **Value comparisons**: [`cmp`], [`eq`], [`ne`], [`lt`], [`le`], [`gt`], [`ge`],
//!   plus `const` `PartialEq` / `Eq` / `PartialOrd` / `Ord` implementations across
//!   [`Int<N>`] and [`int`].
//! - **Three-valued static logic**: [`static_true`], [`static_false`], [`static_not`],
//!   [`static_and`], [`static_or`].
//! - **Static comparisons**: type-level [`static_eq_of`], [`static_lt_of`],
//!   [`static_le_of`], [`static_cmp_of`]; value-level [`static_eq`], [`static_ne`],
//!   [`static_lt`], [`static_le`], [`static_gt`], [`static_ge`], [`static_cmp`];
//!   and boolean consts [`STATIC_TRUE_EQ`], [`STATIC_TRUE_LT`].
//! - **Extrema**: [`min`], [`max`], and output type aliases [`Min`], [`Max`].
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::{Int, max, min, static_lt, static_true};
//!
//! assert!(Int::<-1> < 0i32);
//! assert_eq!(min(Int::<2>, Int::<5>), Int::<2>);
//! assert_eq!(max(Int::<2>, 5i32), 5i32);
//! assert!(static_true(static_lt(Int::<2>, Int::<5>)));
//! assert_eq!(static_lt(Int::<2>, 5i32), None);
//! ```
//!
//! # Notes
//!
//! - Static predicates return `None` whenever either operand is dynamic;
//!   [`static_true`] and [`static_false`] map `None` to `false`.
//! - Using `==`, `<`, etc. on [`Int<N>`] inside `const` contexts requires
//!   `#![feature(const_cmp)]`.

use core::cmp::Ordering;

use super::{BOTH_STATIC, Int, Integer, StaticInteger, int};

const impl<const N: int, B: [const] Integer> PartialEq<B> for Int<N> {
    fn eq(&self, b: &B) -> bool {
        eq(*self, *b)
    }
}

const impl<const N: int> Eq for Int<N> {}

const impl<const N: int, B: [const] Integer> PartialOrd<B> for Int<N> {
    fn partial_cmp(&self, b: &B) -> Option<Ordering> {
        Some(cmp(*self, *b))
    }
}

const impl<const N: int> Ord for Int<N> {
    fn cmp(&self, _: &Self) -> Ordering {
        Ordering::Equal
    }
}

const impl<const N: int> PartialEq<Int<N>> for int {
    fn eq(&self, _: &Int<N>) -> bool {
        *self == N
    }
}

const impl<const N: int> PartialOrd<Int<N>> for int {
    fn partial_cmp(&self, _: &Int<N>) -> Option<Ordering> {
        self.partial_cmp(&N)
    }
}

/// Compares any two CuTe integers mathematically in `const` or runtime contexts.
///
/// Handles any combination of static [`Int<N>`] and dynamic [`int`] values.
pub const fn cmp<A: [const] Integer, B: [const] Integer>(a: A, b: B) -> Ordering {
    a.value().cmp(&b.value())
}

/// Returns `true` iff `a` and `b` represent the same mathematical integer value.
pub const fn eq<A: [const] Integer, B: [const] Integer>(a: A, b: B) -> bool {
    a.value().eq(&b.value())
}

/// Returns `true` iff `a` and `b` represent distinct mathematical integer values.
pub const fn ne<A: [const] Integer, B: [const] Integer>(a: A, b: B) -> bool {
    !eq(a, b)
}

/// Returns `true` iff `a < b` mathematically.
pub const fn lt<A: [const] Integer, B: [const] Integer>(a: A, b: B) -> bool {
    matches!(cmp(a, b), Ordering::Less)
}

/// Returns `true` iff `a <= b` mathematically.
pub const fn le<A: [const] Integer, B: [const] Integer>(a: A, b: B) -> bool {
    !matches!(cmp(a, b), Ordering::Greater)
}

/// Returns `true` iff `a > b` mathematically.
pub const fn gt<A: [const] Integer, B: [const] Integer>(a: A, b: B) -> bool {
    matches!(cmp(a, b), Ordering::Greater)
}

/// Returns `true` iff `a >= b` mathematically.
pub const fn ge<A: [const] Integer, B: [const] Integer>(a: A, b: B) -> bool {
    !matches!(cmp(a, b), Ordering::Less)
}

/// Returns `true` only when a three-valued static predicate is definitely `Some(true)`.
pub const fn static_true(cond: Option<bool>) -> bool {
    matches!(cond, Some(true))
}

/// Returns `true` only when a three-valued static predicate is definitely `Some(false)`.
pub const fn static_false(cond: Option<bool>) -> bool {
    matches!(cond, Some(false))
}

/// Three-valued logical negation.
pub const fn static_not(cond: Option<bool>) -> Option<bool> {
    match cond {
        Some(b) => Some(!b),
        None => None,
    }
}

/// Three-valued logical conjunction.
pub const fn static_and(lhs: Option<bool>, rhs: Option<bool>) -> Option<bool> {
    match (lhs, rhs) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

/// Three-valued logical disjunction.
pub const fn static_or(lhs: Option<bool>, rhs: Option<bool>) -> Option<bool> {
    match (lhs, rhs) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

/// Type-level equality check returning `None` if either operand is dynamic.
pub const fn static_eq_of<A: Integer, B: Integer>() -> Option<bool> {
    if let (Some(a), Some(b)) = (A::STATIC_VALUE, B::STATIC_VALUE) {
        Some(eq(a, b))
    } else {
        None
    }
}

/// Type-level strict inequality `A < B`, returning `None` if either operand is dynamic.
pub const fn static_lt_of<A: Integer, B: Integer>() -> Option<bool> {
    if let (Some(a), Some(b)) = (A::STATIC_VALUE, B::STATIC_VALUE) {
        Some(lt(a, b))
    } else {
        None
    }
}

/// Type-level inequality `A <= B`, returning `None` if either operand is dynamic.
pub const fn static_le_of<A: Integer, B: Integer>() -> Option<bool> {
    static_not(static_lt_of::<B, A>())
}

/// Type-level ordering between `A` and `B`, returning `None` if either operand is dynamic.
pub const fn static_cmp_of<A: Integer, B: Integer>() -> Option<Ordering> {
    if static_true(static_lt_of::<A, B>()) {
        Some(Ordering::Less)
    } else if static_true(static_eq_of::<A, B>()) {
        Some(Ordering::Equal)
    } else if static_true(static_lt_of::<B, A>()) {
        Some(Ordering::Greater)
    } else {
        None
    }
}

/// Checks if two integer values are guaranteed equal from their static types alone.
pub const fn static_eq<A: Integer, B: Integer>(_: A, _: B) -> Option<bool> {
    static_eq_of::<A, B>()
}

/// Checks if two integer values are guaranteed unequal from their static types alone.
pub const fn static_ne<A: Integer, B: Integer>(_: A, _: B) -> Option<bool> {
    static_not(static_eq_of::<A, B>())
}

/// Checks if `a < b` is guaranteed from the static types of `A` and `B`.
pub const fn static_lt<A: Integer, B: Integer>(_: A, _: B) -> Option<bool> {
    static_lt_of::<A, B>()
}

/// Checks if `a <= b` is guaranteed from the static types of `A` and `B`.
pub const fn static_le<A: Integer, B: Integer>(_: A, _: B) -> Option<bool> {
    static_le_of::<A, B>()
}

/// Checks if `a > b` is guaranteed from the static types of `A` and `B`.
pub const fn static_gt<A: Integer, B: Integer>(_: A, _: B) -> Option<bool> {
    static_lt_of::<B, A>()
}

/// Checks if `a >= b` is guaranteed from the static types of `A` and `B`.
pub const fn static_ge<A: Integer, B: Integer>(_: A, _: B) -> Option<bool> {
    static_le_of::<B, A>()
}

/// Returns the static ordering between `a` and `b` if guaranteed by their static types.
pub const fn static_cmp<A: Integer, B: Integer>(_: A, _: B) -> Option<Ordering> {
    static_cmp_of::<A, B>()
}

/// `true` iff `A` and `B` are statically known to be equal (`false` when either operand is dynamic).
pub const STATIC_TRUE_EQ<A: Integer, B: Integer>: bool = static_true(static_eq_of::<A, B>());

/// `true` iff `A` is statically known to be strictly less than `B` (`false` when either operand is dynamic).
pub const STATIC_TRUE_LT<A: Integer, B: Integer>: bool = static_true(static_lt_of::<A, B>());

const STATIC_MINIMUM<A: StaticInteger, B: StaticInteger>: int = A::VALUE.min(B::VALUE);
const STATIC_MAXIMUM<A: StaticInteger, B: StaticInteger>: int = A::VALUE.max(B::VALUE);

// Dispatches `min`/`max` between `Self` and `B`: `BOTH_STATIC = true` evaluates
// at compile time to `Int<R>`, `false` evaluates dynamically to `int`.
#[doc(hidden)]
pub const trait ExtremaOp<B: [const] Integer, const BOTH_STATIC: bool>:
    [const] Integer
{
    type MinOutput: [const] Integer;
    type MaxOutput: [const] Integer;
    fn apply_min(self, b: B) -> Self::MinOutput;
    fn apply_max(self, b: B) -> Self::MaxOutput;
}

const impl<A: [const] StaticInteger, B: [const] StaticInteger> ExtremaOp<B, true> for A {
    type MinOutput = Int<{ STATIC_MINIMUM::<A, B> }>;
    type MaxOutput = Int<{ STATIC_MAXIMUM::<A, B> }>;
    fn apply_min(self, _: B) -> Self::MinOutput {
        Int
    }
    fn apply_max(self, _: B) -> Self::MaxOutput {
        Int
    }
}

const impl<A: [const] Integer, B: [const] Integer> ExtremaOp<B, false> for A {
    type MinOutput = int;
    type MaxOutput = int;
    fn apply_min(self, b: B) -> Self::MinOutput {
        let (a, b) = (self.value(), b.value());
        if a <= b { a } else { b }
    }
    fn apply_max(self, b: B) -> Self::MaxOutput {
        let (a, b) = (self.value(), b.value());
        if a >= b { a } else { b }
    }
}

/// Output type of `min(A, B)`.
pub type Min<A, B> = <A as ExtremaOp<B, { BOTH_STATIC::<A, B> }>>::MinOutput;

/// Output type of `max(A, B)`.
pub type Max<A, B> = <A as ExtremaOp<B, { BOTH_STATIC::<A, B> }>>::MaxOutput;

/// Returns the minimum of two CuTe integers.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, min};
///
/// assert_eq!(min(Int::<-2>, Int::<5>), Int::<-2>);
/// assert_eq!(min(Int::<5>, 2i32), 2i32);
/// ```
pub const fn min<A, B>(a: A, b: B) -> Min<A, B>
where
    A: [const] Integer + [const] ExtremaOp<B, { BOTH_STATIC::<A, B> }>,
    B: [const] Integer,
{
    a.apply_min(b)
}

/// Returns the maximum of two CuTe integers.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::{Int, max};
///
/// assert_eq!(max(Int::<3>, Int::<9>), Int::<9>);
/// assert_eq!(max(3i32, Int::<9>), 9i32);
/// ```
pub const fn max<A, B>(a: A, b: B) -> Max<A, B>
where
    A: [const] Integer + [const] ExtremaOp<B, { BOTH_STATIC::<A, B> }>,
    B: [const] Integer,
{
    a.apply_max(b)
}
