//! Exact hierarchical layout and tensor algebra in Rust (`no_std`).
//!
//! The crate is organized in layers from low-level arithmetic primitives up to
//! hierarchical tuples and sparse coordinate bases:
//!
//! - [`integer`]: Static ([`Int<N>`](integer::Int)) and dynamic ([`int`](integer::int))
//!   signed integers, `const` comparisons, three-valued static predicates, and checked
//!   arithmetic.
//! - [`mod@slice`]: `const`-capable Python-style slice specifications ([`Slice`](slice::Slice),
//!   [`Bound`](slice::Bound)) and dimension resolution ([`slice!`], [`slices!`]).
//! - [`mod@tuple`]: Hierarchical tuples ([`TupleContainer`](tuple::TupleContainer),
//!   [`Nil`](tuple::Nil)), structural congruence, mode indexing/slicing, restructuring,
//!   synchronized multi-tree leaf traversal, and product reductions
//!   ([`tuple!`], [`int_tuple!`], [`mode!`], [`walk_mode!`]).
//! - [`basis`]: Sparse, path-sorted coordinate tuples and basis vectors
//!   ([`Basis`](basis::Basis), [`BasisTerm`](basis::BasisTerm),
//!   [`BasisCombination`](basis::BasisCombination)) with linear arithmetic, zero-padded
//!   equality, colexicographical ordering, and tuple conversions
//!   ([`e!`], [`basis_tuple!`]).
//!
//! # Main Root Interfaces
//!
//! - [`NoodleCuteError`]: Error enum for integer overflow, division by zero, and
//!   out-of-range conversions.
//! - [`Result<T>`]: Crate-wide `Result` alias using [`NoodleCuteError`].
//! - [`NoodleCuteResult`]: `const`-capable `.v()` helper that unwraps `Ok` or panics
//!   with the error message on `Err`.
//!
//! # Examples
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{basis::as_tuple, e, int_tuple, integer::Int, walk_mode};
//!
//! const SUM: Int<5> = Int::<2> + Int::<3>;
//! let t = int_tuple![SUM, (Int::<4>, 6i32)];
//! assert_eq!(walk_mode!(t, 1, 0), Int::<4>);
//! assert_eq!(as_tuple(Int::<2> * e![0] + 3i32 * e![1]), int_tuple![Int::<2>, 3i32]);
//! ```
//!
//! # Notes
//!
//! - `#![no_std]` crate requiring nightly Rust features for generic `const` arguments
//!   and `const` trait implementations.
//! - Arithmetic and algebraic operations expose one unified interface: invalid static
//!   operations fail at compile time, while invalid dynamic operations panic at runtime
//!   in both debug and release builds.

#![no_std]
#![feature(
    min_adt_const_params,
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    trait_alias,
    const_trait_impl,
    const_ops,
    const_cmp,
    const_option_ops,
    const_try
)]
#![allow(incomplete_features)]

pub mod basis;
pub mod integer;
pub mod slice;
pub mod tuple;

/// Errors that can occur during integer arithmetic and conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoodleCuteError {
    /// An arithmetic operation produced a value outside the representable [`int`](integer::int) range.
    ArithmeticOverflow,
    /// Division or remainder was attempted with a zero divisor.
    DivisionByZero,
    /// Negating or taking the absolute value of `int::MIN` (`i32::MIN`) would overflow.
    SignedMinValueOverflow,
    /// An unsigned value cannot be represented by the signed [`int`](integer::int) type.
    UnsignedToSignedOutOfRange,
}

impl NoodleCuteError {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ArithmeticOverflow => "integer arithmetic overflow",
            Self::DivisionByZero => "integer division by zero",
            Self::SignedMinValueOverflow => "cannot negate or take the absolute value of int::MIN",
            Self::UnsignedToSignedOutOfRange => "unsigned integer is out of range for signed int",
        }
    }
}

impl core::fmt::Display for NoodleCuteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl core::error::Error for NoodleCuteError {}

/// Result type alias using [`NoodleCuteError`].
pub type Result<T> = core::result::Result<T, NoodleCuteError>;

/// Helper trait for unwrapping [`Result`] values in both `const` and runtime contexts.
pub const trait NoodleCuteResult {
    /// The unwrapped success value type.
    type Output;

    /// Returns the `Ok` value, panicking with the error description on `Err`.
    fn v(self) -> Self::Output;
}

const impl<T: Copy> NoodleCuteResult for Result<T> {
    type Output = T;
    fn v(self) -> Self::Output {
        match self {
            Ok(t) => t,
            Err(e) => panic!("{}", e.as_str()),
        }
    }
}
