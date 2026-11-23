//! CuTe library implementation in Rust.

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

pub mod integer;

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
