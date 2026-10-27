//! Device-independent building blocks for CuTe-style layouts.
//!
//! Integers may be encoded in types or stored at runtime; nested tuples preserve
//! their structure. Checked operations share [`Error`] and [`Result`].

#![no_std]
#![feature(
    gca_min_const_items,
    gca_const_items,
    gca_macroless_args,
    generic_const_items
)]
#![allow(incomplete_features)]

pub mod int;
pub mod tuple;

pub use int::{DynInt, Int, Integer};
pub use tuple::{IntTuple, Nil};

/// Errors reported by checked integer, tuple, and layout operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// A shape extent or another size parameter is invalid for the operation.
    InvalidShape,
    /// Shapes, strides, or coordinates have incompatible nesting or axis counts.
    StructureMismatch,
    /// A logical coordinate lies outside the layout's domain.
    CoordinateOutOfBounds,
    /// An axis, top-level mode, or flattened leaf index is out of bounds.
    AxisOutOfBounds,
    /// Checked arithmetic or an integer conversion exceeds the destination type's range.
    Overflow,
    /// Distinct logical coordinates map to the same address when uniqueness is required.
    NonInjective,
    /// The layout does not cover the codomain required by the operation.
    NotSurjective,
    /// An extent or stride fails a divisibility requirement.
    NotDivisible,
    /// The requested operation is not implemented for this layout or parameter combination.
    Unsupported,
    /// An address lies outside the permitted storage range.
    AddressOutOfBounds,
    /// Operands have incompatible logical sizes.
    SizeMismatch,
    /// Runtime layout metadata exceeds the capacity of its fixed-size storage.
    CapacityExceeded,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let message = match self {
            Self::InvalidShape => "shape extents must be nonnegative",
            Self::StructureMismatch => "shape, stride, or coordinate structures differ",
            Self::CoordinateOutOfBounds => "coordinate is outside the layout domain",
            Self::AxisOutOfBounds => "axis is outside the layout rank",
            Self::Overflow => "integer overflow",
            Self::NonInjective => "layout maps multiple coordinates to the same address",
            Self::NotSurjective => "layout does not cover the requested codomain",
            Self::NotDivisible => "layout or tile extent is not divisible",
            Self::Unsupported => "operation is unsupported for this layout",
            Self::AddressOutOfBounds => "address is outside storage",
            Self::SizeMismatch => "logical sizes differ",
            Self::CapacityExceeded => "layout exceeds fixed metadata capacity",
        };
        f.write_str(message)
    }
}

impl core::error::Error for Error {}

/// The result of a checked operation, using this crate's [`Error`] type.
pub type Result<T> = core::result::Result<T, Error>;

/// Carries a compile-time condition as a type for trait bounds.
#[doc(hidden)]
pub struct Assert<const B: bool>;
/// Marks a compile-time condition that holds.
#[doc(hidden)]
pub trait True {}
// Only `Assert<true>` satisfies this bound, so invalid static extents fail trait checking.
impl True for Assert<true> {}
