//! Device-independent building blocks for CuTe-style layouts.
//!
//! Integers may be encoded in types or stored at runtime; nested tuples preserve
//! their structure. Checked operations share [`Error`] and [`Result`].
//!
//! Use [`layout::DynLayout`] for runtime metadata or [`layout::Layout`] to keep
//! static values and nested modes in types. Indexing returns address offsets;
//! layouts do not own or access a backing allocation.
//!
//! ```
//! use noodle_cute::layout::{DynLayout, LayoutMap};
//!
//! let layout = DynLayout::left(&[2, 3])?;
//! assert_eq!(layout.strides(), &[1, 2]);
//! assert_eq!(layout.index_of_coord(&[1, 2])?, 5);
//! assert_eq!(layout.size()?, 6);
//! # Ok::<(), noodle_cute::Error>(())
//! ```

#![no_std]
#![feature(
    gca_min_const_items,
    gca_const_items,
    gca_macroless_args,
    generic_const_items
)]
#![allow(incomplete_features)]

pub mod algebra;
pub mod int;
pub mod layout;
mod slice;
pub mod tuple;

pub use int::{DynInt, Int, Integer};
pub use layout::Layout;
pub use slice::Slice;
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

/// 可由类型唯一确定的值；用于全静态布局代数的结果构造。
#[doc(hidden)]
pub trait StaticValue: Copy {
    const VALUE: Self;
}
impl<const N: isize> StaticValue for Int<N> {
    const VALUE: Self = Int;
}
impl StaticValue for Nil {
    const VALUE: Self = Nil;
}
impl<H: StaticValue, T: StaticValue> StaticValue for IntTuple<H, T> {
    const VALUE: Self = IntTuple(H::VALUE, T::VALUE);
}
impl<S: StaticValue, D: StaticValue> StaticValue for Layout<S, D> {
    const VALUE: Self = Layout {
        shape: S::VALUE,
        stride: D::VALUE,
    };
}
