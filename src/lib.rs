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
    const_trait_impl,
    const_try,
    decl_macro,
    trait_alias,
    adt_const_params,
    min_adt_const_params,
    gca_min_const_items,
    gca_const_items,
    gca_macroless_args,
    generic_const_items
)]
#![allow(incomplete_features)]

pub mod algebra;
pub mod algorithm;
pub mod atom;
pub mod basis;
pub mod coord;
pub mod int;
pub mod layout;
pub mod numeric;
pub mod pointer;
mod slice;
pub mod swizzle;
pub mod tensor;
pub mod tuple;

pub use int::{DynInt, Int, Integer};
pub use layout::{Layout, LayoutDesc, StaticDesc};
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
            Self::InvalidShape => "invalid shape extent or size parameter",
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

mod marker {
    /// Carries a compile-time condition as a type for trait bounds and impl dispatch.
    pub struct Bool<const B: bool>;
    /// Implemented only by `Bool<true>`, so a false static condition fails trait checking.
    pub trait True {}
    impl True for Bool<true> {}
}
pub(crate) use marker::{Bool, True};

/// A value fully determined by its type.
///
/// Every leaf of a fully static `Layout` is an `Int<N>`, so typed algebra results such as
/// [`algebra::TypedComposition`] can be read as constants:
///
/// ```
/// use noodle_cute::{Int, StaticValue, algebra::TypedComposition, layout::Layout};
/// const C: Layout<Int<3>, Int<4>> =
///     <TypedComposition<Layout<Int<8>, Int<2>>, Layout<Int<3>, Int<2>>> as StaticValue>::VALUE;
/// ```
///
/// Unlike the checked methods, a constant read this way skips the final runtime size and
/// address-range validation; any static arithmetic error is rejected at compile time instead.
#[diagnostic::on_unimplemented(
    message = "type `{Self}` does not determine a constant value",
    label = "runtime values cannot be read through StaticValue",
    note = "use Int<N> leaves for type-determined constants; DynInt and isize store runtime values"
)]
pub trait StaticValue: Copy {
    /// The value encoded by the type.
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
