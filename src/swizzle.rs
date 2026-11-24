//! Layout swizzling primitives: binary-field stride scalar [`F2`] and bit-field functor [`Swizzle`].
//!
//! A swizzle permutes the offsets a layout produces to spread accesses across shared-memory
//! banks:
//! - [`F2`] expresses a swizzle as a stride over $\mathbb{F}_2\[x\]$ (`+` is XOR, `*` is
//!   carry-less multiplication), allowing `F2`-strided [`Layout`](crate::layout::Layout)s
//!   to pass directly through the layout algebra.
//! - [`Swizzle`] expresses the same bit-field XOR permutation as a direct functor on integer
//!   or [`F2`] offsets.
//!
//! # Main Interfaces
//!
//! - [`mod@f2`]: [`F2`], [`StaticF2`], [`F2Scalar`], [`IntoF2`], [`f2()`], [`f2_add`],
//!   [`f2_sub`], [`f2_mul`], [`f2_divmod`], and [`crate::f2!`].
//! - [`mod@functor`]: [`Swizzle`], [`StaticSwizzle`], [`SwizzleApply`], and [`crate::swizzle!`].
//! - [`shiftl`] / [`shiftr`]: Signed shifts re-exported from [`crate::integer`].
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::swizzle::{Swizzle, f2, shiftr};
//! use noodle_cute::{int_tuple, tuple};
//!
//! let sw = Swizzle::new(Int::<3>, Int::<0>, Int::<3>);
//! let layout = Layout::new(
//!     int_tuple![Int::<8>, Int::<8>],
//!     tuple![f2(Int::<1>), f2(Int::<9>)],
//! );
//! assert_eq!(layout(2i32, 5i32), sw(2i32 + 5i32 * 8i32));
//! assert_eq!(shiftr(0b1000, -2), 0b100000);
//! ```
//!
//! # Compile-Time Rejections
//!
//! Carrying prefix products in `idx2crd(F2, shape)` fail at compile time for static shapes:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::shape::idx2crd;
//! use noodle_cute::swizzle::f2;
//!
//! let _ = idx2crd(f2(Int::<0b10110>), &int_tuple![Int::<3>, Int::<3>, Int::<3>]);
//! ```
//!
//! Codomain bit-span overflow (`1 << 31` exceeds `int::MAX`) fails at compile time:
//!
//! ```compile_fail,E0080
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::stride::coshape;
//! use noodle_cute::swizzle::f2;
//!
//! let _ = coshape(Int::<2>, f2(Int::<1_073_741_824>));
//! ```
//!
//! Non-unit-gap `F2` layouts are rejected by `left_inverse`, and `F2` strides are rejected
//! by `complement` at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::left_inverse;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::swizzle::f2;
//! use noodle_cute::{int_tuple, tuple};
//!
//! let _ = left_inverse(Layout::new(
//!     int_tuple![Int::<8>, Int::<8>],
//!     tuple![f2(Int::<1>), f2(Int::<9>)],
//! ));
//! ```
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::complement;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::swizzle::f2;
//!
//! let _ = complement(Layout::new(Int::<4>, f2(Int::<2>)));
//! ```
//!
//! Mixing integer and `F2` strides on the same codomain axis fails `right_inverse` at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::right_inverse;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::swizzle::f2;
//! use noodle_cute::{int_tuple, tuple};
//!
//! let _ = right_inverse(Layout::new(
//!     int_tuple![Int::<4>, Int::<4>],
//!     tuple![Int::<1>, f2(Int::<4>)],
//! ));
//! ```

pub use crate::integer::{shiftl, shiftr};

pub mod f2;
mod f2_traits;
pub mod functor;

pub use f2::*;
pub use functor::*;

/// Constructs an [`F2`] value idempotently from an integer or existing [`F2`].
#[macro_export]
macro_rules! f2 {
    ($val:expr $(,)?) => {
        $crate::swizzle::f2($val)
    };
}

/// Constructs a validated [`Swizzle`] from `(bits, base, shift)` expressions.
#[macro_export]
macro_rules! swizzle {
    ($bits:expr, $base:expr, $shift:expr $(,)?) => {
        $crate::swizzle::Swizzle::new($bits, $base, $shift)
    };
}
