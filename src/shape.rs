//! Tensor shapes, coordinate conversions, and shape compatibility.
//!
//! In CuTe, a **shape** represents the domain of a multi-dimensional tensor. It is either
//! a non-negative scalar integer extent (dynamic [`int`] or static [`Int<N>`] with `N >= 0`),
//! the empty shape [`Nil`], or a nested cons-cell tuple of shapes
//! ([`Til`]).
//!
//! # Core Concepts & Submodules
//!
//! - **Shape abstraction ([`Shape`], [`ShapeScalar`], [`ShapeTuple`])**: Exposes
//!   [`size`](Shape::size) (the total product of extents) and [`has_zero`](Shape::has_zero).
//!   The empty shape [`Nil`] has size 1.
//! - [`mod@compatible`]: Partial-order coarsening ([`Compatible`], [`compatible()`]) and
//!   lattice join/meet operations ([`CommonRefinement`], [`common_refinement`],
//!   [`CommonCoarsening`], [`common_coarsening`]).
//! - [`mod@coord`]: Colexicographical conversions between natural coordinates and 1D linear
//!   indices ([`CoordToIndex`], [`crd2idx`], [`IndexToCoord`], [`idx2crd`]), bounds
//!   checking ([`InBounds`], [`in_bounds`]), and coordinate iteration ([`Coordinates`],
//!   [`coordinates`]).
//!
//! # Examples
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::shape::{Shape, compatible, crd2idx, idx2crd, size};
//!
//! // 1. Constructing shapes and computing size: (3, (2, 4))
//! let s = int_tuple![Int::<3>, (2i32, 4i32)];
//! assert_eq!(s.size(), 24);
//! assert_eq!(size(&s), 24);
//!
//! // 2. Natural coordinate to 1D linear index (colexicographical)
//! let crd = int_tuple![1i32, (0i32, 1i32)];
//! // idx = 1 + (0 + 1 * 2) * 3 = 7
//! assert_eq!(crd2idx(crd, &s), 7);
//!
//! // 3. 1D linear index to natural coordinate
//! assert_eq!(idx2crd(7i32, &s), crd);
//!
//! // 4. Shape compatibility (coarsening)
//! assert!(compatible(&24i32, &s));
//! assert!(compatible(&int_tuple![3i32, 8i32], &s));
//! assert!(!compatible(&s, &24i32)); // Tuples never match scalars
//! ```
//!
//! # Notes
//!
//! - Negative static extents (such as `Int<-1>`) are rejected at compile time:
//!
//! ```compile_fail,E0277
//! use noodle_cute::integer::Int;
//! use noodle_cute::shape::size;
//!
//! let _ = size(&Int::<-1>);
//! ```
//!
//! - Negative dynamic [`int`] extents panic at runtime when queried.

use crate::integer::{IS_NON_NEG, IS_ZERO, Int, Integer, int, is_pow2, mul, mul_maybe_overflows};
use crate::tuple::{FlatTuple, Item, Leaf, Nil, Til, Tuple};

pub mod compatible;
pub mod coord;

pub use compatible::*;
pub use coord::*;

/// Compile-time guard restricting static shape extents to non-negative values (`VALID == true`).
#[diagnostic::on_unimplemented(
    message = "static shape extent must be non-negative",
    label = "negative static extent in shape",
    note = "Int<N> shape extents must satisfy N >= 0"
)]
#[doc(hidden)]
pub const trait ValidShapeExtent<const VALID: bool> {}
const impl ValidShapeExtent<true> for () {}

/// A non-negative extent or a nested tuple of non-negative extents.
///
/// Implementors include [`Nil`], [`int`] (dynamic non-negative integer), [`Int<N>`]
/// (static non-negative integer), and [`Til<H, T>`](crate::tuple::Til)
/// where `H` and `T` are shapes.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::shape::Shape;
/// use noodle_cute::tuple::Nil;
///
/// assert_eq!(Nil.size(), 1);
/// assert_eq!(4i32.size(), 4);
/// assert_eq!(Int::<8>.size(), 8);
///
/// let s = int_tuple![Int::<2>, (3i32, Int::<5>)];
/// assert_eq!(s.size(), 30);
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid layout shape",
    label = "expected a valid shape",
    note = "use non-negative Int<N>, runtime int (i32), Nil, or nested tuples of shapes"
)]
pub const trait Shape: [const] Item {
    /// Compile-time total size when all leaf extents are static and fit in [`int`].
    const STATIC_SIZE: Option<int> = None;

    /// `true` iff any leaf extent is statically known to be zero.
    const STATIC_HAS_ZERO: bool = false;

    /// The product of all extents.
    ///
    /// The empty shape [`Nil`] has size one. If any leaf extent is zero, the total size is zero.
    /// Panics if any dynamic extent is negative or if the product overflows [`int`].
    fn size(&self) -> int;

    /// Returns `true` if any leaf extent in this shape is zero.
    ///
    /// Panics if any dynamic extent is negative.
    fn has_zero(&self) -> bool;
}

const impl Shape for Nil {
    const STATIC_SIZE: Option<int> = Some(1);
    const STATIC_HAS_ZERO: bool = false;
    fn size(&self) -> int {
        1
    }
    fn has_zero(&self) -> bool {
        false
    }
}

const impl Shape for int {
    const STATIC_SIZE: Option<int> = None;
    const STATIC_HAS_ZERO: bool = false;
    fn size(&self) -> int {
        assert!(*self >= 0, "shape extent must be non-negative");
        *self
    }
    fn has_zero(&self) -> bool {
        assert!(*self >= 0, "shape extent must be non-negative");
        *self == 0
    }
}

const impl<const N: int> Shape for Int<N>
where
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<N> }>,
{
    const STATIC_SIZE: Option<int> = Some(N);
    const STATIC_HAS_ZERO: bool = IS_ZERO::<N>;
    fn size(&self) -> int {
        N
    }
    fn has_zero(&self) -> bool {
        IS_ZERO::<N>
    }
}

const impl<H: [const] Shape, T: [const] Tuple + [const] Shape> Shape for Til<H, T> {
    const STATIC_SIZE: Option<int> = match (H::STATIC_SIZE, T::STATIC_SIZE) {
        (Some(0), _) | (_, Some(0)) => Some(0),
        (Some(a), Some(b)) => Some(mul(a, b)),
        _ => None,
    };
    const STATIC_HAS_ZERO: bool = H::STATIC_HAS_ZERO || T::STATIC_HAS_ZERO;
    fn size(&self) -> int {
        if self.has_zero() {
            0
        } else {
            mul(self.0.size(), self.1.size())
        }
    }
    fn has_zero(&self) -> bool {
        self.0.has_zero() | self.1.has_zero()
    }
}

/// Returns the total coordinate count [`Shape::size`] of `shape`.
pub const fn size<S: [const] Shape>(shape: &S) -> int {
    shape.size()
}

/// Returns `true` if any leaf extent in `shape` is zero.
pub const fn has_zero<S: [const] Shape>(shape: &S) -> bool {
    shape.has_zero()
}

mod sealed {
    use crate::integer::{IS_NON_NEG, Int, int};

    use super::ValidShapeExtent;

    /// Seals [`ShapeScalar`](super::ShapeScalar) to `int` and non-negative `Int<N>`.
    #[diagnostic::on_unimplemented(
        message = "`{Self}` is not a scalar shape extent",
        label = "expected a scalar shape leaf (non-negative `Int<N>` or `i32`)",
        note = "tuple shapes require a congruent tuple stride with matching structure"
    )]
    #[doc(hidden)]
    pub trait SealedShapeScalar {}
    impl SealedShapeScalar for int {}
    impl<const N: int> SealedShapeScalar for Int<N> where (): ValidShapeExtent<{ IS_NON_NEG::<N> }> {}
}

/// Sealed marker trait for scalar shape leaves (`int` and non-negative `Int<N>`).
pub const trait ShapeScalar:
    sealed::SealedShapeScalar + [const] Integer + [const] Leaf + [const] Shape
{
}
const impl<S: sealed::SealedShapeScalar + [const] Integer + [const] Leaf + [const] Shape>
    ShapeScalar for S
{
}

/// Marker trait for hierarchical tuple shapes (`Nil` or `Til<H, T>`).
pub const trait ShapeTuple: [const] Tuple + [const] Shape {}
const impl<S: [const] Tuple + [const] Shape> ShapeTuple for S {}

const SHAPE_LEAF_IS_STATIC_ZERO<S: ShapeScalar>: bool = matches!(S::STATIC_VALUE, Some(0));

/// Compile-time guard rejecting static zero shape leaves (`IS_ZERO == false`).
#[doc(hidden)]
pub const trait ValidPositiveShapeLeaf<const IS_ZERO: bool> {}
const impl ValidPositiveShapeLeaf<false> for () {}

/// Validates that every flattened shape leaf is strictly positive (`> 0`),
/// rejecting static `Int<0>` at compile time and panicking on dynamic `<= 0`.
#[doc(hidden)]
pub const trait AssertPositiveShape: FlatTuple {
    fn assert_positive_shape(self);
}

const impl AssertPositiveShape for Nil {
    fn assert_positive_shape(self) {}
}

const impl<H, T> AssertPositiveShape for Til<H, T>
where
    H: [const] ShapeScalar,
    T: [const] AssertPositiveShape,
    (): ValidPositiveShapeLeaf<{ SHAPE_LEAF_IS_STATIC_ZERO::<H> }>,
{
    fn assert_positive_shape(self) {
        if !<H as Integer>::IS_STATIC {
            assert!(self.0.value() > 0, "shape extent must be positive");
        }
        self.1.assert_positive_shape();
    }
}

/// Tests whether domain coordinates occupy disjoint binary fields.
#[doc(hidden)]
pub const trait BinaryExtents: Shape {
    const STATIC_BINARY: Option<bool>;
    fn binary_extents(self) -> bool;
    /// Returns the total domain size when every extent is a power of two and fits in `int`.
    fn binary_extents_size(self) -> Option<int>;
    /// Returns the total domain size when every extent before the last leaf is a power of two.
    fn leading_binary_size(self) -> Option<int>;
}
const impl<S: [const] ShapeScalar> BinaryExtents for S {
    const STATIC_BINARY: Option<bool> = match S::STATIC_VALUE {
        Some(s) => Some(is_pow2(s)),
        None => None,
    };
    fn binary_extents(self) -> bool {
        is_pow2(self.value())
    }
    fn binary_extents_size(self) -> Option<int> {
        let s = self.value();
        if is_pow2(s) { Some(s) } else { None }
    }
    fn leading_binary_size(self) -> Option<int> {
        let s = self.value();
        if s > 0 { Some(s) } else { None }
    }
}
const impl BinaryExtents for Nil {
    const STATIC_BINARY: Option<bool> = Some(true);
    fn binary_extents(self) -> bool {
        true
    }
    fn binary_extents_size(self) -> Option<int> {
        Some(1)
    }
    fn leading_binary_size(self) -> Option<int> {
        Some(1)
    }
}
const impl<H: [const] BinaryExtents, T: [const] BinaryExtents + Tuple> BinaryExtents for Til<H, T> {
    const STATIC_BINARY: Option<bool> = match (H::STATIC_BINARY, T::STATIC_BINARY) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    };
    fn binary_extents(self) -> bool {
        self.0.binary_extents() && self.1.binary_extents()
    }
    fn binary_extents_size(self) -> Option<int> {
        match (self.0.binary_extents_size(), self.1.binary_extents_size()) {
            (Some(a), Some(b)) => mul_maybe_overflows(a, b),
            _ => None,
        }
    }
    fn leading_binary_size(self) -> Option<int> {
        if T::LEAVES == 0 {
            self.0.leading_binary_size()
        } else {
            match (self.0.binary_extents_size(), self.1.leading_binary_size()) {
                (Some(a), Some(b)) => mul_maybe_overflows(a, b),
                _ => None,
            }
        }
    }
}
