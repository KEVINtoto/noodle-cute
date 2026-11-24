//! Strides, linear address mapping, and integer semi-module algebra.
//!
//! In CuTe, a **stride** $D$ pairs with a [`Shape`] $S$ to form a layout. While a shape
//! defines the domain (extents) of a tensor, its stride defines the mapping from natural
//! coordinates to memory addresses or logical coordinate vectors:
//!
//! $$\text{offset} = \langle c, d \rangle = \sum_k c_k \cdot d_k$$
//!
//! # Core Concepts & Submodules
//!
//! - **Integer & Basis Modules**: In standard array layouts, stride leaves are scalar integers
//!   representing physical memory increments. In CuTe, strides generalize to elements of an
//!   **integer module** ($\mathbb{Z}$-modules), admitting:
//!   - Physical integer strides: [`Integer`] (`i32` / [`int`](crate::integer::int), [`Int<N>`]).
//!   - Single basis terms: [`ScaledBasis<V, P>`](type@crate::basis::ScaledBasis) (alias for
//!     [`BasisTerm<V, P>`](crate::basis::BasisTerm)) representing $V$ times the unit step
//!     along coordinate path `P`.
//!   - Multi-term combinations: [`BasisCombination<T>`] representing sparse sums of
//!     [`BasisTerm`](crate::basis::BasisTerm) leaves.
//! - **Stride abstractions ([`Stride`], [`StrideScalar`], [`same_path`])**: Enforce that
//!   a stride has the exact same structural profile as the shape it pairs with
//!   ([`Item::Profile`] matching).
//! - [`mod@ops`]: Leaf stride scaling and accumulation ([`StrideMul`], [`Accumulate`],
//!   [`ScaleBasisFunctor`], [`ScaleStride`]).
//! - [`mod@product`]: Inner product ([`InnerProduct`], [`inner_product`]) and exclusive
//!   prefix product ([`PrefixProduct`], [`prefix_product`], [`prefix_product_with`]).
//! - [`mod@codomain`]: Codomain structural profile ([`CodomainProfile`]) and codomain
//!   shape bounding ([`Coshape`], [`coshape`]).
//!
//! # Examples
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::basis::{Mode, ScaledBasis};
//! use noodle_cute::integer::Int;
//! use noodle_cute::stride::{inner_product, prefix_product};
//! use noodle_cute::{int_tuple, tuple};
//!
//! // 1. Computing default column-major compact strides: (3, 4) -> (1, 3)
//! let s = int_tuple![Int::<3>, Int::<4>];
//! let d = prefix_product(s);
//! assert_eq!(d, int_tuple![Int::<1>, Int::<3>]);
//!
//! // 2. Linear memory address mapping via inner product:
//! // For coordinate (1, 2) in shape (3, 4) with stride (1, 3):
//! // offset = 1 * 1 + 2 * 3 = 7
//! let crd = int_tuple![1i32, 2i32];
//! let strd = int_tuple![1i32, 3i32];
//! assert_eq!(inner_product(crd, strd), 7i32);
//!
//! // 3. Single-basis scaling via inner product: 2 * (1@0) = 2@0
//! let basis0 = ScaledBasis::<Int<1>, Mode<0>>::new(Int::<1>);
//! assert_eq!(inner_product(2i32, basis0), ScaledBasis::<i32, Mode<0>>::new(2i32));
//! ```
//!
//! # Notes
//!
//! - [`Stride<S>`] enforces structural congruence (`Self::Profile == S::Profile`) at compile time.
//! - Mixing prefix-conflicting basis stride paths during accumulation or codomain queries
//!   is rejected at compile time.
//! - Arithmetic overflow follows [`crate::integer`] rules (compile-time error for static
//!   operands, runtime panic for dynamic operands).

use crate::basis::{
    BasisCoeff, BasisCombination, BasisPath, ComparePath, ModeEnd, PathOrdering, ScaledBasis,
};
use crate::integer::{Int, Integer};
use crate::shape::{Shape, ShapeScalar};
use crate::tuple::{Item, Leaf, Nil, Til, Tuple};

pub mod codomain;
pub mod ops;
pub mod product;

pub use codomain::*;
pub use ops::*;
pub use product::*;

/// Returns `true` when two coordinate paths are identical at compile time.
pub const fn same_path<P1: ComparePath<P2>, P2: BasisPath>() -> bool {
    matches!(P1::ORDER, PathOrdering::Equal)
}

/// A single-term element occurring as a leaf in a [`Stride`].
///
/// In CuTe, a stride scalar is either an ordinary integer (with [`Path`](StrideScalar::Path)
/// equal to [`ModeEnd`]) or a single scaled basis term [`ScaledBasis<V, P>`]
/// representing a scalar coefficient along coordinate path `P`.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use core::any::TypeId;
/// use noodle_cute::basis::{Mode, ModeEnd, ScaledBasis};
/// use noodle_cute::integer::Int;
/// use noodle_cute::stride::StrideScalar;
///
/// // Ordinary integers have Path = ModeEnd
/// assert_eq!(5i32.value(), 5i32);
/// assert_eq!(TypeId::of::<<i32 as StrideScalar>::Path>(), TypeId::of::<ModeEnd>());
///
/// // Scaled basis vector: 3 * E(0)
/// let sb = ScaledBasis::<Int<3>, Mode<0>>::new(Int::<3>);
/// assert_eq!(sb.value(), Int::<3>);
/// assert_eq!(
///     TypeId::of::<<ScaledBasis<Int<3>, Mode<0>> as StrideScalar>::Path>(),
///     TypeId::of::<Mode<0>>()
/// );
/// ```
pub const trait StrideScalar: [const] Leaf {
    /// The coefficient type representing the magnitude of the stride.
    type Value: [const] BasisCoeff;

    /// The coordinate path type, or [`ModeEnd`] if this is a conventional integer stride.
    type Path: [const] BasisPath;

    /// Returns the coefficient value.
    fn value(self) -> Self::Value;
}

const impl<I: [const] Integer> StrideScalar for I {
    type Value = Self;
    type Path = ModeEnd;
    fn value(self) -> Self::Value {
        self
    }
}

const impl<V: [const] BasisCoeff, P: [const] BasisPath> StrideScalar for ScaledBasis<V, P> {
    type Value = V;
    type Path = P;
    fn value(self) -> Self::Value {
        self.value
    }
}

const impl StrideScalar for BasisCombination<Nil> {
    type Value = Int<0>;
    type Path = ModeEnd;
    fn value(self) -> Self::Value {
        Int
    }
}

/// A stride for a given shape `S`, matching its hierarchical profile.
///
/// A stride must have the identical nesting structure as the shape it pairs with:
/// `Self::Profile = S::Profile`. Integer shape leaves accept any [`Leaf`] stride
/// ([`Integer`], [`ScaledBasis`], or [`BasisCombination`]).
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::basis::{Mode, ScaledBasis};
/// use noodle_cute::integer::Int;
/// use noodle_cute::shape::Shape;
/// use noodle_cute::stride::Stride;
/// use noodle_cute::tuple::{Nil, TupleContainer};
///
/// fn check_stride<S: Shape, D: Stride<S>>(_: &S, _: &D) {}
///
/// check_stride(&Nil, &Nil);
/// check_stride(&4i32, &1i32);
/// check_stride(&4i32, &ScaledBasis::<Int<1>, Mode<0>>::new(Int::<1>));
/// check_stride(
///     &TupleContainer::new(Int::<2>, TupleContainer::new(3i32, Nil)),
///     &TupleContainer::new(1i32, TupleContainer::new(2i32, Nil)),
/// );
/// ```
pub const trait Stride<S: Shape>: [const] Item<Profile = S::Profile> {}
const impl Stride<Nil> for Nil {}
const impl<D: [const] Leaf, S: ShapeScalar> Stride<S> for D {}
const impl<H, T, SH, ST> Stride<Til<SH, ST>> for Til<H, T>
where
    SH: Shape,
    ST: Tuple + Shape,
    H: [const] Stride<SH>,
    T: [const] Tuple + [const] Stride<ST>,
{
}

/// Classifies stride trees containing a binary-polynomial coefficient.
#[doc(hidden)]
pub trait ContainsF2 {
    const HAS_F2: bool;
    const HAS_INTEGER: bool = false;
}
impl<V: BasisCoeff + Leaf> ContainsF2 for V {
    const HAS_INTEGER: bool = matches!(V::ALGEBRA, crate::basis::CoeffAlgebra::Integer);
    const HAS_F2: bool = matches!(V::ALGEBRA, crate::basis::CoeffAlgebra::F2);
}
impl<V: BasisCoeff, P> ContainsF2 for ScaledBasis<V, P> {
    const HAS_F2: bool = matches!(V::ALGEBRA, crate::basis::CoeffAlgebra::F2);
}
impl<T: ContainsF2 + crate::basis::BasisTuple> ContainsF2 for BasisCombination<T> {
    const HAS_F2: bool = T::HAS_F2;
}
impl ContainsF2 for Nil {
    const HAS_F2: bool = false;
}
impl<H: ContainsF2, T: ContainsF2> ContainsF2 for Til<H, T> {
    const HAS_F2: bool = H::HAS_F2 || T::HAS_F2;
    const HAS_INTEGER: bool = H::HAS_INTEGER || T::HAS_INTEGER;
}
