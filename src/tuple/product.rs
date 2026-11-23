//! Product reductions over hierarchical tuples (`TupleProduct`, `ProductEach`, `ProductLike`).
//!
//! Multiplies integer leaves across an entire tuple hierarchy, per top-level mode,
//! or per subtree aligned with the leaf positions of a structural guide.
//!
//! # Main Interfaces
//!
//! - [`TupleProduct`] / [`tuple_product`]: Recursively multiplies all scalar leaves.
//! - [`ProductEach`] / [`product_each`]: Applies [`TupleProduct`] to each top-level mode.
//! - [`ProductLike`] / [`product_like`]: Contracts each subtree matching a leaf of `guide`
//!   into its scalar product.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{int_tuple, integer::Int};
//! use noodle_cute::tuple::{product_each, product_like, tuple_product};
//!
//! let shape = int_tuple![(Int::<2>, (Int::<3>, Int::<4>)), Int::<5>];
//! assert_eq!(tuple_product(&shape), Int::<120>);
//! assert_eq!(product_each(&shape), int_tuple![Int::<24>, Int::<5>]);
//! assert_eq!(
//!     product_like(&shape, &int_tuple![(0, 0), 0]),
//!     int_tuple![(Int::<2>, Int::<12>), Int::<5>]
//! );
//! ```
//!
//! # Notes
//!
//! - [`Nil`] yields `Int<1>` under [`tuple_product`] and `Nil` under [`product_each`].
//! - Multiplication delegates to [`crate::integer::mul`], so static `Int<0>` absorbs
//!   dynamic leaves, static overflow fails at compile time, and dynamic overflow panics.
//! - [`product_like`] requires `guide` to be weakly congruent with the target tuple.

use super::{ApplyLeafFunctor, Item, Leaf, Nil, Til, Tuple, TupleContainer, ZipTransformLeaf};
use crate::integer::{ArithMul, Int, Integer, Product, mul};

/// Recursively multiplies all scalar leaves in a hierarchical tuple (`product(t)`).
pub const trait TupleProduct: [const] Item {
    /// Integer product type preserving staticness across static leaves.
    type Output: [const] Integer;
    /// Computes the product of all leaves (`Int<1>` for `Nil`).
    fn tuple_product(&self) -> Self::Output;
}

const impl TupleProduct for Nil {
    type Output = Int<1>;
    fn tuple_product(&self) -> Int<1> {
        Int
    }
}

const impl<I: [const] Integer> TupleProduct for I {
    type Output = I;
    fn tuple_product(&self) -> I {
        *self
    }
}

const impl<H, T> TupleProduct for TupleContainer<H, T>
where
    H: [const] TupleProduct,
    T: [const] Tuple + [const] TupleProduct,
    H::Output: [const] ArithMul<H::Output, T::Output>,
{
    type Output = Product<H::Output, T::Output>;
    fn tuple_product(&self) -> Self::Output {
        mul(self.0.tuple_product(), self.1.tuple_product())
    }
}

/// Computes the product of all scalar leaves in `t` (`Int<1>` for `Nil`).
pub const fn tuple_product<T: [const] TupleProduct>(t: &T) -> T::Output {
    t.tuple_product()
}

/// Computes [`TupleProduct`] per top-level mode of `self` (`product_each(t)`).
pub const trait ProductEach: [const] Item {
    /// Mode-wise reduced item type.
    type Output: [const] Item;
    /// Reduces each top-level mode to its scalar product.
    fn product_each(&self) -> Self::Output;
}

const impl ProductEach for Nil {
    type Output = Nil;
    fn product_each(&self) -> Nil {
        Nil
    }
}

const impl<I: [const] Integer> ProductEach for I {
    type Output = I;
    fn product_each(&self) -> I {
        *self
    }
}

const impl<H, T> ProductEach for TupleContainer<H, T>
where
    H: [const] TupleProduct,
    T: [const] Tuple + [const] ProductEach,
    T::Output: [const] Tuple,
{
    type Output = TupleContainer<H::Output, T::Output>;
    fn product_each(&self) -> Self::Output {
        Til(self.0.tuple_product(), self.1.product_each())
    }
}

/// Reduces each top-level mode of `t` via [`tuple_product`].
pub const fn product_each<T: [const] ProductEach>(t: &T) -> T::Output {
    t.product_each()
}

// Internal `ApplyLeafFunctor` used by `ProductLike`: drops guide leaf `G` and reduces subtree `X`.
#[doc(hidden)]
#[derive(Clone, Copy, Default, Debug)]
pub struct ProductLeafFunctor;

const impl<G: [const] Leaf, X: [const] TupleProduct> ApplyLeafFunctor<(G, X)>
    for ProductLeafFunctor
{
    type Output = X::Output;
    fn apply(&self, (_, x): (G, X)) -> Self::Output {
        x.tuple_product()
    }
}

/// Contracts each subtree matching a leaf of `Guide` into its [`TupleProduct`] (`product_like(a, b)`).
pub const trait ProductLike<Guide: [const] Item>: [const] Item {
    /// Item congruent to `Guide` holding subtree products at each leaf.
    type Output: [const] Item;
    /// Contracts `self` to match `guide`'s structural profile.
    fn product_like(&self, guide: &Guide) -> Self::Output;
}

const impl<S: [const] Item, Guide> ProductLike<Guide> for S
where
    Guide: [const] ZipTransformLeaf<ProductLeafFunctor, S>,
{
    type Output = Guide::Output;
    fn product_like(&self, guide: &Guide) -> Self::Output {
        guide.zip_transform(&ProductLeafFunctor, *self)
    }
}

/// Contracts subtrees of `s` to match the leaf positions of `guide` using [`tuple_product`].
pub const fn product_like<S: [const] ProductLike<Guide>, Guide: [const] Item>(
    s: &S,
    guide: &Guide,
) -> S::Output {
    s.product_like(guide)
}
