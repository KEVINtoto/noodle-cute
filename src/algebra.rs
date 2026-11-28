//! CuTe layout algebra operations.
//!
//! This module implements the algebraic transformations on [`Layout`](crate::layout::Layout)
//! and tiler types built on top of [`integer`](crate::integer), [`tuple`](mod@crate::tuple),
//! [`basis`](crate::basis), [`shape`](crate::shape), [`stride`](crate::stride), and
//! [`layout`](crate::layout):
//!
//! - [`mod@coalesce`]: Canonical mode merging and size-1 elimination
//!   ([`Coalesce`], [`CoalesceExtended`], [`coalesce()`], [`coalesce_extended`]).
//! - [`mod@nullspace`]: Kernel (zero-stride domain) extraction
//!   ([`Nullspace`], [`nullspace()`]).
//! - [`mod@composition`]: Functional composition $A \circ B$ and mode-wise tile composition
//!   ([`Composition`], [`ComposeTile`], [`composition()`], [`compose_tile`]).
//! - [`mod@complement`]: Free and extended codomain complement $A^*$
//!   ([`Complement`], [`ComplementWith`], [`complement()`], [`complement_with`]).
//! - [`mod@inverse`]: Right and left layout inverses $A^{-1}_R, A^{-1}_L$
//!   ([`RightInverse`], [`LeftInverse`], [`right_inverse`], [`left_inverse`]).
//! - [`mod@logical_divide`]: Logical, zipped, tiled, and flat division $A \oslash B$
//!   ([`LogicalDivide`], [`ZippedDivide`], [`TiledDivide`], [`FlatDivide`],
//!   [`LogicalDivideAt`], [`ZippedDivideAt`], [`logical_divide()`], [`logical_divide_at`],
//!   [`zipped_divide`], [`zipped_divide_at`], [`tiled_divide`], [`flat_divide`]).
//! - [`mod@logical_product`]: Logical, zipped, tiled, flat, blocked, and raked products $A \otimes B$
//!   ([`LogicalProduct`], [`ZippedProduct`], [`TiledProduct`], [`FlatProduct`],
//!   [`LogicalProductAt`], [`BlockedProduct`], [`RakedProduct`], [`logical_product()`],
//!   [`logical_product_at`], [`zipped_product`], [`tiled_product`], [`flat_product`],
//!   [`blocked_product`], [`raked_product`]).
//! - [`mod@common`]: Common vectorization analysis, shape-targeted tiling, and fragment layouts
//!   ([`MaxCommonVector`], [`MaxCommonLayout`], [`TileToShape`], [`FragmentLike`],
//!   [`max_common_vector`], [`max_common_layout`], [`tile_to_shape`], [`make_fragment_like`]).
//! - [`mod@greatest_common_domain`]: Greatest common domain factorization of two shapes or layouts
//!   ([`IntoGcdShape`], [`GreatestCommonDomain`], [`greatest_common_domain()`]).
//! - [`mod@layout_add`]: Coordinate-wise layout addition $A + B$
//!   ([`LayoutAdd`], [`layout_add()`], [`layout_add_by`], [`layout_add_with_profile`]).
//!
//! # Main Interfaces
//!
//! All algebraic operations are exposed in three equivalent forms:
//! 1. Core `const` traits (e.g., [`Coalesce`], [`Composition`], [`LogicalDivide`]);
//! 2. Inherent `const fn` methods on [`Layout`](crate::layout::Layout) (e.g., `layout.coalesce()`);
//! 3. Top-level `const fn` free functions re-exported here (e.g., [`coalesce()`], [`composition()`],
//!    [`logical_divide()`]).
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::{
//!     blocked_product, coalesce, complement_with, composition, greatest_common_domain,
//!     layout_add, left_inverse, logical_divide, logical_product, nullspace, right_inverse,
//!     zipped_divide,
//! };
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let a = Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<1>, Int::<4>]);
//! assert_eq!(coalesce(a), Layout::new(Int::<12>, Int::<1>));
//!
//! let tile = Layout::new(Int::<4>, Int::<2>);
//! let divided = logical_divide(Layout::new(Int::<24>, Int::<1>), tile);
//! assert_eq!(
//!     divided,
//!     Layout::new(
//!         int_tuple![Int::<4>, (Int::<2>, Int::<3>)],
//!         int_tuple![Int::<2>, (Int::<1>, Int::<8>)],
//!     )
//! );
//!
//! let prod = logical_product(Layout::new(Int::<3>, Int::<1>), Layout::new(Int::<4>, Int::<1>));
//! assert_eq!(
//!     prod,
//!     Layout::new(int_tuple![Int::<3>, Int::<4>], int_tuple![Int::<1>, Int::<3>])
//! );
//!
//! let sum = layout_add(a, Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<3>, Int::<1>]));
//! assert_eq!(
//!     sum,
//!     Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<4>, Int::<5>])
//! );
//! ```
//!
//! # Notes
//!
//! - Every operation is `const`-evaluable and `#![no_std]`-compatible with zero heap allocation.
//! - Operations follow the crate's unified error model: invalid preconditions (such as
//!   non-injective complements, zero/negative extents, out-of-bounds paths, or [`int`](crate::integer::int)
//!   arithmetic overflow) are rejected at compile time for static operands and panic at runtime
//!   for dynamic operands.

pub mod coalesce;
pub mod common;
pub mod complement;
pub mod composition;
pub mod greatest_common_domain;
pub mod inverse;
pub mod layout_add;
pub mod logical_divide;
pub mod logical_product;
pub mod nullspace;

pub use coalesce::{
    Coalesce, CoalesceExtended, CoalesceProfile, CoalesceTarget, coalesce, coalesce_extended,
    coalesce_with_profile, coalesce_z,
};
pub use common::{
    FragmentLike, MaxCommonLayout, MaxCommonVector, TileToShape, make_fragment_like,
    max_common_layout, max_common_vector, tile_to_shape,
};
pub use complement::{Complement, ComplementWith, complement, complement_with};
pub use composition::{ComposeTile, Composition, compose_tile, composition};
pub use greatest_common_domain::{GreatestCommonDomain, IntoGcdShape, greatest_common_domain};
pub use inverse::{LeftInverse, RightInverse, left_inverse, right_inverse};
pub use layout_add::{LayoutAdd, layout_add, layout_add_by, layout_add_with_profile};
pub use logical_divide::{
    FlatDivide, LogicalDivide, LogicalDivideAt, TiledDivide, ZippedDivide, ZippedDivideAt,
    flat_divide, logical_divide, logical_divide_at, tiled_divide, zipped_divide, zipped_divide_at,
};
pub use logical_product::{
    BlockedProduct, FlatProduct, LogicalProduct, LogicalProductAt, RakedProduct, TiledProduct,
    ZippedProduct, blocked_product, flat_product, logical_product, logical_product_at,
    raked_product, tiled_product, zipped_product,
};
pub use nullspace::{Nullspace, nullspace};
