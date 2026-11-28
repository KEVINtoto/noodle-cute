//! CuTe algorithms expressed over [`Tensor`](crate::tensor::Tensor)s (Whitepaper §2.6).
//!
//! Built on top of [`tensor`](crate::tensor) and [`algebra`](crate::algebra), this module
//! provides pure `#![no_std]`, zero-allocation host-side algorithms and tensor view
//! transformations:
//!
//! - [`mod@reference`]: Reference implementations of [`reference::copy`] and batched
//!   [`reference::gemm`] built on [`mod@atom`].
//! - [`mod@copy`]: Layout-analyzed [`copy()`] (`greatest_common_domain` -> `nullspace` &
//!   write-after-write rejection -> `right_inverse` alignment & vectorization) and
//!   [`copy_plan`].
//! - [`mod@elementwise`]: Pure-safe element-wise tensor algorithms ([`fill`], [`clear`],
//!   [`copy_if`], [`axpby`], [`for_each`], [`transform`], [`transform_into`], [`transform_zip`]).
//! - [`mod@atom`]: Device-independent, pure-safe Copy & MMA atoms, thread–value tilings,
//!   fragment retiling, explicit copy policies ([`copy_with`], [`copy_if_with`]), and 5-rank
//!   register-reuse MMA GEMM ([`gemm_mma`], [`gemm_fma`]).
//! - [`mod@einfold`]: Named mode rearrangement ([`einfold()`], [`parse_einfold`],
//!   [`einfold!`](crate::einfold!)) over [`Layout`](crate::layout::Layout)s,
//!   [`Tensor`](crate::tensor::Tensor)s, and hierarchical tuples.
//! - [`mod@einsum`]: Explicit binary tensor contractions ([`einsum()`], [`parse_einsum`],
//!   [`fold_operand`], [`einsum!`](crate::einsum!)) lowered to batched [`gemm`].
//! - [`mod@im2col`]: N-D convolution `im2col` layout views ([`im2col()`], [`im2col_coord`],
//!   [`im2col_padded`], [`im2col_layout`], and [`Im2ColOptions`]).

pub mod atom;
pub mod copy;
pub mod einfold;
pub mod einsum;
pub mod elementwise;
pub mod im2col;
pub mod reference;

pub use atom::{CopyPolicy, copy_if_with, copy_with, gemm_fma, gemm_mma};
pub use copy::{CopyPlan, copy, copy_plan};
pub use einfold::{Einfold, EinfoldSpec, ModeToken, einfold, parse_einfold};
pub use einsum::{
    Einsum, EinsumSpec, FoldGroup, FoldOperand, OperandFold, einsum, fold_operand, parse_einsum,
};
pub use elementwise::{
    axpby, clear, copy_if, fill, for_each, transform, transform_into, transform_zip,
};
pub use im2col::{
    DefaultParam, Im2Col, Im2ColCoord, Im2ColCoordTensor, Im2ColLayout, Im2ColOptions,
    Im2ColPadded, Im2ColPaddedTensor, im2col, im2col_coord, im2col_coord_with, im2col_layout,
    im2col_layout_with, im2col_padded, im2col_padded_with, im2col_with,
};
pub use reference::{Gemm, gemm};
