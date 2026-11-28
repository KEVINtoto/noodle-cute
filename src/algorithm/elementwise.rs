//! Pure-safe element-wise tensor algorithms: CuTe `fill`, `clear`, `copy_if`, `axpby`,
//! `for_each`, `transform`, `transform_into`, and `transform_zip`.
//!
//! Every algorithm visits elements in colexicographic linear order `0..size(tensor)` using
//! signed [`int`] indices over safe [`Tensor`] views or owning [`ArrayEngine`](crate::tensor::ArrayEngine)
//! tensors, with `#![no_std]` compatibility and zero `unsafe`.
//!
//! # Main Interfaces
//!
//! - [`fill`] / [`clear`]: Fills every element with a constant or [`Default::default()`].
//! - [`copy_if`]: Predicated copy `dst(i) = From::from(src(i))` for indices where `pred(i)` holds.
//! - [`axpby`]: Linear combination `y = alpha * x + beta * y` (skipping reads of `y` when
//!   `beta == Beta::default()`).
//! - [`for_each`]: Visits every element by value in colexicographic order.
//! - [`transform`] / [`transform_into`] / [`transform_zip`]: In-place unary, out-of-place unary,
//!   and binary element-wise transformations.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algorithm::{
//!     axpby, clear, fill, for_each, transform, transform_into, transform_zip,
//! };
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::tensor::make_owned_tensor;
//!
//! let layout = Layout::from_shape_right(int_tuple![Int::<2>, Int::<3>]);
//! let mut x = make_owned_tensor::<f32, _>(layout);
//! let mut y = make_owned_tensor::<f32, _>(layout);
//!
//! fill(&mut x, 2.0);
//! transform(&mut x, |v| v + 1.0);
//! fill(&mut y, f32::NAN);
//! // beta == 0 does not read y (NaN-safe).
//! axpby(2.0f32, &x, 0.0f32, &mut y);
//! assert_eq!(y.as_slice(), &[6.0; 6]);
//!
//! axpby(1.0f32, &x, 0.5f32, &mut y);
//! let mut sum = 0.0;
//! for_each(&y, |v| sum += v);
//! assert_eq!(sum, 36.0);
//!
//! transform_into(&x, &mut y, |v| v * 10.0);
//! let yv = y.view_mut();
//! let mut yv_dst = yv;
//! transform_zip(&x.view(), &yv, &mut yv_dst, |a, b| a + b);
//! assert_eq!(y.as_slice(), &[33.0; 6]);
//!
//! clear(&mut x);
//! assert_eq!(x.as_slice(), &[0.0; 6]);
//! ```
//!
//! # Notes
//!
//! - Multi-tensor algorithms validate size equality at compile time for static shapes and
//!   panic with [`NoodleCuteError::SizeMismatch`] for mismatched dynamic shapes.
//! - Because [`CellAccessor`](crate::tensor::CellAccessor) is `Copy` and interior-mutable
//!   without `unsafe`, callers needing aliased input/output views (e.g. `transform_zip(&x, &yv, &mut yv_copy, ...)`)
//!   can copy a `CellAccessor`-backed view obtained from [`Tensor::view_mut`](crate::tensor::Tensor::view_mut).

use core::ops::{Add, Mul};

use crate::NoodleCuteError;
use crate::integer::{add, int};
use crate::layout::Layout;
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tensor::{Tensor, TensorIndex, TensorSet};

use super::reference::{STATIC_SIZE_COMPAT, ValidSizeMatch};

/// Fills every element of `tensor` with `value` in colexicographic order (CuTe `fill`).
pub const fn fill<A, S, D, T>(tensor: &mut Tensor<A, Layout<S, D>>, value: T)
where
    S: [const] Shape,
    D: [const] Stride<S>,
    T: Copy,
    Tensor<A, Layout<S, D>>: [const] TensorSet<int, Item = T>,
{
    let n = tensor.size();
    let mut i: int = 0;
    while i < n {
        tensor.set(i, value);
        i = add(i, 1);
    }
}

/// Fills every element of `tensor` with [`Default::default()`] (CuTe `clear`).
pub fn clear<A, S, D, T>(tensor: &mut Tensor<A, Layout<S, D>>)
where
    S: Shape,
    D: Stride<S>,
    T: Copy + Default,
    Tensor<A, Layout<S, D>>: TensorSet<int, Item = T>,
{
    fill(tensor, T::default());
}

/// Predicated element-wise copy: `dst[i] = From::from(src[i])` wherever `pred(i)` is `true`
/// (CuTe `copy_if`).
///
/// Elements with a `false` predicate are neither read from `src` nor written to `dst`, making
/// this safe for out-of-bounds tail tiles predicated with [`identity_tensor`](crate::tensor::identity_tensor)
/// and [`elem_less`](crate::shape::elem_less).
///
/// Statically mismatched total sizes fail at compile time; dynamic mismatches panic with
/// [`NoodleCuteError::SizeMismatch`].
///
/// ```compile_fail,E0271
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::algorithm::copy_if;
/// use noodle_cute::integer::Int;
/// use noodle_cute::tensor::make_owned_tensor;
///
/// let src = make_owned_tensor::<i32, _>(Int::<4>);
/// let mut dst = make_owned_tensor::<i32, _>(Int::<5>);
/// copy_if(|_| true, &src, &mut dst);
/// ```
pub fn copy_if<AS, SS, DS, AD, SD, DD, TS, TD>(
    mut pred: impl FnMut(int) -> bool,
    src: &Tensor<AS, Layout<SS, DS>>,
    dst: &mut Tensor<AD, Layout<SD, DD>>,
) where
    SS: Shape,
    DS: Stride<SS>,
    SD: Shape,
    DD: Stride<SD>,
    (): ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SS, SD> }>,
    Tensor<AS, Layout<SS, DS>>: TensorIndex<int, Output = TS>,
    Tensor<AD, Layout<SD, DD>>: TensorSet<int, Item = TD>,
    TD: From<TS>,
{
    let n = dst.size();
    NoodleCuteError::SizeMismatch.check_with(src.size() == n, "copy_if: size mismatch");
    let mut i: int = 0;
    while i < n {
        if pred(i) {
            dst.set(i, TD::from(src.at(i)));
        }
        i = add(i, 1);
    }
}

/// Element-wise linear combination `y[i] = alpha * x[i] + beta * y[i]` (CuTe `axpby`).
///
/// When `beta == Beta::default()` (zero), `y[i]` is not read (avoiding `0 * NaN` propagation
/// and matching CuTe's `axpby`). Statically mismatched sizes fail at compile time; dynamic
/// mismatches panic with [`NoodleCuteError::SizeMismatch`].
pub fn axpby<AX, SX, DX, AY, SY, DY, TX, TY, Alpha, Beta>(
    alpha: Alpha,
    x: &Tensor<AX, Layout<SX, DX>>,
    beta: Beta,
    y: &mut Tensor<AY, Layout<SY, DY>>,
) where
    SX: Shape,
    DX: Stride<SX>,
    SY: Shape,
    DY: Stride<SY>,
    (): ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SX, SY> }>,
    Tensor<AX, Layout<SX, DX>>: TensorIndex<int, Output = TX>,
    Tensor<AY, Layout<SY, DY>>: TensorIndex<int, Output = TY> + TensorSet<int, Item = TY>,
    Alpha: Copy + Mul<TX, Output = TY>,
    Beta: Copy + PartialEq + Default + Mul<TY, Output = TY>,
    TY: Add<Output = TY>,
{
    let n = x.size();
    NoodleCuteError::SizeMismatch.check_with(y.size() == n, "axpby: size mismatch");
    let beta_is_zero = beta == Beta::default();
    let mut i: int = 0;
    while i < n {
        let ax = alpha * x.at(i);
        let out = if beta_is_zero {
            ax
        } else {
            ax + beta * y.at(i)
        };
        y.set(i, out);
        i = add(i, 1);
    }
}

/// Applies `op` to every element of `tensor` by value in colexicographic order (CuTe `for_each`).
pub fn for_each<A, S, D, T>(tensor: &Tensor<A, Layout<S, D>>, mut op: impl FnMut(T))
where
    S: Shape,
    D: Stride<S>,
    Tensor<A, Layout<S, D>>: TensorIndex<int, Output = T>,
{
    let n = tensor.size();
    let mut i: int = 0;
    while i < n {
        op(tensor.at(i));
        i = add(i, 1);
    }
}

/// Replaces every element of `tensor` in place with `op(tensor[i])` (CuTe `transform(tensor, op)`).
pub fn transform<A, S, D, T>(tensor: &mut Tensor<A, Layout<S, D>>, mut op: impl FnMut(T) -> T)
where
    S: Shape,
    D: Stride<S>,
    Tensor<A, Layout<S, D>>: TensorIndex<int, Output = T> + TensorSet<int, Item = T>,
{
    let n = tensor.size();
    let mut i: int = 0;
    while i < n {
        let v = op(tensor.at(i));
        tensor.set(i, v);
        i = add(i, 1);
    }
}

/// Unary out-of-place transform `dst[i] = op(src[i])` (CuTe `transform(in, out, op)`).
///
/// Statically mismatched total sizes fail at compile time; dynamic mismatches panic with
/// [`NoodleCuteError::SizeMismatch`].
pub fn transform_into<AS, SS, DS, AD, SD, DD, TS, TD>(
    src: &Tensor<AS, Layout<SS, DS>>,
    dst: &mut Tensor<AD, Layout<SD, DD>>,
    mut op: impl FnMut(TS) -> TD,
) where
    SS: Shape,
    DS: Stride<SS>,
    SD: Shape,
    DD: Stride<SD>,
    (): ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SS, SD> }>,
    Tensor<AS, Layout<SS, DS>>: TensorIndex<int, Output = TS>,
    Tensor<AD, Layout<SD, DD>>: TensorSet<int, Item = TD>,
{
    let n = src.size();
    NoodleCuteError::SizeMismatch.check_with(dst.size() == n, "transform_into: size mismatch");
    let mut i: int = 0;
    while i < n {
        dst.set(i, op(src.at(i)));
        i = add(i, 1);
    }
}

/// Binary element-wise transform `dst[i] = op(a[i], b[i])` (CuTe `transform(in1, in2, out, op)`).
///
/// Statically mismatched total sizes fail at compile time; dynamic mismatches panic with
/// [`NoodleCuteError::SizeMismatch`].
pub fn transform_zip<AA, SA, DA, AB, SB, DB, AD, SD, DD, TA, TB, TD>(
    a: &Tensor<AA, Layout<SA, DA>>,
    b: &Tensor<AB, Layout<SB, DB>>,
    dst: &mut Tensor<AD, Layout<SD, DD>>,
    mut op: impl FnMut(TA, TB) -> TD,
) where
    SA: Shape,
    DA: Stride<SA>,
    SB: Shape,
    DB: Stride<SB>,
    SD: Shape,
    DD: Stride<SD>,
    (): ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SA, SB> }>
        + ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SA, SD> }>
        + ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SB, SD> }>,
    Tensor<AA, Layout<SA, DA>>: TensorIndex<int, Output = TA>,
    Tensor<AB, Layout<SB, DB>>: TensorIndex<int, Output = TB>,
    Tensor<AD, Layout<SD, DD>>: TensorSet<int, Item = TD>,
{
    let n = a.size();
    NoodleCuteError::SizeMismatch.check_with(
        b.size() == n && dst.size() == n,
        "transform_zip: size mismatch",
    );
    let mut i: int = 0;
    while i < n {
        dst.set(i, op(a.at(i), b.at(i)));
        i = add(i, 1);
    }
}
