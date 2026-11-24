//! Stride inner products (`InnerProduct`) and exclusive prefix products (`PrefixProduct`).
//!
//! # Main Interfaces
//!
//! - [`InnerProduct`] / [`inner_product`]: Evaluates $\langle c, d \rangle = \sum_k c_k \cdot d_k$
//!   for scalar or basis strides.
//! - [`PrefixProduct`] / [`prefix_product`] / [`prefix_product_with`]: Computes exclusive
//!   prefix products of shape leaf extents to construct compact column-major strides.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::stride::{inner_product, prefix_product};
//!
//! let shape = int_tuple![Int::<3>, (Int::<2>, Int::<4>)];
//! assert_eq!(prefix_product(shape), int_tuple![Int::<1>, (Int::<3>, Int::<6>)]);
//! assert_eq!(inner_product(int_tuple![1i32, 2i32], int_tuple![1i32, 3i32]), 7i32);
//! ```
//!
//! # Notes
//!
//! - [`PrefixProduct`] skips the unused trailing multiplication `init * s_last`, preventing
//!   spurious overflow when the total shape size exceeds `int::MAX` while all strides fit.

use crate::basis::BasisCoeff;
use crate::integer::{Int, Integer};
use crate::shape::Shape;
use crate::tuple::{Item, Leaf, Nil, Til, Tuple};

use super::{Accumulate, StrideMul};

/// Evaluates the linear inner product $\langle c, d \rangle = \sum_k c_k \cdot d_k$.
///
/// - When strides are scalars, computes linear address offsets.
/// - When strides are basis terms or combinations, computes transformed coordinates.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::stride::InnerProduct;
/// use noodle_cute::tuple::Nil;
///
/// // Nil inner product preserves accumulator:
/// assert_eq!(Nil.inner_product(Nil, Int::<0>), Int::<0>);
///
/// // Integer inner product: 2 * 3 = 6
/// assert_eq!(2i32.inner_product(3i32, Int::<0>), 6i32);
///
/// // Tuple inner product: 2 * 1 + 3 * 4 = 14
/// let crd = int_tuple![2i32, 3i32];
/// let strd = int_tuple![1i32, 4i32];
/// assert_eq!(crd.inner_product(strd, Int::<0>), 14i32);
/// ```
pub const trait InnerProduct<D, Acc = Int<0>>: Copy {
    /// Accumulated inner product output type.
    type Output: Copy;
    /// Accumulates `self` dotted with `stride` onto `acc`.
    fn inner_product(self, stride: D, acc: Acc) -> Self::Output;
}

const impl<Acc: Copy> InnerProduct<Nil, Acc> for Nil {
    type Output = Acc;
    fn inner_product(self, _: Nil, acc: Acc) -> Self::Output {
        acc
    }
}

const impl<S, D, Acc> InnerProduct<D, Acc> for S
where
    S: [const] BasisCoeff,
    D: [const] StrideMul<S>,
    Acc: [const] Accumulate<D::Output>,
{
    type Output = Acc::Output;
    fn inner_product(self, stride: D, acc: Acc) -> Self::Output {
        acc.accumulate(stride.stride_mul(self))
    }
}

const impl<H, T, RH, RT, Acc> InnerProduct<Til<RH, RT>, Acc> for Til<H, T>
where
    Acc: Copy,
    H: [const] InnerProduct<RH, Acc>,
    T: Tuple + [const] InnerProduct<RT, H::Output>,
    RH: Copy,
    RT: Copy,
{
    type Output = T::Output;
    fn inner_product(self, stride: Til<RH, RT>, acc: Acc) -> Self::Output {
        let acc1 = self.0.inner_product(stride.0, acc);
        self.1.inner_product(stride.1, acc1)
    }
}

/// Freestanding function for [`InnerProduct`] initialized with zero (`Int<0>`).
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::stride::inner_product;
///
/// // Memory offset calculation:
/// let crd = int_tuple![2i32, 3i32];
/// let strd = int_tuple![1i32, 4i32];
/// assert_eq!(inner_product(crd, strd), 14i32);
/// ```
pub const fn inner_product<S: [const] InnerProduct<D>, D>(coord: S, stride: D) -> S::Output {
    coord.inner_product(stride, Int::<0>)
}

const HAS_LEAVES<T: Item>: bool = T::LEAVES > 0;
const OR_BOOL<const A: bool, const B: bool>: bool = A || B;

/// Recursive exclusive prefix product carrying seed `Init`;
/// `NEED_NEXT = false` skips the unused final `init * self` multiplication.
#[doc(hidden)]
pub const trait PrefixProductStep<Init, const NEED_NEXT: bool>: Item {
    type Stride: Copy;
    type NextSeed: Copy;

    fn prefix_product_step_impl(self, init: Init) -> (Self::Stride, Self::NextSeed);
}

const impl<Init: Leaf, const NEED_NEXT: bool> PrefixProductStep<Init, NEED_NEXT> for Nil {
    type Stride = Nil;
    type NextSeed = Init;

    fn prefix_product_step_impl(self, init: Init) -> (Nil, Init) {
        (Nil, init)
    }
}

const impl<const NEED_NEXT: bool> PrefixProductStep<Nil, NEED_NEXT> for Nil {
    type Stride = Nil;
    type NextSeed = Nil;

    fn prefix_product_step_impl(self, _: Nil) -> (Nil, Nil) {
        (Nil, Nil)
    }
}

const impl<Init, S> PrefixProductStep<Init, true> for S
where
    S: [const] Integer + Shape,
    Init: Leaf + [const] StrideMul<S>,
{
    type Stride = Init;
    type NextSeed = Init::Output;

    fn prefix_product_step_impl(self, init: Init) -> (Init, Init::Output) {
        let next = init.stride_mul(self);
        (init, next)
    }
}

const impl<Init, S> PrefixProductStep<Init, false> for S
where
    S: [const] Integer + Shape,
    Init: Leaf,
{
    type Stride = Init;
    type NextSeed = Nil;

    fn prefix_product_step_impl(self, init: Init) -> (Init, Nil) {
        (init, Nil)
    }
}

const impl<S, const NEED_NEXT: bool> PrefixProductStep<Nil, NEED_NEXT> for S
where
    S: [const] Integer + Shape,
{
    type Stride = Int<0>;
    type NextSeed = Nil;

    fn prefix_product_step_impl(self, _: Nil) -> (Int<0>, Nil) {
        (Int, Nil)
    }
}

const impl<Init, H, T, const NEED_NEXT: bool> PrefixProductStep<Init, NEED_NEXT> for Til<H, T>
where
    Init: Leaf,
    H: [const] PrefixProductStep<Init, { OR_BOOL::<NEED_NEXT, { HAS_LEAVES::<T> }> }>,
    T: Tuple + [const] PrefixProductStep<H::NextSeed, NEED_NEXT>,
{
    type Stride = Til<H::Stride, T::Stride>;
    type NextSeed = T::NextSeed;

    fn prefix_product_step_impl(self, init: Init) -> (Self::Stride, Self::NextSeed) {
        let (h_stride, h_next) = self.0.prefix_product_step_impl(init);
        let (t_stride, t_next) = self.1.prefix_product_step_impl(h_next);
        (Til(h_stride, t_stride), t_next)
    }
}

const impl<H, T, const NEED_NEXT: bool> PrefixProductStep<Nil, NEED_NEXT> for Til<H, T>
where
    H: [const] PrefixProductStep<Nil, { OR_BOOL::<NEED_NEXT, { HAS_LEAVES::<T> }> }>,
    T: Tuple + [const] PrefixProductStep<H::NextSeed, NEED_NEXT>,
{
    type Stride = Til<H::Stride, T::Stride>;
    type NextSeed = T::NextSeed;

    fn prefix_product_step_impl(self, _: Nil) -> (Self::Stride, Self::NextSeed) {
        let (h_stride, h_next) = self.0.prefix_product_step_impl(Nil);
        let (t_stride, t_next) = self.1.prefix_product_step_impl(h_next);
        (Til(h_stride, t_stride), t_next)
    }
}

const impl<H, T, IH, IT, const NEED_NEXT: bool> PrefixProductStep<Til<IH, IT>, NEED_NEXT>
    for Til<H, T>
where
    IH: Copy,
    IT: Copy,
    H: [const] PrefixProductStep<IH, NEED_NEXT>,
    T: Tuple + [const] PrefixProductStep<IT, NEED_NEXT>,
{
    type Stride = Til<H::Stride, T::Stride>;
    type NextSeed = Til<H::NextSeed, T::NextSeed>;

    fn prefix_product_step_impl(self, init: Til<IH, IT>) -> (Self::Stride, Self::NextSeed) {
        let (h_stride, h_next) = self.0.prefix_product_step_impl(init.0);
        let (t_stride, t_next) = self.1.prefix_product_step_impl(init.1);
        (Til(h_stride, t_stride), Til(h_next, t_next))
    }
}

/// Exclusive prefix product of the leaves of a shape.
///
/// Given a shape $S = (s_0, s_1, \dots)$ and an initial seed `Init` (defaulting to [`Int<1>`]),
/// generates default compact strides:
/// $$(s_0, s_1, s_2, \dots) \mapsto (\text{init}, \text{init} \cdot s_0, \text{init} \cdot s_0 \cdot s_1, \dots)$$
///
/// The final leaf's unused product `init * s_last` is never computed, avoiding spurious overflow
/// when the total shape size exceeds [`int::MAX`](crate::integer::int::MAX).
///
/// # Trait Dispatch Modes
/// 1. **Integer leaf with leaf seed**: yields `init` (and advances `init * extent` only when followed by another leaf).
/// 2. **Tuple with leaf seed**: accumulates sequentially across all leaves depth-first.
/// 3. **Tuple with tuple seed**: distributes seeds by mode, computing prefix product independently per mode.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::stride::PrefixProduct;
///
/// let s = int_tuple![Int::<3>, Int::<4>];
/// let strd = s.prefix_product(Int::<1>);
/// assert_eq!(strd, int_tuple![Int::<1>, Int::<3>]);
/// ```
pub const trait PrefixProduct<Init = Int<1>>: Copy {
    /// Resulting stride type.
    type Stride: Copy;

    /// Evaluates the full exclusive prefix product with `init`.
    fn prefix_product(self, init: Init) -> Self::Stride;
}

const impl<Init, S> PrefixProduct<Init> for S
where
    S: [const] PrefixProductStep<Init, false>,
{
    type Stride = S::Stride;

    fn prefix_product(self, init: Init) -> Self::Stride {
        self.prefix_product_step_impl(init).0
    }
}

/// Computes the prefix product of a shape using an explicit initial seed.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::basis::{Mode, ScaledBasis};
/// use noodle_cute::integer::Int;
/// use noodle_cute::stride::prefix_product_with;
/// use noodle_cute::{int_tuple, tuple};
///
/// // Prefix product starting at stride 2: (4, 8) -> (2, 8)
/// let s = int_tuple![Int::<4>, Int::<8>];
/// assert_eq!(prefix_product_with(s, Int::<2>), int_tuple![Int::<2>, Int::<8>]);
///
/// // Prefix product with ScaledBasis init: (4, 5) with (1@0, 1@1)
/// let b0 = ScaledBasis::<Int<1>, Mode<0>>::new(Int::<1>);
/// let b1 = ScaledBasis::<Int<1>, Mode<1>>::new(Int::<1>);
/// let seeds = tuple![b0, b1];
/// assert_eq!(prefix_product_with(s, seeds), tuple![b0, b1]);
/// ```
pub const fn prefix_product_with<S: [const] PrefixProduct<Init>, Init>(
    shape: S,
    init: Init,
) -> S::Stride {
    shape.prefix_product(init)
}

/// Computes the standard column-major compact stride of a shape, using initial seed `Int<1>`.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::stride::prefix_product;
///
/// let s = int_tuple![Int::<3>, (Int::<2>, Int::<4>)];
/// let d = prefix_product(s);
/// assert_eq!(d, int_tuple![Int::<1>, (Int::<3>, Int::<6>)]);
/// ```
pub const fn prefix_product<S: [const] PrefixProduct>(shape: S) -> S::Stride {
    shape.prefix_product(Int::<1>)
}
