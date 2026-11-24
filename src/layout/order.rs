//! Ordered and stride-induced compact layouts (`make_ordered_layout`, `make_layout_like`).
//!
//! # Main Interfaces
//!
//! - [`OrderKey`]: Leaf key types (`int`, `Int<N>`, `BasisTerm`, `BasisCombination`)
//!   that induce a mode ordering.
//! - [`MakeOrderedLayout`] / [`make_ordered_layout`]: Constructs a compact [`Layout`]
//!   with a given shape ordered by ascending mode keys.
//! - [`MakeLayoutLike`] / [`make_layout_like`]: Constructs a compact [`Layout`] with
//!   the same shape whose non-zero strides follow the relative order of an existing layout's strides.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::{Layout, make_ordered_layout};
//!
//! let s = int_tuple![Int::<4>, Int::<8>];
//! let order = int_tuple![Int::<1>, Int::<0>];
//! assert_eq!(
//!     make_ordered_layout(&s, &order),
//!     Layout::new(s, int_tuple![Int::<8>, Int::<1>])
//! );
//! ```
//!
//! # Notes
//!
//! - Ties and dynamic keys preserve left-to-right mode order; static keys sort before
//!   dynamic keys.
//! - [`make_layout_like`] preserves static zero-stride modes (`Int<0>`, `BasisCombination<Nil>`,
//!   or static-zero `BasisTerm`) as `Int<0>`.

use core::cmp::Ordering;

use crate::basis::{
    BasisCoeff, BasisCombination, BasisPath, BasisTerm, BasisTuple, IntoColexTerms, MergeColex,
};
use crate::integer::{ArithMul, INCREASE, Int, Integer, Product, int, mul};
use crate::shape::{Shape, ShapeScalar};
use crate::stride::Stride;
use crate::tuple::{Congruent, FlatTuple, Flatten, Leaf, Nil, Til, Unflatten, unflatten_like};

use super::Layout;

/// Key item usable to determine mode ordering in [`make_ordered_layout`] and [`make_layout_like`].
pub const trait OrderKey: Leaf {
    /// `true` iff this key is statically known to be zero.
    const IS_STATIC_ZERO: bool = false;
}

const impl OrderKey for int {
    const IS_STATIC_ZERO: bool = false;
}

const impl<const N: int> OrderKey for Int<N> {
    const IS_STATIC_ZERO: bool = N == 0;
}

const impl<V, P> OrderKey for BasisTerm<V, P>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    Self: Leaf,
{
    const IS_STATIC_ZERO: bool = V::IS_STATIC_ZERO;
}

const impl<T: [const] BasisTuple> OrderKey for BasisCombination<T> {
    const IS_STATIC_ZERO: bool = T::RANK == 0;
}

#[doc(hidden)]
pub const ORDER_BOTH_STATIC<L: OrderKey, R: OrderKey>: bool = L::IS_STATIC && R::IS_STATIC;

#[doc(hidden)]
pub const J_LT_I<const J: int, const I: int>: bool = J < I;

#[doc(hidden)]
pub const SKIP_ORDER_PRECEDE<
    KJ: OrderKey,
    const I: int,
    const J: int,
    const SKIP_ZERO: bool,
>: bool = (J == I) || (SKIP_ZERO && KJ::IS_STATIC_ZERO);

#[doc(hidden)]
pub const ZERO_OUT_FLAG<KI: OrderKey, const SKIP_ZERO: bool>: bool =
    SKIP_ZERO && KI::IS_STATIC_ZERO;

/// Determines whether order key `Self` (at flat index `J`) precedes key `R` (at index `I`).
///
/// `J_BEFORE_I` breaks ties and orders dynamic keys left-to-right; `BOTH_STATIC` selects
/// compile-time colexicographical comparison vs mixed/dynamic ordering.
#[doc(hidden)]
pub const trait OrderKeyPrecedes<
    R: OrderKey,
    const J_BEFORE_I: bool,
    const BOTH_STATIC: bool = { ORDER_BOTH_STATIC::<Self, R> },
>: OrderKey
{
    const PRECEDES: bool;
}

const impl<L: OrderKey, R: OrderKey, const J_BEFORE_I: bool> OrderKeyPrecedes<R, J_BEFORE_I, false>
    for L
{
    const PRECEDES: bool = if Self::IS_STATIC {
        true
    } else if R::IS_STATIC {
        false
    } else {
        J_BEFORE_I
    };
}

const impl<L, R, const J_BEFORE_I: bool> OrderKeyPrecedes<R, J_BEFORE_I, true> for L
where
    L: OrderKey + IntoColexTerms,
    R: OrderKey + IntoColexTerms,
    L::Terms: MergeColex<R::Terms>,
{
    const PRECEDES: bool = match <L::Terms as MergeColex<R::Terms>>::STATIC_ORDER {
        Some(Ordering::Less) => true,
        Some(Ordering::Equal) => J_BEFORE_I,
        Some(Ordering::Greater) => false,
        None => J_BEFORE_I,
    };
}

/// Wraps [`OrderKeyPrecedes`] for key pair `(Self, R)` with tie-breaker `J_BEFORE_I`,
/// short-circuiting to `false` when `SKIP` is `true` (self-comparison or skipped static zero).
#[doc(hidden)]
pub const trait CondOrderKeyPrecedes<R: OrderKey, const J_BEFORE_I: bool, const SKIP: bool>:
    OrderKey
{
    const PRECEDES: bool;
}

const impl<L: OrderKey, R: OrderKey, const J_BEFORE_I: bool>
    CondOrderKeyPrecedes<R, J_BEFORE_I, true> for L
{
    const PRECEDES: bool = false;
}

const impl<L, R, const J_BEFORE_I: bool> CondOrderKeyPrecedes<R, J_BEFORE_I, false> for L
where
    L: OrderKey + OrderKeyPrecedes<R, J_BEFORE_I>,
    R: OrderKey,
{
    const PRECEDES: bool = <L as OrderKeyPrecedes<R, J_BEFORE_I>>::PRECEDES;
}

#[doc(hidden)]
pub const PRECEDES_FLAG<
    KJ: CondOrderKeyPrecedes<KI, { J_LT_I::<J, I> }, { SKIP_ORDER_PRECEDE::<KJ, I, J, SKIP_ZERO> }>,
    KI: OrderKey,
    const I: int,
    const J: int,
    const SKIP_ZERO: bool,
>: bool = <KJ as CondOrderKeyPrecedes<
    KI,
    { J_LT_I::<J, I> },
    { SKIP_ORDER_PRECEDE::<KJ, I, J, SKIP_ZERO> },
>>::PRECEDES;

/// Conditionally multiplies running stride accumulator `Self` by shape leaf `S` when
/// `COND` is `true`, returning `Self` unchanged when `COND` is `false`.
#[doc(hidden)]
pub const trait CondMulShape<S, const COND: bool>: [const] Integer {
    type Output: [const] Integer;
    fn cond_mul(self, shape: S) -> Self::Output;
}

const impl<Acc: [const] Integer, S: Copy> CondMulShape<S, false> for Acc {
    type Output = Acc;
    fn cond_mul(self, _: S) -> Acc {
        self
    }
}

const impl<Acc, S> CondMulShape<S, true> for Acc
where
    Acc: [const] Integer + [const] ArithMul<Acc, S>,
    S: [const] Integer,
    Product<Acc, S>: [const] Integer,
{
    type Output = Product<Acc, S>;
    fn cond_mul(self, shape: S) -> Self::Output {
        mul(self, shape)
    }
}

/// Walks flat shape suffix `Self` alongside flat key cursor `FKCursor` (at index `J`),
/// multiplying into `Acc` every extent whose key precedes target key `KI` (at index `I`).
///
/// When `SKIP_ZERO` is `true`, static zero keys are excluded from the product.
#[doc(hidden)]
pub const trait AccumOrderedStride<
    FKCursor: FlatTuple,
    KI: OrderKey,
    const I: int,
    const J: int,
    const SKIP_ZERO: bool,
    Acc: [const] Integer,
>: FlatTuple
{
    type Output: Leaf;
    fn accum_ordered_stride(&self, fk: &FKCursor, acc: Acc) -> Self::Output;
}

const impl<KI: OrderKey, const I: int, const J: int, const SKIP_ZERO: bool, Acc: [const] Integer>
    AccumOrderedStride<Nil, KI, I, J, SKIP_ZERO, Acc> for Nil
where
    Acc: Leaf,
{
    type Output = Acc;
    fn accum_ordered_stride(&self, _: &Nil, acc: Acc) -> Acc {
        acc
    }
}

const impl<SJ, ST, KJ, KT, KI, const I: int, const J: int, const SKIP_ZERO: bool, Acc>
    AccumOrderedStride<Til<KJ, KT>, KI, I, J, SKIP_ZERO, Acc> for Til<SJ, ST>
where
    SJ: ShapeScalar,
    ST: FlatTuple,
    KJ: OrderKey
        + CondOrderKeyPrecedes<KI, { J_LT_I::<J, I> }, { SKIP_ORDER_PRECEDE::<KJ, I, J, SKIP_ZERO> }>,
    KT: FlatTuple,
    KI: OrderKey,
    Acc: [const] CondMulShape<SJ, { PRECEDES_FLAG::<KJ, KI, I, J, SKIP_ZERO> }>,
    ST: [const] AccumOrderedStride<
            KT,
            KI,
            I,
            { INCREASE::<J> },
            SKIP_ZERO,
            <Acc as CondMulShape<SJ, { PRECEDES_FLAG::<KJ, KI, I, J, SKIP_ZERO> }>>::Output,
        >,
{
    type Output = <ST as AccumOrderedStride<
        KT,
        KI,
        I,
        { INCREASE::<J> },
        SKIP_ZERO,
        <Acc as CondMulShape<SJ, { PRECEDES_FLAG::<KJ, KI, I, J, SKIP_ZERO> }>>::Output,
    >>::Output;

    fn accum_ordered_stride(&self, fk: &Til<KJ, KT>, acc: Acc) -> Self::Output {
        let next_acc = acc.cond_mul(self.0);
        self.1.accum_ordered_stride(&fk.1, next_acc)
    }
}

/// Computes the compact ordered stride for key `Self` at flat index `I` using full flat shape
/// `FS` and full flat keys `FKFull`.
///
/// Emits `Int<0>` immediately when `ZERO_OUT` is `true`; otherwise accumulates over preceding
/// modes via [`AccumOrderedStride`] with `SKIP_ZERO`.
#[doc(hidden)]
pub const trait LeafOrderedStride<
    FS: FlatTuple,
    FKFull: FlatTuple,
    const I: int,
    const SKIP_ZERO: bool,
    const ZERO_OUT: bool,
>: OrderKey
{
    type Output: Leaf;
    fn leaf_ordered_stride(fs: &FS, fk_full: &FKFull) -> Self::Output;
}

const impl<FS: FlatTuple, FKFull: FlatTuple, KI: OrderKey, const I: int, const SKIP_ZERO: bool>
    LeafOrderedStride<FS, FKFull, I, SKIP_ZERO, true> for KI
{
    type Output = Int<0>;
    fn leaf_ordered_stride(_: &FS, _: &FKFull) -> Int<0> {
        Int::<0>
    }
}

const impl<FS, FKFull, KI, const I: int, const SKIP_ZERO: bool>
    LeafOrderedStride<FS, FKFull, I, SKIP_ZERO, false> for KI
where
    FS: [const] AccumOrderedStride<FKFull, KI, I, 0, SKIP_ZERO, Int<1>>,
    FKFull: FlatTuple,
    KI: OrderKey,
{
    type Output = <FS as AccumOrderedStride<FKFull, KI, I, 0, SKIP_ZERO, Int<1>>>::Output;
    fn leaf_ordered_stride(fs: &FS, fk_full: &FKFull) -> Self::Output {
        fs.accum_ordered_stride(fk_full, Int::<1>)
    }
}

/// Maps each key in flat key suffix `Self` (starting at flat index `I`) to its ordered compact
/// stride against full flat shape `FS` and full flat keys `FKFull`.
///
/// Setting `SKIP_ZERO = true` preserves static zero-stride modes for [`MakeLayoutLike`].
#[doc(hidden)]
pub const trait MapOrderedStrides<
    FS: FlatTuple,
    FKFull: FlatTuple,
    const I: int,
    const SKIP_ZERO: bool,
>: FlatTuple
{
    type Output: FlatTuple;
    fn map_ordered_strides(&self, fs: &FS, fk_full: &FKFull) -> Self::Output;
}

const impl<FS: FlatTuple, FKFull: FlatTuple, const I: int, const SKIP_ZERO: bool>
    MapOrderedStrides<FS, FKFull, I, SKIP_ZERO> for Nil
{
    type Output = Nil;
    fn map_ordered_strides(&self, _: &FS, _: &FKFull) -> Nil {
        Nil
    }
}

const impl<FS, FKFull, KI, KT, const I: int, const SKIP_ZERO: bool>
    MapOrderedStrides<FS, FKFull, I, SKIP_ZERO> for Til<KI, KT>
where
    FS: FlatTuple,
    FKFull: FlatTuple,
    KI: [const] LeafOrderedStride<FS, FKFull, I, SKIP_ZERO, { ZERO_OUT_FLAG::<KI, SKIP_ZERO> }>,
    KT: [const] MapOrderedStrides<FS, FKFull, { INCREASE::<I> }, SKIP_ZERO>,
{
    type Output =
        Til<
            <KI as LeafOrderedStride<
                FS,
                FKFull,
                I,
                SKIP_ZERO,
                { ZERO_OUT_FLAG::<KI, SKIP_ZERO> },
            >>::Output,
            <KT as MapOrderedStrides<FS, FKFull, { INCREASE::<I> }, SKIP_ZERO>>::Output,
        >;

    fn map_ordered_strides(&self, fs: &FS, fk_full: &FKFull) -> Self::Output {
        Til(
            KI::leaf_ordered_stride(fs, fk_full),
            self.1.map_ordered_strides(fs, fk_full),
        )
    }
}

/// Builds a compact [`Layout`] with shape `self` whose strides follow the ascending
/// order induced by `order` (ties and dynamic keys preserve left-to-right order).
pub const trait MakeOrderedLayout<O>: Shape {
    /// Resulting compact [`Layout`].
    type Output;

    /// Constructs the compact ordered layout from `self` and `order`.
    fn make_ordered_layout(&self, order: &O) -> Self::Output;
}

const impl<S, O> MakeOrderedLayout<O> for S
where
    S: [const] Shape + [const] Flatten + [const] Congruent<O>,
    O: [const] Flatten,
    O::Output: [const] MapOrderedStrides<S::Output, O::Output, 0, false>,
    <O::Output as MapOrderedStrides<S::Output, O::Output, 0, false>>::Output:
        [const] Unflatten<S::Profile>,
    <<O::Output as MapOrderedStrides<S::Output, O::Output, 0, false>>::Output as Unflatten<
        S::Profile,
    >>::Output: Stride<S>,
{
    type Output = Layout<
        S,
        <<O::Output as MapOrderedStrides<S::Output, O::Output, 0, false>>::Output as Unflatten<
            S::Profile,
        >>::Output,
    >;

    fn make_ordered_layout(&self, order: &O) -> Self::Output {
        let flat_s = self.flatten();
        let flat_o = order.flatten();
        let flat_d = flat_o.map_ordered_strides(&flat_s, &flat_o);
        let stride = unflatten_like(&flat_d, self);
        Layout::new(*self, stride)
    }
}

/// Constructs a compact [`Layout`] with `shape` ordered by ascending `order`.
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::{Layout, make_ordered_layout};
///
/// let s = int_tuple![Int::<4>, Int::<8>];
/// let order = int_tuple![Int::<1>, Int::<0>];
/// assert_eq!(
///     make_ordered_layout(&s, &order),
///     Layout::new(s, int_tuple![Int::<8>, Int::<1>])
/// );
/// ```
pub const fn make_ordered_layout<S: [const] MakeOrderedLayout<O>, O>(
    shape: &S,
    order: &O,
) -> S::Output {
    shape.make_ordered_layout(order)
}

/// Builds a compact [`Layout`] with the same shape as `self` whose non-zero strides
/// follow the relative ordering induced by `self`'s strides.
pub const trait MakeLayoutLike: Copy {
    /// Resulting compact [`Layout`].
    type Output;

    /// Constructs the compact layout preserving static stride-0 modes.
    fn make_layout_like(&self) -> Self::Output;
}

const impl<S, D> MakeLayoutLike for Layout<S, D>
where
    S: [const] Shape + [const] Flatten,
    D: Stride<S> + [const] Flatten,
    D::Output: [const] MapOrderedStrides<S::Output, D::Output, 0, true>,
    <D::Output as MapOrderedStrides<S::Output, D::Output, 0, true>>::Output:
        [const] Unflatten<S::Profile>,
    <<D::Output as MapOrderedStrides<S::Output, D::Output, 0, true>>::Output as Unflatten<
        S::Profile,
    >>::Output: Stride<S>,
{
    type Output = Layout<
        S,
        <<D::Output as MapOrderedStrides<S::Output, D::Output, 0, true>>::Output as Unflatten<
            S::Profile,
        >>::Output,
    >;

    fn make_layout_like(&self) -> Self::Output {
        let flat_s = self.shape.flatten();
        let flat_d_in = self.stride.flatten();
        let flat_d = flat_d_in.map_ordered_strides(&flat_s, &flat_d_in);
        let stride = unflatten_like(&flat_d, &self.shape);
        Layout::new(self.shape, stride)
    }
}

/// Constructs a compact [`Layout`] matching `layout`'s shape and relative stride order.
pub const fn make_layout_like<L: [const] MakeLayoutLike>(layout: &L) -> L::Output {
    layout.make_layout_like()
}
