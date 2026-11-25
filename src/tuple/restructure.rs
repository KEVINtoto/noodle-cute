//! Structural transformations on hierarchical tuples.
//!
//! Provides compile-time and runtime restructuring operations that reshape tuple
//! hierarchies while preserving concrete leaf types and staticness.
//!
//! # Main Interfaces
//!
//! - **Wrapping & concatenation**: [`Wrap`], [`Unwrap`], [`Append`], [`Prepend`].
//! - **Flattening & reconstruction**: [`Flatten`], [`Unflatten`], [`unflatten`],
//!   [`unflatten_like`], [`RepeatLike`], [`repeat_like`].
//! - **Path-addressed updates & lifting**: [`ReplaceAt`], [`Replace`], [`replace`],
//!   [`Lift`], [`lift`].
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{int_tuple, mode};
//! use noodle_cute::tuple::{Flatten, MakeTuple, lift, repeat_like, replace, unflatten_like};
//!
//! let t = int_tuple![1, (2, 3), 4];
//! let flat = t.flatten();
//! assert_eq!(unflatten_like(&flat, &t), t);
//! assert_eq!(repeat_like(0, &t), int_tuple![0, (0, 0), 0]);
//! assert_eq!(replace::<mode![1, -1], _, _>(&t, 9), int_tuple![1, (2, 9), 4]);
//! assert_eq!(lift::<mode![1, 0], _, _, _>(&7, &MakeTuple, 0), int_tuple![0, (7)]);
//! ```
//!
//! # Notes
//!
//! - [`Unwrap`] strips only enclosing singleton tuples, stopping at leaves, [`Nil`],
//!   or tuples of rank `>= 2`.
//! - [`Flatten`] drops nested [`Nil`]s, whereas [`Unflatten`] restores [`Nil`] positions
//!   dictated by the target profile and rejects mismatched leaf counts at compile time.
//! - [`Replace`] wraps a scalar leaf for single-step paths (`mode!(0)` / `mode!(-1)`).
//! - [`Lift`] inserts padding before the child for non-negative indices and after for
//!   negative indices.

use crate::integer::{DECREASE, IS_NON_NEG, IS_ZERO, int, sub};

use super::{
    FlatTuple, Get, GetOutput, Item, Leaf, LeafProfile, MakeNodeFunctor, Mode, ModeEnd,
    NORMALIZED_INDEX, Nil, NilProfile, Til, Tuple, TupleProfile,
};

/// Wraps scalar leaves in a singleton tuple and preserves built-in tuple containers.
///
/// A leaf becomes `Til::new(self, Nil)`. [`Nil`] and
/// [`Til`](super::Til) return copies of themselves. Custom leaf
/// types automatically inherit the [`Leaf`] implementation.
///
/// # Examples
///
/// ```
/// # #![feature(const_trait_impl, const_ops, const_cmp)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::tuple::{Nil, Til, Wrap};
///
/// const WRAPPED: Til<Int<3>, Nil> = Int::<3>.wrap();
/// assert_eq!(WRAPPED, Til::new(Int::<3>, Nil));
/// assert_eq!(7i32.wrap(), Til::new(7i32, Nil));
/// let nested = Til::new(1i32, Til::new(Til::new(2i32, Nil), Nil));
/// assert_eq!(nested.wrap(), nested);
/// assert_eq!(Nil.wrap(), Nil);
/// ```
pub const trait Wrap: [const] Item {
    /// The singleton tuple for a leaf, or the original tuple type.
    type Wrapped: [const] Tuple;
    /// Copies this item into its wrapped representation.
    fn wrap(&self) -> Self::Wrapped;
}

const impl Wrap for Nil {
    type Wrapped = Self;
    fn wrap(&self) -> Self::Wrapped {
        Nil
    }
}

const impl<L: [const] Leaf> Wrap for L {
    type Wrapped = Til<Self, Nil>;
    fn wrap(&self) -> Self::Wrapped {
        Til(*self, Nil)
    }
}

const impl<H: [const] Item, T: [const] Tuple> Wrap for Til<H, T> {
    type Wrapped = Self;
    fn wrap(&self) -> Self::Wrapped {
        *self
    }
}

/// Removes enclosing singleton tuples recursively.
///
/// Scalar leaves and [`Nil`] return copies of themselves. A singleton delegates
/// to its head's `Unwrap` implementation. A tuple with at least two top-level
/// modes returns a copy of itself, preserving its nested heads.
///
/// # Examples
///
/// ```
/// # #![feature(const_trait_impl, const_ops, const_cmp)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::tuple::{Nil, Til, Unwrap};
///
/// const UNWRAPPED: Int<3> = Til::new(Til::new(Int::<3>, Nil), Nil).unwrap();
/// assert_eq!(UNWRAPPED, Int::<3>);
/// assert_eq!(7i32.unwrap(), 7i32);
/// assert_eq!(Til::new(Nil, Nil).unwrap(), Nil);
/// assert_eq!(Nil.unwrap(), Nil);
///
/// let pair = Til::new(Til::new(1i32, Nil), Til::new(2i32, Nil));
/// assert_eq!(Til::new(Til::new(pair, Nil), Nil).unwrap(), pair);
/// ```
pub const trait Unwrap: [const] Item {
    /// The item reached by recursively unwrapping enclosing singletons.
    type Unwrapped: [const] Item;
    /// Copies this item and recursively removes enclosing singleton tuples.
    fn unwrap(&self) -> Self::Unwrapped;
}

const impl Unwrap for Nil {
    type Unwrapped = Self;
    fn unwrap(&self) -> Self::Unwrapped {
        Nil
    }
}

const impl<L: [const] Leaf> Unwrap for L {
    type Unwrapped = Self;
    fn unwrap(&self) -> Self::Unwrapped {
        *self
    }
}

const impl<H: [const] Unwrap> Unwrap for Til<H, Nil> {
    type Unwrapped = H::Unwrapped;
    fn unwrap(&self) -> Self::Unwrapped {
        self.0.unwrap()
    }
}

const impl<H: [const] Item, TH: [const] Item, TT: [const] Tuple> Unwrap for Til<H, Til<TH, TT>> {
    type Unwrapped = Self;
    fn unwrap(&self) -> Self::Unwrapped {
        *self
    }
}

/// Replaces a tuple chain's terminating [`Nil`] with `rhs`.
///
/// Both operands and the result implement [`Tuple`]. Concatenation preserves
/// concrete head types and nesting. Appending to `Nil` returns `rhs`.
///
/// # Examples
///
/// ```
/// # #![feature(const_trait_impl, const_ops, const_cmp)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::tuple::{Append, Nil, Til};
///
/// const APPENDED: Til<i32, Til<Int<2>, Nil>> =
///     Til::new(1i32, Nil).append(Til::new(Int::<2>, Nil));
/// assert_eq!(APPENDED, Til::new(1i32, Til::new(Int::<2>, Nil)));
///
/// let nested = Til::new(Til::new(1i32, Til::new(2i32, Nil)), Nil);
/// assert_eq!(
///     nested.append(Til::new(3i32, Nil)),
///     Til::new(nested.0, Til::new(3i32, Nil))
/// );
/// assert_eq!(Nil.append(nested), nested);
/// assert_eq!(nested.append(Nil), nested);
/// assert_eq!(Nil.append(Nil), Nil);
/// ```
///
/// An integer right operand fails the tuple bound (including when appending to `Nil`).
///
/// ```compile_fail,E0277
/// use noodle_cute::tuple::{Append, Nil, Til};
/// let _ = Til::new(1i32, Nil).append(2i32);
/// ```
///
/// ```compile_fail,E0277
/// use noodle_cute::tuple::{Append, Nil};
/// let _ = Nil.append(2i32);
/// ```
///
/// A native Rust pair also fails the tuple bound.
///
/// ```compile_fail,E0277
/// use noodle_cute::tuple::{Append, Nil};
/// let _ = Nil.append((1i32, 2i32));
/// ```
pub const trait Append<R: [const] Tuple>: [const] Tuple {
    /// The chain obtained by replacing the terminal `Nil` with `R`.
    type Output: [const] Tuple;
    /// Copies this chain and attaches `rhs` after its last head.
    fn append(&self, rhs: R) -> Self::Output;
}

const impl<R: [const] Tuple> Append<R> for Nil {
    type Output = R;
    fn append(&self, rhs: R) -> Self::Output {
        rhs
    }
}

const impl<H, T, R> Append<R> for Til<H, T>
where
    R: [const] Tuple,
    H: [const] Item,
    T: [const] Tuple + [const] Append<R>,
{
    type Output = Til<H, <T as Append<R>>::Output>;
    fn append(&self, rhs: R) -> Self::Output {
        Til(self.0, self.1.append(rhs))
    }
}

/// Inserts `rhs` as the new head mode of a [`Tuple`].
///
/// [`Nil`] becomes the singleton tuple `Til::new(rhs, Nil)`, and a
/// [`Til`](super::Til) becomes `Til::new(rhs, *self)`
/// in $O(1)$, preserving its existing modes as the tail.
///
/// # Examples
///
/// ```
/// # #![feature(const_trait_impl, const_ops, const_cmp)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::tuple::{Nil, Prepend, Til};
///
/// const PREPENDED: Til<i32, Til<Int<2>, Nil>> =
///     Til::new(Int::<2>, Nil).prepend(1i32);
/// assert_eq!(PREPENDED, Til::new(1i32, Til::new(Int::<2>, Nil)));
/// assert_eq!(Nil.prepend(0i32), Til::new(0i32, Nil));
/// assert_eq!(
///     Til::new(2i32, Nil).prepend(Til::new(1i32, Nil)),
///     Til::new(Til::new(1i32, Nil), Til::new(2i32, Nil))
/// );
/// ```
///
/// Calling `prepend` on a scalar leaf fails the tuple bound.
///
/// ```compile_fail,E0599
/// use noodle_cute::tuple::Prepend;
/// let _ = 1i32.prepend(0i32);
/// ```
///
/// A native Rust pair fails the `Item` bound on `R`.
///
/// ```compile_fail,E0277
/// use noodle_cute::tuple::{Nil, Prepend};
/// let _ = Nil.prepend((1i32, 2i32));
/// ```
pub const trait Prepend<R: [const] Item>: [const] Tuple {
    /// The tuple obtained by inserting `R` before `Self`.
    type Output: [const] Tuple;
    /// Copies this tuple and attaches `rhs` as its new head mode.
    fn prepend(&self, rhs: R) -> Self::Output;
}

const impl<T: [const] Tuple, R: [const] Item> Prepend<R> for T {
    type Output = Til<R, Self>;
    fn prepend(&self, rhs: R) -> Self::Output {
        Til(rhs, *self)
    }
}

/// Flattens `Self` right-to-left onto flat accumulator `Tail` in a single pass.
#[doc(hidden)]
pub const trait FlattenTo<Tail: [const] FlatTuple>: [const] Item {
    type Output: [const] FlatTuple;
    fn flatten_to(&self, tail: Tail) -> Self::Output;
}

const impl<Tail: [const] FlatTuple> FlattenTo<Tail> for Nil {
    type Output = Tail;
    fn flatten_to(&self, tail: Tail) -> Tail {
        tail
    }
}

const impl<L: [const] Leaf, Tail: [const] FlatTuple> FlattenTo<Tail> for L {
    type Output = Til<Self, Tail>;
    fn flatten_to(&self, tail: Tail) -> Self::Output {
        Til(*self, tail)
    }
}

const impl<H, T, Tail> FlattenTo<Tail> for Til<H, T>
where
    Tail: [const] FlatTuple,
    T: [const] Tuple + [const] FlattenTo<Tail>,
    H: [const] Item + [const] FlattenTo<T::Output>,
{
    type Output = H::Output;
    fn flatten_to(&self, tail: Tail) -> Self::Output {
        let tail_flat = self.1.flatten_to(tail);
        self.0.flatten_to(tail_flat)
    }
}

/// Flattens nested heads into one tuple chain in head-first, depth-first order.
///
/// Scalar leaves become singleton tuples, preserving their types. [`Nil`]
/// produces an empty result, including when it appears as a nested head.
///
/// # Examples
///
/// ```
/// # #![feature(const_trait_impl, const_ops, const_cmp)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::tuple::{Flatten, Nil, Til};
///
/// const FLAT: Til<Int<1>, Til<i32, Nil>> =
///     Til::new(
///         Til::new(Int::<1>, Til::new(2i32, Nil)),
///         Til::new(Nil, Nil),
///     )
///     .flatten();
/// assert_eq!(FLAT, Til::new(Int::<1>, Til::new(2i32, Nil)));
/// assert_eq!(42i32.flatten(), Til::new(42i32, Nil));
/// assert_eq!(Nil.flatten(), Nil);
/// ```
pub const trait Flatten: [const] Item {
    /// A flat tuple chain retaining the original leaf types and order.
    type Output: [const] FlatTuple;
    /// Copies the structure and concatenates its flattened heads.
    fn flatten(&self) -> Self::Output;
}

const impl Flatten for Nil {
    type Output = Nil;
    fn flatten(&self) -> Self::Output {
        Nil
    }
}

const impl<L: [const] Leaf> Flatten for L {
    type Output = Til<Self, Nil>;
    fn flatten(&self) -> Self::Output {
        self.wrap()
    }
}

const impl<H, T> Flatten for Til<H, T>
where
    H: [const] Item,
    T: [const] Tuple,
    Self: [const] FlattenTo<Nil>,
{
    type Output = <Self as FlattenTo<Nil>>::Output;
    fn flatten(&self) -> Self::Output {
        self.flatten_to(Nil)
    }
}

/// Consumes the prefix of a `FlatTuple` matching structural profile `P`, returning `(Output, Rest)`.
#[doc(hidden)]
pub const trait UnflattenPrefix<P>: [const] FlatTuple {
    type Output: [const] Item;
    type Rest: [const] FlatTuple;
    fn unflatten_prefix(&self) -> (Self::Output, Self::Rest);
}

const impl<F: [const] FlatTuple> UnflattenPrefix<NilProfile> for F {
    type Output = Nil;
    type Rest = F;
    fn unflatten_prefix(&self) -> (Nil, F) {
        (Nil, *self)
    }
}

const impl<H: [const] Leaf, T: [const] FlatTuple> UnflattenPrefix<LeafProfile> for Til<H, T> {
    type Output = H;
    type Rest = T;
    fn unflatten_prefix(&self) -> (H, T) {
        (self.0, self.1)
    }
}

const impl<F, HP, TP> UnflattenPrefix<TupleProfile<HP, TP>> for F
where
    F: [const] FlatTuple + [const] UnflattenPrefix<HP>,
    <F as UnflattenPrefix<HP>>::Rest: [const] UnflattenPrefix<TP>,
    <<F as UnflattenPrefix<HP>>::Rest as UnflattenPrefix<TP>>::Output: [const] Tuple,
{
    type Output = Til<
        <F as UnflattenPrefix<HP>>::Output,
        <<F as UnflattenPrefix<HP>>::Rest as UnflattenPrefix<TP>>::Output,
    >;
    type Rest = <<F as UnflattenPrefix<HP>>::Rest as UnflattenPrefix<TP>>::Rest;
    fn unflatten_prefix(&self) -> (Self::Output, Self::Rest) {
        let (head, rest1) = <F as UnflattenPrefix<HP>>::unflatten_prefix(self);
        let (tail, rest2) =
            <<F as UnflattenPrefix<HP>>::Rest as UnflattenPrefix<TP>>::unflatten_prefix(&rest1);
        (Til(head, tail), rest2)
    }
}

/// Reconstructs a nested tuple matching structural profile `P` from a [`FlatTuple`],
/// rejecting mismatched leaf counts at compile time.
pub const trait Unflatten<P>: [const] FlatTuple {
    /// Reconstructed nested item whose `Profile` equals `P`.
    type Output: [const] Item;
    /// Converts `self` into the nested structure prescribed by `P`.
    fn unflatten(&self) -> Self::Output;
}

const impl<F: [const] FlatTuple + [const] UnflattenPrefix<P, Rest = Nil>, P> Unflatten<P> for F {
    type Output = F::Output;
    fn unflatten(&self) -> Self::Output {
        self.unflatten_prefix().0
    }
}

/// Reconstructs a nested item matching profile `P` from a flat tuple `flat`.
pub const fn unflatten<P, F: [const] Unflatten<P>>(flat: &F) -> F::Output {
    flat.unflatten()
}

/// Reconstructs a nested item matching the structure of guide item `guide` from `flat`.
pub const fn unflatten_like<F: [const] Unflatten<G::Profile>, G: Item>(
    flat: &F,
    _: &G,
) -> F::Output {
    flat.unflatten()
}

/// Constructs a hierarchy matching `Self`'s structural profile with copies of leaf item `X`.
#[doc(hidden)]
pub const trait RepeatLikeProfile<X: [const] Item> {
    type Output: [const] Item;
    fn repeat_like_profile(x: X) -> Self::Output;
}

const impl<X: [const] Item> RepeatLikeProfile<X> for NilProfile {
    type Output = Nil;
    fn repeat_like_profile(_: X) -> Nil {
        Nil
    }
}

const impl<X: [const] Item> RepeatLikeProfile<X> for LeafProfile {
    type Output = X;
    fn repeat_like_profile(x: X) -> X {
        x
    }
}

const impl<X: [const] Item, HP: [const] RepeatLikeProfile<X>, TP: [const] RepeatLikeProfile<X>>
    RepeatLikeProfile<X> for TupleProfile<HP, TP>
where
    TP::Output: [const] Tuple,
{
    type Output = Til<HP::Output, TP::Output>;
    fn repeat_like_profile(x: X) -> Self::Output {
        Til(HP::repeat_like_profile(x), TP::repeat_like_profile(x))
    }
}

/// Builds a structure congruent to `Self` with every leaf position holding `x`.
pub const trait RepeatLike<X: [const] Item>: [const] Item {
    /// Output item congruent to `Self`.
    type Output: [const] Item;
    /// Fills every leaf slot of `Self`'s profile with `x`.
    fn repeat_like(&self, x: X) -> Self::Output;
}

const impl<G: [const] Item, X: [const] Item> RepeatLike<X> for G
where
    G::Profile: [const] RepeatLikeProfile<X>,
{
    type Output = <G::Profile as RepeatLikeProfile<X>>::Output;
    fn repeat_like(&self, x: X) -> Self::Output {
        <G::Profile as RepeatLikeProfile<X>>::repeat_like_profile(x)
    }
}

/// Repeats `x` across every leaf position matching the structure of `guide`.
pub const fn repeat_like<X: [const] Item, G: [const] RepeatLike<X>>(x: X, guide: &G) -> G::Output {
    guide.repeat_like(x)
}

/// Recursively replaces normalized mode `NORM` with `X`; `FIRST = true` replaces the current head.
#[doc(hidden)]
pub const trait ReplaceImpl<X: [const] Item, const FIRST: bool, const NORM: int>:
    [const] Item
{
    type Output: [const] Tuple;
    fn replace_impl(&self, x: X) -> Self::Output;
}

const impl<H: [const] Item, T: [const] Tuple, X: [const] Item, const NORM: int>
    ReplaceImpl<X, true, NORM> for Til<H, T>
{
    type Output = Til<X, T>;
    fn replace_impl(&self, x: X) -> Self::Output {
        Til(x, self.1)
    }
}

const impl<H: [const] Item, T: [const] Tuple, X: [const] Item, const NORM: int>
    ReplaceImpl<X, false, NORM> for Til<H, T>
where
    T: [const] ReplaceImpl<X, { IS_ZERO::<{ DECREASE::<NORM> }> }, { DECREASE::<NORM> }>,
{
    type Output = Til<
        H,
        <T as ReplaceImpl<X, { IS_ZERO::<{ DECREASE::<NORM> }> }, { DECREASE::<NORM> }>>::Output,
    >;
    fn replace_impl(&self, x: X) -> Self::Output {
        Til(self.0, self.1.replace_impl(x))
    }
}

/// Replaces top-level mode `I` of a [`Til`](super::Til) with `x`.
pub const trait ReplaceAt<X: [const] Item, const I: int>: [const] Item {
    /// Resulting tuple after replacing mode `I` with `X`.
    type Output: [const] Tuple;
    /// Replaces mode `I` (supporting negative indices) with `x`.
    fn replace_at(&self, x: X) -> Self::Output;
}

const impl<H: [const] Item, T: [const] Tuple, X: [const] Item, const I: int> ReplaceAt<X, I>
    for Til<H, T>
where
    Self: [const] ReplaceImpl<
            X,
            { IS_ZERO::<{ NORMALIZED_INDEX::<I, Self> }> },
            { NORMALIZED_INDEX::<I, Self> },
        >,
{
    type Output = <Self as ReplaceImpl<
        X,
        { IS_ZERO::<{ NORMALIZED_INDEX::<I, Self> }> },
        { NORMALIZED_INDEX::<I, Self> },
    >>::Output;
    fn replace_at(&self, x: X) -> Self::Output {
        <Self as ReplaceImpl<
            X,
            { IS_ZERO::<{ NORMALIZED_INDEX::<I, Self> }> },
            { NORMALIZED_INDEX::<I, Self> },
        >>::replace_impl(self, x)
    }
}

/// Recursively replaces the subtree located at signed mode `Path` with `x`.
pub const trait Replace<X: [const] Item, Path>: [const] Item {
    /// Resulting item after replacement.
    type Output: [const] Item;
    /// Copies `self` with the entry at `Path` replaced by `x`.
    fn replace(&self, x: X) -> Self::Output;
}

const impl<T: [const] Item, X: [const] Item> Replace<X, ModeEnd> for T {
    type Output = X;
    fn replace(&self, x: X) -> X {
        x
    }
}

const impl<T, X: [const] Item, const I: int, Next> Replace<X, Mode<I, Next>> for T
where
    T: [const] Wrap,
    T::Wrapped: [const] Get<I>,
    GetOutput<T::Wrapped, I>: [const] Replace<X, Next>,
    T::Wrapped: [const] ReplaceAt<<GetOutput<T::Wrapped, I> as Replace<X, Next>>::Output, I>,
{
    type Output = <T::Wrapped as ReplaceAt<
        <GetOutput<T::Wrapped, I> as Replace<X, Next>>::Output,
        I,
    >>::Output;
    fn replace(&self, x: X) -> Self::Output {
        let wrapped = self.wrap();
        let sub = <T::Wrapped as Get<I>>::get(&wrapped);
        let updated = <GetOutput<T::Wrapped, I> as Replace<X, Next>>::replace(&sub, x);
        <T::Wrapped as ReplaceAt<_, I>>::replace_at(&wrapped, updated)
    }
}

/// Replaces the subtree at signed `Path` in `t` with `x`.
pub const fn replace<Path, T: [const] Replace<X, Path>, X: [const] Item>(t: &T, x: X) -> T::Output {
    t.replace(x)
}

/// Leading sibling pad count for [`Lift`] at signed mode index `I`.
#[doc(hidden)]
pub const LIFT_BEFORE<const I: int>: int = if IS_NON_NEG::<I> { I } else { 0 };
/// Trailing sibling pad count for [`Lift`] at signed mode index `I`.
#[doc(hidden)]
pub const LIFT_AFTER<const I: int>: int = if IS_NON_NEG::<I> { 0 } else { sub(-1, I) };

/// Prepends `N` copies of `Pad` before `Tail` (`DONE = true` when `N == 0`).
#[doc(hidden)]
pub const trait PadRepeat<Pad: Copy, Tail: Copy, const N: int, const DONE: bool> {
    type Output: Copy;
    fn pad_repeat(pad: Pad, tail: Tail) -> Self::Output;
}

const impl<Pad: Copy, Tail: Copy, const N: int> PadRepeat<Pad, Tail, N, true> for () {
    type Output = Tail;
    fn pad_repeat(_: Pad, tail: Tail) -> Self::Output {
        tail
    }
}

const impl<Pad: Copy, Tail: Copy, const N: int> PadRepeat<Pad, Tail, N, false> for ()
where
    (): [const] PadRepeat<Pad, Tail, { DECREASE::<N> }, { IS_ZERO::<{ DECREASE::<N> }> }>,
{
    type Output = Til<
        Pad,
        <() as PadRepeat<Pad, Tail, { DECREASE::<N> }, { IS_ZERO::<{ DECREASE::<N> }> }>>::Output,
    >;
    fn pad_repeat(pad: Pad, tail: Tail) -> Self::Output {
        Til(
            pad,
            <() as PadRepeat<Pad, Tail, { DECREASE::<N> }, { IS_ZERO::<{ DECREASE::<N> }> }>>::pad_repeat(pad, tail),
        )
    }
}

/// Trailing `Pad` chain placed after the lifted child at signed mode index `I`.
#[doc(hidden)]
pub type AfterPads<Pad, const I: int> =
    <() as PadRepeat<Pad, Nil, { LIFT_AFTER::<I> }, { IS_ZERO::<{ LIFT_AFTER::<I> }> }>>::Output;

/// Full child chain `[Pad..., Inner, Pad...]` constructed at one [`Lift`] level for mode index `I`.
#[doc(hidden)]
pub type LiftChildren<Inner, Pad, const I: int> = <() as PadRepeat<
    Pad,
    Til<Inner, AfterPads<Pad, I>>,
    { LIFT_BEFORE::<I> },
    { IS_ZERO::<{ LIFT_BEFORE::<I> }> },
>>::Output;

/// Wraps an item in enclosing tuple levels at signed `Path`, padding sibling slots with `pad`.
pub const trait Lift<Make, Pad: Copy, Path>: Copy {
    /// Resulting lifted node type.
    type Output: Copy;
    /// Lifts `self` along `Path` using `make` and filler `pad`.
    fn lift(&self, make: &Make, pad: Pad) -> Self::Output;
}

const impl<T: Copy, Make, Pad: Copy> Lift<Make, Pad, ModeEnd> for T {
    type Output = T;
    fn lift(&self, _: &Make, _: Pad) -> T {
        *self
    }
}

const impl<T, Make, Pad: Copy, const I: int, Next> Lift<Make, Pad, Mode<I, Next>> for T
where
    T: [const] Lift<Make, Pad, Next>,
    (): [const] PadRepeat<Pad, Nil, { LIFT_AFTER::<I> }, { IS_ZERO::<{ LIFT_AFTER::<I> }> }>,
    (): [const] PadRepeat<
            Pad,
            Til<<T as Lift<Make, Pad, Next>>::Output, AfterPads<Pad, I>>,
            { LIFT_BEFORE::<I> },
            { IS_ZERO::<{ LIFT_BEFORE::<I> }> },
        >,
    Make: [const] MakeNodeFunctor<LiftChildren<<T as Lift<Make, Pad, Next>>::Output, Pad, I>>,
    <Make as MakeNodeFunctor<LiftChildren<<T as Lift<Make, Pad, Next>>::Output, Pad, I>>>::Output:
        Copy,
{
    type Output = <Make as MakeNodeFunctor<
        LiftChildren<<T as Lift<Make, Pad, Next>>::Output, Pad, I>,
    >>::Output;
    fn lift(&self, make: &Make, pad: Pad) -> Self::Output {
        let inner = <T as Lift<Make, Pad, Next>>::lift(self, make, pad);
        let after = <() as PadRepeat<
            Pad,
            Nil,
            { LIFT_AFTER::<I> },
            { IS_ZERO::<{ LIFT_AFTER::<I> }> },
        >>::pad_repeat(pad, Nil);
        let children = <() as PadRepeat<
            Pad,
            Til<<T as Lift<Make, Pad, Next>>::Output, AfterPads<Pad, I>>,
            { LIFT_BEFORE::<I> },
            { IS_ZERO::<{ LIFT_BEFORE::<I> }> },
        >>::pad_repeat(pad, Til(inner, after));
        make.make(children)
    }
}

/// Lifts `t` along `Path` using node builder `make` and padding element `pad`.
pub const fn lift<Path, Make, Pad: Copy, T: [const] Lift<Make, Pad, Path>>(
    t: &T,
    make: &Make,
    pad: Pad,
) -> T::Output {
    t.lift(make, pad)
}
