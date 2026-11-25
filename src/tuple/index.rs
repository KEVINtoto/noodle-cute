//! Mode paths, single-mode indexing (`Get`), multi-mode selection (`Select`), and
//! top-level slicing (`SliceRange`).
//!
//! # Main Interfaces
//!
//! - **Mode paths**: [`Mode`] and [`ModeEnd`], built with [`mode!`](crate::mode!).
//! - **Single-mode indexing**: [`Get`], [`GetOutput`], [`NORMALIZED_INDEX`],
//!   [`Til::get`](super::Til::get), and [`walk_mode!`](crate::walk_mode!).
//! - **Multi-mode selection**: [`Select`] and [`select`].
//! - **Top-level slicing**: [`SliceRange`], [`slice_range`], [`Til::slice`](super::Til::slice),
//!   [`resolve_tuple_slice`], and [`SLICE_COUNT`].
//!
//! # Usage
//!
//! ```
//! # #![feature(min_adt_const_params, min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{mode, slice, tuple, walk_mode};
//! use noodle_cute::tuple::select;
//!
//! let t = tuple!(10i32, (20i32, 30i32), 40i32);
//! assert_eq!(t.get::<-1>(), 40);
//! assert_eq!(walk_mode!(t, 1, 0), 20);
//! assert_eq!(select::<mode!(-1, 0), _>(&t), tuple!(40i32, 10i32));
//! assert_eq!(t.slice::<{ slice!([..] : -2) }>(), tuple!(40i32, 10i32));
//! ```
//!
//! # Notes
//!
//! - Signed indices range over `-RANK..RANK` with negative indices counting from the end.
//! - Out-of-range positive indices fail trait resolution (`E0277`), whereas out-of-range
//!   negative indices fail `const` evaluation (`E0080`) during normalization.
//! - [`SliceRange`] wraps scalar items into a singleton tuple before slicing and fails
//!   at compile time if `S.step == 0`.

use core::marker::PhantomData;

use crate::integer::{DECREASE, IS_NON_NEG, IS_ONE, IS_ZERO, NOT_ZERO, abs, add, int, mul, sub};
use crate::slice::{ResolvedSlice, Slice};

use super::{Append, Item, Leaf, Nil, Til, Tuple, Wrap};

/// Type-level mode index path node constructed by [`mode!`](crate::mode).
///
/// Tuple operations ([`Replace`](super::Replace), [`Lift`](super::Lift), [`Select`])
/// accept signed [`int`] indices.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Mode<const I: int, Next = ModeEnd>(PhantomData<Next>);

/// Terminates a type-level mode index path for [`Replace`](super::Replace),
/// [`Lift`](super::Lift), and [`Select`].
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct ModeEnd;

/// Normalizes a signed mode index `I` into a zero-based index using [`Item::RANK`].
///
/// Nonnegative indices retain their value. Negative indices add `I` to `T::RANK`
/// via [`crate::integer::add`]. An out-of-range negative index triggers a
/// compile-time assertion when the constant is evaluated. [`Get`] resolves the
/// upper rank bound.
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::tuple::{NORMALIZED_INDEX, Nil, Til};
///
/// type Pair = Til<i32, Til<i32, Nil>>;
/// assert_eq!(NORMALIZED_INDEX::<0, Pair>, 0);
/// assert_eq!(NORMALIZED_INDEX::<-1, Pair>, 1);
/// assert_eq!(NORMALIZED_INDEX::<-2, Pair>, 0);
/// // Normalization alone also accepts an index equal to the rank.
/// assert_eq!(NORMALIZED_INDEX::<2, Pair>, 2);
/// ```
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::tuple::{NORMALIZED_INDEX, Nil, Til};
/// const INVALID: i32 = NORMALIZED_INDEX::<-2, Til<i32, Nil>>;
/// ```
pub const NORMALIZED_INDEX<const I: int, T: Item>: int = {
    if IS_NON_NEG::<I> {
        I
    } else {
        let idx = add(T::RANK, I);
        assert!(idx >= 0, "negative tuple index out of range");
        idx
    }
};

/// Recursively resolves normalized mode index `NORM` along a `Til` chain.
/// `FIRST = true` selects the current head; `FIRST = false` decrements `NORM` and recurses on the tail.
#[doc(hidden)]
pub const trait GetImpl<const FIRST: bool, const NORM: int>: [const] Item {
    type Output: [const] Item;
    fn get(&self) -> Self::Output;
}

const impl<H, T, const NORM: int> GetImpl<true, NORM> for Til<H, T>
where
    H: [const] Item,
    T: [const] Tuple,
{
    type Output = H;
    fn get(&self) -> Self::Output {
        self.0
    }
}

const impl<H, T, const NORM: int> GetImpl<false, NORM> for Til<H, T>
where
    H: [const] Item,
    T: [const] Tuple + [const] GetImpl<{ IS_ZERO::<{ DECREASE::<NORM> }> }, { DECREASE::<NORM> }>,
{
    type Output = <T as GetImpl<{ IS_ZERO::<{ DECREASE::<NORM> }> }, { DECREASE::<NORM> }>>::Output;
    fn get(&self) -> Self::Output {
        self.1.get()
    }
}

/// Selects mode `I` of a [`Til`] by copying the selected item.
///
/// Valid indices range from `-RANK` through `RANK - 1`. Negative indices count
/// backwards from the end. Selection preserves the concrete item type and any
/// nested structure. An invalid index fails at compile time through index
/// normalization or trait resolution.
///
/// The trait method supports const evaluation. [`Til::get`](super::Til::get)
/// provides an inherent convenience method for both `const` and run-time calls.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::tuple::{Get, Nil, Til};
///
/// let t = Til::new(
///     10i32,
///     Til::new(
///         Til::new(20i32, Til::new(30i32, Nil)),
///         Til::new(40i32, Nil),
///     ),
/// );
/// assert_eq!(Get::<0>::get(&t), 10);
/// assert_eq!(Get::<-1>::get(&t), 40);
/// assert_eq!(Get::<-3>::get(&t), 10);
/// assert_eq!(Get::<1>::get(&t), Til::new(20i32, Til::new(30i32, Nil)));
/// const LAST: Int<2> =
///     Get::<-1>::get(&Til::new(Int::<1>, Til::new(Int::<2>, Nil)));
/// assert_eq!(LAST, Int::<2>);
/// ```
///
/// # Invalid indices
///
/// An index equal to the rank reaches the end of the chain without a matching implementation.
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::tuple::{Nil, Til};
/// let _ = Til::new(10i32, Til::new(20i32, Nil)).get::<2>();
/// ```
///
/// An index below `-RANK` fails the normalization assertion.
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::tuple::{Nil, Til};
/// let _ = Til::new(10i32, Til::new(20i32, Nil)).get::<-3>();
/// ```
pub const trait Get<const I: int>: [const] Item {
    /// The concrete type of the item at mode `I`.
    type Output: [const] Item;
    /// Copies the item at mode `I`.
    fn get(&self) -> Self::Output;
}

const impl<H, T, const I: int> Get<I> for Til<H, T>
where
    H: [const] Item,
    T: [const] Tuple,
    Self: [const] GetImpl<
            { IS_ZERO::<{ NORMALIZED_INDEX::<I, Self> }> },
            { NORMALIZED_INDEX::<I, Self> },
        >,
{
    type Output = <Self as GetImpl<
        { IS_ZERO::<{ NORMALIZED_INDEX::<I, Self> }> },
        { NORMALIZED_INDEX::<I, Self> },
    >>::Output;
    fn get(&self) -> Self::Output {
        <Self as GetImpl<
            { IS_ZERO::<{ NORMALIZED_INDEX::<I, Self> }> },
            { NORMALIZED_INDEX::<I, Self> },
        >>::get(self)
    }
}

/// The concrete item type selected by [`Get<I>`] from tuple `T`.
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::tuple::{GetOutput, Nil, Til};
///
/// type Pair = Til<Int<3>, Til<Int<4>, Nil>>;
/// let pair = Til::new(Int::<3>, Til::new(Int::<4>, Nil));
/// let last: GetOutput<Pair, -1> = pair.get::<-1>();
/// let _: Int<4> = last;
/// ```
pub type GetOutput<T, const I: int> = <T as Get<I>>::Output;

/// Collects a flat tuple of modes at the list of signed indices `Indices` ([`Mode`]).
pub const trait Select<Indices>: [const] Item {
    /// Resulting tuple containing the selected modes in order.
    type Output: [const] Tuple;
    /// Selects all modes listed in `Indices`.
    fn select(&self) -> Self::Output;
}

const impl<T: [const] Item> Select<ModeEnd> for T {
    type Output = Nil;
    fn select(&self) -> Nil {
        Nil
    }
}

const impl<T, const I: int, Next> Select<Mode<I, Next>> for T
where
    T: [const] Get<I> + [const] Select<Next>,
{
    type Output = Til<GetOutput<T, I>, <T as Select<Next>>::Output>;
    fn select(&self) -> Self::Output {
        Til(<T as Get<I>>::get(self), <T as Select<Next>>::select(self))
    }
}

/// Selects modes listed in `Indices` from `t` into a [`Tuple`].
pub const fn select<Indices, T: [const] Select<Indices>>(t: &T) -> T::Output {
    t.select()
}

/// Resolves `S` against a tuple of rank `rank`, panicking if `S.step == 0`
/// or `rank < 0`.
pub const fn resolve_tuple_slice(s: Slice, rank: int) -> ResolvedSlice {
    match s.resolve(rank) {
        Some(resolved) => resolved,
        None => panic!("invalid tuple slice: step cannot be zero and rank must be non-negative"),
    }
}

/// Element count of a slice on item `T`.
pub const SLICE_COUNT<T: Item, const S: Slice>: int = resolve_tuple_slice(S, T::RANK).len;

const SLICE_ABS_STEP<T: Item, const S: Slice>: int = {
    let r = resolve_tuple_slice(S, T::RANK);
    if r.len <= 1 { 1 } else { abs(r.step) }
};

const SLICE_IS_REVERSE<const S: Slice>: bool = S.step < 0;

// Lowest visited index so both forward and reverse walkers traverse left-to-right in one pass.
const SLICE_FIRST_INDEX<T: Item, const S: Slice>: int = {
    let r = resolve_tuple_slice(S, T::RANK);
    if r.len == 0 {
        0
    } else if r.step > 0 || r.len == 1 {
        r.start
    } else {
        add(r.start, mul(sub(r.len, 1), r.step))
    }
};

/// Single-pass forward slice walker over a `Tuple`.
/// `SKIP`: elements to skip before taking; `STEP`: stride between taken elements;
/// `COUNT`: remaining elements to emit; `DONE`: `COUNT == 0`; `SKIPPING`: `SKIP != 0`; `LAST`: `COUNT == 1`.
#[doc(hidden)]
pub const trait SliceForward<
    const SKIP: int,
    const STEP: int,
    const COUNT: int,
    const DONE: bool,
    const SKIPPING: bool,
    const LAST: bool,
>: [const] Tuple
{
    type Output: [const] Tuple;
    fn slice_forward(&self) -> Self::Output;
}

const impl<
    T: [const] Tuple,
    const SKIP: int,
    const STEP: int,
    const COUNT: int,
    const SKIPPING: bool,
    const LAST: bool,
> SliceForward<SKIP, STEP, COUNT, true, SKIPPING, LAST> for T
{
    type Output = Nil;
    fn slice_forward(&self) -> Nil {
        Nil
    }
}

const impl<H: [const] Item, T, const SKIP: int, const STEP: int, const COUNT: int, const LAST: bool>
    SliceForward<SKIP, STEP, COUNT, false, true, LAST> for Til<H, T>
where
    T: [const] Tuple
        + [const] SliceForward<
            { DECREASE::<SKIP> },
            STEP,
            COUNT,
            false,
            { NOT_ZERO::<{ DECREASE::<SKIP> }> },
            LAST,
        >,
{
    type Output = <T as SliceForward<
        { DECREASE::<SKIP> },
        STEP,
        COUNT,
        false,
        { NOT_ZERO::<{ DECREASE::<SKIP> }> },
        LAST,
    >>::Output;
    fn slice_forward(&self) -> Self::Output {
        self.1.slice_forward()
    }
}

const impl<H: [const] Item, T: [const] Tuple, const SKIP: int, const STEP: int, const COUNT: int>
    SliceForward<SKIP, STEP, COUNT, false, false, true> for Til<H, T>
{
    type Output = Til<H, Nil>;
    fn slice_forward(&self) -> Self::Output {
        Til(self.0, Nil)
    }
}

const impl<H: [const] Item, T, const SKIP: int, const STEP: int, const COUNT: int>
    SliceForward<SKIP, STEP, COUNT, false, false, false> for Til<H, T>
where
    T: [const] Tuple
        + [const] SliceForward<
            { DECREASE::<STEP> },
            STEP,
            { DECREASE::<COUNT> },
            false,
            { NOT_ZERO::<{ DECREASE::<STEP> }> },
            { IS_ONE::<{ DECREASE::<COUNT> }> },
        >,
{
    type Output = Til<
        H,
        <T as SliceForward<
            { DECREASE::<STEP> },
            STEP,
            { DECREASE::<COUNT> },
            false,
            { NOT_ZERO::<{ DECREASE::<STEP> }> },
            { IS_ONE::<{ DECREASE::<COUNT> }> },
        >>::Output,
    >;
    fn slice_forward(&self) -> Self::Output {
        Til(self.0, self.1.slice_forward())
    }
}

/// Single-pass reverse slice walker: visits selected elements left-to-right and prepends
/// each onto `Acc` to reverse output order. Const flags match `SliceForward`.
#[doc(hidden)]
pub const trait SliceReverse<
    Acc: [const] Tuple,
    const SKIP: int,
    const STEP: int,
    const COUNT: int,
    const DONE: bool,
    const SKIPPING: bool,
    const LAST: bool,
>: [const] Tuple
{
    type Output: [const] Tuple;
    fn slice_reverse(&self, acc: Acc) -> Self::Output;
}

const impl<
    T: [const] Tuple,
    Acc: [const] Tuple,
    const SKIP: int,
    const STEP: int,
    const COUNT: int,
    const SKIPPING: bool,
    const LAST: bool,
> SliceReverse<Acc, SKIP, STEP, COUNT, true, SKIPPING, LAST> for T
{
    type Output = Acc;
    fn slice_reverse(&self, acc: Acc) -> Self::Output {
        acc
    }
}

const impl<
    H: [const] Item,
    T,
    Acc: [const] Tuple,
    const SKIP: int,
    const STEP: int,
    const COUNT: int,
    const LAST: bool,
> SliceReverse<Acc, SKIP, STEP, COUNT, false, true, LAST> for Til<H, T>
where
    T: [const] Tuple
        + [const] SliceReverse<
            Acc,
            { DECREASE::<SKIP> },
            STEP,
            COUNT,
            false,
            { NOT_ZERO::<{ DECREASE::<SKIP> }> },
            LAST,
        >,
{
    type Output = <T as SliceReverse<
        Acc,
        { DECREASE::<SKIP> },
        STEP,
        COUNT,
        false,
        { NOT_ZERO::<{ DECREASE::<SKIP> }> },
        LAST,
    >>::Output;
    fn slice_reverse(&self, acc: Acc) -> Self::Output {
        self.1.slice_reverse(acc)
    }
}

const impl<
    H: [const] Item,
    T: [const] Tuple,
    Acc: [const] Tuple,
    const SKIP: int,
    const STEP: int,
    const COUNT: int,
> SliceReverse<Acc, SKIP, STEP, COUNT, false, false, true> for Til<H, T>
{
    type Output = Til<H, Acc>;
    fn slice_reverse(&self, acc: Acc) -> Self::Output {
        Til(self.0, acc)
    }
}

const impl<
    H: [const] Item,
    T,
    Acc: [const] Tuple,
    const SKIP: int,
    const STEP: int,
    const COUNT: int,
> SliceReverse<Acc, SKIP, STEP, COUNT, false, false, false> for Til<H, T>
where
    T: [const] Tuple
        + [const] SliceReverse<
            Til<H, Acc>,
            { DECREASE::<STEP> },
            STEP,
            { DECREASE::<COUNT> },
            false,
            { NOT_ZERO::<{ DECREASE::<STEP> }> },
            { IS_ONE::<{ DECREASE::<COUNT> }> },
        >,
{
    type Output = <T as SliceReverse<
        Til<H, Acc>,
        { DECREASE::<STEP> },
        STEP,
        { DECREASE::<COUNT> },
        false,
        { NOT_ZERO::<{ DECREASE::<STEP> }> },
        { IS_ONE::<{ DECREASE::<COUNT> }> },
    >>::Output;
    fn slice_reverse(&self, acc: Acc) -> Self::Output {
        self.1.slice_reverse(Til(self.0, acc))
    }
}

/// Routes a normalized tuple slice to `SliceForward` (`REVERSE = false`) or `SliceReverse`
/// (`REVERSE = true`), starting at `FIRST` with stride `ABS_STEP` and length `COUNT`.
#[doc(hidden)]
pub const trait SliceDispatch<
    const FIRST: int,
    const ABS_STEP: int,
    const COUNT: int,
    const REVERSE: bool,
>: [const] Tuple
{
    type Output: [const] Tuple;
    fn slice_dispatch(&self) -> Self::Output;
}

const impl<T, const FIRST: int, const ABS_STEP: int, const COUNT: int>
    SliceDispatch<FIRST, ABS_STEP, COUNT, false> for T
where
    T: [const] Tuple
        + [const] SliceForward<
            FIRST,
            ABS_STEP,
            COUNT,
            { IS_ZERO::<COUNT> },
            { NOT_ZERO::<FIRST> },
            { IS_ONE::<COUNT> },
        >,
{
    type Output = <T as SliceForward<
        FIRST,
        ABS_STEP,
        COUNT,
        { IS_ZERO::<COUNT> },
        { NOT_ZERO::<FIRST> },
        { IS_ONE::<COUNT> },
    >>::Output;
    fn slice_dispatch(&self) -> Self::Output {
        self.slice_forward()
    }
}

const impl<T, const FIRST: int, const ABS_STEP: int, const COUNT: int>
    SliceDispatch<FIRST, ABS_STEP, COUNT, true> for T
where
    T: [const] Tuple
        + [const] SliceReverse<
            Nil,
            FIRST,
            ABS_STEP,
            COUNT,
            { IS_ZERO::<COUNT> },
            { NOT_ZERO::<FIRST> },
            { IS_ONE::<COUNT> },
        >,
{
    type Output = <T as SliceReverse<
        Nil,
        FIRST,
        ABS_STEP,
        COUNT,
        { IS_ZERO::<COUNT> },
        { NOT_ZERO::<FIRST> },
        { IS_ONE::<COUNT> },
    >>::Output;
    fn slice_dispatch(&self) -> Self::Output {
        self.slice_reverse(Nil)
    }
}

/// Python-style top-level mode slice (`start:stop:step`), wrapping non-tuple items first.
pub const trait SliceRange<const S: Slice>: [const] Item {
    /// Resulting sliced [`Tuple`].
    type Output: [const] Tuple;
    /// Extracts the modes selected by `S`.
    fn slice_range(&self) -> Self::Output;
}

const impl<T, const S: Slice> SliceRange<S> for T
where
    T: [const] Wrap,
    T::Wrapped: [const] SliceDispatch<
            { SLICE_FIRST_INDEX::<T::Wrapped, S> },
            { SLICE_ABS_STEP::<T::Wrapped, S> },
            { SLICE_COUNT::<T::Wrapped, S> },
            { SLICE_IS_REVERSE::<S> },
        >,
{
    type Output = <T::Wrapped as SliceDispatch<
        { SLICE_FIRST_INDEX::<T::Wrapped, S> },
        { SLICE_ABS_STEP::<T::Wrapped, S> },
        { SLICE_COUNT::<T::Wrapped, S> },
        { SLICE_IS_REVERSE::<S> },
    >>::Output;
    fn slice_range(&self) -> Self::Output {
        self.wrap().slice_dispatch()
    }
}

/// Slices `t` according to `S` into a [`Tuple`].
pub const fn slice_range<const S: Slice, T: [const] SliceRange<S>>(t: &T) -> T::Output {
    t.slice_range()
}

impl<H: Item, T: Tuple> Til<H, T> {
    /// Slices top-level modes according to `S` (e.g. `t.slice::<{ slice!([1..4]) }>()`).
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(min_adt_const_params, min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
    /// # #![allow(incomplete_features)]
    /// use noodle_cute::{slice, tuple};
    ///
    /// let t = tuple!(10i32, 20i32, 30i32, 40i32);
    /// assert_eq!(t.slice::<{ slice!([1..3]) }>(), tuple!(20i32, 30i32));
    /// assert_eq!(t.slice::<{ slice!([..] : -2) }>(), tuple!(40i32, 20i32));
    /// ```
    pub const fn slice<const S: Slice>(&self) -> <Self as SliceRange<S>>::Output
    where
        Self: [const] SliceRange<S>,
    {
        self.slice_range()
    }
}

/// Evaluates whether leaf `L` is a slicing wildcard (`..`).
#[doc(hidden)]
pub const IS_SLICE_WILDCARD<L: Item>: bool = L::IS_WILDCARD;

/// Leaf filter step backing [`SliceBy`]: `KEEP = true` retains `target` in a singleton tuple,
/// while `KEEP = false` drops it to [`Nil`].
#[doc(hidden)]
pub const trait SliceByLeaf<B: [const] Item, const KEEP: bool>: [const] Leaf {
    type Output: [const] Tuple;
    fn slice_by_leaf(&self, target: &B) -> Self::Output;
}

const impl<L: [const] Leaf, B: [const] Item> SliceByLeaf<B, true> for L {
    type Output = Til<B, Nil>;
    fn slice_by_leaf(&self, target: &B) -> Self::Output {
        Til(*target, Nil)
    }
}

const impl<L: [const] Leaf, B: [const] Item> SliceByLeaf<B, false> for L {
    type Output = Nil;
    fn slice_by_leaf(&self, _: &B) -> Self::Output {
        Nil
    }
}

/// Collects the subtrees of `B` whose counterpart in `Self` is a slicing wildcard (`..`).
///
/// Mirrors PyCuTe's `slice_(coord, B)`: walks `Self` as the driver (requiring
/// [`WeaklyCongruent<B>`](super::WeaklyCongruent)), retaining each subtree of `B` aligned
/// with a `..` ([`core::ops::RangeFull`]) leaf in a flat output [`Tuple`].
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::tuple;
/// use noodle_cute::tuple::{Nil, slice_by};
///
/// let target = tuple!((2i32, 3i32), (5i32, 7i32, 9i32));
/// assert_eq!(slice_by(&tuple!(.., 1i32), &target), tuple!((2i32, 3i32)));
/// assert_eq!(slice_by(&tuple!(.., ..), &tuple!(2i32, 3i32)), tuple!(2i32, 3i32));
/// assert_eq!(slice_by(&tuple!(1i32, 2i32), &tuple!(5i32, 7i32)), Nil);
/// ```
pub const trait SliceBy<B: [const] Item>: [const] Item {
    /// Flat [`Tuple`] containing the retained subtrees of `B` in walk order.
    type Output: [const] Tuple;
    /// Collects subtrees of `target` corresponding to `..` leaves in `self`.
    fn slice_by(&self, target: &B) -> Self::Output;
}

const impl SliceBy<Nil> for Nil {
    type Output = Nil;
    fn slice_by(&self, _: &Nil) -> Self::Output {
        Nil
    }
}

const impl<L, B> SliceBy<B> for L
where
    L: [const] Leaf + [const] SliceByLeaf<B, { IS_SLICE_WILDCARD::<L> }>,
    B: [const] Item,
{
    type Output = <L as SliceByLeaf<B, { IS_SLICE_WILDCARD::<L> }>>::Output;
    fn slice_by(&self, target: &B) -> Self::Output {
        self.slice_by_leaf(target)
    }
}

const impl<H, T, BH, BT> SliceBy<Til<BH, BT>> for Til<H, T>
where
    H: [const] SliceBy<BH>,
    T: [const] Tuple + [const] SliceBy<BT>,
    BH: [const] Item,
    BT: [const] Tuple,
    H::Output: [const] Append<T::Output>,
{
    type Output = <H::Output as Append<T::Output>>::Output;
    fn slice_by(&self, target: &Til<BH, BT>) -> Self::Output {
        self.0
            .slice_by(&target.0)
            .append(self.1.slice_by(&target.1))
    }
}

/// Collects the subtrees of `target` whose counterpart in `coord` is `..` (PyCuTe `slice_`).
pub const fn slice_by<C: [const] SliceBy<B>, B: [const] Item>(coord: &C, target: &B) -> C::Output {
    coord.slice_by(target)
}

/// Leaf filter step backing [`DiceBy`]: `DROP = true` drops `target` to [`Nil`],
/// while `DROP = false` retains `target` in a singleton tuple.
#[doc(hidden)]
pub const trait DiceByLeaf<B: [const] Item, const DROP: bool>: [const] Leaf {
    type Output: [const] Tuple;
    fn dice_by_leaf(&self, target: &B) -> Self::Output;
}

const impl<L: [const] Leaf, B: [const] Item> DiceByLeaf<B, true> for L {
    type Output = Nil;
    fn dice_by_leaf(&self, _: &B) -> Self::Output {
        Nil
    }
}

const impl<L: [const] Leaf, B: [const] Item> DiceByLeaf<B, false> for L {
    type Output = Til<B, Nil>;
    fn dice_by_leaf(&self, target: &B) -> Self::Output {
        Til(*target, Nil)
    }
}

/// Collects the subtrees of `B` whose counterpart in `Self` is NOT a slicing wildcard (`..`).
///
/// Mirrors PyCuTe's `dice_(coord, B)`: the complement of [`SliceBy`].
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::tuple;
/// use noodle_cute::tuple::{Nil, dice_by};
///
/// let target = tuple!((2i32, 3i32), (5i32, 7i32, 9i32));
/// assert_eq!(dice_by(&tuple!(.., 1i32), &target), tuple!((5i32, 7i32, 9i32)));
/// assert_eq!(dice_by(&tuple!(.., ..), &tuple!(2i32, 3i32)), Nil);
/// assert_eq!(dice_by(&tuple!(1i32, 2i32), &tuple!(5i32, 7i32)), tuple!(5i32, 7i32));
/// ```
pub const trait DiceBy<B: [const] Item>: [const] Item {
    /// Flat [`Tuple`] containing the non-wildcard subtrees of `B` in walk order.
    type Output: [const] Tuple;
    /// Collects subtrees of `target` corresponding to non-wildcard leaves in `self`.
    fn dice_by(&self, target: &B) -> Self::Output;
}

const impl DiceBy<Nil> for Nil {
    type Output = Nil;
    fn dice_by(&self, _: &Nil) -> Self::Output {
        Nil
    }
}

const impl<L, B> DiceBy<B> for L
where
    L: [const] Leaf + [const] DiceByLeaf<B, { IS_SLICE_WILDCARD::<L> }>,
    B: [const] Item,
{
    type Output = <L as DiceByLeaf<B, { IS_SLICE_WILDCARD::<L> }>>::Output;
    fn dice_by(&self, target: &B) -> Self::Output {
        self.dice_by_leaf(target)
    }
}

const impl<H, T, BH, BT> DiceBy<Til<BH, BT>> for Til<H, T>
where
    H: [const] DiceBy<BH>,
    T: [const] Tuple + [const] DiceBy<BT>,
    BH: [const] Item,
    BT: [const] Tuple,
    H::Output: [const] Append<T::Output>,
{
    type Output = <H::Output as Append<T::Output>>::Output;
    fn dice_by(&self, target: &Til<BH, BT>) -> Self::Output {
        self.0.dice_by(&target.0).append(self.1.dice_by(&target.1))
    }
}

/// Collects the subtrees of `target` whose counterpart in `coord` is not `..` (PyCuTe `dice_`).
pub const fn dice_by<C: [const] DiceBy<B>, B: [const] Item>(coord: &C, target: &B) -> C::Output {
    coord.dice_by(target)
}
