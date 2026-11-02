//! Nested, heterogeneous tuples of static and dynamic integers.
//!
//! [`IntTuple`] stores a head item and a tail ending in [`Nil`]. A head may
//! itself be a tuple, so top-level modes and flattened integer leaves are
//! distinct: `(2, (3, 4))` has two modes and three leaves.
//!
//! ```
//! use noodle_cute::{Int, int_tuple};
//! use noodle_cute::tuple::{Flatten, Item};
//!
//! let tuple = int_tuple![Int::<2>, (3isize, 4isize)];
//! assert_eq!(tuple.rank(), 2);
//! assert_eq!(tuple.mode_span(1)?, (1, 2));
//! assert_eq!(tuple.product()?, 24);
//! let flat = tuple.flatten();
//! assert_eq!(flat.rank(), 3);
//! assert_eq!(flat.leaf(2)?, 4);
//! # Ok::<(), noodle_cute::Error>(())
//! ```

use core::fmt::Debug;
use core::marker::PhantomData;

use crate::int::{Arith, Int, Integer, Op, Prod, mul, prod};
use crate::{Bool, Error, Result};

/// An empty tuple and the terminator of an [`IntTuple`] chain.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Nil;

/// A head item followed by a tail tuple; nested heads add tuple levels.
///
/// `repr(C)` keeps the field order identical on host and device targets, so
/// tuples and layouts can be passed by value as kernel parameters; static
/// leaves remain zero-sized.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct IntTuple<H, T>(pub H, pub T);

/// Structural marker for an empty tuple.
pub struct NilProfile;
/// Structural marker shared by all integer leaves, regardless of value or type.
pub struct LeafProfile;
/// Structural marker combining the profiles of a tuple's head and tail.
pub struct TupleProfile<H, T>(PhantomData<(H, T)>);

/// An integer leaf or a tuple that supports structural queries and leaf access.
///
/// Built-in leaves and tuples support constant evaluation with the caller's
/// `const_trait_impl` feature. Downstream crates can still provide ordinary
/// `impl Item` blocks for runtime use without enabling a feature.
///
/// ```
/// #![feature(const_trait_impl)]
/// use noodle_cute::tuple::Item;
/// use noodle_cute::{DynInt, Int, Result, int_tuple};
/// const PRODUCT: Result<isize> = int_tuple![Int::<2>, DynInt(3)].product();
/// assert_eq!(PRODUCT, Ok(6));
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not an integer leaf or a nested integer tuple",
    note = "use Int<N>, DynInt, isize, i32, Nil, or an IntTuple with an integer or tuple head and a tuple tail"
)]
pub const trait Item: Clone + Debug {
    /// The nested structure, independent of leaf values and integer types.
    type Profile;
    /// Number of top-level modes: one for an integer and zero for [`Nil`].
    const RANK: usize;
    /// Nesting depth: zero for an integer and one for [`Nil`].
    ///
    /// A nonempty tuple adds one to the greatest depth of its head items;
    /// extending the tail chain alone does not increase depth.
    const DEPTH: usize;
    /// Total number of integer leaves across all nesting levels.
    const LEAVES: usize;

    /// Returns a leaf value in head-first, depth-first order.
    ///
    /// Returns [`Error::AxisOutOfBounds`] if `index` is outside `0..Self::LEAVES`.
    fn leaf(&self, index: usize) -> Result<isize>;
    /// Returns the leaf's type-level constant, or `None` for a runtime integer.
    ///
    /// Uses the same indexing and bounds errors as [`Item::leaf`].
    fn leaf_static(&self, index: usize) -> Result<Option<isize>>;
    /// Returns `(start, len)` for a top-level mode in the flattened leaf sequence.
    ///
    /// Returns [`Error::AxisOutOfBounds`] if `axis` is outside `0..Self::RANK`.
    fn mode_span(&self, axis: usize) -> Result<(usize, usize)>;

    /// Multiplies all leaf values with checked arithmetic; the empty product is one.
    ///
    /// Any zero leaf makes the result zero. Otherwise, an overflowing
    /// multiplication returns [`Error::Overflow`].
    fn product(&self) -> Result<isize> {
        let mut i = 0;
        while i < Self::LEAVES {
            if self.leaf(i)? == 0 {
                return Ok(0);
            }
            i += 1;
        }
        let mut product = 1;
        i = 0;
        while i < Self::LEAVES {
            product = mul(product, self.leaf(i)?)?;
            i += 1;
        }
        Ok(product)
    }

    /// Returns [`Item::RANK`].
    fn rank(&self) -> usize {
        Self::RANK
    }

    /// Returns [`Item::DEPTH`].
    fn depth(&self) -> usize {
        Self::DEPTH
    }
}

const impl Item for Nil {
    type Profile = NilProfile;
    const RANK: usize = 0;
    const DEPTH: usize = 1;
    const LEAVES: usize = 0;

    fn leaf(&self, _: usize) -> Result<isize> {
        Err(Error::AxisOutOfBounds)
    }

    fn leaf_static(&self, _: usize) -> Result<Option<isize>> {
        Err(Error::AxisOutOfBounds)
    }

    fn mode_span(&self, _: usize) -> Result<(usize, usize)> {
        Err(Error::AxisOutOfBounds)
    }
}

#[diagnostic::do_not_recommend]
const impl<I: [const] Integer + Debug> Item for I {
    type Profile = LeafProfile;
    const RANK: usize = 1;
    const DEPTH: usize = 0;
    const LEAVES: usize = 1;

    fn leaf(&self, index: usize) -> Result<isize> {
        if index == 0 {
            Ok(self.value())
        } else {
            Err(Error::AxisOutOfBounds)
        }
    }

    fn leaf_static(&self, index: usize) -> Result<Option<isize>> {
        if index == 0 {
            Ok(<Self as Integer>::STATIC)
        } else {
            Err(Error::AxisOutOfBounds)
        }
    }

    fn mode_span(&self, axis: usize) -> Result<(usize, usize)> {
        if axis == 0 {
            Ok((0, 1))
        } else {
            Err(Error::AxisOutOfBounds)
        }
    }
}

/// A tuple chain, either [`Nil`] or an [`IntTuple`] with a tuple tail.
pub trait Tuple: Item {}

impl Tuple for Nil {}

const impl<H: [const] Item, T: Tuple + [const] Item> Item for IntTuple<H, T> {
    type Profile = TupleProfile<H::Profile, T::Profile>;
    const RANK: usize = 1 + T::RANK;
    // A tail's depth counts its own tuple level, which this tuple already adds.
    const DEPTH: usize = 1 + if H::DEPTH > T::DEPTH - 1 {
        H::DEPTH
    } else {
        T::DEPTH - 1
    };
    const LEAVES: usize = H::LEAVES + T::LEAVES;

    fn leaf(&self, index: usize) -> Result<isize> {
        if index < H::LEAVES {
            self.0.leaf(index)
        } else {
            self.1.leaf(index - H::LEAVES)
        }
    }

    fn leaf_static(&self, index: usize) -> Result<Option<isize>> {
        if index < H::LEAVES {
            self.0.leaf_static(index)
        } else {
            self.1.leaf_static(index - H::LEAVES)
        }
    }

    fn mode_span(&self, axis: usize) -> Result<(usize, usize)> {
        if axis == 0 {
            Ok((0, H::LEAVES))
        } else {
            let (start, len) = self.1.mode_span(axis - 1)?;
            Ok((start + H::LEAVES, len))
        }
    }
}

impl<H: Item, T: Tuple> Tuple for IntTuple<H, T> {}

// Compile-time mode access.

/// Borrows a top-level mode selected by a compile-time index.
///
/// Nested heads remain tuples; call `get` again to access their modes.
/// A bare integer supports index zero. Unsupported indices fail to compile.
pub trait Get<const INDEX: usize> {
    /// Concrete type of the selected mode.
    type Output;
    /// Returns a reference to the selected mode.
    fn get(&self) -> &Self::Output;
}

impl<I: Integer + Debug> Get<0> for I {
    type Output = Self;
    fn get(&self) -> &Self::Output {
        self
    }
}

const IS_ZERO<const N: usize>: bool = N == 0;
// Keep this expression valid at zero, even though only the head branch is used.
const SUB_1<const N: usize>: usize = N.saturating_sub(1);
// Underflows, and so fails to compile, when `B > E`.
const SPAN<const B: usize, const E: usize>: usize = E - B;
const BEFORE<const B: usize, const E: usize>: bool = B < E;
const INDEX<const N: isize>: usize = {
    assert!(N >= 0, "mode indices must be nonnegative");
    N as usize
};

// Implementation traits that appear in public bounds but are not part of the API.
mod detail {
    use super::{Get, IntTuple, Nil, SUB_1};
    use crate::Bool;
    use crate::int::Integer;

    /// One head-or-tail step of compile-time tuple access.
    pub trait Step<H, T, const INDEX: usize> {
        type Output;
        fn step(tuple: &IntTuple<H, T>) -> &Self::Output;
    }

    impl<H, T, const INDEX: usize> Step<H, T, INDEX> for Bool<true> {
        type Output = H;
        fn step(tuple: &IntTuple<H, T>) -> &Self::Output {
            &tuple.0
        }
    }

    impl<H, T: Get<{ SUB_1::<INDEX> }>, const INDEX: usize> Step<H, T, INDEX> for Bool<false> {
        type Output = T::Output;
        fn step(tuple: &IntTuple<H, T>) -> &Self::Output {
            T::get(&tuple.1)
        }
    }

    /// Borrows the chain after its first `N` modes.
    pub trait Skip<const N: usize> {
        type Output;
        fn skip(&self) -> &Self::Output;
    }

    pub trait SkipStep<H, T, const N: usize> {
        type Output;
        fn skip(tuple: &IntTuple<H, T>) -> &Self::Output;
    }

    impl<H, T, const N: usize> SkipStep<H, T, N> for Bool<true> {
        type Output = IntTuple<H, T>;
        fn skip(tuple: &IntTuple<H, T>) -> &Self::Output {
            tuple
        }
    }

    impl<H, T: Skip<{ SUB_1::<N> }>, const N: usize> SkipStep<H, T, N> for Bool<false> {
        type Output = T::Output;
        fn skip(tuple: &IntTuple<H, T>) -> &Self::Output {
            tuple.1.skip()
        }
    }

    /// Copies the first `N` modes into a new chain.
    pub trait First<const N: usize> {
        type Output;
        fn first(&self) -> Self::Output;
    }

    pub trait FirstStep<H, T, const N: usize> {
        type Output;
        fn first(tuple: &IntTuple<H, T>) -> Self::Output;
    }

    impl<H, T, const N: usize> FirstStep<H, T, N> for Bool<true> {
        type Output = Nil;
        fn first(_: &IntTuple<H, T>) -> Nil {
            Nil
        }
    }

    impl<H: Clone, T: First<{ SUB_1::<N> }>, const N: usize> FirstStep<H, T, N> for Bool<false> {
        type Output = IntTuple<H, T::Output>;
        fn first(tuple: &IntTuple<H, T>) -> Self::Output {
            IntTuple(tuple.0.clone(), tuple.1.first())
        }
    }

    /// The single mode inserted by `group`, absent when the range is empty.
    pub trait Single<X> {
        type Output;
        fn single(mode: X) -> Self::Output;
    }

    impl<X> Single<X> for Bool<true> {
        type Output = IntTuple<X, Nil>;
        fn single(mode: X) -> Self::Output {
            IntTuple(mode, Nil)
        }
    }

    impl<X> Single<X> for Bool<false> {
        type Output = Nil;
        fn single(_: X) -> Nil {
            Nil
        }
    }

    /// CuTe `wrap`: an integer becomes a rank-1 tuple and a tuple is unchanged.
    pub trait Wrap {
        type Output;
        fn wrap(self) -> Self::Output;
    }

    impl<I: Integer> Wrap for I {
        type Output = IntTuple<I, Nil>;
        fn wrap(self) -> Self::Output {
            IntTuple(self, Nil)
        }
    }

    impl Wrap for Nil {
        type Output = Nil;
        fn wrap(self) -> Nil {
            self
        }
    }

    impl<H, T> Wrap for IntTuple<H, T> {
        type Output = Self;
        fn wrap(self) -> Self {
            self
        }
    }

    /// Takes the only mode out of a one-mode tuple and keeps a longer tuple unchanged.
    ///
    /// Unlike CuTe's recursive `unwrap`, this removes a single level; callers only pass flat
    /// tuples, where both agree.
    pub trait Unwrap {
        type Output;
        fn unwrap(self) -> Self::Output;
    }
    impl<H> Unwrap for IntTuple<H, Nil> {
        type Output = H;
        fn unwrap(self) -> H {
            self.0
        }
    }
    impl<H, TH, TT> Unwrap for IntTuple<H, IntTuple<TH, TT>> {
        type Output = Self;
        fn unwrap(self) -> Self {
            self
        }
    }
}
use detail::{First, FirstStep, Single, Skip, SkipStep, Step};
pub(crate) use detail::{Unwrap, Wrap};

impl<H, T, const INDEX: usize> Get<INDEX> for IntTuple<H, T>
where
    Bool<{ IS_ZERO::<INDEX> }>: Step<H, T, INDEX>,
{
    type Output = <Bool<{ IS_ZERO::<INDEX> }> as Step<H, T, INDEX>>::Output;
    fn get(&self) -> &Self::Output {
        Bool::<{ IS_ZERO::<INDEX> }>::step(self)
    }
}

impl Skip<0> for Nil {
    type Output = Nil;
    fn skip(&self) -> &Nil {
        self
    }
}

impl<H, T, const N: usize> Skip<N> for IntTuple<H, T>
where
    Bool<{ IS_ZERO::<N> }>: SkipStep<H, T, N>,
{
    type Output = <Bool<{ IS_ZERO::<N> }> as SkipStep<H, T, N>>::Output;
    fn skip(&self) -> &Self::Output {
        <Bool<{ IS_ZERO::<N> }> as SkipStep<H, T, N>>::skip(self)
    }
}

impl First<0> for Nil {
    type Output = Nil;
    fn first(&self) -> Nil {
        Nil
    }
}

impl<H, T, const N: usize> First<N> for IntTuple<H, T>
where
    Bool<{ IS_ZERO::<N> }>: FirstStep<H, T, N>,
{
    type Output = <Bool<{ IS_ZERO::<N> }> as FirstStep<H, T, N>>::Output;
    fn first(&self) -> Self::Output {
        <Bool<{ IS_ZERO::<N> }> as FirstStep<H, T, N>>::first(self)
    }
}

impl<H, T> IntTuple<H, T> {
    /// Borrows a top-level mode without flattening nested heads.
    ///
    /// The index must be supported by this tuple's [`Get`] implementation.
    pub fn get<const INDEX: usize>(&self) -> &<Self as Get<INDEX>>::Output
    where
        Self: Get<INDEX>,
    {
        <Self as Get<INDEX>>::get(self)
    }
}

// Structural operations.

/// Replaces a tuple chain's terminating [`Nil`] with `rhs`.
///
/// Appending another tuple joins the chains while preserving concrete types
/// and nested heads. Appending to [`Nil`] returns `rhs` unchanged.
pub trait Append<R> {
    /// The chain obtained by replacing the terminal `Nil` with `R`.
    type Output;
    /// Consumes this chain and attaches `rhs` after its last head.
    fn append(self, rhs: R) -> Self::Output;
}

impl<R> Append<R> for Nil {
    type Output = R;
    fn append(self, rhs: R) -> Self::Output {
        rhs
    }
}

impl<H, T: Append<R>, R> Append<R> for IntTuple<H, T> {
    type Output = IntTuple<H, T::Output>;
    fn append(self, rhs: R) -> Self::Output {
        IntTuple(self.0, self.1.append(rhs))
    }
}

/// Flattens nested heads into one tuple chain in head-first, depth-first order.
///
/// Leaf types are preserved. A bare integer becomes a singleton tuple, and
/// [`Nil`] remains empty. This is CuTe's `flatten_to_tuple`; CuTe's `flatten`
/// instead returns a bare integer unchanged.
pub trait Flatten {
    /// A flat tuple chain retaining the original leaf types and order.
    type Output;
    /// Consumes the structure and concatenates its flattened heads.
    fn flatten(self) -> Self::Output;
}

impl<I: Integer + Debug> Flatten for I {
    type Output = IntTuple<Self, Nil>;
    fn flatten(self) -> Self::Output {
        IntTuple(self, Nil)
    }
}

impl Flatten for Nil {
    type Output = Self;
    fn flatten(self) -> Self::Output {
        self
    }
}

impl<H: Flatten, T: Flatten> Flatten for IntTuple<H, T>
where
    H::Output: Append<T::Output>,
{
    type Output = <H::Output as Append<T::Output>>::Output;
    fn flatten(self) -> Self::Output {
        self.0.flatten().append(self.1.flatten())
    }
}

/// Copies the top-level modes in `[B, E)` into a new tuple: CuTe `take<B, E>`.
///
/// `B > E` and ranges beyond the tuple fail to compile; `B == E` gives [`Nil`].
/// Modes may be any cloneable values, including layouts.
///
/// ```
/// use noodle_cute::tuple::Take;
/// use noodle_cute::{Int, int_tuple};
///
/// let t = int_tuple![Int::<2>, (Int::<3>, Int::<4>), 5i32];
/// assert_eq!(Take::<1, 3>::take(&t), int_tuple![(Int::<3>, Int::<4>), 5i32]);
/// ```
pub trait Take<const B: usize, const E: usize> {
    /// The selected modes.
    type Output;
    /// Returns the selected modes.
    fn take(&self) -> Self::Output;
}

impl<T, const B: usize, const E: usize> Take<B, E> for T
where
    T: Skip<B>,
    <T as Skip<B>>::Output: First<{ SPAN::<B, E> }>,
{
    type Output = <<T as Skip<B>>::Output as First<{ SPAN::<B, E> }>>::Output;
    fn take(&self) -> Self::Output {
        self.skip().first()
    }
}

/// Copies top-level modes by static index: CuTe `select<I...>`.
///
/// The indices are a tuple of `Int<N>` values, such as
/// `int_tuple![Int::<2>, Int::<0>]`; they may repeat or reorder modes. A bare
/// integer supports index zero, like [`Get`]. Out-of-range or negative indices
/// fail to compile.
///
/// ```
/// use noodle_cute::tuple::Select;
/// use noodle_cute::{Int, int_tuple};
///
/// let t = int_tuple![Int::<2>, 3i32, Int::<4>];
/// let picked = t.select(int_tuple![Int::<2>, Int::<0>]);
/// assert_eq!(picked, int_tuple![Int::<4>, Int::<2>]);
/// ```
pub trait Select<I> {
    /// The selected modes, in index order.
    type Output;
    /// Returns the modes named by `indices`.
    fn select(&self, indices: I) -> Self::Output;
}

impl<T> Select<Nil> for T {
    type Output = Nil;
    fn select(&self, _: Nil) -> Nil {
        Nil
    }
}

impl<T, R, const N: isize> Select<IntTuple<Int<N>, R>> for T
where
    T: Get<{ INDEX::<N> }> + Select<R>,
    <T as Get<{ INDEX::<N> }>>::Output: Clone,
{
    type Output = IntTuple<<T as Get<{ INDEX::<N> }>>::Output, <T as Select<R>>::Output>;
    fn select(&self, indices: IntTuple<Int<N>, R>) -> Self::Output {
        IntTuple(
            <T as Get<{ INDEX::<N> }>>::get(self).clone(),
            <T as Select<R>>::select(self, indices.1),
        )
    }
}

/// Groups the top-level modes in `[B, E)` into one mode: CuTe `group<B, E>`.
///
/// `B == E` leaves the tuple unchanged; `B > E` and ranges beyond the tuple
/// fail to compile.
///
/// ```
/// use noodle_cute::tuple::Group;
/// use noodle_cute::{Int, int_tuple};
///
/// let t = int_tuple![Int::<1>, Int::<2>, Int::<3>, Int::<4>];
/// assert_eq!(Group::<1, 3>::group(&t), int_tuple![Int::<1>, (Int::<2>, Int::<3>), Int::<4>]);
/// ```
pub trait Group<const B: usize, const E: usize> {
    /// The regrouped tuple.
    type Output;
    /// Returns the regrouped tuple.
    fn group(&self) -> Self::Output;
}

type Middle<T, const B: usize, const E: usize> =
    <Bool<{ BEFORE::<B, E> }> as Single<<T as Take<B, E>>::Output>>::Output;
type Rest<T, const B: usize, const E: usize> =
    <Middle<T, B, E> as Append<<T as Skip<E>>::Output>>::Output;

impl<T, const B: usize, const E: usize> Group<B, E> for T
where
    T: First<B> + Skip<E> + Take<B, E>,
    <T as Skip<E>>::Output: Clone,
    Bool<{ BEFORE::<B, E> }>: Single<<T as Take<B, E>>::Output>,
    Middle<T, B, E>: Append<<T as Skip<E>>::Output>,
    <T as First<B>>::Output: Append<Rest<T, B, E>>,
{
    type Output = <<T as First<B>>::Output as Append<Rest<T, B, E>>>::Output;
    fn group(&self) -> Self::Output {
        let middle = <Bool<{ BEFORE::<B, E> }> as Single<_>>::single(Take::<B, E>::take(self));
        First::<B>::first(self).append(middle.append(Skip::<E>::skip(self).clone()))
    }
}

/// Multiplies leaves while retaining a static result whenever CuTe's rules allow it.
///
/// A tuple of [`Int`] leaves produces an `Int` whose type encodes the product;
/// [`Nil`] produces `Int<1>`, and a static zero leaf makes the product `Int<0>`.
/// Otherwise a runtime leaf makes the result runtime, combined as described in
/// [`Integer`]. Unlike [`Item::product`], there is no zero pre-scan or checked
/// error; runtime overflow follows Rust's overflow settings.
pub trait Product {
    /// The static or runtime integer representation of the result.
    type Output;
    /// Consumes the structure and multiplies its leaves.
    fn product_typed(self) -> Self::Output;
}

impl<I: Integer + Debug> Product for I {
    type Output = Self;
    fn product_typed(self) -> Self::Output {
        self
    }
}

impl Product for Nil {
    type Output = Int<1>;
    fn product_typed(self) -> Self::Output {
        Int
    }
}

impl<H: Product, T: Product> Product for IntTuple<H, T>
where
    H::Output: Arith<T::Output, { Op::Mul }>,
{
    type Output = Prod<H::Output, T::Output>;
    fn product_typed(self) -> Self::Output {
        prod(self.0.product_typed(), self.1.product_typed())
    }
}

/// The product of each top-level mode: CuTe `product_each`.
///
/// A bare integer is returned unchanged; a tuple yields one [`Product`] per mode.
///
/// ```
/// use noodle_cute::tuple::ProductEach;
/// use noodle_cute::{Int, int_tuple};
///
/// let sizes = int_tuple![(Int::<2>, Int::<4>), 3i32].product_each();
/// assert_eq!(sizes, int_tuple![Int::<8>, 3i32]);
/// ```
pub trait ProductEach {
    /// The per-mode products.
    type Output;
    /// Returns the per-mode products.
    fn product_each(self) -> Self::Output;
}

impl<I: Integer> ProductEach for I {
    type Output = Self;
    fn product_each(self) -> Self {
        self
    }
}

impl ProductEach for Nil {
    type Output = Nil;
    fn product_each(self) -> Nil {
        Nil
    }
}

impl<H: Product, T: ProductEach> ProductEach for IntTuple<H, T> {
    type Output = IntTuple<H::Output, T::Output>;
    fn product_each(self) -> Self::Output {
        IntTuple(self.0.product_typed(), self.1.product_each())
    }
}

/// Builds an [`IntTuple`] chain; parenthesized groups create nested tuple heads.
///
/// For example, `int_tuple![Int::<2>, (Int::<3>, Int::<4>)]` has two top-level
/// modes and three leaves. An empty invocation produces [`Nil`].
#[macro_export]
macro_rules! int_tuple {
    () => { $crate::Nil };
    // Match groups before expressions so parentheses create a new tuple level.
    (($($inner:tt)*), $($tail:tt)*) => { $crate::IntTuple($crate::int_tuple!($($inner)*), $crate::int_tuple!($($tail)*)) };
    (($($inner:tt)*)) => { $crate::IntTuple($crate::int_tuple!($($inner)*), $crate::Nil) };
    ($head:expr, $($tail:tt)*) => { $crate::IntTuple($head, $crate::int_tuple!($($tail)*)) };
    ($head:expr) => { $crate::IntTuple($head, $crate::Nil) };
}
