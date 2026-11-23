//! Hierarchical tuples, structural relations, and leaf transformations.
//!
//! A [`TupleContainer`] stores a head [`Item`] and a tail [`Tuple`]. Tuple chains
//! terminate in [`Nil`], and nested heads introduce additional hierarchy levels
//! ([`Nil`] can also appear as an empty nested head).
//!
//! # Main Interfaces & Submodules
//!
//! - **Core types & markers**: [`Nil`], [`TupleContainer`], [`Item`], [`Leaf`],
//!   [`Tuple`], [`FlatTuple`], [`IntTuple`], [`IS_STATIC`], and structural profiles
//!   ([`NilProfile`], [`LeafProfile`], [`TupleProfile`]).
//! - **Congruence relations**: [`WeaklyCongruent`] and [`Congruent`].
//! - [`mod@index`]: Type-level mode paths ([`Mode`], [`ModeEnd`]), signed indexing
//!   ([`Get`], [`GetOutput`]), multi-mode selection ([`Select`], [`select`]), and
//!   top-level slicing ([`SliceRange`], [`slice_range`]).
//! - [`mod@restructure`]: [`Wrap`], [`Unwrap`], [`Append`], [`Prepend`], [`Flatten`],
//!   [`Unflatten`], [`RepeatLike`], [`ReplaceAt`], [`Replace`], and [`Lift`].
//! - [`mod@traversal`]: Single-tree [`TransformLeaf`] and synchronized multi-tree
//!   [`TransformApplyLeaf`], [`ZipTransformLeaf`], [`ZipLeaves`], [`FoldLeaf`],
//!   [`AllLeaves`], and [`AnyLeaf`] (up to 6 trees in lockstep).
//! - [`mod@product`]: [`TupleProduct`], [`ProductEach`], and [`ProductLike`].
//! - **Macros**: [`tuple!`](crate::tuple!), [`int_tuple!`](crate::int_tuple!),
//!   [`mode!`](crate::mode!), and [`walk_mode!`](crate::walk_mode!).
//!
//! # Examples
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{tuple, walk_mode};
//! use noodle_cute::tuple::{Flatten, Item, Nil, Til, TransformLeaf, TransformLeafFunctor};
//!
//! // The hierarchy (10, (20, 30), 40).
//! let t = tuple!(10i32, (20i32, 30i32), 40i32);
//! assert_eq!((t.rank(), t.depth(), t.length()), (3, 2, 4));
//! assert_eq!(t.get::<-1>(), 40);
//! assert_eq!(walk_mode!(t, 1, 0), 20);
//! assert_eq!(t.flatten(), Til(10, Til(20, Til(30, Til(40, Nil)))));
//!
//! #[derive(Clone, Copy)]
//! struct Increment;
//! impl TransformLeafFunctor<i32> for Increment {
//!     type Output = i32;
//!     fn apply(&self, leaf: i32) -> i32 {
//!         leaf + 1
//!     }
//! }
//! let incremented = tuple!(11i32, (21i32, 31i32), 41i32);
//! assert_eq!(t.transform(&Increment), incremented);
//! ```
//!
//! # Notes
//!
//! - Native Rust tuples `(a, b)` do not implement [`Item`]; construct hierarchical
//!   tuples with [`tuple!`](crate::tuple!), [`int_tuple!`](crate::int_tuple!), or
//!   [`TupleContainer::new`].
//! - All [`Item`] types must be `Copy`. Scalar leaves have `RANK = 1, DEPTH = 0, LEAVES = 1`,
//!   whereas [`Nil`] has `RANK = 0, DEPTH = 1, LEAVES = 0`.
//! - Custom leaf types implement [`Item`] with `type Profile = LeafProfile` and
//!   automatically receive the blanket [`Leaf`] implementation.

use core::marker::PhantomData;

use crate::integer::{Integer, add, int, max, sub};

pub mod index;
pub mod product;
pub mod restructure;
pub mod traversal;

pub use index::*;
pub use product::*;
pub use restructure::*;
pub use traversal::*;

/// The empty tuple and terminator of a built-in tuple chain.
///
/// `Nil` has rank 0, depth 1, and zero leaves. As a [`TupleContainer`] head it
/// represents an empty nested tuple.
///
/// # Examples
///
/// ```
/// use noodle_cute::tuple::{Item, Nil, Til};
///
/// assert_eq!((Nil::RANK, Nil::DEPTH, Nil::LEAVES), (0, 1, 0));
/// let nested_empty = Til(Nil, Nil);
/// assert_eq!(
///     (
///         nested_empty.rank(),
///         nested_empty.depth(),
///         nested_empty.length()
///     ),
///     (1, 2, 0)
/// );
/// ```
#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct Nil;

const impl PartialEq for Nil {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

const impl Eq for Nil {}

// Cons-cell backing `TupleContainer`, storing head item `H` and tail tuple `T`.
#[doc(hidden)]
#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct Til<H, T>(pub H, pub T);

const impl<HL, TL, HR, TR> PartialEq<Til<HR, TR>> for Til<HL, TL>
where
    HL: [const] PartialEq<HR>,
    TL: [const] PartialEq<TR>,
{
    fn eq(&self, rhs: &Til<HR, TR>) -> bool {
        self.0 == rhs.0 && self.1 == rhs.1
    }
}

const impl<H: [const] Eq, T: [const] Eq> Eq for Til<H, T> {}

/// Tuple container representing a hierarchical or flat tuple chain.
pub type TupleContainer<H, T> = Til<H, T>;

// Formats remaining modes of a non-empty tuple chain after the leading head.
trait TupleTailDisplay {
    fn fmt_tail(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result;
}

impl TupleTailDisplay for Nil {
    fn fmt_tail(&self, _: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Ok(())
    }
}

impl<H: core::fmt::Display, T: TupleTailDisplay> TupleTailDisplay for Til<H, T> {
    fn fmt_tail(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(", ")?;
        core::fmt::Display::fmt(&self.0, f)?;
        self.1.fmt_tail(f)
    }
}

impl core::fmt::Display for Nil {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("()")
    }
}

impl<H: core::fmt::Display, T: Tuple + TupleTailDisplay> core::fmt::Display for Til<H, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("(")?;
        core::fmt::Display::fmt(&self.0, f)?;
        if T::RANK == 0 {
            f.write_str(",)")
        } else {
            self.1.fmt_tail(f)?;
            f.write_str(")")
        }
    }
}

/// Structural profile of the empty tuple [`Nil`].
///
/// An empty nested head contributes `NilProfile` to its enclosing [`TupleProfile`].
pub struct NilProfile;

/// Structural profile shared by scalar leaves.
///
/// Integer leaves use this profile, and custom leaf items can select it through
/// [`Item::Profile`].
pub struct LeafProfile;

/// Structural profile of a nonempty [`TupleContainer`].
///
/// `H` and `T` are the head and tail profiles. Each cons-cell contributes one
/// `TupleProfile`, including the cons-cells that form the tail chain.
///
/// ```
/// use noodle_cute::integer::Int;
/// use noodle_cute::tuple::{Item, LeafProfile, Nil, NilProfile, Til, TupleProfile};
///
/// fn require_profile<T: Item<Profile = P>, P>(_: T) {}
/// type PairProfile = TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>;
/// require_profile::<_, PairProfile>(Til(1i32, Til(Int::<2>, Nil)));
/// require_profile::<_, TupleProfile<NilProfile, NilProfile>>(Til(Nil, Nil));
/// ```
pub struct TupleProfile<H, T>(PhantomData<(H, T)>);

/// A scalar leaf or a nested tuple supporting structural queries.
///
/// Built-in implementations cover [`int`] (`i32`),
/// [`Int<N>`](crate::integer::Int), [`Nil`], and [`TupleContainer`] with an
/// `Item` head and a [`Tuple`] tail. Custom leaf items supply their own
/// structural constants and [`LeafProfile`].
///
/// # Examples
///
/// ```
/// use noodle_cute::integer::Int;
/// use noodle_cute::tuple::{Item, Nil, TupleContainer};
///
/// assert_eq!((i32::RANK, i32::DEPTH, i32::LEAVES), (1, 0, 1));
/// assert_eq!(
///     (Int::<5>::RANK, Int::<5>::DEPTH, Int::<5>::LEAVES),
///     (1, 0, 1)
/// );
/// assert_eq!((Nil::RANK, Nil::DEPTH, Nil::LEAVES), (0, 1, 0));
///
/// type Nested = TupleContainer<
///     Int<1>,
///     TupleContainer<TupleContainer<Int<2>, TupleContainer<i32, Nil>>, Nil>,
/// >;
/// assert_eq!((Nested::RANK, Nested::DEPTH, Nested::LEAVES), (2, 2, 3));
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a hierarchical tuple item",
    note = "use an integer, a custom Item, Nil, or a TupleContainer with an Item head and a tuple tail"
)]
pub const trait Item: Copy {
    /// Number of top-level modes, with one for a leaf and zero for [`Nil`].
    const RANK: int;
    /// Nesting depth, with zero for a leaf and one for [`Nil`].
    ///
    /// A nonempty tuple has depth one greater than its deepest head item.
    const DEPTH: int;
    /// Total number of leaves across all nesting levels.
    ///
    /// Each scalar leaf contributes one, and [`Nil`] contributes zero.
    const LEAVES: int;
    /// Whole-tree staticness flag (`true` iff all scalar leaves are static).
    ///
    /// Defaults to `false` for custom user items.
    const IS_STATIC: bool = false;

    /// The nested structure, including empty heads and the terminal tail.
    ///
    /// Built-in profiles depend on leaf positions and nesting, independently of
    /// leaf values and integer types.
    type Profile;

    /// Returns [`Item::RANK`].
    fn rank(&self) -> int {
        Self::RANK
    }
    /// Returns [`Item::DEPTH`].
    fn depth(&self) -> int {
        Self::DEPTH
    }
    /// Returns [`Item::LEAVES`].
    fn length(&self) -> int {
        Self::LEAVES
    }
    /// Returns [`Item::IS_STATIC`].
    fn is_static(&self) -> bool {
        Self::IS_STATIC
    }
}

const impl Item for Nil {
    const RANK: int = 0;
    const DEPTH: int = 1;
    const LEAVES: int = 0;
    const IS_STATIC: bool = true;
    type Profile = NilProfile;
}

const impl<I: [const] Integer> Item for I {
    const RANK: int = 1;
    const DEPTH: int = 0;
    const LEAVES: int = 1;
    const IS_STATIC: bool = I::IS_STATIC;
    type Profile = LeafProfile;
}

const impl<H: [const] Item, T: [const] Tuple> Item for Til<H, T> {
    const RANK: int = add(1, T::RANK);
    const DEPTH: int = add(1, max(H::DEPTH, sub(T::DEPTH, 1)));
    const LEAVES: int = add(H::LEAVES, T::LEAVES);
    const IS_STATIC: bool = H::IS_STATIC && T::IS_STATIC;
    type Profile = TupleProfile<H::Profile, T::Profile>;
}

/// Marker trait for scalar [`Item`] leaves (`Profile = LeafProfile`).
pub const trait Leaf: [const] Item<Profile = LeafProfile> {}
const impl<L: [const] Item<Profile = LeafProfile>> Leaf for L {}

/// Marker trait for built-in tuple chains (`Nil` or `TupleContainer<H, T>`).
pub const trait Tuple: [const] Item {}
const impl Tuple for Nil {}
const impl<H: [const] Item, T: [const] Tuple> Tuple for Til<H, T> {}

/// Marker trait for flat tuples whose elements are all [`Leaf`] items.
pub const trait FlatTuple: [const] Tuple {}
const impl FlatTuple for Nil {}
const impl<H: [const] Leaf, T: [const] FlatTuple> FlatTuple for Til<H, T> {}

/// Hierarchical [`Tuple`] containing integers (`int`, `Int<N>`) and [`Nil`].
pub const trait IntTuple: [const] Tuple {}
const impl IntTuple for Nil {}
const impl<T: [const] IntTuple> IntTuple for Til<Nil, T> {}
const impl<I: [const] Integer, T: [const] IntTuple> IntTuple for Til<I, T> {}
const impl<HH: [const] Item, HT: [const] Tuple, T: [const] IntTuple> IntTuple
    for Til<Til<HH, HT>, T>
where
    Til<HH, HT>: [const] IntTuple,
{
}

impl<H: Item, T: Tuple> Til<H, T> {
    /// Constructs a tuple cons-cell from a head and a tail.
    pub const fn new(head: H, tail: T) -> Self {
        Til(head, tail)
    }

    /// Copies the item at mode `I`, counting negative indices from the end.
    ///
    /// Uses the bounds and indexing rules of [`Get`].
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
    /// # #![allow(incomplete_features)]
    /// use noodle_cute::tuple::{Nil, Til};
    ///
    /// let inner = Til(20i32, Til(30i32, Nil));
    /// let t = Til(10i32, Til(inner, Til(40i32, Nil)));
    /// assert_eq!(t.get::<0>(), 10);
    /// assert_eq!(t.get::<1>(), inner);
    /// assert_eq!(t.get::<-1>(), 40);
    /// assert_eq!(t.get::<-2>().get::<-1>(), 30);
    /// ```
    pub const fn get<const I: int>(&self) -> GetOutput<Self, I>
    where
        Self: [const] Get<I>,
    {
        Get::<I>::get(self)
    }

    /// Returns the selected item's [`RANK`](Item::RANK) using [`Get`] indexing.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
    /// # #![allow(incomplete_features)]
    /// use noodle_cute::tuple::{Nil, Til};
    ///
    /// let inner = Til(20i32, Til(Til(30i32, Til(40i32, Nil)), Nil));
    /// let t = Til(10i32, Til(inner, Nil));
    /// assert_eq!(t.rank_at::<0>(), 1);
    /// assert_eq!(t.rank_at::<1>(), 2);
    /// assert_eq!(t.rank_at::<-1>(), 2);
    /// ```
    pub const fn rank_at<const I: int>(&self) -> int
    where
        Self: Get<I>,
    {
        GetOutput::<Self, I>::RANK
    }

    /// Returns the selected item's [`DEPTH`](Item::DEPTH) using [`Get`] indexing.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
    /// # #![allow(incomplete_features)]
    /// use noodle_cute::tuple::{Nil, Til};
    ///
    /// let inner = Til(20i32, Til(Til(30i32, Til(40i32, Nil)), Nil));
    /// let t = Til(10i32, Til(inner, Nil));
    /// assert_eq!(t.depth_at::<0>(), 0);
    /// assert_eq!(t.depth_at::<1>(), 2);
    /// assert_eq!(t.depth_at::<-1>(), 2);
    /// ```
    pub const fn depth_at<const I: int>(&self) -> int
    where
        Self: Get<I>,
    {
        GetOutput::<Self, I>::DEPTH
    }

    /// Returns the selected item's [`LEAVES`](Item::LEAVES) using [`Get`] indexing.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
    /// # #![allow(incomplete_features)]
    /// use noodle_cute::tuple::{Nil, Til};
    ///
    /// let inner = Til(20i32, Til(Til(30i32, Til(40i32, Nil)), Nil));
    /// let t = Til(10i32, Til(inner, Nil));
    /// assert_eq!(t.length_at::<0>(), 1);
    /// assert_eq!(t.length_at::<1>(), 3);
    /// assert_eq!(t.length_at::<-1>(), 3);
    /// ```
    pub const fn length_at<const I: int>(&self) -> int
    where
        Self: Get<I>,
    {
        GetOutput::<Self, I>::LEAVES
    }

    /// Returns the selected item's [`IS_STATIC`](Item::IS_STATIC) using [`Get`] indexing.
    pub const fn is_static_at<const I: int>(&self) -> bool
    where
        Self: Get<I>,
    {
        GetOutput::<Self, I>::IS_STATIC
    }
}

/// Compile-time staticness query for any [`Item`].
pub const IS_STATIC<T: Item>: bool = T::IS_STATIC;

#[doc(hidden)]
pub const fn require_tuple<T: Tuple>(value: T) -> T {
    value
}

#[doc(hidden)]
pub const fn require_int_tuple<T: IntTuple>(value: T) -> T {
    value
}

// Profile-level coarsening relation backing `WeaklyCongruent` and `Congruent`:
// `LeafProfile` coarsens any profile `R`, `NilProfile` coarsens `NilProfile`,
// and `TupleProfile` compares head and tail profiles recursively.
#[doc(hidden)]
pub const trait WeaklyCongruentImpl<R> {}
const impl WeaklyCongruentImpl<NilProfile> for NilProfile {}
const impl<R> WeaklyCongruentImpl<R> for LeafProfile {}
const impl<HL, TL, HR, TR> WeaklyCongruentImpl<TupleProfile<HR, TR>> for TupleProfile<HL, TL>
where
    HL: WeaklyCongruentImpl<HR>,
    TL: WeaklyCongruentImpl<TR>,
{
}

/// Marker relation indicating that `Self` coarsens `R`.
///
/// The blanket implementation compares [`Item::Profile`], ignoring values and
/// leaf types. A [`LeafProfile`] coarsens any subtree, including [`NilProfile`].
/// `NilProfile` coarsens itself. Two tuple profiles must have matching tail-chain
/// lengths and recursively coarsening heads.
///
/// Require this trait in a generic bound to enforce the directional relation.
///
/// # Examples
///
/// ```
/// use noodle_cute::integer::Int;
/// use noodle_cute::tuple::{Item, Nil, Til, WeaklyCongruent};
///
/// fn require<L: WeaklyCongruent<R>, R: Item>(_: &L, _: &R) {}
/// require(
///     &Int::<32>,
///     &Til(3i32, Til(Til(4i32, Til(5i32, Nil)), Nil)),
/// );
/// require(
///     &Til(3i32, Nil),
///     &Til(Til(4i32, Til(5i32, Nil)), Nil),
/// );
/// require(&32i32, &Nil);
/// require(&Nil, &Nil);
/// ```
///
/// A tuple-to-scalar comparison fails the relation.
///
/// ```compile_fail,E0277
/// use noodle_cute::tuple::{Item, Nil, Til, WeaklyCongruent};
/// fn require<L: WeaklyCongruent<R>, R: Item>(_: &L, _: &R) {}
/// require(&Til(3i32, Nil), &32i32);
/// ```
///
/// Different top-level ranks fail the relation.
///
/// ```compile_fail,E0277
/// use noodle_cute::tuple::{Item, Nil, Til, WeaklyCongruent};
/// fn require<L: WeaklyCongruent<R>, R: Item>(_: &L, _: &R) {}
/// require(&Til(1i32, Til(2i32, Nil)), &Til(1i32, Nil));
/// ```
///
/// Comparisons from `Nil` to a nonempty tuple or scalar leaf fail the relation.
///
/// ```compile_fail,E0277
/// use noodle_cute::tuple::{Item, Nil, Til, WeaklyCongruent};
/// fn require<L: WeaklyCongruent<R>, R: Item>(_: &L, _: &R) {}
/// require(&Nil, &Til(1i32, Nil));
/// ```
///
/// ```compile_fail,E0277
/// use noodle_cute::tuple::{Item, Nil, WeaklyCongruent};
/// fn require<L: WeaklyCongruent<R>, R: Item>(_: &L, _: &R) {}
/// require(&Nil, &1i32);
/// ```
///
/// A comparison from a nonempty tuple to `Nil` fails the relation.
///
/// ```compile_fail,E0277
/// use noodle_cute::tuple::{Item, Nil, Til, WeaklyCongruent};
/// fn require<L: WeaklyCongruent<R>, R: Item>(_: &L, _: &R) {}
/// require(&Til(1i32, Nil), &Nil);
/// ```
///
/// Corresponding nested heads must coarsen in the same direction.
///
/// ```compile_fail,E0277
/// use noodle_cute::tuple::{Item, Nil, Til, WeaklyCongruent};
/// fn require<L: WeaklyCongruent<R>, R: Item>(_: &L, _: &R) {}
/// require(&Til(Til(1i32, Nil), Nil), &Til(2i32, Nil));
/// ```
pub const trait WeaklyCongruent<R: Item>: Item {}
const impl<L: Item, R: Item> WeaklyCongruent<R> for L where
    L::Profile: WeaklyCongruentImpl<R::Profile>
{
}

/// Marker relation indicating identical tuple and leaf structure.
///
/// The blanket implementation requires profile coarsening in both directions.
/// Leaf values and integer types do not affect the relation. Empty tuple
/// positions are part of the structure.
///
/// # Examples
///
/// ```
/// use noodle_cute::integer::Int;
/// use noodle_cute::tuple::{Congruent, Item, Nil, Til};
///
/// fn require<L: Congruent<R>, R: Item>(_: &L, _: &R) {}
/// require(
///     &Til(Int::<4>, Til(8i32, Nil)),
///     &Til(5i32, Til(Int::<7>, Nil)),
/// );
/// require(&Int::<31>, &42i32);
/// require(&Til(Nil, Nil), &Til(Nil, Nil));
/// require(&Nil, &Nil);
/// ```
///
/// One-way coarsening is insufficient for congruence.
///
/// ```compile_fail,E0277
/// use noodle_cute::tuple::{Congruent, Item, Nil, Til};
/// fn require<L: Congruent<R>, R: Item>(_: &L, _: &R) {}
/// require(&Til(4i32, Nil), &Til(Til(2i32, Til(4i32, Nil)), Nil));
/// ```
///
/// Equal ranks and leaf counts can have different nested structures.
///
/// ```compile_fail,E0277
/// use noodle_cute::tuple::{Congruent, Item, Nil, Til};
/// fn require<L: Congruent<R>, R: Item>(_: &L, _: &R) {}
/// let left = Til(Til(1i32, Til(2i32, Nil)), Til(3i32, Nil));
/// let right = Til(1i32, Til(Til(2i32, Til(3i32, Nil)), Nil));
/// require(&left, &right);
/// ```
pub const trait Congruent<R: Item>: Item {}
const impl<L: Item, R: Item> Congruent<R> for L
where
    L::Profile: WeaklyCongruentImpl<R::Profile>,
    R::Profile: WeaklyCongruentImpl<L::Profile>,
{
}

/// Constructs a hierarchical [`Tuple`] from nested parenthesized items.
///
/// - `tuple!()` produces [`Nil`].
/// - `tuple!(a, b, ...)` produces `Til(a, Til(b, ...))`.
/// - Inside `tuple!(...)`, `()` builds a nested [`Nil`], `(a)` or `(a,)` builds
///   a nested singleton tuple, and `(a, b, ...)` builds a nested tuple.
///
/// # Examples
///
/// ```
/// # #![feature(const_trait_impl, const_ops, const_cmp)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::tuple;
/// use noodle_cute::tuple::{Nil, Til, TupleContainer};
///
/// const T: TupleContainer<Int<1>, TupleContainer<TupleContainer<Int<2>, TupleContainer<Int<3>, Nil>>, Nil>> =
///     tuple!(Int::<1>, (Int::<2>, Int::<3>));
/// assert_eq!(T, Til(Int::<1>, Til(Til(Int::<2>, Til(Int::<3>, Nil)), Nil)));
/// assert_eq!(tuple!(), Nil);
/// assert_eq!(
///     tuple!((1 + 2) * 2, (3), ()),
///     Til(6i32, Til(Til(3i32, Nil), Til(Nil, Nil)))
/// );
/// ```
#[macro_export]
macro_rules! tuple {
    (@items) => {
        $crate::tuple::Nil
    };
    (@items ($($inner:tt)*) $(, $($rest:tt)*)?) => {
        $crate::tuple::Til(
            $crate::tuple!(@items $($inner)*),
            $crate::tuple!(@items $($($rest)*)?),
        )
    };
    (@items $head:expr $(, $($rest:tt)*)?) => {
        $crate::tuple::Til(
            $head,
            $crate::tuple!(@items $($($rest)*)?),
        )
    };

    ($($input:tt)*) => {
        $crate::tuple::require_tuple($crate::tuple!(@items $($input)*))
    };
}

/// Constructs a hierarchical [`IntTuple`] checked at compile time to contain only
/// integers and nested [`Nil`]s.
///
/// # Examples
///
/// ```
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::tuple::{Nil, Til};
///
/// let t = int_tuple!(Int::<2>, (3i32, Int::<4>));
/// assert_eq!(t, Til(Int::<2>, Til(Til(3i32, Til(Int::<4>, Nil)), Nil)));
/// ```
#[macro_export]
macro_rules! int_tuple {
    ($($input:tt)*) => {
        $crate::tuple::require_int_tuple($crate::tuple!($($input)*))
    };
}

/// Builds a type-level mode index path ([`Mode`] / [`ModeEnd`]) for [`Select`],
/// [`Replace`], and [`Lift`].
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{mode, tuple};
/// use noodle_cute::tuple::{replace, select};
///
/// let t = tuple!(10i32, (20i32, 30i32), 40i32);
/// assert_eq!(select::<mode!(0, -1), _>(&t), tuple!(10i32, 40i32));
/// assert_eq!(replace::<mode!(1, -1), _, _>(&t, 99i32), tuple!(10i32, (20i32, 99i32), 40i32));
/// ```
#[macro_export]
macro_rules! mode {
    () => {
        $crate::tuple::ModeEnd
    };
    ($i:expr $(,)?) => {
        $crate::tuple::Mode<{ $i }, $crate::tuple::ModeEnd>
    };
    ($i:expr, $($rest:tt)+) => {
        $crate::tuple::Mode<{ $i }, $crate::mode!($($rest)+)>
    };
}

/// Follows a sequence of signed mode indices through nested [`Get`] calls.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{tuple, walk_mode};
///
/// let t = tuple!(10i32, (20i32, (30i32, 40i32)));
/// assert_eq!(walk_mode!(t), t);
/// assert_eq!(walk_mode!(t, 1, 1, 0), 30);
/// assert_eq!(walk_mode!(t, -1, -1, -1), 40);
/// ```
#[macro_export]
macro_rules! walk_mode {
    ($tuple:expr $(,)?) => {
        $tuple
    };
    ($tuple:expr, $i:expr $(, $($rest:tt)*)?) => {
        $crate::walk_mode!(
            $crate::tuple::Get::<{ $i }>::get(&($tuple))
            $(, $($rest)*)?
        )
    };
}
