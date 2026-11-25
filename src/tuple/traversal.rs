//! Single-tree and synchronized multi-tree depth-first leaf traversal.
//!
//! Traverses one primary [`Item`] tree alongside `0..=5` peer trees in lockstep,
//! supporting leaf mapping, custom node reconstruction, leaf zipping, type-evolving
//! left folds, and short-circuiting leaf predicates.
//!
//! # Main Interfaces
//!
//! - **Single-tree mapping**: [`TransformLeaf`] and [`TransformLeafFunctor`].
//! - **Generalized multi-tree transform**: [`TransformApplyLeaf`], [`transform_apply_leaf`],
//!   [`ApplyLeafFunctor`], [`MakeNodeFunctor`], [`MakeTuple`], [`ZipTransformLeaf`], and
//!   [`zip_transform_leaf`].
//! - **Leaf zipping**: [`ZipLeaves`], [`zip_leaves`], and [`ZipLeaf`].
//! - **Folds & predicates**: [`FoldLeaf`], [`fold_leaf`], [`FoldLeafFunctor`],
//!   [`AllLeaves`], [`all_leaves`], [`AnyLeaf`], [`any_leaf`], and [`LeafPredicate`].
//!
//! # Usage
//!
//! Pass peers as `()` (0 peers, functor receives `L`), `peer` (1 peer, functor receives
//! `(L, B)`), or a Rust tuple `(p0, p1, ...)` (up to 5 peers, functor receives `(L, B0, ...)`):
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{int_tuple, tuple};
//! use noodle_cute::tuple::{ZipLeaf, zip_leaves};
//!
//! let a = int_tuple![1, (2, 3)];
//! let b = int_tuple![10, (20, 30)];
//! assert_eq!(
//!     zip_leaves(&a, b),
//!     tuple![ZipLeaf((1, 10)), ZipLeaf((2, 20)), ZipLeaf((3, 30))]
//! );
//! ```
//!
//! # Notes
//!
//! - The primary tree must be weakly congruent with every peer: when the primary reaches
//!   a [`Leaf`], each peer's corresponding subtree is passed directly to the functor.
//! - [`MakeNodeFunctor::make`] is called once per tuple root and once for [`Nil`], never
//!   for intermediate cons tails.
//! - [`AllLeaves`] is vacuously `true` and [`AnyLeaf`] is `false` on [`Nil`].

use crate::integer::int;

use super::{FlatTuple, Item, Leaf, LeafProfile, Nil, Til, Tuple};

/// A mapping from one leaf type to an [`Item`] for [`TransformLeaf`].
///
/// Implement this trait for every input leaf type the functor handles. The
/// output can be an integer, custom item, [`Nil`], or a nested [`Til`].
/// A const implementation enables const evaluation of the mapping.
///
/// # Examples
///
/// ```
/// # #![feature(const_trait_impl)]
/// use noodle_cute::tuple::TransformLeafFunctor;
///
/// #[derive(Clone, Copy)]
/// struct Double;
/// const impl TransformLeafFunctor<i32> for Double {
///     type Output = i32;
///     fn apply(&self, leaf: i32) -> i32 {
///         leaf * 2
///     }
/// }
/// const SIX: i32 = Double.apply(3);
/// assert_eq!(SIX, 6);
/// ```
pub const trait TransformLeafFunctor<L: [const] Leaf> {
    /// The mapped value type.
    type Output: [const] Item;
    /// Maps one leaf.
    fn apply(&self, leaf: L) -> Self::Output;
}

/// Maps scalar leaves while retaining the enclosing tuple chains.
///
/// Leaf dispatch requires [`Item::Profile`] to be [`LeafProfile`] and `F` to
/// implement [`TransformLeafFunctor`] for that leaf type. This covers integers
/// and registered custom leaves. Each mapped result implements [`Item`] and is
/// inserted directly into the surrounding structure.
///
/// Nonempty tuples visit heads before tails in depth-first order. Empty tuples
/// return `Nil` without invoking the functor. A mapped result can introduce
/// nesting or an empty head, and its contents receive no further mapping during
/// this call.
///
/// # Examples
///
/// ```
/// # #![feature(const_trait_impl)]
/// use noodle_cute::tuple::{Item, LeafProfile, Nil, TransformLeaf, TransformLeafFunctor, Til};
///
/// #[derive(Clone, Copy, Debug, PartialEq)]
/// struct Pair(i32, i32);
/// const impl Item for Pair {
///     const RANK: i32 = 1;
///     const DEPTH: i32 = 0;
///     const LEAVES: i32 = 1;
///     type Profile = LeafProfile;
/// }
/// #[derive(Clone, Copy)]
/// struct ToPair;
/// const impl TransformLeafFunctor<i32> for ToPair {
///     type Output = Pair;
///     fn apply(&self, leaf: i32) -> Pair {
///         Pair(leaf, leaf + 1)
///     }
/// }
/// const MAPPED: Til<Pair, Nil> = Til::new(3i32, Nil).transform(&ToPair);
/// assert_eq!(MAPPED, Til::new(Pair(3, 4), Nil));
/// let nested = Til::new(1i32, Til::new(Til::new(2i32, Nil), Nil));
/// assert_eq!(
///     nested.transform(&ToPair),
///     Til::new(
///         Pair(1, 2),
///         Til::new(Til::new(Pair(2, 3), Nil), Nil),
///     )
/// );
/// assert_eq!(Nil.transform(&ToPair), Nil);
/// ```
///
/// A functor may map leaves to empty or nonempty tuples.
///
/// ```
/// use noodle_cute::tuple::{Nil, TransformLeaf, TransformLeafFunctor, Til};
///
/// #[derive(Clone, Copy)]
/// struct Empty;
/// impl TransformLeafFunctor<i32> for Empty {
///     type Output = Nil;
///     fn apply(&self, _: i32) -> Nil {
///         Nil
///     }
/// }
/// assert_eq!(1i32.transform(&Empty), Nil);
/// assert_eq!(Til::new(1i32, Nil).transform(&Empty), Til::new(Nil, Nil));
///
/// #[derive(Clone, Copy)]
/// struct Duplicate;
/// impl TransformLeafFunctor<i32> for Duplicate {
///     type Output = Til<i32, Til<i32, Nil>>;
///     fn apply(&self, leaf: i32) -> Self::Output {
///         Til::new(leaf, Til::new(leaf, Nil))
///     }
/// }
/// assert_eq!(
///     Til::new(2i32, Nil).transform(&Duplicate),
///     Til::new(Til::new(2i32, Til::new(2i32, Nil)), Nil)
/// );
/// ```
///
/// A native Rust pair fails the mapped output's `Item` bound.
///
/// ```compile_fail,E0277
/// use noodle_cute::tuple::TransformLeafFunctor;
/// struct ToPair;
/// impl TransformLeafFunctor<i32> for ToPair {
///     type Output = (i32, i32);
///     fn apply(&self, leaf: i32) -> Self::Output { (leaf, leaf) }
/// }
/// ```
pub const trait TransformLeaf<F>: [const] Item {
    /// The mapped item, retaining enclosing tuple chains around mapped leaves.
    type Output: [const] Item;
    /// Copies the input and applies `f` to its leaves.
    fn transform(&self, f: &F) -> Self::Output;
}

const impl<F> TransformLeaf<F> for Nil {
    type Output = Nil;
    fn transform(&self, _: &F) -> Self::Output {
        Nil
    }
}

const impl<L, F> TransformLeaf<F> for L
where
    L: [const] Leaf,
    F: [const] TransformLeafFunctor<L>,
{
    type Output = F::Output;
    fn transform(&self, f: &F) -> Self::Output {
        f.apply(*self)
    }
}

const impl<H, T, F> TransformLeaf<F> for Til<H, T>
where
    H: [const] Item + [const] TransformLeaf<F>,
    T: [const] Tuple + [const] TransformLeaf<F>,
    H::Output: [const] Item,
    T::Output: [const] Tuple,
{
    type Output = Til<H::Output, T::Output>;
    fn transform(&self, f: &F) -> Self::Output {
        Til(self.0.transform(f), self.1.transform(f))
    }
}

/// Computes combined `IS_STATIC` over a native Rust tuple stored inside [`ZipLeaf`].
#[doc(hidden)]
pub const trait TupleIsStatic {
    const IS_STATIC: bool = false;
}

const impl TupleIsStatic for () {
    const IS_STATIC: bool = true;
}

macro_rules! impl_tuple_is_static {
    ($($T:ident),+) => {
        const impl<$($T: Item),+> TupleIsStatic for ($($T,)+) {
            const IS_STATIC: bool = true $(&& $T::IS_STATIC)+;
        }
    };
}

impl_tuple_is_static!(A);
impl_tuple_is_static!(A, B);
impl_tuple_is_static!(A, B, C);
impl_tuple_is_static!(A, B, C, D);
impl_tuple_is_static!(A, B, C, D, E);
impl_tuple_is_static!(A, B, C, D, E, F);

/// Transparent [`Leaf`] wrapper around a native Rust tuple produced by [`ZipLeaves`].
///
/// Native tuples do not implement [`Item`], so [`ZipLeaves`] stores each
/// zipped leaf tuple inside `ZipLeaf<T>` to yield a valid [`FlatTuple`].
#[derive(Clone, Copy, Default, Debug)]
#[repr(transparent)]
pub struct ZipLeaf<T>(pub T);

const impl<L: [const] PartialEq<R>, R> PartialEq<ZipLeaf<R>> for ZipLeaf<L> {
    fn eq(&self, rhs: &ZipLeaf<R>) -> bool {
        self.0 == rhs.0
    }
}

const impl<T: [const] Eq> Eq for ZipLeaf<T> {}

impl<T: Copy> ZipLeaf<T> {
    /// Extracts the inner native Rust tuple.
    pub const fn into_inner(self) -> T {
        self.0
    }
}

impl<T> core::ops::Deref for ZipLeaf<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

const impl<T: Copy + TupleIsStatic> Item for ZipLeaf<T> {
    const RANK: int = 1;
    const DEPTH: int = 0;
    const LEAVES: int = 1;
    const IS_STATIC: bool = T::IS_STATIC;
    type Profile = LeafProfile;
}

/// Verifies that 0..=5 peer trees all match a [`Nil`] node during lockstep traversal.
#[doc(hidden)]
pub const trait PeersMatchNil: Copy {}

/// Attaches primary leaf `L` to 0..=5 synchronized peer subtrees:
/// `Args` is `L` (0 peers) or `(L, B...)`; `ZippedTuple` is always a tuple `(L, ...)` for [`ZipLeaf`].
#[doc(hidden)]
pub const trait PeersAttachLeaf<L: [const] Leaf>: Copy {
    type Args: Copy;
    type ZippedTuple: Copy + TupleIsStatic;
    fn attach_leaf(self, leaf: L) -> Self::Args;
    fn attach_zip(self, leaf: L) -> Self::ZippedTuple;
}

/// Splits 0..=5 peer `Til` nodes into corresponding `(Heads, Tails)` for recursion.
#[doc(hidden)]
pub const trait PeersSplit: Copy {
    type Heads: Copy;
    type Tails: Copy;
    fn split(self) -> (Self::Heads, Self::Tails);
}

const impl PeersMatchNil for () {}
const impl<L: [const] Leaf> PeersAttachLeaf<L> for () {
    type Args = L;
    type ZippedTuple = (L,);
    fn attach_leaf(self, leaf: L) -> Self::Args {
        leaf
    }
    fn attach_zip(self, leaf: L) -> Self::ZippedTuple {
        (leaf,)
    }
}
const impl PeersSplit for () {
    type Heads = ();
    type Tails = ();
    fn split(self) -> (Self::Heads, Self::Tails) {
        ((), ())
    }
}

const impl PeersMatchNil for Nil {}
const impl<L: [const] Leaf, B: [const] Item> PeersAttachLeaf<L> for B {
    type Args = (L, B);
    type ZippedTuple = (L, B);
    fn attach_leaf(self, leaf: L) -> Self::Args {
        (leaf, self)
    }
    fn attach_zip(self, leaf: L) -> Self::ZippedTuple {
        (leaf, self)
    }
}
const impl<H: [const] Item, T: [const] Tuple> PeersSplit for Til<H, T> {
    type Heads = H;
    type Tails = T;
    fn split(self) -> (Self::Heads, Self::Tails) {
        (self.0, self.1)
    }
}

macro_rules! impl_peers_tuple {
    ($(($P:ident, $H:ident, $T:ident, $v:ident)),+) => {
        const impl<$($P: PeersMatchNil),+> PeersMatchNil for ($($P,)+) {}

        const impl<L: [const] Leaf, $($P: [const] Item),+> PeersAttachLeaf<L> for ($($P,)+) {
            type Args = (L, $($P),+);
            type ZippedTuple = (L, $($P),+);
            fn attach_leaf(self, leaf: L) -> Self::Args {
                let ($($v,)+) = self;
                (leaf, $($v),+)
            }
            fn attach_zip(self, leaf: L) -> Self::ZippedTuple {
                let ($($v,)+) = self;
                (leaf, $($v),+)
            }
        }

        const impl<$($H: [const] Item, $T: [const] Tuple),+> PeersSplit
            for ($(Til<$H, $T>,)+)
        {
            type Heads = ($($H,)+);
            type Tails = ($($T,)+);
            fn split(self) -> (Self::Heads, Self::Tails) {
                let ($($v,)+) = self;
                (($($v.0,)+), ($($v.1,)+))
            }
        }
    };
}

impl_peers_tuple!((B0, H0, T0, p0));
impl_peers_tuple!((B0, H0, T0, p0), (B1, H1, T1, p1));
impl_peers_tuple!((B0, H0, T0, p0), (B1, H1, T1, p1), (B2, H2, T2, p2));
impl_peers_tuple!(
    (B0, H0, T0, p0),
    (B1, H1, T1, p1),
    (B2, H2, T2, p2),
    (B3, H3, T3, p3)
);
impl_peers_tuple!(
    (B0, H0, T0, p0),
    (B1, H1, T1, p1),
    (B2, H2, T2, p2),
    (B3, H3, T3, p3),
    (B4, H4, T4, p4)
);

/// Functor mapping synchronized leaf arguments `Args` in [`TransformApplyLeaf`].
pub const trait ApplyLeafFunctor<Args> {
    /// Mapped leaf output type (need not implement [`Item`]).
    type Output;
    /// Maps synchronized leaf arguments.
    fn apply(&self, args: Args) -> Self::Output;
}

/// Functor combining child results at tuple level roots and [`Nil`] nodes in [`TransformApplyLeaf`] / [`Lift`](super::Lift).
pub const trait MakeNodeFunctor<Children> {
    /// Reconstructed node output type.
    type Output;
    /// Combines mapped children at one tuple root or `Nil`.
    fn make(&self, children: Children) -> Self::Output;
}

/// Default [`MakeNodeFunctor`] reconstructing [`Nil`] and [`Til`] nodes unchanged.
#[derive(Clone, Copy, Default, Debug)]
pub struct MakeTuple;

const impl MakeNodeFunctor<Nil> for MakeTuple {
    type Output = Nil;
    fn make(&self, _: Nil) -> Nil {
        Nil
    }
}

const impl<H: [const] Item, T: [const] Tuple> MakeNodeFunctor<Til<H, T>> for MakeTuple {
    type Output = Til<H, T>;
    fn make(&self, children: Til<H, T>) -> Til<H, T> {
        children
    }
}

/// Generalized multi-tree leaf traversal with a customizable node builder `Make`.
///
/// Unlike [`TransformLeaf`], neither leaf outputs nor `Make::Output` are required
/// to implement [`Item`], and `make.make(...)` is called once per tuple root (not per cons tail).
pub const trait TransformApplyLeaf<Make, F, Peers = ()>: [const] Item {
    /// Output produced at this subtree.
    type Output;
    /// Recursively transforms leaves and rebuilds tuple nodes with `make`.
    fn transform_apply(&self, make: &Make, f: &F, peers: Peers) -> Self::Output;
}

/// Traverses remaining cons cells at one tuple level without re-invoking `Make`,
/// ensuring [`MakeNodeFunctor::make`] runs once per tuple root (`F`: leaf functor, `Peers`: peer tails).
#[doc(hidden)]
pub const trait TransformApplyLevel<Make, F, Peers = ()>: [const] Tuple {
    type Output;
    fn transform_apply_level(&self, make: &Make, f: &F, peers: Peers) -> Self::Output;
}

const impl<Make, F, Peers: PeersMatchNil> TransformApplyLevel<Make, F, Peers> for Nil {
    type Output = Nil;
    fn transform_apply_level(&self, _: &Make, _: &F, _: Peers) -> Nil {
        Nil
    }
}

const impl<H, T, Make, F, Peers> TransformApplyLevel<Make, F, Peers> for Til<H, T>
where
    H: [const] Item + [const] TransformApplyLeaf<Make, F, Peers::Heads>,
    T: [const] Tuple + [const] TransformApplyLevel<Make, F, Peers::Tails>,
    Peers: [const] PeersSplit,
{
    type Output = Til<H::Output, T::Output>;
    fn transform_apply_level(&self, make: &Make, f: &F, peers: Peers) -> Self::Output {
        let (ph, pt) = peers.split();
        Til(
            <H as TransformApplyLeaf<Make, F, Peers::Heads>>::transform_apply(&self.0, make, f, ph),
            <T as TransformApplyLevel<Make, F, Peers::Tails>>::transform_apply_level(
                &self.1, make, f, pt,
            ),
        )
    }
}

const impl<Make, F, Peers: PeersMatchNil> TransformApplyLeaf<Make, F, Peers> for Nil
where
    Make: [const] MakeNodeFunctor<Nil>,
{
    type Output = Make::Output;
    fn transform_apply(&self, make: &Make, _: &F, _: Peers) -> Self::Output {
        make.make(Nil)
    }
}

const impl<L: [const] Leaf, Make, F, Peers> TransformApplyLeaf<Make, F, Peers> for L
where
    Peers: [const] PeersAttachLeaf<L>,
    F: [const] ApplyLeafFunctor<Peers::Args>,
{
    type Output = F::Output;
    fn transform_apply(&self, _: &Make, f: &F, peers: Peers) -> Self::Output {
        f.apply(peers.attach_leaf(*self))
    }
}

const impl<H, T, Make, F, Peers> TransformApplyLeaf<Make, F, Peers> for Til<H, T>
where
    H: [const] Item + [const] TransformApplyLeaf<Make, F, Peers::Heads>,
    T: [const] Tuple + [const] TransformApplyLevel<Make, F, Peers::Tails>,
    Peers: [const] PeersSplit,
    Make: [const] MakeNodeFunctor<Til<H::Output, T::Output>>,
{
    type Output = Make::Output;
    fn transform_apply(&self, make: &Make, f: &F, peers: Peers) -> Self::Output {
        let (ph, pt) = peers.split();
        let children = Til(
            <H as TransformApplyLeaf<Make, F, Peers::Heads>>::transform_apply(&self.0, make, f, ph),
            <T as TransformApplyLevel<Make, F, Peers::Tails>>::transform_apply_level(
                &self.1, make, f, pt,
            ),
        );
        make.make(children)
    }
}

/// Applies [`TransformApplyLeaf`] across synchronized peer trees.
pub const fn transform_apply_leaf<T, Make, F, Peers>(
    t: &T,
    make: &Make,
    f: &F,
    peers: Peers,
) -> T::Output
where
    T: [const] TransformApplyLeaf<Make, F, Peers>,
{
    t.transform_apply(make, f, peers)
}

/// Synchronized leaf transformation preserving enclosing [`Til`] structure via [`MakeTuple`].
pub const trait ZipTransformLeaf<F, Peers = ()>: [const] Item {
    /// Mapped [`Item`] retaining matching tuple structure.
    type Output: [const] Item;
    /// Maps synchronized leaves using `f`.
    fn zip_transform(&self, f: &F, peers: Peers) -> Self::Output;
}

const impl<T, F, Peers> ZipTransformLeaf<F, Peers> for T
where
    T: [const] TransformApplyLeaf<MakeTuple, F, Peers>,
    T::Output: [const] Item,
{
    type Output = T::Output;
    fn zip_transform(&self, f: &F, peers: Peers) -> Self::Output {
        self.transform_apply(&MakeTuple, f, peers)
    }
}

/// Maps leaves across synchronized peer trees while preserving tuple hierarchy.
pub const fn zip_transform_leaf<T, F, Peers>(t: &T, f: &F, peers: Peers) -> T::Output
where
    T: [const] ZipTransformLeaf<F, Peers>,
{
    t.zip_transform(f, peers)
}

/// Zips leaves of `Self` and `Peers` right-to-left onto flat accumulator `Tail` in a single pass.
#[doc(hidden)]
pub const trait ZipLeavesTo<Peers, Tail: [const] FlatTuple>: [const] Item {
    type Output: [const] FlatTuple;
    fn zip_leaves_to(&self, peers: Peers, tail: Tail) -> Self::Output;
}

const impl<Peers: PeersMatchNil, Tail: [const] FlatTuple> ZipLeavesTo<Peers, Tail> for Nil {
    type Output = Tail;
    fn zip_leaves_to(&self, _: Peers, tail: Tail) -> Tail {
        tail
    }
}

const impl<L: [const] Leaf, Peers: [const] PeersAttachLeaf<L>, Tail: [const] FlatTuple>
    ZipLeavesTo<Peers, Tail> for L
{
    type Output = Til<ZipLeaf<Peers::ZippedTuple>, Tail>;
    fn zip_leaves_to(&self, peers: Peers, tail: Tail) -> Self::Output {
        Til(ZipLeaf(peers.attach_zip(*self)), tail)
    }
}

const impl<H, T, Peers, Tail> ZipLeavesTo<Peers, Tail> for Til<H, T>
where
    Tail: [const] FlatTuple,
    Peers: [const] PeersSplit,
    T: [const] Tuple + [const] ZipLeavesTo<Peers::Tails, Tail>,
    H: [const] Item + [const] ZipLeavesTo<Peers::Heads, T::Output>,
{
    type Output = H::Output;
    fn zip_leaves_to(&self, peers: Peers, tail: Tail) -> Self::Output {
        let (ph, pt) = peers.split();
        let tail_zipped = <T as ZipLeavesTo<Peers::Tails, Tail>>::zip_leaves_to(&self.1, pt, tail);
        <H as ZipLeavesTo<Peers::Heads, T::Output>>::zip_leaves_to(&self.0, ph, tail_zipped)
    }
}

/// Zips corresponding leaves across `self` and `peers` into a flat list of [`ZipLeaf`] entries.
pub const trait ZipLeaves<Peers = ()>: [const] Item {
    /// Flat tuple of [`ZipLeaf`] values in depth-first order.
    type Output: [const] FlatTuple;
    /// Collects synchronized leaves into a flat tuple.
    fn zip_leaves(&self, peers: Peers) -> Self::Output;
}

const impl<Peers: PeersMatchNil> ZipLeaves<Peers> for Nil {
    type Output = Nil;
    fn zip_leaves(&self, _: Peers) -> Nil {
        Nil
    }
}

const impl<L: [const] Leaf, Peers: [const] PeersAttachLeaf<L>> ZipLeaves<Peers> for L {
    type Output = Til<ZipLeaf<Peers::ZippedTuple>, Nil>;
    fn zip_leaves(&self, peers: Peers) -> Self::Output {
        Til(ZipLeaf(peers.attach_zip(*self)), Nil)
    }
}

const impl<H, T, Peers> ZipLeaves<Peers> for Til<H, T>
where
    H: [const] Item,
    T: [const] Tuple,
    Self: [const] ZipLeavesTo<Peers, Nil>,
{
    type Output = <Self as ZipLeavesTo<Peers, Nil>>::Output;
    fn zip_leaves(&self, peers: Peers) -> Self::Output {
        self.zip_leaves_to(peers, Nil)
    }
}

/// Zips corresponding leaves across `t` and `peers` in depth-first order.
pub const fn zip_leaves<T: [const] ZipLeaves<Peers>, Peers>(t: &T, peers: Peers) -> T::Output {
    t.zip_leaves(peers)
}

/// Heterogeneous accumulator step functor for [`FoldLeaf`].
pub const trait FoldLeafFunctor<Acc, Args> {
    /// Updated accumulator type after visiting one leaf position.
    type Output;
    /// Combines accumulator `acc` with synchronized leaf arguments `args`.
    fn apply(&self, acc: Acc, args: Args) -> Self::Output;
}

/// Depth-first left fold over synchronized leaves, allowing the accumulator type to evolve per step.
pub const trait FoldLeaf<F, Acc, Peers = ()>: [const] Item {
    /// Final accumulator type after visiting all leaves.
    type Output;
    /// Folds synchronized leaves from left to right starting with `init`.
    fn fold_leaf(&self, init: Acc, f: &F, peers: Peers) -> Self::Output;
}

const impl<F, Acc, Peers: PeersMatchNil> FoldLeaf<F, Acc, Peers> for Nil {
    type Output = Acc;
    fn fold_leaf(&self, init: Acc, _: &F, _: Peers) -> Acc {
        init
    }
}

const impl<L: [const] Leaf, F, Acc, Peers> FoldLeaf<F, Acc, Peers> for L
where
    Peers: [const] PeersAttachLeaf<L>,
    F: [const] FoldLeafFunctor<Acc, Peers::Args>,
{
    type Output = F::Output;
    fn fold_leaf(&self, init: Acc, f: &F, peers: Peers) -> Self::Output {
        f.apply(init, peers.attach_leaf(*self))
    }
}

const impl<H, T, F, Acc, Peers> FoldLeaf<F, Acc, Peers> for Til<H, T>
where
    Peers: [const] PeersSplit,
    H: [const] Item + [const] FoldLeaf<F, Acc, Peers::Heads>,
    T: [const] Tuple + [const] FoldLeaf<F, H::Output, Peers::Tails>,
{
    type Output = T::Output;
    fn fold_leaf(&self, init: Acc, f: &F, peers: Peers) -> Self::Output {
        let (ph, pt) = peers.split();
        let acc1 = <H as FoldLeaf<F, Acc, Peers::Heads>>::fold_leaf(&self.0, init, f, ph);
        <T as FoldLeaf<F, H::Output, Peers::Tails>>::fold_leaf(&self.1, acc1, f, pt)
    }
}

/// Depth-first left fold across `t` and synchronized `peers`.
pub const fn fold_leaf<T, F, Acc, Peers>(t: &T, init: Acc, f: &F, peers: Peers) -> T::Output
where
    T: [const] FoldLeaf<F, Acc, Peers>,
{
    t.fold_leaf(init, f, peers)
}

/// Predicate over synchronized leaf arguments for [`AllLeaves`] and [`AnyLeaf`].
pub const trait LeafPredicate<Args> {
    /// Evaluates the predicate on synchronized leaf arguments.
    fn test(&self, args: Args) -> bool;
}

/// Returns `true` iff every synchronized leaf satisfies `pred` (vacuously `true` for `Nil`).
pub const trait AllLeaves<P, Peers = ()>: [const] Item {
    /// Tests `pred` across all leaves with short-circuit conjunction.
    fn all_leaves(&self, pred: &P, peers: Peers) -> bool;
}

const impl<P, Peers: PeersMatchNil> AllLeaves<P, Peers> for Nil {
    fn all_leaves(&self, _: &P, _: Peers) -> bool {
        true
    }
}

const impl<L: [const] Leaf, P, Peers> AllLeaves<P, Peers> for L
where
    Peers: [const] PeersAttachLeaf<L>,
    P: [const] LeafPredicate<Peers::Args>,
{
    fn all_leaves(&self, pred: &P, peers: Peers) -> bool {
        pred.test(peers.attach_leaf(*self))
    }
}

const impl<H, T, P, Peers> AllLeaves<P, Peers> for Til<H, T>
where
    Peers: [const] PeersSplit,
    H: [const] Item + [const] AllLeaves<P, Peers::Heads>,
    T: [const] Tuple + [const] AllLeaves<P, Peers::Tails>,
{
    fn all_leaves(&self, pred: &P, peers: Peers) -> bool {
        let (ph, pt) = peers.split();
        <H as AllLeaves<P, Peers::Heads>>::all_leaves(&self.0, pred, ph)
            && <T as AllLeaves<P, Peers::Tails>>::all_leaves(&self.1, pred, pt)
    }
}

/// Returns `true` iff any synchronized leaf satisfies `pred` (`false` for `Nil`).
pub const trait AnyLeaf<P, Peers = ()>: [const] Item {
    /// Tests `pred` across all leaves with short-circuit disjunction.
    fn any_leaf(&self, pred: &P, peers: Peers) -> bool;
}

const impl<P, Peers: PeersMatchNil> AnyLeaf<P, Peers> for Nil {
    fn any_leaf(&self, _: &P, _: Peers) -> bool {
        false
    }
}

const impl<L: [const] Leaf, P, Peers> AnyLeaf<P, Peers> for L
where
    Peers: [const] PeersAttachLeaf<L>,
    P: [const] LeafPredicate<Peers::Args>,
{
    fn any_leaf(&self, pred: &P, peers: Peers) -> bool {
        pred.test(peers.attach_leaf(*self))
    }
}

const impl<H, T, P, Peers> AnyLeaf<P, Peers> for Til<H, T>
where
    Peers: [const] PeersSplit,
    H: [const] Item + [const] AnyLeaf<P, Peers::Heads>,
    T: [const] Tuple + [const] AnyLeaf<P, Peers::Tails>,
{
    fn any_leaf(&self, pred: &P, peers: Peers) -> bool {
        let (ph, pt) = peers.split();
        <H as AnyLeaf<P, Peers::Heads>>::any_leaf(&self.0, pred, ph)
            || <T as AnyLeaf<P, Peers::Tails>>::any_leaf(&self.1, pred, pt)
    }
}

/// Evaluates whether all synchronized leaves in `t` (with `peers`) satisfy `pred`.
pub const fn all_leaves<T, P, Peers>(t: &T, pred: &P, peers: Peers) -> bool
where
    T: [const] AllLeaves<P, Peers>,
{
    t.all_leaves(pred, peers)
}

/// Evaluates whether any synchronized leaf in `t` (with `peers`) satisfy `pred`.
pub const fn any_leaf<T, P, Peers>(t: &T, pred: &P, peers: Peers) -> bool
where
    T: [const] AnyLeaf<P, Peers>,
{
    t.any_leaf(pred, peers)
}
