//! Nested, heterogeneous tuples of static and dynamic integers.
//!
//! [`IntTuple`] stores a head item and a tail ending in [`Nil`]. A head may
//! itself be a tuple, so top-level modes and flattened integer leaves are
//! distinct: `(2, (3, 4))` has two modes and three leaves.

use core::fmt::Debug;
use core::marker::PhantomData;
use core::ops::Mul;

use crate::int::{Int, Integer, mul};
use crate::{Error, Result};

/// An empty tuple and the terminator of an [`IntTuple`] chain.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Nil;

/// A head item followed by a tail tuple; nested heads add tuple levels.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct IntTuple<H, T>(pub H, pub T);

/// Structural marker for an empty tuple.
pub struct NilProfile;
/// Structural marker shared by all integer leaves, regardless of value or type.
pub struct LeafProfile;
/// Structural marker combining the profiles of a tuple's head and tail.
pub struct TupleProfile<H, T>(PhantomData<(H, T)>);

/// An integer leaf or a tuple that supports structural queries and leaf access.
pub trait Item: Clone + Debug {
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
        for i in 0..Self::LEAVES {
            // Scan for zero before multiplying so a zero product cannot overflow.
            if self.leaf(i)? == 0 {
                return Ok(0);
            }
        }
        let mut result = 1;
        for i in 0..Self::LEAVES {
            result = mul(result, self.leaf(i)?)?;
        }
        Ok(result)
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

impl Item for Nil {
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

impl<I: Integer + Debug> Item for I {
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
pub trait Tuple: Item {
    /// Number of heads in the chain, equal to the tuple's top-level rank.
    const LEN: usize;
    /// Greatest depth among the heads, or zero for [`Nil`].
    ///
    /// This excludes the chain's own tuple level and its terminating [`Nil`].
    const MAX_DEPTH: usize;
}

impl Tuple for Nil {
    const LEN: usize = 0;
    const MAX_DEPTH: usize = 0;
}

impl<H: Item, T: Tuple> Item for IntTuple<H, T> {
    type Profile = TupleProfile<H::Profile, T::Profile>;
    const RANK: usize = 1 + T::LEN;
    const DEPTH: usize = 1 + if H::DEPTH > T::MAX_DEPTH {
        H::DEPTH
    } else {
        T::MAX_DEPTH
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

impl<H: Item, T: Tuple> Tuple for IntTuple<H, T> {
    const LEN: usize = 1 + T::LEN;
    const MAX_DEPTH: usize = if H::DEPTH > T::MAX_DEPTH {
        H::DEPTH
    } else {
        T::MAX_DEPTH
    };
}

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

/// Dispatch marker: `true` selects the head; `false` continues into the tail.
pub struct GetHead<const B: bool>;

/// Implements one head-or-tail step of compile-time tuple access.
pub trait Step<H, T, const INDEX: usize> {
    type Output;
    fn step(tuple: &IntTuple<H, T>) -> &Self::Output;
}

impl<H, T, const INDEX: usize> Step<H, T, INDEX> for GetHead<true> {
    type Output = H;
    fn step(tuple: &IntTuple<H, T>) -> &Self::Output {
        &tuple.0
    }
}

impl<H, T: Get<{ SUB_1::<INDEX> }>, const INDEX: usize> Step<H, T, INDEX> for GetHead<false> {
    type Output = T::Output;
    fn step(tuple: &IntTuple<H, T>) -> &Self::Output {
        T::get(&tuple.1)
    }
}

impl<H, T, const INDEX: usize> Get<INDEX> for IntTuple<H, T>
where
    GetHead<{ IS_ZERO::<INDEX> }>: Step<H, T, INDEX>,
{
    type Output = <GetHead<{ IS_ZERO::<INDEX> }> as Step<H, T, INDEX>>::Output;
    fn get(&self) -> &Self::Output {
        GetHead::<{ IS_ZERO::<INDEX> }>::step(self)
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
    type Output;
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
/// [`Nil`] remains empty.
pub trait Flatten {
    type Output;
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

/// Multiplies leaves while retaining a static result whenever all leaves are static.
///
/// A tuple of [`Int`] leaves produces an `Int` whose type encodes the product;
/// [`Nil`] produces `Int<1>`. A runtime leaf makes a tuple's result dynamic.
/// Unlike [`Item::product`], this uses multiplication operators without a zero
/// pre-scan or checked errors; runtime overflow follows Rust's overflow settings.
pub trait Product {
    type Output;
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
    H::Output: Mul<T::Output>,
    <H::Output as Mul<T::Output>>::Output: Integer,
{
    type Output = <H::Output as Mul<T::Output>>::Output;
    fn product_typed(self) -> Self::Output {
        self.0.product_typed() * self.1.product_typed()
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
