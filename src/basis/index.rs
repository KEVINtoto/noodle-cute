//! Signed mode indexing (`Get<I>`) for [`BasisCombination`] and [`BasisTerm`].
//!
//! Enables indexing, [`walk_mode!`](crate::walk_mode!), and [`select`](crate::tuple::select)
//! directly on sparse basis elements with implicit zero-padding for omitted modes.
//!
//! # Main Interfaces
//!
//! - [`BASIS_NORMALIZED_INDEX`]: Normalizes a signed mode index against `MAX_MODE`.
//! - [`Get`] implementations for [`BasisTerm`] and [`BasisCombination`] (also exposed
//!   via [`BasisTerm::get`] and [`BasisCombination::get`]).
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{basis_tuple, integer::Int, walk_mode};
//!
//! let c = basis_tuple![1i32, (2i32, 3i32), 4i32];
//! assert_eq!(c.get::<0>(), 1i32);
//! assert_eq!(c.get::<-1>(), 4i32);
//! assert_eq!(c.get::<5>(), Int::<0>);
//! assert_eq!(walk_mode!(c, 1, 0), 2i32);
//! ```
//!
//! # Notes
//!
//! - Non-negative indices absent from the sparse chain yield [`Int<0>`].
//! - Negative indices count backward from `MAX_MODE + 1` (computed without overflowing
//!   `int::MAX`); indices below `-(MAX_MODE + 1)` or any negative index on
//!   `BasisCombination<Nil>` fail at compile time (`E0080`).

use crate::integer::{IS_NON_NEG, Int, add, eq as scalar_eq, int, max};
use crate::tuple::{Get, Item, Mode, ModeEnd, Nil, Til};

use super::{
    BasisCoeff, BasisCombination, BasisPath, BasisTerm, BasisTuple, COEFF_IS_STATIC_ZERO,
    CanonicalBasisTuple, EQ_MODE, EmitBasisTerm,
};

/// Normalizes a signed mode index `I` against the maximum leading mode `MAX_MODE`
/// (`-1` when no leading mode is present).
///
/// Non-negative indices pass through for zero padding; negative indices count
/// backward from `MAX_MODE + 1` without overflowing [`int`], failing at compile
/// time when below `-(MAX_MODE + 1)`.
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::basis::BASIS_NORMALIZED_INDEX;
///
/// assert_eq!(BASIS_NORMALIZED_INDEX::<0, -1>, 0);
/// assert_eq!(BASIS_NORMALIZED_INDEX::<-1, 2>, 2);
/// assert_eq!(BASIS_NORMALIZED_INDEX::<-3, 2>, 0);
/// ```
///
/// ```compile_fail,E0080
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::basis::BASIS_NORMALIZED_INDEX;
/// const INVALID: i32 = BASIS_NORMALIZED_INDEX::<-1, -1>;
/// ```
pub const BASIS_NORMALIZED_INDEX<const I: int, const MAX_MODE: int>: int = {
    assert!(MAX_MODE >= -1, "basis max mode must be at least -1");
    if IS_NON_NEG::<I> {
        I
    } else {
        assert!(IS_NON_NEG::<MAX_MODE>, "basis index out of bounds");
        let idx = add(MAX_MODE, add(I, 1));
        assert!(idx >= 0, "basis index out of bounds");
        idx
    }
};

const TERM_INDEX_MATCH<const I: int, const M: int>: bool =
    scalar_eq(BASIS_NORMALIZED_INDEX::<I, M>, M);

/// Computes the largest leading mode index in a `BasisTuple` (`-1` when no leading mode exists).
#[doc(hidden)]
pub const trait MaxTopMode: [const] BasisTuple {
    const MAX_MODE: int;
}

const impl MaxTopMode for Nil {
    const MAX_MODE: int = -1;
}

const impl<V: [const] BasisCoeff, T: [const] MaxTopMode> MaxTopMode
    for Til<BasisTerm<V, ModeEnd>, T>
{
    const MAX_MODE: int = T::MAX_MODE;
}

const impl<V: [const] BasisCoeff, const M: int, Next: [const] BasisPath, T: [const] MaxTopMode>
    MaxTopMode for Til<BasisTerm<V, Mode<M, Next>>, T>
where
    Mode<M, Next>: [const] BasisPath,
{
    const MAX_MODE: int = max(M, T::MAX_MODE);
}

/// Extracts terms whose leading mode equals `TARGET`, stripping that outer mode index.
#[doc(hidden)]
pub const trait CombFilterMode<const TARGET: int>: [const] BasisTuple {
    type Output: [const] BasisTuple;
    fn filter_mode(&self) -> Self::Output;
}

/// Single-step dispatch for `CombFilterMode`: `MATCH = true` emits the inner term for `TARGET`.
#[doc(hidden)]
pub const trait CombFilterModeStep<const TARGET: int, const MATCH: bool>:
    [const] BasisTuple
{
    type Output: [const] BasisTuple;
    fn filter_mode_step(&self) -> Self::Output;
}

const impl<const TARGET: int> CombFilterMode<TARGET> for Nil {
    type Output = Nil;
    fn filter_mode(&self) -> Self::Output {
        Nil
    }
}

const impl<const TARGET: int, V: [const] BasisCoeff, T: [const] CombFilterMode<TARGET>>
    CombFilterMode<TARGET> for Til<BasisTerm<V, ModeEnd>, T>
{
    type Output = T::Output;
    fn filter_mode(&self) -> Self::Output {
        self.1.filter_mode()
    }
}

const impl<const TARGET: int, V, const M: int, Next, T> CombFilterMode<TARGET>
    for Til<BasisTerm<V, Mode<M, Next>>, T>
where
    V: [const] BasisCoeff,
    Next: [const] BasisPath,
    Mode<M, Next>: [const] BasisPath,
    T: [const] BasisTuple,
    Self: [const] CombFilterModeStep<TARGET, { EQ_MODE::<M, TARGET> }>,
{
    type Output = <Self as CombFilterModeStep<TARGET, { EQ_MODE::<M, TARGET> }>>::Output;
    fn filter_mode(&self) -> Self::Output {
        self.filter_mode_step()
    }
}

const impl<const TARGET: int, V, const M: int, Next, T> CombFilterModeStep<TARGET, true>
    for Til<BasisTerm<V, Mode<M, Next>>, T>
where
    V: [const] BasisCoeff,
    Next: [const] BasisPath,
    Mode<M, Next>: [const] BasisPath,
    T: [const] CombFilterMode<TARGET>,
    T::Output: [const] EmitBasisTerm<V, Next, { COEFF_IS_STATIC_ZERO::<V> }>,
{
    type Output = <T::Output as EmitBasisTerm<V, Next, { COEFF_IS_STATIC_ZERO::<V> }>>::Output;
    fn filter_mode_step(&self) -> Self::Output {
        self.1.filter_mode().emit(self.0.value)
    }
}

const impl<const TARGET: int, V, const M: int, Next, T> CombFilterModeStep<TARGET, false>
    for Til<BasisTerm<V, Mode<M, Next>>, T>
where
    V: [const] BasisCoeff,
    Next: [const] BasisPath,
    Mode<M, Next>: [const] BasisPath,
    T: [const] CombFilterMode<TARGET>,
{
    type Output = T::Output;
    fn filter_mode_step(&self) -> Self::Output {
        self.1.filter_mode()
    }
}

/// Collapses filtered terms for one mode into `Int<0>`, scalar `V`, `BasisTerm`, or `BasisCombination`.
#[doc(hidden)]
pub const trait UnwrapModeOutput: [const] BasisTuple {
    type Output: Copy;
    fn unwrap_mode(self) -> Self::Output;
}

const impl UnwrapModeOutput for Nil {
    type Output = Int<0>;
    fn unwrap_mode(self) -> Self::Output {
        Int
    }
}

const impl<V: [const] BasisCoeff> UnwrapModeOutput for Til<BasisTerm<V, ModeEnd>, Nil> {
    type Output = V;
    fn unwrap_mode(self) -> Self::Output {
        self.0.value
    }
}

const impl<V: [const] BasisCoeff, const M: int, Next: [const] BasisPath> UnwrapModeOutput
    for Til<BasisTerm<V, Mode<M, Next>>, Nil>
where
    Mode<M, Next>: [const] BasisPath,
{
    type Output = BasisTerm<V, Mode<M, Next>>;
    fn unwrap_mode(self) -> Self::Output {
        self.0
    }
}

const impl<V0, P0, V1, P1, T> UnwrapModeOutput for Til<BasisTerm<V0, P0>, Til<BasisTerm<V1, P1>, T>>
where
    V0: [const] BasisCoeff,
    P0: [const] BasisPath,
    V1: [const] BasisCoeff,
    P1: [const] BasisPath,
    T: [const] BasisTuple,
    Self: [const] CanonicalBasisTuple,
{
    type Output = BasisCombination<Self>;
    fn unwrap_mode(self) -> Self::Output {
        BasisCombination(self)
    }
}

const impl<T, const I: int> Get<I> for BasisCombination<T>
where
    T: [const] MaxTopMode
        + [const] CombFilterMode<{ BASIS_NORMALIZED_INDEX::<I, { T::MAX_MODE }> }>,
    <T as CombFilterMode<{ BASIS_NORMALIZED_INDEX::<I, { T::MAX_MODE }> }>>::Output:
        [const] UnwrapModeOutput,
    <<T as CombFilterMode<{ BASIS_NORMALIZED_INDEX::<I, { T::MAX_MODE }> }>>::Output as UnwrapModeOutput>::Output:
        [const] Item,
{
    type Output = <<T as CombFilterMode<{ BASIS_NORMALIZED_INDEX::<I, { T::MAX_MODE }> }>>::Output as UnwrapModeOutput>::Output;
    fn get(&self) -> Self::Output {
        self.0.filter_mode().unwrap_mode()
    }
}

/// Unwraps the remaining path of a matched `BasisTerm<V, Mode<M, Next>>`:
/// `ModeEnd` yields scalar `V`, nested `Mode` yields `BasisTerm<V, Self>`.
#[doc(hidden)]
pub const trait UnwrapTermNext<V: [const] BasisCoeff>: [const] BasisPath {
    type Output: Copy;
    fn unwrap_term(value: V) -> Self::Output;
}

const impl<V: [const] BasisCoeff> UnwrapTermNext<V> for ModeEnd {
    type Output = V;
    fn unwrap_term(value: V) -> Self::Output {
        value
    }
}

const impl<V: [const] BasisCoeff, const M: int, Next: [const] BasisPath> UnwrapTermNext<V>
    for Mode<M, Next>
where
    Mode<M, Next>: [const] BasisPath,
{
    type Output = BasisTerm<V, Mode<M, Next>>;
    fn unwrap_term(value: V) -> Self::Output {
        BasisTerm::new(value)
    }
}

/// Dispatches `Get<I>` on a single `BasisTerm`: `MATCH = false` pads with `Int<0>`,
/// `MATCH = true` strips the leading mode via `UnwrapTermNext`.
#[doc(hidden)]
pub const trait BasisTermGetImpl<const MATCH: bool> {
    type Output: Copy;
    fn term_get(&self) -> Self::Output;
}

const impl<V: [const] BasisCoeff, const M: int, Next: [const] BasisPath> BasisTermGetImpl<false>
    for BasisTerm<V, Mode<M, Next>>
{
    type Output = Int<0>;
    fn term_get(&self) -> Self::Output {
        Int
    }
}

const impl<V: [const] BasisCoeff, const M: int, Next: [const] UnwrapTermNext<V>>
    BasisTermGetImpl<true> for BasisTerm<V, Mode<M, Next>>
{
    type Output = <Next as UnwrapTermNext<V>>::Output;
    fn term_get(&self) -> Self::Output {
        Next::unwrap_term(self.value)
    }
}

const impl<V, const M: int, Next, const I: int> Get<I> for BasisTerm<V, Mode<M, Next>>
where
    V: [const] BasisCoeff,
    Next: [const] BasisPath,
    Mode<M, Next>: [const] BasisPath,
    Self: [const] BasisTermGetImpl<{ TERM_INDEX_MATCH::<I, M> }>,
    <Self as BasisTermGetImpl<{ TERM_INDEX_MATCH::<I, M> }>>::Output: [const] Item,
{
    type Output = <Self as BasisTermGetImpl<{ TERM_INDEX_MATCH::<I, M> }>>::Output;
    fn get(&self) -> Self::Output {
        self.term_get()
    }
}
