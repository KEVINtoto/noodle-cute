//! Type-level non-negative coordinate paths (`BasisPath`, `PathOrdering`, `ComparePath`).
//!
//! Validates that [`Mode`] / [`ModeEnd`] coordinate paths contain only non-negative
//! indices, tracks path depth, and compares paths lexicographically at compile time
//! while detecting prefix conflicts.
//!
//! # Main Interfaces
//!
//! - [`BasisPath`]: Marker and depth query (`DEPTH`) implemented for [`ModeEnd`] and
//!   non-negative [`Mode<M, Next>`] chains.
//! - [`PathOrdering`]: Compile-time comparison result (`Less`, `Equal`, `Greater`,
//!   `PrefixConflict`).
//! - [`ComparePath`]: Lexicographical comparison between two [`BasisPath`] types.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{basis::{BasisPath, ComparePath, PathOrdering}, mode};
//!
//! assert_eq!(<mode![1, 2] as BasisPath>::DEPTH, 2);
//! assert_eq!(<mode![0, 1] as ComparePath<mode![0, 2]>>::ORDER, PathOrdering::Less);
//! assert_eq!(<mode![0] as ComparePath<mode![0, 1]>>::ORDER, PathOrdering::PrefixConflict);
//! ```
//!
//! # Notes
//!
//! - Negative mode indices fail the [`BasisPath`] bound at compile time.
//! - When one path is a strict prefix of another (e.g. `mode![0]` vs `mode![0, 1]`),
//!   [`ComparePath::ORDER`] evaluates to [`PathOrdering::PrefixConflict`].

use core::cmp::Ordering;
use core::marker::ConstParamTy;

use crate::integer::{IS_NON_NEG, add, cmp as scalar_cmp, eq as scalar_eq, int, lt};
use crate::tuple::{Mode, ModeEnd};

/// Compile-time guard restricting basis path mode indices to non-negative values (`VALID == true`).
#[doc(hidden)]
pub const trait ValidBasisMode<const VALID: bool> {}
const impl ValidBasisMode<true> for () {}

/// Marker and depth query trait for non-negative type-level coordinate paths
/// ([`Mode`] and [`ModeEnd`]).
pub const trait BasisPath: Copy {
    /// Number of mode indices in this path (`0` for [`ModeEnd`]).
    const DEPTH: int;
}

const impl BasisPath for ModeEnd {
    const DEPTH: int = 0;
}

const impl<const M: int, Next: [const] BasisPath> BasisPath for Mode<M, Next>
where
    (): [const] ValidBasisMode<{ IS_NON_NEG::<M> }>,
{
    const DEPTH: int = add(1, Next::DEPTH);
}

/// Appends non-negative mode index `M` to the end of `Self`, producing a longer `BasisPath`.
#[doc(hidden)]
pub const trait AppendMode<const M: int>: [const] BasisPath {
    type Output: [const] BasisPath;
}

const impl<const M: int> AppendMode<M> for ModeEnd
where
    (): [const] ValidBasisMode<{ IS_NON_NEG::<M> }>,
{
    type Output = Mode<M, ModeEnd>;
}

const impl<const H: int, Next: [const] AppendMode<M>, const M: int> AppendMode<M> for Mode<H, Next>
where
    (): [const] ValidBasisMode<{ IS_NON_NEG::<H> }>,
{
    type Output = Mode<H, Next::Output>;
}

/// Compile-time ordering between two [`BasisPath`] types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum PathOrdering {
    /// Default dispatch sentinel used in trait bounds; never returned by [`ComparePath`].
    Dispatch,
    /// Left path precedes right path lexicographically.
    Less,
    /// Both paths are identical.
    Equal,
    /// Left path follows right path lexicographically.
    Greater,
    /// One path is a strict prefix of the other (incompatible hierarchy depths).
    PrefixConflict,
}

/// Lexicographical comparison between two type-level paths, flagging prefix conflicts.
pub const trait ComparePath<R: BasisPath>: BasisPath {
    /// Compile-time ordering between `Self` and `R`.
    const ORDER: PathOrdering;
}

const impl ComparePath<ModeEnd> for ModeEnd {
    const ORDER: PathOrdering = PathOrdering::Equal;
}

const impl<const M: int, Next: BasisPath> ComparePath<Mode<M, Next>> for ModeEnd
where
    Mode<M, Next>: BasisPath,
{
    const ORDER: PathOrdering = PathOrdering::PrefixConflict;
}

const impl<const M: int, Next: BasisPath> ComparePath<ModeEnd> for Mode<M, Next>
where
    Mode<M, Next>: BasisPath,
{
    const ORDER: PathOrdering = PathOrdering::PrefixConflict;
}

const impl<const L: int, NextL: BasisPath, const R: int, NextR: BasisPath>
    ComparePath<Mode<R, NextR>> for Mode<L, NextL>
where
    Mode<L, NextL>: BasisPath,
    Mode<R, NextR>: BasisPath,
    NextL: ComparePath<NextR>,
{
    const ORDER: PathOrdering = match scalar_cmp(L, R) {
        Ordering::Less => PathOrdering::Less,
        Ordering::Greater => PathOrdering::Greater,
        Ordering::Equal => NextL::ORDER,
    };
}

pub(crate) const PATH_ORDER<PL: ComparePath<PR>, PR: BasisPath>: PathOrdering = PL::ORDER;
// Distinguishes which side is the shorter prefix on a `PathOrdering::PrefixConflict`.
pub(crate) const IS_LEFT_PREFIX<PL: BasisPath, PR: BasisPath>: bool = lt(PL::DEPTH, PR::DEPTH);
pub(crate) const EQ_MODE<const M: int, const TARGET: int>: bool = scalar_eq(M, TARGET);
