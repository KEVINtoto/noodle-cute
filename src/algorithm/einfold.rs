//! Named mode rearrangement (`einfold`) for [`Layout`]s, [`Tensor`]s, and hierarchical tuples.
//!
//! `einfold!("in_modes -> out_modes", value)` is a generalized zero-copy transpose that
//! permutes, groups, ungroups, duplicates, and drops modes by naming them with single
//! alphanumeric characters. Parentheses descend into nested source modes on the left and
//! construct nested mode groups on the right. Top-level modes left unnamed on the input
//! side are appended unchanged as a passthrough suffix.
//!
//! # Main Interfaces
//!
//! - [`parse_einfold`] / [`EinfoldSpec`]: `const` parser compiling an `"in -> out"` mode
//!   expression into a compile-time plan, rejecting malformed syntax, duplicate input
//!   names, and unknown output names at compile time (or panicking at runtime).
//! - [`Einfold`] / [`einfold()`]: Applies a parsed [`EinfoldSpec`] to a [`Layout`],
//!   [`Tensor`], or hierarchical tuple (`Shape`, `Stride`, `Tiler`, coordinate, or scalar
//!   leaf), checking profile congruence at compile time.
//! - [`einfold!`](crate::einfold!): Convenience macro expanding
//!   `einfold!("in -> out", value)` to `einfold::<{ parse_einfold("in -> out") }>(value)`.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_adt_const_params, min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::{einfold, int_tuple};
//!
//! let layout = Layout::from_shape(int_tuple![(Int::<12>, Int::<4>), Int::<42>, Int::<5>, Int::<7>]);
//! assert_eq!(
//!     einfold!("ijkm -> imkj", layout).shape(),
//!     int_tuple![(Int::<12>, Int::<4>), Int::<7>, Int::<5>, Int::<42>]
//! );
//! assert_eq!(
//!     einfold!("(ab)cde -> c(ade)b", layout).shape(),
//!     int_tuple![Int::<42>, (Int::<12>, Int::<5>, Int::<7>), Int::<4>]
//! );
//! assert_eq!(
//!     einfold!("ij -> (ij)", layout).shape(),
//!     int_tuple![((Int::<12>, Int::<4>), Int::<42>), Int::<5>, Int::<7>]
//! );
//! assert_eq!(
//!     einfold!("ijk -> ik", layout).shape(),
//!     int_tuple![(Int::<12>, Int::<4>), Int::<5>, Int::<7>]
//! );
//! ```
//!
//! # Compile-Time Rejections
//!
//! Malformed expressions, duplicate input names, and unknown output names fail at compile time:
//!
//! ```compile_fail,E0080
//! # #![feature(min_adt_const_params, min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::{einfold, int_tuple};
//!
//! let l = Layout::from_shape(int_tuple![Int::<2>, Int::<3>]);
//! let _ = einfold!("ij -> ik", l);
//! ```
//!
//! Input profiles that do not coarsen the source fail at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_adt_const_params, min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::{einfold, int_tuple};
//!
//! let l = Layout::from_shape(int_tuple![Int::<2>, Int::<3>]);
//! let _ = einfold!("(ij)k -> ijk", l);
//! ```
//!
//! Unwrapping a scalar layout fails at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_adt_const_params, min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::einfold;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let l = Layout::new(Int::<4>, Int::<5>);
//! let _ = einfold!("(i) -> i", l);
//! ```
//!
//! A Tiler's [`Layout`] elements are atomic leaves, so descending into one fails at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_adt_const_params, min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::{einfold, int_tuple, tuple};
//!
//! let tiler = tuple![Layout::from_shape(int_tuple![Int::<2>, Int::<3>]), Layout::from_shape(Int::<5>)];
//! let _ = einfold!("(ab)c -> abc", tiler);
//! ```
//!
//! # Notes
//!
//! - Because each side denotes a profile, a lone name (`"i"`) binds the entire source rather
//!   than a single top-level mode: `"i -> (i)"` wraps a source one level deeper, and
//!   `"(i) -> i"` unwraps a rank-1 tuple source (or passes through trailing modes on rank > 1).
//! - [`Tensor`] inputs produce a new [`Tensor`] sharing the source accessor with zero copy.

use core::marker::ConstParamTy;
use core::ops::RangeFull;

use crate::basis::{BasisCombination, BasisTerm, BasisTuple};
use crate::f2::F2;
use crate::integer::{INCREASE, Int, int};
use crate::layout::{Layout, MakeLayout};
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tensor::Tensor;
use crate::tuple::{
    Get, GetOutput, Item, LeafProfile, MakeNodeFunctor, MakeTuple, Nil, NilProfile, Til, Tuple,
    TupleProfile,
};

/// Maximum number of tree tokens per side of an `einfold` subscript expression.
pub const MAX_EINFOLD_TOKENS: usize = 64;
/// Maximum number of distinct named input modes in an `einfold` expression.
pub const MAX_EINFOLD_LEAVES: usize = 32;
/// Maximum nesting depth of a named mode path in `einfold`.
pub const MAX_EINFOLD_DEPTH: usize = 16;

/// One token in the flattened parenthesized tree representation of a mode profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum ModeToken {
    /// Unused trailing slot.
    End,
    /// Opening delimiter of a mode group (`(`).
    Open,
    /// Closing delimiter of a mode group (`)`).
    Close,
    /// Leaf mode index (`0..num_leaves`).
    Leaf(int),
}

/// Tag classifying a [`ModeToken`] for compile-time trait dispatch.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum ModeTokenKind {
    End,
    Open,
    Close,
    Leaf,
}

const fn token_kind(tok: ModeToken) -> ModeTokenKind {
    match tok {
        ModeToken::End => ModeTokenKind::End,
        ModeToken::Open => ModeTokenKind::Open,
        ModeToken::Close => ModeTokenKind::Close,
        ModeToken::Leaf(_) => ModeTokenKind::Leaf,
    }
}

const fn token_leaf(tok: ModeToken) -> int {
    match tok {
        ModeToken::Leaf(k) => k,
        _ => 0,
    }
}

/// Compile-time parsed rearrangement plan produced by [`parse_einfold`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub struct EinfoldSpec {
    /// Input mode tree token stream (`[Leaf(0)]` or `[Open, ..., Close]`).
    pub in_tokens: [ModeToken; MAX_EINFOLD_TOKENS],
    /// Next-sibling token index for every node in `in_tokens`.
    pub in_next: [int; MAX_EINFOLD_TOKENS],
    /// `true` iff the parsed input profile is a tuple (`in_tokens[0] == ModeToken::Open`).
    pub in_is_tuple: bool,
    /// Number of top-level modes named on the input side (`0` when `in_is_tuple == false`).
    pub in_rank: int,
    /// Number of distinct input leaves.
    pub num_leaves: int,
    /// Per-leaf mode index paths within the input profile.
    pub leaf_paths: [[int; MAX_EINFOLD_DEPTH]; MAX_EINFOLD_LEAVES],
    /// Depth of each leaf path (`0` when a lone name binds the whole source).
    pub leaf_depths: [int; MAX_EINFOLD_LEAVES],
    /// Output mode tree token stream (`[Leaf(k)]` or `[Open, ..., Close]`).
    pub out_tokens: [ModeToken; MAX_EINFOLD_TOKENS],
    /// Next-sibling token index for every node in `out_tokens`.
    pub out_next: [int; MAX_EINFOLD_TOKENS],
    /// `true` iff the parsed output profile is a tuple (`out_tokens[0] == ModeToken::Open`).
    pub out_is_tuple: bool,
}

const fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

const fn parse_side(
    bytes: &[u8],
    start: usize,
    end: usize,
    is_input: bool,
    in_names: &mut [u8; MAX_EINFOLD_LEAVES],
    in_name_count: &mut usize,
) -> (
    [ModeToken; MAX_EINFOLD_TOKENS],
    [int; MAX_EINFOLD_TOKENS],
    bool,
    bool,
) {
    let mut raw = [ModeToken::End; MAX_EINFOLD_TOKENS];
    let mut raw_len: usize = 0;

    // Wrap top-level parse in an implicit outer Open .. Close pair.
    raw[0] = ModeToken::Open;
    raw_len += 1;

    let mut group_counts = [0usize; MAX_EINFOLD_DEPTH + 1];
    let mut depth: usize = 0;
    let mut total_leaves: usize = 0;
    let mut name_err = false;

    let mut i = start;
    while i < end {
        let b = bytes[i];
        i += 1;
        if is_ws(b) {
            continue;
        }
        if b == b'(' {
            assert!(
                depth < MAX_EINFOLD_DEPTH,
                "einfold: mode nesting exceeds MAX_EINFOLD_DEPTH"
            );
            assert!(
                raw_len < MAX_EINFOLD_TOKENS,
                "einfold: expression exceeds MAX_EINFOLD_TOKENS"
            );
            raw[raw_len] = ModeToken::Open;
            raw_len += 1;
            depth += 1;
            group_counts[depth] = 0;
        } else if b == b')' {
            assert!(depth > 0, "einfold: unmatched ')'");
            assert!(group_counts[depth] > 0, "einfold: empty mode group");
            assert!(
                raw_len < MAX_EINFOLD_TOKENS,
                "einfold: expression exceeds MAX_EINFOLD_TOKENS"
            );
            raw[raw_len] = ModeToken::Close;
            raw_len += 1;
            depth -= 1;
            group_counts[depth] += 1;
        } else if b == b',' {
            panic!("einfold: separators between modes are implied, so ',' is not allowed");
        } else if b.is_ascii_alphanumeric() {
            let leaf_idx = if is_input {
                let mut k = 0;
                let mut duplicate = false;
                while k < *in_name_count {
                    if in_names[k] == b {
                        duplicate = true;
                        break;
                    }
                    k += 1;
                }
                if duplicate {
                    name_err = true;
                    0
                } else {
                    assert!(
                        *in_name_count < MAX_EINFOLD_LEAVES,
                        "einfold: too many named modes"
                    );
                    let idx = *in_name_count;
                    in_names[idx] = b;
                    *in_name_count += 1;
                    idx
                }
            } else {
                let mut k = 0;
                let mut found: Option<usize> = None;
                while k < *in_name_count {
                    if in_names[k] == b {
                        found = Some(k);
                        break;
                    }
                    k += 1;
                }
                match found {
                    Some(idx) => idx,
                    None => {
                        name_err = true;
                        0
                    }
                }
            };
            assert!(
                raw_len < MAX_EINFOLD_TOKENS,
                "einfold: expression exceeds MAX_EINFOLD_TOKENS"
            );
            raw[raw_len] = ModeToken::Leaf(leaf_idx as int);
            raw_len += 1;
            total_leaves += 1;
            group_counts[depth] += 1;
        } else {
            panic!("einfold: invalid mode name");
        }
    }

    assert!(depth == 0, "einfold: unclosed '('");
    assert!(group_counts[0] > 0, "einfold: empty mode expression");
    assert!(
        raw_len < MAX_EINFOLD_TOKENS,
        "einfold: expression exceeds MAX_EINFOLD_TOKENS"
    );
    raw[raw_len] = ModeToken::Close;
    raw_len += 1;

    // Match PyCuTe `_parse_modes`: a lone leaf strips one enclosing tuple level.
    let mut tokens = [ModeToken::End; MAX_EINFOLD_TOKENS];
    let len = if total_leaves == 1 {
        let inner_len = raw_len - 2;
        let mut j = 0;
        while j < inner_len {
            tokens[j] = raw[j + 1];
            j += 1;
        }
        inner_len
    } else {
        let mut j = 0;
        while j < raw_len {
            tokens[j] = raw[j];
            j += 1;
        }
        raw_len
    };

    // Precompute next-sibling indices for O(1) subtree skipping.
    let mut next = [0 as int; MAX_EINFOLD_TOKENS];
    let mut open_stack = [0usize; MAX_EINFOLD_DEPTH + 2];
    let mut sp: usize = 0;
    let mut p: usize = 0;
    while p < len {
        match tokens[p] {
            ModeToken::Open => {
                open_stack[sp] = p;
                sp += 1;
            }
            ModeToken::Close => {
                sp -= 1;
                let open_pos = open_stack[sp];
                next[open_pos] = (p + 1) as int;
            }
            ModeToken::Leaf(_) => {
                next[p] = (p + 1) as int;
            }
            ModeToken::End => {}
        }
        p += 1;
    }

    let is_tuple = matches!(tokens[0], ModeToken::Open);
    (tokens, next, is_tuple, name_err)
}

/// Parses an `"in_modes -> out_modes"` mode expression into a compile-time [`EinfoldSpec`].
///
/// Panics (causing a compile-time error in `const` contexts or a runtime panic otherwise)
/// if `subscripts` does not contain exactly one `"->"`, contains malformed syntax, repeats
/// an input name, or references an unknown output name.
pub const fn parse_einfold(subscripts: &str) -> EinfoldSpec {
    let bytes = subscripts.as_bytes();
    let mut arrow_pos: Option<usize> = None;
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'-' && bytes[i + 1] == b'>' {
            assert!(arrow_pos.is_none(), "einfold: expected exactly one '->'");
            arrow_pos = Some(i);
            i += 2;
        } else {
            i += 1;
        }
    }
    let arrow = match arrow_pos {
        Some(pos) => pos,
        None => panic!("einfold: expected exactly one '->'"),
    };

    let mut in_names = [0u8; MAX_EINFOLD_LEAVES];
    let mut in_name_count: usize = 0;

    let (in_tokens, in_next, in_is_tuple, in_repeated) =
        parse_side(bytes, 0, arrow, true, &mut in_names, &mut in_name_count);
    let (out_tokens, out_next, out_is_tuple, out_unknown) = parse_side(
        bytes,
        arrow + 2,
        bytes.len(),
        false,
        &mut in_names,
        &mut in_name_count,
    );
    assert!(!in_repeated, "einfold: input contains repeated names");
    assert!(!out_unknown, "einfold: output contains unknown names");

    // Extract top-level input rank and each leaf's index path.
    let mut leaf_paths = [[0 as int; MAX_EINFOLD_DEPTH]; MAX_EINFOLD_LEAVES];
    let mut leaf_depths = [0 as int; MAX_EINFOLD_LEAVES];
    let mut in_rank: int = 0;

    if !in_is_tuple {
        // Lone input name binds the entire source at depth 0.
        leaf_depths[0] = 0;
    } else {
        let mut path = [0 as int; MAX_EINFOLD_DEPTH];
        let mut depth: usize = 0;
        let mut p: usize = 1; // Inside outermost Open
        loop {
            match in_tokens[p] {
                ModeToken::Open => {
                    depth += 1;
                    path[depth] = 0;
                    p += 1;
                }
                ModeToken::Close => {
                    if depth == 0 {
                        in_rank = path[0];
                        break;
                    }
                    depth -= 1;
                    path[depth] += 1;
                    p += 1;
                }
                ModeToken::Leaf(k) => {
                    let ku = k as usize;
                    let d = depth + 1;
                    leaf_depths[ku] = d as int;
                    let mut j = 0;
                    while j < d {
                        leaf_paths[ku][j] = path[j];
                        j += 1;
                    }
                    path[depth] += 1;
                    p += 1;
                }
                ModeToken::End => break,
            }
        }
    }

    EinfoldSpec {
        in_tokens,
        in_next,
        in_is_tuple,
        in_rank,
        num_leaves: in_name_count as int,
        leaf_paths,
        leaf_depths,
        out_tokens,
        out_next,
        out_is_tuple,
    }
}

// ------------------------------------------------------------------
// Compile-time const projection helpers over EinfoldSpec
// ------------------------------------------------------------------

#[doc(hidden)]
pub const IN_KIND<const SPEC: EinfoldSpec, const POS: int>: ModeTokenKind =
    token_kind(SPEC.in_tokens[POS as usize]);
#[doc(hidden)]
pub const IN_NEXT<const SPEC: EinfoldSpec, const POS: int>: int = SPEC.in_next[POS as usize];
#[doc(hidden)]
pub const IN_IS_TUPLE<const SPEC: EinfoldSpec>: bool = SPEC.in_is_tuple;
#[doc(hidden)]
pub const IN_RANK<const SPEC: EinfoldSpec>: int = SPEC.in_rank;

#[doc(hidden)]
pub const OUT_KIND<const SPEC: EinfoldSpec, const POS: int>: ModeTokenKind =
    token_kind(SPEC.out_tokens[POS as usize]);
#[doc(hidden)]
pub const OUT_LEAF<const SPEC: EinfoldSpec, const POS: int>: int =
    token_leaf(SPEC.out_tokens[POS as usize]);
#[doc(hidden)]
pub const OUT_NEXT<const SPEC: EinfoldSpec, const POS: int>: int = SPEC.out_next[POS as usize];
#[doc(hidden)]
pub const OUT_IS_TUPLE<const SPEC: EinfoldSpec>: bool = SPEC.out_is_tuple;

#[doc(hidden)]
pub const LEAF_AT_END<const SPEC: EinfoldSpec, const LEAF: int, const DEPTH: int>: bool =
    DEPTH >= SPEC.leaf_depths[LEAF as usize];
#[doc(hidden)]
pub const LEAF_PATH_STEP<const SPEC: EinfoldSpec, const LEAF: int, const DEPTH: int>: int =
    SPEC.leaf_paths[LEAF as usize][DEPTH as usize];

#[doc(hidden)]
pub const HAS_PASSTHROUGH<const SPEC: EinfoldSpec, const RANK: int>: bool =
    SPEC.in_is_tuple && (RANK > SPEC.in_rank);
#[doc(hidden)]
pub const MODES_DONE<const IDX: int, const RANK: int>: bool = IDX >= RANK;

// ------------------------------------------------------------------
// Compile-time input profile congruence validation
// ------------------------------------------------------------------

/// Validates that a single input node at `POS` (with tag `KIND`) coarsens profile `Self`.
#[doc(hidden)]
pub const trait CheckInputNode<const SPEC: EinfoldSpec, const POS: int, const KIND: ModeTokenKind> {}

const impl<P, const SPEC: EinfoldSpec, const POS: int>
    CheckInputNode<SPEC, POS, { ModeTokenKind::Leaf }> for P
{
}

const impl<H, T, const SPEC: EinfoldSpec, const POS: int>
    CheckInputNode<SPEC, POS, { ModeTokenKind::Open }> for TupleProfile<H, T>
where
    TupleProfile<H, T>:
        CheckInputGroup<SPEC, { INCREASE::<POS> }, { IN_KIND::<SPEC, { INCREASE::<POS> }> }>,
{
}

/// Validates a nested input group (exact length match until `Close`) against profile `Self`.
#[doc(hidden)]
pub const trait CheckInputGroup<const SPEC: EinfoldSpec, const POS: int, const KIND: ModeTokenKind>
{
}

const impl<const SPEC: EinfoldSpec, const POS: int>
    CheckInputGroup<SPEC, POS, { ModeTokenKind::Close }> for NilProfile
{
}

const impl<H, T, const SPEC: EinfoldSpec, const POS: int>
    CheckInputGroup<SPEC, POS, { ModeTokenKind::Leaf }> for TupleProfile<H, T>
where
    H: CheckInputNode<SPEC, POS, { ModeTokenKind::Leaf }>,
    T: CheckInputGroup<
            SPEC,
            { IN_NEXT::<SPEC, POS> },
            { IN_KIND::<SPEC, { IN_NEXT::<SPEC, POS> }> },
        >,
{
}

const impl<H, T, const SPEC: EinfoldSpec, const POS: int>
    CheckInputGroup<SPEC, POS, { ModeTokenKind::Open }> for TupleProfile<H, T>
where
    H: CheckInputNode<SPEC, POS, { ModeTokenKind::Open }>,
    T: CheckInputGroup<
            SPEC,
            { IN_NEXT::<SPEC, POS> },
            { IN_KIND::<SPEC, { IN_NEXT::<SPEC, POS> }> },
        >,
{
}

/// Validates top-level input modes (permitting unnamed trailing passthrough modes at `Close`).
#[doc(hidden)]
pub const trait CheckInputTop<const SPEC: EinfoldSpec, const POS: int, const KIND: ModeTokenKind> {}

const impl<const SPEC: EinfoldSpec, const POS: int>
    CheckInputTop<SPEC, POS, { ModeTokenKind::Close }> for NilProfile
{
}

const impl<H, T, const SPEC: EinfoldSpec, const POS: int>
    CheckInputTop<SPEC, POS, { ModeTokenKind::Close }> for TupleProfile<H, T>
{
}

const impl<H, T, const SPEC: EinfoldSpec, const POS: int>
    CheckInputTop<SPEC, POS, { ModeTokenKind::Leaf }> for TupleProfile<H, T>
where
    H: CheckInputNode<SPEC, POS, { ModeTokenKind::Leaf }>,
    T: CheckInputTop<SPEC, { IN_NEXT::<SPEC, POS> }, { IN_KIND::<SPEC, { IN_NEXT::<SPEC, POS> }> }>,
{
}

const impl<H, T, const SPEC: EinfoldSpec, const POS: int>
    CheckInputTop<SPEC, POS, { ModeTokenKind::Open }> for TupleProfile<H, T>
where
    H: CheckInputNode<SPEC, POS, { ModeTokenKind::Open }>,
    T: CheckInputTop<SPEC, { IN_NEXT::<SPEC, POS> }, { IN_KIND::<SPEC, { IN_NEXT::<SPEC, POS> }> }>,
{
}

/// Root input profile validator: lone name (`IS_TUPLE == false`) matches any profile;
/// tuple input (`IS_TUPLE == true`) requires a `TupleProfile` whose prefix matches `in_tokens`.
#[doc(hidden)]
pub const trait CheckEinfoldInput<const SPEC: EinfoldSpec, const IS_TUPLE: bool> {}

const impl<P, const SPEC: EinfoldSpec> CheckEinfoldInput<SPEC, false> for P {}

const impl<H, T, const SPEC: EinfoldSpec> CheckEinfoldInput<SPEC, true> for TupleProfile<H, T> where
    TupleProfile<H, T>: CheckInputTop<SPEC, 1, { IN_KIND::<SPEC, 1> }>
{
}

// ------------------------------------------------------------------
// Path extraction, passthrough collection, and output tree evaluation
// ------------------------------------------------------------------

/// Extracts the sub-mode at `SPEC.leaf_paths[LEAF][DEPTH..]` via recursive [`Get`].
#[doc(hidden)]
pub const trait ExtractLeafPath<
    const SPEC: EinfoldSpec,
    const LEAF: int,
    const DEPTH: int,
    const DONE: bool,
>: Copy
{
    type Output: Copy;
    fn extract_leaf_path(&self) -> Self::Output;
}

const impl<T: Copy, const SPEC: EinfoldSpec, const LEAF: int, const DEPTH: int>
    ExtractLeafPath<SPEC, LEAF, DEPTH, true> for T
{
    type Output = T;

    fn extract_leaf_path(&self) -> T {
        *self
    }
}

const impl<T, const SPEC: EinfoldSpec, const LEAF: int, const DEPTH: int>
    ExtractLeafPath<SPEC, LEAF, DEPTH, false> for T
where
    T: Copy + [const] Get<{ LEAF_PATH_STEP::<SPEC, LEAF, DEPTH> }>,
    GetOutput<T, { LEAF_PATH_STEP::<SPEC, LEAF, DEPTH> }>: [const] ExtractLeafPath<
            SPEC,
            LEAF,
            { INCREASE::<DEPTH> },
            { LEAF_AT_END::<SPEC, LEAF, { INCREASE::<DEPTH> }> },
        >,
{
    type Output = <GetOutput<T, { LEAF_PATH_STEP::<SPEC, LEAF, DEPTH> }> as ExtractLeafPath<
        SPEC,
        LEAF,
        { INCREASE::<DEPTH> },
        { LEAF_AT_END::<SPEC, LEAF, { INCREASE::<DEPTH> }> },
    >>::Output;

    fn extract_leaf_path(&self) -> Self::Output {
        let next = Get::<{ LEAF_PATH_STEP::<SPEC, LEAF, DEPTH> }>::get(self);
        next.extract_leaf_path()
    }
}

/// Collects unnamed top-level passthrough modes `IDX..RANK` into a tuple chain.
#[doc(hidden)]
pub const trait CollectModesFrom<const IDX: int, const RANK: int, const DONE: bool>:
    Copy
{
    type Output: [const] Tuple;
    fn collect_modes_from(&self) -> Self::Output;
}

const impl<T: Copy, const IDX: int, const RANK: int> CollectModesFrom<IDX, RANK, true> for T {
    type Output = Nil;

    fn collect_modes_from(&self) -> Nil {
        Nil
    }
}

const impl<T, const IDX: int, const RANK: int> CollectModesFrom<IDX, RANK, false> for T
where
    T: Copy
        + [const] Get<IDX>
        + [const] CollectModesFrom<
            { INCREASE::<IDX> },
            RANK,
            { MODES_DONE::<{ INCREASE::<IDX> }, RANK> },
        >,
{
    type Output = Til<
        GetOutput<T, IDX>,
        <T as CollectModesFrom<
            { INCREASE::<IDX> },
            RANK,
            { MODES_DONE::<{ INCREASE::<IDX> }, RANK> },
        >>::Output,
    >;

    fn collect_modes_from(&self) -> Self::Output {
        Til(Get::<IDX>::get(self), self.collect_modes_from())
    }
}

/// Evaluates a sibling sequence in `SPEC.out_tokens` starting at `POS` until `Close`,
/// appending `Tail` at the end and rebuilding nested groups with `Make`.
#[doc(hidden)]
pub const trait EvalOutList<
    const SPEC: EinfoldSpec,
    const POS: int,
    const KIND: ModeTokenKind,
    Make,
    Tail: Copy,
>: Copy
{
    type Output: Copy;
    fn eval_out_list(&self, make: &Make, tail: Tail) -> Self::Output;
}

const impl<T: Copy, const SPEC: EinfoldSpec, const POS: int, Make, Tail: Copy>
    EvalOutList<SPEC, POS, { ModeTokenKind::Close }, Make, Tail> for T
{
    type Output = Tail;

    fn eval_out_list(&self, _: &Make, tail: Tail) -> Tail {
        tail
    }
}

const impl<T: Copy, const SPEC: EinfoldSpec, const POS: int, Make, Tail: Copy>
    EvalOutList<SPEC, POS, { ModeTokenKind::End }, Make, Tail> for T
{
    type Output = Tail;

    fn eval_out_list(&self, _: &Make, tail: Tail) -> Tail {
        tail
    }
}

const impl<T, const SPEC: EinfoldSpec, const POS: int, Make, Tail: Copy>
    EvalOutList<SPEC, POS, { ModeTokenKind::Leaf }, Make, Tail> for T
where
    T: Copy
        + [const] ExtractLeafPath<
            SPEC,
            { OUT_LEAF::<SPEC, POS> },
            0,
            { LEAF_AT_END::<SPEC, { OUT_LEAF::<SPEC, POS> }, 0> },
        > + [const] EvalOutList<
            SPEC,
            { OUT_NEXT::<SPEC, POS> },
            { OUT_KIND::<SPEC, { OUT_NEXT::<SPEC, POS> }> },
            Make,
            Tail,
        >,
{
    type Output = Til<
        <T as ExtractLeafPath<
            SPEC,
            { OUT_LEAF::<SPEC, POS> },
            0,
            { LEAF_AT_END::<SPEC, { OUT_LEAF::<SPEC, POS> }, 0> },
        >>::Output,
        <T as EvalOutList<
            SPEC,
            { OUT_NEXT::<SPEC, POS> },
            { OUT_KIND::<SPEC, { OUT_NEXT::<SPEC, POS> }> },
            Make,
            Tail,
        >>::Output,
    >;

    fn eval_out_list(&self, make: &Make, tail: Tail) -> Self::Output {
        let head = <T as ExtractLeafPath<
            SPEC,
            { OUT_LEAF::<SPEC, POS> },
            0,
            { LEAF_AT_END::<SPEC, { OUT_LEAF::<SPEC, POS> }, 0> },
        >>::extract_leaf_path(self);
        let rest = <T as EvalOutList<
            SPEC,
            { OUT_NEXT::<SPEC, POS> },
            { OUT_KIND::<SPEC, { OUT_NEXT::<SPEC, POS> }> },
            Make,
            Tail,
        >>::eval_out_list(self, make, tail);
        Til(head, rest)
    }
}

const impl<T, const SPEC: EinfoldSpec, const POS: int, Make, Tail: Copy>
    EvalOutList<SPEC, POS, { ModeTokenKind::Open }, Make, Tail> for T
where
    T: Copy
        + [const] EvalOutList<
            SPEC,
            { INCREASE::<POS> },
            { OUT_KIND::<SPEC, { INCREASE::<POS> }> },
            Make,
            Nil,
        > + [const] EvalOutList<
            SPEC,
            { OUT_NEXT::<SPEC, POS> },
            { OUT_KIND::<SPEC, { OUT_NEXT::<SPEC, POS> }> },
            Make,
            Tail,
        >,
    Make: [const] MakeNodeFunctor<
        <T as EvalOutList<
            SPEC,
            { INCREASE::<POS> },
            { OUT_KIND::<SPEC, { INCREASE::<POS> }> },
            Make,
            Nil,
        >>::Output,
    >,
    Make::Output: Copy,
{
    type Output = Til<
        Make::Output,
        <T as EvalOutList<
            SPEC,
            { OUT_NEXT::<SPEC, POS> },
            { OUT_KIND::<SPEC, { OUT_NEXT::<SPEC, POS> }> },
            Make,
            Tail,
        >>::Output,
    >;

    fn eval_out_list(&self, make: &Make, tail: Tail) -> Self::Output {
        let children = <T as EvalOutList<
            SPEC,
            { INCREASE::<POS> },
            { OUT_KIND::<SPEC, { INCREASE::<POS> }> },
            Make,
            Nil,
        >>::eval_out_list(self, make, Nil);
        let head = make.make(children);
        let rest = <T as EvalOutList<
            SPEC,
            { OUT_NEXT::<SPEC, POS> },
            { OUT_KIND::<SPEC, { OUT_NEXT::<SPEC, POS> }> },
            Make,
            Tail,
        >>::eval_out_list(self, make, tail);
        Til(head, rest)
    }
}

/// Top-level output builder dispatching on `(OUT_TUPLE, HAS_PASS)`.
#[doc(hidden)]
pub const trait EvalEinfoldRoot<
    const SPEC: EinfoldSpec,
    const RANK: int,
    const OUT_TUPLE: bool,
    const HAS_PASS: bool,
    Make,
>: Copy
{
    type Output: Copy;
    fn eval_einfold_root(&self, make: &Make) -> Self::Output;
}

// Single output leaf, no passthrough -> return leaf directly (unwrapped).
const impl<T, const SPEC: EinfoldSpec, const RANK: int, Make>
    EvalEinfoldRoot<SPEC, RANK, false, false, Make> for T
where
    T: Copy
        + [const] ExtractLeafPath<
            SPEC,
            { OUT_LEAF::<SPEC, 0> },
            0,
            { LEAF_AT_END::<SPEC, { OUT_LEAF::<SPEC, 0> }, 0> },
        >,
{
    type Output = <T as ExtractLeafPath<
        SPEC,
        { OUT_LEAF::<SPEC, 0> },
        0,
        { LEAF_AT_END::<SPEC, { OUT_LEAF::<SPEC, 0> }, 0> },
    >>::Output;

    fn eval_einfold_root(&self, _: &Make) -> Self::Output {
        self.extract_leaf_path()
    }
}

// Single output leaf with passthrough -> wrap output leaf and append passthrough modes.
const impl<T, const SPEC: EinfoldSpec, const RANK: int, Make>
    EvalEinfoldRoot<SPEC, RANK, false, true, Make> for T
where
    T: Copy
        + [const] ExtractLeafPath<
            SPEC,
            { OUT_LEAF::<SPEC, 0> },
            0,
            { LEAF_AT_END::<SPEC, { OUT_LEAF::<SPEC, 0> }, 0> },
        > + [const] CollectModesFrom<
            { IN_RANK::<SPEC> },
            RANK,
            { MODES_DONE::<{ IN_RANK::<SPEC> }, RANK> },
        >,
    Make: [const] MakeNodeFunctor<
        Til<
            <T as ExtractLeafPath<
                SPEC,
                { OUT_LEAF::<SPEC, 0> },
                0,
                { LEAF_AT_END::<SPEC, { OUT_LEAF::<SPEC, 0> }, 0> },
            >>::Output,
            <T as CollectModesFrom<
                { IN_RANK::<SPEC> },
                RANK,
                { MODES_DONE::<{ IN_RANK::<SPEC> }, RANK> },
            >>::Output,
        >,
    >,
    Make::Output: Copy,
{
    type Output = Make::Output;

    fn eval_einfold_root(&self, make: &Make) -> Self::Output {
        let head = self.extract_leaf_path();
        let passthrough = self.collect_modes_from();
        make.make(Til(head, passthrough))
    }
}

// Tuple output, no passthrough.
const impl<T, const SPEC: EinfoldSpec, const RANK: int, Make>
    EvalEinfoldRoot<SPEC, RANK, true, false, Make> for T
where
    T: Copy + [const] EvalOutList<SPEC, 1, { OUT_KIND::<SPEC, 1> }, Make, Nil>,
    Make: [const] MakeNodeFunctor<
        <T as EvalOutList<SPEC, 1, { OUT_KIND::<SPEC, 1> }, Make, Nil>>::Output,
    >,
    Make::Output: Copy,
{
    type Output = Make::Output;

    fn eval_einfold_root(&self, make: &Make) -> Self::Output {
        make.make(self.eval_out_list(make, Nil))
    }
}

// Tuple output with passthrough suffix appended to outermost mode tuple.
const impl<T, const SPEC: EinfoldSpec, const RANK: int, Make>
    EvalEinfoldRoot<SPEC, RANK, true, true, Make> for T
where
    T: Copy
        + [const] CollectModesFrom<
            { IN_RANK::<SPEC> },
            RANK,
            { MODES_DONE::<{ IN_RANK::<SPEC> }, RANK> },
        > + [const] EvalOutList<
            SPEC,
            1,
            { OUT_KIND::<SPEC, 1> },
            Make,
            <T as CollectModesFrom<
                { IN_RANK::<SPEC> },
                RANK,
                { MODES_DONE::<{ IN_RANK::<SPEC> }, RANK> },
            >>::Output,
        >,
    Make: [const] MakeNodeFunctor<
        <T as EvalOutList<
            SPEC,
            1,
            { OUT_KIND::<SPEC, 1> },
            Make,
            <T as CollectModesFrom<
                { IN_RANK::<SPEC> },
                RANK,
                { MODES_DONE::<{ IN_RANK::<SPEC> }, RANK> },
            >>::Output,
        >>::Output,
    >,
    Make::Output: Copy,
{
    type Output = Make::Output;

    fn eval_einfold_root(&self, make: &Make) -> Self::Output {
        let passthrough = self.collect_modes_from();
        make.make(self.eval_out_list(make, passthrough))
    }
}

// ------------------------------------------------------------------
// Public Einfold trait, free function, and einfold! macro
// ------------------------------------------------------------------

/// Rearranges the modes of `self` according to compile-time [`EinfoldSpec`] `SPEC`.
///
/// Implemented for [`Tensor`], [`Layout`], hierarchical tuples ([`Til`], [`Nil`]), and
/// built-in scalar leaves (`int`, [`Int<N>`], [`BasisTerm`], [`BasisCombination`], [`F2`],
/// [`RangeFull`]).
pub const trait Einfold<const SPEC: EinfoldSpec>: Copy {
    /// Rearranged output type (preserving [`Tensor`], [`Layout`], or tuple/leaf kind).
    type Output: Copy;

    /// Rearranges `self` according to `SPEC`.
    fn einfold(self) -> Self::Output;
}

const impl<A, S, D, const SPEC: EinfoldSpec> Einfold<SPEC> for Tensor<A, Layout<S, D>>
where
    A: Copy,
    S: Shape,
    D: Stride<S>,
    Layout<S, D>: [const] Einfold<SPEC>,
{
    type Output = Tensor<A, <Layout<S, D> as Einfold<SPEC>>::Output>;

    fn einfold(self) -> Self::Output {
        Tensor {
            accessor: self.accessor,
            layout: self.layout.einfold(),
        }
    }
}

const impl<S, D, const SPEC: EinfoldSpec> Einfold<SPEC> for Layout<S, D>
where
    S: Shape,
    D: Stride<S>,
    S::Profile: CheckEinfoldInput<SPEC, { IN_IS_TUPLE::<SPEC> }>,
    Self: [const] EvalEinfoldRoot<
            SPEC,
            { S::RANK },
            { OUT_IS_TUPLE::<SPEC> },
            { HAS_PASSTHROUGH::<SPEC, { S::RANK }> },
            MakeLayout,
        >,
{
    type Output = <Self as EvalEinfoldRoot<
        SPEC,
        { S::RANK },
        { OUT_IS_TUPLE::<SPEC> },
        { HAS_PASSTHROUGH::<SPEC, { S::RANK }> },
        MakeLayout,
    >>::Output;

    fn einfold(self) -> Self::Output {
        self.eval_einfold_root(&MakeLayout)
    }
}

const ITEM_RANK<T: Item>: int = T::RANK;

const impl<H, T, const SPEC: EinfoldSpec> Einfold<SPEC> for Til<H, T>
where
    H: [const] Item,
    T: [const] Tuple,
    TupleProfile<H::Profile, T::Profile>: CheckEinfoldInput<SPEC, { IN_IS_TUPLE::<SPEC> }>,
    Self: [const] EvalEinfoldRoot<
            SPEC,
            { ITEM_RANK::<Self> },
            { OUT_IS_TUPLE::<SPEC> },
            { HAS_PASSTHROUGH::<SPEC, { ITEM_RANK::<Self> }> },
            MakeTuple,
        >,
{
    type Output = <Self as EvalEinfoldRoot<
        SPEC,
        { ITEM_RANK::<Self> },
        { OUT_IS_TUPLE::<SPEC> },
        { HAS_PASSTHROUGH::<SPEC, { ITEM_RANK::<Self> }> },
        MakeTuple,
    >>::Output;

    fn einfold(self) -> Self::Output {
        self.eval_einfold_root(&MakeTuple)
    }
}

const impl<const SPEC: EinfoldSpec> Einfold<SPEC> for Nil
where
    NilProfile: CheckEinfoldInput<SPEC, { IN_IS_TUPLE::<SPEC> }>,
    Self: [const] EvalEinfoldRoot<
            SPEC,
            0,
            { OUT_IS_TUPLE::<SPEC> },
            { HAS_PASSTHROUGH::<SPEC, 0> },
            MakeTuple,
        >,
{
    type Output = <Self as EvalEinfoldRoot<
        SPEC,
        0,
        { OUT_IS_TUPLE::<SPEC> },
        { HAS_PASSTHROUGH::<SPEC, 0> },
        MakeTuple,
    >>::Output;

    fn einfold(self) -> Self::Output {
        self.eval_einfold_root(&MakeTuple)
    }
}

macro_rules! impl_einfold_leaf {
    ([$($g:tt)*] $T:ty $(where $($w:tt)*)?) => {
        const impl<$($g)* const SPEC: EinfoldSpec> Einfold<SPEC> for $T
        where
            $($($w)*,)?
            LeafProfile: CheckEinfoldInput<SPEC, { IN_IS_TUPLE::<SPEC> }>,
            Self: [const] EvalEinfoldRoot<
                    SPEC,
                    1,
                    { OUT_IS_TUPLE::<SPEC> },
                    { HAS_PASSTHROUGH::<SPEC, 1> },
                    MakeTuple,
                >,
        {
            type Output = <Self as EvalEinfoldRoot<
                SPEC,
                1,
                { OUT_IS_TUPLE::<SPEC> },
                { HAS_PASSTHROUGH::<SPEC, 1> },
                MakeTuple,
            >>::Output;

            fn einfold(self) -> Self::Output {
                self.eval_einfold_root(&MakeTuple)
            }
        }
    };
}

impl_einfold_leaf!([] int);
impl_einfold_leaf!([const N: int,] Int<N>);
impl_einfold_leaf!([V: Copy, P: Copy,] BasisTerm<V, P>);
impl_einfold_leaf!([T: BasisTuple,] BasisCombination<T>);
impl_einfold_leaf!([V: Copy,] F2<V>);
impl_einfold_leaf!([] RangeFull);

/// Rearranges the modes of `value` according to compile-time [`EinfoldSpec`] `SPEC`.
pub const fn einfold<const SPEC: EinfoldSpec, T>(value: T) -> T::Output
where
    T: [const] Einfold<SPEC>,
{
    value.einfold()
}

/// Rearranges the modes of a [`Layout`](crate::layout::Layout),
/// [`Tensor`](crate::tensor::Tensor), or hierarchical tuple using an
/// `"in_modes -> out_modes"` subscript string parsed at compile time.
///
/// # Examples
///
/// ```
/// # #![feature(min_adt_const_params, min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
/// use noodle_cute::{einfold, int_tuple};
///
/// let a = Layout::new(int_tuple![Int::<2>, Int::<3>], int_tuple![Int::<1>, Int::<2>]);
/// assert_eq!(
///     einfold!("ij -> ji", a),
///     Layout::new(int_tuple![Int::<3>, Int::<2>], int_tuple![Int::<2>, Int::<1>])
/// );
/// ```
#[macro_export]
macro_rules! einfold {
    ($subscripts:expr, $value:expr $(,)?) => {{
        const __SPEC: $crate::algorithm::einfold::EinfoldSpec =
            $crate::algorithm::einfold::parse_einfold($subscripts);
        $crate::algorithm::einfold::einfold::<__SPEC, _>($value)
    }};
}
