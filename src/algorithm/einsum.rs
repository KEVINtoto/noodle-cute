//! Binary tensor contractions (`einsum`) lowered to batched [`gemm`].
//!
//! Evaluates explicit binary contractions `"a_modes,b_modes->c_modes"` by classifying each
//! subscript label at compile time into one of the four batched-GEMM index roles
//! (Whitepaper §2.2, §2.6.2):
//!
//! - **row (`M`)**: appears in `A` and `C` (not `B`)
//! - **column (`N`)**: appears in `B` and `C` (not `A`)
//! - **reduction (`K`)**: appears in `A` and `B` (not `C`)
//! - **batch (`L`)**: appears in `A`, `B`, and `C`
//!
//! Each operand is then folded — as a zero-copy view sharing its original accessor — into
//! the canonical rank-3 `(M, K, L)`, `(N, K, L)`, and `(M, N, L)` layouts consumed by
//! [`reference::gemm`](super::reference::gemm). Results accumulate in place into `C`
//! (`C += A * B`).
//!
//! # Main Interfaces
//!
//! - [`parse_einsum`] / [`EinsumSpec`] / [`OperandFold`] / [`FoldGroup`]: `const` parser
//!   and label classifier producing the compile-time fold plan for all three operands.
//! - [`fold_operand`] / [`FoldOperand`]: Zero-copy rank-3 folding of one tensor according
//!   to an [`OperandFold`].
//! - [`einsum()`] / [`Einsum`]: Validates ranks and per-label extent agreement, folds
//!   `(A, B, C)`, and dispatches to [`reference::gemm`](super::reference::gemm).
//! - [`einsum!`](crate::einsum!): Convenience macro expanding
//!   `einsum!("a,b->c", a, b, &mut c)` to
//!   `einsum::<{ parse_einsum("a,b->c") }>(a, b, &mut c)`.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_adt_const_params, min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::tensor::make_tensor;
//! use noodle_cute::integer::Int;
//! use noodle_cute::{einsum, int_tuple};
//!
//! let a_buf = [1i32, 2, 3, 4, 5, 6];
//! let b_buf = [1i32, 2, 3, 4, 5, 6, 7, 8];
//! let mut c_buf = [0i32; 12];
//! let a = make_tensor(&a_buf, int_tuple![Int::<3>, Int::<2>]);
//! let b = make_tensor(&b_buf, int_tuple![Int::<4>, Int::<2>]);
//! let mut c = make_tensor(&mut c_buf, int_tuple![Int::<3>, Int::<4>]);
//!
//! einsum!("mk,nk->mn", a, b, &mut c);
//! assert_eq!(c(0i32, 0i32), a(0i32, 0i32) * b(0i32, 0i32) + a(0i32, 1i32) * b(0i32, 1i32));
//! ```
//!
//! # Compile-Time Rejections
//!
//! Invalid equations (missing `"->"`, wrong input count, repeated label within an operand,
//! or a label appearing in only one operand) fail at compile time:
//!
//! ```compile_fail,E0080
//! # #![feature(min_adt_const_params, min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::tensor::make_tensor;
//! use noodle_cute::{einsum, int_tuple};
//!
//! let a_buf = [0i32; 12];
//! let b_buf = [0i32; 8];
//! let mut c_buf = [0i32; 6];
//! let a = make_tensor(&a_buf, int_tuple![Int::<2>, Int::<3>, Int::<2>]);
//! let b = make_tensor(&b_buf, int_tuple![Int::<4>, Int::<2>]);
//! let mut c = make_tensor(&mut c_buf, int_tuple![Int::<2>, Int::<3>]);
//! einsum!("mkx,nk->mn", a, b, &mut c);
//! ```
//!
//! Rank mismatches and statically inconsistent label extents fail at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_adt_const_params, min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::tensor::make_tensor;
//! use noodle_cute::{einsum, int_tuple};
//!
//! let a_buf = [0i32; 15];
//! let b_buf = [0i32; 24];
//! let mut c_buf = [0i32; 12];
//! let a = make_tensor(&a_buf, int_tuple![Int::<3>, Int::<5>]);
//! let b = make_tensor(&b_buf, int_tuple![Int::<4>, Int::<6>]);
//! let mut c = make_tensor(&mut c_buf, int_tuple![Int::<3>, Int::<4>]);
//! einsum!("mk,nk->mn", a, b, &mut c);
//! ```

use core::marker::ConstParamTy;

use crate::integer::{INCREASE, Int, int};
use crate::layout::{Concatenation, Layout, LayoutBase, make_layout};
use crate::shape::Shape;
use crate::stride::Stride;
use crate::tensor::Tensor;
use crate::tuple::{Get, GetOutput, Nil, Til, Tuple};

use super::reference::{Gemm, STATIC_SIZE_COMPAT, ValidSizeMatch, gemm};

/// Maximum rank of any single `einsum` operand.
pub const MAX_EINSUM_RANK: usize = 16;
/// Maximum total number of distinct labels in an `einsum` equation.
pub const MAX_EINSUM_LABELS: usize = 32;

/// One folded multi-mode group storing the source mode indices concatenated into it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub struct FoldGroup {
    /// Number of source modes in this group (`0` produces a size-1 `Layout(1, 0)` mode).
    pub len: int,
    /// Zero-based mode indices into the source operand (`indices[0..len]`).
    pub indices: [int; MAX_EINSUM_RANK],
}

impl FoldGroup {
    /// Constructs an empty [`FoldGroup`] (folding into `Layout(1, 0)`).
    pub const fn empty() -> Self {
        Self {
            len: 0,
            indices: [0; MAX_EINSUM_RANK],
        }
    }
}

/// Rank-3 fold specification for one operand of [`einsum`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub struct OperandFold {
    /// Required rank of the source operand.
    pub expected_rank: int,
    /// Mode group folded into output mode 0.
    pub g0: FoldGroup,
    /// Mode group folded into output mode 1.
    pub g1: FoldGroup,
    /// Mode group folded into output mode 2.
    pub g2: FoldGroup,
}

/// Compile-time parsed and classified binary `einsum` plan produced by [`parse_einsum`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub struct EinsumSpec {
    /// Fold specification for `A -> (M, K, L)`.
    pub a_fold: OperandFold,
    /// Fold specification for `B -> (N, K, L)`.
    pub b_fold: OperandFold,
    /// Fold specification for `C -> (M, N, L)`.
    pub c_fold: OperandFold,
    /// Total number of distinct labels in the equation.
    pub num_labels: int,
    /// Per-label position in `A`, or `-1` if absent.
    pub label_pos_a: [int; MAX_EINSUM_LABELS],
    /// Per-label position in `B`, or `-1` if absent.
    pub label_pos_b: [int; MAX_EINSUM_LABELS],
    /// Per-label position in `C`, or `-1` if absent.
    pub label_pos_c: [int; MAX_EINSUM_LABELS],
    /// Number of row (`M`) labels.
    pub row_len: int,
    /// Number of column (`N`) labels.
    pub col_len: int,
    /// Number of reduction (`K`) labels.
    pub red_len: int,
    /// Number of batch (`L`) labels.
    pub bat_len: int,
    /// Row (`M`) label characters in first-appearance order.
    pub row_labels: [char; MAX_EINSUM_RANK],
    /// Column (`N`) label characters in first-appearance order.
    pub col_labels: [char; MAX_EINSUM_RANK],
    /// Reduction (`K`) label characters in first-appearance order.
    pub red_labels: [char; MAX_EINSUM_RANK],
    /// Batch (`L`) label characters in first-appearance order.
    pub bat_labels: [char; MAX_EINSUM_RANK],
}

impl EinsumSpec {
    /// Returns the classified row (`M`) labels in first-appearance order.
    pub const fn row_labels(&self) -> &[char] {
        self.row_labels.split_at(self.row_len as usize).0
    }

    /// Returns the classified column (`N`) labels in first-appearance order.
    pub const fn col_labels(&self) -> &[char] {
        self.col_labels.split_at(self.col_len as usize).0
    }

    /// Returns the classified reduction (`K`) labels in first-appearance order.
    pub const fn red_labels(&self) -> &[char] {
        self.red_labels.split_at(self.red_len as usize).0
    }

    /// Returns the classified batch (`L`) labels in first-appearance order.
    pub const fn bat_labels(&self) -> &[char] {
        self.bat_labels.split_at(self.bat_len as usize).0
    }
}

const fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

const fn collect_operand(
    bytes: &[u8],
    start: usize,
    end: usize,
    dup_msg: &'static str,
) -> ([u8; MAX_EINSUM_RANK], usize) {
    let mut out = [0u8; MAX_EINSUM_RANK];
    let mut len = 0usize;
    let mut i = start;
    while i < end {
        let b = bytes[i];
        i += 1;
        if is_ws(b) {
            continue;
        }
        assert!(b.is_ascii_alphanumeric(), "einsum: invalid mode label");
        let mut j = 0;
        while j < len {
            if out[j] == b {
                panic!("{}", dup_msg);
            }
            j += 1;
        }
        assert!(
            len < MAX_EINSUM_RANK,
            "einsum: operand rank exceeds MAX_EINSUM_RANK"
        );
        out[len] = b;
        len += 1;
    }
    (out, len)
}

const fn find_in(arr: &[u8; MAX_EINSUM_RANK], len: usize, ch: u8) -> int {
    let mut i = 0;
    while i < len {
        if arr[i] == ch {
            return i as int;
        }
        i += 1;
    }
    -1
}

const fn push_group(group: &mut FoldGroup, idx: int) {
    assert!(
        (group.len as usize) < MAX_EINSUM_RANK,
        "einsum: group size exceeds MAX_EINSUM_RANK"
    );
    group.indices[group.len as usize] = idx;
    group.len += 1;
}

/// Parses an explicit binary `"a_modes,b_modes->c_modes"` contraction equation and
/// classifies every label into `(row, col, red, bat)` roles at compile time.
pub const fn parse_einsum(subscripts: &str) -> EinsumSpec {
    let bytes = subscripts.as_bytes();
    let mut arrow_pos: Option<usize> = None;
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'-' && bytes[i + 1] == b'>' {
            assert!(arrow_pos.is_none(), "einsum: expected exactly one '->'");
            arrow_pos = Some(i);
            i += 2;
        } else {
            i += 1;
        }
    }
    let arrow = match arrow_pos {
        Some(pos) => pos,
        None => panic!("einsum: expected exactly one '->'"),
    };

    let mut comma_pos: Option<usize> = None;
    let mut j = 0;
    while j < arrow {
        if bytes[j] == b',' {
            assert!(
                comma_pos.is_none(),
                "einsum: expected exactly two ','-separated inputs"
            );
            comma_pos = Some(j);
        }
        j += 1;
    }
    let comma = match comma_pos {
        Some(pos) => pos,
        None => panic!("einsum: expected exactly two ','-separated inputs"),
    };

    let (a_modes, a_len) =
        collect_operand(bytes, 0, comma, "einsum: repeated label within operand A");
    let (b_modes, b_len) = collect_operand(
        bytes,
        comma + 1,
        arrow,
        "einsum: repeated label within operand B",
    );
    let (c_modes, c_len) = collect_operand(
        bytes,
        arrow + 2,
        bytes.len(),
        "einsum: repeated label within operand C",
    );

    // Collect unique labels in first-appearance order across A + B + C.
    let mut order = [0u8; MAX_EINSUM_LABELS];
    let mut order_len = 0usize;

    let mut k = 0;
    while k < a_len {
        let ch = a_modes[k];
        assert!(
            order_len < MAX_EINSUM_LABELS,
            "einsum: too many distinct labels"
        );
        order[order_len] = ch;
        order_len += 1;
        k += 1;
    }
    k = 0;
    while k < b_len {
        let ch = b_modes[k];
        let mut seen = false;
        let mut m = 0;
        while m < order_len {
            if order[m] == ch {
                seen = true;
                break;
            }
            m += 1;
        }
        if !seen {
            assert!(
                order_len < MAX_EINSUM_LABELS,
                "einsum: too many distinct labels"
            );
            order[order_len] = ch;
            order_len += 1;
        }
        k += 1;
    }
    k = 0;
    while k < c_len {
        let ch = c_modes[k];
        let mut seen = false;
        let mut m = 0;
        while m < order_len {
            if order[m] == ch {
                seen = true;
                break;
            }
            m += 1;
        }
        if !seen {
            assert!(
                order_len < MAX_EINSUM_LABELS,
                "einsum: too many distinct labels"
            );
            order[order_len] = ch;
            order_len += 1;
        }
        k += 1;
    }

    let mut a_row = FoldGroup::empty();
    let mut a_red = FoldGroup::empty();
    let mut a_bat = FoldGroup::empty();
    let mut b_col = FoldGroup::empty();
    let mut b_red = FoldGroup::empty();
    let mut b_bat = FoldGroup::empty();
    let mut c_row = FoldGroup::empty();
    let mut c_col = FoldGroup::empty();
    let mut c_bat = FoldGroup::empty();

    let mut label_pos_a = [-1 as int; MAX_EINSUM_LABELS];
    let mut label_pos_b = [-1 as int; MAX_EINSUM_LABELS];
    let mut label_pos_c = [-1 as int; MAX_EINSUM_LABELS];

    let mut row_labels = ['\0'; MAX_EINSUM_RANK];
    let mut col_labels = ['\0'; MAX_EINSUM_RANK];
    let mut red_labels = ['\0'; MAX_EINSUM_RANK];
    let mut bat_labels = ['\0'; MAX_EINSUM_RANK];
    let mut row_len = 0usize;
    let mut col_len = 0usize;
    let mut red_len = 0usize;
    let mut bat_len = 0usize;

    let mut idx = 0;
    while idx < order_len {
        let ch = order[idx];
        let pa = find_in(&a_modes, a_len, ch);
        let pb = find_in(&b_modes, b_len, ch);
        let pc = find_in(&c_modes, c_len, ch);
        label_pos_a[idx] = pa;
        label_pos_b[idx] = pb;
        label_pos_c[idx] = pc;

        match (pa >= 0, pb >= 0, pc >= 0) {
            (true, false, true) => {
                // row M
                push_group(&mut a_row, pa);
                push_group(&mut c_row, pc);
                row_labels[row_len] = ch as char;
                row_len += 1;
            }
            (false, true, true) => {
                // col N
                push_group(&mut b_col, pb);
                push_group(&mut c_col, pc);
                col_labels[col_len] = ch as char;
                col_len += 1;
            }
            (true, true, false) => {
                // red K
                push_group(&mut a_red, pa);
                push_group(&mut b_red, pb);
                red_labels[red_len] = ch as char;
                red_len += 1;
            }
            (true, true, true) => {
                // bat L
                push_group(&mut a_bat, pa);
                push_group(&mut b_bat, pb);
                push_group(&mut c_bat, pc);
                bat_labels[bat_len] = ch as char;
                bat_len += 1;
            }
            _ => {
                panic!(
                    "einsum: label appears in only one operand; each label must appear in at least two of A, B, C"
                );
            }
        }
        idx += 1;
    }

    EinsumSpec {
        a_fold: OperandFold {
            expected_rank: a_len as int,
            g0: a_row,
            g1: a_red,
            g2: a_bat,
        },
        b_fold: OperandFold {
            expected_rank: b_len as int,
            g0: b_col,
            g1: b_red,
            g2: b_bat,
        },
        c_fold: OperandFold {
            expected_rank: c_len as int,
            g0: c_row,
            g1: c_col,
            g2: c_bat,
        },
        num_labels: order_len as int,
        label_pos_a,
        label_pos_b,
        label_pos_c,
        row_len: row_len as int,
        col_len: col_len as int,
        red_len: red_len as int,
        bat_len: bat_len as int,
        row_labels,
        col_labels,
        red_labels,
        bat_labels,
    }
}

// ------------------------------------------------------------------
// Zero-copy rank-3 operand folding
// ------------------------------------------------------------------

/// Kind of a [`FoldGroup`] based on its mode count (`0`, `1`, or `>= 2`).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum FoldGroupKind {
    Empty,
    Single,
    Multi,
}

const fn fold_group_kind(g: FoldGroup) -> FoldGroupKind {
    if g.len == 0 {
        FoldGroupKind::Empty
    } else if g.len == 1 {
        FoldGroupKind::Single
    } else {
        FoldGroupKind::Multi
    }
}

#[doc(hidden)]
pub const GROUP_KIND<const G: FoldGroup>: FoldGroupKind = fold_group_kind(G);
#[doc(hidden)]
pub const GROUP_INDEX<const G: FoldGroup, const POS: int>: int = G.indices[POS as usize];
#[doc(hidden)]
pub const GROUP_DONE<const G: FoldGroup, const POS: int>: bool = POS >= G.len;

/// Collects sub-layouts at `G.indices[POS..G.len]` into a tuple chain for [`make_layout`].
#[doc(hidden)]
pub const trait CollectGroupModes<const G: FoldGroup, const POS: int, const DONE: bool>:
    Copy
{
    type Output: [const] Tuple;
    fn collect_group_modes(&self) -> Self::Output;
}

const impl<L: Copy, const G: FoldGroup, const POS: int> CollectGroupModes<G, POS, true> for L {
    type Output = Nil;

    fn collect_group_modes(&self) -> Nil {
        Nil
    }
}

const impl<L, const G: FoldGroup, const POS: int> CollectGroupModes<G, POS, false> for L
where
    L: Copy
        + [const] Get<{ GROUP_INDEX::<G, POS> }>
        + [const] CollectGroupModes<G, { INCREASE::<POS> }, { GROUP_DONE::<G, { INCREASE::<POS> }> }>,
{
    type Output = Til<
        GetOutput<L, { GROUP_INDEX::<G, POS> }>,
        <L as CollectGroupModes<
            G,
            { INCREASE::<POS> },
            { GROUP_DONE::<G, { INCREASE::<POS> }> },
        >>::Output,
    >;

    fn collect_group_modes(&self) -> Self::Output {
        Til(
            Get::<{ GROUP_INDEX::<G, POS> }>::get(self),
            self.collect_group_modes(),
        )
    }
}

/// Folds one [`FoldGroup`] of a [`Layout`] into a single mode sub-layout.
#[doc(hidden)]
pub const trait EvalFoldGroup<const G: FoldGroup, const KIND: FoldGroupKind>: Copy {
    type Shape: Shape;
    type Stride: Stride<Self::Shape>;
    fn eval_fold_group(&self) -> Layout<Self::Shape, Self::Stride>;
}

const impl<L: Copy, const G: FoldGroup> EvalFoldGroup<G, { FoldGroupKind::Empty }> for L {
    type Shape = Int<1>;
    type Stride = Int<0>;

    fn eval_fold_group(&self) -> Layout<Int<1>, Int<0>> {
        Layout::new(Int::<1>, Int::<0>)
    }
}

const impl<L, const G: FoldGroup, S, D> EvalFoldGroup<G, { FoldGroupKind::Single }> for L
where
    L: Copy + [const] Get<{ GROUP_INDEX::<G, 0> }, Output = Layout<S, D>>,
    S: Shape,
    D: Stride<S>,
{
    type Shape = S;
    type Stride = D;

    fn eval_fold_group(&self) -> Layout<S, D> {
        Get::<{ GROUP_INDEX::<G, 0> }>::get(self)
    }
}

const impl<L, const G: FoldGroup> EvalFoldGroup<G, { FoldGroupKind::Multi }> for L
where
    L: Copy + [const] CollectGroupModes<G, 0, { GROUP_DONE::<G, 0> }>,
    <L as CollectGroupModes<G, 0, { GROUP_DONE::<G, 0> }>>::Output: [const] Concatenation,
{
    type Shape =
        <<L as CollectGroupModes<G, 0, { GROUP_DONE::<G, 0> }>>::Output as Concatenation>::Shape;
    type Stride =
        <<L as CollectGroupModes<G, 0, { GROUP_DONE::<G, 0> }>>::Output as Concatenation>::Stride;

    fn eval_fold_group(&self) -> Layout<Self::Shape, Self::Stride> {
        make_layout(self.collect_group_modes())
    }
}

/// Compile-time guard verifying an operand's rank matches the label count in `einsum`.
#[doc(hidden)]
pub const trait ValidRankMatch<const MATCH: bool> {}
const impl ValidRankMatch<true> for () {}

#[doc(hidden)]
pub const RANK_MATCHES<S: Shape, const EXPECTED: int>: bool = S::RANK == EXPECTED;
#[doc(hidden)]
pub const FOLD_G0<const F: OperandFold>: FoldGroup = F.g0;
#[doc(hidden)]
pub const FOLD_G1<const F: OperandFold>: FoldGroup = F.g1;
#[doc(hidden)]
pub const FOLD_G2<const F: OperandFold>: FoldGroup = F.g2;
#[doc(hidden)]
pub const FOLD_RANK<const F: OperandFold>: int = F.expected_rank;

type FoldGroupShape<L, const G: FoldGroup> = <L as EvalFoldGroup<G, { GROUP_KIND::<G> }>>::Shape;
type FoldGroupStride<L, const G: FoldGroup> = <L as EvalFoldGroup<G, { GROUP_KIND::<G> }>>::Stride;

/// Folds a [`Tensor`] into its canonical rank-3 view per [`OperandFold`] `FOLD`, sharing
/// the original accessor with zero data copy.
pub const trait FoldOperand<const FOLD: OperandFold>: Copy {
    /// Folded rank-3 tensor view type.
    type Output: Copy;

    /// Returns the zero-copy rank-3 folded view of `self`.
    fn fold_operand(self) -> Self::Output;
}

const impl<A, S, D, const FOLD: OperandFold> FoldOperand<FOLD> for Tensor<A, Layout<S, D>>
where
    A: Copy,
    S: Shape,
    D: Stride<S>,
    (): ValidRankMatch<{ RANK_MATCHES::<S, { FOLD_RANK::<FOLD> }> }>,
    Layout<S, D>: [const] EvalFoldGroup<{ FOLD_G0::<FOLD> }, { GROUP_KIND::<{ FOLD_G0::<FOLD> }> }>
        + [const] EvalFoldGroup<{ FOLD_G1::<FOLD> }, { GROUP_KIND::<{ FOLD_G1::<FOLD> }> }>
        + [const] EvalFoldGroup<{ FOLD_G2::<FOLD> }, { GROUP_KIND::<{ FOLD_G2::<FOLD> }> }>,
{
    type Output = Tensor<
        A,
        Layout<
            Til<
                FoldGroupShape<Layout<S, D>, { FOLD_G0::<FOLD> }>,
                Til<
                    FoldGroupShape<Layout<S, D>, { FOLD_G1::<FOLD> }>,
                    Til<FoldGroupShape<Layout<S, D>, { FOLD_G2::<FOLD> }>, Nil>,
                >,
            >,
            Til<
                FoldGroupStride<Layout<S, D>, { FOLD_G0::<FOLD> }>,
                Til<
                    FoldGroupStride<Layout<S, D>, { FOLD_G1::<FOLD> }>,
                    Til<FoldGroupStride<Layout<S, D>, { FOLD_G2::<FOLD> }>, Nil>,
                >,
            >,
        >,
    >;

    fn fold_operand(self) -> Self::Output {
        let m0 = <Layout<S, D> as EvalFoldGroup<
            { FOLD_G0::<FOLD> },
            { GROUP_KIND::<{ FOLD_G0::<FOLD> }> },
        >>::eval_fold_group(&self.layout);
        let m1 = <Layout<S, D> as EvalFoldGroup<
            { FOLD_G1::<FOLD> },
            { GROUP_KIND::<{ FOLD_G1::<FOLD> }> },
        >>::eval_fold_group(&self.layout);
        let m2 = <Layout<S, D> as EvalFoldGroup<
            { FOLD_G2::<FOLD> },
            { GROUP_KIND::<{ FOLD_G2::<FOLD> }> },
        >>::eval_fold_group(&self.layout);
        Tensor {
            accessor: self.accessor,
            layout: Layout::new(
                Til(m0.shape, Til(m1.shape, Til(m2.shape, Nil))),
                Til(m0.stride, Til(m1.stride, Til(m2.stride, Nil))),
            ),
        }
    }
}

/// Folds `tensor` into a zero-copy rank-3 view according to `FOLD`.
pub const fn fold_operand<const FOLD: OperandFold, T>(tensor: T) -> T::Output
where
    T: [const] FoldOperand<FOLD>,
{
    tensor.fold_operand()
}

// ------------------------------------------------------------------
// Per-label extent agreement validation across (A, B, C)
// ------------------------------------------------------------------

/// Role of one label across operands `(A, B, C)` for per-label extent validation.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum LabelRole {
    Done,
    Row,
    Col,
    Red,
    Bat,
}

const fn label_role(spec: EinsumSpec, idx: int) -> LabelRole {
    if idx >= spec.num_labels {
        return LabelRole::Done;
    }
    let u = idx as usize;
    match (
        spec.label_pos_a[u] >= 0,
        spec.label_pos_b[u] >= 0,
        spec.label_pos_c[u] >= 0,
    ) {
        (true, false, true) => LabelRole::Row,
        (false, true, true) => LabelRole::Col,
        (true, true, false) => LabelRole::Red,
        (true, true, true) => LabelRole::Bat,
        _ => LabelRole::Done,
    }
}

#[doc(hidden)]
pub const LABEL_ROLE<const SPEC: EinsumSpec, const IDX: int>: LabelRole = label_role(SPEC, IDX);
#[doc(hidden)]
pub const LABEL_POS_A<const SPEC: EinsumSpec, const IDX: int>: int = SPEC.label_pos_a[IDX as usize];
#[doc(hidden)]
pub const LABEL_POS_B<const SPEC: EinsumSpec, const IDX: int>: int = SPEC.label_pos_b[IDX as usize];
#[doc(hidden)]
pub const LABEL_POS_C<const SPEC: EinsumSpec, const IDX: int>: int = SPEC.label_pos_c[IDX as usize];

/// Validates that every label `IDX..SPEC.num_labels` has agreeing extents across every
/// operand in which it appears (compile-time error when static, runtime panic when dynamic).
#[doc(hidden)]
pub const trait CheckEinsumLabels<
    const SPEC: EinsumSpec,
    const IDX: int,
    const ROLE: LabelRole,
    LB,
    LC,
>
{
    fn check_einsum_labels(&self, b: &LB, c: &LC);
}

const impl<LA, LB, LC, const SPEC: EinsumSpec, const IDX: int>
    CheckEinsumLabels<SPEC, IDX, { LabelRole::Done }, LB, LC> for LA
{
    fn check_einsum_labels(&self, _: &LB, _: &LC) {}
}

const impl<LA, LB, LC, const SPEC: EinsumSpec, const IDX: int>
    CheckEinsumLabels<SPEC, IDX, { LabelRole::Row }, LB, LC> for LA
where
    LA: [const] Get<{ LABEL_POS_A::<SPEC, IDX> }>
        + [const] CheckEinsumLabels<
            SPEC,
            { INCREASE::<IDX> },
            { LABEL_ROLE::<SPEC, { INCREASE::<IDX> }> },
            LB,
            LC,
        >,
    LC: [const] Get<{ LABEL_POS_C::<SPEC, IDX> }>,
    GetOutput<LA, { LABEL_POS_A::<SPEC, IDX> }>: [const] LayoutBase,
    GetOutput<LC, { LABEL_POS_C::<SPEC, IDX> }>: [const] LayoutBase,
    <GetOutput<LA, { LABEL_POS_A::<SPEC, IDX> }> as LayoutBase>::Shape: [const] Shape,
    <GetOutput<LC, { LABEL_POS_C::<SPEC, IDX> }> as LayoutBase>::Shape: [const] Shape,
    (): ValidSizeMatch<
        {
            STATIC_SIZE_COMPAT::<
                <GetOutput<LA, { LABEL_POS_A::<SPEC, IDX> }> as LayoutBase>::Shape,
                <GetOutput<LC, { LABEL_POS_C::<SPEC, IDX> }> as LayoutBase>::Shape,
            >
        },
    >,
{
    fn check_einsum_labels(&self, b: &LB, c: &LC) {
        let sa = Get::<{ LABEL_POS_A::<SPEC, IDX> }>::get(self)
            .shape()
            .size();
        let sc = Get::<{ LABEL_POS_C::<SPEC, IDX> }>::get(c).shape().size();
        assert!(sa == sc, "einsum: label has inconsistent extents");
        <Self as CheckEinsumLabels<
            SPEC,
            { INCREASE::<IDX> },
            { LABEL_ROLE::<SPEC, { INCREASE::<IDX> }> },
            LB,
            LC,
        >>::check_einsum_labels(self, b, c);
    }
}

const impl<LA, LB, LC, const SPEC: EinsumSpec, const IDX: int>
    CheckEinsumLabels<SPEC, IDX, { LabelRole::Col }, LB, LC> for LA
where
    LA: [const] CheckEinsumLabels<
            SPEC,
            { INCREASE::<IDX> },
            { LABEL_ROLE::<SPEC, { INCREASE::<IDX> }> },
            LB,
            LC,
        >,
    LB: [const] Get<{ LABEL_POS_B::<SPEC, IDX> }>,
    LC: [const] Get<{ LABEL_POS_C::<SPEC, IDX> }>,
    GetOutput<LB, { LABEL_POS_B::<SPEC, IDX> }>: [const] LayoutBase,
    GetOutput<LC, { LABEL_POS_C::<SPEC, IDX> }>: [const] LayoutBase,
    <GetOutput<LB, { LABEL_POS_B::<SPEC, IDX> }> as LayoutBase>::Shape: [const] Shape,
    <GetOutput<LC, { LABEL_POS_C::<SPEC, IDX> }> as LayoutBase>::Shape: [const] Shape,
    (): ValidSizeMatch<
        {
            STATIC_SIZE_COMPAT::<
                <GetOutput<LB, { LABEL_POS_B::<SPEC, IDX> }> as LayoutBase>::Shape,
                <GetOutput<LC, { LABEL_POS_C::<SPEC, IDX> }> as LayoutBase>::Shape,
            >
        },
    >,
{
    fn check_einsum_labels(&self, b: &LB, c: &LC) {
        let sb = Get::<{ LABEL_POS_B::<SPEC, IDX> }>::get(b).shape().size();
        let sc = Get::<{ LABEL_POS_C::<SPEC, IDX> }>::get(c).shape().size();
        assert!(sb == sc, "einsum: label has inconsistent extents");
        <Self as CheckEinsumLabels<
            SPEC,
            { INCREASE::<IDX> },
            { LABEL_ROLE::<SPEC, { INCREASE::<IDX> }> },
            LB,
            LC,
        >>::check_einsum_labels(self, b, c);
    }
}

const impl<LA, LB, LC, const SPEC: EinsumSpec, const IDX: int>
    CheckEinsumLabels<SPEC, IDX, { LabelRole::Red }, LB, LC> for LA
where
    LA: [const] Get<{ LABEL_POS_A::<SPEC, IDX> }>
        + [const] CheckEinsumLabels<
            SPEC,
            { INCREASE::<IDX> },
            { LABEL_ROLE::<SPEC, { INCREASE::<IDX> }> },
            LB,
            LC,
        >,
    LB: [const] Get<{ LABEL_POS_B::<SPEC, IDX> }>,
    GetOutput<LA, { LABEL_POS_A::<SPEC, IDX> }>: [const] LayoutBase,
    GetOutput<LB, { LABEL_POS_B::<SPEC, IDX> }>: [const] LayoutBase,
    <GetOutput<LA, { LABEL_POS_A::<SPEC, IDX> }> as LayoutBase>::Shape: [const] Shape,
    <GetOutput<LB, { LABEL_POS_B::<SPEC, IDX> }> as LayoutBase>::Shape: [const] Shape,
    (): ValidSizeMatch<
        {
            STATIC_SIZE_COMPAT::<
                <GetOutput<LA, { LABEL_POS_A::<SPEC, IDX> }> as LayoutBase>::Shape,
                <GetOutput<LB, { LABEL_POS_B::<SPEC, IDX> }> as LayoutBase>::Shape,
            >
        },
    >,
{
    fn check_einsum_labels(&self, b: &LB, c: &LC) {
        let sa = Get::<{ LABEL_POS_A::<SPEC, IDX> }>::get(self)
            .shape()
            .size();
        let sb = Get::<{ LABEL_POS_B::<SPEC, IDX> }>::get(b).shape().size();
        assert!(sa == sb, "einsum: label has inconsistent extents");
        <Self as CheckEinsumLabels<
            SPEC,
            { INCREASE::<IDX> },
            { LABEL_ROLE::<SPEC, { INCREASE::<IDX> }> },
            LB,
            LC,
        >>::check_einsum_labels(self, b, c);
    }
}

const impl<LA, LB, LC, const SPEC: EinsumSpec, const IDX: int>
    CheckEinsumLabels<SPEC, IDX, { LabelRole::Bat }, LB, LC> for LA
where
    LA: [const] Get<{ LABEL_POS_A::<SPEC, IDX> }>
        + [const] CheckEinsumLabels<
            SPEC,
            { INCREASE::<IDX> },
            { LABEL_ROLE::<SPEC, { INCREASE::<IDX> }> },
            LB,
            LC,
        >,
    LB: [const] Get<{ LABEL_POS_B::<SPEC, IDX> }>,
    LC: [const] Get<{ LABEL_POS_C::<SPEC, IDX> }>,
    GetOutput<LA, { LABEL_POS_A::<SPEC, IDX> }>: [const] LayoutBase,
    GetOutput<LB, { LABEL_POS_B::<SPEC, IDX> }>: [const] LayoutBase,
    GetOutput<LC, { LABEL_POS_C::<SPEC, IDX> }>: [const] LayoutBase,
    <GetOutput<LA, { LABEL_POS_A::<SPEC, IDX> }> as LayoutBase>::Shape: [const] Shape,
    <GetOutput<LB, { LABEL_POS_B::<SPEC, IDX> }> as LayoutBase>::Shape: [const] Shape,
    <GetOutput<LC, { LABEL_POS_C::<SPEC, IDX> }> as LayoutBase>::Shape: [const] Shape,
    (): ValidSizeMatch<
            {
                STATIC_SIZE_COMPAT::<
                    <GetOutput<LA, { LABEL_POS_A::<SPEC, IDX> }> as LayoutBase>::Shape,
                    <GetOutput<LC, { LABEL_POS_C::<SPEC, IDX> }> as LayoutBase>::Shape,
                >
            },
        > + ValidSizeMatch<
            {
                STATIC_SIZE_COMPAT::<
                    <GetOutput<LB, { LABEL_POS_B::<SPEC, IDX> }> as LayoutBase>::Shape,
                    <GetOutput<LC, { LABEL_POS_C::<SPEC, IDX> }> as LayoutBase>::Shape,
                >
            },
        > + ValidSizeMatch<
            {
                STATIC_SIZE_COMPAT::<
                    <GetOutput<LA, { LABEL_POS_A::<SPEC, IDX> }> as LayoutBase>::Shape,
                    <GetOutput<LB, { LABEL_POS_B::<SPEC, IDX> }> as LayoutBase>::Shape,
                >
            },
        >,
{
    fn check_einsum_labels(&self, b: &LB, c: &LC) {
        let sa = Get::<{ LABEL_POS_A::<SPEC, IDX> }>::get(self)
            .shape()
            .size();
        let sb = Get::<{ LABEL_POS_B::<SPEC, IDX> }>::get(b).shape().size();
        let sc = Get::<{ LABEL_POS_C::<SPEC, IDX> }>::get(c).shape().size();
        assert!(
            sa == sc && sb == sc,
            "einsum: label has inconsistent extents"
        );
        <Self as CheckEinsumLabels<
            SPEC,
            { INCREASE::<IDX> },
            { LABEL_ROLE::<SPEC, { INCREASE::<IDX> }> },
            LB,
            LC,
        >>::check_einsum_labels(self, b, c);
    }
}

#[doc(hidden)]
pub const SPEC_A_FOLD<const SPEC: EinsumSpec>: OperandFold = SPEC.a_fold;
#[doc(hidden)]
pub const SPEC_B_FOLD<const SPEC: EinsumSpec>: OperandFold = SPEC.b_fold;
#[doc(hidden)]
pub const SPEC_C_FOLD<const SPEC: EinsumSpec>: OperandFold = SPEC.c_fold;

/// Binary tensor contraction trait backing [`einsum()`] and [`einsum!`](crate::einsum!).
pub const trait Einsum<const SPEC: EinsumSpec, B, C> {
    /// Accumulates the binary contraction specified by `SPEC` into `c` in place.
    fn einsum(self, b: B, c: &mut C);
}

const impl<const SPEC: EinsumSpec, AA, SA, DA, AB, SB, DB, AC, SC, DC>
    Einsum<SPEC, Tensor<AB, Layout<SB, DB>>, Tensor<AC, Layout<SC, DC>>>
    for Tensor<AA, Layout<SA, DA>>
where
    AA: Copy,
    AB: Copy,
    AC: Copy,
    SA: Shape,
    DA: Stride<SA>,
    SB: Shape,
    DB: Stride<SB>,
    SC: Shape,
    DC: Stride<SC>,
    Self: [const] FoldOperand<{ SPEC_A_FOLD::<SPEC> }>,
    Tensor<AB, Layout<SB, DB>>: [const] FoldOperand<{ SPEC_B_FOLD::<SPEC> }>,
    Tensor<AC, Layout<SC, DC>>: [const] FoldOperand<{ SPEC_C_FOLD::<SPEC> }>,
    Layout<SA, DA>: [const] CheckEinsumLabels<
            SPEC,
            0,
            { LABEL_ROLE::<SPEC, 0> },
            Layout<SB, DB>,
            Layout<SC, DC>,
        >,
    <Self as FoldOperand<{ SPEC_A_FOLD::<SPEC> }>>::Output: [const] Gemm<
            <Tensor<AB, Layout<SB, DB>> as FoldOperand<{ SPEC_B_FOLD::<SPEC> }>>::Output,
            <Tensor<AC, Layout<SC, DC>> as FoldOperand<{ SPEC_C_FOLD::<SPEC> }>>::Output,
        >,
{
    fn einsum(self, b: Tensor<AB, Layout<SB, DB>>, c: &mut Tensor<AC, Layout<SC, DC>>) {
        self.layout.check_einsum_labels(&b.layout, &c.layout);
        let a3 = fold_operand::<{ SPEC_A_FOLD::<SPEC> }, _>(self);
        let b3 = fold_operand::<{ SPEC_B_FOLD::<SPEC> }, _>(b);
        let mut c3 = fold_operand::<{ SPEC_C_FOLD::<SPEC> }, _>(*c);
        gemm(a3, b3, &mut c3);
    }
}

/// Evaluates an explicit binary tensor contraction according to `SPEC`, accumulating into `c`
/// (`C += A * B`) via zero-copy folding and [`reference::gemm`](super::reference::gemm).
pub const fn einsum<const SPEC: EinsumSpec, A, B, C>(a: A, b: B, c: &mut C)
where
    A: [const] Einsum<SPEC, B, C>,
{
    a.einsum(b, c);
}

/// Evaluates an explicit binary `"a_modes,b_modes->c_modes"` tensor contraction in place,
/// accumulating into `c` (`C += A * B`) by zero-copy folding into batched
/// [`gemm`](crate::algorithm::reference::gemm).
///
/// # Examples
///
/// ```
/// # #![feature(min_adt_const_params, min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::tensor::make_tensor;
/// use noodle_cute::{einsum, int_tuple};
///
/// let a_buf = [1i32, 2, 3];
/// let b_buf = [4i32, 5, 6];
/// let mut c_buf = [0i32; 1];
/// let a = make_tensor(&a_buf, Int::<3>);
/// let b = make_tensor(&b_buf, Int::<3>);
/// let mut c = make_tensor(&mut c_buf, noodle_cute::tuple::Nil);
/// einsum!("i,i->", a, b, &mut c);
/// assert_eq!(c_buf[0], 1 * 4 + 2 * 5 + 3 * 6);
/// ```
#[macro_export]
macro_rules! einsum {
    ($subscripts:expr, $a:expr, $b:expr, $c:expr $(,)?) => {{
        const __SPEC: $crate::algorithm::einsum::EinsumSpec =
            $crate::algorithm::einsum::parse_einsum($subscripts);
        $crate::algorithm::einsum::einsum::<__SPEC, _, _, _>($a, $b, $c)
    }};
}
