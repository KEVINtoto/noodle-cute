//! Coordinate-wise layout addition (`layout_add`).
//!
//! Given two [`Layout`]s $A$ and $B$ with equal domain size $|A| = |B|$, `layout_add` constructs
//! the canonical coalesced layout $R$ such that:
//!
//! $$|R| = |A| = |B|, \qquad R(i) = A(i) + B(i) \quad \forall i \in [0, |R|)$$
//!
//! # Algorithm
//!
//! For whole-layout addition (`P = LeafProfile`, matching PyCuTe's default `profile = 1`):
//!
//! 1. **Identity shortcut**: Adding [`Nil`] on either side returns the other operand unchanged,
//!    under any profile.
//! 2. **Canonicalize operands**: Reduce $A_{\text{co}} = \mathrm{coalesce}(A)$ and
//!    $B_{\text{co}} = \mathrm{coalesce}(B)$ so [`greatest_common_domain`](super::greatest_common_domain())
//!    sees the maximally merged shape factorizations exposed by each layout's strides.
//! 3. **Common domain**: Compute $G = \mathrm{greatest\_common\_domain}(A_{\text{co}}, B_{\text{co}})$
//!    with shape $S = (s_0, s_1, \dots)$ and stride $D = (d_0, d_1, \dots)$.
//! 4. **Refinement check**: Require $|G| = |A| = |B|$. Every index $i \in [0, |A|)$ then
//!    decomposes uniquely as $i = \sum_k c_k d_k$ ($0 \le c_k < s_k$), and both $A_{\text{co}}$
//!    and $B_{\text{co}}$ act linearly along each generator $d_k$:
//!    $$A(i) + B(i) = \sum_k c_k \cdot \bigl(A_{\text{co}}(d_k) + B_{\text{co}}(d_k)\bigr)$$
//! 5. **Leaf-wise stride evaluation & canonicalization**: Evaluate $d'_k = A_{\text{co}}(d_k) + B_{\text{co}}(d_k)$
//!    at each leaf of $G.\text{stride}$ and return $\mathrm{coalesce}(\mathrm{Layout}(G.\text{shape}, D'))$.
//!
//! For by-mode addition (`P = TupleProfile<PH, PT>`, or via [`layout_add_with_profile`]):
//! the profile must weakly coarsen both `shape(A)` and `shape(B)` (`weakly_congruent(P, shape)`),
//! and addition recurses mode-by-mode, preserving the outer mode structure.
//!
//! # Main Interfaces
//!
//! - [`LayoutAdd`] trait (`layout_add`)
//! - [`layout_add`] / [`Layout::layout_add`] (whole-layout addition)
//! - [`layout_add_by`] / [`Layout::layout_add_by`] (explicit type-level profile `P`)
//! - [`layout_add_with_profile`] / [`Layout::layout_add_with_profile`] (value-guided profile, e.g. `int_tuple![1, 1]`)
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::layout_add::{layout_add, layout_add_with_profile};
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::int_tuple;
//!
//! let a = Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<1>, Int::<4>]);
//! let b = Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<3>, Int::<1>]);
//!
//! // Whole-layout addition
//! assert_eq!(
//!     layout_add(a, b),
//!     Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<4>, Int::<5>])
//! );
//!
//! // Whole-layout vs by-mode addition on A + A
//! assert_eq!(layout_add(a, a), Layout::new(Int::<12>, Int::<2>));
//! assert_eq!(
//!     layout_add_with_profile(a, a, int_tuple![1, 1]),
//!     Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<2>, Int::<8>])
//! );
//! ```
//!
//! # Notes
//!
//! - Both layouts must have strictly positive extents, equal total domain sizes, and a common
//!   refinement of size `|A|`.
//! - Violations and arithmetic overflows fail at compile time for static operands and panic at
//!   runtime for dynamic operands.
//!
//! # Compile-Time Rejections
//!
//! Splitting a nonzero F2 stride at an interior non-binary radix introduces carries:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{algebra::layout_add, f2::f2, integer::Int, layout::Layout, int_tuple, tuple};
//! let a = Layout::new(Int::<9>, f2(Int::<1>));
//! let b = Layout::new(int_tuple![Int::<3>, Int::<3>], tuple![f2(Int::<2>), f2(Int::<4>)]);
//! let _ = layout_add(a, b);
//! ```
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{algebra::layout_add, f2::f2, integer::Int, layout::Layout, int_tuple, tuple};
//! let a = Layout::new(int_tuple![Int::<3>, Int::<3>], tuple![f2(Int::<2>), f2(Int::<4>)]);
//! let b = Layout::new(Int::<9>, f2(Int::<1>));
//! let _ = layout_add(a, b);
//! ```
//!
//! Size mismatch is rejected at compile time for static shapes:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::layout_add::layout_add;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let _ = layout_add(Layout::new(Int::<6>, Int::<1>), Layout::new(Int::<8>, Int::<1>));
//! ```
//!
//! Layouts without a common refinement of size `|A|` are rejected at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::layout_add::layout_add;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::int_tuple;
//!
//! let _ = layout_add(
//!     Layout::new(int_tuple![Int::<5>, Int::<3>], int_tuple![Int::<3>, Int::<1>]),
//!     Layout::new(int_tuple![Int::<3>, Int::<5>], int_tuple![Int::<5>, Int::<1>]),
//! );
//! ```
//!
//! Non-layout operands and incompatible stride algebras are rejected at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::layout_add::layout_add;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! let _ = layout_add(Layout::new(Int::<4>, Int::<1>), Int::<4>);
//! ```
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::layout_add::layout_add;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::e;
//!
//! let _ = layout_add(Layout::new(Int::<8>, e![0]), Layout::new(Int::<8>, Int::<1>));
//! ```
//!
//! By-mode profile rank/depth mismatches are rejected at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::layout_add::layout_add_with_profile;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::int_tuple;
//!
//! // Short profile (1,) is refused on a rank-3 layout
//! let a = Layout::new(
//!     int_tuple![Int::<4>, Int::<3>, Int::<2>],
//!     int_tuple![Int::<1>, Int::<4>, Int::<12>],
//! );
//! let _ = layout_add_with_profile(a, a, int_tuple![1]);
//! ```
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::layout_add::layout_add_with_profile;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::int_tuple;
//!
//! // Rank-2 profile (1, 1) is refused on a scalar layout
//! let s = Layout::new(Int::<12>, Int::<1>);
//! let _ = layout_add_with_profile(s, s, int_tuple![1, 1]);
//! ```
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::layout_add::layout_add_with_profile;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::int_tuple;
//!
//! // Nested profile ((1, 1), 1) does not coarsen flat (4, 3)
//! let a = Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<1>, Int::<4>]);
//! let _ = layout_add_with_profile(a, a, int_tuple![(1, 1), 1]);
//! ```

use crate::integer::{div, int, is_pow2, rem};
use crate::layout::Layout;
use crate::shape::{AssertPositiveShape, IndexToCoord, Shape, ShapeScalar};
use crate::stride::{Accumulate, ContainsF2, InnerProduct, Stride};
use crate::tuple::{
    FlatTuple, Flatten, Item, Leaf, LeafProfile, Nil, NilProfile, Til, Tuple, TupleProfile,
};

use super::coalesce::{Coalesce, DropTrailingUnit};
use super::greatest_common_domain::GreatestCommonDomain;

const fn static_sizes_compatible<SA: Shape, SB: Shape>() -> bool {
    if SA::IS_STATIC && SA::STATIC_SIZE.is_none() {
        return false;
    }
    if SB::IS_STATIC && SB::STATIC_SIZE.is_none() {
        return false;
    }
    match (SA::STATIC_SIZE, SB::STATIC_SIZE) {
        (Some(a), Some(b)) => a == b,
        _ => true,
    }
}

const STATIC_SIZES_COMPATIBLE<SA: Shape, SB: Shape>: bool = static_sizes_compatible::<SA, SB>();

/// Type-level guard asserting that two static shapes have equal non-overflowing sizes.
#[doc(hidden)]
pub const trait ValidLayoutAddSizes<const OK: bool> {}
const impl ValidLayoutAddSizes<true> for () {}

/// Canonicalizes a layout by flattening, running `Coalesce<true>`, and dropping a trailing
/// static unit mode when rank >= 2 (preserving standalone `1:d`, matching PyCuTe's `_coalesce`).
#[doc(hidden)]
pub const trait CanonicalCoalesce: Copy {
    type Shape: [const] Shape;
    type Stride: Stride<Self::Shape>;
    fn canonical_coalesce(self) -> Layout<Self::Shape, Self::Stride>;
}

const impl<S, D, CoalOut> CanonicalCoalesce for Layout<S, D>
where
    S: Shape + [const] Flatten,
    D: Stride<S> + [const] Flatten,
    S::Output: [const] AssertPositiveShape,
    Layout<S::Output, D::Output>: [const] Coalesce<true, Output = CoalOut>,
    CoalOut: [const] DropTrailingUnit,
    CoalOut::Shape: [const] Shape,
{
    type Shape = CoalOut::Shape;
    type Stride = CoalOut::Stride;

    fn canonical_coalesce(self) -> Layout<Self::Shape, Self::Stride> {
        let flat_s = self.shape.flatten();
        let flat_d = self.stride.flatten();
        flat_s.assert_positive_shape();
        let coal = Coalesce::<true>::coalesce(Layout {
            shape: flat_s,
            stride: flat_d,
        });
        coal.drop_trailing_unit()
    }
}

type CanonShape<S, D> = <Layout<S, D> as CanonicalCoalesce>::Shape;
type CanonStride<S, D> = <Layout<S, D> as CanonicalCoalesce>::Stride;
type CanonLayout<S, D> = Layout<CanonShape<S, D>, CanonStride<S, D>>;

/// Checks a common-domain generator against the radix of its original mode.
/// An interior split of a nonzero F2 mode must separate disjoint bit fields.
#[doc(hidden)]
pub const trait RefinementBoundary: Copy {
    const HAS_F2: bool;
    fn valid_boundary(layout: Option<Self>, weight: int) -> Option<bool>;
}

const impl<S: ShapeScalar, D: [const] ContainsF2 + Copy> RefinementBoundary for Layout<S, D> {
    const HAS_F2: bool = D::HAS_F2;
    fn valid_boundary(layout: Option<Self>, weight: int) -> Option<bool> {
        if weight <= 1 || is_pow2(weight) {
            return Some(true);
        }
        match D::active_f2(match layout {
            Some(l) => Some(l.stride),
            None => None,
        }) {
            Some(active) => Some(!active),
            None => None,
        }
    }
}

const impl RefinementBoundary for Layout<Nil, Nil> {
    const HAS_F2: bool = false;
    fn valid_boundary(_: Option<Self>, _: int) -> Option<bool> {
        Some(true)
    }
}

const impl<H: [const] Shape, T: Copy, D: Copy, R: Copy> RefinementBoundary
    for Layout<Til<H, T>, Til<D, R>>
where
    Layout<H, D>: [const] RefinementBoundary,
    Layout<T, R>: [const] RefinementBoundary,
{
    const HAS_F2: bool = Layout::<H, D>::HAS_F2 || Layout::<T, R>::HAS_F2;
    fn valid_boundary(layout: Option<Self>, weight: int) -> Option<bool> {
        if weight <= 1 {
            return Some(true);
        }
        let (extent, head, tail) = match layout {
            Some(l) => (
                Some(l.shape.0.size()),
                Some(Layout {
                    shape: l.shape.0,
                    stride: l.stride.0,
                }),
                Some(Layout {
                    shape: l.shape.1,
                    stride: l.stride.1,
                }),
            ),
            None => (H::STATIC_SIZE, None, None),
        };
        let Some(n) = extent else { return None };
        if n == 0 {
            return None;
        }
        if weight < n {
            Layout::<H, D>::valid_boundary(head, weight)
        } else if rem(weight, n) != 0 {
            Some(false)
        } else {
            Layout::<T, R>::valid_boundary(tail, div(weight, n))
        }
    }
}

const STATIC_BOUNDARY<L: const RefinementBoundary, W: ShapeScalar>: Option<bool> =
    if !L::HAS_F2 { Some(true) } else {
        match W::STATIC_VALUE {
            Some(w) => L::valid_boundary(None, w),
            None => None,
        }
    };

const VALID_BOUNDARY<L: const RefinementBoundary, W: ShapeScalar>: bool =
    !matches!(STATIC_BOUNDARY::<L, W>, Some(false));

/// Allows a generator whose radix is compatible with the original F2 mode.
#[doc(hidden)]
#[diagnostic::on_unimplemented(
    message = "layout_add refinement carries in a nonzero F2 mode",
    note = "splitting an F2 mode across a common domain requires power-of-two sub-mode radices"
)]
pub trait ValidRefinementBoundary<const VALID: bool> {}
impl ValidRefinementBoundary<true> for () {}

const fn check_boundary<L: const RefinementBoundary, W: [const] ShapeScalar>(layout: L, weight: W) {
    if STATIC_BOUNDARY::<L, W>.is_none() {
        assert!(
            matches!(L::valid_boundary(Some(layout), weight.value()), Some(true)),
            "layout_add refinement carries in an F2 mode"
        );
    }
}

/// Evaluates `A_co(d_k) + B_co(d_k)` at every leaf `d_k` of `G.stride`.
#[doc(hidden)]
pub const trait EvalSumStrides<SA, DA, SB, DB>: FlatTuple {
    type Output: FlatTuple;
    fn eval_sum_strides(self, a: Layout<SA, DA>, b: Layout<SB, DB>) -> Self::Output;
}

const impl<SA: Copy, DA: Copy, SB: Copy, DB: Copy> EvalSumStrides<SA, DA, SB, DB> for Nil {
    type Output = Nil;

    fn eval_sum_strides(self, _: Layout<SA, DA>, _: Layout<SB, DB>) -> Nil {
        Nil
    }
}

const impl<DH, DT, SA, DA, SB, DB, ValA, ValB> EvalSumStrides<SA, DA, SB, DB> for Til<DH, DT>
where
    Layout<SA, DA>: const RefinementBoundary,
    Layout<SB, DB>: const RefinementBoundary,
    (): ValidRefinementBoundary<{ VALID_BOUNDARY::<Layout<SA, DA>, DH> }>
        + ValidRefinementBoundary<{ VALID_BOUNDARY::<Layout<SB, DB>, DH> }>,
    DH: [const] ShapeScalar + [const] IndexToCoord<SA> + [const] IndexToCoord<SB>,
    DT: [const] EvalSumStrides<SA, DA, SB, DB>,
    SA: Shape,
    DA: Stride<SA>,
    SB: Shape,
    DB: Stride<SB>,
    <DH as IndexToCoord<SA>>::Output: [const] InnerProduct<DA, Output = ValA>,
    <DH as IndexToCoord<SB>>::Output: [const] InnerProduct<DB, Output = ValB>,
    ValA: [const] Accumulate<ValB>,
    ValB: Copy,
    <ValA as Accumulate<ValB>>::Output: Leaf,
{
    type Output = Til<<ValA as Accumulate<ValB>>::Output, DT::Output>;

    fn eval_sum_strides(self, a: Layout<SA, DA>, b: Layout<SB, DB>) -> Self::Output {
        check_boundary(a, self.0);
        check_boundary(b, self.0);
        let va = a.call(self.0);
        let vb = b.call(self.0);
        let sum = va.accumulate(vb);
        let tail = self.1.eval_sum_strides(a, b);
        Til(sum, tail)
    }
}

/// Coordinate-wise addition of two [`Layout`]s under structural profile `P`.
///
/// - `P = LeafProfile` (default): whole-layout addition (`R(i) == A(i) + B(i)` for `i in 0..|A|`).
/// - `P = TupleProfile<PH, PT>` / `NilProfile`: by-mode addition preserving the profile's hierarchy.
/// - [`Nil`] acts as the additive identity on either side under any profile.
pub const trait LayoutAdd<B, P = LeafProfile>: Copy {
    /// Resulting sum [`Layout`] (or [`Nil`] when both operands are [`Nil`]).
    type Output: Copy;

    /// Layout addition, panicking on dynamic precondition failure or arithmetic overflow.
    fn layout_add(self, b: B) -> Self::Output;
}

const impl<P> LayoutAdd<Nil, P> for Nil {
    type Output = Nil;

    fn layout_add(self, _: Nil) -> Nil {
        Nil
    }
}

const impl<SA: Shape, DA: Stride<SA>, P> LayoutAdd<Nil, P> for Layout<SA, DA> {
    type Output = Self;

    fn layout_add(self, _: Nil) -> Self {
        self
    }
}

const impl<SB: Shape, DB: Stride<SB>, P> LayoutAdd<Layout<SB, DB>, P> for Nil {
    type Output = Layout<SB, DB>;

    fn layout_add(self, b: Layout<SB, DB>) -> Self::Output {
        b
    }
}

type GcdShapeOf<SA, DA, SB, DB> =
    <CanonLayout<SA, DA> as GreatestCommonDomain<CanonLayout<SB, DB>>>::Shape;
type GcdStrideOf<SA, DA, SB, DB> =
    <CanonLayout<SA, DA> as GreatestCommonDomain<CanonLayout<SB, DB>>>::Stride;
type SumStrideOf<SA, DA, SB, DB> = <GcdStrideOf<SA, DA, SB, DB> as EvalSumStrides<
    CanonShape<SA, DA>,
    CanonStride<SA, DA>,
    CanonShape<SB, DB>,
    CanonStride<SB, DB>,
>>::Output;
type PreCoalesceSumLayout<SA, DA, SB, DB> =
    Layout<GcdShapeOf<SA, DA, SB, DB>, SumStrideOf<SA, DA, SB, DB>>;

const impl<SA, DA, SB, DB, FinalCoal> LayoutAdd<Layout<SB, DB>, LeafProfile> for Layout<SA, DA>
where
    SA: [const] Shape,
    DA: Stride<SA>,
    SB: [const] Shape,
    DB: Stride<SB>,
    (): ValidLayoutAddSizes<{ STATIC_SIZES_COMPATIBLE::<SA, SB> }>,
    Layout<SA, DA>: [const] CanonicalCoalesce,
    Layout<SB, DB>: [const] CanonicalCoalesce,
    CanonLayout<SA, DA>: [const] GreatestCommonDomain<CanonLayout<SB, DB>>,
    (): ValidLayoutAddSizes<{ STATIC_SIZES_COMPATIBLE::<GcdShapeOf<SA, DA, SB, DB>, SA> }>,
    GcdStrideOf<SA, DA, SB, DB>: [const] EvalSumStrides<
            CanonShape<SA, DA>,
            CanonStride<SA, DA>,
            CanonShape<SB, DB>,
            CanonStride<SB, DB>,
        >,
    SumStrideOf<SA, DA, SB, DB>: Stride<GcdShapeOf<SA, DA, SB, DB>>,
    PreCoalesceSumLayout<SA, DA, SB, DB>: [const] Coalesce<true, Output = FinalCoal>,
    FinalCoal: [const] DropTrailingUnit,
{
    type Output = Layout<FinalCoal::Shape, FinalCoal::Stride>;

    fn layout_add(self, b: Layout<SB, DB>) -> Self::Output {
        let size_a = self.shape.size();
        let size_b = b.shape.size();
        assert!(size_a == size_b, "layout_add requires equal domain sizes");
        let a_co = self.canonical_coalesce();
        let b_co = b.canonical_coalesce();
        let g = a_co.greatest_common_domain(b_co);
        let size_g = g.shape.size();
        assert!(
            size_g == size_a,
            "layout_add requires a common refinement of size |A|"
        );
        let result_stride = g.stride.eval_sum_strides(a_co, b_co);
        let coal = Coalesce::<true>::coalesce(Layout::new(g.shape, result_stride));
        coal.drop_trailing_unit()
    }
}

const impl LayoutAdd<Layout<Nil, Nil>, NilProfile> for Layout<Nil, Nil> {
    type Output = Layout<Nil, Nil>;

    fn layout_add(self, _: Layout<Nil, Nil>) -> Layout<Nil, Nil> {
        Layout::new(Nil, Nil)
    }
}

const impl<ASH, AST, ADH, ADT, BSH, BST, BDH, BDT, PH, PT, HS, HD, TS, TD>
    LayoutAdd<Layout<Til<BSH, BST>, Til<BDH, BDT>>, TupleProfile<PH, PT>>
    for Layout<Til<ASH, AST>, Til<ADH, ADT>>
where
    ASH: Shape,
    AST: Tuple + Shape,
    ADH: Stride<ASH>,
    ADT: Tuple + Stride<AST>,
    BSH: Shape,
    BST: Tuple + Shape,
    BDH: Stride<BSH>,
    BDT: Tuple + Stride<BST>,
    Layout<ASH, ADH>: [const] LayoutAdd<Layout<BSH, BDH>, PH, Output = Layout<HS, HD>>,
    Layout<AST, ADT>: [const] LayoutAdd<Layout<BST, BDT>, PT, Output = Layout<TS, TD>>,
    HS: Shape,
    HD: Stride<HS>,
    TS: Tuple + Shape,
    TD: Tuple + Stride<TS>,
{
    type Output = Layout<Til<HS, TS>, Til<HD, TD>>;

    fn layout_add(self, b: Layout<Til<BSH, BST>, Til<BDH, BDT>>) -> Self::Output {
        let head = <Layout<ASH, ADH> as LayoutAdd<Layout<BSH, BDH>, PH>>::layout_add(
            Layout::new(self.shape.0, self.stride.0),
            Layout::new(b.shape.0, b.stride.0),
        );
        let tail = <Layout<AST, ADT> as LayoutAdd<Layout<BST, BDT>, PT>>::layout_add(
            Layout::new(self.shape.1, self.stride.1),
            Layout::new(b.shape.1, b.stride.1),
        );
        Layout::new(Til(head.shape, tail.shape), Til(head.stride, tail.stride))
    }
}

impl<S, D> Layout<S, D> {
    /// Whole-layout coordinate-wise addition `self + b`.
    pub const fn layout_add<B>(self, b: B) -> <Self as LayoutAdd<B, LeafProfile>>::Output
    where
        Self: [const] LayoutAdd<B, LeafProfile>,
    {
        LayoutAdd::<B, LeafProfile>::layout_add(self, b)
    }

    /// Layout addition under explicit type-level profile `P`.
    pub const fn layout_add_by<P, B>(self, b: B) -> <Self as LayoutAdd<B, P>>::Output
    where
        Self: [const] LayoutAdd<B, P>,
    {
        LayoutAdd::<B, P>::layout_add(self, b)
    }

    /// Layout addition guided by the structural profile of `profile` (e.g. `int_tuple![1, 1]`).
    pub const fn layout_add_with_profile<B, Prof: Item>(
        self,
        b: B,
        _profile: Prof,
    ) -> <Self as LayoutAdd<B, Prof::Profile>>::Output
    where
        Self: [const] LayoutAdd<B, Prof::Profile>,
    {
        LayoutAdd::<B, Prof::Profile>::layout_add(self, b)
    }
}

/// Whole-layout coordinate-wise addition `a + b`.
pub const fn layout_add<A, B>(a: A, b: B) -> <A as LayoutAdd<B, LeafProfile>>::Output
where
    A: [const] LayoutAdd<B, LeafProfile>,
{
    LayoutAdd::<B, LeafProfile>::layout_add(a, b)
}

/// Coordinate-wise addition `a + b` under explicit type-level profile `P`.
pub const fn layout_add_by<P, A, B>(a: A, b: B) -> <A as LayoutAdd<B, P>>::Output
where
    A: [const] LayoutAdd<B, P>,
{
    LayoutAdd::<B, P>::layout_add(a, b)
}

/// Coordinate-wise addition `a + b` guided by the structural profile of `profile`.
pub const fn layout_add_with_profile<A, B, Prof: Item>(
    a: A,
    b: B,
    _profile: Prof,
) -> <A as LayoutAdd<B, Prof::Profile>>::Output
where
    A: [const] LayoutAdd<B, Prof::Profile>,
{
    LayoutAdd::<B, Prof::Profile>::layout_add(a, b)
}
