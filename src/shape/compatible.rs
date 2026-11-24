//! Shape compatibility (`Compatible`) and lattice operations (`CommonRefinement`, `CommonCoarsening`).
//!
//! Models CuTe's partial order $A \unlhd B$ ("$A$ coarsens $B$"), together with its join
//! (least upper bound) and meet (greatest lower bound) operations.
//!
//! # Main Interfaces
//!
//! - [`Compatible`] / [`compatible`]: Tests whether `lhs` coarsens `rhs`.
//! - [`CommonRefinement`] / [`common_refinement`]: Computes the minimal shape refining both inputs.
//! - [`CommonCoarsening`] / [`common_coarsening`]: Computes the maximal shape coarsening both inputs.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::shape::{common_coarsening, common_refinement, compatible};
//!
//! let a = int_tuple![4i32, (3i32, 5i32)];
//! let b = int_tuple![(2i32, 2i32), 15i32];
//! assert_eq!(common_refinement(&a, &b), int_tuple![(2i32, 2i32), (3i32, 5i32)]);
//! assert_eq!(common_coarsening(&a, &b), int_tuple![4i32, 15i32]);
//! assert!(compatible(&Int::<60>, &a));
//! ```
//!
//! # Notes
//!
//! - A scalar matches any shape with the same total size; a tuple never matches a scalar.
//! - [`CommonRefinement`] panics at runtime (or fails in `const`) if no common refinement exists.
//! - [`CommonCoarsening`] falls back to total size when static tuples have incompatible
//!   modes of equal total size, and panics when sizes disagree or dynamic modes mismatch.

use crate::integer::{IS_NON_NEG, Int, int};
use crate::tuple::{Nil, Til, Tuple};

use super::{Shape, ShapeScalar, ShapeTuple, ValidShapeExtent};

/// Whether the left shape coarsens the right, including matching mode sizes.
///
/// CuTe formalizes shape compatibility as a partial order $A \unlhd B$ (read "$A$ coarsens $B$").
///
/// # Compatibility Rules
/// 1. A scalar matches any shape with the same total size.
/// 2. Tuples must have equal ranks and compatible corresponding modes.
/// 3. A tuple never matches a scalar.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::shape::Compatible;
///
/// // Integer coarsens matching tuple:
/// assert!(30i32.compatible(&int_tuple![2i32, 15i32]));
///
/// // Mode-by-mode compatibility:
/// assert!(int_tuple![2i32, 15i32].compatible(&int_tuple![2i32, (3i32, 5i32)]));
///
/// // Tuple never matches scalar:
/// assert!(!int_tuple![30i32].compatible(&30i32));
/// ```
pub const trait Compatible<R: Shape>: Shape {
    /// Returns `true` iff `self` coarsens `rhs`.
    fn compatible(&self, rhs: &R) -> bool;
}

const impl<R: Shape> Compatible<R> for Nil {
    fn compatible(&self, _: &R) -> bool {
        R::RANK == 0
    }
}

const impl<L: [const] ShapeScalar, R: [const] Shape> Compatible<R> for L {
    fn compatible(&self, rhs: &R) -> bool {
        self.size() == rhs.size()
    }
}

const impl<HL: Shape, TL: ShapeTuple> Compatible<Nil> for Til<HL, TL> {
    fn compatible(&self, _: &Nil) -> bool {
        false
    }
}

const impl<HL: Shape, TL: Tuple + Shape, R: ShapeScalar> Compatible<R> for Til<HL, TL> {
    fn compatible(&self, _: &R) -> bool {
        false
    }
}

const impl<HL, TL, HR, TR> Compatible<Til<HR, TR>> for Til<HL, TL>
where
    HL: [const] Compatible<HR>,
    TL: Tuple + [const] Compatible<TR>,
    HR: Shape,
    TR: Tuple + Shape,
{
    fn compatible(&self, rhs: &Til<HR, TR>) -> bool {
        self.0.compatible(&rhs.0) && self.1.compatible(&rhs.1)
    }
}

/// Returns whether `lhs` coarsens `rhs`.
///
/// A scalar matches any shape with the same total size. Tuples must have equal
/// ranks and compatible corresponding modes. A tuple never matches a scalar.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::shape::compatible;
///
/// assert!(compatible(&30i32, &int_tuple![2i32, 15i32]));
/// assert!(compatible(
///     &int_tuple![Int::<2>, 15i32],
///     &int_tuple![2i32, (3i32, 5i32)],
/// ));
/// assert!(!compatible(&int_tuple![30i32], &30i32));
/// ```
pub const fn compatible<L: [const] Compatible<R>, R: Shape>(lhs: &L, rhs: &R) -> bool {
    lhs.compatible(rhs)
}

const SHAPE_IS_STATIC<S: Shape>: bool = S::IS_STATIC;
const STATIC_SCALAR_VAL<S: ShapeScalar>: int = S::STATIC_SIZE.unwrap();

/// Dispatches scalar refinement between `Self` and `R`; `L_STATIC` and `R_STATIC`
/// preserve static `Int<N>` whenever either side is compile-time static.
#[doc(hidden)]
pub const trait CommonScalarRefinement<R: ShapeScalar, const L_STATIC: bool, const R_STATIC: bool>:
    ShapeScalar
{
    type Output: ShapeScalar;
    fn refine_scalar(&self, rhs: &R) -> Self::Output;
}

const impl<L: [const] ShapeScalar, R: [const] ShapeScalar> CommonScalarRefinement<R, false, false>
    for L
{
    type Output = int;
    fn refine_scalar(&self, rhs: &R) -> int {
        let sa = self.size();
        let sb = rhs.size();
        assert!(
            sa == sb,
            "common_refinement: shapes have no common refinement"
        );
        sa
    }
}

const impl<L: [const] ShapeScalar, R: [const] ShapeScalar> CommonScalarRefinement<R, true, false>
    for L
where
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_SCALAR_VAL::<L> }> }>,
{
    type Output = Int<{ STATIC_SCALAR_VAL::<L> }>;
    fn refine_scalar(&self, rhs: &R) -> Self::Output {
        assert!(
            STATIC_SCALAR_VAL::<L> == rhs.size(),
            "common_refinement: shapes have no common refinement"
        );
        Int
    }
}

const impl<L: [const] ShapeScalar, R: [const] ShapeScalar> CommonScalarRefinement<R, false, true>
    for L
where
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_SCALAR_VAL::<R> }> }>,
{
    type Output = Int<{ STATIC_SCALAR_VAL::<R> }>;
    fn refine_scalar(&self, _: &R) -> Self::Output {
        assert!(
            self.size() == STATIC_SCALAR_VAL::<R>,
            "common_refinement: shapes have no common refinement"
        );
        Int
    }
}

const impl<L: [const] ShapeScalar, R: [const] ShapeScalar> CommonScalarRefinement<R, true, true>
    for L
where
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_SCALAR_VAL::<L> }> }>,
{
    type Output = Int<{ STATIC_SCALAR_VAL::<L> }>;
    fn refine_scalar(&self, _: &R) -> Self::Output {
        assert!(
            STATIC_SCALAR_VAL::<L> == STATIC_SCALAR_VAL::<R>,
            "common_refinement: shapes have no common refinement"
        );
        Int
    }
}

/// Computes the join (least upper bound) of two shapes under [`Compatible`].
pub const trait CommonRefinement<R: Shape>: Shape {
    /// The refined output shape type.
    type Output: Shape;
    /// Returns the minimal shape refining both `self` and `rhs`, panicking if incompatible.
    fn common_refinement(&self, rhs: &R) -> Self::Output;
}

const impl CommonRefinement<Nil> for Nil {
    type Output = Nil;
    fn common_refinement(&self, _: &Nil) -> Nil {
        Nil
    }
}

const impl<R: [const] ShapeScalar> CommonRefinement<R> for Nil {
    type Output = Nil;
    fn common_refinement(&self, rhs: &R) -> Nil {
        assert!(
            rhs.size() == 1,
            "common_refinement: shapes have no common refinement"
        );
        Nil
    }
}

const impl<HR: Shape, TR: Tuple + Shape> CommonRefinement<Til<HR, TR>> for Nil {
    type Output = Nil;
    fn common_refinement(&self, _: &Til<HR, TR>) -> Nil {
        panic!("common_refinement: shapes have no common refinement")
    }
}

const impl<L: [const] ShapeScalar> CommonRefinement<Nil> for L {
    type Output = Nil;
    fn common_refinement(&self, _: &Nil) -> Nil {
        assert!(
            self.size() == 1,
            "common_refinement: shapes have no common refinement"
        );
        Nil
    }
}

const impl<L, R> CommonRefinement<R> for L
where
    L: [const] ShapeScalar
        + [const] CommonScalarRefinement<R, { SHAPE_IS_STATIC::<L> }, { SHAPE_IS_STATIC::<R> }>,
    R: [const] ShapeScalar,
{
    type Output = <L as CommonScalarRefinement<
        R,
        { SHAPE_IS_STATIC::<L> },
        { SHAPE_IS_STATIC::<R> },
    >>::Output;
    fn common_refinement(&self, rhs: &R) -> Self::Output {
        self.refine_scalar(rhs)
    }
}

const impl<L: [const] ShapeScalar, HR: [const] Shape, TR: [const] Tuple + [const] Shape>
    CommonRefinement<Til<HR, TR>> for L
{
    type Output = Til<HR, TR>;
    fn common_refinement(&self, rhs: &Til<HR, TR>) -> Self::Output {
        assert!(
            self.size() == rhs.size(),
            "common_refinement: shapes have no common refinement"
        );
        *rhs
    }
}

const impl<HL: Shape, TL: Tuple + Shape> CommonRefinement<Nil> for Til<HL, TL> {
    type Output = Nil;
    fn common_refinement(&self, _: &Nil) -> Nil {
        panic!("common_refinement: shapes have no common refinement")
    }
}

const impl<HL: [const] Shape, TL: [const] Tuple + [const] Shape, R: [const] ShapeScalar>
    CommonRefinement<R> for Til<HL, TL>
{
    type Output = Self;
    fn common_refinement(&self, rhs: &R) -> Self::Output {
        assert!(
            self.size() == rhs.size(),
            "common_refinement: shapes have no common refinement"
        );
        *self
    }
}

const impl<HL, TL, HR, TR> CommonRefinement<Til<HR, TR>> for Til<HL, TL>
where
    HL: [const] CommonRefinement<HR>,
    HR: [const] Shape,
    TL: Tuple + [const] CommonRefinement<TR>,
    TR: Tuple + [const] Shape,
    TL::Output: Tuple,
{
    type Output = Til<HL::Output, TL::Output>;
    fn common_refinement(&self, rhs: &Til<HR, TR>) -> Self::Output {
        Til(
            self.0.common_refinement(&rhs.0),
            self.1.common_refinement(&rhs.1),
        )
    }
}

/// Returns the minimal shape refining both `lhs` and `rhs`, panicking if no common refinement exists.
pub const fn common_refinement<L: [const] CommonRefinement<R>, R: Shape>(
    lhs: &L,
    rhs: &R,
) -> L::Output {
    lhs.common_refinement(rhs)
}

/// Checks whether tuple shapes `Self` and `R` have equal ranks and statically non-conflicting mode sizes.
#[doc(hidden)]
pub const trait TupleModesCompatibleSize<R: Shape>: Shape {
    const MODES_MATCH: bool;
}

const impl TupleModesCompatibleSize<Nil> for Nil {
    const MODES_MATCH: bool = true;
}

const impl<HR: Shape, TR: Tuple + Shape> TupleModesCompatibleSize<Til<HR, TR>> for Nil {
    const MODES_MATCH: bool = false;
}

const impl<HL: Shape, TL: Tuple + Shape> TupleModesCompatibleSize<Nil> for Til<HL, TL> {
    const MODES_MATCH: bool = false;
}

const impl<HL, TL, HR, TR> TupleModesCompatibleSize<Til<HR, TR>> for Til<HL, TL>
where
    HL: Shape,
    TL: Tuple + TupleModesCompatibleSize<TR>,
    HR: Shape,
    TR: Tuple + Shape,
{
    const MODES_MATCH: bool = !matches!(
        (HL::STATIC_SIZE, HR::STATIC_SIZE),
        (Some(sa), Some(sb)) if sa != sb
    ) && TL::MODES_MATCH;
}

const MODES_COMPAT_SIZE<L: TupleModesCompatibleSize<R>, R: Shape>: bool = L::MODES_MATCH;
const SHAPE_HAS_STATIC_SIZE<S: Shape>: bool = S::STATIC_SIZE.is_some();
const COARSENED_STATIC_SIZE<S: Shape>: int = S::STATIC_SIZE.unwrap();

/// Scalar fallback meet between `Self` and `R` when ranks or static mode sizes differ;
/// `L_STATIC` and `R_STATIC` preserve a static total size `Int<SIZE>` when available.
#[doc(hidden)]
pub const trait ScalarMeet<R: Shape, const L_STATIC: bool, const R_STATIC: bool>:
    Shape
{
    type Output: ShapeScalar;
    fn scalar_meet(&self, rhs: &R) -> Self::Output;
}

const impl<L: [const] Shape, R: [const] Shape> ScalarMeet<R, false, false> for L {
    type Output = int;
    fn scalar_meet(&self, rhs: &R) -> int {
        let sa = self.size();
        let sb = rhs.size();
        assert!(
            sa == sb,
            "common_coarsening: shapes have no common coarsening"
        );
        sa
    }
}

const impl<L: [const] Shape, R: [const] Shape> ScalarMeet<R, true, false> for L
where
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ COARSENED_STATIC_SIZE::<L> }> }>,
{
    type Output = Int<{ COARSENED_STATIC_SIZE::<L> }>;
    fn scalar_meet(&self, rhs: &R) -> Self::Output {
        assert!(
            COARSENED_STATIC_SIZE::<L> == rhs.size(),
            "common_coarsening: shapes have no common coarsening"
        );
        Int
    }
}

const impl<L: [const] Shape, R: [const] Shape> ScalarMeet<R, false, true> for L
where
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ COARSENED_STATIC_SIZE::<R> }> }>,
{
    type Output = Int<{ COARSENED_STATIC_SIZE::<R> }>;
    fn scalar_meet(&self, _: &R) -> Self::Output {
        assert!(
            self.size() == COARSENED_STATIC_SIZE::<R>,
            "common_coarsening: shapes have no common coarsening"
        );
        Int
    }
}

const impl<L: Shape, R: Shape> ScalarMeet<R, true, true> for L
where
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ COARSENED_STATIC_SIZE::<L> }> }>,
{
    type Output = Int<{ COARSENED_STATIC_SIZE::<L> }>;
    fn scalar_meet(&self, _: &R) -> Self::Output {
        assert!(
            COARSENED_STATIC_SIZE::<L> == COARSENED_STATIC_SIZE::<R>,
            "common_coarsening: shapes have no common coarsening"
        );
        Int
    }
}

/// Dispatches tuple coarsening between `Self` and `R`: `AS_TUPLE = true` meets mode-by-mode,
/// `false` falls back to `ScalarMeet`.
#[doc(hidden)]
pub const trait TupleCoarseningStep<R: Shape, const AS_TUPLE: bool>: Shape {
    type Output: Shape;
    fn tuple_coarsen(&self, rhs: &R) -> Self::Output;
}

/// Recursively meets the equal-rank tail chains `Self` and `R` mode-by-mode.
#[doc(hidden)]
pub const trait TupleTailCoarsening<R: Tuple + Shape>: Tuple + Shape {
    type Output: Tuple + Shape;
    fn tail_coarsen(&self, rhs: &R) -> Self::Output;
}

const impl TupleTailCoarsening<Nil> for Nil {
    type Output = Nil;
    fn tail_coarsen(&self, _: &Nil) -> Nil {
        Nil
    }
}

const impl<HL, TL, HR, TR> TupleTailCoarsening<Til<HR, TR>> for Til<HL, TL>
where
    HL: [const] CommonCoarsening<HR>,
    HR: [const] Shape,
    TL: [const] TupleTailCoarsening<TR>,
    TR: Tuple + [const] Shape,
{
    type Output = Til<HL::Output, TL::Output>;
    fn tail_coarsen(&self, rhs: &Til<HR, TR>) -> Self::Output {
        Til(
            self.0.common_coarsening(&rhs.0),
            self.1.tail_coarsen(&rhs.1),
        )
    }
}

const impl<HL, TL, HR, TR> TupleCoarseningStep<Til<HR, TR>, false> for Til<HL, TL>
where
    HL: [const] Shape,
    TL: Tuple + [const] Shape,
    HR: [const] Shape,
    TR: Tuple + [const] Shape,
    Self: [const] ScalarMeet<
            Til<HR, TR>,
            { SHAPE_HAS_STATIC_SIZE::<Self> },
            { SHAPE_HAS_STATIC_SIZE::<Til<HR, TR>> },
        >,
{
    type Output = <Self as ScalarMeet<
        Til<HR, TR>,
        { SHAPE_HAS_STATIC_SIZE::<Self> },
        { SHAPE_HAS_STATIC_SIZE::<Til<HR, TR>> },
    >>::Output;
    fn tuple_coarsen(&self, rhs: &Til<HR, TR>) -> Self::Output {
        self.scalar_meet(rhs)
    }
}

const impl<HL, TL, HR, TR> TupleCoarseningStep<Til<HR, TR>, true> for Til<HL, TL>
where
    HL: [const] CommonCoarsening<HR>,
    HR: [const] Shape,
    TL: [const] TupleTailCoarsening<TR>,
    TR: Tuple + [const] Shape,
{
    type Output = Til<HL::Output, TL::Output>;
    fn tuple_coarsen(&self, rhs: &Til<HR, TR>) -> Self::Output {
        Til(
            self.0.common_coarsening(&rhs.0),
            self.1.tail_coarsen(&rhs.1),
        )
    }
}

/// Computes the meet (greatest lower bound) of two shapes under [`Compatible`].
///
/// Static shapes with equal total size but incompatible mode ranks/extents automatically fall
/// back to their scalar total size (`Int<SIZE>`). Dynamic equal-rank tuples commit to a
/// tuple return type and panic if runtime mode sizes disagree.
pub const trait CommonCoarsening<R: Shape>: Shape {
    /// The coarsened output shape type.
    type Output: Shape;
    /// Returns the maximal shape coarsening both `self` and `rhs`, panicking if sizes differ.
    fn common_coarsening(&self, rhs: &R) -> Self::Output;
}

const impl CommonCoarsening<Nil> for Nil {
    type Output = Nil;
    fn common_coarsening(&self, _: &Nil) -> Nil {
        Nil
    }
}

const impl<R: ShapeScalar> CommonCoarsening<R> for Nil
where
    Self: [const] ScalarMeet<R, { SHAPE_HAS_STATIC_SIZE::<Self> }, { SHAPE_HAS_STATIC_SIZE::<R> }>,
{
    type Output = <Self as ScalarMeet<
        R,
        { SHAPE_HAS_STATIC_SIZE::<Self> },
        { SHAPE_HAS_STATIC_SIZE::<R> },
    >>::Output;
    fn common_coarsening(&self, rhs: &R) -> Self::Output {
        self.scalar_meet(rhs)
    }
}

const impl<HR: [const] Shape, TR: Tuple + [const] Shape> CommonCoarsening<Til<HR, TR>> for Nil
where
    Self: [const] ScalarMeet<
            Til<HR, TR>,
            { SHAPE_HAS_STATIC_SIZE::<Self> },
            { SHAPE_HAS_STATIC_SIZE::<Til<HR, TR>> },
        >,
{
    type Output = <Self as ScalarMeet<
        Til<HR, TR>,
        { SHAPE_HAS_STATIC_SIZE::<Self> },
        { SHAPE_HAS_STATIC_SIZE::<Til<HR, TR>> },
    >>::Output;
    fn common_coarsening(&self, rhs: &Til<HR, TR>) -> Self::Output {
        self.scalar_meet(rhs)
    }
}

const impl<L: [const] ShapeScalar, R: [const] Shape> CommonCoarsening<R> for L
where
    Self: [const] ScalarMeet<R, { SHAPE_HAS_STATIC_SIZE::<Self> }, { SHAPE_HAS_STATIC_SIZE::<R> }>,
{
    type Output = <Self as ScalarMeet<
        R,
        { SHAPE_HAS_STATIC_SIZE::<Self> },
        { SHAPE_HAS_STATIC_SIZE::<R> },
    >>::Output;
    fn common_coarsening(&self, rhs: &R) -> Self::Output {
        self.scalar_meet(rhs)
    }
}

const impl<HL: [const] Shape, TL: Tuple + [const] Shape> CommonCoarsening<Nil> for Til<HL, TL>
where
    Self: [const] ScalarMeet<Nil, { SHAPE_HAS_STATIC_SIZE::<Self> }, { SHAPE_HAS_STATIC_SIZE::<Nil> }>,
{
    type Output = <Self as ScalarMeet<
        Nil,
        { SHAPE_HAS_STATIC_SIZE::<Self> },
        { SHAPE_HAS_STATIC_SIZE::<Nil> },
    >>::Output;
    fn common_coarsening(&self, rhs: &Nil) -> Self::Output {
        self.scalar_meet(rhs)
    }
}

const impl<HL: [const] Shape, TL: Tuple + [const] Shape, R: [const] ShapeScalar> CommonCoarsening<R>
    for Til<HL, TL>
where
    Self: [const] ScalarMeet<R, { SHAPE_HAS_STATIC_SIZE::<Self> }, { SHAPE_HAS_STATIC_SIZE::<R> }>,
{
    type Output = <Self as ScalarMeet<
        R,
        { SHAPE_HAS_STATIC_SIZE::<Self> },
        { SHAPE_HAS_STATIC_SIZE::<R> },
    >>::Output;
    fn common_coarsening(&self, rhs: &R) -> Self::Output {
        self.scalar_meet(rhs)
    }
}

const impl<HL, TL, HR, TR> CommonCoarsening<Til<HR, TR>> for Til<HL, TL>
where
    HL: [const] Shape,
    TL: Tuple + [const] Shape,
    HR: [const] Shape,
    TR: Tuple + [const] Shape,
    Self: TupleModesCompatibleSize<Til<HR, TR>>
        + [const] TupleCoarseningStep<Til<HR, TR>, { MODES_COMPAT_SIZE::<Self, Til<HR, TR>> }>,
{
    type Output = <Self as TupleCoarseningStep<
        Til<HR, TR>,
        { MODES_COMPAT_SIZE::<Self, Til<HR, TR>> },
    >>::Output;
    fn common_coarsening(&self, rhs: &Til<HR, TR>) -> Self::Output {
        self.tuple_coarsen(rhs)
    }
}

/// Returns the maximal shape coarsening both `lhs` and `rhs`, panicking if no common coarsening exists.
pub const fn common_coarsening<L: [const] CommonCoarsening<R>, R: Shape>(
    lhs: &L,
    rhs: &R,
) -> L::Output {
    lhs.common_coarsening(rhs)
}
