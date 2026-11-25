//! Coordinate-index conversions (`crd2idx`, `idx2crd`), bounds checking (`in_bounds`),
//! and colexicographical coordinate iteration (`coordinates`).
//!
//! # Main Interfaces
//!
//! - [`CoordToIndex`] / [`coord_to_index`] / [`crd2idx`]: Maps natural, partial, or basis
//!   coordinates to a 1D linear index in colexicographical order.
//! - [`IndexToCoord`] / [`index_to_coord`] / [`idx2crd`]: Unpacks a 1D linear index or
//!   basis coordinate into a natural coordinate matching a [`Shape`].
//! - [`InBounds`] / [`in_bounds`]: Checks whether a coordinate satisfies $0 \le c_i < s_i$
//!   at every shape leaf.
//! - [`Coordinates`] / [`coordinates`]: Iterates over all natural coordinates of a shape.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::int_tuple;
//! use noodle_cute::shape::{crd2idx, idx2crd, in_bounds};
//!
//! let s = int_tuple![3i32, (2i32, 4i32)];
//! let crd = int_tuple![1i32, (0i32, 1i32)];
//! assert_eq!(crd2idx(crd, &s), 7);
//! assert_eq!(idx2crd(7i32, &s), crd);
//! assert!(in_bounds(crd, &s));
//! ```
//!
//! # Notes
//!
//! - Index unpacking uses floor division (`floor_divmod`), so negative linear indices wrap
//!   modulo earlier modes and leave the final mode signed.
//! - Empty shapes (`Nil`, `(Nil, Nil)`) are vacuously in bounds, whereas any zero extent
//!   in `shape` yields `false` under [`in_bounds`].

use core::iter::FusedIterator;
use core::ops::RangeFull;

use crate::basis::{
    BasisCoeff, BasisCoeffAdd, BasisCoeffMul, BasisCombination, BasisPath, BasisTerm, BasisTuple,
    COEFF_IS_STATIC_ZERO, EQ_MODE, EmitBasisTerm, Mode, ModeEnd, SplitModePrefix,
};
use crate::integer::{
    ArithFloorDivMod, FloorQuotient, FloorRemainder, INCREASE, Int, Integer, floor_divmod, ge, int,
    lt,
};
use crate::tuple::{FlatTuple, Item, Leaf, Nil, RepeatLike, Til, Tuple, TupleProduct, repeat_like};

use super::{Shape, ShapeScalar};

/// Converts natural coordinates into a 1D linear index.
///
/// In CuTe, indexing follows **colexicographical ordering** (generalized column-major).
/// For a coordinate $(c_0, c_1, \dots, c_{R-1})$ and a shape $(s_0, s_1, \dots, s_{R-1})$:
/// $$\text{idx} = c_0 + c_1 \cdot s_0 + c_2 \cdot (s_0 \cdot s_1) + \dots$$
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::shape::CoordToIndex;
///
/// let s = int_tuple![3i32, 4i32];
/// let crd = int_tuple![1i32, 2i32];
/// // idx = 1 + 2 * 3 = 7
/// assert_eq!(crd.coord_to_index(&s), 7);
/// ```
pub const trait CoordToIndex<S: Shape>: Copy {
    /// Linear index scalar type (an [`Integer`] or algebraic [`BasisCoeff`] such as `F2`).
    type Output: [const] BasisCoeff;
    /// Converts `self` into a 1D linear index within `shape`.
    fn coord_to_index(self, shape: &S) -> Self::Output;
}

const impl CoordToIndex<Nil> for Nil {
    type Output = Int<0>;
    fn coord_to_index(self, _: &Nil) -> Self::Output {
        Int
    }
}

const impl<S: [const] Shape, I: [const] Integer> CoordToIndex<S> for I {
    type Output = I;
    fn coord_to_index(self, shape: &S) -> Self::Output {
        let _ = shape.has_zero();
        self
    }
}

const impl<V, P, S> CoordToIndex<S> for BasisTerm<V, P>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    S: [const] Shape,
    Self: [const] IndexToCoord<S>,
    <Self as IndexToCoord<S>>::Output: [const] CoordToIndex<S>,
{
    type Output = <<Self as IndexToCoord<S>>::Output as CoordToIndex<S>>::Output;
    fn coord_to_index(self, shape: &S) -> Self::Output {
        self.index_to_coord(shape).coord_to_index(shape)
    }
}

const impl<T, S> CoordToIndex<S> for BasisCombination<T>
where
    T: [const] BasisTuple,
    S: [const] Shape,
    Self: [const] IndexToCoord<S>,
    <Self as IndexToCoord<S>>::Output: [const] CoordToIndex<S>,
{
    type Output = <<Self as IndexToCoord<S>>::Output as CoordToIndex<S>>::Output;
    fn coord_to_index(self, shape: &S) -> Self::Output {
        self.index_to_coord(shape).coord_to_index(shape)
    }
}

const impl<H, SH> CoordToIndex<Til<SH, Nil>> for Til<H, Nil>
where
    H: [const] CoordToIndex<SH>,
    SH: Shape,
{
    type Output = H::Output;
    fn coord_to_index(self, shape: &Til<SH, Nil>) -> Self::Output {
        self.0.coord_to_index(&shape.0)
    }
}

const impl<H, TH, TT, SH, STH, STT> CoordToIndex<Til<SH, Til<STH, STT>>> for Til<H, Til<TH, TT>>
where
    H: [const] CoordToIndex<SH>,
    TH: Item,
    TT: Tuple,
    Til<TH, TT>: [const] CoordToIndex<Til<STH, STT>>,
    SH: [const] Shape + [const] TupleProduct,
    STH: Shape,
    STT: Tuple + Shape,
    <Til<TH, TT> as CoordToIndex<Til<STH, STT>>>::Output: [const] BasisCoeffMul<SH::Output>,
    H::Output: [const] BasisCoeffAdd<
        <<Til<TH, TT> as CoordToIndex<Til<STH, STT>>>::Output as BasisCoeffMul<SH::Output>>::Output,
    >,
{
    type Output = <H::Output as BasisCoeffAdd<
        <<Til<TH, TT> as CoordToIndex<Til<STH, STT>>>::Output as BasisCoeffMul<SH::Output>>::Output,
    >>::Output;
    fn coord_to_index(self, shape: &Til<SH, Til<STH, STT>>) -> Self::Output {
        let head = self.0.coord_to_index(&shape.0);
        let tail = self.1.coord_to_index(&shape.1);
        let stride = shape.0.tuple_product();
        head.coeff_add(tail.coeff_mul(stride))
    }
}

/// Converts a coordinate to a linear index within a shape.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::shape::coord_to_index;
///
/// let s = int_tuple![3i32, 4i32];
/// let crd = int_tuple![1i32, 2i32];
/// assert_eq!(coord_to_index(crd, &s), 7);
/// ```
pub const fn coord_to_index<C: [const] CoordToIndex<S>, S: Shape>(
    coord: C,
    shape: &S,
) -> C::Output {
    coord.coord_to_index(shape)
}

/// Alias for [`coord_to_index`] matching CuTe naming (`crd2idx`).
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::shape::crd2idx;
///
/// let s = int_tuple![3i32, (2i32, 3i32)];
/// let crd = int_tuple![1i32, (0i32, 1i32)];
/// assert_eq!(crd2idx(crd, &s), 7);
/// ```
pub const fn crd2idx<C: [const] CoordToIndex<S>, S: Shape>(coord: C, shape: &S) -> C::Output {
    coord.coord_to_index(shape)
}

/// Converts a 1D linear index (or nested coordinate) into a natural coordinate.
///
/// This performs the inverse mapping of [`CoordToIndex`]. For a shape $S = (s_0, s_1, \dots)$,
/// the remainder modulo $s_0$ gives the coordinate in mode 0, while the quotient is
/// recursively unpacked into the remaining modes.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::shape::IndexToCoord;
///
/// let s = int_tuple![3i32, 4i32];
/// // 7 % 3 = 1, 7 / 3 = 2 -> (1, 2)
/// assert_eq!(7i32.index_to_coord(&s), int_tuple![1i32, 2i32]);
/// ```
pub const trait IndexToCoord<S: Shape> {
    /// Natural coordinate type matching `S`.
    type Output: Item;
    /// Unpacks `self` into a natural coordinate for `shape`.
    fn index_to_coord(self, shape: &S) -> Self::Output;
}

const impl IndexToCoord<Nil> for Nil {
    type Output = Nil;
    fn index_to_coord(self, _: &Nil) -> Self::Output {
        Nil
    }
}

const impl<I: [const] Integer> IndexToCoord<Nil> for I {
    type Output = Nil;
    fn index_to_coord(self, _: &Nil) -> Self::Output {
        Nil
    }
}

const impl<S: [const] ShapeScalar, I: [const] Integer> IndexToCoord<S> for I {
    type Output = I;
    fn index_to_coord(self, shape: &S) -> Self::Output {
        let _ = shape.size();
        self
    }
}

const impl<S> IndexToCoord<S> for RangeFull
where
    S: [const] Shape + [const] RepeatLike<Int<0>>,
{
    type Output = <S as RepeatLike<Int<0>>>::Output;
    fn index_to_coord(self, shape: &S) -> Self::Output {
        let _ = shape.has_zero();
        repeat_like(Int::<0>, shape)
    }
}

const impl<H, T, SH, ST> IndexToCoord<Til<SH, ST>> for Til<H, T>
where
    SH: Shape,
    ST: Tuple + Shape,
    H: Copy + [const] IndexToCoord<SH>,
    T: Tuple + [const] IndexToCoord<ST>,
    T::Output: Tuple,
{
    type Output = Til<H::Output, T::Output>;
    fn index_to_coord(self, shape: &Til<SH, ST>) -> Self::Output {
        Til(
            self.0.index_to_coord(&shape.0),
            self.1.index_to_coord(&shape.1),
        )
    }
}

const HAS_NO_LEAVES<T: Item>: bool = T::LEAVES == 0;
const AND_BOOL<const A: bool, const B: bool>: bool = A && B;

/// Leaf-by-leaf colexicographical decomposition carrying remaining quotient `RemIndex` across
/// shape subtree `S`; `IS_LAST = true` skips `floor_divmod` at the final leaf.
#[doc(hidden)]
pub const trait IndexToCoordCarry<S: Shape, const IS_LAST: bool>: [const] Integer {
    type Output: Item;
    type RemIndex: [const] Integer;
    fn index_to_coord_carry(self, shape: &S) -> (Self::Output, Self::RemIndex);
}

const impl<I: [const] Integer, const IS_LAST: bool> IndexToCoordCarry<Nil, IS_LAST> for I {
    type Output = Nil;
    type RemIndex = I;
    fn index_to_coord_carry(self, _: &Nil) -> (Nil, I) {
        (Nil, self)
    }
}

const impl<S: [const] ShapeScalar, I: [const] Integer> IndexToCoordCarry<S, true> for I {
    type Output = I;
    type RemIndex = Int<0>;
    fn index_to_coord_carry(self, shape: &S) -> (I, Int<0>) {
        let _ = shape.size();
        (self, Int)
    }
}

const impl<S, I> IndexToCoordCarry<S, false> for I
where
    S: [const] ShapeScalar,
    I: [const] Integer + [const] ArithFloorDivMod<I, S>,
{
    type Output = FloorRemainder<I, S>;
    type RemIndex = FloorQuotient<I, S>;
    fn index_to_coord_carry(self, shape: &S) -> (Self::Output, Self::RemIndex) {
        let _ = shape.size();
        let (quot, rem) = floor_divmod(self, *shape);
        (rem, quot)
    }
}

const impl<SH, ST, I, const IS_LAST: bool> IndexToCoordCarry<Til<SH, ST>, IS_LAST> for I
where
    SH: Shape,
    ST: Tuple + Shape,
    I: [const] Integer
        + [const] IndexToCoordCarry<SH, { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> }>,
    <I as IndexToCoordCarry<SH, { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> }>>::RemIndex:
        [const] IndexToCoordCarry<ST, IS_LAST>,
    <<I as IndexToCoordCarry<SH, { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> }>>::RemIndex as IndexToCoordCarry<ST, IS_LAST>>::Output:
        Tuple,
{
    type Output = Til<
        <I as IndexToCoordCarry<SH, { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> }>>::Output,
        <<I as IndexToCoordCarry<SH, { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> }>>::RemIndex as IndexToCoordCarry<ST, IS_LAST>>::Output,
    >;
    type RemIndex = <<I as IndexToCoordCarry<
        SH,
        { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> },
    >>::RemIndex as IndexToCoordCarry<ST, IS_LAST>>::RemIndex;

    fn index_to_coord_carry(self, shape: &Til<SH, ST>) -> (Self::Output, Self::RemIndex) {
        let (head, rem) = self.index_to_coord_carry(&shape.0);
        let (tail, rem2) = rem.index_to_coord_carry(&shape.1);
        (Til(head, tail), rem2)
    }
}

const impl<SH, ST, I> IndexToCoord<Til<SH, ST>> for I
where
    SH: Shape,
    ST: Tuple + Shape,
    I: [const] Integer + [const] IndexToCoordCarry<Til<SH, ST>, true>,
{
    type Output = <I as IndexToCoordCarry<Til<SH, ST>, true>>::Output;
    fn index_to_coord(self, shape: &Til<SH, ST>) -> Self::Output {
        self.index_to_coord_carry(shape).0
    }
}

/// Converts a 1D linear index into a natural coordinate for a shape.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::shape::index_to_coord;
///
/// let s = int_tuple![3i32, 4i32];
/// assert_eq!(index_to_coord(7i32, &s), int_tuple![1i32, 2i32]);
/// ```
pub const fn index_to_coord<I: [const] IndexToCoord<S>, S: Shape>(
    index: I,
    shape: &S,
) -> I::Output {
    index.index_to_coord(shape)
}

/// Alias for [`index_to_coord`] matching CuTe naming (`idx2crd`).
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::shape::idx2crd;
///
/// let s = int_tuple![3i32, (2i32, 3i32)];
/// assert_eq!(idx2crd(7i32, &s), int_tuple![1i32, (0i32, 1i32)]);
/// ```
pub const fn idx2crd<I: [const] IndexToCoord<S>, S: Shape>(index: I, shape: &S) -> I::Output {
    index.index_to_coord(shape)
}

/// Recursively materializes a path-sorted `BasisTuple` into natural coordinates for `S`
/// starting at mode index `MODE`, zero-padding absent modes with `Int<0>`.
#[doc(hidden)]
pub const trait BasisTermsToCoord<const MODE: int, S: Shape>: FlatTuple {
    type Output: Item;
    fn basis_terms_to_coord(self, shape: &S) -> Self::Output;
}

/// Single-step dispatch for `BasisTermsToCoord`: `MATCH = true` splits terms under `MODE` for `S`.
#[doc(hidden)]
pub const trait BasisTermsToCoordStep<const MODE: int, S: Shape, const MATCH: bool>:
    FlatTuple
{
    type Output: Item;
    fn basis_terms_step(self, shape: &S) -> Self::Output;
}

const impl<const MODE: int> BasisTermsToCoord<MODE, Nil> for Nil {
    type Output = Nil;
    fn basis_terms_to_coord(self, _: &Nil) -> Nil {
        Nil
    }
}

const impl<S: [const] ShapeScalar> BasisTermsToCoord<0, S> for Nil
where
    Int<0>: [const] IndexToCoord<S>,
{
    type Output = <Int<0> as IndexToCoord<S>>::Output;
    fn basis_terms_to_coord(self, shape: &S) -> Self::Output {
        Int::<0>.index_to_coord(shape)
    }
}

const impl<const MODE: int, SH, ST> BasisTermsToCoord<MODE, Til<SH, ST>> for Nil
where
    SH: [const] Shape,
    ST: Tuple + [const] Shape,
    Int<0>: [const] IndexToCoord<SH>,
    Self: [const] BasisTermsToCoord<{ INCREASE::<MODE> }, ST>,
    <Self as BasisTermsToCoord<{ INCREASE::<MODE> }, ST>>::Output: Tuple,
{
    type Output = Til<
        <Int<0> as IndexToCoord<SH>>::Output,
        <Self as BasisTermsToCoord<{ INCREASE::<MODE> }, ST>>::Output,
    >;
    fn basis_terms_to_coord(self, shape: &Til<SH, ST>) -> Self::Output {
        Til(
            Int::<0>.index_to_coord(&shape.0),
            self.basis_terms_to_coord(&shape.1),
        )
    }
}

const impl<V, S> BasisTermsToCoord<0, S> for Til<BasisTerm<V, ModeEnd>, Nil>
where
    V: [const] BasisCoeff + [const] IndexToCoord<S>,
    S: [const] Shape,
    BasisTerm<V, ModeEnd>: [const] Leaf,
{
    type Output = V::Output;
    fn basis_terms_to_coord(self, shape: &S) -> Self::Output {
        self.0.value.index_to_coord(shape)
    }
}

const impl<const MODE: int, V, const M: int, P, Rest, SH, ST> BasisTermsToCoord<MODE, Til<SH, ST>>
    for Til<BasisTerm<V, Mode<M, P>>, Rest>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    Mode<M, P>: [const] BasisPath,
    Rest: FlatTuple,
    SH: [const] Shape,
    ST: Tuple + [const] Shape,
    BasisTerm<V, Mode<M, P>>: Item,
    Self: [const] BasisTermsToCoordStep<MODE, Til<SH, ST>, { EQ_MODE::<M, MODE> }>,
{
    type Output =
        <Self as BasisTermsToCoordStep<MODE, Til<SH, ST>, { EQ_MODE::<M, MODE> }>>::Output;
    fn basis_terms_to_coord(self, shape: &Til<SH, ST>) -> Self::Output {
        self.basis_terms_step(shape)
    }
}

const impl<const MODE: int, V, const M: int, P, Rest, SH, ST>
    BasisTermsToCoordStep<MODE, Til<SH, ST>, false> for Til<BasisTerm<V, Mode<M, P>>, Rest>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    Mode<M, P>: [const] BasisPath,
    Rest: FlatTuple,
    SH: [const] Shape,
    ST: Tuple + [const] Shape,
    BasisTerm<V, Mode<M, P>>: Item,
    Int<0>: [const] IndexToCoord<SH>,
    Self: [const] BasisTermsToCoord<{ INCREASE::<MODE> }, ST>,
    <Self as BasisTermsToCoord<{ INCREASE::<MODE> }, ST>>::Output: Tuple,
{
    type Output = Til<
        <Int<0> as IndexToCoord<SH>>::Output,
        <Self as BasisTermsToCoord<{ INCREASE::<MODE> }, ST>>::Output,
    >;
    fn basis_terms_step(self, shape: &Til<SH, ST>) -> Self::Output {
        Til(
            Int::<0>.index_to_coord(&shape.0),
            self.basis_terms_to_coord(&shape.1),
        )
    }
}

const impl<const MODE: int, V, const M: int, P, Rest, SH, ST>
    BasisTermsToCoordStep<MODE, Til<SH, ST>, true> for Til<BasisTerm<V, Mode<M, P>>, Rest>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    Mode<M, P>: [const] BasisPath,
    Rest: FlatTuple,
    SH: [const] Shape,
    ST: Tuple + [const] Shape,
    BasisTerm<V, Mode<M, P>>: Item,
    Self: [const] SplitModePrefix<MODE>,
    <Self as SplitModePrefix<MODE>>::Head: [const] BasisTermsToCoord<0, SH>,
    <Self as SplitModePrefix<MODE>>::Tail: [const] BasisTermsToCoord<{ INCREASE::<MODE> }, ST>,
    <<Self as SplitModePrefix<MODE>>::Tail as BasisTermsToCoord<{ INCREASE::<MODE> }, ST>>::Output:
        Tuple,
{
    type Output =
        Til<
            <<Self as SplitModePrefix<MODE>>::Head as BasisTermsToCoord<0, SH>>::Output,
            <<Self as SplitModePrefix<MODE>>::Tail as BasisTermsToCoord<
                { INCREASE::<MODE> },
                ST,
            >>::Output,
        >;
    fn basis_terms_step(self, shape: &Til<SH, ST>) -> Self::Output {
        let (head, tail) = self.split_mode_prefix();
        Til(
            head.basis_terms_to_coord(&shape.0),
            tail.basis_terms_to_coord(&shape.1),
        )
    }
}

const impl<V, P, S> IndexToCoord<S> for BasisTerm<V, P>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    S: [const] Shape,
    Nil: [const] EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>,
    <Nil as EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>>::Output:
        [const] BasisTermsToCoord<0, S>,
{
    type Output = <<Nil as EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>>::Output as BasisTermsToCoord<0, S>>::Output;
    fn index_to_coord(self, shape: &S) -> Self::Output {
        Nil.emit(self.value).basis_terms_to_coord(shape)
    }
}

const impl<T, S> IndexToCoord<S> for BasisCombination<T>
where
    T: [const] BasisTuple + [const] BasisTermsToCoord<0, S>,
    S: [const] Shape,
{
    type Output = <T as BasisTermsToCoord<0, S>>::Output;
    fn index_to_coord(self, shape: &S) -> Self::Output {
        self.terms().basis_terms_to_coord(shape)
    }
}

/// Verifies `0 <= c_i < s_i` at every leaf of an unpacked natural coordinate against `S`.
#[doc(hidden)]
pub const trait LeavesInBounds<S: Shape>: Item {
    fn leaves_in_bounds(&self, shape: &S) -> bool;
}

const impl LeavesInBounds<Nil> for Nil {
    fn leaves_in_bounds(&self, _: &Nil) -> bool {
        true
    }
}

const impl<C: [const] Integer, S: [const] ShapeScalar> LeavesInBounds<S> for C {
    fn leaves_in_bounds(&self, shape: &S) -> bool {
        let extent = shape.size();
        ge(*self, Int::<0>) && lt(*self, extent)
    }
}

const impl<CH, CT, SH, ST> LeavesInBounds<Til<SH, ST>> for Til<CH, CT>
where
    CH: [const] LeavesInBounds<SH>,
    CT: Tuple + [const] LeavesInBounds<ST>,
    SH: Shape,
    ST: Tuple + Shape,
{
    fn leaves_in_bounds(&self, shape: &Til<SH, ST>) -> bool {
        self.0.leaves_in_bounds(&shape.0) && self.1.leaves_in_bounds(&shape.1)
    }
}

const SHAPE_IS_STATIC_ZERO<S: Shape>: bool = S::STATIC_HAS_ZERO;

/// Dispatches scalar `InBounds` for shape `S`: `STATIC_ZERO = true` short-circuits to `false` without dividing by 0.
#[doc(hidden)]
pub const trait ScalarInBounds<S: Shape, const STATIC_ZERO: bool>: Integer {
    fn scalar_in_bounds(self, shape: &S) -> bool;
}

const impl<I: [const] Integer, S: Shape> ScalarInBounds<S, true> for I {
    fn scalar_in_bounds(self, _: &S) -> bool {
        false
    }
}

const impl<I, S> ScalarInBounds<S, false> for I
where
    I: [const] Integer + [const] IndexToCoord<S>,
    S: [const] Shape,
    I::Output: [const] LeavesInBounds<S>,
{
    fn scalar_in_bounds(self, shape: &S) -> bool {
        if shape.has_zero() {
            false
        } else {
            self.index_to_coord(shape).leaves_in_bounds(shape)
        }
    }
}

/// Checks bounds of a sorted `BasisTuple` against `S` starting at `MODE` (`Int<0>` for absent modes).
#[doc(hidden)]
pub const trait BasisTermsInBounds<const MODE: int, S: Shape>: FlatTuple {
    fn basis_terms_in_bounds(self, shape: &S) -> bool;
}

/// Single-step dispatch for `BasisTermsInBounds`: `MATCH = true` splits terms under `MODE` for `S`.
#[doc(hidden)]
pub const trait BasisTermsInBoundsStep<const MODE: int, S: Shape, const MATCH: bool>:
    FlatTuple
{
    fn basis_terms_in_bounds_step(self, shape: &S) -> bool;
}

const impl<const MODE: int> BasisTermsInBounds<MODE, Nil> for Nil {
    fn basis_terms_in_bounds(self, _: &Nil) -> bool {
        true
    }
}

const impl<S: [const] ShapeScalar> BasisTermsInBounds<0, S> for Nil
where
    Int<0>: [const] InBounds<S>,
{
    fn basis_terms_in_bounds(self, shape: &S) -> bool {
        Int::<0>.in_bounds(shape)
    }
}

const impl<const MODE: int, SH, ST> BasisTermsInBounds<MODE, Til<SH, ST>> for Nil
where
    SH: [const] Shape,
    ST: Tuple + [const] Shape,
    Int<0>: [const] InBounds<SH>,
    Self: [const] BasisTermsInBounds<{ INCREASE::<MODE> }, ST>,
{
    fn basis_terms_in_bounds(self, shape: &Til<SH, ST>) -> bool {
        Int::<0>.in_bounds(&shape.0) && self.basis_terms_in_bounds(&shape.1)
    }
}

const impl<V, S> BasisTermsInBounds<0, S> for Til<BasisTerm<V, ModeEnd>, Nil>
where
    V: [const] BasisCoeff + [const] InBounds<S>,
    S: [const] Shape,
    BasisTerm<V, ModeEnd>: [const] Leaf,
{
    fn basis_terms_in_bounds(self, shape: &S) -> bool {
        self.0.value.in_bounds(shape)
    }
}

const impl<const MODE: int, V, const M: int, P, Rest, SH, ST> BasisTermsInBounds<MODE, Til<SH, ST>>
    for Til<BasisTerm<V, Mode<M, P>>, Rest>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    Mode<M, P>: [const] BasisPath,
    Rest: FlatTuple,
    SH: [const] Shape,
    ST: Tuple + [const] Shape,
    BasisTerm<V, Mode<M, P>>: Item,
    Self: [const] BasisTermsInBoundsStep<MODE, Til<SH, ST>, { EQ_MODE::<M, MODE> }>,
{
    fn basis_terms_in_bounds(self, shape: &Til<SH, ST>) -> bool {
        self.basis_terms_in_bounds_step(shape)
    }
}

const impl<const MODE: int, V, const M: int, P, Rest, SH, ST>
    BasisTermsInBoundsStep<MODE, Til<SH, ST>, false> for Til<BasisTerm<V, Mode<M, P>>, Rest>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    Mode<M, P>: [const] BasisPath,
    Rest: FlatTuple,
    SH: [const] Shape,
    ST: Tuple + [const] Shape,
    BasisTerm<V, Mode<M, P>>: Item,
    Int<0>: [const] InBounds<SH>,
    Self: [const] BasisTermsInBounds<{ INCREASE::<MODE> }, ST>,
{
    fn basis_terms_in_bounds_step(self, shape: &Til<SH, ST>) -> bool {
        Int::<0>.in_bounds(&shape.0) && self.basis_terms_in_bounds(&shape.1)
    }
}

const impl<const MODE: int, V, const M: int, P, Rest, SH, ST>
    BasisTermsInBoundsStep<MODE, Til<SH, ST>, true> for Til<BasisTerm<V, Mode<M, P>>, Rest>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    Mode<M, P>: [const] BasisPath,
    Rest: FlatTuple,
    SH: [const] Shape,
    ST: Tuple + [const] Shape,
    BasisTerm<V, Mode<M, P>>: Item,
    Self: [const] SplitModePrefix<MODE>,
    <Self as SplitModePrefix<MODE>>::Head: [const] BasisTermsInBounds<0, SH>,
    <Self as SplitModePrefix<MODE>>::Tail: [const] BasisTermsInBounds<{ INCREASE::<MODE> }, ST>,
{
    fn basis_terms_in_bounds_step(self, shape: &Til<SH, ST>) -> bool {
        let (head, tail) = self.split_mode_prefix();
        head.basis_terms_in_bounds(&shape.0) && tail.basis_terms_in_bounds(&shape.1)
    }
}

/// Tests whether `self` is an in-bounds coordinate of `shape` ($0 \le c_i < s_i$ at every leaf).
///
/// Matches PyCuTe semantics: empty leaf sets (`Nil`, `(Nil, Nil)`, etc.) are vacuously `true`,
/// while any zero leaf extent in `shape` yields `false`.
pub const trait InBounds<S: Shape>: Copy {
    /// Returns `true` iff `self` lies within `shape`.
    fn in_bounds(self, shape: &S) -> bool;
}

const impl InBounds<Nil> for Nil {
    fn in_bounds(self, _: &Nil) -> bool {
        true
    }
}

const impl<I, S> InBounds<S> for I
where
    S: Shape,
    I: [const] ScalarInBounds<S, { SHAPE_IS_STATIC_ZERO::<S> }>,
{
    fn in_bounds(self, shape: &S) -> bool {
        self.scalar_in_bounds(shape)
    }
}

const impl<CH, CT, SH, ST> InBounds<Til<SH, ST>> for Til<CH, CT>
where
    CH: [const] InBounds<SH>,
    CT: Tuple + [const] InBounds<ST>,
    SH: Shape,
    ST: Tuple + Shape,
{
    fn in_bounds(self, shape: &Til<SH, ST>) -> bool {
        self.0.in_bounds(&shape.0) && self.1.in_bounds(&shape.1)
    }
}

const impl<V, P, S> InBounds<S> for BasisTerm<V, P>
where
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    S: [const] Shape,
    Nil: [const] EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>,
    <Nil as EmitBasisTerm<V, P, { COEFF_IS_STATIC_ZERO::<V> }>>::Output:
        [const] BasisTermsInBounds<0, S>,
{
    fn in_bounds(self, shape: &S) -> bool {
        Nil.emit(self.value).basis_terms_in_bounds(shape)
    }
}

const impl<T, S> InBounds<S> for BasisCombination<T>
where
    T: [const] BasisTuple + [const] BasisTermsInBounds<0, S>,
    S: [const] Shape,
{
    fn in_bounds(self, shape: &S) -> bool {
        self.terms().basis_terms_in_bounds(shape)
    }
}

/// Tests whether coordinate `coord` is in bounds for `shape`.
///
/// Any zero extent in `shape` yields `false`, negative coordinates yield `false`,
/// and scalar or basis coordinates are decomposed with [`idx2crd`] first.
pub const fn in_bounds<C: [const] InBounds<S>, S: Shape>(coord: C, shape: &S) -> bool {
    coord.in_bounds(shape)
}

/// Iterator over all natural coordinates of a [`Shape`] in colexicographical order.
#[derive(Clone, Debug)]
pub struct Coordinates<S: Shape> {
    shape: S,
    next: int,
    total: int,
}

/// Constructs an iterator yielding every natural coordinate of `shape` in colexicographical order.
///
/// Panics if the total size of `shape` overflows [`int`] or any dynamic extent is negative.
pub const fn coordinates<S: [const] Shape>(shape: S) -> Coordinates<S> {
    let total = shape.size();
    Coordinates {
        shape,
        next: 0,
        total,
    }
}

impl<S: Shape> Iterator for Coordinates<S>
where
    int: IndexToCoord<S>,
{
    type Item = <int as IndexToCoord<S>>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next < self.total {
            let idx = self.next;
            self.next += 1;
            Some(idx.index_to_coord(&self.shape))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = (self.total - self.next) as usize;
        (rem, Some(rem))
    }
}

impl<S: Shape> ExactSizeIterator for Coordinates<S> where int: IndexToCoord<S> {}
impl<S: Shape> FusedIterator for Coordinates<S> where int: IndexToCoord<S> {}
