//! Hierarchical coordinates: CuTe's `crd2idx`, `idx2crd`, `crd2crd`,
//! `elem_less`, and slicing with [`Underscore`] (`stride.hpp`,
//! `int_tuple.hpp`, `underscore.hpp`).
//!
//! These maps are unchecked device-side evaluation: they perform no bounds or
//! overflow checks and return typed integers. Static operands keep static
//! results and runtime leaves combine as described in [`Integer`]; overflow
//! panics in debug builds and wraps in release builds, and a zero divisor
//! panics. Validate layouts with [`Layout::new`] before evaluating them.
//!
//! A coordinate is an integer or a tuple of coordinates. An integer coordinate
//! for a tuple mode is decoded in left-major (colexicographic) order, and the
//! last mode takes the remaining quotient without a remainder, which is CuTe's
//! extended domain.
//!
//! ```
//! use noodle_cute::coord::crd2idx;
//! use noodle_cute::{Int, shape, stride};
//!
//! let shape = shape![Int::<2>, (Int::<2>, Int::<2>)];
//! let stride = stride![Int::<4>, (Int::<2>, Int::<1>)];
//! let nested: Int<5> = crd2idx(shape![Int::<1>, (Int::<0>, Int::<1>)], &shape, &stride);
//! assert_eq!(nested.value(), 5);
//! assert_eq!(crd2idx(shape![1i32, 2i32], &shape, &stride), 5);
//! assert_eq!(crd2idx(5i32, &shape, &stride), 5);
//! ```

use crate::Bool;
use crate::basis::{OffsetAdd, ScaledBasis};
use crate::int::{Arith, Int, Integer, Modulo, Op, Prod, Quot, Sum, modulo, prod, quot, sum};
use crate::layout::Layout;
use crate::tuple::{Append, IntTuple, Nil, Product};

/// CuTe's `_`: keeps a mode when slicing and acts as a static zero coordinate.
///
/// `_` is not a Rust expression, so use `Underscore` or its upstream alias [`X`].
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Underscore;

/// Upstream alias of [`Underscore`], also used to drop modes in `Step` projections.
pub use self::Underscore as X;

// Implementation traits that appear in public bounds but are not part of the API.
mod detail {
    use crate::int::Integer;

    pub const UNIT<S: Integer>: bool = matches!(S::STATIC, Some(1));

    /// `crd2idx_itt`: an integer coordinate for a tuple of modes.
    pub trait Itt<S, D> {
        type Output;
        fn itt(self, shape: &S, stride: &D) -> Self::Output;
    }

    /// `idx2crd` on one integer mode; a static unit extent gives a static zero.
    pub trait LeafIdx2Crd<I, S, D> {
        type Output: Integer;
        fn leaf(index: I, shape: S, stride: D) -> Self::Output;
    }

    pub trait LiftSlice<B> {
        type Output;
        fn lift_slice(self, target: &B) -> Self::Output;
    }

    pub trait LiftDice<B> {
        type Output;
        fn lift_dice(self, target: &B) -> Self::Output;
    }
}
use detail::{Itt, LeafIdx2Crd, LiftDice, LiftSlice, UNIT};

//-----------------------------------------------------------------------------
// crd2idx(coord, shape, stride)

/// Maps a coordinate within a shape and stride to an index: CuTe `crd2idx(c, s, d)`.
///
/// * integer coordinate, shape, and stride: `c * d`;
/// * tuple coordinate, shape, and stride of equal rank: the sum over modes;
/// * integer coordinate for a tuple shape: `c % size(s0)` maps into the first
///   mode and `c / size(s0)` into the remaining modes; the last mode receives
///   the whole quotient.
///
/// A tuple coordinate for an integer mode, or modes of different ranks, do not
/// implement this trait. [`Underscore`] is a static zero coordinate. Unlike
/// upstream, where an empty fold is ill-formed, the empty coordinate maps the
/// empty shape to `Int<0>`.
///
/// With [`ScaledBasis`] strides, as in an identity layout, the result is a
/// coordinate offset rather than an integer: `c * ScaledBasis<V, M>` is
/// `ScaledBasis<c * V, M>`, and contributions add mode by mode like CuTe's
/// `ArithmeticTuple`.
pub trait Crd2Idx<S, D> {
    /// The typed index or coordinate offset.
    type Output;
    /// Evaluates the index without checks.
    fn crd2idx(self, shape: &S, stride: &D) -> Self::Output;
}

impl<C, S, D> Crd2Idx<S, D> for C
where
    C: Integer + Arith<D, { Op::Mul }>,
    S: Integer,
    D: Integer,
{
    type Output = Prod<C, D>;
    fn crd2idx(self, _: &S, stride: &D) -> Self::Output {
        prod(self, *stride)
    }
}

impl<C, S, V, const M: usize> Crd2Idx<S, ScaledBasis<V, M>> for C
where
    C: Integer + Arith<V, { Op::Mul }>,
    S: Integer,
    V: Copy,
{
    type Output = ScaledBasis<Prod<C, V>, M>;
    fn crd2idx(self, _: &S, stride: &ScaledBasis<V, M>) -> Self::Output {
        ScaledBasis(prod(self, stride.0))
    }
}

impl<C, SH, ST, DH, DT> Crd2Idx<IntTuple<SH, ST>, IntTuple<DH, DT>> for C
where
    C: Integer + Itt<IntTuple<SH, ST>, IntTuple<DH, DT>>,
{
    type Output = <C as Itt<IntTuple<SH, ST>, IntTuple<DH, DT>>>::Output;
    fn crd2idx(self, shape: &IntTuple<SH, ST>, stride: &IntTuple<DH, DT>) -> Self::Output {
        self.itt(shape, stride)
    }
}

impl Crd2Idx<Nil, Nil> for Nil {
    type Output = Int<0>;
    fn crd2idx(self, _: &Nil, _: &Nil) -> Int<0> {
        Int
    }
}

impl<CH, CT, SH, ST, DH, DT> Crd2Idx<IntTuple<SH, ST>, IntTuple<DH, DT>> for IntTuple<CH, CT>
where
    CH: Crd2Idx<SH, DH>,
    CT: Crd2Idx<ST, DT>,
    CH::Output: OffsetAdd<CT::Output>,
{
    type Output = <CH::Output as OffsetAdd<CT::Output>>::Output;
    fn crd2idx(self, shape: &IntTuple<SH, ST>, stride: &IntTuple<DH, DT>) -> Self::Output {
        self.0
            .crd2idx(&shape.0, &stride.0)
            .offset_add(self.1.crd2idx(&shape.1, &stride.1))
    }
}

impl<S, D> Crd2Idx<S, D> for Underscore
where
    Int<0>: Crd2Idx<S, D>,
{
    type Output = <Int<0> as Crd2Idx<S, D>>::Output;
    fn crd2idx(self, shape: &S, stride: &D) -> Self::Output {
        Int::<0>.crd2idx(shape, stride)
    }
}

// The last mode is not divided, so the extended domain continues along it.
impl<C, SH, DH> Itt<IntTuple<SH, Nil>, IntTuple<DH, Nil>> for C
where
    C: Crd2Idx<SH, DH>,
{
    type Output = C::Output;
    fn itt(self, shape: &IntTuple<SH, Nil>, stride: &IntTuple<DH, Nil>) -> Self::Output {
        self.crd2idx(&shape.0, &stride.0)
    }
}

type ModeSize<S> = <S as Product>::Output;
type HeadIdx<C, SH, DH> = <Modulo<C, ModeSize<SH>> as Crd2Idx<SH, DH>>::Output;
type TailIdx<C, SH, S, D> = <Quot<C, ModeSize<SH>> as Itt<S, D>>::Output;

impl<C, SH, SN, ST, DH, DN, DT> Itt<IntTuple<SH, IntTuple<SN, ST>>, IntTuple<DH, IntTuple<DN, DT>>>
    for C
where
    C: Integer + Arith<ModeSize<SH>, { Op::Rem }> + Arith<ModeSize<SH>, { Op::Div }>,
    SH: Product + Clone,
    ModeSize<SH>: Integer,
    Modulo<C, ModeSize<SH>>: Crd2Idx<SH, DH>,
    Quot<C, ModeSize<SH>>: Itt<IntTuple<SN, ST>, IntTuple<DN, DT>>,
    HeadIdx<C, SH, DH>: OffsetAdd<TailIdx<C, SH, IntTuple<SN, ST>, IntTuple<DN, DT>>>,
{
    type Output = <HeadIdx<C, SH, DH> as OffsetAdd<
        TailIdx<C, SH, IntTuple<SN, ST>, IntTuple<DN, DT>>,
    >>::Output;
    fn itt(
        self,
        shape: &IntTuple<SH, IntTuple<SN, ST>>,
        stride: &IntTuple<DH, IntTuple<DN, DT>>,
    ) -> Self::Output {
        let size = shape.0.clone().product_typed();
        modulo(self, size)
            .crd2idx(&shape.0, &stride.0)
            .offset_add(quot(self, size).itt(&shape.1, &stride.1))
    }
}

/// Free-function form of [`Crd2Idx`], CuTe `crd2idx(coord, shape, stride)`.
pub fn crd2idx<C: Crd2Idx<S, D>, S, D>(coord: C, shape: &S, stride: &D) -> C::Output {
    coord.crd2idx(shape, stride)
}

//-----------------------------------------------------------------------------
// idx2crd(index, shape, stride)

/// Splits an index into a coordinate within a shape and stride: CuTe `idx2crd(i, s, d)`.
///
/// Each integer mode yields `(i / d) % s`, or a static zero when `s` is a static
/// one. Tuple shapes produce one coordinate per mode, and a tuple index splits
/// mode by mode. As upstream notes, this inverts only compact layouts. Unlike
/// upstream, the stride must have the same profile as the shape; the overload
/// that derives compact strides from an integer stride is not provided.
pub trait Idx2Crd<S, D> {
    /// The coordinate, congruent with the shape.
    type Output;
    /// Evaluates the coordinate without checks.
    fn idx2crd(self, shape: &S, stride: &D) -> Self::Output;
}

impl<I, S, D> Idx2Crd<S, D> for I
where
    I: Integer,
    S: Integer,
    D: Integer,
    Bool<{ UNIT::<S> }>: LeafIdx2Crd<I, S, D>,
{
    type Output = <Bool<{ UNIT::<S> }> as LeafIdx2Crd<I, S, D>>::Output;
    fn idx2crd(self, shape: &S, stride: &D) -> Self::Output {
        <Bool<{ UNIT::<S> }> as LeafIdx2Crd<I, S, D>>::leaf(self, *shape, *stride)
    }
}

impl<I, S, D> LeafIdx2Crd<I, S, D> for Bool<true> {
    type Output = Int<0>;
    fn leaf(_: I, _: S, _: D) -> Int<0> {
        Int
    }
}

impl<I, S, D> LeafIdx2Crd<I, S, D> for Bool<false>
where
    I: Arith<D, { Op::Div }>,
    Quot<I, D>: Arith<S, { Op::Rem }>,
{
    type Output = Modulo<Quot<I, D>, S>;
    fn leaf(index: I, shape: S, stride: D) -> Self::Output {
        modulo(quot(index, stride), shape)
    }
}

impl<I: Integer> Idx2Crd<Nil, Nil> for I {
    type Output = Nil;
    fn idx2crd(self, _: &Nil, _: &Nil) -> Nil {
        Nil
    }
}

impl<I, SH, ST, DH, DT> Idx2Crd<IntTuple<SH, ST>, IntTuple<DH, DT>> for I
where
    I: Integer + Idx2Crd<SH, DH> + Idx2Crd<ST, DT>,
{
    type Output = IntTuple<<I as Idx2Crd<SH, DH>>::Output, <I as Idx2Crd<ST, DT>>::Output>;
    fn idx2crd(self, shape: &IntTuple<SH, ST>, stride: &IntTuple<DH, DT>) -> Self::Output {
        IntTuple(
            Idx2Crd::<SH, DH>::idx2crd(self, &shape.0, &stride.0),
            Idx2Crd::<ST, DT>::idx2crd(self, &shape.1, &stride.1),
        )
    }
}

impl Idx2Crd<Nil, Nil> for Nil {
    type Output = Nil;
    fn idx2crd(self, _: &Nil, _: &Nil) -> Nil {
        Nil
    }
}

impl<IH, IT, SH, ST, DH, DT> Idx2Crd<IntTuple<SH, ST>, IntTuple<DH, DT>> for IntTuple<IH, IT>
where
    IH: Idx2Crd<SH, DH>,
    IT: Idx2Crd<ST, DT>,
{
    type Output = IntTuple<IH::Output, IT::Output>;
    fn idx2crd(self, shape: &IntTuple<SH, ST>, stride: &IntTuple<DH, DT>) -> Self::Output {
        IntTuple(
            self.0.idx2crd(&shape.0, &stride.0),
            self.1.idx2crd(&shape.1, &stride.1),
        )
    }
}

/// Free-function form of [`Idx2Crd`], CuTe `idx2crd(index, shape, stride)`.
pub fn idx2crd<I: Idx2Crd<S, D>, S, D>(index: I, shape: &S, stride: &D) -> I::Output {
    index.idx2crd(shape, stride)
}

//-----------------------------------------------------------------------------
// Colexicographic crd2idx(coord, shape) and idx2crd(index, shape)

/// Maps a coordinate to its colexicographic index within a shape: CuTe `crd2idx(c, s)`.
///
/// An integer coordinate is already an index. Otherwise the coordinate and the
/// shape are tuples of equal rank, and the index is Horner's
/// `c0 + size(s0) * (c1 + size(s1) * (...))`, recursing into nested
/// coordinates; the last coordinate is not bounded by its extent.
pub trait Crd2IdxColex<S> {
    /// The typed index.
    type Output: Integer;
    /// Evaluates the index without checks.
    fn crd2idx_colex(self, shape: &S) -> Self::Output;
}

impl<C: Integer, S> Crd2IdxColex<S> for C {
    type Output = C;
    fn crd2idx_colex(self, _: &S) -> C {
        self
    }
}

impl<CH, SH> Crd2IdxColex<IntTuple<SH, Nil>> for IntTuple<CH, Nil>
where
    CH: Crd2IdxColex<SH>,
{
    type Output = CH::Output;
    fn crd2idx_colex(self, shape: &IntTuple<SH, Nil>) -> Self::Output {
        self.0.crd2idx_colex(&shape.0)
    }
}

type ColexTail<C, S> = <C as Crd2IdxColex<S>>::Output;

impl<CH, CN, CT, SH, SN, ST> Crd2IdxColex<IntTuple<SH, IntTuple<SN, ST>>>
    for IntTuple<CH, IntTuple<CN, CT>>
where
    CH: Crd2IdxColex<SH>,
    SH: Product + Clone,
    IntTuple<CN, CT>: Crd2IdxColex<IntTuple<SN, ST>>,
    ModeSize<SH>: Arith<ColexTail<IntTuple<CN, CT>, IntTuple<SN, ST>>, { Op::Mul }>,
    CH::Output:
        Arith<Prod<ModeSize<SH>, ColexTail<IntTuple<CN, CT>, IntTuple<SN, ST>>>, { Op::Add }>,
{
    type Output =
        Sum<CH::Output, Prod<ModeSize<SH>, ColexTail<IntTuple<CN, CT>, IntTuple<SN, ST>>>>;
    fn crd2idx_colex(self, shape: &IntTuple<SH, IntTuple<SN, ST>>) -> Self::Output {
        sum(
            self.0.crd2idx_colex(&shape.0),
            prod(
                shape.0.clone().product_typed(),
                self.1.crd2idx_colex(&shape.1),
            ),
        )
    }
}

/// Free-function form of [`Crd2IdxColex`], CuTe `crd2idx(coord, shape)`.
pub fn crd2idx_colex<C: Crd2IdxColex<S>, S>(coord: C, shape: &S) -> C::Output {
    coord.crd2idx_colex(shape)
}

/// Splits an index into a colexicographic coordinate within a shape: CuTe `idx2crd(i, s)`.
///
/// An integer shape returns the index unchanged. For a tuple shape the first
/// mode receives `i % size(s0)`, decoded recursively, and the remaining modes
/// receive `i / size(s0)`; the last mode takes the whole quotient, so indices
/// beyond the size extend along it. A tuple index splits mode by mode.
pub trait Idx2CrdColex<S> {
    /// The coordinate, congruent with the shape.
    type Output;
    /// Evaluates the coordinate without checks.
    fn idx2crd_colex(self, shape: &S) -> Self::Output;
}

impl<I: Integer, S: Integer> Idx2CrdColex<S> for I {
    type Output = I;
    fn idx2crd_colex(self, _: &S) -> I {
        self
    }
}

impl<I, SH> Idx2CrdColex<IntTuple<SH, Nil>> for I
where
    I: Integer + Idx2CrdColex<SH>,
{
    type Output = IntTuple<I::Output, Nil>;
    fn idx2crd_colex(self, shape: &IntTuple<SH, Nil>) -> Self::Output {
        IntTuple(self.idx2crd_colex(&shape.0), Nil)
    }
}

impl<I, SH, SN, ST> Idx2CrdColex<IntTuple<SH, IntTuple<SN, ST>>> for I
where
    I: Integer + Arith<ModeSize<SH>, { Op::Rem }> + Arith<ModeSize<SH>, { Op::Div }>,
    SH: Product + Clone,
    ModeSize<SH>: Integer,
    Modulo<I, ModeSize<SH>>: Idx2CrdColex<SH>,
    Quot<I, ModeSize<SH>>: Idx2CrdColex<IntTuple<SN, ST>>,
{
    type Output = IntTuple<
        <Modulo<I, ModeSize<SH>> as Idx2CrdColex<SH>>::Output,
        <Quot<I, ModeSize<SH>> as Idx2CrdColex<IntTuple<SN, ST>>>::Output,
    >;
    fn idx2crd_colex(self, shape: &IntTuple<SH, IntTuple<SN, ST>>) -> Self::Output {
        let size = shape.0.clone().product_typed();
        IntTuple(
            modulo(self, size).idx2crd_colex(&shape.0),
            quot(self, size).idx2crd_colex(&shape.1),
        )
    }
}

impl Idx2CrdColex<Nil> for Nil {
    type Output = Nil;
    fn idx2crd_colex(self, _: &Nil) -> Nil {
        Nil
    }
}

impl<IH, IT, SH, ST> Idx2CrdColex<IntTuple<SH, ST>> for IntTuple<IH, IT>
where
    IH: Idx2CrdColex<SH>,
    IT: Idx2CrdColex<ST>,
{
    type Output = IntTuple<IH::Output, IT::Output>;
    fn idx2crd_colex(self, shape: &IntTuple<SH, ST>) -> Self::Output {
        IntTuple(
            self.0.idx2crd_colex(&shape.0),
            self.1.idx2crd_colex(&shape.1),
        )
    }
}

/// Free-function form of [`Idx2CrdColex`], CuTe `idx2crd(index, shape)`.
pub fn idx2crd_colex<I: Idx2CrdColex<S>, S>(index: I, shape: &S) -> I::Output {
    index.idx2crd_colex(shape)
}

//-----------------------------------------------------------------------------
// crd2crd

/// Converts a coordinate between two shapes of equal size: CuTe `crd2crd(c, src, dst)`.
///
/// When the coordinate and both shapes are tuples of equal rank, each mode is
/// converted separately; otherwise the result is
/// `idx2crd_colex(crd2idx_colex(c, src), dst)`. The sizes are not checked.
///
/// ```
/// use noodle_cute::coord::crd2crd;
/// use noodle_cute::{Int, shape};
///
/// let src = shape![Int::<2>, Int::<6>];
/// let dst = shape![Int::<2>, (Int::<2>, Int::<3>)];
/// let c = crd2crd(shape![1i32, 5i32], &src, &dst);
/// assert_eq!(c, shape![1i32, (1i32, 2i32)]);
/// ```
pub trait Crd2Crd<S, D> {
    /// The coordinate within `D`.
    type Output;
    /// Converts the coordinate without checks.
    fn crd2crd(self, src: &S, dst: &D) -> Self::Output;
}

impl<C, S, D> Crd2Crd<S, D> for C
where
    C: Integer + Idx2CrdColex<D>,
{
    type Output = C::Output;
    fn crd2crd(self, _: &S, dst: &D) -> Self::Output {
        self.idx2crd_colex(dst)
    }
}

impl<CH, CT, SH, ST, D> Crd2Crd<IntTuple<SH, ST>, D> for IntTuple<CH, CT>
where
    Self: Crd2IdxColex<IntTuple<SH, ST>>,
    D: Integer,
{
    type Output = ColexTail<Self, IntTuple<SH, ST>>;
    fn crd2crd(self, src: &IntTuple<SH, ST>, _: &D) -> Self::Output {
        self.crd2idx_colex(src)
    }
}

impl Crd2Crd<Nil, Nil> for Nil {
    type Output = Nil;
    fn crd2crd(self, _: &Nil, _: &Nil) -> Nil {
        Nil
    }
}

impl<CH, CT, SH, ST, DH, DT> Crd2Crd<IntTuple<SH, ST>, IntTuple<DH, DT>> for IntTuple<CH, CT>
where
    CH: Crd2Crd<SH, DH>,
    CT: Crd2Crd<ST, DT>,
{
    type Output = IntTuple<CH::Output, CT::Output>;
    fn crd2crd(self, src: &IntTuple<SH, ST>, dst: &IntTuple<DH, DT>) -> Self::Output {
        IntTuple(
            self.0.crd2crd(&src.0, &dst.0),
            self.1.crd2crd(&src.1, &dst.1),
        )
    }
}

/// Free-function form of [`Crd2Crd`], CuTe `crd2crd(coord, src_shape, dst_shape)`.
pub fn crd2crd<C: Crd2Crd<S, D>, S, D>(coord: C, src: &S, dst: &D) -> C::Output {
    coord.crd2crd(src, dst)
}

//-----------------------------------------------------------------------------
// elem_less

/// Elementwise "all less than" comparison: CuTe `elem_less(a, b)`.
///
/// Integers compare with `<`. Tuples compare mode by mode; modes of `a` beyond
/// the rank of `b` make the result false, while extra modes of `b` are ignored.
/// The result is a runtime `bool` even for static operands.
///
/// ```
/// use noodle_cute::coord::elem_less;
/// use noodle_cute::{Int, shape};
///
/// let extent = shape![Int::<4>, (Int::<2>, 3i32)];
/// assert!(elem_less(&shape![3i32, (1i32, 2i32)], &extent));
/// assert!(!elem_less(&shape![3i32, (1i32, 3i32)], &extent));
/// ```
pub trait ElemLess<B> {
    /// Returns whether every mode of `self` is less than the matching mode of `rhs`.
    fn elem_less(&self, rhs: &B) -> bool;
}

impl<A: Integer, B: Integer> ElemLess<B> for A {
    fn elem_less(&self, rhs: &B) -> bool {
        self.value() < rhs.value()
    }
}

impl ElemLess<Nil> for Nil {
    fn elem_less(&self, _: &Nil) -> bool {
        true
    }
}

impl<BH, BT> ElemLess<IntTuple<BH, BT>> for Nil {
    fn elem_less(&self, _: &IntTuple<BH, BT>) -> bool {
        true
    }
}

impl<AH, AT> ElemLess<Nil> for IntTuple<AH, AT> {
    fn elem_less(&self, _: &Nil) -> bool {
        false
    }
}

impl<AH, AT, BH, BT> ElemLess<IntTuple<BH, BT>> for IntTuple<AH, AT>
where
    AH: ElemLess<BH>,
    AT: ElemLess<BT>,
{
    fn elem_less(&self, rhs: &IntTuple<BH, BT>) -> bool {
        self.0.elem_less(&rhs.0) && self.1.elem_less(&rhs.1)
    }
}

/// Free-function form of [`ElemLess`].
pub fn elem_less<A: ElemLess<B>, B>(a: &A, b: &B) -> bool {
    a.elem_less(b)
}

//-----------------------------------------------------------------------------
// slice and dice

/// Keeps the modes of a target selected by [`Underscore`]: CuTe `slice(coord, target)`.
///
/// `slice(_, b)` is `b` and `slice(i, b)` is the empty tuple for an integer `i`.
/// For tuples, each `_` keeps its mode, each integer drops it, and nested
/// coordinate tuples splice their kept modes into the enclosing result. The
/// target may hold any cloneable elements, such as integers or layouts.
pub trait SliceOf<B> {
    /// The kept modes.
    type Output;
    /// Returns the kept modes of `target`.
    fn slice_of(self, target: &B) -> Self::Output;
}

/// Keeps the modes of a target selected by integers: CuTe `dice(coord, target)`.
///
/// `dice(i, b)` is `b` for an integer `i` and `dice(_, b)` is the empty tuple.
/// This is the complement of [`SliceOf`]. With a `Step` such as
/// `(Int<1>, X, Int<1>)` it projects a tiler or coordinate onto the modes marked
/// by integers.
pub trait DiceOf<B> {
    /// The kept modes.
    type Output;
    /// Returns the kept modes of `target`.
    fn dice_of(self, target: &B) -> Self::Output;
}

impl<B: Clone> SliceOf<B> for Underscore {
    type Output = B;
    fn slice_of(self, target: &B) -> B {
        target.clone()
    }
}

impl<C: Integer, B> SliceOf<B> for C {
    type Output = Nil;
    fn slice_of(self, _: &B) -> Nil {
        Nil
    }
}

impl SliceOf<Nil> for Nil {
    type Output = Nil;
    fn slice_of(self, _: &Nil) -> Nil {
        Nil
    }
}

impl<CH, CT, BH, BT> SliceOf<IntTuple<BH, BT>> for IntTuple<CH, CT>
where
    CH: LiftSlice<BH>,
    CT: SliceOf<BT>,
    CH::Output: Append<CT::Output>,
{
    type Output = <CH::Output as Append<CT::Output>>::Output;
    fn slice_of(self, target: &IntTuple<BH, BT>) -> Self::Output {
        self.0
            .lift_slice(&target.0)
            .append(self.1.slice_of(&target.1))
    }
}

impl<B: Clone> LiftSlice<B> for Underscore {
    type Output = IntTuple<B, Nil>;
    fn lift_slice(self, target: &B) -> Self::Output {
        IntTuple(target.clone(), Nil)
    }
}

impl<C: Integer, B> LiftSlice<B> for C {
    type Output = Nil;
    fn lift_slice(self, _: &B) -> Nil {
        Nil
    }
}

impl LiftSlice<Nil> for Nil {
    type Output = Nil;
    fn lift_slice(self, _: &Nil) -> Nil {
        Nil
    }
}

impl<CH, CT, BH, BT> LiftSlice<IntTuple<BH, BT>> for IntTuple<CH, CT>
where
    Self: SliceOf<IntTuple<BH, BT>>,
{
    type Output = <Self as SliceOf<IntTuple<BH, BT>>>::Output;
    fn lift_slice(self, target: &IntTuple<BH, BT>) -> Self::Output {
        self.slice_of(target)
    }
}

impl<B> DiceOf<B> for Underscore {
    type Output = Nil;
    fn dice_of(self, _: &B) -> Nil {
        Nil
    }
}

impl<C: Integer, B: Clone> DiceOf<B> for C {
    type Output = B;
    fn dice_of(self, target: &B) -> B {
        target.clone()
    }
}

impl DiceOf<Nil> for Nil {
    type Output = Nil;
    fn dice_of(self, _: &Nil) -> Nil {
        Nil
    }
}

impl<CH, CT, BH, BT> DiceOf<IntTuple<BH, BT>> for IntTuple<CH, CT>
where
    CH: LiftDice<BH>,
    CT: DiceOf<BT>,
    CH::Output: Append<CT::Output>,
{
    type Output = <CH::Output as Append<CT::Output>>::Output;
    fn dice_of(self, target: &IntTuple<BH, BT>) -> Self::Output {
        self.0
            .lift_dice(&target.0)
            .append(self.1.dice_of(&target.1))
    }
}

impl<B> LiftDice<B> for Underscore {
    type Output = Nil;
    fn lift_dice(self, _: &B) -> Nil {
        Nil
    }
}

impl<C: Integer, B: Clone> LiftDice<B> for C {
    type Output = IntTuple<B, Nil>;
    fn lift_dice(self, target: &B) -> Self::Output {
        IntTuple(target.clone(), Nil)
    }
}

impl LiftDice<Nil> for Nil {
    type Output = Nil;
    fn lift_dice(self, _: &Nil) -> Nil {
        Nil
    }
}

impl<CH, CT, BH, BT> LiftDice<IntTuple<BH, BT>> for IntTuple<CH, CT>
where
    Self: DiceOf<IntTuple<BH, BT>>,
{
    type Output = <Self as DiceOf<IntTuple<BH, BT>>>::Output;
    fn lift_dice(self, target: &IntTuple<BH, BT>) -> Self::Output {
        self.dice_of(target)
    }
}

/// Free-function form of [`SliceOf`], CuTe `slice(coord, target)`.
///
/// ```
/// use noodle_cute::coord::{X, slice};
/// use noodle_cute::{Int, shape};
///
/// let s = shape![(Int::<2>, Int::<3>), Int::<4>];
/// let kept = slice(shape![(X, 1i32), X], &s);
/// assert_eq!(kept, shape![Int::<2>, Int::<4>]);
/// ```
pub fn slice<C: SliceOf<B>, B>(coord: C, target: &B) -> C::Output {
    coord.slice_of(target)
}

/// Free-function form of [`DiceOf`], CuTe `dice(coord, target)`.
///
/// ```
/// use noodle_cute::coord::{X, dice};
/// use noodle_cute::{Int, shape};
///
/// let tiler = shape![Int::<8>, Int::<16>, Int::<4>];
/// let mn = dice(shape![Int::<1>, Int::<1>, X], &tiler);
/// assert_eq!(mn, shape![Int::<8>, Int::<16>]);
/// ```
pub fn dice<C: DiceOf<B>, B>(coord: C, target: &B) -> C::Output {
    coord.dice_of(target)
}

//-----------------------------------------------------------------------------
// Layout entry points

impl<S, D> Layout<S, D> {
    /// Maps a coordinate to an address without checks: CuTe `layout(coord)`.
    ///
    /// Accepts an integer or a hierarchical coordinate as described by
    /// [`Crd2Idx`]. Static leaves fold to constants, and the result keeps a
    /// static type when every contributing operand is static. Here the runtime
    /// stride makes the result a `DynInt`; with `i32` coordinates and static
    /// strides it would be an `i32`.
    ///
    /// ```
    /// use noodle_cute::{DynInt, Int, Layout, shape, stride};
    ///
    /// let layout = Layout::new(
    ///     shape![Int::<2>, (Int::<2>, Int::<2>)],
    ///     stride![DynInt(4), (Int::<2>, Int::<1>)],
    /// )?;
    /// assert_eq!(layout.crd2idx(shape![1i32, (0i32, 1i32)]), DynInt(5));
    /// assert_eq!(layout.crd2idx(5i32), DynInt(5));
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn crd2idx<C: Crd2Idx<S, D>>(&self, coord: C) -> C::Output {
        coord.crd2idx(&self.shape, &self.stride)
    }

    /// Keeps the modes selected by [`Underscore`]: CuTe `slice(coord, layout)`.
    ///
    /// The result is not validated; see [`SliceOf`] for the selection rules.
    pub fn slice<C>(&self, coord: C) -> Sliced<C, S, D>
    where
        C: SliceOf<S> + SliceOf<D> + Clone,
    {
        Layout {
            shape: SliceOf::<S>::slice_of(coord.clone(), &self.shape),
            stride: SliceOf::<D>::slice_of(coord, &self.stride),
        }
    }

    /// Keeps the modes selected by integers: CuTe `dice(coord, layout)`.
    ///
    /// The result is not validated; see [`DiceOf`] for the selection rules.
    pub fn dice<C>(&self, coord: C) -> Layout<<C as DiceOf<S>>::Output, <C as DiceOf<D>>::Output>
    where
        C: DiceOf<S> + DiceOf<D> + Clone,
    {
        Layout {
            shape: DiceOf::<S>::dice_of(coord.clone(), &self.shape),
            stride: DiceOf::<D>::dice_of(coord, &self.stride),
        }
    }

    /// Returns the slice and the address of its origin: CuTe `slice_and_offset`.
    ///
    /// Each [`Underscore`] contributes a static zero to the offset, so the
    /// original address is `offset + slice.crd2idx(u)` for any coordinate `u`
    /// of the kept modes.
    ///
    /// ```
    /// use noodle_cute::coord::X;
    /// use noodle_cute::{Int, Layout, shape, stride};
    ///
    /// let layout = Layout::new(shape![Int::<3>, Int::<4>], stride![Int::<1>, Int::<3>])?;
    /// let (column, offset) = layout.slice_and_offset(shape![X, 2i32]);
    /// assert_eq!(column, Layout { shape: shape![Int::<3>], stride: stride![Int::<1>] });
    /// assert_eq!(offset, 6);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn slice_and_offset<C>(&self, coord: C) -> (Sliced<C, S, D>, <C as Crd2Idx<S, D>>::Output)
    where
        C: SliceOf<S> + SliceOf<D> + Crd2Idx<S, D> + Clone,
    {
        (self.slice(coord.clone()), self.crd2idx(coord))
    }
}

type Sliced<C, S, D> = Layout<<C as SliceOf<S>>::Output, <C as SliceOf<D>>::Output>;

//-----------------------------------------------------------------------------
// Partition coordinates

// Implementation traits for the coordinates of `inner_partition` and `outer_partition`.
mod partition {
    use super::Underscore;
    use crate::int::Integer;
    use crate::tuple::{IntTuple, Nil};

    /// CuTe `repeat<rank(S)>(v)`: `v` for one mode, otherwise one `v` per mode.
    pub trait RepeatModes<V> {
        type Output: Clone;
        fn repeat_modes(value: V) -> Self::Output;
    }

    /// One `v` per mode of a tuple chain.
    pub trait ChainModes<V> {
        type Output: Clone;
        fn chain_modes(value: V) -> Self::Output;
    }

    /// CuTe `append<rank(S)>(coord, _)` for a tuple coordinate; integers and `_` pass through.
    pub trait PadModes<S> {
        type Output: Clone;
        fn pad_modes(self) -> Self::Output;
    }

    impl<V: Clone> ChainModes<V> for Nil {
        type Output = Nil;
        fn chain_modes(_: V) -> Nil {
            Nil
        }
    }

    impl<V: Clone, H, T: ChainModes<V>> ChainModes<V> for IntTuple<H, T> {
        type Output = IntTuple<V, T::Output>;
        fn chain_modes(value: V) -> Self::Output {
            IntTuple(value.clone(), T::chain_modes(value))
        }
    }

    impl<V: Clone, I: Integer> RepeatModes<V> for I {
        type Output = V;
        fn repeat_modes(value: V) -> V {
            value
        }
    }

    impl<V: Clone> RepeatModes<V> for Nil {
        type Output = Nil;
        fn repeat_modes(_: V) -> Nil {
            Nil
        }
    }

    impl<V: Clone, H> RepeatModes<V> for IntTuple<H, Nil> {
        type Output = V;
        fn repeat_modes(value: V) -> V {
            value
        }
    }

    impl<V: Clone, H, TH, TT> RepeatModes<V> for IntTuple<H, IntTuple<TH, TT>>
    where
        Self: ChainModes<V>,
    {
        type Output = <Self as ChainModes<V>>::Output;
        fn repeat_modes(value: V) -> Self::Output {
            <Self as ChainModes<V>>::chain_modes(value)
        }
    }

    impl<C: Integer, S> PadModes<S> for C {
        type Output = C;
        fn pad_modes(self) -> C {
            self
        }
    }

    impl<S> PadModes<S> for Underscore {
        type Output = Underscore;
        fn pad_modes(self) -> Underscore {
            self
        }
    }

    impl<S: ChainModes<Underscore>> PadModes<S> for Nil {
        type Output = S::Output;
        fn pad_modes(self) -> S::Output {
            S::chain_modes(Underscore)
        }
    }

    impl<CH: Clone, CT, SH, ST> PadModes<IntTuple<SH, ST>> for IntTuple<CH, CT>
    where
        CT: PadModes<ST>,
    {
        type Output = IntTuple<CH, CT::Output>;
        fn pad_modes(self) -> Self::Output {
            IntTuple(self.0, self.1.pad_modes())
        }
    }

    /// The coordinate slicing a zipped `(tile, rest)` layout for `inner_partition`.
    pub trait InnerCoord<C> {
        type Output: Clone;
        fn inner_coord(coord: C) -> Self::Output;
    }

    /// The coordinate slicing a zipped `(tile, rest)` layout for `outer_partition`.
    pub trait OuterCoord<C> {
        type Output: Clone;
        fn outer_coord(coord: C) -> Self::Output;
    }

    impl<S0, S1, C> InnerCoord<C> for IntTuple<S0, IntTuple<S1, Nil>>
    where
        S0: RepeatModes<Underscore>,
        C: PadModes<S1>,
    {
        type Output = IntTuple<S0::Output, IntTuple<C::Output, Nil>>;
        fn inner_coord(coord: C) -> Self::Output {
            IntTuple(
                S0::repeat_modes(Underscore),
                IntTuple(coord.pad_modes(), Nil),
            )
        }
    }

    impl<S0, S1, C> OuterCoord<C> for IntTuple<S0, IntTuple<S1, Nil>>
    where
        C: PadModes<S0>,
        S1: RepeatModes<Underscore>,
    {
        type Output = IntTuple<C::Output, IntTuple<S1::Output, Nil>>;
        fn outer_coord(coord: C) -> Self::Output {
            IntTuple(
                coord.pad_modes(),
                IntTuple(S1::repeat_modes(Underscore), Nil),
            )
        }
    }
}
pub(crate) use partition::{InnerCoord, OuterCoord, PadModes, RepeatModes};

type FlatCoord<I, S, D> =
    <<I as Idx2Crd<S, D>>::Output as Crd2Crd<S, <S as RepeatModes<Int<1>>>::Output>>::Output;

impl<S, D> Layout<S, D> {
    /// The flat coordinate of index `index` in a compact layout: CuTe
    /// `Layout::get_flat_coord(idx)`.
    ///
    /// Splits the index with `idx2crd(index, shape, stride)` and then reduces
    /// each mode to its colexicographic index, giving one integer per top-level
    /// mode, or one integer for a single mode. Only compact layouts are inverted.
    ///
    /// ```
    /// use noodle_cute::{Int, Layout, shape, stride};
    ///
    /// let threads = Layout::new(shape![(Int::<2>, Int::<4>), Int::<16>], stride![(Int::<16>, Int::<32>), Int::<1>])?;
    /// // 37 = 1 * 32 + 5 is the hierarchical coordinate ((0, 1), 5), flattened to (0 + 2 * 1, 5).
    /// assert_eq!(threads.get_flat_coord(37i32), shape![2i32, 5i32]);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn get_flat_coord<I>(&self, index: I) -> FlatCoord<I, S, D>
    where
        I: Idx2Crd<S, D>,
        S: RepeatModes<Int<1>>,
        I::Output: Crd2Crd<S, <S as RepeatModes<Int<1>>>::Output>,
    {
        index
            .idx2crd(&self.shape, &self.stride)
            .crd2crd(&self.shape, &S::repeat_modes(Int))
    }
}
