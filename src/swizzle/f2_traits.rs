//! Integrations connecting [`F2`] to `tuple`, `basis`, `shape`, `stride`, `layout`, and `algebra`.

use core::cmp::Ordering;
use core::ops::{Add, Mul, Sub};

use crate::algebra::composition::AtomCompose;
use crate::algebra::nullspace::NullspaceStrideLeaf;
use crate::basis::ops::{BasisAdd, BasisMul, BasisSub};
use crate::basis::{
    AsTuple, BasisCoeff, BasisCoeffAdd, BasisCoeffEq, BasisCoeffMul, BasisCoeffNeg, BasisCoeffOrd,
    BasisCoeffSub, BasisCombination, BasisPath, BasisTerm, BasisTuple, CoeffAlgebra,
    IntoBasisCombination, ModeEnd, ScaledBasis,
};
use crate::integer::ops::{bit_length_non_neg, checked_clmul};
use crate::integer::{
    ArithMul, BOTH_STATIC, IS_NON_NEG, Int, Integer, Product, cmp, eq, int, lt, mul, shiftl,
    static_cmp, static_cmp_of, static_eq, static_eq_of, static_true,
};
use crate::layout::order::OrderKey;
use crate::layout::recast::{MakeStaticRecastStride, RecastStrideScalar};
use crate::layout::tiler::TilerFactor;
use crate::layout::{IsLayout, Layout, LayoutItemEq};
use crate::shape::{
    CoordToIndex, InBounds, IndexToCoord, LeavesInBounds, Shape, ShapeScalar, ValidShapeExtent,
};
use crate::stride::{
    Accumulate, CoshapeAccumulate, CoshapeCoeffAccum, InnerProduct, OffsetExtent,
    ScaleBasisFunctor, Stride, StrideMul, StrideScalar, StrideSkeleton,
};
use crate::tuple::{Item, Leaf, LeafProfile, Nil, Til, TransformLeafFunctor, Tuple};

use super::f2::{
    F2, F2_DIV_DISPATCH, F2_MUL_DISPATCH, F2DivModOp, F2MulOp, F2OrOutput, F2OrScalar, F2Product,
    F2Quotient, F2Remainder, F2Scalar, F2XorOutput, F2XorScalar, ValidF2Value, f2_add, f2_divmod,
    f2_mul,
};

// ------------------------------------------------------------------
// tuple::Item
// ------------------------------------------------------------------

const impl<V: [const] F2Scalar> Item for F2<V> {
    const RANK: int = 1;
    const DEPTH: int = 0;
    const LEAVES: int = 1;
    const IS_STATIC: bool = V::IS_STATIC;
    type Profile = LeafProfile;
}

// ------------------------------------------------------------------
// basis::BasisCoeff & basis algebra/comparison/conversion traits
// ------------------------------------------------------------------

const impl<V: [const] F2Scalar> BasisCoeff for F2<V> {
    const IS_STATIC_ZERO: bool = static_true(static_eq_of::<V, Int<0>>());
    const IS_STATIC: bool = V::IS_STATIC;
    const STATIC_CMP_ZERO: Option<Ordering> = static_cmp_of::<V, Int<0>>();
    const STATIC_VALUE: Option<int> = V::STATIC_VALUE;
    const ALGEBRA: CoeffAlgebra = CoeffAlgebra::F2;
    type Unit = F2<Int<1>>;

    fn is_zero(&self) -> bool {
        eq(self.0, Int::<0>)
    }

    fn cmp_zero(&self) -> Ordering {
        cmp(self.0, Int::<0>)
    }

    fn static_cmp_zero(&self) -> Option<Ordering> {
        static_cmp(self.0, Int::<0>)
    }

    fn algebra_unit() -> Self::Unit {
        F2(Int)
    }
}

const impl<L, R> BasisCoeffAdd<F2<R>> for F2<L>
where
    L: [const] F2Scalar + [const] F2XorScalar<R, { BOTH_STATIC::<L, R> }>,
    R: [const] F2Scalar,
{
    type Output = F2<F2XorOutput<L, R>>;
    fn coeff_add(self, rhs: F2<R>) -> Self::Output {
        f2_add(self, rhs)
    }
}

const impl<L: [const] F2Scalar> BasisCoeffAdd<Int<0>> for F2<L> {
    type Output = Self;
    fn coeff_add(self, _: Int<0>) -> Self {
        self
    }
}

const impl<R: [const] F2Scalar> BasisCoeffAdd<F2<R>> for Int<0> {
    type Output = F2<R>;
    fn coeff_add(self, rhs: F2<R>) -> F2<R> {
        rhs
    }
}

const impl<V: [const] F2Scalar> BasisCoeffNeg for F2<V> {
    type Output = Self;
    fn coeff_neg(self) -> Self {
        self
    }
}

const impl<L, R> BasisCoeffSub<F2<R>> for F2<L>
where
    L: [const] F2Scalar + [const] F2XorScalar<R, { BOTH_STATIC::<L, R> }>,
    R: [const] F2Scalar,
{
    type Output = F2<F2XorOutput<L, R>>;
    fn coeff_sub(self, rhs: F2<R>) -> Self::Output {
        f2_add(self, rhs)
    }
}

const impl<L: [const] F2Scalar> BasisCoeffSub<Int<0>> for F2<L> {
    type Output = Self;
    fn coeff_sub(self, _: Int<0>) -> Self {
        self
    }
}

const impl<R: [const] F2Scalar> BasisCoeffSub<F2<R>> for Int<0> {
    type Output = F2<R>;
    fn coeff_sub(self, rhs: F2<R>) -> F2<R> {
        rhs
    }
}

const impl<L, R> BasisCoeffMul<F2<R>> for F2<L>
where
    L: [const] F2Scalar,
    R: [const] F2Scalar,
    F2<L>: [const] F2MulOp<F2<R>, { F2_MUL_DISPATCH::<F2<L>, F2<R>> }>,
{
    type Output = F2Product<F2<L>, F2<R>>;
    fn coeff_mul(self, rhs: F2<R>) -> Self::Output {
        f2_mul(self, rhs)
    }
}

const impl<L, R> BasisCoeffMul<R> for F2<L>
where
    L: [const] F2Scalar,
    R: [const] Integer + [const] super::f2::F2Operand,
    F2<L>: [const] F2MulOp<R, { F2_MUL_DISPATCH::<F2<L>, R> }>,
{
    type Output = F2Product<F2<L>, R>;
    fn coeff_mul(self, rhs: R) -> Self::Output {
        f2_mul(self, rhs)
    }
}

const impl<L, R> BasisCoeffMul<F2<R>> for L
where
    L: [const] Integer + [const] super::f2::F2Operand,
    R: [const] F2Scalar,
    L: [const] F2MulOp<F2<R>, { F2_MUL_DISPATCH::<L, F2<R>> }>,
{
    type Output = F2Product<L, F2<R>>;
    fn coeff_mul(self, rhs: F2<R>) -> Self::Output {
        f2_mul(self, rhs)
    }
}

const impl<L: [const] F2Scalar, R: [const] F2Scalar> BasisCoeffEq<F2<R>> for F2<L> {
    fn coeff_eq(&self, rhs: &F2<R>) -> bool {
        eq(self.0, rhs.0)
    }
    fn coeff_static_eq(&self, rhs: &F2<R>) -> Option<bool> {
        static_eq(self.0, rhs.0)
    }
}

const impl<L: [const] F2Scalar, R: [const] Integer> BasisCoeffEq<R> for F2<L> {
    fn coeff_eq(&self, rhs: &R) -> bool {
        eq(self.0, *rhs)
    }
    fn coeff_static_eq(&self, rhs: &R) -> Option<bool> {
        static_eq(self.0, *rhs)
    }
}

const impl<L: [const] Integer, R: [const] F2Scalar> BasisCoeffEq<F2<R>> for L {
    fn coeff_eq(&self, rhs: &F2<R>) -> bool {
        eq(*self, rhs.0)
    }
    fn coeff_static_eq(&self, rhs: &F2<R>) -> Option<bool> {
        static_eq(*self, rhs.0)
    }
}

const impl<L: [const] F2Scalar, R: [const] F2Scalar> BasisCoeffOrd<F2<R>> for F2<L> {
    const STATIC_CMP: Option<Ordering> = static_cmp_of::<L, R>();
    fn coeff_cmp(&self, rhs: &F2<R>) -> Ordering {
        cmp(self.0, rhs.0)
    }
    fn coeff_static_cmp(&self, rhs: &F2<R>) -> Option<Ordering> {
        static_cmp(self.0, rhs.0)
    }
}

const impl<L: [const] F2Scalar, R: [const] Integer> BasisCoeffOrd<R> for F2<L> {
    const STATIC_CMP: Option<Ordering> = static_cmp_of::<L, R>();
    fn coeff_cmp(&self, rhs: &R) -> Ordering {
        cmp(self.0, *rhs)
    }
    fn coeff_static_cmp(&self, rhs: &R) -> Option<Ordering> {
        static_cmp(self.0, *rhs)
    }
}

const impl<L: [const] Integer, R: [const] F2Scalar> BasisCoeffOrd<F2<R>> for L {
    const STATIC_CMP: Option<Ordering> = static_cmp_of::<L, R>();
    fn coeff_cmp(&self, rhs: &F2<R>) -> Ordering {
        cmp(*self, rhs.0)
    }
    fn coeff_static_cmp(&self, rhs: &F2<R>) -> Option<Ordering> {
        static_cmp(*self, rhs.0)
    }
}

const impl IntoBasisCombination for F2<Int<0>> {
    type Terms = Nil;
    fn into_basis_terms(self) -> Nil {
        Nil
    }
}

const impl<V: [const] BasisCoeff, P: [const] BasisPath> Add<BasisTerm<V, P>> for F2<Int<0>>
where
    Self: [const] BasisAdd<BasisTerm<V, P>>,
{
    type Output = <Self as BasisAdd<BasisTerm<V, P>>>::Output;
    fn add(self, rhs: BasisTerm<V, P>) -> Self::Output {
        self.basis_add(rhs)
    }
}

const impl<V: [const] BasisCoeff, P: [const] BasisPath> Sub<BasisTerm<V, P>> for F2<Int<0>>
where
    Self: [const] BasisSub<BasisTerm<V, P>>,
{
    type Output = <Self as BasisSub<BasisTerm<V, P>>>::Output;
    fn sub(self, rhs: BasisTerm<V, P>) -> Self::Output {
        self.basis_sub(rhs)
    }
}

const impl<T: [const] BasisTuple> Add<BasisCombination<T>> for F2<Int<0>>
where
    Self: [const] BasisAdd<BasisCombination<T>>,
{
    type Output = <Self as BasisAdd<BasisCombination<T>>>::Output;
    fn add(self, rhs: BasisCombination<T>) -> Self::Output {
        self.basis_add(rhs)
    }
}

const impl<T: [const] BasisTuple> Sub<BasisCombination<T>> for F2<Int<0>>
where
    Self: [const] BasisSub<BasisCombination<T>>,
{
    type Output = <Self as BasisSub<BasisCombination<T>>>::Output;
    fn sub(self, rhs: BasisCombination<T>) -> Self::Output {
        self.basis_sub(rhs)
    }
}

const impl<V: [const] F2Scalar> AsTuple for F2<V> {
    type Output = Self;
    fn as_tuple(self) -> Self {
        self
    }
}

const impl<L, V, P> Mul<BasisTerm<V, P>> for F2<L>
where
    L: [const] F2Scalar,
    V: [const] BasisCoeff,
    P: [const] BasisPath,
    Self: [const] BasisMul<BasisTerm<V, P>>,
{
    type Output = <Self as BasisMul<BasisTerm<V, P>>>::Output;
    fn mul(self, rhs: BasisTerm<V, P>) -> Self::Output {
        self.basis_mul(rhs)
    }
}

const impl<L, T> Mul<BasisCombination<T>> for F2<L>
where
    L: [const] F2Scalar,
    T: [const] BasisTuple,
    Self: [const] BasisMul<BasisCombination<T>>,
{
    type Output = <Self as BasisMul<BasisCombination<T>>>::Output;
    fn mul(self, rhs: BasisCombination<T>) -> Self::Output {
        self.basis_mul(rhs)
    }
}

// ------------------------------------------------------------------
// shape::coord (CoordToIndex, IndexToCoord with no-carry check, InBounds)
// ------------------------------------------------------------------

const impl<S: [const] Shape, V: [const] F2Scalar> CoordToIndex<S> for F2<V> {
    type Output = Self;
    fn coord_to_index(self, shape: &S) -> Self {
        let _ = shape.has_zero();
        self
    }
}

const HAS_NO_LEAVES<T: Item>: bool = T::LEAVES == 0;
const AND_BOOL<const A: bool, const B: bool>: bool = A && B;

const fn static_prefix_no_carry<Pps: Integer, S: ShapeScalar>() -> bool {
    match (Pps::STATIC_VALUE, S::STATIC_VALUE) {
        (Some(p), Some(s)) => {
            if p < 0 || s <= 0 {
                return false;
            }
            let Some(z_prod) = p.checked_mul(s) else {
                return false;
            };
            matches!(checked_clmul(p, s), Ok(f2_prod) if f2_prod == z_prod)
        }
        _ => true,
    }
}

const F2_PREFIX_NO_CARRY<Pps: Integer, S: ShapeScalar>: bool = static_prefix_no_carry::<Pps, S>();

/// Compile-time guard verifying that `F2(pps) * F2(s) == pps * s` during `idx2crd(F2, shape)`.
#[doc(hidden)]
pub const trait ValidF2PrefixProduct<const OK: bool> {}
const impl ValidF2PrefixProduct<true> for () {}

/// Colexicographical carry-less decomposition of an [`F2`] index across shape `S`,
/// threading remaining [`F2`] quotient and running $\mathbb{Z}$ prefix product `Pps`.
#[doc(hidden)]
pub const trait F2IndexToCoordCarry<S: Shape, Pps: [const] Integer, const IS_LAST: bool>:
    Copy
{
    type Output: Item;
    type RemScalar: [const] F2Scalar;
    type NextPps: [const] Integer;

    fn f2_index_to_coord_carry(
        self,
        shape: &S,
        pps: Pps,
    ) -> (Self::Output, F2<Self::RemScalar>, Self::NextPps);
}

const impl<V: [const] F2Scalar, Pps: [const] Integer, const IS_LAST: bool>
    F2IndexToCoordCarry<Nil, Pps, IS_LAST> for F2<V>
{
    type Output = Nil;
    type RemScalar = V;
    type NextPps = Pps;

    fn f2_index_to_coord_carry(self, _: &Nil, pps: Pps) -> (Nil, F2<V>, Pps) {
        (Nil, self, pps)
    }
}

const impl<S: [const] ShapeScalar, V: [const] F2Scalar, Pps: [const] Integer>
    F2IndexToCoordCarry<S, Pps, true> for F2<V>
{
    type Output = F2<V>;
    type RemScalar = Int<0>;
    type NextPps = Pps;

    fn f2_index_to_coord_carry(self, shape: &S, pps: Pps) -> (F2<V>, F2<Int<0>>, Pps) {
        let _ = shape.size();
        (self, F2(Int), pps)
    }
}

const impl<S, V, Pps> F2IndexToCoordCarry<S, Pps, false> for F2<V>
where
    S: [const] ShapeScalar + [const] super::f2::F2Operand,
    V: [const] F2Scalar,
    Pps: [const] Integer + [const] ArithMul<Pps, S>,
    (): [const] ValidF2PrefixProduct<{ F2_PREFIX_NO_CARRY::<Pps, S> }>,
    F2<V>: [const] F2DivModOp<S, { F2_DIV_DISPATCH::<V, S> }>,
{
    type Output = F2Remainder<V, S>;
    type RemScalar = <F2<V> as F2DivModOp<S, { F2_DIV_DISPATCH::<V, S> }>>::QuotScalar;
    type NextPps = Product<Pps, S>;

    fn f2_index_to_coord_carry(
        self,
        shape: &S,
        pps: Pps,
    ) -> (Self::Output, F2<Self::RemScalar>, Self::NextPps) {
        let s_val = shape.size();
        let next_pps = mul(pps, *shape);
        let f2_prod = crate::NoodleCuteResult::v(checked_clmul(Integer::value(pps), s_val));
        assert!(
            f2_prod == Integer::value(next_pps),
            "F2 idx2crd requires carry-less prefix product agreement (F2(pps) * F2(s) == pps * s)"
        );
        let (quot, rem): (F2Quotient<V, S>, F2Remainder<V, S>) = f2_divmod(self, *shape);
        (rem, quot, next_pps)
    }
}

const impl<SH, ST, V, Pps, const IS_LAST: bool> F2IndexToCoordCarry<Til<SH, ST>, Pps, IS_LAST>
    for F2<V>
where
    SH: Shape,
    ST: Tuple + Shape,
    V: [const] F2Scalar,
    Pps: [const] Integer,
    F2<V>: [const] F2IndexToCoordCarry<SH, Pps, { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> }>,
    F2<
        <F2<V> as F2IndexToCoordCarry<SH, Pps, { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> }>>::RemScalar,
    >: [const] F2IndexToCoordCarry<
            ST,
            <F2<V> as F2IndexToCoordCarry<
                SH,
                Pps,
                { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> },
            >>::NextPps,
            IS_LAST,
        >,
    <F2<
        <F2<V> as F2IndexToCoordCarry<SH, Pps, { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> }>>::RemScalar,
    > as F2IndexToCoordCarry<
        ST,
        <F2<V> as F2IndexToCoordCarry<SH, Pps, { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> }>>::NextPps,
        IS_LAST,
    >>::Output: Tuple,
{
    type Output = Til<
        <F2<V> as F2IndexToCoordCarry<SH, Pps, { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> }>>::Output,
        <F2<
            <F2<V> as F2IndexToCoordCarry<
                SH,
                Pps,
                { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> },
            >>::RemScalar,
        > as F2IndexToCoordCarry<
            ST,
            <F2<V> as F2IndexToCoordCarry<
                SH,
                Pps,
                { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> },
            >>::NextPps,
            IS_LAST,
        >>::Output,
    >;
    type RemScalar = <F2<
        <F2<V> as F2IndexToCoordCarry<SH, Pps, { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> }>>::RemScalar,
    > as F2IndexToCoordCarry<
        ST,
        <F2<V> as F2IndexToCoordCarry<SH, Pps, { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> }>>::NextPps,
        IS_LAST,
    >>::RemScalar;
    type NextPps = <F2<
        <F2<V> as F2IndexToCoordCarry<SH, Pps, { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> }>>::RemScalar,
    > as F2IndexToCoordCarry<
        ST,
        <F2<V> as F2IndexToCoordCarry<SH, Pps, { AND_BOOL::<IS_LAST, { HAS_NO_LEAVES::<ST> }> }>>::NextPps,
        IS_LAST,
    >>::NextPps;

    fn f2_index_to_coord_carry(
        self,
        shape: &Til<SH, ST>,
        pps: Pps,
    ) -> (Self::Output, F2<Self::RemScalar>, Self::NextPps) {
        let (head, rem, next_pps) = self.f2_index_to_coord_carry(&shape.0, pps);
        let (tail, rem2, next_pps2) = rem.f2_index_to_coord_carry(&shape.1, next_pps);
        (Til(head, tail), rem2, next_pps2)
    }
}

const impl<V: [const] F2Scalar> IndexToCoord<Nil> for F2<V> {
    type Output = Nil;
    fn index_to_coord(self, _: &Nil) -> Nil {
        Nil
    }
}

const impl<S: [const] ShapeScalar, V: [const] F2Scalar> IndexToCoord<S> for F2<V> {
    type Output = Self;
    fn index_to_coord(self, shape: &S) -> Self {
        let _ = shape.size();
        self
    }
}

const impl<SH, ST, V> IndexToCoord<Til<SH, ST>> for F2<V>
where
    SH: Shape,
    ST: Tuple + Shape,
    V: [const] F2Scalar,
    F2<V>: [const] F2IndexToCoordCarry<Til<SH, ST>, Int<1>, true>,
{
    type Output = <F2<V> as F2IndexToCoordCarry<Til<SH, ST>, Int<1>, true>>::Output;
    fn index_to_coord(self, shape: &Til<SH, ST>) -> Self::Output {
        self.f2_index_to_coord_carry(shape, Int::<1>).0
    }
}

const impl<V: [const] F2Scalar, S: [const] ShapeScalar> LeavesInBounds<S> for F2<V> {
    fn leaves_in_bounds(&self, shape: &S) -> bool {
        let extent = shape.size();
        lt(self.0, extent)
    }
}

const F2_SHAPE_IS_STATIC_ZERO<S: Shape>: bool = S::STATIC_HAS_ZERO;

/// Dispatches `InBounds<S>` for `F2<V>`, short-circuiting static-zero shapes to `false`.
#[doc(hidden)]
pub const trait F2InBounds<S: Shape, const STATIC_ZERO: bool>: Copy {
    fn f2_in_bounds(self, shape: &S) -> bool;
}

const impl<V: [const] F2Scalar, S: Shape> F2InBounds<S, true> for F2<V> {
    fn f2_in_bounds(self, _: &S) -> bool {
        false
    }
}

const impl<V, S> F2InBounds<S, false> for F2<V>
where
    V: [const] F2Scalar,
    S: [const] Shape,
    F2<V>: [const] IndexToCoord<S>,
    <F2<V> as IndexToCoord<S>>::Output: [const] LeavesInBounds<S>,
{
    fn f2_in_bounds(self, shape: &S) -> bool {
        if shape.has_zero() {
            false
        } else {
            self.index_to_coord(shape).leaves_in_bounds(shape)
        }
    }
}

const impl<V, S> InBounds<S> for F2<V>
where
    V: [const] F2Scalar,
    S: Shape,
    Self: [const] F2InBounds<S, { F2_SHAPE_IS_STATIC_ZERO::<S> }>,
{
    fn in_bounds(self, shape: &S) -> bool {
        self.f2_in_bounds(shape)
    }
}

// ------------------------------------------------------------------
// stride (StrideScalar, StrideMul, Accumulate, ScaleBasisFunctor, Coshape)
// ------------------------------------------------------------------

const impl<V: [const] F2Scalar> StrideScalar for F2<V> {
    type Value = Self;
    type Path = ModeEnd;

    fn value(self) -> Self {
        self
    }
}

const impl<V, S> StrideMul<S> for F2<V>
where
    V: [const] F2Scalar,
    S: [const] super::f2::F2Operand,
    F2<V>: [const] F2MulOp<S, { F2_MUL_DISPATCH::<F2<V>, S> }>,
{
    type Output = F2Product<F2<V>, S>;
    fn stride_mul(self, shape: S) -> Self::Output {
        f2_mul(self, shape)
    }
}

const impl<A, VS> StrideMul<F2<VS>> for A
where
    A: [const] Integer + [const] super::f2::F2Operand,
    VS: [const] F2Scalar,
    A: [const] F2MulOp<F2<VS>, { F2_MUL_DISPATCH::<A, F2<VS>> }>,
{
    type Output = F2Product<A, F2<VS>>;
    fn stride_mul(self, shape: F2<VS>) -> Self::Output {
        f2_mul(self, shape)
    }
}

// Accumulate (XOR for F2, Int<0>/Nil/BasisCombination<Nil> as additive identities)
const impl<V: [const] F2Scalar> Accumulate<F2<V>> for Nil {
    type Output = F2<V>;
    fn accumulate(self, rhs: F2<V>) -> F2<V> {
        rhs
    }
}

const impl<V: [const] F2Scalar> Accumulate<Nil> for F2<V> {
    type Output = Self;
    fn accumulate(self, _: Nil) -> Self {
        self
    }
}

const impl<V: [const] F2Scalar> Accumulate<Int<0>> for F2<V> {
    type Output = Self;
    fn accumulate(self, _: Int<0>) -> Self {
        self
    }
}

const impl<V: [const] F2Scalar> Accumulate<F2<V>> for Int<0> {
    type Output = F2<V>;
    fn accumulate(self, rhs: F2<V>) -> F2<V> {
        rhs
    }
}

const impl<L, R> Accumulate<F2<R>> for F2<L>
where
    L: [const] F2Scalar + [const] F2XorScalar<R, { BOTH_STATIC::<L, R> }>,
    R: [const] F2Scalar,
{
    type Output = F2<F2XorOutput<L, R>>;
    fn accumulate(self, rhs: F2<R>) -> Self::Output {
        f2_add(self, rhs)
    }
}

const impl<V: [const] F2Scalar> Accumulate<BasisCombination<Nil>> for F2<V> {
    type Output = Self;
    fn accumulate(self, _: BasisCombination<Nil>) -> Self {
        self
    }
}

const impl<V: [const] F2Scalar> Accumulate<F2<V>> for BasisCombination<Nil> {
    type Output = F2<V>;
    fn accumulate(self, rhs: F2<V>) -> F2<V> {
        rhs
    }
}

const impl<V: BasisCoeff, P: Copy> Accumulate<ScaledBasis<V, P>> for F2<Int<0>> {
    type Output = ScaledBasis<V, P>;
    fn accumulate(self, rhs: ScaledBasis<V, P>) -> Self::Output {
        rhs
    }
}

const impl<V: BasisCoeff, P: Copy> Accumulate<F2<Int<0>>> for ScaledBasis<V, P> {
    type Output = Self;
    fn accumulate(self, _: F2<Int<0>>) -> Self::Output {
        self
    }
}

const impl<H, T> Accumulate<BasisCombination<Til<H, T>>> for F2<Int<0>>
where
    H: Copy,
    T: BasisTuple,
    Til<H, T>: BasisTuple,
{
    type Output = BasisCombination<Til<H, T>>;
    fn accumulate(self, rhs: BasisCombination<Til<H, T>>) -> Self::Output {
        rhs
    }
}

const impl<H, T> Accumulate<F2<Int<0>>> for BasisCombination<Til<H, T>>
where
    H: Copy,
    T: BasisTuple,
    Til<H, T>: BasisTuple,
{
    type Output = Self;
    fn accumulate(self, _: F2<Int<0>>) -> Self::Output {
        self
    }
}

// ScaleBasisFunctor
const impl<V, VF> TransformLeafFunctor<V> for ScaleBasisFunctor<F2<VF>>
where
    V: [const] Integer + [const] Leaf + [const] super::f2::F2Operand,
    VF: [const] F2Scalar,
    V: [const] F2MulOp<F2<VF>, { F2_MUL_DISPATCH::<V, F2<VF>> }>,
{
    type Output = F2Product<V, F2<VF>>;
    fn apply(&self, leaf: V) -> Self::Output {
        f2_mul(leaf, self.0)
    }
}

const impl<V, Factor> TransformLeafFunctor<F2<V>> for ScaleBasisFunctor<Factor>
where
    V: [const] F2Scalar,
    Factor: [const] Integer + [const] super::f2::F2Operand,
    F2<V>: [const] F2MulOp<Factor, { F2_MUL_DISPATCH::<F2<V>, Factor> }>,
{
    type Output = F2Product<F2<V>, Factor>;
    fn apply(&self, leaf: F2<V>) -> Self::Output {
        f2_mul(leaf, self.0)
    }
}

const impl<V, VF> TransformLeafFunctor<F2<V>> for ScaleBasisFunctor<F2<VF>>
where
    V: [const] F2Scalar,
    VF: [const] F2Scalar,
    F2<V>: [const] F2MulOp<F2<VF>, { F2_MUL_DISPATCH::<F2<V>, F2<VF>> }>,
{
    type Output = F2Product<F2<V>, F2<VF>>;
    fn apply(&self, leaf: F2<V>) -> Self::Output {
        f2_mul(leaf, self.0)
    }
}

const impl<V, VF, PF> TransformLeafFunctor<F2<V>> for ScaleBasisFunctor<ScaledBasis<VF, PF>>
where
    V: [const] F2Scalar,
    VF: BasisCoeff,
    PF: BasisPath,
    ScaledBasis<VF, PF>: [const] StrideMul<F2<V>>,
    <ScaledBasis<VF, PF> as StrideMul<F2<V>>>::Output: [const] Item,
{
    type Output = <ScaledBasis<VF, PF> as StrideMul<F2<V>>>::Output;
    fn apply(&self, leaf: F2<V>) -> Self::Output {
        self.0.stride_mul(leaf)
    }
}

const impl<V, T> TransformLeafFunctor<F2<V>> for ScaleBasisFunctor<BasisCombination<T>>
where
    V: [const] F2Scalar,
    T: BasisTuple,
    BasisCombination<T>: [const] StrideMul<F2<V>>,
    <BasisCombination<T> as StrideMul<F2<V>>>::Output: [const] Item,
{
    type Output = <BasisCombination<T> as StrideMul<F2<V>>>::Output;
    fn apply(&self, leaf: F2<V>) -> Self::Output {
        self.0.stride_mul(leaf)
    }
}

// StrideSkeleton, OffsetExtent (1 << bit_length), CoshapeCoeffAccum (bitwise OR), CoshapeAccumulate
const impl<V: [const] F2Scalar> StrideSkeleton for F2<V> {
    type Terms = Nil;
    fn skeleton_terms(self) -> Nil {
        Nil
    }
}

const F2_COSHAPE_EXTENT<const N: int>: int = shiftl(1i32, bit_length_non_neg(N));

const impl OffsetExtent for F2<int> {
    type Output = int;
    fn offset_extent(self) -> int {
        shiftl(1i32, bit_length_non_neg(self.value()))
    }
}

const impl<const N: int> OffsetExtent for F2<Int<N>>
where
    (): [const] super::f2::ValidF2Value<{ IS_NON_NEG::<N> }>,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ F2_COSHAPE_EXTENT::<N> }> }>,
{
    type Output = Int<{ F2_COSHAPE_EXTENT::<N> }>;
    fn offset_extent(self) -> Self::Output {
        Int
    }
}

const impl<L, R> CoshapeCoeffAccum<F2<R>> for F2<L>
where
    L: [const] F2Scalar + [const] F2OrScalar<R, { BOTH_STATIC::<L, R> }>,
    R: [const] F2Scalar,
{
    type Output = F2<F2OrOutput<L, R>>;
    fn coshape_coeff_accum(self, rhs: F2<R>) -> Self::Output {
        F2(self.0.or_scalar(rhs.0))
    }
}

const impl<L: [const] F2Scalar> CoshapeCoeffAccum<Int<0>> for F2<L> {
    type Output = Self;
    fn coshape_coeff_accum(self, _: Int<0>) -> Self {
        self
    }
}

const impl<R: [const] F2Scalar> CoshapeCoeffAccum<F2<R>> for Int<0> {
    type Output = F2<R>;
    fn coshape_coeff_accum(self, rhs: F2<R>) -> F2<R> {
        rhs
    }
}

const impl<V: [const] F2Scalar> CoshapeAccumulate<F2<V>> for Nil {
    type Output = F2<V>;
    fn coshape_accumulate(self, rhs: F2<V>) -> F2<V> {
        rhs
    }
}

const impl<V: [const] F2Scalar> CoshapeAccumulate<Nil> for F2<V> {
    type Output = Self;
    fn coshape_accumulate(self, _: Nil) -> Self {
        self
    }
}

const impl<V: [const] F2Scalar> CoshapeAccumulate<Int<0>> for F2<V> {
    type Output = Self;
    fn coshape_accumulate(self, _: Int<0>) -> Self {
        self
    }
}

const impl<V: [const] F2Scalar> CoshapeAccumulate<F2<V>> for Int<0> {
    type Output = F2<V>;
    fn coshape_accumulate(self, rhs: F2<V>) -> F2<V> {
        rhs
    }
}

const impl<L, R> CoshapeAccumulate<F2<R>> for F2<L>
where
    L: [const] F2Scalar + [const] F2OrScalar<R, { BOTH_STATIC::<L, R> }>,
    R: [const] F2Scalar,
{
    type Output = F2<F2OrOutput<L, R>>;
    fn coshape_accumulate(self, rhs: F2<R>) -> Self::Output {
        F2(self.0.or_scalar(rhs.0))
    }
}

const impl<V: [const] F2Scalar> CoshapeAccumulate<BasisCombination<Nil>> for F2<V> {
    type Output = Self;
    fn coshape_accumulate(self, _: BasisCombination<Nil>) -> Self {
        self
    }
}

const impl<V: [const] F2Scalar> CoshapeAccumulate<F2<V>> for BasisCombination<Nil> {
    type Output = F2<V>;
    fn coshape_accumulate(self, rhs: F2<V>) -> F2<V> {
        rhs
    }
}

const impl<V: BasisCoeff, P: Copy> CoshapeAccumulate<BasisTerm<V, P>> for F2<Int<0>> {
    type Output = BasisTerm<V, P>;
    fn coshape_accumulate(self, rhs: BasisTerm<V, P>) -> Self::Output {
        rhs
    }
}

const impl<V: BasisCoeff, P: Copy> CoshapeAccumulate<F2<Int<0>>> for BasisTerm<V, P> {
    type Output = Self;
    fn coshape_accumulate(self, _: F2<Int<0>>) -> Self::Output {
        self
    }
}

const impl<H, T> CoshapeAccumulate<BasisCombination<Til<H, T>>> for F2<Int<0>>
where
    H: Copy,
    T: BasisTuple,
    Til<H, T>: BasisTuple,
{
    type Output = BasisCombination<Til<H, T>>;
    fn coshape_accumulate(self, rhs: BasisCombination<Til<H, T>>) -> Self::Output {
        rhs
    }
}

const impl<H, T> CoshapeAccumulate<F2<Int<0>>> for BasisCombination<Til<H, T>>
where
    H: Copy,
    T: BasisTuple,
    Til<H, T>: BasisTuple,
{
    type Output = Self;
    fn coshape_accumulate(self, _: F2<Int<0>>) -> Self::Output {
        self
    }
}

// ------------------------------------------------------------------
// layout (IsLayout, LayoutItemEq, OrderKey, TilerFactor, Recast)
// ------------------------------------------------------------------

const impl<V: Copy> IsLayout for F2<V> {
    const IS_LAYOUT: bool = false;
}

const impl<V: [const] F2Scalar> OrderKey for F2<V> {
    const IS_STATIC_ZERO: bool = static_true(static_eq_of::<V, Int<0>>());
}

// F2<Int<1>> must NOT be treated as a no-op tiler factor, since scaling an integer-strided
// layout by F2(1) lifts its strides into the F2 algebra.
const impl<V: Copy> TilerFactor for F2<V> {
    const IS_UNIT: bool = false;
}

const impl<L: [const] F2Scalar, R: [const] F2Scalar> LayoutItemEq<F2<R>> for F2<L> {
    fn layout_item_eq(&self, rhs: &F2<R>) -> bool {
        eq(self.0, rhs.0)
    }
}

const impl<V: [const] F2Scalar, const N: int> LayoutItemEq<Int<N>> for F2<V> {
    fn layout_item_eq(&self, rhs: &Int<N>) -> bool {
        eq(self.0, *rhs)
    }
}

const impl<const N: int, V: [const] F2Scalar> LayoutItemEq<F2<V>> for Int<N> {
    fn layout_item_eq(&self, rhs: &F2<V>) -> bool {
        eq(*self, rhs.0)
    }
}

const impl<V: [const] F2Scalar> LayoutItemEq<int> for F2<V> {
    fn layout_item_eq(&self, rhs: &int) -> bool {
        eq(self.0, *rhs)
    }
}

const impl<V: [const] F2Scalar> LayoutItemEq<F2<V>> for int {
    fn layout_item_eq(&self, rhs: &F2<V>) -> bool {
        eq(*self, rhs.0)
    }
}

const impl<V: [const] F2Scalar> LayoutItemEq<BasisCombination<Nil>> for F2<V> {
    fn layout_item_eq(&self, _: &BasisCombination<Nil>) -> bool {
        eq(self.0, Int::<0>)
    }
}

const impl<V: [const] F2Scalar> LayoutItemEq<F2<V>> for BasisCombination<Nil> {
    fn layout_item_eq(&self, rhs: &F2<V>) -> bool {
        eq(rhs.0, Int::<0>)
    }
}

macro_rules! impl_f2_false_eq {
    ([$($g:tt)*] $Other:ty $(where $($wb:tt)+)?) => {
        const impl<V: [const] F2Scalar, $($g)*> LayoutItemEq<$Other> for F2<V>
        $(where $($wb)+)?
        {
            fn layout_item_eq(&self, _: &$Other) -> bool {
                false
            }
        }
        const impl<V: [const] F2Scalar, $($g)*> LayoutItemEq<F2<V>> for $Other
        $(where $($wb)+)?
        {
            fn layout_item_eq(&self, _: &F2<V>) -> bool {
                false
            }
        }
    };
}

impl_f2_false_eq!([] Nil);
impl_f2_false_eq!([H, T] Til<H, T>);
impl_f2_false_eq!([VB, P] BasisTerm<VB, P>);
impl_f2_false_eq!([H, T] BasisCombination<Til<H, T>> where Til<H, T>: BasisTuple);

const impl<V: [const] F2Scalar, const NEW_D: int> MakeStaticRecastStride<NEW_D> for F2<V>
where
    (): [const] ValidF2Value<{ IS_NON_NEG::<NEW_D> }>,
{
    type Output = F2<Int<NEW_D>>;
    fn make_static_recast_stride() -> F2<Int<NEW_D>> {
        F2(Int)
    }
}

const impl<V: [const] F2Scalar> RecastStrideScalar for F2<V> {
    const IS_STATIC_COEFF: bool = V::IS_STATIC;
    const STATIC_SIGNED: Option<int> = V::STATIC_VALUE;
    const IS_F2_COEFF: bool = true;
    type DynamicOut = F2<int>;

    fn signed_coeff(&self) -> int {
        self.value()
    }
    fn make_dynamic_out(new_d: int) -> F2<int> {
        F2::new(new_d)
    }
}

// ------------------------------------------------------------------
// algebra (Nullspace, AtomCompose for shape-1 F2 RHS)
// ------------------------------------------------------------------

const impl<V: [const] F2Scalar> NullspaceStrideLeaf for F2<V> {
    const IS_STATIC_ZERO: bool = static_true(static_eq_of::<V, Int<0>>());
}

const impl<S0, TS, D0, TD, M> AtomCompose<Layout<Int<1>, F2<M>>>
    for Layout<Til<S0, TS>, Til<D0, TD>>
where
    S0: ShapeScalar,
    D0: StrideScalar + Stride<S0>,
    TS: Tuple + Shape,
    TD: Tuple + Stride<TS>,
    M: [const] F2Scalar,
    F2<M>: [const] IndexToCoord<Til<S0, TS>>,
    <F2<M> as IndexToCoord<Til<S0, TS>>>::Output: [const] InnerProduct<Til<D0, TD>>,
    <<F2<M> as IndexToCoord<Til<S0, TS>>>::Output as InnerProduct<Til<D0, TD>>>::Output:
        Stride<Int<1>>,
{
    type Output = Layout<
        Int<1>,
        <<F2<M> as IndexToCoord<Til<S0, TS>>>::Output as InnerProduct<Til<D0, TD>>>::Output,
    >;

    fn atom_compose(self, b: Layout<Int<1>, F2<M>>) -> Self::Output {
        Layout::new(Int::<1>, self.call(b.stride))
    }
}
