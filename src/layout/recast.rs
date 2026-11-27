//! Element scale recasting (`Fraction`, `Recast`, `recast`).
//!
//! Adjusts a [`Layout`]'s shape and stride when the underlying element size changes by a
//! positive integer or rational factor `num / den`.
//!
//! # Main Interfaces
//!
//! - [`Fraction`]: Rational scale factor `num / den` (defaulting `den` to [`Int<1>`]).
//! - [`RecastScale`]: Positive integer or [`Fraction`] scale accepted by [`recast`].
//! - [`Recast`] / [`recast`] / [`recast_leaf_values`]: Rescales static or dynamic layouts
//!   while enforcing divisibility and checked [`int`] arithmetic.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::{Fraction, Layout, recast};
//!
//! let l = Layout::new(Int::<24>, Int::<1>);
//! assert_eq!(recast(&l, Int::<8>), Layout::new(Int::<3>, Int::<1>));
//! assert_eq!(
//!     recast(&l, Fraction::new(Int::<1>, Int::<2>)),
//!     Layout::new(Int::<48>, Int::<1>)
//! );
//! ```
//!
//! # Notes
//!
//! - Requires non-negative shape extents and strictly positive scale numerators/denominators.
//! - Zero-stride leaves (`0`, `Int<0>`, or empty `BasisCombination<Nil>`) preserve `(shape, 0)`.
//! - F2 strides support aligned binary scaling and carry-free integer progressions.
//!   Packing a smaller power-of-two stride groups adjacent coordinates into one element.
//! - Divisibility violations or [`int`] overflow fail at compile time when shape, stride, and
//!   scale are all static, and panic at runtime when any operand is dynamic.

use crate::basis::{BasisCoeff, BasisCombination, BasisPath, BasisTerm};
use crate::f2::{F2, F2Scalar, ValidF2Value};
use crate::integer::{
    IS_NON_NEG, Int, Integer, add, carry_free_mask, ceil_div, div, floor_divmod, gcd, int, is_pow2,
    mul, rem,
};
use crate::shape::{Shape, ShapeScalar, ValidShapeExtent};
use crate::stride::Stride;
use crate::tuple::{Leaf, Nil, Til, Tuple};

use super::Layout;

/// Positive rational element scale factor `num / den` for [`recast`].
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Fraction<N, D = Int<1>>(pub N, pub D);

impl<N, D> Fraction<N, D> {
    /// Constructs a rational scale `num / den`.
    pub const fn new(num: N, den: D) -> Self {
        Self(num, den)
    }
}

/// Computes `(new_shape, new_stride)` for one scalar leaf under scale `num / den`,
/// panicking if `s < 0`, `num <= 0`, `den <= 0`, divisibility fails, or [`int`] overflows.
pub const fn recast_leaf_values(s: int, dd: int, num: int, den: int) -> (int, int) {
    assert!(s >= 0, "recast shape extent must be non-negative");
    assert!(num > 0 && den > 0, "recast scale must be positive");
    let g = gcd(num, den);
    let n = div(num, g);
    let d = div(den, g);
    if dd == 0 {
        return (s, 0);
    }
    if dd == 1 {
        let (q, r) = floor_divmod(s, n);
        let qd = mul(q, d);
        let rd = mul(r, d);
        let extra = ceil_div(rd, n);
        let new_s = add(qd, extra);
        return (new_s, 1);
    }
    let g2 = gcd(dd, n);
    let dd2 = div(dd, g2);
    let n2 = div(n, g2);
    if n2 == 1 {
        return (s, mul(dd2, d));
    }
    if dd2 == 1 && d == 1 {
        return (ceil_div(s, n2), 1);
    }
    panic!("recast divisibility condition violated")
}

/// Computes `(new_shape, new_stride)` for one `F2` stride leaf under scale `num / den`.
pub(crate) const fn recast_f2_leaf_values(s: int, dd: int, num: int, den: int) -> (int, int) {
    assert!(num > 0 && den > 0, "recast scale must be positive");
    let g = gcd(num, den);
    let n = div(num, g);
    let d = div(den, g);
    let result = recast_leaf_values(s, dd, n, d);
    // Binary scaling shifts the polynomial coefficients directly. Other
    // ratios are exact when both address sequences are ordinary progressions.
    let binary = is_pow2(n) && is_pow2(d) && (rem(dd, n) == 0 || is_pow2(dd));
    assert!(
        dd <= 1
            || s <= 1
            || binary
            || (carry_free_mask(s, dd).is_some() && carry_free_mask(result.0, result.1).is_some()),
        "F2 recast changes the element offsets"
    );
    result
}

/// Positive integer or rational scale accepted by [`recast`].
pub const trait RecastScale: Copy {
    /// `true` iff both numerator and denominator are compile-time constants.
    const IS_STATIC: bool;
    /// Static numerator (`0` when dynamic or non-positive).
    const STATIC_NUM: int;
    /// Static denominator (`1` for integers, `0` when dynamic or non-positive).
    const STATIC_DEN: int;

    /// Returns positive `(num, den)`, panicking if zero or negative.
    fn to_num_den(self) -> (int, int);
}

const fn static_pos_int<T: Integer>() -> int {
    match T::STATIC_VALUE {
        Some(v) if v > 0 => v,
        _ => 0,
    }
}

const impl<const N: int> RecastScale for Int<N> {
    const IS_STATIC: bool = true;
    const STATIC_NUM: int = if N > 0 { N } else { 0 };
    const STATIC_DEN: int = 1;

    fn to_num_den(self) -> (int, int) {
        assert!(N > 0, "recast scale must be positive");
        (N, 1)
    }
}

const impl RecastScale for int {
    const IS_STATIC: bool = false;
    const STATIC_NUM: int = 0;
    const STATIC_DEN: int = 0;

    fn to_num_den(self) -> (int, int) {
        assert!(self > 0, "recast scale must be positive");
        (self, 1)
    }
}

const impl<N: [const] Integer, D: [const] Integer> RecastScale for Fraction<N, D> {
    const IS_STATIC: bool = N::IS_STATIC && D::IS_STATIC;
    const STATIC_NUM: int = static_pos_int::<N>();
    const STATIC_DEN: int = static_pos_int::<D>();

    fn to_num_den(self) -> (int, int) {
        let n = self.0.value();
        let d = self.1.value();
        assert!(n > 0 && d > 0, "recast scale must be positive");
        (n, d)
    }
}

/// Constructs the static recast output stride leaf for concrete value `NEW_D`.
#[doc(hidden)]
pub const trait MakeStaticRecastStride<const NEW_D: int>: Leaf {
    type Output: Leaf;
    fn make_static_recast_stride() -> Self::Output;
}

const impl<const NEW_D: int> MakeStaticRecastStride<NEW_D> for int {
    type Output = Int<NEW_D>;
    fn make_static_recast_stride() -> Int<NEW_D> {
        Int::<NEW_D>
    }
}

const impl<const D: int, const NEW_D: int> MakeStaticRecastStride<NEW_D> for Int<D> {
    type Output = Int<NEW_D>;
    fn make_static_recast_stride() -> Int<NEW_D> {
        Int::<NEW_D>
    }
}

const impl<V: [const] F2Scalar, const NEW_D: int> MakeStaticRecastStride<NEW_D> for F2<V>
where
    (): [const] ValidF2Value<{ IS_NON_NEG::<NEW_D> }>,
{
    type Output = F2<Int<NEW_D>>;
    fn make_static_recast_stride() -> F2<Int<NEW_D>> {
        F2::new(Int::<NEW_D>)
    }
}

const impl<V, P, const NEW_D: int> MakeStaticRecastStride<NEW_D> for BasisTerm<V, P>
where
    V: [const] BasisCoeff + [const] MakeStaticRecastStride<NEW_D>,
    <V as MakeStaticRecastStride<NEW_D>>::Output: [const] BasisCoeff,
    P: [const] BasisPath,
{
    type Output = BasisTerm<<V as MakeStaticRecastStride<NEW_D>>::Output, P>;
    fn make_static_recast_stride() -> Self::Output {
        BasisTerm::new(V::make_static_recast_stride())
    }
}

const impl<const NEW_D: int> MakeStaticRecastStride<NEW_D> for BasisCombination<Nil> {
    type Output = BasisCombination<Nil>;
    fn make_static_recast_stride() -> Self::Output {
        BasisCombination::<Nil>::zero()
    }
}

/// Extracts and rebuilds scalar stride coefficients (`int`, `Int<D>`, or `F2`) across static
/// and dynamic (`DynamicOut`) recast paths.
#[doc(hidden)]
pub const trait RecastStrideScalar: [const] BasisCoeff + Leaf {
    const IS_STATIC_COEFF: bool;
    const STATIC_SIGNED: Option<int>;
    const IS_F2_COEFF: bool = false;
    type DynamicOut: [const] BasisCoeff + Leaf;

    fn signed_coeff(&self) -> int;
    fn make_dynamic_out(new_d: int) -> Self::DynamicOut;
}

const impl RecastStrideScalar for int {
    const IS_STATIC_COEFF: bool = false;
    const STATIC_SIGNED: Option<int> = None;
    type DynamicOut = int;

    fn signed_coeff(&self) -> int {
        *self
    }
    fn make_dynamic_out(new_d: int) -> int {
        new_d
    }
}

const impl<const D: int> RecastStrideScalar for Int<D> {
    const IS_STATIC_COEFF: bool = true;
    const STATIC_SIGNED: Option<int> = Some(D);
    type DynamicOut = int;

    fn signed_coeff(&self) -> int {
        D
    }
    fn make_dynamic_out(new_d: int) -> int {
        new_d
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

/// Abstracts over scalar, [`BasisTerm`], and empty `BasisCombination<Nil>` stride leaves in
/// [`RecastPair`], providing signed coefficient extraction and dynamic reconstruction.
#[doc(hidden)]
pub const trait RecastStrideLeaf: Leaf {
    const IS_STATIC_STRIDE: bool;
    const STATIC_SIGNED: Option<int>;
    const IS_F2_STRIDE: bool = false;
    type DynamicOut: Leaf;

    fn signed_coeff(&self) -> int;
    fn make_dynamic_out(new_d: int) -> Self::DynamicOut;
}

const impl<V: [const] RecastStrideScalar> RecastStrideLeaf for V {
    const IS_STATIC_STRIDE: bool = V::IS_STATIC_COEFF;
    const STATIC_SIGNED: Option<int> = V::STATIC_SIGNED;
    const IS_F2_STRIDE: bool = V::IS_F2_COEFF;
    type DynamicOut = V::DynamicOut;

    fn signed_coeff(&self) -> int {
        V::signed_coeff(self)
    }
    fn make_dynamic_out(new_d: int) -> Self::DynamicOut {
        V::make_dynamic_out(new_d)
    }
}

const impl<V: [const] RecastStrideScalar, P: [const] BasisPath> RecastStrideLeaf
    for BasisTerm<V, P>
{
    const IS_STATIC_STRIDE: bool = V::IS_STATIC_COEFF;
    const STATIC_SIGNED: Option<int> = V::STATIC_SIGNED;
    const IS_F2_STRIDE: bool = V::IS_F2_COEFF;
    type DynamicOut = BasisTerm<V::DynamicOut, P>;

    fn signed_coeff(&self) -> int {
        self.value.signed_coeff()
    }
    fn make_dynamic_out(new_d: int) -> Self::DynamicOut {
        BasisTerm::new(V::make_dynamic_out(new_d))
    }
}

const impl RecastStrideLeaf for BasisCombination<Nil> {
    const IS_STATIC_STRIDE: bool = true;
    const STATIC_SIGNED: Option<int> = Some(0);
    type DynamicOut = BasisCombination<Nil>;

    fn signed_coeff(&self) -> int {
        0
    }
    fn make_dynamic_out(_: int) -> Self::DynamicOut {
        BasisCombination::<Nil>::zero()
    }
}

const STATIC_RECAST_LEAF<S: Integer, D: RecastStrideLeaf, Scale: RecastScale>: (int, int) = {
    let s = S::STATIC_VALUE.expect("recast static shape missing");
    let dd = D::STATIC_SIGNED.expect("recast static stride missing");
    if D::IS_F2_STRIDE {
        recast_f2_leaf_values(s, dd, Scale::STATIC_NUM, Scale::STATIC_DEN)
    } else {
        recast_leaf_values(s, dd, Scale::STATIC_NUM, Scale::STATIC_DEN)
    }
};

const STATIC_RECAST_S<S: Integer, D: RecastStrideLeaf, Scale: RecastScale>: int =
    STATIC_RECAST_LEAF::<S, D, Scale>.0;

const STATIC_RECAST_D<S: Integer, D: RecastStrideLeaf, Scale: RecastScale>: int =
    STATIC_RECAST_LEAF::<S, D, Scale>.1;

#[doc(hidden)]
pub const RECAST_ALL_STATIC<S: Integer, D: RecastStrideLeaf, Scale: RecastScale>: bool =
    <S as Integer>::IS_STATIC && D::IS_STATIC_STRIDE && Scale::IS_STATIC;

/// Dispatches single-leaf `(Self, D)` recasting under `Scale` to compile-time evaluation
/// (`ALL_STATIC = true`) or runtime evaluation (`ALL_STATIC = false`).
#[doc(hidden)]
pub const trait RecastLeafDispatch<
    D: [const] RecastStrideLeaf,
    Scale: [const] RecastScale,
    const ALL_STATIC: bool,
>: [const] ShapeScalar
{
    type NewShape: [const] ShapeScalar;
    type NewStride: Leaf;

    fn recast_leaf(self, d: D, scale: Scale) -> (Self::NewShape, Self::NewStride);
}

const impl<S, D, Scale> RecastLeafDispatch<D, Scale, true> for S
where
    S: [const] ShapeScalar,
    D: [const] RecastStrideLeaf
        + [const] MakeStaticRecastStride<{ STATIC_RECAST_D::<S, D, Scale> }>,
    Scale: [const] RecastScale,
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<{ STATIC_RECAST_S::<S, D, Scale> }> }>,
{
    type NewShape = Int<{ STATIC_RECAST_S::<S, D, Scale> }>;
    type NewStride = <D as MakeStaticRecastStride<{ STATIC_RECAST_D::<S, D, Scale> }>>::Output;

    fn recast_leaf(self, _: D, _: Scale) -> (Self::NewShape, Self::NewStride) {
        let _ = (
            STATIC_RECAST_S::<S, D, Scale>,
            STATIC_RECAST_D::<S, D, Scale>,
        );
        (
            Int,
            <D as MakeStaticRecastStride<{ STATIC_RECAST_D::<S, D, Scale> }>>::make_static_recast_stride(),
        )
    }
}

const impl<S, D, Scale> RecastLeafDispatch<D, Scale, false> for S
where
    S: [const] ShapeScalar,
    D: [const] RecastStrideLeaf,
    Scale: [const] RecastScale,
{
    type NewShape = int;
    type NewStride = D::DynamicOut;

    fn recast_leaf(self, d: D, scale: Scale) -> (Self::NewShape, Self::NewStride) {
        let (num, den) = scale.to_num_den();
        let dd = d.signed_coeff();
        let (new_s, new_d) = if D::IS_F2_STRIDE {
            recast_f2_leaf_values(self.value(), dd, num, den)
        } else {
            recast_leaf_values(self.value(), dd, num, den)
        };
        (new_s, D::make_dynamic_out(new_d))
    }
}

/// Recursively recasts a congruent `(Self, D)` shape-stride pair under `Scale` across `Nil`,
/// scalar leaves, and nested tuples, producing `(NewShape, NewStride)`.
#[doc(hidden)]
pub const trait RecastPair<D, Scale: RecastScale>: Copy {
    type NewShape: Shape;
    type NewStride: Stride<Self::NewShape>;

    fn recast_pair(self, d: D, scale: Scale) -> (Self::NewShape, Self::NewStride);
}

const impl<Scale: [const] RecastScale> RecastPair<Nil, Scale> for Nil {
    type NewShape = Nil;
    type NewStride = Nil;

    fn recast_pair(self, _: Nil, scale: Scale) -> (Nil, Nil) {
        let _ = scale.to_num_den();
        (Nil, Nil)
    }
}

macro_rules! impl_recast_pair_leaf {
    ([$($g:tt)*] $S:ty $(where $($wb:tt)+)?) => {
        const impl<$($g)* D, Scale> RecastPair<D, Scale> for $S
        where
            D: [const] RecastStrideLeaf,
            Scale: [const] RecastScale,
            $S: [const] RecastLeafDispatch<D, Scale, { RECAST_ALL_STATIC::<$S, D, Scale> }>,
            $($($wb)+)?
        {
            type NewShape = <$S as RecastLeafDispatch<
                D,
                Scale,
                { RECAST_ALL_STATIC::<$S, D, Scale> },
            >>::NewShape;
            type NewStride = <$S as RecastLeafDispatch<
                D,
                Scale,
                { RECAST_ALL_STATIC::<$S, D, Scale> },
            >>::NewStride;

            fn recast_pair(self, d: D, scale: Scale) -> (Self::NewShape, Self::NewStride) {
                self.recast_leaf(d, scale)
            }
        }
    };
}

impl_recast_pair_leaf!([] int);
impl_recast_pair_leaf!([const N: int,] Int<N> where (): [const] ValidShapeExtent<{ IS_NON_NEG::<N> }>);

const impl<SH, ST, DH, DT, Scale: [const] RecastScale> RecastPair<Til<DH, DT>, Scale>
    for Til<SH, ST>
where
    DH: Copy,
    DT: Copy,
    SH: [const] RecastPair<DH, Scale>,
    ST: Tuple + [const] RecastPair<DT, Scale>,
    ST::NewShape: Tuple + Shape,
    ST::NewStride: Tuple + Stride<ST::NewShape>,
{
    type NewShape = Til<SH::NewShape, ST::NewShape>;
    type NewStride = Til<SH::NewStride, ST::NewStride>;

    fn recast_pair(self, d: Til<DH, DT>, scale: Scale) -> (Self::NewShape, Self::NewStride) {
        let Til(dh, dt) = d;
        let (hs, hd) = self.0.recast_pair(dh, scale);
        let (ts, td) = self.1.recast_pair(dt, scale);
        (Til(hs, ts), Til(hd, td))
    }
}

/// Rescales a layout's element unit by a positive integer or [`Fraction`] scale.
pub const trait Recast<Scale: RecastScale>: Copy {
    /// Rescaled [`Layout`] type.
    type Output;

    /// Rescales `self` by `scale`, panicking if divisibility fails or arithmetic overflows.
    fn recast(&self, scale: Scale) -> Self::Output;
}

const impl<S, D, Scale> Recast<Scale> for Layout<S, D>
where
    S: Shape + [const] RecastPair<D, Scale>,
    D: Stride<S>,
    Scale: [const] RecastScale,
{
    type Output = Layout<S::NewShape, S::NewStride>;

    fn recast(&self, scale: Scale) -> Self::Output {
        let (shape, stride) = self.shape.recast_pair(self.stride, scale);
        Layout::new(shape, stride)
    }
}

/// Rescales `layout` by `scale`.
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::{Fraction, Layout, recast};
///
/// let l = Layout::new(Int::<24>, Int::<1>);
/// assert_eq!(recast(&l, Int::<8>), Layout::new(Int::<3>, Int::<1>));
/// assert_eq!(
///     recast(&l, Fraction::new(Int::<1>, Int::<2>)),
///     Layout::new(Int::<48>, Int::<1>)
/// );
/// ```
pub const fn recast<L: [const] Recast<Scale>, Scale: [const] RecastScale>(
    layout: &L,
    scale: Scale,
) -> L::Output {
    layout.recast(scale)
}
