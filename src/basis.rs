//! Basis strides and coordinate offsets: CuTe's `ScaledBasis` and `ArithmeticTuple`
//! (`arithmetic_tuple.hpp`), and the identity layout built from them.
//!
//! A [`ScaledBasis<V, M>`] stride contributes `coord * V` to mode `M` of a
//! coordinate instead of to an integer address, so an identity layout maps each
//! coordinate to itself and each mode stays independent: indices past a tail
//! tile do not carry into the next mode. Offsets add mode by mode, padding the
//! shorter coordinate with zeros, exactly as `ArithmeticTuple` addition does;
//! coordinate tuples are plain [`IntTuple`] chains.
//!
//! Only flat bases are provided: an integral shape gets the stride `Int<1>`
//! (upstream `E<>`) and a flat tuple shape gets `(E<0>, E<1>, ...)`. Nested
//! shapes, whose upstream bases are multi-index `E<i, j>`, are not supported.
//!
//! ```
//! use noodle_cute::basis::make_identity_layout;
//! use noodle_cute::{DynInt, Int, shape};
//!
//! let identity = make_identity_layout(shape![DynInt(10), Int::<8>]);
//! assert_eq!(identity.crd2idx(shape![3i32, 5i32]).0, 3);
//! assert_eq!(identity.crd2idx(13i32).1.0, DynInt(1));
//! ```

use crate::Bool;
use crate::int::{Arith, Int, Integer, Op, Sum};
use crate::layout::Layout;
use crate::tuple::{IntTuple, Nil};

/// The value `V` in mode `M` of a coordinate: CuTe `ScaledBasis<V, M>`.
///
/// Used as a stride, `c * ScaledBasis<V, M>` is `ScaledBasis<c * V, M>`. As a
/// coordinate offset it equals the tuple `(0, ..., 0, V)` with `M` leading zeros.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct ScaledBasis<V, const M: usize>(pub V);

/// The unit basis of mode `M`: CuTe `E<M>`.
pub type E<const M: usize> = ScaledBasis<Int<1>, M>;

const IS_ZERO<const M: usize>: bool = M == 0;
const PREV<const M: usize>: usize = M.saturating_sub(1);
const NEXT<const M: usize>: usize = M + 1;

// Implementation traits that appear in public bounds but are not part of the API.
mod detail {
    use super::{IntTuple, Nil, ScaledBasis};

    /// Coordinate offset addition (`ArithmeticTuple::operator+`).
    pub trait OffsetAdd<B> {
        type Output;
        fn offset_add(self, rhs: B) -> Self::Output;
    }

    /// `as_arithmetic_tuple` of a basis: `M` zeros followed by the value.
    pub trait AsTuple {
        type Output;
        fn into_tuple(self) -> Self::Output;
    }

    pub trait ZerosThen<V, const M: usize> {
        type Output;
        fn zeros_then(value: V) -> Self::Output;
    }

    /// Whether a stride tree contains a basis stride.
    pub trait StrideKind {
        type Kind;
        const BASIS: bool;
    }

    /// `make_basis_like` and the zero coordinate of a flat tuple shape, from mode `POS`.
    pub trait BasisModes<const POS: usize> {
        type Stride: Copy;
        type Zero: Copy;
        fn basis() -> Self::Stride;
        fn zero() -> Self::Zero;
    }

    impl<const POS: usize> BasisModes<POS> for Nil {
        type Stride = Nil;
        type Zero = Nil;
        fn basis() -> Nil {
            Nil
        }
        fn zero() -> Nil {
            Nil
        }
    }

    impl<V, const M: usize> AsTuple for ScaledBasis<V, M>
    where
        crate::Bool<{ super::IS_ZERO::<M> }>: ZerosThen<V, M>,
    {
        type Output = <crate::Bool<{ super::IS_ZERO::<M> }> as ZerosThen<V, M>>::Output;
        fn into_tuple(self) -> Self::Output {
            <crate::Bool<{ super::IS_ZERO::<M> }> as ZerosThen<V, M>>::zeros_then(self.0)
        }
    }

    impl<H, T> AsTuple for IntTuple<H, T> {
        type Output = Self;
        fn into_tuple(self) -> Self {
            self
        }
    }
}
use detail::{AsTuple, BasisModes, ZerosThen};
pub(crate) use detail::{OffsetAdd, StrideKind};

impl<V, const M: usize> ZerosThen<V, M> for Bool<true> {
    type Output = IntTuple<V, Nil>;
    fn zeros_then(value: V) -> Self::Output {
        IntTuple(value, Nil)
    }
}

impl<V, const M: usize> ZerosThen<V, M> for Bool<false>
where
    Bool<{ IS_ZERO::<{ PREV::<M> }> }>: ZerosThen<V, { PREV::<M> }>,
{
    type Output = IntTuple<
        Int<0>,
        <Bool<{ IS_ZERO::<{ PREV::<M> }> }> as ZerosThen<V, { PREV::<M> }>>::Output,
    >;
    fn zeros_then(value: V) -> Self::Output {
        IntTuple(
            Int,
            <Bool<{ IS_ZERO::<{ PREV::<M> }> }> as ZerosThen<V, { PREV::<M> }>>::zeros_then(value),
        )
    }
}

// Integers add as integers.
impl<A, B> OffsetAdd<B> for A
where
    A: Integer + Arith<B, { Op::Add }>,
    B: Integer,
{
    type Output = Sum<A, B>;
    fn offset_add(self, rhs: B) -> Self::Output {
        self.apply(rhs)
    }
}

// A static zero is the additive identity of a basis or a coordinate tuple.
impl<V, const M: usize> OffsetAdd<Int<0>> for ScaledBasis<V, M> {
    type Output = Self;
    fn offset_add(self, _: Int<0>) -> Self {
        self
    }
}

impl<V, const M: usize> OffsetAdd<ScaledBasis<V, M>> for Int<0> {
    type Output = ScaledBasis<V, M>;
    fn offset_add(self, rhs: ScaledBasis<V, M>) -> Self::Output {
        rhs
    }
}

impl<H, T> OffsetAdd<Int<0>> for IntTuple<H, T> {
    type Output = Self;
    fn offset_add(self, _: Int<0>) -> Self {
        self
    }
}

impl<H, T> OffsetAdd<IntTuple<H, T>> for Int<0> {
    type Output = IntTuple<H, T>;
    fn offset_add(self, rhs: IntTuple<H, T>) -> Self::Output {
        rhs
    }
}

// Tuples add mode by mode; the shorter one is padded with zeros.
impl OffsetAdd<Nil> for Nil {
    type Output = Nil;
    fn offset_add(self, _: Nil) -> Nil {
        Nil
    }
}

impl<H, T> OffsetAdd<Nil> for IntTuple<H, T> {
    type Output = Self;
    fn offset_add(self, _: Nil) -> Self {
        self
    }
}

impl<H, T> OffsetAdd<IntTuple<H, T>> for Nil {
    type Output = IntTuple<H, T>;
    fn offset_add(self, rhs: IntTuple<H, T>) -> Self::Output {
        rhs
    }
}

impl<AH, AT, BH, BT> OffsetAdd<IntTuple<BH, BT>> for IntTuple<AH, AT>
where
    AH: OffsetAdd<BH>,
    AT: OffsetAdd<BT>,
{
    type Output = IntTuple<AH::Output, AT::Output>;
    fn offset_add(self, rhs: IntTuple<BH, BT>) -> Self::Output {
        IntTuple(self.0.offset_add(rhs.0), self.1.offset_add(rhs.1))
    }
}

// Bases add through their tuple form.
impl<V, const M: usize, W, const N: usize> OffsetAdd<ScaledBasis<W, N>> for ScaledBasis<V, M>
where
    Self: AsTuple,
    ScaledBasis<W, N>: AsTuple,
    <Self as AsTuple>::Output: OffsetAdd<<ScaledBasis<W, N> as AsTuple>::Output>,
{
    type Output =
        <<Self as AsTuple>::Output as OffsetAdd<<ScaledBasis<W, N> as AsTuple>::Output>>::Output;
    fn offset_add(self, rhs: ScaledBasis<W, N>) -> Self::Output {
        self.into_tuple().offset_add(rhs.into_tuple())
    }
}

impl<V, const M: usize, H, T> OffsetAdd<IntTuple<H, T>> for ScaledBasis<V, M>
where
    Self: AsTuple,
    <Self as AsTuple>::Output: OffsetAdd<IntTuple<H, T>>,
{
    type Output = <<Self as AsTuple>::Output as OffsetAdd<IntTuple<H, T>>>::Output;
    fn offset_add(self, rhs: IntTuple<H, T>) -> Self::Output {
        self.into_tuple().offset_add(rhs)
    }
}

impl<V, const M: usize, H, T> OffsetAdd<ScaledBasis<V, M>> for IntTuple<H, T>
where
    ScaledBasis<V, M>: AsTuple,
    Self: OffsetAdd<<ScaledBasis<V, M> as AsTuple>::Output>,
{
    type Output = <Self as OffsetAdd<<ScaledBasis<V, M> as AsTuple>::Output>>::Output;
    fn offset_add(self, rhs: ScaledBasis<V, M>) -> Self::Output {
        self.offset_add(rhs.into_tuple())
    }
}

impl<I: Integer> StrideKind for I {
    type Kind = Bool<false>;
    const BASIS: bool = false;
}

impl<V, const M: usize> StrideKind for ScaledBasis<V, M> {
    type Kind = Bool<true>;
    const BASIS: bool = true;
}

impl StrideKind for Nil {
    type Kind = Bool<false>;
    const BASIS: bool = false;
}

pub trait BasisUnion<R> {
    type Output;
}
impl<R> BasisUnion<R> for Bool<false> {
    type Output = R;
}
impl<R> BasisUnion<R> for Bool<true> {
    type Output = Self;
}

impl<H: StrideKind, T: StrideKind> StrideKind for IntTuple<H, T>
where
    H::Kind: BasisUnion<T::Kind>,
{
    type Kind = <H::Kind as BasisUnion<T::Kind>>::Output;
    const BASIS: bool = H::BASIS || T::BASIS;
}

impl<H: Integer, T, const POS: usize> BasisModes<POS> for IntTuple<H, T>
where
    T: BasisModes<{ NEXT::<POS> }>,
{
    type Stride = IntTuple<E<POS>, T::Stride>;
    type Zero = IntTuple<Int<0>, T::Zero>;
    fn basis() -> Self::Stride {
        IntTuple(ScaledBasis(Int), T::basis())
    }
    fn zero() -> Self::Zero {
        IntTuple(Int, T::zero())
    }
}

/// Shapes with an identity layout: integers and flat tuples of integers.
///
/// `Stride` is CuTe's `make_basis_like(shape)` and `Zero` is the zero coordinate
/// of the same profile, upstream's `coprofile` of the identity layout.
pub trait IdentityShape {
    /// The basis strides.
    type Stride: Copy;
    /// The zero coordinate.
    type Zero: Copy;
    /// Returns the basis strides.
    fn basis() -> Self::Stride;
    /// Returns the zero coordinate.
    fn zero() -> Self::Zero;
}

impl<I: Integer> IdentityShape for I {
    type Stride = Int<1>;
    type Zero = Int<0>;
    fn basis() -> Int<1> {
        Int
    }
    fn zero() -> Int<0> {
        Int
    }
}

impl IdentityShape for Nil {
    type Stride = Nil;
    type Zero = Nil;
    fn basis() -> Nil {
        Nil
    }
    fn zero() -> Nil {
        Nil
    }
}

impl<H, T> IdentityShape for IntTuple<H, T>
where
    Self: BasisModes<0>,
{
    type Stride = <Self as BasisModes<0>>::Stride;
    type Zero = <Self as BasisModes<0>>::Zero;
    fn basis() -> Self::Stride {
        <Self as BasisModes<0>>::basis()
    }
    fn zero() -> Self::Zero {
        <Self as BasisModes<0>>::zero()
    }
}

/// Maps each coordinate of `shape` to itself: CuTe `make_identity_layout(shape)`.
///
/// The strides are `make_basis_like(shape)`, so `layout.crd2idx(c)` is the
/// coordinate `c` in the profile of `shape`; an integer coordinate for a tuple
/// shape is decoded into one component per mode, with the last mode unbounded.
pub fn make_identity_layout<S: IdentityShape>(shape: S) -> Layout<S, S::Stride> {
    Layout {
        shape,
        stride: S::basis(),
    }
}

impl<V: crate::StaticValue, const M: usize> crate::StaticValue for ScaledBasis<V, M> {
    const VALUE: Self = Self(V::VALUE);
}

impl<V, const M: usize> crate::tuple::Flatten for ScaledBasis<V, M> {
    type Output = IntTuple<Self, Nil>;
    fn flatten(self) -> Self::Output {
        IntTuple(self, Nil)
    }
}

pub(crate) trait StrideProfile<S> {}
impl<S: Integer, D: Integer> StrideProfile<S> for D {}
impl<S: Integer, V: Integer, const M: usize> StrideProfile<S> for ScaledBasis<V, M> {}
impl StrideProfile<Nil> for Nil {}
impl<SH, ST, DH: StrideProfile<SH>, DT: StrideProfile<ST>> StrideProfile<IntTuple<SH, ST>>
    for IntTuple<DH, DT>
{
}
