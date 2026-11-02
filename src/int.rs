//! Integer leaves with compile-time or runtime values.
//!
//! Arithmetic on two [`Int`] values keeps the result in its type. Mixing an
//! `Int` with a [`DynInt`] or `isize` produces a `DynInt`, as does arithmetic
//! involving `DynInt`. Operators follow Rust's integer arithmetic rules.
//!
//! Unlike CuTe, a static zero does not absorb a runtime operand: `Int<0> * DynInt` is a
//! `DynInt`, while CuTe's `_0 * x` is `_0`. The operator impls choose one output type per
//! operand kind rather than dispatching on `N`, which keeps their outputs concrete types;
//! the layout algebra applies CuTe's static-zero rules itself where they change a result type.
//!
//! Native `i32` leaves store 32-bit runtime values, matching CuTe's common `int` extents. The
//! operators above do not accept `i32` operands; tuple products, compact strides, and
//! coordinate maps combine leaves with the rules described in [`Integer`].
//!
//! ```
//! use noodle_cute::{DynInt, Int};
//!
//! let sum: Int<5> = Int::<2> + Int::<3>;
//! assert_eq!(sum.value(), 5);
//! let product: DynInt = Int::<2> * DynInt(3);
//! assert_eq!(product.value(), 6);
//! ```

use core::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::{Error, Result};

/// A zero-sized integer whose value `N` is encoded in its type.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(C)]
pub struct Int<const N: isize>;

impl<const N: isize> Int<N> {
    /// The integer value encoded in this type.
    pub const VALUE: isize = N;

    /// Returns `N` as an `isize`.
    pub const fn value(self) -> isize {
        N
    }
}

impl<const N: isize> From<Int<N>> for isize {
    fn from(_: Int<N>) -> Self {
        N
    }
}

/// An integer stored at runtime rather than encoded in its type.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct DynInt(pub isize);

impl DynInt {
    /// Returns the stored value.
    pub const fn value(self) -> isize {
        self.0
    }
}

impl From<isize> for DynInt {
    fn from(value: isize) -> Self {
        Self(value)
    }
}

// Restrict `Integer` implementations to the integer representations below.
mod sealedcontainer {
    use super::{DynInt, Int};
    pub trait SealedInteger {}
    impl<const N: isize> SealedInteger for Int<N> {}
    impl SealedInteger for DynInt {}
    impl SealedInteger for isize {}
    impl SealedInteger for i32 {}
}

/// A shared interface for `Int<N>`, `DynInt`, and native `isize` and `i32` leaves.
///
/// This trait is sealed; downstream crates cannot implement it.
///
/// Tuple products, compact strides, and coordinate maps combine leaves with
/// CuTe's rules: two static operands give a static result, and a static zero
/// absorbs multiplication (`_0 * x = _0`), division of zero (`_0 / x = _0`),
/// and remainders (`_0 % x = x % _1 = x % _-1 = _0`). Other runtime results are
/// `i32` when every runtime operand is `i32`, and [`DynInt`] when any operand is
/// `DynInt` or `isize`. A static operand combined with an `i32` must fit in `i32`.
///
/// ```
/// use noodle_cute::tuple::Product;
/// use noodle_cute::{DynInt, Int, int_tuple};
///
/// let narrow: i32 = int_tuple![Int::<2>, 3i32].product_typed();
/// let wide: DynInt = int_tuple![4i32, DynInt(5)].product_typed();
/// let zero: Int<0> = int_tuple![Int::<0>, DynInt(5)].product_typed();
/// assert_eq!((narrow, wide, zero.value()), (6, DynInt(20), 0));
/// ```
///
/// A static value outside the `i32` range cannot combine with an `i32` leaf:
///
/// ```compile_fail
/// use noodle_cute::tuple::Product;
/// use noodle_cute::{Int, int_tuple};
/// let _ = int_tuple![Int::<{ 1 << 40 }>, 3i32].product_typed();
/// ```
/// Its implementations support constant evaluation. A generic const caller
/// uses `[const] Integer` and enables `const_trait_impl`:
///
/// ```
/// #![feature(const_trait_impl)]
/// use noodle_cute::{DynInt, Integer};
/// const fn value<I: [const] Integer>(integer: I) -> isize {
///     integer.value()
/// }
/// const N: isize = value(DynInt(7));
/// assert_eq!(N, 7);
/// ```
pub const trait Integer: sealedcontainer::SealedInteger + Copy {
    /// The value known from the type, or `None` for a runtime representation.
    ///
    /// A `DynInt` or `isize` always reports `None`, even if its value is constant.
    const STATIC: Option<isize>;

    /// Returns the leaf's integer value, regardless of how it is represented.
    fn value(self) -> isize;
}

const impl<const N: isize> Integer for Int<N> {
    const STATIC: Option<isize> = Some(N);
    fn value(self) -> isize {
        N
    }
}

const impl Integer for DynInt {
    const STATIC: Option<isize> = None;
    fn value(self) -> isize {
        self.0
    }
}

const impl Integer for isize {
    const STATIC: Option<isize> = None;
    fn value(self) -> isize {
        self
    }
}

const impl Integer for i32 {
    const STATIC: Option<isize> = None;
    fn value(self) -> isize {
        self as isize
    }
}

// Keep static results in `Int`; promote every supported runtime combination
// to `DynInt`. Native `isize`-to-`isize` operators are provided by Rust itself.
macro_rules! binary {
    ($trait:ident, $method:ident, $op:tt, $result:ident) => {
        pub(crate) const $result<const A: isize, const B: isize>: isize = A $op B;

        impl<const A: isize, const B: isize> $trait<Int<B>> for Int<A>
        where
            Int<{ $result::<A, B> }>: Sized,
        {
            type Output = Int<{ $result::<A, B> }>;
            fn $method(self, _: Int<B>) -> Self::Output { Int }
        }

        impl<const A: isize> $trait<DynInt> for Int<A> {
            type Output = DynInt;
            fn $method(self, rhs: DynInt) -> Self::Output { DynInt(A $op rhs.0) }
        }

        impl<const A: isize> $trait<isize> for Int<A> {
            type Output = DynInt;
            fn $method(self, rhs: isize) -> Self::Output { DynInt(A $op rhs) }
        }

        impl<const B: isize> $trait<Int<B>> for DynInt {
            type Output = DynInt;
            fn $method(self, _: Int<B>) -> Self::Output { DynInt(self.0 $op B) }
        }

        impl $trait for DynInt {
            type Output = DynInt;
            fn $method(self, rhs: DynInt) -> Self::Output { DynInt(self.0 $op rhs.0) }
        }

        impl $trait<isize> for DynInt {
            type Output = DynInt;
            fn $method(self, rhs: isize) -> Self::Output { DynInt(self.0 $op rhs) }
        }

        impl<const B: isize> $trait<Int<B>> for isize {
            type Output = DynInt;
            fn $method(self, _: Int<B>) -> Self::Output { DynInt(self $op B) }
        }

        impl $trait<DynInt> for isize {
            type Output = DynInt;
            fn $method(self, rhs: DynInt) -> Self::Output { DynInt(self $op rhs.0) }
        }
    };
}

binary!(Add, add, +, ADD_RESULT);
binary!(Sub, sub, -, SUB_RESULT);
binary!(Mul, mul, *, MUL_RESULT);
binary!(Div, div, /, DIV_RESULT);
binary!(Rem, rem, %, REM_RESULT);

pub(crate) const NEG_RESULT<const N: isize>: isize = -N;

impl<const N: isize> Neg for Int<N>
where
    Int<{ NEG_RESULT::<N> }>: Sized,
{
    type Output = Int<{ NEG_RESULT::<N> }>;
    fn neg(self) -> Self::Output {
        Int
    }
}

impl Neg for DynInt {
    type Output = DynInt;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

pub(crate) const NON_NEGATIVE<const N: isize>: bool = N >= 0;

// CuTe's mixed static/dynamic integer rules for tuple products, compact strides, and
// coordinate maps; the public operators above keep their simpler output types.
mod typed {
    use core::marker::ConstParamTy;

    use super::{DynInt, Int, Integer, ceil_div};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
    pub enum Op {
        Add,
        Mul,
        Div,
        Rem,
        /// Ceiling division of a nonnegative dividend by a positive divisor.
        CeilDiv,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
    pub enum Repr {
        Static,
        Narrow,
        Wide,
    }

    pub trait Leaf: Integer {
        const REPR: Repr;
    }
    impl<const N: isize> Leaf for Int<N> {
        const REPR: Repr = Repr::Static;
    }
    impl Leaf for i32 {
        const REPR: Repr = Repr::Narrow;
    }
    impl Leaf for DynInt {
        const REPR: Repr = Repr::Wide;
    }
    impl Leaf for isize {
        const REPR: Repr = Repr::Wide;
    }

    const fn fold(op: Op, a: isize, b: isize) -> isize {
        match op {
            Op::Add => a + b,
            Op::Mul => a * b,
            Op::Div => a / b,
            Op::Rem => a % b,
            Op::CeilDiv => ceil_div(a, b),
        }
    }

    // integral_constant.hpp: `_0 * x`, `x * _0`, `_0 / x`, `_0 % x`, and `x % _1` stay static.
    const fn static_result(op: Op, a: Option<isize>, b: Option<isize>) -> Option<isize> {
        match (op, a, b) {
            (_, Some(a), Some(b)) => Some(fold(op, a, b)),
            (Op::Mul, Some(0), _)
            | (Op::Mul, _, Some(0))
            | (Op::Div, Some(0), _)
            | (Op::Rem, Some(0), _)
            | (Op::Rem, _, Some(1 | -1)) => Some(0),
            _ => None,
        }
    }

    const fn fits_narrow(value: Option<isize>) -> bool {
        match value {
            Some(v) => v >= i32::MIN as isize && v <= i32::MAX as isize,
            None => true,
        }
    }

    pub const RESULT_REPR<A: Leaf, B: Leaf, const OP: Op>: Repr =
        if static_result(OP, A::STATIC, B::STATIC).is_some() {
            Repr::Static
        } else if matches!(A::REPR, Repr::Wide) || matches!(B::REPR, Repr::Wide) {
            Repr::Wide
        } else {
            assert!(
                fits_narrow(A::STATIC) && fits_narrow(B::STATIC),
                "a static operand combined with i32 must fit in i32"
            );
            Repr::Narrow
        };
    pub const RESULT_VALUE<A: Leaf, B: Leaf, const OP: Op>: isize =
        match static_result(OP, A::STATIC, B::STATIC) {
            Some(v) => v,
            None => 0,
        };

    pub struct Width<const R: Repr>;

    pub trait Pick<const N: isize> {
        type Output: Integer;
        fn apply(op: Op, a: isize, b: isize) -> Self::Output;
        fn checked(op: Op, a: isize, b: isize) -> Option<Self::Output>;
    }
    impl<const N: isize> Pick<N> for Width<{ Repr::Static }> {
        type Output = Int<N>;
        fn apply(_: Op, _: isize, _: isize) -> Int<N> {
            Int
        }
        fn checked(_: Op, _: isize, _: isize) -> Option<Int<N>> {
            Some(Int)
        }
    }
    // Narrow operands are `i32` leaves or static values checked by `RESULT_REPR`.
    impl<const N: isize> Pick<N> for Width<{ Repr::Narrow }> {
        type Output = i32;
        fn apply(op: Op, a: isize, b: isize) -> i32 {
            let (a, b) = (a as i32, b as i32);
            match op {
                Op::Add => a + b,
                Op::Mul => a * b,
                Op::Div => a / b,
                Op::Rem => a % b,
                Op::CeilDiv => a / b + (a % b != 0) as i32,
            }
        }
        fn checked(op: Op, a: isize, b: isize) -> Option<i32> {
            let (a, b) = (a as i32, b as i32);
            match op {
                Op::Add => a.checked_add(b),
                Op::Mul => a.checked_mul(b),
                Op::Div => a.checked_div(b),
                Op::Rem => a.checked_rem(b),
                Op::CeilDiv => a
                    .checked_div(b)?
                    .checked_add((a.checked_rem(b)? != 0) as i32),
            }
        }
    }
    impl<const N: isize> Pick<N> for Width<{ Repr::Wide }> {
        type Output = DynInt;
        fn apply(op: Op, a: isize, b: isize) -> DynInt {
            DynInt(fold(op, a, b))
        }
        fn checked(op: Op, a: isize, b: isize) -> Option<DynInt> {
            match op {
                Op::Add => a.checked_add(b),
                Op::Mul => a.checked_mul(b),
                Op::Div => a.checked_div(b),
                Op::Rem => a.checked_rem(b),
                Op::CeilDiv => a
                    .checked_div(b)
                    .zip(a.checked_rem(b))
                    .map(|(q, r)| q + (r != 0) as isize),
            }
            .map(DynInt)
        }
    }

    pub trait Arith<B, const OP: Op>: Integer {
        type Output: Integer;
        /// Uses Rust's integer operators: overflow panics in debug builds.
        fn apply(self, rhs: B) -> Self::Output;
        fn checked(self, rhs: B) -> Option<Self::Output>;
    }
    impl<A: Leaf, B: Leaf, const OP: Op> Arith<B, OP> for A
    where
        Width<{ RESULT_REPR::<A, B, OP> }>: Pick<{ RESULT_VALUE::<A, B, OP> }>,
    {
        type Output =
            <Width<{ RESULT_REPR::<A, B, OP> }> as Pick<{ RESULT_VALUE::<A, B, OP> }>>::Output;
        fn apply(self, rhs: B) -> Self::Output {
            <Width<{ RESULT_REPR::<A, B, OP> }> as Pick<{ RESULT_VALUE::<A, B, OP> }>>::apply(
                OP,
                self.value(),
                rhs.value(),
            )
        }
        fn checked(self, rhs: B) -> Option<Self::Output> {
            <Width<{ RESULT_REPR::<A, B, OP> }> as Pick<{ RESULT_VALUE::<A, B, OP> }>>::checked(
                OP,
                self.value(),
                rhs.value(),
            )
        }
    }
}
pub(crate) use typed::{Arith, Op};

pub(crate) type Sum<A, B> = <A as Arith<B, { Op::Add }>>::Output;
pub(crate) type Prod<A, B> = <A as Arith<B, { Op::Mul }>>::Output;
pub(crate) type Quot<A, B> = <A as Arith<B, { Op::Div }>>::Output;
pub(crate) type Modulo<A, B> = <A as Arith<B, { Op::Rem }>>::Output;
pub(crate) type CeilQuot<A, B> = <A as Arith<B, { Op::CeilDiv }>>::Output;

pub(crate) fn sum<A: Arith<B, { Op::Add }>, B>(a: A, b: B) -> Sum<A, B> {
    a.apply(b)
}
pub(crate) fn prod<A: Arith<B, { Op::Mul }>, B>(a: A, b: B) -> Prod<A, B> {
    a.apply(b)
}
pub(crate) fn quot<A: Arith<B, { Op::Div }>, B>(a: A, b: B) -> Quot<A, B> {
    a.apply(b)
}
pub(crate) fn modulo<A: Arith<B, { Op::Rem }>, B>(a: A, b: B) -> Modulo<A, B> {
    a.apply(b)
}
pub(crate) fn ceil_quot<A: Arith<B, { Op::CeilDiv }>, B>(a: A, b: B) -> CeilQuot<A, B> {
    a.apply(b)
}

/// Adds two integers, returning `Error::Overflow` if the sum cannot fit.
pub(crate) const fn add(a: isize, b: isize) -> Result<isize> {
    match a.checked_add(b) {
        Some(value) => Ok(value),
        None => Err(Error::Overflow),
    }
}

/// Multiplies two integers, returning `Error::Overflow` if the product cannot fit.
pub(crate) const fn mul(a: isize, b: isize) -> Result<isize> {
    match a.checked_mul(b) {
        Some(value) => Ok(value),
        None => Err(Error::Overflow),
    }
}

/// Computes ceiling division without forming the potentially overflowing `a + b - 1`.
/// The caller must ensure `a >= 0` and `b > 0`.
pub(crate) const fn ceil_div(a: isize, b: isize) -> isize {
    a / b + (a % b != 0) as isize
}

/// Creates a compile-time integer.
///
/// This macro can be used to create an `Int<N>` type for a given integer literal `N`.
/// Example:
/// ```
/// use noodle_cute::{Int, i};
///
/// let x: Int<5> = i!(5);
/// ```
#[macro_export]
macro_rules! i {
    ($n:expr) => {
        $crate::Int::<$n>
    };
}
