//! Integer leaves with compile-time or runtime values.
//!
//! Arithmetic on two [`Int`] values keeps the result in its type. Mixing an
//! `Int` with a [`DynInt`] or `isize` produces a `DynInt`, as does arithmetic
//! involving `DynInt`. Operators follow Rust's integer arithmetic rules.
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
}

/// A shared interface for `Int<N>`, `DynInt`, and native `isize` leaves.
///
/// This trait is sealed; downstream crates cannot implement it.
pub trait Integer: sealedcontainer::SealedInteger + Copy {
    /// The value known from the type, or `None` for a runtime representation.
    ///
    /// A `DynInt` or `isize` always reports `None`, even if its value is constant.
    const STATIC: Option<isize>;

    /// Returns the leaf's integer value, regardless of how it is represented.
    fn value(self) -> isize;
}

impl<const N: isize> Integer for Int<N> {
    const STATIC: Option<isize> = Some(N);
    fn value(self) -> isize {
        N
    }
}

impl Integer for DynInt {
    const STATIC: Option<isize> = None;
    fn value(self) -> isize {
        self.0
    }
}

impl Integer for isize {
    const STATIC: Option<isize> = None;
    fn value(self) -> isize {
        self
    }
}

// Keep static results in `Int`; promote every supported runtime combination
// to `DynInt`. Native `isize`-to-`isize` operators are provided by Rust itself.
macro_rules! binary {
    ($trait:ident, $method:ident, $op:tt, $result:ident) => {
        /// The result of applying this operator to compile-time operands.
        pub const $result<const A: isize, const B: isize>: isize = A $op B;

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

/// The negation of a compile-time integer, used as the result type of `-Int<N>`.
pub const NEG_RESULT<const N: isize>: isize = -N;

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

/// Whether `N` is nonnegative, for use in compile-time shape constraints.
pub const NON_NEGATIVE<const N: isize>: bool = N >= 0;

/// Adds two runtime values, returning `Error::Overflow` if the sum cannot fit.
pub(crate) fn add(a: isize, b: isize) -> Result<isize> {
    a.checked_add(b).ok_or(Error::Overflow)
}

/// Multiplies two runtime values, returning `Error::Overflow` if the product cannot fit.
pub(crate) fn mul(a: isize, b: isize) -> Result<isize> {
    a.checked_mul(b).ok_or(Error::Overflow)
}

#[allow(dead_code)]
/// Computes ceiling division without forming the potentially overflowing `a + b - 1`.
/// The caller must ensure `a >= 0` and `b > 0`.
pub(crate) fn ceil_div(a: isize, b: isize) -> isize {
    a / b + isize::from(a % b != 0)
}

#[allow(dead_code)]
/// Computes a checked product, treating an empty slice as one and any zero as zero.
/// This helper does not validate the signs of shape extents.
pub(crate) fn product_slice(values: &[isize]) -> Result<isize> {
    // Detect empty domains before multiplying to avoid irrelevant intermediate overflow.
    if values.contains(&0) {
        return Ok(0);
    }
    let mut n = 1;
    for &v in values {
        n = mul(n, v)?;
    }
    Ok(n)
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
        Int::<$n>
    };
}
