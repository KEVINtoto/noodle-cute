//! Address-bit swizzles: CuTe `Swizzle<B, M, S>` (`swizzle.hpp`).
//!
//! A swizzle XORs the `B` bits starting at bit `M + max(0, S)` (the YYY mask)
//! into the `B` bits starting at bit `M - min(0, S)` (the ZZZ mask). Composed
//! after a layout it permutes shared-memory addresses to avoid bank conflicts
//! without changing which coordinates exist. Applying it twice is the identity.
//!
//! ```
//! use noodle_cute::swizzle::Swizzle;
//!
//! type Sw = Swizzle<3, 3, 3>;
//! assert_eq!(Sw::apply(64), 72); // bit 6 (YYY) flips bit 3 (ZZZ)
//! assert_eq!(Sw::apply(72), 64);
//! assert_eq!(Sw::apply(7), 7); // the low M = 3 bits never change
//! ```

use crate::int::{DynInt, Int};

/// XORs the `B`-bit field at `M + max(0, S)` into the field at `M - min(0, S)`:
/// CuTe `Swizzle<BBits, MBase, SShift>`.
///
/// As upstream, `|S| >= B` keeps the two fields disjoint; violating it fails to compile.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Swizzle<const B: u32, const M: u32, const S: i32>;

impl<const B: u32, const M: u32, const S: i32> Swizzle<B, M, S> {
    const VALID: () = assert!(S.unsigned_abs() >= B, "abs(SShift) must be at least BBits");
    /// The `B`-bit mask `(1 << B) - 1`.
    pub const BIT_MSK: isize = (1 << B) - 1;
    /// The source field, CuTe `yyy_msk`.
    pub const YYY_MSK: isize = Self::BIT_MSK << (M as i32 + if S > 0 { S } else { 0 });
    /// The destination field, CuTe `zzz_msk`.
    pub const ZZZ_MSK: isize = Self::BIT_MSK << (M as i32 - if S < 0 { S } else { 0 });
    /// Both fields, CuTe `swizzle_code`.
    pub const SWIZZLE_CODE: u32 = (Self::YYY_MSK | Self::ZZZ_MSK) as u32;

    /// Swizzles an offset: `offset ^ shiftr(offset & yyy_msk, S)`.
    pub const fn apply(offset: isize) -> isize {
        let () = Self::VALID;
        let y = offset & Self::YYY_MSK;
        offset ^ if S >= 0 { y >> S } else { y << -S }
    }
}

// Implementation traits that appear in public bounds but are not part of the API.
mod detail {
    use crate::int::Integer;

    /// Applies a swizzle to a typed offset, keeping static offsets static.
    pub trait Swizzled<const B: u32, const M: u32, const S: i32>: Integer {
        type Output: Integer;
        fn swizzled(self) -> Self::Output;
    }
}
pub(crate) use detail::Swizzled;

/// The swizzled value of a static offset.
pub const SWIZZLED<const B: u32, const M: u32, const S: i32, const N: isize>: isize =
    Swizzle::<B, M, S>::apply(N);

impl<const B: u32, const M: u32, const S: i32, const N: isize> Swizzled<B, M, S> for Int<N> {
    type Output = Int<{ SWIZZLED::<B, M, S, N> }>;
    fn swizzled(self) -> Self::Output {
        Int
    }
}

impl<const B: u32, const M: u32, const S: i32> Swizzled<B, M, S> for DynInt {
    type Output = DynInt;
    fn swizzled(self) -> DynInt {
        DynInt(Swizzle::<B, M, S>::apply(self.0))
    }
}

impl<const B: u32, const M: u32, const S: i32> Swizzled<B, M, S> for isize {
    type Output = isize;
    fn swizzled(self) -> isize {
        Swizzle::<B, M, S>::apply(self)
    }
}

// The masks of a valid swizzle on an `i32` offset lie below bit 31 whenever the
// offset itself fits, so the result fits in `i32` too.
impl<const B: u32, const M: u32, const S: i32> Swizzled<B, M, S> for i32 {
    type Output = i32;
    fn swizzled(self) -> i32 {
        Swizzle::<B, M, S>::apply(self as isize) as i32
    }
}

/// Element-width changes of a swizzle: CuTe `upcast` / `downcast` of `Swizzle`.
///
/// Upcasting by `N = 2^k` moves the base down by `k` bits, clamping at zero by
/// dropping low bits of the mask; downcasting moves it up. `N` must be a power
/// of two.
pub const fn upcast_swizzle(b: u32, m: u32, n: u32) -> (u32, u32) {
    assert!(n.is_power_of_two(), "N must be a power of two");
    let log2 = n.trailing_zeros();
    if m >= log2 {
        (b, m - log2)
    } else {
        // Swizzle<max(B + NewM, 0), 0, S> with NewM = M - log2 < 0.
        (b.saturating_sub(log2 - m), 0)
    }
}

/// See [`upcast_swizzle`].
pub const fn downcast_swizzle(m: u32, n: u32) -> u32 {
    assert!(n.is_power_of_two(), "N must be a power of two");
    m + n.trailing_zeros()
}

/// The `(B, M, S)` of the swizzle with source mask `y` and destination mask `z`:
/// CuTe `make_swizzle<Y, Z>()`.
///
/// The masks must have the same number of bits, and together they must be the
/// fields of a swizzle.
pub const fn swizzle_params(y: u32, z: u32) -> (u32, u32, i32) {
    let b = y.count_ones();
    assert!(b == z.count_ones(), "Number of bits in Y and Z don't match");
    let (ty, tz) = (y.trailing_zeros(), z.trailing_zeros());
    let m = (if ty < tz { ty } else { tz }) % 32;
    let s = ty as i32 - tz as i32;
    let bits = if b == 0 { 0 } else { u32::MAX >> (32 - b) };
    let yyy = bits << (m as i32 + if s > 0 { s } else { 0 });
    let zzz = bits << (m as i32 - if s < 0 { s } else { 0 });
    assert!(
        y | z == yyy | zzz,
        "the masks are not the fields of a swizzle"
    );
    (b, m, s)
}

/// `B` of [`MakeSwizzle`].
pub const MADE_B<const Y: u32, const Z: u32>: u32 = swizzle_params(Y, Z).0;
/// `M` of [`MakeSwizzle`].
pub const MADE_M<const Y: u32, const Z: u32>: u32 = swizzle_params(Y, Z).1;
/// `S` of [`MakeSwizzle`].
pub const MADE_S<const Y: u32, const Z: u32>: i32 = swizzle_params(Y, Z).2;

/// The swizzle with source mask `Y` and destination mask `Z`: the type of CuTe `make_swizzle<Y, Z>()`.
pub type MakeSwizzle<const Y: u32, const Z: u32> =
    Swizzle<{ MADE_B::<Y, Z> }, { MADE_M::<Y, Z> }, { MADE_S::<Y, Z> }>;

/// CuTe `make_swizzle<Y, Z>()`.
///
/// ```
/// use noodle_cute::swizzle::{Swizzle, make_swizzle};
///
/// assert_eq!(make_swizzle::<0b1000, 0b0100>(), Swizzle::<1, 2, 1>);
/// assert_eq!(make_swizzle::<0b1100_0000, 0b0000_0110>(), Swizzle::<2, 1, 5>);
/// ```
pub fn make_swizzle<const Y: u32, const Z: u32>() -> MakeSwizzle<Y, Z> {
    Swizzle
}

/// The source mask of `composition(Swizzle<B0, M0, S0>, Swizzle<B1, M1, S1>)`.
pub const MERGED_Y<const B0: u32, const M0: u32, const S0: i32, const B1: u32, const M1: u32, const S1: i32>: u32 = {
    assert!(S0 == S1, "Can only merge swizzles of the same shift.");
    (Swizzle::<B0, M0, S0>::YYY_MSK ^ Swizzle::<B1, M1, S1>::YYY_MSK) as u32
};
/// The destination mask of `composition(Swizzle<B0, M0, S0>, Swizzle<B1, M1, S1>)`.
pub const MERGED_Z<const B0: u32, const M0: u32, const S0: i32, const B1: u32, const M1: u32, const S1: i32>: u32 =
    (Swizzle::<B0, M0, S0>::ZZZ_MSK ^ Swizzle::<B1, M1, S1>::ZZZ_MSK) as u32;
