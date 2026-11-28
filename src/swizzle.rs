//! Bit-field XOR permutation functor (`Swizzle`).
//!
//! Applies the CuTe bit-swizzle transformation `offset ^ shiftr(offset & yyy_msk, shift)`
//! to integer offsets (and [`F2`] values), with compile-time mask folding and
//! zero-sized representation when parameters are static [`Int<N>`](crate::integer::Int).
//!
//! # Main Interfaces
//!
//! - [`Swizzle<Bits, Base, Shift>`]: Generic static/dynamic swizzle functor, callable via
//!   `Fn`/`FnMut`/`FnOnce` or [`Swizzle::call`].
//! - [`StaticSwizzle<BITS, BASE, SHIFT>`]: Type alias for compile-time static swizzles.
//! - [`crate::swizzle!`]: Constructs a swizzle from its bit count, base and shift.
//! - [`SwizzleApply`]: Trait backing [`Swizzle::call`] and function-call operators.
//! - **Swizzle helper algebra** (requires involution `bits == 0 || |shift| >= bits`):
//!   - [`swizzle_params`], [`MakeSwizzle`], and [`make_swizzle`]: Derives `(bits, base, shift)`
//!     from disjoint source/target bit masks `(Y, Z)`.
//!   - [`ComposeSwizzle`] / [`compose_swizzle`]: Merges two involutive swizzles sharing the same shift.
//!   - [`UpcastSwizzle`] / [`upcast_swizzle`], [`DowncastSwizzle`] / [`downcast_swizzle`],
//!     and [`recast_swizzle`]: Adjusts swizzle bit base/width across power-of-two element scales.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::swizzle::Swizzle;
//!
//! const SW: Swizzle<Int<1>, Int<0>, Int<1>> = Swizzle::new(Int::<1>, Int::<0>, Int::<1>);
//! assert_eq!(SW(Int::<0b10>), Int::<0b11>);
//! assert_eq!(SW(SW(0b10i32)), 0b10i32);
//! ```
//!
//! # Notes
//!
//! - Pre-conditions: `bits >= 0`, `base >= 0`, `shift >= 0 || |shift| >= bits`, and
//!   (when `bits > 0`) `base + |shift| + bits <= 31` so masks fit within non-negative [`int`].
//! - Static violations fail at compile time; dynamic violations panic at runtime.

use core::fmt::{Display, Formatter};

use crate::f2::{F2, F2Scalar, IntoF2, f2};
use crate::integer::{
    ArithBitAnd, ArithBitXor, ArithShiftR, BitAndResult, BitXorResult, Int, Integer, ShiftRResult,
    add, bit_length_non_neg, bitand, bitxor, eq, int, is_pow2, max, min, neg, shiftl, shiftr, sub,
    uint,
};
use crate::layout::{RECAST_DEN, RECAST_NUM};
use crate::tuple::{Item, LeafProfile};
use crate::{NoodleCuteError, StaticValue};

const MAX_SWIZZLE_BITS: int = bit_length_non_neg(int::MAX);

const fn fits_swizzle_width(bits: int, base: int, shift: int) -> bool {
    if bits == 0 {
        return true;
    }
    let max_base = sub(MAX_SWIZZLE_BITS, bits);
    if base > max_base {
        return false;
    }
    let max_shift = sub(max_base, base);
    shift >= neg(max_shift) && shift <= max_shift
}

const fn validate_swizzle_values(bits: int, base: int, shift: int) {
    NoodleCuteError::InvalidShape.check_with(bits >= 0, "Swizzle requires bits >= 0");
    NoodleCuteError::InvalidShape.check_with(base >= 0, "Swizzle requires base >= 0");
    NoodleCuteError::InvalidShape.check_with(
        shift >= 0 || shift <= neg(bits),
        "Swizzle requires shift >= 0 or |shift| >= bits",
    );
    NoodleCuteError::InvalidShape.check_with(
        fits_swizzle_width(bits, base, shift),
        "Swizzle bit masks exceed representable non-negative int width",
    );
}

const fn static_swizzle_valid<Bits: Integer, Base: Integer, Shift: Integer>() -> bool {
    if let Some(b) = Bits::STATIC_VALUE
        && (b < 0 || b > MAX_SWIZZLE_BITS)
    {
        return false;
    }
    if let Some(m) = Base::STATIC_VALUE
        && m < 0
    {
        return false;
    }
    if let (Some(b), Some(s)) = (Bits::STATIC_VALUE, Shift::STATIC_VALUE)
        && ((s < 0 && s > neg(b)) || !fits_swizzle_width(b, 0, s))
    {
        return false;
    }
    if let (Some(b), Some(m)) = (Bits::STATIC_VALUE, Base::STATIC_VALUE)
        && !fits_swizzle_width(b, m, 0)
    {
        return false;
    }
    if let (Some(b), Some(m), Some(s)) =
        (Bits::STATIC_VALUE, Base::STATIC_VALUE, Shift::STATIC_VALUE)
    {
        return fits_swizzle_width(b, m, s);
    }
    true
}

/// Compile-time guard validating static [`Swizzle`] parameters (`VALID == true`).
#[doc(hidden)]
pub const trait ValidSwizzle<const VALID: bool> {}
const impl ValidSwizzle<true> for () {}

/// Compile-time validity predicate for [`Swizzle`] parameter types.
#[doc(hidden)]
pub const SWIZZLE_VALID<Bits: Integer, Base: Integer, Shift: Integer>: bool =
    static_swizzle_valid::<Bits, Base, Shift>();

const fn bit_msk_val(bits: int) -> int {
    shiftr(int::MAX, sub(MAX_SWIZZLE_BITS, bits))
}

const fn yyy_msk_val(bits: int, base: int, shift: int) -> int {
    if bits == 0 {
        0
    } else {
        shiftl(bit_msk_val(bits), add(base, max(0, shift)))
    }
}

const fn zzz_msk_val(bits: int, base: int, shift: int) -> int {
    if bits == 0 {
        0
    } else {
        let extra = if shift < 0 { neg(shift) } else { 0 };
        shiftl(bit_msk_val(bits), add(base, extra))
    }
}

const BIT_MASK_IS_STATIC<Bits: Integer>: bool = Bits::IS_STATIC;
const STATIC_BIT_MASK<Bits: Integer>: int =
    bit_msk_val(Bits::STATIC_VALUE.expect("expected static bits"));

const FIELD_MASKS_ARE_STATIC<Bits: Integer, Base: Integer, Shift: Integer>: bool =
    matches!(Bits::STATIC_VALUE, Some(0))
        || (Bits::IS_STATIC && Base::IS_STATIC && Shift::IS_STATIC);

const STATIC_YYY_MASK<Bits: Integer, Base: Integer, Shift: Integer>: int =
    if matches!(Bits::STATIC_VALUE, Some(0)) {
        0
    } else {
        yyy_msk_val(
            Bits::STATIC_VALUE.expect("expected static bits"),
            Base::STATIC_VALUE.expect("expected static base"),
            Shift::STATIC_VALUE.expect("expected static shift"),
        )
    };

const STATIC_ZZZ_MASK<Bits: Integer, Base: Integer, Shift: Integer>: int =
    if matches!(Bits::STATIC_VALUE, Some(0)) {
        0
    } else {
        zzz_msk_val(
            Bits::STATIC_VALUE.expect("expected static bits"),
            Base::STATIC_VALUE.expect("expected static base"),
            Shift::STATIC_VALUE.expect("expected static shift"),
        )
    };

/// Computes `bit_msk = (1 << bits) - 1`, preserving static `Int<M>` when `STATIC == true`.
#[doc(hidden)]
pub const trait SwizzleBitMask<const STATIC: bool>: [const] Integer {
    type Output: [const] Integer;
    fn bit_mask(self) -> Self::Output;
}

const impl<Bits: [const] Integer> SwizzleBitMask<true> for Bits {
    type Output = Int<{ STATIC_BIT_MASK::<Bits> }>;
    fn bit_mask(self) -> Self::Output {
        Int
    }
}

const impl<Bits: [const] Integer> SwizzleBitMask<false> for Bits {
    type Output = int;
    fn bit_mask(self) -> Self::Output {
        bit_msk_val(self.value())
    }
}

/// Computes `yyy_msk` and `zzz_msk`, preserving static `Int<M>` when `STATIC == true`.
#[doc(hidden)]
pub const trait SwizzleFieldMasks<Base: [const] Integer, Shift: [const] Integer, const STATIC: bool>:
    [const] Integer
{
    type YyyOutput: [const] Integer;
    type ZzzOutput: [const] Integer;
    fn yyy_mask(self, base: Base, shift: Shift) -> Self::YyyOutput;
    fn zzz_mask(self, base: Base, shift: Shift) -> Self::ZzzOutput;
}

const impl<Bits: [const] Integer, Base: [const] Integer, Shift: [const] Integer>
    SwizzleFieldMasks<Base, Shift, true> for Bits
{
    type YyyOutput = Int<{ STATIC_YYY_MASK::<Bits, Base, Shift> }>;
    type ZzzOutput = Int<{ STATIC_ZZZ_MASK::<Bits, Base, Shift> }>;

    fn yyy_mask(self, _: Base, _: Shift) -> Self::YyyOutput {
        Int
    }

    fn zzz_mask(self, _: Base, _: Shift) -> Self::ZzzOutput {
        Int
    }
}

const impl<Bits: [const] Integer, Base: [const] Integer, Shift: [const] Integer>
    SwizzleFieldMasks<Base, Shift, false> for Bits
{
    type YyyOutput = int;
    type ZzzOutput = int;

    fn yyy_mask(self, base: Base, shift: Shift) -> Self::YyyOutput {
        yyy_msk_val(self.value(), base.value(), shift.value())
    }

    fn zzz_mask(self, base: Base, shift: Shift) -> Self::ZzzOutput {
        zzz_msk_val(self.value(), base.value(), shift.value())
    }
}

/// Output type of [`Swizzle::bit_msk`].
pub type SwizzleBitMaskOutput<Bits> =
    <Bits as SwizzleBitMask<{ BIT_MASK_IS_STATIC::<Bits> }>>::Output;

/// Output type of [`Swizzle::yyy_msk`].
pub type SwizzleYyyMaskOutput<Bits, Base, Shift> = <Bits as SwizzleFieldMasks<
    Base,
    Shift,
    { FIELD_MASKS_ARE_STATIC::<Bits, Base, Shift> },
>>::YyyOutput;

/// Output type of [`Swizzle::zzz_msk`].
pub type SwizzleZzzMaskOutput<Bits, Base, Shift> = <Bits as SwizzleFieldMasks<
    Base,
    Shift,
    { FIELD_MASKS_ARE_STATIC::<Bits, Base, Shift> },
>>::ZzzOutput;

/// Generic bit-field XOR permutation functor `Swizzle(bits, base, shift)`.
///
/// Transforms bit pattern `0b...YYY...ZZZ...` into `0b...YYY...(ZZZ ^ YYY)...` via
/// `offset ^ shiftr(offset & yyy_msk, shift)`.
///
/// When all three parameters are static [`Int<N>`], `Swizzle` is zero-sized and
/// evaluates static offsets at compile time.
///
/// Invalid static parameters are rejected at compile time:
///
/// ```compile_fail,E0271
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::swizzle::Swizzle;
/// let _ = Swizzle::new(Int::<-1>, Int::<0>, Int::<1>);
/// ```
///
/// ```compile_fail,E0271
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::swizzle::Swizzle;
/// let _ = Swizzle::new(Int::<3>, Int::<0>, Int::<-2>);
/// ```
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Swizzle<Bits = int, Base = int, Shift = int> {
    bits: Bits,
    base: Base,
    shift: Shift,
}

/// Type alias for a compile-time static [`Swizzle`].
pub type StaticSwizzle<const BITS: int, const BASE: int, const SHIFT: int> =
    Swizzle<Int<BITS>, Int<BASE>, Int<SHIFT>>;

const impl<Bits: Item, Base: Item, Shift: Item> Item for Swizzle<Bits, Base, Shift> {
    const RANK: int = 1;
    const DEPTH: int = 0;
    const LEAVES: int = 1;
    const IS_STATIC: bool = Bits::IS_STATIC && Base::IS_STATIC && Shift::IS_STATIC;
    type Profile = LeafProfile;
}

const impl<const B: int, const M: int, const S: int> StaticValue
    for Swizzle<Int<B>, Int<M>, Int<S>>
where
    (): ValidSwizzle<{ SWIZZLE_VALID::<Int<B>, Int<M>, Int<S>> }>,
{
    const VALUE: Self = Self {
        bits: Int,
        base: Int,
        shift: Int,
    };
}

const impl<B1, M1, S1, B2, M2, S2> PartialEq<Swizzle<B2, M2, S2>> for Swizzle<B1, M1, S1>
where
    B1: [const] Integer,
    M1: [const] Integer,
    S1: [const] Integer,
    B2: [const] Integer,
    M2: [const] Integer,
    S2: [const] Integer,
{
    fn eq(&self, other: &Swizzle<B2, M2, S2>) -> bool {
        eq(self.bits, other.bits) && eq(self.base, other.base) && eq(self.shift, other.shift)
    }
}

const impl<B: [const] Integer, M: [const] Integer, S: [const] Integer> Eq for Swizzle<B, M, S> {}

impl<Bits: Display, Base: Display, Shift: Display> Display for Swizzle<Bits, Base, Shift> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "SW_{}_{}_{}", self.bits, self.base, self.shift)
    }
}

impl<Bits, Base, Shift> Swizzle<Bits, Base, Shift> {
    /// Constructs a validated `Swizzle(bits, base, shift)`.
    ///
    /// Static precondition violations fail at compile time; dynamic violations panic at runtime.
    pub const fn new(bits: Bits, base: Base, shift: Shift) -> Self
    where
        Bits: [const] Integer,
        Base: [const] Integer,
        Shift: [const] Integer,
        (): [const] ValidSwizzle<{ SWIZZLE_VALID::<Bits, Base, Shift> }>,
    {
        validate_swizzle_values(bits.value(), base.value(), shift.value());
        Self { bits, base, shift }
    }

    /// Number of bits in the `YYY` / `ZZZ` masks.
    pub const fn bits(&self) -> Bits
    where
        Bits: Copy,
    {
        self.bits
    }

    /// Number of least-significant bits kept constant.
    pub const fn base(&self) -> Base
    where
        Base: Copy,
    {
        self.base
    }

    /// Signed distance to shift the `YYY` mask (`>= 0` shifts right, `< 0` shifts left).
    pub const fn shift(&self) -> Shift
    where
        Shift: Copy,
    {
        self.shift
    }

    /// Base `bits`-wide mask `(1 << bits) - 1`.
    pub const fn bit_msk(&self) -> SwizzleBitMaskOutput<Bits>
    where
        Bits: [const] Integer + [const] SwizzleBitMask<{ BIT_MASK_IS_STATIC::<Bits> }>,
    {
        self.bits.bit_mask()
    }

    /// Source bit-field mask `bit_msk << (base + max(0, shift))`.
    pub const fn yyy_msk(&self) -> SwizzleYyyMaskOutput<Bits, Base, Shift>
    where
        Bits: [const] Integer
            + [const] SwizzleFieldMasks<Base, Shift, { FIELD_MASKS_ARE_STATIC::<Bits, Base, Shift> }>,
        Base: [const] Integer,
        Shift: [const] Integer,
    {
        self.bits.yyy_mask(self.base, self.shift)
    }

    /// Target bit-field mask `bit_msk << (base - min(0, shift))`.
    pub const fn zzz_msk(&self) -> SwizzleZzzMaskOutput<Bits, Base, Shift>
    where
        Bits: [const] Integer
            + [const] SwizzleFieldMasks<Base, Shift, { FIELD_MASKS_ARE_STATIC::<Bits, Base, Shift> }>,
        Base: [const] Integer,
        Shift: [const] Integer,
    {
        self.bits.zzz_mask(self.base, self.shift)
    }

    /// Returns `true` iff `YYY` and `ZZZ` do not overlap (`bits == 0 || |shift| >= bits`),
    /// guaranteeing involution `S(S(x)) == x`.
    pub const fn is_involution(&self) -> bool
    where
        Bits: [const] Integer,
        Shift: [const] Integer,
    {
        let b = self.bits.value();
        let s = self.shift.value();
        s >= b || s <= neg(b)
    }

    /// Applies the swizzle transformation to `offset`.
    pub const fn call<O>(&self, offset: O) -> <Self as SwizzleApply<O>>::Output
    where
        Self: [const] SwizzleApply<O>,
    {
        SwizzleApply::apply_swizzle(self, offset)
    }

    /// Merges `self` with another involutive swizzle sharing the same shift.
    pub const fn compose<R>(self, rhs: R) -> <Self as ComposeSwizzle<R>>::Output
    where
        Self: [const] ComposeSwizzle<R>,
    {
        ComposeSwizzle::compose_swizzle(self, rhs)
    }

    /// Upcasts this involutive swizzle by power-of-two factor `factor`.
    pub const fn upcast<N>(self, factor: N) -> <Self as UpcastSwizzle<N>>::Output
    where
        Self: [const] UpcastSwizzle<N>,
    {
        UpcastSwizzle::upcast_swizzle(self, factor)
    }

    /// Downcasts this involutive swizzle by power-of-two factor `factor`.
    pub const fn downcast<N>(self, factor: N) -> <Self as DowncastSwizzle<N>>::Output
    where
        Self: [const] DowncastSwizzle<N>,
    {
        DowncastSwizzle::downcast_swizzle(self, factor)
    }
}

// ------------------------------------------------------------------
// Swizzle helper algebra (involution-preserving masks, merge, upcast/downcast)
// ------------------------------------------------------------------

const fn assert_involution(bits: int, shift: int) {
    NoodleCuteError::InvalidShape.check_with(
        bits == 0 || shift >= bits || shift <= neg(bits),
        "Swizzle algebra requires an involution (bits == 0 or |shift| >= bits)",
    );
}

/// Derives `(bits, base, shift)` for an involutive [`Swizzle`] from its source (`y`)
/// and destination (`z`) bit-field masks (CuTe `make_swizzle<Y, Z>` parameters).
///
/// Both masks must be non-negative, have equal popcounts, be disjoint (`|shift| >= bits`),
/// and form contiguous `bits`-wide fields matching `yyy_msk` and `zzz_msk`.
pub const fn swizzle_params(y: int, z: int) -> (int, int, int) {
    NoodleCuteError::InvalidShape.check_with(
        y >= 0 && z >= 0,
        "Swizzle masks must be non-negative",
    );
    let uy = y as uint;
    let uz = z as uint;
    let b = uy.count_ones() as int;
    NoodleCuteError::InvalidShape.check_with(
        b == uz.count_ones() as int,
        "Number of bits in Y and Z masks must match",
    );
    if b == 0 {
        return (0, 0, 0);
    }
    let ty = uy.trailing_zeros() as int;
    let tz = uz.trailing_zeros() as int;
    let m = min(ty, tz);
    let s = sub(ty, tz);
    assert_involution(b, s);
    NoodleCuteError::InvalidShape.check_with(
        fits_swizzle_width(b, m, s) && y == yyy_msk_val(b, m, s) && z == zzz_msk_val(b, m, s),
        "masks are not the disjoint fields of a valid Swizzle",
    );
    (b, m, s)
}

/// `bits` parameter derived by [`make_swizzle`].
#[doc(hidden)]
pub const MADE_SWIZZLE_B<const Y: int, const Z: int>: int = swizzle_params(Y, Z).0;
/// `base` parameter derived by [`make_swizzle`].
#[doc(hidden)]
pub const MADE_SWIZZLE_M<const Y: int, const Z: int>: int = swizzle_params(Y, Z).1;
/// `shift` parameter derived by [`make_swizzle`].
#[doc(hidden)]
pub const MADE_SWIZZLE_S<const Y: int, const Z: int>: int = swizzle_params(Y, Z).2;

/// Static [`Swizzle`] type constructed from source mask `Y` and destination mask `Z`.
pub type MakeSwizzle<const Y: int, const Z: int> = StaticSwizzle<
    { MADE_SWIZZLE_B::<Y, Z> },
    { MADE_SWIZZLE_M::<Y, Z> },
    { MADE_SWIZZLE_S::<Y, Z> },
>;

/// Constructs the static involutive [`Swizzle`] with source mask `Y` and destination mask `Z`
/// (CuTe `make_swizzle<Y, Z>()`).
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::swizzle::{Swizzle, make_swizzle};
///
/// assert_eq!(
///     make_swizzle::<0b1000, 0b0100>(),
///     Swizzle::new(Int::<1>, Int::<2>, Int::<1>)
/// );
/// assert_eq!(
///     make_swizzle::<0b1100_0000, 0b0000_0110>(),
///     Swizzle::new(Int::<2>, Int::<1>, Int::<5>)
/// );
/// ```
pub const fn make_swizzle<const Y: int, const Z: int>() -> MakeSwizzle<Y, Z> {
    Swizzle {
        bits: Int,
        base: Int,
        shift: Int,
    }
}

const fn merge_swizzle_params(
    b0: int,
    m0: int,
    s0: int,
    b1: int,
    m1: int,
    s1: int,
) -> (int, int, int) {
    validate_swizzle_values(b0, m0, s0);
    validate_swizzle_values(b1, m1, s1);
    assert_involution(b0, s0);
    assert_involution(b1, s1);
    if b0 == 0 {
        return (b1, m1, s1);
    }
    if b1 == 0 {
        return (b0, m0, s0);
    }
    NoodleCuteError::Unsupported.check_with(
        s0 == s1,
        "can only merge involutive swizzles with the same shift",
    );
    let y = bitxor(yyy_msk_val(b0, m0, s0), yyy_msk_val(b1, m1, s1));
    let z = bitxor(zzz_msk_val(b0, m0, s0), zzz_msk_val(b1, m1, s1));
    swizzle_params(y, z)
}

/// Computes `(bits, base, shift)` after upcasting an involutive swizzle by power-of-two `n`.
pub const fn upcast_swizzle_params(bits: int, base: int, shift: int, n: int) -> (int, int, int) {
    validate_swizzle_values(bits, base, shift);
    assert_involution(bits, shift);
    NoodleCuteError::InvalidShape.check_with(
        is_pow2(n),
        "upcast_swizzle requires a positive power-of-two factor",
    );
    let log2 = (n as uint).trailing_zeros() as int;
    if base >= log2 {
        (bits, sub(base, log2), shift)
    } else {
        let drop = sub(log2, base);
        let new_b = max(0, sub(bits, drop));
        (new_b, 0, shift)
    }
}

/// Computes `(bits, base, shift)` after downcasting an involutive swizzle by power-of-two `n`.
pub const fn downcast_swizzle_params(bits: int, base: int, shift: int, n: int) -> (int, int, int) {
    validate_swizzle_values(bits, base, shift);
    assert_involution(bits, shift);
    NoodleCuteError::InvalidShape.check_with(
        is_pow2(n),
        "downcast_swizzle requires a positive power-of-two factor",
    );
    let log2 = (n as uint).trailing_zeros() as int;
    let new_m = add(base, log2);
    validate_swizzle_values(bits, new_m, shift);
    (bits, new_m, shift)
}

const SWIZZLE_PAIR_STATIC<
    B0: Integer,
    M0: Integer,
    S0: Integer,
    B1: Integer,
    M1: Integer,
    S1: Integer,
>: bool = B0::IS_STATIC
    && M0::IS_STATIC
    && S0::IS_STATIC
    && B1::IS_STATIC
    && M1::IS_STATIC
    && S1::IS_STATIC;

const SWIZZLE_FACTOR_STATIC<B: Integer, M: Integer, S: Integer, N: Integer>: bool =
    B::IS_STATIC && M::IS_STATIC && S::IS_STATIC && N::IS_STATIC;

const fn static_merge_params<
    B0: Integer,
    M0: Integer,
    S0: Integer,
    B1: Integer,
    M1: Integer,
    S1: Integer,
>() -> (int, int, int) {
    merge_swizzle_params(
        B0::STATIC_VALUE.expect("expected static bits"),
        M0::STATIC_VALUE.expect("expected static base"),
        S0::STATIC_VALUE.expect("expected static shift"),
        B1::STATIC_VALUE.expect("expected static bits"),
        M1::STATIC_VALUE.expect("expected static base"),
        S1::STATIC_VALUE.expect("expected static shift"),
    )
}

#[doc(hidden)]
pub const MERGED_SWIZZLE_B<
    B0: Integer,
    M0: Integer,
    S0: Integer,
    B1: Integer,
    M1: Integer,
    S1: Integer,
>: int = static_merge_params::<B0, M0, S0, B1, M1, S1>().0;
#[doc(hidden)]
pub const MERGED_SWIZZLE_M<
    B0: Integer,
    M0: Integer,
    S0: Integer,
    B1: Integer,
    M1: Integer,
    S1: Integer,
>: int = static_merge_params::<B0, M0, S0, B1, M1, S1>().1;
#[doc(hidden)]
pub const MERGED_SWIZZLE_S<
    B0: Integer,
    M0: Integer,
    S0: Integer,
    B1: Integer,
    M1: Integer,
    S1: Integer,
>: int = static_merge_params::<B0, M0, S0, B1, M1, S1>().2;

/// Static/dynamic dispatch for [`ComposeSwizzle`].
#[doc(hidden)]
pub const trait ComposeSwizzleImpl<R, const STATIC: bool>: Copy {
    type Output: Copy;
    fn compose_swizzle_impl(self, rhs: R) -> Self::Output;
}

const impl<B0, M0, S0, B1, M1, S1> ComposeSwizzleImpl<Swizzle<B1, M1, S1>, true>
    for Swizzle<B0, M0, S0>
where
    B0: [const] Integer,
    M0: [const] Integer,
    S0: [const] Integer,
    B1: [const] Integer,
    M1: [const] Integer,
    S1: [const] Integer,
{
    type Output = StaticSwizzle<
        { MERGED_SWIZZLE_B::<B0, M0, S0, B1, M1, S1> },
        { MERGED_SWIZZLE_M::<B0, M0, S0, B1, M1, S1> },
        { MERGED_SWIZZLE_S::<B0, M0, S0, B1, M1, S1> },
    >;

    fn compose_swizzle_impl(self, _: Swizzle<B1, M1, S1>) -> Self::Output {
        Swizzle {
            bits: Int,
            base: Int,
            shift: Int,
        }
    }
}

const impl<B0, M0, S0, B1, M1, S1> ComposeSwizzleImpl<Swizzle<B1, M1, S1>, false>
    for Swizzle<B0, M0, S0>
where
    B0: [const] Integer,
    M0: [const] Integer,
    S0: [const] Integer,
    B1: [const] Integer,
    M1: [const] Integer,
    S1: [const] Integer,
{
    type Output = Swizzle<int, int, int>;

    fn compose_swizzle_impl(self, rhs: Swizzle<B1, M1, S1>) -> Self::Output {
        let (b, m, s) = merge_swizzle_params(
            self.bits.value(),
            self.base.value(),
            self.shift.value(),
            rhs.bits.value(),
            rhs.base.value(),
            rhs.shift.value(),
        );
        Swizzle::new(b, m, s)
    }
}

/// Composes two involutive [`Swizzle`] functors sharing the same shift.
///
/// Static violations (non-involution, differing shifts, or non-contiguous merged masks) fail
/// at compile time; dynamic violations panic at runtime.
pub const trait ComposeSwizzle<R>: Copy {
    /// Merged [`Swizzle`] type.
    type Output: Copy;
    /// Merges `self` and `rhs`.
    fn compose_swizzle(self, rhs: R) -> Self::Output;
}

const impl<B0, M0, S0, B1, M1, S1> ComposeSwizzle<Swizzle<B1, M1, S1>> for Swizzle<B0, M0, S0>
where
    B0: [const] Integer,
    M0: [const] Integer,
    S0: [const] Integer,
    B1: [const] Integer,
    M1: [const] Integer,
    S1: [const] Integer,
    Self: [const] ComposeSwizzleImpl<
            Swizzle<B1, M1, S1>,
            { SWIZZLE_PAIR_STATIC::<B0, M0, S0, B1, M1, S1> },
        >,
{
    type Output = <Self as ComposeSwizzleImpl<
        Swizzle<B1, M1, S1>,
        { SWIZZLE_PAIR_STATIC::<B0, M0, S0, B1, M1, S1> },
    >>::Output;

    fn compose_swizzle(self, rhs: Swizzle<B1, M1, S1>) -> Self::Output {
        self.compose_swizzle_impl(rhs)
    }
}

/// Composes two involutive [`Swizzle`] functors sharing the same shift.
pub const fn compose_swizzle<L: [const] ComposeSwizzle<R>, R>(lhs: L, rhs: R) -> L::Output {
    lhs.compose_swizzle(rhs)
}

const fn static_upcast_params<B: Integer, M: Integer, S: Integer, N: Integer>() -> (int, int, int) {
    upcast_swizzle_params(
        B::STATIC_VALUE.expect("expected static bits"),
        M::STATIC_VALUE.expect("expected static base"),
        S::STATIC_VALUE.expect("expected static shift"),
        N::STATIC_VALUE.expect("expected static factor"),
    )
}

const fn static_downcast_params<B: Integer, M: Integer, S: Integer, N: Integer>() -> (int, int, int)
{
    downcast_swizzle_params(
        B::STATIC_VALUE.expect("expected static bits"),
        M::STATIC_VALUE.expect("expected static base"),
        S::STATIC_VALUE.expect("expected static shift"),
        N::STATIC_VALUE.expect("expected static factor"),
    )
}

#[doc(hidden)]
pub const UPCAST_SWIZZLE_B<B: Integer, M: Integer, S: Integer, N: Integer>: int =
    static_upcast_params::<B, M, S, N>().0;
#[doc(hidden)]
pub const UPCAST_SWIZZLE_M<B: Integer, M: Integer, S: Integer, N: Integer>: int =
    static_upcast_params::<B, M, S, N>().1;
#[doc(hidden)]
pub const UPCAST_SWIZZLE_S<B: Integer, M: Integer, S: Integer, N: Integer>: int =
    static_upcast_params::<B, M, S, N>().2;

#[doc(hidden)]
pub const DOWNCAST_SWIZZLE_B<B: Integer, M: Integer, S: Integer, N: Integer>: int =
    static_downcast_params::<B, M, S, N>().0;
#[doc(hidden)]
pub const DOWNCAST_SWIZZLE_M<B: Integer, M: Integer, S: Integer, N: Integer>: int =
    static_downcast_params::<B, M, S, N>().1;
#[doc(hidden)]
pub const DOWNCAST_SWIZZLE_S<B: Integer, M: Integer, S: Integer, N: Integer>: int =
    static_downcast_params::<B, M, S, N>().2;

/// Static/dynamic dispatch for [`UpcastSwizzle`].
#[doc(hidden)]
pub const trait UpcastSwizzleImpl<N, const STATIC: bool>: Copy {
    type Output: Copy;
    fn upcast_swizzle_impl(self, n: N) -> Self::Output;
}

const impl<B, M, S, N> UpcastSwizzleImpl<N, true> for Swizzle<B, M, S>
where
    B: [const] Integer,
    M: [const] Integer,
    S: [const] Integer,
    N: [const] Integer,
{
    type Output = StaticSwizzle<
        { UPCAST_SWIZZLE_B::<B, M, S, N> },
        { UPCAST_SWIZZLE_M::<B, M, S, N> },
        { UPCAST_SWIZZLE_S::<B, M, S, N> },
    >;

    fn upcast_swizzle_impl(self, _: N) -> Self::Output {
        Swizzle {
            bits: Int,
            base: Int,
            shift: Int,
        }
    }
}

const impl<B, M, S, N> UpcastSwizzleImpl<N, false> for Swizzle<B, M, S>
where
    B: [const] Integer,
    M: [const] Integer,
    S: [const] Integer,
    N: [const] Integer,
{
    type Output = Swizzle<int, int, int>;

    fn upcast_swizzle_impl(self, n: N) -> Self::Output {
        let (b, m, s) =
            upcast_swizzle_params(self.bits.value(), self.base.value(), self.shift.value(), n.value());
        Swizzle::new(b, m, s)
    }
}

/// Upcasts an involutive [`Swizzle`] by a power-of-two factor `N` (CuTe `upcast<N>(swizzle)`).
pub const trait UpcastSwizzle<N>: Copy {
    /// Upcast [`Swizzle`] type.
    type Output: Copy;
    /// Upcasts `self` by `factor`.
    fn upcast_swizzle(self, factor: N) -> Self::Output;
}

const impl<B, M, S, N> UpcastSwizzle<N> for Swizzle<B, M, S>
where
    B: [const] Integer,
    M: [const] Integer,
    S: [const] Integer,
    N: [const] Integer,
    Self: [const] UpcastSwizzleImpl<N, { SWIZZLE_FACTOR_STATIC::<B, M, S, N> }>,
{
    type Output =
        <Self as UpcastSwizzleImpl<N, { SWIZZLE_FACTOR_STATIC::<B, M, S, N> }>>::Output;

    fn upcast_swizzle(self, factor: N) -> Self::Output {
        self.upcast_swizzle_impl(factor)
    }
}

/// Upcasts involutive `sw` by power-of-two factor `factor`.
pub const fn upcast_swizzle<Sw: [const] UpcastSwizzle<N>, N>(sw: Sw, factor: N) -> Sw::Output {
    sw.upcast_swizzle(factor)
}

/// Static/dynamic dispatch for [`DowncastSwizzle`].
#[doc(hidden)]
pub const trait DowncastSwizzleImpl<N, const STATIC: bool>: Copy {
    type Output: Copy;
    fn downcast_swizzle_impl(self, n: N) -> Self::Output;
}

const impl<B, M, S, N> DowncastSwizzleImpl<N, true> for Swizzle<B, M, S>
where
    B: [const] Integer,
    M: [const] Integer,
    S: [const] Integer,
    N: [const] Integer,
{
    type Output = StaticSwizzle<
        { DOWNCAST_SWIZZLE_B::<B, M, S, N> },
        { DOWNCAST_SWIZZLE_M::<B, M, S, N> },
        { DOWNCAST_SWIZZLE_S::<B, M, S, N> },
    >;

    fn downcast_swizzle_impl(self, _: N) -> Self::Output {
        Swizzle {
            bits: Int,
            base: Int,
            shift: Int,
        }
    }
}

const impl<B, M, S, N> DowncastSwizzleImpl<N, false> for Swizzle<B, M, S>
where
    B: [const] Integer,
    M: [const] Integer,
    S: [const] Integer,
    N: [const] Integer,
{
    type Output = Swizzle<int, int, int>;

    fn downcast_swizzle_impl(self, n: N) -> Self::Output {
        let (b, m, s) = downcast_swizzle_params(
            self.bits.value(),
            self.base.value(),
            self.shift.value(),
            n.value(),
        );
        Swizzle::new(b, m, s)
    }
}

/// Downcasts an involutive [`Swizzle`] by a power-of-two factor `N` (CuTe `downcast<N>(swizzle)`).
pub const trait DowncastSwizzle<N>: Copy {
    /// Downcast [`Swizzle`] type.
    type Output: Copy;
    /// Downcasts `self` by `factor`.
    fn downcast_swizzle(self, factor: N) -> Self::Output;
}

const impl<B, M, S, N> DowncastSwizzle<N> for Swizzle<B, M, S>
where
    B: [const] Integer,
    M: [const] Integer,
    S: [const] Integer,
    N: [const] Integer,
    Self: [const] DowncastSwizzleImpl<N, { SWIZZLE_FACTOR_STATIC::<B, M, S, N> }>,
{
    type Output =
        <Self as DowncastSwizzleImpl<N, { SWIZZLE_FACTOR_STATIC::<B, M, S, N> }>>::Output;

    fn downcast_swizzle(self, factor: N) -> Self::Output {
        self.downcast_swizzle_impl(factor)
    }
}

/// Downcasts involutive `sw` by power-of-two factor `factor`.
pub const fn downcast_swizzle<Sw: [const] DowncastSwizzle<N>, N>(sw: Sw, factor: N) -> Sw::Output {
    sw.downcast_swizzle(factor)
}

/// Recasts an involutive [`Swizzle`] from `Old` to `New` element units (upcasting by the reduced
/// numerator `size_of::<New>() / g` and downcasting by `size_of::<Old>() / g`, both powers of two).
pub const fn recast_swizzle<Old, New, Sw>(
    sw: Sw,
) -> <<Sw as UpcastSwizzle<Int<{ RECAST_NUM::<Old, New> }>>>::Output as DowncastSwizzle<
    Int<{ RECAST_DEN::<Old, New> }>,
>>::Output
where
    Sw: [const] UpcastSwizzle<Int<{ RECAST_NUM::<Old, New> }>>,
    Sw::Output: [const] DowncastSwizzle<Int<{ RECAST_DEN::<Old, New> }>>,
{
    downcast_swizzle(
        upcast_swizzle(sw, Int::<{ RECAST_NUM::<Old, New> }>),
        Int::<{ RECAST_DEN::<Old, New> }>,
    )
}

/// Applies a [`Swizzle`] functor to an integer (or [`F2`]) offset.
pub const trait SwizzleApply<O>: Copy {
    /// Resulting swizzled offset type.
    type Output: Copy;

    /// Evaluates `offset ^ shiftr(offset & yyy_msk, shift)`.
    fn apply_swizzle(&self, offset: O) -> Self::Output;
}

const impl<Bits, Base, Shift, O> SwizzleApply<O> for Swizzle<Bits, Base, Shift>
where
    Bits: [const] Integer
        + [const] SwizzleFieldMasks<Base, Shift, { FIELD_MASKS_ARE_STATIC::<Bits, Base, Shift> }>,
    Base: [const] Integer,
    Shift: [const] Integer,
    O: [const] Integer
        + [const] ArithBitAnd<O, SwizzleYyyMaskOutput<Bits, Base, Shift>>
        + [const] ArithBitXor<
            O,
            ShiftRResult<BitAndResult<O, SwizzleYyyMaskOutput<Bits, Base, Shift>>, Shift>,
        >,
    BitAndResult<O, SwizzleYyyMaskOutput<Bits, Base, Shift>>: [const] Integer
        + [const] ArithShiftR<BitAndResult<O, SwizzleYyyMaskOutput<Bits, Base, Shift>>, Shift>,
    ShiftRResult<BitAndResult<O, SwizzleYyyMaskOutput<Bits, Base, Shift>>, Shift>: [const] Integer,
{
    type Output = BitXorResult<
        O,
        ShiftRResult<BitAndResult<O, SwizzleYyyMaskOutput<Bits, Base, Shift>>, Shift>,
    >;

    fn apply_swizzle(&self, offset: O) -> Self::Output {
        let masked = bitand(offset, self.yyy_msk());
        let shifted = shiftr(masked, self.shift);
        bitxor(offset, shifted)
    }
}

const impl<Bits, Base, Shift, V> SwizzleApply<F2<V>> for Swizzle<Bits, Base, Shift>
where
    V: [const] F2Scalar,
    Self: [const] SwizzleApply<V>,
    <Self as SwizzleApply<V>>::Output: [const] IntoF2,
{
    type Output = <<Self as SwizzleApply<V>>::Output as IntoF2>::Output;

    fn apply_swizzle(&self, offset: F2<V>) -> Self::Output {
        f2(SwizzleApply::<V>::apply_swizzle(self, offset.inner()))
    }
}

const impl<Bits, Base, Shift, O> FnOnce<(O,)> for Swizzle<Bits, Base, Shift>
where
    O: Copy,
    Self: [const] SwizzleApply<O>,
{
    type Output = <Self as SwizzleApply<O>>::Output;

    extern "rust-call" fn call_once(self, args: (O,)) -> Self::Output {
        let (offset,) = args;
        self.apply_swizzle(offset)
    }
}

const impl<Bits, Base, Shift, O> FnMut<(O,)> for Swizzle<Bits, Base, Shift>
where
    O: Copy,
    Self: [const] SwizzleApply<O>,
{
    extern "rust-call" fn call_mut(&mut self, args: (O,)) -> Self::Output {
        let (offset,) = args;
        self.apply_swizzle(offset)
    }
}

const impl<Bits, Base, Shift, O> Fn<(O,)> for Swizzle<Bits, Base, Shift>
where
    O: Copy,
    Self: [const] SwizzleApply<O>,
{
    extern "rust-call" fn call(&self, args: (O,)) -> Self::Output {
        let (offset,) = args;
        self.apply_swizzle(offset)
    }
}

/// Original module path for the swizzle functor.
pub mod functor {
    pub use super::{
        SWIZZLE_VALID, StaticSwizzle, Swizzle, SwizzleApply, SwizzleBitMask, SwizzleBitMaskOutput,
        SwizzleFieldMasks, SwizzleYyyMaskOutput, SwizzleZzzMaskOutput, ValidSwizzle,
    };
}

/// Constructs a validated [`Swizzle`] from `(bits, base, shift)` expressions.
#[macro_export]
macro_rules! swizzle {
    ($bits:expr, $base:expr, $shift:expr $(,)?) => {
        $crate::swizzle::Swizzle::new($bits, $base, $shift)
    };
}
