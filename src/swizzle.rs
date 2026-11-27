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
    add, bit_length_non_neg, bitand, bitxor, eq, int, max, neg, shiftl, shiftr, sub,
};
use crate::tuple::{Item, LeafProfile};

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
    assert!(bits >= 0, "Swizzle requires bits >= 0");
    assert!(base >= 0, "Swizzle requires base >= 0");
    assert!(
        shift >= 0 || shift <= neg(bits),
        "Swizzle requires shift >= 0 or |shift| >= bits"
    );
    assert!(
        fits_swizzle_width(bits, base, shift),
        "Swizzle bit masks exceed representable non-negative int width"
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
