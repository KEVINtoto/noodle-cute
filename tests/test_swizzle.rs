#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use std::panic::catch_unwind;

use noodle_cute::integer::{Int, int, shiftr};
use noodle_cute::swizzle;
use noodle_cute::swizzle::{StaticSwizzle, Swizzle};
use noodle_cute::tuple::Item;

#[test]
fn test_swizzle_static_and_dynamic_masks_and_repr() {
    assert_eq!(core::mem::size_of::<StaticSwizzle<3, 0, 3>>(), 0);
    const {
        assert!(<StaticSwizzle<3, 0, 3> as Item>::IS_STATIC);
        assert!(!<Swizzle<int, Int<0>, Int<3>> as Item>::IS_STATIC);
    }

    const SW_POS: StaticSwizzle<3, 4, 3> = Swizzle::new(Int::<3>, Int::<4>, Int::<3>);
    const BIT_MSK: Int<0b111> = SW_POS.bit_msk();
    const YYY_MSK: Int<0b11_1000_0000> = SW_POS.yyy_msk();
    const ZZZ_MSK: Int<0b111_0000> = SW_POS.zzz_msk();
    assert_eq!(BIT_MSK, Int::<0b111>);
    assert_eq!(YYY_MSK, Int::<{ 0b111 << 7 }>);
    assert_eq!(ZZZ_MSK, Int::<{ 0b111 << 4 }>);
    assert!(SW_POS.is_involution());
    assert_eq!(format!("{SW_POS}"), "SW_3_4_3");

    const SW_NEG: StaticSwizzle<2, 3, -2> = Swizzle::new(Int::<2>, Int::<3>, Int::<-2>);
    assert_eq!(SW_NEG.bit_msk(), Int::<0b11>);
    assert_eq!(SW_NEG.yyy_msk(), Int::<{ 0b11 << 3 }>);
    assert_eq!(SW_NEG.zzz_msk(), Int::<{ 0b11 << 5 }>);
    assert!(SW_NEG.is_involution());
    assert_eq!(format!("{SW_NEG}"), "SW_2_3_-2");

    // Zero bits: identity swizzle, static masks collapse to Int<0> even with dynamic base/shift
    let sw_zero = swizzle!(Int::<0>, 5i32, -100i32);
    let yz: (Int<0>, Int<0>) = (sw_zero.yyy_msk(), sw_zero.zzz_msk());
    assert_eq!(yz, (Int::<0>, Int::<0>));
    assert!(sw_zero.is_involution());
    assert_eq!(sw_zero(0x1234_5678), 0x1234_5678);

    // Heterogeneous equality between static and dynamic Swizzles
    let sw_dyn = Swizzle::new(3i32, 4i32, 3i32);
    assert_eq!(SW_POS, sw_dyn);
    assert_eq!(sw_dyn.bit_msk(), 0b111);
    assert_eq!(sw_dyn.yyy_msk(), 0b111 << 7);
    assert_eq!(sw_dyn.zzz_msk(), 0b111 << 4);
}

#[test]
fn test_swizzle_evaluation_and_involution() {
    // Matches pycute/swizzle.py docstring examples
    const SW1: StaticSwizzle<1, 0, 1> = Swizzle::new(Int::<1>, Int::<0>, Int::<1>);
    const R0: Int<0b00> = SW1(Int::<0b00>);
    const R2: Int<0b11> = SW1(Int::<0b10>);
    const R2_INV: Int<0b10> = SW1(R2);
    assert_eq!((R0, R2, R2_INV), (Int::<0b00>, Int::<0b11>, Int::<0b10>));

    // Involution across positive and negative non-overlapping shifts
    for &(bits, base, shift) in &[
        (0, 0, 0),
        (1, 0, 1),
        (3, 0, 3),
        (3, 4, 3),
        (2, 3, -2),
        (3, 1, -4),
    ] {
        let sw = Swizzle::new(bits, base, shift);
        assert!(sw.is_involution());
        for x in 0..1024 {
            assert_eq!(sw(sw(x)), x);
            assert_eq!(sw.call(x), x ^ shiftr(x & sw.yyy_msk(), shift));
        }
    }

    // Overlapping positive shift (0 <= shift < bits): valid, non-involutive
    let sw_overlap = Swizzle::new(2i32, 0i32, 1i32);
    assert!(!sw_overlap.is_involution());
    // 0b110: yyy_msk = 0b110 -> shiftr(0b110, 1) = 0b011 -> 0b110 ^ 0b011 = 0b101
    assert_eq!(sw_overlap(0b110), 0b101);
    assert_eq!(sw_overlap(0b101), 0b111); // not involutive

    // shift == 0 clears the bit field
    let sw_zero_shift = Swizzle::new(Int::<2>, Int::<1>, Int::<0>);
    assert!(!sw_zero_shift.is_involution());
    assert_eq!(sw_zero_shift(0b1111), 0b1001);
}

#[test]
fn test_swizzle_dynamic_validation_panics() {
    assert!(catch_unwind(|| Swizzle::new(-1i32, 0i32, 1i32)).is_err());
    assert!(catch_unwind(|| Swizzle::new(1i32, -1i32, 1i32)).is_err());
    assert!(catch_unwind(|| Swizzle::new(3i32, 0i32, -2i32)).is_err());
    assert!(catch_unwind(|| Swizzle::new(1i32, 0i32, 31i32)).is_err());
    assert!(catch_unwind(|| Swizzle::new(1i32, 0i32, -31i32)).is_err());
    assert!(catch_unwind(|| Swizzle::new(1i32, 0i32, i32::MIN)).is_err());
    assert!(catch_unwind(|| Swizzle::new(31i32, 1i32, 0i32)).is_err());

    // Exact 31-bit boundary succeeds
    let sw_edge = Swizzle::new(1i32, 0i32, -30i32);
    assert_eq!(sw_edge.yyy_msk(), 1);
    assert_eq!(sw_edge.zzz_msk(), 1 << 30);
    assert_eq!(sw_edge(1), 1 | (1 << 30));
    assert_eq!(sw_edge(sw_edge(1)), 1);
}
