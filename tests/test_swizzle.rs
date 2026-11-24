#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl,
    const_ops
)]
#![allow(incomplete_features)]

use std::panic::catch_unwind;

use noodle_cute::algebra::{
    coalesce, coalesce_extended, composition, layout_add, left_inverse, logical_divide, nullspace,
    right_inverse, zipped_divide,
};
use noodle_cute::basis::{BasisTerm, Mode, as_tuple, basis_eq, colex_cmp, unit};
use noodle_cute::integer::{Int, int};
use noodle_cute::layout::{Fraction, Layout, make_layout_like, recast, tiler_to_layout_with};
use noodle_cute::shape::{crd2idx, idx2crd, in_bounds};
use noodle_cute::stride::{coshape, inner_product, prefix_product_with};
use noodle_cute::swizzle::{F2, StaticF2, StaticSwizzle, Swizzle, f2, f2_divmod, shiftl, shiftr};
use noodle_cute::tuple::Item;
use noodle_cute::{e, f2 as f2_macro, int_tuple, swizzle, tuple};

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

    // Re-exported shift helpers match pycute examples
    assert_eq!(shiftr(0b1000, 2), 0b10);
    assert_eq!(shiftr(0b1000, -2), 0b100000);
    assert_eq!(shiftl(0b1000, 2), 0b100000);
    assert_eq!(shiftl(0b1000, -2), 0b10);
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

#[test]
fn test_f2_arithmetic_comparison_and_swizzle_apply() {
    assert_eq!(core::mem::size_of::<StaticF2<9>>(), 0);
    assert_eq!(core::mem::size_of::<F2<int>>(), core::mem::size_of::<int>());
    const {
        assert!(<StaticF2<9> as Item>::IS_STATIC);
        assert!(!<F2<int> as Item>::IS_STATIC);
    }

    // Idempotent constructor and Display
    const A: StaticF2<0b1010> = f2(Int::<0b1010>);
    const B: StaticF2<0b1100> = f2(f2(Int::<0b1100>));
    assert_eq!(f2_macro!(A), A);
    assert_eq!(format!("{A}"), "F10");
    assert_eq!(format!("{}", f2(9i32)), "F9");

    // XOR addition, subtraction, negation, additive identity Int<0>
    const SUM: StaticF2<0b0110> = A + B;
    const DIFF: StaticF2<0b0110> = A - B;
    const SELF_CANCEL: StaticF2<0> = A + A;
    const NEG_A: StaticF2<0b1010> = -A;
    const ID_R: StaticF2<0b1010> = A + Int::<0>;
    const ID_L: StaticF2<0b1010> = Int::<0> + A;
    assert_eq!(SUM, Int::<0b0110>);
    assert_eq!(DIFF, f2(Int::<0b0110>));
    assert_eq!(SELF_CANCEL, Int::<0>);
    assert_eq!(NEG_A, A);
    assert_eq!((ID_R, ID_L), (A, A));

    // Carry-less multiplication (static, mixed, dynamic, zero-absorbing)
    const P1: StaticF2<0b11110> = Int::<3> * f2(Int::<0b1010>);
    const P2: StaticF2<0b101> = f2(Int::<0b11>) * Int::<0b11>;
    const P3: StaticF2<0b101> = f2(Int::<0b11>) * f2(Int::<0b11>);
    assert_eq!(P1, Int::<0b11110>);
    assert_eq!((P2, P3), (f2(Int::<0b101>), f2(Int::<0b101>)));

    let z_f2: StaticF2<0> = f2(Int::<0>) * f2(123i32);
    let z_int: StaticF2<0> = f2(Int::<0>) * 123i32;
    let z_int_l: StaticF2<0> = 123i32 * f2(Int::<0>);
    assert_eq!(
        (z_f2, z_int, z_int_l),
        (f2(Int::<0>), f2(Int::<0>), f2(Int::<0>))
    );

    assert_eq!(3i32 * f2(0b1010i32), f2(0b11110i32));
    assert_eq!(f2(0b11i32) * 0b11i32, f2(0b101i32));

    // Euclidean division in F2[x]
    const DM1: (StaticF2<0b101>, StaticF2<0b10>) = f2_divmod(f2(Int::<0b10110>), Int::<4>);
    const DM2: (StaticF2<0b110>, StaticF2<0b1>) = f2_divmod(f2(Int::<0b1011>), Int::<3>);
    const DM3: (StaticF2<42>, StaticF2<0>) = f2_divmod(f2(Int::<42>), Int::<1>);
    assert_eq!(DM1, (f2(Int::<0b101>), f2(Int::<0b10>)));
    assert_eq!(DM2, (f2(Int::<0b110>), f2(Int::<0b1>)));
    assert_eq!(DM3, (f2(Int::<42>), f2(Int::<0>)));

    // Verify q * d + r == a && r < d across dynamic pairs
    for a in 0..64i32 {
        for d in 1..16i32 {
            let (q, r) = f2_divmod(f2(a), d);
            assert_eq!(q * d + r, f2(a));
            assert!(r < d);
        }
    }

    // Ordering and equality with int / Int
    assert!(f2(Int::<3>) < f2(Int::<4>));
    assert!(f2(Int::<3>) == 3i32);
    assert!(3i32 == f2(Int::<3>));
    assert!(f2(3i32) < Int::<8>);

    // Swizzle applied to F2 preserves F2 type
    const SW: StaticSwizzle<3, 0, 3> = Swizzle::new(Int::<3>, Int::<0>, Int::<3>);
    const SW_F2: StaticF2<0b101_111> = SW(f2(Int::<0b101_010>));
    assert_eq!(SW_F2, f2(Int::<0b101_111>));
    assert_eq!(SW(SW(f2(0b101_010i32))), f2(0b101_010i32));

    // Dynamic error panics
    assert!(catch_unwind(|| F2::new(-1i32)).is_err());
    assert!(catch_unwind(|| f2(Int::<0>) * -1i32).is_err());
    assert!(catch_unwind(|| -1i32 * f2(Int::<0>)).is_err());
    assert!(catch_unwind(|| f2(2i32) * (1i32 << 30)).is_err());
    assert!(catch_unwind(|| f2_divmod(f2(10i32), 0i32)).is_err());
    assert!(catch_unwind(|| f2_divmod(f2(10i32), -3i32)).is_err());
    assert!(catch_unwind(|| f2_divmod(f2(Int::<0>), 0i32)).is_err());
}

#[test]
fn test_f2_basis_shape_and_stride_integration() {
    use core::cmp::Ordering;

    // 1. basis: unit, BasisTerm<F2, P>, mixed axes, basis_eq, colex_cmp, as_tuple
    assert_eq!(unit(f2(Int::<9>)), f2(Int::<1>));
    let b0 = f2(Int::<9>) * e![0];
    assert_eq!(unit(b0), f2(Int::<1>) * e![0]);
    let mixed = f2(Int::<3>) * e![0] + Int::<5> * e![1];
    assert_eq!(as_tuple(mixed), tuple![f2(Int::<3>), Int::<5>]);
    assert!(basis_eq(f2(Int::<0>), Int::<0>));
    assert_eq!(colex_cmp(f2(Int::<3>), f2(Int::<8>)), Ordering::Less);
    assert_eq!(f2(Int::<0>) + e![0], e![0]);

    // 2. shape: idx2crd / crd2idx / in_bounds for F2
    let s48 = int_tuple![Int::<4>, Int::<8>];
    const C48: noodle_cute::tuple::TupleContainer<
        StaticF2<0b10>,
        noodle_cute::tuple::TupleContainer<StaticF2<0b101>, noodle_cute::tuple::Nil>,
    > = idx2crd(f2(Int::<0b10110>), &int_tuple![Int::<4>, Int::<8>]);
    assert_eq!(C48, tuple![f2(Int::<0b10>), f2(Int::<0b101>)]);
    assert_eq!(crd2idx(C48, &s48), f2(Int::<0b10110>));
    assert!(in_bounds(f2(Int::<0b10110>), &s48));
    assert!(!in_bounds(f2(Int::<32>), &s48));
    assert!(!in_bounds(f2(Int::<0>), &int_tuple![Int::<0>, Int::<4>]));

    // (3, 3) only divides by 3 (pps = 1), so carry-less prefix product agrees
    let s33 = int_tuple![Int::<3>, Int::<3>];
    let c33 = idx2crd(f2(Int::<0b10110>), &s33);
    assert_eq!(c33, tuple![f2(Int::<0b1>), f2(Int::<0b1101>)]);
    assert_eq!(crd2idx(c33, &s33), f2(Int::<0b10110>));

    // Dynamic (3, 3, 3) panics because F2(3)*F2(3) == F2(5) != 9
    assert!(catch_unwind(|| idx2crd(f2(0b10110i32), &int_tuple![3i32, 3i32, 3i32])).is_err());

    // 3. stride: inner_product, prefix_product_with, coshape (including XOR anti-cancellation)
    assert_eq!(
        prefix_product_with(int_tuple![Int::<4>, Int::<8>], f2(Int::<1>)),
        tuple![f2(Int::<1>), f2(Int::<4>)]
    );
    assert_eq!(
        inner_product(int_tuple![2i32, 1i32], tuple![f2(Int::<1>), f2(Int::<8>)]),
        f2(2 ^ 8)
    );
    // Feeding F2 coordinates into an F2 or integer stride
    assert_eq!(
        inner_product(
            tuple![f2(Int::<2>), f2(Int::<1>)],
            tuple![f2(Int::<1>), f2(Int::<9>)]
        ),
        f2(Int::<{ 2 ^ 9 }>)
    );
    assert_eq!(
        inner_product(
            tuple![f2(Int::<2>), f2(Int::<1>)],
            int_tuple![Int::<1>, Int::<4>]
        ),
        f2(Int::<6>)
    );

    // coshape: Layout(4, F2(3)) -> 8 (max contrib 3*F2(3)=F2(5), bit_len=3, 1<<3=8)
    const CS_4_3: Int<8> = coshape(Int::<4>, f2(Int::<3>));
    assert_eq!(CS_4_3, Int::<8>);

    // coshape anti-cancellation: Layout((2, 2), (F2(4), F2(4))) -> 8 (NOT 1!)
    const CS_NO_CANCEL: Int<8> = coshape(
        int_tuple![Int::<2>, Int::<2>],
        tuple![f2(Int::<4>), f2(Int::<4>)],
    );
    assert_eq!(CS_NO_CANCEL, Int::<8>);

    // coshape of 8x8 swizzled layout: (8, 8):(F2(1), F2(9)) -> 64
    const CS_SW88: Int<64> = coshape(
        int_tuple![Int::<8>, Int::<8>],
        tuple![f2(Int::<1>), f2(Int::<9>)],
    );
    assert_eq!(CS_SW88, Int::<64>);

    // coshape of mixed multi-axis layout: axis 0 is F2, axis 1 is Z
    let mixed_cs = coshape(
        int_tuple![Int::<4>, Int::<8>],
        tuple![BasisTerm::<StaticF2<3>, Mode<0>>::new(f2(Int::<3>)), e![1]],
    );
    assert_eq!(mixed_cs, int_tuple![Int::<8>, Int::<8>]);

    // coshape overflow when bit_length == 31 panics dynamically
    assert!(catch_unwind(|| coshape(2i32, f2(1i32 << 30))).is_err());

    // 4. Layout evaluation matches Swizzle pointwise on (8, 8)
    let sw = Swizzle::new(Int::<3>, Int::<0>, Int::<3>);
    let sw_layout = Layout::new(
        int_tuple![Int::<8>, Int::<8>],
        tuple![f2(Int::<1>), f2(Int::<9>)],
    );
    for c1 in 0..8 {
        for c0 in 0..8 {
            assert_eq!(sw_layout(c0, c1), sw(c0 + 8 * c1));
        }
    }
}

#[test]
fn test_f2_layout_and_core_algebra() {
    // 1. Coalesce: power-of-two F2 modes merge; non-power-of-two do NOT merge
    let merge_pow2 = Layout::new(
        int_tuple![Int::<2>, Int::<2>],
        tuple![f2(Int::<3>), f2(Int::<6>)],
    );
    assert_eq!(coalesce(merge_pow2), Layout::new(Int::<4>, f2(Int::<3>)));

    let merge_std = Layout::new(
        int_tuple![Int::<4>, Int::<8>],
        tuple![f2(Int::<1>), f2(Int::<4>)],
    );
    assert_eq!(coalesce(merge_std), Layout::new(Int::<32>, f2(Int::<1>)));

    // (3, 2):(F2(1), F2(3)) satisfies s_a * d_a == d_b, but fails linearity at (s_a - 1, 1)
    let no_merge_non_pow2 = Layout::new(
        int_tuple![Int::<3>, Int::<2>],
        tuple![f2(Int::<1>), f2(Int::<3>)],
    );
    assert_eq!(coalesce(no_merge_non_pow2), no_merge_non_pow2);
    // Pointwise verification why it cannot merge to 6:F2(1): at idx 4, (1, 1) -> 1 ^ 3 = 2 != 4
    assert_eq!(no_merge_non_pow2(4i32), f2(2i32));

    // (8, 8):(F2(1), F2(9)) stays unmerged; zero F2 strides merge for any extent
    let sw88 = Layout::new(
        int_tuple![Int::<8>, Int::<8>],
        tuple![f2(Int::<1>), f2(Int::<9>)],
    );
    assert_eq!(coalesce(sw88), sw88);
    assert_eq!(
        coalesce(Layout::new(
            int_tuple![Int::<3>, Int::<5>],
            tuple![f2(Int::<0>), f2(Int::<0>)],
        )),
        Layout::new(Int::<15>, f2(Int::<0>))
    );
    // Trailing size-1 handling in coalesce vs coalesce_extended
    let trail_unit = Layout::new(
        int_tuple![Int::<4>, Int::<1>],
        tuple![f2(Int::<1>), f2(Int::<9>)],
    );
    assert_eq!(coalesce(trail_unit), Layout::new(Int::<4>, f2(Int::<1>)));
    assert_eq!(coalesce_extended(trail_unit), trail_unit);

    // 2. Nullspace on F2 layouts
    assert_eq!(nullspace(sw88), Layout::new(Int::<1>, Int::<0>));
    let with_zero = Layout::new(
        int_tuple![Int::<4>, Int::<3>, Int::<2>],
        tuple![f2(Int::<1>), f2(Int::<0>), f2(Int::<4>)],
    );
    assert_eq!(nullspace(with_zero), Layout::new(Int::<3>, Int::<4>));

    // 3. make_layout_like and tiler_to_layout_with(F2(1))
    let sw88_rev = Layout::new(
        int_tuple![Int::<8>, Int::<8>],
        tuple![f2(Int::<9>), f2(Int::<1>)],
    );
    assert_eq!(
        make_layout_like(&sw88_rev),
        Layout::new(
            int_tuple![Int::<8>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>]
        )
    );
    let lifted = tiler_to_layout_with(
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>],
        ),
        f2(Int::<1>),
    );
    assert_eq!(
        lifted,
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            tuple![f2(Int::<1>), f2(Int::<4>)],
        )
    );

    // 4. layout_add constructs a swizzled layout from linear F2 + XOR correction
    let base_f2 = Layout::new(
        int_tuple![Int::<8>, Int::<8>],
        tuple![f2(Int::<1>), f2(Int::<8>)],
    );
    let bank_xor = Layout::new(
        int_tuple![Int::<8>, Int::<8>],
        tuple![f2(Int::<0>), f2(Int::<1>)],
    );
    assert_eq!(layout_add(base_f2, bank_xor), sw88);

    // 5. Composition, logical_divide, zipped_divide with F2 layouts
    let tiler = Layout::new(
        int_tuple![Int::<4>, Int::<4>],
        int_tuple![Int::<2>, Int::<8>],
    );
    let comp = composition(sw88, tiler);
    for i in 0..comp.size() {
        assert_eq!(comp(i), sw88(tiler(i)));
    }
    // Rank-1 composition with F2 RHS stride
    assert_eq!(
        composition(
            Layout::new(Int::<8>, f2(Int::<1>)),
            Layout::new(Int::<4>, f2(Int::<2>)),
        ),
        Layout::new(Int::<4>, f2(Int::<2>))
    );
    // Multi-mode F2 LHS composed with shape-1 F2 RHS: A o 1:F2(9) == 1:A(F2(9))
    assert_eq!(
        composition(sw88, Layout::new(Int::<1>, f2(Int::<9>))),
        Layout::new(Int::<1>, f2(Int::<8>))
    );

    // logical_divide & zipped_divide on swizzled layout
    let by_mode = logical_divide(sw88, int_tuple![Int::<4>, Int::<4>]);
    let zipped = zipped_divide(sw88, int_tuple![Int::<4>, Int::<4>]);
    assert_eq!(by_mode.size(), sw88.size());
    assert_eq!(zipped.size(), sw88.size());
    for m1 in 0..8 {
        for m0 in 0..8 {
            assert_eq!(
                by_mode(int_tuple![m0 % 4, m0 / 4], int_tuple![m1 % 4, m1 / 4]),
                sw88(m0, m1)
            );
            assert_eq!(
                zipped(int_tuple![m0 % 4, m1 % 4], int_tuple![m0 / 4, m1 / 4]),
                sw88(m0, m1)
            );
        }
    }

    // 6. Recast on F2 layouts
    let f2_4x4 = Layout::new(
        int_tuple![Int::<4>, Int::<4>],
        tuple![f2(Int::<4>), f2(Int::<1>)],
    );
    assert_eq!(
        recast(&f2_4x4, Int::<4>),
        Layout::new(
            int_tuple![Int::<4>, Int::<1>],
            tuple![f2(Int::<1>), f2(Int::<1>)],
        )
    );
    assert_eq!(
        recast(&f2_4x4, Fraction::new(Int::<1>, Int::<2>)),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            tuple![f2(Int::<8>), f2(Int::<1>)],
        )
    );
    // Dynamic indivisible F2 recast panics
    let dyn_sw = Layout::new(int_tuple![8i32, 8i32], tuple![f2(1i32), f2(9i32)]);
    assert!(catch_unwind(|| recast(&dyn_sw, 4i32)).is_err());
}

#[test]
fn test_f2_right_and_left_inverse() {
    // 1. Self-inverse 8x8 swizzle: right_inverse((8, 8):(F1, F9)) == (8, 8):(F1, F9)
    type Pair<A, B> = noodle_cute::tuple::TupleContainer<
        A,
        noodle_cute::tuple::TupleContainer<B, noodle_cute::tuple::Nil>,
    >;
    type Sw88Layout = Layout<Pair<Int<8>, Int<8>>, Pair<StaticF2<1>, StaticF2<9>>>;
    const SW88: Sw88Layout = Layout::new(
        int_tuple![Int::<8>, Int::<8>],
        tuple![f2(Int::<1>), f2(Int::<9>)],
    );
    const R_SW88: Sw88Layout = right_inverse(SW88);
    assert_eq!(R_SW88, SW88);
    for i in 0..64 {
        let ri = R_SW88(i);
        assert_eq!(SW88(ri), f2(i));
        assert_eq!(R_SW88(SW88(ri)), ri);
    }

    // 2. Transposed 8x8 swizzle: right_inverse((8, 8):(F9, F1)) == (8, 8):(F8, F9)
    let sw88_t = Layout::new(
        int_tuple![Int::<8>, Int::<8>],
        tuple![f2(Int::<9>), f2(Int::<1>)],
    );
    let r_sw88_t = right_inverse(sw88_t);
    assert_eq!(
        r_sw88_t,
        Layout::new(
            int_tuple![Int::<8>, Int::<8>],
            tuple![f2(Int::<8>), f2(Int::<9>)],
        )
    );
    for i in 0..64 {
        let ri = r_sw88_t(i);
        assert_eq!(sw88_t(ri), f2(i));
        assert_eq!(r_sw88_t(sw88_t(ri)), ri);
    }

    // 3. 3-mode Swizzle(3, 2, 3) layout: (4, 8, 8):(F1, F4, F36)
    let sw_323 = Swizzle::new(Int::<3>, Int::<2>, Int::<3>);
    let l_323 = Layout::new(
        int_tuple![Int::<4>, Int::<8>, Int::<8>],
        tuple![f2(Int::<1>), f2(Int::<4>), f2(Int::<36>)],
    );
    let r_323 = right_inverse(l_323);
    assert_eq!(
        r_323,
        Layout::new(
            int_tuple![Int::<32>, Int::<8>],
            tuple![f2(Int::<1>), f2(Int::<36>)],
        )
    );
    for i in 0..256 {
        assert_eq!(l_323(i), sw_323(i));
        let ri = r_323(i);
        assert_eq!(l_323(ri), f2(i));
        assert_eq!(r_323(l_323(ri)), ri);
    }

    // 4. Out-of-reach residue stops the chain cleanly (valid partial right inverse)
    let bad_reach = Layout::new(
        int_tuple![Int::<4>, Int::<8>],
        tuple![f2(Int::<1>), f2(Int::<5>)],
    );
    let r_bad_reach = right_inverse(bad_reach);
    assert_eq!(r_bad_reach, Layout::new(Int::<4>, f2(Int::<1>)));
    for i in 0..r_bad_reach.size() {
        assert_eq!(bad_reach(r_bad_reach(i)), f2(i));
    }

    // 5. Negative-shift Swizzle(2, 3, -2) pointwise match with F2 layout + partial right_inverse
    let sw_neg = Swizzle::new(Int::<2>, Int::<3>, Int::<-2>);
    let l_neg = Layout::new(
        int_tuple![Int::<8>, Int::<4>, Int::<4>],
        tuple![f2(Int::<1>), f2(Int::<40>), f2(Int::<32>)],
    );
    for i in 0..128 {
        assert_eq!(l_neg(i), sw_neg(i));
    }
    let r_neg = right_inverse(l_neg);
    assert_eq!(r_neg, Layout::new(Int::<8>, f2(Int::<1>)));
    for i in 0..r_neg.size() {
        assert_eq!(l_neg(r_neg(i)), f2(i));
    }

    // 6. Degenerate F2 layouts (all zero stride / size-1)
    assert_eq!(
        right_inverse(Layout::new(Int::<4>, f2(Int::<0>))),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        right_inverse(Layout::new(Int::<1>, f2(Int::<9>))),
        Layout::new(Int::<1>, Int::<0>)
    );

    // 7. Dynamic F2 right_inverse (dynamic extents and dynamic F2 strides)
    let dyn_sw = Layout::new(int_tuple![8i32, 8i32], tuple![f2(1i32), f2(9i32)]);
    let r_dyn = right_inverse(dyn_sw);
    assert_eq!(r_dyn, dyn_sw);
    for i in 0..64 {
        let ri = r_dyn(i);
        assert_eq!(dyn_sw(ri), f2(i));
        assert_eq!(r_dyn(dyn_sw(ri)), ri);
    }

    // Dynamic extent on zero-residue mode still continues the chain
    let dyn_s_zero_res = Layout::new(
        int_tuple![8i32, Int::<4>],
        tuple![f2(Int::<1>), f2(Int::<8>)],
    );
    let r_dyn_s = right_inverse(dyn_s_zero_res);
    for i in 0..32 {
        assert_eq!(dyn_s_zero_res(r_dyn_s(i)), f2(i));
    }

    // 8. Multi-axis layout: axis 0 is F2-swizzled, axis 1 is ordinary Z
    let multi_axis = Layout::new(
        int_tuple![(Int::<8>, Int::<8>), Int::<4>],
        tuple![(f2(Int::<1>) * e![0], f2(Int::<9>) * e![0]), e![1]],
    );
    let r_multi = right_inverse(multi_axis);
    assert_eq!(
        r_multi,
        Layout::new(
            int_tuple![(Int::<8>, Int::<8>), Int::<4>],
            tuple![(f2(Int::<1>), f2(Int::<9>)), Int::<64>],
        )
    );

    for i in 0..256 {
        assert_eq!(r_multi(multi_axis(i)), f2(i));
    }

    // 9. left_inverse on F2: supported when every gap is 1 (single unit-stride mode)
    assert_eq!(
        left_inverse(Layout::new(Int::<8>, f2(Int::<1>))),
        Layout::new(Int::<8>, Int::<1>)
    );
    assert_eq!(
        left_inverse(Layout::new(Int::<4>, f2(Int::<0>))),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        left_inverse(Layout::new(8i32, f2(1i32))),
        Layout::new(8i32, Int::<1>)
    );
    assert!(catch_unwind(|| left_inverse(Layout::new(8i32, f2(2i32)))).is_err());
}

#[test]
fn recast_f2_element_offsets() {
    assert_eq!(
        recast(&Layout::new(8i32, f2(2i32)), 4i32),
        Layout::new(4i32, f2(1i32))
    );
    assert!(catch_unwind(|| recast(&Layout::new(2i32, f2(5i32)), 3i32)).is_err());
    let packed = recast(&Layout::new(Int::<8>, f2(Int::<2>)), Int::<4>);
    assert_eq!(packed, Layout::new(Int::<4>, f2(Int::<1>)));
    let original = Layout::new(3i32, f2(3i32));
    let scaled = recast(&original, 3i32);
    for i in 0..3 {
        assert_eq!(scaled(i).value() * 3, original(i).value());
    }
}

#[test]
fn default_preserves_the_scalar_type() {
    assert_eq!(StaticF2::<7>::default().value(), 7);
    assert_eq!(F2::<int>::default().value(), 0);
}

#[test]
fn right_inverse_recovers_each_binary_column() {
    let a = Layout::new(
        int_tuple![Int::<2>, Int::<2>, Int::<4>],
        tuple![f2(Int::<1>), f2(Int::<3>), f2(Int::<5>)],
    );
    let r = right_inverse(a);
    assert_eq!(r.size(), 16);
    for i in 0..16 {
        assert_eq!(a(r(i)), f2(i));
    }
}

#[test]
fn integer_composition_detects_carries() {
    let a = Layout::new(16i32, f2(1i32));
    assert!(catch_unwind(|| composition(a, Layout::new(4i32, 3i32))).is_err());
    assert!(
        catch_unwind(|| composition(
            a,
            Layout::new(int_tuple![2i32, 2i32], int_tuple![1i32, 1i32])
        ))
        .is_err()
    );
}

#[test]
fn inverse_of_partial_binary_radix() {
    let a = Layout::new(
        int_tuple![Int::<3>, Int::<2>],
        tuple![f2(Int::<1>), f2(Int::<3>)],
    );
    let r = right_inverse(a);
    assert_eq!(r.size(), 4);
    for i in 0..4 {
        assert_eq!(a(r(i)), f2(i));
    }
    let a = Layout::new(int_tuple![3i32, 2i32], tuple![f2(1i32), f2(3i32)]);
    let r = right_inverse(a);
    assert_eq!(r.size(), 4);
    for i in 0..4 {
        assert_eq!(a(r(i)), f2(i));
    }
}

#[test]
fn runtime_inverse_solves_each_binary_column() {
    let a = Layout::new(
        int_tuple![2i32, 2i32, 4i32],
        tuple![f2(1i32), f2(3i32), f2(5i32)],
    );
    let r = right_inverse(a);
    assert_eq!(r.size(), 16);
    for i in 0..16 {
        assert_eq!(a(r(i)), f2(i));
    }
}
