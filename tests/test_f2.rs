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
use noodle_cute::f2::{F2, StaticF2, f2, f2_divmod};
use noodle_cute::integer::{Int, add, bitand, int, mul};
use noodle_cute::layout::{Fraction, Layout, make_layout_like, recast, tiler_to_layout_with};
use noodle_cute::shape::{crd2idx, idx2crd, in_bounds};
use noodle_cute::stride::{coshape, inner_product, prefix_product_with};
use noodle_cute::swizzle::{StaticSwizzle, Swizzle};
use noodle_cute::tuple::Item;
use noodle_cute::{e, f2 as f2_macro, int_tuple, tuple};

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
    const C48: noodle_cute::tuple::Til<
        StaticF2<0b10>,
        noodle_cute::tuple::Til<StaticF2<0b101>, noodle_cute::tuple::Nil>,
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
    type Pair<A, B> =
        noodle_cute::tuple::Til<A, noodle_cute::tuple::Til<B, noodle_cute::tuple::Nil>>;
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

#[test]
fn hierarchical_swizzle_keeps_the_non_power_of_two_tail() {
    // docs/06_swizzle.md: the two leading coordinates mix by XOR within each slab.
    let layout = Layout::new(
        int_tuple![Int::<4>, (Int::<4>, Int::<3>)],
        tuple![f2(Int::<1>), (f2(Int::<5>), f2(Int::<16>))],
    );
    let rows = [[0, 1, 2, 3], [5, 4, 7, 6], [10, 11, 8, 9], [15, 14, 13, 12]];
    for (j, row) in (0..4).zip(rows) {
        for k in 0..3 {
            for (i, offset) in (0..4).zip(row) {
                let expected = f2(add(offset, mul(k, 16i32)));
                assert_eq!(layout(i, int_tuple![j, k]), expected);
            }
        }
    }
}

#[test]
fn coalesce_f2_can_merge_a_non_power_of_two_final_extent() {
    // test/test_coalesce.py: only the preceding radix must be a power of two.
    let layout = Layout::new(
        int_tuple![Int::<2>, Int::<2>, Int::<3>],
        tuple![f2(Int::<1>), f2(Int::<2>), f2(Int::<4>)],
    );
    let result = coalesce(layout);
    assert_eq!(result, Layout::new(Int::<12>, f2(Int::<1>)));
    for i in 0..layout.size() {
        assert_eq!(result(i), layout(i));
    }
}

#[test]
fn refined_swizzle_columns_coalesce_back_to_the_same_layout() {
    // docs/GPUMODE_114_PyCuTe.pdf, p. 88: each binary coordinate supplies one column.
    let layout = Layout::new(
        int_tuple![Int::<4>, Int::<4>],
        tuple![f2(Int::<1>), f2(Int::<5>)],
    );
    let refined = Layout::new(
        int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<2>)],
        tuple![(f2(Int::<1>), f2(Int::<2>)), (f2(Int::<5>), f2(Int::<10>))],
    );
    assert_eq!(coalesce(refined), layout);
    for i in 0..layout.size() {
        assert_eq!(refined(i), layout(i));
    }
}

#[test]
fn swizzled_rows_and_columns_each_span_eight_banks() {
    // docs/GPUMODE_114_PyCuTe.pdf, p. 86: use the low three address bits as bank IDs.
    let shape = int_tuple![Int::<8>, Int::<8>];
    let row_major = Layout::new(shape, int_tuple![Int::<8>, Int::<1>]);
    let row_swizzle = Layout::new(shape, tuple![f2(9i32), f2(1i32)]);
    let column_swizzle = Layout::new(shape, tuple![f2(1i32), f2(9i32)]);
    for fixed in 0..8 {
        assert!((0..8).all(|row| bitand(row_major(row, fixed), 7i32) == fixed));
        for swizzled in [row_swizzle, column_swizzle] {
            let mut row_banks: Vec<_> = (0..8)
                .map(|column| bitand(swizzled(fixed, column).value(), 7i32))
                .collect();
            let mut col_banks: Vec<_> = (0..8)
                .map(|row| bitand(swizzled(row, fixed).value(), 7i32))
                .collect();
            row_banks.sort_unstable();
            col_banks.sort_unstable();
            assert_eq!(row_banks, [0, 1, 2, 3, 4, 5, 6, 7]);
            assert_eq!(col_banks, row_banks);
        }
    }
}

#[test]
fn right_inverse_covers_the_full_hierarchical_swizzle() {
    // test/test_inverse_right.py and docs/GPUMODE_114_PyCuTe.pdf, p. 89.
    let layout = Layout::new(
        int_tuple![Int::<4>, (Int::<4>, Int::<3>)],
        tuple![f2(Int::<1>), (f2(Int::<5>), f2(Int::<16>))],
    );
    const {
        let static_layout = Layout::new(
            int_tuple![Int::<4>, (Int::<4>, Int::<3>)],
            tuple![f2(Int::<1>), (f2(Int::<5>), f2(Int::<16>))],
        );
        let static_inv = right_inverse(static_layout);
        assert!(static_inv.size() == 48);
        assert!(static_layout(static_inv(47i32)).value() == 47);
    }
    let inverse = right_inverse(layout);
    assert_eq!(inverse.size(), 48);
    for i in 0..layout.size() {
        assert_eq!(layout(inverse(i)), f2(i));
    }

    let dynamic_layout = Layout::new(
        int_tuple![4i32, (4i32, 3i32)],
        tuple![f2(1i32), (f2(5i32), f2(16i32))],
    );
    let dynamic_inverse = right_inverse(dynamic_layout);
    assert_eq!(dynamic_inverse.size(), 48);
    for i in 0..dynamic_layout.size() {
        assert_eq!(dynamic_layout(dynamic_inverse(i)), f2(i));
    }

    // Coalesced trailing non-power-of-two segment `(2, 3):(F2(16), F2(32)) -> 6:F2(16)`.
    let coalesced_tail = Layout::new(
        int_tuple![Int::<4>, Int::<4>, Int::<2>, Int::<3>],
        tuple![f2(Int::<1>), f2(Int::<5>), f2(Int::<16>), f2(Int::<32>)],
    );
    let coalesced_inv = right_inverse(coalesced_tail);
    assert_eq!(coalesced_inv.size(), 96);
    for i in 0..coalesced_tail.size() {
        assert_eq!(coalesced_tail(coalesced_inv(i)), f2(i));
    }

    // Multi-axis layout whose trailing non-power-of-two mode sits at the end of the domain.
    let multi_axis = Layout::new(
        int_tuple![Int::<2>, (Int::<4>, (Int::<4>, Int::<3>))],
        tuple![
            f2(Int::<1>) * e![1],
            (
                f2(Int::<1>) * e![0],
                (f2(Int::<5>) * e![0], f2(Int::<16>) * e![0])
            )
        ],
    );
    let multi_inv = right_inverse(multi_axis);
    assert_eq!(multi_inv.size(), 96);
    for i0 in 0..48 {
        for i1 in 0..2 {
            assert_eq!(
                multi_axis(multi_inv(int_tuple![i0, i1])),
                f2(i0) * e![0] + f2(i1) * e![1]
            );
        }
    }
}

#[test]
fn right_inverse_preserves_partial_inverse_on_holes_and_non_trailing_modes() {
    // Gap at 16 prevents extending into the trailing mode; keep the 0..16 partial inverse.
    let with_hole = Layout::new(
        int_tuple![Int::<4>, Int::<4>, Int::<3>],
        tuple![f2(Int::<1>), f2(Int::<5>), f2(Int::<32>)],
    );
    let hole_inv = right_inverse(with_hole);
    assert_eq!(hole_inv.size(), 16);
    for i in 0..hole_inv.size() {
        assert_eq!(with_hole(hole_inv(i)), f2(i));
    }

    // Trailing stride F2(17) != F2(16): fall back to the pure binary partial inverse (32).
    let with_residue = Layout::new(
        int_tuple![Int::<4>, Int::<4>, Int::<3>],
        tuple![f2(Int::<1>), f2(Int::<5>), f2(Int::<17>)],
    );
    let residue_inv = right_inverse(with_residue);
    assert_eq!(residue_inv.size(), 32);
    for i in 0..residue_inv.size() {
        assert_eq!(with_residue(residue_inv(i)), f2(i));
    }

    // Trailing broadcast leaf after non-power-of-two mode 3: idx2crd divides by 3,
    // so only the binary prefix 0..32 is safely invertible.
    let non_last_broadcast = Layout::new(
        int_tuple![Int::<4>, Int::<4>, Int::<3>, Int::<2>],
        tuple![f2(Int::<1>), f2(Int::<5>), f2(Int::<16>), f2(Int::<0>)],
    );
    let non_last_inv = right_inverse(non_last_broadcast);
    assert_eq!(non_last_inv.size(), 32);
    for i in 0..non_last_inv.size() {
        assert_eq!(non_last_broadcast(non_last_inv(i)), f2(i));
    }

    let non_last_unit = Layout::new(
        int_tuple![Int::<4>, Int::<4>, Int::<3>, Int::<1>],
        tuple![f2(Int::<1>), f2(Int::<5>), f2(Int::<16>), f2(Int::<0>)],
    );
    let non_last_unit_inv = right_inverse(non_last_unit);
    assert_eq!(non_last_unit_inv.size(), 32);
    for i in 0..non_last_unit_inv.size() {
        assert_eq!(non_last_unit(non_last_unit_inv(i)), f2(i));
    }
}

#[test]
fn layout_add_rejects_carry_in_the_common_f2_domain() {
    // test/test_layout_add.py: the extent-three radix introduces a carry at index 4.
    let a = Layout::new(9i32, f2(1i32));
    let b = Layout::new(int_tuple![3i32, 3i32], tuple![f2(2i32), f2(4i32)]);
    assert!(catch_unwind(|| layout_add(a, b)).is_err());
    assert!(catch_unwind(|| layout_add(b, a)).is_err());

    // Static shape with dynamic nonzero F2 stride also panics at runtime.
    let mixed_nonzero = Layout::new(Int::<9>, f2(1i32));
    let static_b = Layout::new(
        int_tuple![Int::<3>, Int::<3>],
        tuple![f2(Int::<2>), f2(Int::<4>)],
    );
    assert!(catch_unwind(|| layout_add(mixed_nonzero, static_b)).is_err());
    assert!(catch_unwind(|| layout_add(static_b, mixed_nonzero)).is_err());

    // Multi-mode F2 refinement that splits a nonzero extent-6 mode at radix 3.
    let left = Layout::new(int_tuple![3i32, 6i32], tuple![f2(1i32), f2(4i32)]);
    let right = Layout::new(int_tuple![6i32, 3i32], tuple![f2(2i32), f2(8i32)]);
    assert!(catch_unwind(|| layout_add(left, right)).is_err());
}

#[test]
fn layout_add_preserves_valid_f2_refinements() {
    let a_static = Layout::new(
        int_tuple![Int::<3>, Int::<3>],
        tuple![f2(Int::<1>), f2(Int::<2>)],
    );
    let b_static = Layout::new(
        int_tuple![Int::<3>, Int::<3>],
        tuple![f2(Int::<2>), f2(Int::<4>)],
    );
    let sum_static = layout_add(a_static, b_static);
    const {
        let a = Layout::new(
            int_tuple![Int::<3>, Int::<3>],
            tuple![f2(Int::<1>), f2(Int::<2>)],
        );
        let b = Layout::new(
            int_tuple![Int::<3>, Int::<3>],
            tuple![f2(Int::<2>), f2(Int::<4>)],
        );
        let s = layout_add(a, b);
        assert!(s(4i32).value() == (a(4i32) + b(4i32)).value());
    }
    assert_eq!(
        sum_static,
        Layout::new(
            int_tuple![Int::<3>, Int::<3>],
            tuple![f2(Int::<3>), f2(Int::<6>)],
        )
    );
    for i in 0..9 {
        assert_eq!(sum_static(i), a_static(i) + b_static(i));
    }

    let a_dyn = Layout::new(int_tuple![3i32, 3i32], tuple![f2(1i32), f2(2i32)]);
    let b = Layout::new(int_tuple![3i32, 3i32], tuple![f2(2i32), f2(4i32)]);
    let sum_dyn = layout_add(a_dyn, b);
    let zero = Layout::new(9i32, f2(0i32));
    let broadcast_sum = layout_add(zero, b);
    let mixed_zero = Layout::new(Int::<9>, f2(0i32));
    let mixed_broadcast = layout_add(mixed_zero, b_static);
    let cancelled = layout_add(b, b);
    for i in 0..9 {
        assert_eq!(sum_dyn(i), a_dyn(i) + b(i));
        assert_eq!(broadcast_sum(i), b(i));
        assert_eq!(mixed_broadcast(i), b_static(i));
        assert_eq!(cancelled(i), f2(0i32));
    }

    let a = Layout::new(12i32, f2(3i32));
    let b = Layout::new(int_tuple![4i32, 3i32], tuple![f2(2i32), f2(4i32)]);
    let sum = layout_add(a, b);
    for i in 0..12 {
        assert_eq!(sum(i), a(i) + b(i));
    }

    // Splitting extent 6 into (2, 3) uses power-of-two sub-mode radix 2 and is carry-free.
    let split_a = Layout::new(
        int_tuple![Int::<6>, Int::<2>],
        tuple![f2(Int::<1>), f2(Int::<8>)],
    );
    let split_b = Layout::new(
        int_tuple![Int::<2>, Int::<3>, Int::<2>],
        tuple![f2(Int::<2>), f2(Int::<4>), f2(Int::<16>)],
    );
    let split_sum = layout_add(split_a, split_b);
    for i in 0..12 {
        assert_eq!(split_sum(i), split_a(i) + split_b(i));
    }
}
