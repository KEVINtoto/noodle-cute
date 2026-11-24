#![feature(const_trait_impl)]

use noodle_cute::basis::BasisCombination;
use noodle_cute::integer::{Int, int};
use noodle_cute::shape::{
    Shape, common_coarsening, common_refinement, compatible, coord_to_index, coordinates, crd2idx,
    has_zero, idx2crd, in_bounds, index_to_coord, size,
};
use noodle_cute::tuple::{Nil, Til, TupleContainer};
use noodle_cute::{basis_tuple, e, int_tuple, tuple};

#[test]
fn size_works_across_shapes() {
    assert_eq!(Nil.size(), 1);
    assert!(!Nil.has_zero());

    assert_eq!(4i32.size(), 4);
    assert_eq!(Int::<8>.size(), 8);
    assert_eq!(0i32.size(), 0);
    assert!(Int::<0>.has_zero());

    let nested = Til(Int::<2>, Til(Til(3i32, Til(Int::<5>, Nil)), Nil));
    assert_eq!(nested.size(), 30);
    assert!(!nested.has_zero());

    let with_zero = Til(0i32, Til(Til(i32::MAX, Til(i32::MAX, Nil)), Nil));
    assert!(with_zero.has_zero());
    assert_eq!(with_zero.size(), 0);

    let overflow = Til(65536i32, Til(65536i32, Nil));
    assert!(std::panic::catch_unwind(|| overflow.size()).is_err());

    // Negative dynamic extents panic at runtime
    assert!(std::panic::catch_unwind(|| (-1i32).size()).is_err());
    assert!(std::panic::catch_unwind(|| (-1i32).has_zero()).is_err());
    assert!(std::panic::catch_unwind(|| int_tuple![2i32, -3i32].size()).is_err());
    assert!(std::panic::catch_unwind(|| int_tuple![0i32, -3i32].has_zero()).is_err());
}

#[test]
fn compatibility_checks_rank_and_size() {
    let s = Til(2i32, Til(Til(3i32, Til(5i32, Nil)), Nil));
    assert!(compatible(&30i32, &s));
    assert!(compatible(&Til(2i32, Til(15i32, Nil)), &s));
    assert!(!compatible(&s, &30i32));
    assert!(!compatible(
        &Til(4i32, Til(6i32, Nil)),
        &Til(Til(2i32, Til(3i32, Nil)), Til(8i32, Nil))
    ));

    assert!(compatible(&Nil, &Nil));
    assert!(compatible(&1i32, &Nil));
    assert!(!compatible(&2i32, &Nil));
    assert!(!compatible(&Nil, &1i32));
    assert!(!compatible(&Til(1i32, Nil), &Nil));
    assert!(compatible(&Til(Nil, Nil), &Til(Nil, Nil)));
    assert!(
        std::panic::catch_unwind(|| compatible(&0i32, &Til(65536i32, Til(65536i32, Nil)))).is_err()
    );
    assert!(compatible(
        &0i32,
        &Til(0i32, Til(Til(i32::MAX, Til(i32::MAX, Nil)), Nil))
    ));
}

#[test]
fn coord_conversion_supports_signed_unsigned_and_nested_nil() {
    assert_eq!(coord_to_index(Nil, &Nil), Int::<0>);
    assert_eq!(index_to_coord(Nil, &Nil), Nil);
    assert_eq!(index_to_coord(0i32, &Nil), Nil);

    let s = Til(3i32, Til(Til(2i32, Til(3i32, Nil)), Nil));
    let crd = Til(1i32, Til(Til(0i32, Til(1i32, Nil)), Nil));
    assert_eq!(crd2idx(crd, &s), 7i32);
    assert_eq!(idx2crd(7i32, &s), crd);

    // Signed scalar and tuple coordinates work as well.
    assert_eq!(
        crd2idx(Til(1i32, Til(Int::<2>, Nil)), &Til(3i32, Til(4i32, Nil))),
        7i32
    );
    assert_eq!(
        idx2crd(7i32, &Til(3i32, Til(4i32, Nil))),
        Til(1i32, Til(2i32, Nil))
    );
    assert_eq!(idx2crd(Int::<2>, &Til(5i32, Nil)), Til(Int::<2>, Nil));

    // Nested Nil mode inside shape.
    let s_nil = Til(Nil, Til(4i32, Nil));
    assert_eq!(idx2crd(2i32, &s_nil), Til(Nil, Til(2i32, Nil)));
    assert_eq!(crd2idx(Til(Nil, Til(2i32, Nil)), &s_nil), 2i32);
}

#[test]
fn shape_operations_support_const_eval() {
    const S: TupleContainer<Int<3>, TupleContainer<Int<4>, Nil>> =
        Til(Int::<3>, Til(Int::<4>, Nil));
    const SZ: int = S.size();
    const MATCH: bool = compatible(&Int::<12>, &S);
    const CRD: TupleContainer<int, TupleContainer<int, Nil>> = Til(1i32, Til(2i32, Nil));
    const IDX: int = coord_to_index(CRD, &S);
    const BACK: TupleContainer<int, TupleContainer<int, Nil>> = index_to_coord(7i32, &S);
    assert_eq!(SZ, 12);
    const { assert!(MATCH) };
    assert_eq!(IDX, 7);
    assert_eq!(BACK, CRD);
}

#[test]
fn compatibility_matches_python_examples() {
    assert!(compatible(&30i32, &int_tuple![2i32, 15i32]));
    assert!(compatible(
        &int_tuple![2i32, 15i32],
        &int_tuple![2i32, (3i32, 5i32)]
    ));
    assert!(compatible(&30i32, &int_tuple![2i32, (3i32, 5i32)]));
    assert!(compatible(&24i32, &int_tuple![(2i32, 2i32), (3i32, 2i32)]));
    assert!(!compatible(&24i32, &32i32));
    assert!(!compatible(
        &int_tuple![4i32, 6i32],
        &int_tuple![(2i32, 3i32), 8i32]
    ));
    assert!(!compatible(
        &int_tuple![2i32, (3i32, 5i32)],
        &int_tuple![(3i32, 2i32), 5i32]
    ));
    assert!(compatible(&24i32, &int_tuple![24i32]));
    assert!(!compatible(&int_tuple![24i32], &24i32));
}

#[test]
fn empty_shapes_and_rank_mismatches() {
    assert!(compatible(&Nil, &Nil));
    assert!(compatible(&1i32, &Nil));
    assert!(!compatible(&2i32, &Nil));
    assert!(!compatible(&Nil, &1i32));
    assert!(!compatible(&Nil, &int_tuple![1i32]));
    assert!(!compatible(&int_tuple![1i32], &Nil));
    assert!(!compatible(&int_tuple![2i32], &int_tuple![2i32, 1i32]));
    assert!(!compatible(&int_tuple![2i32, 1i32], &int_tuple![2i32]));
    assert!(compatible(&tuple![Nil], &tuple![Nil]));
}

#[test]
fn static_and_dynamic_extents_support_const_evaluation() {
    const MATCH: bool = compatible(
        &int_tuple![Int::<2>, 15i32],
        &int_tuple![2i32, (Int::<3>, 5i32)],
    );
    const MISMATCH: bool = compatible(&Int::<24>, &int_tuple![Int::<2>, Int::<15>]);
    const { assert!(MATCH) };
    const { assert!(!MISMATCH) };

    // The same Rust types can yield different results at run time.
    for extent in [14i32, 15i32, 16i32] {
        assert_eq!(
            compatible(&30i32, &int_tuple![Int::<2>, extent]),
            extent == 15
        );
    }
}

#[test]
fn size_comparison_panics_on_dynamic_overflow_and_short_circuits_on_zero() {
    assert!(compatible(&i32::MAX, &int_tuple![Int::<1>, i32::MAX]));
    assert!(
        std::panic::catch_unwind(|| compatible(&0i32, &int_tuple![i32::MAX, i32::MAX, 2i32]))
            .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| compatible(&1i32, &int_tuple![i32::MAX, i32::MAX])).is_err()
    );
    assert!(
        std::panic::catch_unwind(|| compatible(&0i32, &int_tuple![65536i32, 65536i32])).is_err()
    );
    assert!(compatible(&0i32, &int_tuple![(i32::MAX, i32::MAX), 0i32]));
    assert!(compatible(&0i32, &int_tuple![0i32, (i32::MAX, i32::MAX)]));
}

#[test]
fn test_shape_size() {
    // Nil
    assert_eq!(Nil.size(), 1);
    assert!(!Nil.has_zero());

    // Integer dynamic
    assert_eq!(4i32.size(), 4);
    assert!(!4i32.has_zero());
    assert!(0i32.has_zero());
    assert_eq!(0i32.size(), 0);

    // Integer static
    assert_eq!(Int::<8>.size(), 8);
    assert!(!Int::<8>.has_zero());
    assert!(Int::<0>.has_zero());

    // Flat tuples
    let s_flat = int_tuple![Int::<3>, 4i32, Int::<5>];
    assert_eq!(s_flat.size(), 60);
    assert!(!s_flat.has_zero());

    // Nested tuples
    let s_nested = int_tuple![Int::<2>, (3i32, (Int::<4>, 5i32))];
    assert_eq!(s_nested.size(), 120);
    assert!(!s_nested.has_zero());

    // Zero leaf extents
    let s_with_zero = int_tuple![Int::<2>, (0i32, Int::<5>)];
    assert_eq!(s_with_zero.size(), 0);
    assert!(s_with_zero.has_zero());

    // Overflow detection
    let s_overflow = int_tuple![65536i32, 65536i32];
    assert!(std::panic::catch_unwind(|| s_overflow.size()).is_err());

    let s_overflow_nested = int_tuple![Int::<2>, (i32::MAX, Int::<2>)];
    assert!(std::panic::catch_unwind(|| s_overflow_nested.size()).is_err());

    // Zero absorbs overflow
    let s_zero_overflow = int_tuple![0i32, (i32::MAX, i32::MAX)];
    assert_eq!(s_zero_overflow.size(), 0);
}

#[test]
fn test_coord_to_index_and_index_to_coord() {
    // 1. Nil shape & coord
    assert_eq!(coord_to_index(Nil, &Nil), Int::<0>);
    assert_eq!(index_to_coord(Nil, &Nil), Nil);

    // 2. Integer shape
    let s_scalar = 5i32;
    assert_eq!(coord_to_index(3i32, &s_scalar), 3i32);
    assert_eq!(index_to_coord(3i32, &s_scalar), 3i32);
    assert_eq!(crd2idx(3i32, &s_scalar), 3i32);
    assert_eq!(idx2crd(3i32, &s_scalar), 3i32);

    // 3. 1D tuple shape (rank-1 tuple)
    let s_rank1 = int_tuple![4i32];
    assert_eq!(coord_to_index(int_tuple![2i32], &s_rank1), 2i32);
    assert_eq!(index_to_coord(2i32, &s_rank1), int_tuple![2i32]);

    // 4. 2D flat tuple: (3, 4)
    let s_flat = int_tuple![3i32, 4i32];
    // idx = 1 + 2 * 3 = 7
    let crd_flat = int_tuple![1i32, 2i32];
    assert_eq!(coord_to_index(crd_flat, &s_flat), 7i32);
    assert_eq!(index_to_coord(7i32, &s_flat), crd_flat);

    // 5. Nested shape: (3, (2, 3)) from PyCuTe doc
    // 7 -> (1, (0, 1))
    let s_nested = int_tuple![3i32, (2i32, 3i32)];
    let crd_nested = int_tuple![1i32, (0i32, 1i32)];
    assert_eq!(coord_to_index(crd_nested, &s_nested), 7i32);
    assert_eq!(index_to_coord(7i32, &s_nested), crd_nested);

    // 6. 3D flat tuple: (2, 3, 4)
    // idx = c0 + c1*2 + c2*(2*3)
    let s_3d = int_tuple![2i32, 3i32, 4i32];
    let crd_3d = int_tuple![1i32, 2i32, 3i32];
    // 1 + 2*2 + 3*6 = 1 + 4 + 18 = 23
    assert_eq!(coord_to_index(crd_3d, &s_3d), 23i32);
    assert_eq!(index_to_coord(23i32, &s_3d), crd_3d);

    // 7. Deeply nested shape: ((2, 2), (3, 2))
    let s_deep = int_tuple![(2i32, 2i32), (3i32, 2i32)];
    assert_eq!(s_deep.size(), 24);

    // Roundtrip verification for all indices
    for i in 0..s_nested.size() {
        let crd = index_to_coord(i, &s_nested);
        assert_eq!(coord_to_index(crd, &s_nested), i);
    }
    for i in 0..s_3d.size() {
        let crd = index_to_coord(i, &s_3d);
        assert_eq!(coord_to_index(crd, &s_3d), i);
    }
    for i in 0..s_deep.size() {
        let crd = index_to_coord(i, &s_deep);
        assert_eq!(coord_to_index(crd, &s_deep), i);
    }
}

#[test]
fn test_shape_const_eval() {
    const S: TupleContainer<Int<3>, TupleContainer<Int<4>, Nil>> = int_tuple![Int::<3>, Int::<4>];
    const SZ: int = S.size();
    assert_eq!(SZ, 12);

    const CRD: TupleContainer<int, TupleContainer<int, Nil>> = int_tuple![1i32, 2i32];
    const IDX: int = coord_to_index(CRD, &S);
    assert_eq!(IDX, 7);

    const CRD_FROM_IDX: TupleContainer<int, TupleContainer<int, Nil>> = index_to_coord(7i32, &S);
    assert_eq!(CRD_FROM_IDX, CRD);

    const COMPAT_MATCH: bool = compatible(&12i32, &S);
    const COMPAT_MISMATCH: bool = compatible(&10i32, &S);
    const { assert!(COMPAT_MATCH) };
    const { assert!(!COMPAT_MISMATCH) };
}

#[test]
fn pycute_htuple_idx2crd_in_bounds_out_of_bounds_and_round_trip() {
    // In bounds (test_htuple.py::TestHTuple::test_idx2crd)
    assert_eq!(idx2crd(7i32, &14i32), 7i32);
    assert_eq!(
        idx2crd(7i32, &int_tuple![3i32, 2i32, 4i32]),
        int_tuple![1i32, 0i32, 1i32]
    );
    assert_eq!(
        idx2crd(7i32, &int_tuple![3i32, (2i32, 4i32)]),
        int_tuple![1i32, (0i32, 1i32)]
    );
    assert_eq!(
        idx2crd(7i32, &int_tuple![(3i32, 2i32), 4i32]),
        int_tuple![(1i32, 0i32), 1i32]
    );

    // Out of bounds: excess absorbed by the last leaf
    assert_eq!(idx2crd(7i32, &5i32), 7i32);
    assert_eq!(
        idx2crd(42i32, &int_tuple![3i32, 7i32, 2i32]),
        int_tuple![0i32, 0i32, 2i32]
    );
    assert_eq!(
        idx2crd(42i32, &int_tuple![3i32, 7i32, 2i32, 1i32]),
        int_tuple![0i32, 0i32, 0i32, 1i32]
    );
    assert_eq!(
        idx2crd(42i32, &int_tuple![3i32, 7i32, (2i32, 1i32)]),
        int_tuple![0i32, 0i32, (0i32, 1i32)]
    );

    // General round-trip beyond shape size
    let shape = int_tuple![(2i32, 3i32), (2i32, 1i32, 2i32)];
    for idx in 0..7 * shape.size() {
        assert_eq!(crd2idx(idx, &shape), idx);
        let natural = idx2crd(idx, &shape);
        assert_eq!(crd2idx(natural, &shape), idx);
    }

    // Colexicographical coordinate sequence (test_htuple.py::TestHTuple::test_coordinates)
    let coords_6: Vec<_> = (0..6i32.size()).map(|i| idx2crd(i, &6i32)).collect();
    assert_eq!(coords_6, vec![0, 1, 2, 3, 4, 5]);

    let s32 = int_tuple![3i32, 2i32];
    let coords_32: Vec<_> = (0..s32.size()).map(|i| idx2crd(i, &s32)).collect();
    assert_eq!(
        coords_32,
        vec![
            int_tuple![0i32, 0i32],
            int_tuple![1i32, 0i32],
            int_tuple![2i32, 0i32],
            int_tuple![0i32, 1i32],
            int_tuple![1i32, 1i32],
            int_tuple![2i32, 1i32],
        ]
    );

    let s222 = int_tuple![2i32, (2i32, 2i32)];
    let coords_222: Vec<_> = (0..s222.size()).map(|i| idx2crd(i, &s222)).collect();
    assert_eq!(
        coords_222,
        vec![
            int_tuple![0i32, (0i32, 0i32)],
            int_tuple![1i32, (0i32, 0i32)],
            int_tuple![0i32, (1i32, 0i32)],
            int_tuple![1i32, (1i32, 0i32)],
            int_tuple![0i32, (0i32, 1i32)],
            int_tuple![1i32, (0i32, 1i32)],
            int_tuple![0i32, (1i32, 1i32)],
            int_tuple![1i32, (1i32, 1i32)],
        ]
    );
}

#[test]
fn free_functions_size_has_zero_and_crd2idx_mixed_and_basis() {
    assert_eq!(size(&Nil), 1);
    assert!(!has_zero(&Nil));

    let nested = Til(Int::<2>, Til(Til(3i32, Til(Int::<5>, Nil)), Nil));
    assert_eq!(size(&nested), 30);
    assert!(!has_zero(&nested));

    let with_zero = Til(0i32, Til(Til(i32::MAX, Til(i32::MAX, Nil)), Nil));
    assert!(has_zero(&with_zero));
    assert_eq!(size(&with_zero), 0);
    assert!(std::panic::catch_unwind(|| size(&Til(65536i32, Til(65536i32, Nil)))).is_err());

    // Partial/sub-index coordinate: 2 + 5 * 3 == 17
    assert_eq!(
        crd2idx(int_tuple![2i32, 5i32], &int_tuple![3i32, (2i32, 3i32)]),
        17i32
    );

    // Nil rank strictness and basis coordinates
    assert_eq!(crd2idx(Nil, &Nil), Int::<0>);
    assert_eq!(crd2idx(BasisCombination::<Nil>::zero(), &Nil), Int::<0>);
    assert_eq!(idx2crd(BasisCombination::<Nil>::zero(), &Nil), Nil);

    let shape = int_tuple![(2i32, 3i32), 4i32];
    assert_eq!(
        idx2crd(e![0] * 5i32, &shape),
        int_tuple![(1i32, 2i32), Int::<0>]
    );
    assert_eq!(crd2idx(e![0] * 5i32, &shape), 5i32);

    assert_eq!(
        idx2crd(e![0, 1] * 2i32, &shape),
        int_tuple![(Int::<0>, 2i32), Int::<0>]
    );
    assert_eq!(crd2idx(e![0, 1] * 2i32, &shape), 4i32);

    let comb = e![0] * 5i32 + e![1] * 2i32;
    assert_eq!(idx2crd(comb, &shape), int_tuple![(1i32, 2i32), 2i32]);
    assert_eq!(crd2idx(comb, &shape), 17i32);

    let nested_basis = basis_tuple![e![1] * 2i32, 1i32];
    assert_eq!(
        crd2idx(nested_basis, &int_tuple![(2i32, 3i32), 4i32]),
        10i32
    );

    const _: () = {
        assert!(size(&Til(Int::<3>, Til(Int::<4>, Nil))) == 12);
        assert!(has_zero(&Til(Int::<0>, Til(Int::<4>, Nil))));
    };
}

#[test]
fn common_refinement_cases() {
    fn assert_refines<
        A: noodle_cute::shape::Compatible<B> + noodle_cute::shape::Shape,
        B: noodle_cute::shape::Shape,
    >(
        a: &A,
        b: &B,
    ) {
        assert!(compatible(a, b));
    }

    // Nil and scalars
    assert_eq!(common_refinement(&Nil, &Nil), Nil);
    assert_eq!(common_refinement(&Int::<1>, &Int::<1>), Int::<1>);
    assert_eq!(common_refinement(&30i32, &30i32), 30i32);
    assert_eq!(common_refinement(&Int::<30>, &30i32), Int::<30>);
    assert_eq!(common_refinement(&30i32, &Int::<30>), Int::<30>);

    // Integer vs tuple of matching size (and symmetry)
    let t2_5 = int_tuple![2i32, 5i32];
    assert_eq!(common_refinement(&10i32, &t2_5), t2_5);
    assert_eq!(common_refinement(&t2_5, &10i32), t2_5);
    assert_refines(&10i32, &t2_5);

    let t2_3_5 = int_tuple![2i32, (3i32, 5i32)];
    assert_eq!(common_refinement(&30i32, &t2_3_5), t2_3_5);
    assert_eq!(common_refinement(&t2_3_5, &30i32), t2_3_5);

    let t22_6 = int_tuple![(2i32, 2i32), 6i32];
    assert_eq!(common_refinement(&24i32, &t22_6), t22_6);
    assert_eq!(common_refinement(&t22_6, &24i32), t22_6);

    let t10_single = tuple![10i32];
    assert_eq!(common_refinement(&10i32, &t10_single), t10_single);
    assert_eq!(common_refinement(&t10_single, &10i32), t10_single);

    // Reflexive
    let s1 = int_tuple![3i32, 4i32];
    let s2 = int_tuple![(1i32, 2i32), 3i32];
    let s3 = int_tuple![(1i32, (2i32, 3i32)), 4i32];
    assert_eq!(common_refinement(&7i32, &7i32), 7i32);
    assert_eq!(common_refinement(&s1, &s1), s1);
    assert_eq!(common_refinement(&s2, &s2), s2);
    assert_eq!(common_refinement(&s3, &s3), s3);
    assert_eq!(common_refinement(&t2_3_5, &t2_3_5), t2_3_5);

    // Whitepaper chains
    let t2_15 = int_tuple![2i32, 15i32];
    assert_eq!(common_refinement(&30i32, &t2_15), t2_15);
    assert_eq!(common_refinement(&t2_15, &t2_3_5), t2_3_5);
    assert_eq!(common_refinement(&t2_3_5, &t2_15), t2_3_5);

    let t6_5 = int_tuple![6i32, 5i32];
    let t32_5 = int_tuple![(3i32, 2i32), 5i32];
    assert_eq!(common_refinement(&30i32, &t6_5), t6_5);
    assert_eq!(common_refinement(&t6_5, &t32_5), t32_5);
    assert_eq!(common_refinement(&t32_5, &t6_5), t32_5);

    // Mixed modes and deeper nesting
    let a = int_tuple![4i32, (3i32, 5i32)];
    let b = int_tuple![(2i32, 2i32), 15i32];
    let c = common_refinement(&a, &b);
    assert_eq!(c, int_tuple![(2i32, 2i32), (3i32, 5i32)]);
    assert_refines(&a, &c);
    assert_refines(&b, &c);
    assert_eq!(common_refinement(&b, &a), c);

    let d1 = int_tuple![2i32, (6i32, 5i32)];
    let d2 = int_tuple![2i32, ((3i32, 2i32), 5i32)];
    assert_eq!(common_refinement(&d1, &d2), d2);

    let j1 = int_tuple![(2i32, 3i32), 20i32];
    let j2 = int_tuple![6i32, (4i32, 5i32)];
    let jr = common_refinement(&j1, &j2);
    assert_eq!(jr, int_tuple![(2i32, 3i32), (4i32, 5i32)]);
    assert_refines(&j1, &jr);
    assert_refines(&j2, &jr);

    // Non-existence cases panic at runtime for dynamic inputs
    assert!(std::panic::catch_unwind(|| common_refinement(&3i32, &4i32)).is_err());
    assert!(
        std::panic::catch_unwind(|| common_refinement(&7i32, &int_tuple![2i32, 3i32])).is_err()
    );
    assert!(
        std::panic::catch_unwind(|| common_refinement(&int_tuple![2i32, 3i32], &7i32)).is_err()
    );
    assert!(
        std::panic::catch_unwind(|| common_refinement(
            &int_tuple![2i32, 3i32],
            &int_tuple![2i32, 3i32, 1i32]
        ))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| common_refinement(&tuple![6i32], &int_tuple![2i32, 3i32]))
            .is_err()
    );
    assert!(std::panic::catch_unwind(|| common_refinement(&t2_3_5, &t32_5)).is_err());
    assert!(std::panic::catch_unwind(|| common_refinement(&t32_5, &t2_3_5)).is_err());
    assert!(
        std::panic::catch_unwind(|| common_refinement(
            &int_tuple![2i32, 15i32],
            &int_tuple![3i32, 10i32]
        ))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| common_refinement(
            &int_tuple![4i32, (3i32, 5i32)],
            &int_tuple![4i32, (2i32, 5i32)]
        ))
        .is_err()
    );
    assert!(std::panic::catch_unwind(|| common_refinement(&Nil, &int_tuple![2i32, 3i32])).is_err());
    assert!(std::panic::catch_unwind(|| common_refinement(&int_tuple![2i32, 3i32], &Nil)).is_err());

    // Static const eval
    const _: () = {
        let sa = Til(Int::<4>, Til(Til(Int::<3>, Til(Int::<5>, Nil)), Nil));
        let sb = Til(Til(Int::<2>, Til(Int::<2>, Nil)), Til(Int::<15>, Nil));
        let sc = common_refinement(&sa, &sb);
        assert!(size(&sc) == 60);
    };
}

#[test]
fn common_coarsening_and_duality_cases() {
    // Nil & scalars
    assert_eq!(common_coarsening(&Nil, &Nil), Nil);
    assert_eq!(common_coarsening(&Int::<1>, &Int::<1>), Int::<1>);
    assert_eq!(common_coarsening(&30i32, &30i32), 30i32);
    assert_eq!(common_coarsening(&Int::<30>, &30i32), Int::<30>);
    assert_eq!(common_coarsening(&30i32, &Int::<30>), Int::<30>);

    // Integer forces scalar meet
    let t2_5 = int_tuple![2i32, 5i32];
    assert_eq!(common_coarsening(&10i32, &t2_5), 10i32);
    assert_eq!(common_coarsening(&t2_5, &10i32), 10i32);

    let t2_3_5 = int_tuple![2i32, (3i32, 5i32)];
    assert_eq!(common_coarsening(&30i32, &t2_3_5), 30i32);
    assert_eq!(common_coarsening(&t2_3_5, &30i32), 30i32);

    let t22_6 = int_tuple![(2i32, 2i32), 6i32];
    assert_eq!(common_coarsening(&24i32, &t22_6), 24i32);
    assert_eq!(common_coarsening(&t22_6, &24i32), 24i32);

    let t10_single = tuple![10i32];
    assert_eq!(common_coarsening(&10i32, &t10_single), 10i32);
    assert_eq!(common_coarsening(&t10_single, &10i32), 10i32);

    // Reflexive
    let s1 = int_tuple![3i32, 4i32];
    let s2 = int_tuple![(1i32, 2i32), 3i32];
    let s3 = int_tuple![(1i32, (2i32, 3i32)), 4i32];
    assert_eq!(common_coarsening(&s1, &s1), s1);
    assert_eq!(common_coarsening(&s2, &s2), s2);
    assert_eq!(common_coarsening(&s3, &s3), s3);

    // Whitepaper examples (including incompatible profiles with equal size)
    let t2_15 = int_tuple![2i32, 15i32];
    let t32_5 = int_tuple![(3i32, 2i32), 5i32];
    assert_eq!(common_coarsening(&t2_15, &t2_3_5), t2_15);
    assert_eq!(common_coarsening(&t2_3_5, &t2_15), t2_15);
    // Dynamic equal-rank tuples with incompatible mode sizes panic (type commits to tuple)
    assert!(std::panic::catch_unwind(|| common_coarsening(&t2_3_5, &t32_5)).is_err());
    assert!(std::panic::catch_unwind(|| common_coarsening(&t32_5, &t2_3_5)).is_err());

    let st2_3_5 = Til(Int::<2>, Til(Til(Int::<3>, Til(Int::<5>, Nil)), Nil));
    let st32_5 = Til(Til(Int::<3>, Til(Int::<2>, Nil)), Til(Int::<5>, Nil));
    assert_eq!(common_coarsening(&st2_3_5, &st32_5), Int::<30>);
    assert_eq!(common_coarsening(&st32_5, &st2_3_5), Int::<30>);

    // Mode mismatch on static shapes falls back to top-level static total size
    assert_eq!(
        common_coarsening(&tuple![Int::<6>, Int::<5>], &tuple![Int::<2>, Int::<15>]),
        Int::<30>
    );
    assert_eq!(
        common_coarsening(&tuple![Int::<6>, Int::<5>], &tuple![Int::<3>, Int::<10>]),
        Int::<30>
    );
    assert_eq!(
        common_coarsening(&tuple![Int::<4>, Int::<6>], &tuple![Int::<3>, Int::<8>]),
        Int::<24>
    );
    assert!(
        std::panic::catch_unwind(|| common_coarsening(
            &int_tuple![6i32, 5i32],
            &int_tuple![2i32, 15i32]
        ))
        .is_err()
    );

    // Partial tuple meet (dynamic matching-mode & static fallback per-mode)
    assert_eq!(
        common_coarsening(
            &int_tuple![4i32, (3i32, 5i32)],
            &int_tuple![(2i32, 2i32), 15i32]
        ),
        int_tuple![4i32, 15i32]
    );
    assert_eq!(
        common_coarsening(
            &tuple![(Int::<2>, Int::<3>), Int::<4>],
            &tuple![(Int::<3>, Int::<2>), Int::<4>]
        ),
        tuple![Int::<6>, Int::<4>]
    );
    assert!(
        std::panic::catch_unwind(|| common_coarsening(
            &int_tuple![(2i32, 3i32), 4i32],
            &int_tuple![(3i32, 2i32), 4i32]
        ))
        .is_err()
    );

    // Rank mismatch falls back to scalar meet
    assert_eq!(
        common_coarsening(&int_tuple![2i32, 3i32], &int_tuple![2i32, 3i32, 1i32]),
        6i32
    );
    assert_eq!(common_coarsening(&tuple![6i32], &6i32), 6i32);
    assert_eq!(
        common_coarsening(&int_tuple![5i32, 3i32, 4i32], &int_tuple![10i32, 6i32]),
        60i32
    );

    // Non-existence cases (size mismatch)
    assert!(std::panic::catch_unwind(|| common_coarsening(&3i32, &4i32)).is_err());
    assert!(
        std::panic::catch_unwind(|| common_coarsening(&7i32, &int_tuple![2i32, 3i32])).is_err()
    );
    assert!(
        std::panic::catch_unwind(|| common_coarsening(&int_tuple![2i32, 3i32], &7i32)).is_err()
    );
    assert!(
        std::panic::catch_unwind(|| common_coarsening(
            &int_tuple![2i32, 3i32],
            &int_tuple![4i32, 5i32]
        ))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| common_coarsening(
            &int_tuple![2i32, 3i32, 4i32],
            &int_tuple![6i32, 5i32]
        ))
        .is_err()
    );

    // Duality checks: meet <= join, meet <= a, meet <= b, a <= join, b <= join
    let c1 = common_coarsening(&30i32, &t2_15);
    let r1 = common_refinement(&30i32, &t2_15);
    assert!(compatible(&c1, &r1) && compatible(&c1, &30i32) && compatible(&c1, &t2_15));
    assert!(compatible(&30i32, &r1) && compatible(&t2_15, &r1));

    let p2a = int_tuple![4i32, (3i32, 5i32)];
    let p2b = int_tuple![(2i32, 2i32), 15i32];
    let c2 = common_coarsening(&p2a, &p2b);
    let r2 = common_refinement(&p2a, &p2b);
    assert!(compatible(&c2, &r2) && compatible(&c2, &p2a) && compatible(&c2, &p2b));
    assert!(compatible(&p2a, &r2) && compatible(&p2b, &r2));

    let p3a = int_tuple![(2i32, 3i32), 20i32];
    let p3b = int_tuple![6i32, (4i32, 5i32)];
    let c3 = common_coarsening(&p3a, &p3b);
    let r3 = common_refinement(&p3a, &p3b);
    assert!(compatible(&c3, &r3) && compatible(&c3, &p3a) && compatible(&c3, &p3b));
    assert!(compatible(&p3a, &r3) && compatible(&p3b, &r3));

    const _: () = {
        let sa = Til(Int::<2>, Til(Til(Int::<3>, Til(Int::<5>, Nil)), Nil));
        let sb = Til(Til(Int::<3>, Til(Int::<2>, Nil)), Til(Int::<5>, Nil));
        let mc = common_coarsening(&sa, &sb);
        assert!(size(&mc) == 30);
    };
}

#[test]
fn in_bounds_and_coordinates_cases() {
    let s34 = int_tuple![3i32, 4i32];
    assert!(in_bounds(int_tuple![1i32, 2i32], &s34));
    assert!(in_bounds(0i32, &s34));
    assert!(in_bounds(5i32, &s34));
    assert!(in_bounds(11i32, &s34));
    assert!(!in_bounds(12i32, &s34));
    assert!(!in_bounds(int_tuple![3i32, 0i32], &s34));
    assert!(!in_bounds(int_tuple![0i32, 4i32], &s34));

    // Negative coordinates
    assert!(!in_bounds(-1i32, &4i32));
    assert!(!in_bounds(-1i32, &s34));
    assert!(!in_bounds(int_tuple![-1i32, 0i32], &s34));
    assert!(!in_bounds(int_tuple![0i32, -1i32], &s34));

    // Nested and partial sub-index coordinates
    let s3_2_6 = int_tuple![3i32, (2i32, 6i32)];
    assert!(in_bounds(int_tuple![2i32, (1i32, 5i32)], &s3_2_6));
    assert!(in_bounds(int_tuple![0i32, 11i32], &s3_2_6));
    assert!(!in_bounds(int_tuple![0i32, 12i32], &s3_2_6));

    // Zero-extent shapes always return false
    assert!(!in_bounds(0i32, &0i32));
    assert!(!in_bounds(int_tuple![0i32, 0i32], &int_tuple![0i32, 4i32]));
    assert!(!in_bounds(int_tuple![0i32, 0i32], &int_tuple![4i32, 0i32]));
    assert!(!in_bounds(
        int_tuple![0i32, 0i32],
        &int_tuple![(4i32, 0i32), 2i32]
    ));

    // Nil and nested Nil (matches PyCuTe: empty leaf sets are vacuously in bounds)
    assert!(in_bounds(Nil, &Nil));
    assert!(in_bounds(0i32, &Nil));
    assert!(in_bounds(1i32, &Nil));
    assert!(in_bounds(BasisCombination::<Nil>::zero(), &Nil));
    assert!(in_bounds(tuple![1i32, Nil], &tuple![Int::<2>, Nil]));
    assert!(!in_bounds(tuple![2i32, Nil], &tuple![Int::<2>, Nil]));

    // Basis coordinates
    assert!(in_bounds(e![0] * 2i32, &s34));
    assert!(!in_bounds(e![0] * 3i32, &s34));
    assert!(in_bounds(e![1] * 2i32 + e![0] * 1i32, &s34));
    assert!(!in_bounds(e![1] * 4i32 + e![0] * 1i32, &s34));
    assert!(in_bounds(
        basis_tuple![1i32, e![1] * 2i32],
        &int_tuple![3i32, (2i32, 3i32)]
    ));
    assert!(!in_bounds(
        basis_tuple![1i32, e![1] * 3i32],
        &int_tuple![3i32, (2i32, 3i32)]
    ));

    // Const eval of in_bounds
    const _: () = {
        let s = Til(Int::<3>, Til(Int::<4>, Nil));
        assert!(in_bounds(Til(Int::<2>, Til(Int::<3>, Nil)), &s));
        assert!(!in_bounds(Til(Int::<3>, Til(Int::<0>, Nil)), &s));
        assert!(in_bounds(Int::<11>, &s));
        assert!(!in_bounds(Int::<12>, &s));
        assert!(!in_bounds(Int::<0>, &Til(Int::<0>, Til(Int::<4>, Nil))));
    };

    // coordinates() iterator & ExactSizeIterator
    let mut it_nil = coordinates(Nil);
    assert_eq!(it_nil.len(), 1);
    assert_eq!(it_nil.next(), Some(Nil));
    assert_eq!(it_nil.len(), 0);
    assert_eq!(it_nil.next(), None);

    let mut it_0 = coordinates(0i32);
    assert_eq!(it_0.len(), 0);
    assert_eq!(it_0.next(), None);

    let it_zero_tup = coordinates(int_tuple![3i32, 0i32]);
    assert_eq!(it_zero_tup.len(), 0);
    assert!(it_zero_tup.collect::<Vec<_>>().is_empty());

    let it_scalar = coordinates(4i32);
    assert_eq!(it_scalar.len(), 4);
    assert_eq!(it_scalar.collect::<Vec<_>>(), vec![0i32, 1i32, 2i32, 3i32]);

    let mut it_32 = coordinates(int_tuple![3i32, 2i32]);
    assert_eq!(it_32.len(), 6);
    assert_eq!(it_32.next(), Some(int_tuple![0i32, 0i32]));
    assert_eq!(it_32.len(), 5);
    let mut rest_32 = vec![int_tuple![0i32, 0i32]];
    rest_32.extend(it_32);
    assert_eq!(
        rest_32,
        vec![
            int_tuple![0i32, 0i32],
            int_tuple![1i32, 0i32],
            int_tuple![2i32, 0i32],
            int_tuple![0i32, 1i32],
            int_tuple![1i32, 1i32],
            int_tuple![2i32, 1i32],
        ]
    );

    let coords_222: Vec<_> = coordinates(int_tuple![2i32, (2i32, 2i32)]).collect();
    assert_eq!(
        coords_222,
        vec![
            int_tuple![0i32, (0i32, 0i32)],
            int_tuple![1i32, (0i32, 0i32)],
            int_tuple![0i32, (1i32, 0i32)],
            int_tuple![1i32, (1i32, 0i32)],
            int_tuple![0i32, (0i32, 1i32)],
            int_tuple![1i32, (0i32, 1i32)],
            int_tuple![0i32, (1i32, 1i32)],
            int_tuple![1i32, (1i32, 1i32)],
        ]
    );
}

#[test]
fn shape_regression_trailing_nil_floor_divmod_signed_crd2idx_and_refinement() {
    // 1. Trailing leaf-less tail skips mod (T::LEAVES == 0), matching PyCuTe
    let s_4_nil = tuple![4i32, Nil];
    assert_eq!(idx2crd(5i32, &s_4_nil), tuple![5i32, Nil]);
    assert!(!in_bounds(5i32, &s_4_nil));
    assert!(in_bounds(3i32, &s_4_nil));

    let s_nested_tail_nil = tuple![(2i32, 3i32), (Nil, Nil)];
    assert_eq!(
        idx2crd(7i32, &s_nested_tail_nil),
        tuple![(1i32, 3i32), (Nil, Nil)]
    );
    assert!(!in_bounds(7i32, &s_nested_tail_nil));

    // Empty leaf sets are vacuously in bounds (matching PyCuTe)
    assert!(in_bounds(100i32, &Nil));
    assert!(in_bounds(100i32, &tuple![Nil, Nil]));
    assert!(in_bounds(tuple![1i32, 2i32], &s_4_nil));

    // 2. Floor division semantics for negative indices & no wrap/overflow on large extents
    assert_eq!(
        idx2crd(-1i32, &int_tuple![3i32, 4i32]),
        int_tuple![2i32, -1i32]
    );
    assert_eq!(
        idx2crd(10i32, &int_tuple![i32::MAX, 2i32]),
        int_tuple![10i32, 0i32]
    );
    assert!(in_bounds(10i32, &int_tuple![i32::MAX, 2i32]));
    assert!(!in_bounds(-1i32, &int_tuple![i32::MAX, 2i32]));

    // Leaf-by-leaf divmod avoids subtree size overflow on ((65536, 65536), 4)
    assert_eq!(
        idx2crd(5i32, &int_tuple![(65536i32, 65536i32), 4i32]),
        int_tuple![(5i32, 0i32), 0i32]
    );

    // Static idx2crd and crd2idx preserve static types
    type InnerCrd = TupleContainer<Int<1>, TupleContainer<Int<2>, Nil>>;
    const STATIC_CRD: TupleContainer<Int<2>, TupleContainer<InnerCrd, Nil>> =
        idx2crd(Int::<17>, &int_tuple![Int::<3>, (Int::<2>, Int::<4>)]);
    assert_eq!(STATIC_CRD, int_tuple![Int::<2>, (Int::<1>, Int::<2>)]);
    const STATIC_IDX: Int<17> = crd2idx(STATIC_CRD, &int_tuple![Int::<3>, (Int::<2>, Int::<4>)]);
    assert_eq!(STATIC_IDX, Int::<17>);

    // 3. Signed negative coordinates in crd2idx do not wrap to uint or panic
    assert_eq!(crd2idx(-1i32, &4i32), -1i32);
    assert_eq!(
        crd2idx(int_tuple![-1i32, 1i32], &int_tuple![3i32, 4i32]),
        2i32
    );
    assert_eq!(
        crd2idx(
            idx2crd(-1i32, &int_tuple![3i32, 4i32]),
            &int_tuple![3i32, 4i32]
        ),
        -1i32
    );

    // 4. CommonRefinement satisfies lattice axiom for (1, Nil) / (Nil, 1)
    assert!(compatible(&Int::<1>, &Nil));
    assert_eq!(common_refinement(&Int::<1>, &Nil), Nil);
    assert_eq!(common_refinement(&Nil, &Int::<1>), Nil);
    assert_eq!(common_refinement(&1i32, &Nil), Nil);
    assert_eq!(common_refinement(&Nil, &1i32), Nil);
    assert!(std::panic::catch_unwind(|| common_refinement(&2i32, &Nil)).is_err());
    assert!(std::panic::catch_unwind(|| common_refinement(&Nil, &2i32)).is_err());

    // 5. Negative dynamic extents panic across shape operations
    assert!(std::panic::catch_unwind(|| compatible(&-1i32, &4i32)).is_err());
    assert!(std::panic::catch_unwind(|| crd2idx(0i32, &-1i32)).is_err());
    assert!(std::panic::catch_unwind(|| idx2crd(0i32, &-1i32)).is_err());
    assert!(std::panic::catch_unwind(|| idx2crd(0i32, &int_tuple![-1i32, 4i32])).is_err());
    assert!(std::panic::catch_unwind(|| in_bounds(0i32, &-1i32)).is_err());
    assert!(std::panic::catch_unwind(|| in_bounds(0i32, &int_tuple![-1i32, 4i32])).is_err());
    assert!(std::panic::catch_unwind(|| coordinates(-1i32)).is_err());
}
