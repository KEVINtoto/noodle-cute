//! CuTe composition 的规范化、惰性回退与边界契约。

use noodle_cute::algebra::complement::complement_extended;
use noodle_cute::algebra::composition::{Composition, composition, composition_layout};
use noodle_cute::layout::{AffineLayoutMap, DynLayout, GroupedLayout, Layout, LayoutMap};
use noodle_cute::{DynInt, Error, Int, IntTuple, Nil, shape, stride};

mod common;
use common::{TableMap, assert_grouped, dyn_layout, grouped, values};

type Pair<A, B> = IntTuple<A, IntTuple<B, Nil>>;
type Row = Layout<Pair<Int<4>, Int<6>>, Pair<Int<6>, Int<1>>>;

const ROW: Row = Layout {
    shape: shape![Int::<4>, Int::<6>],
    stride: stride![Int::<6>, Int::<1>],
};

#[test]
fn mixed_integral_composition_preserves_nested_rhs_and_static_leaves() {
    let lhs = Layout::new(DynInt(8), Int::<2>).unwrap();
    let rhs = Layout::new(
        shape![(Int::<2>, DynInt(3)), Int::<1>],
        stride![(Int::<1>, DynInt(4)), Int::<0>],
    )
    .unwrap();
    type Output = Layout<Pair<Pair<Int<2>, DynInt>, Int<1>>, Pair<Pair<Int<2>, DynInt>, Int<0>>>;
    let result: Output = lhs.composition(rhs).unwrap();
    assert_eq!(result.shape, rhs.shape);
    assert_eq!(result.stride.0.1.0, DynInt(8));
    assert_eq!(values(&result), [0, 2, 8, 10, 16, 18]);

    let zero = Layout::new(DynInt(8), Int::<0>).unwrap();
    let result: Layout<Int<1>, Int<0>> = zero
        .composition(Layout::new(Int::<1>, DynInt(isize::MIN)).unwrap())
        .unwrap();
    assert_eq!(values(&result), [0]);
}

#[test]
fn mixed_weak_divisibility_retains_runtime_unit_modes() {
    for (n, expected_shape) in [(1, [1, 1]), (2, [2, 1]), (8, [2, 4])] {
        type Output = Layout<Pair<DynInt, DynInt>, Pair<Int<18>, Int<1>>>;
        let result: Output = ROW
            .composition(Layout::new(DynInt(n), Int::<3>).unwrap())
            .unwrap();
        assert_eq!(result.as_flat_layout().unwrap().shapes(), &expected_shape);
        assert_eq!(result.rank(), 2);
        assert_eq!(values(&result), [0, 18, 1, 19, 2, 20, 3, 21][..n as usize]);
    }
}

#[test]
fn mixed_lhs_metadata_only_promotes_dependent_result_leaves() {
    let dynamic_shape =
        Layout::new(shape![DynInt(4), Int::<6>], stride![Int::<6>, Int::<1>]).unwrap();
    let a: Layout<Pair<DynInt, DynInt>, Pair<Int<18>, DynInt>> = dynamic_shape
        .composition(Layout::new(Int::<8>, Int::<3>).unwrap())
        .unwrap();
    assert_eq!(values(&a), [0, 18, 1, 19, 2, 20, 3, 21]);

    let dynamic_stride =
        Layout::new(shape![Int::<4>, Int::<6>], stride![DynInt(6), Int::<1>]).unwrap();
    let b: Layout<Pair<Int<2>, Int<4>>, Pair<DynInt, Int<1>>> = dynamic_stride
        .composition(Layout::new(Int::<8>, Int::<3>).unwrap())
        .unwrap();
    assert_eq!(values(&b), [0, 18, 1, 19, 2, 20, 3, 21]);

    let native = Layout::new(shape![4isize, 6], stride![6isize, 1]).unwrap();
    let c: Layout<Pair<DynInt, DynInt>, Pair<DynInt, DynInt>> = native
        .composition(Layout::new(8isize, 3isize).unwrap())
        .unwrap();
    assert_eq!(values(&c), [0, 18, 1, 19, 2, 20, 3, 21]);
}

#[test]
fn mixed_weak_divisibility_uses_type_known_coalesce_boundaries() {
    let lhs = Layout::new(shape![DynInt(4), DynInt(6)], stride![DynInt(1), DynInt(4)]).unwrap();
    let rhs = Layout::new(Int::<8>, Int::<3>).unwrap();
    let typed = lhs.composition(rhs).unwrap();
    assert_eq!(values(&typed), [0, 3, 4, 7, 8, 11, 12, 15]);
    let runtime = lhs.as_flat_layout().unwrap().composition(&rhs).unwrap();
    assert_eq!(values(&runtime), [0, 3, 6, 9, 12, 15, 18, 21]);
}

#[test]
fn mixed_composition_checks_dynamic_divisibility_shapes_and_overflow() {
    for (s, d, error) in [
        (8, 5, Error::NotDivisible),
        (3, 3, Error::NotDivisible),
        (0, 1, Error::InvalidShape),
        (5, 0, Error::InvalidShape),
        (1, isize::MIN, Error::Overflow),
    ] {
        assert_eq!(
            ROW.composition(Layout::new(DynInt(s), DynInt(d)).unwrap())
                .err(),
            Some(error)
        );
    }
    assert_eq!(
        ROW.composition(Layout {
            shape: DynInt(-1),
            stride: Int::<1>
        })
        .err(),
        Some(Error::InvalidShape)
    );
    let large_stride = Layout::new(Int::<1>, DynInt(isize::MAX)).unwrap();
    assert_eq!(
        large_stride
            .composition(Layout::new(Int::<2>, Int::<2>).unwrap())
            .err(),
        Some(Error::Overflow)
    );
}

#[test]
fn mixed_composition_covers_broadcast_empty_tuples_and_static_zero_shape() {
    let empty_lhs = Layout::new(shape![DynInt(0), Int::<6>], stride![DynInt(6), Int::<1>]).unwrap();
    let broadcast: Layout<DynInt, Int<0>> = empty_lhs
        .composition(Layout::new(DynInt(5), Int::<0>).unwrap())
        .unwrap();
    assert_eq!(values(&broadcast), [0; 5]);

    let empty_rhs: Layout<Nil, Nil> = empty_lhs
        .composition(Layout::new(Nil, Nil).unwrap())
        .unwrap();
    assert_eq!(empty_rhs.size(), Ok(1));
    let scalar: Layout<DynInt, Int<0>> = Layout::new(Nil, Nil)
        .unwrap()
        .composition(Layout::new(DynInt(3), Int::<1>).unwrap())
        .unwrap();
    assert_eq!(values(&scalar), [0, 0, 0]);

    let zero_shape: Layout<Pair<DynInt, Int<0>>, Pair<DynInt, DynInt>> = ROW
        .composition(Layout::new(Int::<0>, DynInt(1)).unwrap())
        .unwrap();
    assert_eq!(zero_shape.size(), Ok(0));
    assert_eq!(zero_shape.as_flat_layout().unwrap().shapes(), &[0, 0]);
    let skipped_zero: Layout<DynInt, Int<1>> = ROW
        .composition(Layout::new(DynInt(0), Int::<4>).unwrap())
        .unwrap();
    assert_eq!(skipped_zero.size(), Ok(0));

    let runtime_zero_stride: Layout<DynInt, DynInt> = Layout::new(DynInt(8), DynInt(2))
        .unwrap()
        .composition(Layout::new(DynInt(5), DynInt(0)).unwrap())
        .unwrap();
    assert_eq!(values(&runtime_zero_stride), [0; 5]);
}

#[test]
fn mixed_composition_preserves_extended_stride_and_signed_rules() {
    let lhs = Layout::new(shape![Int::<2>, Int::<1>], stride![Int::<1>, DynInt(8)]).unwrap();
    let result: Layout<Pair<DynInt, DynInt>, Pair<Int<1>, DynInt>> = lhs
        .composition(Layout::new(DynInt(6), Int::<1>).unwrap())
        .unwrap();
    assert_eq!(values(&result), [0, 1, 8, 9, 16, 17]);

    let negative: Layout<Int<1>, DynInt> = ROW
        .composition(Layout::new(Int::<1>, DynInt(-5)).unwrap())
        .unwrap();
    assert_eq!(negative.stride, DynInt(-2));
    assert_eq!(values(&negative), [0]);

    let discarded = Layout::new(
        shape![Int::<2>, Int::<2>],
        stride![Int::<1>, DynInt(isize::MIN)],
    )
    .unwrap();
    let result: Layout<Int<2>, Int<-1>> = discarded
        .composition(Layout::new(Int::<2>, Int::<-1>).unwrap())
        .unwrap();
    assert_eq!(values(&result), [0, -1]);
}

#[test]
fn typed_composition_has_no_runtime_leaf_buffer_limit() {
    let block = shape![Int::<1>, Int::<1>, Int::<1>, Int::<1>];
    let zeros = stride![Int::<0>, Int::<0>, Int::<0>, Int::<0>];
    let rhs = Layout::new(
        shape![
            block, block, block, block, block, block, block, block, block
        ],
        stride![
            zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros
        ],
    )
    .unwrap();
    assert_eq!(rhs.as_flat_layout(), Err(Error::CapacityExceeded));
    let lhs = Layout::new(DynInt(8), Int::<1>).unwrap();
    let result = lhs.composition(rhs).unwrap();
    assert_eq!(result.rank(), 9);
    assert_eq!(values(&result), [0]);
}

#[test]
fn runtime_composition_method_keeps_rhs_grouping() {
    let lhs = dyn_layout(&[4, 6], &[6, 1]);
    let rhs = grouped(&[8], &[3], &[1]);
    let result = lhs.composition(&rhs).unwrap();
    assert_grouped(&result, &[2, 4], &[18, 1], &[2]);
    assert_eq!(values(&result), [0, 18, 1, 19, 2, 20, 3, 21]);
}

// ---------------------------------------------------------------------------
// 运行时规范化与惰性组合 (`src/algebra/composition.rs`)
// ---------------------------------------------------------------------------

#[test]
fn composition_layout_and_lazy_composition_match_cute_oracle_cases() {
    let row = dyn_layout(&[4, 6], &[6, 1]);
    let rhs_nested = grouped(&[2, 2, 3], &[1, 2, 4], &[2, 1]);

    struct Case {
        lhs: DynLayout,
        rhs: GroupedLayout,
        shape: &'static [isize],
        stride: &'static [isize],
        eager_groups: &'static [usize],
    }

    let cases = [
        Case {
            lhs: dyn_layout(&[4], &[1]),
            rhs: GroupedLayout::from_flat(dyn_layout(&[8], &[1])).unwrap(),
            shape: &[8],
            stride: &[1],
            eager_groups: &[1],
        },
        Case {
            lhs: dyn_layout(&[4], &[2]),
            rhs: GroupedLayout::from_flat(dyn_layout(&[8], &[3])).unwrap(),
            shape: &[8],
            stride: &[6],
            eager_groups: &[1],
        },
        Case {
            lhs: row,
            rhs: GroupedLayout::from_flat(dyn_layout(&[24], &[1])).unwrap(),
            shape: &[4, 6],
            stride: &[6, 1],
            eager_groups: &[1, 1],
        },
        Case {
            lhs: row,
            rhs: GroupedLayout::from_flat(dyn_layout(&[8], &[3])).unwrap(),
            shape: &[2, 4],
            stride: &[18, 1],
            eager_groups: &[1, 1],
        },
        Case {
            lhs: row,
            rhs: GroupedLayout::from_flat(dyn_layout(&[2, 3], &[2, 4])).unwrap(),
            shape: &[2, 3],
            stride: &[12, 1],
            eager_groups: &[1, 1],
        },
        Case {
            lhs: row,
            rhs: rhs_nested,
            shape: &[2, 2, 3],
            stride: &[6, 12, 1],
            eager_groups: &[2, 1],
        },
        Case {
            lhs: row,
            rhs: GroupedLayout::from_flat(dyn_layout(&[4, 6], &[0, 4])).unwrap(),
            shape: &[4, 6],
            stride: &[0, 1],
            eager_groups: &[1, 1],
        },
        Case {
            lhs: dyn_layout(&[2, 4], &[1, 10]),
            rhs: GroupedLayout::from_flat(dyn_layout(&[16], &[1])).unwrap(),
            shape: &[2, 8],
            stride: &[1, 10],
            eager_groups: &[1, 1],
        },
        Case {
            lhs: dyn_layout(&[2, 4], &[-1, 10]),
            rhs: GroupedLayout::from_flat(dyn_layout(&[2, 2], &[1, 2])).unwrap(),
            shape: &[2, 2],
            stride: &[-1, 10],
            eager_groups: &[1, 1],
        },
    ];

    for c in cases {
        let eager = composition_layout(&c.lhs, &c.rhs).unwrap();
        assert_grouped(&eager, c.shape, c.stride, c.eager_groups);

        let lazy: Composition<_, _> = composition(c.lhs, c.rhs).unwrap();
        assert!(lazy.is_normalized());
        assert_eq!(lazy.rank(), c.rhs.rank());
        assert_eq!(lazy.size(), c.rhs.size());
        for a in 0..c.rhs.rank() {
            assert_eq!(lazy.extent(a), c.rhs.extent(a));
        }
        assert_eq!(values(&lazy), values(&eager));

        let cached = lazy.as_affine().unwrap().unwrap();
        assert_eq!(lazy.index_extended(20), cached.index_extended(20));
    }

    // Independent CuTe tutorial table: (6,2):(8,2) o (4,3):(3,1).
    let tutorial = composition(dyn_layout(&[6, 2], &[8, 2]), dyn_layout(&[4, 3], &[3, 1])).unwrap();
    assert_eq!(
        values(&tutorial),
        [0, 24, 2, 26, 8, 32, 10, 34, 16, 40, 18, 42]
    );

    // Extended domain continuation for (2,4):(1,10) o 16:1.
    let extended = composition(dyn_layout(&[2, 4], &[1, 10]), dyn_layout(&[16], &[1])).unwrap();
    assert_eq!(
        values(&extended),
        [0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61, 70, 71]
    );
}

#[test]
fn composition_preserves_trailing_unit_stride_for_extended_domain() {
    // `complement_extended(4:2)` is `(2, 1):(1, 8)`. Ordinary `coalesce` would
    // drop `1:8` to `2:1`, losing the repetition stride 8; `coalesce_extended`
    // retains it so `composition` steps by 8 once past the two gap slots.
    let comp = complement_extended(dyn_layout(&[4], &[2])).unwrap();
    assert_eq!((comp.shapes(), comp.strides()), (&[2, 1][..], &[1, 8][..]));

    let eager = composition_layout(&comp, &dyn_layout(&[6], &[1])).unwrap();
    assert_grouped(&eager, &[2, 3], &[1, 8], &[1, 1]);
    assert_eq!(values(&eager), [0, 1, 8, 9, 16, 17]);

    let lazy = composition(comp, dyn_layout(&[6], &[1])).unwrap();
    assert_eq!(values(&lazy), [0, 1, 8, 9, 16, 17]);
}

#[test]
fn composition_distinguishes_integral_shapes_from_single_mode_tuples() {
    let row = dyn_layout(&[4, 6], &[6, 1]);

    // An integral shape (Int<8>) expands its leaves into top-level modes in
    // `composition_layout`, while `composition` merges them to match inner's rank 1.
    let integral_rhs = Layout::new(Int::<8>, Int::<3>).unwrap();
    let eager_int = composition_layout(&row, &integral_rhs).unwrap();
    assert_grouped(&eager_int, &[2, 4], &[18, 1], &[1, 1]);
    assert_eq!(eager_int.rank(), 2);

    let lazy_int = composition(row, integral_rhs).unwrap();
    assert_eq!(lazy_int.rank(), 1);
    assert_eq!(lazy_int.extent(0), Ok(8));
    assert_grouped(
        &lazy_int.as_affine().unwrap().unwrap(),
        &[2, 4],
        &[18, 1],
        &[2],
    );

    // A 1-element tuple `(Int<8>,)` keeps all output leaves inside mode 0 in both APIs.
    let tuple_rhs = Layout::new(shape![Int::<8>], stride![Int::<3>]).unwrap();
    let eager_tuple = composition_layout(&row, &tuple_rhs).unwrap();
    assert_grouped(&eager_tuple, &[2, 4], &[18, 1], &[2]);
    assert_eq!(eager_tuple.rank(), 1);

    // Empty 0-leaf tuple modes in GroupedLayout remain intact through right-distributivity.
    let with_empty_mode = GroupedLayout::new(dyn_layout(&[8], &[2]), &[0, 1]).unwrap();
    let eager_empty_mode = composition_layout(&row, &with_empty_mode).unwrap();
    assert_grouped(&eager_empty_mode, &[2, 4], &[12, 1], &[0, 2]);
}

#[test]
fn composition_weak_divisibility_follows_cute_while_not_divisible_falls_back_pointwise() {
    let row = dyn_layout(&[4, 6], &[6, 1]);

    // Weak divisibility: 8:3 steps by 3 across extent 4, restarting at the next
    // column in CuTe ((2,4):(18,1) gives 1 at i=2) rather than carrying (row(6) == 13).
    let weak = composition(row, dyn_layout(&[8], &[3])).unwrap();
    assert!(weak.is_normalized());
    assert_eq!(weak.index(2), Ok(1));
    assert_eq!(row.index_extended(6), Ok(13));
    assert_eq!(values(&weak), [0, 18, 1, 19, 2, 20, 3, 21]);

    // Under weak divisibility, `composition` normalizes before probing `outer.index_extended(max)`:
    // here `outer.index_extended(45)` overflows, whereas the normalized layout `(2, 8):(3, d_big)`
    // restarts at the second mode after 2 steps and stays well within `isize::MAX`.
    let d_big = isize::MAX / 10;
    let outer_big = dyn_layout(&[4, 6], &[1, d_big]);
    let inner_weak = dyn_layout(&[16], &[3]);
    assert_eq!(outer_big.index_extended(45), Err(Error::Overflow));
    let weak_big = composition(outer_big, inner_weak).unwrap();
    assert!(weak_big.is_normalized());
    assert_grouped(
        &weak_big.as_affine().unwrap().unwrap(),
        &[2, 8],
        &[3, d_big],
        &[2],
    );

    // Stride divisibility failure (5 >= 4 and 5 % 4 != 0): eager errors, lazy evaluates pointwise.
    let bad_stride = dyn_layout(&[2], &[5]);
    assert_eq!(
        composition_layout(&row, &bad_stride),
        Err(Error::NotDivisible)
    );
    let lazy_stride = composition(row, bad_stride).unwrap();
    assert_eq!(lazy_stride.outer(), &row);
    assert_eq!(lazy_stride.inner(), &bad_stride);
    assert!(!lazy_stride.is_normalized());
    assert_eq!(lazy_stride.as_affine(), Ok(None));
    assert_eq!(values(&lazy_stride), [0, 7]);

    // Shape divisibility failure (ceil(4/2) = 2 does not divide rest_shape = 3):
    // eager errors, while lazy evaluates pointwise in both bounded and extended domains.
    let bad_shape = dyn_layout(&[3], &[2]);
    assert_eq!(
        composition_layout(&row, &bad_shape),
        Err(Error::NotDivisible)
    );
    let lazy_shape = composition(row, bad_shape).unwrap();
    assert!(!lazy_shape.is_normalized());
    assert_eq!(lazy_shape.as_affine(), Ok(None));
    assert_eq!(values(&lazy_shape), [0, 12, 1]);
    // At extended index 4, bad_shape(4) = 8, and row.index_extended(8) = 2.
    assert_eq!(lazy_shape.index_extended(4), Ok(2));
    assert_eq!(lazy_shape.index_extended_of_coord(&[4]), Ok(2));
}

#[test]
fn composition_handles_broadcast_empty_unit_and_non_affine_maps() {
    let row = dyn_layout(&[4, 6], &[6, 1]);

    // Broadcast, unit, and first-mode-consumed leaves in compose_leaf.
    assert_grouped(
        &composition_layout(&row, &dyn_layout(&[5], &[0])).unwrap(),
        &[5],
        &[0],
        &[1],
    );
    assert_grouped(
        &composition_layout(&row, &dyn_layout(&[1], &[3])).unwrap(),
        &[1],
        &[1],
        &[1],
    );
    assert_grouped(
        &composition_layout(&row, &dyn_layout(&[2], &[1])).unwrap(),
        &[2],
        &[6],
        &[1],
    );

    // An empty inner domain produces no addresses, so even an empty outer is allowed.
    let empty_composed = composition(TableMap(&[]), dyn_layout(&[0], &[3])).unwrap();
    assert_eq!(empty_composed.size(), Ok(0));
    assert_eq!(empty_composed.address_range(), Ok(None));
    assert_eq!(empty_composed.index(0), Err(Error::CoordinateOutOfBounds));

    // Normalized `Composition::address_range` delegates to the cached affine layout in O(leaves).
    let huge_comp =
        composition(dyn_layout(&[1 << 30], &[2]), dyn_layout(&[1 << 30], &[1])).unwrap();
    assert!(huge_comp.is_normalized());
    assert_eq!(
        huge_comp.address_range(),
        Ok(Some((0, ((1isize << 30) - 1) * 2)))
    );

    // Non-affine outer (Gray-code swizzle) composed with affine inner.
    let swizzled = TableMap(&[0, 1, 3, 2, 4, 5, 7, 6]);
    let sub = composition(swizzled, dyn_layout(&[2, 2], &[1, 4])).unwrap();
    assert!(!sub.is_normalized());
    assert_eq!(sub.as_affine(), Ok(None));
    assert_eq!(values(&sub), [0, 1, 4, 5]);
    assert_eq!(sub.address_range(), Ok(Some((0, 5))));

    // Exceeding a non-affine outer's domain is rejected at construction.
    assert_eq!(
        composition(TableMap(&[0, 1, 3, 2]), dyn_layout(&[8], &[1])).err(),
        Some(Error::CoordinateOutOfBounds)
    );

    // Affine outer composed with non-affine inner reaching into outer's extended domain.
    let into_ext = composition(dyn_layout(&[2, 2], &[1, 4]), TableMap(&[0, 5, 10])).unwrap();
    assert!(!into_ext.is_normalized());
    assert_eq!(values(&into_ext), [0, 9, 20]);
}

#[test]
fn composition_injectivity_covers_normalized_within_domain_and_extended_fallbacks() {
    let row = dyn_layout(&[4, 6], &[6, 1]);

    // Normalized: delegates to cached GroupedLayout.
    let norm_inj = composition(row, dyn_layout(&[2, 3], &[2, 4])).unwrap();
    assert_eq!(norm_inj.is_injective(), Ok(true));
    let norm_dup = composition(row, dyn_layout(&[4, 6], &[0, 4])).unwrap();
    assert_eq!(norm_dup.is_injective(), Ok(false));

    // Normalized into extended domain where outer's strides overlap: (2,3):(3,1) o 8:1 -> (2,4):(3,1).
    let wrapped = composition(dyn_layout(&[2, 3], &[3, 1]), dyn_layout(&[8], &[1])).unwrap();
    assert_eq!(wrapped.is_injective(), Ok(false));

    // Unnormalized within outer's domain + injective outer: delegates to inner.is_injective().
    let unnorm_inj = composition(row, dyn_layout(&[3], &[2])).unwrap();
    assert!(!unnorm_inj.is_normalized());
    assert_eq!(unnorm_inj.is_injective(), Ok(true));
    let unnorm_dup = composition(row, dyn_layout(&[3, 2], &[2, 0])).unwrap();
    assert!(!unnorm_dup.is_normalized());
    assert_eq!(unnorm_dup.is_injective(), Ok(false));

    // Unnormalized into outer's extended domain: enumerates pairs because row(1) == row_ext(24) == 6.
    let ext_collide = composition(row, TableMap(&[1, 24])).unwrap();
    assert_eq!(ext_collide.is_injective(), Ok(false));
    let ext_distinct = composition(row, TableMap(&[0, 24])).unwrap();
    assert_eq!(ext_distinct.is_injective(), Ok(true));

    // Unnormalized with non-injective outer: enumerates rather than assuming non-injective.
    let non_inj_outer = TableMap(&[0, 0, 5, 9]);
    let picks_distinct = composition(non_inj_outer, dyn_layout(&[2], &[2])).unwrap();
    assert_eq!(picks_distinct.is_injective(), Ok(true));
    let picks_alias = composition(non_inj_outer, dyn_layout(&[2], &[1])).unwrap();
    assert_eq!(picks_alias.is_injective(), Ok(false));

    // When `outer`'s stride-separation proof is inconclusive, `is_injective` enumerates
    // the small composed domain rather than scanning all elements of `outer`.
    let huge_inconclusive_outer = dyn_layout(&[1_000_000, 2], &[2, 3]);
    let tiny_comp = composition(huge_inconclusive_outer, TableMap(&[0, 1])).unwrap();
    assert!(!tiny_comp.is_normalized());
    assert_eq!(tiny_comp.is_injective(), Ok(true));
}

#[test]
fn composition_reports_negative_indices_invalid_shapes_and_overflow() {
    assert_eq!(
        composition(dyn_layout(&[4], &[1]), dyn_layout(&[2], &[-1])).err(),
        Some(Error::CoordinateOutOfBounds)
    );
    assert_eq!(
        composition_layout(&dyn_layout(&[4], &[1]), &dyn_layout(&[2], &[-1])),
        Err(Error::CoordinateOutOfBounds)
    );
    assert_eq!(
        composition_layout(&dyn_layout(&[4, 4], &[1, 4]), &dyn_layout(&[2], &[-1])),
        Err(Error::CoordinateOutOfBounds)
    );

    // A unit-extent leaf with a non-divisible negative stride has non-negative reachable
    // addresses [0, 1], but still violates CuTe's stride divisibility check `|-5| >= 4`.
    assert_eq!(
        composition_layout(
            &dyn_layout(&[4, 6], &[6, 1]),
            &dyn_layout(&[1, 2], &[-5, 1])
        ),
        Err(Error::NotDivisible)
    );
    let unit_neg =
        composition(dyn_layout(&[4, 6], &[6, 1]), dyn_layout(&[1, 2], &[-5, 1])).unwrap();
    assert!(!unit_neg.is_normalized());
    assert_eq!(values(&unit_neg), [0, 6]);

    // An uncoalesceable leading zero extent in `outer` fails multi-mode composition.
    assert_eq!(
        composition_layout(&dyn_layout(&[0, 4], &[1, 4]), &dyn_layout(&[2], &[1])),
        Err(Error::InvalidShape)
    );

    // A unit leaf with stride `isize::MIN` has address range [0, 1], but |isize::MIN| overflows.
    assert_eq!(
        composition(
            dyn_layout(&[4, 6], &[6, 1]),
            dyn_layout(&[1, 2], &[isize::MIN, 1])
        )
        .err(),
        Some(Error::Overflow)
    );

    // Extended evaluation probe at `max` catches address overflow at construction time.
    assert_eq!(
        composition(dyn_layout(&[2], &[isize::MAX / 2]), dyn_layout(&[4], &[1])).err(),
        Some(Error::Overflow)
    );
}

#[test]
fn composition_associativity_and_leaf_capacity_limits_hold() {
    let a = dyn_layout(&[4, 6], &[10, 1]);
    let b = dyn_layout(&[2, 6], &[2, 4]);
    let c = grouped(&[2, 3], &[1, 4], &[1, 1]);

    // `(A o B) o C` and `A o (B o C)` agree in grouping and pointwise mapping.
    let ab = composition_layout(&a, &b).unwrap();
    let bc = composition_layout(&b, &c).unwrap();
    let left_assoc = composition_layout(&ab, &c).unwrap();
    let right_assoc = composition_layout(&a, &bc).unwrap();
    assert_eq!(left_assoc.rank(), right_assoc.rank());
    assert_eq!(values(&left_assoc), values(&right_assoc));

    // Each `4:1` mode expands against `(2,2):(1,4)` into 2 leaves; 17 modes -> 34 > MAX_RANK.
    let outer = dyn_layout(&[2, 2], &[1, 4]);
    let inner_17 = dyn_layout(&[4; 17], &[1; 17]);
    assert_eq!(
        composition_layout(&outer, &inner_17),
        Err(Error::CapacityExceeded)
    );
    assert_eq!(
        composition(outer, inner_17).err(),
        Some(Error::CapacityExceeded)
    );
}
