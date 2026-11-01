//! CuTe complement 的全静态、混合与运行时契约。

use noodle_cute::algebra::{TypedComplement, complement, complement_extended, complement_shape};
use noodle_cute::layout::{AffineLayoutMap, DynLayout, Layout, LayoutMap, MAX_RANK};
use noodle_cute::{DynInt, Error, Int, IntTuple, Nil, StaticValue, shape, stride};

pub mod common;
use common::{complemented, complemented_extended};

type Pair<A, B> = IntTuple<A, IntTuple<B, Nil>>;

fn dyn_layout(shape: &[isize], stride: &[isize]) -> DynLayout {
    DynLayout::new(shape, stride).unwrap()
}

fn values<L: LayoutMap>(layout: &L) -> Vec<isize> {
    (0..layout.size().unwrap())
        .map(|i| layout.index(i).unwrap())
        .collect()
}

macro_rules! leaves_33 {
    ($leaf:expr) => {
        shape![
            ($leaf, $leaf, $leaf, $leaf, $leaf, $leaf, $leaf, $leaf),
            ($leaf, $leaf, $leaf, $leaf, $leaf, $leaf, $leaf, $leaf),
            ($leaf, $leaf, $leaf, $leaf, $leaf, $leaf, $leaf, $leaf),
            ($leaf, $leaf, $leaf, $leaf, $leaf, $leaf, $leaf, $leaf),
            $leaf
        ]
    };
}

type Tile = Layout<Pair<Int<2>, Int<2>>, Pair<Int<1>, Int<6>>>;
const TILE: Tile = Layout {
    shape: shape![Int::<2>, Int::<2>],
    stride: stride![Int::<1>, Int::<6>],
};

#[test]
fn static_results_are_const_zero_sized_and_tile_the_cotarget() {
    const RESULT: Layout<Pair<Int<3>, Int<2>>, Pair<Int<2>, Int<12>>> =
        complemented(TILE, Int::<24>);
    const ASSOCIATED: TypedComplement<Tile, Int<24>> =
        <TypedComplement<Tile, Int<24>> as StaticValue>::VALUE;
    assert_eq!(RESULT, ASSOCIATED);
    assert_eq!(RESULT, TILE.complement(Int::<24>).unwrap());
    assert_eq!(core::mem::size_of_val(&RESULT), 0);
    assert_eq!(values(&RESULT), [0, 2, 4, 12, 14, 16]);
    let mut visits = [0; 24];
    for a in values(&TILE) {
        for b in values(&RESULT) {
            visits[(a + b) as usize] += 1;
        }
    }
    assert_eq!(visits, [1; 24]);
}

#[test]
fn static_filter_sort_and_floor_gaps_follow_cute() {
    const NESTED: Layout<Pair<Int<2>, Int<2>>, Pair<Int<6>, Int<24>>> = complemented(
        Layout {
            shape: shape![(Int::<2>, Int::<2>), Int::<3>],
            stride: stride![(Int::<1>, Int::<12>), Int::<2>],
        },
        Int::<48>,
    );
    assert_eq!(values(&NESTED), [0, 6, 24, 30]);
    const SORTED: Layout<Int<2>, Int<6>> = complemented(
        Layout {
            shape: shape![Int::<2>, Int::<3>],
            stride: stride![Int::<3>, Int::<1>],
        },
        Int::<12>,
    );
    assert_eq!(values(&SORTED), [0, 6]);
    const FLOOR: Layout<Int<3>, Int<6>> = complemented(
        Layout {
            shape: shape![Int::<2>, Int::<2>],
            stride: stride![Int::<1>, Int::<3>],
        },
        Int::<13>,
    );
    assert_eq!(values(&FLOOR), [0, 6, 12]);
    const FILTERED: Layout<Int<2>, Int<1>> = complemented(
        Layout {
            shape: shape![Int::<1>, Int::<4>, Int::<9>],
            stride: stride![Int::<-9>, Int::<2>, Int::<0>],
        },
        Int::<8>,
    );
    assert_eq!(values(&FILTERED), [0, 1]);
}

#[test]
fn static_tuple_cotargets_and_broadcast_special_case_preserve_semantics() {
    const TUPLE: Layout<Int<12>, Int<3>> = complemented(
        Layout {
            shape: Int::<3>,
            stride: Int::<1>,
        },
        shape![(Int::<4>, Int::<6>)],
    );
    const INTEGER: Layout<Int<8>, Int<3>> = complemented(
        Layout {
            shape: Int::<3>,
            stride: Int::<1>,
        },
        Int::<24>,
    );
    assert_eq!(TUPLE.size(), Ok(12));
    assert_eq!(INTEGER.size(), Ok(8));
    const BROADCAST: Layout<Int<6>, Int<1>> = complemented(
        Layout {
            shape: Int::<4>,
            stride: Int::<0>,
        },
        shape![Int::<2>, Int::<3>],
    );
    assert_eq!(values(&BROADCAST), [0, 1, 2, 3, 4, 5]);
    const SCALAR: Layout<Int<1>, Int<1>> = complemented(
        Layout {
            shape: Nil,
            stride: Nil,
        },
        Nil,
    );
    const EMPTY_TARGET: Layout<Int<2>, Int<1>> = complemented(
        Layout {
            shape: Int::<4>,
            stride: Int::<2>,
        },
        Nil,
    );
    const UNIT: Layout<Int<1>, Int<0>> = complemented(
        Layout {
            shape: Int::<4>,
            stride: Int::<1>,
        },
        Int::<4>,
    );
    const ZERO: Layout<Int<0>, Int<3>> = complemented(
        Layout {
            shape: Int::<3>,
            stride: Int::<1>,
        },
        shape![Int::<0>, Int::<6>],
    );
    assert_eq!(SCALAR.stride, Int::<1>);
    assert_eq!(EMPTY_TARGET.shape, Int::<2>);
    assert_eq!(UNIT.index(0), Ok(0));
    assert_eq!(ZERO.size(), Ok(0));
}

#[test]
fn static_extended_complement_keeps_the_terminal_stride() {
    const GAPS: Layout<Pair<Int<3>, Int<1>>, Pair<Int<2>, Int<12>>> = complemented_extended(TILE);
    const COMPACT: Layout<Int<1>, Int<4>> = complemented_extended(Layout {
        shape: Int::<4>,
        stride: Int::<1>,
    });
    const FLOOR: Layout<Int<1>, Int<6>> = complemented_extended(Layout {
        shape: shape![Int::<2>, Int::<2>],
        stride: stride![Int::<1>, Int::<3>],
    });
    const BROADCAST: Layout<Int<1>, Int<1>> = complemented_extended(Layout {
        shape: Int::<4>,
        stride: Int::<0>,
    });
    const SCALAR: Layout<Int<1>, Int<1>> = complemented_extended(Layout {
        shape: Nil,
        stride: Nil,
    });
    assert_eq!(GAPS, TILE.complement_extended().unwrap());
    assert_eq!(GAPS.index_extended(3), Ok(12));
    assert_eq!(COMPACT.index_extended(2), Ok(8));
    assert_eq!(FLOOR.index_extended(2), Ok(12));
    assert_eq!(BROADCAST, SCALAR);
}

#[test]
fn static_filter_accepts_more_than_max_rank_input_leaves() {
    const RESULT: Layout<Int<8>, Int<1>> = complemented(
        Layout {
            shape: leaves_33!(Int::<1>),
            stride: leaves_33!(Int::<1>),
        },
        Int::<8>,
    );
    assert_eq!(RESULT.size(), Ok(8));
    assert_eq!(core::mem::size_of_val(&RESULT), 0);
}

#[test]
fn mixed_shapes_with_static_strides_keep_only_dependent_leaves_dynamic() {
    for n in [1, 2] {
        let input = Layout::new(shape![DynInt(n), Int::<2>], stride![Int::<1>, Int::<6>]).unwrap();
        let result: Layout<Pair<DynInt, Int<2>>, Pair<DynInt, Int<12>>> =
            input.complement(Int::<24>).unwrap();
        assert_eq!(result.shape, shape![DynInt(6 / n), Int::<2>]);
        assert_eq!(result.stride, stride![DynInt(n), Int::<12>]);
    }
    let reversed = Layout::new(shape![Int::<2>, DynInt(2)], stride![Int::<6>, Int::<1>]).unwrap();
    let result: Layout<Pair<DynInt, Int<2>>, Pair<DynInt, Int<12>>> =
        reversed.complement(Int::<24>).unwrap();
    assert_eq!(values(&result), [0, 2, 4, 12, 14, 16]);
}

#[test]
fn rank_one_dynamic_strides_work_without_erasing_static_facts() {
    let input = Layout::new(Int::<4>, DynInt(2)).unwrap();
    let result: Layout<Pair<DynInt, DynInt>, Pair<Int<1>, DynInt>> =
        input.complement(Int::<24>).unwrap();
    assert_eq!(result.shape, shape![DynInt(2), DynInt(3)]);
    assert_eq!(result.stride, stride![Int::<1>, DynInt(8)]);
    let native = Layout::new(4isize, 2isize).unwrap();
    let native_result: Layout<Pair<DynInt, DynInt>, Pair<Int<1>, isize>> =
        native.complement(24isize).unwrap();
    assert_eq!(values(&result), values(&native_result));
    let extended: Layout<Pair<DynInt, Int<1>>, Pair<Int<1>, DynInt>> =
        input.complement_extended().unwrap();
    assert_eq!(extended.shape, shape![DynInt(2), Int::<1>]);
    assert_eq!(extended.stride, stride![Int::<1>, DynInt(8)]);
}

#[test]
fn mixed_cotarget_coalesces_adjacent_leaves_of_the_same_kind() {
    let input = Layout::new(Int::<3>, Int::<1>).unwrap();
    let target = shape![Int::<4>, DynInt(2), DynInt(3), Int::<2>];
    type Output = Layout<Pair<Int<2>, DynInt>, Pair<Int<3>, Int<6>>>;
    let result: Output = input.complement(target).unwrap();
    assert_eq!(result.shape, shape![Int::<2>, DynInt(12)]);
    assert_eq!(result.stride, stride![Int::<3>, Int::<6>]);
    assert_eq!(result.size(), Ok(24));

    let broadcast = Layout::new(DynInt(9), Int::<0>).unwrap();
    type Broadcast = Layout<
        IntTuple<Int<4>, IntTuple<DynInt, Pair<Int<1>, DynInt>>>,
        IntTuple<Int<1>, IntTuple<Int<4>, Pair<DynInt, DynInt>>>,
    >;
    let result: Broadcast = broadcast
        .complement(shape![Int::<2>, Int::<2>, DynInt(3), Int::<1>, DynInt(1)])
        .unwrap();
    assert_eq!(
        result.shape,
        shape![Int::<4>, DynInt(3), Int::<1>, DynInt(1)]
    );
    assert_eq!(
        result.stride,
        stride![Int::<1>, Int::<4>, DynInt(12), DynInt(12)]
    );
}

#[test]
fn runtime_unit_extents_keep_their_typed_modes() {
    let input = Layout::new(Int::<4>, Int::<1>).unwrap();
    let result: Layout<DynInt, Int<4>> = input.complement(DynInt(4)).unwrap();
    assert_eq!(result.shape, DynInt(1));
    assert_eq!(result.stride, Int::<4>);
    let input = Layout::new(DynInt(1), Int::<4>).unwrap();
    let extended: Layout<Pair<Int<4>, Int<1>>, Pair<Int<1>, DynInt>> =
        input.complement_extended().unwrap();
    assert_eq!(extended.stride, stride![Int::<1>, DynInt(4)]);
    let filtered = Layout::new(shape![Int::<1>, Int::<4>], stride![DynInt(-9), DynInt(2)]).unwrap();
    let result: Layout<Pair<DynInt, DynInt>, Pair<Int<1>, DynInt>> =
        filtered.complement(Int::<24>).unwrap();
    assert_eq!(result.shape, shape![DynInt(2), DynInt(3)]);

    let input = Layout::new(Int::<2>, Int::<1>).unwrap();
    let result: Layout<DynInt, Int<2>> = input.complement(DynInt(isize::MAX)).unwrap();
    assert_eq!(result.size(), Ok(isize::MAX / 2 + 1));
    assert_eq!(result.index(isize::MAX / 2), Ok(isize::MAX - 1));
}

#[test]
fn mixed_errors_are_checked_before_unchecked_integer_operators() {
    assert_eq!(
        Layout::new(DynInt(0), Int::<1>)
            .unwrap()
            .complement(Int::<8>),
        Err(Error::Unsupported)
    );
    assert_eq!(
        Layout {
            shape: DynInt(-1),
            stride: Int::<1>
        }
        .complement(Int::<8>),
        Err(Error::InvalidShape)
    );
    assert_eq!(
        Layout::new(Int::<2>, DynInt(-1))
            .unwrap()
            .complement(Int::<8>),
        Err(Error::Unsupported)
    );
    assert_eq!(
        Layout::new(Int::<2>, DynInt(0))
            .unwrap()
            .complement(Int::<8>),
        Err(Error::NonInjective)
    );
    assert_eq!(
        Layout::new(Int::<2>, Int::<1>)
            .unwrap()
            .complement(DynInt(-1)),
        Err(Error::InvalidShape)
    );
    assert_eq!(
        Layout::new(Int::<2>, DynInt(isize::MAX))
            .unwrap()
            .complement_extended(),
        Err(Error::Overflow)
    );
    let overlap = Layout::new(shape![DynInt(4), Int::<2>], stride![Int::<2>, Int::<3>]).unwrap();
    assert_eq!(overlap.complement(Int::<16>), Err(Error::NonInjective));
    let broadcast = Layout::new(Int::<1>, Int::<0>).unwrap();
    assert_eq!(
        broadcast.complement(shape![DynInt(isize::MAX), DynInt(2)]),
        Err(Error::Overflow)
    );
}

#[test]
fn equal_static_strides_keep_the_original_leaf_order() {
    let input = Layout::new(shape![DynInt(1), Int::<2>], stride![Int::<2>, Int::<2>]).unwrap();
    type Triple<A, B, C> = IntTuple<A, Pair<B, C>>;
    let result: Layout<Triple<Int<2>, DynInt, Int<2>>, Triple<Int<1>, DynInt, Int<4>>> =
        input.complement(Int::<8>).unwrap();
    assert_eq!(result.shape, shape![Int::<2>, DynInt(1), Int::<2>]);
    assert_eq!(result.stride, stride![Int::<1>, DynInt(2), Int::<4>]);
    assert_eq!(values(&result), [0, 1, 4, 5]);
}

#[test]
fn typed_complement_is_not_limited_by_runtime_leaf_capacity() {
    let input = Layout::new(leaves_33!(DynInt(1)), leaves_33!(Int::<1>)).unwrap();
    let result = input.complement(Int::<8>).unwrap();
    assert_eq!(result.rank(), MAX_RANK + 1);
    assert_eq!(result.as_flat_layout(), Err(Error::CapacityExceeded));
    assert_eq!(values(&result), [0, 1, 2, 3, 4, 5, 6, 7]);
    let extended = input.complement_extended().unwrap();
    assert_eq!(extended.rank(), MAX_RANK + 1);
    assert_eq!(extended.index_extended(8), Ok(8));
}

#[test]
fn zero_cotarget_preserves_static_zero_strides_and_checked_fold_order() {
    let broadcast = Layout::new(Int::<2>, Int::<0>).unwrap();
    type Triple<A, B, C> = IntTuple<A, Pair<B, C>>;
    let result: Layout<Triple<DynInt, Int<0>, DynInt>, Triple<Int<1>, DynInt, Int<0>>> = broadcast
        .complement(shape![DynInt(isize::MAX), Int::<0>, DynInt(2)])
        .unwrap();
    assert_eq!(result.size(), Ok(0));
    assert_eq!(
        result.stride,
        stride![Int::<1>, DynInt(isize::MAX), Int::<0>]
    );
    assert_eq!(
        broadcast.complement(shape![DynInt(isize::MAX), DynInt(2), DynInt(0)]),
        Err(Error::Overflow)
    );
    let input = Layout::new(Int::<3>, Int::<1>).unwrap();
    let result: Layout<Pair<Int<0>, DynInt>, Pair<Int<3>, Int<0>>> =
        input.complement(shape![Int::<0>, DynInt(6)]).unwrap();
    assert_eq!(result.size(), Ok(0));
    assert_eq!(result.shape, shape![Int::<0>, DynInt(2)]);
}

#[test]
fn complement_matches_cute_integer_cotarget_fixtures_and_covers_lattices() {
    struct Case {
        shape: &'static [isize],
        stride: &'static [isize],
        cotarget: isize,
        expected_shape: &'static [isize],
        expected_stride: &'static [isize],
    }

    let cases = [
        Case {
            shape: &[4],
            stride: &[2],
            cotarget: 24,
            expected_shape: &[2, 3],
            expected_stride: &[1, 8],
        },
        Case {
            shape: &[2, 2],
            stride: &[1, 6],
            cotarget: 24,
            expected_shape: &[3, 2],
            expected_stride: &[2, 12],
        },
        Case {
            shape: &[2, 2],
            stride: &[1, 3],
            cotarget: 13,
            expected_shape: &[3],
            expected_stride: &[6],
        },
        Case {
            shape: &[2, 2],
            stride: &[1, 3],
            cotarget: 12,
            expected_shape: &[2],
            expected_stride: &[6],
        },
        Case {
            shape: &[4, 3],
            stride: &[0, 2],
            cotarget: 24,
            expected_shape: &[2, 4],
            expected_stride: &[1, 6],
        },
        Case {
            shape: &[4],
            stride: &[0],
            cotarget: 6,
            expected_shape: &[6],
            expected_stride: &[1],
        },
        Case {
            shape: &[4],
            stride: &[1],
            cotarget: 6,
            expected_shape: &[2],
            expected_stride: &[4],
        },
        Case {
            shape: &[4],
            stride: &[1],
            cotarget: 0,
            expected_shape: &[0],
            expected_stride: &[4],
        },
        Case {
            shape: &[4],
            stride: &[1],
            cotarget: 4,
            expected_shape: &[1],
            expected_stride: &[0],
        },
        Case {
            shape: &[4],
            stride: &[2],
            cotarget: 8,
            expected_shape: &[2],
            expected_stride: &[1],
        },
        Case {
            shape: &[2, 3],
            stride: &[3, 1],
            cotarget: 12,
            expected_shape: &[2],
            expected_stride: &[6],
        },
    ];

    for c in cases {
        let actual = complement(dyn_layout(c.shape, c.stride), c.cotarget).unwrap();
        assert_eq!(actual.shapes(), c.expected_shape, "{:?}", c.shape);
        assert_eq!(actual.strides(), c.expected_stride, "{:?}", c.stride);
    }

    // Official CuTe tutorial check: (2,2):(1,6) and its complement mod 24 tile [0, 24) bijectively.
    let tile = dyn_layout(&[2, 2], &[1, 6]);
    let rest = complement(tile, 24).unwrap();
    assert_eq!(values(&rest), [0, 2, 4, 12, 14, 16]);
    let mut visits = [0usize; 24];
    for shift in values(&rest) {
        for base in values(&tile) {
            visits[(base + shift) as usize] += 1;
        }
    }
    assert!(visits.into_iter().all(|n| n == 1));

    // Typed nested, scalar, and unit-extent layouts work directly via AffineLayoutMap.
    let nested = Layout::new(
        shape![(Int::<2>, Int::<2>), Int::<3>],
        stride![(Int::<1>, Int::<12>), Int::<2>],
    )
    .unwrap();
    let nested_comp = complement(nested, 48).unwrap();
    assert_eq!(
        (nested_comp.shapes(), nested_comp.strides()),
        (&[2, 2][..], &[6, 24][..])
    );

    let scalar = Layout::new(Nil, Nil).unwrap();
    assert_eq!(complement(scalar, 5), Ok(dyn_layout(&[5], &[1])));

    // A negative stride on a size-1 leaf is removed by filter/coalesce before sign validation.
    let unit_neg = Layout::new(shape![Int::<1>, Int::<4>], stride![Int::<-9>, Int::<2>]).unwrap();
    assert_eq!(complement(unit_neg, 8), Ok(dyn_layout(&[2], &[1])));
}

#[test]
fn complement_shape_folds_tuple_cotargets_leaf_by_leaf() {
    // Tuple ceil_div folds leaf by leaf: complement(3:1, (4, 6)) = 12:3, whereas
    // complement(3:1, 24) = 8:3 even though product((4, 6)) == 24.
    assert_eq!(
        complement_shape(dyn_layout(&[3], &[1]), &[4, 6]),
        Ok(dyn_layout(&[12], &[3]))
    );
    assert_eq!(
        complement(dyn_layout(&[3], &[1]), 24),
        Ok(dyn_layout(&[8], &[3]))
    );

    type ShapeCotargetCase = (
        &'static [isize],
        &'static [isize],
        &'static [isize],
        &'static [isize],
        &'static [isize],
    );
    let cases: [ShapeCotargetCase; 8] = [
        (&[4], &[0], &[2, 3], &[6], &[1]),
        (&[4], &[1], &[4, 6], &[6], &[4]),
        (&[5], &[1], &[4, 6], &[3], &[5]),
        (&[4], &[0], &[2, 0], &[0], &[1]),
        (&[3], &[1], &[0, 6], &[0], &[3]),
        // `2:3` and `0:6` coalesce (`2 * 3 == 6`) to `0:3`; `3:2` with `[12, 0]` retains `(2, 0):(1, 6)`.
        (&[3], &[1], &[4, 0], &[0], &[3]),
        (&[3], &[2], &[12, 0], &[2, 0], &[1, 6]),
        (&[2, 3], &[1, 2], &[2, 3], &[1], &[0]),
    ];
    for (s, d, cotarget, rs, rd) in cases {
        let actual = complement_shape(dyn_layout(s, d), cotarget).unwrap();
        assert_eq!((actual.shapes(), actual.strides()), (rs, rd));
    }

    // Empty cotarget slice acts as multiplicative identity 1.
    assert_eq!(
        complement_shape(dyn_layout(&[4], &[0]), &[]),
        Ok(dyn_layout(&[1], &[1]))
    );
    assert_eq!(
        complement_shape(dyn_layout(&[4], &[2]), &[]),
        Ok(dyn_layout(&[2], &[1]))
    );
}

#[test]
fn complement_extended_marks_repetition_stride_without_cotarget() {
    // Trailing 1:current carries s_last * d_last (8 for 4:2, not cosize 7).
    type ExtendedCase = (
        &'static [isize],
        &'static [isize],
        &'static [isize],
        &'static [isize],
    );
    let cases: [ExtendedCase; 6] = [
        (&[4], &[2], &[2, 1], &[1, 8]),
        (&[2, 2], &[1, 6], &[3, 1], &[2, 12]),
        (&[2, 2], &[1, 3], &[1], &[6]),
        (&[4], &[1], &[1], &[4]),
        (&[4], &[0], &[1], &[1]),
        (&[], &[], &[1], &[1]),
    ];
    for (s, d, rs, rd) in cases {
        let actual = complement_extended(dyn_layout(s, d)).unwrap();
        assert_eq!((actual.shapes(), actual.strides()), (rs, rd));
    }
}

#[test]
fn complement_rejects_empty_negative_overlapping_overflowing_and_over_capacity_inputs() {
    for empty in [dyn_layout(&[0], &[1]), dyn_layout(&[3, 0], &[1, 3])] {
        assert_eq!(complement(empty, 8), Err(Error::Unsupported));
        assert_eq!(complement_shape(empty, &[2, 4]), Err(Error::Unsupported));
        assert_eq!(complement_extended(empty), Err(Error::Unsupported));
    }

    assert_eq!(
        complement(dyn_layout(&[2], &[-1]), 4),
        Err(Error::Unsupported)
    );
    assert_eq!(
        complement_extended(dyn_layout(&[2, 3], &[1, -2])),
        Err(Error::Unsupported)
    );

    assert_eq!(
        complement(dyn_layout(&[2], &[1]), -1),
        Err(Error::InvalidShape)
    );
    assert_eq!(
        complement_shape(dyn_layout(&[2], &[1]), &[4, -2]),
        Err(Error::InvalidShape)
    );

    for overlapping in [
        dyn_layout(&[2, 2], &[2, 3]),
        dyn_layout(&[2, 2], &[1, 1]),
        dyn_layout(&[2, 2], &[2, 2]),
    ] {
        assert_eq!(complement(overlapping, 16), Err(Error::NonInjective));
        assert_eq!(complement_extended(overlapping), Err(Error::NonInjective));
    }

    // `2:isize::MAX` is a valid layout, but its next repetition stride `2 * isize::MAX` overflows.
    assert_eq!(
        complement_extended(dyn_layout(&[2], &[isize::MAX])),
        Err(Error::Overflow)
    );
    assert_eq!(
        complement_shape(dyn_layout(&[1], &[0]), &[isize::MAX, 2]),
        Err(Error::Overflow)
    );

    // 20 internal gaps + 13 non-unit cotarget extents exceed MAX_RANK (32) before coalescing.
    let mut shapes = [0isize; 20];
    let mut strides = [0isize; 20];
    for k in 0..20 {
        shapes[k] = 2;
        strides[k] = 2 * (1isize << (2 * k));
    }
    let gapped = dyn_layout(&shapes, &strides);
    let mut cotarget = [2isize; 13];
    cotarget[0] = (1isize << 40) + 1;
    assert_eq!(
        complement_shape(gapped, &cotarget),
        Err(Error::CapacityExceeded)
    );
}
