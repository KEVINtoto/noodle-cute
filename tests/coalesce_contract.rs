//! CuTe bw_coalesce / coalesce_x 的静态、混合与运行时契约。

use core::mem::size_of_val;

use noodle_cute::algebra::{StaticCoalesce, CoalescedLayout, static_coalesce, static_coalesce_extended};
use noodle_cute::layout::{AffineLayoutMap, DynLayout, Layout, LayoutMap};
use noodle_cute::{DynInt, Error, Int, IntTuple, Nil, shape, stride};

type Pair<A, B> = IntTuple<A, IntTuple<B, Nil>>;
type Triple<A, B, C> = IntTuple<A, Pair<B, C>>;
type Single<S, D> = Layout<IntTuple<S, Nil>, IntTuple<D, Nil>>;

#[test]
fn static_entries_produce_const_zero_sized_results() {
    type Input = Layout<Pair<Int<2>, Pair<Int<1>, Int<6>>>, Pair<Int<1>, Pair<Int<99>, Int<2>>>>;
    const INPUT: Input = Layout {
        shape: shape![Int::<2>, (Int::<1>, Int::<6>)],
        stride: stride![Int::<1>, (Int::<99>, Int::<2>)],
    };
    const OUTPUT: Single<Int<12>, Int<1>> = static_coalesce(INPUT);
    assert_eq!(OUTPUT, INPUT.coalesce().unwrap());
    assert_eq!(size_of_val(&OUTPUT), 0);
    for i in 0..12 {
        assert_eq!(OUTPUT.index(i), INPUT.index(i));
    }

    type Unit = Layout<Int<1>, Int<8>>;
    const NORMAL: Layout<Int<1>, Int<0>> = <Unit as StaticCoalesce>::OUTPUT;
    const EXTENDED: Single<Int<2>, Int<8>> = <Unit as StaticCoalesce<true>>::OUTPUT;
    assert_eq!(NORMAL.index(0), Ok(0));
    assert_eq!(EXTENDED.index_extended(3), Ok(24));

    const EMPTY: Layout<Int<1>, Int<0>> = static_coalesce(Layout {
        shape: Nil,
        stride: Nil,
    });
    const EMPTY_EXTENDED: Single<Int<1>, Int<0>> = static_coalesce_extended(Layout {
        shape: Nil,
        stride: Nil,
    });
    assert_eq!(EMPTY.size(), Ok(1));
    assert_eq!(EMPTY_EXTENDED.index_extended(5), Ok(0));
}

#[test]
fn static_extended_seed_preserves_the_terminal_stride() {
    const INPUT: Layout<Pair<Int<2>, Int<1>>, Pair<Int<1>, Int<8>>> = Layout {
        shape: shape![Int::<2>, Int::<1>],
        stride: stride![Int::<1>, Int::<8>],
    };
    const NORMAL: Single<Int<2>, Int<1>> = static_coalesce(INPUT);
    const EXTENDED: Layout<Pair<Int<2>, Int<2>>, Pair<Int<1>, Int<8>>> =
        static_coalesce_extended(INPUT);
    let _: CoalescedLayout<Pair<Int<2>, Int<1>>, Pair<Int<1>, Int<8>>, true> = EXTENDED;
    assert_eq!(NORMAL.index_extended(2), Ok(2));
    for i in 0..16 {
        assert_eq!(EXTENDED.index_extended(i), INPUT.index_extended(i));
    }

    const MERGED: Single<Int<4>, Int<1>> = static_coalesce_extended(Layout {
        shape: shape![Int::<2>, Int::<1>],
        stride: stride![Int::<1>, Int::<2>],
    });
    assert_eq!(MERGED.size(), Ok(4));
    assert_eq!(MERGED.index_extended(9), Ok(9));
}

#[test]
fn mixed_folds_require_a_static_accumulated_shape() {
    for n in [1, 3] {
        let input = Layout::new(shape![Int::<2>, DynInt(n)], stride![Int::<1>, Int::<2>]).unwrap();
        let normal: Layout<Pair<Int<2>, DynInt>, Pair<Int<1>, Int<2>>> = input.coalesce().unwrap();
        let extended: Layout<Pair<Int<2>, DynInt>, Pair<Int<1>, Int<2>>> =
            input.coalesce_extended().unwrap();
        assert_eq!(normal.shape.1.0, DynInt(n));
        assert_eq!(extended.shape.1.0, DynInt(n));
        for i in 0..12 {
            assert_eq!(extended.index_extended(i), input.index_extended(i));
        }
        let runtime = input.as_flat_layout().unwrap().coalesce().unwrap();
        assert_eq!(runtime.shapes(), &[2 * n]);
    }

    let empty = Layout::new(
        shape![Int::<{ isize::MAX }>, DynInt(2), Int::<0>],
        stride![Int::<0>, Int::<0>, Int::<1>],
    )
    .unwrap();
    let output = empty.coalesce().unwrap();
    assert_eq!(output.rank(), 3);
    assert_eq!(output.size(), Ok(0));
}

#[test]
fn mixed_extended_fold_retains_runtime_values_and_static_merges() {
    let input = Layout::new(
        shape![Int::<2>, (Int::<3>, Int::<1>)],
        stride![Int::<1>, (Int::<2>, DynInt(8))],
    )
    .unwrap();
    let normal: Single<Int<6>, Int<1>> = input.coalesce().unwrap();
    let extended: Layout<Pair<Int<6>, Int<2>>, Pair<Int<1>, DynInt>> =
        input.coalesce_extended().unwrap();
    assert_eq!(normal.index_extended(6), Ok(6));
    assert_eq!(extended.stride.1.0, DynInt(8));
    for i in 0..24 {
        assert_eq!(extended.index_extended(i), input.index_extended(i));
    }

    let all_dynamic =
        Layout::new(shape![DynInt(2), DynInt(1)], stride![DynInt(1), DynInt(8)]).unwrap();
    assert_eq!(all_dynamic.coalesce().unwrap(), all_dynamic);
    assert_eq!(all_dynamic.coalesce_extended().unwrap(), all_dynamic);
}

#[test]
fn mixed_entries_validate_runtime_inputs_and_extended_output() {
    let negative = Layout {
        shape: DynInt(-1),
        stride: Int::<1>,
    };
    assert_eq!(negative.coalesce(), Err(Error::InvalidShape));
    assert_eq!(negative.coalesce_extended(), Err(Error::InvalidShape));

    let terminal = Layout::new(
        shape![Int::<2>, Int::<1>],
        stride![Int::<1>, DynInt(isize::MAX)],
    )
    .unwrap();
    assert!(terminal.coalesce().is_ok());
    assert_eq!(terminal.coalesce_extended(), Err(Error::Overflow));
}

#[test]
fn runtime_backward_fold_agrees_with_static_zero_extent_order() {
    const INPUT: Layout<
        Triple<Int<{ isize::MAX }>, Int<2>, Int<0>>,
        Triple<Int<0>, Int<0>, Int<0>>,
    > = Layout {
        shape: shape![Int::<{ isize::MAX }>, Int::<2>, Int::<0>],
        stride: stride![Int::<0>, Int::<0>, Int::<0>],
    };
    const OUTPUT: Single<Int<0>, Int<0>> = static_coalesce(INPUT);
    let input = INPUT.as_flat_layout().unwrap();
    let normal = input.coalesce().unwrap();
    assert_eq!(normal.shapes(), &[0]);
    assert_eq!(normal.strides(), &[0]);
    assert_eq!(normal, OUTPUT.as_flat_layout().unwrap());
    assert_eq!(input.coalesce_extended(), Ok(normal));

    let tail_overflows = DynLayout::new(&[0, isize::MAX, 2], &[0, 0, 0]).unwrap();
    assert_eq!(tail_overflows.coalesce(), Err(Error::Overflow));
    assert_eq!(tail_overflows.coalesce_extended(), Err(Error::Overflow));
}

#[test]
fn runtime_folds_preserve_bounded_and_extended_maps() {
    for a in 1..=3 {
        for b in 1..=3 {
            for da in -2..=2 {
                for db in -2..=2 {
                    let input = DynLayout::new(&[a, b], &[da, db]).unwrap();
                    let normal = input.coalesce().unwrap();
                    let extended = input.coalesce_extended().unwrap();
                    assert_eq!(normal.size(), input.size());
                    assert_eq!(extended.size(), input.size());
                    for i in 0..a * b {
                        assert_eq!(normal.index(i), input.index(i));
                    }
                    for i in 0..a * b * 3 {
                        assert_eq!(extended.index_extended(i), input.index_extended(i));
                    }
                }
            }
        }
    }
    let scalar = DynLayout::new(&[], &[]).unwrap();
    for output in [
        scalar.coalesce().unwrap(),
        scalar.coalesce_extended().unwrap(),
    ] {
        assert_eq!(output.shapes(), &[1]);
        assert_eq!(output.strides(), &[0]);
    }
}
