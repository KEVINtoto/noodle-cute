//! Contract tests for hierarchical coordinates: `crd2idx`, `idx2crd`, `crd2crd`,
//! `elem_less`, and `Underscore` slicing, following CuTe's `stride.hpp`,
//! `int_tuple.hpp`, `underscore.hpp`, and `layout.hpp`.
//!
//! Expected values are derived from the upstream definitions, and the typed
//! results are checked against the checked `LayoutMap` path where they overlap.

use noodle_cute::coord::{
    X, crd2crd, crd2idx, crd2idx_colex, dice, elem_less, idx2crd, idx2crd_colex, slice,
};
use noodle_cute::layout::LayoutLeft;
use noodle_cute::{DynInt, Int, IntTuple, Layout, Nil, shape, stride};

// stride.hpp: `op((c,C), (s,S), (d,D)) => op(c, s, d) + op((C), (S), (D))`.
#[test]
fn tuple_coordinates_sum_their_mode_contributions() {
    let shape = shape![Int::<2>, (Int::<2>, Int::<2>)];
    let stride = stride![Int::<4>, (Int::<2>, Int::<1>)];
    let nested: Int<5> = crd2idx(shape![Int::<1>, (Int::<0>, Int::<1>)], &shape, &stride);
    assert_eq!(nested.value(), 5);
    let runtime: i32 = crd2idx(shape![1i32, (0i32, 1i32)], &shape, &stride);
    assert_eq!(runtime, 5);
    // A mode coordinate may be an integer for a nested mode: 2 decodes to (0, 1).
    assert_eq!(crd2idx(shape![1i32, 2i32], &shape, &stride), 5);
}

// stride.hpp: `op(c, (s,S), (d,D)) => op(c % prod(s), s, d) + op(c / prod(s), (S), (D))`.
#[test]
fn integer_coordinates_decode_left_major_across_nested_modes() {
    let layout = Layout::new(
        shape![Int::<2>, (Int::<3>, DynInt(4))],
        stride![Int::<12>, (Int::<1>, Int::<3>)],
    )
    .unwrap();
    for i in 0..layout.size().unwrap() {
        let typed: i32 = layout.crd2idx(i as i32);
        assert_eq!(typed as isize, layout.index(i).unwrap(), "index {i}");
    }
}

// The last mode takes the whole quotient: `(2,3):(3,1)` maps 7 to 1*3 + 3*1.
#[test]
fn integer_coordinates_extend_along_the_last_mode() {
    let shape = shape![Int::<2>, Int::<3>];
    let stride = stride![Int::<3>, Int::<1>];
    assert_eq!(crd2idx(7i32, &shape, &stride), 6);
    let nested = shape![Int::<2>, (Int::<2>, Int::<2>)];
    let nested_stride = stride![Int::<1>, (Int::<2>, Int::<4>)];
    // 9 -> (1, 4 -> (0, 2)): the innermost last leaf is not reduced either.
    assert_eq!(crd2idx(9i32, &nested, &nested_stride), 9);
}

#[test]
fn coordinate_results_follow_static_and_width_promotion() {
    let shape = shape![DynInt(3), DynInt(4)];
    let stride = stride![DynInt(1), DynInt(3)];
    // integral_constant.hpp: `_0 * x`, `_0 / x`, and `_0 % x` remain `_0`.
    let zero: Int<0> = crd2idx(Int::<0>, &shape, &stride);
    assert_eq!(zero.value(), 0);
    let wide: DynInt = crd2idx(shape![1i32, 2i32], &shape, &stride);
    assert_eq!(wide, DynInt(7));
    let narrow: i32 = crd2idx(
        shape![1i32, 2i32],
        &shape![Int::<3>, Int::<4>],
        &stride![Int::<1>, 3i32],
    );
    assert_eq!(narrow, 7);
    let negative: i32 = crd2idx(3i32, &Int::<4>, &Int::<-1>);
    assert_eq!(negative, -3);
    let empty: Int<0> = crd2idx(Nil, &Nil, &Nil);
    assert_eq!(empty.value(), 0);
}

// stride.hpp: `idx2crd(i, s, d)` yields `(i / d) % s`, with a static zero for `_1` modes.
#[test]
fn idx2crd_inverts_compact_layouts() {
    let layout = LayoutLeft::make(shape![Int::<2>, (Int::<3>, Int::<4>)]).unwrap();
    for i in 0..24i32 {
        let c = idx2crd(i, &layout.shape, &layout.stride);
        assert_eq!(layout.crd2idx(c), i);
    }
    assert_eq!(
        idx2crd(17i32, &layout.shape, &layout.stride),
        shape![1i32, (2i32, 2i32)]
    );
    // A static unit extent skips the division, so a zero stride is harmless.
    let unit: Int<0> = idx2crd(5i32, &Int::<1>, &Int::<0>);
    assert_eq!(unit.value(), 0);
    // A tuple index splits mode by mode.
    let per_mode = idx2crd(shape![5i32, 3i32], &layout.shape, &layout.stride);
    assert_eq!(per_mode, shape![1i32, (1i32, 0i32)]);
}

// stride.hpp: `crd2idx(c, s)` is Horner's colexicographic index.
#[test]
fn colexicographic_index_and_coordinate_round_trip() {
    let shape = shape![Int::<2>, (Int::<3>, Int::<4>)];
    assert_eq!(crd2idx_colex(shape![1i32, (2i32, 2i32)], &shape), 17);
    // An integer mode coordinate is already that mode's index.
    assert_eq!(crd2idx_colex(shape![1i32, 8i32], &shape), 17);
    assert_eq!(crd2idx_colex(17i32, &shape), 17);
    let static_index: Int<17> = crd2idx_colex(shape![Int::<1>, (Int::<2>, Int::<2>)], &shape);
    assert_eq!(static_index.value(), 17);

    assert_eq!(idx2crd_colex(17i32, &shape), shape![1i32, (2i32, 2i32)]);
    for i in 0..24i32 {
        assert_eq!(crd2idx_colex(idx2crd_colex(i, &shape), &shape), i);
    }
    // The last leaf is unbounded, matching the extended domain.
    assert_eq!(
        idx2crd_colex(30i32, &shape![Int::<2>, Int::<3>]),
        shape![0i32, 15i32]
    );
    // An integer shape returns the index unchanged.
    assert_eq!(idx2crd_colex(30i32, &Int::<4>), 30);
}

// stride.hpp: `crd2crd` maps mode by mode when all three are tuples.
#[test]
fn crd2crd_converts_between_shapes_of_equal_size() {
    let src = shape![Int::<2>, Int::<6>];
    let dst = shape![Int::<2>, (Int::<2>, Int::<3>)];
    assert_eq!(
        crd2crd(shape![1i32, 5i32], &src, &dst),
        shape![1i32, (1i32, 2i32)]
    );
    // An integer coordinate is decoded in the destination shape.
    assert_eq!(crd2crd(11i32, &src, &dst), shape![1i32, (1i32, 2i32)]);
    // An integer destination receives the colexicographic index.
    assert_eq!(crd2crd(shape![1i32, 5i32], &src, &Int::<12>), 11);
}

// int_tuple.hpp: `elem_less` stops at the rank of `a`.
#[test]
fn elem_less_compares_every_mode() {
    let extent = shape![Int::<4>, (Int::<2>, DynInt(3))];
    assert!(elem_less(&shape![3i32, (1i32, 2i32)], &extent));
    assert!(!elem_less(&shape![4i32, (1i32, 2i32)], &extent));
    assert!(!elem_less(&shape![3i32, (1i32, 3i32)], &extent));
    assert!(elem_less(&shape![1i32], &shape![2i32, 0i32]));
    assert!(!elem_less(&shape![1i32, 0i32], &shape![2i32]));
    assert!(elem_less(&Nil, &Nil));
}

// underscore.hpp: slice keeps `_` modes, dice keeps integer modes, nested results splice.
#[test]
fn slice_and_dice_select_complementary_modes() {
    let target = shape![(Int::<2>, Int::<3>), Int::<4>];
    assert_eq!(slice(X, &target), target);
    assert_eq!(slice(1i32, &target), Nil);
    assert_eq!(
        slice(shape![X, 1i32], &target),
        shape![(Int::<2>, Int::<3>)]
    );
    assert_eq!(
        slice(shape![(X, 0i32), X], &target),
        shape![Int::<2>, Int::<4>]
    );

    assert_eq!(dice(1i32, &target), target);
    assert_eq!(dice(X, &target), Nil);
    assert_eq!(dice(shape![X, 1i32], &target), shape![Int::<4>]);
    assert_eq!(dice(shape![(X, 0i32), X], &target), shape![Int::<3>]);
}

#[test]
fn slice_and_offset_splits_the_address() {
    // Fixing the second mode of (3,4):(1,3) at 2 leaves 3:1 at offset 6.
    let layout = Layout::new(shape![Int::<3>, Int::<4>], stride![Int::<1>, Int::<3>]).unwrap();
    let (column, offset) = layout.slice_and_offset(shape![X, 2i32]);
    assert_eq!(
        column,
        Layout {
            shape: shape![Int::<3>],
            stride: stride![Int::<1>]
        }
    );
    assert_eq!(offset, 6);
    for u in 0..3i32 {
        assert_eq!(
            offset + column.crd2idx(shape![u]),
            layout.crd2idx(shape![u, 2i32])
        );
    }
    let row = layout.dice(shape![X, 2i32]);
    assert_eq!(
        row,
        Layout {
            shape: shape![Int::<4>],
            stride: stride![Int::<3>]
        }
    );
    // Every `_` contributes a static zero to the offset.
    let (whole, origin) = layout.slice_and_offset(shape![X, X]);
    assert_eq!(whole, layout);
    let _: Int<0> = origin;
}

// A `Step` such as `(_1, X, _1)` projects a tiler of layouts onto the marked modes.
#[test]
fn step_projections_dice_tuples_of_layouts() {
    let tiler = IntTuple(
        Layout::new(Int::<8>, Int::<1>).unwrap(),
        IntTuple(
            Layout::new(Int::<16>, Int::<1>).unwrap(),
            IntTuple(Layout::new(Int::<4>, Int::<1>).unwrap(), Nil),
        ),
    );
    let mk = dice(shape![Int::<1>, X, Int::<1>], &tiler);
    assert_eq!(mk.0, tiler.0);
    assert_eq!(mk.1.0, tiler.1.1.0);
    let n = slice(shape![Int::<1>, X, Int::<1>], &tiler);
    assert_eq!(n.0, tiler.1.0);
}
