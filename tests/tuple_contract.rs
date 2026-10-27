//! Contract tests for heterogeneous integer tuples and their operations.
//!
//! These tests distinguish top-level modes from flattened leaves and verify
//! that tuple operations preserve leaf order, types, and static values.

use noodle_cute::int::{DynInt, Int};
use noodle_cute::tuple::{Append, Flatten, Item, Product, Tuple};
use noodle_cute::{Error, IntTuple, Nil, int_tuple};

#[test]
fn nil_is_the_empty_leaf_free_tuple() {
    assert_eq!(Nil::RANK, 0);
    assert_eq!(Nil::DEPTH, 1);
    assert_eq!(Nil::LEAVES, 0);
    assert_eq!(<Nil as Tuple>::LEN, 0);
    assert_eq!(<Nil as Tuple>::MAX_DEPTH, 0);
    assert_eq!(Nil.rank(), 0);
    assert_eq!(Nil.depth(), 1);

    // An empty tuple has neither modes nor leaves to index.
    assert_eq!(Nil.leaf(0), Err(Error::AxisOutOfBounds));
    assert_eq!(Nil.leaf_static(0), Err(Error::AxisOutOfBounds));
    assert_eq!(Nil.mode_span(0), Err(Error::AxisOutOfBounds));

    // The empty product is the multiplicative identity.
    assert_eq!(Nil.product(), Ok(1));
}

#[test]
fn bare_integers_behave_as_single_leaf_items() {
    // Static, wrapped dynamic, and native integers share the single-leaf contract.
    let (rank, depth, leaves) = (Int::<9>::RANK, Int::<9>::DEPTH, Int::<9>::LEAVES);
    assert_eq!((rank, depth, leaves), (1, 0, 1));
    assert_eq!(DynInt::RANK, 1);
    assert_eq!(DynInt::DEPTH, 0);
    assert_eq!(DynInt::LEAVES, 1);
    assert_eq!(<isize as Item>::RANK, 1);

    let leaf = Int::<9>;
    assert_eq!(leaf.leaf(0), Ok(9));
    assert_eq!(leaf.leaf(1), Err(Error::AxisOutOfBounds));
    assert_eq!(leaf.leaf_static(0), Ok(Some(9)));
    assert_eq!(leaf.mode_span(0), Ok((0, 1)));
    assert_eq!(leaf.mode_span(1), Err(Error::AxisOutOfBounds));
    assert_eq!(leaf.product(), Ok(9));

    let dyn_leaf = DynInt(9);
    assert_eq!(dyn_leaf.leaf(0), Ok(9));
    assert_eq!(dyn_leaf.leaf_static(0), Ok(None));

    let isize_leaf = 9isize;
    assert_eq!(Item::leaf(&isize_leaf, 0), Ok(9));
    assert_eq!(Item::leaf_static(&isize_leaf, 0), Ok(None));
}

#[test]
fn flat_int_tuple_reports_rank_depth_and_leaves() {
    // Adding modes to a flat tuple increases its rank without increasing its depth.
    let flat = int_tuple![Int::<2>, Int::<3>, DynInt(4)];
    assert_eq!(flat.rank(), 3);
    assert_eq!(flat.depth(), 1);
    assert_eq!(
        IntTuple::<Int<2>, IntTuple<Int<3>, IntTuple<DynInt, Nil>>>::LEAVES,
        3
    );
    assert_eq!(flat.leaf(0), Ok(2));
    assert_eq!(flat.leaf(1), Ok(3));
    assert_eq!(flat.leaf(2), Ok(4));
    assert_eq!(flat.leaf(3), Err(Error::AxisOutOfBounds));
    assert_eq!(flat.leaf_static(0), Ok(Some(2)));
    assert_eq!(flat.leaf_static(2), Ok(None));
    assert_eq!(flat.mode_span(0), Ok((0, 1)));
    assert_eq!(flat.mode_span(1), Ok((1, 1)));
    assert_eq!(flat.mode_span(2), Ok((2, 1)));
    assert_eq!(flat.mode_span(3), Err(Error::AxisOutOfBounds));
}

#[test]
fn single_element_tuple_matches_the_bare_leaf_contract() {
    // Wrapping one leaf adds a tuple level while preserving its value and rank.
    let singleton = int_tuple![Int::<5>];
    assert_eq!(singleton.rank(), 1);
    assert_eq!(singleton.depth(), 1);
    assert_eq!(singleton.leaf(0), Ok(5));
    assert_eq!(singleton, IntTuple(Int::<5>, Nil));
}

#[test]
fn nested_head_increases_depth_but_not_the_trailing_chain() {
    // A parenthesized group adds nesting while remaining one top-level mode.
    let nested = int_tuple![Int::<2>, (Int::<3>, Int::<4>)];
    assert_eq!(nested.rank(), 2);
    assert_eq!(nested.depth(), 2);
    assert_eq!(nested.leaf(0), Ok(2));
    assert_eq!(nested.leaf(1), Ok(3));
    assert_eq!(nested.leaf(2), Ok(4));
    assert_eq!(nested.mode_span(0), Ok((0, 1)));
    assert_eq!(nested.mode_span(1), Ok((1, 2)));
    assert_eq!(nested.product(), Ok(24));
}

#[test]
fn product_short_circuits_on_a_zero_leaf_before_overflowing() {
    // A zero makes the product zero even when the other factors would overflow.
    let has_zero = int_tuple![DynInt(isize::MAX), DynInt(2), DynInt(0)];
    assert_eq!(has_zero.product(), Ok(0));
}

#[test]
fn product_reports_overflow_for_genuinely_too_large_results() {
    let overflowing = int_tuple![DynInt(isize::MAX), DynInt(2)];
    assert_eq!(overflowing.product(), Err(Error::Overflow));
}

#[test]
fn get_reads_a_top_level_mode_without_expanding_nested_heads() {
    // Access a nested mode first, then index its children with another `get` call.
    let nested = int_tuple![Int::<2>, (Int::<3>, Int::<4>)];
    let _: &Int<2> = nested.get::<0>();
    let second_mode: &IntTuple<Int<3>, IntTuple<Int<4>, Nil>> = nested.get::<1>();
    assert_eq!(second_mode.get::<0>().value(), 3);
    assert_eq!(second_mode.get::<1>().value(), 4);
}

#[test]
fn append_preserves_concrete_leaf_types_on_both_sides() {
    assert_eq!(Nil.append(5isize), 5isize);
    let lhs = int_tuple![Int::<1>, Int::<2>];
    let rhs = int_tuple![Int::<3>];
    let joined = lhs.append(rhs);
    assert_eq!(joined, int_tuple![Int::<1>, Int::<2>, Int::<3>]);
    assert_eq!(joined.rank(), 3);
}

#[test]
fn flatten_removes_nesting_while_keeping_the_same_leaves() {
    let nested = int_tuple![Int::<2>, (Int::<3>, Int::<4>), DynInt(5)];
    assert_eq!(nested.rank(), 3);
    assert_eq!(nested.depth(), 2);
    let flat = nested.flatten();
    // Flattening makes each leaf a top-level mode and preserves leaf order.
    assert_eq!(flat.rank(), 4);
    assert_eq!(flat.depth(), 1);
    assert_eq!(
        (flat.leaf(0), flat.leaf(1), flat.leaf(2), flat.leaf(3)),
        (Ok(2), Ok(3), Ok(4), Ok(5))
    );

    // A bare leaf becomes a singleton tuple; an empty tuple stays empty.
    let leaf_flat = Int::<7>.flatten();
    assert_eq!(leaf_flat, IntTuple(Int::<7>, Nil));
    assert_eq!(Nil.flatten(), Nil);
}

#[test]
fn product_typed_keeps_a_fully_static_tuple_as_an_int() {
    // The output type encodes the product when every leaf is static.
    let shape = int_tuple![Int::<2>, (Int::<3>, Int::<4>)];
    let total: Int<24> = shape.product_typed();
    assert_eq!(total.value(), 24);

    // The empty product also preserves its value in the type.
    let empty: Int<1> = Nil.product_typed();
    assert_eq!(empty.value(), 1);
}

#[test]
fn product_typed_degrades_to_dynint_once_any_leaf_is_dynamic() {
    let mixed = int_tuple![Int::<2>, DynInt(3), Int::<4>];
    let total: DynInt = mixed.product_typed();
    assert_eq!(total, DynInt(24));
}

#[test]
fn int_tuple_macro_parenthesized_groups_build_nested_structure_explicitly() {
    // Parenthesized groups must expand to the same structure as nested constructors.
    let via_macro = int_tuple![Int::<1>, (Int::<2>, Int::<3>)];
    let by_hand = IntTuple(
        Int::<1>,
        IntTuple(IntTuple(Int::<2>, IntTuple(Int::<3>, Nil)), Nil),
    );
    assert_eq!(via_macro, by_hand);
    assert_eq!(int_tuple![], Nil);
}
