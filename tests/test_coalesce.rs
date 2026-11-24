#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use noodle_cute::algebra::coalesce::{
    Coalesce, CoalesceExtended, CoalescePrepend, DropTrailingUnit, PrependAction, coalesce,
    coalesce_extended,
};
use noodle_cute::basis::{BasisCombination, Mode, ScaledBasis};
use noodle_cute::integer::Int;
use noodle_cute::layout::Layout;
use noodle_cute::tuple::{LeafProfile, Nil, NilProfile, Til, TupleProfile};
use noodle_cute::{e, int_tuple, tuple};

type PairLayout<S0, S1, D0, D1> = Layout<Til<S0, Til<S1, Nil>>, Til<D0, Til<D1, Nil>>>;

#[test]
fn test_coalesce_simple_merge() {
    // Layout((2, 4), (1, 2)) -> Layout(8, 1)
    let l = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        int_tuple![Int::<1>, Int::<2>],
    );
    assert_eq!(l.coalesce(), Layout::new(Int::<8>, Int::<1>));
    assert_eq!(coalesce(l), Layout::new(Int::<8>, Int::<1>));
}

#[test]
fn test_coalesce_cannot_merge() {
    // Layout((2, 4), (4, 1)) -> Layout((2, 4), (4, 1))
    let l = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        int_tuple![Int::<4>, Int::<1>],
    );
    assert_eq!(
        l.coalesce(),
        Layout::new(
            int_tuple![Int::<2>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>]
        )
    );
}

#[test]
fn test_coalesce_with_size_one_in_middle() {
    // Layout((2, 1, 6), (1, 7, 2)) -> Layout(12, 1)
    let l = Layout::new(
        int_tuple![Int::<2>, Int::<1>, Int::<6>],
        int_tuple![Int::<1>, Int::<7>, Int::<2>],
    );
    assert_eq!(l.coalesce(), Layout::new(Int::<12>, Int::<1>));
}

#[test]
fn test_coalesce_trailing_size_one_dropped() {
    // Layout((2, 1), (1, 0)) -> Layout(2, 1)
    let l = Layout::new(
        int_tuple![Int::<2>, Int::<1>],
        int_tuple![Int::<1>, Int::<0>],
    );
    assert_eq!(l.coalesce(), Layout::new(Int::<2>, Int::<1>));
}

#[test]
fn test_coalesce_const_eval() {
    const L: PairLayout<Int<2>, Int<4>, Int<1>, Int<2>> = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        int_tuple![Int::<1>, Int::<2>],
    );
    const C: Layout<Int<8>, Int<1>> = L.coalesce();
    const C_FREE: Layout<Int<8>, Int<1>> = coalesce(L);
    assert_eq!(C, Layout::new(Int::<8>, Int::<1>));
    assert_eq!(C_FREE, C);
}

#[test]
fn test_coalesce_scaled_basis_merge() {
    // Layout((2, 4), (1@0, 2@0)) -> Layout(8, 1@0)
    let l = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        tuple![
            ScaledBasis::<Int<1>, Mode<0>>::new(Int::<1>),
            ScaledBasis::<Int<2>, Mode<0>>::new(Int::<2>),
        ],
    );
    assert_eq!(
        l.coalesce(),
        Layout::new(Int::<8>, ScaledBasis::<Int<1>, Mode<0>>::new(Int::<1>))
    );
}

#[test]
fn test_coalesce_scaled_basis_cannot_merge_different_modes() {
    // Layout((2, 4), (1@0, 2@1)) -> Layout((2, 4), (1@0, 2@1))
    let l = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        tuple![
            ScaledBasis::<Int<1>, Mode<0>>::new(Int::<1>),
            ScaledBasis::<Int<2>, Mode<1>>::new(Int::<2>),
        ],
    );
    assert_eq!(l.coalesce(), l);

    // Nested paths with same outer index but distinct inner index must not merge.
    let nested_diff = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        tuple![
            ScaledBasis::<Int<1>, Mode<1, Mode<0>>>::new(Int::<1>),
            ScaledBasis::<Int<2>, Mode<1, Mode<1>>>::new(Int::<2>),
        ],
    );
    assert_eq!(nested_diff.coalesce(), nested_diff);

    // Nested paths with identical full path merge.
    let nested_same = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        tuple![
            ScaledBasis::<Int<1>, Mode<1, Mode<0>>>::new(Int::<1>),
            ScaledBasis::<Int<2>, Mode<1, Mode<0>>>::new(Int::<2>),
        ],
    );
    assert_eq!(
        nested_same.coalesce(),
        Layout::new(
            Int::<8>,
            ScaledBasis::<Int<1>, Mode<1, Mode<0>>>::new(Int::<1>)
        )
    );
}

#[test]
fn test_coalesce_mixed_int_and_scaled_basis() {
    // Layout((2, 4), (1, 2@0)) -> cannot merge across scalar int and ScaledBasis
    let l = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        tuple![Int::<1>, ScaledBasis::<Int<2>, Mode<0>>::new(Int::<2>)],
    );
    assert_eq!(l.coalesce(), l);
}

#[test]
fn test_coalesce_extended_empty_layout() {
    let empty = Layout::new(Nil, Nil);
    assert_eq!(empty.coalesce(), Layout::new(Nil, Nil));
    assert_eq!(empty.coalesce_extended(), Layout::new(Nil, Nil));
    assert_eq!(coalesce_extended(empty), Layout::new(Nil, Nil));
}

#[test]
fn test_coalesce_extended_single_size_one() {
    // Layout(1, 0) in extended mode preserves 1:0 instead of collapsing to Nil
    let l = Layout::new(Int::<1>, Int::<0>);
    assert_eq!(l.coalesce_extended(), Layout::new(Int::<1>, Int::<0>));
}

#[test]
fn test_coalesce_extended_trailing_size_one_preserved() {
    let l = Layout::new(
        int_tuple![Int::<2>, Int::<1>],
        int_tuple![Int::<1>, Int::<0>],
    );
    assert_eq!(l.coalesce(), Layout::new(Int::<2>, Int::<1>));
    assert_eq!(l.coalesce_extended(), l);
}

#[test]
fn test_coalesce_extended_middle_size_one_eliminated() {
    // Layout((2, 1, 6), (1, 7, 2)) -> Layout(12, 1)
    let l = Layout::new(
        int_tuple![Int::<2>, Int::<1>, Int::<6>],
        int_tuple![Int::<1>, Int::<7>, Int::<2>],
    );
    assert_eq!(l.coalesce_extended(), Layout::new(Int::<12>, Int::<1>));
}

#[test]
fn test_coalesce_extended_multiple_size_ones() {
    // Layout((1, 1), (5, 7)) -> Layout(1, 7)
    let l = Layout::new(
        int_tuple![Int::<1>, Int::<1>],
        int_tuple![Int::<5>, Int::<7>],
    );
    assert_eq!(l.coalesce_extended(), Layout::new(Int::<1>, Int::<7>));
}

#[test]
fn test_coalesce_extended_const_eval() {
    const EMPTY: Layout<Nil, Nil> = Layout::new(Nil, Nil);
    const EXT: Layout<Nil, Nil> = EMPTY.coalesce_extended();
    assert_eq!(EXT, Layout::new(Nil, Nil));

    const L: PairLayout<Int<2>, Int<4>, Int<1>, Int<2>> = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        int_tuple![Int::<1>, Int::<2>],
    );
    const C: Layout<Int<8>, Int<1>> = L.coalesce_extended();
    const C_FREE: Layout<Int<8>, Int<1>> = coalesce_extended(L);
    assert_eq!(C, Layout::new(Int::<8>, Int::<1>));
    assert_eq!(C_FREE, C);
}

#[test]
fn test_coalesce_trait_modes_and_scalar_base_cases() {
    const UNIT: Layout<Int<1>, Int<7>> = Layout::new(Int::<1>, Int::<7>);
    const STANDARD: Layout<Nil, Nil> = Coalesce::<false>::coalesce(UNIT);
    const EXTENDED: Layout<Int<1>, Int<7>> = Coalesce::<true>::coalesce(UNIT);
    assert_eq!(STANDARD, Layout::new(Nil, Nil));
    assert_eq!(EXTENDED, UNIT);

    // A scalar seeds an empty suffix even if its zero stride could self-merge.
    let scalar = Layout::new(Int::<4>, Int::<0>);
    assert_eq!(scalar.coalesce(), scalar);
    assert_eq!(scalar.coalesce_extended(), scalar);
}

#[test]
fn test_prepend_actions_const_eval() {
    const MERGED: Layout<Int<8>, Int<1>> = <Layout<Int<4>, Int<2>> as CoalescePrepend<
        Int<2>,
        Int<1>,
        { PrependAction::Merge },
    >>::prepend(
        Layout::new(Int::<4>, Int::<2>), Int::<2>, Int::<1>
    );
    const DISCARDED: Layout<Int<8>, Int<1>> = <Layout<Int<8>, Int<1>> as CoalescePrepend<
        Int<1>,
        Int<99>,
        { PrependAction::Discard },
    >>::prepend(MERGED, Int::<1>, Int::<99>);
    assert_eq!(MERGED, Layout::new(Int::<8>, Int::<1>));
    assert_eq!(DISCARDED, MERGED);
}

#[test]
fn test_coalesce_actions_on_tuple_suffix() {
    // Retain several modes, merge into a tuple head, then discard a leading unit.
    let layout = Layout::new(
        int_tuple![Int::<1>, Int::<2>, Int::<3>, Int::<4>, Int::<5>],
        int_tuple![Int::<7>, Int::<1>, Int::<2>, Int::<99>, Int::<1>],
    );
    assert_eq!(
        layout.coalesce(),
        Layout::new(
            int_tuple![Int::<6>, Int::<4>, Int::<5>],
            int_tuple![Int::<1>, Int::<99>, Int::<1>],
        ),
    );
}

#[test]
fn test_coalesce_negative_strides() {
    let layout = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        int_tuple![Int::<-1>, Int::<-2>],
    );
    assert_eq!(layout.coalesce(), Layout::new(Int::<8>, Int::<-1>));

    let basis_layout = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        tuple![
            ScaledBasis::<Int<-1>, Mode<0>>::new(Int::<-1>),
            ScaledBasis::<Int<-2>, Mode<0>>::new(Int::<-2>),
        ],
    );
    assert_eq!(
        basis_layout.coalesce(),
        Layout::new(Int::<8>, ScaledBasis::<Int<-1>, Mode<0>>::new(Int::<-1>)),
    );
}

#[test]
fn test_coalesce_large_signed_boundary_and_non_overflowing_comparison() {
    let near_max = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        int_tuple![Int::<1_073_741_823>, Int::<2_147_483_646>],
    );
    assert_eq!(
        near_max.coalesce(),
        Layout::new(Int::<8>, Int::<1_073_741_823>),
    );

    let near_min = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        int_tuple![Int::<-1_073_741_824>, Int::<-2_147_483_648>],
    );
    assert_eq!(
        near_min.coalesce(),
        Layout::new(Int::<8>, Int::<-1_073_741_824>),
    );

    let different_signs = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        int_tuple![Int::<-1>, Int::<2>],
    );
    assert_eq!(different_signs.coalesce(), different_signs);

    // Overflowing static comparison d_a * s_a != d_b must safely retain without panicking.
    let overflow_cmp = Layout::new(
        int_tuple![Int::<2_147_483_647>, Int::<2>],
        int_tuple![Int::<2_147_483_647>, Int::<1>],
    );
    assert_eq!(overflow_cmp.coalesce(), overflow_cmp);
}

#[test]
fn test_coalesce_dynamic_facts_are_retained_in_const_context() {
    // Const evaluation does not turn a primitive value into a static type.
    const SHAPE: Layout<Til<i32, Nil>, Til<Int<1>, Nil>> =
        Layout::new(int_tuple![1i32], int_tuple![Int::<1>]);
    const RESULT: Layout<i32, Int<1>> = SHAPE.coalesce();
    assert_eq!(RESULT, Layout::new(1i32, Int::<1>));

    let strides = Layout::new(int_tuple![Int::<2>, Int::<4>], int_tuple![1i32, Int::<2>]);
    assert_eq!(strides.coalesce(), strides);

    let basis = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        tuple![
            ScaledBasis::<i32, Mode<0>>::new(1),
            ScaledBasis::<Int<2>, Mode<0>>::new(Int::<2>)
        ],
    );
    assert_eq!(basis.coalesce(), basis);
}

#[test]
fn test_coalesce_static_strides_can_merge_a_dynamic_suffix_extent() {
    let layout = Layout::new(int_tuple![Int::<2>, 3i32], int_tuple![Int::<1>, Int::<2>]);
    assert_eq!(layout.coalesce(), Layout::new(6i32, Int::<1>));
}

#[test]
fn test_coalesce_discards_static_unit_with_dynamic_stride() {
    let layout = Layout::new(
        int_tuple![Int::<2>, Int::<1>, Int::<4>],
        int_tuple![Int::<1>, 99i32, Int::<2>],
    );
    assert_eq!(layout.coalesce(), Layout::new(Int::<8>, Int::<1>));
}

#[test]
fn test_layout_coalesce_flattens_before_const_eval() {
    const RESULT: Layout<Int<12>, Int<1>> = Layout::new(
        int_tuple![Int::<2>, (Int::<1>, Int::<6>)],
        int_tuple![Int::<1>, (Int::<7>, Int::<2>)],
    )
    .coalesce();
    assert_eq!(RESULT, Layout::new(Int::<12>, Int::<1>));
}

#[test]
fn test_layout_coalesce_flattens_basis_leaves() {
    const RESULT: Layout<Int<24>, ScaledBasis<Int<1>, Mode<0>>> = Layout::new(
        int_tuple![(Int::<2>, Int::<3>), Int::<4>],
        tuple![
            (
                ScaledBasis::<Int<1>, Mode<0>>::new(Int::<1>),
                ScaledBasis::<Int<2>, Mode<0>>::new(Int::<2>)
            ),
            ScaledBasis::<Int<6>, Mode<0>>::new(Int::<6>),
        ],
    )
    .coalesce();
    assert_eq!(
        RESULT,
        Layout::new(Int::<24>, ScaledBasis::<Int<1>, Mode<0>>::new(Int::<1>))
    );
}

#[test]
fn test_layout_coalesce_flattens_empty_modes() {
    let layout = Layout::new(
        int_tuple![(), ((), Int::<2>), ()],
        int_tuple![(), ((), Int::<1>), ()],
    );
    assert_eq!(layout.coalesce(), Layout::new(Int::<2>, Int::<1>));
    assert_eq!(layout.coalesce_extended(), Layout::new(Int::<2>, Int::<1>));
}

#[test]
fn test_layout_coalesce_flattens_unit_modes() {
    let layout = Layout::new(
        int_tuple![Int::<1>, (Int::<1>, Int::<1>), ()],
        int_tuple![Int::<3>, (Int::<5>, Int::<7>), ()],
    );
    assert_eq!(layout.coalesce(), Layout::new(Nil, Nil));
    assert_eq!(layout.coalesce_extended(), Layout::new(Int::<1>, Int::<7>));
}

#[test]
fn test_coalesce_zero_extent_and_zero_strides() {
    let zero_extent = Layout::new(
        int_tuple![Int::<0>, Int::<4>],
        int_tuple![Int::<1>, Int::<0>],
    );
    assert_eq!(zero_extent.coalesce(), Layout::new(Int::<0>, Int::<1>));

    let zero_strides = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        int_tuple![Int::<0>, Int::<0>],
    );
    assert_eq!(zero_strides.coalesce(), Layout::new(Int::<8>, Int::<0>));
}

#[test]
fn test_coalesce_stride_zero_with_trailing_unit() {
    let l = Layout::new(
        int_tuple![Int::<2>, Int::<1>],
        int_tuple![Int::<0>, Int::<0>],
    );
    assert_eq!(l.coalesce(), Layout::new(Int::<2>, Int::<0>));
}

#[test]
fn test_coalesce_profile_leaf_flattens_and_preserves_trailing_unit() {
    type ResultShape = Til<Int<12>, Til<Int<1>, Nil>>;
    type ResultStride = Til<Int<1>, Til<Int<0>, Nil>>;

    const RESULT: Layout<ResultShape, ResultStride> =
        CoalesceExtended::<LeafProfile>::coalesce_z(Layout::new(
            int_tuple![Int::<2>, (Int::<1>, Int::<6>), Int::<1>],
            int_tuple![Int::<1>, (Int::<7>, Int::<2>), Int::<0>],
        ));

    assert_eq!(
        RESULT,
        Layout::new(
            int_tuple![Int::<12>, Int::<1>],
            int_tuple![Int::<1>, Int::<0>],
        )
    );
}

#[test]
fn test_coalesce_profile_preserves_mode_boundaries() {
    type Profile = TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>;
    type ResultShape = Til<Int<6>, Til<Int<4>, Nil>>;
    type ResultStride = Til<Int<1>, Til<Int<6>, Nil>>;

    const RESULT: Layout<ResultShape, ResultStride> =
        CoalesceExtended::<Profile>::coalesce_z(Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<4>],
            int_tuple![(Int::<1>, Int::<2>), Int::<6>],
        ));

    assert_eq!(
        RESULT,
        Layout::new(
            int_tuple![Int::<6>, Int::<4>],
            int_tuple![Int::<1>, Int::<6>],
        )
    );
}

#[test]
fn test_coalesce_profile_nested_and_empty() {
    type InnerProfile = TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>;
    type Profile = TupleProfile<InnerProfile, TupleProfile<LeafProfile, NilProfile>>;

    let result = CoalesceExtended::<Profile>::coalesce_z(Layout::new(
        int_tuple![((Int::<2>, Int::<3>), (Int::<4>, Int::<5>)), Int::<7>],
        int_tuple![((Int::<1>, Int::<2>), (Int::<6>, Int::<24>)), Int::<120>],
    ));
    assert_eq!(
        result,
        Layout::new(
            int_tuple![(Int::<6>, Int::<20>), Int::<7>],
            int_tuple![(Int::<1>, Int::<6>), Int::<120>],
        )
    );

    const EMPTY: Layout<Nil, Nil> =
        CoalesceExtended::<NilProfile>::coalesce_z(Layout::new(Nil, Nil));
    assert_eq!(EMPTY, Layout::new(Nil, Nil));
}

#[test]
fn test_coalesce_overflow_negative_and_drop_trailing_unit() {
    // Coalesce catches dynamic extent merge overflow and negative extents
    let ok = Layout::new(int_tuple![Int::<2>, 5i32], int_tuple![Int::<1>, Int::<2>]);
    assert_eq!(ok.coalesce(), Layout::new(10i32, Int::<1>));
    assert_eq!(ok.coalesce_extended(), Layout::new(10i32, Int::<1>));

    let overflow = Layout::new(
        int_tuple![Int::<2>, 2_000_000_000i32],
        int_tuple![Int::<1>, Int::<2>],
    );
    assert!(std::panic::catch_unwind(|| overflow.coalesce()).is_err());
    assert!(std::panic::catch_unwind(|| overflow.coalesce_extended()).is_err());

    let negative = Layout::new(-3i32, Int::<1>);
    assert!(std::panic::catch_unwind(|| negative.coalesce()).is_err());

    // DropTrailingUnit preserves standalone `1:d`, drops trailing `1` for rank >= 2
    assert_eq!(
        Layout::new(Int::<1>, Int::<4>).drop_trailing_unit(),
        Layout::new(Int::<1>, Int::<4>)
    );
    assert_eq!(
        Layout::new(int_tuple![Int::<1>], int_tuple![Int::<4>]).drop_trailing_unit(),
        Layout::new(Int::<1>, Int::<4>)
    );
    assert_eq!(
        Layout::new(int_tuple![Int::<2>, Int::<1>], int_tuple![Int::<1>, Int::<8>])
            .drop_trailing_unit(),
        Layout::new(Int::<2>, Int::<1>)
    );
    assert_eq!(
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<1>],
            int_tuple![Int::<1>, Int::<4>, Int::<24>],
        )
        .drop_trailing_unit(),
        Layout::new(int_tuple![Int::<2>, Int::<3>], int_tuple![Int::<1>, Int::<4>])
    );
    assert_eq!(
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>, Int::<1>],
            int_tuple![Int::<1>, Int::<4>, Int::<24>, Int::<100>],
        )
        .drop_trailing_unit(),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            int_tuple![Int::<1>, Int::<4>, Int::<24>],
        )
    );
    assert_eq!(
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            int_tuple![Int::<1>, Int::<4>, Int::<24>],
        )
        .drop_trailing_unit(),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            int_tuple![Int::<1>, Int::<4>, Int::<24>],
        )
    );
}

#[test]
fn test_coalesce_basis_combination_strides() {
    // Matching multi-term BasisCombination strides merge
    let mergeable = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        tuple![e![0] + e![1], Int::<2> * e![0] + Int::<2> * e![1]],
    );
    assert_eq!(mergeable.coalesce(), Layout::new(Int::<8>, e![0] + e![1]));

    // Non-matching multi-term BasisCombination strides are retained
    let mismatch = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        tuple![e![0] + e![1], Int::<2> * e![0] + Int::<3> * e![1]],
    );
    assert_eq!(mismatch.coalesce(), mismatch);

    // Trailing unit mode dropped after multi-term BasisCombination
    let trailing_unit = Layout::new(
        int_tuple![Int::<2>, Int::<1>],
        tuple![e![0] + e![1], e![2]],
    );
    assert_eq!(
        trailing_unit.coalesce(),
        Layout::new(Int::<2>, e![0] + e![1])
    );

    // Zero extent with BasisCombination followed by canonical zero merges
    let zero_ext = Layout::new(
        int_tuple![Int::<0>, Int::<4>],
        tuple![e![0] + e![1], BasisCombination::<Nil>::zero()],
    );
    assert_eq!(zero_ext.coalesce(), Layout::new(Int::<0>, e![0] + e![1]));
}
