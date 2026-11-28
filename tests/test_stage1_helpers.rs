#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use std::panic::catch_unwind;

use noodle_cute::algebra::{
    MaxCommonVector, blocked_product, coalesce, compose_tile, composition, flat_divide,
    flat_product, logical_divide, logical_product, make_fragment_like, max_common_layout,
    max_common_vector, raked_product, tile_to_shape, tiled_divide, tiled_product, zipped_divide,
    zipped_product,
};
use noodle_cute::f2::f2;
use noodle_cute::integer::{Int, int};
use noodle_cute::layout::{Layout, get_flat_coord, make_identity_layout, recast_layout};
use noodle_cute::shape::{crd2crd, elem_less};
use noodle_cute::stride::{idx2crd_with_stride, suffix_product, suffix_product_with};
use noodle_cute::swizzle::{
    StaticSwizzle, Swizzle, compose_swizzle, downcast_swizzle, make_swizzle, recast_swizzle,
    swizzle_params, upcast_swizzle,
};
use noodle_cute::tuple::{Nil, Til, group, pad_modes, repeat_modes, zip2};
use noodle_cute::{e, int_tuple, mode, static_value, tuple};

type Pair<A, B> = Til<A, Til<B, Nil>>;
type Triple<A, B, C> = Til<A, Pair<B, C>>;
type Quad<A, B, C, D> = Til<A, Triple<B, C, D>>;

#[test]
fn test_tuple_group_zip2_pad_and_repeat_modes() {
    // group::<B, E>
    const T4: Quad<Int<1>, Int<2>, Int<3>, Int<4>> =
        int_tuple![Int::<1>, Int::<2>, Int::<3>, Int::<4>];
    const G_MID: Triple<Int<1>, Pair<Int<2>, Int<3>>, Int<4>> = group::<1, 3, _>(&T4);
    assert_eq!(G_MID, int_tuple![Int::<1>, (Int::<2>, Int::<3>), Int::<4>]);
    assert_eq!(
        T4.group::<0, 2>(),
        int_tuple![(Int::<1>, Int::<2>), Int::<3>, Int::<4>]
    );
    assert_eq!(
        T4.group::<2, 4>(),
        int_tuple![Int::<1>, Int::<2>, (Int::<3>, Int::<4>)]
    );
    assert_eq!(
        T4.group::<0, 4>(),
        tuple![int_tuple![Int::<1>, Int::<2>, Int::<3>, Int::<4>]]
    );
    // Single-element group wraps that mode in a 1-tuple
    assert_eq!(
        T4.group::<1, 2>(),
        int_tuple![Int::<1>, (Int::<2>,), Int::<3>, Int::<4>]
    );
    // Empty range (B == E) leaves tuple unchanged
    assert_eq!(T4.group::<2, 2>(), T4);
    assert_eq!(group::<0, 0, _>(&Nil), Nil);

    // zip2 on tuples and layouts
    const PAIRS: Pair<Pair<Int<1>, Int<2>>, Pair<Int<3>, Int<4>>> =
        int_tuple![(Int::<1>, Int::<2>), (Int::<3>, Int::<4>)];
    const ZIPPED: Pair<Pair<Int<1>, Int<3>>, Pair<Int<2>, Int<4>>> = zip2(PAIRS);
    assert_eq!(
        ZIPPED,
        int_tuple![(Int::<1>, Int::<3>), (Int::<2>, Int::<4>)]
    );
    assert_eq!(zip2(Nil), tuple![Nil, Nil]);
    assert_eq!(
        zip2(int_tuple![
            (Int::<1>, Int::<2>),
            (Int::<3>, Int::<4>),
            (Int::<5>, Int::<6>)
        ]),
        int_tuple![(Int::<1>, Int::<3>, Int::<5>), (Int::<2>, Int::<4>, Int::<6>)]
    );

    let pair_layout = Layout::new(
        int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<4>)],
        int_tuple![(Int::<1>, Int::<4>), (Int::<2>, Int::<8>)],
    );
    assert_eq!(
        pair_layout.zip2(),
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<4>)],
            int_tuple![(Int::<1>, Int::<2>), (Int::<4>, Int::<8>)],
        )
    );

    // pad_modes
    let short = int_tuple![Int::<2>];
    let target3 = int_tuple![Int::<1>, Int::<1>, Int::<1>];
    assert_eq!(
        pad_modes(short, &target3, Int::<0>),
        int_tuple![Int::<2>, Int::<0>, Int::<0>]
    );
    assert_eq!(pad_modes(target3, &short, Int::<9>), target3);
    assert_eq!(pad_modes(Nil, &short, Int::<1>), int_tuple![Int::<1>]);

    // repeat_modes
    let nested = int_tuple![Int::<2>, (Int::<3>, Int::<4>), Int::<5>];
    assert_eq!(
        repeat_modes(Int::<1>, &nested),
        int_tuple![Int::<1>, Int::<1>, Int::<1>]
    );
    assert_eq!(repeat_modes(Int::<7>, &Int::<4>), Int::<7>);
    assert_eq!(repeat_modes(Int::<1>, &Nil), Nil);
}

#[test]
fn test_shape_crd2crd_and_elem_less() {
    let s = int_tuple![(Int::<2>, Int::<2>), Int::<3>];
    let flat_dst = int_tuple![Int::<4>, Int::<3>];

    // Nested -> flat mode-wise (same rank)
    assert_eq!(
        crd2crd(int_tuple![(1i32, 1i32), 2i32], &s, &flat_dst),
        int_tuple![3i32, 2i32]
    );
    // Flat -> nested mode-wise (same rank)
    assert_eq!(
        crd2crd(int_tuple![3i32, 2i32], &flat_dst, &s),
        int_tuple![(1i32, 1i32), 2i32]
    );
    // Scalar -> nested
    const SCALAR_TO_NESTED: Til<Til<Int<1>, Til<Int<1>, Nil>>, Til<Int<2>, Nil>> = crd2crd(
        Int::<11>,
        &Int::<12>,
        &int_tuple![(Int::<2>, Int::<2>), Int::<3>],
    );
    assert_eq!(SCALAR_TO_NESTED, int_tuple![(Int::<1>, Int::<1>), Int::<2>]);
    // Nested -> scalar
    assert_eq!(
        crd2crd(int_tuple![(1i32, 1i32), 2i32], &s, &Int::<12>),
        11i32
    );
    // Different-rank tuple -> tuple via flat linear index
    assert_eq!(
        crd2crd(
            int_tuple![1i32, 2i32],
            &int_tuple![Int::<2>, Int::<3>],
            &int_tuple![Int::<3>, Int::<2>, Int::<1>],
        ),
        int_tuple![2i32, 1i32, 0i32]
    );

    // elem_less
    assert!(elem_less(&2i32, &Int::<4>));
    assert!(!elem_less(&4i32, &Int::<4>));
    assert!(elem_less(
        &int_tuple![1i32, 2i32],
        &int_tuple![Int::<2>, Int::<3>]
    ));
    assert!(!elem_less(
        &int_tuple![2i32, 2i32],
        &int_tuple![Int::<2>, Int::<3>]
    ));
    assert!(elem_less(
        &int_tuple![(1i32, 0i32), 2i32],
        &int_tuple![(Int::<2>, Int::<2>), Int::<3>]
    ));
    assert!(!elem_less(
        &int_tuple![(1i32, 2i32), 2i32],
        &int_tuple![(Int::<2>, Int::<2>), Int::<3>]
    ));
    assert!(elem_less(&Nil, &Nil));
}

#[test]
fn test_stride_suffix_product_idx2crd_and_layout_helpers() {
    // suffix_product & overflow-safe leading leaf
    const RIGHT: Til<Int<12>, Til<Int<4>, Til<Int<1>, Nil>>> =
        suffix_product(int_tuple![Int::<2>, Int::<3>, Int::<4>]);
    assert_eq!(RIGHT, int_tuple![Int::<12>, Int::<4>, Int::<1>]);
    assert_eq!(
        suffix_product_with(int_tuple![Int::<2>, Int::<3>], Int::<2>),
        int_tuple![Int::<6>, Int::<2>]
    );
    assert_eq!(
        suffix_product(int_tuple![(Int::<2>, Int::<3>), Int::<4>]),
        int_tuple![(Int::<12>, Int::<4>), Int::<1>]
    );
    // Leading leaf i32::MAX must NOT overflow because its extent is never multiplied in
    const MAX_I32: int = int::MAX;
    assert_eq!(
        suffix_product(int_tuple![Int::<MAX_I32>, Int::<4>]),
        int_tuple![Int::<4>, Int::<1>]
    );

    // Layout::from_shape_right
    const ROW: Layout<Pair<Int<3>, Int<4>>, Pair<Int<4>, Int<1>>> =
        Layout::from_shape_right(int_tuple![Int::<3>, Int::<4>]);
    assert_eq!(
        ROW,
        Layout::new(
            int_tuple![Int::<3>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>]
        )
    );
    assert_eq!(
        Layout::from_shape_right_with(int_tuple![Int::<3>, Int::<4>], Int::<2>),
        Layout::new(
            int_tuple![Int::<3>, Int::<4>],
            int_tuple![Int::<8>, Int::<2>]
        )
    );

    // idx2crd_with_stride (row-major, col-major, and static unit-extent stride-0)
    assert_eq!(
        idx2crd_with_stride(7i32, &ROW.shape(), &ROW.stride()),
        int_tuple![1i32, 3i32]
    );
    assert_eq!(
        idx2crd_with_stride(
            5i32,
            &int_tuple![Int::<1>, Int::<8>],
            &int_tuple![Int::<0>, Int::<1>],
        ),
        int_tuple![Int::<0>, 5i32]
    );

    // get_flat_coord
    let nested_layout = Layout::new(
        int_tuple![(Int::<2>, Int::<2>), Int::<3>],
        int_tuple![(Int::<1>, Int::<2>), Int::<4>],
    );
    for idx in 0..12 {
        assert_eq!(
            get_flat_coord(&nested_layout, idx),
            int_tuple![idx % 4, idx / 4]
        );
        assert_eq!(
            nested_layout.get_flat_coord(idx),
            int_tuple![idx % 4, idx / 4]
        );
    }

    // make_identity_layout + zipped_divide on basis-strided identity layout
    let id = make_identity_layout(int_tuple![10i32, 8i32]);
    assert_eq!(id.stride(), tuple![e!(0), e!(1)]);
    assert_eq!(id.call(int_tuple![3i32, 5i32]), 3 * e!(0) + 5 * e!(1));
    let id_tiled = zipped_divide(id, tuple![Int::<4>, Int::<4>]);
    // Tail tile (2, 0), intra-tile coord (3, 0) reaches row 11, col 0 (no carry into col!)
    assert_eq!(
        id_tiled.call(int_tuple![(3i32, 0i32), (2i32, 0i32)]),
        11 * e!(0) + 0 * e!(1)
    );

    // recast_layout::<Old, New>
    let u8_row = Layout::from_shape_right(int_tuple![Int::<8>, Int::<32>]);
    assert_eq!(
        recast_layout::<u8, u32, _>(&u8_row),
        Layout::from_shape_right(int_tuple![Int::<8>, Int::<8>])
    );
    assert_eq!(
        u8_row.recast_layout::<u8, u32>().recast_layout::<u32, u8>(),
        u8_row
    );
    assert_eq!(recast_layout::<i32, f32, _>(&u8_row), u8_row);

    // Layout forwarding methods: select, group, append, prepend
    let l3 = Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<4>]);
    assert_eq!(
        l3.select::<mode![2, 0]>(),
        Layout::new(
            int_tuple![Int::<4>, Int::<2>],
            int_tuple![Int::<6>, Int::<1>]
        )
    );
    assert_eq!(
        l3.group::<0, 2>(),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<4>],
            int_tuple![(Int::<1>, Int::<2>), Int::<6>],
        )
    );
    let l1 = Layout::new(Int::<5>, Int::<24>);
    assert_eq!(
        l3.append(l1),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>, Int::<5>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>, Int::<24>],
        )
    );
    assert_eq!(
        l3.prepend(l1),
        Layout::new(
            int_tuple![Int::<5>, Int::<2>, Int::<3>, Int::<4>],
            int_tuple![Int::<24>, Int::<1>, Int::<2>, Int::<6>],
        )
    );
}

#[test]
fn test_swizzle_helper_algebra() {
    // make_swizzle & swizzle_params matching CuTe upstream comments
    const SW1: StaticSwizzle<1, 2, 1> = make_swizzle::<0b1000, 0b0100>();
    const SW2: StaticSwizzle<2, 1, 5> = make_swizzle::<0b1100_0000, 0b0000_0110>();
    const SW_NEG: StaticSwizzle<1, 0, -1> = make_swizzle::<0b01, 0b10>();
    const SW_ZERO: StaticSwizzle<0, 0, 0> = make_swizzle::<0, 0>();
    assert_eq!(SW1, static_value::<StaticSwizzle<1, 2, 1>>());
    assert_eq!(SW2, static_value::<StaticSwizzle<2, 1, 5>>());
    assert_eq!(SW_NEG, static_value::<StaticSwizzle<1, 0, -1>>());
    assert_eq!(SW_ZERO, static_value::<StaticSwizzle<0, 0, 0>>());
    assert_eq!(SW_NEG.call(1i32), 3);
    assert_eq!(SW_NEG.call(2i32), 2);
    assert_eq!(swizzle_params(0b111 << 6, 0b111 << 3), (3, 3, 3));

    // compose_swizzle (same shift merge, partial cancel, self-cancel, identity)
    const MERGED: StaticSwizzle<2, 3, 3> = compose_swizzle(
        static_value::<StaticSwizzle<1, 3, 3>>(),
        static_value::<StaticSwizzle<1, 4, 3>>(),
    );
    assert_eq!(MERGED, Swizzle::new(Int::<2>, Int::<3>, Int::<3>));
    assert_eq!(
        static_value::<StaticSwizzle<2, 3, 3>>().compose(static_value::<StaticSwizzle<1, 4, 3>>()),
        Swizzle::new(Int::<1>, Int::<3>, Int::<3>)
    );
    assert_eq!(
        static_value::<StaticSwizzle<3, 3, 3>>().compose(static_value::<StaticSwizzle<3, 3, 3>>()),
        Swizzle::new(Int::<0>, Int::<0>, Int::<0>)
    );
    assert_eq!(
        static_value::<StaticSwizzle<0, 4, 3>>().compose(static_value::<StaticSwizzle<2, 1, 5>>()),
        Swizzle::new(Int::<2>, Int::<1>, Int::<5>)
    );
    assert_eq!(
        Swizzle::new(1i32, 3i32, 3i32).compose(Swizzle::new(1i32, 4i32, 3i32)),
        Swizzle::new(2i32, 3i32, 3i32)
    );

    // upcast_swizzle, downcast_swizzle, recast_swizzle
    const SW333: StaticSwizzle<3, 3, 3> = static_value();
    const UP8: StaticSwizzle<3, 0, 3> = upcast_swizzle(SW333, Int::<8>);
    assert_eq!(UP8, Swizzle::new(Int::<3>, Int::<0>, Int::<3>));
    assert_eq!(downcast_swizzle(UP8, Int::<8>), SW333);
    // Below base, bits shrinks: max(B + M - log2(N), 0)
    assert_eq!(
        static_value::<StaticSwizzle<3, 1, 3>>().upcast(Int::<8>),
        Swizzle::new(Int::<1>, Int::<0>, Int::<3>)
    );
    assert_eq!(
        static_value::<StaticSwizzle<2, 0, 3>>().upcast(Int::<8>),
        Swizzle::new(Int::<0>, Int::<0>, Int::<3>)
    );
    assert_eq!(
        recast_swizzle::<u16, u8, _>(SW333),
        Swizzle::new(Int::<3>, Int::<4>, Int::<3>)
    );
    assert_eq!(
        recast_swizzle::<u8, u64, _>(SW333),
        Swizzle::new(Int::<3>, Int::<0>, Int::<3>)
    );
    assert_eq!(recast_swizzle::<i32, f32, _>(SW333), SW333);

    // Dynamic error checks (involution & parameter validation)
    assert!(catch_unwind(|| swizzle_params(-1, 0)).is_err());
    assert!(catch_unwind(|| swizzle_params(0b11, 0b100)).is_err()); // mismatched popcount
    assert!(catch_unwind(|| swizzle_params(0b110, 0b011)).is_err()); // overlapping (non-involution)
    assert!(catch_unwind(|| swizzle_params(0b101_000, 0b101)).is_err()); // non-contiguous
    // Non-involution rejected by compose/upcast/downcast
    let non_inv = Swizzle::new(2i32, 0i32, 1i32);
    assert!(!non_inv.is_involution());
    assert!(catch_unwind(|| non_inv.compose(non_inv)).is_err());
    assert!(catch_unwind(|| non_inv.upcast(2i32)).is_err());
    assert!(catch_unwind(|| non_inv.downcast(2i32)).is_err());
    // Different shifts rejected by compose
    assert!(
        catch_unwind(|| Swizzle::new(1i32, 3i32, 3i32).compose(Swizzle::new(1i32, 3i32, 2i32)))
            .is_err()
    );
    // Non-power-of-two factor rejected by upcast/downcast
    assert!(catch_unwind(|| Swizzle::new(3i32, 3i32, 3i32).upcast(3i32)).is_err());
    assert!(catch_unwind(|| Swizzle::new(3i32, 3i32, 3i32).downcast(0i32)).is_err());
}

#[test]
fn test_algebra_compose_tile_divide_and_product_variants() {
    // compose_tile with `..` (RangeFull) and mode truncation
    let l = Layout::new(
        int_tuple![Int::<8>, Int::<4>, Int::<2>],
        int_tuple![Int::<1>, Int::<8>, Int::<32>],
    );
    let tile = tuple![Layout::new(Int::<4>, Int::<2>), ..];
    assert_eq!(
        compose_tile(l, tile),
        Layout::new(
            int_tuple![Int::<4>, Int::<4>],
            int_tuple![Int::<2>, Int::<8>]
        )
    );
    assert_eq!(l.compose_tile(tile), composition(l, tile));
    assert_eq!(compose_tile(l, Nil), Layout::new(Nil, Nil));

    // logical_divide / logical_product with `..`
    assert_eq!(logical_divide(l, ..), l);
    assert_eq!(logical_product(l, ..), l);
    assert_eq!(
        logical_divide(
            Layout::new(
                int_tuple![Int::<6>, Int::<4>],
                int_tuple![Int::<4>, Int::<1>]
            ),
            tuple![Int::<2>, ..]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<4>],
            int_tuple![(Int::<4>, Int::<8>), Int::<1>],
        )
    );

    // tiled_divide & flat_divide
    let a = Layout::new(
        int_tuple![Int::<6>, Int::<4>, Int::<5>],
        int_tuple![Int::<1>, Int::<6>, Int::<24>],
    );
    let t = tuple![Int::<2>, Int::<2>];
    assert_eq!(
        zipped_divide(a, t),
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), (Int::<3>, Int::<2>, Int::<5>)],
            int_tuple![(Int::<1>, Int::<6>), (Int::<2>, Int::<12>, Int::<24>)],
        )
    );
    assert_eq!(
        tiled_divide(a, t),
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), Int::<3>, Int::<2>, Int::<5>],
            int_tuple![(Int::<1>, Int::<6>), Int::<2>, Int::<12>, Int::<24>],
        )
    );
    assert_eq!(
        flat_divide(a, t),
        Layout::new(
            int_tuple![Int::<2>, Int::<2>, Int::<3>, Int::<2>, Int::<5>],
            int_tuple![Int::<1>, Int::<6>, Int::<2>, Int::<12>, Int::<24>],
        )
    );
    assert_eq!(
        tiled_divide(Layout::new(Int::<8>, Int::<1>), Int::<2>),
        Layout::new(
            int_tuple![Int::<2>, Int::<4>],
            int_tuple![Int::<1>, Int::<2>]
        )
    );
    assert_eq!(
        flat_divide(Layout::new(Int::<8>, Int::<1>), Int::<2>),
        Layout::new(
            int_tuple![Int::<2>, Int::<4>],
            int_tuple![Int::<1>, Int::<2>]
        )
    );

    // zipped_product, tiled_product, flat_product (single layout & partial tuple tiler)
    let blk = Layout::from_shape(int_tuple![Int::<2>, Int::<3>]);
    let grid = Layout::from_shape(int_tuple![Int::<4>, Int::<2>]);
    assert_eq!(
        zipped_product(blk, grid),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), (Int::<4>, Int::<2>)],
            int_tuple![(Int::<1>, Int::<2>), (Int::<6>, Int::<24>)],
        )
    );
    assert_eq!(
        tiled_product(blk, grid),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<4>, Int::<2>],
            int_tuple![(Int::<1>, Int::<2>), Int::<6>, Int::<24>],
        )
    );
    assert_eq!(
        flat_product(blk, grid),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>, Int::<24>],
        )
    );

    let blk3 = Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<5>]);
    let tiler2 = tuple![Int::<4>, Int::<2>];
    // Mode 0: 2:1 x 4:1 -> (2, 4):(1, 2)
    // Mode 1: 3:2 x 2:1 -> complement(3:2) = (2, 1):(1, 6), so rest_1 = 2:1
    // Mode 2: uncovered 5:6 moves into Rest
    assert_eq!(
        zipped_product(blk3, tiler2),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), (Int::<4>, Int::<2>, Int::<5>)],
            int_tuple![(Int::<1>, Int::<2>), (Int::<2>, Int::<1>, Int::<6>)],
        )
    );
    assert_eq!(
        tiled_product(blk3, tiler2),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<4>, Int::<2>, Int::<5>],
            int_tuple![(Int::<1>, Int::<2>), Int::<2>, Int::<1>, Int::<6>],
        )
    );
    assert_eq!(
        flat_product(blk3, tiler2),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>, Int::<2>, Int::<5>],
            int_tuple![Int::<1>, Int::<2>, Int::<2>, Int::<1>, Int::<6>],
        )
    );

    // Unequal-rank blocked_product & raked_product (both rank(A) < rank(B) and rank(A) > rank(B))
    assert_eq!(
        blocked_product(
            Layout::from_shape(Int::<8>),
            Layout::from_shape(int_tuple![Int::<2>, Int::<2>]),
        ),
        Layout::new(
            int_tuple![(Int::<8>, Int::<2>), (Int::<1>, Int::<2>)],
            int_tuple![(Int::<1>, Int::<8>), (Int::<0>, Int::<16>)],
        )
    );
    assert_eq!(
        blocked_product(
            Layout::from_shape(int_tuple![Int::<2>, Int::<3>]),
            Layout::from_shape(Int::<4>),
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<4>), (Int::<3>, Int::<1>)],
            int_tuple![(Int::<1>, Int::<6>), (Int::<2>, Int::<0>)],
        )
    );
    assert_eq!(
        raked_product(
            Layout::from_shape(int_tuple![Int::<2>, Int::<3>]),
            Layout::from_shape(Int::<4>),
        ),
        Layout::new(
            int_tuple![(Int::<4>, Int::<2>), (Int::<1>, Int::<3>)],
            int_tuple![(Int::<6>, Int::<1>), (Int::<0>, Int::<2>)],
        )
    );
}

#[test]
fn test_algebra_common_vector_layout_tile_and_fragment() {
    // max_common_vector & max_common_layout
    let col = Layout::from_shape(int_tuple![Int::<4>, Int::<8>]);
    let row = Layout::from_shape_right(int_tuple![Int::<4>, Int::<8>]);
    assert_eq!(max_common_vector(col, col), Int::<32>);
    type Col4x8 = Layout<Til<Int<4>, Til<Int<8>, Nil>>, Til<Int<1>, Til<Int<4>, Nil>>>;
    assert_eq!(<Col4x8 as MaxCommonVector<Col4x8>>::VALUE, 32);
    assert_eq!(max_common_vector(row, col), Int::<1>);
    assert_eq!(max_common_vector(row, row), Int::<32>);

    let padded = Layout::new(
        int_tuple![Int::<4>, Int::<3>],
        int_tuple![Int::<1>, Int::<5>],
    );
    assert_eq!(max_common_vector(padded, padded), Int::<4>);

    let dyn_stride = Layout::new(
        int_tuple![Int::<4>, Int::<8>],
        int_tuple![Int::<1>, 4i32],
    );
    assert_eq!(max_common_vector(dyn_stride, col), Int::<4>);

    assert_eq!(max_common_layout(col, col), Layout::new(Int::<32>, Int::<1>));
    assert_eq!(max_common_layout(row, col), Layout::new(Int::<1>, Int::<0>));
    let common_rr = max_common_layout(row, row);
    assert_eq!(coalesce(composition(row, common_rr)), Layout::new(Int::<32>, Int::<1>));

    // F2-swizzled layout vectorization via algebra mainline (no ComposedLayout wrapper needed!):
    // 1. Row-major 8x64 with Swizzle<3, 3, 3> at origin 0: row 0 (64 elements) has r == 0 so
    //    bits 3..6 are untouched until index 64 -> exact leading contiguous vector is 64.
    let plain_8x64 = Layout::new(
        int_tuple![Int::<8>, Int::<8>, Int::<8>],
        int_tuple![Int::<1>, Int::<8>, Int::<64>],
    );
    let swiz_8x64 = Layout::new(
        int_tuple![Int::<8>, Int::<8>, Int::<8>],
        tuple![f2(Int::<1>), f2(Int::<8>), f2(Int::<72>)],
    );
    assert_eq!(max_common_vector(swiz_8x64, plain_8x64), Int::<64>);
    assert_eq!(max_common_vector(plain_8x64, swiz_8x64), Int::<64>);
    assert_eq!(
        max_common_layout(swiz_8x64, plain_8x64),
        Layout::new(Int::<64>, Int::<1>)
    );
    // 2. When the swizzled mode immediately follows the 8-element unswizzled vector,
    //    divergence from flat 512:1 starts at index 8 -> exact leading contiguous vector is 8.
    let swiz_vec8 = Layout::new(
        int_tuple![Int::<8>, Int::<8>, Int::<8>],
        tuple![f2(Int::<1>), f2(Int::<72>), f2(Int::<8>)],
    );
    assert_eq!(max_common_vector(swiz_vec8, plain_8x64), Int::<8>);
    assert_eq!(max_common_vector(plain_8x64, swiz_vec8), Int::<8>);
    assert_eq!(
        max_common_layout(swiz_vec8, plain_8x64),
        Layout::new(Int::<8>, Int::<1>)
    );
    let f2_linear = Layout::new(Int::<16>, f2(Int::<1>));
    assert_eq!(max_common_vector(f2_linear, f2_linear), Int::<16>);
    assert_eq!(
        max_common_layout(f2_linear, f2_linear),
        Layout::new(Int::<16>, f2(Int::<1>))
    );

    // tile_to_shape
    let atom = Layout::from_shape_right(int_tuple![Int::<8>, Int::<8>]);
    assert_eq!(
        tile_to_shape(atom, int_tuple![Int::<16>, Int::<16>]),
        Layout::new(
            int_tuple![(Int::<8>, Int::<2>), (Int::<8>, Int::<2>)],
            int_tuple![(Int::<8>, Int::<64>), (Int::<1>, Int::<128>)],
        )
    );
    // Rank-1 block padded with 1:0 to target rank
    assert_eq!(
        Layout::from_shape(Int::<8>).tile_to_shape(int_tuple![Int::<16>, Int::<2>]),
        Layout::new(
            int_tuple![(Int::<8>, Int::<2>), (Int::<1>, Int::<2>)],
            int_tuple![(Int::<1>, Int::<8>), (Int::<0>, Int::<16>)],
        )
    );
    // SM80 nested block tiled to (128, 64, 3)
    let sm80_atom = Layout::new(
        int_tuple![Int::<8>, (Int::<8>, Int::<8>)],
        int_tuple![Int::<8>, (Int::<1>, Int::<64>)],
    );
    let s_a = sm80_atom.tile_to_shape(int_tuple![Int::<128>, Int::<64>, Int::<3>]);
    assert_eq!(
        s_a.shape(),
        int_tuple![
            (Int::<8>, Int::<16>),
            ((Int::<8>, Int::<8>), Int::<1>),
            (Int::<1>, Int::<3>)
        ]
    );
    assert_eq!(s_a.size(), 128 * 64 * 3);
    // Dynamic target works & checks divisibility at runtime
    let dyn_tiled = tile_to_shape(atom, int_tuple![16i32, 32i32]);
    assert_eq!(dyn_tiled.size(), 16 * 32);
    assert!(catch_unwind(|| tile_to_shape(atom, int_tuple![10i32, 16i32])).is_err());
    assert!(catch_unwind(|| tile_to_shape(atom, int_tuple![-8i32, 16i32])).is_err());

    // make_fragment_like
    // 1. Multi-mode with static strides: mode 0 compact left, rest ordered by stride
    let part = Layout::new(
        int_tuple![Int::<2>, Int::<4>, Int::<2>],
        int_tuple![Int::<1>, Int::<64>, Int::<8>],
    );
    assert_eq!(
        make_fragment_like(&part),
        Layout::new(
            int_tuple![Int::<2>, Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<4>, Int::<2>],
        )
    );
    // 2. Multi-mode with dynamic strides: rest falls back to left-to-right order
    let part_dyn = Layout::new(part.shape(), int_tuple![Int::<1>, 64i32, 8i32]);
    assert_eq!(
        part_dyn.make_fragment_like(),
        Layout::new(
            int_tuple![Int::<2>, Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<2>, Int::<8>],
        )
    );
    // 3. Nested mode 0 with static stride-0 preserved, rest ordered
    let part_zero0 = Layout::new(
        int_tuple![(Int::<2>, Int::<2>), Int::<3>],
        int_tuple![(Int::<1>, Int::<0>), Int::<8>],
    );
    assert_eq!(
        make_fragment_like(&part_zero0),
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), Int::<3>],
            int_tuple![(Int::<1>, Int::<0>), Int::<2>],
        )
    );
    // 4. All-zero mode 0 preserved
    let part_all_zero0 = Layout::new(
        int_tuple![Int::<4>, Int::<3>],
        int_tuple![Int::<0>, Int::<1>],
    );
    assert_eq!(
        make_fragment_like(&part_all_zero0),
        Layout::new(
            int_tuple![Int::<4>, Int::<3>],
            int_tuple![Int::<0>, Int::<1>],
        )
    );
    // 5. Rest mode with stride 0 still receives non-zero register stride (sorted first)
    let part_zero_rest = Layout::new(
        int_tuple![Int::<2>, Int::<4>, Int::<2>],
        int_tuple![Int::<1>, Int::<0>, Int::<8>],
    );
    assert_eq!(
        make_fragment_like(&part_zero_rest),
        Layout::new(
            int_tuple![Int::<2>, Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<2>, Int::<8>],
        )
    );
    // 6. Single-mode and rank-0 layouts use compact left-major
    assert_eq!(
        make_fragment_like(&Layout::new(Int::<8>, Int::<4>)),
        Layout::new(Int::<8>, Int::<1>)
    );
    assert_eq!(
        make_fragment_like(&Layout::new(int_tuple![Int::<8>], int_tuple![Int::<4>])),
        Layout::new(int_tuple![Int::<8>], int_tuple![Int::<1>])
    );
    assert_eq!(
        make_fragment_like(&Layout::new(Nil, Nil)),
        Layout::new(Nil, Nil)
    );
}
