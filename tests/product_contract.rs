//! CuTe product 的全静态、混合与运行时契约。
#![feature(min_adt_const_params, macro_metavar_expr)]

use noodle_cute::Error;
use noodle_cute::algebra::*;
use noodle_cute::layout::{AffineLayoutMap, DynLayout, GroupedLayout, Layout, LayoutMap, MAX_RANK};

pub mod common;
use common::{TableMap, assert_grouped, assert_same_map, dyn_layout, grouped, producted, values};
use noodle_cute::{DynInt, Int, IntTuple, Nil, StaticValue, int_tuple, shape, stride};

type Pair<A, B> = IntTuple<A, IntTuple<B, Nil>>;

type ModePairs<const A: isize, const B: isize, const C: isize, const D: isize> =
    Pair<Pair<Int<A>, Int<B>>, Pair<Int<C>, Int<D>>>;

#[test]
fn static_products_are_const_layouts_with_the_terminal_repetition_stride() {
    const R: Layout<Pair<Int<2>, Int<3>>, Pair<Int<1>, Int<2>>> =
        producted::<{ ProductKind::Logical }, _, _>(
            Layout {
                shape: Int::<2>,
                stride: Int::<1>,
            },
            Int::<3>,
        );
    assert_eq!(R.index(5), Ok(5));
    let mixed = Layout::new(DynInt(2), Int::<1>)
        .unwrap()
        .logical_product(Int::<3>)
        .unwrap();
    assert_eq!(mixed.index(5), Ok(5));
}

#[test]
fn tuple_product_uses_each_modes_own_complement() {
    let block = Layout::new(shape![Int::<4>, Int::<2>], stride![Int::<1>, Int::<10>]).unwrap();
    let tiler = int_tuple![Int::<2>, Int::<3>];
    let logical: Layout<ModePairs<4, 2, 2, 3>, ModePairs<1, 4, 10, 1>> =
        block.logical_product(tiler).unwrap();
    assert_eq!(logical.size(), Ok(48));
    let zipped = block.zipped_product(tiler).unwrap();
    assert_eq!(zipped.as_flat_layout().unwrap().strides(), &[1, 10, 4, 1]);
    let flat = block.flat_product(tiler).unwrap();
    assert_eq!(flat.rank(), 4);
    assert_eq!(block.tiled_product(tiler).unwrap().rank(), 3);
}

#[test]
fn rank_sensitive_static_products_pad_unit_modes_and_zip_in_order() {
    const B: Layout<Pair<Int<2>, Int<3>>, Pair<Int<1>, Int<2>>> = Layout {
        shape: shape![Int::<2>, Int::<3>],
        stride: stride![Int::<1>, Int::<2>],
    };
    const T: Layout<Int<4>, Int<1>> = Layout {
        shape: Int::<4>,
        stride: Int::<1>,
    };
    const BLOCK: Layout<ModePairs<2, 4, 3, 1>, ModePairs<1, 6, 2, 0>> =
        producted::<{ ProductKind::Blocked }, _, _>(B, T);
    const RAKE: Layout<ModePairs<4, 2, 1, 3>, ModePairs<6, 1, 0, 2>> =
        producted::<{ ProductKind::Raked }, _, _>(B, T);
    assert_eq!(BLOCK.size(), Ok(24));
    assert_eq!(RAKE.size(), Ok(24));
    assert_eq!(BLOCK, B.blocked_product(T).unwrap());
    assert_eq!(RAKE, B.raked_product(T).unwrap());
    let t = Layout::new(DynInt(4), Int::<1>).unwrap();
    assert_eq!(B.blocked_product(t).unwrap().size(), Ok(24));
}

// ---------------------------------------------------------------------------
// 6. Product (`src/algebra/product.rs`)
// ---------------------------------------------------------------------------

#[test]
fn product_variants_repeat_tiles_across_cotargetless_complements() {
    let blk = dyn_layout(&[2, 2], &[1, 4]);
    let t23 = grouped(&[2, 3], &[1, 2], &[1, 1]);

    let logical: Produced<_, _> = logical_product(blk, t23).unwrap();
    let zipped = zipped_product(blk, t23).unwrap();
    let tiled = tiled_product(blk, t23).unwrap();
    let flat = flat_product(blk, t23).unwrap();

    assert_grouped(
        &logical.as_affine().unwrap().unwrap(),
        &[2, 2, 2, 3],
        &[1, 4, 2, 8],
        &[2, 2],
    );
    assert_grouped(
        &zipped.as_affine().unwrap().unwrap(),
        &[2, 2, 2, 3],
        &[1, 4, 2, 8],
        &[2, 2],
    );
    assert_grouped(
        &tiled.as_affine().unwrap().unwrap(),
        &[2, 2, 2, 3],
        &[1, 4, 2, 8],
        &[2, 1, 1],
    );
    assert_grouped(
        &flat.as_affine().unwrap().unwrap(),
        &[2, 2, 2, 3],
        &[1, 4, 2, 8],
        &[1, 1, 1, 1],
    );
    assert_eq!(
        values(&logical),
        [
            0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15, 16, 17, 20, 21, 18, 19, 22, 23
        ]
    );

    for rj in 0..3 {
        for ri in 0..2 {
            for tj in 0..2 {
                for ti in 0..2 {
                    let t = ti + 2 * tj;
                    let r = ri + 2 * rj;
                    let expected = ti + 4 * tj + 2 * (r % 2) + 8 * (r / 2);
                    assert_eq!(logical.index_of_coord(&[t, r]), Ok(expected));
                    assert_eq!(zipped.index_of_coord(&[t, r]), Ok(expected));
                    assert_eq!(tiled.index_of_coord(&[t, ri, rj]), Ok(expected));
                    assert_eq!(flat.index_of_coord(&[ti, tj, ri, rj]), Ok(expected));
                }
            }
        }
    }

    // Additional CuTe oracle product fixtures and broadcast tile case.
    type ProductCase = (
        GroupedLayout,
        GroupedLayout,
        &'static [isize],
        &'static [isize],
        &'static [usize],
    );
    let cases: [ProductCase; 4] = [
        (
            grouped(&[4], &[2], &[1]),
            grouped(&[2, 2], &[1, 2], &[2]),
            &[4, 2, 2],
            &[2, 1, 8],
            &[1, 2],
        ),
        (
            grouped(&[3], &[1], &[1]),
            grouped(&[4], &[2], &[1]),
            &[3, 4],
            &[1, 6],
            &[1, 1],
        ),
        (
            grouped(&[2, 2], &[2, 1], &[2]),
            grouped(&[3], &[1], &[1]),
            &[2, 2, 3],
            &[2, 1, 4],
            &[2, 1],
        ),
        (
            grouped(&[4], &[0], &[1]),
            grouped(&[3], &[1], &[1]),
            &[4, 3],
            &[0, 1],
            &[1, 1],
        ),
    ];
    for (block, tiler, s, d, g) in cases {
        let prod = logical_product(block, tiler).unwrap();
        let eager = prod.as_affine().unwrap().unwrap();
        assert_grouped(&eager, s, d, g);
        assert_eq!(values(&prod), values(&eager));
    }

    // Non-affine `rest` works lazily across product variants.
    let lazy_rest = logical_product(blk, TableMap(&[0, 2, 1])).unwrap();
    assert_eq!(lazy_rest.as_affine(), Ok(None));
    assert_eq!(values(&lazy_rest), [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7]);

    // Error bounds: invalid block, negative rest addresses, and product size overflow.
    assert_eq!(
        logical_product(dyn_layout(&[2], &[-1]), dyn_layout(&[2], &[1])).err(),
        Some(Error::Unsupported)
    );
    assert_eq!(
        logical_product(dyn_layout(&[0], &[1]), dyn_layout(&[2], &[1])).err(),
        Some(Error::Unsupported)
    );
    assert_eq!(
        logical_product(dyn_layout(&[2, 2], &[2, 3]), dyn_layout(&[2], &[1])).err(),
        Some(Error::NonInjective)
    );
    // CuTe: complement(2:1) = 1:2, so logical_product(2:1, 2:-1) = (2:1, 2:-2).
    assert_eq!(
        values(&logical_product(dyn_layout(&[2], &[1]), dyn_layout(&[2], &[-1])).unwrap()),
        [0, 1, -2, -1]
    );
    assert_eq!(
        logical_product(dyn_layout(&[1 << 32], &[0]), dyn_layout(&[1 << 32], &[1])).err(),
        Some(Error::Overflow)
    );
}

#[test]
fn blocked_and_raked_products_zip_modes_pad_missing_axes_and_build_tv_layouts() {
    let col = grouped(&[2, 2], &[1, 2], &[1, 1]);
    let t23 = grouped(&[2, 3], &[1, 2], &[1, 1]);
    let row = grouped(&[2, 2], &[2, 1], &[1, 1]);
    let r32 = grouped(&[3, 2], &[2, 1], &[1, 1]);
    let vec4 = grouped(&[4], &[1], &[1]);
    let vec3 = grouped(&[3], &[1], &[1]);

    type ZipProductCase = (
        GroupedLayout,
        &'static [isize],
        &'static [isize],
        &'static [usize],
    );
    let cases: [ZipProductCase; 8] = [
        (
            blocked_product(col, t23).unwrap(),
            &[2, 2, 2, 3],
            &[1, 4, 2, 8],
            &[2, 2],
        ),
        (
            raked_product(col, t23).unwrap(),
            &[2, 2, 3, 2],
            &[4, 1, 8, 2],
            &[2, 2],
        ),
        // Rank-1 block padded with `1:0` to match rank-2 tiler.
        (
            blocked_product(vec4, t23).unwrap(),
            &[4, 2, 1, 3],
            &[1, 4, 0, 8],
            &[2, 2],
        ),
        (
            raked_product(vec4, t23).unwrap(),
            &[2, 4, 3, 1],
            &[4, 1, 8, 0],
            &[2, 2],
        ),
        // Rank-1 tiler padded with `1:0` to match rank-2 block.
        (
            blocked_product(col, vec3).unwrap(),
            &[2, 3, 2, 1],
            &[1, 4, 2, 0],
            &[2, 2],
        ),
        (
            raked_product(col, vec3).unwrap(),
            &[3, 2, 1, 2],
            &[4, 1, 0, 2],
            &[2, 2],
        ),
        (
            blocked_product(row, r32).unwrap(),
            &[2, 3, 2, 2],
            &[2, 8, 1, 4],
            &[2, 2],
        ),
        (
            raked_product(row, r32).unwrap(),
            &[3, 2, 2, 2],
            &[8, 2, 4, 1],
            &[2, 2],
        ),
    ];
    for (actual, s, d, g) in cases {
        assert_grouped(&actual, s, d, g);
    }

    let blocked = blocked_product(col, t23).unwrap();
    let raked = raked_product(col, t23).unwrap();
    assert_eq!(
        values(&raked),
        [
            0, 4, 1, 5, 8, 12, 9, 13, 16, 20, 17, 21, 2, 6, 3, 7, 10, 14, 11, 15, 18, 22, 19, 23
        ]
    );
    // In mode 0, blocked steps within the block first, whereas raked steps across tiles first.
    assert_eq!((blocked.extent(0), blocked.extent(1)), (Ok(4), Ok(6)));
    assert_eq!(blocked.index_of_coord(&[1, 0]), Ok(1));
    assert_eq!(raked.index_of_coord(&[1, 0]), Ok(4));

    // Scalar rank-0 operands produce a rank-0 scalar GroupedLayout.
    let scalar = Layout::new(Nil, Nil).unwrap();
    assert_grouped(&blocked_product(scalar, scalar).unwrap(), &[], &[], &[]);

    // Nested multi-leaf modes sum their submode leaf counts in each zipped mode.
    let nested_block = Layout::new(
        shape![(Int::<2>, Int::<2>), Int::<2>],
        stride![(Int::<1>, Int::<4>), Int::<2>],
    )
    .unwrap();
    assert_grouped(
        &blocked_product(nested_block, t23).unwrap(),
        &[2, 2, 2, 2, 3],
        &[1, 4, 8, 2, 16],
        &[3, 2],
    );

    // End-to-end CuTe `make_tiled_copy` TV-layout construction:
    // `tv = composition_layout(&right_inverse(&raked_product(thr, val)), &(T, V):(1, T))`.
    struct TvCase {
        thr: GroupedLayout,
        val: GroupedLayout,
        mn_shape: &'static [isize],
        mn_stride: &'static [isize],
        tv_shape: &'static [isize],
        tv_stride: &'static [isize],
        tv_groups: &'static [usize],
    }

    let tv_cases = [
        TvCase {
            thr: grouped(&[4, 2], &[2, 1], &[1, 1]),
            val: grouped(&[2, 2], &[1, 2], &[1, 1]),
            mn_shape: &[2, 4, 2, 2],
            mn_stride: &[8, 2, 16, 1],
            tv_shape: &[2, 4, 2, 2],
            tv_stride: &[16, 2, 1, 8],
            tv_groups: &[2, 2],
        },
        TvCase {
            thr: grouped(&[2, 4], &[1, 2], &[1, 1]),
            val: grouped(&[1, 4], &[1, 1], &[1, 1]),
            mn_shape: &[1, 2, 4, 4],
            mn_stride: &[8, 1, 8, 2],
            tv_shape: &[2, 4, 4],
            tv_stride: &[1, 8, 2],
            tv_groups: &[2, 1],
        },
        TvCase {
            thr: grouped(&[4, 2], &[1, 4], &[1, 1]),
            val: grouped(&[2, 1], &[1, 1], &[1, 1]),
            mn_shape: &[2, 4, 1, 2],
            mn_stride: &[8, 1, 8, 4],
            tv_shape: &[8, 2],
            tv_stride: &[2, 1],
            tv_groups: &[1, 1],
        },
    ];

    for c in tv_cases {
        let mn = raked_product(c.thr, c.val).unwrap();
        assert_grouped(&mn, c.mn_shape, c.mn_stride, &[2, 2]);
        let rinv = right_inverse(mn).unwrap();
        let domain = GroupedLayout::from_flat(
            DynLayout::left(&[c.thr.size().unwrap(), c.val.size().unwrap()]).unwrap(),
        )
        .unwrap();
        let tv = composition_layout(&rinv, &domain).unwrap();
        assert_grouped(&tv, c.tv_shape, c.tv_stride, c.tv_groups);

        let threads = c.thr.size().unwrap();
        for t in 0..threads {
            for v in 0..c.val.size().unwrap() {
                let tile_coord = tv.index_of_coord(&[t, v]).unwrap();
                assert_eq!(mn.index(tile_coord), Ok(t + threads * v));
            }
        }
    }

    // Error bounds: indivisibility against `complement_extended(block)` and > MAX_RANK leaves.
    assert_eq!(
        blocked_product(dyn_layout(&[4], &[2]), dyn_layout(&[2], &[3])),
        Err(Error::NotDivisible)
    );
    // Negative tiler strides compose through `signum`: ((2,2)):((1,-2)).
    assert_grouped(
        &blocked_product(dyn_layout(&[2], &[1]), dyn_layout(&[2], &[-1])).unwrap(),
        &[2, 2],
        &[1, -2],
        &[2],
    );
    let rank_17 = DynLayout::left(&[2; 17]).unwrap();
    assert_eq!(
        raked_product(rank_17, rank_17),
        Err(Error::CapacityExceeded)
    );
}

#[test]
fn nested_partial_and_empty_product_tilers_preserve_profiles() {
    let input = Layout::new(
        shape![(Int::<8>, Int::<6>), Int::<5>],
        stride![(Int::<1>, Int::<8>), Int::<48>],
    )
    .unwrap();
    let zipped = input
        .zipped_product(int_tuple![(Int::<2>, Int::<3>)])
        .unwrap();
    assert_eq!(
        zipped.shape,
        shape![((Int::<8>, Int::<6>)), ((Int::<2>, Int::<3>), Int::<5>)]
    );
    assert_eq!(
        zipped.stride,
        stride![((Int::<1>, Int::<8>)), ((Int::<8>, Int::<1>), Int::<48>)]
    );
    assert_eq!(input.logical_product(Nil).unwrap(), input);
    assert_eq!(input.flat_product(Nil).unwrap(), input);
    assert_eq!(input.zipped_product(Nil).unwrap().extent(0), Ok(1));
}

#[test]
fn static_product_grouping_entries_are_const() {
    type A = Layout<Int<4>, Int<2>>;
    type T = Layout<Int<3>, Int<2>>;
    const A: A = Layout {
        shape: Int::<4>,
        stride: Int::<2>,
    };
    const T: T = Layout {
        shape: Int::<3>,
        stride: Int::<2>,
    };
    const Z: TypedProductLayout<A, T, { ProductKind::Zipped }> =
        <TypedProductLayout<A, T, { ProductKind::Zipped }> as StaticValue>::VALUE;
    const F: TypedProductLayout<A, T, { ProductKind::Flat }> =
        producted::<{ ProductKind::Flat }, _, _>(A, T);
    const TL: TypedProductLayout<A, T, { ProductKind::Tiled }> =
        producted::<{ ProductKind::Tiled }, _, _>(A, T);
    assert_eq!(Z.shape, shape![Int::<4>, Int::<3>]);
    assert_eq!(Z.stride, stride![Int::<2>, Int::<8>]);
    assert_eq!(F.shape, shape![Int::<4>, Int::<3>]);
    assert_eq!(TL.shape, shape![Int::<4>, Int::<3>]);
}

#[test]
fn dynamic_tuple_product_matches_per_mode_cute_formulas() {
    let block = dyn_layout(&[4, 2], &[1, 10]);
    let tiler = [dyn_layout(&[2], &[1]), dyn_layout(&[3], &[1])];
    assert_grouped(
        &logical_product_modes(&block, &tiler).unwrap(),
        &[4, 2, 2, 3],
        &[1, 4, 10, 1],
        &[2, 2],
    );
    assert_grouped(
        &zipped_product_modes(&block, &tiler).unwrap(),
        &[4, 2, 2, 3],
        &[1, 10, 4, 1],
        &[2, 2],
    );
    assert_grouped(
        &tiled_product_modes(&block, &tiler).unwrap(),
        &[4, 2, 2, 3],
        &[1, 10, 4, 1],
        &[2, 1, 1],
    );
    assert_grouped(
        &flat_product_modes(&block, &tiler).unwrap(),
        &[4, 2, 2, 3],
        &[1, 10, 4, 1],
        &[1, 1, 1, 1],
    );
    assert_grouped(
        &logical_product_modes(&block, &tiler[..1]).unwrap(),
        &[4, 2, 2],
        &[1, 4, 10],
        &[2, 1],
    );
    assert_eq!(
        logical_product_modes(&block, &[tiler[0]; 3]),
        Err(Error::StructureMismatch)
    );
    assert_grouped(
        &logical_product_modes(&block, &[dyn_layout(&[2], &[-1])]).unwrap(),
        &[4, 2, 2],
        &[1, -4, 10],
        &[2, 1],
    );
}

#[test]
fn mixed_product_errors_and_zero_tilers_are_checked() {
    let zero = Layout::new(DynInt(0), Int::<1>).unwrap();
    assert_eq!(zero.logical_product(Int::<2>), Err(Error::Unsupported));
    let block = Layout::new(DynInt(2), Int::<1>).unwrap();
    let reversed = block
        .logical_product(Layout::new(DynInt(3), DynInt(-1)).unwrap())
        .unwrap();
    assert_eq!(values(&reversed), [0, 1, -2, -1, -4, -3]);
    assert_eq!(
        block.logical_product(Layout::new(DynInt(isize::MAX), Int::<0>).unwrap()),
        Err(Error::Overflow)
    );
    assert_eq!(
        Layout::new(DynInt(2), DynInt(isize::MAX))
            .unwrap()
            .logical_product(Int::<2>),
        Err(Error::Overflow)
    );
    let gapped = Layout::new(Int::<2>, Int::<2>).unwrap();
    assert_eq!(
        gapped.logical_product(Layout::new(DynInt(2), DynInt(3)).unwrap()),
        Err(Error::NotDivisible)
    );
    let empty = block.logical_product(Int::<0>).unwrap();
    assert_eq!(empty.size(), Ok(0));
}

#[test]
fn static_product_can_return_more_than_max_rank_leaves() {
    macro_rules! broadcast {
        ($($slot:literal),*) => { Layout {
            shape: shape![$( ${ignore($slot)} Int::<2> ),*],
            stride: stride![$( ${ignore($slot)} Int::<0> ),*],
        } };
    }
    let tiler = broadcast!(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32
    );
    let result = producted::<{ ProductKind::Flat }, _, _>(
        Layout {
            shape: Int::<2>,
            stride: Int::<1>,
        },
        tiler,
    );
    assert_eq!(result.rank(), MAX_RANK + 2);
    assert_eq!(result.as_flat_layout(), Err(Error::CapacityExceeded));
    assert_eq!(result.index((1isize << 33) + 1), Ok(1));
}

#[test]
fn product_scalar_layout_and_empty_tuple_tilers_keep_distinct_profiles() {
    let block = Layout::new(Int::<2>, Int::<1>).unwrap();
    let scalar = block
        .logical_product(Layout::new(Nil, Nil).unwrap())
        .unwrap();
    assert_eq!(scalar.shape, shape![Int::<2>, ()]);
    assert_eq!(block.logical_product(Nil).unwrap(), block);
    let empty = Layout::new(Nil, Nil).unwrap();
    let rank_product: Layout<Nil, Nil> = producted::<{ ProductKind::Blocked }, _, _>(empty, empty);
    assert_eq!(rank_product.size(), Ok(1));
    assert_eq!(rank_product.rank(), 0);
}

#[test]
fn tiled_and_flat_product_unpack_the_composed_rest() {
    // CuTe: complement(2:2) = (2,1):(1,4) and composing it with 6:1 gives (2,3):(1,4), so
    // tiled_product(2:2, 6:1) = (2,2,3):(2,1,4) with rank 3.
    let block = dyn_layout(&[2], &[2]);
    let tiler = dyn_layout(&[6], &[1]);
    let typed = Layout::new(Int::<2>, Int::<2>).unwrap();
    let typed_tiler = Layout::new(Int::<6>, Int::<1>).unwrap();

    let tiled = tiled_product(block, tiler).unwrap();
    assert_grouped(
        &tiled.as_affine().unwrap().unwrap(),
        &[2, 2, 3],
        &[2, 1, 4],
        &[1, 1, 1],
    );
    assert_same_map(&tiled, &typed.tiled_product(typed_tiler).unwrap());

    let flat = flat_product(block, tiler).unwrap();
    assert_grouped(
        &flat.as_affine().unwrap().unwrap(),
        &[2, 2, 3],
        &[2, 1, 4],
        &[1, 1, 1],
    );
    assert_same_map(&flat, &typed.flat_product(typed_tiler).unwrap());

    let logical = logical_product(block, tiler).unwrap();
    assert_grouped(
        &logical.as_affine().unwrap().unwrap(),
        &[2, 2, 3],
        &[2, 1, 4],
        &[1, 2],
    );
    assert_same_map(&logical, &typed.logical_product(typed_tiler).unwrap());
}

#[test]
fn rank_sensitive_products_keep_a_split_tiler_mode_nested() {
    // CuTe pads with append<R> before logical_product: complement((2,2):(2,8)) =
    // (2,2,1):(1,4,16) composed with 8:1 is (2,2,2):(1,4,16), which stays inside mode 0:
    //   blocked = ((2,(2,2,2)),(2,1)) : ((2,(1,4,16)),(8,0))
    //   raked   = (((2,2,2),2),(1,2)) : (((1,4,16),2),(0,8))
    let block = dyn_layout(&[2, 2], &[2, 8]);
    let tiler = dyn_layout(&[8], &[1]);
    let typed = Layout::new(shape![Int::<2>, Int::<2>], stride![Int::<2>, Int::<8>]).unwrap();
    let typed_tiler = Layout::new(Int::<8>, Int::<1>).unwrap();

    let blocked = blocked_product(block, tiler).unwrap();
    assert_grouped(&blocked, &[2, 2, 2, 2, 2, 1], &[2, 1, 4, 16, 8, 0], &[4, 2]);
    assert_same_map(&blocked, &typed.blocked_product(typed_tiler).unwrap());

    let raked = raked_product(block, tiler).unwrap();
    assert_grouped(&raked, &[2, 2, 2, 2, 1, 2], &[1, 4, 16, 2, 0, 8], &[4, 2]);
    assert_same_map(&raked, &typed.raked_product(typed_tiler).unwrap());

    // Rank 1: blocked_product(2:2, 6:1) = ((2,(2,3))):((2,(1,4))).
    let line = blocked_product(dyn_layout(&[2], &[2]), dyn_layout(&[6], &[1])).unwrap();
    assert_grouped(&line, &[2, 2, 3], &[2, 1, 4], &[3]);
    assert_same_map(
        &line,
        &Layout::new(Int::<2>, Int::<2>)
            .unwrap()
            .blocked_product(Layout::new(Int::<6>, Int::<1>).unwrap())
            .unwrap(),
    );
}
