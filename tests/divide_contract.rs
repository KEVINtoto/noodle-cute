//! CuTe divide 的全静态、混合与运行时契约。
use noodle_cute::Error;
use noodle_cute::algebra::Grouping;
use noodle_cute::algebra::divide::*;
use noodle_cute::layout::{AffineLayoutMap, DynLayout, GroupedLayout, Layout, LayoutMap, MAX_RANK};

pub mod common;
use common::{TableMap, assert_grouped, assert_same_map, dyn_layout, grouped, values};
use noodle_cute::{DynInt, Int, IntTuple, Nil, int_tuple, shape, stride};

type Pair<A, B> = IntTuple<A, IntTuple<B, Nil>>;

type ModePairs<const A: isize, const B: isize, const C: isize, const D: isize> =
    Pair<Pair<Int<A>, Int<B>>, Pair<Int<C>, Int<D>>>;

#[test]
fn static_single_layout_division_is_a_const_layout() {
    const R: Layout<Pair<Int<3>, Int<4>>, Pair<Int<2>, Int<6>>> = static_logical_divide(
        Layout {
            shape: Int::<12>,
            stride: Int::<2>,
        },
        Int::<3>,
    );
    assert_eq!(R.size(), Ok(12));
    for rest in 0..4 {
        for tile in 0..3 {
            assert_eq!(R.index_of_coord(&[tile, rest]), Ok(2 * (tile + 3 * rest)));
        }
    }
}

#[test]
fn tuple_division_preserves_and_transposes_the_full_profile() {
    let input = Layout::new(shape![Int::<8>, Int::<6>], stride![Int::<1>, Int::<8>]).unwrap();
    let tiler = int_tuple![Int::<2>, Int::<3>];
    let logical: Layout<ModePairs<2, 4, 3, 2>, ModePairs<1, 2, 8, 24>> =
        input.logical_divide(tiler).unwrap();
    let zipped: Layout<ModePairs<2, 3, 4, 2>, ModePairs<1, 8, 2, 24>> =
        input.zipped_divide(tiler).unwrap();
    assert_eq!(logical.size(), Ok(48));
    assert_eq!(zipped.index_of_coord(&[5, 7]), Ok(47));
    let flat = input.flat_divide(tiler).unwrap();
    assert_eq!(flat.as_flat_layout().unwrap().shapes(), &[2, 3, 4, 2]);
    let tiled = input.tiled_divide(tiler).unwrap();
    assert_eq!(tiled.rank(), 3);
    let mixed = Layout::new(DynInt(10), Int::<2>)
        .unwrap()
        .logical_divide(Int::<4>)
        .unwrap();
    assert_eq!(mixed.size(), Ok(12));
    assert_eq!(mixed.index_of_coord(&[3, 2]), Ok(22));
}

#[test]
fn strict_axes_support_const_mixed_and_zero_extent_inputs() {
    const R: Layout<ModePairs<2, 4, 3, 2>, ModePairs<1, 2, 8, 24>> = static_logical_divide_axes(
        Layout {
            shape: shape![Int::<8>, Int::<6>],
            stride: stride![Int::<1>, Int::<8>],
        },
        shape![Int::<2>, Int::<3>],
    );
    assert_eq!(R.size(), Ok(48));
    let input = Layout::new(shape![DynInt(8), Int::<6>], stride![Int::<1>, DynInt(8)]).unwrap();
    let result = input
        .zipped_divide_axes(shape![Int::<2>, DynInt(3)])
        .unwrap();
    assert_eq!(result.index_of_coord(&[5, 7]), Ok(47));
    let empty = Layout::new(Int::<0>, Int::<2>)
        .unwrap()
        .logical_divide_axes(Int::<3>)
        .unwrap();
    assert_eq!(empty.size(), Ok(0));
    assert_eq!(empty.as_flat_layout().unwrap().shapes(), &[3, 0]);
}

// ---------------------------------------------------------------------------
// 4. Divide (`src/algebra/divide.rs`)
// ---------------------------------------------------------------------------

#[test]
fn single_tiler_divides_match_cute_groupings_and_pad_into_extended_domain() {
    // Non-compact tile (2,2):(1,6) over 24:1 produces rest (3,2):(2,12).
    let base = DynLayout::left(&[24]).unwrap();
    let tile = dyn_layout(&[2, 2], &[1, 6]);

    let logical: Divided<_, _> = logical_divide(base, tile).unwrap();
    let zipped = zipped_divide(base, tile).unwrap();
    let tiled = tiled_divide(base, tile).unwrap();
    let flat = flat_divide(base, tile).unwrap();

    assert_grouped(
        &logical.as_affine().unwrap().unwrap(),
        &[2, 2, 3, 2],
        &[1, 6, 2, 12],
        &[2, 2],
    );
    assert_grouped(
        &zipped.as_affine().unwrap().unwrap(),
        &[2, 2, 3, 2],
        &[1, 6, 2, 12],
        &[2, 2],
    );
    assert_grouped(
        &tiled.as_affine().unwrap().unwrap(),
        &[2, 2, 3, 2],
        &[1, 6, 2, 12],
        &[2, 1, 1],
    );
    assert_grouped(
        &flat.as_affine().unwrap().unwrap(),
        &[2, 2, 3, 2],
        &[1, 6, 2, 12],
        &[1, 1, 1, 1],
    );

    let mut seen = [false; 24];
    for r1 in 0..2 {
        for r0 in 0..3 {
            for t1 in 0..2 {
                for t0 in 0..2 {
                    let t = t0 + 2 * t1;
                    let r = r0 + 3 * r1;
                    let expected = t0 + 6 * t1 + 2 * r0 + 12 * r1;
                    assert_eq!(logical.index_of_coord(&[t, r]), Ok(expected));
                    assert_eq!(zipped.index_of_coord(&[t, r]), Ok(expected));
                    assert_eq!(tiled.index_of_coord(&[t, r0, r1]), Ok(expected));
                    assert_eq!(flat.index_of_coord(&[t0, t1, r0, r1]), Ok(expected));
                    assert!(!seen[expected as usize]);
                    seen[expected as usize] = true;
                }
            }
        }
    }
    assert!(seen.into_iter().all(|v| v));

    // Padded CuTe oracle cases: non-dividing tiles extend into outer's extended domain.
    let padded = logical_divide(dyn_layout(&[6], &[1]), dyn_layout(&[4], &[1])).unwrap();
    assert_eq!(
        (padded.size(), padded.extent(0), padded.extent(1)),
        (Ok(8), Ok(4), Ok(2))
    );
    assert_grouped(
        &padded.as_affine().unwrap().unwrap(),
        &[4, 2],
        &[1, 4],
        &[1, 1],
    );
    assert_eq!(values(&padded), (0..8).collect::<Vec<_>>());

    let row_div = logical_divide(dyn_layout(&[4, 6], &[6, 1]), dyn_layout(&[3], &[1])).unwrap();
    assert_grouped(
        &row_div.as_affine().unwrap().unwrap(),
        &[3, 2, 6],
        &[6, 18, 1],
        &[1, 2],
    );
    assert_eq!(
        values(&row_div),
        [
            0, 6, 12, 18, 24, 30, 1, 7, 13, 19, 25, 31, 2, 8, 14, 20, 26, 32, 3, 9, 15, 21, 27, 33,
            4, 10, 16, 22, 28, 34, 5, 11, 17, 23, 29, 35,
        ]
    );

    let col = dyn_layout(&[4, 6], &[1, 4]);
    assert_grouped(
        &logical_divide(col, dyn_layout(&[5], &[1]))
            .unwrap()
            .as_affine()
            .unwrap()
            .unwrap(),
        &[5, 5],
        &[1, 5],
        &[1, 1],
    );
    assert_grouped(
        &logical_divide(col, dyn_layout(&[4], &[2]))
            .unwrap()
            .as_affine()
            .unwrap()
            .unwrap(),
        &[4, 2, 3],
        &[2, 1, 8],
        &[1, 2],
    );

    // Dividing an empty affine layout yields an empty divided layout.
    let empty_div = logical_divide(dyn_layout(&[0], &[1]), dyn_layout(&[4], &[1])).unwrap();
    assert_eq!(empty_div.size(), Ok(0));

    // Non-affine layouts support exact division via `complement(tile, layout.size())`
    // and reject non-dividing tiles that would require an extended domain.
    let non_affine = TableMap(&[10, 20, 30, 40, 50, 60]);
    let exact = logical_divide(non_affine, dyn_layout(&[3], &[1])).unwrap();
    assert_eq!(exact.as_affine(), Ok(None));
    assert_eq!(exact.index_of_coord(&[2, 1]), Ok(60));
    assert_eq!(
        logical_divide(non_affine, dyn_layout(&[4], &[1])).err(),
        Some(Error::CoordinateOutOfBounds)
    );

    // Invalid tilers are rejected.
    assert_eq!(
        logical_divide(base, dyn_layout(&[0], &[1])).err(),
        Some(Error::InvalidShape)
    );
    assert_eq!(
        logical_divide(base, dyn_layout(&[2], &[-1])).err(),
        Some(Error::Unsupported)
    );
    assert_eq!(
        logical_divide(base, dyn_layout(&[2, 2], &[2, 3])).err(),
        Some(Error::NonInjective)
    );
}

#[test]
fn divide_axes_variants_preserve_addresses_and_transpose_axis_coordinates() {
    let base = dyn_layout(&[6, 8], &[11, 1]);
    let logical: DividedAxes<_> = logical_divide_axes(base, &[2, 4]).unwrap();
    let zipped = zipped_divide_axes(base, &[2, 4]).unwrap();
    let tiled = tiled_divide_axes(base, &[2, 4]).unwrap();
    let flat = flat_divide_axes(base, &[2, 4]).unwrap();

    assert_eq!(*logical.layout(), base);
    assert_eq!(logical.tile_shape(), &[2, 4]);
    assert_eq!(logical.rest_shape(), &[3, 2]);
    assert_eq!(logical.grouping(), Grouping::Logical);
    assert_eq!(zipped.grouping(), Grouping::Zipped);
    assert_eq!(tiled.grouping(), Grouping::Tiled);
    assert_eq!(flat.grouping(), Grouping::Flat);

    assert_eq!(
        (logical.rank(), zipped.rank(), tiled.rank(), flat.rank()),
        (2, 2, 3, 4)
    );
    assert_eq!(
        (logical.extent(0), logical.extent(1), logical.extent(2)),
        (Ok(6), Ok(8), Err(Error::AxisOutOfBounds))
    );
    assert_eq!(
        (zipped.extent(0), zipped.extent(1), zipped.extent(2)),
        (Ok(8), Ok(6), Err(Error::AxisOutOfBounds))
    );
    assert_eq!(
        (
            tiled.extent(0),
            tiled.extent(1),
            tiled.extent(2),
            tiled.extent(3)
        ),
        (Ok(8), Ok(3), Ok(2), Err(Error::AxisOutOfBounds))
    );
    assert_eq!(
        (
            flat.extent(0),
            flat.extent(1),
            flat.extent(2),
            flat.extent(3),
            flat.extent(4)
        ),
        (Ok(2), Ok(4), Ok(3), Ok(2), Err(Error::AxisOutOfBounds))
    );

    for variant in [&logical, &zipped, &tiled, &flat] {
        assert_eq!(variant.size(), Ok(48));
        assert_eq!(variant.address_range(), base.address_range());
        assert_eq!(variant.is_injective(), Ok(true));
        assert_eq!(variant.as_affine(), Ok(None));
        assert_eq!(variant.index(-1), Err(Error::CoordinateOutOfBounds));
        assert_eq!(variant.index(48), Err(Error::CoordinateOutOfBounds));
    }
    // Logical preserves linear order; Zipped/Tiled/Flat share tile-first linear order.
    assert_eq!(values(&logical), values(&base));
    assert_eq!(values(&zipped), values(&tiled));
    assert_eq!(values(&tiled), values(&flat));

    for rj in 0..2 {
        for ri in 0..3 {
            for tj in 0..4 {
                for ti in 0..2 {
                    let expected = (ti + 2 * ri) * 11 + tj + 4 * rj;
                    assert_eq!(
                        logical.index_of_coord(&[ti + 2 * ri, tj + 4 * rj]),
                        Ok(expected)
                    );
                    assert_eq!(
                        zipped.index_of_coord(&[ti + 2 * tj, ri + 3 * rj]),
                        Ok(expected)
                    );
                    assert_eq!(tiled.index_of_coord(&[ti + 2 * tj, ri, rj]), Ok(expected));
                    assert_eq!(flat.index_of_coord(&[ti, tj, ri, rj]), Ok(expected));
                }
            }
        }
    }

    // Broadcast injectivity and empty axes propagate cleanly.
    let broadcast = zipped_divide_axes(dyn_layout(&[4, 6], &[0, 1]), &[2, 3]).unwrap();
    assert_eq!(broadcast.is_injective(), Ok(false));

    let empty = flat_divide_axes(dyn_layout(&[0, 8], &[1, 1]), &[2, 4]).unwrap();
    assert_eq!(empty.rest_shape(), &[0, 2]);
    assert_eq!(empty.size(), Ok(0));

    // Error bounds: rank mismatch, nonpositive tile, indivisibility, and > MAX_RANK.
    assert_eq!(
        logical_divide_axes(base, &[2]).err(),
        Some(Error::StructureMismatch)
    );
    assert_eq!(
        logical_divide_axes(base, &[0, 4]).err(),
        Some(Error::InvalidShape)
    );
    assert_eq!(
        logical_divide_axes(base, &[-2, 4]).err(),
        Some(Error::InvalidShape)
    );
    assert_eq!(
        logical_divide_axes(base, &[4, 4]).err(),
        Some(Error::NotDivisible)
    );

    let rank_33 = Layout::new(
        shape![
            2isize, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2
        ],
        stride![
            0isize, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ],
    )
    .unwrap();
    assert_eq!(
        logical_divide_axes(rank_33, &[1; MAX_RANK + 1]).err(),
        Some(Error::CapacityExceeded)
    );
}

#[test]
fn divide_modes_variants_match_cute_tuple_tilers_and_partial_tilers() {
    let tiler = |extents: &[isize]| -> Vec<DynLayout> {
        extents.iter().map(|&t| dyn_layout(&[t], &[1])).collect()
    };

    let m86 = grouped(&[8, 6], &[1, 8], &[1, 1]);
    let m65 = grouped(&[6, 5], &[1, 6], &[1, 1]);
    let r46 = grouped(&[4, 6], &[6, 1], &[1, 1]);
    let m462 = grouped(&[4, 6, 2], &[1, 4, 24], &[1, 1, 1]);
    let n46 = Layout::new(
        shape![(Int::<2>, DynInt(4)), Int::<6>],
        stride![(Int::<1>, Int::<2>), Int::<8>],
    )
    .unwrap();
    let t43 = tiler(&[4, 3]);
    let t23 = tiler(&[2, 3]);

    assert_grouped(
        &logical_divide_modes(&m86, &t43).unwrap(),
        &[4, 2, 3, 2],
        &[1, 4, 8, 24],
        &[2, 2],
    );
    assert_grouped(
        &zipped_divide_modes(&m86, &t43).unwrap(),
        &[4, 3, 2, 2],
        &[1, 8, 4, 24],
        &[2, 2],
    );
    assert_grouped(
        &tiled_divide_modes(&m86, &t43).unwrap(),
        &[4, 3, 2, 2],
        &[1, 8, 4, 24],
        &[2, 1, 1],
    );
    assert_grouped(
        &flat_divide_modes(&m86, &t43).unwrap(),
        &[4, 3, 2, 2],
        &[1, 8, 4, 24],
        &[1, 1, 1, 1],
    );

    // Padded, row-major, and strided-layout tilers.
    assert_grouped(
        &zipped_divide_modes(&m65, &tiler(&[4, 2])).unwrap(),
        &[4, 2, 2, 3],
        &[1, 6, 4, 12],
        &[2, 2],
    );
    assert_grouped(
        &zipped_divide_modes(&r46, &tiler(&[2, 4])).unwrap(),
        &[2, 4, 2, 2],
        &[6, 1, 12, 4],
        &[2, 2],
    );
    let strided = [dyn_layout(&[2], &[4]), dyn_layout(&[3], &[2])];
    assert_grouped(
        &zipped_divide_modes(&m86, &strided).unwrap(),
        &[2, 3, 4, 2],
        &[4, 16, 1, 8],
        &[2, 2],
    );

    // Partial tilers leave uncovered trailing modes in `rests`.
    assert_grouped(
        &logical_divide_modes(&m462, &t23).unwrap(),
        &[2, 2, 3, 2, 2],
        &[1, 2, 4, 12, 24],
        &[2, 2, 1],
    );
    assert_grouped(
        &zipped_divide_modes(&m462, &t23).unwrap(),
        &[2, 3, 2, 2, 2],
        &[1, 4, 2, 12, 24],
        &[2, 3],
    );
    assert_grouped(
        &tiled_divide_modes(&m462, &t23).unwrap(),
        &[2, 3, 2, 2, 2],
        &[1, 4, 2, 12, 24],
        &[2, 1, 1, 1],
    );
    assert_grouped(
        &flat_divide_modes(&m462, &t23).unwrap(),
        &[2, 3, 2, 2, 2],
        &[1, 4, 2, 12, 24],
        &[1, 1, 1, 1, 1],
    );

    // Empty tiler keeps all modes in `rests` with an empty scalar `tiles` group.
    assert_grouped(
        &zipped_divide_modes(&m86, &[]).unwrap(),
        &[8, 6],
        &[1, 8],
        &[0, 2],
    );

    // Nested typed layout mode coalescing and multi-leaf tiler mode preservation.
    assert_grouped(
        &zipped_divide_modes(&n46, &t43).unwrap(),
        &[4, 3, 2, 2],
        &[1, 8, 4, 24],
        &[2, 2],
    );
    let multi = [dyn_layout(&[2, 2], &[1, 4]), dyn_layout(&[3], &[1])];
    assert_grouped(
        &flat_divide_modes(&m86, &multi).unwrap(),
        &[2, 2, 3, 2, 2],
        &[1, 4, 8, 2, 24],
        &[2, 1, 1, 1],
    );

    // Error checks: tiler longer than rank, empty tiler item, non-divisible mode, mode capacity.
    assert_eq!(
        zipped_divide_modes(&m86, &tiler(&[2, 2, 2])),
        Err(Error::StructureMismatch)
    );
    assert_eq!(
        zipped_divide_modes(&m86, &tiler(&[0, 2])),
        Err(Error::InvalidShape)
    );
    let nested_row = grouped(&[4, 6], &[6, 1], &[2]);
    assert_eq!(
        logical_divide_modes(&nested_row, &[dyn_layout(&[3], &[2])]),
        Err(Error::NotDivisible)
    );
    let max_modes = GroupedLayout::new(dyn_layout(&[], &[]), &[0; MAX_RANK]).unwrap();
    assert_eq!(
        tiled_divide_modes(&max_modes, &[dyn_layout(&[1], &[1])]),
        Err(Error::CapacityExceeded)
    );
}

#[test]
fn nested_partial_and_empty_tilers_preserve_the_cute_profile() {
    let input = Layout::new(
        shape![(Int::<8>, Int::<6>), Int::<5>],
        stride![(Int::<1>, Int::<8>), Int::<48>],
    )
    .unwrap();
    let tiler = int_tuple![(Int::<2>, Int::<3>)];
    let zipped = input.zipped_divide(tiler).unwrap();
    assert_eq!(
        zipped.shape,
        shape![((Int::<2>, Int::<3>)), ((Int::<4>, Int::<2>), Int::<5>)]
    );
    assert_eq!(
        zipped.stride,
        stride![((Int::<1>, Int::<8>)), ((Int::<2>, Int::<24>), Int::<48>)]
    );
    assert_eq!(zipped.index_of_coord(&[5, 39]), Ok(239));
    assert_eq!(input.logical_divide(Nil).unwrap(), input);
    assert_eq!(input.flat_divide(Nil).unwrap(), input);
    let empty = input.zipped_divide(Nil).unwrap();
    assert_eq!(empty.extent(0), Ok(1));
    assert_eq!(empty.extent(1), Ok(240));
    let scalar = Layout::new(Nil, Nil)
        .unwrap()
        .logical_divide_axes(Nil)
        .unwrap();
    assert_eq!(scalar.rank(), 0);
    assert_eq!(scalar.size(), Ok(1));
}

#[test]
fn static_grouping_entries_and_axes_entries_are_const() {
    type A = Layout<Pair<Int<8>, Int<6>>, Pair<Int<1>, Int<8>>>;
    type T = Pair<Int<2>, Int<3>>;
    const INPUT: A = Layout {
        shape: shape![Int::<8>, Int::<6>],
        stride: stride![Int::<1>, Int::<8>],
    };
    const TILER: T = shape![Int::<2>, Int::<3>];
    const Z: <A as StaticDivide<T, 1>>::Output = static_zipped_divide(INPUT, TILER);
    const F: <A as StaticDivide<T, 3>>::Output = static_flat_divide(INPUT, TILER);
    const TILED: <A as StaticDivide<T, 2>>::Output = static_tiled_divide(INPUT, TILER);
    const AZ: <A as StaticDivideAxes<T, 1>>::Output = static_zipped_divide_axes(INPUT, TILER);
    const AF: <A as StaticDivideAxes<T, 3>>::Output = static_flat_divide_axes(INPUT, TILER);
    const AT: <A as StaticDivideAxes<T, 2>>::Output = static_tiled_divide_axes(INPUT, TILER);
    assert_eq!(Z.shape, shape![(Int::<2>, Int::<3>), (Int::<4>, Int::<2>)]);
    assert_eq!(F.shape, shape![Int::<2>, Int::<3>, Int::<4>, Int::<2>]);
    assert_eq!(
        TILED.shape,
        shape![(Int::<2>, Int::<3>), Int::<4>, Int::<2>]
    );
    assert_eq!((Z, F, TILED), (AZ, AF, AT));
}

#[test]
fn mixed_division_checks_shapes_strict_divisibility_and_composition() {
    let input = Layout::new(DynInt(10), Int::<1>).unwrap();
    assert_eq!(
        input.logical_divide_axes(DynInt(3)),
        Err(Error::NotDivisible)
    );
    assert_eq!(
        input.logical_divide_axes(DynInt(0)),
        Err(Error::InvalidShape)
    );
    assert_eq!(input.logical_divide(DynInt(0)), Err(Error::InvalidShape));
    assert_eq!(input.logical_divide(DynInt(-1)), Err(Error::InvalidShape));
    let outer = Layout::new(shape![Int::<4>, Int::<6>], stride![Int::<6>, Int::<1>]).unwrap();
    let bad = Layout::new(DynInt(8), DynInt(5)).unwrap();
    assert_eq!(outer.logical_divide(bad), Err(Error::NotDivisible));
    let dynamic = dyn_layout(&[8], &[2]);
    let tile = dyn_layout(&[2], &[1]);
    assert_eq!(
        values(&dynamic.logical_divide(tile).unwrap()),
        values(&logical_divide(dynamic, tile).unwrap())
    );
    assert_eq!(
        dynamic.zipped_divide_modes(&[tile]).unwrap(),
        zipped_divide_modes(&dynamic, &[tile]).unwrap()
    );
    assert_eq!(
        values(&dynamic.flat_divide_axes(&[2]).unwrap()),
        [0, 2, 4, 6, 8, 10, 12, 14]
    );
}

#[test]
fn typed_division_can_return_more_than_max_rank_leaves() {
    macro_rules! broadcast {
        ($($slot:literal),*) => { Layout {
            shape: shape![$(Int::<{2 + 0 * $slot}>),*],
            stride: stride![$(Int::<{0 * $slot}>),*],
        } };
    }
    let tile = broadcast!(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32
    );
    let input = Layout::new(Int::<7>, Int::<1>).unwrap();
    let result = static_flat_divide(input, tile);
    assert_eq!(result.rank(), MAX_RANK + 2);
    assert_eq!(result.as_flat_layout(), Err(Error::CapacityExceeded));
    assert_eq!(result.index(1isize << 33), Ok(1));
}

#[test]
fn integral_and_singleton_tuple_tilers_have_distinct_profiles() {
    let input = Layout::new(Int::<8>, Int::<1>).unwrap();
    let integral = input.zipped_divide(Int::<2>).unwrap();
    let singleton = input.zipped_divide(int_tuple![Int::<2>]).unwrap();
    assert_eq!(integral.shape, shape![Int::<2>, Int::<4>]);
    assert_eq!(singleton.shape, shape![(Int::<2>), (Int::<4>)]);
    let scalar_tile = input
        .logical_divide(Layout::new(Nil, Nil).unwrap())
        .unwrap();
    assert_eq!(scalar_tile.shape, shape![(), Int::<8>]);
}

#[test]
fn tiled_and_flat_divide_unpack_modes_after_composition() {
    // CuTe: logical_divide((8,6):(1,16), 2:1) = (2:1, (4,6):(2,16)); the rest mode splits
    // across both input modes, and tiled/flat_divide unpack it after composing.
    let layout = dyn_layout(&[8, 6], &[1, 16]);
    let tile = dyn_layout(&[2], &[1]);
    let typed = Layout::new(shape![Int::<8>, Int::<6>], stride![Int::<1>, Int::<16>]).unwrap();
    let typed_tile = Layout::new(Int::<2>, Int::<1>).unwrap();

    let logical = logical_divide(layout, tile).unwrap();
    assert_grouped(
        &logical.as_affine().unwrap().unwrap(),
        &[2, 4, 6],
        &[1, 2, 16],
        &[1, 2],
    );
    assert_same_map(&logical, &typed.logical_divide(typed_tile).unwrap());

    let zipped = zipped_divide(layout, tile).unwrap();
    assert_grouped(
        &zipped.as_affine().unwrap().unwrap(),
        &[2, 4, 6],
        &[1, 2, 16],
        &[1, 2],
    );
    assert_same_map(&zipped, &typed.zipped_divide(typed_tile).unwrap());

    let tiled = tiled_divide(layout, tile).unwrap();
    assert_eq!(tiled.rank(), 3);
    assert_eq!(tiled.index_of_coord(&[1, 3, 5]), Ok(1 + 6 + 80));
    assert_grouped(
        &tiled.as_affine().unwrap().unwrap(),
        &[2, 4, 6],
        &[1, 2, 16],
        &[1, 1, 1],
    );
    assert_same_map(&tiled, &typed.tiled_divide(typed_tile).unwrap());

    let flat = flat_divide(layout, tile).unwrap();
    assert_grouped(
        &flat.as_affine().unwrap().unwrap(),
        &[2, 4, 6],
        &[1, 2, 16],
        &[1, 1, 1],
    );
    assert_same_map(&flat, &typed.flat_divide(typed_tile).unwrap());
}
