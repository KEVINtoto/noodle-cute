//! Grouping、TileRest 与跨模块布局代数契约。
//!
//! Expected shapes, strides, top-level groupings, and address sequences come
//! from independent coordinate formulas and upstream CUTLASS/CuTe oracle cases,
//! rather than computing expectations through another transformation under test.

#![feature(min_adt_const_params)]

use noodle_cute::Error;
use noodle_cute::algebra::logical_divide;
use noodle_cute::algebra::logical_product;
use noodle_cute::algebra::{Grouping, TileRest};
use noodle_cute::algebra::{left_inverse, right_inverse};
use noodle_cute::layout::{DynLayout, GroupedLayout, LayoutMap};

pub mod common;
use common::{TableMap, assert_grouped, dyn_layout, grouped, values};

// ---------------------------------------------------------------------------
// 1. Grouping & TileRest (`src/algebra.rs`)
// ---------------------------------------------------------------------------

#[test]
fn grouping_and_tile_rest_expose_expected_ranks_extents_and_coordinates() {
    assert_ne!(Grouping::Logical, Grouping::Zipped);
    assert_ne!(Grouping::Tiled, Grouping::Flat);
    let copied = Grouping::Logical;
    assert_eq!(copied, Grouping::Logical);

    let tile = dyn_layout(&[2, 3], &[1, 8]);
    let rest = dyn_layout(&[2, 2], &[2, 24]);

    let make = |grouping| TileRest::new(tile, rest, grouping);
    let logical = make(Grouping::Logical);
    let zipped = make(Grouping::Zipped);
    let tiled = make(Grouping::Tiled);
    let flat = make(Grouping::Flat);

    assert_eq!(logical.rank(), 2);
    assert_eq!(zipped.rank(), 2);
    assert_eq!(tiled.rank(), 3);
    assert_eq!(flat.rank(), 4);

    // Logical and Zipped both expose two axes: the tile dimensions are
    // combined into 2 * 3 = 6, and the rest dimensions into 2 * 2 = 4.
    // Their grouping affects coordinate mapping, not these extents.
    assert_eq!(
        (logical.extent(0), logical.extent(1), logical.extent(2)),
        (Ok(6), Ok(4), Err(Error::AxisOutOfBounds))
    );
    assert_eq!(
        (zipped.extent(0), zipped.extent(1), zipped.extent(2)),
        (Ok(6), Ok(4), Err(Error::AxisOutOfBounds))
    );
    assert_eq!(
        (
            tiled.extent(0),
            tiled.extent(1),
            tiled.extent(2),
            tiled.extent(3)
        ),
        (Ok(6), Ok(2), Ok(2), Err(Error::AxisOutOfBounds))
    );
    assert_eq!(
        (
            flat.extent(0),
            flat.extent(1),
            flat.extent(2),
            flat.extent(3),
            flat.extent(4)
        ),
        (Ok(2), Ok(3), Ok(2), Ok(2), Err(Error::AxisOutOfBounds))
    );

    // Every grouping traverses the exact same linear sequence (tile fastest).
    for variant in [&logical, &zipped, &tiled, &flat] {
        assert_eq!(variant.size(), Ok(24));
        assert_eq!(variant.index(-1), Err(Error::CoordinateOutOfBounds));
        assert_eq!(variant.index(24), Err(Error::CoordinateOutOfBounds));
    }

    for r1 in 0..2 {
        for r0 in 0..2 {
            for t1 in 0..3 {
                for t0 in 0..2 {
                    let t = t0 + 2 * t1;
                    let r = r0 + 2 * r1;
                    let linear = t + 6 * r;
                    let expected = t0 + 8 * t1 + 2 * r0 + 24 * r1;

                    assert_eq!(logical.index(linear), Ok(expected));
                    assert_eq!(zipped.index(linear), Ok(expected));
                    assert_eq!(tiled.index(linear), Ok(expected));
                    assert_eq!(flat.index(linear), Ok(expected));

                    assert_eq!(logical.index_of_coord(&[t, r]), Ok(expected));
                    assert_eq!(zipped.index_of_coord(&[t, r]), Ok(expected));
                    assert_eq!(tiled.index_of_coord(&[t, r0, r1]), Ok(expected));
                    assert_eq!(flat.index_of_coord(&[t0, t1, r0, r1]), Ok(expected));
                }
            }
        }
    }

    // Rank-0 scalar operands collapse cleanly in Tiled and Flat hierarchies.
    let scalar = dyn_layout(&[], &[]);
    let scalar_rest = TileRest::new(tile, scalar, Grouping::Tiled);
    assert_eq!(scalar_rest.tile(), &tile);
    assert_eq!(scalar_rest.rest(), &scalar);
    assert_eq!(scalar_rest.grouping(), Grouping::Tiled);
    assert_eq!(scalar_rest.rank(), 1);
    assert_eq!(scalar_rest.extent(0), Ok(6));
    assert_eq!(scalar_rest.extent(1), Err(Error::AxisOutOfBounds));
    assert_eq!(
        scalar_rest.index_extended(7),
        scalar_rest.as_affine().unwrap().unwrap().index_extended(7)
    );

    let scalar_tile = TileRest::new(scalar, rest, Grouping::Flat);
    assert_eq!(scalar_tile.rank(), 2);
    assert_eq!(scalar_tile.extent(0), Ok(2));
    assert_eq!(scalar_tile.extent(1), Ok(2));
}

#[test]
fn tile_rest_extended_indexing_wraps_tile_and_extends_only_rest() {
    let tr = TileRest::new(
        dyn_layout(&[2, 2], &[1, 4]),
        dyn_layout(&[3, 2], &[2, 8]),
        Grouping::Tiled,
    );
    let affine = tr.as_affine().unwrap().unwrap();
    for i in 0..24 {
        assert_eq!(tr.index_extended(i), tr.index(i));
    }
    for i in 24..40 {
        // Tile wraps modulo 4; rest extends along its final leaf (stride 8).
        let t = i % 4;
        let r = i / 4;
        let expected = (t % 2) + 4 * (t / 2) + 2 * (r % 3) + 8 * (r / 3);
        assert_eq!(tr.index_extended(i), Ok(expected));
        assert_eq!(affine.index_extended(i), Ok(expected));
    }
    assert_eq!(tr.index_extended(-1), Err(Error::CoordinateOutOfBounds));
    assert_eq!(
        tr.index_extended_of_coord(&[5, 4, 3]),
        affine.index_extended_of_coord(&[5, 4, 3])
    );

    // An empty tile cannot wrap modulo zero in the extended domain.
    let empty_tile = TileRest::new(
        dyn_layout(&[0], &[1]),
        dyn_layout(&[3], &[1]),
        Grouping::Logical,
    );
    assert_eq!(empty_tile.size(), Ok(0));
    assert_eq!(empty_tile.index(0), Err(Error::CoordinateOutOfBounds));
    assert_eq!(
        empty_tile.index_extended(0),
        Err(Error::CoordinateOutOfBounds)
    );

    // A non-affine rest works inside the logical domain and rejects extension beyond it.
    let non_affine = TileRest::new(
        dyn_layout(&[2], &[1]),
        TableMap(&[0, 10]),
        Grouping::Logical,
    );
    assert_eq!(non_affine.as_affine(), Ok(None));
    assert_eq!(values(&non_affine), [0, 1, 10, 11]);
    assert_eq!(non_affine.index_extended(3), Ok(11));
    assert_eq!(
        non_affine.index_extended(4),
        Err(Error::CoordinateOutOfBounds)
    );
}

#[test]
fn tile_rest_affine_export_injectivity_and_capacity_or_overflow_errors() {
    let tile = grouped(&[2, 2, 3], &[1, 2, 4], &[2, 1]);
    let rest = grouped(&[2, 3, 2], &[12, 24, 72], &[1, 2]);

    for (grouping, expected_groups) in [
        (Grouping::Logical, &[3usize, 3][..]),
        (Grouping::Zipped, &[3, 3]),
        (Grouping::Tiled, &[3, 1, 2]),
        (Grouping::Flat, &[2, 1, 1, 2]),
    ] {
        let tr = TileRest::new(tile, rest, grouping);
        let exported = tr.as_affine().unwrap().unwrap();
        assert_grouped(
            &exported,
            &[2, 2, 3, 2, 3, 2],
            &[1, 2, 4, 12, 24, 72],
            expected_groups,
        );
        assert_eq!(tr.is_injective(), Ok(true));
    }

    let overlapping = TileRest::new(
        dyn_layout(&[2], &[1]),
        dyn_layout(&[2], &[1]),
        Grouping::Logical,
    );
    assert_eq!(overlapping.is_injective(), Ok(false));

    // Combined leaves exceeding MAX_RANK fail during leaf collection, while index_extended
    // and is_injective fall back to component-wise / enumerated evaluation.
    let mut shapes_20 = [1isize; 20];
    let mut strides_20 = [0isize; 20];
    shapes_20[19] = 2;
    strides_20[19] = 1;
    let mut shapes_13 = [1isize; 13];
    let mut strides_13 = [0isize; 13];
    shapes_13[12] = 2;
    strides_13[12] = 10;
    let leaves_20 = dyn_layout(&shapes_20, &strides_20);
    let leaves_13 = dyn_layout(&shapes_13, &strides_13);
    let too_many_leaves = TileRest::new(leaves_20, leaves_13, Grouping::Logical);
    assert_eq!(too_many_leaves.as_affine(), Err(Error::CapacityExceeded));
    assert_eq!(too_many_leaves.index_extended(5), Ok(21));
    assert_eq!(too_many_leaves.is_injective(), Ok(true));

    // Empty 0-leaf modes keep total leaves small while exceeding MAX_RANK groups.
    let mut mode_groups = [0usize; 20];
    mode_groups[0] = 1;
    let modes_20 = GroupedLayout::new(dyn_layout(&[2], &[1]), &mode_groups).unwrap();
    let too_many_groups = TileRest::new(modes_20, modes_20, Grouping::Flat);
    assert_eq!(too_many_groups.as_affine(), Err(Error::CapacityExceeded));
    assert_eq!(too_many_groups.index_extended(5), Ok(3));

    // When `rest` has 0 affine leaves and group count exceeds MAX_RANK, the fallback
    // still extends along `tile`'s last leaf rather than dividing by `tile_size`.
    let empty_modes_20 = GroupedLayout::new(dyn_layout(&[], &[]), &[0; 20]).unwrap();
    let scalar_rest_overflow = TileRest::new(modes_20, empty_modes_20, Grouping::Flat);
    assert_eq!(
        scalar_rest_overflow.as_affine(),
        Err(Error::CapacityExceeded)
    );
    assert_eq!(scalar_rest_overflow.index_extended(5), Ok(5));

    // Size and address arithmetic overflow must be reported rather than wrapping.
    let huge = dyn_layout(&[1 << 32], &[0]);
    let size_overflow = TileRest::new(huge, huge, Grouping::Logical);
    assert_eq!(size_overflow.size(), Err(Error::Overflow));

    let addr_overflow = TileRest::new(
        dyn_layout(&[2], &[isize::MAX]),
        dyn_layout(&[2], &[1]),
        Grouping::Logical,
    );
    assert_eq!(addr_overflow.index(3), Err(Error::Overflow));
    assert_eq!(addr_overflow.as_affine(), Err(Error::Overflow));
}

// ---------------------------------------------------------------------------
// 7. Cross-Module Algebraic Laws & Capacity Boundaries
// ---------------------------------------------------------------------------

#[test]
fn product_divide_and_inverse_form_round_trip_bijections() {
    // `logical_product((2,2):(1,4), 6:1)` tiles `[0, 24)` identically to
    // `logical_divide(24:1, (2,2):(1,4))`, and both structural inverses invert it.
    let tile = dyn_layout(&[2, 2], &[1, 4]);
    let prod = logical_product(tile, dyn_layout(&[6], &[1]))
        .unwrap()
        .as_affine()
        .unwrap()
        .unwrap();
    let div = logical_divide(DynLayout::left(&[24]).unwrap(), tile)
        .unwrap()
        .as_affine()
        .unwrap()
        .unwrap();
    assert_eq!(prod, div);

    let linv = left_inverse(prod).unwrap();
    let rinv = right_inverse(prod).unwrap();
    assert_eq!(linv, rinv);
    for i in 0..24 {
        let addr = prod.index(i).unwrap();
        assert_eq!(linv.index(addr), Ok(i));
        assert_eq!(prod.index(rinv.index(i).unwrap()), Ok(i));
    }
}
