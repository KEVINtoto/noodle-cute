//! Contract tests for CuTe-style layout algebra: `Grouping`, `TileRest`,
//! `complement`, `composition`, `divide`, `inverse`, and `product`.
//!
//! Expected shapes, strides, top-level groupings, and address sequences come
//! from independent coordinate formulas and upstream CUTLASS/CuTe oracle cases,
//! rather than computing expectations through another transformation under test.

use noodle_cute::algebra::complement::{complement, complement_extended, complement_shape};
use noodle_cute::algebra::composition::{Composition, composition, composition_layout};
use noodle_cute::algebra::divide::{
    Divided, DividedAxes, flat_divide, flat_divide_axes, flat_divide_modes, logical_divide,
    logical_divide_axes, logical_divide_modes, tiled_divide, tiled_divide_axes, tiled_divide_modes,
    zipped_divide, zipped_divide_axes, zipped_divide_modes,
};
use noodle_cute::algebra::inverse::{
    Inverse, dense_left_inverse, dense_right_inverse, left_inverse, right_inverse,
};
use noodle_cute::algebra::product::{
    Product, blocked_product, flat_product, logical_product, raked_product, tiled_product,
    zipped_product,
};
use noodle_cute::algebra::{Grouping, TileRest};
use noodle_cute::int::{DynInt, Int};
use noodle_cute::layout::{DynLayout, GroupedLayout, Layout, LayoutMap, MAX_RANK};
use noodle_cute::{Error, Nil, Result, shape, stride};

fn dyn_layout(shape: &[isize], stride: &[isize]) -> DynLayout {
    DynLayout::new(shape, stride).unwrap()
}

fn grouped(shape: &[isize], stride: &[isize], groups: &[usize]) -> GroupedLayout {
    GroupedLayout::new(dyn_layout(shape, stride), groups).unwrap()
}

fn assert_grouped(actual: &GroupedLayout, shape: &[isize], stride: &[isize], groups: &[usize]) {
    assert_eq!(actual.shapes(), shape, "shape of {actual:?}");
    assert_eq!(actual.strides(), stride, "stride of {actual:?}");
    assert_eq!(
        actual.groups().collect::<Vec<_>>(),
        groups,
        "groups of {actual:?}"
    );
}

fn values<L: LayoutMap>(layout: &L) -> Vec<isize> {
    (0..layout.size().unwrap())
        .map(|i| layout.index(i).unwrap())
        .collect()
}

/// Non-affine lookup map (`as_affine() == Ok(None)`) used to test lazy fallback
/// paths that represent swizzled or otherwise non-affine layouts.
#[derive(Clone, Copy, Debug)]
struct TableMap(&'static [isize]);

impl LayoutMap for TableMap {
    fn rank(&self) -> usize {
        1
    }

    fn extent(&self, axis: usize) -> Result<isize> {
        if axis == 0 {
            Ok(self.0.len() as isize)
        } else {
            Err(Error::AxisOutOfBounds)
        }
    }

    fn size(&self) -> Result<isize> {
        Ok(self.0.len() as isize)
    }

    fn index(&self, linear: isize) -> Result<isize> {
        if linear < 0 {
            return Err(Error::CoordinateOutOfBounds);
        }
        self.0
            .get(linear as usize)
            .copied()
            .ok_or(Error::CoordinateOutOfBounds)
    }

    fn is_injective(&self) -> Result<bool> {
        for (i, &a) in self.0.iter().enumerate() {
            if self.0[..i].contains(&a) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn as_affine(&self) -> Result<Option<GroupedLayout>> {
        Ok(None)
    }
}

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
// 2. Complement (`src/algebra/complement.rs`)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 3. Composition (`src/algebra/composition.rs`)
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

// ---------------------------------------------------------------------------
// 5. Inverse (`src/algebra/inverse.rs`)
// ---------------------------------------------------------------------------

#[test]
fn structural_right_and_left_inverses_match_cute_and_satisfy_inverse_identities() {
    let nested = Layout::new(
        shape![(Int::<2>, Int::<2>), Int::<3>],
        stride![(Int::<6>, Int::<1>), Int::<2>],
    )
    .unwrap();

    type InverseCase = (
        &'static [isize],
        &'static [isize],
        &'static [isize],
        &'static [isize],
    );

    // `right_inverse` advances only along contiguous stride prefixes, skipping
    // broadcast, negative, overlapping, and post-hole modes while preserving preprod.
    let right_cases: [InverseCase; 9] = [
        (&[], &[], &[1], &[0]),
        (&[4], &[1], &[4], &[1]),
        (&[4], &[2], &[1], &[0]),
        (&[4, 2], &[2, 1], &[2, 4], &[4, 1]),
        (&[3, 2], &[0, 1], &[2], &[3]),
        (&[2, 3, 2], &[1, 0, 2], &[2, 2], &[1, 6]),
        (&[2, 4], &[1, 10], &[2], &[1]),
        (&[2, 3], &[-1, 2], &[1], &[0]),
        (&[2, 2], &[1, 1], &[2], &[1]),
    ];
    for (s, d, rs, rd) in right_cases {
        let layout = dyn_layout(s, d);
        let inv = right_inverse(layout).unwrap();
        assert_eq!((inv.shapes(), inv.strides()), (rs, rd), "{s:?}:{d:?}");
        for a in 0..inv.size().unwrap() {
            assert_eq!(layout.index(inv.index(a).unwrap()), Ok(a));
        }
    }
    let nested_rinv = right_inverse(nested).unwrap();
    assert_eq!(
        (nested_rinv.shapes(), nested_rinv.strides()),
        (&[6, 2][..], &[2, 1][..])
    );

    // `left_inverse` absorbs address holes into mixed-radix extents and ignores stride-0 leaves.
    let left_cases: [InverseCase; 8] = [
        (&[], &[], &[1], &[0]),
        (&[4], &[1], &[4], &[1]),
        (&[4], &[2], &[2, 4], &[0, 1]),
        (&[4, 2], &[2, 1], &[2, 4], &[4, 1]),
        (&[2, 2], &[1, 6], &[6, 2], &[1, 2]),
        (&[3, 2], &[1, 8], &[8, 2], &[1, 3]),
        (&[3, 2], &[0, 1], &[2], &[3]),
        (&[3], &[0], &[3], &[0]),
    ];
    for (s, d, rs, rd) in left_cases {
        let layout = dyn_layout(s, d);
        let inv = left_inverse(layout).unwrap();
        assert_eq!((inv.shapes(), inv.strides()), (rs, rd), "{s:?}:{d:?}");
        if layout.is_injective().unwrap() {
            for i in 0..layout.size().unwrap() {
                assert_eq!(inv.index(layout.index(i).unwrap()), Ok(i));
            }
        } else {
            // With broadcast leaves ignored, `left_inverse` acts as a generalized inverse L(R(L(i))) == L(i).
            for i in 0..layout.size().unwrap() {
                let addr = layout.index(i).unwrap();
                assert_eq!(layout.index(inv.index(addr).unwrap()), Ok(addr));
            }
        }
    }
    let nested_linv = left_inverse(nested).unwrap();
    assert_eq!(
        (nested_linv.shapes(), nested_linv.strides()),
        (&[6, 2][..], &[2, 1][..])
    );
    assert_eq!(
        values(&left_inverse(dyn_layout(&[2, 2], &[1, 6])).unwrap()),
        [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7]
    );
    // Unit negative strides coalesce away before `left_inverse` checks signs.
    assert_eq!(
        left_inverse(dyn_layout(&[1, 3], &[-5, 1])),
        Ok(dyn_layout(&[3], &[1]))
    );

    // Exhaustive round-trip over compact row-major permutations.
    for m in 1..7 {
        for n in 1..7 {
            let layout = DynLayout::right(&[m, n]).unwrap();
            let linv = left_inverse(layout).unwrap();
            let rinv = right_inverse(layout).unwrap();
            assert_eq!(linv, rinv);
            for j in 0..n {
                for i in 0..m {
                    let address = i * n + j;
                    let logical = i + j * m;
                    assert_eq!(linv.index(address), Ok(logical));
                    assert_eq!(layout.index(logical), Ok(address));
                }
            }
        }
    }

    // Coalescing runs before stride-chain divisibility: `(2,2,2):(1,2,5)` first merges
    // `(2,2):(1,2)` into `4:1`, so `covered` stays 1 and `5 % 1 == 0` succeeds as `(5,2):(1,4)`.
    assert_eq!(
        left_inverse(dyn_layout(&[2, 2, 2], &[1, 2, 5])),
        Ok(dyn_layout(&[5, 2], &[1, 4]))
    );

    // Error bounds for `right_inverse` and `left_inverse`.
    for empty in [
        dyn_layout(&[0], &[1]),
        dyn_layout(&[0], &[2]),
        dyn_layout(&[0, 2], &[0, 1]),
    ] {
        assert_eq!(right_inverse(empty), Err(Error::Unsupported));
        assert_eq!(left_inverse(empty), Err(Error::Unsupported));
    }
    assert_eq!(
        left_inverse(dyn_layout(&[2], &[-1])),
        Err(Error::Unsupported)
    );
    assert_eq!(
        left_inverse(dyn_layout(&[2, 2], &[2, 3])),
        Err(Error::NotDivisible)
    );
    assert_eq!(
        left_inverse(dyn_layout(&[2, 2, 2], &[1, 4, 6])),
        Err(Error::NotDivisible)
    );
    for overlapping in [
        dyn_layout(&[2, 2], &[1, 1]),
        dyn_layout(&[3, 2], &[1, 2]),
        dyn_layout(&[3, 2, 2], &[1, 2, 6]),
    ] {
        assert_eq!(left_inverse(overlapping), Err(Error::NonInjective));
    }
    // 32 non-coalescing leaves starting at stride 1 fit in MAX_RANK for both inverses;
    // starting at stride 2 requires 33 output slots (a leading hole plus 32 leaves).
    let mut rev_stride = [0isize; MAX_RANK];
    let mut rev_stride_from_2 = [0isize; MAX_RANK];
    for i in 0..MAX_RANK {
        rev_stride[i] = 1isize << (MAX_RANK - 1 - i);
        rev_stride_from_2[i] = 1isize << (MAX_RANK - i);
    }
    let rev_32 = dyn_layout(&[2; MAX_RANK], &rev_stride);
    assert_eq!(right_inverse(rev_32).unwrap().rank(), MAX_RANK);
    assert_eq!(left_inverse(rev_32).unwrap().rank(), MAX_RANK);
    assert_eq!(
        left_inverse(dyn_layout(&[2; MAX_RANK], &rev_stride_from_2)),
        Err(Error::CapacityExceeded)
    );
}

#[test]
fn dense_enumerated_inverses_invert_non_affine_bijections_and_broadcast_surjections() {
    // Non-affine Gray-code bijection onto [0, 8).
    let swizzled = TableMap(&[0, 1, 3, 2, 4, 5, 7, 6]);
    let linv: Inverse<_> = dense_left_inverse(swizzled).unwrap();
    let rinv = dense_right_inverse(swizzled).unwrap();

    assert_eq!(linv.layout().size(), Ok(8));
    assert_eq!(linv.rank(), 1);
    assert_eq!(linv.size(), Ok(8));
    assert_eq!(linv.extent(0), Ok(8));
    assert_eq!(linv.extent(1), Err(Error::AxisOutOfBounds));
    assert_eq!(linv.is_injective(), Ok(true));
    assert_eq!(linv.as_affine(), Ok(None));
    assert_eq!(linv.index(-1), Err(Error::CoordinateOutOfBounds));
    assert_eq!(linv.index(8), Err(Error::CoordinateOutOfBounds));
    assert_eq!(linv.index_extended(8), Err(Error::CoordinateOutOfBounds));

    for i in 0..8 {
        let addr = swizzled.index(i).unwrap();
        assert_eq!(linv.index(addr), Ok(i));
        assert_eq!(rinv.index(addr), Ok(i));
        assert_eq!(swizzled.index(rinv.index(i).unwrap()), Ok(i));
    }

    // Empty domains are vacuously dense bijections/surjections of size 0.
    let inv = dense_left_inverse(TableMap(&[])).unwrap();
    assert_eq!(inv.size(), Ok(0));
    assert_eq!(inv.address_range(), Ok(None));
    assert_eq!(dense_right_inverse(TableMap(&[])).unwrap().size(), Ok(0));
    assert_eq!(
        dense_left_inverse(dyn_layout(&[0], &[1])).unwrap().size(),
        Ok(0)
    );
    assert_eq!(
        dense_right_inverse(dyn_layout(&[0], &[1])).unwrap().size(),
        Ok(0)
    );

    // `dense_right_inverse` chooses the smallest logical preimage for broadcast/overlapping maps.
    let broadcast = dyn_layout(&[3, 2], &[0, 1]);
    assert_eq!(
        dense_left_inverse(broadcast).err(),
        Some(Error::NonInjective)
    );
    let dense_bcast = dense_right_inverse(broadcast).unwrap();
    assert_eq!(values(&dense_bcast), [0, 3]);

    // `(2,2):(1,1)` reaches {0, 1, 1, 2}: dense_right_inverse covers all 3 addresses,
    // unlike structural `right_inverse` which stops at `2:1`.
    let sum_map = dyn_layout(&[2, 2], &[1, 1]);
    assert_eq!(values(&dense_right_inverse(sum_map).unwrap()), [0, 1, 3]);

    let non_affine_surj = TableMap(&[2, 0, 2, 1, 0]);
    assert_eq!(
        values(&dense_right_inverse(non_affine_surj).unwrap()),
        [1, 3, 0]
    );

    // Sparse holes, non-zero minimum addresses, negative addresses, and endpoint overflow.
    // Dense affine bijections use the O(leaves^2) stride-separation proof without enumeration,
    // and sparse maps with `address_end > domain_size` fail immediately by pigeonhole principle.
    let huge_identity = dyn_layout(&[1_000_000], &[1]);
    assert_eq!(
        dense_left_inverse(huge_identity).unwrap().size(),
        Ok(1_000_000)
    );
    assert_eq!(
        dense_right_inverse(huge_identity).unwrap().size(),
        Ok(1_000_000)
    );
    assert_eq!(
        dense_right_inverse(dyn_layout(&[1_000_000], &[2])).err(),
        Some(Error::NotSurjective)
    );
    assert_eq!(
        dense_left_inverse(dyn_layout(&[3], &[2])).err(),
        Some(Error::NotSurjective)
    );
    assert_eq!(
        dense_right_inverse(dyn_layout(&[3], &[2])).err(),
        Some(Error::NotSurjective)
    );
    // Range is [0, 2], so the endpoint check passes and the inner scan catches missing address 1.
    assert_eq!(
        dense_right_inverse(TableMap(&[0, 0, 2])).err(),
        Some(Error::NotSurjective)
    );
    assert_eq!(
        dense_right_inverse(TableMap(&[1, 2, 3])).err(),
        Some(Error::NotSurjective)
    );
    assert_eq!(
        dense_right_inverse(dyn_layout(&[2], &[-1])).err(),
        Some(Error::NotSurjective)
    );
    assert_eq!(
        dense_right_inverse(dyn_layout(&[2], &[isize::MAX])).err(),
        Some(Error::Overflow)
    );
}

// ---------------------------------------------------------------------------
// 6. Product (`src/algebra/product.rs`)
// ---------------------------------------------------------------------------

#[test]
fn product_variants_repeat_tiles_across_cotargetless_complements() {
    let blk = dyn_layout(&[2, 2], &[1, 4]);
    let t23 = grouped(&[2, 3], &[1, 2], &[1, 1]);

    let logical: Product<_, _> = logical_product(blk, t23).unwrap();
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
    assert_eq!(
        logical_product(dyn_layout(&[2], &[1]), dyn_layout(&[2], &[-1])).err(),
        Some(Error::CoordinateOutOfBounds)
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

    // Error bounds: indivisibility against `complement_extended(block)`, negative tiler addresses,
    // and > MAX_RANK leaves.
    assert_eq!(
        blocked_product(dyn_layout(&[4], &[2]), dyn_layout(&[2], &[3])),
        Err(Error::NotDivisible)
    );
    assert_eq!(
        blocked_product(dyn_layout(&[2], &[1]), dyn_layout(&[2], &[-1])),
        Err(Error::CoordinateOutOfBounds)
    );
    let rank_17 = DynLayout::left(&[2; 17]).unwrap();
    assert_eq!(
        raked_product(rank_17, rank_17),
        Err(Error::CapacityExceeded)
    );
}

// ---------------------------------------------------------------------------
// 7. Cross-Module Algebraic Laws & Capacity Boundaries
// ---------------------------------------------------------------------------

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

#[test]
fn divide_axes_and_divide_modes_agree_across_all_four_groupings() {
    let base = dyn_layout(&[6, 8], &[11, 1]);
    let tile_extents = [2, 4];
    let mode_tiler = [dyn_layout(&[2], &[1]), dyn_layout(&[4], &[1])];

    let check_pair = |axes: &DividedAxes<DynLayout>, modes: &GroupedLayout| {
        assert_eq!(axes.rank(), modes.rank());
        assert_eq!(axes.size(), modes.size());
        for a in 0..axes.rank() {
            assert_eq!(axes.extent(a), modes.extent(a));
        }
        assert_eq!(values(axes), values(modes));
    };

    check_pair(
        &logical_divide_axes(base, &tile_extents).unwrap(),
        &logical_divide_modes(&base, &mode_tiler).unwrap(),
    );
    check_pair(
        &zipped_divide_axes(base, &tile_extents).unwrap(),
        &zipped_divide_modes(&base, &mode_tiler).unwrap(),
    );
    check_pair(
        &tiled_divide_axes(base, &tile_extents).unwrap(),
        &tiled_divide_modes(&base, &mode_tiler).unwrap(),
    );
    check_pair(
        &flat_divide_axes(base, &tile_extents).unwrap(),
        &flat_divide_modes(&base, &mode_tiler).unwrap(),
    );
}

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
