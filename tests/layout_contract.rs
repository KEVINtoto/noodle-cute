//! Contract tests for affine capabilities shared by typed and runtime layouts.
//!
//! Static type assertions cover compact strides and coalescing; independent
//! coordinate fixtures cover address mappings, signed spans, and error bounds.
//!
//! Typed layouts evaluate extended-domain coordinates and prove injectivity
//! directly from their leaves, so neither operation depends on `MAX_RANK`.
//! Exporting runtime metadata through `to_dyn` or `as_affine` is capacity-limited.

use core::mem::{size_of, size_of_val};

use noodle_cute::int::{DynInt, Int};
use noodle_cute::layout::{
    AffineLayoutMap, DynLayout, GroupedLayout, Layout, LayoutLeft, LayoutMap, LayoutRight, MAX_RANK,
};
use noodle_cute::tuple::{Flatten, Item, Product};
use noodle_cute::{Error, IntTuple, Nil, shape, stride};

// Flattened shape and stride are (2, 2, 3):(1, 6, 2), with top-level groups [2, 1].
macro_rules! nested {
    () => {
        Layout::new(
            shape![(Int::<2>, Int::<2>), Int::<3>],
            stride![(Int::<1>, Int::<6>), Int::<2>],
        )
        .unwrap()
    };
}

#[test]
fn typed_layout_runs_into_the_extended_domain() {
    let layout = nested!();
    // Left-major decoding maps 9 to leaf coordinates (1, 0, 2), at address 5.
    assert_eq!(layout.index(9), Ok(5));
    assert_eq!(layout.index(12), Err(Error::CoordinateOutOfBounds));
    // Extended decoding leaves the final quotient unreduced: 13 becomes (1, 0, 3).
    // The final coordinate reaches its extent, yielding address 1 * 1 + 3 * 2.
    assert_eq!(layout.index_extended(13), Ok(7));
    assert_eq!(layout.index_extended(-1), Err(Error::CoordinateOutOfBounds));
    // Extended and bounded indexing must agree throughout the logical domain.
    for i in 0..12 {
        assert_eq!(layout.index_extended(i), layout.index(i));
    }
}

#[test]
fn typed_extended_coordinates_stay_within_their_own_mode() {
    let layout = nested!();
    // Each mode extends independently: mode 0 maps 5 to 1 * 1 + 2 * 6,
    // while mode 1 maps 1 to 1 * 2. No quotient carries between modes.
    assert_eq!(layout.index_extended_of_coord(&[5, 1]), Ok(15));
    assert_eq!(
        layout.index_extended_of_coord(&[1, 2]),
        layout.index_of_coord(&[1, 2])
    );
    assert_eq!(
        layout.index_extended_of_coord(&[1]),
        Err(Error::StructureMismatch)
    );
    assert_eq!(
        layout.index_extended_of_coord(&[-1, 0]),
        Err(Error::CoordinateOutOfBounds)
    );
}

#[test]
fn typed_layout_exports_grouped_affine_metadata() {
    let layout = nested!();
    let flat = layout.as_flat_layout().unwrap();
    assert_eq!(flat.shapes(), &[2, 2, 3]);
    assert_eq!(flat.strides(), &[1, 6, 2]);

    let grouped = layout.as_affine().unwrap().unwrap();
    assert_eq!(grouped.groups().collect::<Vec<_>>(), [2, 1]);
    assert!(!grouped.is_integral());
    // Preserving top-level groups preserves both forms of extended indexing.
    assert_eq!(grouped.index_extended_of_coord(&[5, 1]), Ok(15));
    assert_eq!(grouped.index_extended(13), layout.index_extended(13));

    // A leaf shape represents a CuTe integral shape; a one-element tuple does not.
    let integral = Layout::new(Int::<8>, Int::<1>).unwrap();
    assert!(integral.as_affine().unwrap().unwrap().is_integral());
    let tuple = Layout::new(shape![Int::<8>], stride![Int::<1>]).unwrap();
    assert!(!tuple.as_affine().unwrap().unwrap().is_integral());
}

#[test]
fn typed_injectivity_uses_the_affine_proof() {
    // Roughly 5e11 address pairs make exhaustive comparison impractical here;
    // these large compact layouts must be handled by the affine proof.
    let big = LayoutLeft::make(shape![Int::<1024>, Int::<1024>]).unwrap();
    assert_eq!(big.is_injective(), Ok(true));
    let reversed = Layout::new(
        shape![Int::<1024>, Int::<1024>],
        stride![Int::<-1024>, Int::<-1>],
    )
    .unwrap();
    assert_eq!(reversed.is_injective(), Ok(true));
    // A zero stride with extent greater than one proves overlap without enumeration.
    let broadcast = Layout::new(
        shape![Int::<1024>, Int::<1024>],
        stride![Int::<1>, Int::<0>],
    )
    .unwrap();
    assert_eq!(broadcast.is_injective(), Ok(false));
}

#[test]
fn injectivity_falls_back_to_enumeration_when_the_proof_fails() {
    // (3, 2):(2, 3) reaches distinct addresses {0, 2, 4, 3, 5, 7}, but stride 3
    // does not exceed the accumulated span 4, so the proof requires enumeration.
    // (3, 2):(2, 4) reaches address 4 twice. The singleton and empty cases remain
    // injective despite zero strides: singleton axes add no aliases, and empty
    // domains have no coordinates to compare.
    let cases: [([isize; 2], [isize; 2], bool); 5] = [
        ([3, 2], [2, 3], true),
        ([3, 2], [2, 4], false),
        ([2, 2], [1, 1], false),
        ([1, 4], [0, 1], true),
        ([0, 4], [0, 0], true),
    ];
    for (s, d, expected) in cases {
        let typed = Layout::new(shape![s[0], s[1]], stride![d[0], d[1]]).unwrap();
        let flat = DynLayout::new(&s, &d).unwrap();
        assert_eq!(typed.is_injective(), Ok(expected), "{s:?}:{d:?}");
        assert_eq!(flat.is_injective(), Ok(expected), "{s:?}:{d:?}");
    }
}

#[test]
fn typed_layouts_beyond_max_rank_keep_leaf_level_capabilities() {
    // These 33 leaves have extent 2 and strides 2^i, giving size 2^33.
    // Injectivity must be proved without exporting bounded runtime metadata.
    let layout = Layout::new(
        shape![
            2isize, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2
        ],
        stride![
            1isize,
            1 << 1,
            1 << 2,
            1 << 3,
            1 << 4,
            1 << 5,
            1 << 6,
            1 << 7,
            1 << 8,
            1 << 9,
            1 << 10,
            1 << 11,
            1 << 12,
            1 << 13,
            1 << 14,
            1 << 15,
            1 << 16,
            1 << 17,
            1 << 18,
            1 << 19,
            1 << 20,
            1 << 21,
            1 << 22,
            1 << 23,
            1 << 24,
            1 << 25,
            1 << 26,
            1 << 27,
            1 << 28,
            1 << 29,
            1 << 30,
            1 << 31,
            1 << 32
        ],
    )
    .unwrap();
    assert_eq!(layout.rank(), MAX_RANK + 1);
    assert_eq!(layout.is_injective(), Ok(true));
    // Compact addresses equal linear indices. Beyond the size, indexing extends
    // along the final leaf, whose stride is 2^32.
    assert_eq!(layout.index_extended(12345), Ok(12345));
    assert_eq!(layout.index_extended(1 << 34), Ok(1 << 34));
    let mut coord = [0isize; MAX_RANK + 1];
    coord[MAX_RANK] = 3;
    assert_eq!(layout.index_extended_of_coord(&coord), Ok(3 << 32));

    // Runtime metadata still enforces its fixed capacity for the same layout.
    assert_eq!(layout.as_flat_layout(), Err(Error::CapacityExceeded));
    assert_eq!(layout.as_affine(), Err(Error::CapacityExceeded));
}

#[test]
fn injectivity_reports_overflow_for_unrepresentable_stride_magnitude() {
    // The proof cannot represent |isize::MIN| as an isize; both implementations
    // report arithmetic overflow rather than returning an injectivity result.
    let flat = DynLayout::new(&[2, 2], &[1, isize::MIN]).unwrap();
    assert_eq!(flat.is_injective(), Err(Error::Overflow));
    let typed = Layout::new(shape![2isize, 2], stride![1isize, isize::MIN]).unwrap();
    assert_eq!(typed.is_injective(), Err(Error::Overflow));
}

#[test]
fn address_range_allows_isize_max_endpoint() {
    // Address bounds can fit even when a storage length of max + 1 would not.
    assert_eq!(
        DynLayout::new(&[2], &[isize::MAX]).unwrap().address_range(),
        Ok(Some((0, isize::MAX)))
    );
}

#[test]
fn compact_layouts_preserve_static_values_and_store_dynamic_dependencies() {
    // Type annotations prove static results, while storage sizes catch runtime promotion.
    let _: Int<42> = Int::<6> * Int::<7>;
    let _: Int<9> = Int::<12> - Int::<3>;
    let _: Int<4> = Int::<12> / Int::<3>;
    let _: Int<-1> = Int::<-13> % Int::<4>;
    let _: Int<-3> = -Int::<3>;
    let _: DynInt = Int::<2> + DynInt(4);
    let shape = shape![Int::<2>, (Int::<3>, Int::<4>)];
    let _: Int<24> = shape.product_typed();
    assert_eq!(shape.rank(), 2);
    assert_eq!(shape.depth(), 2);
    assert_eq!(shape.get::<1>().get::<0>().value(), 3);
    let flat = shape.flatten();
    assert_eq!(flat.rank(), 3);
    assert_eq!(flat.depth(), 1);
    let left = LayoutLeft::make(shape).unwrap();
    let _: Int<1> = left.stride.0;
    let _: Int<2> = left.stride.1.0.0;
    let _: Int<6> = left.stride.1.0.1.0;
    assert_eq!(size_of_val(&left), 0);
    let right = LayoutRight::make(shape).unwrap();
    let _: Int<12> = right.stride.0;
    let _: Int<4> = right.stride.1.0.0;
    let _: Int<1> = right.stride.1.0.1.0;
    assert_eq!(size_of_val(&right), 0);
    let mixed = LayoutLeft::make(shape![Int::<2>, DynInt(3), Int::<4>]).unwrap();
    // The dynamic extent and the later stride that depends on it both occupy storage.
    assert_eq!(size_of_val(&mixed), 2 * size_of::<isize>());
}

#[test]
fn coalesce_merges_static_contiguous_axes_and_preserves_dynamic_axes() {
    // Remove static unit axes and merge contiguous leaves using only static facts.
    let original = Layout::new(
        shape![Int::<2>, (Int::<1>, Int::<6>)],
        stride![Int::<1>, (Int::<6>, Int::<2>)],
    )
    .unwrap();
    let coalesced = original.coalesce().unwrap();
    let _: Layout<IntTuple<Int<12>, Nil>, IntTuple<Int<1>, Nil>> = coalesced;
    assert_eq!(size_of_val(&coalesced), 0);
    assert_eq!(coalesced.rank(), 1);
    for i in 0..12 {
        assert_eq!(coalesced.index(i), original.index(i));
    }
    let all_units = Layout::new(shape![Int::<1>, Int::<1>], stride![Int::<7>, Int::<9>])
        .unwrap()
        .coalesce()
        .unwrap();
    let _: Layout<Int<1>, Int<0>> = all_units;
    // A runtime extent equal to one must not change the output type's structure.
    for n in [1, 3] {
        let mixed =
            Layout::new(shape![DynInt(n), DynInt(4)], stride![Int::<1>, DynInt(n)]).unwrap();
        let result = mixed.coalesce().unwrap();
        assert_eq!(result.rank(), 2);
        assert_eq!(size_of_val(&result), 3 * size_of::<isize>());
        for i in 0..n * 4 {
            assert_eq!(result.index(i), mixed.index(i));
        }
    }
}

#[test]
fn typed_coalesce_handles_empty_singleton_and_broadcast_layouts() {
    let scalar = Layout::new(Nil, Nil).unwrap().coalesce().unwrap();
    let _: Layout<Int<1>, Int<0>> = scalar;
    assert_eq!(scalar.index(0), Ok(0));

    let unit = Layout::new(Int::<1>, Int::<8>).unwrap().coalesce().unwrap();
    let _: Layout<Int<1>, Int<0>> = unit;
    assert_eq!(unit.index(0), Ok(0));

    let dynamic_unit = Layout::new(DynInt(1), Int::<8>)
        .unwrap()
        .coalesce()
        .unwrap();
    let _: Layout<IntTuple<DynInt, Nil>, IntTuple<Int<8>, Nil>> = dynamic_unit;
    assert_eq!(dynamic_unit.shape.0, DynInt(1));

    let broadcast = Layout::new(shape![Int::<2>, Int::<3>], stride![Int::<0>, Int::<0>])
        .unwrap()
        .coalesce()
        .unwrap();
    let _: Layout<IntTuple<Int<6>, Nil>, IntTuple<Int<0>, Nil>> = broadcast;
    for i in 0..6 {
        assert_eq!(broadcast.index(i), Ok(0));
    }
}

#[test]
fn runtime_layout_addresses_and_injectivity_match_coordinate_enumeration() {
    // Enumerate empty, padded, broadcast, and reversed layouts with a direct address formula.
    for m in 0..5 {
        for n in 0..5 {
            for sm in -3..4 {
                for sn in -3..4 {
                    let layout = DynLayout::new(&[m, n], &[sm, sn]).unwrap();
                    let mut addresses = Vec::new();
                    for c in 0..n {
                        for r in 0..m {
                            let expected = r * sm + c * sn;
                            addresses.push(expected);
                            assert_eq!(layout.index(r + m * c).unwrap(), expected);
                            assert_eq!(layout.index_of_coord(&[r, c]).unwrap(), expected);
                        }
                    }
                    let range = addresses
                        .iter()
                        .min()
                        .zip(addresses.iter().max())
                        .map(|(&a, &b)| (a, b));
                    assert_eq!(layout.address_range().unwrap(), range);
                    addresses.sort_unstable();
                    let old = addresses.len();
                    addresses.dedup();
                    assert_eq!(layout.is_injective().unwrap(), addresses.len() == old);
                }
            }
        }
    }
}

#[test]
fn layout_construction_checks_invalid_extents_overflow_and_empty_domains() {
    // Empty domains must short-circuit size overflow without hiding invalid extents.
    assert!(matches!(
        DynLayout::new(&[-1], &[1]),
        Err(Error::InvalidShape)
    ));
    assert!(matches!(
        DynLayout::new(&[isize::MAX, 2], &[0, 0]),
        Err(Error::Overflow)
    ));
    assert!(matches!(
        DynLayout::new(&[3], &[isize::MAX]),
        Err(Error::Overflow)
    ));
    let empty = DynLayout::new(&[isize::MAX, 2, 0], &[0, 0, 0]).unwrap();
    assert_eq!(empty.size().unwrap(), 0);
    assert_eq!(empty.address_range().unwrap(), None);
    let empty_typed = Layout::new(
        shape![DynInt(isize::MAX), DynInt(2), DynInt(0)],
        stride![Int::<0>, Int::<0>, Int::<0>],
    )
    .unwrap();
    assert_eq!(empty_typed.size().unwrap(), 0);
    // Compact stride construction can overflow before reaching the zero extent.
    assert!(matches!(
        LayoutLeft::make(shape![DynInt(isize::MAX), DynInt(2), DynInt(0)]),
        Err(Error::Overflow)
    ));
    assert_eq!(
        DynLayout::new(&[2], &[-1]).unwrap().address_end(),
        Err(Error::Unsupported)
    );
    assert_eq!(DynLayout::new(&[2], &[-1]).unwrap().cosize(), Ok(2));
}

#[test]
fn nested_mode_coordinates_decode_in_left_major_order() {
    // CuTe (2,(2,2)):(4,(2,1)) expands the second top-level coordinate
    // as (n % 2, n / 2), independently of the stride order.
    let layout = Layout::new(
        shape![Int::<2>, (DynInt(2), Int::<2>)],
        stride![Int::<4>, (Int::<2>, Int::<1>)],
    )
    .unwrap();
    let expected = [[0, 2, 1, 3], [4, 6, 5, 7]];
    for (m, row) in expected.iter().enumerate() {
        for (n, &address) in row.iter().enumerate() {
            assert_eq!(
                layout.index_of_coord(&[m as isize, n as isize]).unwrap(),
                address
            );
        }
    }
    // Only one shape leaf is dynamic; all stride leaves occupy zero bytes.
    assert_eq!(size_of_val(&layout), size_of::<isize>());
}

#[test]
fn linear_indexing_uses_left_major_order_across_stride_patterns() {
    // The leftmost logical axis varies fastest for compact, padded, and
    // permuted address orders alike.
    let fixtures: &[(&[isize], &[isize], &[isize])] = &[
        (&[2, 4], &[1, 2], &[0, 1, 2, 3, 4, 5, 6, 7]),
        (&[2, 4], &[12, 1], &[0, 12, 1, 13, 2, 14, 3, 15]),
        (&[2, 2, 2], &[4, 2, 1], &[0, 4, 2, 6, 1, 5, 3, 7]),
        (&[4, 2], &[2, 1], &[0, 2, 4, 6, 1, 3, 5, 7]),
    ];
    for &(shape, stride, expected) in fixtures {
        let layout = DynLayout::new(shape, stride).unwrap();
        assert_eq!(layout.size().unwrap() as usize, expected.len());
        for (linear, &address) in expected.iter().enumerate() {
            assert_eq!(layout.index(linear as isize).unwrap(), address);
        }
    }
}

#[test]
fn negative_stride_indexing_preserves_signed_addresses_and_checks_bounds() {
    // Negative strides may produce negative addresses within a valid domain.
    let layout = DynLayout::new(&[3, 5], &[-7, 1]).unwrap();
    for col in 0..5 {
        for row in 0..3 {
            assert_eq!(layout.index(row + col * 3).unwrap(), col - row * 7);
            assert_eq!(layout.index_of_coord(&[row, col]).unwrap(), col - row * 7);
        }
    }
    assert_eq!(layout.index(-1), Err(Error::CoordinateOutOfBounds));
    assert_eq!(layout.index(15), Err(Error::CoordinateOutOfBounds));
    assert!(layout.index_of_coord(&[0, 5]).is_err());
}

#[test]
fn zero_extent_layouts_reject_all_logical_indices_and_coordinates() {
    // A zero extent produces an empty domain, unlike a rank-0 scalar.
    for shape in [[0, 7], [3, 0], [0, 0]] {
        let layout = DynLayout::new(&shape, &[7, 1]).unwrap();
        assert_eq!(layout.size().unwrap(), 0);
        assert!(layout.index(0).is_err());
        assert!(layout.index_of_coord(&[0, 0]).is_err());
    }
}

#[test]
fn cosize_measures_address_span_for_signed_broadcast_and_empty_layouts() {
    // Address spans include holes and measure negative strides by magnitude.
    let cases: [(&[isize], &[isize], isize); 6] = [
        (&[2], &[-1], 2),
        (&[2, 3], &[-1, 3], 8),
        (&[3, 4], &[-7, 2], 21),
        (&[2, 2], &[2, 1], 4),
        (&[4, 3], &[0, 5], 11),
        (&[0], &[1], 0),
    ];
    for (shape, stride, expected) in cases {
        let layout = DynLayout::new(shape, stride).unwrap();
        assert_eq!(layout.cosize(), Ok(expected), "{shape:?}:{stride:?}");
    }

    let nested = Layout::new(
        shape![Int::<3>, (Int::<2>, Int::<2>)],
        stride![Int::<7>, (Int::<2>, Int::<1>)],
    )
    .unwrap();
    assert_eq!(nested.cosize(), Ok(18));

    // A span is valid for negative addresses, but storage from address zero is not.
    let negative = DynLayout::new(&[2, 3], &[-1, 3]).unwrap();
    assert_eq!(negative.address_end(), Err(Error::Unsupported));
}

#[test]
fn extended_indexing_continues_beyond_linear_and_mode_extents() {
    // Extended linear decoding leaves the final leaf's quotient unreduced.
    let flat = DynLayout::new(&[2, 3], &[3, 1]).unwrap();
    assert_eq!(flat.index(7), Err(Error::CoordinateOutOfBounds));
    assert_eq!(flat.index_extended(7), Ok(6));
    assert_eq!(flat.index_extended(13), Ok(9));
    assert_eq!(flat.index_extended_of_coord(&[1, 5]), Ok(8));
    assert_eq!(flat.index_extended(-1), Err(Error::CoordinateOutOfBounds));

    let nested = Layout::new(
        shape![(Int::<2>, Int::<2>), Int::<3>],
        stride![(Int::<1>, Int::<6>), Int::<2>],
    )
    .unwrap();
    assert_eq!(nested.index_extended(9), Ok(5));
    // Each top-level mode extends along its own final leaf.
    assert_eq!(nested.index_extended_of_coord(&[5, 1]), Ok(15));
    for i in 0..12 {
        assert_eq!(nested.index_extended(i), nested.index(i));
    }
}

#[test]
fn grouped_mode_queries_do_not_revalidate_address_ranges() {
    // The empty domain makes the whole address range `None`, so the huge stride
    // is accepted. Evaluating mode 1 alone must not re-run that range check.
    let flat = DynLayout::new(&[0, 3], &[1, isize::MAX]).unwrap();
    let grouped = GroupedLayout::new(flat, &[1, 1]).unwrap();
    assert_eq!(grouped.extent(0), Ok(0));
    assert_eq!(grouped.extent(1), Ok(3));
    assert_eq!(grouped.index_extended_of_coord(&[0, 1]), Ok(isize::MAX));

    let typed = Layout::new(
        shape![Int::<0>, Int::<3>],
        stride![Int::<1>, Int::<{ isize::MAX }>],
    )
    .unwrap();
    assert_eq!(typed.extent(1), grouped.extent(1));
    assert_eq!(
        typed.index_extended_of_coord(&[0, 1]),
        grouped.index_extended_of_coord(&[0, 1])
    );
}

#[test]
fn grouped_layout_rejects_overflowing_group_counts() {
    let flat = DynLayout::new(&[2], &[1]).unwrap();
    for groups in [&[usize::MAX, 2][..], &[usize::MAX, 1, 1], &[2]] {
        assert_eq!(
            GroupedLayout::new(flat, groups),
            Err(Error::StructureMismatch)
        );
    }
    let grouped = GroupedLayout::new(flat, &[0, 1, 0]).unwrap();
    assert_eq!(grouped.groups().collect::<Vec<_>>(), [0, 1, 0]);
    assert_eq!(grouped.extent(0), Ok(1));
    assert_eq!(grouped.mode(1), Ok(flat));
}

#[test]
fn runtime_coalesce_retains_axes_when_merge_predicate_overflows() {
    for stride in [isize::MAX / 2 + 1, isize::MIN / 2 - 1] {
        let layout = DynLayout::new(&[2, 2], &[stride, 1]).unwrap();
        let result = layout.coalesce().unwrap();
        assert_eq!(result.shapes(), &[2, 2]);
        assert_eq!(result.strides(), &[stride, 1]);
        for i in 0..4 {
            assert_eq!(result.index(i), layout.index(i));
        }
    }
    // A genuine merge still must reject an unrepresentable extent, even when
    // a later zero extent makes the original domain empty.
    let empty = DynLayout::new(&[isize::MAX, 2, 0], &[0, 0, 1]).unwrap();
    assert_eq!(empty.coalesce(), Err(Error::Overflow));
}

#[test]
fn filter_revalidates_size_and_address_bounds_after_removing_empty_axes() {
    let address_overflow = DynLayout::new(&[3, 0], &[isize::MAX, 0]).unwrap();
    assert_eq!(address_overflow.filter(), Err(Error::Overflow));
    let size_overflow = DynLayout::new(&[isize::MAX, 2, 0], &[1, 1, 0]).unwrap();
    assert_eq!(size_overflow.filter(), Err(Error::Overflow));

    let valid = DynLayout::new(&[3, 0], &[2, 0]).unwrap().filter().unwrap();
    assert_eq!(valid.size(), Ok(3));
    assert_eq!(valid.address_range(), Ok(Some((0, 4))));
}

#[test]
fn grouped_extent_depends_only_on_mode_shape() {
    let flat = DynLayout::new(&[3, 0], &[isize::MAX, 0]).unwrap();
    let grouped = GroupedLayout::new(flat, &[1, 1]).unwrap();
    assert_eq!(grouped.extent(0), Ok(3));
    assert_eq!(grouped.extent(1), Ok(0));
    assert_eq!(grouped.extent(2), Err(Error::AxisOutOfBounds));
    // Extracting an actual standalone layout must still validate its bounds.
    assert_eq!(grouped.mode(0), Err(Error::Overflow));
    let huge = DynLayout::new(&[isize::MAX, 2, 0], &[0, 0, 0]).unwrap();
    assert_eq!(
        GroupedLayout::new(huge, &[2, 1]).unwrap().extent(0),
        Err(Error::Overflow)
    );
}

#[test]
fn grouped_extended_coordinates_check_only_requested_address_arithmetic() {
    let flat = DynLayout::new(&[3, 0], &[isize::MAX, 0]).unwrap();
    let grouped = GroupedLayout::new(flat, &[1, 1]).unwrap();
    for coord in [[0, 0], [1, 0], [2, 0], [-1, 0]] {
        assert_eq!(
            grouped.index_extended_of_coord(&coord),
            flat.index_extended_of_coord(&coord)
        );
    }
    assert_eq!(
        grouped.index_extended_of_coord(&[0]),
        Err(Error::StructureMismatch)
    );

    let with_empty_mode = GroupedLayout::new(flat, &[0, 1, 1]).unwrap();
    assert_eq!(with_empty_mode.index_extended_of_coord(&[0, 0, 0]), Ok(0));
    assert_eq!(
        with_empty_mode.index_extended_of_coord(&[1, 0, 0]),
        Err(Error::CoordinateOutOfBounds)
    );
}
