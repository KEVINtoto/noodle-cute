//! CuTe inverse 的全静态、混合与运行时契约。
use noodle_cute::Error;
use noodle_cute::algebra::inverse::*;
use noodle_cute::layout::{AffineLayoutMap, DynLayout, Layout, LayoutMap, MAX_RANK};

pub mod common;
use common::{TableMap, dyn_layout, values};
use noodle_cute::{DynInt, Int, IntTuple, Nil, shape, stride};

type Pair<A, B> = IntTuple<A, IntTuple<B, Nil>>;

#[test]
fn static_inverses_are_const_and_preserve_the_colexicographic_preimage() {
    const R: Layout<Pair<Int<3>, Int<2>>, Pair<Int<2>, Int<1>>> = static_right_inverse(Layout {
        shape: shape![Int::<2>, Int::<3>],
        stride: stride![Int::<3>, Int::<1>],
    });
    const L: Layout<Pair<Int<3>, Int<2>>, Pair<Int<2>, Int<1>>> = static_left_inverse(Layout {
        shape: shape![Int::<2>, Int::<3>],
        stride: stride![Int::<3>, Int::<1>],
    });
    assert_eq!(R, L);
    assert_eq!(
        (0..6).map(|i| R.index(i).unwrap()).collect::<Vec<_>>(),
        [0, 2, 4, 1, 3, 5]
    );
    const HOLES: Layout<Pair<Int<2>, Int<4>>, Pair<Int<0>, Int<1>>> = static_left_inverse(Layout {
        shape: Int::<4>,
        stride: Int::<2>,
    });
    for i in 0..4 {
        assert_eq!(HOLES.index(2 * i), Ok(i));
    }
    const UNIT: Layout<Int<1>, Int<0>> = static_right_inverse(Layout {
        shape: Nil,
        stride: Nil,
    });
    assert_eq!(UNIT.size(), Ok(1));
}

#[test]
fn mixed_inverse_only_uses_type_known_stride_facts() {
    let input = Layout::new(shape![DynInt(3), Int::<2>], stride![Int::<1>, Int::<3>]).unwrap();
    let right: Layout<DynInt, Int<1>> = input.right_inverse().unwrap();
    assert_eq!(right.size(), Ok(3));
    let left: Layout<Pair<Int<3>, Int<2>>, Pair<Int<1>, DynInt>> = input.left_inverse().unwrap();
    for i in 0..6 {
        assert_eq!(left.index(input.index(i).unwrap()), Ok(i));
    }
    let dynamic = Layout::new(Int::<4>, DynInt(1)).unwrap();
    let right: Layout<Int<1>, Int<0>> = dynamic.right_inverse().unwrap();
    assert_eq!(right.size(), Ok(1));
    assert_eq!(
        dynamic
            .as_flat_layout()
            .unwrap()
            .right_inverse()
            .unwrap()
            .size(),
        Ok(4)
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

#[test]
fn mixed_inverse_errors_broadcasts_and_dynamic_unit_extents_are_checked() {
    let broadcast = Layout::new(DynInt(3), Int::<0>).unwrap();
    let left: Layout<DynInt, Int<0>> = broadcast.left_inverse().unwrap();
    assert_eq!(values(&left), [0, 0, 0]);
    let unit = Layout::new(DynInt(1), Int::<1>).unwrap();
    let right: Layout<DynInt, Int<1>> = unit.right_inverse().unwrap();
    assert_eq!(right.shape, DynInt(1));
    assert_eq!(
        Layout::new(DynInt(0), Int::<1>).unwrap().right_inverse(),
        Err(Error::Unsupported)
    );
    assert_eq!(
        Layout::new(DynInt(0), Int::<1>).unwrap().left_inverse(),
        Err(Error::Unsupported)
    );
    assert_eq!(
        Layout {
            shape: DynInt(-1),
            stride: Int::<1>
        }
        .right_inverse(),
        Err(Error::InvalidShape)
    );
    let overlapping =
        Layout::new(shape![DynInt(4), Int::<2>], stride![Int::<1>, Int::<3>]).unwrap();
    assert_eq!(overlapping.left_inverse(), Err(Error::NonInjective));
    let overflow = Layout::new(DynInt(2), Int::<{ isize::MAX }>).unwrap();
    assert_eq!(overflow.left_inverse(), Err(Error::Overflow));
    assert_eq!(overflow.right_inverse().unwrap().size(), Ok(1));
    let dynamic = dyn_layout(&[2, 3], &[3, 1]);
    assert_eq!(dynamic.right_inverse(), right_inverse(dynamic));
    assert_eq!(dynamic.left_inverse(), left_inverse(dynamic));
}

#[test]
fn static_inversion_exceeds_the_runtime_leaf_capacity() {
    macro_rules! reverse_layout {
        ($($bit:literal),*) => { Layout {
            shape: shape![$(Int::<{2 + 0 * $bit}>),*],
            stride: stride![$(Int::<{1isize << $bit}>),*],
        } };
    }
    let input = reverse_layout!(
        32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0
    );
    let right = static_right_inverse(input);
    let left = static_left_inverse(input);
    assert_eq!(right.rank(), MAX_RANK + 1);
    assert_eq!(right.as_flat_layout(), Err(Error::CapacityExceeded));
    assert_eq!(core::mem::size_of_val(&right), 0);
    for i in [0, 1, 2, 3, 12345, (1isize << 32) + 1] {
        let expected = (0..33).fold(0, |acc, bit| acc | (((i >> bit) & 1) << (32 - bit)));
        assert_eq!(right.index(i), Ok(expected));
        assert_eq!(left.index(i), Ok(expected));
    }
}
