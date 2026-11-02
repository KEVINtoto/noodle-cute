//! Contract tests for swizzles, composed layouts, and the layout utilities
//! built on them, following CuTe's `swizzle.hpp`, `swizzle_layout.hpp`,
//! `layout_composed.hpp`, and `layout.hpp`.
//!
//! Swizzled addresses are computed from upstream's definition
//! `offset ^ shiftr(offset & yyy_msk, S)`.

use noodle_cute::algebra::{MaxCommonLayout, NonSwizzlePortion, TileToShape, max_common_vector};
use noodle_cute::coord::X;
use noodle_cute::layout::{LayoutLeft, LayoutRight, MapCoord, SliceAt};
use noodle_cute::swizzle::Swizzle;
use noodle_cute::tensor::make_tensor;
use noodle_cute::{DynInt, Int, Layout, shape, stride};

#[test]
fn swizzle_masks_and_offsets_follow_the_upstream_definition() {
    type Sw = Swizzle<3, 3, 3>;
    assert_eq!((Sw::YYY_MSK, Sw::ZZZ_MSK), (0b111 << 6, 0b111 << 3));
    assert_eq!(Sw::apply(0b101_000_011), 0b101_101_011);
    for x in 0..1024 {
        assert_eq!(Sw::apply(Sw::apply(x)), x);
    }
    // A negative shift moves the YYY bits up into the ZZZ bits.
    type Up = Swizzle<1, 0, -1>;
    assert_eq!((Up::YYY_MSK, Up::ZZZ_MSK), (0b01, 0b10));
    assert_eq!(Up::apply(1), 3);
    assert_eq!(Up::apply(2), 2);
}

#[test]
fn swizzled_layouts_map_and_slice_through_the_offset() {
    // An 8x64 row-major tile: row r XORs r into bits 3..6 of the column.
    let smem = Swizzle::<3, 3, 3>.compose(LayoutRight::make(shape![Int::<8>, Int::<64>]).unwrap());
    for r in 0..8isize {
        for c in 0..64isize {
            assert_eq!(
                smem.map_coord(shape![r, c]),
                DynInt(r * 64 + (c ^ (r << 3)))
            );
        }
    }
    // Slicing keeps the swizzle outside: row 2 adds 128 to the offset, not the pointer.
    let (row, offset) = smem.slice_and_offset(shape![2isize, X]);
    assert_eq!(offset, Int::<0>);
    assert_eq!(row.offset, DynInt(128));
    assert_eq!(row.map_coord(8isize), DynInt(128 + (8 ^ 16)));

    let data: [i32; 512] = core::array::from_fn(|i| i as i32);
    let tile = make_tensor(data.as_ptr(), smem);
    // SAFETY: the swizzle permutes addresses within 0..512.
    assert_eq!(unsafe { tile.read(shape![1i32, 0i32]) }, 72);
    let row = tile.slice(shape![2i32, X]);
    assert_eq!(row.data(), data.as_ptr());
    assert_eq!(unsafe { row.read(8i32) }, 152);
}

#[test]
fn max_common_vector_and_layout_match_upstream() {
    let col = LayoutLeft::make(shape![Int::<4>, Int::<8>]).unwrap();
    let row = LayoutRight::make(shape![Int::<4>, Int::<8>]).unwrap();
    assert_eq!(max_common_vector(&col, &col), 32);
    assert_eq!(max_common_vector(&row, &col), 1);
    // Only the leading contiguous mode counts.
    let padded = Layout::new(shape![Int::<4>, Int::<3>], stride![Int::<1>, Int::<5>]).unwrap();
    assert_eq!(max_common_vector(&padded, &padded), 4);
    // A dynamic stride hides the contiguity of the modes after it.
    let dynamic = Layout::new(shape![Int::<4>, Int::<8>], stride![Int::<1>, DynInt(4)]).unwrap();
    assert_eq!(max_common_vector(&dynamic, &col), 4);

    // A swizzle keeps 2^M unswizzled elements, unless both sides share it.
    let smem = Swizzle::<3, 3, 3>.compose(LayoutRight::make(shape![Int::<8>, Int::<64>]).unwrap());
    let plain = LayoutRight::make(shape![Int::<8>, Int::<64>]).unwrap();
    assert_eq!(max_common_vector(&smem, &plain), 8);
    assert_eq!(max_common_vector(&plain, &smem), 8);
    assert_eq!(max_common_vector(&smem, &smem), 512);
    assert_eq!(
        max_common_vector(&Swizzle::<2, 3, 3>.compose(plain), &smem),
        8
    );
    assert_eq!(
        max_common_vector(&Swizzle::<2, 2, 3>.compose(plain), &smem),
        4
    );

    // composition(right_inverse(b), layout<0>(common)) with the coalesced 32:1.
    let common = col.max_common_layout(&col).unwrap();
    assert_eq!((common.shape, common.stride), (Int::<32>, Int::<1>));
    let none = row.max_common_layout(&col).unwrap();
    assert_eq!((none.shape, none.stride), (Int::<1>, Int::<0>));
}

#[test]
fn tile_to_shape_repeats_blocks_and_keeps_swizzles() {
    // upstream: tile_to_shape((8,8):(8,1), (16,16)) uses a column-major order of blocks.
    let atom = LayoutRight::make(shape![Int::<8>, Int::<8>]).unwrap();
    let tile = atom.tile_to_shape(shape![Int::<16>, Int::<16>]).unwrap();
    assert_eq!(
        tile.shape,
        shape![(Int::<8>, Int::<2>), (Int::<8>, Int::<2>)]
    );
    assert_eq!(
        tile.stride,
        stride![(Int::<8>, Int::<64>), (Int::<1>, Int::<128>)]
    );

    // A rank-1 block is padded with 1:0 to the rank of the target.
    let column = LayoutLeft::make(shape![Int::<8>]).unwrap();
    let tile = column.tile_to_shape(shape![Int::<16>, Int::<2>]).unwrap();
    assert_eq!(
        tile.shape,
        shape![(Int::<8>, Int::<2>), (Int::<1>, Int::<2>)]
    );
    assert_eq!(
        tile.stride,
        stride![(Int::<1>, Int::<8>), (Int::<0>, Int::<16>)]
    );

    let swizzled = Swizzle::<3, 3, 3>
        .compose(atom)
        .tile_to_shape(shape![Int::<16>, Int::<16>])
        .unwrap();
    assert_eq!(swizzled.layout_a, Swizzle::<3, 3, 3>);
    assert_eq!(
        swizzled.get_nonswizzle_portion().stride,
        stride![(Int::<8>, Int::<64>), (Int::<1>, Int::<128>)]
    );
    assert_eq!(atom.get_nonswizzle_portion(), atom);
}

#[test]
fn make_swizzle_and_swizzle_composition_follow_upstream_comments() {
    use noodle_cute::swizzle::make_swizzle;
    // swizzle.hpp: make_swizzle<0b1000, 0b0100>() -> Swizzle<1,2,1>,
    // make_swizzle<0b11000000, 0b00000110>() -> Swizzle<2,1,5>.
    assert_eq!(make_swizzle::<0b1000, 0b0100>(), Swizzle::<1, 2, 1>);
    assert_eq!(
        make_swizzle::<0b1100_0000, 0b0000_0110>(),
        Swizzle::<2, 1, 5>
    );
    assert_eq!(Swizzle::<3, 3, 3>::SWIZZLE_CODE, 0b111_111_000);
    // Masks XOR: adjacent fields merge, and a swizzle composed with itself is the identity.
    assert_eq!(
        Swizzle::<1, 3, 3>.compose(Swizzle::<1, 4, 3>),
        Swizzle::<2, 3, 3>
    );
    assert_eq!(
        Swizzle::<2, 3, 3>.compose(Swizzle::<1, 4, 3>),
        Swizzle::<1, 3, 3>
    );
    assert_eq!(
        Swizzle::<3, 3, 3>.compose(Swizzle::<3, 3, 3>),
        Swizzle::<0, 0, 0>
    );
    // composition(Swizzle<0, M, S>, layout) is the layout itself.
    let layout = LayoutRight::make(shape![Int::<8>, Int::<64>]).unwrap();
    assert_eq!(Swizzle::<0, 4, 3>.compose(layout), layout);
}

#[test]
fn sm80_tutorial_shared_memory_layout() {
    // sgemm_sm80.cu: composition(Swizzle<3,3,3>{}, Layout<Shape<_8,Shape<_8,_8>>, Stride<_8,Stride<_1,_64>>>{})
    let atom = Swizzle::<3, 3, 3>.compose(
        Layout::new(
            shape![Int::<8>, (Int::<8>, Int::<8>)],
            stride![Int::<8>, (Int::<1>, Int::<64>)],
        )
        .unwrap(),
    );
    assert_eq!(
        (atom.size(), atom.cosize(), atom.rank(), atom.depth()),
        (Ok(512), Ok(512), 2, 2)
    );
    // Offset bits 3..6 hold the row m, bits 6..9 the vector k1: the physical row is m ^ k1.
    for m in 0..8isize {
        for k in 0..64isize {
            let (k0, k1) = (k % 8, k / 8);
            assert_eq!(
                atom.map_coord(shape![m, k]),
                DynInt(8 * (m ^ k1) + k0 + 64 * k1)
            );
        }
    }

    // tile_to_shape(swizzle_atom, (bM, bK, bP)) with bM = 128, bK = 64, bP = 3.
    let s_a = atom
        .tile_to_shape(shape![Int::<128>, Int::<64>, Int::<3>])
        .unwrap();
    assert_eq!(s_a.layout_a, Swizzle::<3, 3, 3>);
    assert_eq!(
        s_a.layout_b.shape,
        shape![
            (Int::<8>, Int::<16>),
            ((Int::<8>, Int::<8>), Int::<1>),
            (Int::<1>, Int::<3>)
        ]
    );
    assert_eq!(
        s_a.layout_b.stride,
        stride![
            (Int::<8>, Int::<512>),
            ((Int::<1>, Int::<64>), Int::<0>),
            (Int::<0>, Int::<8192>)
        ]
    );
    assert_eq!(s_a.size(), Ok(128 * 64 * 3));

    // sA(_, _, stage) keeps the swizzle and moves the offset, not the pointer.
    let (stage, offset) = s_a.slice_and_offset(shape![X, X, 2isize]);
    assert_eq!((offset, stage.offset), (Int::<0>, DynInt(2 * 8192)));
    // Row 9 is (m0, m1) = (1, 1) and column 10 is (k0, k1) = (2, 1): the swizzle
    // XORs k1 into m0, so the in-atom row becomes 0.
    assert_eq!(
        stage.map_coord(shape![9isize, 10isize]),
        DynInt(2 * 8192 + 512 + 2 + 64)
    );
}

#[test]
fn composed_layouts_forward_the_algebra_to_layout_b() {
    use noodle_cute::algebra::{CoalesceProfile, Zip2};
    let sw = Swizzle::<2, 0, 2>;
    let col = sw.compose(LayoutLeft::make(shape![Int::<4>, Int::<8>]).unwrap());
    let plain = col.layout_b;

    // Mode operations keep the swizzle and the offset.
    assert_eq!(col.get::<1>().layout_b, plain.get::<1>());
    assert_eq!(col.take::<0, 1>().layout_b, plain.take::<0, 1>());
    assert_eq!(col.group::<0, 2>().layout_b, plain.group::<0, 2>());
    assert_eq!(col.flatten().layout_b, plain.flatten());
    let wider = col.append(Layout::new(Int::<2>, Int::<32>).unwrap());
    assert_eq!((wider.layout_a, wider.rank()), (sw, 3));

    // Divides, products, composition, and coalesce act on layout_b.
    let tile = shape![Int::<2>, Int::<4>];
    assert_eq!(
        col.logical_divide(tile).unwrap().layout_b,
        plain.logical_divide(tile).unwrap()
    );
    assert_eq!(
        col.tiled_divide(tile).unwrap().layout_b,
        plain.tiled_divide(tile).unwrap()
    );
    assert_eq!(
        col.flat_divide(tile).unwrap().layout_b,
        plain.flat_divide(tile).unwrap()
    );
    let rep = Layout::new(Int::<2>, Int::<1>).unwrap();
    assert_eq!(
        col.logical_product(rep).unwrap().layout_b,
        plain.logical_product(rep).unwrap()
    );
    assert_eq!(
        col.zipped_product(rep).unwrap().layout_b,
        plain.zipped_product(rep).unwrap()
    );
    assert_eq!(
        col.tiled_product(rep).unwrap().layout_b,
        plain.tiled_product(rep).unwrap()
    );
    assert_eq!(
        col.flat_product(rep).unwrap().layout_b,
        plain.flat_product(rep).unwrap()
    );
    let rep2 = LayoutLeft::make(shape![Int::<2>, Int::<1>]).unwrap();
    assert_eq!(
        col.blocked_product(rep2).unwrap().layout_b,
        plain.blocked_product(rep2).unwrap()
    );
    assert_eq!(
        col.raked_product(rep2).unwrap().layout_b,
        plain.raked_product(rep2).unwrap()
    );
    let picked = col
        .composition(Layout::new(Int::<8>, Int::<4>).unwrap())
        .unwrap();
    // Stride 4 skips the 4 rows of mode 0: every column, 8:4.
    assert_eq!(
        (picked.layout_b.shape, picked.layout_b.stride),
        (Int::<8>, Int::<4>)
    );
    let merged = col.coalesce().unwrap();
    assert_eq!(
        (merged.layout_a, merged.layout_b),
        (sw, plain.coalesce().unwrap())
    );
    let per_mode = col.coalesce_profile(shape![Int::<1>, Int::<1>]).unwrap();
    assert_eq!(
        per_mode.layout_b,
        plain.coalesce_profile(shape![Int::<1>, Int::<1>]).unwrap()
    );
    let pairs = sw.compose(
        Layout::new(
            shape![(Int::<2>, Int::<2>), (Int::<2>, Int::<4>)],
            stride![(Int::<1>, Int::<4>), (Int::<2>, Int::<8>)],
        )
        .unwrap(),
    );
    assert_eq!(
        pairs.zip2().layout_b.stride,
        stride![(Int::<1>, Int::<2>), (Int::<4>, Int::<8>)]
    );
    // complement(composed, cotarget) is the plain complement of layout_b.
    let half = sw.compose(Layout::new(Int::<4>, Int::<1>).unwrap());
    let rest = half.complement(Int::<32>).unwrap();
    assert_eq!((rest.shape, rest.stride), (Int::<8>, Int::<4>));
}

#[test]
fn swizzled_layouts_upcast_downcast_and_recast_together() {
    // swizzle_layout.hpp: upcast<N> lowers M by log2(N); downcast<N> raises it.
    let halves =
        Swizzle::<3, 3, 3>.compose(LayoutRight::make(shape![Int::<8>, Int::<64>]).unwrap());
    let vectors = halves.upcast::<8>();
    assert_eq!(vectors.layout_a, Swizzle::<3, 0, 3>);
    assert_eq!(
        vectors.layout_b,
        LayoutRight::make(shape![Int::<8>, Int::<8>]).unwrap()
    );
    let back = vectors.downcast::<8>();
    assert_eq!(
        (back.layout_a, back.layout_b),
        (halves.layout_a, halves.layout_b)
    );
    // Below the unswizzled bits the mask shrinks: Swizzle<max(B + M - log2 N, 0), 0, S>.
    assert_eq!(
        Swizzle::<3, 1, 3>
            .compose(LayoutLeft::make(shape![Int::<64>]).unwrap())
            .upcast::<8>()
            .layout_a,
        Swizzle::<1, 0, 3>
    );
    // The offset scales with the elements.
    let moved = halves.slice_and_offset(shape![1isize, X]).0;
    assert_eq!(moved.upcast::<8>().offset, DynInt(8));
    assert_eq!(moved.recast_layout::<u16, u8>().offset, DynInt(128));
    assert_eq!(
        moved.recast_layout::<u16, u8>().layout_a,
        Swizzle::<3, 4, 3>
    );
}
