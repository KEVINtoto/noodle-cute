//! Contract tests for tensors, engines, iterators, recasting, and partitioning,
//! following CuTe's `tensor_impl.hpp`, `pointer.hpp`, `arithmetic_tuple.hpp`,
//! `layout.hpp`, and `stride.hpp`.
//!
//! Expected layouts are taken from upstream source comments where they exist;
//! partitions are checked element by element against the global coordinates.

use core::mem::size_of;

use noodle_cute::coord::{X, elem_less};
use noodle_cute::layout::{LayoutLeft, OrderedLayout};
use noodle_cute::pointer::{
    Advance, CoordIter, GmemPtr, Read, RecastPtr, Write, make_gmem_ptr, make_rmem_ptr,
    make_smem_ptr,
};
use noodle_cute::tensor::{
    ArrayEngine, ViewEngine, make_fragment_like, make_identity_tensor, make_owned_tensor,
    make_tensor, make_tensor_like,
};
use noodle_cute::{DynInt, Int, IntTuple, Layout, Nil, StaticValue, shape, stride};

fn iota<const N: usize>() -> [i32; N] {
    core::array::from_fn(|i| i as i32)
}

#[test]
fn pointers_advance_read_write_and_recast_in_every_memory_space() {
    let mut data = iota::<8>();
    let base = data.as_mut_ptr();
    let g = make_gmem_ptr(base);
    // SAFETY: every offset below stays inside `data`.
    unsafe {
        assert_eq!(g.advance(Int::<3>).read(), 3);
        make_smem_ptr(base).advance(DynInt(5)).write(50);
        assert_eq!(make_rmem_ptr(base).advance(5i32).read(), 50);
        let words: GmemPtr<*mut i64> = g.recast_ptr();
        words.write(0);
        assert_eq!((data[0], data[1]), (0, 0));
    }
    let coords = CoordIter(shape![Int::<0>, Int::<0>]).advance(shape![2i32, 3i32]);
    assert_eq!(coords.0, shape![2i32, 3i32]);
}

#[test]
fn view_tensors_read_and_write_hierarchical_coordinates() {
    let mut data = iota::<24>();
    let layout = Layout::new(
        shape![Int::<2>, (Int::<3>, Int::<4>)],
        stride![Int::<12>, (Int::<1>, Int::<3>)],
    )
    .unwrap();
    let t = make_tensor(data.as_mut_ptr(), layout);
    // SAFETY: the layout addresses 0..24 within `data`.
    unsafe {
        assert_eq!(t.read(shape![1i32, (2i32, 1i32)]), 17);
        assert_eq!(t.read(shape![1i32, 5i32]), 17);
        assert_eq!(t.read(11i32), 17);
        t.write(shape![0i32, (0i32, 2i32)], -6);
    }
    assert_eq!(data[6], -6);
    // A view is a pointer and the dynamic layout fields only.
    let v = make_tensor(data.as_ptr(), Layout::new(DynInt(24), Int::<1>).unwrap());
    assert_eq!(size_of_val(&v), 2 * size_of::<usize>());
    assert_eq!(size_of::<ViewEngine<*const i32>>(), size_of::<usize>());
}

#[test]
fn owned_tensors_are_sized_by_cosize_and_bounds_checked() {
    let layout = Layout::new(shape![Int::<2>, Int::<3>], stride![Int::<1>, Int::<4>]).unwrap();
    let mut owned = make_owned_tensor::<i32, _>(layout);
    // cosize((2,3):(1,4)) = 1 + 2 * 4 + 1 = 10, with holes at addresses 2, 3, 6, 7.
    let _: &ArrayEngine<i32, 10> = &owned.engine;
    assert_eq!(core::mem::align_of::<ArrayEngine<u8, 1>>(), 16);
    *owned.get_mut(shape![1i32, 2i32]) = 7;
    assert_eq!(owned.get(5i32), 7);
    assert_eq!(owned.engine.0[9], 7);
    let column = owned.view_mut().slice(shape![X, 1i32]);
    // SAFETY: the view points into `owned`, which outlives it.
    unsafe { column.write(shape![1i32], 3) };
    assert_eq!(owned.get(shape![1i32, 1i32]), 3);
}

#[test]
#[should_panic(expected = "outside the owned storage")]
fn owned_tensors_reject_coordinates_outside_the_storage() {
    let owned = make_owned_tensor::<i32, _>(Layout::new(Int::<4>, Int::<1>).unwrap());
    owned.get(4i32);
}

// layout.hpp: make_layout_like(Layout<Shape<_2,_2,_2,_2>, Stride<_0,_2,_4,_1>>)
//   -> (_2,_2,_2,_2):(_0,_2,_4,_1)
#[test]
fn make_tensor_like_follows_the_stride_order_and_keeps_zero_strides() {
    let reference = Layout::new(
        shape![Int::<2>, Int::<2>, Int::<2>, Int::<2>],
        stride![Int::<0>, Int::<2>, Int::<4>, Int::<1>],
    )
    .unwrap();
    let like = make_tensor_like::<f32, _, _>(&reference);
    assert_eq!(like.layout, reference);
    let _: &ArrayEngine<f32, 8> = &like.engine;
}

// stride.hpp: make_ordered_layout(Shape<_2,_2,_2,_2>, Step<_0,_2,_3,_1>) -> (_2,_2,_2,_2):(_1,_4,_8,_2)
// make_ordered_layout((2,3,4,5), (_2,67,42,_50)) -> (2,3,4,5):(_1,10,30,2): dynamic orders
// count as larger than the static ones, from left to right.
#[test]
fn ordered_layouts_rank_static_orders_before_dynamic_ones() {
    type S4 = IntTuple<Int<2>, IntTuple<Int<2>, IntTuple<Int<2>, IntTuple<Int<2>, Nil>>>>;
    type Step = IntTuple<Int<0>, IntTuple<Int<2>, IntTuple<Int<3>, IntTuple<Int<1>, Nil>>>>;
    let ordered = <OrderedLayout<S4, Step> as StaticValue>::VALUE;
    assert_eq!(
        ordered.stride,
        stride![Int::<1>, Int::<4>, Int::<8>, Int::<2>]
    );

    type S2345 = IntTuple<Int<2>, IntTuple<Int<3>, IntTuple<Int<4>, IntTuple<Int<5>, Nil>>>>;
    type Mixed = IntTuple<Int<2>, IntTuple<DynInt, IntTuple<DynInt, IntTuple<Int<50>, Nil>>>>;
    let mixed = <OrderedLayout<S2345, Mixed> as StaticValue>::VALUE;
    assert_eq!(
        mixed.stride,
        stride![Int::<1>, Int::<10>, Int::<30>, Int::<2>]
    );
}

// layout.hpp make_fragment_like: mode 0 compact left-major with zero strides kept,
// tiled_product with the remaining modes ordered by stride.
#[test]
fn make_fragment_like_keeps_mode_zero_left_major_and_broadcasts() {
    let partition = Layout::new(
        shape![(Int::<2>, Int::<2>), Int::<4>, Int::<2>],
        stride![(Int::<0>, Int::<1>), Int::<64>, Int::<8>],
    )
    .unwrap();
    let frag = make_fragment_like::<f32, _>(&partition);
    assert_eq!(
        frag.layout,
        Layout {
            shape: shape![(Int::<2>, Int::<2>), Int::<4>, Int::<2>],
            stride: stride![(Int::<0>, Int::<1>), Int::<4>, Int::<2>],
        }
    );
    let _: &ArrayEngine<f32, 16> = &frag.engine;
    // A single mode is compact left-major, whatever its strides.
    let single =
        make_fragment_like::<f32, _>(&Layout::new(shape![Int::<4>], stride![Int::<0>]).unwrap());
    assert_eq!(single.layout.stride, stride![Int::<1>]);
}

// layout.hpp upcast/downcast: stride-1 modes change their extent, other strides scale.
#[test]
fn upcast_downcast_and_recast_rescale_layouts_by_element_width() {
    let layout = Layout::new(shape![Int::<4>, Int::<3>], stride![Int::<1>, Int::<4>]).unwrap();
    assert_eq!(
        layout.upcast::<2>(),
        Layout {
            shape: shape![Int::<2>, Int::<3>],
            stride: stride![Int::<1>, Int::<2>]
        }
    );
    assert_eq!(
        layout.downcast::<2>(),
        Layout {
            shape: shape![Int::<8>, Int::<3>],
            stride: stride![Int::<1>, Int::<8>]
        }
    );
    // Static zero strides are kept, dynamic strides are divided, extents round up.
    let mixed = Layout::new(
        shape![DynInt(5), Int::<3>, Int::<2>],
        stride![Int::<1>, Int::<0>, DynInt(8)],
    )
    .unwrap();
    let up = mixed.upcast::<4>();
    assert_eq!(up.shape, shape![DynInt(2), Int::<3>, Int::<2>]);
    assert_eq!(up.stride, stride![Int::<1>, Int::<0>, DynInt(2)]);
    // u16 -> (u8, u32): a 3:2 ratio upcasts by 3, then downcasts by 2.
    let bytes = Layout::new(shape![Int::<6>, Int::<2>], stride![Int::<1>, Int::<6>]).unwrap();
    assert_eq!(
        bytes.recast_layout::<u16, [u8; 3]>(),
        Layout {
            shape: shape![Int::<4>, Int::<2>],
            stride: stride![Int::<1>, Int::<4>]
        }
    );
}

// tensor_impl.hpp recast: an upcast moves the pointer by sum(min((old - new) * stride, 0)).
#[test]
fn recast_tensors_move_the_pointer_for_negative_strides() {
    let data: [u16; 8] = core::array::from_fn(|i| i as u16);
    let base = data.as_ptr();
    let forward = make_tensor(base, Layout::new(Int::<8>, Int::<1>).unwrap()).recast::<u32>();
    assert_eq!(forward.data(), base.cast::<u32>());
    // SAFETY: the read stays inside `data`; `[u16; 8]` is only 2-byte aligned, so read unaligned.
    let first = unsafe { forward.data().read_unaligned() };
    assert_eq!(first, u32::from_ne_bytes(as_bytes(&data[0..2])));

    let reversed = make_tensor(
        base.wrapping_add(7),
        Layout::new(Int::<4>, Int::<-1>).unwrap(),
    );
    let words = reversed.recast::<u32>();
    assert_eq!(
        words.layout,
        Layout {
            shape: Int::<2>,
            stride: Int::<-1>
        }
    );
    // (4 - 2) * -1 = -2 elements of u16, as upstream.
    assert_eq!(words.data().cast::<u16>(), base.wrapping_add(5));
}

fn as_bytes(pair: &[u16]) -> [u8; 4] {
    let (a, b) = (pair[0].to_ne_bytes(), pair[1].to_ne_bytes());
    [a[0], a[1], b[0], b[1]]
}

#[test]
fn slicing_and_domain_offset_move_the_origin() {
    let data = iota::<12>();
    let t = make_tensor(
        data.as_ptr(),
        LayoutLeft::make(shape![Int::<3>, Int::<4>]).unwrap(),
    );
    let row = t.slice(shape![1i32, X]);
    assert_eq!(row.layout.shape, shape![Int::<4>]);
    // SAFETY: all reads below stay inside `data`.
    unsafe {
        assert_eq!(row.read(shape![2i32]), 7);
        let shifted = t.domain_offset(shape![1i32, 1i32]);
        assert_eq!(shifted.layout, t.layout);
        assert_eq!(shifted.read(shape![0i32, 0i32]), 4);
        assert_eq!(shifted.read(shape![1i32, 2i32]), 11);
    }
}

#[test]
fn zipped_inner_and_outer_partitions_split_tile_and_rest() {
    let data = iota::<96>();
    let t = make_tensor(
        data.as_ptr(),
        LayoutLeft::make(shape![DynInt(12), DynInt(8)]).unwrap(),
    );
    let tiler = shape![Int::<4>, Int::<2>];
    let zipped = t.zipped_divide(tiler).unwrap();
    assert_eq!(zipped.data(), t.data());
    assert_eq!(
        zipped.layout.shape,
        shape![(Int::<4>, Int::<2>), (DynInt(3), DynInt(4))]
    );
    // SAFETY: every read below stays inside `data`.
    unsafe {
        // inner: the tile at rest coordinate (2, 3) -> rows 8..12, columns 6..8.
        let tile = t.inner_partition(tiler, shape![2i32, 3i32]).unwrap();
        assert_eq!(tile.layout.shape, shape![Int::<4>, Int::<2>]);
        assert_eq!(tile.read(shape![1i32, 1i32]), 9 + 7 * 12);
        // An integer coordinate indexes the rest mode as one left-major mode: 11 = (2, 3).
        let flat = t.inner_partition(tiler, 11i32).unwrap();
        assert_eq!(flat.read(shape![1i32, 1i32]), 9 + 7 * 12);
        // A shorter tuple is padded with `_`, keeping the remaining rest modes.
        let column_of_tiles = t.inner_partition(tiler, shape![2i32]).unwrap();
        assert_eq!(
            column_of_tiles.layout.shape,
            shape![Int::<4>, Int::<2>, DynInt(4)]
        );
        assert_eq!(column_of_tiles.read(shape![1i32, 1i32, 3i32]), 9 + 7 * 12);
        // outer: element (1, 1) of every tile.
        let spread = t.outer_partition(tiler, shape![1i32, 1i32]).unwrap();
        assert_eq!(spread.layout.shape, shape![DynInt(3), DynInt(4)]);
        assert_eq!(spread.read(shape![2i32, 3i32]), 9 + 7 * 12);
    }
}

// tensor_impl.hpp local_tile: cta_tiler = (_32,_64,_4), cta_coord = (bx, by, _);
//   ctaA (M,K) Step<_1,X,_1> -> (_32,_4,k), ctaB (N,K) Step<X,_1,_1> -> (_64,_4,k),
//   ctaC (M,N) Step<_1,_1,X> -> (_32,_64).
#[test]
fn local_tile_projections_serve_a_b_and_c_from_one_tiler() {
    let (m, n, k) = (64, 128, 16);
    let a = [0f32; 64 * 16];
    let b = [0f32; 128 * 16];
    let c = [0f32; 64 * 128];
    let ma = make_tensor(
        a.as_ptr(),
        LayoutLeft::make(shape![DynInt(m), DynInt(k)]).unwrap(),
    );
    let mb = make_tensor(
        b.as_ptr(),
        LayoutLeft::make(shape![DynInt(n), DynInt(k)]).unwrap(),
    );
    let mc = make_tensor(
        c.as_ptr(),
        LayoutLeft::make(shape![DynInt(m), DynInt(n)]).unwrap(),
    );
    let cta_tiler = shape![Int::<32>, Int::<64>, Int::<4>];
    let cta_coord = shape![1i32, 1i32, X];

    let ga = ma
        .local_tile_proj(cta_tiler, cta_coord, shape![Int::<1>, X, Int::<1>])
        .unwrap();
    assert_eq!(ga.layout.shape, shape![Int::<32>, Int::<4>, DynInt(4)]);
    assert_eq!(ga.data(), a.as_ptr().wrapping_add(32));

    let gb = mb
        .local_tile_proj(cta_tiler, cta_coord, shape![X, Int::<1>, Int::<1>])
        .unwrap();
    assert_eq!(gb.layout.shape, shape![Int::<64>, Int::<4>, DynInt(4)]);
    assert_eq!(gb.data(), b.as_ptr().wrapping_add(64));

    let gc = mc
        .local_tile_proj(cta_tiler, cta_coord, shape![Int::<1>, Int::<1>, X])
        .unwrap();
    assert_eq!(gc.layout.shape, shape![Int::<32>, Int::<64>]);
    assert_eq!(gc.data(), c.as_ptr().wrapping_add(32 + 64 * 64));
}

// tensor_impl.hpp local_partition: (_16,_64) by Layout<Shape<_2,_16>> -> (_8,_4) per thread.
#[test]
fn local_partition_gives_each_thread_a_strided_share() {
    let data = iota::<{ 16 * 64 }>();
    let tile = make_tensor(
        data.as_ptr(),
        LayoutLeft::make(shape![Int::<16>, Int::<64>]).unwrap(),
    );
    let threads = LayoutLeft::make(shape![Int::<2>, Int::<16>]).unwrap();
    for t in 0..32i32 {
        let mine = tile.local_partition(&threads, t).unwrap();
        assert_eq!(mine.layout.shape, shape![Int::<8>, Int::<4>]);
        for i in 0..8i32 {
            for j in 0..4i32 {
                let (row, col) = (t % 2 + 2 * i, t / 2 + 16 * j);
                // SAFETY: the partition stays inside the 16x64 tile.
                assert_eq!(unsafe { mine.read(shape![i, j]) }, row + col * 16);
            }
        }
    }
}

// tensor_impl.hpp: thr_layout = Layout<Shape<_2,_16,_1>, Stride<_16,_1,_0>>;
//   thrA Step<_1,X,_1> -> (M/2,K/1), thrB Step<X,_1,_1> -> (N/16,K/1), thrC Step<_1,_1,X> -> (M/2,N/16).
#[test]
fn local_partition_projections_split_one_thread_layout() {
    let a = iota::<{ 8 * 4 }>();
    let b = iota::<{ 32 * 4 }>();
    let c = iota::<{ 8 * 32 }>();
    let ta = make_tensor(
        a.as_ptr(),
        LayoutLeft::make(shape![Int::<8>, Int::<4>]).unwrap(),
    );
    let tb = make_tensor(
        b.as_ptr(),
        LayoutLeft::make(shape![Int::<32>, Int::<4>]).unwrap(),
    );
    let tc = make_tensor(
        c.as_ptr(),
        LayoutLeft::make(shape![Int::<8>, Int::<32>]).unwrap(),
    );
    let thr = Layout::new(
        shape![Int::<2>, Int::<16>, Int::<1>],
        stride![Int::<16>, Int::<1>, Int::<0>],
    )
    .unwrap();
    // Thread 21 is (1, 5, 0) in the thread layout.
    let idx = 21i32;
    let pa = ta
        .local_partition_proj(&thr, idx, shape![Int::<1>, X, Int::<1>])
        .unwrap();
    let pb = tb
        .local_partition_proj(&thr, idx, shape![X, Int::<1>, Int::<1>])
        .unwrap();
    let pc = tc
        .local_partition_proj(&thr, idx, shape![Int::<1>, Int::<1>, X])
        .unwrap();
    assert_eq!(pa.layout.shape, shape![Int::<4>, Int::<4>]);
    assert_eq!(pb.layout.shape, shape![Int::<2>, Int::<4>]);
    assert_eq!(pc.layout.shape, shape![Int::<4>, Int::<2>]);
    // SAFETY: each partition stays inside its source array.
    unsafe {
        assert_eq!(pa.read(shape![2i32, 3i32]), (1 + 2 * 2) + 3 * 8);
        assert_eq!(pb.read(shape![1i32, 3i32]), (5 + 16) + 3 * 32);
        assert_eq!(pc.read(shape![2i32, 1i32]), (1 + 2 * 2) + (5 + 16) * 8);
    }
}

// arithmetic_tuple.hpp / tensor_impl.hpp: an identity tensor partitioned like the data tensor
// yields global coordinates; tail tiles extend past the shape without wrapping into other modes.
#[test]
fn identity_tensors_predicate_tail_tiles() {
    let extent = shape![DynInt(10), DynInt(6)];
    let data = iota::<60>();
    let mdata = make_tensor(data.as_ptr(), LayoutLeft::make(extent).unwrap());
    let identity = make_identity_tensor(extent);
    let tiler = shape![Int::<4>, Int::<4>];
    let threads = LayoutLeft::make(shape![Int::<2>, Int::<2>]).unwrap();
    let mut seen = [0u8; 60];
    for bx in 0..3i32 {
        for by in 0..2i32 {
            let gd = mdata.local_tile(tiler, shape![bx, by]).unwrap();
            let gi = identity.local_tile(tiler, shape![bx, by]).unwrap();
            for t in 0..4i32 {
                let td = gd.local_partition(&threads, t).unwrap();
                let ti = gi.local_partition(&threads, t).unwrap();
                for i in 0..2i32 {
                    for j in 0..2i32 {
                        let c = ti.get(shape![i, j]);
                        assert_eq!(c, shape![bx * 4 + t % 2 + 2 * i, by * 4 + t / 2 + 2 * j]);
                        if elem_less(&c, &extent) {
                            // SAFETY: the predicate keeps the read inside `data`.
                            let v = unsafe { td.read(shape![i, j]) };
                            assert_eq!(v, c.0 + c.1.0 * 10);
                            seen[v as usize] += 1;
                        }
                    }
                }
            }
        }
    }
    // Every in-bounds element is visited exactly once; out-of-bounds slots are predicated off.
    assert!(seen.iter().all(|&n| n == 1));
}

#[test]
fn identity_tensors_of_integral_shapes_return_integers() {
    let line = make_identity_tensor(DynInt(10));
    let tail = line.local_tile(Int::<4>, 2i32).unwrap();
    assert_eq!(tail.get(3i32), 11);
    let flat = make_identity_tensor(shape![Int::<3>, Int::<4>]);
    // An integer coordinate decodes left-major into one component per mode.
    assert_eq!(flat.get(7i32), shape![1i32, 2i32]);
}
