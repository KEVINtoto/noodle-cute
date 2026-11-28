#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

//! Contract tests for Stage 2: owning `ArrayEngine` tensors, `domain_offset`, CuTe
//! partitioning (`inner_partition`, `outer_partition`, `local_tile_proj`,
//! `local_partition`, `local_partition_proj`), and pure-safe element-wise algorithms
//! (`fill`, `clear`, `copy_if`, `axpby`, `for_each`, `transform`, `transform_into`,
//! `transform_zip`).

use core::mem::align_of;

use noodle_cute::algebra::{compose_tile, flat_divide, tiled_divide};
use noodle_cute::algorithm::{
    axpby, clear, copy_if, fill, for_each, transform, transform_into, transform_zip,
};
use noodle_cute::basis::as_tuple;
use noodle_cute::f2::F2;
use noodle_cute::integer::Int;
use noodle_cute::layout::Layout;
use noodle_cute::shape::elem_less;
use noodle_cute::tensor::{
    ArrayEngine, F2OffsetAccessor, OWNED_STORAGE_LEN, Tensor, domain_offset, identity_tensor,
    inner_partition, local_partition, local_partition_proj, local_tile, local_tile_proj,
    make_fragment_like, make_owned_tensor, make_owned_tensor_with, make_tensor, make_tensor_like,
    outer_partition,
};
use noodle_cute::tuple::Nil;
use noodle_cute::{f2, int_tuple, tuple};

fn iota<const N: usize>() -> [i32; N] {
    core::array::from_fn(|i| i as i32)
}

// ==================================================================
// 1. ArrayEngine & owning tensor constructors
// ==================================================================

#[test]
fn owned_tensors_are_sized_by_cosize_and_bounds_checked() {
    let layout = Layout::new(int_tuple![Int::<2>, Int::<3>], int_tuple![Int::<1>, Int::<4>]);
    let mut owned = make_owned_tensor::<i32, _>(layout);
    // cosize((2,3):(1,4)) = 1 + 2 * 4 + 1 = 10, with holes at addresses 2, 3, 6, 7.
    let _: &ArrayEngine<i32, 10> = &owned.accessor;
    assert_eq!(align_of::<ArrayEngine<u8, 1>>(), 16);
    assert_eq!(owned.accessor.len(), 10);
    assert!(!owned.accessor.is_empty());

    *owned.get_mut(int_tuple![1i32, 2i32]) = 7;
    assert_eq!(owned(5i32), 7);
    assert_eq!(*owned.get_ref(5i32), 7);
    assert_eq!(owned.as_slice()[9], 7);

    let mut column = owned.view_mut()(.., 1i32);
    column.set(int_tuple![1i32], 3);
    assert_eq!(owned(1i32, 1i32), 3);

    let ro_col = owned.view()(.., 2i32);
    assert_eq!(ro_col(1i32), 7);
}

#[test]
fn owned_storage_len_handles_empty_unit_zero_stride_and_f2_layouts() {
    let empty = make_owned_tensor::<i32, _>(Layout::new(Int::<0>, Int::<1>));
    let _: &ArrayEngine<i32, 0> = &empty.accessor;
    assert!(empty.accessor.is_empty());

    let unit_nil = make_owned_tensor::<i32, _>(Nil);
    let _: &ArrayEngine<i32, 1> = &unit_nil.accessor;
    assert_eq!(unit_nil.at(Nil), 0);

    // Extent 1 with negative stride still only reaches offset 0.
    let unit_neg = make_owned_tensor::<i32, _>(Layout::new(Int::<1>, Int::<-5>));
    let _: &ArrayEngine<i32, 1> = &unit_neg.accessor;
    assert_eq!(unit_neg(0i32), 0);

    // Broadcast mode (stride 0) does not inflate backing storage.
    let bcast = make_owned_tensor::<i32, _>(Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        int_tuple![Int::<0>, Int::<1>],
    ));
    let _: &ArrayEngine<i32, 4> = &bcast.accessor;

    // F2-swizzled 8x8 tile: linear offsets span 0..64.
    let f2_layout = Layout::new(
        int_tuple![Int::<8>, Int::<8>],
        tuple![f2!(Int::<1>), f2!(Int::<9>)],
    );
    let f2_owned = make_owned_tensor::<i32, _>(f2_layout);
    let _: &ArrayEngine<i32, 64> = &f2_owned.accessor;

    assert_eq!(
        OWNED_STORAGE_LEN::<
            noodle_cute::tuple::Til<Int<2>, noodle_cute::tuple::Til<Int<3>, Nil>>,
            noodle_cute::tuple::Til<Int<1>, noodle_cute::tuple::Til<Int<4>, Nil>>,
        >,
        10
    );
}

#[test]
fn owned_tensors_support_const_construction_and_fill() {
    type L = Layout<
        noodle_cute::tuple::Til<Int<2>, noodle_cute::tuple::Til<Int<3>, Nil>>,
        noodle_cute::tuple::Til<Int<1>, noodle_cute::tuple::Til<Int<2>, Nil>>,
    >;
    const BUILT: Tensor<ArrayEngine<i32, 6>, L> = {
        let mut t = make_owned_tensor_with(int_tuple![Int::<2>, Int::<3>], 5i32);
        fill(&mut t, 9i32);
        t.set(int_tuple![1i32, 2i32], 42);
        t
    };
    assert_eq!(BUILT(0i32, 0i32), 9);
    assert_eq!(BUILT(1i32, 2i32), 42);
    assert_eq!(BUILT.as_slice(), &[9, 9, 9, 9, 9, 42]);
}

#[test]
#[should_panic(expected = "accessor index out of bounds")]
fn owned_tensors_reject_coordinates_outside_the_storage() {
    let owned = make_owned_tensor::<i32, _>(Layout::new(Int::<4>, Int::<1>));
    let _ = owned(4i32);
}

#[test]
#[should_panic(expected = "accessor index is negative")]
fn owned_tensors_reject_negative_coordinates() {
    let owned = make_owned_tensor::<i32, _>(Layout::new(Int::<4>, Int::<1>));
    let _ = owned(-1i32);
}

#[test]
fn make_tensor_like_follows_stride_order_and_keeps_zero_strides() {
    let reference = Layout::new(
        int_tuple![Int::<2>, Int::<2>, Int::<2>, Int::<2>],
        int_tuple![Int::<0>, Int::<2>, Int::<4>, Int::<1>],
    );
    let like = make_tensor_like::<f32, _>(&reference);
    assert_eq!(like.layout, reference);
    let _: &ArrayEngine<f32, 8> = &like.accessor;

    // Also works directly from a Tensor reference.
    let buf = [0i32; 8];
    let t = make_tensor(&buf, reference);
    let like_from_tensor = make_tensor_like::<f32, _>(&t);
    assert_eq!(like_from_tensor.layout, reference);
    let _: &ArrayEngine<f32, 8> = &like_from_tensor.accessor;
}

#[test]
fn make_fragment_like_keeps_mode_zero_left_major_and_broadcasts() {
    let partition = Layout::new(
        int_tuple![(Int::<2>, Int::<2>), Int::<4>, Int::<2>],
        int_tuple![(Int::<0>, Int::<1>), Int::<64>, Int::<8>],
    );
    let expected = Layout::new(
        int_tuple![(Int::<2>, Int::<2>), Int::<4>, Int::<2>],
        int_tuple![(Int::<0>, Int::<1>), Int::<4>, Int::<2>],
    );
    let frag = make_fragment_like::<f32, _>(&partition);
    assert_eq!(frag.layout, expected);
    let _: &ArrayEngine<f32, 16> = &frag.accessor;

    // Also accepts a Tensor directly.
    let buf = [0i32; 256];
    let t = make_tensor(&buf, partition);
    let frag_t = make_fragment_like::<f32, _>(&t);
    assert_eq!(frag_t.layout, expected);
    let _: &ArrayEngine<f32, 16> = &frag_t.accessor;

    // Single mode is compact left-major regardless of its original stride.
    let single =
        make_fragment_like::<f32, _>(&Layout::new(int_tuple![Int::<4>], int_tuple![Int::<0>]));
    assert_eq!(single.layout.stride, int_tuple![Int::<1>]);
    let _: &ArrayEngine<f32, 4> = &single.accessor;
}

// ==================================================================
// 2. domain_offset & algebra pass-through on Tensor
// ==================================================================

#[test]
fn slicing_and_domain_offset_move_the_origin() {
    let data = iota::<12>();
    let t = make_tensor(&data, int_tuple![Int::<3>, Int::<4>]);

    let row = t(1i32, ..);
    assert_eq!(row.shape(), int_tuple![Int::<4>]);
    assert_eq!(row(2i32), 7);

    let shifted = domain_offset(t, int_tuple![1i32, 1i32]);
    assert_eq!(shifted.layout, t.layout);
    assert_eq!(shifted.accessor.offset_index(), 4);
    assert_eq!(shifted(0i32, 0i32), 4);
    assert_eq!(shifted(1i32, 2i32), 11);

    // Method form + mutable CellAccessor.
    let mut buf = iota::<12>();
    let mut mut_shifted = make_tensor(&mut buf, int_tuple![Int::<3>, Int::<4>])
        .domain_offset(int_tuple![1i32, 1i32]);
    mut_shifted.set(int_tuple![0i32, 0i32], 99);
    assert_eq!(buf[4], 99);

    // F2-strided tensor preserves XOR origin through F2OffsetAccessor.
    let f2_data = iota::<16>();
    let f2_t = make_tensor(
        &f2_data,
        Layout::new(
            int_tuple![Int::<4>, Int::<4>],
            tuple![f2!(Int::<1>), f2!(Int::<4>)],
        ),
    );
    let f2_shifted = f2_t.domain_offset(int_tuple![1i32, 1i32]);
    let _: F2OffsetAccessor<_, F2<i32>> = f2_shifted.accessor;
    assert_eq!(f2_shifted.accessor.origin, f2!(5i32));
    // (1^1, 1^0) = (0, 1) -> linear offset 4.
    assert_eq!(f2_shifted(1i32, 0i32), 4);

    // Identity tensor shifts coordinate origin without allocation.
    let id = identity_tensor(int_tuple![Int::<3>, Int::<4>]).domain_offset(int_tuple![1i32, 2i32]);
    assert_eq!(as_tuple(id(0i32, 0i32)), int_tuple![1i32, 2i32]);
    assert_eq!(as_tuple(id(1i32, 1i32)), int_tuple![2i32, 3i32]);

    // Const evaluation of domain_offset.
    const DATA: [i32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    const CONST_VAL: i32 = {
        let ct = make_tensor(&DATA, int_tuple![Int::<3>, Int::<4>]);
        let cs = ct.domain_offset(int_tuple![1i32, 1i32]);
        cs.at(int_tuple![1i32, 2i32])
    };
    assert_eq!(CONST_VAL, 11);
}

#[test]
fn tensor_compose_tile_tiled_divide_and_flat_divide() {
    let data = iota::<96>();
    let t = make_tensor(&data, int_tuple![Int::<12>, Int::<8>]);

    let ct = compose_tile(t, tuple![Layout::new(Int::<4>, Int::<2>), ..]);
    assert_eq!(ct.shape(), int_tuple![Int::<4>, Int::<8>]);
    assert_eq!(ct(3i32, 2i32), 6 + 2 * 12);

    let td = tiled_divide(t, int_tuple![Int::<4>, Int::<2>]);
    assert_eq!(td.shape(), int_tuple![(Int::<4>, Int::<2>), Int::<3>, Int::<4>]);
    assert_eq!(td.at(tuple![(1i32, 1i32), 2i32, 3i32]), 9 + 7 * 12);

    let fd = flat_divide(t, int_tuple![Int::<4>, Int::<2>]);
    assert_eq!(fd.shape(), int_tuple![Int::<4>, Int::<2>, Int::<3>, Int::<4>]);
    assert_eq!(fd(1i32, 1i32, 2i32, 3i32), 9 + 7 * 12);

    // Inherent method forwarding matches free functions.
    assert_eq!(t.tiled_divide(int_tuple![Int::<4>, Int::<2>]).layout, td.layout);
    assert_eq!(t.flat_divide(int_tuple![Int::<4>, Int::<2>]).layout, fd.layout);
}

// ==================================================================
// 3. CuTe partitioning: inner/outer/local_tile_proj/local_partition
// ==================================================================

#[test]
fn zipped_inner_and_outer_partitions_split_tile_and_rest() {
    let data = iota::<96>();
    let t = make_tensor(&data, int_tuple![12i32, 8i32]);
    let tiler = int_tuple![Int::<4>, Int::<2>];

    let zipped = t.zipped_divide(tiler);
    assert_eq!(zipped.accessor.offset_index(), 0);
    assert_eq!(
        zipped.shape(),
        int_tuple![(Int::<4>, Int::<2>), (3i32, 4i32)]
    );

    // inner: tile at rest coordinate (2, 3) -> rows 8..12, columns 6..8.
    let tile = inner_partition(t, tiler, int_tuple![2i32, 3i32]);
    assert_eq!(tile.shape(), int_tuple![Int::<4>, Int::<2>]);
    assert_eq!(tile(1i32, 1i32), 9 + 7 * 12);

    // Flat integer coordinate indexes rest mode colexicographically: 11 = (2, 3).
    let flat = t.inner_partition(tiler, 11i32);
    assert_eq!(flat.shape(), int_tuple![Int::<4>, Int::<2>]);
    assert_eq!(flat(1i32, 1i32), 9 + 7 * 12);

    // Shorter tuple coordinate is padded with `..`, keeping trailing rest modes.
    let col_of_tiles = t.inner_partition(tiler, int_tuple![2i32]);
    assert_eq!(col_of_tiles.shape(), int_tuple![Int::<4>, Int::<2>, 4i32]);
    assert_eq!(col_of_tiles(1i32, 1i32, 3i32), 9 + 7 * 12);

    // outer: element (1, 1) of every tile.
    let spread = outer_partition(t, tiler, int_tuple![1i32, 1i32]);
    assert_eq!(spread.shape(), int_tuple![3i32, 4i32]);
    assert_eq!(spread(2i32, 3i32), 9 + 7 * 12);

    // Existing PyCuTe local_tile keeps tile grouped in mode 0.
    let pycute = local_tile(t, tiler, int_tuple![2i32, 3i32]);
    assert_eq!(pycute.shape(), int_tuple![(Int::<4>, Int::<2>)]);
    assert_eq!(pycute.at(tuple![(1i32, 1i32)]), 9 + 7 * 12);
    assert_eq!(t.local_tile(tiler, int_tuple![2i32, 3i32]).layout, pycute.layout);

    // Const evaluation of inner_partition and outer_partition.
    const BUF: [i32; 24] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    ];
    const CONST_INNER: i32 = {
        let ct = make_tensor(&BUF, int_tuple![Int::<6>, Int::<4>]);
        let tile = ct.inner_partition(int_tuple![Int::<2>, Int::<2>], int_tuple![2i32, 1i32]);
        tile.at(int_tuple![1i32, 1i32])
    };
    const CONST_OUTER: i32 = {
        let ct = make_tensor(&BUF, int_tuple![Int::<6>, Int::<4>]);
        let spread = ct.outer_partition(int_tuple![Int::<2>, Int::<2>], int_tuple![1i32, 1i32]);
        spread.at(int_tuple![2i32, 1i32])
    };
    assert_eq!(CONST_INNER, 5 + 3 * 6);
    assert_eq!(CONST_OUTER, 5 + 3 * 6);
}

#[test]
fn local_tile_projections_serve_a_b_and_c_from_one_tiler() {
    let (m, n, k) = (64i32, 128i32, 16i32);
    let a = iota::<{ 64 * 16 }>();
    let b = iota::<{ 128 * 16 }>();
    let c = iota::<{ 64 * 128 }>();
    let ma = make_tensor(&a, int_tuple![m, k]);
    let mb = make_tensor(&b, int_tuple![n, k]);
    let mc = make_tensor(&c, int_tuple![m, n]);

    let cta_tiler = int_tuple![Int::<32>, Int::<64>, Int::<4>];
    let cta_coord = tuple![1i32, 1i32, ..];

    let ga = ma.local_tile_proj(cta_tiler, cta_coord, tuple![Int::<1>, .., Int::<1>]);
    assert_eq!(ga.shape(), int_tuple![Int::<32>, Int::<4>, 4i32]);
    assert_eq!(ga.accessor.offset_index(), 32);
    assert_eq!(ga(3i32, 2i32, 1i32), (32 + 3) + (4 + 2) * 64);

    let gb = local_tile_proj(mb, cta_tiler, cta_coord, tuple![.., Int::<1>, Int::<1>]);
    assert_eq!(gb.shape(), int_tuple![Int::<64>, Int::<4>, 4i32]);
    assert_eq!(gb.accessor.offset_index(), 64);
    assert_eq!(gb(5i32, 1i32, 2i32), (64 + 5) + (8 + 1) * 128);

    let gc = mc.local_tile_proj(cta_tiler, cta_coord, tuple![Int::<1>, Int::<1>, ..]);
    assert_eq!(gc.shape(), int_tuple![Int::<32>, Int::<64>]);
    assert_eq!(gc.accessor.offset_index(), 32 + 64 * 64);
    assert_eq!(gc(7i32, 9i32), (32 + 7) + (64 + 9) * 64);
}

#[test]
fn local_partition_gives_each_thread_a_strided_share() {
    let data = iota::<{ 16 * 64 }>();
    let tile = make_tensor(&data, int_tuple![Int::<16>, Int::<64>]);
    let threads = Layout::from_shape(int_tuple![Int::<2>, Int::<16>]);

    for t in 0..32i32 {
        let mine = local_partition(tile, &threads, t);
        assert_eq!(mine.shape(), int_tuple![Int::<8>, Int::<4>]);
        for i in 0..8i32 {
            for j in 0..4i32 {
                let (row, col) = (t % 2 + 2 * i, t / 2 + 16 * j);
                assert_eq!(mine(i, j), row + col * 16);
            }
        }
    }
}

#[test]
fn local_partition_projections_split_one_thread_layout() {
    let a = iota::<{ 8 * 4 }>();
    let b = iota::<{ 32 * 4 }>();
    let c = iota::<{ 8 * 32 }>();
    let ta = make_tensor(&a, int_tuple![Int::<8>, Int::<4>]);
    let tb = make_tensor(&b, int_tuple![Int::<32>, Int::<4>]);
    let tc = make_tensor(&c, int_tuple![Int::<8>, Int::<32>]);

    let thr = Layout::new(
        int_tuple![Int::<2>, Int::<16>, Int::<1>],
        int_tuple![Int::<16>, Int::<1>, Int::<0>],
    );
    // Thread 21 is (1, 5, 0) in `thr`.
    let idx = 21i32;
    let pa = ta.local_partition_proj(&thr, idx, tuple![Int::<1>, .., Int::<1>]);
    let pb = local_partition_proj(tb, &thr, idx, tuple![.., Int::<1>, Int::<1>]);
    let pc = tc.local_partition_proj(&thr, idx, tuple![Int::<1>, Int::<1>, ..]);

    assert_eq!(pa.shape(), int_tuple![Int::<4>, Int::<4>]);
    assert_eq!(pb.shape(), int_tuple![Int::<2>, Int::<4>]);
    assert_eq!(pc.shape(), int_tuple![Int::<4>, Int::<2>]);

    assert_eq!(pa(2i32, 3i32), (1 + 2 * 2) + 3 * 8);
    assert_eq!(pb(1i32, 3i32), (5 + 16) + 3 * 32);
    assert_eq!(pc(2i32, 1i32), (1 + 2 * 2) + (5 + 16) * 8);
}

#[test]
fn identity_tensors_predicate_tail_tiles() {
    let extent = int_tuple![10i32, 6i32];
    let data = iota::<60>();
    let mdata = make_tensor(&data, extent);
    let identity = identity_tensor(extent);
    let tiler = int_tuple![Int::<4>, Int::<4>];
    let threads = Layout::from_shape(int_tuple![Int::<2>, Int::<2>]);
    let mut seen = [0u8; 60];

    for bx in 0..3i32 {
        for by in 0..2i32 {
            let gd = mdata.inner_partition(tiler, int_tuple![bx, by]);
            let gi = identity.inner_partition(tiler, int_tuple![bx, by]);
            for t in 0..4i32 {
                let td = gd.local_partition(&threads, t);
                let ti = gi.local_partition(&threads, t);
                for i in 0..2i32 {
                    for j in 0..2i32 {
                        let c = as_tuple(ti(i, j));
                        assert_eq!(c, int_tuple![bx * 4 + t % 2 + 2 * i, by * 4 + t / 2 + 2 * j]);
                        if elem_less(&c, &extent) {
                            let v = td(i, j);
                            assert_eq!(v, c.0 + c.1.0 * 10);
                            seen[v as usize] += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(seen.iter().all(|&n| n == 1));
}

#[test]
fn identity_tensors_of_integral_and_flat_shapes() {
    let line = identity_tensor(10i32);
    let tail = line.inner_partition(Int::<4>, 2i32);
    assert_eq!(as_tuple(tail(3i32)), 11);

    let flat = identity_tensor(int_tuple![Int::<3>, Int::<4>]);
    // Colexicographic flat index 7 decodes to (1, 2).
    assert_eq!(as_tuple(flat(7i32)), int_tuple![1i32, 2i32]);
}

// ==================================================================
// 4. Pure-safe element-wise algorithms
// ==================================================================

#[test]
fn copy_if_predicates_tail_tile_and_converts_types() {
    // A 4x4 tile at block (1, 1) of a 6x6 matrix: only rows/cols 4, 5 exist.
    let extent = int_tuple![6i32, 6i32];
    let tile = int_tuple![Int::<4>, Int::<4>];
    let data = iota::<36>();
    let src = make_tensor(&data, extent).inner_partition(tile, int_tuple![1i32, 1i32]);
    let coords = identity_tensor(extent).inner_partition(tile, int_tuple![1i32, 1i32]);
    let mut out = [-1i64; 16];
    let mut dst = make_tensor(&mut out, tile);

    copy_if(
        |i| elem_less(&as_tuple(coords.at(i)), &extent),
        &src,
        &mut dst,
    );
    assert_eq!(out[..2], [28, 29]);
    assert_eq!(out[4..6], [34, 35]);
    assert!(out[2..4].iter().chain(&out[6..]).all(|&x| x == -1));
}

#[test]
fn elementwise_algorithms_on_owned_and_aliased_views() {
    let layout = Layout::from_shape_right(int_tuple![Int::<2>, Int::<3>]);
    let mut x = make_owned_tensor::<f32, _>(layout);
    let mut y = make_owned_tensor::<f32, _>(layout);

    fill(&mut x, 2.0);
    transform(&mut x, |v| v + 1.0);
    fill(&mut y, f32::NAN);

    // beta = 0 does not read y (NaN-safe).
    axpby(2.0f32, &x, 0.0f32, &mut y);
    assert_eq!(y.as_slice(), &[6.0; 6]);

    let mut sum = 0.0;
    axpby(1.0f32, &x, 0.5f32, &mut y);
    for_each(&y, |v| sum += v);
    assert_eq!(sum, 36.0);

    transform_into(&x, &mut y, |v| v * 10.0);

    // Aliased input/output view via Copy CellAccessor (zero unsafe).
    let yv = y.view_mut();
    let mut yv_dst = yv;
    transform_zip(&x.view(), &yv, &mut yv_dst, |a, b| a + b);
    assert_eq!(y.as_slice(), &[33.0; 6]);

    clear(&mut x);
    assert_eq!(x.as_slice(), &[0.0; 6]);
}

#[test]
fn elementwise_algorithms_on_f2_swizzled_tensors() {
    let f2_layout = Layout::new(
        int_tuple![Int::<8>, Int::<8>],
        tuple![f2!(Int::<1>), f2!(Int::<9>)],
    );
    let mut a = make_owned_tensor::<i32, _>(f2_layout);
    let mut b = make_owned_tensor::<i32, _>(f2_layout);

    fill(&mut a, 3);
    transform(&mut a, |v| v * 2);
    copy_if(|i| i % 2 == 0, &a, &mut b);

    let mut sum = 0;
    for_each(&b, |v| sum += v);
    assert_eq!(sum, 32 * 6);
}

#[test]
#[should_panic(expected = "copy_if: size mismatch")]
fn copy_if_rejects_mismatched_dynamic_sizes() {
    let src_buf = [1i32; 4];
    let mut dst_buf = [0i32; 6];
    let src = make_tensor(&src_buf, 4i32);
    let mut dst = make_tensor(&mut dst_buf, 6i32);
    copy_if(|_| true, &src, &mut dst);
}

#[test]
#[should_panic(expected = "axpby: size mismatch")]
fn axpby_rejects_mismatched_dynamic_sizes() {
    let x_buf = [1.0f32; 4];
    let mut y_buf = [0.0f32; 5];
    let x = make_tensor(&x_buf, 4i32);
    let mut y = make_tensor(&mut y_buf, 5i32);
    axpby(1.0f32, &x, 1.0f32, &mut y);
}

#[test]
#[should_panic(expected = "transform_into: size mismatch")]
fn transform_into_rejects_mismatched_dynamic_sizes() {
    let src_buf = [1i32; 4];
    let mut dst_buf = [0i32; 5];
    let src = make_tensor(&src_buf, 4i32);
    let mut dst = make_tensor(&mut dst_buf, 5i32);
    transform_into(&src, &mut dst, |v| v + 1);
}

#[test]
#[should_panic(expected = "transform_zip: size mismatch")]
fn transform_zip_rejects_mismatched_dynamic_sizes() {
    let a_buf = [1i32; 4];
    let b_buf = [2i32; 4];
    let mut dst_buf = [0i32; 5];
    let a = make_tensor(&a_buf, 4i32);
    let b = make_tensor(&b_buf, 4i32);
    let mut dst = make_tensor(&mut dst_buf, 5i32);
    transform_zip(&a, &b, &mut dst, |x, y| x + y);
}
