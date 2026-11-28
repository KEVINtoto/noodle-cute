#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use std::collections::BTreeMap;
use std::panic::catch_unwind;

use noodle_cute::algorithm::im2col::{
    Im2ColOptions, im2col, im2col_coord, im2col_coord_with, im2col_layout, im2col_layout_with,
    im2col_padded, im2col_padded_with, im2col_with,
};
use noodle_cute::algorithm::reference::gemm;
use noodle_cute::integer::{Int, int};
use noodle_cute::layout::{Layout, make_ordered_layout};
use noodle_cute::shape::{Shape, idx2crd, in_bounds};
use noodle_cute::stride::Stride;
use noodle_cute::tensor::{
    Accessor, OutOfBoundsAccessor, Tensor, TensorIndex, TensorSet, make_tensor,
};
use noodle_cute::{einfold, int_tuple};

fn zpq_oracle(
    shape_dhw: &[int],
    stencil: &[int],
    lower_dhw: &[int],
    upper_dhw: &[int],
    stride_dhw: &[int],
    stride_trs: &[int],
) -> Vec<int> {
    (0..shape_dhw.len())
        .map(|i| {
            let num =
                shape_dhw[i] + lower_dhw[i] + upper_dhw[i] - ((stencil[i] - 1) * stride_trs[i] + 1);
            1 + num.div_euclid(stride_dhw[i])
        })
        .collect()
}

fn fill_fn<A, S, D>(t: &mut Tensor<A, Layout<S, D>>, f: impl Fn(int) -> i64)
where
    S: Shape,
    D: Stride<S>,
    Tensor<A, Layout<S, D>>: TensorSet<int, Item = i64>,
{
    for i in 0..t.size() {
        t.set(i, f(i));
    }
}

fn flat_i64<A, S, D>(t: &Tensor<A, Layout<S, D>>) -> Vec<i64>
where
    S: Shape,
    D: Stride<S>,
    Tensor<A, Layout<S, D>>: TensorIndex<int, Output = i64>,
{
    (0..t.size()).map(|i| t.at(i)).collect()
}

struct ConvSpec<'a> {
    n: int,
    c: int,
    k: int,
    shape_dhw: &'a [int],
    stencil: &'a [int],
    lower_dhw: &'a [int],
    upper_dhw: &'a [int],
    stride_dhw: &'a [int],
    stride_trs: &'a [int],
}

/// Direct N-D cross-correlation oracle matching `examples/im2col_test.py::conv_reference`.
fn conv_oracle_nd(
    spec: &ConvSpec<'_>,
    act_at: impl Fn(int, &[int], int) -> i64,
    flt_at: impl Fn(int, &[int], int) -> i64,
) -> BTreeMap<(int, Vec<int>, int), i64> {
    let zpq = zpq_oracle(
        spec.shape_dhw,
        spec.stencil,
        spec.lower_dhw,
        spec.upper_dhw,
        spec.stride_dhw,
        spec.stride_trs,
    );
    let rank = spec.shape_dhw.len();

    fn enumerate(extents: &[int]) -> Vec<Vec<int>> {
        let mut acc = vec![Vec::new()];
        for &e in extents {
            let mut next = Vec::with_capacity(acc.len() * (e as usize));
            for prefix in &acc {
                for v in 0..e {
                    let mut p = prefix.clone();
                    p.push(v);
                    next.push(p);
                }
            }
            acc = next;
        }
        acc
    }

    let positions = enumerate(&zpq);
    let taps = enumerate(spec.stencil);
    let mut out = BTreeMap::new();

    for n in 0..spec.n {
        for k in 0..spec.k {
            for pos in &positions {
                let mut total = 0i64;
                for tap in &taps {
                    let mut crd = Vec::with_capacity(rank);
                    let mut ok = true;
                    for i in 0..rank {
                        let coord_i = spec.stride_dhw[i] * pos[i] + spec.stride_trs[i] * tap[i]
                            - spec.lower_dhw[i];
                        if coord_i < 0 || coord_i >= spec.shape_dhw[i] {
                            ok = false;
                            break;
                        }
                        crd.push(coord_i);
                    }
                    if ok {
                        for c in 0..spec.c {
                            total += act_at(n, &crd, c) * flt_at(k, tap, c);
                        }
                    }
                }
                out.insert((n, pos.clone(), k), total);
            }
        }
    }
    out
}

// ------------------------------------------------------------------
// TestConvShape
// ------------------------------------------------------------------

#[test]
fn test_conv_shape_matches_cutlass_formula_exhaustively() {
    for d in 1..10 {
        for t in 1..5 {
            for pad in 0..3 {
                for u in 1..4 {
                    for s in 1..4 {
                        if (t - 1) * s + 1 > d + 2 * pad {
                            // Verify non-fitting dynamic stencil panics.
                            let res = catch_unwind(|| {
                                let l = make_ordered_layout(
                                    &int_tuple![1i32, d, 1i32],
                                    &int_tuple![Int::<2>, Int::<1>, Int::<0>],
                                );
                                let opts = Im2ColOptions::new()
                                    .lower_dhw(pad)
                                    .stride_dhw(u)
                                    .stride_trs(s);
                                im2col_layout_with(l, t, opts)
                            });
                            assert!(
                                res.is_err(),
                                "expected non-fitting panic for d={d}, t={t}, pad={pad}, u={u}, s={s}"
                            );
                            continue;
                        }
                        let l = make_ordered_layout(
                            &int_tuple![1i32, d, 1i32],
                            &int_tuple![Int::<2>, Int::<1>, Int::<0>],
                        );
                        let opts = Im2ColOptions::new()
                            .lower_dhw(pad)
                            .stride_dhw(u)
                            .stride_trs(s);
                        let (out_l, _) = im2col_layout_with(l, t, opts);
                        let got_z = out_l.shape.0.1.0.0;
                        let want_z = 1 + (d + 2 * pad - ((t - 1) * s + 1)).div_euclid(u);
                        assert_eq!(got_z, want_z, "d={d}, t={t}, pad={pad}, u={u}, s={s}");
                    }
                }
            }
        }
    }
}

#[test]
fn test_conv_shape_defaults_asymmetric_and_explicit_zpq() {
    let a2 = make_ordered_layout(
        &int_tuple![Int::<1>, Int::<5>, Int::<5>, Int::<1>],
        &int_tuple![Int::<3>, Int::<2>, Int::<1>, Int::<0>],
    );
    assert_eq!(
        im2col_layout(a2, int_tuple![Int::<3>, Int::<3>])
            .0
            .shape
            .0
            .1
            .0,
        int_tuple![Int::<3>, Int::<3>]
    );
    assert_eq!(
        im2col_layout_with(
            a2,
            int_tuple![Int::<3>, Int::<3>],
            Im2ColOptions::new().lower_dhw(int_tuple![Int::<1>, Int::<1>]),
        )
        .0
        .shape
        .0
        .1
        .0,
        int_tuple![Int::<5>, Int::<5>]
    );

    let a3 = make_ordered_layout(
        &int_tuple![Int::<1>, Int::<5>, Int::<6>, Int::<7>, Int::<1>],
        &int_tuple![Int::<4>, Int::<3>, Int::<2>, Int::<1>, Int::<0>],
    );
    assert_eq!(
        im2col_layout_with(
            a3,
            int_tuple![Int::<3>, Int::<3>, Int::<3>],
            Im2ColOptions::new().lower_dhw(int_tuple![Int::<1>, Int::<1>, Int::<1>]),
        )
        .0
        .shape
        .0
        .1
        .0,
        int_tuple![Int::<5>, Int::<6>, Int::<7>]
    );

    // Asymmetric padding on 1D: lower_dhw = 1, upper_dhw = 0
    let a1 = make_ordered_layout(
        &int_tuple![Int::<1>, Int::<5>, Int::<1>],
        &int_tuple![Int::<2>, Int::<1>, Int::<0>],
    );
    assert_eq!(
        im2col_layout_with(
            a1,
            int_tuple![Int::<2>],
            Im2ColOptions::new().lower_dhw(Int::<1>).upper_dhw(Int::<0>),
        )
        .0
        .shape
        .0
        .1
        .0,
        int_tuple![Int::<5>]
    );

    // Explicit shape_zpq overrides fprop formula
    let a6 = make_ordered_layout(
        &int_tuple![Int::<1>, Int::<6>, Int::<1>],
        &int_tuple![Int::<2>, Int::<1>, Int::<0>],
    );
    assert_eq!(
        im2col_layout_with(
            a6,
            int_tuple![Int::<3>],
            Im2ColOptions::new().shape_zpq(int_tuple![Int::<4>])
        )
        .0
        .shape
        .0
        .1
        .0,
        int_tuple![Int::<4>]
    );

    // Dynamic non-positive explicit shape_zpq panics
    assert!(
        catch_unwind(|| {
            let _ = im2col_layout_with(
                a6,
                int_tuple![Int::<3>],
                Im2ColOptions::new().shape_zpq(0i32),
            );
        })
        .is_err()
    );

    // Scalar integer is admissible for 1D spatial and matches 1-tuple
    let buf = [0i64; 8];
    let t1d = make_tensor(
        &buf,
        make_ordered_layout(
            &int_tuple![Int::<1>, Int::<8>, Int::<1>],
            &int_tuple![Int::<2>, Int::<1>, Int::<0>],
        ),
    );
    assert_eq!(
        im2col_with(t1d, Int::<3>, Im2ColOptions::new().lower_dhw(Int::<1>))
            .layout
            .shape,
        im2col_with(
            t1d,
            int_tuple![Int::<3>],
            Im2ColOptions::new().lower_dhw(int_tuple![Int::<1>])
        )
        .layout
        .shape
    );
}

// ------------------------------------------------------------------
// TestLayout
// ------------------------------------------------------------------

#[test]
fn test_im2col_layout_properties_and_cutlass_example_59() {
    let buf = [0i64; 16];
    let act = make_tensor(
        &buf,
        Layout::new(
            int_tuple![Int::<1>, Int::<4>, Int::<4>, Int::<1>],
            int_tuple![Int::<16>, Int::<4>, Int::<1>, Int::<1>],
        ),
    );

    // Basic shape, stride, shared base, non-injectivity
    let b = im2col(act, int_tuple![Int::<2>, Int::<2>]);
    assert_eq!(
        b.layout.shape,
        int_tuple![
            (Int::<1>, (Int::<3>, Int::<3>)),
            ((Int::<2>, Int::<2>), Int::<1>)
        ]
    );
    assert_eq!(
        b.layout.stride,
        int_tuple![
            (Int::<16>, (Int::<4>, Int::<1>)),
            ((Int::<4>, Int::<1>), Int::<1>)
        ]
    );
    assert!(b.accessor.same_base(&act.accessor));
    assert_eq!(b.size(), 36);
    assert_eq!(b.layout.cosize(), 16);

    // Traversal stride scales position modes only
    let b_str = im2col_with(
        act,
        int_tuple![Int::<2>, Int::<2>],
        Im2ColOptions::new().stride_dhw(int_tuple![Int::<2>, Int::<2>]),
    );
    assert_eq!(
        b_str.layout.stride,
        int_tuple![
            (Int::<16>, (Int::<8>, Int::<2>)),
            ((Int::<4>, Int::<1>), Int::<1>)
        ]
    );

    // Dilation scales tap modes only
    let b_dil = im2col_with(
        act,
        int_tuple![Int::<2>, Int::<2>],
        Im2ColOptions::new().stride_trs(int_tuple![Int::<2>, Int::<2>]),
    );
    assert_eq!(
        b_dil.layout.stride,
        int_tuple![
            (Int::<16>, (Int::<4>, Int::<1>)),
            ((Int::<8>, Int::<2>), Int::<1>)
        ]
    );

    // Unit 1x1 stencil degenerates to activation size/cosize
    let b_1x1 = im2col(act, int_tuple![Int::<1>, Int::<1>]);
    assert_eq!(
        b_1x1.layout.shape,
        int_tuple![
            (Int::<1>, (Int::<4>, Int::<4>)),
            ((Int::<1>, Int::<1>), Int::<1>)
        ]
    );
    assert_eq!(b_1x1.size(), act.size());
    assert_eq!(b_1x1.layout.cosize(), act.layout.cosize());

    // Rank-1 and Rank-3 spatial
    let buf1 = [0i64; 48];
    let act1 = make_tensor(
        &buf1,
        make_ordered_layout(
            &int_tuple![Int::<2>, Int::<8>, Int::<3>],
            &int_tuple![Int::<2>, Int::<1>, Int::<0>],
        ),
    );
    assert_eq!(
        im2col(act1, Int::<3>).layout.shape,
        int_tuple![(Int::<2>, (Int::<6>,)), ((Int::<3>,), Int::<3>)]
    );

    let buf3 = [0i64; 720];
    let act3 = make_tensor(
        &buf3,
        make_ordered_layout(
            &int_tuple![Int::<2>, Int::<4>, Int::<5>, Int::<6>, Int::<3>],
            &int_tuple![Int::<4>, Int::<3>, Int::<2>, Int::<1>, Int::<0>],
        ),
    );
    assert_eq!(
        im2col(act3, int_tuple![Int::<2>, Int::<3>, Int::<4>])
            .layout
            .shape,
        int_tuple![
            (Int::<2>, (Int::<3>, Int::<3>, Int::<3>)),
            ((Int::<2>, Int::<3>, Int::<4>), Int::<3>)
        ]
    );

    // Hierarchical channel mode passes through unchanged
    let buf_hc = vec![0i64; 512];
    let act_hc = make_tensor(
        &buf_hc[..],
        Layout::new(
            int_tuple![Int::<2>, Int::<4>, Int::<4>, (Int::<2>, Int::<8>)],
            int_tuple![Int::<256>, Int::<64>, Int::<16>, (Int::<8>, Int::<1>)],
        ),
    );
    assert_eq!(
        im2col(act_hc, int_tuple![Int::<2>, Int::<2>]).layout.shape,
        int_tuple![
            (Int::<2>, (Int::<3>, Int::<3>)),
            ((Int::<2>, Int::<2>), (Int::<2>, Int::<8>))
        ]
    );

    // CUTLASS example 59 (NDHWC)
    let got59 = im2col(act3, int_tuple![Int::<2>, Int::<3>, Int::<2>]);
    assert_eq!(
        got59.layout.shape,
        int_tuple![
            (Int::<2>, (Int::<3>, Int::<3>, Int::<5>)),
            ((Int::<2>, Int::<3>, Int::<2>), Int::<3>)
        ]
    );
    assert_eq!(
        got59.layout.stride,
        int_tuple![
            (Int::<360>, (Int::<90>, Int::<18>, Int::<3>)),
            ((Int::<90>, Int::<18>, Int::<3>), Int::<1>)
        ]
    );
}

#[test]
fn test_im2col_hierarchical_and_blocked_spatial_modes() {
    // A renesting that coalesces ((2,4):(1,2) == 8:1) produces the exact flat view.
    let mut flat_buf = [0i64; 32];
    let mut hier_buf = [0i64; 32];
    let mut flat = make_tensor(
        &mut flat_buf,
        Layout::new(
            int_tuple![Int::<1>, Int::<4>, Int::<8>, Int::<1>],
            int_tuple![Int::<32>, Int::<8>, Int::<1>, Int::<1>],
        ),
    );
    let mut hier = make_tensor(
        &mut hier_buf,
        Layout::new(
            int_tuple![Int::<1>, Int::<4>, (Int::<2>, Int::<4>), Int::<1>],
            int_tuple![Int::<32>, Int::<8>, (Int::<1>, Int::<2>), Int::<1>],
        ),
    );
    fill_fn(&mut flat, |i| i as i64);
    fill_fn(&mut hier, |i| i as i64);

    macro_rules! check_renest {
        ($r:literal, $s:literal) => {{
            let want = im2col(flat, int_tuple![Int::<$r>, Int::<$s>]);
            let got = im2col(hier, int_tuple![Int::<$r>, Int::<$s>]);
            assert_eq!(got.layout.shape, want.layout.shape);
            assert_eq!(got.layout.stride, want.layout.stride);
            assert_eq!(flat_i64(&got), flat_i64(&want));
        }};
    }
    check_renest!(2, 2);
    check_renest!(3, 3);
    check_renest!(2, 4);

    // Blocked spatial mode: (2,4):(1,8) keeps blocking in memory view, coalesces in coord view
    let mut blk_buf = [0i64; 32];
    let mut blk = make_tensor(
        &mut blk_buf,
        Layout::new(
            int_tuple![Int::<1>, (Int::<2>, Int::<4>), Int::<1>],
            int_tuple![Int::<32>, (Int::<1>, Int::<8>), Int::<1>],
        ),
    );
    fill_fn(&mut blk, |i| i as i64);

    let b = im2col_with(
        blk,
        int_tuple![Int::<2>],
        Im2ColOptions::new().stride_trs(int_tuple![Int::<2>]),
    );
    assert_eq!(
        b.layout.shape,
        int_tuple![(Int::<1>, ((Int::<2>, Int::<3>),)), ((Int::<2>,), Int::<1>)]
    );
    assert_eq!(
        b.layout.stride,
        int_tuple![
            (Int::<32>, ((Int::<1>, Int::<8>),)),
            ((Int::<8>,), Int::<1>)
        ]
    );
    for w in 0..2 {
        for t in 0..3 {
            for tap in 0..2 {
                assert_eq!(
                    b.at(int_tuple![(0i32, ((w, t),)), ((tap,), 0i32)]),
                    blk.at(int_tuple![0i32, (w, t + tap), 0i32])
                );
            }
        }
    }

    // Both views admit blocked spatial modes and name the same elements
    macro_rules! check_blocked_pair {
        ($st:expr, $opts:expr) => {{
            let mem = im2col_with(blk, $st, $opts);
            let crd = im2col_coord_with(blk, $st, $opts);
            assert_eq!(mem.layout.shape.0.size(), crd.layout.shape.0.size());
            assert_eq!(mem.layout.shape.1.size(), crd.layout.shape.1.size());
            for i in 0..mem.size() {
                let c = idx2crd(crd.at(i), &blk.layout.shape);
                assert_eq!(mem.at(i), blk.at(c));
            }
        }};
    }
    check_blocked_pair!(
        int_tuple![Int::<2>],
        Im2ColOptions::new().stride_trs(int_tuple![Int::<2>])
    );
    check_blocked_pair!(
        int_tuple![Int::<3>],
        Im2ColOptions::new().stride_trs(int_tuple![Int::<2>])
    );
    check_blocked_pair!(
        int_tuple![Int::<4>],
        Im2ColOptions::new().stride_dhw(int_tuple![Int::<2>])
    );
}

// ------------------------------------------------------------------
// TestCoord & TestPadded
// ------------------------------------------------------------------

#[test]
fn test_im2col_coord_and_agreement_with_offset_view() {
    let mut buf = [0i64; 16];
    let mut act = make_tensor(
        &mut buf,
        Layout::new(
            int_tuple![Int::<1>, Int::<4>, Int::<4>, Int::<1>],
            int_tuple![Int::<16>, Int::<4>, Int::<1>, Int::<1>],
        ),
    );
    fill_fn(&mut act, |i| (i + 1) as i64);

    let b = im2col_coord(act, int_tuple![Int::<2>, Int::<2>]);
    assert_eq!(
        idx2crd(
            b.at(int_tuple![(0i32, (0i32, 0i32)), ((0i32, 0i32), 0i32)]),
            &act.layout.shape
        ),
        int_tuple![0i32, 0i32, 0i32, 0i32]
    );
    assert_eq!(
        idx2crd(
            b.at(int_tuple![(0i32, (1i32, 2i32)), ((1i32, 1i32), 0i32)]),
            &act.layout.shape
        ),
        int_tuple![0i32, 2i32, 3i32, 0i32]
    );

    // Padding origin lands in each view's codomain (-5 offset vs (0,-1,-1,0) coord)
    let pad_opts = Im2ColOptions::new().lower_dhw(int_tuple![Int::<1>, Int::<1>]);
    let off = im2col_with(act, int_tuple![Int::<3>, Int::<3>], pad_opts);
    let crd = im2col_coord_with(act, int_tuple![Int::<3>, Int::<3>], pad_opts);
    assert_eq!(
        off.accessor.offset_index() - act.accessor.offset_index(),
        -5
    );
    let corner = crd.at(int_tuple![(0i32, (0i32, 0i32)), ((0i32, 0i32), 0i32)]);
    assert_eq!(
        idx2crd(corner, &act.layout.shape),
        int_tuple![0i32, -1i32, -1i32, 0i32]
    );
    assert!(!in_bounds(corner, &act.layout.shape));
    let center = crd.at(int_tuple![(0i32, (1i32, 1i32)), ((1i32, 1i32), 0i32)]);
    assert!(in_bounds(center, &act.layout.shape));

    // Agreement across 5 configurations
    macro_rules! check_agree {
        ($shape:expr, $order:expr, $stencil:expr, $opts:expr) => {{
            let l = make_ordered_layout(&$shape, &$order);
            let mut data = vec![0i64; l.cosize() as usize];
            let mut a = make_tensor(&mut data[..], l);
            fill_fn(&mut a, |i| (i + 1) as i64);
            let off_v = im2col_with(a, $stencil, $opts);
            let crd_v = im2col_coord_with(a, $stencil, $opts);
            assert_eq!(off_v.size(), crd_v.size());
            for i in 0..off_v.size() {
                let c = idx2crd(crd_v.at(i), &a.layout.shape);
                if in_bounds(c, &a.layout.shape) {
                    assert_eq!(off_v.at(i), a.at(c), "mismatch at flat index {i}");
                }
            }
        }};
    }
    let ord4 = int_tuple![Int::<3>, Int::<2>, Int::<1>, Int::<0>];
    let ord5 = int_tuple![Int::<4>, Int::<3>, Int::<2>, Int::<1>, Int::<0>];
    check_agree!(
        int_tuple![Int::<2>, Int::<5>, Int::<6>, Int::<3>],
        ord4,
        int_tuple![Int::<2>, Int::<3>],
        Im2ColOptions::new()
    );
    check_agree!(
        int_tuple![Int::<2>, Int::<5>, Int::<6>, Int::<3>],
        ord4,
        int_tuple![Int::<2>, Int::<3>],
        Im2ColOptions::new().stride_dhw(int_tuple![Int::<2>, Int::<2>])
    );
    check_agree!(
        int_tuple![Int::<1>, Int::<7>, Int::<7>, Int::<2>],
        ord4,
        int_tuple![Int::<3>, Int::<3>],
        Im2ColOptions::new().stride_trs(int_tuple![Int::<2>, Int::<2>])
    );
    check_agree!(
        int_tuple![Int::<1>, Int::<7>, Int::<7>, Int::<2>],
        ord4,
        int_tuple![Int::<3>, Int::<3>],
        Im2ColOptions::new().lower_dhw(int_tuple![Int::<1>, Int::<1>])
    );
    check_agree!(
        int_tuple![Int::<2>, Int::<4>, Int::<5>, Int::<6>, Int::<2>],
        ord5,
        int_tuple![Int::<2>, Int::<2>, Int::<2>],
        Im2ColOptions::new()
            .stride_dhw(int_tuple![Int::<1>, Int::<2>, Int::<1>])
            .stride_trs(int_tuple![Int::<1>, Int::<1>, Int::<2>])
    );
}

#[test]
fn test_im2col_padded_fill_slicing_hierarchical_and_direct_accessor() {
    let mut buf = [0i64; 9];
    let mut act = make_tensor(
        &mut buf,
        make_ordered_layout(
            &int_tuple![Int::<1>, Int::<3>, Int::<3>, Int::<1>],
            &int_tuple![Int::<3>, Int::<2>, Int::<1>, Int::<0>],
        ),
    );
    fill_fn(&mut act, |i| (i + 1) as i64);

    let opts = Im2ColOptions::new().lower_dhw(int_tuple![Int::<1>, Int::<1>]);
    let b = im2col_padded_with(act, int_tuple![Int::<3>, Int::<3>], 0i64, opts);
    assert_eq!(
        b.at(int_tuple![(0i32, (1i32, 1i32)), ((1i32, 1i32), 0i32)]),
        act.at(int_tuple![0i32, 1i32, 1i32, 0i32])
    );
    assert_eq!(
        b.at(int_tuple![(0i32, (0i32, 0i32)), ((0i32, 0i32), 0i32)]),
        0
    );

    let b_fill = im2col_padded_with(act, int_tuple![Int::<3>, Int::<3>], -7i64, opts);
    assert_eq!(
        b_fill.at(int_tuple![(0i32, (0i32, 0i32)), ((0i32, 0i32), 0i32)]),
        -7
    );

    // Slicing keeps bounds checking (exact 9 elements of corner window)
    let row = b.at(noodle_cute::tuple![(0i32, (0i32, 0i32)), ..]);
    assert_eq!(
        flat_i64(&row),
        vec![
            0,
            0,
            0,
            0,
            act.at(int_tuple![0i32, 0i32, 0i32, 0i32]),
            act.at(int_tuple![0i32, 1i32, 0i32, 0i32]),
            0,
            act.at(int_tuple![0i32, 0i32, 1i32, 0i32]),
            act.at(int_tuple![0i32, 1i32, 1i32, 0i32]),
        ]
    );

    // Hierarchical channel mode indexed as one nested mode
    let mut hc_buf = [0i64; 36];
    let mut act_hc = make_tensor(
        &mut hc_buf,
        Layout::new(
            int_tuple![Int::<1>, Int::<3>, Int::<3>, (Int::<2>, Int::<2>)],
            int_tuple![Int::<36>, Int::<12>, Int::<4>, (Int::<2>, Int::<1>)],
        ),
    );
    fill_fn(&mut act_hc, |i| (i + 1) as i64);
    let b_hc = im2col_padded_with(act_hc, int_tuple![Int::<3>, Int::<3>], 0i64, opts);
    assert_eq!(
        b_hc.layout.shape,
        int_tuple![
            (Int::<1>, (Int::<3>, Int::<3>)),
            ((Int::<3>, Int::<3>), (Int::<2>, Int::<2>))
        ]
    );
    assert_eq!(
        b_hc.at(int_tuple![
            (0i32, (1i32, 1i32)),
            ((1i32, 1i32), (1i32, 1i32))
        ]),
        act_hc.at(int_tuple![0i32, 1i32, 1i32, (1i32, 1i32)])
    );
    assert_eq!(
        b_hc.at(int_tuple![
            (0i32, (0i32, 0i32)),
            ((0i32, 0i32), (0i32, 0i32))
        ]),
        0
    );

    // Direct OutOfBoundsAccessor usage
    let mut small_buf = [0.0f64; 4];
    let mut small = make_tensor(
        &mut small_buf,
        Layout::new(
            int_tuple![Int::<2>, Int::<2>],
            int_tuple![Int::<2>, Int::<1>],
        ),
    );
    small.set(int_tuple![1i32, 1i32], 5.0);
    assert_eq!(
        OutOfBoundsAccessor::new(small, 0.0).load(int_tuple![1i32, 1i32]),
        5.0
    );
    assert_eq!(
        OutOfBoundsAccessor::new(small, 0.0).load(int_tuple![2i32, 0i32]),
        0.0
    );
    assert_eq!(
        OutOfBoundsAccessor::new(small, -1.0).load(int_tuple![-1i32, 0i32]),
        -1.0
    );
}

// ------------------------------------------------------------------
// TestConvolution: ALL 12 convolution cases via gemm + einfold + im2col
// ------------------------------------------------------------------

struct ConvCase<const D: usize> {
    n: int,
    c: int,
    k: int,
    dhw: [int; D],
    st: [int; D],
    lo: [int; D],
    hi: [int; D],
    u: [int; D],
    s: [int; D],
}

fn check_conv_1d(case: ConvCase<1>) {
    let ConvCase {
        n,
        c,
        k,
        dhw: [w],
        st: [s_tap],
        lo: [lower_dhw],
        hi: [upper_dhw],
        u: [stride_dhw],
        s: [stride_trs],
    } = case;
    let zpq = zpq_oracle(
        &[w],
        &[s_tap],
        &[lower_dhw],
        &[upper_dhw],
        &[stride_dhw],
        &[stride_trs],
    );
    let act_l = make_ordered_layout(
        &int_tuple![n, w, c],
        &int_tuple![Int::<2>, Int::<1>, Int::<0>],
    );
    let flt_l = make_ordered_layout(
        &int_tuple![k, s_tap, c],
        &int_tuple![Int::<2>, Int::<1>, Int::<0>],
    );
    let out_l = make_ordered_layout(
        &int_tuple![n, zpq[0], k],
        &int_tuple![Int::<2>, Int::<1>, Int::<0>],
    );

    let mut act_buf = vec![0i64; act_l.cosize() as usize];
    let mut flt_buf = vec![0i64; flt_l.cosize() as usize];
    let mut out_buf = vec![0i64; out_l.cosize() as usize];

    let mut act = make_tensor(&mut act_buf[..], act_l);
    let mut flt = make_tensor(&mut flt_buf[..], flt_l);
    let out = make_tensor(&mut out_buf[..], out_l);
    fill_fn(&mut act, |i| (i % 7 - 3) as i64);
    fill_fn(&mut flt, |i| (i % 5 - 2) as i64);

    let opts = Im2ColOptions::new()
        .lower_dhw(lower_dhw)
        .upper_dhw(upper_dhw)
        .stride_dhw(stride_dhw)
        .stride_trs(stride_trs);
    let b_nk = einfold!("ksc -> k((s)c)", flt);
    let mut c_mn = einfold!("nqk -> (n(q))k", out);

    if lower_dhw != 0 || upper_dhw != 0 {
        let a_mk = im2col_padded_with(act, s_tap, 0i64, opts);
        gemm(a_mk, b_nk, &mut c_mn);
    } else {
        let a_mk = im2col_with(act, s_tap, opts);
        gemm(a_mk, b_nk, &mut c_mn);
    }

    let spec = ConvSpec {
        n,
        c,
        k,
        shape_dhw: &[w],
        stencil: &[s_tap],
        lower_dhw: &[lower_dhw],
        upper_dhw: &[upper_dhw],
        stride_dhw: &[stride_dhw],
        stride_trs: &[stride_trs],
    };
    let want = conv_oracle_nd(
        &spec,
        |ni, crd, ci| act.at(int_tuple![ni, crd[0], ci]),
        |ki, tap, ci| flt.at(int_tuple![ki, tap[0], ci]),
    );
    for ((ni, pos, ki), val) in want {
        assert_eq!(out.at(int_tuple![ni, pos[0], ki]), val);
    }
}

fn check_conv_2d(case: ConvCase<2>) {
    let ConvCase {
        n,
        c,
        k,
        dhw,
        st,
        lo,
        hi,
        u,
        s,
    } = case;
    let zpq = zpq_oracle(&dhw, &st, &lo, &hi, &u, &s);
    let ord = int_tuple![Int::<3>, Int::<2>, Int::<1>, Int::<0>];
    let act_l = make_ordered_layout(&int_tuple![n, dhw[0], dhw[1], c], &ord);
    let flt_l = make_ordered_layout(&int_tuple![k, st[0], st[1], c], &ord);
    let out_l = make_ordered_layout(&int_tuple![n, zpq[0], zpq[1], k], &ord);

    let mut act_buf = vec![0i64; act_l.cosize() as usize];
    let mut flt_buf = vec![0i64; flt_l.cosize() as usize];
    let mut out_buf = vec![0i64; out_l.cosize() as usize];

    let mut act = make_tensor(&mut act_buf[..], act_l);
    let mut flt = make_tensor(&mut flt_buf[..], flt_l);
    let out = make_tensor(&mut out_buf[..], out_l);
    fill_fn(&mut act, |i| (i % 7 - 3) as i64);
    fill_fn(&mut flt, |i| (i % 5 - 2) as i64);

    let opts = Im2ColOptions::new()
        .lower_dhw(int_tuple![lo[0], lo[1]])
        .upper_dhw(int_tuple![hi[0], hi[1]])
        .stride_dhw(int_tuple![u[0], u[1]])
        .stride_trs(int_tuple![s[0], s[1]]);
    let b_nk = einfold!("krsc -> k((rs)c)", flt);
    let mut c_mn = einfold!("npqk -> (n(pq))k", out);

    if lo.iter().any(|&v| v != 0) || hi.iter().any(|&v| v != 0) {
        let a_mk = im2col_padded_with(act, int_tuple![st[0], st[1]], 0i64, opts);
        gemm(a_mk, b_nk, &mut c_mn);
    } else {
        let a_mk = im2col_with(act, int_tuple![st[0], st[1]], opts);
        gemm(a_mk, b_nk, &mut c_mn);
    }

    let spec = ConvSpec {
        n,
        c,
        k,
        shape_dhw: &dhw,
        stencil: &st,
        lower_dhw: &lo,
        upper_dhw: &hi,
        stride_dhw: &u,
        stride_trs: &s,
    };
    let want = conv_oracle_nd(
        &spec,
        |ni, crd, ci| act.at(int_tuple![ni, crd[0], crd[1], ci]),
        |ki, tap, ci| flt.at(int_tuple![ki, tap[0], tap[1], ci]),
    );
    for ((ni, pos, ki), val) in want {
        assert_eq!(out.at(int_tuple![ni, pos[0], pos[1], ki]), val);
    }
}

fn check_conv_3d(case: ConvCase<3>) {
    let ConvCase {
        n,
        c,
        k,
        dhw,
        st,
        lo,
        hi,
        u,
        s,
    } = case;
    let zpq = zpq_oracle(&dhw, &st, &lo, &hi, &u, &s);
    let ord = int_tuple![Int::<4>, Int::<3>, Int::<2>, Int::<1>, Int::<0>];
    let act_l = make_ordered_layout(&int_tuple![n, dhw[0], dhw[1], dhw[2], c], &ord);
    let flt_l = make_ordered_layout(&int_tuple![k, st[0], st[1], st[2], c], &ord);
    let out_l = make_ordered_layout(&int_tuple![n, zpq[0], zpq[1], zpq[2], k], &ord);

    let mut act_buf = vec![0i64; act_l.cosize() as usize];
    let mut flt_buf = vec![0i64; flt_l.cosize() as usize];
    let mut out_buf = vec![0i64; out_l.cosize() as usize];

    let mut act = make_tensor(&mut act_buf[..], act_l);
    let mut flt = make_tensor(&mut flt_buf[..], flt_l);
    let out = make_tensor(&mut out_buf[..], out_l);
    fill_fn(&mut act, |i| (i % 7 - 3) as i64);
    fill_fn(&mut flt, |i| (i % 5 - 2) as i64);

    let opts = Im2ColOptions::new()
        .lower_dhw(int_tuple![lo[0], lo[1], lo[2]])
        .upper_dhw(int_tuple![hi[0], hi[1], hi[2]])
        .stride_dhw(int_tuple![u[0], u[1], u[2]])
        .stride_trs(int_tuple![s[0], s[1], s[2]]);
    let b_nk = einfold!("ktrsc -> k((trs)c)", flt);
    let mut c_mn = einfold!("nzpqk -> (n(zpq))k", out);

    if lo.iter().any(|&v| v != 0) || hi.iter().any(|&v| v != 0) {
        let a_mk = im2col_padded_with(act, int_tuple![st[0], st[1], st[2]], 0i64, opts);
        gemm(a_mk, b_nk, &mut c_mn);
    } else {
        let a_mk = im2col_with(act, int_tuple![st[0], st[1], st[2]], opts);
        gemm(a_mk, b_nk, &mut c_mn);
    }

    let spec = ConvSpec {
        n,
        c,
        k,
        shape_dhw: &dhw,
        stencil: &st,
        lower_dhw: &lo,
        upper_dhw: &hi,
        stride_dhw: &u,
        stride_trs: &s,
    };
    let want = conv_oracle_nd(
        &spec,
        |ni, crd, ci| act.at(int_tuple![ni, crd[0], crd[1], crd[2], ci]),
        |ki, tap, ci| flt.at(int_tuple![ki, tap[0], tap[1], tap[2], ci]),
    );
    for ((ni, pos, ki), val) in want {
        assert_eq!(out.at(int_tuple![ni, pos[0], pos[1], pos[2], ki]), val);
    }
}

#[test]
fn test_convolution_all_12_cases() {
    // 1. 2D single channel
    check_conv_2d(ConvCase {
        n: 1,
        c: 1,
        k: 1,
        dhw: [4, 4],
        st: [2, 2],
        lo: [0, 0],
        hi: [0, 0],
        u: [1, 1],
        s: [1, 1],
    });
    // 2. 2D multi-channel batched
    check_conv_2d(ConvCase {
        n: 2,
        c: 3,
        k: 4,
        dhw: [5, 6],
        st: [2, 3],
        lo: [0, 0],
        hi: [0, 0],
        u: [1, 1],
        s: [1, 1],
    });
    // 3. 2D traversal stride
    check_conv_2d(ConvCase {
        n: 2,
        c: 2,
        k: 3,
        dhw: [7, 7],
        st: [3, 3],
        lo: [0, 0],
        hi: [0, 0],
        u: [2, 2],
        s: [1, 1],
    });
    // 4. 2D dilation
    check_conv_2d(ConvCase {
        n: 1,
        c: 2,
        k: 2,
        dhw: [7, 7],
        st: [3, 3],
        lo: [0, 0],
        hi: [0, 0],
        u: [1, 1],
        s: [2, 2],
    });
    // 5. 2D padding
    check_conv_2d(ConvCase {
        n: 2,
        c: 2,
        k: 3,
        dhw: [5, 5],
        st: [3, 3],
        lo: [1, 1],
        hi: [1, 1],
        u: [1, 1],
        s: [1, 1],
    });
    // 6. 2D asymmetric padding
    check_conv_2d(ConvCase {
        n: 1,
        c: 1,
        k: 2,
        dhw: [5, 5],
        st: [2, 2],
        lo: [1, 1],
        hi: [0, 0],
        u: [1, 1],
        s: [1, 1],
    });
    // 7. 2D padding + stride + dilation
    check_conv_2d(ConvCase {
        n: 2,
        c: 2,
        k: 2,
        dhw: [9, 9],
        st: [3, 3],
        lo: [2, 2],
        hi: [2, 2],
        u: [2, 2],
        s: [2, 2],
    });
    // 8. 1D unpadded
    check_conv_1d(ConvCase {
        n: 2,
        c: 3,
        k: 4,
        dhw: [8],
        st: [3],
        lo: [0],
        hi: [0],
        u: [1],
        s: [1],
    });
    // 9. 1D padded
    check_conv_1d(ConvCase {
        n: 1,
        c: 2,
        k: 2,
        dhw: [8],
        st: [3],
        lo: [1],
        hi: [1],
        u: [1],
        s: [1],
    });
    // 10. 3D unpadded
    check_conv_3d(ConvCase {
        n: 2,
        c: 2,
        k: 3,
        dhw: [4, 5, 6],
        st: [2, 3, 2],
        lo: [0, 0, 0],
        hi: [0, 0, 0],
        u: [1, 1, 1],
        s: [1, 1, 1],
    });
    // 11. 3D mixed parameters
    check_conv_3d(ConvCase {
        n: 1,
        c: 2,
        k: 2,
        dhw: [6, 7, 8],
        st: [2, 3, 2],
        lo: [0, 1, 1],
        hi: [0, 1, 1],
        u: [1, 2, 1],
        s: [1, 1, 2],
    });
    // 12. 1x1 plain GEMM
    check_conv_2d(ConvCase {
        n: 2,
        c: 4,
        k: 3,
        dhw: [5, 5],
        st: [1, 1],
        lo: [0, 0],
        hi: [0, 0],
        u: [1, 1],
        s: [1, 1],
    });
}

// ------------------------------------------------------------------
// TestDgrad
// ------------------------------------------------------------------

#[test]
fn test_dgrad_reversed_stencil() {
    let buf = [0i64; 6];
    let act = make_tensor(
        &buf,
        make_ordered_layout(
            &int_tuple![Int::<1>, Int::<6>, Int::<1>],
            &int_tuple![Int::<2>, Int::<1>, Int::<0>],
        ),
    );
    let opts = Im2ColOptions::new()
        .lower_trs(Int::<2>)
        .stride_trs(Int::<-1>)
        .shape_zpq(Int::<6>)
        .lower_dhw(Int::<2>);
    let b = im2col_coord_with(act, Int::<3>, opts);

    for z in 0..6 {
        for t in 0..3 {
            let crd = idx2crd(
                b.at(int_tuple![(0i32, (z,)), ((t,), 0i32)]),
                &act.layout.shape,
            );
            assert_eq!(crd.1.0, z - t);
        }
    }

    // Also test unpadded `im2col_padded` convenience wrapper
    let p = im2col_padded(act, Int::<3>, -9i64);
    assert_eq!(
        p.layout.shape,
        int_tuple![(Int::<1>, (Int::<4>,)), ((Int::<3>,), Int::<1>)]
    );
}

#[test]
fn crossing_a_block_requires_coordinate_addressing() {
    let buf: [i32; 32] = core::array::from_fn(|i| i as i32);
    let act = make_tensor(
        &buf,
        Layout::new(
            int_tuple![Int::<1>, (Int::<2>, Int::<4>), Int::<1>],
            int_tuple![Int::<32>, (Int::<1>, Int::<8>), Int::<1>],
        ),
    );
    let opts = Im2ColOptions::new().shape_zpq(Int::<2>).lower_trs(0i32);
    assert!(std::panic::catch_unwind(|| im2col_with(act, Int::<2>, opts)).is_err());
    let view = im2col_padded_with(act, Int::<2>, -1, opts);
    for z in 0..2 {
        for t in 0..2 {
            assert_eq!(view(z, t), act(0i32, z + t, 0i32));
        }
    }
}
