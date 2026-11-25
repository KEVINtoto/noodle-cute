#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use std::panic::{AssertUnwindSafe, catch_unwind};

use noodle_cute::algorithm::reference::gemm;
use noodle_cute::int_tuple;
use noodle_cute::integer::{Int, int};
use noodle_cute::layout::Layout;
use noodle_cute::tensor::make_tensor;

/// Direct `(M, K, L) x (N, K, L) -> (M, N, L)` oracle.
fn gemm_oracle(
    m: int,
    n: int,
    k: int,
    l: int,
    a_fn: impl Fn(int, int, int) -> i64,
    b_fn: impl Fn(int, int, int) -> i64,
    c_init: impl Fn(int, int, int) -> i64,
) -> Vec<Vec<Vec<i64>>> {
    let mut out = vec![vec![vec![0i64; l as usize]; n as usize]; m as usize];
    for mi in 0..m {
        for ni in 0..n {
            for li in 0..l {
                let mut acc = c_init(mi, ni, li);
                for ki in 0..k {
                    acc += a_fn(mi, ki, li) * b_fn(ni, ki, li);
                }
                out[mi as usize][ni as usize][li as usize] = acc;
            }
        }
    }
    out
}

#[test]
fn test_gemm_blas_variants_nt_tn_nn_tt() {
    let (m, n, k) = (4, 5, 3);
    let a_vals: Vec<i64> = (1..=(m * k) as i64).collect();
    let b_vals: Vec<i64> = (1..=(n * k) as i64).map(|x| x * 2 - 5).collect();

    // NT: A = (M,K):(1,M) col-major, B = (N,K):(1,N) col-major, C = (M,N):(1,M) col-major
    {
        let a = make_tensor(
            &a_vals[..],
            Layout::new(
                int_tuple![Int::<4>, Int::<3>],
                int_tuple![Int::<1>, Int::<4>],
            ),
        );
        let b = make_tensor(
            &b_vals[..],
            Layout::new(
                int_tuple![Int::<5>, Int::<3>],
                int_tuple![Int::<1>, Int::<5>],
            ),
        );
        let mut c_buf = vec![7i64; (m * n) as usize];
        let mut c = make_tensor(
            &mut c_buf[..],
            Layout::new(
                int_tuple![Int::<4>, Int::<5>],
                int_tuple![Int::<1>, Int::<4>],
            ),
        );
        gemm(a, b, &mut c);
        let want = gemm_oracle(
            m,
            n,
            k,
            1,
            |mi, ki, _| a.at(int_tuple![mi, ki]),
            |ni, ki, _| b.at(int_tuple![ni, ki]),
            |_, _, _| 7,
        );
        for mi in 0..m {
            for ni in 0..n {
                assert_eq!(c.at(int_tuple![mi, ni]), want[mi as usize][ni as usize][0]);
            }
        }
    }

    // TN: A = (M,K):(K,1) row-major, B = (N,K):(K,1) row-major, C = (M,N):(N,1) row-major
    {
        let a = make_tensor(
            &a_vals[..],
            Layout::new(
                int_tuple![Int::<4>, Int::<3>],
                int_tuple![Int::<3>, Int::<1>],
            ),
        );
        let b = make_tensor(
            &b_vals[..],
            Layout::new(
                int_tuple![Int::<5>, Int::<3>],
                int_tuple![Int::<3>, Int::<1>],
            ),
        );
        let mut c_buf = vec![0i64; (m * n) as usize];
        let mut c = make_tensor(
            &mut c_buf[..],
            Layout::new(
                int_tuple![Int::<4>, Int::<5>],
                int_tuple![Int::<5>, Int::<1>],
            ),
        );
        gemm(a, b, &mut c);
        let want = gemm_oracle(
            m,
            n,
            k,
            1,
            |mi, ki, _| a.at(int_tuple![mi, ki]),
            |ni, ki, _| b.at(int_tuple![ni, ki]),
            |_, _, _| 0,
        );
        for mi in 0..m {
            for ni in 0..n {
                assert_eq!(c.at(int_tuple![mi, ni]), want[mi as usize][ni as usize][0]);
            }
        }
    }

    // NN: A = (M,K):(1,M) col-major, B = (N,K):(K,1) row-major, C = (M,N):(1,M)
    {
        let a = make_tensor(
            &a_vals[..],
            Layout::new(
                int_tuple![Int::<4>, Int::<3>],
                int_tuple![Int::<1>, Int::<4>],
            ),
        );
        let b = make_tensor(
            &b_vals[..],
            Layout::new(
                int_tuple![Int::<5>, Int::<3>],
                int_tuple![Int::<3>, Int::<1>],
            ),
        );
        let mut c_buf = vec![0i64; (m * n) as usize];
        let mut c = make_tensor(
            &mut c_buf[..],
            Layout::new(
                int_tuple![Int::<4>, Int::<5>],
                int_tuple![Int::<1>, Int::<4>],
            ),
        );
        gemm(a, b, &mut c);
        let want = gemm_oracle(
            m,
            n,
            k,
            1,
            |mi, ki, _| a.at(int_tuple![mi, ki]),
            |ni, ki, _| b.at(int_tuple![ni, ki]),
            |_, _, _| 0,
        );
        for mi in 0..m {
            for ni in 0..n {
                assert_eq!(c.at(int_tuple![mi, ni]), want[mi as usize][ni as usize][0]);
            }
        }
    }

    // TT: A = (M,K):(K,1) row-major, B = (N,K):(1,N) col-major, C = (M,N):(N,1)
    {
        let a = make_tensor(
            &a_vals[..],
            Layout::new(
                int_tuple![Int::<4>, Int::<3>],
                int_tuple![Int::<3>, Int::<1>],
            ),
        );
        let b = make_tensor(
            &b_vals[..],
            Layout::new(
                int_tuple![Int::<5>, Int::<3>],
                int_tuple![Int::<1>, Int::<5>],
            ),
        );
        let mut c_buf = vec![0i64; (m * n) as usize];
        let mut c = make_tensor(
            &mut c_buf[..],
            Layout::new(
                int_tuple![Int::<4>, Int::<5>],
                int_tuple![Int::<5>, Int::<1>],
            ),
        );
        gemm(a, b, &mut c);
        let want = gemm_oracle(
            m,
            n,
            k,
            1,
            |mi, ki, _| a.at(int_tuple![mi, ki]),
            |ni, ki, _| b.at(int_tuple![ni, ki]),
            |_, _, _| 0,
        );
        for mi in 0..m {
            for ni in 0..n {
                assert_eq!(c.at(int_tuple![mi, ni]), want[mi as usize][ni as usize][0]);
            }
        }
    }
}

#[test]
fn test_gemm_batched_and_broadcast_batch() {
    // Standard batched GEMM: M=3, N=2, K=4, L=3
    let a_buf: Vec<i64> = (1..=36).collect();
    let b_buf: Vec<i64> = (1..=24).map(|x| x - 10).collect();
    let mut c_buf = [0i64; 18];

    let a = make_tensor(
        &a_buf[..],
        Layout::from_shape(int_tuple![Int::<3>, Int::<4>, Int::<3>]),
    );
    let b = make_tensor(
        &b_buf[..],
        Layout::from_shape(int_tuple![Int::<2>, Int::<4>, Int::<3>]),
    );
    let mut c = make_tensor(
        &mut c_buf[..],
        Layout::from_shape(int_tuple![Int::<3>, Int::<2>, Int::<3>]),
    );
    gemm(a, b, &mut c);

    let want = gemm_oracle(
        3,
        2,
        4,
        3,
        |mi, ki, li| a.at(int_tuple![mi, ki, li]),
        |ni, ki, li| b.at(int_tuple![ni, ki, li]),
        |_, _, _| 0,
    );
    for mi in 0..3 {
        for ni in 0..2 {
            for li in 0..3 {
                assert_eq!(
                    c.at(int_tuple![mi, ni, li]),
                    want[mi as usize][ni as usize][li as usize]
                );
            }
        }
    }

    // Broadcast batch on B: stride 0 along L shares one weight matrix across all batches
    let b_shared_buf: Vec<i64> = (1..=8).collect();
    let b_shared = make_tensor(
        &b_shared_buf[..],
        Layout::new(
            int_tuple![Int::<2>, Int::<4>, Int::<3>],
            int_tuple![Int::<1>, Int::<2>, Int::<0>],
        ),
    );
    let mut c2_buf = [0i64; 18];
    let mut c2 = make_tensor(
        &mut c2_buf[..],
        Layout::from_shape(int_tuple![Int::<3>, Int::<2>, Int::<3>]),
    );
    gemm(a, b_shared, &mut c2);

    let want2 = gemm_oracle(
        3,
        2,
        4,
        3,
        |mi, ki, li| a.at(int_tuple![mi, ki, li]),
        |ni, ki, li| b_shared.at(int_tuple![ni, ki, li]),
        |_, _, _| 0,
    );
    for mi in 0..3 {
        for ni in 0..2 {
            for li in 0..3 {
                assert_eq!(
                    c2.at(int_tuple![mi, ni, li]),
                    want2[mi as usize][ni as usize][li as usize]
                );
            }
        }
    }
}

#[test]
fn test_gemm_hierarchical_gett_and_f64() {
    // Hierarchical M=(2,3), N=(2,2), K=(2,2), L=(2,2) (Whitepaper Table 3 GETT)
    let a_buf: Vec<f64> = (0..96).map(|i| (i % 11) as f64 - 5.0).collect();
    let b_buf: Vec<f64> = (0..64).map(|i| (i % 7) as f64 - 3.0).collect();
    let mut c_buf = vec![1.5f64; 96];

    let a = make_tensor(
        &a_buf[..],
        Layout::from_shape(int_tuple![
            (Int::<2>, Int::<3>),
            (Int::<2>, Int::<2>),
            (Int::<2>, Int::<2>)
        ]),
    );
    let b = make_tensor(
        &b_buf[..],
        Layout::from_shape(int_tuple![
            (Int::<2>, Int::<2>),
            (Int::<2>, Int::<2>),
            (Int::<2>, Int::<2>)
        ]),
    );
    let mut c = make_tensor(
        &mut c_buf[..],
        Layout::from_shape(int_tuple![
            (Int::<2>, Int::<3>),
            (Int::<2>, Int::<2>),
            (Int::<2>, Int::<2>)
        ]),
    );
    gemm(a, b, &mut c);

    for mi in 0..6 {
        for ni in 0..4 {
            for li in 0..4 {
                let mut acc = 1.5;
                for ki in 0..4 {
                    acc += a.at(int_tuple![mi, ki, li]) * b.at(int_tuple![ni, ki, li]);
                }
                assert!((c.at(int_tuple![mi, ni, li]) - acc).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn test_gemm_dynamic_dimension_mismatch_panics() {
    let a_buf = [0i32; 24];
    let b_buf = [0i32; 24];
    let mut c_buf = [0i32; 24];

    // M mismatch (3 vs 4)
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let a = make_tensor(&a_buf, Layout::from_shape(int_tuple![3i32, 2i32]));
            let b = make_tensor(&b_buf, Layout::from_shape(int_tuple![2i32, 2i32]));
            let mut c = make_tensor(&mut c_buf[..], Layout::from_shape(int_tuple![4i32, 2i32]));
            gemm(a, b, &mut c);
        }))
        .is_err()
    );

    // N mismatch (2 vs 3)
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let a = make_tensor(&a_buf, Layout::from_shape(int_tuple![3i32, 2i32]));
            let b = make_tensor(&b_buf, Layout::from_shape(int_tuple![2i32, 2i32]));
            let mut c = make_tensor(&mut c_buf[..], Layout::from_shape(int_tuple![3i32, 3i32]));
            gemm(a, b, &mut c);
        }))
        .is_err()
    );

    // K mismatch (2 vs 3)
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let a = make_tensor(&a_buf, Layout::from_shape(int_tuple![2i32, 2i32]));
            let b = make_tensor(&b_buf, Layout::from_shape(int_tuple![2i32, 3i32]));
            let mut c = make_tensor(&mut c_buf[..], Layout::from_shape(int_tuple![2i32, 2i32]));
            gemm(a, b, &mut c);
        }))
        .is_err()
    );

    // L mismatch (2 vs 3)
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let a = make_tensor(&a_buf, Layout::from_shape(int_tuple![2i32, 2i32, 2i32]));
            let b = make_tensor(&b_buf, Layout::from_shape(int_tuple![2i32, 2i32, 3i32]));
            let mut c = make_tensor(
                &mut c_buf[..],
                Layout::from_shape(int_tuple![2i32, 2i32, 2i32]),
            );
            gemm(a, b, &mut c);
        }))
        .is_err()
    );
}
