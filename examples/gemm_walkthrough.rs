//! Runnable walkthrough mirroring `examples/algorithms/gemm.ipynb`.
//!
//! Shows how one unified CuTe GEMM definition `C(m, n, l) += sum_k A(m, k, l) * B(n, k, l)`
//! covers BLAS NT/TN/NN/TT, strided/batched/broadcast GEMM, multi-mode tensor contractions
//! (GETT via `einfold!` / `einsum!`), and N-D convolution via `im2col_padded`.
//!
//! Run with: `cargo run --example gemm_walkthrough`

#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use noodle_cute::algorithm::im2col::{Im2ColOptions, im2col_padded_with};
use noodle_cute::algorithm::reference::gemm;
use noodle_cute::integer::Int;
use noodle_cute::layout::{Layout, make_ordered_layout};
use noodle_cute::tensor::make_tensor;
use noodle_cute::{einfold, einsum, int_tuple};

fn main() {
    println!("=== 1. Rank-2 BLAS GEMM (NT col-major vs TN row-major) ===");
    let a_buf = [1i64, 2, 3, 4, 5, 6];
    let b_buf = [7i64, 8, 9, 10, 11, 12];
    let mut c_nt = [0i64; 4];
    {
        let a = make_tensor(
            &a_buf,
            Layout::new(
                int_tuple![Int::<2>, Int::<3>],
                int_tuple![Int::<1>, Int::<2>],
            ),
        );
        let b = make_tensor(
            &b_buf,
            Layout::new(
                int_tuple![Int::<2>, Int::<3>],
                int_tuple![Int::<1>, Int::<2>],
            ),
        );
        let mut c = make_tensor(
            &mut c_nt,
            Layout::new(
                int_tuple![Int::<2>, Int::<2>],
                int_tuple![Int::<1>, Int::<2>],
            ),
        );
        gemm(a, b, &mut c);
    }
    println!("NT 2x2 result (col-major buffer): {c_nt:?}");

    println!("\n=== 2. Batched GEMM with Broadcast Weight (L stride = 0) ===");
    let a_batch: Vec<i64> = (1..=12).collect();
    let b_shared = [1i64, 0, 0, 1];
    let mut c_batch = [0i64; 12];
    {
        let a = make_tensor(
            &a_batch[..],
            Layout::from_shape(int_tuple![Int::<2>, Int::<2>, Int::<3>]),
        );
        let b = make_tensor(
            &b_shared,
            Layout::new(
                int_tuple![Int::<2>, Int::<2>, Int::<3>],
                int_tuple![Int::<1>, Int::<2>, Int::<0>],
            ),
        );
        let mut c = make_tensor(
            &mut c_batch[..],
            Layout::from_shape(int_tuple![Int::<2>, Int::<2>, Int::<3>]),
        );
        gemm(a, b, &mut c);
    }
    assert_eq!(&c_batch[..], &a_batch[..]);
    println!(
        "Batched identity-weight check passed ({} elements).",
        c_batch.len()
    );

    println!("\n=== 3. Hierarchical GETT via einsum! (Whitepaper C_stqp = A_stupr * B_qtru) ===");
    let a_gett: Vec<i64> = (1..=72).collect();
    let b_gett: Vec<i64> = (1..=36).map(|x| x % 5 - 2).collect();
    let mut c_gett = [0i64; 24];
    {
        let a = make_tensor(
            &a_gett[..],
            Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<2>, Int::<2>, Int::<3>]),
        );
        let b = make_tensor(
            &b_gett[..],
            Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<3>, Int::<2>]),
        );
        let mut c = make_tensor(
            &mut c_gett[..],
            Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<2>, Int::<2>]),
        );
        einsum!("stupr,qtru->stqp", a, b, &mut c);
    }
    println!("GETT C_stqp first 8 outputs: {:?}", &c_gett[..8]);

    println!("\n=== 4. 2-D Padded Convolution via im2col_padded + einfold! + gemm ===");
    let ord = int_tuple![Int::<3>, Int::<2>, Int::<1>, Int::<0>];
    let act_buf = [1i64, 2, 3, 4, 5, 6, 7, 8, 9];
    let flt_buf = [1i64; 9]; // 3x3 box sum filter
    let mut out_buf = [0i64; 9];
    {
        let act = make_tensor(
            &act_buf,
            make_ordered_layout(&int_tuple![Int::<1>, Int::<3>, Int::<3>, Int::<1>], &ord),
        );
        let flt = make_tensor(
            &flt_buf,
            make_ordered_layout(&int_tuple![Int::<1>, Int::<3>, Int::<3>, Int::<1>], &ord),
        );
        let out = make_tensor(
            &mut out_buf,
            make_ordered_layout(&int_tuple![Int::<1>, Int::<3>, Int::<3>, Int::<1>], &ord),
        );

        let a_mk = im2col_padded_with(
            act,
            int_tuple![Int::<3>, Int::<3>],
            0i64,
            Im2ColOptions::new().lower_dhw(int_tuple![Int::<1>, Int::<1>]),
        );
        let b_nk = einfold!("krsc -> k((rs)c)", flt);
        let mut c_mn = einfold!("npqk -> (n(pq))k", out);
        gemm(a_mk, b_nk, &mut c_mn);
    }
    assert_eq!(out_buf, [12, 21, 16, 27, 45, 33, 24, 39, 28]);
    println!("3x3 padded box-filter output = {out_buf:?}");
}
