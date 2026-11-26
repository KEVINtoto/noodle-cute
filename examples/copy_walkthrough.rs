//! Runnable walkthrough mirroring `examples/algorithms/copy.ipynb`.
//!
//! Demonstrates the three-stage CuTe copy layout analysis (`greatest_common_domain` ->
//! `nullspace` & write-after-write rejection -> `right_inverse` & vectorization) and compares
//! write patterns against the naive reference copy.
//!
//! Run with: `cargo run --example copy_walkthrough`

#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    const_trait_impl
)]
#![allow(incomplete_features)]

use noodle_cute::algebra::{
    coalesce, greatest_common_domain, logical_divide, nullspace, right_inverse,
};
use noodle_cute::algorithm::copy::{copy, copy_plan};
use noodle_cute::algorithm::reference::copy as copy_ref;
use noodle_cute::integer::Int;
use noodle_cute::layout::Layout;
use noodle_cute::tensor::{ImplicitAccessor, Tensor, make_tensor};
use noodle_cute::{int_tuple, tuple};

macro_rules! show_stage3 {
    ($label:expr, $src:expr, $dst:expr) => {{
        let src_l = $src;
        let dst_l = $dst;
        let (src_v, _) = copy_plan(
            Tensor::new(ImplicitAccessor::new(Int::<0>), src_l),
            Tensor::new(ImplicitAccessor::new(Int::<0>), dst_l),
        );
        let run = src_v.at(tuple![.., 0i32]).layout;
        println!(
            "{:<14} | src={:<28} dst={:<28} -> vec={:>2}, c_src={}",
            $label,
            format!("{src_l}"),
            format!("{dst_l}"),
            run.size(),
            coalesce(run),
        );
    }};
}

fn main() {
    println!("=== Stage 1: Greatest Common Domain & Tiling ===");
    let src_l = Layout::new(
        int_tuple![Int::<4>, (Int::<2>, Int::<3>)],
        int_tuple![Int::<1>, (Int::<4>, Int::<12>)],
    );
    let dst_l = Layout::new(
        int_tuple![Int::<4>, Int::<6>],
        int_tuple![Int::<1>, Int::<4>],
    );
    let common = greatest_common_domain(src_l, dst_l);
    println!("src    = {src_l}");
    println!("dst    = {dst_l}");
    println!("common = {common}");
    println!("src / common = {}", logical_divide(src_l, common));
    println!("dst / common = {}", logical_divide(dst_l, common));

    println!("\n=== Stage 2: Nullspace Redundancy Elimination ===");
    let const_l = Layout::new(Int::<7>, Int::<0>);
    println!(
        "7:0 nullspace = {} (all 7 coordinates alias offset 0 -> 1 write)",
        nullspace(const_l)
    );

    println!("\n=== Stage 3: Destination-Aligned Vectorization Table ===");
    let scat_src = Layout::new(Int::<6>, Int::<1>);
    let scat_dst = Layout::new(
        int_tuple![Int::<2>, Int::<3>],
        int_tuple![Int::<4>, Int::<1>],
    );
    println!("scatter right_inverse(dst) = {}", right_inverse(scat_dst));

    show_stage3!(
        "memcpy",
        Layout::new(Int::<32>, Int::<1>),
        Layout::new(Int::<32>, Int::<1>)
    );
    show_stage3!(
        "col-major",
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );
    show_stage3!(
        "row-major",
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>]
        )
    );
    show_stage3!(
        "transpose",
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );
    show_stage3!(
        "broadcast",
        Layout::new(Int::<7>, Int::<0>),
        Layout::new(Int::<7>, Int::<1>)
    );
    show_stage3!(
        "constant",
        Layout::new(Int::<7>, Int::<0>),
        Layout::new(Int::<7>, Int::<0>)
    );
    show_stage3!(
        "part bcast",
        Layout::new(
            int_tuple![Int::<5>, Int::<4>],
            int_tuple![Int::<0>, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<5>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>]
        )
    );
    show_stage3!("hierarchical", src_l, dst_l);
    show_stage3!(
        "paper hier",
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<2>)],
            int_tuple![(Int::<8>, Int::<2>), (Int::<4>, Int::<1>)]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<2>)],
            int_tuple![(Int::<4>, Int::<2>), (Int::<8>, Int::<1>)]
        )
    );
    show_stage3!(
        "subdomain",
        Layout::new(
            int_tuple![Int::<4>, Int::<3>, Int::<5>],
            int_tuple![Int::<1>, Int::<7>, Int::<42>]
        ),
        Layout::new(
            int_tuple![Int::<6>, Int::<10>],
            int_tuple![Int::<1>, Int::<9>]
        )
    );
    show_stage3!(
        "coprime",
        Layout::new(
            int_tuple![Int::<5>, Int::<7>],
            int_tuple![Int::<7>, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<7>, Int::<5>],
            int_tuple![Int::<5>, Int::<1>]
        )
    );
    show_stage3!("scatter", scat_src, scat_dst);

    println!("\n=== End-to-End Verification ===");
    let src_buf = [10, 20, 30, 40, 50, 60];
    let mut ref_buf = [-1; 7];
    let mut opt_buf = [-1; 7];
    {
        let src = make_tensor(&src_buf, scat_src);
        let mut dst_ref = make_tensor(&mut ref_buf, scat_dst);
        let mut dst_opt = make_tensor(&mut opt_buf, scat_dst);
        copy_ref(src, &mut dst_ref);
        copy(src, &mut dst_opt);
    }
    assert_eq!(ref_buf, opt_buf);
    println!("scatter memory buffer = {opt_buf:?} (gap at slot 3 preserved)");
}
