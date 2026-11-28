#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use noodle_cute::algebra::{composition, logical_divide};
use noodle_cute::algorithm::atom::{
    CopyAtom, MmaAtom, UniversalCopy, UniversalFma, make_tiled_copy, make_tiled_copy_c,
    make_tiled_mma,
};
use noodle_cute::algorithm::{copy_with, gemm_fma};
use noodle_cute::int_tuple;
use noodle_cute::integer::Int;
use noodle_cute::layout::Layout;
use noodle_cute::tensor::{make_owned_tensor, make_tensor};

fn main() {
    let l = Layout::from_shape(int_tuple![Int::<4>, Int::<4>]);
    let _ = composition(l, Layout::new(Int::<8>, Int::<2>));
    let _ = logical_divide(l, int_tuple![Int::<2>, Int::<2>]);

    let mut owned = make_owned_tensor::<f32, _>(l);
    let data = [1.0f32; 16];
    let src = make_tensor(&data, l);

    let tiled_copy = make_tiled_copy(
        CopyAtom::<UniversalCopy<f32>, f32>::new(),
        Layout::from_shape(int_tuple![Int::<2>, Int::<2>]),
        Layout::from_shape(int_tuple![Int::<2>, Int::<2>]),
    );
    let mut dst_view = owned.view_mut();
    let s0 = tiled_copy.get_slice(0i32).partition_s(src);
    let mut d0 = tiled_copy.get_slice(0i32).partition_d(dst_view);
    copy_with(&tiled_copy, &s0, &mut d0);

    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<f32>>::new(),
        Layout::from_shape(int_tuple![Int::<2>, Int::<2>]),
        ..,
    );
    let _ = make_tiled_copy_c(CopyAtom::<UniversalCopy<f32>, f32>::new(), &mma);

    let a = make_owned_tensor::<f32, _>(int_tuple![Int::<4>, Int::<4>]);
    let b = make_owned_tensor::<f32, _>(int_tuple![Int::<4>, Int::<4>]);
    gemm_fma(&a, &b, &mut dst_view);
}
