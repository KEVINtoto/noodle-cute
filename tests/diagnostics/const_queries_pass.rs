#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use noodle_cute::algebra::composition;
use noodle_cute::algorithm::atom::{
    CopyAtom, MmaAtom, UniversalCopy, UniversalFma, make_tiled_copy_c, make_tiled_mma,
};
use noodle_cute::int_tuple;
use noodle_cute::integer::Int;
use noodle_cute::layout::Layout;
use noodle_cute::tensor::make_tensor;

const VAL: i32 = {
    let l = Layout::from_shape(int_tuple![Int::<4>, Int::<4>]);
    let c = composition(l, Layout::new(Int::<8>, Int::<2>));
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<i32>>::new(),
        Layout::from_shape(int_tuple![Int::<2>, Int::<2>]),
        ..,
    );
    let tc = make_tiled_copy_c(CopyAtom::<UniversalCopy<i32>, i32>::new(), &mma);
    let data = [0i32; 16];
    let t = make_tensor(&data, l);
    let p = tc.get_slice(0i32).partition_s(t);
    c.size() + p.at(int_tuple![0i32, 0i32, 0i32])
};

fn main() {
    let _ = VAL;
}
