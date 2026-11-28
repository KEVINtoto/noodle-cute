#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use noodle_cute::algorithm::atom::{
    CopyAtom, MmaAtom, UniversalCopy, UniversalFma, make_tiled_copy_c, make_tiled_mma,
};
use noodle_cute::int_tuple;
use noodle_cute::integer::Int;
use noodle_cute::layout::Layout;

fn main() {
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<f32>>::new(),
        Layout::from_shape(int_tuple![Int::<2>, Int::<2>]),
        ..,
    );
    let _ = make_tiled_copy_c(CopyAtom::<UniversalCopy<u128>, f32>::new(), &mma);
}
