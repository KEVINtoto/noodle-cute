#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    const_trait_impl
)]
#![allow(incomplete_features)]

use noodle_cute::algebra::composition;
use noodle_cute::int_tuple;
use noodle_cute::integer::Int;
use noodle_cute::layout::Layout;

const _: () = {
    let a = Layout::new(int_tuple![Int::<3>, Int::<2>], int_tuple![Int::<1>, Int::<8>]);
    let b = Layout::new(int_tuple![Int::<2>, Int::<2>], int_tuple![Int::<1>, Int::<2>]);
    let _ = composition(a, b);
};

fn main() {}
