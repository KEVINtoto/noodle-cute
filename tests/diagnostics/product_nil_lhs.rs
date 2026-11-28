#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use noodle_cute::algebra::logical_product;
use noodle_cute::integer::Int;
use noodle_cute::layout::Layout;
use noodle_cute::tuple::Nil;

fn main() {
    let _ = logical_product(Nil, Layout::new(Int::<2>, Int::<1>));
}
