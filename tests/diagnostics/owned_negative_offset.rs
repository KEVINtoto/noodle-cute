#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use noodle_cute::integer::Int;
use noodle_cute::layout::Layout;
use noodle_cute::tensor::make_owned_tensor;

fn main() {
    let _ = make_owned_tensor::<i32, _>(Layout::new(Int::<4>, Int::<-1>));
}
