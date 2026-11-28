#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use noodle_cute::int_tuple;
use noodle_cute::integer::Int;
use noodle_cute::layout::Layout;

fn main() {
    let _ = Layout::new(int_tuple![Int::<2>, Int::<3>], Int::<1>);
}
