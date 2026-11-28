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

fn main() {
    let _ = Layout::new((), Int::<1>);
}
