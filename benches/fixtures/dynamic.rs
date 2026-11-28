#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    const_trait_impl
)]
#![allow(incomplete_features)]

use core::hint::black_box;

use noodle_cute::algebra::{coalesce, composition};
use noodle_cute::int_tuple;
use noodle_cute::layout::Layout;

#[inline(never)]
fn instance(n: i32) {
    let layout = black_box(Layout::new(
        int_tuple![black_box(2i32), (black_box(3i32), black_box(n))],
        int_tuple![black_box(1i32), (black_box(2i32), black_box(6i32))],
    ));
    let size = 6 * n;
    let last = black_box(size - 1);
    let coalesced = coalesce(layout);
    let flat = black_box(Layout::new(size, black_box(1i32)));
    let composed = composition(flat, Layout::new(black_box(3i32), black_box(2i32)));
    let actual = black_box([
        layout.size(),
        layout.cosize(),
        layout(last),
        coalesced.size(),
        composed(black_box(2i32)),
    ]);
    assert_eq!(actual, [size, size, size - 1, size, 4]);
}
