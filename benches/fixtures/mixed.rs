#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    const_trait_impl
)]
#![allow(incomplete_features)]

include!("typed.rs");

#[inline(never)]
fn instance(n: i32) {
    typed_workload!(
        Layout::new(
            int_tuple![Int::<2>, (Int::<3>, black_box(n))],
            int_tuple![Int::<1>, (Int::<2>, Int::<6>)]
        ),
        n,
        black_box(12 * n)
    );
}
