#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    const_trait_impl
)]
#![allow(incomplete_features)]

include!("typed.rs");

macro_rules! instance {
    ($name:ident, $n:literal) => {
        #[inline(never)]
        fn $name() {
            typed_workload!(
                Layout::new(
                    int_tuple![Int::<2>, (Int::<3>, Int::<$n>)],
                    int_tuple![Int::<1>, (Int::<2>, Int::<6>)]
                ),
                $n,
                Int::<{ 12 * $n }>
            );
        }
    };
}
