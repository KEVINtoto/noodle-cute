include!("typed.rs");

macro_rules! instance {
    ($name:ident, $n:literal) => {
        #[inline(never)]
        fn $name() {
            typed_workload!(
                Layout::new(
                    shape![Int::<2>, (Int::<3>, Int::<$n>)],
                    stride![Int::<1>, (Int::<2>, Int::<6>)]
                )
                .unwrap(),
                $n,
                Int::<{ 12 * $n }>
            );
        }
    };
}
