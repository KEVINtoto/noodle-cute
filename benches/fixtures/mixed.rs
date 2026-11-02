include!("typed.rs");

use noodle_cute::DynInt;

#[inline(never)]
fn instance(n: isize) {
    typed_workload!(
        Layout::new(
            shape![Int::<2>, (Int::<3>, DynInt(black_box(n)))],
            stride![Int::<1>, (Int::<2>, Int::<6>)]
        )
        .unwrap(),
        n,
        DynInt(black_box(12 * n))
    );
}
