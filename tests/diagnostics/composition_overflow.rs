use noodle_cute::{Int, StaticValue, algebra::TypedComposition, layout::Layout};

const C: TypedComposition<Layout<Int<2>, Int<{ isize::MAX }>>, Layout<Int<2>, Int<2>>> =
    StaticValue::VALUE;

fn main() {
    let _ = C;
}
