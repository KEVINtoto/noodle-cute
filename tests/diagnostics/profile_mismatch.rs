use noodle_cute::{Int, IntTuple, Nil, StaticValue, algebra::TypedComposition, layout::Layout};

const C: TypedComposition<Layout<Int<4>, Int<1>>, Layout<IntTuple<Int<2>, Nil>, Int<1>>> =
    StaticValue::VALUE;

fn main() {
    let _ = C;
}
