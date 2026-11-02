use noodle_cute::{DynInt, Int, StaticValue, algebra::TypedComposition, layout::Layout};

const C: TypedComposition<Layout<Int<8>, Int<1>>, Layout<DynInt, Int<1>>> = StaticValue::VALUE;

fn main() {
    let _ = C;
}
