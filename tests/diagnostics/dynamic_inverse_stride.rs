use noodle_cute::algebra::InverseSide;
use noodle_cute::{DynInt, Int, StaticValue, algebra::TypedInverse, layout::Layout};

const C: TypedInverse<Layout<Int<4>, DynInt>, { InverseSide::Left }> = StaticValue::VALUE;

fn main() {
    let _ = C;
}
