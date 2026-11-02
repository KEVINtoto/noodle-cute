use noodle_cute::{Int, StaticValue, algebra::TypedComplement, layout::Layout};

const C: TypedComplement<Layout<Int<0>, Int<1>>, Int<8>> = StaticValue::VALUE;

fn main() {
    let _ = C;
}
