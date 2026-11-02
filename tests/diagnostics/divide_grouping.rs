use noodle_cute::{Int, algebra::TypedDivide, layout::Layout};

fn divide<L: TypedDivide<Int<2>, 7>>() {}

fn main() {
    divide::<Layout<Int<8>, Int<1>>>();
}
