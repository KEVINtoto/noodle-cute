use noodle_cute::{Int, algebra::TypedCompose, layout::Layout};

fn composition<L: TypedCompose<R>, R>() {}

fn main() {
    composition::<Layout<Int<4>, Int<1>>, ()>();
}
