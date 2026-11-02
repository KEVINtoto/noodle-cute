use noodle_cute::{Int, algebra::TypedProduct, layout::Layout};

fn product<L: TypedProduct<()>>() {}

fn main() {
    product::<Layout<Int<2>, Int<1>>>();
}
