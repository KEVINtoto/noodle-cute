use noodle_cute::{Int, algebra::TypedInverse, layout::Layout};

type C = TypedInverse<Layout<Int<4>, Int<1>>, true>;

fn main() {
    let _: Option<C> = None;
}
