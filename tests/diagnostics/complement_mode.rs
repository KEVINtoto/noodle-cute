use noodle_cute::{Int, Nil, algebra::TypedComplement, layout::Layout};

type C = TypedComplement<Layout<Int<4>, Int<2>>, Nil, true>;

fn main() {
    let _: Option<C> = None;
}
