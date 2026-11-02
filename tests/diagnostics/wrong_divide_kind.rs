use noodle_cute::{
    Int,
    algebra::{ProductKind, TypedDivided},
    layout::Layout,
};

type C = TypedDivided<Layout<Int<8>, Int<1>>, Int<2>, { ProductKind::Blocked }>;

fn main() {
    let _: Option<C> = None;
}
