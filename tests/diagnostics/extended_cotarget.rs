use noodle_cute::{
    Int, StaticValue,
    algebra::{ComplementMode, TypedComplement},
    layout::Layout,
};

const C: TypedComplement<Layout<Int<4>, Int<2>>, Int<16>, { ComplementMode::Extended }> =
    StaticValue::VALUE;

fn main() {
    let _ = C;
}
