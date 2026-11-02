use noodle_cute::{Int, StaticValue, layout::Layout};

const L: Layout<Int<2>, Int<1>> = StaticValue::VALUE;
const SIZE: noodle_cute::Result<isize> = L.size();

fn main() {
    let _ = SIZE;
}
