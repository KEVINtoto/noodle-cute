use noodle_cute::{Int, int_tuple, layout::Layout};

fn main() {
    let block = Layout::new(Int::<2>, Int::<1>).unwrap();
    let _ = block.blocked_product(int_tuple![Int::<3>]);
}
