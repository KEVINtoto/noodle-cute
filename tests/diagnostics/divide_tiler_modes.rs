use noodle_cute::{Int, int_tuple, layout::Layout};

fn main() {
    let layout = Layout::new(Int::<8>, Int::<1>).unwrap();
    let _ = layout.logical_divide(int_tuple![Int::<2>, Int::<2>]);
}
