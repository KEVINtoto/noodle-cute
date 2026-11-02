use noodle_cute::{DynInt, Int, layout::Layout, shape, stride};

fn main() {
    let layout = Layout::new(shape![Int::<2>, Int::<2>], stride![DynInt(1), DynInt(6)]).unwrap();
    let _ = layout.complement(Int::<24>);
}
