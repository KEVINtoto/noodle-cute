#![feature(const_trait_impl)]

use noodle_cute::{DynInt, Int, layout::Layout};

const L: Layout<Int<2>, DynInt> = Layout {
    shape: Int,
    stride: DynInt(isize::MAX),
};
const STORAGE: usize = match L.cosize() {
    Ok(n) => n as usize,
    Err(_) => panic!("layout storage span overflows"),
};

fn main() {
    let _ = STORAGE;
}
