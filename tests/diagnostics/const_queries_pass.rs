#![feature(const_trait_impl)]

use noodle_cute::layout::{Layout, Shape, StrideOf};
use noodle_cute::tuple::Item;
use noodle_cute::{DynInt, Int, IntTuple, Nil, Result, shape, stride};

type Pair<A, B> = IntTuple<A, IntTuple<B, Nil>>;
const L: Layout<Pair<Int<2>, DynInt>, Pair<Int<1>, Int<2>>> = Layout {
    shape: shape![Int::<2>, DynInt(3)],
    stride: stride![Int::<1>, Int::<2>],
};

const fn elements<S: Shape + [const] Item, D: [const] StrideOf<S>>(
    layout: &Layout<S, D>,
) -> Result<isize> {
    layout.size()
}

const SIZE: isize = match elements(&L) {
    Ok(n) => n,
    Err(_) => panic!("invalid layout"),
};
const STORAGE: usize = match L.cosize() {
    Ok(n) => n as usize,
    Err(_) => panic!("invalid storage span"),
};
const LAST: isize = match L.index(5) {
    Ok(n) => n,
    Err(_) => panic!("invalid coordinate"),
};

fn main() {
    let buffer: [u8; STORAGE] = [0; STORAGE];
    assert_eq!(SIZE, 6);
    assert_eq!(buffer.len(), 6);
    assert_eq!(LAST, 5);
}
