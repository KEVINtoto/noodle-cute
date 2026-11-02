#![feature(min_adt_const_params)]

use noodle_cute::algebra::{Grouping, ProductKind, TypedDivide, TypedProduct};
use noodle_cute::{Int, Result, layout::Layout};

fn divide<const G: Grouping, L: TypedDivide<T, G>, T>(layout: L, tiler: T) -> Result<L::Output> {
    layout.divide_typed(tiler)
}

fn product<const K: ProductKind, L: TypedProduct<T, K>, T>(
    layout: L,
    tiler: T,
) -> Result<L::Output> {
    layout.product_typed(tiler)
}

fn main() {
    let layout = Layout::new(Int::<8>, Int::<1>).unwrap();
    let _ = divide::<{ Grouping::Zipped }, _, _>(layout, Int::<2>).unwrap();
    let _ = product::<{ ProductKind::Flat }, _, _>(layout, Int::<2>).unwrap();
}
