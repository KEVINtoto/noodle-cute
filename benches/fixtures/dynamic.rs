use core::hint::black_box;

use noodle_cute::algebra::{
    complement, composition_layout, logical_divide, logical_product, right_inverse,
};
use noodle_cute::layout::{DynLayout, LayoutMap};

#[inline(never)]
fn instance(n: isize) {
    let layout = DynLayout::new(black_box(&[2, 3, n]), black_box(&[1, 2, 6])).unwrap();
    let size = 6 * n;
    let last = black_box(size - 1);
    let coalesced = layout.coalesce().unwrap();
    let composed = composition_layout(&layout, &DynLayout::new(&[3], &[2]).unwrap()).unwrap();
    let complement = complement(layout, black_box(12 * n)).unwrap();
    let divided = logical_divide(layout, DynLayout::new(&[3], &[1]).unwrap()).unwrap();
    let product = logical_product(layout, DynLayout::new(&[2], &[1]).unwrap()).unwrap();
    let inverse = right_inverse(layout).unwrap();
    let actual = black_box([
        layout.size().unwrap(),
        layout.cosize().unwrap(),
        layout.index(last).unwrap(),
        coalesced.size().unwrap(),
        composed.index(black_box(2)).unwrap(),
        complement.index(black_box(1)).unwrap(),
        divided.index(last).unwrap(),
        product.index(black_box(2 * size - 1)).unwrap(),
        inverse.index(last).unwrap(),
    ]);
    assert_eq!(
        actual,
        [
            size,
            size,
            size - 1,
            size,
            4,
            size,
            size - 1,
            2 * size - 1,
            size - 1
        ]
    );
}
