use core::hint::black_box;

use noodle_cute::layout::Layout;
use noodle_cute::{Int, shape, stride};

macro_rules! typed_workload {
    ($layout:expr, $n:expr, $cotarget:expr) => {{
        let layout = black_box($layout);
        let size = 6 * $n;
        let last = black_box(size - 1);
        let coalesced = layout.coalesce().unwrap();
        let composed = layout
            .composition(Layout::new(Int::<3>, Int::<2>).unwrap())
            .unwrap();
        let complement = layout.complement($cotarget).unwrap();
        let divided = layout.logical_divide(Int::<3>).unwrap();
        let product = layout.logical_product(Int::<2>).unwrap();
        let inverse = layout.right_inverse().unwrap();
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
    }};
}
