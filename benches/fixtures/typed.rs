use core::hint::black_box;

use noodle_cute::algebra::{
    coalesce, complement_with, composition, logical_divide, logical_product, right_inverse,
};
use noodle_cute::int_tuple;
use noodle_cute::integer::Int;
use noodle_cute::layout::Layout;

macro_rules! typed_workload {
    ($layout:expr, $n:expr, $cotarget:expr) => {{
        let layout = black_box($layout);
        let size: i32 = 6 * ($n as i32);
        let last = black_box(size - 1);
        let coalesced = coalesce(layout);
        let composed = composition(layout, Layout::new(Int::<3>, Int::<2>));
        let complement = complement_with(layout, $cotarget);
        let divided = logical_divide(layout, Int::<3>);
        let product = logical_product(layout, Int::<2>);
        let inverse = right_inverse(layout);
        let actual = black_box([
            layout.size(),
            layout.cosize(),
            layout(last),
            coalesced.size(),
            composed(black_box(2i32)),
            complement(black_box(1i32)),
            divided(last),
            product(black_box(2 * size - 1)),
            inverse(last),
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
                size - 1,
            ]
        );
    }};
}
