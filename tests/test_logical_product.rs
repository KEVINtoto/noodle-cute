#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl,
    const_cmp
)]
#![allow(incomplete_features)]

use noodle_cute::algebra::complement::complement;
use noodle_cute::algebra::composition::composition;
use noodle_cute::algebra::logical_product::{
    blocked_product, logical_product, logical_product_at, raked_product,
};
use noodle_cute::integer::{Int, int};
use noodle_cute::layout::{Layout, tiler_to_layout};
use noodle_cute::shape::compatible;
use noodle_cute::tuple::{Nil, Til};
use noodle_cute::{e, int_tuple, make_layout, mode, tuple};

macro_rules! postcondition_logical_product {
    ($a:expr, $b:expr $(,)?) => {{
        let a = $a;
        let b = $b;
        let r = logical_product(a, b);

        assert_eq!(r.rank(), 2);
        assert_eq!(r.size(), a.size() * b.size());
        assert_eq!(r.get::<0>(), a);
        assert!(compatible(&b.shape(), &r.get::<1>().shape()));

        r
    }};
}

macro_rules! postcondition_blocked_product {
    ($tile:expr, $grid:expr $(,)?) => {{
        let tile = $tile;
        let grid = $grid;
        let r = blocked_product(tile, grid);

        assert_eq!(r.rank(), tile.rank());
        assert_eq!(r.rank(), grid.rank());
        assert_eq!(r.size(), tile.size() * grid.size());

        r
    }};
}

macro_rules! postcondition_raked_product {
    ($tile:expr, $grid:expr $(,)?) => {{
        let tile = $tile;
        let grid = $grid;
        let r = raked_product(tile, grid);

        assert_eq!(r.rank(), tile.rank());
        assert_eq!(r.rank(), grid.rank());
        assert_eq!(r.size(), tile.size() * grid.size());

        r
    }};
}

#[test]
fn test_logical_product() {
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(Int::<1>, Int::<0>),
            Layout::new(Int::<1>, Int::<0>),
        ),
        Layout::new(
            int_tuple![Int::<1>, Int::<1>],
            int_tuple![Int::<0>, Int::<0>],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(Int::<1>, Int::<1>),
            Layout::new(Int::<1>, Int::<0>),
        ),
        Layout::new(
            int_tuple![Int::<1>, Int::<1>],
            int_tuple![Int::<1>, Int::<0>],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(Int::<1>, Int::<0>),
            Layout::new(Int::<1>, Int::<1>),
        ),
        Layout::new(
            int_tuple![Int::<1>, Int::<1>],
            int_tuple![Int::<0>, Int::<1>],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(Int::<1>, Int::<1>),
            Layout::new(Int::<1>, Int::<1>),
        ),
        Layout::new(
            int_tuple![Int::<1>, Int::<1>],
            int_tuple![Int::<1>, Int::<1>],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(Int::<3>, Int::<1>),
            Layout::new(Int::<4>, Int::<1>),
        ),
        Layout::new(
            int_tuple![Int::<3>, Int::<4>],
            int_tuple![Int::<1>, Int::<3>],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(Int::<3>, Int::<1>),
            Layout::new(Int::<4>, Int::<0>),
        ),
        Layout::new(
            int_tuple![Int::<3>, Int::<4>],
            int_tuple![Int::<1>, Int::<0>],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(Int::<3>, Int::<1>),
            Layout::new(
                int_tuple![Int::<2>, Int::<4>],
                int_tuple![Int::<1>, Int::<2>],
            ),
        ),
        Layout::new(
            int_tuple![Int::<3>, (Int::<2>, Int::<4>)],
            int_tuple![Int::<1>, (Int::<3>, Int::<6>)],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(
                int_tuple![Int::<2>, Int::<4>],
                int_tuple![Int::<1>, Int::<2>],
            ),
            Layout::new(Int::<3>, Int::<1>),
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<4>), Int::<3>],
            int_tuple![(Int::<1>, Int::<2>), Int::<8>],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::from_shape(int_tuple![Int::<8>, (Int::<2>, Int::<2>)]),
            Layout::new(Int::<4>, Int::<2>),
        ),
        Layout::new(
            int_tuple![(Int::<8>, (Int::<2>, Int::<2>)), Int::<4>],
            int_tuple![(Int::<1>, (Int::<8>, Int::<16>)), Int::<64>],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::from_shape(int_tuple![Int::<2>, Int::<2>]),
            Layout::new(
                int_tuple![Int::<3>, Int::<3>],
                int_tuple![Int::<3>, Int::<1>],
            ),
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), (Int::<3>, Int::<3>)],
            int_tuple![(Int::<1>, Int::<2>), (Int::<12>, Int::<4>)],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(Int::<3>, Int::<32>),
            Layout::new(Int::<32>, Int::<1>),
        ),
        Layout::new(
            int_tuple![Int::<3>, Int::<32>],
            int_tuple![Int::<32>, Int::<1>],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(Int::<3>, Int::<2>),
            Layout::new(Int::<4>, Int::<1>),
        ),
        Layout::new(
            int_tuple![Int::<3>, (Int::<2>, Int::<2>)],
            int_tuple![Int::<2>, (Int::<1>, Int::<6>)],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(Int::<3>, Int::<32>),
            Layout::new(Int::<128>, Int::<1>),
        ),
        Layout::new(
            int_tuple![Int::<3>, (Int::<32>, Int::<4>)],
            int_tuple![Int::<32>, (Int::<1>, Int::<96>)],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(Int::<3>, Int::<32>),
            Layout::from_shape(int_tuple![Int::<8>, Int::<8>]),
        ),
        Layout::new(
            int_tuple![Int::<3>, (Int::<8>, (Int::<4>, Int::<2>))],
            int_tuple![Int::<32>, (Int::<1>, (Int::<8>, Int::<96>))],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(Int::<3>, Int::<32>),
            Layout::new(
                int_tuple![Int::<8>, Int::<8>],
                int_tuple![Int::<8>, Int::<1>],
            ),
        ),
        Layout::new(
            int_tuple![Int::<3>, ((Int::<4>, Int::<2>), Int::<8>)],
            int_tuple![Int::<32>, ((Int::<8>, Int::<96>), Int::<1>)],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(
                int_tuple![Int::<4>, Int::<2>],
                int_tuple![Int::<1>, Int::<16>],
            ),
            Layout::from_shape(int_tuple![Int::<4>, Int::<4>]),
        ),
        Layout::new(
            int_tuple![(Int::<4>, Int::<2>), (Int::<4>, Int::<4>)],
            int_tuple![(Int::<1>, Int::<16>), (Int::<4>, Int::<32>)],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(
                int_tuple![Int::<4>, Int::<2>],
                int_tuple![Int::<1>, Int::<16>],
            ),
            Layout::new(
                int_tuple![Int::<4>, Int::<2>],
                int_tuple![Int::<2>, Int::<1>],
            ),
        ),
        Layout::new(
            int_tuple![(Int::<4>, Int::<2>), ((Int::<2>, Int::<2>), Int::<2>)],
            int_tuple![(Int::<1>, Int::<16>), ((Int::<8>, Int::<32>), Int::<4>)],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(
                int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<2>)],
                int_tuple![(Int::<1>, Int::<4>), (Int::<8>, Int::<32>)],
            ),
            Layout::new(
                int_tuple![Int::<2>, Int::<2>],
                int_tuple![Int::<1>, Int::<2>],
            ),
        ),
        Layout::new(
            int_tuple![
                ((Int::<2>, Int::<2>), (Int::<2>, Int::<2>)),
                (Int::<2>, Int::<2>)
            ],
            int_tuple![
                ((Int::<1>, Int::<4>), (Int::<8>, Int::<32>)),
                (Int::<2>, Int::<16>)
            ],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(
                int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<2>)],
                int_tuple![(Int::<1>, Int::<4>), (Int::<8>, Int::<32>)],
            ),
            Layout::new(
                int_tuple![Int::<2>, Int::<2>],
                int_tuple![Int::<2>, Int::<1>],
            ),
        ),
        Layout::new(
            int_tuple![
                ((Int::<2>, Int::<2>), (Int::<2>, Int::<2>)),
                (Int::<2>, Int::<2>)
            ],
            int_tuple![
                ((Int::<1>, Int::<4>), (Int::<8>, Int::<32>)),
                (Int::<16>, Int::<2>)
            ],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(
                int_tuple![Int::<4>, Int::<6>],
                int_tuple![Int::<1>, Int::<6>],
            ),
            Layout::new(Int::<3>, Int::<1>),
        ),
        Layout::new(
            int_tuple![(Int::<4>, Int::<6>), Int::<3>],
            int_tuple![(Int::<1>, Int::<6>), Int::<36>],
        )
    );
    // Docstring worked example: Layout((2, 2), (4, 1)) x Layout(6, 1)
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(
                int_tuple![Int::<2>, Int::<2>],
                int_tuple![Int::<4>, Int::<1>],
            ),
            Layout::new(Int::<6>, Int::<1>),
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<3>)],
            int_tuple![(Int::<4>, Int::<1>), (Int::<2>, Int::<8>)],
        )
    );
}

#[test]
fn test_logical_product_tiler() {
    let a = Layout::new(
        int_tuple![Int::<6>, Int::<4>],
        int_tuple![Int::<4>, Int::<1>],
    );
    let tiler = tuple![
        Layout::new(Int::<2>, Int::<1>),
        Layout::new(Int::<2>, Int::<1>),
    ];
    let expected = make_layout!(
        logical_product(a.get::<0>(), tiler.get::<0>()),
        logical_product(a.get::<1>(), tiler.get::<1>())
    );
    assert_eq!(logical_product(a, tiler), expected);

    // Scalar A with 1-tuple tiler wraps A to rank 1
    assert_eq!(
        logical_product(
            Layout::new(Int::<3>, Int::<1>),
            tuple![Layout::new(Int::<2>, Int::<1>)],
        ),
        Layout::new(
            int_tuple![(Int::<3>, Int::<2>)],
            int_tuple![(Int::<1>, Int::<3>)],
        )
    );

    // Nested by-mode tiler
    let nested_a = Layout::new(
        int_tuple![(Int::<3>, Int::<5>), Int::<7>],
        int_tuple![(Int::<1>, Int::<3>), Int::<15>],
    );
    let nested_b = tuple![
        (Layout::new(Int::<2>, Int::<1>), Nil),
        Layout::new(Int::<4>, Int::<1>),
    ];
    assert_eq!(
        logical_product(nested_a, nested_b),
        Layout::new(
            int_tuple![((Int::<3>, Int::<2>), Int::<5>), (Int::<7>, Int::<4>)],
            int_tuple![((Int::<1>, Int::<3>), Int::<3>), (Int::<15>, Int::<1>)],
        )
    );
}

#[test]
fn test_logical_product_mode() {
    let a = Layout::new(
        int_tuple![Int::<3>, Int::<5>],
        int_tuple![Int::<1>, Int::<20>],
    );
    let b = Layout::new(Int::<4>, Int::<1>);

    assert_eq!(
        logical_product_at::<mode![0], _, _>(a, b),
        make_layout!(logical_product(a.get::<0>(), b), a.get::<1>())
    );
    assert_eq!(
        logical_product_at::<mode![1], _, _>(a, b),
        make_layout!(a.get::<0>(), logical_product(a.get::<1>(), b))
    );
    assert_eq!(
        logical_product_at::<mode![0], _, _>(a, b),
        logical_product(a, tuple![b])
    );
    assert_eq!(
        logical_product_at::<mode![], _, _>(a, b),
        logical_product(a, b)
    );

    // Negative indices count from the end
    assert_eq!(
        logical_product_at::<mode![-1], _, _>(a, b),
        logical_product_at::<mode![1], _, _>(a, b)
    );
    assert_eq!(
        logical_product_at::<mode![-2], _, _>(a, b),
        logical_product_at::<mode![0], _, _>(a, b)
    );
    assert_eq!(
        a.logical_product_at::<mode![0], _>(b),
        Layout::new(
            int_tuple![(Int::<3>, Int::<4>), Int::<5>],
            int_tuple![(Int::<1>, Int::<3>), Int::<20>],
        )
    );

    // Post-conditions hold of the reproduced mode
    let r = logical_product_at::<mode![0], _, _>(a, b);
    assert_eq!(r.get::<0>().rank(), 2);
    assert_eq!(r.get::<0>().get::<0>(), a.get::<0>());
    assert!(compatible(&b.shape(), &r.get::<0>().get::<1>().shape()));

    // Nested mode path (positive and negative)
    let nested = Layout::new(
        int_tuple![(Int::<3>, Int::<5>), Int::<7>],
        int_tuple![(Int::<1>, Int::<3>), Int::<15>],
    );
    let b2 = Layout::new(Int::<2>, Int::<1>);
    assert_eq!(
        logical_product_at::<mode![0, 1], _, _>(nested, b2),
        Layout::new(
            int_tuple![(Int::<3>, (Int::<5>, Int::<2>)), Int::<7>],
            int_tuple![(Int::<1>, (Int::<3>, Int::<1>)), Int::<15>],
        )
    );
    assert_eq!(
        logical_product_at::<mode![-2, -1], _, _>(nested, b2),
        logical_product_at::<mode![0, 1], _, _>(nested, b2)
    );

    // Scalar LHS with mode![0] wraps to rank-1
    assert_eq!(
        logical_product_at::<mode![0], _, _>(Layout::new(Int::<3>, Int::<1>), b),
        Layout::new(
            int_tuple![(Int::<3>, Int::<4>)],
            int_tuple![(Int::<1>, Int::<3>)],
        )
    );

    // Tiler LHS with mode path
    assert_eq!(
        logical_product_at::<mode![1], _, _>(tuple![Int::<3>, Int::<5>], b),
        logical_product_at::<mode![1], _, _>(tiler_to_layout(tuple![Int::<3>, Int::<5>]), b)
    );
}

#[test]
fn test_logical_product_types() {
    let a = Layout::new(
        int_tuple![Int::<6>, Int::<4>],
        int_tuple![Int::<4>, Int::<1>],
    );

    // A promoted from scalar / tuple tiler
    assert_eq!(
        logical_product(Int::<6>, Layout::new(Int::<2>, Int::<1>)),
        logical_product(
            Layout::new(Int::<6>, Int::<1>),
            Layout::new(Int::<2>, Int::<1>),
        )
    );
    assert_eq!(
        logical_product(tuple![Int::<2>, Int::<3>], Layout::new(Int::<2>, Int::<1>)),
        logical_product(
            tiler_to_layout(tuple![Int::<2>, Int::<3>]),
            Layout::new(Int::<2>, Int::<1>),
        )
    );

    // B promotions: Nil is a no-op; scalar promotes to N:1
    assert_eq!(logical_product(a, Nil), a);
    assert_eq!(
        logical_product(a, Int::<2>),
        logical_product(a, Layout::new(Int::<2>, Int::<1>))
    );
    assert_eq!(
        logical_product(a, 2i32),
        logical_product(a, Layout::new(2i32, Int::<1>))
    );

    // Empty Layout(Nil, Nil)
    let empty = Layout::new(Nil, Nil);
    assert_eq!(logical_product(empty, Nil), empty);
    assert_eq!(
        logical_product(empty, Int::<2>),
        Layout::new(tuple![Nil, Int::<2>], tuple![Nil, Int::<1>])
    );
    assert_eq!(
        logical_product(empty, Layout::new(Int::<2>, Int::<1>)),
        Layout::new(tuple![Nil, Int::<2>], tuple![Nil, Int::<1>])
    );
    assert_eq!(
        logical_product(empty, empty),
        Layout::new(tuple![Nil, Nil], tuple![Nil, Nil])
    );
}

#[test]
fn test_logical_product_arg_matrix() {
    // A = Int<24>
    assert_eq!(
        logical_product(Int::<24>, Nil),
        Layout::new(Int::<24>, Int::<1>)
    );
    assert_eq!(
        logical_product(Int::<24>, Int::<2>),
        Layout::new(
            int_tuple![Int::<24>, Int::<2>],
            int_tuple![Int::<1>, Int::<24>],
        )
    );
    assert_eq!(
        logical_product(Int::<24>, Layout::new(Int::<2>, Int::<1>)),
        Layout::new(
            int_tuple![Int::<24>, Int::<2>],
            int_tuple![Int::<1>, Int::<24>],
        )
    );

    // A = (6, 4)
    let t_a = tuple![Int::<6>, Int::<4>];
    assert_eq!(
        logical_product(t_a, Nil),
        Layout::new(int_tuple![Int::<6>, Int::<4>], tuple![e!(0), e!(1)])
    );
    assert_eq!(
        logical_product(t_a, Int::<2>),
        Layout::new(
            int_tuple![(Int::<6>, Int::<4>), Int::<2>],
            tuple![(e!(0), e!(1)), Int::<4> * e!(1)],
        )
    );
    assert_eq!(
        logical_product(t_a, tuple![Int::<2>, Int::<2>]),
        Layout::new(
            int_tuple![(Int::<6>, Int::<2>), (Int::<4>, Int::<2>)],
            tuple![(e!(0), Int::<6> * e!(0)), (e!(1), Int::<4> * e!(1))],
        )
    );
    assert_eq!(
        logical_product(t_a, Layout::new(Int::<2>, Int::<1>)),
        Layout::new(
            int_tuple![(Int::<6>, Int::<4>), Int::<2>],
            tuple![(e!(0), e!(1)), Int::<4> * e!(1)],
        )
    );
    assert_eq!(
        logical_product(
            t_a,
            tuple![
                Layout::new(Int::<2>, Int::<1>),
                Layout::new(Int::<2>, Int::<2>),
            ],
        ),
        Layout::new(
            int_tuple![(Int::<6>, Int::<2>), (Int::<4>, Int::<2>)],
            tuple![(e!(0), Int::<6> * e!(0)), (e!(1), Int::<8> * e!(1))],
        )
    );

    // A = Layout((6, 4), (4, 1))
    let l_a = Layout::new(
        int_tuple![Int::<6>, Int::<4>],
        int_tuple![Int::<4>, Int::<1>],
    );
    assert_eq!(logical_product(l_a, Nil), l_a);
    assert_eq!(
        logical_product(l_a, Int::<2>),
        Layout::new(
            int_tuple![(Int::<6>, Int::<4>), Int::<2>],
            int_tuple![(Int::<4>, Int::<1>), Int::<24>],
        )
    );
    assert_eq!(
        logical_product(l_a, tuple![Int::<2>, Int::<2>]),
        Layout::new(
            int_tuple![(Int::<6>, Int::<2>), (Int::<4>, Int::<2>)],
            int_tuple![(Int::<4>, Int::<1>), (Int::<1>, Int::<4>)],
        )
    );
    assert_eq!(
        logical_product(l_a, Layout::new(Int::<2>, Int::<1>)),
        Layout::new(
            int_tuple![(Int::<6>, Int::<4>), Int::<2>],
            int_tuple![(Int::<4>, Int::<1>), Int::<24>],
        )
    );
    assert_eq!(
        logical_product(
            l_a,
            tuple![
                Layout::new(Int::<2>, Int::<1>),
                Layout::new(Int::<2>, Int::<2>),
            ],
        ),
        Layout::new(
            int_tuple![(Int::<6>, Int::<2>), (Int::<4>, Int::<2>)],
            int_tuple![(Int::<4>, Int::<1>), (Int::<1>, Int::<8>)],
        )
    );

    // A = (Layout(6, 2), Layout(4, 1))
    let tl_a = tuple![
        Layout::new(Int::<6>, Int::<2>),
        Layout::new(Int::<4>, Int::<1>),
    ];
    assert_eq!(
        logical_product(tl_a, Nil),
        Layout::new(
            int_tuple![Int::<6>, Int::<4>],
            tuple![Int::<2> * e!(0), e!(1)],
        )
    );
    assert_eq!(
        logical_product(tl_a, Int::<2>),
        Layout::new(
            int_tuple![(Int::<6>, Int::<4>), Int::<2>],
            tuple![(Int::<2> * e!(0), e!(1)), e!(0)],
        )
    );
    assert_eq!(
        logical_product(tl_a, tuple![Int::<2>, Int::<2>]),
        Layout::new(
            int_tuple![(Int::<6>, Int::<2>), (Int::<4>, Int::<2>)],
            tuple![(Int::<2> * e!(0), e!(0)), (e!(1), Int::<4> * e!(1))],
        )
    );
    assert_eq!(
        logical_product(tl_a, Layout::new(Int::<2>, Int::<1>)),
        Layout::new(
            int_tuple![(Int::<6>, Int::<4>), Int::<2>],
            tuple![(Int::<2> * e!(0), e!(1)), e!(0)],
        )
    );
    assert_eq!(
        logical_product(
            tl_a,
            tuple![
                Layout::new(Int::<2>, Int::<1>),
                Layout::new(Int::<2>, Int::<2>),
            ],
        ),
        Layout::new(
            int_tuple![(Int::<6>, Int::<2>), (Int::<4>, Int::<2>)],
            tuple![(Int::<2> * e!(0), e!(0)), (e!(1), Int::<8> * e!(1))],
        )
    );
}

#[test]
fn test_logical_product_promotes_tiler_a() {
    macro_rules! check_promo {
        ($a:expr) => {{
            let a = $a;
            let l = tiler_to_layout(a);
            assert_eq!(logical_product(a, Nil), logical_product(l, Nil));
            assert_eq!(logical_product(a, Int::<2>), logical_product(l, Int::<2>));
            assert_eq!(
                logical_product(a, Layout::new(Int::<2>, Int::<1>)),
                logical_product(l, Layout::new(Int::<2>, Int::<1>))
            );
        }};
    }
    macro_rules! check_promo_rank2 {
        ($a:expr) => {{
            let a = $a;
            let l = tiler_to_layout(a);
            check_promo!(a);
            assert_eq!(
                logical_product(a, tuple![Int::<2>, Int::<2>]),
                logical_product(l, tuple![Int::<2>, Int::<2>])
            );
            assert_eq!(
                logical_product(
                    a,
                    tuple![
                        Layout::new(Int::<2>, Int::<1>),
                        Layout::new(Int::<2>, Int::<2>),
                    ],
                ),
                logical_product(
                    l,
                    tuple![
                        Layout::new(Int::<2>, Int::<1>),
                        Layout::new(Int::<2>, Int::<2>),
                    ],
                )
            );
            assert_eq!(
                logical_product(a, tuple![Layout::new(Int::<2>, Int::<1>), Nil]),
                logical_product(l, tuple![Layout::new(Int::<2>, Int::<1>), Nil])
            );
        }};
    }

    check_promo!(Int::<6>);
    check_promo!(Int::<24>);
    check_promo_rank2!(tuple![Int::<6>, Int::<4>]);
    check_promo_rank2!(tuple![Int::<2>, Int::<3>, Int::<4>]);
    check_promo_rank2!(tuple![
        Layout::new(Int::<6>, Int::<2>),
        Layout::new(Int::<4>, Int::<1>),
    ]);
    check_promo_rank2!(tuple![Int::<2>, Layout::new(Int::<4>, Int::<1>)]);
}

#[test]
fn test_logical_product_short_and_padded_tiler() {
    macro_rules! check_short {
        ($a:expr) => {{
            let a = $a;
            let l = tiler_to_layout(a);
            let t2 = Layout::new(Int::<2>, Int::<1>);
            assert_eq!(
                logical_product(a, tuple![t2]),
                make_layout!(logical_product(l.get::<0>(), t2), l.get::<1>())
            );
            assert_eq!(
                logical_product(a, tuple![t2, Nil]),
                logical_product(a, tuple![t2])
            );
            assert_eq!(
                logical_product(a, tuple![Nil, t2]),
                make_layout!(l.get::<0>(), logical_product(l.get::<1>(), t2))
            );
            assert_eq!(logical_product(a, tuple![Nil, Nil]), l);
            assert_eq!(logical_product(a, tuple![]), l);

            assert_eq!(logical_product(a, tuple![t2]).size(), l.size() * 2);
            assert_eq!(logical_product(a, tuple![t2, Nil]).size(), l.size() * 2);
            assert_eq!(logical_product(a, tuple![Nil, Nil]).size(), l.size());
            assert_eq!(logical_product(a, tuple![]).size(), l.size());
        }};
    }

    check_short!(Layout::new(
        int_tuple![Int::<6>, Int::<4>],
        int_tuple![Int::<4>, Int::<1>],
    ));
    check_short!(tuple![Int::<6>, Int::<4>]);
    check_short!(tuple![
        Layout::new(Int::<6>, Int::<2>),
        Layout::new(Int::<4>, Int::<1>),
    ]);
}

#[test]
fn test_logical_product_is_complement_composition() {
    macro_rules! check_equiv {
        ($a:expr, $b:expr) => {{
            let a = $a;
            let b = $b;
            assert_eq!(
                logical_product(a, b),
                make_layout!(a, composition(complement(a), b))
            );
        }};
    }

    check_equiv!(
        Layout::new(Int::<3>, Int::<1>),
        Layout::new(Int::<4>, Int::<1>)
    );
    check_equiv!(
        Layout::new(
            int_tuple![Int::<2>, Int::<2>],
            int_tuple![Int::<4>, Int::<1>],
        ),
        Layout::new(Int::<6>, Int::<1>)
    );
    check_equiv!(
        Layout::new(
            int_tuple![Int::<6>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>],
        ),
        Layout::new(Int::<2>, Int::<1>)
    );
    check_equiv!(
        Layout::new(Int::<3>, Int::<32>),
        Layout::new(
            int_tuple![Int::<8>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>],
        )
    );
    check_equiv!(
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<2>)],
            int_tuple![(Int::<1>, Int::<4>), (Int::<8>, Int::<32>)],
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<2>],
            int_tuple![Int::<2>, Int::<1>],
        )
    );
}

#[test]
fn test_logical_product_coord() {
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(Int::<3>, e!(0)),
            Layout::new(Int::<4>, Int::<1>),
        ),
        Layout::new(
            int_tuple![Int::<3>, Int::<4>],
            tuple![e!(0), Int::<3> * e!(0)],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(Int::<3>, Int::<2> * e!(0)),
            Layout::new(Int::<4>, Int::<1>),
        ),
        Layout::new(
            int_tuple![Int::<3>, (Int::<2>, Int::<2>)],
            tuple![Int::<2> * e!(0), (e!(0), Int::<6> * e!(0))],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(int_tuple![Int::<3>, Int::<5>], tuple![e!(0), e!(1)]),
            Layout::new(Int::<4>, Int::<1>),
        ),
        Layout::new(
            int_tuple![(Int::<3>, Int::<5>), Int::<4>],
            tuple![(e!(0), e!(1)), Int::<5> * e!(1)],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(int_tuple![Int::<3>, Int::<5>], tuple![e!(0), e!(1)]),
            Layout::new(int_tuple![Int::<2>, Int::<4>], tuple![e!(0), e!(1)]),
        ),
        Layout::new(
            int_tuple![(Int::<3>, Int::<5>), (Int::<2>, Int::<4>)],
            tuple![(e!(0), e!(1)), (Int::<3> * e!(0), Int::<5> * e!(1))],
        )
    );
    assert_eq!(
        postcondition_logical_product!(
            Layout::new(
                int_tuple![Int::<3>, Int::<5>],
                tuple![Int::<2> * e!(0), e!(1)],
            ),
            Layout::new(int_tuple![Int::<2>, Int::<4>], tuple![e!(0), e!(1)]),
        ),
        Layout::new(
            int_tuple![(Int::<3>, Int::<5>), (Int::<2>, Int::<4>)],
            tuple![(Int::<2> * e!(0), e!(1)), (e!(0), Int::<5> * e!(1))],
        )
    );
}

#[test]
fn test_blocked_and_raked_product() {
    // 2D worked example from docs/04_layout_algebra.md
    let tile = Layout::new(
        int_tuple![Int::<2>, Int::<5>],
        int_tuple![Int::<5>, Int::<1>],
    );
    let grid = Layout::new(
        int_tuple![Int::<3>, Int::<4>],
        int_tuple![Int::<1>, Int::<3>],
    );
    let bp = postcondition_blocked_product!(tile, grid);
    assert_eq!(bp.get::<0>().size(), 6);
    assert_eq!(bp.get::<1>().size(), 20);
    assert_eq!(
        bp,
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), (Int::<5>, Int::<4>)],
            int_tuple![(Int::<5>, Int::<10>), (Int::<1>, Int::<30>)],
        )
    );

    let rp = postcondition_raked_product!(tile, grid);
    assert_eq!(rp.get::<0>().size(), 6);
    assert_eq!(rp.get::<1>().size(), 20);
    assert_eq!(
        rp,
        Layout::new(
            int_tuple![(Int::<3>, Int::<2>), (Int::<4>, Int::<5>)],
            int_tuple![(Int::<10>, Int::<5>), (Int::<30>, Int::<1>)],
        )
    );

    // Distinct offsets across the whole bounding box
    let tile_23 = Layout::from_shape(int_tuple![Int::<2>, Int::<3>]);
    let grid_45 = Layout::from_shape(int_tuple![Int::<4>, Int::<5>]);
    let bp_23 = postcondition_blocked_product!(tile_23, grid_45);
    let mut seen = [false; 120];
    for i in 0..bp_23.size() {
        let offset = bp_23.call(i) as usize;
        assert!(!seen[offset], "duplicate offset {offset} at index {i}");
        seen[offset] = true;
    }
    assert!(seen.iter().all(|&v| v));

    // Blocked vs raked: same image set, different iteration order
    let tile_22 = Layout::from_shape(int_tuple![Int::<2>, Int::<2>]);
    let grid_33 = Layout::from_shape(int_tuple![Int::<3>, Int::<3>]);
    let bp_22 = postcondition_blocked_product!(tile_22, grid_33);
    let rp_22 = postcondition_raked_product!(tile_22, grid_33);
    assert_eq!(bp_22.size(), rp_22.size());
    assert_ne!(bp_22, rp_22);
    let mut bp_offsets = [0i32; 36];
    let mut rp_offsets = [0i32; 36];
    for i in 0..36i32 {
        bp_offsets[i as usize] = bp_22.call(i);
        rp_offsets[i as usize] = rp_22.call(i);
    }
    bp_offsets.sort_unstable();
    rp_offsets.sort_unstable();
    assert_eq!(bp_offsets, rp_offsets);

    // Rank 0 empty layouts
    let empty = Layout::new(Nil, Nil);
    assert_eq!(postcondition_blocked_product!(empty, empty), empty);
    assert_eq!(postcondition_raked_product!(empty, empty), empty);

    // Rank 1 scalar layouts & 1-tuple layout
    let a1 = Layout::new(Int::<3>, Int::<1>);
    let b1 = Layout::new(Int::<4>, Int::<1>);
    let bp1 = postcondition_blocked_product!(a1, b1);
    let rp1 = postcondition_raked_product!(a1, b1);
    assert_eq!(
        bp1,
        Layout::new(
            int_tuple![(Int::<3>, Int::<4>)],
            int_tuple![(Int::<1>, Int::<3>)],
        )
    );
    assert_eq!(
        rp1,
        Layout::new(
            int_tuple![(Int::<4>, Int::<3>)],
            int_tuple![(Int::<3>, Int::<1>)],
        )
    );
    assert_eq!(blocked_product(a1, Int::<4>), bp1);
    assert_eq!(raked_product(a1, Int::<4>), rp1);
    assert_eq!(blocked_product(Int::<3>, Int::<4>), bp1);
    assert_eq!(raked_product(Int::<3>, Int::<4>), rp1);
    assert_eq!(
        blocked_product(a1, Layout::new(int_tuple![Int::<4>], int_tuple![Int::<1>])),
        bp1
    );

    // Rank 1 scalar whose complement splits: preserves full size=12 and rank=1!
    let a_gap = Layout::new(Int::<3>, Int::<2>);
    let bp_gap = postcondition_blocked_product!(a_gap, b1);
    let rp_gap = postcondition_raked_product!(a_gap, b1);
    assert_eq!(
        bp_gap,
        Layout::new(
            int_tuple![(Int::<3>, (Int::<2>, Int::<2>))],
            int_tuple![(Int::<2>, (Int::<1>, Int::<6>))],
        )
    );
    assert_eq!(
        rp_gap,
        Layout::new(
            int_tuple![((Int::<2>, Int::<2>), Int::<3>)],
            int_tuple![((Int::<1>, Int::<6>), Int::<2>)],
        )
    );
    let mut gap_seen = [false; 12];
    for i in 0..12i32 {
        gap_seen[bp_gap.call(i) as usize] = true;
    }
    assert!(gap_seen.iter().all(|&v| v));

    // Rank 3 layouts
    let a3 = Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<4>]);
    let b3 = Layout::from_shape(int_tuple![Int::<5>, Int::<6>, Int::<7>]);
    assert_eq!(
        postcondition_blocked_product!(a3, b3),
        Layout::new(
            int_tuple![
                (Int::<2>, Int::<5>),
                (Int::<3>, Int::<6>),
                (Int::<4>, Int::<7>)
            ],
            int_tuple![
                (Int::<1>, Int::<24>),
                (Int::<2>, Int::<120>),
                (Int::<6>, Int::<720>)
            ],
        )
    );
    assert_eq!(
        postcondition_raked_product!(a3, b3),
        Layout::new(
            int_tuple![
                (Int::<5>, Int::<2>),
                (Int::<6>, Int::<3>),
                (Int::<7>, Int::<4>)
            ],
            int_tuple![
                (Int::<24>, Int::<1>),
                (Int::<120>, Int::<2>),
                (Int::<720>, Int::<6>)
            ],
        )
    );

    // Nested tile modes & coordinate strides & broadcast grid stride
    let a_nested = Layout::new(
        int_tuple![(Int::<2>, Int::<2>), Int::<5>],
        int_tuple![(Int::<1>, Int::<10>), Int::<2>],
    );
    assert_eq!(
        postcondition_blocked_product!(a_nested, grid),
        Layout::new(
            int_tuple![((Int::<2>, Int::<2>), Int::<3>), (Int::<5>, Int::<4>)],
            int_tuple![((Int::<1>, Int::<10>), Int::<20>), (Int::<2>, Int::<60>)],
        )
    );
    assert_eq!(
        postcondition_raked_product!(a_nested, grid),
        Layout::new(
            int_tuple![(Int::<3>, (Int::<2>, Int::<2>)), (Int::<4>, Int::<5>)],
            int_tuple![(Int::<20>, (Int::<1>, Int::<10>)), (Int::<60>, Int::<2>)],
        )
    );

    let a_coord = Layout::new(
        int_tuple![Int::<2>, Int::<3>],
        tuple![Int::<2> * e!(0), e!(1)],
    );
    let b_coord = Layout::new(int_tuple![Int::<4>, Int::<5>], tuple![e!(0), e!(1)]);
    assert_eq!(
        postcondition_blocked_product!(a_coord, b_coord),
        Layout::new(
            int_tuple![(Int::<2>, (Int::<2>, Int::<2>)), (Int::<3>, Int::<5>)],
            tuple![
                (Int::<2> * e!(0), (e!(0), Int::<4> * e!(0))),
                (e!(1), Int::<3> * e!(1))
            ],
        )
    );
    assert_eq!(
        postcondition_raked_product!(a_coord, b_coord),
        Layout::new(
            int_tuple![((Int::<2>, Int::<2>), Int::<2>), (Int::<5>, Int::<3>)],
            tuple![
                ((e!(0), Int::<4> * e!(0)), Int::<2> * e!(0)),
                (Int::<3> * e!(1), e!(1))
            ],
        )
    );

    // Tiler LHS promotion in blocked_product / raked_product
    assert_eq!(
        blocked_product(tuple![Int::<2>, Int::<5>], grid),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), (Int::<5>, Int::<4>)],
            tuple![
                (e!(0), Int::<5> * e!(1)),
                (e!(1), Int::<15> * e!(1))
            ],
        )
    );
    assert_eq!(
        raked_product(tuple![Int::<2>, Int::<5>], grid),
        Layout::new(
            int_tuple![(Int::<3>, Int::<2>), (Int::<4>, Int::<5>)],
            tuple![
                (Int::<5> * e!(1), e!(0)),
                (Int::<15> * e!(1), e!(1))
            ],
        )
    );
}

#[test]
fn test_logical_product_const_eval() {
    type Pair2<S0, S1, D0, D1> = Layout<Til<S0, Til<S1, Nil>>, Til<D0, Til<D1, Nil>>>;

    const A: Layout<Int<3>, Int<1>> = Layout::new(Int::<3>, Int::<1>);
    const B: Layout<Int<4>, Int<1>> = Layout::new(Int::<4>, Int::<1>);
    const LP: Pair2<Int<3>, Int<4>, Int<1>, Int<3>> = logical_product(A, B);
    assert_eq!(
        LP,
        Layout::new(
            int_tuple![Int::<3>, Int::<4>],
            int_tuple![Int::<1>, Int::<3>],
        )
    );
    assert_eq!(A.logical_product(B), LP);

    const TILE: Pair2<Int<2>, Int<5>, Int<5>, Int<1>> = Layout::new(
        int_tuple![Int::<2>, Int::<5>],
        int_tuple![Int::<5>, Int::<1>],
    );
    const GRID: Pair2<Int<3>, Int<4>, Int<1>, Int<3>> = Layout::new(
        int_tuple![Int::<3>, Int::<4>],
        int_tuple![Int::<1>, Int::<3>],
    );
    const {
        assert!(blocked_product(TILE, GRID) == TILE.blocked_product(GRID));
        assert!(raked_product(TILE, GRID) == TILE.raked_product(GRID));
    }

    const PAIR: Pair2<Int<3>, Int<5>, Int<1>, Int<20>> = Layout::new(
        int_tuple![Int::<3>, Int::<5>],
        int_tuple![Int::<1>, Int::<20>],
    );
    const {
        assert!(
            logical_product_at::<mode![0], _, _>(PAIR, B)
                == PAIR.logical_product_at::<mode![0], _>(B)
        );
    }
}

#[test]
fn test_logical_product_dynamic_and_panics() {
    // Dynamic success via free functions and Layout methods
    let dyn_a = Layout::new(3i32, Int::<1>);
    let dyn_b = Layout::new(4i32, 2i32);
    let expected = Layout::new(
        int_tuple![3i32, 4i32],
        int_tuple![Int::<1>, 6i32],
    );
    assert_eq!(logical_product(dyn_a, dyn_b), expected);
    assert_eq!(dyn_a.logical_product(dyn_b), expected);

    // Dynamic extent with static stride gap -> static gap + dynamic tail stride
    let dyn_gap = Layout::new(3i32, Int::<2>);
    let b_static = Layout::new(Int::<4>, Int::<1>);
    let expected_gap = Layout::new(
        int_tuple![3i32, (Int::<2>, Int::<2>)],
        int_tuple![Int::<2>, (Int::<1>, 6i32)],
    );
    assert_eq!(logical_product(dyn_gap, b_static), expected_gap);

    // Dynamic blocked_product and raked_product
    let dyn_tile = Layout::new(
        int_tuple![Int::<2>, 5i32],
        int_tuple![Int::<1>, Int::<2>],
    );
    let dyn_grid = Layout::new(
        int_tuple![3i32, 4i32],
        int_tuple![1i32, 3i32],
    );
    let expected_bp = Layout::new(
        int_tuple![(Int::<2>, 3i32), (5i32, 4i32)],
        int_tuple![(Int::<1>, 10i32), (Int::<2>, 30i32)],
    );
    let expected_rp = Layout::new(
        int_tuple![(3i32, Int::<2>), (4i32, 5i32)],
        int_tuple![(10i32, Int::<1>), (30i32, Int::<2>)],
    );
    assert_eq!(blocked_product(dyn_tile, dyn_grid), expected_bp);
    assert_eq!(dyn_tile.blocked_product(dyn_grid), expected_bp);
    assert_eq!(raked_product(dyn_tile, dyn_grid), expected_rp);
    assert_eq!(dyn_tile.raked_product(dyn_grid), expected_rp);

    // Mode-targeted variant
    let mode_a = Layout::new(
        int_tuple![3i32, 5i32],
        int_tuple![Int::<1>, Int::<20>],
    );
    assert_eq!(
        logical_product_at::<mode![0], _, _>(mode_a, dyn_b),
        mode_a.logical_product_at::<mode![0], _>(dyn_b)
    );

    // Dynamic failures panic:
    // 1. Zero extent with non-zero stride in A (complement precondition failure)
    assert!(
        std::panic::catch_unwind(|| {
            logical_product(Layout::new(0i32, Int::<1>), Layout::new(4i32, 1i32))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            blocked_product(Layout::new(0i32, Int::<1>), Layout::new(4i32, 1i32))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            raked_product(Layout::new(0i32, Int::<1>), Layout::new(4i32, 1i32))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            logical_product_at::<mode![0], _, _>(
                Layout::new(int_tuple![0i32, 5i32], int_tuple![Int::<1>, Int::<20>]),
                Layout::new(4i32, 1i32),
            )
        })
        .is_err()
    );

    // 2. Complement running-stride overflow
    assert!(
        std::panic::catch_unwind(|| {
            logical_product(
                Layout::new(int::MAX, Int::<2>),
                Layout::new(Int::<4>, Int::<1>),
            )
        })
        .is_err()
    );

    // 3. Stride multiplication overflow in composition(complement(A), B)
    assert!(
        std::panic::catch_unwind(|| {
            logical_product(
                Layout::new(Int::<3>, Int::<1>),
                Layout::new(4i32, int::MAX),
            )
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            blocked_product(
                Layout::new(Int::<3>, Int::<1>),
                Layout::new(4i32, int::MAX),
            )
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            raked_product(
                Layout::new(Int::<3>, Int::<1>),
                Layout::new(4i32, int::MAX),
            )
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            logical_product_at::<mode![0], _, _>(
                Layout::new(int_tuple![Int::<3>, Int::<5>], int_tuple![Int::<1>, Int::<20>]),
                Layout::new(4i32, int::MAX),
            )
        })
        .is_err()
    );

    // 4. Dynamic scalar tiler promotion & negative stride in B
    assert_eq!(
        logical_product(3i32, 4i32),
        Layout::new(int_tuple![3i32, 4i32], int_tuple![Int::<1>, 3i32])
    );
    assert_eq!(
        blocked_product(3i32, 4i32),
        Layout::new(int_tuple![(3i32, 4i32)], int_tuple![(Int::<1>, 3i32)])
    );
    assert_eq!(
        raked_product(3i32, 4i32),
        Layout::new(int_tuple![(4i32, 3i32)], int_tuple![(3i32, Int::<1>)])
    );
    assert!(std::panic::catch_unwind(|| logical_product(0i32, 4i32)).is_err());
    assert!(std::panic::catch_unwind(|| blocked_product(0i32, 4i32)).is_err());
    assert!(std::panic::catch_unwind(|| raked_product(0i32, 4i32)).is_err());
    assert_eq!(
        logical_product(
            Layout::new(Int::<3>, Int::<1>),
            Layout::new(Int::<4>, Int::<-1>),
        ),
        Layout::new(
            int_tuple![Int::<3>, Int::<4>],
            int_tuple![Int::<1>, Int::<-3>],
        )
    );

    // 5. By-mode tuple error propagation (head failure and tail failure)
    assert!(
        std::panic::catch_unwind(|| {
            logical_product(
                Layout::new(int_tuple![0i32, 5i32], int_tuple![Int::<1>, Int::<20>]),
                tuple![4i32, Int::<2>],
            )
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            logical_product(
                Layout::new(int_tuple![3i32, 0i32], int_tuple![Int::<1>, Int::<20>]),
                tuple![4i32, Int::<2>],
            )
        })
        .is_err()
    );

    // 6. Negative dynamic extents panic
    assert!(
        std::panic::catch_unwind(|| {
            logical_product(Layout::new(-3i32, Int::<1>), Layout::new(4i32, 1i32))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            logical_product(Layout::new(3i32, Int::<1>), Layout::new(-4i32, 1i32))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            blocked_product(Layout::new(-3i32, Int::<1>), Layout::new(4i32, 1i32))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            raked_product(Layout::new(3i32, Int::<1>), Layout::new(-4i32, 1i32))
        })
        .is_err()
    );
}

#[test]
#[should_panic]
fn test_logical_product_panics_on_dynamic_complement_failure() {
    let _ = logical_product(Layout::new(0i32, Int::<1>), Layout::new(4i32, 1i32));
}

#[test]
#[should_panic]
fn test_blocked_product_panics_on_dynamic_overflow() {
    let _ = blocked_product(
        Layout::new(Int::<3>, Int::<1>),
        Layout::new(4i32, int::MAX),
    );
}

#[test]
#[should_panic]
fn test_raked_product_panics_on_dynamic_overflow() {
    let _ = raked_product(
        Layout::new(Int::<3>, Int::<1>),
        Layout::new(4i32, int::MAX),
    );
}
