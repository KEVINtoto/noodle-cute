#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl,
    const_cmp
)]
#![allow(incomplete_features)]

use noodle_cute::algebra::complement::complement_with;
use noodle_cute::algebra::composition::composition;
use noodle_cute::algebra::logical_divide::{
    logical_divide, logical_divide_at, zipped_divide, zipped_divide_at,
};
use noodle_cute::basis::basis_eq;
use noodle_cute::integer::{Int, int};
use noodle_cute::layout::{Layout, tiler_to_layout};
use noodle_cute::shape::compatible;
use noodle_cute::tuple::{Item, Nil, Til, WeaklyCongruent};
use noodle_cute::{e, int_tuple, make_layout, mode, tuple};

fn assert_weak_congruent<A: WeaklyCongruent<B>, B: Item>(_: &A, _: &B) {}

macro_rules! postcondition_zipped_divide {
    ($a:expr, $b:expr $(,)?) => {{
        let a = $a;
        let b = $b;
        let r = zipped_divide(a, b);
        let tiler = tiler_to_layout(b);

        assert_eq!(r.rank(), 2);

        let r0 = r.get::<0>();
        let r1 = r.get::<1>();
        assert!(compatible(&tiler.shape(), &r0.shape()));

        for i in 0..tiler.size() {
            assert!(basis_eq(r.call(tuple![i, 0i32]), a.call(tiler.call(i))));
        }

        assert_weak_congruent(&r1, &tiler.coprofile());

        for i in 0..a.size() {
            let ai = a.call(i);
            assert!((0..r.size()).any(|j| basis_eq(r.call(j), ai)));
        }
    }};
}

#[test]
fn test_zipped_divide() {
    postcondition_zipped_divide!(
        Layout::new(Int::<1>, Int::<0>),
        Layout::new(Int::<1>, Int::<0>),
    );
    postcondition_zipped_divide!(
        Layout::new(Int::<1>, Int::<0>),
        Layout::new(Int::<1>, Int::<1>),
    );
    postcondition_zipped_divide!(
        Layout::new(Int::<1>, Int::<1>),
        Layout::new(Int::<1>, Int::<0>),
    );
    postcondition_zipped_divide!(
        Layout::new(Int::<1>, Int::<1>),
        Layout::new(Int::<1>, Int::<1>),
    );
    postcondition_zipped_divide!(
        Layout::new(Int::<6>, Int::<1>),
        Layout::new(Int::<2>, Int::<1>),
    );
    postcondition_zipped_divide!(
        Layout::new(Int::<6>, Int::<1>),
        Layout::new(Int::<2>, Int::<3>),
    );
    postcondition_zipped_divide!(
        Layout::new(Int::<6>, Int::<1>),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<3>, Int::<1>],
        ),
    );
    postcondition_zipped_divide!(
        Layout::new(Int::<6>, Int::<2>),
        Layout::new(Int::<2>, Int::<1>),
    );
    postcondition_zipped_divide!(
        Layout::new(Int::<6>, Int::<2>),
        Layout::new(Int::<2>, Int::<3>),
    );
    postcondition_zipped_divide!(
        Layout::new(Int::<6>, Int::<2>),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<3>, Int::<1>],
        ),
    );
    postcondition_zipped_divide!(
        Layout::new(
            int_tuple![Int::<6>, Int::<6>],
            int_tuple![Int::<1>, Int::<12>],
        ),
        Layout::new(
            int_tuple![Int::<6>, Int::<3>],
            int_tuple![Int::<3>, Int::<1>],
        ),
    );
    postcondition_zipped_divide!(
        Layout::new(
            int_tuple![Int::<6>, Int::<6>],
            int_tuple![Int::<12>, Int::<1>],
        ),
        Layout::new(
            int_tuple![Int::<6>, Int::<3>],
            int_tuple![Int::<3>, Int::<1>],
        ),
    );
    postcondition_zipped_divide!(
        Layout::new(Int::<32>, Int::<1>),
        Layout::new(Int::<2>, Int::<8>),
    );
    postcondition_zipped_divide!(
        Layout::new(
            int_tuple![Int::<4>, Int::<1>],
            int_tuple![Int::<1>, Int::<1>],
        ),
        Layout::new(Int::<2>, Int::<1>),
    );
    postcondition_zipped_divide!(
        Layout::new(
            int_tuple![Int::<4>, Int::<1>],
            int_tuple![Int::<1>, Int::<1>],
        ),
        Layout::new(Int::<2>, Int::<2>),
    );
    postcondition_zipped_divide!(
        Layout::new(
            int_tuple![Int::<8>, Int::<8>],
            int_tuple![Int::<1>, Int::<8>],
        ),
        Layout::new(Int::<32>, Int::<2>),
    );
    postcondition_zipped_divide!(
        Layout::new(
            int_tuple![Int::<8>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>],
        ),
        Layout::new(Int::<32>, Int::<2>),
    );
}

#[test]
fn test_zipped_divide_coord() {
    let a = Layout::new(
        int_tuple![Int::<8>, Int::<8>],
        int_tuple![Int::<9>, Int::<1>],
    );
    postcondition_zipped_divide!(a, Layout::new(Int::<4>, e!(0)));
    postcondition_zipped_divide!(a, Layout::new(Int::<4>, e!(1)));
    postcondition_zipped_divide!(a, Layout::new(Int::<4>, Int::<2> * e!(0)));
    postcondition_zipped_divide!(a, Layout::new(Int::<4>, Int::<2> * e!(1)));
    postcondition_zipped_divide!(
        a,
        Layout::new(
            int_tuple![Int::<4>, Int::<4>],
            tuple![e!(1), Int::<2> * e!(0)],
        ),
    );
    postcondition_zipped_divide!(
        a,
        Layout::new(
            int_tuple![Int::<5>, Int::<7>],
            tuple![Int::<3> * e!(1), Int::<2> * e!(0)],
        ),
    );
    let r_e0 = zipped_divide(a, Layout::new(Int::<4>, e!(0)));
    assert_eq!(
        r_e0,
        Layout::new(
            int_tuple![Int::<4>, (Int::<2>, Int::<8>)],
            int_tuple![Int::<9>, (Int::<36>, Int::<1>)],
        )
    );
}

#[test]
fn test_zipped_divide_associativity() {
    let tiler_mn = tuple![Int::<8>, Int::<8>];
    let clayout_tv = Layout::new(
        int_tuple![
            (Int::<2>, Int::<2>, Int::<2>),
            (Int::<2>, Int::<2>, Int::<2>)
        ],
        int_tuple![
            (Int::<1>, Int::<16>, Int::<4>),
            (Int::<8>, Int::<2>, Int::<32>)
        ],
    );

    let data_col = Layout::from_shape(int_tuple![Int::<32>, Int::<64>]);
    let c_mma_0 = composition(zipped_divide(data_col, tiler_mn), tuple![clayout_tv, Nil]);
    let c_mma_1 = zipped_divide(data_col, composition(tiler_mn, clayout_tv));
    assert_eq!(c_mma_0, c_mma_1);

    let data_row = Layout::new(
        int_tuple![Int::<32>, Int::<64>],
        int_tuple![Int::<64>, Int::<1>],
    );
    let c_mma_0 = composition(zipped_divide(data_row, tiler_mn), tuple![clayout_tv, Nil]);
    let c_mma_1 = zipped_divide(data_row, composition(tiler_mn, clayout_tv));
    assert_eq!(c_mma_0, c_mma_1);
}

#[test]
fn test_logical_divide_tiler() {
    let a1 = Layout::new(
        int_tuple![Int::<6>, Int::<4>],
        int_tuple![Int::<4>, Int::<1>],
    );
    let t1 = tuple![
        Layout::new(Int::<2>, Int::<1>),
        Layout::new(Int::<2>, Int::<1>),
    ];
    postcondition_zipped_divide!(a1, t1);
    let expected1 = make_layout!(
        logical_divide(a1.get::<0>(), t1.get::<0>()),
        logical_divide(a1.get::<1>(), t1.get::<1>())
    );
    assert_eq!(logical_divide(a1, t1), expected1);
    assert_ne!(logical_divide(a1, t1), zipped_divide(a1, t1));

    let a2 = Layout::new(
        int_tuple![Int::<8>, Int::<8>],
        int_tuple![Int::<8>, Int::<1>],
    );
    let t2 = tuple![
        Layout::new(Int::<2>, Int::<1>),
        Layout::new(Int::<4>, Int::<1>),
    ];
    postcondition_zipped_divide!(a2, t2);
    let expected2 = make_layout!(
        logical_divide(a2.get::<0>(), t2.get::<0>()),
        logical_divide(a2.get::<1>(), t2.get::<1>())
    );
    assert_eq!(logical_divide(a2, t2), expected2);
    assert_ne!(logical_divide(a2, t2), zipped_divide(a2, t2));
}

#[test]
fn test_logical_divide_mode() {
    let a = Layout::new(
        int_tuple![(Int::<3>, Int::<8>), Int::<5>],
        int_tuple![(Int::<1>, Int::<3>), Int::<24>],
    );
    let b = Layout::new(Int::<4>, Int::<2>);

    assert_eq!(
        logical_divide_at::<mode![0], _, _>(a, b),
        make_layout!(logical_divide(a.get::<0>(), b), a.get::<1>())
    );
    assert_eq!(
        logical_divide_at::<mode![1], _, _>(a, b),
        make_layout!(a.get::<0>(), logical_divide(a.get::<1>(), b))
    );
    assert_eq!(
        logical_divide_at::<mode![0, 1], _, _>(a, b),
        make_layout!(
            make_layout!(
                a.get::<0>().get::<0>(),
                logical_divide(a.get::<0>().get::<1>(), b)
            ),
            a.get::<1>()
        )
    );

    assert_eq!(
        logical_divide_at::<mode![], _, _>(a, b),
        logical_divide(a, b)
    );
    assert_eq!(
        logical_divide_at::<mode![-1], _, _>(a, b),
        logical_divide_at::<mode![1], _, _>(a, b)
    );
    assert_eq!(
        logical_divide_at::<mode![-2, -1], _, _>(a, b),
        logical_divide_at::<mode![0, 1], _, _>(a, b)
    );
    assert_eq!(
        a.logical_divide_at::<mode![0, 1], _>(b),
        logical_divide_at::<mode![0, 1], _, _>(a, b)
    );
}

#[test]
fn test_zipped_divide_mode() {
    let a = Layout::from_shape(int_tuple![(Int::<9>, Int::<32>), Int::<7>]);
    let tiler = tuple![
        Layout::new(Int::<3>, Int::<3>),
        Layout::new(
            int_tuple![Int::<2>, Int::<4>],
            int_tuple![Int::<1>, Int::<8>],
        ),
    ];

    assert_eq!(
        zipped_divide_at::<mode![0], _, _>(a, tiler),
        make_layout!(zipped_divide(a.get::<0>(), tiler), a.get::<1>())
    );
    assert_eq!(
        zipped_divide_at::<mode![0], _, _>(a, tiler),
        logical_divide_at::<mode![0], _, _>(a, tiler_to_layout(tiler))
    );
    assert_eq!(
        zipped_divide_at::<mode![], _, _>(a, tiler),
        zipped_divide(a, tiler)
    );
    assert_eq!(
        zipped_divide_at::<mode![-2], _, _>(a, tiler),
        zipped_divide_at::<mode![0], _, _>(a, tiler)
    );
    assert_eq!(
        zipped_divide_at::<mode![0, 1], _, _>(a, Layout::new(Int::<4>, Int::<2>)),
        make_layout!(
            make_layout!(
                a.get::<0>().get::<0>(),
                zipped_divide(a.get::<0>().get::<1>(), Layout::new(Int::<4>, Int::<2>))
            ),
            a.get::<1>()
        )
    );

    let r = zipped_divide_at::<mode![0], _, _>(a, tiler);
    assert_eq!(r.get::<0>().rank(), 2);
    assert!(compatible(&tiler_to_layout(tiler).shape(), &r.get::<0>().get::<0>().shape()));
}

#[test]
fn test_logical_divide_types() {
    let a = Layout::new(
        int_tuple![Int::<6>, Int::<4>],
        int_tuple![Int::<4>, Int::<1>],
    );

    assert_eq!(
        logical_divide(Int::<6>, Layout::new(Int::<2>, Int::<1>)),
        logical_divide(Layout::new(Int::<6>, Int::<1>), Layout::new(Int::<2>, Int::<1>))
    );
    assert_eq!(
        logical_divide(tuple![Int::<6>, Int::<4>], Layout::new(Int::<2>, Int::<1>)),
        logical_divide(
            tiler_to_layout(tuple![Int::<6>, Int::<4>]),
            Layout::new(Int::<2>, Int::<1>)
        )
    );

    assert_eq!(
        logical_divide(Nil, Layout::new(Int::<2>, Int::<1>)),
        Layout::new(int_tuple![Int::<2>, Int::<1>], int_tuple![Int::<1>, Int::<2>])
    );
    assert_eq!(
        logical_divide(Nil, tuple![Int::<3>, Int::<5>]),
        Layout::new(
            int_tuple![(Int::<3>, Int::<1>), (Int::<5>, Int::<1>)],
            tuple![(e!(0), Int::<3> * e!(0)), (e!(1), Int::<5> * e!(1))],
        )
    );
    assert_eq!(logical_divide(Nil, Nil), Nil);
    assert_eq!(
        logical_divide(Nil, tuple![Layout::new(Int::<2>, Int::<1>), Nil]),
        Layout::new(
            int_tuple![(Int::<2>, Int::<1>), (Int::<1>, Int::<1>)],
            tuple![(e!(0), Int::<2> * e!(0)), (e!(1), e!(1))],
        )
    );
    assert_eq!(
        zipped_divide(Nil, Layout::new(Int::<2>, Int::<1>)),
        Layout::new(int_tuple![Int::<2>, Int::<1>], int_tuple![Int::<1>, Int::<2>])
    );
    assert_eq!(
        zipped_divide(Nil, tuple![Int::<3>, Int::<5>]),
        Layout::new(
            int_tuple![(Int::<3>, Int::<5>), (Int::<1>, Int::<1>)],
            tuple![(e!(0), e!(1)), (Int::<3> * e!(0), Int::<5> * e!(1))],
        )
    );

    assert_eq!(
        logical_divide(Nil, Layout::new(Int::<4>, Int::<2>)),
        Layout::new(
            int_tuple![Int::<4>, (Int::<2>, Int::<1>)],
            int_tuple![Int::<2>, (Int::<1>, Int::<8>)],
        )
    );
    assert_eq!(
        logical_divide(Layout::from_shape(Int::<24>), Layout::new(Int::<4>, Int::<2>)),
        Layout::new(
            int_tuple![Int::<4>, (Int::<2>, Int::<3>)],
            int_tuple![Int::<2>, (Int::<1>, Int::<8>)],
        )
    );

    assert_eq!(
        logical_divide_at::<mode![0], _, _>(Nil, tuple![Int::<3>, Int::<5>]),
        make_layout!(logical_divide(Nil, tuple![Int::<3>, Int::<5>]))
    );
    assert_eq!(
        logical_divide_at::<mode![1], _, _>(Nil, Layout::new(Int::<2>, Int::<1>)),
        make_layout!(
            Layout::new(Int::<1>, Int::<0>),
            logical_divide(Nil, Layout::new(Int::<2>, Int::<1>))
        )
    );

    assert_eq!(logical_divide(a, Nil), a);
    assert_eq!(
        logical_divide(a, Int::<2>),
        logical_divide(a, Layout::new(Int::<2>, Int::<1>))
    );
}

#[test]
fn test_logical_divide_arg_matrix() {
    // A = Nil
    assert_eq!(logical_divide(Nil, Nil), Nil);
    assert_eq!(
        logical_divide(Nil, Int::<2>),
        Layout::new(int_tuple![Int::<2>, Int::<1>], int_tuple![Int::<1>, Int::<2>])
    );
    assert_eq!(
        logical_divide(Nil, tuple![Int::<2>, Int::<2>]),
        Layout::new(
            int_tuple![(Int::<2>, Int::<1>), (Int::<2>, Int::<1>)],
            tuple![(e!(0), Int::<2> * e!(0)), (e!(1), Int::<2> * e!(1))],
        )
    );
    assert_eq!(
        logical_divide(Nil, Layout::new(Int::<2>, Int::<1>)),
        Layout::new(int_tuple![Int::<2>, Int::<1>], int_tuple![Int::<1>, Int::<2>])
    );
    assert_eq!(
        logical_divide(
            Nil,
            tuple![
                Layout::new(Int::<2>, Int::<1>),
                Layout::new(Int::<2>, Int::<2>),
            ]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<1>), (Int::<2>, (Int::<2>, Int::<1>))],
            tuple![
                (e!(0), Int::<2> * e!(0)),
                (Int::<2> * e!(1), (e!(1), Int::<4> * e!(1)))
            ],
        )
    );

    // A = Int<24>
    assert_eq!(logical_divide(Int::<24>, Nil), Layout::new(Int::<24>, Int::<1>));
    assert_eq!(
        logical_divide(Int::<24>, Int::<2>),
        Layout::new(int_tuple![Int::<2>, Int::<12>], int_tuple![Int::<1>, Int::<2>])
    );
    assert_eq!(
        logical_divide(Int::<24>, Layout::new(Int::<2>, Int::<1>)),
        Layout::new(int_tuple![Int::<2>, Int::<12>], int_tuple![Int::<1>, Int::<2>])
    );

    // A = (6, 4)
    let t_a = tuple![Int::<6>, Int::<4>];
    assert_eq!(
        logical_divide(t_a, Nil),
        Layout::new(int_tuple![Int::<6>, Int::<4>], tuple![e!(0), e!(1)])
    );
    assert_eq!(
        logical_divide(t_a, Int::<2>),
        Layout::new(
            int_tuple![Int::<2>, (Int::<3>, Int::<4>)],
            tuple![e!(0), (Int::<2> * e!(0), e!(1))],
        )
    );
    assert_eq!(
        logical_divide(t_a, tuple![Int::<2>, Int::<2>]),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), (Int::<2>, Int::<2>)],
            tuple![(e!(0), Int::<2> * e!(0)), (e!(1), Int::<2> * e!(1))],
        )
    );
    assert_eq!(
        logical_divide(t_a, Layout::new(Int::<2>, Int::<1>)),
        Layout::new(
            int_tuple![Int::<2>, (Int::<3>, Int::<4>)],
            tuple![e!(0), (Int::<2> * e!(0), e!(1))],
        )
    );
    assert_eq!(
        logical_divide(
            t_a,
            tuple![
                Layout::new(Int::<2>, Int::<1>),
                Layout::new(Int::<2>, Int::<2>),
            ]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), (Int::<2>, Int::<2>)],
            tuple![(e!(0), Int::<2> * e!(0)), (Int::<2> * e!(1), e!(1))],
        )
    );

    // A = Layout((6, 4), (4, 1))
    let l_a = Layout::new(
        int_tuple![Int::<6>, Int::<4>],
        int_tuple![Int::<4>, Int::<1>],
    );
    assert_eq!(logical_divide(l_a, Nil), l_a);
    assert_eq!(
        logical_divide(l_a, Int::<2>),
        Layout::new(
            int_tuple![Int::<2>, (Int::<3>, Int::<4>)],
            int_tuple![Int::<4>, (Int::<8>, Int::<1>)],
        )
    );
    assert_eq!(
        logical_divide(l_a, tuple![Int::<2>, Int::<2>]),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), (Int::<2>, Int::<2>)],
            int_tuple![(Int::<4>, Int::<8>), (Int::<1>, Int::<2>)],
        )
    );
    assert_eq!(
        logical_divide(l_a, Layout::new(Int::<2>, Int::<1>)),
        Layout::new(
            int_tuple![Int::<2>, (Int::<3>, Int::<4>)],
            int_tuple![Int::<4>, (Int::<8>, Int::<1>)],
        )
    );
    assert_eq!(
        logical_divide(
            l_a,
            tuple![
                Layout::new(Int::<2>, Int::<1>),
                Layout::new(Int::<2>, Int::<2>),
            ]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), (Int::<2>, Int::<2>)],
            int_tuple![(Int::<4>, Int::<8>), (Int::<2>, Int::<1>)],
        )
    );

    // A = (Layout(6, 2), Layout(4, 1))
    let tl_a = tuple![
        Layout::new(Int::<6>, Int::<2>),
        Layout::new(Int::<4>, Int::<1>),
    ];
    assert_eq!(
        logical_divide(tl_a, Nil),
        Layout::new(
            int_tuple![Int::<6>, Int::<4>],
            tuple![Int::<2> * e!(0), e!(1)],
        )
    );
    assert_eq!(
        logical_divide(tl_a, Int::<2>),
        Layout::new(
            int_tuple![Int::<2>, (Int::<3>, Int::<4>)],
            tuple![Int::<2> * e!(0), (Int::<4> * e!(0), e!(1))],
        )
    );
    assert_eq!(
        logical_divide(tl_a, tuple![Int::<2>, Int::<2>]),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), (Int::<2>, Int::<2>)],
            tuple![
                (Int::<2> * e!(0), Int::<4> * e!(0)),
                (e!(1), Int::<2> * e!(1))
            ],
        )
    );
    assert_eq!(
        logical_divide(tl_a, Layout::new(Int::<2>, Int::<1>)),
        Layout::new(
            int_tuple![Int::<2>, (Int::<3>, Int::<4>)],
            tuple![Int::<2> * e!(0), (Int::<4> * e!(0), e!(1))],
        )
    );
    assert_eq!(
        logical_divide(
            tl_a,
            tuple![
                Layout::new(Int::<2>, Int::<1>),
                Layout::new(Int::<2>, Int::<2>),
            ]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), (Int::<2>, Int::<2>)],
            tuple![
                (Int::<2> * e!(0), Int::<4> * e!(0)),
                (Int::<2> * e!(1), e!(1))
            ],
        )
    );
}

#[test]
fn test_logical_divide_promotes_tiler_a() {
    macro_rules! check_promo {
        ($a:expr) => {{
            let a = $a;
            let l = tiler_to_layout(a);
            assert_eq!(logical_divide(a, Nil), logical_divide(l, Nil));
            assert_eq!(logical_divide(a, Int::<2>), logical_divide(l, Int::<2>));
            assert_eq!(
                logical_divide(a, Layout::new(Int::<2>, Int::<1>)),
                logical_divide(l, Layout::new(Int::<2>, Int::<1>))
            );
        }};
    }
    macro_rules! check_promo_rank2 {
        ($a:expr) => {{
            let a = $a;
            let l = tiler_to_layout(a);
            check_promo!(a);
            assert_eq!(
                logical_divide(a, tuple![Int::<2>, Int::<2>]),
                logical_divide(l, tuple![Int::<2>, Int::<2>])
            );
            assert_eq!(
                logical_divide(
                    a,
                    tuple![
                        Layout::new(Int::<2>, Int::<1>),
                        Layout::new(Int::<2>, Int::<2>),
                    ]
                ),
                logical_divide(
                    l,
                    tuple![
                        Layout::new(Int::<2>, Int::<1>),
                        Layout::new(Int::<2>, Int::<2>),
                    ]
                )
            );
            assert_eq!(
                logical_divide(a, tuple![Layout::new(Int::<2>, Int::<1>), Nil]),
                logical_divide(l, tuple![Layout::new(Int::<2>, Int::<1>), Nil])
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
fn test_logical_divide_short_and_padded_tiler() {
    macro_rules! check_short {
        ($a:expr) => {{
            let a = $a;
            let l = tiler_to_layout(a);
            let t2 = Layout::new(Int::<2>, Int::<1>);
            assert_eq!(
                logical_divide(a, tuple![t2]),
                make_layout!(logical_divide(l.get::<0>(), t2), l.get::<1>())
            );
            assert_eq!(
                logical_divide(a, tuple![t2, Nil]),
                logical_divide(a, tuple![t2])
            );
            assert_eq!(
                logical_divide(a, tuple![Nil, t2]),
                make_layout!(l.get::<0>(), logical_divide(l.get::<1>(), t2))
            );
            assert_eq!(logical_divide(a, tuple![Nil, Nil]), l);
            assert_eq!(logical_divide(a, Nil), l);

            assert_eq!(logical_divide(a, tuple![t2]).size(), l.size());
            assert_eq!(logical_divide(a, tuple![t2, Nil]).size(), l.size());
            assert_eq!(logical_divide(a, tuple![Nil, Nil]).size(), l.size());
            assert_eq!(logical_divide(a, tuple![]).size(), l.size());
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
fn test_logical_divide_is_composition_with_complement() {
    macro_rules! check_equiv {
        ($a:expr, $b:expr) => {{
            let a = $a;
            let b = $b;
            assert_eq!(
                logical_divide(a, b),
                composition(a, make_layout!(b, complement_with(b, a.shape())))
            );
        }};
    }

    check_equiv!(
        Layout::new(Int::<24>, Int::<1>),
        Layout::new(Int::<4>, Int::<2>)
    );
    check_equiv!(
        Layout::new(
            int_tuple![Int::<6>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>],
        ),
        Layout::new(Int::<2>, Int::<1>)
    );
    check_equiv!(
        Layout::new(
            int_tuple![Int::<8>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>],
        ),
        Layout::new(Int::<32>, Int::<2>)
    );
    check_equiv!(
        Layout::new(
            int_tuple![Int::<6>, Int::<6>],
            int_tuple![Int::<1>, Int::<12>],
        ),
        Layout::new(
            int_tuple![Int::<6>, Int::<3>],
            int_tuple![Int::<3>, Int::<1>],
        )
    );
    check_equiv!(
        Layout::new(
            int_tuple![Int::<8>, Int::<8>],
            int_tuple![Int::<9>, Int::<1>],
        ),
        Layout::new(Int::<4>, e!(1))
    );
}

#[test]
fn test_logical_divide_const_eval() {
    type PairLayout<S0, S1, D0, D1> = Layout<Til<S0, Til<S1, Nil>>, Til<D0, Til<D1, Nil>>>;
    type DividedLayout = PairLayout<
        Int<4>,
        Til<Int<2>, Til<Int<3>, Nil>>,
        Int<2>,
        Til<Int<1>, Til<Int<8>, Nil>>,
    >;

    const A: Layout<Int<24>, Int<1>> = Layout::new(Int::<24>, Int::<1>);
    const B: Layout<Int<4>, Int<2>> = Layout::new(Int::<4>, Int::<2>);
    const LD: DividedLayout = logical_divide(A, B);
    const ZD: DividedLayout = zipped_divide(A, B);
    assert_eq!(LD, ZD);

    const PAIR: PairLayout<Int<6>, Int<4>, Int<4>, Int<1>> =
        Layout::new(int_tuple![Int::<6>, Int::<4>], int_tuple![Int::<4>, Int::<1>]);
    const {
        assert!(
            logical_divide_at::<mode![0], _, _>(PAIR, Layout::new(Int::<2>, Int::<1>))
                == PAIR.logical_divide_at::<mode![0], _>(Layout::new(Int::<2>, Int::<1>))
        );
    }
}

#[test]
fn test_logical_divide_dynamic_and_panics() {
    // Dynamic success via free functions and Layout methods
    let dyn_a = Layout::new(24i32, 1i32);
    let dyn_b = Layout::new(4i32, 2i32);
    let expected = Layout::new(
        int_tuple![4i32, (2i32, 3i32)],
        int_tuple![2i32, (1i32, 8i32)],
    );
    assert_eq!(logical_divide(dyn_a, dyn_b), expected);
    assert_eq!(dyn_a.logical_divide(dyn_b), expected);
    assert_eq!(zipped_divide(dyn_a, dyn_b), expected);
    assert_eq!(dyn_a.zipped_divide(dyn_b), expected);

    // Mode-targeted variants
    let mode_a = Layout::new(int_tuple![6i32, 4i32], int_tuple![4i32, 1i32]);
    let mode_b = Layout::new(2i32, 1i32);
    assert_eq!(
        logical_divide_at::<mode![0], _, _>(mode_a, mode_b),
        mode_a.logical_divide_at::<mode![0], _>(mode_b)
    );
    assert_eq!(
        zipped_divide_at::<mode![1], _, _>(mode_a, mode_b),
        mode_a.zipped_divide_at::<mode![1], _>(mode_b)
    );

    // Dynamic ceil-div padding in complement succeeds
    assert_eq!(
        logical_divide(Layout::new(5i32, 1i32), Layout::new(2i32, 1i32)),
        Layout::new(
            int_tuple![2i32, (1i32, 3i32)],
            int_tuple![1i32, (1i32, 2i32)],
        )
    );

    // Dynamic failure panics:
    // 1. Negative stride or zero-extent/non-zero-stride complement failure
    assert!(
        std::panic::catch_unwind(|| {
            logical_divide(Layout::new(8i32, 1i32), Layout::new(4i32, -2i32))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            zipped_divide(Layout::new(8i32, 1i32), Layout::new(0i32, 2i32))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            logical_divide_at::<mode![0], _, _>(
                Layout::new(int_tuple![8i32, 4i32], int_tuple![1i32, 8i32]),
                Layout::new(int_tuple![4i32, 2i32], int_tuple![1i32, 2i32]),
            )
        })
        .is_err()
    );
    // 2. Stride overflow in composition / complement
    assert!(
        std::panic::catch_unwind(|| {
            logical_divide(Layout::new(4i32, int::MAX), Layout::new(2i32, 2i32))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            zipped_divide_at::<mode![0], _, _>(
                Layout::new(int_tuple![4i32, 2i32], int_tuple![int::MAX, 1i32]),
                Layout::new(2i32, 2i32),
            )
        })
        .is_err()
    );
    // 3. Negative dynamic extent panics
    assert!(
        std::panic::catch_unwind(|| {
            logical_divide(Layout::new(-8i32, 1i32), Layout::new(4i32, 1i32))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            logical_divide(Layout::new(8i32, 1i32), Layout::new(-4i32, 1i32))
        })
        .is_err()
    );
}
