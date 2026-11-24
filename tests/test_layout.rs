#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl,
    const_cmp
)]
#![allow(incomplete_features)]

use core::cmp::Ordering;

use noodle_cute::basis::{BasisCoeff, BasisCombination, BasisTerm, make_basis_like};
use noodle_cute::integer::{Int, int};
use noodle_cute::layout::{
    Fraction, Layout, LayoutBase, MakeLayout, coprofile, coshape, is_layout, make_layout,
    make_layout_like, make_ordered_layout, recast, tiler_to_layout, tiler_to_layout_with,
};
use noodle_cute::tuple::{Congruent, Item, LeafProfile, Nil, TupleContainer, lift, select};
use noodle_cute::{e, int_tuple, make_layout, mode, slice, tuple, walk_mode};

const fn require_congruent<A: [const] Congruent<B>, B: Item>(_: &A, _: &B) {}

const fn check_layout_base<L: [const] LayoutBase>(l: &L) -> (L::Shape, L::Stride) {
    (l.shape(), l.stride())
}

#[test]
fn test_layout_and_is_layout() {
    let a = Layout::new(
        int_tuple![Int::<3>, (Int::<2>, Int::<4>)],
        int_tuple![Int::<2>, (Int::<1>, Int::<6>)],
    );

    assert!(is_layout(&a));
    assert!(!is_layout(&a.shape));
    assert!(!is_layout(&42i32));
    assert!(!is_layout(&Int::<42>));
    assert!(!is_layout(&Nil));
    assert!(!is_layout(&e![0]));
    let (sh, st) = check_layout_base(&a);
    assert_eq!(sh, a.shape);
    assert_eq!(st, a.stride);

    assert_eq!(a.size(), 24);
    assert!(!a.has_zero());
    assert_eq!(a.rank(), 2);
    assert_eq!(a.depth(), 2);
    assert_eq!(a.length(), 3);
    assert!(a.is_static());

    assert_eq!(a.get::<0>(), Layout::new(Int::<3>, Int::<2>));
    assert_eq!(
        a.get::<1>(),
        Layout::new(
            int_tuple![Int::<2>, Int::<4>],
            int_tuple![Int::<1>, Int::<6>]
        )
    );
    assert_eq!(walk_mode!(a, 1, 0), Layout::new(Int::<2>, Int::<1>));

    let rebuilt = make_layout!(
        Layout::new(Int::<3>, Int::<2>),
        make_layout!(
            Layout::new(Int::<2>, Int::<1>),
            Layout::new(Int::<4>, Int::<6>)
        )
    );
    assert_eq!(a, rebuilt);

    let expected_image: [int; 24] = [
        0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11, 12, 14, 16, 13, 15, 17, 18, 20, 22, 19, 21, 23,
    ];
    let s0 = a.get::<0>().size();
    let s1 = a.get::<1>().size();
    for i in 0..s0 {
        for j in 0..s1 {
            let idx = i + j * s0;
            assert_eq!(a(idx), expected_image[idx as usize]);
            assert_eq!(a(i, j), expected_image[idx as usize]);
        }
    }
}

#[test]
fn test_layout_construction_and_from_shape_with() {
    // Default compact column-major strides
    assert_eq!(
        Layout::from_shape(Int::<8>),
        Layout::new(Int::<8>, Int::<1>)
    );
    assert_eq!(
        Layout::from_shape(int_tuple![Int::<4>, Int::<8>]),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );
    assert_eq!(
        Layout::from_shape(int_tuple![Int::<3>, (Int::<2>, Int::<4>)]),
        Layout::new(
            int_tuple![Int::<3>, (Int::<2>, Int::<4>)],
            int_tuple![Int::<1>, (Int::<3>, Int::<6>)]
        )
    );

    // Integer base stride scaling
    assert_eq!(
        Layout::from_shape_with(int_tuple![Int::<4>, Int::<8>], Int::<2>),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<2>, Int::<8>]
        )
    );
    assert_eq!(
        Layout::from_shape_with(int_tuple![Int::<3>, (Int::<2>, Int::<4>)], Int::<5>),
        Layout::new(
            int_tuple![Int::<3>, (Int::<2>, Int::<4>)],
            int_tuple![Int::<5>, (Int::<15>, Int::<30>)]
        )
    );

    // Weakly-congruent tuple base stride
    assert_eq!(
        Layout::from_shape_with(
            int_tuple![(Int::<2>, Int::<3>), (Int::<4>, Int::<5>)],
            int_tuple![Int::<1>, Int::<100>],
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), (Int::<4>, Int::<5>)],
            int_tuple![(Int::<1>, Int::<2>), (Int::<100>, Int::<400>)]
        )
    );

    // Basis seed
    assert_eq!(
        Layout::from_shape_with(int_tuple![Int::<2>, Int::<3>], tuple![e![0], e![1]],),
        Layout::new(int_tuple![Int::<2>, Int::<3>], tuple![e![0], e![1]])
    );

    // Explicit stride & flatten & dynamic size
    let dyn_l = Layout::new(int_tuple![4i32, 8i32], int_tuple![8i32, 1i32]);
    assert!(!dyn_l.is_static());
    assert_eq!(dyn_l.stride(), int_tuple![8i32, 1i32]);
    assert_eq!(
        Layout::new(
            int_tuple![3i32, (2i32, 4i32)],
            int_tuple![24i32, (1i32, 6i32)]
        )
        .flatten(),
        Layout::new(int_tuple![3i32, 2i32, 4i32], int_tuple![24i32, 1i32, 6i32])
    );

    let overflow_l = Layout::new(int_tuple![0x1_0000i32, 0x1_0000i32], int_tuple![1i32, 1i32]);
    assert!(std::panic::catch_unwind(|| overflow_l.size()).is_err());
    let zero_l = Layout::new(int_tuple![0i32, 8i32], int_tuple![1i32, 1i32]);
    assert!(zero_l.has_zero());
    assert_eq!(zero_l.size(), 0);

    // Compile-time constant construction
    type ConstL = Layout<
        TupleContainer<Int<3>, TupleContainer<Int<4>, Nil>>,
        TupleContainer<Int<1>, TupleContainer<Int<3>, Nil>>,
    >;
    const CONST_L: ConstL = Layout::from_shape(int_tuple![Int::<3>, Int::<4>]);
    assert_eq!(CONST_L.shape, int_tuple![Int::<3>, Int::<4>]);
    assert_eq!(CONST_L.stride, int_tuple![Int::<1>, Int::<3>]);
}

#[test]
fn test_three_coordinate_forms_and_const_eval() {
    let a = Layout::new(
        int_tuple![Int::<3>, (Int::<2>, Int::<4>)],
        int_tuple![Int::<2>, (Int::<1>, Int::<6>)],
    );

    for i in 0..a.size() {
        let crd_2d = int_tuple![i % 3, i / 3];
        let crd_nat = int_tuple![i % 3, ((i / 3) % 2, (i / 3) / 2)];
        assert_eq!(a(i), a.call(crd_2d));
        assert_eq!(a(i), a(crd_2d));
        assert_eq!(a(i), a(crd_nat));
        assert_eq!(a(i), a(i % 3, int_tuple![(i / 3) % 2, (i / 3) / 2]));
    }

    assert_eq!(
        a(int_tuple![2i32, (1i32, 2i32)]),
        a(2i32, int_tuple![1i32, 2i32])
    );

    // Out-of-bounds extended-domain coordinate
    assert_eq!(a(100i32), 99);

    // Multi-arg up to 6 modes and const evaluation
    let l4 = Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<2>, Int::<2>]);
    assert_eq!(l4(1i32, 2i32, 0i32, 1i32), 1 + 2 * 2 + 12);

    type StaticInnerShape = TupleContainer<Int<2>, TupleContainer<Int<4>, Nil>>;
    type StaticInnerStride = TupleContainer<Int<1>, TupleContainer<Int<6>, Nil>>;
    type StaticA = Layout<
        TupleContainer<Int<3>, TupleContainer<StaticInnerShape, Nil>>,
        TupleContainer<Int<2>, TupleContainer<StaticInnerStride, Nil>>,
    >;
    const STATIC_A: StaticA = Layout::new(
        int_tuple![Int::<3>, (Int::<2>, Int::<4>)],
        int_tuple![Int::<2>, (Int::<1>, Int::<6>)],
    );
    const STATIC_V1: Int<17> = STATIC_A(Int::<17>);
    const STATIC_V2: Int<17> = STATIC_A(Int::<2>, Int::<5>);
    const STATIC_V3: Int<17> = STATIC_A(Int::<2>, int_tuple![Int::<1>, Int::<2>]);
    assert_eq!(STATIC_V1, Int::<17>);
    assert_eq!(STATIC_V2, Int::<17>);
    assert_eq!(STATIC_V3, Int::<17>);
}

#[test]
#[allow(clippy::reversed_empty_ranges)]
fn test_layout_mode_indexing_and_slicing() {
    let a = Layout::new(
        int_tuple![Int::<2>, Int::<3>, Int::<5>, Int::<7>],
        int_tuple![Int::<1>, Int::<2>, Int::<6>, Int::<30>],
    );

    // Single mode indexing unwraps
    assert_eq!(a.get::<1>(), Layout::new(Int::<3>, Int::<2>));
    assert_eq!(a.get::<-1>(), Layout::new(Int::<7>, Int::<30>));
    let nested = Layout::new(
        int_tuple![Int::<2>, (Int::<3>, Int::<5>)],
        int_tuple![Int::<1>, (Int::<2>, Int::<6>)],
    );
    assert_eq!(
        nested.get::<-1>(),
        Layout::new(
            int_tuple![Int::<3>, Int::<5>],
            int_tuple![Int::<2>, Int::<6>]
        )
    );

    // Slicing keeps modes wrapped
    assert_eq!(
        a.slice::<{ slice!([1..-1]) }>(),
        Layout::new(
            int_tuple![Int::<3>, Int::<5>],
            int_tuple![Int::<2>, Int::<6>]
        )
    );
    assert_eq!(
        a.slice::<{ slice!([1..]) }>(),
        Layout::new(
            int_tuple![Int::<3>, Int::<5>, Int::<7>],
            int_tuple![Int::<2>, Int::<6>, Int::<30>]
        )
    );
    assert_eq!(
        a.slice::<{ slice!([..2]) }>(),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<1>, Int::<2>]
        )
    );
    assert_eq!(
        a.slice::<{ slice!([1..2]) }>(),
        Layout::new(int_tuple![Int::<3>], int_tuple![Int::<2>])
    );

    // Clamped and empty slices
    assert_eq!(
        a.slice::<{ slice!([1..99]) }>(),
        a.slice::<{ slice!([1..]) }>()
    );
    assert_eq!(a.slice::<{ slice!([2..2]) }>(), make_layout!());
    assert_eq!(a.slice::<{ slice!([9..]) }>(), make_layout!());

    // Step and reverse step
    assert_eq!(
        a.slice::<{ slice!([..]:2) }>(),
        Layout::new(
            int_tuple![Int::<2>, Int::<5>],
            int_tuple![Int::<1>, Int::<6>]
        )
    );
    assert_eq!(
        a.slice::<{ slice!([..]:-1) }>(),
        Layout::new(
            int_tuple![Int::<7>, Int::<5>, Int::<3>, Int::<2>],
            int_tuple![Int::<30>, Int::<6>, Int::<2>, Int::<1>]
        )
    );

    // Integer shape indexes as a 1-tuple
    let l = Layout::new(Int::<8>, Int::<1>);
    assert_eq!(l.rank(), 1);
    assert_eq!(l.get::<0>(), l);
    assert_eq!(l.get::<-1>(), l);
    assert_eq!(
        l.slice::<{ slice!([0..1]) }>(),
        Layout::new(int_tuple![Int::<8>], int_tuple![Int::<1>])
    );
    assert_eq!(l.slice::<{ slice!([1..]) }>(), make_layout!());

    // Select + make_layout agrees with arbitrary mode selection
    let sel_1_3 = make_layout(select::<mode![1, 3], _>(&a));
    assert_eq!(
        sel_1_3,
        Layout::new(
            int_tuple![Int::<3>, Int::<7>],
            int_tuple![Int::<2>, Int::<30>]
        )
    );
    let sel_3_1 = make_layout(select::<mode![3, 1], _>(&a));
    assert_eq!(
        sel_3_1,
        Layout::new(
            int_tuple![Int::<7>, Int::<3>],
            int_tuple![Int::<30>, Int::<2>]
        )
    );
}

#[test]
fn test_layout_equality_and_display() {
    // Structural equality across static & dynamic scalar types
    assert_eq!(
        Layout::from_shape(int_tuple![Int::<3>, Int::<4>]),
        Layout::new(int_tuple![3i32, 4i32], int_tuple![1i32, 3i32])
    );
    assert_ne!(
        Layout::from_shape(int_tuple![Int::<3>, Int::<4>]),
        Layout::from_shape(int_tuple![Int::<4>, Int::<3>])
    );
    assert_ne!(
        Layout::new(
            int_tuple![Int::<3>, Int::<4>],
            int_tuple![Int::<1>, Int::<3>]
        ),
        Layout::new(
            int_tuple![Int::<3>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>]
        )
    );

    // Functionally equivalent layouts differ structurally
    let a = Layout::new(
        int_tuple![Int::<4>, Int::<8>],
        int_tuple![Int::<1>, Int::<4>],
    );
    let b = Layout::new(Int::<32>, Int::<1>);
    assert_ne!(a, b);
    for i in 0..32i32 {
        assert_eq!(a(i), b(i));
    }

    // Basis equality across layouts
    assert_eq!(
        Layout::new(int_tuple![Int::<4>, Int::<5>], tuple![e![0], e![1]]),
        Layout::new(int_tuple![4i32, 5i32], tuple![1i32 * e![0], 1i32 * e![1]])
    );
    assert_ne!(Layout::new(Int::<4>, e![0]), Layout::new(Int::<4>, e![1]));
    assert_ne!(
        Layout::new(Int::<4>, e![0]),
        Layout::new(Int::<4>, Int::<1>)
    );

    // Display formatting
    let nested = Layout::new(
        int_tuple![Int::<3>, (Int::<2>, Int::<4>)],
        int_tuple![Int::<2>, (Int::<1>, Int::<6>)],
    );
    assert_eq!(format!("{nested}"), "(3, (2, 4)):(2, (1, 6))");
    assert_eq!(format!("{}", make_layout!()), "():()");
    assert_eq!(
        format!(
            "{}",
            Layout::new(int_tuple![Int::<8>], int_tuple![Int::<1>])
        ),
        "(8,):(1,)"
    );
    assert_eq!(
        format!(
            "{}",
            Layout::new(
                int_tuple![Int::<4>, Int::<5>],
                tuple![Int::<2> * e![0], Int::<3> * e![1, 0]]
            )
        ),
        "(4, 5):(2@0, 3@(1, 0))"
    );
}

#[test]
fn test_coshape_cosize_and_coprofile() {
    let l_col = Layout::new(
        int_tuple![Int::<4>, Int::<8>],
        int_tuple![Int::<1>, Int::<4>],
    );
    assert_eq!(l_col.coshape(), Int::<32>);
    assert_eq!(coshape(&l_col), Int::<32>);
    assert_eq!(l_col.cosize(), 32);
    require_congruent(&coprofile(&l_col), &coshape(&l_col));

    let l_row = Layout::new(
        int_tuple![Int::<4>, Int::<8>],
        int_tuple![Int::<8>, Int::<1>],
    );
    assert_eq!(coshape(&l_row), Int::<32>);
    assert_eq!(l_row.cosize(), 32);

    let l_bcast = Layout::new(Int::<8>, Int::<0>);
    assert_eq!(coshape(&l_bcast), Int::<1>);
    assert_eq!(l_bcast.cosize(), 1);
    require_congruent(&coprofile(&l_bcast), &coshape(&l_bcast));

    // Coordinate basis strides
    let l_e01 = Layout::new(int_tuple![Int::<4>, Int::<8>], tuple![e![0], e![1]]);
    assert_eq!(coshape(&l_e01), int_tuple![Int::<4>, Int::<8>]);
    assert_eq!(l_e01.cosize(), 32);
    require_congruent(&coprofile(&l_e01), &coshape(&l_e01));

    let l_e10 = Layout::new(int_tuple![Int::<4>, Int::<8>], tuple![e![1], e![0]]);
    assert_eq!(coshape(&l_e10), int_tuple![Int::<8>, Int::<4>]);
    require_congruent(&coprofile(&l_e10), &coshape(&l_e10));

    let l_nested_basis = Layout::new(int_tuple![Int::<4>, Int::<5>], tuple![e![1], e![4, 1]]);
    require_congruent(&coprofile(&l_nested_basis), &coshape(&l_nested_basis));

    // Custom non-scalar BasisCoeff (XorF2) supports coprofile without extent arithmetic
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct XorF2(u8);

    const impl Item for XorF2 {
        const RANK: int = 1;
        const DEPTH: int = 0;
        const LEAVES: int = 1;
        type Profile = LeafProfile;
    }

    const impl BasisCoeff for XorF2 {
        type Unit = XorF2;
        fn is_zero(&self) -> bool {
            self.0 == 0
        }
        fn cmp_zero(&self) -> Ordering {
            self.0.cmp(&0)
        }
        fn static_cmp_zero(&self) -> Option<Ordering> {
            None
        }
        fn algebra_unit() -> Self::Unit {
            XorF2(1)
        }
    }

    let l_xor = Layout::new(
        int_tuple![Int::<8>, Int::<8>],
        tuple![
            noodle_cute::basis::ScaledBasis::<XorF2, mode![0]>::new(XorF2(1)),
            e![1]
        ],
    );
    let cp_xor = coprofile(&l_xor);
    require_congruent(&cp_xor, &int_tuple![0i32, 0i32]);
    assert_eq!(make_basis_like(cp_xor), tuple![e![0], e![1]]);

    // Coprofile of tilers agrees with coprofile(tiler_to_layout(tiler))
    assert_eq!(coprofile(&Int::<24>), Int::<0>);
    let t1 = int_tuple![Int::<4>, Int::<8>];
    require_congruent(&coprofile(&t1), &coprofile(&tiler_to_layout(t1)));
    let t2 = int_tuple![(Int::<2>, Int::<3>), Int::<4>];
    require_congruent(&coprofile(&t2), &coprofile(&tiler_to_layout(t2)));
    let t3 = tuple![
        Layout::new(Int::<4>, Int::<2>),
        Layout::new(Int::<5>, Int::<3>)
    ];
    require_congruent(&coprofile(&t3), &coprofile(&tiler_to_layout(t3)));
    let t4 = tuple![
        Layout::new(
            int_tuple![Int::<2>, Int::<2>],
            int_tuple![Int::<1>, Int::<4>]
        ),
        Int::<8>
    ];
    require_congruent(&coprofile(&t4), &coprofile(&tiler_to_layout(t4)));
}

#[test]
fn test_make_layout_and_tiler_to_layout() {
    let empty = make_layout!();
    assert_eq!(empty.shape, Nil);
    assert_eq!(empty.stride, Nil);
    assert_eq!(empty.rank(), 0);
    assert_eq!(empty.size(), 1);

    let l1 = Layout::new(Int::<4>, Int::<1>);
    let single = make_layout!(l1);
    assert_eq!(single.shape, int_tuple![Int::<4>]);
    assert_eq!(single.stride, int_tuple![Int::<1>]);

    let a = Layout::new(
        int_tuple![Int::<3>, (Int::<2>, Int::<4>)],
        int_tuple![Int::<2>, (Int::<1>, Int::<6>)],
    );
    let round_trip = make_layout!(
        a.get::<0>(),
        make_layout!(a.get::<1>().get::<0>(), a.get::<1>().get::<1>())
    );
    assert_eq!(a, round_trip);

    let with_empty = make_layout!(
        Layout::new(
            int_tuple![Int::<3>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>]
        ),
        make_layout!()
    );
    assert_eq!(with_empty.shape, tuple![(Int::<3>, Int::<4>), Nil]);
    for c in 0..12i32 {
        assert_eq!(
            with_empty(c, Nil),
            Layout::new(
                int_tuple![Int::<3>, Int::<4>],
                int_tuple![Int::<4>, Int::<1>]
            )(c)
        );
    }

    // Lift with MakeLayout
    let lifted = lift::<mode![1], _, _, _>(
        &Layout::new(Int::<4>, Int::<2>),
        &MakeLayout,
        Layout::new(Int::<1>, Int::<0>),
    );
    assert_eq!(
        lifted,
        Layout::new(
            int_tuple![Int::<1>, Int::<4>],
            int_tuple![Int::<0>, Int::<2>]
        )
    );

    // tiler_to_layout
    assert_eq!(tiler_to_layout(Int::<3>), Layout::new(Int::<3>, Int::<1>));
    let lt = Layout::new(
        int_tuple![Int::<7>, Int::<2>],
        int_tuple![Int::<3>, Int::<1>],
    );
    assert_eq!(tiler_to_layout(lt), lt);
    assert_eq!(
        tiler_to_layout(int_tuple![Int::<4>, Int::<5>]),
        Layout::new(int_tuple![Int::<4>, Int::<5>], tuple![e![0], e![1]])
    );
    assert_eq!(
        tiler_to_layout(int_tuple![Int::<2>, Int::<3>, Int::<5>]),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            tuple![e![0], e![1], e![2]]
        )
    );
    assert_eq!(
        tiler_to_layout(tuple![
            Layout::new(Int::<4>, Int::<2>),
            Layout::new(Int::<5>, Int::<3>)
        ]),
        Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            tuple![Int::<2> * e![0], Int::<3> * e![1]]
        )
    );
    assert_eq!(
        tiler_to_layout_with(int_tuple![Int::<4>, Int::<5>], Int::<2>),
        Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            tuple![Int::<2> * e![0], Int::<2> * e![1]]
        )
    );

    // Nested tiler tuple
    let nested_tiler = tuple![(Int::<2>, Int::<3>), Layout::new(Int::<4>, Int::<2>)];
    assert_eq!(
        tiler_to_layout(nested_tiler),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<4>],
            tuple![(e![0, 0], e![0, 1]), Int::<2> * e![1]]
        )
    );
}

#[test]
fn test_make_layout_like() {
    // Already compact
    assert_eq!(
        make_layout_like(&Layout::new(Int::<8>, Int::<1>)),
        Layout::new(Int::<8>, Int::<1>)
    );
    assert_eq!(
        make_layout_like(&Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>]
        )),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );
    assert_eq!(
        make_layout_like(&Layout::new(
            int_tuple![Int::<3>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>]
        )),
        Layout::new(
            int_tuple![Int::<3>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>]
        )
    );

    // Reordered strides
    assert_eq!(
        make_layout_like(&Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<100>, Int::<1>]
        )),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>]
        )
    );
    assert_eq!(
        make_layout_like(&Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            int_tuple![Int::<1>, Int::<100>, Int::<20>]
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            int_tuple![Int::<1>, Int::<8>, Int::<2>]
        )
    );

    // Documented examples + stride-0 + size-1
    assert_eq!(
        make_layout_like(&Layout::new(
            int_tuple![Int::<2>, Int::<2>, Int::<2>, Int::<2>],
            int_tuple![Int::<0>, Int::<2>, Int::<4>, Int::<1>]
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<2>, Int::<2>, Int::<2>],
            int_tuple![Int::<0>, Int::<2>, Int::<4>, Int::<1>]
        )
    );
    assert_eq!(
        make_layout_like(&Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>, Int::<5>],
            int_tuple![Int::<0>, Int::<42>, Int::<1>, Int::<0>]
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>, Int::<5>],
            int_tuple![Int::<0>, Int::<4>, Int::<1>, Int::<0>]
        )
    );
    assert_eq!(
        make_layout_like(&Layout::new(
            int_tuple![Int::<1>, Int::<4>],
            int_tuple![Int::<7>, Int::<2>]
        )),
        Layout::new(
            int_tuple![Int::<1>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>]
        )
    );
    assert_eq!(
        make_layout_like(&Layout::new(
            int_tuple![Int::<4>, Int::<1>, Int::<8>],
            int_tuple![Int::<1>, Int::<5>, Int::<4>]
        )),
        Layout::new(
            int_tuple![Int::<4>, Int::<1>, Int::<8>],
            int_tuple![Int::<1>, Int::<32>, Int::<4>]
        )
    );
    assert_eq!(
        make_layout_like(&Layout::new(
            int_tuple![Int::<1>, Int::<1>],
            int_tuple![Int::<5>, Int::<7>]
        )),
        Layout::new(
            int_tuple![Int::<1>, Int::<1>],
            int_tuple![Int::<1>, Int::<1>]
        )
    );

    // Nested shapes
    assert_eq!(
        make_layout_like(&Layout::new(
            int_tuple![Int::<2>, (Int::<3>, Int::<4>)],
            int_tuple![Int::<50>, (Int::<1>, Int::<8>)]
        )),
        Layout::new(
            int_tuple![Int::<2>, (Int::<3>, Int::<4>)],
            int_tuple![Int::<12>, (Int::<1>, Int::<3>)]
        )
    );

    // Dynamic / symbolic strides
    let n_dyn = 5i32;
    let x_dyn = 99i32;
    let y_dyn = 7i32;
    assert_eq!(
        make_layout_like(&Layout::new(
            tuple![n_dyn, Int::<4>],
            tuple![Int::<1>, n_dyn]
        )),
        Layout::new(tuple![n_dyn, Int::<4>], tuple![Int::<1>, 5i32])
    );
    assert_eq!(
        make_layout_like(&Layout::new(
            tuple![Int::<2>, n_dyn, Int::<4>],
            tuple![Int::<1>, x_dyn, Int::<2>]
        )),
        Layout::new(
            tuple![Int::<2>, n_dyn, Int::<4>],
            tuple![Int::<1>, Int::<8>, Int::<2>]
        )
    );
    assert_eq!(
        make_layout_like(&Layout::new(tuple![n_dyn, 6i32], tuple![x_dyn, y_dyn])),
        Layout::new(tuple![n_dyn, 6i32], tuple![Int::<1>, 5i32])
    );

    // Coordinate basis strides
    assert_eq!(
        make_layout_like(&Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            tuple![e![0], e![1], e![2]]
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>]
        )
    );
    assert_eq!(
        make_layout_like(&Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            tuple![e![2], e![1], e![0]]
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            int_tuple![Int::<12>, Int::<4>, Int::<1>]
        )
    );
    assert_eq!(
        make_layout_like(&Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            tuple![e![1], e![2], e![0]]
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            int_tuple![Int::<4>, Int::<8>, Int::<1>]
        )
    );
    assert_eq!(
        make_layout_like(&Layout::new(
            int_tuple![Int::<4>, Int::<3>],
            tuple![Int::<3> * e![0], Int::<1> * e![0]]
        )),
        Layout::new(
            int_tuple![Int::<4>, Int::<3>],
            int_tuple![Int::<3>, Int::<1>]
        )
    );
    assert_eq!(
        make_layout_like(&Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            tuple![e![0], Int::<0>, e![1]]
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            int_tuple![Int::<1>, Int::<0>, Int::<2>]
        )
    );
    assert_eq!(
        make_layout_like(&Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            tuple![e![0, 0], e![0, 1], e![1]]
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>]
        )
    );
}

#[test]
fn test_make_ordered_layout() {
    assert_eq!(
        make_ordered_layout(&Int::<8>, &Int::<0>),
        Layout::new(Int::<8>, Int::<1>)
    );
    assert_eq!(
        make_ordered_layout(
            &int_tuple![Int::<4>, Int::<8>],
            &int_tuple![Int::<0>, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );
    assert_eq!(
        make_ordered_layout(
            &int_tuple![Int::<4>, Int::<8>],
            &int_tuple![Int::<1>, Int::<0>]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>]
        )
    );

    // Documented example and permutations
    assert_eq!(
        make_ordered_layout(
            &int_tuple![Int::<2>, Int::<2>, Int::<2>, Int::<2>],
            &int_tuple![Int::<0>, Int::<2>, Int::<3>, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<2>, Int::<2>, Int::<2>],
            int_tuple![Int::<1>, Int::<4>, Int::<8>, Int::<2>]
        )
    );
    assert_eq!(
        make_ordered_layout(
            &int_tuple![Int::<2>, Int::<3>, Int::<4>, Int::<5>],
            &int_tuple![Int::<2>, Int::<67>, Int::<42>, Int::<50>]
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>, Int::<5>],
            int_tuple![Int::<1>, Int::<40>, Int::<2>, Int::<8>]
        )
    );

    // Ties break by left-to-right position
    assert_eq!(
        make_ordered_layout(
            &int_tuple![Int::<2>, Int::<3>, Int::<4>, Int::<2>],
            &int_tuple![Int::<0>, Int::<2>, Int::<3>, Int::<0>]
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<4>, Int::<12>, Int::<2>]
        )
    );

    // Nested shapes and size-1 modes
    assert_eq!(
        make_ordered_layout(
            &int_tuple![Int::<2>, (Int::<3>, Int::<4>)],
            &int_tuple![Int::<2>, (Int::<1>, Int::<0>)]
        ),
        Layout::new(
            int_tuple![Int::<2>, (Int::<3>, Int::<4>)],
            int_tuple![Int::<12>, (Int::<4>, Int::<1>)]
        )
    );
    assert_eq!(
        make_ordered_layout(
            &int_tuple![Int::<4>, Int::<1>, Int::<8>],
            &int_tuple![Int::<0>, Int::<1>, Int::<2>]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<1>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>, Int::<4>]
        )
    );

    // Dynamic orders sort after static orders and preserve left-to-right order
    let p_dyn = 67i32;
    let q_dyn = 42i32;
    assert_eq!(
        make_ordered_layout(
            &int_tuple![Int::<4>, Int::<8>],
            &int_tuple![Int::<1>, p_dyn]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );
    assert_eq!(
        make_ordered_layout(
            &int_tuple![Int::<4>, Int::<8>],
            &int_tuple![p_dyn, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>]
        )
    );
    assert_eq!(
        make_ordered_layout(
            &int_tuple![Int::<2>, Int::<3>, Int::<4>, Int::<5>],
            &int_tuple![Int::<2>, p_dyn, q_dyn, Int::<50>]
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>, Int::<5>],
            int_tuple![Int::<1>, Int::<10>, Int::<30>, Int::<2>]
        )
    );
    assert_eq!(
        make_ordered_layout(
            &int_tuple![Int::<2>, Int::<3>, Int::<4>],
            &int_tuple![p_dyn, Int::<0>, q_dyn]
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            int_tuple![Int::<3>, Int::<1>, Int::<6>]
        )
    );

    // Compile-time constant ordered layout
    type ConstOrdered = Layout<
        TupleContainer<Int<4>, TupleContainer<Int<8>, Nil>>,
        TupleContainer<Int<8>, TupleContainer<Int<1>, Nil>>,
    >;
    const CONST_ORDERED: ConstOrdered = make_ordered_layout(
        &int_tuple![Int::<4>, Int::<8>],
        &int_tuple![Int::<1>, Int::<0>],
    );
    assert_eq!(CONST_ORDERED.stride, int_tuple![Int::<8>, Int::<1>]);
}

#[test]
fn test_recast() {
    // 1D contiguous stride-1
    let l1 = Layout::new(Int::<24>, Int::<1>);
    assert_eq!(recast(&l1, Int::<8>), Layout::new(Int::<3>, Int::<1>));
    assert_eq!(recast(&l1, Int::<6>), Layout::new(Int::<4>, Int::<1>));
    assert_eq!(recast(&l1, Int::<4>), Layout::new(Int::<6>, Int::<1>));
    assert_eq!(recast(&l1, Int::<2>), Layout::new(Int::<12>, Int::<1>));
    assert_eq!(recast(&l1, Int::<1>), Layout::new(Int::<24>, Int::<1>));
    assert_eq!(
        recast(&l1, Fraction::new(Int::<1>, Int::<2>)),
        Layout::new(Int::<48>, Int::<1>)
    );
    assert_eq!(
        recast(&l1, Fraction::new(Int::<1>, Int::<8>)),
        Layout::new(Int::<192>, Int::<1>)
    );

    // 1D stride-2
    let l2 = Layout::new(Int::<24>, Int::<2>);
    assert_eq!(recast(&l2, Int::<8>), Layout::new(Int::<6>, Int::<1>));
    assert_eq!(recast(&l2, Int::<6>), Layout::new(Int::<8>, Int::<1>));
    assert_eq!(recast(&l2, Int::<4>), Layout::new(Int::<12>, Int::<1>));
    assert_eq!(recast(&l2, Int::<2>), Layout::new(Int::<24>, Int::<1>));
    assert_eq!(recast(&l2, Int::<1>), Layout::new(Int::<24>, Int::<2>));
    assert_eq!(
        recast(&l2, Fraction::new(Int::<1>, Int::<2>)),
        Layout::new(Int::<24>, Int::<4>)
    );
    assert_eq!(
        recast(&l2, Fraction::new(Int::<1>, Int::<8>)),
        Layout::new(Int::<24>, Int::<16>)
    );

    // 2D row-major & 3D with stride-0
    let l_2d = Layout::new(
        int_tuple![Int::<4>, Int::<4>],
        int_tuple![Int::<4>, Int::<1>],
    );
    assert_eq!(
        recast(&l_2d, Int::<8>),
        Layout::new(
            int_tuple![Int::<2>, Int::<1>],
            int_tuple![Int::<1>, Int::<1>]
        )
    );
    assert_eq!(
        recast(&l_2d, Int::<4>),
        Layout::new(
            int_tuple![Int::<4>, Int::<1>],
            int_tuple![Int::<1>, Int::<1>]
        )
    );
    assert_eq!(
        recast(&l_2d, Fraction::new(Int::<1>, Int::<2>)),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>]
        )
    );

    let l_3d = Layout::new(
        int_tuple![Int::<8>, Int::<4>, Int::<6>],
        int_tuple![Int::<1>, Int::<0>, Int::<2>],
    );
    assert_eq!(
        recast(&l_3d, Int::<8>),
        Layout::new(
            int_tuple![Int::<1>, Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<0>, Int::<1>]
        )
    );
    assert_eq!(
        recast(&l_3d, Fraction::new(Int::<1>, Int::<2>)),
        Layout::new(
            int_tuple![Int::<16>, Int::<4>, Int::<6>],
            int_tuple![Int::<1>, Int::<0>, Int::<4>]
        )
    );

    // Nested shape
    let l_nested = Layout::new(
        int_tuple![(Int::<4>, Int::<6>), Int::<8>],
        int_tuple![(Int::<1>, Int::<4>), Int::<24>],
    );
    assert_eq!(
        recast(&l_nested, Int::<8>),
        Layout::new(
            int_tuple![(Int::<1>, Int::<3>), Int::<8>],
            int_tuple![(Int::<1>, Int::<1>), Int::<3>]
        )
    );
    assert_eq!(
        recast(&l_nested, Fraction::new(Int::<1>, Int::<2>)),
        Layout::new(
            int_tuple![(Int::<8>, Int::<6>), Int::<8>],
            int_tuple![(Int::<1>, Int::<8>), Int::<48>]
        )
    );

    // BasisTerm stride & BasisCombination<Nil> stride
    let l_basis = Layout::new(Int::<8>, e![0]);
    assert_eq!(
        recast(&l_basis, Int::<4>),
        Layout::new(Int::<2>, BasisTerm::<Int<1>, mode![0]>::new(Int::<1>))
    );
    let l_zero_comb = Layout::new(Int::<8>, BasisCombination::<Nil>::zero());
    assert_eq!(
        recast(&l_zero_comb, Int::<4>),
        Layout::new(Int::<8>, BasisCombination::<Nil>::zero())
    );

    // Compile-time const recast
    const RECAST_STATIC: Layout<Int<3>, Int<1>> =
        recast(&Layout::new(Int::<24>, Int::<1>), Int::<8>);
    assert_eq!(RECAST_STATIC, Layout::new(Int::<3>, Int::<1>));

    // Dynamic recast panicking on divisibility failure, invalid scale, or overflow
    let l_dyn = Layout::new(24i32, 2i32);
    assert_eq!(recast(&l_dyn, 4i32), Layout::new(12i32, 1i32));
    assert!(std::panic::catch_unwind(|| recast(&l_dyn, 3i32)).is_err());
    assert!(std::panic::catch_unwind(|| recast(&l_dyn, 0i32)).is_err());
    assert!(std::panic::catch_unwind(|| recast(&l_dyn, -2i32)).is_err());
    assert!(std::panic::catch_unwind(|| recast(&l_dyn, Fraction::new(1i32, 0i32))).is_err());
    assert!(
        std::panic::catch_unwind(|| {
            recast(&Layout::new(i32::MAX, 1i32), Fraction::new(1i32, 2i32))
        })
        .is_err()
    );
}

#[test]
fn layout_regression_recast_tiler_nil_zero_basis_eq_and_integration() {
    // 1. B8: Negative stride where dd * d == -n and extreme i32::MIN
    assert_eq!(
        recast(&Layout::new(4i32, -2i32), 2i32),
        Layout::new(4i32, -1i32)
    );
    assert_eq!(
        recast(&Layout::new(Int::<4>, Int::<-2>), Int::<2>),
        Layout::new(Int::<4>, Int::<-1>)
    );
    assert_eq!(
        recast(&Layout::new(4i32, -4i32), 2i32),
        Layout::new(4i32, -2i32)
    );
    assert_eq!(
        recast(&Layout::new(4i32, -1i32), Fraction::new(1i32, 2i32)),
        Layout::new(4i32, -2i32)
    );
    assert!(std::panic::catch_unwind(|| recast(&Layout::new(4i32, -1i32), 2i32)).is_err());
    assert!(std::panic::catch_unwind(|| recast(&Layout::new(4i32, -2i32), 4i32)).is_err());
    assert_eq!(
        recast(&Layout::new(4i32, -4i32), Fraction::new(2i32, 3i32)),
        Layout::new(4i32, -6i32)
    );
    assert_eq!(
        recast(&Layout::new(4i32, i32::MIN), 2i32),
        Layout::new(4i32, i32::MIN / 2)
    );
    assert_eq!(
        recast(&Layout::new(4i32, i32::MIN), 1i32),
        Layout::new(4i32, i32::MIN)
    );
    assert!(
        std::panic::catch_unwind(|| {
            recast(&Layout::new(4i32, i32::MIN), Fraction::new(1i32, 2i32))
        })
        .is_err()
    );

    // B8b: Reduce gcd(|dd|, n) first to avoid spurious dd * d overflow
    assert_eq!(
        recast(
            &Layout::new(4i32, 2_000_000_000i32),
            Fraction::new(2_000_000_000i32, 3i32)
        ),
        Layout::new(4i32, 3i32)
    );
    assert_eq!(
        recast(
            &Layout::new(Int::<4>, Int::<2_000_000_000>),
            Fraction::new(Int::<2_000_000_000>, Int::<3>)
        ),
        Layout::new(Int::<4>, Int::<3>)
    );

    // B8c: Large power-of-two scale numerator divisible by positive stride
    assert_eq!(
        recast(&Layout::new(4i32, 2i32), 1i32 << 30),
        Layout::new(1i32, 1i32)
    );
    assert_eq!(
        recast(&Layout::new(i32::MAX, 2i32), 1i32 << 30),
        Layout::new(4i32, 1i32)
    );

    // 2. M7: Tiler<E> for Nil with arbitrary scaling factor E
    assert_eq!(tiler_to_layout_with(Nil, Int::<2>), Layout::new(Nil, Nil));
    assert_eq!(tiler_to_layout_with(Nil, e![0]), Layout::new(Nil, Nil));

    // 3. M7: LayoutItemEq aligns BasisCombination<Nil> with scalar zero
    assert_eq!(
        Layout::new(Int::<4>, Int::<0> * e![0]),
        Layout::new(Int::<4>, Int::<0>)
    );
    assert_eq!(
        Layout::new(Int::<4>, Int::<0>),
        Layout::new(Int::<4>, BasisCombination::<Nil>::zero())
    );
    assert_eq!(
        Layout::new(Int::<4>, BasisCombination::<Nil>::zero()),
        Layout::new(Int::<4>, 0i32)
    );
    assert_ne!(
        Layout::new(Int::<4>, BasisCombination::<Nil>::zero()),
        Layout::new(Int::<4>, Int::<1>)
    );
    assert_ne!(
        Layout::new(Int::<4>, BasisCombination::<Nil>::zero()),
        Layout::new(Int::<4>, 1i32)
    );

    // 4. Cross-layer integration: trailing Nil shape & non-overflowing Layout::from_shape
    let l_trailing_nil = Layout::new(tuple![Int::<4>, Nil], tuple![Int::<1>, Nil]);
    assert_eq!(l_trailing_nil(5i32), 5i32);

    let l_large = Layout::from_shape(int_tuple![Int::<2>, Int::<2_000_000_000>]);
    assert_eq!(
        l_large,
        Layout::new(
            int_tuple![Int::<2>, Int::<2_000_000_000>],
            int_tuple![Int::<1>, Int::<2>]
        )
    );
}
