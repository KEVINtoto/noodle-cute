#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use noodle_cute::algebra::composition::{Composition, composition};
use noodle_cute::basis::basis_eq;
use noodle_cute::integer::{Int, int};
use noodle_cute::layout::{Layout, tiler_to_layout};
use noodle_cute::shape::compatible;
use noodle_cute::tuple::Nil;
use noodle_cute::{e, int_tuple, make_layout, tuple};

#[test]
fn test_composition_rank1_base() {
    // Layout(20, 2) o Layout(5, 4) -> Layout(5, 8)
    let a = Layout::new(Int::<20>, Int::<2>);
    let b = Layout::new(Int::<5>, Int::<4>);
    assert_eq!(composition(a, b), Layout::new(Int::<5>, Int::<8>));
}

#[test]
fn test_composition_rhs_distributive() {
    // Layout(20, 2) o Layout((5, 4), (4, 1)) -> Layout((5, 4), (8, 2))
    let a = Layout::new(Int::<20>, Int::<2>);
    let b = Layout::new(
        int_tuple![Int::<5>, Int::<4>],
        int_tuple![Int::<4>, Int::<1>],
    );
    assert_eq!(
        composition(a, b),
        Layout::new(
            int_tuple![Int::<5>, Int::<4>],
            int_tuple![Int::<8>, Int::<2>],
        )
    );
}

#[test]
fn test_composition_precoalesced_lhs() {
    // Layout((4, 3), (1, 4)) o Layout(6, 2) -> Layout(6, 2)
    let a = Layout::new(
        int_tuple![Int::<4>, Int::<3>],
        int_tuple![Int::<1>, Int::<4>],
    );
    let b = Layout::new(Int::<6>, Int::<2>);
    assert_eq!(composition(a, b), Layout::new(Int::<6>, Int::<2>));
}

#[test]
fn test_composition_multimode_pipeline() {
    // Layout((6, 2), (8, 2)) o Layout((4, 3), (3, 1)) -> Layout(((2, 2), 3), ((24, 2), 8))
    let a = Layout::new(
        int_tuple![Int::<6>, Int::<2>],
        int_tuple![Int::<8>, Int::<2>],
    );
    let b = Layout::new(
        int_tuple![Int::<4>, Int::<3>],
        int_tuple![Int::<3>, Int::<1>],
    );
    assert_eq!(
        composition(a, b),
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), Int::<3>],
            int_tuple![(Int::<24>, Int::<2>), Int::<8>],
        )
    );
}

#[test]
fn test_composition_mid_layout_truncation() {
    // Layout((4, 6, 8, 10), (2, 3, 5, 7)) o Layout(6, 12) -> Layout((2, 3), (9, 5))
    let a = Layout::new(
        int_tuple![Int::<4>, Int::<6>, Int::<8>, Int::<10>],
        int_tuple![Int::<2>, Int::<3>, Int::<5>, Int::<7>],
    );
    let b = Layout::new(Int::<6>, Int::<12>);
    assert_eq!(
        composition(a, b),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<9>, Int::<5>],
        )
    );
}

#[test]
fn test_composition_const_eval() {
    const A: Layout<Int<20>, Int<2>> = Layout::new(Int::<20>, Int::<2>);
    const B: Layout<Int<5>, Int<4>> = Layout::new(Int::<5>, Int::<4>);
    const RES: Layout<Int<5>, Int<8>> = composition(A, B);
    assert_eq!(RES, Layout::new(Int::<5>, Int::<8>));
}

#[test]
fn composition_preserves_large_signed_stride_range() {
    let a = Layout::new(
        int_tuple![Int::<2>, Int::<2>],
        int_tuple![Int::<0>, Int::<2>],
    );
    let b = Layout::new(Int::<2>, Int::<2_147_483_646>);
    let result: Layout<Int<2>, Int<2_147_483_646>> = composition(a, b);
    assert_eq!(result(0i32), 0);
    assert_eq!(result(1i32), 2_147_483_646);
    assert_eq!(result(1i32), a(b(1i32)));
}

#[test]
fn composition_with_negative_stride_skips_whole_modes() {
    let a = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        int_tuple![Int::<1>, Int::<3>],
    );
    let b = Layout::new(Int::<4>, Int::<-2>);
    let result: Layout<Int<4>, Int<-3>> = composition(a, b);
    for index in 0..4i32 {
        assert_eq!(result(index), a(b(index)));
    }
}

#[test]
fn composition_with_zero_stride_preserves_rhs_extent() {
    let a = Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        int_tuple![Int::<1>, Int::<3>],
    );
    let b = Layout::new(Int::<3>, Int::<0>);
    let result: Layout<Int<3>, Int<0>> = composition(a, b);
    assert_eq!(result, Layout::new(Int::<3>, Int::<0>));
}

#[test]
fn test_composition_nil_lhs() {
    assert_eq!(composition(Nil, Nil), Nil);
    assert_eq!(composition(Nil, 4i32), Layout::new(4i32, Int::<1>));
    assert_eq!(composition(Nil, Int::<4>), Layout::new(Int::<4>, Int::<1>));
    let b = Layout::new(Int::<4>, Int::<2>);
    assert_eq!(composition(Nil, b), b);
    assert_eq!(
        composition(Nil, tuple![Int::<3>, Nil, Int::<5>]),
        Layout::new(
            int_tuple![Int::<3>, Int::<1>, Int::<5>],
            tuple![e!(0), e!(1), e!(2)],
        )
    );
}

#[test]
fn test_composition_tiler_lhs_and_rhs() {
    // Scalar tiler LHS & dynamic scalar tiler
    assert_eq!(
        composition(Int::<20>, Layout::new(Int::<5>, Int::<4>)),
        Layout::new(Int::<5>, Int::<4>)
    );
    assert_eq!(
        composition(20i32, Layout::new(5i32, 4i32)),
        Layout::new(5i32, 4i32)
    );

    // Tuple tiler LHS promoted to basis-strided layout + tuple tiler RHS (by-mode)
    let lhs_tiler = tuple![Int::<6>, Int::<4>];
    let rhs_tiler = tuple![Int::<2>, Layout::new(Int::<2>, Int::<2>)];
    assert_eq!(
        composition(lhs_tiler, rhs_tiler),
        Layout::new(
            int_tuple![Int::<2>, Int::<2>],
            tuple![e!(0), Int::<2> * e!(1)],
        )
    );

    // By-mode tuple composition on Layout LHS, plus Nil / empty Layout RHS
    let a = Layout::new(
        int_tuple![Int::<12>, (Int::<4>, Int::<8>)],
        int_tuple![Int::<1>, (Int::<12>, Int::<48>)],
    );
    assert_eq!(a.composition(Nil), a);
    assert_eq!(a.composition(Layout::new(Nil, Nil)), Layout::new(Nil, Nil));
    assert_eq!(
        a.composition(tuple![
            Layout::new(Int::<3>, Int::<4>),
            Layout::new(Int::<8>, Int::<2>),
        ]),
        Layout::new(
            int_tuple![Int::<3>, Int::<8>],
            int_tuple![Int::<4>, Int::<24>]
        )
    );
}

#[test]
fn test_composition_scaled_basis_rhs() {
    let a = Layout::new(
        int_tuple![Int::<8>, Int::<8>],
        int_tuple![Int::<9>, Int::<1>],
    );
    assert_eq!(
        a.composition(Layout::new(Int::<4>, e!(1))),
        Layout::new(Int::<4>, Int::<1>)
    );
    assert_eq!(
        a.composition(Layout::new(Int::<4>, Int::<2> * e!(0))),
        Layout::new(Int::<4>, Int::<18>)
    );

    // Coalescing sub-modes under a single-mode scaled-basis coprofile
    let nested = Layout::new(
        int_tuple![(Int::<2>, Int::<4>), Int::<6>],
        int_tuple![(Int::<1>, Int::<2>), Int::<10>],
    );
    assert_eq!(
        nested.composition(Layout::new(Int::<4>, Int::<2> * e!(0))),
        Layout::new(Int::<4>, Int::<2>)
    );
}

#[test]
fn test_composition_dynamic_and_const() {
    const CONST_OK: Layout<Int<5>, Int<8>> = composition(
        Layout::new(Int::<20>, Int::<2>),
        Layout::new(Int::<5>, Int::<4>),
    );
    assert_eq!(CONST_OK, Layout::new(Int::<5>, Int::<8>));

    assert!(
        std::panic::catch_unwind(|| {
            composition(Layout::new(20i32, int::MAX), Layout::new(5i32, 2i32))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            composition(Layout::new(20i32, 2i32), Layout::new(-5i32, 4i32))
        })
        .is_err()
    );

    let ok_dyn = Layout::new(20i32, 2i32).composition(Layout::new(5i32, 4i32));
    assert_eq!(ok_dyn, Layout::new(5i32, 8i32));
}

#[test]
fn composition_rank1_postconditions() {
    // test/test_composition.py: exhaust small shapes and strides, including broadcasts.
    for a_extent in 1..4 {
        for a_stride in 0..4 {
            for b_extent in 1..4 {
                for b_stride in 0..4 {
                    let a = Layout::new(a_extent, a_stride);
                    let b = Layout::new(b_extent, b_stride);
                    let result = composition(a, b);
                    assert!(compatible(&b.shape(), &result.shape()));
                    for i in 0..b.size() {
                        assert_eq!(result(i), a(b(i)), "{a} composed with {b} at {i}");
                    }
                }
            }
        }
    }
}

#[test]
fn composition_hierarchical_postconditions() {
    // docs/04_layout_algebra.md: column-major reshape crosses a mode boundary.
    let a = Layout::new(
        int_tuple![Int::<10>, Int::<2>],
        int_tuple![Int::<16>, Int::<4>],
    );
    let b = Layout::from_shape(int_tuple![Int::<5>, Int::<4>]);
    let result = composition(a, b);
    assert_eq!(
        result,
        Layout::new(
            int_tuple![Int::<5>, (Int::<2>, Int::<2>)],
            int_tuple![Int::<16>, (Int::<80>, Int::<4>)],
        )
    );
    assert!(compatible(&b.shape(), &result.shape()));
    for i in 0..b.size() {
        assert_eq!(result(i), a(b(i)));
    }
}

#[test]
fn composition_associativity_on_nested_domains() {
    // test/test_composition.py: both intermediate coordinates are within their domains.
    let a = Layout::new(
        int_tuple![Int::<4>, Int::<3>],
        int_tuple![Int::<3>, Int::<1>],
    );
    let b = Layout::from_shape(int_tuple![Int::<4>, Int::<3>]);
    let c = Layout::new(Int::<6>, Int::<2>);
    let left = composition(composition(a, b), c);
    let right = composition(a, composition(b, c));
    for i in 0..c.size() {
        let expected = a(b(c(i)));
        assert_eq!(left(i), expected);
        assert_eq!(right(i), expected);
    }
}

#[test]
fn composition_short_tiler_and_its_promotion_agree() {
    // test/test_composition.py: omitted modes do not contribute to the result.
    let a = Layout::new(
        int_tuple![(Int::<4>, Int::<2>), Int::<5>],
        int_tuple![(Int::<5>, Int::<40>), Int::<1>],
    );
    let tiler = tuple![Layout::new(Int::<2>, Int::<2>)];
    let result = composition(a, tiler);
    assert_eq!(result, composition(a, tiler_to_layout(tiler)));
    assert_eq!(
        result,
        Layout::new(int_tuple![Int::<2>], int_tuple![Int::<10>])
    );
}

#[test]
fn composition_within_a_mode_can_use_nondividing_stride() {
    // test/test_composition.py: offsets 0 and 3 both lie in A's first mode.
    let a = Layout::new(
        int_tuple![Int::<5>, Int::<3>],
        int_tuple![Int::<7>, Int::<1>],
    );
    let b = Layout::new(Int::<2>, Int::<3>);
    let result = composition(a, b);
    assert_eq!(result, Layout::new(Int::<2>, Int::<21>));
    for i in 0..b.size() {
        assert_eq!(result(i), a(b(i)));
    }
}

#[test]
fn composition_rejects_static_shape_and_stride_divisibility_violations() {
    // A valid Composition implementation would make the marker type ambiguous.
    trait RejectedComposition<B, Marker> {
        fn check(&self, _: B) {}
    }
    impl<A, B> RejectedComposition<B, ()> for A {}
    struct Implemented;
    impl<A: Composition<B>, B> RejectedComposition<B, Implemented> for A {}

    // test/test_composition.py: shape truncation cannot keep a partial second mode.
    let a = Layout::new(
        int_tuple![Int::<5>, Int::<3>],
        int_tuple![Int::<7>, Int::<1>],
    );
    RejectedComposition::<_, _>::check(&a, Layout::new(Int::<7>, Int::<1>));
    // The same nondividing stride as above now crosses a mode boundary.
    RejectedComposition::<_, _>::check(&a, Layout::new(Int::<3>, Int::<3>));

    let whitepaper = Layout::new(
        int_tuple![Int::<4>, Int::<6>, Int::<8>],
        int_tuple![Int::<2>, Int::<3>, Int::<5>],
    );
    RejectedComposition::<_, _>::check(&whitepaper, Layout::new(Int::<6>, Int::<3>));
    RejectedComposition::<_, _>::check(&whitepaper, Layout::new(Int::<6>, Int::<1>));

    let coalescible = Layout::new(
        int_tuple![Int::<4>, Int::<2>, Int::<8>],
        int_tuple![Int::<3>, Int::<12>, Int::<97>],
    );
    let gapped = Layout::new(
        coalescible.shape(),
        int_tuple![Int::<3>, Int::<15>, Int::<97>],
    );
    let within = Layout::new(Int::<3>, Int::<3>);
    assert_eq!(
        composition(coalescible, within),
        Layout::new(Int::<3>, Int::<9>)
    );
    RejectedComposition::<_, _>::check(&coalescible, Layout::new(Int::<4>, Int::<3>));
    RejectedComposition::<_, _>::check(&gapped, within);
}

#[test]
fn composition_preserves_coordinate_codomain() {
    // test/test_composition.py: factor a walk that crosses distinct basis axes.
    let a = Layout::new(int_tuple![Int::<4>, Int::<3>], tuple![e!(1), e!(0)]);
    let b = Layout::new(Int::<6>, Int::<2>);
    let result = composition(a, b);
    assert_eq!(
        result,
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            tuple![Int::<2> * e!(1), e!(0)],
        )
    );
    assert!(compatible(&b.shape(), &result.shape()));
    for i in 0..b.size() {
        assert!(basis_eq(result(i), a(b(i))));
    }
}

#[test]
fn composition_of_one_mode_preserves_its_siblings() {
    // pycute/algebra.py: selecting mode 1 is equivalent to an explicit padded tiler.
    let a = Layout::from_shape(int_tuple![Int::<4>, Int::<6>]);
    let b = Layout::new(Int::<3>, Int::<2>);
    let result = composition(a, tuple![Nil, b]);
    assert_eq!(
        result,
        make_layout!(a.get::<0>(), composition(a.get::<1>(), b))
    );
    assert!(compatible(&b.shape(), &result.get::<1>().shape()));
    for i in 0..b.size() {
        assert_eq!(result.get::<1>()(i), a.get::<1>()(b(i)));
    }
}
