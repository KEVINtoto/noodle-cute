#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use core::fmt::Debug;
use core::mem::size_of_val;

use noodle_cute::algebra::composition::composition;
use noodle_cute::algebra::nullspace::{Nullspace, nullspace};
use noodle_cute::basis::{BasisCombination, BasisTerm, IntoColexTerms, MergeEq, basis_eq};
use noodle_cute::integer::{Int, Integer, eq as int_eq, int};
use noodle_cute::layout::{Layout, tiler_to_layout};
use noodle_cute::shape::{IndexToCoord, Shape};
use noodle_cute::stride::{InnerProduct, Stride};
use noodle_cute::tuple::{Nil, Til};
use noodle_cute::{e, int_tuple, mode, tuple};

type PairLayout<S0, S1, D0, D1> = Layout<Til<S0, Til<S1, Nil>>, Til<D0, Til<D1, Nil>>>;

// Post-condition checkers mirroring `test/test_nullspace.py`

fn assert_nullspace_int<SL, DL, SN, DN>(l: Layout<SL, DL>) -> Layout<SN, DN>
where
    SL: Shape + Debug,
    DL: Stride<SL> + Debug,
    SN: Shape + Debug,
    DN: Stride<SN> + Debug,
    Layout<SL, DL>: Nullspace<Output = Layout<SN, DN>>,
    int: IndexToCoord<SN>,
    <int as IndexToCoord<SN>>::Output: InnerProduct<DN>,
    <<int as IndexToCoord<SN>>::Output as InnerProduct<DN>>::Output:
        Integer + IndexToCoord<SL> + Debug,
    <<<int as IndexToCoord<SN>>::Output as InnerProduct<DN>>::Output as IndexToCoord<SL>>::Output:
        InnerProduct<DL>,
    <<<<int as IndexToCoord<SN>>::Output as InnerProduct<DN>>::Output as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output:
        Integer + Debug,
{
    let n = nullspace(l);
    for i in 0..n.size() {
        let ni = n(i);
        let lni = l(ni);
        assert!(
            int_eq(lni, Int::<0>),
            "nullspace post-condition failed for L={l:?}, N={n:?} at i={i}: L(N(i))={lni:?} != 0"
        );
    }
    n
}

fn assert_nullspace_coord<SL, DL, SN, DN>(l: Layout<SL, DL>) -> Layout<SN, DN>
where
    SL: Shape + Debug,
    DL: Stride<SL> + Debug,
    SN: Shape + Debug,
    DN: Stride<SN> + Debug,
    Layout<SL, DL>: Nullspace<Output = Layout<SN, DN>>,
    int: IndexToCoord<SN>,
    <int as IndexToCoord<SN>>::Output: InnerProduct<DN>,
    <<int as IndexToCoord<SN>>::Output as InnerProduct<DN>>::Output:
        Integer + IndexToCoord<SL> + Debug,
    <<<int as IndexToCoord<SN>>::Output as InnerProduct<DN>>::Output as IndexToCoord<SL>>::Output:
        InnerProduct<DL>,
    <<<<int as IndexToCoord<SN>>::Output as InnerProduct<DN>>::Output as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output:
        IntoColexTerms + Debug,
    <<<<<int as IndexToCoord<SN>>::Output as InnerProduct<DN>>::Output as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IntoColexTerms>::Terms:
        MergeEq<<Int<0> as IntoColexTerms>::Terms>,
{
    let n = nullspace(l);
    for i in 0..n.size() {
        let ni = n(i);
        let lni = l(ni);
        assert!(
            basis_eq(lni, Int::<0>),
            "coord nullspace post-condition failed for L={l:?}, N={n:?} at i={i}: L(N(i))={lni:?} != 0"
        );
    }
    n
}

#[test]
fn test_nullspace_int_suite() {
    assert_eq!(
        assert_nullspace_int(Layout::new(Int::<1>, Int::<0>)),
        Layout::new(Int::<1>, Int::<1>)
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(Int::<1>, Int::<1>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(Int::<1>, Int::<2>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(Int::<1>, Int::<4>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(Int::<4>, Int::<0>)),
        Layout::new(Int::<4>, Int::<1>)
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(Int::<4>, Int::<1>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(Int::<4>, Int::<2>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(Int::<4>, Int::<4>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(
            int_tuple![Int::<1>, Int::<1>],
            int_tuple![Int::<0>, Int::<0>],
        )),
        Layout::new(
            int_tuple![Int::<1>, Int::<1>],
            int_tuple![Int::<1>, Int::<1>],
        )
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(
            int_tuple![Int::<3>, Int::<7>],
            int_tuple![Int::<0>, Int::<0>],
        )),
        Layout::new(
            int_tuple![Int::<3>, Int::<7>],
            int_tuple![Int::<1>, Int::<3>],
        )
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(
            int_tuple![Int::<2>, Int::<4>],
            int_tuple![Int::<0>, Int::<2>],
        )),
        Layout::new(Int::<2>, Int::<1>)
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(
            int_tuple![Int::<8>, Int::<4>],
            int_tuple![Int::<1>, Int::<8>],
        )),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(
            int_tuple![Int::<8>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>],
        )),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(
            int_tuple![Int::<2>, Int::<4>, Int::<6>],
            int_tuple![Int::<1>, Int::<2>, Int::<0>],
        )),
        Layout::new(Int::<6>, Int::<8>)
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(
            int_tuple![Int::<2>, Int::<4>, Int::<6>],
            int_tuple![Int::<0>, Int::<1>, Int::<0>],
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<6>],
            int_tuple![Int::<1>, Int::<8>],
        )
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(
            int_tuple![Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<16>],
        )),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(
            int_tuple![Int::<4>, Int::<5>, Int::<6>],
            int_tuple![Int::<1>, Int::<1>, Int::<0>],
        )),
        Layout::new(Int::<6>, Int::<20>)
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(
            int_tuple![Int::<7>, Int::<5>, Int::<9>],
            int_tuple![Int::<2>, Int::<0>, Int::<1>],
        )),
        Layout::new(Int::<5>, Int::<7>)
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(
            int_tuple![Int::<7>, Int::<5>, Int::<9>],
            int_tuple![Int::<2>, Int::<0>, Int::<0>],
        )),
        Layout::new(
            int_tuple![Int::<5>, Int::<9>],
            int_tuple![Int::<7>, Int::<35>],
        )
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(
            int_tuple![Int::<7>, Int::<5>, Int::<9>],
            int_tuple![Int::<0>, Int::<0>, Int::<0>],
        )),
        Layout::new(
            int_tuple![Int::<7>, Int::<5>, Int::<9>],
            int_tuple![Int::<1>, Int::<7>, Int::<35>],
        )
    );
}

#[test]
fn test_nullspace_coord_suite() {
    assert_eq!(
        assert_nullspace_coord(Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            tuple![e![0], e![1]],
        )),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_nullspace_coord(Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            tuple![Int::<0>, e![0]],
        )),
        Layout::new(Int::<4>, Int::<1>)
    );
    assert_eq!(
        assert_nullspace_coord(Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            tuple![Int::<0>, e![4, 1]],
        )),
        Layout::new(Int::<4>, Int::<1>)
    );
    assert_eq!(
        assert_nullspace_coord(Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            tuple![Int::<2> * e![0], Int::<0>],
        )),
        Layout::new(Int::<5>, Int::<4>)
    );

    // SM70 MMA 8x8x4 C TV inverse variants
    assert_eq!(
        assert_nullspace_coord(Layout::new(
            int_tuple![
                (Int::<2>, Int::<2>, Int::<2>),
                (Int::<2>, Int::<2>, Int::<2>)
            ],
            tuple![
                (e![0], Int::<2> * e![1], Int::<4> * e![0]),
                (e![1], Int::<2> * e![0], Int::<4> * e![1]),
            ],
        )),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_nullspace_coord(Layout::new(
            int_tuple![
                (Int::<2>, Int::<2>, Int::<2>),
                (Int::<2>, Int::<2>, Int::<2>)
            ],
            tuple![
                (e![0], Int::<2> * e![1], Int::<5> * e![0]),
                (e![1], Int::<0>, Int::<5> * e![1]),
            ],
        )),
        Layout::new(Int::<2>, Int::<16>)
    );
    assert_eq!(
        assert_nullspace_coord(Layout::new(
            int_tuple![
                (Int::<2>, Int::<2>, Int::<2>),
                (Int::<2>, Int::<2>, Int::<2>)
            ],
            tuple![
                (e![0], Int::<0>, Int::<5> * e![0]),
                (e![1], Int::<2> * e![0], Int::<4> * e![1]),
            ],
        )),
        Layout::new(Int::<2>, Int::<2>)
    );

    // SM70 MMA 8x8x4 A TV inverse
    let sm70_a = composition(
        tiler_to_layout(int_tuple![Int::<8>, Int::<4>]),
        Layout::new(
            int_tuple![(Int::<4>, Int::<2>), Int::<4>],
            int_tuple![(Int::<8>, Int::<4>), Int::<0>],
        ),
    );
    assert_eq!(
        assert_nullspace_coord(sm70_a),
        Layout::new(Int::<4>, Int::<8>)
    );

    // SM80 MMA 16x8 TV inverse
    let sm80 = composition(
        tiler_to_layout(int_tuple![Int::<16>, Int::<8>]),
        Layout::new(
            int_tuple![(Int::<4>, Int::<8>), (Int::<2>, Int::<2>)],
            int_tuple![(Int::<0>, Int::<1>), (Int::<16>, Int::<8>)],
        ),
    );
    assert_eq!(
        assert_nullspace_coord(sm80),
        Layout::new(Int::<4>, Int::<1>)
    );
}

#[test]
fn test_nullspace_const_eval_tilers_basis_comb_and_dynamic() {
    // Const evaluation & zero-sized static output
    const N_CONST: PairLayout<Int<2>, Int<6>, Int<1>, Int<8>> = nullspace(Layout::new(
        int_tuple![Int::<2>, Int::<4>, Int::<6>],
        int_tuple![Int::<0>, Int::<1>, Int::<0>],
    ));
    assert_eq!(size_of_val(&N_CONST), 0);
    assert_eq!(
        N_CONST,
        Layout::new(
            int_tuple![Int::<2>, Int::<6>],
            int_tuple![Int::<1>, Int::<8>],
        )
    );

    // Inherent methods match free functions
    let sample = Layout::new(
        int_tuple![Int::<2>, Int::<4>, Int::<6>],
        int_tuple![Int::<0>, Int::<1>, Int::<0>],
    );
    assert_eq!(sample.nullspace(), nullspace(sample));

    // Nil, empty layout, nested Nil
    assert_eq!(nullspace(Nil), Layout::new(Int::<1>, Int::<0>));
    assert_eq!(
        nullspace(Layout::new(Nil, Nil)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        nullspace(Layout::new(
            int_tuple![(), Int::<4>, ()],
            int_tuple![(), Int::<0>, ()],
        )),
        Layout::new(Int::<4>, Int::<1>)
    );

    // Tilers (scalars, integer tuples, and tuples containing zero-stride layouts)
    assert_eq!(nullspace(Int::<8>), Layout::new(Int::<1>, Int::<0>));
    assert_eq!(nullspace(8i32), Layout::new(Int::<1>, Int::<0>));
    assert_eq!(
        nullspace(int_tuple![Int::<4>, Int::<5>]),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        nullspace(tuple![Layout::new(Int::<4>, Int::<0>), Int::<5>]),
        Layout::new(Int::<4>, Int::<1>)
    );

    // BasisCombination<Nil>, static-zero BasisTerm, and multi-term BasisCombination
    assert_eq!(
        assert_nullspace_coord(Layout::new(
            int_tuple![Int::<3>, Int::<5>],
            tuple![BasisCombination::<Nil>::zero(), e![0] + e![1]],
        )),
        Layout::new(Int::<3>, Int::<1>)
    );
    assert_eq!(
        assert_nullspace_coord(Layout::new(
            int_tuple![Int::<3>, Int::<5>],
            tuple![e![0] + e![1], BasisTerm::<Int<0>, mode![2]>::new(Int::<0>)],
        )),
        Layout::new(Int::<5>, Int::<3>)
    );

    // Dynamic shapes with static zero strides
    assert_eq!(
        assert_nullspace_int(Layout::new(
            int_tuple![3i32, 5i32, 7i32],
            int_tuple![Int::<1>, Int::<0>, Int::<0>],
        )),
        Layout::new(int_tuple![5i32, 7i32], int_tuple![3i32, 15i32])
    );

    // Dynamic stride 0 is NOT statically zero -> returns trivial 1:0 (still satisfies L(N(i)) == 0)
    assert_eq!(
        assert_nullspace_int(Layout::new(Int::<4>, 0i32)),
        Layout::new(Int::<1>, Int::<0>)
    );

    // Negative strides
    assert_eq!(
        assert_nullspace_int(Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            int_tuple![Int::<-1>, Int::<0>],
        )),
        Layout::new(Int::<5>, Int::<4>)
    );
    assert_eq!(
        assert_nullspace_int(Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            int_tuple![Int::<-1>, Int::<-2>],
        )),
        Layout::new(Int::<1>, Int::<0>)
    );

    // Modes after the last zero-stride mode do NOT trigger overflow
    let no_spurious_overflow = Layout::new(
        int_tuple![2_000_000_000i32, 2i32, 4i32],
        int_tuple![Int::<0>, Int::<1>, Int::<2>],
    );
    assert_eq!(
        nullspace(no_spurious_overflow),
        Layout::new(2_000_000_000i32, Int::<1>)
    );

    // Large static prefix product within i32::MAX succeeds cleanly
    const BIG_INT: Layout<Int<2>, Int<2_000_000_000>> = nullspace(Layout::new(
        int_tuple![Int::<2_000_000_000>, Int::<2>],
        int_tuple![Int::<1>, Int::<0>],
    ));
    assert_eq!(BIG_INT, Layout::new(Int::<2>, Int::<2_000_000_000>));
}

#[test]
#[should_panic]
fn test_nullspace_panics_on_dynamic_prefix_overflow() {
    let overflow = Layout::new(
        int_tuple![2_000_000_000i32, 2i32, 4i32],
        int_tuple![Int::<1>, Int::<1>, Int::<0>],
    );
    let _ = nullspace(overflow);
}

#[test]
#[should_panic]
fn test_nullspace_panics_on_negative_dynamic_extent() {
    let negative = Layout::new(int_tuple![-3i32, 4i32], int_tuple![Int::<1>, Int::<0>]);
    let _ = nullspace(negative);
}
