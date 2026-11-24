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

use noodle_cute::algebra::complement::{Complement, complement, complement_with};
use noodle_cute::algebra::composition::composition;
use noodle_cute::basis::{
    BasisCombination, BasisTerm, IntoColexTerms, MergeColex, MergeEq, basis_eq, colex_lt,
};
use noodle_cute::integer::{Int, Integer, ge, int};
use noodle_cute::layout::{Coprofile, Layout};
use noodle_cute::shape::{IndexToCoord, Shape, coordinates};
use noodle_cute::stride::{Coshape, InnerProduct, Stride};
use noodle_cute::tuple::{Item, Nil, TupleContainer, WeaklyCongruent};
use noodle_cute::{e, int_tuple, make_layout, mode, tuple};

// Post-condition checkers mirroring `test/test_complement.py`

type PairLayout<S0, S1, D0, D1> = Layout<
    TupleContainer<S0, TupleContainer<S1, Nil>>,
    TupleContainer<D0, TupleContainer<D1, Nil>>,
>;

fn assert_weak_congruent<A: WeaklyCongruent<B>, B: Item>(_: &A, _: &B) {}

fn assert_complement_weak<SL, DL, SR, DR>(l: Layout<SL, DL>) -> Layout<SR, DR>
where
    SL: Shape + Debug,
    DL: Stride<SL> + Debug,
    SR: Shape + Debug,
    DR: Stride<SR> + Debug,
    Layout<SL, DL>: Complement<Output = Layout<SR, DR>> + Coprofile,
    <Layout<SL, DL> as Coprofile>::Output: WeaklyCongruent<SR>,
    int: IndexToCoord<SL> + IndexToCoord<SR>,
    <int as IndexToCoord<SL>>::Output: InnerProduct<DL>,
    <int as IndexToCoord<SR>>::Output: InnerProduct<DR>,
    <<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output: IntoColexTerms + Debug,
    <<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output: IntoColexTerms + Debug,
    <<<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IntoColexTerms>::Terms:
        MergeColex<
                <<<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IntoColexTerms>::Terms,
            > + MergeEq<
                <<<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IntoColexTerms>::Terms,
            >,
{
    let r = complement(l);

    // 1. Weak congruence with codomain profile
    assert_weak_congruent(&l.coprofile(), &r.shape());

    // 2. Orderedness & disjointness over the extended domain `1..(10 + size(R))`
    let r_span = 10 + r.size();
    let l_size = l.size();
    for i in 1..r_span {
        let prev = r(i - 1);
        let curr = r(i);
        assert!(
            colex_lt(prev, curr),
            "orderedness failed for L={l:?}, R={r:?} at i={i}: R(i-1)={prev:?}, R(i)={curr:?}"
        );
        for j in 0..l_size {
            let lv = l(j);
            assert!(
                !basis_eq(curr, lv),
                "disjointness failed for L={l:?}, R={r:?} at i={i}, j={j}: R(i)={curr:?}, L(j)={lv:?}"
            );
        }
    }

    r
}

fn assert_complement_strong_1d<SL, DL, SR, DR>(l: Layout<SL, DL>) -> Layout<SR, DR>
where
    SL: Shape + Debug,
    DL: Stride<SL> + Debug,
    SR: Shape + Debug,
    DR: Stride<SR> + Debug,
    Layout<SL, DL>: Complement<Output = Layout<SR, DR>> + Coprofile,
    <Layout<SL, DL> as Coprofile>::Output: WeaklyCongruent<SR>,
    int: IndexToCoord<SL> + IndexToCoord<SR>,
    <int as IndexToCoord<SL>>::Output: InnerProduct<DL>,
    <int as IndexToCoord<SR>>::Output: InnerProduct<DR>,
    <<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output: IntoColexTerms + Debug,
    <<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output: IntoColexTerms + Debug,
    <<<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IntoColexTerms>::Terms:
        MergeColex<
                <<<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IntoColexTerms>::Terms,
            > + MergeEq<
                <<<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IntoColexTerms>::Terms,
            >,
    TupleContainer<SL, TupleContainer<SR, Nil>>:
        Coshape<TupleContainer<DL, TupleContainer<DR, Nil>>>,
    <TupleContainer<SL, TupleContainer<SR, Nil>> as Coshape<
        TupleContainer<DL, TupleContainer<DR, Nil>>,
    >>::Output: Shape,
    int: IndexToCoord<TupleContainer<SL, TupleContainer<SR, Nil>>>,
    <int as IndexToCoord<TupleContainer<SL, TupleContainer<SR, Nil>>>>::Output:
        InnerProduct<TupleContainer<DL, TupleContainer<DR, Nil>>>,
    <<int as IndexToCoord<TupleContainer<SL, TupleContainer<SR, Nil>>>>::Output as InnerProduct<
        TupleContainer<DL, TupleContainer<DR, Nil>>,
    >>::Output: Integer,
{
    let r = assert_complement_weak(l);

    // Divisible tiling condition: `make_layout!(l, r)` covers every offset in `0..cosize`
    // with uniform multiplicity (exact bijection when `l` has no stride-0 nullspace).
    let completed = make_layout!(l, r);
    let cosize = completed.cosize();
    let total = completed.size();
    assert_eq!(
        total % cosize,
        0,
        "completed={completed:?} size={total} cosize={cosize}"
    );
    let expected_mult = total / cosize;
    let mut counts = vec![0i32; cosize as usize];
    for i in 0..total {
        let offset = completed(i);
        assert!(ge(offset, 0i32), "negative offset");
        let v = offset.value();
        assert!(v < cosize, "offset {v} out of range 0..{cosize}");
        counts[v as usize] += 1;
    }
    for (v, &c) in counts.iter().enumerate() {
        assert_eq!(
            c, expected_mult,
            "offset {v} hit {c} times (expected {expected_mult}) for L={l:?}, R={r:?}"
        );
    }

    r
}

fn assert_complement_strong_coord<SL, DL, SR, DR>(l: Layout<SL, DL>) -> Layout<SR, DR>
where
    SL: Shape + Debug,
    DL: Stride<SL> + Debug,
    SR: Shape + Debug,
    DR: Stride<SR> + Debug,
    Layout<SL, DL>: Complement<Output = Layout<SR, DR>> + Coprofile,
    <Layout<SL, DL> as Coprofile>::Output: WeaklyCongruent<SR>,
    int: IndexToCoord<SL> + IndexToCoord<SR>,
    <int as IndexToCoord<SL>>::Output: InnerProduct<DL>,
    <int as IndexToCoord<SR>>::Output: InnerProduct<DR>,
    <<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output: IntoColexTerms + Debug,
    <<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output: IntoColexTerms + Debug,
    <<<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IntoColexTerms>::Terms:
        MergeColex<
                <<<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IntoColexTerms>::Terms,
            > + MergeEq<
                <<<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IntoColexTerms>::Terms,
            >,
    TupleContainer<SL, TupleContainer<SR, Nil>>:
        Coshape<TupleContainer<DL, TupleContainer<DR, Nil>>>,
    <TupleContainer<SL, TupleContainer<SR, Nil>> as Coshape<
        TupleContainer<DL, TupleContainer<DR, Nil>>,
    >>::Output: Shape,
    int: IndexToCoord<TupleContainer<SL, TupleContainer<SR, Nil>>>,
    <int as IndexToCoord<TupleContainer<SL, TupleContainer<SR, Nil>>>>::Output:
        InnerProduct<TupleContainer<DL, TupleContainer<DR, Nil>>>,
    <<int as IndexToCoord<TupleContainer<SL, TupleContainer<SR, Nil>>>>::Output as InnerProduct<
        TupleContainer<DL, TupleContainer<DR, Nil>>,
    >>::Output: IntoColexTerms,
    <<<int as IndexToCoord<TupleContainer<SL, TupleContainer<SR, Nil>>>>::Output as InnerProduct<
        TupleContainer<DL, TupleContainer<DR, Nil>>,
    >>::Output as IntoColexTerms>::Terms: MergeEq<
        <<<int as IndexToCoord<TupleContainer<SL, TupleContainer<SR, Nil>>>>::Output as InnerProduct<
            TupleContainer<DL, TupleContainer<DR, Nil>>,
        >>::Output as IntoColexTerms>::Terms,
    >,
    noodle_cute::shape::Coordinates<
        <TupleContainer<SL, TupleContainer<SR, Nil>> as Coshape<
            TupleContainer<DL, TupleContainer<DR, Nil>>,
        >>::Output,
    >: Iterator,
{
    let r = assert_complement_weak(l);

    // Every point in the completed layout's codomain is reached uniformly.
    let completed = make_layout!(l, r);
    let coshape = completed.coshape();
    let cosize = coshape.size();
    let total = completed.size();
    assert_eq!(total % cosize, 0, "completed={completed:?}");
    let expected_mult = total / cosize;

    let mut image = Vec::with_capacity(total as usize);
    for i in 0..total {
        image.push(completed(i));
    }
    let mut unique_count = 0i32;
    for (idx, v) in image.iter().enumerate() {
        let already_seen = image[..idx].iter().any(|u| basis_eq(*u, *v));
        if !already_seen {
            let mult = image.iter().filter(|u| basis_eq(**u, *v)).count() as i32;
            assert_eq!(mult, expected_mult);
            unique_count += 1;
        }
    }
    assert_eq!(unique_count, cosize);
    assert_eq!(coordinates(coshape).count() as i32, cosize);

    r
}

// 1. All strong 1D cases from `test/test_complement.py::test_complement`

#[test]
fn test_complement_strong_1d_suite() {
    assert_eq!(
        assert_complement_strong_1d(Layout::new(Int::<1>, Int::<0>)),
        Layout::new(Int::<1>, Int::<1>)
    );
    assert_eq!(
        assert_complement_strong_1d(Layout::new(Int::<1>, Int::<1>)),
        Layout::new(Int::<1>, Int::<1>)
    );
    assert_eq!(
        assert_complement_strong_1d(Layout::new(Int::<1>, Int::<2>)),
        Layout::new(Int::<1>, Int::<1>)
    );
    assert_eq!(
        assert_complement_strong_1d(Layout::new(Int::<1>, Int::<4>)),
        Layout::new(Int::<1>, Int::<1>)
    );
    assert_eq!(
        assert_complement_strong_1d(Layout::new(
            int_tuple![Int::<1>, Int::<1>],
            int_tuple![Int::<0>, Int::<0>],
        )),
        Layout::new(Int::<1>, Int::<1>)
    );
    assert_eq!(
        assert_complement_strong_1d(Layout::new(
            int_tuple![Int::<3>, Int::<7>],
            int_tuple![Int::<0>, Int::<0>],
        )),
        Layout::new(Int::<1>, Int::<1>)
    );
    assert_eq!(
        assert_complement_strong_1d(Layout::new(Int::<5>, Int::<1>)),
        Layout::new(Int::<1>, Int::<5>)
    );
    assert_eq!(
        assert_complement_strong_1d(Layout::new(Int::<5>, Int::<3>)),
        Layout::new(
            int_tuple![Int::<3>, Int::<1>],
            int_tuple![Int::<1>, Int::<15>],
        )
    );
    assert_eq!(
        assert_complement_strong_1d(Layout::new(Int::<4>, Int::<0>)),
        Layout::new(Int::<1>, Int::<1>)
    );
    assert_eq!(
        assert_complement_strong_1d(Layout::new(Int::<4>, Int::<1>)),
        Layout::new(Int::<1>, Int::<4>)
    );
    assert_eq!(
        assert_complement_strong_1d(Layout::new(Int::<4>, Int::<2>)),
        Layout::new(
            int_tuple![Int::<2>, Int::<1>],
            int_tuple![Int::<1>, Int::<8>],
        )
    );
    assert_eq!(
        assert_complement_strong_1d(Layout::new(Int::<4>, Int::<4>)),
        Layout::new(
            int_tuple![Int::<4>, Int::<1>],
            int_tuple![Int::<1>, Int::<16>],
        )
    );

    assert_complement_strong_1d(Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        int_tuple![Int::<1>, Int::<2>],
    ));
    assert_complement_strong_1d(Layout::new(
        int_tuple![Int::<2>, Int::<3>],
        int_tuple![Int::<1>, Int::<2>],
    ));
    assert_complement_strong_1d(Layout::new(
        int_tuple![Int::<2>, Int::<4>],
        int_tuple![Int::<1>, Int::<4>],
    ));
    assert_complement_strong_1d(Layout::new(
        int_tuple![Int::<8>, Int::<4>],
        int_tuple![Int::<1>, Int::<8>],
    ));
    assert_complement_strong_1d(Layout::new(
        int_tuple![Int::<8>, Int::<4>],
        int_tuple![Int::<4>, Int::<1>],
    ));
    assert_complement_strong_1d(Layout::new(
        int_tuple![Int::<2>, Int::<4>, Int::<6>],
        int_tuple![Int::<1>, Int::<2>, Int::<8>],
    ));
    assert_complement_strong_1d(Layout::new(
        int_tuple![Int::<2>, Int::<4>, Int::<6>],
        int_tuple![Int::<4>, Int::<1>, Int::<8>],
    ));
    assert_complement_strong_1d(Layout::new(
        int_tuple![Int::<2>, Int::<4>, Int::<8>],
        int_tuple![Int::<8>, Int::<1>, Int::<64>],
    ));
    assert_complement_strong_1d(Layout::new(
        int_tuple![Int::<2>, Int::<4>, Int::<8>],
        int_tuple![Int::<32>, Int::<0>, Int::<2>],
    ));
    assert_complement_strong_1d(Layout::new(
        int_tuple![Int::<2>, Int::<4>, Int::<8>],
        int_tuple![Int::<2>, Int::<0>, Int::<32>],
    ));
    assert_complement_strong_1d(Layout::new(
        int_tuple![Int::<2>, Int::<4>, Int::<4>, Int::<4>, Int::<2>],
        int_tuple![Int::<32>, Int::<0>, Int::<2>, Int::<0>, Int::<512>],
    ));
    assert_complement_strong_1d(Layout::new(
        int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<2>)],
        int_tuple![(Int::<1>, Int::<4>), (Int::<8>, Int::<32>)],
    ));
    assert_complement_strong_1d(Layout::new(
        int_tuple![Int::<2>, (Int::<3>, Int::<4>)],
        int_tuple![Int::<3>, (Int::<1>, Int::<6>)],
    ));
    assert_complement_strong_1d(Layout::new(
        int_tuple![Int::<4>, Int::<2>],
        int_tuple![Int::<1>, Int::<16>],
    ));

    // Docstring example: Layout((2, 2), (1, 6)) -> Layout((3, 1), (2, 12))
    assert_eq!(
        assert_complement_strong_1d(Layout::new(
            int_tuple![Int::<2>, Int::<2>],
            int_tuple![Int::<1>, Int::<6>],
        )),
        Layout::new(
            int_tuple![Int::<3>, Int::<1>],
            int_tuple![Int::<2>, Int::<12>],
        )
    );
}

// 2. All weak (non-divisible) cases from `test/test_complement.py::test_complement_weak`

#[test]
fn test_complement_weak_suite() {
    assert_eq!(
        assert_complement_weak(Layout::new(
            int_tuple![Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<6>],
        )),
        Layout::new(Int::<1>, Int::<12>)
    );
    assert_eq!(
        assert_complement_weak(Layout::new(
            int_tuple![Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<5>],
        )),
        Layout::new(Int::<1>, Int::<10>)
    );
    assert_eq!(
        assert_complement_weak(Layout::new(
            int_tuple![Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<10>],
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<1>],
            int_tuple![Int::<4>, Int::<20>],
        )
    );
    assert_eq!(
        assert_complement_weak(Layout::new(
            int_tuple![Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<11>],
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<1>],
            int_tuple![Int::<4>, Int::<22>],
        )
    );
    assert_eq!(
        assert_complement_weak(Layout::new(
            int_tuple![Int::<2>, Int::<4>],
            int_tuple![Int::<11>, Int::<1>],
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<1>],
            int_tuple![Int::<4>, Int::<22>],
        )
    );
}

// 3. Coordinate basis cases from `test/test_complement.py::test_complement_coord`

#[test]
fn test_complement_coord_suite() {
    // 3:E(0) -> (1,):(3@0,)
    assert_eq!(
        assert_complement_strong_coord(Layout::new(Int::<3>, e![0])),
        Layout::new(int_tuple![Int::<1>], tuple![Int::<3> * e![0]])
    );

    // 3:4*E(2) -> (1, 1, (4, 1)):(1@0, 1@1, (1@2, 12@2))
    assert_eq!(
        assert_complement_strong_coord(Layout::new(Int::<3>, Int::<4> * e![2])),
        Layout::new(
            int_tuple![Int::<1>, Int::<1>, (Int::<4>, Int::<1>)],
            tuple![e![0], e![1], (e![2], Int::<12> * e![2])],
        )
    );

    // (2, 5, 3):(4*E(1), 5*E(0), 16*E(1)) -> ((5, 1), (4, 2, 1)):((1@0, 25@0), (1@1, 8@1, 48@1))
    assert_eq!(
        assert_complement_strong_coord(Layout::new(
            int_tuple![Int::<2>, Int::<5>, Int::<3>],
            tuple![Int::<4> * e![1], Int::<5> * e![0], Int::<16> * e![1]],
        )),
        Layout::new(
            int_tuple![(Int::<5>, Int::<1>), (Int::<4>, Int::<2>, Int::<1>)],
            tuple![
                (e![0], Int::<25> * e![0]),
                (e![1], Int::<8> * e![1], Int::<48> * e![1]),
            ],
        )
    );

    // (2, 3, 5):(4*E(1), 5*E(0), 7*E(2, 1))
    assert_eq!(
        assert_complement_strong_coord(Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            tuple![Int::<4> * e![1], Int::<5> * e![0], Int::<7> * e![2, 1]],
        )),
        Layout::new(
            int_tuple![
                (Int::<5>, Int::<1>),
                (Int::<4>, Int::<1>),
                (Int::<1>, (Int::<7>, Int::<1>)),
            ],
            tuple![
                (e![0], Int::<15> * e![0]),
                (e![1], Int::<8> * e![1]),
                (e![2, 0], (e![2, 1], Int::<35> * e![2, 1])),
            ],
        )
    );

    // (2, 3, 5):(4*E(1), 0, 7*E(2, 1)) -- canonical zero basis and static scalar 0 mixed with basis strides
    let expected_zero_mixed = Layout::new(
        int_tuple![
            Int::<1>,
            (Int::<4>, Int::<1>),
            (Int::<1>, (Int::<7>, Int::<1>)),
        ],
        tuple![
            e![0],
            (e![1], Int::<8> * e![1]),
            (e![2, 0], (e![2, 1], Int::<35> * e![2, 1])),
        ],
    );
    assert_eq!(
        assert_complement_strong_coord(Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            tuple![
                Int::<4> * e![1],
                BasisCombination::<Nil>::zero(),
                Int::<7> * e![2, 1]
            ],
        )),
        expected_zero_mixed
    );
    assert_eq!(
        complement(Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            tuple![Int::<4> * e![1], Int::<0>, Int::<7> * e![2, 1]],
        )),
        expected_zero_mixed
    );
}

// 4. `complement_with` (`extend`) cases & cross-checks with `composition`

#[test]
fn test_complement_extend_1d_and_multileaf() {
    // complement(4:2, extend=20) == (2, 3):(1, 8)
    assert_eq!(
        complement_with(Layout::new(Int::<4>, Int::<2>), Int::<20>),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<1>, Int::<8>],
        )
    );

    // Exact coverage drops trailing `1:8` when rank >= 2: complement(4:2, extend=8) == 2:1
    assert_eq!(
        complement_with(Layout::new(Int::<4>, Int::<2>), Int::<8>),
        Layout::new(Int::<2>, Int::<1>)
    );

    // Multi-leaf exact coverage: complement(4:2, extend=(4, 2)) == 2:1
    assert_eq!(
        complement_with(
            Layout::new(Int::<4>, Int::<2>),
            int_tuple![Int::<4>, Int::<2>],
        ),
        Layout::new(Int::<2>, Int::<1>)
    );

    // Standalone `1:d` preserved: complement(4:1, extend=4) == 1:4
    assert_eq!(
        complement_with(Layout::new(Int::<4>, Int::<1>), Int::<4>),
        Layout::new(Int::<1>, Int::<4>)
    );
    assert_eq!(
        complement_with(Layout::new(Int::<4>, Int::<1>), Int::<2>),
        Layout::new(Int::<1>, Int::<4>)
    );
    assert_eq!(
        complement_with(
            Layout::new(Int::<256>, Int::<1>),
            int_tuple![Int::<8>, Int::<8>],
        ),
        Layout::new(Int::<1>, Int::<256>)
    );

    // Zero stride extended: 4:0 extended to 20 -> 20:1
    assert_eq!(
        complement_with(Layout::new(Int::<4>, Int::<0>), Int::<20>),
        Layout::new(Int::<20>, Int::<1>)
    );
    assert_eq!(
        complement_with(Layout::new(Int::<1>, Int::<0>), Int::<1>),
        Layout::new(Int::<1>, Int::<1>)
    );

    // Nil extend treated as extent 1
    assert_eq!(
        complement_with(Layout::new(Int::<4>, Int::<2>), Nil),
        Layout::new(Int::<2>, Int::<1>)
    );
    assert_eq!(
        complement_with(Layout::new(Int::<4>, Int::<1>), Nil),
        Layout::new(Int::<1>, Int::<4>)
    );

    // Multi-mode complement where trailing unit is dropped on length >= 3:
    assert_eq!(
        complement_with(
            Layout::new(
                int_tuple![Int::<2>, Int::<2>],
                int_tuple![Int::<2>, Int::<16>],
            ),
            Int::<32>,
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<4>],
            int_tuple![Int::<1>, Int::<4>],
        )
    );
}

#[test]
fn test_complement_extend_coord_and_higher_rank() {
    // B = 4:E(0), extend = (8, 8) -> (2, 8):(4@0, 1@1)
    assert_eq!(
        complement_with(
            Layout::new(Int::<4>, e![0]),
            int_tuple![Int::<8>, Int::<8>],
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<8>],
            tuple![Int::<4> * e![0], e![1]],
        )
    );

    // B = 4:E(0), extend = (8, (4, 6)) -> nested trailing untouched mode keeps its bases
    assert_eq!(
        complement_with(
            Layout::new(Int::<4>, e![0]),
            int_tuple![Int::<8>, (Int::<4>, Int::<6>)],
        ),
        Layout::new(
            int_tuple![Int::<2>, (Int::<4>, Int::<6>)],
            tuple![Int::<4> * e![0], (e![1, 0], e![1, 1])],
        )
    );

    // B = 4:E(1), extend = (8, 8) -> (8, 2):(1@0, 4@1)
    assert_eq!(
        complement_with(
            Layout::new(Int::<4>, e![1]),
            int_tuple![Int::<8>, Int::<8>],
        ),
        Layout::new(
            int_tuple![Int::<8>, Int::<2>],
            tuple![e![0], Int::<4> * e![1]],
        )
    );

    // B = (4, 4):(E(1), 2*E(0)), extend = (8, 8) -> (2, 2):(1@0, 4@1)
    assert_eq!(
        complement_with(
            Layout::new(
                int_tuple![Int::<4>, Int::<4>],
                tuple![e![1], Int::<2> * e![0]],
            ),
            int_tuple![Int::<8>, Int::<8>],
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<2>],
            tuple![e![0], Int::<4> * e![1]],
        )
    );

    // B = 4:E(0), extend = (4, 8) -> preserves standalone 1:4@0 on mode 0
    assert_eq!(
        complement_with(
            Layout::new(Int::<4>, e![0]),
            int_tuple![Int::<4>, Int::<8>],
        ),
        Layout::new(
            int_tuple![Int::<1>, Int::<8>],
            tuple![Int::<4> * e![0], e![1]],
        )
    );

    // Cross-check with `composition`: logical_divide(24:1, 4:2) == (4, (2, 3)):(2, (1, 8))
    let a = Layout::new(Int::<24>, Int::<1>);
    let b = Layout::new(Int::<4>, Int::<2>);
    let div_res = composition(a, make_layout!(b, complement_with(b, a.shape())));
    assert_eq!(
        div_res,
        Layout::new(
            int_tuple![Int::<4>, (Int::<2>, Int::<3>)],
            int_tuple![Int::<2>, (Int::<1>, Int::<8>)],
        )
    );

    // logical_divide((6, 6):(1, 12), (6, 3):(3, 1)) == (((2, 3), 3), 2):(((3, 12), 1), 36)
    let a2 = Layout::new(
        int_tuple![Int::<6>, Int::<6>],
        int_tuple![Int::<1>, Int::<12>],
    );
    let b2 = Layout::new(
        int_tuple![Int::<6>, Int::<3>],
        int_tuple![Int::<3>, Int::<1>],
    );
    let div_res2 = composition(a2, make_layout!(b2, complement_with(b2, a2.shape())));
    assert_eq!(
        div_res2,
        Layout::new(
            int_tuple![((Int::<2>, Int::<3>), Int::<3>), Int::<2>],
            int_tuple![((Int::<3>, Int::<12>), Int::<1>), Int::<36>],
        )
    );
}

// 5. Dynamic shapes, dynamic strides, and SymPy-equivalent tests

#[test]
fn test_complement_dynamic_and_sympy_equivalents() {
    for n in [1i32, 2, 3, 5] {
        // Layout(N, 1) -> Layout(1, N)
        let r1 = assert_complement_weak(Layout::new(n, Int::<1>));
        assert_eq!(r1, Layout::new(Int::<1>, n));

        // Layout((4, N), (1, 4)) -> Layout(1, 4*N)
        let r2 = assert_complement_weak(Layout::new(
            int_tuple![Int::<4>, n],
            int_tuple![Int::<1>, Int::<4>],
        ));
        assert_eq!(r2, Layout::new(Int::<1>, 4 * n));

        // Layout((2, N), (1, 2)) -> Layout(1, 2*N)
        let r3 = assert_complement_weak(Layout::new(
            int_tuple![Int::<2>, n],
            int_tuple![Int::<1>, Int::<2>],
        ));
        assert_eq!(r3, Layout::new(Int::<1>, 2 * n));

        // Layout((N, 4), (1, N)) -> obeys ordered/disjoint postconditions and R(i) == 4*N*i
        let r4 = assert_complement_weak(Layout::new(
            int_tuple![n, Int::<4>],
            int_tuple![Int::<1>, n],
        ));
        for i in 0..8i32 {
            assert_eq!(r4(i), 4 * n * i);
        }

        // Layout(256, 1).complement_with((32, 4, 2, 2, N)) == Layout(2*N, 256)
        let ext = Layout::new(Int::<256>, Int::<1>).complement_with(int_tuple![
            Int::<32>,
            Int::<4>,
            Int::<2>,
            Int::<2>,
            n
        ]);
        assert_eq!(ext, Layout::new(2 * n, Int::<256>));
    }

    // Layout(N, X) -> Layout((X, 1), (1, N*X))
    for n in [2i32, 4] {
        for x in [1i32, 3, 5] {
            let r = assert_complement_weak(Layout::new(n, x));
            assert_eq!(
                r,
                Layout::new(int_tuple![x, Int::<1>], int_tuple![Int::<1>, n * x])
            );
        }
    }

    // Dynamic degenerate runtime values: d == 0 and s == 1 skip cleanly
    let r_zero_d = assert_complement_weak(Layout::new(4i32, 0i32));
    for i in 0..10i32 {
        assert_eq!(r_zero_d(i), i);
    }
    let r_unit_s = assert_complement_weak(Layout::new(
        int_tuple![Int::<4>, 1i32],
        int_tuple![Int::<1>, 0i32],
    ));
    for i in 0..10i32 {
        assert_eq!(r_unit_s(i), 4 * i);
    }

    // Sorted dynamic strides succeed and pass postconditions
    let r_dyn_sorted = assert_complement_weak(Layout::new(
        int_tuple![4i32, 2i32],
        int_tuple![1i32, 4i32],
    ));
    for i in 0..10i32 {
        assert_eq!(r_dyn_sorted(i), 8 * i);
    }

    // Dynamic extend on static complement keeps static strides
    let r_dyn_ext: PairLayout<Int<2>, int, Int<1>, Int<8>> =
        complement_with(Layout::new(Int::<4>, Int::<2>), 20i32);
    assert_eq!(
        r_dyn_ext,
        Layout::new(int_tuple![Int::<2>, 3i32], int_tuple![Int::<1>, Int::<8>])
    );

    // Dynamic sizeC paired with static Int<1> leaves in extend (`CeilStepAction::UnitShape`)
    let r_unit_shape_ext = complement_with(
        Layout::new(4i32, Int::<1>),
        int_tuple![Int::<1>, Int::<20>],
    );
    assert_eq!(r_unit_shape_ext, Layout::new(5i32, 4i32));

    // Dynamic basis stride (including runtime 0 coefficient) obeys ordered & disjoint postconditions
    let r_dyn_basis = assert_complement_weak(Layout::new(
        Int::<4>,
        BasisTerm::<i32, mode![1]>::new(0i32),
    ));
    assert!(basis_eq(r_dyn_basis(0i32), 0i32));
    assert!(basis_eq(r_dyn_basis(1i32), e![1]));
    assert!(basis_eq(r_dyn_basis(2i32), Int::<2> * e![1]));

    // Dynamic shape on basis axis extended to multi-mode target shape
    let r_dyn_basis_ext = complement_with(
        Layout::new(4i32, e![0]),
        int_tuple![Int::<8>, Int::<8>],
    );
    assert_eq!(
        r_dyn_basis_ext,
        Layout::new(
            int_tuple![2i32, Int::<8>],
            tuple![BasisTerm::<int, mode![0]>::new(4i32), e![1]],
        )
    );
}

// 6. Tilers, Nil, empty/zero basis, const eval, and large `int` range

#[test]
fn test_complement_tilers_nil_and_const_eval() {
    // Const evaluation & zero-sized static output
    const FREE: PairLayout<Int<2>, Int<1>, Int<1>, Int<8>> =
        complement(Layout::new(Int::<4>, Int::<2>));
    const EXT: PairLayout<Int<2>, Int<3>, Int<1>, Int<8>> =
        complement_with(Layout::new(Int::<4>, Int::<2>), Int::<20>);
    assert_eq!(size_of_val(&FREE), 0);
    assert_eq!(size_of_val(&EXT), 0);
    assert_eq!(
        FREE,
        Layout::new(int_tuple![Int::<2>, Int::<1>], int_tuple![Int::<1>, Int::<8>])
    );
    assert_eq!(
        EXT,
        Layout::new(int_tuple![Int::<2>, Int::<3>], int_tuple![Int::<1>, Int::<8>])
    );

    // Tiler inputs
    assert_eq!(complement(Int::<4>), Layout::new(Int::<1>, Int::<4>));
    assert_eq!(complement(4i32), Layout::new(Int::<1>, 4i32));
    assert_eq!(complement(Nil), Layout::new(Int::<1>, Int::<1>));
    assert_eq!(
        complement(int_tuple![Int::<3>, Int::<5>]),
        Layout::new(
            int_tuple![Int::<1>, Int::<1>],
            tuple![Int::<3> * e![0], Int::<5> * e![1]],
        )
    );
    assert_eq!(
        complement_with(
            int_tuple![Int::<3>, Int::<5>],
            int_tuple![Int::<12>, Int::<20>],
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<4>],
            tuple![Int::<3> * e![0], Int::<5> * e![1]],
        )
    );

    // Empty & nested Nil layouts
    assert_eq!(
        complement(Layout::new(Nil, Nil)),
        Layout::new(Int::<1>, Int::<1>)
    );
    assert_eq!(
        complement(Layout::new(
            int_tuple![(), Int::<4>, ()],
            int_tuple![(), Int::<2>, ()],
        )),
        Layout::new(int_tuple![Int::<2>, Int::<1>], int_tuple![Int::<1>, Int::<8>])
    );

    // Canonical zero BasisCombination<Nil> and static-zero BasisTerm
    assert_eq!(
        complement(Layout::new(Int::<4>, BasisCombination::<Nil>::zero())),
        Layout::new(Int::<1>, Int::<1>)
    );
    assert_eq!(
        complement(Layout::new(
            Int::<4>,
            BasisTerm::<Int<0>, mode![1]>::new(Int::<0>),
        )),
        Layout::new(Int::<1>, Int::<1>)
    );

    // Near i32::MAX without overflow
    const BIG_FREE: PairLayout<Int<1_073_741_823>, Int<1>, Int<1>, Int<2_147_483_646>> =
        complement(Layout::new(Int::<2>, Int::<1_073_741_823>));
    assert_eq!(
        BIG_FREE,
        Layout::new(
            int_tuple![Int::<1_073_741_823>, Int::<1>],
            int_tuple![Int::<1>, Int::<2_147_483_646>],
        )
    );

    // Overflow-safe ceil_div near i32::MAX in extend (`s + sizeC` would overflow i32!)
    const BIG_EXT: PairLayout<Int<1_000_000_000>, Int<2>, Int<1>, Int<2_000_000_000>> =
        complement_with(
            Layout::new(Int::<2>, Int::<1_000_000_000>),
            Int::<2_000_000_001>,
        );
    assert_eq!(
        BIG_EXT,
        Layout::new(
            int_tuple![Int::<1_000_000_000>, Int::<2>],
            int_tuple![Int::<1>, Int::<2_000_000_000>],
        )
    );
}

// 7. Dynamic failure modes (`complement` / `complement_with` panics)

#[test]
fn test_complement_dynamic_failures() {
    // Dynamic zero / negative extent with non-zero stride
    assert!(std::panic::catch_unwind(|| complement(Layout::new(0i32, Int::<2>))).is_err());
    assert!(std::panic::catch_unwind(|| complement(Layout::new(0i32, 2i32))).is_err());
    assert!(std::panic::catch_unwind(|| complement(Layout::new(-2i32, Int::<2>))).is_err());
    assert!(std::panic::catch_unwind(|| complement(Layout::new(-2i32, 2i32))).is_err());
    // Dynamic zero extent with static zero stride succeeds (1:1)
    assert_eq!(
        complement(Layout::new(0i32, Int::<0>)),
        Layout::new(Int::<1>, Int::<1>)
    );

    // Dynamic negative stride
    assert!(std::panic::catch_unwind(|| complement(Layout::new(Int::<4>, -2i32))).is_err());
    assert!(std::panic::catch_unwind(|| complement(Layout::new(4i32, -2i32))).is_err());

    // Dynamic overlapping / non-injective strides
    assert!(
        std::panic::catch_unwind(|| {
            complement(Layout::new(
                int_tuple![4i32, 2i32],
                int_tuple![1i32, 2i32],
            ))
        })
        .is_err()
    );
    // Dynamic unsorted strides fail `d >= curr_d`
    assert!(
        std::panic::catch_unwind(|| {
            complement(Layout::new(
                int_tuple![2i32, 4i32],
                int_tuple![4i32, 1i32],
            ))
        })
        .is_err()
    );
    // Dynamic shape causing static next stride to overlap
    assert!(
        std::panic::catch_unwind(|| {
            complement(Layout::new(
                int_tuple![10i32, Int::<2>],
                int_tuple![Int::<1>, Int::<4>],
            ))
        })
        .is_err()
    );

    // Dynamic i32 overflow in d * s
    assert!(
        std::panic::catch_unwind(|| complement(Layout::new(2_000_000_000i32, Int::<2>))).is_err()
    );
    assert!(
        std::panic::catch_unwind(|| complement(Layout::new(2i32, 2_000_000_000i32))).is_err()
    );

    // `complement_with` failures:
    // Dynamic zero / negative extend
    assert!(
        std::panic::catch_unwind(|| complement_with(Layout::new(Int::<4>, Int::<2>), 0i32))
            .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| complement_with(Layout::new(Int::<4>, Int::<2>), -4i32))
            .is_err()
    );
    // Dynamic zero extend when sizeC already reduced to static Int<1>
    assert!(
        std::panic::catch_unwind(|| {
            complement_with(
                Layout::new(Int::<256>, Int::<1>),
                int_tuple![Int::<32>, Int::<4>, Int::<2>, Int::<2>, 0i32],
            )
        })
        .is_err()
    );
    // Dynamic overflow in extend coalesce (`2 * 2_000_000_000` overflows i32)
    assert!(
        std::panic::catch_unwind(|| {
            complement_with(
                Layout::new(Int::<256>, Int::<1>),
                int_tuple![Int::<32>, Int::<4>, Int::<2>, Int::<2>, 2_000_000_000i32],
            )
        })
        .is_err()
    );
    // Dynamic overflow in extend prefix_product (`8 * 125_000 * 10_000` overflows i32)
    assert!(
        std::panic::catch_unwind(|| {
            complement_with(
                Layout::new(Int::<4>, Int::<2>),
                int_tuple![1_000_000i32, 10_000i32, 2i32],
            )
        })
        .is_err()
    );
}

#[test]
#[should_panic]
fn test_complement_panics_on_dynamic_non_injective() {
    let _ = complement(Layout::new(
        int_tuple![4i32, 2i32],
        int_tuple![1i32, 2i32],
    ));
}

#[test]
#[should_panic]
fn test_complement_with_panics_on_dynamic_zero_extend() {
    let _ = complement_with(Layout::new(Int::<4>, Int::<2>), 0i32);
}
