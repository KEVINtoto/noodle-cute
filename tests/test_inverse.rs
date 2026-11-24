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
use noodle_cute::algebra::inverse::{LeftInverse, RightInverse, left_inverse, right_inverse};
use noodle_cute::basis::{BasisCombination, BasisTerm, IntoColexTerms, MergeEq, basis_eq};
use noodle_cute::integer::{Int, Integer, eq as int_eq, int};
use noodle_cute::layout::{Coprofile, Layout, tiler_to_layout};
use noodle_cute::shape::{IndexToCoord, Shape};
use noodle_cute::stride::{InnerProduct, Stride};
use noodle_cute::tuple::{Item, Nil, TupleContainer, WeaklyCongruent};
use noodle_cute::{e, int_tuple, make_layout, mode, tuple};

// Post-condition checkers mirroring `test_inverse_right.py` & `test_inverse_left.py`

type PairLayout<S0, S1, D0, D1> = Layout<
    TupleContainer<S0, TupleContainer<S1, Nil>>,
    TupleContainer<D0, TupleContainer<D1, Nil>>,
>;
type TripleLayout<S0, S1, S2, D0, D1, D2> = Layout<
    TupleContainer<S0, TupleContainer<S1, TupleContainer<S2, Nil>>>,
    TupleContainer<D0, TupleContainer<D1, TupleContainer<D2, Nil>>>,
>;

fn assert_weak_congruent<A: WeaklyCongruent<B>, B: Item>(_: &A, _: &B) {}

fn assert_right_inverse_1d<SL, DL, SR, DR>(l: Layout<SL, DL>) -> Layout<SR, DR>
where
    SL: Shape + Debug,
    DL: Stride<SL> + Debug,
    SR: Shape + Debug,
    DR: Stride<SR> + Debug,
    Layout<SL, DL>: RightInverse<Output = Layout<SR, DR>> + Coprofile,
    <Layout<SL, DL> as Coprofile>::Output: WeaklyCongruent<SR>,
    int: IndexToCoord<SR>,
    <int as IndexToCoord<SR>>::Output: InnerProduct<DR>,
    <<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output:
        Integer + IndexToCoord<SL> + Debug,
    <<<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IndexToCoord<SL>>::Output:
        InnerProduct<DL>,
    <<<<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output:
        Integer + IndexToCoord<SR> + Debug,
    <<<<<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IndexToCoord<SR>>::Output:
        InnerProduct<DR>,
    <<<<<<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output:
        Integer + Debug,
{
    let inv = right_inverse(l);
    assert_weak_congruent(&l.coprofile(), &inv.shape());

    for i in 0..inv.size() {
        let ri = inv(i);
        let lri = l(ri);
        let rlri = inv(lri);
        assert!(
            int_eq(rlri, ri),
            "generalized right inverse failed for L={l:?}, R={inv:?} at i={i}: R(L(R(i)))={rlri:?} != R(i)={ri:?}"
        );
        assert!(
            int_eq(lri, i),
            "canonical right inverse failed for L={l:?}, R={inv:?} at i={i}: L(R(i))={lri:?} != {i}"
        );
    }

    inv
}

fn assert_right_inverse_coord<SL, DL, SR, DR>(l: Layout<SL, DL>) -> Layout<SR, DR>
where
    SL: Shape + Debug,
    DL: Stride<SL> + Debug,
    SR: Shape + Debug,
    DR: Stride<SR> + Debug,
    Layout<SL, DL>: RightInverse<Output = Layout<SR, DR>> + Coprofile,
    <Layout<SL, DL> as Coprofile>::Output: WeaklyCongruent<SR>,
    int: IndexToCoord<SR>,
    <int as IndexToCoord<SR>>::Output: InnerProduct<DR>,
    <<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output:
        Integer + IndexToCoord<SL> + Debug,
    <<<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IndexToCoord<SL>>::Output:
        InnerProduct<DL>,
    <<<<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output:
        IndexToCoord<SR> + Debug,
    <<<<<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IndexToCoord<SR>>::Output:
        InnerProduct<DR>,
    <<<<<<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output:
        Integer + Debug,
{
    let inv = right_inverse(l);
    assert_weak_congruent(&l.coprofile(), &inv.shape());

    for i in 0..inv.size() {
        let ri = inv(i);
        let lri = l(ri);
        let rlri = inv(lri);
        assert!(
            int_eq(rlri, ri),
            "coord right inverse failed for L={l:?}, R={inv:?} at i={i}: R(L(R(i)))={rlri:?} != R(i)={ri:?}"
        );
    }

    inv
}

fn assert_left_inverse_1d<SL, DL, SR, DR>(l: Layout<SL, DL>) -> Layout<SR, DR>
where
    SL: Shape + Debug,
    DL: Stride<SL> + Debug,
    SR: Shape + Debug,
    DR: Stride<SR> + Debug,
    Layout<SL, DL>: LeftInverse<Output = Layout<SR, DR>> + Coprofile,
    <Layout<SL, DL> as Coprofile>::Output: WeaklyCongruent<SR>,
    int: IndexToCoord<SL>,
    <int as IndexToCoord<SL>>::Output: InnerProduct<DL>,
    <<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output:
        Integer + IndexToCoord<SR> + Debug,
    <<<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IndexToCoord<SR>>::Output:
        InnerProduct<DR>,
    <<<<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output:
        Integer + IndexToCoord<SL> + Debug,
    <<<<<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IndexToCoord<SL>>::Output:
        InnerProduct<DL>,
    <<<<<<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output:
        Integer + Debug,
{
    let inv = left_inverse(l);
    assert_weak_congruent(&l.coprofile(), &inv.shape());

    for i in 0..l.size() {
        let li = l(i);
        let rli = inv(li);
        let lrli = l(rli);
        assert!(
            int_eq(lrli, li),
            "left inverse failed for L={l:?}, R={inv:?} at i={i}: L(R(L(i)))={lrli:?} != L(i)={li:?}"
        );
    }

    inv
}

fn assert_left_inverse_coord<SL, DL, SR, DR>(l: Layout<SL, DL>) -> Layout<SR, DR>
where
    SL: Shape + Debug,
    DL: Stride<SL> + Debug,
    SR: Shape + Debug,
    DR: Stride<SR> + Debug,
    Layout<SL, DL>: LeftInverse<Output = Layout<SR, DR>> + Coprofile,
    <Layout<SL, DL> as Coprofile>::Output: WeaklyCongruent<SR>,
    int: IndexToCoord<SL>,
    <int as IndexToCoord<SL>>::Output: InnerProduct<DL>,
    <<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output:
        IndexToCoord<SR> + IntoColexTerms + Debug,
    <<<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IndexToCoord<SR>>::Output:
        InnerProduct<DR>,
    <<<<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output:
        Integer + IndexToCoord<SL> + Debug,
    <<<<<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IndexToCoord<SL>>::Output:
        InnerProduct<DL>,
    <<<<<<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output:
        IntoColexTerms + Debug,
    <<<<<<<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IntoColexTerms>::Terms:
        MergeEq<
            <<<int as IndexToCoord<SL>>::Output as InnerProduct<DL>>::Output as IntoColexTerms>::Terms,
        >,
{
    let inv = left_inverse(l);
    assert_weak_congruent(&l.coprofile(), &inv.shape());

    for i in 0..l.size() {
        let li = l(i);
        let rli = inv(li);
        let lrli = l(rli);
        assert!(
            basis_eq(lrli, li),
            "coord left inverse failed for L={l:?}, R={inv:?} at i={i}: L(R(L(i)))={lrli:?} != L(i)={li:?}"
        );
    }

    inv
}

#[test]
fn test_right_inverse_1d_suite() {
    assert_eq!(
        assert_right_inverse_1d(Layout::new(Int::<1>, Int::<0>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(Int::<1>, Int::<1>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(Int::<1>, Int::<2>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(Int::<1>, Int::<4>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(Int::<4>, Int::<0>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(Int::<4>, Int::<1>)),
        Layout::new(Int::<4>, Int::<1>)
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(Int::<4>, Int::<2>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(Int::<4>, Int::<4>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(
            int_tuple![Int::<1>, Int::<1>],
            int_tuple![Int::<0>, Int::<0>],
        )),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(
            int_tuple![Int::<3>, Int::<7>],
            int_tuple![Int::<0>, Int::<0>],
        )),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(
            int_tuple![Int::<2>, Int::<4>],
            int_tuple![Int::<0>, Int::<2>],
        )),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(
            int_tuple![Int::<8>, Int::<4>],
            int_tuple![Int::<1>, Int::<8>],
        )),
        Layout::new(Int::<32>, Int::<1>)
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(
            int_tuple![Int::<8>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>],
        )),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>]
        )
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>],
        )),
        Layout::new(
            int_tuple![Int::<8>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>]
        )
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(
            int_tuple![Int::<2>, Int::<4>, Int::<6>],
            int_tuple![Int::<1>, Int::<2>, Int::<8>],
        )),
        Layout::new(Int::<48>, Int::<1>)
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(
            int_tuple![Int::<2>, Int::<4>, Int::<6>],
            int_tuple![Int::<4>, Int::<1>, Int::<8>],
        )),
        Layout::new(
            int_tuple![Int::<4>, Int::<2>, Int::<6>],
            int_tuple![Int::<2>, Int::<1>, Int::<8>],
        )
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(
            int_tuple![Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<16>],
        )),
        Layout::new(Int::<4>, Int::<1>)
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<1>, Int::<5>],
        )),
        Layout::new(Int::<4>, Int::<1>)
    );

    // Non-injective layouts: overlapping modes are skipped and the chain continues where possible
    assert_eq!(
        assert_right_inverse_1d(Layout::new(
            int_tuple![Int::<4>, Int::<5>, Int::<6>],
            int_tuple![Int::<1>, Int::<1>, Int::<4>],
        )),
        Layout::new(
            int_tuple![Int::<4>, Int::<6>],
            int_tuple![Int::<1>, Int::<20>],
        )
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(
            int_tuple![Int::<7>, Int::<5>, Int::<9>],
            int_tuple![Int::<2>, Int::<0>, Int::<1>],
        )),
        Layout::new(Int::<9>, Int::<35>)
    );

    // Negative strides do not continue the positive chain and are cleanly skipped
    assert_eq!(
        assert_right_inverse_1d(Layout::new(Int::<4>, Int::<-1>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_right_inverse_1d(Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<-1>, Int::<1>],
        )),
        Layout::new(Int::<8>, Int::<4>)
    );
}

#[test]
fn test_right_inverse_coord_suite() {
    assert_eq!(
        assert_right_inverse_coord(Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            tuple![e![0], e![1]],
        )),
        Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );
    assert_eq!(
        assert_right_inverse_coord(Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            tuple![e![1], e![0]],
        )),
        Layout::new(
            int_tuple![Int::<5>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>]
        )
    );
    assert_eq!(
        assert_right_inverse_coord(Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            tuple![e![1], e![4, 1]],
        )),
        Layout::new(
            int_tuple![Int::<1>, Int::<4>, Int::<1>, Int::<1>, (Int::<1>, Int::<5>)],
            int_tuple![Int::<0>, Int::<1>, Int::<0>, Int::<0>, (Int::<0>, Int::<4>)],
        )
    );
    assert_eq!(
        assert_right_inverse_coord(Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            tuple![Int::<2> * e![0], Int::<2> * e![1]],
        )),
        Layout::new(
            int_tuple![Int::<1>, Int::<1>],
            int_tuple![Int::<0>, Int::<0>]
        )
    );

    // Size-1 mode carrying an extra codomain axis preserves coprofile
    assert_eq!(
        assert_right_inverse_coord(Layout::new(
            int_tuple![Int::<1>, Int::<4>],
            tuple![e![1], e![0]],
        )),
        Layout::new(
            int_tuple![Int::<4>, Int::<1>],
            int_tuple![Int::<1>, Int::<0>]
        )
    );

    // SM70 MMA 8x8x4 C TV inverse
    assert_eq!(
        assert_right_inverse_coord(Layout::new(
            int_tuple![(Int::<2>, Int::<2>, Int::<2>), (Int::<2>, Int::<2>, Int::<2>)],
            tuple![
                (e![0], Int::<2> * e![1], Int::<4> * e![0]),
                (e![1], Int::<2> * e![0], Int::<4> * e![1]),
            ],
        )),
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>, Int::<2>), (Int::<2>, Int::<2>, Int::<2>)],
            int_tuple![(Int::<1>, Int::<16>, Int::<4>), (Int::<8>, Int::<2>, Int::<32>)],
        )
    );
    assert_eq!(
        assert_right_inverse_coord(Layout::new(
            int_tuple![(Int::<2>, Int::<2>, Int::<2>), (Int::<2>, Int::<2>, Int::<2>)],
            tuple![
                (e![0], Int::<2> * e![1], Int::<5> * e![0]),
                (e![1], Int::<2> * e![0], Int::<5> * e![1]),
            ],
        )),
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<2>)],
            int_tuple![(Int::<1>, Int::<16>), (Int::<8>, Int::<2>)],
        )
    );
    assert_eq!(
        assert_right_inverse_coord(Layout::new(
            int_tuple![(Int::<2>, Int::<2>, Int::<2>), (Int::<2>, Int::<2>, Int::<2>)],
            tuple![
                (e![0], Int::<2> * e![1], Int::<5> * e![0]),
                (e![1], Int::<2> * e![0], Int::<4> * e![1]),
            ],
        )),
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<2>, Int::<2>)],
            int_tuple![(Int::<1>, Int::<16>), (Int::<8>, Int::<2>, Int::<32>)],
        )
    );

    // SM70 MMA 8x8x4 A TV inverse
    let sm70_a = composition(
        tiler_to_layout(int_tuple![Int::<8>, Int::<4>]),
        Layout::new(
            int_tuple![(Int::<4>, Int::<2>), Int::<4>],
            int_tuple![(Int::<8>, Int::<4>), Int::<1>],
        ),
    );
    assert_eq!(
        assert_right_inverse_coord(sm70_a),
        Layout::new(
            int_tuple![(Int::<4>, Int::<2>), Int::<4>],
            int_tuple![(Int::<8>, Int::<4>), Int::<1>],
        )
    );

    // SM80 MMA 16x8 TV inverse
    let sm80 = composition(
        tiler_to_layout(int_tuple![Int::<16>, Int::<8>]),
        Layout::new(
            int_tuple![(Int::<4>, Int::<8>), (Int::<2>, Int::<2>)],
            int_tuple![(Int::<32>, Int::<1>), (Int::<16>, Int::<8>)],
        ),
    );
    assert_eq!(
        assert_right_inverse_coord(sm80),
        Layout::new(
            int_tuple![(Int::<8>, Int::<2>), (Int::<2>, Int::<4>)],
            int_tuple![(Int::<4>, Int::<64>), (Int::<32>, Int::<1>)],
        )
    );
}

#[test]
fn test_right_inverse_dynamic_and_sympy_equivalents() {
    for &(n, m, x) in &[(1i32, 1i32, 1i32), (2, 3, 2), (3, 2, 5), (5, 4, 3)] {
        // Layout(N, 1) -> Layout(N, 1)
        assert_eq!(
            assert_right_inverse_1d(Layout::new(n, Int::<1>)),
            Layout::new(n, Int::<1>)
        );

        // Layout((4, N), (1, 4)) -> Layout(4*N, 1)
        assert_eq!(
            assert_right_inverse_1d(Layout::new(
                int_tuple![Int::<4>, n],
                int_tuple![Int::<1>, Int::<4>],
            )),
            Layout::new(4 * n, Int::<1>)
        );

        // Layout((N, 4), (1, N)) -> full size 4*N right inverse
        let r_n4 = assert_right_inverse_1d(Layout::new(
            int_tuple![n, Int::<4>],
            int_tuple![Int::<1>, n],
        ));
        assert_eq!(r_n4.size(), 4 * n);

        // Layout((N, M), (1, N)) -> full size N*M right inverse
        let r_nm = assert_right_inverse_1d(Layout::new(
            int_tuple![n, m],
            int_tuple![Int::<1>, n],
        ));
        assert_eq!(r_nm.size(), n * m);

        // Layout((N, M), (M, 1)) -> full size N*M transposed right inverse
        let r_mn = assert_right_inverse_1d(Layout::new(
            int_tuple![n, m],
            int_tuple![m, Int::<1>],
        ));
        assert_eq!(r_mn.size(), n * m);

        // Layout(N, X): full inverse if X == 1, trivial 1:0 otherwise
        let r_nx = assert_right_inverse_1d(Layout::new(n, x));
        assert_eq!(r_nx.size(), if x == 1 { n } else { 1 });

        // Layout((N, M), (1, 2*N)) -> stops at gap, covers first N elements
        let r_gap = assert_right_inverse_1d(Layout::new(
            int_tuple![n, m],
            int_tuple![Int::<1>, 2 * n],
        ));
        assert_eq!(r_gap.size(), n);

        // Layout((N, 2), (1, 1)) -> non-injective when N >= 2, full size 2 when N == 1
        let r_dup = assert_right_inverse_1d(Layout::new(
            int_tuple![n, Int::<2>],
            int_tuple![Int::<1>, Int::<1>],
        ));
        assert_eq!(r_dup.size(), if n == 1 { 2 } else { n });

        // Layout(N, 0) -> 1:0
        assert_eq!(
            assert_right_inverse_1d(Layout::new(n, Int::<0>)),
            Layout::new(Int::<1>, Int::<0>)
        );

        // Layout((N, M), (0, N)): size M when N == 1, size 1 otherwise
        let r_zero_n = assert_right_inverse_1d(Layout::new(
            int_tuple![n, m],
            int_tuple![Int::<0>, n],
        ));
        assert_eq!(r_zero_n.size(), if n == 1 { m } else { 1 });
    }

    // Dynamic strides: sorted succeeds fully, unsorted still produces a valid right inverse
    let r_dyn_sorted = assert_right_inverse_1d(Layout::new(
        int_tuple![4i32, 8i32],
        int_tuple![1i32, 4i32],
    ));
    assert_eq!(r_dyn_sorted.size(), 32);

    let r_dyn_unsorted = assert_right_inverse_1d(Layout::new(
        int_tuple![4i32, 8i32],
        int_tuple![8i32, 1i32],
    ));
    assert_eq!(r_dyn_unsorted.size(), 8);

    // Dynamic negative / zero strides
    assert_eq!(
        assert_right_inverse_1d(Layout::new(4i32, -2i32)).size(),
        1
    );
    assert_eq!(assert_right_inverse_1d(Layout::new(4i32, 0i32)).size(), 1);

    // Dynamic basis stride (including runtime 0)
    let r_dyn_basis_zero = assert_right_inverse_coord(Layout::new(
        Int::<4>,
        BasisTerm::<i32, mode![1]>::new(0i32),
    ));
    assert_eq!(r_dyn_basis_zero.size(), 1);
}

#[test]
fn test_left_inverse_1d_suite() {
    assert_eq!(
        assert_left_inverse_1d(Layout::new(Int::<1>, Int::<0>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(Int::<1>, Int::<1>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(Int::<1>, Int::<2>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(Int::<1>, Int::<4>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<1>, Int::<1>],
            int_tuple![Int::<0>, Int::<0>],
        )),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<3>, Int::<7>],
            int_tuple![Int::<0>, Int::<0>],
        )),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(Int::<4>, Int::<0>)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(Int::<4>, Int::<1>)),
        Layout::new(Int::<4>, Int::<1>)
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(Int::<4>, Int::<2>)),
        Layout::new(
            int_tuple![Int::<2>, Int::<4>],
            int_tuple![Int::<0>, Int::<1>]
        )
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(Int::<4>, Int::<4>)),
        Layout::new(
            int_tuple![Int::<4>, Int::<4>],
            int_tuple![Int::<0>, Int::<1>]
        )
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<8>, Int::<4>],
            int_tuple![Int::<1>, Int::<8>],
        )),
        Layout::new(Int::<32>, Int::<1>)
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<8>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>],
        )),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>]
        )
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<2>, Int::<4>, Int::<6>],
            int_tuple![Int::<1>, Int::<2>, Int::<8>],
        )),
        Layout::new(Int::<48>, Int::<1>)
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<2>, Int::<4>, Int::<6>],
            int_tuple![Int::<4>, Int::<1>, Int::<8>],
        )),
        Layout::new(
            int_tuple![Int::<4>, Int::<2>, Int::<6>],
            int_tuple![Int::<2>, Int::<1>, Int::<8>],
        )
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<2>, Int::<4>, Int::<8>],
            int_tuple![Int::<32>, Int::<0>, Int::<2>],
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<16>, Int::<2>],
            int_tuple![Int::<0>, Int::<8>, Int::<1>],
        )
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<2>, Int::<4>, Int::<8>],
            int_tuple![Int::<2>, Int::<0>, Int::<32>],
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<16>, Int::<8>],
            int_tuple![Int::<0>, Int::<1>, Int::<8>],
        )
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<2>, Int::<4>, Int::<4>, Int::<4>, Int::<2>],
            int_tuple![Int::<32>, Int::<0>, Int::<2>, Int::<0>, Int::<512>],
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<16>, Int::<16>, Int::<2>],
            int_tuple![Int::<0>, Int::<8>, Int::<1>, Int::<128>],
        )
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<16>],
        )),
        Layout::new(
            int_tuple![Int::<16>, Int::<2>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<5>],
        )),
        Layout::new(
            int_tuple![Int::<5>, Int::<2>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<10>],
        )),
        Layout::new(
            int_tuple![Int::<10>, Int::<2>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<11>],
        )),
        Layout::new(
            int_tuple![Int::<11>, Int::<2>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );

    // TMEM inspired layouts from `test_inverse_left.py`
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<32>, Int::<8>],
            int_tuple![Int::<65536>, Int::<1>],
        )),
        Layout::new(
            int_tuple![Int::<65536>, Int::<32>],
            int_tuple![Int::<32>, Int::<1>]
        )
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<32>, Int::<12>],
            int_tuple![Int::<65536>, Int::<1>],
        )),
        Layout::new(
            int_tuple![Int::<65536>, Int::<32>],
            int_tuple![Int::<32>, Int::<1>]
        )
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<32>, Int::<3>, Int::<8>],
            int_tuple![Int::<65536>, Int::<512>, Int::<1>],
        )),
        Layout::new(
            int_tuple![Int::<512>, Int::<128>, Int::<32>],
            int_tuple![Int::<96>, Int::<32>, Int::<1>],
        )
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![Int::<32>, Int::<8>],
            int_tuple![Int::<131072>, Int::<2>],
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<65536>, Int::<32>],
            int_tuple![Int::<0>, Int::<32>, Int::<1>],
        )
    );
    assert_eq!(
        assert_left_inverse_1d(Layout::new(
            int_tuple![
                ((((Int::<2>, Int::<4>), Int::<1>), (Int::<2>, Int::<2>)), Int::<4>),
                Int::<1>,
                (Int::<2>, Int::<2>),
                Int::<2>
            ],
            int_tuple![
                ((((Int::<262144>, Int::<4>), Int::<0>), (Int::<0>, Int::<1>)), Int::<8388608>),
                Int::<0>,
                (Int::<2>, Int::<16>),
                Int::<32>
            ],
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<2>, Int::<4>, Int::<16384>, Int::<128>],
            int_tuple![Int::<16>, Int::<128>, Int::<2>, Int::<256>, Int::<1>],
        )
    );
}

#[test]
fn test_left_inverse_coord_suite() {
    assert_eq!(
        assert_left_inverse_coord(Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            tuple![e![0], e![1]],
        )),
        Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );
    assert_eq!(
        assert_left_inverse_coord(Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            tuple![e![1], e![0]],
        )),
        Layout::new(
            int_tuple![Int::<5>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>]
        )
    );
    assert_eq!(
        assert_left_inverse_coord(Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            tuple![e![1], e![4, 1]],
        )),
        Layout::new(
            int_tuple![Int::<1>, Int::<4>, Int::<1>, Int::<1>, (Int::<1>, Int::<5>)],
            int_tuple![Int::<0>, Int::<1>, Int::<0>, Int::<0>, (Int::<0>, Int::<4>)],
        )
    );
    assert_eq!(
        assert_left_inverse_coord(Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            tuple![Int::<2> * e![0], Int::<2> * e![1]],
        )),
        Layout::new(
            int_tuple![(Int::<2>, Int::<4>), (Int::<2>, Int::<5>)],
            int_tuple![(Int::<0>, Int::<1>), (Int::<0>, Int::<4>)],
        )
    );
    assert_eq!(
        assert_left_inverse_coord(Layout::new(
            int_tuple![Int::<3>, (Int::<2>, Int::<2>)],
            tuple![Int::<34> * e![0], (Int::<2> * e![0], Int::<2> * e![1])],
        )),
        Layout::new(
            int_tuple![(Int::<2>, Int::<17>, Int::<3>), (Int::<2>, Int::<2>)],
            int_tuple![(Int::<0>, Int::<3>, Int::<1>), (Int::<0>, Int::<6>)],
        )
    );

    // Size-1 mode carrying an extra codomain axis preserves coprofile
    assert_eq!(
        assert_left_inverse_coord(Layout::new(
            int_tuple![Int::<1>, Int::<4>],
            tuple![e![1], e![0]],
        )),
        Layout::new(
            int_tuple![Int::<4>, Int::<1>],
            int_tuple![Int::<1>, Int::<0>]
        )
    );

    // SM70 MMA 8x8x4 C TV inverse
    assert_eq!(
        assert_left_inverse_coord(Layout::new(
            int_tuple![(Int::<2>, Int::<2>, Int::<2>), (Int::<2>, Int::<2>, Int::<2>)],
            tuple![
                (e![0], Int::<2> * e![1], Int::<4> * e![0]),
                (e![1], Int::<2> * e![0], Int::<4> * e![1]),
            ],
        )),
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>, Int::<2>), (Int::<2>, Int::<2>, Int::<2>)],
            int_tuple![(Int::<1>, Int::<16>, Int::<4>), (Int::<8>, Int::<2>, Int::<32>)],
        )
    );
    assert_eq!(
        assert_left_inverse_coord(Layout::new(
            int_tuple![(Int::<2>, Int::<2>, Int::<2>), (Int::<2>, Int::<2>, Int::<2>)],
            tuple![
                (e![0], Int::<2> * e![1], Int::<6> * e![0]),
                (e![1], Int::<2> * e![0], Int::<6> * e![1]),
            ],
        )),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>, Int::<2>), (Int::<2>, Int::<3>, Int::<2>)],
            int_tuple![(Int::<1>, Int::<16>, Int::<4>), (Int::<8>, Int::<2>, Int::<32>)],
        )
    );
    assert_eq!(
        assert_left_inverse_coord(Layout::new(
            int_tuple![(Int::<2>, Int::<2>, Int::<2>), (Int::<2>, Int::<2>, Int::<2>)],
            tuple![
                (e![0], Int::<2> * e![1], Int::<6> * e![0]),
                (e![1], Int::<2> * e![0], Int::<4> * e![1]),
            ],
        )),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>, Int::<2>), (Int::<2>, Int::<2>, Int::<2>)],
            int_tuple![(Int::<1>, Int::<16>, Int::<4>), (Int::<8>, Int::<2>, Int::<32>)],
        )
    );

    // SM70 MMA 8x8x4 A TV inverse
    let sm70_a = composition(
        tiler_to_layout(int_tuple![Int::<8>, Int::<4>]),
        Layout::new(
            int_tuple![(Int::<4>, Int::<2>), Int::<4>],
            int_tuple![(Int::<8>, Int::<4>), Int::<1>],
        ),
    );
    assert_eq!(
        assert_left_inverse_coord(sm70_a),
        Layout::new(
            int_tuple![(Int::<4>, Int::<2>), Int::<4>],
            int_tuple![(Int::<8>, Int::<4>), Int::<1>],
        )
    );

    // SM80 MMA 16x8 TV inverse
    let sm80 = composition(
        tiler_to_layout(int_tuple![Int::<16>, Int::<8>]),
        Layout::new(
            int_tuple![(Int::<4>, Int::<8>), (Int::<2>, Int::<2>)],
            int_tuple![(Int::<32>, Int::<1>), (Int::<16>, Int::<8>)],
        ),
    );
    assert_eq!(
        assert_left_inverse_coord(sm80),
        Layout::new(
            int_tuple![(Int::<8>, Int::<2>), (Int::<2>, Int::<4>)],
            int_tuple![(Int::<4>, Int::<64>), (Int::<32>, Int::<1>)],
        )
    );
}

#[test]
fn test_left_inverse_dynamic_and_sympy_equivalents() {
    for &(n, m, x) in &[(1i32, 1i32, 1i32), (2, 3, 2), (3, 2, 5), (5, 4, 3)] {
        // Layout(N, 1) -> Layout(N, 1)
        assert_eq!(
            assert_left_inverse_1d(Layout::new(n, Int::<1>)),
            Layout::new(n, Int::<1>)
        );

        // Layout((4, N), (1, 4)) -> Layout(4*N, 1)
        assert_eq!(
            assert_left_inverse_1d(Layout::new(
                int_tuple![Int::<4>, n],
                int_tuple![Int::<1>, Int::<4>],
            )),
            Layout::new(4 * n, Int::<1>)
        );

        // Layout((N, 4), (1, N))
        assert_left_inverse_1d(Layout::new(
            int_tuple![n, Int::<4>],
            int_tuple![Int::<1>, n],
        ));

        // Layout((N, M), (1, N))
        assert_left_inverse_1d(Layout::new(
            int_tuple![n, m],
            int_tuple![Int::<1>, n],
        ));

        // Layout((N, M), (M, 1))
        assert_left_inverse_1d(Layout::new(
            int_tuple![n, m],
            int_tuple![m, Int::<1>],
        ));

        // Layout(N, X)
        let r_nx = assert_left_inverse_1d(Layout::new(n, x));
        if n > 1 {
            assert_eq!(
                r_nx,
                Layout::new(int_tuple![x, n], int_tuple![Int::<0>, Int::<1>])
            );
        }

        // Layout((N, M), (1, 2*N))
        let r_2n = assert_left_inverse_1d(Layout::new(
            int_tuple![n, m],
            int_tuple![Int::<1>, 2 * n],
        ));
        if m > 1 {
            assert_eq!(
                r_2n,
                Layout::new(int_tuple![2 * n, m], int_tuple![Int::<1>, n])
            );
        }

        // Layout(N, 0) -> Layout(1, 0)
        assert_eq!(
            assert_left_inverse_1d(Layout::new(n, Int::<0>)),
            Layout::new(Int::<1>, Int::<0>)
        );

        // Layout((N, M), (0, N))
        let r_zero_n = assert_left_inverse_1d(Layout::new(
            int_tuple![n, m],
            int_tuple![Int::<0>, n],
        ));
        if m > 1 {
            assert_eq!(
                r_zero_n,
                Layout::new(int_tuple![n, m], int_tuple![Int::<0>, n])
            );
        }
    }

    // Coalescing merges trailing `(4, 2):(8, 32) -> 8:8` before folding
    let r_coalesced_tail = assert_left_inverse_1d(Layout::new(
        int_tuple![3i32, Int::<4>, Int::<2>],
        int_tuple![Int::<1>, Int::<8>, Int::<32>],
    ));
    assert_eq!(
        r_coalesced_tail,
        Layout::new(int_tuple![8i32, Int::<8>], int_tuple![Int::<1>, 3i32],)
    );

    // Dynamic `PrevS` followed by static `D` and static `S` (`DynamicNoSkip`) recovering static `PrevD`
    // so the third mode's gap `64 / 8 = Int<8>` is computed purely at compile time!
    let r_recover: TripleLayout<int, Int<8>, Int<2>, Int<1>, int, int> =
        assert_left_inverse_1d(Layout::new(
            int_tuple![3i32, Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<8>, Int::<64>],
        ));
    assert_eq!(
        r_recover,
        Layout::new(
            int_tuple![8i32, Int::<8>, Int::<2>],
            int_tuple![Int::<1>, 3i32, 12i32],
        )
    );

    // Dynamic `n == 1` on `Layout((n, Int<2>), (Int<1>, Int<1>))` succeeds via `DynamicNoSkip`
    assert_left_inverse_1d(Layout::new(
        int_tuple![1i32, Int::<2>],
        int_tuple![Int::<1>, Int::<1>],
    ));

    // Dynamic `StaticMatchDynamicNext` overflow saturation on `(2B, 2):(2, 1)`
    let r_big_dyn = right_inverse(Layout::new(
        int_tuple![2_000_000_000i32, Int::<2>],
        int_tuple![Int::<2>, Int::<1>],
    ));
    assert_eq!(
        r_big_dyn,
        Layout::new(
            int_tuple![Int::<2>, 2_000_000_000i32],
            int_tuple![2_000_000_000i32, Int::<1>],
        )
    );

    // Dynamic stride 0 and size 1 skipped cleanly at runtime
    assert_left_inverse_1d(Layout::new(
        int_tuple![4i32, 8i32],
        int_tuple![0i32, 2i32],
    ));
    assert_left_inverse_1d(Layout::new(
        int_tuple![1i32, 4i32],
        int_tuple![-2i32, 1i32],
    ));
    assert_left_inverse_coord(Layout::new(
        Int::<4>,
        BasisTerm::<i32, mode![1]>::new(0i32),
    ));
}

#[test]
fn test_left_inverse_app_cotiling() {
    // Mirrors `test_inverse_left.py::test_left_inverse_app`:
    //   D o (Di o TV) == TV  (coalesced)
    let atom_tv_layout = Layout::new(
        int_tuple![(Int::<32>, Int::<4>), (Int::<16>, Int::<32>)],
        int_tuple![(Int::<0>, Int::<2097152>), (Int::<1>, Int::<65536>)],
    );
    let data_layout = Layout::new(
        int_tuple![Int::<128>, Int::<16>],
        int_tuple![Int::<65536>, Int::<1>],
    );
    let inv_data_layout = make_layout!(left_inverse(data_layout), Layout::new(Int::<1>, Int::<0>));
    let layout_tv_data = composition(inv_data_layout, atom_tv_layout);
    assert_eq!(
        composition(data_layout, layout_tv_data).coalesce(),
        atom_tv_layout.coalesce()
    );
}

#[test]
fn test_inverse_tilers_nil_const_eval_and_int_range() {
    // Const evaluation & zero-sized static outputs
    const R_CONST: PairLayout<Int<8>, Int<4>, Int<4>, Int<1>> = right_inverse(Layout::new(
        int_tuple![Int::<4>, Int::<8>],
        int_tuple![Int::<8>, Int::<1>],
    ));
    const L_CONST: PairLayout<Int<5>, Int<8>, Int<1>, Int<4>> = left_inverse(Layout::new(
        int_tuple![Int::<4>, Int::<8>],
        int_tuple![Int::<1>, Int::<5>],
    ));
    assert_eq!(size_of_val(&R_CONST), 0);
    assert_eq!(size_of_val(&L_CONST), 0);
    assert_eq!(
        R_CONST,
        Layout::new(
            int_tuple![Int::<8>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>]
        )
    );
    assert_eq!(
        L_CONST,
        Layout::new(
            int_tuple![Int::<5>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );

    // Inherent Layout methods match free functions
    let sample = Layout::new(
        int_tuple![Int::<4>, Int::<8>],
        int_tuple![Int::<1>, Int::<5>],
    );
    assert_eq!(sample.right_inverse(), right_inverse(sample));
    assert_eq!(sample.left_inverse(), left_inverse(sample));

    // Tilers and Nil
    assert_eq!(right_inverse(Int::<4>), Layout::new(Int::<4>, Int::<1>));
    assert_eq!(left_inverse(Int::<4>), Layout::new(Int::<4>, Int::<1>));
    assert_eq!(right_inverse(4i32), Layout::new(4i32, Int::<1>));
    assert_eq!(left_inverse(4i32), Layout::new(4i32, Int::<1>));
    assert_eq!(right_inverse(Nil), Layout::new(Int::<1>, Int::<0>));
    assert_eq!(left_inverse(Nil), Layout::new(Int::<1>, Int::<0>));
    assert_eq!(
        right_inverse(int_tuple![Int::<4>, Int::<5>]),
        Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );
    assert_eq!(
        left_inverse(int_tuple![Int::<4>, Int::<5>]),
        Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );

    // Empty & nested Nil layouts
    assert_eq!(
        right_inverse(Layout::new(Nil, Nil)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        left_inverse(Layout::new(Nil, Nil)),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        right_inverse(Layout::new(
            int_tuple![(), Int::<4>, ()],
            int_tuple![(), Int::<1>, ()],
        )),
        Layout::new(Int::<4>, Int::<1>)
    );
    assert_eq!(
        left_inverse(Layout::new(
            int_tuple![(), Int::<4>, ()],
            int_tuple![(), Int::<2>, ()],
        )),
        Layout::new(
            int_tuple![Int::<2>, Int::<4>],
            int_tuple![Int::<0>, Int::<1>]
        )
    );

    // Canonical zero BasisCombination<Nil> and static-zero BasisTerm
    assert_eq!(
        right_inverse(Layout::new(Int::<4>, BasisCombination::<Nil>::zero())),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        left_inverse(Layout::new(Int::<4>, BasisCombination::<Nil>::zero())),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        right_inverse(Layout::new(
            Int::<4>,
            BasisTerm::<Int<0>, mode![1]>::new(Int::<0>),
        )),
        Layout::new(Int::<1>, Int::<0>)
    );
    assert_eq!(
        left_inverse(Layout::new(
            Int::<4>,
            BasisTerm::<Int<0>, mode![1]>::new(Int::<0>),
        )),
        Layout::new(Int::<1>, Int::<0>)
    );

    // Near i32::MAX without spurious overflow!
    // For `(2_000_000_000, 2):(2, 1)`, running extent `2 * 2_000_000_000` exceeds `i32::MAX`,
    // yet both `right_inverse` and `left_inverse` succeed because the result fits in `i32`.
    const BIG_RIGHT: PairLayout<Int<2>, Int<2_000_000_000>, Int<2_000_000_000>, Int<1>> =
        right_inverse(Layout::new(
            int_tuple![Int::<2_000_000_000>, Int::<2>],
            int_tuple![Int::<2>, Int::<1>],
        ));
    const BIG_LEFT: PairLayout<Int<2>, Int<2_000_000_000>, Int<2_000_000_000>, Int<1>> =
        left_inverse(Layout::new(
            int_tuple![Int::<2_000_000_000>, Int::<2>],
            int_tuple![Int::<2>, Int::<1>],
        ));
    assert_eq!(BIG_RIGHT, BIG_LEFT);

    // Large strides in left_inverse:
    // `(2, 2):(1B, 2B)` coalesces to `4:1B` -> `(1B, 4):(0, 1)` without overflow;
    // `(2, 2):(500M, 2B)` does not coalesce and checks `gap = 2B / 500M = 4 >= 2` via division!
    const BIG_CHAIN_COALESCED: PairLayout<Int<1_000_000_000>, Int<4>, Int<0>, Int<1>> =
        left_inverse(Layout::new(
            int_tuple![Int::<2>, Int::<2>],
            int_tuple![Int::<1_000_000_000>, Int::<2_000_000_000>],
        ));
    const BIG_CHAIN_GAPPED: TripleLayout<
        Int<500_000_000>,
        Int<4>,
        Int<2>,
        Int<0>,
        Int<1>,
        Int<2>,
    > = left_inverse(Layout::new(
        int_tuple![Int::<2>, Int::<2>],
        int_tuple![Int::<500_000_000>, Int::<2_000_000_000>],
    ));
    assert_eq!(size_of_val(&BIG_CHAIN_COALESCED), 0);
    assert_eq!(size_of_val(&BIG_CHAIN_GAPPED), 0);
}

#[test]
fn test_inverse_dynamic_failures() {
    // Zero and negative dynamic extents fail both right_inverse and left_inverse
    assert!(std::panic::catch_unwind(|| right_inverse(Layout::new(0i32, Int::<1>))).is_err());
    assert!(std::panic::catch_unwind(|| right_inverse(Layout::new(0i32, Int::<0>))).is_err());
    assert!(std::panic::catch_unwind(|| right_inverse(Layout::new(-3i32, Int::<1>))).is_err());
    assert!(
        std::panic::catch_unwind(|| {
            right_inverse(Layout::new(
                int_tuple![0i32, Int::<4>],
                int_tuple![Int::<0>, Int::<1>],
            ))
        })
        .is_err()
    );
    assert!(std::panic::catch_unwind(|| left_inverse(Layout::new(0i32, Int::<1>))).is_err());
    assert!(std::panic::catch_unwind(|| left_inverse(Layout::new(0i32, Int::<0>))).is_err());
    assert!(std::panic::catch_unwind(|| left_inverse(Layout::new(-3i32, Int::<1>))).is_err());
    assert!(
        std::panic::catch_unwind(|| {
            left_inverse(Layout::new(
                int_tuple![0i32, Int::<4>],
                int_tuple![Int::<0>, Int::<1>],
            ))
        })
        .is_err()
    );

    // Prefix product / coalesce i32 overflow panics in both
    assert!(
        std::panic::catch_unwind(|| {
            right_inverse(Layout::new(
                int_tuple![2i32, 2_000_000_000i32, 2i32],
                int_tuple![Int::<1>, Int::<2>, Int::<4>],
            ))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            left_inverse(Layout::new(
                int_tuple![2i32, 2_000_000_000i32, 2i32],
                int_tuple![Int::<1>, Int::<2>, Int::<4>],
            ))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            right_inverse(Layout::new(
                int_tuple![Int::<2>, 2_000_000_000i32],
                int_tuple![Int::<1>, Int::<2>],
            ))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            left_inverse(Layout::new(
                int_tuple![Int::<2>, 2_000_000_000i32],
                int_tuple![Int::<1>, Int::<2>],
            ))
        })
        .is_err()
    );

    // Left-inverse dynamic precondition failures:
    // 1. Negative stride
    assert!(std::panic::catch_unwind(|| left_inverse(Layout::new(Int::<4>, -2i32))).is_err());
    assert!(std::panic::catch_unwind(|| left_inverse(Layout::new(4i32, -2i32))).is_err());
    assert!(
        std::panic::catch_unwind(|| {
            left_inverse(Layout::new(
                Int::<4>,
                BasisTerm::<i32, mode![1]>::new(-2i32),
            ))
        })
        .is_err()
    );

    // 2. Non-injective / overlapping strides
    assert!(
        std::panic::catch_unwind(|| {
            left_inverse(Layout::new(
                int_tuple![63i32, 2i32],
                int_tuple![1i32, 1i32],
            ))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            left_inverse(Layout::new(
                int_tuple![2i32, 3i32],
                int_tuple![2i32, 1i32],
            ))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            left_inverse(Layout::new(
                int_tuple![5i32, Int::<2>],
                int_tuple![Int::<1>, Int::<1>],
            ))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            left_inverse(Layout::new(
                int_tuple![5i32, 3i32],
                int_tuple![Int::<1>, 3i32],
            ))
        })
        .is_err()
    );

    // 3. Coprime / non-divisible ordered chain violation
    assert!(
        std::panic::catch_unwind(|| {
            left_inverse(Layout::new(
                int_tuple![2i32, 2i32],
                int_tuple![2i32, 3i32],
            ))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            left_inverse(Layout::new(
                int_tuple![4i32, 3i32],
                int_tuple![2i32, 5i32],
            ))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            left_inverse(Layout::new(
                int_tuple![Int::<4>, 3i32],
                int_tuple![2i32, 9i32],
            ))
        })
        .is_err()
    );
}

#[test]
#[should_panic]
fn test_right_inverse_panics_on_dynamic_zero_extent() {
    let _ = right_inverse(Layout::new(0i32, Int::<1>));
}

#[test]
#[should_panic]
fn test_left_inverse_panics_on_dynamic_non_injective() {
    let _ = left_inverse(Layout::new(
        int_tuple![4i32, 2i32],
        int_tuple![1i32, 2i32],
    ));
}
