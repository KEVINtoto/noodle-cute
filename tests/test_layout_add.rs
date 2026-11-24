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
use core::fmt::Debug;
use core::mem::size_of_val;

use noodle_cute::algebra::layout_add::{
    LayoutAdd, layout_add, layout_add_by, layout_add_with_profile,
};
use noodle_cute::basis::{
    BasisCoeff, BasisCoeffAdd, BasisCoeffEq, BasisCoeffMul, BasisCoeffNeg, BasisCoeffOrd,
    BasisCoeffSub, BasisTerm, IntoColexTerms, MergeEq, ModeEnd, basis_eq,
};
use noodle_cute::integer::{Int, Integer, eq as int_eq, int};
use noodle_cute::layout::Layout;
use noodle_cute::shape::{IndexToCoord, Shape};
use noodle_cute::stride::{Accumulate, InnerProduct, Stride};
use noodle_cute::tuple::{
    Congruent, Item, LeafProfile, Nil, NilProfile, Til, TupleProfile, WeaklyCongruent,
};
use noodle_cute::{basis_tuple, e, int_tuple, tuple};

// Custom F2 (GF(2)[x] carry-less ring: XOR addition, clmul scaling)
// Mirrors PyCuTe's `F2` stride scalar tests in `test/test_layout_add.py`.

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct F2Val(u32);

const fn clmul_u32(a: u32, b: u32) -> u32 {
    let mut acc = 0u32;
    let mut i = 0u32;
    while i < 32 {
        if ((b >> i) & 1) != 0 {
            acc ^= a << i;
        }
        i += 1;
    }
    acc
}

const impl Item for F2Val {
    const RANK: int = 1;
    const DEPTH: int = 0;
    const LEAVES: int = 1;
    type Profile = LeafProfile;
}

const impl BasisCoeff for F2Val {
    type Unit = F2Val;
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
        F2Val(1)
    }
}

const impl BasisCoeffAdd<F2Val> for F2Val {
    type Output = F2Val;
    fn coeff_add(self, rhs: F2Val) -> F2Val {
        F2Val(self.0 ^ rhs.0)
    }
}

const impl BasisCoeffNeg for F2Val {
    type Output = F2Val;
    fn coeff_neg(self) -> F2Val {
        self
    }
}

const impl BasisCoeffSub<F2Val> for F2Val {
    type Output = F2Val;
    fn coeff_sub(self, rhs: F2Val) -> F2Val {
        F2Val(self.0 ^ rhs.0)
    }
}

const impl<I: [const] Integer> BasisCoeffMul<I> for F2Val {
    type Output = F2Val;
    fn coeff_mul(self, rhs: I) -> F2Val {
        F2Val(clmul_u32(self.0, rhs.value() as u32))
    }
}

const impl BasisCoeffEq<F2Val> for F2Val {
    fn coeff_eq(&self, rhs: &F2Val) -> bool {
        self.0 == rhs.0
    }
    fn coeff_static_eq(&self, _: &F2Val) -> Option<bool> {
        None
    }
}

const impl BasisCoeffOrd<F2Val> for F2Val {
    fn coeff_cmp(&self, rhs: &F2Val) -> Ordering {
        self.0.cmp(&rhs.0)
    }
    fn coeff_static_cmp(&self, _: &F2Val) -> Option<Ordering> {
        None
    }
}

type F2Stride = BasisTerm<F2Val, ModeEnd>;

const fn f2(v: u32) -> F2Stride {
    BasisTerm::new(F2Val(v))
}

// Universal post-condition checkers mirroring `test/test_layout_add.py`

fn assert_congruent<A: Congruent<B>, B: Item>(_: &A, _: &B) {}
fn assert_weak_congruent<A: WeaklyCongruent<B>, B: Item>(_: &A, _: &B) {}

fn assert_layout_add_int<SA, DA, SB, DB, SR, DR, SR2, DR2>(
    a: Layout<SA, DA>,
    b: Layout<SB, DB>,
) -> Layout<SR, DR>
where
    SA: Shape + Debug,
    DA: Stride<SA> + Debug,
    SB: Shape + Debug,
    DB: Stride<SB> + Debug,
    SR: Shape + Debug,
    DR: Stride<SR> + Debug,
    SR2: Shape + Debug,
    DR2: Stride<SR2> + Debug,
    Layout<SA, DA>: LayoutAdd<Layout<SB, DB>, LeafProfile, Output = Layout<SR, DR>>,
    Layout<SB, DB>: LayoutAdd<Layout<SA, DA>, LeafProfile, Output = Layout<SR2, DR2>>,
    Layout<SR, DR>: PartialEq<Layout<SR2, DR2>>,
    int: IndexToCoord<SA> + IndexToCoord<SB> + IndexToCoord<SR>,
    <int as IndexToCoord<SA>>::Output: InnerProduct<DA>,
    <int as IndexToCoord<SB>>::Output: InnerProduct<DB>,
    <int as IndexToCoord<SR>>::Output: InnerProduct<DR>,
    <<int as IndexToCoord<SA>>::Output as InnerProduct<DA>>::Output: Integer
        + Debug
        + Accumulate<<<int as IndexToCoord<SB>>::Output as InnerProduct<DB>>::Output>,
    <<int as IndexToCoord<SB>>::Output as InnerProduct<DB>>::Output: Integer + Debug,
    <<int as IndexToCoord<SR>>::Output as InnerProduct<DR>>::Output: Integer + Debug,
    <<<int as IndexToCoord<SA>>::Output as InnerProduct<DA>>::Output as Accumulate<
        <<int as IndexToCoord<SB>>::Output as InnerProduct<DB>>::Output,
    >>::Output: Integer + Debug,
{
    let r = layout_add(a, b);
    assert_eq!(r.size(), a.size());
    assert_eq!(r.size(), b.size());

    for i in 0..r.size() {
        let lhs = r(i);
        let rhs = a(i).accumulate(b(i));
        assert!(
            int_eq(lhs, rhs),
            "layout_add pointwise mismatch at i={i}: R(i)={lhs:?} != A(i)+B(i)={rhs:?} (A={a:?}, B={b:?}, R={r:?})"
        );
    }

    let r_ba = layout_add(b, a);
    assert_eq!(r, r_ba, "layout_add not symmetric: {r:?} vs {r_ba:?}");
    r
}

fn assert_layout_add_basis<SA, DA, SB, DB, SR, DR, SR2, DR2, ValA, ValB, ValR, SumVal>(
    a: Layout<SA, DA>,
    b: Layout<SB, DB>,
) -> Layout<SR, DR>
where
    SA: Shape + Debug,
    DA: Stride<SA> + Debug,
    SB: Shape + Debug,
    DB: Stride<SB> + Debug,
    SR: Shape + Debug,
    DR: Stride<SR> + Debug,
    SR2: Shape + Debug,
    DR2: Stride<SR2> + Debug,
    Layout<SA, DA>: LayoutAdd<Layout<SB, DB>, LeafProfile, Output = Layout<SR, DR>>,
    Layout<SB, DB>: LayoutAdd<Layout<SA, DA>, LeafProfile, Output = Layout<SR2, DR2>>,
    Layout<SR, DR>: PartialEq<Layout<SR2, DR2>>,
    int: IndexToCoord<SA> + IndexToCoord<SB> + IndexToCoord<SR>,
    <int as IndexToCoord<SA>>::Output: InnerProduct<DA, Output = ValA>,
    <int as IndexToCoord<SB>>::Output: InnerProduct<DB, Output = ValB>,
    <int as IndexToCoord<SR>>::Output: InnerProduct<DR, Output = ValR>,
    ValA: Debug + Accumulate<ValB, Output = SumVal>,
    ValB: Debug,
    ValR: IntoColexTerms + Debug + Copy,
    SumVal: IntoColexTerms + Debug + Copy,
    ValR::Terms: MergeEq<SumVal::Terms>,
{
    let r = layout_add(a, b);
    assert_eq!(r.size(), a.size());
    assert_eq!(r.size(), b.size());

    for i in 0..r.size() {
        let lhs = r(i);
        let rhs = a(i).accumulate(b(i));
        assert!(
            basis_eq(lhs, rhs),
            "basis layout_add pointwise mismatch at i={i}: R(i)={lhs:?} != A(i)+B(i)={rhs:?} (A={a:?}, B={b:?}, R={r:?})"
        );
    }

    let r_ba = layout_add(b, a);
    assert_eq!(r, r_ba, "basis layout_add not symmetric: {r:?} vs {r_ba:?}");
    r
}

#[test]
fn test_singleton_and_zero_stride_broadcast() {
    // Size-1 layouts evaluate to 0 regardless of stride
    let r = assert_layout_add_int(Layout::new(Int::<1>, Int::<0>), Layout::new(Int::<1>, Int::<0>));
    assert_eq!(r.size(), 1);
    assert_eq!(r, Layout::new(Int::<1>, Int::<0>));

    let r = assert_layout_add_int(Layout::new(Int::<1>, Int::<5>), Layout::new(Int::<1>, Int::<7>));
    assert_eq!(r.size(), 1);
    assert_eq!(r, Layout::new(Int::<1>, Int::<0>));

    // Singleton with F2 and Basis strides
    let r_f2 = assert_layout_add_basis(
        Layout::new(Int::<1>, f2(7)),
        Layout::new(Int::<1>, f2(3)),
    );
    assert_eq!(r_f2.size(), 1);

    let r_at = assert_layout_add_basis(
        Layout::new(Int::<1>, e![0]),
        Layout::new(Int::<1>, Int::<2> * e![1]),
    );
    assert_eq!(r_at.size(), 1);

    // Zero-stride broadcast identity
    assert_eq!(
        assert_layout_add_int(Layout::new(Int::<5>, Int::<0>), Layout::new(Int::<5>, Int::<1>)),
        Layout::new(Int::<5>, Int::<1>)
    );
    assert_eq!(
        assert_layout_add_int(Layout::new(Int::<5>, Int::<0>), Layout::new(Int::<5>, Int::<0>)),
        Layout::new(Int::<5>, Int::<0>)
    );
}

#[test]
fn test_int_strides_suite() {
    // Same shape, same stride
    assert_eq!(
        assert_layout_add_int(
            Layout::new(Int::<12>, Int::<1>),
            Layout::new(Int::<12>, Int::<1>),
        ),
        Layout::new(Int::<12>, Int::<2>)
    );
    assert_eq!(
        assert_layout_add_int(
            Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<1>, Int::<4>]),
            Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<1>, Int::<4>]),
        ),
        Layout::new(Int::<12>, Int::<2>)
    );
    assert_eq!(
        assert_layout_add_int(
            Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<3>, Int::<1>]),
            Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<3>, Int::<1>]),
        ),
        Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<6>, Int::<2>])
    );

    // Same shape, diff stride
    assert_eq!(
        assert_layout_add_int(
            Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<1>, Int::<4>]),
            Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<3>, Int::<1>]),
        ),
        Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<4>, Int::<5>])
    );
    assert_eq!(
        assert_layout_add_int(
            Layout::new(int_tuple![Int::<6>], int_tuple![Int::<5>]),
            Layout::new(int_tuple![Int::<6>], int_tuple![Int::<3>]),
        ),
        Layout::new(Int::<6>, Int::<8>)
    );

    // Both row-major same size, different factorizations
    assert_eq!(
        assert_layout_add_int(
            Layout::new(
                int_tuple![Int::<5>, Int::<3>, Int::<4>],
                int_tuple![Int::<1>, Int::<5>, Int::<15>],
            ),
            Layout::new(
                int_tuple![Int::<10>, Int::<6>],
                int_tuple![Int::<1>, Int::<10>],
            ),
        ),
        Layout::new(Int::<60>, Int::<2>)
    );
    assert_eq!(
        assert_layout_add_int(
            Layout::new(int_tuple![Int::<6>], int_tuple![Int::<1>]),
            Layout::new(int_tuple![Int::<3>, Int::<2>], int_tuple![Int::<1>, Int::<3>]),
        ),
        Layout::new(Int::<6>, Int::<2>)
    );

    // One flat, one decomposed
    assert_eq!(
        assert_layout_add_int(
            Layout::new(int_tuple![Int::<6>], int_tuple![Int::<1>]),
            Layout::new(int_tuple![Int::<3>, Int::<2>], int_tuple![Int::<2>, Int::<1>]),
        ),
        Layout::new(int_tuple![Int::<3>, Int::<2>], int_tuple![Int::<3>, Int::<4>])
    );
    assert_eq!(
        assert_layout_add_int(
            Layout::new(Int::<12>, Int::<2>),
            Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<3>, Int::<1>]),
        ),
        Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<5>, Int::<9>])
    );

    // Nested shapes
    assert_eq!(
        assert_layout_add_int(
            Layout::new(
                int_tuple![Int::<2>, (Int::<3>, Int::<4>)],
                int_tuple![Int::<12>, (Int::<4>, Int::<1>)],
            ),
            Layout::new(
                int_tuple![Int::<6>, Int::<4>],
                int_tuple![Int::<4>, Int::<1>],
            ),
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            int_tuple![Int::<16>, Int::<12>, Int::<2>],
        )
    );

    // Zero stride in one mode
    assert_eq!(
        assert_layout_add_int(
            Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<1>, Int::<0>]),
            Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<3>, Int::<1>]),
        ),
        Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<4>, Int::<1>])
    );

    // Cancellation & coalescing after addition: (4,3):(3,1) + (4,3):(-2,3) -> (4,3):(1,4) -> 12:1
    assert_eq!(
        assert_layout_add_int(
            Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<3>, Int::<1>]),
            Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<-2>, Int::<3>]),
        ),
        Layout::new(Int::<12>, Int::<1>)
    );
    // Exact cancellation to zero coalesces multi-mode to 12:0
    assert_eq!(
        assert_layout_add_int(
            Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<3>, Int::<1>]),
            Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<-3>, Int::<-1>]),
        ),
        Layout::new(Int::<12>, Int::<0>)
    );
    // Nested Nil modes flatten cleanly in layout_add
    assert_eq!(
        assert_layout_add_int(
            Layout::new(int_tuple![(), Int::<4>, ()], int_tuple![(), Int::<1>, ()]),
            Layout::new(Int::<4>, Int::<2>),
        ),
        Layout::new(Int::<4>, Int::<3>)
    );
}

#[test]
fn test_basis_strides_suite() {
    // Disjoint basis paths: E(0) + E(1)
    let r = assert_layout_add_basis(
        Layout::new(Int::<8>, e![0]),
        Layout::new(Int::<8>, e![1]),
    );
    assert_eq!(r, Layout::new(Int::<8>, e![0] + e![1]));
    for i in 0..8i32 {
        assert!(basis_eq(r(i), basis_tuple![i, i]));
    }

    // Same basis path: 2*E(0) + 3*E(0) = 5*E(0)
    let r = assert_layout_add_basis(
        Layout::new(Int::<8>, Int::<2> * e![0]),
        Layout::new(Int::<8>, Int::<3> * e![0]),
    );
    assert_eq!(r, Layout::new(Int::<8>, Int::<5> * e![0]));

    // Multi-mode disjoint basis
    let r = assert_layout_add_basis(
        Layout::new(int_tuple![Int::<4>, Int::<3>], tuple![e![0], e![1]]),
        Layout::new(int_tuple![Int::<4>, Int::<3>], tuple![e![1], e![0]]),
    );
    for c0 in 0..4i32 {
        for c1 in 0..3i32 {
            assert!(basis_eq(r(c0, c1), basis_tuple![c0 + c1, c0 + c1]));
        }
    }

    // Universal int-zero identity on Basis layout
    let a = Layout::new(int_tuple![Int::<4>, Int::<3>], tuple![e![0], e![1]]);
    let r = assert_layout_add_basis(a, Layout::new(Int::<12>, Int::<0>));
    assert_eq!(r, a);

    // Different shapes sharing a refinement: 12:E(0) + (4,3):(E(1), E(2))
    let a = Layout::new(Int::<12>, e![0]);
    let b = Layout::new(int_tuple![Int::<4>, Int::<3>], tuple![e![1], e![2]]);
    let r = assert_layout_add_basis(a, b);
    for i in 0..12i32 {
        assert!(basis_eq(r(i), basis_tuple![i, i % 4, i / 4]));
    }

    // Associativity with multi-term BasisCombination intermediates!
    let c = Layout::new(int_tuple![Int::<4>, Int::<3>], tuple![e![0], e![1]]);
    let abc1 = layout_add(layout_add(a, b), c);
    let abc2 = layout_add(a, layout_add(b, c));
    for i in 0..12i32 {
        assert!(basis_eq(abc1(i), abc2(i)));
        assert!(basis_eq(abc1(i), (a(i) + b(i)) + c(i)));
    }

    // Multi-term BasisCombination cancellation to zero coalesces to 12:0
    let pos_b = Layout::new(int_tuple![Int::<4>, Int::<3>], tuple![e![0], e![1]]);
    let neg_b = Layout::new(int_tuple![Int::<4>, Int::<3>], tuple![-e![0], -e![1]]);
    assert_eq!(
        assert_layout_add_basis(pos_b, neg_b),
        Layout::new(Int::<12>, Int::<0>)
    );

    // Multi-term BasisCombination strides coalesce cleanly after addition!
    // (4, 3):(e![0], 4*e![0] + e![1]) + (4, 3):(e![1], 3*e![1]) -> (4, 3):(e![0]+e![1], 4*e![0]+4*e![1]) -> 12:(e![0]+e![1])
    let m1 = Layout::new(
        int_tuple![Int::<4>, Int::<3>],
        tuple![e![0], Int::<4> * e![0] + e![1]],
    );
    let m2 = Layout::new(
        int_tuple![Int::<4>, Int::<3>],
        tuple![e![1], Int::<3> * e![1]],
    );
    let m_sum = assert_layout_add_basis(m1, m2);
    assert_eq!(m_sum, Layout::new(Int::<12>, e![0] + e![1]));
    // Also verify standalone coalesce on BasisCombination strides directly
    assert_eq!(
        Layout::new(
            int_tuple![Int::<4>, Int::<3>],
            tuple![e![0] + e![1], Int::<4> * e![0] + Int::<4> * e![1]],
        )
        .coalesce(),
        Layout::new(Int::<12>, e![0] + e![1])
    );
}

#[test]
fn test_f2_strides_and_algebraic_properties() {
    // Disjoint bits: F2(1) + F2(8) = F2(9)
    assert_eq!(
        assert_layout_add_basis(Layout::new(Int::<8>, f2(1)), Layout::new(Int::<8>, f2(8))),
        Layout::new(Int::<8>, f2(9))
    );

    // Overlapping bits same shape: F2(1) + F2(2) = F2(3)
    assert_eq!(
        assert_layout_add_basis(Layout::new(Int::<8>, f2(1)), Layout::new(Int::<8>, f2(2))),
        Layout::new(Int::<8>, f2(3))
    );

    // F2(0) identity and universal Int<0> identity
    let a = Layout::new(Int::<8>, f2(7));
    assert_eq!(assert_layout_add_basis(a, Layout::new(Int::<8>, f2(0))), a);
    assert_eq!(assert_layout_add_basis(a, Layout::new(Int::<8>, Int::<0>)), a);

    // Multi-mode F2 strides
    assert_eq!(
        assert_layout_add_basis(
            Layout::new(int_tuple![Int::<4>, Int::<8>], tuple![f2(1), f2(8)]),
            Layout::new(int_tuple![Int::<4>, Int::<8>], tuple![f2(2), f2(16)]),
        ),
        Layout::new(int_tuple![Int::<4>, Int::<8>], tuple![f2(3), f2(24)])
    );

    // Different shapes refining via power-of-two modes
    let p1 = Layout::new(Int::<16>, f2(1));
    let p2 = Layout::new(int_tuple![Int::<4>, Int::<4>], tuple![f2(2), f2(8)]);
    let r_pow2 = assert_layout_add_basis(p1, p2);
    for i in 0..16i32 {
        assert_eq!(r_pow2(i), f2(clmul_u32(3, i as u32)));
    }

    // Self-inverse: A + A == 0 pointwise for F2
    for &s in &[0u32, 1, 8, 9, 64] {
        let l = Layout::new(Int::<16>, f2(s));
        let sum = assert_layout_add_basis(l, l);
        for i in 0..16i32 {
            assert!(basis_eq(sum(i), Int::<0>));
        }
    }
    let multi = Layout::new(int_tuple![Int::<4>, Int::<4>], tuple![f2(1), f2(8)]);
    let multi_sum = assert_layout_add_basis(multi, multi);
    for i in 0..16i32 {
        assert!(basis_eq(multi_sum(i), Int::<0>));
    }

    // F2 associativity
    let f_a = Layout::new(Int::<16>, f2(1));
    let f_b = Layout::new(int_tuple![Int::<4>, Int::<4>], tuple![f2(2), f2(8)]);
    let f_c = Layout::new(int_tuple![Int::<4>, Int::<4>], tuple![f2(1), f2(4)]);
    let abc1 = layout_add(layout_add(f_a, f_b), f_c);
    let abc2 = layout_add(f_a, layout_add(f_b, f_c));
    for i in 0..16i32 {
        assert!(basis_eq(abc1(i), abc2(i)));
    }

    // F2 by-mode swizzle (mirroring `test_f2_by_mode_is_the_swizzle_it_should_be`)
    let tile = Layout::new(int_tuple![Int::<8>, Int::<64>], tuple![f2(64), f2(1)]);
    let corr = Layout::new(int_tuple![Int::<8>, Int::<64>], tuple![f2(8), f2(0)]);
    let swiz = layout_add_with_profile(tile, corr, int_tuple![1, 1]);
    assert_eq!(
        swiz,
        Layout::new(int_tuple![Int::<8>, Int::<64>], tuple![f2(72), f2(1)])
    );
    for m in 0..8i32 {
        for k in 0..64i32 {
            assert_eq!(swiz(m, k), f2(((64 * m) ^ (8 * m) ^ k) as u32));
        }
    }
}

#[test]
fn test_layout_add_by_mode_profile() {
    let a = Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<1>, Int::<4>]);
    let b = Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<3>, Int::<1>]);

    // By-mode keeps axes
    let r = layout_add_with_profile(a, b, int_tuple![1, 1]);
    assert_eq!(
        r,
        Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<4>, Int::<5>])
    );
    assert_congruent(&a.shape(), &r.shape());

    // Whole-layout vs by-mode on A + A
    let by_mode = layout_add_with_profile(a, a, int_tuple![1, 1]);
    let whole = layout_add(a, a);
    assert_eq!(
        by_mode,
        Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<2>, Int::<8>])
    );
    assert_eq!(whole, Layout::new(Int::<12>, Int::<2>));
    assert_congruent(&a.shape(), &by_mode.shape());
    for i in 0..4i32 {
        for j in 0..3i32 {
            assert_eq!(whole(i + 4 * j), by_mode(i, j));
        }
    }

    // By-mode pointwise agreement & symmetry
    assert_eq!(
        layout_add_with_profile(a, b, int_tuple![1, 1]),
        layout_add_with_profile(b, a, int_tuple![1, 1])
    );
    let p_a = Layout::new(int_tuple![Int::<5>, Int::<3>], int_tuple![Int::<3>, Int::<1>]);
    let p_b = Layout::new(int_tuple![Int::<5>, Int::<3>], int_tuple![Int::<1>, Int::<5>]);
    let p_r = layout_add_with_profile(p_a, p_b, int_tuple![1, 1]);
    for i in 0..5i32 {
        for j in 0..3i32 {
            assert_eq!(p_r(i, j), p_a(i, j) + p_b(i, j));
        }
    }

    // Profile weakly coarsens both arguments & sum refines profile
    let n = Layout::new(
        int_tuple![(Int::<2>, Int::<2>), Int::<3>],
        int_tuple![(Int::<1>, Int::<2>), Int::<4>],
    );
    let r_an = layout_add_with_profile(a, n, int_tuple![1, 1]);
    assert_weak_congruent(&int_tuple![1, 1], &r_an.shape());
    assert_eq!(r_an.size(), a.size());

    let r_nn = layout_add_with_profile(n, n, int_tuple![(1, 1), 1]);
    assert_weak_congruent(&int_tuple![(1, 1), 1], &r_nn.shape());
    assert_eq!(r_nn.size(), n.size());
    assert_eq!(
        r_nn,
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), Int::<3>],
            int_tuple![(Int::<2>, Int::<4>), Int::<8>],
        )
    );

    // Explicit type-level `layout_add_by::<P, _, _>` & inherent methods
    type PairProfile = TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>;
    assert_eq!(layout_add_by::<PairProfile, _, _>(a, a), by_mode);
    assert_eq!(a.layout_add_by::<PairProfile, _>(a), by_mode);
    assert_eq!(a.layout_add_with_profile(a, int_tuple![1, 1]), by_mode);
    assert_eq!(a.layout_add(b), layout_add(a, b));

    // Nil is the additive identity under any profile (whole and by-mode)
    assert_eq!(layout_add(a, Nil), a);
    assert_eq!(layout_add(Nil, a), a);
    assert_eq!(layout_add(Nil, Nil), Nil);
    assert_eq!(layout_add_with_profile(a, Nil, int_tuple![1, 1]), a);
    assert_eq!(layout_add_with_profile(Nil, a, int_tuple![1, 1]), a);
    assert_eq!(layout_add_with_profile(Nil, Nil, int_tuple![1, 1]), Nil);

    // Empty layout with NilProfile and LeafProfile
    assert_eq!(
        layout_add_by::<NilProfile, _, _>(Layout::new(Nil, Nil), Layout::new(Nil, Nil)),
        Layout::new(Nil, Nil)
    );
    assert_eq!(
        layout_add(Layout::new(Nil, Nil), Layout::new(Nil, Nil)),
        Layout::new(Int::<1>, Int::<0>)
    );
}

#[test]
fn test_const_eval_associativity_and_dynamic() {
    type PairLayout<S0, S1, D0, D1> = Layout<Til<S0, Til<S1, Nil>>, Til<D0, Til<D1, Nil>>>;

    // Const evaluation & zero-sized static output
    const SUM_CONST: PairLayout<Int<4>, Int<3>, Int<4>, Int<5>> = layout_add(
        Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<1>, Int::<4>]),
        Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<3>, Int::<1>]),
    );
    assert_eq!(size_of_val(&SUM_CONST), 0);

    // Associativity across int layouts of size 12
    let a = Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<1>, Int::<4>]);
    let b = Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![Int::<3>, Int::<1>]);
    let c = Layout::new(
        int_tuple![(Int::<2>, Int::<2>), Int::<3>],
        int_tuple![(Int::<2>, Int::<1>), Int::<4>],
    );
    let abc1 = layout_add(layout_add(a, b), c);
    let abc2 = layout_add(a, layout_add(b, c));
    for i in 0..12i32 {
        assert_eq!(abc1(i), abc2(i));
        assert_eq!(abc1(i), a(i) + b(i) + c(i));
    }

    // Static shapes with dynamic strides -> exact minimal rank!
    let dyn_d1 = Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![1i32, 4i32]);
    let dyn_d2 = Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![3i32, 1i32]);
    assert_eq!(
        assert_layout_add_int(dyn_d1, dyn_d2),
        Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![4i32, 5i32])
    );

    // Dynamic rank-1 shape -> exact rank-1 output!
    let n = 12i32;
    assert_eq!(
        assert_layout_add_int(Layout::new(n, 2i32), Layout::new(n, 5i32)),
        Layout::new(12i32, 7i32)
    );
    // Dynamic rank-1 shape with static zero identity + Basis stride!
    assert_eq!(
        assert_layout_add_basis(Layout::new(n, Int::<0>), Layout::new(n, e![0])),
        Layout::new(12i32, BasisTerm::<int, noodle_cute::mode![0]>::new(1i32))
    );

    // Dynamic multi-mode shapes with common refinement
    let da = Layout::new(int_tuple![4i32, 3i32], int_tuple![1i32, 4i32]);
    let db = Layout::new(int_tuple![4i32, 3i32], int_tuple![3i32, 1i32]);
    let dr = assert_layout_add_int(da, db);
    assert_eq!(dr.size(), 12);
    // By-mode with dynamic shapes preserves exact rank-2 structure!
    let dr_by = layout_add_with_profile(da, db, int_tuple![1, 1]);
    assert_eq!(
        dr_by,
        Layout::new(int_tuple![4i32, 3i32], int_tuple![4i32, 5i32])
    );
}

#[test]
fn test_layout_add_dynamic_failures() {
    // Dynamic size mismatch
    assert!(
        std::panic::catch_unwind(|| layout_add(Layout::new(6i32, 1i32), Layout::new(8i32, 1i32)))
            .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            layout_add(Layout::new(Int::<6>, 1i32), Layout::new(8i32, 1i32))
        })
        .is_err()
    );

    // Dynamic no common refinement of size |A|
    assert!(
        std::panic::catch_unwind(|| {
            layout_add(
                Layout::new(int_tuple![5i32, 3i32], int_tuple![3i32, 1i32]),
                Layout::new(int_tuple![3i32, 5i32], int_tuple![5i32, 1i32]),
            )
        })
        .is_err()
    );

    // Dynamic zero and negative extents
    assert!(
        std::panic::catch_unwind(|| layout_add(Layout::new(-4i32, 1i32), Layout::new(-4i32, 2i32)))
            .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| layout_add(Layout::new(0i32, 1i32), Layout::new(0i32, 2i32)))
            .is_err()
    );

    // Dynamic total size overflow
    assert!(
        std::panic::catch_unwind(|| {
            layout_add(
                Layout::new(int_tuple![65536i32, 65536i32], int_tuple![1i32, 65536i32]),
                Layout::new(int_tuple![65536i32, 65536i32], int_tuple![1i32, 65536i32]),
            )
        })
        .is_err()
    );

    // Dynamic stride addition overflow
    assert!(
        std::panic::catch_unwind(|| {
            layout_add(
                Layout::new(Int::<4>, int::MAX),
                Layout::new(Int::<4>, 1i32),
            )
        })
        .is_err()
    );
    // Dynamic stride evaluation overflow inside A(d)
    assert!(
        std::panic::catch_unwind(|| {
            layout_add(
                Layout::new(Int::<12>, 1_000_000_000i32),
                Layout::new(int_tuple![Int::<4>, Int::<3>], int_tuple![1i32, 4i32]),
            )
        })
        .is_err()
    );
    // Dynamic BasisTerm coefficient addition overflow
    assert!(
        std::panic::catch_unwind(|| {
            layout_add(
                Layout::new(Int::<4>, int::MAX * e![0]),
                Layout::new(Int::<4>, 1i32 * e![0]),
            )
        })
        .is_err()
    );

    // By-mode panic propagates from failing mode
    assert!(
        std::panic::catch_unwind(|| {
            layout_add_with_profile(
                Layout::new(int_tuple![4i32, 6i32], int_tuple![1i32, 4i32]),
                Layout::new(int_tuple![4i32, 8i32], int_tuple![1i32, 4i32]),
                int_tuple![1, 1],
            )
        })
        .is_err()
    );
}

#[test]
#[should_panic]
fn test_layout_add_panics_on_dynamic_size_mismatch() {
    let _ = layout_add(Layout::new(6i32, 1i32), Layout::new(8i32, 1i32));
}
