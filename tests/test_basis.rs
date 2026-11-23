#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl,
    const_ops,
    const_cmp
)]
#![allow(incomplete_features)]

use core::any::TypeId;
use core::cmp::Ordering;
use std::panic::catch_unwind;

use noodle_cute::basis::{
    BASIS_NORMALIZED_INDEX, Basis, BasisAdd, BasisCoeff, BasisCoeffAdd, BasisCoeffEq,
    BasisCoeffMul, BasisCoeffNeg, BasisCoeffOrd, BasisCoeffSub, BasisCombination, BasisMul,
    BasisPath, BasisTerm, ComparePath, MergeColex, Mode, ModeEnd, PathOrdering, ScaledBasis,
    as_tuple, basis_eq, basis_repr, colex_cmp, colex_ge, colex_gt, colex_le, colex_lt, e,
    from_tuple, is_basis, make_basis_like, proj, static_basis_eq, static_colex_cmp, unit,
};
use noodle_cute::integer::{Int, int};
use noodle_cute::tuple::{
    Congruent, FoldLeafFunctor, Item, LeafProfile, Nil, Til, WeaklyCongruent, fold_leaf, select,
};
use noodle_cute::{basis_tuple, e, int_tuple, mode, tuple, walk_mode};

const I32_MIN: int = int::MIN;
const I32_MAX: int = int::MAX;

#[derive(Clone, Copy)]
struct BasisDot;

const impl<Acc, C, S> FoldLeafFunctor<Acc, (C, S)> for BasisDot
where
    C: [const] BasisMul<S>,
    S: Copy,
    Acc: [const] BasisAdd<<C as BasisMul<S>>::Output>,
{
    type Output = <Acc as BasisAdd<<C as BasisMul<S>>::Output>>::Output;
    fn apply(&self, acc: Acc, (c, s): (C, S)) -> Self::Output {
        acc.basis_add(c.basis_mul(s))
    }
}

#[derive(Clone, Copy)]
struct BasisRightDot;

const impl<Acc, C, S> FoldLeafFunctor<Acc, (C, S)> for BasisRightDot
where
    C: Copy,
    S: [const] BasisMul<C>,
    Acc: [const] BasisAdd<<S as BasisMul<C>>::Output>,
{
    type Output = <Acc as BasisAdd<<S as BasisMul<C>>::Output>>::Output;
    fn apply(&self, acc: Acc, (c, s): (C, S)) -> Self::Output {
        acc.basis_add(s.basis_mul(c))
    }
}

#[test]
fn paths_unit_constructors_display_and_make_basis_like() {
    const {
        assert!(ModeEnd::DEPTH == 0);
        assert!(<mode![0] as BasisPath>::DEPTH == 1);
        assert!(<mode![1, 2, 3] as BasisPath>::DEPTH == 3);
        assert!(matches!(
            <ModeEnd as ComparePath<ModeEnd>>::ORDER,
            PathOrdering::Equal
        ));
        assert!(matches!(
            <mode![0] as ComparePath<mode![0, 1]>>::ORDER,
            PathOrdering::PrefixConflict
        ));
        assert!(matches!(
            <mode![0, 1] as ComparePath<mode![0]>>::ORDER,
            PathOrdering::PrefixConflict
        ));
        assert!(matches!(
            <mode![1, 0] as ComparePath<mode![1, 1]>>::ORDER,
            PathOrdering::Less
        ));
        assert!(matches!(
            <mode![2] as ComparePath<mode![1, 5]>>::ORDER,
            PathOrdering::Greater
        ));

        assert!(e![] == Int::<1>);
        assert!(e![0] == Basis::<mode![0]>::UNIT);
        assert!(e![1, 0] == e::<mode![1, 0]>());
        assert!(e![1 + 1, 0,] == e![2, 0]);
        assert!(e![0].at::<1>() == e![1, 0]);
        assert!(e![0] == 1i32 * e![0]);
        assert!(1i32 * e![0] == e![0]);
        assert!(e![0] != 2i32 * e![0]);

        assert!(<Basis<mode![0]> as Item>::RANK == 1);
        assert!(<Basis<mode![0]> as Item>::DEPTH == 0);
        assert!(<Basis<mode![0]> as Item>::LEAVES == 1);
        assert!(<Basis<mode![0]> as Item>::IS_STATIC);
        assert!(!<ScaledBasis<int, mode![0]> as Item>::IS_STATIC);
        assert!(<BasisCombination<Nil> as Item>::IS_STATIC);
        assert!((e![0] + e![1]).is_static());
        assert!(!(e![0] + 2i32 * e![1]).is_static());
    }

    assert_eq!(
        TypeId::of::<Basis<mode![1]>>(),
        TypeId::of::<ScaledBasis<Int<1>, Mode<1, ModeEnd>>>()
    );
    assert_eq!(core::mem::size_of::<Basis<mode![1000]>>(), 0);
    assert_eq!(
        core::mem::size_of::<ScaledBasis<int, mode![1000]>>(),
        core::mem::size_of::<int>()
    );

    // Display formatting
    assert_eq!(format!("{}", BasisTerm::<int, ModeEnd>::new(7)), "7");
    assert_eq!(format!("{}", 3i32 * e![0]), "3@0");
    assert_eq!(format!("{}", Int::<5> * e![1, 2]), "5@(1, 2)");
    assert_eq!(format!("{}", 7i32 * e![0, 1, 2]), "7@(0, 1, 2)");
    assert_eq!(format!("{}", BasisCombination::<Nil>::zero()), "0");
    assert_eq!(
        format!("{}", basis_tuple![1i32, (2i32, 3i32), 4i32]),
        "1@0 + 2@(1, 0) + 3@(1, 1) + 4@2"
    );

    // make_basis_like on scalar, Nil, flat, nested, and tuple-with-Nil
    const LIKE_SCALAR: Int<1> = make_basis_like(10i32);
    const LIKE_NIL: Nil = make_basis_like(Nil);
    assert_eq!(LIKE_SCALAR, e![]);
    assert_eq!(LIKE_NIL, Nil);

    let shape = tuple![Int::<10>, (20i32, Int::<30>)];
    let basis_like = make_basis_like(shape);
    assert_eq!(basis_like, tuple![e![0], (e![1, 0], e![1, 1])]);

    fn require_congruent<L: Congruent<R>, R: Item>(_: &L, _: &R) {}
    require_congruent(&basis_like, &shape);

    let with_nil = tuple![10i32, Nil, (20i32)];
    let basis_with_nil = make_basis_like(with_nil);
    assert_eq!(basis_with_nil, tuple![e![0], Nil, (e![2, 0])]);
    require_congruent(&basis_with_nil, &with_nil);
}

#[test]
fn canonical_arithmetic_sub_neg_mul_and_overflow() {
    let a = basis_tuple![1i32, (2i32, 3i32), 4i32];
    let b = basis_tuple![5i32, (6i32)];
    let expected = basis_tuple![6i32, (8i32, 3i32), 4i32];
    assert_eq!(a + b, expected);
    assert_eq!(b + a, expected);
    assert_eq!(a + basis_tuple![], a);
    assert_eq!(basis_tuple![] + a, a);
    assert_eq!(a + Int::<0>, a);
    assert_eq!(Int::<0> + a, a);

    // Static zero normalization in construction & cancellation
    let zero_norm = basis_tuple![Int::<0>, (Int::<0>), Nil];
    assert_eq!(zero_norm, BasisCombination::<Nil>::zero());
    assert_eq!(zero_norm + a, a);
    assert_eq!(a + zero_norm, a);
    assert_eq!(BasisTerm::<Int<0>, mode![1]>::new(Int) + e![1, 0], e![1, 0]);

    // Cancellation to 1 term (BasisTerm) or 0 terms (BasisCombination<Nil>)
    let single: ScaledBasis<Int<5>, mode![0]> =
        (Int::<2> * e![1] + Int::<5> * e![0]) + Int::<-2> * e![1];
    assert_eq!(single, Int::<5> * e![0]);
    assert_eq!(single + e![1, 0], basis_tuple![Int::<5>, (Int::<1>)]);
    assert_eq!(e![0] - e![0], BasisCombination::<Nil>::zero());
    assert_eq!(e![0] + (-e![0]), BasisCombination::<Nil>::zero());

    // Subtraction & unary negation
    assert_eq!(-e![0], BasisTerm::<Int<-1>, mode![0]>::new(Int));
    assert_eq!(Int::<0> - e![0], -e![0]);
    assert_eq!(e![0] - Int::<0>, e![0]);
    assert_eq!(Int::<0> - (e![0] + e![1]), -(e![0] + e![1]));
    assert_eq!(a - b, basis_tuple![-4i32, (-4i32, 3i32), 4i32]);
    assert_eq!(-(e![0] + 2i32 * e![1]), Int::<-1> * e![0] + (-2i32) * e![1]);
    assert_eq!(-(e![0] + 2i32 * e![1]), -1i32 * e![0] + (-2i32) * e![1]);
    assert!(basis_eq(
        -(e![0] + 2i32 * e![1]),
        -1i32 * e![0] + (-2i32) * e![1]
    ));

    // Const signed boundary evaluation without wider intermediates
    const {
        let cancel_max: BasisCombination<Nil> = Int::<I32_MAX> * e![0] - Int::<I32_MAX> * e![0];
        let cancel_min: BasisCombination<Nil> = Int::<I32_MIN> * e![0] - Int::<I32_MIN> * e![0];
        let max_plus_min: ScaledBasis<Int<-1>, mode![0]> =
            Int::<I32_MAX> * e![0] + Int::<I32_MIN> * e![0];
        let neg_max: ScaledBasis<Int<{ -I32_MAX }>, mode![0]> = -(Int::<I32_MAX> * e![0]);
        let absorb_l: BasisCombination<Nil> = Int::<0> * (I32_MIN * e![0]);
        let absorb_r: BasisCombination<Nil> = (I32_MIN * e![0]) * Int::<0>;

        assert!(cancel_max == BasisCombination::<Nil>::zero());
        assert!(cancel_min == BasisCombination::<Nil>::zero());
        assert!(max_plus_min == Int::<-1> * e![0]);
        assert!(neg_max == Int::<{ -I32_MAX }> * e![0]);
        assert!(absorb_l == BasisCombination::<Nil>::zero());
        assert!(absorb_r == BasisCombination::<Nil>::zero());
    }

    assert_eq!(
        BasisCombination::<Nil>::default(),
        BasisCombination::<Nil>::zero()
    );
    assert_eq!((e![0] + e![1]).terms(), Til(e![0], Til(e![1], Nil)));

    // Left and right scalar multiplication (including Int<0> folding)
    let signed = basis_tuple![Int::<2>, Int::<-3>];
    assert_eq!(signed * Int::<0>, BasisCombination::<Nil>::zero());
    assert_eq!(Int::<0> * signed, BasisCombination::<Nil>::zero());
    assert_eq!(Int::<0> * e![2], BasisCombination::<Nil>::zero());
    assert_eq!(e![2] * Int::<0>, BasisCombination::<Nil>::zero());
    assert_eq!(Int::<-2> * signed, basis_tuple![Int::<-4>, Int::<6>]);
    assert_eq!(signed * Int::<-2>, basis_tuple![Int::<-4>, Int::<6>]);
    assert_eq!(2i32 * (3i32 * e![0]), 6i32 * e![0]);
    assert_eq!((3i32 * e![0]) * 2i32, 6i32 * e![0]);

    // from_tuple with sparse zeros, nested BasisTerm, and BasisCombination
    assert_eq!(
        from_tuple(tuple![Int::<0>, 7i32]),
        BasisTerm::<int, mode![1]>::new(7)
    );
    assert_eq!(from_tuple(tuple![e![1], e![0]]), e![0, 1] + e![1, 0]);
    assert_eq!(
        from_tuple(tuple![basis_tuple![1i32, 2i32], 3i32]),
        basis_tuple![(1i32, 2i32), 3i32]
    );

    // Dynamic overflow panics consistently with integer::ops
    assert!(catch_unwind(|| I32_MAX * e![0] + 1i32 * e![0]).is_err());
    assert!(catch_unwind(|| I32_MIN * e![0] - 1i32 * e![0]).is_err());
    assert!(catch_unwind(|| -(I32_MIN * e![0])).is_err());
    assert!(catch_unwind(|| 2i32 * (I32_MAX * e![0])).is_err());
    assert!(catch_unwind(|| (I32_MAX * e![0]) * 2i32).is_err());
}

#[test]
fn basis_eq_and_colex_ordering_support_zero_padding_and_static_eval() {
    // Zero-padded equality including PrefixConflict with dynamic zeros
    assert!(basis_eq(0i32, BasisCombination::<Nil>::zero()));
    assert!(basis_eq(0i32, basis_tuple![0i32, (0i32, 0i32)]));
    assert!(basis_eq(
        basis_tuple![0i32, 5i32],
        basis_tuple![(0i32, 0i32), 5i32]
    ));
    assert!(!basis_eq(
        basis_tuple![1i32, 5i32],
        basis_tuple![(0i32, 0i32), 5i32]
    ));
    assert!(basis_eq(e![0] + 0i32 * e![1], e![0]));

    const {
        assert!(matches!(
            static_basis_eq(Int::<0>, BasisCombination::<Nil>::zero()),
            Some(true)
        ));
        assert!(matches!(static_basis_eq(e![0], e![0] + e![1]), Some(false)));
        assert!(matches!(static_basis_eq(0i32, e![0]), Some(false)));
        assert!(static_basis_eq(0i32, BasisCombination::<Nil>::zero()).is_none());
    }

    // Colexicographical comparison (higher-index modes are most significant)
    let low_big = basis_tuple![100i32, 2i32];
    let high_big = basis_tuple![1i32, 3i32];
    assert_eq!(colex_cmp(low_big, high_big), Ordering::Less);
    assert!(colex_lt(low_big, high_big));
    assert!(colex_le(low_big, high_big));
    assert!(colex_le(low_big, low_big));
    assert!(colex_gt(high_big, low_big));
    assert!(colex_ge(high_big, low_big));
    assert!(colex_ge(high_big, high_big));
    assert_eq!(colex_cmp(e![0], e![0] + 0i32 * e![1]), Ordering::Equal);
    assert_eq!(colex_cmp(Int::<0>, e![0]), Ordering::Less);
    assert_eq!(colex_cmp(Int::<0>, -e![0]), Ordering::Greater);
    assert_eq!(colex_cmp(5i32 * e![0], -1i32 * e![1]), Ordering::Greater);
    assert_eq!(colex_cmp(e![0, 1], e![1]), Ordering::Less);

    const {
        assert!(matches!(
            static_colex_cmp(
                basis_tuple![Int::<100>, Int::<2>],
                basis_tuple![Int::<1>, Int::<3>]
            ),
            Some(Ordering::Less)
        ));
        // Higher mode decides statically even when lower mode is dynamic:
        assert!(matches!(
            static_colex_cmp(basis_tuple![100i32, Int::<2>], basis_tuple![1i32, Int::<3>]),
            Some(Ordering::Less)
        ));
        // Equal higher mode, dynamic lower mode -> statically undecided:
        assert!(
            static_colex_cmp(basis_tuple![100i32, Int::<2>], basis_tuple![1i32, Int::<2>])
                .is_none()
        );
    }
}

#[test]
fn get_walk_mode_select_basis_repr_is_basis_proj_and_unit() {
    let comb = basis_tuple![1i32, (2i32, 3i32), 4i32];
    assert_eq!(comb.get::<0>(), 1i32);
    assert_eq!(comb.get::<1>(), basis_tuple![2i32, 3i32]);
    assert_eq!(comb.get::<-1>(), 4i32);
    assert_eq!(comb.get::<-3>(), 1i32);
    assert_eq!(comb.get::<10>(), Int::<0>);
    assert_eq!(walk_mode!(comb, 1, 0), 2i32);
    assert_eq!(walk_mode!(comb, 1, 1), 3i32);
    assert_eq!(
        select::<mode![0, -1, 5], _>(&comb),
        tuple![1i32, 4i32, Int::<0>]
    );

    // Single nested mode unwraps to BasisTerm and still supports .get() / walk_mode!
    let sparse_nested = basis_tuple![1i32, (Int::<0>, 3i32)];
    let sub: ScaledBasis<int, mode![1]> = sparse_nested.get::<1>();
    assert_eq!(sub, BasisTerm::<int, mode![1]>::new(3));
    assert_eq!(sub.get::<0>(), Int::<0>);
    assert_eq!(sub.get::<1>(), 3i32);
    assert_eq!(sub.get::<-1>(), 3i32);
    assert_eq!(sub.get::<-2>(), Int::<0>);
    assert_eq!(walk_mode!(sparse_nested, 1, 1), 3i32);
    assert_eq!(BasisCombination::<Nil>::zero().get::<0>(), Int::<0>);

    // Extreme leading mode int::MAX does not overflow negative index normalization
    const {
        assert!(e![I32_MAX].get::<-1>() == Int::<1>);
        assert!(e![I32_MAX].get::<I32_MIN>() == Int::<0>);
        assert!((e![0] + e![I32_MAX]).get::<-1>() == Int::<1>);
        assert!((e![0] + e![I32_MAX]).get::<I32_MIN>() == Int::<1>);
    }

    // Const get & walk_mode!
    type StaticPairTerms =
        Til<ScaledBasis<Int<1>, mode![0]>, Til<ScaledBasis<Int<5>, mode![1, 1]>, Nil>>;
    const C_COMB: BasisCombination<StaticPairTerms> = basis_tuple![Int::<1>, (Int::<0>, Int::<5>)];
    const C_SUB: ScaledBasis<Int<5>, mode![1]> = C_COMB.get::<1>();
    const C_VAL: Int<5> = C_SUB.get::<1>();
    const C_WALK: Int<5> = walk_mode!(C_COMB, 1, 1);
    assert_eq!((C_VAL, C_WALK), (Int::<5>, Int::<5>));

    // basis_repr & is_basis
    assert_eq!(
        basis_repr(5i32 * e![1, 2]),
        Til(BasisTerm::<int, mode![1, 2]>::new(5), Nil)
    );
    assert_eq!(
        basis_repr(3i32 * e![0] + 5i32 * e![1]),
        Til(
            BasisTerm::<int, mode![0]>::new(3),
            Til(BasisTerm::<int, mode![1]>::new(5), Nil)
        )
    );
    assert_eq!(
        basis_repr(Int::<0>),
        Til(BasisTerm::<Int<0>, ModeEnd>::new(Int), Nil)
    );
    assert_eq!(
        basis_repr(BasisCombination::<Nil>::zero()),
        Til(BasisTerm::<Int<0>, ModeEnd>::new(Int), Nil)
    );
    assert_eq!(
        basis_repr(BasisTerm::<Int<0>, mode![1]>::new(Int)),
        Til(BasisTerm::<Int<0>, ModeEnd>::new(Int), Nil)
    );

    const {
        assert!(is_basis(5i32));
        assert!(is_basis(Int::<0>));
        assert!(is_basis(BasisCombination::<Nil>::zero()));
        assert!(is_basis(e![1, 2]));
        assert!(!is_basis(e![0] + e![1]));
    }

    // proj & unit
    let tup = tuple![7i32, (8i32, 9i32)];
    assert_eq!(proj(tup, e![0]), 7i32);
    assert_eq!(proj(tup, 2i32 * e![1]), tuple![8i32, 9i32]);
    assert_eq!(proj(tup, e![1, 0]), 8i32);
    assert_eq!(proj(tup, 42i32), tup);
    assert_eq!(proj(tup, BasisCombination::<Nil>::zero()), tup);
    assert_eq!(proj(tup, BasisTerm::<Int<0>, mode![99]>::new(Int)), tup);
    assert_eq!(proj(comb, e![1, 0]), 2i32);

    assert_eq!(unit(5i32), Int::<1>);
    assert_eq!(unit(2i32 * e![1]), e![1]);
    assert_eq!(unit(BasisCombination::<Nil>::zero()), Int::<1>);
    assert_eq!(unit(BasisTerm::<Int<0>, mode![1]>::new(Int)), Int::<1>);
}

#[test]
fn custom_non_commutative_basis_coeff_and_tuple_integration() {
    // Verify BasisCoeff decoupling works for a non-Integer custom type (like future F2)
    // and preserves non-commutative left/right multiplication order.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct TagCoeff(u8, u8);

    const impl Item for TagCoeff {
        const RANK: int = 1;
        const DEPTH: int = 0;
        const LEAVES: int = 1;
        type Profile = LeafProfile;
    }

    const impl BasisCoeff for TagCoeff {
        type Unit = TagCoeff;
        fn is_zero(&self) -> bool {
            self.0 == 0 && self.1 == 0
        }
        fn cmp_zero(&self) -> Ordering {
            (self.0, self.1).cmp(&(0, 0))
        }
        fn static_cmp_zero(&self) -> Option<Ordering> {
            None
        }
        fn algebra_unit() -> Self::Unit {
            TagCoeff(1, 0)
        }
    }

    const impl BasisCoeffAdd<TagCoeff> for TagCoeff {
        type Output = TagCoeff;
        fn coeff_add(self, rhs: TagCoeff) -> Self::Output {
            TagCoeff(self.0 ^ rhs.0, self.1 ^ rhs.1)
        }
    }

    const impl BasisCoeffNeg for TagCoeff {
        type Output = TagCoeff;
        fn coeff_neg(self) -> Self::Output {
            self
        }
    }

    const impl BasisCoeffSub<TagCoeff> for TagCoeff {
        type Output = TagCoeff;
        fn coeff_sub(self, rhs: TagCoeff) -> Self::Output {
            TagCoeff(self.0 ^ rhs.0, self.1 ^ rhs.1)
        }
    }

    const impl BasisCoeffMul<TagCoeff> for TagCoeff {
        type Output = TagCoeff;
        fn coeff_mul(self, rhs: TagCoeff) -> Self::Output {
            // Non-commutative product encoding (lhs.0, rhs.1)
            TagCoeff(self.0, rhs.1)
        }
    }

    const impl BasisCoeffEq<TagCoeff> for TagCoeff {
        fn coeff_eq(&self, rhs: &TagCoeff) -> bool {
            self.0 == rhs.0 && self.1 == rhs.1
        }
        fn coeff_static_eq(&self, _: &TagCoeff) -> Option<bool> {
            None
        }
    }

    const impl BasisCoeffOrd<TagCoeff> for TagCoeff {
        fn coeff_cmp(&self, rhs: &TagCoeff) -> Ordering {
            (self.0, self.1).cmp(&(rhs.0, rhs.1))
        }
        fn coeff_static_cmp(&self, _: &TagCoeff) -> Option<Ordering> {
            None
        }
    }

    let a = TagCoeff(2, 3);
    let b = TagCoeff(5, 7);
    let term = BasisTerm::<TagCoeff, mode![1, 0]>::new(b);
    assert_eq!(
        BasisMul::basis_mul(a, term),
        BasisTerm::<TagCoeff, mode![1, 0]>::new(TagCoeff(2, 7))
    );
    assert_eq!(
        term * a,
        BasisTerm::<TagCoeff, mode![1, 0]>::new(TagCoeff(5, 3))
    );
    assert_eq!(
        unit(term),
        BasisTerm::<TagCoeff, mode![1, 0]>::new(TagCoeff(1, 0))
    );
    assert_eq!(walk_mode!(term, 1, 0), b);
    assert_eq!(as_tuple(term), tuple![Int::<0>, (b)]);

    // Tuple leaf integration: fold_leaf inner products with basis vectors and static zeros
    assert_eq!(
        fold_leaf(
            &tuple![Int::<0>, 3i32],
            Int::<0>,
            &BasisDot,
            tuple![e![0], e![1]]
        ),
        3i32 * e![1]
    );
    assert_eq!(
        fold_leaf(
            &tuple![Int::<2>, (3i32, 4i32)],
            Int::<0>,
            &BasisDot,
            tuple![e![0], (e![1, 0], e![1, 1])]
        ),
        basis_tuple![Int::<2>, (3i32, 4i32)]
    );
}

#[test]
fn pycute_atuple_add_sub_scale_and_cross_axis_non_cancellation() {
    // test_atuple.py::TestArithTuple::{test_atuple, test_atuple_sub, test_atuple_sub_of_different_axes_cannot_cancel}
    assert_eq!(
        basis_tuple![1i32, 2i32, 3i32] + basis_tuple![7i32, 8i32, 9i32],
        basis_tuple![8i32, 10i32, 12i32]
    );
    assert_eq!(
        basis_tuple![1i32, 2i32, (3i32, 4i32)] + basis_tuple![7i32, 8i32, (9i32, 10i32)],
        basis_tuple![8i32, 10i32, (12i32, 14i32)]
    );
    assert_eq!(
        basis_tuple![1i32, 2i32, 3i32] + basis_tuple![7i32],
        basis_tuple![8i32, 2i32, 3i32]
    );
    assert_eq!(
        basis_tuple![1i32, 2i32, 3i32] * 4i32,
        basis_tuple![4i32, 8i32, 12i32]
    );
    assert_eq!(
        4i32 * basis_tuple![1i32, 2i32, 3i32],
        basis_tuple![4i32, 8i32, 12i32]
    );
    assert!(!is_basis(basis_tuple![1i32, 2i32, 3i32]));

    assert_eq!(
        basis_tuple![1i32, 2i32, 3i32] - basis_tuple![7i32, 8i32, 9i32],
        basis_tuple![-6i32, -6i32, -6i32]
    );
    assert_eq!(
        basis_tuple![7i32, 8i32, 9i32] - basis_tuple![1i32, 2i32, 3i32],
        basis_tuple![6i32, 6i32, 6i32]
    );
    assert_eq!(
        basis_tuple![1i32, 2i32, (3i32, 4i32)] - basis_tuple![7i32, 8i32, (9i32, 10i32)],
        basis_tuple![-6i32, -6i32, (-6i32, -6i32)]
    );

    assert_eq!(4i32 * e![0] - e![0], 3i32 * e![0]);
    assert_eq!(e![0] - e![0], BasisCombination::<Nil>::zero());
    assert_eq!(e![0] - Int::<0>, e![0]);
    assert_eq!(Int::<0> - e![0], Int::<-1> * e![0]);

    let r1 = 4i32 * e![0] - e![1];
    assert!(!basis_eq(r1, 0i32));
    assert!(!basis_eq(r1 + r1, 0i32));

    let r2 = e![1] - e![0];
    assert!(!basis_eq(r2, 0i32));
    assert!(!basis_eq(r2 + r2, 0i32));

    let r3 = e![0, 1] - e![1, 0];
    assert!(!basis_eq(r3, 0i32));
    assert!(!basis_eq(r3 + r3, 0i32));

    assert!(basis_eq((e![0] - e![0]) + (e![0] - e![0]), 0i32));
}

#[test]
fn pycute_atuple_colex_ordering_over_shape_coordinates() {
    // test_atuple.py::TestArithTuple::test_atuple_lt over shape (4, (5, 6), 2)
    let crd_at = |idx: int| {
        let c0 = idx % 4;
        let c10 = (idx / 4) % 5;
        let c11 = (idx / 20) % 6;
        let c2 = idx / 120;
        from_tuple(int_tuple![c0, (c10, c11), c2])
    };
    let n = 4 * 5 * 6 * 2;
    for i in 0..n {
        assert!(colex_lt(Int::<0>, crd_at(i + 1)));
        let ci = crd_at(i);
        for j in (i + 1)..n {
            let cj = crd_at(j);
            assert!(colex_lt(ci, cj));
            assert!(colex_gt(cj, ci));
        }
    }
}

#[test]
fn pycute_atuple_scaled_basis_equality_and_coord_layouts() {
    // test_atuple.py::TestArithTuple::{test_sbasis, test_sbasis_lt, test_coord_layout}
    assert_eq!(42i32 * e![], 42i32);
    assert!(basis_eq(42i32 * e![0], basis_tuple![42i32, 0, 0, 0, 0]));
    assert!(basis_eq(42i32 * e![1], basis_tuple![0, 42i32, 0, 0, 0]));
    assert!(basis_eq(
        42i32 * e![0, 0],
        basis_tuple![(42i32, 0, 0, 0, 0), 0, 0, 0, 0]
    ));
    assert!(basis_eq(
        42i32 * e![0, 1],
        basis_tuple![(0, 42i32, 0, 0, 0), 0, 0, 0, 0]
    ));
    assert!(basis_eq(
        42i32 * e![1, 0],
        basis_tuple![0, (42i32, 0, 0, 0, 0), 0, 0, 0]
    ));
    assert!(basis_eq(
        42i32 * e![1, 1],
        basis_tuple![0, (0, 42i32, 0, 0, 0), 0, 0, 0]
    ));

    assert_eq!(2i32 * (42i32 * e![]), 84i32);
    assert!(basis_eq(
        2i32 * (42i32 * e![0]),
        basis_tuple![84i32, 0, 0, 0, 0]
    ));
    assert!(basis_eq(
        2i32 * (42i32 * e![1]),
        basis_tuple![0, 84i32, 0, 0, 0]
    ));
    assert!(basis_eq(
        2i32 * (42i32 * e![0, 0]),
        basis_tuple![(84i32, 0, 0, 0, 0), 0, 0, 0, 0]
    ));
    assert!(basis_eq(
        2i32 * (42i32 * e![0, 1]),
        basis_tuple![(0, 84i32, 0, 0, 0), 0, 0, 0, 0]
    ));
    assert!(basis_eq(
        2i32 * (42i32 * e![1, 0]),
        basis_tuple![0, (84i32, 0, 0, 0, 0), 0, 0, 0]
    ));
    assert!(basis_eq(
        2i32 * (42i32 * e![1, 1]),
        basis_tuple![0, (0, 84i32, 0, 0, 0), 0, 0, 0]
    ));

    // Check comparisons and colex ordering across the 13 basis paths from `test_sbasis`
    macro_rules! check_basis_group {
        ($ref_zero:expr; $($term:expr),+ $(,)?) => {
            $(
                assert!(colex_lt(Int::<0>, $term));
                assert!(basis_eq(0i32 * $term, 0i32 * $ref_zero));
                assert!(!basis_eq(0i32 * $term, 3i32 * $ref_zero));
                assert!(basis_eq($term, $term));
                assert!(basis_eq(3i32 * $term, 3i32 * $term));
                assert!(!basis_eq(3i32 * $term, 4i32 * $term));
            )+
        };
    }

    assert!(colex_lt(Int::<0>, e![]));
    check_basis_group!(e![]; e![0], e![1], e![2]);
    check_basis_group!(e![0]; e![0, 0], e![0, 1], e![0, 2]);
    check_basis_group!(e![1]; e![1, 0], e![1, 1], e![1, 2]);
    check_basis_group!(e![2]; e![2, 0], e![2, 1], e![2, 2]);
    assert!(!basis_eq(e![0], e![1]));
    assert!(!basis_eq(e![0], e![0, 0]));
    assert!(!basis_eq(e![0, 1], e![1, 0]));

    // 6 coordinate layout evaluations from `test_coord_layout`
    for i in 0i32..5 {
        for j in 0i32..4 {
            assert!(basis_eq(
                fold_leaf(&int_tuple![i, j], Int::<0>, &BasisDot, tuple![e![0], e![1]]),
                basis_tuple![i, j]
            ));
            assert!(basis_eq(
                fold_leaf(&int_tuple![i, j], Int::<0>, &BasisDot, tuple![e![0], e![2]]),
                basis_tuple![i, 0i32, j]
            ));
            assert!(basis_eq(
                fold_leaf(&int_tuple![i, j], Int::<0>, &BasisDot, tuple![e![2], e![1]]),
                basis_tuple![0i32, j, i]
            ));
            assert!(basis_eq(
                fold_leaf(
                    &int_tuple![i, j],
                    Int::<0>,
                    &BasisDot,
                    tuple![e![2], BasisCombination::<Nil>::zero()]
                ),
                basis_tuple![0i32, 0i32, i]
            ));
            assert!(basis_eq(
                fold_leaf(
                    &int_tuple![i, j],
                    Int::<0>,
                    &BasisDot,
                    tuple![e![2], e![1, 3]]
                ),
                basis_tuple![0i32, (0i32, 0i32, 0i32, j), i]
            ));
        }
    }
    for i in 0i32..4 {
        for j in 0i32..8 {
            let nat = int_tuple![j % 4, j / 4];
            assert!(basis_eq(
                fold_leaf(
                    &tuple![i, nat],
                    Int::<0>,
                    &BasisDot,
                    tuple![e![1], (e![0], 4i32 * e![1])]
                ),
                basis_tuple![j % 4, i + 4 * (j / 4)]
            ));
        }
    }
}

#[test]
fn pycute_atuple_implicit_zero_equality_congruence_and_f2_leaves() {
    // test_atuple.py::TestImplicitZeroEquality
    let a = e![0];
    let b = basis_tuple![1i32, 0i32];
    let c = basis_tuple![1i32];
    assert!(basis_eq(a, b));
    assert!(basis_eq(b, c));
    assert!(basis_eq(basis_tuple![], 0i32));
    assert!(basis_eq(basis_tuple![0i32, 0i32, 0i32], 0i32));
    assert!(basis_eq(basis_tuple![(0i32, 0i32)], 0i32));
    assert!(basis_eq(0i32 * e![2], 0i32));
    assert!(basis_eq(BasisTerm::<int, mode![3]>::new(0), 0i32));
    assert!(!basis_eq(basis_tuple![5i32], 5i32));
    assert!(!basis_eq(5i32, basis_tuple![5i32]));

    // test_atuple.py::TestWeaklyCongruentLeaf & TestCongruentStrideLeaf
    fn require_weak<L: WeaklyCongruent<R>, R: Item>(_: &L, _: &R) {}
    fn require_congruent<L: Congruent<R>, R: Item>(_: &L, _: &R) {}

    let pair = int_tuple![8, 8];
    require_weak(&1i32, &pair);
    require_weak(&basis_tuple![1], &pair);
    require_weak(&basis_tuple![1, 1], &pair);
    require_weak(&basis_tuple![1, 1, 1], &pair);
    require_weak(&e![0, 0], &int_tuple![(4, 5), 8]);
    require_weak(&e![0, 0], &pair);
    require_weak(&e![0, 0], &10i32);
    require_weak(&10i32, &int_tuple![10, 20]);
    require_weak(&e![0, 0], &int_tuple![10, 20]);

    require_congruent(&e![0], &10i32);
    require_congruent(&10i32, &e![0]);
    require_congruent(&e![1, 0], &10i32);
    require_congruent(&basis_tuple![1, 2], &5i32);
    require_congruent(&make_basis_like(10i32), &10i32);
    require_congruent(&10i32, &make_basis_like(10i32));
    require_congruent(&make_basis_like(int_tuple![10, 20]), &int_tuple![10, 20]);
    require_congruent(
        &make_basis_like(int_tuple![10, (20, 30)]),
        &int_tuple![10, (20, 30)],
    );
    require_congruent(
        &make_basis_like(int_tuple![(1, 2), (3, (4, 5))]),
        &int_tuple![(1, 2), (3, (4, 5))],
    );
    require_weak(
        &make_basis_like(int_tuple![10, (20, 30)]),
        &int_tuple![10, (20, 30)],
    );

    // test_atuple.py::TestArithTupleF2Leaves
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

    const impl BasisCoeffAdd<XorF2> for XorF2 {
        type Output = XorF2;
        fn coeff_add(self, rhs: XorF2) -> Self::Output {
            XorF2(self.0 ^ rhs.0)
        }
    }

    const impl BasisCoeffNeg for XorF2 {
        type Output = XorF2;
        fn coeff_neg(self) -> Self::Output {
            self
        }
    }

    const impl BasisCoeffSub<XorF2> for XorF2 {
        type Output = XorF2;
        fn coeff_sub(self, rhs: XorF2) -> Self::Output {
            XorF2(self.0 ^ rhs.0)
        }
    }

    const impl BasisCoeffMul<XorF2> for XorF2 {
        type Output = XorF2;
        fn coeff_mul(self, rhs: XorF2) -> Self::Output {
            XorF2(self.0 & rhs.0)
        }
    }

    const impl BasisCoeffMul<int> for XorF2 {
        type Output = XorF2;
        fn coeff_mul(self, rhs: int) -> Self::Output {
            XorF2(self.0 * (rhs as u8))
        }
    }

    const impl BasisCoeffEq<XorF2> for XorF2 {
        fn coeff_eq(&self, rhs: &XorF2) -> bool {
            self.0 == rhs.0
        }
        fn coeff_static_eq(&self, _: &XorF2) -> Option<bool> {
            None
        }
    }

    const impl BasisCoeffOrd<XorF2> for XorF2 {
        fn coeff_cmp(&self, rhs: &XorF2) -> Ordering {
            self.0.cmp(&rhs.0)
        }
        fn coeff_static_cmp(&self, _: &XorF2) -> Option<Ordering> {
            None
        }
    }

    let f0 = BasisTerm::<XorF2, mode![0]>::new(XorF2(1));
    let f4 = BasisTerm::<XorF2, mode![0]>::new(XorF2(4));
    let f5 = BasisTerm::<XorF2, mode![0]>::new(XorF2(5));

    assert!(is_basis(f0));
    assert_eq!(proj(f0, f0), XorF2(1));
    assert_eq!(unit(f4), f0);
    assert_eq!(as_tuple(f0), tuple![XorF2(1)]);
    assert_eq!(f0 + f0, BasisTerm::<XorF2, mode![0]>::new(XorF2(0)));
    assert!(basis_eq(f0 + f0, 0i32));
    assert_eq!(f4 + f0, f5);
    assert_eq!(-f0, f0);
    assert_eq!(Int::<0> + f0, f0);
    assert_eq!(f0 + Int::<0>, f0);
    assert_eq!(Int::<0> - f0, f0);
    assert_eq!(f0 - Int::<0>, f0);
    assert_eq!(Int::<0> * f0, BasisCombination::zero());
    assert_eq!(f0 * Int::<0>, BasisCombination::zero());
    assert_eq!(f4 - f0, f5);
    assert!(basis_eq(f0 - f0, 0i32));
    assert!(colex_lt(f0, f4));
    assert!(colex_gt(f4, f0));
    assert_eq!(
        BasisMul::basis_mul(XorF2(3), f0),
        BasisTerm::<XorF2, mode![0]>::new(XorF2(1 & 3))
    );
    assert_eq!(f0 * 2i32, BasisTerm::<XorF2, mode![0]>::new(XorF2(2)));
    assert_eq!(
        f0 * 2i32 + 3i32 * e![1],
        BasisCombination::<Nil>::new(tuple![XorF2(2), 3i32])
    );
    assert_eq!(
        fold_leaf(
            &int_tuple![2i32, 3i32],
            Int::<0>,
            &BasisRightDot,
            tuple![f0, e![1]]
        ),
        f0 * 2i32 + 3i32 * e![1]
    );
}

#[test]
fn pycute_atuple_as_tuple_and_const_eval() {
    // Scalars and zero combination
    assert_eq!(as_tuple(0i32), 0i32);
    assert_eq!(as_tuple(5i32), 5i32);
    assert_eq!(as_tuple(Int::<7>), Int::<7>);
    assert_eq!(as_tuple(Int::<-3>), Int::<-3>);
    assert_eq!(as_tuple(BasisCombination::zero()), Int::<0>);
    assert_eq!(as_tuple(BasisTerm::<Int<0>, mode![1]>::new(Int)), Int::<0>);
    assert_eq!(as_tuple(BasisTerm::<Int<5>, ModeEnd>::new(Int)), Int::<5>);

    // Unit and scaled single-path basis terms
    assert_eq!(as_tuple(e![0]), tuple![Int::<1>]);
    assert_eq!(as_tuple(3i32 * e![0]), tuple![3i32]);
    assert_eq!(as_tuple(e![1]), tuple![Int::<0>, Int::<1>]);
    assert_eq!(as_tuple(e![0, 1]), tuple![(Int::<0>, Int::<1>)]);
    assert_eq!(
        as_tuple(e![1, 2]),
        tuple![Int::<0>, (Int::<0>, Int::<0>, Int::<1>)]
    );

    // Multi-term basis combination
    assert_eq!(
        as_tuple(Int::<3> * e![0] + Int::<5> * e![1, 2]),
        Til(
            Int::<3>,
            Til(Til(Int::<0>, Til(Int::<0>, Til(Int::<5>, Nil))), Nil)
        )
    );
    assert_eq!(
        as_tuple(3i32 * e![0] + 5i32 * e![1, 2]),
        Til(3i32, Til(Til(Int::<0>, Til(Int::<0>, Til(5i32, Nil))), Nil))
    );

    // Nested tuples mixing scalars, zero combinations, and basis elements
    assert_eq!(
        as_tuple(basis_tuple![1i32, 2i32, (3i32, 4i32)]),
        int_tuple![1i32, 2i32, (3i32, 4i32)]
    );
    assert_eq!(
        as_tuple(tuple![e![0], 4i32, BasisCombination::<Nil>::zero()]),
        tuple![(Int::<1>,), 4i32, Int::<0>]
    );

    // Const evaluation
    const _: () = {
        assert!(as_tuple(BasisCombination::<Nil>::zero()) == Int::<0>);
        let t1 = as_tuple(e![1]);
        assert!(t1.0 == Int::<0> && t1.1.0 == Int::<1>);
        let _: Nil = t1.1.1;
        let t2 = as_tuple(Int::<3> * e![0] + Int::<5> * e![1, 2]);
        assert!(
            t2.0 == Int::<3>
                && t2.1.0.0 == Int::<0>
                && t2.1.0.1.0 == Int::<0>
                && t2.1.0.1.1.0 == Int::<5>
        );
        let _: Nil = t2.1.0.1.1.1;
        let _: Nil = t2.1.1;
    };
}

#[test]
fn merge_add_preserves_operand_order_normalized_index_and_static_cmp() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct OrderedTag(u8);

    const impl Item for OrderedTag {
        const RANK: int = 1;
        const DEPTH: int = 0;
        const LEAVES: int = 1;
        type Profile = LeafProfile;
    }

    const impl BasisCoeff for OrderedTag {
        type Unit = OrderedTag;
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
            OrderedTag(1)
        }
    }

    const impl BasisCoeffAdd<OrderedTag> for OrderedTag {
        type Output = OrderedTag;
        fn coeff_add(self, rhs: OrderedTag) -> OrderedTag {
            OrderedTag(self.0 * 10 + rhs.0)
        }
    }

    // PL = [1] > PR = [0] triggers MergeAdd<..., PathOrdering::Greater>;
    // mode [1] must still evaluate lhs.coeff_add(rhs) = 1*10 + 2 = 12 (not 21).
    let lhs = BasisTerm::<OrderedTag, Mode<1>>::new(OrderedTag(1));
    let rhs = BasisTerm::<OrderedTag, Mode<0>>::new(OrderedTag(9))
        + BasisTerm::<OrderedTag, Mode<1>>::new(OrderedTag(2));
    let sum = lhs + rhs;
    assert_eq!(sum.get::<0>(), OrderedTag(9));
    assert_eq!(sum.get::<1>(), OrderedTag(12));

    // BASIS_NORMALIZED_INDEX handles boundary modes including MAX_MODE == int::MAX without overflow.
    const {
        assert!(BASIS_NORMALIZED_INDEX::<0, -1> == 0);
        assert!(BASIS_NORMALIZED_INDEX::<5, -1> == 5);
        assert!(BASIS_NORMALIZED_INDEX::<-1, 0> == 0);
        assert!(BASIS_NORMALIZED_INDEX::<-2, 1> == 0);
        assert!(BASIS_NORMALIZED_INDEX::<-1, 1> == 1);
        assert!(BASIS_NORMALIZED_INDEX::<-1, I32_MAX> == I32_MAX);
        assert!(BASIS_NORMALIZED_INDEX::<I32_MIN, I32_MAX> == 0);
    }

    // BasisCoeffOrd::STATIC_CMP and BasisCoeff static constants agree with integer::cmp.
    const {
        assert!(matches!(
            <Int<-1> as BasisCoeffOrd<Int<0>>>::STATIC_CMP,
            Some(Ordering::Less)
        ));
        assert!(<Int<-1> as BasisCoeffOrd<int>>::STATIC_CMP.is_none());
        assert!(matches!(
            <Int<-1> as BasisCoeff>::STATIC_CMP_ZERO,
            Some(Ordering::Less)
        ));
        assert!(matches!(<Int<0> as BasisCoeff>::STATIC_VALUE, Some(0)));
        assert!(<int as BasisCoeff>::STATIC_VALUE.is_none());

        type LowBigTerms =
            Til<ScaledBasis<Int<100>, mode![0]>, Til<ScaledBasis<Int<2>, mode![1]>, Nil>>;
        type HighBigTerms =
            Til<ScaledBasis<Int<1>, mode![0]>, Til<ScaledBasis<Int<3>, mode![1]>, Nil>>;
        assert!(matches!(
            <LowBigTerms as MergeColex<HighBigTerms>>::STATIC_ORDER,
            Some(Ordering::Less)
        ));
    }
}
