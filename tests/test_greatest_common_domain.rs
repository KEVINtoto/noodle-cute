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

use noodle_cute::algebra::composition::Composition;
use noodle_cute::algebra::greatest_common_domain::{
    GreatestCommonDomain, IntoGcdShape, greatest_common_domain,
};
use noodle_cute::algebra::logical_divide::LogicalDivide;
use noodle_cute::int_tuple;
use noodle_cute::integer::{Int, gcd};
use noodle_cute::layout::{Layout, LayoutBase};
use noodle_cute::shape::Shape;
use noodle_cute::tuple::{Get, GetOutput, Nil, Til};

// Universal post-condition checker mirroring `test/test_greatest_common_domain.py`

fn assert_gcd_postcondition<A, B, SA, SB, SR, DR, DA, DB, QSA, QSB>(a: A, b: B) -> Layout<SR, DR>
where
    A: IntoGcdShape<Shape = SA> + GreatestCommonDomain<B, Shape = SR, Stride = DR> + Debug,
    B: IntoGcdShape<Shape = SB> + GreatestCommonDomain<A> + Debug,
    SA: Shape + Debug + Composition<Layout<SR, DR>> + LogicalDivide<Layout<SR, DR>, Output = DA>,
    SB: Shape + Debug + Composition<Layout<SR, DR>> + LogicalDivide<Layout<SR, DR>, Output = DB>,
    SR: Shape + Debug,
    DR: noodle_cute::stride::Stride<SR> + Debug,
    Layout<SR, DR>: PartialEq<
        Layout<<B as GreatestCommonDomain<A>>::Shape, <B as GreatestCommonDomain<A>>::Stride>,
    >,
    Layout<<B as GreatestCommonDomain<A>>::Shape, <B as GreatestCommonDomain<A>>::Stride>: Debug,
    DA: LayoutBase + Get<1, Output = QSA>,
    DB: LayoutBase + Get<1, Output = QSB>,
    QSA: GreatestCommonDomain<QSB>,
    <QSA as GreatestCommonDomain<QSB>>::Shape: Shape,
{
    let r = greatest_common_domain(a, b);
    let shape_a = a.into_gcd_shape();
    let shape_b = b.into_gcd_shape();

    // Post-condition: depth is 1
    assert_eq!(r.depth(), 1, "depth != 1 for gcd({a:?}, {b:?}) = {r:?}");

    // Post-condition: size(R) | size(A), size(R) | size(B), size(R) | gcd(size(A), size(B))
    let size_a = shape_a.size();
    let size_b = shape_b.size();
    let size_r = r.size();
    assert_eq!(size_a % size_r, 0, "size(R) does not divide size(A)");
    assert_eq!(size_b % size_r, 0, "size(R) does not divide size(B)");
    assert_eq!(
        gcd(size_a, size_b) % size_r,
        0,
        "size(R) does not divide gcd(size(A), size(B))"
    );

    // Post-condition: composition(shape(A), R) and composition(shape(B), R) are admissible
    let _ = shape_a.composition(r);
    let _ = shape_b.composition(r);

    // Post-condition: symmetric in A and B
    let r_ba = greatest_common_domain(b, a);
    assert_eq!(r, r_ba, "asymmetric gcd({a:?}, {b:?})={r:?} vs {r_ba:?}");

    // Post-condition: GCD(logical_divide(shape(A), R)[1], logical_divide(shape(B), R)[1]) has size 1
    let ra = shape_a.logical_divide(r);
    let rb = shape_b.logical_divide(r);
    let q_a: GetOutput<DA, 1> = Get::<1>::get(&ra);
    let q_b: GetOutput<DB, 1> = Get::<1>::get(&rb);
    let rem_gcd = greatest_common_domain(q_a, q_b);
    assert_eq!(
        rem_gcd.size(),
        1,
        "quotient gcd not trivial for gcd({a:?}, {b:?}) = {r:?}"
    );

    r
}

macro_rules! check_gcd {
    ($a:expr, $b:expr => ($($es:expr),+ $(,)?) : ($($ed:expr),+ $(,)?)) => {{
        let expected = Layout::new(int_tuple![$(Int::<$es>),+], int_tuple![$(Int::<$ed>),+]);
        let r = assert_gcd_postcondition($a, $b);
        assert_eq!(r, expected);
    }};
}

#[test]
fn test_known_expected_results_scalars_and_canonical() {
    // Int vs Ints
    check_gcd!(Int::<1>, Int::<1> => (1) : (0));
    check_gcd!(Int::<10>, Int::<10> => (10) : (1));
    check_gcd!(int_tuple![Int::<10>], int_tuple![Int::<10>] => (10) : (1));
    check_gcd!(Int::<7>, Int::<9> => (1) : (0));
    check_gcd!(Int::<12>, Int::<18> => (6) : (1));

    // Canonical examples
    check_gcd!(
        int_tuple![Int::<16>, Int::<3>],
        int_tuple![Int::<16>, Int::<3>]
        => (16, 3) : (1, 16)
    );
    check_gcd!(
        int_tuple![Int::<5>, Int::<2>],
        Int::<10>
        => (5, 2) : (1, 5)
    );
    check_gcd!(
        int_tuple![Int::<5>, Int::<3>, Int::<4>],
        int_tuple![Int::<10>, Int::<6>]
        => (5, 2) : (1, 30)
    );
    check_gcd!(
        int_tuple![Int::<4>, Int::<3>, Int::<5>],
        int_tuple![Int::<6>, Int::<10>]
        => (2, 5) : (1, 12)
    );
    check_gcd!(
        int_tuple![Int::<5>, Int::<6>, Int::<3>],
        int_tuple![Int::<6>, Int::<15>]
        => (3) : (30)
    );
    check_gcd!(
        int_tuple![Int::<5>, Int::<3>, Int::<3>, Int::<4>],
        int_tuple![Int::<10>, Int::<6>, Int::<3>]
        => (5) : (1)
    );
    check_gcd!(
        int_tuple![Int::<1>, Int::<5>, Int::<3>, Int::<3>, Int::<4>],
        int_tuple![Int::<10>, Int::<6>, Int::<3>]
        => (5) : (1)
    );
    check_gcd!(
        int_tuple![Int::<5>, Int::<3>, Int::<3>, Int::<4>],
        int_tuple![Int::<10>, Int::<6>, Int::<1>, Int::<3>]
        => (5) : (1)
    );
    check_gcd!(
        int_tuple![Int::<2>, Int::<21>],
        int_tuple![Int::<3>, Int::<14>]
        => (7) : (6)
    );
    check_gcd!(
        int_tuple![Int::<6>, Int::<35>],
        int_tuple![Int::<15>, Int::<14>]
        => (3, 7) : (1, 30)
    );
    check_gcd!(
        int_tuple![Int::<16>, Int::<64>],
        int_tuple![Int::<4>, Int::<16>, Int::<16>]
        => (4, 4, 4, 16) : (1, 4, 16, 64)
    );
    check_gcd!(
        int_tuple![Int::<5>, Int::<3>],
        int_tuple![Int::<3>, Int::<5>]
        => (1) : (0)
    );
    check_gcd!(
        int_tuple![Int::<5>, Int::<5>, Int::<3>],
        int_tuple![Int::<5>, Int::<3>, Int::<5>]
        => (5) : (1)
    );
    check_gcd!(
        int_tuple![Int::<7>, Int::<2>, Int::<3>, Int::<2>, Int::<2>, Int::<3>],
        int_tuple![Int::<504>]
        => (7, 2, 3, 2, 2, 3) : (1, 7, 14, 42, 84, 168)
    );
    check_gcd!(
        int_tuple![Int::<7>, Int::<2>, Int::<3>, Int::<2>, Int::<2>, Int::<3>, Int::<2>],
        int_tuple![Int::<7>, Int::<3>, Int::<2>, Int::<2>, Int::<3>, Int::<2>, Int::<2>]
        => (7, 2, 2) : (1, 42, 504)
    );
    check_gcd!(
        int_tuple![Int::<7>, Int::<2>, Int::<2>, Int::<2>, Int::<2>, Int::<3>, Int::<5>],
        int_tuple![Int::<7>, Int::<3>, Int::<2>, Int::<2>, Int::<2>, Int::<2>, Int::<5>]
        => (7, 5) : (1, 336)
    );
}

#[test]
fn test_known_expected_results_unequal_and_mode_boundaries() {
    // Unequal sizes
    check_gcd!(
        int_tuple![Int::<4>, Int::<3>],
        int_tuple![Int::<6>, Int::<12>, Int::<10>]
        => (2) : (1)
    );
    check_gcd!(
        int_tuple![Int::<16>, Int::<12>],
        int_tuple![Int::<6>, Int::<3>, Int::<3>]
        => (2) : (1)
    );
    check_gcd!(
        int_tuple![Int::<12>, Int::<10>],
        int_tuple![Int::<7>, Int::<12>]
        => (1) : (0)
    );
    check_gcd!(
        int_tuple![Int::<10>, Int::<8>],
        int_tuple![Int::<8>, Int::<5>, Int::<4>]
        => (2, 2) : (1, 40)
    );
    check_gcd!(
        int_tuple![Int::<3>, Int::<5>, Int::<4>],
        int_tuple![Int::<5>, Int::<12>, Int::<5>]
        => (4) : (15)
    );
    check_gcd!(
        int_tuple![Int::<3>, Int::<5>, Int::<8>],
        int_tuple![Int::<6>, Int::<5>, Int::<2>, Int::<10>]
        => (3, 2, 2) : (1, 30, 60)
    );
    check_gcd!(
        int_tuple![Int::<8>, Int::<12>],
        int_tuple![Int::<6>, Int::<8>, Int::<6>, Int::<8>]
        => (2, 2, 2) : (1, 24, 48)
    );
    check_gcd!(
        int_tuple![Int::<12>, Int::<12>, Int::<5>],
        int_tuple![Int::<9>, Int::<12>, Int::<16>, Int::<16>]
        => (3) : (1)
    );
    check_gcd!(
        int_tuple![Int::<8>, Int::<6>, Int::<4>, Int::<2>],
        int_tuple![Int::<12>, Int::<6>, Int::<12>, Int::<3>]
        => (4) : (1)
    );
    check_gcd!(
        int_tuple![Int::<9>, Int::<6>, Int::<6>, Int::<9>],
        int_tuple![Int::<6>, Int::<6>, Int::<9>, Int::<2>]
        => (3, 3) : (1, 108)
    );
    check_gcd!(
        int_tuple![Int::<6>, Int::<9>, Int::<4>, Int::<6>],
        int_tuple![Int::<12>, Int::<9>, Int::<4>, Int::<6>]
        => (6, 2, 2, 3) : (1, 108, 216, 432)
    );

    // Mode boundary alignment
    check_gcd!(
        int_tuple![Int::<9>, Int::<6>],
        int_tuple![Int::<6>, Int::<6>, Int::<8>]
        => (3) : (1)
    );
    check_gcd!(
        int_tuple![Int::<8>, Int::<6>, Int::<5>],
        int_tuple![Int::<3>, Int::<8>]
        => (1) : (0)
    );
    check_gcd!(
        int_tuple![Int::<3>, Int::<10>],
        int_tuple![Int::<2>, Int::<6>, Int::<5>, Int::<6>]
        => (1) : (0)
    );
    check_gcd!(
        int_tuple![Int::<2>, Int::<6>, Int::<9>],
        int_tuple![Int::<3>, Int::<6>, Int::<3>]
        => (1) : (0)
    );
    check_gcd!(
        int_tuple![Int::<5>, Int::<10>, Int::<12>],
        int_tuple![Int::<2>, Int::<30>, Int::<2>, Int::<5>]
        => (1) : (0)
    );
    check_gcd!(
        int_tuple![Int::<8>, Int::<3>, Int::<2>, Int::<6>],
        int_tuple![Int::<6>, Int::<12>, Int::<2>]
        => (2) : (1)
    );
    check_gcd!(
        int_tuple![Int::<8>, Int::<9>, Int::<6>, Int::<12>],
        int_tuple![Int::<6>, Int::<6>, Int::<2>]
        => (2) : (1)
    );
    check_gcd!(
        int_tuple![Int::<12>, Int::<10>, Int::<6>, Int::<10>],
        int_tuple![Int::<9>, Int::<10>, Int::<12>]
        => (3) : (1)
    );
    check_gcd!(
        int_tuple![Int::<2>, Int::<10>, Int::<10>],
        int_tuple![Int::<25>, Int::<4>, Int::<2>]
        => (2) : (100)
    );
    check_gcd!(
        int_tuple![Int::<14>, Int::<6>, Int::<3>],
        int_tuple![Int::<6>, Int::<42>]
        => (2, 2, 3) : (1, 42, 84)
    );
}

#[test]
fn test_known_expected_results_greedy_and_equal_sizes() {
    // Greedy trade-offs & later aligned factors
    check_gcd!(
        int_tuple![Int::<3>, Int::<12>],
        int_tuple![Int::<2>, Int::<12>]
        => (2) : (6)
    );
    check_gcd!(
        int_tuple![Int::<4>, Int::<10>],
        int_tuple![Int::<5>, Int::<2>, Int::<4>]
        => (2) : (20)
    );
    check_gcd!(
        int_tuple![Int::<4>, Int::<18>],
        int_tuple![Int::<9>, Int::<2>, Int::<4>]
        => (2) : (36)
    );
    check_gcd!(
        int_tuple![Int::<12>, Int::<4>],
        int_tuple![Int::<8>, Int::<12>]
        => (4, 2) : (1, 24)
    );
    check_gcd!(
        int_tuple![Int::<6>, Int::<2>, Int::<6>, Int::<2>],
        int_tuple![Int::<4>, Int::<12>]
        => (2, 2) : (1, 12)
    );
    check_gcd!(
        int_tuple![Int::<6>, Int::<2>, Int::<8>],
        int_tuple![Int::<2>, Int::<2>, Int::<12>]
        => (2, 4) : (1, 12)
    );
    check_gcd!(
        int_tuple![Int::<7>, Int::<2>, Int::<25>, Int::<2>],
        int_tuple![Int::<10>, Int::<70>]
        => (5, 2) : (70, 350)
    );

    // Equal sizes, unrelated factorizations
    check_gcd!(
        int_tuple![Int::<7>, Int::<35>, Int::<10>],
        int_tuple![Int::<5>, Int::<14>, Int::<35>]
        => (5) : (490)
    );
    check_gcd!(
        int_tuple![Int::<6>, Int::<42>, Int::<9>],
        int_tuple![Int::<9>, Int::<2>, Int::<3>, Int::<42>]
        => (3, 3) : (1, 756)
    );
    check_gcd!(
        int_tuple![Int::<2>, Int::<14>, Int::<3>, Int::<3>],
        int_tuple![Int::<42>, Int::<6>]
        => (2, 7, 3) : (1, 2, 84)
    );
    check_gcd!(
        int_tuple![Int::<2>, Int::<6>, Int::<3>, Int::<2>, Int::<6>],
        int_tuple![Int::<18>, Int::<6>, Int::<2>, Int::<2>]
        => (2, 3, 2) : (1, 2, 216)
    );
    check_gcd!(
        int_tuple![Int::<12>, Int::<2>, Int::<42>],
        int_tuple![Int::<3>, Int::<28>, Int::<3>, Int::<2>, Int::<2>]
        => (3, 4, 2) : (1, 3, 504)
    );
    check_gcd!(
        int_tuple![Int::<15>, Int::<14>, Int::<6>],
        int_tuple![Int::<10>, Int::<3>, Int::<42>]
        => (5, 7, 6) : (1, 30, 210)
    );
    check_gcd!(
        int_tuple![Int::<3>, Int::<4>, Int::<30>],
        int_tuple![Int::<2>, Int::<30>, Int::<2>, Int::<3>]
        => (2, 5, 2, 3) : (6, 12, 60, 120)
    );
}

#[test]
fn test_singleton_leaves_layout_strides_and_nested() {
    // Leading / interior `1`s are transparent
    let r1 = greatest_common_domain(
        int_tuple![Int::<5>, Int::<3>, Int::<3>, Int::<4>],
        int_tuple![Int::<10>, Int::<6>, Int::<3>],
    );
    let r2 = greatest_common_domain(
        int_tuple![Int::<1>, Int::<5>, Int::<3>, Int::<3>, Int::<4>],
        int_tuple![Int::<10>, Int::<6>, Int::<3>],
    );
    let r3 = greatest_common_domain(
        int_tuple![Int::<5>, Int::<3>, Int::<3>, Int::<4>],
        int_tuple![Int::<10>, Int::<6>, Int::<1>, Int::<3>],
    );
    let r4 = greatest_common_domain(
        int_tuple![
            Int::<5>, Int::<1>, Int::<1>, Int::<3>, Int::<1>, Int::<3>, Int::<4>
        ],
        int_tuple![Int::<10>, Int::<6>, Int::<3>],
    );
    assert_eq!(r1, r2);
    assert_eq!(r1, r3);
    assert_eq!(r1, r4);

    // Depends only on shape; input strides are ignored
    let l1 = Layout::new(
        int_tuple![Int::<5>, Int::<3>, Int::<4>],
        int_tuple![Int::<1>, Int::<5>, Int::<15>],
    );
    let l2 = Layout::new(
        int_tuple![Int::<5>, Int::<3>, Int::<4>],
        int_tuple![Int::<12>, Int::<4>, Int::<1>],
    );
    let l3 = Layout::new(
        int_tuple![Int::<5>, Int::<3>, Int::<4>],
        int_tuple![Int::<0>, Int::<0>, Int::<0>],
    );
    let b = Layout::new(
        int_tuple![Int::<10>, Int::<6>],
        int_tuple![Int::<1>, Int::<10>],
    );
    let r_shape = greatest_common_domain(
        int_tuple![Int::<5>, Int::<3>, Int::<4>],
        int_tuple![Int::<10>, Int::<6>],
    );
    assert_eq!(greatest_common_domain(l1, b), r_shape);
    assert_eq!(l1.greatest_common_domain(b), r_shape);
    assert_eq!(greatest_common_domain(l2, b), r_shape);
    assert_eq!(greatest_common_domain(l3, b), r_shape);
    assert_eq!(
        greatest_common_domain(l1, int_tuple![Int::<10>, Int::<6>]),
        r_shape
    );

    // Nested shapes share the same leaf sequence
    let r_nested = greatest_common_domain(
        int_tuple![Int::<2>, (Int::<3>, Int::<4>)],
        int_tuple![Int::<6>, Int::<4>],
    );
    let r_flat = greatest_common_domain(
        int_tuple![Int::<2>, Int::<3>, Int::<4>],
        int_tuple![Int::<24>],
    );
    assert_eq!(r_nested, r_flat);
    assert_eq!(
        r_nested,
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<4>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>],
        )
    );
}

type PairLayout<S0, S1, D0, D1> = Layout<Til<S0, Til<S1, Nil>>, Til<D0, Til<D1, Nil>>>;

#[test]
fn test_gcd_const_eval_nil_and_large_i32() {
    // Const evaluation & zero-sized static output
    const G_CONST: PairLayout<Int<3>, Int<7>, Int<1>, Int<30>> = greatest_common_domain(
        int_tuple![Int::<6>, Int::<35>],
        int_tuple![Int::<15>, Int::<14>],
    );
    assert_eq!(size_of_val(&G_CONST), 0);
    assert_eq!(
        G_CONST,
        Layout::new(
            int_tuple![Int::<3>, Int::<7>],
            int_tuple![Int::<1>, Int::<30>],
        )
    );

    // Nil shapes
    assert_eq!(
        greatest_common_domain(Nil, Nil),
        Layout::new(int_tuple![Int::<1>], int_tuple![Int::<0>])
    );
    assert_eq!(
        greatest_common_domain(Nil, Int::<8>),
        Layout::new(int_tuple![Int::<1>], int_tuple![Int::<0>])
    );
    assert_eq!(
        greatest_common_domain(Layout::new(Nil, Nil), Layout::new(Int::<8>, Int::<1>)),
        Layout::new(int_tuple![Int::<1>], int_tuple![Int::<0>])
    );

    // Full i32 range: final leaf product `2 * 1_500_000_000` exceeds i32::MAX,
    // yet no further leaves remain so no spurious overflow occurs!
    const BIG_EXACT: PairLayout<Int<2>, Int<1_500_000_000>, Int<1>, Int<2>> =
        greatest_common_domain(
            int_tuple![Int::<2>, Int::<1_500_000_000>],
            int_tuple![Int::<2>, Int::<1_500_000_000>],
        );
    assert_eq!(size_of_val(&BIG_EXACT), 0);

    const BIG_PARTIAL: PairLayout<Int<2>, Int<500_000_000>, Int<1>, Int<2>> =
        greatest_common_domain(
            int_tuple![Int::<2>, Int::<1_500_000_000>],
            int_tuple![Int::<2>, Int::<1_000_000_000>],
        );
    assert_eq!(size_of_val(&BIG_PARTIAL), 0);
}

#[test]
fn test_gcd_dynamic_and_mixed_shapes() {
    // Dynamic rank-1 shapes -> exact rank-1 output
    assert_eq!(
        greatest_common_domain(12i32, 18i32),
        Layout::new(int_tuple![6i32], int_tuple![1i32])
    );
    assert_eq!(
        greatest_common_domain(7i32, 9i32),
        Layout::new(int_tuple![1i32], int_tuple![0i32])
    );
    assert_eq!(
        greatest_common_domain(1i32, 1i32),
        Layout::new(int_tuple![1i32], int_tuple![0i32])
    );
    assert_eq!(
        greatest_common_domain(Int::<12>, 18i32),
        Layout::new(int_tuple![6i32], int_tuple![1i32])
    );

    // Static prefix preserved before dynamic suffix!
    const MIXED_CONST: PairLayout<Int<4>, i32, Int<1>, i32> =
        greatest_common_domain(int_tuple![Int::<4>, 12i32], int_tuple![Int::<4>, 18i32]);
    assert_eq!(
        MIXED_CONST,
        Layout::new(int_tuple![Int::<4>, 6i32], int_tuple![Int::<1>, 4i32],)
    );

    // Mixed static/dynamic symmetry & size preservation on multi-mode dynamic inputs
    let a = int_tuple![5i32, 3i32, 4i32];
    let b = int_tuple![10i32, 6i32];
    let g_ab = greatest_common_domain(a, b);
    let g_ba = greatest_common_domain(b, a);
    assert_eq!(g_ab, g_ba);
    assert_eq!(g_ab.depth(), 1);
    assert_eq!(g_ab.size(), 10);
    for i in 0..10i32 {
        let expected = Layout::new(
            int_tuple![Int::<5>, Int::<2>],
            int_tuple![Int::<1>, Int::<30>],
        );
        assert_eq!(g_ab(i), expected(i));
    }

    // Dynamic trailing leaf beyond static prefix: stops early with pure static output,
    // while still validating the dynamic leaf is positive!
    assert_eq!(
        greatest_common_domain(int_tuple![Int::<4>], int_tuple![Int::<4>, 7i32]),
        Layout::new(int_tuple![Int::<4>], int_tuple![Int::<1>])
    );
    assert!(
        std::panic::catch_unwind(|| {
            greatest_common_domain(int_tuple![Int::<4>], int_tuple![Int::<4>, 0i32])
        })
        .is_err()
    );

    // Dynamic full i32 range without spurious overflow
    assert_eq!(
        greatest_common_domain(
            int_tuple![2i32, 1_500_000_000i32],
            int_tuple![2i32, 1_000_000_000i32],
        )
        .size(),
        1_000_000_000
    );

    // Both static and dynamic divides_product paths:
    // At (pA=5, a=2) vs (pB=3, b=1.05B), eB = 3.15B overflows i32 while eA = 10 fits and divides 3.15B,
    // allowing A to advance to a_2 = 21 at pA = 10 and match factor 7 at offset lcm(10, 3) = 30!
    const STATIC_DIV_PROD: Layout<Til<Int<7>, Nil>, Til<Int<30>, Nil>> = greatest_common_domain(
        int_tuple![Int::<5>, Int::<2>, Int::<21>],
        int_tuple![Int::<3>, Int::<1_050_000_000>],
    );
    assert_eq!(
        STATIC_DIV_PROD,
        Layout::new(int_tuple![Int::<7>], int_tuple![Int::<30>])
    );
    let g_div_prod = greatest_common_domain(
        int_tuple![5i32, 2i32, 21i32],
        int_tuple![3i32, 1_050_000_000i32],
    );
    assert_eq!(g_div_prod.size(), 7);
    assert_eq!(g_div_prod(0i32), 0);
    assert_eq!(g_div_prod(1i32), 30);
    assert_eq!(g_div_prod(6i32), 180);
}

#[test]
fn test_gcd_dynamic_failures() {
    // Dynamic zero and negative extents rejected
    assert!(std::panic::catch_unwind(|| greatest_common_domain(-4i32, 4i32)).is_err());
    assert!(std::panic::catch_unwind(|| greatest_common_domain(4i32, -4i32)).is_err());
    assert!(
        std::panic::catch_unwind(|| {
            greatest_common_domain(int_tuple![4i32, -2i32], int_tuple![4i32, 2i32])
        })
        .is_err()
    );
    assert!(std::panic::catch_unwind(|| greatest_common_domain(0i32, 4i32)).is_err());
    assert!(std::panic::catch_unwind(|| greatest_common_domain(4i32, 0i32)).is_err());
    assert!(
        std::panic::catch_unwind(|| {
            greatest_common_domain(int_tuple![4i32, 0i32], int_tuple![4i32, 2i32])
        })
        .is_err()
    );

    // Dynamic lcm / running offset overflow panics
    assert!(
        std::panic::catch_unwind(|| {
            greatest_common_domain(
                int_tuple![1_500_000_000i32, 2i32, 5i32],
                int_tuple![1_500_000_000i32, 2i32, 5i32],
            )
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            greatest_common_domain(int_tuple![65536i32, 5i32], int_tuple![65537i32, 5i32])
        })
        .is_err()
    );
}

#[test]
#[should_panic]
fn test_gcd_panics_on_dynamic_zero_extent() {
    let _ = greatest_common_domain(0i32, 4i32);
}
