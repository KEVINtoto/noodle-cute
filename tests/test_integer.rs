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

use core::cmp::Ordering;
use std::panic::catch_unwind;

use noodle_cute::integer::*;
use noodle_cute::{NoodleCuteError, NoodleCuteResult, Result};

const I32_MIN: int = int::MIN;
const I32_MAX: int = int::MAX;

#[test]
fn test_error_and_result_trait() {
    const OK_VAL: int = Result::Ok(42).v();
    assert_eq!(OK_VAL, 42);
    assert_eq!(Result::Ok(-7).v(), -7);

    let err_ovf = NoodleCuteError::ArithmeticOverflow;
    let err_div = NoodleCuteError::DivisionByZero;
    let err_neg = NoodleCuteError::SignedMinValueOverflow;
    let err_cast = NoodleCuteError::UnsignedToSignedOutOfRange;
    assert_eq!(err_ovf.clone(), err_ovf);
    assert_ne!(err_ovf, err_div);
    assert_eq!(err_ovf.to_string(), "integer arithmetic overflow");
    assert_eq!(err_div.to_string(), "integer division by zero");
    assert_eq!(
        err_neg.to_string(),
        "cannot negate or take the absolute value of int::MIN"
    );
    assert_eq!(
        err_cast.to_string(),
        "unsigned integer is out of range for signed int"
    );

    let _: &dyn core::error::Error = &err_ovf;
    assert!(catch_unwind(|| Result::<int>::Err(err_ovf).v()).is_err());
    assert!(catch_unwind(|| Result::<int>::Err(err_div).v()).is_err());
    assert!(catch_unwind(|| Result::<int>::Err(err_neg).v()).is_err());
    assert!(catch_unwind(|| Result::<int>::Err(err_cast).v()).is_err());
}

#[test]
fn test_integer_trait_and_representation() {
    assert_eq!(core::mem::size_of::<Int<42>>(), 0);
    assert_eq!(core::mem::size_of::<Int<I32_MIN>>(), 0);

    // Int<N> properties
    let s = Int::<42>;
    assert_eq!(s.value(), 42);
    assert_eq!(Int::<42>::STATIC_VALUE, Some(42));
    assert_eq!(<Int<42> as StaticInteger>::VALUE, 42);
    assert_eq!(format!("{s}"), "42");
    assert_eq!(format!("{s:?}"), "Int");
    assert_eq!(<Int<42> as Default>::default(), s);
    assert_eq!(s.clone(), s);

    // Negative and boundary Int<N>
    let neg_s = Int::<-5>;
    assert_eq!(neg_s.value(), -5);
    assert_eq!(Int::<-5>::STATIC_VALUE, Some(-5));
    assert_eq!(format!("{neg_s}"), "-5");
    assert_eq!(Int::<I32_MIN>::STATIC_VALUE, Some(i32::MIN));
    assert_eq!(Int::<I32_MAX>::VALUE, i32::MAX);

    // Primitive int (i32) properties
    let d = 17i32;
    assert_eq!(d.value(), 17);
    assert_eq!(<int as Integer>::STATIC_VALUE, None);

    // Const access and compile-time predicates
    const {
        assert!(Int::<42>::IS_STATIC);
        assert!(Int::<-5>::IS_STATIC);
        assert!(!int::IS_STATIC);
        assert!(Int::<-5>.value() == -5);
        assert!((-5i32).value() == -5);
        assert!(BOTH_STATIC::<Int<1>, Int<2>>);
        assert!(!BOTH_STATIC::<Int<1>, int>);
        assert!(!BOTH_STATIC::<int, Int<2>>);
        assert!(!BOTH_STATIC::<int, int>);

        assert!(IS_NON_NEG::<0>);
        assert!(IS_NON_NEG::<I32_MAX>);
        assert!(!IS_NON_NEG::<-1>);
        assert!(!IS_NON_NEG::<I32_MIN>);
    }
}

#[test]
fn test_const_cmp_and_heterogeneous_ordering() {
    const {
        assert!(Int::<3> == Int::<3>);
        assert!(Int::<3> != Int::<4>);
        assert!(Int::<-1> < Int::<0>);
        assert!(Int::<3> <= Int::<3>);
        assert!(Int::<3> <= Int::<4>);
        assert!(Int::<5> > 3i32);
        assert!(Int::<5> >= 5i32);
        assert!(5i32 == Int::<5>);
        assert!(3i32 < Int::<5>);
        assert!(3i32 <= Int::<3>);
        assert!(5i32 >= Int::<3>);
        assert!(Int::<I32_MIN> < Int::<I32_MAX>);
        assert!(Int::<I32_MIN> <= i32::MIN);
        assert!(i32::MAX >= Int::<I32_MAX>);
        assert!(eq(Int::<-1>, -1i32));
        assert!(ne(Int::<-1>, 0i32));
        assert!(lt(Int::<-1>, 0i32));
        assert!(le(-1i32, Int::<-1>));
        assert!(gt(5i32, Int::<3>));
        assert!(ge(Int::<5>, Int::<5>));
        assert!(matches!(cmp(Int::<-1>, 0i32), Ordering::Less));
        assert!(matches!(cmp(Int::<5>, 5i32), Ordering::Equal));
        assert!(matches!(cmp(10i32, Int::<5>), Ordering::Greater));
    }

    let dyn_three = std::hint::black_box(3i32);
    assert!(Int::<3> <= dyn_three);
    assert!(dyn_three <= Int::<4>);
    assert!(Int::<4> >= dyn_three);
    assert!(dyn_three >= Int::<3>);
    assert!(!(Int::<4> <= dyn_three));
    assert!(!(dyn_three >= Int::<4>));

    assert_eq!(Int::<3>.cmp(&Int::<3>), Ordering::Equal);
    assert_eq!(Int::<3>.partial_cmp(&4i32), Some(Ordering::Less));
    assert_eq!(4i32.partial_cmp(&Int::<3>), Some(Ordering::Greater));
    assert!(eq(Int::<-1>, -1i32));
    assert!(!eq(-1i32, i32::MAX));
    assert!(lt(i32::MIN, Int::<0>));
    assert!(le(Int::<I32_MIN>, i32::MIN));
    assert!(gt(i32::MAX, Int::<0>));
    assert!(ge(Int::<I32_MAX>, i32::MAX));
    assert_eq!(cmp(i32::MAX, -1i32), Ordering::Greater);
}

#[test]
fn test_static_predicates_cover_decided_and_undecided_cases() {
    const {
        // Three-valued combinators
        assert!(static_true(Some(true)));
        assert!(!static_true(Some(false)));
        assert!(!static_true(None));

        assert!(static_false(Some(false)));
        assert!(!static_false(Some(true)));
        assert!(!static_false(None));

        assert!(matches!(static_not(Some(true)), Some(false)));
        assert!(matches!(static_not(Some(false)), Some(true)));
        assert!(static_not(None).is_none());

        assert!(matches!(static_and(Some(true), Some(true)), Some(true)));
        assert!(matches!(static_and(Some(true), Some(false)), Some(false)));
        assert!(matches!(static_and(Some(false), Some(true)), Some(false)));
        assert!(matches!(static_and(Some(false), Some(false)), Some(false)));
        assert!(matches!(static_and(Some(false), None), Some(false)));
        assert!(matches!(static_and(None, Some(false)), Some(false)));
        assert!(static_and(Some(true), None).is_none());
        assert!(static_and(None, Some(true)).is_none());
        assert!(static_and(None, None).is_none());

        assert!(matches!(static_or(Some(true), Some(true)), Some(true)));
        assert!(matches!(static_or(Some(true), Some(false)), Some(true)));
        assert!(matches!(static_or(Some(false), Some(true)), Some(true)));
        assert!(matches!(static_or(Some(true), None), Some(true)));
        assert!(matches!(static_or(None, Some(true)), Some(true)));
        assert!(matches!(static_or(Some(false), Some(false)), Some(false)));
        assert!(static_or(Some(false), None).is_none());
        assert!(static_or(None, Some(false)).is_none());
        assert!(static_or(None, None).is_none());

        // Type-level predicates
        assert!(matches!(static_eq_of::<Int<5>, Int<5>>(), Some(true)));
        assert!(matches!(static_eq_of::<Int<-1>, Int<5>>(), Some(false)));
        assert!(static_eq_of::<Int<5>, int>().is_none());

        assert!(matches!(static_lt_of::<Int<-1>, Int<0>>(), Some(true)));
        assert!(matches!(static_lt_of::<Int<5>, Int<5>>(), Some(false)));
        assert!(static_lt_of::<int, Int<5>>().is_none());

        assert!(matches!(static_le_of::<Int<5>, Int<5>>(), Some(true)));
        assert!(matches!(static_le_of::<Int<6>, Int<5>>(), Some(false)));
        assert!(static_le_of::<Int<5>, int>().is_none());

        assert!(matches!(
            static_cmp_of::<Int<2>, Int<5>>(),
            Some(Ordering::Less)
        ));
        assert!(matches!(
            static_cmp_of::<Int<5>, Int<5>>(),
            Some(Ordering::Equal)
        ));
        assert!(matches!(
            static_cmp_of::<Int<6>, Int<5>>(),
            Some(Ordering::Greater)
        ));
        assert!(static_cmp_of::<Int<5>, int>().is_none());

        // Value-level static predicates
        assert!(static_true(static_eq(Int::<4>, Int::<4>)));
        assert!(static_false(static_eq(Int::<4>, Int::<5>)));
        assert!(static_true(static_ne(Int::<4>, Int::<5>)));
        assert!(static_true(static_lt(Int::<-1>, Int::<0>)));
        assert!(static_false(static_lt(Int::<5>, Int::<0>)));
        assert!(static_true(static_le(Int::<0>, Int::<5>)));
        assert!(static_true(static_gt(Int::<5>, Int::<0>)));
        assert!(static_true(static_ge(Int::<5>, Int::<5>)));
        assert!(matches!(
            static_cmp(Int::<-1>, Int::<0>),
            Some(Ordering::Less)
        ));

        assert!(static_eq(Int::<5>, 5i32).is_none());
        assert!(static_ne(5i32, Int::<5>).is_none());
        assert!(static_lt(5i32, Int::<6>).is_none());
        assert!(static_le(Int::<5>, 5i32).is_none());
        assert!(static_gt(6i32, Int::<5>).is_none());
        assert!(static_ge(Int::<5>, 5i32).is_none());
        assert!(static_cmp(Int::<5>, 5i32).is_none());

        // Const bool helpers
        assert!(STATIC_TRUE_EQ::<Int<5>, Int<5>>);
        assert!(!STATIC_TRUE_EQ::<Int<-1>, Int<5>>);
        assert!(!STATIC_TRUE_EQ::<Int<5>, int>);
        assert!(!STATIC_TRUE_EQ::<int, int>);

        assert!(STATIC_TRUE_LT::<Int<-1>, Int<0>>);
        assert!(!STATIC_TRUE_LT::<Int<0>, Int<0>>);
        assert!(!STATIC_TRUE_LT::<Int<-1>, int>);
    }

    let dyn_x = 0i32;
    assert!(!static_true(static_ne(dyn_x, Int::<0>)));
    assert!(!static_true(static_eq(dyn_x, Int::<0>)));
    assert!(!static_true(static_and(Some(true), None)));
    assert!(static_true(static_or(Some(true), None)));
}

#[test]
fn test_min_and_max() {
    const M_MIN: Int<-2> = min(Int::<-2>, Int::<5>);
    const M_MAX: Int<9> = max(Int::<3>, Int::<9>);
    const M_EQ_MIN: Int<4> = min(Int::<4>, Int::<4>);
    const M_EQ_MAX: Int<4> = max(Int::<4>, Int::<4>);
    const M_EXT_MIN: Int<I32_MIN> = min(Int::<I32_MAX>, Int::<I32_MIN>);
    const M_EXT_MAX: Int<I32_MAX> = max(Int::<I32_MIN>, Int::<I32_MAX>);
    const M_DYN_MIN: int = min(Int::<10>, 3i32);
    const M_DYN_MAX: int = max(3i32, Int::<10>);
    assert_eq!(M_MIN, Int::<-2>);
    assert_eq!(M_MAX, Int::<9>);
    assert_eq!((M_EQ_MIN, M_EQ_MAX), (Int::<4>, Int::<4>));
    assert_eq!(M_EXT_MIN, Int::<I32_MIN>);
    assert_eq!(M_EXT_MAX, Int::<I32_MAX>);
    assert_eq!((M_DYN_MIN, M_DYN_MAX), (3, 10));

    let _: Min<Int<-2>, Int<5>> = Int::<-2>;
    let _: Max<Int<3>, Int<9>> = Int::<9>;
    let _: Min<Int<3>, int> = 2i32;
    let _: Max<int, Int<9>> = 9i32;

    assert_eq!(min(10i32, 20i32), 10i32);
    assert_eq!(max(10i32, 20i32), 20i32);
    assert_eq!(min(Int::<5>, -3i32), -3i32);
    assert_eq!(max(-3i32, Int::<-10>), -3i32);
}

#[test]
fn test_static_arithmetic_and_operators() {
    // Function API: Int × Int -> Int
    let _: Int<5> = add(Int::<2>, Int::<3>);
    let _: Int<-1> = sub(Int::<2>, Int::<3>);
    let _: Int<6> = mul(Int::<2>, Int::<3>);
    let _: Int<2> = div(Int::<7>, Int::<3>);
    let _: Int<1> = rem(Int::<7>, Int::<3>);

    // Negative numbers
    let _: Int<-6> = mul(Int::<2>, Int::<-3>);
    let _: Int<6> = mul(Int::<-2>, Int::<-3>);
    let _: Int<-2> = div(Int::<-7>, Int::<3>);
    let _: Int<-1> = rem(Int::<-7>, Int::<3>);

    // Operator syntax
    let a: Int<5> = Int::<2> + Int::<3>;
    let b: Int<-1> = Int::<2> - Int::<3>;
    let c: Int<6> = Int::<2> * Int::<3>;
    let d: Int<3> = Int::<7> / Int::<2>;
    let e: Int<1> = Int::<7> % Int::<3>;
    assert_eq!(
        (a, b, c, d, e),
        (Int::<5>, Int::<-1>, Int::<6>, Int::<3>, Int::<1>)
    );

    // Const evaluation in const context
    const C_ADD: Int<7> = add(Int::<3>, Int::<4>);
    const C_SUB: Int<-1> = sub(Int::<3>, Int::<4>);
    const C_MUL: Int<24> = mul(Int::<6>, Int::<4>);
    const C_DIV: Int<4> = div(Int::<18>, Int::<4>);
    const C_REM: Int<2> = rem(Int::<18>, Int::<4>);
    const C_EXPR: Int<23> = (Int::<2> + Int::<3>) * Int::<5> - Int::<4> / Int::<2>;
    assert_eq!(C_ADD, Int::<7>);
    assert_eq!(C_SUB, Int::<-1>);
    assert_eq!(C_MUL, Int::<24>);
    assert_eq!(C_DIV, Int::<4>);
    assert_eq!(C_REM, Int::<2>);
    assert_eq!(C_EXPR, Int::<23>);

    // Output type aliases
    let _: Sum<Int<2>, Int<3>> = Int::<5>;
    let _: Difference<Int<2>, Int<3>> = Int::<-1>;
    let _: Product<Int<2>, Int<3>> = Int::<6>;
    let _: Quotient<Int<7>, Int<3>> = Int::<2>;
    let _: Remainder<Int<7>, Int<3>> = Int::<1>;
    let _: CeilQuotient<Int<7>, Int<3>> = Int::<3>;
    let _: ArithOpResult<Int<2>, Int<3>, { ArithOp::Add }> = Int::<5>;

    let _: Sum<Int<2>, int> = 5i32;
    let _: Difference<int, Int<3>> = -1i32;
    let _: Product<Int<2>, int> = 6i32;
    let _: Quotient<int, Int<3>> = 2i32;
    let _: Remainder<int, int> = 1i32;
    let _: CeilQuotient<int, Int<3>> = 3i32;

    // Trait aliases in generic const context
    type ArithTuple<A, B> = (
        Sum<A, B>,
        Difference<A, B>,
        Product<A, B>,
        Quotient<A, B>,
        Remainder<A, B>,
        CeilQuotient<A, B>,
    );
    const fn generic_arith<A, B>(a: A, b: B) -> ArithTuple<A, B>
    where
        A: [const] Integer
            + [const] ArithAdd<A, B>
            + [const] ArithSub<A, B>
            + [const] ArithMul<A, B>
            + [const] ArithDiv<A, B>
            + [const] ArithRem<A, B>
            + [const] ArithCeilDiv<A, B>,
        B: [const] Integer,
    {
        (
            add(a, b),
            sub(a, b),
            mul(a, b),
            div(a, b),
            rem(a, b),
            ceil_div(a, b),
        )
    }
    const GEN_STATIC: (Int<10>, Int<4>, Int<21>, Int<2>, Int<1>, Int<3>) =
        generic_arith(Int::<7>, Int::<3>);
    const GEN_DYN: (int, int, int, int, int, int) = generic_arith(7i32, Int::<3>);
    assert_eq!(
        GEN_STATIC,
        (Int::<10>, Int::<4>, Int::<21>, Int::<2>, Int::<1>, Int::<3>)
    );
    assert_eq!(GEN_DYN, (10, 4, 21, 2, 1, 3));
}

#[test]
fn test_dynamic_and_mixed_arithmetic_and_operators() {
    // Const evaluation with at least one dynamic operand yields primitive int
    const ADD: int = add(Int::<2>, 3i32);
    const SUB: int = sub(2i32, Int::<3>);
    const MUL: int = mul(Int::<2>, 3i32);
    const DIV: int = div(-7i32, Int::<3>);
    const REM: int = rem(Int::<-7>, 3i32);
    assert_eq!((ADD, SUB, MUL, DIV, REM), (5, -1, 6, -2, -1));

    const OP_EXPR: int = (Int::<2> + 3i32) * Int::<5> - 4i32 / Int::<2>;
    assert_eq!(OP_EXPR, 23);

    // Function calls across (int, int), (Int, int), (int, Int)
    assert_eq!(add(10i32, 20i32), 30i32);
    assert_eq!(add(Int::<1>, 2i32), 3i32);
    assert_eq!(add(3i32, Int::<-5>), -2i32);

    assert_eq!(sub(-5i32, 1i32), -6i32);
    assert_eq!(sub(0i32, Int::<1>), -1i32);
    assert_eq!(sub(Int::<10>, 4i32), 6i32);

    assert_eq!(mul(-5i32, 2i32), -10i32);
    assert_eq!(mul(Int::<2>, 5i32), 10i32);
    assert_eq!(mul(-2i32, Int::<4>), -8i32);

    assert_eq!(div(15i32, Int::<4>), 3i32);
    assert_eq!(div(Int::<15>, 4i32), 3i32);
    assert_eq!(div(-12i32, Int::<4>), -3i32);

    assert_eq!(rem(15i32, Int::<4>), 3i32);
    assert_eq!(rem(Int::<15>, 4i32), 3i32);
    assert_eq!(rem(-13i32, Int::<4>), -1i32);

    // Operator calls (both sides)
    assert_eq!(Int::<10> + 5i32, 15i32);
    assert_eq!(5i32 + Int::<10>, 15i32);
    assert_eq!(Int::<10> - 5i32, 5i32);
    assert_eq!(5i32 - Int::<10>, -5i32);
    assert_eq!(Int::<10> * 5i32, 50i32);
    assert_eq!(5i32 * Int::<10>, 50i32);
    assert_eq!(Int::<10> / 5i32, 2i32);
    assert_eq!(12i32 / Int::<5>, 2i32);
    assert_eq!(Int::<10> % 4i32, 2i32);
    assert_eq!(12i32 % Int::<5>, 2i32);
}

#[test]
fn test_arithmetic_boundaries_and_panics() {
    // Valid extremes
    assert_eq!(add(i32::MAX, 0i32), i32::MAX);
    assert_eq!(add(i32::MIN, 0i32), i32::MIN);
    assert_eq!(sub(i32::MIN, 0i32), i32::MIN);
    assert_eq!(mul(i32::MIN, 1i32), i32::MIN);
    assert_eq!(div(i32::MIN, 1i32), i32::MIN);
    assert_eq!(rem(i32::MIN, 1i32), 0i32);

    // Overflows and division by zero panic in both debug and release builds
    assert!(catch_unwind(|| add(i32::MAX, 1i32)).is_err());
    assert!(catch_unwind(|| add(i32::MAX, Int::<1>)).is_err());
    assert!(catch_unwind(|| add(i32::MIN, -1i32)).is_err());
    assert!(catch_unwind(|| i32::MAX + Int::<1>).is_err());

    assert!(catch_unwind(|| sub(i32::MIN, 1i32)).is_err());
    assert!(catch_unwind(|| sub(i32::MAX, -1i32)).is_err());

    assert!(catch_unwind(|| mul(i32::MAX, 2i32)).is_err());
    assert!(catch_unwind(|| mul(i32::MIN, -1i32)).is_err());

    assert!(catch_unwind(|| div(7i32, 0i32)).is_err());
    assert!(catch_unwind(|| div(Int::<7>, 0i32)).is_err());
    assert!(catch_unwind(|| div(i32::MIN, -1i32)).is_err());
    assert!(catch_unwind(|| div(i32::MIN, Int::<-1>)).is_err());
    assert!(catch_unwind(|| Int::<10> / 0i32).is_err());
    assert!(catch_unwind(|| Int::<I32_MIN> / -1i32).is_err());

    assert!(catch_unwind(|| rem(7i32, 0i32)).is_err());
    assert!(catch_unwind(|| rem(Int::<7>, 0i32)).is_err());
    assert!(catch_unwind(|| rem(i32::MIN, -1i32)).is_err());
    assert!(catch_unwind(|| Int::<10> % 0i32).is_err());
    assert!(catch_unwind(|| i32::MIN % Int::<-1>).is_err());
}

#[test]
fn test_unary_neg_and_abs() {
    const C_NEG_POS: Int<-5> = -Int::<5>;
    const C_NEG_NEG: Int<7> = -Int::<-7>;
    const C_NEG_ZERO: Int<0> = -Int::<0>;
    const C_NEG_EXT: Int<I32_MAX> = -Int::<{ -I32_MAX }>;
    const C_NEG_DYN: int = neg(12i32);
    assert_eq!(C_NEG_POS, Int::<-5>);
    assert_eq!(C_NEG_NEG, Int::<7>);
    assert_eq!(C_NEG_ZERO, Int::<0>);
    assert_eq!(C_NEG_EXT, Int::<I32_MAX>);
    assert_eq!(C_NEG_DYN, -12);

    assert_eq!(Int::<-9>.neg(), Int::<9>);
    assert_eq!(neg(Int::<4>), Int::<-4>);
    assert_eq!(neg(0i32), 0i32);
    assert_eq!(neg(-i32::MAX), i32::MAX);
    assert!(catch_unwind(|| neg(i32::MIN)).is_err());

    const C_ABS_NEG: Int<12> = abs(Int::<-12>);
    const C_ABS_POS: Int<12> = abs(Int::<12>);
    const C_ABS_ZERO: Int<0> = abs(Int::<0>);
    const C_ABS_EXT: Int<I32_MAX> = abs(Int::<{ -I32_MAX }>);
    const C_ABS_DYN: int = abs(-42i32);
    assert_eq!(C_ABS_NEG, Int::<12>);
    assert_eq!(C_ABS_POS, Int::<12>);
    assert_eq!(C_ABS_ZERO, Int::<0>);
    assert_eq!(C_ABS_EXT, Int::<I32_MAX>);
    assert_eq!(C_ABS_DYN, 42);

    assert_eq!(abs(0i32), 0i32);
    assert_eq!(abs(-i32::MAX), i32::MAX);
    assert!(catch_unwind(|| abs(i32::MIN)).is_err());
}

#[test]
fn test_ceil_div_and_floor_divmod() {
    // Ceiling division across all four sign quadrants and boundaries
    const C_PP: Int<3> = ceil_div(Int::<7>, Int::<3>);
    const C_NP: Int<-2> = ceil_div(Int::<-7>, Int::<3>);
    const C_PN: Int<-2> = ceil_div(Int::<7>, Int::<-3>);
    const C_NN: Int<3> = ceil_div(Int::<-7>, Int::<-3>);
    const C_ZERO: Int<0> = ceil_div(Int::<0>, Int::<5>);
    const C_ZERO_NEG: Int<0> = ceil_div(Int::<0>, Int::<-5>);
    const C_EXACT: Int<-3> = ceil_div(Int::<-9>, Int::<3>);
    const C_HALF_MAX: Int<{ (I32_MAX / 2) + 1 }> = ceil_div(Int::<I32_MAX>, Int::<2>);
    assert_eq!(C_PP, Int::<3>);
    assert_eq!(C_NP, Int::<-2>);
    assert_eq!(C_PN, Int::<-2>);
    assert_eq!(C_NN, Int::<3>);
    assert_eq!(
        (C_ZERO, C_ZERO_NEG, C_EXACT),
        (Int::<0>, Int::<0>, Int::<-3>)
    );
    assert_eq!(C_HALF_MAX, Int::<1_073_741_824>);

    assert_eq!(ceil_div(7i32, -3i32), -2i32);
    assert_eq!(ceil_div(i32::MAX, 2i32), 1_073_741_824);
    assert_eq!(ceil_div(i32::MIN, 2i32), -(1 << 30));
    assert_eq!(ceil_div(i32::MIN, 3i32), -715_827_882);
    assert_eq!(ceil_div(i32::MIN, -2i32), 1 << 30);
    assert_eq!(ceil_div(i32::MIN, -3i32), 715_827_883);
    assert!(catch_unwind(|| ceil_div(5i32, 0i32)).is_err());
    assert!(catch_unwind(|| ceil_div(Int::<5>, 0i32)).is_err());
    assert!(catch_unwind(|| ceil_div(i32::MIN, -1i32)).is_err());

    // Floor division/remainder across all four sign quadrants
    const F_PP: (Int<2>, Int<1>) = floor_divmod(Int::<7>, Int::<3>);
    const F_NP: (Int<-3>, Int<2>) = floor_divmod(Int::<-7>, Int::<3>);
    const F_PN: (Int<-3>, Int<-2>) = floor_divmod(Int::<7>, Int::<-3>);
    const F_NN: (Int<2>, Int<-1>) = floor_divmod(Int::<-7>, Int::<-3>);
    const F_EXACT: (Int<-3>, Int<0>) = floor_divmod(Int::<-9>, Int::<3>);
    const FQ: Int<-3> = floor_div(Int::<-7>, Int::<3>);
    const FR: Int<2> = floor_rem(Int::<-7>, Int::<3>);
    assert_eq!(F_PP, (Int::<2>, Int::<1>));
    assert_eq!(F_NP, (Int::<-3>, Int::<2>));
    assert_eq!(F_PN, (Int::<-3>, Int::<-2>));
    assert_eq!(F_NN, (Int::<2>, Int::<-1>));
    assert_eq!(F_EXACT, (Int::<-3>, Int::<0>));
    assert_eq!((FQ, FR), (Int::<-3>, Int::<2>));

    let _: (
        FloorQuotient<Int<-7>, Int<3>>,
        FloorRemainder<Int<-7>, Int<3>>,
    ) = (FQ, FR);
    let _: (FloorQuotient<int, Int<3>>, FloorRemainder<int, Int<3>>) = (-3, 2);

    assert_eq!(floor_div(-1i32, 4i32), -1i32);
    assert_eq!(floor_rem(-1i32, 4i32), 3i32);
    assert_eq!(floor_divmod(-7i32, Int::<-3>), (2i32, -1i32));
    assert_eq!(floor_divmod(Int::<7>, -3i32), (-3i32, -2i32));
    assert_eq!(floor_divmod(i32::MIN, 1i32), (i32::MIN, 0));
    assert_eq!(floor_divmod(i32::MIN, 2i32), (-(1 << 30), 0));
    assert_eq!(floor_divmod(i32::MIN, 3i32), (-715_827_883, 1));
    assert_eq!(floor_divmod(i32::MIN, -3i32), (715_827_882, -2));
    assert_eq!(floor_divmod(i32::MIN, i32::MAX), (-2, i32::MAX - 1));

    assert!(catch_unwind(|| floor_divmod(5i32, 0i32)).is_err());
    assert!(catch_unwind(|| floor_div(Int::<5>, 0i32)).is_err());
    assert!(catch_unwind(|| floor_rem(Int::<5>, 0i32)).is_err());
    assert!(catch_unwind(|| floor_divmod(i32::MIN, -1i32)).is_err());
    assert!(catch_unwind(|| floor_rem(i32::MIN, Int::<-1>)).is_err());
}

#[test]
fn test_gcd_and_lcm() {
    // Compile-time static GCD and LCM (always non-negative across all sign combinations)
    const G_PP: Int<6> = gcd(Int::<12>, Int::<18>);
    const G_NP: Int<6> = gcd(Int::<-12>, Int::<18>);
    const G_PN: Int<6> = gcd(Int::<12>, Int::<-18>);
    const G_NN: Int<6> = gcd(Int::<-12>, Int::<-18>);
    const G_ZERO_ZERO: Int<0> = gcd(Int::<0>, Int::<0>);
    const G_NEG_ZERO: Int<12> = gcd(Int::<-12>, Int::<0>);
    const G_ZERO_NEG: Int<12> = gcd(Int::<0>, Int::<-12>);
    const G_MIN_2: Int<2> = gcd(Int::<I32_MIN>, Int::<2>);
    const G_MIN_NEG1: Int<1> = gcd(Int::<I32_MIN>, Int::<-1>);
    assert_eq!(
        (
            G_PP,
            G_NP,
            G_PN,
            G_NN,
            G_ZERO_ZERO,
            G_NEG_ZERO,
            G_ZERO_NEG,
            G_MIN_2,
            G_MIN_NEG1
        ),
        (
            Int::<6>, Int::<6>, Int::<6>, Int::<6>, Int::<0>, Int::<12>, Int::<12>, Int::<2>,
            Int::<1>,
        )
    );

    const L_PP: Int<12> = lcm(Int::<4>, Int::<6>);
    const L_NP: Int<12> = lcm(Int::<-4>, Int::<6>);
    const L_PN: Int<12> = lcm(Int::<4>, Int::<-6>);
    const L_NN: Int<12> = lcm(Int::<-4>, Int::<-6>);
    const L_ZERO_L: Int<0> = lcm(Int::<0>, Int::<-9>);
    const L_ZERO_R: Int<0> = lcm(Int::<I32_MIN>, Int::<0>);
    const L_MAX: Int<I32_MAX> = lcm(Int::<I32_MAX>, Int::<I32_MAX>);
    assert_eq!(
        (L_PP, L_NP, L_PN, L_NN, L_ZERO_L, L_ZERO_R, L_MAX),
        (
            Int::<12>,
            Int::<12>,
            Int::<12>,
            Int::<12>,
            Int::<0>,
            Int::<0>,
            Int::<I32_MAX>,
        )
    );

    // Type aliases and trait alias
    let _: Gcd<Int<-12>, Int<18>> = Int::<6>;
    let _: Lcm<Int<4>, Int<-6>> = Int::<12>;
    let _: Gcd<int, Int<18>> = 6i32;
    let _: Lcm<Int<4>, int> = 12i32;

    const fn generic_nt<A, B>(a: A, b: B) -> (Gcd<A, B>, Lcm<A, B>)
    where
        A: [const] Integer + [const] ArithGcd<A, B> + [const] ArithLcm<A, B>,
        B: [const] Integer,
    {
        (gcd(a, b), lcm(a, b))
    }
    const NT_STATIC: (Int<6>, Int<36>) = generic_nt(Int::<-12>, Int::<18>);
    const NT_DYN: (int, int) = generic_nt(-12i32, Int::<18>);
    assert_eq!(NT_STATIC, (Int::<6>, Int::<36>));
    assert_eq!(NT_DYN, (6, 36));

    // Dynamic & mixed GCD / LCM
    assert_eq!(gcd(0i32, 0i32), 0i32);
    assert_eq!(gcd(12i32, -18i32), 6i32);
    assert_eq!(gcd(-12i32, Int::<-18>), 6i32);
    assert_eq!(gcd(i32::MIN, 1i32), 1i32);
    assert_eq!(gcd(i32::MIN, -1i32), 1i32);
    assert_eq!(gcd(i32::MIN, 2i32), 2i32);
    assert_eq!(gcd(i32::MAX, i32::MAX), i32::MAX);

    assert_eq!(lcm(4i32, -6i32), 12i32);
    assert_eq!(lcm(-4i32, Int::<6>), 12i32);
    assert_eq!(lcm(Int::<-4>, -6i32), 12i32);
    assert_eq!(lcm(0i32, i32::MIN), 0i32);
    assert_eq!(lcm(i32::MIN, 0i32), 0i32);
    assert_eq!(lcm(65536i32, 32768i32), 65536i32);

    // Overflows
    assert!(catch_unwind(|| gcd(i32::MIN, 0i32)).is_err());
    assert!(catch_unwind(|| gcd(i32::MIN, i32::MIN)).is_err());
    assert!(catch_unwind(|| lcm(65536i32, 65537i32)).is_err());
    assert!(catch_unwind(|| lcm(65536i32, 32769i32)).is_err());
    assert!(catch_unwind(|| lcm(i32::MIN, 1i32)).is_err());
    assert!(catch_unwind(|| lcm(i32::MIN, 2i32)).is_err());
    assert!(catch_unwind(|| lcm(i32::MIN, 3i32)).is_err());
    assert!(catch_unwind(|| gcd(i32::MIN, Int::<0>)).is_err());
    assert!(catch_unwind(|| lcm(i32::MIN, Int::<1>)).is_err());
}

#[test]
fn test_static_absorbing_and_checked_div_dispatch() {
    // Unconditional static absorbing rules in const and runtime contexts
    const MUL_L: Int<0> = mul(Int::<0>, I32_MIN);
    const MUL_R: Int<0> = mul(I32_MAX, Int::<0>);
    const OP_MUL_L: Int<0> = Int::<0> * I32_MIN;
    const OP_MUL_R: Int<0> = I32_MAX * Int::<0>;
    const REM_ONE: Int<0> = rem(I32_MIN, Int::<1>);
    const OP_REM_ONE: Int<0> = I32_MIN % Int::<1>;
    const FDIVMOD_ONE: (int, Int<0>) = floor_divmod(I32_MIN, Int::<1>);
    const FREM_ONE: Int<0> = floor_rem(I32_MIN, Int::<1>);
    const GCD_ONE_L: Int<1> = gcd(Int::<1>, I32_MIN);
    const GCD_ONE_R: Int<1> = gcd(0i32, Int::<1>);
    const GCD_NEG1_L: Int<1> = gcd(Int::<-1>, I32_MIN);
    const GCD_NEG1_R: Int<1> = gcd(I32_MIN, Int::<-1>);
    const LCM_ZERO_L: Int<0> = lcm(Int::<0>, I32_MIN);
    const LCM_ZERO_R: Int<0> = lcm(I32_MIN, Int::<0>);

    assert_eq!(
        (
            MUL_L, MUL_R, OP_MUL_L, OP_MUL_R, REM_ONE, OP_REM_ONE, FREM_ONE
        ),
        (
            Int::<0>, Int::<0>, Int::<0>, Int::<0>, Int::<0>, Int::<0>, Int::<0>,
        )
    );
    assert_eq!(FDIVMOD_ONE, (i32::MIN, Int::<0>));
    assert!(static_true(static_eq(FDIVMOD_ONE.1, Int::<0>)));
    assert_eq!(
        (GCD_ONE_L, GCD_ONE_R, GCD_NEG1_L, GCD_NEG1_R),
        (Int::<1>, Int::<1>, Int::<1>, Int::<1>)
    );
    assert_eq!((LCM_ZERO_L, LCM_ZERO_R), (Int::<0>, Int::<0>));

    // StaticCheckedDiv: Int<0> dividend with dynamic non-zero divisor preserves Int<0>
    const DIV_ZERO: Int<0> = div(Int::<0>, -1i32);
    const OP_DIV_ZERO: Int<0> = Int::<0> / I32_MIN;
    const REM_ZERO: Int<0> = rem(Int::<0>, -1i32);
    const OP_REM_ZERO: Int<0> = Int::<0> % I32_MIN;
    const CEIL_ZERO: Int<0> = ceil_div(Int::<0>, -3i32);
    const FDIV_ZERO: Int<0> = floor_div(Int::<0>, -3i32);
    const FREM_ZERO: Int<0> = floor_rem(Int::<0>, -3i32);
    const FDIVMOD_ZERO: (Int<0>, Int<0>) = floor_divmod(Int::<0>, I32_MIN);
    assert_eq!(
        (
            DIV_ZERO,
            OP_DIV_ZERO,
            REM_ZERO,
            OP_REM_ZERO,
            CEIL_ZERO,
            FDIV_ZERO,
            FREM_ZERO,
        ),
        (
            Int::<0>, Int::<0>, Int::<0>, Int::<0>, Int::<0>, Int::<0>, Int::<0>,
        )
    );
    assert_eq!(FDIVMOD_ZERO, (Int::<0>, Int::<0>));

    // StaticCheckedDiv non-const runtime paths with dynamic non-zero divisors
    let d = std::hint::black_box(-3i32);
    let d_min = std::hint::black_box(i32::MIN);
    let rt_div_zero: Int<0> = div(Int::<0>, d);
    let rt_op_div_zero: Int<0> = Int::<0> / d_min;
    let rt_rem_zero: Int<0> = rem(Int::<0>, d);
    let rt_op_rem_zero: Int<0> = Int::<0> % d_min;
    let rt_ceil_zero: Int<0> = ceil_div(Int::<0>, d);
    let rt_fdiv_zero: Int<0> = floor_div(Int::<0>, d);
    let rt_frem_zero: Int<0> = floor_rem(Int::<0>, d);
    let rt_fdivmod_zero: (Int<0>, Int<0>) = floor_divmod(Int::<0>, d_min);
    assert_eq!(
        (
            rt_div_zero,
            rt_op_div_zero,
            rt_rem_zero,
            rt_op_rem_zero,
            rt_ceil_zero,
            rt_fdiv_zero,
            rt_frem_zero,
        ),
        (
            Int::<0>, Int::<0>, Int::<0>, Int::<0>, Int::<0>, Int::<0>, Int::<0>,
        )
    );
    assert_eq!(rt_fdivmod_zero, (Int::<0>, Int::<0>));

    // StaticCheckedDiv still panics at runtime when the dynamic divisor is 0
    assert!(catch_unwind(|| div(Int::<0>, 0i32)).is_err());
    assert!(catch_unwind(|| Int::<0> / 0i32).is_err());
    assert!(catch_unwind(|| rem(Int::<0>, 0i32)).is_err());
    assert!(catch_unwind(|| Int::<0> % 0i32).is_err());
    assert!(catch_unwind(|| ceil_div(Int::<0>, 0i32)).is_err());
    assert!(catch_unwind(|| floor_div(Int::<0>, 0i32)).is_err());
    assert!(catch_unwind(|| floor_rem(Int::<0>, 0i32)).is_err());
    assert!(catch_unwind(|| floor_divmod(Int::<0>, 0i32)).is_err());

    // Verify ArithFloorDiv / ArithFloorRem / ArithFloorDivMod / ArithGcd / ArithLcm bounds
    const fn generic_floor_div<A, B>(a: A, b: B) -> FloorQuotient<A, B>
    where
        A: [const] Integer + [const] ArithFloorDiv<A, B>,
        B: [const] Integer,
    {
        floor_div(a, b)
    }
    const fn generic_floor_rem<A, B>(a: A, b: B) -> FloorRemainder<A, B>
    where
        A: [const] Integer + [const] ArithFloorRem<A, B>,
        B: [const] Integer,
    {
        floor_rem(a, b)
    }
    const fn generic_floor<A, B>(a: A, b: B) -> (FloorQuotient<A, B>, FloorRemainder<A, B>)
    where
        A: [const] Integer + [const] ArithFloorDivMod<A, B>,
        B: [const] Integer,
    {
        let (q, r) = floor_divmod(a, b);
        let _ = (floor_div(a, b), floor_rem(a, b));
        (q, r)
    }
    const fn generic_gcd<A, B>(a: A, b: B) -> Gcd<A, B>
    where
        A: [const] Integer + [const] ArithGcd<A, B>,
        B: [const] Integer,
    {
        gcd(a, b)
    }
    const fn generic_lcm<A, B>(a: A, b: B) -> Lcm<A, B>
    where
        A: [const] Integer + [const] ArithLcm<A, B>,
        B: [const] Integer,
    {
        lcm(a, b)
    }
    const GFD: int = generic_floor_div(-9i32, Int::<4>);
    const GFR: Int<0> = generic_floor_rem(-9i32, Int::<1>);
    const GF: (int, Int<0>) = generic_floor(-9i32, Int::<1>);
    const GG: Int<1> = generic_gcd(-9i32, Int::<1>);
    const GL: Int<0> = generic_lcm(Int::<0>, -9i32);
    assert_eq!(GFD, -3);
    assert_eq!(GFR, Int::<0>);
    assert_eq!(GF, (-9, Int::<0>));
    assert_eq!(GG, Int::<1>);
    assert_eq!(GL, Int::<0>);
}

#[test]
fn test_bitwise_and_signed_shift_ops() {
    // Static bitwise operations and operators
    const AND_SS: Int<0b1000> = Int::<0b1010> & Int::<0b1100>;
    const OR_SS: Int<0b1110> = Int::<0b1010> | Int::<0b1100>;
    const XOR_SS: Int<0b0110> = Int::<0b1010> ^ Int::<0b1100>;
    const SHL_POS: Int<0b100000> = shiftl(Int::<0b1000>, Int::<2>);
    const SHL_NEG: Int<0b10> = shiftl(Int::<0b1000>, Int::<-2>);
    const SHR_POS: Int<0b10> = shiftr(Int::<0b1000>, Int::<2>);
    const SHR_NEG: Int<0b100000> = shiftr(Int::<0b1000>, Int::<-2>);
    const SHL_NEG_ONE_31: Int<I32_MIN> = shiftl(Int::<-1>, Int::<31>);
    const SHL_SAT_ZERO: Int<0> = shiftl(Int::<1>, Int::<I32_MIN>);
    const SHR_SAT_NEG: Int<-1> = shiftr(Int::<-5>, Int::<100>);
    const SHR_SAT_POS: Int<0> = shiftr(Int::<5>, Int::<100>);

    assert_eq!(AND_SS, Int::<0b1000>);
    assert_eq!(OR_SS, Int::<0b1110>);
    assert_eq!(XOR_SS, Int::<0b0110>);
    assert_eq!(SHL_POS, Int::<0b100000>);
    assert_eq!(SHL_NEG, Int::<0b10>);
    assert_eq!(SHR_POS, Int::<0b10>);
    assert_eq!(SHR_NEG, Int::<0b100000>);
    assert_eq!(SHL_NEG_ONE_31, Int::<I32_MIN>);
    assert_eq!(SHL_SAT_ZERO, Int::<0>);
    assert_eq!(SHR_SAT_NEG, Int::<-1>);
    assert_eq!(SHR_SAT_POS, Int::<0>);

    // Absorbing identities preserve static types with dynamic operands
    let dyn_v = 0x1234_5678i32;
    let z_and_l: Int<0> = Int::<0> & dyn_v;
    let z_and_r: Int<0> = dyn_v & Int::<0>;
    let m1_or_l: Int<-1> = Int::<-1> | dyn_v;
    let m1_or_r: Int<-1> = dyn_v | Int::<-1>;
    let z_shl: Int<0> = shiftl(Int::<0>, dyn_v);
    let z_shr: Int<0> = shiftr(Int::<0>, dyn_v);
    assert_eq!((z_and_l, z_and_r), (Int::<0>, Int::<0>));
    assert_eq!((m1_or_l, m1_or_r), (Int::<-1>, Int::<-1>));
    assert_eq!((z_shl, z_shr), (Int::<0>, Int::<0>));

    // Mixed and dynamic bitwise / signed-shift evaluation
    assert_eq!(bitand(0b1010i32, Int::<0b1100>), 0b1000);
    assert_eq!(bitor(0b1010i32, Int::<0b1100>), 0b1110);
    assert_eq!(bitxor(0b1010i32, Int::<0b1100>), 0b0110);
    assert_eq!(0b1010i32 ^ Int::<0b1100>, 0b0110);
    assert_eq!(shiftl(0b1000i32, 2i32), 0b100000);
    assert_eq!(shiftl(0b1000i32, -2i32), 0b10);
    assert_eq!(shiftr(0b1000i32, 2i32), 0b10);
    assert_eq!(shiftr(0b1000i32, -2i32), 0b100000);
    assert_eq!(shiftl(1i32, 30i32), 1 << 30);
    assert_eq!(shiftl(-1i32, 31i32), I32_MIN);
    assert_eq!(shiftr(-7i32, 1i32), -4);
    assert_eq!(shiftl(1i32, I32_MIN), 0);

    // Runtime left-shift overflows panic
    assert!(catch_unwind(|| shiftl(1i32, 31i32)).is_err());
    assert!(catch_unwind(|| shiftl(2i32, 30i32)).is_err());
    assert!(catch_unwind(|| shiftl(-2i32, 31i32)).is_err());
    assert!(catch_unwind(|| shiftl(1i32, 32i32)).is_err());
    assert!(catch_unwind(|| shiftr(1i32, -31i32)).is_err());
    assert!(catch_unwind(|| shiftr(1i32, I32_MIN)).is_err());
}
