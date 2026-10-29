//! Contract tests for static and dynamic integers and the shared error type.
//!
//! These tests cover arithmetic result types, integer metadata, conversions,
//! storage size, and error messages.

use core::mem::{size_of, size_of_val};

use noodle_cute::int::{DynInt, Int, Integer, NON_NEGATIVE};
use noodle_cute::{Error, i};

#[test]
fn static_arithmetic_keeps_results_in_the_type() {
    // Type annotations verify both the result type and its value at compile time.
    let _: Int<5> = Int::<2> + i!(3);
    let _: Int<4> = Int::<6> - i!(2);
    let _: Int<42> = Int::<6> * i!(7);
    let _: Int<4> = Int::<12> / Int::<3>;
    // A nonzero remainder has the dividend's sign, as with native Rust integers.
    let _: Int<1> = Int::<13> % Int::<4>;
    let _: Int<-1> = Int::<-13> % Int::<4>;
    let _: Int<-3> = -Int::<3>;
    let _: Int<3> = -Int::<-3>;

    const {
        assert!(Int::<5>::VALUE == 5);
        assert!(Int::<5>.value() == 5);
    }
    assert!(isize::from(Int::<7>) == 7);
}

#[test]
fn mixed_static_dynamic_operands_promote_to_dynint() {
    // A dynamic operand makes the result dynamic on either side of an operation.
    let a: DynInt = Int::<2> + DynInt(4);
    assert_eq!(a, DynInt(6));
    let b: DynInt = DynInt(4) + Int::<2>;
    assert_eq!(b, DynInt(6));
    let c: DynInt = Int::<10> - DynInt(3);
    assert_eq!(c, DynInt(7));
    let d: DynInt = DynInt(10) * Int::<3>;
    assert_eq!(d, DynInt(30));
    let e: DynInt = Int::<10> / DynInt(3);
    assert_eq!(e, DynInt(3));
    let f: DynInt = Int::<10> % DynInt(3);
    assert_eq!(f, DynInt(1));
}

#[test]
fn native_isize_operands_also_promote_to_dynint() {
    // Native `isize` values follow the same promotion rule as `DynInt`.
    let a: DynInt = Int::<2> + 4isize;
    assert_eq!(a, DynInt(6));
    let b: DynInt = 4isize + Int::<2>;
    assert_eq!(b, DynInt(6));
    let c: DynInt = DynInt(4) + 2isize;
    assert_eq!(c, DynInt(6));
    let d: DynInt = 4isize + DynInt(2);
    assert_eq!(d, DynInt(6));
}

#[test]
fn negation_matches_for_static_and_dynamic_integers() {
    let _: Int<-9> = -Int::<9>;
    assert_eq!(-DynInt(9), DynInt(-9));
    assert_eq!(-DynInt(-9), DynInt(9));
}

#[test]
fn integer_trait_distinguishes_compile_time_constants_from_runtime_values() {
    // Equal values can differ in whether the type encodes their value.
    assert_eq!(<Int<7> as Integer>::STATIC, Some(7));
    assert_eq!(Int::<7>.value(), 7);
    assert_eq!(<DynInt as Integer>::STATIC, None);
    assert_eq!(DynInt(7).value(), 7);
    assert_eq!(<isize as Integer>::STATIC, None);
    assert_eq!(Integer::value(7isize), 7);
}

#[test]
fn dynint_conversions_and_default_round_trip() {
    let value: DynInt = 11isize.into();
    assert_eq!(value, DynInt(11));
    assert_eq!(value.value(), 11);
    assert_eq!(DynInt::default(), DynInt(0));
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn non_negative_const_matches_runtime_sign_check() {
    assert!(NON_NEGATIVE::<0>);
    assert!(NON_NEGATIVE::<5>);
    assert!(!NON_NEGATIVE::<-1>);
}

#[test]
fn static_leaves_are_zero_sized_while_dynamic_leaves_carry_a_machine_word() {
    // Encoding a value in the type requires no runtime storage.
    assert_eq!(size_of::<Int<42>>(), 0);
    assert_eq!(size_of::<DynInt>(), size_of::<isize>());
    assert_eq!(size_of_val(&Int::<42>), 0);
}

#[test]
fn equality_and_ordering_are_derived_structurally() {
    assert_eq!(Int::<3>, Int::<3>);
    assert_ne!(DynInt(3), DynInt(4));
    assert!(DynInt(3) < DynInt(4));
    let mut values = [DynInt(3), DynInt(1), DynInt(2)];
    values.sort();
    assert_eq!(values, [DynInt(1), DynInt(2), DynInt(3)]);
}

#[test]
fn error_variants_report_distinct_human_readable_messages() {
    use std::string::ToString;
    let cases = [
        (Error::InvalidShape, "shape extents must be nonnegative"),
        (
            Error::StructureMismatch,
            "shape, stride, or coordinate structures differ",
        ),
        (
            Error::CoordinateOutOfBounds,
            "coordinate is outside the layout domain",
        ),
        (Error::AxisOutOfBounds, "axis is outside the layout rank"),
        (Error::Overflow, "integer overflow"),
        (
            Error::NonInjective,
            "layout maps multiple coordinates to the same address",
        ),
        (
            Error::NotSurjective,
            "layout does not cover the requested codomain",
        ),
        (
            Error::NotDivisible,
            "layout or tile extent is not divisible",
        ),
        (
            Error::Unsupported,
            "operation is unsupported for this layout",
        ),
        (Error::AddressOutOfBounds, "address is outside storage"),
        (Error::SizeMismatch, "logical sizes differ"),
        (
            Error::CapacityExceeded,
            "layout exceeds fixed metadata capacity",
        ),
    ];
    for (error, message) in cases {
        assert_eq!(error.to_string(), message);
        // Copying an error preserves its identity.
        assert_eq!(error, error);
        let copied = error;
        assert_eq!(copied, error);
    }
}

#[test]
fn error_implements_core_error_trait() {
    fn assert_is_error<E: core::error::Error>(_: &E) {}
    assert_is_error(&Error::Overflow);
    let boxed: &dyn core::error::Error = &Error::NonInjective;
    assert_eq!(
        boxed.to_string(),
        "layout maps multiple coordinates to the same address"
    );
}
