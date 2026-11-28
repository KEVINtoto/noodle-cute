#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use std::ops::RangeFull;
use std::panic::{AssertUnwindSafe, catch_unwind};

use noodle_cute::algebra::{Composition, composition};
use noodle_cute::algorithm::{Im2ColOptions, copy, im2col_with, reference};
use noodle_cute::basis::{Basis, BasisCombination, BasisTerm};
use noodle_cute::f2::{F2, StaticF2, f2};
use noodle_cute::integer::Int;
use noodle_cute::layout::Layout;
use noodle_cute::stride::coshape;
use noodle_cute::swizzle::{StaticSwizzle, Swizzle};
use noodle_cute::tensor::{ImplicitAccessor, Tensor, make_tensor};
use noodle_cute::tuple::{Nil, Til};
use noodle_cute::{
    NoodleCuteError, NoodleCuteResult, Result, StaticValue, e, int_tuple, mode, static_value,
    swizzle, tuple,
};

fn panic_message(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected panic");
    err.downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| err.downcast_ref::<String>().cloned())
        .unwrap_or_default()
}

#[test]
fn static_value_instantiates_across_all_layers_in_const() {
    type StaticCoord = Til<Int<4>, Til<Til<Int<2>, Til<Int<3>, Nil>>, Nil>>;
    type StaticWild = Til<Int<0>, Til<RangeFull, Nil>>;
    type StaticF2Pair = Til<StaticF2<1>, Til<StaticF2<9>, Nil>>;
    type StaticBasisComb = BasisCombination<
        Til<BasisTerm<Int<2>, mode![0]>, Til<Basis<mode![1, 0]>, Nil>>,
    >;
    type StaticLayout = Layout<StaticCoord, Til<Int<1>, Til<Til<Int<4>, Til<Int<8>, Nil>>, Nil>>>;
    type ComposedOut = <Layout<Int<8>, Int<2>> as Composition<Layout<Int<3>, Int<2>>>>::Output;

    const INT_VAL: Int<42> = Int::<42>::VALUE;
    const INT_FN: Int<-7> = static_value::<Int<-7>>();
    const F2_VAL: StaticF2<5> = StaticF2::<5>::VALUE;
    const NIL_VAL: Nil = Nil::VALUE;
    const WILD_VAL: StaticWild = StaticWild::VALUE;
    const F2_TUP: StaticF2Pair = StaticF2Pair::VALUE;
    const BASIS_TERM: BasisTerm<Int<3>, mode![1]> = <BasisTerm<Int<3>, mode![1]>>::VALUE;
    const BASIS_ZERO: BasisCombination<Nil> = <BasisCombination<Nil>>::VALUE;
    const BASIS_COMB: StaticBasisComb = StaticBasisComb::VALUE;
    const LAY: StaticLayout = StaticLayout::VALUE;
    const COMPOSED: ComposedOut = static_value::<ComposedOut>();
    const SW: StaticSwizzle<3, 0, 3> = StaticSwizzle::<3, 0, 3>::VALUE;

    assert_eq!(INT_VAL, Int::<42>);
    assert_eq!(INT_FN, Int::<-7>);
    assert_eq!(F2_VAL, f2(Int::<5>));
    assert_eq!(NIL_VAL, Nil);
    assert_eq!(WILD_VAL, tuple![Int::<0>, ..]);
    assert_eq!(F2_TUP, tuple![f2(Int::<1>), f2(Int::<9>)]);
    assert_eq!(BASIS_TERM, Int::<3> * e![1]);
    assert_eq!(BASIS_ZERO, BasisCombination::zero());
    assert_eq!(BASIS_COMB, Int::<2> * e![0] + e![1, 0]);
    assert_eq!(
        LAY,
        Layout::new(
            int_tuple![Int::<4>, (Int::<2>, Int::<3>)],
            int_tuple![Int::<1>, (Int::<4>, Int::<8>)],
        )
    );
    assert_eq!(COMPOSED, Layout::new(Int::<3>, Int::<4>));
    assert_eq!(SW, swizzle!(Int::<3>, Int::<0>, Int::<3>));
    assert_eq!(size_of_val(&LAY), 0);
    assert_eq!(size_of_val(&SW), 0);
}

#[test]
fn static_value_works_with_f2_and_basis_layouts() {
    type F2Layout = Layout<
        Til<Int<8>, Til<Int<8>, Nil>>,
        Til<F2<Int<1>>, Til<F2<Int<9>>, Nil>>,
    >;
    type IdLayout = Layout<
        Til<Int<3>, Til<Int<4>, Nil>>,
        Til<Basis<mode![0]>, Til<Basis<mode![1]>, Nil>>,
    >;

    const F2L: F2Layout = static_value::<F2Layout>();
    const IDL: IdLayout = static_value::<IdLayout>();

    assert_eq!(F2L(1i32, 1i32), f2(8i32));
    assert_eq!(IDL(2i32, 3i32), 2i32 * e![0] + 3i32 * e![1]);
}

#[test]
fn noodle_cute_error_variants_have_consistent_messages_and_display() {
    let all = [
        NoodleCuteError::ArithmeticOverflow,
        NoodleCuteError::DivisionByZero,
        NoodleCuteError::SignedMinValueOverflow,
        NoodleCuteError::UnsignedToSignedOutOfRange,
        NoodleCuteError::InvalidShape,
        NoodleCuteError::StructureMismatch,
        NoodleCuteError::CoordinateOutOfBounds,
        NoodleCuteError::AxisOutOfBounds,
        NoodleCuteError::NonInjective,
        NoodleCuteError::NotSurjective,
        NoodleCuteError::NotDivisible,
        NoodleCuteError::AddressOutOfBounds,
        NoodleCuteError::SizeMismatch,
        NoodleCuteError::Unsupported,
    ];

    for err in all {
        assert!(!err.as_str().is_empty());
        assert_eq!(err.to_string(), err.as_str());
        let msg = panic_message(|| err.panic());
        assert_eq!(msg, err.as_str());
        let res_msg = panic_message(|| {
            let _: i32 = Result::<i32>::Err(err).v();
        });
        assert_eq!(res_msg, err.as_str());
    }
}

#[test]
fn dynamic_failures_surface_structured_error_categories() {
    // InvalidShape via F2::new / Swizzle::new / coshape
    let msg = panic_message(|| {
        let _ = F2::new(-1i32);
    });
    assert!(msg.contains(NoodleCuteError::InvalidShape.as_str()), "{msg}");

    let msg = panic_message(|| {
        let _ = Swizzle::new(3i32, 0i32, -2i32);
    });
    assert!(msg.contains(NoodleCuteError::InvalidShape.as_str()), "{msg}");

    let msg = panic_message(|| {
        let _ = coshape(int_tuple![2i32, 2i32], int_tuple![-1i32, 2i32]);
    });
    assert!(msg.contains(NoodleCuteError::InvalidShape.as_str()), "{msg}");
    assert!(msg.contains("negative"), "{msg}");

    // AddressOutOfBounds via make_tensor / accessor
    let msg = panic_message(|| {
        let buf = [0i32; 4];
        let _ = make_tensor(&buf, Layout::new(8i32, 1i32));
    });
    assert!(
        msg.contains(NoodleCuteError::AddressOutOfBounds.as_str()),
        "{msg}"
    );

    // NonInjective via copy write-after-write conflict
    let msg = panic_message(|| {
        let sh = int_tuple![Int::<2>, Int::<2>];
        let src = Tensor::new(
            ImplicitAccessor::new(0i32),
            Layout::new(sh, int_tuple![1i32, 2i32]),
        );
        let mut out = [0i32; 3];
        let mut dst = make_tensor(&mut out, Layout::new(sh, int_tuple![Int::<1>, Int::<1>]));
        copy(src, &mut dst);
    });
    assert!(msg.contains(NoodleCuteError::NonInjective.as_str()), "{msg}");
    assert!(msg.to_ascii_lowercase().contains("write-after-write"), "{msg}");

    // SizeMismatch via reference::copy (dynamic extents)
    let msg = panic_message(|| {
        let src_buf = [0i32; 4];
        let mut dst_buf = [0i32; 6];
        let src = make_tensor(&src_buf, 4i32);
        let mut dst = make_tensor(&mut dst_buf, 6i32);
        reference::copy(src, &mut dst);
    });
    assert!(msg.contains(NoodleCuteError::SizeMismatch.as_str()), "{msg}");

    // Unsupported via cross-mode integer carry in composition
    let msg = panic_message(|| {
        let a = Layout::new(int_tuple![Int::<3>, Int::<2>], int_tuple![Int::<1>, 8i32]);
        let b = Layout::new(
            int_tuple![Int::<2>, Int::<2>],
            int_tuple![Int::<1>, Int::<2>],
        );
        let _ = composition(a, b);
    });
    assert!(msg.contains(NoodleCuteError::Unsupported.as_str()), "{msg}");
    assert!(msg.contains("carry"), "{msg}");

    // Unsupported via non-separable im2col spatial window
    let msg = panic_message(|| {
        let buf = [0i32; 32];
        let act = make_tensor(
            &buf,
            Layout::new(
                int_tuple![Int::<1>, (Int::<2>, Int::<4>), Int::<1>],
                int_tuple![Int::<32>, (Int::<1>, Int::<8>), Int::<1>],
            ),
        );
        let opts = Im2ColOptions::new().shape_zpq(Int::<2>).lower_trs(0i32);
        let _ = im2col_with(act, Int::<2>, opts);
    });
    assert!(msg.contains(NoodleCuteError::Unsupported.as_str()), "{msg}");
    assert!(msg.contains("separable"), "{msg}");
}
