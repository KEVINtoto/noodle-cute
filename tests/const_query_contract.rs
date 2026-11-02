#![feature(const_trait_impl)]

use noodle_cute::layout::{AffineLayoutMap, Layout, LayoutMap, Shape, StrideOf};
use noodle_cute::{DynInt, Error, Int, IntTuple, Integer, Nil, Result, StaticValue, shape, stride};

#[path = "diagnostics/runtime_item_pass.rs"]
mod runtime_item_pass;

#[test]
fn ordinary_downstream_item_implementations_still_work() {
    runtime_item_pass::main();
}

type Pair<A, B> = IntTuple<A, IntTuple<B, Nil>>;
type Mixed = Layout<Pair<Int<2>, Pair<DynInt, Int<4>>>, Pair<Int<1>, Pair<Int<2>, Int<6>>>>;

const MIXED: Mixed = Layout {
    shape: shape![Int::<2>, (DynInt(3), Int::<4>)],
    stride: stride![Int::<1>, (Int::<2>, Int::<6>)],
};
const SIZE: Result<isize> = MIXED.size();
const COSIZE: Result<isize> = MIXED.cosize();
const LAST: Result<isize> = MIXED.index(23);
const STORAGE: usize = match COSIZE {
    Ok(n) => n as usize,
    Err(_) => panic!("invalid storage span"),
};
const BUFFER: [u8; STORAGE] = [0; STORAGE];

#[test]
fn mixed_queries_and_buffer_lengths_are_const_evaluated() {
    assert_eq!(SIZE, Ok(24));
    assert_eq!(COSIZE, Ok(24));
    assert_eq!(LAST, Ok(23));
    assert_eq!(BUFFER.len(), 24);

    const NATIVE: Layout<isize, isize> = Layout {
        shape: 4,
        stride: 2,
    };
    const NATIVE_QUERIES: (Result<isize>, Result<isize>, Result<isize>) =
        (NATIVE.size(), NATIVE.cosize(), NATIVE.index(3));
    assert_eq!(NATIVE_QUERIES, (Ok(4), Ok(7), Ok(6)));
}

#[test]
fn const_queries_preserve_signed_broadcast_and_empty_semantics() {
    const SIGNED: Layout<Pair<Int<2>, DynInt>, Pair<Int<1>, DynInt>> = Layout {
        shape: shape![Int::<2>, DynInt(3)],
        stride: stride![Int::<1>, DynInt(-4)],
    };
    const SIGNED_SIZE: Result<isize> = SIGNED.size();
    const SIGNED_SPAN: Result<isize> = SIGNED.cosize();
    const SIGNED_LAST: Result<isize> = SIGNED.index(5);
    assert_eq!(SIGNED_SIZE, Ok(6));
    assert_eq!(SIGNED_SPAN, Ok(10));
    assert_eq!(SIGNED_LAST, Ok(-7));

    const BROADCAST: Layout<Int<8>, Int<0>> = StaticValue::VALUE;
    const BROADCAST_SPAN: Result<isize> = BROADCAST.cosize();
    const BROADCAST_LAST: Result<isize> = BROADCAST.index(7);
    assert_eq!(BROADCAST_SPAN, Ok(1));
    assert_eq!(BROADCAST_LAST, Ok(0));

    type EmptyLayout =
        Layout<IntTuple<DynInt, Pair<Int<2>, Int<0>>>, IntTuple<Int<0>, Pair<Int<0>, Int<1>>>>;
    const EMPTY: EmptyLayout = Layout {
        shape: shape![DynInt(isize::MAX), Int::<2>, Int::<0>],
        stride: stride![Int::<0>, Int::<0>, Int::<1>],
    };
    const EMPTY_SIZE: Result<isize> = EMPTY.size();
    const EMPTY_SPAN: Result<isize> = EMPTY.cosize();
    const EMPTY_INDEX: Result<isize> = EMPTY.index(0);
    assert_eq!(EMPTY_SIZE, Ok(0));
    assert_eq!(EMPTY_SPAN, Ok(0));
    assert_eq!(EMPTY_INDEX, Err(Error::CoordinateOutOfBounds));

    const SCALAR: Layout<Nil, Nil> = StaticValue::VALUE;
    const SCALAR_QUERIES: (Result<isize>, Result<isize>, Result<isize>) =
        (SCALAR.size(), SCALAR.cosize(), SCALAR.index(0));
    assert_eq!(SCALAR_QUERIES, (Ok(1), Ok(1), Ok(0)));
}

#[test]
fn const_queries_return_errors_without_requiring_a_panic() {
    const INVALID: Layout<Pair<Int<0>, DynInt>, Pair<Int<0>, Int<0>>> = Layout {
        shape: shape![Int::<0>, DynInt(-1)],
        stride: stride![Int::<0>, Int::<0>],
    };
    const INVALID_SIZE: Result<isize> = INVALID.size();
    const INVALID_SPAN: Result<isize> = INVALID.cosize();
    const NEGATIVE_INDEX: Result<isize> = INVALID.index(-1);
    assert_eq!(INVALID_SIZE, Err(Error::InvalidShape));
    assert_eq!(INVALID_SPAN, Err(Error::InvalidShape));
    assert_eq!(NEGATIVE_INDEX, Err(Error::CoordinateOutOfBounds));

    const TOO_LARGE: Layout<Pair<DynInt, Int<2>>, Pair<Int<0>, Int<0>>> = Layout {
        shape: shape![DynInt(isize::MAX), Int::<2>],
        stride: stride![Int::<0>, Int::<0>],
    };
    const SIZE_OVERFLOW: Result<isize> = TOO_LARGE.size();
    const INDEX_OVERFLOW: Result<isize> = TOO_LARGE.index(0);
    assert_eq!(SIZE_OVERFLOW, Err(Error::Overflow));
    assert_eq!(INDEX_OVERFLOW, Err(Error::Overflow));

    const WIDE: Layout<Int<3>, DynInt> = Layout {
        shape: Int,
        stride: DynInt(isize::MAX),
    };
    const WIDE_INDEX: Result<isize> = WIDE.index(1);
    const ADDRESS_OVERFLOW: Result<isize> = WIDE.index(2);
    const SPAN_OVERFLOW: Result<isize> = WIDE.cosize();
    assert_eq!(WIDE_INDEX, Ok(isize::MAX));
    assert_eq!(ADDRESS_OVERFLOW, Err(Error::Overflow));
    assert_eq!(SPAN_OVERFLOW, Err(Error::Overflow));

    const MINIMUM: Layout<Int<2>, DynInt> = Layout {
        shape: Int,
        stride: DynInt(isize::MIN),
    };
    const MINIMUM_INDEX: Result<isize> = MINIMUM.index(1);
    const MINIMUM_SPAN: Result<isize> = MINIMUM.cosize();
    assert_eq!(MINIMUM_INDEX, Ok(isize::MIN));
    assert_eq!(MINIMUM_SPAN, Err(Error::Overflow));
}

#[test]
fn inherent_queries_agree_with_generic_runtime_maps() {
    fn check<S: Shape, D: StrideOf<S>>(layout: Layout<S, D>) {
        assert_eq!(layout.size(), LayoutMap::size(&layout));
        assert_eq!(layout.cosize(), LayoutMap::cosize(&layout));
        for i in [-1, 0, 1, 5, 23, 24] {
            assert_eq!(layout.index(i), LayoutMap::index(&layout, i));
        }
    }
    check(MIXED);
    check(Layout::new(shape![DynInt(2), Int::<3>], stride![DynInt(-4), Int::<1>]).unwrap());
    check(Layout::new(Int::<0>, Int::<1>).unwrap());
    check(Layout::new(Nil, Nil).unwrap());
    check(Layout {
        shape: DynInt(-1),
        stride: Int::<1>,
    });
}

#[test]
fn const_integer_and_item_queries_keep_runtime_staticness() {
    const fn value<I: [const] Integer>(integer: I) -> isize {
        integer.value()
    }
    const DYNAMIC_VALUE: isize = value(DynInt(7));
    const LEAF: Result<isize> = MIXED.shape.leaf(1);
    const STATIC_LEAF: Result<Option<isize>> = MIXED.shape.leaf_static(1);
    const MODE_SPAN: Result<(usize, usize)> = MIXED.shape.mode_span(1);
    const PRODUCT: Result<isize> = MIXED.shape.product();
    const ZERO_PRODUCT: Result<isize> =
        shape![Int::<{ isize::MAX }>, DynInt(2), Int::<0>].product();
    assert_eq!(DYNAMIC_VALUE, 7);
    assert_eq!(LEAF, Ok(3));
    assert_eq!(STATIC_LEAF, Ok(None));
    assert_eq!(MODE_SPAN, Ok((1, 2)));
    assert_eq!(PRODUCT, Ok(24));
    assert_eq!(ZERO_PRODUCT, Ok(0));
}

#[test]
fn const_queries_do_not_export_bounded_metadata() {
    type Two<T> = Pair<T, T>;
    type Leaves32<T> = Two<Two<Two<Two<Two<T>>>>>;
    type Units32 = Leaves32<Int<1>>;
    type Zeros32 = Leaves32<Int<0>>;
    type Wide = Layout<IntTuple<Int<1>, Units32>, IntTuple<Int<0>, Zeros32>>;
    const LAYOUT: Wide = StaticValue::VALUE;
    const QUERIES: (Result<isize>, Result<isize>, Result<isize>) =
        (LAYOUT.size(), LAYOUT.cosize(), LAYOUT.index(0));
    assert_eq!(QUERIES, (Ok(1), Ok(1), Ok(0)));
    assert_eq!(LAYOUT.as_flat_layout(), Err(Error::CapacityExceeded));
}

#[test]
fn layout_desc_and_static_desc_unify_type_level_and_const_algebra() {
    use noodle_cute::algebra::{
        ComplementMode, TypedComplement, TypedComposition, complement_desc, compose_desc,
    };
    use noodle_cute::layout::{LayoutDesc, StaticDesc};

    type A = Layout<Pair<Int<4>, Int<4>>, Pair<Int<1>, Int<4>>>;
    type B = Layout<Pair<Int<2>, Int<2>>, Pair<Int<1>, Int<2>>>;
    const DESC_A: LayoutDesc = <A as StaticDesc>::DESC;
    const DESC_B: LayoutDesc = <B as StaticDesc>::DESC;

    assert_eq!(DESC_A.shapes(), &[4, 4]);
    assert_eq!(DESC_A.strides(), &[1, 4]);
    assert_eq!(DESC_B.shapes(), &[2, 2]);
    assert_eq!(DESC_B.strides(), &[1, 2]);

    const COMP_STATIC: LayoutDesc = <TypedComposition<A, B> as StaticDesc>::DESC;
    const COMP_CONST_FN: Result<LayoutDesc> = compose_desc(&DESC_A, &DESC_B);
    const COMP_INHERENT: Result<LayoutDesc> = A::VALUE.compose_desc(&B::VALUE);

    assert_eq!(COMP_CONST_FN, Ok(COMP_STATIC));
    assert_eq!(COMP_INHERENT, Ok(COMP_STATIC));
    assert_eq!(COMP_STATIC.shapes(), &[2, 2]);
    assert_eq!(COMP_STATIC.strides(), &[1, 2]);

    type Tile = Layout<Pair<Int<2>, Int<2>>, Pair<Int<1>, Int<6>>>;
    const TILE_DESC: LayoutDesc = <Tile as StaticDesc>::DESC;
    const COMPL_STATIC: LayoutDesc = <TypedComplement<Tile, Int<24>> as StaticDesc>::DESC;
    const COMPL_CONST_FN: Result<LayoutDesc> =
        complement_desc(&TILE_DESC, &[24], ComplementMode::Standard);
    const COMPL_INHERENT: Result<LayoutDesc> = Tile::VALUE.complement_desc::<24>();

    assert_eq!(COMPL_CONST_FN, Ok(COMPL_STATIC));
    assert_eq!(COMPL_INHERENT, Ok(COMPL_STATIC));
    assert_eq!(COMPL_STATIC.shapes(), &[3, 2]);
    assert_eq!(COMPL_STATIC.strides(), &[2, 12]);

    type ExtSource = Layout<Int<4>, Int<2>>;
    const EXT_STATIC: LayoutDesc =
        <TypedComplement<ExtSource, Nil, { ComplementMode::Extended }> as StaticDesc>::DESC;
    const EXT_CONST_FN: Result<LayoutDesc> = complement_desc(
        &<ExtSource as StaticDesc>::DESC,
        &[],
        ComplementMode::Extended,
    );
    const EXT_INHERENT: Result<LayoutDesc> = ExtSource::VALUE.complement_extended_desc();

    assert_eq!(EXT_CONST_FN, Ok(EXT_STATIC));
    assert_eq!(EXT_INHERENT, Ok(EXT_STATIC));
    assert_eq!(EXT_STATIC.shapes(), &[2, 1]);
    assert_eq!(EXT_STATIC.strides(), &[1, 8]);
}

#[test]
fn layout_desc_inherent_methods_and_const_eval_errors() {
    use noodle_cute::algebra::{CoalesceMode, ComplementMode};
    use noodle_cute::layout::LayoutDesc;

    // 1. Const constructor and basic queries
    const DESC: Result<LayoutDesc> = LayoutDesc::new(&[2, 3], &[1, 2]);
    assert!(DESC.is_ok());
    let desc = DESC.unwrap();
    assert_eq!(desc.size(), Ok(6));
    assert_eq!(desc.cosize(), Ok(6));
    assert_eq!(desc.index(0), Ok(0));
    assert_eq!(desc.index(5), Ok(5));
    assert_eq!(desc.leaf_shape(0), Ok(2));
    assert_eq!(desc.leaf_stride(1), Ok(2));
    assert_eq!(desc.leaf_shape(2), Err(Error::AxisOutOfBounds));

    // 2. Coalescing
    const UNCOALESCED: LayoutDesc = match LayoutDesc::from_flat(&[2, 3, 1], &[1, 2, 99]) {
        Ok(d) => d,
        Err(_) => panic!(),
    };
    const COALESCED: Result<LayoutDesc> = UNCOALESCED.coalesce(CoalesceMode::Standard);
    assert_eq!(COALESCED.unwrap().shapes(), &[6]);
    assert_eq!(COALESCED.unwrap().strides(), &[1]);

    const EXT_COALESCED: Result<LayoutDesc> = UNCOALESCED.coalesce(CoalesceMode::Extended);
    assert_eq!(EXT_COALESCED.unwrap().shapes(), &[6, 1]);
    assert_eq!(EXT_COALESCED.unwrap().strides(), &[1, 99]);

    // 3. Inherent compose and complement
    const OUTER: LayoutDesc = match LayoutDesc::from_flat(&[4, 4], &[1, 4]) {
        Ok(d) => d,
        Err(_) => panic!(),
    };
    const INNER: LayoutDesc = match LayoutDesc::from_flat(&[2, 2], &[1, 2]) {
        Ok(d) => d,
        Err(_) => panic!(),
    };
    const COMPOSED: Result<LayoutDesc> = OUTER.compose(&INNER);
    assert_eq!(COMPOSED.unwrap().shapes(), &[2, 2]);
    assert_eq!(COMPOSED.unwrap().strides(), &[1, 2]);

    const COMPL: Result<LayoutDesc> = OUTER.complement(&[16], ComplementMode::Standard);
    assert_eq!(COMPL.unwrap().shapes(), &[1]);
    assert_eq!(COMPL.unwrap().strides(), &[0]);

    const COMPL_EXTENDED: Result<LayoutDesc> = OUTER.complement(&[32], ComplementMode::Standard);
    assert_eq!(COMPL_EXTENDED.unwrap().shapes(), &[2]);
    assert_eq!(COMPL_EXTENDED.unwrap().strides(), &[16]);

    // 4. const_eq
    assert!(COMPOSED.unwrap().const_eq(&INNER));

    // 5. Const errors
    const NEGATIVE: Result<LayoutDesc> = LayoutDesc::new(&[-2], &[1]);
    assert_eq!(NEGATIVE, Err(Error::InvalidShape));

    const OVERFLOW_DESC: Result<LayoutDesc> = LayoutDesc::new(&[isize::MAX, 2], &[1, isize::MAX]);
    assert_eq!(OVERFLOW_DESC, Err(Error::Overflow));

    const NON_INJECTIVE: Result<LayoutDesc> = match LayoutDesc::from_flat(&[2, 2], &[1, 1]) {
        Ok(d) => d.complement(&[8], ComplementMode::Standard),
        Err(e) => Err(e),
    };
    assert_eq!(NON_INJECTIVE, Err(Error::NonInjective));
}
