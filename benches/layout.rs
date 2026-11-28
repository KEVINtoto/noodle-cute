#![feature(
    test,
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

extern crate test;

use core::hint::black_box;

use noodle_cute::algebra::{coalesce, composition, logical_divide, logical_product, right_inverse};
use noodle_cute::int_tuple;
use noodle_cute::integer::Int;
use noodle_cute::layout::Layout;
use noodle_cute::tuple::{Nil, Til};
use test::Bencher;

type Pair<A, B> = Til<A, Til<B, Nil>>;
type Quad<A, B, C, D> = Til<A, Til<B, Pair<C, D>>>;
type Static2 = Layout<Pair<Int<64>, Int<32>>, Pair<Int<1>, Int<64>>>;
type Mixed2 = Layout<Pair<Int<64>, i32>, Pair<Int<1>, i32>>;
type Dynamic2 = Layout<Pair<i32, i32>, Pair<i32, i32>>;
type FourTwos = Quad<Int<2>, Int<2>, Int<2>, Int<2>>;
type Static8 = Layout<
    Pair<FourTwos, FourTwos>,
    Pair<Quad<Int<1>, Int<2>, Int<4>, Int<8>>, Quad<Int<16>, Int<32>, Int<64>, Int<128>>>,
>;
type Mixed8 = Layout<
    Pair<FourTwos, Quad<Int<2>, Int<2>, Int<2>, i32>>,
    Pair<Quad<Int<1>, Int<2>, Int<4>, Int<8>>, Quad<Int<16>, Int<32>, Int<64>, i32>>,
>;

fn static_2() -> Static2 {
    Layout::new(int_tuple![Int::<64>, Int::<32>], int_tuple![Int::<1>, Int::<64>])
}

fn mixed_2() -> Mixed2 {
    Layout::new(
        int_tuple![Int::<64>, black_box(32i32)],
        int_tuple![Int::<1>, black_box(64i32)],
    )
}

fn dynamic_2() -> Dynamic2 {
    Layout::new(
        int_tuple![black_box(64i32), black_box(32i32)],
        int_tuple![black_box(1i32), black_box(64i32)],
    )
}

fn static_8() -> Static8 {
    Layout::new(
        int_tuple![
            (Int::<2>, Int::<2>, Int::<2>, Int::<2>),
            (Int::<2>, Int::<2>, Int::<2>, Int::<2>)
        ],
        int_tuple![
            (Int::<1>, Int::<2>, Int::<4>, Int::<8>),
            (Int::<16>, Int::<32>, Int::<64>, Int::<128>)
        ],
    )
}

fn mixed_8() -> Mixed8 {
    Layout::new(
        int_tuple![
            (Int::<2>, Int::<2>, Int::<2>, Int::<2>),
            (Int::<2>, Int::<2>, Int::<2>, black_box(2i32))
        ],
        int_tuple![
            (Int::<1>, Int::<2>, Int::<4>, Int::<8>),
            (Int::<16>, Int::<32>, Int::<64>, black_box(128i32))
        ],
    )
}

macro_rules! query_bench {
    ($name:ident, $layout:expr, $expr:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let layout = $layout;
            b.iter(|| {
                let l = black_box(layout);
                black_box(($expr)(l))
            });
        }
    };
}

query_bench!(size_static_leaves2, static_2(), |l: Static2| l.size());
query_bench!(size_mixed_leaves2, mixed_2(), |l: Mixed2| l.size());
query_bench!(size_dynamic_leaves2, dynamic_2(), |l: Dynamic2| l.size());
query_bench!(size_static_leaves8, static_8(), |l: Static8| l.size());
query_bench!(size_mixed_leaves8, mixed_8(), |l: Mixed8| l.size());

query_bench!(cosize_static_leaves2, static_2(), |l: Static2| l.cosize());
query_bench!(cosize_mixed_leaves2, mixed_2(), |l: Mixed2| l.cosize());
query_bench!(cosize_dynamic_leaves2, dynamic_2(), |l: Dynamic2| l.cosize());
query_bench!(cosize_static_leaves8, static_8(), |l: Static8| l.cosize());
query_bench!(cosize_mixed_leaves8, mixed_8(), |l: Mixed8| l.cosize());

query_bench!(index_static_leaves2, static_2(), |l: Static2| l(black_box(977i32)));
query_bench!(index_mixed_leaves2, mixed_2(), |l: Mixed2| l(black_box(977i32)));
query_bench!(index_dynamic_leaves2, dynamic_2(), |l: Dynamic2| l(black_box(977i32)));
query_bench!(index_static_leaves8, static_8(), |l: Static8| l(black_box(173i32)));
query_bench!(index_mixed_leaves8, mixed_8(), |l: Mixed8| l(black_box(173i32)));

query_bench!(coalesce_static_leaves2, static_2(), |l: Static2| coalesce(l));
query_bench!(coalesce_mixed_leaves2, mixed_2(), |l: Mixed2| coalesce(l));
query_bench!(coalesce_dynamic_leaves2, dynamic_2(), |l: Dynamic2| coalesce(l));

#[bench]
fn composition_static(b: &mut Bencher) {
    let outer = static_2();
    let inner = Layout::new(Int::<32>, Int::<4>);
    b.iter(|| black_box(composition(black_box(outer), black_box(inner))));
}

#[bench]
fn composition_mixed(b: &mut Bencher) {
    let outer = mixed_2();
    let inner = Layout::new(Int::<32>, Int::<4>);
    b.iter(|| black_box(composition(black_box(outer), black_box(inner))));
}

#[bench]
fn divide_and_product_static(b: &mut Bencher) {
    let outer = static_2();
    b.iter(|| {
        let d = logical_divide(black_box(outer), int_tuple![Int::<8>, Int::<4>]);
        let p = logical_product(black_box(outer), Layout::new(Int::<2>, Int::<1>));
        let r = right_inverse(black_box(outer));
        black_box((d, p, r))
    });
}

#[test]
fn benchmark_fixtures_agree_across_static_mixed_and_dynamic() {
    let s2 = static_2();
    let m2 = mixed_2();
    let d2 = dynamic_2();
    assert_eq!((s2.size(), m2.size(), d2.size()), (2048, 2048, 2048));
    assert_eq!((s2.cosize(), m2.cosize(), d2.cosize()), (2048, 2048, 2048));
    for i in [0i32, 977, 2047] {
        assert_eq!((s2(i), m2(i), d2(i)), (i, i, i));
    }

    let s8 = static_8();
    let m8 = mixed_8();
    assert_eq!((s8.size(), m8.size()), (256, 256));
    assert_eq!((s8(173i32), m8(173i32)), (173, 173));

    let cs = composition(s2, Layout::new(Int::<32>, Int::<4>));
    let cm = composition(m2, Layout::new(Int::<32>, Int::<4>));
    assert_eq!((cs.size(), cm.size()), (32, 32));
    for i in 0..32i32 {
        assert_eq!((cs(i), cm(i)), (4 * i, 4 * i));
    }
}
