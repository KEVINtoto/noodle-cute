#![feature(test)]

extern crate test;

use core::hint::black_box;

use noodle_cute::algebra::composition_layout;
use noodle_cute::layout::{DynLayout, Layout, LayoutMap};
use noodle_cute::{DynInt, Int, IntTuple, Nil, shape, stride};
use test::Bencher;

type Pair<A, B> = IntTuple<A, IntTuple<B, Nil>>;
type Quad<A, B, C, D> = IntTuple<A, IntTuple<B, Pair<C, D>>>;
type Static2 = Layout<Pair<Int<64>, Int<32>>, Pair<Int<1>, Int<64>>>;
type Mixed2 = Layout<Pair<Int<64>, DynInt>, Pair<Int<1>, DynInt>>;
type FourTwos = Quad<Int<2>, Int<2>, Int<2>, Int<2>>;
type Static8 = Layout<
    Pair<FourTwos, FourTwos>,
    Pair<Quad<Int<1>, Int<2>, Int<4>, Int<8>>, Quad<Int<16>, Int<32>, Int<64>, Int<128>>>,
>;
type Mixed8 = Layout<
    Pair<FourTwos, Quad<Int<2>, Int<2>, Int<2>, DynInt>>,
    Pair<Quad<Int<1>, Int<2>, Int<4>, Int<8>>, Quad<Int<16>, Int<32>, Int<64>, DynInt>>,
>;

fn static_2() -> Static2 {
    Layout::new(shape![Int::<64>, Int::<32>], stride![Int::<1>, Int::<64>]).unwrap()
}

fn mixed_2() -> Mixed2 {
    Layout::new(
        shape![Int::<64>, DynInt(black_box(32))],
        stride![Int::<1>, DynInt(black_box(64))],
    )
    .unwrap()
}

fn dynamic_2() -> DynLayout {
    DynLayout::new(black_box(&[64, 32]), black_box(&[1, 64])).unwrap()
}

fn static_8() -> Static8 {
    Layout::new(
        shape![
            (Int::<2>, Int::<2>, Int::<2>, Int::<2>),
            (Int::<2>, Int::<2>, Int::<2>, Int::<2>)
        ],
        stride![
            (Int::<1>, Int::<2>, Int::<4>, Int::<8>),
            (Int::<16>, Int::<32>, Int::<64>, Int::<128>)
        ],
    )
    .unwrap()
}

fn mixed_8() -> Mixed8 {
    Layout::new(
        shape![
            (Int::<2>, Int::<2>, Int::<2>, Int::<2>),
            (Int::<2>, Int::<2>, Int::<2>, DynInt(black_box(2)))
        ],
        stride![
            (Int::<1>, Int::<2>, Int::<4>, Int::<8>),
            (Int::<16>, Int::<32>, Int::<64>, DynInt(black_box(128)))
        ],
    )
    .unwrap()
}

fn dynamic_8() -> DynLayout {
    DynLayout::new(
        black_box(&[2; 8]),
        black_box(&[1, 2, 4, 8, 16, 32, 64, 128]),
    )
    .unwrap()
}

macro_rules! query {
    ($name:ident, $layout:expr, $method:ident($($arg:expr),*)) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let layout = $layout;
            b.iter(|| black_box(black_box(&layout).$method($(black_box($arg)),*).unwrap()));
        }
    };
}

query!(size_static_leaves2, static_2(), size());
query!(size_mixed_leaves2, mixed_2(), size());
query!(size_dynamic_leaves2, dynamic_2(), size());
query!(size_static_leaves8, static_8(), size());
query!(size_mixed_leaves8, mixed_8(), size());
query!(size_dynamic_leaves8, dynamic_8(), size());
query!(cosize_static_leaves2, static_2(), cosize());
query!(cosize_mixed_leaves2, mixed_2(), cosize());
query!(cosize_dynamic_leaves2, dynamic_2(), cosize());
query!(cosize_static_leaves8, static_8(), cosize());
query!(cosize_mixed_leaves8, mixed_8(), cosize());
query!(cosize_dynamic_leaves8, dynamic_8(), cosize());
query!(index_static_leaves2, static_2(), index(977));
query!(index_mixed_leaves2, mixed_2(), index(977));
query!(index_dynamic_leaves2, dynamic_2(), index(977));
query!(index_static_leaves8, static_8(), index(173));
query!(index_mixed_leaves8, mixed_8(), index(173));
query!(index_dynamic_leaves8, dynamic_8(), index(173));
query!(injective_proof_static_leaves2, static_2(), is_injective());
query!(injective_proof_mixed_leaves2, mixed_2(), is_injective());
query!(injective_proof_dynamic_leaves2, dynamic_2(), is_injective());
query!(injective_proof_static_leaves8, static_8(), is_injective());
query!(injective_proof_mixed_leaves8, mixed_8(), is_injective());
query!(injective_proof_dynamic_leaves8, dynamic_8(), is_injective());

macro_rules! fallback {
    ($static:ident, $mixed:ident, $dynamic:ident, $n:literal) => {
        query!(
            $static,
            Layout::new(shape![Int::<3>, Int::<$n>], stride![Int::<2>, Int::<3>]).unwrap(),
            is_injective()
        );
        query!(
            $mixed,
            Layout::new(
                shape![Int::<3>, DynInt(black_box($n))],
                stride![Int::<2>, Int::<3>]
            )
            .unwrap(),
            is_injective()
        );
        query!(
            $dynamic,
            DynLayout::new(black_box(&[3, $n]), black_box(&[2, 3])).unwrap(),
            is_injective()
        );
    };
}

fallback!(
    injective_fallback_static_domain24,
    injective_fallback_mixed_domain24,
    injective_fallback_dynamic_domain24,
    8
);
fallback!(
    injective_fallback_static_domain96,
    injective_fallback_mixed_domain96,
    injective_fallback_dynamic_domain96,
    32
);

macro_rules! construct {
    ($name:ident, $layout:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            b.iter(|| black_box($layout));
        }
    };
}

construct!(construct_static_leaves2, static_2());
construct!(construct_mixed_leaves2, mixed_2());
construct!(construct_dynamic_leaves2, dynamic_2());

query!(coalesce_static_leaves2, static_2(), coalesce());
query!(coalesce_mixed_leaves2, mixed_2(), coalesce());
query!(coalesce_dynamic_leaves2, dynamic_2(), coalesce());

#[bench]
fn composition_static(b: &mut Bencher) {
    let outer = static_2();
    let inner = Layout::new(Int::<32>, Int::<4>).unwrap();
    b.iter(|| black_box(black_box(&outer).composition(black_box(inner)).unwrap()));
}

#[bench]
fn composition_mixed(b: &mut Bencher) {
    let outer = mixed_2();
    let inner = Layout::new(DynInt(black_box(32)), Int::<4>).unwrap();
    b.iter(|| black_box(black_box(&outer).composition(black_box(inner)).unwrap()));
}

#[bench]
fn composition_dynamic(b: &mut Bencher) {
    let outer = dynamic_2();
    let inner = DynLayout::new(black_box(&[32]), black_box(&[4])).unwrap();
    b.iter(|| black_box(composition_layout(black_box(&outer), black_box(&inner)).unwrap()));
}

fn check_compact<L: LayoutMap>(layout: L, size: isize) {
    assert_eq!(layout.size(), Ok(size));
    assert_eq!(layout.cosize(), Ok(size));
    assert_eq!(layout.is_injective(), Ok(true));
    for i in [0, size / 2, size - 1] {
        assert_eq!(layout.index(i), Ok(i));
    }
}

#[test]
fn compact_fixtures_have_the_same_linear_map() {
    check_compact(static_2(), 2048);
    check_compact(mixed_2(), 2048);
    check_compact(dynamic_2(), 2048);
    check_compact(static_8(), 256);
    check_compact(mixed_8(), 256);
    check_compact(dynamic_8(), 256);
    check_compact(static_2().coalesce().unwrap(), 2048);
    check_compact(mixed_2().coalesce().unwrap(), 2048);
    check_compact(dynamic_2().coalesce().unwrap(), 2048);
}

#[test]
fn fallback_fixtures_exhaust_an_injective_domain() {
    fn check<L: LayoutMap>(layout: L, n: isize) {
        assert_eq!(layout.size(), Ok(3 * n));
        assert_eq!(layout.cosize(), Ok(3 * n + 2));
        assert_eq!(layout.is_injective(), Ok(true));
        for i in 0..3 * n {
            assert_eq!(layout.index(i), Ok(2 * (i % 3) + 3 * (i / 3)));
        }
    }
    check(
        Layout::new(shape![Int::<3>, Int::<8>], stride![Int::<2>, Int::<3>]).unwrap(),
        8,
    );
    check(
        Layout::new(shape![Int::<3>, Int::<32>], stride![Int::<2>, Int::<3>]).unwrap(),
        32,
    );
    for n in [8, 32] {
        check(
            Layout::new(shape![Int::<3>, DynInt(n)], stride![Int::<2>, Int::<3>]).unwrap(),
            n,
        );
        check(DynLayout::new(&[3, n], &[2, 3]).unwrap(), n);
    }
}

#[test]
fn composition_fixtures_have_the_same_linear_map() {
    let a = static_2()
        .composition(Layout::new(Int::<32>, Int::<4>).unwrap())
        .unwrap();
    let b = mixed_2()
        .composition(Layout::new(DynInt(32), Int::<4>).unwrap())
        .unwrap();
    let c = composition_layout(&dynamic_2(), &DynLayout::new(&[32], &[4]).unwrap()).unwrap();
    assert_eq!(a.size(), Ok(32));
    assert_eq!(b.size(), Ok(32));
    assert_eq!(c.size(), Ok(32));
    for i in 0..32 {
        assert_eq!(a.index(i), Ok(4 * i));
        assert_eq!(b.index(i), Ok(4 * i));
        assert_eq!(c.index(i), Ok(4 * i));
    }
}
