#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use std::panic::catch_unwind;

use noodle_cute::algorithm::einfold::{ModeToken, parse_einfold};
use noodle_cute::integer::{Int, int};
use noodle_cute::layout::{Layout, is_layout};
use noodle_cute::swizzle::f2;
use noodle_cute::tensor::{ImplicitAccessor, Tensor, is_tensor, make_tensor};
use noodle_cute::tuple::{Item, LeafProfile};
use noodle_cute::{e, einfold, int_tuple, tuple};

/// Folds `source` as a Layout, as a Tensor (preserving accessor), as its Shape, and as its Stride.
macro_rules! check_einfold {
    ($subscripts:expr, $source:expr, $expected:expr) => {{
        let src = $source;
        let exp = $expected;

        let res = einfold!($subscripts, src);
        assert!(is_layout(&res));
        assert_eq!(res, exp, "Layout mismatch for {}", $subscripts);

        let tensor = Tensor::new(ImplicitAccessor::new(Int::<17>), src);
        let t_res = einfold!($subscripts, tensor);
        assert!(is_tensor(&t_res));
        assert_eq!(t_res.accessor, tensor.accessor);
        assert_eq!(
            t_res.layout, exp,
            "Tensor layout mismatch for {}",
            $subscripts
        );

        assert_eq!(
            einfold!($subscripts, src.shape),
            exp.shape,
            "Shape mismatch for {}",
            $subscripts
        );
        assert_eq!(
            einfold!($subscripts, src.stride),
            exp.stride,
            "Stride mismatch for {}",
            $subscripts
        );
    }};
}

fn assert_parse_panics(subscripts: &str, needle: &str) {
    let err = catch_unwind(|| parse_einfold(subscripts))
        .expect_err(&format!("expected parse_einfold({subscripts:?}) to panic"));
    let msg = err
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| err.downcast_ref::<String>().map(|s| s.as_str()))
        .unwrap_or("");
    assert!(
        msg.contains(needle),
        "for {subscripts:?}, expected panic containing {needle:?}, got {msg:?}"
    );
}

#[test]
fn test_einfold_layout_tensor_shape_stride_examples() {
    // Permuting and folding top-level modes.
    check_einfold!(
        "ijk -> ijk",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            int_tuple![Int::<7>, Int::<11>, Int::<13>]
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            int_tuple![Int::<7>, Int::<11>, Int::<13>]
        )
    );
    check_einfold!(
        "ijkl -> kilj",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>, Int::<7>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>, Int::<30>]
        ),
        Layout::new(
            int_tuple![Int::<5>, Int::<2>, Int::<7>, Int::<3>],
            int_tuple![Int::<6>, Int::<1>, Int::<30>, Int::<2>]
        )
    );
    check_einfold!(
        "abcd -> (ac)(db)",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>, Int::<7>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>, Int::<30>]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<5>), (Int::<7>, Int::<3>)],
            int_tuple![(Int::<1>, Int::<6>), (Int::<30>, Int::<2>)]
        )
    );
    // Size-1 and zero-stride modes.
    check_einfold!(
        "(ab)cd -> d(ac)b",
        Layout::new(
            int_tuple![(Int::<1>, Int::<3>), Int::<1>, Int::<5>],
            int_tuple![(Int::<0>, Int::<7>), Int::<0>, Int::<11>]
        ),
        Layout::new(
            int_tuple![Int::<5>, (Int::<1>, Int::<1>), Int::<3>],
            int_tuple![Int::<11>, (Int::<0>, Int::<0>), Int::<7>]
        )
    );
    // Noncompact strides.
    check_einfold!(
        "(ab)cd -> d(bc)a",
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<5>, Int::<7>],
            int_tuple![(Int::<101>, Int::<17>), Int::<1009>, Int::<43>]
        ),
        Layout::new(
            int_tuple![Int::<7>, (Int::<3>, Int::<5>), Int::<2>],
            int_tuple![Int::<43>, (Int::<17>, Int::<1009>), Int::<101>]
        )
    );

    // Each pair of parentheses adds one level.
    check_einfold!(
        "ij -> ij",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<1>, Int::<2>]
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<1>, Int::<2>]
        )
    );
    check_einfold!(
        "ij -> (ij)",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<1>, Int::<2>]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>)],
            int_tuple![(Int::<1>, Int::<2>)]
        )
    );
    check_einfold!(
        "ij -> ((ij))",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<1>, Int::<2>]
        ),
        Layout::new(
            int_tuple![((Int::<2>, Int::<3>),)],
            int_tuple![((Int::<1>, Int::<2>),)]
        )
    );

    // A group descends into a nested source mode; a name keeps its mode whole.
    check_einfold!(
        "(ab)c -> abc",
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<5>],
            int_tuple![(Int::<1>, Int::<2>), Int::<6>]
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>]
        )
    );
    check_einfold!(
        "(ab)c -> (ba)c",
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<5>],
            int_tuple![(Int::<7>, Int::<11>), Int::<13>]
        ),
        Layout::new(
            int_tuple![(Int::<3>, Int::<2>), Int::<5>],
            int_tuple![(Int::<11>, Int::<7>), Int::<13>]
        )
    );
    check_einfold!(
        "tb -> bt",
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<5>],
            int_tuple![(Int::<7>, Int::<11>), Int::<13>]
        ),
        Layout::new(
            int_tuple![Int::<5>, (Int::<2>, Int::<3>)],
            int_tuple![Int::<13>, (Int::<7>, Int::<11>)]
        )
    );
    check_einfold!(
        "(ab)(cd) -> (acb)d",
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), (Int::<5>, Int::<7>)],
            int_tuple![(Int::<1>, Int::<2>), (Int::<6>, Int::<30>)]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<5>, Int::<3>), Int::<7>],
            int_tuple![(Int::<1>, Int::<6>, Int::<2>), Int::<30>]
        )
    );
    check_einfold!(
        "((ab)(cd))e -> e((da)(bc))",
        Layout::new(
            int_tuple![((Int::<2>, Int::<3>), (Int::<5>, Int::<7>)), Int::<11>],
            int_tuple![((Int::<1>, Int::<2>), (Int::<6>, Int::<30>)), Int::<210>]
        ),
        Layout::new(
            int_tuple![Int::<11>, ((Int::<7>, Int::<2>), (Int::<3>, Int::<5>))],
            int_tuple![Int::<210>, ((Int::<30>, Int::<1>), (Int::<2>, Int::<6>))]
        )
    );

    // Whitespace is ignored.
    check_einfold!(
        "ijk->k(ij)",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>]
        ),
        Layout::new(
            int_tuple![Int::<5>, (Int::<2>, Int::<3>)],
            int_tuple![Int::<6>, (Int::<1>, Int::<2>)]
        )
    );
    check_einfold!(
        " i j k -> k ( i j ) ",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>]
        ),
        Layout::new(
            int_tuple![Int::<5>, (Int::<2>, Int::<3>)],
            int_tuple![Int::<6>, (Int::<1>, Int::<2>)]
        )
    );

    // Repeating an output name duplicates the mode it binds.
    check_einfold!(
        "ij -> iji",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<1>, Int::<2>]
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<2>],
            int_tuple![Int::<1>, Int::<2>, Int::<1>]
        )
    );
    check_einfold!(
        "ij -> (ii)j",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<1>, Int::<2>]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), Int::<3>],
            int_tuple![(Int::<1>, Int::<1>), Int::<2>]
        )
    );
    check_einfold!(
        "ij -> ii",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<1>, Int::<2>]
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<2>],
            int_tuple![Int::<1>, Int::<1>]
        )
    );
    check_einfold!(
        "tb -> tbt",
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<5>],
            int_tuple![(Int::<1>, Int::<2>), Int::<6>]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<5>, (Int::<2>, Int::<3>)],
            int_tuple![(Int::<1>, Int::<2>), Int::<6>, (Int::<1>, Int::<2>)]
        )
    );
    check_einfold!(
        "(ab)c -> (aca)b",
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<5>],
            int_tuple![(Int::<1>, Int::<2>), Int::<6>]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<5>, Int::<2>), Int::<3>],
            int_tuple![(Int::<1>, Int::<6>, Int::<1>), Int::<2>]
        )
    );

    // Omitting a name drops the mode it binds.
    check_einfold!(
        "ijk -> ik",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>]
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<5>],
            int_tuple![Int::<1>, Int::<6>]
        )
    );
    check_einfold!(
        "(ab)c -> ab",
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<5>],
            int_tuple![(Int::<1>, Int::<2>), Int::<6>]
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<1>, Int::<2>]
        )
    );
    check_einfold!(
        "ijk -> ik",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>, Int::<7>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>, Int::<30>]
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<5>, Int::<7>],
            int_tuple![Int::<1>, Int::<6>, Int::<30>]
        )
    );
    check_einfold!(
        "ijk -> (ik)",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>, Int::<7>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>, Int::<30>]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<5>), Int::<7>],
            int_tuple![(Int::<1>, Int::<6>), Int::<30>]
        )
    );

    // Lone name binds the whole shape, however deep.
    check_einfold!(
        "i -> i",
        Layout::new(Int::<4>, Int::<5>),
        Layout::new(Int::<4>, Int::<5>)
    );
    check_einfold!(
        "i -> (i)",
        Layout::new(Int::<4>, Int::<5>),
        Layout::new(int_tuple![Int::<4>], int_tuple![Int::<5>])
    );
    check_einfold!(
        "i -> i",
        Layout::new(int_tuple![Int::<4>], int_tuple![Int::<5>]),
        Layout::new(int_tuple![Int::<4>], int_tuple![Int::<5>])
    );
    check_einfold!(
        "(i) -> i",
        Layout::new(int_tuple![Int::<4>], int_tuple![Int::<5>]),
        Layout::new(Int::<4>, Int::<5>)
    );
    check_einfold!(
        "i -> i",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>]
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>]
        )
    );
    check_einfold!(
        "i -> (i)",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>, Int::<5>)],
            int_tuple![(Int::<1>, Int::<2>, Int::<6>)]
        )
    );
    check_einfold!(
        "i -> ii",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>]
        ),
        Layout::new(
            int_tuple![
                (Int::<2>, Int::<3>, Int::<5>),
                (Int::<2>, Int::<3>, Int::<5>)
            ],
            int_tuple![
                (Int::<1>, Int::<2>, Int::<6>),
                (Int::<1>, Int::<2>, Int::<6>)
            ]
        )
    );

    // Wrapping and unwrapping are inverses.
    let source = Layout::new(
        int_tuple![(Int::<2>, Int::<3>), Int::<5>],
        int_tuple![(Int::<1>, Int::<2>), Int::<6>],
    );
    let wrapped = einfold!("i -> (i)", source);
    assert_eq!(
        wrapped,
        Layout::new(
            int_tuple![((Int::<2>, Int::<3>), Int::<5>)],
            int_tuple![((Int::<1>, Int::<2>), Int::<6>)]
        )
    );
    assert_eq!(einfold!("(i) -> i", wrapped), source);

    // Top-level unnamed modes pass through unchanged.
    check_einfold!(
        "(i) -> i",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>]
        ),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>]
        )
    );
    check_einfold!(
        "(i) -> ((i))",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>]
        ),
        Layout::new(
            int_tuple![(Int::<2>,), Int::<3>, Int::<5>],
            int_tuple![(Int::<1>,), Int::<2>, Int::<6>]
        )
    );
    check_einfold!(
        "ij -> ji",
        Layout::from_shape(int_tuple![
            (Int::<12>, Int::<4>),
            Int::<42>,
            Int::<5>,
            Int::<7>
        ]),
        Layout::new(
            int_tuple![Int::<42>, (Int::<12>, Int::<4>), Int::<5>, Int::<7>],
            int_tuple![Int::<48>, (Int::<1>, Int::<12>), Int::<2016>, Int::<10080>]
        )
    );
    check_einfold!(
        "ij -> (ji)",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>, Int::<7>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>, Int::<30>]
        ),
        Layout::new(
            int_tuple![(Int::<3>, Int::<2>), Int::<5>, Int::<7>],
            int_tuple![(Int::<2>, Int::<1>), Int::<6>, Int::<30>]
        )
    );
    check_einfold!(
        "((ab)(cd))e -> e((da)(bc))",
        Layout::from_shape(int_tuple![
            ((Int::<2>, Int::<3>), (Int::<4>, Int::<5>)),
            Int::<6>,
            Int::<7>
        ]),
        Layout::new(
            int_tuple![
                Int::<6>,
                ((Int::<5>, Int::<2>), (Int::<3>, Int::<4>)),
                Int::<7>
            ],
            int_tuple![
                Int::<120>,
                ((Int::<24>, Int::<1>), (Int::<2>, Int::<6>)),
                Int::<720>
            ]
        )
    );

    // Top-level group is one mode, as anywhere else.
    check_einfold!(
        "ij -> (ij)",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>, Int::<7>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>, Int::<30>]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<5>, Int::<7>],
            int_tuple![(Int::<1>, Int::<2>), Int::<6>, Int::<30>]
        )
    );
    check_einfold!(
        "ijk -> (ij)k",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>, Int::<5>, Int::<7>],
            int_tuple![Int::<1>, Int::<2>, Int::<6>, Int::<30>]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<5>, Int::<7>],
            int_tuple![(Int::<1>, Int::<2>), Int::<6>, Int::<30>]
        )
    );
    check_einfold!(
        "(ij) -> ji",
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<5>, Int::<7>],
            int_tuple![(Int::<1>, Int::<2>), Int::<6>, Int::<30>]
        ),
        Layout::new(
            int_tuple![Int::<3>, Int::<2>, Int::<5>, Int::<7>],
            int_tuple![Int::<2>, Int::<1>, Int::<6>, Int::<30>]
        )
    );
    check_einfold!(
        "(ij)k -> jik",
        Layout::new(
            int_tuple![(Int::<2>, Int::<3>), Int::<5>, Int::<7>],
            int_tuple![(Int::<1>, Int::<2>), Int::<6>, Int::<30>]
        ),
        Layout::new(
            int_tuple![Int::<3>, Int::<2>, Int::<5>, Int::<7>],
            int_tuple![Int::<2>, Int::<1>, Int::<6>, Int::<30>]
        )
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tag(&'static str);

const impl Item for Tag {
    const RANK: int = 1;
    const DEPTH: int = 0;
    const LEAVES: int = 1;
    const IS_STATIC: bool = false;
    type Profile = LeafProfile;
}

#[test]
fn test_einfold_htuples_and_diverse_leaves() {
    assert_eq!(einfold!("ij -> ji", int_tuple![2, 3]), int_tuple![3, 2]);
    assert_eq!(einfold!("ij -> (ij)", int_tuple![2, 3]), int_tuple![(2, 3)]);
    assert_eq!(einfold!("i -> i", int_tuple![2, 3, 5]), int_tuple![2, 3, 5]);
    assert_eq!(
        einfold!("(ab)c -> (ca)b", int_tuple![(2, 3), 5]),
        int_tuple![(5, 2), 3]
    );
    assert_eq!(
        einfold!("ij -> ji", int_tuple![(12, 4), 42, 5, 7]),
        int_tuple![42, (12, 4), 5, 7]
    );

    // Leaves can be F2, Basis, RangeFull (..), custom leaf items, or Layouts.
    assert_eq!(
        einfold!("ij -> ji", tuple![f2(Int::<1>), f2(Int::<2>)]),
        tuple![f2(Int::<2>), f2(Int::<1>)]
    );
    assert_eq!(
        einfold!("ij -> ji", tuple![e![0], e![1]]),
        tuple![e![1], e![0]]
    );
    assert_eq!(einfold!("ij -> ji", tuple![.., 3i32]), tuple![3i32, ..]);
    assert_eq!(
        einfold!("(ab)c -> (cb)a", tuple![(.., 1i32), 2i32]),
        tuple![(2i32, 1i32), ..]
    );
    assert_eq!(
        einfold!("ij -> ji", tuple![Tag("m"), Tag("n")]),
        tuple![Tag("n"), Tag("m")]
    );

    // A Tiler's Layouts are its leaves, so each is one whole mode.
    let l1 = Layout::from_shape(int_tuple![Int::<2>, Int::<3>]);
    let l2 = Layout::from_shape(Int::<5>);
    let tiler = tuple![l1, l2];
    assert_eq!(einfold!("ij -> ji", tiler), tuple![l2, l1]);
    assert_eq!(einfold!("ij -> (ij)", tiler), tuple![(l1, l2)]);

    // Scalar leaves.
    assert_eq!(einfold!("i -> i", 4i32), 4i32);
    assert_eq!(einfold!("i -> (i)", 4i32), int_tuple![4i32]);
    assert_eq!(einfold!("i -> ii", 4i32), int_tuple![4i32, 4i32]);
    assert_eq!(einfold!("i -> (i)", Int::<4>), int_tuple![Int::<4>]);
}

#[test]
fn test_einfold_tensor_views_alias_source() {
    let mut buf = [0i32; 6];
    let mut source = make_tensor(
        &mut buf,
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<1>, Int::<2>],
        ),
    );
    for i in 0..2 {
        for j in 0..3 {
            source.set(int_tuple![i, j], 10 * i + j);
        }
    }

    let mut view = einfold!("ij -> ji", source);
    assert!(is_tensor(&view));
    assert_eq!(view.accessor, source.accessor);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(view.at(int_tuple![j, i]), source.at(int_tuple![i, j]));
        }
    }

    view.set(int_tuple![2i32, 1i32], 123);
    assert_eq!(source.at(int_tuple![1i32, 2i32]), 123);
    source.set(int_tuple![0i32, 1i32], 456);
    assert_eq!(view.at(int_tuple![1i32, 0i32]), 456);
}

#[test]
fn test_einfold_parse_spec_structure() {
    let s1 = parse_einfold("i -> i");
    assert!(!s1.in_is_tuple && !s1.out_is_tuple);
    assert_eq!(s1.num_leaves, 1);

    let s2 = parse_einfold("(i) -> ((i))");
    assert!(s2.in_is_tuple && s2.out_is_tuple);
    assert_eq!(s2.in_rank, 1);
    assert_eq!(s2.out_tokens[0], ModeToken::Open);
    assert_eq!(s2.out_tokens[1], ModeToken::Open);
    assert_eq!(s2.out_tokens[2], ModeToken::Leaf(0));
    assert_eq!(s2.out_tokens[3], ModeToken::Close);
    assert_eq!(s2.out_tokens[4], ModeToken::Close);

    let s3 = parse_einfold("i0j1 -> 1j0i");
    assert!(s3.in_is_tuple && s3.out_is_tuple);
    assert_eq!(s3.in_rank, 4);
    assert_eq!(s3.num_leaves, 4);

    // Const evaluation check
    type FoldedShape =
        noodle_cute::tuple::Til<Int<3>, noodle_cute::tuple::Til<Int<2>, noodle_cute::tuple::Nil>>;
    type FoldedStride =
        noodle_cute::tuple::Til<Int<2>, noodle_cute::tuple::Til<Int<1>, noodle_cute::tuple::Nil>>;
    const FOLDED: Layout<FoldedShape, FoldedStride> = einfold!(
        "ij -> ji",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<1>, Int::<2>]
        )
    );
    assert_eq!(
        FOLDED,
        Layout::new(
            int_tuple![Int::<3>, Int::<2>],
            int_tuple![Int::<2>, Int::<1>]
        )
    );
}

#[test]
fn test_einfold_malformed_expressions_and_errors() {
    let bad_modes = [
        ("", "empty mode expression"),
        ("   ", "empty mode expression"),
        ("()", "empty mode group"),
        ("(", "unclosed '('"),
        ("(a(b)", "unclosed '('"),
        (")", "unmatched ')'"),
        ("a)b", "unmatched ')'"),
        ("i,j", "separators between modes are implied"),
        ("i_j", "invalid mode name"),
        ("i+j", "invalid mode name"),
        ("i.j", "invalid mode name"),
        ("[ij]", "invalid mode name"),
    ];
    for (expr, msg) in bad_modes {
        assert_parse_panics(&format!("{expr} -> ij"), msg);
        assert_parse_panics(&format!("ij -> {expr}"), msg);
    }

    for bad_arrow in ["ij", "ji", "ij>ji", "ij-ji", "ij->ji->k"] {
        assert_parse_panics(bad_arrow, "expected exactly one '->'");
    }

    assert_parse_panics("ii->i", "input contains repeated names");
    assert_parse_panics("(ij)i->ij", "input contains repeated names");
    assert_parse_panics("ij->ik", "output contains unknown name");
    assert_parse_panics("ij->ijk", "output contains unknown name");
    assert_parse_panics("ij->(jk)", "output contains unknown name");
}
