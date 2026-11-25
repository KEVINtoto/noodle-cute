#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use noodle_cute::algorithm::einsum::{EinsumSpec, OperandFold, fold_operand, parse_einsum};
use noodle_cute::integer::{Int, int};
use noodle_cute::layout::Layout;
use noodle_cute::shape::Shape;
use noodle_cute::stride::Stride;
use noodle_cute::tensor::{Tensor, TensorIndex, TensorSet, make_tensor};
use noodle_cute::tuple::Nil;
use noodle_cute::{einsum, int_tuple};

/// Converts a multi-index into a 1D colexicographical linear index for flat `shape`.
fn colex_index(coord: &[usize], shape: &[usize]) -> int {
    let mut idx = 0usize;
    let mut mul = 1usize;
    for (&c, &s) in coord.iter().zip(shape.iter()) {
        idx += c * mul;
        mul *= s;
    }
    idx as int
}

struct OperandData<'a> {
    modes: &'a str,
    vals: &'a [f64],
    shape: &'a [usize],
}

/// Dependency-free brute-force oracle matching `examples/einsum_test.py::_reference`.
fn einsum_oracle(
    a: &OperandData<'_>,
    b: &OperandData<'_>,
    c_mode: &str,
    extents: &[(u8, usize)],
) -> BTreeMap<Vec<usize>, f64> {
    let ext_of = |ch: u8| -> usize {
        extents
            .iter()
            .find(|&&(k, _)| k == ch)
            .map(|&(_, v)| v)
            .expect("missing extent")
    };

    let mut labels = Vec::new();
    for &ch in a
        .modes
        .as_bytes()
        .iter()
        .chain(b.modes.as_bytes())
        .chain(c_mode.as_bytes())
    {
        if !labels.contains(&ch) {
            labels.push(ch);
        }
    }

    let label_exts: Vec<usize> = labels.iter().map(|&ch| ext_of(ch)).collect();
    let total: usize = label_exts.iter().product::<usize>().max(1);
    let mut out: BTreeMap<Vec<usize>, f64> = BTreeMap::new();

    for flat in 0..total {
        let mut rem = flat;
        let mut assign = [0usize; 256];
        for (&ch, &e) in labels.iter().zip(label_exts.iter()) {
            assign[ch as usize] = rem % e;
            rem /= e;
        }
        let a_idx: Vec<usize> = a
            .modes
            .as_bytes()
            .iter()
            .map(|&ch| assign[ch as usize])
            .collect();
        let b_idx: Vec<usize> = b
            .modes
            .as_bytes()
            .iter()
            .map(|&ch| assign[ch as usize])
            .collect();
        let c_idx: Vec<usize> = c_mode
            .as_bytes()
            .iter()
            .map(|&ch| assign[ch as usize])
            .collect();

        let av = a.vals[colex_index(&a_idx, a.shape) as usize];
        let bv = b.vals[colex_index(&b_idx, b.shape) as usize];
        *out.entry(c_idx).or_insert(0.0) += av * bv;
    }

    out
}

fn fill_f64<A, S, D>(t: &mut Tensor<A, Layout<S, D>>, f: impl Fn(int) -> f64)
where
    S: Shape,
    D: Stride<S>,
    Tensor<A, Layout<S, D>>: TensorSet<int, Item = f64>,
{
    for i in 0..t.size() {
        t.set(i, f(i));
    }
}

fn read_logical_f64<A, S, D>(t: &Tensor<A, Layout<S, D>>) -> Vec<f64>
where
    S: Shape,
    D: Stride<S>,
    Tensor<A, Layout<S, D>>: TensorIndex<int, Output = f64>,
{
    (0..t.size()).map(|i| t.at(i)).collect()
}

macro_rules! check_einsum {
    (
        $subscripts:literal,
        $a_layout:expr,
        $b_layout:expr,
        $c_layout:expr,
        $extents:expr,
        $c_init:expr
    ) => {{
        let al = $a_layout;
        let bl = $b_layout;
        let cl = $c_layout;
        let c_init: f64 = $c_init;

        let mut a_buf = vec![0.0f64; (al.cosize() as usize).max(1)];
        let mut b_buf = vec![0.0f64; (bl.cosize() as usize).max(1)];
        let mut c_buf = vec![0.0f64; (cl.cosize() as usize).max(1)];

        let mut a = make_tensor(&mut a_buf[..], al);
        let mut b = make_tensor(&mut b_buf[..], bl);
        let mut c = make_tensor(&mut c_buf[..], cl);

        fill_f64(&mut a, |i| (i + 1) as f64);
        fill_f64(&mut b, |i| (2 * i + 3) as f64);
        fill_f64(&mut c, |_| c_init);

        let a_logical = read_logical_f64(&a);
        let b_logical = read_logical_f64(&b);

        einsum!($subscripts, a, b, &mut c);

        let cleaned: String = $subscripts
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect();
        let (lhs, c_mode) = cleaned.split_once("->").unwrap();
        let (a_mode, b_mode) = lhs.split_once(',').unwrap();
        let exts: &[(u8, usize)] = &$extents;
        let ext_vec = |m: &str| -> Vec<usize> {
            m.as_bytes()
                .iter()
                .map(|&ch| exts.iter().find(|&&(k, _)| k == ch).unwrap().1)
                .collect()
        };
        let a_shape = ext_vec(a_mode);
        let b_shape = ext_vec(b_mode);
        let c_shape = ext_vec(c_mode);

        let expected = einsum_oracle(
            &OperandData {
                modes: a_mode,
                vals: &a_logical,
                shape: &a_shape,
            },
            &OperandData {
                modes: b_mode,
                vals: &b_logical,
                shape: &b_shape,
            },
            c_mode,
            exts,
        );
        for (c_idx, val) in expected {
            let flat = colex_index(&c_idx, &c_shape);
            let got = c.at(flat);
            assert!(
                (got - (c_init + val)).abs() < 1e-7,
                "{}: at {:?}, got {}, expected {}",
                $subscripts,
                c_idx,
                got,
                c_init + val
            );
        }
    }};
}

// ------------------------------------------------------------------
// All 18 contractions from examples/einsum_test.py
// ------------------------------------------------------------------

#[test]
fn test_einsum_gemm_and_batched_contractions() {
    // mk,nk->mn
    check_einsum!(
        "mk,nk->mn",
        Layout::from_shape(int_tuple![Int::<3>, Int::<5>]),
        Layout::from_shape(int_tuple![Int::<4>, Int::<5>]),
        Layout::from_shape(int_tuple![Int::<3>, Int::<4>]),
        [(b'm', 3), (b'n', 4), (b'k', 5)],
        0.0
    );

    // mk,nk->nm (transposed output)
    check_einsum!(
        "mk,nk->nm",
        Layout::from_shape(int_tuple![Int::<3>, Int::<5>]),
        Layout::from_shape(int_tuple![Int::<4>, Int::<5>]),
        Layout::from_shape(int_tuple![Int::<4>, Int::<3>]),
        [(b'm', 3), (b'n', 4), (b'k', 5)],
        0.0
    );

    // Strided operands
    check_einsum!(
        "mk,nk->mn",
        Layout::new(
            int_tuple![Int::<3>, Int::<5>],
            int_tuple![Int::<5>, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<5>],
            int_tuple![Int::<5>, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<3>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>]
        ),
        [(b'm', 3), (b'n', 4), (b'k', 5)],
        0.0
    );

    // Batched leading: bmk,bnk->bmn
    check_einsum!(
        "bmk,bnk->bmn",
        Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<5>]),
        Layout::from_shape(int_tuple![Int::<2>, Int::<4>, Int::<5>]),
        Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<4>]),
        [(b'b', 2), (b'm', 3), (b'n', 4), (b'k', 5)],
        0.0
    );

    // Batched trailing: mkb,nkb->mnb
    check_einsum!(
        "mkb,nkb->mnb",
        Layout::from_shape(int_tuple![Int::<3>, Int::<5>, Int::<2>]),
        Layout::from_shape(int_tuple![Int::<4>, Int::<5>, Int::<2>]),
        Layout::from_shape(int_tuple![Int::<3>, Int::<4>, Int::<2>]),
        [(b'b', 2), (b'm', 3), (b'n', 4), (b'k', 5)],
        0.0
    );

    // Multi-row and reduction: ijp,np->ijn
    check_einsum!(
        "ijp,np->ijn",
        Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<4>]),
        Layout::from_shape(int_tuple![Int::<5>, Int::<4>]),
        Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<5>]),
        [(b'i', 2), (b'j', 3), (b'p', 4), (b'n', 5)],
        0.0
    );

    // Whitepaper tensor contraction: stupr,qtru->stqp
    check_einsum!(
        "stupr,qtru->stqp",
        Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<2>, Int::<2>, Int::<3>]),
        Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<3>, Int::<2>]),
        Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<2>, Int::<2>]),
        [
            (b's', 2),
            (b't', 3),
            (b'u', 2),
            (b'p', 2),
            (b'r', 3),
            (b'q', 2)
        ],
        0.0
    );

    // Accumulates into non-zero C
    check_einsum!(
        "mk,nk->mn",
        Layout::from_shape(int_tuple![Int::<2>, Int::<3>]),
        Layout::from_shape(int_tuple![Int::<2>, Int::<3>]),
        Layout::from_shape(int_tuple![Int::<2>, Int::<2>]),
        [(b'm', 2), (b'n', 2), (b'k', 3)],
        10.0
    );
}

#[test]
fn test_einsum_classic_vector_matrix_and_scalar_products() {
    // Inner product: i,i->
    check_einsum!(
        "i,i->",
        Layout::from_shape(Int::<6>),
        Layout::from_shape(Int::<6>),
        Layout::new(Nil, Nil),
        [(b'i', 6)],
        0.0
    );

    // Outer product: i,j->ij
    check_einsum!(
        "i,j->ij",
        Layout::from_shape(Int::<3>),
        Layout::from_shape(Int::<4>),
        Layout::from_shape(int_tuple![Int::<3>, Int::<4>]),
        [(b'i', 3), (b'j', 4)],
        0.0
    );

    // Elementwise Hadamard 1D: i,i->i
    check_einsum!(
        "i,i->i",
        Layout::from_shape(Int::<6>),
        Layout::from_shape(Int::<6>),
        Layout::from_shape(Int::<6>),
        [(b'i', 6)],
        0.0
    );

    // Elementwise Hadamard 2D: ij,ij->ij
    check_einsum!(
        "ij,ij->ij",
        Layout::from_shape(int_tuple![Int::<3>, Int::<4>]),
        Layout::from_shape(int_tuple![Int::<3>, Int::<4>]),
        Layout::from_shape(int_tuple![Int::<3>, Int::<4>]),
        [(b'i', 3), (b'j', 4)],
        0.0
    );

    // Scalar scaling: i,->i
    check_einsum!(
        "i,->i",
        Layout::from_shape(Int::<5>),
        Layout::new(Nil, Nil),
        Layout::from_shape(Int::<5>),
        [(b'i', 5)],
        0.0
    );

    // Scalar matrix: mn,->mn
    check_einsum!(
        "mn,->mn",
        Layout::from_shape(int_tuple![Int::<5>, Int::<6>]),
        Layout::new(Nil, Nil),
        Layout::from_shape(int_tuple![Int::<5>, Int::<6>]),
        [(b'm', 5), (b'n', 6)],
        0.0
    );

    // Scalar transpose: mn,->nm
    check_einsum!(
        "mn,->nm",
        Layout::from_shape(int_tuple![Int::<5>, Int::<6>]),
        Layout::new(Nil, Nil),
        Layout::from_shape(int_tuple![Int::<6>, Int::<5>]),
        [(b'm', 5), (b'n', 6)],
        0.0
    );

    // Matrix-vector: ij,j->i
    check_einsum!(
        "ij,j->i",
        Layout::from_shape(int_tuple![Int::<3>, Int::<4>]),
        Layout::from_shape(Int::<4>),
        Layout::from_shape(Int::<3>),
        [(b'i', 3), (b'j', 4)],
        0.0
    );

    // Vector-matrix: i,ij->j
    check_einsum!(
        "i,ij->j",
        Layout::from_shape(Int::<3>),
        Layout::from_shape(int_tuple![Int::<3>, Int::<4>]),
        Layout::from_shape(Int::<4>),
        [(b'i', 3), (b'j', 4)],
        0.0
    );

    // Batched matrix-vector: bij,bj->bi
    check_einsum!(
        "bij,bj->bi",
        Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<4>]),
        Layout::from_shape(int_tuple![Int::<2>, Int::<4>]),
        Layout::from_shape(int_tuple![Int::<2>, Int::<3>]),
        [(b'b', 2), (b'i', 3), (b'j', 4)],
        0.0
    );
}

#[test]
fn test_einsum_fold_operand_structure() {
    // Fold A_stupr into (M=s,p; K=u,r; L=t)
    const SPEC_WHITEPAPER: EinsumSpec = parse_einsum("stupr,qtru->stqp");
    const FOLD_A: OperandFold = SPEC_WHITEPAPER.a_fold;
    let buf = [0.0f64; 2 * 3 * 2 * 4 * 5];
    let a = make_tensor(
        &buf,
        Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<2>, Int::<4>, Int::<5>]),
    );
    let a3 = fold_operand::<FOLD_A, _>(a);
    assert_eq!(a3.rank(), 3);
    assert_eq!(a3.layout.shape.0.size(), 2 * 4);
    assert_eq!(a3.layout.shape.1.0.size(), 2 * 5);
    assert_eq!(a3.layout.shape.1.1.0.size(), 3);

    // Empty batch group produces size-1 batch mode with stride 0
    const SPEC_GEMM: EinsumSpec = parse_einsum("mk,nk->mn");
    const FOLD_MK: OperandFold = SPEC_GEMM.a_fold;
    let buf2 = [0.0f64; 15];
    let mk = make_tensor(&buf2, Layout::from_shape(int_tuple![Int::<3>, Int::<5>]));
    let mk3 = fold_operand::<FOLD_MK, _>(mk);
    assert_eq!(mk3.layout.shape.1.1.0, Int::<1>);
    assert_eq!(mk3.layout.stride.1.1.0, Int::<0>);
}

#[test]
fn test_einsum_parse_and_dynamic_extent_errors() {
    for bad in [
        "mk,nk",         // missing arrow
        "mk,nk,xy->mn",  // three inputs
        "mk->mn",        // one input
        "mkx,nk->mn",    // single-operand label 'x'
        "mk,nk->mnx",    // output-only label 'x'
        "mm,nm->mn",     // repeated label in A
        "mk,nn->mn",     // repeated label in B
        "mk,nk->mm",     // repeated label in C
        "mk,nk->mn->ab", // multiple arrows
    ] {
        assert!(
            catch_unwind(|| parse_einsum(bad)).is_err(),
            "expected parse_einsum({bad:?}) to panic"
        );
    }

    // Dynamic single-mode extent mismatch (k: 5 vs 6)
    let a_buf = [0.0f64; 30];
    let b_buf = [0.0f64; 30];
    let mut c_buf = [0.0f64; 30];
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let a = make_tensor(&a_buf[..], Layout::from_shape(int_tuple![3i32, 5i32]));
            let b = make_tensor(&b_buf[..], Layout::from_shape(int_tuple![4i32, 6i32]));
            let mut c = make_tensor(&mut c_buf[..], Layout::from_shape(int_tuple![3i32, 4i32]));
            einsum!("mk,nk->mn", a, b, &mut c);
        }))
        .is_err()
    );

    // Dynamic multi-mode per-label extent mismatch: (2, 3) vs (3, 2) have same product 6,
    // so only per-label validation catches the disagreement!
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let a = make_tensor(&a_buf[..], Layout::from_shape(int_tuple![2i32, 2i32, 3i32]));
            let b = make_tensor(&b_buf[..], Layout::from_shape(int_tuple![2i32, 3i32, 2i32]));
            let mut c = make_tensor(&mut c_buf[..], Layout::from_shape(int_tuple![2i32, 2i32]));
            einsum!("mkr,nkr->mn", a, b, &mut c);
        }))
        .is_err()
    );
}

#[test]
fn test_einsum_const_fold() {
    const fn const_folded_sizes() -> (int, int, int) {
        const SPEC: EinsumSpec = parse_einsum("stupr,qtru->stqp");
        const FOLD_A: OperandFold = SPEC.a_fold;
        let a = Tensor::new(
            noodle_cute::tensor::ImplicitAccessor::new(Int::<0>),
            Layout::from_shape(int_tuple![Int::<2>, Int::<3>, Int::<2>, Int::<4>, Int::<5>]),
        );
        let a3 = fold_operand::<FOLD_A, _>(a);
        (
            a3.layout.shape.0.size(),
            a3.layout.shape.1.0.size(),
            a3.layout.shape.1.1.0.size(),
        )
    }
    const SIZES: (int, int, int) = const_folded_sizes();
    assert_eq!(SIZES, (8, 10, 3));
}
