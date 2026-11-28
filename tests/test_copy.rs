#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use noodle_cute::algebra::{coalesce, greatest_common_domain, nullspace};
use noodle_cute::algorithm::copy::{copy, copy_plan};
use noodle_cute::algorithm::reference::copy as copy_ref;
use noodle_cute::integer::{Int, int};
use noodle_cute::layout::Layout;
use noodle_cute::shape::Shape;
use noodle_cute::stride::Stride;
use noodle_cute::tensor::{
    Accessor, AccessorBase, AccessorOffset, CellAccessor, ImplicitAccessor, LinearOffset,
    MutableAccessor, Tensor, TensorIndex, TensorSet, make_owned_tensor, make_tensor,
};
use noodle_cute::{int_tuple, tuple};

/// Accessor wrapper recording every absolute element offset written through `store`.
#[derive(Debug)]
struct LoggingAccessor<'a, T> {
    inner: CellAccessor<'a, T>,
    log: &'a RefCell<Vec<int>>,
    base: int,
}

impl<T> Clone for LoggingAccessor<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for LoggingAccessor<'_, T> {}

impl<T> AccessorBase for LoggingAccessor<'_, T> {}

impl<'a, T> LoggingAccessor<'a, T> {
    fn new(buf: &'a mut [T], log: &'a RefCell<Vec<int>>) -> Self {
        Self {
            inner: CellAccessor::from_slice_mut(buf),
            log,
            base: 0,
        }
    }
}

impl<'a, T, Off: LinearOffset> AccessorOffset<Off> for LoggingAccessor<'a, T>
where
    CellAccessor<'a, T>: AccessorOffset<Off, Output = CellAccessor<'a, T>>,
{
    type Output = Self;

    fn offset(&self, delta: Off) -> Self::Output {
        Self {
            inner: self.inner.offset(delta),
            log: self.log,
            base: self.base + delta.linear_offset(),
        }
    }
}

impl<'a, T: Copy, Off: LinearOffset> Accessor<Off> for LoggingAccessor<'a, T>
where
    CellAccessor<'a, T>: Accessor<Off, Item = T>,
{
    type Item = T;

    fn load(&self, offset: Off) -> Self::Item {
        self.inner.load(offset)
    }
}

impl<'a, T: Copy, Off: LinearOffset> MutableAccessor<Off> for LoggingAccessor<'a, T>
where
    CellAccessor<'a, T>: MutableAccessor<Off, Item = T>,
{
    fn store(&mut self, offset: Off, value: Self::Item) {
        self.log
            .borrow_mut()
            .push(self.base + offset.linear_offset());
        self.inner.store(offset, value);
    }
}

fn fill_iota<A, S, D>(t: &mut Tensor<A, Layout<S, D>>)
where
    S: Shape,
    D: Stride<S>,
    Tensor<A, Layout<S, D>>: TensorSet<int, Item = f64>,
{
    for i in 0..t.size() {
        t.set(i, i as f64);
    }
}

fn flat_values<A, S, D>(t: &Tensor<A, Layout<S, D>>) -> Vec<f64>
where
    S: Shape,
    D: Stride<S>,
    Tensor<A, Layout<S, D>>: TensorIndex<int, Output = f64>,
{
    (0..t.size()).map(|i| t.at(i)).collect()
}

macro_rules! check_matches_reference {
    ($label:expr, $src_layout:expr, $dst_layout:expr) => {{
        let src_l = $src_layout;
        let dst_l = $dst_layout;
        let mut src_buf = vec![0.0f64; src_l.cosize() as usize];
        let mut src = make_tensor(&mut src_buf[..], src_l);
        fill_iota(&mut src);

        let mut ref_buf = vec![-1.0f64; dst_l.cosize() as usize];
        let mut opt_buf = vec![-1.0f64; dst_l.cosize() as usize];
        {
            let mut dst_ref = make_tensor(&mut ref_buf[..], dst_l);
            let mut dst_opt = make_tensor(&mut opt_buf[..], dst_l);
            copy_ref(src, &mut dst_ref);
            copy(src, &mut dst_opt);

            assert_eq!(
                flat_values(&dst_ref),
                flat_values(&dst_opt),
                "logical values diverged for {}",
                $label
            );
        }
        assert_eq!(
            ref_buf, opt_buf,
            "backing memory diverged (untouched slots disturbed) for {}",
            $label
        );
    }};
}

macro_rules! check_vector_dispatch {
    ($label:expr, $src_layout:expr, $dst_layout:expr, 1) => {{
        let src_t = Tensor::new(ImplicitAccessor::new(Int::<0>), $src_layout);
        let dst_t = Tensor::new(ImplicitAccessor::new(Int::<0>), $dst_layout);
        let (src_v, dst_v) = copy_plan(src_t, dst_t);
        let src_vec = src_v.at(tuple![.., 0i32]).layout;
        let dst_vec = dst_v.at(tuple![.., 0i32]).layout;
        assert_eq!(src_vec.size(), 1, "src_vec size mismatch for {}", $label);
        assert_eq!(dst_vec.size(), 1, "dst_vec size mismatch for {}", $label);
    }};
    ($label:expr, $src_layout:expr, $dst_layout:expr, $expect:literal) => {{
        let src_t = Tensor::new(ImplicitAccessor::new(Int::<0>), $src_layout);
        let dst_t = Tensor::new(ImplicitAccessor::new(Int::<0>), $dst_layout);
        let (src_v, dst_v) = copy_plan(src_t, dst_t);
        let src_vec = src_v.at(tuple![.., 0i32]).layout;
        let dst_vec = dst_v.at(tuple![.., 0i32]).layout;
        assert_eq!(
            src_vec.size(),
            $expect,
            "src_vec size mismatch for {}",
            $label
        );
        assert_eq!(
            dst_vec.size(),
            $expect,
            "dst_vec size mismatch for {}",
            $label
        );
        let c_src = coalesce(src_vec);
        let c_dst = coalesce(dst_vec);
        assert_eq!(c_src, c_dst, "coalesced vector run mismatch for {}", $label);
        assert_eq!(
            c_src.stride, Int::<1>,
            "coalesced vector run must have unit stride for {}",
            $label
        );
    }};
}

// ------------------------------------------------------------------
// Reference copy tests (mirroring TestCopyRef)
// ------------------------------------------------------------------

#[test]
fn test_copy_ref_memcpy() {
    let l = Layout::new(Int::<8>, Int::<1>);
    let mut src_buf = [0.0f64; 8];
    let mut dst_buf = [0.0f64; 8];
    let mut src = make_tensor(&mut src_buf, l);
    fill_iota(&mut src);
    let mut dst = make_tensor(&mut dst_buf, l);
    copy_ref(src, &mut dst);
    assert_eq!(flat_values(&src), flat_values(&dst));
}

#[test]
fn test_copy_ref_transpose() {
    let src_l = Layout::new(
        int_tuple![Int::<4>, Int::<8>],
        int_tuple![Int::<8>, Int::<1>],
    );
    let dst_l = Layout::new(
        int_tuple![Int::<4>, Int::<8>],
        int_tuple![Int::<1>, Int::<4>],
    );
    let mut src_buf = [0.0f64; 32];
    let mut dst_buf = [0.0f64; 32];
    let mut src = make_tensor(&mut src_buf, src_l);
    fill_iota(&mut src);
    let mut dst = make_tensor(&mut dst_buf, dst_l);
    copy_ref(src, &mut dst);

    for i in 0..src.size() {
        assert_eq!(dst.at(i), src.at(i));
    }
    assert_ne!(src_buf, dst_buf);
}

#[test]
fn test_copy_ref_gather_scatter() {
    let sparse = Layout::new(
        int_tuple![Int::<2>, Int::<3>],
        int_tuple![Int::<4>, Int::<1>],
    );
    let dense = Layout::new(Int::<6>, Int::<1>);

    let mut src_buf = [0.0f64; 7];
    let mut dst_buf = [0.0f64; 6];
    let mut src = make_tensor(&mut src_buf, sparse);
    fill_iota(&mut src);
    let mut dst = make_tensor(&mut dst_buf, dense);
    copy_ref(src, &mut dst);
    assert_eq!(flat_values(&src), flat_values(&dst));

    let mut src2_buf = [0.0f64; 6];
    let mut dst2_buf = [0.0f64; 7];
    let mut src2 = make_tensor(&mut src2_buf, dense);
    fill_iota(&mut src2);
    let mut dst2 = make_tensor(&mut dst2_buf, sparse);
    copy_ref(src2, &mut dst2);
    assert_eq!(flat_values(&src2), flat_values(&dst2));
}

#[test]
fn test_copy_ref_broadcast() {
    let mut src_buf = [42.0f64; 1];
    let mut dst_buf = [0.0f64; 7];
    let src = make_tensor(&mut src_buf, Layout::new(Int::<7>, Int::<0>));
    let mut dst = make_tensor(&mut dst_buf, Layout::new(Int::<7>, Int::<1>));
    copy_ref(src, &mut dst);
    assert_eq!(flat_values(&dst), vec![42.0; 7]);
}

#[test]
fn test_copy_ref_dynamic_size_mismatch_panics() {
    let src_buf = [0.0f64; 4];
    let mut dst_buf = [0.0f64; 8];
    let src = make_tensor(&src_buf, Layout::new(4i32, Int::<1>));
    let mut dst = make_tensor(&mut dst_buf, Layout::new(8i32, Int::<1>));
    let res = catch_unwind(AssertUnwindSafe(|| {
        copy_ref(src, &mut dst);
    }));
    assert!(res.is_err());
}

#[test]
fn test_copy_ref_owned_and_empty() {
    let mut src = make_owned_tensor::<i32, _>(int_tuple![Int::<2>, Int::<3>]);
    let mut dst = make_owned_tensor::<i32, _>(Layout::from_shape_right(int_tuple![Int::<2>, Int::<3>]));
    src.as_mut_slice()
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = (i + 1) as i32);
    copy_ref(src, &mut dst);
    for i in 0..6 {
        assert_eq!(dst.at(i), src.at(i));
    }

    let empty_src: [i32; 0] = [];
    let mut empty_dst: [i32; 0] = [];
    let s0 = make_tensor(&empty_src[..], Layout::new(0i32, Int::<1>));
    let mut d0 = make_tensor(&mut empty_dst[..], Layout::new(0i32, Int::<1>));
    copy_ref(s0, &mut d0);
}

// ------------------------------------------------------------------
// Optimized copy tests (mirroring TestCopyOpt)
// ------------------------------------------------------------------

#[test]
fn test_copy_opt_matches_reference_on_all_15_applications() {
    check_matches_reference!(
        "memcpy",
        Layout::new(Int::<8>, Int::<1>),
        Layout::new(Int::<8>, Int::<1>)
    );
    check_matches_reference!(
        "transpose",
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );
    check_matches_reference!(
        "gather",
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<4>, Int::<1>]
        ),
        Layout::new(Int::<6>, Int::<1>)
    );
    check_matches_reference!(
        "scatter",
        Layout::new(Int::<6>, Int::<1>),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<4>, Int::<1>]
        )
    );
    check_matches_reference!(
        "broadcast",
        Layout::new(Int::<7>, Int::<0>),
        Layout::new(Int::<7>, Int::<1>)
    );
    check_matches_reference!(
        "constant",
        Layout::new(Int::<7>, Int::<0>),
        Layout::new(Int::<7>, Int::<0>)
    );
    check_matches_reference!(
        "tensor transpose",
        Layout::new(
            int_tuple![Int::<4>, Int::<6>],
            int_tuple![Int::<1>, Int::<4>]
        ),
        Layout::new(
            int_tuple![Int::<4>, (Int::<2>, Int::<3>)],
            int_tuple![Int::<1>, (Int::<14>, Int::<4>)]
        )
    );
    check_matches_reference!(
        "col-major",
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );
    check_matches_reference!(
        "row-major",
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>]
        )
    );
    check_matches_reference!(
        "part bcast",
        Layout::new(
            int_tuple![Int::<5>, Int::<4>],
            int_tuple![Int::<0>, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<5>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>]
        )
    );
    check_matches_reference!(
        "hierarchical",
        Layout::new(
            int_tuple![Int::<4>, (Int::<2>, Int::<3>)],
            int_tuple![Int::<1>, (Int::<4>, Int::<12>)]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<6>],
            int_tuple![Int::<1>, Int::<4>]
        )
    );
    check_matches_reference!(
        "paper hier",
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<2>)],
            int_tuple![(Int::<8>, Int::<2>), (Int::<4>, Int::<1>)]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<2>)],
            int_tuple![(Int::<4>, Int::<2>), (Int::<8>, Int::<1>)]
        )
    );
    check_matches_reference!(
        "subdomain",
        Layout::new(
            int_tuple![Int::<4>, Int::<3>, Int::<5>],
            int_tuple![Int::<1>, Int::<7>, Int::<42>]
        ),
        Layout::new(
            int_tuple![Int::<6>, Int::<10>],
            int_tuple![Int::<1>, Int::<9>]
        )
    );
    check_matches_reference!(
        "coprime",
        Layout::new(
            int_tuple![Int::<5>, Int::<7>],
            int_tuple![Int::<7>, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<7>, Int::<5>],
            int_tuple![Int::<5>, Int::<1>]
        )
    );
    check_matches_reference!(
        "rank1 to 2",
        Layout::new(Int::<12>, Int::<1>),
        Layout::new(
            int_tuple![Int::<3>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>]
        )
    );
}

#[test]
fn test_copy_opt_dynamic_strides_and_size_mismatch() {
    // Dynamic source strides with static shapes and destination strides succeed and match reference.
    let src_buf = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut dst_buf = [0.0f64; 6];
    let src = make_tensor(
        &src_buf,
        Layout::new(int_tuple![Int::<2>, Int::<3>], int_tuple![1i32, 2i32]),
    );
    let mut dst = make_tensor(
        &mut dst_buf,
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<3>, Int::<1>],
        ),
    );
    copy(src, &mut dst);
    let mut ref_buf = [0.0f64; 6];
    let mut dst_ref = make_tensor(
        &mut ref_buf,
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<3>, Int::<1>],
        ),
    );
    copy_ref(src, &mut dst_ref);
    assert_eq!(dst_buf, ref_buf);

    // Dynamic trailing mode matching static source size succeeds (size 2 == 2 * 1).
    let s2_buf = [10.0f64, 20.0];
    let s2 = make_tensor(&s2_buf, Layout::new(Int::<2>, Int::<1>));
    let mut d2_ok_buf = [0.0f64; 2];
    let mut d2_ok = make_tensor(
        &mut d2_ok_buf,
        Layout::new(int_tuple![Int::<2>, 1i32], int_tuple![Int::<1>, Int::<2>]),
    );
    copy(s2, &mut d2_ok);
    assert_eq!(d2_ok_buf, s2_buf);

    // Dynamic trailing mode with mismatched size panics (size 2 != 2 * 3).
    let mut d2_bad_buf = [0.0f64; 6];
    let mut d2_bad = make_tensor(
        &mut d2_bad_buf,
        Layout::new(int_tuple![Int::<2>, 3i32], int_tuple![Int::<1>, Int::<2>]),
    );
    let res = catch_unwind(AssertUnwindSafe(|| {
        copy(s2, &mut d2_bad);
    }));
    assert!(res.is_err());
}

#[test]
fn test_copy_opt_constant_fill_drops_redundant_writes() {
    let l = Layout::new(Int::<7>, Int::<0>);
    assert_eq!(nullspace(l), Layout::new(Int::<7>, Int::<1>));

    let src_buf = [99.0f64; 1];
    let src = make_tensor(&src_buf, l);

    let ref_log = RefCell::new(Vec::new());
    let mut ref_buf = [0.0f64; 1];
    let mut dst_ref = Tensor::new(LoggingAccessor::new(&mut ref_buf, &ref_log), l);
    copy_ref(src, &mut dst_ref);

    let opt_log = RefCell::new(Vec::new());
    let mut opt_buf = [0.0f64; 1];
    let mut dst_opt = Tensor::new(LoggingAccessor::new(&mut opt_buf, &opt_log), l);
    copy(src, &mut dst_opt);

    assert_eq!(*ref_log.borrow(), vec![0; 7]);
    assert_eq!(*opt_log.borrow(), vec![0]);
    assert_eq!(flat_values(&dst_ref), flat_values(&dst_opt));
}

#[test]
fn test_copy_opt_dynamic_write_after_write_rejected() {
    let src_buf = [0.0f64, 1.0, 2.0, 3.0];
    let src = make_tensor(&src_buf, Layout::new(Int::<4>, 1i32));

    let mut dst_opt_buf = [0.0f64; 1];
    let mut dst_opt = make_tensor(&mut dst_opt_buf, Layout::new(Int::<4>, Int::<0>));
    let err = catch_unwind(AssertUnwindSafe(|| {
        copy(src, &mut dst_opt);
    }))
    .expect_err("dynamic WAW must panic");
    let msg = err
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| err.downcast_ref::<String>().map(|s| s.as_str()))
        .unwrap_or("");
    assert!(
        msg.to_ascii_lowercase().contains("write-after-write"),
        "unexpected panic message: {msg}"
    );

    // Reference copy has no WAW check and leaves the last write standing.
    let mut dst_ref_buf = [0.0f64; 1];
    let mut dst_ref = make_tensor(&mut dst_ref_buf, Layout::new(Int::<4>, Int::<0>));
    copy_ref(src, &mut dst_ref);
    assert_eq!(dst_ref.at(0i32), 3.0);
}

#[test]
fn test_copy_plan_all_12_stage3_vector_dispatch_rows() {
    check_vector_dispatch!(
        "memcpy",
        Layout::new(Int::<32>, Int::<1>),
        Layout::new(Int::<32>, Int::<1>),
        32
    );
    check_vector_dispatch!(
        "col-major",
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>]
        ),
        32
    );
    check_vector_dispatch!(
        "row-major",
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>]
        ),
        32
    );
    check_vector_dispatch!(
        "transpose",
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<8>, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<8>],
            int_tuple![Int::<1>, Int::<4>]
        ),
        1
    );
    check_vector_dispatch!(
        "broadcast",
        Layout::new(Int::<7>, Int::<0>),
        Layout::new(Int::<7>, Int::<1>),
        1
    );
    check_vector_dispatch!(
        "constant",
        Layout::new(Int::<7>, Int::<0>),
        Layout::new(Int::<7>, Int::<0>),
        1
    );
    check_vector_dispatch!(
        "part bcast",
        Layout::new(
            int_tuple![Int::<5>, Int::<4>],
            int_tuple![Int::<0>, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<5>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>]
        ),
        4
    );
    check_vector_dispatch!(
        "hierarchical",
        Layout::new(
            int_tuple![Int::<4>, (Int::<2>, Int::<3>)],
            int_tuple![Int::<1>, (Int::<4>, Int::<12>)]
        ),
        Layout::new(
            int_tuple![Int::<4>, Int::<6>],
            int_tuple![Int::<1>, Int::<4>]
        ),
        8
    );
    check_vector_dispatch!(
        "paper hier",
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<2>)],
            int_tuple![(Int::<8>, Int::<2>), (Int::<4>, Int::<1>)]
        ),
        Layout::new(
            int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<2>)],
            int_tuple![(Int::<4>, Int::<2>), (Int::<8>, Int::<1>)]
        ),
        4
    );
    check_vector_dispatch!(
        "subdomain",
        Layout::new(
            int_tuple![Int::<4>, Int::<3>, Int::<5>],
            int_tuple![Int::<1>, Int::<7>, Int::<42>]
        ),
        Layout::new(
            int_tuple![Int::<6>, Int::<10>],
            int_tuple![Int::<1>, Int::<9>]
        ),
        2
    );
    check_vector_dispatch!(
        "coprime",
        Layout::new(
            int_tuple![Int::<5>, Int::<7>],
            int_tuple![Int::<7>, Int::<1>]
        ),
        Layout::new(
            int_tuple![Int::<7>, Int::<5>],
            int_tuple![Int::<5>, Int::<1>]
        ),
        1
    );
    check_vector_dispatch!(
        "scatter",
        Layout::new(Int::<6>, Int::<1>),
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<4>, Int::<1>]
        ),
        1
    );
}

#[test]
fn test_copy_opt_destination_written_in_ascending_runs() {
    let src_l = Layout::new(Int::<6>, Int::<1>);
    let dst_l = Layout::new(
        int_tuple![Int::<2>, Int::<3>],
        int_tuple![Int::<4>, Int::<1>],
    );

    let src_buf = [0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    let src = make_tensor(&src_buf, src_l);

    let ref_log = RefCell::new(Vec::new());
    let mut ref_buf = [-1.0f64; 7];
    let mut dst_ref = Tensor::new(LoggingAccessor::new(&mut ref_buf, &ref_log), dst_l);
    copy_ref(src, &mut dst_ref);

    let opt_log = RefCell::new(Vec::new());
    let mut opt_buf = [-1.0f64; 7];
    let mut dst_opt = Tensor::new(LoggingAccessor::new(&mut opt_buf, &opt_log), dst_l);
    copy(src, &mut dst_opt);

    let ref_offsets = ref_log.borrow().clone();
    let opt_offsets = opt_log.borrow().clone();
    assert_eq!(ref_offsets, vec![0, 4, 1, 5, 2, 6]);
    assert_eq!(opt_offsets, vec![0, 1, 2, 4, 5, 6]);

    let mut sorted_ref = ref_offsets;
    sorted_ref.sort();
    assert_eq!(sorted_ref, opt_offsets);
    assert!(!opt_offsets.contains(&3));
    assert_eq!(opt_buf[3], -1.0);
}

#[test]
fn test_copy_opt_incompatible_shapes_fallback_to_elementwise() {
    let src_l = Layout::new(
        int_tuple![Int::<5>, Int::<7>],
        int_tuple![Int::<7>, Int::<1>],
    );
    let dst_l = Layout::new(
        int_tuple![Int::<7>, Int::<5>],
        int_tuple![Int::<5>, Int::<1>],
    );

    let gcd = greatest_common_domain(src_l, dst_l);
    assert_eq!(gcd.size(), 1);
    assert_eq!(gcd.rank(), 1);

    let mut src_buf = [0.0f64; 35];
    let mut src = make_tensor(&mut src_buf, src_l);
    fill_iota(&mut src);

    let ref_log = RefCell::new(Vec::new());
    let mut ref_buf = [0.0f64; 35];
    let mut dst_ref = Tensor::new(LoggingAccessor::new(&mut ref_buf, &ref_log), dst_l);
    copy_ref(src, &mut dst_ref);

    let opt_log = RefCell::new(Vec::new());
    let mut opt_buf = [0.0f64; 35];
    let mut dst_opt = Tensor::new(LoggingAccessor::new(&mut opt_buf, &opt_log), dst_l);
    copy(src, &mut dst_opt);

    assert_eq!(*opt_log.borrow(), *ref_log.borrow());
    assert_eq!(opt_log.borrow().len(), 35);
    assert_eq!(flat_values(&dst_ref), flat_values(&dst_opt));
}

#[test]
fn test_copy_plan_const_eval() {
    const fn const_vec_width() -> int {
        let src = Tensor::new(
            ImplicitAccessor::new(Int::<0>),
            Layout::new(
                int_tuple![Int::<4>, (Int::<2>, Int::<3>)],
                int_tuple![Int::<1>, (Int::<4>, Int::<12>)],
            ),
        );
        let dst = Tensor::new(
            ImplicitAccessor::new(Int::<0>),
            Layout::new(
                int_tuple![Int::<4>, Int::<6>],
                int_tuple![Int::<1>, Int::<4>],
            ),
        );
        let (src_v, _) = copy_plan(src, dst);
        src_v.at(tuple![.., 0i32]).layout.size()
    }

    const VEC: int = const_vec_width();
    assert_eq!(VEC, 8);
}

#[test]
fn conflicting_nonzero_strides_are_rejected_before_writing() {
    let sh = int_tuple![Int::<2>, Int::<2>];
    let src = Tensor::new(
        ImplicitAccessor::new(10i32),
        Layout::new(sh, int_tuple![1i32, 2i32]),
    );
    let mut out = [77; 3];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut dst = make_tensor(&mut out, Layout::new(sh, int_tuple![Int::<1>, Int::<1>]));
        copy(src, &mut dst);
    }));
    assert!(result.is_err());
    assert_eq!(out, [77; 3]);
}

#[test]
fn opposing_source_strides_in_broadcast_are_rejected_before_writing() {
    let sh = int_tuple![Int::<2>, Int::<2>];
    let src = Tensor::new(
        ImplicitAccessor::new(10i32),
        Layout::new(sh, int_tuple![1i32, -1i32]),
    );
    let mut out = [77];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut dst = make_tensor(&mut out, Layout::new(sh, int_tuple![Int::<0>, Int::<0>]));
        copy(src, &mut dst);
    }));
    assert!(result.is_err());
    assert_eq!(out, [77]);
}
