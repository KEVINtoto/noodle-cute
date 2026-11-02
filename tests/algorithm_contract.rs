//! Contract tests for the tensor algorithms, following CuTe's
//! `algorithm/copy.hpp`, `fill.hpp`, `clear.hpp`, `axpby.hpp`,
//! `tensor_algorithms.hpp`, and `gemm.hpp`.
//!
//! Vector widths follow upstream's `AutoVectorizingCopyWithAssumedAlignment`
//! rule, and the `gemm` traversal orders are upstream's register-reuse paths.

use core::sync::atomic::{AtomicI32, Ordering};

use noodle_cute::algorithm::{
    VECTOR_BITS, axpby, clear, copy, copy_if, fill, for_each, gemm, gemm_fma, transform,
    transform_into, transform_zip,
};
use noodle_cute::atom::{MmaAtom, MmaTraits, UniversalFma};
use noodle_cute::coord::elem_less;
use noodle_cute::layout::{LayoutLeft, LayoutRight};
use noodle_cute::pointer::{Read, Write};
use noodle_cute::swizzle::Swizzle;
use noodle_cute::tensor::{make_identity_tensor, make_owned_tensor, make_tensor};
use noodle_cute::{DynInt, Int, IntTuple, Layout, Nil, shape, stride};

type Col<const M: isize, const N: isize> =
    Layout<IntTuple<Int<M>, IntTuple<Int<N>, Nil>>, IntTuple<Int<1>, IntTuple<Int<M>, Nil>>>;
type Strided =
    Layout<IntTuple<Int<4>, IntTuple<Int<3>, Nil>>, IntTuple<Int<1>, IntTuple<Int<5>, Nil>>>;
type Dynamic =
    Layout<IntTuple<Int<4>, IntTuple<Int<8>, Nil>>, IntTuple<Int<1>, IntTuple<DynInt, Nil>>>;
type Smem = noodle_cute::layout::ComposedLayout<Swizzle<3, 3, 3>, Int<0>, Col<64, 8>>;

#[test]
fn copy_vector_width_follows_upstream() {
    // gcd(common * bits, alignment, 128) when wider than one element.
    assert_eq!(VECTOR_BITS::<Col<4, 8>, Col<4, 8>, f32>, 128);
    assert_eq!(VECTOR_BITS::<Col<4, 8>, Col<4, 8>, f64>, 128);
    assert_eq!(VECTOR_BITS::<Col<2, 1>, Col<2, 1>, u8>, 16);
    // Six contiguous u16 are 96 bits: the vector is the largest power of two dividing them.
    assert_eq!(VECTOR_BITS::<Col<6, 1>, Col<6, 1>, u16>, 32);
    // Three are 48 bits: 16-bit vectors are no wider than one element.
    assert_eq!(VECTOR_BITS::<Col<3, 1>, Col<3, 1>, u16>, 0);
    // Columns start every 5 elements: no common alignment beyond one element.
    assert_eq!(VECTOR_BITS::<Strided, Strided, f32>, 0);
    // Dynamic strides only assume byte alignment.
    assert_eq!(VECTOR_BITS::<Dynamic, Col<4, 8>, f32>, 0);
    // A swizzle keeps 2^M = 8 contiguous elements.
    assert_eq!(VECTOR_BITS::<Smem, Col<64, 8>, u16>, 128);
    assert_eq!(VECTOR_BITS::<Smem, Col<64, 8>, f64>, 128);
    assert_eq!(VECTOR_BITS::<Col<4, 8>, Col<4, 8>, u128>, 0);
}

#[test]
fn copy_handles_vectors_misalignment_and_swizzles() {
    let layout = LayoutLeft::make(shape![Int::<4>, Int::<8>]).unwrap();
    let data: [f32; 33] = core::array::from_fn(|i| i as f32);
    let mut out = [0f32; 33];
    // A view starting one element in is not 16-byte aligned; the copy falls back.
    let src = make_tensor(data.as_ptr().wrapping_add(1), layout);
    let dst = make_tensor(out.as_mut_ptr().wrapping_add(1), layout);
    // SAFETY: both views cover elements 1..33 of their arrays.
    unsafe { copy(&src, &dst) };
    assert_eq!(out[1..], data[1..]);

    let tile = LayoutLeft::make(shape![Int::<64>, Int::<8>]).unwrap();
    let mut plain = make_owned_tensor::<u16, _>(tile);
    plain
        .engine
        .0
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = i as u16);
    let mut smem = [0u16; 512];
    let swizzled = make_tensor(smem.as_mut_ptr(), Swizzle::<3, 3, 3>.compose(tile));
    // SAFETY: the swizzle permutes 0..512.
    unsafe { copy(&plain.view(), &swizzled) };
    for i in 0..512usize {
        assert_eq!(
            smem[Swizzle::<3, 3, 3>::apply(i as isize) as usize],
            i as u16
        );
    }
}

#[test]
fn copy_if_predicates_with_an_identity_tensor() {
    // A 4x4 tile at block (1, 1) of a 6x6 matrix: only rows and columns 4, 5 exist.
    let extent = shape![DynInt(6), DynInt(6)];
    let tile = shape![Int::<4>, Int::<4>];
    let data: [i32; 36] = core::array::from_fn(|i| i as i32);
    let src = make_tensor(data.as_ptr(), LayoutLeft::make(extent).unwrap())
        .local_tile(tile, shape![1i32, 1i32])
        .unwrap();
    let coords = make_identity_tensor(extent)
        .local_tile(tile, shape![1i32, 1i32])
        .unwrap();
    let mut out = [-1i64; 16];
    let dst = make_tensor(out.as_mut_ptr(), LayoutLeft::make(tile).unwrap());
    // SAFETY: only predicated elements, which lie inside `data`, are read.
    unsafe { copy_if(|i| elem_less(&coords.get(i as i32), &extent), &src, &dst) };
    assert_eq!(out[..2], [28, 29]);
    assert_eq!(out[4..6], [34, 35]);
    assert!(out[2..4].iter().chain(&out[6..]).all(|&x| x == -1));
}

#[test]
fn elementwise_algorithms() {
    let layout = LayoutRight::make(shape![Int::<2>, Int::<3>]).unwrap();
    let mut x = make_owned_tensor::<f32, _>(layout);
    let mut y = make_owned_tensor::<f32, _>(layout);
    // SAFETY: owned fragments, accessed by this thread only.
    unsafe {
        fill(&x.view_mut(), 2.0);
        transform(&x.view_mut(), |v| v + 1.0);
        fill(&y.view_mut(), f32::NAN);
        // beta = 0 does not read y.
        axpby(2.0f32, &x.view(), 0.0f32, &y.view_mut());
    }
    assert_eq!(y.engine.0, [6.0; 6]);
    let mut sum = 0.0;
    // SAFETY: as above.
    unsafe {
        axpby(1.0f32, &x.view(), 0.5f32, &y.view_mut());
        for_each(&y.view(), |v| sum += v);
        transform_into(&x.view(), &y.view_mut(), |v| v * 10.0);
        transform_zip(&x.view(), &y.view(), &y.view_mut(), |a, b| a + b);
    }
    assert_eq!(sum, 36.0);
    assert_eq!(y.engine.0, [33.0; 6]);
    // SAFETY: as above.
    unsafe { clear(&x.view_mut()) };
    assert_eq!(x.engine.0, [0.0; 6]);
}

#[test]
fn gemm_dispatches_by_rank() {
    let fma = MmaAtom::<UniversalFma<i32>>::new();
    let a = [1, 2, 3, 4, 5, 6];
    let b = [10, 20, 30];
    let mut c = [0i32; 9];
    let mut d = [0i32; 9];
    // [1] (V) x (V) => (V)
    let one = LayoutLeft::make(shape![Int::<1>]).unwrap();
    // SAFETY: every view below stays inside its array.
    unsafe {
        gemm(
            &fma,
            &make_tensor(d.as_mut_ptr(), one),
            &make_tensor(a.as_ptr(), one),
            &make_tensor(b.as_ptr(), one),
            &make_tensor(c.as_ptr(), one),
        );
    }
    assert_eq!(d[0], 10);

    // [2] (M) x (N) => (M, N): the outer product, accumulated into C.
    let m3 = LayoutLeft::make(shape![Int::<3>]).unwrap();
    let mn = LayoutLeft::make(shape![Int::<3>, Int::<3>]).unwrap();
    let cc = make_tensor(c.as_mut_ptr(), mn);
    // SAFETY: as above.
    unsafe {
        gemm_fma(
            &make_tensor(a.as_ptr(), m3),
            &make_tensor(b.as_ptr(), m3),
            &cc,
        )
    };
    assert_eq!(c, [10, 20, 30, 20, 40, 60, 30, 60, 90]);

    // [3] (M, K) x (N, K) => (M, N) with K = 2, into a separate D.
    let mk = LayoutLeft::make(shape![Int::<3>, Int::<2>]).unwrap();
    let nk = LayoutLeft::make(shape![Int::<1>, Int::<2>]).unwrap();
    let mn1 = LayoutLeft::make(shape![Int::<3>, Int::<1>]).unwrap();
    // SAFETY: as above.
    unsafe {
        gemm(
            &fma,
            &make_tensor(d.as_mut_ptr(), mn1),
            &make_tensor(a.as_ptr(), mk),
            &make_tensor(b.as_ptr(), nk),
            &make_tensor(c.as_ptr(), mn1),
        );
    }
    // Upstream adds C, not D, for each k, so only k = 1 remains in D.
    assert_eq!(d[..3], [10 + 4 * 20, 20 + 5 * 20, 30 + 6 * 20]);

    // [4] (V, M) x (V, N) => (V, M, N) and [5] (V, M, K) x (V, N, K) => (V, M, N).
    let mut acc = [0i32; 6];
    let vmn = LayoutLeft::make(shape![Int::<1>, Int::<3>, Int::<2>]).unwrap();
    let acc_t = make_tensor(acc.as_mut_ptr(), vmn);
    // SAFETY: as above.
    unsafe {
        let vm = LayoutLeft::make(shape![Int::<1>, Int::<3>]).unwrap();
        let vn = LayoutLeft::make(shape![Int::<1>, Int::<2>]).unwrap();
        gemm_fma(
            &make_tensor(a.as_ptr(), vm),
            &make_tensor(b.as_ptr(), vn),
            &acc_t,
        );
    }
    assert_eq!(acc, [10, 20, 30, 20, 40, 60]);
    let vmk = LayoutLeft::make(shape![Int::<1>, Int::<3>, Int::<2>]).unwrap();
    let vnk = LayoutLeft::make(shape![Int::<1>, Int::<1>, Int::<2>]).unwrap();
    let vm1 = LayoutLeft::make(shape![Int::<1>, Int::<3>, Int::<1>]).unwrap();
    let mut acc = [0i32; 3];
    // SAFETY: as above.
    unsafe {
        gemm_fma(
            &make_tensor(a.as_ptr(), vmk),
            &make_tensor(b.as_ptr(), vnk),
            &make_tensor(acc.as_mut_ptr(), vm1),
        );
    }
    assert_eq!(acc, [10 + 4 * 20, 20 + 5 * 20, 30 + 6 * 20]);
}

/// Records the order of its instructions in D.
struct Order<T>(core::marker::PhantomData<T>);

static STEP: AtomicI32 = AtomicI32::new(0);

impl<T> MmaTraits for Order<T> {
    type ValA = T;
    type ValB = T;
    type ValC = i32;
    type ShapeMnk = IntTuple<Int<1>, IntTuple<Int<1>, IntTuple<Int<1>, Nil>>>;
    type ThrId = Layout<Int<1>, Int<0>>;
    type ALayout = ();
    type BLayout = ();
    type CLayout = ();
    type DRegister = i32;
    type ARegister = T;
    type BRegister = T;
    type CRegister = i32;
    const D_REGISTERS: isize = 1;
    const A_REGISTERS: isize = 1;
    const B_REGISTERS: isize = 1;
    const C_REGISTERS: isize = 1;
    unsafe fn fma<DP, AP, BP, CP>(
        d: impl Fn(isize) -> DP,
        _: impl Fn(isize) -> AP,
        _: impl Fn(isize) -> BP,
        _: impl Fn(isize) -> CP,
    ) where
        DP: Write<Value = i32>,
        AP: Read<Value = T>,
        BP: Read<Value = T>,
        CP: Read<Value = i32>,
    {
        // SAFETY: forwarded to the caller.
        unsafe { d(0).write(STEP.fetch_add(1, Ordering::Relaxed)) }
    }
}

fn order<T: Default + Copy>() -> [i32; 12] {
    STEP.store(0, Ordering::Relaxed);
    let a = [T::default(); 4];
    let b = [T::default(); 3];
    let mut d = [0i32; 12];
    let dt = make_tensor(
        d.as_mut_ptr(),
        LayoutLeft::make(shape![Int::<1>, Int::<4>, Int::<3>]).unwrap(),
    );
    // SAFETY: every view stays inside its array.
    unsafe {
        gemm(
            &MmaAtom::<Order<T>>::new(),
            &dt,
            &make_tensor(
                a.as_ptr(),
                LayoutLeft::make(shape![Int::<1>, Int::<4>]).unwrap(),
            ),
            &make_tensor(
                b.as_ptr(),
                LayoutLeft::make(shape![Int::<1>, Int::<3>]).unwrap(),
            ),
            &dt,
        );
    }
    d
}

#[test]
fn gemm_traverses_in_upstream_register_reuse_order() {
    // d is column-major (M = 4, N = 3): d[m + 4 n] is the step of (m, n).
    // 32-bit A and B: kinked serpentine over pairs of rows.
    assert_eq!(order::<f32>(), [0, 1, 10, 11, 2, 3, 8, 9, 4, 5, 6, 7]);
    // 64-bit A and B: row-major serpentine.
    assert_eq!(order::<f64>(), [0, 5, 6, 11, 1, 4, 7, 10, 2, 3, 8, 9]);
    // Other widths: column-major serpentine.
    assert_eq!(order::<u8>(), [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11]);
}

#[test]
fn layouts_used_above_are_static() {
    // Keeps the strides of the type aliases above honest.
    let col: Col<4, 8> = LayoutLeft::make(shape![Int::<4>, Int::<8>]).unwrap();
    assert_eq!(col.stride, stride![Int::<1>, Int::<4>]);
}
