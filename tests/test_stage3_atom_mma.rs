#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

//! Contract tests for Stage 3: pure-safe `CopyAtom`/`TiledCopy`/`ThrCopy`,
//! `MmaAtom`/`TiledMma`/`ThrMma`, `make_tiled_copy_{a,b,c}`, `Retile`, `PredicatedCopy`,
//! `copy_with`/`copy_if_with`, and 5-rank register-reuse `gemm_mma`/`gemm_fma`.

use core::marker::PhantomData;
use core::sync::atomic::{AtomicI32, Ordering};

use noodle_cute::algorithm::atom::{
    CopyAtom, CopyTraits, MmaAtom, MmaTraits, UniversalCopy, UniversalFma, make_tiled_copy,
    make_tiled_copy_a, make_tiled_copy_b, make_tiled_copy_c, make_tiled_mma,
};
use noodle_cute::algorithm::{clear, copy, copy_if_with, copy_with, gemm_fma, gemm_mma};
use noodle_cute::basis::as_tuple;
use noodle_cute::integer::{Int, int};
use noodle_cute::layout::Layout;
use noodle_cute::swizzle::Swizzle;
use noodle_cute::tensor::{
    Accessor, AccessorBase, Tensor, identity_tensor, make_owned_tensor, make_tensor,
};
use noodle_cute::tuple::{Nil, Til};
use noodle_cute::{f2, int_tuple, tuple};

type P<A, B> = Til<A, Til<B, Nil>>;

// ==================================================================
// 1. TiledCopy & CopyAtom contracts
// ==================================================================

#[test]
fn tiled_copy_tv_layout_and_tiler() {
    let atom = CopyAtom::<UniversalCopy<i32>, i32>::new();
    let threads = Layout::from_shape(int_tuple![Int::<4>, Int::<8>]);
    let values = Layout::from_shape(int_tuple![Int::<4>, Int::<1>]);
    let tiled = make_tiled_copy(atom, threads, values);
    assert_eq!(tiled.tiler, int_tuple![Int::<16>, Int::<8>]);
    // (thread, value) -> column-major index of (4 * (t % 4) + v, t / 4) = 4 * t + v.
    assert_eq!(tiled.layout_tv.shape, int_tuple![Int::<32>, Int::<4>]);
    assert_eq!(tiled.layout_tv.stride, int_tuple![Int::<4>, Int::<1>]);
}

#[test]
fn tiled_copy_partitions_follow_the_thread_value_map() {
    let atom = CopyAtom::<UniversalCopy<i32>, i32>::new();
    let threads = Layout::from_shape(int_tuple![Int::<4>, Int::<8>]);
    let values = Layout::from_shape(int_tuple![Int::<4>, Int::<1>]);
    let tiled = make_tiled_copy(atom, threads, values);

    let data: [i32; 32 * 16] = core::array::from_fn(|i| i as i32);
    let src = make_tensor(&data, int_tuple![Int::<32>, Int::<16>]);
    for t in 0..32i32 {
        let part = tiled.get_slice(t).partition_s(src);
        assert_eq!(
            part.shape(),
            int_tuple![(Int::<1>, Int::<4>), Int::<2>, Int::<2>]
        );
        assert_eq!(
            part.stride(),
            int_tuple![(Int::<0>, Int::<1>), Int::<16>, Int::<256>]
        );
        for v in 0..4i32 {
            for (rm, rn) in [(0i32, 0i32), (1, 0), (0, 1), (1, 1)] {
                let (m, n) = (4 * (t % 4) + v + 16 * rm, t / 4 + 8 * rn);
                let got = part.at(tuple![(0i32, v), rm, rn]);
                assert_eq!(got, m + 32 * n);
            }
        }
    }

    // Destination side partitions a row-major tile with the same TV map.
    let mut out = [0i32; 16 * 8];
    let dst = make_tensor(
        &mut out,
        Layout::from_shape_right(int_tuple![Int::<16>, Int::<8>]),
    );
    let mut part = tiled.get_slice(5i32).partition_d(dst);
    assert_eq!(
        part.shape(),
        int_tuple![(Int::<1>, Int::<4>), Int::<1>, Int::<1>]
    );
    assert_eq!(
        part.stride(),
        int_tuple![(Int::<0>, Int::<8>), Int::<128>, Int::<8>]
    );
    assert_eq!(part.accessor.offset_index(), 4 * 8 + 1);
    part.set(tuple![(0i32, 3i32), 0i32, 0i32], 77);
    assert_eq!(out[(4 + 3) * 8 + 1], 77);
}

#[test]
fn copy_atoms_move_whole_registers_safely() {
    // 128-bit copy moves four consecutive f32 values per thread via pure-safe RegisterCodec.
    let atom = CopyAtom::<UniversalCopy<u128>, f32>::new();
    let threads = Layout::from_shape(int_tuple![Int::<4>, Int::<8>]);
    let values = Layout::from_shape(int_tuple![Int::<4>, Int::<1>]);
    let tiled = make_tiled_copy(atom, threads, values);
    assert_eq!(tiled.tiler, int_tuple![Int::<16>, Int::<8>]);

    let tile = Layout::from_shape(int_tuple![Int::<16>, Int::<8>]);
    let mut src = make_owned_tensor::<f32, _>(tile);
    let mut dst = make_owned_tensor::<f32, _>(tile);
    src.as_mut_slice()
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = i as f32);
    for t in 0..32i32 {
        let thr = tiled.get_slice(t);
        let s = thr.partition_s(src.view());
        let mut d = thr.partition_d(dst.view_mut());
        assert_eq!(s.shape(), int_tuple![(Int::<4>, Int::<1>), Int::<1>, Int::<1>]);
        tiled.atom.call(&s, &mut d);
    }
    assert_eq!(dst.as_slice(), src.as_slice());
}

// ==================================================================
// 2. TiledMma & end-to-end GEMM contracts
// ==================================================================

#[test]
fn tiled_mma_interleaves_atoms_over_the_tile() {
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<f32>>::new(),
        Layout::from_shape(int_tuple![Int::<4>, Int::<2>]),
        tuple![..],
    );
    assert_eq!(
        mma.thr_layout_vmnk.shape,
        int_tuple![Int::<1>, Int::<4>, Int::<2>, Int::<1>]
    );
    assert_eq!(
        mma.thr_layout_vmnk.stride,
        int_tuple![Int::<0>, Int::<1>, Int::<4>, Int::<0>]
    );
    assert_eq!(mma.permutation_mnk, int_tuple![Int::<4>, Int::<2>, Int::<1>]);

    let a = [0f32; 8 * 2];
    let b = [0f32; 4 * 2];
    let c = [0f32; 8 * 4];
    let ga = make_tensor(&a, int_tuple![Int::<8>, Int::<2>]);
    let gb = make_tensor(&b, int_tuple![Int::<4>, Int::<2>]);
    let gc = make_tensor(&c, int_tuple![Int::<8>, Int::<4>]);

    // Thread 5 is (v, m, n, k) = (0, 1, 1, 0).
    let thr = mma.get_slice(5i32);
    let ta = thr.partition_a(ga);
    assert_eq!(ta.shape(), int_tuple![Int::<1>, Int::<2>, Int::<2>]);
    assert_eq!(ta.stride(), int_tuple![Int::<0>, Int::<4>, Int::<8>]);
    assert_eq!(ta.accessor.offset_index(), 1);

    let tb = thr.partition_b(gb);
    assert_eq!(tb.shape(), int_tuple![Int::<1>, Int::<2>, Int::<2>]);
    assert_eq!(tb.stride(), int_tuple![Int::<0>, Int::<2>, Int::<4>]);
    assert_eq!(tb.accessor.offset_index(), 1);

    let tc = thr.partition_c(gc);
    assert_eq!(tc.stride(), int_tuple![Int::<0>, Int::<4>, Int::<16>]);
    assert_eq!(tc.accessor.offset_index(), 1 + 8);

    let fa = thr.partition_fragment_a(ga);
    assert_eq!(fa.shape(), int_tuple![Int::<1>, Int::<2>, Int::<2>]);
    let fc = thr.partition_fragment_c(gc);
    assert_eq!(fc.shape(), int_tuple![Int::<1>, Int::<2>, Int::<2>]);
    assert_eq!(fc.as_slice().len(), 4);
}

#[test]
fn tiled_mma_partitions_compute_a_gemm() {
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<f32>>::new(),
        Layout::from_shape(int_tuple![Int::<4>, Int::<2>]),
        ..,
    );
    let a: [f32; 8 * 3] = core::array::from_fn(|i| (i % 7) as f32);
    let b: [f32; 4 * 3] = core::array::from_fn(|i| (i % 5) as f32 - 2.0);
    let mut c = [0f32; 8 * 4];
    let ga = make_tensor(&a, int_tuple![Int::<8>, Int::<3>]);
    let gb = make_tensor(&b, int_tuple![Int::<4>, Int::<3>]);
    let gc = make_tensor(&mut c, int_tuple![Int::<8>, Int::<4>]);

    for t in 0..8i32 {
        let thr = mma.get_slice(t);
        let ta = thr.partition_a(ga);
        let tb = thr.partition_b(gb);
        let mut tc = thr.partition_c(gc);
        let mut acc = thr.partition_fragment_c(gc);
        clear(&mut acc);
        gemm_fma(&ta, &tb, &mut acc);
        copy(acc.view(), &mut tc);
    }

    for m in 0..8 {
        for n in 0..4 {
            let expected: f32 = (0..3).map(|k| a[m + 8 * k] * b[n + 4 * k]).sum();
            assert_eq!(c[m + 8 * n], expected);
        }
    }
}

#[test]
fn tilings_partition_f2_swizzled_shared_memory() {
    // Exact F2 representation of Swizzle<2, 2, 3> over column-major (16, 8).
    let sw = Swizzle::new(Int::<2>, Int::<2>, Int::<3>);
    let smem_layout = Layout::new(
        int_tuple![Int::<16>, (Int::<2>, Int::<4>)],
        tuple![f2!(Int::<1>), (f2!(Int::<16>), f2!(Int::<36>))],
    );
    let tile = Layout::from_shape(int_tuple![Int::<16>, Int::<8>]);

    // gmem -> F2-swizzled smem through TiledCopy.
    let atom = CopyAtom::<UniversalCopy<i32>, i32>::new();
    let tiled = make_tiled_copy(
        atom,
        Layout::from_shape(int_tuple![Int::<4>, Int::<8>]),
        Layout::from_shape(int_tuple![Int::<4>, Int::<1>]),
    );
    let data: [i32; 128] = core::array::from_fn(|i| i as i32);
    let mut smem = [-1i32; 128];
    let src = make_tensor(&data, tile);
    let dst = make_tensor(&mut smem, smem_layout);
    for t in 0..32i32 {
        let thr = tiled.get_slice(t);
        let s = thr.partition_s(src);
        let mut d = thr.partition_d(dst);
        tiled.atom.call(&s, &mut d);
    }
    for i in 0..128i32 {
        assert_eq!(smem[sw.call(i) as usize], i);
    }

    // F2-swizzled smem -> TiledMma: partition_a reads logical A(m, k).
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<i32>>::new(),
        Layout::from_shape(int_tuple![Int::<4>, Int::<2>]),
        ..,
    );
    let s_a = make_tensor(&smem, smem_layout);
    let thr = mma.get_slice(5i32);
    let t_a = thr.partition_a(s_a);
    for rm in 0..4i32 {
        for k in 0..8i32 {
            // Thread 5 owns rows 1, 5, 9, 13.
            assert_eq!(t_a.at(int_tuple![0i32, rm, k]), 1 + 4 * rm + 16 * k);
        }
    }
    let frag = thr.partition_fragment_a(s_a);
    assert_eq!(frag.as_slice().len(), 32);
}

// ==================================================================
// 3. MMA-compatible TV maps, SM80 WarpMma, Retile, Ldsm4, Predicates
// ==================================================================

#[test]
fn mma_copy_tv_broadcasts_unused_thread_modes() {
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<f32>>::new(),
        Layout::from_shape(int_tuple![Int::<2>, Int::<2>, Int::<2>]),
        ..,
    );
    let atom = CopyAtom::<UniversalCopy<f32>, f32>::new();
    let a = make_tiled_copy_a(atom, &mma);
    let b = make_tiled_copy_b(atom, &mma);
    let c = make_tiled_copy_c(atom, &mma);
    assert_eq!(a.tiler, int_tuple![Int::<2>, Int::<2>]);
    assert_eq!(b.tiler, int_tuple![Int::<2>, Int::<2>]);
    assert_eq!(c.tiler, int_tuple![Int::<2>, Int::<2>]);
    for t in 0..8i32 {
        assert_eq!((a.layout_tv)(t, 0i32), t % 2 + 2 * (t / 4));
        assert_eq!((b.layout_tv)(t, 0i32), (t / 2) % 2 + 2 * (t / 4));
        assert_eq!((c.layout_tv)(t, 0i32), t % 4);
    }
}

#[test]
fn identity_tensors_follow_copy_and_mma_tail_partitions() {
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<f32>>::new(),
        Layout::from_shape(int_tuple![Int::<2>, Int::<2>]),
        ..,
    );
    let tiled = make_tiled_copy_c(CopyAtom::<UniversalCopy<f32>, f32>::new(), &mma);
    let extent = int_tuple![3i32, 3i32];
    let identity = identity_tensor(extent);
    for t in 0..4i32 {
        let cp = tiled.get_slice(t).partition_s(identity);
        let mp = mma.get_slice(t).partition_c(identity);
        for rn in 0..2i32 {
            for rm in 0..2i32 {
                let expected = int_tuple![t % 2 + 2 * rm, t / 2 + 2 * rn];
                assert_eq!(as_tuple(cp(0i32, rm, rn)), expected);
                assert_eq!(as_tuple(mp(0i32, rm, rn)), expected);
            }
        }
    }
}

#[test]
fn mma_fragment_retile_preserves_storage() {
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<f32>>::new(),
        Layout::from_shape(int_tuple![Int::<2>, Int::<2>]),
        int_tuple![Int::<4>, Int::<4>],
    );
    let tiled = make_tiled_copy_c(CopyAtom::<UniversalCopy<f32>, f32>::new(), &mma);
    let values: [f32; 64] = core::array::from_fn(|i| i as f32);
    let tensor = make_tensor(&values, int_tuple![Int::<8>, Int::<8>]);

    let thr = mma.get_slice(1i32);
    let mut fragment = thr.partition_fragment_c(tensor);
    let view = fragment.view_mut();
    let copy_thr = tiled.get_slice(1i32);
    let mut retiled = copy_thr.retile_d(view);
    assert_eq!(retiled.accessor, view.accessor);
    let src = copy_thr.partition_s(tensor);
    copy_with(&tiled, &src, &mut retiled);

    let retiled_s = copy_thr.retile_s(view);
    assert_eq!(retiled_s.layout, retiled.layout);
    assert_eq!(src.size(), retiled.size());

    let mma_part = thr.partition_c(tensor);
    for i in 0..mma_part.size() {
        assert_eq!(fragment(i), mma_part(i));
    }
}

// SM80 16x8x8 warp-level MMA layout-only fixture (`mma_traits_sm80.hpp`).
type Row16x8 =
    Layout<P<P<Int<4>, Int<8>>, P<Int<2>, Int<2>>>, P<P<Int<32>, Int<1>>, P<Int<16>, Int<8>>>>;
type Row8x8 = Layout<P<P<Int<4>, Int<8>>, Int<2>>, P<P<Int<16>, Int<1>>, Int<8>>>;

struct WarpMma;
impl MmaTraits for WarpMma {
    type ValA = u16;
    type ValB = u16;
    type ValC = u16;
    type ShapeMnk = Til<Int<16>, Til<Int<8>, Til<Int<8>, Nil>>>;
    type ThrId = Layout<Int<32>, Int<1>>;
    type ALayout = Row16x8;
    type BLayout = Row8x8;
    type CLayout = Row16x8;
    type DRegister = u32;
    type ARegister = u32;
    type BRegister = u32;
    type CRegister = u32;
    const D_REGISTERS: int = 2;
    const A_REGISTERS: int = 2;
    const B_REGISTERS: int = 1;
    const C_REGISTERS: int = 2;

    fn fma(
        _: impl FnMut(int, u32),
        _: impl FnMut(int) -> u32,
        _: impl FnMut(int) -> u32,
        _: impl FnMut(int) -> u32,
    ) {
        panic!("layout-only fixture");
    }
}

#[test]
fn warp_operand_tv_maps_match_upstream_lane_coordinates() {
    let mma = make_tiled_mma(
        MmaAtom::<WarpMma>::new(),
        Layout::from_shape(int_tuple![Int::<2>, Int::<2>]),
        ..,
    );
    let atom = CopyAtom::<UniversalCopy<u16>, u16>::new();
    let a = make_tiled_copy_a(atom, &mma);
    let b = make_tiled_copy_b(atom, &mma);
    let c = make_tiled_copy_c(atom, &mma);
    assert_eq!(a.layout_tv, mma.get_layout_a_tv());
    assert_eq!(b.layout_tv, mma.get_layout_b_tv());
    assert_eq!(c.layout_tv, mma.get_layout_c_tv());

    for t in 0..128i32 {
        let (lane, wm, wn) = (t % 32, (t / 32) % 2, t / 64);
        for v in 0..4i32 {
            let m = lane / 4 + 16 * wm + 8 * (v / 2);
            let k = 2 * (lane % 4) + v % 2;
            assert_eq!((a.layout_tv)(t, v), m + 32 * k);
            let n = 2 * (lane % 4) + v % 2 + 8 * wn;
            assert_eq!((c.layout_tv)(t, v), m + 32 * n);
        }
        for v in 0..2i32 {
            let n = lane / 4 + 8 * wn;
            let k = 2 * (lane % 4) + v;
            assert_eq!((b.layout_tv)(t, v), n + 16 * k);
        }
    }
}

struct Gather8;
impl CopyTraits for Gather8 {
    type SrcLayout = <UniversalCopy<u128> as CopyTraits>::SrcLayout;
    type DstLayout = Self::SrcLayout;
    type RefLayout = Self::SrcLayout;
    type SRegister = u16;
    type DRegister = u16;
    const SRC_REGISTERS: int = 8;
    const DST_REGISTERS: int = 8;

    fn copy(mut src: impl FnMut(int) -> u16, mut dst: impl FnMut(int, u16)) {
        for i in 0..8i32 {
            dst(i, src(i));
        }
    }
}

#[test]
fn warp_retile_gathers_values_from_mma_rest_modes() {
    let mma = make_tiled_mma(
        MmaAtom::<WarpMma>::new(),
        Layout::from_shape(int_tuple![Int::<1>]),
        int_tuple![Int::<16>, Int::<16>],
    );
    let tiled = make_tiled_copy_c(CopyAtom::<Gather8, u16>::new(), &mma);
    let data: [u16; 16 * 32] = core::array::from_fn(|i| i as u16);
    let tensor = make_tensor(&data, int_tuple![Int::<16>, Int::<32>]);

    for t in 0..32i32 {
        let thr = mma.get_slice(t);
        let mut fragment = thr.partition_fragment_c(tensor);
        let mma_part = thr.partition_c(tensor);
        let copy_thr = tiled.get_slice(t);
        let src = copy_thr.partition_s(tensor);
        let mut dst = copy_thr.retile_d(fragment.view_mut());
        assert_eq!(dst.size(), mma_part.size());
        copy_with(&copy_thr, &src, &mut dst);
        for i in 0..mma_part.size() {
            assert_eq!(fragment(i), mma_part(i));
        }
    }
}

/// Safe panic-on-load accessor proving false predicates never read from `src`.
#[derive(Clone, Copy)]
struct PanicOnLoad<T>(PhantomData<T>);

impl<T: Copy> AccessorBase for PanicOnLoad<T> {}
impl<T: Copy> Accessor<int> for PanicOnLoad<T> {
    type Item = T;
    fn load(&self, _: int) -> T {
        panic!("must not read source when predicate is false");
    }
}

/// Test-local fixed-point newtype replacing `F16` without adding low-precision types to core.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Scaled16(i16);

impl From<f32> for Scaled16 {
    fn from(v: f32) -> Self {
        Self((v * 10.0) as i16)
    }
}

#[test]
fn predicated_copy_dispatch_skips_invalid_source_and_converts_numbers() {
    let input = [1.0f32, 2.0, 3.0, 999.0];
    let mut output = [Scaled16(-10); 4];
    let src = make_tensor(&input, Int::<4>);
    let atom = CopyAtom::<UniversalCopy<f32, Scaled16>, f32>::new();
    {
        let mut dst = make_tensor(&mut output, Int::<4>);
        copy_if_with(&atom, |i| i < 3, &src, &mut dst);
    }
    assert_eq!(
        output,
        [Scaled16(10), Scaled16(20), Scaled16(30), Scaled16(-10)]
    );

    let poison = Tensor::new(
        PanicOnLoad::<f32>(PhantomData),
        Layout::new(Int::<4>, Int::<1>),
    );
    {
        let mut dst = make_tensor(&mut output, Int::<4>);
        copy_with(&atom.with(false), &poison, &mut dst);
    }
    assert_eq!(
        output,
        [Scaled16(10), Scaled16(20), Scaled16(30), Scaled16(-10)]
    );
}

struct ZeroFill;
impl CopyTraits for ZeroFill {
    type SrcLayout = <UniversalCopy<u32> as CopyTraits>::SrcLayout;
    type DstLayout = Self::SrcLayout;
    type RefLayout = Self::SrcLayout;
    type SRegister = u32;
    type DRegister = u32;
    const SRC_REGISTERS: int = 1;
    const DST_REGISTERS: int = 1;

    fn copy(mut src: impl FnMut(int) -> u32, mut dst: impl FnMut(int, u32)) {
        dst(0, src(0));
    }

    fn copy_predicated(pred: bool, src: impl FnMut(int) -> u32, mut dst: impl FnMut(int, u32)) {
        if pred {
            Self::copy(src, dst);
        } else {
            dst(0, 0);
        }
    }
}

#[test]
fn false_predicate_reaches_backend_zero_fill() {
    let mut output = [99u32; 4];
    let poison = Tensor::new(
        PanicOnLoad::<u32>(PhantomData),
        Layout::new(Int::<4>, Int::<1>),
    );
    let mut dst = make_tensor(&mut output, Int::<4>);
    copy_with(&CopyAtom::<ZeroFill, u32>::new().with(false), &poison, &mut dst);
    assert_eq!(output, [0; 4]);
}

#[test]
fn mma_copy_matches_partitions_with_reordered_threads_and_permutations() {
    let threads = Layout::new(
        int_tuple![Int::<2>, Int::<2>, Int::<2>],
        int_tuple![Int::<4>, Int::<2>, Int::<1>],
    );
    let perm = Layout::new(int_tuple![Int::<2>, Int::<2>], int_tuple![Int::<2>, Int::<1>]);
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<f32>>::new(),
        threads,
        tuple![perm, Int::<4>],
    );
    let atom = CopyAtom::<UniversalCopy<f32>, f32>::new();
    let a = make_tiled_copy_a(atom, &mma);
    let b = make_tiled_copy_b(atom, &mma);
    let c = make_tiled_copy_c(atom, &mma);

    let data: [f32; 16] = core::array::from_fn(|i| i as f32);
    let ab = make_tensor(&data, int_tuple![Int::<4>, Int::<2>]);
    let cv = make_tensor(&data, int_tuple![Int::<4>, Int::<4>]);
    for t in 0..8i32 {
        let thr = mma.get_slice(t);
        let ma = thr.partition_a(ab);
        let mb = thr.partition_b(ab);
        let mc = thr.partition_c(cv);
        let ca = a.get_slice(t).partition_s(ab);
        let cb = b.get_slice(t).partition_s(ab);
        let cc = c.get_slice(t).partition_s(cv);
        assert_eq!(ca.size(), ma.size());
        assert_eq!(cb.size(), mb.size());
        assert_eq!(cc.size(), mc.size());
        for i in 0..ma.size() {
            assert_eq!(ca(i), ma(i));
        }
        for i in 0..mb.size() {
            assert_eq!(cb(i), mb(i));
        }
        for i in 0..mc.size() {
            assert_eq!(cc(i), mc(i));
        }
    }
}

#[test]
fn operand_a_b_retile_preserves_broadcast_thread_fragments() {
    let mma = make_tiled_mma(
        MmaAtom::<WarpMma>::new(),
        Layout::from_shape(int_tuple![Int::<2>, Int::<2>]),
        int_tuple![Int::<32>, Int::<16>, Int::<16>],
    );
    let a = make_tiled_copy_a(CopyAtom::<UniversalCopy<u16>, u16>::new(), &mma);
    let b = make_tiled_copy_b(CopyAtom::<UniversalCopy<u16>, u16>::new(), &mma);
    let data: [u16; 32 * 32] = core::array::from_fn(|i| i as u16);
    let ta = make_tensor(&data, int_tuple![Int::<32>, Int::<32>]);
    let tb = make_tensor(&data, int_tuple![Int::<16>, Int::<32>]);

    for t in [0i32, 17, 32, 63, 64, 101, 127] {
        let thr = mma.get_slice(t);
        let mut fa = thr.partition_fragment_a(ta);
        let mut fb = thr.partition_fragment_b(tb);
        let sa = a.get_slice(t).partition_s(ta);
        let sb = b.get_slice(t).partition_s(tb);
        let mut da = a.get_slice(t).retile_d(fa.view_mut());
        let mut db = b.get_slice(t).retile_d(fb.view_mut());
        copy_with(&a, &sa, &mut da);
        copy_with(&b, &sb, &mut db);

        let ma = thr.partition_a(ta);
        let mb = thr.partition_b(tb);
        for i in 0..ma.size() {
            assert_eq!(fa(i), ma(i));
        }
        for i in 0..mb.size() {
            assert_eq!(fb(i), mb(i));
        }
    }
}

// SM75_U32x4_LDSM_N operand layouts (`copy_traits_sm75.hpp`).
struct Ldsm4;
impl CopyTraits for Ldsm4 {
    type SrcLayout = Layout<P<Int<32>, Int<128>>, P<Int<128>, Int<1>>>;
    type DstLayout = Layout<P<Int<32>, P<Int<32>, Int<4>>>, P<Int<32>, P<Int<1>, Int<1024>>>>;
    type RefLayout = Self::DstLayout;
    type SRegister = u128;
    type DRegister = u32;
    const SRC_REGISTERS: int = 1;
    const DST_REGISTERS: int = 4;

    fn copy(_: impl FnMut(int) -> u128, _: impl FnMut(int, u32)) {
        panic!("layout-only fixture");
    }
}

#[test]
fn ldsm_source_partition_connects_f2_swizzled_memory_to_mma_reference_values() {
    // Exact F2 representation of Swizzle<2, 3, 3> over column-major (32, 16).
    let swizzled_layout = Layout::new(
        int_tuple![Int::<32>, (Int::<2>, Int::<4>, Int::<2>)],
        tuple![
            f2!(Int::<1>),
            (f2!(Int::<32>), f2!(Int::<72>), f2!(Int::<256>))
        ],
    );
    let mma = make_tiled_mma(
        MmaAtom::<WarpMma>::new(),
        Layout::from_shape(int_tuple![Int::<2>, Int::<2>]),
        int_tuple![Int::<32>, Int::<16>, Int::<16>],
    );
    let tiled = make_tiled_copy_a(CopyAtom::<Ldsm4, u16>::new(), &mma);
    let data: [u16; 512] = core::array::from_fn(|i| i as u16);
    let memory = make_tensor(&data, swizzled_layout);
    let identity = identity_tensor(int_tuple![Int::<32>, Int::<16>]);

    for t in 0..128i32 {
        let cp = tiled.get_slice(t).partition_s(memory);
        let coords = tiled.get_slice(t).partition_s(identity);
        for v in 0..8i32 {
            // SrcLayout is 8*t+v; invert RefLayout (2*r + rv%2 + 64*(rv/2)).
            let x = 8 * (t % 32) + v;
            let r = (x % 64) / 2;
            let rv = x % 2 + 2 * (x / 64);
            let m = r / 4 + 8 * ((rv / 2) % 2) + 16 * ((t / 32) % 2);
            let k = 2 * (r % 4) + rv % 2 + 8 * (rv / 4);
            assert_eq!(as_tuple(coords(v)), int_tuple![m, k]);
            let raw = m + 32 * k;
            let swizzled = raw ^ ((raw & (3 << 6)) >> 3);
            assert_eq!(cp(v), swizzled as u16);
        }
    }
}

// ==================================================================
// 4. 5-rank gemm_mma / gemm_fma dispatch & serpentine traversal orders
// ==================================================================

#[test]
fn gemm_mma_dispatches_all_five_ranks() {
    let fma = MmaAtom::<UniversalFma<i32>>::new();
    let a = [1i32, 2, 3, 4, 5, 6];
    let b = [10i32, 20, 30];
    let mut c = [0i32; 9];
    let mut d = [0i32; 9];

    // [1] (V) x (V) => (V)
    let one = int_tuple![Int::<1>];
    let mut dt1 = make_tensor(&mut d, one);
    gemm_mma(
        &fma,
        &mut dt1,
        &make_tensor(&a, one),
        &make_tensor(&b, one),
        &make_tensor(&c, one),
    );
    assert_eq!(d[0], 10);

    // [2] (M) x (N) => (M, N): outer product accumulated into C.
    let m3 = int_tuple![Int::<3>];
    let mn = int_tuple![Int::<3>, Int::<3>];
    let mut cc = make_tensor(&mut c, mn);
    gemm_fma(&make_tensor(&a, m3), &make_tensor(&b, m3), &mut cc);
    assert_eq!(c, [10, 20, 30, 20, 40, 60, 30, 60, 90]);

    // [3] (M, K) x (N, K) => (M, N) with K = 2, into separate D.
    let mk = int_tuple![Int::<3>, Int::<2>];
    let nk = int_tuple![Int::<1>, Int::<2>];
    let mn1 = int_tuple![Int::<3>, Int::<1>];
    let mut dt3 = make_tensor(&mut d, mn1);
    gemm_mma(
        &fma,
        &mut dt3,
        &make_tensor(&a, mk),
        &make_tensor(&b, nk),
        &make_tensor(&c, mn1),
    );
    // Upstream adds C (not D) for each k, so k = 1 dominates D.
    assert_eq!(d[..3], [10 + 4 * 20, 20 + 5 * 20, 30 + 6 * 20]);

    // [4] (V, M) x (V, N) => (V, M, N)
    let mut acc = [0i32; 6];
    let mut acc_t = make_tensor(&mut acc, int_tuple![Int::<1>, Int::<3>, Int::<2>]);
    gemm_fma(
        &make_tensor(&a, int_tuple![Int::<1>, Int::<3>]),
        &make_tensor(&b, int_tuple![Int::<1>, Int::<2>]),
        &mut acc_t,
    );
    assert_eq!(acc, [10, 20, 30, 20, 40, 60]);

    // [5] (V, M, K) x (V, N, K) => (V, M, N)
    let mut acc5 = [0i32; 3];
    let mut acc5_t = make_tensor(&mut acc5, int_tuple![Int::<1>, Int::<3>, Int::<1>]);
    gemm_fma(
        &make_tensor(&a, int_tuple![Int::<1>, Int::<3>, Int::<2>]),
        &make_tensor(&b, int_tuple![Int::<1>, Int::<1>, Int::<2>]),
        &mut acc5_t,
    );
    assert_eq!(acc5, [10 + 4 * 20, 20 + 5 * 20, 30 + 6 * 20]);
}

/// Records the order of MMA instructions into D.
struct Order<T>(PhantomData<T>);

static STEP: AtomicI32 = AtomicI32::new(0);

impl<T: Copy> MmaTraits for Order<T> {
    type ValA = T;
    type ValB = T;
    type ValC = i32;
    type ShapeMnk = Til<Int<1>, Til<Int<1>, Til<Int<1>, Nil>>>;
    type ThrId = Layout<Int<1>, Int<0>>;
    type ALayout = ();
    type BLayout = ();
    type CLayout = ();
    type DRegister = i32;
    type ARegister = T;
    type BRegister = T;
    type CRegister = i32;
    const D_REGISTERS: int = 1;
    const A_REGISTERS: int = 1;
    const B_REGISTERS: int = 1;
    const C_REGISTERS: int = 1;

    fn fma(
        mut d: impl FnMut(int, i32),
        _: impl FnMut(int) -> T,
        _: impl FnMut(int) -> T,
        _: impl FnMut(int) -> i32,
    ) {
        d(0, STEP.fetch_add(1, Ordering::Relaxed));
    }
}

fn record_order<T: Default + Copy>() -> [i32; 12] {
    STEP.store(0, Ordering::Relaxed);
    let a = [T::default(); 4];
    let b = [T::default(); 3];
    let mut d = [0i32; 12];
    let ct = make_tensor(&mut d, int_tuple![Int::<1>, Int::<4>, Int::<3>]);
    let mut dt = ct; // Copy CellAccessor: zero-unsafe aliased D/C.
    gemm_mma(
        &MmaAtom::<Order<T>>::new(),
        &mut dt,
        &make_tensor(&a, int_tuple![Int::<1>, Int::<4>]),
        &make_tensor(&b, int_tuple![Int::<1>, Int::<3>]),
        &ct,
    );
    d
}

#[test]
fn gemm_mma_traverses_in_upstream_register_reuse_order() {
    // 32-bit A and B: kinked serpentine over pairs of rows.
    assert_eq!(record_order::<f32>(), [0, 1, 10, 11, 2, 3, 8, 9, 4, 5, 6, 7]);
    // 64-bit A and B: row-major serpentine.
    assert_eq!(record_order::<f64>(), [0, 5, 6, 11, 1, 4, 7, 10, 2, 3, 8, 9]);
    // Other widths: column-major serpentine.
    assert_eq!(record_order::<u8>(), [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11]);
}

// ==================================================================
// 5. Const evaluation of TiledCopy, TiledMma, and Retile
// ==================================================================

#[test]
fn const_eval_tiled_copy_and_tiled_mma() {
    const DATA: [i32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    const VAL: i32 = {
        let mma = make_tiled_mma(
            MmaAtom::<UniversalFma<i32>>::new(),
            Layout::from_shape(int_tuple![Int::<2>, Int::<2>]),
            ..,
        );
        let tiled_c = make_tiled_copy_c(CopyAtom::<UniversalCopy<i32>, i32>::new(), &mma);
        let tensor = make_tensor(&DATA, int_tuple![Int::<4>, Int::<4>]);
        let cp = tiled_c.get_slice(3i32).partition_s(tensor);
        cp.at(int_tuple![0i32, 1i32, 1i32])
    };
    // Thread 3 is (m, n) = (1, 1); rest (1, 1) -> global (3, 3) = 15.
    assert_eq!(VAL, 15);
}
