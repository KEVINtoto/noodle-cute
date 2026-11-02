//! Contract tests for copy and MMA atoms and their tilings, following CuTe's
//! `copy_atom.hpp`, `copy_traits.hpp`, `mma_atom.hpp`, and `mma_traits.hpp`.
//!
//! Thread–value mappings are checked against upstream's documented
//! semantics: `make_tiled_copy(atom, thr, val)` rakes `val` blocks over `thr`,
//! so thread `t` of a column-major `(4, 8)` thread layout with `(4, 1)` values
//! owns rows `4 * (t % 4) + v` of column `t / 4`; `TiledMMA` interleaves the
//! atoms of `(ThrM, ThrN)` over the tile.

use noodle_cute::algorithm::{clear, copy, gemm_fma};
use noodle_cute::atom::{
    CopyAtom, MmaAtom, UniversalCopy, UniversalFma, make_tiled_copy, make_tiled_mma,
};
use noodle_cute::coord::X;
use noodle_cute::layout::{LayoutLeft, LayoutRight};
use noodle_cute::tensor::{make_owned_tensor, make_tensor};
use noodle_cute::{Int, IntTuple, Nil, shape, stride};

#[test]
fn tiled_copy_tv_layout_and_tiler() {
    let atom = CopyAtom::<UniversalCopy<i32>, i32>::new();
    let threads = LayoutLeft::make(shape![Int::<4>, Int::<8>]).unwrap();
    let values = LayoutLeft::make(shape![Int::<4>, Int::<1>]).unwrap();
    let tiled = make_tiled_copy(atom, threads, values).unwrap();
    assert_eq!(tiled.tiler, shape![Int::<16>, Int::<8>]);
    // (thread, value) -> column-major index of (4 * (t % 4) + v, t / 4) = 4 t + v.
    assert_eq!(tiled.layout_tv.shape, shape![Int::<32>, Int::<4>]);
    assert_eq!(tiled.layout_tv.stride, stride![Int::<4>, Int::<1>]);
}

#[test]
fn tiled_copy_partitions_follow_the_thread_value_map() {
    let atom = CopyAtom::<UniversalCopy<i32>, i32>::new();
    let threads = LayoutLeft::make(shape![Int::<4>, Int::<8>]).unwrap();
    let values = LayoutLeft::make(shape![Int::<4>, Int::<1>]).unwrap();
    let tiled = make_tiled_copy(atom, threads, values).unwrap();

    // Two tiles in each mode of a 32x16 column-major tensor.
    let data: [i32; 32 * 16] = core::array::from_fn(|i| i as i32);
    let src = make_tensor(
        data.as_ptr(),
        LayoutLeft::make(shape![Int::<32>, Int::<16>]).unwrap(),
    );
    for t in 0..32i32 {
        let part = tiled.get_slice(t).partition_s(src).unwrap();
        // ((AtomVal, RestVal), RestM, RestN), as upstream partition_S.
        assert_eq!(
            part.layout.shape,
            shape![(Int::<1>, Int::<4>), Int::<2>, Int::<2>]
        );
        assert_eq!(
            part.layout.stride,
            stride![(Int::<0>, Int::<1>), Int::<16>, Int::<256>]
        );
        for v in 0..4 {
            for (rm, rn) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (m, n) = (4 * (t % 4) + v + 16 * rm, t / 4 + 8 * rn);
                // SAFETY: every coordinate maps inside `data`.
                let got = unsafe { part.read(shape![(0i32, v), rm, rn]) };
                assert_eq!(got, m + 32 * n);
            }
        }
    }

    // The destination side partitions a row-major tile with the same map.
    let mut out = [0i32; 16 * 8];
    let dst = make_tensor(
        out.as_mut_ptr(),
        LayoutRight::make(shape![Int::<16>, Int::<8>]).unwrap(),
    );
    let part = tiled.get_slice(5i32).partition_d(dst).unwrap();
    assert_eq!(
        part.layout.stride,
        stride![(Int::<0>, Int::<8>), Int::<0>, Int::<0>]
    );
    assert_eq!(part.data(), out.as_mut_ptr().wrapping_add(4 * 8 + 1));
}

#[test]
fn copy_atoms_move_whole_registers() {
    // A 128-bit copy moves the four consecutive values of each thread at once.
    let atom = CopyAtom::<UniversalCopy<u128>, f32>::new();
    let threads = LayoutLeft::make(shape![Int::<4>, Int::<8>]).unwrap();
    let values = LayoutLeft::make(shape![Int::<4>, Int::<1>]).unwrap();
    let tiled = make_tiled_copy(atom, threads, values).unwrap();
    assert_eq!(tiled.tiler, shape![Int::<16>, Int::<8>]);

    let tile = LayoutLeft::make(shape![Int::<16>, Int::<8>]).unwrap();
    let mut src = make_owned_tensor::<f32, _>(tile);
    let mut dst = make_owned_tensor::<f32, _>(tile);
    src.engine
        .0
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = i as f32);
    for t in 0..32i32 {
        let thr = tiled.get_slice(t);
        let s = thr.partition_s(src.view()).unwrap();
        let d = thr.partition_d(dst.view_mut()).unwrap();
        // (AtomVal, RestVal) = (4, 1): one instruction per thread.
        assert_eq!(
            s.layout.shape,
            shape![(Int::<4>, Int::<1>), Int::<1>, Int::<1>]
        );
        // SAFETY: the partitions stay inside the 16-byte aligned fragments.
        unsafe { tiled.atom.call(&s, &d) };
    }
    assert_eq!(dst.engine.0, src.engine.0);
}

#[test]
fn tiled_mma_interleaves_atoms_over_the_tile() {
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<f32>>::new(),
        LayoutLeft::make(shape![Int::<4>, Int::<2>]).unwrap(),
        IntTuple(X, Nil),
    )
    .unwrap();
    assert_eq!(
        mma.thr_layout_vmnk.shape,
        shape![Int::<1>, Int::<4>, Int::<2>, Int::<1>]
    );
    assert_eq!(
        mma.thr_layout_vmnk.stride,
        stride![Int::<0>, Int::<1>, Int::<4>, Int::<0>]
    );
    assert_eq!(mma.permutation_mnk, shape![Int::<4>, Int::<2>, Int::<1>]);

    let a = [0f32; 8 * 2];
    let b = [0f32; 4 * 2];
    let c = [0f32; 8 * 4];
    let ga = make_tensor(
        a.as_ptr(),
        LayoutLeft::make(shape![Int::<8>, Int::<2>]).unwrap(),
    );
    let gb = make_tensor(
        b.as_ptr(),
        LayoutLeft::make(shape![Int::<4>, Int::<2>]).unwrap(),
    );
    let gc = make_tensor(
        c.as_ptr(),
        LayoutLeft::make(shape![Int::<8>, Int::<4>]).unwrap(),
    );
    // Thread 5 is (v, m, n, k) = (0, 1, 1, 0).
    let thr = mma.get_slice(5i32);
    let ta = thr.partition_a(ga).unwrap();
    assert_eq!(ta.layout.shape, shape![Int::<1>, Int::<2>, Int::<2>]);
    assert_eq!(ta.layout.stride, stride![Int::<0>, Int::<4>, Int::<8>]);
    assert_eq!(ta.data(), a.as_ptr().wrapping_add(1));
    let tb = thr.partition_b(gb).unwrap();
    assert_eq!(tb.layout.shape, shape![Int::<1>, Int::<2>, Int::<2>]);
    assert_eq!(tb.layout.stride, stride![Int::<0>, Int::<2>, Int::<4>]);
    assert_eq!(tb.data(), b.as_ptr().wrapping_add(1));
    let tc = thr.partition_c(gc).unwrap();
    assert_eq!(tc.layout.stride, stride![Int::<0>, Int::<4>, Int::<16>]);
    assert_eq!(tc.data(), c.as_ptr().wrapping_add(1 + 8));

    let fa = thr.partition_fragment_a(ga).unwrap();
    assert_eq!(fa.layout.shape, shape![Int::<1>, Int::<2>, Int::<2>]);
    let fc = thr.partition_fragment_c(gc).unwrap();
    assert_eq!(fc.layout.shape, shape![Int::<1>, Int::<2>, Int::<2>]);
    assert_eq!(fc.engine.0.len(), 4);
}

#[test]
fn tiled_mma_partitions_compute_a_gemm() {
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<f32>>::new(),
        LayoutLeft::make(shape![Int::<4>, Int::<2>]).unwrap(),
        IntTuple(X, Nil),
    )
    .unwrap();
    let a: [f32; 8 * 3] = core::array::from_fn(|i| (i % 7) as f32);
    let b: [f32; 4 * 3] = core::array::from_fn(|i| (i % 5) as f32 - 2.0);
    let mut c = [0f32; 8 * 4];
    let ga = make_tensor(
        a.as_ptr(),
        LayoutLeft::make(shape![Int::<8>, Int::<3>]).unwrap(),
    );
    let gb = make_tensor(
        b.as_ptr(),
        LayoutLeft::make(shape![Int::<4>, Int::<3>]).unwrap(),
    );
    let gc = make_tensor(
        c.as_mut_ptr(),
        LayoutLeft::make(shape![Int::<8>, Int::<4>]).unwrap(),
    );
    for t in 0..8i32 {
        let thr = mma.get_slice(t);
        let (ta, tb, tc) = (
            thr.partition_a(ga).unwrap(),
            thr.partition_b(gb).unwrap(),
            thr.partition_c(gc).unwrap(),
        );
        // Accumulate in registers, then store: (V, M, K) x (V, N, K) => (V, M, N).
        let mut acc = thr.partition_fragment_c(gc).unwrap();
        // SAFETY: each thread touches its own partition; the fragments are owned.
        unsafe {
            clear(&acc.view_mut());
            gemm_fma(&ta, &tb, &acc.view_mut());
            copy(&acc.view(), &tc);
        }
    }
    for m in 0..8 {
        for n in 0..4 {
            let expected: f32 = (0..3).map(|k| a[m + 8 * k] * b[n + 4 * k]).sum();
            assert_eq!(c[m + 8 * n], expected);
        }
    }
}

#[test]
fn tilings_partition_swizzled_shared_memory() {
    use noodle_cute::swizzle::Swizzle;
    type Sw = Swizzle<2, 2, 3>;
    let sw = Swizzle::<2, 2, 3>;
    let tile = LayoutLeft::make(shape![Int::<16>, Int::<8>]).unwrap();
    let smem_layout = sw.compose(tile);

    // gmem -> smem through a tiled copy: every element lands at its swizzled address.
    let atom = CopyAtom::<UniversalCopy<i32>, i32>::new();
    let tiled = make_tiled_copy(
        atom,
        LayoutLeft::make(shape![Int::<4>, Int::<8>]).unwrap(),
        LayoutLeft::make(shape![Int::<4>, Int::<1>]).unwrap(),
    )
    .unwrap();
    let data: [i32; 128] = core::array::from_fn(|i| i as i32);
    let mut smem = [-1i32; 128];
    let src = make_tensor(data.as_ptr(), tile);
    let dst = make_tensor(smem.as_mut_ptr(), smem_layout);
    for t in 0..32i32 {
        let thr = tiled.get_slice(t);
        let (s, d) = (thr.partition_s(src).unwrap(), thr.partition_d(dst).unwrap());
        // The swizzle stays outside; the pointer is not moved.
        assert_eq!((d.layout.layout_a, d.data()), (sw, smem.as_mut_ptr()));
        // SAFETY: both partitions address 0..128 of their arrays.
        unsafe { tiled.atom.call(&s, &d) };
    }
    for i in 0..128isize {
        assert_eq!(smem[Sw::apply(i) as usize], i as i32);
    }

    // smem -> MMA: partitions of the swizzled tile read the logical A(m, k).
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<i32>>::new(),
        LayoutLeft::make(shape![Int::<4>, Int::<2>]).unwrap(),
        IntTuple(X, Nil),
    )
    .unwrap();
    let s_a = make_tensor(smem.as_ptr(), smem_layout);
    let thr = mma.get_slice(5i32);
    let t_a = thr.partition_a(s_a).unwrap();
    assert_eq!(t_a.layout.layout_a, sw);
    assert_eq!(
        t_a.layout.layout_b.shape,
        shape![Int::<1>, Int::<4>, Int::<8>]
    );
    for rm in 0..4i32 {
        for k in 0..8i32 {
            // Thread 5 owns rows 1, 5, 9, 13.
            // SAFETY: the partition addresses swizzled 0..128.
            let got = unsafe { t_a.read(shape![0i32, rm, k]) };
            assert_eq!(got, 1 + 4 * rm + 16 * k);
        }
    }
    // partition_fragment_A of a swizzled tile is the fragment of its layout_b.
    let frag = thr.partition_fragment_a(s_a).unwrap();
    assert_eq!(frag.layout.shape, shape![Int::<1>, Int::<4>, Int::<8>]);
    assert_eq!(frag.engine.0.len(), 32);
}
