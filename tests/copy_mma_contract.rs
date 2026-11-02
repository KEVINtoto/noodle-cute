//! CuTe copy_atom.hpp: MMA-compatible TV maps, fragment retiling and predicates.
use noodle_cute::algorithm::{copy_if_with, copy_with};
use noodle_cute::atom::{
    CopyAtom, MmaAtom, UniversalCopy, UniversalFma, make_tiled_copy_a, make_tiled_copy_b,
    make_tiled_copy_c, make_tiled_mma,
};
use noodle_cute::coord::X;
use noodle_cute::layout::LayoutLeft;
use noodle_cute::tensor::{make_identity_tensor, make_tensor};
use noodle_cute::{Int, IntTuple, Nil, shape, stride};

#[test]
fn mma_copy_tv_broadcasts_unused_thread_modes() {
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<f32>>::new(),
        LayoutLeft::make(shape![Int::<2>, Int::<2>, Int::<2>]).unwrap(),
        IntTuple(X, Nil),
    )
    .unwrap();
    let atom = CopyAtom::<UniversalCopy<f32>, f32>::new();
    let a = make_tiled_copy_a(atom, &mma).unwrap();
    let b = make_tiled_copy_b(atom, &mma).unwrap();
    let c = make_tiled_copy_c(atom, &mma).unwrap();
    assert_eq!(a.tiler, shape![Int::<2>, Int::<2>]);
    assert_eq!(b.tiler, shape![Int::<2>, Int::<2>]);
    assert_eq!(c.tiler, shape![Int::<2>, Int::<2>]);
    for t in 0..8isize {
        assert_eq!(
            a.layout_tv.crd2idx(shape![t, 0isize]).value(),
            t % 2 + 2 * (t / 4)
        );
        assert_eq!(
            b.layout_tv.crd2idx(shape![t, 0isize]).value(),
            (t / 2) % 2 + 2 * (t / 4)
        );
        assert_eq!(c.layout_tv.crd2idx(shape![t, 0isize]).value(), t % 4);
    }
}

#[test]
fn identity_tensors_follow_copy_and_mma_tail_partitions() {
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<f32>>::new(),
        LayoutLeft::make(shape![Int::<2>, Int::<2>]).unwrap(),
        IntTuple(X, Nil),
    )
    .unwrap();
    let tiled = make_tiled_copy_c(CopyAtom::<UniversalCopy<f32>, f32>::new(), &mma).unwrap();
    let extent = shape![3isize, 3isize];
    let identity = make_identity_tensor(extent);
    for t in 0..4isize {
        let cp = tiled.get_slice(t).partition_s(identity).unwrap();
        let mp = mma.get_slice(t).partition_c(identity).unwrap();
        for rn in 0..2isize {
            for rm in 0..2isize {
                let expected = shape![t % 2 + 2 * rm, t / 2 + 2 * rn];
                let got = cp.get(shape![0isize, rm, rn]);
                let mgot = mp.get(shape![0isize, rm, rn]);
                assert_eq!((got.0.value(), got.1.0.value()), (expected.0, expected.1.0));
                assert_eq!(
                    (mgot.0.value(), mgot.1.0.value()),
                    (expected.0, expected.1.0)
                );
            }
        }
    }
}

#[test]
fn mma_fragment_retile_preserves_storage() {
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<f32>>::new(),
        LayoutLeft::make(shape![Int::<2>, Int::<2>]).unwrap(),
        shape![Int::<4>, Int::<4>],
    )
    .unwrap();
    let tiled = make_tiled_copy_c(CopyAtom::<UniversalCopy<f32>, f32>::new(), &mma).unwrap();
    let mut values: [f32; 64] = core::array::from_fn(|i| i as f32);
    let tensor = make_tensor(
        values.as_mut_ptr(),
        LayoutLeft::make(shape![Int::<8>, Int::<8>]).unwrap(),
    );
    let thr = mma.get_slice(1isize);
    let mut fragment = thr.partition_fragment_c(tensor).unwrap();
    let view = fragment.view_mut();
    let copy_thr = tiled.get_slice(1isize);
    let retiled = copy_thr.retile_d(view).unwrap();
    assert_eq!(retiled.data(), view.data());
    let src = copy_thr.partition_s(tensor).unwrap();
    unsafe { copy_with(&tiled, &src, &retiled) };
    let retiled_s = copy_thr.retile_s(view).unwrap();
    assert_eq!(retiled_s.layout, retiled.layout);
    assert_eq!(src.size(), retiled.size());
    let mma_part = thr.partition_c(tensor).unwrap();
    for i in 0..mma_part.size() {
        assert_eq!(unsafe { fragment.view().read(i) }, unsafe {
            mma_part.read(i)
        });
    }
}

use noodle_cute::atom::{CopyTraits, MmaTraits};
use noodle_cute::pointer::{Read, Write};
use noodle_cute::{DynInt, Layout};

type P<A, B> = IntTuple<A, IntTuple<B, Nil>>;
type Row16x8 =
    Layout<P<P<Int<4>, Int<8>>, P<Int<2>, Int<2>>>, P<P<Int<32>, Int<1>>, P<Int<16>, Int<8>>>>;
type Row8x8 = Layout<P<P<Int<4>, Int<8>>, Int<2>>, P<P<Int<16>, Int<1>>, Int<8>>>;
// Layout-only fixture from mma_traits_sm80.hpp; no GPU operation is provided.
struct WarpMma;
impl MmaTraits for WarpMma {
    type ValA = u16;
    type ValB = u16;
    type ValC = u16;
    type ShapeMnk = IntTuple<Int<16>, IntTuple<Int<8>, IntTuple<Int<8>, Nil>>>;
    type ThrId = Layout<Int<32>, Int<1>>;
    type ALayout = Row16x8;
    type BLayout = Row8x8;
    type CLayout = Row16x8;
    type DRegister = u32;
    type ARegister = u32;
    type BRegister = u32;
    type CRegister = u32;
    const D_REGISTERS: isize = 2;
    const A_REGISTERS: isize = 2;
    const B_REGISTERS: isize = 1;
    const C_REGISTERS: isize = 2;
    unsafe fn fma<DP, AP, BP, CP>(
        _: impl Fn(isize) -> DP,
        _: impl Fn(isize) -> AP,
        _: impl Fn(isize) -> BP,
        _: impl Fn(isize) -> CP,
    ) where
        DP: Write<Value = u32>,
        AP: Read<Value = u32>,
        BP: Read<Value = u32>,
        CP: Read<Value = u32>,
    {
        panic!("layout-only fixture");
    }
}

#[test]
fn warp_operand_tv_maps_match_upstream_lane_coordinates() {
    let mma = make_tiled_mma(
        MmaAtom::<WarpMma>::new(),
        LayoutLeft::make(shape![Int::<2>, Int::<2>]).unwrap(),
        IntTuple(X, Nil),
    )
    .unwrap();
    let atom = CopyAtom::<UniversalCopy<u16>, u16>::new();
    let a = make_tiled_copy_a(atom, &mma).unwrap();
    let b = make_tiled_copy_b(atom, &mma).unwrap();
    let c = make_tiled_copy_c(atom, &mma).unwrap();
    for t in 0..128isize {
        let (lane, wm, wn) = (t % 32, (t / 32) % 2, t / 64);
        for v in 0..4isize {
            let m = lane / 4 + 16 * wm + 8 * (v / 2);
            let k = 2 * (lane % 4) + v % 2;
            assert_eq!(a.layout_tv.crd2idx(shape![t, v]).value(), m + 32 * k);
            let n = 2 * (lane % 4) + v % 2 + 8 * wn;
            assert_eq!(c.layout_tv.crd2idx(shape![t, v]).value(), m + 32 * n);
        }
        for v in 0..2isize {
            let n = lane / 4 + 8 * wn;
            let k = 2 * (lane % 4) + v;
            assert_eq!(b.layout_tv.crd2idx(shape![t, v]).value(), n + 16 * k);
        }
    }
}

#[test]
fn warp_retile_gathers_values_from_mma_rest_modes() {
    let mma = make_tiled_mma(
        MmaAtom::<WarpMma>::new(),
        LayoutLeft::make(shape![Int::<1>]).unwrap(),
        shape![Int::<16>, Int::<16>],
    )
    .unwrap();
    let tiled = make_tiled_copy_c(CopyAtom::<Gather8, u16>::new(), &mma).unwrap();
    let data: [u16; 16 * 32] = core::array::from_fn(|i| i as u16);
    let tensor = make_tensor(
        data.as_ptr(),
        LayoutLeft::make(shape![Int::<16>, Int::<32>]).unwrap(),
    );
    for t in 0..32isize {
        let thr = mma.get_slice(t);
        let mut fragment = thr.partition_fragment_c(tensor).unwrap();
        let mma_part = thr.partition_c(tensor).unwrap();
        let copy_thr = tiled.get_slice(t);
        let src = copy_thr.partition_s(tensor).unwrap();
        let dst = copy_thr.retile_d(fragment.view_mut()).unwrap();
        assert_eq!(dst.size(), mma_part.size());
        // Retile gathers two adjacent MMA N repetitions into eight atom values.
        unsafe { copy_with(&copy_thr, &src, &dst) };
        for i in 0..mma_part.size() {
            assert_eq!(unsafe { fragment.view().read(i) }, unsafe {
                mma_part.read(i)
            });
        }
    }
}

#[test]
fn predicated_copy_dispatch_skips_invalid_source_and_converts_numbers() {
    use noodle_cute::numeric::F16;
    let input = [1.0f32, 2.0, 3.0];
    let mut output = [F16::from_f32(-1.0); 4];
    let src = make_tensor(input.as_ptr(), LayoutLeft::make(shape![Int::<4>]).unwrap());
    let dst = make_tensor(
        output.as_mut_ptr(),
        LayoutLeft::make(shape![Int::<4>]).unwrap(),
    );
    let atom = CopyAtom::<UniversalCopy<f32, F16>, f32>::new();
    unsafe { copy_if_with(&atom, |i| i < 3, &src, &dst) };
    assert_eq!(output.map(f32::from), [1.0, 2.0, 3.0, -1.0]);
    let invalid = make_tensor(core::ptr::null::<f32>(), src.layout);
    unsafe { copy_with(&atom.with(false), &invalid, &dst) };
    assert_eq!(output.map(f32::from), [1.0, 2.0, 3.0, -1.0]);
}

// An instruction with a zero-fill predicate, mirroring copy_traits_sm80.hpp.
struct ZeroFill;
impl CopyTraits for ZeroFill {
    type SrcLayout = <UniversalCopy<u32> as CopyTraits>::SrcLayout;
    type DstLayout = Self::SrcLayout;
    type RefLayout = Self::SrcLayout;
    type SRegister = u32;
    type DRegister = u32;
    const SRC_REGISTERS: isize = 1;
    const DST_REGISTERS: isize = 1;
    unsafe fn copy<SP, DP>(src: impl Fn(isize) -> SP, dst: impl Fn(isize) -> DP)
    where
        SP: Read<Value = u32>,
        DP: Write<Value = u32>,
    {
        unsafe { dst(0).write(src(0).read()) }
    }
    unsafe fn copy_predicated<SP, DP>(
        pred: bool,
        src: impl Fn(isize) -> SP,
        dst: impl Fn(isize) -> DP,
    ) where
        SP: Read<Value = u32>,
        DP: Write<Value = u32>,
    {
        unsafe {
            if pred {
                Self::copy(src, dst)
            } else {
                dst(0).write(0)
            }
        }
    }
}

#[test]
fn false_predicate_reaches_backend_zero_fill() {
    let mut output = [99u32; 4];
    let layout = LayoutLeft::make(shape![Int::<4>]).unwrap();
    let src = make_tensor(core::ptr::null::<u32>(), layout);
    let dst = make_tensor(output.as_mut_ptr(), layout);
    unsafe { copy_with(&CopyAtom::<ZeroFill, u32>::new().with(false), &src, &dst) };
    assert_eq!(output, [0; 4]);
}

#[test]
fn basis_composition_handles_nested_zero_and_mixed_strides() {
    use noodle_cute::basis::{ScaledBasis, make_identity_layout};
    let identity = make_identity_layout(shape![Int::<4>, DynInt(6)]);
    let composed = identity
        .composition(
            Layout::new(
                shape![(Int::<2>, Int::<2>), Int::<3>, Int::<2>],
                stride![(Int::<1>, Int::<2>), Int::<4>, Int::<0>],
            )
            .unwrap(),
        )
        .unwrap();
    let coord = composed.crd2idx(shape![3isize, 2isize, 1isize]);
    assert_eq!((coord.0.value(), coord.1.0.value()), (3, 2));
    let empty = identity
        .composition(Layout {
            shape: Nil,
            stride: Nil,
        })
        .unwrap();
    assert_eq!(empty.shape, Nil);
    // Repeated bases coalesce when their scaled strides are contiguous.
    let repeated = Layout {
        shape: shape![Int::<2>, Int::<3>],
        stride: IntTuple(
            ScaledBasis::<Int<1>, 0>(Int),
            IntTuple(ScaledBasis::<Int<2>, 0>(Int), Nil),
        ),
    };
    let mapped = repeated
        .composition(Layout::new(Int::<6>, Int::<1>).unwrap())
        .unwrap();
    assert_eq!(mapped.crd2idx(5isize).0.value(), 5);
}

struct Gather8;
impl CopyTraits for Gather8 {
    type SrcLayout = <UniversalCopy<u128> as CopyTraits>::SrcLayout;
    type DstLayout = Self::SrcLayout;
    type RefLayout = Self::SrcLayout;
    type SRegister = u16;
    type DRegister = u16;
    const SRC_REGISTERS: isize = 8;
    const DST_REGISTERS: isize = 8;
    unsafe fn copy<SP, DP>(src: impl Fn(isize) -> SP, dst: impl Fn(isize) -> DP)
    where
        SP: Read<Value = u16>,
        DP: Write<Value = u16>,
    {
        for i in 0..8 {
            unsafe { dst(i).write(src(i).read()) }
        }
    }
}

#[test]
fn mma_copy_rejects_atom_value_count_that_does_not_fit() {
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<f32>>::new(),
        LayoutLeft::make(shape![Int::<2>, Int::<2>]).unwrap(),
        IntTuple(X, Nil),
    )
    .unwrap();
    let result = make_tiled_copy_c(CopyAtom::<UniversalCopy<u128>, f32>::new(), &mma);
    assert!(matches!(result, Err(noodle_cute::Error::NotDivisible)));
}

#[test]
fn mma_copy_matches_partitions_with_reordered_threads_and_permutations() {
    let threads = Layout::new(
        shape![Int::<2>, Int::<2>, Int::<2>],
        stride![Int::<4>, Int::<2>, Int::<1>],
    )
    .unwrap();
    let perm = Layout::new(shape![Int::<2>, Int::<2>], stride![Int::<2>, Int::<1>]).unwrap();
    let mma = make_tiled_mma(
        MmaAtom::<UniversalFma<f32>>::new(),
        threads,
        IntTuple(perm, IntTuple(Int::<4>, Nil)),
    )
    .unwrap();
    let atom = CopyAtom::<UniversalCopy<f32>, f32>::new();
    let a = make_tiled_copy_a(atom, &mma).unwrap();
    let b = make_tiled_copy_b(atom, &mma).unwrap();
    let c = make_tiled_copy_c(atom, &mma).unwrap();
    let data: [f32; 16] = core::array::from_fn(|i| i as f32);
    let ab = make_tensor(
        data.as_ptr(),
        LayoutLeft::make(shape![Int::<4>, Int::<2>]).unwrap(),
    );
    let cv = make_tensor(
        data.as_ptr(),
        LayoutLeft::make(shape![Int::<4>, Int::<4>]).unwrap(),
    );
    for t in 0..8isize {
        let thr = mma.get_slice(t);
        let ma = thr.partition_a(ab).unwrap();
        let mb = thr.partition_b(ab).unwrap();
        let mc = thr.partition_c(cv).unwrap();
        let ca = a.get_slice(t).partition_s(ab).unwrap();
        let cb = b.get_slice(t).partition_s(ab).unwrap();
        let cc = c.get_slice(t).partition_s(cv).unwrap();
        assert_eq!(ca.size(), ma.size());
        assert_eq!(cb.size(), mb.size());
        assert_eq!(cc.size(), mc.size());
        for i in 0..ma.size() {
            assert_eq!(unsafe { ca.read(i) }, unsafe { ma.read(i) });
        }
        for i in 0..mb.size() {
            assert_eq!(unsafe { cb.read(i) }, unsafe { mb.read(i) });
        }
        for i in 0..mc.size() {
            assert_eq!(unsafe { cc.read(i) }, unsafe { mc.read(i) });
        }
    }
}

#[test]
fn operand_a_b_retile_preserves_broadcast_thread_fragments() {
    let mma = make_tiled_mma(
        MmaAtom::<WarpMma>::new(),
        LayoutLeft::make(shape![Int::<2>, Int::<2>]).unwrap(),
        shape![Int::<32>, Int::<16>, Int::<16>],
    )
    .unwrap();
    let a = make_tiled_copy_a(CopyAtom::<UniversalCopy<u16>, u16>::new(), &mma).unwrap();
    let b = make_tiled_copy_b(CopyAtom::<UniversalCopy<u16>, u16>::new(), &mma).unwrap();
    let data: [u16; 32 * 32] = core::array::from_fn(|i| i as u16);
    let ta = make_tensor(
        data.as_ptr(),
        LayoutLeft::make(shape![Int::<32>, Int::<32>]).unwrap(),
    );
    let tb = make_tensor(
        data.as_ptr(),
        LayoutLeft::make(shape![Int::<16>, Int::<32>]).unwrap(),
    );
    for t in [0isize, 17, 32, 63, 64, 101, 127] {
        let thr = mma.get_slice(t);
        let mut fa = thr.partition_fragment_a(ta).unwrap();
        let mut fb = thr.partition_fragment_b(tb).unwrap();
        let sa = a.get_slice(t).partition_s(ta).unwrap();
        let sb = b.get_slice(t).partition_s(tb).unwrap();
        let da = a.get_slice(t).retile_d(fa.view_mut()).unwrap();
        let db = b.get_slice(t).retile_d(fb.view_mut()).unwrap();
        unsafe {
            copy_with(&a, &sa, &da);
            copy_with(&b, &sb, &db);
        }
        let ma = thr.partition_a(ta).unwrap();
        let mb = thr.partition_b(tb).unwrap();
        for i in 0..ma.size() {
            assert_eq!(unsafe { fa.view().read(i) }, unsafe { ma.read(i) });
        }
        for i in 0..mb.size() {
            assert_eq!(unsafe { fb.view().read(i) }, unsafe { mb.read(i) });
        }
    }
}

// SM75_U32x4_LDSM_N operand layouts; execution requires a GPU backend.
struct Ldsm4;
impl CopyTraits for Ldsm4 {
    type SrcLayout = Layout<P<Int<32>, Int<128>>, P<Int<128>, Int<1>>>;
    type DstLayout = Layout<P<Int<32>, P<Int<32>, Int<4>>>, P<Int<32>, P<Int<1>, Int<1024>>>>;
    type RefLayout = Self::DstLayout;
    type SRegister = u128;
    type DRegister = u32;
    const SRC_REGISTERS: isize = 1;
    const DST_REGISTERS: isize = 4;
    unsafe fn copy<SP, DP>(_: impl Fn(isize) -> SP, _: impl Fn(isize) -> DP)
    where
        SP: Read<Value = u128>,
        DP: Write<Value = u32>,
    {
        panic!("layout-only fixture");
    }
}

#[test]
fn ldsm_source_partition_connects_swizzled_memory_to_mma_reference_values() {
    use noodle_cute::swizzle::Swizzle;
    let mma = make_tiled_mma(
        MmaAtom::<WarpMma>::new(),
        LayoutLeft::make(shape![Int::<2>, Int::<2>]).unwrap(),
        shape![Int::<32>, Int::<16>, Int::<16>],
    )
    .unwrap();
    let tiled = make_tiled_copy_a(CopyAtom::<Ldsm4, u16>::new(), &mma).unwrap();
    let data: [u16; 512] = core::array::from_fn(|i| i as u16);
    let memory = make_tensor(
        data.as_ptr(),
        Swizzle::<2, 3, 3>.compose(LayoutLeft::make(shape![Int::<32>, Int::<16>]).unwrap()),
    );
    let identity = make_identity_tensor(shape![Int::<32>, Int::<16>]);
    for t in 0..128isize {
        let cp = tiled.get_slice(t).partition_s(memory).unwrap();
        let coords = tiled.get_slice(t).partition_s(identity).unwrap();
        for v in 0..8isize {
            // SrcLayout is 8*t+v; invert RefLayout (2*r + rv%2 + 64*(rv/2)).
            let x = 8 * (t % 32) + v;
            let r = (x % 64) / 2;
            let rv = x % 2 + 2 * (x / 64);
            let m = r / 4 + 8 * ((rv / 2) % 2) + 16 * ((t / 32) % 2);
            let k = 2 * (r % 4) + rv % 2 + 8 * (rv / 4);
            let c = coords.get(v);
            assert_eq!((c.0.value(), c.1.0.value()), (m, k));
            let raw = m + 32 * k;
            let swizzled = raw ^ ((raw & (3 << 6)) >> 3);
            assert_eq!(unsafe { cp.read(v) }, swizzled as u16);
        }
    }
}
