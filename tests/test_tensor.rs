#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl
)]
#![allow(incomplete_features)]

use std::panic::{AssertUnwindSafe, catch_unwind};

use noodle_cute::algebra::{
    Composition, LogicalDivide, LogicalDivideAt, ZippedDivide, ZippedDivideAt, blocked_product,
    coalesce, coalesce_extended, coalesce_with_profile, coalesce_z, composition,
    greatest_common_domain, logical_divide, logical_divide_at, zipped_divide, zipped_divide_at,
};
use noodle_cute::basis::{BasisTerm, ModeEnd, as_tuple, basis_eq};
use noodle_cute::f2::f2;
use noodle_cute::integer::{Int, int};
use noodle_cute::layout::{Layout, is_layout};
use noodle_cute::shape::{compatible, idx2crd, in_bounds};
use noodle_cute::stride::linear_bounds;
use noodle_cute::swizzle::Swizzle;
use noodle_cute::tensor::{
    Accessor, AccessorOffset, CellAccessor, ImplicitAccessor, MutableAccessor, Ptr, SliceAccessor,
    Tensor, TransformAccessor, identity_tensor, is_tensor, local_tile, make_tensor,
};
use noodle_cute::tuple::{LeafProfile, Nil, NilProfile, TupleProfile, dice_by, select, slice_by};
use noodle_cute::{basis_tuple, int_tuple, mode, tuple};

// ------------------------------------------------------------------
// Lower-layer helpers: slice_by, dice_by, idx2crd(..), Layout::offset_and_slice
// ------------------------------------------------------------------

#[test]
fn test_tuple_slice_by_and_dice_by() {
    let target = tuple!((2i32, 3i32), (5i32, 7i32, 9i32));
    assert_eq!(slice_by(&tuple!(.., 1i32), &target), tuple!((2i32, 3i32)));
    assert_eq!(
        dice_by(&tuple!(.., 1i32), &target),
        tuple!((5i32, 7i32, 9i32))
    );

    assert_eq!(
        slice_by(&tuple!(.., ..), &tuple!(2i32, 3i32)),
        tuple!(2i32, 3i32)
    );
    assert_eq!(dice_by(&tuple!(.., ..), &tuple!(2i32, 3i32)), Nil);

    assert_eq!(slice_by(&tuple!(1i32, 2i32), &tuple!(5i32, 7i32)), Nil);
    assert_eq!(
        dice_by(&tuple!(1i32, 2i32), &tuple!(5i32, 7i32)),
        tuple!(5i32, 7i32)
    );

    // Wildcard as scalar leaf keeps entire target subtree wrapped in a 1-tuple
    assert_eq!(slice_by(&.., &tuple!(4i32, 6i32)), tuple!((4i32, 6i32)));
    assert_eq!(dice_by(&.., &tuple!(4i32, 6i32)), Nil);

    // Nested partial coordinate (used by copy nullspace slicing: [(0, ..), ..])
    let nested = int_tuple![(Int::<2>, Int::<3>), Int::<4>];
    assert_eq!(
        slice_by(&tuple!((0i32, ..), ..), &nested),
        int_tuple![Int::<3>, Int::<4>]
    );
    assert_eq!(
        dice_by(&tuple!((0i32, ..), ..), &nested),
        int_tuple![Int::<2>]
    );

    const SLICED: noodle_cute::tuple::Til<Int<4>, Nil> =
        slice_by(&tuple!(1i32, ..), &int_tuple![Int::<3>, Int::<4>]);
    assert_eq!(SLICED, int_tuple![Int::<4>]);
}

#[test]
fn test_idx2crd_wildcard_and_layout_offset_and_slice() {
    let shape = int_tuple![Int::<3>, (Int::<2>, Int::<4>)];
    assert_eq!(
        idx2crd(.., &shape),
        int_tuple![Int::<0>, (Int::<0>, Int::<0>)]
    );
    assert_eq!(
        idx2crd(tuple!(1i32, ..), &shape),
        int_tuple![1i32, (Int::<0>, Int::<0>)]
    );

    let layout = Layout::new(
        int_tuple![Int::<4>, Int::<4>],
        int_tuple![Int::<4>, Int::<1>],
    );
    let (off, sliced) = layout.offset_and_slice(tuple!(1i32, ..));
    assert_eq!(off, 4i32);
    assert_eq!(
        sliced,
        Layout::new(int_tuple![Int::<4>], int_tuple![Int::<1>])
    );

    let (off2, sliced2) = layout.offset_and_slice(tuple!(.., 2i32));
    assert_eq!(off2, 2i32);
    assert_eq!(
        sliced2,
        Layout::new(int_tuple![Int::<4>], int_tuple![Int::<4>])
    );

    let (off_full, sliced_full) = layout.offset_and_slice(tuple!(1i32, 2i32));
    assert_eq!(off_full, 6i32);
    assert_eq!(sliced_full, Layout::new(Nil, Nil));
}

// ------------------------------------------------------------------
// TestTensor (matching test/test_tensor.py::TestTensor)
// ------------------------------------------------------------------

#[test]
fn test_make_tensor() {
    let mut buf = [0.0f64; 16];
    let t = make_tensor(
        &mut buf,
        Layout::new(
            int_tuple![Int::<4>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>],
        ),
    );
    assert!(is_tensor(&t));
    assert!(!is_layout(&t));
    assert_eq!(
        t.layout,
        Layout::new(
            int_tuple![Int::<4>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>],
        )
    );

    // Also accepts a shape (promoted to compact column-major layout)
    let mut buf2 = [0.0f64; 15];
    let t2 = make_tensor(&mut buf2, int_tuple![Int::<3>, Int::<5>]);
    assert_eq!(
        t2.layout,
        Layout::new(
            int_tuple![Int::<3>, Int::<5>],
            int_tuple![Int::<1>, Int::<3>],
        )
    );

    // Scalar shape
    let mut buf3 = [0i32; 6];
    let t3 = make_tensor(&mut buf3, Int::<6>);
    assert_eq!(t3.layout, Layout::new(Int::<6>, Int::<1>));
    assert_eq!(t3.size(), 6);
    assert_eq!(t3.cosize(), 6);

    // Const make_tensor over immutable array
    const DATA: [i32; 6] = [1, 2, 3, 4, 5, 6];
    const ELEM: i32 = make_tensor(&DATA, int_tuple![Int::<2>, Int::<3>])(1i32, 2i32);
    assert_eq!(ELEM, 6);

    // Rejects backing storage smaller than coshape(layout)
    let small = [0i32; 10];
    assert!(
        catch_unwind(|| make_tensor(
            &small,
            Layout::new(
                int_tuple![Int::<4>, Int::<4>],
                int_tuple![Int::<4>, Int::<1>],
            )
        ))
        .is_err()
    );
}

#[test]
fn test_tensor_evaluation_three_coords() {
    let mut buf = [0i32; 16];
    let mut t = make_tensor(
        &mut buf,
        Layout::new(
            int_tuple![Int::<4>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>],
        ),
    );
    for i in 0..t.size() {
        t.set(i, i);
    }

    // 1-D integral coordinate and 2-D flat coordinate map to the same backing offset
    assert_eq!(t(6i32), t.accessor.load(t.layout.call(6i32)));
    assert_eq!(
        t(1i32, 2i32),
        t.accessor.load(t.layout.call(int_tuple![1i32, 2i32]))
    );
    assert_eq!(t.at(int_tuple![1i32, 2i32]), t(1i32, 2i32));

    // Hierarchical natural coordinate
    let mut h_buf = [0i32; 24];
    let mut h = make_tensor(
        &mut h_buf,
        Layout::new(
            int_tuple![Int::<3>, (Int::<2>, Int::<4>)],
            int_tuple![Int::<2>, (Int::<1>, Int::<6>)],
        ),
    );
    for i in 0..h.size() {
        h.set(i, i * 10);
    }
    assert_eq!(h(17i32), h(2i32, 5i32));
    assert_eq!(h(17i32), h(2i32, int_tuple![1i32, 2i32]));
}

#[test]
fn test_tensor_set_and_get() {
    let mut buf = [0i32; 16];
    let mut t = make_tensor(
        &mut buf,
        Layout::new(
            int_tuple![Int::<4>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>],
        ),
    );
    t.set(int_tuple![1i32, 2i32], 42);
    t.set(int_tuple![3i32, 0i32], 99);
    assert_eq!(t(1i32, 2i32), 42);
    assert_eq!(t(3i32, 0i32), 99);
    assert_eq!(t(0i32, 0i32), 0);

    // Setting via 1-D index also works
    t.set(0i32, 7);
    assert_eq!(t(0i32, 0i32), 7);
}

#[test]
fn test_tensor_slicing_returns_subtensor() {
    let mut buf = [0i32; 16];
    let mut t = make_tensor(
        &mut buf,
        Layout::new(
            int_tuple![Int::<4>, Int::<4>],
            int_tuple![Int::<4>, Int::<1>],
        ),
    );
    for i in 0..t.size() {
        t.set(i, i);
    }

    // Row 1 of T: T[1, j] for j = 0..3
    let row = t(1i32, ..);
    assert!(is_tensor(&row));
    assert_eq!(
        row.layout,
        Layout::new(int_tuple![Int::<4>], int_tuple![Int::<1>])
    );
    let row_vals: Vec<i32> = (0..4).map(|j| row(j)).collect();
    let expected_row: Vec<i32> = (0..4).map(|j| t(1i32, j)).collect();
    assert_eq!(row_vals, expected_row);
    assert_eq!(row_vals, vec![1, 5, 9, 13]);

    // Column 2 of T: T[i, 2] for i = 0..3
    let col = t(.., 2i32);
    assert!(is_tensor(&col));
    assert_eq!(
        col.layout,
        Layout::new(int_tuple![Int::<4>], int_tuple![Int::<4>])
    );
    let col_vals: Vec<i32> = (0..4).map(|i| col(i)).collect();
    let expected_col: Vec<i32> = (0..4).map(|i| t(i, 2i32)).collect();
    assert_eq!(col_vals, expected_col);
    assert_eq!(col_vals, vec![8, 9, 10, 11]);
}

#[test]
fn test_tensor_slicing_provides_view_semantics() {
    let mut buf = [0.0f64; 120];
    let mut a = make_tensor(&mut buf, int_tuple![Int::<4>, Int::<5>, Int::<6>]);
    a.set(int_tuple![1i32, 2i32, 3i32], 42.0);

    let mut v1 = a(.., 2i32, ..);
    let v2 = a(.., 2i32, ..);

    // Stable printed identity (underlying pointer address)
    assert_eq!(format!("{}", v1.accessor), format!("{}", v2.accessor));
    assert_eq!(format!("{v1}"), format!("{v2}"));

    // Value equality on accessors and tensors
    assert_eq!(v1.accessor, v2.accessor);
    assert_eq!(v1, v2);

    // Mutations through one view propagate to the base and the other view
    v1.set(int_tuple![1i32, 3i32], 99.0);
    assert_eq!(a(1i32, 2i32, 3i32), 99.0);
    assert_eq!(v2(1i32, 3i32), 99.0);
}

#[test]
fn test_tensor_algebra_passes_through_to_layout() {
    let mut buf = [0i32; 16];
    let mut t = make_tensor(
        &mut buf,
        Layout::new(
            int_tuple![Int::<4>, Int::<4>],
            int_tuple![Int::<1>, Int::<4>],
        ),
    );
    t.set(int_tuple![1i32, 2i32], 42); // offset L(1, 2) = 9

    let tc = coalesce(t);
    assert!(is_tensor(&tc));
    assert_eq!(tc.layout, Layout::new(Int::<16>, Int::<1>));
    assert_eq!(tc.accessor, t.accessor);
    assert!(tc.accessor.same_base(&t.accessor));
    assert_eq!(tc(9i32), 42);
    assert_eq!(t(1i32, 2i32), 42);

    // coalesce_extended / coalesce_z
    let mut buf_ext = [0i32; 2];
    let t_ext = make_tensor(
        &mut buf_ext,
        Layout::new(
            int_tuple![Int::<2>, Int::<1>],
            int_tuple![Int::<1>, Int::<0>],
        ),
    );
    assert_eq!(
        coalesce_extended(t_ext).layout,
        Layout::new(
            int_tuple![Int::<2>, Int::<1>],
            int_tuple![Int::<1>, Int::<0>],
        )
    );
    assert_eq!(
        coalesce_z::<TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>, _>(t).layout,
        t.layout
    );

    // composition
    let mut buf24 = [0i32; 24];
    let t_24 = make_tensor(&mut buf24, Layout::new(Int::<24>, Int::<1>));
    let t_comp = composition(t_24, Layout::from_shape(int_tuple![Int::<4>, Int::<6>]));
    assert_eq!(
        t_comp.layout,
        Layout::new(
            int_tuple![Int::<4>, Int::<6>],
            int_tuple![Int::<1>, Int::<4>],
        )
    );
    assert!(t_comp.accessor.same_base(&t_24.accessor));

    // logical_divide & zipped_divide
    let td = logical_divide(t, Layout::new(Int::<2>, Int::<1>));
    assert_eq!(td.rank(), 2);
    assert_eq!(td.accessor, t.accessor);

    let td_at = logical_divide_at::<mode!(1), _, _>(t, Int::<2>);
    assert_eq!(td_at.shape(), int_tuple![Int::<4>, (Int::<2>, Int::<2>)]);
    assert!(td_at.accessor.same_base(&t.accessor));

    let tz = zipped_divide(t, int_tuple![Int::<2>, Int::<2>]);
    assert_eq!(
        tz.shape(),
        int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<2>)]
    );
    assert!(tz.accessor.same_base(&t.accessor));

    let tz_at = zipped_divide_at::<mode!(0), _, _>(t, Int::<2>);
    assert_eq!(tz_at.shape(), int_tuple![(Int::<2>, Int::<2>), Int::<4>]);

    // greatest_common_domain accepts Tensor by value and by reference
    let gcd_val = greatest_common_domain(t, t_comp);
    let gcd_ref = greatest_common_domain(&t, &t_comp);
    assert_eq!(
        gcd_val,
        Layout::new(
            int_tuple![Int::<4>, Int::<2>],
            int_tuple![Int::<1>, Int::<4>],
        )
    );
    assert_eq!(gcd_val, gcd_ref);

    // Mode indexing via const Tensor::get
    let mode1 = t.get::<1>();
    assert_eq!(mode1.layout, Layout::new(Int::<4>, Int::<4>));
    assert!(mode1.accessor.same_base(&t.accessor));
    let mode0 = t.get::<0>();
    assert_eq!(mode0.layout, Layout::new(Int::<4>, Int::<1>));
}

// ------------------------------------------------------------------
// TestAccessor (matching test/test_tensor.py::TestAccessor + pure no_std accessors)
// ------------------------------------------------------------------

#[test]
fn test_cell_accessor_round_trip_offset_and_equality() {
    let mut buf = [0i32; 8];
    let mut a = CellAccessor::from_slice_mut(&mut buf);
    for i in 0..8 {
        a.store(i, i * i);
    }
    let vals: Vec<i32> = (0..8).map(|i| a.load(i)).collect();
    assert_eq!(vals, vec![0, 1, 4, 9, 16, 25, 36, 49]);

    for i in 0..8 {
        a.store(i, i);
    }
    let mut v = a + 3i32;
    assert_eq!(v.load(0i32), 3);
    assert_eq!(v.load(2i32), 5);
    v.store(1i32, 99);
    assert_eq!(a.load(4i32), 99);

    // Repr stability and exact address formatting
    let v_a = a + 3i32;
    let v_b = a + 3i32;
    let v_c = a + 4i32;
    assert_eq!(format!("{v_a}"), format!("{v_b}"));
    assert_ne!(format!("{v_a}"), format!("{v_c}"));
    let expected_addr = a.address() + 3 * core::mem::size_of::<i32>();
    assert!(format!("{v_a}").contains(&format!("{expected_addr:#018x}")));

    // Value equality
    assert_eq!(v_a, v_b);
    assert_ne!(v_a, v_c);
    assert_eq!(a, a + 0i32);

    // Chained offsets flatten to the same base slice
    let v1 = a + 3i32;
    let v2 = v1 + 2i32;
    assert!(v1.same_base(&a));
    assert!(v2.same_base(&a));
    assert_eq!(v2.offset_index(), 5);
    assert_eq!(v2.address(), a.address() + 5 * core::mem::size_of::<i32>());
}

#[test]
fn test_slice_accessor_const_eval_and_algebra() {
    const DATA: [i32; 6] = [10, 20, 30, 40, 50, 60];
    const VAL: i32 = {
        let t = Tensor::from_slice(
            &DATA,
            Layout::new(
                int_tuple![Int::<2>, Int::<3>],
                int_tuple![Int::<3>, Int::<1>],
            ),
        );
        t(1i32, 2i32)
    };
    assert_eq!(VAL, 60);

    const ROW1_ELEM: i32 = {
        let t = Tensor::from_slice(
            &DATA,
            Layout::new(
                int_tuple![Int::<2>, Int::<3>],
                int_tuple![Int::<3>, Int::<1>],
            ),
        );
        t(1i32, ..)(1i32)
    };
    assert_eq!(ROW1_ELEM, 50);

    const CONST_COALESCED: i32 = coalesce(make_tensor(
        &DATA,
        Layout::new(
            int_tuple![Int::<2>, Int::<3>],
            int_tuple![Int::<1>, Int::<2>],
        ),
    ))(5i32);
    assert_eq!(CONST_COALESCED, 60);

    const MODE0_ELEM: i32 = make_tensor(&DATA, int_tuple![Int::<2>, Int::<3>]).get::<0>()(1i32);
    assert_eq!(MODE0_ELEM, 20);

    // SliceAccessor offset equality & get_ref
    let s = SliceAccessor::new(&DATA);
    assert_eq!((s + 1i32) + 2i32, s + 3i32);
    assert!((s + 3i32).same_base(&s));
    assert_eq!(*s.get_ref(4i32), 50);
}

#[test]
fn test_ptr_wraps_foreign_storage_and_offsets() {
    let mut data = ['a', 'b', 'c', 'd'];
    // SAFETY: `data` stays alive and is not accessed via `&`/`&mut` while `p` is used.
    let p = unsafe { Ptr::from_raw(data.as_mut_ptr(), data.len()) };
    let mut t = make_tensor(p, Int::<4>);
    let chars: Vec<char> = (0..4).map(|i| t(i)).collect();
    assert_eq!(chars, vec!['a', 'b', 'c', 'd']);

    t.set(2i32, 'Z');
    assert_eq!(t(2i32), 'Z');

    // Offsetting Ptr steps by size_of::<T>(), preserves base, and supports equality & Display
    let q = p + 2i32;
    assert_eq!(q.load(0i32), 'Z');
    assert_eq!(q.address(), p.address() + 2 * core::mem::size_of::<char>());
    assert_eq!(q.base(), p.base());
    assert!(q.same_base(&p));
    assert_eq!(p + 2i32, q);
    assert_ne!(p + 1i32, q);
    assert_eq!((p + 1i32) + 1i32, q);
    assert!(format!("{q}").contains(&format!("{:#018x}", q.address())));

    // Negative intermediate offset stays valid when effective index is in bounds
    let back = q + (-1i32);
    assert_eq!(back.load(0i32), 'b');

    // Verify mutation landed in backing storage after Ptr is no longer used
    assert_eq!(data, ['a', 'b', 'Z', 'd']);
}

#[test]
fn test_implicit_accessor_passes_offset_through() {
    const IA0: ImplicitAccessor<Int<0>> = ImplicitAccessor::new(Int::<0>);
    assert_eq!(IA0.load(5i32), 5);
    assert_eq!((IA0 + 3i32).load(2i32), 5);
    assert_eq!((IA0 + Int::<100>).origin(), Int::<100>);
    assert_eq!(format!("{}", IA0 + Int::<5>), "{5}");

    // With basis coordinate origin, addition is elementwise
    let ia_b = ImplicitAccessor::new(basis_tuple![10i32, 20i32]);
    let res = ia_b.load(basis_tuple![1i32, 2i32]);
    assert_eq!(as_tuple(res), int_tuple![11i32, 22i32]);
    assert_eq!(
        ia_b + basis_tuple![1i32, 2i32],
        ImplicitAccessor::new(basis_tuple![11i32, 22i32])
    );

    // Zero-padded origin equality (Int<0> == BasisCombination<Nil> == 0*e![0])
    assert_eq!(
        ImplicitAccessor::new(Int::<0>),
        ImplicitAccessor::new(noodle_cute::basis::BasisCombination::zero())
    );
    assert_eq!(
        ImplicitAccessor::new(Int::<0>),
        ImplicitAccessor::new(0i32 * noodle_cute::e![0])
    );
}

#[test]
fn test_transform_accessor_with_closure_and_const_swizzle() {
    let data = [0i32, 1, 2, 3];
    let acc = SliceAccessor::new(&data);
    let t = TransformAccessor::new(acc, |x: i32| x * x);
    assert_eq!(t.load(3i32), 9);
    assert_eq!((t + 2i32).load(1i32), 9);

    let tensor = Tensor::new(
        t,
        Layout::new(
            int_tuple![Int::<2>, Int::<2>],
            int_tuple![Int::<2>, Int::<1>],
        ),
    );
    assert_eq!(tensor(1i32, 1i32), 9);
    assert_eq!(tensor(1i32, ..)(1i32), 9);

    // Const evaluation with Swizzle functor
    const SW: Swizzle<Int<3>, Int<0>, Int<3>> = Swizzle::new(Int::<3>, Int::<0>, Int::<3>);
    const SW_ACC: TransformAccessor<ImplicitAccessor<Int<0>>, Swizzle<Int<3>, Int<0>, Int<3>>> =
        TransformAccessor::new(ImplicitAccessor::new(Int::<0>), SW);
    const SW_VAL: i32 = SW_ACC.load(2i32 + 5i32 * 8i32);
    assert_eq!(SW_VAL, SW(2i32 + 5i32 * 8i32));
}

#[test]
fn test_f2_swizzled_layout_tensor() {
    let mut buf = [0i32; 64];
    for (i, slot) in buf.iter_mut().enumerate() {
        *slot = i as i32;
    }
    let sw = Swizzle::new(Int::<3>, Int::<0>, Int::<3>);
    let layout = Layout::new(
        int_tuple![Int::<8>, Int::<8>],
        tuple![f2(Int::<1>), f2(Int::<9>)],
    );
    let t = make_tensor(&mut buf, layout);
    for i in 0..8i32 {
        for j in 0..8i32 {
            let expected = sw(i + j * 8);
            assert_eq!(t(i, j), expected);
            assert_eq!(t(i, ..)(j), expected);
            assert_eq!(t(.., j)(i), expected);
        }
    }

    // Specific regression case: (1, 1) -> 1 ^ 9 = 8 (not 1 + 9 = 10)
    assert_eq!(t(1i32, 1i32), 8);
    assert_eq!(t(1i32, ..)(1i32), 8);
    assert_eq!(t(.., 1i32)(1i32), 8);

    // Mutating through sliced F2 view writes to buf[8], leaving buf[10] untouched
    let mut row1 = t(1i32, ..);
    let col1 = t(.., 1i32);
    row1.set(1i32, 77);
    assert_eq!(t(1i32, 1i32), 77);
    assert_eq!(col1(1i32), 77);
    assert_eq!(t.accessor.load(8i32), 77);
    assert_eq!(t.accessor.load(10i32), 10);

    // Repeated F2 slices compare equal
    assert_eq!(t(1i32, ..), t(1i32, ..));
    assert_ne!(t(1i32, ..), t(2i32, ..));

    // Multi-level 3D F2 slicing and linear-shifted base + F2 tile
    let mut buf128 = [0i32; 128];
    for (i, slot) in buf128.iter_mut().enumerate() {
        *slot = i as i32;
    }
    let tile_acc = CellAccessor::from_slice_mut(&mut buf128) + 64i32;
    let tile = make_tensor(tile_acc, layout);
    assert_eq!(tile(1i32, 1i32), 64 + 8);
    assert_eq!(tile(1i32, ..)(1i32), 64 + 8);
    tile(1i32, ..).set(1i32, 999);
    assert_eq!(tile(1i32, 1i32), 999);
    assert_eq!(buf128[72], 999);
    assert_eq!(buf128[74], 74);

    // 3D F2 multi-level chained slice flattens XOR origins
    let layout3d = Layout::new(
        int_tuple![Int::<4>, Int::<4>, Int::<4>],
        tuple![f2(Int::<1>), f2(Int::<4>), f2(Int::<20>)],
    );
    let t3 = make_tensor(&buf128[..64], layout3d);
    assert_eq!(t3(1i32, 1i32, 1i32), 1 ^ 4 ^ 20);
    assert_eq!(t3(1i32, .., ..)(1i32, ..)(1i32), 1 ^ 4 ^ 20);

    // Ptr with F2 slicing
    let mut ptr_buf = [0i32; 64];
    for (i, slot) in ptr_buf.iter_mut().enumerate() {
        *slot = i as i32;
    }
    // SAFETY: `ptr_buf` stays alive and unaliased while `p` is used.
    let p = unsafe { Ptr::from_raw(ptr_buf.as_mut_ptr(), ptr_buf.len()) };
    let tp = make_tensor(p, layout);
    tp(1i32, ..).set(1i32, 555);
    assert_eq!(tp(1i32, 1i32), 555);
    assert_eq!(ptr_buf[8], 555);
    assert_eq!(ptr_buf[10], 10);

    // Const F2 slicing on SliceAccessor
    const F2_CONST_VAL: i32 = {
        const BUF: [i32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let t = make_tensor(
            &BUF,
            Layout::new(
                int_tuple![Int::<4>, Int::<4>],
                tuple![f2(Int::<1>), f2(Int::<5>)],
            ),
        );
        t(1i32, ..)(1i32)
    };
    assert_eq!(F2_CONST_VAL, 1 ^ 5);
}

#[test]
fn test_make_tensor_negative_stride_and_zero_length_bounds() {
    // (2, 2):(-1, 2) reaches offsets {-1, 0, 1, 2} -> linear_bounds = Some((-1, 2))
    let neg_layout = Layout::new(
        int_tuple![Int::<2>, Int::<2>],
        int_tuple![Int::<-1>, Int::<2>],
    );
    assert_eq!(neg_layout.linear_bounds(), Some((-1, 2)));
    assert_eq!(
        linear_bounds(
            int_tuple![Int::<2>, Int::<2>],
            int_tuple![Int::<-1>, Int::<2>]
        ),
        Some((-1, 2))
    );

    // 2 elements at offset 0: rejected at construction
    let mut buf2 = [10i32, 20];
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = make_tensor(&mut buf2, neg_layout);
        }))
        .is_err()
    );

    // 4 elements at offset 0: still rejected because min_offset = -1 < 0
    let mut buf4 = [10i32, 20, 30, 40];
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = make_tensor(&mut buf4, neg_layout);
        }))
        .is_err()
    );

    // 4 elements shifted to offset 1: covers indices [0, 3], all 4 coords valid
    let acc4 = CellAccessor::from_slice_mut(&mut buf4) + 1i32;
    let mut t4 = make_tensor(acc4, neg_layout);
    assert_eq!(t4(0i32, 0i32), 20);
    assert_eq!(t4(1i32, 0i32), 10);
    assert_eq!(t4(0i32, 1i32), 40);
    assert_eq!(t4(1i32, 1i32), 30);
    t4.set(int_tuple![1i32, 0i32], 99);
    assert_eq!(buf4[0], 99);

    // 2:-1 bound to 2-element accessor at offset 1 succeeds; at offset 0 panics
    let rev2 = Layout::new(Int::<2>, Int::<-1>);
    assert_eq!(rev2.linear_bounds(), Some((-1, 0)));
    let mut rev_buf = [100i32, 200];
    let rev_acc = CellAccessor::from_slice_mut(&mut rev_buf);
    assert!(catch_unwind(AssertUnwindSafe(|| make_tensor(rev_acc, rev2))).is_err());
    let mut t_rev = make_tensor(rev_acc + 1i32, rev2);
    assert_eq!(t_rev(0i32), 200);
    assert_eq!(t_rev(1i32), 100);
    t_rev.set(1i32, 333);
    assert_eq!(rev_buf[0], 333);

    // Zero-length tensors succeed on empty storage (both dynamic 0 and static Int<0>)
    let empty: [i32; 0] = [];
    let tz_dyn = make_tensor(&empty, 0i32);
    assert_eq!(tz_dyn.size(), 0);
    let tz_static = make_tensor(&empty, Int::<0>);
    assert_eq!(tz_static.size(), 0);
    let tz_2d = make_tensor(&empty, int_tuple![Int::<0>, Int::<4>]);
    assert_eq!(tz_2d.size(), 0);
    let tz_f2 = make_tensor(&empty, Layout::new(Int::<0>, f2(Int::<1>)));
    assert_eq!(tz_f2.size(), 0);

    // Negative dynamic extent still panics
    assert!(catch_unwind(|| make_tensor(&empty, -1i32)).is_err());

    // Nil (rank-0) has size 1 and requires 1 valid element at offset 0
    assert_eq!(Layout::new(Nil, Nil).linear_bounds(), Some((0, 0)));
    assert!(catch_unwind(|| make_tensor(&empty, Nil)).is_err());
    let one = [42i32];
    let t_nil = make_tensor(&one, Nil);
    assert_eq!(t_nil.size(), 1);
    assert_eq!(t_nil(Nil), 42);
}

#[test]
fn test_implicit_accessor_root_basis_equality() {
    let root = BasisTerm::<i32, ModeEnd>::new(5);
    assert!(basis_eq(root, 5i32));
    assert_eq!(ImplicitAccessor::new(root), ImplicitAccessor::new(5i32));
    assert_eq!(ImplicitAccessor::new(5i32), ImplicitAccessor::new(root));
    assert_eq!(ImplicitAccessor::new(root), ImplicitAccessor::new(Int::<5>));
    assert_eq!(ImplicitAccessor::new(Int::<5>), ImplicitAccessor::new(root));
    assert_ne!(ImplicitAccessor::new(root), ImplicitAccessor::new(6i32));
    assert_ne!(ImplicitAccessor::new(6i32), ImplicitAccessor::new(root));
    assert_eq!(ImplicitAccessor::new(Nil), ImplicitAccessor::new(Nil));
    assert_eq!(
        ImplicitAccessor::new(f2(Int::<5>)),
        ImplicitAccessor::new(5i32)
    );
}

#[test]
fn test_algebra_copy_compatibility_and_coalesce_with_profile() {
    fn dup_comp<A: Composition<B>, B>(a: A, b: B) -> (A::Output, A::Output) {
        let x = composition(a, b);
        (x, x)
    }
    fn dup_ldiv<A: LogicalDivide<B>, B>(a: A, b: B) -> (A::Output, A::Output) {
        let x = logical_divide(a, b);
        (x, x)
    }
    fn dup_zdiv<A: ZippedDivide<B>, B>(a: A, b: B) -> (A::Output, A::Output) {
        let x = zipped_divide(a, b);
        (x, x)
    }
    fn dup_ldiv_at<P, A: LogicalDivideAt<B, P>, B>(a: A, b: B) -> (A::Output, A::Output) {
        let x = logical_divide_at::<P, _, _>(a, b);
        (x, x)
    }
    fn dup_zdiv_at<P, A: ZippedDivideAt<B, P>, B>(a: A, b: B) -> (A::Output, A::Output) {
        let x = zipped_divide_at::<P, _, _>(a, b);
        (x, x)
    }

    let l = Layout::new(
        int_tuple![Int::<4>, Int::<4>],
        int_tuple![Int::<1>, Int::<4>],
    );
    let (c1, c2) = dup_comp(l, Layout::new(Int::<4>, Int::<2>));
    assert_eq!(c1, c2);
    let (d1, d2) = dup_ldiv(l, Int::<2>);
    assert_eq!(d1, d2);
    let (z1, z2) = dup_zdiv(l, int_tuple![Int::<2>, Int::<2>]);
    assert_eq!(z1, z2);
    let (da1, da2) = dup_ldiv_at::<mode!(1), _, _>(l, Int::<2>);
    assert_eq!(da1, da2);
    let (za1, za2) = dup_zdiv_at::<mode!(0), _, _>(l, Int::<2>);
    assert_eq!(za1, za2);

    // Profile-guided standard coalesce drops trailing unit mode per leaf subtree,
    // unlike coalesce_z which preserves it.
    type PairProfile = TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>;
    let buf = [0i32; 12];
    let t = make_tensor(
        &buf,
        Layout::new(
            int_tuple![(Int::<2>, Int::<1>), (Int::<3>, Int::<2>)],
            int_tuple![(Int::<1>, Int::<0>), (Int::<2>, Int::<6>)],
        ),
    );
    let tc_prof = coalesce_with_profile::<PairProfile, _>(t);
    assert_eq!(
        tc_prof.layout,
        Layout::new(
            int_tuple![Int::<2>, Int::<6>],
            int_tuple![Int::<1>, Int::<2>],
        )
    );
    assert_eq!(t.coalesce_with_profile::<PairProfile>(), tc_prof);

    let tz_prof = coalesce_z::<PairProfile, _>(t);
    assert_eq!(
        tz_prof.layout,
        Layout::new(
            int_tuple![(Int::<2>, Int::<1>), Int::<6>],
            int_tuple![(Int::<1>, Int::<0>), Int::<2>],
        )
    );
}

// ------------------------------------------------------------------
// TestIdentityTensor (matching test/test_tensor.py::TestIdentityTensor)
// ------------------------------------------------------------------

#[test]
fn test_identity_tensor_returns_coordinate() {
    let id = identity_tensor(int_tuple![Int::<3>, Int::<4>]);
    for i in 0..3i32 {
        for j in 0..4i32 {
            assert_eq!(as_tuple(id(i, j)), int_tuple![i, j]);
        }
    }

    // Hierarchical shape
    let id_h = identity_tensor(int_tuple![Int::<2>, (Int::<3>, Int::<4>)]);
    for i in 0..2i32 {
        for j in 0..3i32 {
            for k in 0..4i32 {
                assert_eq!(as_tuple(id_h(i, int_tuple![j, k])), int_tuple![i, (j, k)]);
            }
        }
    }

    // Const evaluation of identity_tensor
    const COORD: noodle_cute::tuple::Til<Int<1>, noodle_cute::tuple::Til<Int<2>, Nil>> = {
        let id = identity_tensor(int_tuple![Int::<3>, Int::<4>]);
        as_tuple(id(Int::<1>, Int::<2>))
    };
    assert_eq!(COORD, int_tuple![Int::<1>, Int::<2>]);
}

#[test]
fn test_identity_tensor_supports_slicing() {
    let id = identity_tensor(int_tuple![Int::<3>, Int::<4>]);
    let row = id(1i32, ..);
    for j in 0..4i32 {
        assert_eq!(as_tuple(row(j)), int_tuple![1i32, j]);
    }

    let col = id(.., 2i32);
    for i in 0..3i32 {
        assert_eq!(as_tuple(col(i)), int_tuple![i, 2i32]);
    }
}

// ------------------------------------------------------------------
// TestLocalTile (matching test/test_tensor.py::TestLocalTile)
// ------------------------------------------------------------------

#[test]
fn test_local_tile_matches_parent_and_is_a_view() {
    let mut buf = [0i32; 96];
    let mut t = make_tensor(
        &mut buf,
        Layout::new(
            int_tuple![Int::<8>, Int::<12>],
            int_tuple![Int::<12>, Int::<1>],
        ),
    );
    for i in 0..t.size() {
        t.set(i, i);
    }

    for m in 0..2i32 {
        for n in 0..2i32 {
            let tile = local_tile(t, int_tuple![Int::<4>, Int::<6>], int_tuple![m, n]);
            assert_eq!(tile.shape(), int_tuple![(Int::<4>, Int::<6>)]);
            for i in 0..4i32 {
                for j in 0..6i32 {
                    assert_eq!(tile(tuple!((i, j))), t(m * 4 + i, n * 6 + j));
                }
            }
        }
    }

    // View semantics: writing through a tile lands in the parent
    let mut fbuf = [0.0f64; 96];
    let mut tf = make_tensor(&mut fbuf, int_tuple![Int::<8>, Int::<12>]);
    local_tile(tf, int_tuple![Int::<4>, Int::<6>], int_tuple![1i32, 1i32])
        .set(tuple!((3i32, 5i32)), 42.0);
    assert_eq!(tf(7i32, 11i32), 42.0);
    tf.set(int_tuple![0i32, 0i32], 1.5);
    assert_eq!(tf(0i32, 0i32), 1.5);
}

#[test]
fn test_local_tile_none_keeps_grid_modes() {
    let id = identity_tensor(int_tuple![Int::<8>, Int::<12>]);

    let row = local_tile(id, int_tuple![Int::<4>, Int::<6>], tuple!(1i32, ..));
    assert_eq!(row.shape(), int_tuple![(Int::<4>, Int::<6>), Int::<2>]);
    let row_origins: Vec<_> = (0..2i32)
        .map(|n| as_tuple(row(int_tuple![0i32, 0i32], n)))
        .collect();
    assert_eq!(
        row_origins,
        vec![int_tuple![4i32, 0i32], int_tuple![4i32, 6i32]]
    );

    let col = local_tile(id, int_tuple![Int::<4>, Int::<6>], tuple!(.., 1i32));
    assert_eq!(col.shape(), int_tuple![(Int::<4>, Int::<6>), Int::<2>]);
    let col_origins: Vec<_> = (0..2i32)
        .map(|m| as_tuple(col(int_tuple![0i32, 0i32], m)))
        .collect();
    assert_eq!(
        col_origins,
        vec![int_tuple![0i32, 6i32], int_tuple![4i32, 6i32]]
    );

    // pycute/tensor.py and docs/05_tensor.md: grid coordinates slice the zipped division.
    let tiler = int_tuple![Int::<4>, Int::<6>];
    let divided = zipped_divide(id, tiler);
    let block = int_tuple![1i32, 1i32];
    let tile = local_tile(id, tiler, block);
    assert_eq!(tile, divided(.., block));
    assert_eq!(tile, local_tile(id, tiler, 3i32));
    assert!(compatible(&tiler, &tile.shape().get::<0>()));
    assert_eq!(
        as_tuple(tile(tuple!((3i32, 5i32)))),
        int_tuple![7i32, 11i32]
    );
    assert_eq!(local_tile(id, tiler, ..), divided);
}

#[test]
fn test_local_tile_trailing_modes_and_integer_coord() {
    let id = identity_tensor(int_tuple![Int::<8>, Int::<12>, Int::<3>]);

    let tile = local_tile(id, int_tuple![Int::<4>, Int::<6>], tuple!(1i32, 1i32, ..));
    assert_eq!(tile.shape(), int_tuple![(Int::<4>, Int::<6>), Int::<3>]);
    for l in 0..3i32 {
        assert_eq!(
            as_tuple(tile(int_tuple![3i32, 5i32], l)),
            int_tuple![7i32, 11i32, l]
        );
    }
    assert_eq!(
        local_tile(
            id,
            int_tuple![Int::<4>, Int::<6>],
            int_tuple![1i32, 1i32, 2i32]
        ),
        tile(.., 2i32)
    );

    // Integer coord indexes the whole grid colexicographically, trailing modes included
    let grid = zipped_divide(id, int_tuple![Int::<4>, Int::<6>])
        .shape()
        .get::<1>();
    for t_idx in 0..noodle_cute::shape::size(&grid) {
        assert_eq!(
            local_tile(id, int_tuple![Int::<4>, Int::<6>], t_idx),
            local_tile(id, int_tuple![Int::<4>, Int::<6>], idx2crd(t_idx, &grid))
        );
    }
}

#[test]
fn test_local_tile_gemm_operands_share_tiler() {
    let a = identity_tensor(int_tuple![Int::<128>, Int::<16>]);
    let b = identity_tensor(int_tuple![Int::<256>, Int::<16>]);
    let c = identity_tensor(int_tuple![Int::<128>, Int::<256>]);
    let tiler = int_tuple![Int::<32>, Int::<64>, Int::<4>];
    let coord = tuple!(1i32, 2i32, ..);

    let cta_a = local_tile(
        a,
        select::<mode!(0, 2), _>(&tiler),
        select::<mode!(0, 2), _>(&coord),
    );
    let cta_b = local_tile(
        b,
        select::<mode!(1, 2), _>(&tiler),
        select::<mode!(1, 2), _>(&coord),
    );
    let cta_c = local_tile(
        c,
        select::<mode!(0, 1), _>(&tiler),
        select::<mode!(0, 1), _>(&coord),
    );

    assert_eq!(cta_a.shape(), int_tuple![(Int::<32>, Int::<4>), Int::<4>]);
    assert_eq!(cta_b.shape(), int_tuple![(Int::<64>, Int::<4>), Int::<4>]);
    assert_eq!(cta_c.shape(), int_tuple![(Int::<32>, Int::<64>)]);

    for k in 0..4i32 {
        assert_eq!(
            as_tuple(cta_a(int_tuple![0i32, 0i32], k)),
            int_tuple![32i32, 4 * k]
        );
        assert_eq!(
            as_tuple(cta_b(int_tuple![0i32, 0i32], k)),
            int_tuple![128i32, 4 * k]
        );
    }
    assert_eq!(
        as_tuple(cta_c(tuple!((0i32, 0i32)))),
        int_tuple![32i32, 128i32]
    );
}

#[test]
fn test_local_tile_general_tilers_and_residue_predication() {
    let id = identity_tensor(int_tuple![Int::<8>, Int::<12>]);

    // Tiler of Layouts selects strided tiles (every other row, from row 1)
    let tile_strided = local_tile(
        id,
        tuple!(Layout::new(Int::<4>, Int::<2>), Int::<6>),
        int_tuple![1i32, 1i32],
    );
    assert_eq!(tile_strided.shape(), int_tuple![(Int::<4>, Int::<6>)]);
    let col0: Vec<_> = (0..4i32)
        .map(|i| as_tuple(tile_strided(tuple!((i, 0i32)))))
        .collect();
    assert_eq!(
        col0,
        vec![
            int_tuple![1i32, 6i32],
            int_tuple![3i32, 6i32],
            int_tuple![5i32, 6i32],
            int_tuple![7i32, 6i32],
        ]
    );

    // Scalar integer tiler tiles the tensor as 1-D
    let tile_1d = local_tile(id, Int::<16>, 3i32);
    for i in 0..16i32 {
        assert!(basis_eq(tile_1d(i), id(48 + i)));
    }

    // Residue tiles reaching past the tensor for predication
    let id10 = identity_tensor(int_tuple![Int::<10>, Int::<10>]);
    let corner = local_tile(id10, int_tuple![Int::<4>, Int::<4>], int_tuple![2i32, 2i32]);
    assert_eq!(
        as_tuple(corner(tuple!((0i32, 0i32)))),
        int_tuple![8i32, 8i32]
    );
    let bounds: Vec<bool> = (0..4i32)
        .map(|i| in_bounds(corner(tuple!((i, i))), &int_tuple![Int::<10>, Int::<10>]))
        .collect();
    assert_eq!(bounds, vec![true, true, false, false]);
}

#[test]
fn test_local_tile_dynamic_extents_and_coord() {
    let m: int = 128;
    let n: int = 256;
    let bx: int = 2;
    let by: int = 3;

    let tile = local_tile(
        identity_tensor(int_tuple![m, n]),
        int_tuple![Int::<32>, Int::<64>],
        int_tuple![bx, by],
    );
    assert_eq!(tile.shape(), int_tuple![(Int::<32>, Int::<64>)]);
    assert_eq!(
        as_tuple(tile(tuple!((0i32, 0i32)))),
        int_tuple![32 * bx, 64 * by]
    );
    assert_eq!(
        as_tuple(tile(tuple!((31i32, 63i32)))),
        int_tuple![32 * bx + 31, 64 * by + 63]
    );
}

// ------------------------------------------------------------------
// Runtime bounds & overflow panic tests
// ------------------------------------------------------------------

#[test]
fn test_accessor_bounds_and_overflow_panics() {
    let data = [1i32, 2, 3, 4];
    let s = SliceAccessor::new(&data);
    assert!(catch_unwind(|| s.load(-1i32)).is_err());
    assert!(catch_unwind(|| s.load(4i32)).is_err());
    assert!(catch_unwind(|| (s + int::MAX).offset(1i32)).is_err());

    let mut mut_data = [1i32, 2, 3, 4];
    let c = CellAccessor::from_slice_mut(&mut mut_data);
    assert!(catch_unwind(AssertUnwindSafe(|| c.load(4i32))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| c.load(-1i32))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| make_tensor(c + (-1i32), Int::<2>))).is_err());
}

#[test]
fn hierarchical_morton_tensor_coordinates_and_slices() {
    // docs/GPUMODE_057_CuTe.pdf, p. 28: recursive blocked product builds Morton order.
    let morton1 = Layout::from_shape(int_tuple![Int::<2>, Int::<2>]);
    let morton2 = blocked_product(morton1, morton1);
    let morton3 = blocked_product(morton1, morton2);
    let a = Tensor::new(ImplicitAccessor::new(Int::<0>), morton3);

    assert_eq!(a(37i32), 49);
    assert_eq!(a(5i32, 4i32), 49);
    assert_eq!(a(int_tuple![1i32, 2i32], int_tuple![0i32, 2i32]), 49);
    assert_eq!(
        a(int_tuple![(1i32, (0i32, 1i32)), (0i32, (0i32, 1i32))]),
        49
    );

    let column = a(.., 2i32);
    assert_eq!(
        (0..column.size()).map(|i| column(i)).collect::<Vec<_>>(),
        [8, 9, 12, 13, 24, 25, 28, 29],
    );
    let tile = a(tuple!((.., 1i32), (.., 2i32)));
    assert_eq!(tile.shape(), int_tuple![Int::<2>, Int::<2>]);
    assert_eq!([tile(0i32, 0i32), tile(0i32, 1i32)], [36, 38]);
    assert_eq!([tile(1i32, 0i32), tile(1i32, 1i32)], [37, 39]);
}

#[test]
fn thread_value_partition_composes_then_slices() {
    // docs/GPUMODE_057_CuTe.pdf, pp. 50–55, and docs/GPUMODE_114_PyCuTe.pdf, p. 66.
    let data = Tensor::new(
        ImplicitAccessor::new(Int::<0>),
        Layout::from_shape(int_tuple![Int::<4>, Int::<6>]),
    );
    let thread_value = Layout::new(
        int_tuple![(Int::<2>, Int::<2>), (Int::<2>, Int::<3>)],
        int_tuple![(Int::<2>, Int::<12>), (Int::<1>, Int::<4>)],
    );
    let partitioned = composition(data, thread_value);
    let expected = [
        [0, 1, 4, 5, 8, 9],
        [2, 3, 6, 7, 10, 11],
        [12, 13, 16, 17, 20, 21],
        [14, 15, 18, 19, 22, 23],
    ];
    for (tid, offsets) in (0..4).zip(expected) {
        let values = partitioned(tid, ..);
        assert_eq!(values.size(), 6);
        assert_eq!(
            (0..values.size())
                .map(|vid| values(vid))
                .collect::<Vec<_>>(),
            offsets,
        );
    }
}
