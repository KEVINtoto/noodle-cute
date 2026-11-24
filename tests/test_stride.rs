#![feature(const_trait_impl, const_cmp)]

use core::any::TypeId;

use noodle_cute::basis::{
    Basis, BasisCombination, BasisPath, BasisTerm, Mode, ModeEnd, ScaledBasis, from_tuple,
};
use noodle_cute::integer::{Int, int};
use noodle_cute::shape::Shape;
use noodle_cute::stride::{
    Accumulate, CodomainProfile, InnerProduct, PrefixProduct, ScaleBasisFunctor, ScaleStride,
    Stride, StrideMul, StrideScalar, coshape, inner_product, prefix_product, prefix_product_with,
    same_path,
};
use noodle_cute::tuple::{
    Congruent, Item, LeafProfile, Nil, NilProfile, Til, TransformLeafFunctor, TupleContainer,
    TupleProfile,
};
use noodle_cute::{basis_tuple, e, int_tuple, mode, tuple};

fn unit_basis<const M: int>() -> ScaledBasis<Int<1>, Mode<M>>
where
    Mode<M>: BasisPath,
{
    BasisTerm::new(Int::<1>)
}

#[test]
fn stride_scalar_and_same_path() {
    assert_eq!(StrideScalar::value(5i32), 5i32);
    assert_eq!(StrideScalar::value(Int::<7>), Int::<7>);
    assert_eq!(
        TypeId::of::<<i32 as StrideScalar>::Path>(),
        TypeId::of::<ModeEnd>()
    );
    assert_eq!(
        TypeId::of::<<Int<7> as StrideScalar>::Path>(),
        TypeId::of::<ModeEnd>()
    );

    let sb = ScaledBasis::<Int<3>, Mode<1, Mode<0>>>::new(Int::<3>);
    assert_eq!(sb.value(), Int::<3>);
    assert_eq!(
        TypeId::of::<<ScaledBasis<Int<3>, Mode<1, Mode<0>>> as StrideScalar>::Path>(),
        TypeId::of::<Mode<1, Mode<0>>>()
    );

    const {
        assert!(same_path::<ModeEnd, ModeEnd>());
        assert!(!same_path::<ModeEnd, Mode<0>>());
        assert!(same_path::<Mode<1, Mode<0>>, Mode<1, Mode<0>>>());
        assert!(!same_path::<Mode<1, Mode<0>>, Mode<1, Mode<1>>>());
    }
}

#[test]
fn stride_trait_accepts_scalars_scaled_bases_and_combinations() {
    fn require_stride<S: Shape, D: Stride<S>>(_: &S, _: &D) {}

    let b0 = unit_basis::<0>();
    let b1 = unit_basis::<1>();
    let comb = b0 + b1;

    require_stride(&Nil, &Nil);
    require_stride(&4i32, &1i32);
    require_stride(&4i32, &b0);
    require_stride(&4i32, &comb);
    require_stride(&Til(Int::<2>, Til(3i32, Nil)), &Til(b0, Til(comb, Nil)));
}

#[test]
fn stride_mul_and_accumulate_handle_scalars_and_basis_leaves() {
    // Scalar multiplication
    assert_eq!(3i32.stride_mul(4i32), 12i32);
    assert_eq!(Int::<3>.stride_mul(Int::<4>), Int::<12>);

    // ScaledBasis scaling, including static zero
    let sb_dyn = ScaledBasis::<i32, Mode<2>>::new(-3);
    assert_eq!(
        sb_dyn.stride_mul(4i32),
        ScaledBasis::<i32, Mode<2>>::new(-12)
    );
    assert_eq!(
        unit_basis::<0>().stride_mul(Int::<0>),
        BasisCombination::<Nil>::zero()
    );

    // BasisCombination scaling
    let comb = ScaledBasis::<i32, Mode<0>>::new(2) + ScaledBasis::<i32, Mode<1>>::new(3);
    let scaled = comb.stride_mul(4i32);
    assert_eq!(
        scaled.terms(),
        Til(BasisTerm::new(8i32), Til(BasisTerm::new(12i32), Nil))
    );
    let zero_scaled = comb.stride_mul(Int::<0>);
    assert_eq!(zero_scaled.terms(), Nil);

    // Accumulate identities
    assert_eq!(Int::<0>.accumulate(unit_basis::<0>()), unit_basis::<0>());
    assert_eq!(unit_basis::<0>().accumulate(Nil), unit_basis::<0>());
    assert_eq!(unit_basis::<0>().accumulate(Int::<0>), unit_basis::<0>());
    assert_eq!(Int::<0>.accumulate(Nil), Int::<0>);
    assert_eq!(Nil.accumulate(Int::<5>), Int::<5>);
    assert_eq!(3i32.accumulate(4i32), 7i32);
}

#[test]
fn inner_product_and_prefix_product_work_with_const_eval() {
    // Integer inner product
    let crd = Til(1i32, Til(Til(2i32, Til(3i32, Nil)), Nil));
    let strd = Til(1i32, Til(Til(4i32, Til(12i32, Nil)), Nil));
    assert_eq!(inner_product(crd, strd), 45i32);

    // Signed coordinate inner product
    assert_eq!(inner_product(Int::<-2>, Int::<5>), Int::<-10>);

    // Single ScaledBasis + static zero mode in inner_product
    let b1 = unit_basis::<1>();
    let sparse_crd = Til(Int::<0>, Til(3i32, Nil));
    let sparse_strd = Til(unit_basis::<0>(), Til(b1, Nil));
    assert_eq!(
        inner_product(sparse_crd, sparse_strd),
        ScaledBasis::<i32, Mode<1>>::new(3)
    );

    // Const eval inner_product
    const C_CRD: TupleContainer<Int<2>, TupleContainer<Int<3>, Nil>> =
        Til(Int::<2>, Til(Int::<3>, Nil));
    const C_STRD: TupleContainer<Int<1>, TupleContainer<Int<4>, Nil>> =
        Til(Int::<1>, Til(Int::<4>, Nil));
    const IP: Int<14> = inner_product(C_CRD, C_STRD);
    assert_eq!(IP, Int::<14>);

    // Prefix product with scalar, basis, zero-extent, and tuple seeds
    let shape = Til(Int::<3>, Til(Til(Int::<2>, Til(Int::<4>, Nil)), Nil));
    assert_eq!(
        prefix_product(shape),
        Til(Int::<1>, Til(Til(Int::<3>, Til(Int::<6>, Nil)), Nil))
    );

    let b0 = unit_basis::<0>();
    assert_eq!(
        prefix_product_with(Til(Int::<3>, Til(Int::<2>, Nil)), b0),
        Til(b0, Til(ScaledBasis::<Int<3>, Mode<0>>::new(Int::<3>), Nil))
    );

    // Static zero extent followed by another mode with ScaledBasis seed
    assert_eq!(
        prefix_product_with(Til(Int::<0>, Til(Int::<4>, Nil)), b0),
        Til(b0, Til(BasisCombination::<Nil>::zero(), Nil))
    );

    // Tuple seed distribution
    let two_modes = Til(
        Til(Int::<2>, Til(Int::<3>, Nil)),
        Til(Til(Int::<4>, Til(Int::<5>, Nil)), Nil),
    );
    let seeds = Til(unit_basis::<0>(), Til(unit_basis::<1>(), Nil));
    assert_eq!(
        prefix_product_with(two_modes, seeds),
        Til(
            Til(
                unit_basis::<0>(),
                Til(ScaledBasis::<Int<2>, Mode<0>>::new(Int::<2>), Nil)
            ),
            Til(
                Til(
                    unit_basis::<1>(),
                    Til(ScaledBasis::<Int<4>, Mode<1>>::new(Int::<4>), Nil)
                ),
                Nil
            ),
        )
    );

    const P_OUT: TupleContainer<Int<1>, TupleContainer<Int<3>, Nil>> =
        prefix_product(Til(Int::<3>, Til(Int::<4>, Nil)));
    assert_eq!(P_OUT, Til(Int::<1>, Til(Int::<3>, Nil)));
}

#[test]
fn scale_stride_works_for_scalar_and_basis_strides() {
    let int_strd = Til(Int::<2>, Til(Int::<3>, Nil));
    assert_eq!(
        int_strd.scale_stride(Int::<4>),
        Til(Int::<8>, Til(Int::<12>, Nil))
    );

    let b0 = unit_basis::<0>();
    assert_eq!(
        int_strd.scale_stride(b0),
        Til(
            ScaledBasis::<Int<2>, Mode<0>>::new(Int::<2>),
            Til(ScaledBasis::<Int<3>, Mode<0>>::new(Int::<3>), Nil),
        )
    );

    let basis_strd = Til(
        ScaledBasis::<Int<2>, Mode<0>>::new(Int::<2>),
        Til(ScaledBasis::<Int<3>, Mode<1>>::new(Int::<3>), Nil),
    );
    assert_eq!(
        basis_strd.scale_stride(Int::<5>),
        Til(
            ScaledBasis::<Int<10>, Mode<0>>::new(Int::<10>),
            Til(ScaledBasis::<Int<15>, Mode<1>>::new(Int::<15>), Nil),
        )
    );

    const SCALED: TupleContainer<Int<6>, TupleContainer<int, Nil>> =
        Til(Int::<2>, Til(3i32, Nil)).scale_stride(Int::<3>);
    assert_eq!(SCALED, Til(Int::<6>, Til(9i32, Nil)));
}

#[test]
fn codomain_profile_reflects_scalar_basis_and_combination_structures() {
    fn profile_of<T: CodomainProfile>() -> TypeId {
        TypeId::of::<T::Profile>()
    }

    assert_eq!(profile_of::<Nil>(), TypeId::of::<LeafProfile>());
    assert_eq!(profile_of::<i32>(), TypeId::of::<LeafProfile>());
    assert_eq!(profile_of::<Int<4>>(), TypeId::of::<LeafProfile>());

    assert_eq!(
        profile_of::<ScaledBasis<Int<1>, Mode<0>>>(),
        TypeId::of::<TupleProfile<LeafProfile, NilProfile>>()
    );
    assert_eq!(
        profile_of::<ScaledBasis<Int<1>, Mode<2>>>(),
        TypeId::of::<
            TupleProfile<
                LeafProfile,
                TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>,
            >,
        >()
    );
    assert_eq!(
        profile_of::<ScaledBasis<Int<1>, Mode<1, Mode<0>>>>(),
        TypeId::of::<
            TupleProfile<
                LeafProfile,
                TupleProfile<TupleProfile<LeafProfile, NilProfile>, NilProfile>,
            >,
        >()
    );

    type Comb = BasisCombination<
        TupleContainer<
            BasisTerm<i32, Mode<1, Mode<0>>>,
            TupleContainer<
                BasisTerm<i32, Mode<1, Mode<1>>>,
                TupleContainer<BasisTerm<i32, Mode<3>>, Nil>,
            >,
        >,
    >;
    assert_eq!(
        profile_of::<Comb>(),
        TypeId::of::<
            TupleProfile<
                LeafProfile,
                TupleProfile<
                    TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>,
                    TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>,
                >,
            >,
        >()
    );

    type Tup = TupleContainer<Int<1>, TupleContainer<ScaledBasis<Int<1>, Mode<0>>, Nil>>;
    assert_eq!(
        profile_of::<Tup>(),
        TypeId::of::<TupleProfile<LeafProfile, NilProfile>>()
    );
}

#[test]
fn test_stride_scalar() {
    // Dynamic integer
    assert_eq!(5i32.value(), 5i32);
    assert_eq!(
        TypeId::of::<<i32 as StrideScalar>::Path>(),
        TypeId::of::<ModeEnd>()
    );

    // Static signed integer
    assert_eq!(Int::<7>.value(), Int::<7>);
    assert_eq!(
        TypeId::of::<<Int<7> as StrideScalar>::Path>(),
        TypeId::of::<ModeEnd>()
    );

    // Basis<mode![0]>
    assert_eq!(e![0].value(), Int::<1>);
    assert_eq!(
        TypeId::of::<<Basis<mode![0]> as StrideScalar>::Path>(),
        TypeId::of::<mode![0]>()
    );
    assert_eq!(
        TypeId::of::<<Basis<mode![2]> as StrideScalar>::Path>(),
        TypeId::of::<mode![2]>()
    );

    // ScaledBasis<V, P>
    let sb0 = ScaledBasis::<Int<5>, mode![0]>::new(Int::<5>);
    assert_eq!(sb0.value(), Int::<5>);
    assert_eq!(
        TypeId::of::<<ScaledBasis<Int<5>, mode![0]> as StrideScalar>::Path>(),
        TypeId::of::<mode![0]>()
    );

    let sb1 = ScaledBasis::<i32, mode![1, 0]>::new(12i32);
    assert_eq!(sb1.value(), 12i32);
    assert_eq!(
        TypeId::of::<<ScaledBasis<i32, mode![1, 0]> as StrideScalar>::Path>(),
        TypeId::of::<mode![1, 0]>()
    );
}

#[test]
fn test_stride_mul() {
    // 1. Integer * Integer
    assert_eq!(Int::<3>.stride_mul(Int::<4>), Int::<12>);
    assert_eq!(3i32.stride_mul(Int::<4>), 12i32);
    assert_eq!(Int::<3>.stride_mul(4i32), 12i32);
    assert_eq!(3i32.stride_mul(4i32), 12i32);

    // 2. Basis * Integer -> ScaledBasis
    let sb_stat = e![0].stride_mul(Int::<5>);
    assert_eq!(sb_stat, ScaledBasis::<Int<5>, mode![0]>::new(Int::<5>));

    let sb_dyn = e![1].stride_mul(8i32);
    assert_eq!(sb_dyn, ScaledBasis::<i32, mode![1]>::new(8i32));

    // 3. ScaledBasis<V, P> * Integer -> ScaledBasis
    let sb_scaled = sb_stat.stride_mul(Int::<2>);
    assert_eq!(sb_scaled, ScaledBasis::<Int<10>, mode![0]>::new(Int::<10>));

    let sb_dyn_scaled = sb_dyn.stride_mul(3i32);
    assert_eq!(sb_dyn_scaled, ScaledBasis::<i32, mode![1]>::new(24i32));
}

#[test]
fn test_accumulate() {
    // 1. Zero accumulator identity
    assert_eq!(Int::<0>.accumulate(e![0]), e![0]);
    assert_eq!(Int::<0>.accumulate(e![1]), e![1]);
    assert_eq!(
        Int::<0>.accumulate(ScaledBasis::<Int<3>, mode![0]>::new(Int::<3>)),
        ScaledBasis::<Int<3>, mode![0]>::new(Int::<3>)
    );

    // 2. Integer addition
    assert_eq!(Int::<3>.accumulate(Int::<4>), Int::<7>);
    assert_eq!(3i32.accumulate(4i32), 7i32);

    // 3. Matching mode ScaledBasis addition
    let a0 = ScaledBasis::<Int<2>, mode![0]>::new(Int::<2>);
    let b0 = ScaledBasis::<Int<3>, mode![0]>::new(Int::<3>);
    assert_eq!(
        a0.accumulate(b0),
        ScaledBasis::<Int<5>, mode![0]>::new(Int::<5>)
    );

    // 4. Distinct modes 0 and 1 form 2D BasisCombination
    let a1 = ScaledBasis::<Int<4>, mode![1]>::new(Int::<4>);
    let pair = a0.accumulate(a1);
    assert_eq!(pair, basis_tuple![Int::<2>, Int::<4>]);

    // Transposed order: 1 then 0
    let pair_trans = a1.accumulate(a0);
    assert_eq!(pair_trans, basis_tuple![Int::<2>, Int::<4>]);

    // 5. Combination + mode 2 forms 3D BasisCombination
    let a2 = ScaledBasis::<Int<6>, mode![2]>::new(Int::<6>);
    let triple = pair.accumulate(a2);
    assert_eq!(triple, basis_tuple![Int::<2>, Int::<4>, Int::<6>]);
}

#[test]
fn test_accumulate_arbitrary_basis_modes() {
    let a = ScaledBasis::<Int<2>, mode![5]>::new(Int);
    let b = ScaledBasis::<Int<3>, mode![1]>::new(Int);
    let expected = basis_tuple![Int::<0>, Int::<3>, Int::<0>, Int::<0>, Int::<0>, Int::<2>];
    assert_eq!(a.accumulate(b), expected);
    assert_eq!(b.accumulate(a), expected);
    assert_eq!(a.accumulate(a), ScaledBasis::<Int<4>, mode![5]>::new(Int));
    assert_eq!(
        expected.accumulate(ScaledBasis::<Int<7>, mode![1]>::new(Int)),
        basis_tuple![Int::<0>, Int::<10>, Int::<0>, Int::<0>, Int::<0>, Int::<2>]
    );
    assert_eq!(
        basis_tuple![Int::<1>].accumulate(a),
        basis_tuple![Int::<1>, Int::<0>, Int::<0>, Int::<0>, Int::<0>, Int::<2>]
    );
    assert_eq!(
        Int::<0>.accumulate(basis_tuple![1i32, 2i32]),
        basis_tuple![1i32, 2i32]
    );
}

#[test]
fn test_inner_product_high_out_of_order_and_repeated_basis_modes() {
    let coord = int_tuple![2i32, 3i32, 5i32, 7i32];
    let identity = tuple![e![0], e![1], e![2], e![3]];
    assert_eq!(inner_product(coord, identity), from_tuple(coord));

    let sparse = tuple![e![5], e![1], e![5], e![3]];
    assert_eq!(
        inner_product(coord, sparse),
        basis_tuple![Int::<0>, 3i32, Int::<0>, 7i32, Int::<0>, 7i32]
    );

    type Expected = BasisCombination<
        TupleContainer<BasisTerm<Int<2>, Mode<0>>, TupleContainer<BasisTerm<Int<3>, Mode<3>>, Nil>>,
    >;
    const RESULT: Expected = inner_product(int_tuple![Int::<3>, Int::<2>], tuple![e![3], e![0]]);
    assert_eq!(RESULT, basis_tuple![Int::<2>, Int::<0>, Int::<0>, Int::<3>]);

    // Nested basis inner product: (x, (y, z)) under (E(1, 0), (E(0), E(1, 1)))
    let nested_crd = int_tuple![3i32, (2i32, 5i32)];
    let nested_strd = tuple![e![0].at::<1>(), (e![0], e![1].at::<1>())];
    assert_eq!(
        inner_product(nested_crd, nested_strd),
        basis_tuple![2i32, (3i32, 5i32)]
    );
}

#[test]
fn test_arithmetic_tuple_strides_and_mixed_basis_inner_products() {
    let d = basis_tuple![2i32, 3i32];
    assert_eq!(inner_product(3i32, d), basis_tuple![6i32, 9i32]);
    assert_eq!(d.scale_stride(2i32), basis_tuple![4i32, 6i32]);

    let coord = int_tuple![2i32, 3i32, 4i32];
    assert_eq!(
        inner_product(coord, tuple![d, e![3], e![0]]),
        basis_tuple![8i32, 6i32, Int::<0>, 3i32]
    );
    assert_eq!(
        inner_product(coord, tuple![e![3], d, e![0]]),
        basis_tuple![10i32, 9i32, Int::<0>, 2i32]
    );
    assert_eq!(
        inner_product(coord, tuple![e![3], e![0], d]),
        basis_tuple![11i32, 12i32, Int::<0>, 2i32]
    );
    assert_eq!(
        inner_product(int_tuple![2i32, 3i32], tuple![d, d]),
        basis_tuple![10i32, 15i32]
    );

    type Expected = BasisCombination<
        TupleContainer<BasisTerm<Int<6>, Mode<0>>, TupleContainer<BasisTerm<Int<9>, Mode<1>>, Nil>>,
    >;
    const RESULT: Expected = inner_product(Int::<3>, basis_tuple![Int::<2>, Int::<3>]);
    assert_eq!(RESULT, basis_tuple![Int::<6>, Int::<9>]);

    type D = BasisCombination<
        TupleContainer<BasisTerm<i32, Mode<0>>, TupleContainer<BasisTerm<i32, Mode<1>>, Nil>>,
    >;
    assert_eq!(
        TypeId::of::<<D as CodomainProfile>::Profile>(),
        TypeId::of::<TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>>()
    );
}

#[test]
fn test_inner_product() {
    // 1. Nil
    assert_eq!(Nil.inner_product(Nil, Int::<0>), Int::<0>);
    assert_eq!(inner_product(Nil, Nil), Int::<0>);

    // 2. Integer
    assert_eq!(2i32.inner_product(3i32, Int::<0>), 6i32);
    assert_eq!(inner_product(2i32, 3i32), 6i32);
    assert_eq!(inner_product(Int::<3>, Int::<4>), Int::<12>);

    // 3. 2D flat coordinate and stride (column-major and row-major)
    let crd = int_tuple![2i32, 3i32];

    // Col-major stride (1, 4): offset = 2*1 + 3*4 = 14
    let col_strd = int_tuple![1i32, 4i32];
    assert_eq!(inner_product(crd, col_strd), 14i32);

    // Row-major stride (4, 1): offset = 2*4 + 3*1 = 11
    let row_strd = int_tuple![4i32, 1i32];
    assert_eq!(inner_product(crd, row_strd), 11i32);

    // 4. Nested hierarchical coordinates and strides
    // crd = (1, (2, 3)), strd = (1, (4, 12))
    // offset = 1*1 + 2*4 + 3*12 = 1 + 8 + 36 = 45
    let crd_nested = int_tuple![1i32, (2i32, 3i32)];
    let strd_nested = int_tuple![1i32, (4i32, 12i32)];
    assert_eq!(inner_product(crd_nested, strd_nested), 45i32);

    // 5. Basis strides: coordinate projection / identity
    let basis_strd = tuple![e![0], e![1]];
    assert_eq!(inner_product(crd, basis_strd), from_tuple(crd));

    // 6. Transposed Basis: coordinate transposition
    let trans_strd = tuple![e![1], e![0]];
    assert_eq!(inner_product(crd, trans_strd), basis_tuple![3i32, 2i32]);

    // 7. 3D Basis: (c0, c1, c2)
    let crd_3d = int_tuple![10i32, 20i32, 30i32];
    let basis_3d = tuple![e![0], e![1], e![2]];
    assert_eq!(inner_product(crd_3d, basis_3d), from_tuple(crd_3d));
}

#[test]
fn test_inner_product_const_eval() {
    const CRD: TupleContainer<Int<2>, TupleContainer<Int<3>, Nil>> = int_tuple![Int::<2>, Int::<3>];
    const STRD: TupleContainer<Int<1>, TupleContainer<Int<4>, Nil>> =
        int_tuple![Int::<1>, Int::<4>];

    const RES: Int<14> = inner_product(CRD, STRD);
    assert_eq!(RES, Int::<14>);

    const BASIS_STRD: TupleContainer<Basis<Mode<0>>, TupleContainer<Basis<Mode<1>>, Nil>> =
        tuple![e![0], e![1]];
    type ExpectedProj = BasisCombination<
        TupleContainer<BasisTerm<Int<2>, Mode<0>>, TupleContainer<BasisTerm<Int<3>, Mode<1>>, Nil>>,
    >;
    const PROJ: ExpectedProj = inner_product(CRD, BASIS_STRD);
    assert_eq!(PROJ, from_tuple(CRD));
}

#[test]
fn test_prefix_product_single_trait() {
    // 1. Integer shape with scalar init
    assert_eq!(Int::<4>.prefix_product(Int::<1>), Int::<1>);
    assert_eq!(Int::<4>.prefix_product(e![0]), e![0]);

    // 2. Flat tuple with integer scalar init: (3, 2, 4) -> (1, 3, 6)
    let s1 = int_tuple![Int::<3>, Int::<2>, Int::<4>];
    let d1 = prefix_product(s1);
    assert_eq!(d1, int_tuple![Int::<1>, Int::<3>, Int::<6>]);

    // 3. Nested tuple with default init: (3, (2, 4)) -> (1, (3, 6))
    let s2 = int_tuple![Int::<3>, (Int::<2>, Int::<4>)];
    let d2 = prefix_product(s2);
    assert_eq!(d2, int_tuple![Int::<1>, (Int::<3>, Int::<6>)]);

    // 4. Flat tuple with base 2: (4, 8) with base 2 -> (2, 8)
    let s3 = int_tuple![Int::<4>, Int::<8>];
    let d3 = prefix_product_with(s3, Int::<2>);
    assert_eq!(d3, int_tuple![Int::<2>, Int::<8>]);

    // 5. Flat tuple with Basis scalar init: (3, 2) with E(0) -> (1@0, 3@0)
    let s4 = int_tuple![Int::<3>, Int::<2>];
    let d4 = prefix_product_with(s4, e![0]);
    assert_eq!(
        d4,
        tuple![e![0], ScaledBasis::<Int<3>, mode![0]>::new(Int::<3>)]
    );

    // 6. Nested tuple with Basis scalar init: (3, (2, 4)) with E(0) -> (1@0, (3@0, 6@0))
    let s5 = int_tuple![Int::<3>, (Int::<2>, Int::<4>)];
    let d5 = prefix_product_with(s5, e![0]);
    assert_eq!(
        d5,
        tuple![
            e![0],
            (
                ScaledBasis::<Int<3>, mode![0]>::new(Int::<3>),
                ScaledBasis::<Int<6>, mode![0]>::new(Int::<6>)
            )
        ]
    );

    // 7. Tuple shape with Tuple init of Basis: (4, 5) with (E(0), E(1)) -> (E(0), E(1))
    let s6 = int_tuple![Int::<4>, Int::<5>];
    let init6 = tuple![e![0], e![1]];
    let d6 = prefix_product_with(s6, init6);
    assert_eq!(d6, tuple![e![0], e![1]]);

    // 8. Nested tuple with Tuple init of Basis: ((2, 3), (4, 5)) with (E(0), E(1)) -> ((1@0, 2@0), (1@1, 4@1))
    let s7 = int_tuple![(Int::<2>, Int::<3>), (Int::<4>, Int::<5>)];
    let init7 = tuple![e![0], e![1]];
    let d7 = prefix_product_with(s7, init7);
    assert_eq!(
        d7,
        tuple![
            (e![0], ScaledBasis::<Int<2>, mode![0]>::new(Int::<2>)),
            (e![1], ScaledBasis::<Int<4>, mode![1]>::new(Int::<4>))
        ]
    );

    // 9. Nested tuple with Tuple init of Integers: ((2, 3), (4, 5)) with (1, 100) -> ((1, 2), (100, 400))
    let s8 = int_tuple![(Int::<2>, Int::<3>), (Int::<4>, Int::<5>)];
    let init8 = int_tuple![Int::<1>, Int::<100>];
    let d8 = prefix_product_with(s8, init8);
    assert_eq!(
        d8,
        int_tuple![(Int::<1>, Int::<2>), (Int::<100>, Int::<400>)]
    );
}

#[test]
fn test_prefix_product_const_eval() {
    const SHAPE: TupleContainer<Int<3>, TupleContainer<Int<4>, Nil>> =
        int_tuple![Int::<3>, Int::<4>];
    const STRIDE: TupleContainer<Int<1>, TupleContainer<Int<3>, Nil>> = prefix_product(SHAPE);
    assert_eq!(STRIDE, int_tuple![Int::<1>, Int::<3>]);
}

#[test]
fn test_scale_stride_and_functor() {
    // 1. ScaleBasisFunctor directly
    let f0 = ScaleBasisFunctor(e![0]);
    assert_eq!(
        f0.apply(Int::<5>),
        ScaledBasis::<Int<5>, mode![0]>::new(Int::<5>)
    );

    let f_scale = ScaleBasisFunctor(Int::<3>);
    let sb = ScaledBasis::<Int<4>, mode![1]>::new(Int::<4>);
    assert_eq!(
        f_scale.apply(sb),
        ScaledBasis::<Int<12>, mode![1]>::new(Int::<12>)
    );

    // 2. ScaleStride on scalar
    assert_eq!(
        sb.scale_stride(Int::<2>),
        ScaledBasis::<Int<8>, mode![1]>::new(Int::<8>)
    );

    // 3. ScaleStride on flat tuple of ScaledBasis
    let t = tuple![
        ScaledBasis::<Int<2>, mode![0]>::new(Int::<2>),
        ScaledBasis::<Int<3>, mode![1]>::new(Int::<3>)
    ];
    let t_scaled = t.scale_stride(Int::<5>);
    assert_eq!(
        t_scaled,
        tuple![
            ScaledBasis::<Int<10>, mode![0]>::new(Int::<10>),
            ScaledBasis::<Int<15>, mode![1]>::new(Int::<15>)
        ]
    );

    // 4. ScaleStride turning integer tuple into ScaledBasis tuple
    let int_t = int_tuple![Int::<2>, Int::<3>];
    let basis_t = int_t.scale_stride(e![0]);
    assert_eq!(
        basis_t,
        tuple![
            ScaledBasis::<Int<2>, mode![0]>::new(Int::<2>),
            ScaledBasis::<Int<3>, mode![0]>::new(Int::<3>)
        ]
    );
}

#[test]
fn test_codomain_profile() {
    fn profile_of<T: CodomainProfile>() -> TypeId {
        TypeId::of::<T::Profile>()
    }

    // Nil (matches coshape(Nil, Nil) = Int<1> and coprofile = Int<0>)
    assert_eq!(profile_of::<Nil>(), TypeId::of::<LeafProfile>());

    // Integer
    assert_eq!(profile_of::<i32>(), TypeId::of::<LeafProfile>());
    assert_eq!(profile_of::<Int<4>>(), TypeId::of::<LeafProfile>());

    // ScaledBasis (single and sparse/nested)
    assert_eq!(
        profile_of::<ScaledBasis<Int<1>, mode![0]>>(),
        TypeId::of::<TupleProfile<LeafProfile, NilProfile>>()
    );
    assert_eq!(
        profile_of::<ScaledBasis<Int<1>, mode![2]>>(),
        TypeId::of::<
            TupleProfile<
                LeafProfile,
                TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>,
            >,
        >()
    );
    assert_eq!(
        profile_of::<ScaledBasis<Int<1>, mode![1, 0]>>(),
        TypeId::of::<
            TupleProfile<
                LeafProfile,
                TupleProfile<TupleProfile<LeafProfile, NilProfile>, NilProfile>,
            >,
        >()
    );

    // TupleContainer of scalar strides maps to 1D scalar offsets (LeafProfile)
    type T1 = TupleContainer<Int<1>, Nil>;
    assert_eq!(profile_of::<T1>(), TypeId::of::<LeafProfile>());

    type T2 = TupleContainer<Int<1>, TupleContainer<Int<2>, Nil>>;
    assert_eq!(profile_of::<T2>(), TypeId::of::<LeafProfile>());

    // TupleContainer of basis strides merges their active codomain axes
    type B2 = TupleContainer<Basis<mode![0]>, TupleContainer<Basis<mode![1]>, Nil>>;
    assert_eq!(
        profile_of::<B2>(),
        TypeId::of::<TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>>()
    );
}

#[test]
fn pycute_htuple_inner_product_and_prefix_product_examples() {
    // test_htuple.py::TestHTuple::test_inner_product
    assert_eq!(inner_product(2i32, 3i32), 6i32);
    assert_eq!(
        inner_product(int_tuple![1i32, 2i32], int_tuple![3i32, 2i32]),
        7i32
    );
    assert_eq!(
        inner_product(
            int_tuple![(2i32, 3i32), 4i32],
            int_tuple![(2i32, 1i32), 2i32]
        ),
        15i32
    );

    // test_htuple.py::TestHTuple::test_prefix_product
    assert_eq!(prefix_product(2i32), Int::<1>);
    assert_eq!(
        prefix_product(int_tuple![3i32, 2i32]),
        int_tuple![Int::<1>, 3i32]
    );
    assert_eq!(
        prefix_product(int_tuple![3i32, 2i32, 4i32]),
        int_tuple![Int::<1>, 3i32, 6i32]
    );
    assert_eq!(
        prefix_product(int_tuple![(2i32, 3i32), 4i32]),
        int_tuple![(Int::<1>, 2i32), 6i32]
    );
    assert_eq!(
        prefix_product(int_tuple![
            (2i32, 3i32),
            (2i32, 1i32, 2i32),
            (5i32, 2i32, 1i32)
        ]),
        int_tuple![
            (Int::<1>, 2i32),
            (6i32, 12i32, 12i32),
            (24i32, 120i32, 240i32)
        ]
    );
}

#[test]
fn pycute_coshape_examples_and_basis_skeleton_edge_cases() {
    const fn require_congruent<A: [const] Congruent<B>, B: Item>(_: &A, _: &B) {}

    // Integer strides
    assert_eq!(
        coshape(tuple![Int::<4>, Int::<8>], tuple![Int::<1>, Int::<4>]),
        Int::<32>
    );
    assert_eq!(
        coshape(tuple![Int::<4>, Int::<8>], tuple![Int::<8>, Int::<1>]),
        Int::<32>
    );
    assert_eq!(coshape(Int::<8>, Int::<1>), Int::<8>);
    assert_eq!(coshape(Int::<8>, Int::<0>), Int::<1>);
    assert_eq!(coshape(Nil, Nil), Int::<1>);
    assert_eq!(
        coshape(int_tuple![4i32, 8i32], int_tuple![1i32, 4i32]),
        32i32
    );

    // Basis strides
    let c_id = coshape(tuple![Int::<4>, Int::<8>], tuple![e![0], e![1]]);
    assert_eq!(c_id, tuple![Int::<4>, Int::<8>]);
    require_congruent(&c_id, &tuple![Int::<4>, Int::<8>]);

    assert_eq!(
        coshape(tuple![Int::<4>, Int::<8>], tuple![e![1], e![0]]),
        tuple![Int::<8>, Int::<4>]
    );

    // Sparse gap in basis modes -> padded with Int<1>
    assert_eq!(
        coshape(tuple![Int::<4>, Int::<8>], tuple![e![0], e![2]]),
        tuple![Int::<4>, Int::<1>, Int::<8>]
    );

    // Extent-1 modes preserved via stride skeleton
    assert_eq!(
        coshape(tuple![Int::<4>, Int::<1>], tuple![e![0], e![1]]),
        tuple![Int::<4>, Int::<1>]
    );
    assert_eq!(
        coshape(tuple![Int::<1>, Int::<4>], tuple![e![0], e![1]]),
        tuple![Int::<1>, Int::<4>]
    );
    assert_eq!(
        coshape(tuple![Int::<1>, Int::<1>], tuple![e![0], e![1]]),
        tuple![Int::<1>, Int::<1>]
    );
    assert_eq!(coshape(Int::<1>, e![1]), tuple![Int::<1>, Int::<1>]);
    assert_eq!(coshape(Int::<4>, e![0]), tuple![Int::<4>]);

    // Overlapping basis modes combine into 1-tuple
    assert_eq!(
        coshape(tuple![Int::<4>, Int::<2>], tuple![e![0], e![0]]),
        tuple![Int::<5>]
    );

    // Nested extent-1 with nested basis paths
    let c_nested = coshape(
        tuple![(Int::<4>, Int::<1>), Int::<8>],
        tuple![(e![0, 0], e![0, 1]), e![1]],
    );
    assert_eq!(c_nested, tuple![(Int::<4>, Int::<1>), Int::<8>]);
    require_congruent(&c_nested, &tuple![(Int::<4>, Int::<1>), Int::<8>]);

    // Broadcast scalar 0 + basis stride
    assert_eq!(
        coshape(tuple![Int::<4>, Int::<8>], tuple![Int::<0>, e![1]]),
        tuple![Int::<1>, Int::<8>]
    );

    // Dynamic shape + basis stride
    let c_dyn = coshape(int_tuple![4i32, 8i32], tuple![e![0], e![1]]);
    assert_eq!(c_dyn, int_tuple![4i32, 8i32]);
    require_congruent(&c_dyn, &int_tuple![4i32, 8i32]);

    // Compile-time constant evaluation
    const _: () = {
        let cs = coshape(
            Til(Int::<4>, Til(Int::<8>, Nil)),
            Til(Int::<1>, Til(Int::<4>, Nil)),
        );
        assert!(cs == Int::<32>);
        let cb = coshape(
            Til(Int::<4>, Til(Int::<1>, Nil)),
            Til(e![0], Til(e![1], Nil)),
        );
        assert!(cb.0 == Int::<4>);
        assert!(cb.1.0 == Int::<1>);
        let _: Nil = cb.1.1;
        require_congruent(&cb, &Til(Int::<4>, Til(Int::<1>, Nil)));
    };
}

#[test]
fn stride_regression_prefix_product_codomain_profile_accumulate_and_coshape() {
    use core::cmp::Ordering;
    use noodle_cute::basis::{BasisCoeff, BasisCoeffAdd, BasisCoeffMul, BasisCoeffNeg};
    use noodle_cute::stride::CombCoshapeValue;
    use noodle_cute::tuple::Item;

    // 1. B7: PrefixProduct skips unused final leaf multiplication (no spurious overflow)
    assert_eq!(
        prefix_product(int_tuple![Int::<2>, Int::<2_000_000_000>]),
        int_tuple![Int::<1>, Int::<2>]
    );
    assert_eq!(
        prefix_product(int_tuple![2i32, 2_000_000_000i32]),
        int_tuple![Int::<1>, 2i32]
    );
    assert_eq!(
        prefix_product(int_tuple![(Int::<2>, Int::<3>), Int::<2_000_000_000>]),
        int_tuple![(Int::<1>, Int::<2>), Int::<6>]
    );
    assert_eq!(
        prefix_product(tuple![Int::<2>, Int::<2_000_000_000>, Nil]),
        tuple![Int::<1>, Int::<2>, Nil]
    );
    assert_eq!(
        prefix_product(tuple![Nil, Int::<2>, Int::<2_000_000_000>]),
        tuple![Nil, Int::<1>, Int::<2>]
    );
    assert_eq!(
        prefix_product_with(
            tuple![
                (Int::<2>, Int::<2_000_000_000>),
                (Int::<3>, Int::<2_000_000_000>)
            ],
            tuple![e![0], e![1]]
        ),
        tuple![(e![0], Int::<2> * e![0]), (e![1], Int::<3> * e![1])]
    );

    // 2. B6: CodomainProfile reflects actual codomain structure (via StrideSkeleton + CombAsTuple)
    fn profile_of<T: CodomainProfile>() -> TypeId {
        TypeId::of::<T::Profile>()
    }

    type ScalarStride2D = TupleContainer<Int<1>, TupleContainer<Int<4>, Nil>>;
    assert_eq!(profile_of::<ScalarStride2D>(), TypeId::of::<LeafProfile>());

    type BasisStride2D = TupleContainer<Basis<Mode<0>>, TupleContainer<Basis<Mode<1>>, Nil>>;
    assert_eq!(
        profile_of::<BasisStride2D>(),
        TypeId::of::<TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>>()
    );

    type BasisStrideTransposed =
        TupleContainer<Basis<Mode<1>>, TupleContainer<Basis<Mode<0>>, Nil>>;
    assert_eq!(
        profile_of::<BasisStrideTransposed>(),
        TypeId::of::<TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>>()
    );

    type BroadcastAndBasis =
        TupleContainer<Int<0>, TupleContainer<ScaledBasis<Int<1>, Mode<0>>, Nil>>;
    assert_eq!(
        profile_of::<BroadcastAndBasis>(),
        TypeId::of::<TupleProfile<LeafProfile, NilProfile>>()
    );

    type ZeroBasis = ScaledBasis<Int<0>, Mode<2>>;
    assert_eq!(profile_of::<ZeroBasis>(), TypeId::of::<LeafProfile>());
    assert_eq!(profile_of::<Nil>(), TypeId::of::<LeafProfile>());

    // B6b: CombCoshapeValue<Nil> for ModeEnd term (previously dead bound `Self: Leaf`)
    let mode_end_term = Til(BasisTerm::<Int<4>, ModeEnd>::new(Int::<4>), Nil);
    assert_eq!(mode_end_term.comb_coshape_value(Nil), Int::<5>);
    assert_eq!(
        coshape(Int::<4>, ScaledBasis::<Int<2>, ModeEnd>::new(Int::<2>)),
        Int::<7>
    );

    // 3. M4: Accumulate with custom non-Integer BasisCoeff & symmetric BasisCombination<Nil>
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct F2(u8);

    const impl Item for F2 {
        const RANK: int = 1;
        const DEPTH: int = 0;
        const LEAVES: int = 1;
        type Profile = LeafProfile;
    }

    const impl BasisCoeff for F2 {
        type Unit = F2;
        fn is_zero(&self) -> bool {
            (self.0 & 1) == 0
        }
        fn cmp_zero(&self) -> Ordering {
            (self.0 & 1).cmp(&0)
        }
        fn static_cmp_zero(&self) -> Option<Ordering> {
            None
        }
        fn algebra_unit() -> F2 {
            F2(1)
        }
    }
    const impl BasisCoeffAdd<F2> for F2 {
        type Output = F2;
        fn coeff_add(self, rhs: F2) -> F2 {
            F2((self.0 ^ rhs.0) & 1)
        }
    }
    const impl BasisCoeffNeg for F2 {
        type Output = F2;
        fn coeff_neg(self) -> F2 {
            self
        }
    }
    const impl BasisCoeffMul<F2> for F2 {
        type Output = F2;
        fn coeff_mul(self, rhs: F2) -> F2 {
            F2((self.0 & rhs.0) & 1)
        }
    }
    const impl BasisCoeffMul<Int<1>> for F2 {
        type Output = F2;
        fn coeff_mul(self, _: Int<1>) -> F2 {
            self
        }
    }

    let f2_basis = ScaledBasis::<F2, Mode<0>>::new(F2(1));
    assert_eq!(f2_basis.accumulate(Nil), f2_basis);
    assert_eq!(f2_basis.accumulate(Int::<0>), f2_basis);
    assert_eq!(
        inner_product(tuple![Int::<1>, Nil], tuple![f2_basis, Nil]),
        f2_basis
    );

    // Symmetric BasisCombination<Nil> accumulation with non-zero scalar
    assert_eq!(
        inner_product(tuple![3i32, Int::<0>], tuple![5i32, e![0]]),
        15i32
    );
    assert_eq!(
        inner_product(tuple![Int::<0>, 3i32], tuple![e![0], 5i32]),
        15i32
    );
    assert_eq!(
        coshape(tuple![Int::<1>, Int::<1>], tuple![e![1], Int::<0>]),
        tuple![Int::<1>, Int::<1>]
    );
}

#[test]
fn test_stride_mul_and_prefix_product_panics_on_overflow() {
    // StrideMul on integers and ScaledBasis (including static zero and overflow)
    assert_eq!(Int::<4>.stride_mul(Int::<3>), Int::<12>);
    assert!(std::panic::catch_unwind(|| 1_500_000_000i32.stride_mul(2i32)).is_err());
    assert_eq!(
        e![0].stride_mul(Int::<4>),
        ScaledBasis::<Int<4>, mode![0]>::new(Int::<4>)
    );
    assert_eq!(e![0].stride_mul(Int::<0>), BasisCombination::<Nil>::zero());
    assert!(
        std::panic::catch_unwind(|| {
            ScaledBasis::<int, mode![0]>::new(1_500_000_000i32).stride_mul(2i32)
        })
        .is_err()
    );

    // prefix_product_with succeeds and skips final unused multiplication
    assert_eq!(
        prefix_product_with(int_tuple![2i32, 1_500_000_000i32], Int::<4>),
        int_tuple![Int::<4>, 8i32]
    );
    // prefix_product_with panics on intermediate overflow
    assert!(
        std::panic::catch_unwind(|| {
            prefix_product_with(int_tuple![1_000_000i32, 10_000i32, 2i32], Int::<8>)
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            prefix_product_with(int_tuple![1_000_000i32, 10_000i32, 2i32], Int::<8> * e![1])
        })
        .is_err()
    );
}
