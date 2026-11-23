#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    const_trait_impl,
    const_cmp
)]
#![allow(incomplete_features)]

use core::ops::Bound as CoreBound;
use std::cell::{Cell, RefCell};
use std::panic::catch_unwind;

use noodle_cute::integer::{ArithAdd, ArithMul, Int, Integer, Product, Sum, add, ge, int, lt, mul};
use noodle_cute::slice::{Bound, ResolvedSlice, Slice};
use noodle_cute::tuple::*;
use noodle_cute::{int_tuple, mode, slice, slices, tuple, walk_mode};

const I32_MIN: int = int::MIN;
const I32_MAX: int = int::MAX;

const fn require_weak<L: [const] WeaklyCongruent<R>, R: Item>(_: &L, _: &R) {}
const fn require_congruent<L: [const] Congruent<R>, R: Item>(_: &L, _: &R) {}

#[derive(Clone, Copy)]
struct Offset(int);

const impl<I: [const] Integer> TransformLeafFunctor<I> for Offset {
    type Output = int;
    fn apply(&self, leaf: I) -> Self::Output {
        add(leaf.value(), self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RecordedLeaf(int, usize);

impl Item for RecordedLeaf {
    const RANK: int = 1;
    const DEPTH: int = 0;
    const LEAVES: int = 1;
    type Profile = LeafProfile;
}

#[test]
fn test_item_properties_display_and_equality() {
    // Leaf properties
    const {
        assert!(int::RANK == 1);
        assert!(int::DEPTH == 0);
        assert!(int::LEAVES == 1);
        assert!(!<int as Item>::IS_STATIC);

        assert!(Int::<42>::RANK == 1);
        assert!(Int::<42>::DEPTH == 0);
        assert!(Int::<42>::LEAVES == 1);
        assert!(<Int<42> as Item>::IS_STATIC);
        assert!(IS_STATIC::<Int<42>>);

        assert!(Nil::RANK == 0);
        assert!(Nil::DEPTH == 1);
        assert!(Nil::LEAVES == 0);
        assert!(Nil::IS_STATIC);
        assert!(Nil == Nil);
    }

    // Structural queries across flat, mixed, all-static, nested, and deep hierarchies
    type Flat3 = TupleContainer<int, TupleContainer<int, TupleContainer<int, Nil>>>;
    type AllStatic = TupleContainer<
        Int<1>,
        TupleContainer<TupleContainer<Int<2>, TupleContainer<Nil, Nil>>, Nil>,
    >;
    type Mixed = TupleContainer<Int<1>, TupleContainer<int, Nil>>;
    type Nested = TupleContainer<
        int,
        TupleContainer<TupleContainer<int, TupleContainer<int, Nil>>, TupleContainer<int, Nil>>,
    >;
    type DeepInner = TupleContainer<int, TupleContainer<int, Nil>>;
    type DeepRight = TupleContainer<int, TupleContainer<DeepInner, Nil>>;
    type Deep = TupleContainer<DeepInner, TupleContainer<DeepRight, Nil>>;

    const {
        assert!(Flat3::RANK == 3 && Flat3::DEPTH == 1 && Flat3::LEAVES == 3);
        assert!(!IS_STATIC::<Flat3>);

        assert!(IS_STATIC::<AllStatic> && AllStatic::IS_STATIC);
        assert!(!IS_STATIC::<Mixed>);

        assert!(Nested::RANK == 3 && Nested::DEPTH == 2 && Nested::LEAVES == 4);
        assert!(Deep::RANK == 2 && Deep::DEPTH == 3 && Deep::LEAVES == 5);
    }

    let mixed = Til(Int::<1>, Til(2i32, Nil));
    assert!(mixed.is_static_at::<0>());
    assert!(!mixed.is_static_at::<1>());
    assert!(!mixed.is_static());

    // Display formatting
    let t = tuple!(Int::<10>, (20i32, Int::<30>), (), 40i32);
    assert_eq!(
        (t.rank(), t.depth(), t.length(), t.is_static()),
        (4, 2, 4, false)
    );
    assert_eq!(format!("{t}"), "(10, (20, 30), (), 40)");
    assert_eq!(format!("{}", Nil), "()");
    assert_eq!(format!("{}", tuple!(7i32)), "(7,)");
    assert_eq!(format!("{}", tuple!(())), "((),)");

    // All-static hierarchy, ZST layout, and heterogeneous const PartialEq
    type Pair23 = TupleContainer<Int<2>, TupleContainer<Int<3>, Nil>>;
    type StaticTree = TupleContainer<Int<1>, TupleContainer<Pair23, Nil>>;
    const STATIC_T: StaticTree = tuple!(Int::<1>, (Int::<2>, Int::<3>));
    assert_eq!(core::mem::size_of_val(&STATIC_T), 0);
    const {
        assert!(STATIC_T.rank() == 2);
        assert!(STATIC_T.depth() == 2);
        assert!(STATIC_T.length() == 3);
        assert!(STATIC_T.is_static());
        assert!(IS_STATIC::<StaticTree>);
        assert!(STATIC_T == tuple!(1i32, (2i32, 3i32)));
        assert!(tuple!(1i32, (2i32, 3i32)) == STATIC_T);
        assert!(STATIC_T != tuple!(1i32, (2i32, 4i32)));
    }

    // Deeply nested Nil
    let nested_nil = tuple!((), (()));
    assert_eq!(
        (nested_nil.rank(), nested_nil.depth(), nested_nil.length()),
        (2, 3, 0)
    );
    assert!(nested_nil.is_static());

    // Til::new constructor and IntTuple subset relationship
    let constructed = Til::new(Int::<1>, Til::new(2i32, Nil));
    assert_eq!(constructed, int_tuple!(Int::<1>, 2i32));

    const fn require_integer_tuple<T: IntTuple>(value: T) -> T {
        value
    }
    fn require_general<T: Tuple>(_: T) {}

    const EMPTY: Nil = require_integer_tuple(int_tuple![]);
    const NESTED: TupleContainer<Int<1>, TupleContainer<TupleContainer<int, Nil>, Nil>> =
        require_integer_tuple(tuple![Int::<1>, (2i32)]);
    assert_eq!(EMPTY, tuple![]);
    assert_eq!(NESTED, int_tuple![Int::<1>, (2i32)]);
    require_general(NESTED);
    require_integer_tuple(NESTED.flatten());
    require_integer_tuple(NESTED.append(int_tuple![3]));
    require_integer_tuple(NESTED.transform(&Offset(1)));
    require_integer_tuple(int_tuple![Nil, 1]);
    require_integer_tuple(int_tuple!(Int::<1>, ((), (2i32, Int::<3>))));

    // Non-const custom Item works seamlessly with Til at runtime
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct RuntimeLeaf(int);
    impl Item for RuntimeLeaf {
        const RANK: int = 1;
        const DEPTH: int = 0;
        const LEAVES: int = 1;
        type Profile = LeafProfile;
    }
    let custom = tuple!(RuntimeLeaf(1), (RuntimeLeaf(2)));
    require_general(custom);
    assert_eq!((custom.rank(), custom.depth(), custom.length()), (2, 2, 2));
    assert_eq!(
        custom,
        Til(RuntimeLeaf(1), Til(Til(RuntimeLeaf(2), Nil), Nil))
    );
    assert_eq!(custom.get::<0>(), RuntimeLeaf(1));
}

#[test]
fn test_profiles_and_blanket_leaf_with_custom_items() {
    fn require_tuple_trait<T: Tuple>(_: T) {}
    fn require_profile<I: Item<Profile = P>, P>(_: I) {}

    require_tuple_trait(Nil);
    require_tuple_trait(int_tuple![1, (Int::<2>, 3i32)]);
    require_profile::<_, NilProfile>(Nil);
    require_profile::<_, LeafProfile>(Int::<3>);
    require_profile::<_, LeafProfile>(4i32);
    require_profile::<_, TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>>(
        int_tuple![1, 2],
    );

    // Blanket `Leaf` allows custom `Item<Profile = LeafProfile>` without manual `impl Leaf`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct CustomLeaf(int);
    const impl Item for CustomLeaf {
        const RANK: int = 1;
        const DEPTH: int = 0;
        const LEAVES: int = 1;
        type Profile = LeafProfile;
    }

    #[derive(Clone, Copy)]
    struct MapCustom(int);
    impl TransformLeafFunctor<CustomLeaf> for MapCustom {
        type Output = int;
        fn apply(&self, leaf: CustomLeaf) -> int {
            add(leaf.0, self.0)
        }
    }
    impl<I: Integer> TransformLeafFunctor<I> for MapCustom {
        type Output = int;
        fn apply(&self, leaf: I) -> int {
            add(leaf.value(), self.0)
        }
    }

    require_profile::<_, LeafProfile>(CustomLeaf(5));
    let t = tuple![CustomLeaf(1), (Int::<2>, 3i32)];
    assert_eq!(t.transform(&MapCustom(10)), int_tuple![11, (12, 13)]);
    assert_eq!(CustomLeaf(5).wrap(), tuple![CustomLeaf(5)]);
    const {
        assert!(!IS_STATIC::<CustomLeaf>);
    }
}

#[test]
fn test_macros_tuple_mode_walk_mode_and_single_evaluation() {
    assert_eq!(tuple!(), Nil);
    assert_eq!(tuple!(1i32), Til(1i32, Nil));
    assert_eq!(tuple!(1i32,), Til(1i32, Nil));
    assert_eq!(tuple!(()), Til(Nil, Nil));
    assert_eq!(tuple!((),), Til(Nil, Nil));
    assert_eq!(tuple!((), ()), Til(Nil, Til(Nil, Nil)));
    assert_eq!(tuple!((())), Til(Til(Nil, Nil), Nil));
    assert_eq!(tuple!(((),)), Til(Til(Nil, Nil), Nil));
    assert_eq!(tuple!(1i32, (2i32)), Til(1i32, Til(Til(2i32, Nil), Nil)));
    assert_eq!(tuple!(1i32, (2i32,)), Til(1i32, Til(Til(2i32, Nil), Nil)));
    assert_eq!(
        tuple!((1i32 + 2i32) * 3i32, 4i32 + 5i32),
        tuple!(9i32, 9i32)
    );
    assert_eq!(
        tuple!(((1i32 + 2i32) * 3i32, 4i32), (5i32)),
        Til(Til(9i32, Til(4i32, Nil)), Til(Til(5i32, Nil), Nil))
    );

    // Macros evaluate each expression once
    let mut calls = 0;
    let mut next = || {
        calls += 1;
        calls
    };
    assert_eq!(int_tuple![next(), (next(), next()),], int_tuple![1, (2, 3)]);
    assert_eq!(calls, 3);
    let mut next = || {
        calls += 1;
        calls
    };
    assert_eq!(tuple![next(), (next())], tuple![4, (5)]);
    assert_eq!(calls, 5);

    // Const and runtime walk_mode! including expression indices and single evaluation
    const {
        let t = tuple!(Int::<10>, (Int::<20>, (Int::<30>, Int::<40>)));
        assert!(walk_mode!(t) == t);
        assert!(walk_mode!(t, 0) == Int::<10>);
        assert!(walk_mode!(t, 1, 0) == Int::<20>);
        assert!(walk_mode!(t, 1, 1, 0) == Int::<30>);
        assert!(walk_mode!(t, -1, -1, -1) == Int::<40>);
    }

    let flat = int_tuple![10, 20, 30];
    assert_eq!(walk_mode!(flat), flat);
    assert_eq!(walk_mode!(flat, 2), 30);
    assert_eq!(walk_mode!(flat, 1 + 1,), 30);

    let nested = int_tuple![(0, (10, 20, 30))];
    assert_eq!(walk_mode!(nested, 0, 1, 2), 30);
    assert_eq!(walk_mode!(nested, -1, -1, -1,), 30);

    let mut walk_calls = 0;
    assert_eq!(
        walk_mode!(
            {
                walk_calls += 1;
                nested
            },
            0,
            1,
            2
        ),
        30
    );
    assert_eq!(walk_calls, 1);

    let dyn_t = tuple!(10i32, (20i32, (30i32, 40i32)));
    assert_eq!(walk_mode!(dyn_t, 1, 1, 0,), 30);
    assert_eq!(
        select::<mode!(0, -1), _>(&dyn_t),
        tuple!(10i32, (20i32, (30i32, 40i32)))
    );
    assert_eq!(select::<mode!(), _>(&dyn_t), Nil);
    assert_eq!(
        replace::<mode!(1, 1, -1), _, _>(&dyn_t, 99i32),
        tuple!(10i32, (20i32, (30i32, 99i32)))
    );
}

#[test]
fn test_congruence_relations() {
    #[derive(Clone, Copy)]
    struct TagLeaf;
    const impl Item for TagLeaf {
        const RANK: int = 1;
        const DEPTH: int = 0;
        const LEAVES: int = 1;
        type Profile = LeafProfile;
    }

    const {
        require_weak(&Int::<32>, &int_tuple![3, (4, 5)]);
        require_congruent(&int_tuple![Int::<4>, 8i32], &int_tuple![5i32, Int::<7>]);
    }

    let a = tuple!(Int::<2>, (3i32, Int::<4>));
    let b = tuple!(5i32, (Int::<6>, 7i32));
    let coarse = tuple!(Int::<1>, 2i32);

    require_congruent(&a, &b);
    require_congruent(&b, &a);
    require_congruent(&Int::<31>, &42i32);
    require_congruent(&Nil, &Nil);
    require_congruent(&tuple!((), Int::<1>), &tuple!((), 9i32));
    require_congruent(&int_tuple![(Int::<1>, 2i32)], &int_tuple![(3i32, 4i32)]);
    require_congruent(&TagLeaf, &Int::<7>);
    require_congruent(&tuple![TagLeaf, TagLeaf], &int_tuple![4, 8]);

    require_weak(&Int::<0>, &a);
    require_weak(&Int::<0>, &Nil);
    require_weak(&TagLeaf, &int_tuple![4, 5]);
    require_weak(&coarse, &a);
    require_weak(&int_tuple![3, 4], &int_tuple![5, (6, 7)]);
    require_weak(&int_tuple![(1, 2), 3], &int_tuple![(4, (5, 6)), 7]);
    require_weak(&Nil, &Nil);
    require_weak(&a, &b);
}

#[test]
fn test_indexing_get_mode_queries_and_select() {
    // NORMALIZED_INDEX boundaries
    const {
        type Pair = TupleContainer<int, TupleContainer<int, Nil>>;
        assert!(NORMALIZED_INDEX::<0, Pair> == 0);
        assert!(NORMALIZED_INDEX::<1, Pair> == 1);
        assert!(NORMALIZED_INDEX::<2, Pair> == 2);
        assert!(NORMALIZED_INDEX::<-1, Pair> == 1);
        assert!(NORMALIZED_INDEX::<-2, Pair> == 0);

        type Three = TupleContainer<Int<1>, TupleContainer<Int<2>, TupleContainer<Int<3>, Nil>>>;
        assert!(NORMALIZED_INDEX::<0, Three> == 0);
        assert!(NORMALIZED_INDEX::<2, Three> == 2);
        assert!(NORMALIZED_INDEX::<3, Three> == 3);
        assert!(NORMALIZED_INDEX::<-1, Three> == 2);
        assert!(NORMALIZED_INDEX::<-3, Three> == 0);
    }

    // Const Get & mode queries
    const C0: int = Get::<0>::get(&int_tuple![10, 20, 30]);
    const RANK: int = int_tuple![1, (2, 3)].rank();
    const DEPTH: int = int_tuple![1, (2, 3)].depth();
    const LENGTH: int = int_tuple![1, (2, 3)].length();
    const MODE_RANK: int = int_tuple![10, (20, 30), 40].rank_at::<1>();
    const MODE_DEPTH: int = int_tuple![10, (20, 30), 40].depth_at::<-2>();
    const MODE_LEN: int = int_tuple![10, (20, 30), 40].length_at::<1>();
    const MODE_STAT: bool = int_tuple![Int::<1>, (2i32, 3i32)].is_static_at::<0>();
    assert_eq!(C0, 10);
    assert_eq!((RANK, DEPTH, LENGTH), (2, 2, 3));
    assert_eq!(
        (MODE_RANK, MODE_DEPTH, MODE_LEN, MODE_STAT),
        (2, 1, 2, true)
    );

    // Single-element and nested combinations
    let single = int_tuple![99];
    assert_eq!(single.get::<0>(), 99);
    assert_eq!(single.get::<-1>(), 99);

    let multi = int_tuple![(10, 20), (30, (40, 50))];
    assert_eq!(multi.get::<0>(), int_tuple![10, 20]);
    assert_eq!(multi.get::<-2>(), int_tuple![10, 20]);
    assert_eq!(multi.get::<1>(), int_tuple![30, (40, 50)]);
    assert_eq!(multi.get::<-1>(), int_tuple![30, (40, 50)]);
    assert_eq!(walk_mode!(multi, 0, 0), 10);
    assert_eq!(walk_mode!(multi, 0, 1), 20);
    assert_eq!(walk_mode!(multi, -2, -1), 20);
    assert_eq!(walk_mode!(multi, 1, 0), 30);
    assert_eq!(walk_mode!(multi, 1, 1, 0), 40);
    assert_eq!(walk_mode!(multi, 1, 1, 1), 50);
    assert_eq!(walk_mode!(multi, -1, -1, -1), 50);
    assert_eq!(walk_mode!(multi, -1, -1, -2), 40);

    // GetOutput static type checking
    let static_tuple = int_tuple![Int::<3>, (Int::<4>, Int::<5>)];
    type Pair45 = TupleContainer<Int<4>, TupleContainer<Int<5>, Nil>>;
    type StaticTuple = TupleContainer<Int<3>, TupleContainer<Pair45, Nil>>;
    let s0: GetOutput<StaticTuple, 0> = static_tuple.get::<0>();
    assert_eq!(s0, Int::<3>);
    let s1: GetOutput<StaticTuple, 1> = static_tuple.get::<1>();
    let _: Int<4> = s1.get::<0>();
    let _: Int<5> = s1.get::<1>();
    let s_last = static_tuple.get::<-1>();
    let _: Int<5> = s_last.get::<-1>();

    let inner = tuple!(20i32, (Int::<30>, Int::<40>));
    let t = tuple!(Int::<10>, inner, 50i32);
    assert_eq!(Get::<0>::get(&t), Int::<10>);
    assert_eq!(Get::<-1>::get(&t), 50);
    assert_eq!(
        (
            t.rank_at::<0>(),
            t.depth_at::<0>(),
            t.length_at::<0>(),
            t.is_static_at::<0>()
        ),
        (1, 0, 1, true)
    );
    assert_eq!(
        (
            t.rank_at::<1>(),
            t.depth_at::<1>(),
            t.length_at::<1>(),
            t.is_static_at::<1>()
        ),
        (2, 2, 3, false)
    );
    assert_eq!(
        (
            t.rank_at::<-1>(),
            t.depth_at::<-1>(),
            t.length_at::<-1>(),
            t.is_static_at::<-1>()
        ),
        (1, 0, 1, false)
    );

    // Select (const and runtime: reordering, repetition, negative indices, empty selection)
    const SEL: TupleContainer<Int<30>, TupleContainer<Int<10>, Nil>> =
        select::<mode![-1, 0], _>(&int_tuple![Int::<10>, Int::<20>, Int::<30>]);
    assert_eq!(SEL, int_tuple![Int::<30>, Int::<10>]);

    let src = int_tuple![10, (20, 30), 40, 50];
    assert_eq!(select::<mode![2, 0, -2], _>(&src), int_tuple![40, 10, 40]);
    assert_eq!(
        select::<mode!(-1, 0, -1), _>(&t),
        tuple!(50i32, Int::<10>, 50i32)
    );
    assert_eq!(select::<mode!(1), _>(&t), tuple!(inner));
    assert_eq!(select::<mode!(), _>(&t), Nil);
    assert_eq!(select::<mode!(), _>(&42i32), Nil);

    // Select & slice equivalence (PyCuTe TestSelect)
    let tup4 = int_tuple![1, 2, 3, 4];
    assert_eq!(select::<mode![0, 2], _>(&tup4), int_tuple![1, 3]);
    assert_eq!(select::<mode![2], _>(&tup4), int_tuple![3]);
    assert_eq!(select::<mode![0, 1, 3], _>(&tup4), int_tuple![1, 2, 4]);
    assert_eq!(
        select::<mode![0, 1], _>(&int_tuple![2, (3, 4), 5]),
        int_tuple![2, (3, 4)]
    );
    assert_eq!(
        tup4.slice::<{ slice!([1..3]) }>(),
        select::<mode![1, 2], _>(&tup4)
    );
    assert_eq!(tup4.slice::<{ slice!([1..3]) }>(), int_tuple![2, 3]);
}

#[test]
#[allow(clippy::reversed_empty_ranges)]
fn test_slice_resolution_and_tuple_slicing() {
    // Bound conversions
    const {
        assert!(matches!(
            Bound::from_core(CoreBound::Included(3)),
            Bound::Included(3)
        ));
        assert!(matches!(
            Bound::from_core(CoreBound::Excluded(-2)),
            Bound::Excluded(-2)
        ));
        assert!(matches!(
            Bound::from_core(CoreBound::Unbounded),
            Bound::Unbounded
        ));
        assert!(matches!(
            Bound::Included(3).into_core(),
            CoreBound::Included(3)
        ));
    }
    assert_eq!(Bound::from(CoreBound::Included(5)), Bound::Included(5));
    assert_eq!(CoreBound::from(Bound::Excluded(5)), CoreBound::Excluded(5));

    // Slice construction and macros
    const S_ARR: [Slice; 4] = slices!(1..4, ..=2, ..: -1, -2);
    assert_eq!(S_ARR[0], Slice::from(1..4));
    assert_eq!(S_ARR[1], Slice::from(..=2));
    assert_eq!(S_ARR[2], Slice::new(.., -1));
    assert_eq!(S_ARR[3], Slice::from(-2));
    assert_eq!(
        Slice::from((Bound::Excluded(0), Bound::Excluded(4))).get(),
        (Bound::Excluded(0), Bound::Excluded(4), 1)
    );
    assert_eq!(
        Slice::from((CoreBound::Included(1), CoreBound::Unbounded)),
        Slice::from(1..)
    );
    assert_eq!(slices!(), [] as [Slice; 0]);

    // Resolution semantics including negative bounds, clamping, empty dims, and extreme steps
    const {
        assert!(Slice::new(.., 0).resolve(5).is_none());
        assert!(Slice::new(.., 1).resolve(-1).is_none());

        assert!(matches!(
            Slice::from_slice(1..4).resolve(5),
            Some(ResolvedSlice {
                start: 1,
                step: 1,
                len: 3
            })
        ));
        assert!(matches!(
            Slice::from_slice(..=2).resolve(5),
            Some(ResolvedSlice {
                start: 0,
                step: 1,
                len: 3
            })
        ));
        assert!(matches!(
            Slice::new(-100..100, 2).resolve(5),
            Some(ResolvedSlice {
                start: 0,
                step: 2,
                len: 3
            })
        ));
        assert!(matches!(
            Slice::from_slice(4..2).resolve(5),
            Some(ResolvedSlice {
                start: 4,
                step: 1,
                len: 0
            })
        ));
        assert!(matches!(
            Slice::from_slice(I32_MIN..=I32_MAX).resolve(5),
            Some(ResolvedSlice {
                start: 0,
                step: 1,
                len: 5
            })
        ));
        assert!(matches!(
            Slice::new(.., I32_MAX).resolve(I32_MAX),
            Some(ResolvedSlice {
                start: 0,
                step: I32_MAX,
                len: 1
            })
        ));

        assert!(matches!(
            Slice::new(.., -1).resolve(5),
            Some(ResolvedSlice {
                start: 4,
                step: -1,
                len: 5
            })
        ));
        assert!(matches!(
            Slice::new(3..=1, -1).resolve(5),
            Some(ResolvedSlice {
                start: 3,
                step: -1,
                len: 3
            })
        ));
        assert!(matches!(
            Slice::new(3..1, -1).resolve(5),
            Some(ResolvedSlice {
                start: 3,
                step: -1,
                len: 2
            })
        ));
        assert!(matches!(
            Slice::new(1..3, -1).resolve(5),
            Some(ResolvedSlice {
                start: 1,
                step: -1,
                len: 0
            })
        ));
        assert!(matches!(
            Slice::new(.., -1).resolve(0),
            Some(ResolvedSlice {
                start: -1,
                step: -1,
                len: 0
            })
        ));
        const LAST_MAX: int = I32_MAX - 1;
        assert!(matches!(
            Slice::new(.., -1).resolve(I32_MAX),
            Some(ResolvedSlice {
                start: LAST_MAX,
                step: -1,
                len: I32_MAX
            })
        ));
        assert!(matches!(
            Slice::new(.., I32_MIN).resolve(5),
            Some(ResolvedSlice {
                start: 4,
                step: I32_MIN,
                len: 1
            })
        ));
        assert!(matches!(
            Slice::new(.., I32_MIN).resolve(I32_MAX),
            Some(ResolvedSlice {
                start: LAST_MAX,
                step: I32_MIN,
                len: 1
            })
        ));
        assert!(matches!(
            Slice::new((Bound::Excluded(5), Bound::Included(0)), -2).resolve(5),
            Some(ResolvedSlice {
                start: 4,
                step: -2,
                len: 3
            })
        ));
        assert!(matches!(
            Slice::new((Bound::Excluded(I32_MAX), Bound::Included(I32_MIN)), -1).resolve(5),
            Some(ResolvedSlice {
                start: 4,
                step: -1,
                len: 5
            })
        ));
    }

    // Tuple slicing (const and runtime)
    const {
        let t = tuple!(Int::<10>, Int::<20>, Int::<30>, Int::<40>, Int::<50>);
        assert!(t.slice::<{ slice!([1..4]) }>() == tuple!(Int::<20>, Int::<30>, Int::<40>));
        assert!(t.slice::<{ slice!([..3]) }>() == tuple!(Int::<10>, Int::<20>, Int::<30>));
        assert!(t.slice::<{ slice!([..=2]) }>() == tuple!(Int::<10>, Int::<20>, Int::<30>));
        assert!(t.slice::<{ slice!([..] : 2) }>() == tuple!(Int::<10>, Int::<30>, Int::<50>));
        assert!(
            t.slice::<{ slice!([..] : -1) }>()
                == tuple!(Int::<50>, Int::<40>, Int::<30>, Int::<20>, Int::<10>)
        );
        assert!(t.slice::<{ slice!([..] : -2) }>() == tuple!(Int::<50>, Int::<30>, Int::<10>));
        assert!(t.slice::<{ slice!([3..=1] : -1) }>() == tuple!(Int::<40>, Int::<30>, Int::<20>));
        assert!(t.slice::<{ slice!([3..1] : -1) }>() == tuple!(Int::<40>, Int::<30>));
        assert!(t.slice::<{ slice!([-2..]) }>() == tuple!(Int::<40>, Int::<50>));
        assert!(t.slice::<{ slice!([-1]) }>() == tuple!(Int::<50>));
        assert!(t.slice::<{ slice!([-4]) }>() == tuple!(Int::<20>));
        assert!(t.slice::<{ slice!([4..1]) }>() == Nil);
        assert!(t.slice::<{ slice!([..] : I32_MIN) }>() == tuple!(Int::<50>));
        assert!(t.slice::<{ slice!([..] : I32_MAX) }>() == tuple!(Int::<10>));
        assert!(slice_range::<{ slice!([..]) }, _>(&Nil) == Nil);
        assert!(slice_range::<{ slice!([..]) }, _>(&Int::<7>) == tuple!(Int::<7>));
        assert!(slice_range::<{ slice!([1..]) }, _>(&Int::<7>) == Nil);

        let quad = int_tuple![Int::<10>, Int::<20>, Int::<30>, Int::<40>];
        assert!(
            slice_range::<{ slice!([2..0] : -1) }, _>(&quad) == int_tuple![Int::<30>, Int::<20>]
        );

        // Extreme steps without overflowing next index
        let c_pair = int_tuple![Int::<10>, Int::<20>];
        assert!(c_pair.slice::<{ slice!([1..] : I32_MAX) }>() == int_tuple![Int::<20>]);
        assert!(c_pair.slice::<{ slice!([0..] : I32_MIN) }>() == int_tuple![Int::<10>]);
        assert!(c_pair.slice::<{ slice!([1..] : I32_MIN) }>() == int_tuple![Int::<20>]);
    }

    let pair = int_tuple![10i32, 20i32];
    assert_eq!(
        pair.slice::<{ slice!([1..] : I32_MAX) }>(),
        int_tuple![20i32]
    );
    assert_eq!(
        pair.slice::<{ slice!([0..] : I32_MIN) }>(),
        int_tuple![10i32]
    );
    assert_eq!(
        pair.slice::<{ slice!([1..] : I32_MIN) }>(),
        int_tuple![20i32]
    );

    assert!(catch_unwind(|| resolve_tuple_slice(Slice::new(.., 0), 3)).is_err());
    assert!(catch_unwind(|| resolve_tuple_slice(Slice::new(.., 1), -1)).is_err());
}

#[test]
fn test_restructure_wrap_unwrap_append_prepend_flatten_unflatten_repeat() {
    // Wrap / Unwrap (including PyCuTe TestHTuple::test_wrap_unwrap)
    const {
        assert!(Int::<3>.wrap() == int_tuple![Int::<3>]);
        assert!((-7i32).wrap() == int_tuple![-7]);
        assert!(Nil.wrap() == Nil);
        assert!(tuple!(Int::<1>, Int::<2>).wrap() == tuple!(Int::<1>, Int::<2>));

        assert!(Til(Til(Int::<3>, Nil), Nil).unwrap() == Int::<3>);
        assert!(Til(Til(Til(Til(42i32, Nil), Nil), Nil), Nil).unwrap() == 42);
        assert!(tuple!((Int::<5>)).unwrap() == Int::<5>);
        assert!(tuple!(()).unwrap() == Nil);
        assert!(Nil.unwrap() == Nil);
        assert!(Int::<5>.unwrap() == Int::<5>);
        assert!(int_tuple![7].wrap() == int_tuple![7]);
        assert!(int_tuple![7, 8].wrap() == int_tuple![7, 8]);
        assert!(int_tuple![3].wrap().unwrap() == 3);
        assert!(Til(Til(int_tuple![1, 2], Nil), Nil).unwrap() == int_tuple![1, 2]);
        assert!(int_tuple![1, 2].unwrap() == int_tuple![1, 2]);
    }

    // Append / Prepend
    const {
        assert!(Nil.append(Nil) == Nil);
        assert!(Nil.append(int_tuple![1, 2]) == int_tuple![1, 2]);
        assert!(int_tuple![1, 2].append(Nil) == int_tuple![1, 2]);
        assert!(int_tuple![1, 2].append(int_tuple![3, 4]) == int_tuple![1, 2, 3, 4]);
        assert!(int_tuple![(1, 2)].append(int_tuple![3, (4, 5)]) == int_tuple![(1, 2), 3, (4, 5)]);
        assert!(int_tuple![1i32].append(int_tuple![Int::<2>]) == int_tuple![1i32, Int::<2>]);

        assert!(Nil.prepend(Int::<1>) == tuple!(Int::<1>));
        assert!(int_tuple![2, 3].prepend(1) == int_tuple![1, 2, 3]);
        assert!(int_tuple![2, 3].prepend(int_tuple![0, 1]) == int_tuple![(0, 1), 2, 3]);
        assert!(
            int_tuple![Int::<2>, Int::<3>].prepend(Int::<1>)
                == int_tuple![Int::<1>, Int::<2>, Int::<3>]
        );
    }

    // Flatten / Unflatten / unflatten_like / RepeatLike (including deep left/right trees & PyCuTe round-trips)
    const FLAT: TupleContainer<Int<1>, TupleContainer<int, Nil>> =
        int_tuple![(Int::<1>, (2i32))].flatten();
    assert_eq!(FLAT, int_tuple![Int::<1>, 2i32]);
    assert_eq!(Nil.flatten(), Nil);
    assert_eq!(42i32.flatten(), int_tuple![42i32]);
    assert_eq!(Int::<5>.flatten(), int_tuple![Int::<5>]);
    assert_eq!(
        int_tuple![((1, 2), 3), (4, (5, 6))].flatten(),
        int_tuple![1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        int_tuple![(((((1, 2), 3), 4), 5), 6)].flatten(),
        int_tuple![1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        int_tuple![1, (2, (3, (4, (5, 6))))].flatten(),
        int_tuple![1, 2, 3, 4, 5, 6]
    );

    let nested = tuple!(Int::<1>, (2i32, (Int::<3>, 4i32)), (), Int::<5>);
    let flat = nested.flatten();
    assert_eq!(flat, tuple!(Int::<1>, 2i32, Int::<3>, 4i32, Int::<5>));
    assert_eq!(unflatten_like(&flat, &nested), nested);

    type GuideProfile = <TupleContainer<int, TupleContainer<Nil, Nil>> as Item>::Profile;
    const UNFLATTENED: TupleContainer<Int<7>, TupleContainer<Nil, Nil>> =
        unflatten::<GuideProfile, _>(&int_tuple![Int::<7>]);
    assert_eq!(UNFLATTENED, tuple![Int::<7>, Nil]);

    let guide = tuple![10i32, (20i32, Nil), 30i32];
    assert_eq!(
        unflatten_like(&int_tuple![1i32, 2i32, 3i32], &guide),
        tuple![1i32, (2i32, Nil), 3i32]
    );
    assert_eq!(unflatten_like(&7i32.flatten(), &7i32), 7i32);
    assert_eq!(unflatten_like(&Nil, &Nil), Nil);
    assert_eq!(unflatten_like(&Nil, &tuple!((), (()))), tuple!((), (())));

    let r1 = int_tuple![3, 4];
    let r2 = int_tuple![(1, 2), 3];
    let r3 = int_tuple![(1, (2, 3)), 4];
    assert_eq!(unflatten_like(&r1.flatten(), &r1), r1);
    assert_eq!(unflatten_like(&r2.flatten(), &r2), r2);
    assert_eq!(unflatten_like(&r3.flatten(), &r3), r3);

    const {
        assert!(
            repeat_like(Int::<1>, &tuple![10i32, (20i32, Nil)])
                == tuple![Int::<1>, (Int::<1>, Nil)]
        );
    }
    assert_eq!(
        repeat_like(Int::<0>, &nested),
        tuple!(Int::<0>, (Int::<0>, (Int::<0>, Int::<0>)), (), Int::<0>)
    );
    assert_eq!(repeat_like(0, &r3), int_tuple![(0, (0, 0)), 0]);
    assert_eq!(repeat_like(7i32, &Int::<1>), 7i32);
    assert_eq!(repeat_like(7i32, &Nil), Nil);
}

#[test]
fn test_restructure_replace_and_lift_round_trips() {
    // ReplaceAt / Replace (const and runtime)
    const R_SCALAR: TupleContainer<Int<9>, Nil> = replace::<mode![0], _, _>(&Int::<5>, Int::<9>);
    const R_EMPTY: Int<9> = replace::<mode![], _, _>(&Int::<5>, Int::<9>);
    assert_eq!(R_SCALAR, tuple![Int::<9>]);
    assert_eq!(R_EMPTY, Int::<9>);

    let orig = tuple!((1i32, (2i32, 3i32)), 4i32, (5i32, 6i32));
    assert_eq!(
        ReplaceAt::<_, 1>::replace_at(&orig, 44i32),
        tuple!((1i32, (2i32, 3i32)), 44i32, (5i32, 6i32))
    );
    assert_eq!(
        ReplaceAt::<_, -1>::replace_at(&orig, 66i32),
        tuple!((1i32, (2i32, 3i32)), 4i32, 66i32)
    );
    assert_eq!(
        replace::<mode![1, -1], _, _>(&int_tuple![10, (20, 30), 40], Int::<99>),
        tuple![10, (20, Int::<99>), 40]
    );
    assert_eq!(
        replace::<mode![1], _, _>(&int_tuple![1, 2, 3], 42),
        int_tuple![1, 42, 3]
    );
    assert_eq!(
        replace::<mode![-1], _, _>(&int_tuple![1, 2, 3], 42),
        int_tuple![1, 2, 42]
    );
    assert_eq!(
        replace::<mode![-3], _, _>(&int_tuple![1, 2, 3], 42),
        int_tuple![42, 2, 3]
    );
    assert_eq!(
        replace::<mode![0, 2], _, _>(&int_tuple![(1, 2, 3), 4], 42),
        int_tuple![(1, 2, 42), 4]
    );
    assert_eq!(
        replace::<mode![0, -1], _, _>(&int_tuple![(1, 2, 3), 4], 42),
        int_tuple![(1, 2, 42), 4]
    );
    assert_eq!(replace::<mode![-1], _, _>(&7i32, 42i32), int_tuple![42]);
    assert_eq!(replace::<mode![], _, _>(&int_tuple![1, 2], 42), 42);

    // Replace + walk_mode! round-trips (PyCuTe TestGetLift)
    assert_eq!(walk_mode!(replace::<mode![], _, _>(&orig, 99)), 99);
    assert_eq!(walk_mode!(replace::<mode![1], _, _>(&orig, 99), 1), 99);
    assert_eq!(
        walk_mode!(replace::<mode![0, 1, 0], _, _>(&orig, 99), 0, 1, 0),
        99
    );
    assert_eq!(
        walk_mode!(replace::<mode![2, 1], _, _>(&orig, 99), 2, 1),
        99
    );
    assert_eq!(walk_mode!(replace::<mode![-1], _, _>(&orig, 99), -1), 99);
    assert_eq!(
        walk_mode!(replace::<mode![-3, -1, -2], _, _>(&orig, 99), -3, -1, -2),
        99
    );
    assert_eq!(
        walk_mode!(replace::<mode![-1, 0], _, _>(&orig, 99), -1, 0),
        99
    );

    // LIFT_BEFORE / LIFT_AFTER boundaries (including int::MIN)
    const {
        assert!(LIFT_BEFORE::<0> == 0);
        assert!(LIFT_BEFORE::<3> == 3);
        assert!(LIFT_BEFORE::<-1> == 0);
        assert!(LIFT_BEFORE::<I32_MIN> == 0);

        assert!(LIFT_AFTER::<0> == 0);
        assert!(LIFT_AFTER::<3> == 0);
        assert!(LIFT_AFTER::<-1> == 0);
        assert!(LIFT_AFTER::<-3> == 2);
        assert!(LIFT_AFTER::<I32_MIN> == I32_MAX);
    }

    // Lift structure & walk_mode! round-trips (const and runtime, PyCuTe TestGetLift)
    const {
        assert!(lift::<mode![], _, _, _>(&Int::<7>, &MakeTuple, Int::<1>) == Int::<7>);
        assert!(
            lift::<mode![1, 0], _, _, _>(&Int::<7>, &MakeTuple, Int::<1>)
                == tuple![Int::<1>, (Int::<7>)]
        );
        assert!(
            lift::<mode![-2, -2], _, _, _>(&Int::<7>, &MakeTuple, Int::<1>)
                == tuple![(Int::<7>, Int::<1>), Int::<1>]
        );
        let lifted_pos = lift::<mode!(0, 2, 1), _, _, _>(&Int::<42>, &MakeTuple, Int::<0>);
        assert!(walk_mode!(lifted_pos, 0, 2, 1) == Int::<42>);
    }

    let deep = int_tuple![(0, 0, (0, 0, 0, 42))];
    assert_eq!(walk_mode!(deep, 0, 2, 3), 42);
    assert_eq!(walk_mode!(7i32), 7);

    let x = 99i32;
    assert_eq!(walk_mode!(lift::<mode![], _, _, _>(&x, &MakeTuple, 0)), x);
    assert_eq!(
        walk_mode!(lift::<mode![0], _, _, _>(&x, &MakeTuple, 0), 0),
        x
    );
    assert_eq!(
        walk_mode!(lift::<mode![1], _, _, _>(&x, &MakeTuple, 0), 1),
        x
    );
    assert_eq!(
        walk_mode!(lift::<mode![0, 2, 3], _, _, _>(&x, &MakeTuple, 0), 0, 2, 3),
        x
    );
    assert_eq!(
        walk_mode!(lift::<mode![1, 1, 0], _, _, _>(&x, &MakeTuple, 0), 1, 1, 0),
        x
    );
    assert_eq!(
        walk_mode!(lift::<mode![-1], _, _, _>(&x, &MakeTuple, 0), -1),
        x
    );
    assert_eq!(
        walk_mode!(lift::<mode![-2], _, _, _>(&x, &MakeTuple, 0), -2),
        x
    );
    assert_eq!(
        walk_mode!(
            lift::<mode![1, -3, 0], _, _, _>(&x, &MakeTuple, 0),
            1,
            -3,
            0
        ),
        x
    );

    assert_eq!(
        lift::<mode![-1], _, _, _>(&42i32, &MakeTuple, 0),
        tuple![42]
    );
    assert_eq!(
        lift::<mode![-2], _, _, _>(&42i32, &MakeTuple, Nil),
        tuple![42, Nil]
    );
    assert_eq!(
        lift::<mode![1, -2], _, _, _>(&42i32, &MakeTuple, 0),
        int_tuple![0, (42, 0)]
    );
    assert_eq!(
        lift::<mode![0, 2, 3], _, _, _>(&42i32, &MakeTuple, 0),
        int_tuple![(0, 0, (0, 0, 0, 42))]
    );
    assert_eq!(
        lift::<mode![1, 0], _, _, _>(&42i32, &MakeTuple, Nil),
        tuple![Nil, (42)]
    );
    assert_eq!(
        lift::<mode![2], _, _, _>(&42i32, &MakeTuple, Nil),
        tuple![Nil, Nil, 42]
    );
}

#[test]
fn test_product_operations() {
    const {
        let one: Int<1> = tuple_product(&Nil);
        assert!(one == Int::<1>);

        let s: Int<24> = tuple_product(&tuple!((Int::<2>, Int::<3>), Int::<4>));
        assert!(s == Int::<24>);

        // Absorbing static zero with dynamic operands
        let zero_l: Int<0> = tuple_product(&tuple!(Int::<0>, (I32_MIN, 1i32)));
        let zero_r: Int<0> = tuple_product(&tuple!((I32_MIN, 1i32), Int::<0>));
        assert!(zero_l == Int::<0>);
        assert!(zero_r == Int::<0>);

        // product_each
        assert!(
            product_each(&tuple!(
                (Int::<2>, (Int::<3>, Int::<4>)),
                (Int::<5>, Int::<6>),
                Int::<7>
            )) == tuple!(Int::<24>, Int::<30>, Int::<7>)
        );
        assert!(
            product_each(&int_tuple![Int::<2>, (Int::<3>, Int::<4>)])
                == int_tuple![Int::<2>, Int::<12>]
        );
        assert!(product_each(&Nil) == Nil);
        assert!(product_each(&Int::<9>) == Int::<9>);

        // product_like
        let shape = tuple!((Int::<2>, (Int::<3>, Int::<4>)), Int::<5>);
        assert!(
            product_like(&shape, &tuple!((0, 0), 0)) == tuple!((Int::<2>, Int::<12>), Int::<5>)
        );
        assert!(product_like(&shape, &tuple!(0, 0)) == product_each(&shape));
        assert!(product_like(&shape, &0) == tuple_product(&shape));
        assert!(
            product_like(
                &int_tuple![(Int::<2>, Int::<3>), (Int::<4>, Int::<5>)],
                &int_tuple![Int::<0>, Int::<0>]
            ) == int_tuple![Int::<6>, Int::<20>]
        );
    }

    // PyCuTe TestHTuple product / product_each / product_like parity
    assert_eq!(tuple_product(&2i32), 2);
    assert_eq!(tuple_product(&int_tuple![3, 2]), 6);
    assert_eq!(tuple_product(&int_tuple![(2, 3), 4]), 24);
    assert_eq!(tuple_product(&int_tuple![2i32, (3i32, 5i32)]), 30);

    assert_eq!(product_each(&7i32), 7);
    assert_eq!(product_each(&int_tuple![2, 3]), int_tuple![2, 3]);
    assert_eq!(product_each(&int_tuple![2, 3, 4]), int_tuple![2, 3, 4]);
    assert_eq!(product_each(&int_tuple![(2, 3), 4]), int_tuple![6, 4]);
    let complex = int_tuple![(2, (3, 4)), (5, 6), 7];
    let complex_each = product_each(&complex);
    assert_eq!(complex_each, int_tuple![24, 30, 7]);
    assert_eq!(tuple_product(&complex_each), tuple_product(&complex));

    let p_a = int_tuple![(2, 4), 12];
    assert_eq!(product_like(&p_a, &int_tuple![0, 0]), int_tuple![8, 12]);
    assert_eq!(
        product_like(&int_tuple![(2, (3, 4)), 5], &int_tuple![(0, 0), 0]),
        int_tuple![(2, 12), 5]
    );
    assert_eq!(product_like(&p_a, &0i32), tuple_product(&p_a));
    let p_b = int_tuple![(2, 4), 12, (5, 6)];
    assert_eq!(product_like(&p_b, &int_tuple![0, 0, 0]), product_each(&p_b));
    assert_eq!(
        product_like(&int_tuple![8, 12], &int_tuple![0, 0]),
        int_tuple![8, 12]
    );

    // Dynamic mixed product and overflow panic
    let dyn_t = tuple!((Int::<2>, 3i32), (4i32, Int::<5>));
    assert_eq!(tuple_product(&dyn_t), 120i32);
    assert_eq!(product_each(&dyn_t), tuple!(6i32, 20i32));
    assert_eq!(
        product_like(&dyn_t, &tuple!(0, (0, 0))),
        tuple!(6i32, (4i32, Int::<5>))
    );

    let overflow_t = tuple!(int::MAX, 2i32);
    assert!(catch_unwind(|| tuple_product(&overflow_t)).is_err());
}

#[test]
fn test_transform_leaf_preserves_nesting_and_visits_depth_first() {
    assert_eq!(Int::<3>.transform(&Offset(2)), 5i32);
    assert_eq!(Nil.transform(&Offset(2)), Nil);
    assert_eq!(
        int_tuple![Int::<1>, (2i32, Int::<3>), 4i32].transform(&Offset(10)),
        int_tuple![11i32, (12i32, 13i32), 14i32],
    );
    type Mapped = TupleContainer<int, TupleContainer<TupleContainer<int, Nil>, Nil>>;
    const MAPPED: Mapped = int_tuple![Int::<1>, (2i32)].transform(&Offset(3));
    assert_eq!(MAPPED, int_tuple![4i32, (5i32)]);

    #[derive(Clone, Copy)]
    struct Record<'a>(&'a RefCell<Vec<int>>);
    impl<I: Integer> TransformLeafFunctor<I> for Record<'_> {
        type Output = RecordedLeaf;
        fn apply(&self, leaf: I) -> Self::Output {
            let mut seen = self.0.borrow_mut();
            let index = seen.len();
            seen.push(leaf.value());
            RecordedLeaf(leaf.value(), index)
        }
    }

    let seen = RefCell::new(Vec::new());
    let mapped = int_tuple![1, (Int::<2>, (3i32)), 4].transform(&Record(&seen));
    assert_eq!(*seen.borrow(), vec![1, 2, 3, 4]);
    assert_eq!(
        mapped,
        tuple![
            RecordedLeaf(1, 0),
            (RecordedLeaf(2, 1), (RecordedLeaf(3, 2))),
            RecordedLeaf(4, 3)
        ]
    );
    assert_eq!(Nil.transform(&Record(&seen)), Nil);
    assert_eq!(*seen.borrow(), vec![1, 2, 3, 4]);
}

#[test]
fn test_multi_tree_traversal_zip_transform_apply_fold_all_and_any() {
    // 1. TransformApplyLeaf with custom const reducer (sum at every node)
    #[derive(Clone, Copy)]
    struct IdentityLeaf;
    const impl<T> ApplyLeafFunctor<T> for IdentityLeaf {
        type Output = T;
        fn apply(&self, x: T) -> T {
            x
        }
    }

    const trait SumChain: Copy {
        fn sum_chain(self) -> int;
    }
    const impl SumChain for Nil {
        fn sum_chain(self) -> int {
            0
        }
    }
    const impl<H: [const] Integer, T: [const] SumChain> SumChain for Til<H, T> {
        fn sum_chain(self) -> int {
            add(self.0.value(), self.1.sum_chain())
        }
    }

    #[derive(Clone, Copy)]
    struct SumNode;
    const impl<C: [const] SumChain> MakeNodeFunctor<C> for SumNode {
        type Output = int;
        fn make(&self, children: C) -> int {
            children.sum_chain()
        }
    }

    const SUM: int = transform_apply_leaf(
        &tuple!((Int::<1>, Int::<2>), (Int::<3>, Int::<4>)),
        &SumNode,
        &IdentityLeaf,
        (),
    );
    assert_eq!(SUM, 10);
    assert_eq!(transform_apply_leaf(&Nil, &SumNode, &IdentityLeaf, ()), 0);

    // 2. ZipTransformLeaf & root-only MakeNodeFunctor invocation check (non-Item output)
    #[derive(Clone, Copy)]
    struct AddPair;
    const impl<A, B> ApplyLeafFunctor<(A, B)> for AddPair
    where
        A: [const] Integer + [const] ArithAdd<A, B>,
        B: [const] Integer,
    {
        type Output = Sum<A, B>;
        fn apply(&self, (a, b): (A, B)) -> Self::Output {
            add(a, b)
        }
    }

    const {
        let zip_sum = zip_transform_leaf(
            &tuple!(Int::<1>, (2i32, Int::<3>)),
            &AddPair,
            tuple!(Int::<10>, (20i32, Int::<30>)),
        );
        assert!(zip_sum == tuple!(Int::<11>, (22i32, Int::<33>)));
        assert!(
            zip_transform_leaf(
                &int_tuple![Int::<1>, (Int::<2>)],
                &AddPair,
                int_tuple![Int::<4>, (Int::<9>)]
            ) == int_tuple![Int::<5>, (Int::<11>)]
        );
    }

    struct BoxedSum(int); // deliberately does NOT implement Item
    struct FlatMake<'a>(&'a Cell<usize>);
    impl MakeNodeFunctor<Til<int, Til<int, Nil>>> for FlatMake<'_> {
        type Output = BoxedSum;
        fn make(&self, children: Til<int, Til<int, Nil>>) -> BoxedSum {
            self.0.set(self.0.get() + 1);
            BoxedSum(add(children.0, children.1.0))
        }
    }
    impl MakeNodeFunctor<Nil> for FlatMake<'_> {
        type Output = BoxedSum;
        fn make(&self, _: Nil) -> BoxedSum {
            self.0.set(self.0.get() + 1);
            BoxedSum(0)
        }
    }

    let root_calls = Cell::new(0);
    let out = transform_apply_leaf(
        &int_tuple![1i32, 2i32],
        &FlatMake(&root_calls),
        &AddPair,
        int_tuple![10i32, 20i32],
    );
    assert_eq!(out.0, 33);
    assert_eq!(root_calls.get(), 1);

    let nil_calls = Cell::new(0);
    let nil_out = transform_apply_leaf(&Nil, &FlatMake(&nil_calls), &AddPair, Nil);
    assert_eq!(nil_out.0, 0);
    assert_eq!(nil_calls.get(), 1);

    // 3. ZipLeaves: 0, 1, 5 peers, weakly congruent, nested Nil, deep left/right nesting
    let single_zip = int_tuple![Int::<1>, (2i32, Nil)].zip_leaves(());
    assert_eq!(single_zip, tuple![ZipLeaf((Int::<1>,)), ZipLeaf((2i32,))]);
    assert_eq!(single_zip.get::<0>().into_inner(), (Int::<1>,));
    assert_eq!(*single_zip.get::<1>(), (2i32,));
    assert!(!single_zip.is_static());
    assert!(ZipLeaf((Int::<1>, Int::<2>)).is_static());
    assert!(ZipLeaf(()).is_static());
    assert_eq!(
        (
            ZipLeaf((1i32,)).rank(),
            ZipLeaf((1i32,)).depth(),
            ZipLeaf((1i32,)).length()
        ),
        (1, 0, 1)
    );

    let left = int_tuple![Int::<1>, (2i32, 3i32)];
    let right = int_tuple![4i32, (Int::<5>, 6i32)];
    let zipped = left.zip_leaves(right);
    type Zipped = TupleContainer<
        ZipLeaf<(Int<1>, int)>,
        TupleContainer<ZipLeaf<(int, Int<5>)>, TupleContainer<ZipLeaf<(int, int)>, Nil>>,
    >;
    let _: Zipped = zipped;
    assert_eq!(
        zipped,
        tuple![
            ZipLeaf((Int::<1>, 4i32)),
            ZipLeaf((2i32, Int::<5>)),
            ZipLeaf((3i32, 6i32)),
        ]
    );
    assert_eq!(zip_leaves(&Nil, Nil), Nil);
    assert_eq!(1i32.zip_leaves(Nil), tuple![ZipLeaf((1i32, Nil))]);
    assert_eq!(
        zip_leaves(&Int::<1>, tuple!(2i32, 3i32)),
        tuple!(ZipLeaf((Int::<1>, tuple!(2i32, 3i32))))
    );
    assert_eq!(
        int_tuple![1, 2, Nil].zip_leaves(int_tuple![3, (4, 5), Nil]),
        tuple![ZipLeaf((1, 3)), ZipLeaf((2, int_tuple![4, 5]))]
    );

    const C_ZIP: TupleContainer<ZipLeaf<(Int<1>, Int<2>)>, Nil> = zip_leaves(&Int::<1>, Int::<2>);
    assert!(C_ZIP.is_static());
    assert_eq!(C_ZIP.0.into_inner(), (Int::<1>, Int::<2>));

    // 6-tree lockstep synchronization (primary + 5 peers)
    let p0 = int_tuple![1, (2)];
    let p1 = int_tuple![10, (20)];
    let p2 = int_tuple![100, (200)];
    let p3 = int_tuple![1000, (2000)];
    let p4 = int_tuple![10000, (20000)];
    let p5 = int_tuple![100000, (200000)];
    assert_eq!(
        p0.zip_leaves((p1, p2, p3, p4, p5)),
        tuple![
            ZipLeaf((1, 10, 100, 1000, 10000, 100000)),
            ZipLeaf((2, 20, 200, 2000, 20000, 200000)),
        ]
    );

    // Deep left-heavy and right-heavy trees
    let left_a = int_tuple![(((((1, 2), 3), 4), 5), 6)];
    let left_b = int_tuple![(((((10, 20), 30), 40), 50), 60)];
    let right_a = int_tuple![1, (2, (3, (4, (5, 6))))];
    let right_b = int_tuple![10, (20, (30, (40, (50, 60))))];
    let expected_deep_zip = tuple![
        ZipLeaf((1, 10)),
        ZipLeaf((2, 20)),
        ZipLeaf((3, 30)),
        ZipLeaf((4, 40)),
        ZipLeaf((5, 50)),
        ZipLeaf((6, 60)),
    ];
    assert_eq!(left_a.zip_leaves(left_b), expected_deep_zip);
    assert_eq!(right_a.zip_leaves(right_b), expected_deep_zip);

    // 4. FoldLeaf: unary evolving accumulator and binary WeightedSum
    #[derive(Clone, Copy)]
    struct FoldAdd;
    const impl<Acc, X> FoldLeafFunctor<Acc, X> for FoldAdd
    where
        Acc: [const] Integer + [const] ArithAdd<Acc, X>,
        X: [const] Integer,
    {
        type Output = Sum<Acc, X>;
        fn apply(&self, acc: Acc, x: X) -> Self::Output {
            add(acc, x)
        }
    }

    #[derive(Clone, Copy)]
    struct WeightedSum;
    const impl<Acc, C, W> FoldLeafFunctor<Acc, (C, W)> for WeightedSum
    where
        C: [const] Integer,
        W: [const] Integer + [const] ArithMul<W, C>,
        Product<W, C>: [const] Integer,
        Acc: [const] Integer + [const] ArithAdd<Acc, Product<W, C>>,
    {
        type Output = Sum<Acc, Product<W, C>>;
        fn apply(&self, acc: Acc, (c, w): (C, W)) -> Self::Output {
            add(acc, mul(w, c))
        }
    }

    const STATIC_FOLD: Int<6> = fold_leaf(
        &tuple!(Int::<1>, (Int::<2>, Int::<3>)),
        Int::<0>,
        &FoldAdd,
        (),
    );
    const WEIGHTED_FOLD: Int<14> = fold_leaf(
        &int_tuple![Int::<2>, (Int::<3>)],
        Int::<0>,
        &WeightedSum,
        int_tuple![Int::<1>, (Int::<4>)],
    );
    assert_eq!(STATIC_FOLD, Int::<6>);
    assert_eq!(WEIGHTED_FOLD, Int::<14>);
    let mixed_fold: int = fold_leaf(&tuple!(Int::<1>, (2i32, Int::<3>)), Int::<0>, &FoldAdd, ());
    assert_eq!(mixed_fold, 6);
    assert_eq!(fold_leaf(&Nil, Int::<9>, &FoldAdd, ()), Int::<9>);

    // 5. AllLeaves and AnyLeaf (unary & binary bounds checks)
    #[derive(Clone, Copy)]
    struct IsPositive;
    const impl<I: [const] Integer> LeafPredicate<I> for IsPositive {
        fn test(&self, x: I) -> bool {
            x.value() > 0
        }
    }

    #[derive(Clone, Copy)]
    struct InBounds;
    const impl<C: [const] Integer, B: [const] Integer> LeafPredicate<(C, B)> for InBounds {
        fn test(&self, (c, b): (C, B)) -> bool {
            ge(c, Int::<0>) && lt(c, b)
        }
    }

    const {
        assert!(all_leaves(&Nil, &IsPositive, ()));
        assert!(!any_leaf(&Nil, &IsPositive, ()));
        assert!(all_leaves(
            &tuple!(Int::<1>, (Int::<2>, Int::<3>)),
            &IsPositive,
            ()
        ));
        assert!(!all_leaves(
            &tuple!(Int::<1>, (Int::<0>, Int::<3>)),
            &IsPositive,
            ()
        ));
        assert!(any_leaf(
            &tuple!(Int::<-1>, (Int::<2>, Int::<-3>)),
            &IsPositive,
            ()
        ));
        assert!(!any_leaf(
            &tuple!(Int::<-1>, (Int::<0>, Int::<-3>)),
            &IsPositive,
            ()
        ));

        assert!(all_leaves(
            &int_tuple![Int::<1>, (Int::<2>, Int::<3>)],
            &InBounds,
            int_tuple![Int::<2>, (Int::<5>, Int::<4>)]
        ));
        assert!(!all_leaves(
            &int_tuple![Int::<1>, (Int::<5>, Int::<3>)],
            &InBounds,
            int_tuple![Int::<2>, (Int::<5>, Int::<4>)]
        ));
        assert!(any_leaf(
            &int_tuple![Int::<1>, (Int::<0>, Int::<3>)],
            &InBounds,
            int_tuple![Int::<1>, (Int::<2>, Int::<1>)]
        ));
        assert!(Nil.all_leaves(&InBounds, Nil));
        assert!(!Nil.any_leaf(&InBounds, Nil));
    }
}
