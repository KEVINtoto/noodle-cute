use noodle_cute::algebra::{
    CoalesceMode, CoalescedLayout, Complement, ComplementMode, Grouping, InverseSide, ProductKind,
    TypedComplement, TypedCompose, TypedDivide, TypedDivided, TypedInverse, TypedInvert,
    TypedProduct, TypedProductLayout,
};
use noodle_cute::layout::{Layout, LayoutMap, Shape, StrideOf};
use noodle_cute::tuple::Item;
use noodle_cute::{DynInt, Int, StaticValue, shape, stride};

fn supported<L, T>()
where
    L: TypedCompose<T>
        + Complement<Int<16>>
        + TypedDivide<Int<2>>
        + TypedProduct<Int<2>>
        + TypedInvert,
{
}

fn shape_item<S: Shape + Item>() {}

fn same_profile<S: Shape, D: StrideOf<S>>(shape: S, stride: D) -> Layout<S, D> {
    Layout::new(shape, stride).unwrap()
}

fn main() {
    type A = Layout<Int<8>, Int<1>>;
    type T = Layout<Int<2>, Int<1>>;
    macro_rules! divides {
        ($($grouping:path),*) => { $(
            let _: TypedDivided<A, T, { $grouping }> = StaticValue::VALUE;
        )* };
    }
    macro_rules! products {
        ($($kind:path),*) => { $(
            let _: TypedProductLayout<A, T, { $kind }> = StaticValue::VALUE;
        )* };
    }
    divides!(
        Grouping::Logical,
        Grouping::Zipped,
        Grouping::Tiled,
        Grouping::Flat
    );
    products!(
        ProductKind::Logical,
        ProductKind::Zipped,
        ProductKind::Tiled,
        ProductKind::Flat,
        ProductKind::Blocked,
        ProductKind::Raked
    );
    let _: CoalescedLayout<Int<8>, Int<1>, { CoalesceMode::Standard }> = StaticValue::VALUE;
    let _: CoalescedLayout<Int<8>, Int<1>, { CoalesceMode::Extended }> = StaticValue::VALUE;
    let _: TypedComplement<A, Int<16>, { ComplementMode::Standard }> = StaticValue::VALUE;
    let _: TypedComplement<A, noodle_cute::Nil, { ComplementMode::Extended }> = StaticValue::VALUE;
    let _: TypedInverse<A, { InverseSide::Right }> = StaticValue::VALUE;
    let _: TypedInverse<A, { InverseSide::Left }> = StaticValue::VALUE;
    supported::<Layout<Int<8>, Int<1>>, Layout<Int<2>, Int<1>>>();
    shape_item::<Int<0>>();
    let _ = same_profile(shape![Int::<2>, DynInt(3)], stride![Int::<1>, Int::<2>]);
    let constant: Layout<Int<8>, Int<1>> = StaticValue::VALUE;
    let _ = constant.logical_divide(Int::<2>).unwrap();
    let _ = constant.logical_product(Int::<2>).unwrap();
    let _ = constant.composition(Layout::new(Int::<2>, Int::<1>).unwrap());
    let _ = Layout::new(shape![Int::<2>, DynInt(3)], stride![Int::<1>, Int::<2>])
        .unwrap()
        .size();
}
