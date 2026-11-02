use noodle_cute::{Int, IntTuple, Nil, StaticValue, algebra::TypedComposition, layout::Layout};

type Pair<const A: isize, const B: isize> = IntTuple<Int<A>, IntTuple<Int<B>, Nil>>;

const C: TypedComposition<Layout<Pair<4, 6>, Pair<6, 1>>, Layout<Int<8>, Int<5>>> =
    StaticValue::VALUE;

fn main() {
    let _ = C;
}
