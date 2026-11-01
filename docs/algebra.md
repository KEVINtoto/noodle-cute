# Implementing the layout algebra: `composition`, `complement`, `inverse`, `divide`, and `product`

The first three parts of this document explain the three core algorithms of the layout algebra. Together they form the foundation of every higher-level operation, such as divide, product, partitioning, and thread assignment. The last two parts explain divide and product, which are built directly on top of them:

- **Part I · Layout composition `composition`**: how the composed view is constructed, how it decides between evaluating through the CuTe normalized layout and evaluating pointwise, and under what conditions the `Composition::normalized` field is populated. Corresponds to [`src/algebra/composition.rs`](../src/algebra/composition.rs); fully static results (`TypedComposition` read through `StaticValue`) are covered in 7.1, and the entry points for each kind of input in 7.3.
- **Part II · Complement `complement`**: how to find the "gaps" a layout leaves behind, and how `complement_shape` differs from `complement_extended`. Corresponds to [`src/algebra/complement.rs`](../src/algebra/complement.rs).
- **Part III · Inverse `inverse`**: how to recover logical indices from addresses, why `right_inverse` and `left_inverse` give completely different answers for "holes", and how they divide the work with the enumerated inverses `dense_*_inverse`. Corresponds to [`src/algebra/inverse.rs`](../src/algebra/inverse.rs).
- **Part IV · Divide `divide`**: how a layout is split into "intra-tile coordinate × tile index", why the four `Grouping`s are only different parenthesizations of the same map, and the two routes of a single tiler and a per-mode tiler (`*_divide_modes`). Corresponds to [`src/algebra/divide.rs`](../src/algebra/divide.rs) and `TileRest` in [`src/algebra.rs`](../src/algebra.rs).
- **Part V · Product `product`**: how a block is repeated following the pattern of a tiler, why `complement_extended` without a cotarget is used, and how `blocked_product` differs from `raked_product`. Corresponds to [`src/algebra/product.rs`](../src/algebra/product.rs).

The relationship between `composition` and `complement` can be summarized in one formula, which is also the definition of `logical_divide` in Part IV (when A is an affine layout):

```text
logical_divide(A, B) = composition(A, (B, complement(B, shape(coalesce(A)))))
```

For the mathematical background see [Mathematical model](mathematical-model.md); for the overall engineering structure see [Engineering implementation](engineering-implementation.md).

This document uses the `Shape:Stride` layout notation throughout; for example, `8:3` denotes a one-dimensional layout of length 8 and stride 3.

## Part I · Layout composition `composition`

In this part, `outer` is written A and `inner` is written B.

### 1. Overview

`composition(A, B)` returns a lazy view `R` such that

```text
R(c) = A(B(c))
```

Construction computes no mapping; it only does two things:

1. **Codomain check**: whether the outputs of B can be evaluated as inputs of A.
2. **Normalization decision**: when both sides are affine layouts, compute the normalized layout with CuTe's `composition` algorithm and cache it in the `normalized` field; if CuTe would fail an assertion here (`NotDivisible`), nothing is cached.

At evaluation time, the cached layout is used if present; otherwise `A(B(c))` is computed pointwise.

### 2. Construction phase

#### 2.1 Codomain check and normalization decision

```rust
let normalized = match (outer.as_affine()?, inner.as_affine()?) {
    (Some(a), Some(b)) => match compose_grouped(a.flat(), &b) {
        Ok(layout) => Some(layout),
        Err(Error::NotDivisible) => None,
        Err(e) => return Err(e),
    },
    _ => None,
};

// Only pointwise evaluation feeds `inner` values to `outer` as logical indices.
if normalized.is_none()
    && let Some((min, max)) = inner.address_range()?
{
    if min < 0 {
        return Err(Error::CoordinateOutOfBounds);
    }
    outer.index_extended(max)?;
}
```

- The normalized layout follows CuTe `composition_impl`, which handles negative RHS strides through `signum`. Neither the eager `composition_layout` nor a normalized lazy composition restricts the sign of `inner` addresses.
- Values of B may exceed `A.size()`, as in CuTe. They are evaluated on A's **extended domain**: every leaf except the last is reduced modulo its extent, and the last leaf absorbs the remaining quotient (see `LayoutMap::index_extended`).
- Only the pointwise fallback (`normalized.is_none()`) uses B's values as logical indices of A, so only it requires `min >= 0` and probes `outer.index_extended(max)`:
  - When normalization succeeds, `compose_grouped` already validated strides and addresses while building its `DynLayout`. Under weak divisibility the normalized layout restarts at the next mode, so probing `outer.index_extended(max)` could report a spurious overflow caused by carrying across modes.
  - When A has no affine form (`as_affine()` returns `None`, e.g. swizzle-like maps), the default `index_extended` rejects indices outside the domain, so `max >= A.size()` returns `CoordinateOutOfBounds`. If `max` is inside the domain, so is every value in `[0, max]`.
  - When both sides are affine but `NotDivisible` forces pointwise evaluation, the probe checks the multiply-add at the largest index for overflow.
- An empty B has no address range, so the checks are skipped and an empty inner may be composed with an empty outer.

#### 2.2 Normalization outcomes

| `compose_grouped` result | Meaning | Handling |
|---|---|---|
| `Ok(layout)`, integral B | CuTe normalized result | Cached; each leaf of the result is a top-level mode, as in CuTe |
| `Ok(layout)`, tuple B | CuTe normalized result | Cached; B's top-level grouping is kept |
| `Err(NotDivisible)` | CuTe would fail an assertion | Pointwise evaluation is still mathematically defined, so fall back to pointwise |
| Other `Err` | Overflow, `CapacityExceeded`, etc. | Propagated upward |
| Either side non-affine | Corresponds to CuTe's `ComposedLayout` | Pointwise evaluation |

What is passed in is `a.flat()`: A's grouping is discarded, because A only receives linear indices and its hierarchy does not affect the result. B keeps its top-level grouping, because the shape of the result comes from B.

### 3. Core algorithm

#### 3.1 `compose_grouped`

1. Apply `coalesce_extended` to A, i.e. CuTe's `coalesce_x`. It differs from ordinary `coalesce` in that it keeps the stride of a trailing leaf with shape 1, because the extended domain extends along that stride.
2. When B has an integral shape, call `compose_leaf` directly on its single leaf and return the result as a flat layout.
3. When B is a tuple, use the right-distributive law `A ∘ (B0, B1, …) = (A∘B0, A∘B1, …)`: call `compose_leaf` on each leaf of B and record how many leaves each top-level group produces. This way the top-level grouping of the result matches B.

#### 3.2 `compose_leaf`

Composes one leaf `s:d` of B with the coalesced A by passing the stride through the modes of A in turn:

```text
rest_shape = s, rest_stride = d
for the first r-1 modes (cs, cd) of A:
    next_shape  = ceil(cs / |rest_stride|)    // how many steps to take within this mode
    next_stride = ceil(|rest_stride| / cs) * sign(rest_stride)   // stride converted to the next mode
    if next_shape == 1 (the stride spans the whole mode) or rest_shape == 1 (B is exhausted):
        only update rest_stride
    else:
        new_shape = min(next_shape, rest_shape)
        emit leaf new_shape : rest_stride * cd
        rest_shape /= new_shape; rest_stride = next_stride
if no leaf has been emitted yet, or rest_shape != 1:
    put the remaining rest_shape into the last mode of A (extended domain, no modulo)
```

Special cases and checks:

- `d == 0 || s == 0`: B does not read A; the result is `s:0` directly.
- A has a single mode: the result is `s : d*stride0`, and the extended domain extends naturally.
- **Stride divisibility condition**: when the stride is larger than the mode size but does not divide it, return `NotDivisible`.
- **Shape divisibility condition**: when `rest_shape` is not divisible by `new_shape`, return `NotDivisible`.
- **Weak divisibility behavior**: when `cs % |stride| != 0` and B has not yet finished within this mode (`new_shape < rest_shape`), `compose_leaf` does not report an error; following CuTe, it discards the remainder and restarts at the next mode. This is exactly where the CuTe result and pointwise evaluation diverge (see sections 5 and 6). The code no longer records a separate divergence flag; the normalized result is always cached.

### 4. Evaluation phase: `impl LayoutMap for Composition`

- `size` / `rank` / `extent`: all come from inner; the output shape is B's shape.
- `index(i)`: returns `normalized.index(i)` when cached, otherwise `outer.index_extended(inner.index(i)?)`. Here outer uses `index_extended` because values of B may exceed A's domain.
- `index_extended(i)` / `index_extended_of_coord(coord)`: same as `index`; when cached, use the cached layout, otherwise inner is also evaluated on the corresponding extended domain.
- `is_injective`: when cached, ask the cached layout directly. Otherwise, only when A can be proven injective by a cheap $O(\text{leaves}^2)$ affine stride-separation argument (`affine_injective == Some(true)`) and all values of B lie in A's logical domain, delegate directly to `B.is_injective()`; if A is non-affine, the stride-separation argument is inconclusive (to avoid an $O(\text{size}(A)^2)$ enumeration of a huge A), or B enters A's extended domain, fall back to `is_injective_by_enumeration(self)`, which enumerates the composed domain at $O(\text{size}(B)^2)$ cost.
- `address_range`: when cached, delegate directly to the cached `GroupedLayout` (exact endpoints in O(leaves)); without a cache, fall back to traversing `0..size` to find the minimum/maximum address.
- `as_affine`: returns the cached `normalized` directly. `None` means "can be evaluated pointwise, but has no affine form" (either side is non-affine, or `NotDivisible` was encountered).

### 5. The `normalized` field

```rust
// Cached CuTe normalized layout when both sides are affine and divisible.
pub(crate) normalized: Option<GroupedLayout>,
```

The core of this comment is "the normalized layout cached when …"; the first half is the condition:

> The normalized layout that is cached when [both sides are affine layouts] and [the CuTe divisibility condition holds (not `NotDivisible`)].

The field is `Some` only when both conditions hold.

- **Affine layout**: representable by a set of flat shapes/strides, i.e. `as_affine()` returns `Some`. Maps such as swizzle are not affine layouts.
- **CuTe normalized result**: the new layout `R` computed directly by `compose_grouped`.
- **Pointwise evaluation**: no new layout is produced; each evaluation first computes `B(i)` and then passes it to A.

Why cache: the normalized result matches CuTe's `Layout∘Layout`, and the padded `logical_divide` depends on it; within A's logical domain it usually coincides with pointwise evaluation and differs only under weak divisibility (see section 6), so the cached layout is authoritative.

Two points to note:

1. **`None` does not necessarily mean the composition is meaningless**. It means "either side is non-affine, or `NotDivisible` was encountered"; pointwise evaluation is still possible, there is just no affine form. `is_normalized()` returns `normalized.is_some()`, and only indicates whether evaluation uses the cached layout.
2. **The extended-domain behavior of the cached layout differs from pointwise evaluation**. R extends along the last leaf of each mode, whereas pointwise evaluation carries between A's leaves; composed evaluation follows R, consistent with CuTe.

### 6. Illustration: one A, three Bs

The examples below all use `A = (4,6):(6,1)`. It is a layout with 4 rows and 6 columns: logical index `j` corresponds to position `(j % 4, j / 4)`, and its address is `6·row + col`.

```text
logical index j                     address A(j)
   col0   1   2   3   4   5            col0   1   2   3   4   5
row0  0   4   8  12  16  20         row0  0   1   2   3   4   5
row1  1   5   9  13  17  21         row1  6   7   8   9  10  11
row2  2   6  10  14  18  22         row2 12  13  14  15  16  17
row3  3   7  11  15  19  23         row3 18  19  20  21  22  23
```

B selects cells of this grid in order. In the diagrams below, `[i]` marks the `i`-th selected cell.

#### 6.1 Divisible stride: the normalized result equals pointwise evaluation

Take `B = 8:2`, i.e. `j = 0, 2, 4, …, 14`.

```text
pointwise A(B(i))                   CuTe result R = (2,4):(12,1)
   col0   1   2   3   4   5            col0   1   2   3   4   5
row0 [0] [2] [4] [6]  .   .         row0 [0] [2] [4] [6]  .   .
row1  .   .   .   .   .   .         row1  .   .   .   .   .   .
row2 [1] [3] [5] [7]  .   .         row2 [1] [3] [5] [7]  .   .
row3  .   .   .   .   .   .         row3  .   .   .   .   .   .
```

The `compose_leaf` steps:

```text
mode 0 (cs=4, cd=6): stride=2, ceil(4/2)=2 steps, 4%2==0
    → leaf 2 : 2·6 = 12; rest_shape = 4, next-mode stride = ceil(2/4) = 1
last mode: → leaf 4 : 1·1 = 1
R = (2,4):(12,1), addresses 0,12,1,13,2,14,3,15
```

Stride 2 divides the column height 4, so exactly two cells are selected in each column before moving to the next column, and the two diagrams are identical. Here `normalized = Some(R)`, and evaluating through `R` gives the same addresses as pointwise evaluation.

#### 6.2 Non-divisible stride: the normalized result differs from pointwise evaluation

Take `B = 8:3`, i.e. `j = 0, 3, 6, 9, 12, 15, 18, 21`.

```text
pointwise A(B(i))                   CuTe result R = (2,4):(18,1)
   col0   1   2   3   4   5            col0   1   2   3   4   5
row0 [0]  .   .  [4]  .   .         row0 [0] [2] [4] [6]  .   .
row1  .   .  [3]  .   .  [7]        row1  .   .   .   .   .   .
row2  .  [2]  .   .  [6]  .         row2  .   .   .   .   .   .
row3 [1]  .   .  [5]  .   .         row3 [1] [3] [5] [7]  .   .
addresses: 0,18,13,8,3,21,16,11     addresses: 0,18,1,19,2,20,3,21
```

The `compose_leaf` steps:

```text
mode 0 (cs=4, cd=6): stride=3, ceil(4/3)=2 steps
    4%3≠0 and 2<8 → weak divisibility: discard the remainder, restart at the next mode
    → leaf 2 : 3·6 = 18; rest_shape = 4, next-mode stride = ceil(3/4) = 1 (remainder discarded)
last mode: → leaf 4 : 1·1 = 1
R = (2,4):(18,1)
```

- **Pointwise evaluation**: after selecting rows 0 and 3 of column 0, stride 3 steps past the bottom of the column and the remainder carries into the next column, so it lands on row 2 of column 1; the path zigzags.
- **CuTe**: only `ceil(4/3) = 2` cells are taken within a column; then the remainder is discarded and it restarts from the top of the next column, giving a regular 2×4 subgrid.

The two differ. The `R` cached by `composition` reflects the CuTe result, and afterwards `index(i)` returns `R.index(i)`. The padded `logical_divide` depends on this regular "restart at the next mode" structure; otherwise the tiles would be misaligned.

#### 6.3 Non-affine A: `normalized = None`

Replace A with a non-affine map such as swizzle: first compute the affine address, then XOR some of its bits.

```text
A(j) = swizzle(6·row + col)
```

Such a map has no shape/stride form, `as_affine()` returns `None`, `compose_grouped` is never entered, and so there is no CuTe normalized result. Only pointwise evaluation is possible, corresponding to CuTe's `ComposedLayout`:

```text
i ──B──▶ j ──A (with swizzle)──▶ address
```

#### 6.4 Decision flow

```text
                    composition(A, B)
                           │
              Are both A and B affine layouts?
               ┌───────────┴───────────┐
              no                      yes
               │                run compose_grouped
       normalized = None               │
       pointwise (6.3)       ┌─────────┴─────────┐
                           Ok(R)            NotDivisible
                             │                   │
                          Some(R)               None
                    evaluate via R (6.1, 6.2)  pointwise
```

### 7. Implementation notes

1. Only the pointwise fallback rejects an `inner` whose minimum address is negative (`CoordinateOutOfBounds`), because it uses those values as logical indices of `outer`. Normalization follows CuTe's signed stride divisibility condition `rest_stride % cs == 0 || rest_stride < cs` in `checked_step`, shared by the runtime and typed paths, so negative strides compose through `signum` (for example `(4,4):(4,1) ∘ 4:-1 = 4:-4`).
2. The codomain check on the non-normalized branch only probes `max`. Every subsequent evaluation uses checked `add`/`mul`, so even if an intermediate index overflows, the result is only `Err(Overflow)` and never undefined behavior.

#### 7.1 Fully static results

[`src/algebra/composition.rs`](../src/algebra/composition.rs) runs the same CuTe algorithm during trait resolution. When every shape/stride leaf of both sides is an `Int<N>`, the result type `TypedComposition<A, B>` is a zero-sized `Layout` whose value is `<TypedComposition<A, B> as StaticValue>::VALUE`, usable in a `const`. It keeps B's full nesting and never goes through `DynLayout`, so it is not limited by `MAX_RANK`.

```rust
use noodle_cute::algebra::TypedComposition;
use noodle_cute::layout::Layout;
use noodle_cute::{Int, IntTuple, Nil, StaticValue};

type Pair<const A: isize, const B: isize> = IntTuple<Int<A>, IntTuple<Int<B>, Nil>>;
// (6,2):(8,2) ∘ (4,3):(3,1) = ((2,2),3):((24,2),8)
type R = Layout<
    IntTuple<Pair<2, 2>, IntTuple<Int<3>, Nil>>,
    IntTuple<Pair<24, 2>, IntTuple<Int<8>, Nil>>,
>;
type A = Layout<Pair<6, 2>, Pair<8, 2>>;
type B = Layout<Pair<4, 3>, Pair<3, 1>>;
const C: R = <TypedComposition<A, B> as StaticValue>::VALUE;
```

Correspondence with section 3:

| CuTe `layout.hpp` | Type-level implementation |
|---|---|
| `coalesce_x`: `bw_coalesce` after flatten, trailing static 1 replaced by 2 | `CoalesceFold<true>` and `Prepend` in [`coalesce.rs`](../src/algebra/coalesce.rs); ordinary coalesce uses the default branch of the same fold |
| Right distribution when B is a tuple | `ComposeTree` recurses over B's tree; `Nil` is kept as an empty tuple |
| Static stride 0 returns B directly | The `ZERO_STRIDE` branch yields `s:0` |
| Fold over the first r-1 modes of A | `Fold`/`FoldStep`; static values are derived by `const fn step`, mixed values are computed by `TypedStep` |
| Unwrap when the result is empty / `rest_shape == 1`, otherwise append | `Finish`/`FinishNonempty` |

The differences from the dynamic `compose_leaf` all come from "copying the upstream static path exactly":

- Only stride 0 takes the pass-through branch; `s == 0` with a multi-mode A fails at the shape divisibility check, as upstream does, and with a single-mode A gives `0 : d*stride0`.
- The trailing stride discarded by unwrap is never evaluated, so an overflow of `rest_stride * stride[r-1]` does not affect the result.
- Non-divisibility, overflow, profile mismatch, or the presence of a dynamic leaf all fail to compile at instantiation, instead of returning `Err`.

Deviation from upstream: upstream `coalesce_x` takes the last mode, so an empty A cannot be instantiated; here, following this library's scalar convention, an empty A is treated as `1:0`, so `() ∘ s:d = s:0`.

#### 7.2 Fully static, mixed, and runtime entry points for coalesce

[`coalesce.rs`](../src/algebra/coalesce.rs) places the type-level and runtime simplification in the same module:

- A fully static `Layout` reads its result as `<CoalescedLayout<S, D> as StaticValue>::VALUE` (or `CoalescedLayout<S, D, true>` for `coalesce_x`); `CoalesceFold<false/true>` derives the type.
- A mixed `Layout` uses the `coalesce` / `coalesce_extended` methods, which keep static leaves and dynamic fields without converting to a runtime array. The output type is `CoalescedLayout<S,D>` / `CoalescedLayout<S,D,true>`; after the result is constructed, a checked `Result` is returned.
- `DynLayout` uses methods of the same names, which decide from the current values which modes can be removed or merged; the result is still a `DynLayout`.

The ordinary and extended versions share `PrependToEmpty<S,D,EXTENDED>` for initializing the tail. The keep branch shares a single implementation; on a static unit axis, the ordinary version returns an empty tail, while the extended version returns an extent-2 placeholder leaf carrying the original stride.

All three paths fold uniformly from right to left. The type level strictly follows CuTe's staticness condition: only when the current shape/stride and the first shape/stride of the accumulated layout are all static values can they be merged on the basis of `s*d == next_stride`. For example, `(Int<2>, DynInt(3)):(Int<1>, Int<2>)` is kept as two axes, whereas the corresponding `DynLayout` can be merged into `6:1`.

As in CuTe, a static zero factor makes the product static: `x * _0` is `_0` even for a dynamic `x`, so `(DynInt(3), Int<4>):(Int<0>, Int<0>)` merges into `12:_0` (with a dynamic extent) exactly like upstream `bw_coalesce`.

```rust
use noodle_cute::{DynInt, Int, IntTuple, Nil, StaticValue, shape, stride};
use noodle_cute::algebra::CoalescedLayout;
use noodle_cute::layout::{Layout, LayoutMap};

type Shape = IntTuple<Int<2>, IntTuple<Int<1>, Nil>>;
type Stride = IntTuple<Int<1>, IntTuple<Int<2>, Nil>>;
const NORMAL: Layout<IntTuple<Int<2>, Nil>, IntTuple<Int<1>, Nil>> =
    <CoalescedLayout<Shape, Stride> as StaticValue>::VALUE;
const EXTENDED: Layout<IntTuple<Int<4>, Nil>, IntTuple<Int<1>, Nil>> =
    <CoalescedLayout<Shape, Stride, true> as StaticValue>::VALUE;
let mixed = Layout::new(shape![Int::<2>, Int::<1>], stride![Int::<1>, DynInt(8)])?;
let extended = mixed.coalesce_extended()?;
assert_eq!(extended.index_extended(2)?, 8);
# Ok::<(), noodle_cute::Error>(())
```

Reading the constant requires both trees to have `Int<N>` leaves, and performs no runtime size or address-range validation; an overflowing static merge product fails to compile at instantiation. The method entry points check the input shape and the output layout, and dynamic errors return `Err`.

The type-level result of the extended version follows the upstream placeholder rule: a trailing static shape-1 is changed to 2, so the logical size may increase; a dynamic shape-1 in a mixed input is not changed to a static 2. The runtime version keeps the actual trailing extent, so that the placeholder value does not add overflow risk; the finite domain size is unchanged, but the same extended stride is kept. Empty inputs uniformly follow this library's `1:0` scalar convention. Equivalence of the extended map requires the leading extents of the original layout to be positive; with zero extents, simplification still follows CuTe's algebraic rules, and the division-by-zero error of the original extended decoding is not guaranteed to be preserved.

#### 7.3 Fully static, mixed, and runtime entry points for composition

All three kinds of algorithm are implemented in [`composition.rs`](../src/algebra/composition.rs).

| Input and entry point | Result | Structure and checks |
|---|---|---|
| Fully static `<TypedComposition<A,B> as StaticValue>::VALUE` | Zero-sized `Layout` constant | The type determines the value; static illegal shapes, divisibility errors, and product overflow are rejected at instantiation |
| Mixed `A.composition(B)`, A/B are `Layout` | `Result<TypedComposition<A,B>>` | Keeps static leaves and the full RHS nesting; dynamic arguments only change field values, not the type |
| Runtime `composition_layout(&A, &B)` | `Result<GroupedLayout>` | Normalizes with the current values and keeps the RHS top-level grouping |
| General `composition(A,B)` | `Result<Composition<A,B>>` | Keeps the existing normalization cache and the pointwise fallback for non-divisible or non-affine inputs |

The fully static and mixed entry points share `ComposeTree`, `Fold`, and `Finish`. Each stage expresses the same layout result through `Output: LayoutParts`, and relay stages forward `Output` directly. Finally `LayoutParts::into_layout` extracts the shape/stride and then the checks required by the entry point are run; this conversion itself does no normalization or validation. The shape/stride of helper traits should be projected from `Output`, keeping the public result aliases consistent.

The computation order is: first normalize the LHS through the shared `CoalesceFold<true>`, then distribute to the right over the RHS tree. Branches may only be decided by static facts: a static stride-0 broadcasts directly; a mode is skipped only for a static `next_shape == 1` or `rest_shape == 1`; and at the end, unwrap happens only for a static `rest_shape == 1`. Modes that happen to be 1 at runtime are kept; for example, `(4,6):(6,1) ∘ DynInt(2):Int<3>` gives `(DynInt(2),DynInt(1)):(Int<18>,Int<1>)`, and is not turned into a scalar type just because the remaining shape is 1 this time.

Typed coalesce can only use contiguity that is known statically, while runtime coalesce can use the current values, so the two may keep different LHS mode boundaries; weak divisibility depends precisely on these boundaries, so the numeric results may differ as well. For example, composing `(4,6):(1,4)` with dynamic leaves and `8:3`, the typed entry point keeps two LHS modes and the resulting addresses are `[0,3,4,7,8,11,12,15]`; the runtime entry point first merges into `24:1` and the result is `[0,3,6,9,12,15,18,21]`. This continues CuTe's dependence on static information and mode boundaries; the general lazy entry point still follows the runtime normalization behavior.

`TypedStep` derives the types of the extent, the remaining shape, and the remaining stride separately: `Int<N>` when everything they depend on is statically known, `DynInt` otherwise. Stride multiplication also keeps CuTe's static-zero rule: `Int<0> * DynInt` is still `Int<0>`; a static zero numerator divided by a dynamic value also stays `Int<0>`. These rules are limited to composition and do not change this library's type-promotion convention for ordinary integer operators.

```rust
use noodle_cute::{DynInt, Int, IntTuple, Nil, shape, stride};
use noodle_cute::algebra::TypedComposition;
use noodle_cute::layout::{Layout, LayoutMap};

type Pair<A,B> = IntTuple<A, IntTuple<B,Nil>>;
type A = Layout<Pair<Int<4>,Int<6>>, Pair<Int<6>,Int<1>>>;
type B = Layout<DynInt,Int<3>>;
let a: A = Layout::new(shape![Int::<4>,Int::<6>], stride![Int::<6>,Int::<1>])?;
let b: B = Layout::new(DynInt(8),Int::<3>)?;
let c: TypedComposition<A,B> = a.composition(b)?;
assert_eq!(c.index(1)?,18);
assert_eq!(c.index(2)?,1);
# Ok::<(), noodle_cute::Error>(())
```

The typed method checks the input shapes of both sides and the size and address range of the result; it does not go through `LayoutBuilder` or `DynLayout` and is not limited by `MAX_RANK`. It shares the signed `d < cs` stride condition and negative-stride support with the runtime normalization.

Checked differences from upstream: the divisibility assertions that CuTe omits for dynamic arguments are still checked in the typed Rust methods and return `NotDivisible`; a division by zero that cannot be performed returns `InvalidShape`, and an overflow of a dynamic absolute value or product returns `Overflow`. Hence a dynamic stride of 0 can scale directly on a single-mode LHS, but returns an error in a multi-mode LHS where `ceil(cs/abs(d))` must be computed; a static stride-0 broadcasts directly. The same illegal operations on static inputs are rejected at compile time. An empty RHS keeps the empty tuple, and an empty LHS is extended under this library's `1:0` convention.

The runtime normalization and lazy fallback contracts are collected in [`tests/composition_contract.rs`](../tests/composition_contract.rs), which also contains cases for mixed types, zero values, negative strides, overflow, and exceeding `MAX_RANK`. The fully static types and upstream examples are in the same file; the other layout-algebra tests share the layout constructors and non-affine maps in `tests/common/mod.rs` with the composition tests.

## Part II · Complement `complement`

This part explains the fully static, mixed, and runtime entry points in [`src/algebra/complement.rs`](../src/algebra/complement.rs) (see section 14.2 for a comparison of the entry points), as well as the runtime `complement_prefix` / `complement_shape` / `complement_extended`: how they find the gaps a layout leaves behind, why the leaves must be sorted by stride, why floor division is used, and why a leaf such as `1:q` appears at the end.

In this part, the layout being complemented is written `L`, and the result `R`.

### 8. What `complement` computes

The "complement" in the name is **not a set complement**. The layout `L` maps coordinates to addresses; it visits memory with jumps and leaves behind a number of unvisited **gaps**. `complement(L, cotarget)` produces another layout `R` whose addresses are exactly **the set of starting points of these gaps**.

Together, `(L, R)` flatten the address range `[0, cotarget)` into a two-dimensional "intra-tile offset × inter-tile offset" structure; this is where the `logical_divide` formula at the beginning of Part I comes from.

An intuitive analogy: `L` is "how to walk within one tile", and `R` is "how to jump from tile to tile".

#### 8.1 A minimal example

Take `L = 4:1` (reading 4 contiguous elements) and `cotarget = 24`:

```text
addresses visited by L:  0 1 2 3
whole address space:     0 .. 23
```

Cutting the 24 cells into groups of 4 gives 6 groups, 4 apart from each other, so

```text
complement(4:1, 24) = 6:4
```

The addresses of `R = 6:4` are `{0, 4, 8, 12, 16, 20}`, exactly the starting point of each tile. `L(i) + R(j)` covers `[0, 24)` exactly, with no overlaps and no omissions.

### 9. Core invariant: `current`

The algorithm has a single state variable, called `current` in the code, meaning:

> Starting from address 0, the length that has so far been **completely filled** by the leaves processed.

The initial value `current = 1` means "so far only the single point at address 0 is occupied". Then each leaf `s:d` of `L` is processed **in increasing order of stride**, and two things are done for each leaf:

1. **Fill the gap**: if there is a gap between `current` and the leaf's starting stride `d`, add it as a leaf of `R`, i.e. `push(d / current, current)`;
2. **Advance**: the leaf itself covers up to `s · d`, so `current = s · d`.

#### 9.1 Why sorting by stride is required

The gap computation relies on "expanding layer by layer from small to large". If a leaf with a large stride were processed first, `current` would jump straight to a large value, and the correct gaps for the later small-stride leaves could no longer be computed.

```rust
order[..flat.rank()].sort_unstable_by_key(|&a| (flat.strides()[a], a));
```

The sort key is `(stride, original index)`. The second key `a` is a crucial detail: **it turns the unstable sort into a stable sort**, equivalent to CuTe's `stable_sort`, without any extra allocation. Two leaves with the same stride stay in their original order, so the result is deterministic and reproducible.

### 10. `complement_prefix` step by step

```rust
fn complement_prefix<L: AffineLayoutMap>(layout: &L) -> Result<Option<(LayoutBuilder, isize)>>
```

This is the first half shared by `complement_shape` and `complement_extended`; it returns `(prefix leaves, next stride)`.

#### 10.1 Rejecting an empty domain

```rust
let flat = layout.as_flat_layout()?;
if flat.size()? == 0 {
    return Err(Error::Unsupported);
}
```

`size == 0` means some axis has length 0, so the layout visits no address and "gaps" are meaningless; `Unsupported` is reported directly. This is a deliberate `Result` in place of upstream undefined behavior: a zero extent makes the next stride `s * d` zero, and upstream's `ceil_div(cotarget, new_stride)` then divides by zero.

#### 10.2 Filtering out broadcast axes

```rust
let flat = flat.filter()?;
```

`filter` **changes the shape of every stride-0 leaf to 1 and then coalesces**. A stride-0 axis is a **broadcast**: no matter how the coordinate changes, the address stays put. It consumes no address space and therefore contributes nothing to "which addresses are occupied"; it must be removed first, otherwise it would be misjudged as occupying a large range. As a side effect, `coalesce` also merges adjacent leaves that can be merged (when `s0·d0 == d1`), minimizing the number of leaves to process afterwards.

#### 10.3 The all-broadcast special case

```rust
if flat.rank() == 1 && flat.leaf_stride(0)? == 0 {
    return Ok(None);
}
```

If only `1:0` remains after filtering (the original layout is `1:0`, a pure broadcast `4:0`, or all axes have length 1), `L` occupies only the single point at address 0, and **the whole cotarget is a gap**. This is CuTe's special-case branch; `None` is passed upward as a signal, and the caller directly returns the whole contiguous space:

```text
complement(4:0, 6)       = 6:1
complement_extended(4:0) = 1:1
```

#### 10.4 Rejecting negative strides

```rust
for (a, slot) in order.iter_mut().enumerate().take(flat.rank()) {
    if flat.leaf_stride(a)? < 0 {
        return Err(Error::Unsupported);
    }
    *slot = a;
}
```

A negative stride means the address walks backwards, and the "expand from small to large" model breaks down entirely. The same loop also initializes `order` to `[0, 1, 2, …]`.

#### 10.5 Main loop: filling gaps one by one

```rust
let mut prefix = LayoutBuilder::new();
let mut current = 1;
for &a in &order[..flat.rank()] {
    let (s, d) = (flat.leaf_shape(a)?, flat.leaf_stride(a)?);
    let (gap, next) = complement_step(s, d, current)?;
    if gap > 1 {
        prefix.push(gap, current)?;
    }
    current = next;
}
```

Each of the three branches corresponds to a real situation:

| Case | Meaning | Handling |
|---|---|---|
| `gap == 0` | `d < current`: the leaf's starting point falls **inside the already covered region** → addresses overlap, and `(L, R)` cannot be injective | `Err(NonInjective)` |
| `gap == 1` | `d == current`: a seamless fit, **no gap** | Skipped; takes no leaf capacity |
| `gap > 1` | A gap exists, of length `gap` and spacing `current` | `push(gap, current)` |

Pushing only when `gap > 1` is not just a convenience: `LayoutBuilder` is a fixed-length `MAX_RANK` buffer, so every slot saved lowers the risk of hitting `CapacityExceeded`; besides, length-1 leaves would eventually be removed by `coalesce` anyway.

**Floor division is deliberate**: when `d` is not divisible by `current`, the algorithm rounds down instead of reporting an error, at the cost that the result may overlap the original layout in its holes. This aligns with CuTe's behavior and is not a defect; see [14.1](#141-what-the-algorithm-does-not-guarantee) for details.

#### 10.6 Walking through `L = (2,2):(1,6)`

| Step | Leaf | `current` on entry | `gap = d / current` | Action | `current` on exit |
|---|---|---|---|---|---|
| 1 | `2:1` | 1 | `1 / 1 = 1` | No gap, skip | `2 · 1 = 2` |
| 2 | `2:6` | 2 | `6 / 2 = 3` | push `3:2` | `2 · 6 = 12` |

Prefix = `3:2`, `current = 12`, and finally `complement((2,2):(1,6), 24) = (3,2):(2,12)`.

Check: the addresses of `L` are `{0, 1, 6, 7}` and those of `R` are `{0, 2, 4, 12, 14, 16}`; their pairwise sums give 24 distinct addresses, with no overlap.

### 11. `complement_shape`: filling up to the target extent

The prefix only handles the gaps "between leaves"; what remains is the **tail gap** from `current` to `cotarget`.

```rust
let mut rest = current;
let mut stride = current;
for &a in cotarget {
    let extent = ceil_div(a, rest);
    rest = if a == 0 { rest } else { ceil_div(rest, a) };
    if extent != 1 {
        leaves.push(extent, stride)?;
    }
    stride = mul(stride, extent)?;
}
leaves.layout()?.coalesce()
```

#### 11.1 Why `ceil_div`

`cotarget` is not necessarily divisible by `current`. CuTe chooses to **round up**, guaranteeing that the result covers the cotarget: better to cover a little too much than too little:

```text
complement(3:1, 24) = 8:3      // current = 3, ceil(24/3) = 8, covers [0,24)
```

#### 11.2 Why the result differs when the cotarget is a tuple

`cotarget` can be a shape (several numbers), in which case CuTe uses a tuple ceil_div that **folds leaf by leaf**:

```text
extent_i = ceil(a_i / rest)
rest     = ceil(rest / a_i)
```

Each `a_i` first consumes part of `rest` and leaves the remainder to the next one, so the result may be larger than when using the product directly:

```text
complement_shape(3:1, [4,6]) = 12:3      // not 8:3
complement(3:1, 24)          =  8:3      // 24 = 4·6, yet the result differs
```

Checking `[4, 6]` by hand (starting with `current = rest = stride = 3`):

| `a_i` | `extent = ceil(a / rest)` | New `rest` | push | New `stride` |
|---|---|---|---|---|
| 4 | `ceil(4/3) = 2` | `ceil(3/4) = 1` | `2:3` | `3 · 2 = 6` |
| 6 | `ceil(6/1) = 6` | `ceil(1/6) = 1` | `6:6` | 36 |

The leaves are `(2,6):(3,6)`; during coalesce `2 · 3 == 6`, so they merge into `12:3`.

When `a == 0`, the second division is skipped (`rest` stays unchanged) to avoid division by zero.

#### 11.3 Wrap-up and special cases

- Pushing only when `extent != 1` is, again, to save slots.
- A final `coalesce()` is applied uniformly: it merges adjacent leaves and normalizes to `1:0` when everything is 1, consistent with CuTe.
- The entry point catches the `None` returned by the prefix (all broadcast) and directly returns `prod(cotarget):1`:

  ```rust
  let Some((mut leaves, current)) = complement_prefix(&layout)? else {
      let size = cotarget.iter().try_fold(1, |acc, &c| mul(acc, c))?;
      return DynLayout::new(&[size], &[1]);
  };
  ```

### 12. `complement_extended`: marking the repetition stride

```rust
let Some((leaves, current)) = complement_prefix(&layout)? else {
    return DynLayout::new(&[1], &[1]);
};
let prefix = leaves.layout()?.coalesce()?;
if prefix.size()? == 1 {
    return DynLayout::new(&[1], &[current]);
}
// …copy the leaves of prefix into out…
out.push(1, current)?;
```

This version **has no cotarget**: it keeps the internal gaps and then appends a trailing leaf `1:current`.

`1:current` has logical size 1 and **does not increase the size**; its only purpose is to **carry stride information**: it tells the caller "if another copy is to be tiled, it starts at `current`". Product-style operations (`blocked_product`, `tiled_product`, `TiledMmaLayout`) rely on exactly this value to derive the outer stride.

```text
complement_extended(4:2) = (2,1):(1,8)
```

Here **8 comes from `s · d = 4 · 2`, not from the cosize (= 7)**. When reasoning about product operations, this stride must not be replaced by the address span.

The short-circuit branch for `prefix.size() == 1` handles the case of "no internal gaps at all" and directly returns `1:current`.

### 13. `LayoutBuilder`: fixed-capacity result buffer

Defined in [`src/layout/dynamic_layout.rs`](../src/layout/dynamic_layout.rs) and shared by every runtime algebra operation.

```rust
pub(crate) struct LayoutBuilder {
    shape: [isize; MAX_RANK],
    stride: [isize; MAX_RANK],
    leaves: usize,
    groups: [usize; MAX_RANK],
    modes: usize,
    closed: usize,
}
```

It is a fixed-size, stack-allocated, `Copy` collector that returns `Err(CapacityExceeded)` on overflow instead of panicking. `push` / `push_leaves` append leaves to the open mode, `close_mode` ends it (so leaves pushed before one `close_mode` form one mode), and `push_modes` copies every top-level mode of a `GroupedLayout`. `finish_flat` validates the leaves as a `DynLayout`, and `finish` groups them into a `GroupedLayout`. The crate avoids allocation, so it cannot use `Vec`; this is also why [10.5](#105-main-loop-filling-gaps-one-by-one) and [11.3](#113-wrap-up-and-special-cases) skip unit-length leaves.

### 14. Errors and edge cases

| Input | Returns |
|---|---|
| `size == 0` (empty domain) | `Err(Unsupported)` |
| A negative stride remains after filtering | `Err(Unsupported)` |
| The cotarget contains a negative number | `Err(InvalidShape)` |
| `gap == 0` (overlapping strides) | `Err(NonInjective)` |
| The number of leaves exceeds `MAX_RANK` | `Err(CapacityExceeded)` |
| Only `1:0` remains after filtering | `prod(cotarget):1` / `1:1` |

#### 14.1 What the algorithm does not guarantee

**`complement` does not guarantee a tiling with no overlaps and no holes.** A counterexample:

```text
complement((2,2):(1,3), 12) = 2:6
```

The addresses of `L` are `{0, 1, 3, 4}` and `R = 2:6` is `{0, 6}`; together they give `{0,1,3,4,6,7,9,10}`, and **addresses 2, 5, 8, 11 are still missing**. The reason is that `L`'s own stride 3 is not divisible by `current = 2`, so the floor division `3 / 2 = 1` judges it as "no gap", and address 2 is missed.

Therefore conclusions must be drawn from the algorithm's conditions and the actual return value; `complement` cannot be treated as a general set complement or a tool for proving bijectivity. Likewise, the fact that broadcasts are filtered out does not mean the original layout has become injective.

#### 14.2 Fully static, mixed, and runtime entry points

The algorithm follows `filter`, `detail::complement_prefix`, and the two `detail::complement` overloads in [CuTe `layout.hpp`](https://github.com/NVIDIA/cutlass/blob/main/include/cute/layout.hpp). All related implementations are in [`complement.rs`](../src/algebra/complement.rs) and are re-exported from `algebra`.

| Input and entry point | Result and structure |
|---|---|
| Fully static `<TypedComplement<L, C> as StaticValue>::VALUE` | Zero-sized `Layout` constant; the cotarget may be an integer or a nested shape |
| Fully static `<TypedComplement<L, Nil, true> as StaticValue>::VALUE` | Constant layout keeping the terminal `1:repetition-stride` mode |
| Mixed `L.complement(cotarget)`, L is a `Layout` | `Result<TypedComplement<L,C>>`, keeping the static leaves that can be derived |
| Mixed `L.complement_extended()` | `Result<TypedComplement<L,Nil,true>>` |
| Runtime free functions `complement` / `complement_shape` / `complement_extended` | Take any `AffineLayoutMap`, sort and simplify with the current values, and return a `DynLayout`; divide/product call them too |

The fully static and mixed entry points share the following stages and do not convert to runtime arrays, so they are not limited by `MAX_RANK`:

1. `Flatten` flattens the shape/stride; `FilterLeaf` only replaces the shape of a static zero stride with a static 1. `Filter` reuses coalesce's `Prepend` to remove static unit axes and merge static contiguous modes. When everything is removed, an empty intermediate layout represents upstream's `1:0` special case.
2. `Sort` / `Insert` stably sort by static stride. As upstream, **a multi-mode layout after filtering requires all strides to be statically known**; dynamic strides are supported only for a single mode. When multi-mode dynamic sorting is needed, use `as_flat_layout()` and the runtime entry points.
3. `Prefix` computes `gap = d/current` and `next = s*d`; the dynamic and typed paths share `complement_step` to check sign, overlap, and product overflow. Static computation reuses the division and multiplication of `Int`, and results that depend on dynamic values stay dynamic integers.
4. With a cotarget, `CeilTarget` folds ceil_div leaf by leaf; `GroupShape`, like upstream `coalesce(shape)`, only merges adjacent leaves that are both static or both dynamic. `TargetStrides` reuses the static-zero propagation of `ComposeProduct` to build colexicographic strides, without computing the unused trailing product; this avoids an irrelevant overflow in the trailing product of `Compact::compact`. After the prefix and tail are concatenated, `CoalesceFold` / `FinishCoalesce` are reused, and finally `Unwrap` restores a single-mode result to an integer profile.
5. Without a cotarget, the prefix is first simplified with `CoalesceFold`, and then a static unit axis and the final stride are appended; when the prefix is empty, `1:final-stride` is returned directly. The all-broadcast special case likewise unwraps a single mode and directly returns the compact layout of the cotarget or `1:1`, without any further ordinary layout coalesce.

The type structure is affected only by static information. For example, when the dynamic cotarget of `4:1` happens to be 4, the result is still a dynamic `1:4` leaf; if the cotarget is `Int<4>`, ordinary coalesce normalizes the result to a static `1:0`. A dynamic shape that happens to equal 1 also keeps its original mode.

```rust
use noodle_cute::{DynInt, Int, IntTuple, Nil, StaticValue, layout::Layout};
use noodle_cute::algebra::TypedComplement;

type Pair<const A: isize, const B: isize> = IntTuple<Int<A>, IntTuple<Int<B>, Nil>>;
type L = Layout<Int<4>, Int<2>>;
const C: Layout<Pair<2,3>, Pair<1,8>> = <TypedComplement<L, Int<24>> as StaticValue>::VALUE;
const E: Layout<Pair<2,1>, Pair<1,8>> = <TypedComplement<L, Nil, true> as StaticValue>::VALUE;
let input = Layout::new(DynInt(4), Int::<2>)?;
let mixed: Layout<IntTuple<Int<2>, IntTuple<DynInt, Nil>>,
                  IntTuple<Int<1>, IntTuple<DynInt, Nil>>> = input.complement(Int::<24>)?;
assert_eq!(mixed.shape.1.0, DynInt(3));
# Ok::<(), noodle_cute::Error>(())
```

Deviations and check boundaries: a static zero domain, a static negative stride, a static zero gap, and the like are rejected at compile time; dynamic errors are returned through `Result`. A typed dynamic zero stride is not filtered and returns `NonInjective`, whereas the runtime numeric path can filter it. The typed methods check the input size and the final `Layout::new`; reading a fully static result through `StaticValue` skips the final runtime size/address_range validation, as for coalesce. The empty tuple follows this library's scalar convention; the reverse ceil_div with a zero cotarget divides by zero upstream, whereas this library follows the zero-domain extension of the existing dynamic interface: on zero it keeps rest, and the tail size is zero.

The contract tests are collected in [`tests/complement_contract.rs`](../tests/complement_contract.rs), covering fully static constant types, upstream examples and filtering/sorting, mixed-type dependencies, dynamic edge cases, typed outputs exceeding `MAX_RANK`, as well as the existing cotarget, repetition-stride, and error contracts. Compile-failure constraints are covered by rustdoc `compile_fail` examples.

### 15. Flow overview

```text
                    ┌───────────────────────────────────────┐
                    │           complement_prefix           │
   L ──────────────▶│ 1. size == 0?       → Err             │
                    │ 2. filter (drop broadcast + coalesce) │
                    │ 3. only 1:0 left?   → None            │
                    │ 4. negative stride? → Err             │
                    │ 5. sort by (stride, idx)              │
                    │ 6. loop:                              │
                    │      gap = d / current                │
                    │      gap == 0 → NonInjective          │
                    │      gap  > 1 → push(gap,cur)         │
                    │      current = s · d                  │
                    └──────────┬────────────────────────────┘
                               │ (prefix, current)
                   ┌───────────┴────────────┐
                   ▼                        ▼
          complement_shape          complement_extended
      ceil_div per cotarget leaf     append 1:current
         fills the tail gap      (only marks repetition stride)
                   │                        │
                   ▼                        ▼
           coalesce normalize       (gaps…, 1:current)
```

One-line mnemonic: **sort → fill gaps leaf by leaf → fill the tail**. The only state is `current`, and three branches (overlap / seamless / gap) cover every case.

## Part III · Inverse `inverse`

This part explains `sorted_leaves` / `right_inverse` / `left_inverse` in [`src/algebra/inverse.rs`](../src/algebra/inverse.rs), as well as the enumerating `dense_left_inverse` / `dense_right_inverse`: how logical indices are recovered from addresses, why the leaves must be sorted by stride, and why "address holes" make the two inverses give completely different answers.

In this part, the layout being inverted is written `L`, and the result `R`.

### 16. What exactly an inverse inverts

The map defined by an affine layout `L = shape:stride` has two steps:

```text
logical linear index i  ──decompose──▶  coordinate (c0,c1,…)  ──weighted sum──▶  address a
```

- **Decomposition** uses a "left-major mixed radix": `c0 = i % s0`, `c1 = (i / s0) % s1`, …
  so the weight of the `k`-th leaf on the **logical side** is the shape prefix product `p_k = s0·s1·…·s_{k-1}`.
- **Summation** uses the strides: `a = Σ c_k · d_k`, so the weight of the `k`-th leaf on the **address side** is `d_k`.

A layout is therefore essentially a table with two columns of weights:

| Leaf | Range | Logical-side weight | Address-side weight |
|---|---|---|---|
| `k` | `s_k` | `preprod[k]` (prefix product) | `stride[k]` |

**Inverting means swapping these two columns of weights**: the original maps `preprod → stride`, and the inverse maps `stride → preprod`.

All the remaining complexity comes from just two things:

1. The input of the inverse is an **address**, so the leaves must be reordered **by increasing stride** (the same motivation as the sorting in `complement`, see [9.1](#91-why-sorting-by-stride-is-required));
2. The address space may have **holes** (`L` visits with jumps) or **overlaps** (broadcast axes), and a decision must be made about how to handle them.

The difference between `right_inverse` and `left_inverse` is precisely two different attitudes toward "holes".

### 17. The contracts of the two inverses

| | Definition | Intuition |
|---|---|---|
| **Right inverse** `R` | `L(R(a)) == a` for all `a < size(R)` | "Give me an address, and I return an index that generates it". The address asked about must actually be producible by `L`, so it **must not touch a hole** and can only stop before the first hole. |
| **Left inverse** `G` | `G(L(i)) == i` for all `i < size(L)` | "Give me an address that `L` produced, and I return the original index". Addresses in holes are never asked about, so **anything may be returned** there, and it can go all the way to the end. |

This explains the following pair of seemingly contradictory results:

```text
right_inverse(4:2) = 1:0            // the addresses are only {0,2,4,6}; 1 is a hole, so the right inverse can only cover [0,1)
left_inverse (4:2) = (2,4):(0,1)    // the left inverse is defined on all of 0..7; holes hold garbage values, but nobody queries them
```

### 18. Common preparation: `sorted_leaves`

```rust
let flat = layout.as_flat_layout()?.coalesce()?;          // ①
for a in 0..flat.rank() {
    preprod[a] = acc;                                     // ②
    acc = mul(acc, flat.leaf_shape(a)?)?;
    order[a] = a;
}
order[..flat.rank()].sort_unstable_by_key(|&a| (flat.strides()[a], a));   // ③
```

Both structural inverses share this segment, which does only three things:

① **Flatten + coalesce**. Nested structure is meaningless for inversion, so it is first flattened into a one-dimensional sequence of leaves; `coalesce` drops leaves with `shape == 1` (they contribute nothing to the address) and merges adjacent contiguous leaves (`(4,2):(1,4)` becomes `8:1`; neither `(4,2):(2,1)` nor `(2,4):(4,1)` can be merged). The fewer the leaves, the simpler the subsequent loop and the more canonical the output.

② **`preprod[a]` is the logical-side weight**. It is the source of every stride in the inverse layout: the strides of the inverse are the preprods of the original layout.

③ **Stable sort by `(stride, original index)`**. The second key plays the same role as in [9.1](#91-why-sorting-by-stride-is-required): it turns the unstable sort into a stable sort, equivalent to CuTe's `stable_sort`, without allocation.

Note that `order` is only a permutation of indices; `flat` itself is not reordered.

### 19. `right_inverse`: greedily chaining a contiguous address prefix

```rust
let mut current = 1;
for &a in sorted_order {
    let (s, d) = (shape(a), stride(a));
    if d == current {                       // join only on a seamless fit
        leaves.push(s, preprod[a])?;
        current = mul(s, d)?;               // address span covered so far
    }
}
leaves.layout()?.coalesce()
```

An empty layout (`size == 0`) is not rejected: like upstream, the fold simply runs on its coalesced leaves, e.g. `right_inverse(0:1) = 0:1` and `right_inverse(0:2) = 1:0`. `left_inverse` does the same (`left_inverse(0:1) = 0:1`). Only `complement` rejects empty layouts, because upstream divides by zero there.

`current` has the same meaning as in [section 9](#9-core-invariant-current): **the span starting at address 0 that has been covered without holes**, with initial value 1 (only address 0 occupied).

The algorithm greedily asks: is there a leaf whose stride **exactly equals** `current`?

- Yes → it extends the coverage from `current` exactly to `s · d`, so it is joined. In the inverse layout this leaf keeps its shape, and its stride is replaced by `preprod[a]` (the output logical index).
- No → `current` is a hole. The loop does not `break`, but the strides of the remaining leaves only get larger and **can never match again**, which is equivalent to stopping here.

Measured results:

```text
(2,4):(4,1)  → right_inverse = (4,2):(2,1)   // current: 1→4→8, full coverage, a true bijective inverse
(3,2):(1,8)  → right_inverse = 3:1           // current: 1→3, addresses 3..7 are holes, stop
4:2          → right_inverse = 1:0           // no leaf with stride 1, cannot take a single step
(2,4):(0,1)  → right_inverse = 4:2           // the broadcast axis is skipped naturally because d == current >= 1 never holds
```

Three details:

- When no leaf is joined, `leaves` is empty, and the final `coalesce()` reduces it directly to the valid `1:0`, so there is no need to push a `1:0` sentinel in advance (which would take one `MAX_RANK` slot).
- **Negative strides need no explicit rejection**: they likewise never match `current >= 1` and are silently ignored.
- This is a **structural** prefix, not "the longest prefix found by search". For example, `(2,2):(1,1)` actually reaches addresses 0, 1, 2, but the algorithm only acknowledges 0 and 1. This is the same kind of caveat as [14.1](#141-what-the-algorithm-does-not-guarantee): conclusions must be read according to the algorithm's conditions, not according to set semantics.

### 20. `left_inverse`: a decoder that spans holes

The left inverse must be defined on **the entire address interval**, so the idea becomes: **split the address into digits as a mixed-radix number, multiply each digit by the `preprod` of the corresponding leaf, and sum.**

The "radices" used to split the digits come from **the ratios of adjacent strides** after sorting. Let the sorted strides be `d_0 < d_1 < … < d_{k-1}`, the corresponding preprods `p_0 … p_{k-1}`, and the shape of the last leaf `s_last`; then

```text
R = ( d_0 , d_1/d_0 , d_2/d_1 , … , d_{k-1}/d_{k-2} , s_last )
  : (  0  ,   p_0   ,   p_1   , … ,    p_{k-2}      , p_{k-1} )
```

Term by term:

- **The first term `(d_0 : 0)`**: the part of the address below `d_0` is reachable by no leaf (a pure hole). Giving it stride 0 amounts to "discarding these digits".
- **The `i`-th middle term `(d_i/d_{i-1} : p_{i-1})`**: the value of this digit must be multiplied by the logical weight of the **previous** leaf, so the shape and stride are **offset by one slot** in the arrays; this is exactly what the code comment "shape has one fewer leading term than stride" means.
  There is a key trick here: the range of this digit is taken as `d_i/d_{i-1}` rather than leaf `i-1`'s own `s_{i-1}`. Since `d_i/d_{i-1} >= s_{i-1}`, the extra part is exactly the hole after leaf `i-1`. **Merging "the leaf's range" and "the hole after it" into one mode** saves a leaf and is completely valid, because nobody checks the output on holes.
- **The last term `(s_last : p_{k-1})`**: there is no hole after the last leaf that needs to be absorbed, so its actual shape is used.

The code implements this chain of ratios with `covered` and `min_stride`:

```rust
if d % covered != 0 { return Err(Error::NotDivisible); }
if d < min_stride   { return Err(Error::NonInjective); }
shape[len]      = d / covered;        // the radix of this digit
stride[len + 1] = sorted.preprod[a];  // offset by one slot: assigned to the next slot
covered    = mul(covered, shape[len])?; // covered becomes d
min_stride = mul(flat.leaf_shape(a)?, d)?;   // minimum non-overlapping stride for the next non-broadcast leaf
```

**Invariant: after each iteration, `covered == the stride d of the current leaf`** (because `covered · (d/covered) = d`), and `min_stride == s · d` (the upper bound of the addresses covered by the current leaf). Therefore:

- `d % covered != 0` requires **the stride chain to be successively divisible**: `d_0 | d_1 | d_2 | …`; otherwise `NotDivisible` is returned (CuTe fails a static assertion here outright);
- `d < min_stride` requires **non-broadcast leaves not to overlap**: `d_k >= s_{k-1} · d_{k-1}`; otherwise different logical coordinates would map to the same address (e.g. `(2,2):(1,1)` or `(3,2):(1,2)`), the left-inverse identity cannot hold, and `NonInjective` is returned.

Two preprocessing rules: **leaves with stride 0 (broadcasts) are skipped directly**; a broadcast layout is not injective and has no left inverse in the first place, so what is obtained after skipping is actually one of its *right* inverses. **Negative strides** return `Unsupported`.

#### 20.1 Walking through `L = (3,2):(1,8)`

Sorted leaves: `3:1` (`p = 1`), `2:8` (`p = 3`).

| Step | `covered` (before) | Shape slot written | Stride slot written | `covered` (after) |
|---|---|---|---|---|
| Leaf `3:1` | 1 | `1/1 = 1` | next slot `p = 1` | 1 |
| Leaf `2:8` | 1 | `8/1 = 8` | next slot `p = 3` | 8 |
| Finish | | `s_last = 2` | | |

This gives `(1,8,2):(0,1,3)`, which after coalescing away the shape-1 is **`(8,2):(1,3)`**.

Checking the identity: `i = i0 + 3·i1`, address `a = i0 + 8·i1` (`i0 < 3`, `i1 < 2`).
`R` splits `a` into `a0 = a % 8 = i0` and `a1 = a / 8 = i1`, and outputs `1·a0 + 3·a1 = i0 + 3·i1 = i` ✓

Note that the range of the first digit is **8**, not 3: the extra `3..7` is the absorbed hole; the output there is a garbage value, but `L` never produces these addresses.

#### 20.2 Comparison table

```text
L            right_inverse    left_inverse
4:2          1:0              (2,4):(0,1)     // the right inverse stops before the hole, the left inverse absorbs it
(4,2):(2,1)  (2,4):(4,1)      (2,4):(4,1)     // dense bijection, the two agree
(2,4):(4,1)  (4,2):(2,1)      (4,2):(2,1)
(4,2):(1,4)  8:1              8:1             // coalesces to 8:1, self-inverse
(3,2):(1,8)  3:1              (8,2):(1,3)
(2,4):(0,1)  4:2              4:2             // broadcast: the left inverse degenerates to a right inverse
```

### 21. Structural inverse vs enumerated inverse

`Inverse<L>` and its two constructors `dense_left_inverse` / `dense_right_inverse` take a completely different route: they ignore the stride structure and instead, at `index(address)` time, **scan the original layout linearly starting from logical index 0**, returning on the first hit (so with multiple preimages, the smallest is taken).

| | Structural (`left_inverse` / `right_inverse`) | Enumerated (`dense_*_inverse`) |
|---|---|---|
| Applies to | Affine layouts (with shape/stride metadata) | Any `LayoutMap`, including non-affine maps such as swizzle |
| Cost | One sort + one pass; the result is a static structure | An `O(size)` scan per query, but no heap allocation |
| Holes | Right inverse truncates, left inverse absorbs | Always `NotSurjective` |
| Result type | The typed entry points return a `Layout`, the runtime entry points a `DynLayout`; both can take further part in the algebra | A wrapper type; `as_affine()` returns `None` |

`dense_left_inverse` requires the original layout to be injective and its addresses to cover exactly `[0, size)` (an empty layout is allowed); `dense_right_inverse` requires the addresses to cover `[0, max]` contiguously starting from 0, but **allows broadcasts**, and picks the smallest logical preimage for each address (at construction time, sparse maps with `max + 1 > size(L)` are first rejected by the pigeonhole principle; dense bijections with `max + 1 == size(L)` that pass the affine stride-separation argument are accepted directly in $O(\text{leaves}^2)$; in all other cases the preimages are scanned address by address).

Division of labor between the two routes: the internals of the layout algebra (`logical_divide`, `raked_product`, TV layout derivation) use the structural inverses, because the results take further part in algebraic operations; the enumerated inverses are used only when an exact reverse lookup is needed for non-affine maps such as swizzle.

#### 21.1 Fully static, mixed, and runtime entry points for inverse

The implementation corresponds to `right_inverse` / `left_inverse` in [CuTe `layout.hpp`](https://github.com/NVIDIA/cutlass/blob/main/include/cute/layout.hpp), and does not treat equalities that happen to hold at runtime as static facts.

| Input and entry point | Result and constraints |
|---|---|
| `<TypedInverse<L> as StaticValue>::VALUE` / `<TypedInverse<L, true> as StaticValue>::VALUE` | Zero-sized `Layout` constant; every input shape/stride leaf is static |
| `L.right_inverse()`, L is a `Layout` | `Result<TypedInverse<L>>`; ignores dynamic strides and only chains statically known contiguous addresses |
| `L.left_inverse()`, L is a `Layout` | `Result<TypedInverse<L,true>>`; strides after coalesce must all be static, shapes may be mixed |
| Free functions `right_inverse` / `left_inverse` | Coalesce, sort, and invert with the current values, returning a `DynLayout` |
| `dense_left_inverse` / `dense_right_inverse` | Keep the runtime enumeration interface, applicable to any `LayoutMap` |

`TypedInvert<LEFT>` shares the following preparation: first flatten and simplify with `CoalesceFold`, then `Attach` keeps `(shape, original logical prefix product)` for each leaf. The right inverse filters out dynamic strides, and the left inverse requires the stride types to be uniquely determined by `StaticValue`; both reuse complement's stable `Sort`, carrying the original logical prefix product while sorting, since it cannot be recomputed after sorting.

`RightFold` produces a leaf only when the static values of `stride` and `current` are equal. Hence the typed right inverse of `(DynInt(3), Int<2>):(Int<1>, Int<3>)` is `DynInt(3):Int<1>`, whereas after conversion to `DynLayout` it can be merged into `6:1` based on the current values. Once a dynamic shape has been joined, `current` becomes a dynamic type, and subsequent strides cannot be joined on the basis of this run's values.

`LeftFold` computes `gap = d/covered` from the sorted static strides, and the result strides come from the original logical prefix products; dynamic shapes affect only the prefix products, the overlap check, and the final extent. Consistent with the checked runtime entry points, negative strides, non-broadcast overlaps, and non-divisibility are rejected; a dynamic overlap returns `NonInjective`, and product overflow returns `Overflow`. This is a stricter input check than upstream's quasi-inverse path. Broadcasts are still handled as a quasi-inverse, with no claim to recover all broadcast coordinates.

The finish reuses `CoalesceFold` / `FinishCoalesce` and `UnwrapLayout`, and a single-mode result is restored to an integer profile. The empty tuple returns `1:0` under the scalar convention; empty layouts are inverted like upstream (e.g. `0:_1` gives `0:_1`). Static illegal divisibility or arithmetic overflow is rejected at instantiation; reading a fully static result through `StaticValue` skips the final runtime size and address-range validation, as for the other operations. The typed path does not go through fixed-capacity arrays, so both inputs and outputs may exceed `MAX_RANK`.

The contracts are collected in [`tests/inverse_contract.rs`](../tests/inverse_contract.rs): they keep the existing structural-inverse, enumerated-inverse, and capacity-boundary tests, and verify constant types, mixed staticness rules, dynamic errors, and a 33-mode bit-reversal layout against an independently computed preimage formula.

## Part IV · Divide `divide`

This part explains `logical_divide` / `zipped_divide` / `tiled_divide` / `flat_divide` and the per-mode versions `*_divide_modes` in [`src/algebra/divide.rs`](../src/algebra/divide.rs), as well as `TileRest` and `Grouping` in [`src/algebra.rs`](../src/algebra.rs), which they share.

In this part, the layout being divided is written `L`, and the tile (also called the tiler) `T`.

### 22. What dividing does

View `24:1` as a column-major matrix with 6 rows and 4 columns: address `a = row + 6·col`. Now we want to cut it into 2×2 tiles and afterwards access it with two coordinates:

- **Intra-tile coordinate**: which element within a 2×2 tile;
- **Tile index**: which tile.

`T = (2,2):(1,6)` describes exactly the addresses `{0, 1, 6, 7}` occupied by "tile 0" (rows 0–1, columns 0–1). The remaining question is: where do the other tiles start? The diagram below marks which tile each cell belongs to:

```text
      col0 col1 col2 col3
row0     0    0    3    3
row1     0    0    3    3
row2     1    1    4    4
row3     1    1    4    4
row4     2    2    5    5
row5     2    2    5    5
```

"The starting points of the other tiles" are exactly the gaps left by `T`, i.e. the complement `complement(T)` from [Part II](#part-ii--complement-complement). So the result of dividing is a pair:

```text
(intra-tile coordinate t, tile index r)  ↦  T(t) + R(r)          R = complement(T)
```

This "logical index" is then fed into `L` to obtain the address (here `L = 24:1` is the identity map, so the effect is invisible; section 24.2 shows the case of a nontrivial `L`). This is the formula at the beginning.

One-line mnemonic: **divide = intra-tile coordinate + tile index, where the stride of the tile index is given by the complement**.

### 23. Shared model: `TileRest`

#### 23.1 Definition

`TileRest<T, R>` is the concatenated layout "tile address plus rest address". Its domain is the **Cartesian product** of the domains of `T` and `R` (all `(t, r)` pairs), and the map is

```text
TileRest(t, r) = T(t) + R(r)
size           = size(T) · size(R)
```

A linear index `i` is split into two digits in left-major mixed radix, with **the tile digit varying fastest**:

```text
t = i % size(T)
r = i / size(T)
index(i) = T.index(t) + R.index(r)
```

A small example (data from the contract tests): `T = (2,3):(1,8)`, `R = (2,2):(2,24)`, `size = 6 · 4 = 24`.

```text
i = 7  →  t = 7 % 6 = 1,  r = 7 / 6 = 1
T(1) = 1 (t decodes to (1,0)),  R(1) = 2 (r decodes to (1,0))
index(7) = 1 + 2 = 3
```

| Mathematical object | Rust | Location |
|---|---|---|
| `(t, r) ↦ T(t) + R(r)` | `TileRest<T, R>` and its `LayoutMap` implementation | [`src/algebra.rs`](../src/algebra.rs) |
| Parenthesization | `Grouping::{Logical, Zipped, Tiled, Flat}` | Same as above |
| `L ∘ TileRest` | `Divided<A, B> = Composition<A, TileRest<B, DynLayout>>` | [`src/algebra/divide.rs`](../src/algebra/divide.rs) |
| `TileRest(block, complement ∘ tiler)` | `Produced<A, B> = TileRest<A, Composition<DynLayout, B>>` | [`src/algebra/product.rs`](../src/algebra/product.rs) |

**Why it is done this way**: divide and product are both essentially "one tile address + one offset address". Making this a type of its own means each of the two families of algorithms only has to decide what the tile and the rest are.

#### 23.2 Four `Grouping`s: the same map, different parentheses

The leaf order of `TileRest` is always "tile leaves first, rest leaves after", and `index(i)` is likewise independent of the grouping. `Grouping` only decides **how the top-level modes are divided**, i.e. how many numbers are used to write a coordinate:

| `Grouping` | Top-level modes | Extent in the example above | Coordinate of address 3 |
|---|---|---|---|
| `Logical` | `(tile, rest)` | `(6, 4)` | `(1, 1)` |
| `Zipped` | `(tile, rest)` | `(6, 4)` | `(1, 1)` |
| `Tiled` | `(tile, rest_0, rest_1, …)` | `(6, 2, 2)` | `(1, 1, 0)` |
| `Flat` | `(tile_0, tile_1, …, rest_0, rest_1, …)` | `(2, 3, 2, 2)` | `(1, 0, 1, 0)` |

This is like adding different parentheses to the same sequence of digits `t0 t1 r0 r1`: `((t0 t1)(r0 r1))`, `((t0 t1) r0 r1)`, `(t0 t1 r0 r1)`. The linear index and the address stay the same; only "how the multidimensional coordinate is written" changes.

With a single tiler, `Logical` and `Zipped` are exactly the same. They differ only when the tiler is a **per-mode tuple** (section 25):

- logical groups by the original modes: `((t0, r0), (t1, r1))`; inside the m-th mode is "that mode's intra-tile coordinate, tile index";
- zipped gathers the tiles and the rests separately: `((t0, t1), (r0, r1))`.

These two forms are transposes of each other. CuTe calls the operation that turns "paired by mode" into "grouped by role" **unzip**, and the reverse **zip**:

```text
zip  ((a0, a1), (b0, b1)) = ((a0, b0), (a1, b1))
unzip((a0, b0), (a1, b1)) = ((a0, a1), (b0, b1))
```

That is, transposing a 2×n table into an n×2 one. CuTe's `zipped_divide` is `tile_unzip(logical_divide(...))`; `blocked_product` in Part V applies `zip` to the two groups of `logical_product`.

#### 23.3 Why the remaining methods are written this way

- **`index_extended`**: first try the affine form (`as_affine()`); when there is no affine form or the number of leaves/groups exceeds `MAX_RANK` (`Err(CapacityExceeded)`), take the component-wise fallback: if `rest` is the 0-leaf scalar `():()`, the last leaf belongs to `tile`, so delegate directly to `tile.index_extended(i)`; otherwise the tile digits are still reduced modulo, and only the slowest rest digit enters the extended domain. This is consistent with the convention that "the extended domain only extends the last leaf".
- **`address_range`**: the domain is a Cartesian product and the address is the sum of the two, so `min = min_T + min_R` and `max = max_T + max_R`; adding the endpoints term by term gives the **exact** interval, with no enumeration needed.
- **`as_affine`**: when both sides have an affine form, concatenate the tile leaves and the rest leaves in order, then write the top-level grouping according to `Grouping`.
- **`is_injective`**: with an affine form, use the stride-separation argument; otherwise (or when the leaves exceed `MAX_RANK`), fall back to `is_injective_by_enumeration`.

One-line mnemonic: **`TileRest` defines only one map; `Grouping` only changes how coordinates are written, not the addresses**.

### 24. Single tiler: `logical_divide`

```rust
if tile.size()? <= 0 {
    return Err(Error::InvalidShape);
}
let rest = match layout.as_affine()? {
    Some(affine) => complement_shape(tile.clone(), affine.flat().coalesce()?.shapes())?,
    None => complement(tile.clone(), layout.size()?)?,
};
let mut divided = composition(layout, TileRest::new(tile, rest, grouping))?;
// CuTe unpacks tiled/flat groups after composing, so group the composed tile and rest.
if divided.normalized.is_some()
    && let Some(affine) = divided.outer().as_affine()?
{
    let inner = divided.inner();
    let tiles = top_modes(&composition_layout(&affine, inner.tile())?)?;
    let rests = top_modes(&composition_layout(&affine, inner.rest())?)?;
    // A layout tiler's tile_unzip is the identity, so logical matches zipped.
    let grouping = match grouping {
        Grouping::Logical => Grouping::Zipped,
        other => other,
    };
    divided.normalized = Some(super::group_modes(&tiles, &rests, grouping)?);
}
Ok(divided)
```

Steps:

1. Reject an empty tile.
2. Compute rest: when `L` is affine the cotarget is `shape(coalesce(L))`; otherwise only `size(L)` is available.
3. Build `TileRest(T, rest)` in logical-index space and send it through `composition` into `L`.
4. When the result is normalized, regroup it the way CuTe does. Upstream `tiled_divide` and `flat_divide` slice the *composed* result (`result(_, repeat<R1>(_))`), so their unpacked modes are the top-level modes of `L ∘ T` and `L ∘ rest`, not the modes of `T` and `rest`. For example `tiled_divide((8,6):(1,16), 2:1)` composes the rest `24:2` into `(4,6):(2,16)`, so the result is `(2,4,6):(1,2,16)` with rank 3. A non-affine `L` keeps the modes of `TileRest`, because pointwise composition has no other structure.

The four public functions `logical_divide`, `zipped_divide`, `tiled_divide`, and `flat_divide` differ only in the final `Grouping` argument. The return value `Divided<A, B>` is a **lazy composed view**; call `.as_affine()` when a flat shape/stride is needed.

#### 24.1 Walking through `logical_divide(24:1, (2,2):(1,6))`

1. `coalesce(24:1) = 24:1`, cotarget `[24]`.
2. `complement_shape((2,2):(1,6), [24]) = (3,2):(2,12)`, by exactly the same process as in [section 10.6](#106-walking-through-l--2216).
3. inner = `TileRest((2,2):(1,6), (3,2):(2,12))`.
4. `L = 24:1` is the identity map, so the composition changes nothing, and the result is

```text
logical_divide(24:1, (2,2):(1,6)) = ((2,2),(3,2)) : ((1,6),(2,12))
```

Taking `.as_affine()` under each of the four `Grouping`s, the leaves are always `(2,2,3,2):(1,6,2,12)`, and the top-level groupings are `[2,2]`, `[2,2]`, `[2,1,1]`, and `[1,1,1,1]` respectively.

The address formula is `t0 + 6·t1 + 2·r0 + 12·r1`, which is exactly the diagram in section 22:

```text
i  = 0..3  → tile 0: 0, 1, 6, 7
i  = 4..7  → tile 1: 2, 3, 8, 9
…
i  = 20..23 → tile 5: 16, 17, 22, 23
```

The contract tests check these 24 addresses one by one, confirming that they cover `[0, 24)` exactly and without repetition.

#### 24.2 Why the cotarget is `shape(coalesce(L))` rather than `size(L)`

Take `L = (4,6):(6,1)` from [section 6](#6-illustration-one-a-three-bs) (4 rows, 6 columns, row-major) and `T = 3:1` (each tile is 3 consecutive rows within one column). 4 rows are not divisible by 3, so padding is needed.

**If `size(L) = 24` is used**: `complement(3:1, 24) = 8:3`. The composition `L ∘ 8:3` is exactly the example of [section 6.2](#62-non-divisible-stride-the-normalized-result-differs-from-pointwise-evaluation), with result `(2,4):(18,1)`: two row tiles × **4 columns**. Columns 4 and 5 are lost entirely.

**Using `shape(L) = [4,6]`**: by the leaf-by-leaf fold of [section 11.2](#112-why-the-result-differs-when-the-cotarget-is-a-tuple), `complement_shape(3:1, [4,6]) = 12:3`. The composition `L ∘ 12:3` takes `ceil(4/3) = 2` steps within mode 0 and discards the remainder, and the remaining 6 all go into the column mode, giving `(2,6):(18,1)`: two row tiles × **6 columns**.

```text
logical_divide((4,6):(6,1), 3:1) = (3, (2,6)) : (6, (18,1))      size = 36
```

The tile index of each cell (`*` rows are padded rows, addresses 24–35):

```text
      col0 col1 col2 col3 col4 col5
row0     0    2    4    6    8   10
row1     0    2    4    6    8   10
row2     0    2    4    6    8   10
row3     1    3    5    7    9   11
row4*    1    3    5    7    9   11
row5*    1    3    5    7    9   11
```

The addresses of the first 6 linear indices are `0, 6, 12, 18, 24, 30`: tile 0 is rows 0–2 of column 0, and tile 1 is rows 3–5 of column 0.

**Why `coalesce` first**: coalesce keeps mode boundaries only where `L` is genuinely non-contiguous, and padding happens only at these boundaries. `(4,6):(1,4)` (column-major) is 24 contiguous addresses and coalesces to `24:1`, so `T = 5:1` pads the whole range of 24 up to 25:

```text
logical_divide((4,6):(1,4), 5:1) = (5,5):(1,5)        // not padded per "4 rows per column"
logical_divide((4,6):(1,4), 4:2) = (4,(2,3)):(2,(1,8))
```

The rest in the second line is exactly `complement(4:2, 24) = (2,3):(1,8)`: `T` takes the even addresses, and the rest first fills in the odd ones (`2:1`), then repeats every 8 (`3:8`).

#### 24.3 Non-divisible: padding and the extended domain

```text
logical_divide(6:1, 4:1) = (4,2):(1,4)        size = 8, addresses 0..7
```

The tail of `complement_shape(4:1, [6])` is `ceil(6/4) = 2`, so the rest is `2:4` and the result has 8 elements. Addresses 6 and 7 exceed the logical domain of `L`; they fall in the extended domain of `L` (the last mode is not reduced modulo).

**Why the size is allowed to grow**: this is CuTe's semantics. After dividing, one usually slices by tile index; after padding, every tile has the same size, and the out-of-bounds part is masked by the user (e.g. by a predicated copy).

When `L` has no affine form (e.g. a lookup-table map), there is neither a shape to coalesce nor an extended domain:

- the cotarget can only be `size(L)`, i.e. `complement(T, size(L))`;
- it works normally when the division is exact: dividing the 6-value lookup table `[10,20,30,40,50,60]` by `3:1` gives rest `2:3`, and coordinate `(2,1)` corresponds to logical index `2 + 3 = 5`, with value 60;
- when it is not exact (`4:1`), the rest goes beyond `[0, 6)`, and the codomain check of `composition` returns `CoordinateOutOfBounds`.

#### 24.4 Lazy view and `as_affine`

`Divided<A, B>` is `Composition<A, TileRest<B, DynLayout>>`; construction only performs the codomain check and the normalization decision ([section 2](#2-construction-phase)).

- When `L` is affine, `as_affine()` returns the CuTe normalized layout, and `rank`/`extent` follow its regrouped top-level modes;
- When `L` is non-affine, it returns `None`, but pointwise evaluation is still possible.

**Why not return a `GroupedLayout` directly**: `L` can be any `LayoutMap` (e.g. swizzle), which has no flat form at all. The lazy view lets affine and non-affine inputs go through the same API.

One-line mnemonic: **rest = the complement padded according to the mode shapes of `L`; if not divisible, pad, and the out-of-bounds part goes into the extended domain**.

### 25. Per-mode tiler: `*_divide_modes`

A CuTe tiler can also be a tuple: the m-th entry is responsible only for dividing the m-th top-level mode of `L`. For example, dividing a matrix by `(4:1, 3:1)` means "a tile every 4 along the rows and every 3 along the columns". This project represents such a tuple as `&[DynLayout]` (an integer `n` is written `n:1`), corresponding to `logical_divide_modes`, `zipped_divide_modes`, `tiled_divide_modes`, and `flat_divide_modes`.

#### 25.1 Mechanism

The shared part `divide_modes` performs the division of section 24 on each mode separately:

```text
for each top-level mode mode_m of L:
    if the tiler has no m-th entry: mode_m goes into rests unchanged
    else:
        rest   = complement_shape(T_m, shape(coalesce(mode_m)))
        tile_m = mode_m ∘ T_m
        rest_m = mode_m ∘ rest
```

The results are collected in two `LayoutBuilder`s (section 13), one closed mode per input mode, and finished into two `GroupedLayout`s. `group_modes` then pairs or concatenates their modes for the requested grouping.

The four functions then just assemble `tiles` and `rests` in different ways:

| Function | Top-level structure |
|---|---|
| `logical_divide_modes` | `(tile_0 ∪ rest_0, tile_1 ∪ rest_1, …, uncovered modes…)` |
| `zipped_divide_modes` | `((tile_0, tile_1, …), (rest_0, rest_1, …, uncovered modes…))` |
| `tiled_divide_modes` | `((tile_0, tile_1, …), rest_0, rest_1, …, uncovered modes…)` |
| `flat_divide_modes` | `(tile_0, tile_1, …, rest_0, rest_1, …, uncovered modes…)` |

#### 25.2 Walking through `L = (8,6):(1,8)` with tiler `(4:1, 3:1)`

`L` is a column-major matrix with 8 rows and 6 columns, divided into 4×3 tiles, 2×2 tiles in total.

| Mode | `mode_m` | `T_m` | `tile_m = mode_m ∘ T_m` | rest | `rest_m = mode_m ∘ rest` |
|---|---|---|---|---|---|
| 0 (rows) | `8:1` | `4:1` | `4:1` | `complement_shape(4:1,[8]) = 2:4` | `2:4` |
| 1 (columns) | `6:8` | `3:1` | `3:8` | `complement_shape(3:1,[6]) = 2:3` | `2:24` |

Assembled results (data from the contract tests):

```text
logical : ((4,2),(3,2)) : ((1,4),(8,24))     grouping [2,2]
zipped  : ((4,3),(2,2)) : ((1,8),(4,24))     grouping [2,2]
tiled   : ((4,3), 2, 2) : ((1,8), 4, 24)     grouping [2,1,1]
flat    : (4,3,2,2)     : (1,8,4,24)         grouping [1,1,1,1]
```

logical and zipped have exactly the transpose relationship of section 23.2: mode 0 of logical is "(row within tile, tile row index) along the rows", while mode 0 of zipped is "(row, column) within the tile". With zipped, fixing the second coordinate (the tile index) yields an entire 4×3 tile.

Non-divisible cases are padded likewise: dividing `(6,5):(1,6)` by `(4,2)` gives the zipped result `((4,2),(2,3)):((1,6),(4,12))`: 6 rows → `2` row tiles (padded to 8), 5 columns → `3` column tiles (padded to 6).

#### 25.3 Tiler shorter than the rank

`L = (4,6,2):(1,4,24)`, tiler `(2:1, 3:1)` covers only the first two modes:

```text
logical : ((2,2),(3,2),2) : ((1,2),(4,12),24)      grouping [2,2,1]
zipped  : ((2,3),(2,2,2)) : ((1,4),(2,12,24))      grouping [2,3]
```

Mode 2, `2:24`, is not divided and is appended unchanged at the end of the rests. In zipped it belongs to the "tile index" group. This is consistent with CuTe's behavior.

#### 25.4 Difference in the runtime GroupedLayout: only one level of grouping is kept

With a tuple tiler, CuTe's `logical_divide` returns a nested layout of depth 2, `((t0, r0), (t1, r1))`. `GroupedLayout` has only one level of grouping, so the logical version can only merge `(tile_m, rest_m)` into one top-level mode (grouping `[2,2]`), and **the boundary between tile and rest inside each mode is no longer recorded**.

When this boundary is needed, use the typed entry points that keep the full profile (section 27.1), or switch to the zipped / tiled / flat runtime groupings.

One-line mnemonic: **per-mode tiler = one single-tiler division per mode, then assembled in one of four ways**.

### 26. Errors and edge cases

| Input | Returns | Source |
|---|---|---|
| tile size ≤ 0 (e.g. `0:1`) | `Err(InvalidShape)` | `divide` entry point |
| tile contains a negative stride (e.g. `2:-1`) | `Err(Unsupported)` | `complement` |
| tile overlaps itself (e.g. `(2,2):(2,3)`) | `Err(NonInjective)` | `complement` |
| `L` is non-affine and the tile does not divide `size(L)` | `Err(CoordinateOutOfBounds)` | `composition` codomain check |
| `_modes`: tiler longer than the rank | `Err(StructureMismatch)` | `divide_modes` |
| `_modes`: the composition of some mode violates the divisibility condition | `Err(NotDivisible)` | `composition_layout` |
| Number of leaves or modes exceeds `MAX_RANK` | `Err(CapacityExceeded)` | `LayoutBuilder` |
| `L` is empty (e.g. `0:1`) | Result has size 0 | |

In most of these places CuTe has a static assertion or undefined behavior. This project uniformly returns a `Result` instead.

### 27. Correspondence with upstream and flow overview

| CuTe (`include/cute/layout.hpp`) | noodle-cute |
|---|---|
| `logical_divide(layout, tiler)`, tiler is a single layout | `logical_divide`, returns a lazy `Divided` |
| Same as above, tiler is a tuple (per-mode `transform_layout`) | `logical_divide_modes`; the result keeps only one level of grouping |
| `zipped_divide = tile_unzip(logical_divide(...))` | `zipped_divide` / `zipped_divide_modes` |
| `tiled_divide = zipped_divide(...)(_, repeat<R1>(_))` | `tiled_divide` / `tiled_divide_modes` |
| `flat_divide = zipped_divide(...)(repeat<R0>(_), repeat<R1>(_))` | `flat_divide` / `flat_divide_modes` |

```text
                      logical_divide(L, T)
                               │
                       size(T) ≤ 0 ? ── yes ──▶ InvalidShape
                               │
                   Does L have an affine form?
              ┌────────────────┴────────────────┐
             yes                               no
 rest = complement_shape(T,            rest = complement(T, size(L))
         shape(coalesce(L)))                     │
              └────────────────┬────────────────┘
                               ▼
                 inner = TileRest(T, rest, grouping)
                               ▼
                    composition(L, inner)  ──▶  Divided (lazy)
                               │
             L affine: regroup L∘T and L∘rest by Grouping
                               ▼
              flat leaves + top-level groups (.as_affine())
```

One-line mnemonic: **first arrange (intra-tile, tile index) in logical-index space, then feed the whole thing into `L`**.

#### 27.1 Fully static, mixed, and runtime entry points for divide

| Input and entry point | Result |
|---|---|
| `<TypedDivided<L, T, KIND> as StaticValue>::VALUE` | Full-hierarchy `Layout` constant for fully static inputs |
| `logical_divide` / `zipped_divide` / `tiled_divide` / `flat_divide` methods of `Layout` | `Result<TypedDivided<L,T,KIND>>`, keeping static leaves and dynamic fields |
| Free functions `*_divide` and `*_divide_modes` | Single-layout tilers (lazy `Divided`, with the pointwise fallback for non-affine inputs) and `&[DynLayout]` per-mode tilers |

`KIND` is one of `algebra::LOGICAL`, `ZIPPED`, `TILED`, or `FLAT`.

When a single typed tiler is a `Layout`, `DivideOp` fully reuses `composition(L, (T, complement(T, shape(coalesce(L)))))`: `CoalesceTarget` obtains the typed cotarget, complement keeps the static derivation, and composition handles weak divisibility and the extended domain. An integer tiler is automatically treated as `n:1`. When the composition fails a divisibility check, the typed entry point returns `NotDivisible` directly and does not build a lazy fallback.

A nested tuple tiler uses `Transform` from [`typed.rs`](../src/algebra/typed.rs) to recursively process modes with the same profile; a short tuple keeps the trailing modes, and an empty tuple returns the original layout. `Unzip` corresponds to upstream `zip2_by`: it collects tile/rest at the tiler's terminals, and uncovered modes go only into rest. `Regroup` only unpacks the top-level modes without flattening the inner levels; therefore `(tile,rest)` for an integer tiler differs from `((tile),(rest))` for a single-element tuple tiler. All of this structure is kept in the types and is not subject to the one-level grouping limit of `GroupedLayout`.

The public methods validate the input size and the final `Layout::new`; reading a constant through `StaticValue` skips the final runtime size and address-range checks. Only the runtime array representation is limited by `MAX_RANK`. The existing dynamic grouping assembly has been consolidated into `group_modes`, which is also reused by the runtime tuple product.

[`tests/divide_contract.rs`](../tests/divide_contract.rs) contains the migrated dynamic contracts, as well as the four static groupings, mixed metadata, nested/partial/empty tuples, the difference between integer and singleton profiles, the extended domain, dynamic errors, and results exceeding `MAX_RANK`. Compile-time rejection examples are in the module rustdoc.

## Part V · Product `product`

This part explains `logical_product` / `zipped_product` / `tiled_product` / `flat_product` and `blocked_product` / `raked_product` in [`src/algebra/product.rs`](../src/algebra/product.rs).

In this part, the small layout being repeated is written `block`, and the layout describing "how to repeat" is written `tiler`.

### 28. What a product does

Dividing is "cutting something large into small pieces"; a product is the reverse: **treat a block as a tile and lay many copies of it following the pattern of the tiler**.

- block describes the addresses inside one tile;
- tiler describes how "the n-th tile" is arranged;
- the starting point of each tile must fall on an address the block does not use, otherwise they would overlap.

"The positions the block does not use, where another copy can be placed" are again the complement. Hence

```text
logical_product(block, tiler) = (block, complement(block) ∘ tiler)
```

The second term reads: the tiler produces "the n-th free slot", and `complement(block)` translates it into the actual starting address. The result is still a `TileRest`:

```rust
let complement = complement_extended(block.clone())?;
let rest = composition(complement, tiler)?;
let result = TileRest::new(block, rest, grouping);
result.size()?;
Ok(result)
```

`Produced<A, B> = TileRest<A, Composition<DynLayout, B>>`. The final `result.size()?` checks in advance whether `size(block) · size(tiler)` overflows. The block must have an affine form (it has to be complemented), while the tiler can be any `LayoutMap`.

### 29. Why `complement_extended` (without a cotarget)

divide knows how large the target is (`shape(L)`); product does not: how many copies are laid is determined entirely by the tiler, whose values can be arbitrarily large. So the complement here **cannot have a cotarget**, and must instead be able to extend indefinitely.

The `complement_extended` of [section 12](#12-complement_extended-marking-the-repetition-stride) is designed exactly for this: it keeps the gaps inside the block and then appends a trailing leaf `1:current`. This trailing leaf has size 1 and does not change the logical size, but in the extended domain it provides the stride of "the next whole segment". When the tiler's values go beyond the logical size of the complement, placement continues along this stride.

**Counterexample: what if `1:current` were dropped**. Take `block = (2,2):(1,4)`, with addresses `{0, 1, 4, 5}`:

- the correct complement is `(2,1):(2,8)`, which in the extended domain yields `0, 2, 8, 10, 16, 18, …` in order;
- if it were coalesced to `2:2`, the extended domain would extend as `0, 2, 4, 6, …`, and the third tile would start at address 4, overlapping the block's own `{4, 5}`.

This is also why `composition` in [section 3.1](#31-compose_grouped) applies `coalesce_extended` rather than ordinary `coalesce` to A: the stride of a trailing leaf with shape 1 must be kept.

**Relationship with upstream**: CuTe's `logical_product` uses `complement(block)` without a cotarget, which is equivalent to `complement(block, cosize(block))`. The cosize does not exceed `current`, so the tail `ceil_div(cosize, current)` is exactly 1, leaving only a single `1:current` leaf, consistent with this project's `complement_extended`.

### 30. Walking through `logical_product((2,2):(1,4), (2,3):(1,2))`

View the addresses as a column-major matrix with 4 rows (`a = row + 4·col`). The block occupies rows 0–1 and columns 0–1; the tiler is a 2×3 column-major arrangement.

**Step 1: `complement_extended(block)`**

| Step | Leaf | `current` on entry | `gap = d / current` | Action | `current` on exit |
|---|---|---|---|---|---|
| 1 | `2:1` | 1 | 1 | No gap, skip | 2 |
| 2 | `2:4` | 2 | 2 | push `2:2` | 8 |

The size of the prefix `2:2` is not 1, so `1:8` is appended: `complement_extended = (2,1):(2,8)`.

**Step 2: `(2,1):(2,8) ∘ (2,3):(1,2)`**, composed leaf by leaf as in [section 3.2](#32-compose_leaf):

```text
modes of A: mode 0 (cs=2, cd=2), last mode (1, cd=8)

leaf 2:1: mode 0: next_shape = ceil(2/1) = 2, new_shape = min(2, 2) = 2
          emit 2 : 1·2 = 2:2, rest_shape = 1, B is exhausted
leaf 3:2: mode 0: next_shape = ceil(2/2) = 1, the stride spans the whole mode, only update rest_stride = 1
          no leaf emitted yet → put the remaining 3 into the last mode: 3 : 1·8 = 3:8
rest = (2,3):(2,8)
```

**Result**:

```text
logical_product((2,2):(1,4), (2,3):(1,2)) = ((2,2),(2,3)) : ((1,4),(2,8))
```

The starting points of the six tiles are `2·r0 + 8·r1 = 0, 2, 8, 10, 16, 18`. The tile index of each cell:

```text
      col0 col1 col2 col3 col4 col5
row0     0    0    2    2    4    4
row1     0    0    2    2    4    4
row2     1    1    3    3    5    5
row3     1    1    3    3    5    5
```

In linear traversal the block coordinate varies fastest; the address sequence in the contract tests is:

```text
0,1,4,5,  2,3,6,7,  8,9,12,13,  10,11,14,15,  16,17,20,21,  18,19,22,23
```

The 24 addresses tile `[0, 24)` exactly. Note that tiler copy number 2 (index 2) goes beyond the logical size 2 of the complement, and it is precisely the trailing leaf `1:8` that sends it to address 8.

### 31. More examples

The following data come from the contract tests (generated by a CuTe oracle):

```text
block       tiler           logical_product
4:2         (2,2):(1,2)     (4,(2,2)):(2,(1,8))     // complement (2,1):(1,8): fill the odd addresses first, then repeat every 8
3:1         4:2             (3,4):(1,6)             // complement 1:3; the tiler takes every other free slot
(2,2):(2,1) 3:1             ((2,2),3):((2,1),4)     // the block densely covers [0,4), complement 1:4
4:0         3:1             (4,3):(0,1)             // all-broadcast block, complement 1:1
```

The second line deserves a closer look: `complement_extended(3:1) = 1:3` means "a copy can be placed every 3 addresses". The tiler `4:2` takes free slots 0, 2, 4, and 6, so the starting points are `0, 6, 12, 18`, leaving a gap of one copy in between.

The tiler need not be an affine layout. If the tiler is a lookup-table map with values `[0, 2, 1]`, the starting points of the three tiles are the values of the complement at 0, 2, and 1, namely `0, 8, 2`; the result can only be evaluated lazily (`as_affine()` is `None`), and the address sequence is `0,1,4,5, 8,9,12,13, 2,3,6,7`.

### 32. `zipped_product` / `tiled_product` / `flat_product`

As with divide, the four product functions share the same `TileRest` and differ only in `Grouping`. Taking section 30 as the example, the leaves are always `(2,2,2,3):(1,4,2,8)`:

| Function | Top-level grouping | Coordinate form |
|---|---|---|
| `logical_product` | `[2,2]` | `(block coordinate, tile index)` |
| `zipped_product` | `[2,2]` | Same as above (identical to logical for a single tiler) |
| `tiled_product` | `[2,1,1]` | `(block coordinate, tiler axis 0, tiler axis 1)` |
| `flat_product` | `[1,1,1,1]` | Fully unpacked |

`tiled_product` unpacks the top-level modes of the composed rest `complement(block) ∘ tiler`, as upstream `result(_, repeat<R1>(_))` does. A tuple tiler keeps one mode per tiler axis; an integral tiler yields one mode per leaf of the composition. For example `complement(2:2) = (2,1):(1,4)` composed with `6:1` is `(2,3):(1,4)`, so `tiled_product(2:2, 6:1) = (2,2,3):(2,1,4)` with rank 3. The lazy `Composition` reports this structure through its normalized layout.

### 33. `blocked_product` and `raked_product`

The result of `logical_product` is `(block coordinate, tile index)`, not the `(row, col)` of one "large matrix". When both the block and the tiler are two-dimensional, one usually wants a two-dimensional result as well: mode 0 is "the rows of the large matrix", made up of the block's rows and the tiler's rows; mode 1 likewise. This is applying the **zip** of section 23.2 to the two groups:

```text
logical : ((b0, b1), (r0, r1))
zip     : ((b0, r0), (b1, r1))      ← blocked_product
          ((r0, b0), (r1, b1))      ← raked_product
```

The two have exactly the same leaves; they differ only in which comes first (which varies faster) inside each mode.

#### 33.1 A small one-dimensional example

`block = 3:1`, `tiler = 2:1`. `complement_extended(3:1) = 1:3`, `rest = 1:3 ∘ 2:1 = 2:3`.

```text
blocked = (3:1, 2:3) = (3,2):(1,3)    positions 0..5 → addresses 0,1,2,3,4,5
raked   = (2:3, 3:1) = (2,3):(3,1)    positions 0..5 → addresses 0,3,1,4,2,5
```

Address = intra-tile index + 3·tile index. Which tile each position belongs to:

```text
blocked: 0 0 0 1 1 1      // each tile forms one contiguous segment
raked  : 0 1 0 1 0 1      // the two tiles are interleaved
```

"Raked" refers to a rake: the elements of a tile are spread out evenly like the teeth of a rake, instead of forming one contiguous block.

#### 33.2 Mechanism: `rank_product`

```rust
let rank = block.rank().max(tiler.rank());
let complement = complement_extended(block)?;
for m in 0..rank {
    let b = if m < block.rank() { block.mode(m)? } else { 1:0 };
    let r = if m < tiler.rank() { complement ∘ tiler.mode(m) } else { 1:0 };
    let (first, second) = if raked { (r, b) } else { (b, r) };
    // push the leaves of first, then second; leaf count of top-level mode m = sum of the two leaf counts
}
```

(Pseudocode, error handling omitted.) Key points:

1. **The complement is computed only once**, for the whole block.
2. **Mode-by-mode composition**: `complement ∘ (T0, T1, …) = (complement ∘ T0, complement ∘ T1, …)`, i.e. the right-distributive law of [section 3.1](#31-compose_grouped), so it can be computed mode by mode.
3. **Pad with `1:0` when the ranks differ**: CuTe uses `append<R>` to pad both sides to the same rank, and the padded modes are `1:0`. They have no effect on the complement (`complement` filters them out) and are still `1:0` after composition. The result **keeps** these shape-1 leaves, so that the mode structure matches CuTe.
4. **Pad before composing**: the typed `blocked_product`/`raked_product` also wrap the tiler into modes and pad it with `1:0` to the block's rank *before* composing, matching upstream `logical_product(append<R>(block), append<R>(tiler))`. A tiler mode whose composition has several modes therefore stays nested inside one result mode: `blocked_product((2,2):(2,8), 8:1) = ((2,(2,2,2)),(2,1)) : ((2,(1,4,16)),(8,0))`.
5. **Negative tiler strides** compose through `signum` like any other composition, e.g. `blocked_product(2:1, 2:-1) = ((2,2)):((1,-2))`.

#### 33.3 Two-dimensional example: `col = (2,2):(1,2)`, `t23 = (2,3):(1,2)`

`col` densely covers `[0, 4)`, and `complement_extended(col) = 1:4`.

| Mode | `b_m` | `r_m = 1:4 ∘ t23_m` | blocked `(b_m, r_m)` | raked `(r_m, b_m)` |
|---|---|---|---|---|
| 0 | `2:1` | `2:4` | `(2,2):(1,4)` | `(2,2):(4,1)` |
| 1 | `2:2` | `3:8` | `(2,3):(2,8)` | `(3,2):(8,2)` |

```text
blocked_product(col, t23) = ((2,2),(2,3)) : ((1,4),(2,8))      extent (4, 6)
raked_product  (col, t23) = ((2,2),(3,2)) : ((4,1),(8,2))      extent (4, 6)
```

Both can be viewed as a large matrix with 4 rows and 6 columns, with each address = intra-tile address + 4·tile index. The diagram below shows which tile each position `(row, col)` belongs to:

```text
blocked                         raked
      0  1  2  3  4  5                0  1  2  3  4  5
row0  0  0  2  2  4  4          row0  0  2  4  0  2  4
row1  0  0  2  2  4  4          row1  1  3  5  1  3  5
row2  1  1  3  3  5  5          row2  0  2  4  0  2  4
row3  1  1  3  3  5  5          row3  1  3  5  1  3  5
```

In blocked, each tile is a contiguous 2×2 square; in raked, the four elements of tile 0 are at `(0,0)`, `(0,3)`, `(2,0)`, and `(2,3)`, spread out evenly. For the same coordinate `(1,0)`: blocked gives address 1 (element 1 of tile 0), and raked gives address 4 (element 0 of tile 1).

Note that the leaves of blocked, `(2,2,2,3):(1,4,2,8)`, are exactly the same as the leaves of `logical_product` in section 30, but the meaning differs: here mode 0 is "the rows of the large matrix", while in section 30 mode 0 is "the block coordinate".

#### 33.4 Examples with different ranks

```text
blocked_product(4:1, t23) = ((4,2),(1,3)) : ((1,4),(0,8))
raked_product  (4:1, t23) = ((2,4),(3,1)) : ((4,1),(8,0))
blocked_product(col, 3:1) = ((2,3),(2,1)) : ((1,4),(2,0))
raked_product  (col, 3:1) = ((3,2),(1,2)) : ((4,1),(0,2))
```

Missing modes are padded with `1:0` and left in the result as is.

#### 33.5 Application: thread-value (TV) layouts

CuTe's `make_tiled_copy` uses `raked_product(thr, val)` to describe "in the rectangular region copied by a thread block, which value of which thread each position belongs to":

- `thr` maps thread coordinates to thread indices, and `val` maps value coordinates to value indices;
- the stride of `complement_extended(thr)` is exactly `size(thr)`, so the result maps positions in the rectangle to `thread index + size(thr)·value index`;
- applying the `right_inverse` of [section 19](#19-right_inverse-greedily-chaining-a-contiguous-address-prefix) to it then gives the reverse direction `(thread, value) → position`, i.e. the TV layout.

The example in the contract tests: `thr = (4,2):(2,1)`, `val = (2,2):(1,2)`; `raked_product` is `((2,4),(2,2)):((8,2),(16,1))`, and the final TV layout is `((2,4),(2,2)):((16,2),(1,8))`. The test checks one by one that each `(thread, value)`, passed through the TV layout and then through `raked_product`, returns to `thread index + 8·value index`.

### 34. Errors and edge cases

| Input | Returns | Source |
|---|---|---|
| block contains a negative stride (e.g. `2:-1`) | `Err(Unsupported)` | `complement_extended` |
| block is empty (e.g. `0:1`) | `Err(Unsupported)` | `complement_extended` |
| block overlaps itself (e.g. `(2,2):(2,3)`) | `Err(NonInjective)` | `complement_extended` |
| A non-affine tiler produces negative values | `Err(CoordinateOutOfBounds)` | pointwise `composition` fallback |
| `size(block) · size(tiler)` overflows | `Err(Overflow)` | `result.size()?` at the end of `product` |
| blocked/raked: the composition of some mode is not divisible (e.g. `blocked_product(4:2, 2:3)`) | `Err(NotDivisible)` | `composition_layout` |
| Leaves exceed `MAX_RANK` (e.g. two layouts of rank 17) | `Err(CapacityExceeded)` | `LayoutBuilder` |

Why `blocked_product(4:2, 2:3)` fails: mode 0 of the complement `(2,1):(1,8)` has size 2, and the tiler stride 3 is greater than 2 but not a multiple of 2, which triggers the stride divisibility condition of [section 3.2](#32-compose_leaf); in CuTe this is a failed divisibility assertion.

### 35. Correspondence with upstream and relationship with divide

| CuTe (`include/cute/layout.hpp`) | noodle-cute |
|---|---|
| `logical_product = make_layout(block, composition(complement(block), tiler))` | `logical_product`, returns a lazy `Product`; the complement uses `complement_extended` |
| `zipped_product = tile_unzip(logical_product(...))` | `zipped_product` (identical to logical for a single tiler) |
| `tiled_product`, `flat_product` | Functions of the same names, changing only `Grouping` |
| `blocked_product = zip(get<0>(r), get<1>(r))`, `r = logical_product(append<R>(block), append<R>(tiler))` | `blocked_product`, eagerly returns a `GroupedLayout` with one level of grouping |
| `raked_product = zip(get<1>(r), get<0>(r))` | `raked_product`, same as above |

Deviations:

- A single-tiler product returns a lazy `TileRest`, and the tiler may be non-affine; call `.as_affine()` when a flat layout is needed.
- blocked/raked require both sides to be affine layouts; the result keeps only one level of grouping, and the boundary between block and rest inside each mode is no longer recorded.
- CuTe's assertions and undefined behavior become `Result` errors (section 34).

**divide and product are inverse operations of each other**. The contract tests verify:

```text
logical_product((2,2):(1,4), 6:1) = logical_divide(24:1, (2,2):(1,4)) = ((2,2),(2,3)) : ((1,4),(2,8))
```

The two sides use different algorithms but compute the same thing:

- in divide, rest is `complement_shape(block, [24]) = (2,3):(2,8)`;
- in product, rest is `complement_extended(block) ∘ 6:1 = (2,1):(2,8) ∘ 6:1 = (2,3):(2,8)`.

The former starts from "24 in total" and works out how many copies are needed; the latter starts from "6 copies are wanted" and lays out 24. This layout is a bijection on `[0, 24)`; `left_inverse` and `right_inverse` are equal, and both can recover it.

One-line mnemonic: **product = block + complement ∘ tiler; blocked keeps each tile contiguous, raked interleaves the tiles**.

#### 35.1 Fully static, mixed, and runtime entry points for product

| Input and entry point | Result and grouping |
|---|---|
| `<TypedProductLayout<L, T, KIND> as StaticValue>::VALUE` | Zero-sized `Layout` constant keeping the full profile |
| The six methods of the same names on `Layout` | `Result<TypedProductLayout<L,T,KIND>>` |
| Free functions `logical/zipped/tiled/flat_product`, `blocked_product`, `raked_product` | Single tilers keep the lazy `Produced`; blocked/raked return a `GroupedLayout` |
| `*_product_modes(&block, &[DynLayout])` | Each covered mode is complemented and composed independently; uncovered trailing modes are kept |

`KIND` is one of `algebra::LOGICAL`, `ZIPPED`, `TILED`, `FLAT`, `BLOCKED`, or `RAKED`. The last two take a `Layout` tiler; the other four also accept integers and nested tuples.

`ProductOp` reuses the typed complement without a cotarget and the typed composition, avoiding mistakenly replacing the repetition stride with the cosize. Tuple traversal and `tile_unzip` share `Transform` / `Unzip` / `Regroup` with divide, so logical keeps the block/rest boundary inside each mode, and zipped keeps the nested guide when transposing. An empty tuple is a tiler that performs no transformation, whereas `Layout<Nil,Nil>` is an actual scalar layout tiler; the two have different profiles.

blocked/raked first wrap the tiler into modes and pad it with `1:0` to the block's rank (`PadModes`), compose, and then apply `IntoModes` and `ZipModes` to the two groups. Each missing mode on the shorter side is padded with `1:0`; these unit axes are kept in the final profile. In blocked each mode is `(block_m,rest_m)`, and raked swaps the two; when both sides are empty-tuple layouts, the result is still a rank-0 scalar. The zip here performs no additional coalesce.

The typed value entry points check the size and address range of the final layout; failures in the complement or composition return the corresponding error directly, and statically known illegal operations are rejected by type constraints. Reading a constant through `StaticValue` skips the final runtime validation. The general runtime product still keeps the lazy fallback for non-divisible or non-affine tilers. `*_product_modes` handles only affine modes, reports composition failures directly, and its runtime output keeps one level of grouping.

[`tests/product_contract.rs`](../tests/product_contract.rs) keeps the existing product, TV-layout, and capacity-boundary contracts, and adds the six static groupings, mixed metadata, nested/partial/empty tilers, the order of padded axes, the runtime tuple product, zero-domain/overflow/divisibility errors, and typed outputs exceeding `MAX_RANK`. The cross-module divide/product/inverse round-trip laws remain covered by [`algebra_contract.rs`](../tests/algebra_contract.rs).

#### Self-check

What are the addresses at position 4 of `blocked_product(3:1, 2:1)` and `raked_product(3:1, 2:1)`, respectively?

Answer: `blocked = (3,2):(1,3)`; position 4 decomposes into `(1, 1)`, address `1 + 3 = 4`. `raked = (2,3):(3,1)`; position 4 decomposes into `(0, 2)`, address `0 + 2 = 2`.
