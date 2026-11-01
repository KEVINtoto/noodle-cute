# 布局代数的实现：`composition`、`complement`、`inverse`、`divide` 与 `product`

本文前三部分讲解布局代数的三个核心算法，它们一起构成了分块、乘积、分区、线程分工等所有上层操作的底座；后两部分讲解直接建在它们之上的分块与乘积：

- **第一部分 · 布局组合 `composition`**：怎样构造组合视图，怎样决定按 CuTe 规范化布局求值还是逐点求值，以及 `Composition::normalized` 字段在什么条件下被填充。对应 [`src/algebra/composition.rs`](../src/algebra/composition.rs)；fully static results (`TypedComposition` read through `StaticValue`) are covered in 7.1, and the entry points for each kind of input in 7.3.
- **第二部分 · 补布局 `complement`**：怎样求出一个布局留下的"空隙"，以及 `complement_shape` 与 `complement_extended` 的区别。对应 [`src/algebra/complement.rs`](../src/algebra/complement.rs)。
- **第三部分 · 求逆 `inverse`**：怎样从地址反推逻辑下标，`right_inverse` 与 `left_inverse` 为什么对"空洞"给出完全不同的答案，以及它们与枚举逆 `dense_*_inverse` 的分工。对应 [`src/algebra/inverse.rs`](../src/algebra/inverse.rs)。
- **第四部分 · 分块 `divide`**：怎样把一个布局切成"块内坐标 × 块号"，四种 `Grouping` 为什么只是同一映射的不同括号方式，以及单 tiler 与逐模式 tiler（`*_divide_modes`）两条路线。对应 [`src/algebra/divide.rs`](../src/algebra/divide.rs) 与 [`src/algebra.rs`](../src/algebra.rs) 中的 `TileRest`。
- **第五部分 · 乘积 `product`**：怎样按 tiler 的样式重复铺一个 block，为什么用不带 cotarget 的 `complement_extended`，以及 `blocked_product` 与 `raked_product` 的区别。对应 [`src/algebra/product.rs`](../src/algebra/product.rs)。

`composition` 与 `complement` 的关系可以用一行公式概括，它也正是第四部分 `logical_divide` 的定义（A 为仿射布局时）：

```text
logical_divide(A, B) = composition(A, (B, complement(B, shape(coalesce(A)))))
```

数学背景见[数学模型](mathematical-model.md)，整体工程结构见[工程实现](engineering-implementation.md)。

全文使用 `Shape:Stride` 布局记号，例如 `8:3` 表示长度为 8、步长为 3 的一维布局。

## 第一部分 · 布局组合 `composition`

本部分记 `outer` 为 A、`inner` 为 B。

### 1. 概览

`composition(A, B)` 返回一个惰性视图 `R`，满足

```text
R(c) = A(B(c))
```

构造时不计算任何映射，只做两件事：

1. **值域检查**：B 的输出能否作为 A 的输入求值。
2. **规范化判定**：两侧都是仿射布局时，用 CuTe 的 `composition` 算法算出规范化布局，并缓存到 `normalized` 字段；若 CuTe 在此处会断言失败（`NotDivisible`），则不缓存。

求值时，有缓存就查缓存布局，否则逐点计算 `A(B(c))`。

### 2. 构造阶段

#### 2.1 值域检查与规范化判定

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
| `Err(NotDivisible)` | CuTe 会断言失败 | 逐点求值在数学上仍有定义，退回逐点 |
| 其他 `Err` | 溢出、`CapacityExceeded` 等 | 向上返回 |
| 任一侧非仿射 | 对应 CuTe 的 `ComposedLayout` | 逐点求值 |

传入的是 `a.flat()`，A 的分组会被丢弃，因为 A 只接收线性下标，层级不影响结果。B 保留顶层分组，因为结果的形状来自 B。

### 3. 核心算法

#### 3.1 `compose_grouped`

1. 对 A 做 `coalesce_extended`，即 CuTe 的 `coalesce_x`。它与普通 `coalesce` 的区别在于保留末尾 shape 为 1 的叶子的步长，因为扩展域要沿这个步长延伸。
2. B 为整数形状时，直接对它唯一的叶子调用 `compose_leaf`，结果按扁平布局返回。
3. B 为 tuple 时，利用右分配律 `A ∘ (B0, B1, …) = (A∘B0, A∘B1, …)`，对 B 的每个叶子调用 `compose_leaf`，并记录每个顶层组产出的叶子数。这样结果的顶层分组与 B 一致。

#### 3.2 `compose_leaf`

把 B 的一个叶子 `s:d` 与 coalesce 后的 A 组合，做法是让步长依次穿过 A 的各个模式：

```text
rest_shape = s, rest_stride = d
对 A 的前 r-1 个模式 (cs, cd)：
    next_shape  = ceil(cs / |rest_stride|)    // 在该模式内走几步
    next_stride = ceil(|rest_stride| / cs) * sign(rest_stride)   // 换算到下一模式的步长
    若 next_shape == 1（步长跨过整个模式）或 rest_shape == 1（B 已走完）：
        只更新 rest_stride
    否则：
        new_shape = min(next_shape, rest_shape)
        产出叶子 new_shape : rest_stride * cd
        rest_shape /= new_shape; rest_stride = next_stride
若还没产出任何叶子，或 rest_shape != 1：
    把剩余的 rest_shape 放入 A 的最后一个模式（扩展域，不取模）
```

特殊情况和检查：

- `d == 0 || s == 0`：B 不读取 A，直接得到 `s:0`。
- A 只有一个模式：得到 `s : d*stride0`，扩展域自然延伸。
- **stride 整除条件**：步长大于模式大小但不能整除它时，返回 `NotDivisible`。
- **shape 整除条件**：`rest_shape` 不能被 `new_shape` 整除时，返回 `NotDivisible`。
- **弱整除行为**：`cs % |stride| != 0`，且 B 在该模式内还没走完（`new_shape < rest_shape`）时，`compose_leaf` 并不报错，而是照 CuTe 的做法丢弃余数、在下一模式重新起步。这正是 CuTe 结果与逐点求值出现分歧的地方（见第 5、6 节），代码不再单独记录分歧标志，规范化结果一律缓存。

### 4. 求值阶段：`impl LayoutMap for Composition`

- `size` / `rank` / `extent`：都来自 inner，输出形状就是 B 的形状。
- `index(i)`：有缓存时返回 `normalized.index(i)`，否则返回 `outer.index_extended(inner.index(i)?)`。outer 这里用的是 `index_extended`，因为 B 的值可能超出 A 的域。
- `index_extended(i)` / `index_extended_of_coord(coord)`：与 `index` 相同，有缓存时查缓存布局，否则 inner 也走对应的扩展域求值。
- `is_injective`：有缓存时直接询问缓存布局。否则，仅当 A 能通过廉价的 $O(\text{叶子数}^2)$ 仿射步长分离证明（`affine_injective == Some(true)`）且 B 的值都在 A 的逻辑域内时，直接委托给 `B.is_injective()`；若 A 非仿射、步长分离证明无法判定（避免对巨大的 A 做 $O(\text{size}(A)^2)$ 枚举），或 B 进入 A 的扩展域，则退回 `is_injective_by_enumeration(self)` 枚举组合域，复杂度为 $O(\text{size}(B)^2)$。
- `address_range`：有缓存时直接委托给缓存的 `GroupedLayout`（O(叶子数) 精确计算端点）；没有缓存时退回遍历 `0..size` 求最小/最大地址。
- `as_affine`：直接返回缓存的 `normalized`。`None` 表示“可以逐点求值，但没有仿射形式”（任一侧非仿射，或遇到 `NotDivisible`）。

### 5. `normalized` 字段

```rust
// Cached CuTe normalized layout when both sides are affine and divisible.
pub(crate) normalized: Option<GroupedLayout>,
```

这句注释的主干是“……时缓存的规范化布局”，前半句是条件：

> 【两侧均为仿射布局】且【满足 CuTe 整除条件（不是 `NotDivisible`）】时，被缓存下来的规范化布局。

只有两个条件同时满足，字段才是 `Some`。

- **仿射布局**：能用一组扁平 shape/stride 表示，即 `as_affine()` 返回 `Some`。swizzle 这类映射不是仿射布局。
- **CuTe 规范化结果**：由 `compose_grouped` 直接算出的新布局 `R`。
- **逐点求值**：不生成新布局，每次先算 `B(i)`，再交给 A。

缓存的原因：规范化结果与 CuTe 的 `Layout∘Layout` 一致，补齐的 `logical_divide` 依赖它；它在 A 的逻辑域内通常与逐点求值相同，只在弱整除条件下不同（见第 6 节），因此以缓存布局为准。

需要注意两点：

1. **`None` 不一定表示组合没有意义**。它表示“任一侧非仿射，或遇到 `NotDivisible`”，此时仍可逐点求值，只是没有仿射形式。`is_normalized()` 返回 `normalized.is_some()`，只表示求值时是否用了缓存布局。
2. **缓存布局的扩展域行为与逐点求值不同**。R 沿每个模式的最后一个叶子延伸，逐点求值则在 A 的叶子间进位；组合求值遵循 R，与 CuTe 一致。

### 6. 图示：同一个 A，三种 B

下面的例子都使用 `A = (4,6):(6,1)`。它是一个 4 行 6 列的布局：逻辑下标 `j` 对应位置 `(j % 4, j / 4)`，地址为 `6·行 + 列`。

```text
逻辑下标 j                          地址 A(j)
      列0  1   2   3   4   5            列0  1   2   3   4   5
行0     0   4   8  12  16  20      行0    0   1   2   3   4   5
行1     1   5   9  13  17  21      行1    6   7   8   9  10  11
行2     2   6  10  14  18  22      行2   12  13  14  15  16  17
行3     3   7  11  15  19  23      行3   18  19  20  21  22  23
```

B 在这张网格上依次选格子。下图中 `[i]` 表示第 `i` 个选中的格子。

#### 6.1 步长整除：规范化结果与逐点求值相同

取 `B = 8:2`，即 `j = 0, 2, 4, …, 14`。

```text
逐点求值 A(B(i))               CuTe 结果 R = (2,4):(12,1)
      列0  1   2   3   4   5          列0  1   2   3   4   5
行0    [0] [2] [4] [6]  .   .    行0    [0] [2] [4] [6]  .   .
行1     .   .   .   .   .   .    行1     .   .   .   .   .   .
行2    [1] [3] [5] [7]  .   .    行2    [1] [3] [5] [7]  .   .
行3     .   .   .   .   .   .    行3     .   .   .   .   .   .
```

`compose_leaf` 的过程：

```text
模式0 (cs=4, cd=6): stride=2, ceil(4/2)=2 步，4%2==0
    → 叶子 2 : 2·6 = 12；rest_shape = 4，下一模式步长 = ceil(2/4) = 1
最后一个模式: → 叶子 4 : 1·1 = 1
R = (2,4):(12,1)，地址 0,12,1,13,2,14,3,15
```

步长 2 整除列高 4，每列正好选两格后进入下一列，两张图相同。此时 `normalized = Some(R)`，按 `R` 求值与逐点求值给出同样的地址。

#### 6.2 步长不整除：规范化结果与逐点求值不同

取 `B = 8:3`，即 `j = 0, 3, 6, 9, 12, 15, 18, 21`。

```text
逐点求值 A(B(i))               CuTe 结果 R = (2,4):(18,1)
      列0  1   2   3   4   5          列0  1   2   3   4   5
行0    [0]  .   .  [4]  .   .    行0    [0] [2] [4] [6]  .   .
行1     .   .  [3]  .   .  [7]   行1     .   .   .   .   .   .
行2     .  [2]  .   .  [6]  .    行2     .   .   .   .   .   .
行3    [1]  .   .  [5]  .   .    行3    [1] [3] [5] [7]  .   .
地址：0,18,13,8,3,21,16,11       地址：0,18,1,19,2,20,3,21
```

`compose_leaf` 的过程：

```text
模式0 (cs=4, cd=6): stride=3, ceil(4/3)=2 步
    4%3≠0 且 2<8 → 弱整除：丢弃余数，在下一模式重新起步
    → 叶子 2 : 3·6 = 18；rest_shape = 4，下一模式步长 = ceil(3/4) = 1（余数被丢弃）
最后一个模式: → 叶子 4 : 1·1 = 1
R = (2,4):(18,1)
```

- **逐点求值**：选完第 0 列的第 0、3 行后，步长 3 跨过列底，余数带到下一列，于是落在第 1 列第 2 行，路径呈锯齿状。
- **CuTe**：一列内只取 `ceil(4/3) = 2` 格，然后丢掉余数，从下一列顶部重新开始，得到规整的 2×4 子网格。

两者不同，`composition` 缓存的 `R` 反映 CuTe 的结果，之后 `index(i)` 返回 `R.index(i)`。补齐的 `logical_divide` 依赖这种“在下一模式重新起步”的规整结构，否则分块会错位。

#### 6.3 A 非仿射：`normalized = None`

把 A 换成 swizzle 这类非仿射映射：先算仿射地址，再对其中若干位做异或。

```text
A(j) = swizzle(6·行 + 列)
```

这种映射没有 shape/stride 形式，`as_affine()` 返回 `None`，不会进入 `compose_grouped`，也就没有 CuTe 规范化结果，只能逐点求值，对应 CuTe 的 `ComposedLayout`：

```text
i ──B──▶ j ──A（含 swizzle）──▶ 地址
```

#### 6.4 判定流程

```text
                 composition(A, B)
                        │
           A 和 B 都是仿射布局吗？
            ┌───────────┴───────────┐
            否                       是
            │                 运行 compose_grouped
     normalized = None               │
     逐点求值（6.3）        ┌─────────┴─────────┐
                         Ok(R)            NotDivisible
                           │                    │
                      Some(R)                 None
                  按 R 求值（6.1、6.2）      逐点求值
```

### 7. 实现细节备注

1. Only the pointwise fallback rejects an `inner` whose minimum address is negative (`CoordinateOutOfBounds`), because it uses those values as logical indices of `outer`. Normalization follows CuTe's signed stride divisibility condition `rest_stride % cs == 0 || rest_stride < cs` in `checked_step`, shared by the runtime and typed paths, so negative strides compose through `signum` (for example `(4,4):(4,1) ∘ 4:-1 = 4:-4`).
2. 未规范化分支的值域检查只试探 `max`。之后每次求值都使用带检查的 `add`/`mul`，即使中间下标溢出，也只会返回 `Err(Overflow)`，不会产生未定义行为。

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

与第 3 节的对应关系：

| CuTe `layout.hpp` | 类型层实现 |
|---|---|
| `coalesce_x`：flatten 后 `bw_coalesce`，末尾静态 1 换成 2 | [`coalesce.rs`](../src/algebra/coalesce.rs) 的 `CoalesceFold<true>` 与 `Prepend`；普通 coalesce 使用同一折叠的默认分支 |
| B 为 tuple 时右分配 | `ComposeTree` 按 B 的树递归，`Nil` 保留为空 tuple |
| 静态 stride 0 直接返回 B | `ZERO_STRIDE` 分支得到 `s:0` |
| 对 A 前 r-1 个模式的 fold | `Fold`/`FoldStep`，静态值由 `const fn step` 推导，混合值由 `TypedStep` 计算 |
| 结果为空 / `rest_shape == 1` 时 unwrap / 其余 append | `Finish`/`FinishNonempty` |

与动态 `compose_leaf` 的差别都来自"严格照抄上游静态路径"：

- 只有 stride 为 0 时走直通分支；`s == 0` 遇到多模式 A 会和上游一样在 shape 整除检查处失败，A 为单模式时得到 `0 : d*stride0`。
- 被 unwrap 丢弃的末尾 stride 不会被求值，因此 `rest_stride * stride[r-1]` 溢出不影响结果。
- 不可整除、溢出、profile 不一致或出现动态叶子都在实例化时编译失败，而不是返回 `Err`。

偏离上游：上游 `coalesce_x` 要取最后一个模式，空 A 无法实例化；这里按本库的标量约定把空 A 当作 `1:0`，因此 `() ∘ s:d = s:0`。

#### 7.2 coalesce 的全静态、混合与运行时入口

[`coalesce.rs`](../src/algebra/coalesce.rs) 将类型层和运行时化简放在同一模块：

- A fully static `Layout` reads its result as `<CoalescedLayout<S, D> as StaticValue>::VALUE` (or `CoalescedLayout<S, D, true>` for `coalesce_x`); `CoalesceFold<false/true>` derives the type.
- 混合 `Layout` 使用 `coalesce` / `coalesce_extended` 方法，保留静态叶子及动态字段，不转成运行时数组。输出类型是 `CoalescedLayout<S,D>` / `CoalescedLayout<S,D,true>`；结果构造后返回受检的 `Result`。
- `DynLayout` 使用同名方法，根据当前数值决定哪些模式可以删除或合并，结果仍是 `DynLayout`。

普通与扩展版本的末尾初始化共用 `PrependToEmpty<S,D,EXTENDED>`。保留分支共用同一实现；遇到静态单位轴时，普通版本返回空尾部，扩展版本返回携带原 stride 的 extent-2 占位叶子。

三条路径统一从右向左折叠。类型层严格采用 CuTe 的静态性条件：当前 shape/stride 与累积布局的首个 shape/stride 均为静态值，才能依据 `s*d == next_stride` 合并。例如 `(Int<2>, DynInt(3)):(Int<1>, Int<2>)` 保留为两轴，而对应的 `DynLayout` 可以合并为 `6:1`。

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

常量读取要求两棵树都是 `Int<N>` 叶子，不执行运行时大小或地址范围校验；静态合并乘积溢出在实例化时编译失败。方法入口检查输入 shape 与输出布局，动态错误返回 `Err`。

扩展版本的类型层结果遵循上游占位规则：末尾静态 shape-1 改为 2，因此逻辑大小可能增加；混合输入的动态 shape-1 不会改为静态 2。运行时版本沿用末尾真实 extent，以免占位值增加溢出风险，有限域大小不变，但保留相同的扩展步长。空输入统一采用本库的 `1:0` 标量约定。扩展映射等价要求原布局的前导 extent 为正；含零 extent 时仍按 CuTe 的代数规则化简，不保证保留原扩展解码的零除错误。

#### 7.3 composition 的全静态、混合与运行时入口

三类算法都在 [`composition.rs`](../src/algebra/composition.rs) 内实现。

| 输入与入口 | 结果 | 结构与检查 |
|---|---|---|
| Fully static `<TypedComposition<A,B> as StaticValue>::VALUE` | Zero-sized `Layout` constant | The type determines the value; static illegal shapes, divisibility errors, and product overflow are rejected at instantiation |
| 混合 `A.composition(B)`，A/B 为 `Layout` | `Result<TypedComposition<A,B>>` | 保留静态叶子和 RHS 完整嵌套；动态参数仅改变字段值，不改变类型 |
| Runtime `composition_layout(&A, &B)` | `Result<GroupedLayout>` | Normalizes with the current values and keeps the RHS top-level grouping |
| 通用 `composition(A,B)` | `Result<Composition<A,B>>` | 保留原有规范化缓存、不可整除或非仿射时的逐点回退 |

全静态与混合入口共用 `ComposeTree`、`Fold` 和 `Finish`，各阶段通过 `Output: LayoutParts` 表达同一个布局结果，中继阶段直接转发 `Output`。最终通过 `LayoutParts::into_layout` 取出 shape/stride 再执行入口所需校验；这个转换本身不做归一或校验。辅助 trait 的 shape/stride 应从 `Output` 投影，公开结果别名保持一致。

其计算顺序为：先通过公共 `CoalesceFold<true>` 规范化 LHS，再按 RHS 的树右分配。分支只能由静态事实决定：静态 stride-0 直接广播；静态 `next_shape == 1` 或 `rest_shape == 1` 才跳过模式；最后仅静态 `rest_shape == 1` 才 unwrap。运行时碰巧为 1 的模式保留，例如 `(4,6):(6,1) ∘ DynInt(2):Int<3>` 得到 `(DynInt(2),DynInt(1)):(Int<18>,Int<1>)`，不会因为这一次剩余 shape 为 1 就改成标量类型。

类型化 coalesce 只能利用静态已知的连续性，运行时 coalesce 可以利用当前数值，因此两者可能保留不同的 LHS 模式边界；弱整除恰好依赖这些边界，结果数值也可能不同。例如动态叶子的 `(4,6):(1,4)` 与 `8:3` 组合，类型化入口保留两个 LHS 模式，结果地址为 `[0,3,4,7,8,11,12,15]`；运行时入口先合并为 `24:1`，结果为 `[0,3,6,9,12,15,18,21]`。这延续 CuTe 对静态信息和模式边界的依赖，通用惰性入口仍沿用运行时规范化行为。

`TypedStep` 分别推导 extent、剩余 shape 和剩余 stride 的类型：全部依赖已静态确定时使用 `Int<N>`，否则使用 `DynInt`。stride 乘法还保留 CuTe 的静态零规则，`Int<0> * DynInt` 仍为 `Int<0>`；静态零分子除动态值也保留 `Int<0>`。这些规则限于 composition，不改变本库普通整数运算符的类型提升约定。

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

类型化方法检查两侧输入 shape 与结果的大小、地址范围，不经过 `Leaves` 或 `DynLayout`，不受 `MAX_RANK` 限制。It shares the signed `d < cs` stride condition and negative-stride support with the runtime normalization.

与上游的受检差异：CuTe 对动态参数省略的整除断言，在类型化 Rust 方法中仍执行检查并返回 `NotDivisible`；无法执行的除零返回 `InvalidShape`，动态绝对值或乘积溢出返回 `Overflow`。因此动态 stride 为 0 可在单模式 LHS 上直接缩放，但多模式中需要计算 `ceil(cs/abs(d))` 时返回错误；静态 stride-0 则直接广播。静态输入出现同样的非法运算会在编译期拒绝。空 RHS 保留空 tuple，空 LHS 按本库的 `1:0` 约定扩展。

运行时规范化与惰性回退契约集中在 [`tests/composition_contract.rs`](../tests/composition_contract.rs)，其中也包含混合类型、零值、负 stride、溢出及超出 `MAX_RANK` 的用例。全静态类型与上游示例也在同一文件中；其他布局代数测试与组合测试共用 `tests/common/mod.rs` 的布局构造和非仿射映射。

## 第二部分 · 补布局 `complement`

本部分讲解 [`src/algebra/complement.rs`](../src/algebra/complement.rs) 中的全静态、混合与运行时入口（入口对照见 14.2 节），以及运行时 `complement_prefix` / `complement_shape` / `complement_extended`：它们怎样求出一个布局留下的空隙，以及为什么要按步长排序、为什么用向下取整、为什么末尾会出现 `1:q` 这样的叶子。

本部分记被求补的布局为 `L`，结果为 `R`。

### 8. `complement` 求的是什么

名字里的"补"**不是集合补集**。布局 `L` 把坐标映射成地址，它会跳着访问内存，留下一堆没被访问到的**空隙**。`complement(L, cotarget)` 生成的是另一个布局 `R`，它的地址恰好是**这些空隙的起点集合**。

两者合起来 `(L, R)` 就能把 `[0, cotarget)` 这片地址摊平成"块内偏移 × 块间偏移"的二维结构——这正是第一部分开头那条 `logical_divide` 公式的来历。

直观比喻：`L` 是"一块瓷砖内部怎么走"，`R` 是"瓷砖与瓷砖之间怎么跳"。

#### 8.1 最小例子

取 `L = 4:1`（连续读 4 个元素），`cotarget = 24`：

```text
L 访问的地址：  0 1 2 3
整个地址空间：  0 .. 23
```

把 24 个格子按 4 个一组切开，共 6 组，组与组之间相距 4，于是

```text
complement(4:1, 24) = 6:4
```

`R = 6:4` 的地址是 `{0, 4, 8, 12, 16, 20}`，正是每块的起点。`L(i) + R(j)` 刚好不重不漏地覆盖 `[0, 24)`。

### 9. 核心不变量：`current`

算法只有一个状态变量，代码里叫 `current`，含义是：

> 从地址 0 开始，到目前为止已被处理过的叶子**完整铺满**的那段长度。

初值 `current = 1`，表示"目前只占住了地址 0 这一个点"。随后**按步长从小到大**依次处理 `L` 的每个叶子 `s:d`，对每个叶子做两件事：

1. **补空隙**：`current` 与该叶子的起跳步长 `d` 之间若有空档，就把它作为 `R` 的一个叶子补上，即 `push(d / current, current)`；
2. **推进**：该叶子自身覆盖到了 `s · d`，于是 `current = s · d`。

#### 9.1 为什么必须按步长排序

空隙的计算依赖"从小到大逐层扩张"。若先处理大步长的叶子，`current` 会直接跳得很大，后面小步长的叶子就算不出正确空隙了。

```rust
order[..flat.rank()].sort_unstable_by_key(|&a| (flat.strides()[a], a));
```

排序键是 `(步长, 原下标)`。第二个键 `a` 是关键细节：**它把不稳定排序变成了稳定排序**，等价于 CuTe 的 `stable_sort`，且无需额外分配。步长相同的两个叶子仍按原顺序排列，结果确定可复现。

### 10. `complement_prefix` 逐步拆解

```rust
fn complement_prefix<L: AffineLayoutMap>(layout: &L) -> Result<Option<(Leaves, isize)>>
```

这是 `complement_shape` 与 `complement_extended` 共享的前半段，返回 `(前缀叶子, 下一条步长)`。

#### 10.1 拒绝空域

```rust
let flat = layout.as_flat_layout()?;
if flat.size()? == 0 {
    return Err(Error::Unsupported);
}
```

`size == 0` 意味着某个轴长度为 0，布局不访问任何地址，"空隙"此时无意义，直接报 `Unsupported`。 This is a deliberate `Result` in place of upstream undefined behavior: a zero extent makes the next stride `s * d` zero, and upstream's `ceil_div(cotarget, new_stride)` then divides by zero.

#### 10.2 filter 掉广播轴

```rust
let flat = flat.filter()?;
```

`filter` 把**步长为 0 的叶子的 shape 改成 1，然后 coalesce**。stride-0 的轴是**广播**——不管坐标怎么变，地址都不动，它不消耗任何地址空间，因此对"哪些地址被占用"没有贡献，必须先剔除，否则会被误判成占了一大段。顺带地，`coalesce` 还会把能合并的相邻叶子并掉（`s0·d0 == d1` 时），让后续处理的叶子数最少。

#### 10.3 全广播特例

```rust
if flat.rank() == 1 && flat.stride(0)? == 0 {
    return Ok(None);
}
```

若 filter 之后只剩 `1:0`（原布局是 `1:0`、纯广播 `4:0`、或所有轴长度都是 1），说明 `L` 只占住地址 0 一个点，**整个 cotarget 都是空隙**。这是 CuTe 的特例分支，用 `None` 作信号往上传，调用方直接返回整片连续空间：

```text
complement(4:0, 6)       = 6:1
complement_extended(4:0) = 1:1
```

#### 10.4 拒绝负步长

```rust
for (a, slot) in order.iter_mut().enumerate().take(flat.rank()) {
    if flat.stride(a)? < 0 {
        return Err(Error::Unsupported);
    }
    *slot = a;
}
```

负步长意味着地址往回走，"从小到大扩张"的模型直接失效。同一个循环顺手把 `order` 初始化成 `[0, 1, 2, …]`。

#### 10.5 主循环：逐个补空隙

```rust
let mut prefix = Leaves::new();
let mut current = 1;
for &a in &order[..flat.rank()] {
    let (s, d) = (flat.shape(a)?, flat.stride(a)?);
    let (gap, next) = complement_step(s, d, current)?;
    if gap > 1 {
        prefix.push(gap, current)?;
    }
    current = next;
}
```

三个分支各对应一种真实情况：

| 情况 | 含义 | 处理 |
|---|---|---|
| `gap == 0` | `d < current`，叶子的起跳点落在**已覆盖区域内部** → 地址重叠，`(L, R)` 不可能单射 | `Err(NonInjective)` |
| `gap == 1` | `d == current`，严丝合缝，**没有空隙** | 跳过，不占叶子容量 |
| `gap > 1` | 存在空隙，长度 `gap`、间距 `current` | `push(gap, current)` |

`gap > 1` 才 push 不只是省事：`Leaves` 是定长 `MAX_RANK` 缓冲区，省下的每个槽位都能降低触碰 `CapacityExceeded` 的风险；而且长度为 1 的叶子最终也会被 `coalesce` 去掉。

**floor 除法是刻意的**：当 `d` 不被 `current` 整除时，算法选择向下取整而非报错，代价是结果可能与原布局在空洞处重叠。这是为了对齐 CuTe 的行为，不是缺陷，详见 [14.1](#141-算法不保证什么)。

#### 10.6 走一遍 `L = (2,2):(1,6)`

| 步骤 | 叶子 | 进入时 `current` | `gap = d / current` | 动作 | 离开时 `current` |
|---|---|---|---|---|---|
| 1 | `2:1` | 1 | `1 / 1 = 1` | 无空隙，跳过 | `2 · 1 = 2` |
| 2 | `2:6` | 2 | `6 / 2 = 3` | push `3:2` | `2 · 6 = 12` |

前缀 = `3:2`，`current = 12`，最终 `complement((2,2):(1,6), 24) = (3,2):(2,12)`。

验证：`L` 的地址是 `{0, 1, 6, 7}`，`R` 的地址是 `{0, 2, 4, 12, 14, 16}`，两两相加得到 24 个互不相同的地址，没有重叠。

### 11. `complement_shape`：补到目标尺度

前缀只处理了"叶子之间"的空隙，还剩从 `current` 到 `cotarget` 的**尾部空隙**。

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

#### 11.1 为什么用 `ceil_div`

`cotarget` 不一定被 `current` 整除。CuTe 选择**向上取整**，保证结果能覆盖到 cotarget——宁可多盖一点，不可少盖：

```text
complement(3:1, 24) = 8:3      // current = 3，ceil(24/3) = 8，覆盖 [0,24)
```

#### 11.2 cotarget 是 tuple 时为什么结果不同

`cotarget` 可以是一个 shape（多个数），这时 CuTe 用**逐叶子折叠**的 tuple-ceil_div：

```text
extent_i = ceil(a_i / rest)
rest     = ceil(rest / a_i)
```

每个 `a_i` 先吃掉一部分 `rest`，剩下的留给下一个，结果可能比直接用乘积大：

```text
complement_shape(3:1, [4,6]) = 12:3      // 不是 8:3
complement(3:1, 24)          =  8:3      // 24 = 4·6，但结果不同
```

手工验证 `[4, 6]`（起始 `current = rest = stride = 3`）：

| `a_i` | `extent = ceil(a / rest)` | 新 `rest` | push | 新 `stride` |
|---|---|---|---|---|
| 4 | `ceil(4/3) = 2` | `ceil(3/4) = 1` | `2:3` | `3 · 2 = 6` |
| 6 | `ceil(6/1) = 6` | `ceil(1/6) = 1` | `6:6` | 36 |

叶子为 `(2,6):(3,6)`，coalesce 时 `2 · 3 == 6`，合并成 `12:3`。

`a == 0` 时跳过第二项除法（`rest` 保持不变），避免除零。

#### 11.3 收尾与特例

- `extent != 1` 才 push，同样是为了省槽位。
- 最后统一 `coalesce()`：合并相邻叶子，并在全为 1 时归一成 `1:0`，与 CuTe 一致。
- 入口处接住前缀返回的 `None`（全广播），直接返回 `prod(cotarget):1`：

  ```rust
  let Some((mut leaves, current)) = complement_prefix(&layout)? else {
      let size = cotarget.iter().try_fold(1, |acc, &c| mul(acc, c))?;
      return DynLayout::new(&[size], &[1]);
  };
  ```

### 12. `complement_extended`：标记重复步长

```rust
let Some((leaves, current)) = complement_prefix(&layout)? else {
    return DynLayout::new(&[1], &[1]);
};
let prefix = leaves.layout()?.coalesce()?;
if prefix.size()? == 1 {
    return DynLayout::new(&[1], &[current]);
}
// …把 prefix 的叶子复制到 out…
out.push(1, current)?;
```

这个版本**没有 cotarget**，它保留内部空隙，然后追加一个 `1:current` 的尾叶子。

`1:current` 逻辑尺寸是 1、**不增加 size**，唯一作用是**携带步长信息**：告诉调用方"若要重复铺下一份，起点在 `current`"。乘积类操作（`blocked_product`、`tiled_product`、`TiledMmaLayout`）正是靠这个值推出外层步长。

```text
complement_extended(4:2) = (2,1):(1,8)
```

这里的 **8 来自 `s · d = 4 · 2`，不是 cosize（= 7）**。理解乘积运算时不能把这个步长替换成地址跨度。

`prefix.size() == 1` 的短路分支处理"没有任何内部空隙"的情况，直接返回 `1:current`。

### 13. `Leaves`：定长叶子缓冲区

定义在 [`src/algebra.rs`](../src/algebra.rs)，由 `complement`、`composition`、`inverse` 共用。

```rust
struct Leaves {
    shape: [isize; MAX_RANK],
    stride: [isize; MAX_RANK],
    len: usize,
}
```

这是个定长、栈上、`Copy` 的叶子收集器，溢出时返回 `Err(CapacityExceeded)` 而不是 panic。整个 crate 走 no-alloc 路线，不能用 `Vec`，这也解释了 [10.5](#105-主循环逐个补空隙) 与 [11.3](#113-收尾与特例) 两处主动跳过长度为 1 的叶子的做法。

### 14. 错误与边界

| 输入情况 | 返回 |
|---|---|
| `size == 0`（空域） | `Err(Unsupported)` |
| filter 后仍有负步长 | `Err(Unsupported)` |
| cotarget 含负数 | `Err(InvalidShape)` |
| `gap == 0`（步长重叠） | `Err(NonInjective)` |
| 叶子数超 `MAX_RANK` | `Err(CapacityExceeded)` |
| filter 后只剩 `1:0` | `prod(cotarget):1` / `1:1` |

#### 14.1 算法不保证什么

**`complement` 不保证结果无重叠、无空洞地铺满。** 反例：

```text
complement((2,2):(1,3), 12) = 2:6
```

`L` 的地址是 `{0, 1, 3, 4}`，`R = 2:6` 是 `{0, 6}`，合起来是 `{0,1,3,4,6,7,9,10}`——**地址 2、5、8、11 仍然缺失**。原因是 `L` 自身的步长 3 不被 `current = 2` 整除，floor 除法 `3 / 2 = 1` 判定为"无空隙"，于是地址 2 被漏掉。

因此要依据算法条件和实际返回值判断，不能把 `complement` 当成通用的集合补集或双射证明工具。同样，广播被 filter 掉也不意味着原布局变成了单射。

#### 14.2 全静态、混合与运行时入口

算法以 [CuTe `layout.hpp`](https://github.com/NVIDIA/cutlass/blob/main/include/cute/layout.hpp) 的 `filter`、`detail::complement_prefix` 和两个 `detail::complement` 重载为准。相关实现均在 [`complement.rs`](../src/algebra/complement.rs)，并从 `algebra` 重导出。

| 输入与入口 | 结果与结构 |
|---|---|
| Fully static `<TypedComplement<L, C> as StaticValue>::VALUE` | Zero-sized `Layout` constant; the cotarget may be an integer or a nested shape |
| Fully static `<TypedComplement<L, Nil, true> as StaticValue>::VALUE` | Constant layout keeping the terminal `1:repetition-stride` mode |
| 混合 `L.complement(cotarget)`，L 为 `Layout` | `Result<TypedComplement<L,C>>`，保留可推导的静态叶子 |
| 混合 `L.complement_extended()` | `Result<TypedComplement<L,Nil,true>>` |
| Runtime free functions `complement` / `complement_shape` / `complement_extended` | Take any `AffineLayoutMap`, sort and simplify with the current values, and return a `DynLayout`; divide/product call them too |

全静态与混合入口共用以下阶段，不转为运行时数组，因此不受 `MAX_RANK` 限制：

1. `Flatten` 展平 shape/stride；`FilterLeaf` 只把静态零 stride 对应的 shape 换为静态 1。`Filter` 复用 coalesce 的 `Prepend` 删除静态单位轴并合并静态连续模式。全部删除后用空中间布局表示上游的 `1:0` 特例。
2. `Sort` / `Insert` 按静态 stride 稳定排序。与上游一致，**过滤后的多模式布局要求所有 stride 静态已知**；动态 stride 仅支持单模式。需要多模式动态排序时使用 `as_flat_layout()` 和运行时入口。
3. `Prefix` 计算 `gap = d/current` 和 `next = s*d`；动态路径与类型化路径共用 `complement_step` 检查符号、重叠和乘积溢出。静态计算复用 `Int` 的除法、乘法，依赖动态值的结果保留为动态整数。
4. 有 cotarget 时，`CeilTarget` 逐叶折叠 ceil_div；`GroupShape` 与上游 `coalesce(shape)` 一样，只合并相邻同为静态或同为动态的叶子。`TargetStrides` 复用 `ComposeProduct` 的静态零传播构造左序 stride，不计算未使用的末尾乘积；它避免 `Compact::compact` 的末尾乘积发生无关溢出。前缀与尾部拼接后复用 `CoalesceFold` / `FinishCoalesce`，最后用 `Unwrap` 将单模式结果还原为整数 profile。
5. 无 cotarget 时，先用 `CoalesceFold` 化简前缀，再追加静态单位轴和最终 stride；前缀为空时直接返回 `1:最终stride`。全广播特例同样 unwrap 单模式，直接返回 cotarget 的紧凑布局或 `1:1`，不再进行普通布局 coalesce。

类型结构只受静态信息影响。比如 `4:1` 的动态 cotarget 恰好为 4 时，结果仍是动态 `1:4` 叶子；若 cotarget 为 `Int<4>`，普通 coalesce 将结果归一为静态 `1:0`。动态 shape 恰好等于 1 时也会保留原模式。

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

偏离与检查边界：静态零域、静态负 stride、静态零 gap 等在编译期拒绝；动态错误通过 `Result` 返回。类型化动态零 stride 不会被过滤，返回 `NonInjective`，运行时数值路径则可以过滤它。类型化方法检查输入 size 和最终 `Layout::new`；reading a fully static result through `StaticValue` skips the final runtime size/address_range validation, as for coalesce.空 tuple 沿用本库标量约定；零 cotarget 的反向 ceil_div 在上游会零除，本库沿用既有动态接口的零域扩展，遇零时保持 rest，尾部大小为零。

契约测试集中在 [`tests/complement_contract.rs`](../tests/complement_contract.rs)，覆盖全静态常量类型、上游样例与过滤/排序、混合类型依赖、动态边界、超过 `MAX_RANK` 的类型化输出，以及原有 cotarget、重复步长及错误契约。编译失败约束由 rustdoc 的 `compile_fail` 示例覆盖。

### 15. 流程总览

```text
                    ┌─────────────────────────────┐
                    │     complement_prefix       │
   L ──────────────▶│ 1. size == 0?    → Err      │
                    │ 2. filter（去广播 + coalesce）│
                    │ 3. 只剩 1:0?     → None     │
                    │ 4. 负步长?       → Err      │
                    │ 5. 按 (stride, idx) 排序    │
                    │ 6. 循环：                   │
                    │      gap = d / current      │
                    │      gap == 0 → NonInjective│
                    │      gap  > 1 → push(gap,cur)│
                    │      current = s · d        │
                    └──────────┬──────────────────┘
                               │ (prefix, current)
                   ┌───────────┴────────────┐
                   ▼                        ▼
          complement_shape          complement_extended
        逐 cotarget 叶子用          追加 1:current
        ceil_div 补尾部空隙        （只标记重复步长）
                   │                        │
                   ▼                        ▼
             coalesce 归一          (空隙…, 1:current)
```

一句话记忆：**排序 → 逐叶补空隙 → 补尾巴**。状态只有 `current` 一个，三条分支（重叠 / 无缝 / 有空隙）覆盖全部情形。

## 第三部分 · 求逆 `inverse`

本部分讲解 [`src/algebra/inverse.rs`](../src/algebra/inverse.rs) 中的 `sorted_leaves` / `right_inverse` / `left_inverse`，以及枚举版的 `dense_left_inverse` / `dense_right_inverse`：怎样从地址反推逻辑下标，为什么要按步长排序，以及"地址空洞"为什么会让两种逆给出完全不同的答案。

本部分记被求逆的布局为 `L`，结果为 `R`。

### 16. 求逆到底在逆什么

一个仿射布局 `L = shape:stride` 定义的映射分两步：

```text
逻辑线性下标 i  ──分解──▶  坐标 (c0,c1,…)  ──加权求和──▶  地址 a
```

- **分解**用"左优先（left-major）混合进制"：`c0 = i % s0`，`c1 = (i / s0) % s1`，…
  因此第 `k` 个叶子在**逻辑侧**的权重是 shape 前缀积 `p_k = s0·s1·…·s_{k-1}`。
- **求和**用 stride：`a = Σ c_k · d_k`，第 `k` 个叶子在**地址侧**的权重是 `d_k`。

于是一个布局本质上就是一张两列权重表：

| 叶子 | 量程 | 逻辑侧权重 | 地址侧权重 |
|---|---|---|---|
| `k` | `s_k` | `preprod[k]`（前缀积） | `stride[k]` |

**求逆就是把这两列权重对调**：原来 `preprod → stride`，逆就是 `stride → preprod`。

剩下的全部复杂度只来自两件事：

1. 逆的输入是**地址**，所以叶子必须**按步长从小到大**重排（与 [9.1](#91-为什么必须按步长排序) 中 `complement` 的排序动机相同）；
2. 地址空间可能有**空洞**（`L` 跳着访问）或**重叠**（广播轴），必须决定怎么处理。

`right_inverse` 与 `left_inverse` 的差别，正是对"空洞"的两种态度。

### 17. 两种逆的契约

| | 定义 | 直觉 |
|---|---|---|
| **右逆** `R` | `L(R(a)) == a`，对所有 `a < size(R)` | "给我一个地址，还我一个能生成它的下标"。被问到的地址必须真的能由 `L` 产生，所以**不能碰空洞**，只能停在第一个洞之前。 |
| **左逆** `G` | `G(L(i)) == i`，对所有 `i < size(L)` | "给我 `L` 产生过的地址，还我原下标"。空洞上的地址根本不会被问到，所以那里**返回什么都行**，可以一路走到底。 |

这就解释了下面这对看似矛盾的结果：

```text
right_inverse(4:2) = 1:0            // 地址只有 {0,2,4,6}；1 是空洞，右逆只能覆盖 [0,1)
left_inverse (4:2) = (2,4):(0,1)    // 左逆在 0..7 全有定义，空洞处是垃圾值但无人查询
```

### 18. 公共准备：`sorted_leaves`

```rust
let flat = layout.as_flat_layout()?.coalesce()?;          // ①
for a in 0..flat.rank() {
    preprod[a] = acc;                                     // ②
    acc = mul(acc, flat.shape(a)?)?;
    order[a] = a;
}
order[..flat.rank()].sort_unstable_by_key(|&a| (flat.strides()[a], a));   // ③
```

两种结构化逆共享这一段，它只做三件事：

① **拍平 + coalesce**。嵌套结构对求逆没有意义，先摊成一维叶子序列；`coalesce` 丢掉 `shape == 1` 的叶子（它们对地址没有贡献），并合并相邻且连续的叶子（`(4,2):(1,4)` 合成 `8:1`；`(4,2):(2,1)` 与 `(2,4):(4,1)` 都不能合）。叶子越少，后面的循环越简单，输出也更规范。

② **`preprod[a]` 即逻辑侧权重**。这是逆布局里所有 stride 的来源：逆的 stride 就是原布局的 preprod。

③ **按 `(步长, 原下标)` 稳定排序**。第二个键的作用与 [9.1](#91-为什么必须按步长排序) 一致：把不稳定排序变成稳定排序，等价于 CuTe 的 `stable_sort` 且无需分配。

注意 `order` 只是下标排列，`flat` 本身没有被重排。

### 19. `right_inverse`：贪心拼接连续地址前缀

```rust
let mut current = 1;
for &a in sorted_order {
    let (s, d) = (shape(a), stride(a));
    if d == current {                       // 严丝合缝才接上
        leaves.push(s, preprod[a])?;
        current = mul(s, d)?;               // 已覆盖的地址跨度
    }
}
leaves.layout()?.coalesce()
```

An empty layout (`size == 0`) is not rejected: like upstream, the fold simply runs on its coalesced leaves, e.g. `right_inverse(0:1) = 0:1` and `right_inverse(0:2) = 1:0`. `left_inverse` does the same (`left_inverse(0:1) = 0:1`). Only `complement` rejects empty layouts, because upstream divides by zero there.

`current` 的含义与 [第 9 节](#9-核心不变量current)相同：**从地址 0 开始、已被无洞覆盖的跨度**，初值 1（只占住地址 0）。

算法贪心地问：有没有叶子的步长**恰好等于** `current`？

- 有 → 它正好把覆盖范围从 `current` 扩展到 `s · d`，接上它。逆布局中该叶子的 shape 不变，stride 换成 `preprod[a]`（输出逻辑下标）。
- 没有 → 说明 `current` 处是空洞。循环并不 `break`，但后续叶子的步长只会更大，**再也不可能匹配**，等价于停在这里。

实测：

```text
(2,4):(4,1)  → right_inverse = (4,2):(2,1)   // current: 1→4→8，全覆盖，是真正的双射逆
(3,2):(1,8)  → right_inverse = 3:1           // current: 1→3，地址 3..7 是洞，停
4:2          → right_inverse = 1:0           // 没有步长为 1 的叶子，一步都走不动
(2,4):(0,1)  → right_inverse = 4:2           // 广播轴因 d == current >= 1 恒不成立被自然跳过
```

三个细节：

- 一个叶子都没接上时，`leaves` 为空，收尾的 `coalesce()` 会直接把它规约为合法的 `1:0`，无需预先压入 `1:0` 哨兵（避免占掉 `MAX_RANK` 的一个槽位）。
- **负步长无需显式拒绝**：它同样永远匹配不上 `current >= 1`，被静默忽略。
- 这是**结构化**的前缀，不是"搜索出来的最长前缀"。例如 `(2,2):(1,1)` 实际能摸到地址 0、1、2，但算法只承认 0、1。这与 [14.1](#141-算法不保证什么) 是同一类告诫：结论要按算法条件读，不能按集合语义读。

### 20. `left_inverse`：跨越空洞的解码器

左逆必须在**整个地址区间**上有定义，于是思路变成：**把地址当成一个混合进制数拆位，每一位乘以对应叶子的 `preprod` 再求和。**

拆位用的"进制"来自排序后**相邻步长的比值**。设排序后的步长为 `d_0 < d_1 < … < d_{k-1}`，对应 preprod 为 `p_0 … p_{k-1}`，最后一个叶子的 shape 为 `s_last`，则

```text
R = ( d_0 , d_1/d_0 , d_2/d_1 , … , d_{k-1}/d_{k-2} , s_last )
  : (  0  ,   p_0   ,   p_1   , … ,    p_{k-2}      , p_{k-1} )
```

逐项解读：

- **首项 `(d_0 : 0)`**：地址中低于 `d_0` 的那一段任何叶子都碰不到（纯空洞）。给它步长 0，等于"这几位直接丢弃"。
- **中间第 `i` 项 `(d_i/d_{i-1} : p_{i-1})`**：这一位的值要乘以**上一个**叶子的逻辑权重，所以 shape 与 stride 在数组里**错开一格**——这正是代码注释"shape 比 stride 少一个前导项"的含义。
  这里有个关键技巧：这一位的量程取 `d_i/d_{i-1}` 而不是叶 `i-1` 自己的 `s_{i-1}`。由于 `d_i/d_{i-1} >= s_{i-1}`，多出来的那一截恰好是叶 `i-1` 之后的空洞。**把"叶子量程"和"它后面的空洞"合并成一个模式**，既少一个叶子，又因为空洞上的输出无人检查而完全合法。
- **末项 `(s_last : p_{k-1})`**：最后一个叶子后面没有需要吞掉的空洞，用它真实的 shape。

代码用 `covered` 与 `min_stride` 实现这条比值链：

```rust
if d % covered != 0 { return Err(Error::NotDivisible); }
if d < min_stride   { return Err(Error::NonInjective); }
shape[len]      = d / covered;        // 这一位的进制
stride[len + 1] = sorted.preprod[a];  // 错开一格：配给下一个槽位
covered    = mul(covered, shape[len])?; // covered 变成 d
min_stride = mul(flat.shape(a)?, d)?;   // 下一非广播叶子的最小不重叠步长
```

**不变量：每轮结束后 `covered == 当前叶子的步长 d`**（因为 `covered · (d/covered) = d`），`min_stride == s · d`（当前叶子覆盖到的地址上界）。因此：

- `d % covered != 0` 要求**步长链两两整除**：`d_0 | d_1 | d_2 | …`，不满足返回 `NotDivisible`（CuTe 在此直接静态断言失败）；
- `d < min_stride` 要求**非广播叶子互不重叠**：`d_k >= s_{k-1} · d_{k-1}`，否则不同逻辑坐标会映射到同一地址（如 `(2,2):(1,1)` 或 `(3,2):(1,2)`），左逆恒等式无法成立，返回 `NonInjective`。

两条前置处理：**步长为 0 的叶子（广播）直接跳过**——广播布局不单射、本来就没有左逆，跳过后得到的其实是它的某个*右*逆；**负步长**返回 `Unsupported`。

#### 20.1 走一遍 `L = (3,2):(1,8)`

排序后的叶子：`3:1`（`p = 1`）、`2:8`（`p = 3`）。

| 步骤 | `covered`（前） | 写入 shape 位 | 写入 stride 位 | `covered`（后） |
|---|---|---|---|---|
| 叶 `3:1` | 1 | `1/1 = 1` | 下一格 `p = 1` | 1 |
| 叶 `2:8` | 1 | `8/1 = 8` | 下一格 `p = 3` | 8 |
| 收尾 | | `s_last = 2` | | |

得到 `(1,8,2):(0,1,3)`，`coalesce` 掉 shape-1 后是 **`(8,2):(1,3)`**。

验证恒等式：`i = i0 + 3·i1`，地址 `a = i0 + 8·i1`（`i0 < 3`，`i1 < 2`）。
`R` 把 `a` 拆成 `a0 = a % 8 = i0`、`a1 = a / 8 = i1`，输出 `1·a0 + 3·a1 = i0 + 3·i1 = i` ✓

注意第一位量程是 **8** 而不是 3——多出的 `3..7` 就是被吞掉的空洞，那里的输出是垃圾值，但 `L` 永远不会产生这些地址。

#### 20.2 对照表

```text
L            right_inverse    left_inverse
4:2          1:0              (2,4):(0,1)     // 右逆停在洞前，左逆吞掉洞
(4,2):(2,1)  (2,4):(4,1)      (2,4):(4,1)     // 稠密双射，两者一致
(2,4):(4,1)  (4,2):(2,1)      (4,2):(2,1)
(4,2):(1,4)  8:1              8:1             // coalesce 成 8:1，自逆
(3,2):(1,8)  3:1              (8,2):(1,3)
(2,4):(0,1)  4:2              4:2             // 广播：左逆退化为某个右逆
```

### 21. 结构化逆 vs 枚举逆

`Inverse<L>` 及其两个构造函数 `dense_left_inverse` / `dense_right_inverse` 走的是完全不同的路线：不看步长结构，而是在 `index(address)` 时**从逻辑下标 0 开始线性扫描**原布局，首次命中即返回（因此多重原像时取最小者）。

| | 结构化（`left_inverse` / `right_inverse`） | 枚举（`dense_*_inverse`） |
|---|---|---|
| 适用对象 | 仿射布局（有 shape/stride 元数据） | 任意 `LayoutMap`，包括 swizzle 等非仿射映射 |
| 代价 | 一次排序 + 一遍扫描，结果是静态结构 | 每次查询 `O(size)` 扫描，但不分配堆内存 |
| 对空洞 | 右逆截断，左逆吞掉 | 一律 `NotSurjective` |
| 结果类型 | 类型化入口返回 `Layout`，运行时入口返回 `DynLayout`，均可继续参与代数 | 包装类型，`as_affine()` 返回 `None` |

`dense_left_inverse` 要求原布局单射且地址恰好覆盖 `[0, size)`（空布局允许）；`dense_right_inverse` 要求地址从 0 开始连续覆盖 `[0, max]`，但**允许广播**，为每个地址选择最小的逻辑原像（构造时先按抽屉原理拒绝 `max + 1 > size(L)` 的稀疏映射；对 `max + 1 == size(L)` 且通过仿射步长分离证明的稠密双射直接在 $O(\text{叶子数}^2)$ 内放行，其余情形再逐地址扫描原像）。

两条路线的分工：布局代数内部（`logical_divide`、`raked_product`、TV 布局推导）用结构化逆，因为结果还要继续参与代数运算；需要对 swizzle 这类非仿射映射精确反查时才用枚举逆。

#### 21.1 inverse 的全静态、混合与运行时入口

实现对应 [CuTe `layout.hpp`](https://github.com/NVIDIA/cutlass/blob/main/include/cute/layout.hpp) 的 `right_inverse` / `left_inverse`，不把运行时恰好成立的等式当作静态事实。

| 输入与入口 | 结果与约束 |
|---|---|
| `<TypedInverse<L> as StaticValue>::VALUE` / `<TypedInverse<L, true> as StaticValue>::VALUE` | Zero-sized `Layout` constant; every input shape/stride leaf is static |
| `L.right_inverse()`，L 为 `Layout` | `Result<TypedInverse<L>>`；忽略动态 stride，仅接续静态已知的连续地址链 |
| `L.left_inverse()`，L 为 `Layout` | `Result<TypedInverse<L,true>>`；coalesce 后的 stride 必须全静态，shape 可混合 |
| Free functions `right_inverse` / `left_inverse` | Coalesce, sort, and invert with the current values, returning a `DynLayout` |
| `dense_left_inverse` / `dense_right_inverse` | 保留运行时枚举接口，适用于任意 `LayoutMap` |

`TypedInvert<LEFT>` 共用以下准备过程：先用 `CoalesceFold` 展平并化简，再由 `Attach` 为每个叶子保留 `(shape, 原逻辑前缀积)`。右逆过滤动态 stride，左逆要求 stride 类型可由 `StaticValue` 唯一确定；两者复用 complement 的稳定 `Sort`，排序时携带原逻辑前缀积，不能在排序后重新计算它。

`RightFold` 只有在 `stride` 和 `current` 的静态值相等时才产出叶子。因此 `(DynInt(3), Int<2>):(Int<1>, Int<3>)` 的类型化右逆为 `DynInt(3):Int<1>`，而转换成 `DynLayout` 后可按当前数值合并成 `6:1`。动态 shape 被接入后，`current` 变为动态类型，后续 stride 不能凭这次运行的数值继续接入。

`LeftFold` 根据已排序的静态 stride 计算 `gap = d/covered`，结果 stride 来自原逻辑前缀积；动态 shape 只影响前缀积、重叠检查及最后的 extent。与运行时受检入口一致，负 stride、非广播重叠和不可整除被拒绝；动态重叠返回 `NonInjective`，乘积溢出返回 `Overflow`。这是比上游准逆路径更严格的输入检查。广播仍按准逆处理，不声称恢复所有广播坐标。

收尾复用 `CoalesceFold` / `FinishCoalesce` 和 `UnwrapLayout`，单模式结果恢复为整数 profile。空 tuple 按标量约定返回 `1:0`；empty layouts are inverted like upstream (e.g. `0:_1` gives `0:_1`). 静态非法整除或运算溢出在实例化时拒绝；reading a fully static result through `StaticValue` skips the final runtime size and address-range validation, as for the other operations.类型化路径不经过固定容量数组，输入和输出均可超过 `MAX_RANK`。

契约集中在 [`tests/inverse_contract.rs`](../tests/inverse_contract.rs)：保留原有结构逆、枚举逆与容量边界测试，并验证常量类型、混合静态性规则、动态错误及 33 模式的独立位反转原像公式。

## 第四部分 · 分块 `divide`

本部分讲解 [`src/algebra/divide.rs`](../src/algebra/divide.rs) 中的 `logical_divide` / `zipped_divide` / `tiled_divide` / `flat_divide`、逐模式版本 `*_divide_modes`，以及它们共用的 [`src/algebra.rs`](../src/algebra.rs) 中的 `TileRest` 与 `Grouping`。

本部分记被分块的布局为 `L`，tile（也叫 tiler）为 `T`。

### 22. 分块在做什么

把 `24:1` 看成一个 6 行 4 列、按列存放的矩阵：地址 `a = 行 + 6·列`。现在想把它切成 2×2 的小块，并且希望之后能用两个坐标来访问：

- **块内坐标**：在一个 2×2 小块里的第几个元素；
- **块号**：第几个小块。

`T = (2,2):(1,6)` 恰好描述"第 0 块"占用的地址 `{0, 1, 6, 7}`（行 0–1、列 0–1）。剩下的问题是：其余的块从哪里开始？下图标出每个格子属于第几块：

```text
        列0 列1 列2 列3
行0      0   0   3   3
行1      0   0   3   3
行2      1   1   4   4
行3      1   1   4   4
行4      2   2   5   5
行5      2   2   5   5
```

"其余块的起点"就是 `T` 留下的空隙，也就是 [第二部分](#第二部分--补布局-complement) 的补布局 `complement(T)`。所以分块的结果是一个二元组：

```text
(块内坐标 t, 块号 r)  ↦  T(t) + R(r)          R = complement(T)
```

再把这个"逻辑下标"送进 `L` 求地址（这里 `L = 24:1` 是恒等映射，看不出效果；第 24.2 节会看到 `L` 不平凡时的情形）。这就是开头那行公式。

一句话记忆：**divide = 块内坐标 + 块号，块号的步长由补布局给出**。

### 23. 共用模型：`TileRest`

#### 23.1 定义

`TileRest<T, R>` 是"tile 地址加 rest 地址"的拼接布局。它的定义域是 `T` 与 `R` 定义域的**笛卡尔积**（所有 `(t, r)` 配对），映射为

```text
TileRest(t, r) = T(t) + R(r)
size           = size(T) · size(R)
```

线性下标 `i` 按左优先混合进制拆成两位，**tile 位最快变化**：

```text
t = i % size(T)
r = i / size(T)
index(i) = T.index(t) + R.index(r)
```

小例子（契约测试中的数据）：`T = (2,3):(1,8)`，`R = (2,2):(2,24)`，`size = 6 · 4 = 24`。

```text
i = 7  →  t = 7 % 6 = 1,  r = 7 / 6 = 1
T(1) = 1（t 解码为 (1,0)）,  R(1) = 2（r 解码为 (1,0)）
index(7) = 1 + 2 = 3
```

| 数学对象 | Rust | 位置 |
|---|---|---|
| `(t, r) ↦ T(t) + R(r)` | `TileRest<T, R>` 与它的 `LayoutMap` 实现 | [`src/algebra.rs`](../src/algebra.rs) |
| 括号方式 | `Grouping::{Logical, Zipped, Tiled, Flat}` | 同上 |
| `L ∘ TileRest` | `Divided<A, B> = Composition<A, TileRest<B, DynLayout>>` | [`src/algebra/divide.rs`](../src/algebra/divide.rs) |
| `TileRest(block, complement ∘ tiler)` | `Produced<A, B> = TileRest<A, Composition<DynLayout, B>>` | [`src/algebra/product.rs`](../src/algebra/product.rs) |

**为什么这样**：divide 和 product 本质上都是"一份 tile 地址 + 一份偏移地址"。把这件事单独做成一个类型，两族算法只需各自决定 tile 和 rest 是什么。

#### 23.2 四种 `Grouping`：同一个映射，不同的括号

`TileRest` 的叶子顺序永远是"tile 的叶子在前，rest 的叶子在后"，`index(i)` 也与分组无关。`Grouping` 只决定**顶层模式怎么划分**，也就是坐标用几个数来写：

| `Grouping` | 顶层模式 | 上例的 extent | 地址 3 的坐标 |
|---|---|---|---|
| `Logical` | `(tile, rest)` | `(6, 4)` | `(1, 1)` |
| `Zipped` | `(tile, rest)` | `(6, 4)` | `(1, 1)` |
| `Tiled` | `(tile, rest_0, rest_1, …)` | `(6, 2, 2)` | `(1, 1, 0)` |
| `Flat` | `(tile_0, tile_1, …, rest_0, rest_1, …)` | `(2, 3, 2, 2)` | `(1, 0, 1, 0)` |

这像给同一串数字 `t0 t1 r0 r1` 加不同的括号：`((t0 t1)(r0 r1))`、`((t0 t1) r0 r1)`、`(t0 t1 r0 r1)`。线性下标、地址都不变，变的只是"多维坐标该怎么写"。

单个 tiler 时 `Logical` 与 `Zipped` 完全相同。它们的区别只在 tiler 是**逐模式的 tuple**时出现（第 25 节）：

- logical 按原模式分组：`((t0, r0), (t1, r1))`，第 m 个模式内部是"该模式的块内坐标、块号"；
- zipped 把 tile 和 rest 分别收拢：`((t0, t1), (r0, r1))`。

这两种写法互为转置。CuTe 把"按模式成对"变成"按角色成组"的操作叫 **unzip**，反方向叫 **zip**：

```text
zip  ((a0, a1), (b0, b1)) = ((a0, b0), (a1, b1))
unzip((a0, b0), (a1, b1)) = ((a0, a1), (b0, b1))
```

就是把一张 2×n 的表转置成 n×2。CuTe 的 `zipped_divide` 是 `tile_unzip(logical_divide(...))`；第五部分的 `blocked_product` 是对 `logical_product` 的两组做 `zip`。

#### 23.3 其余方法为什么这样写

- **`index_extended`**：先尝试仿射形式（`as_affine()`）；没有仿射形式或叶子/分组数超过 `MAX_RANK`（`Err(CapacityExceeded)`）时走分量回退：若 `rest` 是 0 叶子的标量 `():()`，最后一个叶子属于 `tile`，直接委托 `tile.index_extended(i)`；否则 tile 位仍取模，只让最慢的 rest 位进入扩展域。这与"扩展域只延伸最后一个叶子"的约定一致。
- **`address_range`**：定义域是笛卡尔积、地址是两者之和，所以 `min = min_T + min_R`、`max = max_T + max_R`，端点逐项相加就是**精确**区间，无需枚举。
- **`as_affine`**：两侧都有仿射形式时，把 tile 的叶子和 rest 的叶子依次拼起来，再按 `Grouping` 写入顶层分组。
- **`is_injective`**：有仿射形式就走步长分离证明，否则（或叶子超过 `MAX_RANK`）退回 `is_injective_by_enumeration`。

一句话记忆：**`TileRest` 只定义一个映射；`Grouping` 只改坐标的写法，不改地址**。

### 24. 单 tiler：`logical_divide`

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

四个公开函数 `logical_divide`、`zipped_divide`、`tiled_divide`、`flat_divide` 只差最后一个 `Grouping` 参数。返回值 `Divided<A, B>` 是**惰性的组合视图**，需要扁平 shape/stride 时调用 `.as_affine()`。

#### 24.1 走一遍 `logical_divide(24:1, (2,2):(1,6))`

1. `coalesce(24:1) = 24:1`，cotarget `[24]`。
2. `complement_shape((2,2):(1,6), [24]) = (3,2):(2,12)`，与 [第 10.6 节](#106-走一遍-l--2216) 的过程完全相同。
3. inner = `TileRest((2,2):(1,6), (3,2):(2,12))`。
4. `L = 24:1` 是恒等映射，组合不改变任何东西，结果为

```text
logical_divide(24:1, (2,2):(1,6)) = ((2,2),(3,2)) : ((1,6),(2,12))
```

按四种 `Grouping` 取 `.as_affine()`，叶子都是 `(2,2,3,2):(1,6,2,12)`，顶层分组分别为 `[2,2]`、`[2,2]`、`[2,1,1]`、`[1,1,1,1]`。

地址公式为 `t0 + 6·t1 + 2·r0 + 12·r1`，正好是第 22 节的图：

```text
i  = 0..3  → 块 0：0, 1, 6, 7
i  = 4..7  → 块 1：2, 3, 8, 9
…
i  = 20..23 → 块 5：16, 17, 22, 23
```

契约测试逐个检查了这 24 个地址，确认它们恰好覆盖 `[0, 24)` 且互不重复。

#### 24.2 为什么 cotarget 是 `shape(coalesce(L))` 而不是 `size(L)`

取 [第 6 节](#6-图示同一个-a三种-b) 的 `L = (4,6):(6,1)`（4 行 6 列，行优先），`T = 3:1`（每块是一列中的连续 3 行）。4 行不能被 3 整除，需要补齐。

**如果用 `size(L) = 24`**：`complement(3:1, 24) = 8:3`。组合 `L ∘ 8:3` 正是 [第 6.2 节](#62-步长不整除规范化结果与逐点求值不同) 的例子，结果 `(2,4):(18,1)`：两个行块 × **4 列**。列 4、5 整个丢了。

**用 `shape(L) = [4,6]`**：按 [第 11.2 节](#112-cotarget-是-tuple-时为什么结果不同) 的逐叶折叠，`complement_shape(3:1, [4,6]) = 12:3`。组合 `L ∘ 12:3` 在第 0 模式内取 `ceil(4/3) = 2` 步、丢弃余数，剩下的 6 全部进入列模式，得到 `(2,6):(18,1)`：两个行块 × **6 列**。

```text
logical_divide((4,6):(6,1), 3:1) = (3, (2,6)) : (6, (18,1))      size = 36
```

每个格子所属的块号（`*` 行是补出来的行，地址 24–35）：

```text
        列0 列1 列2 列3 列4 列5
行0      0   2   4   6   8  10
行1      0   2   4   6   8  10
行2      0   2   4   6   8  10
行3      1   3   5   7   9  11
行4*     1   3   5   7   9  11
行5*     1   3   5   7   9  11
```

前 6 个线性下标的地址是 `0, 6, 12, 18, 24, 30`：块 0 是第 0 列的行 0–2，块 1 是第 0 列的行 3–5。

**为什么要先 `coalesce`**：coalesce 只在 `L` 真正不连续的地方保留模式边界，补齐只发生在这些边界上。`(4,6):(1,4)`（列优先）是 24 个连续地址，coalesce 成 `24:1`，所以 `T = 5:1` 在整条 24 上补到 25：

```text
logical_divide((4,6):(1,4), 5:1) = (5,5):(1,5)        // 不按"每列 4 行"补齐
logical_divide((4,6):(1,4), 4:2) = (4,(2,3)):(2,(1,8))
```

第二行的 rest 就是 `complement(4:2, 24) = (2,3):(1,8)`：`T` 取偶数地址，rest 先补奇数（`2:1`），再每 8 重复一次（`3:8`）。

#### 24.3 不整除：补齐与扩展域

```text
logical_divide(6:1, 4:1) = (4,2):(1,4)        size = 8，地址 0..7
```

`complement_shape(4:1, [6])` 的尾部是 `ceil(6/4) = 2`，于是 rest 为 `2:4`，结果有 8 个元素。地址 6、7 超出了 `L` 的逻辑域，它们落在 `L` 的扩展域里（最后一个模式不取模）。

**为什么允许 size 变大**：这是 CuTe 的语义。分块之后通常还要按块号切片，补齐后每块大小相同，越界的部分由使用者（例如带谓词的拷贝）屏蔽。

`L` 没有仿射形式（例如查表映射）时，既没有 shape 可以 coalesce，也没有扩展域：

- cotarget 只能用 `size(L)`，即 `complement(T, size(L))`；
- 刚好整除时正常工作：对 6 个值的查表 `[10,20,30,40,50,60]` 用 `3:1` 分块，rest 为 `2:3`，坐标 `(2,1)` 对应逻辑下标 `2 + 3 = 5`，值为 60；
- 不整除时（`4:1`），rest 会越出 `[0, 6)`，`composition` 的值域检查返回 `CoordinateOutOfBounds`。

#### 24.4 惰性视图与 `as_affine`

`Divided<A, B>` 是 `Composition<A, TileRest<B, DynLayout>>`，构造时只做值域检查和规范化判定（[第 2 节](#2-构造阶段)）。

- When `L` is affine, `as_affine()` returns the CuTe normalized layout, and `rank`/`extent` follow its regrouped top-level modes;
- `L` 非仿射时返回 `None`，但仍可逐点求值。

**为什么不直接返回 `GroupedLayout`**：`L` 可以是任意 `LayoutMap`（例如 swizzle），这时根本没有扁平形式。惰性视图让仿射与非仿射走同一个 API。

一句话记忆：**rest = 按 `L` 的模式形状补出来的补布局；不整除就补齐，越界部分进扩展域**。

### 25. 逐模式 tiler：`*_divide_modes`

CuTe 的 tiler 还可以是一个 tuple：第 m 项只负责分 `L` 的第 m 个顶层模式。例如对一个矩阵用 `(4:1, 3:1)` 分块，意思是"行方向每 4 个一块、列方向每 3 个一块"。本项目用 `&[DynLayout]` 表示这种 tuple（整数 `n` 写成 `n:1`），对应 `logical_divide_modes`、`zipped_divide_modes`、`tiled_divide_modes`、`flat_divide_modes`。

#### 25.1 机制

公共部分 `divide_modes` 对每个模式单独做一次第 24 节的分块：

```text
对 L 的每个顶层模式 mode_m：
    若 tiler 没有第 m 项：mode_m 原样进入 rests
    否则：
        rest   = complement_shape(T_m, shape(coalesce(mode_m)))
        tile_m = mode_m ∘ T_m
        rest_m = mode_m ∘ rest
```

结果放在两个 `Modes` 缓冲区里。`Modes` 是在第 13 节的 `Leaves` 上多记一个"每个模式有几个叶子"的数组：`push` 追加一个模式，`mode(m)` 取出第 m 个模式，`merged(range)` 把一段模式合并成一个顶层模式。

然后四个函数只是用不同方式拼装 `tiles` 和 `rests`：

| 函数 | 顶层结构 |
|---|---|
| `logical_divide_modes` | `(tile_0 ∪ rest_0, tile_1 ∪ rest_1, …, 未覆盖的模式…)` |
| `zipped_divide_modes` | `((tile_0, tile_1, …), (rest_0, rest_1, …, 未覆盖的模式…))` |
| `tiled_divide_modes` | `((tile_0, tile_1, …), rest_0, rest_1, …, 未覆盖的模式…)` |
| `flat_divide_modes` | `(tile_0, tile_1, …, rest_0, rest_1, …, 未覆盖的模式…)` |

#### 25.2 走一遍 `L = (8,6):(1,8)`，tiler `(4:1, 3:1)`

`L` 是 8 行 6 列、列优先的矩阵，按 4×3 分块，一共 2×2 块。

| 模式 | `mode_m` | `T_m` | `tile_m = mode_m ∘ T_m` | rest | `rest_m = mode_m ∘ rest` |
|---|---|---|---|---|---|
| 0（行） | `8:1` | `4:1` | `4:1` | `complement_shape(4:1,[8]) = 2:4` | `2:4` |
| 1（列） | `6:8` | `3:1` | `3:8` | `complement_shape(3:1,[6]) = 2:3` | `2:24` |

拼装结果（契约测试中的数据）：

```text
logical : ((4,2),(3,2)) : ((1,4),(8,24))     分组 [2,2]
zipped  : ((4,3),(2,2)) : ((1,8),(4,24))     分组 [2,2]
tiled   : ((4,3), 2, 2) : ((1,8), 4, 24)     分组 [2,1,1]
flat    : (4,3,2,2)     : (1,8,4,24)         分组 [1,1,1,1]
```

logical 与 zipped 正是 23.2 节的转置关系：logical 的第 0 模式是"行内的 (块内行, 块行号)"，zipped 的第 0 模式是"块内的 (行, 列)"。用 zipped 时，固定第 1 个坐标（块号）就得到一整个 4×3 块。

不整除时同样补齐：`(6,5):(1,6)` 按 `(4,2)` 分块，zipped 结果为 `((4,2),(2,3)):((1,6),(4,12))`：行 6 → `2` 个行块（补到 8），列 5 → `3` 个列块（补到 6）。

#### 25.3 tiler 比 rank 短

`L = (4,6,2):(1,4,24)`，tiler `(2:1, 3:1)` 只覆盖前两个模式：

```text
logical : ((2,2),(3,2),2) : ((1,2),(4,12),24)      分组 [2,2,1]
zipped  : ((2,3),(2,2,2)) : ((1,4),(2,12,24))      分组 [2,3]
```

第 2 个模式 `2:24` 没有被分块，原样追加在 rests 末尾。zipped 中它属于"块号"那一组。这与 CuTe 的行为一致。

#### 25.4 运行时 GroupedLayout 的差异：只保留一层分组

CuTe 的 `logical_divide` 在 tuple tiler 下返回深度为 2 的嵌套布局 `((t0, r0), (t1, r1))`。`GroupedLayout` 只有一层分组，所以 logical 版本只能把 `(tile_m, rest_m)` 合成一个顶层模式（分组 `[2,2]`），**每个模式内部 tile 与 rest 的边界不再记录**。

需要这条边界时，可以使用保留完整 profile 的类型化入口（27.1 节），或改用 zipped / tiled / flat 运行时分组。

一句话记忆：**逐模式 tiler = 对每个模式各做一次单 tiler 分块，再按四种方式拼装**。

### 26. 错误与边界

| 输入情况 | 返回 | 来源 |
|---|---|---|
| tile 的 size ≤ 0（如 `0:1`） | `Err(InvalidShape)` | `divide` 入口 |
| tile 含负步长（如 `2:-1`） | `Err(Unsupported)` | `complement` |
| tile 自身重叠（如 `(2,2):(2,3)`） | `Err(NonInjective)` | `complement` |
| `L` 非仿射且 tile 不整除 `size(L)` | `Err(CoordinateOutOfBounds)` | `composition` 值域检查 |
| `_modes`：tiler 比 rank 长 | `Err(StructureMismatch)` | `divide_modes` |
| `_modes`：某模式的组合不满足整除条件 | `Err(NotDivisible)` | `composition_layout` |
| 叶子或模式数超 `MAX_RANK` | `Err(CapacityExceeded)` | `Leaves` / `Modes` |
| `L` 为空（如 `0:1`） | 结果 size 为 0 | |

CuTe 在这些地方大多是静态断言或未定义行为。本项目统一改为返回 `Result`。

### 27. 与上游的对应与流程总览

| CuTe（`include/cute/layout.hpp`） | noodle-cute |
|---|---|
| `logical_divide(layout, tiler)`，tiler 为单个布局 | `logical_divide`，返回惰性 `Divided` |
| 同上，tiler 为 tuple（逐模式 `transform_layout`） | `logical_divide_modes`，结果只保留一层分组 |
| `zipped_divide = tile_unzip(logical_divide(...))` | `zipped_divide` / `zipped_divide_modes` |
| `tiled_divide = zipped_divide(...)(_, repeat<R1>(_))` | `tiled_divide` / `tiled_divide_modes` |
| `flat_divide = zipped_divide(...)(repeat<R0>(_), repeat<R1>(_))` | `flat_divide` / `flat_divide_modes` |

```text
                      logical_divide(L, T)
                               │
                       size(T) ≤ 0 ? ── 是 ──▶ InvalidShape
                               │
                     L 有仿射形式吗？
              ┌────────────────┴────────────────┐
              是                                 否
 rest = complement_shape(T,            rest = complement(T, size(L))
         shape(coalesce(L)))                     │
              └────────────────┬────────────────┘
                               ▼
                 inner = TileRest(T, rest, grouping)
                               ▼
                    composition(L, inner)  ──▶  Divided（惰性）
                               │
             L affine: regroup L∘T and L∘rest by Grouping
                               ▼
              flat leaves + top-level groups (.as_affine())
```

一句话记忆：**先在逻辑下标空间里排好 (块内, 块号)，再整体送进 `L`**。

#### 27.1 divide 的全静态、混合与运行时入口

| 输入与入口 | 结果 |
|---|---|
| `<TypedDivided<L, T, KIND> as StaticValue>::VALUE` | Full-hierarchy `Layout` constant for fully static inputs |
| `Layout` 的 `logical_divide` / `zipped_divide` / `tiled_divide` / `flat_divide` 方法 | `Result<TypedDivided<L,T,KIND>>`，保留静态叶子和动态字段 |
| Free functions `*_divide` and `*_divide_modes` | Single-layout tilers (lazy `Divided`, with the pointwise fallback for non-affine inputs) and `&[DynLayout]` per-mode tilers |

`KIND` is one of `algebra::LOGICAL`, `ZIPPED`, `TILED`, or `FLAT`.

单个类型化 tiler 为 `Layout` 时，`DivideOp` 完整复用 `composition(L, (T, complement(T, shape(coalesce(L)))))`：`CoalesceTarget` 取得类型化 cotarget，complement 保留静态推导，composition 处理弱整除和扩展域。整数 tiler 自动视为 `n:1`。类型化入口遇到组合整除失败直接返回 `NotDivisible`，不会构造惰性回退。

嵌套 tuple tiler 使用 [`typed.rs`](../src/algebra/typed.rs) 的 `Transform` 递归处理相同 profile 的模式，短 tuple 保留尾部，空 tuple 返回原布局。`Unzip` 对应上游 `zip2_by`：按 tiler 的终端收集 tile/rest，未覆盖模式只进入 rest。`Regroup` 只解开顶层模式，不压平内部层级；所以整数 tiler 的 `(tile,rest)` 与单元素 tuple tiler 的 `((tile),(rest))` 不同。这些结构均由类型保留，不受 `GroupedLayout` 的一层分组限制。

公开方法校验输入 size 和最终 `Layout::new`；reading a constant through `StaticValue` skips the final runtime size and address-range checks.只有运行时数组表示受 `MAX_RANK` 限制。原有动态分组拼装收敛到 `group_modes`，它还被运行时 tuple product 复用。

[`tests/divide_contract.rs`](../tests/divide_contract.rs) 包含迁移后的动态契约，以及四种静态分组、混合元数据、嵌套/部分/空 tuple、整数与 singleton profile 差异、扩展域、动态错误和超过 `MAX_RANK` 的结果。编译期拒绝示例见模块 rustdoc。

## 第五部分 · 乘积 `product`

本部分讲解 [`src/algebra/product.rs`](../src/algebra/product.rs) 中的 `logical_product` / `zipped_product` / `tiled_product` / `flat_product` 与 `blocked_product` / `raked_product`。

本部分记被重复的小布局为 `block`，描述"怎样重复"的布局为 `tiler`。

### 28. 乘积在做什么

分块是"把大的切成小的"，乘积反过来：**把一个 block 当成瓷砖，按 tiler 的样式铺很多份**。

- block 描述一份瓷砖内部的地址；
- tiler 描述"第几份瓷砖"如何排列；
- 每份瓷砖的起点必须落在 block 没有用到的地址上，否则会重叠。

"block 没有用到、可以放下一份的位置"仍然是补布局。于是

```text
logical_product(block, tiler) = (block, complement(block) ∘ tiler)
```

第二项读作：tiler 产出"第几个空位"，`complement(block)` 把它翻译成真实的起点地址。结果仍是一个 `TileRest`：

```rust
let complement = complement_extended(block.clone())?;
let rest = composition(complement, tiler)?;
let result = TileRest::new(block, rest, grouping);
result.size()?;
Ok(result)
```

`Produced<A, B> = TileRest<A, Composition<DynLayout, B>>`。最后的 `result.size()?` 提前检查 `size(block) · size(tiler)` 是否溢出。block 必须有仿射形式（要对它求补），tiler 可以是任意 `LayoutMap`。

### 29. 为什么用 `complement_extended`（不带 cotarget）

divide 知道目标多大（`shape(L)`），product 不知道：要铺多少份完全由 tiler 决定，tiler 的值可以任意大。所以这里的补布局**不能有 cotarget**，而要能无限延伸。

[第 12 节](#12-complement_extended标记重复步长) 的 `complement_extended` 正是为此设计的：它保留 block 内部的空隙，再追加一个 `1:current` 尾叶子。这个尾叶子 size 为 1，不改变逻辑大小，但在扩展域里提供"下一整段"的步长。tiler 的值越过补布局的逻辑大小时，就沿着这个步长继续往后放。

**反例：如果丢掉 `1:current`**。取 `block = (2,2):(1,4)`，地址 `{0, 1, 4, 5}`：

- 正确的补布局是 `(2,1):(2,8)`，扩展域里依次给出 `0, 2, 8, 10, 16, 18, …`；
- 若 coalesce 成 `2:2`，扩展域会按 `0, 2, 4, 6, …` 延伸，第 3 份瓷砖从地址 4 开始，和 block 自己的 `{4, 5}` 重叠。

这也是 [第 3.1 节](#31-compose_grouped) 中 `composition` 对 A 用 `coalesce_extended` 而不是普通 `coalesce` 的原因：末尾 shape 为 1 的叶子的步长必须保留。

**与上游的关系**：CuTe 的 `logical_product` 使用无 cotarget 的 `complement(block)`，等价于 `complement(block, cosize(block))`。cosize 不超过 `current`，尾部的 `ceil_div(cosize, current)` 恰为 1，只剩一个 `1:current` 叶子，与本项目的 `complement_extended` 一致。

### 30. 走一遍 `logical_product((2,2):(1,4), (2,3):(1,2))`

把地址看成 4 行、按列存放的矩阵（`a = 行 + 4·列`）。block 占行 0–1、列 0–1；tiler 是 2×3 的列优先排列。

**第一步：`complement_extended(block)`**

| 步骤 | 叶子 | 进入时 `current` | `gap = d / current` | 动作 | 离开时 `current` |
|---|---|---|---|---|---|
| 1 | `2:1` | 1 | 1 | 无空隙，跳过 | 2 |
| 2 | `2:4` | 2 | 2 | push `2:2` | 8 |

前缀 `2:2` 的 size 不为 1，追加 `1:8`：`complement_extended = (2,1):(2,8)`。

**第二步：`(2,1):(2,8) ∘ (2,3):(1,2)`**，按 [第 3.2 节](#32-compose_leaf) 逐叶组合：

```text
A 的模式：模式0 (cs=2, cd=2)，最后一个模式 (1, cd=8)

叶子 2:1：模式0：next_shape = ceil(2/1) = 2，new_shape = min(2, 2) = 2
          产出 2 : 1·2 = 2:2，rest_shape = 1，B 走完
叶子 3:2：模式0：next_shape = ceil(2/2) = 1，步长跨过整个模式，只更新 rest_stride = 1
          尚未产出叶子 → 剩余 3 放入最后一个模式：3 : 1·8 = 3:8
rest = (2,3):(2,8)
```

**结果**：

```text
logical_product((2,2):(1,4), (2,3):(1,2)) = ((2,2),(2,3)) : ((1,4),(2,8))
```

六份瓷砖的起点是 `2·r0 + 8·r1 = 0, 2, 8, 10, 16, 18`。每格所属的瓷砖号：

```text
        列0 列1 列2 列3 列4 列5
行0      0   0   2   2   4   4
行1      0   0   2   2   4   4
行2      1   1   3   3   5   5
行3      1   1   3   3   5   5
```

线性遍历时 block 坐标最快变化，契约测试中的地址序列为：

```text
0,1,4,5,  2,3,6,7,  8,9,12,13,  10,11,14,15,  16,17,20,21,  18,19,22,23
```

24 个地址恰好铺满 `[0, 24)`。注意 tiler 的第 2 份（下标 2）越过了补布局的逻辑大小 2，正是 `1:8` 尾叶子把它送到了地址 8。

### 31. 更多例子

以下数据来自契约测试（由 CuTe oracle 生成）：

```text
block       tiler           logical_product
4:2         (2,2):(1,2)     (4,(2,2)):(2,(1,8))     // 补布局 (2,1):(1,8)：先填奇数，再每 8 重复
3:1         4:2             (3,4):(1,6)             // 补布局 1:3；tiler 隔一个取一个空位
(2,2):(2,1) 3:1             ((2,2),3):((2,1),4)     // block 稠密覆盖 [0,4)，补布局 1:4
4:0         3:1             (4,3):(0,1)             // 全广播 block，补布局 1:1
```

第二行值得细看：`complement_extended(3:1) = 1:3` 表示"每隔 3 个地址可以放一份"。tiler `4:2` 取第 0、2、4、6 个空位，所以起点是 `0, 6, 12, 18`，中间留出了一份的空档。

tiler 不必是仿射布局。若 tiler 是一个值为 `[0, 2, 1]` 的查表映射，三份瓷砖的起点依次是补布局在 0、2、1 处的值 `0, 8, 2`，结果只能惰性求值（`as_affine()` 为 `None`），地址序列为 `0,1,4,5, 8,9,12,13, 2,3,6,7`。

### 32. `zipped_product` / `tiled_product` / `flat_product`

与 divide 一样，四个乘积函数共用同一个 `TileRest`，只差 `Grouping`。以第 30 节为例，叶子都是 `(2,2,2,3):(1,4,2,8)`：

| 函数 | 顶层分组 | 坐标写法 |
|---|---|---|
| `logical_product` | `[2,2]` | `(block 坐标, 瓷砖号)` |
| `zipped_product` | `[2,2]` | 同上（单个 tiler 时与 logical 相同） |
| `tiled_product` | `[2,1,1]` | `(block 坐标, tiler 第 0 轴, tiler 第 1 轴)` |
| `flat_product` | `[1,1,1,1]` | 全部展开 |

`tiled_product` unpacks the top-level modes of the composed rest `complement(block) ∘ tiler`, as upstream `result(_, repeat<R1>(_))` does. A tuple tiler keeps one mode per tiler axis; an integral tiler yields one mode per leaf of the composition. For example `complement(2:2) = (2,1):(1,4)` composed with `6:1` is `(2,3):(1,4)`, so `tiled_product(2:2, 6:1) = (2,2,3):(2,1,4)` with rank 3. The lazy `Composition` reports this structure through its normalized layout.

### 33. `blocked_product` 与 `raked_product`

`logical_product` 的结果是 `(block 坐标, 瓷砖号)`，不是一个"大矩阵"的 `(行, 列)`。如果 block 和 tiler 都是二维的，通常希望结果也是二维：第 0 模式是"大矩阵的行"，由 block 的行和 tiler 的行组成；第 1 模式同理。这就是对两组做 23.2 节的 **zip**：

```text
logical : ((b0, b1), (r0, r1))
zip     : ((b0, r0), (b1, r1))      ← blocked_product
          ((r0, b0), (r1, b1))      ← raked_product
```

两者的叶子完全相同，只是每个模式内部谁在前（谁变化更快）不同。

#### 33.1 一维小例子

`block = 3:1`，`tiler = 2:1`。`complement_extended(3:1) = 1:3`，`rest = 1:3 ∘ 2:1 = 2:3`。

```text
blocked = (3:1, 2:3) = (3,2):(1,3)    位置 0..5 → 地址 0,1,2,3,4,5
raked   = (2:3, 3:1) = (2,3):(3,1)    位置 0..5 → 地址 0,3,1,4,2,5
```

地址 = 块内下标 + 3·瓷砖号。按位置看属于哪份瓷砖：

```text
blocked: 0 0 0 1 1 1      // 每份瓷砖连成一段
raked  : 0 1 0 1 0 1      // 两份瓷砖交错排列
```

"raked"是"耙"的意思：瓷砖的元素像耙齿一样均匀插开，而不是连成一块。

#### 33.2 机制：`rank_product`

```rust
let rank = block.rank().max(tiler.rank());
let complement = complement_extended(block)?;
for m in 0..rank {
    let b = if m < block.rank() { block.mode(m)? } else { 1:0 };
    let r = if m < tiler.rank() { complement ∘ tiler.mode(m) } else { 1:0 };
    let (first, second) = if raked { (r, b) } else { (b, r) };
    // 依次压入 first、second 的叶子；第 m 个顶层模式的叶子数 = 两者叶子数之和
}
```

（伪代码，省略了错误处理。）要点：

1. **只求一次补布局**，对整个 block。
2. **逐模式组合**：`complement ∘ (T0, T1, …) = (complement ∘ T0, complement ∘ T1, …)`，即 [第 3.1 节](#31-compose_grouped) 的右分配律，所以可以逐模式算。
3. **rank 不同时补 `1:0`**：CuTe 用 `append<R>` 把两边补到相同 rank，补入的模式是 `1:0`。它对补布局没有影响（`complement` 会 filter 掉），组合后仍是 `1:0`。结果中**保留**这些 shape 为 1 的叶子，使模式结构与 CuTe 一致。
4. **Pad before composing**: the typed `blocked_product`/`raked_product` also wrap the tiler into modes and pad it with `1:0` to the block's rank *before* composing, matching upstream `logical_product(append<R>(block), append<R>(tiler))`. A tiler mode whose composition has several modes therefore stays nested inside one result mode: `blocked_product((2,2):(2,8), 8:1) = ((2,(2,2,2)),(2,1)) : ((2,(1,4,16)),(8,0))`.
5. **Negative tiler strides** compose through `signum` like any other composition, e.g. `blocked_product(2:1, 2:-1) = ((2,2)):((1,-2))`.

#### 33.3 二维例子：`col = (2,2):(1,2)`，`t23 = (2,3):(1,2)`

`col` 稠密覆盖 `[0, 4)`，`complement_extended(col) = 1:4`。

| 模式 | `b_m` | `r_m = 1:4 ∘ t23_m` | blocked `(b_m, r_m)` | raked `(r_m, b_m)` |
|---|---|---|---|---|
| 0 | `2:1` | `2:4` | `(2,2):(1,4)` | `(2,2):(4,1)` |
| 1 | `2:2` | `3:8` | `(2,3):(2,8)` | `(3,2):(8,2)` |

```text
blocked_product(col, t23) = ((2,2),(2,3)) : ((1,4),(2,8))      extent (4, 6)
raked_product  (col, t23) = ((2,2),(3,2)) : ((4,1),(8,2))      extent (4, 6)
```

两者都可以看成 4 行 6 列的大矩阵，每个地址 = 块内地址 + 4·瓷砖号。下面画出每个位置 `(行, 列)` 属于第几份瓷砖：

```text
blocked                         raked
     0  1  2  3  4  5                0  1  2  3  4  5
行0  0  0  2  2  4  4           行0  0  2  4  0  2  4
行1  0  0  2  2  4  4           行1  1  3  5  1  3  5
行2  1  1  3  3  5  5           行2  0  2  4  0  2  4
行3  1  1  3  3  5  5           行3  1  3  5  1  3  5
```

blocked 中每份瓷砖是一个连续的 2×2 方块；raked 中第 0 份瓷砖的四个元素在 `(0,0)`、`(0,3)`、`(2,0)`、`(2,3)`，均匀分散。同一个坐标 `(1,0)`：blocked 给出地址 1（瓷砖 0 的第 1 个元素），raked 给出地址 4（瓷砖 1 的第 0 个元素）。

注意 blocked 的叶子 `(2,2,2,3):(1,4,2,8)` 恰好与第 30 节 `logical_product` 的叶子相同，但含义不同：这里第 0 模式是"大矩阵的行"，第 30 节的第 0 模式是"block 坐标"。

#### 33.4 rank 不同的例子

```text
blocked_product(4:1, t23) = ((4,2),(1,3)) : ((1,4),(0,8))
raked_product  (4:1, t23) = ((2,4),(3,1)) : ((4,1),(8,0))
blocked_product(col, 3:1) = ((2,3),(2,1)) : ((1,4),(2,0))
raked_product  (col, 3:1) = ((3,2),(1,2)) : ((4,1),(0,2))
```

缺的模式补成 `1:0`，并原样留在结果里。

#### 33.5 应用：线程-值（TV）布局

CuTe 的 `make_tiled_copy` 用 `raked_product(thr, val)` 描述"一个线程块拷贝的矩形区域里，每个位置归哪个线程的第几个值"：

- `thr` 把线程坐标映射为线程号，`val` 把值坐标映射为值号；
- `complement_extended(thr)` 的步长恰为 `size(thr)`，所以结果把矩形内的位置映射为 `线程号 + size(thr)·值号`；
- 再对它求 [第 19 节](#19-right_inverse贪心拼接连续地址前缀) 的 `right_inverse`，就得到反方向的 `(线程, 值) → 位置`，即 TV 布局。

契约测试中的例子：`thr = (4,2):(2,1)`，`val = (2,2):(1,2)`，`raked_product` 为 `((2,4),(2,2)):((8,2),(16,1))`，最终的 TV 布局为 `((2,4),(2,2)):((16,2),(1,8))`。测试逐个检查了每个 `(线程, 值)` 经 TV 布局再经 `raked_product` 后回到 `线程号 + 8·值号`。

### 34. 错误与边界

| 输入情况 | 返回 | 来源 |
|---|---|---|
| block 含负步长（如 `2:-1`） | `Err(Unsupported)` | `complement_extended` |
| block 为空（如 `0:1`） | `Err(Unsupported)` | `complement_extended` |
| block 自身重叠（如 `(2,2):(2,3)`） | `Err(NonInjective)` | `complement_extended` |
| A non-affine tiler produces negative values | `Err(CoordinateOutOfBounds)` | pointwise `composition` fallback |
| `size(block) · size(tiler)` 溢出 | `Err(Overflow)` | `product` 末尾的 `result.size()?` |
| blocked/raked：某模式组合不整除（如 `blocked_product(4:2, 2:3)`） | `Err(NotDivisible)` | `composition_layout` |
| 叶子超过 `MAX_RANK`（如两个 rank 17 的布局） | `Err(CapacityExceeded)` | `Leaves` |

`blocked_product(4:2, 2:3)` 失败的原因：补布局 `(2,1):(1,8)` 的第 0 模式大小为 2，tiler 步长 3 大于 2 却不是 2 的倍数，触发 [第 3.2 节](#32-compose_leaf) 的 stride 整除条件；CuTe 在这里是整除断言失败。

### 35. 与上游的对应及与 divide 的关系

| CuTe（`include/cute/layout.hpp`） | noodle-cute |
|---|---|
| `logical_product = make_layout(block, composition(complement(block), tiler))` | `logical_product`，返回惰性 `Product`；补布局用 `complement_extended` |
| `zipped_product = tile_unzip(logical_product(...))` | `zipped_product`（单个 tiler 时与 logical 相同） |
| `tiled_product`、`flat_product` | 同名函数，只改 `Grouping` |
| `blocked_product = zip(get<0>(r), get<1>(r))`，`r = logical_product(append<R>(block), append<R>(tiler))` | `blocked_product`，立即返回一层分组的 `GroupedLayout` |
| `raked_product = zip(get<1>(r), get<0>(r))` | `raked_product`，同上 |

偏离点：

- 单 tiler 的乘积返回惰性 `TileRest`，tiler 可以非仿射；需要扁平布局时调用 `.as_affine()`。
- blocked/raked 要求两侧都是仿射布局，结果只保留一层分组，每个模式内部 block 与 rest 的边界不再记录。
- CuTe 中的断言与未定义行为改为 `Result` 错误（第 34 节）。

**divide 与 product 互为逆操作**。契约测试验证了：

```text
logical_product((2,2):(1,4), 6:1) = logical_divide(24:1, (2,2):(1,4)) = ((2,2),(2,3)) : ((1,4),(2,8))
```

两边算法不同，但算的是同一件事：

- divide 中 rest 为 `complement_shape(block, [24]) = (2,3):(2,8)`；
- product 中 rest 为 `complement_extended(block) ∘ 6:1 = (2,1):(2,8) ∘ 6:1 = (2,3):(2,8)`。

前者从"总共 24 个"出发，求需要几份；后者从"要 6 份"出发，铺出 24 个。这个布局是 `[0, 24)` 上的双射，`left_inverse` 与 `right_inverse` 相同，并且都能把它还原。

一句话记忆：**product = block + 补布局 ∘ tiler；blocked 让瓷砖连成块，raked 让瓷砖交错**。

#### 35.1 product 的全静态、混合与运行时入口

| 输入与入口 | 结果与分组 |
|---|---|
| `<TypedProductLayout<L, T, KIND> as StaticValue>::VALUE` | Zero-sized `Layout` constant keeping the full profile |
| `Layout` 上的六个同名方法 | `Result<TypedProductLayout<L,T,KIND>>` |
| Free functions `logical/zipped/tiled/flat_product`, `blocked_product`, `raked_product` | Single tilers keep the lazy `Produced`; blocked/raked return a `GroupedLayout` |
| `*_product_modes(&block, &[DynLayout])` | 每个被覆盖模式独立求补和组合，保留未覆盖尾部模式 |

`KIND` is one of `algebra::LOGICAL`, `ZIPPED`, `TILED`, `FLAT`, `BLOCKED`, or `RAKED`. The last two take a `Layout` tiler; the other four also accept integers and nested tuples.

`ProductOp` 复用不带 cotarget 的类型化 complement 和 composition，避免把重复步长误换成 cosize。tuple 遍历与 `tile_unzip` 和 divide 共用 `Transform` / `Unzip` / `Regroup`，因此 logical 保留各模式内部的 block/rest 边界，zipped 转置时保留嵌套 guide。空 tuple 是不变换的 tiler，`Layout<Nil,Nil>` 则是实际的标量布局 tiler，两者 profile 不同。

blocked/raked first wrap the tiler into modes and pad it with `1:0` to the block's rank (`PadModes`), compose, and then apply `IntoModes` and `ZipModes` to the two groups. 短侧的每个缺失模式补 `1:0`；这些单位轴保留在最终 profile 中。blocked 每模式为 `(block_m,rest_m)`，raked 交换二者；两侧均为空 tuple 布局时仍是 rank-0 标量。此处的 zip 不做额外 coalesce。

类型化值入口检查最终布局大小和地址范围；补布局和组合失败直接返回相应错误，静态已知的非法运算由类型约束拒绝。Reading a constant through `StaticValue` skips the final runtime validation.运行时通用乘积继续保留不可整除或非仿射 tiler 的惰性回退。`*_product_modes` 仅处理仿射模式，组合失败直接报错，运行时输出保留一层分组。

[`tests/product_contract.rs`](../tests/product_contract.rs) 保留原有乘积、TV 布局及容量边界契约，并新增六种静态分组、混合元数据、嵌套/部分/空 tiler、补轴次序、运行时 tuple product、零域/溢出/整除错误及超过 `MAX_RANK` 的类型化输出。跨模块 divide/product/inverse 往返定律继续由 [`algebra_contract.rs`](../tests/algebra_contract.rs) 覆盖。

#### 自测

`blocked_product(3:1, 2:1)` 与 `raked_product(3:1, 2:1)` 在位置 4 的地址分别是多少？

答案：`blocked = (3,2):(1,3)`，位置 4 拆成 `(1, 1)`，地址 `1 + 3 = 4`。`raked = (2,3):(3,1)`，位置 4 拆成 `(0, 2)`，地址 `0 + 2 = 2`。
