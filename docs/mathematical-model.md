# The Mathematical Model of noodle-cute

This document explains the mathematical semantics implemented in the current `noodle-cute/src`. The companion [engineering implementation](engineering-implementation.md) explains how these formulas map onto Rust types, algorithms, and tests. The layouts, thread partitions, and matrix multiplications here are all device-independent descriptions; they do not execute GPU instructions themselves.

The reading order is: coordinates and addresses → layout transformations → thread work distribution → Tensors and expressions. Every mathematical concept is defined where it first appears, and a glossary is provided at the end. Notations such as `Shape:Stride` and `4:2` are layout notation; for example, `4:2` denotes a one-dimensional layout of length 4 and stride 2.

## 1. Understanding a Layout as a Function

A **map** or **function** is a rule that assigns an output to every valid input. The input of a layout is a logical coordinate, and the output is an integer address offset measured in elements:

$$
L:D\longrightarrow\mathbb Z.
$$

Here $D$ is the **domain**, the set of allowed input coordinates; $\mathbb Z$ is the set of integers and is the **codomain** chosen here, the set in which outputs live. The set of addresses actually produced, $L(D)$, is called the **image** or **range**. The codomain is not the allocated storage; a layout may produce negative addresses, and a Tensor adds a starting offset separately and checks the storage bounds.

An ordinary two-dimensional layout can be written as:

$$
S=(s_0,s_1),\qquad D_S=\{0,\ldots,s_0-1\}\times\{0,\ldots,s_1-1\},
\qquad L(c_0,c_1)=c_0d_0+c_1d_1.
$$

Between sets, the symbol $\times$ denotes the **Cartesian product**: take one value from each set to form every possible coordinate combination. $S$ is the shape, $s_j$ is the length of axis $j$, and $d_j$ is its stride. The stride is the number of elements by which the address increases when that coordinate increases by 1.

Suppose there are $n$ leaf axes after flattening; uniformly:

$$
L(c)=\sum_{j=0}^{n-1}c_jd_j.
$$

$\sum$ denotes summation. The coordinate and the stride are each an ordered sequence of numbers and can be regarded as **vectors**; multiplying corresponding components and summing is the **inner product**. The form with a constant offset, $o+\sum c_jd_j$, is an **affine map**; the basic shape/stride layouts of this project take $o=0$. A map that cannot be written as such a weighted sum of the given leaf coordinates is a **non-affine map**. This property refers to the flattened leaf coordinates. Once rewritten as a function of a single linear index, division and remainder appear, and it usually can no longer be written as a simple $ai+b$.

## 2. A Shape Is a Tree

A **tuple** is a finite ordered group of values. Here tuples may contain other tuples, for example:

```text
shape  = (2, (2, 2))
stride = (4, (2, 1))
```

Each top-level member is called a **mode**, also a top-level axis; the innermost integers are called **leaves**. The shape and stride must have the same nesting structure, and every length leaf has a corresponding stride leaf.

Three quantities that are easily confused:

| Name | This example | Meaning |
|---|---:|---|
| rank | 2 | In this library, specifically the number of top-level modes |
| depth | 2 | Maximum nesting level; 0 for a single integer, 1 for a flat tuple |
| leaves | 3 | Number of integers after recursive flattening |

The **size** of a shape is the number of valid coordinates, i.e. the **cardinality** of the set:

$$
\operatorname{size}(S)=\prod_j s_j.
$$

$\prod$ denotes a product. In this example the size is $2\cdot2\cdot2=8$, and the length of the second top-level mode is $2\cdot2=4$. Lengths must be non-negative. If any length is zero, the domain is the **empty set**: there are no valid coordinates and the size is 0.

The empty tuple `()` denotes a **scalar**, a single value with no axes. Its size is 1: this is the convention that **the empty product is 1**. Hence `shape=[]` and `shape=[0]` denote a scalar and an empty domain, respectively.

## 3. Logical Linear Indices and Physical Addresses

### 3.1 Mixed-Radix Indexing

**Linearization** encodes a multi-dimensional coordinate as one integer. This project always makes the leftmost axis vary fastest. Let

$$
p_0=1,\qquad p_j=\prod_{k<j}s_k,
$$

so $p_j$ is the **prefix product** of all lengths before axis $j$, and the logical linear index is:

$$
i=\sum_j c_jp_j,\qquad 0\le i<\operatorname{size}(S).
$$

This is a **mixed-radix** representation: the range of each digit is determined by its own $s_j$, rather than always being 0 to 9 as in decimal. Decoding repeatedly computes, from left to right:

$$
c_j=i_j\bmod s_j,\qquad i_{j+1}=\left\lfloor i_j/s_j\right\rfloor,\qquad i_0=i.
$$

$\bmod$ is the remainder, and $\lfloor x\rfloor$ is the greatest integer not greater than $x$, i.e. rounding down (floor). Index decoding uses non-negative indices and positive divisors, so Rust's integer `/` and `%` agree with these formulas.

`coord_to_linear` performs logical linearization and does not apply address strides. There is no separate function for the reverse decoding: `LayoutMap::index(i)` (`affine_index`) decodes leaf by leaf with remainders and, in the same loop, takes the inner product of the leaf coordinates with the strides.

### 3.2 Same Index, Different Storage Orders

For shape `(2,3)`:

| Logical index `i` | Coordinate `(c0,c1)` | Address under left-compact `(1,2)` | Address under right-compact `(3,1)` |
|---:|---|---:|---:|
| 0 | (0,0) | 0 | 0 |
| 1 | (1,0) | 1 | 3 |
| 2 | (0,1) | 2 | 1 |
| 3 | (1,1) | 3 | 4 |
| 4 | (0,2) | 4 | 2 |
| 5 | (1,2) | 5 | 5 |

A **compact layout** leaves no holes in storage. The left-compact layout takes $d_j=\prod_{k<j}s_k$, and the right-compact layout takes $d_j=\prod_{k>j}s_k$; the latter uses the product of the lengths to the right of that axis, called the **suffix product**. For a matrix indexed as `(row,column)`, they correspond to column-major and row-major order respectively, i.e. adjacent addresses advance first along the row coordinate or the column coordinate.

`index(1)` of the right-compact layout is 3, but logical index 1 still denotes `(1,0)`. Changing the stride does not change the logical-index convention.

### 3.3 Two Entry Points for Nested Coordinates

For `(2,(2,2)):(4,(2,1))`:

- The natural nested coordinate `(1,(0,1))` directly gives $1\cdot4+0\cdot2+1\cdot1=5$; this corresponds to `Layout::at`.
- The top-level coordinate `[1,2]` treats the second mode as a one-dimensional domain of length 4; index 2 decodes to `(0,1)`. This corresponds to `index_of_coord`.
- The global linear index is $1+2\cdot2=5$, corresponding to `index(5)`.

These three entry points yield the same address, but the structure of their inputs differs.

## 4. Address Ranges, Overlap, and Storage Capacity

### 4.1 Closed Address Intervals and `cosize`

A **closed interval** `[a,b]` includes both endpoints; a **half-open interval** `[a,b)` includes the left endpoint but not the right one. For a non-empty affine layout, each axis can independently reach its endpoints, so:

$$
\begin{aligned}
a_{\min}&=\sum_j\min(0,(s_j-1)d_j),\\
a_{\max}&=\sum_j\max(0,(s_j-1)d_j),\\
\operatorname{cosize}(L)&=a_{\max}-a_{\min}+1.
\end{aligned}
$$

`min` and `max` take the minimum and maximum. `cosize` is the **span** of the address interval, including addresses inside the interval that no element uses; these unused addresses are called **holes** or padding. A layout with address holes is also called a **sparse layout**. The `cosize` of an empty domain is 0.

For example, the image of `4:2` is `{0,2,4,6}`, with `size=4` and `cosize=7`. The image of `4:-1` is `{0,-1,-2,-3}`, with `cosize=4`. A **negative stride** denotes reverse traversal; its validity is determined by the Tensor's final storage addresses.

`address_end` only accepts non-negative reachable addresses and returns $a_{\max}+1$. For example, if `4:2` is given an additional address offset of 10, `cosize` is still 7 while `address_end` is 17. The two are not interchangeable.

### 4.2 Injective, Surjective, and Bijective

**Injective** means that different inputs never produce the same output:

$$
L(x)=L(y)\Longrightarrow x=y.
$$

Failing injectivity means addresses **overlap**, also called aliasing. **Broadcasting** is one intentional form of this: an axis of length greater than 1 uses stride zero, so different coordinates read the same element. Broadcasting can be used for read-only Tensors; the current implementation rejects writes through non-injective layouts.

**Surjective** must be defined relative to a specified target set: every address in the target is covered by at least one input. **Bijective** means both injective and surjective, so every target address has a unique source. **Dense coverage** means covering a contiguous address interval without holes; this project's `dense_*_inverse` requires that interval to start at 0.

To check injectivity of an affine layout, first remove axes of length 1, then sort by absolute stride in ascending order. The **absolute value** $|d|$ is the distance from zero. If the largest address difference the existing axes can produce is $w$ and the new axis satisfies $|d|>w$, the address intervals for different coordinates of the new axis do not overlap; then update:

$$
w\leftarrow w+(s-1)|d|.
$$

This is a **sufficient condition**: satisfying it proves injectivity, but failing it does not imply overlap. The implementation therefore falls back to pairwise comparison of addresses. This check does not mistake a sparse layout for an overlapping one.

## 5. Extended Domain: No Remainder on the Last Digit

When a tiling does not evenly divide the original shape, or when composition continues to tile outward repeatedly, intermediate indices may fall outside the original shape. The **extended domain** lets these positions still compute addresses by a uniform rule, for later padding or boundary predicates.

Strict indexing `index` requires $0\le i<\operatorname{size}$. Extended indexing `index_extended` uses the same mixed-radix decoding, but the last leaf directly receives the remaining quotient:

$$
c_j=i_j\bmod s_j\quad(j<n-1),\qquad c_{n-1}=i_{n-1}.
$$

This lets the last digit exceed its original length. For example, `(2,3):(3,1)`:

$$
L_{\mathrm{ext}}(7)=(7\bmod2)\cdot3+\lfloor7/2\rfloor\cdot1=6.
$$

Strict `index(7)` reports out of bounds. `index_extended_of_coord([1,5])` extends each of the two top-level modes separately and gives $1\cdot3+5\cdot1=8$. This differs from first validating `[1,5]` as an ordinary coordinate.

The extended domain still rejects negative inputs. The leading leaves that take a remainder must be positive; the last leaf takes no remainder, so even if its length is zero there may be an algebraic extended result. A leafless scalar only accepts index 0. That the extended formula can be evaluated does not mean the corresponding element belongs to the original tensor, nor that the address may be read or written.

## 6. Simplification, Slicing, and Translation

### 6.1 `flatten`, `coalesce`, and `filter`

**Flattening** (`flatten`) removes nesting boundaries while preserving the order of all leaves; the map from linear index to address on the finite domain is unchanged, while the top-level coordinate interface changes.

**Coalescing** (`coalesce`) removes leaves of length 1 and, when two adjacent axes satisfy $d_1=s_0d_0$, performs:

$$
(s_0,s_1):(d_0,d_1)\longrightarrow(s_0s_1):d_0.
$$

The reason is $c_0d_0+c_1s_0d_0=(c_0+s_0c_1)d_0$. For example, `(2,4):(1,2)` becomes `8:1`. When all axes are removed, the canonical representation is `1:0`.

This simplification guarantees the map on the original finite domain; it should not be taken to mean that the extended domain is always unchanged too. For example, after `(2,1):(2,8)` is simplified to `2:2`, the address of extended index 2 changes from 8 to 4. Composition internally uses `coalesce_extended` instead, which keeps the extended stride carried by the last length-1 leaf.

The typed implementation simplifies using only static information: apart from eliminating unit axes, merging also requires both shapes and both strides to be static values, following CuTe's `bw_coalesce` condition. The runtime `DynLayout` decides using the current values. Both merge from right to left so that the intermediate products of checked arithmetic are formed in the same order; the fully static constant entry point also uses the same type-level fold.

`filter` first changes the length of zero-stride axes to 1, then coalesces. For a non-empty layout, it preserves the set of reachable addresses but removes the duplicate inputs produced by broadcasting, so it does not guarantee that the original logical size or per-element indexing is preserved.

### 6.2 Fixing Coordinates to Obtain a Slice

A **slice** fixes the coordinates on some axes and keeps the rest. After an affine layout fixes the axis set $F$, the original address decomposes as:

$$
o_{\mathrm{slice}}=\sum_{j\in F}c_jd_j,\qquad
L_{\mathrm{sub}}(u)=\sum_{j\notin F}u_jd_j,\qquad
L(c)=o_{\mathrm{slice}}+L_{\mathrm{sub}}(u).
$$

$j\in F$ means axis $j$ belongs to the set $F$. For example, fixing the second axis of `(3,4):(1,3)` at 2 gives `3:1` and offset 6. The code returns the two separately so that the original storage can be reused.

The general `slice` is also defined for non-affine layouts: first embed the sub-coordinate back into the original coordinate, compute the original address, and subtract the address of the subview's origin. "Embedding back" here just means re-inserting the fixed coordinates at their positions.

### 6.3 Two Kinds of Offset

**Translation** adds the same constant to every output:

$$
\operatorname{address\_offset}(L,o)(x)=L(x)+o.
$$

`domain_offset(L,c0)` first validates the original coordinate `c0`, computes $o=L(c_0)$, then applies the address translation above. It keeps the original shape; it does not truncate a leading segment and automatically shrink the Tensor. For the affine formula over independent leaf coordinates, this corresponds to moving the origin; for mode coordinates decoded through mixed radix, $L(c_0)+L(x)$ is authoritative, and it cannot unconditionally be replaced by coordinate addition with carries.

## 7. Composition: Pointwise Evaluation and Normalization

The local index of a thread or small block usually needs to be converted first into a logical index of the original layout and then into a storage address. Composition chains these two steps, so that an existing layout can be reused under a new coordinate organization.

**Function composition** $A\circ B$ means applying $B$ first and then feeding its output to $A$. The ordinary case is:

$$
C(i)=A_{\mathrm{ext}}(B(i)).
$$

Here B's output is a logical index of A, not a physical address of A. B may enter A's extended domain; the general `composition` requires B's reachable values to be non-negative.

`composition_layout` additionally re-expresses the result as shape/stride. This process is called **normalization**: generating new layout metadata by fixed algebraic rules. It requires certain **divisibility**, i.e. one integer divides another exactly with zero remainder; when the requirement is not met it returns `NotDivisible`.

The CuTe rules the current implementation aligns with have a special **weak divisibility** case: some stride is smaller than the current mode length but does not divide it, and the traversal must cross that mode boundary. Normalization may then restart in the next mode and does not equal pointwise composition. A concrete example is:

$$
(4,6):(6,1)\circ(8:3)\quad\rightsquigarrow\quad(2,4):(18,1).
$$

The arrow on the right denotes the current normalization rule, not a general functional identity. The address of index 2 in the result is 1; direct computation gives $B(2)=6$ and $A(6)=13$.

When `composition` encounters such a normalizable affine input, it caches the normalized result and `is_normalized()` returns `true`. If the shape/stride rules report `NotDivisible`, the general `composition` can still keep pointwise evaluation; a non-affine outer also takes the pointwise path. Therefore not every composition can be interpreted as an exception-free $A(B(i))$, and not every `composition` object can be treated as a successfully normalized affine layout.

The typed `Layout::composition` and the fully static `TypedComposition` constant share the algebra above and preserve the nested tree of the RHS; the output structure is governed only by static facts, and modes whose dynamic value is 1 are not eliminated at the type level. The runtime `composition_layout` may simplify the structure using actual values; the result is represented by `GroupedLayout`. The typed Rust methods additionally check the dynamic divisibility assertions that upstream omits, and report divisibility, division-by-zero, and overflow errors through `Result`.

## 8. Complement: Describing Gaps and Repetition Strides

After deciding how to traverse the inside of a small block (tile), we also need to know how to step from block to block in order to cover a larger region. The **complement** describes these gaps and repetition displacements, and is the basis for the tiling and repeated tiling in the next section.

The "complement" in `complement` is not the set-theoretic **complement set**. A complement set is the elements of a target set that do not belong to the original set; a complement layout instead generates a set of displacements that can be added to the original layout's addresses.

Take `B=4:2` as an example; B produces `{0,2,4,6}`. To tile out to 24 addresses, the complement is:

$$
R=(2,3):(1,8),\qquad B(b)+R(r).
$$

The first axis of the complement switches between even and odd addresses, and the second axis translates by 8 each step; added to B, they cover exactly `[0,24)`. These positions generated by integer strides can be called **lattice points**; no continuous space needs to be introduced here.

The current algorithm first rejects empty domains, then applies `filter` to remove broadcasting. If only `1:0` remains (for example, the original layout is `1:0`, a pure broadcast `4:0`, or all axis lengths are 1), a special case applies: with a target extent cotarget it directly returns `prod(cotarget):1`, where `prod` denotes the product of the target lengths; without a cotarget it returns `1:1`. For example, `complement(4:0,6)=6:1`, and the gap check below is never reached.

In the remaining cases, sort by stride, let the current spacing be $q=1$, and for each leaf `(s,d)`:

1. Compute the gap length $g=\lfloor d/q\rfloor$. If $g=0$, return `NonInjective`.
2. If $g>1$, add `g:q` to the complement.
3. Update $q=ds$.

The **cotarget** is the target extent to complete up to. For an integer cotarget $C$, additionally append a mode of length $\lceil C/q\rceil$ and stride $q$, then coalesce. $\lceil x\rceil$ is the smallest integer not less than $x$, i.e. rounding up (ceiling). For non-negative integers, $\operatorname{ceil\_div}(a,b)$ is $\lceil a/b\rceil$; it makes an incomplete group still occupy a group.

A tuple cotarget is not first reduced to its total product; it is updated leaf by leaf. Initially `rest=q` and `stride=q`; each target length $a$ gives:

$$
e=\lceil a/\mathrm{rest}\rceil,\qquad
\mathrm{rest}\leftarrow\lceil\mathrm{rest}/a\rceil,\qquad
\mathrm{stride}\leftarrow\mathrm{stride}\cdot e.
$$

When $e\ne1$, append `e:old_stride`; when $a=0$, the second division is not performed and the old `rest` is kept. Hence `complement_shape(3:1,[4,6])=12:3`, whereas `complement(3:1,24)=8:3`.

Expanding the tuple case, with initial `rest=3` and `stride=3`:

| Target axis | Current length and rest | Appended mode | Updated rest / stride |
|---|---|---|---|
| Axis 0 | $a=4,\ \mathrm{rest}=3$, so $e=\lceil4/3\rceil=2$ | `2:3` | `1 / 6` |
| Axis 1 | $a=6,\ \mathrm{rest}=1$, so $e=\lceil6/1\rceil=6$ | `6:6` | `1 / 36` |

The two modes `(2,6):(3,6)` coalesce to `12:3`. Intuitively, covering length 4 on the first axis needs two blocks of length 3, so the padded length becomes 6; the second axis then repeats 6 times. Multiplying the target into 24 first would need only eight blocks of length 3, losing the per-axis padding boundaries.

Without a cotarget, `complement_extended` keeps the gaps and, when needed, appends `1:q` to record the stride for continued repetition. For `4:2`, the result is `(2,1):(1,8)`. The 8 here comes from $d\cdot s$, **not** from the original layout's `cosize=7`; when reasoning about product operations this stride must not be replaced by the address span.

These rules do not guarantee that every input yields a non-overlapping, hole-free tiling. For example, `(2,2):(1,3)` with cotarget 12 gives `2:6`, and the combined addresses still miss 2, 5, 8, and 11. Filtering out broadcasting also does not make the original layout injective. Empty inputs, and layouts that still have negative strides after filtering, are not supported; judge by the algorithm's conditions and return values here, rather than treating `complement` as a general set complement or a proof of bijectivity.

## 9. Division and Layout Products

### 9.1 Regrouping Coordinates with Tile and Rest

A **tile** is a small block processed at once, and **rest** is the inter-block coordinate needed to cover the remaining positions. The address rule of `TileRest(B,R)` is:

$$
T(b,r)=B(b)+R(r),\qquad
\operatorname{size}(T)=\operatorname{size}(B)\operatorname{size}(R).
$$

The tile varies fastest in its logical linear traversal. The logical division by a single tiler is defined as:

$$
R=\operatorname{complement\_shape}(B,\operatorname{shape}(\operatorname{coalesce}(L))),
\qquad
\operatorname{logical\_divide}(L,B)=\operatorname{composition}(L,T(B,R)).
$$

A **tiler** is a layout that prescribes the indexing within a block; an integer length $t$ can be represented as `t:1`. The composition follows the actual rules of Section 7. When a non-affine input has no shape/stride canonical form, the general divide uses `size(L)` as the integer cotarget instead.

For example, dividing `6:1` by `4:1` gives `(4,2):(1,4)`, and the size grows from 6 to 8. The last two slots are the **padding** part; their addresses are defined in the extended domain but do not belong to the original six elements. The operation itself does not automatically generate validity checks for these positions.

### 9.2 Per-Mode Division and Grouping

`*_divide_modes` assigns a tiler to each top-level mode. For ordinary length tiling, each axis can be written as:

$$
c_j=t_j+b_jr_j,\qquad 0\le t_j<b_j.
$$

Here $b_j$ is the block length, $t_j$ the intra-block coordinate, and $r_j$ the block index. Taking the two-dimensional shape `(8,6)` and block shape `(4,3)` as an example, the rest shape is `(2,2)`. The APIs differ in how coordinates are grouped:

| Form | Conceptual coordinate structure | Top-level extent in this example |
|---|---|---|
| logical | `((t0,r0),(t1,r1))` | `(8,6)` |
| zipped | `((t0,t1),(r0,r1))` | `(12,4)` |
| tiled | `((t0,t1),r0,r1)` | `(12,2,2)` |
| flat | `(t0,t1,r0,r1)` | `(4,3,2,2)` |

`zipped` and `tiled` both gather the intra-block coordinates `(t0,t1)` into mode 0, so that this mode can be used directly as "one tile"; the subsequent modes describe "which tile". The difference is whether the rest coordinates are grouped together or exposed per axis.

**Grouping** changes top-level mode boundaries; swapping coordinate positions is called **reordering**. The leaf order of logical differs from the other three, so the same linear index does not necessarily visit the same original coordinate. `GroupedLayout` keeps only the top-level grouping; the deeper boundaries in the table are conceptual.

`*_divide_modes` allows non-divisible cases with padding, and trailing modes without a tiler go into rest. For the ordinary `logical_divide` and `zipped_divide` with a single layout tiler, both have just the two groups `(tile,rest)`, and there is no per-axis group swapping.

### 9.3 Products Express Repeated Tiling

A layout **product** is not element-wise multiplication of addresses. It combines two logical domains into a larger domain:

$$
\operatorname{logical\_product}(B,V)
=T\bigl(B,\operatorname{composition}(\operatorname{complement\_extended}(B),V)\bigr).
$$

For example, with block `3:1` and tiler `2:1`, the complement is `1:3`. Extending it gives two repetitions; the final address is $b+3v$ and the size is 6.

`zipped_product`, `tiled_product`, and `flat_product` change the top-level grouping of tile/rest. `blocked_product` and `raked_product` instead first pad both sides to the same rank and then pair mode by mode: the former arranges `(block_m,rest_m)`, the latter `(rest_m,block_m)`. Missing modes are padded with `1:0`, and that leaf is kept. In the example above, the leaves of the two are `(3,2):(1,3)` and `(2,3):(3,1)` respectively: the same address set, different enumeration order.

## 10. Inversion and Decoding Thread Indices

The **identity map** $\mathrm{id}(x)=x$ leaves its input unchanged. A **left inverse** $G$ satisfies $G(L(i))=i$, i.e. it recovers the input from an address; a **right inverse** $R$ satisfies $L(R(a))=a$, i.e. it picks an input for a target address. All sources of an address are called its **preimage**.

There are currently two different inversion contracts:

| API | Mathematical meaning and restrictions |
|---|---|
| `right_inverse` | Uses the stride structure to select an address prefix starting at 0 that can be contiguously recovered; does not require covering all original addresses |
| `left_inverse` | Uses the stride structure to build a reverse layout that also has values at address holes; broadcast axes are skipped, so broadcast inputs cannot promise a true left-inverse identity |
| `dense_left_inverse` | The original layout must be a bijection onto `[0,size)`; searches preimages one by one |
| `dense_right_inverse` | The original layout must be surjective onto `[0,max_address]`; allows broadcasting and chooses the preimage with the smallest logical index |

Structural inversion first coalesces leaves and records each leaf's prefix product in the logical linearization, then sorts by address stride. `right_inverse` starts from a covered length of 1 and only accepts leaves whose stride exactly equals the covered length. Hence:

$$
\operatorname{right\_inverse}(4:2)=1:0,\qquad
\operatorname{left\_inverse}(4:2)=(2,4):(0,1).
$$

The right inverse recovers only address 0; the left inverse maps original addresses 0, 2, 4, 6 back to indices 0, 1, 2, 3, and is also defined at addresses 1, 3, 5, 7, but those values carry no requirement to "recover the original element".

The structural right inverse yields the contiguous prefix the algorithm recognizes; it must not be read as the longest prefix found by searching an arbitrary overlapping layout. For example, `(2,2):(1,1)` can reach addresses 0, 1, 2, but the structural right inverse covers only 0 and 1. The structural left inverse returns an error for empty domains, for inputs that still have negative strides after simplification, and for inputs that violate its divisibility rules; the enumerating inverses can handle non-affine layouts that satisfy the density conditions.

`get_flat_coord` is another stride-based decoding operation. For each non-unit leaf it computes:

$$
c_j=\left\lfloor\mathrm{index}/d_j\right\rfloor\bmod s_j,
$$

and then encodes the coordinates within the same top-level mode in left-major order. For a compact arrangement it yields the coordinate corresponding to an address; it does not provide a general inverse for arbitrary layouts with holes or overlaps, and it does not actively verify that the input address is reachable. Non-unit leaves must have positive strides.

Next, layouts are handed to concrete threads, so we need to answer in reverse "which coordinates does this thread index correspond to". `local_partition_layout` in Section 11 and `TiledMmaLayout` in Section 13 use `get_flat_coord` to decode thread coordinates; Section 12.3 uses `right_inverse` to recover tile indices from thread-value indices. Inversion thereby connects the layout algebra to thread work distribution.

## 11. From Tiling to Local Views

### 11.1 `inner_partition` and `outer_partition`

Both first perform per-mode division and then slice: `inner_partition` fixes the rest coordinate and keeps the elements within a block; `outer_partition` fixes the tile coordinate and keeps the elements across blocks. `Slice::All` means the mode is not fixed.

For example, with the original layout `(8,6):(1,8)`:

- Taking block coordinate `(1,1)` with block shape `(4,3)`, the origin is `(4,3)`, the address offset is $4+8\cdot3=28$, and the sublayout is `(4,3):(1,8)`.
- Fixing intra-block coordinate `(1,1)` with thread shape `(2,3)`, the sublayout is `(4,2):(2,24)` with offset 9. The thread takes values every 2 elements along the first axis and every 3 elements along the second axis.

`local_tile_layout` is the first operation; `local_partition_layout` first decodes the thread index with the thread layout via `get_flat_coord`, then performs the second operation. In the thread layout `(2,3):(3,1)`, thread index 4 decodes to `(1,1)`, which gives exactly the example above.

The **projection** in `*_proj` selects a subset of modes, for example keeping `(M,K)` out of `(M,N,K)`. A boolean array currently represents keep versus drop. The selected coordinate components are kept directly; no data summation is performed.

### 11.2 Predicated Tail Tiles

A **predicate** is a function whose result is true or false. `local_tile` additionally provides a view with a built-in boundary predicate. Let the block coordinate be $b$, the block length $t$, and the original shape $s$; then:

$$
o_j=b_jt_j,\qquad g_j=o_j+c_j,\qquad
P(c)=\bigwedge_j(0\le c_j<t_j\ \land\ g_j<s_j).
$$

$\land$ and $\bigwedge$ denote "and"; the latter means all conditions must hold. $g$ is the coordinate in the original layout. This view still has the logical size of a full tile, but defines addresses only for slots where $P$ is true; `valid` / `valid_linear` can be checked before access.

For example, with length 6 and block length 4, the second block contains local slots 0, 1, 2, 3; only the first two are valid. `local_tile_layout` allows computing extended addresses for the last two slots, while `local_tile` refuses to actually index them.

### 11.3 One-Dimensional Round-Robin Distribution

`local_partition(L,T,t)` uses **round-robin assignment**: value $v$ of thread $t$ comes from original logical index $t+vT$. $T$ is the number of threads and $0\le t<T$. The actual size for each thread is:

$$
n_t=\begin{cases}
0,&t\ge n,\\
1+\lfloor(n-1-t)/T\rfloor,&t<n,
\end{cases}\qquad n=\operatorname{size}(L).
$$

It directly drops the invalid trailing slots. For example, distributing 10 elements to 4 threads, thread 1 gets indices 1, 5, 9. This and the coordinate-based distribution of the multi-dimensional `local_partition_layout` are two separate interfaces.

## 12. Thread-Value Layouts and Copy

A **thread-value layout**, TV (thread/value) for short, describes the map from `(thread index, value index within the thread)` to tile coordinates. "Value index" here means which element the thread is processing, not the element's numeric value.

### 12.1 Explicit Two-Dimensional Ownership

`ThreadValueLayout` specifies the thread shape $T=(T_0,T_1)$, the value shape $V=(V_0,V_1)$, and the indexing order of each. After decoding the thread coordinate $t$ and value coordinate $v$:

$$
c_j=t_j+T_jv_j,\qquad S_j=T_jV_j.
$$

`FirstMajor` decodes index $i$ as `(i % s0, i / s0)`; `LastMajor` decodes it as `(i / s1, i % s1)`. From $c_j$ one can uniquely recover $t_j=c_j\bmod T_j$ and $v_j=\lfloor c_j/T_j\rfloor$, so every tile element has exactly one owner.

For example, with `thread_shape=[2,2]` and `value_shape=[2,1]`, both using `FirstMajor`, the tile is `[4,2]`. The two value coordinates of thread 1 are `[1,0]` and `[3,0]`. Only this compact blocked form is currently implemented; it does not describe arbitrary hardware thread arrangements.

### 12.2 Copy Atoms and Cyclic Arrangement

An **atom** is the basic granularity of one operation. Let a Copy atom contain $a$ values, the number of threads be $T$, and the number of tile elements be $E$. `TiledCopy::new` reserves for each thread

$$
V_{\mathrm{thread}}=a\left\lceil E/(Ta)\right\rceil
$$

value slots, and uses:

$$
i(t,v)=\left(\lfloor v/a\rfloor T+t\right)a+(v\bmod a).
$$

Logical indices within an atom are contiguous, and atoms are assigned round-robin; when $a=1$ this degenerates to $t+vT$. Only slots with $i<E$ are valid. The coordinate in a two-dimensional tile `[R,C]` is `(i % R, i / R)`.

For example, with $E=10,T=3,a=2$, each thread has 4 value slots. Thread 0 handles `0,1,6,7`, thread 1 handles `2,3,8,9`, and thread 2's slots are `4,5,10,11`, of which 10 and 11 are inactive.

`CopyAtom::vector` records the number of values and the byte **alignment** requirement, i.e. the starting byte address must be divisible by the specified alignment. Contiguous logical indices do not guarantee contiguous physical addresses; the backend still has to check the actual layout.

### 12.3 Deriving TV from the Layout Algebra

`make_layout_tv(thr,val)` uses another, more general construction:

$$
L_{MN}=\operatorname{raked\_product}(thr,val),\qquad
L_{TV}=\operatorname{composition\_layout}
\left(\operatorname{right\_inverse}(L_{MN}),
\operatorname{LayoutLeft}(\operatorname{size}(thr),\operatorname{size}(val))\right).
$$

Let $T=\operatorname{size}(thr)$ and $V=\operatorname{size}(val)$. The right inverse accepts the one-dimensional index $t+Tv$; composing with `LayoutLeft(T,V)` first encodes `(thread index t, value index v)` into this integer and then passes it to the right inverse. The result therefore exposes two top-level input axes, and callers do not need to compute $t+Tv$ by hand.

For thread and value layouts that satisfy the required coverage conditions, $L_{MN}$ maps tile indices to $t+Tv$, and $L_{TV}$ maps them back to tile indices. `raked` refers to the interleaved arrangement that places rest before block mode by mode. It is not directly interchangeable with the blocked formula of `ThreadValueLayout`; nor can one infer from the API name alone that TV always covers the whole tile for arbitrary irregular inputs.

## 13. Two Levels of Mathematical Description for MMA

A **matrix** is a table of numbers with two coordinate axes. One output element of a matrix multiplication requires summing several products; the operation that combines multiple values into one along some axis is called a **reduction**. MMA (matrix multiply-accumulate) denotes matrix multiplication with accumulation; $M,N$ are the two output axes, and $K$ is the reduction axis over which products are summed. A **fragment** is the part of the elements held by a thread in one operation; a **lane** is one thread position within a cooperating thread group.

### 13.1 `TiledMma`: Logical Thread Groups

`MmaAtom` specifies the atom shape `(aM,aN,aK)` and the number of threads. `TiledMma` divides the output plane into `(GM,GN)` thread groups, each with output region:

$$
(M/GM,N/GN),\qquad
\operatorname{origin}(g)=
\left((g\bmod GM)M/GM,\lfloor g/GM\rfloor N/GN\right).
$$

Each group's region must hold whole atoms: $M$ must be divisible by $GM\cdot aM$, $N$ by $GN\cdot aN$, and $K$ by $aK$. When the grid is chosen automatically, the implementation enumerates valid factorizations and minimizes the absolute difference between the height and width of each group.

At this level, `partition_a/b/c` is a scalar Copy distribution covering the entire corresponding tile, used for staging or output; it does not derive the lane mapping of hardware fragments, and B uses `(K,N)`.

### 13.2 `TiledMmaLayout`: Explicit Atom TV and Thread Grid

`MmaAtomLayout` requires the caller to provide the atom shape, the `ThrID` thread-index layout, and the TV layouts of A/B/C. The outputs of these TV layouts are left-major linear indices in the atom's `(M,K)`, `(N,K)`, and `(M,N)` domains respectively.

The thread grid is denoted `AtomLayoutMNK`, and the construction is:

$$
L_{VMNK}=\operatorname{tiled\_product}(\mathrm{ThrID},\mathrm{AtomLayoutMNK}).
$$

`V` here denotes the atom-internal thread mode `ThrV`; do not confuse it with the per-thread value mode `FrgV`. If the grid has fewer than three axes, it is padded with `1:0`; the thread index is decoded as `(v,m,n,k)` using the strides of $L_{VMNK}$.

Mathematically, a **permutation** is a bijection from a set to itself, used to rearrange elements. The parameter named `PermutationMNK` here actually accepts an optional layout per axis; it can also change the tile extent, so one cannot assume the constructor verifies a bijection merely because it is named permutation. The tile length of axis $j$ is:

$$
\mathrm{tile}_j=
\begin{cases}
\operatorname{size}(P_j),&\text{a layout }P_j\text{ is provided for this axis},\\
\mathrm{atom}_j\cdot\operatorname{extent}(L_{VMNK},j+1),&\text{this axis is None}.
\end{cases}
$$

`thrfrg_*` performs, in order, permutation division, atom division, composition with the atom TV, and thread-grid division, producing a "thread-coordinate contribution" and an "intra-thread fragment-coordinate contribution". After fixing the thread coordinate, `partition_*` returns the fragment layout and address offset.

A has no N axis, so threads along N may share A; B has no M axis, so threads along M may share B. In the TV description this sharing appears as zero strides. `layout_*_tv` gives the thread-value map of the whole tile. When the thread grid splits along K, `layout_c_tv` may also produce extended indices beyond the C tile; the current layout description does not automatically reduce the accumulated results of the K slices into one output.

Both levels of interface are only work-distribution descriptions; they do not execute multiplication, synchronization, or hardware instructions. `universal_fma()` uses a single-thread `1×1×1` atom and can be used to observe these algebraic transformations.

## 14. Swizzle: Invertible Transformations of Address Bits

The Copy/MMA partitions determine which elements each thread processes; next we must consider how these elements are staged. Shared memory is storage accessed jointly by cooperating threads; under some access patterns, affine strides alone cause multiple threads to access the same storage group (bank) simultaneously. Swizzle appends a bitwise rearrangement after the layout address to adjust this distribution; `StagedTile` below uses it for staged buffering, and Section 15 connects the addresses to actual storage through Tensors.

A **bit** is a binary digit 0 or 1. A **bit field** is a contiguous run of bits, and a **mask** selects that run with 1s at the specified positions. Bitwise AND `&` keeps only bits that are 1 on both sides; bitwise exclusive OR `^` (XOR) yields 1 where the two bits differ; left shift `<<` and right shift `>>` move bits by the specified number of positions.

`Swizzle<B,M,S>` applies the following transformation to non-negative addresses. Let $b=2^B-1$; the source bit field starts at $M+\max(S,0)$:

$$
y=x\ \&\ \left(b\ll(M+\max(S,0))\right),
\qquad
W(x)=\begin{cases}
x\mathbin{\mathrm{XOR}}(y\gg S),&S\ge0,\\
x\mathbin{\mathrm{XOR}}(y\ll(-S)),&S<0.
\end{cases}
$$

B is the bit-field width, M is the starting bit of the lower bit field, the absolute value of S is the distance between the two bit fields, and its sign determines the shift direction. The source and target bit fields must not overlap, i.e. $|S|\ge B$, and $M+B+|S|\le63$ to avoid reaching the sign bit of signed `i64`. When B is 0 it is the identity transformation.

Because the source bit field is not modified, XORing the same content twice cancels out:

$$
W(W(x))=x.
$$

A function that returns to the original value when applied twice is called an **involution**. A valid swizzle is therefore a bijection on addresses: it neither creates nor removes address overlaps of the original layout, but it may change the minimum and maximum reachable addresses.

For example, `Swizzle<3,3,3>` equals `x ^ (((x >> 6) & 7) << 3)`. Address 64 becomes 72, and applying it again returns 64. The composed layout is $W(L(c))$, and the logical shape is unchanged.

`StagedTile` replicates the same arrangement into multiple **stages**, i.e. stage buffers. Let each stage have R rows with row pitch `pitch` P, so each stage has $E=RP$ elements:

$$
\mathrm{offset}(s,r,c)=sE+W(rP+c).
$$

Construction requires $E$ to be an integer multiple of $2^{M+B+|S|}$. This number is the block length of the bit-field pattern; bits above the relevant bit fields are unchanged, so the transformation within a complete block does not spill into an adjacent stage. `pitch` is measured in elements; a byte length must not be substituted directly.

In the tests, a **bank** is a storage group accessed in parallel, and a **bank conflict** means that accesses in the same batch hit the same group. `sm89_swizzle.rs` uses a specific model of two-byte elements and 16-byte groups to check whether write and read patterns are spread across different groups. This is a verification within that test model, not a performance guarantee for arbitrary layouts on arbitrary devices.

## 15. Tensors, Copying, and Matrix Computation

### 15.1 A Tensor Connects the Address Map to Values

A **tensor** in this implementation is a shaped collection of logical elements, represented by a storage engine E, a layout L, and a starting offset o:

$$
T(c)=E[o+L(c)].
$$

An **engine** can either access actual storage or generate values from indices. If the storage length is N, a non-empty Tensor requires:

$$
0\le o+a_{\min},\qquad o+a_{\max}<N.
$$

For example, `4:-1` with offset 3 accesses 3, 2, 1, 0. Read-only Tensors allow broadcasting; writable Tensors additionally require an injective layout. An empty Tensor has no reachable addresses and allows the starting offset to equal N exactly.

`identity_layout` leaves linear indices unchanged; the values of `identity_tensor` are the coordinates themselves, e.g. for `[2,3]` the value at logical index 4 is `[0,2]`. It decodes coordinates through a virtual engine and does not store an array of coordinates. "Identity" here denotes the coordinate identity relation; the common **identity matrix** is a numeric matrix with equal numbers of rows and columns, 1 on the diagonal and 0 elsewhere, which means something different.

### 15.2 Copying Does Not Require Identical Physical Arrangement

`copy` specifies $\mathrm{dst}[i]\leftarrow\mathrm{src}[i]$, where i is the logical index on each side; the two sides need only the same logical size and the same element type. `copy_if` performs the read/write only when the predicate is true. `fill` fills all reachable elements and leaves address holes alone. `clear` writes the element type's default value; it does not prove through the type system that this is numeric zero.

### 15.3 Two Matrix Multiplication Contracts

**Transposition** swaps the row and column coordinates of a matrix; the `(k,n)` element of $B^T$ is the `(n,k)` element of B. The current `gemm` is defined as:

$$
C_{mn}\leftarrow C_{mn}+\sum_{k=0}^{K-1}A_{mk}B_{nk},
\qquad A:(M,K),\ B:(N,K),\ C:(M,N).
$$

That is, $C\leftarrow C+AB^T$. The **accumulator** here is the value that holds partial results during summation. `gemm` accumulates starting from the original C; when K is zero the values of C are unchanged, although the current loop still reads and writes back C.

`matmul` instead overwrites the output:

$$
C_{mn}\leftarrow\sum_{k=0}^{K-1}A_{mk}B_{kn},
\qquad B:(K,N).
$$

It starts from the element's default value and requires that value to be the **additive identity**, i.e. a value satisfying $0+x=x+0=x$. When K is zero it writes that identity. When M or N is zero, neither function has output elements to iterate over.

These are scalar reference algorithms executed in a fixed order. Numeric elements use their own type's addition and multiplication, and the checked integer arithmetic of layouts does not automatically extend to element arithmetic. In particular, finite-precision **floating-point numbers** represent only a subset of the real numbers, and every operation may round; changing the summation order does not guarantee bit-identical results.

## 16. Symbolic Expressions and Valid Domains

A **symbolic expression** first records the relationships between operations and then substitutes parameters to evaluate. A **constant** is known when the expression is constructed; a **parameter** is supplied by name at evaluation time; **evaluation** replaces these symbols with numbers and performs the operations. For example, any two-dimensional flat layout `(s0,s1):(d0,d1)` can be exported as:

$$
E(i)=(i\bmod s_0)d_0+
\left(\lfloor i/s_0\rfloor\bmod s_1\right)d_1.
$$

For right-compact layouts $d_0=s_1,d_1=1$; other layouts use their own strides. The order of division and remainder always follows the logical-index convention in which the leftmost axis varies fastest, independent of the physical storage order.

`Expr` represents the address formula, `Pred` represents the valid domain, and `IndexExpr` exports the two together with parameter bindings. The **consistency condition** they must satisfy is: under a valid parameter binding, whenever the predicate is true, evaluating the expression equals evaluating the CPU layout. There is no promise that addresses can still be computed safely at invalid coordinates.

**Constant folding** computes in advance the subexpressions that consist entirely of constants and can be evaluated successfully. **Short-circuiting** skips evaluating later branches once the boolean result is determined; for example, "non-empty and index valid" checks non-emptiness first, avoiding division by zero on an empty shape. `Select(P,a,b)` likewise evaluates only the selected branch.

Signed division in expressions truncates toward zero, and the remainder has the same sign as the dividend or is zero, e.g. `-7 / 3 = -2` and `-7 % 3 = -1`; this differs from the mathematical floor of negative numbers. The earlier index formulas use non-negative indices, so this difference does not arise there.

Metadata and expression arithmetic are also constrained by the representable integer range. **Overflow** means the mathematical result exceeds the range of `i64` or of the target integer type; the address layer and the expression evaluator report the corresponding errors. Rebinding parameters must still satisfy the original construction conditions. In particular, the structure of a normalized composition depends on the dynamic values at that time; after the inputs change it must be reconstructed, not merely have its parameters replaced.

## 17. Glossary and Implementation Map

| Concept | Section where defined | Main implementation |
|---|---|---|
| Map, domain, codomain, image, inner product, affine | 1 | `LayoutMap`, `Layout` |
| Tuple, mode, leaf, rank, depth, cardinality, empty product | 2 | `Node`, `IntTuple`, `Shape` |
| Mixed radix, linearization, prefix product, compact, row/column-major | 3 | `index`, `LayoutLeft/Right` |
| Address interval, span, hole, broadcasting, injective/surjective/bijective | 4 | `address_range`, `cosize`, `is_injective` |
| Extended domain | 5 | `index_extended`, `index_extended_of_coord` |
| Flatten, coalesce, slice, translation | 6 | `flatten`, `coalesce`, `slice`, `domain_offset` |
| Composition, normalization, divisibility, weak divisibility | 7 | `composition`, `composition_layout` |
| Complement set and complement layout, lattice point, cotarget, ceiling | 8 | `complement*` |
| tile/rest, tiler, grouping, padding, layout product | 9 | `TileRest`, `*_divide*`, `*_product` |
| Identity map, preimage, left inverse, right inverse | 10 | `*_inverse`, `get_flat_coord` |
| Projection, predicate, round-robin assignment | 11 | `local_*`, `inner/outer_partition` |
| TV, atom, alignment, blocked/raked | 12 | `ThreadValueLayout`, `TiledCopy`, `make_layout_tv` |
| Matrix, reduction, MMA, fragment, lane, permutation | 13 | `MmaAtom*`, `TiledMma*` |
| Bit field, mask, XOR, involution, stage, bank | 14 | `Swizzle`, `StagedTile` |
| Tensor, transposition, accumulator, additive identity, floating-point rounding | 15 | `Tensor`, `copy`, `gemm`, `matmul` |
| Symbolic expression, constant folding, short-circuiting, overflow | 16 | `Expr`, `Pred`, `IndexExpr` |

The source entry points of the mathematical model are [layout.rs](../src/layout.rs), [algebra.rs](../src/algebra.rs), [partition.rs](../src/partition.rs), [swizzle.rs](../src/swizzle.rs), [tensor.rs](../src/tensor.rs), and [expr.rs](../src/expr.rs). The next document, [engineering implementation](engineering-implementation.md), explains the type constraints, execution paths, and runnable examples module by module.
