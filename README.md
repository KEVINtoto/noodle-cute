# noodle-cute

`noodle-cute` provides a device-independent foundation for CuTe-style layouts and tensor algorithms in Rust. It uses static integers and nested heterogeneous tuples to describe structures such as shapes and strides, keeping compile-time values in their types while supporting runtime parameters.

The core uses `no_std`. The `int` and `tuple` modules covered here require neither heap allocation nor a device backend:

- `int` provides compile-time and runtime integer representations with a shared interface and arithmetic operations.
- `tuple` organizes integers into nested structures, with access, concatenation, flattening, and product operations.

CuTe design and implementation references:

- [NVlabs/CuTe](https://github.com/NVlabs/CuTe): a Python reference implementation of CuTe layouts and layout algebra.
- [CuTe in NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass/tree/main/include/cute): the C++ headers for CuTe integer tuples, layouts, tensors, and related operations.

## Toolchain Dependencies

The repository pins its Rust toolchain in [rust-toolchain.toml](rust-toolchain.toml) and declares its edition and dependencies in [Cargo.toml](Cargo.toml).

| Setting | Value |
| --- | --- |
| Rust toolchain | `nightly-2026-10-11` |
| Rust edition | `2024` |
| Toolchain components | `rust-src`, `rustfmt`, `clippy`, `rust-analyzer` |
| Third-party crate dependencies | None currently |

Type-level integer arithmetic relies on nightly generic constant features. The library enables `gca_min_const_items`, `gca_const_items`, `gca_macroless_args`, and `generic_const_items` internally. The usage examples below do not require feature attributes in the calling crate.

Install the toolchain, then run tests and generate API documentation from the repository root. When invoked through rustup, Cargo automatically uses the repository's pinned toolchain.

```sh
rustup toolchain install nightly-2026-10-11 --component rust-src --component rustfmt --component clippy --component rust-analyzer
cargo test
cargo doc --no-deps
```

## Public Interfaces

This README currently covers only the [`int`](src/int.rs) and [`tuple`](src/tuple.rs) modules.

### `int`: Static and Dynamic Integers

Integer values use `isize`, so their range depends on the target platform's pointer width.

| Interface | Purpose |
| --- | --- |
| `Int<const N: isize>` | A zero-sized integer with its value encoded in its type; read it through `Int::<N>::VALUE` or `.value()` |
| `DynInt(pub isize)` | A runtime integer; read it through `.0` or `.value()` |
| `Integer` | A shared interface for `Int<N>`, `DynInt`, and native `isize`, providing `STATIC` and `value()`; downstream crates cannot implement this trait |
| `NON_NEGATIVE::<N>` | Tests whether the compile-time integer `N` is nonnegative |

`Integer::STATIC` indicates whether the type determines the value. It is `Some(N)` for `Int<N>` and `None` for `DynInt` and `isize`, even when they happen to hold the same value.

`Int` and `DynInt` support `+`, `-`, `*`, `/`, `%`, and unary negation. Arithmetic between two `Int` values produces another `Int`. Operations with a runtime operand and either wrapper produce `DynInt`. These operators follow Rust's integer arithmetic rules and do not return `Result`.

```rust
use noodle_cute::int::{DynInt, Int, Integer, NON_NEGATIVE};

fn main() {
    let sum: Int<5> = Int::<2> + Int::<3>;
    let negated: Int<-5> = -sum;
    assert_eq!(negated.value(), -5);

    let mixed: DynInt = Int::<2> * DynInt(3);
    let native: DynInt = 4isize + Int::<2>;
    assert_eq!(mixed.value(), 6);
    assert_eq!(native.value(), 6);

    assert_eq!(<Int<7> as Integer>::STATIC, Some(7));
    assert_eq!(<DynInt as Integer>::STATIC, None);
    assert_eq!(<isize as Integer>::STATIC, None);
    assert!(NON_NEGATIVE::<0>);

    let runtime: DynInt = 11isize.into();
    assert_eq!(runtime, DynInt(11));
    assert_eq!(isize::from(Int::<7>), 7);
}
```

### `tuple`: Nested Heterogeneous Integer Structures

`IntTuple<H, T>` holds a head `H` and a tail `T`, with the tuple chain ending in `Nil`. A head can be an integer or another tuple, so each position can hold a different integer type or nested structure.

Use `int_tuple!` to construct these structures. Commas separate top-level modes, parentheses create nested modes, and an empty invocation, `int_tuple![]`, returns `Nil`. For example, `(2, (3, 4))` has two top-level modes, three integer leaves, and a nesting depth of two.

| Interface | Purpose |
| --- | --- |
| `Item` | A shared interface for integer leaves and tuples, providing structural metadata, leaf queries, and a checked product |
| `Tuple` | A tuple-chain interface providing the top-level length `LEN` and the greatest head depth `MAX_DEPTH` |
| `Get<INDEX>` / `.get::<INDEX>()` | Borrows a top-level mode by a compile-time index, preserving its concrete type and nesting; invalid indices fail to compile |
| `Append<R>` / `.append(rhs)` | Replaces the tuple chain's terminating `Nil` with `rhs`; passing another tuple concatenates the chains |
| `Flatten` / `.flatten()` | Flattens the structure in head-first, depth-first order, preserving leaf order and types |
| `Product` / `.product_typed()` | Computes the product, keeping a static result type when every leaf is static |

`Item` provides the structural metadata `RANK`, `DEPTH`, `LEAVES`, and the associated type `Profile`:

- `RANK` is the number of top-level modes, also returned by `rank()`.
- `DEPTH` is the nesting depth, also returned by `depth()`. Extending a flat tuple does not increase its depth.
- `LEAVES` is the total number of integer leaves across all levels.
- `Profile` describes the nested structure using `NilProfile`, `LeafProfile`, and `TupleProfile`, independently of leaf values or integer types.

A bare integer has rank `1`, depth `0`, and one leaf. `Nil` has rank `0`, depth `1`, and no leaves. A nonempty flat tuple has depth `1`. `Tuple::MAX_DEPTH` excludes the current tuple level and the terminating `Nil`; it is `0` for an empty chain.

Queries use zero-based indices. `leaf(index)` reads a leaf in flattened order, `leaf_static(index)` queries its static value, and `mode_span(axis)` returns a top-level mode's `(start, leaf_count)` in the flattened sequence. Out-of-range indices return `Error::AxisOutOfBounds`.

```rust
use noodle_cute::int::{DynInt, Int};
use noodle_cute::tuple::Item;
use noodle_cute::{IntTuple, Nil, int_tuple};

fn main() {
    let nested = int_tuple![Int::<2>, (Int::<3>, DynInt(4))];

    assert_eq!(nested.rank(), 2);
    assert_eq!(nested.depth(), 2);
    assert_eq!(nested.leaf(2), Ok(4));
    assert_eq!(nested.leaf_static(0), Ok(Some(2)));
    assert_eq!(nested.leaf_static(2), Ok(None));
    assert_eq!(nested.mode_span(1), Ok((1, 2)));

    let second: &IntTuple<Int<3>, IntTuple<DynInt, Nil>> = nested.get::<1>();
    assert_eq!(second.get::<0>().value(), 3);
    assert_eq!(second.get::<1>().value(), 4);
    assert_eq!(nested.product(), Ok(24));
}
```

The two product interfaces serve different purposes:

| Interface | Result and arithmetic behavior |
| --- | --- |
| `Item::product()` | Returns `Result<isize>`; checks for zero leaves before multiplying, then checks each multiplication for overflow and returns `Error::Overflow` on failure; the empty product is `Ok(1)` |
| `Product::product_typed()` | Returns an integer type: `Int<N>` for an entirely static tuple and `DynInt` for a tuple containing a runtime leaf; uses ordinary multiplication without scanning for zero first; `Nil` produces `Int<1>` |

The following example combines structural operations with a static product. Import the corresponding traits to call their methods.

```rust
use noodle_cute::int::Int;
use noodle_cute::int_tuple;
use noodle_cute::tuple::{Append, Flatten, Product};

fn main() {
    let nested = int_tuple![Int::<2>, (Int::<3>, Int::<4>)];
    let flat = nested.flatten();
    assert_eq!(flat, int_tuple![Int::<2>, Int::<3>, Int::<4>]);

    let joined = flat.append(int_tuple![Int::<5>]);
    assert_eq!(joined, int_tuple![Int::<2>, Int::<3>, Int::<4>, Int::<5>]);

    let total: Int<120> = joined.product_typed();
    assert_eq!(total.value(), 120);
}
```
