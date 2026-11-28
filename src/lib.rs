//! Exact hierarchical layout and tensor algebra in Rust (`no_std`).
//!
//! The crate is organized in layers from arithmetic primitives through
//! hierarchical layout algebra and tensor algorithms:
//!
//! - [`integer`]: Static ([`Int<N>`](integer::Int)) and dynamic ([`int`](integer::int))
//!   signed integers, `const` comparisons, three-valued static predicates, and arithmetic.
//! - [`mod@f2`]: Static and dynamic binary polynomials ([`F2`](f2::F2)) with XOR,
//!   carry-less multiplication, and polynomial division ([`f2!`]).
//! - [`mod@tuple`] and [`mod@slice`]: Hierarchical tuples ([`Tuple`](tuple::Tuple),
//!   [`Nil`](tuple::Nil)), structural congruence, mode indexing/slicing, restructuring,
//!   synchronized multi-tree leaf traversal, and product reductions
//!   ([`tuple!`], [`int_tuple!`], [`mode!`], [`walk_mode!`]). Python-style slice
//!   specifications ([`Slice`](slice::Slice), [`Bound`](slice::Bound)) support
//!   `const` dimension resolution ([`slice!`], [`slices!`]).
//! - [`basis`]: Sparse, path-sorted coordinate tuples and basis vectors
//!   ([`Basis`](basis::Basis), [`BasisTerm`](basis::BasisTerm),
//!   [`BasisCombination`](basis::BasisCombination)) with linear arithmetic, zero-padded
//!   equality, colexicographical ordering, and tuple conversions
//!   ([`e!`], [`basis_tuple!`]).
//! - [`shape`]: Non-negative tensor shapes ([`Shape`](shape::Shape)), partial-order
//!   compatibility and lattice operations ([`Compatible`](shape::Compatible),
//!   [`CommonRefinement`](shape::CommonRefinement), [`CommonCoarsening`](shape::CommonCoarsening)),
//!   colexicographical coordinate-index conversions ([`crd2idx`](shape::crd2idx),
//!   [`idx2crd`](shape::idx2crd)), bounds checks ([`in_bounds`](shape::in_bounds)), and
//!   iteration ([`coordinates`](shape::coordinates)).
//! - [`stride`]: Congruent integer, F2, and basis strides ([`Stride`](stride::Stride),
//!   [`StrideScalar`](stride::StrideScalar)), leaf scaling and accumulation
//!   ([`StrideMul`](stride::StrideMul), [`Accumulate`](stride::Accumulate),
//!   [`ScaleStride`](stride::ScaleStride)), [`inner_product`](stride::inner_product),
//!   [`prefix_product`](stride::prefix_product), and codomain analysis
//!   ([`CodomainProfile`](stride::CodomainProfile), [`coshape`](stride::coshape)).
//! - [`layout`]: Hierarchical [`Layout`](layout::Layout) mapping coordinates to offsets,
//!   mode indexing/slicing, tiler conversion ([`tiler_to_layout`](layout::tiler_to_layout)),
//!   ordered compact layouts ([`make_ordered_layout`](layout::make_ordered_layout),
//!   [`make_layout_like`](layout::make_layout_like)), rational element recasting
//!   ([`recast`](layout::recast())), and [`make_layout!`].
//! - [`algebra`] and [`mod@swizzle`]: Layout algebra transformations including [`coalesce`](algebra::coalesce()),
//!   [`nullspace`](algebra::nullspace()), [`composition`](algebra::composition()),
//!   [`complement`](algebra::complement()), [`right_inverse`](algebra::right_inverse),
//!   [`left_inverse`](algebra::left_inverse), [`logical_divide`](algebra::logical_divide()),
//!   [`zipped_divide`](algebra::zipped_divide), [`logical_product`](algebra::logical_product()),
//!   [`blocked_product`](algebra::blocked_product), [`raked_product`](algebra::raked_product),
//!   [`greatest_common_domain`](algebra::greatest_common_domain()), and
//!   [`layout_add`](algebra::layout_add()). [`Swizzle`](swizzle::Swizzle) applies a
//!   bit-field XOR permutation to offsets ([`swizzle!`]).
//! - [`tensor`]: CuTe [`Tensor`](tensor::Tensor) pairing an [`Accessor`](tensor::Accessor)
//!   with a [`Layout`](layout::Layout), pure `no_std` accessors
//!   ([`SliceAccessor`](tensor::SliceAccessor), [`CellAccessor`](tensor::CellAccessor),
//!   [`ArrayEngine`](tensor::ArrayEngine), [`Ptr`](tensor::Ptr),
//!   [`ImplicitAccessor`](tensor::ImplicitAccessor),
//!   [`TransformAccessor`](tensor::TransformAccessor), and
//!   [`OutOfBoundsAccessor`](tensor::OutOfBoundsAccessor)), coordinate indexing/slicing,
//!   layout-algebra pass-through, [`domain_offset`](tensor::domain_offset), partitioning
//!   ([`inner_partition`](tensor::inner_partition), [`outer_partition`](tensor::outer_partition),
//!   [`local_tile`](tensor::local_tile), [`local_tile_proj`](tensor::local_tile_proj),
//!   [`local_partition`](tensor::local_partition),
//!   [`local_partition_proj`](tensor::local_partition_proj)),
//!   [`identity_tensor`](tensor::identity_tensor), [`make_tensor`](tensor::make_tensor), and
//!   owning constructors ([`make_owned_tensor`](tensor::make_owned_tensor),
//!   [`make_tensor_like`](tensor::make_tensor_like),
//!   [`make_fragment_like`](tensor::make_fragment_like)).
//! - [`algorithm`]: CuTe tensor algorithms and layout views including reference
//!   [`reference::copy`](algorithm::reference::copy) and [`gemm`](algorithm::gemm),
//!   layout-analyzed [`copy`](algorithm::copy()) and [`copy_plan`](algorithm::copy_plan),
//!   pure-safe element-wise operations ([`fill`](algorithm::fill), [`clear`](algorithm::clear),
//!   [`copy_if`](algorithm::copy_if), [`axpby`](algorithm::axpby),
//!   [`for_each`](algorithm::for_each), [`transform`](algorithm::transform),
//!   [`transform_into`](algorithm::transform_into),
//!   [`transform_zip`](algorithm::transform_zip)),
//!   Copy/MMA atoms and thread–value tilings ([`algorithm::atom`],
//!   [`copy_with`](algorithm::copy_with), [`copy_if_with`](algorithm::copy_if_with),
//!   [`gemm_mma`](algorithm::gemm_mma), [`gemm_fma`](algorithm::gemm_fma)),
//!   named mode rearrangement [`einfold`](algorithm::einfold()) ([`einfold!`]),
//!   explicit binary tensor contraction [`einsum`](algorithm::einsum()) ([`einsum!`]),
//!   and N-D convolution views [`im2col`](algorithm::im2col()),
//!   [`im2col_coord`](algorithm::im2col_coord), and [`im2col_padded`](algorithm::im2col_padded).
//!
//! # Main Root Interfaces
//!
//! - [`StaticValue`] and [`static_value`]: Zero-cost compile-time constant instantiation
//!   for strictly static types (`Int<N>`, `F2<Int<N>>`, tuples, canonical basis values,
//!   layouts, and swizzles), never implemented for runtime `int`.
//! - [`NoodleCuteError`]: Unified error classification enum for arithmetic, shape/stride,
//!   layout, storage, and algorithm panics in both `const` and runtime contexts.
//! - [`Result<T>`]: Crate-wide `Result` alias using [`NoodleCuteError`].
//! - [`NoodleCuteResult`]: `const`-capable `.v()` helper that unwraps `Ok` or panics
//!   with the error message on `Err`.
//!
//! # Examples
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{basis::as_tuple, e, int_tuple, integer::Int, walk_mode};
//!
//! const SUM: Int<5> = Int::<2> + Int::<3>;
//! let t = int_tuple![SUM, (Int::<4>, 6i32)];
//! assert_eq!(walk_mode!(t, 1, 0), Int::<4>);
//! assert_eq!(as_tuple(Int::<2> * e![0] + 3i32 * e![1]), int_tuple![Int::<2>, 3i32]);
//! ```
//!
//! # Notes
//!
//! - `#![no_std]` crate requiring nightly Rust features for generic `const` arguments
//!   and `const` trait implementations.
//! - Arithmetic and algebraic operations expose one unified interface: invalid static
//!   operations fail at compile time, while invalid dynamic operations panic at runtime
//!   in both debug and release builds.

#![no_std]
#![feature(
    min_adt_const_params,
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
    trait_alias,
    const_trait_impl,
    const_ops,
    const_cmp,
    const_option_ops,
    const_try,
    unboxed_closures,
    fn_traits
)]
#![allow(incomplete_features)]

pub mod algebra;
pub mod algorithm;
pub mod basis;
pub mod f2;
pub mod integer;
pub mod layout;
pub mod shape;
pub mod slice;
pub mod stride;
pub mod swizzle;
pub mod tensor;
pub mod tuple;

/// A value fully determined by its type.
///
/// Implemented only for zero-sized compile-time constant types (never for runtime
/// [`int`](integer::int) or dynamic wrappers), providing both a strict compile-time
/// staticness bound and zero-cost constant instantiation from type aliases or
/// associated types.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{
///     StaticValue, int_tuple,
///     integer::Int,
///     layout::Layout,
///     static_value,
///     tuple::{Nil, Til},
/// };
///
/// type StaticTile = Layout<
///     Til<Int<4>, Til<Int<8>, Nil>>,
///     Til<Int<1>, Til<Int<4>, Nil>>,
/// >;
/// const TILE: StaticTile = StaticTile::VALUE;
/// assert_eq!(
///     TILE,
///     Layout::new(int_tuple![Int::<4>, Int::<8>], int_tuple![Int::<1>, Int::<4>])
/// );
/// assert_eq!(static_value::<Int<7>>(), Int::<7>);
/// ```
///
/// Dynamic [`int`](integer::int) values never implement `StaticValue`:
///
/// ```compile_fail,E0277
/// # #![feature(const_trait_impl)]
/// use noodle_cute::StaticValue;
/// let _ = <i32 as StaticValue>::VALUE;
/// ```
///
/// Invalid static layouts and swizzles cannot be instantiated via `StaticValue`:
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{StaticValue, integer::Int, layout::Layout};
/// let _ = <Layout<Int<-1>, Int<1>> as StaticValue>::VALUE;
/// ```
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{StaticValue, swizzle::StaticSwizzle};
/// let _ = <StaticSwizzle<3, 0, -2> as StaticValue>::VALUE;
/// ```
///
/// Negative `F2` values and non-canonical (unsorted) `BasisCombination` types are also rejected:
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{StaticValue, f2::StaticF2};
/// let _ = <StaticF2<-1> as StaticValue>::VALUE;
/// ```
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::{
///     StaticValue,
///     basis::{Basis, BasisCombination},
///     mode,
///     tuple::{Nil, Til},
/// };
/// type Unsorted = BasisCombination<Til<Basis<mode![1]>, Til<Basis<mode![0]>, Nil>>>;
/// let _ = <Unsorted as StaticValue>::VALUE;
/// ```
#[diagnostic::on_unimplemented(
    message = "type `{Self}` does not determine a constant value",
    label = "runtime values cannot be read through StaticValue",
    note = "use static Int<N> leaves for type-determined constants; dynamic `int` values cannot implement StaticValue"
)]
pub const trait StaticValue: Copy {
    /// The unique constant value encoded by `Self`.
    const VALUE: Self;
}

/// Instantiates the unique constant value encoded by a [`StaticValue`] type `T`.
pub const fn static_value<T: StaticValue>() -> T {
    T::VALUE
}

/// Errors reported by integer arithmetic, shape/stride/layout validation, and tensor operations.
///
/// Public arithmetic and algebraic operations expose one unified interface (compile-time errors
/// for invalid static operands, runtime panics for invalid dynamic operands). These variants
/// provide structured error categories for unified `const` and runtime panic messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoodleCuteError {
    /// An arithmetic operation produced a value outside the representable [`int`](integer::int) range.
    ArithmeticOverflow,
    /// Division or remainder was attempted with a zero divisor.
    DivisionByZero,
    /// Negating or taking the absolute value of `int::MIN` (`i32::MIN`) would overflow.
    SignedMinValueOverflow,
    /// An unsigned value cannot be represented by the signed [`int`](integer::int) type.
    UnsignedToSignedOutOfRange,
    /// A shape extent, stride sign, or size parameter is invalid for the operation.
    InvalidShape,
    /// Shapes, strides, or coordinates have incompatible nesting or axis counts.
    StructureMismatch,
    /// A logical coordinate lies outside the layout's domain.
    CoordinateOutOfBounds,
    /// An axis, top-level mode, or flattened leaf index is out of bounds.
    AxisOutOfBounds,
    /// Distinct logical coordinates map to the same address when uniqueness is required.
    NonInjective,
    /// The layout does not cover the codomain required by the operation.
    NotSurjective,
    /// An extent or stride fails a divisibility requirement.
    NotDivisible,
    /// An address or storage offset lies outside the permitted storage range.
    AddressOutOfBounds,
    /// Operands have incompatible logical sizes.
    SizeMismatch,
    /// The requested operation is unsupported for this layout or parameter combination.
    Unsupported,
}

impl NoodleCuteError {
    /// Returns the canonical static error description for this error variant.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ArithmeticOverflow => "integer arithmetic overflow",
            Self::DivisionByZero => "integer division by zero",
            Self::SignedMinValueOverflow => "cannot negate or take the absolute value of int::MIN",
            Self::UnsignedToSignedOutOfRange => "unsigned integer is out of range for signed int",
            Self::InvalidShape => "invalid shape extent or size parameter",
            Self::StructureMismatch => "shape, stride, or coordinate structures differ",
            Self::CoordinateOutOfBounds => "coordinate is outside the layout domain",
            Self::AxisOutOfBounds => "axis is outside the layout rank",
            Self::NonInjective => "layout maps multiple coordinates to the same address",
            Self::NotSurjective => "layout does not cover the requested codomain",
            Self::NotDivisible => "layout or tile extent is not divisible",
            Self::AddressOutOfBounds => "address is outside storage",
            Self::SizeMismatch => "logical sizes differ",
            Self::Unsupported => "operation is unsupported for this layout",
        }
    }

    /// Panics (at compile time in `const` contexts or at runtime otherwise) with this error's message.
    #[cold]
    #[track_caller]
    pub const fn panic(self) -> ! {
        panic!("{}", self.as_str())
    }

    /// Panics with both the canonical error category and a specific diagnostic detail.
    #[cold]
    #[track_caller]
    pub(crate) const fn panic_with(self, detail: &str) -> ! {
        if detail.is_empty() {
            self.panic();
        }
        let prefix = self.as_str().as_bytes();
        let sep = b": ";
        let suffix = detail.as_bytes();
        let total = prefix.len() + sep.len() + suffix.len();
        if total > 256 {
            panic!("{}", detail);
        }
        let mut buf = [0u8; 256];
        let mut pos = 0;
        let mut i = 0;
        while i < prefix.len() {
            buf[pos] = prefix[i];
            pos += 1;
            i += 1;
        }
        i = 0;
        while i < sep.len() {
            buf[pos] = sep[i];
            pos += 1;
            i += 1;
        }
        i = 0;
        while i < suffix.len() {
            buf[pos] = suffix[i];
            pos += 1;
            i += 1;
        }
        let msg = match core::str::from_utf8(buf.split_at(pos).0) {
            Ok(s) => s,
            Err(_) => self.as_str(),
        };
        panic!("{}", msg)
    }

    /// Asserts `cond`, panicking with [`Self::panic`] if `false`.
    #[track_caller]
    #[allow(dead_code)]
    pub(crate) const fn check(self, cond: bool) {
        if !cond {
            self.panic();
        }
    }

    /// Asserts `cond`, panicking with [`Self::panic_with`] if `false`.
    #[track_caller]
    pub(crate) const fn check_with(self, cond: bool, detail: &str) {
        if !cond {
            self.panic_with(detail);
        }
    }
}

impl core::fmt::Display for NoodleCuteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl core::error::Error for NoodleCuteError {}

/// Result type alias using [`NoodleCuteError`].
pub type Result<T> = core::result::Result<T, NoodleCuteError>;

/// Helper trait for unwrapping [`Result`] values in both `const` and runtime contexts.
pub const trait NoodleCuteResult {
    /// The unwrapped success value type.
    type Output;

    /// Returns the `Ok` value, panicking with the error description on `Err`.
    fn v(self) -> Self::Output;
}

const impl<T: Copy> NoodleCuteResult for Result<T> {
    type Output = T;
    #[track_caller]
    fn v(self) -> Self::Output {
        match self {
            Ok(t) => t,
            Err(e) => e.panic(),
        }
    }
}
