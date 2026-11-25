//! CuTe [`Tensor`]: an [`Accessor`] composed with a [`Layout`] (`#![no_std]`, zero allocation).
//!
//! A tensor pairs a random-access handle (`accessor`) with a hierarchical [`Layout`]
//! (`layout`):
//!
//! $$T(c) = (e \circ L)(c) = e\[L(c)\]$$
//!
//! Indexing with a complete coordinate dereferences `accessor` at `L(c)`; indexing with a
//! coordinate containing `..` ([`core::ops::RangeFull`]) offsets `accessor` by `L(c)` and
//! returns a sub-[`Tensor`] over the modes left unnamed. Layout-algebra operations
//! ([`coalesce`](crate::algebra::coalesce()), [`coalesce_extended`](crate::algebra::coalesce_extended),
//! [`coalesce_z`](crate::algebra::coalesce_z), [`composition`](crate::algebra::composition()),
//! [`logical_divide`](crate::algebra::logical_divide()),
//! [`logical_divide_at`](crate::algebra::logical_divide_at),
//! [`zipped_divide`](crate::algebra::zipped_divide),
//! [`zipped_divide_at`](crate::algebra::zipped_divide_at)) pass straight through to the
//! layout while keeping the accessor intact.
//!
//! # Core Types & Submodules
//!
//! - [`mod@accessor`]: Accessor traits ([`LinearOffset`], [`AccessorOffset`], [`Accessor`],
//!   [`MutableAccessor`], [`IntoLinearAccessor`]) and pure `#![no_std]` backends
//!   ([`SliceAccessor`], [`CellAccessor`], [`Ptr`], [`ImplicitAccessor`], [`TransformAccessor`],
//!   [`OutOfBoundsAccessor`]).
//! - [`mod@ops`]: Indexing/slicing ([`TensorIndex`], [`TensorSet`]), layout-algebra
//!   pass-through, and block tiling ([`local_tile`]).
//! - **Tensor core**: [`Tensor`], [`IsTensor`], [`is_tensor`], [`identity_tensor`],
//!   [`IntoTensorLayout`], and [`make_tensor`].
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algebra::coalesce;
//! use noodle_cute::basis::as_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::tensor::{identity_tensor, local_tile, make_tensor};
//! use noodle_cute::{int_tuple, tuple};
//!
//! let mut buf = [0i32; 16];
//! let layout = Layout::new(int_tuple![Int::<4>, Int::<4>], int_tuple![Int::<4>, Int::<1>]);
//! let mut t = make_tensor(&mut buf, layout);
//! t.set(tuple!(1i32, 2i32), 42);
//! assert_eq!(t(1i32, 2i32), 42);
//!
//! let row = t(1i32, ..);
//! assert_eq!(row.shape(), int_tuple![Int::<4>]);
//! assert_eq!(row(2i32), 42);
//!
//! let col_major = make_tensor(
//!     &mut buf,
//!     Layout::new(int_tuple![Int::<4>, Int::<4>], int_tuple![Int::<1>, Int::<4>]),
//! );
//! assert_eq!(coalesce(col_major).layout, Layout::new(Int::<16>, Int::<1>));
//!
//! let id = identity_tensor(int_tuple![Int::<8>, Int::<12>]);
//! let tile = local_tile(id, int_tuple![Int::<4>, Int::<6>], int_tuple![1i32, 1i32]);
//! assert_eq!(as_tuple(tile(tuple!((3i32, 5i32)))), int_tuple![7i32, 11i32]);
//! ```
//!
//! # Compile-Time Rejections
//!
//! Writing through a partial (slicing) coordinate fails at compile time:
//!
//! ```compile_fail,E0271
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::tensor::make_tensor;
//! use noodle_cute::{int_tuple, tuple};
//!
//! let mut buf = [0i32; 16];
//! let mut t = make_tensor(
//!     &mut buf,
//!     Layout::new(int_tuple![Int::<4>, Int::<4>], int_tuple![Int::<4>, Int::<1>]),
//! );
//! t.set(tuple!(1i32, ..), 0);
//! ```
//!
//! Writing to a read-only accessor (such as [`ImplicitAccessor`] or [`SliceAccessor`]) fails at
//! compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::tensor::identity_tensor;
//! use noodle_cute::{int_tuple, tuple};
//!
//! let mut id = identity_tensor(int_tuple![Int::<3>, Int::<4>]);
//! id.set(tuple!(1i32, 2i32), Int::<0>);
//! ```
//!
//! Passing a coordinate whose rank does not match the tile grid to [`local_tile`] fails at
//! compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::tensor::{identity_tensor, local_tile};
//! use noodle_cute::int_tuple;
//!
//! let id = identity_tensor(int_tuple![Int::<8>, Int::<12>]);
//! let _ = local_tile(id, int_tuple![Int::<4>, Int::<6>], int_tuple![1i32]);
//! ```
//!
//! Passing a bare [`Layout`] instead of a [`Tensor`] to [`local_tile`] fails at compile time:
//!
//! ```compile_fail,E0308
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::tensor::local_tile;
//! use noodle_cute::int_tuple;
//!
//! let layout = Layout::from_shape(int_tuple![Int::<8>, Int::<12>]);
//! let _ = local_tile(layout, int_tuple![Int::<4>, Int::<6>], int_tuple![1i32, 1i32]);
//! ```
//!
//! Passing a non-accessor value to [`Tensor::new`] fails at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::tensor::Tensor;
//!
//! let _ = Tensor::new(false, Layout::from_shape(Int::<4>));
//! ```
//!
//! # Notes
//!
//! - Every built-in accessor and `Tensor` is `Copy` and `#![no_std]` with zero heap allocation;
//!   [`SliceAccessor`], [`ImplicitAccessor`], [`TransformAccessor`], [`identity_tensor`],
//!   [`make_tensor`] (over immutable slices/arrays), and [`local_tile`] are all `const`-evaluable.
//! - `PartialEq` checks exact structural equality (`accessor == other.accessor && layout ==
//!   other.layout`), matching PyCuTe.

use core::fmt::{self, Display, Formatter};
use core::ops::RangeFull;

use crate::basis::{
    BasisCombination, BasisTerm, BasisTuple, MakeBasisLikeOutput, MakeBasisLikeTopProfile,
    make_basis_like,
};
use crate::integer::{IS_NON_NEG, Int, add, int};
use crate::layout::{Coprofile, IsLayout, Layout};
use crate::shape::{Shape, ValidShapeExtent};
use crate::stride::{Coshape, LinearBounds, PrefixProduct, Stride};
use crate::swizzle::F2;
use crate::tuple::{Get, GetOutput, Item, Nil, Til, Tuple};

pub mod accessor;
pub mod ops;

pub use accessor::*;
pub use ops::*;

/// Compile-time tensor type predicate backing [`is_tensor`].
pub const trait IsTensor {
    /// `true` iff `Self` is a [`Tensor`].
    const IS_TENSOR: bool;
}

/// Returns `true` iff `x` is a [`Tensor`].
///
/// ```
/// # #![feature(const_trait_impl)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
/// use noodle_cute::tensor::{identity_tensor, is_tensor};
///
/// let t = identity_tensor(int_tuple![Int::<2>, Int::<2>]);
/// let l = Layout::from_shape(int_tuple![Int::<2>, Int::<2>]);
/// assert!(is_tensor(&t));
/// assert!(!is_tensor(&l));
/// assert!(!is_tensor(&42i32));
/// ```
pub const fn is_tensor<T: IsTensor>(_: &T) -> bool {
    T::IS_TENSOR
}

/// A CuTe Tensor: an accessor paired with a [`Layout`].
#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct Tensor<A, L> {
    /// Random-access backend mapping layout offsets to values.
    pub accessor: A,
    /// Hierarchical layout mapping coordinates to offsets.
    pub layout: L,
}

const impl<A, L> IsTensor for Tensor<A, L> {
    const IS_TENSOR: bool = true;
}

const impl<S, D> IsTensor for Layout<S, D> {
    const IS_TENSOR: bool = false;
}

const impl IsTensor for Nil {
    const IS_TENSOR: bool = false;
}

const impl IsTensor for int {
    const IS_TENSOR: bool = false;
}

const impl<const N: int> IsTensor for Int<N> {
    const IS_TENSOR: bool = false;
}

const impl<V, P> IsTensor for BasisTerm<V, P> {
    const IS_TENSOR: bool = false;
}

const impl<T: BasisTuple> IsTensor for BasisCombination<T> {
    const IS_TENSOR: bool = false;
}

const impl<H, T> IsTensor for Til<H, T> {
    const IS_TENSOR: bool = false;
}

const impl<V> IsTensor for F2<V> {
    const IS_TENSOR: bool = false;
}

const impl IsTensor for RangeFull {
    const IS_TENSOR: bool = false;
}

const impl<A: Copy, L: Copy> IsLayout for Tensor<A, L> {
    const IS_LAYOUT: bool = false;
}

impl<A, S, D> Tensor<A, Layout<S, D>> {
    /// Constructs a [`Tensor`] pairing an [`AccessorBase`] with a valid `layout`.
    ///
    /// Unlike [`make_tensor`], `Tensor::new` does not check backing storage capacity or
    /// restrict the codomain to linear offsets (making it suitable for virtual accessors,
    /// custom user accessors, or manually positioned buffers); offset-type compatibility
    /// with `accessor` is enforced when indexing or slicing.
    pub const fn new(accessor: A, layout: Layout<S, D>) -> Self
    where
        A: AccessorBase,
        S: Shape,
        D: Stride<S>,
    {
        Self { accessor, layout }
    }

    /// Returns a reference to this tensor's accessor.
    pub const fn accessor(&self) -> &A {
        &self.accessor
    }

    /// Returns a copy of this tensor's layout.
    pub const fn layout(&self) -> Layout<S, D>
    where
        S: Copy,
        D: Copy,
    {
        self.layout
    }

    /// Returns a copy of this tensor's domain shape.
    pub const fn shape(&self) -> S
    where
        S: Copy,
    {
        self.layout.shape
    }

    /// Returns a copy of this tensor's stride.
    pub const fn stride(&self) -> D
    where
        D: Copy,
    {
        self.layout.stride
    }

    /// Returns the total number of integral coordinates in the tensor's domain.
    pub const fn size(&self) -> int
    where
        S: [const] Shape,
    {
        self.layout.size()
    }

    /// Returns the number of top-level modes in the tensor's domain shape.
    pub const fn rank(&self) -> int
    where
        S: Item,
    {
        self.layout.rank()
    }

    /// Returns the nesting depth of the tensor's domain shape.
    pub const fn depth(&self) -> int
    where
        S: Item,
    {
        self.layout.depth()
    }

    /// Returns the total number of leaves in the tensor's domain shape.
    pub const fn length(&self) -> int
    where
        S: Item,
    {
        self.layout.length()
    }

    /// Returns `true` iff both `shape` and `stride` are statically known.
    pub const fn is_static(&self) -> bool
    where
        S: Item,
        D: Item,
    {
        self.layout.is_static()
    }

    /// Computes the codomain shape bounding all offsets produced by `self.layout`.
    pub const fn coshape(&self) -> <S as Coshape<D>>::Output
    where
        S: [const] Coshape<D>,
        D: Stride<S>,
    {
        self.layout.coshape()
    }

    /// Computes the total size of the layout codomain (`size(coshape(self))`).
    pub const fn cosize(&self) -> int
    where
        S: [const] Coshape<D>,
        D: Stride<S>,
        <S as Coshape<D>>::Output: [const] Shape,
    {
        self.layout.cosize()
    }

    /// Computes the hierarchical profile of the layout's codomain.
    pub const fn coprofile(&self) -> <Layout<S, D> as Coprofile>::Output
    where
        Layout<S, D>: [const] Coprofile,
    {
        self.layout.coprofile()
    }

    /// Returns the sub-tensor at top-level mode `I` (supporting negative indices) while
    /// keeping the same accessor and offset.
    pub const fn get<const I: int>(&self) -> Tensor<A, GetOutput<Layout<S, D>, I>>
    where
        A: Copy,
        Layout<S, D>: [const] Get<I>,
    {
        Tensor {
            accessor: self.accessor,
            layout: self.layout.get::<I>(),
        }
    }
}

impl<'a, T, S, D> Tensor<SliceAccessor<'a, T>, Layout<S, D>> {
    /// Creates a read-only tensor viewing borrowed slice `data` through `layout`.
    pub const fn from_slice(data: &'a [T], layout: Layout<S, D>) -> Self
    where
        S: Shape,
        D: Stride<S>,
    {
        Self::new(SliceAccessor::new(data), layout)
    }
}

impl<'a, T, S, D> Tensor<CellAccessor<'a, T>, Layout<S, D>> {
    /// Creates a safe, `Copy` read/write tensor viewing mutable slice `data` through `layout`.
    pub fn from_slice_mut(data: &'a mut [T], layout: Layout<S, D>) -> Self
    where
        S: Shape,
        D: Stride<S>,
    {
        Self::new(CellAccessor::from_slice_mut(data), layout)
    }
}

const impl<A1, L1, A2, L2> PartialEq<Tensor<A2, L2>> for Tensor<A1, L1>
where
    A1: [const] PartialEq<A2>,
    L1: [const] PartialEq<L2>,
{
    fn eq(&self, other: &Tensor<A2, L2>) -> bool {
        self.accessor == other.accessor && self.layout == other.layout
    }
}

const impl<A, L> Eq for Tensor<A, L>
where
    A: [const] Eq,
    L: [const] Eq,
{
}

impl<A: Display, L: Display> Display for Tensor<A, L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} o {}", self.accessor, self.layout)
    }
}

/// Constructs the identity tensor mapping every coordinate in `shape` to itself.
///
/// Uses coordinate-basis strides (`make_basis_like(shape)`) over an [`ImplicitAccessor`]
/// rooted at `Int<0>`, requiring zero allocation. Slicing or tiling an identity tensor
/// alongside a data tensor recovers each element's global coordinates for predication.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::basis::as_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::tensor::identity_tensor;
/// use noodle_cute::int_tuple;
///
/// let id = identity_tensor(int_tuple![Int::<3>, Int::<4>]);
/// assert_eq!(as_tuple(id(1i32, 2i32)), int_tuple![1i32, 2i32]);
/// assert_eq!(as_tuple(id(1i32, ..)(2i32)), int_tuple![1i32, 2i32]);
/// ```
pub const fn identity_tensor<S>(
    shape: S,
) -> Tensor<ImplicitAccessor<Int<0>>, Layout<S, MakeBasisLikeOutput<S>>>
where
    S: [const] Shape,
    S::Profile: [const] MakeBasisLikeTopProfile,
    MakeBasisLikeOutput<S>: Stride<S>,
{
    let stride = make_basis_like(shape);
    Tensor::new(ImplicitAccessor::new(Int::<0>), Layout::new(shape, stride))
}

/// Promotes a [`Layout`] or [`Shape`] into a [`Layout`] for tensor construction.
///
/// - An existing [`Layout<S, D>`] passes through unchanged.
/// - A [`Shape`] promotes via [`Layout::from_shape`] (compact column-major `prefix_product`).
pub const trait IntoTensorLayout {
    /// Resulting layout shape type.
    type Shape: [const] Shape;
    /// Resulting layout stride type.
    type Stride: Stride<Self::Shape>;

    /// Converts `self` into a [`Layout`].
    fn into_tensor_layout(self) -> Layout<Self::Shape, Self::Stride>;
}

const impl<S: [const] Shape, D: Stride<S>> IntoTensorLayout for Layout<S, D> {
    type Shape = S;
    type Stride = D;

    fn into_tensor_layout(self) -> Layout<S, D> {
        self
    }
}

const impl IntoTensorLayout for Nil {
    type Shape = Nil;
    type Stride = Nil;

    fn into_tensor_layout(self) -> Layout<Nil, Nil> {
        Layout::from_shape(Nil)
    }
}

const impl IntoTensorLayout for int {
    type Shape = int;
    type Stride = Int<1>;

    fn into_tensor_layout(self) -> Layout<int, Int<1>> {
        Layout::from_shape(self)
    }
}

const impl<const N: int> IntoTensorLayout for Int<N>
where
    (): [const] ValidShapeExtent<{ IS_NON_NEG::<N> }>,
{
    type Shape = Int<N>;
    type Stride = Int<1>;

    fn into_tensor_layout(self) -> Layout<Int<N>, Int<1>> {
        Layout::from_shape(self)
    }
}

const impl<H, T> IntoTensorLayout for Til<H, T>
where
    H: [const] Shape,
    T: [const] Tuple + [const] Shape,
    Self: [const] PrefixProduct,
    <Self as PrefixProduct>::Stride: Stride<Self>,
{
    type Shape = Self;
    type Stride = <Self as PrefixProduct>::Stride;

    fn into_tensor_layout(self) -> Layout<Self::Shape, Self::Stride> {
        Layout::from_shape(self)
    }
}

/// Binds linear backing `storage` (`&[T]`, `&mut [T]`, `&[T; N]`, `&mut [T; N]`,
/// [`SliceAccessor`], [`CellAccessor`], or [`Ptr`]) to `layout` (a [`Layout`] or [`Shape`]).
///
/// Validates at compile time that `layout` has a linear codomain (rejecting coordinate-basis
/// layouts) and verifies that every in-bounds coordinate's reachable offset interval
/// `[min_offset, max_offset]` (including negative strides) falls within `storage`.
/// Zero-extent shapes have an empty domain and succeed on empty storage, whereas rank-0
/// [`Nil`] shapes contain one element at offset `0` and require `0 <= base_offset < len`.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
/// use noodle_cute::tensor::make_tensor;
///
/// let mut buf = [0.0f64; 16];
/// let mut t = make_tensor(
///     &mut buf,
///     Layout::new(int_tuple![Int::<4>, Int::<4>], int_tuple![Int::<4>, Int::<1>]),
/// );
/// t.set(int_tuple![1i32, 2i32], 42.0);
/// assert_eq!(t(1i32, 2i32), 42.0);
///
/// let mut buf2 = [0i32; 15];
/// let t2 = make_tensor(&mut buf2, int_tuple![Int::<3>, Int::<5>]);
/// assert_eq!(
///     t2.layout,
///     Layout::new(int_tuple![Int::<3>, Int::<5>], int_tuple![Int::<1>, Int::<3>])
/// );
/// ```
///
/// Coordinate-codomain layouts are rejected at compile time:
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
/// use noodle_cute::tensor::make_tensor;
/// use noodle_cute::{e, int_tuple, tuple};
///
/// let mut buf = [0.0f64; 4];
/// let _ = make_tensor(
///     &mut buf,
///     Layout::new(int_tuple![Int::<2>, Int::<2>], tuple![e![0], e![1]]),
/// );
/// ```
pub const fn make_tensor<B, L>(
    storage: B,
    layout: L,
) -> Tensor<B::Accessor, Layout<L::Shape, L::Stride>>
where
    B: [const] IntoLinearAccessor,
    L: [const] IntoTensorLayout,
    L::Shape: [const] LinearBounds<L::Stride>,
{
    let layout = layout.into_tensor_layout();
    let (accessor, base_offset, total_len) = storage.into_linear_accessor();
    if let Some((min_off, max_off)) = layout.linear_bounds() {
        let lo = add(base_offset, min_off);
        let hi = add(base_offset, max_off);
        assert!(
            lo >= 0 && (hi as usize) < total_len,
            "layout codomain exceeds backing storage bounds"
        );
    }
    Tensor::new(accessor, layout)
}
