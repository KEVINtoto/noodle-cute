//! Accessors: pure `#![no_std]` memory and functional backends for CuTe [`Tensor`]s.
//!
//! A [`Layout`] maps a coordinate to a codomain offset; an accessor
//! maps that offset to a value (or shifts itself to a sub-view at that offset). Keeping
//! storage and indexing orthogonal allows the same tensor API and layout algebra to address
//! borrowed slices, mutable cell buffers, raw pointers, coordinate origins, and lazily
//! transformed views without any heap allocation.
//!
//! # Main Interfaces
//!
//! - **Core traits**: [`AccessorBase`], [`LinearOffset`], [`AccessorOffset`], [`Accessor`],
//!   [`MutableAccessor`], and [`IntoLinearAccessor`].
//! - **Memory-backed accessors**:
//!   - [`SliceAccessor`]: Read-only borrowed slice `&[T]` with flattened signed offset (`Copy`, `const`).
//!   - [`CellAccessor`]: Safe, `#![no_std]`, `Copy` read/write view over `&[Cell<T>]` (created from
//!     `&mut [T]` via [`CellAccessor::from_slice_mut`]), supporting simultaneous mutable sub-views.
//!   - [`ArrayEngine`]: Owned 16-byte-aligned stack array `[T; N]` with bounds-checked element
//!     access and zero-`unsafe` [`SliceAccessor`] / [`CellAccessor`] view borrowing.
//!   - [`Ptr`]: Bounds-checked raw pointer `*mut T` into externally owned storage (`Copy`, `unsafe`
//!     construction via [`Ptr::from_raw`]).
//!   - [`F2OffsetAccessor`]: Preserves an $\mathbb{F}_2\[x\]$ slicing origin over a memory-backed
//!     base accessor so subsequent `F2` offsets combine via XOR before linear address conversion.
//! - **Virtual / functional accessors**:
//!   - [`ImplicitAccessor`]: Returns `origin + offset` without memory dereferencing (backs
//!     [`identity_tensor`](super::identity_tensor)).
//!   - [`TransformAccessor`]: Read-only wrapper applying a transform `F` on every load while
//!     propagating offsets to the inner accessor.
//!   - [`OutOfBoundsAccessor`]: Read-only coordinate-indexed wrapper around a
//!     [`Tensor`] that returns a fill value whenever the accumulated coordinate
//!     falls outside the tensor's shape.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::tensor::{
//!     Accessor, CellAccessor, ImplicitAccessor, MutableAccessor, SliceAccessor, TransformAccessor,
//! };
//!
//! let mut buf = [0i32, 1, 2, 3, 4, 5];
//! let acc = CellAccessor::from_slice_mut(&mut buf);
//! let mut shifted = acc + 2i32;
//! shifted.store(1i32, 99);
//! assert_eq!(acc.load(3i32), 99);
//!
//! let ro = SliceAccessor::new(&buf);
//! let sq = TransformAccessor::new(ro, |x: i32| x * x);
//! assert_eq!((sq + 2i32).load(0i32), 4);
//!
//! let ia = ImplicitAccessor::new(Int::<10>);
//! assert_eq!((ia + 3i32).load(2i32), 15);
//! ```
//!
//! # Notes
//!
//! - All memory-backed accessors flatten chained offsets against their base storage and validate
//!   the effective index `base_offset + delta` against storage bounds, panicking on out-of-bounds
//!   access or signed [`int`] overflow.
//! - Offsetting a memory-backed accessor by an [`F2`] value returns an [`F2OffsetAccessor`]
//!   that keeps the origin in $\mathbb{F}_2\[x\]$ and flattens further `F2` offsets with XOR.
//! - `PartialEq` on memory-backed accessors compares effective element addresses (matching
//!   PyCuTe's pointer equality), while [`ImplicitAccessor`] delegates to [`basis_eq`].

use core::any::type_name;
use core::cell::Cell;
use core::fmt::{self, Debug, Display, Formatter};
use core::ops::Add;

use crate::NoodleCuteError;
use crate::basis::{
    BasisCombination, IntoBasisCoord, IntoColexTerms, MergeEq, basis_eq, into_basis_coord,
};
use crate::f2::{F2, F2Scalar};
use crate::integer::{Int, Integer, add, int};
use crate::layout::Layout;
use crate::shape::{InBounds, Shape, in_bounds};
use crate::stride::{Accumulate, Stride};
use crate::tuple::Nil;

use super::{Tensor, TensorIndex};

/// Marker base trait for all CuTe accessor types.
///
/// Implemented by all built-in accessors and open to custom user-defined accessors.
/// Required by [`Tensor::new`](super::Tensor::new) so non-accessor values are rejected
/// at construction time.
pub const trait AccessorBase: Copy {}

/// Extracts a signed linear element offset ([`int`]) from a layout codomain value.
///
/// Implemented for all [`Integer`] scalars (`int`, [`Int<N>`]), binary-field [`F2`] values,
/// the empty [`BasisCombination<Nil>`] (`0`), and [`Nil`] (`0`). Non-empty basis coordinates
/// intentionally do not implement `LinearOffset`, rejecting coordinate-strided layouts on
/// linear memory accessors at compile time.
pub const trait LinearOffset: Copy {
    /// Returns the linear element offset as a signed [`int`].
    fn linear_offset(self) -> int;
}

const impl<I: [const] Integer> LinearOffset for I {
    fn linear_offset(self) -> int {
        self.value()
    }
}

const impl<V: [const] F2Scalar> LinearOffset for F2<V> {
    fn linear_offset(self) -> int {
        self.value()
    }
}

const impl LinearOffset for BasisCombination<Nil> {
    fn linear_offset(self) -> int {
        0
    }
}

const impl LinearOffset for Nil {
    fn linear_offset(self) -> int {
        0
    }
}

/// Computes `base_offset + delta` with overflow and bounds checking against `len`.
const fn resolve_linear_index<Off: [const] LinearOffset>(
    base_offset: int,
    delta: Off,
    len: usize,
) -> usize {
    let idx = add(base_offset, delta.linear_offset());
    NoodleCuteError::AddressOutOfBounds.check_with(idx >= 0, "accessor index is negative");
    let uidx = idx as usize;
    NoodleCuteError::AddressOutOfBounds.check_with(uidx < len, "accessor index out of bounds");
    uidx
}

/// Offsetting an accessor by a layout codomain value `Off` to produce a shifted accessor.
pub const trait AccessorOffset<Off>: AccessorBase {
    /// Shifted accessor type.
    type Output: AccessorBase;

    /// Returns a new accessor shifted by `offset`.
    fn offset(&self, offset: Off) -> Self::Output;
}

/// Read-only random-access handle mapping a codomain offset `Off` to a value.
pub const trait Accessor<Off>: AccessorBase {
    /// Element type returned by [`load`](Self::load).
    type Item;

    /// Reads the value at `offset`.
    fn load(&self, offset: Off) -> Self::Item;
}

/// Read/write random-access handle extending [`Accessor`] with element writes.
pub const trait MutableAccessor<Off>: [const] Accessor<Off> {
    /// Writes `value` at `offset`.
    fn store(&mut self, offset: Off, value: Self::Item);
}

/// Converts linear backing storage into a memory-backed accessor and reports its signed
/// origin offset and total backing length (in elements) for bounds validation in
/// [`make_tensor`](super::make_tensor).
pub const trait IntoLinearAccessor {
    /// Resulting memory-backed accessor type.
    type Accessor: AccessorBase;

    /// Converts `self` into `(accessor, base_offset, total_len)`.
    fn into_linear_accessor(self) -> (Self::Accessor, int, usize);
}

// ------------------------------------------------------------------
// F2OffsetAccessor<A, F2<V>>
// ------------------------------------------------------------------

/// Accessor pairing a linear base accessor `A` with an $\mathbb{F}_2\[x\]$ offset origin.
///
/// Produced when slicing a memory-backed tensor with an [`F2`]-strided layout so that
/// subsequent `F2` offsets combine via XOR (`origin ^ off`) before converting to a linear
/// element index on `base`.
#[derive(Clone, Copy, Debug)]
pub struct F2OffsetAccessor<A, O> {
    /// Underlying linear accessor (may itself carry a linear base offset).
    pub base: A,
    /// Accumulated $\mathbb{F}_2\[x\]$ origin within the swizzled tile.
    pub origin: O,
}

impl<A, O> F2OffsetAccessor<A, O> {
    /// Wraps `base` with an `F2` origin.
    pub const fn new(base: A, origin: O) -> Self {
        Self { base, origin }
    }
}

const impl<A: AccessorBase, V: F2Scalar> AccessorBase for F2OffsetAccessor<A, F2<V>> {}

const impl<A, V, Off, V2> AccessorOffset<Off> for F2OffsetAccessor<A, F2<V>>
where
    A: AccessorBase,
    V: [const] F2Scalar,
    V2: [const] F2Scalar,
    F2<V>: [const] Accumulate<Off, Output = F2<V2>>,
{
    type Output = F2OffsetAccessor<A, F2<V2>>;

    fn offset(&self, offset: Off) -> Self::Output {
        F2OffsetAccessor {
            base: self.base,
            origin: self.origin.accumulate(offset),
        }
    }
}

const impl<A, V, Off, V2> Accessor<Off> for F2OffsetAccessor<A, F2<V>>
where
    A: [const] Accessor<F2<V2>>,
    V: [const] F2Scalar,
    V2: [const] F2Scalar,
    F2<V>: [const] Accumulate<Off, Output = F2<V2>>,
{
    type Item = A::Item;

    fn load(&self, offset: Off) -> Self::Item {
        let total = self.origin.accumulate(offset);
        self.base.load(total)
    }
}

const impl<A, V, Off, V2> MutableAccessor<Off> for F2OffsetAccessor<A, F2<V>>
where
    A: [const] MutableAccessor<F2<V2>>,
    V: [const] F2Scalar,
    V2: [const] F2Scalar,
    F2<V>: [const] Accumulate<Off, Output = F2<V2>>,
{
    fn store(&mut self, offset: Off, value: Self::Item) {
        let total = self.origin.accumulate(offset);
        self.base.store(total, value);
    }
}

const impl<A, V, Off, V2> Add<Off> for F2OffsetAccessor<A, F2<V>>
where
    A: AccessorBase,
    V: [const] F2Scalar,
    V2: [const] F2Scalar,
    F2<V>: [const] Accumulate<Off, Output = F2<V2>>,
{
    type Output = F2OffsetAccessor<A, F2<V2>>;

    fn add(self, rhs: Off) -> Self::Output {
        self.offset(rhs)
    }
}

const impl<A1, V1, A2, V2> PartialEq<F2OffsetAccessor<A2, F2<V2>>> for F2OffsetAccessor<A1, F2<V1>>
where
    A1: [const] PartialEq<A2>,
    V1: [const] F2Scalar,
    V2: [const] F2Scalar,
{
    fn eq(&self, other: &F2OffsetAccessor<A2, F2<V2>>) -> bool {
        self.base == other.base && self.origin == other.origin
    }
}

const impl<A, V> Eq for F2OffsetAccessor<A, F2<V>>
where
    A: [const] Eq,
    V: [const] F2Scalar,
{
}

impl<A: Display, O: Display> Display for F2OffsetAccessor<A, O> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} ^ {}", self.base, self.origin)
    }
}

// ------------------------------------------------------------------
// SliceAccessor<'a, T>
// ------------------------------------------------------------------

/// Read-only accessor backed by a borrowed slice `&'a [T]` with a flattened signed base offset.
///
/// Supports `const` evaluation and `#![no_std]` usage with zero `unsafe`.
#[derive(Debug)]
pub struct SliceAccessor<'a, T> {
    base: &'a [T],
    offset: int,
}

impl<T> Clone for SliceAccessor<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SliceAccessor<'_, T> {}

const impl<T> AccessorBase for SliceAccessor<'_, T> {}

impl<'a, T> SliceAccessor<'a, T> {
    /// Wraps `base` at offset `0`.
    pub const fn new(base: &'a [T]) -> Self {
        Self { base, offset: 0 }
    }

    /// Returns the underlying base slice.
    pub const fn base(&self) -> &'a [T] {
        self.base
    }

    /// Returns `true` iff `self` and `other` share the same base slice pointer.
    pub fn same_base(&self, other: &Self) -> bool {
        core::ptr::eq(self.base.as_ptr(), other.base.as_ptr())
    }

    /// Returns the accumulated signed element offset from `base`.
    pub const fn offset_index(&self) -> int {
        self.offset
    }

    /// Returns the effective raw element pointer at the current offset.
    pub const fn as_ptr(&self) -> *const T {
        self.base.as_ptr().wrapping_offset(self.offset as isize)
    }

    /// Returns the effective byte address of element `0` in this accessor.
    pub fn address(&self) -> usize {
        self.as_ptr() as usize
    }

    /// Borrows the element at `delta` with bounds checking.
    pub const fn get_ref<Off: [const] LinearOffset>(&self, delta: Off) -> &'a T {
        let idx = resolve_linear_index(self.offset, delta, self.base.len());
        &self.base[idx]
    }
}

const impl<'a, T, I: [const] Integer> AccessorOffset<I> for SliceAccessor<'a, T> {
    type Output = Self;

    fn offset(&self, delta: I) -> Self::Output {
        Self {
            base: self.base,
            offset: add(self.offset, delta.value()),
        }
    }
}

const impl<'a, T> AccessorOffset<Nil> for SliceAccessor<'a, T> {
    type Output = Self;

    fn offset(&self, _: Nil) -> Self::Output {
        *self
    }
}

const impl<'a, T> AccessorOffset<BasisCombination<Nil>> for SliceAccessor<'a, T> {
    type Output = Self;

    fn offset(&self, _: BasisCombination<Nil>) -> Self::Output {
        *self
    }
}

const impl<'a, T, V: [const] F2Scalar> AccessorOffset<F2<V>> for SliceAccessor<'a, T> {
    type Output = F2OffsetAccessor<Self, F2<V>>;

    fn offset(&self, delta: F2<V>) -> Self::Output {
        F2OffsetAccessor::new(*self, delta)
    }
}

const impl<'a, T: Copy, Off> Accessor<Off> for SliceAccessor<'a, T>
where
    Off: [const] LinearOffset,
    Self: [const] AccessorOffset<Off>,
{
    type Item = T;

    fn load(&self, delta: Off) -> T {
        let idx = resolve_linear_index(self.offset, delta, self.base.len());
        self.base[idx]
    }
}

const impl<'a, T, Off> Add<Off> for SliceAccessor<'a, T>
where
    Self: [const] AccessorOffset<Off>,
{
    type Output = <Self as AccessorOffset<Off>>::Output;

    fn add(self, rhs: Off) -> Self::Output {
        self.offset(rhs)
    }
}

const impl<'a, T> IntoLinearAccessor for &'a [T] {
    type Accessor = SliceAccessor<'a, T>;

    fn into_linear_accessor(self) -> (Self::Accessor, int, usize) {
        (SliceAccessor::new(self), 0, self.len())
    }
}

const impl<'a, T, const N: usize> IntoLinearAccessor for &'a [T; N] {
    type Accessor = SliceAccessor<'a, T>;

    fn into_linear_accessor(self) -> (Self::Accessor, int, usize) {
        (SliceAccessor::new(self.as_slice()), 0, N)
    }
}

const impl<'a, T> IntoLinearAccessor for SliceAccessor<'a, T> {
    type Accessor = Self;

    fn into_linear_accessor(self) -> (Self::Accessor, int, usize) {
        (self, self.offset, self.base.len())
    }
}

impl<T> PartialEq for SliceAccessor<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self.as_ptr(), other.as_ptr())
    }
}

impl<T> Eq for SliceAccessor<'_, T> {}

impl<T> Display for SliceAccessor<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SliceAccessor({:#018x}, {})",
            self.address(),
            type_name::<T>()
        )
    }
}

// ------------------------------------------------------------------
// CellAccessor<'a, T>
// ------------------------------------------------------------------

/// Safe, `Copy`, `#![no_std]` read/write accessor backed by `&'a [Cell<T>]`.
///
/// Construct directly from any mutable slice via [`CellAccessor::from_slice_mut`] (using
/// [`Cell::from_mut`] and [`Cell::as_slice_of_cells`], with zero `unsafe`). Because
/// `CellAccessor` is `Copy`, multiple sub-tensor views can coexist and mutate shared storage
/// without borrow-checker conflicts.
pub struct CellAccessor<'a, T> {
    base: &'a [Cell<T>],
    offset: int,
}

impl<T> Clone for CellAccessor<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for CellAccessor<'_, T> {}

const impl<T> AccessorBase for CellAccessor<'_, T> {}

impl<'a, T> CellAccessor<'a, T> {
    /// Creates a read/write accessor viewing `slice` with zero `unsafe`.
    pub fn from_slice_mut(slice: &'a mut [T]) -> Self {
        Self {
            base: Cell::from_mut(slice).as_slice_of_cells(),
            offset: 0,
        }
    }

    /// Creates a read/write accessor directly over a slice of [`Cell<T>`].
    pub const fn from_cells(base: &'a [Cell<T>]) -> Self {
        Self { base, offset: 0 }
    }

    /// Returns the underlying cell slice.
    pub const fn base(&self) -> &'a [Cell<T>] {
        self.base
    }

    /// Returns `true` iff `self` and `other` share the same base slice pointer.
    pub fn same_base(&self, other: &Self) -> bool {
        core::ptr::eq(self.base.as_ptr(), other.base.as_ptr())
    }

    /// Returns the accumulated signed element offset from `base`.
    pub const fn offset_index(&self) -> int {
        self.offset
    }

    /// Returns the effective raw element pointer at the current offset.
    pub const fn as_ptr(&self) -> *const Cell<T> {
        self.base.as_ptr().wrapping_offset(self.offset as isize)
    }

    /// Returns the effective byte address of element `0` in this accessor.
    pub fn address(&self) -> usize {
        self.as_ptr() as usize
    }

    /// Writes `value` at `delta` through a shared reference using [`Cell::set`].
    pub fn set<Off: LinearOffset>(&self, delta: Off, value: T) {
        let idx = resolve_linear_index(self.offset, delta, self.base.len());
        self.base[idx].set(value);
    }
}

const impl<'a, T, I: [const] Integer> AccessorOffset<I> for CellAccessor<'a, T> {
    type Output = Self;

    fn offset(&self, delta: I) -> Self::Output {
        Self {
            base: self.base,
            offset: add(self.offset, delta.value()),
        }
    }
}

const impl<'a, T> AccessorOffset<Nil> for CellAccessor<'a, T> {
    type Output = Self;

    fn offset(&self, _: Nil) -> Self::Output {
        *self
    }
}

const impl<'a, T> AccessorOffset<BasisCombination<Nil>> for CellAccessor<'a, T> {
    type Output = Self;

    fn offset(&self, _: BasisCombination<Nil>) -> Self::Output {
        *self
    }
}

const impl<'a, T, V: [const] F2Scalar> AccessorOffset<F2<V>> for CellAccessor<'a, T> {
    type Output = F2OffsetAccessor<Self, F2<V>>;

    fn offset(&self, delta: F2<V>) -> Self::Output {
        F2OffsetAccessor::new(*self, delta)
    }
}

impl<'a, T: Copy, Off> Accessor<Off> for CellAccessor<'a, T>
where
    Off: LinearOffset,
    Self: AccessorOffset<Off>,
{
    type Item = T;

    fn load(&self, delta: Off) -> T {
        let idx = resolve_linear_index(self.offset, delta, self.base.len());
        self.base[idx].get()
    }
}

impl<'a, T: Copy, Off> MutableAccessor<Off> for CellAccessor<'a, T>
where
    Off: LinearOffset,
    Self: AccessorOffset<Off>,
{
    fn store(&mut self, delta: Off, value: T) {
        self.set(delta, value);
    }
}

const impl<'a, T, Off> Add<Off> for CellAccessor<'a, T>
where
    Self: [const] AccessorOffset<Off>,
{
    type Output = <Self as AccessorOffset<Off>>::Output;

    fn add(self, rhs: Off) -> Self::Output {
        self.offset(rhs)
    }
}

impl<'a, T> IntoLinearAccessor for &'a mut [T] {
    type Accessor = CellAccessor<'a, T>;

    fn into_linear_accessor(self) -> (Self::Accessor, int, usize) {
        let len = self.len();
        (CellAccessor::from_slice_mut(self), 0, len)
    }
}

impl<'a, T, const N: usize> IntoLinearAccessor for &'a mut [T; N] {
    type Accessor = CellAccessor<'a, T>;

    fn into_linear_accessor(self) -> (Self::Accessor, int, usize) {
        (CellAccessor::from_slice_mut(self.as_mut_slice()), 0, N)
    }
}

const impl<'a, T> IntoLinearAccessor for CellAccessor<'a, T> {
    type Accessor = Self;

    fn into_linear_accessor(self) -> (Self::Accessor, int, usize) {
        (self, self.offset, self.base.len())
    }
}

impl<T> PartialEq for CellAccessor<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self.as_ptr(), other.as_ptr())
    }
}

impl<T> Eq for CellAccessor<'_, T> {}

impl<T> Debug for CellAccessor<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CellAccessor({:#018x}, {})",
            self.address(),
            type_name::<T>()
        )
    }
}

impl<T> Display for CellAccessor<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

// ------------------------------------------------------------------
// Ptr<T>
// ------------------------------------------------------------------

/// Typed raw-pointer accessor into externally owned memory, mirroring PyCuTe's `Ptr`.
///
/// Tracks `(base, len, offset)` so chained offsets flatten without wrapping and every read
/// or write checks `0 <= offset + delta < len`.
pub struct Ptr<T> {
    base: *mut T,
    len: usize,
    offset: int,
}

impl<T> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Ptr<T> {}

const impl<T> AccessorBase for Ptr<T> {}

impl<T> Ptr<T> {
    /// Creates a [`Ptr`] viewing `len` contiguous elements starting at `base`.
    ///
    /// # Safety
    ///
    /// - `base` must be non-null (when `len > 0`), properly aligned for `T`, and valid for
    ///   reads and writes of `len` contiguous `T` elements for the lifetime of this `Ptr`
    ///   and any offset copies derived from it.
    /// - No live `&T` or `&mut T` references may conflict with writes performed through
    ///   this `Ptr` (aliasing multiple `Ptr<T>` handles from raw / `UnsafeCell` storage is
    ///   permitted because reads and writes use raw pointer operations).
    pub const unsafe fn from_raw(base: *mut T, len: usize) -> Self {
        Self {
            base,
            len,
            offset: 0,
        }
    }

    /// Returns the base raw pointer of the backing region.
    pub const fn base(&self) -> *mut T {
        self.base
    }

    /// Returns `true` iff `self` and `other` share the same base pointer.
    pub fn same_base(&self, other: &Self) -> bool {
        core::ptr::eq(self.base, other.base)
    }

    /// Returns the length (in elements) of the backing region.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` iff the backing region has zero elements.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the accumulated signed element offset from `base`.
    pub const fn offset_index(&self) -> int {
        self.offset
    }

    /// Returns the effective raw pointer at the current offset (`base.wrapping_offset(offset)`).
    pub const fn as_ptr(&self) -> *mut T {
        self.base.wrapping_offset(self.offset as isize)
    }

    /// Returns the effective byte address of element `0` in this pointer.
    pub fn address(&self) -> usize {
        self.as_ptr() as usize
    }
}

const impl<T, I: [const] Integer> AccessorOffset<I> for Ptr<T> {
    type Output = Self;

    fn offset(&self, delta: I) -> Self::Output {
        Self {
            base: self.base,
            len: self.len,
            offset: add(self.offset, delta.value()),
        }
    }
}

const impl<T> AccessorOffset<Nil> for Ptr<T> {
    type Output = Self;

    fn offset(&self, _: Nil) -> Self::Output {
        *self
    }
}

const impl<T> AccessorOffset<BasisCombination<Nil>> for Ptr<T> {
    type Output = Self;

    fn offset(&self, _: BasisCombination<Nil>) -> Self::Output {
        *self
    }
}

const impl<T, V: [const] F2Scalar> AccessorOffset<F2<V>> for Ptr<T> {
    type Output = F2OffsetAccessor<Self, F2<V>>;

    fn offset(&self, delta: F2<V>) -> Self::Output {
        F2OffsetAccessor::new(*self, delta)
    }
}

impl<T: Copy, Off> Accessor<Off> for Ptr<T>
where
    Off: LinearOffset,
    Self: AccessorOffset<Off>,
{
    type Item = T;

    fn load(&self, delta: Off) -> T {
        let idx = resolve_linear_index(self.offset, delta, self.len);
        // SAFETY: Caller of `Ptr::from_raw` guaranteed `[base, base + len)` is valid and aligned.
        unsafe { self.base.add(idx).read() }
    }
}

impl<T: Copy, Off> MutableAccessor<Off> for Ptr<T>
where
    Off: LinearOffset,
    Self: AccessorOffset<Off>,
{
    fn store(&mut self, delta: Off, value: T) {
        let idx = resolve_linear_index(self.offset, delta, self.len);
        // SAFETY: Caller of `Ptr::from_raw` guaranteed `[base, base + len)` is valid for writes.
        unsafe { self.base.add(idx).write(value) }
    }
}

const impl<T, Off> Add<Off> for Ptr<T>
where
    Self: [const] AccessorOffset<Off>,
{
    type Output = <Self as AccessorOffset<Off>>::Output;

    fn add(self, rhs: Off) -> Self::Output {
        self.offset(rhs)
    }
}

const impl<T> IntoLinearAccessor for Ptr<T> {
    type Accessor = Self;

    fn into_linear_accessor(self) -> (Self::Accessor, int, usize) {
        (self, self.offset, self.len)
    }
}

impl<T> PartialEq for Ptr<T> {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self.as_ptr(), other.as_ptr())
    }
}

impl<T> Eq for Ptr<T> {}

impl<T> Debug for Ptr<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Ptr({:#018x}, {})", self.address(), type_name::<T>())
    }
}

impl<T> Display for Ptr<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

// ------------------------------------------------------------------
// ImplicitAccessor<Origin>
// ------------------------------------------------------------------

/// Read-only accessor with no backing memory: reading `offset` returns `origin + offset`
/// via [`Accumulate`].
///
/// Used by [`identity_tensor`](super::identity_tensor) to compute global coordinates
/// without allocation, and by offset-inspection workflows.
#[derive(Clone, Copy, Default, Debug)]
#[repr(transparent)]
pub struct ImplicitAccessor<Origin = Int<0>> {
    /// Accumulated origin value added to every offset.
    pub origin: Origin,
}

const impl<Origin: Copy> AccessorBase for ImplicitAccessor<Origin> {}

impl<Origin> ImplicitAccessor<Origin> {
    /// Creates an [`ImplicitAccessor`] starting at `origin`.
    pub const fn new(origin: Origin) -> Self {
        Self { origin }
    }

    /// Returns a copy of the accessor's origin.
    pub const fn origin(&self) -> Origin
    where
        Origin: Copy,
    {
        self.origin
    }
}

const impl<Origin, Off> AccessorOffset<Off> for ImplicitAccessor<Origin>
where
    Origin: [const] Accumulate<Off>,
{
    type Output = ImplicitAccessor<Origin::Output>;

    fn offset(&self, offset: Off) -> Self::Output {
        ImplicitAccessor {
            origin: self.origin.accumulate(offset),
        }
    }
}

const impl<Origin, Off> Accessor<Off> for ImplicitAccessor<Origin>
where
    Origin: [const] Accumulate<Off>,
{
    type Item = Origin::Output;

    fn load(&self, offset: Off) -> Self::Item {
        self.origin.accumulate(offset)
    }
}

const impl<Origin, Off> Add<Off> for ImplicitAccessor<Origin>
where
    Origin: [const] Accumulate<Off>,
{
    type Output = ImplicitAccessor<Origin::Output>;

    fn add(self, rhs: Off) -> Self::Output {
        self.offset(rhs)
    }
}

const impl<O1, O2> PartialEq<ImplicitAccessor<O2>> for ImplicitAccessor<O1>
where
    O1: [const] IntoColexTerms,
    O2: [const] IntoColexTerms,
    O1::Terms: [const] MergeEq<O2::Terms>,
{
    fn eq(&self, other: &ImplicitAccessor<O2>) -> bool {
        basis_eq(self.origin, other.origin)
    }
}

const impl<O> Eq for ImplicitAccessor<O>
where
    O: [const] IntoColexTerms,
    O::Terms: [const] MergeEq<O::Terms>,
{
}

impl<Origin: Display> Display for ImplicitAccessor<Origin> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{{{}}}", self.origin)
    }
}

// ------------------------------------------------------------------
// TransformAccessor<A, F>
// ------------------------------------------------------------------

/// Read-only accessor wrapping an inner accessor `A` and applying callable `F` on every load:
/// `TransformAccessor::new(a, f).load(off) == f(a.load(off))`.
///
/// Offsetting propagates to `A` and leaves `F` untouched.
#[derive(Clone, Copy, Debug)]
pub struct TransformAccessor<A, F> {
    /// Underlying accessor.
    pub accessor: A,
    /// Post-read transformation function.
    pub transform: F,
}

const impl<A: AccessorBase, F: Copy> AccessorBase for TransformAccessor<A, F> {}

impl<A, F> TransformAccessor<A, F> {
    /// Wraps `accessor` with post-read `transform`.
    pub const fn new(accessor: A, transform: F) -> Self {
        Self {
            accessor,
            transform,
        }
    }
}

const impl<A, F, Off> AccessorOffset<Off> for TransformAccessor<A, F>
where
    A: [const] AccessorOffset<Off>,
    F: Copy,
{
    type Output = TransformAccessor<A::Output, F>;

    fn offset(&self, offset: Off) -> Self::Output {
        TransformAccessor {
            accessor: self.accessor.offset(offset),
            transform: self.transform,
        }
    }
}

const impl<A, F, Off> Accessor<Off> for TransformAccessor<A, F>
where
    A: [const] Accessor<Off>,
    F: Copy + [const] Fn<(A::Item,)>,
{
    type Item = <F as FnOnce<(A::Item,)>>::Output;

    fn load(&self, offset: Off) -> Self::Item {
        (self.transform)(self.accessor.load(offset))
    }
}

const impl<A, F, Off> Add<Off> for TransformAccessor<A, F>
where
    A: [const] AccessorOffset<Off>,
    F: Copy,
{
    type Output = TransformAccessor<A::Output, F>;

    fn add(self, rhs: Off) -> Self::Output {
        self.offset(rhs)
    }
}

const impl<A1, A2, F1, F2> PartialEq<TransformAccessor<A2, F2>> for TransformAccessor<A1, F1>
where
    A1: [const] PartialEq<A2>,
    F1: [const] PartialEq<F2>,
{
    fn eq(&self, other: &TransformAccessor<A2, F2>) -> bool {
        self.accessor == other.accessor && self.transform == other.transform
    }
}

const impl<A, F> Eq for TransformAccessor<A, F>
where
    A: [const] Eq,
    F: [const] Eq,
{
}

impl<A: Display, F: Debug> Display for TransformAccessor<A, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TransformAccessor({}, {:?})",
            self.accessor, self.transform
        )
    }
}

// ------------------------------------------------------------------
// OutOfBoundsAccessor<T, Fill, Origin>
// ------------------------------------------------------------------

/// Read-only accessor that indexes an underlying [`Tensor`] by coordinate and returns
/// `fill` whenever the accumulated coordinate falls outside the tensor's shape.
///
/// Designed for coordinate-codomain layouts (such as [`im2col_padded`](crate::algorithm::im2col::im2col_padded)):
/// offsetting accumulates into `origin` via [`Accumulate`] so slicing a padded view preserves
/// bounds checking without allocating padded storage.
///
/// # Examples
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::int_tuple;
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
/// use noodle_cute::tensor::{Accessor, OutOfBoundsAccessor, make_tensor};
///
/// let mut buf = [0.0f64; 4];
/// let mut t = make_tensor(
///     &mut buf,
///     Layout::new(int_tuple![Int::<2>, Int::<2>], int_tuple![Int::<2>, Int::<1>]),
/// );
/// t.set(int_tuple![1i32, 1i32], 5.0);
///
/// let oob = OutOfBoundsAccessor::new(t, -1.0);
/// assert_eq!(oob.load(int_tuple![1i32, 1i32]), 5.0);
/// assert_eq!(oob.load(int_tuple![2i32, 0i32]), -1.0);
/// assert_eq!(oob.load(int_tuple![-1i32, 0i32]), -1.0);
/// ```
///
/// `OutOfBoundsAccessor` is read-only; writing through a tensor backed by it fails at compile time:
///
/// ```compile_fail,E0277
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::Layout;
/// use noodle_cute::tensor::{OutOfBoundsAccessor, Tensor, make_tensor};
///
/// let mut buf = [0i32; 4];
/// let base = make_tensor(&mut buf, Int::<4>);
/// let mut padded = Tensor::new(OutOfBoundsAccessor::new(base, 0), Layout::from_shape(Int::<4>));
/// padded.set(0i32, 1);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct OutOfBoundsAccessor<T, Fill, Origin = Int<0>> {
    /// Backing tensor indexed when the accumulated coordinate is in bounds.
    pub tensor: T,
    /// Value returned for out-of-bounds coordinates.
    pub fill: Fill,
    /// Accumulated coordinate origin.
    pub origin: Origin,
}

const impl<T: Copy, Fill: Copy, Origin: Copy> AccessorBase
    for OutOfBoundsAccessor<T, Fill, Origin>
{
}

impl<T, Fill> OutOfBoundsAccessor<T, Fill, Int<0>> {
    /// Creates an [`OutOfBoundsAccessor`] rooted at `Int<0>` returning `fill` out of bounds.
    pub const fn new(tensor: T, fill: Fill) -> Self {
        Self {
            tensor,
            fill,
            origin: Int::<0>,
        }
    }
}

impl<T, Fill, Origin> OutOfBoundsAccessor<T, Fill, Origin> {
    /// Creates an [`OutOfBoundsAccessor`] with an explicit coordinate `origin`.
    pub const fn with_origin(tensor: T, fill: Fill, origin: Origin) -> Self {
        Self {
            tensor,
            fill,
            origin,
        }
    }

    /// Returns a reference to the backing tensor.
    pub const fn tensor(&self) -> &T {
        &self.tensor
    }

    /// Returns a copy of the out-of-bounds fill value.
    pub const fn fill(&self) -> Fill
    where
        Fill: Copy,
    {
        self.fill
    }

    /// Returns a copy of the accumulated coordinate origin.
    pub const fn origin(&self) -> Origin
    where
        Origin: Copy,
    {
        self.origin
    }
}

const impl<T, Fill, Origin, Off> AccessorOffset<Off> for OutOfBoundsAccessor<T, Fill, Origin>
where
    T: Copy,
    Fill: Copy,
    Off: [const] IntoBasisCoord,
    Origin: [const] Accumulate<Off::Output>,
{
    type Output = OutOfBoundsAccessor<T, Fill, Origin::Output>;

    fn offset(&self, offset: Off) -> Self::Output {
        OutOfBoundsAccessor {
            tensor: self.tensor,
            fill: self.fill,
            origin: self.origin.accumulate(into_basis_coord(offset)),
        }
    }
}

const impl<A, S, D, Fill, Origin, Off> Accessor<Off>
    for OutOfBoundsAccessor<Tensor<A, Layout<S, D>>, Fill, Origin>
where
    A: Copy,
    S: [const] Shape,
    D: [const] Stride<S>,
    Fill: Copy,
    Off: [const] IntoBasisCoord,
    Origin: [const] Accumulate<Off::Output>,
    Origin::Output: [const] InBounds<S>,
    Tensor<A, Layout<S, D>>: [const] TensorIndex<Origin::Output, Output = Fill>,
{
    type Item = Fill;

    fn load(&self, offset: Off) -> Self::Item {
        let crd = self.origin.accumulate(into_basis_coord(offset));
        if in_bounds(crd, &self.tensor.layout.shape) {
            self.tensor.at(crd)
        } else {
            self.fill
        }
    }
}

const impl<T, Fill, Origin, Off> Add<Off> for OutOfBoundsAccessor<T, Fill, Origin>
where
    Self: [const] AccessorOffset<Off>,
{
    type Output = <Self as AccessorOffset<Off>>::Output;

    fn add(self, rhs: Off) -> Self::Output {
        self.offset(rhs)
    }
}

const impl<T1, F1, O1, T2, F2, O2> PartialEq<OutOfBoundsAccessor<T2, F2, O2>>
    for OutOfBoundsAccessor<T1, F1, O1>
where
    T1: [const] PartialEq<T2>,
    F1: [const] PartialEq<F2>,
    O1: [const] IntoColexTerms,
    O2: [const] IntoColexTerms,
    O1::Terms: [const] MergeEq<O2::Terms>,
{
    fn eq(&self, other: &OutOfBoundsAccessor<T2, F2, O2>) -> bool {
        self.tensor == other.tensor
            && self.fill == other.fill
            && basis_eq(self.origin, other.origin)
    }
}

const impl<T, F, O> Eq for OutOfBoundsAccessor<T, F, O>
where
    T: [const] Eq,
    F: [const] Eq,
    O: [const] IntoColexTerms,
    O::Terms: [const] MergeEq<O::Terms>,
{
}

impl<T: Display, Fill: Debug, Origin: Display> Display for OutOfBoundsAccessor<T, Fill, Origin> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OutOfBoundsAccessor({}, fill={:?}, origin={})",
            self.tensor, self.fill, self.origin
        )
    }
}

// ------------------------------------------------------------------
// ArrayEngine<T, N>
// ------------------------------------------------------------------

/// Owned 16-byte-aligned stack storage for `N` elements: CuTe `ArrayEngine`.
///
/// Alignment to 16 bytes matches upstream `array_aligned`, keeping register fragments
/// naturally aligned for 128-bit vector loads/stores while remaining `#![no_std]` and
/// completely safe (zero `unsafe`).
///
/// Implements [`Accessor`] and [`MutableAccessor`] for direct complete-coordinate element
/// reads and writes, but intentionally does not implement [`AccessorOffset`] so slicing an
/// owning tensor requires borrowing a zero-copy view via [`Tensor::view`](super::Tensor::view)
/// or [`Tensor::view_mut`](super::Tensor::view_mut).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct ArrayEngine<T, const N: usize>(pub [T; N]);

impl<T: Copy + Default, const N: usize> Default for ArrayEngine<T, N> {
    fn default() -> Self {
        Self([T::default(); N])
    }
}

const impl<T: Copy, const N: usize> AccessorBase for ArrayEngine<T, N> {}

impl<T, const N: usize> ArrayEngine<T, N> {
    /// Wraps an existing fixed-size array `data`.
    pub const fn new(data: [T; N]) -> Self {
        Self(data)
    }

    /// Creates an engine filled with copies of `value`.
    pub const fn splat(value: T) -> Self
    where
        T: Copy,
    {
        Self([value; N])
    }

    /// Returns the compile-time element capacity `N`.
    pub const fn len(&self) -> usize {
        N
    }

    /// Returns `true` iff `N == 0`.
    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// Borrows the underlying array as an immutable slice.
    pub const fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// Borrows the underlying array as a mutable slice.
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }

    /// Creates a read-only [`SliceAccessor`] viewing this engine's storage.
    pub const fn slice_accessor(&self) -> SliceAccessor<'_, T> {
        SliceAccessor::new(&self.0)
    }

    /// Creates a safe, `Copy` read/write [`CellAccessor`] viewing this engine's storage.
    pub fn cell_accessor(&mut self) -> CellAccessor<'_, T> {
        CellAccessor::from_slice_mut(&mut self.0)
    }

    /// Borrows the element at `offset` with bounds checking.
    pub const fn get_ref<Off: [const] LinearOffset>(&self, offset: Off) -> &T {
        let idx = resolve_linear_index(0, offset, N);
        &self.0[idx]
    }

    /// Mutably borrows the element at `offset` with bounds checking.
    pub const fn get_mut<Off: [const] LinearOffset>(&mut self, offset: Off) -> &mut T {
        let idx = resolve_linear_index(0, offset, N);
        &mut self.0[idx]
    }
}

const impl<T: Copy, const N: usize, Off: [const] LinearOffset> Accessor<Off> for ArrayEngine<T, N> {
    type Item = T;

    fn load(&self, offset: Off) -> T {
        let idx = resolve_linear_index(0, offset, N);
        self.0[idx]
    }
}

const impl<T: Copy, const N: usize, Off: [const] LinearOffset> MutableAccessor<Off>
    for ArrayEngine<T, N>
{
    fn store(&mut self, offset: Off, value: T) {
        let idx = resolve_linear_index(0, offset, N);
        self.0[idx] = value;
    }
}

impl<T, const N: usize> Display for ArrayEngine<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "ArrayEngine<{}, {}>", type_name::<T>(), N)
    }
}
