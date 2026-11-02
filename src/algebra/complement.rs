//! Complement for static, mixed, and runtime layouts.
//!
//! Follows CuTe `layout.hpp` `filter`, `complement_prefix`, and `complement`. Typed entry
//! points change structure only from static facts; runtime entry points sort and simplify by
//! the current values.

use core::marker::ConstParamTy;
use core::ops::{Div, Mul};

use crate::int::{Integer, ceil_div, mul};
use crate::layout::{
    AffineLayoutMap, DynLayout, Layout, LayoutDesc, LayoutMap, LayoutParts, MAX_RANK, Shape,
    StrideOf,
};
use crate::tuple::{Append, Flatten, Item, Product, Unwrap};
use crate::{Bool, Error, Int, IntTuple, Nil, Result, True};

use super::coalesce::{CoalesceFold, CoalesceMode, FinishCoalesce, Prepend, coalesce_desc};
use super::composition::{ComposeProduct, StepNumber, ZERO_STRIDE};

/// Selects a cotarget complement or the complement's extended repetition mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum ComplementMode {
    /// Complements the supplied cotarget and coalesces the result.
    Standard,
    /// Keeps the terminal `1:repetition-stride` mode without a cotarget.
    Extended,
}

/// CuTe `filter(layout)` on a [`LayoutDesc`]: treats stride-0 leaves as size 1, then coalesces.
pub const fn filter_desc(desc: &LayoutDesc) -> Result<LayoutDesc> {
    let mut zeros = *desc;
    let mut a = 0;
    while a < zeros.rank {
        if zeros.strides[a] == 0 {
            zeros.shapes[a] = 1;
        }
        a += 1;
    }
    coalesce_desc(&zeros, CoalesceMode::Standard)
}

const fn complement_step(s: isize, d: isize, current: isize) -> Result<(isize, isize)> {
    if d < 0 {
        return Err(Error::Unsupported);
    }
    let gap = d / current;
    if gap == 0 {
        return Err(Error::NonInjective);
    }
    let next = match mul(s, d) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    Ok((gap, next))
}

struct PrefixResult {
    prefix: LayoutDesc,
    current: isize,
}

// Common prefix logic of CuTe `detail::complement`: filters unit modes, sorts stably by stride,
// and fills in address gaps one by one.
const fn complement_prefix_desc(desc: &LayoutDesc) -> Result<Option<PrefixResult>> {
    let size = match desc.size() {
        Ok(s) => s,
        Err(e) => return Err(e),
    };
    if size == 0 {
        return Err(Error::Unsupported);
    }
    let flat = match filter_desc(desc) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };
    if flat.rank == 1 && flat.strides[0] == 0 {
        return Ok(None);
    }
    let mut order = [0usize; MAX_RANK];
    let mut a = 0;
    while a < flat.rank {
        if flat.strides[a] < 0 {
            return Err(Error::Unsupported);
        }
        order[a] = a;
        a += 1;
    }
    // Stable insertion sort by (stride, original_index)
    let mut i = 1;
    while i < flat.rank {
        let key_idx = order[i];
        let key_stride = flat.strides[key_idx];
        let mut j = i;
        while j > 0 {
            let prev_idx = order[j - 1];
            let prev_stride = flat.strides[prev_idx];
            if prev_stride > key_stride || (prev_stride == key_stride && prev_idx > key_idx) {
                order[j] = order[j - 1];
                j -= 1;
            } else {
                break;
            }
        }
        order[j] = key_idx;
        i += 1;
    }

    let mut prefix = LayoutDesc::empty();
    let mut current = 1;
    let mut k = 0;
    while k < flat.rank {
        let idx = order[k];
        let s = flat.shapes[idx];
        let d = flat.strides[idx];
        let (gap, next) = match complement_step(s, d, current) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        if gap > 1 {
            prefix = match prefix.push(gap, current) {
                Ok(p) => p,
                Err(e) => return Err(e),
            };
        }
        current = next;
        k += 1;
    }

    Ok(Some(PrefixResult { prefix, current }))
}

/// CuTe complement on a [`LayoutDesc`].
pub const fn complement_desc(
    layout: &LayoutDesc,
    cotarget: &[isize],
    mode: ComplementMode,
) -> Result<LayoutDesc> {
    let mut idx = 0;
    while idx < cotarget.len() {
        if cotarget[idx] < 0 {
            return Err(Error::InvalidShape);
        }
        idx += 1;
    }

    let prefix_opt = match complement_prefix_desc(layout) {
        Ok(opt) => opt,
        Err(e) => return Err(e),
    };

    match prefix_opt {
        None => {
            if matches!(mode, ComplementMode::Extended) {
                let out = LayoutDesc::empty();
                out.push(1, 1)
            } else {
                let mut size = 1;
                let mut i = 0;
                while i < cotarget.len() {
                    size = match mul(size, cotarget[i]) {
                        Ok(s) => s,
                        Err(e) => return Err(e),
                    };
                    i += 1;
                }
                let out = LayoutDesc::empty();
                out.push(size, 1)
            }
        }
        Some(PrefixResult {
            mut prefix,
            current,
        }) => {
            if matches!(mode, ComplementMode::Extended) {
                let coalesced_prefix = match coalesce_desc(&prefix, CoalesceMode::Standard) {
                    Ok(p) => p,
                    Err(e) => return Err(e),
                };
                let prefix_size = match coalesced_prefix.size() {
                    Ok(s) => s,
                    Err(e) => return Err(e),
                };
                if prefix_size == 1 {
                    let out = LayoutDesc::empty();
                    return out.push(1, current);
                }
                let out = coalesced_prefix;
                out.push(1, current)
            } else {
                let mut rest = current;
                let mut stride = current;
                let mut prev_extent = 1;
                let mut i = 0;
                while i < cotarget.len() {
                    let a = cotarget[i];
                    let extent = ceil_div(a, rest);
                    rest = if a == 0 { rest } else { ceil_div(rest, a) };
                    if extent != 1 {
                        stride = match mul(stride, prev_extent) {
                            Ok(s) => s,
                            Err(e) => return Err(e),
                        };
                        prefix = match prefix.push(extent, stride) {
                            Ok(p) => p,
                            Err(e) => return Err(e),
                        };
                        prev_extent = extent;
                    }
                    i += 1;
                }
                coalesce_desc(&prefix, CoalesceMode::Standard)
            }
        }
    }
}

/// CuTe `complement(layout, cotarget)` with a tuple cotarget (a shape slice such as
/// `shape(coalesce(L))`).
///
/// The remaining extent is `ceil_div(cotarget, stride)` folded leaf by leaf, so for example
/// `complement(3:1, (4, 6)) = 12:3` rather than `8:3`.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::complement_shape;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[3], &[1])?;
/// let comp = complement_shape(layout, &[4, 6])?;
/// assert_eq!(comp.shapes(), &[12]);
/// assert_eq!(comp.strides(), &[3]);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// - Returns [`Error::InvalidShape`] if any cotarget extent is negative.
/// - Returns [`Error::Unsupported`] if `layout` has size 0 or negative strides.
/// - Returns [`Error::NonInjective`] if `layout` strides overlap.
/// - Returns [`Error::CapacityExceeded`] or [`Error::Overflow`] on capacity or arithmetic limits.
pub fn complement_shape<L: AffineLayoutMap>(layout: L, cotarget: &[isize]) -> Result<DynLayout> {
    let flat = layout.as_flat_layout()?;
    let desc = LayoutDesc::from_dyn_layout(&flat);
    let comp = complement_desc(&desc, cotarget, ComplementMode::Standard)?;
    Ok(comp.to_dyn_layout())
}

/// CuTe `complement(layout, cotarget)` with an integral cotarget.
///
/// Fills stride gaps and extends to `cotarget` (rounded up).
/// Consistent with CuTe: stride-0 leaves are filtered first; non-divisible stride gaps round down,
/// so the result does not in general prove injectivity or complete coverage of `(layout, R)`.
/// All-broadcast layouts return `cotarget:1`.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::complement;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[4], &[1])?;
/// let comp = complement(layout, 12)?;
/// assert_eq!(comp.shapes(), &[3]);
/// assert_eq!(comp.strides(), &[4]);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// - Returns [`Error::InvalidShape`] if `cotarget < 0`.
/// - Returns [`Error::Unsupported`] if `layout` has size 0 or negative strides.
/// - Returns [`Error::NonInjective`] if `layout` strides overlap.
/// - Returns [`Error::CapacityExceeded`] or [`Error::Overflow`] on capacity or arithmetic limits.
pub fn complement<L: AffineLayoutMap>(layout: L, cotarget: isize) -> Result<DynLayout> {
    complement_shape(layout, &[cotarget])
}

/// CuTe `complement(layout)` without a cotarget: returns the gaps of `layout` plus a
/// terminal `1:(s_last * d_last)` mode marking where the next repetition would start.
///
/// Unlike `complement(layout, cosize)`, this does not extend up to a cotarget; product operations
/// rely on the terminal mode to deduce repetition strides.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::complement_extended;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[2], &[2])?;
/// let comp = complement_extended(layout)?;
/// assert_eq!(comp.shapes(), &[2, 1]);
/// assert_eq!(comp.strides(), &[1, 4]);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// - Returns [`Error::Unsupported`] if `layout` has size 0 or negative strides.
/// - Returns [`Error::NonInjective`] if strides overlap.
/// - Returns [`Error::CapacityExceeded`] or [`Error::Overflow`] on capacity or arithmetic limits.
pub fn complement_extended<L: AffineLayoutMap>(layout: L) -> Result<DynLayout> {
    let flat = layout.as_flat_layout()?;
    let desc = LayoutDesc::from_dyn_layout(&flat);
    let comp = complement_desc(&desc, &[], ComplementMode::Extended)?;
    Ok(comp.to_dyn_layout())
}

/// Result type of typed complement; [`ComplementMode::Extended`] keeps the terminal repetition stride
/// (CuTe `complement(layout)` without a cotarget).
///
/// For fully static inputs, `<TypedComplement<L, C> as StaticValue>::VALUE` is the result as
/// a constant. Static empty domains, negative cotargets, negative strides, overlaps, and
/// arithmetic overflow are rejected at compile time. An empty tuple follows this crate's
/// scalar convention.
///
/// ```
/// use noodle_cute::{Int, IntTuple, Nil, StaticValue, algebra::{ComplementMode, TypedComplement}, layout::Layout};
/// type Pair<const A: isize, const B: isize> = IntTuple<Int<A>, IntTuple<Int<B>, Nil>>;
/// const C: Layout<Pair<3, 2>, Pair<2, 12>> =
///     <TypedComplement<Layout<Pair<2, 2>, Pair<1, 6>>, Int<24>> as StaticValue>::VALUE;
/// const E: Layout<Pair<2, 1>, Pair<1, 8>> =
///     <TypedComplement<Layout<Int<4>, Int<2>>, Nil, { ComplementMode::Extended }> as StaticValue>::VALUE;
/// ```
///
/// ```compile_fail
/// use noodle_cute::{DynInt, Int, StaticValue, algebra::TypedComplement, layout::Layout};
/// const C: TypedComplement<Layout<DynInt, Int<2>>, Int<24>> = StaticValue::VALUE;
/// ```
/// ```compile_fail
/// use noodle_cute::{Int, IntTuple, Nil, StaticValue, algebra::TypedComplement, layout::Layout};
/// type Pair<const A: isize, const B: isize> = IntTuple<Int<A>, IntTuple<Int<B>, Nil>>;
/// const C: TypedComplement<Layout<Pair<2, 2>, Pair<2, 3>>, Int<16>> = StaticValue::VALUE;
/// ```
/// ```compile_fail
/// use noodle_cute::{Int, StaticValue, algebra::TypedComplement, layout::Layout};
/// const C: TypedComplement<Layout<Int<0>, Int<1>>, Int<8>> = StaticValue::VALUE;
/// ```
/// ```compile_fail
/// use noodle_cute::{Int, StaticValue, algebra::TypedComplement, layout::Layout};
/// const C: TypedComplement<Layout<Int<2>, Int<-1>>, Int<8>> = StaticValue::VALUE;
/// ```
/// ```compile_fail
/// use noodle_cute::{Int, StaticValue, algebra::TypedComplement, layout::Layout};
/// const C: TypedComplement<Layout<Int<2>, Int<1>>, Int<-1>> = StaticValue::VALUE;
/// ```
/// ```compile_fail
/// use noodle_cute::{Int, Nil, StaticValue, algebra::{ComplementMode, TypedComplement}, layout::Layout};
/// const C: TypedComplement<Layout<Int<2>, Int<{ isize::MAX }>>, Nil, { ComplementMode::Extended }> = StaticValue::VALUE;
/// ```
pub type TypedComplement<L, C = Nil, const MODE: ComplementMode = { ComplementMode::Standard }> =
    <L as Complement<C, MODE>>::Output;

/// Checked complement shared by static and mixed layouts.
#[diagnostic::on_unimplemented(
    message = "layout `{Self}` has no typed complement for cotarget `{C}`",
    label = "unsupported typed complement",
    note = "typed complement requires a nonempty domain, supported strides, and a valid cotarget",
    note = "ComplementMode::Extended requires Nil instead of a cotarget"
)]
pub trait Complement<C, const MODE: ComplementMode = { ComplementMode::Standard }> {
    type Output: LayoutMap;
    fn complement_typed(self, cotarget: C) -> Result<Self::Output>;
}

impl<S: Shape, D: StrideOf<S>> Layout<S, D> {
    /// Complement keeping static leaves; the cotarget may be an integer or a nested shape.
    ///
    /// As in CuTe, a filtered multi-mode layout needs static strides; dynamic strides are
    /// supported only for a single mode. Dynamic unit shapes and zero strides do not change
    /// the type, and a dynamic zero stride returns `NonInjective`. An empty domain or a negative
    /// stride returns `Unsupported`, a negative cotarget `InvalidShape`, and overflow `Overflow`;
    /// statically known illegal inputs are rejected at compile time. The result is not limited
    /// by `MAX_RANK`.
    ///
    /// ```
    /// use noodle_cute::{DynInt, Int, IntTuple, Nil, layout::Layout};
    /// let a = Layout::new(DynInt(4), Int::<2>)?;
    /// let b: Layout<IntTuple<Int<2>, IntTuple<DynInt, Nil>>,
    ///               IntTuple<Int<1>, IntTuple<DynInt, Nil>>> = a.complement(Int::<24>)?;
    /// assert_eq!(b.shape.1.0, DynInt(3));
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    ///
    /// Multi-mode dynamic strides should go through `DynLayout` instead:
    /// ```compile_fail
    /// use noodle_cute::{DynInt, Int, shape, stride, layout::Layout};
    /// let a = Layout::new(shape![Int::<2>, Int::<2>], stride![DynInt(1), DynInt(6)]).unwrap();
    /// let _ = a.complement(Int::<24>);
    /// ```
    pub fn complement<C>(&self, cotarget: C) -> Result<TypedComplement<Self, C>>
    where
        Self: Complement<C>,
    {
        self.clone().complement_typed(cotarget)
    }

    /// Typed complement without a cotarget, keeping the terminal `1:repetition-stride` mode.
    ///
    /// Errors match [`Self::complement`]; an empty tuple returns `1:1` by the scalar convention.
    pub fn complement_extended(
        &self,
    ) -> Result<TypedComplement<Self, Nil, { ComplementMode::Extended }>>
    where
        Self: Complement<Nil, { ComplementMode::Extended }>,
    {
        self.clone().complement_typed(Nil)
    }
}

type Filtered<S, D> = <Layout<<S as Flatten>::Output, <D as Flatten>::Output> as Filter>::Output;
type Ordered<S, D> = <Filtered<S, D> as Sort>::Output;
type Built<S, D, C, const MODE: ComplementMode> = <Ordered<S, D> as Build<C, MODE>>::Output;

pub const NONEMPTY<S: Product>: bool = !matches!(S::Output::STATIC, Some(0))
where
    S::Output: Integer;

impl<S, D, C, const MODE: ComplementMode> Complement<C, MODE> for Layout<S, D>
where
    S: Shape + Flatten + Product,
    <S as Product>::Output: Integer,
    Bool<{ NONEMPTY::<S> }>: True,
    D: StrideOf<S> + Flatten,
    Layout<<S as Flatten>::Output, D::Output>: Filter,
    Filtered<S, D>: Sort,
    Ordered<S, D>: Build<C, MODE>,
    <Built<S, D, C, MODE> as LayoutParts>::Shape: Shape,
    <Built<S, D, C, MODE> as LayoutParts>::Stride:
        Item<Profile = <<Built<S, D, C, MODE> as LayoutParts>::Shape as Item>::Profile>,
{
    type Output = Layout<
        <Built<S, D, C, MODE> as LayoutParts>::Shape,
        <Built<S, D, C, MODE> as LayoutParts>::Stride,
    >;
    fn complement_typed(self, cotarget: C) -> Result<Self::Output> {
        if self.size()? == 0 {
            return Err(Error::Unsupported);
        }
        let result = self
            .flatten()
            .filter()
            .sort()
            .build(cotarget)?
            .into_layout();
        Layout::new(result.shape, result.stride)
    }
}

pub trait FilterLeaf<S> {
    type Output;
    fn filter_leaf(s: S) -> Self::Output;
}
impl<S> FilterLeaf<S> for Bool<true> {
    type Output = Int<1>;
    fn filter_leaf(_: S) -> Int<1> {
        Int
    }
}
impl<S> FilterLeaf<S> for Bool<false> {
    type Output = S;
    fn filter_leaf(s: S) -> S {
        s
    }
}

pub trait Filter {
    type Output: LayoutParts;
    fn filter(self) -> Self::Output;
}
impl Filter for Layout<Nil, Nil> {
    type Output = Self;
    fn filter(self) -> Self {
        self
    }
}
impl<S, ST, D: Integer, DT> Filter for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    Bool<{ ZERO_STRIDE::<D> }>: FilterLeaf<S>,
    Layout<ST, DT>: Filter,
    <Layout<ST, DT> as Filter>::Output:
        Prepend<<Bool<{ ZERO_STRIDE::<D> }> as FilterLeaf<S>>::Output, D>,
    <<Layout<ST, DT> as Filter>::Output as Prepend<
        <Bool<{ ZERO_STRIDE::<D> }> as FilterLeaf<S>>::Output,
        D,
    >>::Output: LayoutParts,
{
    type Output = <<Layout<ST, DT> as Filter>::Output as Prepend<
        <Bool<{ ZERO_STRIDE::<D> }> as FilterLeaf<S>>::Output,
        D,
    >>::Output;
    fn filter(self) -> Self::Output {
        Layout {
            shape: self.shape.1,
            stride: self.stride.1,
        }
        .filter()
        .prepend(
            <Bool<{ ZERO_STRIDE::<D> }> as FilterLeaf<S>>::filter_leaf(self.shape.0),
            self.stride.0,
        )
    }
}

pub trait Sort {
    type Output: LayoutParts;
    fn sort(self) -> Self::Output;
}
impl Sort for Layout<Nil, Nil> {
    type Output = Self;
    fn sort(self) -> Self {
        self
    }
}
impl<S, D> Sort for Layout<IntTuple<S, Nil>, IntTuple<D, Nil>> {
    type Output = Self;
    fn sort(self) -> Self {
        self
    }
}
impl<S, SH, ST, D, DH, DT> Sort
    for Layout<IntTuple<S, IntTuple<SH, ST>>, IntTuple<D, IntTuple<DH, DT>>>
where
    Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>: Sort,
    <Layout<IntTuple<SH, ST>, IntTuple<DH, DT>> as Sort>::Output: Insert<S, D>,
{
    type Output =
        <<Layout<IntTuple<SH, ST>, IntTuple<DH, DT>> as Sort>::Output as Insert<S, D>>::Output;
    fn sort(self) -> Self::Output {
        Layout {
            shape: self.shape.1,
            stride: self.stride.1,
        }
        .sort()
        .insert(self.shape.0, self.stride.0)
    }
}

pub const BEFORE<D: Integer, E: Integer>: bool = match (D::STATIC, E::STATIC) {
    (Some(d), Some(e)) => d <= e,
    _ => panic!("Dynamic-stride complement only for rank-1 layouts"),
};

pub trait Insert<S, D> {
    type Output: LayoutParts;
    fn insert(self, s: S, d: D) -> Self::Output;
}
impl<S, D> Insert<S, D> for Layout<Nil, Nil> {
    type Output = Layout<IntTuple<S, Nil>, IntTuple<D, Nil>>;
    fn insert(self, s: S, d: D) -> Self::Output {
        Layout {
            shape: IntTuple(s, Nil),
            stride: IntTuple(d, Nil),
        }
    }
}
impl<S, D: Integer, SH, ST, DH: Integer, DT> Insert<S, D>
    for Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>
where
    Bool<{ BEFORE::<D, DH> }>: InsertAt<S, D, SH, ST, DH, DT>,
{
    type Output = <Bool<{ BEFORE::<D, DH> }> as InsertAt<S, D, SH, ST, DH, DT>>::Output;
    fn insert(self, s: S, d: D) -> Self::Output {
        <Bool<{ BEFORE::<D, DH> }> as InsertAt<S, D, SH, ST, DH, DT>>::insert_at(self, s, d)
    }
}

pub trait InsertAt<S, D, SH, ST, DH, DT> {
    type Output: LayoutParts;
    fn insert_at(tail: Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>, s: S, d: D) -> Self::Output;
}
impl<S, D, SH, ST, DH, DT> InsertAt<S, D, SH, ST, DH, DT> for Bool<true> {
    type Output = Layout<IntTuple<S, IntTuple<SH, ST>>, IntTuple<D, IntTuple<DH, DT>>>;
    fn insert_at(tail: Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>, s: S, d: D) -> Self::Output {
        Layout {
            shape: IntTuple(s, tail.shape),
            stride: IntTuple(d, tail.stride),
        }
    }
}
impl<S, D, SH, ST, DH, DT> InsertAt<S, D, SH, ST, DH, DT> for Bool<false>
where
    Layout<ST, DT>: Insert<S, D>,
{
    type Output = Layout<
        IntTuple<SH, <<Layout<ST, DT> as Insert<S, D>>::Output as LayoutParts>::Shape>,
        IntTuple<DH, <<Layout<ST, DT> as Insert<S, D>>::Output as LayoutParts>::Stride>,
    >;
    fn insert_at(tail: Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>, s: S, d: D) -> Self::Output {
        let rest = Layout {
            shape: tail.shape.1,
            stride: tail.stride.1,
        }
        .insert(s, d)
        .into_layout();
        Layout {
            shape: IntTuple(tail.shape.0, rest.shape),
            stride: IntTuple(tail.stride.0, rest.stride),
        }
    }
}

pub const VALID_STEP<S: Integer, D: Integer, C: Integer>: bool = !matches!(S::STATIC, Some(s) if s <= 0)
    && !matches!(D::STATIC, Some(d) if d <= 0)
    && !matches!((D::STATIC, C::STATIC), (Some(d), Some(c)) if c <= 0 || d / c == 0);

pub trait Prefix<C> {
    type Output: LayoutParts;
    type End: Integer;
    fn prefix(self, current: C) -> Result<(Self::Output, Self::End)>;
}
impl<C: Integer> Prefix<C> for Layout<Nil, Nil> {
    type Output = Self;
    type End = C;
    fn prefix(self, current: C) -> Result<(Self, C)> {
        Ok((self, current))
    }
}
impl<S: Integer + Mul<D>, ST, D: Integer + Div<C>, DT, C: Integer> Prefix<C>
    for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    S::Output: Integer,
    D::Output: Integer,
    Bool<{ VALID_STEP::<S, D, C> }>: True,
    Layout<ST, DT>: Prefix<S::Output>,
{
    type Output = Layout<
        IntTuple<D::Output, <<Layout<ST, DT> as Prefix<S::Output>>::Output as LayoutParts>::Shape>,
        IntTuple<C, <<Layout<ST, DT> as Prefix<S::Output>>::Output as LayoutParts>::Stride>,
    >;
    type End = <Layout<ST, DT> as Prefix<S::Output>>::End;
    fn prefix(self, current: C) -> Result<(Self::Output, Self::End)> {
        let (s, d) = (self.shape.0, self.stride.0);
        complement_step(s.value(), d.value(), current.value())?;
        let (tail, end) = Layout {
            shape: self.shape.1,
            stride: self.stride.1,
        }
        .prefix(s * d)?;
        let tail = tail.into_layout();
        Ok((
            Layout {
                shape: IntTuple(d / current, tail.shape),
                stride: IntTuple(current, tail.stride),
            },
            end,
        ))
    }
}

// The reverse ceil_div by a zero cotarget is undefined in C++; like the runtime path, keep `rest`.
const fn target_ceil(a: isize, b: isize) -> isize {
    assert!(a >= 0 && b >= 0, "Invalid complement cotarget");
    if b == 0 { a } else { ceil_div(a, b) }
}
pub const CEIL_STATIC<A: Integer, B: Integer>: bool = A::STATIC.is_some() && B::STATIC.is_some();
pub const CEIL<A: Integer, B: Integer>: isize = match (A::STATIC, B::STATIC) {
    (Some(a), Some(b)) => target_ceil(a, b),
    _ => 0,
};
pub trait Ceil<B> {
    type Output: Integer;
    fn ceil(self, b: B) -> Self::Output;
}
impl<A: Integer, B: Integer> Ceil<B> for A
where
    Bool<{ CEIL_STATIC::<A, B> }>: StepNumber<{ CEIL::<A, B> }>,
{
    type Output = <Bool<{ CEIL_STATIC::<A, B> }> as StepNumber<{ CEIL::<A, B> }>>::Output;
    fn ceil(self, b: B) -> Self::Output {
        <Bool<{ CEIL_STATIC::<A, B> }> as StepNumber<{ CEIL::<A, B> }>>::make(target_ceil(
            self.value(),
            b.value(),
        ))
    }
}
pub trait CeilTarget<C> {
    type Output;
    fn ceil_target(self, current: C) -> Self::Output;
}
impl<C> CeilTarget<C> for Nil {
    type Output = Nil;
    fn ceil_target(self, _: C) -> Nil {
        Nil
    }
}
impl<A: Integer + Ceil<C>, T, C: Integer + Ceil<A>> CeilTarget<C> for IntTuple<A, T>
where
    T: CeilTarget<<C as Ceil<A>>::Output>,
{
    type Output =
        IntTuple<<A as Ceil<C>>::Output, <T as CeilTarget<<C as Ceil<A>>::Output>>::Output>;
    fn ceil_target(self, current: C) -> Self::Output {
        IntTuple(
            self.0.ceil(current),
            self.1.ceil_target(current.ceil(self.0)),
        )
    }
}

pub trait GroupShape {
    type Output;
    fn group_shape(self) -> Result<Self::Output>;
}
impl GroupShape for Nil {
    type Output = IntTuple<Int<1>, Nil>;
    fn group_shape(self) -> Result<Self::Output> {
        Ok(IntTuple(Int, Nil))
    }
}
impl<S> GroupShape for IntTuple<S, Nil> {
    type Output = Self;
    fn group_shape(self) -> Result<Self> {
        Ok(self)
    }
}
pub const SAME_KIND<A: Integer, B: Integer>: bool = A::STATIC.is_some() == B::STATIC.is_some();
impl<S: Integer, H: Integer, T> GroupShape for IntTuple<S, IntTuple<H, T>>
where
    Bool<{ SAME_KIND::<S, H> }>: GroupAt<S, H, T>,
{
    type Output = <Bool<{ SAME_KIND::<S, H> }> as GroupAt<S, H, T>>::Output;
    fn group_shape(self) -> Result<Self::Output> {
        <Bool<{ SAME_KIND::<S, H> }> as GroupAt<S, H, T>>::group_at(self.1, self.0)
    }
}
pub trait GroupAt<S, H, T> {
    type Output;
    fn group_at(tail: IntTuple<H, T>, s: S) -> Result<Self::Output>;
}
impl<S: Integer + Mul<H>, H: Integer, T> GroupAt<S, H, T> for Bool<true>
where
    IntTuple<S::Output, T>: GroupShape,
{
    type Output = <IntTuple<S::Output, T> as GroupShape>::Output;
    fn group_at(tail: IntTuple<H, T>, s: S) -> Result<Self::Output> {
        mul(s.value(), tail.0.value())?;
        IntTuple(s * tail.0, tail.1).group_shape()
    }
}
impl<S, H, T> GroupAt<S, H, T> for Bool<false>
where
    IntTuple<H, T>: GroupShape,
{
    type Output = IntTuple<S, <IntTuple<H, T> as GroupShape>::Output>;
    fn group_at(tail: IntTuple<H, T>, s: S) -> Result<Self::Output> {
        Ok(IntTuple(s, tail.group_shape()?))
    }
}

pub trait TargetStrides<C> {
    type Output: LayoutParts;
    fn target_strides(self, current: C) -> Result<Self::Output>;
}
// `Compact` also computes the unused trailing size*stride; computing only the needed strides
// avoids unrelated overflow.
impl<S, C> TargetStrides<C> for IntTuple<S, Nil> {
    type Output = Layout<Self, IntTuple<C, Nil>>;
    fn target_strides(self, current: C) -> Result<Self::Output> {
        Ok(Layout {
            shape: self,
            stride: IntTuple(current, Nil),
        })
    }
}
type StrideProduct<S, C> = <() as ComposeProduct<S, C>>::Output;

impl<S: Integer, H, T, C: Integer> TargetStrides<C> for IntTuple<S, IntTuple<H, T>>
where
    (): ComposeProduct<S, C>,
    IntTuple<H, T>: TargetStrides<StrideProduct<S, C>>,
{
    type Output = Layout<
        IntTuple<
            S,
            <<IntTuple<H, T> as TargetStrides<StrideProduct<S, C>>>::Output as LayoutParts>::Shape,
        >,
        IntTuple<
            C,
            <<IntTuple<H, T> as TargetStrides<StrideProduct<S, C>>>::Output as LayoutParts>::Stride,
        >,
    >;
    fn target_strides(self, current: C) -> Result<Self::Output> {
        let next = <() as ComposeProduct<S, C>>::product(self.0, current)?;
        let tail = self.1.target_strides(next)?.into_layout();
        Ok(Layout {
            shape: IntTuple(self.0, tail.shape),
            stride: IntTuple(current, tail.stride),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum TargetMode {
    Divide,
    Preserve,
}

pub trait Target<C, const MODE: TargetMode = { TargetMode::Divide }> {
    type Output: LayoutParts;
    fn target(self, current: C) -> Result<Self::Output>;
}
impl<T: Shape + Flatten, C: Integer> Target<C> for T
where
    T::Output: CeilTarget<C>,
    <T::Output as CeilTarget<C>>::Output: GroupShape,
    <<T::Output as CeilTarget<C>>::Output as GroupShape>::Output: TargetStrides<C>,
{
    type Output =
        <<<T::Output as CeilTarget<C>>::Output as GroupShape>::Output as TargetStrides<C>>::Output;
    fn target(self, current: C) -> Result<Self::Output> {
        for i in 0..T::LEAVES {
            if self.leaf(i)? < 0 {
                return Err(Error::InvalidShape);
            }
        }
        self.flatten()
            .ceil_target(current)
            .group_shape()?
            .target_strides(current)
    }
}
impl<T: Shape + Flatten, C: Integer> Target<C, { TargetMode::Preserve }> for T
where
    T::Output: GroupShape,
    <T::Output as GroupShape>::Output: TargetStrides<C>,
{
    type Output = <<T::Output as GroupShape>::Output as TargetStrides<C>>::Output;
    fn target(self, current: C) -> Result<Self::Output> {
        for i in 0..T::LEAVES {
            if self.leaf(i)? < 0 {
                return Err(Error::InvalidShape);
            }
        }
        self.flatten().group_shape()?.target_strides(current)
    }
}

pub trait Complete<C, End, const MODE: ComplementMode> {
    type Output: LayoutParts;
    fn complete(self, cotarget: C, end: End) -> Result<Self::Output>;
}

// The shared coalesce keeps a one-leaf tuple; CuTe's complement result needs an integral profile.
pub trait UnwrapLayout {
    type Output: LayoutParts;
    fn unwrap_layout(self) -> Self::Output;
}
impl<S: Integer, D: Integer> UnwrapLayout for Layout<S, D> {
    type Output = Self;
    fn unwrap_layout(self) -> Self {
        self
    }
}
impl<S, ST, D, DT> UnwrapLayout for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    IntTuple<S, ST>: Unwrap,
    IntTuple<D, DT>: Unwrap,
{
    type Output = Layout<<IntTuple<S, ST> as Unwrap>::Output, <IntTuple<D, DT> as Unwrap>::Output>;
    fn unwrap_layout(self) -> Self::Output {
        Layout {
            shape: self.shape.unwrap(),
            stride: self.stride.unwrap(),
        }
    }
}

impl<S, D, C: Target<End>, End> Complete<C, End, { ComplementMode::Standard }> for Layout<S, D>
where
    S: Append<<C::Output as LayoutParts>::Shape>,
    D: Append<<C::Output as LayoutParts>::Stride>,
    Layout<S::Output, D::Output>: CoalesceFold,
    <Layout<S::Output, D::Output> as CoalesceFold>::Output: FinishCoalesce,
    <<Layout<S::Output, D::Output> as CoalesceFold>::Output as FinishCoalesce>::Output:
        UnwrapLayout,
{
    type Output = <<<Layout<S::Output, D::Output> as CoalesceFold>::Output as FinishCoalesce>::Output as UnwrapLayout>::Output;
    fn complete(self, cotarget: C, end: End) -> Result<Self::Output> {
        let rest = cotarget.target(end)?.into_layout();
        Layout {
            shape: self.shape.append(rest.shape),
            stride: self.stride.append(rest.stride),
        }
        .fold()
        .finish()
        .map(UnwrapLayout::unwrap_layout)
    }
}
impl<End> Complete<Nil, End, { ComplementMode::Extended }> for Layout<Nil, Nil> {
    type Output = Layout<Int<1>, End>;
    fn complete(self, _: Nil, end: End) -> Result<Self::Output> {
        Ok(Layout {
            shape: Int,
            stride: end,
        })
    }
}
impl<S, ST, D, DT, End> Complete<Nil, End, { ComplementMode::Extended }>
    for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    IntTuple<S, ST>: Append<IntTuple<Int<1>, Nil>>,
    IntTuple<D, DT>: Append<IntTuple<End, Nil>>,
{
    type Output = Layout<
        <IntTuple<S, ST> as Append<IntTuple<Int<1>, Nil>>>::Output,
        <IntTuple<D, DT> as Append<IntTuple<End, Nil>>>::Output,
    >;
    fn complete(self, _: Nil, end: End) -> Result<Self::Output> {
        Ok(Layout {
            shape: self.shape.append(IntTuple(Int, Nil)),
            stride: self.stride.append(IntTuple(end, Nil)),
        })
    }
}

pub trait Build<C, const MODE: ComplementMode> {
    type Output: LayoutParts;
    fn build(self, cotarget: C) -> Result<Self::Output>;
}
impl<C: Target<Int<1>, { TargetMode::Preserve }>> Build<C, { ComplementMode::Standard }>
    for Layout<Nil, Nil>
where
    C::Output: UnwrapLayout,
{
    type Output = <C::Output as UnwrapLayout>::Output;
    fn build(self, cotarget: C) -> Result<Self::Output> {
        cotarget.target(Int).map(UnwrapLayout::unwrap_layout)
    }
}
impl Build<Nil, { ComplementMode::Extended }> for Layout<Nil, Nil> {
    type Output = Layout<Int<1>, Int<1>>;
    fn build(self, _: Nil) -> Result<Self::Output> {
        Ok(Layout {
            shape: Int,
            stride: Int,
        })
    }
}
impl<S, ST, D, DT, C> Build<C, { ComplementMode::Standard }>
    for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    Self: Prefix<Int<1>>,
    <Self as Prefix<Int<1>>>::Output:
        Complete<C, <Self as Prefix<Int<1>>>::End, { ComplementMode::Standard }>,
{
    type Output = <<Self as Prefix<Int<1>>>::Output as Complete<
        C,
        <Self as Prefix<Int<1>>>::End,
        { ComplementMode::Standard },
    >>::Output;
    fn build(self, cotarget: C) -> Result<Self::Output> {
        let (prefix, end) = self.prefix(Int)?;
        prefix.complete(cotarget, end)
    }
}
impl<S, ST, D, DT> Build<Nil, { ComplementMode::Extended }>
    for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    Self: Prefix<Int<1>>,
    <Self as Prefix<Int<1>>>::Output: CoalesceFold,
    <<Self as Prefix<Int<1>>>::Output as CoalesceFold>::Output:
        Complete<Nil, <Self as Prefix<Int<1>>>::End, { ComplementMode::Extended }>,
{
    type Output = <<<Self as Prefix<Int<1>>>::Output as CoalesceFold>::Output as Complete<
        Nil,
        <Self as Prefix<Int<1>>>::End,
        { ComplementMode::Extended },
    >>::Output;
    fn build(self, _: Nil) -> Result<Self::Output> {
        let (prefix, end) = self.prefix(Int)?;
        prefix.fold().complete(Nil, end)
    }
}
