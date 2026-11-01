//! Complement for static, mixed, and runtime layouts.
//!
//! Follows CuTe `layout.hpp` `filter`, `complement_prefix`, and `complement`. Typed entry
//! points change structure only from static facts; runtime entry points sort and simplify by
//! the current values.

use core::ops::{Div, Mul};

use crate::int::{Integer, ceil_div, checked_product, mul};
use crate::layout::{
    AffineLayoutMap, DynLayout, Layout, LayoutBuilder, LayoutMap, LayoutParts, MAX_RANK, Shape,
};
use crate::tuple::{Append, Flatten, Item, Product, Unwrap};
use crate::{Bool, Error, Int, IntTuple, Nil, Result, True};

use super::coalesce::{CoalesceFold, FinishCoalesce, Prepend};
use super::composition::{ComposeProduct, StepNumber, ZERO_STRIDE};

fn complement_step(s: isize, d: isize, current: isize) -> Result<(isize, isize)> {
    if d < 0 {
        return Err(Error::Unsupported);
    }
    let gap = d / current;
    if gap == 0 {
        return Err(Error::NonInjective);
    }
    Ok((gap, mul(s, d)?))
}

// Common prefix logic of CuTe `detail::complement`: filters unit modes, sorts stably by stride,
// and fills in address gaps one by one.
// Returns the prefix leaves of the complement layout along with the next stride; returns `None`
// when only `1:0` remains after filtering (CuTe special case).
fn complement_prefix<L: AffineLayoutMap>(layout: &L) -> Result<Option<(LayoutBuilder, isize)>> {
    let flat = layout.as_flat_layout()?;
    // A zero extent makes CuTe's next stride zero, and its cotarget ceil_div then divides by it.
    if flat.size()? == 0 {
        return Err(Error::Unsupported);
    }
    let flat = flat.filter()?;
    if flat.rank() == 1 && flat.leaf_stride(0)? == 0 {
        return Ok(None);
    }
    let mut order = [0usize; MAX_RANK];
    for (a, slot) in order.iter_mut().enumerate().take(flat.rank()) {
        if flat.leaf_stride(a)? < 0 {
            return Err(Error::Unsupported);
        }
        *slot = a;
    }
    // Unstable sort by key (stride, index) is equivalent to CuTe's stable sort without allocations.
    order[..flat.rank()].sort_unstable_by_key(|&a| (flat.strides()[a], a));
    let mut prefix = LayoutBuilder::new();
    let mut current = 1;
    for &a in &order[..flat.rank()] {
        let (s, d) = (flat.leaf_shape(a)?, flat.leaf_stride(a)?);
        // Floor: non-divisible gaps round down as in CuTe; the complement may overlap with the original layout at holes.
        let (gap, next) = complement_step(s, d, current)?;
        // Gaps of length 1 are eliminated by final coalesce and do not consume leaf capacity here.
        if gap > 1 {
            prefix.push(gap, current)?;
        }
        current = next;
    }

    Ok(Some((prefix, current)))
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
    if cotarget.iter().any(|&c| c < 0) {
        return Err(Error::InvalidShape);
    }
    let Some((mut leaves, current)) = complement_prefix(&layout)? else {
        // CuTe: complement(1:0, cotarget) = make_layout(coalesce(cotarget))
        let size = checked_product(cotarget.iter().map(|&c| Ok(c)))?;
        return DynLayout::new(&[size], &[1]);
    };
    // CuTe tuple ceil_div: result_i = ceil(a_i / rest), rest = ceil(rest / a_i)
    let mut rest = current;
    let mut stride = current;
    let mut prev_extent = 1;
    for &a in cotarget {
        let extent = ceil_div(a, rest);
        rest = if a == 0 { rest } else { ceil_div(rest, a) };
        if extent != 1 {
            stride = mul(stride, prev_extent)?;
            leaves.push(extent, stride)?;
            prev_extent = extent;
        }
    }
    // Coalesces to 1:0 when all extents are 1, matching CuTe.
    leaves.finish_flat()?.coalesce()
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
    let Some((leaves, current)) = complement_prefix(&layout)? else {
        return DynLayout::new(&[1], &[1]);
    };
    let prefix = leaves.finish_flat()?.coalesce()?;
    if prefix.size()? == 1 {
        return DynLayout::new(&[1], &[current]);
    }
    let mut out = LayoutBuilder::new();
    out.push_leaves(&prefix)?;
    out.push(1, current)?;
    out.finish_flat()
}

/// Result type of the typed complement; `EXTENDED` keeps the terminal repetition stride
/// (CuTe `complement(layout)` without a cotarget).
///
/// For fully static inputs, `<TypedComplement<L, C> as StaticValue>::VALUE` is the result as
/// a constant. Static empty domains, negative cotargets, negative strides, overlaps, and
/// arithmetic overflow are rejected at compile time. An empty tuple follows this crate's
/// scalar convention.
///
/// ```
/// use noodle_cute::{Int, IntTuple, Nil, StaticValue, algebra::TypedComplement, layout::Layout};
/// type Pair<const A: isize, const B: isize> = IntTuple<Int<A>, IntTuple<Int<B>, Nil>>;
/// const C: Layout<Pair<3, 2>, Pair<2, 12>> =
///     <TypedComplement<Layout<Pair<2, 2>, Pair<1, 6>>, Int<24>> as StaticValue>::VALUE;
/// const E: Layout<Pair<2, 1>, Pair<1, 8>> =
///     <TypedComplement<Layout<Int<4>, Int<2>>, Nil, true> as StaticValue>::VALUE;
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
/// use noodle_cute::{Int, Nil, StaticValue, algebra::TypedComplement, layout::Layout};
/// const C: TypedComplement<Layout<Int<2>, Int<{ isize::MAX }>>, Nil, true> = StaticValue::VALUE;
/// ```
pub type TypedComplement<L, C = Nil, const EXTENDED: bool = false> =
    <L as Complement<C, EXTENDED>>::Output;

/// Checked complement shared by static and mixed layouts.
pub trait Complement<C, const EXTENDED: bool = false> {
    type Output: LayoutMap;
    fn complement_typed(self, cotarget: C) -> Result<Self::Output>;
}

impl<S: Shape, D: Item<Profile = S::Profile>> Layout<S, D> {
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
    pub fn complement_extended(&self) -> Result<TypedComplement<Self, Nil, true>>
    where
        Self: Complement<Nil, true>,
    {
        self.clone().complement_typed(Nil)
    }
}

type Filtered<S, D> = <Layout<<S as Flatten>::Output, <D as Flatten>::Output> as Filter>::Output;
type Ordered<S, D> = <Filtered<S, D> as Sort>::Output;
type Built<S, D, C, const EXTENDED: bool> = <Ordered<S, D> as Build<C, EXTENDED>>::Output;

pub const NONEMPTY<S: Product>: bool = !matches!(S::Output::STATIC, Some(0))
where
    S::Output: Integer;

impl<S, D, C, const EXTENDED: bool> Complement<C, EXTENDED> for Layout<S, D>
where
    S: Shape + Flatten + Product,
    <S as Product>::Output: Integer,
    Bool<{ NONEMPTY::<S> }>: True,
    D: Item<Profile = S::Profile> + Flatten,
    Layout<<S as Flatten>::Output, D::Output>: Filter,
    Filtered<S, D>: Sort,
    Ordered<S, D>: Build<C, EXTENDED>,
    <Built<S, D, C, EXTENDED> as LayoutParts>::Shape: Shape,
    <Built<S, D, C, EXTENDED> as LayoutParts>::Stride:
        Item<Profile = <<Built<S, D, C, EXTENDED> as LayoutParts>::Shape as Item>::Profile>,
{
    type Output = Layout<
        <Built<S, D, C, EXTENDED> as LayoutParts>::Shape,
        <Built<S, D, C, EXTENDED> as LayoutParts>::Stride,
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

pub trait Target<C, const DIVIDE: bool = true> {
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
impl<T: Shape + Flatten, C: Integer> Target<C, false> for T
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

pub trait Complete<C, End, const EXTENDED: bool> {
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

impl<S, D, C: Target<End>, End> Complete<C, End, false> for Layout<S, D>
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
impl<End> Complete<Nil, End, true> for Layout<Nil, Nil> {
    type Output = Layout<Int<1>, End>;
    fn complete(self, _: Nil, end: End) -> Result<Self::Output> {
        Ok(Layout {
            shape: Int,
            stride: end,
        })
    }
}
impl<S, ST, D, DT, End> Complete<Nil, End, true> for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
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

pub trait Build<C, const EXTENDED: bool> {
    type Output: LayoutParts;
    fn build(self, cotarget: C) -> Result<Self::Output>;
}
impl<C: Target<Int<1>, false>> Build<C, false> for Layout<Nil, Nil>
where
    C::Output: UnwrapLayout,
{
    type Output = <C::Output as UnwrapLayout>::Output;
    fn build(self, cotarget: C) -> Result<Self::Output> {
        cotarget.target(Int).map(UnwrapLayout::unwrap_layout)
    }
}
impl Build<Nil, true> for Layout<Nil, Nil> {
    type Output = Layout<Int<1>, Int<1>>;
    fn build(self, _: Nil) -> Result<Self::Output> {
        Ok(Layout {
            shape: Int,
            stride: Int,
        })
    }
}
impl<S, ST, D, DT, C> Build<C, false> for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    Self: Prefix<Int<1>>,
    <Self as Prefix<Int<1>>>::Output: Complete<C, <Self as Prefix<Int<1>>>::End, false>,
{
    type Output = <<Self as Prefix<Int<1>>>::Output as Complete<
        C,
        <Self as Prefix<Int<1>>>::End,
        false,
    >>::Output;
    fn build(self, cotarget: C) -> Result<Self::Output> {
        let (prefix, end) = self.prefix(Int)?;
        prefix.complete(cotarget, end)
    }
}
impl<S, ST, D, DT> Build<Nil, true> for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    Self: Prefix<Int<1>>,
    <Self as Prefix<Int<1>>>::Output: CoalesceFold,
    <<Self as Prefix<Int<1>>>::Output as CoalesceFold>::Output:
        Complete<Nil, <Self as Prefix<Int<1>>>::End, true>,
{
    type Output = <<<Self as Prefix<Int<1>>>::Output as CoalesceFold>::Output as Complete<
        Nil,
        <Self as Prefix<Int<1>>>::End,
        true,
    >>::Output;
    fn build(self, _: Nil) -> Result<Self::Output> {
        let (prefix, end) = self.prefix(Int)?;
        prefix.fold().complete(Nil, end)
    }
}
