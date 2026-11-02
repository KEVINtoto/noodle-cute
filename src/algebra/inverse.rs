//! Structural inverses for static, mixed, and runtime layouts, plus enumerated runtime inverses.
//!
//! The typed right inverse ignores dynamic strides and extends only the statically known
//! stride chain; the typed left inverse requires all strides to be static after coalescing.
//! The runtime free functions use the current values without these type constraints, and
//! `dense_*_inverse` enumerate any `LayoutMap`.
//!
//! ```
//! use noodle_cute::{Int, DynInt, layout::{Layout, LayoutMap}};
//! let layout = Layout::new(DynInt(4), Int::<1>)?;
//! let inverse: Layout<DynInt, Int<1>> = layout.right_inverse()?;
//! assert_eq!(inverse.index(3)?, 3);
//! # Ok::<(), noodle_cute::Error>(())
//! ```
//! ```
//! use noodle_cute::{Int, StaticValue, algebra::{InverseSide, TypedInverse}, layout::Layout};
//! // CuTe: left_inverse(_0:_1) = _0:_1.
//! const R: Layout<Int<0>, Int<1>> = <TypedInverse<Layout<Int<0>, Int<1>>, { InverseSide::Left }> as StaticValue>::VALUE;
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, IntTuple, Nil, StaticValue, algebra::{InverseSide, TypedInverse}, layout::Layout};
//! type Pair<const A: isize, const B: isize> = IntTuple<Int<A>, IntTuple<Int<B>, Nil>>;
//! const R: TypedInverse<Layout<Pair<2, 2>, Pair<2, 3>>, { InverseSide::Left }> = StaticValue::VALUE;
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, StaticValue, algebra::{InverseSide, TypedInverse}, layout::Layout};
//! const R: TypedInverse<Layout<Int<2>, Int<-1>>, { InverseSide::Left }> = StaticValue::VALUE;
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, StaticValue, algebra::{InverseSide, TypedInverse}, layout::Layout};
//! const R: TypedInverse<Layout<Int<2>, Int<{ isize::MAX }>>, { InverseSide::Left }> = StaticValue::VALUE;
//! ```

use core::marker::ConstParamTy;
use core::ops::Div;

use crate::int::{Integer, add, mul};
use crate::layout::{
    AffineLayoutMap, DynLayout, GroupedLayout, Layout, LayoutBuilder, LayoutMap, LayoutParts,
    MAX_RANK, Shape, StrideOf, affine_injective,
};
use crate::tuple::Flatten;
use crate::{Bool, Error, Int, IntTuple, Nil, Result, StaticValue, True};

use super::coalesce::{CoalesceFold, FinishCoalesce};
use super::complement::{Sort, UnwrapLayout};
use super::composition::{ComposeProduct, ZERO_STRIDE};
use super::typed::Checked;

// Common preparation for CuTe inversion: coalesced flat leaves, prefix products
// of leaf shapes, and indices sorted stably by stride.
struct SortedLeaves {
    flat: DynLayout,
    preprod: [isize; MAX_RANK],
    order: [usize; MAX_RANK],
}

/// Prepares the leaves of an affine layout for inversion.
/// Returns the coalesced flat layout, the prefix product
/// of each leaf's shape, and the indices sorted by stride.
fn sorted_leaves<L: AffineLayoutMap>(layout: &L) -> Result<SortedLeaves> {
    let flat = layout.as_flat_layout()?.coalesce()?;
    let mut preprod = [0; MAX_RANK];
    let mut order = [0; MAX_RANK];
    let mut acc = 1;
    for a in 0..flat.rank() {
        preprod[a] = acc;
        acc = mul(acc, flat.leaf_shape(a)?)?;
        order[a] = a;
    }
    order[..flat.rank()].sort_unstable_by_key(|&a| (flat.strides()[a], a));
    Ok(SortedLeaves {
        flat,
        preprod,
        order,
    })
}

/// CuTe `right_inverse(layout)`: returns a layout `R` such that `layout(R(i)) == i` for `i < size(R)`.
///
/// Advances only along modes whose stride equals the accumulated address span. Other modes (broadcasts, negative strides, or modes after
/// a gap) are ignored, so for non-dense layouts `R` covers only the longest contiguous address prefix starting
/// at 0 (e.g. `right_inverse(4:2) = 1:0`).
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::right_inverse;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[2, 3], &[1, 2])?;
/// let inv = right_inverse(layout)?;
/// assert_eq!(inv.shapes(), &[6]);
/// assert_eq!(inv.strides(), &[1]);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// - Returns [`Error::CapacityExceeded`] if leaf capacity is exceeded.
/// - Returns [`Error::Overflow`] if stride arithmetic overflows.
pub fn right_inverse<L: AffineLayoutMap>(layout: L) -> Result<DynLayout> {
    let sorted = sorted_leaves(&layout)?;
    let flat = &sorted.flat;
    let mut leaves = LayoutBuilder::new();
    let mut current = 1;
    for &a in &sorted.order[..flat.rank()] {
        let (s, d) = (flat.leaf_shape(a)?, flat.leaf_stride(a)?);
        if d == current {
            leaves.push(s, sorted.preprod[a])?;
            current = mul(s, d)?;
        }
    }
    leaves.finish_flat()?.coalesce()
}

/// CuTe `left_inverse(layout)`: returns a layout `R` such that `R(layout(i)) == i` for `i < size(layout)`.
///
/// Address holes are also defined (unconstrained values), e.g. `left_inverse(4:2) = (2, 4):(0, 1)`.
/// Stride-0 modes are ignored, so broadcast layouts yield one of their right inverses.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::left_inverse;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[2, 3], &[1, 2])?;
/// let inv = left_inverse(layout)?;
/// assert_eq!(inv.shapes(), &[6]);
/// assert_eq!(inv.strides(), &[1]);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// - Returns [`Error::Unsupported`] if `layout` contains negative strides.
/// - Returns [`Error::NotDivisible`] if a stride is not divisible by the covered size.
/// - Returns [`Error::NonInjective`] if non-broadcast leaves overlap (`stride < prev_shape * prev_stride`).
/// - Returns [`Error::CapacityExceeded`] or [`Error::Overflow`] on capacity or arithmetic limits.
pub fn left_inverse<L: AffineLayoutMap>(layout: L) -> Result<DynLayout> {
    let sorted = sorted_leaves(&layout)?;
    let flat = &sorted.flat;
    let rank = flat.rank();
    // The resulting shape has one fewer leading entry than stride: strides begin with 0, followed by the shape of the last mode.
    let mut shape = [0; MAX_RANK + 1];
    let mut stride = [0; MAX_RANK + 1];
    let mut len = 0;
    let mut covered = 1;
    let mut min_stride = 1;
    for &a in &sorted.order[..rank] {
        let d = flat.leaf_stride(a)?;
        if d == 0 {
            continue;
        }
        if d < 0 {
            return Err(Error::Unsupported);
        }
        if d % covered != 0 {
            return Err(Error::NotDivisible);
        }
        if d < min_stride {
            return Err(Error::NonInjective);
        }
        shape[len] = d / covered;
        stride[len + 1] = sorted.preprod[a];
        covered = mul(covered, shape[len])?;
        min_stride = mul(flat.leaf_shape(a)?, d)?;
        len += 1;
    }
    shape[len] = flat.leaf_shape(sorted.order[rank - 1])?;
    let mut leaves = LayoutBuilder::new();
    // Without leading holes (min stride 1), the initial 1:0 is eliminated by coalesce; skip it to prevent overflow on full-capacity layouts.
    let start = usize::from(len > 0 && shape[0] == 1);
    for i in start..=len {
        leaves.push(shape[i], stride[i])?;
    }
    leaves.finish_flat()?.coalesce()
}

/// An enumerated inverse mapping: logical coordinates represent original addresses,
/// and output is the original layout's colexicographic (left-major) linear index.
///
/// Lookups linearly scan from index 0 upwards (`O(size)`), selecting the smallest preimage in case of duplicates.
#[derive(Clone, Debug)]
pub struct Inverse<L> {
    layout: L,
    size: isize,
}

impl<L> Inverse<L> {
    /// Returns a reference to the underlying layout.
    pub fn layout(&self) -> &L {
        &self.layout
    }
}

impl<L: LayoutMap> LayoutMap for Inverse<L> {
    fn size(&self) -> Result<isize> {
        Ok(self.size)
    }
    fn rank(&self) -> usize {
        1
    }
    fn extent(&self, a: usize) -> Result<isize> {
        if a == 0 {
            Ok(self.size)
        } else {
            Err(Error::AxisOutOfBounds)
        }
    }
    fn index(&self, address: isize) -> Result<isize> {
        if address < 0 || address >= self.size {
            return Err(Error::CoordinateOutOfBounds);
        }
        for i in 0..self.layout.size()? {
            if self.layout.index(i)? == address {
                return Ok(i);
            }
        }
        Err(Error::NotSurjective)
    }
    fn is_injective(&self) -> Result<bool> {
        Ok(true)
    }

    fn as_affine(&self) -> Result<Option<GroupedLayout>> {
        Ok(None)
    }
}

/// Exact left inverse of a dense bijection onto `[0, size)`, for maps without
/// affine metadata (e.g. swizzled layouts). Sparse codomains are rejected
/// instead of being defined in the holes as CuTe's `left_inverse` does.
///
/// The original address set must exactly cover `[0, size)`. Empty layouts are supported.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::dense_left_inverse;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[3], &[1])?;
/// let inv = dense_left_inverse(layout)?;
/// assert_eq!(inv.size()?, 3);
/// assert_eq!(inv.index(2)?, 2);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// - Returns [`Error::NonInjective`] if `layout` is not injective.
/// - Returns [`Error::NotSurjective`] if the address range does not match `[0, size)`.
pub fn dense_left_inverse<L: LayoutMap>(layout: L) -> Result<Inverse<L>> {
    if !layout.is_injective()? {
        return Err(Error::NonInjective);
    }
    let size = layout.size()?;
    if size > 0 && layout.address_range()? != Some((0, size - 1)) {
        return Err(Error::NotSurjective);
    }
    Ok(Inverse { layout, size })
}

/// Exact right inverse of a dense surjection onto `[0, max]`. Broadcast maps
/// are supported: the smallest logical preimage is selected for each address.
///
/// Unlike CuTe's `right_inverse`, sparse codomains return [`Error::NotSurjective`] rather than truncating
/// to a contiguous prefix. Dense affine bijections are verified using the stride-separation proof;
/// other maps linearly scan the domain for each reachable address (`O(cosize * size)`).
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::dense_right_inverse;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let layout = DynLayout::new(&[3], &[0])?;
/// let inv = dense_right_inverse(layout)?;
/// assert_eq!(inv.size()?, 1);
/// assert_eq!(inv.index(0)?, 0);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// - Returns [`Error::NotSurjective`] if any address in `[0, max]` is not covered by `layout`.
/// - Returns [`Error::Overflow`] if address calculation overflows.
pub fn dense_right_inverse<L: LayoutMap>(layout: L) -> Result<Inverse<L>> {
    let domain_size = layout.size()?;
    let size = match layout.address_range()? {
        None => 0,
        Some((0, max)) => add(max, 1)?,
        Some(_) => return Err(Error::NotSurjective),
    };
    if size > domain_size {
        return Err(Error::NotSurjective);
    }
    // When the codomain and domain have equal size, any injective map onto [0, size) is a bijection.
    let proven_bijection = match layout.as_affine() {
        Ok(Some(a)) if size == domain_size => {
            affine_injective(&a, 0..a.leaf_count())? == Some(true)
        }
        Ok(_) | Err(Error::CapacityExceeded) => false,
        Err(e) => return Err(e),
    };
    if !proven_bijection {
        for address in 0..size {
            let mut found = false;
            for i in 0..domain_size {
                if layout.index(i)? == address {
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(Error::NotSurjective);
            }
        }
    }
    Ok(Inverse { layout, size })
}

/// Selects the direction of a typed structural inverse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum InverseSide {
    /// Builds the statically known right-inverse stride chain.
    Right,
    /// Builds the left quasi-inverse, requiring static strides after coalescing.
    Left,
}

/// Typed structural inverse selected by [`InverseSide`].
#[diagnostic::on_unimplemented(
    message = "layout `{Self}` does not support typed inversion on side {SIDE}",
    label = "unsupported typed inverse",
    note = "typed left inversion requires static strides after coalescing; check the selected inverse's stride and overflow requirements"
)]
pub trait TypedInvert<const SIDE: InverseSide = { InverseSide::Right }> {
    type Output: LayoutMap;
    fn invert(self) -> Result<Self::Output>;
}
/// Result type of the typed structural inverse (right inverse by default), keeping static
/// leaves.
///
/// For fully static inputs, `<TypedInverse<L, SIDE> as StaticValue>::VALUE` is the result as a
/// constant. The right inverse ignores broadcasts, negative strides, and gaps in the address
/// chain; the left inverse rejects static negative strides, overlaps, and indivisible strides
/// at compile time and treats broadcast modes as CuTe's quasi-inverse.
///
/// ```
/// use noodle_cute::{Int, IntTuple, Nil, StaticValue, algebra::{InverseSide, TypedInverse}, layout::Layout};
/// type Pair<const A: isize, const B: isize> = IntTuple<Int<A>, IntTuple<Int<B>, Nil>>;
/// const R: Layout<Int<6>, Int<1>> =
///     <TypedInverse<Layout<Pair<2, 3>, Pair<1, 2>>> as StaticValue>::VALUE;
/// const L: Layout<Pair<2, 4>, Pair<0, 1>> =
///     <TypedInverse<Layout<Int<4>, Int<2>>, { InverseSide::Left }> as StaticValue>::VALUE;
/// ```
/// ```compile_fail
/// use noodle_cute::{DynInt, Int, StaticValue, algebra::TypedInverse, layout::Layout};
/// const R: TypedInverse<Layout<DynInt, Int<1>>> = StaticValue::VALUE;
/// ```
/// ```compile_fail
/// use noodle_cute::{DynInt, Int, StaticValue, algebra::{InverseSide, TypedInverse}, layout::Layout};
/// const L: TypedInverse<Layout<Int<4>, DynInt>, { InverseSide::Left }> = StaticValue::VALUE;
/// ```
pub type TypedInverse<L, const SIDE: InverseSide = { InverseSide::Right }> =
    <L as TypedInvert<SIDE>>::Output;

impl<S: Shape, D: StrideOf<S>> Layout<S, D> {
    /// Typed right inverse that uses only static strides and extends only statically matched
    /// address chains.
    ///
    /// A dynamic shape may become a result leaf, but stride equalities that hold only at runtime
    /// do not change the structure. Runtime product overflow returns `Overflow`; the result is
    /// not limited by `MAX_RANK`.
    pub fn right_inverse(&self) -> Result<TypedInverse<Self>>
    where
        Self: TypedInvert,
    {
        self.clone().invert()
    }

    /// Typed left inverse; every stride must be static after coalescing.
    ///
    /// Overlap or overflow caused by dynamic shapes returns `NonInjective` or `Overflow`; address
    /// gaps keep CuTe's quasi-inverse. For dynamic strides, use the runtime [`left_inverse`] on
    /// `as_flat_layout()`.
    pub fn left_inverse(&self) -> Result<TypedInverse<Self, { InverseSide::Left }>>
    where
        Self: TypedInvert<{ InverseSide::Left }>,
    {
        self.clone().invert()
    }
}
type Flat<S, D> = <Layout<<S as Flatten>::Output, <D as Flatten>::Output> as CoalesceFold>::Output;
type Inverted<S, D, const SIDE: InverseSide> = <Flat<S, D> as InverseKernel<SIDE>>::Output;
impl<S, D, const SIDE: InverseSide> TypedInvert<SIDE> for Layout<S, D>
where
    S: Shape + Flatten,
    D: StrideOf<S> + Flatten,
    Layout<<S as Flatten>::Output, D::Output>: CoalesceFold,
    Flat<S, D>: InverseKernel<SIDE>,
    Inverted<S, D, SIDE>: Checked,
{
    type Output = <Inverted<S, D, SIDE> as Checked>::Output;
    fn invert(self) -> Result<Self::Output> {
        self.size()?;
        self.flatten().fold().inverse_kernel()?.checked()
    }
}

pub trait FinishInverse {
    type Output: LayoutParts;
    fn finish_inverse(self) -> Result<Self::Output>;
}
impl<L: CoalesceFold> FinishInverse for L
where
    L::Output: FinishCoalesce,
    <L::Output as FinishCoalesce>::Output: UnwrapLayout,
{
    type Output = <<L::Output as FinishCoalesce>::Output as UnwrapLayout>::Output;
    fn finish_inverse(self) -> Result<Self::Output> {
        self.fold().finish().map(UnwrapLayout::unwrap_layout)
    }
}

pub trait Attach<P> {
    type Output: LayoutParts;
    fn attach(self, prefix: P) -> Result<Self::Output>;
}
impl<P> Attach<P> for Layout<Nil, Nil> {
    type Output = Self;
    fn attach(self, _: P) -> Result<Self> {
        Ok(self)
    }
}
type NumberProduct<A, B> = <() as ComposeProduct<A, B>>::Output;
impl<S: Integer, ST, D, DT, P: Integer> Attach<P> for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    (): ComposeProduct<S, P>,
    Layout<ST, DT>: Attach<NumberProduct<S, P>>,
{
    type Output = Layout<
        IntTuple<
            (S, P),
            <<Layout<ST, DT> as Attach<NumberProduct<S, P>>>::Output as LayoutParts>::Shape,
        >,
        IntTuple<
            D,
            <<Layout<ST, DT> as Attach<NumberProduct<S, P>>>::Output as LayoutParts>::Stride,
        >,
    >;
    fn attach(self, prefix: P) -> Result<Self::Output> {
        let next = <() as ComposeProduct<S, P>>::product(self.shape.0, prefix)?;
        let tail = Layout {
            shape: self.shape.1,
            stride: self.stride.1,
        }
        .attach(next)?
        .into_layout();
        Ok(Layout {
            shape: IntTuple((self.shape.0, prefix), tail.shape),
            stride: IntTuple(self.stride.0, tail.stride),
        })
    }
}
pub trait SelectStrides<const SIDE: InverseSide> {
    type Output: LayoutParts;
    fn select_strides(self) -> Self::Output;
}
impl<S, D: StaticValue> SelectStrides<{ InverseSide::Left }> for Layout<S, D> {
    type Output = Self;
    fn select_strides(self) -> Self {
        self
    }
}
impl SelectStrides<{ InverseSide::Right }> for Layout<Nil, Nil> {
    type Output = Self;
    fn select_strides(self) -> Self {
        self
    }
}
pub const STATIC_STRIDE<D: Integer>: bool = D::STATIC.is_some();
impl<S, ST, D: Integer, DT> SelectStrides<{ InverseSide::Right }>
    for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    Layout<ST, DT>: SelectStrides<{ InverseSide::Right }>,
    Bool<{ STATIC_STRIDE::<D> }>:
        KeepStride<S, D, <Layout<ST, DT> as SelectStrides<{ InverseSide::Right }>>::Output>,
{
    type Output = <Bool<{ STATIC_STRIDE::<D> }> as KeepStride<
        S,
        D,
        <Layout<ST, DT> as SelectStrides<{ InverseSide::Right }>>::Output,
    >>::Output;
    fn select_strides(self) -> Self::Output {
        let tail = Layout {
            shape: self.shape.1,
            stride: self.stride.1,
        }
        .select_strides();
        <Bool<{ STATIC_STRIDE::<D> }> as KeepStride<_, _, _>>::keep(
            self.shape.0,
            self.stride.0,
            tail,
        )
    }
}
pub trait KeepStride<S, D, Tail> {
    type Output: LayoutParts;
    fn keep(s: S, d: D, tail: Tail) -> Self::Output;
}
impl<S, D, Tail: LayoutParts> KeepStride<S, D, Tail> for Bool<false> {
    type Output = Tail;
    fn keep(_: S, _: D, tail: Tail) -> Tail {
        tail
    }
}
impl<S, D, Tail: LayoutParts> KeepStride<S, D, Tail> for Bool<true> {
    type Output = Layout<IntTuple<S, Tail::Shape>, IntTuple<D, Tail::Stride>>;
    fn keep(s: S, d: D, tail: Tail) -> Self::Output {
        let tail = tail.into_layout();
        Layout {
            shape: IntTuple(s, tail.shape),
            stride: IntTuple(d, tail.stride),
        }
    }
}

pub const MATCHES<D: Integer, C: Integer>: bool =
    matches!((D::STATIC, C::STATIC), (Some(d), Some(c)) if d == c);
pub trait RightFold<C> {
    type Output: LayoutParts;
    fn right_fold(self, current: C) -> Result<Self::Output>;
}
impl<C> RightFold<C> for Layout<Nil, Nil> {
    type Output = Self;
    fn right_fold(self, _: C) -> Result<Self> {
        Ok(self)
    }
}
impl<S, P, ST, D: Integer, DT, C: Integer> RightFold<C>
    for Layout<IntTuple<(S, P), ST>, IntTuple<D, DT>>
where
    Bool<{ MATCHES::<D, C> }>: RightStep<S, P, D, Layout<ST, DT>, C>,
{
    type Output = <Bool<{ MATCHES::<D, C> }> as RightStep<S, P, D, Layout<ST, DT>, C>>::Output;
    fn right_fold(self, current: C) -> Result<Self::Output> {
        <Bool<{ MATCHES::<D, C> }> as RightStep<_, _, _, _, _>>::run(
            self.shape.0.0,
            self.shape.0.1,
            self.stride.0,
            Layout {
                shape: self.shape.1,
                stride: self.stride.1,
            },
            current,
        )
    }
}
pub trait RightStep<S, P, D, Tail, C> {
    type Output: LayoutParts;
    fn run(s: S, p: P, d: D, tail: Tail, current: C) -> Result<Self::Output>;
}
impl<S, P, D, Tail: RightFold<C>, C> RightStep<S, P, D, Tail, C> for Bool<false> {
    type Output = Tail::Output;
    fn run(_: S, _: P, _: D, tail: Tail, c: C) -> Result<Self::Output> {
        tail.right_fold(c)
    }
}
impl<S: Integer, P, D: Integer, Tail, C> RightStep<S, P, D, Tail, C> for Bool<true>
where
    (): ComposeProduct<S, D>,
    Tail: RightFold<NumberProduct<S, D>>,
{
    type Output = Layout<
        IntTuple<S, <Tail::Output as LayoutParts>::Shape>,
        IntTuple<P, <Tail::Output as LayoutParts>::Stride>,
    >;
    fn run(s: S, p: P, d: D, tail: Tail, _: C) -> Result<Self::Output> {
        let next = <() as ComposeProduct<S, D>>::product(s, d)?;
        let tail = tail.right_fold(next)?.into_layout();
        Ok(Layout {
            shape: IntTuple(s, tail.shape),
            stride: IntTuple(p, tail.stride),
        })
    }
}

pub const LEFT_VALID<D: Integer, C: Integer, M: Integer>: bool = match (D::STATIC, C::STATIC) {
    (Some(d), Some(c)) => {
        d > 0
            && c > 0
            && d % c == 0
            && match M::STATIC {
                Some(m) => d >= m,
                None => true,
            }
    }
    _ => false,
};
pub trait LeftFold<C, M, P, Last> {
    type Output: LayoutParts;
    fn left_fold(self, covered: C, minimum: M, previous: P, last: Last) -> Result<Self::Output>;
}
impl<C, M, P, Last> LeftFold<C, M, P, Last> for Layout<Nil, Nil> {
    type Output = Layout<IntTuple<Last, Nil>, IntTuple<P, Nil>>;
    fn left_fold(self, _: C, _: M, p: P, last: Last) -> Result<Self::Output> {
        Ok(Layout {
            shape: IntTuple(last, Nil),
            stride: IntTuple(p, Nil),
        })
    }
}
impl<S, PP, ST, D: Integer, DT, C, M, P, Last> LeftFold<C, M, P, Last>
    for Layout<IntTuple<(S, PP), ST>, IntTuple<D, DT>>
where
    Bool<{ ZERO_STRIDE::<D> }>: LeftStep<S, PP, D, Layout<ST, DT>, C, M, P>,
{
    type Output =
        <Bool<{ ZERO_STRIDE::<D> }> as LeftStep<S, PP, D, Layout<ST, DT>, C, M, P>>::Output;
    fn left_fold(self, c: C, m: M, p: P, _: Last) -> Result<Self::Output> {
        <Bool<{ ZERO_STRIDE::<D> }> as LeftStep<_, _, _, _, _, _, _>>::run(
            self.shape.0.0,
            self.shape.0.1,
            self.stride.0,
            Layout {
                shape: self.shape.1,
                stride: self.stride.1,
            },
            c,
            m,
            p,
        )
    }
}
pub trait LeftStep<S, PP, D, Tail, C, M, P> {
    type Output: LayoutParts;
    fn run(
        s: S,
        pp: PP,
        d: D,
        tail: Tail,
        covered: C,
        minimum: M,
        previous: P,
    ) -> Result<Self::Output>;
}
impl<S, PP, D, Tail, C, M, P> LeftStep<S, PP, D, Tail, C, M, P> for Bool<true>
where
    Tail: LeftFold<C, M, P, S>,
{
    type Output = Tail::Output;
    fn run(s: S, _: PP, _: D, tail: Tail, c: C, m: M, p: P) -> Result<Self::Output> {
        tail.left_fold(c, m, p, s)
    }
}
impl<S: Integer, PP, D: Integer + Div<C>, Tail, C: Integer, M: Integer, P>
    LeftStep<S, PP, D, Tail, C, M, P> for Bool<false>
where
    Bool<{ LEFT_VALID::<D, C, M> }>: True,
    (): ComposeProduct<S, D>,
    Tail: LeftFold<D, NumberProduct<S, D>, PP, S>,
{
    type Output = Layout<
        IntTuple<D::Output, <Tail::Output as LayoutParts>::Shape>,
        IntTuple<P, <Tail::Output as LayoutParts>::Stride>,
    >;
    fn run(s: S, pp: PP, d: D, tail: Tail, c: C, m: M, p: P) -> Result<Self::Output> {
        if d.value() < 0 {
            return Err(Error::Unsupported);
        }
        if d.value() % c.value() != 0 {
            return Err(Error::NotDivisible);
        }
        if d.value() < m.value() {
            return Err(Error::NonInjective);
        }
        let minimum = <() as ComposeProduct<S, D>>::product(s, d)?;
        let tail = tail.left_fold(d, minimum, pp, s)?.into_layout();
        Ok(Layout {
            shape: IntTuple(d / c, tail.shape),
            stride: IntTuple(p, tail.stride),
        })
    }
}

pub trait InverseKernel<const SIDE: InverseSide> {
    type Output: LayoutParts;
    fn inverse_kernel(self) -> Result<Self::Output>;
}
type Selected<L, const SIDE: InverseSide> =
    <<L as Attach<Int<1>>>::Output as SelectStrides<SIDE>>::Output;
type Ordered<L, const SIDE: InverseSide> = <Selected<L, SIDE> as Sort>::Output;
impl<L: Attach<Int<1>>> InverseKernel<{ InverseSide::Right }> for L
where
    L::Output: SelectStrides<{ InverseSide::Right }>,
    Selected<L, { InverseSide::Right }>: Sort,
    Ordered<L, { InverseSide::Right }>: RightFold<Int<1>>,
    <Ordered<L, { InverseSide::Right }> as RightFold<Int<1>>>::Output: FinishInverse,
{
    type Output = <<Ordered<L, { InverseSide::Right }> as RightFold<Int<1>>>::Output as FinishInverse>::Output;
    fn inverse_kernel(self) -> Result<Self::Output> {
        self.attach(Int)?
            .select_strides()
            .sort()
            .right_fold(Int)?
            .finish_inverse()
    }
}
impl<L: Attach<Int<1>>> InverseKernel<{ InverseSide::Left }> for L
where
    L::Output: SelectStrides<{ InverseSide::Left }>,
    Selected<L, { InverseSide::Left }>: Sort,
    Ordered<L, { InverseSide::Left }>: LeftFold<Int<1>, Int<1>, Int<0>, Int<1>>,
    <Ordered<L, { InverseSide::Left }> as LeftFold<Int<1>, Int<1>, Int<0>, Int<1>>>::Output:
        FinishInverse,
{
    type Output = <<Ordered<L, { InverseSide::Left }> as LeftFold<Int<1>, Int<1>, Int<0>, Int<1>>>::Output as FinishInverse>::Output;
    fn inverse_kernel(self) -> Result<Self::Output> {
        self.attach(Int)?
            .select_strides()
            .sort()
            .left_fold(Int, Int, Int, Int)?
            .finish_inverse()
    }
}
