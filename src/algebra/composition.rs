//! Composition for static, mixed, and runtime layouts, plus the generic lazy composition.
//!
//! Follows CuTe `layout.hpp` `composition_impl`; `coalesce_x` reuses the shared coalesce module.
//! Fully static results are read through [`TypedComposition`] and `StaticValue`, mixed layouts
//! use [`Layout::composition`], [`composition_layout`] returns runtime metadata, and the generic
//! [`composition`] keeps the lazy fallback.

use crate::basis::{ScaledBasis, StrideKind, StrideProfile};
use core::marker::ConstParamTy;

use crate::int::{Integer, ceil_div, mul};
use crate::layout::{
    AffineLayoutMap, DynLayout, GroupedLayout, Layout, LayoutBuilder, LayoutDesc, LayoutMap,
    LayoutParts, Shape, StrideOf, address_range_by_scan, affine_injective,
    is_injective_by_enumeration,
};
use crate::tuple::Unwrap;
use crate::tuple::{Append, Flatten, Item};
use crate::{Bool, DynInt, Error, Int, IntTuple, Nil, Result};

use super::coalesce::{CoalesceFold, CoalesceMode, coalesce_desc};

/// CuTe `composition_impl` for one integral rhs mode `s:d` against a coalesced lhs descriptor.
pub const fn compose_leaf_desc(lhs: &LayoutDesc, s: isize, d: isize) -> Result<LayoutDesc> {
    let mut out = LayoutDesc::empty();
    let r = lhs.rank;
    if r == 0 || d == 0 || s == 0 {
        return out.push(s, 0);
    }
    if r == 1 {
        let stride = match mul(d, lhs.strides[0]) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        return out.push(s, stride);
    }
    let mut rest_shape = s;
    let mut rest_stride = d;
    let mut i = 0;
    while i + 1 < r {
        let cs = lhs.shapes[i];
        let cd = lhs.strides[i];
        let step = match checked_step(rest_shape, rest_stride, cs) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        if !step.skip {
            let next_s = match mul(rest_stride, cd) {
                Ok(v) => v,
                Err(e) => return Err(e),
            };
            out = match out.push(step.extent, next_s) {
                Ok(o) => o,
                Err(e) => return Err(e),
            };
        }
        rest_shape = step.rest;
        rest_stride = step.stride;
        i += 1;
    }
    if out.rank == 0 || rest_shape != 1 {
        let next_s = match mul(rest_stride, lhs.strides[r - 1]) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        out = match out.push(rest_shape, next_s) {
            Ok(o) => o,
            Err(e) => return Err(e),
        };
    }
    Ok(out)
}

/// Eager CuTe composition of two layout descriptors `outer ∘ inner`.
pub const fn compose_desc(outer: &LayoutDesc, inner: &LayoutDesc) -> Result<LayoutDesc> {
    let lhs = match coalesce_desc(outer, CoalesceMode::Extended) {
        Ok(l) => l,
        Err(e) => return Err(e),
    };
    if inner.rank == 0 {
        return Ok(LayoutDesc::empty());
    }
    let mut out = LayoutDesc::empty();
    let mut leaf = 0;
    while leaf < inner.rank {
        let leaf_desc = match compose_leaf_desc(&lhs, inner.shapes[leaf], inner.strides[leaf]) {
            Ok(d) => d,
            Err(e) => return Err(e),
        };
        let mut j = 0;
        while j < leaf_desc.rank {
            out = match out.push(leaf_desc.shapes[j], leaf_desc.strides[j]) {
                Ok(o) => o,
                Err(e) => return Err(e),
            };
            j += 1;
        }
        leaf += 1;
    }
    Ok(out)
}

// CuTe `composition_impl` for one integral rhs mode `s:d` against a coalesced lhs.
// When the rhs crosses an lhs mode boundary with a stride that does not divide the lhs mode extent
// (CuTe's weak divisibility condition), CuTe restarts at the new mode rather than wrapping around,
// so the result may differ from pointwise evaluation `lhs_ext(s:d (i))`.
fn compose_leaf(lhs: &DynLayout, s: isize, d: isize, out: &mut LayoutBuilder) -> Result<()> {
    let desc = LayoutDesc::from_dyn_layout(lhs);
    let leaf_desc = compose_leaf_desc(&desc, s, d)?;
    for i in 0..leaf_desc.rank {
        out.push(leaf_desc.shapes[i], leaf_desc.strides[i])?;
    }
    Ok(())
}

// Composes an outer flat layout with a grouped rhs layout.
// CuTe uses coalesce_x rather than coalesce: the stride of a trailing shape-1 leaf determines
// the extended domain and must not be discarded.
fn compose_grouped(outer: &DynLayout, rhs: &GroupedLayout) -> Result<GroupedLayout> {
    let lhs = outer.coalesce_extended()?;
    let mut out = LayoutBuilder::new();
    if rhs.is_integral() {
        compose_leaf(&lhs, rhs.leaf_shape(0)?, rhs.leaf_stride(0)?, &mut out)?;
        return GroupedLayout::from_flat(out.finish_flat()?);
    }
    let mut leaf = 0;
    for count in rhs.groups() {
        for _ in 0..count {
            compose_leaf(
                &lhs,
                rhs.leaf_shape(leaf)?,
                rhs.leaf_stride(leaf)?,
                &mut out,
            )?;
            leaf += 1;
        }
        out.close_mode()?;
    }
    out.finish()
}

/// Lazily evaluated layout composition `composition(A, B)(c) = A(B(c))`.
///
/// The size is taken from `inner`; `outer` receives linear coordinates and does not
/// interpret `inner`'s results according to its own address bounds.
/// Consistent with CuTe, `inner` values may exceed the logical domain of `outer`, in which
/// case they are evaluated on the extended domain of `outer` (the last leaf does not wrap modulo,
/// see [`LayoutMap::index_extended`]).
/// An empty `inner` requires no reachable addresses, allowing composition with an empty `outer`.
///
/// When both sides are affine layouts and satisfy CuTe divisibility conditions, the normalized layout
/// `R = composition_layout(outer, inner)` is cached. `index`, `index_extended`, and
/// `index_extended_of_coord` are evaluated through `R`, matching CuTe's `Layout ∘ Layout`.
/// In the logical domain, `R` and pointwise `A(B(c))` differ only under weak divisibility
/// (where a mode of `B` crosses `A`'s mode boundary with a stride not dividing `A`'s mode size, e.g.,
/// `(4, 6):(6, 1) ∘ 8:3 = (2, 4):(18, 1)`, where CuTe restarts at the next mode of `A`, which
/// padded `logical_divide` relies upon). In the extended domain, `R` and pointwise evaluation
/// generally differ: `R` extends along the last leaf of each mode, whereas pointwise evaluation
/// carries across leaves of `A`.
/// `rank` and `extent` follow `R` as in CuTe: an integral-shaped `B` yields one top-level mode
/// per leaf of `R`, while a tuple-shaped `B` keeps its top-level modes.
/// If divisibility conditions are not met ([`Error::NotDivisible`]) or either side is non-affine
/// (corresponding to CuTe's `ComposedLayout`), evaluation falls back to pointwise mapping with
/// the modes of `inner`. Only this fallback requires nonnegative `inner` addresses and an `outer`
/// that supports the extended domain; otherwise, [`Error::CoordinateOutOfBounds`] is returned.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::composition;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let outer = DynLayout::new(&[4, 4], &[1, 4])?;
/// let inner = DynLayout::new(&[2, 2], &[1, 2])?;
/// let comp = composition(outer, inner)?;
/// assert_eq!(comp.size()?, 4);
/// assert_eq!(comp.index_of_coord(&[1, 1])?, 3);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// - Returns [`Error::CoordinateOutOfBounds`] if the pointwise fallback sees negative `inner` addresses or `outer` rejects extended coordinates.
/// - Returns [`Error::CapacityExceeded`] if the normalized layout exceeds [`MAX_RANK`](crate::layout::MAX_RANK) leaves or groups.
/// - Returns other layout errors if computing address ranges or normalized layouts fails.
pub fn composition<A: LayoutMap, B: LayoutMap>(outer: A, inner: B) -> Result<Composition<A, B>> {
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

    Ok(Composition {
        outer,
        inner,
        normalized,
    })
}

/// Eager CuTe `composition(A, B)`: produces the normalized layout `R` satisfying
/// `R(i) = A(B(i))` on `B`'s domain, where `B`'s image may extend into `A`'s extended domain.
///
/// `A` is coalesced (preserving its extended map via `coalesce_extended`), and `B` distributes over
/// its top-level modes, so the result inherits `B`'s top-level grouping (an integral `B` yields the
/// leaves of `composition_impl` as top-level modes, as in CuTe).
/// Divisibility conditions for strides and shapes (asserted statically in CuTe) are checked at runtime,
/// returning [`Error::NotDivisible`] on failure.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::composition_layout;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let a = DynLayout::new(&[4, 4], &[1, 4])?;
/// let b = DynLayout::new(&[2, 2], &[1, 2])?;
/// let r = composition_layout(&a, &b)?;
/// assert_eq!(r.size()?, 4);
/// assert_eq!(r.shapes(), &[2, 2]);
/// assert_eq!(r.strides(), &[1, 2]);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// - Returns [`Error::NotDivisible`] if stride or shape divisibility conditions fail.
/// - Returns [`Error::InvalidShape`] if any non-positive extent is encountered.
/// - Returns [`Error::CapacityExceeded`] if leaf or group counts exceed [`MAX_RANK`](crate::layout::MAX_RANK).
/// - Returns [`Error::Overflow`] if stride arithmetic overflows.
pub fn composition_layout<A: AffineLayoutMap, B: AffineLayoutMap>(
    outer: &A,
    inner: &B,
) -> Result<GroupedLayout> {
    compose_grouped(&outer.as_flat_layout()?, &inner.as_grouped_layout()?)
}

/// Composite layout evaluating `outer(inner(coord))`.
///
/// Constructed via [`composition`].
///
/// Does not directly implement [`AffineLayoutMap`]; to obtain an affine representation, use
/// [`LayoutMap::as_affine`] to query the cached normalized layout, or call [`composition_layout`]
/// to construct a [`GroupedLayout`].
#[derive(Clone, Debug)]
pub struct Composition<A, B> {
    outer: A,
    inner: B,
    // Cached CuTe normalized layout when both sides are affine and divisible.
    pub(crate) normalized: Option<GroupedLayout>,
}

impl<A, B> Composition<A, B> {
    /// Returns a reference to the outer layout.
    pub fn outer(&self) -> &A {
        &self.outer
    }

    /// Returns a reference to the inner layout.
    pub fn inner(&self) -> &B {
        &self.inner
    }

    /// Returns `true` if evaluation uses the cached CuTe normalized layout rather than pointwise `A(B(c))`.
    pub fn is_normalized(&self) -> bool {
        self.normalized.is_some()
    }
}

impl<A: LayoutMap, B: LayoutMap> LayoutMap for Composition<A, B> {
    fn size(&self) -> Result<isize> {
        self.inner.size()
    }

    fn rank(&self) -> usize {
        self.normalized
            .as_ref()
            .map_or_else(|| self.inner.rank(), |layout| layout.rank())
    }

    fn extent(&self, axis: usize) -> Result<isize> {
        match &self.normalized {
            Some(layout) => layout.extent(axis),
            None => self.inner.extent(axis),
        }
    }

    fn index(&self, linear: isize) -> Result<isize> {
        if let Some(layout) = &self.normalized {
            return layout.index(linear);
        }
        self.outer.index_extended(self.inner.index(linear)?)
    }

    fn index_extended(&self, linear: isize) -> Result<isize> {
        if let Some(layout) = &self.normalized {
            return layout.index_extended(linear);
        }
        self.outer
            .index_extended(self.inner.index_extended(linear)?)
    }

    fn index_extended_of_coord(&self, coord: &[isize]) -> Result<isize> {
        if let Some(layout) = &self.normalized {
            return layout.index_extended_of_coord(coord);
        }
        self.outer
            .index_extended(self.inner.index_extended_of_coord(coord)?)
    }

    fn as_affine(&self) -> Result<Option<GroupedLayout>> {
        // If either side is non-affine or divisibility fails, normalized is None:
        // composition is still valid pointwise without an affine form.
        Ok(self.normalized)
    }

    fn address_range(&self) -> Result<Option<(isize, isize)>> {
        if let Some(layout) = &self.normalized {
            return layout.address_range();
        }
        address_range_by_scan(self)
    }

    fn is_injective(&self) -> Result<bool> {
        if let Some(layout) = &self.normalized {
            return layout.is_injective();
        }
        // Fast path when `outer` has a cheap stride-separation proof on its logical domain;
        // if `outer` is non-affine, inconclusive, or exceeded by `inner`, enumerate the composed domain.
        let outer_proven = match self.outer.as_affine() {
            Ok(Some(a)) => affine_injective(&a, 0..a.leaf_count())? == Some(true),
            Ok(None) | Err(Error::CapacityExceeded) => false,
            Err(e) => return Err(e),
        };
        if outer_proven {
            let within = match self.inner.address_range()? {
                Some((_, max)) => max < self.outer.size()?,
                None => true,
            };
            if within {
                return self.inner.is_injective();
            }
        }
        is_injective_by_enumeration(self)
    }
}

type CoalescedLhs<S, D> =
    <Layout<<S as Flatten>::Output, <D as Flatten>::Output> as CoalesceFold<
        { CoalesceMode::Extended },
    >>::Output;

/// Result type of the typed composition, keeping the RHS nesting and every statically
/// derivable leaf.
///
/// For fully static inputs, `<TypedComposition<A, B> as StaticValue>::VALUE` is the result as
/// a constant. Like CuTe it allows the extended domain and negative strides; static
/// intermediate values must be representable as `isize`. An empty RHS stays an empty tuple,
/// and an empty LHS is treated as `1:0` by this crate's scalar convention.
///
/// ```
/// use noodle_cute::{Int, StaticValue, algebra::TypedComposition, layout::{Layout, LayoutMap}};
/// type A = Layout<Int<8>, Int<2>>;
/// type B = Layout<Int<3>, Int<2>>;
/// const C: Layout<Int<3>, Int<4>> = <TypedComposition<A, B> as StaticValue>::VALUE;
/// assert_eq!(C.index(2), Ok(8));
/// ```
///
/// Runtime leaves have no constant value:
/// ```compile_fail
/// use noodle_cute::{DynInt, Int, StaticValue, algebra::TypedComposition, layout::Layout};
/// const C: TypedComposition<Layout<Int<8>, Int<1>>, Layout<DynInt, Int<1>>> = StaticValue::VALUE;
/// ```
/// A static stride that violates the divisibility condition is rejected at compile time:
/// ```compile_fail
/// use noodle_cute::{Int, IntTuple, Nil, StaticValue, algebra::TypedComposition, layout::Layout};
/// type Pair<const A: isize, const B: isize> = IntTuple<Int<A>, IntTuple<Int<B>, Nil>>;
/// const C: TypedComposition<Layout<Pair<4, 6>, Pair<6, 1>>, Layout<Int<8>, Int<5>>> =
///     StaticValue::VALUE;
/// ```
/// So is a static shape that violates it:
/// ```compile_fail
/// use noodle_cute::{Int, IntTuple, Nil, StaticValue, algebra::TypedComposition, layout::Layout};
/// type Pair<const A: isize, const B: isize> = IntTuple<Int<A>, IntTuple<Int<B>, Nil>>;
/// const C: TypedComposition<Layout<Pair<4, 6>, Pair<6, 1>>, Layout<Int<3>, Int<3>>> =
///     StaticValue::VALUE;
/// ```
/// A static stride product that overflows:
/// ```compile_fail
/// use noodle_cute::{Int, StaticValue, algebra::TypedComposition, layout::Layout};
/// const C: TypedComposition<Layout<Int<2>, Int<{ isize::MAX }>>, Layout<Int<2>, Int<2>>> =
///     StaticValue::VALUE;
/// ```
/// And shape/stride profiles that differ:
/// ```compile_fail
/// use noodle_cute::{Int, IntTuple, Nil, StaticValue, algebra::TypedComposition, layout::Layout};
/// const C: TypedComposition<Layout<Int<4>, Int<1>>, Layout<IntTuple<Int<2>, Nil>, Int<1>>> =
///     StaticValue::VALUE;
/// ```
pub type TypedComposition<A, B> = <A as TypedCompose<B>>::Output;

/// Checked composition of typed layouts; fully static results use this trait's output type.
#[diagnostic::on_unimplemented(
    message = "layout `{Self}` cannot be composed with `{Rhs}`",
    label = "incompatible typed layout composition",
    note = "check shape/stride profiles and the static divisibility and overflow requirements"
)]
pub trait TypedCompose<Rhs> {
    type Output;
    fn compose(self, rhs: Rhs) -> Result<Self::Output>;
}

impl<S, D> Layout<S, D> {
    /// Composes static or mixed layouts with CuTe `composition_impl`, keeping the result type.
    ///
    /// Only statically known zero strides and unit shapes change the output structure; modes
    /// that happen to be units at runtime are kept. Negative strides and the extended domain
    /// are allowed, and the output is validated by `Layout::new`. A dynamic divisibility failure
    /// returns `NotDivisible`, and a division by a dynamic zero stride in a multi-mode LHS
    /// returns `InvalidShape`. Purely static errors fail at instantiation; the result is not
    /// limited by `MAX_RANK`.
    ///
    /// ```
    /// use noodle_cute::{DynInt, Int, layout::{Layout, LayoutMap}};
    /// let a = Layout::new(Int::<8>, DynInt(2))?;
    /// let b = Layout::new(DynInt(3), Int::<2>)?;
    /// let c: Layout<DynInt, DynInt> = a.composition(b)?;
    /// assert_eq!(c.index(2)?, 8);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    ///
    /// With mixed inputs, a statically known stride divisibility error is also rejected at
    /// compile time:
    /// ```compile_fail
    /// use noodle_cute::{DynInt, Int, shape, stride, layout::Layout};
    /// let a = Layout::new(shape![Int::<4>, Int::<6>], stride![DynInt(6), Int::<1>]).unwrap();
    /// let b = Layout::new(DynInt(8), Int::<5>).unwrap();
    /// let _ = a.composition(b);
    /// ```
    pub fn composition<RS, RD>(
        &self,
        rhs: Layout<RS, RD>,
    ) -> Result<TypedComposition<Self, Layout<RS, RD>>>
    where
        Self: TypedCompose<Layout<RS, RD>> + Clone,
    {
        self.clone().compose(rhs)
    }
}

type CompositionTree<S, D, RS, RD> = <Layout<RS, RD> as ComposeTree<CoalescedLhs<S, D>>>::Output;

pub trait ComposeResult<L> {
    type Output;
    fn validate(layout: L) -> Result<Self::Output>;
}
impl<S: Shape, D: StrideOf<S>> ComposeResult<Layout<S, D>> for Bool<false> {
    type Output = Layout<S, D>;
    fn validate(layout: Self::Output) -> Result<Self::Output> {
        Layout::new(layout.shape, layout.stride)
    }
}
impl<S: Shape, D> ComposeResult<Layout<S, D>> for Bool<true> {
    type Output = Layout<S, D>;
    fn validate(layout: Self::Output) -> Result<Self::Output> {
        for i in 0..S::LEAVES {
            if layout.shape.leaf(i)? < 0 {
                return Err(Error::InvalidShape);
            }
        }
        layout.shape.product()?;
        Ok(layout)
    }
}
impl<S, D, RS, RD> TypedCompose<Layout<RS, RD>> for Layout<S, D>
where
    S: Shape + Flatten,
    D: Flatten + StrideKind + StrideProfile<S>,
    RS: Shape,
    RD: Item<Profile = RS::Profile>,
    Layout<S::Output, D::Output>: CoalesceFold<{ CoalesceMode::Extended }>,
    Layout<RS, RD>: ComposeTree<CoalescedLhs<S, D>>,
    D::Kind: ComposeResult<CompositionTree<S, D, RS, RD>>,
{
    type Output = <D::Kind as ComposeResult<CompositionTree<S, D, RS, RD>>>::Output;
    fn compose(self, rhs: Layout<RS, RD>) -> Result<Self::Output> {
        for i in 0..S::LEAVES {
            if self.shape.leaf(i)? < 0 {
                return Err(Error::InvalidShape);
            }
        }
        self.shape.product()?;
        rhs.size()?;
        let lhs = Layout {
            shape: self.shape.flatten(),
            stride: self.stride.flatten(),
        }
        .fold();
        let result = rhs.compose_tree(&lhs)?;
        <D::Kind as ComposeResult<_>>::validate(result)
    }
}

pub trait ComposeTree<Lhs> {
    type Output: LayoutParts;
    fn compose_tree(self, lhs: &Lhs) -> Result<Self::Output>;
}
impl<Lhs> ComposeTree<Lhs> for Layout<Nil, Nil> {
    type Output = Layout<Nil, Nil>;
    fn compose_tree(self, _: &Lhs) -> Result<Self> {
        Ok(self)
    }
}
impl<Lhs, SH, ST, DH, DT> ComposeTree<Lhs> for Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>
where
    Layout<SH, DH>: ComposeTree<Lhs>,
    Layout<ST, DT>: ComposeTree<Lhs>,
{
    type Output = Layout<
        IntTuple<
            <<Layout<SH, DH> as ComposeTree<Lhs>>::Output as LayoutParts>::Shape,
            <<Layout<ST, DT> as ComposeTree<Lhs>>::Output as LayoutParts>::Shape,
        >,
        IntTuple<
            <<Layout<SH, DH> as ComposeTree<Lhs>>::Output as LayoutParts>::Stride,
            <<Layout<ST, DT> as ComposeTree<Lhs>>::Output as LayoutParts>::Stride,
        >,
    >;
    fn compose_tree(self, lhs: &Lhs) -> Result<Self::Output> {
        let h = Layout {
            shape: self.shape.0,
            stride: self.stride.0,
        }
        .compose_tree(lhs)?
        .into_layout();
        let t = Layout {
            shape: self.shape.1,
            stride: self.stride.1,
        }
        .compose_tree(lhs)?
        .into_layout();
        Ok(Layout {
            shape: IntTuple(h.shape, t.shape),
            stride: IntTuple(h.stride, t.stride),
        })
    }
}

pub const ZERO_STRIDE<D: Integer>: bool = matches!(D::STATIC, Some(0));
pub trait ComposeLeaf<Lhs, S, D> {
    type Output: LayoutParts;
    fn compose_leaf(lhs: &Lhs, s: S, d: D) -> Result<Self::Output>;
}
impl<Lhs, S: Integer, D: Integer> ComposeTree<Lhs> for Layout<S, D>
where
    Bool<{ ZERO_STRIDE::<D> }>: ComposeLeaf<Lhs, S, D>,
{
    type Output = <Bool<{ ZERO_STRIDE::<D> }> as ComposeLeaf<Lhs, S, D>>::Output;
    fn compose_tree(self, lhs: &Lhs) -> Result<Self::Output> {
        <Bool<{ ZERO_STRIDE::<D> }> as ComposeLeaf<Lhs, S, D>>::compose_leaf(
            lhs,
            self.shape,
            self.stride,
        )
    }
}
impl<Lhs, S, D> ComposeLeaf<Lhs, S, D> for Bool<true> {
    type Output = Layout<S, D>;
    fn compose_leaf(_: &Lhs, s: S, d: D) -> Result<Layout<S, D>> {
        Ok(Layout {
            shape: s,
            stride: d,
        })
    }
}
impl<Lhs, S, D> ComposeLeaf<Lhs, S, D> for Bool<false>
where
    Lhs: Clone + Fold<S, D, Nil, Nil>,
{
    type Output = <Lhs as Fold<S, D, Nil, Nil>>::Output;
    fn compose_leaf(lhs: &Lhs, s: S, d: D) -> Result<Self::Output> {
        lhs.clone().fold(s, d, Nil, Nil)
    }
}

pub struct StepData {
    pub skip: bool,
    pub next_shape: isize,
    pub extent: isize,
    pub rest: isize,
    pub stride: isize,
}

const fn checked_step(s: isize, d: isize, cs: isize) -> Result<StepData> {
    if s < 0 || cs <= 0 || d == 0 {
        return Err(Error::InvalidShape);
    }
    if d % cs != 0 && d >= cs {
        return Err(Error::NotDivisible);
    }
    let magnitude = match d.checked_abs() {
        Some(n) => n,
        None => return Err(Error::Overflow),
    };
    let next_shape = ceil_div(cs, magnitude);
    let next_stride = ceil_div(magnitude, cs) * d.signum();
    if next_shape == 1 || s == 1 {
        Ok(StepData {
            skip: true,
            next_shape,
            extent: 1,
            rest: s,
            stride: next_stride,
        })
    } else {
        let extent = if next_shape < s { next_shape } else { s };
        if extent <= 0 {
            return Err(Error::InvalidShape);
        }
        if s % extent != 0 {
            return Err(Error::NotDivisible);
        }
        Ok(StepData {
            skip: false,
            next_shape,
            extent,
            rest: s / extent,
            stride: next_stride,
        })
    }
}

const fn step(s: isize, d: isize, cs: isize) -> StepData {
    match checked_step(s, d, cs) {
        Ok(v) => v,
        Err(Error::Overflow) => panic!("static composition arithmetic overflow"),
        Err(Error::NotDivisible) => panic!("static composition is not divisible"),
        Err(_) => panic!("invalid static composition shape or stride"),
    }
}

pub const STEP_STATIC<S: Integer, D: Integer, CS: Integer>: bool =
    S::STATIC.is_some() && D::STATIC.is_some() && CS::STATIC.is_some();
pub const REST_STATIC<S: Integer, D: Integer, CS: Integer>: bool =
    matches!(S::STATIC, Some(0)) || STEP_STATIC::<S, D, CS>;
pub const STRIDE_STATIC<D: Integer, CS: Integer>: bool =
    D::STATIC.is_some() && CS::STATIC.is_some();
/// Action taken during composition step traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum StepAction {
    Skip,
    Produce,
}

pub const fn step_action(s: Option<isize>, d: Option<isize>, cs: Option<isize>) -> StepAction {
    if matches!(s, Some(1))
        || match (d, cs) {
            (Some(d_val), Some(cs_val)) => step(1, d_val, cs_val).next_shape == 1,
            _ => false,
        }
    {
        StepAction::Skip
    } else {
        StepAction::Produce
    }
}

pub const STEP_ACTION<S: Integer, D: Integer, CS: Integer>: StepAction =
    step_action(S::STATIC, D::STATIC, CS::STATIC);
pub const EXTENT<S: Integer, D: Integer, CS: Integer>: isize =
    match (S::STATIC, D::STATIC, CS::STATIC) {
        (Some(s), Some(d), Some(cs)) => step(s, d, cs).extent,
        _ => 0,
    };
pub const REST<S: Integer, D: Integer, CS: Integer>: isize =
    match (S::STATIC, D::STATIC, CS::STATIC) {
        (Some(0), _, _) => 0,
        (Some(s), Some(d), Some(cs)) => step(s, d, cs).rest,
        _ => 0,
    };
pub const STRIDE<D: Integer, CS: Integer>: isize = match (D::STATIC, CS::STATIC) {
    (Some(d), Some(cs)) => step(1, d, cs).stride,
    _ => 0,
};

pub trait StepNumber<const N: isize> {
    type Output: Integer;
    fn make(value: isize) -> Self::Output;
}
impl<const N: isize> StepNumber<N> for Bool<true> {
    type Output = Int<N>;
    fn make(_: isize) -> Self::Output {
        Int
    }
}
impl<const N: isize> StepNumber<N> for Bool<false> {
    type Output = DynInt;
    fn make(value: isize) -> Self::Output {
        DynInt(value)
    }
}

pub trait TypedStep<S, D, CS> {
    type Extent: Integer;
    type Rest: Integer;
    type Stride: Integer;
    fn run(s: S, d: D, cs: CS) -> Result<(Self::Extent, Self::Rest, Self::Stride)>;
}
impl<S: Integer, D: Integer, CS: Integer> TypedStep<S, D, CS> for ()
where
    Bool<{ STEP_STATIC::<S, D, CS> }>: StepNumber<{ EXTENT::<S, D, CS> }>,
    Bool<{ REST_STATIC::<S, D, CS> }>: StepNumber<{ REST::<S, D, CS> }>,
    Bool<{ STRIDE_STATIC::<D, CS> }>: StepNumber<{ STRIDE::<D, CS> }>,
{
    type Extent = <Bool<{ STEP_STATIC::<S, D, CS> }> as StepNumber<{ EXTENT::<S, D, CS> }>>::Output;
    type Rest = <Bool<{ REST_STATIC::<S, D, CS> }> as StepNumber<{ REST::<S, D, CS> }>>::Output;
    type Stride = <Bool<{ STRIDE_STATIC::<D, CS> }> as StepNumber<{ STRIDE::<D, CS> }>>::Output;
    fn run(s: S, d: D, cs: CS) -> Result<(Self::Extent, Self::Rest, Self::Stride)> {
        let shape = s.value();
        if shape < 0 {
            return Err(Error::InvalidShape);
        }
        let mut v = checked_step(1, d.value(), cs.value())?;
        if step_action(S::STATIC, D::STATIC, CS::STATIC) == StepAction::Skip {
            v.rest = shape;
        } else {
            v.extent = v.next_shape.min(shape);
            // CuTe's `_0 / x` is a static `_0`, so no runtime division happens.
            v.rest = if matches!(S::STATIC, Some(0)) {
                0
            } else {
                if v.extent == 0 {
                    return Err(Error::InvalidShape);
                }
                if shape % v.extent != 0 {
                    return Err(Error::NotDivisible);
                }
                shape / v.extent
            };
        }
        Ok((
            <Bool<{ STEP_STATIC::<S, D, CS> }> as StepNumber<{ EXTENT::<S, D, CS> }>>::make(
                v.extent,
            ),
            <Bool<{ REST_STATIC::<S, D, CS> }> as StepNumber<{ REST::<S, D, CS> }>>::make(v.rest),
            <Bool<{ STRIDE_STATIC::<D, CS> }> as StepNumber<{ STRIDE::<D, CS> }>>::make(v.stride),
        ))
    }
}

pub const PRODUCT_STATIC<A: Integer, B: Integer>: bool = matches!(A::STATIC, Some(0))
    || matches!(B::STATIC, Some(0))
    || (A::STATIC.is_some() && B::STATIC.is_some());
pub const PRODUCT<A: Integer, B: Integer>: isize = match (A::STATIC, B::STATIC) {
    (Some(0), _) | (_, Some(0)) => 0,
    (Some(a), Some(b)) => a * b,
    _ => 0,
};

// CuTe's `_0 * x` is a static `_0`, which this crate's mixed multiplication does not model.
pub trait ComposeProduct<A, B> {
    type Output;
    fn product(a: A, b: B) -> Result<Self::Output>;
}
impl<A: Integer, B: Integer> ComposeProduct<A, B> for ()
where
    Bool<{ PRODUCT_STATIC::<A, B> }>: StepNumber<{ PRODUCT::<A, B> }>,
{
    type Output = <Bool<{ PRODUCT_STATIC::<A, B> }> as StepNumber<{ PRODUCT::<A, B> }>>::Output;
    fn product(a: A, b: B) -> Result<Self::Output> {
        let value = mul(a.value(), b.value())?;
        Ok(<Bool<{ PRODUCT_STATIC::<A, B> }> as StepNumber<
            { PRODUCT::<A, B> },
        >>::make(value))
    }
}

pub trait ScaleBasis<D, V, const M: usize> {
    type Output;
    fn scale(d: D, value: V) -> Result<Self::Output>;
}
impl<D, V, const M: usize> ScaleBasis<D, V, M> for Bool<true> {
    type Output = Int<0>;
    fn scale(_: D, _: V) -> Result<Self::Output> {
        Ok(Int)
    }
}
impl<D, V, const M: usize> ScaleBasis<D, V, M> for Bool<false>
where
    (): ComposeProduct<D, V>,
{
    type Output = ScaledBasis<<() as ComposeProduct<D, V>>::Output, M>;
    fn scale(d: D, value: V) -> Result<Self::Output> {
        Ok(ScaledBasis(<() as ComposeProduct<D, V>>::product(
            d, value,
        )?))
    }
}
impl<D: Integer, V: Integer, const M: usize> ComposeProduct<D, ScaledBasis<V, M>> for ()
where
    Bool<{ ZERO_STRIDE::<D> }>: ScaleBasis<D, V, M>,
{
    type Output = <Bool<{ ZERO_STRIDE::<D> }> as ScaleBasis<D, V, M>>::Output;
    fn product(d: D, basis: ScaledBasis<V, M>) -> Result<Self::Output> {
        <Bool<{ ZERO_STRIDE::<D> }> as ScaleBasis<D, V, M>>::scale(d, basis.0)
    }
}

pub trait Fold<S, D, OS, OD> {
    type Output: LayoutParts;
    fn fold(self, s: S, d: D, os: OS, od: OD) -> Result<Self::Output>;
}
impl<S: Integer, D, CS, CD> Fold<S, D, Nil, Nil> for Layout<IntTuple<CS, Nil>, IntTuple<CD, Nil>>
where
    Layout<Nil, Nil>: Finish<{ FINISH_EMPTY::<S> }, S, D, CD>,
{
    type Output = <Layout<Nil, Nil> as Finish<{ FINISH_EMPTY::<S> }, S, D, CD>>::Output;
    fn fold(self, s: S, d: D, os: Nil, od: Nil) -> Result<Self::Output> {
        Layout {
            shape: os,
            stride: od,
        }
        .finish(s, d, self.stride.0)
    }
}

impl<S: Integer, D, CS, CD, SH, ST, DH, DT> Fold<S, D, IntTuple<SH, ST>, IntTuple<DH, DT>>
    for Layout<IntTuple<CS, Nil>, IntTuple<CD, Nil>>
where
    Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>: Finish<{ FINISH_NONEMPTY::<S> }, S, D, CD>,
{
    type Output = <Layout<IntTuple<SH, ST>, IntTuple<DH, DT>> as Finish<
        { FINISH_NONEMPTY::<S> },
        S,
        D,
        CD,
    >>::Output;
    fn fold(self, s: S, d: D, os: IntTuple<SH, ST>, od: IntTuple<DH, DT>) -> Result<Self::Output> {
        Layout {
            shape: os,
            stride: od,
        }
        .finish(s, d, self.stride.0)
    }
}
impl<S: Integer, D: Integer, CS: Integer, CD, SH, ST, DH, DT, OS, OD> Fold<S, D, OS, OD>
    for Layout<IntTuple<CS, IntTuple<SH, ST>>, IntTuple<CD, IntTuple<DH, DT>>>
where
    (): FoldStep<
            { STEP_ACTION::<S, D, CS> },
            Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>,
            S,
            D,
            CS,
            CD,
            OS,
            OD,
        >,
{
    type Output = <() as FoldStep<
        { STEP_ACTION::<S, D, CS> },
        Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>,
        S,
        D,
        CS,
        CD,
        OS,
        OD,
    >>::Output;
    fn fold(self, s: S, d: D, os: OS, od: OD) -> Result<Self::Output> {
        <() as FoldStep<
            { STEP_ACTION::<S, D, CS> },
            Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>,
            S,
            D,
            CS,
            CD,
            OS,
            OD,
        >>::run(
            Layout {
                shape: self.shape.1,
                stride: self.stride.1,
            },
            s,
            d,
            self.shape.0,
            self.stride.0,
            os,
            od,
        )
    }
}

pub trait FoldStep<const ACTION: StepAction, Tail, S, D, CS, CD, OS, OD> {
    type Output: LayoutParts;
    fn run(tail: Tail, s: S, d: D, cs: CS, cd: CD, os: OS, od: OD) -> Result<Self::Output>;
}
impl<Tail, S: Integer, D: Integer, CS: Integer, CD, OS, OD>
    FoldStep<{ StepAction::Skip }, Tail, S, D, CS, CD, OS, OD> for ()
where
    (): TypedStep<S, D, CS>,
    Tail: Fold<S, <() as TypedStep<S, D, CS>>::Stride, OS, OD>,
{
    type Output = <Tail as Fold<S, <() as TypedStep<S, D, CS>>::Stride, OS, OD>>::Output;
    fn run(tail: Tail, s: S, d: D, cs: CS, _: CD, os: OS, od: OD) -> Result<Self::Output> {
        let (_, _, next_stride) = <() as TypedStep<S, D, CS>>::run(s, d, cs)?;
        tail.fold(s, next_stride, os, od)
    }
}
impl<Tail, S: Integer, D: Integer, CS: Integer, CD, OS, OD>
    FoldStep<{ StepAction::Produce }, Tail, S, D, CS, CD, OS, OD> for ()
where
    (): ComposeProduct<D, CD>,
    (): TypedStep<S, D, CS>,
    OS: Append<IntTuple<<() as TypedStep<S, D, CS>>::Extent, Nil>>,
    OD: Append<IntTuple<<() as ComposeProduct<D, CD>>::Output, Nil>>,
    Tail: Fold<
            <() as TypedStep<S, D, CS>>::Rest,
            <() as TypedStep<S, D, CS>>::Stride,
            OS::Output,
            OD::Output,
        >,
{
    type Output = <Tail as Fold<
        <() as TypedStep<S, D, CS>>::Rest,
        <() as TypedStep<S, D, CS>>::Stride,
        OS::Output,
        OD::Output,
    >>::Output;
    fn run(tail: Tail, s: S, d: D, cs: CS, cd: CD, os: OS, od: OD) -> Result<Self::Output> {
        let (extent, rest, next_stride) = <() as TypedStep<S, D, CS>>::run(s, d, cs)?;
        let stride = <() as ComposeProduct<D, CD>>::product(d, cd)?;
        tail.fold(
            rest,
            next_stride,
            os.append(IntTuple(extent, Nil)),
            od.append(IntTuple(stride, Nil)),
        )
    }
}

/// Final stage action taken when completing composition fold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum FinishAction {
    Direct,
    DropUnit,
    Append,
}

pub const fn finish_action(is_empty: bool, s: Option<isize>) -> FinishAction {
    if is_empty {
        FinishAction::Direct
    } else if matches!(s, Some(1)) {
        FinishAction::DropUnit
    } else {
        FinishAction::Append
    }
}

pub const FINISH_EMPTY<S: Integer>: FinishAction = finish_action(true, S::STATIC);
pub const FINISH_NONEMPTY<S: Integer>: FinishAction = finish_action(false, S::STATIC);

pub trait Finish<const ACTION: FinishAction, S, D, CD> {
    type Output: LayoutParts;
    fn finish(self, s: S, d: D, cd: CD) -> Result<Self::Output>;
}
impl<S, D: Integer, CD> Finish<{ FinishAction::Direct }, S, D, CD> for Layout<Nil, Nil>
where
    (): ComposeProduct<D, CD>,
{
    type Output = Layout<S, <() as ComposeProduct<D, CD>>::Output>;
    fn finish(
        self,
        s: S,
        d: D,
        cd: CD,
    ) -> Result<Layout<S, <() as ComposeProduct<D, CD>>::Output>> {
        Ok(Layout {
            shape: s,
            stride: <() as ComposeProduct<D, CD>>::product(d, cd)?,
        })
    }
}
impl<S, D, CD, SH, ST, DH, DT> Finish<{ FinishAction::DropUnit }, S, D, CD>
    for Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>
where
    IntTuple<SH, ST>: Unwrap,
    IntTuple<DH, DT>: Unwrap,
{
    type Output =
        Layout<<IntTuple<SH, ST> as Unwrap>::Output, <IntTuple<DH, DT> as Unwrap>::Output>;
    fn finish(self, _: S, _: D, _: CD) -> Result<Self::Output> {
        Ok(Layout {
            shape: self.shape.unwrap(),
            stride: self.stride.unwrap(),
        })
    }
}
impl<S, D: Integer, CD, SH, ST, DH, DT> Finish<{ FinishAction::Append }, S, D, CD>
    for Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>
where
    (): ComposeProduct<D, CD>,
    IntTuple<SH, ST>: Append<IntTuple<S, Nil>>,
    IntTuple<DH, DT>: Append<IntTuple<<() as ComposeProduct<D, CD>>::Output, Nil>>,
{
    type Output = Layout<
        <IntTuple<SH, ST> as Append<IntTuple<S, Nil>>>::Output,
        <IntTuple<DH, DT> as Append<IntTuple<<() as ComposeProduct<D, CD>>::Output, Nil>>>::Output,
    >;
    fn finish(self, s: S, d: D, cd: CD) -> Result<Self::Output> {
        Ok(Layout {
            shape: self.shape.append(IntTuple(s, Nil)),
            stride: self.stride.append(IntTuple(
                <() as ComposeProduct<D, CD>>::product(d, cd)?,
                Nil,
            )),
        })
    }
}
