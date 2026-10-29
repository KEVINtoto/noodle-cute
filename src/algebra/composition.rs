//! Composition algorithms for affine layouts.

use crate::int::{ceil_div, mul};
use crate::layout::{
    AffineLayoutMap, DynLayout, GroupedLayout, LayoutMap, MAX_RANK, is_injective_by_enumeration,
};
use crate::{Error, Result};

use super::Leaves;

// CuTe `composition_impl` for one integral rhs mode `s:d` against a coalesced lhs.
// When the rhs crosses an lhs mode boundary with a stride that does not divide the lhs mode extent
// (CuTe's weak divisibility condition), CuTe restarts at the new mode rather than wrapping around,
// so the result may differ from pointwise evaluation `lhs_ext(s:d (i))`.
fn compose_leaf(lhs: &DynLayout, s: isize, d: isize, out: &mut Leaves) -> Result<()> {
    let r = lhs.rank();
    let (shape, stride) = (lhs.shapes(), lhs.strides());
    // A stride-0 rhs does not read lhs; an integral lhs directly scales the stride (the extended domain extends naturally here).
    if d == 0 || s == 0 {
        return out.push(s, 0);
    }
    if r == 1 {
        return out.push(s, mul(d, stride[0])?);
    }
    let start = out.len;
    let (mut rest_shape, mut rest_stride) = (s, d);
    for i in 0..r - 1 {
        let (cs, cd) = (shape[i], stride[i]);
        if cs <= 0 {
            return Err(Error::InvalidShape);
        }
        // Stride divisibility: statically asserted in CuTe; checked here at runtime.
        let magnitude = rest_stride.checked_abs().ok_or(Error::Overflow)?;
        if rest_stride % cs != 0 && magnitude >= cs {
            return Err(Error::NotDivisible);
        }
        let next_shape = ceil_div(cs, magnitude);
        let next_stride = ceil_div(magnitude, cs) * rest_stride.signum();
        if next_shape == 1 || rest_shape == 1 {
            rest_stride = next_stride;
            continue;
        }
        let new_shape = next_shape.min(rest_shape);
        // Shape divisibility, same as above.
        if rest_shape % new_shape != 0 {
            return Err(Error::NotDivisible);
        }
        out.push(new_shape, mul(rest_stride, cd)?)?;
        rest_shape /= new_shape;
        rest_stride = next_stride;
    }
    if out.len == start || rest_shape != 1 {
        out.push(rest_shape, mul(rest_stride, stride[r - 1])?)?;
    }
    Ok(())
}

// Composes an outer flat layout with a grouped rhs layout.
// CuTe uses coalesce_x rather than coalesce: the stride of a trailing shape-1 leaf determines
// the extended domain and must not be discarded.
fn compose_grouped(outer: &DynLayout, rhs: &GroupedLayout) -> Result<GroupedLayout> {
    let lhs = outer.coalesce_extended()?;
    let mut out = Leaves::new();
    if rhs.is_integral() {
        compose_leaf(&lhs, rhs.shape(0)?, rhs.stride(0)?, &mut out)?;
        return GroupedLayout::from_flat(out.layout()?);
    }
    let mut groups = [0usize; MAX_RANK];
    let mut leaf = 0;
    for (axis, count) in rhs.groups().enumerate() {
        let start = out.len;
        for _ in 0..count {
            compose_leaf(&lhs, rhs.shape(leaf)?, rhs.stride(leaf)?, &mut out)?;
            leaf += 1;
        }
        groups[axis] = out.len - start;
    }
    GroupedLayout::new(out.layout()?, &groups[..rhs.rank()])
}

/// Lazily evaluated layout composition `composition(A, B)(c) = A(B(c))`.
///
/// The output shape is taken from `inner`; `outer` receives linear coordinates and does not
/// interpret `inner`'s results according to its own address bounds.
/// Consistent with CuTe, `inner` values may exceed the logical domain of `outer`, in which
/// case they are evaluated on the extended domain of `outer` (the last leaf does not wrap modulo,
/// see [`LayoutMap::index_extended`]). `inner` coordinates must be non-negative, and `outer`
/// must support an extended domain; otherwise, [`Error::CoordinateOutOfBounds`] is returned.
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
/// An integral-shaped `B` has a single top-level mode; when caching, leaves of `R` are merged into
/// a single mode so that its rank matches `B`.
/// If divisibility conditions are not met ([`Error::NotDivisible`]) or either side is non-affine
/// (corresponding to CuTe's `ComposedLayout`), evaluation falls back to pointwise mapping.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::composition::composition;
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
/// - Returns [`Error::CoordinateOutOfBounds`] if `inner` has negative reachable addresses or if `outer` rejects extended coordinates.
/// - Returns [`Error::CapacityExceeded`] if the normalized layout exceeds [`MAX_RANK`] leaves or groups.
/// - Returns other layout errors if computing address ranges or normalized layouts fails.
pub fn composition<A: LayoutMap, B: LayoutMap>(outer: A, inner: B) -> Result<Composition<A, B>> {
    let range = inner.address_range()?;
    if let Some((min, _)) = range
        && min < 0
    {
        return Err(Error::CoordinateOutOfBounds);
    }

    let normalized = match (outer.as_affine()?, inner.as_affine()?) {
        (Some(a), Some(b)) => match compose_grouped(a.flat(), &b) {
            Ok(layout) if b.is_integral() => {
                Some(GroupedLayout::new(*layout.flat(), &[layout.flat().rank()])?)
            }
            Ok(layout) => Some(layout),
            Err(Error::NotDivisible) => None,
            Err(e) => return Err(e),
        },
        _ => None,
    };

    if normalized.is_none()
        && let Some((_, max)) = range
    {
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
/// use noodle_cute::algebra::composition::composition_layout;
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
/// - Returns [`Error::CoordinateOutOfBounds`] if `inner` has negative reachable addresses.
/// - Returns [`Error::NotDivisible`] if stride or shape divisibility conditions fail.
/// - Returns [`Error::InvalidShape`] if any non-positive extent is encountered.
/// - Returns [`Error::CapacityExceeded`] if leaf or group counts exceed [`MAX_RANK`].
/// - Returns [`Error::Overflow`] if stride arithmetic overflows.
pub fn composition_layout<A: AffineLayoutMap, B: AffineLayoutMap>(
    outer: &A,
    inner: &B,
) -> Result<GroupedLayout> {
    if let Some((min, _)) = inner.address_range()?
        && min < 0
    {
        return Err(Error::CoordinateOutOfBounds);
    }
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
    // Cached CuTe normalized layout when both sides are affine and divisible; rank matches inner.
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
        self.inner.rank()
    }

    fn extent(&self, axis: usize) -> Result<isize> {
        self.inner.extent(axis)
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
        let n = self.size()?;
        if n <= 0 {
            return Ok(None);
        }
        let first = self.index(0)?;
        let (mut min, mut max) = (first, first);
        for i in 1..n {
            let a = self.index(i)?;
            min = min.min(a);
            max = max.max(a);
        }
        Ok(Some((min, max)))
    }

    fn is_injective(&self) -> Result<bool> {
        if let Some(layout) = &self.normalized {
            return layout.is_injective();
        }
        // The outer layout guarantees injectivity only on its own logical domain;
        // if inner enters the extended domain, fall back to pairwise comparison.
        let within = match self.inner.address_range()? {
            Some((_, max)) => max < self.outer.size()?,
            None => true,
        };
        if within && self.outer.is_injective()? {
            self.inner.is_injective()
        } else {
            is_injective_by_enumeration(self)
        }
    }
}
