//! Inverse algorithms for affine layouts.

use crate::int::{add, mul};
use crate::layout::{AffineLayoutMap, DynLayout, GroupedLayout, LayoutMap, MAX_RANK};
use crate::{Error, Result};

use super::Leaves;

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
        acc = mul(acc, flat.shape(a)?)?;
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
/// Advances only along modes whose stride equals the accumulated address span (CuTe uses `abs(stride)`
/// to locate chains, multiplying back `signum`). Other modes (broadcasts, negative strides, or modes after
/// a gap) are ignored, so for non-dense layouts `R` covers only the longest contiguous address prefix starting
/// at 0 (e.g. `right_inverse(4:2) = 1:0`).
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::inverse::right_inverse;
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
/// - Returns [`Error::Unsupported`] if `layout` has size 0.
/// - Returns [`Error::CapacityExceeded`] if leaf capacity is exceeded.
/// - Returns [`Error::Overflow`] if stride arithmetic overflows.
pub fn right_inverse<L: AffineLayoutMap>(layout: L) -> Result<DynLayout> {
    if layout.size()? == 0 {
        return Err(Error::Unsupported);
    }
    let sorted = sorted_leaves(&layout)?;
    let flat = &sorted.flat;
    let mut leaves = Leaves::new();
    let mut current = 1;
    for &a in &sorted.order[..flat.rank()] {
        let (s, d) = (flat.shape(a)?, flat.stride(a)?);
        if d == current {
            leaves.push(s, sorted.preprod[a])?;
            current = mul(s, d)?;
        }
    }
    leaves.layout()?.coalesce()
}

/// CuTe `left_inverse(layout)`: returns a layout `R` such that `R(layout(i)) == i` for `i < size(layout)`.
///
/// Address holes are also defined (unconstrained values), e.g. `left_inverse(4:2) = (2, 4):(0, 1)`.
/// Stride-0 modes are ignored, so broadcast layouts yield one of their right inverses.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::inverse::left_inverse;
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
/// - Returns [`Error::Unsupported`] if `layout` has size 0 or contains negative strides.
/// - Returns [`Error::NotDivisible`] if a stride is not divisible by the covered size.
/// - Returns [`Error::NonInjective`] if non-broadcast leaves overlap (`stride < prev_shape * prev_stride`).
/// - Returns [`Error::CapacityExceeded`] or [`Error::Overflow`] on capacity or arithmetic limits.
pub fn left_inverse<L: AffineLayoutMap>(layout: L) -> Result<DynLayout> {
    if layout.size()? == 0 {
        return Err(Error::Unsupported);
    }
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
        let d = flat.stride(a)?;
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
        min_stride = mul(flat.shape(a)?, d)?;
        len += 1;
    }
    shape[len] = flat.shape(sorted.order[rank - 1])?;
    let mut leaves = Leaves::new();
    // Without leading holes (min stride 1), the initial 1:0 is eliminated by coalesce; skip it to prevent overflow on full-capacity layouts.
    let start = usize::from(len > 0 && shape[0] == 1);
    for i in start..=len {
        leaves.push(shape[i], stride[i])?;
    }
    leaves.layout()?.coalesce()
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
/// use noodle_cute::algebra::inverse::dense_left_inverse;
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
/// to a contiguous prefix. Construction linearly scans the domain for each reachable address (`O(cosize * size)`).
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::inverse::dense_right_inverse;
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
    let size = match layout.address_range()? {
        None => 0,
        Some((0, max)) => add(max, 1)?,
        Some(_) => return Err(Error::NotSurjective),
    };
    for address in 0..size {
        let mut found = false;
        for i in 0..layout.size()? {
            if layout.index(i)? == address {
                found = true;
                break;
            }
        }
        if !found {
            return Err(Error::NotSurjective);
        }
    }
    Ok(Inverse { layout, size })
}
