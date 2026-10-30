//! Complement algorithms for affine layouts.

use crate::int::{ceil_div, mul};
use crate::layout::{AffineLayoutMap, DynLayout, LayoutMap, MAX_RANK};
use crate::{Error, Result};

use super::Leaves;

// Common prefix logic of CuTe `detail::complement`: filters unit modes, sorts stably by stride,
// and fills in address gaps one by one.
// Returns the prefix leaves of the complement layout along with the next stride; returns `None`
// when only `1:0` remains after filtering (CuTe special case).
fn complement_prefix<L: AffineLayoutMap>(layout: &L) -> Result<Option<(Leaves, isize)>> {
    let flat = layout.as_flat_layout()?;
    if flat.size()? == 0 {
        return Err(Error::Unsupported);
    }
    let flat = flat.filter()?;
    if flat.rank() == 1 && flat.stride(0)? == 0 {
        return Ok(None);
    }
    let mut order = [0usize; MAX_RANK];
    for (a, slot) in order.iter_mut().enumerate().take(flat.rank()) {
        if flat.stride(a)? < 0 {
            return Err(Error::Unsupported);
        }
        *slot = a;
    }
    // Unstable sort by key (stride, index) is equivalent to CuTe's stable sort without allocations.
    order[..flat.rank()].sort_unstable_by_key(|&a| (flat.strides()[a], a));
    let mut prefix = Leaves::new();
    let mut current = 1;
    for &a in &order[..flat.rank()] {
        let (s, d) = (flat.shape(a)?, flat.stride(a)?);
        // Floor: non-divisible gaps round down as in CuTe; the complement may overlap with the original layout at holes.
        let gap = d / current;
        if gap == 0 {
            return Err(Error::NonInjective);
        }
        // Gaps of length 1 are eliminated by final coalesce and do not consume leaf capacity here.
        if gap > 1 {
            prefix.push(gap, current)?;
        }
        current = mul(s, d)?;
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
/// use noodle_cute::algebra::complement::complement_shape;
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
        let size = cotarget.iter().try_fold(1, |acc, &c| mul(acc, c))?;
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
    leaves.layout()?.coalesce()
}

/// CuTe `complement(layout, cotarget)` with an integral cotarget.
///
/// Returns a layout `R` such that `(layout, R)` is injective and covers the stride lattice of
/// `[0, cotarget)` (rounded up).
/// Consistent with CuTe: stride-0 leaves are filtered first; non-divisible stride gaps round down;
/// all-broadcast layouts return `cotarget:1`.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::complement::complement;
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
/// terminal `1:cosize` mode marking where the next repetition would start.
///
/// Unlike `complement(layout, cosize)`, this does not extend up to a cotarget; product operations
/// rely on the terminal mode to deduce repetition strides.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::complement::complement_extended;
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
    let prefix = leaves.layout()?.coalesce()?;
    if prefix.size()? == 1 {
        return DynLayout::new(&[1], &[current]);
    }
    let mut out = Leaves::new();
    for (&s, &d) in prefix.shapes().iter().zip(prefix.strides()) {
        out.push(s, d)?;
    }
    out.push(1, current)?;
    out.layout()
}
