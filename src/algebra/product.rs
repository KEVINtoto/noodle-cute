//! Product layout: combines a tile address with the rest address mapped
//! through the complement of the tile.

use crate::layout::{AffineLayoutMap, DynLayout, GroupedLayout, LayoutMap, MAX_RANK};
use crate::{Error, Result};

use super::complement::complement_extended;
use super::composition::{Composition, composition, composition_layout};
use super::{Grouping, Leaves, TileRest};

/// Product layout formed by adding a tile address to a rest address mapped through the complement of the tile.
pub type Product<A, B> = TileRest<A, Composition<DynLayout, B>>;

// Construct a product layout from a block and a tiler layout, returning a composed layout.
fn product<A: AffineLayoutMap, B: LayoutMap>(
    block: A,
    tiler: B,
    grouping: Grouping,
) -> Result<Product<A, B>> {
    // CuTe: logical_product(block, tiler) = (block, complement(block) o tiler), with complement
    // having no cotarget; rest values exceeding the complement's logical domain extend into its terminal 1:cosize mode.
    let complement = complement_extended(block.clone())?;
    let rest = composition(complement, tiler)?;
    let result = TileRest::new(block, rest, grouping);
    result.size()?;
    Ok(result)
}

/// CuTe `logical_product(block, tiler)`: produces two groups `(block, complement(block) ∘ tiler)`.
///
/// `block` must satisfy complement constraints (nonnegative strides, non-overlapping).
/// `tiler` coordinates must be nonnegative. The resulting size is `block.size() * tiler.size()`.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::product::logical_product;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let block = DynLayout::new(&[2], &[1])?;
/// let tiler = DynLayout::new(&[3], &[1])?;
/// let prod = logical_product(block, tiler)?;
/// assert_eq!(prod.size()?, 6);
/// assert_eq!(prod.rank(), 2);
/// assert_eq!(prod.index(5)?, 5);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// - Returns [`Error::Unsupported`] if `block` has negative strides or size 0.
/// - Returns [`Error::NonInjective`] if `block` strides overlap.
/// - Returns [`Error::CoordinateOutOfBounds`] if `tiler` produces negative addresses.
/// - Returns [`Error::Overflow`] if size computation overflows.
pub fn logical_product<A: AffineLayoutMap, B: LayoutMap>(
    block: A,
    tiler: B,
) -> Result<Product<A, B>> {
    product(block, tiler, Grouping::Logical)
}

/// CuTe `zipped_product(block, tiler)`: for a single block layout, zipped product produces the same two groups as logical product.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::product::zipped_product;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let block = DynLayout::new(&[2], &[1])?;
/// let tiler = DynLayout::new(&[3], &[1])?;
/// let prod = zipped_product(block, tiler)?;
/// assert_eq!(prod.size()?, 6);
/// assert_eq!(prod.rank(), 2);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`logical_product`].
pub fn zipped_product<A: AffineLayoutMap, B: LayoutMap>(
    block: A,
    tiler: B,
) -> Result<Product<A, B>> {
    product(block, tiler, Grouping::Zipped)
}

/// CuTe `tiled_product(block, tiler)`: groups `block` into mode 0 and exposes each axis of `tiler` individually.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::product::tiled_product;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let block = DynLayout::new(&[2], &[1])?;
/// let tiler = DynLayout::new(&[3], &[1])?;
/// let prod = tiled_product(block, tiler)?;
/// assert_eq!(prod.size()?, 6);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`logical_product`].
pub fn tiled_product<A: AffineLayoutMap, B: LayoutMap>(
    block: A,
    tiler: B,
) -> Result<Product<A, B>> {
    product(block, tiler, Grouping::Tiled)
}

/// CuTe `flat_product(block, tiler)`: flattens all axes of `block` followed by all axes of `tiler`.
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::product::flat_product;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let block = DynLayout::new(&[2], &[1])?;
/// let tiler = DynLayout::new(&[3], &[1])?;
/// let prod = flat_product(block, tiler)?;
/// assert_eq!(prod.size()?, 6);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`logical_product`].
pub fn flat_product<A: AffineLayoutMap, B: LayoutMap>(block: A, tiler: B) -> Result<Product<A, B>> {
    product(block, tiler, Grouping::Flat)
}

// CuTe blocked/raked product: pads both sides to max(rank) with unit 1:0 modes,
// then zips the two groups of logical_product mode-by-mode; raked places rest before block in each mode.
fn rank_product<A: AffineLayoutMap, B: AffineLayoutMap>(
    block: &A,
    tiler: &B,
    raked: bool,
) -> Result<GroupedLayout> {
    let block = block.as_grouped_layout()?;
    let tiler = tiler.as_grouped_layout()?;
    if let Some((min, _)) = tiler.address_range()?
        && min < 0
    {
        return Err(Error::CoordinateOutOfBounds);
    }
    let rank = block.rank().max(tiler.rank());
    // Padded 1:0 modes are filtered out by complement, not affecting the complement layout.
    let complement = complement_extended(block)?;
    let unit = DynLayout::new(&[1], &[0])?;
    let mut leaves = Leaves::new();
    let mut groups = [0usize; MAX_RANK];
    for (m, group) in groups.iter_mut().enumerate().take(rank) {
        let b = if m < block.rank() {
            block.mode(m)?
        } else {
            unit
        };
        // CuTe composition on a stride-0 rhs produces s:0 directly; padding modes evaluate to 1:0.
        let r = if m < tiler.rank() {
            let mode = GroupedLayout::from_flat(tiler.mode(m)?)?;
            *composition_layout(&complement, &mode)?.flat()
        } else {
            unit
        };
        let (first, second) = if raked { (r, b) } else { (b, r) };
        for part in [first, second] {
            for (&s, &d) in part.shapes().iter().zip(part.strides()) {
                leaves.push(s, d)?;
            }
        }
        *group = first.rank() + second.rank();
    }
    GroupedLayout::new(leaves.layout()?, &groups[..rank])
}

/// CuTe `blocked_product(block, tiler)`: rank is `max(block.rank(), tiler.rank())`, where the `m`-th mode is
/// `(block_m, (complement(block) ∘ tiler)_m)`, repeating the entire block along each mode.
///
/// Missing modes are padded with `1:0`; these shape-1 leaves are retained to match CuTe's mode structure.
/// `tiler` addresses must be nonnegative, and each mode must satisfy the divisibility requirements of [`composition_layout`].
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::product::blocked_product;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let block = DynLayout::new(&[2], &[1])?;
/// let tiler = DynLayout::new(&[3], &[1])?;
/// let bp = blocked_product(block, tiler)?;
/// assert_eq!(bp.rank(), 1);
/// assert_eq!(bp.shapes(), &[2, 3]);
/// assert_eq!(bp.strides(), &[1, 2]);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// - Returns [`Error::CoordinateOutOfBounds`] if `tiler` has negative addresses.
/// - Returns [`Error::NotDivisible`] if mode composition divisibility conditions fail.
/// - Returns [`Error::CapacityExceeded`] if leaf or group storage exceeds [`MAX_RANK`].
pub fn blocked_product<A: AffineLayoutMap, B: AffineLayoutMap>(
    block: A,
    tiler: B,
) -> Result<GroupedLayout> {
    rank_product(&block, &tiler, false)
}

/// CuTe `raked_product(block, tiler)`: has the same leaves as [`blocked_product`], but each mode is
/// ordered as `(rest_m, block_m)`, interleaving (raking) elements of `block` across modes.
///
/// `tiler` addresses must be nonnegative, and each mode must satisfy the divisibility requirements of [`composition_layout`].
///
/// # Examples
///
/// ```
/// use noodle_cute::algebra::product::raked_product;
/// use noodle_cute::layout::{DynLayout, LayoutMap};
///
/// let block = DynLayout::new(&[2], &[1])?;
/// let tiler = DynLayout::new(&[3], &[1])?;
/// let rp = raked_product(block, tiler)?;
/// assert_eq!(rp.rank(), 1);
/// assert_eq!(rp.shapes(), &[3, 2]);
/// assert_eq!(rp.strides(), &[2, 1]);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
///
/// # Errors
///
/// See [`blocked_product`].
pub fn raked_product<A: AffineLayoutMap, B: AffineLayoutMap>(
    block: A,
    tiler: B,
) -> Result<GroupedLayout> {
    rank_product(&block, &tiler, true)
}
