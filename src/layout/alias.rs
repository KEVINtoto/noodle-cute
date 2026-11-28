//! Address aliasing between two layouts over the same flat index domain.
//!
//! `compatible_aliases` certifies regular integer strides by their spans and otherwise compares
//! the input layouts directly. It allocates no storage and does not dereference their addresses.

use crate::integer::{add, int, sub, uint};
use crate::tuple::Item;

use super::Layout;

/// Proves that each active integer stride separates all lower-stride addresses.
/// A broadcast axis may be ignored when the aligned source axis is also broadcast.
const fn separated_writes<
    SS: [const] Item,
    DS: [const] Item,
    SD: [const] Item,
    DD: [const] Item,
>(
    src: Layout<SS, DS>,
    dst: Layout<SD, DD>,
) -> bool {
    let mut same_extents = SS::LEAVES == SD::LEAVES;
    let mut equal_strides = same_extents;
    let mut i = 0;
    while same_extents && i < SD::LEAVES {
        let source_extent = src.shape.integer_leaf(i);
        same_extents = source_extent.is_some() && source_extent == dst.shape.integer_leaf(i);
        let source_stride = src.stride.integer_leaf(i);
        equal_strides &= source_stride.is_some() && source_stride == dst.stride.integer_leaf(i);
        i = add(i, 1);
    }
    if same_extents && equal_strides {
        return true;
    }
    i = 0;
    while i < SD::LEAVES {
        let (Some(extent), Some(stride)) = (dst.shape.integer_leaf(i), dst.stride.integer_leaf(i))
        else {
            return false;
        };
        if extent > 1 {
            if stride == 0 {
                if !(same_extents && src.stride.integer_leaf(i) == Some(0)) {
                    return false;
                }
            } else {
                let step = stride.unsigned_abs();
                let mut span: uint = 0;
                let mut j = 0;
                while j < SD::LEAVES {
                    let (Some(n), Some(d)) =
                        (dst.shape.integer_leaf(j), dst.stride.integer_leaf(j))
                    else {
                        return false;
                    };
                    if j != i && n > 1 && d.unsigned_abs() <= step {
                        span =
                            span.saturating_add(d.unsigned_abs().saturating_mul(sub(n, 1) as uint));
                        if span >= step {
                            return false;
                        }
                    }
                    j = add(j, 1);
                }
            }
        }
        i = add(i, 1);
    }
    true
}

/// Checks whether equal destination offsets imply equal source offsets over an equal-sized domain.
pub(crate) const fn compatible_aliases<SS, DS, SD, DD, X, Y>(
    src: Layout<SS, DS>,
    dst: Layout<SD, DD>,
    size: int,
) -> bool
where
    SS: [const] Item,
    DS: [const] Item,
    SD: [const] Item,
    DD: [const] Item,
    Layout<SS, DS>: [const] Fn(int) -> X,
    Layout<SD, DD>: [const] Fn(int) -> Y,
    X: Copy + [const] PartialEq,
    Y: Copy + [const] PartialEq,
{
    if separated_writes(src, dst) {
        return true;
    }
    let mut i = 1;
    while i < size {
        let destination = dst(i);
        let mut j = 0;
        while j < i {
            if destination == dst(j) && src(i) != src(j) {
                return false;
            }
            j = add(j, 1);
        }
        i = add(i, 1);
    }
    true
}
