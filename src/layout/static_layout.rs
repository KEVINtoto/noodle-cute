//! Typed affine layouts with static or mixed static/runtime leaves.

use core::fmt::Debug;
use core::ops::Mul;

use crate::int::{Int, Integer, mul};
use crate::tuple::{Flatten, Item, Nil, Tuple};
use crate::{Error, IntTuple, Result};

use super::{AffineLayoutMap, DynLayout, GroupedLayout, LayoutMap, MAX_RANK, Shape};

/// An affine map defined by a shape and structurally congruent strides.
///
/// The shape and stride profiles must match at the type level, including nested
/// modes. Each leaf contributes `coordinate * stride` to the address. Static
/// leaves occupy no storage; runtime leaves retain their values in the layout.
/// Negative and zero strides are allowed; [`LayoutMap::is_injective`] checks
/// whether distinct logical coordinates can share an address.
///
/// Public fields allow construction without validation. Use [`Layout::new`] to
/// check the shape and address arithmetic. The coalescing fold also builds
/// unchecked `Layout { shape, stride }` intermediates before validating its result.
///
/// ```
/// use noodle_cute::{DynInt, Int, shape, stride};
/// use noodle_cute::layout::{Layout, LayoutMap};
///
/// let layout = Layout::new(
///     shape![Int::<2>, DynInt(3)],
///     stride![Int::<1>, Int::<2>],
/// )?;
/// assert_eq!(layout.size()?, 6);
/// assert_eq!(layout.index_of_coord(&[1, 2])?, 5);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout<S, D> {
    /// Nonnegative extents arranged as an integer leaf or nested tuple.
    pub shape: S,
    /// Signed address increments with the same nested profile as `shape`.
    pub stride: D,
}

/// 类型化代数结果的 shape/stride 接口，支持尚未校验的中间布局。
#[doc(hidden)]
pub trait LayoutParts {
    type Shape;
    type Stride;
    /// 返回布局值，不执行校验或归一。
    fn into_layout(self) -> Layout<Self::Shape, Self::Stride>;
}
impl<S, D> LayoutParts for Layout<S, D> {
    type Shape = S;
    type Stride = D;
    fn into_layout(self) -> Self {
        self
    }
}

impl<S: Shape, D: Item<Profile = S::Profile>> Layout<S, D> {
    /// Constructs a layout after checking nonnegative extents, logical size,
    /// and reachable address endpoints with checked arithmetic.
    ///
    /// Negative extents return [`Error::InvalidShape`]; unrepresentable size or
    /// address calculations return [`Error::Overflow`]. Addresses may be negative
    /// or repeated, and an empty domain has no address endpoints to check.
    pub fn new(shape: S, stride: D) -> Result<Self> {
        let layout = Self { shape, stride };
        layout.size()?;
        layout.address_range()?;
        Ok(layout)
    }

    /// Flattens nested modes in leaf order without changing the linear address map.
    ///
    /// Static leaf values remain encoded in the result type; flattening neither
    /// merges contiguous leaves nor removes unit extents.
    pub fn flatten(self) -> Layout<<S as Flatten>::Output, <D as Flatten>::Output>
    where
        S: Flatten,
        D: Flatten,
    {
        Layout {
            shape: self.shape.flatten(),
            stride: self.stride.flatten(),
        }
    }
}

impl<S: Shape, D: Item<Profile = S::Profile>> AffineLayoutMap for Layout<S, D> {
    fn affine_rank(&self) -> usize {
        S::RANK
    }

    fn shape(&self, leaf: usize) -> Result<isize> {
        self.shape.leaf(leaf)
    }

    fn stride(&self, leaf: usize) -> Result<isize> {
        self.stride.leaf(leaf)
    }

    fn leaves(&self) -> core::ops::Range<usize> {
        0..S::LEAVES
    }

    fn mode_span(&self, axis: usize) -> Result<(usize, usize)> {
        self.shape.mode_span(axis)
    }

    /// Exports flattened runtime metadata for reflection or backend parameters.
    ///
    /// The result has one axis per leaf and discards the original mode grouping.
    /// Returns [`Error::CapacityExceeded`] if the leaf count exceeds [`MAX_RANK`].
    /// The exported values are validated by [`DynLayout::new`].
    fn as_flat_layout(&self) -> Result<DynLayout> {
        if S::LEAVES > MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        let mut shape = [0; MAX_RANK];
        let mut stride = [0; MAX_RANK];
        for i in 0..S::LEAVES {
            shape[i] = self.shape.leaf(i)?;
            stride[i] = self.stride.leaf(i)?;
        }
        DynLayout::new(&shape[..S::LEAVES], &stride[..S::LEAVES])
    }

    fn as_grouped_layout(&self) -> Result<GroupedLayout> {
        if S::RANK > MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        let mut groups = [0usize; MAX_RANK];
        for (axis, g) in groups[..S::RANK].iter_mut().enumerate() {
            *g = self.shape.mode_span(axis)?.1;
        }
        let mut grouped = GroupedLayout::new(self.as_flat_layout()?, &groups[..S::RANK])?;
        // A leaf type (depth zero) represents a CuTe integral shape, distinct
        // from a one-element tuple even though both have rank one.
        grouped.integral = S::DEPTH == 0;
        Ok(grouped)
    }
}

//-----------------------------------------------------------------------------
// Compact stride construction.

/// Constructs typed compact layouts with the leftmost leaf varying fastest.
/// Also serves as the direction marker for [`Compact`].
#[derive(Clone, Copy, Debug)]
pub struct LayoutLeft;

/// Constructs typed compact layouts with the rightmost leaf varying fastest.
/// Also serves as the direction marker for [`Compact`].
#[derive(Clone, Copy, Debug)]
pub struct LayoutRight;

/// Builds compact strides while preserving tuple structure and static arithmetic.
///
/// `Dir` is [`LayoutLeft`] or [`LayoutRight`] and determines the order in which
/// leaf extents multiply the running stride. The result is
/// `(stride_tree, start * product_of_extents)`. This low-level operation uses
/// ordinary integer multiplication; use the direction marker's `make` method
/// for checked layout construction.
pub trait Compact<Dir, Start: Integer>: Item {
    /// Stride tree with the same nested structure as the shape.
    type Stride: Item<Profile = Self::Profile>;
    /// Running stride after all extents have been multiplied into `Start`.
    type End: Integer;
    /// Assigns strides beginning at `start` and returns the final running stride.
    fn compact(self, start: Start) -> (Self::Stride, Self::End);
}

// Leaves and empty tuples are direction-independent; only `IntTuple` changes
// its recursive evaluation order with the direction marker.
impl<I, Dir, Start> Compact<Dir, Start> for I
where
    I: Integer + Debug + Mul<Start>,
    <I as Mul<Start>>::Output: Integer,
    Start: Integer + Debug,
{
    type Stride = Start;
    type End = <I as Mul<Start>>::Output;
    fn compact(self, start: Start) -> (Start, Self::End) {
        (start, self * start)
    }
}

impl<Dir, Start: Integer> Compact<Dir, Start> for Nil {
    type Stride = Nil;
    type End = Start;
    fn compact(self, start: Start) -> (Nil, Start) {
        (Nil, start)
    }
}

impl<H, T, Start> Compact<LayoutLeft, Start> for IntTuple<H, T>
where
    Start: Integer,
    H: Compact<LayoutLeft, Start>,
    T: Tuple + Compact<LayoutLeft, H::End>,
    T::Stride: Tuple,
{
    type Stride = IntTuple<H::Stride, T::Stride>;
    type End = T::End;
    fn compact(self, start: Start) -> (Self::Stride, Self::End) {
        let (h, end) = self.0.compact(start);
        let (t, end) = self.1.compact(end);
        (IntTuple(h, t), end)
    }
}

impl<H, T, Start> Compact<LayoutRight, Start> for IntTuple<H, T>
where
    Start: Integer,
    T: Tuple + Compact<LayoutRight, Start>,
    H: Compact<LayoutRight, T::End>,
    T::Stride: Tuple,
{
    type Stride = IntTuple<H::Stride, T::Stride>;
    type End = H::End;
    fn compact(self, start: Start) -> (Self::Stride, Self::End) {
        let (t, end) = self.1.compact(start);
        let (h, end) = self.0.compact(end);
        (IntTuple(h, t), end)
    }
}

impl LayoutLeft {
    /// Builds a compact layout with left-to-right cumulative strides.
    ///
    /// Validates extents and partial products before constructing strides,
    /// retaining static values in their types. The first leaf has stride one.
    ///
    /// ```
    /// use noodle_cute::{Int, shape};
    /// use noodle_cute::layout::{AffineLayoutMap, LayoutLeft, LayoutMap};
    ///
    /// let layout = LayoutLeft::make(shape![Int::<2>, (Int::<3>, Int::<4>)])?;
    /// assert_eq!(layout.rank(), 2);
    /// assert_eq!(layout.extent(1)?, 12);
    /// assert_eq!(layout.as_flat_layout()?.strides(), &[1, 2, 6]);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn make<S: Shape + Compact<Self, Int<1>>>(shape: S) -> Result<Layout<S, S::Stride>> {
        make_compact(shape, 0..S::LEAVES)
    }
}

impl LayoutRight {
    /// Builds a compact layout with right-to-left cumulative strides.
    ///
    /// Checks extents and partial products for overflow. The last leaf has
    /// stride one; logical linear indices still use left-major order.
    pub fn make<S: Shape + Compact<Self, Int<1>>>(shape: S) -> Result<Layout<S, S::Stride>> {
        make_compact(shape, (0..S::LEAVES).rev())
    }
}

// Typed stride construction uses ordinary multiplication. Reject negative
// extents first, then preflight partial products in the same leaf order so
// overflow before a zero extent returns an error rather than panicking, even
// when the final logical size is zero.
fn make_compact<Dir, S: Shape + Compact<Dir, Int<1>>>(
    shape: S,
    order: impl Iterator<Item = usize>,
) -> Result<Layout<S, S::Stride>> {
    for i in 0..S::LEAVES {
        if shape.leaf(i)? < 0 {
            return Err(Error::InvalidShape);
        }
    }
    let mut scale = 1;
    for i in order {
        scale = mul(scale, shape.leaf(i)?)?;
    }
    let (stride, _) = shape.clone().compact(Int);
    Layout::new(shape, stride)
}
