//! Flat layout descriptor suitable for constant evaluation and const generics.

use core::marker::ConstParamTy;

use crate::layout::{AffineLayoutMap, DynLayout, GroupedLayout, MAX_RANK, address_span};
use crate::{Error, Result};

/// A flat affine layout descriptor with fixed maximum rank.
///
/// Unlike [`DynLayout`], `LayoutDesc` derives [`ConstParamTy`], allowing it to be used
/// as an ADT const generic parameter. All inactive array entries past `rank` are kept at 0,
/// ensuring structural equality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub struct LayoutDesc {
    /// Extents of each active leaf; entries past `rank` are zero.
    pub shapes: [isize; MAX_RANK],
    /// Strides of each active leaf; entries past `rank` are zero.
    pub strides: [isize; MAX_RANK],
    /// Number of active leaves.
    pub rank: usize,
}

impl LayoutDesc {
    /// The empty rank-0 scalar layout descriptor.
    pub const EMPTY: Self = Self {
        shapes: [0; MAX_RANK],
        strides: [0; MAX_RANK],
        rank: 0,
    };

    /// Creates an empty rank-0 scalar layout descriptor.
    pub const fn empty() -> Self {
        Self::EMPTY
    }

    /// Creates an empty rank-0 scalar layout descriptor.
    pub const fn scalar() -> Self {
        Self::EMPTY
    }

    /// Creates a layout descriptor from flat shape and stride slices.
    pub const fn from_flat(shape: &[isize], stride: &[isize]) -> Result<Self> {
        if shape.len() != stride.len() {
            return Err(Error::StructureMismatch);
        }
        if shape.len() > MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        let mut desc = Self::empty();
        let mut i = 0;
        while i < shape.len() {
            if shape[i] < 0 {
                return Err(Error::InvalidShape);
            }
            desc.shapes[i] = shape[i];
            desc.strides[i] = stride[i];
            i += 1;
        }
        desc.rank = shape.len();
        Ok(desc)
    }

    /// Validates shape non-negativity and address arithmetic, returning a checked layout descriptor.
    pub const fn new(shape: &[isize], stride: &[isize]) -> Result<Self> {
        let desc = match Self::from_flat(shape, stride) {
            Ok(d) => d,
            Err(e) => return Err(e),
        };
        if let Err(e) = desc.size() {
            return Err(e);
        }
        if let Err(e) = desc.address_range() {
            return Err(e);
        }
        Ok(desc)
    }

    /// Appends a leaf to the layout descriptor.
    pub const fn push(mut self, shape: isize, stride: isize) -> Result<Self> {
        if self.rank >= MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        if shape < 0 {
            return Err(Error::InvalidShape);
        }
        self.shapes[self.rank] = shape;
        self.strides[self.rank] = stride;
        self.rank += 1;
        Ok(self)
    }

    /// Returns the number of active leaves.
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Returns a slice of the active shapes.
    pub fn shapes(&self) -> &[isize] {
        &self.shapes[..self.rank]
    }

    /// Returns a slice of the active strides.
    pub fn strides(&self) -> &[isize] {
        &self.strides[..self.rank]
    }

    /// Returns the shape extent of leaf `i`.
    pub const fn leaf_shape(&self, i: usize) -> Result<isize> {
        if i < self.rank {
            Ok(self.shapes[i])
        } else {
            Err(Error::AxisOutOfBounds)
        }
    }

    /// Returns the stride of leaf `i`.
    pub const fn leaf_stride(&self, i: usize) -> Result<isize> {
        if i < self.rank {
            Ok(self.strides[i])
        } else {
            Err(Error::AxisOutOfBounds)
        }
    }

    /// Computes the checked logical size of the layout.
    pub const fn size(&self) -> Result<isize> {
        let mut i = 0;
        while i < self.rank {
            if self.shapes[i] < 0 {
                return Err(Error::InvalidShape);
            }
            i += 1;
        }
        i = 0;
        while i < self.rank {
            if self.shapes[i] == 0 {
                return Ok(0);
            }
            i += 1;
        }
        let mut product = 1;
        i = 0;
        while i < self.rank {
            product = match crate::int::mul(product, self.shapes[i]) {
                Ok(p) => p,
                Err(e) => return Err(e),
            };
            i += 1;
        }
        Ok(product)
    }

    /// Computes the inclusive address bounds of the layout.
    pub const fn address_range(&self) -> Result<Option<(isize, isize)>> {
        let size = match self.size() {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        if size == 0 {
            return Ok(None);
        }
        let mut min = 0;
        let mut max = 0;
        let mut i = 0;
        while i < self.rank {
            let delta = match crate::int::mul(self.shapes[i] - 1, self.strides[i]) {
                Ok(d) => d,
                Err(e) => return Err(e),
            };
            if delta < 0 {
                min = match crate::int::add(min, delta) {
                    Ok(m) => m,
                    Err(e) => return Err(e),
                };
            } else {
                max = match crate::int::add(max, delta) {
                    Ok(m) => m,
                    Err(e) => return Err(e),
                };
            }
            i += 1;
        }
        Ok(Some((min, max)))
    }

    /// Computes the address span `max - min + 1` of the layout.
    pub const fn cosize(&self) -> Result<isize> {
        address_span(match self.address_range() {
            Ok(r) => r,
            Err(e) => return Err(e),
        })
    }

    /// Evaluates the address at logical linear index `linear`.
    pub const fn index(&self, mut linear: isize) -> Result<isize> {
        let size = match self.size() {
            Ok(s) => s,
            Err(e) => return Err(e),
        };
        if linear < 0 || linear >= size {
            return Err(Error::CoordinateOutOfBounds);
        }
        let mut result = 0;
        let mut i = 0;
        while i < self.rank {
            let s = self.shapes[i];
            let term = match crate::int::mul(linear % s, self.strides[i]) {
                Ok(t) => t,
                Err(e) => return Err(e),
            };
            result = match crate::int::add(result, term) {
                Ok(r) => r,
                Err(e) => return Err(e),
            };
            linear /= s;
            i += 1;
        }
        Ok(result)
    }

    /// Checks whether two layout descriptors are equal in a const context.
    pub const fn const_eq(&self, other: &Self) -> bool {
        if self.rank != other.rank {
            return false;
        }
        let mut i = 0;
        while i < self.rank {
            if self.shapes[i] != other.shapes[i] || self.strides[i] != other.strides[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Converts this descriptor to a [`DynLayout`].
    pub const fn to_dyn_layout(&self) -> DynLayout {
        DynLayout {
            shape: self.shapes,
            stride: self.strides,
            rank: self.rank,
        }
    }

    /// Creates a descriptor from a [`DynLayout`].
    pub const fn from_dyn_layout(d: &DynLayout) -> Self {
        let mut desc = Self::empty();
        let mut i = 0;
        while i < d.rank {
            desc.shapes[i] = d.shape[i];
            desc.strides[i] = d.stride[i];
            i += 1;
        }
        desc.rank = d.rank;
        desc
    }

    /// Coalesces this layout descriptor according to CuTe coalescing rules.
    pub const fn coalesce(&self, mode: crate::algebra::CoalesceMode) -> Result<Self> {
        crate::algebra::coalesce_desc(self, mode)
    }

    /// Filters stride-0 modes and coalesces.
    pub const fn filter(&self) -> Result<Self> {
        crate::algebra::filter_desc(self)
    }

    /// Composes this layout descriptor against leaf `s:d`.
    pub const fn compose_leaf(&self, s: isize, d: isize) -> Result<Self> {
        crate::algebra::compose_leaf_desc(self, s, d)
    }

    /// Composes this layout descriptor against an inner layout descriptor.
    pub const fn compose(&self, inner: &Self) -> Result<Self> {
        crate::algebra::compose_desc(self, inner)
    }

    /// Complements this layout descriptor with cotarget extents.
    pub const fn complement(
        &self,
        cotarget: &[isize],
        mode: crate::algebra::ComplementMode,
    ) -> Result<Self> {
        crate::algebra::complement_desc(self, cotarget, mode)
    }

    /// Complements this layout descriptor in extended repetition mode.
    pub const fn complement_extended(&self) -> Result<Self> {
        crate::algebra::complement_desc(self, &[], crate::algebra::ComplementMode::Extended)
    }
}

impl AffineLayoutMap for LayoutDesc {
    fn affine_rank(&self) -> usize {
        self.rank
    }

    fn leaf_shape(&self, leaf: usize) -> Result<isize> {
        self.leaf_shape(leaf)
    }

    fn leaf_stride(&self, leaf: usize) -> Result<isize> {
        self.leaf_stride(leaf)
    }

    fn leaf_count(&self) -> usize {
        self.rank
    }

    fn mode_span(&self, axis: usize) -> Result<(usize, usize)> {
        if axis < self.rank {
            Ok((axis, 1))
        } else {
            Err(Error::AxisOutOfBounds)
        }
    }

    fn as_flat_layout(&self) -> Result<DynLayout> {
        Ok(self.to_dyn_layout())
    }

    fn as_grouped_layout(&self) -> Result<GroupedLayout> {
        GroupedLayout::from_flat(self.to_dyn_layout())
    }
}

impl From<DynLayout> for LayoutDesc {
    fn from(d: DynLayout) -> Self {
        Self::from_dyn_layout(&d)
    }
}

impl From<LayoutDesc> for DynLayout {
    fn from(desc: LayoutDesc) -> Self {
        desc.to_dyn_layout()
    }
}

/// Trait for static layouts that can provide their [`LayoutDesc`] at compile time.
pub trait StaticDesc {
    /// The compile-time layout descriptor.
    const DESC: LayoutDesc;
}
