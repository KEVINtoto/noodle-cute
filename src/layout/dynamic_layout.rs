use crate::{Error, Result};

use super::{
    LayoutMap, MAX_RANK, Slice, add, affine_address_range, affine_index, affine_index_extended,
    affine_injective, checked_size, injective_by_enumeration, mul,
};

/// An allocation-free affine layout with runtime extents and strides.
///
/// Every axis is a leaf, and at most [`MAX_RANK`] leaves can be stored. An empty
/// shape describes a scalar with size 1 and address 0; any zero extent makes the
/// logical domain empty. Only the first `rank` entries of each array are active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynLayout {
    pub(crate) shape: [isize; MAX_RANK],
    pub(crate) stride: [isize; MAX_RANK],
    pub(crate) rank: usize,
}

impl DynLayout {
    /// Creates the rank-0 scalar used as the starting point for layout builders.
    pub(crate) const fn scalar() -> Self {
        Self {
            shape: [0; MAX_RANK],
            stride: [0; MAX_RANK],
            rank: 0,
        }
    }

    /// Creates a layout from equally sized shape and stride slices.
    ///
    /// Extents must be nonnegative, and the rank must not exceed `MAX_RANK`.
    /// Construction checks for overflow in the logical size and reachable
    /// address bounds. Zero and negative strides are allowed; overlapping
    /// addresses are accepted and can be checked with `LayoutMap::is_injective`.
    pub fn new(shape: &[isize], stride: &[isize]) -> Result<Self> {
        if shape.len() != stride.len() {
            return Err(Error::StructureMismatch);
        }
        if shape.len() > MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        let mut out = Self::scalar();
        out.rank = shape.len();
        out.shape[..shape.len()].copy_from_slice(shape);
        out.stride[..stride.len()].copy_from_slice(stride);
        out.size()?;
        out.address_range()?;
        Ok(out)
    }

    /// Creates a compact layout whose leftmost axis has stride 1.
    /// Each subsequent stride is the product of the extents to its left.
    pub fn left(shape: &[isize]) -> Result<Self> {
        Self::compact(shape, false)
    }

    /// Creates a compact layout whose rightmost axis has stride 1.
    /// Each preceding stride is the product of the extents to its right.
    /// `LayoutMap::index` still decodes logical indices with the leftmost axis
    /// varying fastest, regardless of this physical storage order.
    pub fn right(shape: &[isize]) -> Result<Self> {
        Self::compact(shape, true)
    }

    fn compact(shape: &[isize], right: bool) -> Result<Self> {
        if shape.len() > MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        let mut strides = [0; MAX_RANK];
        let mut scale = 1;
        for j in 0..shape.len() {
            let i = if right { shape.len() - 1 - j } else { j };
            if shape[i] < 0 {
                return Err(Error::InvalidShape);
            }
            strides[i] = scale;
            scale = mul(scale, shape[i])?;
        }
        Self::new(shape, &strides[..shape.len()])
    }

    /// Returns the active leaf extents in logical axis order.
    pub fn shape(&self) -> &[isize] {
        &self.shape[..self.rank]
    }

    /// Returns the signed address strides corresponding to `Self::shape`.
    pub fn stride(&self) -> &[isize] {
        &self.stride[..self.rank]
    }

    /// Selects a sublayout using exactly one selector per axis.
    ///
    /// Fixed axes are removed; `Slice::All` axes retain their extents and
    /// strides. The returned offset is the address contribution of the fixed
    /// coordinates and must be added when accessing the original storage.
    /// Fixed coordinates are checked and their offset is computed even when a
    /// retained axis is empty. Out-of-bounds coordinates and offset overflow
    /// return errors.
    pub fn slice(&self, selectors: &[Slice]) -> Result<(Self, isize)> {
        if selectors.len() != self.rank {
            return Err(Error::StructureMismatch);
        }
        let mut result = Self::scalar();
        let mut offset = 0;
        for (i, s) in selectors.iter().enumerate() {
            match *s {
                Slice::All => {
                    let j = result.rank;
                    result.rank += 1;
                    result.shape[j] = self.shape[i];
                    result.stride[j] = self.stride[i];
                }
                Slice::Index(c) => {
                    if c < 0 || c >= self.shape[i] {
                        return Err(Error::CoordinateOutOfBounds);
                    }
                    offset = add(offset, mul(c, self.stride[i])?)?;
                }
            }
        }
        Ok((result, offset))
    }

    /// CuTe `coalesce(layout)`: drops size-1 leaves and merges adjacent leaves
    /// when `s0 * d0 == d1`. A scalar or all-size-1 layout becomes `1:0`.
    ///
    /// The logical size and mapping on `[0, size)` are preserved. The extended
    /// mapping can change when a trailing size-1 leaf is removed.
    pub fn coalesce(&self) -> Result<Self> {
        let mut out = Self::scalar();
        for a in 0..self.rank {
            let (s, d) = (self.shape[a], self.stride[a]);
            if s == 1 {
                continue;
            }
            if out.rank > 0 {
                let j = out.rank - 1;
                if mul(out.shape[j], out.stride[j])? == d {
                    out.shape[j] = mul(out.shape[j], s)?;
                    continue;
                }
            }
            out.shape[out.rank] = s;
            out.stride[out.rank] = d;
            out.rank += 1;
        }
        if out.rank == 0 {
            out.rank = 1;
            out.shape[0] = 1;
            out.stride[0] = 0;
        }
        Ok(out)
    }

    /// CuTe `detail::coalesce_x`: coalesces leaves while retaining the stride
    /// that controls extension beyond the logical domain.
    ///
    /// A trailing size-1 leaf is assigned a placeholder extent of 2, because
    /// extended indexing never reads the final leaf's extent. Thus
    /// `(2,1):(2,8)` becomes `(2,2):(2,8)` instead of `2:2`. The placeholder can
    /// change the logical size; this helper is intended for extended indexing.
    /// A rank-0 scalar is normalized to `1:0`, which additionally accepts
    /// positive extended indices and maps them to address 0.
    #[allow(dead_code)]
    pub(crate) fn coalesce_extended(&self) -> Result<Self> {
        let r = self.rank;
        if r == 0 {
            return self.coalesce();
        }
        // Merge from right to left (CuTe bw_coalesce), storing leaves in reverse
        // order until both metadata arrays are reversed below.
        let mut out = Self::scalar();
        out.rank = 1;
        out.shape[0] = if self.shape[r - 1] == 1 {
            2
        } else {
            self.shape[r - 1]
        };
        out.stride[0] = self.stride[r - 1];
        for a in (0..r - 1).rev() {
            let (s, d) = (self.shape[a], self.stride[a]);
            if s == 1 {
                continue;
            }
            let j = out.rank - 1;
            if mul(s, d)? == out.stride[j] {
                out.shape[j] = mul(s, out.shape[j])?;
                out.stride[j] = d;
                continue;
            }
            out.shape[out.rank] = s;
            out.stride[out.rank] = d;
            out.rank += 1;
        }
        out.shape[..out.rank].reverse();
        out.stride[..out.rank].reverse();
        Ok(out)
    }

    /// CuTe `filter(layout)`: treats stride-0 leaves as size 1, then coalesces.
    ///
    /// For a nonempty domain, this preserves the set of reachable addresses,
    /// but may change the logical size and coordinate mapping. Replacing a
    /// zero extent on a stride-0 leaf can also make an empty domain nonempty.
    pub fn filter(&self) -> Result<Self> {
        let mut zeros = *self;
        for a in 0..zeros.rank {
            if zeros.stride[a] == 0 {
                zeros.shape[a] = 1;
            }
        }
        zeros.coalesce()
    }
}

impl LayoutMap for DynLayout {
    fn size(&self) -> Result<isize> {
        checked_size(0..self.rank, |a| Ok(self.shape[a]))
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn extent(&self, a: usize) -> Result<isize> {
        self.shape().get(a).copied().ok_or(Error::AxisOutOfBounds)
    }

    fn index(&self, linear: isize) -> Result<isize> {
        affine_index(
            0..self.rank,
            |a| Ok(self.shape[a]),
            |a| Ok(self.stride[a]),
            linear,
        )
    }

    fn index_extended(&self, linear: isize) -> Result<isize> {
        self.size()?;
        affine_index_extended(
            0..self.rank,
            |i| Ok(self.shape[i]),
            |i| Ok(self.stride[i]),
            linear,
        )
    }

    fn index_extended_of_coord(&self, coord: &[isize]) -> Result<isize> {
        self.size()?;
        if coord.len() != self.rank {
            return Err(Error::StructureMismatch);
        }
        // Each top-level axis consists of a single leaf, which is therefore
        // its mode's final leaf. Extended coordinates are never reduced modulo
        // the corresponding extent.
        let mut result = 0;
        for (a, &c) in coord.iter().enumerate() {
            if c < 0 {
                return Err(Error::CoordinateOutOfBounds);
            }
            result = add(result, mul(c, self.stride[a])?)?;
        }
        Ok(result)
    }

    fn as_affine(&self) -> Result<Option<GroupedLayout>> {
        GroupedLayout::from_flat(*self).map(Some)
    }

    fn address_range(&self) -> Result<Option<(isize, isize)>> {
        affine_address_range(0..self.rank, |a| Ok(self.shape[a]), |a| Ok(self.stride[a]))
    }

    fn is_injective(&self) -> Result<bool> {
        let proof = affine_injective(0..self.rank, |a| Ok(self.shape[a]), |a| Ok(self.stride[a]))?;
        // The fast proof uses a sufficient condition. Fall back to exact
        // enumeration when it cannot decide an irregular stride pattern.
        match proof {
            Some(injective) => Ok(injective),
            None => injective_by_enumeration(self),
        }
    }
}

/// Runtime affine leaves grouped into top-level modes, as used by CuTe layout
/// algebra operations such as composition, complement, divide, and product.
///
/// Each mode is linearized with its leftmost leaf varying fastest, matching
/// `super::Layout`'s nested modes; deeper nesting is flattened. Both the number
/// of leaves and the number of top-level modes are bounded by `MAX_RANK`.
/// The `integral` flag distinguishes a CuTe integer shape from a tuple shape;
/// an integral shape has exactly one mode containing one leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupedLayout {
    pub(crate) flat: DynLayout,
    pub(crate) groups: [u8; MAX_RANK],
    pub(crate) rank: usize,
    pub(crate) integral: bool,
}

impl GroupedLayout {
    /// Groups consecutive leaves of `flat` into top-level tuple modes.
    ///
    /// `groups[a]` gives the leaf count of mode `a`; the counts must sum to
    /// `flat.rank()`, and there may be at most `MAX_RANK` modes. A mode with no
    /// leaves represents an empty tuple with extent 1. This constructor always
    /// records a tuple shape, including when there is just one mode.
    pub fn new(flat: DynLayout, groups: &[usize]) -> Result<Self> {
        if groups.len() > MAX_RANK {
            return Err(Error::CapacityExceeded);
        }
        if groups.iter().sum::<usize>() != flat.rank {
            return Err(Error::StructureMismatch);
        }
        let mut out = Self {
            flat,
            groups: [0; MAX_RANK],
            rank: groups.len(),
            integral: false,
        };
        for (g, &n) in out.groups.iter_mut().zip(groups) {
            *g = n as u8;
        }
        Ok(out)
    }

    /// Makes each flat leaf a separate top-level mode.
    /// Rank 1 is recorded as a CuTe integer shape, matching `make_layout(n)`.
    pub fn from_flat(flat: DynLayout) -> Result<Self> {
        let groups = [1usize; MAX_RANK];
        let mut out = Self::new(flat, &groups[..flat.rank])?;
        out.integral = flat.rank == 1;
        Ok(out)
    }

    /// Returns the flat affine metadata for all leaves.
    pub fn flat(&self) -> &DynLayout {
        &self.flat
    }

    /// Returns leaf extents; use `LayoutMap::extent` for top-level mode sizes.
    pub fn shape(&self) -> &[isize] {
        self.flat.shape()
    }

    /// Returns the signed address stride of each flat leaf.
    pub fn stride(&self) -> &[isize] {
        self.flat.stride()
    }

    /// Iterates over the leaf count of each top-level mode.
    pub fn groups(&self) -> impl Iterator<Item = usize> + '_ {
        self.groups[..self.rank].iter().map(|&g| g as usize)
    }

    /// Reports whether the shape is a CuTe integer rather than a tuple.
    pub fn is_integral(&self) -> bool {
        self.integral
    }

    /// Returns the starting leaf index and leaf count of a top-level mode.
    pub(crate) fn mode_span(&self, axis: usize) -> Result<(usize, usize)> {
        if axis >= self.rank {
            return Err(Error::AxisOutOfBounds);
        }
        let start = self.groups[..axis].iter().map(|&g| g as usize).sum();
        Ok((start, self.groups[axis] as usize))
    }

    /// Returns the flat sublayout of top-level mode `axis`.
    /// An empty mode becomes a rank-0 scalar; an invalid axis returns an error.
    pub fn mode(&self, axis: usize) -> Result<DynLayout> {
        let (start, len) = self.mode_span(axis)?;
        DynLayout::new(
            &self.flat.shape()[start..start + len],
            &self.flat.stride()[start..start + len],
        )
    }
}
impl LayoutMap for GroupedLayout {
    fn size(&self) -> Result<isize> {
        self.flat.size()
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn extent(&self, axis: usize) -> Result<isize> {
        self.mode(axis)?.size()
    }

    fn index(&self, linear: isize) -> Result<isize> {
        self.flat.index(linear)
    }

    fn index_extended(&self, linear: isize) -> Result<isize> {
        self.flat.index_extended(linear)
    }

    fn index_extended_of_coord(&self, coord: &[isize]) -> Result<isize> {
        if coord.len() != self.rank {
            return Err(Error::StructureMismatch);
        }
        let mut result = 0;
        for (axis, &c) in coord.iter().enumerate() {
            result = add(result, self.mode(axis)?.index_extended(c)?)?;
        }
        Ok(result)
    }

    fn as_affine(&self) -> Result<Option<GroupedLayout>> {
        Ok(Some(*self))
    }

    fn address_range(&self) -> Result<Option<(isize, isize)>> {
        self.flat.address_range()
    }

    fn is_injective(&self) -> Result<bool> {
        self.flat.is_injective()
    }
}
