//! 布局代数契约共用的布局构造、断言与非仿射测试映射。

use noodle_cute::algebra::{
    Complement, ComplementMode, Grouping, InverseSide, ProductKind, TypedCompose, TypedDivide,
    TypedInvert, TypedProduct,
};
use noodle_cute::layout::{DynLayout, GroupedLayout, LayoutMap};
use noodle_cute::{Error, Nil, Result, StaticValue};

// Constant results of typed algebra on fully static inputs, inferred from the arguments.

pub const fn composed<A: TypedCompose<B> + Copy, B: Copy>(_: A, _: B) -> A::Output
where
    A::Output: StaticValue,
{
    <A::Output as StaticValue>::VALUE
}

pub const fn complemented<L: Complement<C> + Copy, C: Copy>(_: L, _: C) -> L::Output
where
    L::Output: StaticValue,
{
    <L::Output as StaticValue>::VALUE
}

pub const fn complemented_extended<L: Complement<Nil, { ComplementMode::Extended }> + Copy>(
    _: L,
) -> L::Output
where
    L::Output: StaticValue,
{
    <L::Output as StaticValue>::VALUE
}

pub const fn divided<const GROUPING: Grouping, L: TypedDivide<T, GROUPING> + Copy, T: Copy>(
    _: L,
    _: T,
) -> L::Output
where
    L::Output: StaticValue,
{
    <L::Output as StaticValue>::VALUE
}

pub const fn producted<const KIND: ProductKind, L: TypedProduct<T, KIND> + Copy, T: Copy>(
    _: L,
    _: T,
) -> L::Output
where
    L::Output: StaticValue,
{
    <L::Output as StaticValue>::VALUE
}

pub const fn inverted<const SIDE: InverseSide, L: TypedInvert<SIDE> + Copy>(_: L) -> L::Output
where
    L::Output: StaticValue,
{
    <L::Output as StaticValue>::VALUE
}

pub fn dyn_layout(shape: &[isize], stride: &[isize]) -> DynLayout {
    DynLayout::new(shape, stride).unwrap()
}

pub fn grouped(shape: &[isize], stride: &[isize], groups: &[usize]) -> GroupedLayout {
    GroupedLayout::new(dyn_layout(shape, stride), groups).unwrap()
}

pub fn assert_grouped(actual: &GroupedLayout, shape: &[isize], stride: &[isize], groups: &[usize]) {
    assert_eq!(actual.shapes(), shape, "shape of {actual:?}");
    assert_eq!(actual.strides(), stride, "stride of {actual:?}");
    assert_eq!(
        actual.groups().collect::<Vec<_>>(),
        groups,
        "groups of {actual:?}"
    );
}

/// Asserts equal top-level structure (rank and extents) and equal mappings.
pub fn assert_same_map<A: LayoutMap, B: LayoutMap>(a: &A, b: &B) {
    assert_eq!(a.rank(), b.rank(), "rank");
    for axis in 0..a.rank() {
        assert_eq!(a.extent(axis), b.extent(axis), "extent of mode {axis}");
    }
    assert_eq!(values(a), values(b));
}

pub fn values<L: LayoutMap>(layout: &L) -> Vec<isize> {
    (0..layout.size().unwrap())
        .map(|i| layout.index(i).unwrap())
        .collect()
}

/// Non-affine lookup map (`as_affine() == Ok(None)`) used to test lazy fallback
/// paths that represent swizzled or otherwise non-affine layouts.
#[derive(Clone, Copy, Debug)]
pub struct TableMap(pub &'static [isize]);

impl LayoutMap for TableMap {
    fn rank(&self) -> usize {
        1
    }

    fn extent(&self, axis: usize) -> Result<isize> {
        if axis == 0 {
            Ok(self.0.len() as isize)
        } else {
            Err(Error::AxisOutOfBounds)
        }
    }

    fn size(&self) -> Result<isize> {
        Ok(self.0.len() as isize)
    }

    fn index(&self, linear: isize) -> Result<isize> {
        if linear < 0 {
            return Err(Error::CoordinateOutOfBounds);
        }
        self.0
            .get(linear as usize)
            .copied()
            .ok_or(Error::CoordinateOutOfBounds)
    }

    fn is_injective(&self) -> Result<bool> {
        for (i, &a) in self.0.iter().enumerate() {
            if self.0[..i].contains(&a) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn as_affine(&self) -> Result<Option<GroupedLayout>> {
        Ok(None)
    }
}
