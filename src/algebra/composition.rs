//! 全静态、混合类型与运行时布局的 composition，以及通用惰性组合。
//!
//! 对应 CuTe layout.hpp 的 composition_impl；coalesce_x 复用公共 coalesce 模块。
//! 全静态使用 [`static_composition`]；混合使用 [`Layout::composition`]，保留类型。
//! [`DynLayout::composition`] 返回运行时元数据；通用 [`composition`] 保留惰性回退。

use crate::int::{Integer, ceil_div, mul};
use crate::layout::{
    AffineLayoutMap, DynLayout, GroupedLayout, Layout, LayoutMap, LayoutParts, MAX_RANK, Shape,
    affine_injective, is_injective_by_enumeration,
};
#[doc(hidden)]
pub use crate::tuple::Unwrap;
use crate::tuple::{Append, Flatten, Item};
use crate::{DynInt, Error, Int, IntTuple, Nil, Result, StaticValue};

use super::Leaves;
use super::coalesce::CoalesceFold;

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
        let step = checked_step(rest_shape, rest_stride, cs)?;
        if !step.skip {
            out.push(step.extent, mul(rest_stride, cd)?)?;
        }
        rest_shape = step.rest;
        rest_stride = step.stride;
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
        // Fast path when `outer` has a cheap stride-separation proof on its logical domain;
        // if `outer` is non-affine, inconclusive, or exceeded by `inner`, enumerate the composed domain.
        let outer_proven = match self.outer.as_affine() {
            Ok(Some(a)) => affine_injective(&a, a.leaves())? == Some(true),
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
    <Layout<<S as Flatten>::Output, <D as Flatten>::Output> as CoalesceFold<true>>::Output;

/// 类型化 composition 的结果，保留 RHS 的嵌套结构和可静态推导的叶子。
pub type TypedComposition<A, B> = <A as TypedCompose<B>>::Output;

/// 类型化布局的受检组合；全静态入口复用此 trait 的结果类型。
pub trait TypedCompose<Rhs> {
    type Output: LayoutMap;
    fn compose(self, rhs: Rhs) -> Result<Self::Output>;
}

impl<S: Shape, D: Item<Profile = S::Profile>> Layout<S, D> {
    /// 按 CuTe composition_impl 组合静态或混合布局，保留结果类型。
    ///
    /// 只有静态已知的 0 stride、单位 shape 才能改变输出结构；运行时巧合的
    /// 单位模式会保留。允许负 stride 和扩展域，输出经过 Layout::new 校验。
    /// 动态整除失败返回 NotDivisible；多模式中动态零 stride 导致的零除返回
    /// InvalidShape。纯静态错误在类型实例化时拒绝，不受 MAX_RANK 限制。
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
    /// 混合输入中，静态已知的 stride 整除错误同样在编译期拒绝：
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
        Self: TypedCompose<Layout<RS, RD>>,
    {
        self.clone().compose(rhs)
    }
}

impl DynLayout {
    /// 组合运行时仿射布局，保留 RHS 的顶层分组，返回受检的运行时元数据。
    ///
    /// 与 composition_layout 使用相同算法；输入的负逻辑编号、不可整除与溢出
    /// 分别返回 CoordinateOutOfBounds、NotDivisible 与 Overflow。
    pub fn composition<B: AffineLayoutMap>(&self, rhs: &B) -> Result<GroupedLayout> {
        composition_layout(self, rhs)
    }
}

type CompositionTree<S, D, RS, RD> = <Layout<RS, RD> as ComposeTree<CoalescedLhs<S, D>>>::Output;

impl<S, D, RS, RD> TypedCompose<Layout<RS, RD>> for Layout<S, D>
where
    S: Shape + Flatten,
    D: Item<Profile = S::Profile> + Flatten,
    RS: Shape,
    RD: Item<Profile = RS::Profile>,
    Layout<S::Output, D::Output>: CoalesceFold<true>,
    Layout<RS, RD>: ComposeTree<CoalescedLhs<S, D>>,
    <CompositionTree<S, D, RS, RD> as LayoutParts>::Shape: Shape,
    <CompositionTree<S, D, RS, RD> as LayoutParts>::Stride:
        Item<Profile = <<CompositionTree<S, D, RS, RD> as LayoutParts>::Shape as Item>::Profile>,
{
    type Output = Layout<
        <CompositionTree<S, D, RS, RD> as LayoutParts>::Shape,
        <CompositionTree<S, D, RS, RD> as LayoutParts>::Stride,
    >;
    fn compose(self, rhs: Layout<RS, RD>) -> Result<Self::Output> {
        self.size()?;
        rhs.size()?;
        let lhs = self.flatten().fold();
        let result = rhs.compose_tree(&lhs)?.into_layout();
        Layout::new(result.shape, result.stride)
    }
}

/// 全静态组合的结果类型与常量值。
///
/// 不经过 `DynLayout`，因此不受运行时 `MAX_RANK` 缓冲区限制。
pub trait StaticCompose<Rhs>: Copy {
    type Output: StaticValue;
    const OUTPUT: Self::Output = Self::Output::VALUE;
}

/// 两个静态布局组合后得到的具体 `Layout` 类型。
pub type StaticComposition<A, B> = <A as StaticCompose<B>>::Output;

/// 在编译期组合两个全静态布局，保留 RHS 的嵌套结构。
///
/// 与 CuTe 一样允许扩展域和负 stride；这不是 `Layout::new` 的运行时
/// 地址范围验证入口。静态中间运算必须可由 `isize` 表示。
/// 空 RHS 保留为空 tuple；空 LHS 按本库标量约定扩展为 `1:0`。
///
/// ```
/// use noodle_cute::{Int, layout::{Layout, LayoutMap}};
/// use noodle_cute::algebra::static_composition;
/// const A: Layout<Int<8>, Int<2>> = Layout { shape: Int, stride: Int };
/// const B: Layout<Int<3>, Int<2>> = Layout { shape: Int, stride: Int };
/// const C: Layout<Int<3>, Int<4>> = static_composition(A, B);
/// assert_eq!(C.index(2), Ok(8));
/// ```
///
/// 动态值不能进入全静态入口：
/// ```compile_fail
/// use noodle_cute::{DynInt, Int, layout::Layout};
/// use noodle_cute::algebra::static_composition;
/// let a = Layout { shape: Int::<8>, stride: Int::<1> };
/// let b = Layout { shape: DynInt(3), stride: Int::<1> };
/// let _ = static_composition(a, b);
/// ```
/// 不可整除的静态 stride 在编译期拒绝：
/// ```compile_fail
/// use noodle_cute::{Int, layout::Layout, shape, stride};
/// use noodle_cute::algebra::static_composition;
/// let a = Layout { shape: shape![Int::<4>, Int::<6>], stride: stride![Int::<6>, Int::<1>] };
/// let b = Layout { shape: Int::<8>, stride: Int::<5> };
/// let _ = static_composition(a, b);
/// ```
///
/// 不可整除的静态 shape 在编译期拒绝：
/// ```compile_fail
/// use noodle_cute::{Int, layout::Layout, shape, stride};
/// use noodle_cute::algebra::static_composition;
/// let a = Layout { shape: shape![Int::<4>, Int::<6>], stride: stride![Int::<6>, Int::<1>] };
/// let b = Layout { shape: Int::<3>, stride: Int::<3> };
/// let _ = static_composition(a, b);
/// ```
///
/// 静态 stride 乘积溢出在编译期拒绝：
/// ```compile_fail
/// use noodle_cute::{Int, layout::Layout};
/// use noodle_cute::algebra::static_composition;
/// let a = Layout { shape: Int::<2>, stride: Int::<{ isize::MAX }> };
/// let b = Layout { shape: Int::<2>, stride: Int::<2> };
/// let _ = static_composition(a, b);
/// ```
///
/// shape 与 stride 的嵌套 profile 必须一致：
/// ```compile_fail
/// use noodle_cute::{Int, layout::Layout, shape};
/// use noodle_cute::algebra::static_composition;
/// let a = Layout { shape: Int::<4>, stride: Int::<1> };
/// let b = Layout { shape: shape![Int::<2>], stride: Int::<1> };
/// let _ = static_composition(a, b);
/// ```
pub const fn static_composition<A: StaticCompose<B>, B: Copy>(_: A, _: B) -> A::Output {
    A::OUTPUT
}

impl<S, D, RS, RD> StaticCompose<Layout<RS, RD>> for Layout<S, D>
where
    S: StaticValue,
    D: StaticValue,
    RS: StaticValue,
    RD: StaticValue,
    Self: TypedCompose<Layout<RS, RD>>,
    TypedComposition<Self, Layout<RS, RD>>: StaticValue,
{
    type Output = TypedComposition<Self, Layout<RS, RD>>;
}

#[doc(hidden)]
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

#[doc(hidden)]
pub use crate::Assert as Select;
#[doc(hidden)]
pub const ZERO_STRIDE<D: Integer>: bool = matches!(D::STATIC, Some(0));
#[doc(hidden)]
pub trait ComposeLeaf<Lhs, S, D> {
    type Output: LayoutParts;
    fn compose_leaf(lhs: &Lhs, s: S, d: D) -> Result<Self::Output>;
}
impl<Lhs, S: Integer, D: Integer> ComposeTree<Lhs> for Layout<S, D>
where
    Select<{ ZERO_STRIDE::<D> }>: ComposeLeaf<Lhs, S, D>,
{
    type Output = <Select<{ ZERO_STRIDE::<D> }> as ComposeLeaf<Lhs, S, D>>::Output;
    fn compose_tree(self, lhs: &Lhs) -> Result<Self::Output> {
        <Select<{ ZERO_STRIDE::<D> }> as ComposeLeaf<Lhs, S, D>>::compose_leaf(
            lhs,
            self.shape,
            self.stride,
        )
    }
}
impl<Lhs, S, D> ComposeLeaf<Lhs, S, D> for Select<true> {
    type Output = Layout<S, D>;
    fn compose_leaf(_: &Lhs, s: S, d: D) -> Result<Layout<S, D>> {
        Ok(Layout {
            shape: s,
            stride: d,
        })
    }
}
impl<Lhs, S, D> ComposeLeaf<Lhs, S, D> for Select<false>
where
    Lhs: Clone + Fold<S, D, Nil, Nil>,
{
    type Output = <Lhs as Fold<S, D, Nil, Nil>>::Output;
    fn compose_leaf(lhs: &Lhs, s: S, d: D) -> Result<Self::Output> {
        lhs.clone().fold(s, d, Nil, Nil)
    }
}

#[doc(hidden)]
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

#[doc(hidden)]
pub const STEP_STATIC<S: Integer, D: Integer, CS: Integer>: bool =
    S::STATIC.is_some() && D::STATIC.is_some() && CS::STATIC.is_some();
#[doc(hidden)]
pub const REST_STATIC<S: Integer, D: Integer, CS: Integer>: bool =
    matches!(S::STATIC, Some(0)) || STEP_STATIC::<S, D, CS>;
#[doc(hidden)]
pub const STRIDE_STATIC<D: Integer, CS: Integer>: bool =
    D::STATIC.is_some() && CS::STATIC.is_some();
#[doc(hidden)]
pub const SKIP<S: Integer, D: Integer, CS: Integer>: bool = matches!(S::STATIC, Some(1))
    || match (D::STATIC, CS::STATIC) {
        (Some(d), Some(cs)) => step(1, d, cs).next_shape == 1,
        _ => false,
    };
#[doc(hidden)]
pub const EXTENT<S: Integer, D: Integer, CS: Integer>: isize =
    match (S::STATIC, D::STATIC, CS::STATIC) {
        (Some(s), Some(d), Some(cs)) => step(s, d, cs).extent,
        _ => 0,
    };
#[doc(hidden)]
pub const REST<S: Integer, D: Integer, CS: Integer>: isize =
    match (S::STATIC, D::STATIC, CS::STATIC) {
        (Some(0), _, _) => 0,
        (Some(s), Some(d), Some(cs)) => step(s, d, cs).rest,
        _ => 0,
    };
#[doc(hidden)]
pub const STRIDE<D: Integer, CS: Integer>: isize = match (D::STATIC, CS::STATIC) {
    (Some(d), Some(cs)) => step(1, d, cs).stride,
    _ => 0,
};

#[doc(hidden)]
pub trait StepNumber<const N: isize> {
    type Output: Integer;
    fn make(value: isize) -> Self::Output;
}
impl<const N: isize> StepNumber<N> for Select<true> {
    type Output = Int<N>;
    fn make(_: isize) -> Self::Output {
        Int
    }
}
impl<const N: isize> StepNumber<N> for Select<false> {
    type Output = DynInt;
    fn make(value: isize) -> Self::Output {
        DynInt(value)
    }
}

#[doc(hidden)]
pub trait TypedStep<S, D, CS> {
    type Extent: Integer;
    type Rest: Integer;
    type Stride: Integer;
    fn run(s: S, d: D, cs: CS) -> Result<(Self::Extent, Self::Rest, Self::Stride)>;
}
impl<S: Integer, D: Integer, CS: Integer> TypedStep<S, D, CS> for ()
where
    Select<{ STEP_STATIC::<S, D, CS> }>: StepNumber<{ EXTENT::<S, D, CS> }>,
    Select<{ REST_STATIC::<S, D, CS> }>: StepNumber<{ REST::<S, D, CS> }>,
    Select<{ STRIDE_STATIC::<D, CS> }>: StepNumber<{ STRIDE::<D, CS> }>,
{
    type Extent =
        <Select<{ STEP_STATIC::<S, D, CS> }> as StepNumber<{ EXTENT::<S, D, CS> }>>::Output;
    type Rest = <Select<{ REST_STATIC::<S, D, CS> }> as StepNumber<{ REST::<S, D, CS> }>>::Output;
    type Stride = <Select<{ STRIDE_STATIC::<D, CS> }> as StepNumber<{ STRIDE::<D, CS> }>>::Output;
    fn run(s: S, d: D, cs: CS) -> Result<(Self::Extent, Self::Rest, Self::Stride)> {
        let shape = s.value();
        if shape < 0 {
            return Err(Error::InvalidShape);
        }
        let mut v = checked_step(1, d.value(), cs.value())?;
        if SKIP::<S, D, CS> {
            v.rest = shape;
        } else {
            v.extent = v.next_shape.min(shape);
            // CuTe 的静态零分子除动态值仍为静态零，不执行运行时除法。
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
            <Select<{ STEP_STATIC::<S, D, CS> }> as StepNumber<{ EXTENT::<S, D, CS> }>>::make(
                v.extent,
            ),
            <Select<{ REST_STATIC::<S, D, CS> }> as StepNumber<{ REST::<S, D, CS> }>>::make(v.rest),
            <Select<{ STRIDE_STATIC::<D, CS> }> as StepNumber<{ STRIDE::<D, CS> }>>::make(v.stride),
        ))
    }
}

#[doc(hidden)]
pub const PRODUCT_STATIC<A: Integer, B: Integer>: bool = matches!(A::STATIC, Some(0))
    || matches!(B::STATIC, Some(0))
    || (A::STATIC.is_some() && B::STATIC.is_some());
#[doc(hidden)]
pub const PRODUCT<A: Integer, B: Integer>: isize = match (A::STATIC, B::STATIC) {
    (Some(0), _) | (_, Some(0)) => 0,
    (Some(a), Some(b)) => a * b,
    _ => 0,
};

// CuTe 的静态零乘动态值仍是静态零，不能用本库的普通混合乘法推导类型。
#[doc(hidden)]
pub trait ComposeProduct<A, B> {
    type Output: Integer;
    fn product(a: A, b: B) -> Result<Self::Output>;
}
impl<A: Integer, B: Integer> ComposeProduct<A, B> for ()
where
    Select<{ PRODUCT_STATIC::<A, B> }>: StepNumber<{ PRODUCT::<A, B> }>,
{
    type Output = <Select<{ PRODUCT_STATIC::<A, B> }> as StepNumber<{ PRODUCT::<A, B> }>>::Output;
    fn product(a: A, b: B) -> Result<Self::Output> {
        let value = mul(a.value(), b.value())?;
        Ok(<Select<{ PRODUCT_STATIC::<A, B> }> as StepNumber<
            { PRODUCT::<A, B> },
        >>::make(value))
    }
}

#[doc(hidden)]
pub trait Fold<S, D, OS, OD> {
    type Output: LayoutParts;
    fn fold(self, s: S, d: D, os: OS, od: OD) -> Result<Self::Output>;
}
impl<S, D, CS, CD, OS, OD> Fold<S, D, OS, OD> for Layout<IntTuple<CS, Nil>, IntTuple<CD, Nil>>
where
    Layout<OS, OD>: Finish<S, D, CD>,
{
    type Output = <Layout<OS, OD> as Finish<S, D, CD>>::Output;
    fn fold(self, s: S, d: D, os: OS, od: OD) -> Result<Self::Output> {
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
    Select<{ SKIP::<S, D, CS> }>:
        FoldStep<Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>, S, D, CS, CD, OS, OD>,
{
    type Output = <Select<{ SKIP::<S, D, CS> }> as FoldStep<
        Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>,
        S,
        D,
        CS,
        CD,
        OS,
        OD,
    >>::Output;
    fn fold(self, s: S, d: D, os: OS, od: OD) -> Result<Self::Output> {
        <Select<{ SKIP::<S, D, CS> }> as FoldStep<
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

#[doc(hidden)]
pub trait FoldStep<Tail, S, D, CS, CD, OS, OD> {
    type Output: LayoutParts;
    fn run(tail: Tail, s: S, d: D, cs: CS, cd: CD, os: OS, od: OD) -> Result<Self::Output>;
}
impl<Tail, S: Integer, D: Integer, CS: Integer, CD, OS, OD> FoldStep<Tail, S, D, CS, CD, OS, OD>
    for Select<true>
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
impl<Tail, S: Integer, D: Integer, CS: Integer, CD: Integer, OS, OD>
    FoldStep<Tail, S, D, CS, CD, OS, OD> for Select<false>
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

#[doc(hidden)]
pub trait Finish<S, D, CD> {
    type Output: LayoutParts;
    fn finish(self, s: S, d: D, cd: CD) -> Result<Self::Output>;
}
impl<S, D: Integer, CD: Integer> Finish<S, D, CD> for Layout<Nil, Nil>
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
#[doc(hidden)]
pub const UNIT<S: Integer>: bool = matches!(S::STATIC, Some(1));
impl<S: Integer, D, CD, SH, ST, DH, DT> Finish<S, D, CD>
    for Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>
where
    Select<{ UNIT::<S> }>: FinishNonempty<IntTuple<SH, ST>, IntTuple<DH, DT>, S, D, CD>,
{
    type Output = <Select<{ UNIT::<S> }> as FinishNonempty<
        IntTuple<SH, ST>,
        IntTuple<DH, DT>,
        S,
        D,
        CD,
    >>::Output;
    fn finish(self, s: S, d: D, cd: CD) -> Result<Self::Output> {
        <Select<{ UNIT::<S> }> as FinishNonempty<IntTuple<SH, ST>, IntTuple<DH, DT>, S, D, CD>>::run(
            self.shape,
            self.stride,
            s,
            d,
            cd,
        )
    }
}
#[doc(hidden)]
pub trait FinishNonempty<OS, OD, S, D, CD> {
    type Output: LayoutParts;
    fn run(os: OS, od: OD, s: S, d: D, cd: CD) -> Result<Self::Output>;
}
impl<OS: Unwrap, OD: Unwrap, S, D, CD> FinishNonempty<OS, OD, S, D, CD> for Select<true> {
    type Output = Layout<OS::Output, OD::Output>;
    fn run(os: OS, od: OD, _: S, _: D, _: CD) -> Result<Self::Output> {
        Ok(Layout {
            shape: os.unwrap(),
            stride: od.unwrap(),
        })
    }
}
impl<OS, OD, S, D: Integer, CD: Integer> FinishNonempty<OS, OD, S, D, CD> for Select<false>
where
    (): ComposeProduct<D, CD>,
    OS: Append<IntTuple<S, Nil>>,
    OD: Append<IntTuple<<() as ComposeProduct<D, CD>>::Output, Nil>>,
{
    type Output = Layout<OS::Output, OD::Output>;
    fn run(os: OS, od: OD, s: S, d: D, cd: CD) -> Result<Self::Output> {
        Ok(Layout {
            shape: os.append(IntTuple(s, Nil)),
            stride: od.append(IntTuple(
                <() as ComposeProduct<D, CD>>::product(d, cd)?,
                Nil,
            )),
        })
    }
}
