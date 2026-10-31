//! 全静态、混合类型与运行时布局的结构 inverse，以及运行时枚举逆。
//!
//! 类型化右逆忽略动态 stride，并只接续静态已知的 stride 链；类型化左逆要求
//! coalesce 后的 stride 全部静态已知。DynLayout 方法按当前数值计算，不受上述
//! 类型约束。`dense_*_inverse` 是面向任意 LayoutMap 的运行时枚举辅助接口。
//!
//! ```
//! use noodle_cute::{Int, DynInt, layout::{Layout, LayoutMap}};
//! let layout = Layout::new(DynInt(4), Int::<1>)?;
//! let inverse: Layout<DynInt, Int<1>> = layout.right_inverse()?;
//! assert_eq!(inverse.index(3)?, 3);
//! # Ok::<(), noodle_cute::Error>(())
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, layout::Layout, algebra::static_left_inverse};
//! let _ = static_left_inverse(Layout { shape: Int::<0>, stride: Int::<1> });
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, shape, stride, layout::Layout, algebra::static_left_inverse};
//! let _ = static_left_inverse(Layout {
//!     shape: shape![Int::<2>, Int::<2>], stride: stride![Int::<2>, Int::<3>],
//! });
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, layout::Layout, algebra::static_left_inverse};
//! let _ = static_left_inverse(Layout { shape: Int::<2>, stride: Int::<-1> });
//! ```
//! ```compile_fail
//! use noodle_cute::{Int, layout::Layout, algebra::static_left_inverse};
//! let _ = static_left_inverse(Layout { shape: Int::<2>, stride: Int::<{isize::MAX}> });
//! ```

use core::ops::Div;

use crate::int::{Integer, add, mul};
use crate::layout::{
    AffineLayoutMap, DynLayout, GroupedLayout, Layout, LayoutMap, LayoutParts, MAX_RANK, Shape,
    affine_injective,
};
use crate::tuple::{Flatten, Item, Product};
use crate::{Assert, Error, Int, IntTuple, Nil, Result, True};

use super::Leaves;
use super::coalesce::{CoalesceFold, FinishCoalesce};
use super::complement::{NONEMPTY, Sort, UnwrapLayout};
use super::composition::{ComposeProduct, Select, ZERO_STRIDE};
use super::typed::Checked;

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
/// Advances only along modes whose stride equals the accumulated address span. Other modes (broadcasts, negative strides, or modes after
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
/// to a contiguous prefix. Dense affine bijections are verified using the stride-separation proof;
/// other maps linearly scan the domain for each reachable address (`O(cosize * size)`).
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
    let domain_size = layout.size()?;
    let size = match layout.address_range()? {
        None => 0,
        Some((0, max)) => add(max, 1)?,
        Some(_) => return Err(Error::NotSurjective),
    };
    if size > domain_size {
        return Err(Error::NotSurjective);
    }
    // When the codomain and domain have equal size, any injective map onto [0, size) is a bijection.
    let proven_bijection = match layout.as_affine() {
        Ok(Some(a)) if size == domain_size => affine_injective(&a, a.leaves())? == Some(true),
        Ok(_) | Err(Error::CapacityExceeded) => false,
        Err(e) => return Err(e),
    };
    if !proven_bijection {
        for address in 0..size {
            let mut found = false;
            for i in 0..domain_size {
                if layout.index(i)? == address {
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(Error::NotSurjective);
            }
        }
    }
    Ok(Inverse { layout, size })
}

/// 类型化结构逆；LEFT 为 false 时求右逆，为 true 时求左逆。
pub trait TypedInvert<const LEFT: bool = false> {
    type Output: LayoutMap;
    fn invert(self) -> Result<Self::Output>;
}
/// 保留静态叶子的结构逆结果。
pub type TypedInverse<L, const LEFT: bool = false> = <L as TypedInvert<LEFT>>::Output;
/// 全静态结构逆的结果和常量值。
pub trait StaticInverse<const LEFT: bool = false>: Copy {
    type Output: crate::StaticValue;
    const OUTPUT: Self::Output = <Self::Output as crate::StaticValue>::VALUE;
}
impl<S: crate::StaticValue, D: crate::StaticValue, const LEFT: bool> StaticInverse<LEFT>
    for Layout<S, D>
where
    Self: TypedInvert<LEFT>,
    TypedInverse<Self, LEFT>: crate::StaticValue,
{
    type Output = TypedInverse<Self, LEFT>;
}

/// 编译期右逆；动态输入使用 Layout::right_inverse。
///
/// 忽略广播、负 stride 和地址链的空隙；静态零域与溢出在实例化时拒绝。
/// 常量入口与 static_composition 一样，不执行最终运行时布局校验。
///
/// ```
/// use noodle_cute::{Int, shape, stride, layout::Layout};
/// use noodle_cute::algebra::inverse::static_right_inverse;
/// const R: Layout<Int<6>, Int<1>> = static_right_inverse(Layout {
///     shape: shape![Int::<2>, Int::<3>], stride: stride![Int::<1>, Int::<2>],
/// });
/// ```
/// ```compile_fail
/// use noodle_cute::{Int, DynInt, layout::Layout, algebra::inverse::static_right_inverse};
/// let _ = static_right_inverse(Layout { shape: DynInt(4), stride: Int::<1> });
/// ```
pub const fn static_right_inverse<L: StaticInverse>(_: L) -> L::Output {
    L::OUTPUT
}

/// 编译期左逆；静态负 stride、重叠或不可整除在实例化时拒绝。
///
/// 广播模式的逆按 CuTe 定义为准逆；空 tuple 采用本库标量约定。
/// 常量入口不执行最终运行时布局校验。
///
/// ```
/// use noodle_cute::{Int, IntTuple, Nil, layout::Layout};
/// use noodle_cute::algebra::inverse::static_left_inverse;
/// const R: Layout<IntTuple<Int<2>, IntTuple<Int<4>, Nil>>,
///                 IntTuple<Int<0>, IntTuple<Int<1>, Nil>>> =
///     static_left_inverse(Layout { shape: Int::<4>, stride: Int::<2> });
/// ```
/// ```compile_fail
/// use noodle_cute::{Int, DynInt, layout::Layout, algebra::inverse::static_left_inverse};
/// let _ = static_left_inverse(Layout { shape: Int::<4>, stride: DynInt(1) });
/// ```
pub const fn static_left_inverse<L: StaticInverse<true>>(_: L) -> L::Output {
    L::OUTPUT
}

impl<S: crate::layout::Shape, D: crate::tuple::Item<Profile = S::Profile>> Layout<S, D> {
    /// 保留类型的右逆，仅使用静态 stride，并只延续静态已知匹配的地址链。
    ///
    /// 动态 shape 可保留为结果叶子，但其后运行时才成立的 stride 等式不会改变结构。
    /// 零域返回 Unsupported，运行时乘积溢出返回 Overflow，不受 MAX_RANK 限制。
    pub fn right_inverse(&self) -> Result<TypedInverse<Self>>
    where
        Self: TypedInvert,
    {
        self.clone().invert()
    }

    /// 保留类型的左逆；coalesce 后的 stride 必须全部静态已知。
    ///
    /// 动态形状导致的重叠和溢出返回 NonInjective 或 Overflow；地址空隙保留为准逆。
    /// 任意动态 stride 求值使用 as_flat_layout 后的运行时入口。
    pub fn left_inverse(&self) -> Result<TypedInverse<Self, true>>
    where
        Self: TypedInvert<true>,
    {
        self.clone().invert()
    }
}
impl DynLayout {
    /// 根据当前数值求结构右逆，与自由函数 right_inverse 共用算法。
    pub fn right_inverse(&self) -> Result<Self> {
        right_inverse(*self)
    }
    /// 根据当前数值求结构左逆，与自由函数 left_inverse 共用算法和错误检查。
    pub fn left_inverse(&self) -> Result<Self> {
        left_inverse(*self)
    }
}

type Flat<S, D> = <Layout<<S as Flatten>::Output, <D as Flatten>::Output> as CoalesceFold>::Output;
type Inverted<S, D, const LEFT: bool> = <Flat<S, D> as InverseKernel<LEFT>>::Output;
impl<S, D, const LEFT: bool> TypedInvert<LEFT> for Layout<S, D>
where
    S: Shape + Flatten + Product,
    <S as Product>::Output: Integer,
    Assert<{ NONEMPTY::<S> }>: True,
    D: Item<Profile = S::Profile> + Flatten,
    Layout<<S as Flatten>::Output, D::Output>: CoalesceFold,
    Flat<S, D>: InverseKernel<LEFT>,
    Inverted<S, D, LEFT>: Checked,
{
    type Output = <Inverted<S, D, LEFT> as Checked>::Output;
    fn invert(self) -> Result<Self::Output> {
        if self.size()? == 0 {
            return Err(Error::Unsupported);
        }
        self.flatten().fold().inverse_kernel()?.checked()
    }
}

#[doc(hidden)]
pub trait FinishInverse {
    type Output: LayoutParts;
    fn finish_inverse(self) -> Result<Self::Output>;
}
impl<L: CoalesceFold> FinishInverse for L
where
    L::Output: FinishCoalesce,
    <L::Output as FinishCoalesce>::Output: UnwrapLayout,
{
    type Output = <<L::Output as FinishCoalesce>::Output as UnwrapLayout>::Output;
    fn finish_inverse(self) -> Result<Self::Output> {
        self.fold().finish().map(UnwrapLayout::unwrap_layout)
    }
}

#[doc(hidden)]
pub trait Attach<P> {
    type Output: LayoutParts;
    fn attach(self, prefix: P) -> Result<Self::Output>;
}
impl<P> Attach<P> for Layout<Nil, Nil> {
    type Output = Self;
    fn attach(self, _: P) -> Result<Self> {
        Ok(self)
    }
}
type NumberProduct<A, B> = <() as ComposeProduct<A, B>>::Output;
impl<S: Integer, ST, D, DT, P: Integer> Attach<P> for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    (): ComposeProduct<S, P>,
    Layout<ST, DT>: Attach<NumberProduct<S, P>>,
{
    type Output = Layout<
        IntTuple<
            (S, P),
            <<Layout<ST, DT> as Attach<NumberProduct<S, P>>>::Output as LayoutParts>::Shape,
        >,
        IntTuple<
            D,
            <<Layout<ST, DT> as Attach<NumberProduct<S, P>>>::Output as LayoutParts>::Stride,
        >,
    >;
    fn attach(self, prefix: P) -> Result<Self::Output> {
        let next = <() as ComposeProduct<S, P>>::product(self.shape.0, prefix)?;
        let tail = Layout {
            shape: self.shape.1,
            stride: self.stride.1,
        }
        .attach(next)?
        .into_layout();
        Ok(Layout {
            shape: IntTuple((self.shape.0, prefix), tail.shape),
            stride: IntTuple(self.stride.0, tail.stride),
        })
    }
}
#[doc(hidden)]
pub trait SelectStrides<const LEFT: bool> {
    type Output: LayoutParts;
    fn select_strides(self) -> Self::Output;
}
impl<S, D: crate::StaticValue> SelectStrides<true> for Layout<S, D> {
    type Output = Self;
    fn select_strides(self) -> Self {
        self
    }
}
impl SelectStrides<false> for Layout<Nil, Nil> {
    type Output = Self;
    fn select_strides(self) -> Self {
        self
    }
}
#[doc(hidden)]
pub const STATIC_STRIDE<D: Integer>: bool = D::STATIC.is_some();
impl<S, ST, D: Integer, DT> SelectStrides<false> for Layout<IntTuple<S, ST>, IntTuple<D, DT>>
where
    Layout<ST, DT>: SelectStrides<false>,
    Select<{ STATIC_STRIDE::<D> }>:
        KeepStride<S, D, <Layout<ST, DT> as SelectStrides<false>>::Output>,
{
    type Output = <Select<{ STATIC_STRIDE::<D> }> as KeepStride<
        S,
        D,
        <Layout<ST, DT> as SelectStrides<false>>::Output,
    >>::Output;
    fn select_strides(self) -> Self::Output {
        let tail = Layout {
            shape: self.shape.1,
            stride: self.stride.1,
        }
        .select_strides();
        <Select<{ STATIC_STRIDE::<D> }> as KeepStride<_, _, _>>::keep(
            self.shape.0,
            self.stride.0,
            tail,
        )
    }
}
#[doc(hidden)]
pub trait KeepStride<S, D, Tail> {
    type Output: LayoutParts;
    fn keep(s: S, d: D, tail: Tail) -> Self::Output;
}
impl<S, D, Tail: LayoutParts> KeepStride<S, D, Tail> for Select<false> {
    type Output = Tail;
    fn keep(_: S, _: D, tail: Tail) -> Tail {
        tail
    }
}
impl<S, D, Tail: LayoutParts> KeepStride<S, D, Tail> for Select<true> {
    type Output = Layout<IntTuple<S, Tail::Shape>, IntTuple<D, Tail::Stride>>;
    fn keep(s: S, d: D, tail: Tail) -> Self::Output {
        let tail = tail.into_layout();
        Layout {
            shape: IntTuple(s, tail.shape),
            stride: IntTuple(d, tail.stride),
        }
    }
}

#[doc(hidden)]
pub const MATCHES<D: Integer, C: Integer>: bool =
    matches!((D::STATIC, C::STATIC), (Some(d), Some(c)) if d == c);
#[doc(hidden)]
pub trait RightFold<C> {
    type Output: LayoutParts;
    fn right_fold(self, current: C) -> Result<Self::Output>;
}
impl<C> RightFold<C> for Layout<Nil, Nil> {
    type Output = Self;
    fn right_fold(self, _: C) -> Result<Self> {
        Ok(self)
    }
}
impl<S, P, ST, D: Integer, DT, C: Integer> RightFold<C>
    for Layout<IntTuple<(S, P), ST>, IntTuple<D, DT>>
where
    Select<{ MATCHES::<D, C> }>: RightStep<S, P, D, Layout<ST, DT>, C>,
{
    type Output = <Select<{ MATCHES::<D, C> }> as RightStep<S, P, D, Layout<ST, DT>, C>>::Output;
    fn right_fold(self, current: C) -> Result<Self::Output> {
        <Select<{ MATCHES::<D, C> }> as RightStep<_, _, _, _, _>>::run(
            self.shape.0.0,
            self.shape.0.1,
            self.stride.0,
            Layout {
                shape: self.shape.1,
                stride: self.stride.1,
            },
            current,
        )
    }
}
#[doc(hidden)]
pub trait RightStep<S, P, D, Tail, C> {
    type Output: LayoutParts;
    fn run(s: S, p: P, d: D, tail: Tail, current: C) -> Result<Self::Output>;
}
impl<S, P, D, Tail: RightFold<C>, C> RightStep<S, P, D, Tail, C> for Select<false> {
    type Output = Tail::Output;
    fn run(_: S, _: P, _: D, tail: Tail, c: C) -> Result<Self::Output> {
        tail.right_fold(c)
    }
}
impl<S: Integer, P, D: Integer, Tail, C> RightStep<S, P, D, Tail, C> for Select<true>
where
    (): ComposeProduct<S, D>,
    Tail: RightFold<NumberProduct<S, D>>,
{
    type Output = Layout<
        IntTuple<S, <Tail::Output as LayoutParts>::Shape>,
        IntTuple<P, <Tail::Output as LayoutParts>::Stride>,
    >;
    fn run(s: S, p: P, d: D, tail: Tail, _: C) -> Result<Self::Output> {
        let next = <() as ComposeProduct<S, D>>::product(s, d)?;
        let tail = tail.right_fold(next)?.into_layout();
        Ok(Layout {
            shape: IntTuple(s, tail.shape),
            stride: IntTuple(p, tail.stride),
        })
    }
}

#[doc(hidden)]
pub const LEFT_VALID<D: Integer, C: Integer, M: Integer>: bool = match (D::STATIC, C::STATIC) {
    (Some(d), Some(c)) => {
        d > 0
            && c > 0
            && d % c == 0
            && match M::STATIC {
                Some(m) => d >= m,
                None => true,
            }
    }
    _ => false,
};
#[doc(hidden)]
pub trait LeftFold<C, M, P, Last> {
    type Output: LayoutParts;
    fn left_fold(self, covered: C, minimum: M, previous: P, last: Last) -> Result<Self::Output>;
}
impl<C, M, P, Last> LeftFold<C, M, P, Last> for Layout<Nil, Nil> {
    type Output = Layout<IntTuple<Last, Nil>, IntTuple<P, Nil>>;
    fn left_fold(self, _: C, _: M, p: P, last: Last) -> Result<Self::Output> {
        Ok(Layout {
            shape: IntTuple(last, Nil),
            stride: IntTuple(p, Nil),
        })
    }
}
impl<S, PP, ST, D: Integer, DT, C, M, P, Last> LeftFold<C, M, P, Last>
    for Layout<IntTuple<(S, PP), ST>, IntTuple<D, DT>>
where
    Select<{ ZERO_STRIDE::<D> }>: LeftStep<S, PP, D, Layout<ST, DT>, C, M, P>,
{
    type Output =
        <Select<{ ZERO_STRIDE::<D> }> as LeftStep<S, PP, D, Layout<ST, DT>, C, M, P>>::Output;
    fn left_fold(self, c: C, m: M, p: P, _: Last) -> Result<Self::Output> {
        <Select<{ ZERO_STRIDE::<D> }> as LeftStep<_, _, _, _, _, _, _>>::run(
            self.shape.0.0,
            self.shape.0.1,
            self.stride.0,
            Layout {
                shape: self.shape.1,
                stride: self.stride.1,
            },
            c,
            m,
            p,
        )
    }
}
#[doc(hidden)]
pub trait LeftStep<S, PP, D, Tail, C, M, P> {
    type Output: LayoutParts;
    fn run(
        s: S,
        pp: PP,
        d: D,
        tail: Tail,
        covered: C,
        minimum: M,
        previous: P,
    ) -> Result<Self::Output>;
}
impl<S, PP, D, Tail, C, M, P> LeftStep<S, PP, D, Tail, C, M, P> for Select<true>
where
    Tail: LeftFold<C, M, P, S>,
{
    type Output = Tail::Output;
    fn run(s: S, _: PP, _: D, tail: Tail, c: C, m: M, p: P) -> Result<Self::Output> {
        tail.left_fold(c, m, p, s)
    }
}
impl<S: Integer, PP, D: Integer + Div<C>, Tail, C: Integer, M: Integer, P>
    LeftStep<S, PP, D, Tail, C, M, P> for Select<false>
where
    Assert<{ LEFT_VALID::<D, C, M> }>: True,
    (): ComposeProduct<S, D>,
    Tail: LeftFold<D, NumberProduct<S, D>, PP, S>,
{
    type Output = Layout<
        IntTuple<D::Output, <Tail::Output as LayoutParts>::Shape>,
        IntTuple<P, <Tail::Output as LayoutParts>::Stride>,
    >;
    fn run(s: S, pp: PP, d: D, tail: Tail, c: C, m: M, p: P) -> Result<Self::Output> {
        if d.value() < 0 {
            return Err(Error::Unsupported);
        }
        if d.value() % c.value() != 0 {
            return Err(Error::NotDivisible);
        }
        if d.value() < m.value() {
            return Err(Error::NonInjective);
        }
        let minimum = <() as ComposeProduct<S, D>>::product(s, d)?;
        let tail = tail.left_fold(d, minimum, pp, s)?.into_layout();
        Ok(Layout {
            shape: IntTuple(d / c, tail.shape),
            stride: IntTuple(p, tail.stride),
        })
    }
}

#[doc(hidden)]
pub trait InverseKernel<const LEFT: bool> {
    type Output: LayoutParts;
    fn inverse_kernel(self) -> Result<Self::Output>;
}
type Selected<L, const LEFT: bool> = <<L as Attach<Int<1>>>::Output as SelectStrides<LEFT>>::Output;
type Ordered<L, const LEFT: bool> = <Selected<L, LEFT> as Sort>::Output;
impl<L: Attach<Int<1>>> InverseKernel<false> for L
where
    L::Output: SelectStrides<false>,
    Selected<L, false>: Sort,
    Ordered<L, false>: RightFold<Int<1>>,
    <Ordered<L, false> as RightFold<Int<1>>>::Output: FinishInverse,
{
    type Output = <<Ordered<L, false> as RightFold<Int<1>>>::Output as FinishInverse>::Output;
    fn inverse_kernel(self) -> Result<Self::Output> {
        self.attach(Int)?
            .select_strides()
            .sort()
            .right_fold(Int)?
            .finish_inverse()
    }
}
impl<L: Attach<Int<1>>> InverseKernel<true> for L
where
    L::Output: SelectStrides<true>,
    Selected<L, true>: Sort,
    Ordered<L, true>: LeftFold<Int<1>, Int<1>, Int<0>, Int<1>>,
    <Ordered<L, true> as LeftFold<Int<1>, Int<1>, Int<0>, Int<1>>>::Output: FinishInverse,
{
    type Output = <<Ordered<L, true> as LeftFold<Int<1>, Int<1>, Int<0>, Int<1>>>::Output as FinishInverse>::Output;
    fn inverse_kernel(self) -> Result<Self::Output> {
        self.attach(Int)?
            .select_strides()
            .sort()
            .left_fold(Int, Int, Int, Int)?
            .finish_inverse()
    }
}
