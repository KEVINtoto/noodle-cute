//! 类型化与运行时布局的 coalesce，以及组合使用的 coalesce_x。
//!
//! 对应 CuTe `layout.hpp` 的 coalesce 与 `bw_coalesce`。类型化布局仅使用
//! 类型已知的事实改变结构，两条类型层路径共用反向折叠与 Prepend。
//! 运行时布局按当前数值化简；空输入按本库标量约定归一为 `1:0`。

use core::ops::Mul;

use crate::int::{Int, Integer, mul};
use crate::layout::{DynLayout, Layout, LayoutMap, Shape};
use crate::tuple::{Flatten, Item, Nil};
use crate::{IntTuple, Result, StaticValue};

/// Selects a coalescing action from facts known at compile time.
///
/// The fold changes types only using static facts: a runtime extent equal to one
/// remains in the output tuple. Action codes are [`DISCARD`] for a static unit
/// extent, [`RETAIN`] to keep a leaf, and [`MERGE`] to merge it with the folded
/// tail's head.
#[doc(hidden)]
pub struct Decision<const N: u8>;

/// Drops a static unit extent.
#[doc(hidden)]
pub const DISCARD: u8 = 0;
/// Keeps the leaf as a separate mode.
#[doc(hidden)]
pub const RETAIN: u8 = 1;
/// Merges the leaf with the folded tail's head.
#[doc(hidden)]
pub const MERGE: u8 = 2;

/// Selects whether a leaf is discarded or retained when the folded tail is empty.
#[doc(hidden)]
pub const fn unit_decision(s: Option<isize>) -> u8 {
    if matches!(s, Some(1)) {
        DISCARD
    } else {
        RETAIN
    }
}

/// Selects whether to discard, retain, or merge a leaf with the folded tail's head.
#[doc(hidden)]
pub const fn merge_decision(
    s: Option<isize>,
    d: Option<isize>,
    tail_shape: Option<isize>,
    next: Option<isize>,
) -> u8 {
    if matches!(s, Some(1)) {
        return DISCARD;
    }
    // 类型层的 None 表示未知；运行时路径将当前数值全部作为已知值传入。
    match (s, d, tail_shape, next) {
        (Some(a), Some(b), Some(_), Some(c)) => match a.checked_mul(b) {
            Some(v) if v == c => MERGE,
            _ => RETAIN,
        },
        _ => RETAIN,
    }
}

// Reuse these constants in bounds and associated-type projections so generic
// const expressions refer to identical definitions.
#[doc(hidden)]
pub const UNIT_DECISION<S: Integer>: u8 = unit_decision(S::STATIC);
#[doc(hidden)]
pub const MERGE_DECISION<S: Integer, D: Integer, TS: Integer, TD: Integer>: u8 =
    merge_decision(S::STATIC, D::STATIC, TS::STATIC, TD::STATIC);

/// 将叶子加入空尾部；EXTENDED 为真时用 extent 2 保留末尾静态单位轴。
#[doc(hidden)]
pub trait PrependToEmpty<S, D, const EXTENDED: bool = false> {
    type Output;
    fn apply(s: S, d: D) -> Self::Output;
}

impl<S, D> PrependToEmpty<S, D, false> for Decision<DISCARD> {
    type Output = Layout<Nil, Nil>;
    fn apply(_: S, _: D) -> Self::Output {
        Layout {
            shape: Nil,
            stride: Nil,
        }
    }
}

impl<S, D, const EXTENDED: bool> PrependToEmpty<S, D, EXTENDED> for Decision<RETAIN> {
    type Output = Layout<IntTuple<S, Nil>, IntTuple<D, Nil>>;
    fn apply(s: S, d: D) -> Self::Output {
        Layout {
            shape: IntTuple(s, Nil),
            stride: IntTuple(d, Nil),
        }
    }
}

/// Prepends `(s, d)` to a folded tail. Type parameters:
/// - `S`, `D`: types of the new shape and stride leaves being prepended.
/// - `TS`, `TD`: types of the tail's first shape and stride leaves.
/// - `TSR`, `TDR`: types of the remaining shape and stride tuples of the tail
///   after its first leaf.
#[doc(hidden)]
pub trait PrependToNonempty<S, D, TS, TSR, TD, TDR> {
    type Output;
    fn apply(s: S, d: D, tail: Layout<IntTuple<TS, TSR>, IntTuple<TD, TDR>>) -> Self::Output;
}

impl<S, D, TS, TSR, TD, TDR> PrependToNonempty<S, D, TS, TSR, TD, TDR> for Decision<DISCARD> {
    type Output = Layout<IntTuple<TS, TSR>, IntTuple<TD, TDR>>;
    fn apply(_: S, _: D, tail: Self::Output) -> Self::Output {
        tail
    }
}

impl<S, D, TS, TSR, TD, TDR> PrependToNonempty<S, D, TS, TSR, TD, TDR> for Decision<RETAIN> {
    type Output = Layout<IntTuple<S, IntTuple<TS, TSR>>, IntTuple<D, IntTuple<TD, TDR>>>;
    fn apply(s: S, d: D, tail: Layout<IntTuple<TS, TSR>, IntTuple<TD, TDR>>) -> Self::Output {
        Layout {
            shape: IntTuple(s, tail.shape),
            stride: IntTuple(d, tail.stride),
        }
    }
}

impl<S: Mul<TS>, D, TS, TSR, TD, TDR> PrependToNonempty<S, D, TS, TSR, TD, TDR>
    for Decision<MERGE>
{
    type Output = Layout<IntTuple<S::Output, TSR>, IntTuple<D, TDR>>;
    fn apply(s: S, d: D, tail: Layout<IntTuple<TS, TSR>, IntTuple<TD, TDR>>) -> Self::Output {
        Layout {
            shape: IntTuple(s * tail.shape.0, tail.shape.1),
            stride: IntTuple(d, tail.stride.1),
        }
    }
}

/// Dispatches the prepend action using the static extent and stride metadata.
///
/// `S`/`D` are the new shape/stride leaf types. For a nonempty tail, `TS`/`TD`
/// are its first shape/stride leaf types and `TSR`/`TDR` the remaining tuples.
#[doc(hidden)]
pub trait Prepend<S, D> {
    type Output;
    fn prepend(self, s: S, d: D) -> Self::Output;
}

impl<S: Integer, D> Prepend<S, D> for Layout<Nil, Nil>
where
    Decision<{ UNIT_DECISION::<S> }>: PrependToEmpty<S, D>,
{
    type Output = <Decision<{ UNIT_DECISION::<S> }> as PrependToEmpty<S, D>>::Output;
    fn prepend(self, s: S, d: D) -> Self::Output {
        <Decision<{ UNIT_DECISION::<S> }> as PrependToEmpty<S, D>>::apply(s, d)
    }
}

impl<S: Integer, D: Integer, TS: Integer, TSR, TD: Integer, TDR> Prepend<S, D>
    for Layout<IntTuple<TS, TSR>, IntTuple<TD, TDR>>
where
    Decision<{ MERGE_DECISION::<S, D, TS, TD> }>: PrependToNonempty<S, D, TS, TSR, TD, TDR>,
{
    type Output = <Decision<{ MERGE_DECISION::<S, D, TS, TD> }> as PrependToNonempty<
        S,
        D,
        TS,
        TSR,
        TD,
        TDR,
    >>::Output;
    fn prepend(self, s: S, d: D) -> Self::Output {
        <Decision<{ MERGE_DECISION::<S, D, TS, TD> }> as PrependToNonempty<
            S,
            D,
            TS,
            TSR,
            TD,
            TDR,
        >>::apply(s, d, self)
    }
}

/// Coalesces a flattened layout at the type level.
///
/// The result can still be empty when the input is empty or every extent is a
/// static one; [`FinishCoalesce`] normalizes that case.
/// `EXTENDED` 选择 coalesce_x 的初始叶子，保留末尾 stride。
#[doc(hidden)]
pub trait CoalesceFold<const EXTENDED: bool = false> {
    type Output;
    fn fold(self) -> Self::Output;
}

impl CoalesceFold for Layout<Nil, Nil> {
    type Output = Self;
    fn fold(self) -> Self {
        self
    }
}

impl<S: Integer, D> CoalesceFold for Layout<IntTuple<S, Nil>, IntTuple<D, Nil>>
where
    Layout<Nil, Nil>: Prepend<S, D>,
{
    type Output = <Layout<Nil, Nil> as Prepend<S, D>>::Output;
    fn fold(self) -> Self::Output {
        Layout {
            shape: Nil,
            stride: Nil,
        }
        .prepend(self.shape.0, self.stride.0)
    }
}

impl CoalesceFold<true> for Layout<Nil, Nil> {
    // 上游要求非空输入；这里沿用本库空布局的标量约定。
    type Output = Layout<IntTuple<Int<1>, Nil>, IntTuple<Int<0>, Nil>>;
    fn fold(self) -> Self::Output {
        Layout {
            shape: IntTuple(Int, Nil),
            stride: IntTuple(Int, Nil),
        }
    }
}

impl<S, D> PrependToEmpty<S, D, true> for Decision<DISCARD> {
    type Output = Layout<IntTuple<Int<2>, Nil>, IntTuple<D, Nil>>;
    fn apply(_: S, d: D) -> Self::Output {
        Layout {
            shape: IntTuple(Int, Nil),
            stride: IntTuple(d, Nil),
        }
    }
}
impl<S: Integer, D> CoalesceFold<true> for Layout<IntTuple<S, Nil>, IntTuple<D, Nil>>
where
    Decision<{ UNIT_DECISION::<S> }>: PrependToEmpty<S, D, true>,
{
    type Output = <Decision<{ UNIT_DECISION::<S> }> as PrependToEmpty<S, D, true>>::Output;
    fn fold(self) -> Self::Output {
        <Decision<{ UNIT_DECISION::<S> }> as PrependToEmpty<S, D, true>>::apply(
            self.shape.0,
            self.stride.0,
        )
    }
}

impl<S, D, SH, ST, DH, DT, const EXTENDED: bool> CoalesceFold<EXTENDED>
    for Layout<IntTuple<S, IntTuple<SH, ST>>, IntTuple<D, IntTuple<DH, DT>>>
where
    Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>: CoalesceFold<EXTENDED>,
    <Layout<IntTuple<SH, ST>, IntTuple<DH, DT>> as CoalesceFold<EXTENDED>>::Output: Prepend<S, D>,
{
    type Output = <<Layout<IntTuple<SH, ST>, IntTuple<DH, DT>> as CoalesceFold<EXTENDED>>::Output as Prepend<S, D>>::Output;
    fn fold(self) -> Self::Output {
        Layout {
            shape: self.shape.1,
            stride: self.stride.1,
        }
        .fold()
        .prepend(self.shape.0, self.stride.0)
    }
}

/// Validates the folded layout and normalizes an empty result to `1:0`.
#[doc(hidden)]
pub trait FinishCoalesce {
    type Output: LayoutMap;
    fn finish(self) -> Result<Self::Output>;
}

impl FinishCoalesce for Layout<Nil, Nil> {
    type Output = Layout<Int<1>, Int<0>>;
    // An empty input or removal of every static unit extent yields a single
    // logical element at address zero.
    fn finish(self) -> Result<Self::Output> {
        Layout::new(Int, Int)
    }
}

// `S`/`D` are the first shape/stride leaf types; `TSR`/`TDR` the remaining tuples.
impl<S, TSR, D, TDR> FinishCoalesce for Layout<IntTuple<S, TSR>, IntTuple<D, TDR>>
where
    IntTuple<S, TSR>: Shape,
    IntTuple<D, TDR>: Item<Profile = <IntTuple<S, TSR> as Item>::Profile>,
{
    type Output = Self;
    fn finish(self) -> Result<Self> {
        Layout::new(self.shape, self.stride)
    }
}

/// 类型化 coalesce 的结果类型；EXTENDED 选择保留扩展映射的版本。
pub type CoalescedLayout<S, D, const EXTENDED: bool = false> = <<Layout<
    <S as Flatten>::Output,
    <D as Flatten>::Output,
> as CoalesceFold<EXTENDED>>::Output as FinishCoalesce>::Output;

impl<S: Shape + Flatten, D: Item<Profile = S::Profile> + Flatten> Layout<S, D>
where
    Layout<S::Output, D::Output>: CoalesceFold,
    <Layout<S::Output, D::Output> as CoalesceFold>::Output: FinishCoalesce,
{
    /// Flattens and coalesces while preserving the logical linear address map.
    ///
    /// Removes static unit extents and merges adjacent leaves only when their
    /// static values prove `shape * stride == next_stride`. Static result leaves
    /// remain encoded in their types; runtime values never alter the tuple
    /// profile. An empty result is normalized to the scalar layout `1:0`.
    /// 合并要求尾部 shape 也为静态值，与 CuTe 的 bw_coalesce 一致。
    /// 输入 shape 和输出布局在运行时校验；静态合并乘积溢出在实例化时报错。
    pub fn coalesce(&self) -> Result<CoalescedLayout<S, D>> {
        self.size()?;
        self.clone().flatten().fold().finish()
    }
}

impl<S: Shape + Flatten, D: Item<Profile = S::Profile> + Flatten> Layout<S, D>
where
    Layout<S::Output, D::Output>: CoalesceFold<true>,
    <Layout<S::Output, D::Output> as CoalesceFold<true>>::Output: FinishCoalesce,
{
    /// 保留扩展域映射的类型化化简，支持静态和动态叶子混合。
    ///
    /// 与 CuTe coalesce_x 一样，末尾静态 shape-1 换成 2，动态 shape-1 保留原值。
    /// 因此结果的逻辑大小可能增大。输入非空且前导 extent 为正时，扩展映射保持不变。
    /// 合并仅使用静态信息；空输入归一为 1:0。
    /// 输入 shape 与输出的大小、地址范围都经过校验。
    ///
    /// ```
    /// use noodle_cute::{DynInt, Int, shape, stride};
    /// use noodle_cute::layout::{Layout, LayoutMap};
    /// let a = Layout::new(shape![Int::<2>, Int::<1>], stride![Int::<1>, DynInt(8)])?;
    /// let b = a.coalesce_extended()?;
    /// assert_eq!(b.size()?, 4);
    /// for i in 0..12 { assert_eq!(a.index_extended(i)?, b.index_extended(i)?); }
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn coalesce_extended(&self) -> Result<CoalescedLayout<S, D, true>> {
        self.size()?;
        self.clone().flatten().fold().finish()
    }
}

/// 全静态 coalesce 的结果类型与常量值，不经过运行时递归或 DynLayout。
///
/// EXTENDED 选择 CuTe coalesce_x；默认选择普通 coalesce。
pub trait StaticCoalesce<const EXTENDED: bool = false>: Copy {
    type Output: StaticValue;
    const OUTPUT: Self::Output = Self::Output::VALUE;
}
impl<S, D, const EXTENDED: bool> StaticCoalesce<EXTENDED> for Layout<S, D>
where
    S: Shape + StaticValue + Flatten,
    D: Item<Profile = S::Profile> + StaticValue + Flatten,
    Layout<S::Output, D::Output>: CoalesceFold<EXTENDED>,
    <Layout<S::Output, D::Output> as CoalesceFold<EXTENDED>>::Output: FinishCoalesce,
    CoalescedLayout<S, D, EXTENDED>: StaticValue,
{
    type Output = CoalescedLayout<S, D, EXTENDED>;
}

/// 在编译期化简全静态布局，静态单位轴被移除，连续模式被合并。
///
/// 空输入与全单位轴结果归一为 1:0。静态合并乘积溢出在实例化时报错。
/// 与 static_composition 一样，此入口不执行 Layout::new 的大小与地址范围校验。
///
/// ```
/// use noodle_cute::{Int, IntTuple, Nil, shape, stride};
/// use noodle_cute::layout::Layout;
/// use noodle_cute::algebra::static_coalesce;
/// const C: Layout<IntTuple<Int<6>, Nil>, IntTuple<Int<1>, Nil>> = static_coalesce(
///     Layout { shape: shape![Int::<2>, Int::<3>], stride: stride![Int::<1>, Int::<2>] }
/// );
/// ```
///
/// 动态叶子应使用 Layout::coalesce：
/// ```compile_fail
/// use noodle_cute::{DynInt, Int, layout::Layout, algebra::static_coalesce};
/// let _ = static_coalesce(Layout { shape: DynInt(3), stride: Int::<1> });
/// ```
///
/// 静态合并乘积必须能由 isize 表示：
/// ```compile_fail
/// use noodle_cute::{Int, shape, stride, layout::Layout, algebra::static_coalesce};
/// let _ = static_coalesce(Layout {
///     shape: shape![Int::<{ isize::MAX }>, Int::<2>],
///     stride: stride![Int::<0>, Int::<0>],
/// });
/// ```
pub const fn static_coalesce<L: StaticCoalesce>(_: L) -> L::Output {
    L::OUTPUT
}

/// 在编译期执行全静态 coalesce_x，保留扩展映射。
///
/// 末尾静态 shape-1 换成 2，因此结果大小可能增大。空输入归一为 1:0。
/// 扩展映射等价要求原布局的前导 extent 为正。
/// 与 static_coalesce 相同，此入口不执行运行时大小与地址范围校验。
///
/// ```
/// use noodle_cute::{Int, IntTuple, Nil, shape, stride};
/// use noodle_cute::layout::Layout;
/// use noodle_cute::algebra::static_coalesce_extended;
/// const C: Layout<IntTuple<Int<4>, Nil>, IntTuple<Int<1>, Nil>> = static_coalesce_extended(
///     Layout { shape: shape![Int::<2>, Int::<1>], stride: stride![Int::<1>, Int::<2>] }
/// );
/// ```
///
/// 动态 stride 也不能进入全静态入口：
/// ```compile_fail
/// use noodle_cute::{DynInt, Int, layout::Layout, algebra::static_coalesce_extended};
/// let _ = static_coalesce_extended(Layout { shape: Int::<1>, stride: DynInt(8) });
/// ```
pub const fn static_coalesce_extended<L: StaticCoalesce<true>>(_: L) -> L::Output {
    L::OUTPUT
}

impl DynLayout {
    /// CuTe `coalesce(layout)`: drops size-1 leaves and merges adjacent leaves
    /// when `s0 * d0 == d1`. A scalar or all-size-1 layout becomes `1:0`.
    ///
    /// The logical size and mapping on `[0, size)` are preserved. The extended
    /// mapping can change when a trailing size-1 leaf is removed.
    ///
    /// ```
    /// use noodle_cute::layout::{DynLayout, LayoutMap};
    ///
    /// let layout = DynLayout::new(&[2, 3, 1], &[1, 2, 99])?;
    /// let compact = layout.coalesce()?;
    /// assert_eq!(compact.shapes(), &[6]);
    /// assert_eq!(compact.strides(), &[1]);
    /// assert_eq!(layout.index(5)?, compact.index(5)?);
    /// // Removing the trailing unit axis changes extension beyond the domain.
    /// assert_eq!(layout.index_extended(6)?, 99);
    /// assert_eq!(compact.index_extended(6)?, 6);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn coalesce(&self) -> Result<Self> {
        self.coalesce_impl(false)
    }

    /// 对应 CuTe coalesce_x，保留决定扩展方向的末尾 stride。
    ///
    /// 末尾 shape-1 保留真实 extent，除非与前一叶子合并。因此 (2,1):(2,8)
    /// 保持不变，(2,1):(1,2) 化为 2:1。逻辑大小不变；输入非空且前导 extent
    /// 为正时扩展映射不变。零 extent 仍按 CuTe 的代数规则化简，不保证保留
    /// 原布局扩展解码因零除而产生的错误。
    /// 空输入归一为 1:0，额外允许正扩展索引并将其映射到地址 0。
    pub fn coalesce_extended(&self) -> Result<Self> {
        self.coalesce_impl(true)
    }

    // 运行时用真实末尾 extent 保留扩展步长，避免静态占位 2 引入额外溢出。
    fn coalesce_impl(&self, extended: bool) -> Result<Self> {
        let mut out = Self::scalar();
        for a in (0..self.rank).rev() {
            let (s, d) = (self.shape[a], self.stride[a]);
            if s == 1 && !(extended && a + 1 == self.rank) {
                continue;
            }
            if out.rank > 0 {
                let j = out.rank - 1;
                if merge_decision(Some(s), Some(d), Some(out.shape[j]), Some(out.stride[j]))
                    == MERGE
                {
                    out.shape[j] = mul(s, out.shape[j])?;
                    out.stride[j] = d;
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
        out.shape[..out.rank].reverse();
        out.stride[..out.rank].reverse();
        Ok(out)
    }
}
