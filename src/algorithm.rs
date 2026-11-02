//! Tensor algorithms: CuTe `copy`, `copy_if`, `fill`, `clear`, `axpby`,
//! `for_each`, `transform`, and `gemm` (`algorithm/*.hpp`).
//!
//! Like upstream's `tensor(i)` loops, the algorithms visit elements by
//! colexicographic index. They are `unsafe` for the same reason as
//! [`Tensor::read`]: a layout cannot prove that its addresses are valid or
//! free of races between threads.

use core::ops::{Add, Mul};

use crate::algebra::MaxCommonVector;
use crate::atom::{MmaAtom, MmaTraits, UniversalFma};
use crate::int::{DynInt, Integer};
use crate::layout::{
    ComposedLayout, Layout, LayoutShape, MAX_RANK, MapCoord, RecastView, StaticLeaves,
};
use crate::pointer::{Address, Advance, Iter, Read, RecastPtr, Write};
use crate::swizzle::Swizzle;
use crate::tensor::{Tensor, ViewEngine};
use crate::tuple::{Get, IntTuple, Nil, Product};

type View<I, L> = Tensor<ViewEngine<I>, L>;

/// A view addressed by colexicographic index: CuTe `size(tensor)` and `&tensor(i)`.
pub trait Indexed {
    /// The iterator at one element.
    type Ptr: Iter;
    /// The number of elements.
    fn size(&self) -> isize;
    /// The iterator at element `i`.
    fn at(&self, i: isize) -> Self::Ptr;
}

impl<I, L> Indexed for View<I, L>
where
    I: Advance<L::Output>,
    L: MapCoord<isize> + LayoutShape<Shape: Product<Output: Integer>>,
{
    type Ptr = I::Output;
    fn size(&self) -> isize {
        Tensor::size(self)
    }
    fn at(&self, i: isize) -> I::Output {
        self.data().advance(self.layout.map_coord(i))
    }
}

/// CuTe `copy_if(pred, src, dst)`: `dst(i) = src(i)` where `pred(i)`, converting with [`From`].
///
/// # Safety
///
/// Every element with a true predicate must be valid for its access and not
/// accessed by another thread concurrently.
pub unsafe fn copy_if<S, D>(pred: impl Fn(isize) -> bool, src: &S, dst: &D)
where
    S: Indexed<Ptr: Read>,
    D: Indexed<Ptr: Write<Value: From<<S::Ptr as Iter>::Value>>>,
{
    for i in 0..dst.size() {
        if pred(i) {
            // SAFETY: forwarded to the caller.
            unsafe { dst.at(i).write(From::from(src.at(i).read())) }
        }
    }
}

/// Explicit copy policy: CuTe's atom, tiled-copy, and thread-copy overloads.
///
/// Rust has no function overloading; use [`copy_with`] / [`copy_if_with`]
/// for explicit policies and [`copy`] / [`copy_if`] for automatic copies.
pub trait CopyPolicy<S, D> {
    /// Copies instruction groups selected by `pred(group_index)`.
    ///
    /// # Safety
    ///
    /// See [`copy_if_with`].
    unsafe fn copy_if_policy(&self, pred: impl Fn(isize) -> bool, src: &S, dst: &D);
}

/// Copies already partitioned tensors using an explicit atom or tiling policy.
///
/// # Safety
///
/// The tensors must satisfy the policy's register sizes, alignment, and access
/// requirements. See [`crate::atom::CopyAtom::call`].
pub unsafe fn copy_with<P: CopyPolicy<S, D>, S, D>(policy: &P, src: &S, dst: &D) {
    unsafe { policy.copy_if_policy(|_| true, src, dst) }
}

/// Predicated explicit copy, with one predicate per instruction group.
///
/// A false predicate skips the operation unless the backend overrides
/// [`crate::atom::CopyTraits::copy_predicated`] (for example, to zero-fill).
/// The group index follows the source's colexicographic instruction order.
///
/// # Safety
///
/// See [`copy_with`]. All values in one instruction must share its predicate;
/// a false predicate must satisfy the backend's destination access requirements.
pub unsafe fn copy_if_with<P: CopyPolicy<S, D>, S, D>(
    policy: &P,
    pred: impl Fn(isize) -> bool,
    src: &S,
    dst: &D,
) {
    unsafe { policy.copy_if_policy(pred, src, dst) }
}

/// The alignment of a layout in elements: CuTe `max_alignment(layout)` for a
/// static layout, or `None` when a shape or stride is dynamic.
pub trait CopyLayout {
    /// The alignment, if the layout is static.
    const ALIGN: Option<isize>;
}

const fn gcd(a: isize, b: isize) -> isize {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

// The contiguous extent reached from stride 1, as `size(right_inverse(layout))`,
// with the gcd of the remaining strides, which move between such runs.
const fn alignment(
    s: [Option<isize>; MAX_RANK],
    d: [Option<isize>; MAX_RANK],
    n: usize,
) -> Option<isize> {
    let mut i = 0;
    while i < n {
        if s[i].is_none() || d[i].is_none() {
            return None;
        }
        i += 1;
    }
    let mut used = [false; MAX_RANK];
    let mut extent = 1;
    'chain: loop {
        let mut j = 0;
        while j < n {
            if let (Some(shape), Some(stride)) = (s[j], d[j])
                && !used[j]
                && shape > 1
                && stride == extent
            {
                used[j] = true;
                extent *= shape;
                continue 'chain;
            }
            j += 1;
        }
        break;
    }
    let mut g = extent;
    let mut j = 0;
    while j < n {
        if let (Some(shape), Some(stride)) = (s[j], d[j])
            && !used[j]
            && shape > 1
        {
            g = gcd(g, stride);
        }
        j += 1;
    }
    Some(g)
}

impl<S: StaticLeaves, D: StaticLeaves> CopyLayout for Layout<S, D> {
    const ALIGN: Option<isize> = alignment(S::STATICS, D::STATICS, S::COUNT);
}

impl<const B: u32, const M: u32, const SH: i32, O: Integer, L: CopyLayout> CopyLayout
    for ComposedLayout<Swizzle<B, M, SH>, O, L>
{
    const ALIGN: Option<isize> = match (L::ALIGN, O::STATIC) {
        (Some(a), Some(o)) => Some(gcd(gcd(a, 1 << M), o)),
        _ => None,
    };
}

const fn vector_bits(common: isize, src: Option<isize>, dst: Option<isize>, bits: isize) -> isize {
    // Static layouts assume 128-bit aligned pointers; dynamic ones only 8 bits.
    let align = match (src, dst) {
        (Some(s), Some(d)) => gcd(gcd(s * bits, d * bits), 128),
        _ => 8,
    };
    let vec = gcd(common * bits, align);
    if common > 1 && vec % 8 == 0 && bits < vec {
        vec
    } else {
        0
    }
}

/// The vector width of [`copy`] in bits, or zero for an element-wise copy.
pub const VECTOR_BITS<SL: CopyLayout + MaxCommonVector<DL>, DL: CopyLayout, T>: isize = vector_bits(
    SL::VALUE,
    SL::ALIGN,
    DL::ALIGN,
    (size_of::<T>() * 8) as isize,
);

/// Selects the vector type of [`copy`] by its width in bits.
pub struct Vector<const BITS: isize>;

// Implementation traits that appear in public bounds but are not part of the API.
mod detail {
    pub trait VectorCopy<S, D> {
        unsafe fn copy(src: &S, dst: &D);
    }
}
use detail::VectorCopy;

impl<S, D, T> VectorCopy<S, D> for Vector<0>
where
    S: Indexed<Ptr: Read<Value = T>>,
    D: Indexed<Ptr: Write<Value = T>>,
{
    unsafe fn copy(src: &S, dst: &D) {
        // SAFETY: forwarded to the caller.
        unsafe { copy_if(|_| true, src, dst) }
    }
}

macro_rules! vector_copy {
    ($($bits:literal => $v:ty),*) => {$(
        impl<SI, SL, DI, DL, T> VectorCopy<View<SI, SL>, View<DI, DL>> for Vector<$bits>
        where
            SI: Iter<Value = T> + Address + Advance<DynInt>,
            DI: Iter<Value = T> + Address + Advance<DynInt>,
            View<SI, SL>: Indexed<Ptr: Read<Value = T>>,
            View<DI, DL>: Indexed<Ptr: Write<Value = T>>,
            SL: RecastView<T, $v> + Clone,
            DL: RecastView<T, $v> + Clone,
            SI::Output: RecastPtr<$v>,
            DI::Output: RecastPtr<$v>,
            View<<SI::Output as RecastPtr<$v>>::Output, SL::Output>: Indexed<Ptr: Read<Value = $v>>,
            View<<DI::Output as RecastPtr<$v>>::Output, DL::Output>: Indexed<Ptr: Write<Value = $v>>,
        {
            unsafe fn copy(src: &View<SI, SL>, dst: &View<DI, DL>) {
                let bytes = $bits / 8;
                // SAFETY: forwarded to the caller; vectors are only used at aligned addresses.
                unsafe {
                    if src.data().addr() % bytes == 0 && dst.data().addr() % bytes == 0 {
                        copy_if(|_| true, &src.clone().recast::<$v>(), &dst.clone().recast::<$v>())
                    } else {
                        copy_if(|_| true, src, dst)
                    }
                }
            }
        }
    )*};
}

vector_copy!(16 => u16, 32 => u32, 64 => u64, 128 => u128);

/// [`copy`] for a pair of views; see there.
pub trait AutoCopy<D> {
    /// Copies `self` into `dst`.
    ///
    /// # Safety
    ///
    /// See [`copy`].
    unsafe fn auto_copy(&self, dst: &D);
}

impl<SI, SL, DI, DL, T> AutoCopy<View<DI, DL>> for View<SI, SL>
where
    SI: Iter<Value = T>,
    DI: Iter<Value = T>,
    SL: CopyLayout + MaxCommonVector<DL>,
    DL: CopyLayout,
    Vector<{ VECTOR_BITS::<SL, DL, T> }>: VectorCopy<Self, View<DI, DL>>,
{
    unsafe fn auto_copy(&self, dst: &View<DI, DL>) {
        // SAFETY: forwarded to the caller.
        unsafe { <Vector<{ VECTOR_BITS::<SL, DL, T> }> as VectorCopy<_, _>>::copy(self, dst) }
    }
}

/// CuTe `copy(src, dst)`: an element-wise copy, vectorized where both layouts allow.
///
/// As upstream, when both layouts are static the copy moves
/// `gcd(max_common_vector * bits, alignment of both layouts, 128)` bits at a
/// time through unsigned integers of that width, if that is wider than one
/// element; layouts with a dynamic shape or stride are copied element by
/// element. Unlike upstream, which assumes 128-bit aligned pointers for static
/// layouts, the pointers are checked at runtime and misaligned tensors fall
/// back to element-wise copies. The source and destination must have the same
/// element type; [`copy_if`] converts. Upstream's filtering of broadcast
/// (stride-0) destination modes is not performed, so such elements are written
/// repeatedly with the same values.
///
/// # Safety
///
/// Every element must be valid for its access and not accessed by another
/// thread concurrently.
///
/// ```
/// use noodle_cute::algorithm::copy;
/// use noodle_cute::layout::LayoutLeft;
/// use noodle_cute::tensor::make_owned_tensor;
/// use noodle_cute::{Int, shape};
///
/// let layout = LayoutLeft::make(shape![Int::<4>, Int::<8>])?;
/// let mut src = make_owned_tensor::<f32, _>(layout);
/// let mut dst = make_owned_tensor::<f32, _>(layout);
/// src.engine.0[7] = 1.5;
/// // Both fragments are contiguous and 16-byte aligned: eight 128-bit copies.
/// unsafe { copy(&src.view(), &dst.view_mut()) };
/// assert_eq!(dst.engine.0[7], 1.5);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
pub unsafe fn copy<S: AutoCopy<D>, D>(src: &S, dst: &D) {
    // SAFETY: forwarded to the caller.
    unsafe { src.auto_copy(dst) }
}

/// CuTe `fill(tensor, value)`.
///
/// # Safety
///
/// Every element must be valid for writes and not accessed by another thread concurrently.
pub unsafe fn fill<T: Indexed<Ptr: Write<Value: Clone>>>(
    tensor: &T,
    value: <T::Ptr as Iter>::Value,
) {
    for i in 0..tensor.size() {
        // SAFETY: forwarded to the caller.
        unsafe { tensor.at(i).write(value.clone()) }
    }
}

/// CuTe `clear(tensor)`: fills with the default value.
///
/// # Safety
///
/// See [`fill`].
pub unsafe fn clear<T: Indexed<Ptr: Write<Value: Clone + Default>>>(tensor: &T) {
    // SAFETY: forwarded to the caller.
    unsafe { fill(tensor, Default::default()) }
}

/// CuTe `axpby(alpha, x, beta, y)`: `y = alpha * x + beta * y`.
///
/// As upstream, `y` is not read when `beta` is zero (its default value).
///
/// # Safety
///
/// Every element must be valid for its access and not accessed by another
/// thread concurrently.
pub unsafe fn axpby<X, Y, A, B>(alpha: A, x: &X, beta: B, y: &Y)
where
    X: Indexed<Ptr: Read>,
    Y: Indexed<Ptr: Read + Write<Value: Add<Output = <Y::Ptr as Iter>::Value>>>,
    A: Copy + Mul<<X::Ptr as Iter>::Value, Output = <Y::Ptr as Iter>::Value>,
    B: Copy + PartialEq + Default + Mul<<Y::Ptr as Iter>::Value, Output = <Y::Ptr as Iter>::Value>,
{
    let beta_is_zero = beta == B::default();
    for i in 0..x.size() {
        // SAFETY: forwarded to the caller.
        unsafe {
            let ax = alpha * x.at(i).read();
            let out = if beta_is_zero {
                ax
            } else {
                ax + beta * y.at(i).read()
            };
            y.at(i).write(out);
        }
    }
}

/// CuTe `for_each(tensor, op)`, passing each element by value.
///
/// # Safety
///
/// Every element must be valid for reads and not written by another thread concurrently.
pub unsafe fn for_each<T: Indexed<Ptr: Read>>(
    tensor: &T,
    mut op: impl FnMut(<T::Ptr as Iter>::Value),
) {
    for i in 0..tensor.size() {
        // SAFETY: forwarded to the caller.
        op(unsafe { tensor.at(i).read() })
    }
}

/// CuTe `transform(tensor, op)`: replaces each element by `op` of it.
///
/// # Safety
///
/// Every element must be valid for reads and writes and not accessed by
/// another thread concurrently.
pub unsafe fn transform<T>(
    tensor: &T,
    mut op: impl FnMut(<T::Ptr as Iter>::Value) -> <T::Ptr as Iter>::Value,
) where
    T: Indexed<Ptr: Read + Write>,
{
    for i in 0..tensor.size() {
        // SAFETY: forwarded to the caller.
        unsafe { tensor.at(i).write(op(tensor.at(i).read())) }
    }
}

/// CuTe `transform(in, out, op)`: `out(i) = op(in(i))`.
///
/// # Safety
///
/// Every element must be valid for its access and not accessed by another
/// thread concurrently.
pub unsafe fn transform_into<S, D>(
    src: &S,
    dst: &D,
    mut op: impl FnMut(<S::Ptr as Iter>::Value) -> <D::Ptr as Iter>::Value,
) where
    S: Indexed<Ptr: Read>,
    D: Indexed<Ptr: Write>,
{
    for i in 0..src.size() {
        // SAFETY: forwarded to the caller.
        unsafe { dst.at(i).write(op(src.at(i).read())) }
    }
}

/// CuTe `transform(in1, in2, out, op)`: `out(i) = op(in1(i), in2(i))`.
///
/// # Safety
///
/// Every element must be valid for its access and not accessed by another
/// thread concurrently.
pub unsafe fn transform_zip<S1, S2, D>(
    a: &S1,
    b: &S2,
    dst: &D,
    mut op: impl FnMut(<S1::Ptr as Iter>::Value, <S2::Ptr as Iter>::Value) -> <D::Ptr as Iter>::Value,
) where
    S1: Indexed<Ptr: Read>,
    S2: Indexed<Ptr: Read>,
    D: Indexed<Ptr: Write>,
{
    for i in 0..a.size() {
        // SAFETY: forwarded to the caller.
        unsafe { dst.at(i).write(op(a.at(i).read(), b.at(i).read())) }
    }
}

/// The rank of a shape or of a view's shape; an integer is rank one.
pub struct Rank<const N: usize>;

/// The [`Rank`] of a shape, for [`gemm`] dispatch.
pub trait RankOf {
    /// The rank marker.
    type Rank;
}

impl<I: Integer> RankOf for I {
    type Rank = Rank<1>;
}

impl<A> RankOf for IntTuple<A, Nil> {
    type Rank = Rank<1>;
}

impl<A, B> RankOf for IntTuple<A, IntTuple<B, Nil>> {
    type Rank = Rank<2>;
}

impl<A, B, C> RankOf for IntTuple<A, IntTuple<B, IntTuple<C, Nil>>> {
    type Rank = Rank<3>;
}

impl<I, L: LayoutShape<Shape: RankOf>> RankOf for View<I, L> {
    type Rank = <L::Shape as RankOf>::Rank;
}

type C2 = IntTuple<isize, IntTuple<isize, Nil>>;
type C3 = IntTuple<isize, C2>;

fn c2(x: isize, y: isize) -> C2 {
    IntTuple(x, IntTuple(y, Nil))
}

fn mode<S: Get<I, Output: Product<Output: Integer> + Clone>, const I: usize>(shape: &S) -> isize {
    shape.get().clone().product_typed().value()
}

// Implementation traits that appear in public bounds but are not part of the API.
mod operand {
    /// How a view of this rank is read as `(v, x, y)` when the accumulator has rank `RC`.
    pub trait OperandKind<RC, I, L> {
        type Ptr;
        fn extents(layout: &L) -> [isize; 3];
        fn ptr(data: I, layout: &L, v: isize, x: isize, y: isize) -> Self::Ptr;
    }
}
use operand::OperandKind;

type Shape<L> = <L as LayoutShape>::Shape;

// (V) with a rank-1 accumulator: [1] `(V) x (V) => (V)`.
impl<I, L> OperandKind<Rank<1>, I, L> for Rank<1>
where
    I: Advance<<L as MapCoord<isize>>::Output>,
    L: MapCoord<isize> + LayoutShape<Shape: Product<Output: Integer>>,
{
    type Ptr = I::Output;
    fn extents(layout: &L) -> [isize; 3] {
        [layout.shape().product_typed().value(), 1, 1]
    }
    fn ptr(data: I, layout: &L, v: isize, _: isize, _: isize) -> I::Output {
        data.advance(layout.map_coord(v))
    }
}

// (M) or (N) with a rank-2 accumulator: [2] `(M) x (N) => (M, N)`.
impl<I, L> OperandKind<Rank<2>, I, L> for Rank<1>
where
    I: Advance<<L as MapCoord<isize>>::Output>,
    L: MapCoord<isize> + LayoutShape<Shape: Product<Output: Integer>>,
{
    type Ptr = I::Output;
    fn extents(layout: &L) -> [isize; 3] {
        [1, layout.shape().product_typed().value(), 1]
    }
    fn ptr(data: I, layout: &L, _: isize, x: isize, _: isize) -> I::Output {
        data.advance(layout.map_coord(x))
    }
}

// (M, K), (N, K), or (M, N) with a rank-2 accumulator: [2] after `append`, and [3].
impl<I, L> OperandKind<Rank<2>, I, L> for Rank<2>
where
    I: Advance<<L as MapCoord<C2>>::Output>,
    L: MapCoord<C2> + LayoutShape<Shape: Get<0, Output: Product<Output: Integer> + Clone>>,
    Shape<L>: Get<1, Output: Product<Output: Integer> + Clone>,
{
    type Ptr = I::Output;
    fn extents(layout: &L) -> [isize; 3] {
        let shape = layout.shape();
        [1, mode::<_, 0>(&shape), mode::<_, 1>(&shape)]
    }
    fn ptr(data: I, layout: &L, _: isize, x: isize, y: isize) -> I::Output {
        data.advance(layout.map_coord(c2(x, y)))
    }
}

// (V, M) or (V, N) with a rank-3 accumulator: [4] `(V, M) x (V, N) => (V, M, N)`.
impl<I, L> OperandKind<Rank<3>, I, L> for Rank<2>
where
    I: Advance<<L as MapCoord<C2>>::Output>,
    L: MapCoord<C2> + LayoutShape<Shape: Get<0, Output: Product<Output: Integer> + Clone>>,
    Shape<L>: Get<1, Output: Product<Output: Integer> + Clone>,
{
    type Ptr = I::Output;
    fn extents(layout: &L) -> [isize; 3] {
        let shape = layout.shape();
        [mode::<_, 0>(&shape), mode::<_, 1>(&shape), 1]
    }
    fn ptr(data: I, layout: &L, v: isize, x: isize, _: isize) -> I::Output {
        data.advance(layout.map_coord(c2(v, x)))
    }
}

// (V, M, K), (V, N, K), or (V, M, N): [4] for the accumulator and [5].
impl<I, L> OperandKind<Rank<3>, I, L> for Rank<3>
where
    I: Advance<<L as MapCoord<C3>>::Output>,
    L: MapCoord<C3> + LayoutShape<Shape: Get<0, Output: Product<Output: Integer> + Clone>>,
    Shape<L>: Get<1, Output: Product<Output: Integer> + Clone>,
    Shape<L>: Get<2, Output: Product<Output: Integer> + Clone>,
{
    type Ptr = I::Output;
    fn extents(layout: &L) -> [isize; 3] {
        let shape = layout.shape();
        [
            mode::<_, 0>(&shape),
            mode::<_, 1>(&shape),
            mode::<_, 2>(&shape),
        ]
    }
    fn ptr(data: I, layout: &L, v: isize, x: isize, y: isize) -> I::Output {
        data.advance(layout.map_coord(IntTuple(v, c2(x, y))))
    }
}

/// A [`gemm`] operand read as `(v, x, y)`, according to its rank and the rank `RC` of C.
pub trait GemmOperand<RC> {
    /// The iterator at one element.
    type Ptr: Iter;
    /// The extents of `(V, X, Y)`; absent modes have extent one.
    fn extents(&self) -> [isize; 3];
    /// The iterator at `(v, x, y)`.
    fn ptr(&self, v: isize, x: isize, y: isize) -> Self::Ptr;
}

impl<I: Iter, L: LayoutShape<Shape: RankOf>, RC> GemmOperand<RC> for View<I, L>
where
    <L::Shape as RankOf>::Rank: OperandKind<RC, I, L, Ptr: Iter>,
{
    type Ptr = <<L::Shape as RankOf>::Rank as OperandKind<RC, I, L>>::Ptr;
    fn extents(&self) -> [isize; 3] {
        <<L::Shape as RankOf>::Rank as OperandKind<RC, I, L>>::extents(&self.layout)
    }
    fn ptr(&self, v: isize, x: isize, y: isize) -> Self::Ptr {
        <<L::Shape as RankOf>::Rank as OperandKind<RC, I, L>>::ptr(
            self.data(),
            &self.layout,
            v,
            x,
            y,
        )
    }
}

/// Values per register of type `R`.
const fn per<R, V>() -> isize {
    let (r, v) = (size_of::<R>(), size_of::<V>());
    assert!(v > 0 && r % v == 0, "a register must hold whole values");
    (r / v) as isize
}

// Upstream's register-reuse orders over (M, N) for dispatch [4], chosen by the
// bytes of one A and one B instruction operand.
fn traverse(
    m_count: isize,
    n_count: isize,
    a_bytes: isize,
    b_bytes: isize,
    mut f: impl FnMut(isize, isize),
) {
    match (a_bytes, b_bytes) {
        (8, 8) | (8, 4) => {
            for m in 0..m_count {
                for n in 0..n_count {
                    f(m, if m & 1 != 0 { n_count - 1 - n } else { n });
                }
            }
        }
        (4, 4) => {
            for m in (0..m_count).step_by(2) {
                for n in 0..n_count {
                    let ns = if m & 2 != 0 { n_count - 1 - n } else { n };
                    f(m, ns);
                    if m + 1 < m_count {
                        f(m + 1, ns);
                    }
                }
            }
        }
        _ => {
            for n in 0..n_count {
                for m in 0..m_count {
                    f(if n & 1 != 0 { m_count - 1 - m } else { m }, n);
                }
            }
        }
    }
}

type Value<P> = <P as Iter>::Value;

/// CuTe `gemm(mma, D, A, B, C)`: `D = A * B + C` on thread-local fragments.
///
/// Upstream's register dispatches are selected by the ranks of A, B, and C:
///
/// | Dispatch | A | B | C, D |
/// |---|---|---|---|
/// | 1 | `(V)` | `(V)` | `(V)` |
/// | 2 | `(M)` | `(N)` | `(M, N)` |
/// | 3 | `(M, K)` | `(N, K)` | `(M, N)` |
/// | 4 | `(V, M)` | `(V, N)` | `(V, M, N)` |
/// | 5 | `(V, M, K)` | `(V, N, K)` | `(V, M, N)` |
///
/// Dispatches 2 and 3 are for one-value instructions, where V is one.
/// Every form runs as dispatch 5: for each `k`, the `(m, n)` instructions are issued
/// in upstream's serpentine order for the operand widths, and each computes
/// `D(_, m, n) = A(_, m, k) * B(_, n, k) + C(_, m, n)`. As upstream, C rather
/// than D is added for every `k`, so accumulate by passing the same tensor as
/// C and D. The instruction's registers are read from the V mode as in
/// [`MmaAtom::call`]. Upstream's shared-memory variants, which copy operands
/// to registers first, are not provided.
///
/// # Safety
///
/// Every element must be valid for its access and not accessed by another
/// thread concurrently; D may alias C.
pub unsafe fn gemm<Op, D, A, B, C>(_: &MmaAtom<Op>, d: &D, a: &A, b: &B, c: &C)
where
    Op: MmaTraits,
    C: RankOf + GemmOperand<C::Rank>,
    D: GemmOperand<C::Rank, Ptr: RecastPtr<Op::DRegister, Output: Write<Value = Op::DRegister>>>,
    A: GemmOperand<C::Rank, Ptr: RecastPtr<Op::ARegister, Output: Read<Value = Op::ARegister>>>,
    B: GemmOperand<C::Rank, Ptr: RecastPtr<Op::BRegister, Output: Read<Value = Op::BRegister>>>,
    C::Ptr: RecastPtr<Op::CRegister, Output: Read<Value = Op::CRegister>>,
{
    let pd = const { per::<Op::DRegister, Value<D::Ptr>>() };
    let pa = const { per::<Op::ARegister, Value<A::Ptr>>() };
    let pb = const { per::<Op::BRegister, Value<B::Ptr>>() };
    let pc = const { per::<Op::CRegister, Value<C::Ptr>>() };
    let [va, m_count, k_count] = a.extents();
    let [vb, n_count, _] = b.extents();
    let a_bytes = va * size_of::<Value<A::Ptr>>() as isize;
    let b_bytes = vb * size_of::<Value<B::Ptr>>() as isize;
    for k in 0..k_count {
        traverse(m_count, n_count, a_bytes, b_bytes, |m, n| {
            // SAFETY: forwarded to the caller.
            unsafe {
                Op::fma(
                    |r| d.ptr(r * pd, m, n).recast_ptr(),
                    |r| a.ptr(r * pa, m, k).recast_ptr(),
                    |r| b.ptr(r * pb, n, k).recast_ptr(),
                    |r| c.ptr(r * pc, m, n).recast_ptr(),
                )
            }
        });
    }
}

type Fma<A, B, C> = UniversalFma<
    Value<<C as GemmOperand<<C as RankOf>::Rank>>::Ptr>,
    Value<<A as GemmOperand<<C as RankOf>::Rank>>::Ptr>,
    Value<<B as GemmOperand<<C as RankOf>::Rank>>::Ptr>,
    Value<<C as GemmOperand<<C as RankOf>::Rank>>::Ptr>,
>;

/// CuTe `gemm(A, B, C)`: `C += A * B` with [`UniversalFma`] on the element types.
///
/// # Safety
///
/// See [`gemm`].
///
/// ```
/// use noodle_cute::algorithm::gemm_fma;
/// use noodle_cute::layout::LayoutLeft;
/// use noodle_cute::tensor::make_owned_tensor;
/// use noodle_cute::{Int, shape};
///
/// // (M, K) x (N, K) => (M, N) with M = 2, N = 3, K = 4.
/// let mut a = make_owned_tensor::<f32, _>(LayoutLeft::make(shape![Int::<2>, Int::<4>])?);
/// let mut b = make_owned_tensor::<f32, _>(LayoutLeft::make(shape![Int::<3>, Int::<4>])?);
/// let mut c = make_owned_tensor::<f32, _>(LayoutLeft::make(shape![Int::<2>, Int::<3>])?);
/// a.engine.0.iter_mut().enumerate().for_each(|(i, x)| *x = i as f32);
/// b.engine.0.fill(1.0);
/// unsafe { gemm_fma(&a.view(), &b.view(), &c.view_mut()) };
/// // c(1, n) = sum_k a(1, k) = 1 + 3 + 5 + 7.
/// assert_eq!(c.get(shape![1i32, 2i32]), 16.0);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
pub unsafe fn gemm_fma<A, B, C>(a: &A, b: &B, c: &C)
where
    C: RankOf + GemmOperand<C::Rank, Ptr: Read + Write>,
    A: GemmOperand<C::Rank, Ptr: Read>,
    B: GemmOperand<C::Rank, Ptr: Read>,
    Fma<A, B, C>: MmaTraits<
            DRegister = Value<C::Ptr>,
            ARegister = Value<A::Ptr>,
            BRegister = Value<B::Ptr>,
            CRegister = Value<C::Ptr>,
        >,
    C::Ptr: RecastPtr<Value<C::Ptr>, Output: Read<Value = Value<C::Ptr>> + Write<Value = Value<C::Ptr>>>,
    A::Ptr: RecastPtr<Value<A::Ptr>, Output: Read<Value = Value<A::Ptr>>>,
    B::Ptr: RecastPtr<Value<B::Ptr>, Output: Read<Value = Value<B::Ptr>>>,
{
    // SAFETY: forwarded to the caller.
    unsafe { gemm(&MmaAtom::<Fma<A, B, C>>::new(), c, a, b, c) }
}
