//! Copy and MMA atoms and their thread tilings: CuTe `Copy_Traits`,
//! `Copy_Atom`, `TiledCopy`, `ThrCopy`, `MMA_Traits`, `MMA_Atom`, `TiledMMA`,
//! and `ThrMMA` (`atom/*.hpp`), with the portable `UniversalCopy` and
//! `UniversalFMA` operations (`arch/copy.hpp`, `arch/mma.hpp`).
//!
//! A trait type describes one instruction: which logical threads take part
//! (`ThrId`), the thread–value layouts of its operands, and how to execute it
//! on registers. Architecture-specific instructions implement the same traits
//! in a backend crate. The tilings arrange atoms over threads and partition
//! tensors per thread with the layout algebra; all atom and tiling layouts are
//! static, while the partitioned tensors may have runtime extents and strides.
//!
//! Like other tensor partitions, the tilings return `Result` from the checked
//! typed algebra, including basis-strided identity tensors for tail predicates.
//! MMA-compatible copies and fragment retiling are implemented in `atom/copy.rs`.

mod copy;
pub use copy::{
    MmaCopyLayout, MmaCopyTile, PredicatedCopy, Retile, make_tiled_copy_a, make_tiled_copy_b,
    make_tiled_copy_c,
};

use core::marker::PhantomData;
use core::ops::{Add, Mul};

use crate::algebra::{
    CoalesceProfile, ComposeTile, LogicalDivide, PadTo, ProductKind, TypedCompose,
    TypedComposition, TypedInverse, TypedInvert, TypedProduct, TypedProductLayout, Zip2,
    ZippedDivide,
};
use crate::coord::{Crd2Crd, Idx2Crd, PadModes, RepeatModes, Underscore};
use crate::int::{Arith, Int, Integer, Op as ArithOp, Prod, prod};
use crate::layout::{
    ColMajorLayout, CompactLeft, FragmentLike, Layout, LayoutParts, LayoutShape, MapCoord, SliceAt,
    Upcast,
};
use crate::pointer::{Advance, Iter, Read, RecastPtr, Write};
use crate::tensor::{
    OwnedStorage, Tensor, ViewEngine, make_fragment_like, make_owned_tensor, make_tensor,
};
use crate::tuple::{Get, IntTuple, Nil, Product, ProductEach, Wrap};
use crate::{Result, StaticValue};

/// CuTe `sizeof_bits_v<T>`; Rust element types occupy whole bytes.
pub const SIZEOF_BITS<T>: isize = (size_of::<T>() * 8) as isize;

/// One copy instruction: CuTe `Copy_Traits<CopyOp>` with the operation's
/// `SRegisters` / `DRegisters` and `copy`.
///
/// `SrcLayout`, `DstLayout`, and `RefLayout` map `(thread, value)` to bits;
/// [`CopyAtom`] rescales them to elements of its value type.
pub trait CopyTraits {
    /// `(thread, value) -> bit` of the source.
    type SrcLayout;
    /// `(thread, value) -> bit` of the destination.
    type DstLayout;
    /// The reference `(thread, value) -> bit` map tilings are expressed in.
    type RefLayout;
    /// The type of one source register.
    type SRegister;
    /// The type of one destination register.
    type DRegister;
    /// The number of source registers.
    const SRC_REGISTERS: isize;
    /// The number of destination registers.
    const DST_REGISTERS: isize;
    /// Executes the instruction on registers `0..SRC_REGISTERS` and `0..DST_REGISTERS`.
    ///
    /// # Safety
    ///
    /// Every register address must be valid for its access, and no other
    /// thread may write them concurrently.
    unsafe fn copy<SP, DP>(src: impl Fn(isize) -> SP, dst: impl Fn(isize) -> DP)
    where
        SP: Read<Value = Self::SRegister>,
        DP: Write<Value = Self::DRegister>;

    /// Predicated instruction call. Backends may override this for zero-fill.
    ///
    /// The default skips a false predicate without evaluating either pointer.
    ///
    /// # Safety
    ///
    /// The registers accessed by the operation must satisfy [`Self::copy`].
    unsafe fn copy_predicated<SP, DP>(
        pred: bool,
        src: impl Fn(isize) -> SP,
        dst: impl Fn(isize) -> DP,
    ) where
        SP: Read<Value = Self::SRegister>,
        DP: Write<Value = Self::DRegister>,
    {
        if pred {
            unsafe { Self::copy(src, dst) }
        }
    }
}

/// Copies one element per instruction: CuTe `UniversalCopy<S, D>`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UniversalCopy<S, D = S>(PhantomData<(S, D)>);

type OneThread = Layout<Int<1>, Int<0>>;
type BitsLayout<T> = Layout<
    IntTuple<Int<1>, IntTuple<Int<{ SIZEOF_BITS::<T> }>, Nil>>,
    IntTuple<Int<0>, IntTuple<Int<1>, Nil>>,
>;

impl<S, D: From<S>> CopyTraits for UniversalCopy<S, D> {
    type SrcLayout = BitsLayout<S>;
    type DstLayout = BitsLayout<D>;
    type RefLayout = BitsLayout<S>;
    type SRegister = S;
    type DRegister = D;
    const SRC_REGISTERS: isize = 1;
    const DST_REGISTERS: isize = 1;
    unsafe fn copy<SP, DP>(src: impl Fn(isize) -> SP, dst: impl Fn(isize) -> DP)
    where
        SP: Read<Value = S>,
        DP: Write<Value = D>,
    {
        // SAFETY: forwarded to the caller.
        unsafe { dst(0).write(D::from(src(0).read())) }
    }
}

/// A copy instruction applied to values of type `V`: CuTe `Copy_Atom<CopyOp, V>`.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct CopyAtom<Op, V>(PhantomData<(Op, V)>);

impl<Op, V> Clone for CopyAtom<Op, V> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<Op, V> Copy for CopyAtom<Op, V> {}

impl<Op, V> CopyAtom<Op, V> {
    /// The atom of `Op` on values of type `V`.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// A bit layout rescaled to elements of `V`: CuTe `recast_layout<uint1_t, V>`.
type ValLayout<L, V> =
    Layout<<L as Upcast<{ SIZEOF_BITS::<V> }>>::Shape, <L as Upcast<{ SIZEOF_BITS::<V> }>>::Stride>;
type Mode<L, const I: usize> = <<L as LayoutParts>::Shape as Get<I>>::Output;
/// The number of threads or values of a static `(thread, value)` layout, as a static integer.
pub type ModeSize<L, const I: usize> = <Mode<L, I> as Product>::Output;

/// The `(ref thread, ref value) -> (src thread, src value)` map of an atom:
/// `right_inverse(ref).compose(src)`.
type RefToTrg<Ref, Trg> = TypedComposition<TypedInverse<Ref>, Trg>;

/// Copy atoms tiled over threads: CuTe `TiledCopy<Copy_Atom, LayoutCopy_TV, ShapeTiler_MN>`.
///
/// `layout_tv` maps `(thread, value)` to a coordinate in the tile `tiler`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TiledCopy<Op, V, TV, Tiler> {
    /// The atom.
    pub atom: CopyAtom<Op, V>,
    /// `(thread, value) -> tile coordinate`.
    pub layout_tv: TV,
    /// The tile shape partitioned by one application of the tiling.
    pub tiler: Tiler,
}

type MnLayout<TS, TD, VS, VD> =
    TypedProductLayout<Layout<TS, TD>, Layout<VS, VD>, { ProductKind::Raked }>;
type ThrVal<TS, VS> = IntTuple<<TS as Product>::Output, IntTuple<<VS as Product>::Output, Nil>>;
type TvLayout<TS, TD, VS, VD> =
    TypedComposition<TypedInverse<MnLayout<TS, TD, VS, VD>>, ColMajorLayout<ThrVal<TS, VS>>>;
type MadeCopy<Op, V, TS, TD, VS, VD> =
    TiledCopy<Op, V, TvLayout<TS, TD, VS, VD>, Tiler<TS, TD, VS, VD>>;
type Tiler<TS, TD, VS, VD> =
    <<MnLayout<TS, TD, VS, VD> as LayoutParts>::Shape as ProductEach>::Output;

/// CuTe `make_tiled_copy(atom, thr_layout, val_layout)`.
///
/// `thr_layout` maps a tile coordinate to a thread index and `val_layout` to a
/// value index. The `(M, N) -> (thread, value)` map is their
/// `raked_product`; its right inverse, reshaped to `(threads, values)`, is the
/// tiling's TV layout, and the tiler is the product's per-mode size. All three
/// layouts must be static.
///
/// ```
/// use noodle_cute::atom::{CopyAtom, UniversalCopy, make_tiled_copy};
/// use noodle_cute::layout::LayoutLeft;
/// use noodle_cute::{Int, Layout, shape, stride};
///
/// // 32 threads in a 4x8 grid, each copying 4 consecutive rows.
/// let atom = CopyAtom::<UniversalCopy<f32>, f32>::new();
/// let threads = LayoutLeft::make(shape![Int::<4>, Int::<8>])?;
/// let values = LayoutLeft::make(shape![Int::<4>, Int::<1>])?;
/// let tiled = make_tiled_copy(atom, threads, values)?;
/// assert_eq!(tiled.tiler, shape![Int::<16>, Int::<8>]);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
pub fn make_tiled_copy<Op, V, TS, TD, VS, VD>(
    atom: CopyAtom<Op, V>,
    thr_layout: Layout<TS, TD>,
    val_layout: Layout<VS, VD>,
) -> Result<MadeCopy<Op, V, TS, TD, VS, VD>>
where
    TS: Product,
    VS: Product,
    ThrVal<TS, VS>: CompactLeft,
    Layout<TS, TD>: TypedProduct<Layout<VS, VD>, { ProductKind::Raked }>,
    MnLayout<TS, TD, VS, VD>: TypedInvert + LayoutParts + Clone,
    <MnLayout<TS, TD, VS, VD> as LayoutParts>::Shape: ProductEach + Clone,
    TypedInverse<MnLayout<TS, TD, VS, VD>>: TypedCompose<ColMajorLayout<ThrVal<TS, VS>>>,
{
    let layout_mn = thr_layout.product_typed(val_layout)?;
    let layout_tv = TypedInvert::invert(layout_mn.clone())?
        .compose(<ColMajorLayout<ThrVal<TS, VS>> as StaticValue>::VALUE)?;
    let tiler = layout_mn.into_layout().shape.product_each();
    Ok(TiledCopy {
        atom,
        layout_tv,
        tiler,
    })
}

/// The `(thread, value) -> element` layouts of a copy atom: CuTe
/// `Copy_Atom::ValLayoutRef`, `ValLayoutSrc`, and `ValLayoutDst`.
pub trait AtomValLayouts {
    /// The reference layout.
    type Ref;
    /// The source layout.
    type Src;
    /// The destination layout.
    type Dst;
}

impl<Op: CopyTraits, V> AtomValLayouts for CopyAtom<Op, V>
where
    Op::RefLayout: Upcast<{ SIZEOF_BITS::<V> }>,
    Op::SrcLayout: Upcast<{ SIZEOF_BITS::<V> }>,
    Op::DstLayout: Upcast<{ SIZEOF_BITS::<V> }>,
{
    type Ref = ValLayout<Op::RefLayout, V>;
    type Src = ValLayout<Op::SrcLayout, V>;
    type Dst = ValLayout<Op::DstLayout, V>;
}

impl<Op: CopyTraits, V> CopyAtom<Op, V> {
    /// CuTe `Copy_Atom::call(src, dst)`: copies with one instruction per group
    /// of values.
    ///
    /// The tensors are read in colexicographic order and recast to the
    /// operation's registers, as upstream `copy_unpack` does. Where upstream
    /// peels the first mode until it matches the instruction, the elements are
    /// split here into consecutive groups of the instruction's value count,
    /// which must divide the source size.
    ///
    /// # Safety
    ///
    /// Every element must be valid for its access and not accessed by another
    /// thread concurrently, and the values of one register must be contiguous
    /// and aligned for the register type.
    pub unsafe fn call<SI, SL, DI, DL>(&self, src: &View<SI, SL>, dst: &View<DI, DL>)
    where
        SI: Advance<SL::Output>,
        SL: MapCoord<isize> + LayoutShape<Shape: Product<Output: Integer>>,
        Elem<SI, SL>: RecastPtr<Op::SRegister>,
        Reg<SI, SL, Op::SRegister>: Read<Value = Op::SRegister>,
        DI: Advance<DL::Output>,
        DL: MapCoord<isize> + LayoutShape<Shape: Product<Output: Integer>>,
        Elem<DI, DL>: RecastPtr<Op::DRegister>,
        Reg<DI, DL, Op::DRegister>: Write<Value = Op::DRegister>,
    {
        unsafe { self.call_if(|_| true, src, dst) }
    }

    /// Copies complete instruction groups selected by `pred(group_index)`.
    /// A false predicate is passed to [`CopyTraits::copy_predicated`], allowing
    /// a backend operation to zero-fill without reading the source.
    ///
    /// # Safety
    ///
    /// See [`Self::call`]; all values within an instruction share one predicate.
    pub unsafe fn call_if<SI, SL, DI, DL>(
        &self,
        pred: impl Fn(isize) -> bool,
        src: &View<SI, SL>,
        dst: &View<DI, DL>,
    ) where
        SI: Advance<SL::Output>,
        SL: MapCoord<isize> + LayoutShape<Shape: Product<Output: Integer>>,
        Elem<SI, SL>: RecastPtr<Op::SRegister>,
        Reg<SI, SL, Op::SRegister>: Read<Value = Op::SRegister>,
        DI: Advance<DL::Output>,
        DL: MapCoord<isize> + LayoutShape<Shape: Product<Output: Integer>>,
        Elem<DI, DL>: RecastPtr<Op::DRegister>,
        Reg<DI, DL, Op::DRegister>: Write<Value = Op::DRegister>,
    {
        let n = values::<Op::SRegister, SI::Value>(Op::SRC_REGISTERS);
        let m = values::<Op::DRegister, DI::Value>(Op::DST_REGISTERS);
        assert!(
            n > 0 && m > 0,
            "copy instructions must contain whole values"
        );
        let size = src.size();
        assert_eq!(size % n, 0, "the source does not split into instructions");
        assert_eq!(dst.size(), size / n * m, "copy operand sizes disagree");
        for i in 0..size / n {
            unsafe {
                Op::copy_predicated(
                    pred(i),
                    registers::<_, _, Op::SRegister>(src, i * n),
                    registers::<_, _, Op::DRegister>(dst, i * m),
                )
            }
        }
    }
}

type ValRef<Op, V> = <CopyAtom<Op, V> as AtomValLayouts>::Ref;
type ValSrc<Op, V> = <CopyAtom<Op, V> as AtomValLayouts>::Src;
type ValDst<Op, V> = <CopyAtom<Op, V> as AtomValLayouts>::Dst;
type AtomTile<Op, V> =
    IntTuple<ModeSize<ValRef<Op, V>, 0>, IntTuple<ModeSize<ValRef<Op, V>, 1>, Nil>>;
type Keep<M> = IntTuple<M, IntTuple<Underscore, Nil>>;
type AtomTv<TV, Op, V> = <TV as ZippedDivide<AtomTile<Op, V>>>::Output;
type TrgTv<TV, Op, V, Trg> =
    <AtomTv<TV, Op, V> as ComposeTile<Keep<RefToTrg<ValRef<Op, V>, Trg>>>>::Output;
type Profile = IntTuple<Int<1>, IntTuple<IntTuple<Int<1>, IntTuple<Int<1>, Nil>>, Nil>>;
type ThrVal2Mn<TV, Op, V, Trg> =
    <<TrgTv<TV, Op, V, Trg> as Zip2>::Output as CoalesceProfile<Profile>>::Output;
type TvTensor<L, Tiler, M> = <<L as ZippedDivide<Tiler>>::Output as ComposeTile<Keep<M>>>::Output;
type Unfold = IntTuple<IntTuple<Underscore, IntTuple<Underscore, Nil>>, IntTuple<Underscore, Nil>>;
/// `(thread, value, (rest...))` of a layout tiled by a copy: the result of `tidfrg_S` / `tidfrg_D`.
pub type Tidfrg<L, Tiler, M> = <TvTensor<L, Tiler, M> as SliceAt<Unfold>>::Sliced;

type Map<C, Trg> = <C as ThrValMap<Trg>>::Output;
type SrcMap<Op, V, TV, Tiler> = Map<TiledCopy<Op, V, TV, Tiler>, ValSrc<Op, V>>;
type DstMap<Op, V, TV, Tiler> = Map<TiledCopy<Op, V, TV, Tiler>, ValDst<Op, V>>;

/// The static steps of CuTe `TiledCopy::tile2thrfrg`, for the target layout `Trg`.
pub trait ThrValMap<Trg> {
    /// `((thread, rest thread), (value, rest value)) -> tile coordinate`, coalesced.
    type Output;
    /// Computes the map.
    fn thrval2mn(&self) -> Result<Self::Output>;
}

impl<Op: CopyTraits, V, TV, Tiler, Trg> ThrValMap<Trg> for TiledCopy<Op, V, TV, Tiler>
where
    CopyAtom<Op, V>: AtomValLayouts,
    ValRef<Op, V>: LayoutParts + TypedInvert,
    <ValRef<Op, V> as LayoutParts>::Shape: Get<0> + Get<1>,
    Mode<ValRef<Op, V>, 0>: Product,
    Mode<ValRef<Op, V>, 1>: Product,
    ModeSize<ValRef<Op, V>, 0>: StaticValue,
    ModeSize<ValRef<Op, V>, 1>: StaticValue,
    TypedInverse<ValRef<Op, V>>: TypedCompose<Trg>,
    RefToTrg<ValRef<Op, V>, Trg>: StaticValue,
    Trg: StaticValue,
    TV: ZippedDivide<AtomTile<Op, V>>,
    AtomTv<TV, Op, V>: ComposeTile<Keep<RefToTrg<ValRef<Op, V>, Trg>>>,
    TrgTv<TV, Op, V, Trg>: Zip2,
    <TrgTv<TV, Op, V, Trg> as Zip2>::Output: CoalesceProfile<Profile>,
{
    type Output = ThrVal2Mn<TV, Op, V, Trg>;
    fn thrval2mn(&self) -> Result<Self::Output> {
        let atom_tv = self.layout_tv.zipped_divide(AtomTile::<Op, V>::VALUE)?;
        let ref2trg = <RefToTrg<ValRef<Op, V>, Trg> as StaticValue>::VALUE;
        atom_tv
            .compose_tile(IntTuple(ref2trg, IntTuple(Underscore, Nil)))?
            .zip2()
            .coalesce_profile(Profile::VALUE)
    }
}

impl<Op: CopyTraits, V, TV, Tiler: Clone> TiledCopy<Op, V, TV, Tiler> {
    /// CuTe `TiledCopy::tile2thrfrg`: tiles `layout` and maps each tile to
    /// `(thread, value, (rest...))` through `trg`'s side of the atom.
    fn tile2thrfrg<L, Trg>(&self, layout: &L) -> Result<Tidfrg<L, Tiler, Map<Self, Trg>>>
    where
        Self: ThrValMap<Trg>,
        L: ZippedDivide<Tiler>,
        L::Output: ComposeTile<Keep<Map<Self, Trg>>>,
        TvTensor<L, Tiler, Map<Self, Trg>>: SliceAt<Unfold>,
    {
        let tiled = layout.zipped_divide(self.tiler.clone())?;
        let tv = tiled.compose_tile(IntTuple(self.thrval2mn()?, IntTuple(Underscore, Nil)))?;
        let unfold = IntTuple(
            IntTuple(Underscore, IntTuple(Underscore, Nil)),
            IntTuple(Underscore, Nil),
        );
        Ok(tv.slice_and_offset(unfold).0)
    }

    /// CuTe `TiledCopy::tidfrg_S`: `(thread, value, (rest...))` of a source layout.
    pub fn tidfrg_s<L>(&self, layout: &L) -> Result<SrcFrg<L, Op, V, TV, Tiler>>
    where
        CopyAtom<Op, V>: AtomValLayouts,
        Self: ThrValMap<ValSrc<Op, V>>,
        L: ZippedDivide<Tiler>,
        L::Output: ComposeTile<Keep<SrcMap<Op, V, TV, Tiler>>>,
        TvTensor<L, Tiler, SrcMap<Op, V, TV, Tiler>>: SliceAt<Unfold>,
    {
        self.tile2thrfrg::<L, ValSrc<Op, V>>(layout)
    }

    /// CuTe `TiledCopy::tidfrg_D`: `(thread, value, (rest...))` of a destination layout.
    pub fn tidfrg_d<L>(&self, layout: &L) -> Result<DstFrg<L, Op, V, TV, Tiler>>
    where
        CopyAtom<Op, V>: AtomValLayouts,
        Self: ThrValMap<ValDst<Op, V>>,
        L: ZippedDivide<Tiler>,
        L::Output: ComposeTile<Keep<DstMap<Op, V, TV, Tiler>>>,
        TvTensor<L, Tiler, DstMap<Op, V, TV, Tiler>>: SliceAt<Unfold>,
    {
        self.tile2thrfrg::<L, ValDst<Op, V>>(layout)
    }

    /// The partitioner of thread `thr_idx`: CuTe `TiledCopy::get_slice(thr_idx)`.
    pub fn get_slice<X: Integer>(&self, thr_idx: X) -> ThrCopy<'_, Op, V, TV, Tiler, X> {
        ThrCopy {
            tiled: self,
            thr_idx,
        }
    }
}

/// One thread's view of a tiled copy: CuTe `ThrCopy`.
#[derive(Clone, Copy, Debug)]
pub struct ThrCopy<'a, Op, V, TV, Tiler, X> {
    tiled: &'a TiledCopy<Op, V, TV, Tiler>,
    thr_idx: X,
}

type RestModes<F> =
    <<<F as LayoutShape>::Shape as Get<2>>::Output as RepeatModes<Underscore>>::Output;
type ThrCoord<X, F> = IntTuple<X, IntTuple<Underscore, IntTuple<RestModes<F>, Nil>>>;
/// One thread's `(value, rest...)` of a `(thread, value, (rest...))` tensor.
pub type ThrPartition<I, F, X> = Tensor<
    ViewEngine<<I as Advance<<F as SliceAt<ThrCoord<X, F>>>::Offset>>::Output>,
    <F as SliceAt<ThrCoord<X, F>>>::Sliced,
>;
type SrcFrg<L, Op, V, TV, Tiler> = Tidfrg<L, Tiler, SrcMap<Op, V, TV, Tiler>>;
type DstFrg<L, Op, V, TV, Tiler> = Tidfrg<L, Tiler, DstMap<Op, V, TV, Tiler>>;
type SrcPartition<I, L, Op, V, TV, Tiler, X> = ThrPartition<I, SrcFrg<L, Op, V, TV, Tiler>, X>;
type DstPartition<I, L, Op, V, TV, Tiler, X> = ThrPartition<I, DstFrg<L, Op, V, TV, Tiler>, X>;

fn thr_coord<X, F>(thr_idx: X) -> ThrCoord<X, F>
where
    F: LayoutShape,
    F::Shape: Get<2>,
    <F::Shape as Get<2>>::Output: RepeatModes<Underscore>,
{
    IntTuple(
        thr_idx,
        IntTuple(
            Underscore,
            IntTuple(
                <<F::Shape as Get<2>>::Output as RepeatModes<Underscore>>::repeat_modes(Underscore),
                Nil,
            ),
        ),
    )
}

impl<Op: CopyTraits, V, TV, Tiler: Clone, X: Integer> ThrCopy<'_, Op, V, TV, Tiler, X> {
    /// CuTe `ThrCopy::partition_S`: this thread's `(value, rest...)` of a source tensor.
    ///
    /// The tensor's leading modes must match the tiler; trailing modes are kept
    /// in the rest modes.
    pub fn partition_s<I, L>(
        &self,
        tensor: Tensor<ViewEngine<I>, L>,
    ) -> Result<SrcPartition<I, L, Op, V, TV, Tiler, X>>
    where
        CopyAtom<Op, V>: AtomValLayouts,
        TiledCopy<Op, V, TV, Tiler>: ThrValMap<ValSrc<Op, V>>,
        L: ZippedDivide<Tiler>,
        L::Output: ComposeTile<Keep<SrcMap<Op, V, TV, Tiler>>>,
        TvTensor<L, Tiler, SrcMap<Op, V, TV, Tiler>>: SliceAt<Unfold>,
        SrcFrg<L, Op, V, TV, Tiler>:
            LayoutShape + SliceAt<ThrCoord<X, SrcFrg<L, Op, V, TV, Tiler>>>,
        <SrcFrg<L, Op, V, TV, Tiler> as LayoutShape>::Shape: Get<2>,
        <<SrcFrg<L, Op, V, TV, Tiler> as LayoutShape>::Shape as Get<2>>::Output:
            RepeatModes<Underscore>,
        I:
            Advance<
                <SrcFrg<L, Op, V, TV, Tiler> as SliceAt<
                    ThrCoord<X, SrcFrg<L, Op, V, TV, Tiler>>,
                >>::Offset,
            >,
    {
        let frg = self.tiled.tidfrg_s(&tensor.layout)?;
        Ok(make_tensor(tensor.data(), frg)
            .slice(thr_coord::<X, SrcFrg<L, Op, V, TV, Tiler>>(self.thr_idx)))
    }

    /// CuTe `ThrCopy::partition_D`: this thread's `(value, rest...)` of a destination tensor.
    pub fn partition_d<I, L>(
        &self,
        tensor: Tensor<ViewEngine<I>, L>,
    ) -> Result<DstPartition<I, L, Op, V, TV, Tiler, X>>
    where
        CopyAtom<Op, V>: AtomValLayouts,
        TiledCopy<Op, V, TV, Tiler>: ThrValMap<ValDst<Op, V>>,
        L: ZippedDivide<Tiler>,
        L::Output: ComposeTile<Keep<DstMap<Op, V, TV, Tiler>>>,
        TvTensor<L, Tiler, DstMap<Op, V, TV, Tiler>>: SliceAt<Unfold>,
        DstFrg<L, Op, V, TV, Tiler>:
            LayoutShape + SliceAt<ThrCoord<X, DstFrg<L, Op, V, TV, Tiler>>>,
        <DstFrg<L, Op, V, TV, Tiler> as LayoutShape>::Shape: Get<2>,
        <<DstFrg<L, Op, V, TV, Tiler> as LayoutShape>::Shape as Get<2>>::Output:
            RepeatModes<Underscore>,
        I:
            Advance<
                <DstFrg<L, Op, V, TV, Tiler> as SliceAt<
                    ThrCoord<X, DstFrg<L, Op, V, TV, Tiler>>,
                >>::Offset,
            >,
    {
        let frg = self.tiled.tidfrg_d(&tensor.layout)?;
        Ok(make_tensor(tensor.data(), frg)
            .slice(thr_coord::<X, DstFrg<L, Op, V, TV, Tiler>>(self.thr_idx)))
    }
}

/// One MMA instruction: CuTe `MMA_Traits<MMAOperation>` with the operation's
/// register arrays and `fma`.
///
/// `ALayout`, `BLayout`, and `CLayout` map `(thread, value)` to `(m, k)`,
/// `(n, k)`, and `(m, n)` within the static `ShapeMnk`. The fragment value
/// types are the `Val*` types; upstream's optional `FrgType*` are not modeled.
pub trait MmaTraits {
    /// The value type of A.
    type ValA;
    /// The value type of B.
    type ValB;
    /// The value type of C.
    type ValC;
    /// The static `(M, N, K)` of one instruction.
    type ShapeMnk;
    /// Logical thread id to thread index.
    type ThrId;
    /// `(thread, value) -> (m, k)`.
    type ALayout;
    /// `(thread, value) -> (n, k)`.
    type BLayout;
    /// `(thread, value) -> (m, n)`.
    type CLayout;
    /// The type of one D register.
    type DRegister;
    /// The type of one A register.
    type ARegister;
    /// The type of one B register.
    type BRegister;
    /// The type of one C register.
    type CRegister;
    /// The number of D registers.
    const D_REGISTERS: isize;
    /// The number of A registers.
    const A_REGISTERS: isize;
    /// The number of B registers.
    const B_REGISTERS: isize;
    /// The number of C registers.
    const C_REGISTERS: isize;
    /// Computes `D = A * B + C` on the registers.
    ///
    /// # Safety
    ///
    /// Every register address must be valid for its access, and no other
    /// thread may access them concurrently. D may alias C.
    unsafe fn fma<DP, AP, BP, CP>(
        d: impl Fn(isize) -> DP,
        a: impl Fn(isize) -> AP,
        b: impl Fn(isize) -> BP,
        c: impl Fn(isize) -> CP,
    ) where
        DP: Write<Value = Self::DRegister>,
        AP: Read<Value = Self::ARegister>,
        BP: Read<Value = Self::BRegister>,
        CP: Read<Value = Self::CRegister>;
}

/// A scalar fused multiply-add on one thread: CuTe `UniversalFMA<D, A, B, C>`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UniversalFma<D, A = D, B = A, C = D>(PhantomData<(D, A, B, C)>);

type Unit2 =
    Layout<IntTuple<Int<1>, IntTuple<Int<1>, Nil>>, IntTuple<Int<0>, IntTuple<Int<0>, Nil>>>;

impl<D, A, B, C> MmaTraits for UniversalFma<D, A, B, C>
where
    A: Mul<B>,
    A::Output: Add<C, Output = D>,
{
    type ValA = A;
    type ValB = B;
    type ValC = C;
    type ShapeMnk = IntTuple<Int<1>, IntTuple<Int<1>, IntTuple<Int<1>, Nil>>>;
    type ThrId = OneThread;
    type ALayout = Unit2;
    type BLayout = Unit2;
    type CLayout = Unit2;
    type DRegister = D;
    type ARegister = A;
    type BRegister = B;
    type CRegister = C;
    const D_REGISTERS: isize = 1;
    const A_REGISTERS: isize = 1;
    const B_REGISTERS: isize = 1;
    const C_REGISTERS: isize = 1;
    unsafe fn fma<DP, AP, BP, CP>(
        d: impl Fn(isize) -> DP,
        a: impl Fn(isize) -> AP,
        b: impl Fn(isize) -> BP,
        c: impl Fn(isize) -> CP,
    ) where
        DP: Write<Value = D>,
        AP: Read<Value = A>,
        BP: Read<Value = B>,
        CP: Read<Value = C>,
    {
        // SAFETY: forwarded to the caller.
        unsafe { d(0).write(a(0).read() * b(0).read() + c(0).read()) }
    }
}

/// An MMA instruction: CuTe `MMA_Atom<MMAOperation>`.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct MmaAtom<Op>(PhantomData<Op>);

impl<Op> Clone for MmaAtom<Op> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<Op> Copy for MmaAtom<Op> {}

impl<Op> MmaAtom<Op> {
    /// The atom of `Op`.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

type View<I, L> = Tensor<ViewEngine<I>, L>;
type Fragment<L> = <L as FragmentLike>::Output;
type Owned<T, L> = Tensor<<L as OwnedStorage<T>>::Engine, L>;
type Elem<I, L> = <I as Advance<<L as MapCoord<isize>>::Output>>::Output;
type Reg<I, L, R> = <Elem<I, L> as RecastPtr<R>>::Output;

/// The register pointers of a fragment: register `r` starts at element
/// `r * size_of::<R>() / size_of::<value>()`, and its elements must be contiguous.
fn registers<I, L, R>(t: &View<I, L>, base: isize) -> impl Fn(isize) -> Reg<I, L, R> + '_
where
    I: Advance<L::Output>,
    L: MapCoord<isize>,
    Elem<I, L>: RecastPtr<R>,
{
    let per = const {
        let (r, v) = (size_of::<R>(), size_of::<I::Value>());
        assert!(v > 0 && r % v == 0, "a register must hold whole values");
        (r / v) as isize
    };
    move |r| {
        t.data()
            .advance(t.layout.map_coord(base + r * per))
            .recast_ptr()
    }
}

/// The number of values held by `n` registers of type `R`.
const fn values<R, V>(n: isize) -> isize {
    n * (size_of::<R>() / size_of::<V>()) as isize
}

impl<Op: MmaTraits> MmaAtom<Op> {
    /// CuTe `MMA_Atom::call(D, A, B, C)`: one instruction on register fragments.
    ///
    /// The fragments are read in colexicographic order and recast to the
    /// operation's registers, as upstream `mma_unpack` does. Unlike upstream,
    /// rank-1 fragments are not required.
    ///
    /// # Safety
    ///
    /// Each fragment must hold the values of the operation's registers, valid
    /// for their access and not accessed by another thread concurrently; the
    /// values of one register must be contiguous and aligned for the register
    /// type. D may alias C.
    pub unsafe fn call<DI, DL, AI, AL, BI, BL, CI, CL>(
        &self,
        d: &View<DI, DL>,
        a: &View<AI, AL>,
        b: &View<BI, BL>,
        c: &View<CI, CL>,
    ) where
        DI: Advance<DL::Output>,
        DL: MapCoord<isize>,
        Elem<DI, DL>: RecastPtr<Op::DRegister>,
        Reg<DI, DL, Op::DRegister>: Write<Value = Op::DRegister>,
        AI: Advance<AL::Output>,
        AL: MapCoord<isize>,
        Elem<AI, AL>: RecastPtr<Op::ARegister>,
        Reg<AI, AL, Op::ARegister>: Read<Value = Op::ARegister>,
        BI: Advance<BL::Output>,
        BL: MapCoord<isize>,
        Elem<BI, BL>: RecastPtr<Op::BRegister>,
        Reg<BI, BL, Op::BRegister>: Read<Value = Op::BRegister>,
        CI: Advance<CL::Output>,
        CL: MapCoord<isize>,
        Elem<CI, CL>: RecastPtr<Op::CRegister>,
        Reg<CI, CL, Op::CRegister>: Read<Value = Op::CRegister>,
    {
        // SAFETY: forwarded to the caller.
        unsafe {
            Op::fma(
                registers::<_, _, Op::DRegister>(d, 0),
                registers::<_, _, Op::ARegister>(a, 0),
                registers::<_, _, Op::BRegister>(b, 0),
                registers::<_, _, Op::CRegister>(c, 0),
            )
        }
    }
}

type Mnk<P0, P1, P2> = IntTuple<P0, IntTuple<P1, IntTuple<P2, Nil>>>;

/// MMA atoms tiled over threads: CuTe `TiledMMA<MMA_Atom, AtomLayoutMNK, PermutationMNK>`.
///
/// `thr_layout_vmnk` is `tiled_product(ThrID, AtomLayoutMNK)`, mapping
/// `(atom thread, m, n, k)` to a thread index. `permutation_mnk` holds one
/// tiler per M, N, K mode; upstream's `_` entries are already resolved to
/// `size(atom shape) * size(thread mode)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TiledMma<Op, Vmnk, Perm> {
    /// The atom.
    pub atom: MmaAtom<Op>,
    /// `(V, M, N, K) -> thread index`.
    pub thr_layout_vmnk: Vmnk,
    /// The M, N, K tilers applied before tiling by the atom.
    pub permutation_mnk: Perm,
}

type Three = Mnk<Int<1>, Int<1>, Int<1>>;
type PadThr<TS, TD> = Layout<
    <Layout<Wrapped<TS>, Wrapped<TD>> as PadTo<Three>>::Shape,
    <Layout<Wrapped<TS>, Wrapped<TD>> as PadTo<Three>>::Stride,
>;
type Wrapped<T> = <T as Wrap>::Output;
type Vmnk<Op, TS, TD> =
    TypedProductLayout<<Op as MmaTraits>::ThrId, PadThr<TS, TD>, { ProductKind::Tiled }>;
type VmnkShape<V> = <V as LayoutParts>::Shape;
type ThrSize<V, const I: usize> = <<VmnkShape<V> as Get<I>>::Output as Product>::Output;
type AtomSize<Op, const I: usize> =
    <<<Op as MmaTraits>::ShapeMnk as Get<I>>::Output as Product>::Output;
type Padded<P> = <P as PadModes<Three>>::Output;

/// Replaces a `_` permutation mode by its default tile size `A * T`.
pub trait PermMode<A, T> {
    /// The resolved tiler.
    type Output;
    /// Resolves the mode.
    fn resolve(self) -> Self::Output;
}

impl<A: Arith<T, { ArithOp::Mul }> + StaticValue, T: StaticValue> PermMode<A, T> for Underscore {
    type Output = Prod<A, T>;
    fn resolve(self) -> Self::Output {
        prod(A::VALUE, T::VALUE)
    }
}

impl<A, T, const N: isize> PermMode<A, T> for Int<N> {
    type Output = Self;
    fn resolve(self) -> Self {
        self
    }
}

impl<A, T, S, D> PermMode<A, T> for Layout<S, D> {
    type Output = Self;
    fn resolve(self) -> Self {
        self
    }
}

type PermOut<Op, V, P, const I: usize, const J: usize> =
    <<Padded<P> as Get<I>>::Output as PermMode<AtomSize<Op, I>, ThrSize<V, J>>>::Output;
type MadeMma<Op, TS, TD, P> = TiledMma<Op, Vmnk<Op, TS, TD>, Perms<Op, Vmnk<Op, TS, TD>, P>>;
type Perms<Op, V, P> =
    Mnk<PermOut<Op, V, P, 0, 1>, PermOut<Op, V, P, 1, 2>, PermOut<Op, V, P, 2, 3>>;

/// CuTe `make_tiled_mma(atom, thr_layout, permutations)`.
///
/// `thr_layout` arranges atoms over M, N, and K and is padded with `1:0` modes
/// to rank 3; `permutations` is padded with `_` to rank 3, and each `_`
/// becomes the atom's extent times the number of atoms in that mode. As
/// upstream, the thread layout and the permutations must be static.
///
/// ```
/// use noodle_cute::atom::{MmaAtom, UniversalFma, make_tiled_mma};
/// use noodle_cute::coord::X;
/// use noodle_cute::layout::LayoutLeft;
/// use noodle_cute::{Int, IntTuple, Nil, shape};
///
/// // 16x16 threads, each computing one FMA per (m, n).
/// let atom = MmaAtom::<UniversalFma<f32>>::new();
/// let mma = make_tiled_mma(atom, LayoutLeft::make(shape![Int::<16>, Int::<16>])?, IntTuple(X, Nil))?;
/// assert_eq!(mma.thr_layout_vmnk.shape, shape![Int::<1>, Int::<16>, Int::<16>, Int::<1>]);
/// assert_eq!(mma.permutation_mnk, shape![Int::<16>, Int::<16>, Int::<1>]);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
pub fn make_tiled_mma<Op, TS, TD, P>(
    atom: MmaAtom<Op>,
    thr_layout: Layout<TS, TD>,
    permutations: P,
) -> Result<MadeMma<Op, TS, TD, P>>
where
    Op: MmaTraits,
    Op::ThrId: StaticValue + TypedProduct<PadThr<TS, TD>, { ProductKind::Tiled }>,
    Op::ShapeMnk: Get<0> + Get<1> + Get<2>,
    <Op::ShapeMnk as Get<0>>::Output: Product,
    <Op::ShapeMnk as Get<1>>::Output: Product,
    <Op::ShapeMnk as Get<2>>::Output: Product,
    TS: Wrap,
    TD: Wrap,
    Layout<Wrapped<TS>, Wrapped<TD>>: PadTo<Three>,
    Vmnk<Op, TS, TD>: LayoutParts,
    VmnkShape<Vmnk<Op, TS, TD>>: Get<1> + Get<2> + Get<3>,
    <VmnkShape<Vmnk<Op, TS, TD>> as Get<1>>::Output: Product,
    <VmnkShape<Vmnk<Op, TS, TD>> as Get<2>>::Output: Product,
    <VmnkShape<Vmnk<Op, TS, TD>> as Get<3>>::Output: Product,
    P: PadModes<Three>,
    Padded<P>: Get<0> + Get<1> + Get<2>,
    <Padded<P> as Get<0>>::Output: Clone + PermMode<AtomSize<Op, 0>, ThrSize<Vmnk<Op, TS, TD>, 1>>,
    <Padded<P> as Get<1>>::Output: Clone + PermMode<AtomSize<Op, 1>, ThrSize<Vmnk<Op, TS, TD>, 2>>,
    <Padded<P> as Get<2>>::Output: Clone + PermMode<AtomSize<Op, 2>, ThrSize<Vmnk<Op, TS, TD>, 3>>,
{
    let thr_mnk = PadTo::<Three>::pad(Layout {
        shape: thr_layout.shape.wrap(),
        stride: thr_layout.stride.wrap(),
    });
    let thr_layout_vmnk = Op::ThrId::VALUE.product_typed(thr_mnk)?;
    let perms = permutations.pad_modes();
    let p0 = Get::<0>::get(&perms).clone().resolve();
    let p1 = Get::<1>::get(&perms).clone().resolve();
    let p2 = Get::<2>::get(&perms).clone().resolve();
    Ok(TiledMma {
        atom,
        thr_layout_vmnk,
        permutation_mnk: IntTuple(p0, IntTuple(p1, IntTuple(p2, Nil))),
    })
}

// Marker types rather than an enum const parameter: rustdoc (nightly
// 2026-11-01) crashes on public impls keyed by enum constants.
/// Selects the `(M, K)` operand A of an MMA.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OperandA;
/// Selects the `(N, K)` operand B of an MMA.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OperandB;
/// Selects the `(M, N)` accumulator C of an MMA.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OperandC;

type Pair<A, B> = IntTuple<A, IntTuple<B, Nil>>;

/// The tilers CuTe `TiledMMA::thrfrg_A`, `thrfrg_B`, and `thrfrg_C` use for one operand.
pub trait MmaOperand<W> {
    /// The two permutation tilers.
    type Perm;
    /// The atom's extents in the operand's two modes.
    type AtomTile;
    /// The atom's `(thread, value) -> coordinate` layout.
    type Tv;
    /// The atom counts in the operand's two modes.
    type ThrTile;
    /// Returns the four parts.
    fn operand(&self) -> (Self::Perm, Self::AtomTile, Self::Tv, Self::ThrTile);
}

macro_rules! mma_operand {
    ($w:ident: perms $p0:literal $p1:literal, atom $m0:literal $m1:literal, $tv:ident, threads $t0:literal $t1:literal) => {
        impl<Op, V, P> MmaOperand<$w> for TiledMma<Op, V, P>
        where
            Op: MmaTraits,
            Op::ShapeMnk: Get<$m0> + Get<$m1>,
            <Op::ShapeMnk as Get<$m0>>::Output: Product,
            <Op::ShapeMnk as Get<$m1>>::Output: Product,
            AtomSize<Op, $m0>: StaticValue,
            AtomSize<Op, $m1>: StaticValue,
            Op::$tv: StaticValue,
            V: LayoutParts,
            VmnkShape<V>: Get<$t0> + Get<$t1>,
            <VmnkShape<V> as Get<$t0>>::Output: Product,
            <VmnkShape<V> as Get<$t1>>::Output: Product,
            ThrSize<V, $t0>: StaticValue,
            ThrSize<V, $t1>: StaticValue,
            P: Get<$p0> + Get<$p1>,
            <P as Get<$p0>>::Output: Clone,
            <P as Get<$p1>>::Output: Clone,
        {
            type Perm = Pair<<P as Get<$p0>>::Output, <P as Get<$p1>>::Output>;
            type AtomTile = Pair<AtomSize<Op, $m0>, AtomSize<Op, $m1>>;
            type Tv = Op::$tv;
            type ThrTile = Pair<ThrSize<V, $t0>, ThrSize<V, $t1>>;
            fn operand(&self) -> (Self::Perm, Self::AtomTile, Self::Tv, Self::ThrTile) {
                let p = &self.permutation_mnk;
                let perm = IntTuple(
                    Get::<$p0>::get(p).clone(),
                    IntTuple(Get::<$p1>::get(p).clone(), Nil),
                );
                (
                    perm,
                    StaticValue::VALUE,
                    StaticValue::VALUE,
                    StaticValue::VALUE,
                )
            }
        }
    };
}

mma_operand!(OperandA: perms 0 2, atom 0 2, ALayout, threads 1 3);
mma_operand!(OperandB: perms 1 2, atom 1 2, BLayout, threads 2 3);
mma_operand!(OperandC: perms 0 1, atom 0 1, CLayout, threads 1 2);

type ThrTiler<TT> = IntTuple<Underscore, IntTuple<TT, Nil>>;
type Permuted<L, M, W> = <L as LogicalDivide<<M as MmaOperand<W>>::Perm>>::Output;
type AtomTiled<L, M, W> =
    <Permuted<L, M, W> as ZippedDivide<<M as MmaOperand<W>>::AtomTile>>::Output;
type OperandTv<L, M, W> =
    <AtomTiled<L, M, W> as ComposeTile<Keep<<M as MmaOperand<W>>::Tv>>>::Output;
/// `((ThrV, (Thr0, Thr1)), (FrgV, (Rest0, Rest1, ...)))`: a layout tiled for operand `W`.
pub type ThrFrg<L, M, W> =
    <OperandTv<L, M, W> as ZippedDivide<ThrTiler<<M as MmaOperand<W>>::ThrTile>>>::Output;

/// CuTe `TiledMMA::thrfrg_A`, `thrfrg_B`, and `thrfrg_C` for operand `W`.
///
/// The layout is divided by the operand's permutations, zipped-divided by
/// the atom's extents, composed with the atom's thread–value layout, and
/// zipped-divided by the atom counts.
pub trait ThrFrgOf<L, W> {
    /// The thread–fragment layout.
    type Output;
    /// Tiles `layout`.
    fn thrfrg(&self, layout: &L) -> Result<Self::Output>;
}

impl<Op, V, P, L, W> ThrFrgOf<L, W> for TiledMma<Op, V, P>
where
    Self: MmaOperand<W>,
    L: LogicalDivide<<Self as MmaOperand<W>>::Perm>,
    Permuted<L, Self, W>: ZippedDivide<<Self as MmaOperand<W>>::AtomTile>,
    AtomTiled<L, Self, W>: ComposeTile<Keep<<Self as MmaOperand<W>>::Tv>>,
    OperandTv<L, Self, W>: ZippedDivide<ThrTiler<<Self as MmaOperand<W>>::ThrTile>>,
{
    type Output = ThrFrg<L, Self, W>;
    fn thrfrg(&self, layout: &L) -> Result<Self::Output> {
        let (perm, atom_tile, tv, thr_tile) = self.operand();
        layout
            .logical_divide_by(perm)?
            .zipped_divide(atom_tile)?
            .compose_tile(IntTuple(tv, IntTuple(Underscore, Nil)))?
            .zipped_divide(IntTuple(Underscore, IntTuple(thr_tile, Nil)))
    }
}

type FlatVmnk<V, X> =
    <<X as Idx2Crd<VmnkShape<V>, <V as LayoutParts>::Stride>>::Output as Crd2Crd<
        VmnkShape<V>,
        <VmnkShape<V> as RepeatModes<Int<1>>>::Output,
    >>::Output;

impl<Op, V: LayoutParts + Clone, P> TiledMma<Op, V, P> {
    /// The partitioner of thread `thr_idx`: CuTe `TiledMMA::get_slice(thr_idx)`,
    /// which keeps `thr_layout_vmnk.get_flat_coord(thr_idx)`.
    pub fn get_slice<X>(&self, thr_idx: X) -> ThrMma<'_, Op, V, P, FlatVmnk<V, X>>
    where
        X: Integer + Idx2Crd<VmnkShape<V>, V::Stride>,
        VmnkShape<V>: RepeatModes<Int<1>>,
        X::Output: Crd2Crd<VmnkShape<V>, <VmnkShape<V> as RepeatModes<Int<1>>>::Output>,
    {
        let vmnk = self.thr_layout_vmnk.clone().into_layout();
        ThrMma {
            tiled: self,
            thr_vmnk: vmnk.get_flat_coord(thr_idx),
        }
    }
}

/// One thread's view of a tiled MMA: CuTe `ThrMMA`.
#[derive(Clone, Copy, Debug)]
pub struct ThrMma<'a, Op, V, P, F> {
    tiled: &'a TiledMma<Op, V, P>,
    /// The thread's flat `(v, m, n, k)` coordinate in `thr_layout_vmnk`.
    pub thr_vmnk: F,
}

/// The thread coordinate of one operand: `(v, (m, k))`, `(v, (n, k))`, or `(v, (m, n))`.
pub trait OperandCoord<W> {
    /// The coordinate.
    type Output;
    /// Selects the coordinate.
    fn operand_coord(&self) -> Self::Output;
}

type Vmnk4<V, M, N, K> = IntTuple<V, Mnk<M, N, K>>;

impl<V: Copy, M: Copy, N, K: Copy> OperandCoord<OperandA> for Vmnk4<V, M, N, K> {
    type Output = Pair<V, Pair<M, K>>;
    fn operand_coord(&self) -> Self::Output {
        let IntTuple(v, IntTuple(m, IntTuple(_, IntTuple(k, Nil)))) = self;
        IntTuple(*v, IntTuple(IntTuple(*m, IntTuple(*k, Nil)), Nil))
    }
}

impl<V: Copy, M, N: Copy, K: Copy> OperandCoord<OperandB> for Vmnk4<V, M, N, K> {
    type Output = Pair<V, Pair<N, K>>;
    fn operand_coord(&self) -> Self::Output {
        let IntTuple(v, IntTuple(_, IntTuple(n, IntTuple(k, Nil)))) = self;
        IntTuple(*v, IntTuple(IntTuple(*n, IntTuple(*k, Nil)), Nil))
    }
}

impl<V: Copy, M: Copy, N: Copy, K> OperandCoord<OperandC> for Vmnk4<V, M, N, K> {
    type Output = Pair<V, Pair<M, N>>;
    fn operand_coord(&self) -> Self::Output {
        let IntTuple(v, IntTuple(m, IntTuple(n, _))) = self;
        IntTuple(*v, IntTuple(IntTuple(*m, IntTuple(*n, Nil)), Nil))
    }
}

type RestOf<F> =
    <<<<F as LayoutShape>::Shape as Get<1>>::Output as Get<1>>::Output as RepeatModes<
        Underscore,
    >>::Output;
type OperandSlice<F, C> = Pair<C, Pair<Underscore, RestOf<F>>>;
type OperandFrg<L, Op, V, P, W> = <TiledMma<Op, V, P> as ThrFrgOf<L, W>>::Output;
type OperandAt<Th, W> = <Th as OperandCoord<W>>::Output;
/// One thread's `(FrgV, Rest0, Rest1, ...)` partition of a tensor for operand `W`.
pub type MmaPartition<I, L, Op, V, P, Th, W> = Tensor<
    ViewEngine<
        <I as Advance<
            <OperandFrg<L, Op, V, P, W> as SliceAt<
                OperandSlice<OperandFrg<L, Op, V, P, W>, OperandAt<Th, W>>,
            >>::Offset,
        >>::Output,
    >,
    <OperandFrg<L, Op, V, P, W> as SliceAt<
        OperandSlice<OperandFrg<L, Op, V, P, W>, OperandAt<Th, W>>,
    >>::Sliced,
>;

/// CuTe `ThrMMA::partition_A`, `partition_B`, and `partition_C` for operand `W`.
pub trait PartitionOperand<T, W> {
    /// The partition.
    type Output;
    /// Partitions `tensor`.
    fn partition_operand(&self, tensor: T) -> Result<Self::Output>;
}

impl<Op, V, P, Th, I, L, W> PartitionOperand<View<I, L>, W> for ThrMma<'_, Op, V, P, Th>
where
    I: Iter,
    Th: OperandCoord<W>,
    TiledMma<Op, V, P>: ThrFrgOf<L, W>,
    OperandFrg<L, Op, V, P, W>:
        LayoutShape + SliceAt<OperandSlice<OperandFrg<L, Op, V, P, W>, OperandAt<Th, W>>>,
    <OperandFrg<L, Op, V, P, W> as LayoutShape>::Shape: Get<1>,
    <<OperandFrg<L, Op, V, P, W> as LayoutShape>::Shape as Get<1>>::Output: Get<1>,
    <<<OperandFrg<L, Op, V, P, W> as LayoutShape>::Shape as Get<1>>::Output as Get<1>>::Output:
        RepeatModes<Underscore>,
    I: Advance<
        <OperandFrg<L, Op, V, P, W> as SliceAt<
            OperandSlice<OperandFrg<L, Op, V, P, W>, OperandAt<Th, W>>,
        >>::Offset,
    >,
{
    type Output = MmaPartition<I, L, Op, V, P, Th, W>;
    fn partition_operand(&self, tensor: View<I, L>) -> Result<Self::Output> {
        let frg = self.tiled.thrfrg(&tensor.layout)?;
        let rest = <<<<OperandFrg<L, Op, V, P, W> as LayoutShape>::Shape as Get<1>>::Output as Get<1>>::Output as RepeatModes<Underscore>>::repeat_modes(Underscore);
        let coord = IntTuple(
            self.thr_vmnk.operand_coord(),
            IntTuple(IntTuple(Underscore, IntTuple(rest, Nil)), Nil),
        );
        Ok(make_tensor(tensor.data(), frg).slice(coord))
    }
}

impl<Op: MmaTraits, V, P, Th> ThrMma<'_, Op, V, P, Th> {
    /// CuTe `ThrMMA::partition_A`: this thread's `(FrgV, RestM, RestK, ...)` of an `(M, K, ...)` tensor.
    pub fn partition_a<T>(
        &self,
        tensor: T,
    ) -> Result<<Self as PartitionOperand<T, OperandA>>::Output>
    where
        Self: PartitionOperand<T, OperandA>,
    {
        self.partition_operand(tensor)
    }

    /// CuTe `ThrMMA::partition_B`: this thread's `(FrgV, RestN, RestK, ...)` of an `(N, K, ...)` tensor.
    pub fn partition_b<T>(
        &self,
        tensor: T,
    ) -> Result<<Self as PartitionOperand<T, OperandB>>::Output>
    where
        Self: PartitionOperand<T, OperandB>,
    {
        self.partition_operand(tensor)
    }

    /// CuTe `ThrMMA::partition_C`: this thread's `(FrgV, RestM, RestN, ...)` of an `(M, N, ...)` tensor.
    ///
    /// ```
    /// use noodle_cute::atom::{MmaAtom, UniversalFma, make_tiled_mma};
    /// use noodle_cute::coord::X;
    /// use noodle_cute::layout::LayoutLeft;
    /// use noodle_cute::tensor::make_tensor;
    /// use noodle_cute::{Int, IntTuple, Nil, shape};
    ///
    /// let mma = make_tiled_mma(
    ///     MmaAtom::<UniversalFma<f32>>::new(),
    ///     LayoutLeft::make(shape![Int::<4>, Int::<2>])?,
    ///     IntTuple(X, Nil),
    /// )?;
    /// let c = [0f32; 8 * 4];
    /// let tile = make_tensor(c.as_ptr(), LayoutLeft::make(shape![Int::<8>, Int::<4>])?);
    /// // Thread 5 is (m, n) = (1, 1): rows 1 and 5 of columns 1 and 3.
    /// let mine = mma.get_slice(5i32).partition_c(tile)?;
    /// assert_eq!(mine.layout.shape, shape![Int::<1>, Int::<2>, Int::<2>]);
    /// assert_eq!(unsafe { mine.data().offset_from(c.as_ptr()) }, 1 + 8);
    /// # Ok::<(), noodle_cute::Error>(())
    /// ```
    pub fn partition_c<T>(
        &self,
        tensor: T,
    ) -> Result<<Self as PartitionOperand<T, OperandC>>::Output>
    where
        Self: PartitionOperand<T, OperandC>,
    {
        self.partition_operand(tensor)
    }

    /// CuTe `ThrMMA::partition_fragment_A`: a register fragment shaped like
    /// [`ThrMma::partition_a`], from `make_fragment_like<ValTypeA>`; a swizzled
    /// shared-memory partition gives the fragment of its `layout_b`.
    pub fn partition_fragment_a<T, E, L>(&self, tensor: T) -> Result<Owned<Op::ValA, Fragment<L>>>
    where
        Self: PartitionOperand<T, OperandA, Output = Tensor<E, L>>,
        L: FragmentLike,
        Fragment<L>: OwnedStorage<Op::ValA>,
    {
        Ok(make_fragment_like::<Op::ValA, L>(
            &self.partition_a(tensor)?.layout,
        ))
    }

    /// CuTe `ThrMMA::partition_fragment_B`: a register fragment shaped like [`ThrMma::partition_b`].
    pub fn partition_fragment_b<T, E, L>(&self, tensor: T) -> Result<Owned<Op::ValB, Fragment<L>>>
    where
        Self: PartitionOperand<T, OperandB, Output = Tensor<E, L>>,
        L: FragmentLike,
        Fragment<L>: OwnedStorage<Op::ValB>,
    {
        Ok(make_fragment_like::<Op::ValB, L>(
            &self.partition_b(tensor)?.layout,
        ))
    }

    /// CuTe `ThrMMA::partition_fragment_C`: a compact column-major accumulator
    /// of `ValTypeC` with the shape of [`ThrMma::partition_c`].
    pub fn partition_fragment_c<T, E, L>(
        &self,
        tensor: T,
    ) -> Result<Owned<Op::ValC, ColMajorLayout<L::Shape>>>
    where
        Self: PartitionOperand<T, OperandC, Output = Tensor<E, L>>,
        L: LayoutShape,
        L::Shape: CompactLeft,
        ColMajorLayout<L::Shape>: OwnedStorage<Op::ValC>,
    {
        self.partition_c(tensor)?;
        Ok(make_owned_tensor(
            <ColMajorLayout<L::Shape> as StaticValue>::VALUE,
        ))
    }
}
