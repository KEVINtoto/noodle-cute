//! Copy atoms, thread–value copy tilings, and explicit copy policies (`CopyTraits`,
//! `UniversalCopy`, `CopyAtom`, `TiledCopy`, `ThrCopy`, `PredicatedCopy`, `CopyPolicy`,
//! `copy_with`, `copy_if_with`).
//!
//! # Main Interfaces
//!
//! - [`CopyTraits`] & [`UniversalCopy`]: Instruction-level bit TV-layouts and pure-safe
//!   register-closure execution contract.
//! - [`CopyAtom`] & [`AtomValLayouts`]: Rescales bit layouts to element units via
//!   [`Recast`] and executes instruction groups over safe [`Tensor`]s.
//! - [`TiledCopy`], [`make_tiled_copy`], [`ThrValMap`], and [`ThrCopy`]: Tiles copy atoms
//!   over threads via [`raked_product`] and partitions source/destination tensors per thread
//!   (supporting linear, `F2`-swizzled, and [`identity_tensor`](crate::tensor::identity_tensor)
//!   layouts).
//! - [`PredicatedCopy`], [`CopyPolicy`], [`copy_with`], and [`copy_if_with`]: Explicit
//!   atom/tiling copy dispatch with instruction-group predication.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algorithm::atom::{CopyAtom, UniversalCopy, make_tiled_copy};
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//!
//! // 32 threads in a 4x8 grid, each copying 4 consecutive rows.
//! let atom = CopyAtom::<UniversalCopy<f32>, f32>::new();
//! let threads = Layout::from_shape(int_tuple![Int::<4>, Int::<8>]);
//! let values = Layout::from_shape(int_tuple![Int::<4>, Int::<1>]);
//! let tiled = make_tiled_copy(atom, threads, values);
//! assert_eq!(tiled.tiler, int_tuple![Int::<16>, Int::<8>]);
//! ```

use core::marker::PhantomData;
use core::ops::RangeFull;

use crate::NoodleCuteError;
use crate::StaticValue;
use crate::algebra::{
    CoalesceProfile, ComposeTile, Composition, RakedProduct, RightInverse, ZippedDivide,
    raked_product, right_inverse,
};
use crate::integer::{Int, Integer, add, div, int, mul, rem};
use crate::layout::{Layout, LayoutBase, OffsetAndSlice, Recast};
use crate::shape::Shape;
use crate::stride::{PrefixProduct, Stride};
use crate::tensor::{AccessorOffset, Tensor, TensorIndex, TensorSet};
use crate::tuple::{
    Get, GetOutput, LeafProfile, Nil, NilProfile, ProductEach, RepeatModes, Til, TupleProduct,
    TupleProfile, Zip2, product_each,
};

use super::{ColMajorLayout, ModeSize, RegisterCodec, SIZEOF_BITS};

/// One copy instruction: CuTe `Copy_Traits<CopyOp>` with pure-safe register execution.
///
/// `SrcLayout`, `DstLayout`, and `RefLayout` map `(thread, value)` to bits; [`CopyAtom`]
/// rescales them to elements of its logical value type `V`.
pub trait CopyTraits {
    /// `(thread, value) -> bit` of the source.
    type SrcLayout;
    /// `(thread, value) -> bit` of the destination.
    type DstLayout;
    /// Reference `(thread, value) -> bit` map tilings are expressed in.
    type RefLayout;
    /// Type of one source register.
    type SRegister: Copy;
    /// Type of one destination register.
    type DRegister: Copy;
    /// Number of source registers per instruction.
    const SRC_REGISTERS: int;
    /// Number of destination registers per instruction.
    const DST_REGISTERS: int;

    /// Executes one copy instruction reading registers `0..SRC_REGISTERS` via `src` and
    /// writing registers `0..DST_REGISTERS` via `dst`.
    fn copy(src: impl FnMut(int) -> Self::SRegister, dst: impl FnMut(int, Self::DRegister));

    /// Predicated instruction call. Backends may override this for zero-fill behavior.
    ///
    /// The default implementation skips a `false` predicate without invoking `src` or `dst`.
    fn copy_predicated(
        pred: bool,
        src: impl FnMut(int) -> Self::SRegister,
        dst: impl FnMut(int, Self::DRegister),
    ) {
        if pred {
            Self::copy(src, dst);
        }
    }
}

/// Copies one element per instruction with [`From`] conversion: CuTe `UniversalCopy<S, D>`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UniversalCopy<S, D = S>(PhantomData<(S, D)>);

/// Bit-level `(1, sizeof_bits(T)) : (0, 1)` layout for a single-thread copy of `T`.
type BitsLayout<T> = Layout<
    Til<Int<1>, Til<Int<{ SIZEOF_BITS::<T> }>, Nil>>,
    Til<Int<0>, Til<Int<1>, Nil>>,
>;

impl<S: Copy, D: Copy + From<S>> CopyTraits for UniversalCopy<S, D> {
    type SrcLayout = BitsLayout<S>;
    type DstLayout = BitsLayout<D>;
    type RefLayout = BitsLayout<S>;
    type SRegister = S;
    type DRegister = D;
    const SRC_REGISTERS: int = 1;
    const DST_REGISTERS: int = 1;

    fn copy(mut src: impl FnMut(int) -> S, mut dst: impl FnMut(int, D)) {
        dst(0, D::from(src(0)));
    }
}

/// A copy instruction applied to logical elements of type `V`: CuTe `Copy_Atom<CopyOp, V>`.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct CopyAtom<Op, V>(PhantomData<(Op, V)>);

impl<Op, V> Clone for CopyAtom<Op, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Op, V> Copy for CopyAtom<Op, V> {}

impl<Op, V> CopyAtom<Op, V> {
    /// Constructs the zero-sized copy atom for `Op` on element type `V`.
    pub const fn new() -> Self {
        Self(PhantomData)
    }

    /// Binds an instruction-wide predicate, forwarded to [`CopyTraits::copy_predicated`].
    pub const fn with(&self, predicate: bool) -> PredicatedCopy<Op, V> {
        PredicatedCopy {
            atom: *self,
            predicate,
        }
    }
}

/// Bit layout `L` rescaled to element units of `V` (`recast(L, sizeof_bits(V))`).
type ValLayout<L, V> = <L as Recast<Int<{ SIZEOF_BITS::<V> }>>>::Output;

/// `(thread, value) -> element` layouts of a [`CopyAtom`] (`ValLayoutRef`/`Src`/`Dst`).
pub trait AtomValLayouts {
    /// Reference element TV layout.
    type Ref;
    /// Source element TV layout.
    type Src;
    /// Destination element TV layout.
    type Dst;
}

impl<Op: CopyTraits, V> AtomValLayouts for CopyAtom<Op, V>
where
    Op::RefLayout: Recast<Int<{ SIZEOF_BITS::<V> }>>,
    Op::SrcLayout: Recast<Int<{ SIZEOF_BITS::<V> }>>,
    Op::DstLayout: Recast<Int<{ SIZEOF_BITS::<V> }>>,
{
    type Ref = ValLayout<Op::RefLayout, V>;
    type Src = ValLayout<Op::SrcLayout, V>;
    type Dst = ValLayout<Op::DstLayout, V>;
}

/// `(ref thread, ref value) -> (trg thread, trg value)` map: `right_inverse(Ref) o Trg`.
type RefToTrg<Ref, Trg> = <<Ref as RightInverse>::Output as Composition<Trg>>::Output;

impl<Op: CopyTraits, V> CopyAtom<Op, V> {
    /// Executes `Op` across consecutive instruction groups of `src` into `dst`
    /// (CuTe `Copy_Atom::call`).
    pub fn call<AS, SS, DS, AD, SD, DD, TS, TD>(
        &self,
        src: &Tensor<AS, Layout<SS, DS>>,
        dst: &mut Tensor<AD, Layout<SD, DD>>,
    ) where
        SS: Shape,
        DS: Stride<SS>,
        SD: Shape,
        DD: Stride<SD>,
        Tensor<AS, Layout<SS, DS>>: TensorIndex<int, Output = TS>,
        Tensor<AD, Layout<SD, DD>>: TensorSet<int, Item = TD>,
        TS: RegisterCodec<Op::SRegister>,
        TD: RegisterCodec<Op::DRegister>,
    {
        self.call_if(|_| true, src, dst);
    }

    /// Executes `Op` on instruction groups selected by `pred(group_index)`.
    ///
    /// A `false` predicate is forwarded to [`CopyTraits::copy_predicated`], allowing custom
    /// atoms to zero-fill without reading `src`.
    pub fn call_if<AS, SS, DS, AD, SD, DD, TS, TD>(
        &self,
        mut pred: impl FnMut(int) -> bool,
        src: &Tensor<AS, Layout<SS, DS>>,
        dst: &mut Tensor<AD, Layout<SD, DD>>,
    ) where
        SS: Shape,
        DS: Stride<SS>,
        SD: Shape,
        DD: Stride<SD>,
        Tensor<AS, Layout<SS, DS>>: TensorIndex<int, Output = TS>,
        Tensor<AD, Layout<SD, DD>>: TensorSet<int, Item = TD>,
        TS: RegisterCodec<Op::SRegister>,
        TD: RegisterCodec<Op::DRegister>,
    {
        let per_s = TS::ELEMS_PER_REG;
        let per_d = TD::ELEMS_PER_REG;
        let n = mul(Op::SRC_REGISTERS, per_s);
        let m = mul(Op::DST_REGISTERS, per_d);
        NoodleCuteError::InvalidShape.check_with(
            n > 0 && m > 0,
            "copy instructions must contain whole values",
        );
        let size = src.size();
        NoodleCuteError::NotDivisible.check_with(
            rem(size, n) == 0,
            "copy source size is not divisible by instruction value count",
        );
        let groups = div(size, n);
        NoodleCuteError::SizeMismatch.check_with(
            dst.size() == mul(groups, m),
            "copy operand sizes disagree",
        );
        let mut g: int = 0;
        while g < groups {
            let src_base = mul(g, n);
            let dst_base = mul(g, m);
            Op::copy_predicated(
                pred(g),
                |r| {
                    let reg_base = add(src_base, mul(r, per_s));
                    TS::pack_reg(|e| src.at(add(reg_base, e)))
                },
                |r, reg| {
                    let reg_base = add(dst_base, mul(r, per_d));
                    TD::unpack_reg(reg, |e, val| dst.set(add(reg_base, e), val));
                },
            );
            g = add(g, 1);
        }
    }
}

/// Copy atoms tiled over threads: CuTe `TiledCopy<Copy_Atom, LayoutCopy_TV, ShapeTiler_MN>`.
///
/// `layout_tv` maps `(thread, value)` to a flat coordinate inside the tile `tiler`.
#[derive(Debug, PartialEq, Eq)]
pub struct TiledCopy<Op, V, TV, Tiler> {
    /// Underlying copy atom.
    pub atom: CopyAtom<Op, V>,
    /// `(thread, value) -> tile coordinate`.
    pub layout_tv: TV,
    /// Tile shape covered by one application of this tiling.
    pub tiler: Tiler,
}

impl<Op, V, TV: Clone, Tiler: Clone> Clone for TiledCopy<Op, V, TV, Tiler> {
    fn clone(&self) -> Self {
        Self {
            atom: self.atom,
            layout_tv: self.layout_tv.clone(),
            tiler: self.tiler.clone(),
        }
    }
}

impl<Op, V, TV: Copy, Tiler: Copy> Copy for TiledCopy<Op, V, TV, Tiler> {}

type MnLayout<TS, TD, VS, VD> = <Layout<TS, TD> as RakedProduct<Layout<VS, VD>>>::Output;
type ThrVal<TS, VS> = Til<<TS as TupleProduct>::Output, Til<<VS as TupleProduct>::Output, Nil>>;
type TvLayout<TS, TD, VS, VD> = <<MnLayout<TS, TD, VS, VD> as RightInverse>::Output as Composition<
    ColMajorLayout<ThrVal<TS, VS>>,
>>::Output;
type CopyTiler<TS, TD, VS, VD> =
    <<MnLayout<TS, TD, VS, VD> as LayoutBase>::Shape as ProductEach>::Output;

/// Resulting [`TiledCopy`] type constructed by [`make_tiled_copy`].
pub type MadeCopy<Op, V, TS, TD, VS, VD> =
    TiledCopy<Op, V, TvLayout<TS, TD, VS, VD>, CopyTiler<TS, TD, VS, VD>>;

/// Constructs a [`TiledCopy`] by raking `val_layout` across `thr_layout`
/// (CuTe `make_tiled_copy(atom, thr_layout, val_layout)`).
///
/// Computes `layout_mn = raked_product(thr_layout, val_layout)`, inverts and reshapes it to
/// `(threads, values)` for `layout_tv`, and sets `tiler = product_each(layout_mn.shape)`.
pub const fn make_tiled_copy<Op, V, TS, TD, VS, VD>(
    atom: CopyAtom<Op, V>,
    thr_layout: Layout<TS, TD>,
    val_layout: Layout<VS, VD>,
) -> MadeCopy<Op, V, TS, TD, VS, VD>
where
    TS: [const] Shape + [const] TupleProduct,
    TD: Stride<TS>,
    VS: [const] Shape + [const] TupleProduct,
    VD: Stride<VS>,
    ThrVal<TS, VS>: Shape + [const] PrefixProduct<Int<1>>,
    <ThrVal<TS, VS> as PrefixProduct<Int<1>>>::Stride: Stride<ThrVal<TS, VS>>,
    ColMajorLayout<ThrVal<TS, VS>>: StaticValue,
    Layout<TS, TD>: [const] RakedProduct<Layout<VS, VD>>,
    MnLayout<TS, TD, VS, VD>: [const] RightInverse + [const] LayoutBase,
    <MnLayout<TS, TD, VS, VD> as LayoutBase>::Shape: [const] ProductEach,
    <MnLayout<TS, TD, VS, VD> as RightInverse>::Output:
        [const] Composition<ColMajorLayout<ThrVal<TS, VS>>>,
{
    let layout_mn = raked_product(thr_layout, val_layout);
    let layout_tv = right_inverse(layout_mn).composition(ColMajorLayout::<ThrVal<TS, VS>>::VALUE);
    let tiler = product_each(&layout_mn.shape());
    TiledCopy {
        atom,
        layout_tv,
        tiler,
    }
}

/// Element-unit reference TV layout of `CopyAtom<Op, V>`.
pub(crate) type ValRef<Op, V> = <CopyAtom<Op, V> as AtomValLayouts>::Ref;
/// Element-unit source TV layout of `CopyAtom<Op, V>`.
pub(crate) type ValSrc<Op, V> = <CopyAtom<Op, V> as AtomValLayouts>::Src;
/// Element-unit destination TV layout of `CopyAtom<Op, V>`.
pub(crate) type ValDst<Op, V> = <CopyAtom<Op, V> as AtomValLayouts>::Dst;

type AtomTile<Op, V> = Til<ModeSize<ValRef<Op, V>, 0>, Til<ModeSize<ValRef<Op, V>, 1>, Nil>>;
type Keep<M> = Til<M, Til<RangeFull, Nil>>;
type AtomTv<TV, Op, V> = <TV as ZippedDivide<AtomTile<Op, V>>>::Output;
type TrgTv<TV, Op, V, Trg> =
    <AtomTv<TV, Op, V> as ComposeTile<Keep<RefToTrg<ValRef<Op, V>, Trg>>>>::Output;
/// Coalesce profile `(Leaf, (Leaf, Leaf))` preserving `(Thr, (AtomVal, RestVal))`.
type AtomTvProfile = TupleProfile<
    LeafProfile,
    TupleProfile<TupleProfile<LeafProfile, TupleProfile<LeafProfile, NilProfile>>, NilProfile>,
>;
type ThrVal2Mn<TV, Op, V, Trg> =
    <<TrgTv<TV, Op, V, Trg> as Zip2>::Output as CoalesceProfile<AtomTvProfile>>::Output;
type TvTensor<L, Tiler, M> = <<L as ZippedDivide<Tiler>>::Output as ComposeTile<Keep<M>>>::Output;
/// Unfold coordinate `((.., ..), ..)` expanding `((Thr, Val), Rest)` into `(Thr, Val, Rest)`.
type Unfold = Til<Til<RangeFull, Til<RangeFull, Nil>>, Til<RangeFull, Nil>>;

/// `(thread, value, (rest...))` layout produced by [`TiledCopy::tidfrg_s`] / [`TiledCopy::tidfrg_d`].
pub type Tidfrg<L, Tiler, M> = Layout<
    <TvTensor<L, Tiler, M> as OffsetAndSlice<Unfold>>::SlicedShape,
    <TvTensor<L, Tiler, M> as OffsetAndSlice<Unfold>>::SlicedStride,
>;

type Map<C, Trg> = <C as ThrValMap<Trg>>::Output;
type SrcMap<Op, V, TV, Tiler> = Map<TiledCopy<Op, V, TV, Tiler>, ValSrc<Op, V>>;
type DstMap<Op, V, TV, Tiler> = Map<TiledCopy<Op, V, TV, Tiler>, ValDst<Op, V>>;
type SrcFrg<L, Op, V, TV, Tiler> = Tidfrg<L, Tiler, SrcMap<Op, V, TV, Tiler>>;
type DstFrg<L, Op, V, TV, Tiler> = Tidfrg<L, Tiler, DstMap<Op, V, TV, Tiler>>;

/// Computes the static `((Thr), (AtomVal, RestVal)) -> tile_coord` map for one side (`Trg`)
/// of a [`TiledCopy`] (CuTe `TiledCopy::tile2thrfrg` inner step).
pub const trait ThrValMap<Trg>: Copy {
    /// Coalesced `(Thr, (AtomVal, RestVal)) -> tile_coord` layout.
    type Output: Copy;

    /// Computes the TV-to-tile map.
    fn thrval2mn(&self) -> Self::Output;
}

const impl<Op, V, TV, Tiler, Trg> ThrValMap<Trg> for TiledCopy<Op, V, TV, Tiler>
where
    Op: CopyTraits,
    CopyAtom<Op, V>: AtomValLayouts,
    TV: Copy + [const] ZippedDivide<AtomTile<Op, V>>,
    Tiler: Copy,
    ValRef<Op, V>: LayoutBase + RightInverse,
    <ValRef<Op, V> as LayoutBase>::Shape: Get<0> + Get<1>,
    GetOutput<<ValRef<Op, V> as LayoutBase>::Shape, 0>: TupleProduct,
    GetOutput<<ValRef<Op, V> as LayoutBase>::Shape, 1>: TupleProduct,
    ModeSize<ValRef<Op, V>, 0>: StaticValue,
    ModeSize<ValRef<Op, V>, 1>: StaticValue,
    <ValRef<Op, V> as RightInverse>::Output: Composition<Trg>,
    RefToTrg<ValRef<Op, V>, Trg>: StaticValue,
    AtomTv<TV, Op, V>: [const] ComposeTile<Keep<RefToTrg<ValRef<Op, V>, Trg>>>,
    TrgTv<TV, Op, V, Trg>: [const] Zip2,
    <TrgTv<TV, Op, V, Trg> as Zip2>::Output: [const] CoalesceProfile<AtomTvProfile>,
{
    type Output = ThrVal2Mn<TV, Op, V, Trg>;

    fn thrval2mn(&self) -> Self::Output {
        let atom_tv = self.layout_tv.zipped_divide(AtomTile::<Op, V>::VALUE);
        let ref2trg = <RefToTrg<ValRef<Op, V>, Trg> as StaticValue>::VALUE;
        atom_tv
            .compose_tile(Til(ref2trg, Til(.., Nil)))
            .zip2()
            .coalesce_with_profile()
    }
}

impl<Op: CopyTraits, V, TV: Copy, Tiler: Copy> TiledCopy<Op, V, TV, Tiler> {
    /// Tiles `layout` by `self.tiler` and maps each tile into `(thread, value, (rest...))`
    /// through target side `Trg`.
    const fn tile2thrfrg<L, Trg>(&self, layout: &L) -> Tidfrg<L, Tiler, Map<Self, Trg>>
    where
        Self: [const] ThrValMap<Trg>,
        L: Copy + [const] ZippedDivide<Tiler>,
        L::Output: [const] ComposeTile<Keep<Map<Self, Trg>>>,
        TvTensor<L, Tiler, Map<Self, Trg>>: [const] OffsetAndSlice<Unfold>,
    {
        let tiled = (*layout).zipped_divide(self.tiler);
        let tv = tiled.compose_tile(Til(self.thrval2mn(), Til(.., Nil)));
        let unfold: Unfold = Til(Til(.., Til(.., Nil)), Til(.., Nil));
        let (_, sliced) = tv.offset_and_slice(unfold);
        sliced
    }

    /// CuTe `TiledCopy::tidfrg_S`: `(thread, value, (rest...))` of a source layout.
    pub const fn tidfrg_s<L>(&self, layout: &L) -> SrcFrg<L, Op, V, TV, Tiler>
    where
        CopyAtom<Op, V>: AtomValLayouts,
        Self: [const] ThrValMap<ValSrc<Op, V>>,
        L: Copy + [const] ZippedDivide<Tiler>,
        L::Output: [const] ComposeTile<Keep<SrcMap<Op, V, TV, Tiler>>>,
        TvTensor<L, Tiler, SrcMap<Op, V, TV, Tiler>>: [const] OffsetAndSlice<Unfold>,
    {
        self.tile2thrfrg::<L, ValSrc<Op, V>>(layout)
    }

    /// CuTe `TiledCopy::tidfrg_D`: `(thread, value, (rest...))` of a destination layout.
    pub const fn tidfrg_d<L>(&self, layout: &L) -> DstFrg<L, Op, V, TV, Tiler>
    where
        CopyAtom<Op, V>: AtomValLayouts,
        Self: [const] ThrValMap<ValDst<Op, V>>,
        L: Copy + [const] ZippedDivide<Tiler>,
        L::Output: [const] ComposeTile<Keep<DstMap<Op, V, TV, Tiler>>>,
        TvTensor<L, Tiler, DstMap<Op, V, TV, Tiler>>: [const] OffsetAndSlice<Unfold>,
    {
        self.tile2thrfrg::<L, ValDst<Op, V>>(layout)
    }

    /// Returns the per-thread partitioner for `thr_idx` (CuTe `TiledCopy::get_slice`).
    pub const fn get_slice<X: Integer>(&self, thr_idx: X) -> ThrCopy<'_, Op, V, TV, Tiler, X> {
        ThrCopy {
            tiled: self,
            thr_idx,
        }
    }
}

/// One thread's view of a [`TiledCopy`]: CuTe `ThrCopy`.
#[derive(Debug)]
pub struct ThrCopy<'a, Op, V, TV, Tiler, X> {
    pub(crate) tiled: &'a TiledCopy<Op, V, TV, Tiler>,
    thr_idx: X,
}

impl<Op, V, TV, Tiler, X: Copy> Clone for ThrCopy<'_, Op, V, TV, Tiler, X> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Op, V, TV, Tiler, X: Copy> Copy for ThrCopy<'_, Op, V, TV, Tiler, X> {}

type RestWild<F> = <GetOutput<<F as LayoutBase>::Shape, 2> as RepeatModes<RangeFull>>::Output;
type ThrCoord<X, F> = Til<X, Til<RangeFull, Til<RestWild<F>, Nil>>>;

/// One thread's `(value, rest...)` sub-[`Tensor`] of a `(thread, value, (rest...))` layout `F`.
pub type ThrPartition<A, F, X> = Tensor<
    <A as AccessorOffset<<F as OffsetAndSlice<ThrCoord<X, F>>>::Offset>>::Output,
    Layout<
        <F as OffsetAndSlice<ThrCoord<X, F>>>::SlicedShape,
        <F as OffsetAndSlice<ThrCoord<X, F>>>::SlicedStride,
    >,
>;

type SrcPartition<A, L, Op, V, TV, Tiler, X> = ThrPartition<A, SrcFrg<L, Op, V, TV, Tiler>, X>;
type DstPartition<A, L, Op, V, TV, Tiler, X> = ThrPartition<A, DstFrg<L, Op, V, TV, Tiler>, X>;

const fn thr_coord<X, F>(thr_idx: X) -> ThrCoord<X, F>
where
    F: LayoutBase,
    F::Shape: Get<2>,
    GetOutput<F::Shape, 2>: [const] RepeatModes<RangeFull>,
{
    Til(
        thr_idx,
        Til(
            ..,
            Til(
                <GetOutput<F::Shape, 2> as RepeatModes<RangeFull>>::repeat_modes(..),
                Nil,
            ),
        ),
    )
}

impl<Op: CopyTraits, V, TV: Copy, Tiler: Copy, X: Integer> ThrCopy<'_, Op, V, TV, Tiler, X> {
    /// CuTe `ThrCopy::partition_S`: returns this thread's `(value, rest...)` partition of a
    /// source tensor (works on linear, `F2`-swizzled, and [`identity_tensor`](crate::tensor::identity_tensor)
    /// views).
    pub const fn partition_s<A, S, D>(
        &self,
        tensor: Tensor<A, Layout<S, D>>,
    ) -> SrcPartition<A, Layout<S, D>, Op, V, TV, Tiler, X>
    where
        S: Shape,
        D: Stride<S>,
        CopyAtom<Op, V>: AtomValLayouts,
        TiledCopy<Op, V, TV, Tiler>: [const] ThrValMap<ValSrc<Op, V>>,
        Layout<S, D>: [const] ZippedDivide<Tiler>,
        <Layout<S, D> as ZippedDivide<Tiler>>::Output:
            [const] ComposeTile<Keep<SrcMap<Op, V, TV, Tiler>>>,
        TvTensor<Layout<S, D>, Tiler, SrcMap<Op, V, TV, Tiler>>: [const] OffsetAndSlice<Unfold>,
        SrcFrg<Layout<S, D>, Op, V, TV, Tiler>: LayoutBase
            + [const] OffsetAndSlice<ThrCoord<X, SrcFrg<Layout<S, D>, Op, V, TV, Tiler>>>,
        <SrcFrg<Layout<S, D>, Op, V, TV, Tiler> as LayoutBase>::Shape: Get<2>,
        GetOutput<<SrcFrg<Layout<S, D>, Op, V, TV, Tiler> as LayoutBase>::Shape, 2>:
            [const] RepeatModes<RangeFull>,
        A: [const] AccessorOffset<
            <SrcFrg<Layout<S, D>, Op, V, TV, Tiler> as OffsetAndSlice<
                ThrCoord<X, SrcFrg<Layout<S, D>, Op, V, TV, Tiler>>,
            >>::Offset,
        >,
    {
        let frg = self.tiled.tidfrg_s(&tensor.layout);
        let coord = thr_coord::<X, SrcFrg<Layout<S, D>, Op, V, TV, Tiler>>(self.thr_idx);
        let (off, sliced) = frg.offset_and_slice(coord);
        Tensor {
            accessor: tensor.accessor.offset(off),
            layout: sliced,
        }
    }

    /// CuTe `ThrCopy::partition_D`: returns this thread's `(value, rest...)` partition of a
    /// destination tensor.
    pub const fn partition_d<A, S, D>(
        &self,
        tensor: Tensor<A, Layout<S, D>>,
    ) -> DstPartition<A, Layout<S, D>, Op, V, TV, Tiler, X>
    where
        S: Shape,
        D: Stride<S>,
        CopyAtom<Op, V>: AtomValLayouts,
        TiledCopy<Op, V, TV, Tiler>: [const] ThrValMap<ValDst<Op, V>>,
        Layout<S, D>: [const] ZippedDivide<Tiler>,
        <Layout<S, D> as ZippedDivide<Tiler>>::Output:
            [const] ComposeTile<Keep<DstMap<Op, V, TV, Tiler>>>,
        TvTensor<Layout<S, D>, Tiler, DstMap<Op, V, TV, Tiler>>: [const] OffsetAndSlice<Unfold>,
        DstFrg<Layout<S, D>, Op, V, TV, Tiler>: LayoutBase
            + [const] OffsetAndSlice<ThrCoord<X, DstFrg<Layout<S, D>, Op, V, TV, Tiler>>>,
        <DstFrg<Layout<S, D>, Op, V, TV, Tiler> as LayoutBase>::Shape: Get<2>,
        GetOutput<<DstFrg<Layout<S, D>, Op, V, TV, Tiler> as LayoutBase>::Shape, 2>:
            [const] RepeatModes<RangeFull>,
        A: [const] AccessorOffset<
            <DstFrg<Layout<S, D>, Op, V, TV, Tiler> as OffsetAndSlice<
                ThrCoord<X, DstFrg<Layout<S, D>, Op, V, TV, Tiler>>,
            >>::Offset,
        >,
    {
        let frg = self.tiled.tidfrg_d(&tensor.layout);
        let coord = thr_coord::<X, DstFrg<Layout<S, D>, Op, V, TV, Tiler>>(self.thr_idx);
        let (off, sliced) = frg.offset_and_slice(coord);
        Tensor {
            accessor: tensor.accessor.offset(off),
            layout: sliced,
        }
    }
}

/// Copy atom paired with an instruction-wide predicate: CuTe `atom.with(pred)`.
#[derive(Debug)]
pub struct PredicatedCopy<Op, V> {
    atom: CopyAtom<Op, V>,
    predicate: bool,
}

impl<Op, V> Clone for PredicatedCopy<Op, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Op, V> Copy for PredicatedCopy<Op, V> {}

/// Explicit copy dispatch policy implemented by [`CopyAtom`], [`PredicatedCopy`],
/// [`TiledCopy`], and [`ThrCopy`].
pub trait CopyPolicy<S, D> {
    /// Copies instruction groups selected by `pred(group_index)` from `src` to `dst`.
    fn copy_if_policy(&self, pred: impl FnMut(int) -> bool, src: &S, dst: &mut D);
}

impl<Op: CopyTraits, V, AS, SS, DS, AD, SD, DD, TS, TD>
    CopyPolicy<Tensor<AS, Layout<SS, DS>>, Tensor<AD, Layout<SD, DD>>> for CopyAtom<Op, V>
where
    SS: Shape,
    DS: Stride<SS>,
    SD: Shape,
    DD: Stride<SD>,
    Tensor<AS, Layout<SS, DS>>: TensorIndex<int, Output = TS>,
    Tensor<AD, Layout<SD, DD>>: TensorSet<int, Item = TD>,
    TS: RegisterCodec<Op::SRegister>,
    TD: RegisterCodec<Op::DRegister>,
{
    fn copy_if_policy(
        &self,
        pred: impl FnMut(int) -> bool,
        src: &Tensor<AS, Layout<SS, DS>>,
        dst: &mut Tensor<AD, Layout<SD, DD>>,
    ) {
        self.call_if(pred, src, dst);
    }
}

impl<Op, V, S, D> CopyPolicy<S, D> for PredicatedCopy<Op, V>
where
    CopyAtom<Op, V>: CopyPolicy<S, D>,
{
    fn copy_if_policy(&self, mut pred: impl FnMut(int) -> bool, src: &S, dst: &mut D) {
        self.atom
            .copy_if_policy(|i| self.predicate && pred(i), src, dst);
    }
}

impl<Op, V, TV, Tiler, S, D> CopyPolicy<S, D> for TiledCopy<Op, V, TV, Tiler>
where
    CopyAtom<Op, V>: CopyPolicy<S, D>,
{
    fn copy_if_policy(&self, pred: impl FnMut(int) -> bool, src: &S, dst: &mut D) {
        self.atom.copy_if_policy(pred, src, dst);
    }
}

impl<Op, V, TV, Tiler, X, S, D> CopyPolicy<S, D> for ThrCopy<'_, Op, V, TV, Tiler, X>
where
    CopyAtom<Op, V>: CopyPolicy<S, D>,
{
    fn copy_if_policy(&self, pred: impl FnMut(int) -> bool, src: &S, dst: &mut D) {
        self.tiled.atom.copy_if_policy(pred, src, dst);
    }
}

/// Copies already partitioned tensors using an explicit copy atom or tiling policy
/// (CuTe `copy(policy, src, dst)`).
pub fn copy_with<P: CopyPolicy<S, D>, S, D>(policy: &P, src: &S, dst: &mut D) {
    policy.copy_if_policy(|_| true, src, dst);
}

/// Predicated explicit copy with one predicate per instruction group
/// (CuTe `copy_if(policy, pred, src, dst)`).
///
/// A `false` predicate skips reading `src` and forwards directly to
/// [`CopyTraits::copy_predicated`] (enabling zero-fill backends).
pub fn copy_if_with<P: CopyPolicy<S, D>, S, D>(
    policy: &P,
    pred: impl FnMut(int) -> bool,
    src: &S,
    dst: &mut D,
) {
    policy.copy_if_policy(pred, src, dst);
}
