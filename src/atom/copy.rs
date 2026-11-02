//! MMA-compatible copy layouts and fragment retiling (`copy_atom.hpp`).

use super::*;

/// The extent of a resolved MMA permutation mode.
pub trait TileExtent {
    /// The static extent.
    type Size;
}
impl<const N: isize> TileExtent for Int<N> {
    type Size = Self;
}
impl<S: Product, D> TileExtent for Layout<S, D> {
    type Size = S::Output;
}

type PermOf<M, W> = <M as MmaOperand<W>>::Perm;
type Extent<M, W, const I: usize> = <<PermOf<M, W> as Get<I>>::Output as TileExtent>::Size;
/// The two-dimensional tile of an MMA operand.
pub type MmaCopyTile<M, W> = Pair<Extent<M, W, 0>, Extent<M, W, 1>>;
type RefLayout<M, W> = ColMajorLayout<MmaCopyTile<M, W>>;
type RefFrg<M, W> = <M as ThrFrgOf<RefLayout<M, W>, W>>::Output;
type Sizes<V> = Vmnk4<ThrSize<V, 0>, ThrSize<V, 1>, ThrSize<V, 2>, ThrSize<V, 3>>;

/// Projects a `(V,M,N,K)` thread coordinate to one operand's thread coordinate.
pub trait CopyProjection<V> {
    /// The projection layout, linearized in the operand's thread modes.
    type Layout;
}
macro_rules! projection {
    ($w:ty, $d1:ty, $d2:ty, $d3:ty) => {
        impl<V: LayoutParts> CopyProjection<V> for $w
        where
            VmnkShape<V>: Get<0> + Get<1> + Get<2> + Get<3>,
            <VmnkShape<V> as Get<0>>::Output: Product,
            <VmnkShape<V> as Get<1>>::Output: Product,
            <VmnkShape<V> as Get<2>>::Output: Product,
            <VmnkShape<V> as Get<3>>::Output: Product,
            ThrSize<V, 0>:
                Arith<ThrSize<V, 1>, { ArithOp::Mul }> + Arith<ThrSize<V, 2>, { ArithOp::Mul }>,
        {
            type Layout = Layout<Sizes<V>, Vmnk4<Int<1>, $d1, $d2, $d3>>;
        }
    };
}
projection!(OperandA, ThrSize<V,0>, Int<0>, Prod<ThrSize<V,0>,ThrSize<V,1>>);
projection!(OperandB, Int<0>, ThrSize<V,0>, Prod<ThrSize<V,0>,ThrSize<V,2>>);
projection!(OperandC, ThrSize<V,0>, Prod<ThrSize<V,0>,ThrSize<V,1>>, Int<0>);

type ThreadMap<V, W> = TypedComposition<<W as CopyProjection<V>>::Layout, TypedInverse<V>>;
type MappedFrg<M, V, W> = <RefFrg<M, W> as ComposeTile<Keep<ThreadMap<V, W>>>>::Output;
type TvProfile = Pair<Int<1>, Int<1>>;

/// CuTe `get_layoutA_TV`, `get_layoutB_TV`, and `get_layoutC_TV`.
///
/// ```
/// use noodle_cute::atom::{CopyAtom, MmaAtom, UniversalCopy, UniversalFma,
///     make_tiled_copy_c, make_tiled_mma};
/// use noodle_cute::algorithm::copy_with;
/// use noodle_cute::coord::X;
/// use noodle_cute::layout::LayoutLeft;
/// use noodle_cute::tensor::make_tensor;
/// use noodle_cute::{Int, IntTuple, Nil, shape};
///
/// let mma = make_tiled_mma(MmaAtom::<UniversalFma<f32>>::new(),
///     LayoutLeft::make(shape![Int::<2>, Int::<2>])?, IntTuple(X, Nil))?;
/// let copy = make_tiled_copy_c(CopyAtom::<UniversalCopy<f32>, f32>::new(), &mma)?;
/// let data = [7.0f32; 16];
/// let tensor = make_tensor(data.as_ptr(), LayoutLeft::make(shape![Int::<4>, Int::<4>])?);
/// let mut fragment = mma.get_slice(1isize).partition_fragment_c(tensor)?;
/// let thread = copy.get_slice(1isize);
/// let src = thread.partition_s(tensor)?;
/// let dst = thread.retile_d(fragment.view_mut())?;
/// unsafe { copy_with(&copy, &src, &dst) };
/// assert_eq!(fragment.get(0isize), 7.0);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
pub trait MmaCopyLayout<W> {
    /// `(thread index, value index) -> operand tile index`.
    type Tv;
    /// The operand's tile extents.
    type Tile;
    /// Returns the TV layout and tile extents.
    fn copy_layout(&self) -> Result<(Self::Tv, Self::Tile)>;
}
impl<Op, V, P, W> MmaCopyLayout<W> for TiledMma<Op, V, P>
where
    Self: MmaOperand<W> + ThrFrgOf<RefLayout<Self, W>, W>,
    PermOf<Self, W>: Get<0> + Get<1>,
    <PermOf<Self, W> as Get<0>>::Output: TileExtent,
    <PermOf<Self, W> as Get<1>>::Output: TileExtent,
    MmaCopyTile<Self, W>: CompactLeft + StaticValue,
    RefLayout<Self, W>: StaticValue,
    V: TypedInvert,
    W: CopyProjection<V>,
    <W as CopyProjection<V>>::Layout: StaticValue + TypedCompose<TypedInverse<V>>,
    ThreadMap<V, W>: StaticValue,
    RefFrg<Self, W>: ComposeTile<Keep<ThreadMap<V, W>>>,
    MappedFrg<Self, V, W>: CoalesceProfile<TvProfile>,
{
    type Tv = <MappedFrg<Self, V, W> as CoalesceProfile<TvProfile>>::Output;
    type Tile = MmaCopyTile<Self, W>;
    fn copy_layout(&self) -> Result<(Self::Tv, Self::Tile)> {
        let frg = self.thrfrg(&RefLayout::<Self, W>::VALUE)?;
        let tv = frg
            .compose_tile(IntTuple(
                ThreadMap::<V, W>::VALUE,
                IntTuple(Underscore, Nil),
            ))?
            .coalesce_profile(TvProfile::VALUE)?;
        Ok((tv, Self::Tile::VALUE))
    }
}

/// Checks the atom's thread and value divisibility against a tiled TV layout.
pub trait CopyTileCheck<Op, V> {
    fn check_copy_tile(&self) -> Result<()>;
}
impl<TV, Op, V> CopyTileCheck<Op, V> for TV
where
    TV: LayoutParts,
    CopyAtom<Op, V>: AtomValLayouts,
    ValRef<Op, V>: LayoutParts,
    <ValRef<Op, V> as LayoutParts>::Shape: Get<0> + Get<1>,
    TV::Shape: Get<0> + Get<1>,
    Mode<ValRef<Op, V>, 0>: Product,
    Mode<ValRef<Op, V>, 1>: Product,
    ModeSize<ValRef<Op, V>, 0>: Integer + StaticValue,
    ModeSize<ValRef<Op, V>, 1>: Integer + StaticValue,
    Mode<TV, 0>: Product,
    Mode<TV, 1>: Product,
    ModeSize<TV, 0>: Integer + StaticValue,
    ModeSize<TV, 1>: Integer + StaticValue,
{
    fn check_copy_tile(&self) -> Result<()> {
        let (t, v) = (
            ModeSize::<TV, 0>::VALUE.value(),
            ModeSize::<TV, 1>::VALUE.value(),
        );
        let (at, av) = (
            ModeSize::<ValRef<Op, V>, 0>::VALUE.value(),
            ModeSize::<ValRef<Op, V>, 1>::VALUE.value(),
        );
        if at <= 0 || av <= 0 {
            return Err(crate::Error::InvalidShape);
        }
        if t % at != 0 || v % av != 0 {
            return Err(crate::Error::NotDivisible);
        }
        Ok(())
    }
}

macro_rules! mma_copy {
    ($function:ident,$method:ident,$w:ty,$doc:literal) => {
        #[doc = $doc]
        pub fn $function<Op, V, M: MmaCopyLayout<$w>>(
            atom: CopyAtom<Op, V>,
            mma: &M,
        ) -> Result<TiledCopy<Op, V, M::Tv, M::Tile>>
        where
            M::Tv: CopyTileCheck<Op, V>,
        {
            let (layout_tv, tiler) = mma.copy_layout()?;
            layout_tv.check_copy_tile()?;
            Ok(TiledCopy {
                atom,
                layout_tv,
                tiler,
            })
        }
        impl<Op, V, P> TiledMma<Op, V, P> {
            /// Returns the MMA operand's `(thread index, value index)` layout.
            pub fn $method(&self) -> Result<<Self as MmaCopyLayout<$w>>::Tv>
            where
                Self: MmaCopyLayout<$w>,
            {
                Ok(<Self as MmaCopyLayout<$w>>::copy_layout(self)?.0)
            }
        }
    };
}
mma_copy!(
    make_tiled_copy_a,
    get_layout_a_tv,
    OperandA,
    "CuTe `make_tiled_copy_A`: copy tiling with the MMA A thread–value map."
);
mma_copy!(
    make_tiled_copy_b,
    get_layout_b_tv,
    OperandB,
    "CuTe `make_tiled_copy_B`: copy tiling with the MMA B thread–value map."
);
mma_copy!(
    make_tiled_copy_c,
    get_layout_c_tv,
    OperandC,
    "CuTe `make_tiled_copy_C`: copy tiling with the MMA C thread–value map."
);

// Retile the reference fragment; source and destination use the same map upstream.
type FrgValues<L> = <<L as LayoutParts>::Shape as Get<0>>::Output;
type VCount<L> = <FrgValues<L> as Product>::Output;
const RETILE_FACTOR<T: Integer, V: Integer>: isize = match (T::STATIC, V::STATIC) {
    (Some(t), Some(v)) => t * v,
    _ => panic!("retile requires a static reference fragment"),
};
type InverseTile<TV, T> = TypedComposition<TypedInverse<TV>, ColMajorLayout<T>>;
type FrgMn<TV, T, L> = Layout<
    <InverseTile<TV, T> as Upcast<{ RETILE_FACTOR::<ModeSize<TV, 0>, VCount<L>> }>>::Shape,
    <InverseTile<TV, T> as Upcast<{ RETILE_FACTOR::<ModeSize<TV, 0>, VCount<L>> }>>::Stride,
>;
type VLayout<L> = Layout<VCount<L>, Int<1>>;
type FrgProduct<TV, T, L> =
    TypedProductLayout<VLayout<L>, TypedInverse<FrgMn<TV, T, L>>, { ProductKind::Logical }>;
type RetileV<TV, T, L, Op, V> =
    <FrgProduct<TV, T, L> as ZippedDivide<Layout<ModeSize<ValRef<Op, V>, 1>, Int<1>>>>::Output;
type RetileTile<TV, T, L> =
    IntTuple<VCount<L>, <<FrgMn<TV, T, L> as LayoutParts>::Shape as ProductEach>::Output>;
type RetileT<TV, T, L> = <L as ZippedDivide<RetileTile<TV, T, L>>>::Output;
type RetileMapped<TV, T, L, Op, V> =
    <RetileT<TV, T, L> as ComposeTile<Keep<RetileV<TV, T, L, Op, V>>>>::Output;
pub trait RestModes {
    type Rest;
}
impl<H, T> RestModes for IntTuple<H, T> {
    type Rest = T;
}
type RestShape<L> = <<L as LayoutParts>::Shape as RestModes>::Rest;
type RetileRest<L> = <RestShape<L> as RepeatModes<Underscore>>::Output;
type RetileCoord<L> = Pair<Underscore, IntTuple<Int<0>, RetileRest<L>>>;

/// CuTe `TiledCopy::retile`, keeping the fragment's storage and changing its layout.
pub trait Retile<L> {
    /// The retiled reference fragment layout.
    type Output;
    /// Computes the new layout.
    fn retile(&self, layout: &L) -> Result<Self::Output>;
}

impl<Op, V, TV, T, L> Retile<L> for TiledCopy<Op, V, TV, T>
where
    CopyAtom<Op, V>: AtomValLayouts,
    ValRef<Op, V>: LayoutParts,
    <ValRef<Op, V> as LayoutParts>::Shape: Get<1>,
    Mode<ValRef<Op, V>, 1>: Product,
    ModeSize<ValRef<Op, V>, 1>: StaticValue,
    TV: LayoutParts + TypedInvert + Clone,
    TV::Shape: Get<0>,
    Mode<TV, 0>: Product,
    ModeSize<TV, 0>: Integer,
    T: CompactLeft,
    ColMajorLayout<T>: StaticValue,
    TypedInverse<TV>: TypedCompose<ColMajorLayout<T>>,
    L: LayoutParts + ZippedDivide<RetileTile<TV, T, L>>,
    L::Shape: Get<0> + RestModes,
    FrgValues<L>: Product,
    VCount<L>: Integer + StaticValue,
    InverseTile<TV, T>: Upcast<{ RETILE_FACTOR::<ModeSize<TV, 0>, VCount<L>> }>,
    FrgMn<TV, T, L>: LayoutParts + TypedInvert,
    <FrgMn<TV, T, L> as LayoutParts>::Shape: ProductEach,
    <<FrgMn<TV, T, L> as LayoutParts>::Shape as ProductEach>::Output: StaticValue,
    VLayout<L>: TypedProduct<TypedInverse<FrgMn<TV, T, L>>, { ProductKind::Logical }>,
    FrgProduct<TV, T, L>: ZippedDivide<Layout<ModeSize<ValRef<Op, V>, 1>, Int<1>>>,
    RetileT<TV, T, L>: ComposeTile<Keep<RetileV<TV, T, L, Op, V>>>,
    RestShape<L>: RepeatModes<Underscore>,
    RetileMapped<TV, T, L, Op, V>: SliceAt<RetileCoord<L>>,
{
    type Output = <RetileMapped<TV, T, L, Op, V> as SliceAt<RetileCoord<L>>>::Sliced;
    fn retile(&self, layout: &L) -> Result<Self::Output> {
        let inverse = TV::invert(self.layout_tv.clone())?;
        let frg_mn = inverse.compose(ColMajorLayout::<T>::VALUE)?.upcast();
        let tile = IntTuple(VCount::<L>::VALUE, StaticValue::VALUE);
        let product = VLayout::<L>::VALUE.product_typed(frg_mn.invert()?)?;
        let frg_v = product.zipped_divide(Layout {
            shape: ModeSize::<ValRef<Op, V>, 1>::VALUE,
            stride: Int::<1>,
        })?;
        let mapped = layout
            .zipped_divide(tile)?
            .compose_tile(IntTuple(frg_v, IntTuple(Underscore, Nil)))?;
        let coord = IntTuple(
            Underscore,
            IntTuple(
                IntTuple(Int::<0>, RestShape::<L>::repeat_modes(Underscore)),
                Nil,
            ),
        );
        Ok(mapped.slice_and_offset(coord).0)
    }
}

type RetiledView<I, L, Op, V, TV, T> = View<I, <TiledCopy<Op, V, TV, T> as Retile<L>>::Output>;

impl<Op, V, TV, T, X> ThrCopy<'_, Op, V, TV, T, X> {
    /// CuTe `retile_S`: views a reference fragment in copy instruction order.
    ///
    /// The input's first mode contains the MMA atom values. The remaining
    /// modes supply the copy's additional values. No data is moved.
    pub fn retile_s<I: Iter, L>(
        &self,
        tensor: View<I, L>,
    ) -> Result<RetiledView<I, L, Op, V, TV, T>>
    where
        TiledCopy<Op, V, TV, T>: Retile<L>,
    {
        Ok(make_tensor(
            tensor.data(),
            self.tiled.retile(&tensor.layout)?,
        ))
    }
    /// CuTe `retile_D`: the same reference-fragment transformation as `retile_S`.
    pub fn retile_d<I: Iter, L>(
        &self,
        tensor: View<I, L>,
    ) -> Result<RetiledView<I, L, Op, V, TV, T>>
    where
        TiledCopy<Op, V, TV, T>: Retile<L>,
    {
        self.retile_s(tensor)
    }
}

/// A copy atom with an instruction-wide predicate: CuTe `atom.with(pred)`.
#[derive(Clone, Copy, Debug)]
pub struct PredicatedCopy<Op, V> {
    atom: CopyAtom<Op, V>,
    predicate: bool,
}
impl<Op, V> CopyAtom<Op, V> {
    /// Binds a predicate, forwarded to the operation's predicated call.
    pub fn with(&self, predicate: bool) -> PredicatedCopy<Op, V> {
        PredicatedCopy {
            atom: *self,
            predicate,
        }
    }
}

impl<Op: CopyTraits, V, SI, SL, DI, DL> crate::algorithm::CopyPolicy<View<SI, SL>, View<DI, DL>>
    for CopyAtom<Op, V>
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
    unsafe fn copy_if_policy(
        &self,
        pred: impl Fn(isize) -> bool,
        src: &View<SI, SL>,
        dst: &View<DI, DL>,
    ) {
        unsafe { self.call_if(pred, src, dst) }
    }
}
impl<Op, V, S, D> crate::algorithm::CopyPolicy<S, D> for PredicatedCopy<Op, V>
where
    CopyAtom<Op, V>: crate::algorithm::CopyPolicy<S, D>,
{
    unsafe fn copy_if_policy(&self, pred: impl Fn(isize) -> bool, src: &S, dst: &D) {
        unsafe {
            self.atom
                .copy_if_policy(|i| self.predicate && pred(i), src, dst)
        }
    }
}
impl<Op, V, TV, T, S, D> crate::algorithm::CopyPolicy<S, D> for TiledCopy<Op, V, TV, T>
where
    CopyAtom<Op, V>: crate::algorithm::CopyPolicy<S, D>,
{
    unsafe fn copy_if_policy(&self, pred: impl Fn(isize) -> bool, src: &S, dst: &D) {
        unsafe { self.atom.copy_if_policy(pred, src, dst) }
    }
}
impl<Op, V, TV, T, X, S, D> crate::algorithm::CopyPolicy<S, D> for ThrCopy<'_, Op, V, TV, T, X>
where
    CopyAtom<Op, V>: crate::algorithm::CopyPolicy<S, D>,
{
    unsafe fn copy_if_policy(&self, pred: impl Fn(isize) -> bool, src: &S, dst: &D) {
        unsafe { self.tiled.atom.copy_if_policy(pred, src, dst) }
    }
}
