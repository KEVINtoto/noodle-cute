//! Mode-wise algebra used by copy and MMA partitioning: CuTe's
//! `composition(layout, tile)`, `zip`, and `coalesce(layout, profile)`
//! (`layout.hpp`, `layout_composed.hpp`).

use crate::Result;
use crate::coord::Underscore;
use crate::int::Int;
use crate::layout::{ComposedLayout, Layout, LayoutParts, Shape, StrideOf};
use crate::tuple::{Flatten, IntTuple, Nil};

use super::coalesce::{CoalesceFold, FinishCoalesce};
use super::complement::UnwrapLayout;
use super::{CoalescedLayout, TypedCompose};

// Implementation traits that appear in public bounds but are not part of the API.
mod detail {
    use crate::Result;

    /// One mode of `composition(layout, tile)`: a layout composes, `_` keeps the mode.
    pub trait ComposeMode<T> {
        type Shape;
        type Stride;
        fn compose_mode(self, tiler: T)
        -> Result<crate::layout::Layout<Self::Shape, Self::Stride>>;
    }
}
use detail::ComposeMode;

impl<S, D> ComposeMode<Underscore> for Layout<S, D> {
    type Shape = S;
    type Stride = D;
    fn compose_mode(self, _: Underscore) -> Result<Self> {
        Ok(self)
    }
}

impl<S, D, RS, RD> ComposeMode<Layout<RS, RD>> for Layout<S, D>
where
    Self: TypedCompose<Layout<RS, RD>>,
    <Self as TypedCompose<Layout<RS, RD>>>::Output: LayoutParts,
{
    type Shape = <<Self as TypedCompose<Layout<RS, RD>>>::Output as LayoutParts>::Shape;
    type Stride = <<Self as TypedCompose<Layout<RS, RD>>>::Output as LayoutParts>::Stride;
    fn compose_mode(self, tiler: Layout<RS, RD>) -> Result<Layout<Self::Shape, Self::Stride>> {
        Ok(self.compose(tiler)?.into_layout())
    }
}

/// CuTe `composition(layout, tile)` for a tuple tile: mode `i` of the layout is
/// composed with element `i` of the tile, where `_` keeps the mode.
///
/// As upstream, modes beyond the tile's rank are dropped and the tile may not
/// be longer than the layout. Each composition is checked like
/// [`Layout::composition`].
///
/// ```
/// use noodle_cute::algebra::ComposeTile;
/// use noodle_cute::coord::X;
/// use noodle_cute::{Int, IntTuple, Layout, Nil, shape, stride};
///
/// let layout = Layout::new(shape![Int::<8>, Int::<4>], stride![Int::<1>, Int::<8>])?;
/// let tile = IntTuple(Layout::new(Int::<4>, Int::<2>)?, IntTuple(X, Nil));
/// let picked = layout.compose_tile(tile)?;
/// assert_eq!(picked.shape, shape![Int::<4>, Int::<4>]);
/// assert_eq!(picked.stride, stride![Int::<2>, Int::<8>]);
/// # Ok::<(), noodle_cute::Error>(())
/// ```
pub trait ComposeTile<T> {
    /// The composed layout.
    type Output;
    /// Composes mode by mode.
    fn compose_tile(&self, tile: T) -> Result<Self::Output>;
}

impl<S, D> ComposeTile<Nil> for Layout<S, D> {
    type Output = Layout<Nil, Nil>;
    fn compose_tile(&self, _: Nil) -> Result<Layout<Nil, Nil>> {
        Ok(Layout {
            shape: Nil,
            stride: Nil,
        })
    }
}

type ModeShape<S, D, T> = <Layout<S, D> as ComposeMode<T>>::Shape;
type ModeStride<S, D, T> = <Layout<S, D> as ComposeMode<T>>::Stride;
type TailOut<S, D, T> = <Layout<S, D> as ComposeTile<T>>::Output;

impl<SH, ST, DH, DT, TH, TT> ComposeTile<IntTuple<TH, TT>>
    for Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>
where
    SH: Clone,
    DH: Clone,
    ST: Clone,
    DT: Clone,
    Layout<SH, DH>: ComposeMode<TH>,
    Layout<ST, DT>: ComposeTile<TT>,
    TailOut<ST, DT, TT>: LayoutParts,
{
    type Output = Layout<
        IntTuple<ModeShape<SH, DH, TH>, <TailOut<ST, DT, TT> as LayoutParts>::Shape>,
        IntTuple<ModeStride<SH, DH, TH>, <TailOut<ST, DT, TT> as LayoutParts>::Stride>,
    >;
    fn compose_tile(&self, tile: IntTuple<TH, TT>) -> Result<Self::Output> {
        let head = Layout {
            shape: self.shape.0.clone(),
            stride: self.stride.0.clone(),
        }
        .compose_mode(tile.0)?;
        let tail = Layout {
            shape: self.shape.1.clone(),
            stride: self.stride.1.clone(),
        }
        .compose_tile(tile.1)?
        .into_layout();
        Ok(Layout {
            shape: IntTuple(head.shape, tail.shape),
            stride: IntTuple(head.stride, tail.stride),
        })
    }
}

// layout_composed.hpp: composition(ComposedLayout, tiler) composes `layout_b`.
impl<A: Clone, O: Clone, B: ComposeTile<T>, T> ComposeTile<T> for ComposedLayout<A, O, B> {
    type Output = ComposedLayout<A, O, B::Output>;
    fn compose_tile(&self, tile: T) -> Result<Self::Output> {
        self.map_b(|b| b.compose_tile(tile))
    }
}

type Pair<A, B> = IntTuple<A, IntTuple<B, Nil>>;

/// CuTe `zip(layout)` for two modes of two modes: `((a, b), (c, d))` becomes
/// `((a, c), (b, d))`, transposing the hierarchy without changing the map.
pub trait Zip2 {
    /// The transposed layout.
    type Output;
    /// Transposes the two levels.
    fn zip2(self) -> Self::Output;
}

impl<A, B, C, E, DA, DB, DC, DE> Zip2
    for Layout<Pair<Pair<A, B>, Pair<C, E>>, Pair<Pair<DA, DB>, Pair<DC, DE>>>
{
    type Output = Layout<Pair<Pair<A, C>, Pair<B, E>>, Pair<Pair<DA, DC>, Pair<DB, DE>>>;
    fn zip2(self) -> Self::Output {
        let (s, d) = (self.shape, self.stride);
        Layout {
            shape: IntTuple(
                IntTuple(s.0.0, IntTuple(s.1.0.0, Nil)),
                IntTuple(IntTuple(s.0.1.0, IntTuple(s.1.0.1.0, Nil)), Nil),
            ),
            stride: IntTuple(
                IntTuple(d.0.0, IntTuple(d.1.0.0, Nil)),
                IntTuple(IntTuple(d.0.1.0, IntTuple(d.1.0.1.0, Nil)), Nil),
            ),
        }
    }
}

// layout_composed.hpp: zip(ComposedLayout) zips `layout_b`.
impl<A, O, B: Zip2> Zip2 for ComposedLayout<A, O, B> {
    type Output = ComposedLayout<A, O, B::Output>;
    fn zip2(self) -> Self::Output {
        ComposedLayout {
            layout_a: self.layout_a,
            offset: self.offset,
            layout_b: self.layout_b.zip2(),
        }
    }
}

/// CuTe `coalesce(layout, trg_profile)`: coalesces each mode named by a tuple
/// profile separately, and a whole (sub)layout where the profile is an integer.
///
/// Modes beyond the profile's rank are kept unchanged. Each coalesce is checked
/// like [`Layout::coalesce`]; as upstream, a (sub)layout that coalesces to one
/// mode becomes integral rather than a one-mode tuple.
pub trait CoalesceProfile<P> {
    /// The coalesced layout.
    type Output;
    /// Coalesces by `profile`.
    fn coalesce_profile(&self, profile: P) -> Result<Self::Output>;
}

impl<S: Clone, D: Clone> CoalesceProfile<Nil> for Layout<S, D> {
    type Output = Self;
    fn coalesce_profile(&self, _: Nil) -> Result<Self> {
        Ok(self.clone())
    }
}

impl<S, D, const N: isize> CoalesceProfile<Int<N>> for Layout<S, D>
where
    S: Shape + Flatten,
    D: StrideOf<S> + Flatten,
    Layout<S::Output, D::Output>: CoalesceFold,
    <Layout<S::Output, D::Output> as CoalesceFold>::Output: FinishCoalesce,
    CoalescedLayout<S, D>: UnwrapLayout,
{
    type Output = <CoalescedLayout<S, D> as UnwrapLayout>::Output;
    fn coalesce_profile(&self, _: Int<N>) -> Result<Self::Output> {
        Ok(self.coalesce()?.unwrap_layout())
    }
}

type ProfOut<S, D, P> = <Layout<S, D> as CoalesceProfile<P>>::Output;

impl<SH, ST, DH, DT, PH, PT> CoalesceProfile<IntTuple<PH, PT>>
    for Layout<IntTuple<SH, ST>, IntTuple<DH, DT>>
where
    SH: Clone,
    DH: Clone,
    ST: Clone,
    DT: Clone,
    Layout<SH, DH>: CoalesceProfile<PH>,
    Layout<ST, DT>: CoalesceProfile<PT>,
    ProfOut<SH, DH, PH>: LayoutParts,
    ProfOut<ST, DT, PT>: LayoutParts,
{
    type Output = Layout<
        IntTuple<
            <ProfOut<SH, DH, PH> as LayoutParts>::Shape,
            <ProfOut<ST, DT, PT> as LayoutParts>::Shape,
        >,
        IntTuple<
            <ProfOut<SH, DH, PH> as LayoutParts>::Stride,
            <ProfOut<ST, DT, PT> as LayoutParts>::Stride,
        >,
    >;
    fn coalesce_profile(&self, profile: IntTuple<PH, PT>) -> Result<Self::Output> {
        let head = Layout {
            shape: self.shape.0.clone(),
            stride: self.stride.0.clone(),
        }
        .coalesce_profile(profile.0)?
        .into_layout();
        let tail = Layout {
            shape: self.shape.1.clone(),
            stride: self.stride.1.clone(),
        }
        .coalesce_profile(profile.1)?
        .into_layout();
        Ok(Layout {
            shape: IntTuple(head.shape, tail.shape),
            stride: IntTuple(head.stride, tail.stride),
        })
    }
}

// layout_composed.hpp: coalesce(ComposedLayout, trg_profile) coalesces `layout_b`.
impl<A: Clone, O: Clone, B: CoalesceProfile<P>, P> CoalesceProfile<P> for ComposedLayout<A, O, B> {
    type Output = ComposedLayout<A, O, B::Output>;
    fn coalesce_profile(&self, profile: P) -> Result<Self::Output> {
        self.map_b(|b| b.coalesce_profile(profile))
    }
}
