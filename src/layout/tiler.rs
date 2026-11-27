//! Layout concatenation, tiler conversion, and codomain queries.
//!
//! # Main Interfaces
//!
//! - [`Concatenation`] / [`MakeLayout`] / [`make_layout`]: Transposes a tuple of [`Layout`]s
//!   into a single [`Layout`] of tuples.
//! - [`Tiler`] / [`tiler_to_layout`] / [`tiler_to_layout_with`]: Converts a tiler (extent,
//!   [`Layout`], or nested tuple of tilers) into a coordinate-strided [`Layout`].
//! - [`coshape`] / [`Coprofile`] / [`coprofile`]: Queries the codomain shape and structural
//!   profile of a [`Layout`] or [`Tiler`].
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::{Layout, tiler_to_layout};
//! use noodle_cute::{e, int_tuple, tuple};
//!
//! assert_eq!(
//!     tiler_to_layout(int_tuple![Int::<4>, Int::<5>]),
//!     Layout::new(int_tuple![Int::<4>, Int::<5>], tuple![e![0], e![1]])
//! );
//! ```
//!
//! # Notes
//!
//! - An integer extent `N` maps to `Layout<N, E>`; a [`Layout`] leaves its stride unchanged when
//!   scaled by static `Int<1>` and otherwise scales via [`ScaleStride`].
//! - A nested tiler tuple pairs each leaf with a congruent unit basis from
//!   [`make_basis_like`] and combines child layouts via [`Concatenation`].
//! - [`coprofile`] yields `Int<0>` for scalar extents and `Nil` for `Nil`.

use crate::basis::{
    BasisCombination, BasisTerm, BasisTuple, CombAsTuple, MakeBasisLikeOutput,
    MakeBasisLikeTopProfile, make_basis_like,
};
use crate::f2::F2;
use crate::integer::{Int, int};
use crate::shape::{Shape, ShapeScalar};
use crate::stride::{Coshape, ScaleStride, Stride, StrideSkeleton};
use crate::tuple::{
    ApplyLeafFunctor, Item, MakeNodeFunctor, Nil, Til, TransformApplyLeaf, Tuple,
    transform_apply_leaf,
};

use super::Layout;

/// Transposes a tuple of [`Layout`]s into a single [`Layout`] of tuples.
pub const trait Concatenation: Copy {
    /// Combined shape tuple type.
    type Shape: Shape;
    /// Combined stride tuple type.
    type Stride: Stride<Self::Shape>;

    /// Concatenates each layout in `self` as one top-level mode.
    fn concat(self) -> Layout<Self::Shape, Self::Stride>;
}

const impl Concatenation for Nil {
    type Shape = Nil;
    type Stride = Nil;

    fn concat(self) -> Layout<Self::Shape, Self::Stride> {
        Layout {
            shape: Nil,
            stride: Nil,
        }
    }
}

const impl<S, D, T: [const] Concatenation> Concatenation for Til<Layout<S, D>, T>
where
    S: Shape,
    D: Stride<S>,
    T::Shape: Tuple + Shape,
    T::Stride: Tuple + Stride<T::Shape>,
{
    type Shape = Til<S, T::Shape>;
    type Stride = Til<D, T::Stride>;

    fn concat(self) -> Layout<Self::Shape, Self::Stride> {
        let tail = self.1.concat();
        Layout {
            shape: Til(self.0.shape, tail.shape),
            stride: Til(self.0.stride, tail.stride),
        }
    }
}

/// Concatenates a tuple of [`Layout`]s so each input becomes one top-level mode.
pub const fn make_layout<T: [const] Concatenation>(layouts: T) -> Layout<T::Shape, T::Stride> {
    layouts.concat()
}

/// Node builder functor for [`transform_apply_leaf`] and [`lift`](crate::tuple::lift)
/// that combines child layouts via [`Concatenation`].
#[derive(Clone, Copy, Default, Debug)]
pub struct MakeLayout;

const impl<Children: [const] Concatenation> MakeNodeFunctor<Children> for MakeLayout {
    type Output = Layout<Children::Shape, Children::Stride>;

    fn make(&self, children: Children) -> Self::Output {
        children.concat()
    }
}

/// Marks tiler scaling factors and identifies exact compile-time multiplicative identity
/// (`IS_UNIT = true` only for `Int<1>`).
#[doc(hidden)]
pub const trait TilerFactor: Copy {
    const IS_UNIT: bool = false;
}

const impl TilerFactor for int {}
const impl<const N: int> TilerFactor for Int<N> {
    const IS_UNIT: bool = N == 1;
}
// Scaling by F2(1) lifts integer strides into the F2 algebra.
const impl<V: Copy> TilerFactor for F2<V> {
    const IS_UNIT: bool = false;
}
const impl<V: Copy, P: Copy> TilerFactor for BasisTerm<V, P> {}
const impl<T: BasisTuple> TilerFactor for BasisCombination<T> {}

/// Scales layout `Self` by factor `E` when `KEEP` is `false`, or returns `Self` unchanged when
/// `KEEP` is `true` (static multiplicative identity `Int<1>`).
#[doc(hidden)]
pub const trait ScaleIfNotOne<E: Copy, const KEEP: bool>: Copy {
    type Output: Copy;
    fn scale_if_not_one(self, factor: E) -> Self::Output;
}

const impl<L: Copy, E: Copy> ScaleIfNotOne<E, true> for L {
    type Output = L;
    fn scale_if_not_one(self, _: E) -> L {
        self
    }
}

const impl<S, D, E> ScaleIfNotOne<E, false> for Layout<S, D>
where
    S: Shape,
    D: Stride<S> + [const] ScaleStride<E>,
    D::Output: Stride<S>,
    E: Copy,
{
    type Output = Layout<S, D::Output>;
    fn scale_if_not_one(self, factor: E) -> Self::Output {
        Layout::new(self.shape, self.stride.scale_stride(factor))
    }
}

/// Converts a tiler (an integer extent, a [`Layout`], or a nested tuple of tilers)
/// into a coordinate-strided [`Layout`] scaled by `E`.
pub const trait Tiler<E = Int<1>>: Copy {
    /// Resulting [`Layout`] type.
    type OutputLayout;

    /// Converts `self` into a [`Layout`] using basis or scaling factor `e`.
    fn to_layout(self, e: E) -> Self::OutputLayout;
}

const impl<E: Copy> Tiler<E> for Nil {
    type OutputLayout = Layout<Nil, Nil>;

    fn to_layout(self, _: E) -> Self::OutputLayout {
        Layout::new(Nil, Nil)
    }
}

const impl<N, E> Tiler<E> for N
where
    N: ShapeScalar,
    E: Stride<N>,
{
    type OutputLayout = Layout<N, E>;

    fn to_layout(self, e: E) -> Self::OutputLayout {
        Layout::new(self, e)
    }
}

#[doc(hidden)]
pub const TILER_FACTOR_IS_UNIT<E: TilerFactor>: bool = E::IS_UNIT;

const impl<S, D, E> Tiler<E> for Layout<S, D>
where
    S: Shape,
    D: Stride<S>,
    E: [const] TilerFactor,
    Self: [const] ScaleIfNotOne<E, { TILER_FACTOR_IS_UNIT::<E> }>,
{
    type OutputLayout = <Self as ScaleIfNotOne<E, { TILER_FACTOR_IS_UNIT::<E> }>>::Output;

    fn to_layout(self, e: E) -> Self::OutputLayout {
        self.scale_if_not_one(e)
    }
}

/// Leaf functor for [`transform_apply_leaf`] that converts each `(tiler_leaf, basis)` pair via
/// [`Tiler::to_layout`].
#[doc(hidden)]
#[derive(Clone, Copy, Default, Debug)]
pub struct TilerLeafFunctor;

const impl<L, B> ApplyLeafFunctor<(L, B)> for TilerLeafFunctor
where
    L: [const] Tiler<B>,
    B: Copy,
{
    type Output = L::OutputLayout;

    fn apply(&self, args: (L, B)) -> Self::Output {
        args.0.to_layout(args.1)
    }
}

const impl<H, T, E> Tiler<E> for Til<H, T>
where
    H: Item,
    T: Tuple,
    Self: [const] TilerTupleBases
        + [const] TransformApplyLeaf<
            MakeLayout,
            TilerLeafFunctor,
            (<Self as TilerTupleBases>::Bases,),
        >,
    E: [const] TilerFactor,
    <Self as TransformApplyLeaf<
        MakeLayout,
        TilerLeafFunctor,
        (<Self as TilerTupleBases>::Bases,),
    >>::Output: [const] ScaleIfNotOne<E, { TILER_FACTOR_IS_UNIT::<E> }>,
{
    type OutputLayout = <<Self as TransformApplyLeaf<
        MakeLayout,
        TilerLeafFunctor,
        (<Self as TilerTupleBases>::Bases,),
    >>::Output as ScaleIfNotOne<E, { TILER_FACTOR_IS_UNIT::<E> }>>::Output;

    fn to_layout(self, e: E) -> Self::OutputLayout {
        let bases = <Self as TilerTupleBases>::make_bases(self);
        let base_layout = transform_apply_leaf(&self, &MakeLayout, &TilerLeafFunctor, (bases,));
        base_layout.scale_if_not_one(e)
    }
}

/// Generates a congruent tuple of unit basis vectors (`Bases`) matching the profile of tiler
/// tuple `Self`.
#[doc(hidden)]
pub const trait TilerTupleBases: Item {
    type Bases: Copy;
    fn make_bases(self) -> Self::Bases;
}

const impl<T: Item> TilerTupleBases for T
where
    T::Profile: [const] MakeBasisLikeTopProfile,
{
    type Bases = MakeBasisLikeOutput<T>;

    fn make_bases(self) -> Self::Bases {
        make_basis_like(self)
    }
}

/// Converts a tiler into its equivalent [`Layout`] with default factor `Int<1>`.
///
/// ```
/// # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::integer::Int;
/// use noodle_cute::layout::{Layout, tiler_to_layout};
/// use noodle_cute::{e, int_tuple, tuple};
///
/// assert_eq!(
///     tiler_to_layout(int_tuple![Int::<4>, Int::<5>]),
///     Layout::new(int_tuple![Int::<4>, Int::<5>], tuple![e![0], e![1]])
/// );
/// ```
pub const fn tiler_to_layout<T: [const] Tiler>(tiler: T) -> T::OutputLayout {
    tiler.to_layout(Int::<1>)
}

/// Converts a tiler into its equivalent [`Layout`] scaled by `e`.
pub const fn tiler_to_layout_with<T: [const] Tiler<E>, E>(tiler: T, e: E) -> T::OutputLayout {
    tiler.to_layout(e)
}

/// Computes the codomain shape bounding all offsets of `layout`.
pub const fn coshape<S: [const] Coshape<D>, D: Stride<S>>(
    layout: &Layout<S, D>,
) -> <S as Coshape<D>>::Output {
    layout.coshape()
}

/// Computes the structural profile of the codomain as an integer tuple congruent to [`coshape`].
pub const trait Coprofile: Copy {
    /// Representative shape/profile value congruent to the codomain.
    type Output: Item;

    /// Returns a value with the codomain's hierarchical profile.
    fn coprofile(&self) -> Self::Output;
}

const impl<S: Copy, D> Coprofile for Layout<S, D>
where
    D: [const] StrideSkeleton,
    <D as StrideSkeleton>::Terms: [const] CombAsTuple,
{
    type Output = <<D as StrideSkeleton>::Terms as CombAsTuple>::Output;

    fn coprofile(&self) -> Self::Output {
        self.stride.skeleton_terms().comb_as_tuple()
    }
}

const impl Coprofile for Nil {
    type Output = Nil;

    fn coprofile(&self) -> Nil {
        Nil
    }
}

const impl Coprofile for int {
    type Output = Int<0>;
    fn coprofile(&self) -> Int<0> {
        Int
    }
}

const impl<const N: int> Coprofile for Int<N> {
    type Output = Int<0>;
    fn coprofile(&self) -> Int<0> {
        Int
    }
}

const impl<H: Copy, T: Copy> Coprofile for Til<H, T>
where
    Self: [const] Tiler,
    <Self as Tiler>::OutputLayout: [const] Coprofile,
{
    type Output = <<Self as Tiler>::OutputLayout as Coprofile>::Output;

    fn coprofile(&self) -> Self::Output {
        tiler_to_layout(*self).coprofile()
    }
}

/// Returns the codomain structural profile of a [`Layout`] or [`Tiler`].
pub const fn coprofile<T: [const] Coprofile>(x: &T) -> T::Output {
    x.coprofile()
}
