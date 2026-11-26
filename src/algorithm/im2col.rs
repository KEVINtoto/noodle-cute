//! The `im2col` layout transformation for N-D convolution.
//!
//! Turns an N-D activation tensor of shape `(N, [D,] [H,] [W,] C)` into a rank-2 logical
//! `(M, K) = ((N, (Z, P, Q)), ((T, R, S), C))` view that an unmodified GEMM can contract
//! against a folded stencil `(K, ((T, R, S), C))` into `((N, (Z, P, Q)), K)`:
//!
//! 1. Split the activation layout `A` into batch `A[0]` (`N`), spatial modes `A[1:-1]`
//!    (`rank >= 1`), and channel `A[-1]` (`C`).
//! 2. Compose the spatial sublayout twice: with position tilers `(Z_i : stride_dhw_i)` and
//!    with tap tilers `(T_i : stride_trs_i)`.
//! 3. Shift the accessor origin by `A((0, lower_trs - lower_dhw, 0))` so every spatial
//!    coordinate evaluates to `stride_dhw * z + stride_trs * t + lower_trs - lower_dhw`.
//!
//! # Main Interfaces
//!
//! - [`Im2ColOptions`] and [`DefaultParam`]: Compile-time-typed builder for optional
//!   convolution parameters (`lower_dhw`, `upper_dhw`, `stride_dhw`, `lower_trs`,
//!   `stride_trs`, `shape_zpq`).
//! - [`im2col_layout`] / [`im2col_layout_with`] ([`Im2ColLayout`]): Core layout + origin
//!   primitive shared by all three tensor views.
//! - [`im2col`] / [`im2col_with`] ([`Im2Col`]): Offset-codomain view sharing the
//!   activation's storage (exact when no window leaves the activation).
//! - [`im2col_coord`] / [`im2col_coord_with`] ([`Im2ColCoord`]): Coordinate-codomain view
//!   producing activation coordinates as basis values (one unit basis per top-level mode).
//! - [`im2col_padded`] / [`im2col_padded_with`] ([`Im2ColPadded`]): Read-only padded view
//!   combining [`im2col_coord`] with [`OutOfBoundsAccessor`]
//!   so out-of-bounds reads return `fill` with zero padded allocation.
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algorithm::im2col::{
//!     Im2ColOptions, im2col, im2col_coord_with, im2col_padded_with,
//! };
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::shape::{idx2crd, in_bounds};
//! use noodle_cute::tensor::make_tensor;
//! use noodle_cute::int_tuple;
//!
//! let mut buf = [0i32; 16];
//! for (i, slot) in buf.iter_mut().enumerate() {
//!     *slot = (i + 1) as i32;
//! }
//! let act = make_tensor(
//!     &buf,
//!     Layout::new(
//!         int_tuple![Int::<1>, Int::<4>, Int::<4>, Int::<1>],
//!         int_tuple![Int::<16>, Int::<4>, Int::<1>, Int::<1>],
//!     ),
//! );
//!
//! let view = im2col(act, int_tuple![Int::<2>, Int::<2>]);
//! assert_eq!(
//!     view.layout.shape,
//!     int_tuple![(Int::<1>, (Int::<3>, Int::<3>)), ((Int::<2>, Int::<2>), Int::<1>)]
//! );
//!
//! let opts = Im2ColOptions::new().lower_dhw(int_tuple![Int::<1>, Int::<1>]);
//! let crd_view = im2col_coord_with(act, int_tuple![Int::<3>, Int::<3>], opts);
//! let corner = crd_view.at(int_tuple![(0i32, (0i32, 0i32)), ((0i32, 0i32), 0i32)]);
//! assert_eq!(idx2crd(corner, &act.layout.shape), int_tuple![0i32, -1i32, -1i32, 0i32]);
//! assert!(!in_bounds(corner, &act.layout.shape));
//!
//! let padded = im2col_padded_with(act, int_tuple![Int::<3>, Int::<3>], -1i32, opts);
//! assert_eq!(padded.at(int_tuple![(0i32, (0i32, 0i32)), ((0i32, 0i32), 0i32)]), -1);
//! assert_eq!(padded.at(int_tuple![(0i32, (1i32, 1i32)), ((1i32, 1i32), 0i32)]), act.at(int_tuple![0i32, 1i32, 1i32, 0i32]));
//! ```
//!
//! Invalid activation rank, parameter rank mismatches (including attempting to broadcast a
//! scalar across multiple spatial modes), and static stencils that do not fit fail at
//! compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algorithm::im2col::im2col;
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::tensor::make_tensor;
//!
//! let buf = [0i32; 16];
//! let act = make_tensor(&buf, Layout::from_shape(int_tuple![Int::<4>, Int::<4>]));
//! let _ = im2col(act, Int::<2>);
//! ```
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algorithm::im2col::{Im2ColOptions, im2col_with};
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::tensor::make_tensor;
//!
//! let buf = [0i32; 16];
//! let act = make_tensor(
//!     &buf,
//!     Layout::new(
//!         int_tuple![Int::<1>, Int::<4>, Int::<4>, Int::<1>],
//!         int_tuple![Int::<16>, Int::<4>, Int::<1>, Int::<1>],
//!     ),
//! );
//! let _ = im2col_with(act, int_tuple![Int::<2>, Int::<2>], Im2ColOptions::new().stride_dhw(Int::<2>));
//! ```
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algorithm::im2col::im2col;
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::tensor::make_tensor;
//!
//! let buf = [0i32; 9];
//! let act = make_tensor(
//!     &buf,
//!     Layout::new(
//!         int_tuple![Int::<1>, Int::<3>, Int::<3>, Int::<1>],
//!         int_tuple![Int::<9>, Int::<3>, Int::<1>, Int::<1>],
//!     ),
//! );
//! let _ = im2col(act, int_tuple![Int::<5>, Int::<5>]);
//! ```
//!
//! # Notes
//!
//! - Every spatial parameter carries one entry per spatial mode (`rank(A) - 2`): scalars are
//!   admissible only for 1-D spatial activations and are never broadcast across multiple modes.
//! - Default options use zero padding (`lower_dhw = 0`), symmetric upper padding
//!   (`upper_dhw = lower_dhw`), unit traversal stride (`stride_dhw = 1`), zero stencil origin
//!   (`lower_trs = 0`), unit dilation (`stride_trs = 1`), and fprop output extents
//!   `1 + floor_div(size(D_i) + lo_i + hi_i - ((T_i - 1) * s_i + 1), u_i)`.
//! - `lower_trs`, negative `stride_trs`, and explicit `shape_zpq` support reversed
//!   stencils such as `dgrad`.
//! - Static invalid extents (`<= 0`) fail at compile time; dynamic invalid extents panic at
//!   runtime.

use crate::algebra::composition::Composition;
use crate::basis::{MakeTopBasisLikeOutput, MakeTopBasisLikeProfile, make_top_basis_like};
use crate::integer::{
    ArithAdd, ArithFloorDiv, ArithMul, ArithSub, Difference, FloorQuotient, Int, Integer, Product,
    Sum, add, floor_div, int, mul, sub,
};
use crate::layout::{Layout, make_layout};
use crate::shape::{IndexToCoord, Shape, ShapeScalar};
use crate::stride::{InnerProduct, PrefixProduct, Stride, StrideScalar};
use crate::tensor::{AccessorBase, AccessorOffset, ImplicitAccessor, OutOfBoundsAccessor, Tensor};
use crate::tuple::{Item, Nil, Til, Tuple, TupleProduct, Wrap};

// ------------------------------------------------------------------
// Options builder
// ------------------------------------------------------------------

/// Marker type representing an omitted optional parameter in [`Im2ColOptions`].
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct DefaultParam;

/// Optional convolution parameters for [`im2col_layout_with`], [`im2col_with`],
/// [`im2col_coord_with`], and [`im2col_padded_with`].
///
/// Every parameter uses activation spatial order (`D, H, W`):
/// - `lower_dhw`: padding before each spatial mode (default: `Int<0>` per mode).
/// - `upper_dhw`: padding after each spatial mode (default: equals `lower_dhw`).
/// - `stride_dhw`: output traversal stride (default: `Int<1>` per mode).
/// - `lower_trs`: stencil origin offset (default: `Int<0>` per mode).
/// - `stride_trs`: stencil dilation, which may be negative for `dgrad` (default: `Int<1>` per mode).
/// - `shape_zpq`: explicit output spatial extents overriding the fprop formula (default: derived).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Im2ColOptions<
    LowerDhw = DefaultParam,
    UpperDhw = DefaultParam,
    StrideDhw = DefaultParam,
    LowerTrs = DefaultParam,
    StrideTrs = DefaultParam,
    ShapeZpq = DefaultParam,
> {
    /// Padding before each spatial mode.
    pub lower_dhw: LowerDhw,
    /// Padding after each spatial mode.
    pub upper_dhw: UpperDhw,
    /// Traversal stride along each spatial mode.
    pub stride_dhw: StrideDhw,
    /// Stencil origin offset along each spatial mode.
    pub lower_trs: LowerTrs,
    /// Dilation stride along each stencil mode.
    pub stride_trs: StrideTrs,
    /// Explicit output spatial extents `(Z, P, Q)`.
    pub shape_zpq: ShapeZpq,
}

impl Im2ColOptions {
    /// Creates an [`Im2ColOptions`] with all optional parameters defaulted.
    pub const fn new() -> Self {
        Self {
            lower_dhw: DefaultParam,
            upper_dhw: DefaultParam,
            stride_dhw: DefaultParam,
            lower_trs: DefaultParam,
            stride_trs: DefaultParam,
            shape_zpq: DefaultParam,
        }
    }
}

impl<LoD: Copy, HiD: Copy, StD: Copy, LoT: Copy, StT: Copy, Zpq: Copy>
    Im2ColOptions<LoD, HiD, StD, LoT, StT, Zpq>
{
    /// Sets lower padding before each spatial mode.
    pub const fn lower_dhw<New>(
        self,
        lower_dhw: New,
    ) -> Im2ColOptions<New, HiD, StD, LoT, StT, Zpq> {
        Im2ColOptions {
            lower_dhw,
            upper_dhw: self.upper_dhw,
            stride_dhw: self.stride_dhw,
            lower_trs: self.lower_trs,
            stride_trs: self.stride_trs,
            shape_zpq: self.shape_zpq,
        }
    }

    /// Sets upper padding after each spatial mode (overrides symmetric padding).
    pub const fn upper_dhw<New>(
        self,
        upper_dhw: New,
    ) -> Im2ColOptions<LoD, New, StD, LoT, StT, Zpq> {
        Im2ColOptions {
            lower_dhw: self.lower_dhw,
            upper_dhw,
            stride_dhw: self.stride_dhw,
            lower_trs: self.lower_trs,
            stride_trs: self.stride_trs,
            shape_zpq: self.shape_zpq,
        }
    }

    /// Sets traversal stride along each spatial mode.
    pub const fn stride_dhw<New>(
        self,
        stride_dhw: New,
    ) -> Im2ColOptions<LoD, HiD, New, LoT, StT, Zpq> {
        Im2ColOptions {
            lower_dhw: self.lower_dhw,
            upper_dhw: self.upper_dhw,
            stride_dhw,
            lower_trs: self.lower_trs,
            stride_trs: self.stride_trs,
            shape_zpq: self.shape_zpq,
        }
    }

    /// Sets stencil origin offset along each spatial mode.
    pub const fn lower_trs<New>(
        self,
        lower_trs: New,
    ) -> Im2ColOptions<LoD, HiD, StD, New, StT, Zpq> {
        Im2ColOptions {
            lower_dhw: self.lower_dhw,
            upper_dhw: self.upper_dhw,
            stride_dhw: self.stride_dhw,
            lower_trs,
            stride_trs: self.stride_trs,
            shape_zpq: self.shape_zpq,
        }
    }

    /// Sets dilation stride along each stencil mode (negative values reverse the stencil).
    pub const fn stride_trs<New>(
        self,
        stride_trs: New,
    ) -> Im2ColOptions<LoD, HiD, StD, LoT, New, Zpq> {
        Im2ColOptions {
            lower_dhw: self.lower_dhw,
            upper_dhw: self.upper_dhw,
            stride_dhw: self.stride_dhw,
            lower_trs: self.lower_trs,
            stride_trs,
            shape_zpq: self.shape_zpq,
        }
    }

    /// Sets explicit output spatial extents `(Z, P, Q)`, overriding the fprop formula.
    pub const fn shape_zpq<New>(
        self,
        shape_zpq: New,
    ) -> Im2ColOptions<LoD, HiD, StD, LoT, StT, New> {
        Im2ColOptions {
            lower_dhw: self.lower_dhw,
            upper_dhw: self.upper_dhw,
            stride_dhw: self.stride_dhw,
            lower_trs: self.lower_trs,
            stride_trs: self.stride_trs,
            shape_zpq,
        }
    }
}

// ------------------------------------------------------------------
// Activation layout decomposition: (N, D0, ..., D_{k-1}, C) with k >= 1
// ------------------------------------------------------------------

/// Pair of `(spatial_layout, channel_layout)` extracted by [`SplitSpatialTail`].
#[doc(hidden)]
pub type SpatialTailPair<SpS, SpD, CS, CD> = (Layout<SpS, SpD>, Layout<CS, CD>);

/// Splits the post-`N` tail `(D0, ..., D_{k-1}, C)` (`k >= 1`) of an activation layout
/// into the non-empty spatial layout `(D0, ..., D_{k-1})` and the trailing channel layout `C`.
#[doc(hidden)]
pub const trait SplitSpatialTail: Copy {
    type SpatialShape: Tuple + Shape;
    type SpatialStride: Tuple + Stride<Self::SpatialShape>;
    type CShape: Shape;
    type CStride: Stride<Self::CShape>;

    fn split_spatial_tail(
        self,
    ) -> SpatialTailPair<Self::SpatialShape, Self::SpatialStride, Self::CShape, Self::CStride>;
}

const impl<SD0, SC, DD0, DC> SplitSpatialTail
    for Layout<Til<SD0, Til<SC, Nil>>, Til<DD0, Til<DC, Nil>>>
where
    SD0: Shape,
    SC: Shape,
    DD0: Stride<SD0>,
    DC: Stride<SC>,
{
    type SpatialShape = Til<SD0, Nil>;
    type SpatialStride = Til<DD0, Nil>;
    type CShape = SC;
    type CStride = DC;

    fn split_spatial_tail(
        self,
    ) -> SpatialTailPair<Self::SpatialShape, Self::SpatialStride, Self::CShape, Self::CStride> {
        (
            Layout::new(Til(self.shape.0, Nil), Til(self.stride.0, Nil)),
            Layout::new(self.shape.1.0, self.stride.1.0),
        )
    }
}

const impl<SD0, SD1, S2, SRest, DD0, DD1, D2, DRest> SplitSpatialTail
    for Layout<Til<SD0, Til<SD1, Til<S2, SRest>>>, Til<DD0, Til<DD1, Til<D2, DRest>>>>
where
    SD0: Shape,
    SD1: Shape,
    S2: Shape,
    SRest: Tuple + Shape,
    DD0: Stride<SD0>,
    DD1: Stride<SD1>,
    D2: Stride<S2>,
    DRest: Tuple + Stride<SRest>,
    Layout<Til<SD1, Til<S2, SRest>>, Til<DD1, Til<D2, DRest>>>: [const] SplitSpatialTail,
{
    type SpatialShape = Til<
        SD0,
        <Layout<Til<SD1, Til<S2, SRest>>, Til<DD1, Til<D2, DRest>>> as SplitSpatialTail>::SpatialShape,
    >;
    type SpatialStride = Til<
        DD0,
        <Layout<Til<SD1, Til<S2, SRest>>, Til<DD1, Til<D2, DRest>>> as SplitSpatialTail>::SpatialStride,
    >;
    type CShape =
        <Layout<Til<SD1, Til<S2, SRest>>, Til<DD1, Til<D2, DRest>>> as SplitSpatialTail>::CShape;
    type CStride =
        <Layout<Til<SD1, Til<S2, SRest>>, Til<DD1, Til<D2, DRest>>> as SplitSpatialTail>::CStride;

    fn split_spatial_tail(
        self,
    ) -> SpatialTailPair<Self::SpatialShape, Self::SpatialStride, Self::CShape, Self::CStride> {
        let (tail_spatial, c_layout) =
            Layout::new(self.shape.1, self.stride.1).split_spatial_tail();
        (
            Layout::new(
                Til(self.shape.0, tail_spatial.shape),
                Til(self.stride.0, tail_spatial.stride),
            ),
            c_layout,
        )
    }
}

/// Triple of `(batch_layout, spatial_layout, channel_layout)` extracted by [`SplitActivationLayout`].
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct ActivationSplit<N, Sp, C>(pub N, pub Sp, pub C);

/// Concrete [`ActivationSplit`] return type for [`SplitActivationLayout`].
#[doc(hidden)]
pub type ActivationParts<T> = ActivationSplit<
    Layout<<T as SplitActivationLayout>::NShape, <T as SplitActivationLayout>::NStride>,
    Layout<<T as SplitActivationLayout>::SpatialShape, <T as SplitActivationLayout>::SpatialStride>,
    Layout<<T as SplitActivationLayout>::CShape, <T as SplitActivationLayout>::CStride>,
>;

/// Splits a rank `>= 3` activation layout `(N, D0, ..., D_{k-1}, C)` into `(N, A_dhw, C)`.
#[doc(hidden)]
pub const trait SplitActivationLayout: Copy {
    type NShape: Shape;
    type NStride: Stride<Self::NShape>;
    type SpatialShape: Tuple + Shape;
    type SpatialStride: Tuple + Stride<Self::SpatialShape>;
    type CShape: Shape;
    type CStride: Stride<Self::CShape>;

    fn split_activation(self) -> ActivationParts<Self>;
}

const impl<SN, ST, DN, DT> SplitActivationLayout for Layout<Til<SN, ST>, Til<DN, DT>>
where
    SN: Shape,
    ST: Tuple + Shape,
    DN: Stride<SN>,
    DT: Tuple + Stride<ST>,
    Layout<ST, DT>: [const] SplitSpatialTail,
{
    type NShape = SN;
    type NStride = DN;
    type SpatialShape = <Layout<ST, DT> as SplitSpatialTail>::SpatialShape;
    type SpatialStride = <Layout<ST, DT> as SplitSpatialTail>::SpatialStride;
    type CShape = <Layout<ST, DT> as SplitSpatialTail>::CShape;
    type CStride = <Layout<ST, DT> as SplitSpatialTail>::CStride;

    fn split_activation(self) -> ActivationParts<Self> {
        let (spatial, c_layout) = Layout::new(self.shape.1, self.stride.1).split_spatial_tail();
        ActivationSplit(Layout::new(self.shape.0, self.stride.0), spatial, c_layout)
    }
}

// ------------------------------------------------------------------
// Spatial parameter validation, defaults, and shape_zpq derivation
// ------------------------------------------------------------------

/// Builds a flat tuple repeating `V` once per top-level mode of spatial shape tuple `Self`.
#[doc(hidden)]
pub const trait FillSpatial<V: [const] Item>: Copy {
    type Output: [const] Tuple;
    fn fill_spatial(value: V) -> Self::Output;
}

const impl<V: [const] Item> FillSpatial<V> for Nil {
    type Output = Nil;
    fn fill_spatial(_: V) -> Nil {
        Nil
    }
}

const impl<H: Copy, T: [const] FillSpatial<V>, V: [const] Item> FillSpatial<V> for Til<H, T> {
    type Output = Til<V, T::Output>;
    fn fill_spatial(value: V) -> Self::Output {
        Til(value, T::fill_spatial(value))
    }
}

/// Validates that `Self` is a flat integer tuple with the exact top-level rank of `SpS`.
#[doc(hidden)]
pub const trait CheckSpatialInts<SpS>: Tuple {}

const impl CheckSpatialInts<Nil> for Nil {}

const impl<I: [const] Integer, T, SH, ST> CheckSpatialInts<Til<SH, ST>> for Til<I, T> where
    T: [const] CheckSpatialInts<ST>
{
}

/// Validates that `Self` is a flat [`ShapeScalar`] tuple with the exact top-level rank of `SpS`.
#[doc(hidden)]
pub const trait CheckSpatialShapes<SpS>: Tuple + Shape {}

const impl CheckSpatialShapes<Nil> for Nil {}

const impl<S: [const] ShapeScalar, T, SH, ST> CheckSpatialShapes<Til<SH, ST>> for Til<S, T> where
    T: [const] CheckSpatialShapes<ST>
{
}

const IS_POSITIVE<const N: int>: bool = N > 0;

/// Compile-time guard requiring a positive output extent (`POSITIVE == true`).
#[doc(hidden)]
pub const trait ValidPositiveExtent<const POSITIVE: bool> {}

const impl ValidPositiveExtent<true> for () {}

/// Ensures an extent is strictly positive (compile-time error for static `Int<N>` with `N <= 0`;
/// runtime panic for dynamic `int <= 0`).
#[doc(hidden)]
pub const trait EnsurePositive: [const] ShapeScalar {
    fn ensure_positive(self) -> Self;
}

const impl<const N: int> EnsurePositive for Int<N>
where
    (): [const] ValidPositiveExtent<{ IS_POSITIVE::<N> }>,
    Self: [const] ShapeScalar,
{
    fn ensure_positive(self) -> Self {
        self
    }
}

const impl EnsurePositive for int {
    fn ensure_positive(self) -> Self {
        assert!(
            self > 0,
            "im2col: stencil does not fit (output extent must be positive)"
        );
        self
    }
}

/// Validates that every mode in an explicit `shape_zpq` tuple matches `SpS`'s rank and is positive.
#[doc(hidden)]
pub const trait CheckSpatialPositive<SpS>: Tuple + Shape {
    fn check_spatial_positive(self) -> Self;
}

const impl CheckSpatialPositive<Nil> for Nil {
    fn check_spatial_positive(self) -> Self {
        Nil
    }
}

const impl<Z: [const] EnsurePositive, T, SH, ST> CheckSpatialPositive<Til<SH, ST>> for Til<Z, T>
where
    T: [const] CheckSpatialPositive<ST>,
{
    fn check_spatial_positive(self) -> Self {
        Til(self.0.ensure_positive(), self.1.check_spatial_positive())
    }
}

/// Resolves an optional spatial integer parameter (defaulting each mode to `Def`, or wrapping
/// and exact-rank-checking a caller-supplied value).
#[doc(hidden)]
pub const trait ResolveSpatialParam<SpS, Def: [const] Item>: Copy {
    type Output: Tuple;
    fn resolve_spatial(self, default_val: Def) -> Self::Output;
}

const impl<SpS: [const] FillSpatial<Def>, Def: [const] Item> ResolveSpatialParam<SpS, Def>
    for DefaultParam
{
    type Output = SpS::Output;
    fn resolve_spatial(self, default_val: Def) -> Self::Output {
        SpS::fill_spatial(default_val)
    }
}

const impl<P, SpS, Def> ResolveSpatialParam<SpS, Def> for P
where
    Def: [const] Item,
    P: [const] Wrap,
    P::Wrapped: [const] CheckSpatialInts<SpS>,
{
    type Output = P::Wrapped;
    fn resolve_spatial(self, _: Def) -> Self::Output {
        self.wrap()
    }
}

/// Resolves `upper_dhw`: defaults to `lower_dhw` (symmetric padding) when omitted.
#[doc(hidden)]
pub const trait ResolveUpperDhw<SpS, LoD: Tuple>: Copy {
    type Output: Tuple;
    fn resolve_upper_dhw(self, lower_dhw: LoD) -> Self::Output;
}

const impl<SpS, LoD: Tuple> ResolveUpperDhw<SpS, LoD> for DefaultParam {
    type Output = LoD;
    fn resolve_upper_dhw(self, lower_dhw: LoD) -> Self::Output {
        lower_dhw
    }
}

const impl<P, SpS, LoD> ResolveUpperDhw<SpS, LoD> for P
where
    LoD: Tuple,
    P: [const] Wrap,
    P::Wrapped: [const] CheckSpatialInts<SpS>,
{
    type Output = P::Wrapped;
    fn resolve_upper_dhw(self, _: LoD) -> Self::Output {
        self.wrap()
    }
}

/// Derives output extents `(Z, P, Q)` per spatial mode via
/// `1 + floor_div(size(D_i) + lo_i + hi_i - ((T_i - 1) * s_i + 1), u_i)` and validates `z > 0`.
#[doc(hidden)]
pub const trait DeriveShapeZpq<St, LoD, HiD, StD, StT>: Copy {
    type Output: Tuple + Shape;
    fn derive_shape_zpq(
        self,
        stencil: St,
        lower_dhw: LoD,
        upper_dhw: HiD,
        stride_dhw: StD,
        stride_trs: StT,
    ) -> Self::Output;
}

const impl DeriveShapeZpq<Nil, Nil, Nil, Nil, Nil> for Nil {
    type Output = Nil;
    fn derive_shape_zpq(self, _: Nil, _: Nil, _: Nil, _: Nil, _: Nil) -> Nil {
        Nil
    }
}

type EffTap<T, S> = Sum<Product<Difference<T, Int<1>>, S>, Int<1>>;
type FitNumerator<DSize, Lo, Hi, T, S> = Difference<Sum<Sum<DSize, Lo>, Hi>, EffTap<T, S>>;
type DerivedExtent<DSize, Lo, Hi, T, U, S> =
    Sum<Int<1>, FloorQuotient<FitNumerator<DSize, Lo, Hi, T, S>, U>>;

const impl<DH, DT, TH, TT, LoH, LoT, HiH, HiT, UH, UT, SH, ST>
    DeriveShapeZpq<Til<TH, TT>, Til<LoH, LoT>, Til<HiH, HiT>, Til<UH, UT>, Til<SH, ST>>
    for Til<DH, DT>
where
    DH: [const] Shape + [const] TupleProduct,
    DH::Output: [const] Integer + [const] ArithAdd<DH::Output, LoH>,
    TH: [const] ShapeScalar + [const] ArithSub<TH, Int<1>>,
    LoH: [const] Integer,
    HiH: [const] Integer,
    UH: [const] EnsurePositive,
    SH: [const] Integer,
    Difference<TH, Int<1>>: [const] ArithMul<Difference<TH, Int<1>>, SH>,
    Product<Difference<TH, Int<1>>, SH>:
        [const] ArithAdd<Product<Difference<TH, Int<1>>, SH>, Int<1>>,
    Sum<DH::Output, LoH>: [const] ArithAdd<Sum<DH::Output, LoH>, HiH>,
    Sum<Sum<DH::Output, LoH>, HiH>:
        [const] ArithSub<Sum<Sum<DH::Output, LoH>, HiH>, EffTap<TH, SH>>,
    FitNumerator<DH::Output, LoH, HiH, TH, SH>:
        [const] ArithFloorDiv<FitNumerator<DH::Output, LoH, HiH, TH, SH>, UH>,
    Int<1>: [const] ArithAdd<Int<1>, FloorQuotient<FitNumerator<DH::Output, LoH, HiH, TH, SH>, UH>>,
    DerivedExtent<DH::Output, LoH, HiH, TH, UH, SH>: [const] EnsurePositive,
    DT: [const] DeriveShapeZpq<TT, LoT, HiT, UT, ST>,
    TT: Copy,
    LoT: Copy,
    HiT: Copy,
    UT: Copy,
    ST: Copy,
{
    type Output = Til<
        DerivedExtent<DH::Output, LoH, HiH, TH, UH, SH>,
        <DT as DeriveShapeZpq<TT, LoT, HiT, UT, ST>>::Output,
    >;

    fn derive_shape_zpq(
        self,
        stencil: Til<TH, TT>,
        lower_dhw: Til<LoH, LoT>,
        upper_dhw: Til<HiH, HiT>,
        stride_dhw: Til<UH, UT>,
        stride_trs: Til<SH, ST>,
    ) -> Self::Output {
        let _ = stencil.0.size();
        let u = stride_dhw.0.ensure_positive();
        let d_size = self.0.tuple_product();
        let eff_t = add(mul(sub(stencil.0, Int::<1>), stride_trs.0), Int::<1>);
        let num = sub(add(add(d_size, lower_dhw.0), upper_dhw.0), eff_t);
        let z = add(Int::<1>, floor_div(num, u)).ensure_positive();
        Til(
            z,
            self.1.derive_shape_zpq(
                stencil.1,
                lower_dhw.1,
                upper_dhw.1,
                stride_dhw.1,
                stride_trs.1,
            ),
        )
    }
}

/// Resolves `shape_zpq`: derives via fprop formula when [`DefaultParam`], or wraps and validates
/// rank and positivity when caller-supplied.
#[doc(hidden)]
pub const trait ResolveShapeZpq<SpS: Copy, St: Copy, LoD: Copy, HiD: Copy, StD: Copy, StT: Copy>:
    Copy
{
    type Output: Tuple + Shape;
    fn resolve_shape_zpq(
        self,
        spatial_shape: SpS,
        stencil: St,
        lower_dhw: LoD,
        upper_dhw: HiD,
        stride_dhw: StD,
        stride_trs: StT,
    ) -> Self::Output;
}

const impl<SpS, St, LoD, HiD, StD, StT> ResolveShapeZpq<SpS, St, LoD, HiD, StD, StT>
    for DefaultParam
where
    SpS: [const] DeriveShapeZpq<St, LoD, HiD, StD, StT>,
    St: Copy,
    LoD: Copy,
    HiD: Copy,
    StD: Copy,
    StT: Copy,
{
    type Output = SpS::Output;
    fn resolve_shape_zpq(
        self,
        spatial_shape: SpS,
        stencil: St,
        lower_dhw: LoD,
        upper_dhw: HiD,
        stride_dhw: StD,
        stride_trs: StT,
    ) -> Self::Output {
        spatial_shape.derive_shape_zpq(stencil, lower_dhw, upper_dhw, stride_dhw, stride_trs)
    }
}

const impl<P, SpS, St, LoD, HiD, StD, StT> ResolveShapeZpq<SpS, St, LoD, HiD, StD, StT> for P
where
    SpS: Copy,
    St: Copy,
    LoD: Copy,
    HiD: Copy,
    StD: Copy,
    StT: Copy,
    P: [const] Wrap,
    P::Wrapped: [const] CheckSpatialPositive<SpS>,
{
    type Output = P::Wrapped;
    fn resolve_shape_zpq(self, _: SpS, _: St, _: LoD, _: HiD, _: StD, _: StT) -> Self::Output {
        self.wrap().check_spatial_positive()
    }
}

// ------------------------------------------------------------------
// Tiler zipping & corner coordinate construction
// ------------------------------------------------------------------

/// Zips a flat shape tuple `Self` and a congruent flat stride tuple `D` into a tuple of rank-1
/// [`Layout`]s for by-mode composition.
#[doc(hidden)]
pub const trait ZipToLayouts<D>: Copy {
    type Output: Tuple;
    fn zip_to_layouts(self, strides: D) -> Self::Output;
}

const impl ZipToLayouts<Nil> for Nil {
    type Output = Nil;
    fn zip_to_layouts(self, _: Nil) -> Nil {
        Nil
    }
}

const impl<SH, ST, DH, DT> ZipToLayouts<Til<DH, DT>> for Til<SH, ST>
where
    SH: [const] ShapeScalar,
    DH: [const] StrideScalar + Stride<SH>,
    ST: [const] ZipToLayouts<DT>,
    DT: Copy,
{
    type Output = Til<Layout<SH, DH>, ST::Output>;
    fn zip_to_layouts(self, strides: Til<DH, DT>) -> Self::Output {
        let _ = self.0.size();
        Til(
            Layout::new(self.0, strides.0),
            self.1.zip_to_layouts(strides.1),
        )
    }
}

/// Computes `self[i] - rhs[i]` per spatial mode and appends a trailing `Int<0>` for channel `C`.
#[doc(hidden)]
pub const trait SubSpatialWithTrail<R>: Copy {
    type Output: Tuple;
    fn sub_spatial_with_trail(self, rhs: R) -> Self::Output;
}

const impl SubSpatialWithTrail<Nil> for Nil {
    type Output = Til<Int<0>, Nil>;
    fn sub_spatial_with_trail(self, _: Nil) -> Til<Int<0>, Nil> {
        Til(Int::<0>, Nil)
    }
}

const impl<LH, LT, RH, RT> SubSpatialWithTrail<Til<RH, RT>> for Til<LH, LT>
where
    LH: [const] Integer + [const] ArithSub<LH, RH>,
    RH: [const] Integer,
    LT: [const] SubSpatialWithTrail<RT>,
    RT: Copy,
{
    type Output = Til<Difference<LH, RH>, LT::Output>;
    fn sub_spatial_with_trail(self, rhs: Til<RH, RT>) -> Self::Output {
        Til(sub(self.0, rhs.0), self.1.sub_spatial_with_trail(rhs.1))
    }
}

// ------------------------------------------------------------------
// Core primitive: Im2ColLayout
// ------------------------------------------------------------------

/// Computes the `((N, (Z, P, Q)), ((T, R, S), C))` `im2col` [`Layout`] and corner `origin`
/// shift for an activation layout `Self` of shape `(N, [D,] [H,] [W,] C)`.
///
/// Serves as the single source of validation and layout construction behind [`im2col`],
/// [`im2col_coord`], and [`im2col_padded`].
pub const trait Im2ColLayout<St, Opts = Im2ColOptions>: Copy {
    /// Resulting rank-2 `((N, (Z, P, Q)), ((T, R, S), C))` shape type.
    type Shape: Shape;
    /// Resulting congruent stride type.
    type Stride: Stride<Self::Shape>;
    /// Codomain origin shift `A((0, lower_trs - lower_dhw, 0))`.
    type Origin: Copy;

    /// Builds the `im2col` layout and its codomain origin shift.
    fn im2col_layout(
        self,
        stencil: St,
        opts: Opts,
    ) -> (Layout<Self::Shape, Self::Stride>, Self::Origin);
}

const impl<
    S,
    D,
    St,
    LoD,
    HiD,
    StD,
    LoT,
    StT,
    Zpq,
    StW,
    LoDW,
    HiDW,
    StDW,
    LoTW,
    StTW,
    ZpqW,
    TilerZpq,
    TilerTrs,
    PosS,
    PosD,
    TapS,
    TapD,
    CornerTail,
> Im2ColLayout<St, Im2ColOptions<LoD, HiD, StD, LoT, StT, Zpq>> for Layout<S, D>
where
    S: Shape,
    D: Stride<S>,
    Self: [const] SplitActivationLayout,
    St: [const] Wrap<Wrapped = StW>,
    StW: [const] CheckSpatialShapes<<Self as SplitActivationLayout>::SpatialShape>,
    LoD: [const] ResolveSpatialParam<
            <Self as SplitActivationLayout>::SpatialShape,
            Int<0>,
            Output = LoDW,
        >,
    LoDW: Tuple,
    HiD:
        [const] ResolveUpperDhw<<Self as SplitActivationLayout>::SpatialShape, LoDW, Output = HiDW>,
    HiDW: Tuple,
    StD: [const] ResolveSpatialParam<
            <Self as SplitActivationLayout>::SpatialShape,
            Int<1>,
            Output = StDW,
        >,
    StDW: Tuple,
    LoT: [const] ResolveSpatialParam<
            <Self as SplitActivationLayout>::SpatialShape,
            Int<0>,
            Output = LoTW,
        >,
    LoTW: [const] SubSpatialWithTrail<LoDW, Output = CornerTail>,
    StT: [const] ResolveSpatialParam<
            <Self as SplitActivationLayout>::SpatialShape,
            Int<1>,
            Output = StTW,
        >,
    StTW: Tuple,
    Zpq: [const] ResolveShapeZpq<
            <Self as SplitActivationLayout>::SpatialShape,
            StW,
            LoDW,
            HiDW,
            StDW,
            StTW,
            Output = ZpqW,
        >,
    ZpqW: [const] ZipToLayouts<StDW, Output = TilerZpq>,
    StW: [const] ZipToLayouts<StTW, Output = TilerTrs>,
    Layout<
        <Self as SplitActivationLayout>::SpatialShape,
        <Self as SplitActivationLayout>::SpatialStride,
    >: [const] Composition<TilerZpq, Output = Layout<PosS, PosD>>
        + [const] Composition<TilerTrs, Output = Layout<TapS, TapD>>,
    PosS: Shape,
    PosD: Stride<PosS>,
    TapS: Shape,
    TapD: Stride<TapS>,
    CornerTail: Tuple,
    Til<Int<0>, CornerTail>: [const] IndexToCoord<S>,
    <Til<Int<0>, CornerTail> as IndexToCoord<S>>::Output: [const] InnerProduct<D>,
{
    type Shape = Til<
        Til<<Self as SplitActivationLayout>::NShape, Til<PosS, Nil>>,
        Til<Til<TapS, Til<<Self as SplitActivationLayout>::CShape, Nil>>, Nil>,
    >;
    type Stride = Til<
        Til<<Self as SplitActivationLayout>::NStride, Til<PosD, Nil>>,
        Til<Til<TapD, Til<<Self as SplitActivationLayout>::CStride, Nil>>, Nil>,
    >;
    type Origin = <<Til<Int<0>, CornerTail> as IndexToCoord<S>>::Output as InnerProduct<D>>::Output;

    fn im2col_layout(
        self,
        stencil: St,
        opts: Im2ColOptions<LoD, HiD, StD, LoT, StT, Zpq>,
    ) -> (Layout<Self::Shape, Self::Stride>, Self::Origin) {
        let ActivationSplit(n_layout, a_dhw, c_layout) = self.split_activation();

        let stencil_w = stencil.wrap();
        let lower_dhw_w = opts.lower_dhw.resolve_spatial(Int::<0>);
        let upper_dhw_w = opts.upper_dhw.resolve_upper_dhw(lower_dhw_w);
        let stride_dhw_w = opts.stride_dhw.resolve_spatial(Int::<1>);
        let lower_trs_w = opts.lower_trs.resolve_spatial(Int::<0>);
        let stride_trs_w = opts.stride_trs.resolve_spatial(Int::<1>);
        let shape_zpq_w = opts.shape_zpq.resolve_shape_zpq(
            a_dhw.shape,
            stencil_w,
            lower_dhw_w,
            upper_dhw_w,
            stride_dhw_w,
            stride_trs_w,
        );

        let tiler_zpq = shape_zpq_w.zip_to_layouts(stride_dhw_w);
        let tiler_trs = stencil_w.zip_to_layouts(stride_trs_w);

        let pos = a_dhw.composition(tiler_zpq);
        let tap = a_dhw.composition(tiler_trs);

        let m_layout = make_layout(Til(n_layout, Til(pos, Nil)));
        let k_layout = make_layout(Til(tap, Til(c_layout, Nil)));
        let layout = make_layout(Til(m_layout, Til(k_layout, Nil)));

        let corner = Til(Int::<0>, lower_trs_w.sub_spatial_with_trail(lower_dhw_w));
        let origin = self.call(corner);

        (layout, origin)
    }
}

/// Builds the `((N, (Z, P, Q)), ((T, R, S), C))` `im2col` [`Layout`] and origin shift with
/// default options (no padding, unit stride and dilation).
pub const fn im2col_layout<L, St>(
    layout: L,
    stencil: St,
) -> (Layout<L::Shape, L::Stride>, L::Origin)
where
    L: [const] Im2ColLayout<St, Im2ColOptions>,
{
    layout.im2col_layout(stencil, Im2ColOptions::new())
}

/// Builds the `((N, (Z, P, Q)), ((T, R, S), C))` `im2col` [`Layout`] and origin shift with
/// explicit [`Im2ColOptions`].
pub const fn im2col_layout_with<L, St, Opts>(
    layout: L,
    stencil: St,
    opts: Opts,
) -> (Layout<L::Shape, L::Stride>, L::Origin)
where
    L: [const] Im2ColLayout<St, Opts>,
{
    layout.im2col_layout(stencil, opts)
}

// ------------------------------------------------------------------
// The three views: im2col, im2col_coord, im2col_padded
// ------------------------------------------------------------------

/// Constructs the offset-codomain `im2col` tensor view sharing `self`'s backing storage.
pub const trait Im2Col<St, Opts = Im2ColOptions>: Copy {
    /// Resulting shifted accessor type.
    type Accessor: AccessorBase;
    /// Resulting rank-2 `((N, (Z, P, Q)), ((T, R, S), C))` shape type.
    type Shape: Shape;
    /// Resulting congruent stride type.
    type Stride: Stride<Self::Shape>;

    /// Builds the `im2col` view with `opts`.
    fn im2col(
        self,
        stencil: St,
        opts: Opts,
    ) -> Tensor<Self::Accessor, Layout<Self::Shape, Self::Stride>>;
}

const impl<A, S, D, St, Opts> Im2Col<St, Opts> for Tensor<A, Layout<S, D>>
where
    S: Shape,
    D: Stride<S>,
    Layout<S, D>: [const] Im2ColLayout<St, Opts>,
    A: [const] AccessorOffset<<Layout<S, D> as Im2ColLayout<St, Opts>>::Origin>,
{
    type Accessor = <A as AccessorOffset<<Layout<S, D> as Im2ColLayout<St, Opts>>::Origin>>::Output;
    type Shape = <Layout<S, D> as Im2ColLayout<St, Opts>>::Shape;
    type Stride = <Layout<S, D> as Im2ColLayout<St, Opts>>::Stride;

    fn im2col(
        self,
        stencil: St,
        opts: Opts,
    ) -> Tensor<Self::Accessor, Layout<Self::Shape, Self::Stride>> {
        let (layout, origin) = self.layout.im2col_layout(stencil, opts);
        Tensor::new(self.accessor.offset(origin), layout)
    }
}

/// Returns the zero-copy `im2col` view `((N, (Z, P, Q)), ((T, R, S), C))` of activation `act`
/// with default options (no padding, unit traversal stride and dilation).
pub const fn im2col<T, St>(act: T, stencil: St) -> Tensor<T::Accessor, Layout<T::Shape, T::Stride>>
where
    T: [const] Im2Col<St, Im2ColOptions>,
{
    act.im2col(stencil, Im2ColOptions::new())
}

/// Returns the zero-copy `im2col` view `((N, (Z, P, Q)), ((T, R, S), C))` of activation `act`
/// with explicit [`Im2ColOptions`].
pub const fn im2col_with<T, St, Opts>(
    act: T,
    stencil: St,
    opts: Opts,
) -> Tensor<T::Accessor, Layout<T::Shape, T::Stride>>
where
    T: [const] Im2Col<St, Opts>,
{
    act.im2col(stencil, opts)
}

/// Coordinate-codomain `im2col` tensor type returned by [`im2col_coord`] and [`im2col_coord_with`].
pub type Im2ColCoordTensor<Origin, S, D> = Tensor<ImplicitAccessor<Origin>, Layout<S, D>>;

/// Read-only padded `im2col` tensor type returned by [`im2col_padded`] and [`im2col_padded_with`].
pub type Im2ColPaddedTensor<T, Fill, Origin, S, D> =
    Tensor<OutOfBoundsAccessor<T, Fill, Origin>, Layout<S, D>>;

/// Constructs the coordinate-codomain `im2col` tensor view backed by [`ImplicitAccessor`].
pub const trait Im2ColCoord<St, Opts = Im2ColOptions>: Copy {
    /// Coordinate origin type carried by [`ImplicitAccessor`].
    type Origin: Copy;
    /// Resulting rank-2 `((N, (Z, P, Q)), ((T, R, S), C))` shape type.
    type Shape: Shape;
    /// Resulting coordinate-basis stride type.
    type Stride: Stride<Self::Shape>;

    /// Builds the coordinate view with `opts`.
    fn im2col_coord(
        self,
        stencil: St,
        opts: Opts,
    ) -> Im2ColCoordTensor<Self::Origin, Self::Shape, Self::Stride>;
}

const impl<A, S, D, St, Opts, CoordStride, ShiftedOrigin> Im2ColCoord<St, Opts>
    for Tensor<A, Layout<S, D>>
where
    A: AccessorBase,
    S: Shape + [const] PrefixProduct<MakeTopBasisLikeOutput<S>, Stride = CoordStride>,
    S::Profile: [const] MakeTopBasisLikeProfile,
    D: Stride<S>,
    CoordStride: Stride<S>,
    Layout<S, CoordStride>: [const] Im2ColLayout<St, Opts>,
    ImplicitAccessor<Int<0>>: [const] AccessorOffset<
            <Layout<S, CoordStride> as Im2ColLayout<St, Opts>>::Origin,
            Output = ImplicitAccessor<ShiftedOrigin>,
        >,
    ShiftedOrigin: Copy,
{
    type Origin = ShiftedOrigin;
    type Shape = <Layout<S, CoordStride> as Im2ColLayout<St, Opts>>::Shape;
    type Stride = <Layout<S, CoordStride> as Im2ColLayout<St, Opts>>::Stride;

    fn im2col_coord(
        self,
        stencil: St,
        opts: Opts,
    ) -> Im2ColCoordTensor<Self::Origin, Self::Shape, Self::Stride> {
        let coord_base =
            Layout::from_shape_with(self.layout.shape, make_top_basis_like(self.layout.shape));
        let (layout, origin) = coord_base.im2col_layout(stencil, opts);
        Tensor::new(ImplicitAccessor::new(Int::<0>).offset(origin), layout)
    }
}

/// Returns the coordinate-codomain `im2col` view of activation `act` with default options.
///
/// Indexing `(m, k)` yields the `(n, [d,] [h,] [w,] c)` basis coordinate of `act` that
/// [`im2col`] would address (suitable for [`in_bounds`](crate::shape::in_bounds) predication
/// or [`idx2crd`](crate::shape::idx2crd) unpacking).
pub const fn im2col_coord<T, St>(
    act: T,
    stencil: St,
) -> Im2ColCoordTensor<T::Origin, T::Shape, T::Stride>
where
    T: [const] Im2ColCoord<St, Im2ColOptions>,
{
    act.im2col_coord(stencil, Im2ColOptions::new())
}

/// Returns the coordinate-codomain `im2col` view of activation `act` with explicit [`Im2ColOptions`].
pub const fn im2col_coord_with<T, St, Opts>(
    act: T,
    stencil: St,
    opts: Opts,
) -> Im2ColCoordTensor<T::Origin, T::Shape, T::Stride>
where
    T: [const] Im2ColCoord<St, Opts>,
{
    act.im2col_coord(stencil, opts)
}

/// Constructs a read-only bounds-checked `im2col` tensor view backed by [`OutOfBoundsAccessor`].
pub const trait Im2ColPadded<St, Fill, Opts = Im2ColOptions>: Copy {
    /// Coordinate origin type carried by [`OutOfBoundsAccessor`].
    type Origin: Copy;
    /// Resulting rank-2 `((N, (Z, P, Q)), ((T, R, S), C))` shape type.
    type Shape: Shape;
    /// Resulting coordinate-basis stride type.
    type Stride: Stride<Self::Shape>;

    /// Builds the padded read-only view with `fill` and `opts`.
    fn im2col_padded(
        self,
        stencil: St,
        fill: Fill,
        opts: Opts,
    ) -> Im2ColPaddedTensor<Self, Fill, Self::Origin, Self::Shape, Self::Stride>;
}

const impl<A, S, D, St, Fill, Opts> Im2ColPadded<St, Fill, Opts> for Tensor<A, Layout<S, D>>
where
    A: AccessorBase,
    S: Shape,
    D: Stride<S>,
    Fill: Copy,
    Self: [const] Im2ColCoord<St, Opts>,
{
    type Origin = <Self as Im2ColCoord<St, Opts>>::Origin;
    type Shape = <Self as Im2ColCoord<St, Opts>>::Shape;
    type Stride = <Self as Im2ColCoord<St, Opts>>::Stride;

    fn im2col_padded(
        self,
        stencil: St,
        fill: Fill,
        opts: Opts,
    ) -> Im2ColPaddedTensor<Self, Fill, Self::Origin, Self::Shape, Self::Stride> {
        let crd_view = self.im2col_coord(stencil, opts);
        Tensor::new(
            OutOfBoundsAccessor::with_origin(self, fill, crd_view.accessor.origin),
            crd_view.layout,
        )
    }
}

/// Returns a read-only padded `im2col` view of activation `act` with default options, returning
/// `fill` for out-of-bounds coordinates.
pub const fn im2col_padded<T, St, Fill>(
    act: T,
    stencil: St,
    fill: Fill,
) -> Im2ColPaddedTensor<T, Fill, T::Origin, T::Shape, T::Stride>
where
    T: [const] Im2ColPadded<St, Fill, Im2ColOptions>,
{
    act.im2col_padded(stencil, fill, Im2ColOptions::new())
}

/// Returns a read-only padded `im2col` view of activation `act` with explicit [`Im2ColOptions`],
/// returning `fill` for out-of-bounds coordinates without allocating padded storage.
pub const fn im2col_padded_with<T, St, Fill, Opts>(
    act: T,
    stencil: St,
    fill: Fill,
    opts: Opts,
) -> Im2ColPaddedTensor<T, Fill, T::Origin, T::Shape, T::Stride>
where
    T: [const] Im2ColPadded<St, Fill, Opts>,
{
    act.im2col_padded(stencil, fill, opts)
}
