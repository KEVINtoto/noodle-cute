//! Optimized COPY with three-stage layout analysis (Whitepaper §2.6.1, §3.4.2).
//!
//! Before planning, every destination address must identify a single source address.
//! Integer layouts with separated strides are certified from their axes (including
//! matching broadcast axes). Remaining layouts are checked exactly by comparing pairs of
//! indices, using constant storage and worst-case quadratic time in the domain size.
//!
//! Reshapes the iteration space of [`reference::copy`] before
//! dispatching, establishing one algebraic invariant at each stage:
//!
//! 1. **Common domain** ([`greatest_common_domain()`]): Finds the maximal common domain
//!    tiler admissible for both `src` and `dst`, dividing each into `(Compat, InCompat)`.
//!    Only `Compat` is permuted; `InCompat` is iterated as-is.
//! 2. **Nullspace** ([`nullspace()`]): Divides `Compat` by `nullspace(dst_c[0])` to isolate
//!    stride-0 coordinates that target the same destination address. If `src` varies across
//!    those coordinates, a **write-after-write** error is rejected (at compile time for
//!    static layouts, or via panic for dynamic layouts). If `src` is constant across them,
//!    all redundant writes are sliced away with `[(0, ..), ..]`.
//! 3. **Alignment & vectorization** ([`right_inverse`]): Divides `NonNull` by
//!    `right_inverse(dst_n[0])` to walk the compatible domain in destination-memory order,
//!    extracts the maximal common contiguous vector width `V` from
//!    `coalesce(src_v[0][0])`, divides both tensors into `(V:1, Grid)`, and dispatches each
//!    slice to [`reference::copy`].
//!
//! # Main Interfaces
//!
//! - [`CopyPlan`] / [`copy_plan`]: Executes stages 1–3 and returns the reshaped
//!   `(V:1, Grid)` tensor views `(src_v, dst_v)`.
//! - [`copy`]: Runs [`copy_plan`] and dispatches every grid slice `0..size[1](dst_v)`
//!   to [`reference::copy`].
//!
//! # Usage
//!
//! ```
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl, const_ops, const_cmp)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algorithm::{copy, copy_plan};
//! use noodle_cute::int_tuple;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::tensor::make_tensor;
//!
//! let src_buf = [0i32, 1, 2, 3, 4, 5, 6, 7];
//! let mut dst_buf = [0i32; 8];
//! let src = make_tensor(
//!     &src_buf,
//!     Layout::new(int_tuple![Int::<2>, Int::<4>], int_tuple![Int::<4>, Int::<1>]),
//! );
//! let mut dst = make_tensor(
//!     &mut dst_buf,
//!     Layout::new(int_tuple![Int::<2>, Int::<4>], int_tuple![Int::<4>, Int::<1>]),
//! );
//!
//! let (src_v, dst_v) = copy_plan(src, dst);
//! assert_eq!(src_v.layout.get::<0>().size(), 8);
//! assert_eq!(dst_v.layout.get::<1>().size(), 1);
//!
//! copy(src, &mut dst);
//! assert_eq!(dst_buf, src_buf);
//! ```
//!
//! # Notes
//!
//! - Layout-algebra stages (`nullspace`, `right_inverse`, `logical_divide`) require static
//!   extents across the common domain and static compatible strides on `dst`; `src` strides
//!   and unmatched trailing modes may be dynamic.
//! - Use [`reference::copy`] for fully dynamic layouts when compile-time
//!   algebraic analysis is not required.
//!
//! # Compile-Time Rejections
//!
//! Static write-after-write conflicts (varying `src` written into a stride-0 `dst`) are
//! rejected at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algorithm::copy;
//! use noodle_cute::integer::Int;
//! use noodle_cute::layout::Layout;
//! use noodle_cute::tensor::make_tensor;
//!
//! let src_buf = [0i32, 1, 2, 3];
//! let mut dst_buf = [0i32; 1];
//! let src = make_tensor(&src_buf, Layout::new(Int::<4>, Int::<1>));
//! let mut dst = make_tensor(&mut dst_buf, Layout::new(Int::<4>, Int::<0>));
//! copy(src, &mut dst);
//! ```
//!
//! Distinct source addresses collide for destination strides `(1, 1)`:
//!
//! ```compile_fail,E0080
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{algorithm::copy_plan, integer::Int, int_tuple, layout::Layout, tensor::{Tensor, ImplicitAccessor}};
//! let shape = int_tuple![Int::<2>, Int::<2>];
//! let source = Tensor::new(ImplicitAccessor::new(0i32), Layout::new(shape, int_tuple![Int::<1>, Int::<2>]));
//! let dest = Tensor::new(ImplicitAccessor::new(0i32), Layout::new(shape, int_tuple![Int::<1>, Int::<1>]));
//! let _ = copy_plan(source, dest);
//! ```
//!
//! Distinct source addresses collide for destination strides `(0, 0)`:
//!
//! ```compile_fail,E0080
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{algorithm::copy_plan, integer::Int, int_tuple, layout::Layout, tensor::{Tensor, ImplicitAccessor}};
//! let shape = int_tuple![Int::<2>, Int::<2>];
//! let source = Tensor::new(ImplicitAccessor::new(0i32), Layout::new(shape, int_tuple![Int::<1>, Int::<-1>]));
//! let dest = Tensor::new(ImplicitAccessor::new(0i32), Layout::new(shape, int_tuple![Int::<0>, Int::<0>]));
//! let _ = copy_plan(source, dest);
//! ```
//!
//! Statically mismatched copy sizes fail at compile time:
//!
//! ```compile_fail,E0277
//! # #![feature(min_generic_const_args, generic_const_args, macroless_generic_const_args, generic_const_items, const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::algorithm::copy;
//! use noodle_cute::integer::Int;
//! use noodle_cute::tensor::make_tensor;
//!
//! let src_buf = [0i32; 4];
//! let mut dst_buf = [0i32; 8];
//! let src = make_tensor(&src_buf, Int::<4>);
//! let mut dst = make_tensor(&mut dst_buf, Int::<8>);
//! copy(src, &mut dst);
//! ```

use core::ops::RangeFull;

use crate::NoodleCuteError;
use crate::algebra::common::LeadingVecSize;
use crate::algebra::{
    CoalesceTarget, GreatestCommonDomain, LogicalDivide, LogicalDivideAt, Nullspace, RightInverse,
    coalesce, greatest_common_domain, logical_divide, logical_divide_at, nullspace, right_inverse,
};
use crate::basis::{BasisCoeff, BasisCombination};
use crate::integer::{ArithSub, Difference, Int, add, int, sub};
use crate::layout::Layout;
use crate::layout::alias::compatible_aliases;
use crate::shape::{Shape, ShapeScalar};
use crate::stride::Stride;
use crate::tensor::{Tensor, TensorIndex, TensorSet};
use crate::tuple::{Get, GetOutput, Item, Mode, ModeEnd, Nil, Til, TupleProduct, tuple_product};

use super::reference::{self, STATIC_SIZE_COMPAT, ValidSizeMatch};

/// Mode path `[0]` targeting the compatible domain in stages 2 and 3.
type Mode0 = Mode<0, ModeEnd>;

/// Partial coordinate `((0, ..), ..)` slicing out the nullspace mode while retaining
/// `(NonNull, InCompat)`.
type DropNullCoord = Til<Til<Int<0>, Til<RangeFull, Nil>>, Til<RangeFull, Nil>>;

/// Partial coordinate `(.., i)` slicing one `V:1` vector transfer at grid index `i`.
type VecSliceCoord = Til<RangeFull, Til<int, Nil>>;

/// Compile-time guard rejecting a statically non-zero value evaluated over a destination
/// nullspace (`NONZERO == false` required).
#[doc(hidden)]
pub const trait ValidNullspaceZero<const NONZERO: bool> {}
const impl ValidNullspaceZero<false> for () {}

/// Classifies whether an evaluated nullspace offset is statically non-zero or dynamically zero.
#[doc(hidden)]
pub const trait NullspaceOffsetValue: Copy {
    /// `true` iff `Self` is statically known to be non-zero.
    const IS_STATIC_NONZERO: bool;
    /// Returns `true` iff `self` evaluates to zero.
    fn is_zero_offset(self) -> bool;
}

const impl<V: [const] BasisCoeff> NullspaceOffsetValue for V {
    const IS_STATIC_NONZERO: bool = V::IS_STATIC && !V::IS_STATIC_ZERO;

    fn is_zero_offset(self) -> bool {
        self.is_zero()
    }
}

const impl NullspaceOffsetValue for BasisCombination<Nil> {
    const IS_STATIC_NONZERO: bool = false;

    fn is_zero_offset(self) -> bool {
        true
    }
}

/// Compile-time flag: `true` when nullspace evaluation `V` is statically non-zero.
#[doc(hidden)]
pub const OFFSET_IS_STATIC_NONZERO<V: NullspaceOffsetValue>: bool = V::IS_STATIC_NONZERO;

/// Compile-time flag: `true` iff vector width `V` is statically `Int<1>`.
#[doc(hidden)]
pub const IS_SCALAR_VEC<V: ShapeScalar>: bool = matches!(V::STATIC_VALUE, Some(1));

/// Splits a Stage-3 tensor into `(V:1, Grid)`:
/// - `SCALAR == true` (`V == Int<1>`): wraps `(Int<1>, layout)` directly without invoking
///   multi-mode composition, supporting arbitrary dynamic layouts.
/// - `SCALAR == false` (`V > 1`): divides via [`logical_divide`].
#[doc(hidden)]
pub const trait DivideByVec<V: ShapeScalar, const SCALAR: bool>: Copy {
    type Output: Copy;
    fn divide_by_vec(self, vec_size: V) -> Self::Output;
}

const impl<A, S, D> DivideByVec<Int<1>, true> for Tensor<A, Layout<S, D>>
where
    A: Copy,
    S: Shape,
    D: Stride<S>,
{
    type Output = Tensor<A, Layout<Til<Int<1>, Til<S, Nil>>, Til<Int<0>, Til<D, Nil>>>>;

    fn divide_by_vec(self, _: Int<1>) -> Self::Output {
        Tensor {
            accessor: self.accessor,
            layout: Layout::new(
                Til(Int::<1>, Til(self.layout.shape, Nil)),
                Til(Int::<0>, Til(self.layout.stride, Nil)),
            ),
        }
    }
}

const impl<T, V> DivideByVec<V, false> for T
where
    V: [const] ShapeScalar,
    T: [const] LogicalDivide<V>,
{
    type Output = T::Output;

    fn divide_by_vec(self, vec_size: V) -> Self::Output {
        logical_divide(self, vec_size)
    }
}

// Type aliases describing the three-stage layout pipeline of `CopyPlan`.
type CommonDom<Src, Dst> =
    Layout<<Src as GreatestCommonDomain<Dst>>::Shape, <Src as GreatestCommonDomain<Dst>>::Stride>;
type NullDst<DstC> = <GetOutput<DstC, 0> as Nullspace>::Output;
type Stage2Div<T, Null> = <T as LogicalDivideAt<Null, Mode0>>::Output;
type Stage2Slice<T> = <T as TensorIndex<DropNullCoord>>::Output;
type InvDst<DstN> = <GetOutput<DstN, 0> as RightInverse>::Output;
type AlignedSrcLead<SrcV> = <GetOutput<GetOutput<SrcV, 0>, 0> as CoalesceTarget<false>>::Output;
type VecWidth<SrcV> = <AlignedSrcLead<SrcV> as LeadingVecSize>::Output;

/// Executes the three-stage COPY layout analysis (`greatest_common_domain` -> `nullspace` ->
/// `right_inverse` + vectorization) to produce `(V:1, Grid)` source and destination views.
pub const trait CopyPlan<Dst>: Copy {
    /// Vectorized `(V:1, Grid)` source tensor type.
    type SrcVec: Copy;
    /// Vectorized `(V:1, Grid)` destination tensor type.
    type DstVec: Copy;

    /// Analyzes `self` and `dst`, rejecting size mismatches and stride-0 write-after-write
    /// conflicts, and returns `(src_v, dst_v)` shaped as `(V:1, Grid)`.
    fn copy_plan(self, dst: Dst) -> (Self::SrcVec, Self::DstVec);
}

const impl<
    AS,
    SS,
    DS,
    AD,
    SD,
    DD,
    SrcOff,
    DstOff,
    SrcCL,
    DstCL,
    NullS,
    NullD,
    SrcNL,
    DstNL,
    DstNullVal,
    SrcNullVal,
    SrcNS,
    SrcND,
    DstNS,
    DstND,
    SrcNA,
    DstNA,
    SrcVL,
    DstVL,
> CopyPlan<Tensor<AD, Layout<SD, DD>>> for Tensor<AS, Layout<SS, DS>>
where
    AS: Copy,
    AD: Copy,
    SS: [const] Shape,
    DS: [const] Stride<SS>,
    SD: [const] Shape,
    DD: [const] Stride<SD>,
    (): ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SS, SD> }>,
    Layout<SS, DS>: const Fn<(int,), Output = SrcOff>,
    Layout<SD, DD>: const Fn<(int,), Output = DstOff>,
    SrcOff: Copy + const PartialEq,
    DstOff: Copy + const PartialEq,
    SS: const Item,
    DS: const Item,
    SD: const Item,
    DD: const Item,
    // Stage 1: greatest_common_domain & logical_divide -> (Compat, InCompat)
    Self: [const] GreatestCommonDomain<Tensor<AD, Layout<SD, DD>>>
        + [const] LogicalDivide<
            CommonDom<Self, Tensor<AD, Layout<SD, DD>>>,
            Output = Tensor<AS, SrcCL>,
        >,
    Tensor<AD, Layout<SD, DD>>: [const] LogicalDivide<
            CommonDom<Self, Tensor<AD, Layout<SD, DD>>>,
            Output = Tensor<AD, DstCL>,
        >,
    SrcCL: Copy,
    DstCL: Copy + [const] Get<0>,
    // Stage 2: nullspace(dst_c[0]), logical_divide[0], WAW check, slice [(0, ..), ..]
    GetOutput<DstCL, 0>: [const] Nullspace<Output = Layout<NullS, NullD>>,
    NullS: [const] Shape + [const] TupleProduct,
    NullD: Stride<NullS>,
    <NullS as TupleProduct>::Output: [const] ArithSub<<NullS as TupleProduct>::Output, Int<1>>,
    Tensor<AS, SrcCL>:
        [const] LogicalDivideAt<Layout<NullS, NullD>, Mode0, Output = Tensor<AS, SrcNL>>,
    Tensor<AD, DstCL>:
        [const] LogicalDivideAt<Layout<NullS, NullD>, Mode0, Output = Tensor<AD, DstNL>>,
    SrcNL: Copy + [const] Get<0>,
    DstNL: Copy + [const] Get<0>,
    GetOutput<SrcNL, 0>: [const] Get<0>,
    GetOutput<DstNL, 0>: [const] Get<0>,
    GetOutput<GetOutput<DstNL, 0>, 0>:
        [const] Fn<(Difference<<NullS as TupleProduct>::Output, Int<1>>,), Output = DstNullVal>,
    GetOutput<GetOutput<SrcNL, 0>, 0>:
        [const] Fn<(Difference<<NullS as TupleProduct>::Output, Int<1>>,), Output = SrcNullVal>,
    DstNullVal: [const] NullspaceOffsetValue,
    SrcNullVal: [const] NullspaceOffsetValue,
    (): ValidNullspaceZero<{ OFFSET_IS_STATIC_NONZERO::<DstNullVal> }>
        + ValidNullspaceZero<{ OFFSET_IS_STATIC_NONZERO::<SrcNullVal> }>,
    Tensor<AS, SrcNL>:
        [const] TensorIndex<DropNullCoord, Output = Tensor<SrcNA, Layout<SrcNS, SrcND>>>,
    Tensor<AD, DstNL>:
        [const] TensorIndex<DropNullCoord, Output = Tensor<DstNA, Layout<DstNS, DstND>>>,
    SrcNA: Copy,
    DstNA: Copy,
    SrcNS: Shape,
    SrcND: Stride<SrcNS>,
    DstNS: Shape,
    DstND: Stride<DstNS>,
    Layout<DstNS, DstND>: [const] Get<0>,
    // Stage 3: right_inverse(dst_n[0]), logical_divide[0], vec_size, logical_divide -> (V:1, Grid)
    GetOutput<Layout<DstNS, DstND>, 0>: [const] RightInverse,
    Tensor<SrcNA, Layout<SrcNS, SrcND>>:
        [const] LogicalDivideAt<InvDst<Layout<DstNS, DstND>>, Mode0, Output = Tensor<SrcNA, SrcVL>>,
    Tensor<DstNA, Layout<DstNS, DstND>>:
        [const] LogicalDivideAt<InvDst<Layout<DstNS, DstND>>, Mode0, Output = Tensor<DstNA, DstVL>>,
    SrcVL: Copy + [const] Get<0>,
    DstVL: Copy,
    GetOutput<SrcVL, 0>: [const] Get<0>,
    GetOutput<GetOutput<SrcVL, 0>, 0>: [const] CoalesceTarget<false>,
    AlignedSrcLead<SrcVL>: [const] LeadingVecSize,
    Tensor<SrcNA, SrcVL>: [const] LogicalDivide<VecWidth<SrcVL>>,
    Tensor<DstNA, DstVL>: [const] LogicalDivide<VecWidth<SrcVL>>,
{
    type SrcVec = <Tensor<SrcNA, SrcVL> as LogicalDivide<VecWidth<SrcVL>>>::Output;
    type DstVec = <Tensor<DstNA, DstVL> as LogicalDivide<VecWidth<SrcVL>>>::Output;

    fn copy_plan(self, dst: Tensor<AD, Layout<SD, DD>>) -> (Self::SrcVec, Self::DstVec) {
        NoodleCuteError::SizeMismatch
            .check_with(self.size() == dst.size(), "copy: size mismatch");

        let checked_statically = const {
            if let (Some(ss), Some(ds), Some(sd), Some(dd), Some(n)) = (
                SS::STATIC_ITEM,
                DS::STATIC_ITEM,
                SD::STATIC_ITEM,
                DD::STATIC_ITEM,
                SS::STATIC_SIZE,
            ) {
                let src = Layout {
                    shape: ss,
                    stride: ds,
                };
                let dst = Layout {
                    shape: sd,
                    stride: dd,
                };
                NoodleCuteError::NonInjective.check_with(
                    compatible_aliases(src, dst, n),
                    "Write-after-write error detected in dst",
                );
                true
            } else {
                false
            }
        };
        if !checked_statically {
            NoodleCuteError::NonInjective.check_with(
                compatible_aliases(self.layout, dst.layout, self.size()),
                "Write-after-write error detected in dst",
            );
        }

        // Stage 1: Extract domain-compatible portions (Compat, InCompat)
        let common_domain = greatest_common_domain(self, dst);
        let src_c: Tensor<AS, SrcCL> = logical_divide(self, common_domain);
        let dst_c: Tensor<AD, DstCL> = logical_divide(dst, common_domain);

        // Stage 2: Account for nullspaces of dst and src within compatible portions
        let null_dst: NullDst<DstCL> = nullspace(Get::<0>::get(&dst_c.layout));
        let src_n: Stage2Div<Tensor<AS, SrcCL>, NullDst<DstCL>> =
            logical_divide_at::<Mode0, _, _>(src_c, null_dst);
        let dst_n: Stage2Div<Tensor<AD, DstCL>, NullDst<DstCL>> =
            logical_divide_at::<Mode0, _, _>(dst_c, null_dst);

        let null_last = sub(tuple_product(&null_dst.shape), Int::<1>);
        let dst_null = Get::<0>::get(&Get::<0>::get(&dst_n.layout));
        let dst_null_end = dst_null(null_last);
        assert!(
            dst_null_end.is_zero_offset(),
            "Sanity: nullspace definition error"
        );
        let src_null = Get::<0>::get(&Get::<0>::get(&src_n.layout));
        let src_null_end = src_null(null_last);
        assert!(
            src_null_end.is_zero_offset(),
            "Write-after-write error detected in dst"
        );

        // Slice the common stride-0 modes out -> (NonNull, InCompat)
        let drop_null = Til(Til(Int::<0>, Til(.., Nil)), Til(.., Nil));
        let src_n: Stage2Slice<Tensor<AS, SrcNL>> = src_n.at(drop_null);
        let dst_n: Stage2Slice<Tensor<AD, DstNL>> = dst_n.at(drop_null);

        // Stage 3: Align compatible portions to destination memory order and extract V
        let inv_dst: InvDst<Layout<DstNS, DstND>> = right_inverse(dst_n.layout.get::<0>());
        let src_v: Tensor<SrcNA, SrcVL> = logical_divide_at::<Mode0, _, _>(src_n, inv_dst);
        let dst_v: Tensor<DstNA, DstVL> = logical_divide_at::<Mode0, _, _>(dst_n, inv_dst);

        let aligned_src = Get::<0>::get(&Get::<0>::get(&src_v.layout));
        let vec_size = coalesce(aligned_src).leading_vec_size();
        let src_v = logical_divide(src_v, vec_size);
        let dst_v = logical_divide(dst_v, vec_size);

        (src_v, dst_v)
    }
}

/// Runs the three-stage COPY layout analysis on `(src, dst)` and returns the reshaped
/// `(V:1, Grid)` source and destination tensor views without performing any element writes.
pub const fn copy_plan<Src, Dst>(src: Src, dst: Dst) -> (Src::SrcVec, Src::DstVec)
where
    Src: [const] CopyPlan<Dst>,
{
    src.copy_plan(dst)
}

/// Optimized COPY with layout analysis (Whitepaper §2.6.1).
///
/// Reshapes `src` and `dst` via [`copy_plan`] (common domain, nullspace deduplication &
/// write-after-write rejection, destination-memory alignment, and maximal common contiguous
/// vectorization), then dispatches every grid slice to [`reference::copy`].
pub fn copy<
    Src,
    Dst,
    SrcVA,
    SrcVS,
    SrcVD,
    DstVA,
    DstVT,
    DstVG,
    DstVDT,
    DstVDG,
    SrcSliceA,
    SrcSliceS,
    SrcSliceD,
    DstSliceA,
    DstSliceS,
    DstSliceD,
    T,
>(
    src: Src,
    dst: &mut Dst,
) where
    Dst: Copy,
    Src: CopyPlan<
            Dst,
            SrcVec = Tensor<SrcVA, Layout<SrcVS, SrcVD>>,
            DstVec = Tensor<
                DstVA,
                Layout<Til<DstVT, Til<DstVG, Nil>>, Til<DstVDT, Til<DstVDG, Nil>>>,
            >,
        >,
    SrcVA: Copy,
    SrcVS: Shape,
    SrcVD: Stride<SrcVS>,
    DstVA: Copy,
    DstVT: Shape,
    DstVG: Shape,
    DstVDT: Stride<DstVT>,
    DstVDG: Stride<DstVG>,
    Tensor<SrcVA, Layout<SrcVS, SrcVD>>:
        TensorIndex<VecSliceCoord, Output = Tensor<SrcSliceA, Layout<SrcSliceS, SrcSliceD>>>,
    Tensor<DstVA, Layout<Til<DstVT, Til<DstVG, Nil>>, Til<DstVDT, Til<DstVDG, Nil>>>>:
        TensorIndex<VecSliceCoord, Output = Tensor<DstSliceA, Layout<DstSliceS, DstSliceD>>>,
    SrcSliceS: Shape,
    SrcSliceD: Stride<SrcSliceS>,
    DstSliceS: Shape,
    DstSliceD: Stride<DstSliceS>,
    T: Copy,
    (): ValidSizeMatch<{ STATIC_SIZE_COMPAT::<SrcSliceS, DstSliceS> }>,
    Tensor<SrcSliceA, Layout<SrcSliceS, SrcSliceD>>: TensorIndex<int, Output = T>,
    Tensor<DstSliceA, Layout<DstSliceS, DstSliceD>>: TensorSet<int, Item = T>,
{
    let (src_v, dst_v) = copy_plan(src, *dst);
    let grid = dst_v.layout.shape.1.0.size();
    let mut i: int = 0;
    while i < grid {
        let src_slice = src_v.at(Til(.., Til(i, Nil)));
        let mut dst_slice = dst_v.at(Til(.., Til(i, Nil)));
        reference::copy(src_slice, &mut dst_slice);
        i = add(i, 1);
    }
}
