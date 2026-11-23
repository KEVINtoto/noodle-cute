//! Python-style `const`-capable slice specifications and dimension resolution.
//!
//! Represents `(start, stop, step)` slices over signed [`int`] coordinates so they
//! can be constructed as `const` generic parameters (e.g. for tuple slicing in
//! [`crate::tuple::SliceRange`]) or resolved at runtime against a dimension length.
//!
//! # Main Interfaces
//!
//! - [`Bound`] (aliased as [`SliceBound`]): `ConstParamTy` endpoint bound
//!   (`Included`, `Excluded`, `Unbounded`) convertible to/from [`core::ops::Bound<int>`].
//! - [`Slice`]: `(start, stop, step)` specification constructed via [`Slice::new`],
//!   [`Slice::from_slice`], [`From`], [`slice!`](crate::slice!), or
//!   [`slices!`](crate::slices!), and resolved via [`Slice::resolve`].
//! - [`ResolvedSlice`]: Normalized `(start, step, len)` slice against a concrete dimension.
//! - [`IntoSlice`]: `const` conversion trait from Rust range types, single indices,
//!   and bound pairs into unit-step [`Slice`]s.
//!
//! # Usage
//!
//! ```
//! # #![feature(const_trait_impl)]
//! # #![allow(incomplete_features)]
//! use noodle_cute::{slice, slices};
//! use noodle_cute::slice::{ResolvedSlice, Slice};
//!
//! const S: Slice = slice!([1..5] : 2);
//! assert_eq!(S.resolve(10), Some(ResolvedSlice { start: 1, step: 2, len: 2 }));
//!
//! const REV: Slice = slice!([..] : -1);
//! assert_eq!(REV.resolve(4), Some(ResolvedSlice { start: 3, step: -1, len: 4 }));
//!
//! let batch: [Slice; 2] = slices!(..-1: 2, -2);
//! assert_eq!(batch[0].resolve(6), Some(ResolvedSlice { start: 0, step: 2, len: 3 }));
//! ```
//!
//! # Notes
//!
//! - Negative endpoints wrap relative to `dim`, out-of-range bounds clamp, and empty
//!   selections yield `len = 0`.
//! - Bounds follow the direction of travel: with a negative `step`, `Bound::Included(stop)`
//!   selects down to and including `stop`.
//! - [`Slice::resolve`] returns `None` when `step == 0` or `dim < 0`.

use core::marker::ConstParamTy;
use core::ops::{
    Bound as CoreBound, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::integer::{add, ceil_div, int, max, min, sub};

/// Endpoint bound of a [`Slice`] using project [`int`] coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum Bound {
    /// Inclusive bound (`x` in `x..y`, `..=y`, or single index `x`).
    Included(int),
    /// Exclusive bound (`y` in `x..y` or `..y`).
    Excluded(int),
    /// Open bound (`..`).
    Unbounded,
}

impl Bound {
    /// Converts a [`core::ops::Bound<int>`] into a [`Bound`].
    pub const fn from_core(bound: CoreBound<int>) -> Self {
        match bound {
            CoreBound::Included(x) => Self::Included(x),
            CoreBound::Excluded(x) => Self::Excluded(x),
            CoreBound::Unbounded => Self::Unbounded,
        }
    }

    /// Converts this [`Bound`] into a [`core::ops::Bound<int>`].
    pub const fn into_core(self) -> CoreBound<int> {
        match self {
            Self::Included(x) => CoreBound::Included(x),
            Self::Excluded(x) => CoreBound::Excluded(x),
            Self::Unbounded => CoreBound::Unbounded,
        }
    }
}

impl From<CoreBound<int>> for Bound {
    fn from(bound: CoreBound<int>) -> Self {
        Self::from_core(bound)
    }
}

impl From<Bound> for CoreBound<int> {
    fn from(bound: Bound) -> Self {
        bound.into_core()
    }
}

/// Alias matching tuple slicing bound terminology.
pub type SliceBound = Bound;

/// A slice resolved against one dimension: `len` elements, the first at
/// `start`, each `step` apart. Signed so that reverse slices are expressible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedSlice {
    /// Index of the first selected element (may be `-1` for an empty reverse slice).
    pub start: int,
    /// Signed stride between consecutive selected elements (non-zero).
    pub step: int,
    /// Number of selected elements (`>= 0`).
    pub len: int,
}

/// Const-capable slice specification `(start, stop, step)` in [`int`] coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub struct Slice {
    /// Starting endpoint bound.
    pub start: Bound,
    /// Stopping endpoint bound.
    pub stop: Bound,
    /// Signed step size (`0` is invalid when resolved).
    pub step: int,
}

/// Const conversion into a unit-step [`Slice`].
pub const trait IntoSlice {
    /// Converts `self` into a [`Slice`] with `step = 1`.
    fn into_slice(self) -> Slice;
}

const impl IntoSlice for Slice {
    fn into_slice(self) -> Slice {
        self
    }
}

const impl IntoSlice for int {
    fn into_slice(self) -> Slice {
        Slice {
            start: Bound::Included(self),
            stop: Bound::Included(self),
            step: 1,
        }
    }
}

const impl IntoSlice for Range<int> {
    fn into_slice(self) -> Slice {
        Slice {
            start: Bound::Included(self.start),
            stop: Bound::Excluded(self.end),
            step: 1,
        }
    }
}

const impl IntoSlice for RangeInclusive<int> {
    fn into_slice(self) -> Slice {
        Slice {
            start: Bound::Included(*self.start()),
            stop: Bound::Included(*self.end()),
            step: 1,
        }
    }
}

const impl IntoSlice for RangeFrom<int> {
    fn into_slice(self) -> Slice {
        Slice {
            start: Bound::Included(self.start),
            stop: Bound::Unbounded,
            step: 1,
        }
    }
}

const impl IntoSlice for RangeTo<int> {
    fn into_slice(self) -> Slice {
        Slice {
            start: Bound::Unbounded,
            stop: Bound::Excluded(self.end),
            step: 1,
        }
    }
}

const impl IntoSlice for RangeToInclusive<int> {
    fn into_slice(self) -> Slice {
        Slice {
            start: Bound::Unbounded,
            stop: Bound::Included(self.end),
            step: 1,
        }
    }
}

const impl IntoSlice for RangeFull {
    fn into_slice(self) -> Slice {
        Slice {
            start: Bound::Unbounded,
            stop: Bound::Unbounded,
            step: 1,
        }
    }
}

const impl IntoSlice for (Bound, Bound) {
    fn into_slice(self) -> Slice {
        Slice {
            start: self.0,
            stop: self.1,
            step: 1,
        }
    }
}

const impl IntoSlice for (CoreBound<int>, CoreBound<int>) {
    fn into_slice(self) -> Slice {
        Slice {
            start: Bound::from_core(self.0),
            stop: Bound::from_core(self.1),
            step: 1,
        }
    }
}

const fn clamp_int(x: int, lo: int, hi: int) -> int {
    min(max(x, lo), hi)
}

const fn wrap_index(x: int, dim: int) -> int {
    if x < 0 { add(x, dim) } else { x }
}

const fn wrap_inc_clamped(x: int, dim: int) -> int {
    let w = wrap_index(x, dim);
    if w < 0 {
        0
    } else if w >= dim {
        dim
    } else {
        add(w, 1)
    }
}

const fn wrap_dec_clamped(x: int, dim: int, last: int) -> int {
    let w = wrap_index(x, dim);
    if w <= 0 {
        -1
    } else if w > dim {
        last
    } else {
        sub(w, 1)
    }
}

impl Slice {
    /// Constructs a [`Slice`] from any const-convertible range or index and explicit `step`.
    pub const fn new<R: [const] IntoSlice>(range: R, step: int) -> Self {
        let s = range.into_slice();
        Self {
            start: s.start,
            stop: s.stop,
            step,
        }
    }

    /// Constructs a unit-step [`Slice`] from a range, single index, or existing [`Slice`].
    pub const fn from_slice<R: [const] IntoSlice>(range: R) -> Self {
        range.into_slice()
    }

    /// Returns `(start, stop, step)`.
    pub const fn get(&self) -> (Bound, Bound, int) {
        (self.start, self.stop, self.step)
    }

    /// Resolve against a dimension of length `dim` with Python semantics:
    /// negative indices count from the end, out-of-range bounds are clipped
    /// and an empty selection is allowed. Bounds are interpreted in the
    /// direction of travel, so `Included(stop)` with a negative step means
    /// "down to and including `stop`".
    ///
    /// Returns `None` if `self.step == 0` or `dim < 0`.
    pub const fn resolve(&self, dim: int) -> Option<ResolvedSlice> {
        let step = self.step;
        if step == 0 || dim < 0 {
            return None;
        }
        if step > 0 {
            let start = match self.start {
                Bound::Included(x) => clamp_int(wrap_index(x, dim), 0, dim),
                Bound::Excluded(x) => wrap_inc_clamped(x, dim),
                Bound::Unbounded => 0,
            };
            let stop = match self.stop {
                Bound::Included(x) => wrap_inc_clamped(x, dim),
                Bound::Excluded(x) => clamp_int(wrap_index(x, dim), 0, dim),
                Bound::Unbounded => dim,
            };
            let len = if start >= stop {
                0
            } else {
                ceil_div(sub(stop, start), step)
            };
            Some(ResolvedSlice { start, step, len })
        } else {
            // Walking backwards; -1 is the "before the first element" sentinel.
            let last = sub(dim, 1);
            let start = match self.start {
                Bound::Included(x) => clamp_int(wrap_index(x, dim), -1, last),
                Bound::Excluded(x) => wrap_dec_clamped(x, dim, last),
                Bound::Unbounded => last,
            };
            let stop = match self.stop {
                Bound::Included(x) => wrap_dec_clamped(x, dim, last),
                Bound::Excluded(x) => clamp_int(wrap_index(x, dim), -1, last),
                Bound::Unbounded => -1,
            };
            let len = if start <= stop {
                0
            } else {
                ceil_div(sub(stop, start), step)
            };
            Some(ResolvedSlice { start, step, len })
        }
    }
}

macro_rules! impl_slice_from_range {
    ($($range:ty),+ $(,)?) => {
        $(impl From<$range> for Slice {
            fn from(range: $range) -> Self {
                Self::from_slice(range)
            }
        })+
    };
}

impl_slice_from_range!(
    int,
    Range<int>,
    RangeInclusive<int>,
    RangeFrom<int>,
    RangeTo<int>,
    RangeToInclusive<int>,
    RangeFull,
    (Bound, Bound),
    (CoreBound<int>, CoreBound<int>),
);

/// Builds a [`Slice`] using Python-like range syntax in const or runtime contexts.
///
/// ```
/// # #![feature(const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::slice;
/// use noodle_cute::slice::{Bound, Slice};
///
/// const S: Slice = slice!([1..5] : 2);
/// assert_eq!(S, Slice::new(1..5, 2));
/// assert_eq!(slice!([-4]), Slice::new(-4, 1));
/// assert_eq!(slice!([..=2]), Slice { start: Bound::Unbounded, stop: Bound::Included(2), step: 1 });
/// ```
#[macro_export]
macro_rules! slice {
    ([$range:expr] : $step:expr) => {
        $crate::slice::Slice::new($range, $step)
    };
    ([$value:expr]) => {{
        let value = $value;
        $crate::slice::Slice::from_slice(value)
    }};
}

/// Builds a homogeneous `[Slice; N]` array from comma-separated range specifications.
///
/// ```
/// # #![feature(const_trait_impl)]
/// # #![allow(incomplete_features)]
/// use noodle_cute::slices;
/// use noodle_cute::slice::Slice;
///
/// const S: [Slice; 3] = slices!(1..4, ..: -1, -2);
/// assert_eq!(S[1].step, -1);
/// ```
#[macro_export]
macro_rules! slices {
    // TT-muncher: first bracket accumulates parsed `Slice` items; second collects current range tokens.
    (@parse [] []) => {
        [] as [$crate::slice::Slice; 0]
    };
    (@parse [$($out:tt)*] []) => {
        [$($out)*]
    };
    (@parse [$($out:tt)*] [$($range:tt)+] : $step:expr, $($rest:tt)*) => {
        $crate::slices!(@parse
            [$($out)* $crate::slice!([$($range)+] : $step),]
            []
            $($rest)*
        )
    };
    (@parse [$($out:tt)*] [$($range:tt)+] : $step:expr) => {
        $crate::slices!(@parse [$($out)* $crate::slice!([$($range)+] : $step),] [])
    };
    (@parse [$($out:tt)*] [$($range:tt)+] , $($rest:tt)*) => {
        $crate::slices!(@parse
            [$($out)* $crate::slice!([$($range)+]),]
            []
            $($rest)*
        )
    };
    (@parse [$($out:tt)*] [$($range:tt)+]) => {
        $crate::slices!(@parse [$($out)* $crate::slice!([$($range)+]),] [])
    };
    (@parse [$($out:tt)*] [$($range:tt)*] $next:tt $($rest:tt)*) => {
        $crate::slices!(@parse
            [$($out)*]
            [$($range)* $next]
            $($rest)*
        )
    };

    ($($input:tt)*) => {
        $crate::slices!(@parse [] [] $($input)*)
    };
}
