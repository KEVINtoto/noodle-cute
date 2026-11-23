#![feature(const_trait_impl)]
#![allow(incomplete_features)]

use noodle_cute::integer::int;
use noodle_cute::slice::{Bound, ResolvedSlice, Slice};
use noodle_cute::{slice, slices};

use Bound::{Excluded, Included, Unbounded};

#[test]
fn mixed_slices_preserve_bounds_and_steps() {
    let end = 10;
    let stride = -2;
    let slices: [Slice; 7] = slices!(1..5, ..-1:2, -4, .., 2..=end, 3.., ..=end:stride + 1,);
    let expected = [
        (Included(1), Excluded(5), 1),
        (Unbounded, Excluded(-1), 2),
        (Included(-4), Included(-4), 1),
        (Unbounded, Unbounded, 1),
        (Included(2), Included(10), 1),
        (Included(3), Unbounded, 1),
        (Unbounded, Included(10), -1),
    ];
    for (slice, (start, stop, step)) in slices.into_iter().zip(expected) {
        assert_eq!(slice, Slice { start, stop, step });
    }
}

#[test]
fn empty_and_single_element_arrays() {
    let empty = slices!();
    assert_eq!(empty, [] as [Slice; 0]);
    let _: [Slice; 1] = slices!(4);
    let _: [Slice; 1] = slices!(..: -1);
    let _: [Slice; 1] = slices!(..,);
}

#[test]
fn expressions_are_evaluated_once_in_order() {
    let mut calls = 0;
    let mut next = || {
        calls += 1;
        calls
    };
    let slices = slices!((next()..next()):next(), next());
    assert_eq!(calls, 4);
    assert_eq!(
        slices[0],
        Slice {
            start: Included(1),
            stop: Excluded(2),
            step: 3
        }
    );
    assert_eq!(slices[1], Slice::from(4));
}

#[test]
fn resolve_follows_python_semantics() {
    let resolve = |s: Slice, dim| {
        let r = s.resolve(dim).unwrap();
        (r.start, r.step, r.len)
    };
    assert_eq!(resolve(slice!([1..5]), 10), (1, 1, 4));
    assert_eq!(resolve(slice!([..-1] : 2), 10), (0, 2, 5));
    assert_eq!(resolve(slice!([-4]), 10), (6, 1, 1));
    assert_eq!(resolve(slice!([3..3]), 10), (3, 1, 0));
    assert_eq!(resolve(slice!([2..100]), 5), (2, 1, 3));
    assert_eq!(resolve(slice!([..] : -1), 5), (4, -1, 5));
    assert_eq!(resolve(slice!([..] : -2), 5), (4, -2, 3));
    assert_eq!(resolve(slice!([..=1] : -1), 5), (4, -1, 4));
    assert_eq!(resolve(slice!([..] : -1), 0), (-1, -1, 0));
    assert!(slice!([..] : 0).resolve(5).is_none());
}

#[test]
fn resolve_handles_const_contexts_and_extreme_bounds() {
    const SLICES: [Slice; 3] = slices!(1..5, ..=2: -1, -3);
    const RESOLVED: ResolvedSlice = slice!([..] : -2).resolve(5).unwrap();
    assert_eq!(SLICES[0].get(), (Included(1), Excluded(5), 1));
    assert_eq!(
        RESOLVED,
        ResolvedSlice {
            start: 4,
            step: -2,
            len: 3
        }
    );

    assert_eq!(
        slice!([0..] : int::MIN).resolve(2),
        Some(ResolvedSlice {
            start: 0,
            step: int::MIN,
            len: 1
        })
    );
    assert_eq!(
        slice!([1..] : int::MIN).resolve(2),
        Some(ResolvedSlice {
            start: 1,
            step: int::MIN,
            len: 1
        })
    );
    assert_eq!(
        slice!([1..] : int::MAX).resolve(2),
        Some(ResolvedSlice {
            start: 1,
            step: int::MAX,
            len: 1
        })
    );
    assert_eq!(
        slice!([..]).resolve(int::MAX),
        Some(ResolvedSlice {
            start: 0,
            step: 1,
            len: int::MAX
        })
    );
    assert_eq!(
        slice!([..] : -1).resolve(int::MAX),
        Some(ResolvedSlice {
            start: int::MAX - 1,
            step: -1,
            len: int::MAX,
        })
    );
}
