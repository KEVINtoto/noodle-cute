use noodle_cute::layout::{
    AffineLayoutMap, Compact, DynLayout, Layout, LayoutRight, Shape, StrideOf,
};
use noodle_cute::{DynInt, Int, shape, stride};

// Layout metadata only: logical order is row-major like NumPy, and a "copy" is a fresh compact layout.
// `L` is a runtime `DynLayout` or a static/mixed `Layout<S, D>`; shape-changing views need runtime
// axes and so exist only for `DynLayout`. Use `to_dyn` to leave the typed world.
struct NDArray<L = DynLayout> {
    layout: L,
}

fn to_isize(values: &[usize]) -> Vec<isize> {
    values
        .iter()
        .map(|&v| isize::try_from(v).expect("extent does not fit in isize"))
        .collect()
}

fn check_permutation(axes: &[usize], ndim: usize) {
    let mut seen = vec![false; ndim];
    assert_eq!(axes.len(), ndim, "axes must name every dimension once");
    for &axis in axes {
        assert!(axis < ndim, "axis {axis} is out of bounds for ndim {ndim}");
        assert!(!seen[axis], "axis {axis} appears more than once");
        seen[axis] = true;
    }
}

impl<L: AffineLayoutMap> NDArray<L> {
    fn from_layout(layout: L) -> Self {
        Self { layout }
    }

    // Typed layouts can only export flat leaves, so nesting is not visible to the queries below.
    fn flat(&self) -> DynLayout {
        self.layout
            .as_flat_layout()
            .expect("layout exceeds MAX_RANK")
    }

    fn shape(&self) -> Vec<isize> {
        self.flat().shapes().to_vec()
    }

    fn strides(&self) -> Vec<isize> {
        self.flat().strides().to_vec()
    }

    fn ndim(&self) -> usize {
        self.layout.leaf_count()
    }

    fn size(&self) -> usize {
        self.layout.size().expect("size overflows") as usize
    }

    fn is_contiguous(&self) -> bool {
        if self.size() == 0 {
            return true;
        }
        let flat = self.flat();
        let mut expected = 1;
        // Strides of unit extents are never used to reach another element.
        for (&extent, &stride) in flat.shapes().iter().zip(flat.strides()).rev() {
            if extent != 1 && stride != expected {
                return false;
            }
            expected *= extent;
        }
        true
    }

    fn flatten(&self) -> NDArray {
        let layout = DynLayout::right(&[self.size() as isize]).expect("invalid shape");
        NDArray { layout }
    }

    fn to_dyn(&self) -> NDArray {
        NDArray {
            layout: self.flat(),
        }
    }
}

// Row-major compact layout of a static or mixed shape; static extents stay in the type.
impl<S: Shape + Compact<LayoutRight, Int<1>>> NDArray<Layout<S, S::Stride>> {
    fn with_shape(shape: S) -> Self {
        Self::from_layout(LayoutRight::make(shape).expect("invalid shape"))
    }
}

impl<S: Shape + Compact<LayoutRight, Int<1>>, D: StrideOf<S>> NDArray<Layout<S, D>> {
    fn compact(&self) -> NDArray<Layout<S, S::Stride>> {
        NDArray::with_shape(self.layout.shape.clone())
    }
}

impl NDArray {
    fn from_parts(shape: &[isize], strides: &[isize]) -> Self {
        let layout = DynLayout::new(shape, strides).expect("invalid shape or strides");
        Self { layout }
    }

    fn new(shape: &[usize]) -> Self {
        let layout = DynLayout::right(&to_isize(shape)).expect("invalid shape");
        Self { layout }
    }

    fn compact(&self) -> NDArray {
        let layout = DynLayout::right(self.layout.shapes()).expect("invalid shape");
        Self { layout }
    }

    // Returns a view when the strides allow it (NumPy `_attempt_nocopy_reshape`), otherwise
    // the compact layout a copy would have.
    fn reshape(&self, shape: &[usize]) -> NDArray {
        let new = to_isize(shape);
        let new_size = new.iter().try_fold(1isize, |a, &e| a.checked_mul(e));
        assert_eq!(
            new_size,
            Some(self.size() as isize),
            "cannot reshape array of size {} into shape {shape:?}",
            self.size()
        );
        if self.size() == 0 {
            return Self::new(shape);
        }

        let (old, old_strides): (Vec<_>, Vec<_>) = self
            .shape()
            .iter()
            .zip(self.layout.strides())
            .filter(|&(&e, _)| e != 1)
            .map(|(&e, &d)| (e, d))
            .unzip();

        let mut strides = vec![0; new.len()];
        let (mut ni, mut nj, mut oi, mut oj) = (0, 1, 0, 1);
        while ni < new.len() && oi < old.len() {
            let (mut np, mut op) = (new[ni], old[oi]);
            while np != op {
                if np < op {
                    np *= new[nj];
                    nj += 1;
                } else {
                    op *= old[oj];
                    oj += 1;
                }
            }
            // The merged old axes must be traversable with a single stride.
            if (oi..oj - 1).any(|k| old_strides[k] != old_strides[k + 1] * old[k + 1]) {
                return Self::new(shape);
            }
            strides[nj - 1] = old_strides[oj - 1];
            for k in (ni + 1..nj).rev() {
                strides[k - 1] = strides[k] * new[k];
            }
            (ni, nj, oi, oj) = (nj, nj + 1, oj, oj + 1);
        }
        let tail = if ni > 0 { strides[ni - 1] } else { 1 };
        strides[ni..].fill(tail);
        Self::from_parts(&new, &strides)
    }

    // Like `numpy.transpose`: no axes reverses the dimension order.
    fn transpose(&self, axes: &[usize]) -> NDArray {
        if axes.is_empty() {
            let reversed: Vec<usize> = (0..self.ndim()).rev().collect();
            self.permute(&reversed)
        } else {
            self.permute(axes)
        }
    }

    fn permute(&self, axes: &[usize]) -> NDArray {
        check_permutation(axes, self.ndim());
        let shape: Vec<isize> = axes.iter().map(|&a| self.layout.shapes()[a]).collect();
        let strides: Vec<isize> = axes.iter().map(|&a| self.layout.strides()[a]).collect();
        Self::from_parts(&shape, &strides)
    }

    fn squeeze(&self) -> NDArray {
        let (shape, strides): (Vec<_>, Vec<_>) = self
            .shape()
            .iter()
            .zip(self.layout.strides())
            .filter(|&(&e, _)| e != 1)
            .map(|(&e, &d)| (e, d))
            .unzip();
        Self::from_parts(&shape, &strides)
    }

    fn unsqueeze(&self, axis: usize) -> NDArray {
        assert!(
            axis <= self.ndim(),
            "axis {axis} is out of bounds for ndim {}",
            self.ndim() + 1
        );
        let stride = match (
            self.layout.shapes().get(axis),
            self.layout.strides().get(axis),
        ) {
            (Some(&e), Some(&d)) => e * d,
            _ => 1,
        };
        let mut shape = self.layout.shapes().to_vec();
        let mut strides = self.layout.strides().to_vec();
        shape.insert(axis, 1);
        strides.insert(axis, stride);
        Self::from_parts(&shape, &strides)
    }

    fn broadcast_to(&self, shape: &[usize]) -> NDArray {
        let new = to_isize(shape);
        let lead = new
            .len()
            .checked_sub(self.ndim())
            .expect("cannot broadcast to fewer dimensions");
        let mut strides = vec![0; new.len()];
        for (i, (&extent, &stride)) in self
            .layout
            .shapes()
            .iter()
            .zip(self.layout.strides())
            .enumerate()
        {
            let target = new[lead + i];
            if extent == target {
                strides[lead + i] = stride;
            } else {
                assert_eq!(
                    extent, 1,
                    "cannot broadcast extent {extent} to {target} on axis {i}"
                );
            }
        }
        Self::from_parts(&new, &strides)
    }
}

fn main() {
    let a = NDArray::new(&[2, 3, 4]);
    assert_eq!(a.shape(), [2, 3, 4]);
    assert_eq!(a.strides(), [12, 4, 1]);
    assert_eq!((a.ndim(), a.size()), (3, 24));
    assert!(a.is_contiguous());

    let t = a.transpose(&[]);
    assert_eq!(t.shape(), [4, 3, 2]);
    assert_eq!(t.strides(), [1, 4, 12]);
    assert!(!t.is_contiguous());

    let p = a.permute(&[1, 0, 2]);
    assert_eq!(p.shape(), [3, 2, 4]);
    assert_eq!(p.strides(), [4, 12, 1]);

    // Contiguous arrays reshape as views; a transposed one cannot merge its axes.
    let r = a.reshape(&[6, 4]);
    assert_eq!(r.strides(), [4, 1]);
    let r = a.reshape(&[2, 12]);
    assert_eq!(r.strides(), [12, 1]);
    assert!(a.transpose(&[1, 0, 2]).reshape(&[2, 3, 4]).is_contiguous());
    let split = t.reshape(&[2, 2, 3, 2]);
    assert_eq!(split.strides(), [2, 1, 4, 12]);
    let copied = t.reshape(&[4, 6]);
    assert_eq!(copied.strides(), [6, 1]);
    assert!(copied.is_contiguous());

    assert_eq!(t.compact().strides(), [6, 2, 1]);
    assert_eq!(t.flatten().shape(), [24]);
    assert_eq!(t.flatten().strides(), [1]);

    let u = NDArray::new(&[3, 1, 4]).unsqueeze(3).unsqueeze(0);
    assert_eq!(u.shape(), [1, 3, 1, 4, 1]);
    assert_eq!(u.squeeze().shape(), [3, 4]);
    assert_eq!(u.squeeze().strides(), [4, 1]);

    let b = NDArray::new(&[3, 1]).broadcast_to(&[2, 3, 5]);
    assert_eq!(b.shape(), [2, 3, 5]);
    assert_eq!(b.strides(), [0, 1, 0]);
    assert!(!b.is_contiguous());

    // Static and mixed layouts: static extents and strides stay in the type.
    let s = NDArray::with_shape(shape![Int::<2>, DynInt(3), Int::<4>]);
    assert_eq!(s.shape(), [2, 3, 4]);
    assert_eq!(s.strides(), [12, 4, 1]);
    assert!(s.is_contiguous());

    let column_major = Layout::new(shape![Int::<2>, DynInt(3)], stride![Int::<1>, Int::<2>])
        .expect("valid layout");
    let m = NDArray::from_layout(column_major);
    assert!(!m.is_contiguous());
    let c = m.compact();
    assert_eq!(c.strides(), [3, 1]);
    assert!(c.is_contiguous());

    // Runtime shape changes go through the dynamic form.
    assert_eq!(m.flatten().shape(), [6]);
    assert_eq!(m.to_dyn().transpose(&[]).strides(), [2, 1]);
}
