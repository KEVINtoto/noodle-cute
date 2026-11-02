use noodle_cute::layout::{Layout, Shape, StrideOf};
use noodle_cute::tuple::{Item, LeafProfile};
use noodle_cute::{Error, Int, Result};

#[derive(Clone, Debug)]
struct RuntimeLeaf(isize);

impl Item for RuntimeLeaf {
    type Profile = LeafProfile;
    const RANK: usize = 1;
    const DEPTH: usize = 0;
    const LEAVES: usize = 1;

    fn leaf(&self, index: usize) -> Result<isize> {
        if index == 0 {
            Ok(self.0)
        } else {
            Err(Error::AxisOutOfBounds)
        }
    }

    fn leaf_static(&self, index: usize) -> Result<Option<isize>> {
        if index == 0 {
            Ok(None)
        } else {
            Err(Error::AxisOutOfBounds)
        }
    }

    fn mode_span(&self, axis: usize) -> Result<(usize, usize)> {
        if axis == 0 {
            Ok((0, 1))
        } else {
            Err(Error::AxisOutOfBounds)
        }
    }
}

impl Shape for RuntimeLeaf {}

fn layout<S: Shape, D: StrideOf<S>>(shape: S, stride: D) -> Layout<S, D> {
    Layout::new(shape, stride).unwrap()
}

pub fn main() {
    let l = layout(RuntimeLeaf(4), Int::<2>);
    assert_eq!(l.size(), Ok(4));
    assert_eq!(l.cosize(), Ok(7));
    assert_eq!(l.index(3), Ok(6));
}
