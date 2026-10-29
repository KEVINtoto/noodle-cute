/// Selects an entire top-level mode or fixes it at one bounded coordinate.
/// Used by [`DynLayout::slice`](crate::layout::DynLayout::slice); strided ranges are not supported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slice {
    /// Retains the mode, preserving its order relative to other retained modes.
    All,
    /// Fixes the mode at this coordinate and removes it from the output shape.
    Index(isize),
}
