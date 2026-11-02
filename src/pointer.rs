//! Iterators a tensor advances by layout offsets: raw pointers, memory-space
//! tagged pointers, and coordinate iterators (`pointer.hpp`, `pointer_base.hpp`,
//! `arithmetic_tuple.hpp`).
//!
//! Advancing an iterator is safe address arithmetic; reading or writing through a
//! pointer is `unsafe`, because a layout cannot prove that an address is valid,
//! aligned, or free of data races between threads. The memory-space tags only
//! record where a pointer lives so that copy and MMA backends can dispatch on it.
//!
//! ```
//! use noodle_cute::pointer::{Advance, Read, make_gmem_ptr};
//! use noodle_cute::Int;
//!
//! let data = [10, 11, 12, 13];
//! let it = make_gmem_ptr(data.as_ptr()).advance(Int::<2>);
//! assert_eq!(unsafe { it.read() }, 12);
//! ```

use crate::basis::OffsetAdd;
use crate::int::Integer;

/// A position with a value type: CuTe's iterator concept.
pub trait Iter: Copy {
    /// The value produced by dereferencing the iterator.
    type Value;
}

/// Advances an iterator by a layout offset: CuTe `iter + offset`.
pub trait Advance<O>: Iter {
    /// The advanced iterator.
    type Output: Iter;
    /// Returns the iterator `offset` elements further, with wrapping pointer arithmetic.
    fn advance(self, offset: O) -> Self::Output;
}

/// Reads the value at an iterator: CuTe `*iter`.
pub trait Read: Iter {
    /// Reads the value.
    ///
    /// # Safety
    ///
    /// For pointers, the address must be valid for reads and properly aligned,
    /// and no other thread may write it concurrently.
    unsafe fn read(self) -> Self::Value;
}

/// Writes the value at an iterator: CuTe `*iter = value`.
pub trait Write: Iter {
    /// Writes the value without dropping the previous one.
    ///
    /// # Safety
    ///
    /// The address must be valid for writes and properly aligned, and no other
    /// thread may access it concurrently.
    unsafe fn write(self, value: Self::Value);
}

/// Reinterprets the element type of a pointer: CuTe `recast_ptr<N>(ptr)`.
pub trait RecastPtr<N>: Iter {
    /// The pointer to `N` at the same address.
    type Output: Iter<Value = N>;
    /// Casts the pointer.
    fn recast_ptr(self) -> Self::Output;
}

/// The address of a pointer, for runtime alignment checks.
pub trait Address: Iter {
    /// Returns the address.
    fn addr(self) -> usize;
}

impl<T> Address for *const T {
    fn addr(self) -> usize {
        self.addr()
    }
}

impl<T> Address for *mut T {
    fn addr(self) -> usize {
        self.addr()
    }
}

impl<T> Iter for *const T {
    type Value = T;
}

impl<T> Iter for *mut T {
    type Value = T;
}

impl<T, O: Integer> Advance<O> for *const T {
    type Output = Self;
    fn advance(self, offset: O) -> Self {
        self.wrapping_offset(offset.value())
    }
}

impl<T, O: Integer> Advance<O> for *mut T {
    type Output = Self;
    fn advance(self, offset: O) -> Self {
        self.wrapping_offset(offset.value())
    }
}

impl<T: Copy> Read for *const T {
    unsafe fn read(self) -> T {
        // SAFETY: forwarded to the caller of `Read::read`.
        unsafe { core::ptr::read(self) }
    }
}

impl<T: Copy> Read for *mut T {
    unsafe fn read(self) -> T {
        // SAFETY: forwarded to the caller of `Read::read`.
        unsafe { core::ptr::read(self) }
    }
}

impl<T: Copy> Write for *mut T {
    unsafe fn write(self, value: T) {
        // SAFETY: forwarded to the caller of `Write::write`.
        unsafe { core::ptr::write(self, value) }
    }
}

impl<T, N> RecastPtr<N> for *const T {
    type Output = *const N;
    fn recast_ptr(self) -> *const N {
        self.cast()
    }
}

impl<T, N> RecastPtr<N> for *mut T {
    type Output = *mut N;
    fn recast_ptr(self) -> *mut N {
        self.cast()
    }
}

macro_rules! memory_space {
    ($(#[$doc:meta])* $name:ident, $make:ident, $upstream:literal) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(transparent)]
        pub struct $name<P>(pub P);

        #[doc = concat!("Tags a pointer with its memory space: CuTe `", $upstream, "`.")]
        pub fn $make<P: Iter>(ptr: P) -> $name<P> {
            $name(ptr)
        }

        impl<P: Iter> Iter for $name<P> {
            type Value = P::Value;
        }

        impl<P: Advance<O>, O> Advance<O> for $name<P> {
            type Output = $name<P::Output>;
            fn advance(self, offset: O) -> Self::Output {
                $name(self.0.advance(offset))
            }
        }

        impl<P: Read> Read for $name<P> {
            unsafe fn read(self) -> P::Value {
                // SAFETY: forwarded to the caller of `Read::read`.
                unsafe { self.0.read() }
            }
        }

        impl<P: Write> Write for $name<P> {
            unsafe fn write(self, value: P::Value) {
                // SAFETY: forwarded to the caller of `Write::write`.
                unsafe { self.0.write(value) }
            }
        }

        impl<P: Address> Address for $name<P> {
            fn addr(self) -> usize {
                self.0.addr()
            }
        }

        impl<P: RecastPtr<N>, N> RecastPtr<N> for $name<P> {
            type Output = $name<P::Output>;
            fn recast_ptr(self) -> Self::Output {
                $name(self.0.recast_ptr())
            }
        }
    };
}

memory_space!(
    /// A pointer into global memory: CuTe `gmem_ptr<P>`.
    GmemPtr,
    make_gmem_ptr,
    "make_gmem_ptr"
);
memory_space!(
    /// A pointer into shared memory: CuTe `smem_ptr<P>`.
    SmemPtr,
    make_smem_ptr,
    "make_smem_ptr"
);
memory_space!(
    /// A pointer into registers: CuTe `rmem_ptr<P>`.
    RmemPtr,
    make_rmem_ptr,
    "make_rmem_ptr"
);

/// An iterator whose value is a coordinate: CuTe `ArithmeticTupleIterator`.
///
/// Advancing adds a coordinate offset mode by mode, so an identity tensor built
/// on this iterator returns each coordinate itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct CoordIter<C>(pub C);

impl<C: Copy> Iter for CoordIter<C> {
    type Value = C;
}

impl<C, O> Advance<O> for CoordIter<C>
where
    C: Copy + OffsetAdd<O>,
    C::Output: Copy,
{
    type Output = CoordIter<C::Output>;
    fn advance(self, offset: O) -> Self::Output {
        CoordIter(self.0.offset_add(offset))
    }
}
