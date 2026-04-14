//! Aligned buffer allocation helpers for SIMD operations.
//!
//! SIMD instruction sets impose strict data-alignment requirements.  Violating
//! them results either in a performance penalty (unaligned loads on newer µarchs)
//! or a `#GP` fault / `SIGBUS` at runtime (older SSE on some platforms).
//!
//! ## Alignment requirements by SIMD tier
//!
//! | Tier      | Register width | Required alignment |
//! |-----------|---------------|--------------------|
//! | SSE4.2    | 128-bit       | **16 bytes**       |
//! | AVX2      | 256-bit       | **32 bytes**       |
//! | AVX-512   | 512-bit       | **64 bytes**       |
//! | NEON      | 128-bit       | **16 bytes**       |
//!
//! Use the alignment constant that matches the SIMD tier you intend to target:
//!
//! ```rust
//! use oximedia_simd::aligned_alloc::AlignedBuffer;
//!
//! // SSE4.2 / NEON — 16-byte alignment
//! let sse = AlignedBuffer::new(128, 16).expect("SSE/NEON buffer");
//! assert_eq!(sse.as_ptr() as usize % 16, 0);
//!
//! // AVX2 — 32-byte alignment
//! let avx2 = AlignedBuffer::new(256, 32).expect("AVX2 buffer");
//! assert_eq!(avx2.as_ptr() as usize % 32, 0);
//!
//! // AVX-512 — 64-byte alignment
//! let avx512 = AlignedBuffer::new(512, 64).expect("AVX-512 buffer");
//! assert_eq!(avx512.as_ptr() as usize % 64, 0);
//! ```
//!
//! Standard `Vec<u8>` allocations only guarantee 1-byte alignment (the
//! alignment of `u8`), which is insufficient for any SIMD tier.  This module
//! provides [`AlignedBuffer`], a heap-allocated byte buffer with a
//! caller-specified alignment guarantee.  The implementation uses Rust's
//! [`std::alloc::Layout`] with the requested alignment and wraps the raw
//! allocation in a safe API.
//!
//! # Examples
//!
//! ```
//! use oximedia_simd::aligned_alloc::AlignedBuffer;
//!
//! // Allocate a 1024-byte buffer aligned to 64 bytes (AVX-512).
//! let buf = AlignedBuffer::new(1024, 64).expect("allocation should succeed");
//! assert_eq!(buf.len(), 1024);
//! assert_eq!(buf.as_ptr() as usize % 64, 0);
//! ```

use crate::{Result, SimdError};

/// A heap-allocated byte buffer with a guaranteed minimum alignment.
///
/// The buffer is zero-initialised on creation.  It implements [`AsRef<[u8]>`]
/// and [`AsMut<[u8]>`] for ergonomic access.
///
/// # Drop
///
/// The underlying allocation is freed via [`std::alloc::dealloc`] when the
/// buffer is dropped.
pub struct AlignedBuffer {
    /// Raw pointer to the allocation.
    ptr: *mut u8,
    /// Layout used for deallocation.
    layout: std::alloc::Layout,
    /// Usable length in bytes.
    len: usize,
}

// SAFETY: AlignedBuffer owns its allocation exclusively — no shared mutable
// state — so it is safe to send across threads.
//
// We allow `unsafe_code` at the crate level (via Cargo.toml lints) for this
// specific use case: raw allocation with `std::alloc` requires `unsafe` for
// `alloc`/`dealloc`.  All invariants (non-null ptr, matching layout) are
// upheld internally.

impl AlignedBuffer {
    /// Allocate a zero-initialised buffer of `len` bytes with the given
    /// `alignment`.
    ///
    /// `alignment` must be a non-zero power of two.  Common values:
    /// - 16 for NEON / SSE
    /// - 32 for AVX2
    /// - 64 for AVX-512
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::InvalidAlignment`] if `alignment` is zero or not a
    /// power of two, or if the allocation fails.
    pub fn new(len: usize, alignment: usize) -> Result<Self> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(SimdError::InvalidAlignment);
        }
        if len == 0 {
            // Zero-length allocation: return a dangling but aligned pointer.
            return Ok(Self {
                ptr: alignment as *mut u8, // non-null sentinel, never dereferenced
                layout: std::alloc::Layout::from_size_align(0, alignment)
                    .map_err(|_| SimdError::InvalidAlignment)?,
                len: 0,
            });
        }

        let layout = std::alloc::Layout::from_size_align(len, alignment)
            .map_err(|_| SimdError::InvalidAlignment)?;

        // SAFETY: layout has non-zero size (checked above) and valid alignment.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(SimdError::InvalidBufferSize);
        }

        Ok(Self { ptr, layout, len })
    }

    /// Allocate a 64-byte-aligned buffer (AVX-512 optimal).
    ///
    /// Convenience wrapper around `AlignedBuffer::new(len, 64)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation fails.
    pub fn new_avx512(len: usize) -> Result<Self> {
        Self::new(len, 64)
    }

    /// Allocate a 32-byte-aligned buffer (AVX2 optimal).
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation fails.
    pub fn new_avx2(len: usize) -> Result<Self> {
        Self::new(len, 32)
    }

    /// Allocate a 16-byte-aligned buffer (NEON / SSE optimal).
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation fails.
    pub fn new_neon(len: usize) -> Result<Self> {
        Self::new(len, 16)
    }

    /// Returns the usable length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer has zero length.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the alignment of the buffer in bytes.
    #[must_use]
    pub fn alignment(&self) -> usize {
        self.layout.align()
    }

    /// Returns a raw const pointer to the buffer.
    #[must_use]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Returns a raw mutable pointer to the buffer.
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Returns the buffer contents as a byte slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: ptr is valid for `len` bytes and properly aligned.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns the buffer contents as a mutable byte slice.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: ptr is valid for `len` bytes and properly aligned.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Copy data from a byte slice into the buffer.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::InvalidBufferSize`] if `data` is larger than the
    /// buffer.
    pub fn copy_from_slice(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > self.len {
            return Err(SimdError::InvalidBufferSize);
        }
        self.as_mut_slice()[..data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Fill the buffer with a constant byte value.
    pub fn fill(&mut self, value: u8) {
        if self.len > 0 {
            self.as_mut_slice().fill(value);
        }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.len > 0 && !self.ptr.is_null() {
            // SAFETY: ptr was allocated with the same layout, and len > 0
            // guarantees a real allocation (not the zero-length sentinel).
            unsafe {
                std::alloc::dealloc(self.ptr, self.layout);
            }
        }
    }
}

impl AsRef<[u8]> for AlignedBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for AlignedBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl std::fmt::Debug for AlignedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuffer")
            .field("len", &self.len)
            .field("alignment", &self.layout.align())
            .finish()
    }
}

// SAFETY: AlignedBuffer owns its data exclusively (no interior mutability or
// shared references).  Sending/sharing the buffer across threads is safe.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_avx512_alignment() {
        let buf = AlignedBuffer::new_avx512(1024).expect("allocation");
        assert_eq!(buf.len(), 1024);
        assert_eq!(buf.alignment(), 64);
        assert_eq!(buf.as_ptr() as usize % 64, 0);
    }

    #[test]
    fn new_avx2_alignment() {
        let buf = AlignedBuffer::new_avx2(512).expect("allocation");
        assert_eq!(buf.alignment(), 32);
        assert_eq!(buf.as_ptr() as usize % 32, 0);
    }

    #[test]
    fn new_neon_alignment() {
        let buf = AlignedBuffer::new_neon(256).expect("allocation");
        assert_eq!(buf.alignment(), 16);
        assert_eq!(buf.as_ptr() as usize % 16, 0);
    }

    #[test]
    fn zero_initialised() {
        let buf = AlignedBuffer::new_avx512(128).expect("allocation");
        for &b in buf.as_slice() {
            assert_eq!(b, 0, "buffer should be zero-initialised");
        }
    }

    #[test]
    fn fill_and_read() {
        let mut buf = AlignedBuffer::new(64, 64).expect("allocation");
        buf.fill(0xAB);
        for &b in buf.as_slice() {
            assert_eq!(b, 0xAB);
        }
    }

    #[test]
    fn copy_from_slice_valid() {
        let mut buf = AlignedBuffer::new(32, 32).expect("allocation");
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        buf.copy_from_slice(&data).expect("copy");
        assert_eq!(&buf.as_slice()[..8], &data);
        // Remaining bytes should still be zero
        for &b in &buf.as_slice()[8..] {
            assert_eq!(b, 0);
        }
    }

    #[test]
    fn copy_from_slice_too_large() {
        let mut buf = AlignedBuffer::new(4, 16).expect("allocation");
        let data = [0u8; 8];
        let result = buf.copy_from_slice(&data);
        assert_eq!(result, Err(SimdError::InvalidBufferSize));
    }

    #[test]
    fn zero_length_buffer() {
        let buf = AlignedBuffer::new(0, 64).expect("zero-length allocation");
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.as_slice().len(), 0);
    }

    #[test]
    fn invalid_alignment_zero() {
        let result = AlignedBuffer::new(64, 0);
        assert_eq!(result.err(), Some(SimdError::InvalidAlignment));
    }

    #[test]
    fn invalid_alignment_not_power_of_two() {
        let result = AlignedBuffer::new(64, 3);
        assert_eq!(result.err(), Some(SimdError::InvalidAlignment));
        let result2 = AlignedBuffer::new(64, 48);
        assert_eq!(result2.err(), Some(SimdError::InvalidAlignment));
    }

    #[test]
    fn as_mut_slice_writes() {
        let mut buf = AlignedBuffer::new(16, 16).expect("allocation");
        let slice = buf.as_mut_slice();
        for (i, b) in slice.iter_mut().enumerate() {
            *b = (i * 17) as u8;
        }
        for (i, &b) in buf.as_slice().iter().enumerate() {
            assert_eq!(b, (i * 17) as u8);
        }
    }

    #[test]
    fn debug_format() {
        let buf = AlignedBuffer::new(128, 64).expect("allocation");
        let dbg = format!("{buf:?}");
        assert!(dbg.contains("128"), "debug should show length");
        assert!(dbg.contains("64"), "debug should show alignment");
    }

    #[test]
    fn multiple_alignments() {
        // Test various valid power-of-two alignments
        for align in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
            let buf = AlignedBuffer::new(256, align)
                .unwrap_or_else(|_| panic!("allocation with align={align}"));
            assert_eq!(
                buf.as_ptr() as usize % align,
                0,
                "alignment {align} not satisfied"
            );
        }
    }

    #[test]
    fn as_ref_as_mut_trait() {
        let mut buf = AlignedBuffer::new(32, 32).expect("allocation");
        let slice_ref: &[u8] = buf.as_ref();
        assert_eq!(slice_ref.len(), 32);
        let slice_mut: &mut [u8] = buf.as_mut();
        slice_mut[0] = 42;
        assert_eq!(buf.as_slice()[0], 42);
    }

    #[test]
    fn large_buffer_alignment() {
        // Allocate a larger buffer to ensure alignment holds beyond small allocs
        let buf = AlignedBuffer::new(65536, 64).expect("large allocation");
        assert_eq!(buf.len(), 65536);
        assert_eq!(buf.as_ptr() as usize % 64, 0);
        // Spot-check it's all zeros
        assert_eq!(buf.as_slice()[0], 0);
        assert_eq!(buf.as_slice()[65535], 0);
    }
}
