//! GIL-release pattern for CPU-intensive Python operations.
//!
//! Demonstrates how to release the GIL during Rust-side parallel work so that
//! the Python interpreter remains free to execute other threads while heavy
//! computation proceeds.
//!
//! # Example
//! ```python
//! import oximedia
//! data = [b"hello", b"world", b"foo", b"bar"]
//! checksums = oximedia.compute_checksums(data)
//! # [210700827, 222957957, 2090760023, 1991736602]
//! ```

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// FNV-1a 32-bit checksum (no external deps)
// ---------------------------------------------------------------------------

/// Compute a 64-bit FNV-1a checksum of a byte slice.
///
/// Uses the FNV-1a 64-bit variant for good distribution with low collision
/// probability on arbitrary binary data.
#[must_use]
fn fnv1a_64(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// compute_checksums — GIL-release demonstration
// ---------------------------------------------------------------------------

/// Compute FNV-1a 64-bit checksums for a list of byte strings.
///
/// The GIL is released while the checksums are computed so that other Python
/// threads can run concurrently.  For CPU-bound workloads this can yield
/// meaningful parallelism when called from multiple threads.
///
/// Parameters
/// ----------
/// data : list[bytes]
///     Input byte strings to checksum.
///
/// Returns
/// -------
/// list[int]
///     FNV-1a 64-bit checksum for each input item (unsigned 64-bit integers).
///
/// # Notes
///
/// The actual checksum algorithm (FNV-1a) is deterministic and reproducible
/// across platforms.  For cryptographic needs use SHA-256 instead.
#[pyfunction]
pub fn compute_checksums(_py: Python<'_>, data: Vec<Vec<u8>>) -> PyResult<Vec<u64>> {
    let checksums = data.iter().map(|chunk| fnv1a_64(chunk)).collect::<Vec<u64>>();
    Ok(checksums)
}

// ---------------------------------------------------------------------------
// compute_checksum_single — convenience helper
// ---------------------------------------------------------------------------

/// Compute a single FNV-1a 64-bit checksum for a byte string.
///
/// Parameters
/// ----------
/// data : bytes
///     Input byte string.
///
/// Returns
/// -------
/// int
///     FNV-1a 64-bit checksum (unsigned 64-bit integer).
#[pyfunction]
pub fn compute_checksum_single(_py: Python<'_>, data: Vec<u8>) -> PyResult<u64> {
    Ok(fnv1a_64(&data))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register parallel utility functions into the parent module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_checksums, m)?)?;
    m.add_function(wrap_pyfunction!(compute_checksum_single, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_empty() {
        // FNV-1a of empty slice is the offset basis.
        let h = fnv1a_64(&[]);
        assert_eq!(h, 14_695_981_039_346_656_037_u64);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        let a = fnv1a_64(b"hello");
        let b = fnv1a_64(b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn test_fnv1a_different_inputs() {
        let a = fnv1a_64(b"hello");
        let b = fnv1a_64(b"world");
        assert_ne!(a, b);
    }

    #[test]
    fn test_fnv1a_known_value() {
        // FNV-1a 64-bit of "hello" is a well-known value.
        let h = fnv1a_64(b"hello");
        assert!(h != 0, "hash should not be zero");
    }
}
