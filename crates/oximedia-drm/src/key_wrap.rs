// SPDX-License-Identifier: Apache-2.0
// Copyright (c) COOLJAPAN OU (Team Kitasan)

//! AES-128 key wrapping and unwrapping.
//!
//! This module provides a simple XOR-based key encapsulation mechanism (KEM)
//! for AES-128 keys.  The approach XOR-combines a Key Encryption Key (`kek`)
//! with a Content Encryption Key (`cek`) to produce a wrapped key.  Because
//! XOR is self-inverse, `unwrap_aes128(kek, wrap_aes128(kek, cek)) == cek`.
//!
//! # Security Notice
//!
//! The XOR stub is **not** cryptographically secure.  It is intended as a
//! placeholder for a real RFC 3394 / NIST SP 800-38F AES Key Wrap
//! implementation that can be provided by the platform HSM or a future
//! `aes-kw` crate integration.
//!
//! # Example
//!
//! ```rust
//! use oximedia_drm::key_wrap::KeyWrapper;
//!
//! let kek: [u8; 16] = [0xAB; 16];
//! let cek: [u8; 16] = [0x12; 16];
//!
//! let wrapped = KeyWrapper::wrap_aes128(&kek, &cek);
//! let unwrapped = KeyWrapper::unwrap_aes128(&kek, &wrapped);
//! assert_eq!(unwrapped, cek);
//! ```

// ---------------------------------------------------------------------------
// KeyWrapper
// ---------------------------------------------------------------------------

/// Stateless AES-128 key wrap / unwrap operations.
pub struct KeyWrapper;

impl KeyWrapper {
    /// Wrap a 128-bit Content Encryption Key (`cek`) with a 128-bit
    /// Key Encryption Key (`kek`).
    ///
    /// The current implementation applies a byte-wise XOR between `kek` and
    /// `cek`.  This is a symmetric, self-inverse operation: calling
    /// [`unwrap_aes128`](Self::unwrap_aes128) with the same `kek` recovers the
    /// original `cek`.
    ///
    /// # Arguments
    ///
    /// * `kek` — 16-byte Key Encryption Key (the master/wrapping key).
    /// * `cek` — 16-byte Content Encryption Key (the key to protect).
    ///
    /// # Returns
    ///
    /// A 16-byte wrapped key.
    #[must_use]
    pub fn wrap_aes128(kek: &[u8; 16], cek: &[u8; 16]) -> [u8; 16] {
        xor_keys(kek, cek)
    }

    /// Unwrap a previously wrapped 128-bit key.
    ///
    /// Applies the same XOR operation as [`wrap_aes128`](Self::wrap_aes128)
    /// to recover the original `cek`.
    ///
    /// # Arguments
    ///
    /// * `kek`     — 16-byte Key Encryption Key (must match the one used
    ///               during wrapping).
    /// * `wrapped` — 16-byte wrapped key produced by
    ///               [`wrap_aes128`](Self::wrap_aes128).
    ///
    /// # Returns
    ///
    /// The original 16-byte Content Encryption Key.
    #[must_use]
    pub fn unwrap_aes128(kek: &[u8; 16], wrapped: &[u8; 16]) -> [u8; 16] {
        xor_keys(kek, wrapped)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Byte-wise XOR of two 16-byte arrays.
#[inline]
fn xor_keys(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for i in 0..16 {
        out[i] = a[i] ^ b[i];
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_unwrap_roundtrip() {
        let kek: [u8; 16] = [
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
            0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
        ];
        let cek: [u8; 16] = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
            0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        ];
        let wrapped = KeyWrapper::wrap_aes128(&kek, &cek);
        let recovered = KeyWrapper::unwrap_aes128(&kek, &wrapped);
        assert_eq!(recovered, cek, "unwrapped key must equal original cek");
    }

    #[test]
    fn test_wrapped_differs_from_cek() {
        let kek: [u8; 16] = [0xAA; 16];
        let cek: [u8; 16] = [0x55; 16];
        let wrapped = KeyWrapper::wrap_aes128(&kek, &cek);
        assert_ne!(wrapped, cek, "wrapped key must differ from plaintext cek");
    }

    #[test]
    fn test_all_zeros_kek_is_identity() {
        // XOR with all-zeros KEK should return the CEK unchanged (identity).
        let kek: [u8; 16] = [0x00; 16];
        let cek: [u8; 16] = [0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF,
                              0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF];
        let wrapped = KeyWrapper::wrap_aes128(&kek, &cek);
        assert_eq!(wrapped, cek, "zero KEK acts as identity");
    }

    #[test]
    fn test_different_keks_produce_different_wraps() {
        let kek_a: [u8; 16] = [0x11; 16];
        let kek_b: [u8; 16] = [0x22; 16];
        let cek: [u8; 16] = [0xFF; 16];
        let wrapped_a = KeyWrapper::wrap_aes128(&kek_a, &cek);
        let wrapped_b = KeyWrapper::wrap_aes128(&kek_b, &cek);
        assert_ne!(wrapped_a, wrapped_b, "different KEKs must produce different wraps");
    }

    #[test]
    fn test_wrap_is_symmetric() {
        // wrap(kek, cek) == wrap(cek, kek)  (XOR is commutative)
        let kek: [u8; 16] = [0xAB; 16];
        let cek: [u8; 16] = [0xCD; 16];
        assert_eq!(
            KeyWrapper::wrap_aes128(&kek, &cek),
            KeyWrapper::wrap_aes128(&cek, &kek),
        );
    }

    #[test]
    fn test_wrap_identical_keys_produces_zeros() {
        let k: [u8; 16] = [0xBB; 16];
        let wrapped = KeyWrapper::wrap_aes128(&k, &k);
        assert_eq!(wrapped, [0u8; 16], "XOR of identical keys must be zero");
    }
}
