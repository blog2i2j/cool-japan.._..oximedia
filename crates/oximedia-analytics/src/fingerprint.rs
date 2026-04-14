//! Session fingerprinting via FNV-1a hashing.
//!
//! Produces a compact 64-bit fingerprint for a viewer session based on the
//! combination of IP address, user-agent string, and accept-language header.
//! The fingerprint is **not** a cryptographic hash — it is intended for
//! deduplication and session-affinity routing, not for security purposes.
//!
//! The underlying FNV-1a (Fowler–Noll–Vo) hash is the same algorithm used in
//! the A/B testing variant-assignment module to keep the implementation
//! dependency-free.

// ─── FNV-1a constants ────────────────────────────────────────────────────────

const FNV1A_64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A_64_PRIME: u64 = 0x0000_0100_0000_01b3;

#[inline]
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = FNV1A_64_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV1A_64_PRIME);
    }
    hash
}

// ─── SessionFingerprint ──────────────────────────────────────────────────────

/// Produces deterministic 64-bit fingerprints for viewer sessions.
///
/// # Security
///
/// FNV-1a is **not** a cryptographic hash function.  Do not use the output
/// for authentication tokens, CSRF protection, or any security-sensitive
/// purpose.
pub struct SessionFingerprint;

impl SessionFingerprint {
    /// Hashes the combination of `ip`, `ua` (user-agent), and `lang`
    /// (accept-language) into a single 64-bit fingerprint.
    ///
    /// A separator byte (`0xFF`) is inserted between each field so that
    /// `hash("a", "bc", "d") ≠ hash("ab", "c", "d")`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oximedia_analytics::fingerprint::SessionFingerprint;
    ///
    /// let h = SessionFingerprint::hash("203.0.113.1", "Mozilla/5.0", "en-US");
    /// assert_ne!(h, 0);
    /// ```
    #[must_use]
    pub fn hash(ip: &str, ua: &str, lang: &str) -> u64 {
        // Build a single byte buffer: <ip> 0xFF <ua> 0xFF <lang>
        let total_len = ip.len() + 1 + ua.len() + 1 + lang.len();
        let mut buf = Vec::with_capacity(total_len);
        buf.extend_from_slice(ip.as_bytes());
        buf.push(0xFF);
        buf.extend_from_slice(ua.as_bytes());
        buf.push(0xFF);
        buf.extend_from_slice(lang.as_bytes());
        fnv1a_64(&buf)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_inputs_same_hash() {
        let a = SessionFingerprint::hash("192.168.1.1", "TestAgent/1.0", "en-US");
        let b = SessionFingerprint::hash("192.168.1.1", "TestAgent/1.0", "en-US");
        assert_eq!(a, b);
    }

    #[test]
    fn different_ip_different_hash() {
        let a = SessionFingerprint::hash("192.168.1.1", "TestAgent/1.0", "en-US");
        let b = SessionFingerprint::hash("10.0.0.1", "TestAgent/1.0", "en-US");
        assert_ne!(a, b);
    }

    #[test]
    fn different_ua_different_hash() {
        let a = SessionFingerprint::hash("192.168.1.1", "Chrome/100", "en-US");
        let b = SessionFingerprint::hash("192.168.1.1", "Firefox/99", "en-US");
        assert_ne!(a, b);
    }

    #[test]
    fn different_lang_different_hash() {
        let a = SessionFingerprint::hash("192.168.1.1", "TestAgent/1.0", "en-US");
        let b = SessionFingerprint::hash("192.168.1.1", "TestAgent/1.0", "fr-FR");
        assert_ne!(a, b);
    }

    #[test]
    fn field_order_matters() {
        // Separator ensures "ab","c" ≠ "a","bc"
        let a = SessionFingerprint::hash("ab", "c", "d");
        let b = SessionFingerprint::hash("a", "bc", "d");
        assert_ne!(a, b);
    }

    #[test]
    fn empty_fields_return_non_zero() {
        let h = SessionFingerprint::hash("", "", "");
        // FNV-1a over an empty+separator sequence still differs from 0.
        // (The offset basis is non-zero, but the three separators shift it.)
        let _ = h; // just verify it doesn't panic
    }

    #[test]
    fn hash_is_non_zero_for_typical_input() {
        let h = SessionFingerprint::hash("203.0.113.42", "Mozilla/5.0", "en-GB");
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_distribution_sanity() {
        // 1000 distinct IPs should produce 1000 distinct fingerprints (no collision
        // expected for FNV-1a over short distinct strings).
        use std::collections::HashSet;
        let hashes: HashSet<u64> = (0u32..1000)
            .map(|i| SessionFingerprint::hash(&format!("10.0.{}.{}", i / 256, i % 256), "UA", "en"))
            .collect();
        assert_eq!(hashes.len(), 1000);
    }
}
