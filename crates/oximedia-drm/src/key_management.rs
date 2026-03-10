//! DRM key management system.
//!
//! Provides content key storage, retrieval, revocation, and lifecycle management.

use std::collections::HashMap;

/// A DRM content key with optional expiry.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ContentKey {
    pub key_id: Vec<u8>,
    pub key: Vec<u8>,
    pub iv: Vec<u8>,
    pub expiry: Option<u64>,
}

impl ContentKey {
    /// Create a new content key with a zero IV.
    pub fn new(key_id: Vec<u8>, key: Vec<u8>) -> Self {
        Self {
            iv: vec![0u8; 16],
            key_id,
            key,
            expiry: None,
        }
    }

    /// Set the expiry timestamp (Unix seconds).
    pub fn with_expiry(mut self, ts: u64) -> Self {
        self.expiry = Some(ts);
        self
    }

    /// Set a custom IV.
    pub fn with_iv(mut self, iv: Vec<u8>) -> Self {
        self.iv = iv;
        self
    }

    /// Returns `true` if the key has expired at the given timestamp.
    pub fn is_expired(&self, now: u64) -> bool {
        match self.expiry {
            Some(exp) => now >= exp,
            None => false,
        }
    }

    /// Return the key ID as a lowercase hex string.
    pub fn key_id_hex(&self) -> String {
        hex_encode(&self.key_id)
    }
}

/// In-memory store for content keys, keyed by their hex-encoded key ID.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct KeyStore {
    keys: HashMap<String, ContentKey>,
}

impl KeyStore {
    /// Create an empty key store.
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
        }
    }

    /// Persist a content key. Overwrites any existing key with the same ID.
    pub fn store(&mut self, key: ContentKey) {
        let hex_id = key.key_id_hex();
        self.keys.insert(hex_id, key);
    }

    /// Retrieve a key by its hex-encoded key ID.
    pub fn retrieve(&self, key_id_hex: &str) -> Option<&ContentKey> {
        self.keys.get(key_id_hex)
    }

    /// Revoke (delete) a key by hex ID. Returns `true` if the key existed.
    pub fn revoke(&mut self, key_id_hex: &str) -> bool {
        self.keys.remove(key_id_hex).is_some()
    }

    /// Remove all expired keys. Returns the number of keys removed.
    pub fn purge_expired(&mut self, now: u64) -> usize {
        let before = self.keys.len();
        self.keys.retain(|_, v| !v.is_expired(now));
        before - self.keys.len()
    }

    /// Return the total number of stored keys.
    pub fn count(&self) -> usize {
        self.keys.len()
    }
}

/// Encode a byte slice as a lowercase hex string.
pub fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Generate a 16-byte pseudo-random key ID using a simple LCG seeded by XOR
/// of a counter and a fixed constant so tests are reproducible when called
/// sequentially in a single-threaded context.
///
/// NOTE: This is NOT cryptographically secure; use a proper CSPRNG in production.
pub fn generate_key_id() -> Vec<u8> {
    use std::cell::Cell;
    thread_local! {
        static COUNTER: Cell<u64> = const { Cell::new(0x517cc1b727220a95u64) };
    }
    COUNTER.with(|c| {
        let mut state = c.get();
        let mut out = Vec::with_capacity(16);
        for _ in 0..2 {
            // splitmix64 step
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            out.extend_from_slice(&z.to_le_bytes());
        }
        c.set(state);
        out
    })
}

/// Policy governing automatic content-key rotation.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct KeyRotationPolicy {
    /// How often keys should be rotated, in milliseconds.
    pub interval_ms: u64,
    /// How many old (rotated-out) keys to retain for decryption of buffered
    /// content that was encrypted with an earlier key.
    pub keep_old_keys_count: usize,
}

impl KeyRotationPolicy {
    /// Create a new rotation policy.
    pub fn new(interval_ms: u64, keep_old_keys_count: usize) -> Self {
        Self {
            interval_ms,
            keep_old_keys_count,
        }
    }

    /// Returns `true` when `now_ms - last_rotation_ms >= interval_ms`.
    pub fn should_rotate(&self, last_rotation_ms: u64, now_ms: u64) -> bool {
        now_ms.saturating_sub(last_rotation_ms) >= self.interval_ms
    }
}

/// Derives content keys from a master key via FNV-1a-based mixing.
///
/// NOTE: This is NOT a production-grade KDF. Use a proper HKDF in production.
pub struct KeyDerivation;

impl KeyDerivation {
    /// Derive a 16-byte content key from a 32-byte master key and a
    /// content-identifier string using FNV-1a mixing.
    #[must_use]
    pub fn derive_key(master_key: &[u8; 32], content_id: &str) -> [u8; 16] {
        // FNV-1a 64-bit hash of content_id
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

        let mut hash = FNV_OFFSET;
        for byte in content_id.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }

        // Mix the hash into the first 8 bytes of the master key using XOR to
        // produce two u64 halves for the derived key.
        let master_lo = u64::from_le_bytes(master_key[0..8].try_into().unwrap_or([0u8; 8]));
        let master_hi = u64::from_le_bytes(master_key[8..16].try_into().unwrap_or([0u8; 8]));

        let lo = hash.wrapping_mul(master_lo ^ 0xdeadbeef_cafebabe);
        let hi = hash.wrapping_add(master_hi) ^ 0x0123456789abcdef;

        let mut out = [0u8; 16];
        out[0..8].copy_from_slice(&lo.to_le_bytes());
        out[8..16].copy_from_slice(&hi.to_le_bytes());
        out
    }
}

/// A content key with fixed-size 16-byte arrays for kid and key material,
/// plus an optional expiry in Unix milliseconds.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedContentKey {
    /// Key ID (16 bytes).
    pub kid: [u8; 16],
    /// Key material (16 bytes).
    pub key: [u8; 16],
    /// Creation timestamp in milliseconds since the Unix epoch.
    pub created_ms: u64,
    /// Optional expiry in milliseconds since the Unix epoch.
    pub expires_ms: Option<u64>,
}

impl FixedContentKey {
    /// Create a new key.
    pub fn new(kid: [u8; 16], key: [u8; 16], created_ms: u64) -> Self {
        Self {
            kid,
            key,
            created_ms,
            expires_ms: None,
        }
    }

    /// Builder: set an expiry.
    pub fn with_expires_ms(mut self, expires_ms: u64) -> Self {
        self.expires_ms = Some(expires_ms);
        self
    }

    /// Returns the KID as a lowercase hex string.
    pub fn kid_hex(&self) -> String {
        hex_encode(&self.kid)
    }

    /// Returns `true` when this key has expired at the given millisecond timestamp.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        match self.expires_ms {
            Some(exp) => now_ms >= exp,
            None => false,
        }
    }
}

/// An in-memory store of [`FixedContentKey`]s.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct FixedKeyStore {
    keys: Vec<FixedContentKey>,
}

impl FixedKeyStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a key.
    pub fn insert(&mut self, key: FixedContentKey) {
        self.keys.push(key);
    }

    /// Find a key by exact KID.
    pub fn find_by_kid(&self, kid: &[u8; 16]) -> Option<&FixedContentKey> {
        self.keys.iter().find(|k| &k.kid == kid)
    }

    /// Return all keys that have not expired at `now_ms`.
    pub fn active_keys(&self, now_ms: u64) -> Vec<&FixedContentKey> {
        self.keys.iter().filter(|k| !k.is_expired(now_ms)).collect()
    }

    /// Remove all expired keys and return the count removed.
    pub fn expire_old_keys(&mut self, now_ms: u64) -> usize {
        let before = self.keys.len();
        self.keys.retain(|k| !k.is_expired(now_ms));
        before - self.keys.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_key_new() {
        let key = ContentKey::new(vec![1, 2, 3], vec![4, 5, 6]);
        assert_eq!(key.key_id, vec![1, 2, 3]);
        assert_eq!(key.key, vec![4, 5, 6]);
        assert_eq!(key.iv.len(), 16);
        assert!(key.expiry.is_none());
    }

    #[test]
    fn test_content_key_with_expiry() {
        let key = ContentKey::new(vec![1], vec![2]).with_expiry(9999);
        assert_eq!(key.expiry, Some(9999));
    }

    #[test]
    fn test_is_expired_no_expiry() {
        let key = ContentKey::new(vec![1], vec![2]);
        assert!(!key.is_expired(u64::MAX));
    }

    #[test]
    fn test_is_expired_future() {
        let key = ContentKey::new(vec![1], vec![2]).with_expiry(2000);
        assert!(!key.is_expired(1000));
    }

    #[test]
    fn test_is_expired_past() {
        let key = ContentKey::new(vec![1], vec![2]).with_expiry(500);
        assert!(key.is_expired(1000));
    }

    #[test]
    fn test_is_expired_at_boundary() {
        let key = ContentKey::new(vec![1], vec![2]).with_expiry(1000);
        assert!(key.is_expired(1000));
    }

    #[test]
    fn test_key_id_hex() {
        let key = ContentKey::new(vec![0xde, 0xad, 0xbe, 0xef], vec![]);
        assert_eq!(key.key_id_hex(), "deadbeef");
    }

    #[test]
    fn test_hex_encode() {
        assert_eq!(hex_encode(&[0x00, 0xff, 0x0a]), "00ff0a");
        assert_eq!(hex_encode(&[]), "");
    }

    #[test]
    fn test_key_store_store_and_retrieve() {
        let mut store = KeyStore::new();
        let key = ContentKey::new(vec![0xaa, 0xbb], vec![0x01]);
        let hex_id = key.key_id_hex();
        store.store(key);
        let retrieved = store.retrieve(&hex_id).expect("retrieved should be valid");
        assert_eq!(retrieved.key_id, vec![0xaa, 0xbb]);
    }

    #[test]
    fn test_key_store_revoke() {
        let mut store = KeyStore::new();
        let key = ContentKey::new(vec![0x01], vec![0x02]);
        let hex_id = key.key_id_hex();
        store.store(key);
        assert!(store.revoke(&hex_id));
        assert!(store.retrieve(&hex_id).is_none());
        // second revoke returns false
        assert!(!store.revoke(&hex_id));
    }

    #[test]
    fn test_key_store_purge_expired() {
        let mut store = KeyStore::new();
        let k1 = ContentKey::new(vec![0x01], vec![]).with_expiry(100);
        let k2 = ContentKey::new(vec![0x02], vec![]).with_expiry(200);
        let k3 = ContentKey::new(vec![0x03], vec![]); // no expiry
        store.store(k1);
        store.store(k2);
        store.store(k3);
        // at t=150, k1 is expired
        let purged = store.purge_expired(150);
        assert_eq!(purged, 1);
        assert_eq!(store.count(), 2);
    }

    #[test]
    fn test_key_store_count() {
        let mut store = KeyStore::new();
        assert_eq!(store.count(), 0);
        store.store(ContentKey::new(vec![0x01], vec![]));
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn test_generate_key_id_length() {
        let id = generate_key_id();
        assert_eq!(id.len(), 16);
    }

    #[test]
    fn test_generate_key_id_sequential_differ() {
        let id1 = generate_key_id();
        let id2 = generate_key_id();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_key_with_custom_iv() {
        let iv = vec![0x01u8; 16];
        let key = ContentKey::new(vec![0x01], vec![0x02]).with_iv(iv.clone());
        assert_eq!(key.iv, iv);
    }

    // ----- KeyRotationPolicy tests -----

    #[test]
    fn test_rotation_policy_should_rotate_true() {
        let policy = KeyRotationPolicy::new(3600_000, 2);
        assert!(policy.should_rotate(0, 3_600_000));
    }

    #[test]
    fn test_rotation_policy_should_rotate_false() {
        let policy = KeyRotationPolicy::new(3600_000, 2);
        assert!(!policy.should_rotate(0, 1_000));
    }

    #[test]
    fn test_rotation_policy_exact_boundary() {
        let policy = KeyRotationPolicy::new(1000, 1);
        assert!(policy.should_rotate(5000, 6000));
        assert!(!policy.should_rotate(5000, 5999));
    }

    // ----- KeyDerivation tests -----

    #[test]
    fn test_key_derivation_produces_16_bytes() {
        let master = [0u8; 32];
        let key = KeyDerivation::derive_key(&master, "test-content");
        assert_eq!(key.len(), 16);
    }

    #[test]
    fn test_key_derivation_deterministic() {
        let master = [0xabu8; 32];
        let k1 = KeyDerivation::derive_key(&master, "movie-123");
        let k2 = KeyDerivation::derive_key(&master, "movie-123");
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_key_derivation_different_content_ids() {
        let master = [0x42u8; 32];
        let k1 = KeyDerivation::derive_key(&master, "content-a");
        let k2 = KeyDerivation::derive_key(&master, "content-b");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_key_derivation_different_masters() {
        let m1 = [0x01u8; 32];
        let m2 = [0x02u8; 32];
        let k1 = KeyDerivation::derive_key(&m1, "same-content");
        let k2 = KeyDerivation::derive_key(&m2, "same-content");
        assert_ne!(k1, k2);
    }

    // ----- FixedContentKey tests -----

    #[test]
    fn test_fixed_content_key_kid_hex() {
        let mut kid = [0u8; 16];
        kid[0] = 0xde;
        kid[1] = 0xad;
        let key = FixedContentKey::new(kid, [0u8; 16], 0);
        assert!(key.kid_hex().starts_with("dead"));
    }

    #[test]
    fn test_fixed_content_key_not_expired_no_expiry() {
        let key = FixedContentKey::new([0u8; 16], [0u8; 16], 0);
        assert!(!key.is_expired(u64::MAX));
    }

    #[test]
    fn test_fixed_content_key_expired() {
        let key = FixedContentKey::new([0u8; 16], [0u8; 16], 0).with_expires_ms(500);
        assert!(key.is_expired(500));
        assert!(key.is_expired(1000));
        assert!(!key.is_expired(499));
    }

    // ----- FixedKeyStore tests -----

    #[test]
    fn test_fixed_key_store_insert_and_find() {
        let mut store = FixedKeyStore::new();
        let kid = [0x01u8; 16];
        let key = FixedContentKey::new(kid, [0xffu8; 16], 1000);
        store.insert(key);
        let found = store.find_by_kid(&kid);
        assert!(found.is_some());
    }

    #[test]
    fn test_fixed_key_store_active_keys() {
        let mut store = FixedKeyStore::new();
        let k1 = FixedContentKey::new([0x01u8; 16], [0u8; 16], 0).with_expires_ms(100);
        let k2 = FixedContentKey::new([0x02u8; 16], [0u8; 16], 0); // no expiry
        store.insert(k1);
        store.insert(k2);
        // at t=200, k1 is expired
        let active = store.active_keys(200);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].kid, [0x02u8; 16]);
    }

    #[test]
    fn test_fixed_key_store_expire_old_keys() {
        let mut store = FixedKeyStore::new();
        store.insert(FixedContentKey::new([0x01u8; 16], [0u8; 16], 0).with_expires_ms(50));
        store.insert(FixedContentKey::new([0x02u8; 16], [0u8; 16], 0).with_expires_ms(150));
        let removed = store.expire_old_keys(100);
        assert_eq!(removed, 1);
    }
}
