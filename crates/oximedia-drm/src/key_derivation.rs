//! Content key derivation functions for DRM systems.
//!
//! Provides pure-Rust implementations of three standard key derivation functions:
//!
//! - **HKDF-SHA-256** (RFC 5869): extract-and-expand for generating keying material
//!   from high-entropy input key material.
//! - **PBKDF2-SHA-256** (RFC 8018): password-based key derivation with configurable
//!   iteration count, suitable for human-memorable passwords/PINs.
//! - **SP 800-108 CTR** (NIST SP 800-108r1): counter-mode KDF using HMAC-SHA-256 as
//!   the PRF, commonly used in broadcast DRM key hierarchies.
//!
//! All algorithms are self-contained and require no external cryptography crates.
//! The SHA-256 and HMAC-SHA-256 primitives are implemented inline.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Public-facing types
// ---------------------------------------------------------------------------

/// The key derivation algorithm to use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyDerivationMethod {
    /// HKDF (RFC 5869) using SHA-256 as the underlying hash.
    HkdfSha256,
    /// PBKDF2 (RFC 8018) using HMAC-SHA-256 as the PRF.
    ///
    /// The inner `u32` is the iteration count (`c` in the RFC).
    Pbkdf2Sha256(u32),
    /// NIST SP 800-108 counter-mode KDF using HMAC-SHA-256 as the PRF.
    Sp800108Ctr,
}

/// A derived content key along with provenance metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedKey {
    /// The raw derived key bytes.
    pub key_bytes: Vec<u8>,
    /// The method used to derive this key.
    pub method: KeyDerivationMethod,
    /// An application-defined identifier for this key.
    pub key_id: String,
}

/// Configuration for a key derivation operation.
#[derive(Debug, Clone)]
pub struct KeyDerivationConfig {
    /// Derivation algorithm.
    pub method: KeyDerivationMethod,
    /// Number of bytes to derive (1–64 for HKDF/SP800-108; 1–64 for PBKDF2).
    pub key_length_bytes: u8,
    /// Context / info bytes (used by HKDF expand and SP800-108 label).
    pub info: Vec<u8>,
    /// Salt bytes.
    pub salt: Vec<u8>,
}

impl KeyDerivationConfig {
    /// Create a new configuration.
    pub fn new(method: KeyDerivationMethod, key_length_bytes: u8) -> Self {
        Self {
            method,
            key_length_bytes,
            info: Vec::new(),
            salt: Vec::new(),
        }
    }

    /// Set the info/context bytes.
    pub fn with_info(mut self, info: Vec<u8>) -> Self {
        self.info = info;
        self
    }

    /// Set the salt bytes.
    pub fn with_salt(mut self, salt: Vec<u8>) -> Self {
        self.salt = salt;
        self
    }
}

/// Errors returned by key derivation operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum KdfError {
    /// The master key length is invalid (e.g. zero).
    #[error("Invalid key length: {0}")]
    InvalidKeyLength(String),
    /// The salt length is invalid for the chosen algorithm.
    #[error("Invalid salt length: {0}")]
    InvalidSaltLength(String),
    /// The derivation itself failed for an algorithm-specific reason.
    #[error("Derivation failed: {0}")]
    DerivationFailed(String),
}

// ---------------------------------------------------------------------------
// Pure-Rust SHA-256
// ---------------------------------------------------------------------------

const SHA256_H: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
    0x5be0cd19,
];

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

/// Compute SHA-256 of `msg` and return the 32-byte digest.
fn sha256(msg: &[u8]) -> [u8; 32] {
    let mut h = SHA256_H;

    // Pre-processing: padding
    let bit_len = (msg.len() as u64).wrapping_mul(8);
    let mut padded: Vec<u8> = msg.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) block
    for block in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, &word) in h.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Pure-Rust HMAC-SHA-256
// ---------------------------------------------------------------------------

const HMAC_BLOCK_SIZE: usize = 64;
const SHA256_DIGEST_SIZE: usize = 32;

/// Compute HMAC-SHA-256(`key`, `data`) and return the 32-byte MAC.
pub(crate) fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    // If key longer than block size, hash it first
    let key_block: Vec<u8> = if key.len() > HMAC_BLOCK_SIZE {
        let hashed = sha256(key);
        let mut block = vec![0u8; HMAC_BLOCK_SIZE];
        block[..SHA256_DIGEST_SIZE].copy_from_slice(&hashed);
        block
    } else {
        let mut block = vec![0u8; HMAC_BLOCK_SIZE];
        block[..key.len()].copy_from_slice(key);
        block
    };

    let mut ipad = [0u8; HMAC_BLOCK_SIZE];
    let mut opad = [0u8; HMAC_BLOCK_SIZE];
    for i in 0..HMAC_BLOCK_SIZE {
        ipad[i] = key_block[i] ^ 0x36;
        opad[i] = key_block[i] ^ 0x5c;
    }

    let mut inner_input = ipad.to_vec();
    inner_input.extend_from_slice(data);
    let inner_hash = sha256(&inner_input);

    let mut outer_input = opad.to_vec();
    outer_input.extend_from_slice(&inner_hash);
    sha256(&outer_input)
}

// ---------------------------------------------------------------------------
// HKDF-SHA-256  (RFC 5869)
// ---------------------------------------------------------------------------

/// HKDF extract step: `PRK = HMAC-Hash(salt, IKM)`.
fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> [u8; 32] {
    // If salt is empty use a string of HashLen zeros as the salt
    if salt.is_empty() {
        hmac_sha256(&[0u8; SHA256_DIGEST_SIZE], ikm)
    } else {
        hmac_sha256(salt, ikm)
    }
}

/// HKDF expand step: derive `length` bytes from `prk` and `info`.
///
/// Implements RFC 5869 §2.3.  Maximum output is 255 × HashLen = 8160 bytes.
fn hkdf_expand(prk: &[u8], info: &[u8], length: usize) -> Result<Vec<u8>, KdfError> {
    if length == 0 || length > 255 * SHA256_DIGEST_SIZE {
        return Err(KdfError::DerivationFailed(
            "HKDF: requested length out of range".to_owned(),
        ));
    }

    let n = length.div_ceil(SHA256_DIGEST_SIZE);
    let mut t = Vec::<u8>::new(); // T(0) = empty
    let mut okm = Vec::with_capacity(n * SHA256_DIGEST_SIZE);

    for i in 1..=n {
        let mut data = t.clone();
        data.extend_from_slice(info);
        data.push(i as u8);
        let ti = hmac_sha256(prk, &data);
        t = ti.to_vec();
        okm.extend_from_slice(&ti);
    }

    okm.truncate(length);
    Ok(okm)
}

// ---------------------------------------------------------------------------
// SP 800-108 Counter-Mode KDF
// ---------------------------------------------------------------------------

/// Derive `length` bytes using NIST SP 800-108r1 counter-mode KDF.
///
/// Label = `info`, Context = empty.  Counter is big-endian 32-bit.
fn sp800108_ctr(key: &[u8], label: &[u8], length: usize) -> Result<Vec<u8>, KdfError> {
    if length == 0 || length > 255 * SHA256_DIGEST_SIZE {
        return Err(KdfError::DerivationFailed(
            "SP800-108 CTR: requested length out of range".to_owned(),
        ));
    }

    let n = length.div_ceil(SHA256_DIGEST_SIZE);
    let l_bits = (length * 8) as u32;
    let mut result = Vec::with_capacity(n * SHA256_DIGEST_SIZE);

    for i in 1u32..=(n as u32) {
        // PRF input = counter(32-bit BE) || label || 0x00 || L(32-bit BE)
        let mut prf_input = Vec::new();
        prf_input.extend_from_slice(&i.to_be_bytes());
        prf_input.extend_from_slice(label);
        prf_input.push(0x00); // separator
        prf_input.extend_from_slice(&l_bits.to_be_bytes());
        let ki = hmac_sha256(key, &prf_input);
        result.extend_from_slice(&ki);
    }

    result.truncate(length);
    Ok(result)
}

// ---------------------------------------------------------------------------
// PBKDF2-SHA-256
// ---------------------------------------------------------------------------

/// PBKDF2 with HMAC-SHA-256.  Derives `length` bytes.
fn pbkdf2_sha256(
    password: &[u8],
    salt: &[u8],
    iterations: u32,
    length: usize,
) -> Result<Vec<u8>, KdfError> {
    if iterations == 0 {
        return Err(KdfError::DerivationFailed(
            "PBKDF2: iteration count must be > 0".to_owned(),
        ));
    }
    if length == 0 {
        return Err(KdfError::DerivationFailed(
            "PBKDF2: requested length must be > 0".to_owned(),
        ));
    }

    let n_blocks = length.div_ceil(SHA256_DIGEST_SIZE);
    let mut dk = Vec::with_capacity(n_blocks * SHA256_DIGEST_SIZE);

    for i in 1u32..=(n_blocks as u32) {
        // U1 = PRF(Password, Salt || INT(i))
        let mut salt_i = salt.to_vec();
        salt_i.extend_from_slice(&i.to_be_bytes());
        let u1 = hmac_sha256(password, &salt_i);
        let mut u_prev = u1;
        let mut block = u1;

        for _ in 1..iterations {
            let u_next = hmac_sha256(password, &u_prev);
            for j in 0..SHA256_DIGEST_SIZE {
                block[j] ^= u_next[j];
            }
            u_prev = u_next;
        }

        dk.extend_from_slice(&block);
    }

    dk.truncate(length);
    Ok(dk)
}

// ---------------------------------------------------------------------------
// KeyDeriver
// ---------------------------------------------------------------------------

/// Derives content keys using the configured KDF algorithm.
pub struct KeyDeriver {
    key_id: String,
}

impl KeyDeriver {
    /// Create a new `KeyDeriver` that will tag derived keys with `key_id`.
    pub fn new(key_id: impl Into<String>) -> Self {
        Self {
            key_id: key_id.into(),
        }
    }

    /// Derive a key from `master_key` using the settings in `config`.
    ///
    /// # Errors
    ///
    /// - [`KdfError::InvalidKeyLength`] — `master_key` is empty.
    /// - [`KdfError::DerivationFailed`] — algorithm-specific failure.
    pub fn derive(
        &self,
        master_key: &[u8],
        config: &KeyDerivationConfig,
    ) -> Result<DerivedKey, KdfError> {
        if master_key.is_empty() {
            return Err(KdfError::InvalidKeyLength(
                "master key must not be empty".to_owned(),
            ));
        }
        let length = config.key_length_bytes as usize;
        if length == 0 {
            return Err(KdfError::InvalidKeyLength(
                "key_length_bytes must be > 0".to_owned(),
            ));
        }

        let key_bytes = match &config.method {
            KeyDerivationMethod::HkdfSha256 => {
                let prk = hkdf_extract(&config.salt, master_key);
                hkdf_expand(&prk, &config.info, length)?
            }
            KeyDerivationMethod::Pbkdf2Sha256(iterations) => {
                if config.salt.is_empty() {
                    return Err(KdfError::InvalidSaltLength(
                        "PBKDF2 requires a non-empty salt".to_owned(),
                    ));
                }
                pbkdf2_sha256(master_key, &config.salt, *iterations, length)?
            }
            KeyDerivationMethod::Sp800108Ctr => {
                sp800108_ctr(master_key, &config.info, length)?
            }
        };

        Ok(DerivedKey {
            key_bytes,
            method: config.method.clone(),
            key_id: self.key_id.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn deriver() -> KeyDeriver {
        KeyDeriver::new("test-key-id")
    }

    // --- HKDF tests ---

    #[test]
    fn test_hkdf_deterministic() {
        let config = KeyDerivationConfig::new(KeyDerivationMethod::HkdfSha256, 32)
            .with_salt(b"test-salt".to_vec())
            .with_info(b"test-info".to_vec());
        let k1 = deriver()
            .derive(b"master-key", &config)
            .expect("derive should succeed");
        let k2 = deriver()
            .derive(b"master-key", &config)
            .expect("derive should succeed");
        assert_eq!(k1.key_bytes, k2.key_bytes);
        assert_eq!(k1.key_bytes.len(), 32);
    }

    #[test]
    fn test_hkdf_different_salts_produce_different_keys() {
        let config_a = KeyDerivationConfig::new(KeyDerivationMethod::HkdfSha256, 32)
            .with_salt(b"salt-A".to_vec());
        let config_b = KeyDerivationConfig::new(KeyDerivationMethod::HkdfSha256, 32)
            .with_salt(b"salt-B".to_vec());
        let k_a = deriver()
            .derive(b"same-master", &config_a)
            .expect("derive A");
        let k_b = deriver()
            .derive(b"same-master", &config_b)
            .expect("derive B");
        assert_ne!(k_a.key_bytes, k_b.key_bytes);
    }

    #[test]
    fn test_hkdf_key_length_respected() {
        for len in [16u8, 24, 32, 48, 64] {
            let config =
                KeyDerivationConfig::new(KeyDerivationMethod::HkdfSha256, len)
                    .with_salt(b"s".to_vec());
            let dk = deriver()
                .derive(b"master", &config)
                .expect("derive should succeed");
            assert_eq!(dk.key_bytes.len(), len as usize);
        }
    }

    #[test]
    fn test_hkdf_empty_master_key_error() {
        let config = KeyDerivationConfig::new(KeyDerivationMethod::HkdfSha256, 16);
        let err = deriver()
            .derive(b"", &config)
            .expect_err("empty key should fail");
        assert!(matches!(err, KdfError::InvalidKeyLength(_)));
    }

    // --- PBKDF2 tests ---

    #[test]
    fn test_pbkdf2_deterministic() {
        let config =
            KeyDerivationConfig::new(KeyDerivationMethod::Pbkdf2Sha256(1000), 32)
                .with_salt(b"pbkdf2-salt".to_vec());
        let k1 = deriver()
            .derive(b"password", &config)
            .expect("derive k1");
        let k2 = deriver()
            .derive(b"password", &config)
            .expect("derive k2");
        assert_eq!(k1.key_bytes, k2.key_bytes);
        assert_eq!(k1.key_bytes.len(), 32);
    }

    #[test]
    fn test_pbkdf2_different_iterations_produce_different_keys() {
        let salt = b"common-salt".to_vec();
        let config_100 =
            KeyDerivationConfig::new(KeyDerivationMethod::Pbkdf2Sha256(100), 32)
                .with_salt(salt.clone());
        let config_200 =
            KeyDerivationConfig::new(KeyDerivationMethod::Pbkdf2Sha256(200), 32)
                .with_salt(salt);
        let k100 = deriver().derive(b"pw", &config_100).expect("100 iters");
        let k200 = deriver().derive(b"pw", &config_200).expect("200 iters");
        assert_ne!(k100.key_bytes, k200.key_bytes);
    }

    #[test]
    fn test_pbkdf2_empty_salt_error() {
        let config = KeyDerivationConfig::new(KeyDerivationMethod::Pbkdf2Sha256(100), 16);
        // salt is empty by default
        let err = deriver()
            .derive(b"password", &config)
            .expect_err("empty salt should fail");
        assert!(matches!(err, KdfError::InvalidSaltLength(_)));
    }

    // --- SP800-108 CTR tests ---

    #[test]
    fn test_sp800108_deterministic() {
        let config =
            KeyDerivationConfig::new(KeyDerivationMethod::Sp800108Ctr, 32)
                .with_info(b"content-key-label".to_vec());
        let k1 = deriver().derive(b"root-key", &config).expect("derive k1");
        let k2 = deriver().derive(b"root-key", &config).expect("derive k2");
        assert_eq!(k1.key_bytes, k2.key_bytes);
        assert_eq!(k1.key_bytes.len(), 32);
    }

    #[test]
    fn test_sp800108_different_labels_produce_different_keys() {
        let config_a = KeyDerivationConfig::new(KeyDerivationMethod::Sp800108Ctr, 32)
            .with_info(b"audio-key".to_vec());
        let config_b = KeyDerivationConfig::new(KeyDerivationMethod::Sp800108Ctr, 32)
            .with_info(b"video-key".to_vec());
        let k_a = deriver().derive(b"root", &config_a).expect("audio");
        let k_b = deriver().derive(b"root", &config_b).expect("video");
        assert_ne!(k_a.key_bytes, k_b.key_bytes);
    }

    #[test]
    fn test_sp800108_key_length_respected() {
        for len in [16u8, 32, 48] {
            let config =
                KeyDerivationConfig::new(KeyDerivationMethod::Sp800108Ctr, len);
            let dk = deriver().derive(b"root", &config).expect("derive");
            assert_eq!(dk.key_bytes.len(), len as usize);
        }
    }

    // --- Metadata tests ---

    #[test]
    fn test_derived_key_method_tag() {
        let config = KeyDerivationConfig::new(KeyDerivationMethod::HkdfSha256, 16)
            .with_salt(b"s".to_vec());
        let dk = deriver().derive(b"mk", &config).expect("derive");
        assert_eq!(dk.method, KeyDerivationMethod::HkdfSha256);
        assert_eq!(dk.key_id, "test-key-id");
    }

    // --- SHA-256 known-answer test ---

    #[test]
    fn test_sha256_empty_message() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let digest = sha256(b"");
        let expected: [u8; 32] = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        assert_eq!(digest, expected);
    }

    #[test]
    fn test_sha256_abc() {
        // SHA-256("abc") known-answer (verified against Python hashlib):
        // ba7816bf 8f01cfea 414140de 5dae2223 b00361a3 96177a9c b410ff61 f20015ad
        let digest = sha256(b"abc");
        let expected: [u8; 32] = [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ];
        assert_eq!(digest, expected, "SHA-256('abc') known-answer test failed");
    }
}
