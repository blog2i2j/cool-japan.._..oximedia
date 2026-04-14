//! Signed URL generation and HMAC-based token authentication for CDN edge nodes.
//!
//! # Overview
//!
//! [`TokenSigner`] generates signed URLs and [`TokenValidator`] verifies them.
//! Both use HMAC-SHA-256 implemented entirely in pure Rust without any
//! external crypto dependency.
//!
//! # Signed URL format
//!
//! A signed URL appends three query parameters:
//!
//! ```text
//! https://cdn.example.com/video/stream.m3u8
//!   ?cdn_expires=<unix_ts>
//!   &cdn_ip=<client_ip_or_*>
//!   &cdn_sig=<hex_hmac_sha256>
//! ```
//!
//! The signature is computed over the canonical string:
//!
//! ```text
//! "<path>\n<expires>\n<ip_constraint>"
//! ```
//!
//! where `<ip_constraint>` is either the literal client IP or `"*"` (no
//! restriction).
//!
//! # Security note
//!
//! This module is designed for **simulation and integration testing** within
//! the OxiMedia framework.  The HMAC implementation is a faithful
//! NIST-compliant pure-Rust construction, but it has not been audited for
//! production cryptographic use.

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;
use thiserror::Error;

type HmacSha256 = Hmac<Sha256>;

// ─── Simple TokenAuth (task requirement) ──────────────────────────────────────

/// A simple HMAC-SHA-256 signed-URL authenticator.
///
/// Uses a shared `secret` to sign URL paths with an expiry timestamp.
/// The generated URL format is:
///
/// ```text
/// <path>?token=<hex_hmac>&expires=<unix_ts>
/// ```
///
/// where `<hex_hmac>` is HMAC-SHA-256 over `"<path>:<expires>"` using the
/// secret, computed via the `hmac` + `sha2` workspace crates.
#[derive(Clone)]
pub struct TokenAuth {
    secret: Vec<u8>,
}

impl std::fmt::Debug for TokenAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenAuth")
            .field("secret", &"<redacted>")
            .finish()
    }
}

impl TokenAuth {
    /// Create a new authenticator with the given secret bytes.
    pub fn new(secret: impl Into<Vec<u8>>) -> Self {
        Self {
            secret: secret.into(),
        }
    }

    /// Sign `path` with an expiry of `expiry_secs` seconds from now.
    ///
    /// Returns a URL string of the form `<path>?token=<hex>&expires=<ts>`.
    /// If the system clock is unavailable, expiry is set to `expiry_secs`
    /// (relative to epoch 0) — verification will immediately reject it.
    pub fn sign_url(&self, path: &str, expiry_secs: u64) -> String {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let expires = now.saturating_add(expiry_secs);
        let token = self.compute_hmac_token(path, expires);
        format!("{path}?token={token}&expires={expires}")
    }

    /// Verify a signed URL produced by [`Self::sign_url`].
    ///
    /// Returns `true` iff:
    /// 1. The URL contains both `token` and `expires` query parameters.
    /// 2. The current time is strictly less than `expires`.
    /// 3. The HMAC digest matches (constant-time comparison).
    pub fn verify_url(&self, url: &str) -> bool {
        let (path, query) = match url.split_once('?') {
            Some(p) => p,
            None => return false,
        };

        let mut token_val: Option<&str> = None;
        let mut expires_val: Option<u64> = None;

        for part in query.split('&') {
            if let Some(v) = part.strip_prefix("token=") {
                token_val = Some(v);
            } else if let Some(v) = part.strip_prefix("expires=") {
                expires_val = v.parse().ok();
            }
        }

        let (token, expires) = match (token_val, expires_val) {
            (Some(t), Some(e)) => (t, e),
            _ => return false,
        };

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if now >= expires {
            return false;
        }

        let expected = self.compute_hmac_token(path, expires);
        token_auth_ct_eq(token.as_bytes(), expected.as_bytes())
    }

    fn compute_hmac_token(&self, path: &str, expires: u64) -> String {
        let message = format!("{path}:{expires}");
        let mut mac =
            HmacSha256::new_from_slice(&self.secret).expect("HMAC accepts any key length");
        mac.update(message.as_bytes());
        let result = mac.finalize().into_bytes();
        token_auth_hex_encode(&result)
    }
}

/// Encode bytes as lowercase hexadecimal (for `TokenAuth`).
fn token_auth_hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// Constant-time equality check (for `TokenAuth`).
fn token_auth_ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ─── Constants ────────────────────────────────────────────────────────────────

/// HMAC-SHA-256 block size in bytes.
const BLOCK_SIZE: usize = 64;
/// SHA-256 digest size in bytes.
const DIGEST_SIZE: usize = 32;

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors that can arise during token signing or validation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TokenError {
    /// The URL is malformed or missing required query parameters.
    #[error("malformed signed URL: {0}")]
    MalformedUrl(String),
    /// The token's `cdn_expires` timestamp has already passed.
    #[error("token has expired (expired at unix={expires}, now={now})")]
    Expired {
        /// The expiry timestamp embedded in the token.
        expires: u64,
        /// The current unix timestamp.
        now: u64,
    },
    /// The IP address of the requester does not match the token's constraint.
    #[error("IP address mismatch: token restricts to '{allowed}', got '{actual}'")]
    IpMismatch {
        /// IP constraint embedded in the token.
        allowed: String,
        /// IP address supplied by the requester.
        actual: String,
    },
    /// The HMAC signature does not match.
    #[error("invalid token signature")]
    InvalidSignature,
    /// The signing key with the given ID is unknown.
    #[error("unknown signing key id: {0}")]
    UnknownKeyId(String),
    /// A system clock error occurred.
    #[error("system clock error")]
    ClockError,
}

// ─── Pure-Rust SHA-256 ────────────────────────────────────────────────────────

/// Compute SHA-256 of `data` and return the 32-byte digest.
fn sha256(data: &[u8]) -> [u8; DIGEST_SIZE] {
    // Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
    let k: [u32; 64] = [
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

    // Pre-processing: padding
    let msg_len_bits: u64 = (data.len() as u64).wrapping_mul(8);
    let mut padded = data.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&msg_len_bits.to_be_bytes());

    // Process each 512-bit chunk
    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, bytes) in chunk.chunks_exact(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
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
                .wrapping_add(k[i])
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

    let mut digest = [0u8; DIGEST_SIZE];
    for (i, val) in h.iter().enumerate() {
        digest[i * 4..(i + 1) * 4].copy_from_slice(&val.to_be_bytes());
    }
    digest
}

/// Compute HMAC-SHA-256 of `message` with `key`.
///
/// Implements RFC 2104 using the pure-Rust `sha256` function above.
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; DIGEST_SIZE] {
    // If key is longer than block size, hash it first.
    let mut k_block = [0u8; BLOCK_SIZE];
    if key.len() > BLOCK_SIZE {
        let hk = sha256(key);
        k_block[..DIGEST_SIZE].copy_from_slice(&hk);
    } else {
        k_block[..key.len()].copy_from_slice(key);
    }

    // Inner and outer padding.
    let mut i_pad = [0x36u8; BLOCK_SIZE];
    let mut o_pad = [0x5cu8; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        i_pad[i] ^= k_block[i];
        o_pad[i] ^= k_block[i];
    }

    // inner = SHA256(i_pad || message)
    let mut inner_input = Vec::with_capacity(BLOCK_SIZE + message.len());
    inner_input.extend_from_slice(&i_pad);
    inner_input.extend_from_slice(message);
    let inner_hash = sha256(&inner_input);

    // HMAC = SHA256(o_pad || inner)
    let mut outer_input = Vec::with_capacity(BLOCK_SIZE + DIGEST_SIZE);
    outer_input.extend_from_slice(&o_pad);
    outer_input.extend_from_slice(&inner_hash);
    sha256(&outer_input)
}

// ─── Hex helpers ─────────────────────────────────────────────────────────────

/// Encode bytes as lowercase hexadecimal.
fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// Decode a lowercase hex string into bytes. Returns `None` on invalid input.
fn from_hex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let bytes: Option<Vec<u8>> = s
        .as_bytes()
        .chunks(2)
        .map(|pair| {
            let hi = hex_val(pair[0])?;
            let lo = hex_val(pair[1])?;
            Some((hi << 4) | lo)
        })
        .collect();
    bytes
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// Constant-time byte slice comparison to prevent timing attacks.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut acc = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        acc |= x ^ y;
    }
    acc == 0
}

// ─── Signing key ─────────────────────────────────────────────────────────────

/// A named HMAC signing key.
#[derive(Debug, Clone)]
pub struct SigningKey {
    /// Unique identifier for key rotation (embedded in `cdn_keyid` parameter).
    pub id: String,
    /// Raw HMAC key material.
    key: Vec<u8>,
}

impl SigningKey {
    /// Create a new signing key.
    pub fn new(id: impl Into<String>, key: impl Into<Vec<u8>>) -> Self {
        Self {
            id: id.into(),
            key: key.into(),
        }
    }

    /// Create a signing key from a UTF-8 passphrase.
    pub fn from_passphrase(id: impl Into<String>, passphrase: &str) -> Self {
        Self::new(id, passphrase.as_bytes().to_vec())
    }

    /// Sign `canonical_string` and return the hex-encoded HMAC-SHA-256.
    fn sign(&self, canonical_string: &str) -> String {
        let mac = hmac_sha256(&self.key, canonical_string.as_bytes());
        to_hex(&mac)
    }

    /// Verify that `signature` matches the HMAC of `canonical_string`.
    fn verify(&self, canonical_string: &str, signature: &str) -> bool {
        let expected = hmac_sha256(&self.key, canonical_string.as_bytes());
        match from_hex(signature) {
            Some(provided) => constant_time_eq(&expected, &provided),
            None => false,
        }
    }
}

// ─── TokenConfig ─────────────────────────────────────────────────────────────

/// Configuration for token signing and validation.
#[derive(Debug, Clone)]
pub struct TokenConfig {
    /// Default token validity duration.
    pub default_ttl: Duration,
    /// Whether to enforce IP address restrictions in tokens.
    pub enforce_ip: bool,
    /// Query parameter name for the expiry timestamp.
    pub param_expires: String,
    /// Query parameter name for the IP constraint.
    pub param_ip: String,
    /// Query parameter name for the key ID (used for key rotation).
    pub param_key_id: String,
    /// Query parameter name for the HMAC signature.
    pub param_sig: String,
}

impl Default for TokenConfig {
    fn default() -> Self {
        Self {
            default_ttl: Duration::from_secs(3600),
            enforce_ip: false,
            param_expires: "cdn_expires".to_string(),
            param_ip: "cdn_ip".to_string(),
            param_key_id: "cdn_keyid".to_string(),
            param_sig: "cdn_sig".to_string(),
        }
    }
}

// ─── SignedUrlClaims ──────────────────────────────────────────────────────────

/// Structured claims extracted from a validated signed URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedUrlClaims {
    /// URL path that was signed (without query string).
    pub path: String,
    /// Unix timestamp at which the token expires.
    pub expires: u64,
    /// IP address constraint (`"*"` means no restriction).
    pub ip_constraint: String,
    /// Key ID used to sign the URL.
    pub key_id: String,
}

impl SignedUrlClaims {
    /// Returns `true` if the token has expired relative to `now_unix`.
    pub fn is_expired(&self, now_unix: u64) -> bool {
        now_unix >= self.expires
    }
}

// ─── TokenSigner ─────────────────────────────────────────────────────────────

/// Generates signed CDN URLs using HMAC-SHA-256.
///
/// Maintains a set of named [`SigningKey`]s for key rotation: one key is
/// designated as *active* for new signatures; all keys are tried when
/// validating (via [`TokenValidator`]).
#[derive(Debug)]
pub struct TokenSigner {
    keys: HashMap<String, SigningKey>,
    active_key_id: String,
    config: TokenConfig,
}

impl TokenSigner {
    /// Create a signer with a single active key.
    pub fn new(key: SigningKey, config: TokenConfig) -> Self {
        let id = key.id.clone();
        let mut keys = HashMap::new();
        keys.insert(id.clone(), key);
        Self {
            keys,
            active_key_id: id,
            config,
        }
    }

    /// Add an additional key (for key rotation — existing tokens remain valid).
    pub fn add_key(&mut self, key: SigningKey) {
        self.keys.insert(key.id.clone(), key);
    }

    /// Rotate to a new active key.  The key must already be registered via
    /// [`Self::add_key`].  Returns `false` if the key ID is unknown.
    pub fn rotate_key(&mut self, key_id: &str) -> bool {
        if self.keys.contains_key(key_id) {
            self.active_key_id = key_id.to_string();
            true
        } else {
            false
        }
    }

    /// Sign `path` with the active key, expiring at `expires_unix`.
    ///
    /// `ip_constraint` should be a dotted-decimal IPv4/IPv6 address to
    /// restrict the token to a specific client, or `"*"` for no restriction.
    ///
    /// Returns the signed URL with CDN parameters appended.
    pub fn sign_url(
        &self,
        base_url: &str,
        path: &str,
        expires_unix: u64,
        ip_constraint: &str,
    ) -> Result<String, TokenError> {
        let key = self
            .keys
            .get(&self.active_key_id)
            .ok_or_else(|| TokenError::UnknownKeyId(self.active_key_id.clone()))?;

        let canonical = build_canonical(path, expires_unix, ip_constraint);
        let sig = key.sign(&canonical);

        let sep = if base_url.contains('?') { '&' } else { '?' };
        let signed = format!(
            "{}{}{}{}={}&{}={}&{}={}&{}={}",
            base_url,
            path,
            sep,
            self.config.param_expires,
            expires_unix,
            self.config.param_ip,
            ip_constraint,
            self.config.param_key_id,
            self.active_key_id,
            self.config.param_sig,
            sig,
        );
        Ok(signed)
    }

    /// Sign `path` using the default TTL from the config, relative to now.
    pub fn sign_url_default_ttl(
        &self,
        base_url: &str,
        path: &str,
        ip_constraint: &str,
    ) -> Result<String, TokenError> {
        let now = current_unix_ts()?;
        let expires = now + self.config.default_ttl.as_secs();
        self.sign_url(base_url, path, expires, ip_constraint)
    }

    /// Access the active key id.
    pub fn active_key_id(&self) -> &str {
        &self.active_key_id
    }

    /// Number of registered keys (including non-active ones).
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }
}

// ─── TokenValidator ───────────────────────────────────────────────────────────

/// Validates CDN signed URLs by verifying their HMAC signature,
/// checking expiry, and optionally enforcing the IP constraint.
#[derive(Debug)]
pub struct TokenValidator {
    keys: HashMap<String, SigningKey>,
    config: TokenConfig,
}

impl TokenValidator {
    /// Create a validator from the same config and key set used by a signer.
    pub fn new(config: TokenConfig) -> Self {
        Self {
            keys: HashMap::new(),
            config,
        }
    }

    /// Register a key that will be accepted during validation.
    pub fn add_key(&mut self, key: SigningKey) {
        self.keys.insert(key.id.clone(), key);
    }

    /// Validate `signed_url` with the given `client_ip`.
    ///
    /// - Parses the CDN query parameters.
    /// - Looks up the key by `cdn_keyid`.
    /// - Verifies the HMAC signature.
    /// - Checks expiry against the current system clock.
    /// - If `config.enforce_ip` is true, verifies the client IP matches the
    ///   token's `cdn_ip` constraint (unless constraint is `"*"`).
    ///
    /// Returns the validated [`SignedUrlClaims`] on success.
    pub fn validate(
        &self,
        signed_url: &str,
        client_ip: &str,
    ) -> Result<SignedUrlClaims, TokenError> {
        let params = parse_query_params(signed_url);

        let expires_str = params
            .get(&self.config.param_expires as &str)
            .ok_or_else(|| TokenError::MalformedUrl("missing cdn_expires".to_string()))?;
        let ip_constraint = params
            .get(&self.config.param_ip as &str)
            .ok_or_else(|| TokenError::MalformedUrl("missing cdn_ip".to_string()))?;
        let key_id = params
            .get(&self.config.param_key_id as &str)
            .ok_or_else(|| TokenError::MalformedUrl("missing cdn_keyid".to_string()))?;
        let sig = params
            .get(&self.config.param_sig as &str)
            .ok_or_else(|| TokenError::MalformedUrl("missing cdn_sig".to_string()))?;

        let expires: u64 = expires_str
            .parse()
            .map_err(|_| TokenError::MalformedUrl("cdn_expires is not a u64".to_string()))?;

        let path = extract_path(signed_url);

        // Signature verification.
        let key = self
            .keys
            .get(*key_id)
            .ok_or_else(|| TokenError::UnknownKeyId((*key_id).to_string()))?;
        let canonical = build_canonical(&path, expires, ip_constraint);
        if !key.verify(&canonical, sig) {
            return Err(TokenError::InvalidSignature);
        }

        // Expiry check.
        let now = current_unix_ts()?;
        if now >= expires {
            return Err(TokenError::Expired { expires, now });
        }

        // IP constraint check.
        if self.config.enforce_ip && *ip_constraint != "*" && *ip_constraint != client_ip {
            return Err(TokenError::IpMismatch {
                allowed: (*ip_constraint).to_string(),
                actual: client_ip.to_string(),
            });
        }

        Ok(SignedUrlClaims {
            path,
            expires,
            ip_constraint: (*ip_constraint).to_string(),
            key_id: (*key_id).to_string(),
        })
    }

    /// Validate using an explicit `now_unix` timestamp instead of the system clock.
    ///
    /// Useful for deterministic tests.
    pub fn validate_at(
        &self,
        signed_url: &str,
        client_ip: &str,
        now_unix: u64,
    ) -> Result<SignedUrlClaims, TokenError> {
        let params = parse_query_params(signed_url);

        let expires_str = params
            .get(&self.config.param_expires as &str)
            .ok_or_else(|| TokenError::MalformedUrl("missing cdn_expires".to_string()))?;
        let ip_constraint = params
            .get(&self.config.param_ip as &str)
            .ok_or_else(|| TokenError::MalformedUrl("missing cdn_ip".to_string()))?;
        let key_id = params
            .get(&self.config.param_key_id as &str)
            .ok_or_else(|| TokenError::MalformedUrl("missing cdn_keyid".to_string()))?;
        let sig = params
            .get(&self.config.param_sig as &str)
            .ok_or_else(|| TokenError::MalformedUrl("missing cdn_sig".to_string()))?;

        let expires: u64 = expires_str
            .parse()
            .map_err(|_| TokenError::MalformedUrl("cdn_expires is not a u64".to_string()))?;

        let path = extract_path(signed_url);

        let key = self
            .keys
            .get(*key_id)
            .ok_or_else(|| TokenError::UnknownKeyId((*key_id).to_string()))?;
        let canonical = build_canonical(&path, expires, ip_constraint);
        if !key.verify(&canonical, sig) {
            return Err(TokenError::InvalidSignature);
        }

        if now_unix >= expires {
            return Err(TokenError::Expired {
                expires,
                now: now_unix,
            });
        }

        if self.config.enforce_ip && *ip_constraint != "*" && *ip_constraint != client_ip {
            return Err(TokenError::IpMismatch {
                allowed: (*ip_constraint).to_string(),
                actual: client_ip.to_string(),
            });
        }

        Ok(SignedUrlClaims {
            path,
            expires,
            ip_constraint: (*ip_constraint).to_string(),
            key_id: (*key_id).to_string(),
        })
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Build the canonical string that is signed:
/// `"<path>\n<expires>\n<ip_constraint>"`.
fn build_canonical(path: &str, expires: u64, ip_constraint: &str) -> String {
    format!("{}\n{}\n{}", path, expires, ip_constraint)
}

/// Extract the path component from a URL (everything before the `?`).
fn extract_path(url: &str) -> String {
    // Strip scheme+host to get the bare path.
    let without_scheme = if let Some(rest) = url.strip_prefix("https://") {
        rest
    } else if let Some(rest) = url.strip_prefix("http://") {
        rest
    } else {
        url
    };

    // The path starts after the first `/` following the host.
    let path_start = without_scheme.find('/').unwrap_or(without_scheme.len());
    let with_path = &without_scheme[path_start..];

    // Strip query string.
    match with_path.find('?') {
        Some(q) => with_path[..q].to_string(),
        None => with_path.to_string(),
    }
}

/// Parse query string parameters from a URL into a `HashMap<&str, &str>`.
fn parse_query_params(url: &str) -> HashMap<&str, &str> {
    let mut map = HashMap::new();
    if let Some(q_start) = url.find('?') {
        let query = &url[q_start + 1..];
        for pair in query.split('&') {
            if let Some(eq) = pair.find('=') {
                let key = &pair[..eq];
                let val = &pair[eq + 1..];
                map.insert(key, val);
            }
        }
    }
    map
}

/// Return the current unix timestamp.
fn current_unix_ts() -> Result<u64, TokenError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .map_err(|_| TokenError::ClockError)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signer() -> (TokenSigner, TokenValidator) {
        let key = SigningKey::from_passphrase("key1", "super-secret-passphrase");
        let config = TokenConfig::default();
        let mut validator = TokenValidator::new(config.clone());
        validator.add_key(SigningKey::from_passphrase(
            "key1",
            "super-secret-passphrase",
        ));
        let signer = TokenSigner::new(key, config);
        (signer, validator)
    }

    // ── SHA-256 ───────────────────────────────────────────────────────────

    // 1. SHA-256 of empty string is the known constant.
    #[test]
    fn test_sha256_empty() {
        let digest = sha256(b"");
        let hex = to_hex(&digest);
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    // 2. SHA-256 of "abc" matches the standard result.
    #[test]
    fn test_sha256_abc() {
        let digest = sha256(b"abc");
        let hex = to_hex(&digest);
        assert_eq!(
            hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    // 3. SHA-256 of the longer NIST test vector.
    #[test]
    fn test_sha256_longer() {
        let digest = sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        let hex = to_hex(&digest);
        assert_eq!(
            hex,
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    // ── HMAC-SHA-256 ──────────────────────────────────────────────────────

    // 4. HMAC-SHA-256 RFC 4231 test vector 1.
    #[test]
    fn test_hmac_sha256_rfc4231_1() {
        let key = vec![0x0bu8; 20];
        let msg = b"Hi There";
        let mac = hmac_sha256(&key, msg);
        let hex = to_hex(&mac);
        assert_eq!(
            hex,
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    // 5. HMAC-SHA-256 RFC 4231 test vector 2.
    #[test]
    fn test_hmac_sha256_rfc4231_2() {
        let key = b"Jefe";
        let msg = b"what do ya want for nothing?";
        let mac = hmac_sha256(key, msg);
        let hex = to_hex(&mac);
        assert_eq!(
            hex,
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    // 6. Different keys produce different MACs.
    #[test]
    fn test_hmac_sha256_different_keys() {
        let mac1 = hmac_sha256(b"key1", b"message");
        let mac2 = hmac_sha256(b"key2", b"message");
        assert_ne!(to_hex(&mac1), to_hex(&mac2));
    }

    // 7. Long key (> block size) is hashed first.
    #[test]
    fn test_hmac_sha256_long_key() {
        let key = vec![0xaau8; 131];
        let msg = b"Test Using Larger Than Block-Size Key - Hash Key First";
        let mac = hmac_sha256(&key, msg);
        let hex = to_hex(&mac);
        assert_eq!(
            hex,
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    // ── Hex helpers ───────────────────────────────────────────────────────

    // 8. to_hex/from_hex round-trip.
    #[test]
    fn test_hex_round_trip() {
        let bytes = vec![0x00u8, 0xffu8, 0xabu8, 0x12u8];
        let hex = to_hex(&bytes);
        assert_eq!(hex, "00ffab12");
        let decoded = from_hex(&hex).expect("valid hex");
        assert_eq!(decoded, bytes);
    }

    // 9. from_hex rejects invalid input.
    #[test]
    fn test_from_hex_invalid() {
        assert!(from_hex("zz").is_none());
        assert!(from_hex("0").is_none()); // odd length
    }

    // ── SigningKey ────────────────────────────────────────────────────────

    // 10. Sign and verify with matching message.
    #[test]
    fn test_signing_key_sign_verify() {
        let key = SigningKey::from_passphrase("k", "secret");
        let canonical = "/videos/sample.m3u8\n9999999999\n*";
        let sig = key.sign(canonical);
        assert!(key.verify(canonical, &sig));
    }

    // 11. Verification fails with tampered message.
    #[test]
    fn test_signing_key_verify_tampered() {
        let key = SigningKey::from_passphrase("k", "secret");
        let sig = key.sign("/path\n100\n*");
        assert!(!key.verify("/path\n101\n*", &sig));
    }

    // 12. Verification fails with wrong key.
    #[test]
    fn test_signing_key_verify_wrong_key() {
        let k1 = SigningKey::from_passphrase("k1", "secret1");
        let k2 = SigningKey::from_passphrase("k2", "secret2");
        let sig = k1.sign("/path\n100\n*");
        assert!(!k2.verify("/path\n100\n*", &sig));
    }

    // ── URL helpers ───────────────────────────────────────────────────────

    // 13. extract_path strips scheme, host and query.
    #[test]
    fn test_extract_path() {
        let url = "https://cdn.example.com/video/live.m3u8?cdn_expires=999";
        assert_eq!(extract_path(url), "/video/live.m3u8");
    }

    // 14. parse_query_params parses all parameters.
    #[test]
    fn test_parse_query_params() {
        let url = "https://cdn.example.com/p?a=1&b=hello&c=xyz";
        let params = parse_query_params(url);
        assert_eq!(params.get("a"), Some(&"1"));
        assert_eq!(params.get("b"), Some(&"hello"));
        assert_eq!(params.get("c"), Some(&"xyz"));
    }

    // ── TokenSigner / TokenValidator ──────────────────────────────────────

    // 15. Sign + validate round-trip succeeds.
    #[test]
    fn test_sign_validate_round_trip() {
        let (signer, validator) = make_signer();
        let future_ts = current_unix_ts().expect("clock ok") + 3600;
        let signed = signer
            .sign_url("https://cdn.example.com", "/video/ep1.m3u8", future_ts, "*")
            .expect("sign ok");
        let claims = validator.validate(&signed, "1.2.3.4").expect("validate ok");
        assert_eq!(claims.path, "/video/ep1.m3u8");
        assert_eq!(claims.expires, future_ts);
        assert_eq!(claims.ip_constraint, "*");
        assert_eq!(claims.key_id, "key1");
    }

    // 16. Expired token returns TokenError::Expired.
    #[test]
    fn test_expired_token() {
        let (signer, validator) = make_signer();
        let past_ts = 1_000_000_000u64; // way in the past
        let signed = signer
            .sign_url("https://cdn.example.com", "/v/file.mp4", past_ts, "*")
            .expect("sign ok");
        let err = validator.validate(&signed, "1.2.3.4").unwrap_err();
        assert!(matches!(err, TokenError::Expired { .. }), "err={err:?}");
    }

    // 17. Tampered signature returns InvalidSignature.
    #[test]
    fn test_tampered_signature() {
        let (signer, validator) = make_signer();
        let future_ts = current_unix_ts().expect("clock ok") + 3600;
        let signed = signer
            .sign_url("https://cdn.example.com", "/v/f.mp4", future_ts, "*")
            .expect("sign ok");
        // Replace the last char of cdn_sig with 'x'.
        let tampered = {
            let mut s = signed.clone();
            s.pop();
            s.push('x');
            s
        };
        let err = validator.validate(&tampered, "1.2.3.4").unwrap_err();
        assert!(matches!(err, TokenError::InvalidSignature), "err={err:?}");
    }

    // 18. validate_at with explicit timestamp: valid.
    #[test]
    fn test_validate_at_valid() {
        let (signer, validator) = make_signer();
        let expires = 2_000_000_000u64;
        let signed = signer
            .sign_url("https://cdn.example.com", "/live.m3u8", expires, "*")
            .expect("sign ok");
        let claims = validator
            .validate_at(&signed, "1.2.3.4", expires - 1)
            .expect("ok");
        assert_eq!(claims.expires, expires);
    }

    // 19. validate_at with expired timestamp.
    #[test]
    fn test_validate_at_expired() {
        let (signer, validator) = make_signer();
        let expires = 2_000_000_000u64;
        let signed = signer
            .sign_url("https://cdn.example.com", "/live.m3u8", expires, "*")
            .expect("sign ok");
        let err = validator
            .validate_at(&signed, "1.2.3.4", expires + 1)
            .unwrap_err();
        assert!(matches!(err, TokenError::Expired { .. }));
    }

    // 20. IP constraint enforced when enabled.
    #[test]
    fn test_ip_constraint_enforced() {
        let key = SigningKey::from_passphrase("k", "secret");
        let mut config = TokenConfig::default();
        config.enforce_ip = true;
        let mut validator = TokenValidator::new(config.clone());
        validator.add_key(SigningKey::from_passphrase("k", "secret"));
        let signer = TokenSigner::new(key, config);

        let expires = 2_000_000_000u64;
        let signed = signer
            .sign_url(
                "https://cdn.example.com",
                "/exclusive.mp4",
                expires,
                "10.0.0.1",
            )
            .expect("sign ok");

        // Correct IP: allowed.
        let ok = validator
            .validate_at(&signed, "10.0.0.1", expires - 1)
            .expect("should pass");
        assert_eq!(ok.ip_constraint, "10.0.0.1");

        // Wrong IP: rejected.
        let err = validator
            .validate_at(&signed, "10.0.0.2", expires - 1)
            .unwrap_err();
        assert!(matches!(err, TokenError::IpMismatch { .. }), "err={err:?}");
    }

    // 21. Wildcard IP always passes even when enforce_ip = true.
    #[test]
    fn test_wildcard_ip_passes() {
        let key = SigningKey::from_passphrase("k", "secret");
        let mut config = TokenConfig::default();
        config.enforce_ip = true;
        let mut validator = TokenValidator::new(config.clone());
        validator.add_key(SigningKey::from_passphrase("k", "secret"));
        let signer = TokenSigner::new(key, config);

        let expires = 2_000_000_000u64;
        let signed = signer
            .sign_url("https://cdn.example.com", "/pub.mp4", expires, "*")
            .expect("sign ok");

        validator
            .validate_at(&signed, "any.ip.address", expires - 1)
            .expect("wildcard always passes");
    }

    // 22. Unknown key ID returns UnknownKeyId.
    #[test]
    fn test_unknown_key_id() {
        let (signer, _) = make_signer();
        // Validator without any keys registered.
        let validator = TokenValidator::new(TokenConfig::default());
        let expires = 2_000_000_000u64;
        let signed = signer
            .sign_url("https://cdn.example.com", "/p.mp4", expires, "*")
            .expect("sign ok");
        let err = validator
            .validate_at(&signed, "1.2.3.4", expires - 1)
            .unwrap_err();
        assert!(matches!(err, TokenError::UnknownKeyId(_)), "err={err:?}");
    }

    // 23. Key rotation: add new key, rotate, old tokens still valid via old key.
    #[test]
    fn test_key_rotation() {
        let key1 = SigningKey::from_passphrase("key1", "old-secret");
        let config = TokenConfig::default();
        let mut signer = TokenSigner::new(key1, config.clone());

        let expires = 2_000_000_000u64;
        let signed_old = signer
            .sign_url("https://cdn.example.com", "/v.mp4", expires, "*")
            .expect("sign ok with key1");

        // Add and rotate to key2.
        signer.add_key(SigningKey::from_passphrase("key2", "new-secret"));
        assert!(signer.rotate_key("key2"));
        assert_eq!(signer.key_count(), 2);
        assert_eq!(signer.active_key_id(), "key2");

        // Sign with key2.
        let signed_new = signer
            .sign_url("https://cdn.example.com", "/v.mp4", expires, "*")
            .expect("sign ok with key2");

        // Validator accepts both keys.
        let mut validator = TokenValidator::new(config);
        validator.add_key(SigningKey::from_passphrase("key1", "old-secret"));
        validator.add_key(SigningKey::from_passphrase("key2", "new-secret"));

        validator
            .validate_at(&signed_old, "1.2.3.4", expires - 1)
            .expect("old token valid");
        validator
            .validate_at(&signed_new, "1.2.3.4", expires - 1)
            .expect("new token valid");
    }

    // 24. rotate_key returns false for unknown key.
    #[test]
    fn test_rotate_key_unknown() {
        let key = SigningKey::from_passphrase("k1", "s");
        let mut signer = TokenSigner::new(key, TokenConfig::default());
        assert!(!signer.rotate_key("nonexistent"));
        assert_eq!(signer.active_key_id(), "k1");
    }

    // 25. MalformedUrl when required param missing.
    #[test]
    fn test_malformed_url_missing_param() {
        let validator = {
            let mut v = TokenValidator::new(TokenConfig::default());
            v.add_key(SigningKey::from_passphrase("k", "s"));
            v
        };
        // URL without any CDN params.
        let err = validator
            .validate("https://cdn.example.com/v.mp4", "1.2.3.4")
            .unwrap_err();
        assert!(matches!(err, TokenError::MalformedUrl(_)), "err={err:?}");
    }

    // 26. sign_url_default_ttl produces a signed URL.
    #[test]
    fn test_sign_url_default_ttl() {
        let (signer, validator) = make_signer();
        let signed = signer
            .sign_url_default_ttl("https://cdn.example.com", "/stream.m3u8", "*")
            .expect("sign ok");
        // Should be a valid URL with cdn_sig.
        assert!(signed.contains("cdn_sig="));
        assert!(signed.contains("cdn_expires="));
        // Validate it.
        validator
            .validate(&signed, "1.2.3.4")
            .expect("valid default TTL token");
    }

    // 27. SignedUrlClaims::is_expired
    #[test]
    fn test_signed_url_claims_is_expired() {
        let claims = SignedUrlClaims {
            path: "/v.mp4".to_string(),
            expires: 1_000,
            ip_constraint: "*".to_string(),
            key_id: "k".to_string(),
        };
        assert!(claims.is_expired(1_000)); // now == expires: expired
        assert!(claims.is_expired(1_001));
        assert!(!claims.is_expired(999));
    }

    // 28. build_canonical produces expected string.
    #[test]
    fn test_build_canonical() {
        let c = build_canonical("/path/to/video.mp4", 1_700_000_000, "192.168.1.1");
        assert_eq!(c, "/path/to/video.mp4\n1700000000\n192.168.1.1");
    }

    // 29. constant_time_eq symmetric correctness.
    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"short", b"longer"));
    }

    // 30. TokenConfig default parameter names.
    #[test]
    fn test_token_config_defaults() {
        let cfg = TokenConfig::default();
        assert_eq!(cfg.param_expires, "cdn_expires");
        assert_eq!(cfg.param_ip, "cdn_ip");
        assert_eq!(cfg.param_key_id, "cdn_keyid");
        assert_eq!(cfg.param_sig, "cdn_sig");
        assert_eq!(cfg.default_ttl, Duration::from_secs(3600));
        assert!(!cfg.enforce_ip);
    }

    // ── TokenAuth (simple HMAC-SHA-256 signed-URL) ────────────────────────

    // 31. Valid URL: freshly signed URL verifies successfully.
    #[test]
    fn test_token_auth_valid_url() {
        let auth = TokenAuth::new(b"cdn-secret-key");
        let url = auth.sign_url("/media/livestream.m3u8", 3600);
        assert!(
            auth.verify_url(&url),
            "freshly signed URL should verify: {url}"
        );
        // Ensure format contains expected params.
        assert!(url.contains("?token="), "must contain token param");
        assert!(url.contains("&expires="), "must contain expires param");
    }

    // 32. Expired URL: zero TTL means expiry == now → rejected.
    #[test]
    fn test_token_auth_expired_url() {
        let auth = TokenAuth::new(b"cdn-secret-key");
        // A TTL of 0 sets expires = now; by the time verify_url runs now >= expires.
        let url = auth.sign_url("/media/clip.mp4", 0);
        assert!(
            !auth.verify_url(&url),
            "URL with expiry == now must be rejected: {url}"
        );
    }

    // 33. Tampered URL: modifying the token byte must fail verification.
    #[test]
    fn test_token_auth_tampered_url() {
        let auth = TokenAuth::new(b"cdn-secret-key");
        let url = auth.sign_url("/protected/asset.mp4", 3600);
        // Flip the very first hex char of the token value.
        let tampered = if url.contains("token=a") {
            url.replacen("token=a", "token=b", 1)
        } else {
            url.replacen("token=b", "token=a", 1)
        };
        // Make sure we actually changed something.
        let actually_tampered = if tampered == url {
            // The flip didn't hit — do a different substitution.
            url.replacen("token=c", "token=d", 1)
                .replacen("token=e", "token=f", 1)
        } else {
            tampered
        };
        // If the URL is unchanged it means our simple flip didn't apply;
        // in that case force a tamper on the last character of the token.
        let final_url = if actually_tampered == url {
            let token_pos = url.find("token=").expect("token param") + 6;
            let token_end = url[token_pos..]
                .find('&')
                .map(|i| token_pos + i)
                .unwrap_or(url.len());
            let mut bytes = url.into_bytes();
            // Flip the last byte of the token hex string (toggle LSB).
            let last = bytes[token_end - 1];
            bytes[token_end - 1] = if last == b'a' { b'b' } else { b'a' };
            String::from_utf8(bytes).expect("valid utf8")
        } else {
            actually_tampered
        };
        assert!(
            !auth.verify_url(&final_url),
            "tampered token must fail: {final_url}"
        );
    }
}
