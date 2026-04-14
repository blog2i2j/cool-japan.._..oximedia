//! Presigned POST support for browser-based direct uploads.
//!
//! Presigned POST allows web browsers (and other HTTP clients) to upload
//! objects directly to a cloud storage provider without routing the data
//! through your application server.  The application server generates a
//! short-lived signed policy document; the browser uses it in a multipart
//! `POST` request.
//!
//! # How it works
//!
//! 1. Your server calls `PresignedPostBuilder::build` with policy conditions
//!    and an expiration time.
//! 2. The resulting `PresignedPostData` contains a URL and a set of form
//!    fields (`policy`, `x-amz-signature`, etc.).
//! 3. Your server returns these to the browser.
//! 4. The browser constructs a `multipart/form-data` POST with the form fields
//!    plus the file content.
//!
//! This module provides the **policy document builder** and **condition types**.
//! Actual signing (HMAC-SHA256) is performed by the backend; this module
//! provides the serialized policy document that should be signed.
//!
//! # Signing
//!
//! The policy document is a Base64-encoded JSON object.  Production
//! implementations feed the raw JSON bytes through an HMAC-SHA256 function
//! keyed with the provider's signing key.  A minimal reference is included
//! here as `sign_policy_hmac_sha256`.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use chrono::{DateTime, Utc};
use std::collections::HashMap;

// ─── PolicyCondition ─────────────────────────────────────────────────────────

/// A single condition in a presigned POST policy document.
#[derive(Debug, Clone, PartialEq)]
pub enum PolicyCondition {
    /// Exact match: `{"field": "value"}`.
    Exact {
        /// Field name (e.g. `"bucket"`, `"key"`, `"Content-Type"`).
        field: String,
        /// Expected exact value.
        value: String,
    },
    /// Prefix match: `["starts-with", "$field", "prefix"]`.
    StartsWith {
        /// Field name (e.g. `"$key"`, `"$Content-Type"`).
        field: String,
        /// Required prefix.
        prefix: String,
    },
    /// Content-length range: `["content-length-range", min, max]`.
    ContentLengthRange {
        /// Minimum allowed content length in bytes.
        min_bytes: u64,
        /// Maximum allowed content length in bytes.
        max_bytes: u64,
    },
}

impl PolicyCondition {
    /// Serialise to a JSON-compatible string fragment.
    pub fn to_json_fragment(&self) -> String {
        match self {
            Self::Exact { field, value } => {
                format!("{{\"{}\":\"{}\"}}", escape_json(field), escape_json(value))
            }
            Self::StartsWith { field, prefix } => {
                format!(
                    "[\"starts-with\",\"{}\",\"{}\"]",
                    escape_json(field),
                    escape_json(prefix)
                )
            }
            Self::ContentLengthRange {
                min_bytes,
                max_bytes,
            } => {
                format!("[\"content-length-range\",{min_bytes},{max_bytes}]")
            }
        }
    }
}

// ─── PresignedPostBuilder ────────────────────────────────────────────────────

/// Builder for presigned POST policy documents.
pub struct PresignedPostBuilder {
    /// Target bucket.
    bucket: String,
    /// Key (object name) or key prefix.
    key: String,
    /// Expiration time of the policy.
    expires_at: DateTime<Utc>,
    /// Additional conditions.
    conditions: Vec<PolicyCondition>,
    /// Form fields to include in the POST (e.g. ACL, Content-Type).
    fields: HashMap<String, String>,
}

impl PresignedPostBuilder {
    /// Create a new builder for `bucket` and `key`.
    pub fn new(
        bucket: impl Into<String>,
        key: impl Into<String>,
        expires_at: DateTime<Utc>,
    ) -> Self {
        Self {
            bucket: bucket.into(),
            key: key.into(),
            expires_at,
            conditions: Vec::new(),
            fields: HashMap::new(),
        }
    }

    /// Restrict the allowed content types using a `starts-with` condition.
    pub fn allow_content_type_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.conditions.push(PolicyCondition::StartsWith {
            field: "$Content-Type".to_string(),
            prefix: prefix.into(),
        });
        self
    }

    /// Restrict content length to the given range.
    pub fn content_length_range(mut self, min_bytes: u64, max_bytes: u64) -> Self {
        self.conditions.push(PolicyCondition::ContentLengthRange {
            min_bytes,
            max_bytes,
        });
        self
    }

    /// Require an exact field value.
    pub fn require_field(mut self, field: impl Into<String>, value: impl Into<String>) -> Self {
        self.conditions.push(PolicyCondition::Exact {
            field: field.into(),
            value: value.into(),
        });
        self
    }

    /// Add an arbitrary form field to include in the upload POST.
    pub fn with_field(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.insert(name.into(), value.into());
        self
    }

    /// Build the [`PresignedPostData`] policy document (unsigned).
    ///
    /// The returned struct contains the raw JSON policy string (for signing)
    /// and the form fields that the browser must include in its POST request.
    pub fn build(self) -> PresignedPostData {
        // Build policy JSON.
        let expiration = self.expires_at.format("%Y-%m-%dT%H:%M:%SZ").to_string();

        let mut conditions_json = Vec::new();
        // Always include the bucket condition.
        conditions_json.push(format!("{{\"bucket\":\"{}\"}}", escape_json(&self.bucket)));
        // Key condition — exact match.
        conditions_json.push(format!("{{\"key\":\"{}\"}}", escape_json(&self.key)));
        // User-specified conditions.
        for cond in &self.conditions {
            conditions_json.push(cond.to_json_fragment());
        }

        let policy_json = format!(
            "{{\"expiration\":\"{expiration}\",\"conditions\":[{}]}}",
            conditions_json.join(",")
        );

        // Base64-encode the policy.
        let policy_b64 = base64_encode(policy_json.as_bytes());

        // Assemble form fields.
        let mut form_fields = self.fields.clone();
        form_fields.insert("key".to_string(), self.key.clone());
        form_fields.insert("policy".to_string(), policy_b64.clone());

        PresignedPostData {
            upload_url: format!("https://{}.s3.amazonaws.com/", self.bucket),
            form_fields,
            policy_json,
            policy_b64,
            expires_at: self.expires_at,
            bucket: self.bucket,
            key: self.key,
        }
    }
}

// ─── PresignedPostData ────────────────────────────────────────────────────────

/// The output of a presigned POST generation: URL + form fields.
///
/// Clients should POST a `multipart/form-data` request to `upload_url`
/// including all `form_fields` plus the file content as the `file` field.
#[derive(Debug, Clone)]
pub struct PresignedPostData {
    /// The URL the browser should POST to.
    pub upload_url: String,
    /// Form fields the browser must include verbatim in its POST body.
    pub form_fields: HashMap<String, String>,
    /// Raw policy JSON (for signing or inspection).
    pub policy_json: String,
    /// Base64-encoded policy (the value that should go in the `policy` field).
    pub policy_b64: String,
    /// When this presigned POST expires.
    pub expires_at: DateTime<Utc>,
    /// Target bucket.
    pub bucket: String,
    /// Target object key.
    pub key: String,
}

impl PresignedPostData {
    /// Returns `true` if the policy has not yet expired.
    pub fn is_valid(&self) -> bool {
        Utc::now() < self.expires_at
    }
}

/// Sign the policy with a simple HMAC-SHA256 reference implementation.
///
/// Returns the hex-encoded signature.  In production deployments this function
/// should be replaced by the provider's SDK signing utilities.
///
/// # Parameters
///
/// - `policy_b64` — the Base64-encoded policy string (as produced by
///   [`PresignedPostBuilder::build`]).
/// - `signing_key` — the HMAC signing key bytes derived from the provider's
///   secret key via the provider-specific key derivation procedure.
pub fn sign_policy_hmac_sha256(policy_b64: &str, signing_key: &[u8]) -> String {
    // Pure-Rust HMAC-SHA256 implementation.
    // Key padding and ipad/opad construction follow RFC 2104.
    let mut padded_key = [0u8; 64];
    let key_bytes = policy_b64.as_bytes(); // simplified: key = policy for demo
                                           // In production, signing_key is the derived secret; we use it here.
    let key_src = if signing_key.len() > 64 {
        // Hash the key if longer than block size.
        sha256(signing_key)
    } else {
        signing_key.to_vec()
    };
    let copy_len = key_src.len().min(64);
    padded_key[..copy_len].copy_from_slice(&key_src[..copy_len]);

    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= padded_key[i];
        opad[i] ^= padded_key[i];
    }

    let mut inner = Vec::with_capacity(64 + key_bytes.len());
    inner.extend_from_slice(&ipad);
    inner.extend_from_slice(key_bytes);
    let inner_hash = sha256(&inner);

    let mut outer = Vec::with_capacity(64 + 32);
    outer.extend_from_slice(&opad);
    outer.extend_from_slice(&inner_hash);
    let result = sha256(&outer);

    hex_encode(&result)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Minimal Base64 encoder (alphabet A-Z a-z 0-9 + /).
fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let combined = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((combined >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((combined >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[((combined >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(combined & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Minimal pure-Rust SHA-256 implementation (FIPS 180-4).
fn sha256(data: &[u8]) -> Vec<u8> {
    // SHA-256 initial hash values (first 32 bits of fractional parts of sqrt of first 8 primes)
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    // Round constants
    const K: [u32; 64] = [
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
    // Pre-processing: add padding
    let msg_len = data.len();
    let bit_len = (msg_len as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) block
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
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
                .wrapping_add(K[i])
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
    let mut result = Vec::with_capacity(32);
    for word in h {
        result.extend_from_slice(&word.to_be_bytes());
    }
    result
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn future_ts(secs: i64) -> DateTime<Utc> {
        Utc::now() + Duration::seconds(secs)
    }

    #[test]
    fn test_build_contains_bucket_key() {
        let data =
            PresignedPostBuilder::new("my-bucket", "uploads/video.mp4", future_ts(3600)).build();
        assert!(
            data.policy_json.contains("my-bucket"),
            "policy should contain bucket"
        );
        assert!(
            data.policy_json.contains("uploads/video.mp4"),
            "policy should contain key"
        );
    }

    #[test]
    fn test_build_contains_expiration() {
        let data = PresignedPostBuilder::new("bucket", "key", future_ts(3600)).build();
        assert!(
            data.policy_json.contains("expiration"),
            "policy should contain expiration"
        );
    }

    #[test]
    fn test_policy_b64_non_empty() {
        let data = PresignedPostBuilder::new("bucket", "key", future_ts(3600)).build();
        assert!(!data.policy_b64.is_empty());
    }

    #[test]
    fn test_form_fields_contain_key_and_policy() {
        let data = PresignedPostBuilder::new("bucket", "my-key", future_ts(3600)).build();
        assert_eq!(
            data.form_fields.get("key").map(String::as_str),
            Some("my-key")
        );
        assert!(data.form_fields.contains_key("policy"));
    }

    #[test]
    fn test_content_length_range_condition() {
        let data = PresignedPostBuilder::new("bucket", "key", future_ts(3600))
            .content_length_range(1024, 10 * 1024 * 1024)
            .build();
        assert!(
            data.policy_json.contains("content-length-range"),
            "policy should contain length range: {}",
            data.policy_json
        );
    }

    #[test]
    fn test_starts_with_condition() {
        let data = PresignedPostBuilder::new("bucket", "key", future_ts(3600))
            .allow_content_type_prefix("video/")
            .build();
        assert!(
            data.policy_json.contains("starts-with"),
            "policy should contain starts-with: {}",
            data.policy_json
        );
    }

    #[test]
    fn test_is_valid_while_not_expired() {
        let data = PresignedPostBuilder::new("bucket", "key", future_ts(3600)).build();
        assert!(data.is_valid());
    }

    #[test]
    fn test_policy_condition_exact_to_json() {
        let cond = PolicyCondition::Exact {
            field: "acl".to_string(),
            value: "public-read".to_string(),
        };
        let json = cond.to_json_fragment();
        assert!(json.contains("\"acl\""), "json: {json}");
        assert!(json.contains("\"public-read\""), "json: {json}");
    }

    #[test]
    fn test_sha256_known_value() {
        // SHA-256 of empty string
        let hash = sha256(b"");
        let hex = hex_encode(&hash);
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_base64_encode_known_value() {
        // base64("Hello") == "SGVsbG8="
        assert_eq!(base64_encode(b"Hello"), "SGVsbG8=");
        assert_eq!(base64_encode(b"Man"), "TWFu");
    }
}
