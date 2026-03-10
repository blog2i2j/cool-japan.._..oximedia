//! DRM license token management.
//!
//! Provides token creation, validation, revocation, and permission checking.

use std::collections::HashSet;

/// Permissions that can be granted by a license token.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LicensePermission {
    /// Standard playback.
    Play,
    /// Download for offline use.
    Download,
    /// Share with other users.
    Share,
    /// HDCP enforcement at the given level (e.g. `2` for HDCP 2.x).
    Hdcp(u32),
    /// Analog/digital output protection required.
    OutputProtection,
}

/// A time-bounded license token granting specific permissions on content.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct LicenseToken {
    pub token_id: String,
    pub content_id: String,
    pub user_id: String,
    pub issued_at: u64,
    pub expires_at: u64,
    pub permissions: Vec<LicensePermission>,
}

/// Simple counter for unique token IDs within a process.
fn next_token_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(1);
    format!("tok-{}", CTR.fetch_add(1, Ordering::Relaxed))
}

impl LicenseToken {
    /// Create a new token issued at `issued_at` that expires after `ttl_s` seconds.
    pub fn new(content_id: &str, user_id: &str, issued_at: u64, ttl_s: u64) -> Self {
        Self {
            token_id: next_token_id(),
            content_id: content_id.to_string(),
            user_id: user_id.to_string(),
            issued_at,
            expires_at: issued_at.saturating_add(ttl_s),
            permissions: Vec::new(),
        }
    }

    /// Builder: add a permission to the token.
    pub fn with_permission(mut self, perm: LicensePermission) -> Self {
        self.permissions.push(perm);
        self
    }

    /// Builder: replace all permissions.
    pub fn with_permissions(mut self, perms: Vec<LicensePermission>) -> Self {
        self.permissions = perms;
        self
    }

    /// Returns `true` if `now` is at or past the expiry timestamp.
    pub fn is_expired(&self, now: u64) -> bool {
        now >= self.expires_at
    }

    /// Returns `true` if this token grants the requested permission.
    ///
    /// For `Hdcp(level)`, matching checks that the token contains an `Hdcp`
    /// entry with a level *at least as restrictive* (i.e. `>= level`).
    pub fn has_permission(&self, perm: &LicensePermission) -> bool {
        match perm {
            LicensePermission::Hdcp(required) => self
                .permissions
                .iter()
                .any(|p| matches!(p, LicensePermission::Hdcp(level) if *level >= *required)),
            other => self.permissions.contains(other),
        }
    }

    /// Returns the number of seconds remaining before expiry, or `None` if
    /// already expired.
    pub fn remaining_s(&self, now: u64) -> Option<u64> {
        if now >= self.expires_at {
            None
        } else {
            Some(self.expires_at - now)
        }
    }
}

/// Result of token validation.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationResult {
    Valid,
    Expired,
    Revoked,
    Invalid(String),
}

/// Validates license tokens against a revocation list.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct TokenValidator {
    revoked_ids: HashSet<String>,
}

impl TokenValidator {
    /// Create a new validator with an empty revocation list.
    pub fn new() -> Self {
        Self {
            revoked_ids: HashSet::new(),
        }
    }

    /// Revoke a token by its ID.
    pub fn revoke(&mut self, token_id: &str) {
        self.revoked_ids.insert(token_id.to_string());
    }

    /// Validate a token at the given current timestamp.
    pub fn validate(&self, token: &LicenseToken, now: u64) -> ValidationResult {
        if self.revoked_ids.contains(&token.token_id) {
            return ValidationResult::Revoked;
        }
        if token.is_expired(now) {
            return ValidationResult::Expired;
        }
        if token.content_id.is_empty() {
            return ValidationResult::Invalid("empty content_id".to_string());
        }
        if token.user_id.is_empty() {
            return ValidationResult::Invalid("empty user_id".to_string());
        }
        ValidationResult::Valid
    }
}

/// Claims carried inside a DRM access token.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenClaims {
    /// Subject (user) identifier.
    pub subject: String,
    /// Content identifier this token grants access to.
    pub content_id: String,
    /// Unix-millisecond timestamp when the token was issued.
    pub issued_at_ms: u64,
    /// Unix-millisecond timestamp when the token expires.
    pub expires_at_ms: u64,
    /// IP addresses explicitly allowed to use this token; empty means any IP.
    pub allowed_ips: Vec<String>,
    /// Maximum number of plays permitted; `None` means unlimited.
    pub max_plays: Option<u32>,
}

impl TokenClaims {
    /// Create new claims.
    pub fn new(
        subject: impl Into<String>,
        content_id: impl Into<String>,
        issued_at_ms: u64,
        expires_at_ms: u64,
    ) -> Self {
        Self {
            subject: subject.into(),
            content_id: content_id.into(),
            issued_at_ms,
            expires_at_ms,
            allowed_ips: Vec::new(),
            max_plays: None,
        }
    }

    /// Returns `true` when `now_ms >= expires_at_ms`.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms >= self.expires_at_ms
    }

    /// Returns `true` when the IP is allowed.
    ///
    /// An empty `allowed_ips` list means any IP is permitted.
    pub fn allows_ip(&self, ip: &str) -> bool {
        self.allowed_ips.is_empty() || self.allowed_ips.iter().any(|a| a == ip)
    }
}

/// Result of validating a DRM access token.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenStatus {
    /// Token is valid and access should be granted.
    Valid,
    /// Token has passed its expiry time.
    Expired,
    /// Token has been explicitly revoked.
    Revoked,
    /// Allowed play count has been exhausted.
    ExceededPlays,
}

impl TokenStatus {
    /// Returns `true` only for the `Valid` variant.
    #[must_use]
    pub fn is_valid(self) -> bool {
        matches!(self, TokenStatus::Valid)
    }
}

/// A simplified JWT-like structure (no real cryptography).
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JwtLike {
    /// Base-64 encoded header.
    pub header_b64: String,
    /// Base-64 encoded payload.
    pub payload_b64: String,
    /// Stub signature (not cryptographically secure).
    pub signature_stub: String,
}

impl JwtLike {
    /// Encode `claims` into a `JwtLike`.
    ///
    /// The encoding is simplified: base-64 of a `key=value;` string.
    pub fn encode(claims: &TokenClaims) -> Self {
        let header = base64_encode("oximedia-drm.v1");
        let payload_raw = format!(
            "sub={};cid={};iat={};exp={};ips={};max_plays={}",
            claims.subject,
            claims.content_id,
            claims.issued_at_ms,
            claims.expires_at_ms,
            claims.allowed_ips.join(","),
            claims
                .max_plays
                .map_or_else(|| "none".to_string(), |n| n.to_string()),
        );
        let payload = base64_encode(&payload_raw);
        // Stub signature: FNV-1a hash of payload bytes, encoded as hex.
        let sig = fnv1a_hex(payload.as_bytes());
        Self {
            header_b64: header,
            payload_b64: payload,
            signature_stub: sig,
        }
    }

    /// Decode a `JwtLike` back into [`TokenClaims`], returning `None` on parse
    /// failure.
    pub fn decode(token: &JwtLike) -> Option<TokenClaims> {
        let raw = base64_decode(&token.payload_b64)?;
        let mut sub = String::new();
        let mut cid = String::new();
        let mut iat: u64 = 0;
        let mut exp: u64 = 0;
        let mut ips: Vec<String> = Vec::new();
        let mut max_plays: Option<u32> = None;

        for part in raw.split(';') {
            if let Some(val) = part.strip_prefix("sub=") {
                sub = val.to_string();
            } else if let Some(val) = part.strip_prefix("cid=") {
                cid = val.to_string();
            } else if let Some(val) = part.strip_prefix("iat=") {
                iat = val.parse().ok()?;
            } else if let Some(val) = part.strip_prefix("exp=") {
                exp = val.parse().ok()?;
            } else if let Some(val) = part.strip_prefix("ips=") {
                if !val.is_empty() {
                    ips = val.split(',').map(str::to_string).collect();
                }
            } else if let Some(val) = part.strip_prefix("max_plays=") {
                if val != "none" {
                    max_plays = val.parse().ok();
                }
            }
        }

        if sub.is_empty() || cid.is_empty() {
            return None;
        }

        Some(TokenClaims {
            subject: sub,
            content_id: cid,
            issued_at_ms: iat,
            expires_at_ms: exp,
            allowed_ips: ips,
            max_plays,
        })
    }
}

// Minimal base-64-like encoding (uses standard chars but no padding trimming).
fn base64_encode(input: &str) -> String {
    // Use the `base64` alphabet manually for a zero-dependency path.
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = input.as_bytes();
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let combined = (b0 << 16) | (b1 << 8) | b2;
        out.push(CHARS[((combined >> 18) & 0x3f) as usize] as char);
        out.push(CHARS[((combined >> 12) & 0x3f) as usize] as char);
        out.push(if chunk.len() > 1 {
            CHARS[((combined >> 6) & 0x3f) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            CHARS[(combined & 0x3f) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn base64_decode(input: &str) -> Option<String> {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out: Vec<u8> = Vec::new();
    let bytes: Vec<u8> = input.bytes().collect();
    for chunk in bytes.chunks(4) {
        if chunk.len() < 4 {
            break;
        }
        let decode_char =
            |c: u8| -> Option<u32> { CHARS.iter().position(|&x| x == c).map(|p| p as u32) };
        let c0 = decode_char(chunk[0])?;
        let c1 = decode_char(chunk[1])?;
        out.push(((c0 << 2) | (c1 >> 4)) as u8);
        if chunk[2] != b'=' {
            let c2 = decode_char(chunk[2])?;
            out.push(((c1 << 4) | (c2 >> 2)) as u8);
        }
        if chunk[3] != b'=' {
            let c2 = decode_char(chunk[2]).unwrap_or(0);
            let c3 = decode_char(chunk[3])?;
            out.push(((c2 << 6) | c3) as u8);
        }
    }
    String::from_utf8(out).ok()
}

fn fnv1a_hex(data: &[u8]) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{:016x}", hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_token() -> LicenseToken {
        LicenseToken::new("content-42", "user-1", 1000, 3600)
    }

    #[test]
    fn test_token_creation_fields() {
        let t = make_token();
        assert_eq!(t.content_id, "content-42");
        assert_eq!(t.user_id, "user-1");
        assert_eq!(t.issued_at, 1000);
        assert_eq!(t.expires_at, 4600);
    }

    #[test]
    fn test_token_not_expired() {
        let t = make_token();
        assert!(!t.is_expired(1000));
        assert!(!t.is_expired(4599));
    }

    #[test]
    fn test_token_expired() {
        let t = make_token();
        assert!(t.is_expired(4600));
        assert!(t.is_expired(9999));
    }

    #[test]
    fn test_has_permission_play() {
        let t = make_token().with_permission(LicensePermission::Play);
        assert!(t.has_permission(&LicensePermission::Play));
        assert!(!t.has_permission(&LicensePermission::Download));
    }

    #[test]
    fn test_has_permission_hdcp() {
        let t = make_token().with_permission(LicensePermission::Hdcp(2));
        // Token has level 2, require level 1 -> ok
        assert!(t.has_permission(&LicensePermission::Hdcp(1)));
        // Token has level 2, require level 2 -> ok
        assert!(t.has_permission(&LicensePermission::Hdcp(2)));
        // Token has level 2, require level 3 -> not ok
        assert!(!t.has_permission(&LicensePermission::Hdcp(3)));
    }

    #[test]
    fn test_remaining_s() {
        let t = make_token();
        assert_eq!(t.remaining_s(1000), Some(3600));
        assert_eq!(t.remaining_s(4000), Some(600));
        assert_eq!(t.remaining_s(4600), None);
        assert_eq!(t.remaining_s(5000), None);
    }

    #[test]
    fn test_validator_valid() {
        let validator = TokenValidator::new();
        let t = make_token();
        assert_eq!(validator.validate(&t, 2000), ValidationResult::Valid);
    }

    #[test]
    fn test_validator_expired() {
        let validator = TokenValidator::new();
        let t = make_token();
        assert_eq!(validator.validate(&t, 9999), ValidationResult::Expired);
    }

    #[test]
    fn test_validator_revoked() {
        let mut validator = TokenValidator::new();
        let t = make_token();
        validator.revoke(&t.token_id);
        // Revocation takes priority over expiry check
        assert_eq!(validator.validate(&t, 2000), ValidationResult::Revoked);
    }

    #[test]
    fn test_validator_invalid_empty_content() {
        let validator = TokenValidator::new();
        let mut t = make_token();
        t.content_id = String::new();
        assert!(matches!(
            validator.validate(&t, 2000),
            ValidationResult::Invalid(_)
        ));
    }

    #[test]
    fn test_validator_invalid_empty_user() {
        let validator = TokenValidator::new();
        let mut t = make_token();
        t.user_id = String::new();
        assert!(matches!(
            validator.validate(&t, 2000),
            ValidationResult::Invalid(_)
        ));
    }

    #[test]
    fn test_with_permissions_replaces() {
        let t = make_token()
            .with_permission(LicensePermission::Play)
            .with_permissions(vec![LicensePermission::Download]);
        assert!(!t.has_permission(&LicensePermission::Play));
        assert!(t.has_permission(&LicensePermission::Download));
    }

    #[test]
    fn test_multiple_tokens_unique_ids() {
        let t1 = LicenseToken::new("c1", "u1", 0, 100);
        let t2 = LicenseToken::new("c2", "u2", 0, 100);
        assert_ne!(t1.token_id, t2.token_id);
    }

    #[test]
    fn test_output_protection_permission() {
        let t = make_token().with_permission(LicensePermission::OutputProtection);
        assert!(t.has_permission(&LicensePermission::OutputProtection));
        assert!(!t.has_permission(&LicensePermission::Share));
    }

    // ----- TokenClaims tests -----

    #[test]
    fn test_token_claims_is_expired_false() {
        let c = TokenClaims::new("user1", "movie1", 0, 10_000);
        assert!(!c.is_expired(9_999));
    }

    #[test]
    fn test_token_claims_is_expired_true() {
        let c = TokenClaims::new("user1", "movie1", 0, 10_000);
        assert!(c.is_expired(10_000));
        assert!(c.is_expired(20_000));
    }

    #[test]
    fn test_token_claims_allows_ip_empty_list() {
        let c = TokenClaims::new("u", "c", 0, 100);
        // No restrictions -> any IP allowed
        assert!(c.allows_ip("1.2.3.4"));
        assert!(c.allows_ip("192.168.0.1"));
    }

    #[test]
    fn test_token_claims_allows_ip_restricted_match() {
        let mut c = TokenClaims::new("u", "c", 0, 100);
        c.allowed_ips = vec!["10.0.0.1".to_string()];
        assert!(c.allows_ip("10.0.0.1"));
        assert!(!c.allows_ip("10.0.0.2"));
    }

    // ----- TokenStatus tests -----

    #[test]
    fn test_token_status_is_valid() {
        assert!(TokenStatus::Valid.is_valid());
        assert!(!TokenStatus::Expired.is_valid());
        assert!(!TokenStatus::Revoked.is_valid());
        assert!(!TokenStatus::ExceededPlays.is_valid());
    }

    // ----- JwtLike encode/decode tests -----

    #[test]
    fn test_jwt_like_encode_decode_roundtrip() {
        let claims = TokenClaims::new("alice", "film-42", 1_000_000, 2_000_000);
        let jwt = JwtLike::encode(&claims);
        let decoded = JwtLike::decode(&jwt).expect("decode should succeed");
        assert_eq!(decoded.subject, "alice");
        assert_eq!(decoded.content_id, "film-42");
        assert_eq!(decoded.issued_at_ms, 1_000_000);
        assert_eq!(decoded.expires_at_ms, 2_000_000);
    }

    #[test]
    fn test_jwt_like_encode_decode_with_ips() {
        let mut claims = TokenClaims::new("bob", "show-1", 0, 9999);
        claims.allowed_ips = vec!["127.0.0.1".to_string()];
        let jwt = JwtLike::encode(&claims);
        let decoded = JwtLike::decode(&jwt).expect("operation should succeed");
        assert!(decoded.allowed_ips.contains(&"127.0.0.1".to_string()));
    }

    #[test]
    fn test_jwt_like_encode_decode_max_plays() {
        let mut claims = TokenClaims::new("carol", "vid-5", 100, 200);
        claims.max_plays = Some(3);
        let jwt = JwtLike::encode(&claims);
        let decoded = JwtLike::decode(&jwt).expect("operation should succeed");
        assert_eq!(decoded.max_plays, Some(3));
    }

    #[test]
    fn test_jwt_like_encode_decode_no_max_plays() {
        let claims = TokenClaims::new("dan", "doc-7", 0, 1000);
        let jwt = JwtLike::encode(&claims);
        let decoded = JwtLike::decode(&jwt).expect("operation should succeed");
        assert_eq!(decoded.max_plays, None);
    }

    #[test]
    fn test_jwt_like_signature_not_empty() {
        let claims = TokenClaims::new("user", "c", 0, 1000);
        let jwt = JwtLike::encode(&claims);
        assert!(!jwt.signature_stub.is_empty());
    }

    #[test]
    fn test_jwt_like_different_contents_differ() {
        let c1 = TokenClaims::new("user", "content-a", 0, 1000);
        let c2 = TokenClaims::new("user", "content-b", 0, 1000);
        let j1 = JwtLike::encode(&c1);
        let j2 = JwtLike::encode(&c2);
        assert_ne!(j1.payload_b64, j2.payload_b64);
    }
}
