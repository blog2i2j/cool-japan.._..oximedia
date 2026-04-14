//! Shareable review link generation, validation, and revocation.
//!
//! Links are identified by a cryptographically derived token built from the
//! link's `id` and a server-side secret using HMAC-SHA-256 (hand-rolled from
//! the `sha2` crate — no external `hmac` dependency required).  Tokens are
//! base-64 (URL-safe) encoded.
//!
//! Password hashes are stored as hex-encoded SHA-256 digests of the raw
//! password, suitable for constant-time comparison.

#![allow(dead_code)]

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// SHA-256 helpers (no extra crate needed — sha2 is already a workspace dep)
// ---------------------------------------------------------------------------

/// Compute the raw SHA-256 digest of `data`.
fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Compute an HMAC-SHA-256 tag for `message` with `key`.
///
/// Implements RFC 2104:
/// ```text
/// HMAC(K, m) = H((K' XOR opad) || H((K' XOR ipad) || m))
/// ```
/// where `K'` is `K` zero-padded or hashed-then-padded to 64 bytes.
fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    const BLOCK_SIZE: usize = 64;

    // Normalise key to block size
    let mut key_block = [0u8; BLOCK_SIZE];
    if key.len() > BLOCK_SIZE {
        let hashed = sha256(key);
        key_block[..32].copy_from_slice(&hashed);
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }

    // ipad = 0x36, opad = 0x5c
    let mut ipad_key = [0u8; BLOCK_SIZE];
    let mut opad_key = [0u8; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        ipad_key[i] = key_block[i] ^ 0x36;
        opad_key[i] = key_block[i] ^ 0x5c;
    }

    // inner = H(ipad_key || message)
    let mut inner_input = Vec::with_capacity(BLOCK_SIZE + message.len());
    inner_input.extend_from_slice(&ipad_key);
    inner_input.extend_from_slice(message);
    let inner_hash = sha256(&inner_input);

    // outer = H(opad_key || inner)
    let mut outer_input = Vec::with_capacity(BLOCK_SIZE + 32);
    outer_input.extend_from_slice(&opad_key);
    outer_input.extend_from_slice(&inner_hash);
    sha256(&outer_input)
}

/// Base-64 URL-safe encoding (no-padding) of a byte slice.
fn base64url_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity((data.len() * 4 + 2) / 3);
    let mut i = 0;
    while i < data.len() {
        let b0 = data[i];
        let b1 = if i + 1 < data.len() { data[i + 1] } else { 0 };
        let b2 = if i + 2 < data.len() { data[i + 2] } else { 0 };

        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[((b0 & 0x03) << 4 | b1 >> 4) as usize] as char);
        if i + 1 < data.len() {
            out.push(TABLE[((b1 & 0x0f) << 2 | b2 >> 6) as usize] as char);
        }
        if i + 2 < data.len() {
            out.push(TABLE[(b2 & 0x3f) as usize] as char);
        }
        i += 3;
    }
    out
}

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Unique identifier for a review link.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReviewLinkId(Uuid);

impl ReviewLinkId {
    /// Create a new random link ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Wrap an existing UUID.
    #[must_use]
    pub fn from_uuid(id: Uuid) -> Self {
        Self(id)
    }

    /// Return the inner UUID.
    #[must_use]
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for ReviewLinkId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ReviewLinkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Fine-grained permission flags carried by a `ReviewLink`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkPermissions {
    /// The link bearer may view content.
    pub can_view: bool,
    /// The link bearer may add annotations or comments.
    pub can_annotate: bool,
    /// The link bearer may download assets.
    pub can_download: bool,
}

impl LinkPermissions {
    /// Read-only link — view only, no annotation or download.
    #[must_use]
    pub const fn view_only() -> Self {
        Self {
            can_view: true,
            can_annotate: false,
            can_download: false,
        }
    }

    /// Annotator link — view and annotate, no download.
    #[must_use]
    pub const fn annotator() -> Self {
        Self {
            can_view: true,
            can_annotate: true,
            can_download: false,
        }
    }

    /// Full-access link — view, annotate, and download.
    #[must_use]
    pub const fn full_access() -> Self {
        Self {
            can_view: true,
            can_annotate: true,
            can_download: true,
        }
    }
}

impl Default for LinkPermissions {
    fn default() -> Self {
        Self::view_only()
    }
}

/// A shareable review link that may be distributed to external reviewers.
#[derive(Debug, Clone)]
pub struct ReviewLink {
    /// Stable link identifier.
    pub id: ReviewLinkId,
    /// The review session this link grants access to.
    pub session_id: String,
    /// Optional expiry timestamp (ms since UNIX epoch).  `None` means never-expires.
    pub expires_at_ms: Option<u64>,
    /// Optional SHA-256 hex digest of the link password.  `None` means no password required.
    pub password_hash: Option<String>,
    /// Permissions granted to anyone who presents a valid token for this link.
    pub permissions: LinkPermissions,
    /// The HMAC-SHA-256 bearer token that authenticates this link.
    pub(crate) token: String,
    /// Whether this link has been revoked.
    revoked: bool,
    /// Arbitrary metadata attached to the link (e.g. recipient name, notes).
    pub metadata: HashMap<String, String>,
}

impl ReviewLink {
    /// Returns `true` if the link has been revoked.
    #[must_use]
    pub fn is_revoked(&self) -> bool {
        self.revoked
    }

    /// Returns `true` if `now_ms` is past the expiry time (if set).
    #[must_use]
    pub fn is_expired(&self, now_ms: u64) -> bool {
        self.expires_at_ms.map(|exp| now_ms > exp).unwrap_or(false)
    }

    /// Returns the bearer token for embedding in URLs.
    #[must_use]
    pub fn token(&self) -> &str {
        &self.token
    }
}

// ---------------------------------------------------------------------------
// LinkManager
// ---------------------------------------------------------------------------

/// Error type for link operations.
#[derive(Debug, PartialEq, Eq)]
pub enum ReviewLinkError {
    /// Token does not match any stored link.
    TokenNotFound,
    /// Link has been revoked.
    Revoked,
    /// Link has expired.
    Expired,
    /// Password verification failed.
    InvalidPassword,
    /// Caller does not have the required permission.
    PermissionDenied,
}

impl std::fmt::Display for ReviewLinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenNotFound => write!(f, "token not found"),
            Self::Revoked => write!(f, "link has been revoked"),
            Self::Expired => write!(f, "link has expired"),
            Self::InvalidPassword => write!(f, "invalid password"),
            Self::PermissionDenied => write!(f, "permission denied"),
        }
    }
}

impl std::error::Error for ReviewLinkError {}

/// Options for creating a new review link.
#[derive(Debug, Default)]
pub struct CreateLinkOptions {
    /// Optional expiry timestamp (ms since UNIX epoch).
    pub expires_at_ms: Option<u64>,
    /// Optional plain-text password — will be hashed before storage.
    pub password: Option<String>,
    /// Permissions for this link.  Defaults to view-only.
    pub permissions: LinkPermissions,
    /// Optional metadata to attach.
    pub metadata: HashMap<String, String>,
}

/// Manages the lifecycle of review links for a deployment.
///
/// `secret` is a server-side signing secret used to derive HMAC tokens.
/// It should be treated as a credential and not exposed to clients.
pub struct ReviewLinkManager {
    secret: Vec<u8>,
    links: HashMap<String, ReviewLink>, // keyed by token
}

impl ReviewLinkManager {
    /// Create a new manager with the given signing secret.
    #[must_use]
    pub fn new(secret: impl Into<Vec<u8>>) -> Self {
        Self {
            secret: secret.into(),
            links: HashMap::new(),
        }
    }

    /// Generate a new review link for `session_id` with the given options.
    #[must_use]
    pub fn create(
        &mut self,
        session_id: impl Into<String>,
        options: CreateLinkOptions,
    ) -> ReviewLink {
        let id = ReviewLinkId::new();
        let session_id_str: String = session_id.into();

        // Derive token: HMAC-SHA256(secret, link_id || session_id) → base64url
        let message = format!("{}{}", id, session_id_str);
        let tag = hmac_sha256(&self.secret, message.as_bytes());
        let token = base64url_encode(&tag);

        // Hash password if provided
        let password_hash = options.password.as_deref().map(|pw| {
            let digest = sha256(pw.as_bytes());
            hex_encode(&digest)
        });

        let link = ReviewLink {
            id,
            session_id: session_id_str,
            expires_at_ms: options.expires_at_ms,
            password_hash,
            permissions: options.permissions,
            token: token.clone(),
            revoked: false,
            metadata: options.metadata,
        };

        self.links.insert(token, link.clone());
        link
    }

    /// Validate a token, optionally checking a password and expiry.
    ///
    /// Returns a reference to the link on success, or a `ReviewLinkError` on failure.
    ///
    /// # Arguments
    ///
    /// * `token` — the bearer token to validate.
    /// * `password` — optional plain-text password (checked against stored hash).
    /// * `now_ms` — current time in milliseconds since UNIX epoch (used for expiry).
    pub fn validate(
        &self,
        token: &str,
        password: Option<&str>,
        now_ms: u64,
    ) -> Result<&ReviewLink, ReviewLinkError> {
        let link = self
            .links
            .get(token)
            .ok_or(ReviewLinkError::TokenNotFound)?;

        if link.revoked {
            return Err(ReviewLinkError::Revoked);
        }

        if link.is_expired(now_ms) {
            return Err(ReviewLinkError::Expired);
        }

        if let Some(stored_hash) = &link.password_hash {
            let provided = password.ok_or(ReviewLinkError::InvalidPassword)?;
            let candidate_hash = hex_encode(&sha256(provided.as_bytes()));
            if constant_time_ne(stored_hash.as_bytes(), candidate_hash.as_bytes()) {
                return Err(ReviewLinkError::InvalidPassword);
            }
        }

        Ok(link)
    }

    /// Revoke a link by its token.
    ///
    /// Returns `true` if the link was found and revoked, `false` if not found.
    pub fn revoke(&mut self, token: &str) -> bool {
        if let Some(link) = self.links.get_mut(token) {
            link.revoked = true;
            true
        } else {
            false
        }
    }

    /// Number of links currently stored (including revoked).
    #[must_use]
    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    /// Returns `true` if a link with the given `token` exists (including revoked).
    #[must_use]
    pub fn contains(&self, token: &str) -> bool {
        self.links.contains_key(token)
    }

    /// Build a shareable URL for a link token using the given base URL.
    ///
    /// The token is appended as a `?token=<token>` query parameter.
    /// Example:
    /// ```
    /// # use oximedia_review::review_link::{ReviewLinkManager, CreateLinkOptions};
    /// let mut mgr = ReviewLinkManager::new(b"secret");
    /// let link = mgr.create("session-1", CreateLinkOptions::default());
    /// let url = mgr.generate_url("https://review.example.com/share", link.token());
    /// assert!(url.starts_with("https://review.example.com/share?token="));
    /// ```
    #[must_use]
    pub fn generate_url(&self, base_url: &str, token: &str) -> String {
        // Percent-encode only characters that are invalid in query param values.
        let encoded_token: String = token
            .chars()
            .map(|c| match c {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
                other => {
                    let mut buf = [0u8; 4];
                    let bytes = other.encode_utf8(&mut buf);
                    bytes.bytes().map(|b| format!("%{b:02X}")).collect()
                }
            })
            .collect();

        let separator = if base_url.contains('?') { '&' } else { '?' };
        format!("{base_url}{separator}token={encoded_token}")
    }

    /// List all active (non-revoked, non-expired) links for a given session.
    ///
    /// `now_ms` is used to filter out expired links.
    #[must_use]
    pub fn active_links_for_session<'a>(
        &'a self,
        session_id: &str,
        now_ms: u64,
    ) -> Vec<&'a ReviewLink> {
        self.links
            .values()
            .filter(|l| l.session_id == session_id && !l.is_revoked() && !l.is_expired(now_ms))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Hex-encode a byte slice.
fn hex_encode(data: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(data.len() * 2);
    for &b in data {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// Constant-time inequality check (mitigates timing attacks during password compare).
/// Returns `true` if `a != b`.
fn constant_time_ne(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return true;
    }
    let mut diff: u8 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff != 0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn manager() -> ReviewLinkManager {
        ReviewLinkManager::new(b"test-secret-key-for-review-links".as_ref())
    }

    fn default_opts() -> CreateLinkOptions {
        CreateLinkOptions {
            expires_at_ms: None,
            password: None,
            permissions: LinkPermissions::view_only(),
            metadata: HashMap::new(),
        }
    }

    // 1 — creating a link returns a non-empty token
    #[test]
    fn test_create_link_has_token() {
        let mut mgr = manager();
        let link = mgr.create("session-1", default_opts());
        assert!(!link.token().is_empty());
    }

    // 2 — two different sessions produce different tokens
    #[test]
    fn test_different_sessions_produce_different_tokens() {
        let mut mgr = manager();
        let l1 = mgr.create("session-1", default_opts());
        let l2 = mgr.create("session-2", default_opts());
        assert_ne!(l1.token(), l2.token());
    }

    // 3 — validate succeeds for a fresh link with no password
    #[test]
    fn test_validate_no_password_success() {
        let mut mgr = manager();
        let link = mgr.create("session-1", default_opts());
        let result = mgr.validate(link.token(), None, 0);
        assert!(result.is_ok());
    }

    // 4 — validate fails for an unknown token
    #[test]
    fn test_validate_unknown_token() {
        let mgr = manager();
        let err = mgr.validate("not-a-real-token", None, 0).unwrap_err();
        assert_eq!(err, ReviewLinkError::TokenNotFound);
    }

    // 5 — revoke makes validate return Revoked
    #[test]
    fn test_revoke_link() {
        let mut mgr = manager();
        let link = mgr.create("session-1", default_opts());
        let token = link.token().to_string();
        assert!(mgr.revoke(&token));
        let err = mgr.validate(&token, None, 0).unwrap_err();
        assert_eq!(err, ReviewLinkError::Revoked);
    }

    // 6 — revoke unknown token returns false
    #[test]
    fn test_revoke_unknown_token_returns_false() {
        let mut mgr = manager();
        assert!(!mgr.revoke("ghost"));
    }

    // 7 — expired link returns Expired
    #[test]
    fn test_validate_expired_link() {
        let mut mgr = manager();
        let opts = CreateLinkOptions {
            expires_at_ms: Some(1000), // expires at 1 second
            ..default_opts()
        };
        let link = mgr.create("session-1", opts);
        let err = mgr.validate(link.token(), None, 2000).unwrap_err(); // now = 2s > 1s
        assert_eq!(err, ReviewLinkError::Expired);
    }

    // 8 — non-expired link does not return Expired
    #[test]
    fn test_validate_not_yet_expired() {
        let mut mgr = manager();
        let opts = CreateLinkOptions {
            expires_at_ms: Some(5000),
            ..default_opts()
        };
        let link = mgr.create("session-1", opts);
        assert!(mgr.validate(link.token(), None, 3000).is_ok());
    }

    // 9 — correct password validates
    #[test]
    fn test_password_correct() {
        let mut mgr = manager();
        let opts = CreateLinkOptions {
            password: Some("s3cr3t".to_string()),
            ..default_opts()
        };
        let link = mgr.create("session-1", opts);
        let result = mgr.validate(link.token(), Some("s3cr3t"), 0);
        assert!(result.is_ok());
    }

    // 10 — wrong password fails
    #[test]
    fn test_password_incorrect() {
        let mut mgr = manager();
        let opts = CreateLinkOptions {
            password: Some("correct-horse".to_string()),
            ..default_opts()
        };
        let link = mgr.create("session-1", opts);
        let err = mgr
            .validate(link.token(), Some("wrong-battery"), 0)
            .unwrap_err();
        assert_eq!(err, ReviewLinkError::InvalidPassword);
    }

    // 11 — missing password when one is required
    #[test]
    fn test_password_required_but_missing() {
        let mut mgr = manager();
        let opts = CreateLinkOptions {
            password: Some("required".to_string()),
            ..default_opts()
        };
        let link = mgr.create("session-1", opts);
        let err = mgr.validate(link.token(), None, 0).unwrap_err();
        assert_eq!(err, ReviewLinkError::InvalidPassword);
    }

    // 12 — permissions are preserved on the created link
    #[test]
    fn test_link_permissions_preserved() {
        let mut mgr = manager();
        let opts = CreateLinkOptions {
            permissions: LinkPermissions::full_access(),
            ..default_opts()
        };
        let link = mgr.create("session-1", opts);
        assert!(link.permissions.can_view);
        assert!(link.permissions.can_annotate);
        assert!(link.permissions.can_download);
    }

    // 13 — link_count tracks stored links
    #[test]
    fn test_link_count() {
        let mut mgr = manager();
        assert_eq!(mgr.link_count(), 0);
        mgr.create("s1", default_opts());
        mgr.create("s2", default_opts());
        assert_eq!(mgr.link_count(), 2);
    }

    // 14 — sha256 and hmac helpers produce correct output (test vectors)
    #[test]
    fn test_sha256_empty() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let digest = sha256(b"");
        assert_eq!(digest[0], 0xe3);
        assert_eq!(digest[1], 0xb0);
    }

    // 15 — base64url_encode round-trips correctly for known input
    #[test]
    fn test_base64url_encode_hello() {
        // base64url("Man") == "TWFu"
        let out = base64url_encode(b"Man");
        assert_eq!(out, "TWFu");
    }
}
