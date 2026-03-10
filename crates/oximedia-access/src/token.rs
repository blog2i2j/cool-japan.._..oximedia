//! Token management for access control.
//!
//! Provides JWT-like tokens with expiry and refresh capabilities for
//! authenticating users and services in media production pipelines.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

/// A set of claims embedded in a token.
#[derive(Debug, Clone)]
pub struct TokenClaims {
    /// Subject (user or service identifier)
    pub sub: String,
    /// Issuer
    pub iss: String,
    /// Issued-at timestamp (Unix seconds)
    pub iat: u64,
    /// Expiry timestamp (Unix seconds)
    pub exp: u64,
    /// Token identifier (jti)
    pub jti: String,
    /// Roles granted to this token
    pub roles: Vec<String>,
    /// Additional custom claims
    pub custom: HashMap<String, String>,
}

impl TokenClaims {
    /// Returns true if the token is expired relative to now.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        let now = current_unix_secs();
        now >= self.exp
    }

    /// Returns true if the token is currently valid (not expired).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.is_expired()
    }

    /// Seconds remaining until expiry. Returns 0 if already expired.
    #[must_use]
    pub fn seconds_remaining(&self) -> u64 {
        let now = current_unix_secs();
        self.exp.saturating_sub(now)
    }

    /// Get a custom claim value.
    #[must_use]
    pub fn get_custom(&self, key: &str) -> Option<&str> {
        self.custom.get(key).map(String::as_str)
    }
}

/// A signed token containing claims and a signature.
#[derive(Debug, Clone)]
pub struct Token {
    /// Token claims
    pub claims: TokenClaims,
    /// Simulated HMAC signature (hash of claims + secret)
    pub signature: u64,
}

impl Token {
    /// Verify the token signature using the provided secret.
    #[must_use]
    pub fn verify_signature(&self, secret: &str) -> bool {
        let expected = compute_signature(&self.claims, secret);
        self.signature == expected
    }

    /// Returns true if the token is valid (not expired and signature matches).
    #[must_use]
    pub fn is_valid(&self, secret: &str) -> bool {
        self.claims.is_valid() && self.verify_signature(secret)
    }
}

/// Token type produced by the manager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    /// Short-lived access token
    Access,
    /// Long-lived refresh token
    Refresh,
}

/// Builder for constructing tokens.
#[derive(Debug)]
pub struct TokenBuilder {
    sub: String,
    iss: String,
    ttl_secs: u64,
    roles: Vec<String>,
    custom: HashMap<String, String>,
    kind: TokenKind,
    secret: String,
}

impl TokenBuilder {
    /// Create a new token builder.
    #[must_use]
    pub fn new(sub: impl Into<String>, secret: impl Into<String>) -> Self {
        Self {
            sub: sub.into(),
            iss: "oximedia-access".to_string(),
            ttl_secs: 3600,
            roles: Vec::new(),
            custom: HashMap::new(),
            kind: TokenKind::Access,
            secret: secret.into(),
        }
    }

    /// Set the issuer.
    #[must_use]
    pub fn issuer(mut self, iss: impl Into<String>) -> Self {
        self.iss = iss.into();
        self
    }

    /// Set the token TTL in seconds.
    #[must_use]
    pub fn ttl(mut self, secs: u64) -> Self {
        self.ttl_secs = secs;
        self
    }

    /// Set the token kind.
    #[must_use]
    pub fn kind(mut self, kind: TokenKind) -> Self {
        self.kind = kind;
        self
    }

    /// Add a role to the token claims.
    #[must_use]
    pub fn role(mut self, role: impl Into<String>) -> Self {
        self.roles.push(role.into());
        self
    }

    /// Add a custom claim.
    #[must_use]
    pub fn custom_claim(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }

    /// Build and sign the token.
    #[must_use]
    pub fn build(self) -> Token {
        let now = current_unix_secs();
        let jti = format!("{}-{}", self.sub, now);
        let claims = TokenClaims {
            sub: self.sub,
            iss: self.iss,
            iat: now,
            exp: now + self.ttl_secs,
            jti,
            roles: self.roles,
            custom: self.custom,
        };
        let signature = compute_signature(&claims, &self.secret);
        Token { claims, signature }
    }
}

/// Manages token issuance, validation, and revocation.
#[derive(Debug)]
pub struct TokenManager {
    secret: String,
    revoked_jtis: HashSet<String>,
    access_ttl_secs: u64,
    refresh_ttl_secs: u64,
}

impl TokenManager {
    /// Create a new token manager.
    #[must_use]
    pub fn new(secret: impl Into<String>) -> Self {
        Self {
            secret: secret.into(),
            revoked_jtis: HashSet::new(),
            access_ttl_secs: 3600,
            refresh_ttl_secs: 86400 * 7,
        }
    }

    /// Configure access token TTL.
    pub fn set_access_ttl(&mut self, secs: u64) {
        self.access_ttl_secs = secs;
    }

    /// Configure refresh token TTL.
    pub fn set_refresh_ttl(&mut self, secs: u64) {
        self.refresh_ttl_secs = secs;
    }

    /// Issue an access token for a user with given roles.
    #[must_use]
    pub fn issue_access_token(&self, sub: &str, roles: &[&str]) -> Token {
        let mut builder = TokenBuilder::new(sub, &self.secret)
            .ttl(self.access_ttl_secs)
            .kind(TokenKind::Access);
        for role in roles {
            builder = builder.role(*role);
        }
        builder.build()
    }

    /// Issue a refresh token for a user.
    #[must_use]
    pub fn issue_refresh_token(&self, sub: &str) -> Token {
        TokenBuilder::new(sub, &self.secret)
            .ttl(self.refresh_ttl_secs)
            .kind(TokenKind::Refresh)
            .build()
    }

    /// Validate a token (signature + expiry + revocation).
    #[must_use]
    pub fn validate(&self, token: &Token) -> TokenValidationResult {
        if !token.verify_signature(&self.secret) {
            return TokenValidationResult::InvalidSignature;
        }
        if token.claims.is_expired() {
            return TokenValidationResult::Expired;
        }
        if self.revoked_jtis.contains(&token.claims.jti) {
            return TokenValidationResult::Revoked;
        }
        TokenValidationResult::Valid
    }

    /// Revoke a token by its jti.
    pub fn revoke(&mut self, token: &Token) {
        self.revoked_jtis.insert(token.claims.jti.clone());
    }

    /// Refresh an access token using a valid refresh token.
    ///
    /// Returns `None` if the refresh token is invalid or expired.
    #[must_use]
    pub fn refresh(&self, refresh_token: &Token, roles: &[&str]) -> Option<Token> {
        if self.validate(refresh_token) != TokenValidationResult::Valid {
            return None;
        }
        Some(self.issue_access_token(&refresh_token.claims.sub, roles))
    }

    /// Number of revoked token JTIs tracked.
    #[must_use]
    pub fn revoked_count(&self) -> usize {
        self.revoked_jtis.len()
    }
}

/// Result of token validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenValidationResult {
    /// Token is valid
    Valid,
    /// Token signature is invalid
    InvalidSignature,
    /// Token has expired
    Expired,
    /// Token has been revoked
    Revoked,
}

impl TokenValidationResult {
    /// Returns true if the token is valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        *self == Self::Valid
    }
}

fn current_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn compute_signature(claims: &TokenClaims, secret: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    claims.sub.hash(&mut hasher);
    claims.iss.hash(&mut hasher);
    claims.iat.hash(&mut hasher);
    claims.exp.hash(&mut hasher);
    claims.jti.hash(&mut hasher);
    for role in &claims.roles {
        role.hash(&mut hasher);
    }
    secret.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &str = "super-secret-key-for-testing";

    #[test]
    fn test_issue_and_validate_access_token() {
        let mgr = TokenManager::new(SECRET);
        let token = mgr.issue_access_token("alice", &["editor"]);
        let result = mgr.validate(&token);
        assert!(result.is_valid());
    }

    #[test]
    fn test_issue_refresh_token() {
        let mgr = TokenManager::new(SECRET);
        let token = mgr.issue_refresh_token("bob");
        let result = mgr.validate(&token);
        assert!(result.is_valid());
        assert_eq!(token.claims.sub, "bob");
    }

    #[test]
    fn test_token_claims_roles() {
        let mgr = TokenManager::new(SECRET);
        let token = mgr.issue_access_token("carol", &["admin", "editor"]);
        assert!(token.claims.roles.contains(&"admin".to_string()));
        assert!(token.claims.roles.contains(&"editor".to_string()));
    }

    #[test]
    fn test_revoke_token() {
        let mut mgr = TokenManager::new(SECRET);
        let token = mgr.issue_access_token("dave", &[]);
        mgr.revoke(&token);
        assert_eq!(mgr.validate(&token), TokenValidationResult::Revoked);
        assert_eq!(mgr.revoked_count(), 1);
    }

    #[test]
    fn test_invalid_signature_detected() {
        let mgr = TokenManager::new(SECRET);
        let mut token = mgr.issue_access_token("eve", &[]);
        token.signature = 0xDEADBEEF;
        assert_eq!(
            mgr.validate(&token),
            TokenValidationResult::InvalidSignature
        );
    }

    #[test]
    fn test_expired_token_detected() {
        let mut mgr = TokenManager::new(SECRET);
        mgr.set_access_ttl(0); // immediately expired
        let token = mgr.issue_access_token("frank", &[]);
        assert_eq!(mgr.validate(&token), TokenValidationResult::Expired);
        assert!(token.claims.is_expired());
    }

    #[test]
    fn test_refresh_issues_new_access_token() {
        let mgr = TokenManager::new(SECRET);
        let refresh = mgr.issue_refresh_token("grace");
        let new_access = mgr.refresh(&refresh, &["viewer"]);
        assert!(new_access.is_some());
        let new_token = new_access.expect("new_token should be valid");
        assert_eq!(new_token.claims.sub, "grace");
        assert!(mgr.validate(&new_token).is_valid());
    }

    #[test]
    fn test_refresh_with_revoked_refresh_token_returns_none() {
        let mut mgr = TokenManager::new(SECRET);
        let refresh = mgr.issue_refresh_token("heidi");
        mgr.revoke(&refresh);
        let result = mgr.refresh(&refresh, &[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_token_builder_custom_claims() {
        let token = TokenBuilder::new("ivan", SECRET)
            .custom_claim("media_project", "project-42")
            .custom_claim("region", "eu-west")
            .build();
        assert_eq!(token.claims.get_custom("media_project"), Some("project-42"));
        assert_eq!(token.claims.get_custom("region"), Some("eu-west"));
        assert!(token.claims.get_custom("nonexistent").is_none());
    }

    #[test]
    fn test_token_builder_issuer() {
        let token = TokenBuilder::new("judy", SECRET)
            .issuer("custom-issuer")
            .build();
        assert_eq!(token.claims.iss, "custom-issuer");
    }

    #[test]
    fn test_seconds_remaining_nonzero() {
        let mgr = TokenManager::new(SECRET);
        let token = mgr.issue_access_token("kim", &[]);
        let remaining = token.claims.seconds_remaining();
        assert!(remaining > 0);
    }

    #[test]
    fn test_seconds_remaining_expired() {
        let mut mgr = TokenManager::new(SECRET);
        mgr.set_access_ttl(0);
        let token = mgr.issue_access_token("lee", &[]);
        assert_eq!(token.claims.seconds_remaining(), 0);
    }

    #[test]
    fn test_validation_result_is_valid() {
        assert!(TokenValidationResult::Valid.is_valid());
        assert!(!TokenValidationResult::Expired.is_valid());
        assert!(!TokenValidationResult::Revoked.is_valid());
        assert!(!TokenValidationResult::InvalidSignature.is_valid());
    }

    #[test]
    fn test_verify_signature_wrong_secret() {
        let mgr = TokenManager::new(SECRET);
        let token = mgr.issue_access_token("mike", &[]);
        assert!(!token.verify_signature("wrong-secret"));
        assert!(token.verify_signature(SECRET));
    }

    #[test]
    fn test_token_is_valid_method() {
        let mgr = TokenManager::new(SECRET);
        let token = mgr.issue_access_token("nina", &[]);
        assert!(token.is_valid(SECRET));
        assert!(!token.is_valid("bad-secret"));
    }
}
