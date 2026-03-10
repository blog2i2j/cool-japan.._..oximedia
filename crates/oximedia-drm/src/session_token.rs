//! Session token issuance and validation for DRM-protected playback sessions.
//!
//! A [`SessionToken`] is a short-lived bearer credential that carries
//! [`TokenClaim`]s about the viewer and the content being accessed.
//! The [`TokenValidator`] verifies the token's integrity and expiry.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A single typed claim carried inside a [`SessionToken`].
#[derive(Debug, Clone, PartialEq)]
pub enum TokenClaim {
    /// String-valued claim (e.g. subject, content ID).
    Text(String),
    /// Integer-valued claim (e.g. iat, exp).
    Integer(i64),
    /// Boolean-valued claim (e.g. offline_allowed).
    Flag(bool),
    /// A list of string values (e.g. allowed regions).
    List(Vec<String>),
}

impl TokenClaim {
    /// Return the text value if this is a [`TokenClaim::Text`].
    pub fn as_text(&self) -> Option<&str> {
        if let Self::Text(s) = self {
            Some(s)
        } else {
            None
        }
    }

    /// Return the integer value if this is a [`TokenClaim::Integer`].
    pub fn as_integer(&self) -> Option<i64> {
        if let Self::Integer(i) = self {
            Some(*i)
        } else {
            None
        }
    }

    /// Return the flag value if this is a [`TokenClaim::Flag`].
    pub fn as_flag(&self) -> Option<bool> {
        if let Self::Flag(b) = self {
            Some(*b)
        } else {
            None
        }
    }
}

/// A bearer credential representing a single playback session.
#[derive(Debug, Clone)]
pub struct SessionToken {
    /// Opaque token identifier.
    pub token_id: String,
    /// Unix timestamp (seconds) at which this token was issued.
    pub issued_at: i64,
    /// Unix timestamp (seconds) at which this token expires.
    pub expires_at: i64,
    /// Claims carried by the token.
    claims: HashMap<String, TokenClaim>,
    /// Simulated HMAC signature bytes (not cryptographically secure in this
    /// pure-Rust implementation — use a proper signing library in production).
    signature: Vec<u8>,
}

impl SessionToken {
    /// Create a new token valid for `ttl` from `now`.
    pub fn new(token_id: impl Into<String>, ttl: Duration) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let expires_at = now + ttl.as_secs() as i64;
        Self {
            token_id: token_id.into(),
            issued_at: now,
            expires_at,
            claims: HashMap::new(),
            signature: Vec::new(),
        }
    }

    /// Create a token with explicit timestamps (useful for testing).
    pub fn with_timestamps(token_id: impl Into<String>, issued_at: i64, expires_at: i64) -> Self {
        Self {
            token_id: token_id.into(),
            issued_at,
            expires_at,
            claims: HashMap::new(),
            signature: Vec::new(),
        }
    }

    /// Insert a claim.
    pub fn set_claim(&mut self, key: impl Into<String>, value: TokenClaim) {
        self.claims.insert(key.into(), value);
    }

    /// Retrieve a claim by key.
    pub fn claim(&self, key: &str) -> Option<&TokenClaim> {
        self.claims.get(key)
    }

    /// Attach a simulated signature.
    pub fn sign(&mut self, secret: &[u8]) {
        // XOR-fold the secret over the token_id bytes as a toy "signature".
        self.signature = self
            .token_id
            .bytes()
            .zip(secret.iter().cycle())
            .map(|(a, &b)| a ^ b)
            .collect();
    }

    /// Return `true` if a signature has been attached.
    pub fn is_signed(&self) -> bool {
        !self.signature.is_empty()
    }

    /// Return `true` if the token's expiry is in the future relative to
    /// `now_secs` (Unix epoch seconds).
    pub fn is_valid_at(&self, now_secs: i64) -> bool {
        now_secs < self.expires_at && now_secs >= self.issued_at
    }
}

/// Reason why token validation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenValidationError {
    /// The token has passed its `expires_at` timestamp.
    Expired,
    /// The `issued_at` is in the future (clock skew or tampered token).
    NotYetValid,
    /// The token carries no signature.
    NotSigned,
    /// The signature does not match the expected value.
    SignatureMismatch,
    /// A required claim is missing.
    MissingClaim(String),
}

impl std::fmt::Display for TokenValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Expired => write!(f, "token has expired"),
            Self::NotYetValid => write!(f, "token is not yet valid"),
            Self::NotSigned => write!(f, "token is not signed"),
            Self::SignatureMismatch => write!(f, "signature mismatch"),
            Self::MissingClaim(c) => write!(f, "missing required claim: {c}"),
        }
    }
}

/// Validates [`SessionToken`]s.
pub struct TokenValidator {
    secret: Vec<u8>,
    /// Claims that must be present in every token this validator accepts.
    required_claims: Vec<String>,
}

impl TokenValidator {
    /// Create a validator with the given secret and no required claims.
    pub fn new(secret: impl Into<Vec<u8>>) -> Self {
        Self {
            secret: secret.into(),
            required_claims: Vec::new(),
        }
    }

    /// Add a required claim name.
    pub fn require_claim(mut self, claim: impl Into<String>) -> Self {
        self.required_claims.push(claim.into());
        self
    }

    /// Validate `token` against the current wall-clock time.
    ///
    /// Returns `Ok(())` when the token is valid, or a
    /// [`TokenValidationError`] describing the first failure found.
    pub fn validate(&self, token: &SessionToken) -> Result<(), TokenValidationError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        self.validate_at(token, now)
    }

    /// Validate `token` as if the current time is `now_secs` (Unix epoch).
    ///
    /// Useful for deterministic unit tests.
    pub fn validate_at(
        &self,
        token: &SessionToken,
        now_secs: i64,
    ) -> Result<(), TokenValidationError> {
        if !token.is_signed() {
            return Err(TokenValidationError::NotSigned);
        }

        // Re-derive expected signature and compare.
        let expected: Vec<u8> = token
            .token_id
            .bytes()
            .zip(self.secret.iter().cycle())
            .map(|(a, &b)| a ^ b)
            .collect();
        if token.signature != expected {
            return Err(TokenValidationError::SignatureMismatch);
        }

        if now_secs < token.issued_at {
            return Err(TokenValidationError::NotYetValid);
        }
        if now_secs >= token.expires_at {
            return Err(TokenValidationError::Expired);
        }

        for claim in &self.required_claims {
            if token.claim(claim).is_none() {
                return Err(TokenValidationError::MissingClaim(claim.clone()));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &[u8] = b"test-secret-key";

    fn valid_token() -> SessionToken {
        // Use explicit timestamps: issued 100s ago, expires 3600s from now.
        let now: i64 = 1_700_000_000;
        let mut t = SessionToken::with_timestamps("tok-001", now - 100, now + 3600);
        t.sign(SECRET);
        t
    }

    fn validator() -> TokenValidator {
        TokenValidator::new(SECRET)
    }

    #[test]
    fn test_valid_token_passes() {
        let token = valid_token();
        let v = validator();
        let now: i64 = 1_700_000_000;
        assert!(v.validate_at(&token, now).is_ok());
    }

    #[test]
    fn test_expired_token_rejected() {
        let t = SessionToken::with_timestamps("tok-exp", 1_000_000, 2_000_000);
        let mut t = t;
        t.sign(SECRET);
        let v = validator();
        let err = v
            .validate_at(&t, 3_000_000)
            .expect_err("should return error");
        assert_eq!(err, TokenValidationError::Expired);
    }

    #[test]
    fn test_not_yet_valid_rejected() {
        let mut t = SessionToken::with_timestamps("tok-future", 5_000_000, 6_000_000);
        t.sign(SECRET);
        let v = validator();
        let err = v
            .validate_at(&t, 1_000_000)
            .expect_err("should return error");
        assert_eq!(err, TokenValidationError::NotYetValid);
    }

    #[test]
    fn test_unsigned_token_rejected() {
        let t = SessionToken::with_timestamps("tok-nosig", 1_000_000, 9_000_000);
        let v = validator();
        let err = v
            .validate_at(&t, 5_000_000)
            .expect_err("should return error");
        assert_eq!(err, TokenValidationError::NotSigned);
    }

    #[test]
    fn test_wrong_secret_rejected() {
        let mut t = SessionToken::with_timestamps("tok-wrong", 1_000_000, 9_000_000);
        t.sign(b"wrong-secret");
        let v = validator(); // uses SECRET
        let err = v
            .validate_at(&t, 5_000_000)
            .expect_err("should return error");
        assert_eq!(err, TokenValidationError::SignatureMismatch);
    }

    #[test]
    fn test_missing_required_claim_rejected() {
        let token = valid_token();
        let v = validator().require_claim("sub");
        let now: i64 = 1_700_000_000;
        let err = v.validate_at(&token, now).expect_err("should return error");
        assert_eq!(err, TokenValidationError::MissingClaim("sub".to_string()));
    }

    #[test]
    fn test_required_claim_present_passes() {
        let mut token = valid_token();
        token.set_claim("sub", TokenClaim::Text("user-42".to_string()));
        let v = validator().require_claim("sub");
        let now: i64 = 1_700_000_000;
        assert!(v.validate_at(&token, now).is_ok());
    }

    #[test]
    fn test_set_and_retrieve_text_claim() {
        let mut t = valid_token();
        t.set_claim("content_id", TokenClaim::Text("movie-999".to_string()));
        assert_eq!(
            t.claim("content_id").expect("claim should exist").as_text(),
            Some("movie-999")
        );
    }

    #[test]
    fn test_set_and_retrieve_integer_claim() {
        let mut t = valid_token();
        t.set_claim("quality", TokenClaim::Integer(1080));
        assert_eq!(
            t.claim("quality").expect("claim should exist").as_integer(),
            Some(1080)
        );
    }

    #[test]
    fn test_set_and_retrieve_flag_claim() {
        let mut t = valid_token();
        t.set_claim("offline", TokenClaim::Flag(true));
        assert_eq!(
            t.claim("offline").expect("claim should exist").as_flag(),
            Some(true)
        );
    }

    #[test]
    fn test_claim_as_text_wrong_variant_returns_none() {
        let c = TokenClaim::Integer(42);
        assert!(c.as_text().is_none());
    }

    #[test]
    fn test_claim_as_integer_wrong_variant_returns_none() {
        let c = TokenClaim::Text("hello".to_string());
        assert!(c.as_integer().is_none());
    }

    #[test]
    fn test_is_valid_at_boundary() {
        let t = SessionToken::with_timestamps("boundary", 1000, 2000);
        assert!(t.is_valid_at(1000));
        assert!(t.is_valid_at(1999));
        assert!(!t.is_valid_at(2000));
        assert!(!t.is_valid_at(999));
    }

    #[test]
    fn test_is_signed_false_before_sign() {
        let t = SessionToken::new("unsign", Duration::from_secs(3600));
        assert!(!t.is_signed());
    }

    #[test]
    fn test_is_signed_true_after_sign() {
        let mut t = SessionToken::new("signed", Duration::from_secs(3600));
        t.sign(SECRET);
        assert!(t.is_signed());
    }

    #[test]
    fn test_token_claim_list_variant() {
        let c = TokenClaim::List(vec!["US".to_string(), "GB".to_string()]);
        assert!(c.as_text().is_none());
        assert!(c.as_flag().is_none());
    }

    #[test]
    fn test_validation_error_display() {
        let e = TokenValidationError::MissingClaim("sub".to_string());
        assert!(e.to_string().contains("sub"));
        assert!(TokenValidationError::Expired
            .to_string()
            .contains("expired"));
    }
}
