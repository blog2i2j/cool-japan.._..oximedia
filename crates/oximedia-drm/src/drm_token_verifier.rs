//! DRM token verification: structured claims, HMAC-SHA256 signature validation,
//! and a multi-key verifier for license-gating workflows.
//!
//! License servers that sit behind a CDN edge (Akamai EdgeAuth, Fastly token
//! auth, etc.) rely on short-lived signed tokens to prevent unauthorized
//! license requests.  This module implements a **DRM-specific token format**
//! that carries the claims most relevant to DRM workflows:
//!
//! - Content key identifiers (`kid`)
//! - DRM system (widevine / playready / fairplay / clearkey)
//! - Validity window (`nbf` / `exp`)
//! - Subscriber / device identifier
//! - Optional geo-restriction (`allow_regions`)
//!
//! ## Token wire format
//!
//! A [`DrmToken`] is serialised as `BASE64URL(header).BASE64URL(payload).SIG`
//! where `SIG` is the lower-case hex encoding of an HMAC-SHA256 over the first
//! two parts joined by a period, using a server-side secret.
//!
//! This is intentionally **not** JWT — it avoids the `alg` confusion
//! vulnerability and uses a fixed algorithm family (HMAC-SHA256).

use crate::{DrmError, DrmSystem, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Token header and claims
// ---------------------------------------------------------------------------

/// Fixed token header (always HMAC-SHA256).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenHeader {
    /// Algorithm identifier — always `"HS256"` for this implementation.
    pub alg: String,
    /// Token type — always `"DRM"`.
    pub typ: String,
    /// Key ID that identifies which signing secret was used.
    pub kid: String,
}

impl TokenHeader {
    /// Create a new header for the given signing key ID.
    pub fn new(signing_key_id: impl Into<String>) -> Self {
        Self {
            alg: "HS256".to_string(),
            typ: "DRM".to_string(),
            kid: signing_key_id.into(),
        }
    }
}

/// DRM-specific token claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrmClaims {
    /// Subject — subscriber or device identifier.
    pub sub: String,
    /// Issued-at (Unix timestamp, seconds).
    pub iat: u64,
    /// Not-before (Unix timestamp, seconds).
    pub nbf: u64,
    /// Expiry (Unix timestamp, seconds).
    pub exp: u64,
    /// Content key IDs (hex-encoded) that this token grants access to.
    pub key_ids: Vec<String>,
    /// DRM systems allowed for this token.
    pub drm_systems: Vec<String>,
    /// Allowed ISO 3166-1 alpha-2 region codes.  Empty = no restriction.
    #[serde(default)]
    pub allow_regions: Vec<String>,
    /// Optional nonce to prevent replay.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nonce: Option<String>,
    /// Arbitrary extension claims.
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl DrmClaims {
    /// Create minimal claims for `sub`, valid from `nbf` until `exp`.
    pub fn new(sub: impl Into<String>, nbf: u64, exp: u64) -> Self {
        Self {
            sub: sub.into(),
            iat: nbf,
            nbf,
            exp,
            key_ids: Vec::new(),
            drm_systems: Vec::new(),
            allow_regions: Vec::new(),
            nonce: None,
            extra: HashMap::new(),
        }
    }

    /// Builder: add an allowed content key ID (hex-encoded).
    pub fn with_key_id(mut self, kid: impl Into<String>) -> Self {
        self.key_ids.push(kid.into());
        self
    }

    /// Builder: allow a DRM system.
    pub fn with_drm_system(mut self, drm: DrmSystem) -> Self {
        self.drm_systems.push(drm.to_string().to_lowercase());
        self
    }

    /// Builder: restrict to a set of regions.
    pub fn with_regions(mut self, regions: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.allow_regions = regions.into_iter().map(Into::into).collect();
        self
    }

    /// Builder: set a replay-prevention nonce.
    pub fn with_nonce(mut self, nonce: impl Into<String>) -> Self {
        self.nonce = Some(nonce.into());
        self
    }

    /// Return `true` if these claims are valid at the given Unix timestamp.
    pub fn is_valid_at(&self, now: u64) -> bool {
        now >= self.nbf && now < self.exp
    }

    /// Return `true` if `key_id` (hex-encoded) is in the claims.
    pub fn allows_key(&self, key_id_hex: &str) -> bool {
        self.key_ids.iter().any(|k| k.eq_ignore_ascii_case(key_id_hex))
    }

    /// Return `true` if `drm` is permitted.
    pub fn allows_drm(&self, drm: DrmSystem) -> bool {
        if self.drm_systems.is_empty() {
            return true; // empty = no restriction
        }
        let name = drm.to_string().to_lowercase();
        self.drm_systems.iter().any(|d| d == &name)
    }

    /// Return `true` if `region` (ISO 3166-1 alpha-2, case-insensitive) is
    /// permitted.  An empty `allow_regions` list means no geo-restriction.
    pub fn allows_region(&self, region: &str) -> bool {
        if self.allow_regions.is_empty() {
            return true;
        }
        self.allow_regions
            .iter()
            .any(|r| r.eq_ignore_ascii_case(region))
    }
}

// ---------------------------------------------------------------------------
// Signed DRM token
// ---------------------------------------------------------------------------

/// A signed DRM token comprising a header, claims, and HMAC-SHA256 signature.
#[derive(Debug, Clone)]
pub struct DrmToken {
    /// Token header (algorithm + key ID).
    pub header: TokenHeader,
    /// DRM-specific claims payload.
    pub claims: DrmClaims,
    /// Hex-encoded HMAC-SHA256 signature.
    signature: String,
}

impl DrmToken {
    /// Serialise the token to the wire format:
    /// `BASE64URL(header).BASE64URL(payload).hex(sig)`
    pub fn to_string_repr(&self) -> String {
        let h = URL_SAFE_NO_PAD.encode(
            serde_json::to_string(&self.header)
                .unwrap_or_default()
                .as_bytes(),
        );
        let p = URL_SAFE_NO_PAD.encode(
            serde_json::to_string(&self.claims)
                .unwrap_or_default()
                .as_bytes(),
        );
        format!("{h}.{p}.{}", self.signature)
    }

    /// Return `true` if the token is valid at `now` (no signature check).
    pub fn is_temporally_valid(&self, now: u64) -> bool {
        self.claims.is_valid_at(now)
    }

    /// Access the header.
    pub fn header(&self) -> &TokenHeader {
        &self.header
    }

    /// Access the claims.
    pub fn claims(&self) -> &DrmClaims {
        &self.claims
    }
}

impl fmt::Display for DrmToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_repr())
    }
}

// ---------------------------------------------------------------------------
// Token signer
// ---------------------------------------------------------------------------

/// Signs DRM tokens with HMAC-SHA256.
#[derive(Debug, Clone)]
pub struct DrmTokenSigner {
    /// Key ID used in the token header.
    key_id: String,
    /// Raw signing secret bytes.
    secret: Vec<u8>,
}

impl DrmTokenSigner {
    /// Create a new signer with the given key ID and secret.
    pub fn new(key_id: impl Into<String>, secret: Vec<u8>) -> Self {
        Self {
            key_id: key_id.into(),
            secret,
        }
    }

    /// Sign `claims` and return a [`DrmToken`].
    pub fn sign(&self, claims: DrmClaims) -> Result<DrmToken> {
        let header = TokenHeader::new(&self.key_id);
        let h_enc = URL_SAFE_NO_PAD.encode(
            serde_json::to_string(&header)
                .map_err(DrmError::JsonError)?
                .as_bytes(),
        );
        let p_enc = URL_SAFE_NO_PAD.encode(
            serde_json::to_string(&claims)
                .map_err(DrmError::JsonError)?
                .as_bytes(),
        );
        let signing_input = format!("{h_enc}.{p_enc}");
        let sig = hmac_sha256(&self.secret, signing_input.as_bytes());
        let sig_hex: String = sig.iter().map(|b| format!("{b:02x}")).collect();

        Ok(DrmToken {
            header,
            claims,
            signature: sig_hex,
        })
    }
}

// ---------------------------------------------------------------------------
// Token verifier
// ---------------------------------------------------------------------------

/// Verification result with detailed failure reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyOutcome {
    /// Token is valid and all policy checks passed.
    Valid,
    /// HMAC signature does not match.
    InvalidSignature,
    /// Token is expired (current time ≥ exp).
    Expired,
    /// Token is not yet valid (current time < nbf).
    NotYetValid,
    /// Signing key ID is not registered.
    UnknownKeyId(String),
    /// DRM system is not permitted by the token.
    DrmNotAllowed,
    /// Region is not permitted by the token.
    RegionNotAllowed,
    /// Token wire format is malformed.
    MalformedToken(String),
}

impl fmt::Display for VerifyOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Valid => write!(f, "valid"),
            Self::InvalidSignature => write!(f, "invalid signature"),
            Self::Expired => write!(f, "token expired"),
            Self::NotYetValid => write!(f, "token not yet valid"),
            Self::UnknownKeyId(kid) => write!(f, "unknown signing key '{kid}'"),
            Self::DrmNotAllowed => write!(f, "DRM system not permitted"),
            Self::RegionNotAllowed => write!(f, "region not permitted"),
            Self::MalformedToken(msg) => write!(f, "malformed token: {msg}"),
        }
    }
}

/// Multi-key DRM token verifier.
///
/// Maintains a registry of named signing secrets and verifies tokens against
/// them, enforcing expiry, geo-restriction, and DRM system policies.
#[derive(Debug, Default)]
pub struct DrmTokenVerifier {
    /// Signing key registry: key_id → raw secret bytes.
    secrets: HashMap<String, Vec<u8>>,
    /// Revoked nonces (prevents replay attacks).
    revoked_nonces: HashSet<String>,
}

impl DrmTokenVerifier {
    /// Create an empty verifier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a signing secret under `key_id`.
    pub fn add_secret(&mut self, key_id: impl Into<String>, secret: Vec<u8>) {
        self.secrets.insert(key_id.into(), secret);
    }

    /// Remove a signing secret.
    pub fn remove_secret(&mut self, key_id: &str) -> bool {
        self.secrets.remove(key_id).is_some()
    }

    /// Revoke a nonce so it cannot be reused.
    pub fn revoke_nonce(&mut self, nonce: impl Into<String>) {
        self.revoked_nonces.insert(nonce.into());
    }

    /// Parse and verify a token wire string.
    ///
    /// Returns [`VerifyOutcome::Valid`] and the parsed token on success, or
    /// the specific failure reason otherwise.
    pub fn verify(
        &self,
        token_str: &str,
        now: u64,
        drm: Option<DrmSystem>,
        region: Option<&str>,
    ) -> (VerifyOutcome, Option<DrmToken>) {
        // Parse wire format
        let parts: Vec<&str> = token_str.splitn(3, '.').collect();
        if parts.len() != 3 {
            return (
                VerifyOutcome::MalformedToken("expected 3 dot-separated parts".to_string()),
                None,
            );
        }
        let (h_enc, p_enc, sig_hex) = (parts[0], parts[1], parts[2]);

        // Decode header
        let header: TokenHeader = match URL_SAFE_NO_PAD
            .decode(h_enc)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
        {
            Some(h) => h,
            None => {
                return (
                    VerifyOutcome::MalformedToken("cannot decode header".to_string()),
                    None,
                )
            }
        };

        // Look up signing secret
        let secret = match self.secrets.get(&header.kid) {
            Some(s) => s,
            None => return (VerifyOutcome::UnknownKeyId(header.kid.clone()), None),
        };

        // Verify signature
        let signing_input = format!("{h_enc}.{p_enc}");
        let expected_sig = hmac_sha256(secret, signing_input.as_bytes());
        let expected_hex: String = expected_sig.iter().map(|b| format!("{b:02x}")).collect();
        // Constant-time comparison
        if !constant_time_eq(sig_hex.as_bytes(), expected_hex.as_bytes()) {
            return (VerifyOutcome::InvalidSignature, None);
        }

        // Decode claims
        let claims: DrmClaims = match URL_SAFE_NO_PAD
            .decode(p_enc)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
        {
            Some(c) => c,
            None => {
                return (
                    VerifyOutcome::MalformedToken("cannot decode claims".to_string()),
                    None,
                )
            }
        };

        // Temporal validity
        if now >= claims.exp {
            return (VerifyOutcome::Expired, None);
        }
        if now < claims.nbf {
            return (VerifyOutcome::NotYetValid, None);
        }

        // Nonce replay check
        if let Some(ref nonce) = claims.nonce {
            if self.revoked_nonces.contains(nonce.as_str()) {
                return (VerifyOutcome::InvalidSignature, None);
            }
        }

        // DRM system check
        if let Some(d) = drm {
            if !claims.allows_drm(d) {
                return (VerifyOutcome::DrmNotAllowed, None);
            }
        }

        // Region check
        if let Some(r) = region {
            if !claims.allows_region(r) {
                return (VerifyOutcome::RegionNotAllowed, None);
            }
        }

        let token = DrmToken {
            header,
            claims,
            signature: sig_hex.to_string(),
        };
        (VerifyOutcome::Valid, Some(token))
    }

    /// Number of registered secrets.
    pub fn secret_count(&self) -> usize {
        self.secrets.len()
    }
}

// ---------------------------------------------------------------------------
// HMAC-SHA256 (pure Rust, reused from key_rotation.rs pattern)
// ---------------------------------------------------------------------------

/// SHA-256 digest (FIPS 180-4).
fn sha256(msg: &[u8]) -> [u8; 32] {
    #[allow(clippy::unreadable_literal)]
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
    #[allow(clippy::unreadable_literal)]
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    let bit_len = (msg.len() as u64).wrapping_mul(8);
    let mut padded = msg.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for block in padded.chunks(64) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g; g = f; f = e; e = d.wrapping_add(t1);
            d = c; c = b; b = a; a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a); h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c); h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e); h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g); h[7] = h[7].wrapping_add(hh);
    }
    let mut out = [0u8; 32];
    for (i, &word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK {
        let h = sha256(key);
        k[..32].copy_from_slice(&h);
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let ipad: Vec<u8> = k.iter().map(|b| b ^ 0x36).collect();
    let opad: Vec<u8> = k.iter().map(|b| b ^ 0x5C).collect();
    let mut inner = ipad;
    inner.extend_from_slice(data);
    let ih = sha256(&inner);
    let mut outer = opad;
    outer.extend_from_slice(&ih);
    sha256(&outer)
}

/// Constant-time byte-slice equality.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signer() -> DrmTokenSigner {
        DrmTokenSigner::new("key-1", b"super-secret-signing-key".to_vec())
    }

    fn make_verifier() -> DrmTokenVerifier {
        let mut v = DrmTokenVerifier::new();
        v.add_secret("key-1", b"super-secret-signing-key".to_vec());
        v
    }

    fn basic_claims(nbf: u64, exp: u64) -> DrmClaims {
        DrmClaims::new("user-123", nbf, exp)
            .with_key_id("aabbccdd")
            .with_drm_system(DrmSystem::Widevine)
    }

    // ── DrmClaims ────────────────────────────────────────────────────────────

    #[test]
    fn test_claims_temporal_validity() {
        let claims = basic_claims(1000, 2000);
        assert!(!claims.is_valid_at(999));
        assert!(claims.is_valid_at(1000));
        assert!(claims.is_valid_at(1999));
        assert!(!claims.is_valid_at(2000));
    }

    #[test]
    fn test_claims_allows_key() {
        let claims = basic_claims(0, 9999).with_key_id("deadbeef");
        assert!(claims.allows_key("aabbccdd"));
        assert!(claims.allows_key("AABBCCDD")); // case-insensitive
        assert!(claims.allows_key("deadbeef"));
        assert!(!claims.allows_key("00000000"));
    }

    #[test]
    fn test_claims_allows_drm() {
        let claims = basic_claims(0, 9999); // widevine only
        assert!(claims.allows_drm(DrmSystem::Widevine));
        assert!(!claims.allows_drm(DrmSystem::FairPlay));
    }

    #[test]
    fn test_claims_empty_drm_allows_all() {
        let claims = DrmClaims::new("u", 0, 9999);
        assert!(claims.allows_drm(DrmSystem::Widevine));
        assert!(claims.allows_drm(DrmSystem::FairPlay));
    }

    #[test]
    fn test_claims_region_restriction() {
        let claims = basic_claims(0, 9999).with_regions(["US", "GB"]);
        assert!(claims.allows_region("US"));
        assert!(claims.allows_region("us")); // case-insensitive
        assert!(claims.allows_region("GB"));
        assert!(!claims.allows_region("DE"));
    }

    #[test]
    fn test_claims_empty_regions_allows_all() {
        let claims = basic_claims(0, 9999);
        assert!(claims.allows_region("JP"));
        assert!(claims.allows_region("XX"));
    }

    // ── Sign + verify round-trip ─────────────────────────────────────────────

    #[test]
    fn test_sign_and_verify_valid() {
        let signer = make_signer();
        let verifier = make_verifier();
        let claims = basic_claims(1000, 9000);
        let token = signer.sign(claims).expect("sign");
        let token_str = token.to_string_repr();
        let (outcome, parsed) = verifier.verify(&token_str, 5000, None, None);
        assert_eq!(outcome, VerifyOutcome::Valid, "expected Valid, got {outcome}");
        assert!(parsed.is_some());
    }

    #[test]
    fn test_verify_expired_token() {
        let signer = make_signer();
        let verifier = make_verifier();
        let token = signer.sign(basic_claims(1000, 2000)).expect("sign");
        let (outcome, _) = verifier.verify(&token.to_string_repr(), 3000, None, None);
        assert_eq!(outcome, VerifyOutcome::Expired);
    }

    #[test]
    fn test_verify_not_yet_valid() {
        let signer = make_signer();
        let verifier = make_verifier();
        let token = signer.sign(basic_claims(5000, 9000)).expect("sign");
        let (outcome, _) = verifier.verify(&token.to_string_repr(), 4999, None, None);
        assert_eq!(outcome, VerifyOutcome::NotYetValid);
    }

    #[test]
    fn test_verify_tampered_signature() {
        let signer = make_signer();
        let verifier = make_verifier();
        let token = signer.sign(basic_claims(0, 9999)).expect("sign");
        let mut token_str = token.to_string_repr();
        // Flip last char of signature
        let last = token_str.pop().unwrap_or('0');
        token_str.push(if last == 'a' { 'b' } else { 'a' });
        let (outcome, _) = verifier.verify(&token_str, 5000, None, None);
        assert_eq!(outcome, VerifyOutcome::InvalidSignature);
    }

    #[test]
    fn test_verify_unknown_key_id() {
        let signer = DrmTokenSigner::new("unknown-key", b"some-secret".to_vec());
        let verifier = make_verifier(); // does not know "unknown-key"
        let token = signer.sign(basic_claims(0, 9999)).expect("sign");
        let (outcome, _) = verifier.verify(&token.to_string_repr(), 5000, None, None);
        assert!(matches!(outcome, VerifyOutcome::UnknownKeyId(_)));
    }

    #[test]
    fn test_verify_drm_not_allowed() {
        let signer = make_signer();
        let verifier = make_verifier();
        let claims = DrmClaims::new("u", 0, 9999).with_drm_system(DrmSystem::Widevine);
        let token = signer.sign(claims).expect("sign");
        let (outcome, _) =
            verifier.verify(&token.to_string_repr(), 5000, Some(DrmSystem::FairPlay), None);
        assert_eq!(outcome, VerifyOutcome::DrmNotAllowed);
    }

    #[test]
    fn test_verify_region_not_allowed() {
        let signer = make_signer();
        let verifier = make_verifier();
        let claims = basic_claims(0, 9999).with_regions(["US"]);
        let token = signer.sign(claims).expect("sign");
        let (outcome, _) = verifier.verify(&token.to_string_repr(), 5000, None, Some("DE"));
        assert_eq!(outcome, VerifyOutcome::RegionNotAllowed);
    }

    #[test]
    fn test_verify_malformed_token() {
        let verifier = make_verifier();
        let (outcome, _) = verifier.verify("notadotted", 0, None, None);
        assert!(matches!(outcome, VerifyOutcome::MalformedToken(_)));
    }

    #[test]
    fn test_revoked_nonce_rejected() {
        let signer = make_signer();
        let mut verifier = make_verifier();
        let claims = basic_claims(0, 9999).with_nonce("nonce-abc");
        let token = signer.sign(claims).expect("sign");
        verifier.revoke_nonce("nonce-abc");
        let (outcome, _) = verifier.verify(&token.to_string_repr(), 5000, None, None);
        assert_eq!(outcome, VerifyOutcome::InvalidSignature);
    }

    #[test]
    fn test_verify_outcome_display() {
        assert_eq!(VerifyOutcome::Valid.to_string(), "valid");
        assert_eq!(VerifyOutcome::Expired.to_string(), "token expired");
        assert_eq!(
            VerifyOutcome::UnknownKeyId("k".to_string()).to_string(),
            "unknown signing key 'k'"
        );
    }

    #[test]
    fn test_verifier_add_remove_secret() {
        let mut v = DrmTokenVerifier::new();
        v.add_secret("k1", vec![0u8; 32]);
        assert_eq!(v.secret_count(), 1);
        v.remove_secret("k1");
        assert_eq!(v.secret_count(), 0);
    }
}
