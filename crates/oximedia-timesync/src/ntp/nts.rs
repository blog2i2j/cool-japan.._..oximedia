//! NTS (Network Time Security) authentication hooks per RFC 8915.
//!
//! NTS extends NTPv4 with cryptographic authentication using AEAD
//! (Authenticated Encryption with Associated Data) key material obtained
//! from a TLS-based NTS-KE (Key Exchange) server.
//!
//! This module defines:
//! - AEAD algorithm identifiers ([`NtsAeadAlgorithm`]).
//! - Key-exchange result types ([`NtsKeResult`], [`NtsKeyMaterial`]).
//! - A trait-based authentication hook ([`NtsAuthHook`]) for plugging in
//!   backend AEAD implementations.
//! - A stateful session object ([`NtsSession`]) tracking cookies and key
//!   material between NTP exchanges.
//! - A no-op hook ([`NullNtsAuthHook`]) for testing and disabled-auth mode.

use std::net::SocketAddr;
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// Algorithm identifiers (RFC 8915 §5.1, IANA NTS AEAD registry)
// ---------------------------------------------------------------------------

/// AEAD algorithm identifiers assigned by IANA for NTS (RFC 8915 §5.1).
///
/// All three variants are AES-SIV-CMAC as specified in RFC 5297.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum NtsAeadAlgorithm {
    /// AES-SIV-CMAC-256 (AEAD identifier 15).
    AesSivCmac256 = 15,
    /// AES-SIV-CMAC-384 (AEAD identifier 16).
    AesSivCmac384 = 16,
    /// AES-SIV-CMAC-512 (AEAD identifier 17).
    AesSivCmac512 = 17,
}

impl NtsAeadAlgorithm {
    /// Returns the key length in bytes required by this algorithm.
    ///
    /// * AesSivCmac256 → 32 bytes (two 128-bit keys).
    /// * AesSivCmac384 → 48 bytes.
    /// * AesSivCmac512 → 64 bytes.
    #[must_use]
    pub fn key_length_bytes(&self) -> usize {
        match self {
            Self::AesSivCmac256 => 32,
            Self::AesSivCmac384 => 48,
            Self::AesSivCmac512 => 64,
        }
    }

    /// Returns the IANA numeric value as a `u16`.
    #[must_use]
    pub fn as_u16(self) -> u16 {
        self as u16
    }

    /// Attempts to construct an [`NtsAeadAlgorithm`] from a raw IANA value.
    pub fn from_u16(value: u16) -> Result<Self, NtsError> {
        match value {
            15 => Ok(Self::AesSivCmac256),
            16 => Ok(Self::AesSivCmac384),
            17 => Ok(Self::AesSivCmac512),
            _ => Err(NtsError::InvalidExtField(format!(
                "unknown AEAD algorithm id {value}"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Cookie and key types
// ---------------------------------------------------------------------------

/// An opaque NTS cookie, obtained from the NTS-KE server and included in
/// NTP requests as an NTS Cookie extension field (RFC 8915 §5.7).
///
/// A fresh cookie is returned by the server with each NTP response, allowing
/// the client to maintain a pool and avoid key-re-exchange.
#[derive(Debug, Clone)]
pub struct NtsCookie(pub Vec<u8>);

impl NtsCookie {
    /// Creates a new cookie wrapping the given byte vector.
    #[must_use]
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Returns a reference to the raw cookie bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Returns the number of bytes in this cookie.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the cookie byte vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Key material derived from the NTS-KE TLS session (RFC 8915 §4).
///
/// Two separate keys are exported from the TLS master secret:
/// - `c2s_key`: used to authenticate requests (client → server).
/// - `s2c_key`: used to authenticate responses (server → client).
#[derive(Debug, Clone)]
pub struct NtsKeyMaterial {
    /// Client-to-server AEAD key.
    pub c2s_key: Vec<u8>,
    /// Server-to-client AEAD key.
    pub s2c_key: Vec<u8>,
    /// AEAD algorithm these keys are intended for.
    pub algorithm: NtsAeadAlgorithm,
}

impl NtsKeyMaterial {
    /// Creates new key material, validating key lengths against the algorithm.
    pub fn new(
        c2s_key: Vec<u8>,
        s2c_key: Vec<u8>,
        algorithm: NtsAeadAlgorithm,
    ) -> Result<Self, NtsError> {
        let expected = algorithm.key_length_bytes();
        if c2s_key.len() != expected {
            return Err(NtsError::AuthFailed(format!(
                "c2s_key length {} does not match algorithm requirement {}",
                c2s_key.len(),
                expected
            )));
        }
        if s2c_key.len() != expected {
            return Err(NtsError::AuthFailed(format!(
                "s2c_key length {} does not match algorithm requirement {}",
                s2c_key.len(),
                expected
            )));
        }
        Ok(Self {
            c2s_key,
            s2c_key,
            algorithm,
        })
    }
}

// ---------------------------------------------------------------------------
// NTS-KE result
// ---------------------------------------------------------------------------

/// The result of a successful NTS-KE (Key Exchange) handshake (RFC 8915 §4).
///
/// Contains the NTP server to use for subsequent time queries, the derived
/// key material, and a set of pre-provisioned cookies.
#[derive(Debug, Clone)]
pub struct NtsKeResult {
    /// Address of the NTP server to use for time queries (may differ from
    /// the NTS-KE server).
    pub server_addr: SocketAddr,
    /// Key material exported from the TLS session.
    pub key_material: NtsKeyMaterial,
    /// Pre-provisioned cookies (RFC 8915 §4.1.6).
    pub cookies: Vec<NtsCookie>,
    /// Next-protocol identifier (0 = NTPv4, RFC 8915 §4.1.2).
    pub next_protocol: u16,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to NTS operations.
#[derive(Debug, thiserror::Error)]
pub enum NtsError {
    /// AEAD authentication failed (e.g. tag mismatch).
    #[error("NTS authentication failed: {0}")]
    AuthFailed(String),

    /// No cookies remain in the session pool; a new KE handshake is required.
    #[error("No NTS cookies available")]
    NoCookies,

    /// An extension field is malformed or unrecognised.
    #[error("Invalid NTS extension field: {0}")]
    InvalidExtField(String),

    /// The key material has expired or been revoked.
    #[error("NTS key material expired")]
    KeyExpired,

    /// The requested algorithm is not supported by this implementation.
    #[error("Unsupported AEAD algorithm: {0:?}")]
    UnsupportedAlgorithm(NtsAeadAlgorithm),
}

// ---------------------------------------------------------------------------
// Authentication hook trait
// ---------------------------------------------------------------------------

/// Hook trait enabling pluggable AEAD backends for NTS packet authentication.
///
/// Implementors authenticate outgoing NTP requests and verify incoming
/// responses by constructing / inspecting the NTS Extension Fields defined
/// in RFC 8915 §5.
///
/// # Extension field layout (simplified)
/// ```text
/// ┌──────────────────────┐
/// │  Unique Identifier   │  (EF type 0x0104, 32 bytes)
/// │  NTS Cookie          │  (EF type 0x0204, opaque)
/// │  NTS Authenticator   │  (EF type 0x0404, nonce || ciphertext)
/// └──────────────────────┘
/// ```
pub trait NtsAuthHook: Send + Sync {
    /// Constructs the NTS Extension Fields to append to an outgoing NTP
    /// request packet.
    ///
    /// # Arguments
    /// * `ntp_header` — the 48-byte NTP header that will be sent (associated
    ///   data for AEAD).
    /// * `key`        — C2S key material.
    /// * `cookie`     — a single cookie to include in the request.
    /// * `unique_id`  — 32-byte nonce that uniquely identifies this exchange.
    ///
    /// # Returns
    /// Raw bytes of the NTS Extension Fields to append, or an [`NtsError`] on
    /// failure.
    fn authenticate_request(
        &self,
        ntp_header: &[u8],
        key: &NtsKeyMaterial,
        cookie: &NtsCookie,
        unique_id: &[u8],
    ) -> Result<Vec<u8>, NtsError>;

    /// Verifies the NTS Extension Fields in an incoming NTP response.
    ///
    /// # Arguments
    /// * `ntp_header`   — the 48-byte NTP response header (associated data).
    /// * `nts_ef_data`  — the NTS Extension Fields from the response.
    /// * `key`          — S2C key material.
    /// * `unique_id`    — the 32-byte nonce sent in the corresponding request.
    ///
    /// # Returns
    /// `Ok(())` if authentication succeeds, or an [`NtsError`] otherwise.
    fn verify_response(
        &self,
        ntp_header: &[u8],
        nts_ef_data: &[u8],
        key: &NtsKeyMaterial,
        unique_id: &[u8],
    ) -> Result<(), NtsError>;
}

// ---------------------------------------------------------------------------
// Null (no-op) implementation for testing / disabled-auth mode
// ---------------------------------------------------------------------------

/// A no-op [`NtsAuthHook`] that never performs real cryptography.
///
/// * `authenticate_request` returns an empty byte vector.
/// * `verify_response` always succeeds.
///
/// Suitable for unit testing or running in environments where NTS is not
/// required.
pub struct NullNtsAuthHook;

impl NtsAuthHook for NullNtsAuthHook {
    fn authenticate_request(
        &self,
        _ntp_header: &[u8],
        _key: &NtsKeyMaterial,
        _cookie: &NtsCookie,
        _unique_id: &[u8],
    ) -> Result<Vec<u8>, NtsError> {
        Ok(Vec::new())
    }

    fn verify_response(
        &self,
        _ntp_header: &[u8],
        _nts_ef_data: &[u8],
        _key: &NtsKeyMaterial,
        _unique_id: &[u8],
    ) -> Result<(), NtsError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// NTS client configuration
// ---------------------------------------------------------------------------

/// Configuration for an NTS-aware NTP client.
#[derive(Debug, Clone)]
pub struct NtsClientConfig {
    /// Address of the NTS-KE server (TCP/TLS, default port 4460 per RFC 8915
    /// §4).
    pub ke_server: SocketAddr,
    /// Preferred AEAD algorithm for key derivation.
    pub algorithm: NtsAeadAlgorithm,
    /// Maximum number of cookies to maintain in the pool.
    pub max_cookies: usize,
}

impl Default for NtsClientConfig {
    fn default() -> Self {
        Self {
            // Use a placeholder; callers must set the real server address.
            ke_server: SocketAddr::from(([127, 0, 0, 1], 4460)),
            algorithm: NtsAeadAlgorithm::AesSivCmac256,
            max_cookies: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// NTS session state machine
// ---------------------------------------------------------------------------

/// Stateful NTS session tracking key material and cookie pool.
///
/// After a successful NTS-KE handshake (external to this module), call
/// `set_ke_result` to populate the session.  Use `consume_cookie` to
/// obtain a fresh cookie for each NTP request.
///
/// When the cookie pool is exhausted a new KE handshake must be performed.
#[derive(Debug)]
pub struct NtsSession {
    config: NtsClientConfig,
    ke_result: Option<NtsKeResult>,
    /// Index into `ke_result.cookies` pointing to the next cookie to use.
    cookie_index: usize,
}

impl NtsSession {
    /// Creates a new session with the given configuration.
    #[must_use]
    pub fn new(config: NtsClientConfig) -> Self {
        Self {
            config,
            ke_result: None,
            cookie_index: 0,
        }
    }

    /// Returns `true` if a KE result has been set and at least one cookie
    /// remains in the pool.
    #[must_use]
    pub fn is_authenticated(&self) -> bool {
        self.ke_result
            .as_ref()
            .map_or(false, |r| self.cookie_index < r.cookies.len())
    }

    /// Consumes and returns the next available cookie, advancing the internal
    /// index.  Returns `None` when no cookies remain.
    pub fn consume_cookie(&mut self) -> Option<NtsCookie> {
        let result = self.ke_result.as_ref()?;
        if self.cookie_index >= result.cookies.len() {
            return None;
        }
        let cookie = result.cookies[self.cookie_index].clone();
        self.cookie_index += 1;
        Some(cookie)
    }

    /// Returns the number of cookies remaining in the pool.
    #[must_use]
    pub fn remaining_cookies(&self) -> usize {
        self.ke_result
            .as_ref()
            .map_or(0, |r| r.cookies.len().saturating_sub(self.cookie_index))
    }

    /// Stores a new KE result, resetting the cookie index.
    pub fn set_ke_result(&mut self, result: NtsKeResult) {
        self.ke_result = Some(result);
        self.cookie_index = 0;
    }

    /// Returns a reference to the current key material, or `None` if no KE
    /// result has been set.
    #[must_use]
    pub fn key_material(&self) -> Option<&NtsKeyMaterial> {
        self.ke_result.as_ref().map(|r| &r.key_material)
    }

    /// Returns a reference to the session configuration.
    #[must_use]
    pub fn config(&self) -> &NtsClientConfig {
        &self.config
    }

    /// Generates a 32-byte Unique Identifier (RFC 8915 §5.3) derived from
    /// the current [`SystemTime`] nanosecond timestamp mixed with a counter.
    ///
    /// This is intentionally simple — in production, a cryptographically
    /// random source should be used.  The primary requirement is uniqueness
    /// across concurrent outstanding requests.
    #[must_use]
    pub fn unique_identifier() -> Vec<u8> {
        // Use nanoseconds since UNIX epoch spread across 32 bytes.
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0u128);

        let mut uid = [0u8; 32];
        // Fill with the raw nanos bytes and then XOR-mix for diversity.
        let nanos_bytes = nanos.to_le_bytes(); // 16 bytes
        uid[..16].copy_from_slice(&nanos_bytes);
        uid[16..32].copy_from_slice(&nanos_bytes);
        // XOR with a simple spread pattern to distinguish the two halves.
        for (i, b) in uid[16..32].iter_mut().enumerate() {
            *b ^= (i as u8).wrapping_add(0xA5);
        }
        uid.to_vec()
    }

    /// Invalidates the current session, requiring a fresh KE handshake.
    pub fn invalidate(&mut self) {
        self.ke_result = None;
        self.cookie_index = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key_material(algorithm: NtsAeadAlgorithm) -> NtsKeyMaterial {
        let len = algorithm.key_length_bytes();
        NtsKeyMaterial::new(vec![0xAB; len], vec![0xCD; len], algorithm)
            .expect("valid key material")
    }

    fn make_ke_result(n_cookies: usize) -> NtsKeResult {
        NtsKeResult {
            server_addr: "127.0.0.1:123".parse().expect("valid addr"),
            key_material: make_key_material(NtsAeadAlgorithm::AesSivCmac256),
            cookies: (0..n_cookies)
                .map(|i| NtsCookie::new(vec![i as u8; 32]))
                .collect(),
            next_protocol: 0,
        }
    }

    // -----------------------------------------------------------------------
    // NtsAeadAlgorithm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_algorithm_key_lengths() {
        assert_eq!(NtsAeadAlgorithm::AesSivCmac256.key_length_bytes(), 32);
        assert_eq!(NtsAeadAlgorithm::AesSivCmac384.key_length_bytes(), 48);
        assert_eq!(NtsAeadAlgorithm::AesSivCmac512.key_length_bytes(), 64);
    }

    #[test]
    fn test_algorithm_round_trip() {
        for algo in [
            NtsAeadAlgorithm::AesSivCmac256,
            NtsAeadAlgorithm::AesSivCmac384,
            NtsAeadAlgorithm::AesSivCmac512,
        ] {
            let v = algo.as_u16();
            let back = NtsAeadAlgorithm::from_u16(v).expect("round trip");
            assert_eq!(back, algo);
        }
    }

    #[test]
    fn test_algorithm_unknown_id_errors() {
        assert!(NtsAeadAlgorithm::from_u16(0).is_err());
        assert!(NtsAeadAlgorithm::from_u16(999).is_err());
    }

    // -----------------------------------------------------------------------
    // NtsKeyMaterial tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_key_material_wrong_length_errors() {
        let res = NtsKeyMaterial::new(
            vec![0u8; 16], // too short for AesSivCmac256
            vec![0u8; 32],
            NtsAeadAlgorithm::AesSivCmac256,
        );
        assert!(res.is_err(), "wrong c2s key length should error");
    }

    #[test]
    fn test_key_material_correct_length_ok() {
        let km = make_key_material(NtsAeadAlgorithm::AesSivCmac256);
        assert_eq!(km.c2s_key.len(), 32);
        assert_eq!(km.s2c_key.len(), 32);
    }

    // -----------------------------------------------------------------------
    // NtsSession tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_session_new_not_authenticated() {
        let session = NtsSession::new(NtsClientConfig::default());
        assert!(!session.is_authenticated(), "new session has no cookies");
        assert_eq!(session.remaining_cookies(), 0);
    }

    #[test]
    fn test_session_set_ke_result_authenticated() {
        let mut session = NtsSession::new(NtsClientConfig::default());
        session.set_ke_result(make_ke_result(4));
        assert!(session.is_authenticated());
        assert_eq!(session.remaining_cookies(), 4);
    }

    #[test]
    fn test_session_consume_cookie_depletes_pool() {
        let mut session = NtsSession::new(NtsClientConfig::default());
        session.set_ke_result(make_ke_result(3));

        for i in 0..3 {
            let cookie = session.consume_cookie();
            assert!(cookie.is_some(), "cookie {i} should be available");
            assert_eq!(session.remaining_cookies(), 2 - i);
        }
        assert!(session.consume_cookie().is_none(), "pool exhausted");
        assert!(!session.is_authenticated());
    }

    #[test]
    fn test_session_set_ke_result_resets_index() {
        let mut session = NtsSession::new(NtsClientConfig::default());
        session.set_ke_result(make_ke_result(2));
        let _ = session.consume_cookie();
        let _ = session.consume_cookie();
        assert!(!session.is_authenticated());

        // Setting a new result resets the index.
        session.set_ke_result(make_ke_result(5));
        assert!(session.is_authenticated());
        assert_eq!(session.remaining_cookies(), 5);
    }

    #[test]
    fn test_session_key_material_accessible() {
        let mut session = NtsSession::new(NtsClientConfig::default());
        assert!(session.key_material().is_none());
        session.set_ke_result(make_ke_result(1));
        assert!(session.key_material().is_some());
    }

    #[test]
    fn test_session_invalidate() {
        let mut session = NtsSession::new(NtsClientConfig::default());
        session.set_ke_result(make_ke_result(3));
        assert!(session.is_authenticated());
        session.invalidate();
        assert!(!session.is_authenticated());
        assert!(session.key_material().is_none());
    }

    // -----------------------------------------------------------------------
    // NullNtsAuthHook tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_null_hook_authenticate_returns_empty() {
        let hook = NullNtsAuthHook;
        let km = make_key_material(NtsAeadAlgorithm::AesSivCmac256);
        let cookie = NtsCookie::new(vec![0u8; 32]);
        let uid = NtsSession::unique_identifier();
        let ntp_header = [0u8; 48];
        let result = hook.authenticate_request(&ntp_header, &km, &cookie, &uid);
        assert!(result.is_ok());
        assert!(result.expect("authenticate should succeed").is_empty());
    }

    #[test]
    fn test_null_hook_verify_always_ok() {
        let hook = NullNtsAuthHook;
        let km = make_key_material(NtsAeadAlgorithm::AesSivCmac256);
        let uid = NtsSession::unique_identifier();
        let ntp_header = [0u8; 48];
        let ef_data = [0u8; 100];
        let result = hook.verify_response(&ntp_header, &ef_data, &km, &uid);
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // Unique identifier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_unique_identifier_length() {
        let uid = NtsSession::unique_identifier();
        assert_eq!(uid.len(), 32, "unique identifier must be 32 bytes");
    }

    #[test]
    fn test_unique_identifier_non_zero() {
        let uid = NtsSession::unique_identifier();
        // At minimum the nanos bytes should be non-zero (system clock is
        // well past epoch).
        let all_zero = uid.iter().all(|&b| b == 0);
        assert!(!all_zero, "unique identifier should not be all-zero");
    }

    // -----------------------------------------------------------------------
    // NtsCookie tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cookie_accessors() {
        let data = vec![0xAB, 0xCD, 0xEF];
        let cookie = NtsCookie::new(data.clone());
        assert_eq!(cookie.as_bytes(), data.as_slice());
        assert_eq!(cookie.len(), 3);
        assert!(!cookie.is_empty());
    }

    #[test]
    fn test_empty_cookie() {
        let cookie = NtsCookie::new(vec![]);
        assert!(cookie.is_empty());
        assert_eq!(cookie.len(), 0);
    }
}
