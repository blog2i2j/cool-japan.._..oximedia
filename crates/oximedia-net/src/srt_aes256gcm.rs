//! SRT AES-256-GCM authenticated encryption layer.
//!
//! This module provides AES-256-GCM (Galois/Counter Mode) authenticated
//! encryption for SRT transport streams, complementing the existing AES-128-CTR
//! crypto in `crate::srt::crypto`.  AES-256-GCM adds authentication tags
//! to every ciphertext block, detecting tampering without a separate HMAC pass.
//!
//! # Design
//!
//! - **Key derivation**: PBKDF2-SHA256 with a configurable iteration count
//!   derives a 32-byte (256-bit) session key from a passphrase and a random
//!   16-byte salt.
//! - **IV construction**: A 12-byte nonce = 4-byte stream-ID XOR packet-seq
//!   prefix ‖ 8-byte random base.  The nonce is never reused per key.
//! - **Packet wire format**: `[salt (16)] [nonce (12)] [ciphertext] [tag (16)]`
//! - **Key rotation**: The [`KeyRotationPolicy`] triggers re-keying after N
//!   packets or T seconds, whichever comes first.
//!
//! All types are `Send + Sync` and operate without allocation in the hot path
//! (encrypt/decrypt work on caller-supplied buffers).

use std::fmt;
use std::time::{Duration, Instant};

use aes_gcm::aead::{AeadInPlace, KeyInit};
use aes_gcm::{Aes256Gcm, Key, Nonce};

use crate::error::{NetError, NetResult};

// ─── Key derivation ──────────────────────────────────────────────────────────

/// PBKDF2 iteration count used for key derivation.
///
/// Increase for stronger key-stretching at the cost of handshake latency.
pub const PBKDF2_ITERATIONS: u32 = 65_536;

/// Length of the AES-256 key in bytes.
pub const KEY_LEN: usize = 32;

/// Length of the GCM nonce in bytes (96-bit, per NIST SP 800-38D).
pub const NONCE_LEN: usize = 12;

/// Length of the GCM authentication tag appended to every ciphertext.
pub const TAG_LEN: usize = 16;

/// Random salt length used in key derivation.
pub const SALT_LEN: usize = 16;

/// Derives a 256-bit session key from `passphrase` and `salt` using
/// PBKDF2-SHA256.
///
/// The returned array is suitable for use as an [`Aes256Gcm`] key.
///
/// # Errors
///
/// Returns [`NetError::Encoding`] if the passphrase is empty.
pub fn derive_key(passphrase: &[u8], salt: &[u8], iterations: u32) -> NetResult<[u8; KEY_LEN]> {
    if passphrase.is_empty() {
        return Err(NetError::encoding("passphrase must not be empty"));
    }
    // Manual PBKDF2-SHA256 using only sha2 (already a workspace dep).
    // PBKDF2(PRF, Password, Salt, c, dkLen):
    //   U1  = PRF(Password, Salt || INT(i))
    //   Ui  = PRF(Password, U_{i-1})
    //   Ti  = U1 XOR U2 XOR … XOR Uc
    use sha2::Sha256;
    // We need HMAC-SHA256 as PRF; hmac crate is a workspace dep.
    use hmac::{Hmac, KeyInit, Mac};

    type HmacSha256 = Hmac<Sha256>;

    // Only one block needed (dkLen = 32 = SHA256 output length).
    let block_index: u32 = 1;
    let mut u = {
        let mut mac = HmacSha256::new_from_slice(passphrase)
            .map_err(|e| NetError::encoding(format!("HMAC init: {e}")))?;
        mac.update(salt);
        mac.update(&block_index.to_be_bytes());
        mac.finalize().into_bytes()
    };

    let mut result = u;

    for _ in 1..iterations {
        let mut mac = HmacSha256::new_from_slice(passphrase)
            .map_err(|e| NetError::encoding(format!("HMAC init: {e}")))?;
        mac.update(&u);
        u = mac.finalize().into_bytes();
        for (r, &b) in result.iter_mut().zip(u.iter()) {
            *r ^= b;
        }
    }

    let mut key = [0u8; KEY_LEN];
    key.copy_from_slice(&result[..KEY_LEN]);
    Ok(key)
}

// ─── Nonce management ────────────────────────────────────────────────────────

/// 8-byte random base combined with a 4-byte counter to produce 12-byte nonces.
///
/// The random base is generated once per session key; the counter increments
/// for every encrypted packet.  This guarantees nonce uniqueness within the
/// lifetime of a single key (up to 2³² packets ≈ 4 billion).
#[derive(Debug, Clone)]
pub struct NonceGenerator {
    base: [u8; 8],
    counter: u32,
    stream_prefix: u32,
}

impl NonceGenerator {
    /// Creates a new generator with the given 8-byte random `base` and a
    /// 4-byte `stream_prefix` (typically derived from the SRT stream ID).
    #[must_use]
    pub fn new(base: [u8; 8], stream_prefix: u32) -> Self {
        Self {
            base,
            counter: 0,
            stream_prefix,
        }
    }

    /// Returns the next 12-byte nonce and advances the counter.
    ///
    /// Layout: `[stream_prefix XOR counter (4 BE)] [base (8)]`
    ///
    /// # Errors
    ///
    /// Returns `Err` after counter wrap-around (2³² packets), at which point
    /// the session key **must** be rotated before further use.
    pub fn next(&mut self) -> NetResult<[u8; NONCE_LEN]> {
        let ctr = self
            .counter
            .checked_add(1)
            .ok_or_else(|| NetError::encoding("nonce counter overflow — rotate session key"))?;
        self.counter = ctr;
        let prefix = self.stream_prefix ^ ctr;
        let mut nonce = [0u8; NONCE_LEN];
        nonce[..4].copy_from_slice(&prefix.to_be_bytes());
        nonce[4..].copy_from_slice(&self.base);
        Ok(nonce)
    }

    /// Returns the current counter value (number of nonces issued).
    #[must_use]
    pub const fn counter(&self) -> u32 {
        self.counter
    }
}

// ─── Key rotation policy ─────────────────────────────────────────────────────

/// Policy that decides when to rotate the AES-256-GCM session key.
///
/// Key rotation fires whenever either of the two limits is reached first.
#[derive(Debug, Clone)]
pub struct KeyRotationPolicy {
    /// Maximum number of packets encrypted under a single key before rotation.
    pub max_packets: u64,
    /// Maximum wall-clock duration before rotation.
    pub max_duration: Duration,
}

impl Default for KeyRotationPolicy {
    fn default() -> Self {
        Self {
            max_packets: 1_000_000,
            max_duration: Duration::from_secs(3600),
        }
    }
}

/// Tracks when the next key rotation is due.
#[derive(Debug)]
pub struct RotationTracker {
    policy: KeyRotationPolicy,
    packets_since_rotation: u64,
    rotated_at: Instant,
}

impl RotationTracker {
    /// Creates a new tracker using `policy`.
    #[must_use]
    pub fn new(policy: KeyRotationPolicy) -> Self {
        Self {
            policy,
            packets_since_rotation: 0,
            rotated_at: Instant::now(),
        }
    }

    /// Records that one more packet was processed and returns `true` if the
    /// key should be rotated.
    pub fn tick(&mut self) -> bool {
        self.packets_since_rotation += 1;
        self.packets_since_rotation >= self.policy.max_packets
            || self.rotated_at.elapsed() >= self.policy.max_duration
    }

    /// Resets the tracker after a successful key rotation.
    pub fn reset(&mut self) {
        self.packets_since_rotation = 0;
        self.rotated_at = Instant::now();
    }

    /// Packets processed under the current key.
    #[must_use]
    pub const fn packets(&self) -> u64 {
        self.packets_since_rotation
    }
}

// ─── Session ──────────────────────────────────────────────────────────────────

/// AES-256-GCM session: holds the cipher, nonce generator, and rotation state.
///
/// # Wire format
///
/// Each encrypted frame produced by [`Session::encrypt`] has the layout:
///
/// ```text
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │  salt (16 B)  │  nonce (12 B)  │  ciphertext  │  GCM tag (16 B)    │
/// └──────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// The salt is included so that the receiver can re-derive the key on a
/// renegotiation without extra signalling.  For normal packet flow the
/// receiver caches the derived key and only uses the nonce field.
pub struct Session {
    cipher: Aes256Gcm,
    nonce_gen: NonceGenerator,
    rotation: RotationTracker,
    salt: [u8; SALT_LEN],
    /// Bytes of additional associated data prepended before the GCM tag.
    aad: Vec<u8>,
}

impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("counter", &self.nonce_gen.counter())
            .field("packets", &self.rotation.packets())
            .finish()
    }
}

impl Session {
    /// Creates a new session from a raw 256-bit key material, nonce base,
    /// stream prefix, salt, and rotation policy.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `key_bytes` is not exactly 32 bytes.
    pub fn from_key(
        key_bytes: &[u8; KEY_LEN],
        nonce_base: [u8; 8],
        stream_prefix: u32,
        salt: [u8; SALT_LEN],
        policy: KeyRotationPolicy,
        aad: Vec<u8>,
    ) -> NetResult<Self> {
        let key = Key::<Aes256Gcm>::from_slice(key_bytes);
        let cipher = Aes256Gcm::new(key);
        Ok(Self {
            cipher,
            nonce_gen: NonceGenerator::new(nonce_base, stream_prefix),
            rotation: RotationTracker::new(policy),
            salt,
            aad,
        })
    }

    /// Creates a session by deriving the key from `passphrase` and `salt`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if key derivation fails.
    pub fn from_passphrase(
        passphrase: &[u8],
        salt: [u8; SALT_LEN],
        nonce_base: [u8; 8],
        stream_prefix: u32,
        policy: KeyRotationPolicy,
        aad: Vec<u8>,
    ) -> NetResult<Self> {
        let key_bytes = derive_key(passphrase, &salt, PBKDF2_ITERATIONS)?;
        Self::from_key(&key_bytes, nonce_base, stream_prefix, salt, policy, aad)
    }

    /// Returns `true` if the rotation policy says the key should be rotated.
    #[must_use]
    pub fn needs_rotation(&self) -> bool {
        // peek without mutating
        self.rotation.packets_since_rotation >= self.rotation.policy.max_packets
            || self.rotation.rotated_at.elapsed() >= self.rotation.policy.max_duration
    }

    /// Encrypts `plaintext` in-place, writing the wire frame into `out`.
    ///
    /// `out` must be at least `SALT_LEN + NONCE_LEN + plaintext.len() + TAG_LEN`
    /// bytes long.  Returns the number of bytes written.
    ///
    /// # Errors
    ///
    /// Returns `Err` on nonce exhaustion or AES-GCM failure.
    pub fn encrypt(&mut self, plaintext: &[u8], out: &mut Vec<u8>) -> NetResult<usize> {
        let nonce_bytes = self.nonce_gen.next()?;
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Build buffer: salt | nonce | plaintext (to be encrypted in-place).
        // We encrypt into a temporary Vec because encrypt_in_place requires a
        // Vec<u8> (which implements the aead::Buffer trait).
        let header_len = SALT_LEN + NONCE_LEN;
        let mut ciphertext_buf: Vec<u8> = plaintext.to_vec();

        self.cipher
            .encrypt_in_place(nonce, &self.aad, &mut ciphertext_buf)
            .map_err(|e| NetError::encoding(format!("AES-GCM encrypt: {e}")))?;

        out.clear();
        out.extend_from_slice(&self.salt);
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext_buf);
        let _ = header_len;

        let written = out.len();
        let _ = self.rotation.tick();
        Ok(written)
    }

    /// Decrypts a wire frame produced by [`Session::encrypt`].
    ///
    /// Returns the recovered plaintext.  The `salt` field in the frame is not
    /// re-used for key derivation by this method; callers should derive the key
    /// once and supply it to [`Session::from_key`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if the frame is too short, the nonce is malformed, or the
    /// GCM authentication tag fails (tampering detected).
    pub fn decrypt(&mut self, frame: &[u8]) -> NetResult<Vec<u8>> {
        let min_len = SALT_LEN + NONCE_LEN + TAG_LEN;
        if frame.len() < min_len {
            return Err(NetError::encoding(format!(
                "frame too short: {} < {min_len}",
                frame.len()
            )));
        }
        // Skip salt (bytes 0..16); extract nonce (bytes 16..28).
        let nonce_bytes = &frame[SALT_LEN..SALT_LEN + NONCE_LEN];
        let nonce = Nonce::from_slice(nonce_bytes);

        let mut buf = frame[SALT_LEN + NONCE_LEN..].to_vec();
        self.cipher
            .decrypt_in_place(nonce, &self.aad, &mut buf)
            .map_err(|_| NetError::authentication("AES-GCM tag mismatch — possible tampering"))?;

        let _ = self.rotation.tick();
        Ok(buf)
    }

    /// Returns the salt embedded in this session (for renegotiation).
    #[must_use]
    pub const fn salt(&self) -> &[u8; SALT_LEN] {
        &self.salt
    }

    /// Resets the rotation tracker (call after issuing a new key).
    pub fn reset_rotation(&mut self) {
        self.rotation.reset();
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_session() -> Session {
        let salt = [0xABu8; SALT_LEN];
        let nonce_base = [0x11u8; 8];
        Session::from_passphrase(
            b"correct-horse-battery-staple",
            salt,
            nonce_base,
            42,
            Default::default(),
            b"srt-stream".to_vec(),
        )
        .expect("session creation")
    }

    // 1. Key derivation produces a 32-byte key
    #[test]
    fn test_derive_key_length() {
        let key = derive_key(b"passphrase", b"randomsalt12345!", 1).expect("derive_key");
        assert_eq!(key.len(), KEY_LEN);
    }

    // 2. Same inputs produce the same key (determinism)
    #[test]
    fn test_derive_key_deterministic() {
        let k1 = derive_key(b"secret", b"saltsalt", 100).expect("k1");
        let k2 = derive_key(b"secret", b"saltsalt", 100).expect("k2");
        assert_eq!(k1, k2);
    }

    // 3. Different salts produce different keys
    #[test]
    fn test_derive_key_salt_sensitivity() {
        let k1 = derive_key(b"secret", b"salt_aaa", 10).expect("k1");
        let k2 = derive_key(b"secret", b"salt_bbb", 10).expect("k2");
        assert_ne!(k1, k2);
    }

    // 4. Empty passphrase returns an error
    #[test]
    fn test_derive_key_empty_passphrase_error() {
        let result = derive_key(b"", b"anysalt", 1);
        assert!(result.is_err());
    }

    // 5. Nonce generator advances counter
    #[test]
    fn test_nonce_generator_counter() {
        let mut gen = NonceGenerator::new([0u8; 8], 0);
        assert_eq!(gen.counter(), 0);
        gen.next().expect("nonce 1");
        assert_eq!(gen.counter(), 1);
        gen.next().expect("nonce 2");
        assert_eq!(gen.counter(), 2);
    }

    // 6. Successive nonces differ
    #[test]
    fn test_nonce_generator_unique() {
        let mut gen = NonceGenerator::new([7u8; 8], 99);
        let n1 = gen.next().expect("n1");
        let n2 = gen.next().expect("n2");
        assert_ne!(n1, n2, "nonces must be unique");
    }

    // 7. Round-trip encrypt → decrypt recovers plaintext
    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let mut session = test_session();
        let plaintext = b"SRT AES-256-GCM test payload";
        let mut frame = Vec::new();
        let written = session.encrypt(plaintext, &mut frame).expect("encrypt");
        assert!(written > plaintext.len());

        // Decrypt with a fresh session using the same key material.
        let mut dec = test_session();
        let recovered = dec.decrypt(&frame).expect("decrypt");
        assert_eq!(recovered, plaintext);
    }

    // 8. Tampered ciphertext fails authentication
    #[test]
    fn test_decrypt_tamper_detection() {
        let mut enc = test_session();
        let mut frame = Vec::new();
        enc.encrypt(b"secret payload", &mut frame).expect("encrypt");

        // Flip a byte in the ciphertext region.
        let ciphertext_offset = SALT_LEN + NONCE_LEN;
        frame[ciphertext_offset] ^= 0xFF;

        let mut dec = test_session();
        let result = dec.decrypt(&frame);
        assert!(result.is_err(), "tampered frame must fail authentication");
    }

    // 9. Frame shorter than minimum returns parse error
    #[test]
    fn test_decrypt_short_frame() {
        let mut session = test_session();
        let short = vec![0u8; SALT_LEN + NONCE_LEN - 1];
        let result = session.decrypt(&short);
        assert!(result.is_err());
    }

    // 10. Rotation tracker fires after max_packets
    #[test]
    fn test_rotation_tracker_fires() {
        let policy = KeyRotationPolicy {
            max_packets: 5,
            max_duration: Duration::from_secs(3600),
        };
        let mut tracker = RotationTracker::new(policy);
        for _ in 0..4 {
            assert!(!tracker.tick(), "should not fire yet");
        }
        assert!(tracker.tick(), "should fire at packet 5");
    }

    // 11. Rotation tracker resets after reset()
    #[test]
    fn test_rotation_tracker_reset() {
        let policy = KeyRotationPolicy {
            max_packets: 3,
            max_duration: Duration::from_secs(3600),
        };
        let mut tracker = RotationTracker::new(policy);
        tracker.tick();
        tracker.tick();
        tracker.tick();
        tracker.reset();
        assert_eq!(tracker.packets(), 0);
    }

    // 12. Session needs_rotation reflects tracker state
    #[test]
    fn test_needs_rotation_packet_limit() {
        let salt = [0u8; SALT_LEN];
        let policy = KeyRotationPolicy {
            max_packets: 2,
            max_duration: Duration::from_secs(3600),
        };
        let mut session =
            Session::from_passphrase(b"pass", salt, [0u8; 8], 0, policy, vec![]).expect("session");
        assert!(!session.needs_rotation());
        let mut buf = Vec::new();
        session.encrypt(b"x", &mut buf).expect("enc1");
        session.encrypt(b"y", &mut buf).expect("enc2");
        assert!(session.needs_rotation());
    }

    // 13. salt() accessor returns the session salt
    #[test]
    fn test_salt_accessor() {
        let salt = [0xCAu8; SALT_LEN];
        let session =
            Session::from_passphrase(b"pw", salt, [0u8; 8], 0, Default::default(), vec![])
                .expect("session");
        assert_eq!(session.salt(), &salt);
    }

    // 14. Wire frame starts with salt bytes
    #[test]
    fn test_wire_frame_starts_with_salt() {
        let salt = [0xBBu8; SALT_LEN];
        let mut session =
            Session::from_passphrase(b"pw", salt, [0u8; 8], 0, Default::default(), vec![])
                .expect("session");
        let mut frame = Vec::new();
        session.encrypt(b"data", &mut frame).expect("encrypt");
        assert_eq!(&frame[..SALT_LEN], &salt, "frame must begin with salt");
    }

    // 15. Encrypt with different AADs produces different ciphertexts
    #[test]
    fn test_aad_differentiation() {
        let salt = [0u8; SALT_LEN];
        let mut s1 = Session::from_passphrase(
            b"pw",
            salt,
            [0u8; 8],
            0,
            Default::default(),
            b"aad-a".to_vec(),
        )
        .expect("s1");
        let mut s2 = Session::from_passphrase(
            b"pw",
            salt,
            [0u8; 8],
            0,
            Default::default(),
            b"aad-b".to_vec(),
        )
        .expect("s2");
        let plaintext = b"same plaintext";
        let mut f1 = Vec::new();
        let mut f2 = Vec::new();
        s1.encrypt(plaintext, &mut f1).expect("enc1");
        s2.encrypt(plaintext, &mut f2).expect("enc2");
        // Ciphertexts differ because nonce sequence starts the same but AAD differs.
        // At minimum, the GCM tags should differ.
        assert_ne!(
            &f1[SALT_LEN + NONCE_LEN..],
            &f2[SALT_LEN + NONCE_LEN..],
            "different AAD must produce different ciphertext+tag"
        );
    }
}
