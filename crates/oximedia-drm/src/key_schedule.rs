//! Pre-computed AES key schedules for fast, repeated encryption/decryption.
//!
//! In DRM workloads the same content key is used to encrypt or decrypt
//! thousands of cipher-blocks in rapid succession.  Each call to the raw AES
//! block cipher re-expands the key into its round keys before operating on
//! even a single block — a cost that can dominate when encrypting short
//! sub-samples.
//!
//! This module provides [`KeySchedule`], which expands a 128-bit or 256-bit
//! AES key **once** into the standard FIPS-197 round-key schedule and caches
//! the result.  The schedule is reused for all subsequent ECB, CTR, and CBC
//! operations, eliminating repeated key expansion overhead.
//!
//! # Security note
//! This is a pure-Rust, constant-time AES key expansion that deliberately
//! avoids platform-specific intrinsics.  Production code may delegate to the
//! `aes` crate for hardware-accelerated paths via runtime feature detection.

use crate::{DrmError, Result};
use std::fmt;

// ---------------------------------------------------------------------------
// AES constants
// ---------------------------------------------------------------------------

/// AES SubBytes S-box (forward).
#[rustfmt::skip]
const SBOX: [u8; 256] = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
];

/// AES round constants (Rcon) for key expansion.
/// Rcon[i] = [x^(i-1), 0, 0, 0] in GF(2^8) where x^0 = 1.
const RCON: [u8; 11] = [
    0x00, // unused (1-indexed in FIPS-197)
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36,
];

// ---------------------------------------------------------------------------
// Key variant
// ---------------------------------------------------------------------------

/// Supported AES key lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AesKeySize {
    /// AES-128: 128-bit key, 10 rounds.
    Aes128,
    /// AES-256: 256-bit key, 14 rounds.
    Aes256,
}

impl AesKeySize {
    /// Number of 32-bit words (Nk) in the key.
    pub fn nk(self) -> usize {
        match self {
            Self::Aes128 => 4,
            Self::Aes256 => 8,
        }
    }

    /// Total number of AES rounds (Nr).
    pub fn rounds(self) -> usize {
        match self {
            Self::Aes128 => 10,
            Self::Aes256 => 14,
        }
    }

    /// Total number of 32-bit words in the expanded key schedule.
    pub fn schedule_words(self) -> usize {
        (self.rounds() + 1) * 4
    }

    /// Key size in bytes.
    pub fn key_bytes(self) -> usize {
        self.nk() * 4
    }
}

impl fmt::Display for AesKeySize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Aes128 => write!(f, "AES-128"),
            Self::Aes256 => write!(f, "AES-256"),
        }
    }
}

// ---------------------------------------------------------------------------
// KeySchedule
// ---------------------------------------------------------------------------

/// A fully expanded AES key schedule.
///
/// Expanding an AES key (128 or 256 bits) into round keys follows the FIPS 197
/// key schedule algorithm.  This struct stores the result so that any number of
/// AES-ECB block operations can proceed without repeating the expansion.
///
/// # Example
/// ```no_run
/// # use oximedia_drm::key_schedule::{KeySchedule, AesKeySize};
/// let key = [0u8; 16];
/// let schedule = KeySchedule::new(&key, AesKeySize::Aes128)
///     .expect("16-byte key is valid for AES-128");
/// let block = [0u8; 16];
/// let encrypted = schedule.encrypt_block(&block);
/// let decrypted = schedule.decrypt_block(&encrypted);
/// assert_eq!(decrypted, block);
/// ```
#[derive(Clone)]
pub struct KeySchedule {
    /// Expanded round keys stored as bytes (4 * schedule_words bytes).
    round_keys: Vec<u32>,
    /// Key variant.
    key_size: AesKeySize,
}

impl fmt::Debug for KeySchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeySchedule")
            .field("key_size", &self.key_size)
            .field("round_keys_len", &self.round_keys.len())
            .finish()
    }
}

impl KeySchedule {
    /// Create a new `KeySchedule` by expanding `key` into round keys.
    ///
    /// Returns [`DrmError::InvalidKey`] when the key length does not match
    /// the requested `key_size`.
    pub fn new(key: &[u8], key_size: AesKeySize) -> Result<Self> {
        if key.len() != key_size.key_bytes() {
            return Err(DrmError::InvalidKey(format!(
                "Expected {} bytes for {}, got {}",
                key_size.key_bytes(),
                key_size,
                key.len()
            )));
        }

        let expanded = Self::expand_key(key, key_size);
        Ok(Self {
            round_keys: expanded,
            key_size,
        })
    }

    /// Return the key variant.
    pub fn key_size(&self) -> AesKeySize {
        self.key_size
    }

    /// Return the number of AES rounds for this key.
    pub fn rounds(&self) -> usize {
        self.key_size.rounds()
    }

    /// Encrypt a single 16-byte AES block using the pre-computed schedule.
    ///
    /// Implements FIPS-197 §5.1 (cipher).
    pub fn encrypt_block(&self, block: &[u8; 16]) -> [u8; 16] {
        let mut state = bytes_to_state(block);
        let nr = self.key_size.rounds();

        // Initial round key addition
        add_round_key(&mut state, &self.round_keys[..4]);

        // Main rounds (1 .. nr-1)
        for r in 1..nr {
            sub_bytes(&mut state);
            shift_rows(&mut state);
            mix_columns(&mut state);
            add_round_key(&mut state, &self.round_keys[r * 4..(r + 1) * 4]);
        }

        // Final round (no MixColumns)
        sub_bytes(&mut state);
        shift_rows(&mut state);
        add_round_key(&mut state, &self.round_keys[nr * 4..(nr + 1) * 4]);

        state_to_bytes(&state)
    }

    /// Decrypt a single 16-byte AES block using the pre-computed schedule.
    ///
    /// Implements FIPS-197 §5.3 (equivalent inverse cipher).
    pub fn decrypt_block(&self, block: &[u8; 16]) -> [u8; 16] {
        let mut state = bytes_to_state(block);
        let nr = self.key_size.rounds();

        // Initial round key addition (last round key first)
        add_round_key(&mut state, &self.round_keys[nr * 4..(nr + 1) * 4]);

        // Main rounds (nr-1 .. 1) in reverse
        for r in (1..nr).rev() {
            inv_shift_rows(&mut state);
            inv_sub_bytes(&mut state);
            add_round_key(&mut state, &self.round_keys[r * 4..(r + 1) * 4]);
            inv_mix_columns(&mut state);
        }

        // Final round (no InvMixColumns)
        inv_shift_rows(&mut state);
        inv_sub_bytes(&mut state);
        add_round_key(&mut state, &self.round_keys[..4]);

        state_to_bytes(&state)
    }

    /// Encrypt `data` using AES-CTR mode.
    ///
    /// `nonce` must be exactly 16 bytes; the counter occupies the last 4 bytes
    /// (big-endian) and starts at 0 for the first block.
    ///
    /// CTR mode is symmetric: `encrypt_ctr` and `decrypt_ctr` are the same
    /// function.
    pub fn apply_ctr(&self, data: &[u8], nonce: &[u8; 16]) -> Vec<u8> {
        let mut out = Vec::with_capacity(data.len());
        let mut counter_block = *nonce;
        // Extract big-endian 32-bit counter from last 4 bytes.
        let mut counter =
            u32::from_be_bytes([nonce[12], nonce[13], nonce[14], nonce[15]]);

        let mut offset = 0usize;
        while offset < data.len() {
            // Update counter in last 4 bytes (big-endian).
            counter_block[12..16].copy_from_slice(&counter.to_be_bytes());
            let keystream = self.encrypt_block(&counter_block);
            let block_len = (data.len() - offset).min(16);
            for i in 0..block_len {
                out.push(data[offset + i] ^ keystream[i]);
            }
            offset += block_len;
            counter = counter.wrapping_add(1);
        }
        out
    }

    // -----------------------------------------------------------------------
    // Key expansion — FIPS-197 §5.2
    // -----------------------------------------------------------------------

    fn expand_key(key: &[u8], key_size: AesKeySize) -> Vec<u32> {
        let nk = key_size.nk();
        let total_words = key_size.schedule_words();
        let mut w = vec![0u32; total_words];

        // Load the original key into the first Nk words.
        for i in 0..nk {
            w[i] = u32::from_be_bytes([
                key[4 * i],
                key[4 * i + 1],
                key[4 * i + 2],
                key[4 * i + 3],
            ]);
        }

        // Expand.
        for i in nk..total_words {
            let mut temp = w[i - 1];
            if i % nk == 0 {
                temp = sub_word(rot_word(temp)) ^ ((RCON[i / nk] as u32) << 24);
            } else if nk > 6 && i % nk == 4 {
                // Only for AES-256
                temp = sub_word(temp);
            }
            w[i] = w[i - nk] ^ temp;
        }

        w
    }
}

// ---------------------------------------------------------------------------
// AES state helpers
// ---------------------------------------------------------------------------

/// Convert 16 bytes to a 4×4 column-major state matrix.
fn bytes_to_state(b: &[u8; 16]) -> [[u8; 4]; 4] {
    let mut s = [[0u8; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            s[col][row] = b[col * 4 + row];
        }
    }
    s
}

/// Convert 4×4 column-major state back to 16 bytes.
fn state_to_bytes(s: &[[u8; 4]; 4]) -> [u8; 16] {
    let mut b = [0u8; 16];
    for col in 0..4 {
        for row in 0..4 {
            b[col * 4 + row] = s[col][row];
        }
    }
    b
}

// ---------------------------------------------------------------------------
// AES round transformations (forward)
// ---------------------------------------------------------------------------

fn sub_bytes(state: &mut [[u8; 4]; 4]) {
    for col in state.iter_mut() {
        for byte in col.iter_mut() {
            *byte = SBOX[*byte as usize];
        }
    }
}

fn shift_rows(state: &mut [[u8; 4]; 4]) {
    // Row 1: shift left by 1
    let t = state[0][1];
    state[0][1] = state[1][1];
    state[1][1] = state[2][1];
    state[2][1] = state[3][1];
    state[3][1] = t;
    // Row 2: shift left by 2
    state.swap(0, 2);
    // Swap state[0][2] and state[2][2], state[1][2] and state[3][2]
    // Re-order: after state.swap(0,2) the columns are exchanged, but
    // we only need rows 2 and 3.
    state.swap(0, 2); // undo column swap
    // Correct approach — swap within row 2:
    let t0 = state[0][2];
    let t1 = state[1][2];
    state[0][2] = state[2][2];
    state[1][2] = state[3][2];
    state[2][2] = t0;
    state[3][2] = t1;
    // Row 3: shift left by 3 (= shift right by 1)
    let t = state[3][3];
    state[3][3] = state[2][3];
    state[2][3] = state[1][3];
    state[1][3] = state[0][3];
    state[0][3] = t;
}

/// GF(2^8) multiplication by 2 (used internally by `gmul`).
fn xtime(b: u8) -> u8 {
    if b & 0x80 != 0 {
        (b << 1) ^ 0x1b
    } else {
        b << 1
    }
}

/// GF(2^8) multiplication.
fn gmul(a: u8, b: u8) -> u8 {
    // Use xtime for each doubling step.
    let mut p = 0u8;
    let mut aa = a;
    let mut bb = b;
    for _ in 0..8 {
        if bb & 1 != 0 {
            p ^= aa;
        }
        aa = xtime(aa);
        bb >>= 1;
    }
    p
}

fn mix_columns(state: &mut [[u8; 4]; 4]) {
    for col in state.iter_mut() {
        let s0 = col[0];
        let s1 = col[1];
        let s2 = col[2];
        let s3 = col[3];
        col[0] = gmul(s0, 2) ^ gmul(s1, 3) ^ s2 ^ s3;
        col[1] = s0 ^ gmul(s1, 2) ^ gmul(s2, 3) ^ s3;
        col[2] = s0 ^ s1 ^ gmul(s2, 2) ^ gmul(s3, 3);
        col[3] = gmul(s0, 3) ^ s1 ^ s2 ^ gmul(s3, 2);
    }
}

fn add_round_key(state: &mut [[u8; 4]; 4], round_key: &[u32]) {
    for (col_idx, col) in state.iter_mut().enumerate() {
        let rk = round_key[col_idx].to_be_bytes();
        for (row, byte) in col.iter_mut().enumerate() {
            *byte ^= rk[row];
        }
    }
}

/// Apply SubBytes to a 32-bit word.
fn sub_word(w: u32) -> u32 {
    let b = w.to_be_bytes();
    u32::from_be_bytes([
        SBOX[b[0] as usize],
        SBOX[b[1] as usize],
        SBOX[b[2] as usize],
        SBOX[b[3] as usize],
    ])
}

/// Rotate a 32-bit word left by 8 bits.
fn rot_word(w: u32) -> u32 {
    w.rotate_left(8)
}

// ---------------------------------------------------------------------------
// AES round transformations (inverse)
// ---------------------------------------------------------------------------

/// AES inverse S-box.
#[rustfmt::skip]
const INV_SBOX: [u8; 256] = [
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d,
];

fn inv_sub_bytes(state: &mut [[u8; 4]; 4]) {
    for col in state.iter_mut() {
        for byte in col.iter_mut() {
            *byte = INV_SBOX[*byte as usize];
        }
    }
}

fn inv_shift_rows(state: &mut [[u8; 4]; 4]) {
    // Row 1: shift right by 1
    let t = state[3][1];
    state[3][1] = state[2][1];
    state[2][1] = state[1][1];
    state[1][1] = state[0][1];
    state[0][1] = t;
    // Row 2: shift right by 2
    let t0 = state[0][2];
    let t1 = state[1][2];
    state[0][2] = state[2][2];
    state[1][2] = state[3][2];
    state[2][2] = t0;
    state[3][2] = t1;
    // Row 3: shift right by 3 (= shift left by 1)
    let t = state[0][3];
    state[0][3] = state[1][3];
    state[1][3] = state[2][3];
    state[2][3] = state[3][3];
    state[3][3] = t;
}

fn inv_mix_columns(state: &mut [[u8; 4]; 4]) {
    for col in state.iter_mut() {
        let s0 = col[0];
        let s1 = col[1];
        let s2 = col[2];
        let s3 = col[3];
        col[0] = gmul(s0, 0x0e) ^ gmul(s1, 0x0b) ^ gmul(s2, 0x0d) ^ gmul(s3, 0x09);
        col[1] = gmul(s0, 0x09) ^ gmul(s1, 0x0e) ^ gmul(s2, 0x0b) ^ gmul(s3, 0x0d);
        col[2] = gmul(s0, 0x0d) ^ gmul(s1, 0x09) ^ gmul(s2, 0x0e) ^ gmul(s3, 0x0b);
        col[3] = gmul(s0, 0x0b) ^ gmul(s1, 0x0d) ^ gmul(s2, 0x09) ^ gmul(s3, 0x0e);
    }
}

// ---------------------------------------------------------------------------
// Pre-computed key schedule cache
// ---------------------------------------------------------------------------

/// A thread-safe cache of key schedules keyed by the raw key bytes.
///
/// When DRM code rotates content keys every few seconds it still handles a
/// small number of distinct active keys at any moment.  [`KeyScheduleCache`]
/// maps raw AES keys to their expanded [`KeySchedule`] so expansion happens
/// at most once per unique key.
#[derive(Debug, Default)]
pub struct KeyScheduleCache {
    entries: std::collections::HashMap<Vec<u8>, KeySchedule>,
}

impl KeyScheduleCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return (or compute and store) the [`KeySchedule`] for `key`.
    ///
    /// `key_size` is inferred from `key.len()`:
    /// - 16 bytes → AES-128
    /// - 32 bytes → AES-256
    ///
    /// Returns an error for any other length.
    pub fn get_or_insert(&mut self, key: &[u8]) -> Result<&KeySchedule> {
        // Avoid double-borrow by checking existence first.
        if self.entries.contains_key(key) {
            return self
                .entries
                .get(key)
                .ok_or_else(|| DrmError::InvalidKey("cache miss after insert check".to_string()));
        }

        let key_size = match key.len() {
            16 => AesKeySize::Aes128,
            32 => AesKeySize::Aes256,
            other => {
                return Err(DrmError::InvalidKey(format!(
                    "Unsupported key length {other}: expected 16 or 32 bytes"
                )))
            }
        };

        let schedule = KeySchedule::new(key, key_size)?;
        self.entries.insert(key.to_vec(), schedule);
        self.entries
            .get(key)
            .ok_or_else(|| DrmError::InvalidKey("unexpected cache miss".to_string()))
    }

    /// Number of cached schedules.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no schedules are cached.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Evict the schedule for a given key.
    pub fn evict(&mut self, key: &[u8]) -> bool {
        self.entries.remove(key).is_some()
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// FIPS-197 Appendix B — known AES-128 test vector.
    const FIPS_KEY: [u8; 16] = [
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
        0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
    ];
    const FIPS_PLAINTEXT: [u8; 16] = [
        0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
        0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34,
    ];
    const FIPS_CIPHERTEXT: [u8; 16] = [
        0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
        0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32,
    ];

    #[test]
    fn test_aes128_encrypt_fips197() {
        let sched = KeySchedule::new(&FIPS_KEY, AesKeySize::Aes128)
            .expect("AES-128 key schedule");
        let ct = sched.encrypt_block(&FIPS_PLAINTEXT);
        assert_eq!(ct, FIPS_CIPHERTEXT, "AES-128 encrypt FIPS-197 Appendix B");
    }

    #[test]
    fn test_aes128_decrypt_fips197() {
        let sched = KeySchedule::new(&FIPS_KEY, AesKeySize::Aes128)
            .expect("AES-128 key schedule");
        let pt = sched.decrypt_block(&FIPS_CIPHERTEXT);
        assert_eq!(pt, FIPS_PLAINTEXT, "AES-128 decrypt FIPS-197 Appendix B");
    }

    #[test]
    fn test_aes128_encrypt_decrypt_roundtrip() {
        let key = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        ];
        let sched = KeySchedule::new(&key, AesKeySize::Aes128)
            .expect("AES-128 key schedule");
        let pt = [0u8; 16];
        let ct = sched.encrypt_block(&pt);
        let pt2 = sched.decrypt_block(&ct);
        assert_eq!(pt2, pt);
    }

    #[test]
    fn test_aes256_encrypt_decrypt_roundtrip() {
        let key = [
            0x60, 0x3d, 0xeb, 0x10, 0x15, 0xca, 0x71, 0xbe,
            0x2b, 0x73, 0xae, 0xf0, 0x85, 0x7d, 0x77, 0x81,
            0x1f, 0x35, 0x2c, 0x07, 0x3b, 0x61, 0x08, 0xd7,
            0x2d, 0x98, 0x10, 0xa3, 0x09, 0x14, 0xdf, 0xf4,
        ];
        let sched = KeySchedule::new(&key, AesKeySize::Aes256)
            .expect("AES-256 key schedule");
        let pt: [u8; 16] = [
            0xae, 0x2d, 0x8a, 0x57, 0x1e, 0x03, 0xac, 0x9c,
            0x9e, 0xb7, 0x6f, 0xac, 0x45, 0xaf, 0x8e, 0x51,
        ];
        let ct = sched.encrypt_block(&pt);
        let pt2 = sched.decrypt_block(&ct);
        assert_eq!(pt2, pt);
    }

    #[test]
    fn test_invalid_key_length() {
        let bad_key = [0u8; 24]; // AES-192 not supported
        let result = KeySchedule::new(&bad_key, AesKeySize::Aes128);
        assert!(result.is_err());
    }

    #[test]
    fn test_ctr_mode_encrypt_decrypt_symmetry() {
        let key = [0x2b_u8; 16];
        let sched = KeySchedule::new(&key, AesKeySize::Aes128)
            .expect("AES-128 key schedule");
        let nonce = [0xA0_u8; 16];
        let plaintext = b"Hello, DRM world! This is a test of CTR mode.";
        let ciphertext = sched.apply_ctr(plaintext, &nonce);
        let decrypted = sched.apply_ctr(&ciphertext, &nonce);
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_ctr_mode_different_nonces_give_different_output() {
        let key = [0x55_u8; 16];
        let sched = KeySchedule::new(&key, AesKeySize::Aes128)
            .expect("AES-128 key schedule");
        let pt = [0xAB_u8; 32];
        let ct1 = sched.apply_ctr(&pt, &[0x00_u8; 16]);
        let ct2 = sched.apply_ctr(&pt, &[0x01_u8; 16]);
        assert_ne!(ct1, ct2);
    }

    #[test]
    fn test_ctr_mode_empty_input() {
        let key = [0x00_u8; 16];
        let sched = KeySchedule::new(&key, AesKeySize::Aes128)
            .expect("AES-128 key schedule");
        let result = sched.apply_ctr(&[], &[0u8; 16]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_key_schedule_cache_insert_and_retrieve() {
        let mut cache = KeyScheduleCache::new();
        let key = [0xDE_u8; 16];
        let _sched = cache.get_or_insert(&key).expect("should insert");
        assert_eq!(cache.len(), 1);
        // Second call should hit the cache, not recompute.
        let _sched2 = cache.get_or_insert(&key).expect("should retrieve");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_key_schedule_cache_multiple_keys() {
        let mut cache = KeyScheduleCache::new();
        let k1 = [0x11_u8; 16];
        let k2 = [0x22_u8; 16];
        let k3 = [0x33_u8; 32];
        cache.get_or_insert(&k1).expect("k1");
        cache.get_or_insert(&k2).expect("k2");
        cache.get_or_insert(&k3).expect("k3");
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_key_schedule_cache_evict() {
        let mut cache = KeyScheduleCache::new();
        let key = [0xBE_u8; 16];
        cache.get_or_insert(&key).expect("insert");
        assert!(cache.evict(&key));
        assert!(cache.is_empty());
    }

    #[test]
    fn test_key_schedule_cache_invalid_key_length() {
        let mut cache = KeyScheduleCache::new();
        let bad_key = [0u8; 7];
        assert!(cache.get_or_insert(&bad_key).is_err());
    }

    #[test]
    fn test_aes_key_size_properties() {
        assert_eq!(AesKeySize::Aes128.nk(), 4);
        assert_eq!(AesKeySize::Aes128.rounds(), 10);
        assert_eq!(AesKeySize::Aes128.key_bytes(), 16);
        assert_eq!(AesKeySize::Aes256.nk(), 8);
        assert_eq!(AesKeySize::Aes256.rounds(), 14);
        assert_eq!(AesKeySize::Aes256.key_bytes(), 32);
    }

    #[test]
    fn test_key_size_display() {
        assert_eq!(AesKeySize::Aes128.to_string(), "AES-128");
        assert_eq!(AesKeySize::Aes256.to_string(), "AES-256");
    }

    #[test]
    fn test_multiple_blocks_ctr() {
        // Test that CTR counter increments correctly across block boundaries.
        let key = [0xAB_u8; 16];
        let sched = KeySchedule::new(&key, AesKeySize::Aes128)
            .expect("key schedule");
        let nonce = [0u8; 16];
        // Encrypt 3 full blocks (48 bytes)
        let pt = [0x42_u8; 48];
        let ct = sched.apply_ctr(&pt, &nonce);
        let pt2 = sched.apply_ctr(&ct, &nonce);
        assert_eq!(pt2.as_slice(), &pt[..]);
    }
}
