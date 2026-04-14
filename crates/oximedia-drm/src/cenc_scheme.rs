//! CENC (Common Encryption) scheme registry and validation.
//!
//! ISO/IEC 23001-7 defines four encryption scheme variants that differ in the
//! encryption algorithm, the number of encrypted bytes per NAL unit, and the
//! initialization vector treatment.  Choosing the wrong scheme for a given
//! codec or DRM system causes silent decryption failures in real deployments.
//!
//! This module provides:
//! - [`CencScheme`] — strongly-typed enum for the four scheme FourCCs
//! - [`SchemeDescriptor`] — structured metadata about each scheme's
//!   algorithm, IV size, and codec compatibility
//! - [`SubsamplePattern`] — CENC subsample encryption pattern (used by `cens`
//!   and `cbcs` in pattern mode)
//! - [`CencSchemeSelector`] — policy helper that recommends the correct scheme
//!   for a (DRM system, codec, mode) combination
//!
//! # CENC scheme summary
//!
//! | FourCC | Algorithm | Pattern | FairPlay | Widevine | PlayReady |
//! |--------|-----------|---------|----------|----------|-----------|
//! | `cenc` | AES-CTR   | No      | No       | Yes      | Yes       |
//! | `cbc1` | AES-CBC   | No      | Partial  | No       | Yes       |
//! | `cens` | AES-CTR   | Yes     | No       | Yes      | Yes       |
//! | `cbcs` | AES-CBC   | Yes     | Yes      | Yes      | Yes       |

use crate::{DrmError, DrmSystem, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// CencScheme
// ---------------------------------------------------------------------------

/// The four CENC protection schemes defined in ISO/IEC 23001-7:2022.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CencScheme {
    /// AES-128-CTR full-sample encryption (no pattern).
    ///
    /// The original Common Encryption scheme.  Supported by all CDMs but
    /// **not** by Apple FairPlay.  Uses an 8-byte IV (64-bit, big-endian
    /// counter).
    Cenc,

    /// AES-128-CBC full-sample encryption (no pattern).
    ///
    /// Introduced in CENC version 3 as a base for `cbcs`.  Not widely
    /// supported by Widevine L1.  Uses a 16-byte IV.
    Cbc1,

    /// AES-128-CTR encryption with subsample pattern.
    ///
    /// Identical to `cenc` algorithm but applies a (crypt_byte_count,
    /// skip_byte_count) pattern so only a fraction of each subsample is
    /// encrypted.  Used for video NAL units in DASH low-latency profiles.
    Cens,

    /// AES-128-CBC encryption with subsample pattern.
    ///
    /// Required by Apple FairPlay Streaming and supported by all major CDMs.
    /// Each subsample has a cleartext leader followed by full-block-aligned
    /// ciphertext, with the IV reset to the constant Initialization Vector
    /// per subsample.  Uses a 16-byte IV.
    Cbcs,
}

impl CencScheme {
    /// Return the ISO/IEC 23001-7 four-character code (FourCC) for this scheme.
    pub fn fourcc(self) -> &'static str {
        match self {
            Self::Cenc => "cenc",
            Self::Cbc1 => "cbc1",
            Self::Cens => "cens",
            Self::Cbcs => "cbcs",
        }
    }

    /// Parse a scheme from its FourCC string (case-insensitive).
    ///
    /// Returns [`DrmError::UnsupportedDrmSystem`] for unknown FourCCs.
    pub fn from_fourcc(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cenc" => Ok(Self::Cenc),
            "cbc1" => Ok(Self::Cbc1),
            "cens" => Ok(Self::Cens),
            "cbcs" => Ok(Self::Cbcs),
            other => Err(DrmError::UnsupportedDrmSystem(format!(
                "Unknown CENC scheme FourCC '{other}'"
            ))),
        }
    }

    /// Return the underlying block cipher mode for this scheme.
    pub fn cipher_mode(self) -> CipherMode {
        match self {
            Self::Cenc | Self::Cens => CipherMode::AesCtr,
            Self::Cbc1 | Self::Cbcs => CipherMode::AesCbc,
        }
    }

    /// Return the required IV size in bytes.
    ///
    /// CENC CTR-mode uses 8-byte IVs (MPEG-CENC §10.1); CBC-mode uses
    /// full 16-byte IVs (MPEG-CENC §10.4).
    pub fn iv_size_bytes(self) -> usize {
        match self {
            Self::Cenc | Self::Cens => 8,
            Self::Cbc1 | Self::Cbcs => 16,
        }
    }

    /// Return `true` if this scheme uses a subsample encryption pattern.
    pub fn uses_pattern(self) -> bool {
        matches!(self, Self::Cens | Self::Cbcs)
    }

    /// Return `true` if the scheme is supported by the given DRM system.
    pub fn supported_by(self, drm: DrmSystem) -> bool {
        match (drm, self) {
            // Widevine does not natively support cbc1
            (DrmSystem::Widevine, Self::Cbc1) => false,
            // FairPlay only supports cbcs
            (DrmSystem::FairPlay, Self::Cenc)
            | (DrmSystem::FairPlay, Self::Cbc1)
            | (DrmSystem::FairPlay, Self::Cens) => false,
            _ => true,
        }
    }

    /// Return the canonical `schemeIdUri` value for DASH ContentProtection.
    pub fn dash_scheme_uri(self) -> &'static str {
        "urn:mpeg:dash:mp4protection:2011"
    }

    /// Return the `value` attribute used in a DASH `ContentProtection` element.
    pub fn dash_value(self) -> &'static str {
        self.fourcc()
    }
}

impl fmt::Display for CencScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.fourcc())
    }
}

// ---------------------------------------------------------------------------
// CipherMode
// ---------------------------------------------------------------------------

/// Block cipher mode used by a CENC scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CipherMode {
    /// AES Counter (CTR) mode.
    AesCtr,
    /// AES Cipher Block Chaining (CBC) mode.
    AesCbc,
}

impl fmt::Display for CipherMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AesCtr => write!(f, "AES-CTR"),
            Self::AesCbc => write!(f, "AES-CBC"),
        }
    }
}

// ---------------------------------------------------------------------------
// SubsamplePattern
// ---------------------------------------------------------------------------

/// CENC pattern encryption parameters (ISO/IEC 23001-7 §10.3).
///
/// Applied to `cens` and `cbcs` schemes to reduce the fraction of data that
/// is encrypted, improving performance for high-bit-rate video streams.
///
/// For each NAL unit the pattern repeats over full 16-byte blocks:
/// - `crypt_byte_count` blocks are encrypted
/// - `skip_byte_count` blocks are left in the clear
///
/// A cleartext leading area (`constant_iv_offset`) precedes the pattern for
/// `cbcs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubsamplePattern {
    /// Number of 16-byte blocks to encrypt in each pattern cycle.
    pub crypt_byte_count: u8,
    /// Number of 16-byte blocks to leave clear in each pattern cycle.
    pub skip_byte_count: u8,
}

impl SubsamplePattern {
    /// Create a new subsample pattern.
    ///
    /// Returns an error when both values are zero (degenerate pattern).
    pub fn new(crypt: u8, skip: u8) -> Result<Self> {
        if crypt == 0 && skip == 0 {
            return Err(DrmError::ConfigError(
                "SubsamplePattern: both crypt and skip cannot be zero".to_string(),
            ));
        }
        Ok(Self {
            crypt_byte_count: crypt,
            skip_byte_count: skip,
        })
    }

    /// Recommended CMAF/DASH pattern for HEVC/AVC HLS (`cbcs`).
    ///
    /// Per the Common Media Application Format spec: 1 encrypted block,
    /// 9 clear blocks.
    pub fn cmaf_video() -> Self {
        Self {
            crypt_byte_count: 1,
            skip_byte_count: 9,
        }
    }

    /// Full encryption (all blocks encrypted, no skip).
    ///
    /// Equivalent to `cenc`/`cbc1` full-sample mode but expressed as a
    /// pattern.
    pub fn full() -> Self {
        Self {
            crypt_byte_count: 10,
            skip_byte_count: 0,
        }
    }

    /// Fraction of data that is encrypted (0.0–1.0).
    pub fn encrypted_fraction(self) -> f64 {
        let total = self.crypt_byte_count as f64 + self.skip_byte_count as f64;
        if total == 0.0 {
            0.0
        } else {
            self.crypt_byte_count as f64 / total
        }
    }

    /// Number of bytes encrypted per full pattern period given `block_count`
    /// 16-byte blocks.
    pub fn encrypted_blocks(self, block_count: usize) -> usize {
        let period = self.crypt_byte_count as usize + self.skip_byte_count as usize;
        if period == 0 {
            return 0;
        }
        let full_periods = block_count / period;
        let remainder = block_count % period;
        let crypt = self.crypt_byte_count as usize;
        full_periods * crypt + remainder.min(crypt)
    }
}

impl fmt::Display for SubsamplePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.crypt_byte_count, self.skip_byte_count)
    }
}

// ---------------------------------------------------------------------------
// SchemeDescriptor
// ---------------------------------------------------------------------------

/// Structured metadata describing a CENC scheme.
#[derive(Debug, Clone)]
pub struct SchemeDescriptor {
    /// The scheme.
    pub scheme: CencScheme,
    /// Human-readable description.
    pub description: &'static str,
    /// Cipher mode.
    pub cipher_mode: CipherMode,
    /// Required IV size in bytes.
    pub iv_size: usize,
    /// Whether the scheme uses a subsample encryption pattern.
    pub uses_pattern: bool,
    /// DRM systems that support this scheme.
    pub supported_by: &'static [DrmSystem],
}

impl SchemeDescriptor {
    /// Return the descriptor for `scheme`.
    pub fn for_scheme(scheme: CencScheme) -> &'static Self {
        match scheme {
            CencScheme::Cenc => &CENC_DESC,
            CencScheme::Cbc1 => &CBC1_DESC,
            CencScheme::Cens => &CENS_DESC,
            CencScheme::Cbcs => &CBCS_DESC,
        }
    }
}

static CENC_DESC: SchemeDescriptor = SchemeDescriptor {
    scheme: CencScheme::Cenc,
    description: "AES-128-CTR full-sample encryption (original CENC)",
    cipher_mode: CipherMode::AesCtr,
    iv_size: 8,
    uses_pattern: false,
    supported_by: &[DrmSystem::Widevine, DrmSystem::PlayReady, DrmSystem::ClearKey],
};

static CBC1_DESC: SchemeDescriptor = SchemeDescriptor {
    scheme: CencScheme::Cbc1,
    description: "AES-128-CBC full-sample encryption (CENC v3)",
    cipher_mode: CipherMode::AesCbc,
    iv_size: 16,
    uses_pattern: false,
    supported_by: &[DrmSystem::PlayReady, DrmSystem::ClearKey],
};

static CENS_DESC: SchemeDescriptor = SchemeDescriptor {
    scheme: CencScheme::Cens,
    description: "AES-128-CTR pattern-mode encryption",
    cipher_mode: CipherMode::AesCtr,
    iv_size: 8,
    uses_pattern: true,
    supported_by: &[DrmSystem::Widevine, DrmSystem::PlayReady, DrmSystem::ClearKey],
};

static CBCS_DESC: SchemeDescriptor = SchemeDescriptor {
    scheme: CencScheme::Cbcs,
    description: "AES-128-CBC pattern-mode encryption (FairPlay-compatible)",
    cipher_mode: CipherMode::AesCbc,
    iv_size: 16,
    uses_pattern: true,
    supported_by: &[
        DrmSystem::Widevine,
        DrmSystem::PlayReady,
        DrmSystem::FairPlay,
        DrmSystem::ClearKey,
    ],
};

// ---------------------------------------------------------------------------
// CencSchemeSelector
// ---------------------------------------------------------------------------

/// Codec class that influences scheme selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecClass {
    /// H.264 / AVC video.
    Avc,
    /// H.265 / HEVC video.
    Hevc,
    /// VP8 / VP9 video.
    Vpx,
    /// AV1 video.
    Av1,
    /// AAC or any other audio codec.
    Audio,
}

/// Encryption coverage preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncryptionCoverage {
    /// Encrypt all bytes (maximum DRM protection).
    Full,
    /// Encrypt using the default pattern for the codec / scheme.
    Pattern,
}

/// Policy helper that selects the best CENC scheme for given constraints.
#[derive(Debug, Default)]
pub struct CencSchemeSelector;

impl CencSchemeSelector {
    /// Create a new selector.
    pub fn new() -> Self {
        Self
    }

    /// Recommend the best CENC scheme for `drm_systems` (the set of DRM
    /// systems that must all be satisfied), `codec`, and `coverage`.
    ///
    /// Returns an error when no single scheme satisfies all DRM systems.
    pub fn recommend(
        &self,
        drm_systems: &[DrmSystem],
        codec: CodecClass,
        coverage: EncryptionCoverage,
    ) -> Result<CencScheme> {
        // FairPlay always requires cbcs.
        if drm_systems.contains(&DrmSystem::FairPlay) {
            return Ok(CencScheme::Cbcs);
        }

        // For video codecs in pattern mode, prefer cbcs for broader support.
        let is_video = matches!(
            codec,
            CodecClass::Avc | CodecClass::Hevc | CodecClass::Av1 | CodecClass::Vpx
        );

        let candidate = match (coverage, is_video) {
            (EncryptionCoverage::Pattern, true) => CencScheme::Cbcs,
            (EncryptionCoverage::Full, _) => CencScheme::Cenc,
            (EncryptionCoverage::Pattern, false) => CencScheme::Cenc,
        };

        // Validate all DRM systems support the candidate.
        for &drm in drm_systems {
            if !candidate.supported_by(drm) {
                return Err(DrmError::UnsupportedDrmSystem(format!(
                    "CENC scheme '{}' is not supported by {}",
                    candidate, drm
                )));
            }
        }

        Ok(candidate)
    }

    /// Return all schemes supported by every DRM system in `drm_systems`.
    pub fn compatible_schemes(&self, drm_systems: &[DrmSystem]) -> Vec<CencScheme> {
        [CencScheme::Cenc, CencScheme::Cbc1, CencScheme::Cens, CencScheme::Cbcs]
            .iter()
            .copied()
            .filter(|&s| drm_systems.iter().all(|&d| s.supported_by(d)))
            .collect()
    }

    /// Validate that `iv` has the correct length for `scheme`.
    pub fn validate_iv(&self, scheme: CencScheme, iv: &[u8]) -> Result<()> {
        let expected = scheme.iv_size_bytes();
        if iv.len() != expected {
            return Err(DrmError::InvalidIv(format!(
                "Scheme '{}' requires a {}-byte IV, got {} bytes",
                scheme,
                expected,
                iv.len()
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── CencScheme basic API ────────────────────────────────────────────────

    #[test]
    fn test_fourcc_values() {
        assert_eq!(CencScheme::Cenc.fourcc(), "cenc");
        assert_eq!(CencScheme::Cbc1.fourcc(), "cbc1");
        assert_eq!(CencScheme::Cens.fourcc(), "cens");
        assert_eq!(CencScheme::Cbcs.fourcc(), "cbcs");
    }

    #[test]
    fn test_from_fourcc_valid() {
        assert_eq!(CencScheme::from_fourcc("cenc").expect("cenc"), CencScheme::Cenc);
        assert_eq!(CencScheme::from_fourcc("CBCS").expect("cbcs"), CencScheme::Cbcs);
        assert_eq!(CencScheme::from_fourcc("CbC1").expect("cbc1"), CencScheme::Cbc1);
    }

    #[test]
    fn test_from_fourcc_invalid() {
        assert!(CencScheme::from_fourcc("mpeg").is_err());
        assert!(CencScheme::from_fourcc("").is_err());
    }

    #[test]
    fn test_cipher_modes() {
        assert_eq!(CencScheme::Cenc.cipher_mode(), CipherMode::AesCtr);
        assert_eq!(CencScheme::Cens.cipher_mode(), CipherMode::AesCtr);
        assert_eq!(CencScheme::Cbc1.cipher_mode(), CipherMode::AesCbc);
        assert_eq!(CencScheme::Cbcs.cipher_mode(), CipherMode::AesCbc);
    }

    #[test]
    fn test_iv_sizes() {
        assert_eq!(CencScheme::Cenc.iv_size_bytes(), 8);
        assert_eq!(CencScheme::Cens.iv_size_bytes(), 8);
        assert_eq!(CencScheme::Cbc1.iv_size_bytes(), 16);
        assert_eq!(CencScheme::Cbcs.iv_size_bytes(), 16);
    }

    #[test]
    fn test_uses_pattern() {
        assert!(!CencScheme::Cenc.uses_pattern());
        assert!(!CencScheme::Cbc1.uses_pattern());
        assert!(CencScheme::Cens.uses_pattern());
        assert!(CencScheme::Cbcs.uses_pattern());
    }

    #[test]
    fn test_fairplay_only_supports_cbcs() {
        assert!(!CencScheme::Cenc.supported_by(DrmSystem::FairPlay));
        assert!(!CencScheme::Cbc1.supported_by(DrmSystem::FairPlay));
        assert!(!CencScheme::Cens.supported_by(DrmSystem::FairPlay));
        assert!(CencScheme::Cbcs.supported_by(DrmSystem::FairPlay));
    }

    #[test]
    fn test_widevine_does_not_support_cbc1() {
        assert!(!CencScheme::Cbc1.supported_by(DrmSystem::Widevine));
        assert!(CencScheme::Cenc.supported_by(DrmSystem::Widevine));
        assert!(CencScheme::Cbcs.supported_by(DrmSystem::Widevine));
    }

    #[test]
    fn test_scheme_display() {
        assert_eq!(CencScheme::Cenc.to_string(), "cenc");
        assert_eq!(CencScheme::Cbcs.to_string(), "cbcs");
    }

    // ── SubsamplePattern ────────────────────────────────────────────────────

    #[test]
    fn test_subsample_pattern_new_valid() {
        let p = SubsamplePattern::new(1, 9).expect("valid pattern");
        assert_eq!(p.crypt_byte_count, 1);
        assert_eq!(p.skip_byte_count, 9);
    }

    #[test]
    fn test_subsample_pattern_new_degenerate() {
        assert!(SubsamplePattern::new(0, 0).is_err());
    }

    #[test]
    fn test_subsample_pattern_cmaf_video() {
        let p = SubsamplePattern::cmaf_video();
        assert_eq!(p.crypt_byte_count, 1);
        assert_eq!(p.skip_byte_count, 9);
    }

    #[test]
    fn test_subsample_pattern_full() {
        let p = SubsamplePattern::full();
        assert_eq!(p.skip_byte_count, 0);
        assert_eq!(p.encrypted_fraction(), 1.0);
    }

    #[test]
    fn test_subsample_pattern_encrypted_fraction() {
        let p = SubsamplePattern::cmaf_video(); // 1:9
        let frac = p.encrypted_fraction();
        assert!((frac - 0.1).abs() < 1e-10, "fraction should be 0.1, got {frac}");
    }

    #[test]
    fn test_subsample_pattern_encrypted_blocks() {
        let p = SubsamplePattern::cmaf_video(); // 1:9 → period = 10
        // 30 blocks → 3 full periods → 3 encrypted blocks
        assert_eq!(p.encrypted_blocks(30), 3);
        // 25 blocks → 2 full periods (20) + 5 remainder → 2 + 1 = 3
        assert_eq!(p.encrypted_blocks(25), 3);
        // 0 blocks → 0 encrypted
        assert_eq!(p.encrypted_blocks(0), 0);
    }

    #[test]
    fn test_subsample_pattern_display() {
        let p = SubsamplePattern::cmaf_video();
        assert_eq!(p.to_string(), "1:9");
    }

    // ── SchemeDescriptor ────────────────────────────────────────────────────

    #[test]
    fn test_scheme_descriptor_cbcs() {
        let desc = SchemeDescriptor::for_scheme(CencScheme::Cbcs);
        assert_eq!(desc.scheme, CencScheme::Cbcs);
        assert!(desc.uses_pattern);
        assert_eq!(desc.iv_size, 16);
        assert!(desc.supported_by.contains(&DrmSystem::FairPlay));
    }

    #[test]
    fn test_scheme_descriptor_cenc() {
        let desc = SchemeDescriptor::for_scheme(CencScheme::Cenc);
        assert!(!desc.uses_pattern);
        assert_eq!(desc.iv_size, 8);
        assert!(!desc.supported_by.contains(&DrmSystem::FairPlay));
    }

    // ── CencSchemeSelector ──────────────────────────────────────────────────

    #[test]
    fn test_selector_fairplay_always_cbcs() {
        let sel = CencSchemeSelector::new();
        let scheme = sel
            .recommend(
                &[DrmSystem::Widevine, DrmSystem::FairPlay],
                CodecClass::Hevc,
                EncryptionCoverage::Full,
            )
            .expect("recommend");
        assert_eq!(scheme, CencScheme::Cbcs);
    }

    #[test]
    fn test_selector_widevine_only_full_gives_cenc() {
        let sel = CencSchemeSelector::new();
        let scheme = sel
            .recommend(&[DrmSystem::Widevine], CodecClass::Avc, EncryptionCoverage::Full)
            .expect("recommend");
        assert_eq!(scheme, CencScheme::Cenc);
    }

    #[test]
    fn test_selector_video_pattern_gives_cbcs() {
        let sel = CencSchemeSelector::new();
        let scheme = sel
            .recommend(
                &[DrmSystem::Widevine, DrmSystem::PlayReady],
                CodecClass::Hevc,
                EncryptionCoverage::Pattern,
            )
            .expect("recommend");
        assert_eq!(scheme, CencScheme::Cbcs);
    }

    #[test]
    fn test_selector_compatible_schemes_fairplay() {
        let sel = CencSchemeSelector::new();
        let schemes = sel.compatible_schemes(&[DrmSystem::FairPlay]);
        assert!(schemes.contains(&CencScheme::Cbcs));
        assert!(!schemes.contains(&CencScheme::Cenc));
    }

    #[test]
    fn test_selector_validate_iv_correct_length() {
        let sel = CencSchemeSelector::new();
        // cenc requires 8-byte IV
        assert!(sel.validate_iv(CencScheme::Cenc, &[0u8; 8]).is_ok());
        // cbcs requires 16-byte IV
        assert!(sel.validate_iv(CencScheme::Cbcs, &[0u8; 16]).is_ok());
    }

    #[test]
    fn test_selector_validate_iv_wrong_length() {
        let sel = CencSchemeSelector::new();
        // cenc with 16-byte IV should fail
        assert!(sel.validate_iv(CencScheme::Cenc, &[0u8; 16]).is_err());
        // cbcs with 8-byte IV should fail
        assert!(sel.validate_iv(CencScheme::Cbcs, &[0u8; 8]).is_err());
    }

    #[test]
    fn test_dash_scheme_uri() {
        assert_eq!(
            CencScheme::Cenc.dash_scheme_uri(),
            "urn:mpeg:dash:mp4protection:2011"
        );
    }
}
