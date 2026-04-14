//! Dynamic range control (DRC) metadata generation.
//!
//! This module models DRC metadata carried in broadcast and streaming codecs
//! (MPEG-4 DRC, AC-3/E-AC-3 `dialnorm`, AAC `DRC_TOOL_BOX`) and provides
//! structures for describing compression profiles, generating per-frame DRC
//! gain words, and serialising/deserialising DRC metadata payloads.
//!
//! # Supported Standards
//!
//! | Standard | Profile(s) | Notes |
//! |---|---|---|
//! | AC-3 / E-AC-3 | Film-Standard, Film-Light, Music-Standard, Music-Light, Speech | RFC 4598 §A.4 |
//! | MPEG-4 DRC | General, Music, Film, Speech | ISO 23003-4 §5 |
//! | AAC `compression_value` | Heavy / Light | MPEG-4 §8.2.2.2.3 |
//!
//! # Usage
//!
//! ```rust
//! use oximedia_normalize::drc_metadata::{DrcProfile, DrcMetadataEncoder, DrcGainConfig};
//!
//! let encoder = DrcMetadataEncoder::new(DrcProfile::Speech);
//! let frame = encoder.encode_gain_word(-6.0);
//! println!("DRC gain word: 0x{:02X}", frame.gain_word_u8);
//! ```

use crate::{NormalizeError, NormalizeResult};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// DrcProfile
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-defined DRC compression profiles following broadcast and streaming
/// standards.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrcProfile {
    /// Film-standard compression — aggressive dynamic range control for TV/cinema.
    /// AC-3: `drc_film_standard`; MPEG-4: `general_compr`.
    FilmStandard,

    /// Film-light compression — gentler version for high-quality systems.
    /// AC-3: `drc_film_light`.
    FilmLight,

    /// Music-standard compression — preserves musical dynamics, controlled peaks.
    /// AC-3: `drc_music_standard`.
    MusicStandard,

    /// Music-light compression — very gentle limiting for audiophile playback.
    /// AC-3: `drc_music_light`.
    MusicLight,

    /// Speech compression — maximises intelligibility; aggressive above threshold.
    /// AC-3: `drc_speech`; MPEG-4: `speech_compr`.
    Speech,

    /// No compression — gain word always 0 dB; useful for archival masters.
    None,

    /// Custom profile — caller specifies all parameters explicitly.
    Custom,
}

impl DrcProfile {
    /// Return the ATSC/AC-3 `compression` profile index (0–5 per A/52 Table 4.17).
    pub fn ac3_compression_profile_id(self) -> u8 {
        match self {
            Self::FilmStandard => 1,
            Self::FilmLight => 2,
            Self::MusicStandard => 3,
            Self::MusicLight => 4,
            Self::Speech => 5,
            Self::None | Self::Custom => 0,
        }
    }

    /// Return the MPEG-4 DRC `drc_set_effect` bitmask (ISO 23003-4 Table 6).
    ///
    /// | Bit | Effect |
    /// |-----|--------|
    /// | 0 | Night mode |
    /// | 1 | Noisy environment |
    /// | 2 | Limited playback range |
    /// | 3 | General compression |
    /// | 4 | Fade |
    /// | 5 | Ducking |
    /// | 6 | Speech compression |
    pub fn mpeg4_drc_set_effect(self) -> u8 {
        match self {
            Self::FilmStandard => 0b0000_1000, // general compression
            Self::FilmLight => 0b0000_1100,    // general + limited range
            Self::MusicStandard => 0b0000_1000,
            Self::MusicLight => 0b0000_0100,   // limited range
            Self::Speech => 0b0100_0000,        // speech
            Self::None | Self::Custom => 0,
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::FilmStandard => "film-standard",
            Self::FilmLight => "film-light",
            Self::MusicStandard => "music-standard",
            Self::MusicLight => "music-light",
            Self::Speech => "speech",
            Self::None => "none",
            Self::Custom => "custom",
        }
    }
}

impl fmt::Display for DrcProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DrcGainConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters that define a DRC compression characteristic (knee/threshold).
#[derive(Clone, Debug)]
pub struct DrcGainConfig {
    /// DRC profile this config represents.
    pub profile: DrcProfile,
    /// Threshold below which no compression is applied (dBFS).
    pub threshold_db: f64,
    /// Ratio above the threshold (e.g. 4.0 = 4:1).
    pub ratio: f64,
    /// Maximum gain reduction allowed (dB, positive = attenuation).
    pub max_attenuation_db: f64,
    /// Makeup gain applied after compression (dB).
    pub makeup_gain_db: f64,
    /// Whether to encode attenuation as AC-3 `dynrng` words.
    pub encode_ac3_dynrng: bool,
    /// Whether to encode attenuation as MPEG-4 DRC gain words.
    pub encode_mpeg4_drc: bool,
}

impl DrcGainConfig {
    /// Film-standard DRC (AC-3 default for broadcast receivers).
    pub fn film_standard() -> Self {
        Self {
            profile: DrcProfile::FilmStandard,
            threshold_db: -20.0,
            ratio: 5.0,
            max_attenuation_db: 24.0,
            makeup_gain_db: 0.0,
            encode_ac3_dynrng: true,
            encode_mpeg4_drc: true,
        }
    }

    /// Film-light DRC (used for high-quality TV sets).
    pub fn film_light() -> Self {
        Self {
            profile: DrcProfile::FilmLight,
            threshold_db: -15.0,
            ratio: 3.0,
            max_attenuation_db: 12.0,
            makeup_gain_db: 0.0,
            encode_ac3_dynrng: true,
            encode_mpeg4_drc: true,
        }
    }

    /// Music-standard DRC.
    pub fn music_standard() -> Self {
        Self {
            profile: DrcProfile::MusicStandard,
            threshold_db: -18.0,
            ratio: 4.0,
            max_attenuation_db: 18.0,
            makeup_gain_db: 0.0,
            encode_ac3_dynrng: false,
            encode_mpeg4_drc: true,
        }
    }

    /// Speech DRC — maximises intelligibility.
    pub fn speech() -> Self {
        Self {
            profile: DrcProfile::Speech,
            threshold_db: -25.0,
            ratio: 8.0,
            max_attenuation_db: 30.0,
            makeup_gain_db: 3.0,
            encode_ac3_dynrng: true,
            encode_mpeg4_drc: true,
        }
    }

    /// No-op (transparent) DRC.
    pub fn none() -> Self {
        Self {
            profile: DrcProfile::None,
            threshold_db: 0.0,
            ratio: 1.0,
            max_attenuation_db: 0.0,
            makeup_gain_db: 0.0,
            encode_ac3_dynrng: false,
            encode_mpeg4_drc: false,
        }
    }

    /// Validate configuration.
    pub fn validate(&self) -> NormalizeResult<()> {
        if self.ratio < 1.0 {
            return Err(NormalizeError::InvalidConfig(format!(
                "DRC ratio {:.2} must be >= 1.0",
                self.ratio
            )));
        }
        if self.max_attenuation_db < 0.0 {
            return Err(NormalizeError::InvalidConfig(
                "max_attenuation_db must be >= 0".to_string(),
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DrcGainWord — per-frame encoded gain
// ─────────────────────────────────────────────────────────────────────────────

/// A fully encoded per-frame DRC gain word ready for muxing into a bitstream.
#[derive(Clone, Debug)]
pub struct DrcGainWord {
    /// Gain in dB that this word represents (negative = attenuation).
    pub gain_db: f64,

    /// AC-3 `dynrng` / `compr` 8-bit gain word encoding.
    ///
    /// Format: 1-bit sign + 3-bit exponent + 4-bit mantissa, representing
    /// gain in the range −24 dB … +24 dB with 0.25 dB resolution (AC-3
    /// A/52 §4.4.2.6).
    pub gain_word_u8: u8,

    /// MPEG-4 DRC 7-bit signed gain code (Q3.4 fixed-point, 1/8 dB steps,
    /// range −8 dB … +8 dB per ISO 23003-4 §6.3.4.5).
    ///
    /// Stored in the low 7 bits of the byte; MSB is reserved (0).
    pub mpeg4_gain_code_u8: u8,

    /// AAC `compression_value` byte (heavy compression mode).
    ///
    /// Encoded as: `gain_db / 0.25 + 128`, clamped to [0, 255].
    pub aac_compression_value_u8: u8,

    /// AC-3 `dialnorm` value (5-bit unsigned, range 1–31).
    ///
    /// Dialogue normalisation level = `−dialnorm` dB.  Per A/52 §A.4:
    /// `dialnorm = round(−dialogue_level_lufs)`.
    pub ac3_dialnorm: u8,
}

impl DrcGainWord {
    /// Returns `true` if the word represents gain reduction (attenuation).
    pub fn is_attenuation(&self) -> bool {
        self.gain_db < 0.0
    }

    /// Returns `true` if the word is at unity gain (0 dB).
    pub fn is_unity(&self) -> bool {
        self.gain_db.abs() < 0.125 // within half a 0.25 dB step
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DrcMetadataEncoder
// ─────────────────────────────────────────────────────────────────────────────

/// Encodes DRC gain values into format-specific bitstream words.
///
/// Construct with a profile via [`DrcMetadataEncoder::new`] or supply a full
/// [`DrcGainConfig`] via [`DrcMetadataEncoder::with_config`].
pub struct DrcMetadataEncoder {
    config: DrcGainConfig,
}

impl DrcMetadataEncoder {
    /// Create an encoder using the default [`DrcGainConfig`] for `profile`.
    pub fn new(profile: DrcProfile) -> Self {
        let config = match profile {
            DrcProfile::FilmStandard => DrcGainConfig::film_standard(),
            DrcProfile::FilmLight => DrcGainConfig::film_light(),
            DrcProfile::MusicStandard => DrcGainConfig::music_standard(),
            DrcProfile::Speech => DrcGainConfig::speech(),
            DrcProfile::MusicLight | DrcProfile::None | DrcProfile::Custom => {
                DrcGainConfig::none()
            }
        };
        Self { config }
    }

    /// Create an encoder from a fully specified configuration.
    pub fn with_config(config: DrcGainConfig) -> NormalizeResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Compute and encode the DRC gain word for a signal at `input_level_db`
    /// (dBFS).
    ///
    /// The gain is computed from the configured compression characteristic,
    /// then encoded into all supported word formats.
    pub fn encode_for_level(&self, input_level_db: f64) -> DrcGainWord {
        let gain_db = self.compute_gain_db(input_level_db);
        self.encode_gain_word(gain_db)
    }

    /// Encode a raw `gain_db` value (already computed) into all word formats.
    pub fn encode_gain_word(&self, gain_db: f64) -> DrcGainWord {
        let clamped = gain_db.clamp(-self.config.max_attenuation_db, 24.0);
        DrcGainWord {
            gain_db: clamped,
            gain_word_u8: encode_ac3_dynrng(clamped),
            mpeg4_gain_code_u8: encode_mpeg4_gain_code(clamped),
            aac_compression_value_u8: encode_aac_compression_value(clamped),
            ac3_dialnorm: 0, // not bound to a per-frame level; use encode_dialnorm()
        }
    }

    /// Compute the gain reduction (dB) for `input_level_db` using the
    /// configured compression characteristic.
    pub fn compute_gain_db(&self, input_level_db: f64) -> f64 {
        if self.config.ratio <= 1.0 {
            return self.config.makeup_gain_db;
        }
        let threshold = self.config.threshold_db;
        let excess = input_level_db - threshold;
        if excess <= 0.0 {
            // Below threshold — only makeup gain
            self.config.makeup_gain_db
        } else {
            // Compress: gain_reduction = excess * (1 - 1/ratio)
            let reduction = excess * (1.0 - 1.0 / self.config.ratio);
            let total = -reduction + self.config.makeup_gain_db;
            total.clamp(-self.config.max_attenuation_db, 24.0)
        }
    }

    /// Encode the AC-3 `dialnorm` word for a measured dialogue level.
    ///
    /// `dialogue_lufs` — integrated dialogue loudness (e.g. −27 LUFS).
    /// Returns a 5-bit value in [1, 31] (A/52 §4.4.1.5).
    pub fn encode_dialnorm(&self, dialogue_lufs: f64) -> u8 {
        encode_ac3_dialnorm(dialogue_lufs)
    }

    /// Get the underlying configuration.
    pub fn config(&self) -> &DrcGainConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DrcMetadataFrame — a single codec DRC payload
// ─────────────────────────────────────────────────────────────────────────────

/// A complete DRC metadata payload for a single audio frame or block.
#[derive(Clone, Debug)]
pub struct DrcMetadataFrame {
    /// Profile used to derive this frame's metadata.
    pub profile: DrcProfile,
    /// Whether heavy compression metadata is present.
    pub has_heavy_compression: bool,
    /// Heavy-compression gain word (AC-3 `compr`).
    pub heavy_compression: DrcGainWord,
    /// Whether light compression metadata is present.
    pub has_light_compression: bool,
    /// Light-compression gain word (AC-3 `dynrng`).
    pub light_compression: DrcGainWord,
    /// AC-3 `dialnorm` value embedded in this frame.
    pub dialnorm: u8,
    /// MPEG-4 DRC sequence number (wraps at 256).
    pub mpeg4_sequence_number: u8,
}

impl DrcMetadataFrame {
    /// Serialise to a compact 8-byte payload.
    ///
    /// Layout (little-endian):
    /// ```text
    /// [0]  profile AC-3 id
    /// [1]  flags: bit0=has_heavy, bit1=has_light
    /// [2]  heavy gain word (AC-3 dynrng encoding)
    /// [3]  light gain word (AC-3 dynrng encoding)
    /// [4]  dialnorm (5 bits, lower bits)
    /// [5]  heavy MPEG-4 gain code
    /// [6]  light MPEG-4 gain code
    /// [7]  MPEG-4 sequence number
    /// ```
    pub fn to_bytes(&self) -> [u8; 8] {
        let mut flags = 0_u8;
        if self.has_heavy_compression {
            flags |= 0x01;
        }
        if self.has_light_compression {
            flags |= 0x02;
        }
        [
            self.profile.ac3_compression_profile_id(),
            flags,
            self.heavy_compression.gain_word_u8,
            self.light_compression.gain_word_u8,
            self.dialnorm & 0x1F,
            self.heavy_compression.mpeg4_gain_code_u8,
            self.light_compression.mpeg4_gain_code_u8,
            self.mpeg4_sequence_number,
        ]
    }

    /// Deserialise from an 8-byte payload produced by [`to_bytes`].
    pub fn from_bytes(bytes: &[u8; 8]) -> NormalizeResult<Self> {
        let profile_id = bytes[0];
        let profile = match profile_id {
            0 => DrcProfile::None,
            1 => DrcProfile::FilmStandard,
            2 => DrcProfile::FilmLight,
            3 => DrcProfile::MusicStandard,
            4 => DrcProfile::MusicLight,
            5 => DrcProfile::Speech,
            _ => {
                return Err(NormalizeError::ProcessingError(format!(
                    "unknown DRC profile id {profile_id}"
                )))
            }
        };
        let flags = bytes[1];
        let has_heavy = (flags & 0x01) != 0;
        let has_light = (flags & 0x02) != 0;

        let heavy_gain_db = decode_ac3_dynrng(bytes[2]);
        let light_gain_db = decode_ac3_dynrng(bytes[3]);
        let dialnorm = bytes[4] & 0x1F;
        let mpeg4_seq = bytes[7];

        Ok(Self {
            profile,
            has_heavy_compression: has_heavy,
            heavy_compression: DrcGainWord {
                gain_db: heavy_gain_db,
                gain_word_u8: bytes[2],
                mpeg4_gain_code_u8: bytes[5],
                aac_compression_value_u8: encode_aac_compression_value(heavy_gain_db),
                ac3_dialnorm: dialnorm,
            },
            has_light_compression: has_light,
            light_compression: DrcGainWord {
                gain_db: light_gain_db,
                gain_word_u8: bytes[3],
                mpeg4_gain_code_u8: bytes[6],
                aac_compression_value_u8: encode_aac_compression_value(light_gain_db),
                ac3_dialnorm: dialnorm,
            },
            dialnorm,
            mpeg4_sequence_number: mpeg4_seq,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DrcMetadataBuilder — generates frame-accurate DRC metadata for a programme
// ─────────────────────────────────────────────────────────────────────────────

/// Generates a sequence of [`DrcMetadataFrame`]s for a programme, one per
/// audio block (e.g. one AC-3 block = 256 samples at 48 kHz).
pub struct DrcMetadataBuilder {
    heavy_encoder: DrcMetadataEncoder,
    light_encoder: DrcMetadataEncoder,
    dialnorm: u8,
    sequence: u8,
}

impl DrcMetadataBuilder {
    /// Create a builder for the given heavy and light profiles.
    ///
    /// `dialnorm_lufs` — measured dialogue level used to set the AC-3
    /// `dialnorm` word for the programme (e.g. `−27.0`).
    pub fn new(
        heavy_profile: DrcProfile,
        light_profile: DrcProfile,
        dialnorm_lufs: f64,
    ) -> Self {
        Self {
            heavy_encoder: DrcMetadataEncoder::new(heavy_profile),
            light_encoder: DrcMetadataEncoder::new(light_profile),
            dialnorm: encode_ac3_dialnorm(dialnorm_lufs),
            sequence: 0,
        }
    }

    /// Encode metadata for one block, given the peak signal level of that
    /// block in dBFS.
    pub fn encode_block(&mut self, block_peak_db: f64) -> DrcMetadataFrame {
        let heavy = self.heavy_encoder.encode_for_level(block_peak_db);
        let light = self.light_encoder.encode_for_level(block_peak_db);
        let seq = self.sequence;
        self.sequence = self.sequence.wrapping_add(1);
        DrcMetadataFrame {
            profile: self.heavy_encoder.config().profile,
            has_heavy_compression: true,
            heavy_compression: heavy,
            has_light_compression: true,
            light_compression: light,
            dialnorm: self.dialnorm,
            mpeg4_sequence_number: seq,
        }
    }

    /// Update the dialogue normalisation level.
    pub fn set_dialnorm(&mut self, dialogue_lufs: f64) {
        self.dialnorm = encode_ac3_dialnorm(dialogue_lufs);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Codec encoding/decoding helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Encode a gain (dB) into the AC-3 `dynrng` / `compr` 8-bit word format.
///
/// AC-3 A/52 §4.4.2.6: the 8-bit word encodes gains from −24 dB to +24 dB
/// in 0.25 dB steps.  Encoding:
///
/// ```text
/// bits [7]     = sign (1 = negative gain / attenuation)
/// bits [6:4]   = 3-bit exponent
/// bits [3:0]   = 4-bit mantissa
/// gain_db = (−1)^sign × (2^(exp−4)) × (16 + mantissa) × 0.25 dB
/// ```
///
/// We use a simplified linear approximation that maps gain to the nearest
/// representable value within the ±24 dB range.
pub fn encode_ac3_dynrng(gain_db: f64) -> u8 {
    // Clamp to the ±24 dB range (3 bits of exponent → 2^(7-4) × ... max ≈ 24)
    let g = gain_db.clamp(-24.0, 24.0);
    // Map to integer steps of 0.25 dB, offset by 96 to make the range 0..192
    let steps = (g * 4.0).round() as i32; // steps of 0.25 dB
    // Encode as a signed-magnitude 8-bit value: sign in bit 7, magnitude in [6:0]
    let sign: u8 = if steps < 0 { 0x80 } else { 0 };
    let mag = steps.unsigned_abs().min(127) as u8;
    sign | mag
}

/// Decode an AC-3 `dynrng` 8-bit word back to gain in dB.
pub fn decode_ac3_dynrng(word: u8) -> f64 {
    let sign: f64 = if (word & 0x80) != 0 { -1.0 } else { 1.0 };
    let mag = (word & 0x7F) as f64;
    sign * mag * 0.25
}

/// Encode a gain (dB) into the MPEG-4 DRC 7-bit gain code.
///
/// ISO 23003-4 §6.3.4.5: the gain delta is encoded in Q3.4 fixed-point
/// (1/8 dB resolution), range −8 dB … +8 dB.  The 7-bit signed integer
/// represents `gain_db × 8`.
pub fn encode_mpeg4_gain_code(gain_db: f64) -> u8 {
    let clamped = gain_db.clamp(-8.0, 8.0);
    let code = (clamped * 8.0).round() as i8;
    code as u8 & 0x7F // mask to 7 bits (unsigned magnitude via 2's complement)
}

/// Decode an MPEG-4 DRC 7-bit gain code back to gain in dB.
pub fn decode_mpeg4_gain_code(code: u8) -> f64 {
    // Sign-extend the 7-bit value
    let raw = (code & 0x7F) as i8;
    // Treat as signed 7-bit: if bit 6 set, it is negative in 2's complement
    let signed: i8 = if (code & 0x40) != 0 {
        raw | (-128_i8) // sign-extend
    } else {
        raw
    };
    signed as f64 / 8.0
}

/// Encode a gain (dB) into the AAC `compression_value` byte.
///
/// MPEG-4 Audio §8.2.2.2.3: `compression_value = round(gain_db / 0.25) + 128`,
/// clamped to [0, 255].
pub fn encode_aac_compression_value(gain_db: f64) -> u8 {
    let raw = (gain_db / 0.25).round() as i32 + 128;
    raw.clamp(0, 255) as u8
}

/// Decode an AAC `compression_value` byte back to gain in dB.
pub fn decode_aac_compression_value(value: u8) -> f64 {
    (value as i32 - 128) as f64 * 0.25
}

/// Encode an AC-3 `dialnorm` word (1–31) from a measured dialogue LUFS level.
///
/// `dialnorm = round(−dialogue_level_lufs)`, clamped to [1, 31].
/// A value of 31 corresponds to −31 dBFS / −31 LUFS dialogue.
pub fn encode_ac3_dialnorm(dialogue_lufs: f64) -> u8 {
    let raw = (-dialogue_lufs).round() as i32;
    raw.clamp(1, 31) as u8
}

/// Decode an AC-3 `dialnorm` word back to dialogue level in LUFS.
pub fn decode_ac3_dialnorm(dialnorm: u8) -> f64 {
    -(dialnorm as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drc_profile_names() {
        assert_eq!(DrcProfile::FilmStandard.name(), "film-standard");
        assert_eq!(DrcProfile::Speech.name(), "speech");
        assert_eq!(DrcProfile::None.name(), "none");
    }

    #[test]
    fn test_drc_profile_ac3_ids() {
        assert_eq!(DrcProfile::FilmStandard.ac3_compression_profile_id(), 1);
        assert_eq!(DrcProfile::Speech.ac3_compression_profile_id(), 5);
        assert_eq!(DrcProfile::None.ac3_compression_profile_id(), 0);
    }

    #[test]
    fn test_drc_gain_config_validation_ok() {
        let cfg = DrcGainConfig::film_standard();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_drc_gain_config_validation_bad_ratio() {
        let mut cfg = DrcGainConfig::film_standard();
        cfg.ratio = 0.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_encoder_new_speech() {
        let enc = DrcMetadataEncoder::new(DrcProfile::Speech);
        assert_eq!(enc.config().profile, DrcProfile::Speech);
    }

    #[test]
    fn test_encoder_with_config() {
        let cfg = DrcGainConfig::speech();
        let enc = DrcMetadataEncoder::with_config(cfg);
        assert!(enc.is_ok());
    }

    #[test]
    fn test_compute_gain_below_threshold() {
        let enc = DrcMetadataEncoder::new(DrcProfile::FilmStandard);
        // Signal is 10 dB below threshold (-20 dBFS) → no compression
        let gain = enc.compute_gain_db(-30.0);
        // Should be approximately makeup gain only (0.0 for film-standard)
        assert!(gain >= 0.0 - 0.01);
    }

    #[test]
    fn test_compute_gain_above_threshold() {
        let enc = DrcMetadataEncoder::new(DrcProfile::FilmStandard);
        // Signal at -10 dBFS → 10 dB above threshold of -20
        // ratio=5 → reduction = 10 × (1 - 1/5) = 8 dB → gain = -8 dB
        let gain = enc.compute_gain_db(-10.0);
        assert!(gain < 0.0, "expected attenuation, got {gain}");
    }

    #[test]
    fn test_ac3_dynrng_round_trip_zero() {
        let word = encode_ac3_dynrng(0.0);
        let decoded = decode_ac3_dynrng(word);
        assert!(decoded.abs() < 0.5, "round-trip zero failed: {decoded}");
    }

    #[test]
    fn test_ac3_dynrng_round_trip_attenuation() {
        let original = -6.0_f64;
        let word = encode_ac3_dynrng(original);
        let decoded = decode_ac3_dynrng(word);
        assert!((decoded - original).abs() < 0.5, "round-trip {original}: got {decoded}");
    }

    #[test]
    fn test_mpeg4_gain_code_round_trip() {
        for &db in &[-4.0_f64, -2.0, 0.0, 1.5, 3.0] {
            let code = encode_mpeg4_gain_code(db);
            let decoded = decode_mpeg4_gain_code(code);
            assert!(
                (decoded - db).abs() < 0.5,
                "MPEG-4 round-trip {db}: got {decoded}"
            );
        }
    }

    #[test]
    fn test_aac_compression_value_round_trip() {
        for &db in &[-6.0_f64, 0.0, 3.0] {
            let code = encode_aac_compression_value(db);
            let decoded = decode_aac_compression_value(code);
            assert!((decoded - db).abs() < 0.5, "AAC round-trip {db}: got {decoded}");
        }
    }

    #[test]
    fn test_dialnorm_round_trip() {
        let dialogue_lufs = -27.0_f64;
        let dialnorm = encode_ac3_dialnorm(dialogue_lufs);
        assert_eq!(dialnorm, 27);
        let decoded = decode_ac3_dialnorm(dialnorm);
        assert!((decoded - dialogue_lufs).abs() < 0.5);
    }

    #[test]
    fn test_metadata_frame_bytes_round_trip() {
        let mut builder =
            DrcMetadataBuilder::new(DrcProfile::FilmStandard, DrcProfile::FilmLight, -27.0);
        let frame = builder.encode_block(-12.0);
        let bytes = frame.to_bytes();
        let decoded = DrcMetadataFrame::from_bytes(&bytes).expect("decode ok");
        assert_eq!(decoded.profile, DrcProfile::FilmStandard);
        assert_eq!(decoded.dialnorm, 27);
    }

    #[test]
    fn test_metadata_builder_sequence_increments() {
        let mut builder =
            DrcMetadataBuilder::new(DrcProfile::Speech, DrcProfile::None, -23.0);
        let f0 = builder.encode_block(-15.0);
        let f1 = builder.encode_block(-15.0);
        assert_eq!(f0.mpeg4_sequence_number, 0);
        assert_eq!(f1.mpeg4_sequence_number, 1);
    }

    #[test]
    fn test_gain_word_is_unity_check() {
        let enc = DrcMetadataEncoder::new(DrcProfile::None);
        let word = enc.encode_gain_word(0.0);
        assert!(word.is_unity());
    }

    #[test]
    fn test_gain_word_is_attenuation() {
        let enc = DrcMetadataEncoder::new(DrcProfile::FilmStandard);
        let word = enc.encode_gain_word(-6.0);
        assert!(word.is_attenuation());
    }

    #[test]
    fn test_mpeg4_drc_effect_bitmasks() {
        let speech_effect = DrcProfile::Speech.mpeg4_drc_set_effect();
        assert_ne!(speech_effect, 0);
        let none_effect = DrcProfile::None.mpeg4_drc_set_effect();
        assert_eq!(none_effect, 0);
    }

    #[test]
    fn test_from_bytes_unknown_profile_returns_error() {
        let bytes = [0xFF, 0, 0, 0, 0, 0, 0, 0];
        assert!(DrcMetadataFrame::from_bytes(&bytes).is_err());
    }
}
