//! ABR encode-ladder validation against CMAF, HLS, and DASH specifications.
//!
//! An *encode ladder* (or ABR ladder) is the ordered set of quality rungs
//! used in adaptive-bitrate streaming.  Each rung is a `(resolution, bitrate,
//! codec)` triplet.  Poorly designed ladders can cause:
//!
//! - Bitrate crossovers (lower resolution at higher bitrate than a taller rung).
//! - Gap violations (jumps too large for smooth adaptive switching).
//! - Resolution/bitrate mismatches (bitrate too low to fill the target
//!   resolution, wasting pixels).
//! - Non-compliant CMAF / HLS / DASH segment durations.
//!
//! This module provides [`LadderValidator`] which runs all checks on a
//! [`EncodeLadder`] and returns a structured [`LadderReport`].
//!
//! # Supported specifications
//!
//! | Spec | Version checked |
//! |------|----------------|
//! | Apple HLS | [RFC 8216 §4.3.4.2](https://datatracker.ietf.org/doc/html/rfc8216) |
//! | MPEG-DASH | ISO/IEC 23009-1:2022 §5 |
//! | CMAF | ISO/IEC 23000-19:2020 §7 |
//! | Apple low-latency HLS | draft-pantos-hls-rfc8216bis-15 |
//!
//! # Example
//!
//! ```rust
//! use oximedia_transcode::encode_ladder_validator::{
//!     EncodeLadder, LadderRung, LadderSpec, LadderValidator,
//! };
//!
//! let ladder = EncodeLadder::new(vec![
//!     LadderRung::new(1920, 1080, 4_500_000, 30.0, "vp9"),
//!     LadderRung::new(1280, 720,  2_500_000, 30.0, "vp9"),
//!     LadderRung::new(854,  480,  1_000_000, 30.0, "vp9"),
//!     LadderRung::new(640,  360,    500_000, 30.0, "vp9"),
//! ]);
//!
//! let report = LadderValidator::new(LadderSpec::Hls).validate(&ladder);
//! assert!(report.is_ok(), "validation errors: {:?}", report.errors());
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use std::collections::HashSet;
use std::fmt;

// ─── LadderRung ───────────────────────────────────────────────────────────────

/// A single quality rung in an adaptive-bitrate encode ladder.
#[derive(Debug, Clone, PartialEq)]
pub struct LadderRung {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Target video bitrate in bits per second.
    pub bitrate_bps: u64,
    /// Frame rate in frames per second.
    pub frame_rate: f64,
    /// Video codec identifier (e.g. "vp9", "av1", "h264").
    pub codec: String,
    /// Audio bitrate in bits per second (0 = no audio / video-only rung).
    pub audio_bps: u64,
    /// Segment duration in seconds for streaming (0.0 = not streaming).
    pub segment_duration_s: f64,
}

impl LadderRung {
    /// Creates a rung with the given dimensions, bitrate, frame rate and codec.
    ///
    /// `audio_bps` and `segment_duration_s` default to `0`.
    #[must_use]
    pub fn new(width: u32, height: u32, bitrate_bps: u64, frame_rate: f64, codec: &str) -> Self {
        Self {
            width,
            height,
            bitrate_bps,
            frame_rate,
            codec: codec.to_owned(),
            audio_bps: 0,
            segment_duration_s: 0.0,
        }
    }

    /// Sets the audio bitrate for this rung.
    #[must_use]
    pub fn with_audio(mut self, audio_bps: u64) -> Self {
        self.audio_bps = audio_bps;
        self
    }

    /// Sets the segment duration.
    #[must_use]
    pub fn with_segment_duration(mut self, secs: f64) -> Self {
        self.segment_duration_s = secs;
        self
    }

    /// Returns the total pixels per frame.
    #[must_use]
    pub fn pixels(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Returns the bits-per-pixel metric for this rung.
    ///
    /// Higher bpp → over-provisioned for the resolution;
    /// lower bpp → under-provisioned.
    #[must_use]
    pub fn bits_per_pixel(&self) -> f64 {
        if self.pixels() == 0 || self.frame_rate <= 0.0 {
            return 0.0;
        }
        self.bitrate_bps as f64 / (self.pixels() as f64 * self.frame_rate)
    }

    /// Returns the aspect ratio as a `(width, height)` pair reduced to lowest terms.
    #[must_use]
    pub fn aspect_ratio(&self) -> (u32, u32) {
        let g = gcd(self.width, self.height);
        if g == 0 {
            return (0, 0);
        }
        (self.width / g, self.height / g)
    }
}

impl fmt::Display for LadderRung {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}×{}@{:.0}fps {:.1}Mbps {}",
            self.width,
            self.height,
            self.frame_rate,
            self.bitrate_bps as f64 / 1_000_000.0,
            self.codec
        )
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

// ─── EncodeLadder ─────────────────────────────────────────────────────────────

/// An ordered set of [`LadderRung`]s from highest to lowest quality.
///
/// Rungs should be inserted in descending order (highest bitrate first),
/// though [`LadderValidator`] will catch ordering violations.
#[derive(Debug, Clone)]
pub struct EncodeLadder {
    /// Ordered rungs (highest quality first by convention).
    pub rungs: Vec<LadderRung>,
}

impl EncodeLadder {
    /// Creates a ladder from a `Vec` of rungs.
    #[must_use]
    pub fn new(rungs: Vec<LadderRung>) -> Self {
        Self { rungs }
    }

    /// Returns the number of rungs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rungs.len()
    }

    /// Returns `true` if the ladder has no rungs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rungs.is_empty()
    }

    /// Sorts rungs from highest to lowest bitrate in place.
    pub fn sort_descending(&mut self) {
        self.rungs.sort_by(|a, b| {
            b.bitrate_bps.cmp(&a.bitrate_bps)
        });
    }

    /// Returns the codec set used across all rungs.
    #[must_use]
    pub fn codec_set(&self) -> HashSet<&str> {
        self.rungs.iter().map(|r| r.codec.as_str()).collect()
    }
}

// ─── LadderSpec ───────────────────────────────────────────────────────────────

/// Streaming specification against which the ladder is validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LadderSpec {
    /// Apple HTTP Live Streaming (RFC 8216).
    Hls,
    /// MPEG-DASH (ISO/IEC 23009-1).
    Dash,
    /// Common Media Application Format (ISO/IEC 23000-19).
    Cmaf,
    /// Low-latency HLS (LL-HLS, draft-pantos-hls-rfc8216bis).
    LlHls,
    /// Custom / no specific spec; only generic checks are applied.
    Generic,
}

impl LadderSpec {
    /// Returns the recommended segment duration range `(min_s, max_s)` for
    /// this specification, or `None` for [`LadderSpec::Generic`].
    #[must_use]
    pub fn segment_duration_range(self) -> Option<(f64, f64)> {
        match self {
            Self::Hls => Some((2.0, 10.0)),
            Self::Dash => Some((1.0, 10.0)),
            Self::Cmaf => Some((1.0, 6.0)),
            Self::LlHls => Some((0.5, 2.0)),
            Self::Generic => None,
        }
    }

    /// Returns the maximum number of rungs recommended for this spec.
    #[must_use]
    pub fn max_rungs(self) -> usize {
        match self {
            Self::LlHls => 6,
            Self::Hls | Self::Dash | Self::Cmaf => 8,
            Self::Generic => usize::MAX,
        }
    }

    /// Returns `true` if the codec is allowed by this specification.
    #[must_use]
    pub fn allows_codec(self, codec: &str) -> bool {
        match self {
            // HLS requires H.264 or HEVC; patent-free codecs via fMP4.
            Self::Hls => matches!(codec, "h264" | "h265" | "hevc" | "vp9" | "av1"),
            // DASH / CMAF: same.
            Self::Dash | Self::Cmaf | Self::LlHls => {
                matches!(codec, "h264" | "h265" | "hevc" | "vp9" | "av1")
            }
            Self::Generic => true,
        }
    }
}

// ─── Validation findings ──────────────────────────────────────────────────────

/// The severity of a ladder finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FindingSeverity {
    /// Informational; does not affect compliance.
    Info,
    /// A sub-optimal configuration that might affect quality.
    Warning,
    /// A violation that makes the ladder non-compliant.
    Error,
}

/// A single finding produced by [`LadderValidator`].
#[derive(Debug, Clone)]
pub struct LadderFinding {
    /// Severity of this finding.
    pub severity: FindingSeverity,
    /// Optional rung index (0-indexed, `None` = applies to whole ladder).
    pub rung_index: Option<usize>,
    /// Human-readable description of the finding.
    pub message: String,
}

impl LadderFinding {
    fn error(rung: Option<usize>, msg: impl Into<String>) -> Self {
        Self {
            severity: FindingSeverity::Error,
            rung_index: rung,
            message: msg.into(),
        }
    }

    fn warning(rung: Option<usize>, msg: impl Into<String>) -> Self {
        Self {
            severity: FindingSeverity::Warning,
            rung_index: rung,
            message: msg.into(),
        }
    }

    fn info(rung: Option<usize>, msg: impl Into<String>) -> Self {
        Self {
            severity: FindingSeverity::Info,
            rung_index: rung,
            message: msg.into(),
        }
    }
}

// ─── LadderReport ─────────────────────────────────────────────────────────────

/// The output of a [`LadderValidator`] run.
#[derive(Debug, Clone)]
pub struct LadderReport {
    /// All findings (errors, warnings, infos) in order of discovery.
    pub findings: Vec<LadderFinding>,
    /// The specification against which the ladder was validated.
    pub spec: LadderSpec,
    /// Number of rungs validated.
    pub rung_count: usize,
}

impl LadderReport {
    /// Returns `true` when there are no error-level findings.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        !self
            .findings
            .iter()
            .any(|f| f.severity == FindingSeverity::Error)
    }

    /// Returns only error-level findings.
    #[must_use]
    pub fn errors(&self) -> Vec<&LadderFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity == FindingSeverity::Error)
            .collect()
    }

    /// Returns only warning-level findings.
    #[must_use]
    pub fn warnings(&self) -> Vec<&LadderFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity == FindingSeverity::Warning)
            .collect()
    }

    /// Returns only info-level findings.
    #[must_use]
    pub fn infos(&self) -> Vec<&LadderFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity == FindingSeverity::Info)
            .collect()
    }

    /// Returns the number of error-level findings.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors().len()
    }
}

// ─── Validator configuration ──────────────────────────────────────────────────

/// Configuration knobs for [`LadderValidator`].
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    /// Minimum bits-per-pixel.  Rungs below this are flagged as a warning.
    /// Default: `0.03` (empirically good lower bound for AV1/VP9 at 480p+).
    pub min_bpp: f64,

    /// Maximum bits-per-pixel.  Rungs above this are wasteful.
    /// Default: `0.5`.
    pub max_bpp: f64,

    /// Maximum allowed bitrate ratio between adjacent rungs (upper/lower).
    /// A ratio > `max_bitrate_gap_ratio` is flagged as a warning.
    /// Default: `3.0`.
    pub max_bitrate_gap_ratio: f64,

    /// Minimum allowed bitrate ratio between adjacent rungs (upper/lower).
    /// A ratio < `min_bitrate_gap_ratio` means the rungs are too close.
    /// Default: `1.2`.
    pub min_bitrate_gap_ratio: f64,

    /// Whether to require that rungs use the same codec throughout.
    pub require_uniform_codec: bool,

    /// Whether segment durations must be within spec bounds.
    pub check_segment_duration: bool,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            min_bpp: 0.03,
            max_bpp: 0.50,
            max_bitrate_gap_ratio: 3.0,
            min_bitrate_gap_ratio: 1.2,
            require_uniform_codec: false,
            check_segment_duration: true,
        }
    }
}

// ─── LadderValidator ─────────────────────────────────────────────────────────

/// Validates an [`EncodeLadder`] against a streaming specification.
///
/// See the [module-level documentation](crate::encode_ladder_validator) for
/// an overview of what is checked.
#[derive(Debug, Clone)]
pub struct LadderValidator {
    spec: LadderSpec,
    config: ValidatorConfig,
}

impl LadderValidator {
    /// Creates a validator with default configuration for the given spec.
    #[must_use]
    pub fn new(spec: LadderSpec) -> Self {
        Self {
            spec,
            config: ValidatorConfig::default(),
        }
    }

    /// Creates a validator with custom configuration.
    #[must_use]
    pub fn with_config(spec: LadderSpec, config: ValidatorConfig) -> Self {
        Self { spec, config }
    }

    /// Validates the ladder and returns a [`LadderReport`].
    #[must_use]
    pub fn validate(&self, ladder: &EncodeLadder) -> LadderReport {
        let mut findings: Vec<LadderFinding> = Vec::new();

        // 1. Empty ladder check.
        if ladder.is_empty() {
            findings.push(LadderFinding::error(None, "Ladder has no rungs"));
            return LadderReport {
                findings,
                spec: self.spec,
                rung_count: 0,
            };
        }

        // 2. Too many rungs.
        if ladder.len() > self.spec.max_rungs() {
            findings.push(LadderFinding::warning(
                None,
                format!(
                    "Ladder has {} rungs; spec {:?} recommends at most {}",
                    ladder.len(),
                    self.spec,
                    self.spec.max_rungs()
                ),
            ));
        }

        // 3. Per-rung checks.
        for (i, rung) in ladder.rungs.iter().enumerate() {
            self.check_rung(i, rung, &mut findings);
        }

        // 4. Cross-rung checks (ordering, gaps).
        self.check_ordering(ladder, &mut findings);

        // 5. Codec uniformity.
        if self.config.require_uniform_codec {
            let codecs = ladder.codec_set();
            if codecs.len() > 1 {
                findings.push(LadderFinding::warning(
                    None,
                    format!(
                        "Ladder uses multiple codecs ({:?}); consider a uniform codec for \
                         simpler packaging",
                        codecs
                    ),
                ));
            }
        }

        // 6. Spec-level codec compliance.
        for (i, rung) in ladder.rungs.iter().enumerate() {
            if !self.spec.allows_codec(&rung.codec) {
                findings.push(LadderFinding::error(
                    Some(i),
                    format!("Codec '{}' is not allowed by {:?}", rung.codec, self.spec),
                ));
            }
        }

        LadderReport {
            findings,
            spec: self.spec,
            rung_count: ladder.len(),
        }
    }

    fn check_rung(&self, i: usize, rung: &LadderRung, findings: &mut Vec<LadderFinding>) {
        // Zero dimensions.
        if rung.width == 0 || rung.height == 0 {
            findings.push(LadderFinding::error(
                Some(i),
                format!("Rung {} has zero dimension ({}×{})", i, rung.width, rung.height),
            ));
        }

        // Zero bitrate.
        if rung.bitrate_bps == 0 {
            findings.push(LadderFinding::error(
                Some(i),
                format!("Rung {} has zero bitrate", i),
            ));
        }

        // Invalid frame rate.
        if rung.frame_rate <= 0.0 || !rung.frame_rate.is_finite() {
            findings.push(LadderFinding::error(
                Some(i),
                format!("Rung {} has invalid frame rate {:.3}", i, rung.frame_rate),
            ));
        }

        // Bits-per-pixel range.
        let bpp = rung.bits_per_pixel();
        if bpp < self.config.min_bpp && bpp > 0.0 {
            findings.push(LadderFinding::warning(
                Some(i),
                format!(
                    "Rung {} bpp {:.4} is below minimum {:.4}; may appear blocky",
                    i, bpp, self.config.min_bpp
                ),
            ));
        }
        if bpp > self.config.max_bpp {
            findings.push(LadderFinding::info(
                Some(i),
                format!(
                    "Rung {} bpp {:.4} exceeds maximum {:.4}; possibly over-provisioned",
                    i, bpp, self.config.max_bpp
                ),
            ));
        }

        // Segment duration.
        if self.config.check_segment_duration && rung.segment_duration_s > 0.0 {
            if let Some((min_s, max_s)) = self.spec.segment_duration_range() {
                if rung.segment_duration_s < min_s || rung.segment_duration_s > max_s {
                    findings.push(LadderFinding::error(
                        Some(i),
                        format!(
                            "Rung {} segment duration {:.2}s is outside spec range \
                             [{min_s:.2}, {max_s:.2}] for {:?}",
                            i, rung.segment_duration_s, self.spec
                        ),
                    ));
                }
            }
        }
    }

    fn check_ordering(&self, ladder: &EncodeLadder, findings: &mut Vec<LadderFinding>) {
        let rungs = &ladder.rungs;
        for i in 1..rungs.len() {
            let upper = &rungs[i - 1];
            let lower = &rungs[i];

            // Bitrate crossover: higher resolution rung should have higher bitrate.
            if upper.pixels() > lower.pixels() && upper.bitrate_bps < lower.bitrate_bps {
                findings.push(LadderFinding::error(
                    Some(i),
                    format!(
                        "Bitrate crossover: rung {} ({}×{} @ {}bps) has higher resolution \
                         but lower bitrate than rung {} ({}×{} @ {}bps)",
                        i - 1,
                        upper.width,
                        upper.height,
                        upper.bitrate_bps,
                        i,
                        lower.width,
                        lower.height,
                        lower.bitrate_bps
                    ),
                ));
            }

            // Bitrate gap.
            if lower.bitrate_bps > 0 {
                let ratio = upper.bitrate_bps as f64 / lower.bitrate_bps as f64;

                if ratio > self.config.max_bitrate_gap_ratio {
                    findings.push(LadderFinding::warning(
                        Some(i),
                        format!(
                            "Bitrate gap between rung {} ({} bps) and rung {} ({} bps) is \
                             {ratio:.2}× — may cause large quality jumps during ABR switching",
                            i - 1,
                            upper.bitrate_bps,
                            i,
                            lower.bitrate_bps,
                        ),
                    ));
                }

                if ratio < self.config.min_bitrate_gap_ratio && ratio > 0.0 {
                    findings.push(LadderFinding::info(
                        Some(i),
                        format!(
                            "Bitrate gap between rung {} and rung {} is only {ratio:.2}× \
                             — rungs may be redundant",
                            i - 1,
                            i,
                        ),
                    ));
                }
            }

            // Duplicate resolution check.
            if upper.width == lower.width && upper.height == lower.height {
                findings.push(LadderFinding::warning(
                    Some(i),
                    format!(
                        "Rungs {} and {} have the same resolution {}×{}",
                        i - 1,
                        i,
                        upper.width,
                        upper.height
                    ),
                ));
            }
        }
    }
}

// ─── Convenience constructors ─────────────────────────────────────────────────

/// Builds a standard 4-rung HLS ladder suitable for VP9 streaming.
///
/// Rungs: 1080p @ 4.5 Mbps, 720p @ 2.5 Mbps, 480p @ 1 Mbps, 360p @ 500 kbps.
#[must_use]
pub fn vp9_hls_ladder() -> EncodeLadder {
    EncodeLadder::new(vec![
        LadderRung::new(1920, 1080, 4_500_000, 30.0, "vp9"),
        LadderRung::new(1280, 720, 2_500_000, 30.0, "vp9"),
        LadderRung::new(854, 480, 1_000_000, 30.0, "vp9"),
        LadderRung::new(640, 360, 500_000, 30.0, "vp9"),
    ])
}

/// Builds a standard 4-rung CMAF ladder suitable for AV1.
///
/// Segment duration is set to 2.0 s for CMAF compliance.
#[must_use]
pub fn av1_cmaf_ladder() -> EncodeLadder {
    EncodeLadder::new(vec![
        LadderRung::new(1920, 1080, 3_500_000, 30.0, "av1")
            .with_segment_duration(2.0),
        LadderRung::new(1280, 720, 1_800_000, 30.0, "av1")
            .with_segment_duration(2.0),
        LadderRung::new(854, 480, 800_000, 30.0, "av1")
            .with_segment_duration(2.0),
        LadderRung::new(640, 360, 350_000, 30.0, "av1")
            .with_segment_duration(2.0),
    ])
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_vp9_hls_ladder() -> EncodeLadder {
        vp9_hls_ladder()
    }

    #[test]
    fn test_valid_ladder_passes() {
        let ladder = valid_vp9_hls_ladder();
        let report = LadderValidator::new(LadderSpec::Hls).validate(&ladder);
        assert!(report.is_ok(), "errors: {:?}", report.errors());
    }

    #[test]
    fn test_empty_ladder_errors() {
        let ladder = EncodeLadder::new(vec![]);
        let report = LadderValidator::new(LadderSpec::Hls).validate(&ladder);
        assert!(!report.is_ok());
        assert!(report.error_count() >= 1);
    }

    #[test]
    fn test_zero_bitrate_errors() {
        let ladder = EncodeLadder::new(vec![
            LadderRung::new(1920, 1080, 0, 30.0, "vp9"), // zero bitrate
        ]);
        let report = LadderValidator::new(LadderSpec::Generic).validate(&ladder);
        assert!(!report.is_ok());
        let msgs: Vec<&str> = report.errors().iter().map(|f| f.message.as_str()).collect();
        assert!(
            msgs.iter().any(|m| m.contains("zero bitrate")),
            "msgs: {msgs:?}"
        );
    }

    #[test]
    fn test_bitrate_crossover_detected() {
        // Higher resolution with lower bitrate than the rung below.
        let ladder = EncodeLadder::new(vec![
            LadderRung::new(1920, 1080, 500_000, 30.0, "vp9"), // crossover
            LadderRung::new(1280, 720, 2_500_000, 30.0, "vp9"),
        ]);
        let report = LadderValidator::new(LadderSpec::Hls).validate(&ladder);
        let error_msgs: Vec<&str> = report.errors().iter().map(|f| f.message.as_str()).collect();
        assert!(
            error_msgs.iter().any(|m| m.contains("crossover")),
            "errors: {error_msgs:?}"
        );
    }

    #[test]
    fn test_duplicate_resolution_warns() {
        let ladder = EncodeLadder::new(vec![
            LadderRung::new(1280, 720, 2_000_000, 30.0, "vp9"),
            LadderRung::new(1280, 720, 1_000_000, 30.0, "vp9"), // duplicate res
        ]);
        let report = LadderValidator::new(LadderSpec::Hls).validate(&ladder);
        let warn_msgs: Vec<&str> = report
            .warnings()
            .iter()
            .map(|f| f.message.as_str())
            .collect();
        assert!(
            warn_msgs.iter().any(|m| m.contains("same resolution")),
            "warnings: {warn_msgs:?}"
        );
    }

    #[test]
    fn test_invalid_codec_for_spec_errors() {
        let ladder = EncodeLadder::new(vec![
            LadderRung::new(1920, 1080, 4_000_000, 30.0, "theora"), // not in HLS spec
        ]);
        let report = LadderValidator::new(LadderSpec::Hls).validate(&ladder);
        assert!(report.error_count() > 0, "expected error for unsupported codec");
    }

    #[test]
    fn test_segment_duration_out_of_range_errors() {
        // CMAF requires segment duration between 1.0 and 6.0 s.
        let ladder = EncodeLadder::new(vec![
            LadderRung::new(1920, 1080, 4_000_000, 30.0, "av1")
                .with_segment_duration(0.1), // too short for CMAF
        ]);
        let report = LadderValidator::new(LadderSpec::Cmaf).validate(&ladder);
        let error_msgs: Vec<&str> = report.errors().iter().map(|f| f.message.as_str()).collect();
        assert!(
            error_msgs.iter().any(|m| m.contains("segment duration")),
            "errors: {error_msgs:?}"
        );
    }

    #[test]
    fn test_av1_cmaf_ladder_passes() {
        let ladder = av1_cmaf_ladder();
        let report = LadderValidator::new(LadderSpec::Cmaf).validate(&ladder);
        assert!(report.is_ok(), "errors: {:?}", report.errors());
    }

    #[test]
    fn test_bits_per_pixel() {
        let rung = LadderRung::new(1920, 1080, 8_294_400, 30.0, "vp9");
        // bpp = 8_294_400 / (1920 * 1080 * 30) ≈ 0.1333...
        let bpp = rung.bits_per_pixel();
        assert!((bpp - 0.1333).abs() < 0.001, "bpp {bpp}");
    }

    #[test]
    fn test_aspect_ratio() {
        let rung = LadderRung::new(1920, 1080, 4_000_000, 30.0, "vp9");
        assert_eq!(rung.aspect_ratio(), (16, 9));
        let rung4_3 = LadderRung::new(640, 480, 1_000_000, 30.0, "vp9");
        assert_eq!(rung4_3.aspect_ratio(), (4, 3));
    }

    #[test]
    fn test_large_bitrate_gap_warns() {
        let ladder = EncodeLadder::new(vec![
            LadderRung::new(1920, 1080, 10_000_000, 30.0, "vp9"),
            LadderRung::new(1280, 720, 100_000, 30.0, "vp9"), // 100× gap
        ]);
        let report = LadderValidator::new(LadderSpec::Hls).validate(&ladder);
        assert!(
            !report.warnings().is_empty(),
            "expected bitrate gap warning"
        );
    }

    #[test]
    fn test_rung_display() {
        let rung = LadderRung::new(1920, 1080, 4_500_000, 30.0, "vp9");
        let s = rung.to_string();
        assert!(s.contains("1920"), "display: {s}");
        assert!(s.contains("1080"), "display: {s}");
        assert!(s.contains("vp9"), "display: {s}");
    }

    #[test]
    fn test_codec_set() {
        let ladder = EncodeLadder::new(vec![
            LadderRung::new(1920, 1080, 4_000_000, 30.0, "vp9"),
            LadderRung::new(1280, 720, 2_000_000, 30.0, "av1"),
        ]);
        let codecs = ladder.codec_set();
        assert!(codecs.contains("vp9"));
        assert!(codecs.contains("av1"));
        assert_eq!(codecs.len(), 2);
    }

    #[test]
    fn test_sort_descending() {
        let mut ladder = EncodeLadder::new(vec![
            LadderRung::new(640, 360, 500_000, 30.0, "vp9"),
            LadderRung::new(1920, 1080, 4_500_000, 30.0, "vp9"),
            LadderRung::new(1280, 720, 2_500_000, 30.0, "vp9"),
        ]);
        ladder.sort_descending();
        assert_eq!(ladder.rungs[0].bitrate_bps, 4_500_000);
        assert_eq!(ladder.rungs[1].bitrate_bps, 2_500_000);
        assert_eq!(ladder.rungs[2].bitrate_bps, 500_000);
    }

    #[test]
    fn test_spec_segment_duration_range() {
        assert!(LadderSpec::Hls.segment_duration_range().is_some());
        assert!(LadderSpec::LlHls.segment_duration_range().is_some());
        assert!(LadderSpec::Generic.segment_duration_range().is_none());
        let (min, max) = LadderSpec::LlHls.segment_duration_range().unwrap();
        assert!(max < 3.0, "LL-HLS max segment should be short");
        assert!(min < max);
    }
}
