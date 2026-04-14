//! Enhanced transcode time and resource estimator.
//!
//! This module provides a structured analytical estimator for transcoding
//! operations, modelling CPU time, RAM usage, output file size, and thread
//! requirements for patent-free codecs (AV1, VP9, VP8, FLAC, Opus, Vorbis).
//!
//! All calculations are pure-function approximations suitable for pre-flight
//! planning, job scheduling, and storage budgeting. No actual encoding is
//! performed.
//!
//! # Example
//!
//! ```rust
//! use oximedia_transcode::transcode_estimator::{
//!     EstimateInput, TargetCodec, TranscodeEstimatorV2,
//! };
//!
//! let input = EstimateInput {
//!     duration_secs: 60.0,
//!     width: 1920,
//!     height: 1080,
//!     input_bitrate_kbps: 8_000,
//!     target_bitrate_kbps: 4_000,
//!     codec: TargetCodec::Av1,
//!     has_hdr: false,
//! };
//! let result = TranscodeEstimatorV2::default().estimate(&input, 8);
//! assert!(result.estimated_secs > 0.0);
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use serde::{Deserialize, Serialize};

// ─── TargetCodec ────────────────────────────────────────────────────────────

/// Codec target for a transcode operation.
///
/// Each variant carries codec-specific complexity characteristics used to
/// refine the time and resource estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetCodec {
    /// AV1 (libaom-av1 / SVT-AV1).  Highest quality, highest CPU cost.
    Av1,
    /// VP9 (libvpx-vp9).  Good quality, moderate CPU cost.
    Vp9,
    /// VP8 (libvpx).  Legacy codec, lowest CPU cost among video codecs.
    Vp8,
    /// FLAC lossless audio.
    Flac,
    /// Opus perceptual audio codec.
    Opus,
    /// Vorbis perceptual audio codec.
    Vorbis,
}

impl TargetCodec {
    /// Returns a dimensionless complexity factor relative to VP8 = 1.0.
    ///
    /// The factor is used to scale the estimated encoding time.
    /// Higher values → more CPU time per pixel.
    #[must_use]
    pub fn complexity_factor(self) -> f32 {
        match self {
            Self::Av1 => 4.0,
            Self::Vp9 => 2.0,
            Self::Vp8 => 1.0,
            Self::Flac => 0.1,
            Self::Opus => 0.08,
            Self::Vorbis => 0.09,
        }
    }

    /// Returns the approximate RAM per megapixel of input (in MiB) at default
    /// encoder settings.
    #[must_use]
    pub fn ram_per_megapixel_mib(self) -> f32 {
        match self {
            Self::Av1 => 48.0,  // large lookahead buffer, CDEF, film grain
            Self::Vp9 => 24.0,  // multi-threaded tile buffers
            Self::Vp8 => 12.0,  // single-loop filter pass
            Self::Flac => 2.0,  // minimal buffering
            Self::Opus => 1.5,  // SILK / CELT frame buffers
            Self::Vorbis => 2.0, // MDCT window buffers
        }
    }

    /// Returns whether this codec is audio-only (no pixel processing).
    #[must_use]
    pub fn is_audio(self) -> bool {
        matches!(self, Self::Flac | Self::Opus | Self::Vorbis)
    }

    /// Returns a human-readable codec name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Av1 => "AV1",
            Self::Vp9 => "VP9",
            Self::Vp8 => "VP8",
            Self::Flac => "FLAC",
            Self::Opus => "Opus",
            Self::Vorbis => "Vorbis",
        }
    }
}

// ─── EstimateInput ───────────────────────────────────────────────────────────

/// Descriptor for the media to be transcoded.
///
/// All fields must be positive; zero/negative values cause the estimator to
/// return a zero-time, zero-size result so callers can detect invalid inputs
/// via the `confidence` field in `EstimateResult`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimateInput {
    /// Source duration in seconds.
    pub duration_secs: f64,
    /// Frame width in pixels (0 for audio-only).
    pub width: u32,
    /// Frame height in pixels (0 for audio-only).
    pub height: u32,
    /// Source stream bitrate in kbps (used for complexity heuristics).
    pub input_bitrate_kbps: u32,
    /// Target output bitrate in kbps.
    pub target_bitrate_kbps: u32,
    /// Target codec.
    pub codec: TargetCodec,
    /// Whether the source carries HDR metadata (slightly increases RAM).
    pub has_hdr: bool,
}

impl EstimateInput {
    /// Returns the total pixel count (width × height).
    #[must_use]
    pub fn total_pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Returns the megapixel count (fractional).
    #[must_use]
    pub fn megapixels(&self) -> f64 {
        self.total_pixels() as f64 / 1_000_000.0
    }

    /// Returns the estimated fps based on typical content heuristics.
    ///
    /// If `input_bitrate_kbps` is 0 or the content appears to be audio-only
    /// (based on codec), returns 0.0.
    #[must_use]
    pub fn estimated_fps(&self) -> f64 {
        if self.codec.is_audio() || self.total_pixels() == 0 {
            return 0.0;
        }
        // Simple heuristic: broadcast-like content tends to be 25–60 fps.
        // We use 30.0 as the baseline reference fps.
        30.0_f64
    }
}

// ─── EstimateResult ──────────────────────────────────────────────────────────

/// Estimated resources required to transcode the described input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimateResult {
    /// Estimated wall-clock encoding time in seconds.
    pub estimated_secs: f64,
    /// Suggested number of CPU threads to use for encoding.
    pub cpu_threads_suggested: u8,
    /// Estimated peak RAM consumption in MiB.
    pub ram_mb_estimated: u32,
    /// Estimated output file size in bytes.
    pub output_size_bytes_estimated: u64,
    /// Confidence score in `[0.0, 1.0]`.
    ///
    /// 1.0 indicates all inputs are well-defined.  Values below 0.5 indicate
    /// that one or more input parameters were missing or implausible.
    pub confidence: f32,
}

impl EstimateResult {
    /// Returns `true` when the estimate has a confidence above 0.5.
    #[must_use]
    pub fn is_reliable(&self) -> bool {
        self.confidence >= 0.5
    }

    /// Returns a human-readable summary of the estimate.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "~{:.0}s encode, ~{} MB RAM, ~{} MB output ({:.0}% confidence)",
            self.estimated_secs,
            self.ram_mb_estimated,
            self.output_size_bytes_estimated / 1_048_576,
            self.confidence * 100.0
        )
    }
}

// ─── EstimatorCalibration ────────────────────────────────────────────────────

/// Reference measurements used to calibrate the estimator model.
///
/// The default values represent empirically observed throughput on a modern
/// 8-core x86-64 workstation at a medium encoder preset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimatorCalibration {
    /// Reference pixels-per-second throughput for VP8 at the reference
    /// thread count (used as the baseline for all codecs).
    ///
    /// Default: 120_000_000 px/s (≈ 60 fps 1080p real-time on a single thread).
    pub reference_pixels_per_sec: f64,

    /// Reference thread count for the above throughput measurement.
    ///
    /// Default: 1 (single-threaded measurement).
    pub reference_threads: u8,

    /// Thread efficiency factor: the fraction of linear speedup achieved per
    /// additional thread.  1.0 = perfectly linear, 0.7 = Amdahl-limited.
    ///
    /// Default: 0.75
    pub thread_efficiency: f32,

    /// Additional RAM overhead in MiB added unconditionally (OS, container
    /// library allocations, etc.).
    ///
    /// Default: 64 MiB
    pub base_ram_overhead_mib: u32,

    /// Extra RAM multiplier applied when `has_hdr` is `true`.
    ///
    /// Default: 1.15 (+15%)
    pub hdr_ram_multiplier: f32,
}

impl Default for EstimatorCalibration {
    fn default() -> Self {
        Self {
            reference_pixels_per_sec: 120_000_000.0,
            reference_threads: 1,
            thread_efficiency: 0.75,
            base_ram_overhead_mib: 64,
            hdr_ram_multiplier: 1.15,
        }
    }
}

impl EstimatorCalibration {
    /// Validates calibration parameters.
    ///
    /// # Errors
    ///
    /// Returns an error string if any parameter is out of range.
    pub fn validate(&self) -> Result<(), String> {
        if self.reference_pixels_per_sec <= 0.0 {
            return Err("reference_pixels_per_sec must be > 0".to_string());
        }
        if self.reference_threads == 0 {
            return Err("reference_threads must be >= 1".to_string());
        }
        if !(0.0..=1.0).contains(&self.thread_efficiency) {
            return Err(format!(
                "thread_efficiency {} must be in [0.0, 1.0]",
                self.thread_efficiency
            ));
        }
        if self.hdr_ram_multiplier < 1.0 {
            return Err(format!(
                "hdr_ram_multiplier {} must be >= 1.0",
                self.hdr_ram_multiplier
            ));
        }
        Ok(())
    }

    /// Computes the effective thread speedup for `n` threads vs. the reference.
    ///
    /// Uses the Amdahl-limited model: `speedup = 1 + (n - ref) * efficiency`.
    #[must_use]
    pub fn thread_speedup(&self, n: u8) -> f64 {
        let ref_n = f64::from(self.reference_threads);
        let eff = f64::from(self.thread_efficiency);
        let n_f = f64::from(n);
        if n_f <= ref_n {
            (n_f / ref_n).max(0.1)
        } else {
            1.0 + (n_f - ref_n) * eff
        }
    }
}

// ─── TranscodeEstimatorV2 ────────────────────────────────────────────────────

/// Enhanced transcode time and resource estimator.
///
/// Uses a calibrated analytical model to estimate encoding duration, RAM,
/// output size, and optimal thread count for patent-free codec targets.
#[derive(Debug, Clone)]
pub struct TranscodeEstimatorV2 {
    /// Calibration parameters governing the model accuracy.
    pub calibration: EstimatorCalibration,
}

impl Default for TranscodeEstimatorV2 {
    fn default() -> Self {
        Self {
            calibration: EstimatorCalibration::default(),
        }
    }
}

impl TranscodeEstimatorV2 {
    /// Creates an estimator with the given calibration.
    #[must_use]
    pub fn with_calibration(calibration: EstimatorCalibration) -> Self {
        Self { calibration }
    }

    /// Returns the current calibration.
    #[must_use]
    pub fn calibration(&self) -> &EstimatorCalibration {
        &self.calibration
    }

    /// Produces a resource estimate for the given input and available threads.
    ///
    /// When `cpu_threads_available` is 0 the estimator treats it as 1.
    #[must_use]
    pub fn estimate(&self, input: &EstimateInput, cpu_threads_available: u8) -> EstimateResult {
        let threads = cpu_threads_available.max(1);
        let confidence = self.compute_confidence(input);

        if confidence < 0.01 {
            return EstimateResult {
                estimated_secs: 0.0,
                cpu_threads_suggested: 1,
                ram_mb_estimated: self.calibration.base_ram_overhead_mib,
                output_size_bytes_estimated: 0,
                confidence,
            };
        }

        let estimated_secs = self.estimate_time(input, threads);
        let cpu_threads_suggested = self.suggest_threads(input, threads);
        let ram_mb_estimated = self.estimate_ram(input);
        let output_size_bytes_estimated = self.estimate_output_size(input);

        EstimateResult {
            estimated_secs,
            cpu_threads_suggested,
            ram_mb_estimated,
            output_size_bytes_estimated,
            confidence,
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Estimates encoding wall time in seconds.
    fn estimate_time(&self, input: &EstimateInput, threads: u8) -> f64 {
        if input.duration_secs <= 0.0 {
            return 0.0;
        }

        if input.codec.is_audio() {
            // Audio codecs are nearly always faster than real-time; apply a
            // small complexity multiplier to the duration.
            let complexity = f64::from(input.codec.complexity_factor());
            return (input.duration_secs * complexity).max(0.001);
        }

        let total_pixels = input.total_pixels();
        if total_pixels == 0 {
            return 0.0;
        }

        let fps = input.estimated_fps();
        // Total pixels to encode
        let total_pixel_ops = total_pixels as f64 * fps * input.duration_secs;

        // Effective throughput = reference / complexity * thread_speedup
        let complexity = f64::from(input.codec.complexity_factor());
        let thread_speedup = self.calibration.thread_speedup(threads);
        let effective_px_per_sec =
            self.calibration.reference_pixels_per_sec / complexity * thread_speedup;

        if effective_px_per_sec <= 0.0 {
            return input.duration_secs;
        }

        (total_pixel_ops / effective_px_per_sec).max(0.001)
    }

    /// Estimates peak RAM in MiB.
    fn estimate_ram(&self, input: &EstimateInput) -> u32 {
        let base = f64::from(self.calibration.base_ram_overhead_mib);

        let codec_ram = if input.codec.is_audio() {
            // Audio: minimal per-channel buffers
            f64::from(input.codec.ram_per_megapixel_mib()) * 2.0
        } else {
            let mp = input.megapixels().max(0.01);
            f64::from(input.codec.ram_per_megapixel_mib()) * mp
        };

        let total = if input.has_hdr {
            (base + codec_ram) * f64::from(self.calibration.hdr_ram_multiplier)
        } else {
            base + codec_ram
        };

        total.ceil() as u32
    }

    /// Estimates output file size in bytes using target_bitrate_kbps.
    fn estimate_output_size(&self, input: &EstimateInput) -> u64 {
        if input.duration_secs <= 0.0 || input.target_bitrate_kbps == 0 {
            return 0;
        }
        // bytes = (kbps * 1000 / 8) * duration_secs
        let bytes_per_sec = u64::from(input.target_bitrate_kbps) * 1000 / 8;
        (bytes_per_sec as f64 * input.duration_secs) as u64
    }

    /// Suggests the number of threads based on codec and available threads.
    fn suggest_threads(&self, input: &EstimateInput, available: u8) -> u8 {
        if input.codec.is_audio() {
            // Audio encoding does not benefit significantly from many threads.
            return 1_u8.min(available);
        }

        // AV1 and VP9 benefit greatly from tiles/threads.  VP8 is mostly
        // single-threaded.
        let max_useful: u8 = match input.codec {
            TargetCodec::Av1 => 16,
            TargetCodec::Vp9 => 8,
            TargetCodec::Vp8 => 4,
            _ => 1,
        };

        available.min(max_useful)
    }

    /// Computes a confidence score based on input parameter plausibility.
    fn compute_confidence(&self, input: &EstimateInput) -> f32 {
        let mut score: f32 = 1.0;

        if input.duration_secs <= 0.0 {
            score *= 0.0; // completely invalid without duration
        }
        if input.duration_secs > 86_400.0 {
            score *= 0.6; // >24h content is unusual
        }
        if !input.codec.is_audio() && input.total_pixels() == 0 {
            score *= 0.0; // video codec with no resolution
        }
        if input.target_bitrate_kbps == 0 {
            score *= 0.7; // output size estimate is meaningless
        }
        if input.input_bitrate_kbps == 0 {
            score *= 0.9; // complexity heuristic degraded
        }
        if !input.codec.is_audio() {
            let mp = input.megapixels();
            if mp > 50.0 {
                score *= 0.7; // unusual resolution (>50 MP)
            }
        }

        score.clamp(0.0, 1.0)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_1080p_av1(duration: f64) -> EstimateInput {
        EstimateInput {
            duration_secs: duration,
            width: 1920,
            height: 1080,
            input_bitrate_kbps: 8_000,
            target_bitrate_kbps: 4_000,
            codec: TargetCodec::Av1,
            has_hdr: false,
        }
    }

    fn make_720p_vp8(duration: f64) -> EstimateInput {
        EstimateInput {
            duration_secs: duration,
            width: 1280,
            height: 720,
            input_bitrate_kbps: 4_000,
            target_bitrate_kbps: 2_000,
            codec: TargetCodec::Vp8,
            has_hdr: false,
        }
    }

    fn make_4k_av1(duration: f64) -> EstimateInput {
        EstimateInput {
            duration_secs: duration,
            width: 3840,
            height: 2160,
            input_bitrate_kbps: 30_000,
            target_bitrate_kbps: 15_000,
            codec: TargetCodec::Av1,
            has_hdr: true,
        }
    }

    // ── TargetCodec tests ────────────────────────────────────────────────────

    #[test]
    fn test_codec_complexity_ordering() {
        // AV1 > VP9 > VP8 > audio codecs
        let av1 = TargetCodec::Av1.complexity_factor();
        let vp9 = TargetCodec::Vp9.complexity_factor();
        let vp8 = TargetCodec::Vp8.complexity_factor();
        let opus = TargetCodec::Opus.complexity_factor();

        assert!(av1 > vp9, "AV1 should be more complex than VP9");
        assert!(vp9 > vp8, "VP9 should be more complex than VP8");
        assert!(vp8 > opus, "VP8 should be more complex than Opus (audio)");
    }

    #[test]
    fn test_audio_codecs_identified_correctly() {
        assert!(TargetCodec::Flac.is_audio());
        assert!(TargetCodec::Opus.is_audio());
        assert!(TargetCodec::Vorbis.is_audio());
        assert!(!TargetCodec::Av1.is_audio());
        assert!(!TargetCodec::Vp9.is_audio());
        assert!(!TargetCodec::Vp8.is_audio());
    }

    #[test]
    fn test_codec_names_non_empty() {
        let codecs = [
            TargetCodec::Av1,
            TargetCodec::Vp9,
            TargetCodec::Vp8,
            TargetCodec::Flac,
            TargetCodec::Opus,
            TargetCodec::Vorbis,
        ];
        for codec in codecs {
            assert!(!codec.name().is_empty(), "{codec:?} name should not be empty");
        }
    }

    // ── EstimatorCalibration tests ───────────────────────────────────────────

    #[test]
    fn test_calibration_default_is_valid() {
        let cal = EstimatorCalibration::default();
        assert!(cal.validate().is_ok(), "Default calibration should be valid");
    }

    #[test]
    fn test_calibration_thread_speedup_single() {
        let cal = EstimatorCalibration::default(); // reference_threads = 1
        // Single thread vs single reference → speedup = 1.0
        let speedup = cal.thread_speedup(1);
        assert!((speedup - 1.0).abs() < 0.01, "1-thread speedup should be ~1.0");
    }

    #[test]
    fn test_calibration_thread_speedup_increases_with_threads() {
        let cal = EstimatorCalibration::default();
        let s1 = cal.thread_speedup(1);
        let s4 = cal.thread_speedup(4);
        let s8 = cal.thread_speedup(8);
        assert!(s4 > s1, "4 threads should be faster than 1");
        assert!(s8 > s4, "8 threads should be faster than 4");
    }

    #[test]
    fn test_calibration_invalid_zero_reference() {
        let cal = EstimatorCalibration {
            reference_pixels_per_sec: 0.0,
            ..EstimatorCalibration::default()
        };
        assert!(cal.validate().is_err());
    }

    #[test]
    fn test_calibration_invalid_thread_efficiency() {
        let cal = EstimatorCalibration {
            thread_efficiency: 1.5,
            ..EstimatorCalibration::default()
        };
        assert!(cal.validate().is_err());
    }

    // ── TranscodeEstimatorV2 tests ───────────────────────────────────────────

    #[test]
    fn test_4k_takes_longer_than_720p_same_duration() {
        let est = TranscodeEstimatorV2::default();
        let r_4k = est.estimate(&make_4k_av1(60.0), 4);
        let r_720p = est.estimate(&make_720p_vp8(60.0), 4);

        assert!(
            r_4k.estimated_secs > r_720p.estimated_secs,
            "4K AV1 should take longer than 720p VP8"
        );
    }

    #[test]
    fn test_av1_slower_than_vp8_same_resolution() {
        let est = TranscodeEstimatorV2::default();
        let av1_input = make_1080p_av1(60.0);
        let vp8_input = EstimateInput {
            codec: TargetCodec::Vp8,
            ..make_1080p_av1(60.0)
        };

        let r_av1 = est.estimate(&av1_input, 4);
        let r_vp8 = est.estimate(&vp8_input, 4);

        assert!(
            r_av1.estimated_secs > r_vp8.estimated_secs,
            "AV1 should take longer than VP8 at same resolution"
        );
    }

    #[test]
    fn test_output_size_formula() {
        let est = TranscodeEstimatorV2::default();
        let input = EstimateInput {
            duration_secs: 10.0,
            width: 1920,
            height: 1080,
            input_bitrate_kbps: 8_000,
            target_bitrate_kbps: 4_000,
            codec: TargetCodec::Vp9,
            has_hdr: false,
        };
        let result = est.estimate(&input, 4);

        // Expected: 4000 kbps * 1000 / 8 * 10 = 5_000_000 bytes
        let expected_bytes: u64 = 4_000 * 1_000 / 8 * 10;
        assert_eq!(
            result.output_size_bytes_estimated, expected_bytes,
            "Output size formula mismatch"
        );
    }

    #[test]
    fn test_more_threads_reduces_estimated_time() {
        let est = TranscodeEstimatorV2::default();
        let input = make_4k_av1(300.0);

        let r1 = est.estimate(&input, 1);
        let r8 = est.estimate(&input, 8);

        assert!(
            r8.estimated_secs < r1.estimated_secs,
            "More threads should reduce estimated time"
        );
    }

    #[test]
    fn test_thread_suggestion_capped_by_codec() {
        let est = TranscodeEstimatorV2::default();
        let vp8_input = make_720p_vp8(60.0);
        let av1_input = make_4k_av1(60.0);

        let r_vp8 = est.estimate(&vp8_input, 16);
        let r_av1 = est.estimate(&av1_input, 16);

        // VP8 is capped at 4 threads, AV1 at 16
        assert!(
            r_av1.cpu_threads_suggested >= r_vp8.cpu_threads_suggested,
            "AV1 should suggest at least as many threads as VP8"
        );
    }

    #[test]
    fn test_hdr_increases_ram() {
        let est = TranscodeEstimatorV2::default();
        let sdr = make_1080p_av1(60.0);
        let hdr = EstimateInput {
            has_hdr: true,
            ..make_1080p_av1(60.0)
        };

        let r_sdr = est.estimate(&sdr, 4);
        let r_hdr = est.estimate(&hdr, 4);

        assert!(
            r_hdr.ram_mb_estimated >= r_sdr.ram_mb_estimated,
            "HDR content should require at least as much RAM as SDR"
        );
    }

    #[test]
    fn test_zero_duration_returns_zero_output_size() {
        let est = TranscodeEstimatorV2::default();
        let input = EstimateInput {
            duration_secs: 0.0,
            ..make_1080p_av1(0.0)
        };
        let result = est.estimate(&input, 4);
        assert_eq!(
            result.output_size_bytes_estimated, 0,
            "Zero duration should produce zero output size"
        );
    }

    #[test]
    fn test_audio_codec_estimate_is_fast() {
        let est = TranscodeEstimatorV2::default();
        let input = EstimateInput {
            duration_secs: 3600.0, // 1 hour of audio
            width: 0,
            height: 0,
            input_bitrate_kbps: 1_411,
            target_bitrate_kbps: 128,
            codec: TargetCodec::Opus,
            has_hdr: false,
        };
        let result = est.estimate(&input, 4);

        // Opus encoding 1h of audio should complete well within the source
        // duration (real-time factor << 1)
        assert!(
            result.estimated_secs < input.duration_secs,
            "Opus encoding 1h should be faster than real-time"
        );
    }

    #[test]
    fn test_confidence_invalid_inputs() {
        let est = TranscodeEstimatorV2::default();

        // Zero duration → confidence = 0
        let zero_dur = EstimateInput {
            duration_secs: 0.0,
            width: 1920,
            height: 1080,
            input_bitrate_kbps: 8_000,
            target_bitrate_kbps: 4_000,
            codec: TargetCodec::Av1,
            has_hdr: false,
        };
        let r = est.estimate(&zero_dur, 4);
        assert_eq!(r.confidence, 0.0, "Zero duration should yield confidence 0");

        // Valid input → confidence = 1.0
        let valid = make_1080p_av1(60.0);
        let r2 = est.estimate(&valid, 4);
        assert!(
            r2.confidence > 0.8,
            "Valid input should yield high confidence"
        );
    }

    #[test]
    fn test_estimate_result_summary_non_empty() {
        let est = TranscodeEstimatorV2::default();
        let result = est.estimate(&make_1080p_av1(120.0), 8);
        let summary = result.summary();
        assert!(!summary.is_empty());
        assert!(result.is_reliable());
    }
}
