#![allow(dead_code)]
//! Audio delivery specification types for broadcast, cinema, streaming, and
//! podcast targets.

// ---------------------------------------------------------------------------
// DeliveryTarget
// ---------------------------------------------------------------------------

/// Destination platform / medium for an audio deliverable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeliveryTarget {
    /// Broadcast (TV / radio) delivery.
    Broadcast,
    /// Cinema theatrical delivery.
    Cinema,
    /// Online video streaming platform.
    Streaming,
    /// Podcast / spoken-word audio.
    Podcast,
}

impl DeliveryTarget {
    /// Maximum integrated loudness in LKFS (Loudness, K-weighted, relative to
    /// Full Scale) as per common industry standards.
    ///
    /// | Target     | Standard / typical target |
    /// |------------|--------------------------|
    /// | Broadcast  | EBU R128 / ATSC A/85: −23 LKFS |
    /// | Cinema     | SMPTE ST 428-2: −27 LKFS |
    /// | Streaming  | Spotify / YouTube: −14 LKFS |
    /// | Podcast    | Apple Podcasts: −16 LKFS |
    #[must_use]
    pub fn max_loudness_lkfs(self) -> f32 {
        match self {
            Self::Broadcast => -23.0,
            Self::Cinema => -27.0,
            Self::Streaming => -14.0,
            Self::Podcast => -16.0,
        }
    }

    /// Maximum true-peak level in dBTP.
    #[must_use]
    pub fn max_true_peak_dbtp(self) -> f32 {
        match self {
            Self::Broadcast => -1.0,
            Self::Cinema => -2.0,
            Self::Streaming => -1.0,
            Self::Podcast => -1.0,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Broadcast => "broadcast",
            Self::Cinema => "cinema",
            Self::Streaming => "streaming",
            Self::Podcast => "podcast",
        }
    }
}

// ---------------------------------------------------------------------------
// AudioDeliverySpec
// ---------------------------------------------------------------------------

/// Full specification for an audio deliverable.
#[derive(Debug, Clone)]
pub struct AudioDeliverySpec {
    /// Delivery target type.
    pub target: DeliveryTarget,
    /// Number of audio channels (1 = mono, 2 = stereo, 6 = 5.1, etc.).
    pub channels: u8,
    /// Sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Bit depth (16, 24, 32 …).
    pub bit_depth: u8,
    /// Maximum integrated loudness in LKFS (overrides the target default when
    /// set explicitly).
    pub max_loudness_lkfs: f32,
    /// Maximum true-peak in dBTP.
    pub max_true_peak_dbtp: f32,
    /// Optional codec name (e.g. "PCM", "AAC", "AC-3").
    pub codec: Option<String>,
}

impl AudioDeliverySpec {
    /// Build a spec from the target's default values.
    #[must_use]
    pub fn from_target(target: DeliveryTarget, channels: u8, sample_rate_hz: u32) -> Self {
        Self {
            max_loudness_lkfs: target.max_loudness_lkfs(),
            max_true_peak_dbtp: target.max_true_peak_dbtp(),
            target,
            channels: channels.max(1),
            sample_rate_hz,
            bit_depth: 24,
            codec: None,
        }
    }

    /// Returns `true` if the spec is stereo (exactly 2 channels).
    #[must_use]
    pub fn is_stereo(&self) -> bool {
        self.channels == 2
    }

    /// Returns `true` if the spec is mono (exactly 1 channel).
    #[must_use]
    pub fn is_mono(&self) -> bool {
        self.channels == 1
    }

    /// Returns `true` if the spec is surround (more than 2 channels).
    #[must_use]
    pub fn is_surround(&self) -> bool {
        self.channels > 2
    }

    /// Attach a codec name and return `self`.
    #[must_use]
    pub fn with_codec(mut self, codec: impl Into<String>) -> Self {
        self.codec = Some(codec.into());
        self
    }
}

// ---------------------------------------------------------------------------
// DeliverySpecChecker
// ---------------------------------------------------------------------------

/// Reports whether a measured audio signal conforms to a [`AudioDeliverySpec`].
#[derive(Debug)]
pub struct DeliverySpecChecker {
    spec: AudioDeliverySpec,
}

/// Result of a conformance check.
#[derive(Debug, Clone)]
pub struct CheckResult {
    /// `true` if all measured values pass.
    pub passed: bool,
    /// List of non-conformances (empty when `passed` is `true`).
    pub violations: Vec<String>,
}

impl CheckResult {
    fn ok() -> Self {
        Self {
            passed: true,
            violations: vec![],
        }
    }
}

impl DeliverySpecChecker {
    /// Create a checker for the given spec.
    #[must_use]
    pub fn new(spec: AudioDeliverySpec) -> Self {
        Self { spec }
    }

    /// Check measured `integrated_lkfs` and `true_peak_dbtp` against the spec.
    ///
    /// Both values must be **≤** their respective spec limits (audio loudness
    /// values are negative, so "within limit" means not exceeding the target).
    #[must_use]
    pub fn check(&self, integrated_lkfs: f32, true_peak_dbtp: f32) -> CheckResult {
        let mut violations = Vec::new();

        // Integrated loudness: measured value should not exceed limit
        // (loudness is negative, so exceeding means being less negative, i.e. louder)
        if integrated_lkfs > self.spec.max_loudness_lkfs {
            violations.push(format!(
                "Integrated loudness {:.1} LKFS exceeds limit {:.1} LKFS",
                integrated_lkfs, self.spec.max_loudness_lkfs
            ));
        }

        // True-peak: measured value should not exceed limit
        if true_peak_dbtp > self.spec.max_true_peak_dbtp {
            violations.push(format!(
                "True-peak {:.1} dBTP exceeds limit {:.1} dBTP",
                true_peak_dbtp, self.spec.max_true_peak_dbtp
            ));
        }

        if violations.is_empty() {
            CheckResult::ok()
        } else {
            CheckResult {
                passed: false,
                violations,
            }
        }
    }

    /// Reference to the spec this checker uses.
    #[must_use]
    pub fn spec(&self) -> &AudioDeliverySpec {
        &self.spec
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_loudness() {
        assert!((DeliveryTarget::Broadcast.max_loudness_lkfs() - (-23.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cinema_loudness() {
        assert!((DeliveryTarget::Cinema.max_loudness_lkfs() - (-27.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_streaming_loudness() {
        assert!((DeliveryTarget::Streaming.max_loudness_lkfs() - (-14.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_podcast_loudness() {
        assert!((DeliveryTarget::Podcast.max_loudness_lkfs() - (-16.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_delivery_target_labels() {
        assert_eq!(DeliveryTarget::Broadcast.label(), "broadcast");
        assert_eq!(DeliveryTarget::Cinema.label(), "cinema");
        assert_eq!(DeliveryTarget::Streaming.label(), "streaming");
        assert_eq!(DeliveryTarget::Podcast.label(), "podcast");
    }

    #[test]
    fn test_spec_is_stereo() {
        let spec = AudioDeliverySpec::from_target(DeliveryTarget::Broadcast, 2, 48_000);
        assert!(spec.is_stereo());
        assert!(!spec.is_mono());
        assert!(!spec.is_surround());
    }

    #[test]
    fn test_spec_is_mono() {
        let spec = AudioDeliverySpec::from_target(DeliveryTarget::Podcast, 1, 44_100);
        assert!(spec.is_mono());
        assert!(!spec.is_stereo());
    }

    #[test]
    fn test_spec_is_surround() {
        let spec = AudioDeliverySpec::from_target(DeliveryTarget::Cinema, 6, 48_000);
        assert!(spec.is_surround());
    }

    #[test]
    fn test_spec_with_codec() {
        let spec =
            AudioDeliverySpec::from_target(DeliveryTarget::Streaming, 2, 48_000).with_codec("AAC");
        assert_eq!(spec.codec.expect("codec should be valid"), "AAC");
    }

    #[test]
    fn test_checker_passes_compliant_signal() {
        let spec = AudioDeliverySpec::from_target(DeliveryTarget::Broadcast, 2, 48_000);
        let checker = DeliverySpecChecker::new(spec);
        let result = checker.check(-24.0, -2.0);
        assert!(result.passed);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_checker_fails_loud_signal() {
        let spec = AudioDeliverySpec::from_target(DeliveryTarget::Broadcast, 2, 48_000);
        let checker = DeliverySpecChecker::new(spec);
        let result = checker.check(-20.0, -2.0); // -20 > -23: too loud
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 1);
    }

    #[test]
    fn test_checker_fails_high_peak() {
        let spec = AudioDeliverySpec::from_target(DeliveryTarget::Broadcast, 2, 48_000);
        let checker = DeliverySpecChecker::new(spec);
        let result = checker.check(-24.0, 0.0); // 0 dBTP > -1 dBTP
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 1);
    }

    #[test]
    fn test_checker_fails_both_violations() {
        let spec = AudioDeliverySpec::from_target(DeliveryTarget::Broadcast, 2, 48_000);
        let checker = DeliverySpecChecker::new(spec);
        let result = checker.check(-18.0, 0.5);
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 2);
    }

    #[test]
    fn test_spec_min_channel_clamped() {
        let spec = AudioDeliverySpec::from_target(DeliveryTarget::Podcast, 0, 44_100);
        assert_eq!(spec.channels, 1);
    }
}
