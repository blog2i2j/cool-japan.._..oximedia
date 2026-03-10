//! Audio conformance checking for professional broadcast and streaming delivery.
//!
//! Validates audio specifications and loudness levels against common
//! broadcast standards: EBU R128, ATSC A/85, Netflix.

#![allow(dead_code)]

/// Audio format specification.
#[derive(Debug, Clone)]
pub struct AudioSpec {
    /// Sample rate in Hz (e.g. 48000).
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u8,
    /// Bit depth per sample (e.g. 16, 24).
    pub bit_depth: u8,
    /// Codec identifier (e.g. "PCM", "AAC").
    pub codec: String,
    /// Integrated loudness in LUFS, if measured.
    pub loudness_lufs: Option<f32>,
}

impl AudioSpec {
    /// Returns `true` if the spec meets broadcast minimum requirements
    /// (48 kHz sample rate, 16-bit or higher).
    #[must_use]
    pub fn is_broadcast(&self) -> bool {
        self.sample_rate == 48_000 && self.bit_depth >= 16
    }
}

/// Checks a measured audio spec against a target [`AudioSpec`].
pub struct AudioConformChecker {
    /// The target (delivery) specification.
    pub target: AudioSpec,
}

impl AudioConformChecker {
    /// Create a new checker with the given target spec.
    #[must_use]
    pub fn new(target: AudioSpec) -> Self {
        Self { target }
    }

    /// Check sample rate; returns an issue string if it does not match.
    #[must_use]
    pub fn check_sample_rate(&self, actual: u32) -> Option<String> {
        if actual == self.target.sample_rate {
            None
        } else {
            Some(format!(
                "Sample rate mismatch: expected {} Hz, found {} Hz",
                self.target.sample_rate, actual
            ))
        }
    }

    /// Check channel count; returns an issue string if it does not match.
    #[must_use]
    pub fn check_channels(&self, actual: u8) -> Option<String> {
        if actual == self.target.channels {
            None
        } else {
            Some(format!(
                "Channel count mismatch: expected {}, found {}",
                self.target.channels, actual
            ))
        }
    }

    /// Check bit depth; returns an issue string if it does not match.
    #[must_use]
    pub fn check_bit_depth(&self, actual: u8) -> Option<String> {
        if actual == self.target.bit_depth {
            None
        } else {
            Some(format!(
                "Bit depth mismatch: expected {}-bit, found {}-bit",
                self.target.bit_depth, actual
            ))
        }
    }

    /// Check loudness against the target spec's loudness (if set).
    /// Returns an issue string when the deviation exceeds 1 LU.
    #[must_use]
    pub fn check_loudness(&self, lufs: f32) -> Option<String> {
        if let Some(target_lufs) = self.target.loudness_lufs {
            if (lufs - target_lufs).abs() > 1.0 {
                return Some(format!(
                    "Loudness mismatch: expected {target_lufs:.1} LUFS, found {lufs:.1} LUFS"
                ));
            }
        }
        None
    }

    /// Run all checks against an actual [`AudioSpec`] and return the list of
    /// issue strings.
    #[must_use]
    pub fn check_all(&self, actual: &AudioSpec) -> Vec<String> {
        let mut issues = Vec::new();
        if let Some(msg) = self.check_sample_rate(actual.sample_rate) {
            issues.push(msg);
        }
        if let Some(msg) = self.check_channels(actual.channels) {
            issues.push(msg);
        }
        if let Some(msg) = self.check_bit_depth(actual.bit_depth) {
            issues.push(msg);
        }
        if let Some(lufs) = actual.loudness_lufs {
            if let Some(msg) = self.check_loudness(lufs) {
                issues.push(msg);
            }
        }
        issues
    }
}

/// Loudness delivery specification.
#[derive(Debug, Clone)]
pub struct LoudnessSpec {
    /// Integrated programme loudness target in LUFS.
    pub target_lufs: f32,
    /// Maximum true-peak level in dBTP.
    pub max_true_peak: f32,
    /// Maximum loudness range (LRA) in LU.
    pub max_lra: f32,
}

impl LoudnessSpec {
    /// EBU R128 specification: −23 LUFS, −1 dBTP, 20 LU LRA.
    #[must_use]
    pub fn ebu_r128() -> Self {
        Self {
            target_lufs: -23.0,
            max_true_peak: -1.0,
            max_lra: 20.0,
        }
    }

    /// ATSC A/85 (US broadcast) specification: −24 LUFS, −2 dBTP, 15 LU LRA.
    #[must_use]
    pub fn atsc_a85() -> Self {
        Self {
            target_lufs: -24.0,
            max_true_peak: -2.0,
            max_lra: 15.0,
        }
    }

    /// Netflix delivery specification: −27 LUFS, −2 dBTP, unlimited LRA.
    #[must_use]
    pub fn netflix() -> Self {
        Self {
            target_lufs: -27.0,
            max_true_peak: -2.0,
            max_lra: f32::MAX,
        }
    }
}

/// Result of a loudness conformance check.
#[derive(Debug, Clone)]
pub struct LoudnessConformResult {
    /// Whether the audio passes the loudness spec.
    pub passed: bool,
    /// Measured integrated loudness in LUFS.
    pub integrated_lufs: f32,
    /// Gain adjustment required to reach the target, in dB.
    pub gain_db: f32,
}

impl LoudnessConformResult {
    /// Compute the gain (in dB) needed to reach the given target spec.
    #[must_use]
    pub fn gain_needed(&self, target: &LoudnessSpec) -> f32 {
        target.target_lufs - self.integrated_lufs
    }
}

/// Check measured integrated loudness against a [`LoudnessSpec`].
///
/// Tolerance is ±1 LU.
#[must_use]
pub fn check_loudness(measured_lufs: f32, spec: &LoudnessSpec) -> LoudnessConformResult {
    let gain_db = spec.target_lufs - measured_lufs;
    let passed = gain_db.abs() <= 1.0;
    LoudnessConformResult {
        passed,
        integrated_lufs: measured_lufs,
        gain_db,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn broadcast_spec() -> AudioSpec {
        AudioSpec {
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            codec: "PCM".to_string(),
            loudness_lufs: Some(-23.0),
        }
    }

    #[test]
    fn test_is_broadcast_passes() {
        let spec = broadcast_spec();
        assert!(spec.is_broadcast());
    }

    #[test]
    fn test_is_broadcast_fails_sample_rate() {
        let spec = AudioSpec {
            sample_rate: 44_100,
            ..broadcast_spec()
        };
        assert!(!spec.is_broadcast());
    }

    #[test]
    fn test_is_broadcast_fails_bit_depth() {
        let spec = AudioSpec {
            bit_depth: 8,
            ..broadcast_spec()
        };
        assert!(!spec.is_broadcast());
    }

    #[test]
    fn test_check_sample_rate_match() {
        let checker = AudioConformChecker::new(broadcast_spec());
        assert!(checker.check_sample_rate(48_000).is_none());
    }

    #[test]
    fn test_check_sample_rate_mismatch() {
        let checker = AudioConformChecker::new(broadcast_spec());
        let issue = checker.check_sample_rate(44_100);
        assert!(issue.is_some());
        assert!(issue.expect("test expectation failed").contains("44100"));
    }

    #[test]
    fn test_check_channels_match() {
        let checker = AudioConformChecker::new(broadcast_spec());
        assert!(checker.check_channels(2).is_none());
    }

    #[test]
    fn test_check_channels_mismatch() {
        let checker = AudioConformChecker::new(broadcast_spec());
        let issue = checker.check_channels(6);
        assert!(issue.is_some());
    }

    #[test]
    fn test_check_bit_depth_match() {
        let checker = AudioConformChecker::new(broadcast_spec());
        assert!(checker.check_bit_depth(24).is_none());
    }

    #[test]
    fn test_check_bit_depth_mismatch() {
        let checker = AudioConformChecker::new(broadcast_spec());
        let issue = checker.check_bit_depth(16);
        assert!(issue.is_some());
        assert!(issue.expect("test expectation failed").contains("16-bit"));
    }

    #[test]
    fn test_check_loudness_within_tolerance() {
        let checker = AudioConformChecker::new(broadcast_spec());
        // target is -23.0, actual is -23.5 → within 1 LU
        assert!(checker.check_loudness(-23.5).is_none());
    }

    #[test]
    fn test_check_loudness_outside_tolerance() {
        let checker = AudioConformChecker::new(broadcast_spec());
        // target is -23.0, actual is -18.0 → 5 LU difference
        let issue = checker.check_loudness(-18.0);
        assert!(issue.is_some());
    }

    #[test]
    fn test_check_all_clean() {
        let spec = broadcast_spec();
        let checker = AudioConformChecker::new(spec.clone());
        let issues = checker.check_all(&spec);
        assert!(issues.is_empty(), "Expected no issues: {issues:?}");
    }

    #[test]
    fn test_check_all_multiple_issues() {
        let checker = AudioConformChecker::new(broadcast_spec());
        let bad = AudioSpec {
            sample_rate: 44_100,
            channels: 1,
            bit_depth: 16,
            codec: "AAC".to_string(),
            loudness_lufs: Some(-18.0),
        };
        let issues = checker.check_all(&bad);
        assert!(issues.len() >= 3, "Expected >=3 issues: {issues:?}");
    }

    #[test]
    fn test_ebu_r128_spec() {
        let spec = LoudnessSpec::ebu_r128();
        assert_eq!(spec.target_lufs, -23.0);
        assert_eq!(spec.max_true_peak, -1.0);
    }

    #[test]
    fn test_atsc_a85_spec() {
        let spec = LoudnessSpec::atsc_a85();
        assert_eq!(spec.target_lufs, -24.0);
    }

    #[test]
    fn test_netflix_spec() {
        let spec = LoudnessSpec::netflix();
        assert_eq!(spec.target_lufs, -27.0);
    }

    #[test]
    fn test_loudness_conform_passed() {
        let spec = LoudnessSpec::ebu_r128();
        let result = check_loudness(-23.0, &spec);
        assert!(result.passed);
        assert!((result.gain_db).abs() < 0.01);
    }

    #[test]
    fn test_loudness_conform_needs_gain() {
        let spec = LoudnessSpec::ebu_r128();
        let result = check_loudness(-30.0, &spec);
        assert!(!result.passed);
        // Need +7 dB to reach -23 LUFS
        assert!(
            (result.gain_needed(&spec) - 7.0).abs() < 0.01,
            "gain: {}",
            result.gain_needed(&spec)
        );
    }
}
