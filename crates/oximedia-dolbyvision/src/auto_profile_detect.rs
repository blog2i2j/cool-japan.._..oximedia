//! Automatic Dolby Vision profile detection from RPU header fields.
//!
//! Analyzes RPU header flags and VDR sequence information to determine
//! which DV profile (5, 7, 8, 8.1, 8.4) a given RPU conforms to.

use crate::{DolbyVisionRpu, Profile};

/// A detected Dolby Vision profile with confidence score and rationale.
#[derive(Debug, Clone)]
pub struct DetectedProfile {
    /// The detected profile.
    pub profile: Profile,
    /// Confidence score in the range 0.0 to 1.0.
    pub confidence: f64,
    /// Per-feature rationale explaining why each flag matched or mismatched.
    pub rationale: Vec<FeatureRationale>,
}

impl DetectedProfile {
    /// Returns `true` when confidence exceeds the given threshold.
    #[must_use]
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }

    /// Returns a summary string listing profile and confidence.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Profile {:?} (confidence {:.1}%)",
            self.profile,
            self.confidence * 100.0
        )
    }
}

/// Rationale for a single feature check during profile detection.
#[derive(Debug, Clone)]
pub struct FeatureRationale {
    /// Name of the feature/flag being checked.
    pub feature_name: String,
    /// Whether the feature matched the expected value for the profile.
    pub matched: bool,
    /// Human-readable explanation.
    pub explanation: String,
}

/// RPU header fields relevant to profile detection.
///
/// These fields are extracted from the RPU header and VDR sequence info.
/// When fields are unknown (not available), set them to `None` and the
/// detector will reduce confidence accordingly.
#[derive(Debug, Clone, Default)]
pub struct RpuHeaderFields {
    /// VDR RPU profile field (from RPU header).
    pub vdr_rpu_profile: Option<u8>,
    /// Base layer video full range flag.
    pub bl_video_full_range_flag: Option<bool>,
    /// Enhancement layer spatial resampling filter flag.
    pub el_spatial_resampling_filter_flag: Option<bool>,
    /// Disable residual flag (true = no residual EL).
    pub disable_residual_flag: Option<bool>,
    /// Signal EOTF from VDR DM data (0 = BT.1886, 1 = PQ, 2 = HLG).
    pub signal_eotf: Option<u16>,
    /// Mapping color space (0 = YCbCr, 1 = RGB, 2 = IPT).
    pub mapping_color_space: Option<u8>,
    /// Component order (0 = RGB, 2 = IPT ordering).
    pub component_order: Option<u8>,
    /// NLQ (Non-Linear Quantization) method present.
    pub nlq_method_idc: Option<u8>,
    /// VDR bit depth.
    pub vdr_bit_depth: Option<u8>,
    /// Base layer bit depth.
    pub bl_bit_depth: Option<u8>,
    /// Enhancement layer bit depth.
    pub el_bit_depth: Option<u8>,
    /// Whether VDR DM metadata is present.
    pub vdr_dm_metadata_present: Option<bool>,
    /// Low-latency mode hint (from external signalling or container).
    pub low_latency_hint: Option<bool>,
}

impl RpuHeaderFields {
    /// Extract header fields from a parsed `DolbyVisionRpu`.
    #[must_use]
    pub fn from_rpu(rpu: &DolbyVisionRpu) -> Self {
        let vdr_seq = rpu.header.vdr_seq_info.as_ref();
        Self {
            vdr_rpu_profile: None, // Not directly stored in the parsed RPU header
            bl_video_full_range_flag: None,
            el_spatial_resampling_filter_flag: None,
            disable_residual_flag: None,
            signal_eotf: rpu.vdr_dm_data.as_ref().map(|dm| dm.signal_eotf),
            mapping_color_space: Some(rpu.header.mapping_color_space),
            component_order: Some(rpu.header.component_order),
            nlq_method_idc: None,
            vdr_bit_depth: vdr_seq.map(|s| s.vdr_bit_depth),
            bl_bit_depth: vdr_seq.map(|s| s.bl_bit_depth),
            el_bit_depth: vdr_seq.map(|s| s.el_bit_depth),
            vdr_dm_metadata_present: Some(rpu.vdr_dm_data.is_some()),
            low_latency_hint: None,
        }
    }
}

/// Profile detector that analyzes RPU header fields to identify the DV profile.
///
/// Each candidate profile is scored against a set of feature checks. The
/// profile with the highest score is returned as the detected profile.
#[derive(Debug, Clone)]
pub struct ProfileDetector {
    /// Minimum confidence threshold for a valid detection.
    pub min_confidence: f64,
}

impl Default for ProfileDetector {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
        }
    }
}

impl ProfileDetector {
    /// Create a new detector with the given minimum confidence threshold.
    #[must_use]
    pub fn new(min_confidence: f64) -> Self {
        Self {
            min_confidence: min_confidence.clamp(0.0, 1.0),
        }
    }

    /// Detect the most likely profile from RPU header fields.
    ///
    /// Returns `None` if no profile meets the minimum confidence threshold.
    #[must_use]
    pub fn detect(&self, fields: &RpuHeaderFields) -> Option<DetectedProfile> {
        let candidates = [
            Profile::Profile5,
            Profile::Profile7,
            Profile::Profile8,
            Profile::Profile8_1,
            Profile::Profile8_4,
        ];

        let mut best: Option<DetectedProfile> = None;

        for &profile in &candidates {
            let detected = self.score_profile(profile, fields);
            if detected.confidence >= self.min_confidence {
                if let Some(ref current_best) = best {
                    if detected.confidence > current_best.confidence {
                        best = Some(detected);
                    }
                } else {
                    best = Some(detected);
                }
            }
        }

        best
    }

    /// Detect all profiles with their scores, sorted by confidence descending.
    #[must_use]
    pub fn detect_all(&self, fields: &RpuHeaderFields) -> Vec<DetectedProfile> {
        let candidates = [
            Profile::Profile5,
            Profile::Profile7,
            Profile::Profile8,
            Profile::Profile8_1,
            Profile::Profile8_4,
        ];

        let mut results: Vec<DetectedProfile> = candidates
            .iter()
            .map(|&p| self.score_profile(p, fields))
            .collect();

        results.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Score a specific profile against the given header fields.
    fn score_profile(&self, profile: Profile, fields: &RpuHeaderFields) -> DetectedProfile {
        let mut rationale = Vec::new();
        let mut total_weight = 0.0_f64;
        let mut matched_weight = 0.0_f64;

        // Helper macro to add a check
        macro_rules! check {
            ($name:expr, $weight:expr, $field:expr, $expected:expr, $explanation:expr) => {
                if let Some(val) = $field {
                    let m = val == $expected;
                    total_weight += $weight;
                    if m {
                        matched_weight += $weight;
                    }
                    rationale.push(FeatureRationale {
                        feature_name: $name.to_string(),
                        matched: m,
                        explanation: if m {
                            format!("{} (matched: {:?})", $explanation, val)
                        } else {
                            format!(
                                "{} (expected {:?}, got {:?})",
                                $explanation, $expected, val
                            )
                        },
                    });
                }
            };
        }

        match profile {
            Profile::Profile5 => {
                // Profile 5: IPT-PQ, mapping_color_space = 2 (IPT), component_order = 2
                check!(
                    "mapping_color_space",
                    3.0,
                    fields.mapping_color_space,
                    2u8,
                    "IPT color space"
                );
                check!(
                    "component_order",
                    2.0,
                    fields.component_order,
                    2u8,
                    "IPT component order"
                );
                check!(
                    "signal_eotf",
                    2.0,
                    fields.signal_eotf,
                    1u16,
                    "PQ transfer function"
                );
                check!(
                    "disable_residual_flag",
                    1.5,
                    fields.disable_residual_flag,
                    true,
                    "No residual enhancement layer"
                );
                check!(
                    "el_spatial_resampling_filter_flag",
                    1.0,
                    fields.el_spatial_resampling_filter_flag,
                    false,
                    "No spatial resampling"
                );
                check!(
                    "vdr_rpu_profile",
                    3.0,
                    fields.vdr_rpu_profile,
                    5u8,
                    "RPU profile field"
                );
                check!(
                    "bl_bit_depth",
                    1.0,
                    fields.bl_bit_depth,
                    10u8,
                    "10-bit base layer"
                );
            }
            Profile::Profile7 => {
                // Profile 7: MEL, el_spatial = true, disable_residual = false
                check!(
                    "el_spatial_resampling_filter_flag",
                    3.0,
                    fields.el_spatial_resampling_filter_flag,
                    true,
                    "Spatial resampling for MEL"
                );
                check!(
                    "disable_residual_flag",
                    3.0,
                    fields.disable_residual_flag,
                    false,
                    "Residual EL present"
                );
                check!(
                    "mapping_color_space",
                    2.0,
                    fields.mapping_color_space,
                    1u8,
                    "RGB mapping color space"
                );
                check!(
                    "signal_eotf",
                    1.5,
                    fields.signal_eotf,
                    1u16,
                    "PQ transfer function"
                );
                check!(
                    "vdr_rpu_profile",
                    3.0,
                    fields.vdr_rpu_profile,
                    7u8,
                    "RPU profile field"
                );
                check!(
                    "el_bit_depth",
                    1.0,
                    fields.el_bit_depth,
                    8u8,
                    "8-bit enhancement layer"
                );
            }
            Profile::Profile8 => {
                // Profile 8: BL only, HDR10 compatible, disable_residual = true
                check!(
                    "disable_residual_flag",
                    3.0,
                    fields.disable_residual_flag,
                    true,
                    "No residual EL (BL only)"
                );
                check!(
                    "mapping_color_space",
                    2.0,
                    fields.mapping_color_space,
                    1u8,
                    "RGB mapping color space"
                );
                check!(
                    "signal_eotf",
                    2.5,
                    fields.signal_eotf,
                    1u16,
                    "PQ transfer function"
                );
                check!(
                    "el_spatial_resampling_filter_flag",
                    2.0,
                    fields.el_spatial_resampling_filter_flag,
                    false,
                    "No spatial resampling"
                );
                check!(
                    "vdr_rpu_profile",
                    3.0,
                    fields.vdr_rpu_profile,
                    8u8,
                    "RPU profile field"
                );
                check!(
                    "component_order",
                    1.5,
                    fields.component_order,
                    0u8,
                    "RGB component order"
                );
                // Distinguish from 8.1: no low-latency hint
                check!(
                    "low_latency_hint",
                    2.0,
                    fields.low_latency_hint,
                    false,
                    "Not low-latency mode"
                );
            }
            Profile::Profile8_1 => {
                // Profile 8.1: Low-latency variant of Profile 8
                check!(
                    "disable_residual_flag",
                    3.0,
                    fields.disable_residual_flag,
                    true,
                    "No residual EL (BL only)"
                );
                check!(
                    "mapping_color_space",
                    2.0,
                    fields.mapping_color_space,
                    1u8,
                    "RGB mapping color space"
                );
                check!(
                    "signal_eotf",
                    2.0,
                    fields.signal_eotf,
                    1u16,
                    "PQ transfer function"
                );
                check!(
                    "low_latency_hint",
                    3.0,
                    fields.low_latency_hint,
                    true,
                    "Low-latency mode"
                );
                check!(
                    "vdr_rpu_profile",
                    3.0,
                    fields.vdr_rpu_profile,
                    81u8,
                    "RPU profile field"
                );
                check!(
                    "el_spatial_resampling_filter_flag",
                    1.5,
                    fields.el_spatial_resampling_filter_flag,
                    false,
                    "No spatial resampling"
                );
            }
            Profile::Profile8_4 => {
                // Profile 8.4: HLG-based, signal_eotf = 2 (HLG)
                check!(
                    "signal_eotf",
                    3.5,
                    fields.signal_eotf,
                    2u16,
                    "HLG transfer function"
                );
                check!(
                    "disable_residual_flag",
                    2.5,
                    fields.disable_residual_flag,
                    true,
                    "No residual EL (BL only)"
                );
                check!(
                    "mapping_color_space",
                    2.0,
                    fields.mapping_color_space,
                    1u8,
                    "RGB mapping color space"
                );
                check!(
                    "el_spatial_resampling_filter_flag",
                    1.5,
                    fields.el_spatial_resampling_filter_flag,
                    false,
                    "No spatial resampling"
                );
                check!(
                    "vdr_rpu_profile",
                    3.0,
                    fields.vdr_rpu_profile,
                    84u8,
                    "RPU profile field"
                );
                check!(
                    "component_order",
                    1.0,
                    fields.component_order,
                    0u8,
                    "RGB component order"
                );
            }
        }

        // Confidence is the match ratio, but penalized when few fields are available.
        // A full profile check typically involves 5-7 fields; fewer fields means
        // lower certainty even if all match.
        let expected_min_checks = 4.0_f64;
        let field_coverage = (total_weight / expected_min_checks).min(1.0);
        let raw_confidence = if total_weight > 0.0 {
            matched_weight / total_weight
        } else {
            0.0
        };
        let confidence = (raw_confidence * field_coverage).clamp(0.0, 1.0);

        DetectedProfile {
            profile,
            confidence,
            rationale,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile5_fields() -> RpuHeaderFields {
        RpuHeaderFields {
            vdr_rpu_profile: Some(5),
            bl_video_full_range_flag: None,
            el_spatial_resampling_filter_flag: Some(false),
            disable_residual_flag: Some(true),
            signal_eotf: Some(1), // PQ
            mapping_color_space: Some(2), // IPT
            component_order: Some(2),
            nlq_method_idc: None,
            vdr_bit_depth: Some(12),
            bl_bit_depth: Some(10),
            el_bit_depth: None,
            vdr_dm_metadata_present: Some(true),
            low_latency_hint: None,
        }
    }

    fn profile7_fields() -> RpuHeaderFields {
        RpuHeaderFields {
            vdr_rpu_profile: Some(7),
            el_spatial_resampling_filter_flag: Some(true),
            disable_residual_flag: Some(false),
            signal_eotf: Some(1), // PQ
            mapping_color_space: Some(1), // RGB
            component_order: Some(0),
            el_bit_depth: Some(8),
            vdr_dm_metadata_present: Some(true),
            ..Default::default()
        }
    }

    fn profile8_fields() -> RpuHeaderFields {
        RpuHeaderFields {
            vdr_rpu_profile: Some(8),
            el_spatial_resampling_filter_flag: Some(false),
            disable_residual_flag: Some(true),
            signal_eotf: Some(1), // PQ
            mapping_color_space: Some(1), // RGB
            component_order: Some(0),
            low_latency_hint: Some(false),
            vdr_dm_metadata_present: Some(true),
            ..Default::default()
        }
    }

    fn profile8_1_fields() -> RpuHeaderFields {
        RpuHeaderFields {
            vdr_rpu_profile: Some(81),
            el_spatial_resampling_filter_flag: Some(false),
            disable_residual_flag: Some(true),
            signal_eotf: Some(1), // PQ
            mapping_color_space: Some(1), // RGB
            component_order: Some(0),
            low_latency_hint: Some(true),
            vdr_dm_metadata_present: Some(true),
            ..Default::default()
        }
    }

    fn profile8_4_fields() -> RpuHeaderFields {
        RpuHeaderFields {
            vdr_rpu_profile: Some(84),
            el_spatial_resampling_filter_flag: Some(false),
            disable_residual_flag: Some(true),
            signal_eotf: Some(2), // HLG
            mapping_color_space: Some(1), // RGB
            component_order: Some(0),
            vdr_dm_metadata_present: Some(true),
            ..Default::default()
        }
    }

    #[test]
    fn test_detect_profile5() {
        let detector = ProfileDetector::default();
        let result = detector.detect(&profile5_fields());
        assert!(result.is_some());
        let detected = result.expect("detection should succeed");
        assert_eq!(detected.profile, Profile::Profile5);
        assert!(detected.confidence > 0.8, "confidence={}", detected.confidence);
    }

    #[test]
    fn test_detect_profile7() {
        let detector = ProfileDetector::default();
        let result = detector.detect(&profile7_fields());
        assert!(result.is_some());
        let detected = result.expect("detection should succeed");
        assert_eq!(detected.profile, Profile::Profile7);
        assert!(detected.confidence > 0.8, "confidence={}", detected.confidence);
    }

    #[test]
    fn test_detect_profile8() {
        let detector = ProfileDetector::default();
        let result = detector.detect(&profile8_fields());
        assert!(result.is_some());
        let detected = result.expect("detection should succeed");
        assert_eq!(detected.profile, Profile::Profile8);
        assert!(detected.confidence > 0.8, "confidence={}", detected.confidence);
    }

    #[test]
    fn test_detect_profile8_1() {
        let detector = ProfileDetector::default();
        let result = detector.detect(&profile8_1_fields());
        assert!(result.is_some());
        let detected = result.expect("detection should succeed");
        assert_eq!(detected.profile, Profile::Profile8_1);
        assert!(detected.confidence > 0.7, "confidence={}", detected.confidence);
    }

    #[test]
    fn test_detect_profile8_4() {
        let detector = ProfileDetector::default();
        let result = detector.detect(&profile8_4_fields());
        assert!(result.is_some());
        let detected = result.expect("detection should succeed");
        assert_eq!(detected.profile, Profile::Profile8_4);
        assert!(detected.confidence > 0.8, "confidence={}", detected.confidence);
    }

    #[test]
    fn test_detect_all_returns_sorted() {
        let detector = ProfileDetector::default();
        let results = detector.detect_all(&profile5_fields());
        assert_eq!(results.len(), 5);
        // First should be Profile5
        assert_eq!(results[0].profile, Profile::Profile5);
        // Sorted by confidence descending
        for w in results.windows(2) {
            assert!(
                w[0].confidence >= w[1].confidence,
                "{:?} >= {:?}",
                w[0].confidence,
                w[1].confidence
            );
        }
    }

    #[test]
    fn test_empty_fields_low_confidence() {
        let detector = ProfileDetector::new(0.01);
        let fields = RpuHeaderFields::default();
        let results = detector.detect_all(&fields);
        // All profiles should have 0 confidence since no fields are set
        for r in &results {
            assert!(
                r.confidence < f64::EPSILON,
                "profile {:?} confidence should be 0, got {}",
                r.profile,
                r.confidence
            );
        }
    }

    #[test]
    fn test_is_confident_threshold() {
        let detected = DetectedProfile {
            profile: Profile::Profile8,
            confidence: 0.75,
            rationale: Vec::new(),
        };
        assert!(detected.is_confident(0.7));
        assert!(detected.is_confident(0.75));
        assert!(!detected.is_confident(0.76));
    }

    #[test]
    fn test_summary_format() {
        let detected = DetectedProfile {
            profile: Profile::Profile8,
            confidence: 0.95,
            rationale: Vec::new(),
        };
        let summary = detected.summary();
        assert!(summary.contains("Profile8"));
        assert!(summary.contains("95.0%"));
    }

    #[test]
    fn test_rationale_populated() {
        let detector = ProfileDetector::default();
        let detected = detector.score_profile(Profile::Profile5, &profile5_fields());
        assert!(!detected.rationale.is_empty());
        // All features should match for profile5_fields
        for r in &detected.rationale {
            assert!(
                r.matched,
                "feature '{}' should match: {}",
                r.feature_name, r.explanation
            );
        }
    }

    #[test]
    fn test_from_rpu_default_profile8() {
        let rpu = DolbyVisionRpu::new(Profile::Profile8);
        let fields = RpuHeaderFields::from_rpu(&rpu);
        assert_eq!(fields.mapping_color_space, Some(1)); // RGB
        assert_eq!(fields.component_order, Some(0)); // RGB order
    }

    #[test]
    fn test_from_rpu_profile5() {
        let rpu = DolbyVisionRpu::new(Profile::Profile5);
        let fields = RpuHeaderFields::from_rpu(&rpu);
        assert_eq!(fields.mapping_color_space, Some(2)); // IPT
        assert_eq!(fields.component_order, Some(2)); // IPT order
    }

    #[test]
    fn test_min_confidence_clamped() {
        let detector = ProfileDetector::new(1.5);
        assert!((detector.min_confidence - 1.0).abs() < f64::EPSILON);
        let detector2 = ProfileDetector::new(-0.5);
        assert!(detector2.min_confidence.abs() < f64::EPSILON);
    }

    #[test]
    fn test_partial_fields_reduce_confidence() {
        // Only mapping_color_space set -- partial information
        let fields = RpuHeaderFields {
            mapping_color_space: Some(2), // IPT -> suggests Profile5
            ..Default::default()
        };
        let detector = ProfileDetector::new(0.01);
        let result = detector.detect(&fields);
        assert!(result.is_some());
        let detected = result.expect("detection should succeed");
        // Should have lower confidence than full field set
        assert!(detected.confidence < 1.0);
    }
}
