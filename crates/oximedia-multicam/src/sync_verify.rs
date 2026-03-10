//! Sync pipeline verification for multi-camera sessions.
//!
//! Validates synchronisation results for consistency, confidence thresholds,
//! offset plausibility, and drift tolerance. This module acts as a quality gate
//! between the raw sync algorithms and downstream consumers (editing, switching,
//! composition).

use crate::sync::{SyncConfig, SyncMethod, SyncOffset, SyncResult};
use crate::{AngleId, MultiCamError, Result};

/// Severity level for verification diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticLevel {
    /// Informational — no action needed.
    Info,
    /// Warning — sync may be degraded but usable.
    Warning,
    /// Error — sync result should not be used as-is.
    Error,
}

/// A single diagnostic produced during sync verification.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Severity level.
    pub level: DiagnosticLevel,
    /// Which angle triggered this diagnostic, if applicable.
    pub angle: Option<AngleId>,
    /// Human-readable description.
    pub message: String,
}

impl Diagnostic {
    /// Create a new diagnostic.
    pub fn new(level: DiagnosticLevel, angle: Option<AngleId>, message: impl Into<String>) -> Self {
        Self {
            level,
            angle,
            message: message.into(),
        }
    }
}

/// Outcome of a sync pipeline verification pass.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Whether the sync result passed all critical checks.
    pub passed: bool,
    /// Number of angles that were verified.
    pub angles_checked: usize,
    /// Mean confidence across all offsets.
    pub mean_confidence: f64,
    /// Maximum absolute offset in frames.
    pub max_offset_frames: f64,
    /// Standard deviation of sub-frame offsets (consistency measure).
    pub offset_std_dev: f64,
    /// All diagnostics produced.
    pub diagnostics: Vec<Diagnostic>,
}

impl VerificationReport {
    /// Returns the number of error-level diagnostics.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Error)
            .count()
    }

    /// Returns the number of warning-level diagnostics.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Warning)
            .count()
    }

    /// Returns only the diagnostics at or above the given severity.
    #[must_use]
    pub fn diagnostics_at_level(&self, min_level: DiagnosticLevel) -> Vec<&Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.level >= min_level)
            .collect()
    }
}

/// Configuration for the sync pipeline verifier.
#[derive(Debug, Clone)]
pub struct VerifierConfig {
    /// Minimum acceptable per-offset confidence (0.0–1.0).
    pub min_confidence: f64,
    /// Minimum acceptable overall (mean) confidence.
    pub min_mean_confidence: f64,
    /// Maximum acceptable absolute offset in frames.
    /// Offsets beyond this indicate likely mis-sync.
    pub max_plausible_offset: f64,
    /// Maximum tolerable drift between any two non-reference offsets (frames).
    /// A large spread suggests inconsistent sync across angles.
    pub max_pairwise_drift: f64,
    /// Whether to require that the reference angle has zero offset.
    pub enforce_zero_reference: bool,
    /// Maximum tolerable standard deviation of sub-frame offsets.
    pub max_sub_frame_std_dev: f64,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            min_mean_confidence: 0.7,
            max_plausible_offset: 1200.0, // ±50 seconds at 24 fps
            max_pairwise_drift: 48.0,     // ±2 seconds at 24 fps
            enforce_zero_reference: true,
            max_sub_frame_std_dev: 0.4,
        }
    }
}

/// Verifies synchronisation pipeline output against quality criteria.
///
/// The verifier checks:
/// 1. **Confidence thresholds** — per-offset and aggregate.
/// 2. **Offset plausibility** — no offset exceeds the physical search window.
/// 3. **Reference consistency** — the reference angle's offset should be zero.
/// 4. **Pairwise drift** — the spread between any two offsets is bounded.
/// 5. **Sub-frame consistency** — sub-frame values are well-behaved.
/// 6. **Angle completeness** — every expected angle has an offset entry.
pub struct SyncPipelineVerifier {
    config: VerifierConfig,
}

impl SyncPipelineVerifier {
    /// Create a verifier with the given configuration.
    #[must_use]
    pub fn new(config: VerifierConfig) -> Self {
        Self { config }
    }

    /// Create a verifier from a `SyncConfig`, inheriting its thresholds.
    #[must_use]
    pub fn from_sync_config(sync_config: &SyncConfig) -> Self {
        Self {
            config: VerifierConfig {
                min_confidence: sync_config.min_confidence,
                min_mean_confidence: sync_config.min_confidence,
                max_plausible_offset: f64::from(sync_config.max_offset),
                ..VerifierConfig::default()
            },
        }
    }

    /// Verify a sync result.
    ///
    /// # Arguments
    /// * `result` — the sync result to verify.
    /// * `expected_angles` — the number of camera angles expected in the session.
    ///
    /// # Errors
    ///
    /// Returns an error if the sync result is fundamentally unusable (e.g. empty offsets).
    pub fn verify(
        &self,
        result: &SyncResult,
        expected_angles: usize,
    ) -> Result<VerificationReport> {
        if result.offsets.is_empty() {
            return Err(MultiCamError::SyncFailed(
                "Sync result has no offset entries".into(),
            ));
        }

        let mut diagnostics = Vec::new();

        // 1. Check angle completeness
        self.check_angle_completeness(result, expected_angles, &mut diagnostics);

        // 2. Check per-offset confidence
        self.check_per_offset_confidence(&result.offsets, &mut diagnostics);

        // 3. Check overall confidence
        let mean_conf = self.check_mean_confidence(result, &mut diagnostics);

        // 4. Check offset plausibility
        let max_offset = self.check_offset_plausibility(&result.offsets, &mut diagnostics);

        // 5. Check reference angle
        self.check_reference_angle(result, &mut diagnostics);

        // 6. Check pairwise drift
        self.check_pairwise_drift(&result.offsets, &mut diagnostics);

        // 7. Check sub-frame consistency
        let std_dev = self.check_sub_frame_consistency(&result.offsets, &mut diagnostics);

        // 8. Check method-specific constraints
        self.check_method_constraints(result, &mut diagnostics);

        let has_errors = diagnostics
            .iter()
            .any(|d| d.level == DiagnosticLevel::Error);

        Ok(VerificationReport {
            passed: !has_errors,
            angles_checked: result.offsets.len(),
            mean_confidence: mean_conf,
            max_offset_frames: max_offset,
            offset_std_dev: std_dev,
            diagnostics,
        })
    }

    fn check_angle_completeness(
        &self,
        result: &SyncResult,
        expected: usize,
        diags: &mut Vec<Diagnostic>,
    ) {
        let found = result.offsets.len();
        if found < expected {
            diags.push(Diagnostic::new(
                DiagnosticLevel::Warning,
                None,
                format!("Expected {expected} angle offsets, found {found}"),
            ));
        }
        // Check for duplicate angles
        let mut seen = Vec::with_capacity(found);
        for offset in &result.offsets {
            if seen.contains(&offset.angle) {
                diags.push(Diagnostic::new(
                    DiagnosticLevel::Error,
                    Some(offset.angle),
                    format!("Duplicate offset entry for angle {}", offset.angle),
                ));
            } else {
                seen.push(offset.angle);
            }
        }
    }

    fn check_per_offset_confidence(&self, offsets: &[SyncOffset], diags: &mut Vec<Diagnostic>) {
        for offset in offsets {
            if offset.confidence < self.config.min_confidence {
                let level = if offset.confidence < self.config.min_confidence * 0.5 {
                    DiagnosticLevel::Error
                } else {
                    DiagnosticLevel::Warning
                };
                diags.push(Diagnostic::new(
                    level,
                    Some(offset.angle),
                    format!(
                        "Angle {} confidence {:.3} below threshold {:.3}",
                        offset.angle, offset.confidence, self.config.min_confidence
                    ),
                ));
            }
        }
    }

    fn check_mean_confidence(&self, result: &SyncResult, diags: &mut Vec<Diagnostic>) -> f64 {
        if result.offsets.is_empty() {
            return 0.0;
        }
        let sum: f64 = result.offsets.iter().map(|o| o.confidence).sum();
        let mean = sum / result.offsets.len() as f64;

        if mean < self.config.min_mean_confidence {
            diags.push(Diagnostic::new(
                DiagnosticLevel::Warning,
                None,
                format!(
                    "Mean confidence {mean:.3} below threshold {:.3}",
                    self.config.min_mean_confidence
                ),
            ));
        }

        // Also check the overall result confidence
        if result.confidence < self.config.min_mean_confidence {
            diags.push(Diagnostic::new(
                DiagnosticLevel::Warning,
                None,
                format!(
                    "Overall sync confidence {:.3} below threshold {:.3}",
                    result.confidence, self.config.min_mean_confidence
                ),
            ));
        }

        mean
    }

    fn check_offset_plausibility(
        &self,
        offsets: &[SyncOffset],
        diags: &mut Vec<Diagnostic>,
    ) -> f64 {
        let mut max_abs = 0.0_f64;
        for offset in offsets {
            let abs_total = offset.total_frames().abs();
            if abs_total > max_abs {
                max_abs = abs_total;
            }
            if abs_total > self.config.max_plausible_offset {
                diags.push(Diagnostic::new(
                    DiagnosticLevel::Error,
                    Some(offset.angle),
                    format!(
                        "Angle {} offset {:.1} frames exceeds plausible limit {:.1}",
                        offset.angle, abs_total, self.config.max_plausible_offset
                    ),
                ));
            }
        }
        max_abs
    }

    fn check_reference_angle(&self, result: &SyncResult, diags: &mut Vec<Diagnostic>) {
        if !self.config.enforce_zero_reference {
            return;
        }
        let ref_angle = result.reference_angle;
        if let Some(ref_offset) = result.offsets.iter().find(|o| o.angle == ref_angle) {
            let total = ref_offset.total_frames().abs();
            if total > 0.01 {
                diags.push(Diagnostic::new(
                    DiagnosticLevel::Warning,
                    Some(ref_angle),
                    format!(
                        "Reference angle {} has non-zero offset {:.3} frames",
                        ref_angle, total
                    ),
                ));
            }
        } else {
            diags.push(Diagnostic::new(
                DiagnosticLevel::Warning,
                Some(ref_angle),
                format!("Reference angle {} not found in offset entries", ref_angle),
            ));
        }
    }

    fn check_pairwise_drift(&self, offsets: &[SyncOffset], diags: &mut Vec<Diagnostic>) {
        if offsets.len() < 2 {
            return;
        }
        for i in 0..offsets.len() {
            for j in (i + 1)..offsets.len() {
                let drift = (offsets[i].total_frames() - offsets[j].total_frames()).abs();
                if drift > self.config.max_pairwise_drift {
                    diags.push(Diagnostic::new(
                        DiagnosticLevel::Warning,
                        None,
                        format!(
                            "Pairwise drift between angle {} and angle {} is {:.1} frames (limit {:.1})",
                            offsets[i].angle,
                            offsets[j].angle,
                            drift,
                            self.config.max_pairwise_drift
                        ),
                    ));
                }
            }
        }
    }

    fn check_sub_frame_consistency(
        &self,
        offsets: &[SyncOffset],
        diags: &mut Vec<Diagnostic>,
    ) -> f64 {
        if offsets.len() < 2 {
            return 0.0;
        }
        let n = offsets.len() as f64;
        let mean_sf: f64 = offsets.iter().map(|o| o.sub_frame).sum::<f64>() / n;
        let variance: f64 = offsets
            .iter()
            .map(|o| {
                let d = o.sub_frame - mean_sf;
                d * d
            })
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt();

        if std_dev > self.config.max_sub_frame_std_dev {
            diags.push(Diagnostic::new(
                DiagnosticLevel::Info,
                None,
                format!(
                    "Sub-frame offset std dev {std_dev:.4} exceeds recommended {:.4}",
                    self.config.max_sub_frame_std_dev
                ),
            ));
        }

        // Check for out-of-range sub-frame values
        for offset in offsets {
            if offset.sub_frame < -1.0 || offset.sub_frame > 1.0 {
                diags.push(Diagnostic::new(
                    DiagnosticLevel::Error,
                    Some(offset.angle),
                    format!(
                        "Angle {} sub-frame value {:.4} is out of expected [-1.0, 1.0] range",
                        offset.angle, offset.sub_frame
                    ),
                ));
            }
        }

        std_dev
    }

    fn check_method_constraints(&self, result: &SyncResult, diags: &mut Vec<Diagnostic>) {
        match result.method {
            SyncMethod::Genlock => {
                // Genlock should produce very tight sync — all offsets near zero
                for offset in &result.offsets {
                    if offset.total_frames().abs() > 1.0 {
                        diags.push(Diagnostic::new(
                            DiagnosticLevel::Warning,
                            Some(offset.angle),
                            format!(
                                "Genlock sync: angle {} has offset {:.2} frames (expected <1.0)",
                                offset.angle,
                                offset.total_frames().abs()
                            ),
                        ));
                    }
                }
            }
            SyncMethod::Manual => {
                // Manual sync: just log an info diagnostic
                diags.push(Diagnostic::new(
                    DiagnosticLevel::Info,
                    None,
                    "Manual sync offsets — automated verification is limited",
                ));
            }
            _ => {}
        }
    }
}

impl Default for SyncPipelineVerifier {
    fn default() -> Self {
        Self::new(VerifierConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(offsets: Vec<SyncOffset>, confidence: f64) -> SyncResult {
        SyncResult {
            reference_angle: 0,
            offsets,
            confidence,
            method: SyncMethod::Audio,
        }
    }

    #[test]
    fn test_passing_verification() {
        let verifier = SyncPipelineVerifier::default();
        let result = make_result(
            vec![
                SyncOffset::new(0, 0, 0.0, 0.95),
                SyncOffset::new(1, 2, 0.3, 0.90),
                SyncOffset::new(2, -1, 0.1, 0.88),
            ],
            0.91,
        );
        let report = verifier.verify(&result, 3).expect("should succeed");
        assert!(report.passed);
        assert_eq!(report.error_count(), 0);
        assert_eq!(report.angles_checked, 3);
    }

    #[test]
    fn test_empty_offsets_error() {
        let verifier = SyncPipelineVerifier::default();
        let result = make_result(Vec::new(), 0.0);
        assert!(verifier.verify(&result, 2).is_err());
    }

    #[test]
    fn test_low_confidence_warning() {
        let verifier = SyncPipelineVerifier::default();
        let result = make_result(
            vec![
                SyncOffset::new(0, 0, 0.0, 0.95),
                SyncOffset::new(1, 1, 0.0, 0.4), // below 0.6 threshold
            ],
            0.67,
        );
        let report = verifier.verify(&result, 2).expect("should succeed");
        assert!(report.warning_count() > 0 || report.error_count() > 0);
    }

    #[test]
    fn test_very_low_confidence_error() {
        let verifier = SyncPipelineVerifier::default();
        let result = make_result(
            vec![
                SyncOffset::new(0, 0, 0.0, 0.95),
                SyncOffset::new(1, 1, 0.0, 0.2), // below 0.3 (half of 0.6)
            ],
            0.5,
        );
        let report = verifier.verify(&result, 2).expect("should succeed");
        assert!(!report.passed);
        assert!(report.error_count() > 0);
    }

    #[test]
    fn test_implausible_offset() {
        let verifier = SyncPipelineVerifier::default();
        let result = make_result(
            vec![
                SyncOffset::new(0, 0, 0.0, 0.95),
                SyncOffset::new(1, 5000, 0.0, 0.90), // way beyond 1200 frame limit
            ],
            0.92,
        );
        let report = verifier.verify(&result, 2).expect("should succeed");
        assert!(!report.passed);
        assert!(report.error_count() > 0);
    }

    #[test]
    fn test_duplicate_angle_error() {
        let verifier = SyncPipelineVerifier::default();
        let result = make_result(
            vec![
                SyncOffset::new(0, 0, 0.0, 0.95),
                SyncOffset::new(1, 2, 0.0, 0.90),
                SyncOffset::new(1, 3, 0.0, 0.88), // duplicate angle 1
            ],
            0.91,
        );
        let report = verifier.verify(&result, 3).expect("should succeed");
        assert!(!report.passed);
    }

    #[test]
    fn test_reference_non_zero_warning() {
        let verifier = SyncPipelineVerifier::default();
        let result = make_result(
            vec![
                SyncOffset::new(0, 5, 0.0, 0.95), // reference not at zero
                SyncOffset::new(1, 7, 0.0, 0.90),
            ],
            0.92,
        );
        let report = verifier.verify(&result, 2).expect("should succeed");
        assert!(report.warning_count() > 0);
    }

    #[test]
    fn test_pairwise_drift_warning() {
        let config = VerifierConfig {
            max_pairwise_drift: 10.0,
            enforce_zero_reference: false,
            ..VerifierConfig::default()
        };
        let verifier = SyncPipelineVerifier::new(config);
        let result = make_result(
            vec![
                SyncOffset::new(0, 0, 0.0, 0.95),
                SyncOffset::new(1, 20, 0.0, 0.90), // drift of 20 > limit 10
            ],
            0.92,
        );
        let report = verifier.verify(&result, 2).expect("should succeed");
        assert!(report.warning_count() > 0);
    }

    #[test]
    fn test_sub_frame_out_of_range() {
        let verifier = SyncPipelineVerifier::default();
        let result = make_result(
            vec![
                SyncOffset::new(0, 0, 0.0, 0.95),
                SyncOffset::new(1, 1, 2.5, 0.90), // sub_frame > 1.0
            ],
            0.92,
        );
        let report = verifier.verify(&result, 2).expect("should succeed");
        assert!(!report.passed);
    }

    #[test]
    fn test_genlock_tight_constraint() {
        let verifier = SyncPipelineVerifier::default();
        let result = SyncResult {
            reference_angle: 0,
            offsets: vec![
                SyncOffset::new(0, 0, 0.0, 0.99),
                SyncOffset::new(1, 3, 0.0, 0.99), // > 1.0 frame for genlock
            ],
            confidence: 0.99,
            method: SyncMethod::Genlock,
        };
        let report = verifier.verify(&result, 2).expect("should succeed");
        assert!(report.warning_count() > 0);
    }

    #[test]
    fn test_from_sync_config() {
        let sync_config = SyncConfig {
            min_confidence: 0.8,
            max_offset: 300,
            ..SyncConfig::default()
        };
        let verifier = SyncPipelineVerifier::from_sync_config(&sync_config);
        assert_eq!(verifier.config.min_confidence, 0.8);
        assert_eq!(verifier.config.max_plausible_offset, 300.0);
    }

    #[test]
    fn test_missing_angles_warning() {
        let verifier = SyncPipelineVerifier::default();
        let result = make_result(vec![SyncOffset::new(0, 0, 0.0, 0.95)], 0.95);
        let report = verifier.verify(&result, 3).expect("should succeed");
        assert!(report.warning_count() > 0);
    }

    #[test]
    fn test_diagnostics_at_level_filter() {
        let verifier = SyncPipelineVerifier::default();
        let result = SyncResult {
            reference_angle: 0,
            offsets: vec![
                SyncOffset::new(0, 0, 0.0, 0.95),
                SyncOffset::new(1, 1, 0.0, 0.90),
            ],
            confidence: 0.92,
            method: SyncMethod::Manual, // produces an Info diagnostic
        };
        let report = verifier.verify(&result, 2).expect("should succeed");
        let info_and_above = report.diagnostics_at_level(DiagnosticLevel::Info);
        assert!(!info_and_above.is_empty());
        let errors_only = report.diagnostics_at_level(DiagnosticLevel::Error);
        assert!(errors_only.is_empty());
    }

    #[test]
    fn test_max_offset_reported() {
        let verifier = SyncPipelineVerifier::default();
        let result = make_result(
            vec![
                SyncOffset::new(0, 0, 0.0, 0.95),
                SyncOffset::new(1, 10, 0.5, 0.90),
                SyncOffset::new(2, -5, 0.2, 0.88),
            ],
            0.91,
        );
        let report = verifier.verify(&result, 3).expect("should succeed");
        assert!((report.max_offset_frames - 10.5).abs() < 0.01);
    }
}
