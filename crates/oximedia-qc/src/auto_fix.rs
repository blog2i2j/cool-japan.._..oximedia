//! Auto-fix suggestions and application for common QC failures.
//!
//! Provides the `QcAutoFixer` which analyzes a QC report and suggests
//! automated remediation actions. Each action can be simulated via
//! `QcAutoFixer::apply_fix` which returns a description of what would change.

#![allow(dead_code)]

use crate::qc_report::{FindingSeverity, QcReport};

/// Unique identifier for a specific QC check result.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CheckId(pub String);

impl CheckId {
    /// Creates a new check identifier from a check name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

impl std::fmt::Display for CheckId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// An automated fix action that can be applied to address a QC failure.
#[derive(Debug, Clone, PartialEq)]
pub enum AutoFixAction {
    /// Normalize audio loudness to the specified LUFS target.
    NormalizeLoudness {
        /// Target integrated loudness in LUFS (e.g. -23.0 for EBU R128).
        target_lufs: f32,
    },
    /// Re-encode the stream to achieve the target bitrate.
    AdjustBitrate {
        /// Target average bitrate in kbps.
        target_kbps: u32,
    },
    /// Remove leading and trailing black frames from the content.
    TrimBlackFrames,
    /// Convert the content to the specified color space.
    NormalizeColorSpace {
        /// Target color space name (e.g. "bt709", "bt2020").
        target_color_space: String,
    },
    /// Scale the video to the specified resolution.
    ScaleResolution {
        /// Target width in pixels.
        width: u32,
        /// Target height in pixels.
        height: u32,
    },
    /// Fix audio clipping by applying a limiter or gain reduction.
    FixAudioClipping {
        /// Maximum output level in dBFS (e.g. -1.0).
        ceiling_dbfs: f32,
    },
}

impl std::fmt::Display for AutoFixAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NormalizeLoudness { target_lufs } => {
                write!(f, "NormalizeLoudness(target={target_lufs:.1} LUFS)")
            }
            Self::AdjustBitrate { target_kbps } => {
                write!(f, "AdjustBitrate(target={target_kbps} kbps)")
            }
            Self::TrimBlackFrames => write!(f, "TrimBlackFrames"),
            Self::NormalizeColorSpace { target_color_space } => {
                write!(f, "NormalizeColorSpace(target={target_color_space})")
            }
            Self::ScaleResolution { width, height } => {
                write!(f, "ScaleResolution(target={width}x{height})")
            }
            Self::FixAudioClipping { ceiling_dbfs } => {
                write!(f, "FixAudioClipping(ceiling={ceiling_dbfs:.1} dBFS)")
            }
        }
    }
}

/// Outcome of applying an auto-fix action.
#[derive(Debug, Clone)]
pub struct FixResult {
    /// Whether the fix was successfully applied (or simulated).
    pub success: bool,
    /// Human-readable description of what was (or would be) changed.
    pub description: String,
    /// The action that was applied.
    pub action: AutoFixAction,
    /// Any caveats or warnings about the fix.
    pub caveats: Vec<String>,
}

impl FixResult {
    /// Creates a successful fix result.
    fn success(action: AutoFixAction, description: impl Into<String>) -> Self {
        Self {
            success: true,
            description: description.into(),
            action,
            caveats: Vec::new(),
        }
    }

    /// Creates a failed fix result.
    fn failure(action: AutoFixAction, description: impl Into<String>) -> Self {
        Self {
            success: false,
            description: description.into(),
            action,
            caveats: Vec::new(),
        }
    }

    /// Adds a caveat to this fix result.
    pub fn with_caveat(mut self, caveat: impl Into<String>) -> Self {
        self.caveats.push(caveat.into());
        self
    }
}

/// Analyzes QC reports and suggests automated fix actions.
///
/// The fixer examines the findings in a [`QcReport`] and maps them to
/// concrete [`AutoFixAction`] values. The suggested fixes can then be
/// applied (or simulated) through [`apply_fix`](QcAutoFixer::apply_fix).
#[derive(Debug, Clone, Default)]
pub struct QcAutoFixer {
    /// Default loudness target in LUFS. Defaults to -23.0 (EBU R128 broadcast).
    pub default_loudness_target_lufs: f32,
    /// Default bitrate target in kbps for bitrate violations. Defaults to 5000.
    pub default_bitrate_target_kbps: u32,
    /// Audio clipping ceiling in dBFS. Defaults to -1.0.
    pub audio_clipping_ceiling_dbfs: f32,
}

impl QcAutoFixer {
    /// Creates a new fixer with broadcast defaults.
    pub fn new() -> Self {
        Self {
            default_loudness_target_lufs: -23.0,
            default_bitrate_target_kbps: 5000,
            audio_clipping_ceiling_dbfs: -1.0,
        }
    }

    /// Sets the default loudness target.
    pub fn with_loudness_target(mut self, lufs: f32) -> Self {
        self.default_loudness_target_lufs = lufs;
        self
    }

    /// Sets the default bitrate target.
    pub fn with_bitrate_target(mut self, kbps: u32) -> Self {
        self.default_bitrate_target_kbps = kbps;
        self
    }

    /// Analyzes the QC report and returns a list of suggested fix actions.
    ///
    /// Each tuple contains the check identifier the suggestion relates to
    /// and the corresponding [`AutoFixAction`].
    pub fn suggest_fixes(&self, report: &QcReport) -> Vec<(CheckId, AutoFixAction)> {
        let mut suggestions: Vec<(CheckId, AutoFixAction)> = Vec::new();

        for result in report.all_results() {
            for finding in &result.findings {
                // Only suggest fixes for non-info findings
                if finding.severity == FindingSeverity::Info {
                    continue;
                }

                let check_id = CheckId::new(&finding.check_name);
                let msg_lower = finding.message.to_lowercase();

                // Loudness violations
                if msg_lower.contains("loudness")
                    || msg_lower.contains("lufs")
                    || msg_lower.contains("lkfs")
                {
                    suggestions.push((
                        check_id.clone(),
                        AutoFixAction::NormalizeLoudness {
                            target_lufs: self.default_loudness_target_lufs,
                        },
                    ));
                }

                // Bitrate violations
                if msg_lower.contains("bitrate") || msg_lower.contains("bit rate") {
                    suggestions.push((
                        check_id.clone(),
                        AutoFixAction::AdjustBitrate {
                            target_kbps: self.default_bitrate_target_kbps,
                        },
                    ));
                }

                // Black frame violations
                if msg_lower.contains("black frame")
                    || msg_lower.contains("blackframe")
                    || msg_lower.contains("black segment")
                {
                    suggestions.push((check_id.clone(), AutoFixAction::TrimBlackFrames));
                }

                // Color space violations
                if msg_lower.contains("color space")
                    || msg_lower.contains("colour space")
                    || msg_lower.contains("colorspace")
                {
                    suggestions.push((
                        check_id.clone(),
                        AutoFixAction::NormalizeColorSpace {
                            target_color_space: "bt709".to_string(),
                        },
                    ));
                }

                // Audio clipping
                if msg_lower.contains("clip")
                    || msg_lower.contains("saturation")
                    || msg_lower.contains("overload")
                {
                    suggestions.push((
                        check_id.clone(),
                        AutoFixAction::FixAudioClipping {
                            ceiling_dbfs: self.audio_clipping_ceiling_dbfs,
                        },
                    ));
                }
            }
        }

        // Deduplicate by CheckId + action type (keep first occurrence)
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        suggestions.retain(|(id, action)| {
            let key = format!("{}:{}", id, action);
            seen.insert(key)
        });

        suggestions
    }

    /// Simulates applying the given fix action and returns what would be changed.
    ///
    /// In a production integration this would call into the transcoding pipeline.
    /// This implementation returns a descriptive result without modifying any files.
    pub fn apply_fix(&self, action: &AutoFixAction) -> FixResult {
        match action {
            AutoFixAction::NormalizeLoudness { target_lufs } => FixResult::success(
                action.clone(),
                format!(
                    "Would apply integrated loudness normalization to {target_lufs:.1} LUFS \
                         using EBU R128 two-pass analysis and a linear gain adjustment. \
                         True peak ceiling of -1.0 dBTP will be enforced."
                ),
            )
            .with_caveat(
                "Dynamic range may be reduced if source loudness is significantly above target.",
            ),

            AutoFixAction::AdjustBitrate { target_kbps } => FixResult::success(
                action.clone(),
                format!(
                    "Would re-encode video stream at {target_kbps} kbps average bitrate \
                         using constrained VBR mode with ±20% tolerance."
                ),
            )
            .with_caveat("Quality may degrade if re-encoding from a lossy source."),

            AutoFixAction::TrimBlackFrames => FixResult::success(
                action.clone(),
                "Would scan the first and last 10 seconds of the file for black frames \
                 (luma < 16/255) and trim any leading/trailing black segments \
                 longer than 2 frames."
                    .to_string(),
            ),

            AutoFixAction::NormalizeColorSpace { target_color_space } => FixResult::success(
                action.clone(),
                format!(
                    "Would apply colorspace conversion to {target_color_space} using \
                     Bradford chromatic adaptation and signaling update in container metadata."
                ),
            )
            .with_caveat("Verify HDR metadata is updated if converting from HDR color space."),

            AutoFixAction::ScaleResolution { width, height } => FixResult::success(
                action.clone(),
                format!(
                    "Would scale video to {width}x{height} using Lanczos3 resampling \
                     with correct SAR/DAR signaling."
                ),
            ),

            AutoFixAction::FixAudioClipping { ceiling_dbfs } => FixResult::success(
                action.clone(),
                format!(
                    "Would apply a true peak limiter at {ceiling_dbfs:.1} dBFS using \
                     4x oversampling. Gain reduction events will be logged."
                ),
            )
            .with_caveat("Aggressive limiting may introduce distortion if clipping is severe."),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qc_report::{FindingSeverity, QcCheckResult, QcFinding, QcReport};

    fn make_report_with_finding(
        check_name: &str,
        severity: FindingSeverity,
        msg: &str,
    ) -> QcReport {
        let mut report = QcReport::new();
        let mut result = QcCheckResult::pass(check_name);
        result.add_finding(QcFinding::new(check_name, severity, msg));
        if severity >= FindingSeverity::Error {
            // recreate as fail
        }
        report.add_result(result);
        report
    }

    #[test]
    fn test_suggest_fixes_loudness() {
        let fixer = QcAutoFixer::new();
        let report = make_report_with_finding(
            "audio_loudness",
            FindingSeverity::Error,
            "Loudness is -18 LUFS, must be -23 LUFS",
        );
        let fixes = fixer.suggest_fixes(&report);
        assert!(!fixes.is_empty());
        assert!(fixes
            .iter()
            .any(|(_, a)| matches!(a, AutoFixAction::NormalizeLoudness { .. })));
    }

    #[test]
    fn test_suggest_fixes_bitrate() {
        let fixer = QcAutoFixer::new();
        let report = make_report_with_finding(
            "video_bitrate",
            FindingSeverity::Warning,
            "Bitrate exceeds maximum allowed",
        );
        let fixes = fixer.suggest_fixes(&report);
        assert!(fixes
            .iter()
            .any(|(_, a)| matches!(a, AutoFixAction::AdjustBitrate { .. })));
    }

    #[test]
    fn test_suggest_fixes_black_frames() {
        let fixer = QcAutoFixer::new();
        let report = make_report_with_finding(
            "black_silence",
            FindingSeverity::Warning,
            "Black frame segment detected at start",
        );
        let fixes = fixer.suggest_fixes(&report);
        assert!(fixes
            .iter()
            .any(|(_, a)| matches!(a, AutoFixAction::TrimBlackFrames)));
    }

    #[test]
    fn test_suggest_fixes_color_space() {
        let fixer = QcAutoFixer::new();
        let report = make_report_with_finding(
            "color_qc",
            FindingSeverity::Error,
            "Color space is not bt709",
        );
        let fixes = fixer.suggest_fixes(&report);
        assert!(fixes
            .iter()
            .any(|(_, a)| matches!(a, AutoFixAction::NormalizeColorSpace { .. })));
    }

    #[test]
    fn test_suggest_fixes_audio_clipping() {
        let fixer = QcAutoFixer::new();
        let report = make_report_with_finding(
            "audio_clip",
            FindingSeverity::Error,
            "Audio clipping detected",
        );
        let fixes = fixer.suggest_fixes(&report);
        assert!(fixes
            .iter()
            .any(|(_, a)| matches!(a, AutoFixAction::FixAudioClipping { .. })));
    }

    #[test]
    fn test_suggest_fixes_info_ignored() {
        let fixer = QcAutoFixer::new();
        let report = make_report_with_finding(
            "check",
            FindingSeverity::Info,
            "Loudness is -23 LUFS (informational)",
        );
        let fixes = fixer.suggest_fixes(&report);
        // Info findings should not generate fix suggestions
        assert!(fixes.is_empty());
    }

    #[test]
    fn test_apply_fix_normalize_loudness() {
        let fixer = QcAutoFixer::new();
        let action = AutoFixAction::NormalizeLoudness { target_lufs: -23.0 };
        let result = fixer.apply_fix(&action);
        assert!(result.success);
        assert!(result.description.contains("-23.0 LUFS"));
    }

    #[test]
    fn test_apply_fix_adjust_bitrate() {
        let fixer = QcAutoFixer::new();
        let action = AutoFixAction::AdjustBitrate { target_kbps: 8000 };
        let result = fixer.apply_fix(&action);
        assert!(result.success);
        assert!(result.description.contains("8000 kbps"));
    }

    #[test]
    fn test_apply_fix_trim_black_frames() {
        let fixer = QcAutoFixer::new();
        let result = fixer.apply_fix(&AutoFixAction::TrimBlackFrames);
        assert!(result.success);
        assert!(result.description.contains("black frames"));
    }

    #[test]
    fn test_apply_fix_normalize_color_space() {
        let fixer = QcAutoFixer::new();
        let action = AutoFixAction::NormalizeColorSpace {
            target_color_space: "bt2020".to_string(),
        };
        let result = fixer.apply_fix(&action);
        assert!(result.success);
        assert!(result.description.contains("bt2020"));
    }

    #[test]
    fn test_apply_fix_scale_resolution() {
        let fixer = QcAutoFixer::new();
        let action = AutoFixAction::ScaleResolution {
            width: 1920,
            height: 1080,
        };
        let result = fixer.apply_fix(&action);
        assert!(result.success);
        assert!(result.description.contains("1920x1080"));
    }

    #[test]
    fn test_apply_fix_audio_clipping() {
        let fixer = QcAutoFixer::new();
        let action = AutoFixAction::FixAudioClipping { ceiling_dbfs: -2.0 };
        let result = fixer.apply_fix(&action);
        assert!(result.success);
        assert!(result.description.contains("-2.0 dBFS"));
    }

    #[test]
    fn test_fix_result_with_caveat() {
        let fixer = QcAutoFixer::new();
        let action = AutoFixAction::NormalizeLoudness { target_lufs: -23.0 };
        let result = fixer.apply_fix(&action);
        assert!(!result.caveats.is_empty());
    }

    #[test]
    fn test_suggest_fixes_deduplication() {
        let fixer = QcAutoFixer::new();
        let mut report = QcReport::new();
        let mut r1 = QcCheckResult::pass("audio_loudness");
        r1.add_finding(QcFinding::new(
            "audio_loudness",
            FindingSeverity::Warning,
            "Loudness too high, LUFS issue",
        ));
        r1.add_finding(QcFinding::new(
            "audio_loudness",
            FindingSeverity::Error,
            "Loudness out of range, LUFS violation",
        ));
        report.add_result(r1);
        let fixes = fixer.suggest_fixes(&report);
        // Should deduplicate identical action types for same check
        let loudness_count = fixes
            .iter()
            .filter(|(id, a)| {
                id.0 == "audio_loudness" && matches!(a, AutoFixAction::NormalizeLoudness { .. })
            })
            .count();
        assert_eq!(loudness_count, 1);
    }

    #[test]
    fn test_check_id_display() {
        let id = CheckId::new("audio_loudness");
        assert_eq!(id.to_string(), "audio_loudness");
    }

    #[test]
    fn test_auto_fix_action_display() {
        let action = AutoFixAction::NormalizeLoudness { target_lufs: -23.0 };
        assert!(action.to_string().contains("LUFS"));
    }

    // ── Additional auto-fix tests ──────────────────────────────────────────

    #[test]
    fn test_fixer_builder_loudness_target() {
        let fixer = QcAutoFixer::new().with_loudness_target(-16.0);
        assert!((fixer.default_loudness_target_lufs - (-16.0)).abs() < 1e-6);
    }

    #[test]
    fn test_fixer_builder_bitrate_target() {
        let fixer = QcAutoFixer::new().with_bitrate_target(12_000);
        assert_eq!(fixer.default_bitrate_target_kbps, 12_000);
    }

    #[test]
    fn test_suggest_fixes_returns_empty_for_empty_report() {
        let fixer = QcAutoFixer::new();
        let report = QcReport::new();
        let fixes = fixer.suggest_fixes(&report);
        assert!(fixes.is_empty());
    }

    #[test]
    fn test_suggest_fixes_lkfs_alias() {
        // "LKFS" is an alias for "LUFS" — should still trigger NormalizeLoudness
        let fixer = QcAutoFixer::new();
        let report = make_report_with_finding(
            "audio_lkfs",
            FindingSeverity::Error,
            "LKFS measurement is out of range",
        );
        let fixes = fixer.suggest_fixes(&report);
        assert!(fixes
            .iter()
            .any(|(_, a)| matches!(a, AutoFixAction::NormalizeLoudness { .. })));
    }

    #[test]
    fn test_suggest_fixes_colorspace_keyword() {
        let fixer = QcAutoFixer::new();
        let report = make_report_with_finding(
            "colorspace_check",
            FindingSeverity::Warning,
            "colorspace does not match delivery spec",
        );
        let fixes = fixer.suggest_fixes(&report);
        assert!(fixes
            .iter()
            .any(|(_, a)| matches!(a, AutoFixAction::NormalizeColorSpace { .. })));
    }

    #[test]
    fn test_suggest_fixes_saturation_triggers_clip_fix() {
        let fixer = QcAutoFixer::new();
        let report = make_report_with_finding(
            "peak_check",
            FindingSeverity::Error,
            "Audio saturation detected at 0 dBFS",
        );
        let fixes = fixer.suggest_fixes(&report);
        assert!(fixes
            .iter()
            .any(|(_, a)| matches!(a, AutoFixAction::FixAudioClipping { .. })));
    }

    #[test]
    fn test_apply_fix_loudness_has_true_peak_caveat() {
        let fixer = QcAutoFixer::new();
        let action = AutoFixAction::NormalizeLoudness { target_lufs: -23.0 };
        let result = fixer.apply_fix(&action);
        // The description should mention EBU R128 or true peak
        assert!(
            result.description.to_lowercase().contains("r128")
                || result.description.to_lowercase().contains("true peak"),
            "Expected true peak or R128 mention in description"
        );
    }

    #[test]
    fn test_apply_fix_bitrate_has_cbr_or_vbr_caveat() {
        let fixer = QcAutoFixer::new();
        let action = AutoFixAction::AdjustBitrate { target_kbps: 5000 };
        let result = fixer.apply_fix(&action);
        assert!(
            !result.caveats.is_empty(),
            "Bitrate fix should include a caveat"
        );
    }

    #[test]
    fn test_apply_fix_color_space_has_hdr_caveat() {
        let fixer = QcAutoFixer::new();
        let action = AutoFixAction::NormalizeColorSpace {
            target_color_space: "bt709".to_string(),
        };
        let result = fixer.apply_fix(&action);
        // Should warn about HDR metadata when converting color space
        assert!(
            result
                .caveats
                .iter()
                .any(|c| c.to_lowercase().contains("hdr")),
            "NormalizeColorSpace should mention HDR metadata in caveats"
        );
    }

    #[test]
    fn test_check_id_equality() {
        let a = CheckId::new("video_bitrate");
        let b = CheckId::new("video_bitrate");
        let c = CheckId::new("audio_loudness");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_auto_fix_action_display_variants() {
        let actions = vec![
            AutoFixAction::AdjustBitrate { target_kbps: 5000 },
            AutoFixAction::TrimBlackFrames,
            AutoFixAction::NormalizeColorSpace {
                target_color_space: "bt709".to_string(),
            },
            AutoFixAction::ScaleResolution {
                width: 1920,
                height: 1080,
            },
            AutoFixAction::FixAudioClipping { ceiling_dbfs: -1.0 },
        ];
        for action in &actions {
            let display = action.to_string();
            assert!(
                !display.is_empty(),
                "Display string should not be empty for {action:?}"
            );
        }
    }

    #[test]
    fn test_suggest_fixes_no_duplicates_across_results() {
        // Two different check results, each with a loudness finding
        // — after deduplication both should be present (different check IDs).
        let fixer = QcAutoFixer::new();
        let mut report = QcReport::new();
        let mut r1 = QcCheckResult::pass("audio_loudness_1");
        r1.add_finding(QcFinding::new(
            "audio_loudness_1",
            FindingSeverity::Error,
            "LUFS too high",
        ));
        let mut r2 = QcCheckResult::pass("audio_loudness_2");
        r2.add_finding(QcFinding::new(
            "audio_loudness_2",
            FindingSeverity::Error,
            "LUFS too low",
        ));
        report.add_result(r1);
        report.add_result(r2);
        let fixes = fixer.suggest_fixes(&report);
        let loudness_count = fixes
            .iter()
            .filter(|(_, a)| matches!(a, AutoFixAction::NormalizeLoudness { .. }))
            .count();
        // Different check IDs → two distinct (id, action) pairs
        assert_eq!(
            loudness_count, 2,
            "Different check IDs should not be deduplicated"
        );
    }
}
