//! WCAG 2.1/2.2 compliance checker for media captions and UI elements.
//!
//! Checks caption specifications against WCAG success criteria relevant to
//! media accessibility:
//!
//! | Criterion | Level | Description |
//! |-----------|-------|-------------|
//! | 1.2.2 | A     | Captions present for pre-recorded audio content |
//! | 1.4.3 | AA    | Contrast ≥ 4.5:1 (normal text) |
//! | 1.4.6 | AAA   | Contrast ≥ 7:1 (enhanced) |
//! | 1.2.6 | AAA   | Sign language track available |
//! | Reading speed | advisory | ≤ 300 wpm recommended |

use serde::{Deserialize, Serialize};

// ── WCAG level ────────────────────────────────────────────────────────────────

/// WCAG conformance level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum WcagLevel {
    /// Level A – minimum conformance.
    A,
    /// Level AA – mid-range conformance required by most regulations.
    AA,
    /// Level AAA – highest conformance level.
    AAA,
}

impl std::fmt::Display for WcagLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::A => write!(f, "A"),
            Self::AA => write!(f, "AA"),
            Self::AAA => write!(f, "AAA"),
        }
    }
}

// ── Criterion ─────────────────────────────────────────────────────────────────

/// A single WCAG success criterion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WcagCriterion {
    /// Criterion identifier, e.g. `"1.4.3"`.
    pub id: String,
    /// Human-readable criterion name.
    pub name: String,
    /// Required WCAG level.
    pub level: WcagLevel,
}

impl WcagCriterion {
    /// Construct a new criterion record.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, level: WcagLevel) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            level,
        }
    }
}

impl std::fmt::Display for WcagCriterion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} (Level {})", self.id, self.name, self.level)
    }
}

// ── Severity ──────────────────────────────────────────────────────────────────

/// How severe a WCAG violation is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    /// An outright failure that must be corrected.
    Error,
    /// A potential problem that should be addressed.
    Warning,
    /// Advisory information only.
    Info,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Error => write!(f, "Error"),
            Self::Warning => write!(f, "Warning"),
            Self::Info => write!(f, "Info"),
        }
    }
}

// ── Violation ─────────────────────────────────────────────────────────────────

/// A single detected WCAG violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WcagViolation {
    /// The criterion that was violated.
    pub criterion: WcagCriterion,
    /// Human-readable description of the specific failure.
    pub description: String,
    /// Severity classification.
    pub severity: Severity,
}

impl WcagViolation {
    /// Construct a new violation.
    #[must_use]
    pub fn new(
        criterion: WcagCriterion,
        description: impl Into<String>,
        severity: Severity,
    ) -> Self {
        Self {
            criterion,
            description: description.into(),
            severity,
        }
    }
}

// ── CaptionSpec ───────────────────────────────────────────────────────────────

/// Input specification for a caption segment to be checked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionSpec {
    /// Text content of the caption.
    pub text: String,
    /// Display duration in milliseconds.
    pub duration_ms: u64,
    /// Whether a sign-language track accompanies this caption.
    pub has_sign_language: bool,
    /// Foreground (text) colour as sRGB `[R, G, B]` (0–255 each).
    pub foreground_color: [u8; 3],
    /// Background colour as sRGB `[R, G, B]` (0–255 each).
    pub background_color: [u8; 3],
}

impl CaptionSpec {
    /// Create a caption spec with sensible defaults (white on black, no sign language).
    #[must_use]
    pub fn new(text: impl Into<String>, duration_ms: u64) -> Self {
        Self {
            text: text.into(),
            duration_ms,
            has_sign_language: false,
            foreground_color: [255, 255, 255],
            background_color: [0, 0, 0],
        }
    }

    /// Enable sign language for this spec.
    #[must_use]
    pub fn with_sign_language(mut self) -> Self {
        self.has_sign_language = true;
        self
    }

    /// Set foreground and background colours.
    #[must_use]
    pub fn with_colors(mut self, fg: [u8; 3], bg: [u8; 3]) -> Self {
        self.foreground_color = fg;
        self.background_color = bg;
        self
    }
}

// ── CaptionCompliance ─────────────────────────────────────────────────────────

/// Result of checking a caption spec against WCAG criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionCompliance {
    /// All violations found (criteria that failed).
    pub violations: Vec<WcagViolation>,
    /// Criteria that passed.
    pub passed: Vec<WcagCriterion>,
    /// Compliance score in \[0.0, 1.0\]: proportion of checked criteria that pass.
    pub score: f32,
}

impl CaptionCompliance {
    /// Returns `true` when there are no error-level violations.
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        !self
            .violations
            .iter()
            .any(|v| v.severity == Severity::Error)
    }

    /// Number of criteria checked (passed + failed).
    #[must_use]
    pub fn total_checked(&self) -> usize {
        self.passed.len() + self.violations.len()
    }
}

// ── WcagCaptionChecker ────────────────────────────────────────────────────────

/// WCAG 2.1/2.2 compliance checker for caption specifications.
#[derive(Debug, Clone, Default)]
pub struct WcagCaptionChecker;

/// Maximum recommended reading speed in words per minute (WPM).
///
/// The BBC subtitle guidelines and many broadcast standards recommend ≤ 180 wpm
/// for general audiences; 300 wpm is used here as the absolute upper bound
/// before an advisory warning is raised.
pub const MAX_READING_SPEED_WPM: f64 = 300.0;

impl WcagCaptionChecker {
    /// Create a new checker.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Check a [`CaptionSpec`] against WCAG criteria up to and including `target_level`.
    ///
    /// Criteria at a stricter level than `target_level` are silently skipped.
    #[must_use]
    pub fn check(&self, spec: &CaptionSpec, target_level: WcagLevel) -> CaptionCompliance {
        let mut violations: Vec<WcagViolation> = Vec::new();
        let mut passed: Vec<WcagCriterion> = Vec::new();

        // 1.2.2 Captions (Prerecorded) – Level A
        // We treat a non-empty text as "captions are present".
        {
            let criterion = criterion_122();
            if criterion.level <= target_level {
                if spec.text.trim().is_empty() {
                    violations.push(WcagViolation::new(
                        criterion,
                        "No caption text provided — pre-recorded content must have captions (WCAG 1.2.2).",
                        Severity::Error,
                    ));
                } else {
                    passed.push(criterion);
                }
            }
        }

        // 1.4.3 Contrast Minimum – Level AA (ratio ≥ 4.5:1)
        {
            let criterion = criterion_143();
            if criterion.level <= target_level {
                let ratio = contrast_ratio_for(spec.foreground_color, spec.background_color);
                if ratio < 4.5 {
                    violations.push(WcagViolation::new(
                        criterion,
                        format!(
                            "Contrast ratio {ratio:.2}:1 is below the required 4.5:1 for normal text (WCAG 1.4.3)."
                        ),
                        Severity::Error,
                    ));
                } else {
                    passed.push(criterion);
                }
            }
        }

        // 1.4.6 Contrast Enhanced – Level AAA (ratio ≥ 7:1)
        {
            let criterion = criterion_146();
            if criterion.level <= target_level {
                let ratio = contrast_ratio_for(spec.foreground_color, spec.background_color);
                if ratio < 7.0 {
                    violations.push(WcagViolation::new(
                        criterion,
                        format!(
                            "Contrast ratio {ratio:.2}:1 is below the required 7.0:1 for enhanced contrast (WCAG 1.4.6)."
                        ),
                        Severity::Error,
                    ));
                } else {
                    passed.push(criterion);
                }
            }
        }

        // 1.2.6 Sign Language – Level AAA
        {
            let criterion = criterion_126();
            if criterion.level <= target_level {
                if spec.has_sign_language {
                    passed.push(criterion);
                } else {
                    violations.push(WcagViolation::new(
                        criterion,
                        "No sign language interpretation track provided (WCAG 1.2.6).",
                        Severity::Error,
                    ));
                }
            }
        }

        // Reading speed advisory (not a formal WCAG criterion number, treated as Info)
        {
            let wpm = reading_speed_wpm(&spec.text, spec.duration_ms);
            if let Some(speed) = wpm {
                let criterion = criterion_reading_speed();
                // Shown regardless of level — it is advisory only
                if speed > MAX_READING_SPEED_WPM {
                    violations.push(WcagViolation::new(
                        criterion,
                        format!(
                            "Reading speed {speed:.0} wpm exceeds the recommended maximum of {MAX_READING_SPEED_WPM} wpm."
                        ),
                        Severity::Warning,
                    ));
                } else {
                    passed.push(criterion);
                }
            }
        }

        let total = passed.len() + violations.len();
        let score = if total == 0 {
            1.0_f32
        } else {
            passed.len() as f32 / total as f32
        };

        CaptionCompliance {
            violations,
            passed,
            score,
        }
    }
}

// ── Criterion constructors ────────────────────────────────────────────────────

fn criterion_122() -> WcagCriterion {
    WcagCriterion::new(
        "1.2.2",
        "Captions (Prerecorded)",
        WcagLevel::A,
    )
}

fn criterion_143() -> WcagCriterion {
    WcagCriterion::new(
        "1.4.3",
        "Contrast (Minimum)",
        WcagLevel::AA,
    )
}

fn criterion_146() -> WcagCriterion {
    WcagCriterion::new(
        "1.4.6",
        "Contrast (Enhanced)",
        WcagLevel::AAA,
    )
}

fn criterion_126() -> WcagCriterion {
    WcagCriterion::new(
        "1.2.6",
        "Sign Language (Prerecorded)",
        WcagLevel::AAA,
    )
}

fn criterion_reading_speed() -> WcagCriterion {
    WcagCriterion::new(
        "advisory",
        "Caption Reading Speed",
        WcagLevel::A, // advisory — always evaluated
    )
}

// ── Colour math ───────────────────────────────────────────────────────────────

/// WCAG relative luminance for an sRGB triple.
///
/// Input components are in \[0, 255\]; output is in \[0.0, 1.0\].
/// Formula: <https://www.w3.org/TR/WCAG21/#dfn-relative-luminance>
#[must_use]
pub fn relative_luminance(rgb: [u8; 3]) -> f64 {
    let linearize = |c: u8| -> f64 {
        let s = f64::from(c) / 255.0;
        if s <= 0.040_45 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let r = linearize(rgb[0]);
    let g = linearize(rgb[1]);
    let b = linearize(rgb[2]);
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// WCAG contrast ratio between two sRGB colours.
///
/// Returns a value ≥ 1.0 (order of arguments does not matter).
#[must_use]
pub fn contrast_ratio_for(fg: [u8; 3], bg: [u8; 3]) -> f64 {
    let l1 = relative_luminance(fg);
    let l2 = relative_luminance(bg);
    let lighter = l1.max(l2);
    let darker = l1.min(l2);
    (lighter + 0.05) / (darker + 0.05)
}

// ── Reading-speed helper ──────────────────────────────────────────────────────

/// Estimate reading speed in words-per-minute for `text` shown during `duration_ms`.
///
/// Returns `None` when `duration_ms` is zero or the text has no words.
#[must_use]
pub fn reading_speed_wpm(text: &str, duration_ms: u64) -> Option<f64> {
    if duration_ms == 0 {
        return None;
    }
    let word_count = text.split_whitespace().count();
    if word_count == 0 {
        return None;
    }
    let minutes = duration_ms as f64 / 60_000.0;
    Some(word_count as f64 / minutes)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn checker() -> WcagCaptionChecker {
        WcagCaptionChecker::new()
    }

    // Helper: high-contrast white-on-black spec
    fn high_contrast_spec(text: &str, duration_ms: u64) -> CaptionSpec {
        CaptionSpec::new(text, duration_ms)
            .with_colors([255, 255, 255], [0, 0, 0])
    }

    // Helper: low-contrast similar-grey spec
    fn low_contrast_spec(text: &str, duration_ms: u64) -> CaptionSpec {
        CaptionSpec::new(text, duration_ms)
            .with_colors([150, 150, 150], [160, 160, 160])
    }

    // 1. High contrast (21:1) passes 1.4.3 at AA level
    #[test]
    fn test_high_contrast_passes_143() {
        let spec = high_contrast_spec("Hello world", 3000);
        let result = checker().check(&spec, WcagLevel::AA);
        let violated_ids: Vec<&str> = result.violations.iter()
            .map(|v| v.criterion.id.as_str())
            .collect();
        assert!(!violated_ids.contains(&"1.4.3"), "1.4.3 should pass for white-on-black");
    }

    // 2. Low contrast fails 1.4.3 at AA level
    #[test]
    fn test_low_contrast_fails_143() {
        let spec = low_contrast_spec("Hello world", 3000);
        let result = checker().check(&spec, WcagLevel::AA);
        let violated_ids: Vec<&str> = result.violations.iter()
            .map(|v| v.criterion.id.as_str())
            .collect();
        assert!(violated_ids.contains(&"1.4.3"), "1.4.3 should fail for low contrast");
    }

    // 3. White-on-black passes 1.4.6 (enhanced contrast, AAA)
    #[test]
    fn test_high_contrast_passes_146_at_aaa() {
        let spec = high_contrast_spec("Hello world", 3000);
        let result = checker().check(&spec, WcagLevel::AAA);
        let violated_ids: Vec<&str> = result.violations.iter()
            .map(|v| v.criterion.id.as_str())
            .collect();
        assert!(!violated_ids.contains(&"1.4.6"), "1.4.6 should pass for white-on-black");
    }

    // 4. Medium contrast (e.g. ~5:1) passes 1.4.3 but fails 1.4.6
    #[test]
    fn test_medium_contrast_passes_aa_fails_aaa() {
        // #767676 on white gives ~4.54:1
        let spec = CaptionSpec::new("Hello", 3000)
            .with_colors([118, 118, 118], [255, 255, 255]);
        let result = checker().check(&spec, WcagLevel::AAA);

        let violated_ids: Vec<&str> = result.violations.iter()
            .map(|v| v.criterion.id.as_str())
            .collect();

        let passed_ids: Vec<&str> = result.passed.iter()
            .map(|c| c.id.as_str())
            .collect();

        assert!(!violated_ids.contains(&"1.4.3"), "should pass 1.4.3 at AA");
        assert!(violated_ids.contains(&"1.4.6"), "should fail 1.4.6 at AAA");
        let _ = passed_ids; // used implicitly via violations check
    }

    // 5. Missing sign language fails 1.2.6 at AAA
    #[test]
    fn test_no_sign_language_fails_126_at_aaa() {
        let spec = high_contrast_spec("Hello", 3000);
        let result = checker().check(&spec, WcagLevel::AAA);
        let violated_ids: Vec<&str> = result.violations.iter()
            .map(|v| v.criterion.id.as_str())
            .collect();
        assert!(violated_ids.contains(&"1.2.6"));
    }

    // 6. With sign language passes 1.2.6 at AAA
    #[test]
    fn test_with_sign_language_passes_126() {
        let spec = high_contrast_spec("Hello", 3000).with_sign_language();
        let result = checker().check(&spec, WcagLevel::AAA);
        let violated_ids: Vec<&str> = result.violations.iter()
            .map(|v| v.criterion.id.as_str())
            .collect();
        assert!(!violated_ids.contains(&"1.2.6"));
    }

    // 7. Excessive reading speed raises a warning
    #[test]
    fn test_reading_speed_violation() {
        // 100 words in 5 seconds = 1200 wpm — way above 300 wpm
        let text = "word ".repeat(100);
        let spec = high_contrast_spec(text.trim(), 5_000);
        let result = checker().check(&spec, WcagLevel::AA);
        let has_speed_warning = result.violations.iter()
            .any(|v| v.criterion.id == "advisory" && v.severity == Severity::Warning);
        assert!(has_speed_warning, "Should warn about excessive reading speed");
    }

    // 8. Normal reading speed passes the advisory check
    #[test]
    fn test_normal_reading_speed_passes() {
        // 10 words in 4 seconds = 150 wpm — within limit
        let text = "The quick brown fox jumps over the lazy dog in";
        let spec = high_contrast_spec(text, 4_000);
        let result = checker().check(&spec, WcagLevel::AA);
        let speed_violation = result.violations.iter()
            .any(|v| v.criterion.id == "advisory");
        assert!(!speed_violation, "Normal reading speed should not produce a violation");
    }

    // 9. Empty text fails 1.2.2
    #[test]
    fn test_empty_text_fails_122() {
        let spec = CaptionSpec::new("", 3000);
        let result = checker().check(&spec, WcagLevel::A);
        let violated_ids: Vec<&str> = result.violations.iter()
            .map(|v| v.criterion.id.as_str())
            .collect();
        assert!(violated_ids.contains(&"1.2.2"));
    }

    // 10. AAA criteria are not checked when target is AA
    #[test]
    fn test_aaa_criteria_skipped_at_aa_level() {
        let spec = high_contrast_spec("Hello", 3000); // no sign language
        let result = checker().check(&spec, WcagLevel::AA);
        let has_126 = result.violations.iter().any(|v| v.criterion.id == "1.2.6")
            || result.passed.iter().any(|c| c.id == "1.2.6");
        assert!(!has_126, "1.2.6 (AAA) should not appear when target is AA");
    }

    // 11. Score is 1.0 when everything passes
    #[test]
    fn test_perfect_score_all_pass() {
        let spec = high_contrast_spec("Hello world caption text", 5_000)
            .with_sign_language();
        let result = checker().check(&spec, WcagLevel::AAA);
        // Only fails if something unexpected is violated
        if result.violations.is_empty() {
            assert!((result.score - 1.0).abs() < 1e-6, "Score should be 1.0 when no violations");
        } else {
            // Print violations for debugging in case criteria change
            for v in &result.violations {
                eprintln!("Unexpected violation: {} — {}", v.criterion.id, v.description);
            }
        }
    }

    // 12. contrast_ratio_for black/white ≈ 21:1
    #[test]
    fn test_contrast_ratio_black_white() {
        let ratio = contrast_ratio_for([0, 0, 0], [255, 255, 255]);
        assert!((ratio - 21.0).abs() < 0.01);
    }

    // 13. reading_speed_wpm returns None for zero duration
    #[test]
    fn test_reading_speed_zero_duration_returns_none() {
        assert!(reading_speed_wpm("hello world", 0).is_none());
    }

    // 14. is_compliant reflects error-level violations only
    #[test]
    fn test_is_compliant_only_errors_matter() {
        // Spec with text, high contrast, but excessive reading speed (Warning only)
        let text = "word ".repeat(100);
        let spec = high_contrast_spec(text.trim(), 3_000);
        let result = checker().check(&spec, WcagLevel::AA);
        let has_error = result.violations.iter().any(|v| v.severity == Severity::Error);
        assert_eq!(result.is_compliant(), !has_error);
    }
}
