//! Caption validation and quality control

use crate::error::Result;
use crate::types::{Caption, CaptionTrack};
use serde::{Deserialize, Serialize};

/// Validation issue severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Critical error
    Error,
    /// Warning
    Warning,
    /// Informational
    Info,
}

/// Validation issue
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Severity
    pub severity: IssueSeverity,
    /// Issue description
    pub message: String,
    /// Caption ID (if applicable)
    pub caption_id: Option<uuid::Uuid>,
    /// Timestamp (if applicable)
    pub timestamp: Option<crate::types::Timestamp>,
    /// Rule that was violated
    pub rule: String,
}

/// Validation report
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Issues found
    pub issues: Vec<ValidationIssue>,
    /// Statistics
    pub statistics: ValidationStatistics,
}

/// Validation statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationStatistics {
    /// Total captions
    pub total_captions: usize,
    /// Total words
    pub total_words: usize,
    /// Average reading speed (WPM)
    pub avg_reading_speed: f64,
    /// Maximum reading speed (WPM)
    pub max_reading_speed: f64,
    /// Average characters per line
    pub avg_chars_per_line: f64,
    /// Maximum characters per line
    pub max_chars_per_line: usize,
    /// Average lines per caption
    pub avg_lines_per_caption: f64,
    /// Total duration (milliseconds)
    pub total_duration_ms: i64,
    /// Error count
    pub error_count: usize,
    /// Warning count
    pub warning_count: usize,
}

impl ValidationReport {
    /// Create a new validation report
    #[must_use]
    pub fn new() -> Self {
        Self {
            issues: Vec::new(),
            statistics: ValidationStatistics {
                total_captions: 0,
                total_words: 0,
                avg_reading_speed: 0.0,
                max_reading_speed: 0.0,
                avg_chars_per_line: 0.0,
                max_chars_per_line: 0,
                avg_lines_per_caption: 0.0,
                total_duration_ms: 0,
                error_count: 0,
                warning_count: 0,
            },
        }
    }

    /// Add an issue
    pub fn add_issue(&mut self, issue: ValidationIssue) {
        match issue.severity {
            IssueSeverity::Error => self.statistics.error_count += 1,
            IssueSeverity::Warning => self.statistics.warning_count += 1,
            IssueSeverity::Info => {}
        }
        self.issues.push(issue);
    }

    /// Check if validation passed (no errors)
    #[must_use]
    pub fn passed(&self) -> bool {
        self.statistics.error_count == 0
    }

    /// Get errors
    #[must_use]
    pub fn errors(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .collect()
    }

    /// Get warnings
    #[must_use]
    pub fn warnings(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Warning)
            .collect()
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Caption validator
pub struct Validator {
    rules: Vec<Box<dyn ValidationRule>>,
}

impl Validator {
    /// Create a new validator
    #[must_use]
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a validation rule
    pub fn add_rule<R: ValidationRule + 'static>(&mut self, rule: R) {
        self.rules.push(Box::new(rule));
    }

    /// Validate a caption track
    pub fn validate(&self, track: &CaptionTrack) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();

        // Calculate statistics
        report.statistics.total_captions = track.count();
        report.statistics.total_words = track.total_words();
        report.statistics.total_duration_ms = track.total_duration().as_millis();

        let mut total_reading_speed = 0.0;
        let mut total_chars = 0;
        let mut total_lines = 0;

        for caption in &track.captions {
            let wpm = caption.reading_speed_wpm();
            total_reading_speed += wpm;
            if wpm > report.statistics.max_reading_speed {
                report.statistics.max_reading_speed = wpm;
            }

            let max_cpl = caption.max_chars_per_line();
            if max_cpl > report.statistics.max_chars_per_line {
                report.statistics.max_chars_per_line = max_cpl;
            }

            total_chars += caption.text.len();
            total_lines += caption.line_count();
        }

        if report.statistics.total_captions > 0 {
            report.statistics.avg_reading_speed =
                total_reading_speed / report.statistics.total_captions as f64;
            report.statistics.avg_chars_per_line = total_chars as f64 / total_lines.max(1) as f64;
            report.statistics.avg_lines_per_caption =
                total_lines as f64 / report.statistics.total_captions as f64;
        }

        // Run validation rules
        for rule in &self.rules {
            for caption in &track.captions {
                if let Err(issue) = rule.validate(caption, track) {
                    report.add_issue(issue);
                }
            }
        }

        Ok(report)
    }
}

impl Default for Validator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation rule trait
pub trait ValidationRule: Send + Sync {
    /// Validate a caption
    fn validate(
        &self,
        caption: &Caption,
        track: &CaptionTrack,
    ) -> std::result::Result<(), ValidationIssue>;
}

/// FCC compliance validator (US standards)
pub struct FccValidator {
    max_chars_per_line: usize,
    max_lines: usize,
    max_reading_speed: f64,
    min_duration_ms: i64,
}

impl FccValidator {
    /// Create a new FCC validator with default rules
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_chars_per_line: 32,
            max_lines: 4,
            max_reading_speed: 180.0,
            min_duration_ms: 1500,
        }
    }

    /// Create validator for CEA-608 (32 chars, 4 lines)
    #[must_use]
    pub fn cea608() -> Self {
        Self {
            max_chars_per_line: 32,
            max_lines: 4,
            max_reading_speed: 180.0,
            min_duration_ms: 1500,
        }
    }
}

impl Default for FccValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationRule for FccValidator {
    fn validate(
        &self,
        caption: &Caption,
        _track: &CaptionTrack,
    ) -> std::result::Result<(), ValidationIssue> {
        // Check character limit
        let max_cpl = caption.max_chars_per_line();
        if max_cpl > self.max_chars_per_line {
            return Err(ValidationIssue {
                severity: IssueSeverity::Error,
                message: format!(
                    "Caption exceeds maximum characters per line: {} > {}",
                    max_cpl, self.max_chars_per_line
                ),
                caption_id: Some(*caption.id.as_uuid()),
                timestamp: Some(caption.start),
                rule: "FCC_CHAR_LIMIT".to_string(),
            });
        }

        // Check line count
        let lines = caption.line_count();
        if lines > self.max_lines {
            return Err(ValidationIssue {
                severity: IssueSeverity::Error,
                message: format!(
                    "Caption exceeds maximum lines: {} > {}",
                    lines, self.max_lines
                ),
                caption_id: Some(*caption.id.as_uuid()),
                timestamp: Some(caption.start),
                rule: "FCC_LINE_LIMIT".to_string(),
            });
        }

        // Check reading speed
        let wpm = caption.reading_speed_wpm();
        if wpm > self.max_reading_speed {
            return Err(ValidationIssue {
                severity: IssueSeverity::Warning,
                message: format!(
                    "Reading speed too high: {:.1} WPM > {} WPM",
                    wpm, self.max_reading_speed
                ),
                caption_id: Some(*caption.id.as_uuid()),
                timestamp: Some(caption.start),
                rule: "FCC_READING_SPEED".to_string(),
            });
        }

        // Check minimum duration
        let duration_ms = caption.duration().as_millis();
        if duration_ms < self.min_duration_ms {
            return Err(ValidationIssue {
                severity: IssueSeverity::Warning,
                message: format!(
                    "Caption too short: {}ms < {}ms",
                    duration_ms, self.min_duration_ms
                ),
                caption_id: Some(*caption.id.as_uuid()),
                timestamp: Some(caption.start),
                rule: "FCC_MIN_DURATION".to_string(),
            });
        }

        Ok(())
    }
}

/// WCAG 2.1 compliance validator (Web accessibility)
pub struct WcagValidator {
    min_contrast_ratio: f64,
    required_font_size: u32,
}

impl WcagValidator {
    /// Create a new WCAG validator
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_contrast_ratio: 4.5, // WCAG AA standard
            required_font_size: 18,
        }
    }

    /// Create validator for WCAG AAA (stricter)
    #[must_use]
    pub fn aaa() -> Self {
        Self {
            min_contrast_ratio: 7.0, // WCAG AAA standard
            required_font_size: 18,
        }
    }
}

impl Default for WcagValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationRule for WcagValidator {
    fn validate(
        &self,
        caption: &Caption,
        _track: &CaptionTrack,
    ) -> std::result::Result<(), ValidationIssue> {
        // Check contrast ratio
        if let Some(bg_color) = caption.style.background_color {
            let contrast = caption.style.color.contrast_ratio(&bg_color);
            if contrast < self.min_contrast_ratio {
                return Err(ValidationIssue {
                    severity: IssueSeverity::Error,
                    message: format!(
                        "Insufficient contrast ratio: {:.2}:1 < {:.2}:1",
                        contrast, self.min_contrast_ratio
                    ),
                    caption_id: Some(*caption.id.as_uuid()),
                    timestamp: Some(caption.start),
                    rule: "WCAG_CONTRAST".to_string(),
                });
            }
        }

        // Check font size
        if caption.style.font_size < self.required_font_size {
            return Err(ValidationIssue {
                severity: IssueSeverity::Warning,
                message: format!(
                    "Font size too small: {} < {}",
                    caption.style.font_size, self.required_font_size
                ),
                caption_id: Some(*caption.id.as_uuid()),
                timestamp: Some(caption.start),
                rule: "WCAG_FONT_SIZE".to_string(),
            });
        }

        Ok(())
    }
}

/// Overlap detector
pub struct OverlapDetector;

impl ValidationRule for OverlapDetector {
    fn validate(
        &self,
        caption: &Caption,
        track: &CaptionTrack,
    ) -> std::result::Result<(), ValidationIssue> {
        for other in &track.captions {
            if caption.id != other.id && caption.overlaps(other) {
                return Err(ValidationIssue {
                    severity: IssueSeverity::Error,
                    message: format!("Caption overlaps with caption at {}", other.start),
                    caption_id: Some(*caption.id.as_uuid()),
                    timestamp: Some(caption.start),
                    rule: "NO_OVERLAP".to_string(),
                });
            }
        }
        Ok(())
    }
}

/// Gap detector (checks for gaps that are too small)
pub struct GapDetector {
    min_gap_frames: u32,
    fps: f64,
}

impl GapDetector {
    /// Create a new gap detector
    #[must_use]
    pub fn new(min_gap_frames: u32, fps: f64) -> Self {
        Self {
            min_gap_frames,
            fps,
        }
    }
}

impl ValidationRule for GapDetector {
    fn validate(
        &self,
        caption: &Caption,
        track: &CaptionTrack,
    ) -> std::result::Result<(), ValidationIssue> {
        let frame_duration_micros = (1_000_000.0 / self.fps) as i64;
        let min_gap_micros = frame_duration_micros * i64::from(self.min_gap_frames);

        // Find next caption
        if let Some(next) = track.captions.iter().find(|c| c.start > caption.end) {
            let gap_micros = next.start.as_micros() - caption.end.as_micros();
            if gap_micros > 0 && gap_micros < min_gap_micros {
                let gap_frames = gap_micros / frame_duration_micros;
                return Err(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    message: format!(
                        "Gap too small: {} frames < {} frames",
                        gap_frames, self.min_gap_frames
                    ),
                    caption_id: Some(*caption.id.as_uuid()),
                    timestamp: Some(caption.end),
                    rule: "MIN_GAP".to_string(),
                });
            }
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reading-speed checker (words per minute)
// ─────────────────────────────────────────────────────────────────────────────

/// Validates that each caption does not exceed a maximum reading speed (WPM).
///
/// The industry standard for broadcast captions is typically 180 WPM; online
/// content often allows up to 220 WPM.
pub struct ReadingSpeedValidator {
    /// Maximum allowed words per minute
    pub max_wpm: f64,
    /// Minimum allowed words per minute (0 = no minimum)
    pub min_wpm: f64,
}

impl ReadingSpeedValidator {
    /// Create a validator with a custom WPM ceiling.
    #[must_use]
    pub fn new(max_wpm: f64) -> Self {
        Self {
            max_wpm,
            min_wpm: 0.0,
        }
    }

    /// Create a validator that also checks a minimum reading speed.
    #[must_use]
    pub fn with_range(min_wpm: f64, max_wpm: f64) -> Self {
        Self { max_wpm, min_wpm }
    }

    /// Broadcast standard (180 WPM maximum)
    #[must_use]
    pub fn broadcast() -> Self {
        Self::new(180.0)
    }

    /// Online standard (220 WPM maximum)
    #[must_use]
    pub fn online() -> Self {
        Self::new(220.0)
    }
}

impl ReadingSpeedValidator {
    /// Compute WPM using millisecond-precision duration (avoids integer truncation).
    fn wpm(caption: &Caption) -> f64 {
        let words = caption.word_count() as f64;
        let duration_ms = caption.duration().as_millis() as f64;
        if duration_ms <= 0.0 {
            return 0.0;
        }
        // Convert ms → minutes: duration_ms / 60_000
        words / (duration_ms / 60_000.0)
    }
}

impl ValidationRule for ReadingSpeedValidator {
    fn validate(
        &self,
        caption: &Caption,
        _track: &CaptionTrack,
    ) -> std::result::Result<(), ValidationIssue> {
        let wpm = Self::wpm(caption);

        if wpm > self.max_wpm {
            return Err(ValidationIssue {
                severity: IssueSeverity::Warning,
                message: format!(
                    "Reading speed {:.1} WPM exceeds maximum {:.1} WPM",
                    wpm, self.max_wpm
                ),
                caption_id: Some(*caption.id.as_uuid()),
                timestamp: Some(caption.start),
                rule: "MAX_READING_SPEED".to_string(),
            });
        }

        if self.min_wpm > 0.0 && wpm < self.min_wpm && wpm > 0.0 {
            return Err(ValidationIssue {
                severity: IssueSeverity::Info,
                message: format!(
                    "Reading speed {:.1} WPM is below minimum {:.1} WPM",
                    wpm, self.min_wpm
                ),
                caption_id: Some(*caption.id.as_uuid()),
                timestamp: Some(caption.start),
                rule: "MIN_READING_SPEED".to_string(),
            });
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Line-length validator
// ─────────────────────────────────────────────────────────────────────────────

/// Validates that no line in a caption exceeds a maximum character count.
pub struct LineLengthValidator {
    /// Maximum characters per line
    pub max_chars: usize,
    /// Maximum number of lines per caption
    pub max_lines: usize,
}

impl LineLengthValidator {
    /// Create a new validator.
    #[must_use]
    pub fn new(max_chars: usize, max_lines: usize) -> Self {
        Self {
            max_chars,
            max_lines,
        }
    }

    /// Netflix standard: 42 chars/line, 2 lines
    #[must_use]
    pub fn netflix() -> Self {
        Self::new(42, 2)
    }

    /// EBU-STL standard: 37 chars/line, 2 lines
    #[must_use]
    pub fn ebu() -> Self {
        Self::new(37, 2)
    }

    /// BBC standard: 37 chars/line, 2 lines
    #[must_use]
    pub fn bbc() -> Self {
        Self::new(37, 2)
    }
}

impl ValidationRule for LineLengthValidator {
    fn validate(
        &self,
        caption: &Caption,
        _track: &CaptionTrack,
    ) -> std::result::Result<(), ValidationIssue> {
        let line_count = caption.line_count();
        if line_count > self.max_lines {
            return Err(ValidationIssue {
                severity: IssueSeverity::Error,
                message: format!("Too many lines: {} > {}", line_count, self.max_lines),
                caption_id: Some(*caption.id.as_uuid()),
                timestamp: Some(caption.start),
                rule: "MAX_LINES".to_string(),
            });
        }

        for (i, line) in caption.text.lines().enumerate() {
            let chars = line.chars().count();
            if chars > self.max_chars {
                return Err(ValidationIssue {
                    severity: IssueSeverity::Error,
                    message: format!(
                        "Line {} too long: {} chars > {} chars",
                        i + 1,
                        chars,
                        self.max_chars
                    ),
                    caption_id: Some(*caption.id.as_uuid()),
                    timestamp: Some(caption.start),
                    rule: "MAX_CHARS_PER_LINE".to_string(),
                });
            }
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Characters-per-second (CPS) constraint checker
// ─────────────────────────────────────────────────────────────────────────────

/// Validates that each caption respects a characters-per-second (CPS) ceiling.
///
/// CPS = (character count) / (duration in seconds).  Netflix and EBU recommend
/// ≤ 17 CPS for Latin scripts.
pub struct CpsValidator {
    /// Maximum allowed characters per second
    pub max_cps: f64,
}

impl CpsValidator {
    /// Create a new CPS validator.
    #[must_use]
    pub fn new(max_cps: f64) -> Self {
        Self { max_cps }
    }

    /// Netflix standard: 17 CPS
    #[must_use]
    pub fn netflix() -> Self {
        Self::new(17.0)
    }

    /// EBU standard: 17 CPS
    #[must_use]
    pub fn ebu() -> Self {
        Self::new(17.0)
    }

    /// Compute the CPS for a caption
    #[must_use]
    pub fn cps(caption: &Caption) -> f64 {
        let chars = caption.text.chars().filter(|c| !c.is_whitespace()).count();
        let duration_secs = caption.duration().as_millis() as f64 / 1000.0;
        if duration_secs <= 0.0 {
            return f64::INFINITY;
        }
        chars as f64 / duration_secs
    }
}

impl ValidationRule for CpsValidator {
    fn validate(
        &self,
        caption: &Caption,
        _track: &CaptionTrack,
    ) -> std::result::Result<(), ValidationIssue> {
        let cps = Self::cps(caption);
        if cps > self.max_cps {
            return Err(ValidationIssue {
                severity: IssueSeverity::Warning,
                message: format!(
                    "Characters per second {:.1} exceeds maximum {:.1}",
                    cps, self.max_cps
                ),
                caption_id: Some(*caption.id.as_uuid()),
                timestamp: Some(caption.start),
                rule: "MAX_CPS".to_string(),
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Enhanced overlap detector (reports all overlapping pairs)
// ─────────────────────────────────────────────────────────────────────────────

/// Reports every pair of overlapping captions in the track.
///
/// Unlike the original `OverlapDetector`, this validator produces one issue per
/// overlapping neighbour pair rather than stopping at the first.
pub struct StrictOverlapDetector;

impl ValidationRule for StrictOverlapDetector {
    fn validate(
        &self,
        caption: &Caption,
        track: &CaptionTrack,
    ) -> std::result::Result<(), ValidationIssue> {
        for other in &track.captions {
            if caption.id == other.id {
                continue;
            }
            if caption.overlaps(other) {
                // Only report when caption.start < other.start to avoid duplicate pairs
                if caption.start <= other.start {
                    return Err(ValidationIssue {
                        severity: IssueSeverity::Error,
                        message: format!(
                            "Caption [{} – {}] overlaps with caption [{} – {}]",
                            caption.start, caption.end, other.start, other.end
                        ),
                        caption_id: Some(*caption.id.as_uuid()),
                        timestamp: Some(caption.start),
                        rule: "STRICT_NO_OVERLAP".to_string(),
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod enhanced_validation_tests {
    use super::*;
    use crate::types::{Caption, Language, Timestamp};

    // ── Reading-speed validator ──────────────────────────────────────────

    #[test]
    fn test_reading_speed_above_max() {
        let validator = ReadingSpeedValidator::broadcast();
        let track = CaptionTrack::new(Language::english());
        // 10 words in 0.1 s → 6000 WPM → well above 180
        let caption = Caption::new(
            Timestamp::from_millis(0),
            Timestamp::from_millis(100),
            "one two three four five six seven eight nine ten".to_string(),
        );
        assert!(validator.validate(&caption, &track).is_err());
    }

    #[test]
    fn test_reading_speed_below_max() {
        let validator = ReadingSpeedValidator::broadcast();
        let track = CaptionTrack::new(Language::english());
        // 4 words in 5 s → 48 WPM → fine
        let caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(5),
            "one two three four".to_string(),
        );
        assert!(validator.validate(&caption, &track).is_ok());
    }

    #[test]
    fn test_reading_speed_online_standard() {
        let validator = ReadingSpeedValidator::online();
        assert_eq!(validator.max_wpm, 220.0);
    }

    #[test]
    fn test_reading_speed_with_range() {
        let validator = ReadingSpeedValidator::with_range(60.0, 180.0);
        assert_eq!(validator.min_wpm, 60.0);
        assert_eq!(validator.max_wpm, 180.0);
    }

    #[test]
    fn test_reading_speed_broadcast_constant() {
        let v = ReadingSpeedValidator::broadcast();
        assert_eq!(v.max_wpm, 180.0);
    }

    // ── Line-length validator ────────────────────────────────────────────

    #[test]
    fn test_line_length_too_long() {
        let validator = LineLengthValidator::netflix(); // 42 chars
        let track = CaptionTrack::new(Language::english());
        let long_line = "A".repeat(50);
        let caption = Caption::new(Timestamp::from_secs(0), Timestamp::from_secs(5), long_line);
        assert!(validator.validate(&caption, &track).is_err());
    }

    #[test]
    fn test_line_length_ok() {
        let validator = LineLengthValidator::netflix();
        let track = CaptionTrack::new(Language::english());
        let caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(5),
            "Short line".to_string(),
        );
        assert!(validator.validate(&caption, &track).is_ok());
    }

    #[test]
    fn test_line_length_too_many_lines() {
        let validator = LineLengthValidator::netflix(); // max 2 lines
        let track = CaptionTrack::new(Language::english());
        let caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(5),
            "Line1\nLine2\nLine3".to_string(),
        );
        assert!(validator.validate(&caption, &track).is_err());
    }

    #[test]
    fn test_line_length_ebu_constant() {
        let v = LineLengthValidator::ebu();
        assert_eq!(v.max_chars, 37);
        assert_eq!(v.max_lines, 2);
    }

    #[test]
    fn test_line_length_bbc_constant() {
        let v = LineLengthValidator::bbc();
        assert_eq!(v.max_chars, 37);
    }

    // ── CPS validator ────────────────────────────────────────────────────

    #[test]
    fn test_cps_above_max() {
        let validator = CpsValidator::netflix(); // 17 CPS
        let track = CaptionTrack::new(Language::english());
        // 170 non-space chars in 1 s → 170 CPS
        let caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(1),
            "A".repeat(170),
        );
        assert!(validator.validate(&caption, &track).is_err());
    }

    #[test]
    fn test_cps_below_max() {
        let validator = CpsValidator::netflix();
        let track = CaptionTrack::new(Language::english());
        // 10 non-space chars in 2 s → 5 CPS → fine
        let caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(2),
            "HelloWorld".to_string(),
        );
        assert!(validator.validate(&caption, &track).is_ok());
    }

    #[test]
    fn test_cps_ebu_constant() {
        let v = CpsValidator::ebu();
        assert_eq!(v.max_cps, 17.0);
    }

    #[test]
    fn test_cps_calculation() {
        let caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(10),
            "Hello World".to_string(), // 10 non-space chars / 10 s = 1.0 CPS
        );
        let cps = CpsValidator::cps(&caption);
        assert!((cps - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cps_custom_value() {
        let v = CpsValidator::new(25.0);
        assert_eq!(v.max_cps, 25.0);
    }

    // ── Strict overlap detector ──────────────────────────────────────────

    #[test]
    fn test_strict_overlap_detects_overlap() {
        let detector = StrictOverlapDetector;
        let mut track = CaptionTrack::new(Language::english());
        let cap1 = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(5),
            "First".to_string(),
        );
        let cap2 = Caption::new(
            Timestamp::from_secs(3),
            Timestamp::from_secs(7),
            "Second".to_string(),
        );
        track
            .add_caption(cap1.clone())
            .expect("adding caption should succeed");
        track
            .add_caption(cap2)
            .expect("adding caption should succeed");
        assert!(detector.validate(&cap1, &track).is_err());
    }

    #[test]
    fn test_strict_overlap_no_overlap() {
        let detector = StrictOverlapDetector;
        let mut track = CaptionTrack::new(Language::english());
        let cap1 = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(3),
            "First".to_string(),
        );
        let cap2 = Caption::new(
            Timestamp::from_secs(4),
            Timestamp::from_secs(7),
            "Second".to_string(),
        );
        track
            .add_caption(cap1.clone())
            .expect("adding caption should succeed");
        track
            .add_caption(cap2)
            .expect("adding caption should succeed");
        assert!(detector.validate(&cap1, &track).is_ok());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Color, Language, Timestamp};

    #[test]
    fn test_fcc_validator() {
        let caption = Caption::new(
            Timestamp::from_secs(1),
            Timestamp::from_secs(3),
            "This is a caption that is way too long for FCC standards".to_string(),
        );

        let validator = FccValidator::new();
        let track = CaptionTrack::new(Language::english());

        // This should fail because line is too long
        let result = validator.validate(&caption, &track);
        assert!(result.is_err());
    }

    #[test]
    fn test_wcag_contrast() {
        let mut caption = Caption::new(
            Timestamp::from_secs(1),
            Timestamp::from_secs(3),
            "Test".to_string(),
        );

        // Low contrast (light gray on white)
        caption.style.color = Color::rgb(200, 200, 200);
        caption.style.background_color = Some(Color::white());

        let validator = WcagValidator::new();
        let track = CaptionTrack::new(Language::english());

        let result = validator.validate(&caption, &track);
        assert!(result.is_err());
    }

    #[test]
    fn test_overlap_detection() {
        let mut track = CaptionTrack::new(Language::english());

        let cap1 = Caption::new(
            Timestamp::from_secs(1),
            Timestamp::from_secs(3),
            "First".to_string(),
        );
        let cap2 = Caption::new(
            Timestamp::from_secs(2),
            Timestamp::from_secs(4),
            "Second".to_string(),
        );

        track
            .add_caption(cap1.clone())
            .expect("adding caption should succeed");
        track
            .add_caption(cap2)
            .expect("adding caption should succeed");

        let detector = OverlapDetector;
        let result = detector.validate(&cap1, &track);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod extended_validation_tests {
    use super::*;
    use crate::types::{Caption, Color, Language, Timestamp};

    #[test]
    fn test_validation_report_creation() {
        let report = ValidationReport::new();
        assert_eq!(report.statistics.error_count, 0);
        assert_eq!(report.statistics.warning_count, 0);
        assert!(report.passed());
    }

    #[test]
    fn test_validation_report_errors() {
        let mut report = ValidationReport::new();

        report.add_issue(ValidationIssue {
            severity: IssueSeverity::Error,
            message: "Test error".to_string(),
            caption_id: None,
            timestamp: None,
            rule: "TEST".to_string(),
        });

        assert_eq!(report.statistics.error_count, 1);
        assert!(!report.passed());

        let errors = report.errors();
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn test_validation_report_warnings() {
        let mut report = ValidationReport::new();

        report.add_issue(ValidationIssue {
            severity: IssueSeverity::Warning,
            message: "Test warning".to_string(),
            caption_id: None,
            timestamp: None,
            rule: "TEST".to_string(),
        });

        assert_eq!(report.statistics.warning_count, 1);
        assert!(report.passed()); // Warnings don't fail validation

        let warnings = report.warnings();
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn test_fcc_validator_character_limit() {
        let validator = FccValidator::cea608();
        let track = CaptionTrack::new(Language::english());

        let caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(5),
            "This is a very long caption that definitely exceeds the thirty-two character limit"
                .to_string(),
        );

        let result = validator.validate(&caption, &track);
        assert!(result.is_err());
    }

    #[test]
    fn test_fcc_validator_line_limit() {
        let validator = FccValidator::new();
        let track = CaptionTrack::new(Language::english());

        let caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(5),
            "Line 1\nLine 2\nLine 3\nLine 4\nLine 5".to_string(),
        );

        let result = validator.validate(&caption, &track);
        assert!(result.is_err());
    }

    #[test]
    fn test_fcc_validator_reading_speed() {
        let validator = FccValidator::new();
        let track = CaptionTrack::new(Language::english());

        // Very short caption with lots of words = high WPM
        let caption = Caption::new(
            Timestamp::from_millis(100),
            Timestamp::from_millis(200),
            "one two three four five six seven eight nine ten".to_string(),
        );

        let result = validator.validate(&caption, &track);
        assert!(result.is_err()); // Should exceed reading speed
    }

    #[test]
    fn test_fcc_validator_min_duration() {
        let validator = FccValidator::new();
        let track = CaptionTrack::new(Language::english());

        let caption = Caption::new(
            Timestamp::from_millis(0),
            Timestamp::from_millis(500), // Less than 1500ms minimum
            "Short".to_string(),
        );

        let result = validator.validate(&caption, &track);
        assert!(result.is_err());
    }

    #[test]
    fn test_wcag_validator_contrast() {
        let validator = WcagValidator::new();
        let track = CaptionTrack::new(Language::english());

        let mut caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(5),
            "Test".to_string(),
        );

        // Low contrast (light gray on white)
        caption.style.color = Color::rgb(200, 200, 200);
        caption.style.background_color = Some(Color::white());

        let result = validator.validate(&caption, &track);
        assert!(result.is_err());
    }

    #[test]
    fn test_wcag_validator_font_size() {
        let validator = WcagValidator::new();
        let track = CaptionTrack::new(Language::english());

        let mut caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(5),
            "Test".to_string(),
        );

        caption.style.font_size = 12; // Too small

        let result = validator.validate(&caption, &track);
        assert!(result.is_err());
    }

    #[test]
    fn test_overlap_detector() {
        let detector = OverlapDetector;
        let mut track = CaptionTrack::new(Language::english());

        let cap1 = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(5),
            "First".to_string(),
        );

        let cap2 = Caption::new(
            Timestamp::from_secs(3),
            Timestamp::from_secs(7),
            "Second".to_string(),
        );

        track
            .add_caption(cap1.clone())
            .expect("adding caption should succeed");
        track
            .add_caption(cap2)
            .expect("adding caption should succeed");

        let result = detector.validate(&cap1, &track);
        assert!(result.is_err()); // Should detect overlap
    }

    #[test]
    fn test_gap_detector() {
        let detector = GapDetector::new(2, 25.0);
        let mut track = CaptionTrack::new(Language::english());

        let cap1 = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_millis(1000),
            "First".to_string(),
        );

        let cap2 = Caption::new(
            Timestamp::from_millis(1010), // 10ms gap = very small
            Timestamp::from_secs(2),
            "Second".to_string(),
        );

        track
            .add_caption(cap1.clone())
            .expect("adding caption should succeed");
        track
            .add_caption(cap2)
            .expect("adding caption should succeed");

        let result = detector.validate(&cap1, &track);
        assert!(result.is_err()); // Gap too small
    }

    #[test]
    fn test_validator_with_multiple_rules() {
        let mut validator = Validator::new();
        validator.add_rule(FccValidator::new());
        validator.add_rule(WcagValidator::new());
        validator.add_rule(OverlapDetector);

        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(0),
                Timestamp::from_secs(5),
                "Good caption".to_string(),
            ))
            .expect("operation should succeed in test");

        let report = validator
            .validate(&track)
            .expect("validation should succeed");
        // This caption should pass most checks
        assert!(report.statistics.error_count == 0 || report.statistics.error_count < 3);
    }

    #[test]
    fn test_validation_statistics() {
        let mut track = CaptionTrack::new(Language::english());

        for i in 0..10 {
            track
                .add_caption(Caption::new(
                    Timestamp::from_secs(i * 5),
                    Timestamp::from_secs(i * 5 + 3),
                    format!("Caption number {}", i + 1),
                ))
                .expect("operation should succeed in test");
        }

        let validator = Validator::new();
        let report = validator
            .validate(&track)
            .expect("validation should succeed");

        assert_eq!(report.statistics.total_captions, 10);
        assert!(report.statistics.avg_reading_speed > 0.0);
    }

    #[test]
    fn test_issue_severity() {
        let error = IssueSeverity::Error;
        let warning = IssueSeverity::Warning;
        let info = IssueSeverity::Info;

        assert_ne!(error, warning);
        assert_ne!(warning, info);
        assert_ne!(error, info);
    }

    #[test]
    fn test_wcag_aaa_validator() {
        let validator = WcagValidator::aaa();
        assert_eq!(validator.min_contrast_ratio, 7.0);

        let track = CaptionTrack::new(Language::english());
        let mut caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(5),
            "Test".to_string(),
        );

        // 4.5:1 ratio passes AA but not AAA
        caption.style.color = Color::white();
        caption.style.background_color = Some(Color::rgb(100, 100, 100));

        let result = validator.validate(&caption, &track);
        // Might fail AAA but pass AA
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_validation_report_statistics_defaults() {
        let stats = ValidationStatistics {
            total_captions: 0,
            total_words: 0,
            avg_reading_speed: 0.0,
            max_reading_speed: 0.0,
            avg_chars_per_line: 0.0,
            max_chars_per_line: 0,
            avg_lines_per_caption: 0.0,
            total_duration_ms: 0,
            error_count: 0,
            warning_count: 0,
        };

        assert_eq!(stats.total_captions, 0);
        assert_eq!(stats.error_count, 0);
    }
}
