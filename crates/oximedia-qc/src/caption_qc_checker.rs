//! QC checker for closed captions and subtitles.
//!
//! This module validates caption lines for common broadcast and streaming issues
//! such as excessive reading speed, overlapping timing, and line length.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Category of error found during caption QC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptionError {
    /// Caption line exceeds the maximum allowed character count.
    TooLong,
    /// Words per minute exceeds the maximum reading speed.
    TooFast,
    /// Caption timing is invalid (start >= end).
    BadTiming,
    /// Caption overlaps with another caption in time.
    Overlap,
    /// Caption line does not end with a period or sentence-ending punctuation.
    MissingPeriod,
    /// Caption contains a detected spelling error (placeholder).
    SpellingError,
}

impl CaptionError {
    /// Returns a numeric severity score for this error type.
    ///
    /// Higher values indicate more critical failures.
    #[must_use]
    pub fn severity(&self) -> u32 {
        match self {
            Self::BadTiming => 100,
            Self::Overlap => 90,
            Self::TooFast => 70,
            Self::TooLong => 60,
            Self::MissingPeriod => 30,
            Self::SpellingError => 20,
        }
    }
}

/// A single line of closed caption text with timing information.
#[derive(Debug, Clone)]
pub struct CaptionLine {
    /// The caption text.
    pub text: String,
    /// Start frame (inclusive).
    pub start_frame: u64,
    /// End frame (exclusive).
    pub end_frame: u64,
}

impl CaptionLine {
    /// Creates a new caption line.
    #[must_use]
    pub fn new(text: impl Into<String>, start_frame: u64, end_frame: u64) -> Self {
        Self {
            text: text.into(),
            start_frame,
            end_frame,
        }
    }

    /// Returns the number of frames this caption is displayed.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Returns the number of characters in the caption text.
    #[must_use]
    pub fn char_count(&self) -> usize {
        self.text.chars().count()
    }

    /// Calculates words per minute at the given frame rate.
    ///
    /// Returns `0.0` if duration is zero or `fps` is non-positive.
    #[must_use]
    pub fn words_per_minute(&self, fps: f32) -> f32 {
        if fps <= 0.0 {
            return 0.0;
        }
        let duration_frames = self.duration_frames();
        if duration_frames == 0 {
            return 0.0;
        }
        let duration_secs = duration_frames as f32 / fps;
        let word_count = self.text.split_whitespace().count() as f32;
        word_count / duration_secs * 60.0
    }
}

/// Timing gap between two consecutive caption lines (in frames).
///
/// A gap of zero means consecutive captions are displayed back-to-back
/// with no blank frames in between.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptionGap {
    /// Index of the earlier caption line.
    pub from_line: usize,
    /// Index of the later caption line.
    pub to_line: usize,
    /// Gap duration in frames (0 if back-to-back).
    pub gap_frames: u64,
}

/// A QC report for a collection of caption lines.
#[derive(Debug, Clone)]
pub struct CaptionQcReport {
    /// The caption lines that were checked.
    pub lines: Vec<CaptionLine>,
    /// Errors found, as (line_index, error) pairs.
    pub errors: Vec<(usize, CaptionError)>,
    /// Timing gaps between consecutive captions (populated when fps is provided).
    pub gaps: Vec<CaptionGap>,
}

impl CaptionQcReport {
    /// Creates a new report.
    #[must_use]
    pub fn new(lines: Vec<CaptionLine>, errors: Vec<(usize, CaptionError)>) -> Self {
        Self {
            lines,
            errors,
            gaps: Vec::new(),
        }
    }

    /// Creates a new report with timing gap information.
    #[must_use]
    pub fn with_gaps(
        lines: Vec<CaptionLine>,
        errors: Vec<(usize, CaptionError)>,
        gaps: Vec<CaptionGap>,
    ) -> Self {
        Self {
            lines,
            errors,
            gaps,
        }
    }

    /// Returns `true` if any errors were found.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Returns the number of errors found.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Returns the ratio of erroneous lines to total lines.
    ///
    /// Returns `0.0` if there are no lines.
    #[must_use]
    pub fn error_rate(&self) -> f32 {
        if self.lines.is_empty() {
            return 0.0;
        }
        // Count unique line indices with errors
        let mut errored_lines: Vec<usize> = self.errors.iter().map(|(i, _)| *i).collect();
        errored_lines.sort_unstable();
        errored_lines.dedup();
        errored_lines.len() as f32 / self.lines.len() as f32
    }

    /// Returns all timing gaps between consecutive captions in seconds, given `fps`.
    ///
    /// Returns an empty vec if no gap information was collected.
    #[must_use]
    pub fn gap_durations_secs(&self, fps: f32) -> Vec<f32> {
        if fps <= 0.0 {
            return Vec::new();
        }
        self.gaps
            .iter()
            .map(|g| g.gap_frames as f32 / fps)
            .collect()
    }
}

/// Checker that validates caption lines against configurable thresholds.
#[derive(Debug, Clone)]
pub struct CaptionQcChecker {
    /// Maximum allowed characters per caption line.
    pub max_chars_per_line: usize,
    /// Maximum allowed reading speed in words per minute.
    pub max_wpm: f32,
    /// Maximum allowed caption duration in frames.
    pub max_duration_frames: u64,
    /// Maximum allowed character rate in characters per second (CPS).
    ///
    /// Typical broadcast limit is 17 CPS; streaming platforms often allow 20–25 CPS.
    /// `None` disables the CPS check.
    pub max_chars_per_second: Option<f32>,
    /// Frames-per-second of the associated video.
    ///
    /// Used for WPM / CPS calculations and gap duration reporting.
    /// Defaults to 25.0 if not set.
    pub fps: f32,
    /// Minimum gap between consecutive captions in frames.
    ///
    /// Captions with a smaller gap are flagged with [`CaptionError::Overlap`].
    /// A value of 0 disables the minimum-gap check.
    pub min_gap_frames: u64,
}

impl CaptionQcChecker {
    /// Returns a checker configured with typical broadcast defaults:
    /// - 42 characters per line
    /// - 180 WPM
    /// - 6 seconds at 25 fps = 150 frames
    /// - 17 CPS (characters per second)
    /// - 25.0 fps
    /// - 2-frame minimum gap between captions
    #[must_use]
    pub fn broadcast_default() -> Self {
        Self {
            max_chars_per_line: 42,
            max_wpm: 180.0,
            max_duration_frames: 150,
            max_chars_per_second: Some(17.0),
            fps: 25.0,
            min_gap_frames: 2,
        }
    }

    /// Returns a checker tuned for online/streaming platforms:
    /// - 42 characters per line
    /// - 220 WPM
    /// - 10 seconds at 25 fps = 250 frames
    /// - 25 CPS
    /// - 25.0 fps
    /// - 1-frame minimum gap
    #[must_use]
    pub fn streaming_default() -> Self {
        Self {
            max_chars_per_line: 42,
            max_wpm: 220.0,
            max_duration_frames: 250,
            max_chars_per_second: Some(25.0),
            fps: 25.0,
            min_gap_frames: 1,
        }
    }

    /// Overrides the frame rate used for timing computations.
    #[must_use]
    pub const fn with_fps(mut self, fps: f32) -> Self {
        self.fps = fps;
        self
    }

    /// Sets the maximum characters-per-second threshold. Pass `None` to disable.
    #[must_use]
    pub const fn with_max_cps(mut self, max_cps: Option<f32>) -> Self {
        self.max_chars_per_second = max_cps;
        self
    }

    /// Sets the minimum gap between consecutive captions in frames.
    #[must_use]
    pub const fn with_min_gap_frames(mut self, frames: u64) -> Self {
        self.min_gap_frames = frames;
        self
    }

    /// Computes the character rate (characters per second) of a caption at the
    /// configured frame rate.
    ///
    /// Returns `0.0` if the duration is zero or fps is non-positive.
    #[must_use]
    pub fn chars_per_second(&self, line: &CaptionLine) -> f32 {
        if self.fps <= 0.0 {
            return 0.0;
        }
        let dur_frames = line.duration_frames();
        if dur_frames == 0 {
            return 0.0;
        }
        let dur_secs = dur_frames as f32 / self.fps;
        line.char_count() as f32 / dur_secs
    }

    /// Runs all checks against the provided caption lines and returns a report.
    ///
    /// Checks performed (in order):
    /// 1. Bad timing (start ≥ end) — skips remaining checks for that line.
    /// 2. Line too long (exceeds `max_chars_per_line`).
    /// 3. Reading speed too fast (exceeds `max_wpm`).
    /// 4. Character rate too fast (exceeds `max_chars_per_second`, if set).
    /// 5. Duration too long (exceeds `max_duration_frames`).
    /// 6. Overlap / insufficient gap with next caption.
    ///
    /// Gap information is always collected and stored in [`CaptionQcReport::gaps`].
    #[must_use]
    pub fn check(&self, lines: &[CaptionLine]) -> CaptionQcReport {
        let fps = self.fps;
        let mut errors: Vec<(usize, CaptionError)> = Vec::new();
        let mut gaps: Vec<CaptionGap> = Vec::new();

        for (i, line) in lines.iter().enumerate() {
            // ── 1. Bad timing ─────────────────────────────────────────────
            if line.start_frame >= line.end_frame {
                errors.push((i, CaptionError::BadTiming));
                // Still compute gap to the next line (if it has valid timing)
                if let Some(next) = lines.get(i + 1) {
                    if next.start_frame > line.end_frame {
                        gaps.push(CaptionGap {
                            from_line: i,
                            to_line: i + 1,
                            gap_frames: next.start_frame - line.end_frame,
                        });
                    }
                }
                continue; // Skip quality checks on a malformed line
            }

            // ── 2. Line too long ─────────────────────────────────────────
            if line.char_count() > self.max_chars_per_line {
                errors.push((i, CaptionError::TooLong));
            }

            // ── 3. Reading speed (WPM) ───────────────────────────────────
            let wpm = line.words_per_minute(fps);
            if wpm > self.max_wpm {
                errors.push((i, CaptionError::TooFast));
            }

            // ── 4. Character rate (CPS) ──────────────────────────────────
            if let Some(max_cps) = self.max_chars_per_second {
                let cps = self.chars_per_second(line);
                if cps > max_cps {
                    errors.push((i, CaptionError::TooFast));
                }
            }

            // ── 5. Duration too long ─────────────────────────────────────
            if line.duration_frames() > self.max_duration_frames {
                errors.push((i, CaptionError::TooFast));
            }

            // ── 6. Gap / overlap with next caption ───────────────────────
            if let Some(next) = lines.get(i + 1) {
                if next.start_frame < line.end_frame {
                    // Actual overlap
                    errors.push((i, CaptionError::Overlap));
                    gaps.push(CaptionGap {
                        from_line: i,
                        to_line: i + 1,
                        gap_frames: 0,
                    });
                } else {
                    let gap = next.start_frame - line.end_frame;
                    // Insufficient gap (but not an overlap)
                    if self.min_gap_frames > 0 && gap < self.min_gap_frames {
                        errors.push((i, CaptionError::Overlap));
                    }
                    gaps.push(CaptionGap {
                        from_line: i,
                        to_line: i + 1,
                        gap_frames: gap,
                    });
                }
            }
        }

        CaptionQcReport::with_gaps(lines.to_vec(), errors, gaps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CaptionError tests ---

    #[test]
    fn test_error_severity_ordering() {
        assert!(CaptionError::BadTiming.severity() > CaptionError::TooFast.severity());
        assert!(CaptionError::TooFast.severity() > CaptionError::MissingPeriod.severity());
        assert!(CaptionError::MissingPeriod.severity() > CaptionError::SpellingError.severity());
    }

    #[test]
    fn test_error_overlap_severity() {
        assert_eq!(CaptionError::Overlap.severity(), 90);
    }

    // --- CaptionLine tests ---

    #[test]
    fn test_caption_line_duration() {
        let line = CaptionLine::new("Hello world", 0, 50);
        assert_eq!(line.duration_frames(), 50);
    }

    #[test]
    fn test_caption_line_duration_inverted() {
        let line = CaptionLine::new("Hello", 100, 50);
        assert_eq!(line.duration_frames(), 0);
    }

    #[test]
    fn test_caption_line_char_count() {
        let line = CaptionLine::new("Hello", 0, 50);
        assert_eq!(line.char_count(), 5);
    }

    #[test]
    fn test_caption_line_words_per_minute() {
        // 2 words in 25 frames at 25fps => 2 words / 1 sec => 120 WPM
        let line = CaptionLine::new("Hello world", 0, 25);
        let wpm = line.words_per_minute(25.0);
        assert!((wpm - 120.0).abs() < 1.0);
    }

    #[test]
    fn test_caption_line_wpm_zero_fps() {
        let line = CaptionLine::new("Hello", 0, 25);
        assert!((line.words_per_minute(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_caption_line_wpm_zero_duration() {
        let line = CaptionLine::new("Hello", 50, 50);
        assert!((line.words_per_minute(25.0) - 0.0).abs() < 1e-6);
    }

    // --- CaptionQcReport tests ---

    #[test]
    fn test_report_has_no_errors() {
        let report = CaptionQcReport::new(vec![], vec![]);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_report_has_errors() {
        let line = CaptionLine::new("Bad", 10, 5);
        let report = CaptionQcReport::new(vec![line], vec![(0, CaptionError::BadTiming)]);
        assert!(report.has_errors());
        assert_eq!(report.error_count(), 1);
    }

    #[test]
    fn test_report_error_rate_no_lines() {
        let report = CaptionQcReport::new(vec![], vec![]);
        assert!((report.error_rate() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_report_error_rate_half() {
        let lines = vec![
            CaptionLine::new("OK", 0, 50),
            CaptionLine::new("Bad", 50, 40),
        ];
        let errors = vec![(1, CaptionError::BadTiming)];
        let report = CaptionQcReport::new(lines, errors);
        assert!((report.error_rate() - 0.5).abs() < 1e-6);
    }

    // --- CaptionQcChecker tests ---

    #[test]
    fn test_checker_broadcast_default() {
        let checker = CaptionQcChecker::broadcast_default();
        assert_eq!(checker.max_chars_per_line, 42);
        assert!((checker.max_wpm - 180.0).abs() < 1e-6);
    }

    #[test]
    fn test_checker_clean_lines() {
        let checker = CaptionQcChecker::broadcast_default();
        let lines = vec![CaptionLine::new("Hello world.", 0, 50)];
        let report = checker.check(&lines);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_checker_detects_bad_timing() {
        let checker = CaptionQcChecker::broadcast_default();
        let lines = vec![CaptionLine::new("Bad timing", 100, 50)];
        let report = checker.check(&lines);
        assert!(report
            .errors
            .iter()
            .any(|(_, e)| *e == CaptionError::BadTiming));
    }

    #[test]
    fn test_checker_detects_overlap() {
        let checker = CaptionQcChecker::broadcast_default();
        let lines = vec![
            CaptionLine::new("Line one.", 0, 60),
            CaptionLine::new("Line two.", 50, 100),
        ];
        let report = checker.check(&lines);
        assert!(report
            .errors
            .iter()
            .any(|(_, e)| *e == CaptionError::Overlap));
    }

    // ── Additional caption QC checker tests ────────────────────────────────

    #[test]
    fn test_checker_detects_too_long_line() {
        let checker = CaptionQcChecker::broadcast_default(); // max_chars = 42
                                                             // 43 characters — exceeds limit
        let long_text = "A".repeat(43);
        let lines = vec![CaptionLine::new(long_text, 0, 50)];
        let report = checker.check(&lines);
        assert!(
            report
                .errors
                .iter()
                .any(|(_, e)| *e == CaptionError::TooLong),
            "Expected TooLong error for a line with 43 characters"
        );
    }

    #[test]
    fn test_checker_no_error_for_exactly_max_chars() {
        let checker = CaptionQcChecker::broadcast_default(); // max_chars = 42
        let text = "A".repeat(42);
        let lines = vec![CaptionLine::new(text, 0, 50)];
        let report = checker.check(&lines);
        assert!(
            !report
                .errors
                .iter()
                .any(|(_, e)| *e == CaptionError::TooLong),
            "Exactly max_chars should not produce TooLong"
        );
    }

    #[test]
    fn test_checker_detects_reading_speed_too_fast() {
        // At 25 fps, 1 word in 1 frame = 25*60 = 1500 WPM >> 180 WPM limit
        let checker = CaptionQcChecker::broadcast_default();
        let lines = vec![CaptionLine::new("Fast", 0, 1)];
        let report = checker.check(&lines);
        assert!(
            report
                .errors
                .iter()
                .any(|(_, e)| *e == CaptionError::TooFast),
            "Expected TooFast error for extremely short display duration"
        );
    }

    #[test]
    fn test_checker_detects_insufficient_gap() {
        // min_gap_frames = 2 (broadcast_default); gap of 1 frame should flag Overlap
        let checker = CaptionQcChecker::broadcast_default();
        let lines = vec![
            CaptionLine::new("First.", 0, 50),
            CaptionLine::new("Second.", 51, 100), // gap of 1 frame < 2
        ];
        let report = checker.check(&lines);
        assert!(
            report
                .errors
                .iter()
                .any(|(_, e)| *e == CaptionError::Overlap),
            "Gap of 1 frame should be flagged when min_gap_frames = 2"
        );
    }

    #[test]
    fn test_checker_sufficient_gap_no_error() {
        // Gap of exactly 2 frames should NOT be flagged
        let checker = CaptionQcChecker::broadcast_default();
        let lines = vec![
            CaptionLine::new("First.", 0, 50),
            CaptionLine::new("Second.", 52, 100), // gap of 2 frames
        ];
        let report = checker.check(&lines);
        assert!(
            !report
                .errors
                .iter()
                .any(|(_, e)| *e == CaptionError::Overlap),
            "Gap of exactly min_gap_frames should not be flagged"
        );
    }

    #[test]
    fn test_checker_gap_collection() {
        let checker = CaptionQcChecker::broadcast_default();
        let lines = vec![
            CaptionLine::new("One.", 0, 25),
            CaptionLine::new("Two.", 30, 55),   // gap = 5 frames
            CaptionLine::new("Three.", 60, 85), // gap = 5 frames
        ];
        let report = checker.check(&lines);
        assert_eq!(report.gaps.len(), 2, "Should record 2 gaps for 3 lines");
        assert_eq!(report.gaps[0].gap_frames, 5);
        assert_eq!(report.gaps[1].gap_frames, 5);
    }

    #[test]
    fn test_checker_gap_durations_secs() {
        let checker = CaptionQcChecker::broadcast_default();
        let lines = vec![
            CaptionLine::new("A.", 0, 25),
            CaptionLine::new("B.", 50, 75), // gap = 25 frames = 1.0 sec at 25fps
        ];
        let report = checker.check(&lines);
        let secs = report.gap_durations_secs(25.0);
        assert_eq!(secs.len(), 1);
        assert!((secs[0] - 1.0).abs() < 0.01, "gap should be 1.0 second");
    }

    #[test]
    fn test_checker_cps_limit_enforced() {
        // 17 chars / 1 second = 17 CPS exactly at limit (should pass)
        // 18 chars / 1 second = 18 CPS > 17 CPS (should fail)
        let checker = CaptionQcChecker::broadcast_default(); // max_cps = Some(17.0)
                                                             // 25 frames = 1 second at 25 fps; 18 chars → 18 CPS
        let lines = vec![CaptionLine::new("A".repeat(18), 0, 25)];
        let report = checker.check(&lines);
        assert!(
            report
                .errors
                .iter()
                .any(|(_, e)| *e == CaptionError::TooFast),
            "18 chars in 1s exceeds 17 CPS limit"
        );
    }

    #[test]
    fn test_checker_cps_disabled() {
        // Disable CPS check; the same fast line should not produce TooFast for CPS
        let checker = CaptionQcChecker::broadcast_default()
            .with_max_cps(None)
            // Also raise WPM limit so only CPS would have caught it
            ;
        let lines = vec![CaptionLine::new("A".repeat(18), 0, 25)];
        // Without CPS check, high chars-per-second no longer flags TooFast.
        // (WPM is word-based, not char-based, so a single 18-char token at
        // 25fps = 60 WPM which is under the 180 WPM limit.)
        let report = checker.check(&lines);
        assert!(
            !report
                .errors
                .iter()
                .any(|(_, e)| *e == CaptionError::TooFast),
            "TooFast should not be triggered when CPS check is disabled and WPM is in range"
        );
    }

    #[test]
    fn test_checker_chars_per_second_calculation() {
        let checker = CaptionQcChecker::broadcast_default();
        // 10 chars in 25 frames at 25 fps = 10 chars / 1 sec = 10 CPS
        let line = CaptionLine::new("0123456789", 0, 25);
        let cps = checker.chars_per_second(&line);
        assert!((cps - 10.0).abs() < 0.1, "Expected ~10.0 CPS, got {cps}");
    }

    #[test]
    fn test_checker_empty_lines_no_panic() {
        let checker = CaptionQcChecker::broadcast_default();
        let report = checker.check(&[]);
        assert!(!report.has_errors());
        assert!(report.gaps.is_empty());
    }

    #[test]
    fn test_checker_single_line_no_gap() {
        let checker = CaptionQcChecker::broadcast_default();
        let lines = vec![CaptionLine::new("Hello world.", 0, 50)];
        let report = checker.check(&lines);
        assert!(report.gaps.is_empty(), "Single line produces no gaps");
    }

    #[test]
    fn test_streaming_checker_higher_wpm_limit() {
        // Streaming allows up to 220 WPM vs broadcast 180 WPM
        let broadcast = CaptionQcChecker::broadcast_default();
        let streaming = CaptionQcChecker::streaming_default();
        assert!(streaming.max_wpm > broadcast.max_wpm);
    }

    #[test]
    fn test_gap_back_to_back_is_zero() {
        let checker = CaptionQcChecker::broadcast_default().with_min_gap_frames(0);
        let lines = vec![
            CaptionLine::new("A.", 0, 25),
            CaptionLine::new("B.", 25, 50), // gap = 0 (back-to-back)
        ];
        let report = checker.check(&lines);
        assert_eq!(report.gaps[0].gap_frames, 0);
        // With min_gap_frames=0 no Overlap error should be generated
        assert!(!report
            .errors
            .iter()
            .any(|(_, e)| *e == CaptionError::Overlap));
    }
}
