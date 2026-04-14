//! Caption reading-speed validation.
//!
//! Validates whether a caption cue can be comfortably read in the available
//! display duration, based on a configurable words-per-second (WPS) target.
//!
//! # Standards
//!
//! | Audience         | WPS target |
//! |------------------|-----------|
//! | Adults           | 3.0 WPS   |
//! | Children (< 10)  | 1.5 WPS   |
//! | SDH / mixed      | 2.0 WPS   |
//! | BBC / EBU        | 3.3 WPS   |
//!
//! # Example
//!
//! ```rust
//! use oximedia_caption_gen::reading_speed::ReadingSpeedValidator;
//!
//! // "Hello world" = 2 words in 1000 ms = 2 WPS ≤ 3.0 WPS → ok
//! assert!(ReadingSpeedValidator::check_cue("Hello world", 1_000, 3.0));
//!
//! // Same text but only 200 ms → 10 WPS > 3.0 WPS → too fast
//! assert!(!ReadingSpeedValidator::check_cue("Hello world", 200, 3.0));
//! ```

// ─── ReadingSpeedValidator ────────────────────────────────────────────────────

/// Validates caption reading speed against a words-per-second threshold.
#[derive(Debug, Clone)]
pub struct ReadingSpeedValidator {
    /// Target audience maximum WPS.
    pub max_wps: f32,
}

impl Default for ReadingSpeedValidator {
    fn default() -> Self {
        Self { max_wps: 3.0 }
    }
}

impl ReadingSpeedValidator {
    /// Create a new validator with the given WPS limit.
    #[must_use]
    pub fn new(max_wps: f32) -> Self {
        Self { max_wps }
    }

    /// Adult standard (3.0 WPS).
    #[must_use]
    pub fn adult() -> Self {
        Self::new(3.0)
    }

    /// Children's standard (1.5 WPS).
    #[must_use]
    pub fn children() -> Self {
        Self::new(1.5)
    }

    /// BBC/EBU standard (3.3 WPS).
    #[must_use]
    pub fn bbc() -> Self {
        Self::new(3.3)
    }

    /// SDH / mixed-audience standard (2.0 WPS).
    #[must_use]
    pub fn sdh() -> Self {
        Self::new(2.0)
    }

    /// Check a single cue using this validator's `max_wps`.
    ///
    /// Returns `true` if the cue is within the reading speed limit.
    #[must_use]
    pub fn check(&self, text: &str, duration_ms: u64) -> bool {
        Self::check_cue(text, duration_ms, self.max_wps)
    }

    /// Check whether `text` can be read in `duration_ms` at `max_wps`.
    ///
    /// # Arguments
    ///
    /// * `text`        – Caption text (whitespace-separated words are counted).
    /// * `duration_ms` – Display duration in milliseconds.
    /// * `max_wps`     – Maximum allowed words per second.
    ///
    /// # Returns
    ///
    /// `true` if the cue is within the reading speed limit, `false` if it
    /// exceeds `max_wps`.  An empty cue or zero-duration always returns `true`.
    #[must_use]
    pub fn check_cue(text: &str, duration_ms: u64, max_wps: f32) -> bool {
        if text.is_empty() || duration_ms == 0 {
            return true;
        }
        let word_count = word_count(text);
        if word_count == 0 {
            return true;
        }
        let duration_secs = duration_ms as f32 / 1000.0;
        let actual_wps = word_count as f32 / duration_secs;
        actual_wps <= max_wps
    }

    /// Compute the actual WPS for a cue.
    ///
    /// Returns `None` if `duration_ms` is zero or `text` is empty.
    #[must_use]
    pub fn compute_wps(text: &str, duration_ms: u64) -> Option<f32> {
        if text.is_empty() || duration_ms == 0 {
            return None;
        }
        let wc = word_count(text);
        if wc == 0 {
            return None;
        }
        let secs = duration_ms as f32 / 1000.0;
        Some(wc as f32 / secs)
    }

    /// Recommended minimum duration (ms) for `text` at the validator's WPS limit.
    ///
    /// Returns `0` for empty text.
    #[must_use]
    pub fn min_duration_ms(&self, text: &str) -> u64 {
        let wc = word_count(text);
        if wc == 0 || self.max_wps <= 0.0 {
            return 0;
        }
        let min_secs = wc as f32 / self.max_wps;
        (min_secs * 1000.0).ceil() as u64
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Count words in `text` (whitespace-split tokens, ignoring empty splits).
fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_text_ok() {
        assert!(ReadingSpeedValidator::check_cue("", 1000, 3.0));
    }

    #[test]
    fn test_zero_duration_ok() {
        assert!(ReadingSpeedValidator::check_cue("hello world", 0, 3.0));
    }

    #[test]
    fn test_within_limit() {
        // 2 words / 1 s = 2.0 WPS ≤ 3.0
        assert!(ReadingSpeedValidator::check_cue("hello world", 1_000, 3.0));
    }

    #[test]
    fn test_exceeds_limit() {
        // 2 words / 0.2 s = 10.0 WPS > 3.0
        assert!(!ReadingSpeedValidator::check_cue("hello world", 200, 3.0));
    }

    #[test]
    fn test_exactly_at_limit() {
        // 3 words / 1 s = 3.0 WPS == 3.0 → ok
        assert!(ReadingSpeedValidator::check_cue(
            "one two three",
            1_000,
            3.0
        ));
    }

    #[test]
    fn test_children_limit_stricter() {
        // 3 words / 1 s = 3.0 WPS > 1.5 → fail
        assert!(!ReadingSpeedValidator::check_cue(
            "one two three",
            1_000,
            1.5
        ));
    }

    #[test]
    fn test_compute_wps_basic() {
        let wps = ReadingSpeedValidator::compute_wps("hello world", 1_000);
        let wps = wps.expect("should compute wps");
        assert!((wps - 2.0).abs() < 0.01, "expected 2.0 wps, got {wps}");
    }

    #[test]
    fn test_compute_wps_empty_returns_none() {
        assert!(ReadingSpeedValidator::compute_wps("", 1_000).is_none());
    }

    #[test]
    fn test_compute_wps_zero_duration_returns_none() {
        assert!(ReadingSpeedValidator::compute_wps("hello", 0).is_none());
    }

    #[test]
    fn test_validator_check_method() {
        let v = ReadingSpeedValidator::adult();
        assert!(v.check("hello world", 1_000));
        assert!(!v.check("hello world", 100));
    }

    #[test]
    fn test_min_duration_ms() {
        let v = ReadingSpeedValidator::new(3.0);
        // 3 words at 3.0 WPS → min 1 s = 1000 ms
        let min = v.min_duration_ms("one two three");
        assert_eq!(min, 1000);
    }

    #[test]
    fn test_min_duration_ms_empty() {
        let v = ReadingSpeedValidator::adult();
        assert_eq!(v.min_duration_ms(""), 0);
    }

    #[test]
    fn test_adult_preset_wps() {
        let v = ReadingSpeedValidator::adult();
        assert!((v.max_wps - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_children_preset_wps() {
        let v = ReadingSpeedValidator::children();
        assert!((v.max_wps - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bbc_preset_wps() {
        let v = ReadingSpeedValidator::bbc();
        assert!((v.max_wps - 3.3).abs() < f32::EPSILON);
    }
}
