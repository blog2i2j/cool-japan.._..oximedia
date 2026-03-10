//! Utility functions for caption processing

use crate::types::{CaptionTrack, Duration, Timestamp};
use unicode_segmentation::UnicodeSegmentation;

/// Text utilities
pub struct TextUtils;

impl TextUtils {
    /// Count grapheme clusters (visible characters)
    #[must_use]
    pub fn grapheme_count(text: &str) -> usize {
        text.graphemes(true).count()
    }

    /// Count words
    #[must_use]
    pub fn word_count(text: &str) -> usize {
        text.split_whitespace().count()
    }

    /// Trim whitespace and normalize line endings
    #[must_use]
    pub fn normalize(text: &str) -> String {
        text.lines().map(str::trim).collect::<Vec<_>>().join("\n")
    }

    /// Remove formatting tags (HTML, ASS, etc.)
    #[must_use]
    pub fn strip_tags(text: &str) -> String {
        let mut result = String::new();
        let mut in_tag = false;

        for ch in text.chars() {
            match ch {
                '<' | '{' => in_tag = true,
                '>' | '}' => {
                    in_tag = false;
                    continue;
                }
                _ if !in_tag => result.push(ch),
                _ => {}
            }
        }

        result
    }

    /// Calculate text width (approximate, based on character count)
    #[must_use]
    pub fn text_width(text: &str, font_size: u32) -> u32 {
        let chars = Self::grapheme_count(text);
        // Approximate: average character width is 60% of font size
        ((chars as f32) * (font_size as f32 * 0.6)) as u32
    }
}

/// Time utilities
pub struct TimeUtils;

impl TimeUtils {
    /// Convert frame number to timestamp
    #[must_use]
    pub fn frame_to_timestamp(frame: u64, fps: f64) -> Timestamp {
        let micros = ((frame as f64 / fps) * 1_000_000.0) as i64;
        Timestamp::from_micros(micros)
    }

    /// Convert timestamp to frame number
    #[must_use]
    pub fn timestamp_to_frame(timestamp: Timestamp, fps: f64) -> u64 {
        ((timestamp.as_micros() as f64 / 1_000_000.0) * fps) as u64
    }

    /// Calculate duration in frames
    #[must_use]
    pub fn duration_in_frames(duration: Duration, fps: f64) -> u64 {
        ((duration.as_micros() as f64 / 1_000_000.0) * fps) as u64
    }

    /// Format timestamp as SMPTE timecode (HH:MM:SS:FF)
    #[must_use]
    pub fn format_smpte(timestamp: Timestamp, fps: f64) -> String {
        let (h, m, s, ms) = timestamp.as_hmsm();
        let frames = ((f64::from(ms) / 1000.0) * fps) as u32;
        format!("{h:02}:{m:02}:{s:02}:{frames:02}")
    }
}

/// Statistics calculator
pub struct Statistics;

impl Statistics {
    /// Calculate caption statistics
    #[must_use]
    pub fn calculate(track: &CaptionTrack) -> CaptionStatistics {
        let mut stats = CaptionStatistics::default();

        stats.total_captions = track.count();
        stats.total_words = track.total_words();
        stats.total_duration = track.total_duration();

        if stats.total_captions == 0 {
            return stats;
        }

        let mut total_wpm = 0.0;
        let mut total_chars = 0;
        let mut total_lines = 0;

        for caption in &track.captions {
            let wpm = caption.reading_speed_wpm();
            total_wpm += wpm;
            stats.max_wpm = stats.max_wpm.max(wpm);

            let chars = caption.max_chars_per_line();
            total_chars += chars;
            stats.max_chars_per_line = stats.max_chars_per_line.max(chars);

            let lines = caption.line_count();
            total_lines += lines;
            stats.max_lines = stats.max_lines.max(lines);

            let duration = caption.duration().as_millis();
            stats.min_duration = if stats.min_duration == 0 {
                duration
            } else {
                stats.min_duration.min(duration)
            };
            stats.max_duration = stats.max_duration.max(duration);
        }

        stats.avg_wpm = total_wpm / (stats.total_captions as f64);
        stats.avg_chars_per_line = total_chars as f64 / (stats.total_captions as f64);
        stats.avg_lines = total_lines as f64 / (stats.total_captions as f64);
        stats.avg_duration = stats.total_duration.as_millis() / (stats.total_captions as i64);

        stats
    }
}

/// Caption statistics
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CaptionStatistics {
    /// Total number of captions
    pub total_captions: usize,
    /// Total words
    pub total_words: usize,
    /// Total duration
    pub total_duration: Duration,
    /// Average words per minute
    pub avg_wpm: f64,
    /// Maximum words per minute
    pub max_wpm: f64,
    /// Average characters per line
    pub avg_chars_per_line: f64,
    /// Maximum characters per line
    pub max_chars_per_line: usize,
    /// Average lines per caption
    pub avg_lines: f64,
    /// Maximum lines
    pub max_lines: usize,
    /// Average duration (milliseconds)
    pub avg_duration: i64,
    /// Minimum duration (milliseconds)
    pub min_duration: i64,
    /// Maximum duration (milliseconds)
    pub max_duration: i64,
}

// ─────────────────────────────────────────────
// SRT / WebVTT format conversion
// ─────────────────────────────────────────────

/// Format a `Timestamp` as an SRT timestamp (`HH:MM:SS,mmm`)
#[must_use]
pub fn timestamp_to_srt(ts: Timestamp) -> String {
    let (h, m, s, ms) = ts.as_hmsm();
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Parse an SRT timestamp string (`HH:MM:SS,mmm`) into a `Timestamp`.
///
/// Returns `None` if parsing fails.
#[allow(dead_code)]
#[must_use]
pub fn srt_to_timestamp(s: &str) -> Option<Timestamp> {
    // Accept both comma and period as the milliseconds separator
    let s = s.replace(',', ".");
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let hours: i64 = parts[0].parse().ok()?;
    let minutes: i64 = parts[1].parse().ok()?;
    // seconds may include fractional part after "."
    let sec_parts: Vec<&str> = parts[2].split('.').collect();
    let secs: i64 = sec_parts[0].parse().ok()?;
    let millis: i64 = if sec_parts.len() > 1 {
        let raw = sec_parts[1];
        // Normalise to exactly 3 digits
        let padded = format!("{raw:0<3}");
        padded[..3].parse().ok()?
    } else {
        0
    };
    Some(Timestamp::from_millis(
        hours * 3_600_000 + minutes * 60_000 + secs * 1_000 + millis,
    ))
}

/// Format a `Timestamp` as a `WebVTT` timestamp (`HH:MM:SS.mmm`)
#[must_use]
pub fn timestamp_to_webvtt(ts: Timestamp) -> String {
    let (h, m, s, ms) = ts.as_hmsm();
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

/// Parse a `WebVTT` timestamp string (`HH:MM:SS.mmm`) into a `Timestamp`.
///
/// Returns `None` if parsing fails.
#[allow(dead_code)]
#[must_use]
pub fn webvtt_to_timestamp(s: &str) -> Option<Timestamp> {
    // WebVTT uses "." as separator; reuse the SRT parser after normalising
    srt_to_timestamp(s)
}

/// Convert a single SRT caption block to `WebVTT` format.
///
/// `index` is the SRT sequence number (1-based).  The output is a single
/// `WebVTT` cue block without the file header.
#[allow(dead_code)]
#[must_use]
pub fn srt_block_to_webvtt(index: usize, start: Timestamp, end: Timestamp, text: &str) -> String {
    format!(
        "{index}\n{} --> {}\n{text}",
        timestamp_to_webvtt(start),
        timestamp_to_webvtt(end)
    )
}

/// Convert a single `WebVTT` cue block to SRT format.
#[allow(dead_code)]
#[must_use]
pub fn webvtt_block_to_srt(index: usize, start: Timestamp, end: Timestamp, text: &str) -> String {
    format!(
        "{index}\n{} --> {}\n{text}",
        timestamp_to_srt(start),
        timestamp_to_srt(end)
    )
}

// ─────────────────────────────────────────────
// Timestamp normalisation utilities
// ─────────────────────────────────────────────

/// Snap a `Timestamp` to the nearest frame boundary at the given frame rate.
#[allow(dead_code)]
#[must_use]
pub fn snap_to_frame(ts: Timestamp, fps: f64) -> Timestamp {
    let frame = TimeUtils::timestamp_to_frame(ts, fps);
    TimeUtils::frame_to_timestamp(frame, fps)
}

/// Clamp a `Timestamp` to `[min, max]`.
#[allow(dead_code)]
#[must_use]
pub fn clamp_timestamp(ts: Timestamp, min: Timestamp, max: Timestamp) -> Timestamp {
    if ts < min {
        min
    } else if ts > max {
        max
    } else {
        ts
    }
}

/// Round a `Timestamp` to the nearest millisecond boundary.
#[allow(dead_code)]
#[must_use]
pub fn round_to_millis(ts: Timestamp) -> Timestamp {
    let millis = ts.as_millis();
    Timestamp::from_millis(millis)
}

// ─────────────────────────────────────────────
// Word-level timing calculation
// ─────────────────────────────────────────────

/// A word annotated with its start and end timestamp within a caption.
#[derive(Debug, Clone, PartialEq)]
pub struct WordTiming {
    /// The word text
    pub word: String,
    /// Start of this word
    pub start: Timestamp,
    /// End of this word
    pub end: Timestamp,
}

/// Distribute a caption's time span evenly across its words, returning one
/// `WordTiming` per whitespace-delimited word.
///
/// This is a linear interpolation model: each word gets the same share of the
/// total caption duration.  For higher accuracy, combine with ASR word
/// timestamps.
#[allow(dead_code)]
#[must_use]
pub fn calculate_word_timings(
    text: &str,
    caption_start: Timestamp,
    caption_end: Timestamp,
) -> Vec<WordTiming> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }

    let total_micros = caption_end.as_micros() - caption_start.as_micros();
    let per_word = total_micros / words.len() as i64;

    words
        .into_iter()
        .enumerate()
        .map(|(i, word)| {
            let start = Timestamp::from_micros(caption_start.as_micros() + i as i64 * per_word);
            let end = Timestamp::from_micros(start.as_micros() + per_word);
            WordTiming {
                word: word.to_string(),
                start,
                end,
            }
        })
        .collect()
}

/// Like `calculate_word_timings` but weights each word's share by its
/// character count, giving longer words more time.
#[allow(dead_code)]
#[must_use]
pub fn calculate_word_timings_weighted(
    text: &str,
    caption_start: Timestamp,
    caption_end: Timestamp,
) -> Vec<WordTiming> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }

    let total_chars: usize = words.iter().map(|w| w.len()).sum();
    let total_micros = caption_end.as_micros() - caption_start.as_micros();

    let mut result = Vec::with_capacity(words.len());
    let mut cursor = caption_start.as_micros();

    for word in words {
        let share = if total_chars == 0 {
            total_micros / result.capacity() as i64
        } else {
            total_micros * word.len() as i64 / total_chars as i64
        };
        let start = Timestamp::from_micros(cursor);
        let end = Timestamp::from_micros(cursor + share);
        result.push(WordTiming {
            word: word.to_string(),
            start,
            end,
        });
        cursor += share;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Caption, Language};

    // ── SRT / WebVTT conversion ──────────────────────────────────────────

    #[test]
    fn test_timestamp_to_srt() {
        let ts = Timestamp::from_hmsm(1, 2, 3, 456);
        assert_eq!(timestamp_to_srt(ts), "01:02:03,456");
    }

    #[test]
    fn test_srt_to_timestamp_roundtrip() {
        let ts = Timestamp::from_hmsm(0, 1, 30, 500);
        let s = timestamp_to_srt(ts);
        let parsed = srt_to_timestamp(&s).expect("timestamp parsing should succeed");
        assert_eq!(parsed.as_millis(), ts.as_millis());
    }

    #[test]
    fn test_timestamp_to_webvtt() {
        let ts = Timestamp::from_hmsm(0, 0, 5, 100);
        assert_eq!(timestamp_to_webvtt(ts), "00:00:05.100");
    }

    #[test]
    fn test_webvtt_to_timestamp_roundtrip() {
        let ts = Timestamp::from_hmsm(2, 45, 10, 0);
        let s = timestamp_to_webvtt(ts);
        let parsed = webvtt_to_timestamp(&s).expect("timestamp parsing should succeed");
        assert_eq!(parsed.as_millis(), ts.as_millis());
    }

    #[test]
    fn test_srt_to_timestamp_invalid() {
        assert!(srt_to_timestamp("not-a-timestamp").is_none());
    }

    #[test]
    fn test_srt_block_to_webvtt() {
        let start = Timestamp::from_hmsm(0, 0, 1, 0);
        let end = Timestamp::from_hmsm(0, 0, 3, 500);
        let block = srt_block_to_webvtt(1, start, end, "Hello world");
        assert!(block.contains("00:00:01.000 --> 00:00:03.500"));
        assert!(block.contains("Hello world"));
    }

    #[test]
    fn test_webvtt_block_to_srt() {
        let start = Timestamp::from_hmsm(0, 0, 1, 0);
        let end = Timestamp::from_hmsm(0, 0, 3, 500);
        let block = webvtt_block_to_srt(2, start, end, "Test caption");
        assert!(block.contains("00:00:01,000 --> 00:00:03,500"));
        assert!(block.contains("Test caption"));
    }

    // ── Timestamp normalisation ──────────────────────────────────────────

    #[test]
    fn test_snap_to_frame() {
        // 1001ms at 25 fps: frame = 25, which snaps to exactly 1000ms
        let ts = Timestamp::from_millis(1001);
        let snapped = snap_to_frame(ts, 25.0);
        // frame 25 = 1000 ms
        assert_eq!(snapped.as_millis(), 1000);
    }

    #[test]
    fn test_clamp_timestamp_below() {
        let ts = Timestamp::from_secs(0);
        let min = Timestamp::from_secs(1);
        let max = Timestamp::from_secs(10);
        assert_eq!(clamp_timestamp(ts, min, max), min);
    }

    #[test]
    fn test_clamp_timestamp_above() {
        let ts = Timestamp::from_secs(20);
        let min = Timestamp::from_secs(1);
        let max = Timestamp::from_secs(10);
        assert_eq!(clamp_timestamp(ts, min, max), max);
    }

    #[test]
    fn test_clamp_timestamp_within() {
        let ts = Timestamp::from_secs(5);
        let min = Timestamp::from_secs(1);
        let max = Timestamp::from_secs(10);
        assert_eq!(clamp_timestamp(ts, min, max), ts);
    }

    #[test]
    fn test_round_to_millis() {
        let ts = Timestamp::from_micros(1_500_999);
        let rounded = round_to_millis(ts);
        // should truncate to 1500ms (1_500_000 µs)
        assert_eq!(rounded.as_millis(), 1500);
    }

    // ── Word-level timing ────────────────────────────────────────────────

    #[test]
    fn test_calculate_word_timings_count() {
        let timings = calculate_word_timings(
            "one two three",
            Timestamp::from_secs(0),
            Timestamp::from_secs(3),
        );
        assert_eq!(timings.len(), 3);
    }

    #[test]
    fn test_calculate_word_timings_coverage() {
        let start = Timestamp::from_secs(0);
        let end = Timestamp::from_secs(6);
        let timings = calculate_word_timings("a b c", start, end);
        // First word starts at caption start
        assert_eq!(timings[0].start, start);
        // Last word ends at or before caption end (due to integer division)
        assert!(
            timings
                .last()
                .expect("last element should exist")
                .end
                .as_micros()
                <= end.as_micros()
        );
    }

    #[test]
    fn test_calculate_word_timings_empty() {
        let timings = calculate_word_timings("", Timestamp::from_secs(0), Timestamp::from_secs(5));
        assert!(timings.is_empty());
    }

    #[test]
    fn test_calculate_word_timings_weighted() {
        // Longer word "three" should get more time than "a"
        let timings = calculate_word_timings_weighted(
            "a three",
            Timestamp::from_secs(0),
            Timestamp::from_secs(6),
        );
        assert_eq!(timings.len(), 2);
        let dur_a = timings[0].end.as_micros() - timings[0].start.as_micros();
        let dur_three = timings[1].end.as_micros() - timings[1].start.as_micros();
        assert!(dur_three > dur_a);
    }

    #[test]
    fn test_word_timing_fields() {
        let timings = calculate_word_timings(
            "hello world",
            Timestamp::from_secs(0),
            Timestamp::from_secs(2),
        );
        assert_eq!(timings[0].word, "hello");
        assert_eq!(timings[1].word, "world");
    }

    #[test]
    fn test_grapheme_count() {
        assert_eq!(TextUtils::grapheme_count("hello"), 5);
        assert_eq!(TextUtils::grapheme_count("hello world"), 11);
        // Unicode: emoji counts as 1 grapheme
        assert_eq!(TextUtils::grapheme_count("👍"), 1);
    }

    #[test]
    fn test_word_count() {
        assert_eq!(TextUtils::word_count("hello world"), 2);
        assert_eq!(TextUtils::word_count("  hello   world  "), 2);
    }

    #[test]
    fn test_strip_tags() {
        let text = "<b>Bold</b> and {\\i1}italic{\\i0}";
        let stripped = TextUtils::strip_tags(text);
        assert_eq!(stripped, "Bold and italic");
    }

    #[test]
    fn test_frame_conversion() {
        let ts = TimeUtils::frame_to_timestamp(25, 25.0);
        assert_eq!(ts.as_secs(), 1);

        let frame = TimeUtils::timestamp_to_frame(Timestamp::from_secs(1), 25.0);
        assert_eq!(frame, 25);
    }

    #[test]
    fn test_smpte_format() {
        let ts = Timestamp::from_hmsm(1, 30, 45, 500);
        let smpte = TimeUtils::format_smpte(ts, 25.0);
        assert!(smpte.contains("01:30:45"));
    }

    #[test]
    fn test_statistics() {
        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(0),
                Timestamp::from_secs(5),
                "Test caption one".to_string(),
            ))
            .expect("operation should succeed in test");
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(10),
                Timestamp::from_secs(15),
                "Test caption two".to_string(),
            ))
            .expect("operation should succeed in test");

        let stats = Statistics::calculate(&track);
        assert_eq!(stats.total_captions, 2);
        assert_eq!(stats.total_words, 6);
        assert!(stats.avg_wpm > 0.0);
    }
}
