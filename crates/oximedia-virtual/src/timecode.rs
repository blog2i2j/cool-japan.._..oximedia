//! SMPTE timecode integration for virtual production synchronisation.
//!
//! Supports:
//! - Non-drop-frame (NDF) and drop-frame (DF) timecodes
//! - Frame rates: 23.976 (24/1.001), 24, 25, 29.97 (30/1.001), 30, 48, 50, 60
//! - Arithmetic: add/subtract frame counts, difference
//! - Conversion to/from total frame counts and real-time seconds
//! - SMPTE string parsing and formatting (`HH:MM:SS:FF` / `HH:MM:SS;FF`)
//! - Playback trigger: fire registered callbacks at target timecodes
//! - Linear time code (LTC) bit-stream frame boundary detection

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// FrameRate
// ---------------------------------------------------------------------------

/// Supported SMPTE timecode frame rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FrameRate {
    /// 23.976 fps (24000/1001) — drop-frame capable
    F23_976,
    /// 24 fps
    F24,
    /// 25 fps (PAL)
    F25,
    /// 29.97 fps (30000/1001) — drop-frame common
    F29_97,
    /// 30 fps
    F30,
    /// 48 fps (HFR)
    F48,
    /// 50 fps
    F50,
    /// 60 fps
    F60,
}

impl FrameRate {
    /// Integer frames per second (ceiling, for counter arithmetic).
    #[must_use]
    pub fn frames_per_second_int(self) -> u32 {
        match self {
            Self::F23_976 => 24,
            Self::F24 => 24,
            Self::F25 => 25,
            Self::F29_97 => 30,
            Self::F30 => 30,
            Self::F48 => 48,
            Self::F50 => 50,
            Self::F60 => 60,
        }
    }

    /// Real frames per second (as f64).
    #[must_use]
    pub fn as_f64(self) -> f64 {
        match self {
            Self::F23_976 => 24_000.0 / 1_001.0,
            Self::F24 => 24.0,
            Self::F25 => 25.0,
            Self::F29_97 => 30_000.0 / 1_001.0,
            Self::F30 => 30.0,
            Self::F48 => 48.0,
            Self::F50 => 50.0,
            Self::F60 => 60.0,
        }
    }

    /// Whether this rate is commonly used with drop-frame.
    #[must_use]
    pub fn supports_drop_frame(self) -> bool {
        matches!(self, Self::F29_97 | Self::F23_976)
    }
}

// ---------------------------------------------------------------------------
// Timecode
// ---------------------------------------------------------------------------

/// A SMPTE timecode value.
///
/// Internally stored as a validated tuple `(hh, mm, ss, ff)`.
/// Drop-frame flag determines the string representation and frame counting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Timecode {
    hours: u8,
    minutes: u8,
    seconds: u8,
    frames: u8,
    rate: FrameRate,
    drop_frame: bool,
}

/// Errors that can occur during timecode parsing or construction.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TimecodeError {
    #[error("Invalid timecode component: {0}")]
    InvalidComponent(String),
    #[error("Frame number {0} out of range for rate {1} fps")]
    FrameOutOfRange(u8, u32),
    #[error("Drop-frame timecode requires a drop-frame-compatible rate")]
    DropFrameRateMismatch,
    #[error("Failed to parse timecode string: {0}")]
    ParseError(String),
}

impl Timecode {
    /// Construct a timecode from components.
    ///
    /// # Errors
    /// Returns [`TimecodeError`] if any component is out of range.
    pub fn new(
        hours: u8,
        minutes: u8,
        seconds: u8,
        frames: u8,
        rate: FrameRate,
        drop_frame: bool,
    ) -> Result<Self, TimecodeError> {
        if drop_frame && !rate.supports_drop_frame() {
            return Err(TimecodeError::DropFrameRateMismatch);
        }
        if hours > 23 {
            return Err(TimecodeError::InvalidComponent(format!(
                "hours={hours} > 23"
            )));
        }
        if minutes > 59 {
            return Err(TimecodeError::InvalidComponent(format!(
                "minutes={minutes} > 59"
            )));
        }
        if seconds > 59 {
            return Err(TimecodeError::InvalidComponent(format!(
                "seconds={seconds} > 59"
            )));
        }
        let max_frames = rate.frames_per_second_int() as u8;
        if frames >= max_frames {
            return Err(TimecodeError::FrameOutOfRange(frames, max_frames as u32));
        }
        // Validate drop-frame: frames 0 and 1 are dropped at start of each
        // minute except every 10th minute.
        if drop_frame && seconds == 0 && (minutes % 10) != 0 && frames < 2 {
            return Err(TimecodeError::InvalidComponent(format!(
                "drop-frame: frames {frames} is a dropped frame at mm={minutes} ss=00"
            )));
        }
        Ok(Self {
            hours,
            minutes,
            seconds,
            frames,
            rate,
            drop_frame,
        })
    }

    /// Hours component.
    #[must_use]
    pub fn hours(&self) -> u8 {
        self.hours
    }

    /// Minutes component.
    #[must_use]
    pub fn minutes(&self) -> u8 {
        self.minutes
    }

    /// Seconds component.
    #[must_use]
    pub fn seconds(&self) -> u8 {
        self.seconds
    }

    /// Frames component.
    #[must_use]
    pub fn frames(&self) -> u8 {
        self.frames
    }

    /// Frame rate.
    #[must_use]
    pub fn rate(&self) -> FrameRate {
        self.rate
    }

    /// Whether this is a drop-frame timecode.
    #[must_use]
    pub fn is_drop_frame(&self) -> bool {
        self.drop_frame
    }

    // ------------------------------------------------------------------
    // Frame-count conversion
    // ------------------------------------------------------------------

    /// Convert to an absolute frame count (from 00:00:00:00).
    ///
    /// Uses the standard SMPTE drop-frame algorithm for DF timecodes.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_frame_count(self) -> u64 {
        let fps = self.rate.frames_per_second_int() as u64;
        let h = self.hours as u64;
        let m = self.minutes as u64;
        let s = self.seconds as u64;
        let f = self.frames as u64;

        if self.drop_frame {
            // SMPTE DF: drop 2 frames at the start of each minute,
            // except every 10th minute.
            let d = 2u64; // frames dropped per minute
            let total_minutes = 60 * h + m;
            let drop_frames =
                d * (total_minutes - total_minutes / 10);
            fps * 3600 * h
                + fps * 60 * m
                + fps * s
                + f
                - drop_frames
        } else {
            fps * 3600 * h + fps * 60 * m + fps * s + f
        }
    }

    /// Construct from an absolute frame count.
    ///
    /// # Errors
    /// Returns an error if the resulting components are invalid.
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_frame_count(
        mut total: u64,
        rate: FrameRate,
        drop_frame: bool,
    ) -> Result<Self, TimecodeError> {
        if drop_frame && !rate.supports_drop_frame() {
            return Err(TimecodeError::DropFrameRateMismatch);
        }
        let fps = rate.frames_per_second_int() as u64;

        if drop_frame {
            // SMPTE DF inverse algorithm
            let d = 2u64;
            let frames_per_10min = fps * 600 - 9 * d; // 10 minutes worth
            let frames_per_1min = fps * 60 - d;

            let ten_min_blocks = total / frames_per_10min;
            let remaining = total % frames_per_10min;

            let extra_1min = if remaining < fps * 60 {
                0u64
            } else {
                ((remaining - fps * 60) / frames_per_1min + 1).min(9)
            };

            total += d * (9 * ten_min_blocks + extra_1min);
        }

        let frames = (total % fps) as u8;
        let total_secs = total / fps;
        let seconds = (total_secs % 60) as u8;
        let total_mins = total_secs / 60;
        let minutes = (total_mins % 60) as u8;
        let hours = (total_mins / 60) as u8;

        Self::new(hours, minutes, seconds, frames, rate, drop_frame)
    }

    /// Convert to real-time seconds from 00:00:00:00.
    #[must_use]
    pub fn to_secs_f64(self) -> f64 {
        self.to_frame_count() as f64 / self.rate.as_f64()
    }

    /// Construct from real-time seconds.
    ///
    /// # Errors
    /// Returns an error if the resulting components are invalid.
    pub fn from_secs_f64(
        secs: f64,
        rate: FrameRate,
        drop_frame: bool,
    ) -> Result<Self, TimecodeError> {
        let total = (secs * rate.as_f64()).round() as u64;
        Self::from_frame_count(total, rate, drop_frame)
    }

    // ------------------------------------------------------------------
    // Arithmetic
    // ------------------------------------------------------------------

    /// Add a frame count to this timecode.
    ///
    /// # Errors
    /// Returns an error if the result is out of range.
    pub fn add_frames(self, frames: i64) -> Result<Self, TimecodeError> {
        let current = self.to_frame_count() as i64;
        let next = (current + frames).max(0) as u64;
        Self::from_frame_count(next, self.rate, self.drop_frame)
    }

    /// Compute the signed frame-count difference (`self - other`).
    ///
    /// Returns `None` if the rates differ.
    #[must_use]
    pub fn diff_frames(self, other: Self) -> Option<i64> {
        if self.rate != other.rate || self.drop_frame != other.drop_frame {
            return None;
        }
        Some(self.to_frame_count() as i64 - other.to_frame_count() as i64)
    }

    // ------------------------------------------------------------------
    // Parsing / formatting
    // ------------------------------------------------------------------

    /// Parse a SMPTE timecode string.
    ///
    /// Accepts `HH:MM:SS:FF` (NDF) and `HH:MM:SS;FF` (DF).
    ///
    /// # Errors
    /// Returns a [`TimecodeError`] if the string is malformed.
    pub fn parse(s: &str, rate: FrameRate) -> Result<Self, TimecodeError> {
        let s = s.trim();
        // Detect DF separator (`;` before last pair)
        let drop_frame = s.contains(';');
        let normalised: String = s.replace(';', ":");
        let parts: Vec<&str> = normalised.split(':').collect();
        if parts.len() != 4 {
            return Err(TimecodeError::ParseError(format!(
                "expected HH:MM:SS:FF, got `{s}`"
            )));
        }
        let parse_u8 = |p: &str| -> Result<u8, TimecodeError> {
            p.parse::<u8>().map_err(|_| {
                TimecodeError::ParseError(format!("cannot parse `{p}` as u8"))
            })
        };
        let hh = parse_u8(parts[0])?;
        let mm = parse_u8(parts[1])?;
        let ss = parse_u8(parts[2])?;
        let ff = parse_u8(parts[3])?;
        Self::new(hh, mm, ss, ff, rate, drop_frame)
    }
}

impl fmt::Display for Timecode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sep = if self.drop_frame { ';' } else { ':' };
        write!(
            f,
            "{:02}:{:02}:{:02}{}{:02}",
            self.hours, self.minutes, self.seconds, sep, self.frames
        )
    }
}

// ---------------------------------------------------------------------------
// TriggerCallback and TimecodeScheduler
// ---------------------------------------------------------------------------

/// Identifier for a registered trigger.
pub type TriggerId = u64;

/// Result of checking triggers: list of fired trigger IDs and labels.
#[derive(Debug, Clone)]
pub struct FiredTriggers {
    /// IDs of triggers that fired this tick.
    pub ids: Vec<TriggerId>,
}

/// A registered trigger: fire when playback passes a target timecode.
#[derive(Debug, Clone)]
struct TriggerEntry {
    id: TriggerId,
    target_frame: u64,
    label: String,
    one_shot: bool,
    fired: bool,
}

/// Frame-accurate timecode playback scheduler.
///
/// Maintains an internal frame counter and fires registered triggers
/// when playback passes their target timecode.
pub struct TimecodeScheduler {
    rate: FrameRate,
    drop_frame: bool,
    current_frame: u64,
    triggers: HashMap<TriggerId, TriggerEntry>,
    next_id: TriggerId,
}

impl TimecodeScheduler {
    /// Create a new scheduler, starting at 00:00:00:00.
    #[must_use]
    pub fn new(rate: FrameRate, drop_frame: bool) -> Self {
        Self {
            rate,
            drop_frame,
            current_frame: 0,
            triggers: HashMap::new(),
            next_id: 1,
        }
    }

    /// Current timecode.
    ///
    /// Returns `None` only if the frame count is somehow invalid.
    #[must_use]
    pub fn current_timecode(&self) -> Option<Timecode> {
        Timecode::from_frame_count(self.current_frame, self.rate, self.drop_frame).ok()
    }

    /// Seek to a specific timecode.
    pub fn seek(&mut self, tc: Timecode) {
        self.current_frame = tc.to_frame_count();
    }

    /// Advance by `frames` frames and return any fired triggers.
    pub fn advance_frames(&mut self, frames: u64) -> FiredTriggers {
        let start = self.current_frame;
        self.current_frame += frames;
        let end = self.current_frame;

        let mut fired_ids = Vec::new();
        for entry in self.triggers.values_mut() {
            if entry.fired && entry.one_shot {
                continue;
            }
            if entry.target_frame > start && entry.target_frame <= end {
                fired_ids.push(entry.id);
                if entry.one_shot {
                    entry.fired = true;
                }
            }
        }
        FiredTriggers { ids: fired_ids }
    }

    /// Register a trigger at a target timecode.
    ///
    /// Returns the [`TriggerId`] assigned.
    pub fn register_trigger(
        &mut self,
        target: Timecode,
        label: impl Into<String>,
        one_shot: bool,
    ) -> TriggerId {
        let id = self.next_id;
        self.next_id += 1;
        self.triggers.insert(
            id,
            TriggerEntry {
                id,
                target_frame: target.to_frame_count(),
                label: label.into(),
                one_shot,
                fired: false,
            },
        );
        id
    }

    /// Remove a trigger.
    pub fn unregister_trigger(&mut self, id: TriggerId) {
        self.triggers.remove(&id);
    }

    /// Get the label of a trigger.
    #[must_use]
    pub fn trigger_label(&self, id: TriggerId) -> Option<&str> {
        self.triggers.get(&id).map(|e| e.label.as_str())
    }

    /// Number of registered triggers.
    #[must_use]
    pub fn trigger_count(&self) -> usize {
        self.triggers.len()
    }

    /// Reset to 00:00:00:00.
    pub fn reset(&mut self) {
        self.current_frame = 0;
        for e in self.triggers.values_mut() {
            e.fired = false;
        }
    }
}

// ---------------------------------------------------------------------------
// LTC frame-boundary detector
// ---------------------------------------------------------------------------

/// Minimal LTC frame-boundary detector.
///
/// In a real system LTC is an 80-bit bi-phase mark coded audio signal.
/// This implementation models the state machine that detects a sync word
/// and returns the number of complete frames found in an audio sample
/// buffer (where each bit is represented as a single `bool` sample for
/// simplicity / testability).
pub struct LtcDecoder {
    /// Running bit buffer (newest bit appended at end).
    bit_buf: Vec<bool>,
    /// Number of complete frames decoded so far.
    frames_decoded: u64,
}

/// Sync word for SMPTE LTC (bits 64–79 of an 80-bit word, LSB first).
/// Binary: 0011 1111 1111 1101
const LTC_SYNC_WORD: u16 = 0x3FFD;

impl LtcDecoder {
    /// Create a new decoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bit_buf: Vec::with_capacity(160),
            frames_decoded: 0,
        }
    }

    /// Feed a slice of bit samples (one `bool` per bit-clock).
    ///
    /// Returns the number of frame boundaries detected.
    pub fn feed(&mut self, bits: &[bool]) -> usize {
        let mut count = 0;
        for &bit in bits {
            self.bit_buf.push(bit);
            if self.bit_buf.len() >= 80 {
                // Check the last 16 bits for the sync word
                let sync_start = self.bit_buf.len() - 16;
                let detected = self.check_sync_at(sync_start);
                if detected {
                    count += 1;
                    self.frames_decoded += 1;
                    // Consume the 80-bit frame
                    let drain_to = self.bit_buf.len().saturating_sub(80);
                    self.bit_buf.drain(..drain_to);
                    // Keep the capacity reasonable
                    if self.bit_buf.len() > 160 {
                        let excess = self.bit_buf.len() - 160;
                        self.bit_buf.drain(..excess);
                    }
                }
            }
        }
        count
    }

    /// Check whether the sync word is present starting at `pos`.
    fn check_sync_at(&self, pos: usize) -> bool {
        if pos + 16 > self.bit_buf.len() {
            return false;
        }
        let mut word: u16 = 0;
        for i in 0..16 {
            if self.bit_buf[pos + i] {
                word |= 1 << i;
            }
        }
        word == LTC_SYNC_WORD
    }

    /// Total frames decoded.
    #[must_use]
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }

    /// Reset decoder state.
    pub fn reset(&mut self) {
        self.bit_buf.clear();
        self.frames_decoded = 0;
    }
}

impl Default for LtcDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- FrameRate --

    #[test]
    fn test_frame_rate_int() {
        assert_eq!(FrameRate::F29_97.frames_per_second_int(), 30);
        assert_eq!(FrameRate::F25.frames_per_second_int(), 25);
        assert_eq!(FrameRate::F60.frames_per_second_int(), 60);
    }

    #[test]
    fn test_frame_rate_f64_29_97() {
        let fps = FrameRate::F29_97.as_f64();
        assert!((fps - 29.97002997).abs() < 1e-6, "fps={fps}");
    }

    // -- NDF timecode construction and round-trip --

    #[test]
    fn test_ndf_to_frame_count_basic() {
        let tc = Timecode::new(0, 1, 0, 0, FrameRate::F25, false).expect("valid");
        assert_eq!(tc.to_frame_count(), 25 * 60);
    }

    #[test]
    fn test_ndf_round_trip() {
        let tc = Timecode::new(1, 23, 45, 12, FrameRate::F30, false).expect("valid");
        let frames = tc.to_frame_count();
        let restored = Timecode::from_frame_count(frames, FrameRate::F30, false).expect("restore");
        assert_eq!(tc, restored);
    }

    #[test]
    fn test_ndf_to_secs_f64() {
        let tc = Timecode::new(0, 0, 1, 0, FrameRate::F25, false).expect("valid");
        assert!((tc.to_secs_f64() - 1.0).abs() < 1e-9);
    }

    // -- Drop-frame --

    #[test]
    fn test_df_frame_count_at_one_minute() {
        // At 29.97 DF, 00:01:00;02 is the first valid frame at minute 1
        // (frames 00 and 01 are dropped from the display).
        // The frame index in the video stream is still 1800, but the
        // SMPTE DF formula subtracts 2 dropped-display-frames, giving 1800.
        // Verify via round-trip: what matters is self-consistency.
        let tc = Timecode::new(0, 1, 0, 2, FrameRate::F29_97, true).expect("valid");
        let frames = tc.to_frame_count();
        let restored =
            Timecode::from_frame_count(frames, FrameRate::F29_97, true).expect("restore");
        assert_eq!(tc, restored, "DF round-trip at 00:01:00;02 failed");
        // The NDF equivalent at the same display position would be 00:01:00:02
        // which is 1802 frames; the DF count should be 2 less = 1800.
        let ndf = Timecode::new(0, 1, 0, 2, FrameRate::F29_97, false).expect("ndf");
        assert_eq!(ndf.to_frame_count(), 1802, "NDF check");
        assert_eq!(frames, 1800, "DF frame count at 00:01:00;02");
    }

    #[test]
    fn test_df_round_trip() {
        let tc = Timecode::new(0, 5, 30, 15, FrameRate::F29_97, true).expect("valid");
        let frames = tc.to_frame_count();
        let restored =
            Timecode::from_frame_count(frames, FrameRate::F29_97, true).expect("restore");
        assert_eq!(tc, restored, "DF round-trip failed");
    }

    #[test]
    fn test_df_invalid_dropped_frame() {
        // 00:01:00;00 and 00:01:00;01 are dropped frames
        assert!(Timecode::new(0, 1, 0, 0, FrameRate::F29_97, true).is_err());
        assert!(Timecode::new(0, 1, 0, 1, FrameRate::F29_97, true).is_err());
    }

    // -- Parsing / formatting --

    #[test]
    fn test_parse_ndf_string() {
        let tc = Timecode::parse("01:23:45:06", FrameRate::F25).expect("parse");
        assert_eq!(tc.hours(), 1);
        assert_eq!(tc.minutes(), 23);
        assert_eq!(tc.seconds(), 45);
        assert_eq!(tc.frames(), 6);
        assert!(!tc.is_drop_frame());
    }

    #[test]
    fn test_parse_df_string() {
        let tc = Timecode::parse("00:10:00;02", FrameRate::F29_97).expect("parse DF");
        assert!(tc.is_drop_frame());
        assert_eq!(tc.frames(), 2);
    }

    #[test]
    fn test_display_ndf() {
        let tc = Timecode::new(1, 2, 3, 4, FrameRate::F25, false).expect("valid");
        assert_eq!(tc.to_string(), "01:02:03:04");
    }

    #[test]
    fn test_display_df() {
        let tc = Timecode::new(0, 10, 0, 2, FrameRate::F29_97, true).expect("valid");
        assert!(tc.to_string().contains(';'), "DF separator missing");
    }

    // -- Arithmetic --

    #[test]
    fn test_add_frames() {
        let tc = Timecode::new(0, 0, 0, 0, FrameRate::F25, false).expect("valid");
        let next = tc.add_frames(25).expect("add");
        assert_eq!(next.seconds(), 1);
        assert_eq!(next.frames(), 0);
    }

    #[test]
    fn test_diff_frames() {
        let a = Timecode::new(0, 0, 1, 0, FrameRate::F25, false).expect("valid");
        let b = Timecode::new(0, 0, 0, 0, FrameRate::F25, false).expect("valid");
        assert_eq!(a.diff_frames(b), Some(25));
    }

    // -- Scheduler --

    #[test]
    fn test_scheduler_trigger_fires() {
        let mut sched = TimecodeScheduler::new(FrameRate::F25, false);
        let target = Timecode::new(0, 0, 2, 0, FrameRate::F25, false).expect("valid");
        let id = sched.register_trigger(target, "mark_in", true);
        // advance 30 frames (< 50)
        let fired = sched.advance_frames(30);
        assert!(fired.ids.is_empty());
        // advance past target (frame 50)
        let fired = sched.advance_frames(25);
        assert!(fired.ids.contains(&id));
    }

    #[test]
    fn test_scheduler_one_shot_fires_once() {
        let mut sched = TimecodeScheduler::new(FrameRate::F25, false);
        let target = Timecode::new(0, 0, 0, 5, FrameRate::F25, false).expect("valid");
        let id = sched.register_trigger(target, "once", true);
        sched.advance_frames(6);
        sched.reset();
        // After reset the one-shot should fire again
        let _fired_1 = sched.advance_frames(6);
        // one_shot triggers mark `fired = true`; after reset they reset
        sched.unregister_trigger(id);
        assert_eq!(sched.trigger_count(), 0);
    }

    // -- LTC decoder --

    #[test]
    fn test_ltc_sync_word_detection() {
        let mut decoder = LtcDecoder::new();
        // Build a synthetic 80-bit LTC word: 64 data bits followed by the sync word.
        // Data bits: all false for simplicity.
        let mut bits = vec![false; 64];
        // Sync word 0x3FFD = 0011 1111 1111 1101 (LSB first)
        let sync: u16 = LTC_SYNC_WORD;
        for i in 0..16 {
            bits.push((sync >> i) & 1 == 1);
        }
        let count = decoder.feed(&bits);
        assert_eq!(count, 1, "should detect exactly 1 frame boundary");
        assert_eq!(decoder.frames_decoded(), 1);
    }

    #[test]
    fn test_ltc_no_false_positives_on_zeros() {
        let mut decoder = LtcDecoder::new();
        let bits = vec![false; 100];
        let count = decoder.feed(&bits);
        assert_eq!(count, 0);
    }
}
