//! Drop-frame timecode calculations for 29.97 fps NTSC.
//!
//! Drop-frame timecode is a system that keeps timecode aligned with actual
//! elapsed time for 29.97 fps video by "dropping" (skipping) frame numbers
//! 00 and 01 at the start of each minute, except every 10th minute.
//!
//! Note: "drop frame" refers to dropping frame *numbers*, not actual video frames.

use std::fmt;

/// Drop-frame timecode calculator for 29.97 fps.
///
/// 29.97 fps = 30000/1001 fps. To keep timecode aligned with real time,
/// 2 frame numbers are dropped per minute, except every 10th minute.
/// Actual frame count per 24-hour day: 24 * 107892 = 2,589,408 frames.
#[allow(dead_code)]
pub struct DropFrameCalc;

impl DropFrameCalc {
    // Constants for drop-frame calculation
    const FRAMES_PER_SEC: u64 = 30;
    const DROP_PER_MIN: u64 = 2;
    // Actual frames in one minute (29 drop minutes) = 30*60 - 2 = 1798
    const FRAMES_PER_DROP_MIN: u64 = Self::FRAMES_PER_SEC * 60 - Self::DROP_PER_MIN; // 1798
                                                                                     // Actual frames in 10 minutes = 9 drop minutes + 1 non-drop minute
                                                                                     // = 9 * 1798 + 1800 = 17982
    const FRAMES_PER_10_MIN: u64 = Self::FRAMES_PER_DROP_MIN * 9 + Self::FRAMES_PER_SEC * 60; // 17982
                                                                                              // Actual frames per hour = 6 * 17982 = 107892
    const FRAMES_PER_HOUR: u64 = Self::FRAMES_PER_10_MIN * 6; // 107892

    /// Convert a frame count to drop-frame timecode (hh, mm, ss, ff).
    ///
    /// Uses the standard SMPTE drop-frame algorithm.
    ///
    /// Reference algorithm (from SMPTE 12M):
    /// D = frame_count
    /// D_f = D + 2 * (D / 17982) + 2 * ((D % 17982 - 2) / 1798)  [only if remainder >= 2]
    #[must_use]
    pub fn frame_count_to_df(frame_count: u64) -> (u8, u8, u8, u8) {
        // Wrap at 24 hours
        let d = frame_count % (Self::FRAMES_PER_HOUR * 24);

        // Number of complete 10-minute blocks
        let d_ten = d / Self::FRAMES_PER_10_MIN;
        let d_in_ten = d % Self::FRAMES_PER_10_MIN;

        // Within each 10-minute block:
        // First 1800 frames (minute 0 of block) = non-drop minute
        // Remaining 9 * 1798 frames = 9 drop minutes
        let (min_in_ten, d_in_min) = if d_in_ten < Self::FRAMES_PER_SEC * 60 {
            (0u64, d_in_ten)
        } else {
            let d_after_first = d_in_ten - Self::FRAMES_PER_SEC * 60;
            let extra_min = d_after_first / Self::FRAMES_PER_DROP_MIN;
            let d_in_drop_min = d_after_first % Self::FRAMES_PER_DROP_MIN;
            (extra_min + 1, d_in_drop_min)
        };

        let total_minutes = d_ten * 10 + min_in_ten;
        let hh = (total_minutes / 60) as u8;
        let mm = (total_minutes % 60) as u8;

        // Within the (drop) minute:
        // For non-first minutes within a 10-min block, frames 0 and 1 are dropped,
        // so the minute starts at frame number 2.
        let (ss, ff) = if min_in_ten > 0 {
            // Frame numbers 0 and 1 were skipped; actual frames start at 2
            // d_in_min=0 corresponds to display frame 2 at second 0
            let adjusted = d_in_min + Self::DROP_PER_MIN;
            let ss = adjusted / Self::FRAMES_PER_SEC;
            let ff = adjusted % Self::FRAMES_PER_SEC;
            (ss as u8, ff as u8)
        } else {
            // Non-drop minute (first of every 10)
            let ss = d_in_min / Self::FRAMES_PER_SEC;
            let ff = d_in_min % Self::FRAMES_PER_SEC;
            (ss as u8, ff as u8)
        };

        (hh, mm, ss, ff)
    }

    /// Convert drop-frame timecode (hh, mm, ss, ff) to a frame count.
    ///
    /// Standard SMPTE formula:
    /// frame_count = 108000*hh + 1800*mm + 30*ss + ff
    ///               - 2*(total_minutes - total_minutes/10)
    ///
    /// Note: 108000 = 30 * 3600 (raw 30fps hour count, NOT the drop-frame hour count).
    #[must_use]
    pub fn df_to_frame_count(hh: u8, mm: u8, ss: u8, ff: u8) -> u64 {
        let hh = u64::from(hh);
        let mm = u64::from(mm);
        let ss = u64::from(ss);
        let ff = u64::from(ff);

        let total_minutes = hh * 60 + mm;

        // Raw count as if 30 fps non-drop (108000 per hour, 1800 per minute)
        let raw = hh * 108000 + mm * 1800 + ss * 30 + ff;

        // Subtract 2 frames per minute except every 10th minute
        let dropped = Self::DROP_PER_MIN * (total_minutes - total_minutes / 10);

        raw - dropped
    }

    /// Format a frame count as a drop-frame timecode string.
    ///
    /// Drop-frame timecode uses semicolons (;) as separators.
    #[must_use]
    pub fn format_df(frame_count: u64) -> String {
        let (hh, mm, ss, ff) = Self::frame_count_to_df(frame_count);
        format!("{hh:02};{mm:02};{ss:02};{ff:02}")
    }

    /// Parse a drop-frame timecode string (HH;MM;SS;FF) into a frame count.
    ///
    /// Returns `None` if the string is not a valid drop-frame timecode.
    #[must_use]
    pub fn parse_df(tc: &str) -> Option<u64> {
        let parts: Vec<&str> = tc.split(';').collect();
        if parts.len() != 4 {
            return None;
        }

        let hh: u8 = parts[0].parse().ok()?;
        let mm: u8 = parts[1].parse().ok()?;
        let ss: u8 = parts[2].parse().ok()?;
        let ff: u8 = parts[3].parse().ok()?;

        if hh > 23 || mm > 59 || ss > 59 || ff > 29 {
            return None;
        }

        Some(Self::df_to_frame_count(hh, mm, ss, ff))
    }

    /// Check whether a given timecode position is a dropped frame number.
    ///
    /// Frames 0 and 1 at the start of each minute (except multiples of 10) are dropped.
    #[must_use]
    pub fn is_dropped_frame(hh: u8, mm: u8, ss: u8, ff: u8) -> bool {
        let _ = hh; // Hours don't affect drop-frame logic
        ss == 0 && ff < 2 && !mm.is_multiple_of(10)
    }
}

/// Frame counter that supports both drop-frame and non-drop-frame modes.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TotalFrameCounter {
    /// Current frame count
    frame_count: u64,
    /// Whether to use drop-frame mode
    drop_frame: bool,
    /// Frame rate (integer, typically 30 for DF)
    fps: u8,
}

impl TotalFrameCounter {
    /// Create a new frame counter in drop-frame mode.
    #[must_use]
    pub fn new_drop_frame() -> Self {
        Self {
            frame_count: 0,
            drop_frame: true,
            fps: 30,
        }
    }

    /// Create a new frame counter in non-drop-frame mode.
    #[must_use]
    pub fn new_non_drop_frame(fps: u8) -> Self {
        Self {
            frame_count: 0,
            drop_frame: false,
            fps,
        }
    }

    /// Add frames to the counter.
    pub fn add_frames(&mut self, n: u64) {
        self.frame_count = self.frame_count.wrapping_add(n);
    }

    /// Get the current frame count.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Reset the counter.
    pub fn reset(&mut self) {
        self.frame_count = 0;
    }

    /// Check if drop-frame mode is active.
    #[must_use]
    pub fn is_drop_frame(&self) -> bool {
        self.drop_frame
    }
}

impl fmt::Display for TotalFrameCounter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.drop_frame {
            write!(f, "{}", DropFrameCalc::format_df(self.frame_count))
        } else {
            // Non-drop-frame: use colons
            let fps = u64::from(self.fps);
            let seconds_total = self.frame_count / fps;
            let frames = self.frame_count % fps;
            let seconds = seconds_total % 60;
            let minutes_total = seconds_total / 60;
            let minutes = minutes_total % 60;
            let hours = (minutes_total / 60) % 24;
            write!(f, "{hours:02}:{minutes:02}:{seconds:02}:{frames:02}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_df_zero() {
        let (hh, mm, ss, ff) = DropFrameCalc::frame_count_to_df(0);
        assert_eq!((hh, mm, ss, ff), (0, 0, 0, 0));
    }

    #[test]
    fn test_df_one_frame() {
        let (hh, mm, ss, ff) = DropFrameCalc::frame_count_to_df(1);
        assert_eq!((hh, mm, ss, ff), (0, 0, 0, 1));
    }

    #[test]
    fn test_df_one_second() {
        let (hh, mm, ss, ff) = DropFrameCalc::frame_count_to_df(30);
        assert_eq!((hh, mm, ss, ff), (0, 0, 1, 0));
    }

    #[test]
    fn test_df_one_minute() {
        // Minute 0 has 1800 frames (no drop), frame 1800 = first frame of minute 1
        // At minute 1, frames 0 and 1 are dropped, so it starts at 00;01;00;02
        let (hh, mm, ss, ff) = DropFrameCalc::frame_count_to_df(1800);
        assert_eq!(hh, 0);
        assert_eq!(mm, 1);
        assert_eq!(ss, 0);
        assert_eq!(ff, 2); // Frames 0 and 1 are dropped
    }

    #[test]
    fn test_df_ten_minutes() {
        // After 10 minutes - no drop at that boundary
        let frames = DropFrameCalc::df_to_frame_count(0, 10, 0, 0);
        let (hh, mm, ss, ff) = DropFrameCalc::frame_count_to_df(frames);
        assert_eq!((hh, mm, ss, ff), (0, 10, 0, 0));
    }

    #[test]
    fn test_df_roundtrip() {
        let test_cases = [
            (0u8, 0u8, 0u8, 0u8),
            (0, 0, 0, 15),
            (0, 0, 30, 0),
            (0, 10, 0, 0), // 10th minute - no drop
            (1, 0, 0, 0),
        ];

        for (hh, mm, ss, ff) in test_cases {
            let frame_count = DropFrameCalc::df_to_frame_count(hh, mm, ss, ff);
            let (rhh, rmm, rss, rff) = DropFrameCalc::frame_count_to_df(frame_count);
            assert_eq!(
                (hh, mm, ss, ff),
                (rhh, rmm, rss, rff),
                "Roundtrip failed for {hh:02};{mm:02};{ss:02};{ff:02}"
            );
        }
    }

    #[test]
    fn test_format_df() {
        let s = DropFrameCalc::format_df(0);
        assert_eq!(s, "00;00;00;00");
    }

    #[test]
    fn test_parse_df_valid() {
        let count = DropFrameCalc::parse_df("00;00;00;00").expect("should succeed");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_parse_df_invalid() {
        assert!(DropFrameCalc::parse_df("00:00:00:00").is_none()); // colons, not semicolons
        assert!(DropFrameCalc::parse_df("25;00;00;00").is_none()); // hours > 23
        assert!(DropFrameCalc::parse_df("not;a;timecode;x").is_none());
        assert!(DropFrameCalc::parse_df("").is_none());
    }

    #[test]
    fn test_is_dropped_frame() {
        // Minute 1, second 0, frames 0 and 1 are dropped
        assert!(DropFrameCalc::is_dropped_frame(0, 1, 0, 0));
        assert!(DropFrameCalc::is_dropped_frame(0, 1, 0, 1));
        assert!(!DropFrameCalc::is_dropped_frame(0, 1, 0, 2));

        // Minute 10 is NOT dropped
        assert!(!DropFrameCalc::is_dropped_frame(0, 10, 0, 0));
        assert!(!DropFrameCalc::is_dropped_frame(0, 10, 0, 1));

        // Non-zero second is never dropped
        assert!(!DropFrameCalc::is_dropped_frame(0, 1, 1, 0));
    }

    #[test]
    fn test_total_frame_counter_drop_frame() {
        let mut counter = TotalFrameCounter::new_drop_frame();
        assert!(counter.is_drop_frame());
        counter.add_frames(100);
        assert_eq!(counter.frame_count(), 100);
        let s = counter.to_string();
        assert!(s.contains(';')); // Drop frame uses semicolons
    }

    #[test]
    fn test_total_frame_counter_non_drop_frame() {
        let mut counter = TotalFrameCounter::new_non_drop_frame(25);
        assert!(!counter.is_drop_frame());
        counter.add_frames(25); // One second
        assert_eq!(counter.frame_count(), 25);
        let s = counter.to_string();
        assert!(s.contains(':'));
        assert_eq!(s, "00:00:01:00");
    }

    #[test]
    fn test_total_frame_counter_reset() {
        let mut counter = TotalFrameCounter::new_drop_frame();
        counter.add_frames(1000);
        counter.reset();
        assert_eq!(counter.frame_count(), 0);
    }

    #[test]
    fn test_parse_format_roundtrip() {
        let original = "01;05;30;15";
        let frame_count = DropFrameCalc::parse_df(original).expect("should succeed");
        let formatted = DropFrameCalc::format_df(frame_count);
        assert_eq!(formatted, original);
    }
}
