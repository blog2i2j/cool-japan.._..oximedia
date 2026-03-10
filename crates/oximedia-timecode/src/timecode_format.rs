#![allow(dead_code)]
//! Timecode display format and parsing utilities.

/// The visual format used to display a timecode value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimecodeFormat {
    /// SMPTE HH:MM:SS:FF / HH:MM:SS;FF notation.
    Smpte,
    /// Film-style feet+frames (e.g. `1234+08`).
    Feet,
    /// Absolute frame count as a plain integer.
    Frames,
    /// Wall-clock seconds expressed as a decimal (e.g. `3723.04`).
    Seconds,
}

impl TimecodeFormat {
    /// The separator character used between the seconds and frames fields
    /// for SMPTE notation.  Non-SMPTE formats return `':'` as a placeholder.
    pub fn separator(&self) -> char {
        match self {
            TimecodeFormat::Smpte => ':',
            _ => ':',
        }
    }

    /// Separator for drop-frame SMPTE timecode.
    pub fn drop_frame_separator(&self) -> char {
        match self {
            TimecodeFormat::Smpte => ';',
            _ => ':',
        }
    }
}

/// Formats and parses timecode values in various [`TimecodeFormat`]s.
#[derive(Debug, Clone)]
pub struct TimecodeFormatter {
    /// The display format to use.
    pub format: TimecodeFormat,
    /// Frames per second (nominal integer, e.g. 25 or 30).
    pub fps: u32,
    /// Whether the timecode uses drop-frame counting.
    pub drop_frame: bool,
}

impl TimecodeFormatter {
    /// Create a new formatter.
    ///
    /// `fps` must be non-zero; returns `None` otherwise.
    pub fn new(format: TimecodeFormat, fps: u32, drop_frame: bool) -> Option<Self> {
        if fps == 0 {
            None
        } else {
            Some(Self {
                format,
                fps,
                drop_frame,
            })
        }
    }

    /// Convert an absolute frame count into a human-readable string.
    pub fn format_frames(&self, total_frames: u64) -> String {
        match self.format {
            TimecodeFormat::Frames => format!("{}", total_frames),

            TimecodeFormat::Seconds => {
                let secs = total_frames as f64 / self.fps as f64;
                format!("{:.3}", secs)
            }

            TimecodeFormat::Feet => {
                // 35 mm film: 16 frames per foot.
                let feet = total_frames / 16;
                let rem = total_frames % 16;
                format!("{}+{:02}", feet, rem)
            }

            TimecodeFormat::Smpte => {
                let fps = self.fps as u64;
                let hours = total_frames / (fps * 3600);
                let rem = total_frames % (fps * 3600);
                let minutes = rem / (fps * 60);
                let rem = rem % (fps * 60);
                let seconds = rem / fps;
                let frames = rem % fps;

                let sep = if self.drop_frame { ';' } else { ':' };
                format!(
                    "{:02}:{:02}:{:02}{}{:02}",
                    hours, minutes, seconds, sep, frames
                )
            }
        }
    }

    /// Parse a SMPTE timecode string (HH:MM:SS:FF or HH:MM:SS;FF) into total
    /// frames using the formatter's `fps`.
    ///
    /// Returns `None` if the string is not valid SMPTE notation.
    pub fn parse_smpte(&self, s: &str) -> Option<u64> {
        // Accept both ':' and ';' as separators between SS and FF.
        let normalized: String = s.chars().map(|c| if c == ';' { ':' } else { c }).collect();
        let parts: Vec<&str> = normalized.split(':').collect();
        if parts.len() != 4 {
            return None;
        }

        let h: u64 = parts[0].parse().ok()?;
        let m: u64 = parts[1].parse().ok()?;
        let sec: u64 = parts[2].parse().ok()?;
        let f: u64 = parts[3].parse().ok()?;

        if m >= 60 || sec >= 60 || f >= self.fps as u64 {
            return None;
        }

        let fps = self.fps as u64;
        Some(h * 3600 * fps + m * 60 * fps + sec * fps + f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_enum_separator() {
        assert_eq!(TimecodeFormat::Smpte.separator(), ':');
        assert_eq!(TimecodeFormat::Frames.separator(), ':');
    }

    #[test]
    fn test_drop_frame_separator() {
        assert_eq!(TimecodeFormat::Smpte.drop_frame_separator(), ';');
        assert_eq!(TimecodeFormat::Feet.drop_frame_separator(), ':');
    }

    #[test]
    fn test_formatter_new_zero_fps_returns_none() {
        assert!(TimecodeFormatter::new(TimecodeFormat::Smpte, 0, false).is_none());
    }

    #[test]
    fn test_format_frames_frames() {
        let fmt = TimecodeFormatter::new(TimecodeFormat::Frames, 25, false)
            .expect("valid timecode formatter");
        assert_eq!(fmt.format_frames(1234), "1234");
    }

    #[test]
    fn test_format_frames_seconds() {
        let fmt = TimecodeFormatter::new(TimecodeFormat::Seconds, 25, false)
            .expect("valid timecode formatter");
        // 25 frames at 25 fps = 1.000 s
        assert_eq!(fmt.format_frames(25), "1.000");
    }

    #[test]
    fn test_format_frames_feet() {
        let fmt = TimecodeFormatter::new(TimecodeFormat::Feet, 24, false)
            .expect("valid timecode formatter");
        // 16 frames = 1 foot + 0 frames
        assert_eq!(fmt.format_frames(16), "1+00");
        // 17 frames = 1 foot + 1 frame
        assert_eq!(fmt.format_frames(17), "1+01");
    }

    #[test]
    fn test_format_frames_smpte_ndf() {
        let fmt = TimecodeFormatter::new(TimecodeFormat::Smpte, 25, false)
            .expect("valid timecode formatter");
        // 1 hour = 25 * 3600 = 90000 frames
        assert_eq!(fmt.format_frames(90000), "01:00:00:00");
    }

    #[test]
    fn test_format_frames_smpte_df() {
        let fmt = TimecodeFormatter::new(TimecodeFormat::Smpte, 30, true)
            .expect("valid timecode formatter");
        // Frame 30 → 0h 0m 1s 0f
        assert_eq!(fmt.format_frames(30), "00:00:01;00");
    }

    #[test]
    fn test_format_frames_smpte_mixed() {
        let fmt = TimecodeFormatter::new(TimecodeFormat::Smpte, 25, false)
            .expect("valid timecode formatter");
        // 01:02:03:04 = (1*3600 + 2*60 + 3)*25 + 4 = (3723)*25+4 = 93079
        assert_eq!(fmt.format_frames(93079), "01:02:03:04");
    }

    #[test]
    fn test_parse_smpte_valid_colon() {
        let fmt = TimecodeFormatter::new(TimecodeFormat::Smpte, 25, false)
            .expect("valid timecode formatter");
        let frames = fmt.parse_smpte("01:02:03:04").expect("valid SMPTE parse");
        // Should round-trip with format_frames
        assert_eq!(fmt.format_frames(frames), "01:02:03:04");
    }

    #[test]
    fn test_parse_smpte_valid_semicolon() {
        let fmt = TimecodeFormatter::new(TimecodeFormat::Smpte, 30, true)
            .expect("valid timecode formatter");
        let frames = fmt.parse_smpte("00:00:01;00").expect("valid SMPTE parse");
        assert_eq!(frames, 30);
    }

    #[test]
    fn test_parse_smpte_invalid_too_few_parts() {
        let fmt = TimecodeFormatter::new(TimecodeFormat::Smpte, 25, false)
            .expect("valid timecode formatter");
        assert!(fmt.parse_smpte("01:02:03").is_none());
    }

    #[test]
    fn test_parse_smpte_invalid_frames_exceed_fps() {
        let fmt = TimecodeFormatter::new(TimecodeFormat::Smpte, 25, false)
            .expect("valid timecode formatter");
        assert!(fmt.parse_smpte("00:00:00:25").is_none());
    }

    #[test]
    fn test_parse_smpte_invalid_minutes() {
        let fmt = TimecodeFormatter::new(TimecodeFormat::Smpte, 25, false)
            .expect("valid timecode formatter");
        assert!(fmt.parse_smpte("00:60:00:00").is_none());
    }
}
