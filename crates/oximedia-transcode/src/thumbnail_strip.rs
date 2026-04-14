//! VTT-compatible sprite sheet / thumbnail strip generator.
//!
//! A [`ThumbnailStrip`] divides a video's timeline into equally-spaced frames,
//! computes the grid position of each frame within a single sprite-sheet image,
//! and can emit a WebVTT cue file pointing into that image with `#xywh` media
//! fragment identifiers.
//!
//! # WebVTT output example
//!
//! ```text
//! WEBVTT
//!
//! 00:00:00.000 --> 00:00:05.000
//! thumbnails.jpg#xywh=0,0,160,90
//!
//! 00:00:05.000 --> 00:00:10.000
//! thumbnails.jpg#xywh=160,0,160,90
//! ```

/// Configuration for a thumbnail strip / sprite sheet.
#[derive(Debug, Clone)]
pub struct ThumbnailStripConfig {
    /// Width of a single thumbnail frame in pixels.
    pub frame_width: u32,
    /// Height of a single thumbnail frame in pixels.
    pub frame_height: u32,
    /// Time between consecutive thumbnails in seconds.
    pub interval_secs: f32,
    /// Number of thumbnail columns in the sprite sheet.
    pub columns: u32,
}

impl ThumbnailStripConfig {
    /// Creates a new config with common defaults (160×90, 5 s interval, 5 columns).
    #[must_use]
    pub fn default_web() -> Self {
        Self {
            frame_width: 160,
            frame_height: 90,
            interval_secs: 5.0,
            columns: 5,
        }
    }
}

/// Grid position and timestamp for a single thumbnail frame.
#[derive(Debug, Clone, PartialEq)]
pub struct ThumbnailPosition {
    /// Timestamp of this frame in seconds.
    pub time_secs: f32,
    /// Zero-based row index in the sprite sheet.
    pub row: u32,
    /// Zero-based column index in the sprite sheet.
    pub col: u32,
    /// Pixel X offset of this frame within the sprite sheet.
    pub x: u32,
    /// Pixel Y offset of this frame within the sprite sheet.
    pub y: u32,
}

/// VTT-compatible sprite sheet descriptor.
///
/// Does not store pixel data — it stores the *layout* (timestamps and grid
/// positions) so that callers can render individual frames and assemble them
/// into the correct positions in a sprite sheet image.
#[derive(Debug, Clone)]
pub struct ThumbnailStrip {
    /// Strip configuration.
    pub config: ThumbnailStripConfig,
    /// Total video duration in seconds.
    pub total_duration_secs: f32,
    /// Pre-computed list of all thumbnail positions.
    positions: Vec<ThumbnailPosition>,
}

impl ThumbnailStrip {
    /// Creates a new [`ThumbnailStrip`] and pre-computes all frame positions.
    ///
    /// `total_duration_secs` must be positive; if it is zero or negative the
    /// strip will contain no frames.
    #[must_use]
    pub fn new(config: ThumbnailStripConfig, total_duration_secs: f32) -> Self {
        let positions = if total_duration_secs > 0.0 && config.interval_secs > 0.0 {
            let columns = config.columns.max(1);
            let frame_width = config.frame_width;
            let frame_height = config.frame_height;
            let interval = config.interval_secs;

            let count =
                (total_duration_secs / interval).ceil() as u32;

            (0..count)
                .map(|i| {
                    let time_secs = i as f32 * interval;
                    let row = i / columns;
                    let col = i % columns;
                    ThumbnailPosition {
                        time_secs,
                        row,
                        col,
                        x: col * frame_width,
                        y: row * frame_height,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        Self {
            config,
            total_duration_secs,
            positions,
        }
    }

    /// Returns the total number of thumbnail frames in this strip.
    #[must_use]
    pub fn frame_count(&self) -> u32 {
        self.positions.len() as u32
    }

    /// Returns the thumbnail position for the frame nearest to `time_secs`.
    ///
    /// If the strip contains no frames the returned position has all fields
    /// set to zero.
    #[must_use]
    pub fn position_at(&self, time_secs: f32) -> ThumbnailPosition {
        if self.positions.is_empty() {
            return ThumbnailPosition {
                time_secs: 0.0,
                row: 0,
                col: 0,
                x: 0,
                y: 0,
            };
        }

        // Find the position whose timestamp is nearest to the requested time.
        let best = self
            .positions
            .iter()
            .min_by(|a, b| {
                let da = (a.time_secs - time_secs).abs();
                let db = (b.time_secs - time_secs).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });

        match best {
            Some(p) => p.clone(),
            None => ThumbnailPosition {
                time_secs: 0.0,
                row: 0,
                col: 0,
                x: 0,
                y: 0,
            },
        }
    }

    /// Returns the total pixel dimensions `(width, height)` of the sprite sheet.
    ///
    /// Width = `columns * frame_width`.
    /// Height = `rows * frame_height` where `rows = ceil(frame_count / columns)`.
    #[must_use]
    pub fn sprite_sheet_dims(&self) -> (u32, u32) {
        if self.positions.is_empty() {
            return (0, 0);
        }
        let columns = self.config.columns.max(1);
        let frame_count = self.frame_count();
        let rows = (frame_count + columns - 1) / columns;
        let total_width = columns * self.config.frame_width;
        let total_height = rows * self.config.frame_height;
        (total_width, total_height)
    }

    /// Generates a WebVTT string suitable for use with `<track kind="metadata">`.
    ///
    /// Each cue covers the interval `[frame_time, next_frame_time)`.  The last
    /// cue ends at `total_duration_secs`.  `image_url` is the URL of the
    /// assembled sprite sheet image.
    #[must_use]
    pub fn to_vtt_string(&self, image_url: &str) -> String {
        let mut out = String::from("WEBVTT\n");

        for (idx, pos) in self.positions.iter().enumerate() {
            let start = pos.time_secs;
            let end = if idx + 1 < self.positions.len() {
                self.positions[idx + 1].time_secs
            } else {
                self.total_duration_secs
            };

            out.push('\n');
            out.push_str(&format_vtt_timestamp(start));
            out.push_str(" --> ");
            out.push_str(&format_vtt_timestamp(end));
            out.push('\n');
            out.push_str(&format!(
                "{}#xywh={},{},{},{}\n",
                image_url, pos.x, pos.y, self.config.frame_width, self.config.frame_height
            ));
        }

        out
    }
}

/// Formats seconds as a WebVTT timestamp `HH:MM:SS.mmm`.
#[must_use]
fn format_vtt_timestamp(secs: f32) -> String {
    let total_ms = (secs * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{:02}:{:02}:{:02}.{:03}", h, m, s, ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_strip() -> ThumbnailStrip {
        ThumbnailStrip::new(
            ThumbnailStripConfig {
                frame_width: 160,
                frame_height: 90,
                interval_secs: 5.0,
                columns: 5,
            },
            60.0,
        )
    }

    #[test]
    fn test_frame_count_60s_5s_interval() {
        let strip = default_strip();
        // 60 / 5 = 12 frames
        assert_eq!(strip.frame_count(), 12);
    }

    #[test]
    fn test_sprite_sheet_dims() {
        let strip = default_strip();
        // 5 columns × 160 = 800 wide; 12 frames / 5 cols = 3 rows × 90 = 270 high
        let (w, h) = strip.sprite_sheet_dims();
        assert_eq!(w, 800);
        assert_eq!(h, 270);
    }

    #[test]
    fn test_position_at_exact() {
        let strip = default_strip();
        let pos = strip.position_at(0.0);
        assert_eq!(pos.time_secs, 0.0);
        assert_eq!(pos.row, 0);
        assert_eq!(pos.col, 0);
        assert_eq!(pos.x, 0);
        assert_eq!(pos.y, 0);

        let pos5 = strip.position_at(5.0);
        assert_eq!(pos5.col, 1);
        assert_eq!(pos5.x, 160);
        assert_eq!(pos5.y, 0);
    }

    #[test]
    fn test_position_at_nearest() {
        let strip = default_strip();
        // Requesting 6 s should snap to the 5 s frame (nearest).
        let pos = strip.position_at(6.0);
        assert_eq!(pos.time_secs, 5.0);
    }

    #[test]
    fn test_vtt_string_header() {
        let strip = default_strip();
        let vtt = strip.to_vtt_string("thumbnails.jpg");
        assert!(vtt.starts_with("WEBVTT\n"));
    }

    #[test]
    fn test_vtt_first_cue() {
        let strip = default_strip();
        let vtt = strip.to_vtt_string("thumbnails.jpg");
        assert!(
            vtt.contains("00:00:00.000 --> 00:00:05.000"),
            "first cue not found:\n{vtt}"
        );
        assert!(vtt.contains("thumbnails.jpg#xywh=0,0,160,90"));
    }

    #[test]
    fn test_vtt_second_cue_offset() {
        let strip = default_strip();
        let vtt = strip.to_vtt_string("sprite.jpg");
        // Second frame (index 1, col 1) → x = 160
        assert!(
            vtt.contains("sprite.jpg#xywh=160,0,160,90"),
            "second cue offset not found:\n{vtt}"
        );
    }

    #[test]
    fn test_empty_strip_for_zero_duration() {
        let strip = ThumbnailStrip::new(ThumbnailStripConfig::default_web(), 0.0);
        assert_eq!(strip.frame_count(), 0);
        assert_eq!(strip.sprite_sheet_dims(), (0, 0));
        let vtt = strip.to_vtt_string("x.jpg");
        assert_eq!(vtt, "WEBVTT\n");
    }

    #[test]
    fn test_format_vtt_timestamp_seconds() {
        assert_eq!(format_vtt_timestamp(0.0), "00:00:00.000");
        assert_eq!(format_vtt_timestamp(5.0), "00:00:05.000");
        assert_eq!(format_vtt_timestamp(65.5), "00:01:05.500");
        assert_eq!(format_vtt_timestamp(3661.0), "01:01:01.000");
    }

    #[test]
    fn test_row_wrapping() {
        // 5 columns: frames 0-4 in row 0, frames 5-9 in row 1
        let strip = default_strip();
        let pos10 = strip.position_at(25.0); // frame index 5
        assert_eq!(pos10.row, 1);
        assert_eq!(pos10.col, 0);
        assert_eq!(pos10.y, 90);
    }
}
