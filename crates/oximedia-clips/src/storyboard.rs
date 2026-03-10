//! Storyboard generation from clips.
//!
//! Provides tools to generate visual storyboards from a collection of clips
//! by sampling frames at configurable intervals and producing panel-based layouts.

#![allow(dead_code)]

/// A single frame within a storyboard, with thumbnail metadata.
#[derive(Debug, Clone)]
pub struct StoryboardFrame {
    /// ID of the source clip.
    pub clip_id: String,
    /// Timestamp within the clip in milliseconds.
    pub timestamp_ms: u64,
    /// Thumbnail dimensions as (width, height).
    pub thumbnail_size: (u32, u32),
    /// Optional caption for this frame.
    pub caption: Option<String>,
}

impl StoryboardFrame {
    /// Create a new storyboard frame.
    #[must_use]
    pub fn new(clip_id: &str, timestamp_ms: u64, thumbnail_size: (u32, u32)) -> Self {
        Self {
            clip_id: clip_id.to_owned(),
            timestamp_ms,
            thumbnail_size,
            caption: None,
        }
    }

    /// Return the aspect ratio (width / height) of the thumbnail.
    ///
    /// Returns `0.0` if height is zero.
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        let (w, h) = self.thumbnail_size;
        if h == 0 {
            return 0.0;
        }
        f64::from(w) / f64::from(h)
    }
}

/// A panel in the storyboard, wrapping one frame with display metadata.
#[derive(Debug, Clone)]
pub struct StoryboardPanel {
    /// The frame represented by this panel.
    pub frame: StoryboardFrame,
    /// 1-based panel number within the storyboard.
    pub panel_number: u32,
    /// Duration this panel represents, in milliseconds.
    pub duration_ms: u64,
}

impl StoryboardPanel {
    /// Create a new storyboard panel.
    #[must_use]
    pub fn new(frame: StoryboardFrame, panel_number: u32, duration_ms: u64) -> Self {
        Self {
            frame,
            panel_number,
            duration_ms,
        }
    }

    /// Return `true` if this panel's duration exceeds the given threshold.
    #[must_use]
    pub fn is_long(&self, threshold_ms: u64) -> bool {
        self.duration_ms > threshold_ms
    }
}

/// Configuration for storyboard generation.
#[derive(Debug, Clone)]
pub struct StoryboardConfig {
    /// Number of frames to sample per clip.
    pub frames_per_clip: u32,
    /// Width of each panel thumbnail in pixels.
    pub panel_width_px: u32,
    /// Height of each panel thumbnail in pixels.
    pub panel_height_px: u32,
    /// Whether to include captions derived from clip names.
    pub include_captions: bool,
}

impl StoryboardConfig {
    /// Return a sensible default configuration.
    #[must_use]
    pub fn default_config() -> Self {
        Self {
            frames_per_clip: 6,
            panel_width_px: 320,
            panel_height_px: 180,
            include_captions: true,
        }
    }
}

impl Default for StoryboardConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Metadata about a clip used as input to the storyboard generator.
#[derive(Debug, Clone)]
pub struct ClipInfo {
    /// Unique clip identifier.
    pub id: String,
    /// Duration of the clip in milliseconds.
    pub duration_ms: u64,
    /// Human-readable name of the clip.
    pub name: String,
}

impl ClipInfo {
    /// Create a new `ClipInfo`.
    #[must_use]
    pub fn new(id: &str, duration_ms: u64, name: &str) -> Self {
        Self {
            id: id.to_owned(),
            duration_ms,
            name: name.to_owned(),
        }
    }

    /// Return evenly spaced timestamp positions (in ms) for sampling at `fps` frames-per-second.
    ///
    /// Timestamps are clamped to `[0, duration_ms]`.  At least one timestamp
    /// (at position `0`) is always returned even if `fps` is `0.0` or if the
    /// clip has zero duration.
    #[must_use]
    pub fn fps_timestamps(&self, fps: f32) -> Vec<u64> {
        if fps <= 0.0 || self.duration_ms == 0 {
            return vec![0];
        }
        let interval_ms = (1_000.0 / fps) as u64;
        let interval_ms = interval_ms.max(1);
        let mut ts = 0u64;
        let mut out = Vec::new();
        while ts <= self.duration_ms {
            out.push(ts);
            ts = ts.saturating_add(interval_ms);
        }
        out
    }
}

/// Generates a storyboard from a list of clips.
#[derive(Debug, Clone)]
pub struct StoryboardGenerator {
    /// Configuration driving generation behaviour.
    pub config: StoryboardConfig,
}

impl StoryboardGenerator {
    /// Create a new generator with the given configuration.
    #[must_use]
    pub fn new(config: StoryboardConfig) -> Self {
        Self { config }
    }

    /// Generate a flat list of [`StoryboardPanel`]s from the provided clips.
    ///
    /// Each clip contributes `config.frames_per_clip` evenly spaced panels.
    /// Panel numbers start at `1` and increase monotonically across all clips.
    #[must_use]
    pub fn generate(&self, clips: &[ClipInfo]) -> Vec<StoryboardPanel> {
        let mut panels = Vec::new();
        let mut panel_number: u32 = 1;
        let frames_per_clip = self.config.frames_per_clip.max(1);
        let size = (self.config.panel_width_px, self.config.panel_height_px);

        for clip in clips {
            let timestamps = self.sample_timestamps(clip, frames_per_clip);
            // Per-panel duration: spread the clip's total duration across the panels.
            let panel_duration = if frames_per_clip > 1 {
                clip.duration_ms / u64::from(frames_per_clip)
            } else {
                clip.duration_ms
            };

            for ts in timestamps {
                let mut frame = StoryboardFrame::new(&clip.id, ts, size);
                if self.config.include_captions {
                    frame.caption = Some(format!("{} @ {}ms", clip.name, ts));
                }
                panels.push(StoryboardPanel::new(frame, panel_number, panel_duration));
                panel_number += 1;
            }
        }

        panels
    }

    /// Sample `count` evenly spaced timestamps from a clip.
    fn sample_timestamps(&self, clip: &ClipInfo, count: u32) -> Vec<u64> {
        if clip.duration_ms == 0 || count == 0 {
            return vec![0];
        }
        if count == 1 {
            return vec![0];
        }
        (0..count)
            .map(|i| {
                let ratio = f64::from(i) / f64::from(count - 1);
                (ratio * clip.duration_ms as f64) as u64
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- StoryboardFrame tests ---

    #[test]
    fn test_frame_aspect_ratio_normal() {
        let f = StoryboardFrame::new("clip1", 0, (1920, 1080));
        let ratio = f.aspect_ratio();
        assert!((ratio - 16.0 / 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_aspect_ratio_zero_height() {
        let f = StoryboardFrame::new("clip1", 0, (1280, 0));
        assert_eq!(f.aspect_ratio(), 0.0);
    }

    #[test]
    fn test_frame_aspect_ratio_square() {
        let f = StoryboardFrame::new("clip1", 500, (100, 100));
        assert!((f.aspect_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_no_caption_by_default() {
        let f = StoryboardFrame::new("c", 100, (320, 180));
        assert!(f.caption.is_none());
    }

    // --- StoryboardPanel tests ---

    #[test]
    fn test_panel_is_long_true() {
        let frame = StoryboardFrame::new("c", 0, (320, 180));
        let panel = StoryboardPanel::new(frame, 1, 5_001);
        assert!(panel.is_long(5_000));
    }

    #[test]
    fn test_panel_is_long_false_equal() {
        let frame = StoryboardFrame::new("c", 0, (320, 180));
        let panel = StoryboardPanel::new(frame, 1, 5_000);
        // Exactly at threshold → not long (uses strict >)
        assert!(!panel.is_long(5_000));
    }

    #[test]
    fn test_panel_is_long_false_below() {
        let frame = StoryboardFrame::new("c", 0, (320, 180));
        let panel = StoryboardPanel::new(frame, 2, 999);
        assert!(!panel.is_long(1_000));
    }

    // --- StoryboardConfig tests ---

    #[test]
    fn test_config_default_values() {
        let cfg = StoryboardConfig::default_config();
        assert_eq!(cfg.frames_per_clip, 6);
        assert_eq!(cfg.panel_width_px, 320);
        assert_eq!(cfg.panel_height_px, 180);
        assert!(cfg.include_captions);
    }

    // --- ClipInfo tests ---

    #[test]
    fn test_fps_timestamps_basic() {
        let clip = ClipInfo::new("c1", 2_000, "Test");
        // 1 fps → timestamps at 0, 1000, 2000
        let ts = clip.fps_timestamps(1.0);
        assert_eq!(ts, vec![0, 1_000, 2_000]);
    }

    #[test]
    fn test_fps_timestamps_zero_fps() {
        let clip = ClipInfo::new("c1", 5_000, "Test");
        let ts = clip.fps_timestamps(0.0);
        assert_eq!(ts, vec![0]);
    }

    #[test]
    fn test_fps_timestamps_zero_duration() {
        let clip = ClipInfo::new("c1", 0, "Test");
        let ts = clip.fps_timestamps(30.0);
        assert_eq!(ts, vec![0]);
    }

    #[test]
    fn test_fps_timestamps_negative_fps() {
        let clip = ClipInfo::new("c1", 1_000, "Test");
        let ts = clip.fps_timestamps(-1.0);
        assert_eq!(ts, vec![0]);
    }

    // --- StoryboardGenerator tests ---

    #[test]
    fn test_generate_empty_clips() {
        let gen = StoryboardGenerator::new(StoryboardConfig::default_config());
        let panels = gen.generate(&[]);
        assert!(panels.is_empty());
    }

    #[test]
    fn test_generate_panel_count() {
        let mut cfg = StoryboardConfig::default_config();
        cfg.frames_per_clip = 3;
        let gen = StoryboardGenerator::new(cfg);
        let clips = vec![
            ClipInfo::new("c1", 10_000, "Clip One"),
            ClipInfo::new("c2", 5_000, "Clip Two"),
        ];
        let panels = gen.generate(&clips);
        // 2 clips × 3 panels each = 6 panels
        assert_eq!(panels.len(), 6);
    }

    #[test]
    fn test_generate_panel_numbers_sequential() {
        let mut cfg = StoryboardConfig::default_config();
        cfg.frames_per_clip = 2;
        let gen = StoryboardGenerator::new(cfg);
        let clips = vec![
            ClipInfo::new("c1", 4_000, "A"),
            ClipInfo::new("c2", 4_000, "B"),
        ];
        let panels = gen.generate(&clips);
        for (i, p) in panels.iter().enumerate() {
            assert_eq!(p.panel_number, (i + 1) as u32);
        }
    }

    #[test]
    fn test_generate_captions_present() {
        let cfg = StoryboardConfig::default_config();
        let gen = StoryboardGenerator::new(cfg);
        let clips = vec![ClipInfo::new("c1", 2_000, "Interview")];
        let panels = gen.generate(&clips);
        for p in &panels {
            assert!(p.frame.caption.is_some());
            assert!(p
                .frame
                .caption
                .as_ref()
                .expect("as_ref should succeed")
                .contains("Interview"));
        }
    }

    #[test]
    fn test_generate_no_captions_when_disabled() {
        let mut cfg = StoryboardConfig::default_config();
        cfg.include_captions = false;
        let gen = StoryboardGenerator::new(cfg);
        let clips = vec![ClipInfo::new("c1", 2_000, "Interview")];
        let panels = gen.generate(&clips);
        for p in &panels {
            assert!(p.frame.caption.is_none());
        }
    }

    #[test]
    fn test_generate_clip_id_matches() {
        let mut cfg = StoryboardConfig::default_config();
        cfg.frames_per_clip = 2;
        let gen = StoryboardGenerator::new(cfg);
        let clips = vec![ClipInfo::new("myClipId", 3_000, "X")];
        let panels = gen.generate(&clips);
        for p in &panels {
            assert_eq!(p.frame.clip_id, "myClipId");
        }
    }

    #[test]
    fn test_generate_thumbnail_size_matches_config() {
        let mut cfg = StoryboardConfig::default_config();
        cfg.panel_width_px = 640;
        cfg.panel_height_px = 360;
        cfg.frames_per_clip = 1;
        let gen = StoryboardGenerator::new(cfg);
        let clips = vec![ClipInfo::new("c", 1_000, "Y")];
        let panels = gen.generate(&clips);
        assert_eq!(panels[0].frame.thumbnail_size, (640, 360));
    }
}
