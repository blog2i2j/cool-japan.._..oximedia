//! Timeline export settings and codec presets.

/// Codec preset for timeline export.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct ExportCodecPreset {
    /// Video codec identifier (e.g. "h264", "hevc", "prores").
    pub video_codec: String,
    /// Audio codec identifier (e.g. "aac", "pcm").
    pub audio_codec: String,
    /// Container format (e.g. "mp4", "mov", "mxf").
    pub container: String,
    /// Target video bitrate in kbps.
    pub video_bitrate_kbps: u32,
    /// Target audio bitrate in kbps.
    pub audio_bitrate_kbps: u32,
}

impl ExportCodecPreset {
    /// H.264 HD preset suitable for web delivery.
    #[must_use]
    pub fn h264_hd() -> Self {
        Self {
            video_codec: "h264".to_string(),
            audio_codec: "aac".to_string(),
            container: "mp4".to_string(),
            video_bitrate_kbps: 8_000,
            audio_bitrate_kbps: 192,
        }
    }

    /// HEVC 4K preset for high-quality delivery.
    #[must_use]
    pub fn hevc_4k() -> Self {
        Self {
            video_codec: "hevc".to_string(),
            audio_codec: "aac".to_string(),
            container: "mp4".to_string(),
            video_bitrate_kbps: 40_000,
            audio_bitrate_kbps: 256,
        }
    }

    /// Apple `ProRes` 4444 preset for post-production interchange.
    #[must_use]
    pub fn prores_4444() -> Self {
        Self {
            video_codec: "prores_4444".to_string(),
            audio_codec: "pcm".to_string(),
            container: "mov".to_string(),
            video_bitrate_kbps: 500_000,
            audio_bitrate_kbps: 2_304,
        }
    }

    /// Estimate the output file size in bytes for the given duration.
    ///
    /// Uses combined audio+video bitrate. Bitrate is in kbps (kilobits per
    /// second), so: `bytes = (kbps * 1000 / 8) * duration_seconds`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimated_size_bytes(&self, duration_ms: u64) -> u64 {
        let total_kbps = self.video_bitrate_kbps + self.audio_bitrate_kbps;
        // bytes per ms = kbps * 1000 / 8 / 1000 = kbps / 8
        u64::from(total_kbps) * duration_ms / 8
    }
}

/// The portion of the timeline to export.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum ExportRange {
    /// Export all content from the first frame to the last.
    EntireTimeline,
    /// Export a custom work-area defined by (`start_frame`, `end_frame`).
    WorkArea(u64, u64),
    /// Export from the In point to the Out point (`start_frame`, `end_frame`).
    InToOut(u64, u64),
}

impl ExportRange {
    /// Return the first frame of the export range.
    #[must_use]
    pub fn start_frame(&self) -> u64 {
        match self {
            Self::EntireTimeline => 0,
            Self::WorkArea(s, _) | Self::InToOut(s, _) => *s,
        }
    }

    /// Return the last frame of the export range.
    ///
    /// For [`ExportRange::EntireTimeline`] the caller must supply the total
    /// frame count of the timeline.
    #[must_use]
    pub fn end_frame(&self, total: u64) -> u64 {
        match self {
            Self::EntireTimeline => total,
            Self::WorkArea(_, e) | Self::InToOut(_, e) => *e,
        }
    }

    /// Number of frames covered by this range.
    #[must_use]
    pub fn frame_count(&self, total: u64) -> u64 {
        self.end_frame(total).saturating_sub(self.start_frame())
    }
}

/// Complete set of parameters controlling a timeline export operation.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ExportSettings {
    /// Codec and container configuration.
    pub preset: ExportCodecPreset,
    /// Which portion of the timeline to export.
    pub range: ExportRange,
    /// Destination file path.
    pub output_path: String,
    /// Burn subtitle track into the video picture.
    pub include_subtitles: bool,
    /// Burn timecode into the video picture.
    pub burn_in_tc: bool,
}

impl ExportSettings {
    /// Create new export settings.
    #[must_use]
    pub fn new(
        preset: ExportCodecPreset,
        range: ExportRange,
        output_path: impl Into<String>,
    ) -> Self {
        Self {
            preset,
            range,
            output_path: output_path.into(),
            include_subtitles: false,
            burn_in_tc: false,
        }
    }

    /// Returns `true` when the settings are considered valid for export.
    ///
    /// Validity checks:
    /// - Output path must be non-empty.
    /// - For ranged exports the end frame must be greater than the start frame.
    /// - Both video and audio bitrates must be non-zero.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if self.output_path.is_empty() {
            return false;
        }
        match &self.range {
            ExportRange::WorkArea(s, e) | ExportRange::InToOut(s, e) => {
                if e <= s {
                    return false;
                }
            }
            ExportRange::EntireTimeline => {}
        }
        self.preset.video_bitrate_kbps > 0 && self.preset.audio_bitrate_kbps > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-timeline-export-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_h264_hd_preset() {
        let p = ExportCodecPreset::h264_hd();
        assert_eq!(p.video_codec, "h264");
        assert_eq!(p.container, "mp4");
        assert_eq!(p.video_bitrate_kbps, 8_000);
    }

    #[test]
    fn test_hevc_4k_preset() {
        let p = ExportCodecPreset::hevc_4k();
        assert_eq!(p.video_codec, "hevc");
        assert_eq!(p.video_bitrate_kbps, 40_000);
    }

    #[test]
    fn test_prores_4444_preset() {
        let p = ExportCodecPreset::prores_4444();
        assert_eq!(p.video_codec, "prores_4444");
        assert_eq!(p.container, "mov");
        assert_eq!(p.audio_codec, "pcm");
    }

    #[test]
    fn test_estimated_size_bytes_h264() {
        let p = ExportCodecPreset::h264_hd();
        // 1 second @ (8000+192) kbps = 8192 * 1000 / 8 bytes = 1_024_000 bytes
        let size = p.estimated_size_bytes(1_000);
        assert_eq!(size, (8_000u64 + 192) * 1_000 / 8);
    }

    #[test]
    fn test_estimated_size_bytes_zero_duration() {
        let p = ExportCodecPreset::h264_hd();
        assert_eq!(p.estimated_size_bytes(0), 0);
    }

    #[test]
    fn test_export_range_entire_timeline_start() {
        assert_eq!(ExportRange::EntireTimeline.start_frame(), 0);
    }

    #[test]
    fn test_export_range_entire_timeline_end() {
        assert_eq!(ExportRange::EntireTimeline.end_frame(500), 500);
    }

    #[test]
    fn test_export_range_work_area() {
        let r = ExportRange::WorkArea(10, 90);
        assert_eq!(r.start_frame(), 10);
        assert_eq!(r.end_frame(500), 90);
        assert_eq!(r.frame_count(500), 80);
    }

    #[test]
    fn test_export_range_in_to_out() {
        let r = ExportRange::InToOut(24, 120);
        assert_eq!(r.start_frame(), 24);
        assert_eq!(r.end_frame(9999), 120);
        assert_eq!(r.frame_count(9999), 96);
    }

    #[test]
    fn test_export_settings_valid() {
        let s = ExportSettings::new(
            ExportCodecPreset::h264_hd(),
            ExportRange::EntireTimeline,
            tmp_str("out.mp4"),
        );
        assert!(s.is_valid());
    }

    #[test]
    fn test_export_settings_empty_path_invalid() {
        let s = ExportSettings::new(
            ExportCodecPreset::h264_hd(),
            ExportRange::EntireTimeline,
            "",
        );
        assert!(!s.is_valid());
    }

    #[test]
    fn test_export_settings_inverted_range_invalid() {
        let s = ExportSettings::new(
            ExportCodecPreset::h264_hd(),
            ExportRange::WorkArea(100, 50),
            tmp_str("out.mp4"),
        );
        assert!(!s.is_valid());
    }

    #[test]
    fn test_export_settings_equal_range_invalid() {
        let s = ExportSettings::new(
            ExportCodecPreset::h264_hd(),
            ExportRange::InToOut(50, 50),
            tmp_str("out.mp4"),
        );
        assert!(!s.is_valid());
    }

    #[test]
    fn test_export_settings_in_to_out_valid() {
        let s = ExportSettings::new(
            ExportCodecPreset::hevc_4k(),
            ExportRange::InToOut(0, 240),
            tmp_str("out_4k.mp4"),
        );
        assert!(s.is_valid());
    }

    #[test]
    fn test_export_settings_flags() {
        let mut s = ExportSettings::new(
            ExportCodecPreset::prores_4444(),
            ExportRange::EntireTimeline,
            tmp_str("out.mov"),
        );
        s.include_subtitles = true;
        s.burn_in_tc = true;
        assert!(s.include_subtitles);
        assert!(s.burn_in_tc);
        assert!(s.is_valid());
    }
}
