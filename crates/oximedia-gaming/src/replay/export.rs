//! Replay export functionality.
//!
//! Provides segment definitions, export configuration, and a stub exporter that
//! generates synthetic frame data for testing without FFmpeg/system dependencies.

/// A labelled time range within a replay.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ReplaySegment {
    /// Segment start in milliseconds
    pub start_ms: u64,
    /// Segment end in milliseconds
    pub end_ms: u64,
    /// Human-readable label (e.g. "Epic triple kill")
    pub label: String,
    /// Arbitrary tags for searching/filtering
    pub tags: Vec<String>,
}

/// Output container format for exported replays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// MPEG-4 container
    Mp4,
    /// `WebM` container
    Webm,
    /// Animated GIF
    Gif,
    /// Matroska container
    Mkv,
}

/// Output quality preset for exported replays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportQuality {
    /// Low quality (fast encode, small file)
    Low,
    /// Medium quality (balanced)
    Medium,
    /// High quality (slow encode, large file)
    High,
    /// Lossless (largest file, best quality)
    Lossless,
}

/// Configuration for a replay export operation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ReplayExportConfig {
    /// Output container format
    pub format: ExportFormat,
    /// Encoding quality
    pub quality: ExportQuality,
    /// Whether to include the audio track
    pub include_audio: bool,
}

/// Metadata manifest describing a collection of exported clips.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct ReplayClipManifest {
    /// The individual clips in the manifest
    pub clips: Vec<ReplaySegment>,
    /// Total duration across all clips in milliseconds
    pub total_duration_ms: u64,
    /// Estimated total file size in bytes
    pub file_size_bytes: u64,
}

/// Exports replay segments to various formats.
///
/// In this stub implementation the actual encoding pipeline is not connected;
/// instead a synthetic byte payload is produced so that all public API paths can
/// be exercised in unit tests without native library dependencies.
pub struct ReplayExporter {
    /// Base bytes-per-millisecond used for size estimation
    base_bpm: u64,
}

// ── ExportQuality ────────────────────────────────────────────────────────────

impl ExportQuality {
    /// Return the Constant Rate Factor (CRF) value used by libx264/libvpx-vp9.
    ///
    /// Lower CRF → better quality, larger file.
    #[must_use]
    pub fn to_crf(&self) -> u8 {
        match self {
            Self::Low => 35,
            Self::Medium => 23,
            Self::High => 16,
            Self::Lossless => 0,
        }
    }

    /// Quality multiplier relative to `Medium` (used for size estimation).
    #[must_use]
    fn size_multiplier(&self) -> u64 {
        match self {
            Self::Low => 1,
            Self::Medium => 2,
            Self::High => 4,
            Self::Lossless => 10,
        }
    }
}

// ── ExportFormat ─────────────────────────────────────────────────────────────

impl ExportFormat {
    /// Return a file extension string for this format.
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Webm => "webm",
            Self::Gif => "gif",
            Self::Mkv => "mkv",
        }
    }
}

// ── ReplaySegment ─────────────────────────────────────────────────────────────

impl ReplaySegment {
    /// Create a new segment.
    #[must_use]
    pub fn new(start_ms: u64, end_ms: u64, label: impl Into<String>, tags: Vec<String>) -> Self {
        Self {
            start_ms,
            end_ms,
            label: label.into(),
            tags,
        }
    }

    /// Duration of the segment in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Return `true` if the segment has a non-zero positive duration.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.end_ms > self.start_ms
    }
}

// ── ReplayExportConfig ────────────────────────────────────────────────────────

impl Default for ReplayExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Mp4,
            quality: ExportQuality::Medium,
            include_audio: true,
        }
    }
}

// ── ReplayExporter ────────────────────────────────────────────────────────────

impl ReplayExporter {
    /// Create a new exporter.
    ///
    /// `base_bpm` is the baseline number of bytes generated per millisecond of
    /// content at `ExportQuality::Medium`.
    #[must_use]
    pub fn new(base_bpm: u64) -> Self {
        Self { base_bpm }
    }

    /// Export a single segment using the given configuration.
    ///
    /// Returns a synthetic `Vec<u8>` whose length is proportional to the
    /// segment duration, quality multiplier, and whether audio is included.
    ///
    /// # Errors
    ///
    /// Returns an error string if the segment is invalid (end ≤ start).
    pub fn export_segment(
        &self,
        segment: &ReplaySegment,
        config: &ReplayExportConfig,
    ) -> Result<Vec<u8>, String> {
        if !segment.is_valid() {
            return Err(format!(
                "Invalid segment '{}': end_ms ({}) must be > start_ms ({})",
                segment.label, segment.end_ms, segment.start_ms
            ));
        }

        let duration_ms = segment.duration_ms();
        let quality_mult = config.quality.size_multiplier();
        let audio_mult: u64 = u64::from(config.include_audio);

        // Synthetic payload size
        let size = (self.base_bpm * duration_ms * quality_mult) / 1000
            + (audio_mult * duration_ms * 16) / 1000;
        let size = size.max(1) as usize;

        // Fill with a repeating pattern derived from format/quality so that the
        // output is deterministic but not just zeros.
        let seed = config.quality.to_crf();
        let payload: Vec<u8> = (0..size).map(|i| seed.wrapping_add(i as u8)).collect();
        Ok(payload)
    }

    /// Estimate the file size of a segment without allocating the full buffer.
    #[must_use]
    pub fn estimate_size(&self, segment: &ReplaySegment, config: &ReplayExportConfig) -> u64 {
        if !segment.is_valid() {
            return 0;
        }
        let duration_ms = segment.duration_ms();
        let quality_mult = config.quality.size_multiplier();
        let audio_mult: u64 = u64::from(config.include_audio);
        (self.base_bpm * duration_ms * quality_mult) / 1000 + (audio_mult * duration_ms * 16) / 1000
    }
}

impl Default for ReplayExporter {
    fn default() -> Self {
        // ~125 bytes/ms at Medium quality ≈ 1 Mbit/s video
        Self::new(125)
    }
}

// ── ReplayClipManifest ────────────────────────────────────────────────────────

impl ReplayClipManifest {
    /// Build a manifest from a set of segments and a known total file size.
    #[must_use]
    pub fn new(clips: Vec<ReplaySegment>, file_size_bytes: u64) -> Self {
        let total_duration_ms = clips.iter().map(ReplaySegment::duration_ms).sum();
        Self {
            clips,
            total_duration_ms,
            file_size_bytes,
        }
    }

    /// Number of clips in the manifest.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: u64, end: u64) -> ReplaySegment {
        ReplaySegment::new(start, end, "test", vec!["gaming".to_string()])
    }

    // --- ReplaySegment ---

    #[test]
    fn test_segment_duration() {
        let s = seg(1000, 6000);
        assert_eq!(s.duration_ms(), 5000);
    }

    #[test]
    fn test_segment_valid() {
        assert!(seg(0, 1000).is_valid());
        // end <= start is invalid
        let bad = ReplaySegment::new(5000, 5000, "empty", vec![]);
        assert!(!bad.is_valid());
    }

    #[test]
    fn test_segment_invalid_reversed() {
        let bad = ReplaySegment::new(10000, 5000, "reversed", vec![]);
        assert!(!bad.is_valid());
        assert_eq!(bad.duration_ms(), 0); // saturating_sub
    }

    // --- ExportQuality ---

    #[test]
    fn test_export_quality_crf_values() {
        assert_eq!(ExportQuality::Low.to_crf(), 35);
        assert_eq!(ExportQuality::Medium.to_crf(), 23);
        assert_eq!(ExportQuality::High.to_crf(), 16);
        assert_eq!(ExportQuality::Lossless.to_crf(), 0);
    }

    #[test]
    fn test_export_quality_crf_ordered() {
        // Lower CRF = better quality
        assert!(ExportQuality::Lossless.to_crf() < ExportQuality::High.to_crf());
        assert!(ExportQuality::High.to_crf() < ExportQuality::Medium.to_crf());
        assert!(ExportQuality::Medium.to_crf() < ExportQuality::Low.to_crf());
    }

    // --- ExportFormat ---

    #[test]
    fn test_export_format_extensions() {
        assert_eq!(ExportFormat::Mp4.extension(), "mp4");
        assert_eq!(ExportFormat::Webm.extension(), "webm");
        assert_eq!(ExportFormat::Gif.extension(), "gif");
        assert_eq!(ExportFormat::Mkv.extension(), "mkv");
    }

    // --- ReplayExporter ---

    #[test]
    fn test_exporter_returns_data() {
        let exporter = ReplayExporter::default();
        let segment = seg(0, 10_000); // 10 seconds
        let config = ReplayExportConfig::default();
        let data = exporter
            .export_segment(&segment, &config)
            .expect("export should succeed");
        assert!(!data.is_empty());
    }

    #[test]
    fn test_exporter_invalid_segment() {
        let exporter = ReplayExporter::default();
        let bad = ReplaySegment::new(5000, 1000, "bad", vec![]);
        let config = ReplayExportConfig::default();
        let result = exporter.export_segment(&bad, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_exporter_high_quality_larger() {
        let exporter = ReplayExporter::default();
        let segment = seg(0, 10_000);
        let low_config = ReplayExportConfig {
            format: ExportFormat::Mp4,
            quality: ExportQuality::Low,
            include_audio: false,
        };
        let high_config = ReplayExportConfig {
            format: ExportFormat::Mp4,
            quality: ExportQuality::High,
            include_audio: false,
        };
        let low_data = exporter
            .export_segment(&segment, &low_config)
            .expect("export should succeed");
        let high_data = exporter
            .export_segment(&segment, &high_config)
            .expect("export should succeed");
        assert!(high_data.len() > low_data.len());
    }

    #[test]
    fn test_exporter_audio_adds_size() {
        let exporter = ReplayExporter::default();
        let segment = seg(0, 10_000);
        let no_audio = ReplayExportConfig {
            format: ExportFormat::Mp4,
            quality: ExportQuality::Medium,
            include_audio: false,
        };
        let with_audio = ReplayExportConfig {
            format: ExportFormat::Mp4,
            quality: ExportQuality::Medium,
            include_audio: true,
        };
        let no_audio_size = exporter
            .export_segment(&segment, &no_audio)
            .expect("export should succeed")
            .len();
        let with_audio_size = exporter
            .export_segment(&segment, &with_audio)
            .expect("should succeed")
            .len();
        assert!(with_audio_size >= no_audio_size);
    }

    #[test]
    fn test_estimate_size_consistent() {
        let exporter = ReplayExporter::default();
        let segment = seg(0, 10_000);
        let config = ReplayExportConfig::default();
        let estimated = exporter.estimate_size(&segment, &config);
        let actual = exporter
            .export_segment(&segment, &config)
            .expect("export should succeed")
            .len() as u64;
        // The estimate should match the actual payload size
        assert_eq!(estimated, actual);
    }

    // --- ReplayClipManifest ---

    #[test]
    fn test_manifest_total_duration() {
        let clips = vec![seg(0, 5000), seg(10_000, 15_000), seg(20_000, 22_000)];
        let manifest = ReplayClipManifest::new(clips, 1_000_000);
        assert_eq!(manifest.total_duration_ms, 12_000);
        assert_eq!(manifest.clip_count(), 3);
    }

    #[test]
    fn test_manifest_empty() {
        let manifest = ReplayClipManifest::default();
        assert_eq!(manifest.clip_count(), 0);
        assert_eq!(manifest.total_duration_ms, 0);
    }
}
