//! Segment writer for HLS/DASH.
//!
//! Provides efficient segment generation for adaptive streaming protocols.

#![forbid(unsafe_code)]

use oximedia_core::{OxiError, OxiResult};
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

use super::mp4::Mp4Fragment;

/// Configuration for segment writer.
#[derive(Clone, Debug)]
pub struct SegmentWriterConfig {
    /// Output directory for segments.
    pub output_dir: PathBuf,
    /// Segment filename pattern (e.g., "segment_%05d.m4s").
    pub filename_pattern: String,
    /// Whether to delete old segments.
    pub delete_old_segments: bool,
    /// Maximum number of segments to keep (if `delete_old_segments` is true).
    pub max_segments: Option<usize>,
    /// Whether to generate a playlist file.
    pub generate_playlist: bool,
}

impl SegmentWriterConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new<P: AsRef<Path>>(output_dir: P) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
            filename_pattern: "segment_%05d.m4s".into(),
            delete_old_segments: false,
            max_segments: None,
            generate_playlist: false,
        }
    }

    /// Sets the filename pattern.
    #[must_use]
    pub fn with_filename_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.filename_pattern = pattern.into();
        self
    }

    /// Enables deletion of old segments.
    #[must_use]
    pub const fn with_delete_old_segments(mut self, enabled: bool) -> Self {
        self.delete_old_segments = enabled;
        self
    }

    /// Sets the maximum number of segments to keep.
    #[must_use]
    pub const fn with_max_segments(mut self, max: usize) -> Self {
        self.max_segments = Some(max);
        self
    }

    /// Enables playlist generation.
    #[must_use]
    pub const fn with_playlist_generation(mut self, enabled: bool) -> Self {
        self.generate_playlist = enabled;
        self
    }
}

/// Information about a written segment.
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    /// Segment sequence number.
    pub sequence: u32,
    /// File path to the segment.
    pub path: PathBuf,
    /// Size of the segment in bytes.
    pub size: u64,
    /// Duration of the segment in seconds.
    pub duration_secs: f64,
    /// Whether the segment contains a keyframe.
    pub has_keyframe: bool,
}

impl SegmentInfo {
    /// Creates a new segment info.
    #[must_use]
    pub fn new(sequence: u32, path: PathBuf, size: u64, duration_secs: f64) -> Self {
        Self {
            sequence,
            path,
            size,
            duration_secs,
            has_keyframe: false,
        }
    }

    /// Returns the filename of the segment.
    #[must_use]
    pub fn filename(&self) -> Option<&str> {
        self.path.file_name().and_then(|n| n.to_str())
    }
}

/// Writer for HLS/DASH segments.
pub struct SegmentWriter {
    config: SegmentWriterConfig,
    segments: Vec<SegmentInfo>,
    init_segment_path: Option<PathBuf>,
}

impl SegmentWriter {
    /// Creates a new segment writer.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the output directory cannot be created.
    pub async fn new(config: SegmentWriterConfig) -> OxiResult<Self> {
        // Create output directory if it doesn't exist
        fs::create_dir_all(&config.output_dir)
            .await
            .map_err(|e: std::io::Error| OxiError::from(e))?;

        Ok(Self {
            config,
            segments: Vec::new(),
            init_segment_path: None,
        })
    }

    /// Writes the initialization segment.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the fragment is not an init segment or if writing fails.
    pub async fn write_init_segment(&mut self, fragment: &Mp4Fragment) -> OxiResult<PathBuf> {
        if !fragment.is_init() {
            return Err(OxiError::InvalidData(
                "Fragment is not an init segment".into(),
            ));
        }

        let path = self.config.output_dir.join("init.mp4");
        let mut file = fs::File::create(&path)
            .await
            .map_err(|e: std::io::Error| OxiError::from(e))?;

        file.write_all(&fragment.data)
            .await
            .map_err(|e: std::io::Error| OxiError::from(e))?;

        file.flush()
            .await
            .map_err(|e: std::io::Error| OxiError::from(e))?;

        self.init_segment_path = Some(path.clone());
        Ok(path)
    }

    /// Writes a media segment.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the fragment is not a media segment or if writing fails.
    pub async fn write_segment(&mut self, fragment: &Mp4Fragment) -> OxiResult<SegmentInfo> {
        if !fragment.is_media() {
            return Err(OxiError::InvalidData(
                "Fragment is not a media segment".into(),
            ));
        }

        // Generate filename
        let filename = self
            .config
            .filename_pattern
            .replace("%05d", &format!("{:05}", fragment.sequence))
            .replace("%d", &fragment.sequence.to_string());

        let path = self.config.output_dir.join(filename);

        // Write segment
        let mut file = fs::File::create(&path)
            .await
            .map_err(|e: std::io::Error| OxiError::from(e))?;

        file.write_all(&fragment.data)
            .await
            .map_err(|e: std::io::Error| OxiError::from(e))?;

        file.flush()
            .await
            .map_err(|e: std::io::Error| OxiError::from(e))?;

        // Create segment info
        #[allow(clippy::cast_precision_loss)]
        let duration_secs = fragment.duration_us as f64 / 1_000_000.0;
        let mut info = SegmentInfo::new(
            fragment.sequence,
            path,
            fragment.data.len() as u64,
            duration_secs,
        );
        info.has_keyframe = fragment.has_keyframe;

        self.segments.push(info.clone());

        // Delete old segments if configured
        if self.config.delete_old_segments {
            self.cleanup_old_segments().await?;
        }

        // Generate playlist if configured
        if self.config.generate_playlist {
            self.generate_playlist().await?;
        }

        Ok(info)
    }

    /// Returns information about all written segments.
    #[must_use]
    pub fn segments(&self) -> &[SegmentInfo] {
        &self.segments
    }

    /// Returns the path to the initialization segment.
    #[must_use]
    pub fn init_segment_path(&self) -> Option<&Path> {
        self.init_segment_path.as_deref()
    }

    /// Cleans up old segments beyond the maximum count.
    async fn cleanup_old_segments(&mut self) -> OxiResult<()> {
        if let Some(max_segments) = self.config.max_segments {
            while self.segments.len() > max_segments {
                if let Some(segment) = self.segments.first() {
                    let path = segment.path.clone();
                    if let Err(e) = fs::remove_file(&path).await {
                        // Log error but don't fail
                        eprintln!("Failed to delete segment {}: {e}", path.display());
                    }
                }
                self.segments.remove(0);
            }
        }
        Ok(())
    }

    /// Generates an HLS playlist file.
    async fn generate_playlist(&self) -> OxiResult<()> {
        let playlist_path = self.config.output_dir.join("playlist.m3u8");

        let mut content = String::new();
        content.push_str("#EXTM3U\n");
        content.push_str("#EXT-X-VERSION:6\n");
        content.push_str("#EXT-X-TARGETDURATION:10\n");
        content.push_str("#EXT-X-MEDIA-SEQUENCE:1\n");

        // Add init segment
        if let Some(init_path) = &self.init_segment_path {
            if let Some(filename) = init_path.file_name().and_then(|n| n.to_str()) {
                let _ = writeln!(content, "#EXT-X-MAP:URI=\"{filename}\"");
            }
        }

        // Add media segments
        for segment in &self.segments {
            let _ = writeln!(content, "#EXTINF:{:.6},", segment.duration_secs);
            if let Some(filename) = segment.filename() {
                content.push_str(filename);
                content.push('\n');
            }
        }

        fs::write(&playlist_path, content)
            .await
            .map_err(|e: std::io::Error| OxiError::from(e))?;

        Ok(())
    }
}

/// Helper for generating DASH manifest (MPD).
pub struct DashManifestGenerator {
    presentation_duration_secs: f64,
    min_buffer_time_secs: f64,
}

impl DashManifestGenerator {
    /// Creates a new DASH manifest generator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            presentation_duration_secs: 0.0,
            min_buffer_time_secs: 4.0,
        }
    }

    /// Sets the presentation duration.
    #[must_use]
    pub const fn with_duration(mut self, duration_secs: f64) -> Self {
        self.presentation_duration_secs = duration_secs;
        self
    }

    /// Sets the minimum buffer time.
    #[must_use]
    pub const fn with_min_buffer_time(mut self, time_secs: f64) -> Self {
        self.min_buffer_time_secs = time_secs;
        self
    }

    /// Generates a DASH manifest for the given segments.
    #[must_use]
    pub fn generate(&self, segments: &[SegmentInfo], init_segment: &str) -> String {
        let mut mpd = String::new();
        mpd.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        mpd.push_str("<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" ");
        mpd.push_str("type=\"static\" ");
        let _ = write!(
            mpd,
            "mediaPresentationDuration=\"PT{:.3}S\" ",
            self.presentation_duration_secs
        );
        let _ = writeln!(
            mpd,
            "minBufferTime=\"PT{:.3}S\">",
            self.min_buffer_time_secs
        );

        mpd.push_str("  <Period>\n");
        mpd.push_str("    <AdaptationSet>\n");
        mpd.push_str("      <Representation>\n");
        let _ = writeln!(mpd, "        <BaseURL>{init_segment}</BaseURL>");
        mpd.push_str("        <SegmentList>\n");

        for segment in segments {
            if let Some(filename) = segment.filename() {
                let _ = writeln!(mpd, "          <SegmentURL media=\"{filename}\" />");
            }
        }

        mpd.push_str("        </SegmentList>\n");
        mpd.push_str("      </Representation>\n");
        mpd.push_str("    </AdaptationSet>\n");
        mpd.push_str("  </Period>\n");
        mpd.push_str("</MPD>\n");

        mpd
    }
}

impl Default for DashManifestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("oximedia-container-fragment-segment-{name}"))
    }

    #[test]
    fn test_segment_writer_config() {
        let config = SegmentWriterConfig::new(tmp_path("segments"))
            .with_filename_pattern("seg_%d.m4s")
            .with_delete_old_segments(true)
            .with_max_segments(10)
            .with_playlist_generation(true);

        assert_eq!(config.filename_pattern, "seg_%d.m4s");
        assert!(config.delete_old_segments);
        assert_eq!(config.max_segments, Some(10));
        assert!(config.generate_playlist);
    }

    #[test]
    fn test_segment_info() {
        let info = SegmentInfo::new(1, tmp_path("seg1.m4s"), 1024, 2.0);
        assert_eq!(info.sequence, 1);
        assert_eq!(info.size, 1024);
        assert_eq!(info.duration_secs, 2.0);
        assert_eq!(
            info.filename(),
            Some("oximedia-container-fragment-segment-seg1.m4s")
        );
    }

    #[test]
    fn test_dash_manifest_generator() {
        let generator = DashManifestGenerator::new()
            .with_duration(10.0)
            .with_min_buffer_time(2.0);

        let segments = vec![
            SegmentInfo::new(1, PathBuf::from("seg1.m4s"), 1024, 2.0),
            SegmentInfo::new(2, PathBuf::from("seg2.m4s"), 1024, 2.0),
        ];

        let manifest = generator.generate(&segments, "init.mp4");
        assert!(manifest.contains("<?xml version=\"1.0\""));
        assert!(manifest.contains("MPD"));
        assert!(manifest.contains("seg1.m4s"));
        assert!(manifest.contains("seg2.m4s"));
    }
}
