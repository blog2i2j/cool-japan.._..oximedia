// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Streaming output support (HLS, DASH).

use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Streaming output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamingFormat {
    /// HTTP Live Streaming (HLS)
    Hls,
    /// Dynamic Adaptive Streaming over HTTP (DASH)
    Dash,
}

/// Streaming configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Streaming format
    pub format: StreamingFormat,
    /// Segment duration in seconds
    pub segment_duration: f64,
    /// Number of segments in playlist
    pub playlist_size: u32,
    /// Enable fast start (moov atom at beginning)
    pub fast_start: bool,
    /// Output directory
    pub output_dir: PathBuf,
    /// Playlist filename
    pub playlist_name: String,
}

impl StreamingConfig {
    /// Create HLS configuration.
    #[must_use]
    pub fn hls(output_dir: PathBuf) -> Self {
        Self {
            format: StreamingFormat::Hls,
            segment_duration: 6.0,
            playlist_size: 5,
            fast_start: true,
            output_dir,
            playlist_name: "playlist.m3u8".to_string(),
        }
    }

    /// Create DASH configuration.
    #[must_use]
    pub fn dash(output_dir: PathBuf) -> Self {
        Self {
            format: StreamingFormat::Dash,
            segment_duration: 4.0,
            playlist_size: 5,
            fast_start: true,
            output_dir,
            playlist_name: "manifest.mpd".to_string(),
        }
    }

    /// Set segment duration.
    #[must_use]
    pub fn with_segment_duration(mut self, duration: f64) -> Self {
        self.segment_duration = duration;
        self
    }

    /// Set playlist size.
    #[must_use]
    pub fn with_playlist_size(mut self, size: u32) -> Self {
        self.playlist_size = size;
        self
    }
}

/// ABR (Adaptive Bitrate) ladder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbrLadder {
    /// Bitrate variants
    pub variants: Vec<BitrateVariant>,
}

/// Bitrate variant for ABR streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitrateVariant {
    /// Resolution width
    pub width: u32,
    /// Resolution height
    pub height: u32,
    /// Video bitrate in bits per second
    pub video_bitrate: u64,
    /// Audio bitrate in bits per second
    pub audio_bitrate: u64,
    /// Variant name (e.g., "1080p", "720p")
    pub name: String,
}

impl AbrLadder {
    /// Create a standard ABR ladder.
    #[must_use]
    pub fn standard() -> Self {
        Self {
            variants: vec![
                BitrateVariant {
                    name: "2160p".to_string(),
                    width: 3840,
                    height: 2160,
                    video_bitrate: 35_000_000,
                    audio_bitrate: 192_000,
                },
                BitrateVariant {
                    name: "1080p".to_string(),
                    width: 1920,
                    height: 1080,
                    video_bitrate: 8_000_000,
                    audio_bitrate: 192_000,
                },
                BitrateVariant {
                    name: "720p".to_string(),
                    width: 1280,
                    height: 720,
                    video_bitrate: 5_000_000,
                    audio_bitrate: 128_000,
                },
                BitrateVariant {
                    name: "480p".to_string(),
                    width: 854,
                    height: 480,
                    video_bitrate: 2_500_000,
                    audio_bitrate: 128_000,
                },
                BitrateVariant {
                    name: "360p".to_string(),
                    width: 640,
                    height: 360,
                    video_bitrate: 1_000_000,
                    audio_bitrate: 96_000,
                },
            ],
        }
    }

    /// Add a custom variant.
    #[must_use]
    pub fn add_variant(mut self, variant: BitrateVariant) -> Self {
        self.variants.push(variant);
        self
    }
}

/// Information about a single media segment used in a playlist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentInfo {
    /// URL (or relative path) of the segment file.
    pub url: String,
    /// Duration of the segment in seconds.
    pub duration: f64,
    /// Sequence number (0-based).
    pub sequence: u32,
}

// ---------------------------------------------------------------------------
// Playlist / manifest generators
// ---------------------------------------------------------------------------

/// Generate an HLS master playlist (M3U8) for an [`AbrLadder`].
///
/// Each variant is represented by a `#EXT-X-STREAM-INF` line referencing a
/// per-rendition playlist named `<base_name>_<variant.name>.m3u8`.
#[must_use]
pub fn generate_hls_master_playlist(ladder: &AbrLadder, base_name: &str) -> String {
    let mut out = String::new();
    out.push_str("#EXTM3U\n");
    out.push_str("#EXT-X-VERSION:3\n");

    for variant in &ladder.variants {
        let bandwidth = variant.video_bitrate + variant.audio_bitrate;
        out.push_str(&format!(
            "#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},RESOLUTION={}x{},CODECS=\"avc1.42c01e,mp4a.40.2\"\n",
            variant.width, variant.height
        ));
        out.push_str(&format!("{base_name}_{}.m3u8\n", variant.name));
    }

    out
}

/// Generate an HLS media playlist (M3U8) for a list of segments.
///
/// `target_duration` should be the maximum segment duration (in seconds) across
/// all segments; it is rounded up to the nearest integer as required by RFC 8216.
#[must_use]
pub fn generate_hls_media_playlist(segments: &[SegmentInfo], target_duration: f64) -> String {
    let target_secs = target_duration.ceil() as u64;
    let media_sequence = segments.first().map_or(0, |s| s.sequence);

    let mut out = String::new();
    out.push_str("#EXTM3U\n");
    out.push_str("#EXT-X-VERSION:3\n");
    out.push_str(&format!("#EXT-X-TARGETDURATION:{target_secs}\n"));
    out.push_str(&format!("#EXT-X-MEDIA-SEQUENCE:{media_sequence}\n"));

    for seg in segments {
        out.push_str(&format!("#EXTINF:{:.6},\n", seg.duration));
        out.push_str(&format!("{}\n", seg.url));
    }

    out.push_str("#EXT-X-ENDLIST\n");
    out
}

/// Generate a DASH MPD (Media Presentation Description) manifest for an [`AbrLadder`].
///
/// Produces a minimal but valid MPEG-DASH On-Demand profile MPD with one
/// `AdaptationSet` containing one `Representation` per ladder variant.
#[must_use]
pub fn generate_dash_mpd(
    ladder: &AbrLadder,
    total_duration: Duration,
    segment_duration: f64,
) -> String {
    // ISO 8601 duration: PT<hours>H<minutes>M<seconds>S
    let total_secs = total_duration.as_secs_f64();
    let hours = (total_secs / 3600.0) as u64;
    let minutes = ((total_secs % 3600.0) / 60.0) as u64;
    let secs = total_secs % 60.0;
    let media_duration = format!("PT{hours}H{minutes}M{secs:.3}S");

    let seg_dur_iso = format!("PT{segment_duration:.3}S");

    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(&format!(
        "<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" \
         profiles=\"urn:mpeg:dash:profile:isoff-on-demand:2011\" \
         type=\"static\" \
         mediaPresentationDuration=\"{media_duration}\" \
         minBufferTime=\"{seg_dur_iso}\">\n"
    ));
    out.push_str("  <Period>\n");
    out.push_str(
        "    <AdaptationSet mimeType=\"video/mp4\" \
         codecs=\"avc1.42c01e\" \
         segmentAlignment=\"true\" \
         bitstreamSwitching=\"true\">\n",
    );

    for variant in &ladder.variants {
        out.push_str(&format!(
            "      <Representation id=\"{name}\" \
             bandwidth=\"{bw}\" \
             width=\"{w}\" \
             height=\"{h}\">\n",
            name = variant.name,
            bw = variant.video_bitrate,
            w = variant.width,
            h = variant.height,
        ));
        out.push_str(&format!(
            "        <BaseURL>{name}/</BaseURL>\n",
            name = variant.name
        ));
        out.push_str("      </Representation>\n");
    }

    out.push_str("    </AdaptationSet>\n");
    out.push_str("  </Period>\n");
    out.push_str("</MPD>\n");
    out
}

// ---------------------------------------------------------------------------
// Packager
// ---------------------------------------------------------------------------

/// Streaming packager.
#[derive(Debug, Clone)]
pub struct StreamingPackager {
    config: StreamingConfig,
}

impl StreamingPackager {
    /// Create a new streaming packager.
    #[must_use]
    pub fn new(config: StreamingConfig) -> Self {
        Self { config }
    }

    /// Package video for streaming.
    ///
    /// Creates the output directory (if necessary) and writes a placeholder
    /// manifest to disk, returning the path and basic metadata.
    pub async fn package(&self, _input: &Path) -> Result<PackageResult> {
        std::fs::create_dir_all(&self.config.output_dir)?;

        let manifest_path = self.config.output_dir.join(&self.config.playlist_name);

        // Generate a minimal valid playlist/manifest to write to disk.
        let content = match self.config.format {
            StreamingFormat::Hls => {
                // Single-quality HLS media playlist with no segments yet.
                generate_hls_media_playlist(&[], self.config.segment_duration)
            }
            StreamingFormat::Dash => {
                // Minimal single-rendition DASH MPD.
                let minimal_ladder = AbrLadder {
                    variants: vec![BitrateVariant {
                        name: "default".to_string(),
                        width: 1920,
                        height: 1080,
                        video_bitrate: 8_000_000,
                        audio_bitrate: 192_000,
                    }],
                };
                generate_dash_mpd(
                    &minimal_ladder,
                    Duration::from_secs(0),
                    self.config.segment_duration,
                )
            }
        };

        std::fs::write(&manifest_path, content)?;

        Ok(PackageResult {
            manifest_path,
            segment_count: 0,
            total_duration: Duration::from_secs(0),
        })
    }

    /// Package with ABR ladder.
    ///
    /// Writes an HLS master playlist (or DASH MPD) for the given ladder to
    /// disk and returns the resulting manifest path.
    pub async fn package_abr(&self, _input: &Path, ladder: &AbrLadder) -> Result<PackageResult> {
        std::fs::create_dir_all(&self.config.output_dir)?;

        let manifest_path = self.config.output_dir.join(&self.config.playlist_name);

        let content = match self.config.format {
            StreamingFormat::Hls => {
                let base_name = self
                    .config
                    .playlist_name
                    .trim_end_matches(".m3u8")
                    .to_string();
                generate_hls_master_playlist(ladder, &base_name)
            }
            StreamingFormat::Dash => {
                generate_dash_mpd(ladder, Duration::from_secs(0), self.config.segment_duration)
            }
        };

        std::fs::write(&manifest_path, content)?;

        Ok(PackageResult {
            manifest_path,
            segment_count: 0,
            total_duration: Duration::from_secs(0),
        })
    }
}

/// Package result.
#[derive(Debug, Clone)]
pub struct PackageResult {
    /// Path to manifest file
    pub manifest_path: PathBuf,
    /// Number of segments created
    pub segment_count: usize,
    /// Total duration
    pub total_duration: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Config / ladder tests (preserved from original)
    // -----------------------------------------------------------------------

    #[test]
    fn test_hls_config() {
        let config = StreamingConfig::hls(PathBuf::from("/tmp/output"));
        assert_eq!(config.format, StreamingFormat::Hls);
        assert_eq!(config.playlist_name, "playlist.m3u8");
    }

    #[test]
    fn test_dash_config() {
        let config = StreamingConfig::dash(PathBuf::from("/tmp/output"));
        assert_eq!(config.format, StreamingFormat::Dash);
        assert_eq!(config.playlist_name, "manifest.mpd");
    }

    #[test]
    fn test_streaming_config_builder() {
        let config = StreamingConfig::hls(PathBuf::from("/tmp/output"))
            .with_segment_duration(10.0)
            .with_playlist_size(10);

        assert_eq!(config.segment_duration, 10.0);
        assert_eq!(config.playlist_size, 10);
    }

    #[test]
    fn test_abr_ladder() {
        let ladder = AbrLadder::standard();
        assert_eq!(ladder.variants.len(), 5);
        assert_eq!(ladder.variants[0].name, "2160p");
        assert_eq!(ladder.variants[1].width, 1920);
    }

    #[test]
    fn test_abr_ladder_add_variant() {
        let ladder = AbrLadder::standard().add_variant(BitrateVariant {
            name: "custom".to_string(),
            width: 1024,
            height: 576,
            video_bitrate: 3_000_000,
            audio_bitrate: 128_000,
        });

        assert_eq!(ladder.variants.len(), 6);
    }

    // -----------------------------------------------------------------------
    // HLS master playlist generator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_hls_master_playlist_header() {
        let ladder = AbrLadder::standard();
        let playlist = generate_hls_master_playlist(&ladder, "stream");

        assert!(
            playlist.starts_with("#EXTM3U\n"),
            "Master playlist must start with #EXTM3U"
        );
        assert!(
            playlist.contains("#EXT-X-VERSION:3"),
            "Master playlist must contain version tag"
        );
    }

    #[test]
    fn test_generate_hls_master_playlist_variants() {
        let ladder = AbrLadder::standard();
        let playlist = generate_hls_master_playlist(&ladder, "stream");

        // Five variants means five STREAM-INF lines.
        let count = playlist.matches("#EXT-X-STREAM-INF").count();
        assert_eq!(count, 5, "Should have one STREAM-INF per variant");

        // Each variant playlist filename should appear.
        assert!(playlist.contains("stream_2160p.m3u8"));
        assert!(playlist.contains("stream_720p.m3u8"));
        assert!(playlist.contains("stream_360p.m3u8"));
    }

    #[test]
    fn test_generate_hls_master_playlist_bandwidth() {
        let ladder = AbrLadder {
            variants: vec![BitrateVariant {
                name: "1080p".to_string(),
                width: 1920,
                height: 1080,
                video_bitrate: 8_000_000,
                audio_bitrate: 192_000,
            }],
        };
        let playlist = generate_hls_master_playlist(&ladder, "video");

        // bandwidth = 8_000_000 + 192_000 = 8_192_000
        assert!(
            playlist.contains("BANDWIDTH=8192000"),
            "Bandwidth must be sum of video + audio bitrate"
        );
        assert!(playlist.contains("RESOLUTION=1920x1080"));
        assert!(playlist.contains("CODECS=\"avc1.42c01e,mp4a.40.2\""));
    }

    // -----------------------------------------------------------------------
    // HLS media playlist generator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_hls_media_playlist_empty() {
        let playlist = generate_hls_media_playlist(&[], 6.0);

        assert!(playlist.contains("#EXTM3U"));
        assert!(playlist.contains("#EXT-X-VERSION:3"));
        assert!(playlist.contains("#EXT-X-TARGETDURATION:6"));
        assert!(playlist.contains("#EXT-X-MEDIA-SEQUENCE:0"));
        assert!(playlist.contains("#EXT-X-ENDLIST"));
    }

    #[test]
    fn test_generate_hls_media_playlist_with_segments() {
        let segments = vec![
            SegmentInfo {
                url: "seg0.ts".to_string(),
                duration: 6.0,
                sequence: 0,
            },
            SegmentInfo {
                url: "seg1.ts".to_string(),
                duration: 5.5,
                sequence: 1,
            },
            SegmentInfo {
                url: "seg2.ts".to_string(),
                duration: 6.0,
                sequence: 2,
            },
        ];
        let playlist = generate_hls_media_playlist(&segments, 6.0);

        assert!(playlist.contains("#EXT-X-MEDIA-SEQUENCE:0"));
        // Three EXTINF lines
        let count = playlist.matches("#EXTINF:").count();
        assert_eq!(count, 3);
        assert!(playlist.contains("seg0.ts"));
        assert!(playlist.contains("seg1.ts"));
        assert!(playlist.contains("seg2.ts"));
        assert!(playlist.contains("#EXT-X-ENDLIST"));
    }

    #[test]
    fn test_generate_hls_media_playlist_target_duration_ceil() {
        let segments = vec![SegmentInfo {
            url: "seg.ts".to_string(),
            duration: 5.1,
            sequence: 10,
        }];
        // target_duration 5.1 should be ceiled to 6
        let playlist = generate_hls_media_playlist(&segments, 5.1);
        assert!(
            playlist.contains("#EXT-X-TARGETDURATION:6"),
            "Target duration must be ceiled to nearest integer"
        );
        // Media sequence starts at segment's sequence number
        assert!(playlist.contains("#EXT-X-MEDIA-SEQUENCE:10"));
    }

    // -----------------------------------------------------------------------
    // DASH MPD generator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_dash_mpd_structure() {
        let ladder = AbrLadder::standard();
        let mpd = generate_dash_mpd(&ladder, Duration::from_secs(3600), 4.0);

        assert!(
            mpd.starts_with("<?xml"),
            "MPD must start with XML declaration"
        );
        assert!(mpd.contains("<MPD "));
        assert!(mpd.contains("<Period>"));
        assert!(mpd.contains("<AdaptationSet "));
        assert!(mpd.contains("</AdaptationSet>"));
        assert!(mpd.contains("</Period>"));
        assert!(mpd.contains("</MPD>"));
    }

    #[test]
    fn test_generate_dash_mpd_representations() {
        let ladder = AbrLadder::standard();
        let mpd = generate_dash_mpd(&ladder, Duration::from_secs(60), 4.0);

        // Five variants
        let count = mpd.matches("<Representation ").count();
        assert_eq!(count, 5, "Should have one Representation per variant");

        assert!(mpd.contains("id=\"1080p\""));
        assert!(mpd.contains("width=\"1920\""));
        assert!(mpd.contains("height=\"1080\""));
        assert!(mpd.contains("bandwidth=\"8000000\""));
    }

    #[test]
    fn test_generate_dash_mpd_duration() {
        let mpd = generate_dash_mpd(&AbrLadder::standard(), Duration::from_secs(90), 4.0);
        // 90 seconds = PT0H1M30.000S
        assert!(
            mpd.contains("mediaPresentationDuration="),
            "MPD must contain mediaPresentationDuration"
        );
        assert!(
            mpd.contains("PT0H1M30."),
            "Duration should encode 90 seconds as PT0H1M30.xxxS"
        );
    }

    // -----------------------------------------------------------------------
    // Packager integration tests (writes to /tmp)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_package_hls_writes_manifest() {
        let dir = std::path::PathBuf::from("/tmp/oximedia_test_hls_package");
        let config = StreamingConfig::hls(dir.clone());
        let packager = StreamingPackager::new(config);
        let result = packager
            .package(std::path::Path::new("/dev/null"))
            .await
            .unwrap();

        assert!(
            result.manifest_path.exists(),
            "HLS manifest must be written to disk"
        );
        let content = std::fs::read_to_string(&result.manifest_path).unwrap();
        assert!(content.contains("#EXTM3U"));
        // cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_package_abr_hls_writes_master_playlist() {
        let dir = std::path::PathBuf::from("/tmp/oximedia_test_hls_abr");
        let config = StreamingConfig::hls(dir.clone());
        let packager = StreamingPackager::new(config);
        let ladder = AbrLadder::standard();
        let result = packager
            .package_abr(std::path::Path::new("/dev/null"), &ladder)
            .await
            .unwrap();

        assert!(
            result.manifest_path.exists(),
            "Master playlist must be written to disk"
        );
        let content = std::fs::read_to_string(&result.manifest_path).unwrap();
        assert!(content.contains("#EXTM3U"));
        assert!(content.contains("#EXT-X-STREAM-INF"));
        // cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_package_dash_writes_mpd() {
        let dir = std::path::PathBuf::from("/tmp/oximedia_test_dash_package");
        let config = StreamingConfig::dash(dir.clone());
        let packager = StreamingPackager::new(config);
        let result = packager
            .package(std::path::Path::new("/dev/null"))
            .await
            .unwrap();

        assert!(
            result.manifest_path.exists(),
            "DASH MPD must be written to disk"
        );
        let content = std::fs::read_to_string(&result.manifest_path).unwrap();
        assert!(content.contains("<MPD "));
        // cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_package_abr_dash_writes_mpd() {
        let dir = std::path::PathBuf::from("/tmp/oximedia_test_dash_abr");
        let config = StreamingConfig::dash(dir.clone());
        let packager = StreamingPackager::new(config);
        let ladder = AbrLadder::standard();
        let result = packager
            .package_abr(std::path::Path::new("/dev/null"), &ladder)
            .await
            .unwrap();

        assert!(
            result.manifest_path.exists(),
            "DASH ABR MPD must be written to disk"
        );
        let content = std::fs::read_to_string(&result.manifest_path).unwrap();
        assert!(content.contains("<Representation "));
        // cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }
}
