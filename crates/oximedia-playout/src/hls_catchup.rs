//! HLS manifest generation for catchup / VOD platform export.
//!
//! This module integrates with the `catchup` module to produce HTTP Live
//! Streaming (HLS) manifests (`.m3u8`) that can be consumed by VOD platforms,
//! CDNs, and player clients.
//!
//! # Overview
//!
//! Given a set of [`VodSegment`] records produced from a completed catch-up
//! recording, [`HlsManifestBuilder`] constructs:
//!
//! - A **Media Playlist** (RFC 8216 §4.3.3) — a flat list of `#EXTINF` entries
//!   pointing to individual MPEG-TS or CMAF segments.
//! - A **Master Playlist** (RFC 8216 §4.3.4) — an optional multi-bitrate
//!   rendition manifest that references separate Media Playlists for each
//!   quality level (HD, SD, mobile proxy).
//!
//! # Retention policy
//!
//! [`RetentionPolicy`] controls how long a finished VOD asset remains
//! accessible before segments can be purged.  [`VodCatalogue`] enforces the
//! policy: `purge_expired` removes assets whose age exceeds the configured
//! window.
//!
//! # References
//! - RFC 8216 — HTTP Live Streaming
//! - SMPTE ST 2042-1 (VC-2) is *not* required; standard MPEG-TS/CMAF muxing
//!   is assumed.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// Segment
// ---------------------------------------------------------------------------

/// A single HLS media segment.
#[derive(Debug, Clone)]
pub struct VodSegment {
    /// Segment index within the rendition playlist (0-based).
    pub index: u32,
    /// URI path or URL referencing the segment file.
    pub uri: String,
    /// Duration of this segment in seconds (as a float, e.g. `6.0`).
    pub duration_secs: f64,
    /// Byte range (start, length) within the media file, if using byte-range
    /// addressing (`#EXT-X-BYTERANGE`).
    pub byte_range: Option<(u64, u64)>,
    /// Whether this segment marks a discontinuity boundary.
    pub discontinuity: bool,
}

impl VodSegment {
    /// Create a standard segment (no byte-range, no discontinuity).
    pub fn new(index: u32, uri: impl Into<String>, duration_secs: f64) -> Self {
        Self {
            index,
            uri: uri.into(),
            duration_secs,
            byte_range: None,
            discontinuity: false,
        }
    }

    /// Mark this segment as a discontinuity point.
    pub fn with_discontinuity(mut self) -> Self {
        self.discontinuity = true;
        self
    }
}

// ---------------------------------------------------------------------------
// Rendition
// ---------------------------------------------------------------------------

/// A single rendition (quality level) for a Master Playlist entry.
#[derive(Debug, Clone)]
pub struct HlsRendition {
    /// Human-readable label (e.g. "HD 1080p", "SD 480p").
    pub label: String,
    /// Peak bandwidth in bits per second (used in `BANDWIDTH=…`).
    pub bandwidth_bps: u32,
    /// Average bandwidth (optional, used in `AVERAGE-BANDWIDTH=…`).
    pub average_bandwidth_bps: Option<u32>,
    /// Codec string for the `CODECS=…` attribute (e.g. `"avc1.64001f,mp4a.40.2"`).
    pub codecs: String,
    /// Video resolution (width, height) for the `RESOLUTION=…` attribute.
    pub resolution: Option<(u32, u32)>,
    /// Frame rate (for `FRAME-RATE=…` attribute).
    pub frame_rate: Option<f64>,
    /// URI of the Media Playlist for this rendition.
    pub playlist_uri: String,
    /// Segments that make up this rendition.
    pub segments: Vec<VodSegment>,
}

impl HlsRendition {
    /// Create a new rendition descriptor.
    pub fn new(
        label: impl Into<String>,
        bandwidth_bps: u32,
        codecs: impl Into<String>,
        playlist_uri: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            bandwidth_bps,
            average_bandwidth_bps: None,
            codecs: codecs.into(),
            resolution: None,
            frame_rate: None,
            playlist_uri: playlist_uri.into(),
            segments: Vec::new(),
        }
    }

    /// Set the video resolution for the `RESOLUTION=` attribute.
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.resolution = Some((width, height));
        self
    }

    /// Set the frame rate for the `FRAME-RATE=` attribute.
    pub fn with_frame_rate(mut self, fps: f64) -> Self {
        self.frame_rate = Some(fps);
        self
    }
}

// ---------------------------------------------------------------------------
// HLS Manifest Builder
// ---------------------------------------------------------------------------

/// Configuration for the HLS manifest builder.
#[derive(Debug, Clone)]
pub struct HlsManifestConfig {
    /// HLS version to declare (default: 7).
    pub version: u8,
    /// Target segment duration in seconds (used in `#EXT-X-TARGETDURATION`).
    pub target_duration: u32,
    /// Whether to include `#EXT-X-ENDLIST` (always true for completed VOD).
    pub end_list: bool,
    /// Optional playlist type (`VOD`, `EVENT`).
    pub playlist_type: Option<HlsPlaylistType>,
    /// Optional `#EXT-X-ALLOW-CACHE` directive.
    pub allow_cache: Option<bool>,
}

/// HLS playlist type tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HlsPlaylistType {
    /// `#EXT-X-PLAYLIST-TYPE:VOD` — segments are immutable.
    Vod,
    /// `#EXT-X-PLAYLIST-TYPE:EVENT` — segments are appended; history is fixed.
    Event,
}

impl Default for HlsManifestConfig {
    fn default() -> Self {
        Self {
            version: 7,
            target_duration: 6,
            end_list: true,
            playlist_type: Some(HlsPlaylistType::Vod),
            allow_cache: None,
        }
    }
}

/// Builds HLS `.m3u8` manifests from a set of [`VodSegment`] records.
#[derive(Debug)]
pub struct HlsManifestBuilder {
    config: HlsManifestConfig,
}

impl HlsManifestBuilder {
    /// Create a new builder with the provided configuration.
    pub fn new(config: HlsManifestConfig) -> Self {
        Self { config }
    }

    /// Build a Media Playlist (single rendition) from `segments`.
    ///
    /// The returned string is a complete, valid `.m3u8` file.
    pub fn build_media_playlist(&self, segments: &[VodSegment]) -> String {
        let mut out = String::with_capacity(512 + segments.len() * 80);

        out.push_str("#EXTM3U\n");
        out.push_str(&format!("#EXT-X-VERSION:{}\n", self.config.version));
        out.push_str(&format!(
            "#EXT-X-TARGETDURATION:{}\n",
            self.config.target_duration
        ));

        if let Some(pt) = self.config.playlist_type {
            match pt {
                HlsPlaylistType::Vod => out.push_str("#EXT-X-PLAYLIST-TYPE:VOD\n"),
                HlsPlaylistType::Event => out.push_str("#EXT-X-PLAYLIST-TYPE:EVENT\n"),
            }
        }

        if let Some(allow) = self.config.allow_cache {
            if allow {
                out.push_str("#EXT-X-ALLOW-CACHE:YES\n");
            } else {
                out.push_str("#EXT-X-ALLOW-CACHE:NO\n");
            }
        }

        for segment in segments {
            if segment.discontinuity {
                out.push_str("#EXT-X-DISCONTINUITY\n");
            }

            // #EXTINF:<duration>,<title>
            out.push_str(&format!("#EXTINF:{:.6},\n", segment.duration_secs));

            if let Some((start, len)) = segment.byte_range {
                out.push_str(&format!("#EXT-X-BYTERANGE:{}@{}\n", len, start));
            }

            out.push_str(&segment.uri);
            out.push('\n');
        }

        if self.config.end_list {
            out.push_str("#EXT-X-ENDLIST\n");
        }

        out
    }

    /// Build a Master Playlist from a list of `HlsRendition` descriptors.
    ///
    /// Each rendition is listed as a `#EXT-X-STREAM-INF` entry.
    pub fn build_master_playlist(&self, renditions: &[HlsRendition]) -> String {
        let mut out = String::with_capacity(256 + renditions.len() * 120);

        out.push_str("#EXTM3U\n");
        out.push_str(&format!("#EXT-X-VERSION:{}\n", self.config.version));

        for rendition in renditions {
            // Build the STREAM-INF attribute list.
            let mut attrs = format!("BANDWIDTH={}", rendition.bandwidth_bps);

            if let Some(avg) = rendition.average_bandwidth_bps {
                attrs.push_str(&format!(",AVERAGE-BANDWIDTH={avg}"));
            }

            if !rendition.codecs.is_empty() {
                attrs.push_str(&format!(",CODECS=\"{}\"", rendition.codecs));
            }

            if let Some((w, h)) = rendition.resolution {
                attrs.push_str(&format!(",RESOLUTION={w}x{h}"));
            }

            if let Some(fps) = rendition.frame_rate {
                attrs.push_str(&format!(",FRAME-RATE={fps:.3}"));
            }

            out.push_str(&format!("#EXT-X-STREAM-INF:{attrs}\n"));
            out.push_str(&rendition.playlist_uri);
            out.push('\n');
        }

        out
    }

    /// Compute the total duration of a segment list in seconds.
    pub fn total_duration(segments: &[VodSegment]) -> f64 {
        segments.iter().map(|s| s.duration_secs).sum()
    }

    /// Infer the `#EXT-X-TARGETDURATION` value from `segments`.
    ///
    /// The target duration must be the ceiling of the maximum segment duration.
    pub fn infer_target_duration(segments: &[VodSegment]) -> u32 {
        segments
            .iter()
            .map(|s| s.duration_secs.ceil() as u32)
            .max()
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Retention policy
// ---------------------------------------------------------------------------

/// Governs how long completed VOD assets are retained before purging.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Maximum age of a VOD asset before it may be purged.
    pub max_age: Duration,
    /// Whether to always retain assets with `pinned == true`.
    pub respect_pin: bool,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_age: Duration::from_hours(168), // 7 days
            respect_pin: true,
        }
    }
}

impl RetentionPolicy {
    /// Return `true` if `created_at` is within the retention window.
    pub fn is_within_window(&self, created_at: SystemTime) -> bool {
        match SystemTime::now().duration_since(created_at) {
            Ok(age) => age <= self.max_age,
            Err(_) => true, // future timestamp — always retain
        }
    }
}

// ---------------------------------------------------------------------------
// VOD Asset
// ---------------------------------------------------------------------------

/// A completed catch-up / VOD asset ready for HLS export.
#[derive(Debug, Clone)]
pub struct VodAsset {
    /// Unique identifier.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// Time at which this asset was recorded / created.
    pub created_at: SystemTime,
    /// Whether this asset is pinned (exempted from automatic purging).
    pub pinned: bool,
    /// Renditions available for this asset (multi-bitrate).
    pub renditions: Vec<HlsRendition>,
}

impl VodAsset {
    /// Create a new VOD asset.
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            created_at: SystemTime::now(),
            pinned: false,
            renditions: Vec::new(),
        }
    }

    /// Add a rendition to this asset.
    pub fn add_rendition(&mut self, rendition: HlsRendition) {
        self.renditions.push(rendition);
    }

    /// Return the primary (highest-bandwidth) rendition, if any.
    pub fn primary_rendition(&self) -> Option<&HlsRendition> {
        self.renditions
            .iter()
            .max_by_key(|r| r.bandwidth_bps)
    }
}

// ---------------------------------------------------------------------------
// VOD Catalogue
// ---------------------------------------------------------------------------

/// Maintains a catalogue of [`VodAsset`] entries and enforces
/// [`RetentionPolicy`] on them.
#[derive(Debug, Default)]
pub struct VodCatalogue {
    assets: HashMap<String, VodAsset>,
    policy: RetentionPolicy,
}

impl VodCatalogue {
    /// Create an empty catalogue with the default retention policy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a catalogue with a custom retention policy.
    pub fn with_policy(policy: RetentionPolicy) -> Self {
        Self {
            assets: HashMap::new(),
            policy,
        }
    }

    /// Register a new VOD asset.
    pub fn register(&mut self, asset: VodAsset) {
        self.assets.insert(asset.id.clone(), asset);
    }

    /// Retrieve an asset by ID.
    pub fn get(&self, id: &str) -> Option<&VodAsset> {
        self.assets.get(id)
    }

    /// Number of assets in the catalogue.
    pub fn count(&self) -> usize {
        self.assets.len()
    }

    /// Remove an asset by ID.  Returns `true` if the asset existed.
    pub fn remove(&mut self, id: &str) -> bool {
        self.assets.remove(id).is_some()
    }

    /// Purge all assets that have exceeded the retention window.
    ///
    /// Pinned assets (`pinned == true`) are always retained if
    /// `RetentionPolicy::respect_pin` is enabled.
    ///
    /// Returns the IDs of purged assets.
    pub fn purge_expired(&mut self) -> Vec<String> {
        let policy = &self.policy;
        let to_purge: Vec<String> = self
            .assets
            .iter()
            .filter(|(_, asset)| {
                if policy.respect_pin && asset.pinned {
                    return false;
                }
                !policy.is_within_window(asset.created_at)
            })
            .map(|(id, _)| id.clone())
            .collect();

        for id in &to_purge {
            self.assets.remove(id);
        }
        to_purge
    }

    /// Return all asset IDs.
    pub fn asset_ids(&self) -> Vec<&str> {
        self.assets.keys().map(|s| s.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// Full catchup-to-HLS pipeline helper
// ---------------------------------------------------------------------------

/// Convenience function: convert a list of segment URIs (each with a uniform
/// duration) into a ready-to-serve HLS media playlist string.
///
/// `segment_duration_secs` must be positive; the target duration is inferred
/// automatically.
pub fn segments_to_hls(
    segment_uris: &[&str],
    segment_duration_secs: f64,
) -> Result<String, String> {
    if segment_duration_secs <= 0.0 {
        return Err(format!(
            "segment_duration_secs must be positive, got {segment_duration_secs}"
        ));
    }

    let segments: Vec<VodSegment> = segment_uris
        .iter()
        .enumerate()
        .map(|(i, &uri)| VodSegment::new(i as u32, uri, segment_duration_secs))
        .collect();

    let target = HlsManifestBuilder::infer_target_duration(&segments);
    let config = HlsManifestConfig {
        target_duration: target,
        ..HlsManifestConfig::default()
    };
    let builder = HlsManifestBuilder::new(config);
    Ok(builder.build_media_playlist(&segments))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── VodSegment ────────────────────────────────────────────────────────────

    #[test]
    fn test_vod_segment_new() {
        let seg = VodSegment::new(0, "seg0.ts", 6.0);
        assert_eq!(seg.index, 0);
        assert_eq!(seg.uri, "seg0.ts");
        assert!(!seg.discontinuity);
        assert!(seg.byte_range.is_none());
    }

    #[test]
    fn test_vod_segment_with_discontinuity() {
        let seg = VodSegment::new(5, "gap.ts", 3.0).with_discontinuity();
        assert!(seg.discontinuity);
    }

    // ── HlsManifestBuilder::build_media_playlist ─────────────────────────────

    #[test]
    fn test_build_media_playlist_basic() {
        let builder = HlsManifestBuilder::new(HlsManifestConfig::default());
        let segments = vec![
            VodSegment::new(0, "seg0.ts", 6.0),
            VodSegment::new(1, "seg1.ts", 6.0),
            VodSegment::new(2, "seg2.ts", 4.5),
        ];
        let m3u8 = builder.build_media_playlist(&segments);

        assert!(m3u8.starts_with("#EXTM3U\n"), "must start with #EXTM3U");
        assert!(m3u8.contains("#EXT-X-VERSION:7"), "must contain version 7");
        assert!(
            m3u8.contains("#EXT-X-TARGETDURATION:6"),
            "must contain target duration"
        );
        assert!(m3u8.contains("#EXT-X-PLAYLIST-TYPE:VOD"));
        assert!(m3u8.contains("seg0.ts"));
        assert!(m3u8.contains("seg1.ts"));
        assert!(m3u8.contains("seg2.ts"));
        assert!(m3u8.contains("#EXT-X-ENDLIST"), "VOD must have ENDLIST");
    }

    #[test]
    fn test_build_media_playlist_discontinuity_tag() {
        let builder = HlsManifestBuilder::new(HlsManifestConfig::default());
        let segments = vec![
            VodSegment::new(0, "a.ts", 6.0),
            VodSegment::new(1, "b.ts", 6.0).with_discontinuity(),
            VodSegment::new(2, "c.ts", 6.0),
        ];
        let m3u8 = builder.build_media_playlist(&segments);
        assert!(
            m3u8.contains("#EXT-X-DISCONTINUITY"),
            "discontinuity tag must be present"
        );
    }

    #[test]
    fn test_build_media_playlist_extinf_format() {
        let builder = HlsManifestBuilder::new(HlsManifestConfig::default());
        let segments = vec![VodSegment::new(0, "s.ts", 6.006)];
        let m3u8 = builder.build_media_playlist(&segments);
        // Duration should be formatted to 6 decimal places
        assert!(m3u8.contains("#EXTINF:6.006000,"));
    }

    #[test]
    fn test_build_media_playlist_empty_segments() {
        let builder = HlsManifestBuilder::new(HlsManifestConfig::default());
        let m3u8 = builder.build_media_playlist(&[]);
        assert!(m3u8.contains("#EXTM3U"));
        assert!(m3u8.contains("#EXT-X-ENDLIST"));
    }

    // ── HlsManifestBuilder::build_master_playlist ────────────────────────────

    #[test]
    fn test_build_master_playlist() {
        let builder = HlsManifestBuilder::new(HlsManifestConfig::default());
        let renditions = vec![
            HlsRendition::new("HD", 4_000_000, "avc1.64001f,mp4a.40.2", "hd/playlist.m3u8")
                .with_resolution(1920, 1080)
                .with_frame_rate(25.0),
            HlsRendition::new("SD", 800_000, "avc1.64001e,mp4a.40.2", "sd/playlist.m3u8")
                .with_resolution(854, 480),
        ];
        let master = builder.build_master_playlist(&renditions);

        assert!(master.starts_with("#EXTM3U\n"));
        assert!(master.contains("#EXT-X-STREAM-INF:"));
        assert!(master.contains("BANDWIDTH=4000000"));
        assert!(master.contains("RESOLUTION=1920x1080"));
        assert!(master.contains("hd/playlist.m3u8"));
        assert!(master.contains("sd/playlist.m3u8"));
        assert!(master.contains("CODECS=\"avc1.64001f,mp4a.40.2\""));
    }

    // ── HlsManifestBuilder utility ────────────────────────────────────────────

    #[test]
    fn test_total_duration() {
        let segments = vec![
            VodSegment::new(0, "a.ts", 6.0),
            VodSegment::new(1, "b.ts", 6.0),
            VodSegment::new(2, "c.ts", 3.5),
        ];
        let total = HlsManifestBuilder::total_duration(&segments);
        assert!((total - 15.5).abs() < 1e-6);
    }

    #[test]
    fn test_infer_target_duration() {
        let segments = vec![
            VodSegment::new(0, "a.ts", 6.0),
            VodSegment::new(1, "b.ts", 7.3),
            VodSegment::new(2, "c.ts", 4.9),
        ];
        // max ceil = ceil(7.3) = 8
        assert_eq!(HlsManifestBuilder::infer_target_duration(&segments), 8);
    }

    // ── RetentionPolicy ───────────────────────────────────────────────────────

    #[test]
    fn test_retention_policy_within_window() {
        let policy = RetentionPolicy {
            max_age: Duration::from_secs(3600),
            respect_pin: true,
        };
        // Just created → within window
        assert!(policy.is_within_window(SystemTime::now()));
    }

    #[test]
    fn test_retention_policy_expired() {
        let policy = RetentionPolicy {
            max_age: Duration::from_secs(1),
            respect_pin: true,
        };
        // 2 hours ago → expired
        let old = SystemTime::now() - Duration::from_secs(7200);
        assert!(!policy.is_within_window(old));
    }

    // ── VodCatalogue ──────────────────────────────────────────────────────────

    #[test]
    fn test_vod_catalogue_register_and_get() {
        let mut cat = VodCatalogue::new();
        let asset = VodAsset::new("a1", "News at Ten");
        cat.register(asset);
        assert_eq!(cat.count(), 1);
        assert!(cat.get("a1").is_some());
        assert!(cat.get("missing").is_none());
    }

    #[test]
    fn test_vod_catalogue_remove() {
        let mut cat = VodCatalogue::new();
        cat.register(VodAsset::new("a2", "Documentary"));
        assert!(cat.remove("a2"));
        assert_eq!(cat.count(), 0);
        assert!(!cat.remove("a2")); // idempotent
    }

    #[test]
    fn test_vod_catalogue_purge_expired() {
        let policy = RetentionPolicy {
            max_age: Duration::from_secs(1),
            respect_pin: false,
        };
        let mut cat = VodCatalogue::with_policy(policy);

        // Create an asset with a timestamp 2 hours in the past.
        let mut old_asset = VodAsset::new("old", "Old Show");
        old_asset.created_at = SystemTime::now() - Duration::from_secs(7200);
        cat.register(old_asset);

        let fresh_asset = VodAsset::new("fresh", "Live News");
        cat.register(fresh_asset);

        let purged = cat.purge_expired();
        assert_eq!(purged.len(), 1, "only the old asset should be purged");
        assert_eq!(purged[0], "old");
        assert_eq!(cat.count(), 1);
        assert!(cat.get("fresh").is_some());
    }

    #[test]
    fn test_vod_catalogue_purge_respects_pin() {
        let policy = RetentionPolicy {
            max_age: Duration::from_secs(1),
            respect_pin: true,
        };
        let mut cat = VodCatalogue::with_policy(policy);

        let mut pinned = VodAsset::new("pinned", "Award Show");
        pinned.created_at = SystemTime::now() - Duration::from_secs(7200);
        pinned.pinned = true;
        cat.register(pinned);

        let purged = cat.purge_expired();
        assert!(purged.is_empty(), "pinned asset should not be purged");
        assert_eq!(cat.count(), 1);
    }

    // ── segments_to_hls ───────────────────────────────────────────────────────

    #[test]
    fn test_segments_to_hls_valid() {
        let uris = ["seg0.ts", "seg1.ts", "seg2.ts"];
        let m3u8 = segments_to_hls(&uris, 6.0).expect("should succeed");
        assert!(m3u8.contains("seg0.ts"));
        assert!(m3u8.contains("#EXT-X-ENDLIST"));
    }

    #[test]
    fn test_segments_to_hls_invalid_duration() {
        let result = segments_to_hls(&["seg.ts"], 0.0);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("segment_duration_secs must be positive"),
        );
    }

    // ── VodAsset ──────────────────────────────────────────────────────────────

    #[test]
    fn test_vod_asset_primary_rendition() {
        let mut asset = VodAsset::new("x", "Test");
        asset.add_rendition(HlsRendition::new("SD", 800_000, "", "sd.m3u8"));
        asset.add_rendition(HlsRendition::new("HD", 4_000_000, "", "hd.m3u8"));
        let primary = asset.primary_rendition().expect("should have primary");
        assert_eq!(primary.label, "HD");
    }

    #[test]
    fn test_vod_asset_no_renditions() {
        let asset = VodAsset::new("empty", "No Renditions");
        assert!(asset.primary_rendition().is_none());
    }
}
