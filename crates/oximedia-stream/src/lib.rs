//! `oximedia-stream` — Adaptive streaming pipeline, segment lifecycle management,
//! and stream health monitoring for the OxiMedia framework.
//!
//! # Modules
//!
//! | Module | Purpose |
//! |---|---|
//! | [`adaptive_pipeline`] | Quality ladder, BOLA-inspired ABR switching |
//! | [`bola`] | BOLA-E buffer-occupancy ABR algorithm implementation |
//! | [`cdn_health`] | CDN health checking with sliding-window probe tracking |
//! | [`cdn_upload`] | Parallel multi-CDN segment upload fan-out |
//! | [`cmaf`] | CMAF chunk muxer for fragmented MP4 output |
//! | [`cmaf_sequencer`] | CMAF chunk accumulator / sequencer |
//! | [`dash_mpd_updater`] | Incremental DASH MPD period update helper |
//! | [`drm_signaling`] | DRM system signaling (Widevine, FairPlay, PlayReady) |
//! | [`dvr_recorder`] | DVR sliding-window recorder with VOD playlist generation |
//! | [`file_recorder`] | Filesystem-backed live stream recorder |
//! | [`ll_dash`] | Low-Latency DASH with CMAF chunked transfer encoding |
//! | [`ll_hls`] | Low-Latency HLS with partial segments (RFC 8216bis) |
//! | [`manifest_builder`] | HLS master/media playlist and DASH MPD generation |
//! | [`multi_audio`] | Multiple audio track variants and language management |
//! | [`multi_cdn`] | Multi-CDN failover routing with EWMA latency tracking |
//! | [`prefetch_scheduler`] | Bandwidth-aware segment prefetch depth scheduler |
//! | [`retry`] | Exponential back-off retry policy for segment fetches |
//! | [`scte35`] | SCTE-35 splice information encoding/parsing/scheduling |
//! | [`segment`] | Segment lifecycle state machine and buffer |
//! | [`segment_cache`] | In-memory LRU segment cache for repeated request serving |
//! | [`segment_manager`] | Segment state machine, prefetch/eviction |
//! | [`srt_ingest`] | SRT protocol ingest as input to the streaming pipeline |
//! | [`stream_analytics`] | Viewer-side playback metrics and QoE aggregation |
//! | [`stream_health`] | QoE scoring, issue detection, history |
//! | [`stream_packager`] | Media unit accumulation and segment packaging |
//! | [`stream_recorder`] | Live stream recorder with DVR sliding-window |
//! | [`subtitle_track`] | WebVTT subtitle segment packaging and manifest integration |
//! | [`thumbnail_track`] | I-frame-only playlists and trick-play manifests |

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(dead_code)]

// ─── Internal error types (used by bola, segment) ────────────────────────────
pub(crate) mod error;

pub mod adaptive_pipeline;
pub mod bola;
pub mod cdn_health;
pub mod cdn_upload;
pub mod cmaf;
pub mod cmaf_sequencer;
pub mod dash_mpd_updater;
pub mod drm_signaling;
pub mod dvr_recorder;
pub mod file_recorder;
pub mod ll_dash;
pub mod ll_hls;
pub mod manifest_builder;
pub mod multi_audio;
pub mod multi_cdn;
pub mod prefetch_scheduler;
pub mod retry;
pub mod scte35;
pub mod segment;
pub mod segment_cache;
pub mod segment_manager;
pub mod srt_ingest;
pub mod stream_analytics;
pub mod stream_health;
pub mod stream_packager;
pub mod stream_recorder;
pub mod subtitle_track;
pub mod throughput_abr;
pub mod thumbnail_track;

// ─── Crate-level error type ───────────────────────────────────────────────────

/// Top-level error type for `oximedia-stream`.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    /// A binary parsing operation encountered invalid or truncated data.
    #[error("parse error: {0}")]
    ParseError(String),

    /// A CDN routing operation could not find an eligible provider.
    #[error("routing error: {0}")]
    RoutingError(String),

    /// An I/O error occurred while reading or writing segment data.
    #[error("I/O error: {0}")]
    IoError(String),

    /// A generic stream-processing error.
    #[error("stream error: {0}")]
    Generic(String),
}

// ─── Re-exports ───────────────────────────────────────────────────────────────

// adaptive_pipeline
pub use adaptive_pipeline::{
    AbrAlgorithm, AdaptivePipeline, BandwidthEstimator, QualityLadder, QualitySwitch, QualityTier,
    SwitchReason,
};

// segment_manager
pub use segment_manager::{MediaSegment, PrefetchConfig, SegmentManager, SegmentState};

// stream_health
pub use stream_health::{
    HealthIssue, QoeConfig, QoeScore, StreamHealthMonitor, StreamHealthReport,
};

// scte35
pub use scte35::{
    encode_bandwidth_reservation, encode_splice_insert, encode_splice_null, parse_splice_info,
    BreakDuration, ScheduledCommand, SpliceCommand, SpliceCommandType, SpliceDescriptor,
    SpliceInfoSection, SpliceInsert, SpliceScheduler, TimeSignal,
};

// multi_cdn
pub use multi_cdn::{CdnProvider, FailoverPolicy, MultiCdnRouter, RoutingStrategy};

// manifest_builder
pub use manifest_builder::{
    build_dash_mpd, build_master_playlist, build_media_playlist, DashMpd, DashRepresentation,
    HlsManifest, HlsSegment, ManifestFormat, SegmentTemplate, StreamVariant,
};

// stream_packager
pub use stream_packager::{
    pack_segment, FileSegmentWriter, MediaUnit, PackagedSegment, PackagerConfig, SegmentPackager,
    SegmentWriter, StreamType,
};

// cmaf
pub use cmaf::{CmafChunk, CmafMuxer};

// ll_hls
pub use ll_hls::{
    BlockingReloadRequest, HintType, LlHlsConfig, LlHlsPlaylist, LlHlsPlaylistState, LlHlsSegment,
    PartialSegment, PreloadHint,
};

// ll_dash
pub use ll_dash::{LlDashChunk, LlDashConfig, LlDashSegment, LlDashTimeline};

// drm_signaling
pub use drm_signaling::{DrmManifestBuilder, DrmSignal, DrmSystem};

// thumbnail_track
pub use thumbnail_track::{ImageFormat, ThumbnailSegment, ThumbnailTrack, ThumbnailTrackBuilder};

// multi_audio
pub use multi_audio::{AudioCodecId, AudioTrack, AudioTrackManager};

// subtitle_track
pub use subtitle_track::{
    SubtitleCue, SubtitlePackager, SubtitleSegment, SubtitleTrack, SubtitleTrackManager,
};

// stream_analytics
pub use stream_analytics::{PlaybackEvent, PlaybackStats, StreamAnalytics};

// dvr_recorder
pub use dvr_recorder::{DvrConfig, DvrRecorder, DvrSegment};

// throughput_abr
pub use throughput_abr::{ThroughputAbr, ThroughputMeasurement};

// bola
pub use bola::{BolaConfig, BolaState};

// cdn_health
pub use cdn_health::{CdnHealthRegistry, HealthCheckConfig, ProbeOutcome, ProviderStatus};

// cdn_upload
pub use cdn_upload::{CdnUploadManager, UploadBatch, UploadTarget};

// srt_ingest
pub use srt_ingest::{SrtConfig, SrtIngest, SrtPacket, SrtStream};

// segment_cache
pub use segment_cache::{CacheStats, SegmentCache};

// prefetch_scheduler
pub use prefetch_scheduler::{PrefetchConfig as PrefetchSchedulerConfig, PrefetchScheduler};

// stream_recorder
pub use stream_recorder::{
    DvrWindow, LiveRecorder, RecordingConfig, RecordingStats, StreamRecorder,
};
