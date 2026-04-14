// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Container operation utilities for media remuxing and inspection.
//!
//! Provides container-format metadata, stream inspection, and remux planning
//! without requiring any external FFI or process invocation.

/// Supported media container formats for remux planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaContainerFormat {
    /// MPEG-4 Part 14 (.mp4)
    Mp4,
    /// Matroska (.mkv)
    Mkv,
    /// `QuickTime` Movie (.mov)
    Mov,
    /// Audio Video Interleave (.avi)
    Avi,
    /// MPEG-2 Transport Stream (.ts)
    Ts,
    /// Flash Video (.flv)
    Flv,
    /// `WebM` (.webm)
    Webm,
}

impl MediaContainerFormat {
    /// Canonical file extension (without leading dot).
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Mkv => "mkv",
            Self::Mov => "mov",
            Self::Avi => "avi",
            Self::Ts => "ts",
            Self::Flv => "flv",
            Self::Webm => "webm",
        }
    }

    /// Whether the format supports embedded chapter markers.
    #[must_use]
    pub const fn supports_chapters(self) -> bool {
        matches!(self, Self::Mp4 | Self::Mkv | Self::Mov)
    }

    /// Whether the format supports embedded subtitle streams.
    #[must_use]
    pub const fn supports_subtitles(self) -> bool {
        matches!(self, Self::Mp4 | Self::Mkv | Self::Mov | Self::Webm)
    }
}

/// Type classification of a media stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamType {
    /// Compressed video elementary stream.
    Video,
    /// Compressed audio elementary stream.
    Audio,
    /// Text or bitmap subtitle stream.
    Subtitle,
    /// Generic binary data / side-data stream.
    Data,
    /// Font or image attachment stream.
    Attachment,
}

impl StreamType {
    /// Returns `true` for video, audio, and subtitle streams.
    #[must_use]
    pub const fn is_media(self) -> bool {
        matches!(self, Self::Video | Self::Audio | Self::Subtitle)
    }
}

/// Metadata describing a single stream inside a container.
#[derive(Debug, Clone)]
pub struct StreamInfo {
    /// Zero-based stream index within the container.
    pub stream_id: u32,
    /// Codec identifier string (e.g. `"h264"`, `"aac"`).
    pub codec: String,
    /// Functional type of the stream.
    pub stream_type: StreamType,
    /// Nominal bitrate in kbps (0 if unknown).
    pub bitrate_kbps: u32,
    /// BCP-47 language tag, if present.
    pub language: Option<String>,
}

impl StreamInfo {
    /// Create a new `StreamInfo`.
    #[must_use]
    pub fn new(
        stream_id: u32,
        codec: impl Into<String>,
        stream_type: StreamType,
        bitrate_kbps: u32,
        language: Option<String>,
    ) -> Self {
        Self {
            stream_id,
            codec: codec.into(),
            stream_type,
            bitrate_kbps,
            language,
        }
    }

    /// Returns `true` when the stream is the first stream of its type
    /// (`stream_id` == 0 by convention when building test manifests).
    #[must_use]
    pub const fn is_default(&self) -> bool {
        self.stream_id == 0
    }
}

/// Chapter list entry: `(start_time_ms, title)`.
pub type ChapterEntry = (u64, String);

/// High-level description of a media container's contents.
#[derive(Debug, Clone)]
pub struct ContainerManifest {
    /// Container format.
    pub format: MediaContainerFormat,
    /// Total duration in milliseconds.
    pub duration_ms: u64,
    /// All streams in declaration order.
    pub streams: Vec<StreamInfo>,
    /// Chapter markers `(start_ms, title)`.
    pub chapters: Vec<ChapterEntry>,
}

impl ContainerManifest {
    /// Create a new, empty `ContainerManifest`.
    #[must_use]
    pub fn new(format: MediaContainerFormat, duration_ms: u64) -> Self {
        Self {
            format,
            duration_ms,
            streams: Vec::new(),
            chapters: Vec::new(),
        }
    }

    /// All video streams.
    #[must_use]
    pub fn video_streams(&self) -> Vec<&StreamInfo> {
        self.streams
            .iter()
            .filter(|s| s.stream_type == StreamType::Video)
            .collect()
    }

    /// All audio streams.
    #[must_use]
    pub fn audio_streams(&self) -> Vec<&StreamInfo> {
        self.streams
            .iter()
            .filter(|s| s.stream_type == StreamType::Audio)
            .collect()
    }

    /// Find a stream by its `stream_id`.
    #[must_use]
    pub fn find_stream(&self, id: u32) -> Option<&StreamInfo> {
        self.streams.iter().find(|s| s.stream_id == id)
    }

    /// Returns `true` if at least one subtitle stream is present.
    #[must_use]
    pub fn has_subtitles(&self) -> bool {
        self.streams
            .iter()
            .any(|s| s.stream_type == StreamType::Subtitle)
    }
}

/// A plan describing which streams to copy during a remux operation.
#[derive(Debug, Clone)]
pub struct RemuxPlan {
    /// Source container format.
    pub src_format: MediaContainerFormat,
    /// Destination container format.
    pub dst_format: MediaContainerFormat,
    /// Stream IDs to copy verbatim (no re-encoding).
    pub streams_to_copy: Vec<u32>,
    /// If `true`, at least one stream must be re-encoded (e.g. codec mismatch).
    pub needs_transcode: bool,
}

impl RemuxPlan {
    /// Create a new `RemuxPlan`.
    #[must_use]
    pub fn new(
        src_format: MediaContainerFormat,
        dst_format: MediaContainerFormat,
        streams_to_copy: Vec<u32>,
        needs_transcode: bool,
    ) -> Self {
        Self {
            src_format,
            dst_format,
            streams_to_copy,
            needs_transcode,
        }
    }

    /// Number of streams scheduled for copying.
    #[must_use]
    pub fn stream_count(&self) -> usize {
        self.streams_to_copy.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- MediaContainerFormat ---

    #[test]
    fn test_extensions() {
        assert_eq!(MediaContainerFormat::Mp4.extension(), "mp4");
        assert_eq!(MediaContainerFormat::Mkv.extension(), "mkv");
        assert_eq!(MediaContainerFormat::Mov.extension(), "mov");
        assert_eq!(MediaContainerFormat::Avi.extension(), "avi");
        assert_eq!(MediaContainerFormat::Ts.extension(), "ts");
        assert_eq!(MediaContainerFormat::Flv.extension(), "flv");
        assert_eq!(MediaContainerFormat::Webm.extension(), "webm");
    }

    #[test]
    fn test_supports_chapters() {
        assert!(MediaContainerFormat::Mp4.supports_chapters());
        assert!(MediaContainerFormat::Mkv.supports_chapters());
        assert!(MediaContainerFormat::Mov.supports_chapters());
        assert!(!MediaContainerFormat::Avi.supports_chapters());
        assert!(!MediaContainerFormat::Flv.supports_chapters());
    }

    #[test]
    fn test_supports_subtitles() {
        assert!(MediaContainerFormat::Mp4.supports_subtitles());
        assert!(MediaContainerFormat::Mkv.supports_subtitles());
        assert!(MediaContainerFormat::Webm.supports_subtitles());
        assert!(!MediaContainerFormat::Avi.supports_subtitles());
        assert!(!MediaContainerFormat::Ts.supports_subtitles());
    }

    // --- StreamType ---

    #[test]
    fn test_stream_type_is_media() {
        assert!(StreamType::Video.is_media());
        assert!(StreamType::Audio.is_media());
        assert!(StreamType::Subtitle.is_media());
        assert!(!StreamType::Data.is_media());
        assert!(!StreamType::Attachment.is_media());
    }

    // --- StreamInfo ---

    #[test]
    fn test_stream_info_is_default() {
        let s0 = StreamInfo::new(0, "h264", StreamType::Video, 5000, None);
        let s1 = StreamInfo::new(1, "aac", StreamType::Audio, 128, None);
        assert!(s0.is_default());
        assert!(!s1.is_default());
    }

    // --- ContainerManifest ---

    fn sample_manifest() -> ContainerManifest {
        let mut m = ContainerManifest::new(MediaContainerFormat::Mkv, 120_000);
        m.streams
            .push(StreamInfo::new(0, "av1", StreamType::Video, 4000, None));
        m.streams.push(StreamInfo::new(
            1,
            "opus",
            StreamType::Audio,
            128,
            Some("en".to_string()),
        ));
        m.streams.push(StreamInfo::new(
            2,
            "srt",
            StreamType::Subtitle,
            0,
            Some("fr".to_string()),
        ));
        m.chapters.push((0, "Intro".to_string()));
        m.chapters.push((60_000, "Main".to_string()));
        m
    }

    #[test]
    fn test_video_streams() {
        let m = sample_manifest();
        assert_eq!(m.video_streams().len(), 1);
        assert_eq!(m.video_streams()[0].codec, "av1");
    }

    #[test]
    fn test_audio_streams() {
        let m = sample_manifest();
        assert_eq!(m.audio_streams().len(), 1);
        assert_eq!(m.audio_streams()[0].codec, "opus");
    }

    #[test]
    fn test_find_stream_found() {
        let m = sample_manifest();
        let s = m.find_stream(1);
        assert!(s.is_some());
        assert_eq!(s.unwrap().codec, "opus");
    }

    #[test]
    fn test_find_stream_not_found() {
        let m = sample_manifest();
        assert!(m.find_stream(99).is_none());
    }

    #[test]
    fn test_has_subtitles_true() {
        let m = sample_manifest();
        assert!(m.has_subtitles());
    }

    #[test]
    fn test_has_subtitles_false() {
        let m = ContainerManifest::new(MediaContainerFormat::Mp4, 60_000);
        assert!(!m.has_subtitles());
    }

    // --- RemuxPlan ---

    #[test]
    fn test_remux_plan_stream_count() {
        let plan = RemuxPlan::new(
            MediaContainerFormat::Mkv,
            MediaContainerFormat::Mp4,
            vec![0, 1],
            false,
        );
        assert_eq!(plan.stream_count(), 2);
    }

    #[test]
    fn test_remux_plan_needs_transcode() {
        let plan = RemuxPlan::new(
            MediaContainerFormat::Avi,
            MediaContainerFormat::Webm,
            vec![0],
            true,
        );
        assert!(plan.needs_transcode);
    }

    #[test]
    fn test_remux_plan_empty_streams() {
        let plan = RemuxPlan::new(
            MediaContainerFormat::Ts,
            MediaContainerFormat::Mp4,
            vec![],
            false,
        );
        assert_eq!(plan.stream_count(), 0);
    }
}
