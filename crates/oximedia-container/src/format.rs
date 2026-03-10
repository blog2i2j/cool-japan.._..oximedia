//! Container format definitions.

/// Supported container formats (patent-free).
///
/// `OxiMedia` focuses on patent-free, royalty-free container formats.
/// The `Mp4` variant is supported only for AV1/VP9 content.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContainerFormat {
    /// Matroska container (.mkv).
    ///
    /// Full-featured container supporting multiple audio/video/subtitle tracks,
    /// chapters, attachments, and rich metadata.
    Matroska,

    /// `WebM` container (.webm).
    ///
    /// Matroska subset optimized for web delivery, supporting VP8/VP9/AV1 video
    /// and Vorbis/Opus audio.
    WebM,

    /// ISOBMFF/MP4 container (.mp4, .m4a, .m4v).
    ///
    /// Only supported for AV1 and VP9 content to maintain patent-free status.
    Mp4,

    /// Ogg container (.ogg, .opus, .oga).
    ///
    /// Xiph.org container format, commonly used for Vorbis and Opus audio,
    /// as well as Theora video.
    Ogg,

    /// WAV/RIFF container (.wav).
    ///
    /// Simple audio container for PCM and other uncompressed audio formats.
    Wav,

    /// FLAC native container (.flac).
    ///
    /// Native container for FLAC lossless audio, supporting metadata and seek tables.
    Flac,

    /// MPEG Transport Stream (.ts, .m2ts, .mts).
    ///
    /// Transport stream format commonly used for broadcast and streaming.
    /// Supports multiplexed video, audio, and subtitle streams with
    /// precise timing via Program Clock Reference (PCR).
    /// Only patent-free codecs (AV1, VP9, VP8, Opus, FLAC) are supported.
    MpegTs,

    /// `WebVTT` subtitle format (.vtt).
    ///
    /// Text-based subtitle format commonly used for web video.
    WebVtt,

    /// `SubRip` subtitle format (.srt).
    ///
    /// Simple text-based subtitle format widely supported.
    Srt,

    /// YUV4MPEG2 container (.y4m).
    ///
    /// Simple uncompressed video sequence format widely used for testing
    /// and piping raw YUV video between tools. Supports various chroma
    /// subsampling modes (420, 422, 444, mono).
    Y4m,
}

impl ContainerFormat {
    /// Returns the common file extensions for this container format.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_container::ContainerFormat;
    ///
    /// let exts = ContainerFormat::Matroska.file_extensions();
    /// assert!(exts.contains(&"mkv"));
    /// assert!(exts.contains(&"mka"));
    /// ```
    #[must_use]
    pub const fn file_extensions(self) -> &'static [&'static str] {
        match self {
            Self::Matroska => &["mkv", "mka", "mks", "mk3d"],
            Self::WebM => &["webm"],
            Self::Mp4 => &["mp4", "m4a", "m4v", "mov"],
            Self::Ogg => &["ogg", "opus", "oga", "ogv", "ogx", "spx"],
            Self::Wav => &["wav", "wave"],
            Self::Flac => &["flac"],
            Self::MpegTs => &["ts", "m2ts", "mts"],
            Self::WebVtt => &["vtt", "webvtt"],
            Self::Srt => &["srt"],
            Self::Y4m => &["y4m", "yuv4mpeg"],
        }
    }

    /// Returns the primary MIME type for this container format.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_container::ContainerFormat;
    ///
    /// assert_eq!(ContainerFormat::WebM.mime_type(), "video/webm");
    /// assert_eq!(ContainerFormat::Ogg.mime_type(), "audio/ogg");
    /// ```
    #[must_use]
    pub const fn mime_type(self) -> &'static str {
        match self {
            Self::Matroska => "video/x-matroska",
            Self::WebM => "video/webm",
            Self::Mp4 => "video/mp4",
            Self::Ogg => "audio/ogg",
            Self::Wav => "audio/wav",
            Self::Flac => "audio/flac",
            Self::MpegTs => "video/mp2t",
            Self::WebVtt => "text/vtt",
            Self::Srt => "application/x-subrip",
            Self::Y4m => "video/x-raw-yuv4mpeg2",
        }
    }

    /// Returns the format name as a human-readable string.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_container::ContainerFormat;
    ///
    /// assert_eq!(ContainerFormat::Matroska.name(), "Matroska");
    /// assert_eq!(ContainerFormat::WebM.name(), "WebM");
    /// ```
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Matroska => "Matroska",
            Self::WebM => "WebM",
            Self::Mp4 => "MP4/ISOBMFF",
            Self::Ogg => "Ogg",
            Self::Wav => "WAV/RIFF",
            Self::Flac => "FLAC",
            Self::MpegTs => "MPEG-TS",
            Self::WebVtt => "WebVTT",
            Self::Srt => "SubRip",
            Self::Y4m => "YUV4MPEG2",
        }
    }

    /// Returns true if this format supports video streams.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_container::ContainerFormat;
    ///
    /// assert!(ContainerFormat::Matroska.supports_video());
    /// assert!(!ContainerFormat::Flac.supports_video());
    /// ```
    #[must_use]
    pub const fn supports_video(self) -> bool {
        match self {
            Self::Matroska | Self::WebM | Self::Mp4 | Self::Ogg | Self::MpegTs | Self::Y4m => true,
            Self::Wav | Self::Flac | Self::WebVtt | Self::Srt => false,
        }
    }

    /// Returns true if this format supports audio streams.
    #[must_use]
    pub const fn supports_audio(self) -> bool {
        match self {
            Self::Matroska
            | Self::WebM
            | Self::Mp4
            | Self::Ogg
            | Self::Wav
            | Self::Flac
            | Self::MpegTs => true,
            Self::WebVtt | Self::Srt | Self::Y4m => false,
        }
    }

    /// Returns true if this format supports subtitle streams.
    #[must_use]
    pub const fn supports_subtitles(self) -> bool {
        match self {
            Self::Matroska | Self::WebM | Self::Mp4 | Self::MpegTs | Self::WebVtt | Self::Srt => {
                true
            }
            Self::Ogg | Self::Wav | Self::Flac | Self::Y4m => false,
        }
    }
}

impl std::fmt::Display for ContainerFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_extensions() {
        assert!(ContainerFormat::Matroska.file_extensions().contains(&"mkv"));
        assert!(ContainerFormat::WebM.file_extensions().contains(&"webm"));
        assert!(ContainerFormat::Ogg.file_extensions().contains(&"opus"));
    }

    #[test]
    fn test_mime_types() {
        assert_eq!(ContainerFormat::Matroska.mime_type(), "video/x-matroska");
        assert_eq!(ContainerFormat::WebM.mime_type(), "video/webm");
        assert_eq!(ContainerFormat::Flac.mime_type(), "audio/flac");
    }

    #[test]
    fn test_supports_video() {
        assert!(ContainerFormat::Matroska.supports_video());
        assert!(ContainerFormat::WebM.supports_video());
        assert!(!ContainerFormat::Wav.supports_video());
        assert!(!ContainerFormat::Flac.supports_video());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ContainerFormat::Matroska), "Matroska");
        assert_eq!(format!("{}", ContainerFormat::WebM), "WebM");
    }
}
