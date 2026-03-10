//! Codec and media type identifiers.
//!
//! This module provides the [`CodecId`] enum for identifying codecs
//! and the [`MediaType`] enum for categorizing streams.
//!
//! **Important**: Only patent-free (Green List) codecs are supported.

/// Media type for stream classification.
///
/// Categorizes streams by their content type.
///
/// # Examples
///
/// ```
/// use oximedia_core::types::MediaType;
///
/// let media = MediaType::Video;
/// assert!(matches!(media, MediaType::Video));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MediaType {
    /// Video content (moving images).
    Video,
    /// Audio content (sound).
    Audio,
    /// Subtitle/caption content.
    Subtitle,
    /// Arbitrary data stream.
    Data,
    /// Attachment (fonts, images, etc.).
    Attachment,
}

impl std::fmt::Display for MediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Video => "video",
            Self::Audio => "audio",
            Self::Subtitle => "subtitle",
            Self::Data => "data",
            Self::Attachment => "attachment",
        };
        write!(f, "{name}")
    }
}

/// Codec identifier for supported codecs.
///
/// **Green List Only**: Only patent-free codecs are supported.
/// Using patent-encumbered codecs (H.264, H.265, AAC, etc.) will
/// result in [`OxiError::PatentViolation`](crate::error::OxiError::PatentViolation).
///
/// # Supported Video Codecs
///
/// - [`Av1`](Self::Av1) - AV1 (Alliance for Open Media)
/// - [`Vp9`](Self::Vp9) - VP9 (Google/WebM)
/// - [`Vp8`](Self::Vp8) - VP8 (Google/WebM)
/// - [`Theora`](Self::Theora) - Theora (Xiph.org)
///
/// # Supported Audio Codecs
///
/// - [`Opus`](Self::Opus) - Opus (IETF/Xiph.org)
/// - [`Vorbis`](Self::Vorbis) - Vorbis (Xiph.org)
/// - [`Flac`](Self::Flac) - FLAC (Xiph.org)
/// - [`Mp3`](Self::Mp3) - MP3 (MPEG-1/2 Layer III, patents expired 2017)
/// - [`Pcm`](Self::Pcm) - Uncompressed PCM
///
/// # Supported Subtitle Formats
///
/// - [`WebVtt`](Self::WebVtt) - `WebVTT`
/// - [`Ass`](Self::Ass) - Advanced `SubStation` Alpha
/// - [`Ssa`](Self::Ssa) - `SubStation` Alpha
/// - [`Srt`](Self::Srt) - `SubRip`
///
/// # Examples
///
/// ```
/// use oximedia_core::types::{CodecId, MediaType};
///
/// let codec = CodecId::Av1;
/// assert_eq!(codec.media_type(), MediaType::Video);
/// assert!(codec.is_video());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum CodecId {
    // Video codecs (Green List)
    /// AV1 video codec (Alliance for Open Media).
    Av1,
    /// VP9 video codec (Google/WebM).
    Vp9,
    /// VP8 video codec (Google/WebM).
    Vp8,
    /// Theora video codec (Xiph.org).
    Theora,
    /// H.263 video codec (patents expired 2019, for education/compatibility).
    H263,
    /// Raw uncompressed video (e.g., Y4M / YUV4MPEG2).
    RawVideo,

    // Image codecs (Green List)
    /// JPEG-XL image codec (ISO/IEC 18181, royalty-free).
    JpegXl,
    /// DNG (Digital Negative) RAW image format (Adobe open standard, royalty-free).
    Dng,
    /// FFV1 lossless video codec (RFC 9043 / ISO/IEC 24114).
    Ffv1,

    // Audio codecs (Green List)
    /// Opus audio codec (IETF/Xiph.org).
    Opus,
    /// Vorbis audio codec (Xiph.org).
    Vorbis,
    /// FLAC lossless audio codec (Xiph.org).
    Flac,
    /// MP3 audio codec (MPEG-1/2 Layer III, patents expired 2017).
    Mp3,
    /// Uncompressed PCM audio.
    Pcm,

    // Subtitle formats
    /// `WebVTT` subtitle format.
    WebVtt,
    /// Advanced `SubStation` Alpha subtitle format.
    Ass,
    /// `SubStation` Alpha subtitle format.
    Ssa,
    /// `SubRip` subtitle format.
    Srt,
}

impl CodecId {
    /// Returns the media type for this codec.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::{CodecId, MediaType};
    ///
    /// assert_eq!(CodecId::Av1.media_type(), MediaType::Video);
    /// assert_eq!(CodecId::Opus.media_type(), MediaType::Audio);
    /// assert_eq!(CodecId::WebVtt.media_type(), MediaType::Subtitle);
    /// ```
    #[must_use]
    pub const fn media_type(&self) -> MediaType {
        match self {
            Self::Av1
            | Self::Vp9
            | Self::Vp8
            | Self::Theora
            | Self::H263
            | Self::RawVideo
            | Self::JpegXl
            | Self::Dng
            | Self::Ffv1 => MediaType::Video,
            Self::Opus | Self::Vorbis | Self::Flac | Self::Mp3 | Self::Pcm => MediaType::Audio,
            Self::WebVtt | Self::Ass | Self::Ssa | Self::Srt => MediaType::Subtitle,
        }
    }

    /// Returns true if this is a video codec.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::CodecId;
    ///
    /// assert!(CodecId::Av1.is_video());
    /// assert!(!CodecId::Opus.is_video());
    /// ```
    #[must_use]
    pub const fn is_video(&self) -> bool {
        matches!(self.media_type(), MediaType::Video)
    }

    /// Returns true if this is an audio codec.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::CodecId;
    ///
    /// assert!(CodecId::Opus.is_audio());
    /// assert!(!CodecId::Av1.is_audio());
    /// ```
    #[must_use]
    pub const fn is_audio(&self) -> bool {
        matches!(self.media_type(), MediaType::Audio)
    }

    /// Returns true if this is a subtitle format.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::CodecId;
    ///
    /// assert!(CodecId::WebVtt.is_subtitle());
    /// assert!(!CodecId::Opus.is_subtitle());
    /// ```
    #[must_use]
    pub const fn is_subtitle(&self) -> bool {
        matches!(self.media_type(), MediaType::Subtitle)
    }

    /// Returns the codec name as a string.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::CodecId;
    ///
    /// assert_eq!(CodecId::Av1.name(), "av1");
    /// assert_eq!(CodecId::Opus.name(), "opus");
    /// ```
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Av1 => "av1",
            Self::Vp9 => "vp9",
            Self::Vp8 => "vp8",
            Self::Theora => "theora",
            Self::H263 => "h263",
            Self::RawVideo => "rawvideo",
            Self::JpegXl => "jpegxl",
            Self::Dng => "dng",
            Self::Ffv1 => "ffv1",
            Self::Opus => "opus",
            Self::Vorbis => "vorbis",
            Self::Flac => "flac",
            Self::Mp3 => "mp3",
            Self::Pcm => "pcm",
            Self::WebVtt => "webvtt",
            Self::Ass => "ass",
            Self::Ssa => "ssa",
            Self::Srt => "srt",
        }
    }

    /// Returns true if this codec supports lossless compression.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::CodecId;
    ///
    /// assert!(CodecId::Flac.is_lossless());
    /// assert!(CodecId::Pcm.is_lossless());
    /// assert!(!CodecId::Opus.is_lossless());
    /// ```
    #[must_use]
    pub const fn is_lossless(&self) -> bool {
        matches!(
            self,
            Self::Flac | Self::Pcm | Self::RawVideo | Self::JpegXl | Self::Dng | Self::Ffv1
        )
    }
}

impl std::fmt::Display for CodecId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_media_type() {
        assert_eq!(CodecId::Av1.media_type(), MediaType::Video);
        assert_eq!(CodecId::Vp9.media_type(), MediaType::Video);
        assert_eq!(CodecId::Vp8.media_type(), MediaType::Video);
        assert_eq!(CodecId::Theora.media_type(), MediaType::Video);

        assert_eq!(CodecId::Opus.media_type(), MediaType::Audio);
        assert_eq!(CodecId::Vorbis.media_type(), MediaType::Audio);
        assert_eq!(CodecId::Flac.media_type(), MediaType::Audio);
        assert_eq!(CodecId::Mp3.media_type(), MediaType::Audio);
        assert_eq!(CodecId::Pcm.media_type(), MediaType::Audio);

        assert_eq!(CodecId::WebVtt.media_type(), MediaType::Subtitle);
        assert_eq!(CodecId::Ass.media_type(), MediaType::Subtitle);
        assert_eq!(CodecId::Ssa.media_type(), MediaType::Subtitle);
        assert_eq!(CodecId::Srt.media_type(), MediaType::Subtitle);
    }

    #[test]
    fn test_is_video() {
        assert!(CodecId::Av1.is_video());
        assert!(CodecId::Vp9.is_video());
        assert!(!CodecId::Opus.is_video());
        assert!(!CodecId::WebVtt.is_video());
    }

    #[test]
    fn test_is_audio() {
        assert!(CodecId::Opus.is_audio());
        assert!(CodecId::Flac.is_audio());
        assert!(!CodecId::Av1.is_audio());
        assert!(!CodecId::Srt.is_audio());
    }

    #[test]
    fn test_is_subtitle() {
        assert!(CodecId::WebVtt.is_subtitle());
        assert!(CodecId::Ass.is_subtitle());
        assert!(!CodecId::Av1.is_subtitle());
        assert!(!CodecId::Opus.is_subtitle());
    }

    #[test]
    fn test_name() {
        assert_eq!(CodecId::Av1.name(), "av1");
        assert_eq!(CodecId::Opus.name(), "opus");
        assert_eq!(CodecId::WebVtt.name(), "webvtt");
    }

    #[test]
    fn test_is_lossless() {
        assert!(CodecId::Flac.is_lossless());
        assert!(CodecId::Pcm.is_lossless());
        assert!(!CodecId::Opus.is_lossless());
        assert!(!CodecId::Av1.is_lossless());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", CodecId::Av1), "av1");
        assert_eq!(format!("{}", MediaType::Video), "video");
    }
}
