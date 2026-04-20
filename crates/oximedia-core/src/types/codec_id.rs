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
/// - [`Mjpeg`](Self::Mjpeg) - Motion JPEG (patents expired, royalty-free)
/// - [`Apv`](Self::Apv) - Advanced Professional Video (ISO/IEC 23009-13, royalty-free)
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
    /// Motion JPEG (sequence of baseline JPEG frames).
    /// Baseline JPEG patents expired (1992 filing + 20 years). Royalty-free.
    Mjpeg,
    /// Advanced Professional Video (APV).
    /// ISO/IEC 23009-13. Royalty-free intra-frame professional codec.
    Apv,

    // Image codecs (Green List)
    /// JPEG-XL image codec (ISO/IEC 18181, royalty-free).
    JpegXl,
    /// DNG (Digital Negative) RAW image format (Adobe open standard, royalty-free).
    Dng,
    /// FFV1 lossless video codec (RFC 9043 / ISO/IEC 24114).
    Ffv1,
    /// WebP image format (Google, royalty-free).
    WebP,
    /// GIF image format (patents expired, royalty-free since 2004).
    Gif,
    /// PNG lossless image format (W3C royalty-free).
    Png,
    /// TIFF image format (Adobe, royalty-free for core spec).
    Tiff,
    /// OpenEXR HDR image format (Academy Software Foundation, BSD-3).
    OpenExr,

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
            | Self::Mjpeg
            | Self::Apv
            | Self::JpegXl
            | Self::Dng
            | Self::Ffv1
            | Self::WebP
            | Self::Gif
            | Self::Png
            | Self::Tiff
            | Self::OpenExr => MediaType::Video,
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
            Self::Mjpeg => "mjpeg",
            Self::Apv => "apv",
            Self::JpegXl => "jpegxl",
            Self::Dng => "dng",
            Self::Ffv1 => "ffv1",
            Self::WebP => "webp",
            Self::Gif => "gif",
            Self::Png => "png",
            Self::Tiff => "tiff",
            Self::OpenExr => "openexr",
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
            Self::Flac
                | Self::Pcm
                | Self::RawVideo
                | Self::JpegXl
                | Self::Dng
                | Self::Ffv1
                | Self::Png
                | Self::Tiff
                | Self::OpenExr
        )
    }

    /// Returns the canonical lower-case name for this codec.
    ///
    /// This is an alias for [`name()`](Self::name) provided for API symmetry
    /// with `FromStr`, where parsing is case-insensitive but serialization
    /// always uses the canonical name.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::CodecId;
    ///
    /// assert_eq!(CodecId::Av1.canonical_name(), "av1");
    /// assert_eq!(CodecId::WebP.canonical_name(), "webp");
    /// assert_eq!(CodecId::Gif.canonical_name(), "gif");
    /// assert_eq!(CodecId::JpegXl.canonical_name(), "jpegxl");
    /// ```
    #[must_use]
    pub const fn canonical_name(&self) -> &'static str {
        self.name()
    }
}

impl std::fmt::Display for CodecId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for CodecId {
    type Err = crate::OxiError;

    /// Parses a codec name string into a [`CodecId`].
    ///
    /// Parsing is **case-insensitive** and accepts several common aliases.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::CodecId;
    ///
    /// assert_eq!("av1".parse::<CodecId>().unwrap(), CodecId::Av1);
    /// assert_eq!("WEBP".parse::<CodecId>().unwrap(), CodecId::WebP);
    /// assert_eq!("jxl".parse::<CodecId>().unwrap(), CodecId::JpegXl);
    /// assert_eq!("gif".parse::<CodecId>().unwrap(), CodecId::Gif);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            // Video codecs
            "av1" => Ok(Self::Av1),
            "vp9" => Ok(Self::Vp9),
            "vp8" => Ok(Self::Vp8),
            "theora" => Ok(Self::Theora),
            "h263" => Ok(Self::H263),
            "rawvideo" | "raw" | "yuv4mpeg" | "y4m" => Ok(Self::RawVideo),
            "mjpeg" | "mjpg" | "motion_jpeg" | "motionjpeg" => Ok(Self::Mjpeg),
            "apv" => Ok(Self::Apv),
            // Image codecs
            "jpegxl" | "jpeg-xl" | "jxl" | "jxl " => Ok(Self::JpegXl),
            "dng" => Ok(Self::Dng),
            "ffv1" => Ok(Self::Ffv1),
            "webp" => Ok(Self::WebP),
            "gif" => Ok(Self::Gif),
            "png" => Ok(Self::Png),
            "tiff" | "tif" => Ok(Self::Tiff),
            "openexr" | "exr" => Ok(Self::OpenExr),
            // Audio codecs
            "opus" => Ok(Self::Opus),
            "vorbis" => Ok(Self::Vorbis),
            "flac" => Ok(Self::Flac),
            "mp3" | "mpeg1audio" | "mpeg1_audio" => Ok(Self::Mp3),
            "pcm" | "raw_audio" | "rawaudio" => Ok(Self::Pcm),
            // Subtitle formats
            "webvtt" | "vtt" => Ok(Self::WebVtt),
            "ass" => Ok(Self::Ass),
            "ssa" => Ok(Self::Ssa),
            "srt" | "subrip" => Ok(Self::Srt),
            _ => Err(crate::OxiError::Unsupported(format!(
                "Unknown or patent-encumbered codec: {s}"
            ))),
        }
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
        assert!(CodecId::Png.is_lossless());
        assert!(CodecId::Tiff.is_lossless());
        assert!(CodecId::OpenExr.is_lossless());
        assert!(!CodecId::WebP.is_lossless());
        assert!(!CodecId::Gif.is_lossless());
        assert!(!CodecId::Opus.is_lossless());
        assert!(!CodecId::Av1.is_lossless());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", CodecId::Av1), "av1");
        assert_eq!(format!("{}", MediaType::Video), "video");
    }

    #[test]
    fn test_new_image_codecs_are_video() {
        assert_eq!(CodecId::WebP.media_type(), MediaType::Video);
        assert_eq!(CodecId::Gif.media_type(), MediaType::Video);
        assert_eq!(CodecId::Png.media_type(), MediaType::Video);
        assert_eq!(CodecId::Tiff.media_type(), MediaType::Video);
        assert_eq!(CodecId::OpenExr.media_type(), MediaType::Video);
        assert!(CodecId::WebP.is_video());
        assert!(CodecId::Gif.is_video());
        assert!(CodecId::Png.is_video());
        assert!(!CodecId::WebP.is_audio());
        assert!(!CodecId::Png.is_subtitle());
    }

    #[test]
    fn test_new_codec_names() {
        assert_eq!(CodecId::WebP.name(), "webp");
        assert_eq!(CodecId::Gif.name(), "gif");
        assert_eq!(CodecId::Png.name(), "png");
        assert_eq!(CodecId::Tiff.name(), "tiff");
        assert_eq!(CodecId::OpenExr.name(), "openexr");
    }

    #[test]
    fn test_mjpeg_apv_media_type() {
        assert_eq!(CodecId::Mjpeg.media_type(), MediaType::Video);
        assert_eq!(CodecId::Apv.media_type(), MediaType::Video);
        assert!(CodecId::Mjpeg.is_video());
        assert!(CodecId::Apv.is_video());
        assert!(!CodecId::Mjpeg.is_audio());
        assert!(!CodecId::Apv.is_subtitle());
    }

    #[test]
    fn test_mjpeg_apv_names() {
        assert_eq!(CodecId::Mjpeg.name(), "mjpeg");
        assert_eq!(CodecId::Apv.name(), "apv");
        assert_eq!(format!("{}", CodecId::Mjpeg), "mjpeg");
        assert_eq!(format!("{}", CodecId::Apv), "apv");
    }

    #[test]
    fn test_mjpeg_apv_is_lossless() {
        // Both are DCT-based lossy codecs by default.
        assert!(!CodecId::Mjpeg.is_lossless());
        assert!(!CodecId::Apv.is_lossless());
    }

    // ── canonical_name tests ─────────────────────────────────────────────────

    #[test]
    fn test_canonical_name_equals_name() {
        // canonical_name() is a const-fn alias for name() — must always agree.
        let all = [
            CodecId::Av1,
            CodecId::Vp9,
            CodecId::Vp8,
            CodecId::Theora,
            CodecId::H263,
            CodecId::RawVideo,
            CodecId::Mjpeg,
            CodecId::Apv,
            CodecId::JpegXl,
            CodecId::Dng,
            CodecId::Ffv1,
            CodecId::WebP,
            CodecId::Gif,
            CodecId::Png,
            CodecId::Tiff,
            CodecId::OpenExr,
            CodecId::Opus,
            CodecId::Vorbis,
            CodecId::Flac,
            CodecId::Mp3,
            CodecId::Pcm,
            CodecId::WebVtt,
            CodecId::Ass,
            CodecId::Ssa,
            CodecId::Srt,
        ];
        for codec in &all {
            assert_eq!(
                codec.canonical_name(),
                codec.name(),
                "{codec:?} canonical_name != name"
            );
        }
    }

    #[test]
    fn test_canonical_name_webp_gif_jpegxl() {
        assert_eq!(CodecId::WebP.canonical_name(), "webp");
        assert_eq!(CodecId::Gif.canonical_name(), "gif");
        assert_eq!(CodecId::JpegXl.canonical_name(), "jpegxl");
    }

    // ── FromStr tests ────────────────────────────────────────────────────────

    #[test]
    fn test_from_str_all_video_codecs() {
        assert_eq!("av1".parse::<CodecId>().expect("parse"), CodecId::Av1);
        assert_eq!("vp9".parse::<CodecId>().expect("parse"), CodecId::Vp9);
        assert_eq!("vp8".parse::<CodecId>().expect("parse"), CodecId::Vp8);
        assert_eq!("theora".parse::<CodecId>().expect("parse"), CodecId::Theora);
        assert_eq!("h263".parse::<CodecId>().expect("parse"), CodecId::H263);
        assert_eq!(
            "rawvideo".parse::<CodecId>().expect("parse"),
            CodecId::RawVideo
        );
        assert_eq!("mjpeg".parse::<CodecId>().expect("parse"), CodecId::Mjpeg);
        assert_eq!("apv".parse::<CodecId>().expect("parse"), CodecId::Apv);
    }

    #[test]
    fn test_from_str_image_codecs() {
        assert_eq!("jpegxl".parse::<CodecId>().expect("parse"), CodecId::JpegXl);
        assert_eq!("dng".parse::<CodecId>().expect("parse"), CodecId::Dng);
        assert_eq!("ffv1".parse::<CodecId>().expect("parse"), CodecId::Ffv1);
        assert_eq!("webp".parse::<CodecId>().expect("parse"), CodecId::WebP);
        assert_eq!("gif".parse::<CodecId>().expect("parse"), CodecId::Gif);
        assert_eq!("png".parse::<CodecId>().expect("parse"), CodecId::Png);
        assert_eq!("tiff".parse::<CodecId>().expect("parse"), CodecId::Tiff);
        assert_eq!(
            "openexr".parse::<CodecId>().expect("parse"),
            CodecId::OpenExr
        );
    }

    #[test]
    fn test_from_str_all_audio_codecs() {
        assert_eq!("opus".parse::<CodecId>().expect("parse"), CodecId::Opus);
        assert_eq!("vorbis".parse::<CodecId>().expect("parse"), CodecId::Vorbis);
        assert_eq!("flac".parse::<CodecId>().expect("parse"), CodecId::Flac);
        assert_eq!("mp3".parse::<CodecId>().expect("parse"), CodecId::Mp3);
        assert_eq!("pcm".parse::<CodecId>().expect("parse"), CodecId::Pcm);
    }

    #[test]
    fn test_from_str_subtitles() {
        assert_eq!("webvtt".parse::<CodecId>().expect("parse"), CodecId::WebVtt);
        assert_eq!("ass".parse::<CodecId>().expect("parse"), CodecId::Ass);
        assert_eq!("ssa".parse::<CodecId>().expect("parse"), CodecId::Ssa);
        assert_eq!("srt".parse::<CodecId>().expect("parse"), CodecId::Srt);
    }

    #[test]
    fn test_from_str_case_insensitive() {
        assert_eq!("AV1".parse::<CodecId>().expect("parse"), CodecId::Av1);
        assert_eq!("WEBP".parse::<CodecId>().expect("parse"), CodecId::WebP);
        assert_eq!("GIF".parse::<CodecId>().expect("parse"), CodecId::Gif);
        assert_eq!("JpegXl".parse::<CodecId>().expect("parse"), CodecId::JpegXl);
        assert_eq!("OPUS".parse::<CodecId>().expect("parse"), CodecId::Opus);
        assert_eq!("WEBVTT".parse::<CodecId>().expect("parse"), CodecId::WebVtt);
    }

    #[test]
    fn test_from_str_aliases() {
        // mjpg alias for Mjpeg
        assert_eq!("mjpg".parse::<CodecId>().expect("parse"), CodecId::Mjpeg);
        // jxl alias for JpegXl
        assert_eq!("jxl".parse::<CodecId>().expect("parse"), CodecId::JpegXl);
        // jpeg-xl alias for JpegXl
        assert_eq!(
            "jpeg-xl".parse::<CodecId>().expect("parse"),
            CodecId::JpegXl
        );
        // vtt alias for WebVtt
        assert_eq!("vtt".parse::<CodecId>().expect("parse"), CodecId::WebVtt);
        // exr alias for OpenExr
        assert_eq!("exr".parse::<CodecId>().expect("parse"), CodecId::OpenExr);
        // raw alias for RawVideo
        assert_eq!("raw".parse::<CodecId>().expect("parse"), CodecId::RawVideo);
        // tif alias for Tiff
        assert_eq!("tif".parse::<CodecId>().expect("parse"), CodecId::Tiff);
        // subrip alias for Srt
        assert_eq!("subrip".parse::<CodecId>().expect("parse"), CodecId::Srt);
    }

    #[test]
    fn test_from_str_unknown_returns_err() {
        assert!("h264".parse::<CodecId>().is_err());
        assert!("h265".parse::<CodecId>().is_err());
        assert!("aac".parse::<CodecId>().is_err());
        assert!("hevc".parse::<CodecId>().is_err());
        assert!("unknown".parse::<CodecId>().is_err());
    }

    #[test]
    fn test_from_str_roundtrip() {
        // Every codec round-trips through canonical_name
        let all = [
            CodecId::Av1,
            CodecId::Vp9,
            CodecId::Vp8,
            CodecId::Theora,
            CodecId::H263,
            CodecId::RawVideo,
            CodecId::Mjpeg,
            CodecId::Apv,
            CodecId::JpegXl,
            CodecId::Dng,
            CodecId::Ffv1,
            CodecId::WebP,
            CodecId::Gif,
            CodecId::Png,
            CodecId::Tiff,
            CodecId::OpenExr,
            CodecId::Opus,
            CodecId::Vorbis,
            CodecId::Flac,
            CodecId::Mp3,
            CodecId::Pcm,
            CodecId::WebVtt,
            CodecId::Ass,
            CodecId::Ssa,
            CodecId::Srt,
        ];
        for codec in &all {
            let name = codec.canonical_name();
            let parsed: CodecId = name
                .parse()
                .unwrap_or_else(|_| panic!("{name} should round-trip via FromStr"));
            assert_eq!(*codec, parsed, "Round-trip failed for {name}");
        }
    }
}
