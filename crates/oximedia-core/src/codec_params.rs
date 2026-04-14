//! Codec parameter types for video and audio streams.
//!
//! Provides codec-agnostic wrappers around common encoding/decoding parameters
//! such as resolution, frame rate, sample rate, channel count, and bit depth.
//! These types decouple codec configuration from specific codec implementations
//! and allow parameter passing through the pipeline without importing
//! heavy codec crates.
//!
//! # Structure
//!
//! - [`VideoParams`] — width, height, frame rate, pixel format, colour space
//! - [`AudioParams`] — sample rate, channel count, sample format, bit rate
//! - [`CodecParams`] — codec-agnostic union of video or audio params with a
//!   [`CodecId`] label
//! - `CodecParamsBuilder` — ergonomic builder for [`CodecParams`]
//!
//! # Example
//!
//! ```
//! use oximedia_core::codec_params::{AudioParams, CodecParams, VideoParams};
//! use oximedia_core::types::{CodecId, PixelFormat, Rational, SampleFormat};
//!
//! let video = CodecParams::video(
//!     CodecId::Av1,
//!     VideoParams::new(1920, 1080, Rational::new(30, 1)),
//! );
//!
//! assert!(video.is_video());
//! assert_eq!(video.video_params().map(|v| v.width), Some(1920));
//!
//! let audio = CodecParams::audio(
//!     CodecId::Opus,
//!     AudioParams::new(48_000, 2),
//! );
//!
//! assert!(audio.is_audio());
//! assert_eq!(audio.audio_params().map(|a| a.sample_rate), Some(48_000));
//! ```

use crate::types::{CodecId, PixelFormat, Rational, SampleFormat};

// ---------------------------------------------------------------------------
// ColorSpace
// ---------------------------------------------------------------------------

/// Colour-space / matrix coefficients used for YCbCr ↔ RGB conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColorSpace {
    /// ITU-R BT.601 (standard definition).
    Bt601,
    /// ITU-R BT.709 (high definition, most common).
    #[default]
    Bt709,
    /// ITU-R BT.2020 (ultra-high definition, HDR).
    Bt2020,
    /// Display P3 (wide colour gamut, common on Apple devices).
    DisplayP3,
    /// Identity / RGB pass-through (no conversion required).
    Rgb,
    /// Unknown / unspecified.
    Unknown,
}

impl std::fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Bt601 => "bt601",
            Self::Bt709 => "bt709",
            Self::Bt2020 => "bt2020",
            Self::DisplayP3 => "display_p3",
            Self::Rgb => "rgb",
            Self::Unknown => "unknown",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// ChromaLocation
// ---------------------------------------------------------------------------

/// Chroma sample location for sub-sampled formats (e.g. YUV 4:2:0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ChromaLocation {
    /// Chroma samples are co-sited with the top-left luma sample.
    #[default]
    Left,
    /// Chroma samples are centred vertically between luma rows (MPEG-1 style).
    Centre,
    /// Chroma samples are co-sited with the top-right luma sample.
    Right,
    /// Unspecified / unknown location.
    Unspecified,
}

// ---------------------------------------------------------------------------
// VideoParams
// ---------------------------------------------------------------------------

/// Video stream encoding / decoding parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoParams {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame rate as a rational number (e.g. `Rational::new(30, 1)` for 30 fps,
    /// `Rational::new(30000, 1001)` for ≈29.97 fps).
    pub frame_rate: Rational,
    /// Pixel format (default: [`PixelFormat::Yuv420p`]).
    pub pixel_format: PixelFormat,
    /// Colour space (default: [`ColorSpace::Bt709`]).
    pub color_space: ColorSpace,
    /// Chroma sample location (default: [`ChromaLocation::Left`]).
    pub chroma_location: ChromaLocation,
    /// Display aspect ratio, if different from the storage aspect ratio.
    pub display_aspect_ratio: Option<Rational>,
    /// Bit depth per colour component (8 for standard, 10 or 12 for HDR).
    pub bit_depth: u8,
    /// Target or measured peak bitrate in bits per second, if known.
    pub bitrate_bps: Option<u64>,
}

impl VideoParams {
    /// Creates minimal `VideoParams` with `width`, `height`, and `frame_rate`.
    ///
    /// All other fields receive sensible defaults:
    /// - `pixel_format` = [`PixelFormat::Yuv420p`]
    /// - `color_space` = [`ColorSpace::Bt709`]
    /// - `bit_depth` = 8
    #[must_use]
    pub fn new(width: u32, height: u32, frame_rate: Rational) -> Self {
        Self {
            width,
            height,
            frame_rate,
            pixel_format: PixelFormat::Yuv420p,
            color_space: ColorSpace::Bt709,
            chroma_location: ChromaLocation::Left,
            display_aspect_ratio: None,
            bit_depth: 8,
            bitrate_bps: None,
        }
    }

    /// Returns the storage aspect ratio `width / height` as a `Rational`.
    #[must_use]
    pub fn storage_aspect_ratio(&self) -> Rational {
        Rational::new(self.width as i64, self.height as i64)
    }

    /// Returns the display aspect ratio, falling back to the storage aspect
    /// ratio when not explicitly set.
    #[must_use]
    pub fn effective_aspect_ratio(&self) -> Rational {
        self.display_aspect_ratio
            .unwrap_or_else(|| self.storage_aspect_ratio())
    }

    /// Returns `true` if the frame dimensions are valid (both non-zero).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0 && self.frame_rate.den > 0
    }

    /// Returns the total number of pixels per frame.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Returns the frame rate as a floating-point number.
    #[must_use]
    pub fn fps(&self) -> f64 {
        self.frame_rate.to_f64()
    }

    /// Builder-style setter for pixel format.
    #[must_use]
    pub fn with_pixel_format(mut self, fmt: PixelFormat) -> Self {
        self.pixel_format = fmt;
        self
    }

    /// Builder-style setter for colour space.
    #[must_use]
    pub fn with_color_space(mut self, cs: ColorSpace) -> Self {
        self.color_space = cs;
        self
    }

    /// Builder-style setter for bit depth.
    #[must_use]
    pub fn with_bit_depth(mut self, depth: u8) -> Self {
        self.bit_depth = depth;
        self
    }

    /// Builder-style setter for bitrate.
    #[must_use]
    pub fn with_bitrate(mut self, bps: u64) -> Self {
        self.bitrate_bps = Some(bps);
        self
    }
}

// ---------------------------------------------------------------------------
// AudioParams
// ---------------------------------------------------------------------------

/// Audio stream encoding / decoding parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioParams {
    /// Sample rate in Hz (e.g. 48_000 for broadcast audio).
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo, 6 = 5.1, …).
    pub channels: u16,
    /// Sample format (default: [`SampleFormat::F32`]).
    pub sample_format: SampleFormat,
    /// Target or measured peak bitrate in bits per second, if known.
    pub bitrate_bps: Option<u64>,
    /// Frame size (samples per channel per frame), if fixed by the codec.
    ///
    /// For example, Opus uses 20 ms frames at 48 kHz → 960 samples.
    pub frame_size: Option<u32>,
    /// Normalisation loudness target in LUFS, if known (e.g. −23 LUFS for EBU R128).
    pub loudness_lufs: Option<f32>,
}

impl AudioParams {
    /// Creates minimal `AudioParams` with `sample_rate` and `channels`.
    ///
    /// Defaults: `sample_format` = [`SampleFormat::F32`], other fields `None`.
    #[must_use]
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        Self {
            sample_rate,
            channels,
            sample_format: SampleFormat::F32,
            bitrate_bps: None,
            frame_size: None,
            loudness_lufs: None,
        }
    }

    /// Returns `true` if the parameters are logically valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.sample_rate > 0 && self.channels > 0
    }

    /// Returns the duration of a single frame in seconds, or `None` if no
    /// fixed frame size is set.
    #[must_use]
    pub fn frame_duration_secs(&self) -> Option<f64> {
        self.frame_size
            .map(|fs| f64::from(fs) / f64::from(self.sample_rate))
    }

    /// Builder-style setter for sample format.
    #[must_use]
    pub fn with_sample_format(mut self, fmt: SampleFormat) -> Self {
        self.sample_format = fmt;
        self
    }

    /// Builder-style setter for bitrate.
    #[must_use]
    pub fn with_bitrate(mut self, bps: u64) -> Self {
        self.bitrate_bps = Some(bps);
        self
    }

    /// Builder-style setter for frame size.
    #[must_use]
    pub fn with_frame_size(mut self, samples: u32) -> Self {
        self.frame_size = Some(samples);
        self
    }

    /// Builder-style setter for loudness target.
    #[must_use]
    pub fn with_loudness(mut self, lufs: f32) -> Self {
        self.loudness_lufs = Some(lufs);
        self
    }
}

// ---------------------------------------------------------------------------
// CodecParamsInner
// ---------------------------------------------------------------------------

/// Inner payload of a [`CodecParams`] discriminated by media type.
#[derive(Debug, Clone, PartialEq)]
pub enum CodecParamsInner {
    /// Video codec parameters.
    Video(VideoParams),
    /// Audio codec parameters.
    Audio(AudioParams),
    /// Data / subtitle / muxed stream with no further type-specific fields.
    Data,
}

// ---------------------------------------------------------------------------
// CodecParams
// ---------------------------------------------------------------------------

/// Codec-agnostic parameter descriptor for a single elementary stream.
///
/// Combines a [`CodecId`] with either [`VideoParams`], [`AudioParams`], or a
/// bare `Data` marker for subtitle/attachment streams.
#[derive(Debug, Clone, PartialEq)]
pub struct CodecParams {
    /// The codec used to encode this stream.
    pub codec_id: CodecId,
    /// Type-specific parameters.
    pub inner: CodecParamsInner,
    /// Optional stream index within the container (0-based).
    pub stream_index: Option<u32>,
    /// Optional stream language tag (BCP-47, e.g. `"en"`, `"ja"`).
    pub language: Option<String>,
}

impl CodecParams {
    /// Creates a `CodecParams` for a video stream.
    #[must_use]
    pub fn video(codec_id: CodecId, params: VideoParams) -> Self {
        Self {
            codec_id,
            inner: CodecParamsInner::Video(params),
            stream_index: None,
            language: None,
        }
    }

    /// Creates a `CodecParams` for an audio stream.
    #[must_use]
    pub fn audio(codec_id: CodecId, params: AudioParams) -> Self {
        Self {
            codec_id,
            inner: CodecParamsInner::Audio(params),
            stream_index: None,
            language: None,
        }
    }

    /// Creates a `CodecParams` for a data/subtitle stream.
    #[must_use]
    pub fn data(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            inner: CodecParamsInner::Data,
            stream_index: None,
            language: None,
        }
    }

    /// Returns `true` if these are video codec parameters.
    #[must_use]
    pub fn is_video(&self) -> bool {
        matches!(self.inner, CodecParamsInner::Video(_))
    }

    /// Returns `true` if these are audio codec parameters.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(self.inner, CodecParamsInner::Audio(_))
    }

    /// Returns a reference to the [`VideoParams`], if present.
    #[must_use]
    pub fn video_params(&self) -> Option<&VideoParams> {
        if let CodecParamsInner::Video(ref v) = self.inner {
            Some(v)
        } else {
            None
        }
    }

    /// Returns a reference to the [`AudioParams`], if present.
    #[must_use]
    pub fn audio_params(&self) -> Option<&AudioParams> {
        if let CodecParamsInner::Audio(ref a) = self.inner {
            Some(a)
        } else {
            None
        }
    }

    /// Builder-style setter for stream index.
    #[must_use]
    pub fn with_stream_index(mut self, index: u32) -> Self {
        self.stream_index = Some(index);
        self
    }

    /// Builder-style setter for language tag.
    #[must_use]
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }
}

// ---------------------------------------------------------------------------
// CodecParamSet
// ---------------------------------------------------------------------------

/// A collection of [`CodecParams`] indexed by stream index, representing
/// all streams present in a container.
#[derive(Debug, Default, Clone)]
pub struct CodecParamSet {
    params: Vec<CodecParams>,
}

impl CodecParamSet {
    /// Creates an empty `CodecParamSet`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a [`CodecParams`] entry to the set.
    pub fn add(&mut self, p: CodecParams) {
        self.params.push(p);
    }

    /// Returns the number of streams.
    #[must_use]
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Returns `true` if the set contains no streams.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Returns a reference to the params at position `index`, or `None`.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&CodecParams> {
        self.params.get(index)
    }

    /// Returns an iterator over all params.
    pub fn iter(&self) -> impl Iterator<Item = &CodecParams> {
        self.params.iter()
    }

    /// Returns an iterator over video stream params.
    pub fn video_streams(&self) -> impl Iterator<Item = &CodecParams> {
        self.params.iter().filter(|p| p.is_video())
    }

    /// Returns an iterator over audio stream params.
    pub fn audio_streams(&self) -> impl Iterator<Item = &CodecParams> {
        self.params.iter().filter(|p| p.is_audio())
    }

    /// Returns the first video stream, if any.
    #[must_use]
    pub fn first_video(&self) -> Option<&CodecParams> {
        self.params.iter().find(|p| p.is_video())
    }

    /// Returns the first audio stream, if any.
    #[must_use]
    pub fn first_audio(&self) -> Option<&CodecParams> {
        self.params.iter().find(|p| p.is_audio())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CodecId, PixelFormat, Rational, SampleFormat};

    // --- ColorSpace / ChromaLocation ---

    #[test]
    fn test_color_space_display() {
        assert_eq!(format!("{}", ColorSpace::Bt709), "bt709");
        assert_eq!(format!("{}", ColorSpace::Bt2020), "bt2020");
        assert_eq!(format!("{}", ColorSpace::Rgb), "rgb");
    }

    #[test]
    fn test_color_space_default() {
        let cs = ColorSpace::default();
        assert_eq!(cs, ColorSpace::Bt709);
    }

    // --- VideoParams ---

    #[test]
    fn test_video_params_basic() {
        let vp = VideoParams::new(1920, 1080, Rational::new(30, 1));
        assert_eq!(vp.width, 1920);
        assert_eq!(vp.height, 1080);
        assert!(vp.is_valid());
        assert_eq!(vp.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_video_params_fps() {
        let vp = VideoParams::new(1280, 720, Rational::new(30000, 1001));
        let fps = vp.fps();
        assert!((fps - 29.97).abs() < 0.01, "fps={fps}");
    }

    #[test]
    fn test_video_params_aspect_ratio_fallback() {
        let vp = VideoParams::new(1920, 1080, Rational::new(30, 1));
        let sar = vp.storage_aspect_ratio();
        let ear = vp.effective_aspect_ratio();
        assert_eq!(sar, ear);
    }

    #[test]
    fn test_video_params_display_aspect_override() {
        let mut vp = VideoParams::new(720, 576, Rational::new(25, 1));
        vp.display_aspect_ratio = Some(Rational::new(16, 9));
        let ear = vp.effective_aspect_ratio();
        assert_eq!(ear, Rational::new(16, 9));
    }

    #[test]
    fn test_video_params_builder_chain() {
        let vp = VideoParams::new(3840, 2160, Rational::new(60, 1))
            .with_pixel_format(PixelFormat::Yuv420p)
            .with_color_space(ColorSpace::Bt2020)
            .with_bit_depth(10)
            .with_bitrate(20_000_000);
        assert_eq!(vp.bit_depth, 10);
        assert_eq!(vp.color_space, ColorSpace::Bt2020);
        assert_eq!(vp.bitrate_bps, Some(20_000_000));
    }

    // --- AudioParams ---

    #[test]
    fn test_audio_params_basic() {
        let ap = AudioParams::new(48_000, 2);
        assert_eq!(ap.sample_rate, 48_000);
        assert_eq!(ap.channels, 2);
        assert!(ap.is_valid());
    }

    #[test]
    fn test_audio_params_frame_duration() {
        let ap = AudioParams::new(48_000, 2).with_frame_size(960);
        let dur = ap.frame_duration_secs().expect("frame size set");
        assert!((dur - 0.02).abs() < 1e-9, "expected 20ms, got {dur}");
    }

    #[test]
    fn test_audio_params_builder_chain() {
        let ap = AudioParams::new(44_100, 1)
            .with_sample_format(SampleFormat::S16)
            .with_bitrate(128_000)
            .with_loudness(-23.0);
        assert_eq!(ap.sample_format, SampleFormat::S16);
        assert_eq!(ap.bitrate_bps, Some(128_000));
        assert!((ap.loudness_lufs.unwrap_or(0.0) - (-23.0_f32)).abs() < 1e-5);
    }

    // --- CodecParams ---

    #[test]
    fn test_codec_params_video() {
        let cp = CodecParams::video(
            CodecId::Av1,
            VideoParams::new(1920, 1080, Rational::new(30, 1)),
        );
        assert!(cp.is_video());
        assert!(!cp.is_audio());
        assert_eq!(cp.video_params().map(|v| v.width), Some(1920));
        assert!(cp.audio_params().is_none());
    }

    #[test]
    fn test_codec_params_audio() {
        let cp = CodecParams::audio(CodecId::Opus, AudioParams::new(48_000, 2));
        assert!(cp.is_audio());
        assert!(!cp.is_video());
        assert_eq!(cp.audio_params().map(|a| a.channels), Some(2));
        assert!(cp.video_params().is_none());
    }

    #[test]
    fn test_codec_params_data() {
        let cp = CodecParams::data(CodecId::WebVtt);
        assert!(!cp.is_video());
        assert!(!cp.is_audio());
    }

    #[test]
    fn test_codec_params_language_and_stream_index() {
        let cp = CodecParams::audio(CodecId::Vorbis, AudioParams::new(44_100, 2))
            .with_stream_index(1)
            .with_language("ja");
        assert_eq!(cp.stream_index, Some(1));
        assert_eq!(cp.language.as_deref(), Some("ja"));
    }

    // --- CodecParamSet ---

    #[test]
    fn test_codec_param_set_push_and_query() {
        let mut set = CodecParamSet::new();
        assert!(set.is_empty());

        set.add(CodecParams::video(
            CodecId::Vp9,
            VideoParams::new(1280, 720, Rational::new(24, 1)),
        ));
        set.add(CodecParams::audio(
            CodecId::Opus,
            AudioParams::new(48_000, 2),
        ));
        set.add(CodecParams::audio(
            CodecId::Flac,
            AudioParams::new(96_000, 2),
        ));

        assert_eq!(set.len(), 3);
        assert_eq!(set.video_streams().count(), 1);
        assert_eq!(set.audio_streams().count(), 2);
        assert!(set.first_video().is_some());
        assert!(set.first_audio().is_some());
    }
}
