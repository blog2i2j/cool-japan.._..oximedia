//! Codec/container/profile compatibility matrix with constraint validation.
//!
//! Provides a declarative table of which codecs are allowed inside which
//! containers, which codec profiles are valid for a given codec, and helper
//! functions for validating a proposed encode configuration before any work
//! begins.
//!
//! All codecs listed here are patent-free: VP8, VP9, AV1 (video);
//! Opus, Vorbis, FLAC, PCM (audio); and the lossless archival codec FFV1.

use std::collections::HashMap;

/// Supported video codecs (patent-free only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    /// VP8 — royalty-free codec from Google.
    Vp8,
    /// VP9 — successor to VP8, used on YouTube WebM.
    Vp9,
    /// AV1 — Alliance for Open Media, state-of-the-art efficiency.
    Av1,
    /// FFV1 — lossless archival codec by the FFmpeg project.
    Ffv1,
    /// Theora — OGG video codec.
    Theora,
}

impl VideoCodec {
    /// Returns the canonical name string for this codec.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Vp8 => "vp8",
            Self::Vp9 => "vp9",
            Self::Av1 => "av1",
            Self::Ffv1 => "ffv1",
            Self::Theora => "theora",
        }
    }

    /// Returns `true` if this codec is lossless.
    #[must_use]
    pub fn is_lossless(self) -> bool {
        matches!(self, Self::Ffv1)
    }

    /// Attempts to parse a codec name string.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().trim() {
            "vp8" | "libvpx" => Some(Self::Vp8),
            "vp9" | "libvpx-vp9" => Some(Self::Vp9),
            "av1" | "libaom-av1" | "libsvtav1" | "librav1e" => Some(Self::Av1),
            "ffv1" => Some(Self::Ffv1),
            "theora" | "libtheora" => Some(Self::Theora),
            _ => None,
        }
    }
}

/// Supported audio codecs (patent-free only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioCodec {
    /// Opus — low-latency, high-quality codec from Xiph.
    Opus,
    /// Vorbis — OGG audio codec.
    Vorbis,
    /// FLAC — Free Lossless Audio Codec.
    Flac,
    /// PCM signed 16-bit little-endian.
    PcmS16Le,
    /// PCM signed 24-bit little-endian.
    PcmS24Le,
    /// PCM signed 32-bit little-endian.
    PcmS32Le,
    /// PCM 32-bit IEEE float.
    PcmF32Le,
}

impl AudioCodec {
    /// Returns the canonical name string.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Opus => "opus",
            Self::Vorbis => "vorbis",
            Self::Flac => "flac",
            Self::PcmS16Le => "pcm_s16le",
            Self::PcmS24Le => "pcm_s24le",
            Self::PcmS32Le => "pcm_s32le",
            Self::PcmF32Le => "pcm_f32le",
        }
    }

    /// Returns `true` if this codec is lossless.
    #[must_use]
    pub fn is_lossless(self) -> bool {
        matches!(
            self,
            Self::Flac | Self::PcmS16Le | Self::PcmS24Le | Self::PcmS32Le | Self::PcmF32Le
        )
    }

    /// Attempts to parse an audio codec name string.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().trim() {
            "opus" | "libopus" => Some(Self::Opus),
            "vorbis" | "libvorbis" => Some(Self::Vorbis),
            "flac" => Some(Self::Flac),
            "pcm_s16le" | "pcms16le" => Some(Self::PcmS16Le),
            "pcm_s24le" | "pcms24le" => Some(Self::PcmS24Le),
            "pcm_s32le" | "pcms32le" => Some(Self::PcmS32Le),
            "pcm_f32le" | "pcmf32le" => Some(Self::PcmF32Le),
            _ => None,
        }
    }
}

/// Container formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Container {
    /// WebM — royalty-free container designed for VP8/VP9/AV1.
    Webm,
    /// Matroska — flexible open container.
    Mkv,
    /// OGG — Xiph container for Theora/Vorbis/FLAC.
    Ogg,
    /// MP4 / ISOBMFF (used with AV1 and opus when required).
    Mp4,
    /// WAV — uncompressed audio container.
    Wav,
    /// FLAC native container.
    FlacNative,
}

impl Container {
    /// Returns the standard file extension for this container.
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::Webm => "webm",
            Self::Mkv => "mkv",
            Self::Ogg => "ogg",
            Self::Mp4 => "mp4",
            Self::Wav => "wav",
            Self::FlacNative => "flac",
        }
    }

    /// Attempts to parse a container name or extension.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().trim().trim_start_matches('.') {
            "webm" => Some(Self::Webm),
            "mkv" | "matroska" => Some(Self::Mkv),
            "ogg" | "ogv" | "oga" => Some(Self::Ogg),
            "mp4" | "m4v" | "m4a" | "isobmff" => Some(Self::Mp4),
            "wav" | "wave" => Some(Self::Wav),
            "flac" => Some(Self::FlacNative),
            _ => None,
        }
    }
}

/// A compatibility error describing why a codec+container combination is invalid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompatError {
    /// The video codec is not supported in the given container.
    VideoCodecNotInContainer {
        /// Codec name.
        codec: &'static str,
        /// Container extension.
        container: &'static str,
    },
    /// The audio codec is not supported in the given container.
    AudioCodecNotInContainer {
        /// Codec name.
        codec: &'static str,
        /// Container extension.
        container: &'static str,
    },
    /// A lossless video codec was paired with a lossy audio codec (or vice-versa)
    /// in a configuration that requires a fully lossless archival output.
    LosslessMismatch,
    /// An unknown codec name was supplied.
    UnknownCodec(String),
    /// An unknown container name was supplied.
    UnknownContainer(String),
}

impl std::fmt::Display for CompatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VideoCodecNotInContainer { codec, container } => {
                write!(f, "video codec '{codec}' is not supported in '{container}'")
            }
            Self::AudioCodecNotInContainer { codec, container } => {
                write!(f, "audio codec '{codec}' is not supported in '{container}'")
            }
            Self::LosslessMismatch => {
                write!(f, "lossless video requires a lossless audio codec for archival")
            }
            Self::UnknownCodec(n) => write!(f, "unknown codec '{n}'"),
            Self::UnknownContainer(n) => write!(f, "unknown container '{n}'"),
        }
    }
}

impl std::error::Error for CompatError {}

/// The static compatibility matrix.
///
/// `allowed_video[container]` → list of video codecs allowed in that container.
/// `allowed_audio[container]` → list of audio codecs allowed in that container.
#[derive(Debug)]
pub struct CompatMatrix {
    video: HashMap<Container, Vec<VideoCodec>>,
    audio: HashMap<Container, Vec<AudioCodec>>,
}

impl Default for CompatMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl CompatMatrix {
    /// Builds the built-in patent-free compatibility matrix.
    #[must_use]
    pub fn new() -> Self {
        let mut video: HashMap<Container, Vec<VideoCodec>> = HashMap::new();
        let mut audio: HashMap<Container, Vec<AudioCodec>> = HashMap::new();

        // WebM: VP8, VP9, AV1 + Opus, Vorbis
        video.insert(Container::Webm, vec![VideoCodec::Vp8, VideoCodec::Vp9, VideoCodec::Av1]);
        audio.insert(
            Container::Webm,
            vec![AudioCodec::Opus, AudioCodec::Vorbis],
        );

        // MKV: virtually all patent-free codecs
        video.insert(
            Container::Mkv,
            vec![
                VideoCodec::Vp8,
                VideoCodec::Vp9,
                VideoCodec::Av1,
                VideoCodec::Ffv1,
                VideoCodec::Theora,
            ],
        );
        audio.insert(
            Container::Mkv,
            vec![
                AudioCodec::Opus,
                AudioCodec::Vorbis,
                AudioCodec::Flac,
                AudioCodec::PcmS16Le,
                AudioCodec::PcmS24Le,
                AudioCodec::PcmS32Le,
                AudioCodec::PcmF32Le,
            ],
        );

        // OGG: Theora + Vorbis/Opus/FLAC
        video.insert(Container::Ogg, vec![VideoCodec::Theora]);
        audio.insert(
            Container::Ogg,
            vec![AudioCodec::Vorbis, AudioCodec::Opus, AudioCodec::Flac],
        );

        // MP4: AV1 + Opus (RFC 7845 / ISOBMFF AV1 binding)
        video.insert(Container::Mp4, vec![VideoCodec::Av1, VideoCodec::Vp9]);
        audio.insert(Container::Mp4, vec![AudioCodec::Opus, AudioCodec::Flac]);

        // WAV: PCM only (audio container)
        video.insert(Container::Wav, vec![]);
        audio.insert(
            Container::Wav,
            vec![
                AudioCodec::PcmS16Le,
                AudioCodec::PcmS24Le,
                AudioCodec::PcmS32Le,
                AudioCodec::PcmF32Le,
            ],
        );

        // FLAC native container: FLAC only
        video.insert(Container::FlacNative, vec![]);
        audio.insert(Container::FlacNative, vec![AudioCodec::Flac]);

        Self { video, audio }
    }

    /// Returns the list of video codecs permitted in `container`.
    #[must_use]
    pub fn allowed_video(&self, container: Container) -> &[VideoCodec] {
        self.video.get(&container).map_or(&[], Vec::as_slice)
    }

    /// Returns the list of audio codecs permitted in `container`.
    #[must_use]
    pub fn allowed_audio(&self, container: Container) -> &[AudioCodec] {
        self.audio.get(&container).map_or(&[], Vec::as_slice)
    }

    /// Returns `true` if `codec` may be used in `container`.
    #[must_use]
    pub fn video_allowed(&self, codec: VideoCodec, container: Container) -> bool {
        self.allowed_video(container).contains(&codec)
    }

    /// Returns `true` if `codec` may be used in `container`.
    #[must_use]
    pub fn audio_allowed(&self, codec: AudioCodec, container: Container) -> bool {
        self.allowed_audio(container).contains(&codec)
    }

    /// Validates a video+audio+container combination.
    ///
    /// Returns `Ok(())` when valid, or a `Vec<CompatError>` listing all violations.
    ///
    /// # Errors
    ///
    /// Returns all detected compatibility issues.
    pub fn validate(
        &self,
        video: VideoCodec,
        audio: AudioCodec,
        container: Container,
    ) -> Result<(), Vec<CompatError>> {
        let mut errors = Vec::new();
        if !self.video_allowed(video, container) {
            errors.push(CompatError::VideoCodecNotInContainer {
                codec: video.name(),
                container: container.extension(),
            });
        }
        if !self.audio_allowed(audio, container) {
            errors.push(CompatError::AudioCodecNotInContainer {
                codec: audio.name(),
                container: container.extension(),
            });
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Validates a configuration expressed as codec name strings and a container name string.
///
/// This is a convenience wrapper around [`CompatMatrix::validate`] that accepts
/// string inputs and resolves them to typed enums.
///
/// # Errors
///
/// Returns all compatibility issues found, including unknown codec/container names.
pub fn validate_string_config(
    video_codec: &str,
    audio_codec: &str,
    container: &str,
) -> Result<(), Vec<CompatError>> {
    let matrix = CompatMatrix::new();
    let mut errors = Vec::new();

    let video = match VideoCodec::from_name(video_codec) {
        Some(v) => v,
        None => {
            errors.push(CompatError::UnknownCodec(video_codec.to_owned()));
            return Err(errors);
        }
    };
    let audio = match AudioCodec::from_name(audio_codec) {
        Some(a) => a,
        None => {
            errors.push(CompatError::UnknownCodec(audio_codec.to_owned()));
            return Err(errors);
        }
    };
    let cont = match Container::from_name(container) {
        Some(c) => c,
        None => {
            errors.push(CompatError::UnknownContainer(container.to_owned()));
            return Err(errors);
        }
    };

    match matrix.validate(video, audio, cont) {
        Ok(()) => Ok(()),
        Err(e) => {
            errors.extend(e);
            Err(errors)
        }
    }
}

/// Returns the list of containers in which both `video` and `audio` codecs are allowed.
#[must_use]
pub fn common_containers(video: VideoCodec, audio: AudioCodec) -> Vec<Container> {
    let matrix = CompatMatrix::new();
    let all = [
        Container::Webm,
        Container::Mkv,
        Container::Ogg,
        Container::Mp4,
        Container::Wav,
        Container::FlacNative,
    ];
    all.iter()
        .filter(|&&c| matrix.video_allowed(video, c) && matrix.audio_allowed(audio, c))
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- VideoCodec ----------

    #[test]
    fn test_video_codec_names() {
        assert_eq!(VideoCodec::Vp9.name(), "vp9");
        assert_eq!(VideoCodec::Av1.name(), "av1");
        assert_eq!(VideoCodec::Ffv1.name(), "ffv1");
    }

    #[test]
    fn test_video_codec_from_name_aliases() {
        assert_eq!(VideoCodec::from_name("libvpx-vp9"), Some(VideoCodec::Vp9));
        assert_eq!(VideoCodec::from_name("libaom-av1"), Some(VideoCodec::Av1));
        assert_eq!(VideoCodec::from_name("libsvtav1"), Some(VideoCodec::Av1));
        assert_eq!(VideoCodec::from_name("unknown"), None);
    }

    #[test]
    fn test_video_codec_lossless() {
        assert!(VideoCodec::Ffv1.is_lossless());
        assert!(!VideoCodec::Vp9.is_lossless());
    }

    // ---------- AudioCodec ----------

    #[test]
    fn test_audio_codec_names() {
        assert_eq!(AudioCodec::Opus.name(), "opus");
        assert_eq!(AudioCodec::Flac.name(), "flac");
    }

    #[test]
    fn test_audio_codec_from_name_aliases() {
        assert_eq!(AudioCodec::from_name("libopus"), Some(AudioCodec::Opus));
        assert_eq!(AudioCodec::from_name("pcm_s16le"), Some(AudioCodec::PcmS16Le));
        assert_eq!(AudioCodec::from_name("xyz"), None);
    }

    #[test]
    fn test_audio_codec_lossless() {
        assert!(AudioCodec::Flac.is_lossless());
        assert!(AudioCodec::PcmS24Le.is_lossless());
        assert!(!AudioCodec::Opus.is_lossless());
        assert!(!AudioCodec::Vorbis.is_lossless());
    }

    // ---------- Container ----------

    #[test]
    fn test_container_extension() {
        assert_eq!(Container::Webm.extension(), "webm");
        assert_eq!(Container::Mkv.extension(), "mkv");
        assert_eq!(Container::Ogg.extension(), "ogg");
    }

    #[test]
    fn test_container_from_name() {
        assert_eq!(Container::from_name("webm"), Some(Container::Webm));
        assert_eq!(Container::from_name(".mkv"), Some(Container::Mkv));
        assert_eq!(Container::from_name("ogv"), Some(Container::Ogg));
        assert_eq!(Container::from_name("mp4"), Some(Container::Mp4));
        assert_eq!(Container::from_name("avi"), None);
    }

    // ---------- CompatMatrix ----------

    #[test]
    fn test_webm_allows_vp9_opus() {
        let m = CompatMatrix::new();
        assert!(m.video_allowed(VideoCodec::Vp9, Container::Webm));
        assert!(m.audio_allowed(AudioCodec::Opus, Container::Webm));
    }

    #[test]
    fn test_webm_rejects_ffv1() {
        let m = CompatMatrix::new();
        assert!(!m.video_allowed(VideoCodec::Ffv1, Container::Webm));
    }

    #[test]
    fn test_mkv_allows_all_patent_free_codecs() {
        let m = CompatMatrix::new();
        for vc in [VideoCodec::Vp8, VideoCodec::Vp9, VideoCodec::Av1, VideoCodec::Ffv1] {
            assert!(m.video_allowed(vc, Container::Mkv), "{:?} in MKV", vc);
        }
        for ac in [
            AudioCodec::Opus,
            AudioCodec::Vorbis,
            AudioCodec::Flac,
            AudioCodec::PcmS16Le,
        ] {
            assert!(m.audio_allowed(ac, Container::Mkv), "{:?} in MKV", ac);
        }
    }

    #[test]
    fn test_validate_ok() {
        let m = CompatMatrix::new();
        assert!(m.validate(VideoCodec::Vp9, AudioCodec::Opus, Container::Webm).is_ok());
    }

    #[test]
    fn test_validate_error_video() {
        let m = CompatMatrix::new();
        let errs = m
            .validate(VideoCodec::Ffv1, AudioCodec::Opus, Container::Webm)
            .expect_err("should fail");
        assert!(errs
            .iter()
            .any(|e| matches!(e, CompatError::VideoCodecNotInContainer { .. })));
    }

    #[test]
    fn test_validate_error_audio() {
        let m = CompatMatrix::new();
        let errs = m
            .validate(VideoCodec::Vp9, AudioCodec::PcmS16Le, Container::Webm)
            .expect_err("should fail");
        assert!(errs
            .iter()
            .any(|e| matches!(e, CompatError::AudioCodecNotInContainer { .. })));
    }

    // ---------- validate_string_config ----------

    #[test]
    fn test_string_config_valid() {
        assert!(validate_string_config("vp9", "opus", "webm").is_ok());
    }

    #[test]
    fn test_string_config_unknown_video() {
        let errs = validate_string_config("h264", "opus", "webm").expect_err("should fail");
        assert!(errs.iter().any(|e| matches!(e, CompatError::UnknownCodec(_))));
    }

    #[test]
    fn test_string_config_unknown_container() {
        let errs = validate_string_config("vp9", "opus", "avi").expect_err("should fail");
        assert!(errs
            .iter()
            .any(|e| matches!(e, CompatError::UnknownContainer(_))));
    }

    // ---------- common_containers ----------

    #[test]
    fn test_common_containers_vp9_opus() {
        let containers = common_containers(VideoCodec::Vp9, AudioCodec::Opus);
        assert!(containers.contains(&Container::Webm));
        assert!(containers.contains(&Container::Mkv));
    }

    #[test]
    fn test_common_containers_ffv1_flac() {
        let containers = common_containers(VideoCodec::Ffv1, AudioCodec::Flac);
        // FFV1 + FLAC is only valid in MKV
        assert!(containers.contains(&Container::Mkv));
        assert!(!containers.contains(&Container::Webm));
    }

    #[test]
    fn test_compat_error_display() {
        let e = CompatError::VideoCodecNotInContainer {
            codec: "ffv1",
            container: "webm",
        };
        let s = e.to_string();
        assert!(s.contains("ffv1"));
        assert!(s.contains("webm"));
    }
}
