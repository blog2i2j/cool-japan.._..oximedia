//! Metadata structures describing audio files and streams.
#![allow(dead_code)]

/// Technical description of an audio codec.
#[derive(Debug, Clone)]
pub struct AudioCodecInfo {
    /// Codec identifier string (e.g. "flac", "opus", "pcm_s24le").
    pub codec_id: String,
    /// Human-readable name.
    pub name: String,
    /// Bit depth (e.g. 16, 24, 32).
    pub bit_depth: u8,
    /// Whether the codec is lossless.
    lossless: bool,
    /// Whether the codec is lossy.
    lossy: bool,
}

impl AudioCodecInfo {
    /// Create a new codec info record.
    pub fn new(
        codec_id: impl Into<String>,
        name: impl Into<String>,
        bit_depth: u8,
        lossless: bool,
    ) -> Self {
        Self {
            codec_id: codec_id.into(),
            name: name.into(),
            bit_depth,
            lossless,
            lossy: !lossless,
        }
    }

    /// Returns `true` if the codec preserves all audio information exactly.
    pub fn is_lossless(&self) -> bool {
        self.lossless
    }

    /// Returns `true` if the codec discards audio information.
    pub fn is_lossy(&self) -> bool {
        self.lossy
    }

    /// Returns the codec identifier string.
    pub fn codec_id(&self) -> &str {
        &self.codec_id
    }

    /// Convenience constructor for FLAC.
    pub fn flac() -> Self {
        Self::new("flac", "FLAC", 24, true)
    }

    /// Convenience constructor for Opus.
    pub fn opus() -> Self {
        Self::new("opus", "Opus", 32, false)
    }

    /// Convenience constructor for MP3.
    pub fn mp3() -> Self {
        Self::new("mp3", "MP3", 16, false)
    }
}

/// Metadata for a single audio stream within a container.
#[derive(Debug, Clone)]
pub struct AudioStreamMeta {
    /// Zero-based stream index within the container.
    pub stream_index: u32,
    /// Codec details.
    pub codec: AudioCodecInfo,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Bit rate in bits per second (0 = variable / unknown).
    pub bitrate_bps: u64,
    /// Total number of samples (0 = unknown).
    pub total_samples: u64,
    /// Language tag (e.g. "en", "fr").
    pub language: Option<String>,
}

impl AudioStreamMeta {
    /// Create a new stream metadata record.
    pub fn new(
        stream_index: u32,
        codec: AudioCodecInfo,
        sample_rate: u32,
        channels: u8,
        total_samples: u64,
    ) -> Self {
        Self {
            stream_index,
            codec,
            sample_rate,
            channels,
            bitrate_bps: 0,
            total_samples,
            language: None,
        }
    }

    /// Duration of this stream in milliseconds.
    ///
    /// Returns 0 when `total_samples` or `sample_rate` is 0.
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_ms(&self) -> u64 {
        if self.sample_rate == 0 || self.total_samples == 0 {
            return 0;
        }
        // Compute via integer arithmetic to avoid floating-point rounding.
        self.total_samples * 1000 / u64::from(self.sample_rate)
    }

    /// Returns `true` when the stream language is known.
    pub fn has_language(&self) -> bool {
        self.language.is_some()
    }

    /// Set the language tag.
    pub fn set_language(&mut self, lang: impl Into<String>) {
        self.language = Some(lang.into());
    }

    /// Returns the language tag if set.
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }
}

/// Top-level metadata about an audio file.
#[derive(Debug, Clone)]
pub struct AudioFileInfo {
    /// Originating file path.
    pub path: String,
    /// Container format (e.g. "matroska", "ogg", "wave").
    pub container: String,
    /// All audio streams within the file.
    pub streams: Vec<AudioStreamMeta>,
    /// File size in bytes (0 = unknown).
    pub file_size_bytes: u64,
}

impl AudioFileInfo {
    /// Create a new file info record.
    pub fn new(
        path: impl Into<String>,
        container: impl Into<String>,
        streams: Vec<AudioStreamMeta>,
    ) -> Self {
        Self {
            path: path.into(),
            container: container.into(),
            streams,
            file_size_bytes: 0,
        }
    }

    /// Returns the first (primary) audio stream, if any.
    pub fn primary_stream(&self) -> Option<&AudioStreamMeta> {
        self.streams.first()
    }

    /// Total duration across all streams (uses the longest stream).
    pub fn total_duration_ms(&self) -> u64 {
        self.streams
            .iter()
            .map(|s| s.duration_ms())
            .max()
            .unwrap_or(0)
    }

    /// Number of audio streams in the file.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Returns `true` if all streams use lossless codecs.
    pub fn all_lossless(&self) -> bool {
        !self.streams.is_empty() && self.streams.iter().all(|s| s.codec.is_lossless())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- AudioCodecInfo ---

    #[test]
    fn test_flac_is_lossless() {
        let c = AudioCodecInfo::flac();
        assert!(c.is_lossless());
        assert!(!c.is_lossy());
    }

    #[test]
    fn test_opus_is_lossy() {
        let c = AudioCodecInfo::opus();
        assert!(c.is_lossy());
        assert!(!c.is_lossless());
    }

    #[test]
    fn test_mp3_is_lossy() {
        assert!(AudioCodecInfo::mp3().is_lossy());
    }

    #[test]
    fn test_codec_id_accessor() {
        let c = AudioCodecInfo::flac();
        assert_eq!(c.codec_id(), "flac");
    }

    #[test]
    fn test_custom_codec_info() {
        let c = AudioCodecInfo::new("pcm_s24le", "PCM 24-bit LE", 24, true);
        assert_eq!(c.bit_depth, 24);
        assert!(c.is_lossless());
    }

    // --- AudioStreamMeta ---

    #[test]
    fn test_duration_ms_correct() {
        let meta = AudioStreamMeta::new(0, AudioCodecInfo::flac(), 48_000, 2, 96_000);
        assert_eq!(meta.duration_ms(), 2_000); // 2 seconds
    }

    #[test]
    fn test_duration_ms_zero_samples() {
        let meta = AudioStreamMeta::new(0, AudioCodecInfo::flac(), 48_000, 2, 0);
        assert_eq!(meta.duration_ms(), 0);
    }

    #[test]
    fn test_duration_ms_zero_sample_rate() {
        let meta = AudioStreamMeta::new(0, AudioCodecInfo::flac(), 0, 2, 48_000);
        assert_eq!(meta.duration_ms(), 0);
    }

    #[test]
    fn test_language_initially_none() {
        let meta = AudioStreamMeta::new(0, AudioCodecInfo::opus(), 48_000, 2, 1000);
        assert!(!meta.has_language());
    }

    #[test]
    fn test_set_language() {
        let mut meta = AudioStreamMeta::new(0, AudioCodecInfo::opus(), 48_000, 2, 1000);
        meta.set_language("en");
        assert!(meta.has_language());
        assert_eq!(meta.language(), Some("en"));
    }

    #[test]
    fn test_stream_channels() {
        let meta = AudioStreamMeta::new(1, AudioCodecInfo::mp3(), 44_100, 2, 44_100);
        assert_eq!(meta.channels, 2);
    }

    // --- AudioFileInfo ---

    #[test]
    fn test_primary_stream_returns_first() {
        let s1 = AudioStreamMeta::new(0, AudioCodecInfo::flac(), 48_000, 2, 48_000);
        let s2 = AudioStreamMeta::new(1, AudioCodecInfo::opus(), 48_000, 2, 48_000);
        let info = AudioFileInfo::new("file.mkv", "matroska", vec![s1, s2]);
        assert_eq!(
            info.primary_stream().expect("should succeed").stream_index,
            0
        );
    }

    #[test]
    fn test_primary_stream_empty_returns_none() {
        let info = AudioFileInfo::new("file.mkv", "matroska", vec![]);
        assert!(info.primary_stream().is_none());
    }

    #[test]
    fn test_total_duration_ms_max_of_streams() {
        let s1 = AudioStreamMeta::new(0, AudioCodecInfo::flac(), 48_000, 2, 48_000); // 1 s
        let s2 = AudioStreamMeta::new(1, AudioCodecInfo::flac(), 48_000, 2, 96_000); // 2 s
        let info = AudioFileInfo::new("f.wav", "wave", vec![s1, s2]);
        assert_eq!(info.total_duration_ms(), 2_000);
    }

    #[test]
    fn test_stream_count() {
        let streams: Vec<_> = (0..3)
            .map(|i| AudioStreamMeta::new(i, AudioCodecInfo::flac(), 48_000, 2, 1000))
            .collect();
        let info = AudioFileInfo::new("f.flac", "flac", streams);
        assert_eq!(info.stream_count(), 3);
    }

    #[test]
    fn test_all_lossless_true() {
        let streams = vec![
            AudioStreamMeta::new(0, AudioCodecInfo::flac(), 48_000, 2, 1000),
            AudioStreamMeta::new(1, AudioCodecInfo::flac(), 48_000, 2, 1000),
        ];
        let info = AudioFileInfo::new("f.flac", "flac", streams);
        assert!(info.all_lossless());
    }

    #[test]
    fn test_all_lossless_false_with_lossy() {
        let streams = vec![
            AudioStreamMeta::new(0, AudioCodecInfo::flac(), 48_000, 2, 1000),
            AudioStreamMeta::new(1, AudioCodecInfo::opus(), 48_000, 2, 1000),
        ];
        let info = AudioFileInfo::new("f.mkv", "matroska", streams);
        assert!(!info.all_lossless());
    }

    #[test]
    fn test_all_lossless_empty_streams_false() {
        let info = AudioFileInfo::new("f.flac", "flac", vec![]);
        assert!(!info.all_lossless());
    }
}
