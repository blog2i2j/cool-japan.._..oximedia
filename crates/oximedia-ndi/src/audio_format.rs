//! NDI audio format configuration and buffer management.
#![allow(dead_code)]

/// Supported NDI audio sample formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NdiSampleFormat {
    /// 16-bit signed integer, interleaved.
    I16,
    /// 32-bit signed integer, interleaved.
    I32,
    /// 32-bit IEEE 754 float, non-interleaved (planar).
    F32Planar,
}

impl NdiSampleFormat {
    /// Bytes consumed by a single sample of this format.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            NdiSampleFormat::I16 => 2,
            NdiSampleFormat::I32 => 4,
            NdiSampleFormat::F32Planar => 4,
        }
    }

    /// Returns `true` for planar (non-interleaved) layouts.
    pub fn is_planar(self) -> bool {
        matches!(self, NdiSampleFormat::F32Planar)
    }
}

/// Configuration for an NDI audio stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NdiAudioConfig {
    /// Sample rate in Hz (e.g. 48000).
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u16,
    /// Sample format.
    pub sample_format: NdiSampleFormat,
    /// Target samples per video frame.
    pub samples_per_frame_hint: u32,
}

impl NdiAudioConfig {
    /// Create a new `NdiAudioConfig`.
    pub fn new(
        sample_rate: u32,
        channels: u16,
        sample_format: NdiSampleFormat,
        samples_per_frame_hint: u32,
    ) -> Self {
        Self {
            sample_rate,
            channels,
            sample_format,
            samples_per_frame_hint,
        }
    }

    /// Standard 48 kHz stereo float-planar configuration (1080p30 hint).
    pub fn stereo_48k_30fps() -> Self {
        // 48000 / 30 = 1600 samples per frame
        Self::new(48000, 2, NdiSampleFormat::F32Planar, 1600)
    }

    /// Standard 48 kHz stereo 16-bit interleaved (1080p60 hint).
    pub fn stereo_48k_60fps() -> Self {
        // 48000 / 60 = 800 samples per frame
        Self::new(48000, 2, NdiSampleFormat::I16, 800)
    }

    /// Standard 48 kHz 5.1 surround float-planar (30 fps hint).
    pub fn surround_48k_30fps() -> Self {
        Self::new(48000, 6, NdiSampleFormat::F32Planar, 1600)
    }

    /// Compute the nominal samples per video frame given a frame rate.
    ///
    /// `fps_num` / `fps_den` is the video frame rate.
    pub fn samples_per_frame(&self, fps_num: u32, fps_den: u32) -> u32 {
        if fps_num == 0 {
            return 0;
        }
        self.sample_rate * fps_den / fps_num
    }

    /// Total bytes required for one audio frame using the hint.
    pub fn frame_bytes_hint(&self) -> usize {
        self.samples_per_frame_hint as usize
            * self.channels as usize
            * self.sample_format.bytes_per_sample()
    }
}

/// A buffer holding one NDI audio frame.
#[derive(Debug, Clone)]
pub struct NdiAudioBuffer {
    config: NdiAudioConfig,
    /// Raw sample data (layout depends on `config.sample_format`).
    data: Vec<u8>,
    /// Number of valid samples per channel stored in `data`.
    samples: u32,
    /// Presentation timestamp in 100-ns ticks.
    timestamp_100ns: i64,
}

impl NdiAudioBuffer {
    /// Create a new audio buffer.
    pub fn new(config: NdiAudioConfig, data: Vec<u8>, samples: u32, timestamp_100ns: i64) -> Self {
        Self {
            config,
            data,
            samples,
            timestamp_100ns,
        }
    }

    /// Allocate a zeroed buffer for `samples` samples per channel.
    pub fn zeroed(config: NdiAudioConfig, samples: u32) -> Self {
        let byte_count =
            samples as usize * config.channels as usize * config.sample_format.bytes_per_sample();
        Self::new(config, vec![0u8; byte_count], samples, 0)
    }

    /// Returns the raw byte data for a given channel index.
    ///
    /// For interleaved formats this returns all bytes (channel interleaving is the
    /// caller's responsibility). For planar formats it returns the plane for
    /// `channel`.
    pub fn channel_data(&self, channel: u16) -> Option<&[u8]> {
        if channel >= self.config.channels {
            return None;
        }
        let bps = self.config.sample_format.bytes_per_sample();
        if self.config.sample_format.is_planar() {
            let plane_bytes = self.samples as usize * bps;
            let start = channel as usize * plane_bytes;
            let end = start + plane_bytes;
            self.data.get(start..end)
        } else {
            // Interleaved: return the full buffer (caller de-interleaves).
            Some(&self.data)
        }
    }

    /// Number of samples per channel.
    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// Presentation timestamp in 100-nanosecond ticks.
    pub fn timestamp_100ns(&self) -> i64 {
        self.timestamp_100ns
    }

    /// Configuration this buffer was created with.
    pub fn config(&self) -> &NdiAudioConfig {
        &self.config
    }
}

/// Format descriptor for an NDI audio stream, combining config with stream-level metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NdiAudioFormat {
    /// Underlying configuration (sample rate, channels, etc.).
    pub config: NdiAudioConfig,
    /// Stride in bytes between audio frames (0 = tightly packed).
    pub frame_stride: u32,
}

impl NdiAudioFormat {
    /// Create an `NdiAudioFormat`.
    pub fn new(config: NdiAudioConfig, frame_stride: u32) -> Self {
        Self {
            config,
            frame_stride,
        }
    }

    /// Tightly-packed format (stride = 0).
    pub fn packed(config: NdiAudioConfig) -> Self {
        Self::new(config, 0)
    }

    /// Byte size of one audio frame (samples × channels × bytes-per-sample).
    pub fn frame_size_bytes(&self, fps_num: u32, fps_den: u32) -> usize {
        let samples = self.config.samples_per_frame(fps_num, fps_den);
        samples as usize
            * self.config.channels as usize
            * self.config.sample_format.bytes_per_sample()
    }

    /// Effective stride: if `frame_stride` is 0, returns `frame_size_bytes`.
    pub fn effective_stride(&self, fps_num: u32, fps_den: u32) -> usize {
        if self.frame_stride == 0 {
            self.frame_size_bytes(fps_num, fps_den)
        } else {
            self.frame_stride as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_format_bytes_i16() {
        assert_eq!(NdiSampleFormat::I16.bytes_per_sample(), 2);
    }

    #[test]
    fn test_sample_format_bytes_i32() {
        assert_eq!(NdiSampleFormat::I32.bytes_per_sample(), 4);
    }

    #[test]
    fn test_sample_format_is_planar() {
        assert!(NdiSampleFormat::F32Planar.is_planar());
        assert!(!NdiSampleFormat::I16.is_planar());
    }

    #[test]
    fn test_stereo_48k_30fps_samples_per_frame() {
        let cfg = NdiAudioConfig::stereo_48k_30fps();
        assert_eq!(cfg.samples_per_frame(30, 1), 1600);
    }

    #[test]
    fn test_stereo_48k_60fps_samples_per_frame() {
        let cfg = NdiAudioConfig::stereo_48k_60fps();
        assert_eq!(cfg.samples_per_frame(60, 1), 800);
    }

    #[test]
    fn test_frame_bytes_hint() {
        let cfg = NdiAudioConfig::stereo_48k_30fps();
        // 1600 samples * 2 channels * 4 bytes (f32)
        assert_eq!(cfg.frame_bytes_hint(), 1600 * 2 * 4);
    }

    #[test]
    fn test_samples_per_frame_zero_fps() {
        let cfg = NdiAudioConfig::stereo_48k_30fps();
        assert_eq!(cfg.samples_per_frame(0, 1), 0);
    }

    #[test]
    fn test_audio_buffer_zeroed_size() {
        let cfg = NdiAudioConfig::stereo_48k_30fps();
        let buf = NdiAudioBuffer::zeroed(cfg, 1600);
        // planar: 2 planes of 1600 * 4 bytes
        assert_eq!(buf.data.len(), 1600 * 2 * 4);
    }

    #[test]
    fn test_audio_buffer_channel_data_planar() {
        let cfg = NdiAudioConfig::stereo_48k_30fps();
        let buf = NdiAudioBuffer::zeroed(cfg, 1600);
        let ch0 = buf.channel_data(0).expect("expected channel data");
        assert_eq!(ch0.len(), 1600 * 4);
        let ch1 = buf.channel_data(1).expect("expected channel data");
        assert_eq!(ch1.len(), 1600 * 4);
    }

    #[test]
    fn test_audio_buffer_channel_out_of_range() {
        let cfg = NdiAudioConfig::stereo_48k_30fps();
        let buf = NdiAudioBuffer::zeroed(cfg, 1600);
        assert!(buf.channel_data(2).is_none());
    }

    #[test]
    fn test_audio_buffer_samples() {
        let cfg = NdiAudioConfig::stereo_48k_30fps();
        let buf = NdiAudioBuffer::zeroed(cfg, 1600);
        assert_eq!(buf.samples(), 1600);
    }

    #[test]
    fn test_ndi_audio_format_frame_size() {
        let cfg = NdiAudioConfig::stereo_48k_30fps();
        let fmt = NdiAudioFormat::packed(cfg);
        // 1600 * 2 * 4 = 12800
        assert_eq!(fmt.frame_size_bytes(30, 1), 12800);
    }

    #[test]
    fn test_ndi_audio_format_effective_stride_packed() {
        let cfg = NdiAudioConfig::stereo_48k_30fps();
        let fmt = NdiAudioFormat::packed(cfg);
        assert_eq!(fmt.effective_stride(30, 1), fmt.frame_size_bytes(30, 1));
    }

    #[test]
    fn test_ndi_audio_format_effective_stride_custom() {
        let cfg = NdiAudioConfig::stereo_48k_30fps();
        let fmt = NdiAudioFormat::new(cfg, 16384);
        assert_eq!(fmt.effective_stride(30, 1), 16384);
    }
}
