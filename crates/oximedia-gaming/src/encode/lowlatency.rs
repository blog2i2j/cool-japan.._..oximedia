//! Low-latency encoding pipeline.
//!
//! Implements ultra-low latency encoding for game streaming.

use crate::{GamingError, GamingResult};
use std::time::Duration;

/// Low-latency encoder.
pub struct LowLatencyEncoder {
    /// Encoder configuration.
    pub config: EncoderConfig,
    frames_encoded: u64,
}

/// Encoder configuration.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Input resolution
    pub resolution: (u32, u32),
    /// Target framerate
    pub framerate: u32,
    /// Bitrate in kbps
    pub bitrate: u32,
    /// Latency mode
    pub latency_mode: LatencyMode,
    /// Keyframe interval in frames
    pub keyframe_interval: u32,
    /// Use B-frames
    pub use_b_frames: bool,
    /// Rate control mode
    pub rate_control: RateControlMode,
}

/// Latency mode for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyMode {
    /// Ultra-low latency (<50ms) - no B-frames, minimal buffering
    UltraLow,
    /// Low latency (<100ms) - minimal B-frames
    Low,
    /// Normal latency - standard encoding
    Normal,
}

/// Rate control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControlMode {
    /// Constant bitrate
    Cbr,
    /// Variable bitrate
    Vbr,
    /// Constant quality
    Cq,
}

/// Encoded frame data.
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    /// Encoded data
    pub data: Vec<u8>,
    /// Presentation timestamp
    pub pts: Duration,
    /// Decode timestamp
    pub dts: Duration,
    /// Is keyframe
    pub is_keyframe: bool,
}

impl LowLatencyEncoder {
    /// Create a new low-latency encoder.
    ///
    /// # Errors
    ///
    /// Returns error if encoder initialization fails.
    pub fn new(config: EncoderConfig) -> GamingResult<Self> {
        if config.resolution.0 == 0 || config.resolution.1 == 0 {
            return Err(GamingError::InvalidConfig(
                "Resolution must be non-zero".to_string(),
            ));
        }

        if config.framerate == 0 {
            return Err(GamingError::InvalidConfig(
                "Framerate must be non-zero".to_string(),
            ));
        }

        if config.bitrate < 500 {
            return Err(GamingError::InvalidConfig(
                "Bitrate must be at least 500 kbps".to_string(),
            ));
        }

        Ok(Self {
            config,
            frames_encoded: 0,
        })
    }

    /// Encode a frame.
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails.
    pub fn encode_frame(&mut self, _frame_data: &[u8]) -> GamingResult<EncodedFrame> {
        self.frames_encoded += 1;

        // In a real implementation, this would encode the frame
        Ok(EncodedFrame {
            data: Vec::new(),
            pts: Duration::from_millis(self.frames_encoded * 16),
            dts: Duration::from_millis(self.frames_encoded * 16),
            is_keyframe: self.frames_encoded % u64::from(self.config.keyframe_interval) == 1,
        })
    }

    /// Get encoder statistics.
    #[must_use]
    pub fn get_stats(&self) -> EncoderStats {
        EncoderStats {
            frames_encoded: self.frames_encoded,
            average_encoding_time: Duration::from_millis(5),
            current_bitrate: self.config.bitrate,
        }
    }

    /// Flush encoder and get remaining frames.
    ///
    /// # Errors
    ///
    /// Returns error if flush fails.
    pub fn flush(&mut self) -> GamingResult<Vec<EncodedFrame>> {
        Ok(Vec::new())
    }
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            resolution: (1920, 1080),
            framerate: 60,
            bitrate: 6000,
            latency_mode: LatencyMode::Low,
            keyframe_interval: 120,
            use_b_frames: false,
            rate_control: RateControlMode::Cbr,
        }
    }
}

/// Encoder statistics.
#[derive(Debug, Clone)]
pub struct EncoderStats {
    /// Total frames encoded
    pub frames_encoded: u64,
    /// Average encoding time per frame
    pub average_encoding_time: Duration,
    /// Current bitrate in kbps
    pub current_bitrate: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let config = EncoderConfig::default();
        let encoder = LowLatencyEncoder::new(config).expect("valid encoder");
        assert_eq!(encoder.frames_encoded, 0);
    }

    #[test]
    fn test_invalid_resolution() {
        let mut config = EncoderConfig::default();
        config.resolution = (0, 0);
        let result = LowLatencyEncoder::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_bitrate() {
        let mut config = EncoderConfig::default();
        config.bitrate = 100;
        let result = LowLatencyEncoder::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_frame() {
        let config = EncoderConfig::default();
        let mut encoder = LowLatencyEncoder::new(config).expect("valid encoder");

        let frame_data = vec![0u8; 1920 * 1080 * 4];
        let encoded = encoder
            .encode_frame(&frame_data)
            .expect("encode should succeed");

        assert!(!encoded.is_keyframe || encoder.frames_encoded == 1);
    }

    #[test]
    fn test_keyframe_interval() {
        let mut config = EncoderConfig::default();
        config.keyframe_interval = 30;

        let mut encoder = LowLatencyEncoder::new(config).expect("valid encoder");
        let frame_data = vec![0u8; 1920 * 1080 * 4];

        // First frame should be keyframe
        let frame1 = encoder
            .encode_frame(&frame_data)
            .expect("encode should succeed");
        assert!(frame1.is_keyframe);

        // Next 29 frames should not be keyframes
        for _ in 0..29 {
            let frame = encoder
                .encode_frame(&frame_data)
                .expect("encode should succeed");
            assert!(!frame.is_keyframe);
        }

        // 31st frame should be keyframe
        let frame31 = encoder
            .encode_frame(&frame_data)
            .expect("encode should succeed");
        assert!(frame31.is_keyframe);
    }

    #[test]
    fn test_ultra_low_latency_mode() {
        let mut config = EncoderConfig::default();
        config.latency_mode = LatencyMode::UltraLow;
        config.use_b_frames = false;

        let encoder = LowLatencyEncoder::new(config).expect("valid encoder");
        assert_eq!(encoder.config.latency_mode, LatencyMode::UltraLow);
        assert!(!encoder.config.use_b_frames);
    }

    #[test]
    fn test_encoder_stats() {
        let config = EncoderConfig::default();
        let mut encoder = LowLatencyEncoder::new(config).expect("valid encoder");

        let frame_data = vec![0u8; 1920 * 1080 * 4];
        encoder
            .encode_frame(&frame_data)
            .expect("encode should succeed");

        let stats = encoder.get_stats();
        assert_eq!(stats.frames_encoded, 1);
    }
}
