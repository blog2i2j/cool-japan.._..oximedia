//! Unified media watermarking that coordinates audio and video watermark embedding.
//!
//! Provides a single entry point for applying watermarks across both audio streams
//! and video frames, with consistent payload encoding, key management, and quality
//! verification. This bridges the gap between [`audio_watermark`] (FSK-based),
//! the generic [`WatermarkEmbedder`] (spread-spectrum/echo/phase/LSB/patchwork/QIM),
//! and [`forensic_watermark`] (DCT-domain QIM for pixel data).

use crate::audio_watermark::{AudioWatermarkConfig, AudioWatermarkDecoder, AudioWatermarkEncoder};
use crate::forensic_watermark::{ForensicDetector, ForensicEmbedder, ForensicPayload};
use crate::{WatermarkConfig, WatermarkDetector, WatermarkEmbedder, WatermarkResult};

/// Which media tracks to watermark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatermarkTarget {
    /// Embed in audio only.
    AudioOnly,
    /// Embed in video only.
    VideoOnly,
    /// Embed in both audio and video tracks.
    Both,
}

/// Configuration for the unified media watermarker.
#[derive(Debug, Clone)]
pub struct MediaWatermarkConfig {
    /// Which tracks to target.
    pub target: WatermarkTarget,
    /// Audio watermark configuration (spread-spectrum / echo / etc.).
    pub audio_config: WatermarkConfig,
    /// Audio sample rate.
    pub audio_sample_rate: u32,
    /// FSK audio watermark configuration (for embedding a 32-bit identifier).
    pub fsk_config: AudioWatermarkConfig,
    /// Whether to also embed a 32-bit FSK identifier in audio.
    pub embed_fsk_id: bool,
    /// Forensic (video) watermark block size.
    pub video_block_size: usize,
    /// Forensic (video) watermark embedding strength (0.0–1.0).
    pub video_strength: f32,
    /// Forensic payload to embed in video frames.
    pub forensic_payload: Option<ForensicPayload>,
    /// FSK gain for mixing into audio (default 0.02 = -34 dB).
    pub fsk_gain: f32,
}

impl Default for MediaWatermarkConfig {
    fn default() -> Self {
        Self {
            target: WatermarkTarget::Both,
            audio_config: WatermarkConfig::default(),
            audio_sample_rate: 48000,
            fsk_config: AudioWatermarkConfig::default(),
            embed_fsk_id: false,
            video_block_size: 8,
            video_strength: 0.5,
            forensic_payload: None,
            fsk_gain: 0.02,
        }
    }
}

impl MediaWatermarkConfig {
    /// Set the watermark target.
    #[must_use]
    pub fn with_target(mut self, target: WatermarkTarget) -> Self {
        self.target = target;
        self
    }

    /// Set the audio watermark algorithm and parameters.
    #[must_use]
    pub fn with_audio_config(mut self, config: WatermarkConfig) -> Self {
        self.audio_config = config;
        self
    }

    /// Set the audio sample rate.
    #[must_use]
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.audio_sample_rate = sample_rate;
        self
    }

    /// Enable FSK identifier embedding with a 32-bit payload.
    #[must_use]
    pub fn with_fsk_id(mut self, enabled: bool) -> Self {
        self.embed_fsk_id = enabled;
        self
    }

    /// Set the forensic payload for video watermarking.
    #[must_use]
    pub fn with_forensic_payload(mut self, payload: ForensicPayload) -> Self {
        self.forensic_payload = Some(payload);
        self
    }

    /// Set the video watermark strength.
    #[must_use]
    pub fn with_video_strength(mut self, strength: f32) -> Self {
        self.video_strength = strength.clamp(0.0, 1.0);
        self
    }
}

/// Result of a media watermark embedding operation.
#[derive(Debug, Clone)]
pub struct MediaWatermarkResult {
    /// Whether audio was watermarked.
    pub audio_watermarked: bool,
    /// Whether video was watermarked.
    pub video_watermarked: bool,
    /// Whether FSK identifier was embedded in audio.
    pub fsk_embedded: bool,
    /// Audio SNR estimate in dB (if audio was watermarked).
    pub audio_snr_db: Option<f64>,
    /// Number of video frames watermarked.
    pub video_frames_watermarked: usize,
}

/// Result of a media watermark detection operation.
#[derive(Debug, Clone)]
pub struct MediaDetectionResult {
    /// Audio payload bytes extracted (if any).
    pub audio_payload: Option<Vec<u8>>,
    /// FSK identifier decoded from audio (if any).
    pub fsk_id: Option<u32>,
    /// Forensic payload extracted from video (if any).
    pub forensic_payload: Option<ForensicPayload>,
    /// Number of tracks where watermark was found.
    pub tracks_detected: u32,
}

/// Unified media watermarker that coordinates audio and video watermark operations.
pub struct MediaWatermarker {
    config: MediaWatermarkConfig,
    audio_embedder: WatermarkEmbedder,
    audio_detector: WatermarkDetector,
    video_embedder: ForensicEmbedder,
    video_detector: ForensicDetector,
    fsk_encoder: AudioWatermarkEncoder,
}

impl MediaWatermarker {
    /// Create a new media watermarker with the given configuration.
    #[must_use]
    pub fn new(config: MediaWatermarkConfig) -> Self {
        let audio_embedder =
            WatermarkEmbedder::new(config.audio_config.clone(), config.audio_sample_rate);
        let audio_detector = WatermarkDetector::new(config.audio_config.clone());
        let video_embedder = ForensicEmbedder::new(config.video_block_size, config.video_strength);
        let video_detector =
            ForensicDetector::with_strength(config.video_block_size, config.video_strength);
        let fsk_encoder = AudioWatermarkEncoder::new(config.fsk_config.clone());

        Self {
            config,
            audio_embedder,
            audio_detector,
            video_embedder,
            video_detector,
            fsk_encoder,
        }
    }

    /// Embed watermark into audio samples.
    ///
    /// Applies the configured spread-spectrum (or other algorithm) watermark,
    /// and optionally also embeds an FSK identifier.
    ///
    /// # Errors
    ///
    /// Returns an error if the audio is too short for the payload or encoding fails.
    pub fn embed_audio(
        &self,
        samples: &[f32],
        payload: &[u8],
    ) -> WatermarkResult<(Vec<f32>, MediaWatermarkResult)> {
        if self.config.target == WatermarkTarget::VideoOnly {
            return Ok((
                samples.to_vec(),
                MediaWatermarkResult {
                    audio_watermarked: false,
                    video_watermarked: false,
                    fsk_embedded: false,
                    audio_snr_db: None,
                    video_frames_watermarked: 0,
                },
            ));
        }

        // Apply primary audio watermark
        let mut watermarked = self.audio_embedder.embed(samples, payload)?;

        // Estimate SNR
        let snr = estimate_snr(samples, &watermarked);

        // Optionally embed FSK identifier
        let fsk_embedded = if self.config.embed_fsk_id {
            if let Some(ref fp) = self.config.forensic_payload {
                let fsk_payload = fp.customer_id;
                let fsk_samples = self
                    .fsk_encoder
                    .encode(fsk_payload, self.config.audio_sample_rate);
                watermarked =
                    AudioWatermarkEncoder::embed(&watermarked, &fsk_samples, self.config.fsk_gain);
                true
            } else {
                false
            }
        } else {
            false
        };

        Ok((
            watermarked,
            MediaWatermarkResult {
                audio_watermarked: true,
                video_watermarked: false,
                fsk_embedded,
                audio_snr_db: Some(snr),
                video_frames_watermarked: 0,
            },
        ))
    }

    /// Embed forensic watermark into a single video frame (luma plane).
    ///
    /// The frame is modified in-place.
    pub fn embed_video_frame(
        &self,
        pixels: &mut [u8],
        width: usize,
        height: usize,
    ) -> MediaWatermarkResult {
        if self.config.target == WatermarkTarget::AudioOnly {
            return MediaWatermarkResult {
                audio_watermarked: false,
                video_watermarked: false,
                fsk_embedded: false,
                audio_snr_db: None,
                video_frames_watermarked: 0,
            };
        }

        if let Some(ref payload) = self.config.forensic_payload {
            self.video_embedder.embed(pixels, width, height, payload);
            MediaWatermarkResult {
                audio_watermarked: false,
                video_watermarked: true,
                fsk_embedded: false,
                audio_snr_db: None,
                video_frames_watermarked: 1,
            }
        } else {
            MediaWatermarkResult {
                audio_watermarked: false,
                video_watermarked: false,
                fsk_embedded: false,
                audio_snr_db: None,
                video_frames_watermarked: 0,
            }
        }
    }

    /// Embed watermarks into both audio and a batch of video frames.
    ///
    /// # Errors
    ///
    /// Returns an error if audio embedding fails.
    pub fn embed_media(
        &self,
        audio_samples: &[f32],
        audio_payload: &[u8],
        video_frames: &mut [VideoFrame],
    ) -> WatermarkResult<MediaWatermarkResult> {
        let mut total_result = MediaWatermarkResult {
            audio_watermarked: false,
            video_watermarked: false,
            fsk_embedded: false,
            audio_snr_db: None,
            video_frames_watermarked: 0,
        };

        // Audio embedding
        if self.config.target != WatermarkTarget::VideoOnly && !audio_samples.is_empty() {
            let (watermarked_audio, audio_result) =
                self.embed_audio(audio_samples, audio_payload)?;
            // Copy watermarked samples back — caller uses the returned result
            // In a real pipeline, the watermarked audio would be forwarded.
            // Here we store the SNR info.
            total_result.audio_watermarked = audio_result.audio_watermarked;
            total_result.fsk_embedded = audio_result.fsk_embedded;
            total_result.audio_snr_db = audio_result.audio_snr_db;
            // The watermarked audio is not stored here because we don't own the input.
            // The caller should use `embed_audio` separately if they need the samples.
            drop(watermarked_audio);
        }

        // Video embedding
        if self.config.target != WatermarkTarget::AudioOnly {
            let mut frame_count = 0_usize;
            for frame in video_frames.iter_mut() {
                let vr = self.embed_video_frame(&mut frame.pixels, frame.width, frame.height);
                if vr.video_watermarked {
                    frame_count += 1;
                }
            }
            if frame_count > 0 {
                total_result.video_watermarked = true;
                total_result.video_frames_watermarked = frame_count;
            }
        }

        Ok(total_result)
    }

    /// Detect watermarks from audio and/or a video frame.
    ///
    /// # Errors
    ///
    /// Returns an error if detection encounters an unrecoverable issue.
    pub fn detect(
        &self,
        audio_samples: Option<&[f32]>,
        audio_expected_bits: usize,
        video_frame: Option<(&[u8], usize, usize)>,
    ) -> WatermarkResult<MediaDetectionResult> {
        let mut result = MediaDetectionResult {
            audio_payload: None,
            fsk_id: None,
            forensic_payload: None,
            tracks_detected: 0,
        };

        // Audio detection
        if let Some(samples) = audio_samples {
            if self.config.target != WatermarkTarget::VideoOnly {
                // Try primary audio watermark detection
                match self.audio_detector.detect(samples, audio_expected_bits) {
                    Ok(payload) => {
                        result.audio_payload = Some(payload);
                        result.tracks_detected += 1;
                    }
                    Err(_) => {
                        // Primary detection failed — continue to try FSK
                    }
                }

                // Try FSK detection
                if self.config.embed_fsk_id {
                    let fsk_id = AudioWatermarkDecoder::decode(
                        samples,
                        &self.config.fsk_config,
                        self.config.audio_sample_rate,
                    );
                    if let Some(id) = fsk_id {
                        result.fsk_id = Some(id);
                        if result.tracks_detected == 0 {
                            result.tracks_detected += 1;
                        }
                    }
                }
            }
        }

        // Video detection
        if let Some((pixels, width, height)) = video_frame {
            if self.config.target != WatermarkTarget::AudioOnly {
                if let Some(payload) = self.video_detector.detect(pixels, width, height) {
                    result.forensic_payload = Some(payload);
                    result.tracks_detected += 1;
                }
            }
        }

        Ok(result)
    }

    /// Check whether the audio watermark capacity is sufficient for the given payload.
    #[must_use]
    pub fn audio_capacity_sufficient(&self, sample_count: usize, payload_bytes: usize) -> bool {
        let capacity_bits = self.audio_embedder.capacity(sample_count);
        capacity_bits >= payload_bytes * 8
    }

    /// Check whether the video frame is large enough for forensic watermarking.
    #[must_use]
    pub fn video_frame_sufficient(&self, width: usize, height: usize) -> bool {
        let bs = self.config.video_block_size.max(8);
        let blocks = (width / bs) * (height / bs);
        blocks >= 64
    }

    /// Cross-verify audio and video watermarks for consistency.
    ///
    /// If both an FSK identifier and a forensic payload are present, verify
    /// that the customer IDs match.
    #[must_use]
    pub fn cross_verify(&self, detection: &MediaDetectionResult) -> CrossVerifyResult {
        let mut result = CrossVerifyResult {
            consistent: true,
            audio_present: detection.audio_payload.is_some() || detection.fsk_id.is_some(),
            video_present: detection.forensic_payload.is_some(),
            id_match: None,
        };

        // Cross-check FSK customer ID against forensic customer ID
        if let (Some(fsk_id), Some(ref fp)) = (detection.fsk_id, &detection.forensic_payload) {
            let ids_match = fsk_id == fp.customer_id;
            result.id_match = Some(ids_match);
            if !ids_match {
                result.consistent = false;
            }
        }

        result
    }
}

/// A simple video frame representation for batch processing.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Raw luma pixel data.
    pub pixels: Vec<u8>,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

impl VideoFrame {
    /// Create a new video frame.
    #[must_use]
    pub fn new(pixels: Vec<u8>, width: usize, height: usize) -> Self {
        Self {
            pixels,
            width,
            height,
        }
    }
}

/// Result of cross-verifying audio and video watermarks.
#[derive(Debug, Clone)]
pub struct CrossVerifyResult {
    /// Whether the detected watermarks are mutually consistent.
    pub consistent: bool,
    /// Whether an audio watermark was present.
    pub audio_present: bool,
    /// Whether a video watermark was present.
    pub video_present: bool,
    /// Whether FSK and forensic customer IDs match (if both present).
    pub id_match: Option<bool>,
}

/// Estimate the signal-to-noise ratio in dB between original and watermarked audio.
fn estimate_snr(original: &[f32], watermarked: &[f32]) -> f64 {
    if original.is_empty() {
        return 0.0;
    }
    let n = original.len().min(watermarked.len());
    let mut signal_power = 0.0_f64;
    let mut noise_power = 0.0_f64;
    for i in 0..n {
        let s = f64::from(original[i]);
        let w = f64::from(watermarked[i]);
        signal_power += s * s;
        noise_power += (w - s) * (w - s);
    }
    if noise_power < 1e-30 {
        return 120.0; // effectively silent noise
    }
    10.0 * (signal_power / noise_power).log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Algorithm;

    fn test_payload() -> ForensicPayload {
        ForensicPayload {
            customer_id: 42,
            session_id: 7,
            timestamp_sec: 1000,
        }
    }

    #[test]
    fn test_media_watermark_config_defaults() {
        let cfg = MediaWatermarkConfig::default();
        assert_eq!(cfg.target, WatermarkTarget::Both);
        assert_eq!(cfg.audio_sample_rate, 48000);
        assert_eq!(cfg.video_block_size, 8);
    }

    #[test]
    fn test_config_builder() {
        let cfg = MediaWatermarkConfig::default()
            .with_target(WatermarkTarget::AudioOnly)
            .with_sample_rate(44100)
            .with_video_strength(0.8)
            .with_fsk_id(true)
            .with_forensic_payload(test_payload());
        assert_eq!(cfg.target, WatermarkTarget::AudioOnly);
        assert_eq!(cfg.audio_sample_rate, 44100);
        assert!(cfg.embed_fsk_id);
        assert!(cfg.forensic_payload.is_some());
    }

    #[test]
    fn test_video_frame_sufficient() {
        let cfg = MediaWatermarkConfig::default();
        let wm = MediaWatermarker::new(cfg);
        // 64x64 = 8x8 grid of 8x8 blocks = 64 blocks, just enough
        assert!(wm.video_frame_sufficient(64, 64));
        // Too small
        assert!(!wm.video_frame_sufficient(16, 16));
    }

    #[test]
    fn test_video_only_skips_audio() {
        let cfg = MediaWatermarkConfig::default().with_target(WatermarkTarget::VideoOnly);
        let wm = MediaWatermarker::new(cfg);
        let samples = vec![0.0_f32; 44100];
        let (result_audio, info) = wm.embed_audio(&samples, b"test").expect("should succeed");
        assert!(!info.audio_watermarked);
        assert_eq!(result_audio, samples);
    }

    #[test]
    fn test_audio_only_skips_video() {
        let cfg = MediaWatermarkConfig::default().with_target(WatermarkTarget::AudioOnly);
        let wm = MediaWatermarker::new(cfg);
        let mut pixels = vec![128_u8; 64 * 64];
        let original = pixels.clone();
        let info = wm.embed_video_frame(&mut pixels, 64, 64);
        assert!(!info.video_watermarked);
        assert_eq!(pixels, original);
    }

    #[test]
    fn test_embed_video_frame_with_payload() {
        let cfg = MediaWatermarkConfig::default()
            .with_forensic_payload(test_payload())
            .with_video_strength(0.5);
        let wm = MediaWatermarker::new(cfg);
        let mut pixels = vec![128_u8; 64 * 64];
        let original = pixels.clone();
        let info = wm.embed_video_frame(&mut pixels, 64, 64);
        assert!(info.video_watermarked);
        assert_ne!(pixels, original);
    }

    #[test]
    fn test_video_embed_detect_roundtrip() {
        let payload = test_payload();
        let cfg = MediaWatermarkConfig::default()
            .with_forensic_payload(payload)
            .with_video_strength(0.5);
        let wm = MediaWatermarker::new(cfg);

        let mut pixels: Vec<u8> = (0..64 * 64)
            .map(|i| ((i as f64 * 1.234).sin() * 50.0 + 128.0) as u8)
            .collect();
        wm.embed_video_frame(&mut pixels, 64, 64);

        let detection = wm
            .detect(None, 0, Some((&pixels, 64, 64)))
            .expect("should succeed");
        assert!(detection.forensic_payload.is_some());
        let fp = detection.forensic_payload.expect("should be present");
        assert_eq!(fp.customer_id, 42);
    }

    #[test]
    fn test_cross_verify_consistent() {
        let cfg = MediaWatermarkConfig::default();
        let wm = MediaWatermarker::new(cfg);
        let detection = MediaDetectionResult {
            audio_payload: Some(vec![1, 2, 3]),
            fsk_id: Some(42),
            forensic_payload: Some(test_payload()),
            tracks_detected: 2,
        };
        let cv = wm.cross_verify(&detection);
        assert!(cv.consistent);
        assert_eq!(cv.id_match, Some(true));
    }

    #[test]
    fn test_cross_verify_inconsistent() {
        let cfg = MediaWatermarkConfig::default();
        let wm = MediaWatermarker::new(cfg);
        let detection = MediaDetectionResult {
            audio_payload: None,
            fsk_id: Some(999), // different from payload's customer_id=42
            forensic_payload: Some(test_payload()),
            tracks_detected: 2,
        };
        let cv = wm.cross_verify(&detection);
        assert!(!cv.consistent);
        assert_eq!(cv.id_match, Some(false));
    }

    #[test]
    fn test_cross_verify_partial_no_fsk() {
        let cfg = MediaWatermarkConfig::default();
        let wm = MediaWatermarker::new(cfg);
        let detection = MediaDetectionResult {
            audio_payload: Some(vec![1]),
            fsk_id: None,
            forensic_payload: Some(test_payload()),
            tracks_detected: 2,
        };
        let cv = wm.cross_verify(&detection);
        assert!(cv.consistent);
        assert!(cv.id_match.is_none());
    }

    #[test]
    fn test_estimate_snr_identical() {
        let samples = vec![0.5_f32; 1000];
        let snr = estimate_snr(&samples, &samples);
        assert!(snr > 100.0);
    }

    #[test]
    fn test_estimate_snr_with_noise() {
        let original: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let noisy: Vec<f32> = original.iter().map(|&s| s + 0.001).collect();
        let snr = estimate_snr(&original, &noisy);
        assert!(snr > 20.0);
    }

    #[test]
    fn test_estimate_snr_empty() {
        assert_eq!(estimate_snr(&[], &[]), 0.0);
    }

    #[test]
    fn test_embed_media_batch() {
        let payload = test_payload();
        let cfg = MediaWatermarkConfig::default()
            .with_target(WatermarkTarget::VideoOnly)
            .with_forensic_payload(payload)
            .with_video_strength(0.5);
        let wm = MediaWatermarker::new(cfg);

        let mut frames: Vec<VideoFrame> = (0..3)
            .map(|_| VideoFrame::new(vec![128_u8; 64 * 64], 64, 64))
            .collect();

        let result = wm
            .embed_media(&[], &[], &mut frames)
            .expect("should succeed");
        assert!(result.video_watermarked);
        assert_eq!(result.video_frames_watermarked, 3);
    }

    #[test]
    fn test_audio_capacity_check() {
        let cfg = MediaWatermarkConfig::default()
            .with_audio_config(WatermarkConfig::default().with_algorithm(Algorithm::Lsb));
        let wm = MediaWatermarker::new(cfg);
        // LSB: capacity = sample_count, so 44100 samples can hold 44100 bits = 5512 bytes
        assert!(wm.audio_capacity_sufficient(44100, 100));
        assert!(!wm.audio_capacity_sufficient(10, 100));
    }
}
