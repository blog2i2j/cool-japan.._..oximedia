//! Basic audio post-production pipeline components.
//!
//! Provides:
//! - [`DialogueLeveler`]   – Normalise dialogue to a target LUFS (default –24 LUFS).
//! - [`SurroundPanner`]    – 5.1 / 7.1 spatial audio panning.
//! - [`AudioExportConfig`] – Export configuration (format / codec / bitrate).

#![allow(dead_code)]

use crate::error::{AudioPostError, AudioPostResult};

// ── DialogueLeveler ────────────────────────────────────────────────────────────

/// Dialogue normalisation to a configurable loudness target (default –24 LUFS).
///
/// This is a simplified single-pass gain leveler.  A real implementation would
/// use an integrated loudness measurement (ITU-R BS.1770-4); here we compute
/// RMS loudness from the input samples and apply a compensating gain.
#[derive(Debug, Clone)]
pub struct DialogueLeveler {
    /// Target integrated loudness in LUFS (negative value, e.g. –24.0).
    pub target_lufs: f32,
    /// Maximum gain that can be applied (dB).  Prevents over-amplification of
    /// very quiet passages.
    pub max_gain_db: f32,
    /// Maximum attenuation that can be applied (dB, stored as positive number).
    pub max_attenuation_db: f32,
}

impl Default for DialogueLeveler {
    fn default() -> Self {
        Self {
            target_lufs: -24.0,
            max_gain_db: 20.0,
            max_attenuation_db: 20.0,
        }
    }
}

impl DialogueLeveler {
    /// Create a leveler with the default –24 LUFS target.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the target loudness in LUFS.
    ///
    /// # Errors
    ///
    /// Returns [`AudioPostError::InvalidLoudnessTarget`] for values outside
    /// the range –60 to 0 LUFS.
    pub fn with_target_lufs(mut self, lufs: f32) -> AudioPostResult<Self> {
        if !(-60.0..=0.0).contains(&lufs) {
            return Err(AudioPostError::InvalidLoudnessTarget(lufs));
        }
        self.target_lufs = lufs;
        Ok(self)
    }

    /// Set maximum gain (dB).
    #[must_use]
    pub fn with_max_gain_db(mut self, db: f32) -> Self {
        self.max_gain_db = db.max(0.0);
        self
    }

    /// Process mono/interleaved PCM samples (f32, full-scale ±1.0).
    ///
    /// Computes the RMS loudness, derives the required gain, clamps it to
    /// `max_gain_db` / `max_attenuation_db`, and applies it in-place.
    ///
    /// # Errors
    ///
    /// Returns [`AudioPostError::InvalidBufferSize`] for empty buffers.
    pub fn process(&self, samples: &mut [f32]) -> AudioPostResult<()> {
        if samples.is_empty() {
            return Err(AudioPostError::InvalidBufferSize(0));
        }

        // Compute RMS in dBFS
        let rms_sq: f64 = samples
            .iter()
            .map(|&s| f64::from(s) * f64::from(s))
            .sum::<f64>()
            / samples.len() as f64;

        // Guard against silence / denormals
        if rms_sq < 1e-12 {
            return Ok(()); // nothing to do
        }

        let rms_dbfs = 10.0 * rms_sq.log10() as f32;

        // Simple approximation: treat RMS dBFS as LUFS offset
        // In a real implementation this would be an ITU-R BS.1770 measurement.
        let current_lufs = rms_dbfs - 3.0; // rough LUFS ≈ RMS – 3 dB

        let required_gain_db = self.target_lufs - current_lufs;
        let clamped_gain_db = required_gain_db
            .min(self.max_gain_db)
            .max(-self.max_attenuation_db);

        let linear_gain = 10_f32.powf(clamped_gain_db / 20.0);

        for s in samples.iter_mut() {
            *s = (*s * linear_gain).clamp(-1.0, 1.0);
        }

        Ok(())
    }

    /// Return the gain (linear) that would be applied to a buffer with the
    /// given RMS level (dBFS).  Useful for dry-run inspection.
    #[must_use]
    pub fn gain_for_rms_dbfs(&self, rms_dbfs: f32) -> f32 {
        let current_lufs = rms_dbfs - 3.0;
        let required_gain_db = self.target_lufs - current_lufs;
        let clamped = required_gain_db
            .min(self.max_gain_db)
            .max(-self.max_attenuation_db);
        10_f32.powf(clamped / 20.0)
    }
}

// ── SurroundPanner ─────────────────────────────────────────────────────────────

/// Surround format for [`SurroundPanner`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurroundFormat {
    /// 5.1 surround (L, R, C, LFE, Ls, Rs).
    Surround51,
    /// 7.1 surround (L, R, C, LFE, Lss, Rss, Lrs, Rrs).
    Surround71,
}

impl SurroundFormat {
    /// Number of output channels.
    #[must_use]
    pub const fn channel_count(self) -> usize {
        match self {
            Self::Surround51 => 6,
            Self::Surround71 => 8,
        }
    }
}

/// 5.1 / 7.1 spatial audio panner.
///
/// Maps a mono dialogue source to surround channels using a simple constant-power
/// panning law based on azimuth (horizontal) and elevation (vertical) angles.
#[derive(Debug, Clone)]
pub struct SurroundPanner {
    format: SurroundFormat,
    /// Azimuth angle in degrees: 0 = front-centre, +90 = right, –90 = left.
    pub azimuth: f32,
    /// Elevation in degrees: 0 = ear level, +90 = overhead.
    pub elevation: f32,
    /// LFE send level (0.0 – 1.0).
    pub lfe_level: f32,
    /// Centre channel percentage (0.0 – 1.0); 1.0 means fully locked to centre.
    pub center_percentage: f32,
}

impl Default for SurroundPanner {
    fn default() -> Self {
        Self {
            format: SurroundFormat::Surround51,
            azimuth: 0.0,
            elevation: 0.0,
            lfe_level: 0.0,
            center_percentage: 1.0,
        }
    }
}

impl SurroundPanner {
    /// Create a new panner for the given surround format.
    #[must_use]
    pub fn new(format: SurroundFormat) -> Self {
        Self {
            format,
            ..Self::default()
        }
    }

    /// Set azimuth (degrees, –180 to +180).
    #[must_use]
    pub fn with_azimuth(mut self, azimuth: f32) -> Self {
        self.azimuth = azimuth.clamp(-180.0, 180.0);
        self
    }

    /// Set elevation (degrees, –90 to +90).
    #[must_use]
    pub fn with_elevation(mut self, elevation: f32) -> Self {
        self.elevation = elevation.clamp(-90.0, 90.0);
        self
    }

    /// Set LFE send level.
    #[must_use]
    pub fn with_lfe_level(mut self, level: f32) -> Self {
        self.lfe_level = level.clamp(0.0, 1.0);
        self
    }

    /// Pan a mono sample to surround channels.
    ///
    /// Returns a `Vec` with `channel_count` gain coefficients (linear, 0–1).
    /// Multiply a mono sample by each coefficient to obtain the per-channel output.
    #[must_use]
    pub fn pan_coefficients(&self) -> Vec<f32> {
        let n = self.format.channel_count();
        let mut gains = vec![0.0f32; n];

        // Normalise azimuth to [–1, 1] where –1 = hard-left, +1 = hard-right
        let az_norm = (self.azimuth / 180.0).clamp(-1.0, 1.0);

        // Elevation attenuation (simple cosine)
        let el_rad = self.elevation.to_radians();
        let el_att = el_rad.cos();

        // Constant-power pan between L (index 0) and R (index 1)
        let pan_angle = (az_norm * 0.5 + 0.5) * std::f32::consts::FRAC_PI_2;
        let pan_r = pan_angle.sin();
        let pan_l = pan_angle.cos();

        match self.format {
            SurroundFormat::Surround51 => {
                // Channels: 0=L, 1=R, 2=C, 3=LFE, 4=Ls, 5=Rs
                let front_spread = 1.0 - self.center_percentage;
                gains[0] = pan_l * front_spread * el_att; // L
                gains[1] = pan_r * front_spread * el_att; // R
                gains[2] = self.center_percentage * el_att; // C
                gains[3] = self.lfe_level; // LFE
                                           // Rear spread based on elevation only (no rear signal from front-pan)
                let rear = (1.0 - el_att).max(0.0);
                gains[4] = pan_l * rear; // Ls
                gains[5] = pan_r * rear; // Rs
            }
            SurroundFormat::Surround71 => {
                // Channels: 0=L, 1=R, 2=C, 3=LFE, 4=Lss, 5=Rss, 6=Lrs, 7=Rrs
                let front_spread = 1.0 - self.center_percentage;
                gains[0] = pan_l * front_spread * el_att;
                gains[1] = pan_r * front_spread * el_att;
                gains[2] = self.center_percentage * el_att;
                gains[3] = self.lfe_level;
                let rear = (1.0 - el_att).max(0.0);
                gains[4] = pan_l * rear * 0.7; // side-surround
                gains[5] = pan_r * rear * 0.7;
                gains[6] = pan_l * rear * 0.3; // rear-surround
                gains[7] = pan_r * rear * 0.3;
            }
        }

        gains
    }

    /// Pan a block of mono samples to an interleaved multi-channel output buffer.
    ///
    /// `output` must have length `samples.len() * channel_count`.
    ///
    /// # Errors
    ///
    /// Returns [`AudioPostError::InvalidBufferSize`] if `output` is not
    /// exactly `samples.len() * channel_count` in length.
    pub fn process(&self, samples: &[f32], output: &mut [f32]) -> AudioPostResult<()> {
        let ch = self.format.channel_count();
        let expected = samples.len() * ch;
        if output.len() != expected {
            return Err(AudioPostError::InvalidBufferSize(output.len()));
        }

        let coeffs = self.pan_coefficients();
        for (i, &s) in samples.iter().enumerate() {
            for (c, &g) in coeffs.iter().enumerate() {
                output[i * ch + c] = s * g;
            }
        }

        Ok(())
    }

    /// Get the surround format.
    #[must_use]
    pub const fn format(&self) -> SurroundFormat {
        self.format
    }
}

// ── AudioExportConfig ──────────────────────────────────────────────────────────

/// Audio codec for export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodec {
    /// PCM (uncompressed).
    Pcm,
    /// FLAC (lossless).
    Flac,
    /// AAC-LC.
    AacLc,
    /// MP3.
    Mp3,
    /// Opus.
    Opus,
}

impl AudioCodec {
    /// Whether the codec is lossless.
    #[must_use]
    pub const fn is_lossless(self) -> bool {
        matches!(self, Self::Pcm | Self::Flac)
    }

    /// Default bitrate (kbps) for lossy codecs; 0 for lossless.
    #[must_use]
    pub const fn default_bitrate_kbps(self) -> u32 {
        match self {
            Self::Pcm | Self::Flac => 0,
            Self::AacLc => 256,
            Self::Mp3 => 320,
            Self::Opus => 192,
        }
    }
}

/// Container format for export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerFormat {
    /// WAV / BWF.
    Wav,
    /// FLAC.
    Flac,
    /// MP4 / M4A.
    Mp4,
    /// MP3.
    Mp3,
    /// Ogg.
    Ogg,
    /// MXF (broadcast).
    Mxf,
}

impl ContainerFormat {
    /// File extension (without leading dot).
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Mp4 => "m4a",
            Self::Mp3 => "mp3",
            Self::Ogg => "ogg",
            Self::Mxf => "mxf",
        }
    }
}

/// Complete export configuration: format, codec, sample rate, bit depth, and bitrate.
#[derive(Debug, Clone)]
pub struct AudioExportConfig {
    /// Container format.
    pub container: ContainerFormat,
    /// Audio codec.
    pub codec: AudioCodec,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Bit depth (e.g. 16, 24, 32).
    pub bit_depth: u8,
    /// Bitrate in kbps for lossy codecs; 0 means auto/codec-default.
    pub bitrate_kbps: u32,
    /// Number of output channels.
    pub channels: u8,
    /// Normalise to this loudness target before export (None = no normalisation).
    pub normalize_lufs: Option<f32>,
}

impl Default for AudioExportConfig {
    fn default() -> Self {
        Self {
            container: ContainerFormat::Wav,
            codec: AudioCodec::Pcm,
            sample_rate: 48_000,
            bit_depth: 24,
            bitrate_kbps: 0,
            channels: 2,
            normalize_lufs: None,
        }
    }
}

impl AudioExportConfig {
    /// Broadcast-quality stereo WAV (48 kHz, 24-bit PCM).
    #[must_use]
    pub fn broadcast_stereo() -> Self {
        Self::default()
    }

    /// Streaming AAC stereo (48 kHz, 256 kbps).
    #[must_use]
    pub fn streaming_aac() -> Self {
        Self {
            container: ContainerFormat::Mp4,
            codec: AudioCodec::AacLc,
            sample_rate: 48_000,
            bit_depth: 16,
            bitrate_kbps: 256,
            channels: 2,
            normalize_lufs: Some(-14.0),
        }
    }

    /// 5.1 surround WAV (48 kHz, 24-bit PCM, –23 LUFS EBU R128).
    #[must_use]
    pub fn surround_51() -> Self {
        Self {
            container: ContainerFormat::Wav,
            codec: AudioCodec::Pcm,
            sample_rate: 48_000,
            bit_depth: 24,
            bitrate_kbps: 0,
            channels: 6,
            normalize_lufs: Some(-23.0),
        }
    }

    /// Validate configuration consistency.
    ///
    /// # Errors
    ///
    /// Returns an error for:
    /// - Zero sample rate.
    /// - Zero channels.
    /// - Bitrate set for a lossless codec (and non-zero).
    pub fn validate(&self) -> AudioPostResult<()> {
        if self.sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(self.sample_rate));
        }
        if self.channels == 0 {
            return Err(AudioPostError::InvalidChannelCount(0));
        }
        if self.codec.is_lossless() && self.bitrate_kbps != 0 {
            return Err(AudioPostError::Generic(
                "Bitrate should be 0 for lossless codecs".into(),
            ));
        }
        Ok(())
    }

    /// Return a human-readable description of this configuration.
    #[must_use]
    pub fn description(&self) -> String {
        format!(
            "{:?}/{:?} {}Hz {}ch {}bit{}",
            self.container,
            self.codec,
            self.sample_rate,
            self.channels,
            self.bit_depth,
            if self.bitrate_kbps > 0 {
                format!(" @ {}kbps", self.bitrate_kbps)
            } else {
                String::new()
            }
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DialogueLeveler tests ────────────────────────────────────────────────

    #[test]
    fn test_dialogue_leveler_default_target() {
        let l = DialogueLeveler::default();
        assert!((l.target_lufs - (-24.0)).abs() < 1e-5);
    }

    #[test]
    fn test_dialogue_leveler_invalid_target() {
        let result = DialogueLeveler::new().with_target_lufs(10.0);
        assert!(result.is_err(), "Positive LUFS should be rejected");
    }

    #[test]
    fn test_dialogue_leveler_amplifies_quiet_signal() {
        let mut samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin() * 0.01).collect();
        let before_rms: f32 =
            (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        let leveler = DialogueLeveler::default();
        leveler
            .process(&mut samples)
            .expect("process should succeed");

        let after_rms: f32 =
            (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        assert!(after_rms > before_rms, "Quiet signal should be amplified");
    }

    #[test]
    fn test_dialogue_leveler_empty_buffer_error() {
        let mut empty: Vec<f32> = Vec::new();
        let result = DialogueLeveler::default().process(&mut empty);
        assert!(result.is_err());
    }

    #[test]
    fn test_dialogue_leveler_gain_for_rms() {
        let l = DialogueLeveler::default();
        // A –30 dBFS RMS ≈ –33 LUFS; target is –24 LUFS → ~9 dB gain expected
        let gain = l.gain_for_rms_dbfs(-30.0);
        assert!(gain > 1.0, "Should need positive gain: {gain}");
    }

    // ── SurroundPanner tests ─────────────────────────────────────────────────

    #[test]
    fn test_surround_panner_51_channel_count() {
        let p = SurroundPanner::new(SurroundFormat::Surround51);
        assert_eq!(p.pan_coefficients().len(), 6);
    }

    #[test]
    fn test_surround_panner_71_channel_count() {
        let p = SurroundPanner::new(SurroundFormat::Surround71);
        assert_eq!(p.pan_coefficients().len(), 8);
    }

    #[test]
    fn test_surround_panner_center_lock() {
        // center_percentage = 1.0, azimuth = 0 → centre channel should dominate
        let p = SurroundPanner::new(SurroundFormat::Surround51).with_azimuth(0.0);
        let g = p.pan_coefficients();
        assert!(
            g[2] > g[0],
            "Centre (g[2]={}) should exceed L (g[0]={})",
            g[2],
            g[0]
        );
    }

    #[test]
    fn test_surround_panner_right_pan() {
        let p = SurroundPanner::new(SurroundFormat::Surround51).with_azimuth(90.0);
        let g = p.pan_coefficients();
        // R should be >= L when panned right with front_spread
        assert!(g[1] >= g[0], "R should dominate when panned right");
    }

    #[test]
    fn test_surround_panner_process_output_size() {
        let p = SurroundPanner::new(SurroundFormat::Surround51);
        let input = vec![0.5f32; 64];
        let mut output = vec![0.0f32; 64 * 6];
        p.process(&input, &mut output)
            .expect("process should succeed");
    }

    #[test]
    fn test_surround_panner_process_wrong_output_size() {
        let p = SurroundPanner::new(SurroundFormat::Surround51);
        let input = vec![0.5f32; 64];
        let mut output = vec![0.0f32; 100]; // wrong
        let result = p.process(&input, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_surround_panner_lfe() {
        let p = SurroundPanner::new(SurroundFormat::Surround51).with_lfe_level(0.8);
        let g = p.pan_coefficients();
        assert!((g[3] - 0.8).abs() < 1e-5, "LFE gain mismatch: {}", g[3]);
    }

    // ── AudioExportConfig tests ──────────────────────────────────────────────

    #[test]
    fn test_export_config_default_is_valid() {
        let cfg = AudioExportConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_export_config_streaming_aac() {
        let cfg = AudioExportConfig::streaming_aac();
        assert_eq!(cfg.codec, AudioCodec::AacLc);
        assert_eq!(cfg.bitrate_kbps, 256);
        assert!(cfg.normalize_lufs.is_some());
    }

    #[test]
    fn test_export_config_surround_51() {
        let cfg = AudioExportConfig::surround_51();
        assert_eq!(cfg.channels, 6);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_export_config_invalid_sample_rate() {
        let mut cfg = AudioExportConfig::default();
        cfg.sample_rate = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_export_config_lossless_with_bitrate_invalid() {
        let mut cfg = AudioExportConfig::default(); // PCM
        cfg.bitrate_kbps = 320; // nonsensical for PCM
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_export_config_description_contains_format() {
        let cfg = AudioExportConfig::streaming_aac();
        let desc = cfg.description();
        assert!(
            desc.contains("48000"),
            "Description should include sample rate: {desc}"
        );
    }

    #[test]
    fn test_codec_is_lossless() {
        assert!(AudioCodec::Pcm.is_lossless());
        assert!(AudioCodec::Flac.is_lossless());
        assert!(!AudioCodec::Mp3.is_lossless());
        assert!(!AudioCodec::AacLc.is_lossless());
    }
}
