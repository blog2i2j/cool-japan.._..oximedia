//! Professional broadcast audio metering for `OxiMedia`.
//!
//! This crate provides comprehensive, standards-compliant audio loudness measurement
//! and metering for broadcast, streaming, and professional audio applications.
//!
//! # Supported Standards
//!
//! - **ITU-R BS.1770-4** - Algorithms to measure audio programme loudness and true-peak level
//! - **ITU-R BS.1771** - Requirements for loudness and true-peak indicating meters
//! - **EBU R128** - Loudness normalisation and permitted maximum level (European standard)
//! - **ATSC A/85** - Techniques for Establishing and Maintaining Audio Loudness (US standard)
//! - **Dolby Metadata** - Dialogue Intelligence metadata generation (metadata only, respects IP)
//!
//! # Features
//!
//! ## Loudness Measurement
//!
//! - **Momentary Loudness** - 400ms sliding window with 75% overlap
//! - **Short-term Loudness** - 3-second sliding window with 75% overlap
//! - **Integrated Loudness** - Gated program loudness (LKFS/LUFS)
//! - **Loudness Range (LRA)** - Dynamic range measurement using percentile-based method
//!
//! ## True Peak Detection
//!
//! - **4x Oversampling** - Detects inter-sample peaks using sinc interpolation
//! - **Per-channel Tracking** - Individual true peak levels for each channel
//! - **dBTP Conversion** - True peak in dB relative to full scale
//!
//! ## Gating Algorithm
//!
//! - **Absolute Gate** - -70 LKFS threshold
//! - **Relative Gate** - -10 LU below ungated loudness
//! - **Two-stage Process** - ITU-R BS.1771 compliant gating
//!
//! ## Multi-channel Support
//!
//! Supports up to 7.1.4 Dolby Atmos layouts with proper channel weighting:
//! - Mono (1.0)
//! - Stereo (2.0)
//! - 5.1 Surround
//! - 7.1 Surround
//! - 7.1.4 Dolby Atmos (bed channels)
//!
//! ## Compliance Checking
//!
//! - EBU R128 compliance (target: -23 LUFS ±1 LU, peak: -1 dBTP)
//! - ATSC A/85 compliance (target: -24 LKFS ±2 dB, peak: -2 dBTP)
//! - Streaming platform targets (Spotify, `YouTube`, Apple Music, etc.)
//!
//! # Example Usage
//!
//! ## Basic Loudness Metering
//!
//! ```rust,no_run
//! use oximedia_metering::{LoudnessMeter, MeterConfig, Standard};
//!
//! // Create meter for EBU R128
//! let config = MeterConfig::new(Standard::EbuR128, 48000.0, 2);
//! let mut meter = LoudnessMeter::new(config).expect("Failed to create meter");
//!
//! // Process audio samples (interleaved f32)
//! # let audio_samples: &[f32] = &[];
//! meter.process_f32(audio_samples);
//!
//! // Get loudness metrics
//! let metrics = meter.metrics();
//! println!("Integrated: {:.1} LUFS", metrics.integrated_lufs);
//! println!("LRA: {:.1} LU", metrics.loudness_range);
//! println!("True Peak: {:.1} dBTP", metrics.true_peak_dbtp);
//!
//! // Check compliance
//! let compliance = meter.check_compliance();
//! if compliance.is_compliant() {
//!     println!("Audio is compliant with {}", compliance.standard_name());
//! }
//!
//! // Generate detailed report
//! let report = meter.generate_report();
//! println!("{}", report);
//! ```
//!
//! ## Peak Metering
//!
//! ```rust,no_run
//! use oximedia_metering::{PeakMeter, PeakMeterType};
//!
//! // Create a VU meter for stereo audio
//! let mut vu_meter = PeakMeter::new(
//!     PeakMeterType::Vu,
//!     48000.0,
//!     2,
//!     2.0  // 2 second peak hold
//! ).expect("Failed to create VU meter");
//!
//! # let audio_samples: &[f64] = &[];
//! vu_meter.process_interleaved(audio_samples);
//!
//! let peaks = vu_meter.peak_dbfs();
//! println!("L: {:.1} dBFS, R: {:.1} dBFS", peaks[0], peaks[1]);
//!
//! // Create an RMS meter with 300ms integration
//! let mut rms_meter = PeakMeter::new(
//!     PeakMeterType::Rms(0.3),
//!     48000.0,
//!     2,
//!     0.0
//! ).expect("Failed to create RMS meter");
//! ```
//!
//! ## K-System Metering
//!
//! ```rust,no_run
//! use oximedia_metering::{KSystemMeter, KSystemType};
//!
//! // Create K-14 meter (mastering standard)
//! let mut k_meter = KSystemMeter::new(
//!     KSystemType::K14,
//!     48000.0,
//!     2
//! ).expect("Failed to create K-meter");
//!
//! # let audio_samples: &[f64] = &[];
//! k_meter.process_interleaved(audio_samples);
//!
//! // Get levels relative to K-14 reference
//! let rms_levels = k_meter.rms_relative_db();
//! println!("RMS relative to K-14: L={:.1} dB, R={:.1} dB",
//!          rms_levels[0], rms_levels[1]);
//!
//! if k_meter.is_overload() {
//!     println!("Warning: Headroom exceeded!");
//! }
//! ```
//!
//! ## Phase Analysis
//!
//! ```rust,no_run
//! use oximedia_metering::{PhaseCorrelationMeter, StereoWidthAnalyzer};
//!
//! // Create phase correlation meter
//! let mut phase_meter = PhaseCorrelationMeter::new(48000.0, 0.4)
//!     .expect("Failed to create phase meter");
//!
//! # let audio_samples: &[f64] = &[];
//! phase_meter.process_interleaved(audio_samples);
//!
//! let correlation = phase_meter.correlation();
//! println!("Phase correlation: {:.2}", correlation);
//!
//! if phase_meter.has_phase_issues() {
//!     println!("Warning: Phase cancellation detected!");
//! }
//!
//! // Stereo width analysis
//! let mut width_analyzer = StereoWidthAnalyzer::new(48000.0)
//!     .expect("Failed to create width analyzer");
//!
//! width_analyzer.process_interleaved(audio_samples);
//! println!("Stereo width: {:.0}%", width_analyzer.width_percentage());
//! ```
//!
//! ## Spectrum Analysis
//!
//! ```rust,no_run
//! use oximedia_metering::{SpectrumAnalyzer, WindowFunction, WeightingCurve};
//!
//! // Create FFT-based spectrum analyzer
//! let mut spectrum = SpectrumAnalyzer::new(
//!     48000.0,
//!     2048,
//!     WindowFunction::Hann,
//!     WeightingCurve::A,
//!     1.0  // 1 second peak hold
//! ).expect("Failed to create spectrum analyzer");
//!
//! # let audio_samples: &[f64] = &[];
//! spectrum.process(audio_samples);
//!
//! let spectrum_db = spectrum.spectrum_db();
//! for (i, &magnitude) in spectrum_db.iter().take(10).enumerate() {
//!     let freq = spectrum.bin_frequency(i);
//!     println!("{:.0} Hz: {:.1} dB", freq, magnitude);
//! }
//! ```
//!
//! ## Video Metering
//!
//! ```rust,no_run
//! use oximedia_metering::{LuminanceMeter, GamutMeter, ColorGamut, QualityAnalyzer};
//! use ndarray::Array2;
//!
//! // Luminance metering for HDR content
//! let mut lum_meter = LuminanceMeter::new(1920, 1080, 1000.0, 256)
//!     .expect("Failed to create luminance meter");
//!
//! # let luminance_frame = Array2::<f64>::zeros((1080, 1920));
//! lum_meter.process(&luminance_frame)?;
//!
//! println!("Peak: {:.1} nits", lum_meter.peak_nits());
//! println!("Average: {:.1} nits", lum_meter.average_nits());
//! println!("Dynamic range: {:.1} stops", lum_meter.dynamic_range_stops());
//!
//! if lum_meter.is_hdr10() {
//!     println!("HDR10 content detected");
//! }
//!
//! // Color gamut analysis
//! let mut gamut_meter = GamutMeter::new(1920, 1080, ColorGamut::Rec2020)
//!     .expect("Failed to create gamut meter");
//!
//! # let r_channel = Array2::<f64>::zeros((1080, 1920));
//! # let g_channel = Array2::<f64>::zeros((1080, 1920));
//! # let b_channel = Array2::<f64>::zeros((1080, 1920));
//! gamut_meter.process(&r_channel, &g_channel, &b_channel)?;
//!
//! println!("Rec.2020 coverage: {:.1}%", gamut_meter.gamut_coverage_percentage());
//! println!("Max saturation: {:.2}", gamut_meter.max_saturation());
//!
//! // Video quality metrics (PSNR, SSIM)
//! let quality = QualityAnalyzer::new(1920, 1080, 1.0)
//!     .expect("Failed to create quality analyzer");
//!
//! # let reference_frame = Array2::<f64>::zeros((1080, 1920));
//! # let distorted_frame = Array2::<f64>::zeros((1080, 1920));
//! let metrics = quality.analyze(&reference_frame, &distorted_frame)?;
//!
//! println!("PSNR: {:.2} dB", metrics.psnr);
//! println!("SSIM: {:.4}", metrics.ssim);
//! println!("Quality: {}", metrics.rating());
//! ```
//!
//! ## Meter Rendering
//!
//! ```rust,no_run
//! use oximedia_metering::{BarMeterConfig, BarMeterData, ColorGradient, Orientation};
//!
//! // Configure a vertical bar meter
//! let config = BarMeterConfig {
//!     orientation: Orientation::Vertical,
//!     width: 30,
//!     height: 200,
//!     min_value: -60.0,
//!     max_value: 0.0,
//!     gradient: ColorGradient::traffic_light(),
//!     show_peak_hold: true,
//!     show_scale: true,
//!     ..Default::default()
//! };
//!
//! // Create meter data from dBFS values
//! let meter_data = BarMeterData::from_dbfs(
//!     -12.0,  // Current level
//!     -6.0,   // Peak hold
//!     -60.0,  // Min range
//!     0.0     // Max range
//! );
//!
//! if meter_data.is_clipping {
//!     println!("Clipping detected!");
//! }
//!
//! // Get color for current level
//! let color = config.gradient.color_at(meter_data.level);
//! println!("Meter color: RGB({}, {}, {})", color.r, color.g, color.b);
//! ```
//!
//! # Technical Implementation
//!
//! ## K-weighting Filter
//!
//! The K-weighting filter chain implements ITU-R BS.1770-4 specification:
//! - **Stage 1**: High-pass filter at 78.5 Hz (head diffraction modeling)
//! - **Stage 2**: High-shelf filter for revised low-frequency B-weighting (RLB)
//!
//! Both filters are implemented as second-order IIR biquad filters with
//! precise coefficients calculated for the given sample rate.
//!
//! ## Block Processing
//!
//! Audio is processed in overlapping blocks:
//! - **Block size**: 100ms (400ms blocks for momentary, 3000ms for short-term)
//! - **Overlap**: 75% (blocks advance by 25% of their duration)
//! - **Gating**: Applied on 400ms blocks with absolute (-70 LKFS) and relative (-10 LU) gates
//!
//! ## True Peak Detection
//!
//! Uses 4x oversampling with windowed sinc interpolation:
//! - Lanczos-windowed sinc function (a=3)
//! - Linear-phase FIR resampling
//! - Per-sample peak tracking
//!
//! # Performance
//!
//! - **Real-time capable**: Processes audio faster than real-time on modern CPUs
//! - **Memory efficient**: Circular buffers for sliding windows
//! - **Zero-copy where possible**: Processes interleaved or planar audio in-place when possible
//!
//! # Standards References
//!
//! - ITU-R BS.1770-4 (10/2015): "Algorithms to measure audio programme loudness and true-peak audio level"
//! - ITU-R BS.1771 (2006): "Requirements for loudness and true-peak indicating meters"
//! - EBU R 128 (2020): "Loudness normalisation and permitted maximum level of audio signals"
//! - ATSC A/85:2013: "Techniques for Establishing and Maintaining Audio Loudness for Digital Television"

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::similar_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::unused_self)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::fn_params_excessive_bools)]
#![allow(clippy::let_and_return)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::format_push_string)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::missing_panics_doc)]
#![allow(dead_code)]
#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::return_self_not_must_use
)]

pub mod atsc;
pub mod ballistics;
pub mod correlation;
pub mod correlation_meter;
pub mod dr_meter;
pub mod dynamic_range_meter;
pub mod dynamics;
pub mod ebu;
pub mod ebu_r128_impl;
pub mod filters;
pub mod gating;
pub mod k_weighting;
pub mod lkfs;
pub mod loudness_gate;
pub mod loudness_history;
pub mod m_s_meter;
pub mod meter_type_config;
pub mod octave_bands;
pub mod peak;
pub mod peak_meter;
pub mod phase;
pub mod phase_analysis;
pub mod phase_scope;
pub mod ppm;
pub mod range;
pub mod render;
pub mod report;
pub mod spectral_balance;
pub mod spectral_energy;
pub mod spectrum;
pub mod spectrum_bands;
pub mod true_peak;
pub mod truepeak;
pub mod video_color;
pub mod video_luminance;
pub mod video_quality;
pub mod vu_meter;

// Wave 12 modules
pub mod crest_factor;
pub mod k_weighted;
pub mod meter_bridge;

// Wave 15 modules
pub mod loudness_trend;
pub mod noise_floor;
pub mod stereo_balance;

use oximedia_core::types::SampleFormat;
use thiserror::Error;

pub use atsc::{AtscA85Compliance, AtscA85Meter};
pub use ballistics::{BallisticProcessor, BallisticType, MultiChannelBallistics};
pub use correlation::{
    CorrelationMeter, FrequencyBand, Goniometer as CorrelationGoniometer,
    GoniometerPoint as CorrelationGoniometerPoint, MultibandMeter, PhaseRelationship,
};
pub use dynamics::{DynamicRangeMeter, PlrMeter};
pub use ebu::{EbuR128Compliance, EbuR128Meter};
pub use filters::{KWeightFilter, KWeightFilterBank};
pub use gating::{GatingProcessor, GatingResult};
pub use lkfs::{LkfsCalculator, LufsValue};
pub use peak::{
    dbfs_to_linear, linear_to_dbfs, KSystemMeter, KSystemType, PeakMeter, PeakMeterType,
};
pub use phase::{Goniometer, GoniometerPoint, PhaseCorrelationMeter, StereoWidthAnalyzer};
pub use range::{LoudnessRange, LraCalculator};
pub use render::{
    colors, generate_db_scale, BarMeterConfig, BarMeterData, CircularMeterConfig, Color,
    ColorGradient, Orientation, ScaleMark, ScaleType,
};
pub use report::{ComplianceReport, LoudnessReport, MeteringReport};
pub use spectrum::{
    OctaveBand, OctaveBandAnalyzer, SpectrumAnalyzer, WeightingCurve, WindowFunction,
};
pub use truepeak::{TruePeak, TruePeakDetector};
pub use video_color::{
    ColorGamut, ColorTemperatureMeter, GamutMeter, HsvColor, RgbColor, SaturationMeter,
};
pub use video_luminance::{BlackWhiteLevelMeter, LuminanceMeter};
pub use video_quality::{
    BlockinessDetector, PsnrCalculator, QualityAnalyzer, QualityMetrics, SsimCalculator,
};

/// Metering error types.
#[derive(Error, Debug)]
pub enum MeteringError {
    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Insufficient data for measurement.
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Sample format not supported.
    #[error("Unsupported sample format: {0:?}")]
    UnsupportedFormat(SampleFormat),

    /// Channel configuration error.
    #[error("Channel error: {0}")]
    ChannelError(String),

    /// Calculation error.
    #[error("Calculation error: {0}")]
    CalculationError(String),
}

/// Metering result type.
pub type MeteringResult<T> = std::result::Result<T, MeteringError>;

/// Broadcast loudness standard.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Standard {
    /// EBU R128 (European Broadcasting Union).
    ///
    /// Target: -23 LUFS ±1 LU
    /// Max True Peak: -1.0 dBTP
    #[default]
    EbuR128,

    /// ATSC A/85 (Advanced Television Systems Committee - US).
    ///
    /// Target: -24 LKFS ±2 dB
    /// Max True Peak: -2.0 dBTP
    AtscA85,

    /// Spotify streaming platform.
    ///
    /// Target: -14 LUFS
    /// Max True Peak: -1.0 dBTP
    Spotify,

    /// `YouTube` streaming platform.
    ///
    /// Target: -14 LUFS
    /// Max True Peak: -1.0 dBTP
    YouTube,

    /// Apple Music streaming platform.
    ///
    /// Target: -16 LUFS
    /// Max True Peak: -1.0 dBTP
    AppleMusic,

    /// Netflix streaming platform.
    ///
    /// Target: -27 LUFS
    /// Max True Peak: -2.0 dBTP
    Netflix,

    /// Amazon Prime Video.
    ///
    /// Target: -24 LUFS
    /// Max True Peak: -2.0 dBTP
    AmazonPrime,

    /// Custom target loudness.
    ///
    /// Specify your own target in LUFS and max true peak in dBTP.
    Custom {
        /// Target loudness in LUFS.
        target_lufs: f64,
        /// Maximum true peak in dBTP.
        max_peak_dbtp: f64,
        /// Tolerance in LU.
        tolerance_lu: f64,
    },
}

impl Standard {
    /// Get the target loudness in LUFS for this standard.
    pub fn target_lufs(&self) -> f64 {
        match self {
            Self::EbuR128 => -23.0,
            Self::AtscA85 | Self::AmazonPrime => -24.0,
            Self::Spotify | Self::YouTube => -14.0,
            Self::AppleMusic => -16.0,
            Self::Netflix => -27.0,
            Self::Custom { target_lufs, .. } => *target_lufs,
        }
    }

    /// Get the maximum true peak in dBTP for this standard.
    pub fn max_true_peak_dbtp(&self) -> f64 {
        match self {
            Self::EbuR128 | Self::Spotify | Self::YouTube | Self::AppleMusic => -1.0,
            Self::AtscA85 | Self::Netflix | Self::AmazonPrime => -2.0,
            Self::Custom { max_peak_dbtp, .. } => *max_peak_dbtp,
        }
    }

    /// Get the tolerance in LU for this standard.
    pub fn tolerance_lu(&self) -> f64 {
        match self {
            Self::EbuR128 | Self::Spotify | Self::YouTube | Self::AppleMusic => 1.0,
            Self::AtscA85 | Self::Netflix | Self::AmazonPrime => 2.0,
            Self::Custom { tolerance_lu, .. } => *tolerance_lu,
        }
    }

    /// Get the standard name as a string.
    pub fn name(&self) -> &str {
        match self {
            Self::EbuR128 => "EBU R128",
            Self::AtscA85 => "ATSC A/85",
            Self::Spotify => "Spotify",
            Self::YouTube => "YouTube",
            Self::AppleMusic => "Apple Music",
            Self::Netflix => "Netflix",
            Self::AmazonPrime => "Amazon Prime Video",
            Self::Custom { .. } => "Custom",
        }
    }
}

/// Meter configuration.
#[derive(Clone, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct MeterConfig {
    /// Broadcast standard to measure against.
    pub standard: Standard,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of audio channels.
    pub channels: usize,
    /// Enable true peak detection (4x oversampling).
    pub enable_true_peak: bool,
    /// Enable loudness range (LRA) calculation.
    pub enable_lra: bool,
    /// Enable momentary loudness tracking.
    pub enable_momentary: bool,
    /// Enable short-term loudness tracking.
    pub enable_short_term: bool,
    /// Enable integrated loudness (gated program loudness).
    pub enable_integrated: bool,
}

impl MeterConfig {
    /// Create a new meter configuration.
    ///
    /// # Arguments
    ///
    /// * `standard` - Broadcast standard
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels
    pub fn new(standard: Standard, sample_rate: f64, channels: usize) -> Self {
        Self {
            standard,
            sample_rate,
            channels,
            enable_true_peak: true,
            enable_lra: true,
            enable_momentary: true,
            enable_short_term: true,
            enable_integrated: true,
        }
    }

    /// Create a minimal configuration (integrated loudness and true peak only).
    pub fn minimal(standard: Standard, sample_rate: f64, channels: usize) -> Self {
        Self {
            standard,
            sample_rate,
            channels,
            enable_true_peak: true,
            enable_lra: false,
            enable_momentary: false,
            enable_short_term: false,
            enable_integrated: true,
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns `MeteringError::InvalidConfig` if any configuration parameters are out of valid range.
    pub fn validate(&self) -> MeteringResult<()> {
        if self.sample_rate < 8000.0 || self.sample_rate > 192_000.0 {
            return Err(MeteringError::InvalidConfig(format!(
                "Sample rate {} Hz is out of valid range (8000-192000 Hz)",
                self.sample_rate
            )));
        }

        if self.channels == 0 || self.channels > 16 {
            return Err(MeteringError::InvalidConfig(format!(
                "Channel count {} is out of valid range (1-16)",
                self.channels
            )));
        }

        if !self.enable_integrated && !self.enable_momentary && !self.enable_short_term {
            return Err(MeteringError::InvalidConfig(
                "At least one loudness measurement must be enabled".to_string(),
            ));
        }

        Ok(())
    }
}

/// Channel configuration for multi-channel audio.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChannelLayout {
    /// Mono (1.0).
    Mono,
    /// Stereo (2.0).
    Stereo,
    /// 5.1 Surround (L, R, C, LFE, Ls, Rs).
    Surround51,
    /// 7.1 Surround (L, R, C, LFE, Ls, Rs, Lrs, Rrs).
    Surround71,
    /// 7.1.4 Dolby Atmos bed (L, R, C, LFE, Ls, Rs, Lrs, Rrs, Ltf, Rtf, Ltb, Rtb).
    Atmos714,
    /// Custom channel configuration.
    Custom(usize),
}

impl ChannelLayout {
    /// Get the number of channels.
    pub fn channel_count(&self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::Atmos714 => 12,
            Self::Custom(n) => *n,
        }
    }

    /// Get ITU-R BS.1770-4 channel weights for this layout.
    ///
    /// Returns a vector of weights to apply to each channel during loudness calculation.
    pub fn channel_weights(&self) -> Vec<f64> {
        match self {
            Self::Mono => vec![1.0],
            Self::Stereo => vec![1.0, 1.0],
            Self::Surround51 => {
                // L, R, C, LFE, Ls, Rs
                vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41]
            }
            Self::Surround71 => {
                // L, R, C, LFE, Ls, Rs, Lrs, Rrs
                vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.41, 1.41]
            }
            Self::Atmos714 => {
                // L, R, C, LFE, Ls, Rs, Lrs, Rrs, Ltf, Rtf, Ltb, Rtb
                vec![
                    1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41,
                ]
            }
            Self::Custom(n) => vec![1.0; *n],
        }
    }

    /// Create from channel count.
    pub fn from_channel_count(count: usize) -> Self {
        match count {
            1 => Self::Mono,
            2 => Self::Stereo,
            6 => Self::Surround51,
            8 => Self::Surround71,
            12 => Self::Atmos714,
            n => Self::Custom(n),
        }
    }
}

/// Loudness measurement metrics.
#[derive(Clone, Debug, Default)]
pub struct LoudnessMetrics {
    /// Momentary loudness in LUFS (400ms window).
    pub momentary_lufs: f64,
    /// Short-term loudness in LUFS (3s window).
    pub short_term_lufs: f64,
    /// Integrated loudness in LUFS (gated program loudness).
    pub integrated_lufs: f64,
    /// Loudness range in LU.
    pub loudness_range: f64,
    /// True peak in dBTP (maximum across all channels).
    pub true_peak_dbtp: f64,
    /// True peak in linear scale.
    pub true_peak_linear: f64,
    /// Maximum momentary loudness seen.
    pub max_momentary: f64,
    /// Maximum short-term loudness seen.
    pub max_short_term: f64,
    /// Per-channel true peaks in dBTP.
    pub channel_peaks_dbtp: Vec<f64>,
}

/// Main loudness meter.
///
/// This is the primary interface for loudness measurement. It combines all
/// measurement algorithms (LKFS, gating, true peak, LRA) into a single meter.
pub struct LoudnessMeter {
    config: MeterConfig,
    lkfs_calculator: LkfsCalculator,
    gating_processor: GatingProcessor,
    true_peak_detector: Option<TruePeakDetector>,
    lra_calculator: Option<LraCalculator>,
    filter_bank: KWeightFilterBank,
    channel_layout: ChannelLayout,
    samples_processed: usize,
}

impl LoudnessMeter {
    /// Create a new loudness meter.
    ///
    /// # Arguments
    ///
    /// * `config` - Meter configuration
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: MeterConfig) -> MeteringResult<Self> {
        config.validate()?;

        let channel_layout = ChannelLayout::from_channel_count(config.channels);
        let filter_bank = KWeightFilterBank::new(config.channels, config.sample_rate);
        let lkfs_calculator = LkfsCalculator::new(config.sample_rate, config.channels);
        let gating_processor = GatingProcessor::new(config.sample_rate, config.channels);

        let true_peak_detector = if config.enable_true_peak {
            Some(TruePeakDetector::new(config.sample_rate, config.channels))
        } else {
            None
        };

        let lra_calculator = if config.enable_lra {
            Some(LraCalculator::new())
        } else {
            None
        };

        Ok(Self {
            config,
            lkfs_calculator,
            gating_processor,
            true_peak_detector,
            lra_calculator,
            filter_bank,
            channel_layout,
            samples_processed: 0,
        })
    }

    /// Process f32 audio samples (interleaved).
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved audio samples normalized to -1.0 to 1.0
    pub fn process_f32(&mut self, samples: &[f32]) {
        let f64_samples: Vec<f64> = samples.iter().map(|&s| f64::from(s)).collect();
        self.process_f64(&f64_samples);
    }

    /// Process f64 audio samples (interleaved).
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved audio samples normalized to -1.0 to 1.0
    pub fn process_f64(&mut self, samples: &[f64]) {
        if samples.is_empty() {
            return;
        }

        // Apply K-weighting filter
        let mut filtered = vec![0.0; samples.len()];
        self.filter_bank
            .process_interleaved(samples, self.config.channels, &mut filtered);

        // Process LKFS calculation
        self.lkfs_calculator.process_interleaved(&filtered);

        // Process gating (for integrated loudness)
        self.gating_processor.process_interleaved(&filtered);

        // Process true peak (on original unfiltered samples)
        if let Some(ref mut detector) = self.true_peak_detector {
            detector.process_interleaved(samples);
        }

        self.samples_processed += samples.len() / self.config.channels;
    }

    /// Get current loudness metrics.
    pub fn metrics(&mut self) -> LoudnessMetrics {
        let momentary = if self.config.enable_momentary {
            self.lkfs_calculator.momentary_loudness()
        } else {
            f64::NEG_INFINITY
        };

        let short_term = if self.config.enable_short_term {
            self.lkfs_calculator.short_term_loudness()
        } else {
            f64::NEG_INFINITY
        };

        let integrated = if self.config.enable_integrated {
            self.gating_processor.integrated_loudness()
        } else {
            f64::NEG_INFINITY
        };

        let loudness_range = if let Some(ref mut lra_calc) = self.lra_calculator {
            let blocks = self.gating_processor.get_blocks_for_lra();
            lra_calc.calculate(&blocks)
        } else {
            0.0
        };

        let (true_peak_dbtp, true_peak_linear, channel_peaks_dbtp) =
            if let Some(ref detector) = self.true_peak_detector {
                let peaks = detector.channel_peaks_dbtp();
                let max_peak = detector.true_peak_dbtp();
                let max_linear = detector.true_peak_linear();
                (max_peak, max_linear, peaks)
            } else {
                (f64::NEG_INFINITY, 0.0, vec![])
            };

        LoudnessMetrics {
            momentary_lufs: momentary,
            short_term_lufs: short_term,
            integrated_lufs: integrated,
            loudness_range,
            true_peak_dbtp,
            true_peak_linear,
            max_momentary: self.lkfs_calculator.max_momentary(),
            max_short_term: self.lkfs_calculator.max_short_term(),
            channel_peaks_dbtp,
        }
    }

    /// Check compliance with the configured standard.
    pub fn check_compliance(&mut self) -> ComplianceResult {
        let metrics = self.metrics();
        let standard = &self.config.standard;

        let target = standard.target_lufs();
        let tolerance = standard.tolerance_lu();
        let max_peak = standard.max_true_peak_dbtp();

        let loudness_compliant = if metrics.integrated_lufs.is_finite() {
            metrics.integrated_lufs >= target - tolerance
                && metrics.integrated_lufs <= target + tolerance
        } else {
            false
        };

        let peak_compliant = metrics.true_peak_dbtp <= max_peak;

        let lra_acceptable = metrics.loudness_range >= 1.0 && metrics.loudness_range <= 30.0;

        ComplianceResult {
            standard: *standard,
            loudness_compliant,
            peak_compliant,
            lra_acceptable,
            integrated_lufs: metrics.integrated_lufs,
            true_peak_dbtp: metrics.true_peak_dbtp,
            loudness_range: metrics.loudness_range,
            target_lufs: target,
            max_peak_dbtp: max_peak,
            deviation_lu: if metrics.integrated_lufs.is_finite() {
                metrics.integrated_lufs - target
            } else {
                0.0
            },
        }
    }

    /// Generate a detailed loudness report.
    #[allow(clippy::cast_precision_loss)]
    pub fn generate_report(&mut self) -> LoudnessReport {
        let metrics = self.metrics();
        let compliance = self.check_compliance();
        let duration_seconds = self.samples_processed as f64 / self.config.sample_rate;

        LoudnessReport::new(metrics, compliance, duration_seconds)
    }

    /// Reset the meter to initial state.
    pub fn reset(&mut self) {
        self.lkfs_calculator.reset();
        self.gating_processor.reset();
        if let Some(ref mut detector) = self.true_peak_detector {
            detector.reset();
        }
        if let Some(ref mut lra_calc) = self.lra_calculator {
            lra_calc.reset();
        }
        self.filter_bank.reset();
        self.samples_processed = 0;
    }

    /// Get the meter configuration.
    pub fn config(&self) -> &MeterConfig {
        &self.config
    }

    /// Get the number of samples processed (per channel).
    pub fn samples_processed(&self) -> usize {
        self.samples_processed
    }

    /// Get the duration of processed audio in seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        self.samples_processed as f64 / self.config.sample_rate
    }
}

/// Compliance result.
#[derive(Clone, Debug)]
pub struct ComplianceResult {
    /// Standard being checked.
    pub standard: Standard,
    /// Is loudness compliant?
    pub loudness_compliant: bool,
    /// Is peak compliant?
    pub peak_compliant: bool,
    /// Is LRA acceptable?
    pub lra_acceptable: bool,
    /// Measured integrated loudness.
    pub integrated_lufs: f64,
    /// Measured true peak.
    pub true_peak_dbtp: f64,
    /// Measured loudness range.
    pub loudness_range: f64,
    /// Target loudness.
    pub target_lufs: f64,
    /// Maximum allowed peak.
    pub max_peak_dbtp: f64,
    /// Deviation from target in LU.
    pub deviation_lu: f64,
}

impl ComplianceResult {
    /// Check if fully compliant (loudness and peak).
    pub fn is_compliant(&self) -> bool {
        self.loudness_compliant && self.peak_compliant
    }

    /// Get the standard name.
    pub fn standard_name(&self) -> &str {
        self.standard.name()
    }

    /// Get recommended gain adjustment to meet target.
    ///
    /// Returns gain in dB (positive = increase, negative = decrease).
    pub fn recommended_gain_db(&self) -> f64 {
        if self.integrated_lufs.is_finite() {
            self.target_lufs - self.integrated_lufs
        } else {
            0.0
        }
    }
}
