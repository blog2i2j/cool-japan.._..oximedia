//! Video quality assessment and objective metrics for `OxiMedia`.
//!
//! This crate provides comprehensive video quality assessment tools including
//! full-reference metrics (PSNR, SSIM, VMAF, etc.) and no-reference metrics
//! (NIQE, BRISQUE, blur, noise).
//!
//! # Supported Metrics
//!
//! ## Full-Reference Metrics
//! - **PSNR** - Peak Signal-to-Noise Ratio
//! - **SSIM** - Structural Similarity Index
//! - **MS-SSIM** - Multi-Scale SSIM
//! - **VMAF** - Video Multi-Method Assessment Fusion
//! - **VIF** - Visual Information Fidelity
//! - **FSIM** - Feature Similarity Index
//!
//! ## No-Reference Metrics
//! - **NIQE** - Natural Image Quality Evaluator
//! - **BRISQUE** - Blind/Referenceless Image Spatial Quality Evaluator
//! - **Blockiness** - DCT-based blockiness detection
//! - **Blur** - Laplacian variance and edge width
//! - **Noise** - Spatial/temporal noise estimation
//!
//! # Example
//!
//! ```
//! use oximedia_quality::{QualityAssessor, MetricType};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let assessor = QualityAssessor::new();
//!
//! // Assess quality between reference and distorted frames
//! // let score = assessor.assess(&reference, &distorted, MetricType::SSIM)?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::let_and_return)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::no_effect_underscore_binding)]
#![allow(clippy::format_push_string)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::unused_self)]
#![allow(dead_code)]

mod batch;
mod blockiness;
mod blur;
mod brisque;
mod fsim;
mod msssim;
mod niqe;
mod noise;
mod psnr;
mod reference;
mod ssim;
mod vif;
mod vmaf;

pub mod ab_compare;
pub mod ab_quality_compare;
pub mod aggregate_score;
pub mod artifact_score;
pub mod audio_quality;
pub mod bitrate_quality;
pub mod blockiness_detector;
pub mod blur_detector;
pub mod budget;
pub mod ciede2000;
pub mod codec_quality;
pub mod color_banding;
pub mod color_fidelity;
pub mod compression_artifact;
pub mod compression_artifacts;
pub mod confidence;
pub mod dynamic_range_quality;
pub mod edge_quality;
pub mod flicker_score;
pub mod golden_tests;
pub mod hdr_quality;
pub mod histogram_quality;
pub mod lpips;
pub mod metrics;
pub mod motion_compensated;
pub mod peaq_like;
pub mod perceptual;
pub mod perceptual_model;
pub mod quality_bitrate_curve;
pub mod quality_gate;
pub mod quality_heatmap;
pub mod quality_history;
pub mod quality_preset;
pub mod quality_report;
pub mod realtime_monitor;
pub mod realtime_quality;
pub mod reference_free;
pub mod region_quality;
pub mod scene_quality;
pub mod sharpness_score;
pub mod spatial_quality;
pub mod ssim_simd;
pub mod temporal_quality;
pub mod temporal_stable;
pub mod vmaf_like;
pub mod vmaf_score;

use oximedia_core::{OxiError, OxiResult, PixelFormat};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use batch::{BatchAssessment, BatchResult};
pub use blockiness::BlockinessDetector;
pub use blur::BlurDetector;
pub use brisque::BrisqueAssessor;
pub use fsim::{
    compute_fsim, gradient_magnitude, phase_congruency_approx, scharr_gradient, FsimCalculator,
};
pub use msssim::MsSsimCalculator;
pub use niqe::NiqeAssessor;
pub use noise::NoiseEstimator;
pub use psnr::PsnrCalculator;
pub use quality_gate::{PipelineQualityGate, QualityGateResult};
pub use reference::ReferenceManager;
pub use ssim::SsimCalculator;
pub use temporal_quality::{
    FrameQuality, QualityStats, SceneAwarePooler, SceneAwarePoolingReport, SceneSegmentStats,
    TemporalQualityAnalysisReport, TemporalQualityAnalyzer,
};
pub use vif::VifCalculator;
pub use vmaf::VmafCalculator;

/// Video frame data for quality assessment.
///
/// Represents a single video frame with planar or packed pixel data.
#[derive(Clone, Debug)]
pub struct Frame {
    /// Frame width in pixels
    pub width: usize,
    /// Frame height in pixels
    pub height: usize,
    /// Pixel format
    pub format: PixelFormat,
    /// Plane data (Y, Cb, Cr for YUV or single plane for RGB/Gray)
    pub planes: Vec<Vec<u8>>,
    /// Stride (bytes per row) for each plane
    pub strides: Vec<usize>,
}

impl Frame {
    /// Creates a new frame with the given dimensions and format.
    ///
    /// # Errors
    ///
    /// Returns an error if the format is not supported.
    pub fn new(width: usize, height: usize, format: PixelFormat) -> OxiResult<Self> {
        let plane_count = format.plane_count() as usize;
        let mut planes = Vec::with_capacity(plane_count);
        let mut strides = Vec::with_capacity(plane_count);

        let (h_subsample, v_subsample) = format.chroma_subsampling();
        let bytes_per_component = format.bits_per_component().div_ceil(8);

        for plane_idx in 0..plane_count {
            let (plane_width, plane_height) = if plane_idx == 0 || !format.is_yuv() {
                // Luma plane or non-YUV format
                (width, height)
            } else {
                // Chroma planes
                (width / h_subsample as usize, height / v_subsample as usize)
            };

            let stride = plane_width * bytes_per_component as usize;
            let size = stride * plane_height;
            planes.push(vec![0; size]);
            strides.push(stride);
        }

        Ok(Self {
            width,
            height,
            format,
            planes,
            strides,
        })
    }

    /// Returns the luma (Y) plane data.
    #[must_use]
    pub fn luma(&self) -> &[u8] {
        &self.planes[0]
    }

    /// Returns the luma (Y) plane data as mutable.
    pub fn luma_mut(&mut self) -> &mut [u8] {
        &mut self.planes[0]
    }

    /// Returns the chroma planes (Cb, Cr) if available.
    #[must_use]
    pub fn chroma(&self) -> Option<(&[u8], &[u8])> {
        if self.planes.len() >= 3 {
            Some((&self.planes[1], &self.planes[2]))
        } else {
            None
        }
    }

    /// Converts frame to grayscale (Y plane only).
    #[must_use]
    pub fn to_gray(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            format: PixelFormat::Gray8,
            planes: vec![self.planes[0].clone()],
            strides: vec![self.strides[0]],
        }
    }
}

/// Type of quality metric to compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MetricType {
    /// Peak Signal-to-Noise Ratio
    Psnr,
    /// Structural Similarity Index
    Ssim,
    /// Multi-Scale SSIM
    MsSsim,
    /// Video Multi-Method Assessment Fusion
    Vmaf,
    /// Visual Information Fidelity
    Vif,
    /// Feature Similarity Index
    Fsim,
    /// Natural Image Quality Evaluator (no-reference)
    Niqe,
    /// Blind/Referenceless Image Spatial Quality Evaluator (no-reference)
    Brisque,
    /// Blockiness detection (no-reference)
    Blockiness,
    /// Blur detection (no-reference)
    Blur,
    /// Noise estimation (no-reference)
    Noise,
}

impl MetricType {
    /// Returns true if this metric requires a reference frame.
    #[must_use]
    pub const fn requires_reference(&self) -> bool {
        matches!(
            self,
            Self::Psnr | Self::Ssim | Self::MsSsim | Self::Vmaf | Self::Vif | Self::Fsim
        )
    }

    /// Returns true if this is a no-reference metric.
    #[must_use]
    pub const fn is_no_reference(&self) -> bool {
        matches!(
            self,
            Self::Niqe | Self::Brisque | Self::Blockiness | Self::Blur | Self::Noise
        )
    }
}

/// Quality score result for a single frame or video.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityScore {
    /// Metric type
    pub metric: MetricType,
    /// Overall score
    pub score: f64,
    /// Per-component scores (Y, Cb, Cr for full-reference metrics)
    pub components: HashMap<String, f64>,
    /// Frame number (if per-frame)
    pub frame_num: Option<usize>,
}

impl QualityScore {
    /// Creates a new quality score.
    #[must_use]
    pub fn new(metric: MetricType, score: f64) -> Self {
        Self {
            metric,
            score,
            components: HashMap::new(),
            frame_num: None,
        }
    }

    /// Adds a component score.
    pub fn add_component(&mut self, name: impl Into<String>, score: f64) {
        self.components.insert(name.into(), score);
    }

    /// Sets the frame number.
    #[must_use]
    pub fn with_frame_num(mut self, frame_num: usize) -> Self {
        self.frame_num = Some(frame_num);
        self
    }
}

/// Temporal pooling method for aggregating per-frame scores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingMethod {
    /// Arithmetic mean
    Mean,
    /// Harmonic mean (emphasizes lower scores)
    HarmonicMean,
    /// Minimum score
    Min,
    /// Percentile (e.g., 10th percentile for VMAF)
    Percentile(u8),
}

impl PoolingMethod {
    /// Applies pooling to a sequence of scores.
    #[must_use]
    pub fn apply(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }

        match self {
            Self::Mean => scores.iter().sum::<f64>() / scores.len() as f64,
            Self::HarmonicMean => {
                let sum_reciprocals: f64 = scores.iter().map(|s| 1.0 / s.max(1e-10)).sum();
                scores.len() as f64 / sum_reciprocals
            }
            Self::Min => scores
                .iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0),
            Self::Percentile(p) => {
                let mut sorted = scores.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx = ((f64::from(*p) / 100.0) * sorted.len() as f64) as usize;
                sorted[idx.min(sorted.len() - 1)]
            }
        }
    }
}

/// Main quality assessment interface.
///
/// Provides unified access to all quality metrics.
pub struct QualityAssessor {
    psnr: PsnrCalculator,
    ssim: SsimCalculator,
    msssim: MsSsimCalculator,
    vif: VifCalculator,
    fsim: FsimCalculator,
    niqe: NiqeAssessor,
    brisque: BrisqueAssessor,
    blockiness: BlockinessDetector,
    blur: BlurDetector,
    noise: NoiseEstimator,
    vmaf: VmafCalculator,
}

impl QualityAssessor {
    /// Creates a new quality assessor with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            psnr: PsnrCalculator::new(),
            ssim: SsimCalculator::new(),
            msssim: MsSsimCalculator::new(),
            vif: VifCalculator::new(),
            fsim: FsimCalculator::new(),
            niqe: NiqeAssessor::new(),
            brisque: BrisqueAssessor::new(),
            blockiness: BlockinessDetector::new(),
            blur: BlurDetector::new(),
            noise: NoiseEstimator::new(),
            vmaf: VmafCalculator::new(),
        }
    }

    /// Assesses quality using the specified metric.
    ///
    /// # Errors
    ///
    /// Returns an error if the metric calculation fails or if dimensions mismatch.
    pub fn assess(
        &self,
        reference: &Frame,
        distorted: &Frame,
        metric: MetricType,
    ) -> OxiResult<QualityScore> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(OxiError::InvalidData(
                "Frame dimensions must match".to_string(),
            ));
        }

        match metric {
            MetricType::Psnr => self.psnr.calculate(reference, distorted),
            MetricType::Ssim => self.ssim.calculate(reference, distorted),
            MetricType::MsSsim => self.msssim.calculate(reference, distorted),
            MetricType::Vif => self.vif.calculate(reference, distorted),
            MetricType::Fsim => self.fsim.calculate(reference, distorted),
            MetricType::Vmaf => self.vmaf.calculate(reference, distorted),
            MetricType::Niqe => self.niqe.assess(distorted),
            MetricType::Brisque => self.brisque.assess(distorted),
            MetricType::Blockiness => self.blockiness.detect(distorted),
            MetricType::Blur => self.blur.detect(distorted),
            MetricType::Noise => self.noise.estimate(distorted),
        }
    }

    /// Assesses no-reference quality of a single frame.
    ///
    /// # Errors
    ///
    /// Returns an error if the metric calculation fails.
    pub fn assess_no_reference(
        &self,
        frame: &Frame,
        metric: MetricType,
    ) -> OxiResult<QualityScore> {
        if !metric.is_no_reference() {
            return Err(OxiError::InvalidData(format!(
                "Metric {metric:?} requires a reference frame"
            )));
        }

        match metric {
            MetricType::Niqe => self.niqe.assess(frame),
            MetricType::Brisque => self.brisque.assess(frame),
            MetricType::Blockiness => self.blockiness.detect(frame),
            MetricType::Blur => self.blur.detect(frame),
            MetricType::Noise => self.noise.estimate(frame),
            _ => Err(OxiError::InvalidData(format!(
                "Metric {metric:?} is not a no-reference metric"
            ))),
        }
    }
}

impl Default for QualityAssessor {
    fn default() -> Self {
        Self::new()
    }
}
