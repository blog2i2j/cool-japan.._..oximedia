//! Interlacing and telecine detection for video content.
//!
//! This module provides comprehensive tools for detecting and analyzing interlaced
//! video content and telecine patterns (film-to-video transfer). It includes:
//!
//! - **Interlace detection**: Identifies interlaced vs progressive content
//! - **Comb detection**: Detects combing artifacts from interlacing
//! - **Telecine detection**: Identifies pulldown patterns (3:2, 2:2, etc.)
//! - **Field analysis**: Separates and analyzes video fields
//! - **Pattern matching**: Detects temporal cadence patterns
//!
//! # Examples
//!
//! ## Basic Interlace Detection
//!
//! ```
//! use oximedia_cv::interlace::{InterlaceDetector, InterlaceDetectorConfig};
//! use oximedia_codec::VideoFrame;
//! use oximedia_core::PixelFormat;
//!
//! let config = InterlaceDetectorConfig::default();
//! let detector = InterlaceDetector::new(config);
//!
//! let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
//! frame.allocate();
//!
//! let info = detector.detect_interlacing(&[frame])?;
//! println!("Content type: {:?}", info.content_type);
//! println!("Confidence: {:.2}", info.confidence);
//! ```
//!
//! ## Telecine Detection
//!
//! ```
//! use oximedia_cv::interlace::{TelecineDetector, TelecineDetectorConfig};
//! use oximedia_codec::VideoFrame;
//! use oximedia_core::PixelFormat;
//!
//! let config = TelecineDetectorConfig::default();
//! let mut detector = TelecineDetector::new(config);
//!
//! // Analyze a sequence of frames
//! let mut frames = Vec::new();
//! for _ in 0..30 {
//!     let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
//!     frame.allocate();
//!     frames.push(frame);
//! }
//!
//! let info = detector.detect(&frames)?;
//! if info.is_telecine {
//!     println!("Detected telecine: {:?}", info.pattern);
//!     println!("Confidence: {:.2}", info.confidence);
//! }
//! ```

pub mod comb;
pub mod field;
pub mod metrics;
pub mod pattern;
pub mod telecine;

use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;

pub use comb::{CombDetector, CombDetectorConfig, CombPattern};
pub use field::{Field, FieldAnalyzer, FieldParity};
pub use metrics::{
    ConfidenceLevel, ContentType, DetectionScore, FieldOrder, InterlaceMetrics, TelecineMetrics,
};
pub use pattern::{
    CadenceMapEntry, CadencePattern, FrameDifference, FramePrediction, PatternMatcher,
    PatternValidation, PatternValidator, PulldownPattern,
};
pub use telecine::{
    IvtcMethod, IvtcRecommendation, TelecineDetector, TelecineDetectorConfig, TelecineInfo,
    TelecineStatistics,
};

/// Configuration for the interlace detector.
#[derive(Debug, Clone)]
pub struct InterlaceDetectorConfig {
    /// Comb detection configuration.
    pub comb_config: CombDetectorConfig,
    /// Confidence threshold for interlace detection (0.0-1.0).
    pub confidence_threshold: f64,
    /// Enable field order detection.
    pub detect_field_order: bool,
    /// Minimum frames required for reliable detection.
    pub min_frames: usize,
}

impl InterlaceDetectorConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            comb_config: CombDetectorConfig::default(),
            confidence_threshold: 0.5,
            detect_field_order: true,
            min_frames: 5,
        }
    }

    /// Creates a configuration optimized for sensitivity.
    #[must_use]
    pub fn sensitive() -> Self {
        Self {
            comb_config: CombDetectorConfig::sensitive(),
            confidence_threshold: 0.4,
            detect_field_order: true,
            min_frames: 3,
        }
    }

    /// Creates a configuration optimized for specificity.
    #[must_use]
    pub fn specific() -> Self {
        Self {
            comb_config: CombDetectorConfig::specific(),
            confidence_threshold: 0.6,
            detect_field_order: true,
            min_frames: 7,
        }
    }
}

impl Default for InterlaceDetectorConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Main interlace detector that combines all detection methods.
pub struct InterlaceDetector {
    config: InterlaceDetectorConfig,
    comb_detector: CombDetector,
    field_analyzer: FieldAnalyzer,
}

impl InterlaceDetector {
    /// Creates a new interlace detector with the given configuration.
    #[must_use]
    pub fn new(config: InterlaceDetectorConfig) -> Self {
        let comb_detector = CombDetector::new(config.comb_config.clone());
        let field_analyzer = FieldAnalyzer::new();

        Self {
            config,
            comb_detector,
            field_analyzer,
        }
    }

    /// Creates an interlace detector with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(InterlaceDetectorConfig::default())
    }

    /// Detects interlacing in a sequence of frames.
    ///
    /// Returns comprehensive information about the interlacing status.
    pub fn detect_interlacing(&self, frames: &[VideoFrame]) -> CvResult<InterlaceInfo> {
        if frames.is_empty() {
            return Err(CvError::insufficient_data(1, 0));
        }

        // Analyze with comb detector
        let metrics = if frames.len() >= 2 {
            self.comb_detector.detect_temporal(frames)?
        } else {
            self.comb_detector.detect(&frames[0])?
        };

        // Detect field order if enabled and confidence is high enough
        let field_order =
            if self.config.detect_field_order && frames.len() >= 2 && metrics.confidence() > 0.5 {
                self.field_analyzer.detect_field_order(frames)?
            } else {
                FieldOrder::Unknown
            };

        // Calculate detection scores
        let detection_score = self.calculate_detection_score(&metrics)?;

        let content_type = detection_score.dominant_type();
        let confidence = detection_score.confidence();
        let is_interlaced = content_type == ContentType::Interlaced
            && confidence >= self.config.confidence_threshold;

        Ok(InterlaceInfo {
            is_interlaced,
            content_type,
            confidence,
            field_order,
            metrics,
            detection_score,
        })
    }

    /// Calculates detection scores from metrics.
    fn calculate_detection_score(&self, metrics: &InterlaceMetrics) -> CvResult<DetectionScore> {
        let interlaced_score = metrics.confidence();
        let progressive_score = 1.0 - interlaced_score;

        // Telecine detection requires temporal analysis (not done here)
        let telecine_score = 0.0;

        // Mixed content detection based on variance in metrics
        let metric_variance = self.calculate_metric_variance(metrics);
        let mixed_score = if metric_variance > 0.3 {
            metric_variance
        } else {
            0.0
        };

        Ok(DetectionScore::from_components(
            progressive_score,
            interlaced_score,
            telecine_score,
            mixed_score,
        ))
    }

    /// Calculates variance in metrics to detect mixed content.
    fn calculate_metric_variance(&self, metrics: &InterlaceMetrics) -> f64 {
        let values = [
            metrics.comb_score,
            metrics.field_diff,
            metrics.spatial_comb,
            metrics.temporal_comb,
            metrics.edge_comb,
        ];

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 = values
            .iter()
            .map(|&v| {
                let diff = v - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64;

        variance.sqrt()
    }

    /// Detects interlacing with detailed analysis.
    ///
    /// Provides more detailed information than `detect_interlacing`.
    pub fn detect_detailed(&self, frames: &[VideoFrame]) -> CvResult<DetailedInterlaceInfo> {
        let basic_info = self.detect_interlacing(frames)?;

        // Generate comb map for the most recent frame
        let comb_map = if frames.is_empty() {
            None
        } else {
            Some(
                self.comb_detector
                    .generate_comb_map(&frames[frames.len() - 1])?,
            )
        };

        // Detect comb patterns
        let comb_patterns = if frames.is_empty() {
            Vec::new()
        } else {
            self.comb_detector
                .detect_comb_patterns(&frames[frames.len() - 1])?
        };

        // Analyze field parity
        let field_parities = if frames.len() >= 2 {
            self.field_analyzer.analyze_field_parity(frames)?
        } else {
            Vec::new()
        };

        Ok(DetailedInterlaceInfo {
            basic_info,
            comb_map,
            comb_patterns,
            field_parities,
        })
    }

    /// Recommends a deinterlacing method based on the detected content.
    pub fn recommend_deinterlace(
        &self,
        frames: &[VideoFrame],
    ) -> CvResult<DeinterlaceRecommendation> {
        let info = self.detect_interlacing(frames)?;

        if !info.is_interlaced {
            return Ok(DeinterlaceRecommendation {
                should_deinterlace: false,
                method: DeinterlaceMethod::None,
                confidence: info.confidence,
                field_order: info.field_order,
            });
        }

        // Choose method based on content characteristics
        let method = if info.metrics.temporal_comb > 0.7 {
            // High temporal combing - use motion adaptive
            DeinterlaceMethod::MotionAdaptive
        } else if info.metrics.spatial_comb > 0.6 {
            // High spatial combing - use edge-directed
            DeinterlaceMethod::EdgeDirected
        } else if info.metrics.field_diff > 0.5 {
            // Significant field differences - use field blending
            DeinterlaceMethod::FieldBlend
        } else {
            // Default to line doubling for simple cases
            DeinterlaceMethod::LineDouble
        };

        Ok(DeinterlaceRecommendation {
            should_deinterlace: true,
            method,
            confidence: info.confidence,
            field_order: info.field_order,
        })
    }

    /// Gets the comb detector for advanced analysis.
    #[must_use]
    pub const fn comb_detector(&self) -> &CombDetector {
        &self.comb_detector
    }

    /// Gets the field analyzer for advanced analysis.
    #[must_use]
    pub const fn field_analyzer(&self) -> &FieldAnalyzer {
        &self.field_analyzer
    }
}

impl Default for InterlaceDetector {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Information about detected interlacing.
#[derive(Debug, Clone)]
pub struct InterlaceInfo {
    /// Whether the content is interlaced.
    pub is_interlaced: bool,
    /// Detected content type.
    pub content_type: ContentType,
    /// Detection confidence (0.0-1.0).
    pub confidence: f64,
    /// Detected field order (if interlaced).
    pub field_order: FieldOrder,
    /// Detailed interlacing metrics.
    pub metrics: InterlaceMetrics,
    /// Overall detection scores.
    pub detection_score: DetectionScore,
}

impl InterlaceInfo {
    /// Returns the confidence level as an enum.
    #[must_use]
    pub fn confidence_level(&self) -> ConfidenceLevel {
        ConfidenceLevel::from_value(self.confidence)
    }

    /// Returns true if the detection is highly confident.
    #[must_use]
    pub fn is_confident(&self) -> bool {
        self.confidence >= 0.7
    }
}

/// Detailed interlacing information with additional analysis.
#[derive(Debug, Clone)]
pub struct DetailedInterlaceInfo {
    /// Basic interlacing information.
    pub basic_info: InterlaceInfo,
    /// Comb map showing detected combing artifacts.
    pub comb_map: Option<Vec<u8>>,
    /// Detected comb patterns.
    pub comb_patterns: Vec<CombPattern>,
    /// Field parity analysis.
    pub field_parities: Vec<FieldParity>,
}

impl DetailedInterlaceInfo {
    /// Returns the number of detected comb patterns.
    #[must_use]
    pub fn comb_pattern_count(&self) -> usize {
        self.comb_patterns.len()
    }

    /// Returns the average comb pattern length.
    #[must_use]
    pub fn average_comb_length(&self) -> f64 {
        if self.comb_patterns.is_empty() {
            return 0.0;
        }

        let total_length: usize = self.comb_patterns.iter().map(|p| p.length).sum();
        total_length as f64 / self.comb_patterns.len() as f64
    }
}

/// Recommendation for deinterlacing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeinterlaceRecommendation {
    /// Whether deinterlacing should be applied.
    pub should_deinterlace: bool,
    /// Recommended deinterlacing method.
    pub method: DeinterlaceMethod,
    /// Confidence in the recommendation.
    pub confidence: f64,
    /// Detected field order.
    pub field_order: FieldOrder,
}

/// Deinterlacing method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeinterlaceMethod {
    /// No deinterlacing needed.
    None,
    /// Simple line doubling.
    LineDouble,
    /// Field blending.
    FieldBlend,
    /// Edge-directed interpolation.
    EdgeDirected,
    /// Motion adaptive deinterlacing.
    MotionAdaptive,
    /// Motion compensated deinterlacing.
    MotionCompensated,
}

impl DeinterlaceMethod {
    /// Returns the computational complexity of the method.
    #[must_use]
    pub const fn complexity(&self) -> Complexity {
        match self {
            Self::None => Complexity::None,
            Self::LineDouble | Self::FieldBlend => Complexity::Low,
            Self::EdgeDirected => Complexity::Medium,
            Self::MotionAdaptive => Complexity::High,
            Self::MotionCompensated => Complexity::VeryHigh,
        }
    }
}

/// Computational complexity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Complexity {
    /// No computation required.
    None,
    /// Low complexity.
    Low,
    /// Medium complexity.
    Medium,
    /// High complexity.
    High,
    /// Very high complexity.
    VeryHigh,
}

/// Combined interlace and telecine detector.
pub struct ContentAnalyzer {
    interlace_detector: InterlaceDetector,
    telecine_detector: TelecineDetector,
}

impl ContentAnalyzer {
    /// Creates a new content analyzer with default configurations.
    #[must_use]
    pub fn new() -> Self {
        Self {
            interlace_detector: InterlaceDetector::default(),
            telecine_detector: TelecineDetector::default(),
        }
    }

    /// Creates a content analyzer with custom configurations.
    #[must_use]
    pub fn with_config(
        interlace_config: InterlaceDetectorConfig,
        telecine_config: TelecineDetectorConfig,
    ) -> Self {
        Self {
            interlace_detector: InterlaceDetector::new(interlace_config),
            telecine_detector: TelecineDetector::new(telecine_config),
        }
    }

    /// Analyzes content to detect both interlacing and telecine.
    pub fn analyze(&mut self, frames: &[VideoFrame]) -> CvResult<ContentAnalysis> {
        let interlace_info = self.interlace_detector.detect_interlacing(frames)?;
        let telecine_info = self.telecine_detector.detect(frames)?;

        // Determine overall content type
        let overall_type =
            if telecine_info.is_telecine && telecine_info.confidence > interlace_info.confidence {
                ContentType::Telecine
            } else if interlace_info.is_interlaced {
                ContentType::Interlaced
            } else {
                ContentType::Progressive
            };

        // Generate processing recommendation
        let recommendation = self.generate_recommendation(&interlace_info, &telecine_info)?;

        Ok(ContentAnalysis {
            overall_type,
            interlace_info,
            telecine_info,
            recommendation,
        })
    }

    /// Generates a processing recommendation based on analysis.
    fn generate_recommendation(
        &self,
        interlace_info: &InterlaceInfo,
        telecine_info: &TelecineInfo,
    ) -> CvResult<ProcessingRecommendation> {
        if telecine_info.is_telecine && telecine_info.confidence > 0.6 {
            // Recommend IVTC for telecine content
            Ok(ProcessingRecommendation::Ivtc {
                pattern: telecine_info.pattern,
                confidence: telecine_info.confidence,
            })
        } else if interlace_info.is_interlaced && interlace_info.confidence > 0.5 {
            // Recommend deinterlacing
            let deinterlace_rec = self.interlace_detector.recommend_deinterlace(&[])?;
            Ok(ProcessingRecommendation::Deinterlace {
                method: deinterlace_rec.method,
                confidence: interlace_info.confidence,
            })
        } else {
            Ok(ProcessingRecommendation::None)
        }
    }

    /// Resets both detectors.
    pub fn reset(&mut self) {
        self.telecine_detector.reset();
    }
}

impl Default for ContentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete content analysis result.
#[derive(Debug, Clone)]
pub struct ContentAnalysis {
    /// Overall detected content type.
    pub overall_type: ContentType,
    /// Interlacing information.
    pub interlace_info: InterlaceInfo,
    /// Telecine information.
    pub telecine_info: TelecineInfo,
    /// Processing recommendation.
    pub recommendation: ProcessingRecommendation,
}

/// Recommended processing for the content.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProcessingRecommendation {
    /// No processing needed (progressive content).
    None,
    /// Apply deinterlacing.
    Deinterlace {
        /// Recommended method.
        method: DeinterlaceMethod,
        /// Confidence in recommendation.
        confidence: f64,
    },
    /// Apply inverse telecine (IVTC).
    Ivtc {
        /// Detected pulldown pattern.
        pattern: PulldownPattern,
        /// Confidence in recommendation.
        confidence: f64,
    },
}

impl ProcessingRecommendation {
    /// Returns true if processing is recommended.
    #[must_use]
    pub const fn requires_processing(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Returns the confidence in this recommendation.
    #[must_use]
    pub const fn confidence(&self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Deinterlace { confidence, .. } | Self::Ivtc { confidence, .. } => *confidence,
        }
    }
}
