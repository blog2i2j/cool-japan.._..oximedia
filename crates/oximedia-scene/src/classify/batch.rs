//! Batch frame processing for classification.
//!
//! Amortizes model/classifier initialization across multiple frames by
//! instantiating classifiers once and running them over an entire batch.
//! Supports parallel processing via rayon when the batch is large enough.

use crate::classify::content::{ContentClassifier, ContentType};
use crate::classify::quality::{QualityClassifier, QualityMetrics};
use crate::classify::scene::{SceneClassifier, SceneType};
use crate::error::{SceneError, SceneResult};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Configuration for batch classification.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Whether to run scene classification.
    pub classify_scene: bool,
    /// Whether to run quality classification.
    pub classify_quality: bool,
    /// Whether to run content classification.
    pub classify_content: bool,
    /// Minimum batch size to trigger parallel processing.
    pub parallel_threshold: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            classify_scene: true,
            classify_quality: true,
            classify_content: true,
            parallel_threshold: 4,
        }
    }
}

/// Classification results for a single frame in a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchFrameResult {
    /// Frame index within the batch.
    pub frame_index: usize,
    /// Scene type classification result (if enabled).
    pub scene_type: Option<SceneType>,
    /// Quality metrics (if enabled).
    pub quality: Option<QualityMetrics>,
    /// Content type classification (if enabled).
    pub content_type: Option<ContentType>,
}

/// A frame reference for batch processing.
///
/// Holds a reference to RGB data and its dimensions.
pub struct FrameRef<'a> {
    /// RGB pixel data.
    pub rgb_data: &'a [u8],
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl<'a> FrameRef<'a> {
    /// Create a new frame reference.
    #[must_use]
    pub fn new(rgb_data: &'a [u8], width: usize, height: usize) -> Self {
        Self {
            rgb_data,
            width,
            height,
        }
    }
}

/// Batch classifier that amortizes initialization across multiple frames.
///
/// Classifiers are constructed once at creation time and reused for every
/// frame in the batch. For batches larger than `parallel_threshold` the
/// per-frame work is distributed across rayon threads.
pub struct BatchClassifier {
    config: BatchConfig,
    scene_classifier: SceneClassifier,
    quality_classifier: QualityClassifier,
    content_classifier: ContentClassifier,
}

impl BatchClassifier {
    /// Create a batch classifier with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: BatchConfig::default(),
            scene_classifier: SceneClassifier::new(),
            quality_classifier: QualityClassifier::new(),
            content_classifier: ContentClassifier::new(),
        }
    }

    /// Create a batch classifier with custom configuration.
    #[must_use]
    pub fn with_config(config: BatchConfig) -> Self {
        Self {
            config,
            scene_classifier: SceneClassifier::new(),
            quality_classifier: QualityClassifier::new(),
            content_classifier: ContentClassifier::new(),
        }
    }

    /// Classify a single frame (internal helper).
    fn classify_frame(
        &self,
        frame: &FrameRef<'_>,
        frame_index: usize,
    ) -> SceneResult<BatchFrameResult> {
        let expected = frame.width * frame.height * 3;
        if frame.rgb_data.len() != expected {
            return Err(SceneError::InvalidDimensions(format!(
                "Frame {}: expected {} bytes for {}x{}, got {}",
                frame_index,
                expected,
                frame.width,
                frame.height,
                frame.rgb_data.len()
            )));
        }

        let scene_type = if self.config.classify_scene {
            Some(
                self.scene_classifier
                    .classify(frame.rgb_data, frame.width, frame.height)?
                    .scene_type,
            )
        } else {
            None
        };

        let quality = if self.config.classify_quality {
            Some(
                self.quality_classifier
                    .analyze(frame.rgb_data, frame.width, frame.height)?,
            )
        } else {
            None
        };

        let content_type = if self.config.classify_content {
            // ContentClassifier requires at least 3 frames for temporal analysis;
            // when classifying a single frame we replicate it to meet the minimum.
            let frames_slice: Vec<&[u8]> = vec![frame.rgb_data; 3];
            Some(
                self.content_classifier
                    .classify(&frames_slice, frame.width, frame.height)?
                    .content_type,
            )
        } else {
            None
        };

        Ok(BatchFrameResult {
            frame_index,
            scene_type,
            quality,
            content_type,
        })
    }

    /// Classify a batch of frames.
    ///
    /// When the number of frames exceeds `parallel_threshold` the work is
    /// distributed across rayon threads. Otherwise frames are processed
    /// sequentially.
    ///
    /// # Errors
    ///
    /// Returns an error if any frame has invalid dimensions.
    pub fn classify_batch(&self, frames: &[FrameRef<'_>]) -> SceneResult<Vec<BatchFrameResult>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        if frames.len() >= self.config.parallel_threshold {
            self.classify_batch_parallel(frames)
        } else {
            self.classify_batch_sequential(frames)
        }
    }

    /// Sequential batch classification.
    fn classify_batch_sequential(
        &self,
        frames: &[FrameRef<'_>],
    ) -> SceneResult<Vec<BatchFrameResult>> {
        let mut results = Vec::with_capacity(frames.len());
        for (i, frame) in frames.iter().enumerate() {
            results.push(self.classify_frame(frame, i)?);
        }
        Ok(results)
    }

    /// Parallel batch classification via rayon.
    fn classify_batch_parallel(
        &self,
        frames: &[FrameRef<'_>],
    ) -> SceneResult<Vec<BatchFrameResult>> {
        let results: Vec<SceneResult<BatchFrameResult>> = frames
            .par_iter()
            .enumerate()
            .map(|(i, frame)| self.classify_frame(frame, i))
            .collect();

        let mut collected = Vec::with_capacity(results.len());
        for r in results {
            collected.push(r?);
        }
        Ok(collected)
    }

    /// Return a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }
}

impl Default for BatchClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_frame(w: usize, h: usize, v: u8) -> Vec<u8> {
        vec![v; w * h * 3]
    }

    fn gradient_frame(w: usize, h: usize) -> Vec<u8> {
        let mut data = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                data[idx] = (x * 255 / w.max(1)) as u8;
                data[idx + 1] = (y * 255 / h.max(1)) as u8;
                data[idx + 2] = 128;
            }
        }
        data
    }

    // 1. Empty batch returns empty results
    #[test]
    fn test_batch_empty() {
        let classifier = BatchClassifier::new();
        let results = classifier.classify_batch(&[]);
        assert!(results.is_ok());
        let r = results.expect("should succeed");
        assert!(r.is_empty());
    }

    // 2. Single frame batch
    #[test]
    fn test_batch_single_frame() {
        let classifier = BatchClassifier::new();
        let data = uniform_frame(100, 100, 128);
        let frames = vec![FrameRef::new(&data, 100, 100)];
        let results = classifier.classify_batch(&frames);
        assert!(results.is_ok());
        let r = results.expect("should succeed");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].frame_index, 0);
        assert!(r[0].scene_type.is_some());
        assert!(r[0].quality.is_some());
        assert!(r[0].content_type.is_some());
    }

    // 3. Multiple frames (sequential path)
    #[test]
    fn test_batch_sequential() {
        let config = BatchConfig {
            parallel_threshold: 100, // force sequential
            ..Default::default()
        };
        let classifier = BatchClassifier::with_config(config);
        let d1 = uniform_frame(50, 50, 100);
        let d2 = uniform_frame(50, 50, 200);
        let d3 = gradient_frame(50, 50);
        let frames = vec![
            FrameRef::new(&d1, 50, 50),
            FrameRef::new(&d2, 50, 50),
            FrameRef::new(&d3, 50, 50),
        ];
        let results = classifier.classify_batch(&frames);
        assert!(results.is_ok());
        let r = results.expect("should succeed");
        assert_eq!(r.len(), 3);
        for (i, res) in r.iter().enumerate() {
            assert_eq!(res.frame_index, i);
        }
    }

    // 4. Parallel path triggers for large batches
    #[test]
    fn test_batch_parallel() {
        let config = BatchConfig {
            parallel_threshold: 2, // force parallel for >= 2
            ..Default::default()
        };
        let classifier = BatchClassifier::with_config(config);
        let frames_data: Vec<Vec<u8>> = (0..5)
            .map(|v| uniform_frame(40, 40, (v * 50) as u8))
            .collect();
        let frames: Vec<FrameRef<'_>> = frames_data
            .iter()
            .map(|d| FrameRef::new(d, 40, 40))
            .collect();
        let results = classifier.classify_batch(&frames);
        assert!(results.is_ok());
        let r = results.expect("should succeed");
        assert_eq!(r.len(), 5);
    }

    // 5. Invalid dimensions produce error
    #[test]
    fn test_batch_invalid_dimensions() {
        let classifier = BatchClassifier::new();
        let data = vec![0u8; 10]; // too small for 100x100
        let frames = vec![FrameRef::new(&data, 100, 100)];
        let results = classifier.classify_batch(&frames);
        assert!(results.is_err());
    }

    // 6. Selective classification (scene only)
    #[test]
    fn test_batch_scene_only() {
        let config = BatchConfig {
            classify_scene: true,
            classify_quality: false,
            classify_content: false,
            parallel_threshold: 10,
        };
        let classifier = BatchClassifier::with_config(config);
        let data = uniform_frame(60, 60, 128);
        let frames = vec![FrameRef::new(&data, 60, 60)];
        let results = classifier.classify_batch(&frames);
        assert!(results.is_ok());
        let r = results.expect("should succeed");
        assert!(r[0].scene_type.is_some());
        assert!(r[0].quality.is_none());
        assert!(r[0].content_type.is_none());
    }

    // 7. Quality only
    #[test]
    fn test_batch_quality_only() {
        let config = BatchConfig {
            classify_scene: false,
            classify_quality: true,
            classify_content: false,
            parallel_threshold: 10,
        };
        let classifier = BatchClassifier::with_config(config);
        let data = gradient_frame(80, 80);
        let frames = vec![FrameRef::new(&data, 80, 80)];
        let results = classifier.classify_batch(&frames);
        assert!(results.is_ok());
        let r = results.expect("should succeed");
        assert!(r[0].scene_type.is_none());
        assert!(r[0].quality.is_some());
        assert!(r[0].content_type.is_none());
    }

    // 8. Default config values
    #[test]
    fn test_batch_config_defaults() {
        let config = BatchConfig::default();
        assert!(config.classify_scene);
        assert!(config.classify_quality);
        assert!(config.classify_content);
        assert_eq!(config.parallel_threshold, 4);
    }

    // 9. Frame indices are correct in parallel mode
    #[test]
    fn test_batch_parallel_frame_indices() {
        let config = BatchConfig {
            parallel_threshold: 1, // always parallel
            ..Default::default()
        };
        let classifier = BatchClassifier::with_config(config);
        let frames_data: Vec<Vec<u8>> = (0..8)
            .map(|v| uniform_frame(30, 30, (v * 30) as u8))
            .collect();
        let frames: Vec<FrameRef<'_>> = frames_data
            .iter()
            .map(|d| FrameRef::new(d, 30, 30))
            .collect();
        let results = classifier.classify_batch(&frames).expect("should succeed");
        // All indices should be present (0..8)
        let mut indices: Vec<usize> = results.iter().map(|r| r.frame_index).collect();
        indices.sort_unstable();
        assert_eq!(indices, (0..8).collect::<Vec<_>>());
    }

    // 10. Config accessor
    #[test]
    fn test_batch_config_accessor() {
        let config = BatchConfig {
            classify_scene: false,
            classify_quality: false,
            classify_content: true,
            parallel_threshold: 42,
        };
        let classifier = BatchClassifier::with_config(config);
        assert!(!classifier.config().classify_scene);
        assert!(!classifier.config().classify_quality);
        assert!(classifier.config().classify_content);
        assert_eq!(classifier.config().parallel_threshold, 42);
    }
}
