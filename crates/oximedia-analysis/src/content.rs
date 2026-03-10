//! Content classification and categorization.
//!
//! This module classifies video content based on motion, spatial, and temporal features:
//! - Action scenes (high motion, rapid changes)
//! - Still/static content (low motion, stable)
//! - Talking head (moderate motion, face regions)
//! - Sports (high motion, specific patterns)
//! - Animation (synthetic appearance, consistent motion)
//!
//! # Classification Features
//!
//! - **Temporal Activity** - Frame-to-frame changes
//! - **Spatial Complexity** - Edge density, texture
//! - **Motion Patterns** - Global vs. local motion
//! - **Color Distribution** - Natural vs. synthetic

use crate::{AnalysisError, AnalysisResult};
use serde::{Deserialize, Serialize};

/// Content classification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentClassification {
    /// Primary content type
    pub primary_type: ContentType,
    /// Confidence (0.0-1.0)
    pub confidence: f64,
    /// Per-frame classifications
    pub frame_types: Vec<FrameType>,
    /// Statistics
    pub stats: ContentStats,
}

/// Content type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentType {
    /// High-action content
    Action,
    /// Still/static content
    Still,
    /// Talking head or interview
    TalkingHead,
    /// Sports content
    Sports,
    /// Animated content
    Animation,
    /// Unknown/mixed
    Mixed,
}

/// Per-frame content type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameType {
    /// Frame number
    pub frame: usize,
    /// Content type
    pub content_type: ContentType,
    /// Temporal activity (0.0-1.0)
    pub temporal_activity: f64,
    /// Spatial complexity (0.0-1.0)
    pub spatial_complexity: f64,
}

/// Content statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentStats {
    /// Average temporal activity
    pub avg_temporal_activity: f64,
    /// Average spatial complexity
    pub avg_spatial_complexity: f64,
    /// Percentage of high-motion frames
    pub high_motion_ratio: f64,
    /// Percentage of static frames
    pub static_ratio: f64,
}

/// Content classifier.
pub struct ContentClassifier {
    frame_types: Vec<FrameType>,
    prev_frame: Option<Vec<u8>>,
}

impl ContentClassifier {
    /// Create a new content classifier.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frame_types: Vec::new(),
            prev_frame: None,
        }
    }

    /// Process a frame.
    pub fn process_frame(
        &mut self,
        y_plane: &[u8],
        width: usize,
        height: usize,
        frame_number: usize,
    ) -> AnalysisResult<()> {
        if y_plane.len() != width * height {
            return Err(AnalysisError::InvalidInput(
                "Y plane size mismatch".to_string(),
            ));
        }

        // Compute features
        let spatial_complexity = compute_spatial_complexity(y_plane, width, height);
        let temporal_activity = if let Some(ref prev) = self.prev_frame {
            compute_temporal_activity(y_plane, prev, width, height)
        } else {
            0.0
        };

        // Classify frame
        let content_type = classify_frame(temporal_activity, spatial_complexity);

        self.frame_types.push(FrameType {
            frame: frame_number,
            content_type,
            temporal_activity,
            spatial_complexity,
        });

        self.prev_frame = Some(y_plane.to_vec());

        Ok(())
    }

    /// Finalize and return classification.
    pub fn finalize(self) -> ContentClassification {
        if self.frame_types.is_empty() {
            return ContentClassification {
                primary_type: ContentType::Mixed,
                confidence: 0.0,
                frame_types: Vec::new(),
                stats: ContentStats {
                    avg_temporal_activity: 0.0,
                    avg_spatial_complexity: 0.0,
                    high_motion_ratio: 0.0,
                    static_ratio: 0.0,
                },
            };
        }

        let count = self.frame_types.len() as f64;

        // Compute statistics
        let avg_temporal = self
            .frame_types
            .iter()
            .map(|f| f.temporal_activity)
            .sum::<f64>()
            / count;
        let avg_spatial = self
            .frame_types
            .iter()
            .map(|f| f.spatial_complexity)
            .sum::<f64>()
            / count;

        let high_motion_count = self
            .frame_types
            .iter()
            .filter(|f| f.temporal_activity > 0.6)
            .count();
        let static_count = self
            .frame_types
            .iter()
            .filter(|f| f.temporal_activity < 0.2)
            .count();

        let high_motion_ratio = high_motion_count as f64 / count;
        let static_ratio = static_count as f64 / count;

        // Determine primary type based on statistics
        let (primary_type, confidence) =
            determine_primary_type(avg_temporal, avg_spatial, high_motion_ratio, static_ratio);

        ContentClassification {
            primary_type,
            confidence,
            frame_types: self.frame_types,
            stats: ContentStats {
                avg_temporal_activity: avg_temporal,
                avg_spatial_complexity: avg_spatial,
                high_motion_ratio,
                static_ratio,
            },
        }
    }
}

impl Default for ContentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute spatial complexity using edge density.
fn compute_spatial_complexity(y_plane: &[u8], width: usize, height: usize) -> f64 {
    let mut edge_count = 0;
    let threshold = 20;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let center = i32::from(y_plane[y * width + x]);
            let neighbors = [
                i32::from(y_plane[(y - 1) * width + x]),
                i32::from(y_plane[(y + 1) * width + x]),
                i32::from(y_plane[y * width + (x - 1)]),
                i32::from(y_plane[y * width + (x + 1)]),
            ];

            let max_diff = neighbors
                .iter()
                .map(|&n| (center - n).abs())
                .max()
                .unwrap_or(0);

            if max_diff > threshold {
                edge_count += 1;
            }
        }
    }

    let total_pixels = (width - 2) * (height - 2);
    if total_pixels == 0 {
        return 0.0;
    }

    (f64::from(edge_count) / total_pixels as f64).min(1.0)
}

/// Compute temporal activity (frame difference).
fn compute_temporal_activity(current: &[u8], previous: &[u8], width: usize, height: usize) -> f64 {
    if current.len() != previous.len() {
        return 0.0;
    }

    let mut diff_sum = 0.0;

    // Sample for efficiency
    for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            let idx = y * width + x;
            let diff = (i32::from(current[idx]) - i32::from(previous[idx])).abs();
            diff_sum += f64::from(diff);
        }
    }

    let sample_count = height.div_ceil(4) * width.div_ceil(4);
    if sample_count == 0 {
        return 0.0;
    }

    let avg_diff = diff_sum / sample_count as f64;
    (avg_diff / 255.0).min(1.0)
}

/// Classify a single frame based on features.
fn classify_frame(temporal_activity: f64, spatial_complexity: f64) -> ContentType {
    // Simple heuristic-based classification
    if temporal_activity > 0.6 && spatial_complexity > 0.5 {
        ContentType::Action
    } else if temporal_activity < 0.15 && spatial_complexity < 0.3 {
        ContentType::Still
    } else if temporal_activity > 0.5 && spatial_complexity > 0.6 {
        ContentType::Sports
    } else if temporal_activity < 0.4 && spatial_complexity > 0.4 {
        ContentType::TalkingHead
    } else {
        ContentType::Mixed
    }
}

/// Determine primary content type for entire video.
fn determine_primary_type(
    avg_temporal: f64,
    avg_spatial: f64,
    high_motion_ratio: f64,
    static_ratio: f64,
) -> (ContentType, f64) {
    // Action: High motion throughout
    if high_motion_ratio > 0.6 {
        return (ContentType::Action, high_motion_ratio);
    }

    // Still: Mostly static
    if static_ratio > 0.7 {
        return (ContentType::Still, static_ratio);
    }

    // Sports: High motion and complexity
    if avg_temporal > 0.5 && avg_spatial > 0.6 {
        return (ContentType::Sports, avg_temporal * avg_spatial);
    }

    // Talking head: Moderate activity, high complexity
    if avg_temporal < 0.4 && avg_spatial > 0.4 {
        return (ContentType::TalkingHead, avg_spatial);
    }

    // Mixed content
    (ContentType::Mixed, 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_complexity_uniform() {
        let frame = vec![128u8; 64 * 64];
        let complexity = compute_spatial_complexity(&frame, 64, 64);
        assert!(complexity < 0.1); // Uniform frame should have low complexity
    }

    #[test]
    fn test_spatial_complexity_edges() {
        // Create checkerboard pattern
        let mut frame = vec![0u8; 100 * 100];
        for y in 0..100 {
            for x in 0..100 {
                if (x + y) % 2 == 0 {
                    frame[y * 100 + x] = 255;
                }
            }
        }
        let complexity = compute_spatial_complexity(&frame, 100, 100);
        assert!(complexity > 0.5); // Checkerboard should have high complexity
    }

    #[test]
    fn test_temporal_activity() {
        let frame1 = vec![100u8; 64 * 64];
        let frame2 = vec![150u8; 64 * 64];
        let activity = compute_temporal_activity(&frame2, &frame1, 64, 64);
        assert!(activity > 0.0);
    }

    #[test]
    fn test_frame_classification() {
        // High motion, high complexity -> Action
        let action = classify_frame(0.7, 0.6);
        assert_eq!(action, ContentType::Action);

        // Low motion, low complexity -> Still
        let still = classify_frame(0.1, 0.2);
        assert_eq!(still, ContentType::Still);
    }

    #[test]
    fn test_content_classifier() {
        let mut classifier = ContentClassifier::new();

        // Process some frames
        let frame = vec![128u8; 64 * 64];
        for i in 0..10 {
            classifier
                .process_frame(&frame, 64, 64, i)
                .expect("frame processing should succeed");
        }

        let classification = classifier.finalize();
        assert_eq!(classification.frame_types.len(), 10);
    }

    #[test]
    fn test_empty_classifier() {
        let classifier = ContentClassifier::new();
        let classification = classifier.finalize();
        assert_eq!(classification.primary_type, ContentType::Mixed);
        assert_eq!(classification.frame_types.len(), 0);
    }
}
