//! Scene understanding and AI-powered video analysis for `OxiMedia`.
//!
//! `oximedia-scene` provides comprehensive scene understanding and intelligent video
//! analysis capabilities for the `OxiMedia` multimedia framework. This includes:
//!
//! - **Scene Classification**: Classify scenes (indoor/outdoor, day/night, landscape, portrait)
//! - **Object Detection**: Lightweight patent-free object detection
//! - **Activity Recognition**: Recognize activities (walking, running, sports)
//! - **Shot Composition**: Analyze framing (rule of thirds, symmetry, leading lines)
//! - **Semantic Segmentation**: Segment image into semantic regions (sky, ground, people)
//! - **Saliency Detection**: Identify visually important regions
//! - **Aesthetic Scoring**: Rate aesthetic quality of frames
//! - **Event Detection**: Detect events in sports and live content
//! - **Face Detection**: Lightweight face detection (Haar cascades)
//! - **Logo Detection**: Detect brand logos and graphics
//!
//! # Patent-Free Algorithms
//!
//! All algorithms are carefully selected to be patent-free:
//!
//! - **HOG (Histogram of Oriented Gradients)**: Object detection
//! - **Haar Cascades**: Face detection
//! - **Color Histograms**: Scene classification
//! - **Motion Histograms**: Activity recognition
//! - **Spectral Saliency**: Attention prediction
//! - **Graph-based Segmentation**: Semantic regions
//! - **Rule-based Composition**: Framing analysis
//!
//! # Modules
//!
//! - [`classify`]: Scene, content, and quality classification
//! - [`detect`]: Object, face, logo, and text detection
//! - [`activity`]: Activity and sports recognition
//! - [`composition`]: Composition rules, balance, and depth analysis
//! - [`segment`]: Semantic and foreground/background segmentation
//! - [`saliency`]: Saliency detection and attention prediction
//! - [`aesthetic`]: Aesthetic quality scoring and feature extraction
//! - [`event`]: Event detection for sports and live content
//! - [`features`]: Feature extraction and descriptors
//!
//! # Example
//!
//! ```
//! use oximedia_scene::classify::scene::SceneClassifier;
//! use oximedia_scene::detect::face::FaceDetector;
//! use oximedia_scene::composition::rules::CompositionAnalyzer;
//!
//! // Example usage
//! let classifier = SceneClassifier::new();
//! let face_detector = FaceDetector::new();
//! let composition = CompositionAnalyzer::new();
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::unused_self)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::single_match_else)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::manual_swap)]
#![allow(clippy::doc_markdown)]
#![allow(dead_code)]

pub mod action_beat;
pub mod activity;
pub mod adaptive_scene;
pub mod aesthetic;
pub mod audio_visual_correlation;
#[path = "camera_motion/mod.rs"]
pub mod camera_motion;
pub mod classification;
pub mod classify;
pub mod color_temperature;
pub mod complexity_detector;
pub mod composition;
pub mod content_moderation;
pub mod continuity_check;
pub mod crowd_density;
pub mod depth_of_field;
pub mod detect;
pub mod emotion_recognition;
pub mod error;
pub mod event;
pub mod face_landmark;
pub mod features;
pub mod lighting_analysis;
pub mod location;
pub mod mood;
pub mod motion_energy;
pub mod object_tracker;
pub mod pacing;
pub mod saliency;
pub mod scene_boundary;
pub mod scene_captioning;
pub mod scene_graph;
pub mod scene_metadata;
pub mod scene_score;
pub mod scene_stats;
pub mod scene_tags;
pub mod segment;
pub mod segmentation;
pub mod shot_type;
pub mod storyboard;
pub mod summarization;
pub mod temporal_graph;
pub mod text_detect;
pub mod thumbnail_selector;
pub mod transition;
pub mod visual_quality_map;
pub mod visual_rhythm;

// Re-export commonly used items at crate root
pub use error::{SceneError, SceneResult};

/// Common types and utilities used across modules.
pub mod common {
    use serde::{Deserialize, Serialize};

    /// A 2D point in image space.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct Point {
        /// X coordinate.
        pub x: f32,
        /// Y coordinate.
        pub y: f32,
    }

    impl Point {
        /// Create a new point.
        #[must_use]
        pub const fn new(x: f32, y: f32) -> Self {
            Self { x, y }
        }

        /// Calculate distance to another point.
        #[must_use]
        pub fn distance(&self, other: &Self) -> f32 {
            let dx = self.x - other.x;
            let dy = self.y - other.y;
            (dx * dx + dy * dy).sqrt()
        }
    }

    /// A rectangular region in image space.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct Rect {
        /// X coordinate of top-left corner.
        pub x: f32,
        /// Y coordinate of top-left corner.
        pub y: f32,
        /// Width of rectangle.
        pub width: f32,
        /// Height of rectangle.
        pub height: f32,
    }

    impl Rect {
        /// Create a new rectangle.
        #[must_use]
        pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
            Self {
                x,
                y,
                width,
                height,
            }
        }

        /// Calculate area of rectangle.
        #[must_use]
        pub const fn area(&self) -> f32 {
            self.width * self.height
        }

        /// Get center point of rectangle.
        #[must_use]
        pub const fn center(&self) -> Point {
            Point {
                x: self.x + self.width / 2.0,
                y: self.y + self.height / 2.0,
            }
        }

        /// Check if this rectangle contains a point.
        #[must_use]
        pub const fn contains(&self, point: &Point) -> bool {
            point.x >= self.x
                && point.x <= self.x + self.width
                && point.y >= self.y
                && point.y <= self.y + self.height
        }

        /// Calculate intersection over union with another rectangle.
        #[must_use]
        pub fn iou(&self, other: &Self) -> f32 {
            let x1 = self.x.max(other.x);
            let y1 = self.y.max(other.y);
            let x2 = (self.x + self.width).min(other.x + other.width);
            let y2 = (self.y + self.height).min(other.y + other.height);

            if x2 <= x1 || y2 <= y1 {
                return 0.0;
            }

            let intersection = (x2 - x1) * (y2 - y1);
            let union = self.area() + other.area() - intersection;

            if union == 0.0 {
                0.0
            } else {
                intersection / union
            }
        }
    }

    /// Confidence score (0.0 to 1.0).
    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
    pub struct Confidence(f32);

    impl Confidence {
        /// Create a new confidence score (clamped to 0.0-1.0).
        #[must_use]
        pub fn new(value: f32) -> Self {
            Self(value.clamp(0.0, 1.0))
        }

        /// Get the confidence value.
        #[must_use]
        pub const fn value(&self) -> f32 {
            self.0
        }

        /// Check if confidence meets threshold.
        #[must_use]
        pub const fn meets_threshold(&self, threshold: f32) -> bool {
            self.0 >= threshold
        }
    }

    impl From<f32> for Confidence {
        fn from(value: f32) -> Self {
            Self::new(value)
        }
    }

    impl From<Confidence> for f32 {
        fn from(conf: Confidence) -> Self {
            conf.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::common::*;

    #[test]
    fn test_point_distance() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(3.0, 4.0);
        assert!((p1.distance(&p2) - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rect_area() {
        let rect = Rect::new(0.0, 0.0, 10.0, 20.0);
        assert!((rect.area() - 200.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rect_center() {
        let rect = Rect::new(0.0, 0.0, 10.0, 20.0);
        let center = rect.center();
        assert!((center.x - 5.0).abs() < f32::EPSILON);
        assert!((center.y - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rect_contains() {
        let rect = Rect::new(0.0, 0.0, 10.0, 20.0);
        assert!(rect.contains(&Point::new(5.0, 10.0)));
        assert!(!rect.contains(&Point::new(15.0, 10.0)));
    }

    #[test]
    fn test_rect_iou() {
        let rect1 = Rect::new(0.0, 0.0, 10.0, 10.0);
        let rect2 = Rect::new(5.0, 5.0, 10.0, 10.0);
        let iou = rect1.iou(&rect2);
        // Intersection: 5x5 = 25
        // Union: 100 + 100 - 25 = 175
        // IoU: 25/175 ≈ 0.1429
        assert!((iou - 0.1428571).abs() < 0.001);
    }

    #[test]
    fn test_confidence() {
        let conf = Confidence::new(0.75);
        assert!((conf.value() - 0.75).abs() < f32::EPSILON);
        assert!(conf.meets_threshold(0.5));
        assert!(!conf.meets_threshold(0.8));

        // Test clamping
        let conf_high = Confidence::new(1.5);
        assert!((conf_high.value() - 1.0).abs() < f32::EPSILON);

        let conf_low = Confidence::new(-0.5);
        assert!((conf_low.value() - 0.0).abs() < f32::EPSILON);
    }
}
