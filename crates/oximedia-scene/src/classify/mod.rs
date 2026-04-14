//! Scene and content classification.
//!
//! This module provides various classification algorithms for video frames:
//!
//! - **Scene classification**: Indoor/outdoor, day/night, landscape/portrait
//! - **Content classification**: Sports, news, drama, action
//! - **Quality classification**: Sharp/blurry, noisy, technical quality
//! - **Mood analysis**: Brightness, warmth, contrast, saturation
//! - **Shot type**: Close-up, medium, wide, establishing
//! - **Color palette**: Dominant colors via k-means

pub mod batch;
pub mod color_palette;
pub mod content;
pub mod mood;
pub mod quality;
pub mod scene;
pub mod shot_type;
pub mod temporal_smooth;

pub use batch::{BatchClassifier, BatchConfig, BatchFrameResult, FrameRef};
pub use color_palette::{ColorPalette, ColorPaletteExtractor, PaletteColor, PaletteConfig};
pub use content::{ContentClassifier, ContentType};
pub use mood::{MoodAnalysis, MoodAnalyzer, MoodCategory, MoodFeatures};
pub use quality::{QualityClassifier, QualityMetrics};
pub use scene::{SceneClassifier, SceneType, SmoothedSceneClassifier};
pub use shot_type::{ShotClassification, ShotFeatures, ShotType, ShotTypeClassifier};
pub use temporal_smooth::TemporalSmoother;

use crate::error::{SceneError, SceneResult};

/// Validate that an RGB frame buffer has the correct size.
pub(crate) fn validate_frame(rgb: &[u8], width: usize, height: usize) -> SceneResult<()> {
    let expected = width * height * 3;
    if rgb.len() != expected {
        return Err(SceneError::InvalidDimensions(format!(
            "Expected {} bytes for {}x{} frame, got {}",
            expected,
            width,
            height,
            rgb.len()
        )));
    }
    Ok(())
}
