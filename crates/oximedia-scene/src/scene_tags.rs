//! Automatic hierarchical scene tagging with confidence-scored labels.
//!
//! This module generates a rich set of descriptive tags for a video frame using
//! colour analysis, edge statistics, and luminance distribution — all
//! patent-free signal processing.
//!
//! # Taxonomy
//!
//! Tags are organised in a two-level hierarchy:
//!
//! ```text
//! Category (e.g. "environment") → Tag (e.g. "outdoor")
//! ```
//!
//! Multiple tags per category are permitted; each tag carries an independent
//! confidence score so that callers can apply their own threshold.
//!
//! # Algorithms
//!
//! | Feature | Algorithm |
//! |---------|-----------|
//! | Sky detection | Blue-bias luminance in top region |
//! | Indoor/outdoor | Saturation distribution + structural lines |
//! | Day/night | Absolute luminance percentile |
//! | Warm/cool colour | Red-to-blue channel ratio |
//! | High/low contrast | Histogram inter-quartile range |
//! | Busy/calm | Edge pixel density |
//!
//! # Example
//!
//! ```
//! use oximedia_scene::scene_tags::SceneTagger;
//!
//! let tagger = SceneTagger::default();
//! let width = 64usize;
//! let height = 64usize;
//! let rgb = vec![120u8; width * height * 3];
//! let result = tagger.tag(&rgb, width, height).unwrap();
//! assert!(!result.tags.is_empty());
//! ```

use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single scene tag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneTag {
    /// Short identifier, e.g. `"outdoor"`.
    pub name: String,
    /// High-level category, e.g. `"environment"`.
    pub category: String,
    /// Confidence (0.0–1.0).
    pub confidence: f32,
}

/// Complete tagging result for one frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagResult {
    /// All generated tags sorted by descending confidence.
    pub tags: Vec<SceneTag>,
    /// Raw feature vector used to derive the tags (for debugging / logging).
    pub features: FrameTagFeatures,
}

impl TagResult {
    /// Return tags that meet a minimum confidence threshold.
    #[must_use]
    pub fn filter_by_confidence(&self, min_confidence: f32) -> Vec<&SceneTag> {
        self.tags
            .iter()
            .filter(|t| t.confidence >= min_confidence)
            .collect()
    }

    /// Return tags belonging to a given category.
    #[must_use]
    pub fn by_category<'a>(&'a self, category: &str) -> Vec<&'a SceneTag> {
        self.tags
            .iter()
            .filter(|t| t.category == category)
            .collect()
    }

    /// Best (highest confidence) tag for a category, if any.
    #[must_use]
    pub fn best_in_category(&self, category: &str) -> Option<&SceneTag> {
        self.tags
            .iter()
            .filter(|t| t.category == category)
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

/// Intermediate feature vector extracted from the frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameTagFeatures {
    /// Mean luminance (0.0–1.0).
    pub mean_luma: f32,
    /// 10th-percentile luma value (0.0–1.0).
    pub luma_p10: f32,
    /// 90th-percentile luma value (0.0–1.0).
    pub luma_p90: f32,
    /// Mean saturation (0.0–1.0).
    pub mean_saturation: f32,
    /// Fraction of pixels classified as sky-like.
    pub sky_ratio: f32,
    /// Edge pixel density (0.0–1.0).
    pub edge_density: f32,
    /// Red-minus-blue channel bias (−1.0–1.0).
    pub warm_bias: f32,
    /// Fraction of bright pixels (luma > 0.7).
    pub bright_ratio: f32,
    /// Fraction of dark pixels (luma < 0.2).
    pub dark_ratio: f32,
    /// Horizontal line density estimate (0.0–1.0).
    pub h_line_density: f32,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the scene tagger.
#[derive(Debug, Clone)]
pub struct SceneTaggerConfig {
    /// Minimum confidence for a tag to be included (0.0–1.0).
    pub min_confidence: f32,
    /// Maximum number of tags to return (0 = unlimited).
    pub max_tags: usize,
    /// Sub-sample factor for faster processing (1 = full resolution).
    pub subsample: usize,
}

impl Default for SceneTaggerConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.15,
            max_tags: 20,
            subsample: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// SceneTagger
// ---------------------------------------------------------------------------

/// Scene tagger that generates hierarchical tags from RGB image data.
#[derive(Debug, Clone, Default)]
pub struct SceneTagger {
    /// Configuration.
    pub config: SceneTaggerConfig,
}

impl SceneTagger {
    /// Create a tagger with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a tagger with custom configuration.
    #[must_use]
    pub fn with_config(config: SceneTaggerConfig) -> Self {
        Self { config }
    }

    /// Tag a single RGB frame.
    ///
    /// `rgb_data` must be a packed `width × height × 3` byte slice in R,G,B order.
    ///
    /// # Errors
    ///
    /// Returns [`SceneError::InvalidDimensions`] if the slice length does not
    /// match `width × height × 3`.
    pub fn tag(&self, rgb_data: &[u8], width: usize, height: usize) -> SceneResult<TagResult> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(format!(
                "Expected {} bytes, got {}",
                width * height * 3,
                rgb_data.len()
            )));
        }
        if width == 0 || height == 0 {
            return Err(SceneError::InvalidDimensions(
                "Width and height must be non-zero".to_string(),
            ));
        }

        let features = self.extract_features(rgb_data, width, height);
        let mut tags = self.derive_tags(&features);

        // Filter and sort
        tags.retain(|t| t.confidence >= self.config.min_confidence);
        tags.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if self.config.max_tags > 0 {
            tags.truncate(self.config.max_tags);
        }

        Ok(TagResult { tags, features })
    }

    // -----------------------------------------------------------------------
    // Feature extraction
    // -----------------------------------------------------------------------

    fn extract_features(&self, rgb: &[u8], width: usize, height: usize) -> FrameTagFeatures {
        let step = self.config.subsample.max(1);
        let pixels = width * height;

        let mut luma_values: Vec<f32> = Vec::with_capacity(pixels / (step * step) + 1);
        let mut sum_sat = 0.0_f32;
        let mut sum_warm = 0.0_f32;
        let mut sky_count = 0usize;
        let mut edge_count = 0usize;
        let mut h_line_count = 0usize;
        let mut bright_count = 0usize;
        let mut dark_count = 0usize;
        let mut sample_count = 0usize;

        // Sky heuristic: top 30% of frame
        let sky_height = (height as f32 * 0.30) as usize;

        for y in (0..height).step_by(step) {
            for x in (0..width).step_by(step) {
                let base = (y * width + x) * 3;
                if base + 2 >= rgb.len() {
                    continue;
                }
                let r = rgb[base] as f32 / 255.0;
                let g = rgb[base + 1] as f32 / 255.0;
                let b = rgb[base + 2] as f32 / 255.0;

                let luma = 0.299 * r + 0.587 * g + 0.114 * b;
                let sat = pixel_saturation(r, g, b);

                luma_values.push(luma);
                sum_sat += sat;
                sum_warm += r - b;

                if luma > 0.7 {
                    bright_count += 1;
                }
                if luma < 0.2 {
                    dark_count += 1;
                }

                // Sky: top region, blue-dominant, moderately bright
                if y < sky_height && b > r && b > g && luma > 0.35 {
                    sky_count += 1;
                }

                // Edge via luminance gradient with right/down neighbour
                if x + step < width && y + step < height {
                    let right_base = (y * width + x + step) * 3;
                    let down_base = ((y + step) * width + x) * 3;
                    if right_base + 2 < rgb.len() && down_base + 2 < rgb.len() {
                        let r_r = rgb[right_base] as f32 / 255.0;
                        let g_r = rgb[right_base + 1] as f32 / 255.0;
                        let b_r = rgb[right_base + 2] as f32 / 255.0;
                        let luma_r = 0.299 * r_r + 0.587 * g_r + 0.114 * b_r;

                        let r_d = rgb[down_base] as f32 / 255.0;
                        let g_d = rgb[down_base + 1] as f32 / 255.0;
                        let b_d = rgb[down_base + 2] as f32 / 255.0;
                        let luma_d = 0.299 * r_d + 0.587 * g_d + 0.114 * b_d;

                        let gx = (luma_r - luma).abs();
                        let gy = (luma_d - luma).abs();
                        let mag = (gx * gx + gy * gy).sqrt();
                        if mag > 0.15 {
                            edge_count += 1;
                        }
                        // Horizontal line: strong vertical gradient, weak horizontal
                        if gy > 0.15 && gx < 0.05 {
                            h_line_count += 1;
                        }
                    }
                }

                sample_count += 1;
            }
        }

        let n = sample_count.max(1) as f32;
        let mean_luma = if luma_values.is_empty() {
            0.0
        } else {
            luma_values.iter().copied().sum::<f32>() / luma_values.len() as f32
        };

        // Sort for percentiles
        luma_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let luma_p10 = percentile_sorted(&luma_values, 0.10);
        let luma_p90 = percentile_sorted(&luma_values, 0.90);

        let sky_samples_in_top =
            ((sky_height / step.max(1)).max(1) * (width / step.max(1)).max(1)) as f32;
        let sky_ratio = (sky_count as f32 / sky_samples_in_top).min(1.0);

        FrameTagFeatures {
            mean_luma,
            luma_p10,
            luma_p90,
            mean_saturation: (sum_sat / n).min(1.0),
            sky_ratio,
            edge_density: (edge_count as f32 / n).min(1.0),
            warm_bias: (sum_warm / n).clamp(-1.0, 1.0),
            bright_ratio: bright_count as f32 / n,
            dark_ratio: dark_count as f32 / n,
            h_line_density: (h_line_count as f32 / n).min(1.0),
        }
    }

    // -----------------------------------------------------------------------
    // Tag derivation
    // -----------------------------------------------------------------------

    fn derive_tags(&self, f: &FrameTagFeatures) -> Vec<SceneTag> {
        let mut tags = Vec::with_capacity(16);

        // ── Environment ──────────────────────────────────────────────────
        {
            // Outdoor: sky present OR high saturation + structured lines
            let outdoor_conf =
                sigmoid(f.sky_ratio * 2.0 + f.mean_saturation * 0.5 + f.h_line_density * 0.3 - 0.6);
            tags.push(tag("outdoor", "environment", outdoor_conf));
            tags.push(tag("indoor", "environment", 1.0 - outdoor_conf));
        }

        // ── Lighting ─────────────────────────────────────────────────────
        {
            // Day: high mean luma, not dominated by darks
            let day_conf = sigmoid(f.mean_luma * 4.0 - 1.5 - f.dark_ratio * 3.0);
            tags.push(tag("day", "lighting", day_conf));
            tags.push(tag("night", "lighting", 1.0 - day_conf));

            let bright_conf = sigmoid(f.bright_ratio * 6.0 - 2.5);
            tags.push(tag("bright", "lighting", bright_conf));
            tags.push(tag("dark", "lighting", sigmoid(f.dark_ratio * 6.0 - 2.5)));

            // High-key / Low-key
            if f.luma_p10 > 0.5 {
                tags.push(tag(
                    "high_key",
                    "lighting",
                    sigmoid((f.luma_p10 - 0.5) * 8.0),
                ));
            }
            if f.luma_p90 < 0.4 {
                tags.push(tag(
                    "low_key",
                    "lighting",
                    sigmoid((0.4 - f.luma_p90) * 8.0),
                ));
            }
        }

        // ── Colour temperature ────────────────────────────────────────────
        {
            let warm_conf = sigmoid(f.warm_bias * 4.0);
            tags.push(tag("warm_tones", "colour", warm_conf));
            tags.push(tag("cool_tones", "colour", 1.0 - warm_conf));

            let vivid_conf = sigmoid(f.mean_saturation * 5.0 - 1.5);
            tags.push(tag("vivid", "colour", vivid_conf));
            tags.push(tag("desaturated", "colour", 1.0 - vivid_conf));
        }

        // ── Contrast ─────────────────────────────────────────────────────
        {
            let contrast = f.luma_p90 - f.luma_p10;
            let high_contrast_conf = sigmoid(contrast * 6.0 - 2.0);
            tags.push(tag("high_contrast", "contrast", high_contrast_conf));
            tags.push(tag("low_contrast", "contrast", 1.0 - high_contrast_conf));
        }

        // ── Scene complexity ──────────────────────────────────────────────
        {
            let busy_conf = sigmoid(f.edge_density * 8.0 - 2.0);
            tags.push(tag("busy", "complexity", busy_conf));
            tags.push(tag("minimalist", "complexity", 1.0 - busy_conf));
        }

        // ── Sky / Nature ──────────────────────────────────────────────────
        if f.sky_ratio > 0.1 {
            tags.push(tag("sky", "subject", sigmoid(f.sky_ratio * 4.0 - 0.4)));
        }

        tags
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tag(name: &str, category: &str, confidence: f32) -> SceneTag {
    SceneTag {
        name: name.to_string(),
        category: category.to_string(),
        confidence: confidence.clamp(0.0, 1.0),
    }
}

/// Logistic sigmoid for mapping a score to [0, 1].
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// HSV-style saturation for an RGB triple, all in 0.0–1.0.
fn pixel_saturation(r: f32, g: f32, b: f32) -> f32 {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    if max < 1e-6 {
        0.0
    } else {
        (max - min) / max
    }
}

/// Get the value at a given fraction in a sorted slice.
fn percentile_sorted(sorted: &[f32], fraction: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f32 * fraction.clamp(0.0, 1.0)) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(r: u8, g: u8, b: u8, w: usize, h: usize) -> Vec<u8> {
        (0..w * h).flat_map(|_| [r, g, b]).collect()
    }

    fn gradient_rgb(w: usize, h: usize) -> Vec<u8> {
        (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    let v = ((x + y) % 256) as u8;
                    [v, (255 - v), v / 2]
                })
            })
            .collect()
    }

    #[test]
    fn test_tag_returns_tags() {
        let tagger = SceneTagger::new();
        let rgb = solid_rgb(200, 220, 240, 64, 64);
        let result = tagger.tag(&rgb, 64, 64).unwrap();
        assert!(!result.tags.is_empty());
    }

    #[test]
    fn test_wrong_size_returns_error() {
        let tagger = SceneTagger::new();
        let rgb = vec![0u8; 10]; // too short
        let err = tagger.tag(&rgb, 64, 64);
        assert!(err.is_err());
    }

    #[test]
    fn test_zero_dimensions_error() {
        let tagger = SceneTagger::new();
        let rgb = vec![];
        let err = tagger.tag(&rgb, 0, 0);
        assert!(err.is_err());
    }

    #[test]
    fn test_dark_frame_tagged_night() {
        let tagger = SceneTagger::new();
        let rgb = solid_rgb(5, 5, 5, 64, 64);
        let result = tagger.tag(&rgb, 64, 64).unwrap();
        let night = result.tags.iter().find(|t| t.name == "night");
        assert!(night.is_some(), "dark frame should produce night tag");
        assert!(night.unwrap().confidence > 0.5);
    }

    #[test]
    fn test_bright_frame_tagged_day() {
        let tagger = SceneTagger::new();
        let rgb = solid_rgb(230, 230, 230, 64, 64);
        let result = tagger.tag(&rgb, 64, 64).unwrap();
        let day = result.tags.iter().find(|t| t.name == "day");
        assert!(day.is_some(), "bright frame should produce day tag");
        assert!(day.unwrap().confidence > 0.5);
    }

    #[test]
    fn test_warm_frame() {
        let tagger = SceneTagger::new();
        let rgb = solid_rgb(240, 120, 20, 64, 64);
        let result = tagger.tag(&rgb, 64, 64).unwrap();
        let warm = result.tags.iter().find(|t| t.name == "warm_tones");
        assert!(warm.is_some());
        assert!(warm.unwrap().confidence > 0.5);
    }

    #[test]
    fn test_tags_sorted_by_confidence() {
        let tagger = SceneTagger::new();
        let rgb = gradient_rgb(64, 64);
        let result = tagger.tag(&rgb, 64, 64).unwrap();
        for pair in result.tags.windows(2) {
            assert!(
                pair[0].confidence >= pair[1].confidence,
                "tags must be sorted descending"
            );
        }
    }

    #[test]
    fn test_filter_by_confidence() {
        let tagger = SceneTagger::new();
        let rgb = gradient_rgb(64, 64);
        let result = tagger.tag(&rgb, 64, 64).unwrap();
        let filtered = result.filter_by_confidence(0.5);
        for t in &filtered {
            assert!(t.confidence >= 0.5);
        }
    }

    #[test]
    fn test_by_category() {
        let tagger = SceneTagger::new();
        let rgb = solid_rgb(180, 180, 180, 64, 64);
        let result = tagger.tag(&rgb, 64, 64).unwrap();
        let env_tags = result.by_category("environment");
        assert!(
            !env_tags.is_empty(),
            "should have environment category tags"
        );
        for t in env_tags {
            assert_eq!(t.category, "environment");
        }
    }

    #[test]
    fn test_best_in_category() {
        let tagger = SceneTagger::new();
        let rgb = gradient_rgb(64, 64);
        let result = tagger.tag(&rgb, 64, 64).unwrap();
        let best = result.best_in_category("lighting");
        assert!(best.is_some(), "should find best lighting tag");
    }

    #[test]
    fn test_max_tags_limit() {
        let config = SceneTaggerConfig {
            min_confidence: 0.0,
            max_tags: 3,
            subsample: 1,
        };
        let tagger = SceneTagger::with_config(config);
        let rgb = gradient_rgb(64, 64);
        let result = tagger.tag(&rgb, 64, 64).unwrap();
        assert!(result.tags.len() <= 3);
    }
}
