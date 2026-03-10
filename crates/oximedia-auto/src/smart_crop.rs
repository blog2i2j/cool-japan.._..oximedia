//! Smart cropping using saliency-based region detection.
//!
//! Provides `SalientRegion`, `SmartCropConfig`, `SmartCropResult`, and
//! `SmartCropper` for suggesting crop parameters that keep the most important
//! content visible.

#![allow(dead_code)]

/// An axis-aligned bounding box representing a salient region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SalientRegion {
    /// Left edge in pixels.
    pub x: u32,
    /// Top edge in pixels.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Saliency score in the range 0.0–1.0.
    pub score: f64,
}

impl SalientRegion {
    /// Create a new `SalientRegion`.
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32, score: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
            score,
        }
    }

    /// Return the pixel area of the region.
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Return the centre point of the region.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn centre(&self) -> (f64, f64) {
        let cx = self.x as f64 + self.width as f64 / 2.0;
        let cy = self.y as f64 + self.height as f64 / 2.0;
        (cx, cy)
    }
}

/// Configuration for the smart cropping algorithm.
#[derive(Debug, Clone)]
pub struct SmartCropConfig {
    /// Desired output width in pixels (0 = same as source).
    pub output_width: u32,
    /// Desired output height in pixels (0 = same as source).
    pub output_height: u32,
    /// Minimum saliency score to consider a region important.
    pub min_saliency: f64,
    /// Allow slight upscaling to fill the output dimensions.
    pub allow_upscale: bool,
}

impl Default for SmartCropConfig {
    fn default() -> Self {
        Self {
            output_width: 1280,
            output_height: 720,
            min_saliency: 0.3,
            allow_upscale: false,
        }
    }
}

impl SmartCropConfig {
    /// Return the target aspect ratio as `width / height`.
    ///
    /// Returns `None` if either dimension is zero.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn aspect_ratio(&self) -> Option<f64> {
        if self.output_height == 0 {
            return None;
        }
        Some(self.output_width as f64 / self.output_height as f64)
    }
}

/// The crop rectangle suggested by `SmartCropper`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CropRect {
    /// Left offset in the source frame.
    pub x: u32,
    /// Top offset in the source frame.
    pub y: u32,
    /// Width of the crop.
    pub width: u32,
    /// Height of the crop.
    pub height: u32,
}

/// Result returned by `SmartCropper::suggest_crop`.
#[derive(Debug, Clone)]
pub struct SmartCropResult {
    /// The suggested crop rectangle in source-image coordinates.
    pub crop: CropRect,
    /// Whether any cropping was actually applied (false = full frame).
    pub crop_applied: bool,
    /// The salient regions that influenced the crop decision.
    pub salient_regions: Vec<SalientRegion>,
    /// Confidence of the suggestion in 0.0–1.0.
    pub confidence: f64,
}

impl SmartCropResult {
    /// Return `true` when the crop differs from the full source frame.
    #[must_use]
    pub fn crop_applied(&self) -> bool {
        self.crop_applied
    }

    /// Return the area of the suggested crop rectangle.
    #[must_use]
    pub fn crop_area(&self) -> u64 {
        u64::from(self.crop.width) * u64::from(self.crop.height)
    }
}

/// Analyses frames and suggests optimal crop parameters.
#[derive(Debug, Default)]
pub struct SmartCropper {
    config: SmartCropConfig,
}

impl SmartCropper {
    /// Create a new `SmartCropper` with the given config.
    #[must_use]
    pub fn new(config: SmartCropConfig) -> Self {
        Self { config }
    }

    /// Analyse a set of `SalientRegion` detections and return all that meet
    /// the minimum saliency threshold.
    #[must_use]
    pub fn analyze(&self, regions: &[SalientRegion]) -> Vec<SalientRegion> {
        let mut kept: Vec<SalientRegion> = regions
            .iter()
            .copied()
            .filter(|r| r.score >= self.config.min_saliency)
            .collect();
        // Sort descending by score.
        kept.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        kept
    }

    /// Suggest a crop rectangle for a frame of `src_width` × `src_height`
    /// containing the given `regions`.
    ///
    /// The algorithm computes a weighted centroid of salient regions and
    /// places the output crop rectangle centred on that point, clamped to
    /// source bounds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn suggest_crop(
        &self,
        src_width: u32,
        src_height: u32,
        regions: &[SalientRegion],
    ) -> SmartCropResult {
        let salient = self.analyze(regions);

        // Determine crop size.
        let crop_w = if self.config.output_width == 0 || self.config.output_width > src_width {
            src_width
        } else {
            self.config.output_width
        };
        let crop_h = if self.config.output_height == 0 || self.config.output_height > src_height {
            src_height
        } else {
            self.config.output_height
        };

        // No crop needed.
        if crop_w == src_width && crop_h == src_height {
            return SmartCropResult {
                crop: CropRect {
                    x: 0,
                    y: 0,
                    width: src_width,
                    height: src_height,
                },
                crop_applied: false,
                salient_regions: salient,
                confidence: 1.0,
            };
        }

        // Compute weighted centroid.
        let (cx, cy, total_weight) = if salient.is_empty() {
            // Fall back to frame centre.
            (src_width as f64 / 2.0, src_height as f64 / 2.0, 1.0)
        } else {
            let (wx, wy, w) = salient
                .iter()
                .fold((0.0f64, 0.0f64, 0.0f64), |(ax, ay, aw), r| {
                    let (rx, ry) = r.centre();
                    (ax + rx * r.score, ay + ry * r.score, aw + r.score)
                });
            if w == 0.0 {
                (src_width as f64 / 2.0, src_height as f64 / 2.0, 1.0)
            } else {
                (wx / w, wy / w, w)
            }
        };

        // Centre the crop on the centroid, clamp to source.
        let x = ((cx - crop_w as f64 / 2.0).max(0.0) as u32).min(src_width.saturating_sub(crop_w));
        let y = ((cy - crop_h as f64 / 2.0).max(0.0) as u32).min(src_height.saturating_sub(crop_h));

        let confidence = (total_weight / salient.len().max(1) as f64).min(1.0);

        SmartCropResult {
            crop: CropRect {
                x,
                y,
                width: crop_w,
                height: crop_h,
            },
            crop_applied: true,
            salient_regions: salient,
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_salient_region_area() {
        let r = SalientRegion::new(0, 0, 100, 50, 0.8);
        assert_eq!(r.area(), 5000);
    }

    #[test]
    fn test_salient_region_area_zero() {
        let r = SalientRegion::new(0, 0, 0, 100, 1.0);
        assert_eq!(r.area(), 0);
    }

    #[test]
    fn test_salient_region_centre() {
        let r = SalientRegion::new(10, 20, 100, 80, 0.5);
        let (cx, cy) = r.centre();
        assert!((cx - 60.0).abs() < 1e-6);
        assert!((cy - 60.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_aspect_ratio() {
        let cfg = SmartCropConfig::default();
        let ratio = cfg.aspect_ratio().expect("ratio should be valid");
        assert!((ratio - 16.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_aspect_ratio_zero_height() {
        let cfg = SmartCropConfig {
            output_height: 0,
            ..Default::default()
        };
        assert!(cfg.aspect_ratio().is_none());
    }

    #[test]
    fn test_cropper_analyze_filters_by_saliency() {
        let cfg = SmartCropConfig {
            min_saliency: 0.5,
            ..Default::default()
        };
        let cropper = SmartCropper::new(cfg);
        let regions = vec![
            SalientRegion::new(0, 0, 10, 10, 0.8),
            SalientRegion::new(0, 0, 10, 10, 0.3),
        ];
        let kept = cropper.analyze(&regions);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].score - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_cropper_no_crop_when_output_equals_source() {
        let cfg = SmartCropConfig {
            output_width: 1920,
            output_height: 1080,
            ..Default::default()
        };
        let cropper = SmartCropper::new(cfg);
        let result = cropper.suggest_crop(1920, 1080, &[]);
        assert!(!result.crop_applied());
        assert_eq!(result.crop.width, 1920);
        assert_eq!(result.crop.height, 1080);
    }

    #[test]
    fn test_cropper_suggest_crop_applied() {
        let cfg = SmartCropConfig {
            output_width: 640,
            output_height: 360,
            min_saliency: 0.1,
            ..Default::default()
        };
        let cropper = SmartCropper::new(cfg);
        let regions = vec![SalientRegion::new(900, 400, 200, 200, 0.9)];
        let result = cropper.suggest_crop(1920, 1080, &regions);
        assert!(result.crop_applied());
        assert_eq!(result.crop.width, 640);
        assert_eq!(result.crop.height, 360);
    }

    #[test]
    fn test_cropper_crop_clamped_to_source() {
        let cfg = SmartCropConfig {
            output_width: 1280,
            output_height: 720,
            min_saliency: 0.0,
            ..Default::default()
        };
        let cropper = SmartCropper::new(cfg);
        // Salient region at far right edge
        let regions = vec![SalientRegion::new(1900, 1070, 20, 10, 1.0)];
        let result = cropper.suggest_crop(1920, 1080, &regions);
        assert!(result.crop.x + result.crop.width <= 1920);
        assert!(result.crop.y + result.crop.height <= 1080);
    }

    #[test]
    fn test_crop_result_crop_area() {
        let r = SmartCropResult {
            crop: CropRect {
                x: 0,
                y: 0,
                width: 640,
                height: 360,
            },
            crop_applied: true,
            salient_regions: vec![],
            confidence: 0.9,
        };
        assert_eq!(r.crop_area(), 230_400);
    }
}
