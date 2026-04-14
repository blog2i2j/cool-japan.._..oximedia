//! Multi-resolution pyramid processing for detection.
//!
//! Instead of running detectors at full resolution, this module builds an
//! image pyramid (sequence of downscaled images) and runs a lightweight
//! edge-density scan at each level. Detections from coarser levels are
//! mapped back to full-resolution coordinates and merged via NMS.
//!
//! This dramatically reduces computation for large frames where objects of
//! interest may appear at various scales.

use crate::common::Rect;
use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Configuration for the image pyramid.
#[derive(Debug, Clone)]
pub struct PyramidConfig {
    /// Number of levels in the pyramid (including the original).
    pub num_levels: usize,
    /// Scale factor between successive levels (0 < factor < 1).
    pub scale_factor: f32,
    /// Minimum image dimension (width or height) to stop building levels.
    pub min_dimension: usize,
    /// Edge density threshold for a block to be considered a detection.
    pub edge_threshold: f32,
    /// Block size (in pixels at that pyramid level) used for scanning.
    pub block_size: usize,
    /// IoU threshold for NMS across pyramid levels.
    pub nms_iou_threshold: f32,
}

impl Default for PyramidConfig {
    fn default() -> Self {
        Self {
            num_levels: 4,
            scale_factor: 0.5,
            min_dimension: 32,
            edge_threshold: 0.08,
            block_size: 16,
            nms_iou_threshold: 0.4,
        }
    }
}

/// A single level in the image pyramid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyramidLevel {
    /// Level index (0 = original resolution).
    pub level: usize,
    /// Width at this level.
    pub width: usize,
    /// Height at this level.
    pub height: usize,
    /// Scale relative to the original image.
    pub scale: f32,
}

/// Detection from the pyramid with original-resolution coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyramidDetection {
    /// Bounding box in original-image coordinates.
    pub bbox: Rect,
    /// Detection confidence (edge density).
    pub confidence: f32,
    /// Pyramid level where the detection originated.
    pub source_level: usize,
}

/// Multi-resolution pyramid detector.
///
/// Builds an image pyramid and scans each level with a block-based edge
/// density measure. Candidate blocks are mapped back to original
/// coordinates and merged via greedy NMS.
pub struct PyramidDetector {
    config: PyramidConfig,
}

impl PyramidDetector {
    /// Create a pyramid detector with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: PyramidConfig::default(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: PyramidConfig) -> Self {
        Self { config }
    }

    /// Build the pyramid level descriptors (does not allocate pixel data).
    #[must_use]
    pub fn build_levels(&self, width: usize, height: usize) -> Vec<PyramidLevel> {
        let mut levels = Vec::with_capacity(self.config.num_levels);
        let mut w = width;
        let mut h = height;
        let mut scale = 1.0_f32;

        for i in 0..self.config.num_levels {
            if w < self.config.min_dimension || h < self.config.min_dimension {
                break;
            }
            levels.push(PyramidLevel {
                level: i,
                width: w,
                height: h,
                scale,
            });
            w = ((w as f32 * self.config.scale_factor) as usize).max(1);
            h = ((h as f32 * self.config.scale_factor) as usize).max(1);
            scale *= self.config.scale_factor;
        }

        levels
    }

    /// Downsample an RGB image by the given factor using box averaging.
    ///
    /// Returns `(downsampled_data, new_width, new_height)`.
    fn downsample(rgb: &[u8], src_w: usize, src_h: usize, factor: f32) -> (Vec<u8>, usize, usize) {
        let dst_w = ((src_w as f32 * factor) as usize).max(1);
        let dst_h = ((src_h as f32 * factor) as usize).max(1);
        let mut out = vec![0u8; dst_w * dst_h * 3];

        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let sx = ((dx as f32 / factor) as usize).min(src_w - 1);
                let sy = ((dy as f32 / factor) as usize).min(src_h - 1);
                let src_idx = (sy * src_w + sx) * 3;
                let dst_idx = (dy * dst_w + dx) * 3;
                out[dst_idx] = rgb[src_idx];
                out[dst_idx + 1] = rgb[src_idx + 1];
                out[dst_idx + 2] = rgb[src_idx + 2];
            }
        }

        (out, dst_w, dst_h)
    }

    /// Compute edge density of a block.
    fn block_edge_density(
        rgb: &[u8],
        width: usize,
        height: usize,
        bx: usize,
        by: usize,
        bw: usize,
        bh: usize,
    ) -> f32 {
        let mut edge_sum = 0.0_f64;
        let mut count = 0_u64;

        let x_end = (bx + bw).min(width.saturating_sub(1));
        let y_end = (by + bh).min(height.saturating_sub(1));

        for y in by..y_end {
            for x in bx..x_end {
                let idx = (y * width + x) * 3;
                let idx_right = (y * width + x + 1) * 3;
                let idx_below = ((y + 1) * width + x) * 3;

                if idx_right + 2 < rgb.len() && idx_below + 2 < rgb.len() {
                    let mut diff = 0.0_f32;
                    for c in 0..3 {
                        diff += (rgb[idx + c] as f32 - rgb[idx_right + c] as f32).abs();
                        diff += (rgb[idx + c] as f32 - rgb[idx_below + c] as f32).abs();
                    }
                    edge_sum += (diff / 6.0 / 255.0) as f64;
                    count += 1;
                }
            }
        }

        if count > 0 {
            (edge_sum / count as f64) as f32
        } else {
            0.0
        }
    }

    /// Detect regions of interest across all pyramid levels.
    ///
    /// # Errors
    ///
    /// Returns error if the input dimensions are inconsistent.
    pub fn detect(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<Vec<PyramidDetection>> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        let levels = self.build_levels(width, height);
        let mut all_detections: Vec<PyramidDetection> = Vec::new();

        let mut current_rgb = rgb_data.to_vec();
        let mut cur_w = width;
        let mut cur_h = height;

        for level_info in &levels {
            // Scan blocks at this level
            let bs = self.config.block_size;
            let blocks_x = (cur_w / bs).max(1);
            let blocks_y = (cur_h / bs).max(1);

            for by_idx in 0..blocks_y {
                for bx_idx in 0..blocks_x {
                    let bx = bx_idx * bs;
                    let by = by_idx * bs;
                    let bw = bs.min(cur_w - bx);
                    let bh = bs.min(cur_h - by);

                    let density =
                        Self::block_edge_density(&current_rgb, cur_w, cur_h, bx, by, bw, bh);

                    if density >= self.config.edge_threshold {
                        // Map back to original coordinates
                        let inv_scale = 1.0 / level_info.scale;
                        let orig_x = bx as f32 * inv_scale;
                        let orig_y = by as f32 * inv_scale;
                        let orig_w = bw as f32 * inv_scale;
                        let orig_h = bh as f32 * inv_scale;

                        all_detections.push(PyramidDetection {
                            bbox: Rect::new(orig_x, orig_y, orig_w, orig_h),
                            confidence: density.clamp(0.0, 1.0),
                            source_level: level_info.level,
                        });
                    }
                }
            }

            // Downsample for next level (skip the last iteration)
            if level_info.level + 1 < levels.len() {
                let (down, dw, dh) =
                    Self::downsample(&current_rgb, cur_w, cur_h, self.config.scale_factor);
                current_rgb = down;
                cur_w = dw;
                cur_h = dh;
            }
        }

        // Apply NMS across all pyramid levels
        self.apply_nms(&mut all_detections);

        Ok(all_detections)
    }

    /// Greedy NMS on pyramid detections (by descending confidence).
    fn apply_nms(&self, detections: &mut Vec<PyramidDetection>) {
        detections.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n = detections.len();
        let mut suppressed = vec![false; n];

        for i in 0..n {
            if suppressed[i] {
                continue;
            }
            for j in (i + 1)..n {
                if suppressed[j] {
                    continue;
                }
                if detections[i].bbox.iou(&detections[j].bbox) > self.config.nms_iou_threshold {
                    suppressed[j] = true;
                }
            }
        }

        let mut out = Vec::with_capacity(n);
        for (i, det) in detections.drain(..).enumerate() {
            if !suppressed[i] {
                out.push(det);
            }
        }
        *detections = out;
    }

    /// Return a reference to the current config.
    #[must_use]
    pub fn config(&self) -> &PyramidConfig {
        &self.config
    }
}

impl Default for PyramidDetector {
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

    fn uniform_image(w: usize, h: usize, v: u8) -> Vec<u8> {
        vec![v; w * h * 3]
    }

    fn edgy_image(w: usize, h: usize) -> Vec<u8> {
        let mut data = vec![0u8; w * h * 3];
        // Create strong vertical stripes
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                let v = if x % 4 < 2 { 200u8 } else { 20u8 };
                data[idx] = v;
                data[idx + 1] = v;
                data[idx + 2] = v;
            }
        }
        data
    }

    // 1. Default config values
    #[test]
    fn test_pyramid_config_defaults() {
        let cfg = PyramidConfig::default();
        assert_eq!(cfg.num_levels, 4);
        assert!((cfg.scale_factor - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.min_dimension, 32);
        assert_eq!(cfg.block_size, 16);
    }

    // 2. Build levels for a large image
    #[test]
    fn test_build_levels() {
        let det = PyramidDetector::new();
        let levels = det.build_levels(640, 480);
        assert!(!levels.is_empty());
        assert_eq!(levels[0].width, 640);
        assert_eq!(levels[0].height, 480);
        assert!((levels[0].scale - 1.0).abs() < f32::EPSILON);
        // Each subsequent level should be smaller
        for i in 1..levels.len() {
            assert!(levels[i].width < levels[i - 1].width);
            assert!(levels[i].height < levels[i - 1].height);
        }
    }

    // 3. Build levels stops at min_dimension
    #[test]
    fn test_build_levels_min_dimension() {
        let cfg = PyramidConfig {
            num_levels: 10,
            scale_factor: 0.5,
            min_dimension: 100,
            ..Default::default()
        };
        let det = PyramidDetector::with_config(cfg);
        let levels = det.build_levels(200, 200);
        // 200 -> 100 -> 50 (below 100, stop)
        assert_eq!(levels.len(), 2);
    }

    // 4. Detect on uniform image (no edges)
    #[test]
    fn test_detect_uniform_no_detections() {
        let det = PyramidDetector::new();
        let w = 128;
        let h = 128;
        let data = uniform_image(w, h, 128);
        let result = det.detect(&data, w, h);
        assert!(result.is_ok());
        let dets = result.expect("should succeed");
        assert!(
            dets.is_empty(),
            "uniform image should produce no detections"
        );
    }

    // 5. Detect on edgy image produces detections
    #[test]
    fn test_detect_edgy_image() {
        let det = PyramidDetector::new();
        let w = 128;
        let h = 128;
        let data = edgy_image(w, h);
        let result = det.detect(&data, w, h);
        assert!(result.is_ok());
        let dets = result.expect("should succeed");
        assert!(!dets.is_empty(), "edgy image should produce detections");
    }

    // 6. Invalid dimensions produce error
    #[test]
    fn test_detect_invalid_dimensions() {
        let det = PyramidDetector::new();
        let result = det.detect(&[0u8; 10], 100, 100);
        assert!(result.is_err());
    }

    // 7. Detections have valid bounding boxes
    #[test]
    fn test_detections_valid_bbox() {
        let det = PyramidDetector::new();
        let w = 128;
        let h = 128;
        let data = edgy_image(w, h);
        let dets = det.detect(&data, w, h).expect("should succeed");
        for d in &dets {
            assert!(d.bbox.x >= 0.0);
            assert!(d.bbox.y >= 0.0);
            assert!(d.bbox.width > 0.0);
            assert!(d.bbox.height > 0.0);
            assert!(d.confidence > 0.0 && d.confidence <= 1.0);
        }
    }

    // 8. Detections from multiple levels exist
    #[test]
    fn test_detections_multi_level() {
        let cfg = PyramidConfig {
            num_levels: 3,
            scale_factor: 0.5,
            min_dimension: 16,
            edge_threshold: 0.01, // low threshold to catch edges at all levels
            block_size: 8,
            nms_iou_threshold: 0.9, // high NMS to keep more detections
        };
        let det = PyramidDetector::with_config(cfg);
        let w = 128;
        let h = 128;
        let data = edgy_image(w, h);
        let dets = det.detect(&data, w, h).expect("should succeed");
        // Should have detections from at least 2 different levels
        let levels_seen: std::collections::HashSet<usize> =
            dets.iter().map(|d| d.source_level).collect();
        assert!(
            levels_seen.len() >= 2,
            "expected detections from multiple levels, got {:?}",
            levels_seen
        );
    }

    // 9. NMS removes overlapping detections
    #[test]
    fn test_nms_reduces_count() {
        let cfg = PyramidConfig {
            nms_iou_threshold: 0.3, // aggressive NMS
            edge_threshold: 0.01,
            block_size: 8,
            ..Default::default()
        };
        let det = PyramidDetector::with_config(cfg);
        let w = 64;
        let h = 64;
        let data = edgy_image(w, h);
        let dets_nms = det.detect(&data, w, h).expect("should succeed");

        // Compare with no-NMS (high threshold)
        let cfg_no_nms = PyramidConfig {
            nms_iou_threshold: 1.0, // effectively disable NMS
            edge_threshold: 0.01,
            block_size: 8,
            ..Default::default()
        };
        let det_no_nms = PyramidDetector::with_config(cfg_no_nms);
        let dets_all = det_no_nms.detect(&data, w, h).expect("should succeed");

        // NMS should keep same or fewer detections
        assert!(
            dets_nms.len() <= dets_all.len(),
            "NMS should not increase count: {} vs {}",
            dets_nms.len(),
            dets_all.len()
        );
    }

    // 10. Config accessor
    #[test]
    fn test_config_accessor() {
        let cfg = PyramidConfig {
            num_levels: 5,
            scale_factor: 0.75,
            min_dimension: 64,
            edge_threshold: 0.1,
            block_size: 32,
            nms_iou_threshold: 0.5,
        };
        let det = PyramidDetector::with_config(cfg);
        assert_eq!(det.config().num_levels, 5);
        assert!((det.config().scale_factor - 0.75).abs() < f32::EPSILON);
        assert_eq!(det.config().min_dimension, 64);
    }

    // 11. Downsample produces correct dimensions
    #[test]
    fn test_downsample_dimensions() {
        let w = 100;
        let h = 80;
        let data = uniform_image(w, h, 128);
        let (down, dw, dh) = PyramidDetector::downsample(&data, w, h, 0.5);
        assert_eq!(dw, 50);
        assert_eq!(dh, 40);
        assert_eq!(down.len(), dw * dh * 3);
    }

    // 12. Single level pyramid
    #[test]
    fn test_single_level_pyramid() {
        let cfg = PyramidConfig {
            num_levels: 1,
            ..Default::default()
        };
        let det = PyramidDetector::with_config(cfg);
        let levels = det.build_levels(256, 256);
        assert_eq!(levels.len(), 1);
        let data = edgy_image(256, 256);
        let result = det.detect(&data, 256, 256);
        assert!(result.is_ok());
    }
}
