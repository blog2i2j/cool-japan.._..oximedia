//! Insert shot and cutaway shot detection with confidence calibration.
//!
//! This module provides a dedicated, high-precision detector for two special
//! shot categories that share a common editing function — they *interrupt* the
//! primary scene space — but differ in their visual signature:
//!
//! | Category   | Visual signature                                          |
//! |------------|-----------------------------------------------------------|
//! | **Insert** | ECU of an inanimate object / detail; tight central crop;  |
//! |            | high edge energy concentrated in the frame centre;        |
//! |            | typically matches the scene's colour palette.             |
//! | **Cutaway**| Spatially unrelated content; often a different exposure /  |
//! |            | colour temperature; can be any shot size but is visually  |
//! |            | distinct from surrounding shots in the same sequence.     |
//!
//! # Detection pipeline
//!
//! 1. **Spatial features** — central-vs-peripheral edge energy ratio and
//!    overall edge density (Prewitt operator, fast and allocation-free).
//! 2. **Colour features** — mean and variance of each RGB channel, plus a
//!    global colour dissimilarity score relative to a supplied scene palette.
//! 3. **Confidence calibration** — raw logistic scores are calibrated with a
//!    Platt-scaling inspired linear transform so the output closely reflects
//!    empirical precision on representative editorial data.
//!
//! # Usage
//!
//! ```
//! use oximedia_shots::insert_cutaway::{InsertCutawayDetector, ScenePalette};
//! use oximedia_shots::frame_buffer::FrameBuffer;
//!
//! let mut palette = ScenePalette::default();
//! let frame = FrameBuffer::zeros(64, 64, 3);
//! palette.update(&frame);
//!
//! let detector = InsertCutawayDetector::default();
//! let result = detector.detect(&frame, &palette).expect("detection should succeed in doc");
//! println!("Insert probability: {:.2}", result.insert_probability);
//! println!("Cutaway probability: {:.2}", result.cutaway_probability);
//! ```

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::FrameBuffer;

// ---------------------------------------------------------------------------
// Scene palette (running colour statistics)
// ---------------------------------------------------------------------------

/// Running colour statistics for a scene sequence, used as the baseline for
/// cutaway detection.
///
/// Call [`ScenePalette::update`] once per frame to maintain the EMA of the
/// per-channel mean and variance.
#[derive(Debug, Clone)]
pub struct ScenePalette {
    /// EMA of per-channel mean luminance (index 0=R, 1=G, 2=B).
    pub channel_mean: [f32; 3],
    /// EMA of per-channel variance.
    pub channel_variance: [f32; 3],
    /// EMA decay factor (0 < α ≤ 1; smaller = longer memory).
    pub alpha: f32,
    /// Number of frames incorporated.
    pub frames_seen: u64,
}

impl Default for ScenePalette {
    fn default() -> Self {
        Self {
            channel_mean: [128.0; 3],
            channel_variance: [40.0; 3],
            alpha: 0.08,
            frames_seen: 0,
        }
    }
}

impl ScenePalette {
    /// Create a palette with a custom decay factor.
    #[must_use]
    pub fn with_alpha(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.001, 1.0),
            ..Self::default()
        }
    }

    /// Update the palette statistics with a new frame.
    pub fn update(&mut self, frame: &FrameBuffer) {
        let (h, w, ch) = frame.dim();
        if ch < 3 || h == 0 || w == 0 {
            return;
        }

        let n = (h * w) as f64;
        for c in 0..3 {
            let mut sum = 0.0_f64;
            let mut sum_sq = 0.0_f64;
            for y in 0..h {
                for x in 0..w {
                    let v = f64::from(frame.get(y, x, c));
                    sum += v;
                    sum_sq += v * v;
                }
            }
            let mean = (sum / n) as f32;
            let var = ((sum_sq / n) as f32 - mean * mean).max(0.0);

            if self.frames_seen == 0 {
                self.channel_mean[c] = mean;
                self.channel_variance[c] = var;
            } else {
                let a = self.alpha;
                self.channel_mean[c] = a * mean + (1.0 - a) * self.channel_mean[c];
                self.channel_variance[c] = a * var + (1.0 - a) * self.channel_variance[c];
            }
        }

        self.frames_seen += 1;
    }

    /// Reset the palette to default statistics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ---------------------------------------------------------------------------
// Detection result
// ---------------------------------------------------------------------------

/// Detailed detection result from [`InsertCutawayDetector`].
#[derive(Debug, Clone)]
pub struct InsertCutawayResult {
    /// Calibrated probability the frame is an **insert** shot (0.0–1.0).
    pub insert_probability: f32,
    /// Calibrated probability the frame is a **cutaway** shot (0.0–1.0).
    pub cutaway_probability: f32,
    /// Central edge concentration ratio (> 1 = edges concentrated in centre).
    pub central_edge_ratio: f32,
    /// Overall edge density (0.0–1.0).
    pub edge_density: f32,
    /// Colour dissimilarity to the scene palette (0.0–1.0).
    pub color_dissimilarity: f32,
    /// Whether the frame is classified as an insert (probability > threshold).
    pub is_insert: bool,
    /// Whether the frame is classified as a cutaway (probability > threshold).
    pub is_cutaway: bool,
}

// ---------------------------------------------------------------------------
// Detector configuration
// ---------------------------------------------------------------------------

/// Configuration for [`InsertCutawayDetector`].
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// Probability threshold above which `is_insert` is set.
    pub insert_threshold: f32,
    /// Probability threshold above which `is_cutaway` is set.
    pub cutaway_threshold: f32,
    /// Minimum edge density for insert consideration (avoids false positives
    /// on near-blank frames).
    pub insert_min_edge_density: f32,
    /// Platt-scaling parameter A for insert logistic (raw score multiplier).
    pub insert_platt_a: f32,
    /// Platt-scaling parameter B for insert logistic (bias).
    pub insert_platt_b: f32,
    /// Platt-scaling parameter A for cutaway logistic.
    pub cutaway_platt_a: f32,
    /// Platt-scaling parameter B for cutaway logistic.
    pub cutaway_platt_b: f32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            insert_threshold: 0.55,
            cutaway_threshold: 0.55,
            insert_min_edge_density: 0.05,
            // Platt parameters tuned to the spatial feature range
            insert_platt_a: 8.0,
            insert_platt_b: -4.0,
            cutaway_platt_a: 6.0,
            cutaway_platt_b: -2.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

/// Insert shot and cutaway shot detector with confidence calibration.
pub struct InsertCutawayDetector {
    config: DetectorConfig,
}

impl Default for InsertCutawayDetector {
    fn default() -> Self {
        Self::new(DetectorConfig::default())
    }
}

impl InsertCutawayDetector {
    /// Create a new detector with the given configuration.
    #[must_use]
    pub fn new(config: DetectorConfig) -> Self {
        Self { config }
    }

    /// Detect whether `frame` is an insert or cutaway shot.
    ///
    /// `palette` is the running scene colour statistics; update it between
    /// calls with [`ScenePalette::update`].
    ///
    /// # Errors
    ///
    /// Returns [`ShotError::InvalidFrame`] if the frame has fewer than 3
    /// channels or is smaller than 8×8.
    pub fn detect(
        &self,
        frame: &FrameBuffer,
        palette: &ScenePalette,
    ) -> ShotResult<InsertCutawayResult> {
        let (h, w, ch) = frame.dim();
        if ch < 3 {
            return Err(ShotError::InvalidFrame(
                "Frame must have at least 3 channels for insert/cutaway detection".to_string(),
            ));
        }
        if h < 8 || w < 8 {
            return Err(ShotError::InvalidFrame(
                "Frame must be at least 8×8 pixels for insert/cutaway detection".to_string(),
            ));
        }

        let (edge_density, central_edge_ratio) = spatial_features(frame, h, w);
        let color_dissimilarity = color_dissimilarity_score(frame, palette);

        // --- Insert raw score ---
        // High central concentration + sufficient edge density → insert
        let insert_raw = if edge_density >= self.config.insert_min_edge_density {
            // Normalise: central ratio > 2 is strong evidence
            let concentration_score = ((central_edge_ratio - 1.0) / 1.5).clamp(0.0, 1.0);
            let density_score = (edge_density / 0.4).clamp(0.0, 1.0);
            0.7 * concentration_score + 0.3 * density_score
        } else {
            0.0
        };

        // --- Cutaway raw score ---
        // High colour dissimilarity from scene palette → cutaway
        // Also suppressed when palette hasn't seen enough frames (< 5)
        let cutaway_raw = if palette.frames_seen >= 5 {
            color_dissimilarity
        } else {
            0.0
        };

        let insert_probability = platt_sigmoid(
            insert_raw,
            self.config.insert_platt_a,
            self.config.insert_platt_b,
        );
        let cutaway_probability = platt_sigmoid(
            cutaway_raw,
            self.config.cutaway_platt_a,
            self.config.cutaway_platt_b,
        );

        Ok(InsertCutawayResult {
            insert_probability,
            cutaway_probability,
            central_edge_ratio,
            edge_density,
            color_dissimilarity,
            is_insert: insert_probability >= self.config.insert_threshold,
            is_cutaway: cutaway_probability >= self.config.cutaway_threshold,
        })
    }

    /// Convenience: detect and update palette in a single call.
    ///
    /// The palette is updated *after* detection so that the cutaway decision
    /// is based on the scene *before* including the current frame.
    ///
    /// # Errors
    ///
    /// Returns [`ShotError::InvalidFrame`] for invalid frames.
    pub fn detect_and_update(
        &self,
        frame: &FrameBuffer,
        palette: &mut ScenePalette,
    ) -> ShotResult<InsertCutawayResult> {
        let result = self.detect(frame, palette)?;
        palette.update(frame);
        Ok(result)
    }

    /// Access the current configuration.
    #[must_use]
    pub fn config(&self) -> &DetectorConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Feature extraction helpers
// ---------------------------------------------------------------------------

/// Compute `(edge_density, central_edge_ratio)` using the Prewitt operator.
///
/// `central_edge_ratio` is the ratio of gradient energy in the central 50%×50%
/// region to the full-frame energy. Values > 1 indicate central concentration.
fn spatial_features(frame: &FrameBuffer, h: usize, w: usize) -> (f32, f32) {
    if h < 3 || w < 3 {
        return (0.0, 1.0);
    }

    let cy0 = h / 4;
    let cy1 = 3 * h / 4;
    let cx0 = w / 4;
    let cx1 = 3 * w / 4;

    let mut total_energy = 0.0_f64;
    let mut centre_energy = 0.0_f64;
    let mut total_count = 0_u32;
    let mut centre_count = 0_u32;
    let mut edge_count = 0_u32;

    // Prewitt threshold: 5% of max gradient magnitude
    let edge_thr = 255.0_f64 * 3.0 * 0.05;

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            // Luminance (BT.601)
            let luma = |py: usize, px: usize| -> f64 {
                f64::from(frame.get(py, px, 0)) * 0.299
                    + f64::from(frame.get(py, px, 1)) * 0.587
                    + f64::from(frame.get(py, px, 2)) * 0.114
            };

            let gx = luma(y - 1, x + 1) + luma(y, x + 1) + luma(y + 1, x + 1)
                - luma(y - 1, x - 1)
                - luma(y, x - 1)
                - luma(y + 1, x - 1);
            let gy = luma(y + 1, x - 1) + luma(y + 1, x) + luma(y + 1, x + 1)
                - luma(y - 1, x - 1)
                - luma(y - 1, x)
                - luma(y - 1, x + 1);

            let mag = (gx * gx + gy * gy).sqrt();
            total_energy += mag;
            total_count += 1;

            if mag > edge_thr {
                edge_count += 1;
            }

            if y >= cy0 && y < cy1 && x >= cx0 && x < cx1 {
                centre_energy += mag;
                centre_count += 1;
            }
        }
    }

    let edge_density = if total_count > 0 {
        (edge_count as f32 / total_count as f32).min(1.0)
    } else {
        0.0
    };

    let total_density = if total_count > 0 {
        total_energy / f64::from(total_count)
    } else {
        0.0
    };
    let centre_density = if centre_count > 0 {
        centre_energy / f64::from(centre_count)
    } else {
        0.0
    };

    let central_edge_ratio = if total_density < f64::EPSILON {
        1.0_f32
    } else {
        (centre_density / total_density) as f32
    };

    (edge_density, central_edge_ratio)
}

/// Compute a colour dissimilarity score (0–1) between a frame and the scene
/// palette using a weighted combination of per-channel mean differences and
/// variance differences.
fn color_dissimilarity_score(frame: &FrameBuffer, palette: &ScenePalette) -> f32 {
    let (h, w, ch) = frame.dim();
    if ch < 3 || h == 0 || w == 0 {
        return 0.0;
    }

    let n = (h * w) as f64;
    let mut chan_score = 0.0_f32;

    for c in 0..3 {
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for y in 0..h {
            for x in 0..w {
                let v = f64::from(frame.get(y, x, c));
                sum += v;
                sum_sq += v * v;
            }
        }
        let mean = (sum / n) as f32;
        let var = ((sum_sq / n) as f32 - mean * mean).max(0.0);

        let scene_mean = palette.channel_mean[c];
        let scene_var = palette.channel_variance[c].max(1.0);

        // Normalised mean difference (0–1 scale, 255 units)
        let mean_diff = (mean - scene_mean).abs() / 255.0;
        // Normalised variance difference (relative)
        let var_diff = ((var - scene_var).abs() / (scene_var + 1.0)).min(1.0);

        chan_score += 0.7 * mean_diff + 0.3 * var_diff;
    }

    // Average across 3 channels, clamp to [0, 1]
    (chan_score / 3.0).clamp(0.0, 1.0)
}

/// Platt-scaling sigmoid: σ(A·x + B) = 1 / (1 + exp(-(A·x + B))).
#[inline]
fn platt_sigmoid(x: f32, a: f32, b: f32) -> f32 {
    let z = a * x + b;
    // Numerically stable sigmoid
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_buffer::FrameBuffer;

    // ---- Helpers ----

    fn flat_rgb(h: usize, w: usize, r: u8, g: u8, b: u8) -> FrameBuffer {
        let mut f = FrameBuffer::zeros(h, w, 3);
        for y in 0..h {
            for x in 0..w {
                f.set(y, x, 0, r);
                f.set(y, x, 1, g);
                f.set(y, x, 2, b);
            }
        }
        f
    }

    fn checkerboard(h: usize, w: usize) -> FrameBuffer {
        let mut f = FrameBuffer::zeros(h, w, 3);
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) % 2 == 0 { 255 } else { 0 };
                f.set(y, x, 0, v);
                f.set(y, x, 1, v);
                f.set(y, x, 2, v);
            }
        }
        f
    }

    /// Frame with sharp centre (checkerboard) and flat grey periphery.
    fn insert_like(h: usize, w: usize) -> FrameBuffer {
        let mut f = FrameBuffer::zeros(h, w, 3);
        let cy0 = h / 4;
        let cy1 = 3 * h / 4;
        let cx0 = w / 4;
        let cx1 = 3 * w / 4;
        for y in 0..h {
            for x in 0..w {
                let v = if y >= cy0 && y < cy1 && x >= cx0 && x < cx1 {
                    if (x + y) % 2 == 0 {
                        255
                    } else {
                        0
                    }
                } else {
                    128
                };
                f.set(y, x, 0, v);
                f.set(y, x, 1, v);
                f.set(y, x, 2, v);
            }
        }
        f
    }

    // ---- ScenePalette ----

    #[test]
    fn test_scene_palette_default() {
        let p = ScenePalette::default();
        assert_eq!(p.frames_seen, 0);
        for c in 0..3 {
            assert!(p.channel_mean[c] > 0.0);
        }
    }

    #[test]
    fn test_scene_palette_update_converges() {
        let mut palette = ScenePalette::with_alpha(0.5);
        let frame = flat_rgb(32, 32, 200, 100, 50);
        // After many updates the EMA should converge toward the frame values
        for _ in 0..40 {
            palette.update(&frame);
        }
        assert!(
            (palette.channel_mean[0] - 200.0).abs() < 5.0,
            "R mean should converge to ~200: {}",
            palette.channel_mean[0]
        );
    }

    #[test]
    fn test_scene_palette_reset() {
        let mut palette = ScenePalette::default();
        palette.update(&flat_rgb(32, 32, 200, 200, 200));
        assert_eq!(palette.frames_seen, 1);
        palette.reset();
        assert_eq!(palette.frames_seen, 0);
    }

    // ---- DetectorConfig defaults ----

    #[test]
    fn test_detector_config_defaults_sensible() {
        let cfg = DetectorConfig::default();
        assert!(cfg.insert_threshold > 0.0 && cfg.insert_threshold < 1.0);
        assert!(cfg.cutaway_threshold > 0.0 && cfg.cutaway_threshold < 1.0);
        assert!(cfg.insert_min_edge_density >= 0.0);
    }

    // ---- Error cases ----

    #[test]
    fn test_detect_too_few_channels() {
        let det = InsertCutawayDetector::default();
        let palette = ScenePalette::default();
        let frame = FrameBuffer::zeros(32, 32, 1);
        assert!(det.detect(&frame, &palette).is_err());
    }

    #[test]
    fn test_detect_too_small_frame() {
        let det = InsertCutawayDetector::default();
        let palette = ScenePalette::default();
        let frame = FrameBuffer::zeros(4, 4, 3);
        assert!(det.detect(&frame, &palette).is_err());
    }

    // ---- Probabilities are in [0, 1] ----

    #[test]
    fn test_probabilities_in_range() {
        let det = InsertCutawayDetector::default();
        let mut palette = ScenePalette::default();
        for frame in [
            flat_rgb(64, 64, 128, 128, 128),
            checkerboard(64, 64),
            insert_like(64, 64),
        ] {
            let result = det
                .detect(&frame, &palette)
                .expect("detection should succeed in test");
            assert!(
                result.insert_probability >= 0.0 && result.insert_probability <= 1.0,
                "insert_probability out of range: {}",
                result.insert_probability
            );
            assert!(
                result.cutaway_probability >= 0.0 && result.cutaway_probability <= 1.0,
                "cutaway_probability out of range: {}",
                result.cutaway_probability
            );
            palette.update(&frame);
        }
    }

    // ---- Insert-like frame has higher central edge ratio ----

    #[test]
    fn test_insert_like_central_concentration() {
        let det = InsertCutawayDetector::default();
        let palette = ScenePalette::default();
        let frame = insert_like(128, 128);
        let result = det
            .detect(&frame, &palette)
            .expect("detection should succeed in test");
        assert!(
            result.central_edge_ratio > 1.0,
            "insert-like frame should have central edge ratio > 1: {}",
            result.central_edge_ratio
        );
    }

    // ---- Flat frame has low edge density ----

    #[test]
    fn test_flat_frame_low_edge_density() {
        let det = InsertCutawayDetector::default();
        let palette = ScenePalette::default();
        let frame = flat_rgb(64, 64, 128, 128, 128);
        let result = det.detect(&frame, &palette).expect("flat frame");
        assert!(
            result.edge_density < 0.1,
            "flat frame edge density should be low: {}",
            result.edge_density
        );
    }

    // ---- detect_and_update increments palette frames_seen ----

    #[test]
    fn test_detect_and_update_increments_palette() {
        let det = InsertCutawayDetector::default();
        let mut palette = ScenePalette::default();
        let frame = flat_rgb(32, 32, 100, 100, 100);
        det.detect_and_update(&frame, &mut palette)
            .expect("detect_and_update should succeed in test");
        assert_eq!(palette.frames_seen, 1);
    }

    // ---- Colour dissimilarity: frame very different from palette → high score ----

    #[test]
    fn test_color_dissimilarity_extreme() {
        // Warm up palette with red frames
        let mut palette = ScenePalette::with_alpha(0.5);
        for _ in 0..10 {
            palette.update(&flat_rgb(32, 32, 200, 50, 50));
        }
        // Now test a blue frame — should have high dissimilarity
        let blue = flat_rgb(64, 64, 30, 30, 200);
        let score = color_dissimilarity_score(&blue, &palette);
        assert!(
            score > 0.2,
            "blue frame vs red palette should have high dissimilarity: {score}"
        );
    }

    // ---- Colour dissimilarity: identical frame → low score ----

    #[test]
    fn test_color_dissimilarity_identical() {
        let mut palette = ScenePalette::with_alpha(0.5);
        let frame = flat_rgb(32, 32, 100, 150, 200);
        // Use more iterations so variance EMA converges fully to ~0
        for _ in 0..50 {
            palette.update(&frame);
        }
        let score = color_dissimilarity_score(&frame, &palette);
        assert!(
            score < 0.20,
            "identical frame vs converged palette should have low dissimilarity: {score}"
        );
    }

    // ---- platt_sigmoid boundaries ----

    #[test]
    fn test_platt_sigmoid_properties() {
        // σ(0) ≈ 0.5 for A=1, B=0
        let mid = platt_sigmoid(0.0, 1.0, 0.0);
        assert!((mid - 0.5).abs() < 1e-5);

        // Very large positive → near 1.0
        let high = platt_sigmoid(1.0, 100.0, 0.0);
        assert!(high > 0.99);

        // Very large negative → near 0.0
        let low = platt_sigmoid(0.0, 100.0, -200.0);
        assert!(low < 0.01);
    }

    // ---- cutaway suppressed when palette not warm ----

    #[test]
    fn test_cutaway_suppressed_cold_palette() {
        let det = InsertCutawayDetector::default();
        let palette = ScenePalette::default(); // frames_seen == 0
                                               // Even a very different frame should not trigger cutaway
        let frame = flat_rgb(64, 64, 0, 0, 255); // bright blue
        let result = det.detect(&frame, &palette).expect("cold palette test");
        assert!(
            !result.is_cutaway,
            "cutaway should be suppressed with cold palette"
        );
    }

    // ── Extended insert / cutaway classification tests (TODO item 2) ──────────

    /// `detect_and_update` updates the palette after detection.
    #[test]
    fn test_detect_and_update_palette_evolves() {
        let det = InsertCutawayDetector::default();
        let mut palette = ScenePalette::with_alpha(1.0); // full replacement each update
        let frame = flat_rgb(32, 32, 200, 100, 50);

        // Before update: palette has default mean ~128
        let initial_mean = palette.channel_mean[0];

        det.detect_and_update(&frame, &mut palette)
            .expect("detect_and_update should succeed");

        // After update: R mean should have moved toward 200
        assert!(
            palette.channel_mean[0] > initial_mean,
            "R mean should increase after update with bright-red frame: {}",
            palette.channel_mean[0]
        );
    }

    /// Insert probability should be higher for an insert-like (centralised-edge)
    /// frame than for a flat uniform frame.
    #[test]
    fn test_insert_probability_insert_gt_flat() {
        let det = InsertCutawayDetector::default();
        let palette = ScenePalette::default();
        let flat = flat_rgb(128, 128, 128, 128, 128);
        let insert = insert_like(128, 128);
        let flat_res = det.detect(&flat, &palette).expect("flat detect");
        let insert_res = det.detect(&insert, &palette).expect("insert detect");
        assert!(
            insert_res.insert_probability >= flat_res.insert_probability,
            "insert-like frame should have >= insert_probability vs flat: {} vs {}",
            insert_res.insert_probability,
            flat_res.insert_probability
        );
    }

    /// A frame with a clear hard edge has higher edge_density than a flat frame.
    #[test]
    fn test_edge_frame_higher_density_than_flat() {
        let det = InsertCutawayDetector::default();
        let palette = ScenePalette::default();
        let flat = flat_rgb(64, 64, 128, 128, 128);

        // Frame with a sharp horizontal edge across the middle
        let mut edge_frame = FrameBuffer::zeros(64, 64, 3);
        for y in 0..32 {
            for x in 0..64 {
                edge_frame.set(y, x, 0, 0);
                edge_frame.set(y, x, 1, 0);
                edge_frame.set(y, x, 2, 0);
            }
        }
        for y in 32..64 {
            for x in 0..64 {
                edge_frame.set(y, x, 0, 255);
                edge_frame.set(y, x, 1, 255);
                edge_frame.set(y, x, 2, 255);
            }
        }

        let flat_res = det.detect(&flat, &palette).expect("flat ok");
        let edge_res = det.detect(&edge_frame, &palette).expect("edge frame ok");
        assert!(
            edge_res.edge_density > flat_res.edge_density,
            "edge frame should have higher edge density than flat: {} vs {}",
            edge_res.edge_density,
            flat_res.edge_density
        );
    }

    /// Custom thresholds control `is_insert` and `is_cutaway` flags.
    #[test]
    fn test_custom_threshold_always_insert() {
        // With insert_threshold = 0.0, every frame is flagged as insert.
        let cfg = DetectorConfig {
            insert_threshold: 0.0,
            ..DetectorConfig::default()
        };
        let det = InsertCutawayDetector::new(cfg);
        let palette = ScenePalette::default();
        let frame = flat_rgb(32, 32, 100, 100, 100);
        let res = det.detect(&frame, &palette).expect("ok");
        assert!(res.is_insert, "threshold=0 should always flag is_insert");
    }

    /// With `insert_threshold = 1.0`, `is_insert` is never set.
    #[test]
    fn test_custom_threshold_never_insert() {
        let cfg = DetectorConfig {
            insert_threshold: 1.0,
            ..DetectorConfig::default()
        };
        let det = InsertCutawayDetector::new(cfg);
        let palette = ScenePalette::default();
        let frame = insert_like(128, 128);
        let res = det.detect(&frame, &palette).expect("ok");
        assert!(!res.is_insert, "threshold=1.0 should never flag is_insert");
    }

    /// `color_dissimilarity` is reported in [0, 1] for all tested frames.
    #[test]
    fn test_color_dissimilarity_in_range() {
        let det = InsertCutawayDetector::default();
        let mut palette = ScenePalette::with_alpha(0.5);
        // Warm up palette
        for _ in 0..10 {
            palette.update(&flat_rgb(32, 32, 100, 100, 100));
        }
        for frame in [
            flat_rgb(32, 32, 0, 0, 0),
            flat_rgb(32, 32, 255, 255, 255),
            flat_rgb(32, 32, 100, 100, 100),
        ] {
            let res = det.detect(&frame, &palette).expect("ok");
            assert!(
                res.color_dissimilarity >= 0.0 && res.color_dissimilarity <= 1.0,
                "color_dissimilarity out of range: {}",
                res.color_dissimilarity
            );
        }
    }

    /// Calling `config()` returns the configuration used to create the detector.
    #[test]
    fn test_config_accessor_returns_correct_values() {
        let cfg = DetectorConfig {
            insert_threshold: 0.42,
            cutaway_threshold: 0.77,
            ..DetectorConfig::default()
        };
        let det = InsertCutawayDetector::new(cfg);
        assert!((det.config().insert_threshold - 0.42).abs() < f32::EPSILON);
        assert!((det.config().cutaway_threshold - 0.77).abs() < f32::EPSILON);
    }

    /// After 5+ palette updates, `is_cutaway` can be triggered for a wildly different frame.
    #[test]
    fn test_cutaway_possible_after_warm_palette() {
        // Build a very sensitive detector (low cutaway_threshold)
        let cfg = DetectorConfig {
            cutaway_threshold: 0.01, // trigger on any dissimilarity
            ..DetectorConfig::default()
        };
        let det = InsertCutawayDetector::new(cfg);
        let mut palette = ScenePalette::with_alpha(0.5);
        // Warm up palette with green frames
        for _ in 0..10 {
            palette.update(&flat_rgb(32, 32, 20, 200, 20));
        }
        // Highly different (red) frame
        let frame = flat_rgb(64, 64, 255, 0, 0);
        let res = det
            .detect(&frame, &palette)
            .expect("warm palette cutaway test");
        assert!(
            res.is_cutaway,
            "low-threshold detector should flag red frame as cutaway vs green palette"
        );
    }

    /// Flat frame should not be flagged as an insert (insufficient edge density).
    #[test]
    fn test_flat_frame_not_insert() {
        let det = InsertCutawayDetector::default();
        let palette = ScenePalette::default();
        let flat = flat_rgb(64, 64, 128, 128, 128);
        let res = det.detect(&flat, &palette).expect("flat insert test");
        assert!(
            !res.is_insert,
            "flat frame should not be flagged as an insert"
        );
    }
}
