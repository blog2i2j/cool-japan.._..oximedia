//! Adaptive quantization matrix (QM) selection for the AV1 encoder.
//!
//! AV1 supports 16 levels of quantization matrices (QM levels 0–15).  Higher
//! levels apply heavier per-frequency weighting, preserving low-frequency
//! content at the cost of high-frequency detail.  The optimal QM level
//! depends on the encoded content type:
//!
//! | Content Type | Recommended QM Level | Rationale |
//! |---|---|---|
//! | Screen content / graphics | 0–2 | Preserve sharp edges & text |
//! | High-motion video (sports) | 4–6 | Reduce ringing on fast objects |
//! | Film / drama              | 7–9 | Balance texture & flatness |
//! | Talking head / webcam     | 10–12| Emphasise faces, drop BG detail |
//! | Slide / presentation      | 0–1 | Lossless-like for graphics |
//! | Animation                 | 5–8 | Smooth shading, no ringing |
//!
//! This module provides:
//!
//! 1. [`ContentClass`] — coarse content classification
//! 2. [`ContentAnalyzer`] — lightweight analyzer that derives [`ContentClass`]
//!    from frame statistics (variance, edge density, motion magnitude)
//! 3. [`AdaptiveQmSelector`] — converts a [`ContentClass`] + base QP into a
//!    recommended QM level and optional per-plane delta-QP adjustments
//! 4. [`QmMatrix`] — the actual 4×4 / 8×8 / 16×16 weighting matrices
//!
//! # References
//!
//! - AV1 Specification §5.9.14 (Quantization Matrix)
//! - libaom `av1/encoder/aq_complexity.c`
//! - Netflix Per-Title Encoding paper (Waggoner 2016)

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]

// ─────────────────────────────────────────────────────────────────────────────
// Content classification
// ─────────────────────────────────────────────────────────────────────────────

/// Coarse classification of encoded content.
///
/// Used to select the most appropriate QM level range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentClass {
    /// Computer-generated graphics, text, slides.
    ScreenContent,
    /// Live-action film or drama.
    Film,
    /// Sports or other fast-moving content.
    HighMotion,
    /// Mostly-static camera (talking head, conference call).
    TalkingHead,
    /// 2D/3D animation.
    Animation,
    /// Default / unknown.
    Generic,
}

impl ContentClass {
    /// Recommended QM level range `[min, max]` for this content class.
    ///
    /// Caller should pick within this range based on the current QP.
    #[must_use]
    pub fn qm_range(self) -> (u8, u8) {
        match self {
            Self::ScreenContent => (0, 2),
            Self::Film => (7, 9),
            Self::HighMotion => (4, 6),
            Self::TalkingHead => (10, 12),
            Self::Animation => (5, 8),
            Self::Generic => (5, 10),
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::ScreenContent => "screen-content",
            Self::Film => "film",
            Self::HighMotion => "high-motion",
            Self::TalkingHead => "talking-head",
            Self::Animation => "animation",
            Self::Generic => "generic",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Content analyzer
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated frame statistics required for content classification.
#[derive(Debug, Clone)]
pub struct FrameStats {
    /// Normalised spatial variance (0 = flat, 1 = maximally complex).
    pub spatial_variance: f32,
    /// Edge density: fraction of pixels that are edges (0–1).
    pub edge_density: f32,
    /// Normalised motion magnitude (0 = static, 1 = full-field motion).
    pub motion_magnitude: f32,
    /// Fraction of the frame covered by palette-like (quantised) colours.
    /// High values suggest screen content or animation.
    pub palette_coverage: f32,
    /// Standard deviation of temporal luma difference.
    pub temporal_std: f32,
}

impl FrameStats {
    /// Compute stats from a luma plane and optional previous frame luma.
    ///
    /// `luma` is the Y plane (one byte per pixel, width × height samples).
    /// `prev_luma` is the previous frame's Y plane; pass `None` for I-frames.
    #[must_use]
    pub fn from_luma(luma: &[u8], width: usize, height: usize, prev_luma: Option<&[u8]>) -> Self {
        let pixels = width * height;
        if pixels == 0 {
            return Self {
                spatial_variance: 0.0,
                edge_density: 0.0,
                motion_magnitude: 0.0,
                palette_coverage: 0.0,
                temporal_std: 0.0,
            };
        }

        // ── Spatial variance ─────────────────────────────────────────────────
        let mean: f64 = luma.iter().map(|&p| p as f64).sum::<f64>() / pixels as f64;
        let variance: f64 = luma
            .iter()
            .map(|&p| {
                let d = p as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / pixels as f64;
        // Max variance of an 8-bit signal ≈ 128² = 16384
        let spatial_variance = (variance as f32 / 16384.0).min(1.0);

        // ── Edge density (Sobel-like horizontal + vertical) ──────────────────
        let edge_density = compute_edge_density(luma, width, height);

        // ── Temporal difference ──────────────────────────────────────────────
        let (motion_magnitude, temporal_std) = if let Some(prev) = prev_luma {
            let n = pixels.min(prev.len());
            let diff_sum: f64 = luma[..n]
                .iter()
                .zip(prev[..n].iter())
                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                .sum();
            let diff_mean = diff_sum / n as f64;
            let diff_var: f64 = luma[..n]
                .iter()
                .zip(prev[..n].iter())
                .map(|(&a, &b)| {
                    let d = (a as f64 - b as f64).abs() - diff_mean;
                    d * d
                })
                .sum::<f64>()
                / n as f64;
            (
                (diff_mean / 255.0) as f32,
                (diff_var.sqrt() / 255.0) as f32,
            )
        } else {
            (0.0, 0.0)
        };

        // ── Palette coverage (simple: fraction of pixels that are exact
        //    multiples of 16, suggesting quantised / synthetic colour) ────────
        let palette_count = luma.iter().filter(|&&p| p % 16 == 0).count();
        let palette_coverage = palette_count as f32 / pixels as f32;

        Self {
            spatial_variance,
            edge_density,
            motion_magnitude,
            palette_coverage,
            temporal_std,
        }
    }
}

/// Lightweight content classifier.
///
/// Uses heuristic thresholds derived from the frame statistics to classify
/// content into one of the [`ContentClass`] variants.
#[derive(Debug, Clone, Default)]
pub struct ContentAnalyzer {
    /// Exponentially-smoothed statistics over recent frames.
    smoothed: Option<FrameStats>,
    /// EMA weight for temporal smoothing.
    ema_alpha: f32,
}

impl ContentAnalyzer {
    /// Create a new analyzer.  `ema_alpha` controls temporal smoothing
    /// (0 = no update, 1 = no smoothing, default = 0.2).
    #[must_use]
    pub fn new(ema_alpha: f32) -> Self {
        Self {
            smoothed: None,
            ema_alpha: ema_alpha.clamp(0.0, 1.0),
        }
    }

    /// Feed frame statistics and return the current [`ContentClass`].
    pub fn feed(&mut self, stats: &FrameStats) -> ContentClass {
        // Temporal smoothing
        let s = match &self.smoothed {
            None => stats.clone(),
            Some(prev) => {
                let a = self.ema_alpha;
                FrameStats {
                    spatial_variance: a * stats.spatial_variance
                        + (1.0 - a) * prev.spatial_variance,
                    edge_density: a * stats.edge_density + (1.0 - a) * prev.edge_density,
                    motion_magnitude: a * stats.motion_magnitude
                        + (1.0 - a) * prev.motion_magnitude,
                    palette_coverage: a * stats.palette_coverage
                        + (1.0 - a) * prev.palette_coverage,
                    temporal_std: a * stats.temporal_std + (1.0 - a) * prev.temporal_std,
                }
            }
        };
        let class = Self::classify_static(&s);
        self.smoothed = Some(s);
        class
    }

    /// Reset the temporal smoother.
    pub fn reset(&mut self) {
        self.smoothed = None;
    }

    // ── Heuristic classifier ─────────────────────────────────────────────────

    fn classify_static(s: &FrameStats) -> ContentClass {
        // Screen content: high palette coverage + high edge density + low motion
        if s.palette_coverage > 0.35 && s.edge_density > 0.10 && s.motion_magnitude < 0.05 {
            return ContentClass::ScreenContent;
        }
        // Animation: high palette, lower edge, low-to-moderate motion
        if s.palette_coverage > 0.30 && s.motion_magnitude < 0.15 {
            return ContentClass::Animation;
        }
        // Talking head: low motion, moderate variance, low edge density
        if s.motion_magnitude < 0.04 && s.spatial_variance < 0.25 && s.edge_density < 0.08 {
            return ContentClass::TalkingHead;
        }
        // High motion (sports, action)
        if s.motion_magnitude > 0.18 {
            return ContentClass::HighMotion;
        }
        // Film: moderate motion with rich texture
        if s.motion_magnitude > 0.05 && s.spatial_variance > 0.15 {
            return ContentClass::Film;
        }
        ContentClass::Generic
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge density computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute edge density using simple horizontal + vertical finite differences.
fn compute_edge_density(luma: &[u8], width: usize, height: usize) -> f32 {
    if width < 2 || height < 2 {
        return 0.0;
    }
    let edge_threshold: i32 = 20;
    let mut edge_count = 0u32;
    let total = ((width - 1) * (height - 1)) as u32;

    for row in 0..height - 1 {
        for col in 0..width - 1 {
            let idx = row * width + col;
            let h_diff = (luma[idx] as i32 - luma[idx + 1] as i32).abs();
            let v_diff = (luma[idx] as i32 - luma[idx + width] as i32).abs();
            if h_diff > edge_threshold || v_diff > edge_threshold {
                edge_count += 1;
            }
        }
    }
    if total > 0 {
        edge_count as f32 / total as f32
    } else {
        0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QM matrix definitions
// ─────────────────────────────────────────────────────────────────────────────

/// A quantization matrix for one transform size.
///
/// Values are scale factors (in units of 1/64) applied to each frequency
/// position before quantization.  Lower values → heavier quantization at
/// that position.
#[derive(Debug, Clone)]
pub struct QmMatrix {
    /// Number of rows (= cols for square transforms).
    pub size: usize,
    /// Row-major flat weights; length == size * size.
    pub weights: Vec<u8>,
    /// QM level this matrix was generated for (0–15).
    pub level: u8,
}

impl QmMatrix {
    /// Generate a QM matrix for a given `size` (4, 8, or 16) and `level` (0–15).
    ///
    /// The weight function is derived from the AV1 reference implementation:
    /// `w[i][j] = 255 - level * ramp(i, j, size)`, where `ramp` grows from
    /// low-frequency (top-left) to high-frequency (bottom-right) positions.
    ///
    /// # Errors
    ///
    /// Returns an error if `size` is not 4, 8, or 16, or if `level > 15`.
    pub fn new(size: usize, level: u8) -> crate::error::CodecResult<Self> {
        use crate::error::CodecError;
        if !matches!(size, 4 | 8 | 16) {
            return Err(CodecError::InvalidData(format!(
                "QmMatrix: unsupported size {size}; must be 4, 8, or 16"
            )));
        }
        if level > 15 {
            return Err(CodecError::InvalidData(format!(
                "QmMatrix: level {level} exceeds max 15"
            )));
        }

        let weights = generate_qm_weights(size, level);
        Ok(Self {
            size,
            weights,
            level,
        })
    }

    /// Weight at position (row, col).  Returns 255 on out-of-bounds.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> u8 {
        if row < self.size && col < self.size {
            self.weights[row * self.size + col]
        } else {
            255
        }
    }

    /// Apply the QM weights to transform coefficients in-place (i32 slice).
    ///
    /// Each coefficient is scaled by `weight / 64` (integer approximation).
    pub fn apply(&self, coeffs: &mut [i32]) {
        for (coeff, &w) in coeffs.iter_mut().zip(self.weights.iter()) {
            *coeff = (*coeff * w as i32) / 64;
        }
    }
}

/// Generate flat weight table for a given size and level.
fn generate_qm_weights(size: usize, level: u8) -> Vec<u8> {
    // Frequency ramp: distance from DC using max(row, col) — zig-zag approximation
    let max_dist = (size - 1) as f32;
    let scale = level as f32 / 15.0; // 0 at level 0, 1 at level 15

    (0..size * size)
        .map(|idx| {
            let row = idx / size;
            let col = idx % size;
            let dist = (row.max(col)) as f32 / max_dist; // 0 = DC, 1 = Nyquist
            // High-frequency positions get lower weights at higher levels
            let weight_f32 = 255.0 - scale * dist * 192.0; // 192 = max reduction
            weight_f32.round().clamp(63.0, 255.0) as u8
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Adaptive QM selector
// ─────────────────────────────────────────────────────────────────────────────

/// Output of the adaptive QM selector.
#[derive(Debug, Clone)]
pub struct QmSelection {
    /// Chosen QM level (0–15).
    pub level: u8,
    /// Luma delta-QP (added to the base QP for the luma plane; typically 0).
    pub luma_delta_qp: i8,
    /// Chroma Cb delta-QP (AV1 supports separate chroma QP offsets).
    pub cb_delta_qp: i8,
    /// Chroma Cr delta-QP.
    pub cr_delta_qp: i8,
    /// Content class that drove this selection.
    pub content_class: ContentClass,
}

/// Converts a [`ContentClass`] and base QP into a concrete [`QmSelection`].
///
/// The selector also recommends per-plane delta-QP offsets:
/// - Luma is kept at the base QP.
/// - Chroma planes receive a +2 boost for screen content (to prevent
///   colour banding) and a –1 offset for talking-head content (faces).
#[derive(Debug, Clone, Default)]
pub struct AdaptiveQmSelector;

impl AdaptiveQmSelector {
    /// Select QM parameters for a given content class and base QP index.
    ///
    /// `base_qp` is the AV1 Q-index (0–255).  The QM level is scaled within
    /// the class's recommended range by the `base_qp`: a lower QP (better
    /// quality) uses the lower end of the range; a higher QP uses the upper end.
    #[must_use]
    pub fn select(&self, content_class: ContentClass, base_qp: u8) -> QmSelection {
        let (qm_min, qm_max) = content_class.qm_range();
        // Map base_qp 0–255 → [qm_min, qm_max]
        let level = qm_min
            + ((qm_max - qm_min) as f32 * base_qp as f32 / 255.0).round() as u8;
        let level = level.min(15);

        let (luma_delta_qp, cb_delta_qp, cr_delta_qp) = per_plane_deltas(content_class);

        QmSelection {
            level,
            luma_delta_qp,
            cb_delta_qp,
            cr_delta_qp,
            content_class,
        }
    }

    /// Select QM and build the actual [`QmMatrix`] for a given transform size.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`QmMatrix::new`] (invalid size).
    pub fn select_matrix(
        &self,
        content_class: ContentClass,
        base_qp: u8,
        transform_size: usize,
    ) -> crate::error::CodecResult<(QmSelection, QmMatrix)> {
        let sel = self.select(content_class, base_qp);
        let matrix = QmMatrix::new(transform_size, sel.level)?;
        Ok((sel, matrix))
    }
}

/// Compute per-plane delta-QP adjustments for a given content class.
fn per_plane_deltas(class: ContentClass) -> (i8, i8, i8) {
    match class {
        // Screen content: raise chroma QP slightly to reduce colour banding
        ContentClass::ScreenContent => (0, 2, 2),
        // Talking head: preserve chroma in faces with a small negative offset
        ContentClass::TalkingHead => (0, -1, -1),
        // Animation: slight chroma boost for vivid colours
        ContentClass::Animation => (0, -2, -2),
        // Default: no per-plane adjustment
        _ => (0, 0, 0),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ContentClass ─────────────────────────────────────────────────────────

    #[test]
    fn test_content_class_qm_ranges_valid() {
        let classes = [
            ContentClass::ScreenContent,
            ContentClass::Film,
            ContentClass::HighMotion,
            ContentClass::TalkingHead,
            ContentClass::Animation,
            ContentClass::Generic,
        ];
        for c in &classes {
            let (lo, hi) = c.qm_range();
            assert!(lo <= hi, "{:?}: lo={lo} > hi={hi}", c);
            assert!(hi <= 15, "{:?}: hi={hi} > 15", c);
        }
    }

    #[test]
    fn test_screen_content_lower_qm_than_talking_head() {
        let (sc_lo, sc_hi) = ContentClass::ScreenContent.qm_range();
        let (th_lo, th_hi) = ContentClass::TalkingHead.qm_range();
        assert!(sc_hi < th_lo, "ScreenContent should have lower QM than TalkingHead");
    }

    // ── FrameStats ───────────────────────────────────────────────────────────

    #[test]
    fn test_frame_stats_flat_frame() {
        // All-128 frame → zero variance, near-zero edge density
        let luma = vec![128u8; 64 * 48];
        let stats = FrameStats::from_luma(&luma, 64, 48, None);
        assert_eq!(stats.spatial_variance, 0.0);
        assert_eq!(stats.edge_density, 0.0);
        assert_eq!(stats.motion_magnitude, 0.0);
    }

    #[test]
    fn test_frame_stats_with_motion() {
        let prev = vec![0u8; 32 * 32];
        let curr = vec![128u8; 32 * 32];
        let stats = FrameStats::from_luma(&curr, 32, 32, Some(&prev));
        assert!(stats.motion_magnitude > 0.0);
    }

    #[test]
    fn test_frame_stats_empty() {
        let stats = FrameStats::from_luma(&[], 0, 0, None);
        assert_eq!(stats.spatial_variance, 0.0);
    }

    // ── ContentAnalyzer ──────────────────────────────────────────────────────

    #[test]
    fn test_content_analyzer_static_classifies_talking_head() {
        // Low everything → TalkingHead
        let stats = FrameStats {
            spatial_variance: 0.10,
            edge_density: 0.03,
            motion_magnitude: 0.01,
            palette_coverage: 0.05,
            temporal_std: 0.01,
        };
        let class = ContentAnalyzer::classify_static(&stats);
        assert_eq!(class, ContentClass::TalkingHead);
    }

    #[test]
    fn test_content_analyzer_screen_content() {
        let stats = FrameStats {
            spatial_variance: 0.30,
            edge_density: 0.20,
            motion_magnitude: 0.01,
            palette_coverage: 0.45,
            temporal_std: 0.01,
        };
        let class = ContentAnalyzer::classify_static(&stats);
        assert_eq!(class, ContentClass::ScreenContent);
    }

    #[test]
    fn test_content_analyzer_high_motion() {
        let stats = FrameStats {
            spatial_variance: 0.50,
            edge_density: 0.15,
            motion_magnitude: 0.40,
            palette_coverage: 0.05,
            temporal_std: 0.10,
        };
        let class = ContentAnalyzer::classify_static(&stats);
        assert_eq!(class, ContentClass::HighMotion);
    }

    // ── QmMatrix ─────────────────────────────────────────────────────────────

    #[test]
    fn test_qm_matrix_size_and_dc_weight() {
        let m = QmMatrix::new(8, 0).unwrap();
        assert_eq!(m.weights.len(), 64);
        // At level 0 the DC weight (top-left) should be 255
        assert_eq!(m.get(0, 0), 255);
    }

    #[test]
    fn test_qm_matrix_high_level_reduces_hf_weights() {
        let lo = QmMatrix::new(8, 0).unwrap();
        let hi = QmMatrix::new(8, 15).unwrap();
        // Bottom-right corner (highest frequency) should have lower weight at level 15
        let lo_w = lo.get(7, 7) as i32;
        let hi_w = hi.get(7, 7) as i32;
        assert!(hi_w < lo_w, "level-15 HF weight ({hi_w}) should be less than level-0 ({lo_w})");
    }

    #[test]
    fn test_qm_matrix_apply_scales_coefficients() {
        let m = QmMatrix::new(4, 15).unwrap();
        let mut coeffs = vec![128i32; 16];
        let dc_weight = m.get(0, 0);
        let hf_weight = m.get(3, 3); // bottom-right corner (Nyquist)
        m.apply(&mut coeffs);
        // DC coefficient should be boosted (weight > 64 → scaled > 128)
        let dc_scaled = 128 * dc_weight as i32 / 64;
        assert!(dc_scaled >= 128, "DC should be >= input at level 15 (weight={dc_weight})");
        // HF corner should be attenuated (weight <= 64 → scaled <= 128)
        let hf_scaled = 128 * hf_weight as i32 / 64;
        assert!(hf_scaled <= 128, "HF corner should be <= input at level 15 (weight={hf_weight})");
        // Verify apply() produced the DC value
        assert_eq!(coeffs[0], dc_scaled, "apply() DC mismatch");
    }

    #[test]
    fn test_qm_matrix_invalid_size_errors() {
        let result = QmMatrix::new(5, 0);
        assert!(result.is_err(), "size=5 should return an error");
    }

    #[test]
    fn test_qm_matrix_invalid_level_errors() {
        let result = QmMatrix::new(8, 16);
        assert!(result.is_err(), "level=16 should return an error");
    }

    // ── AdaptiveQmSelector ───────────────────────────────────────────────────

    #[test]
    fn test_selector_level_within_class_range() {
        let sel = AdaptiveQmSelector;
        for &class in &[
            ContentClass::ScreenContent,
            ContentClass::Film,
            ContentClass::HighMotion,
            ContentClass::TalkingHead,
            ContentClass::Animation,
            ContentClass::Generic,
        ] {
            let (lo, hi) = class.qm_range();
            for qp in [0u8, 64, 128, 192, 255] {
                let s = sel.select(class, qp);
                assert!(
                    s.level >= lo && s.level <= hi,
                    "{:?} qp={qp}: level {} out of [{lo},{hi}]",
                    class,
                    s.level
                );
            }
        }
    }

    #[test]
    fn test_selector_matrix_valid_size() {
        let sel = AdaptiveQmSelector;
        let (_, matrix) = sel
            .select_matrix(ContentClass::Film, 100, 8)
            .expect("should succeed for valid size");
        assert_eq!(matrix.size, 8);
    }
}
