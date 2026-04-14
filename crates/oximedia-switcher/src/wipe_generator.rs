#![allow(dead_code)]
//! Wipe pattern generation for video switcher transitions.
//!
//! Generates per-pixel transition masks for a variety of wipe patterns
//! used in live production switchers. Each pattern produces a normalized
//! [0.0, 1.0] mask that drives the mix between program and preview sources.

/// Standard SMPTE wipe pattern identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WipePatternId {
    /// Horizontal bar wipe (left to right).
    HorizontalBar,
    /// Vertical bar wipe (top to bottom).
    VerticalBar,
    /// Diagonal wipe (top-left to bottom-right).
    DiagonalTlBr,
    /// Diagonal wipe (top-right to bottom-left).
    DiagonalTrBl,
    /// Circle iris from center.
    CircleIris,
    /// Diamond iris from center.
    DiamondIris,
    /// Box iris from center.
    BoxIris,
    /// Horizontal blinds (venetian).
    HorizontalBlinds,
    /// Vertical blinds.
    VerticalBlinds,
    /// Clock wipe (sweeping from 12 o'clock).
    ClockWipe,
    /// Heart shape wipe.
    HeartWipe,
    /// Star shape wipe.
    StarWipe,
    /// Cross wipe.
    CrossWipe,
}

/// Parameters for wipe pattern generation.
#[derive(Debug, Clone)]
pub struct WipeParams {
    /// Width of the output frame in pixels.
    pub width: u32,
    /// Height of the output frame in pixels.
    pub height: u32,
    /// Edge softness in pixels (feather radius).
    pub softness: f32,
    /// Pattern rotation in degrees (0-360).
    pub rotation_deg: f32,
    /// Center X offset (-1.0 to 1.0, 0 = center).
    pub center_x: f32,
    /// Center Y offset (-1.0 to 1.0, 0 = center).
    pub center_y: f32,
    /// Number of segments for blinds/star patterns.
    pub segments: u32,
    /// Whether to reverse (invert) the wipe direction.
    pub reverse: bool,
}

impl WipeParams {
    /// Create default wipe parameters for a given resolution.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            softness: 4.0,
            rotation_deg: 0.0,
            center_x: 0.0,
            center_y: 0.0,
            segments: 6,
            reverse: false,
        }
    }

    /// Validate the parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.width == 0 || self.height == 0 {
            return Err("Width and height must be > 0".to_string());
        }
        if self.softness < 0.0 {
            return Err("Softness must be non-negative".to_string());
        }
        if self.segments == 0 {
            return Err("Segments must be > 0".to_string());
        }
        Ok(())
    }
}

/// A generated wipe mask for one frame.
#[derive(Debug, Clone)]
pub struct WipeMask {
    /// Per-pixel transition value in [0.0, 1.0].
    pub data: Vec<f32>,
    /// Width of the mask.
    pub width: u32,
    /// Height of the mask.
    pub height: u32,
}

impl WipeMask {
    /// Create a new empty mask.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            data: vec![0.0; (width as usize) * (height as usize)],
            width,
            height,
        }
    }

    /// Get the value at pixel (x, y).
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x < self.width && y < self.height {
            self.data[(y as usize) * (self.width as usize) + (x as usize)]
        } else {
            0.0
        }
    }

    /// Set the value at pixel (x, y).
    pub fn set(&mut self, x: u32, y: u32, value: f32) {
        if x < self.width && y < self.height {
            self.data[(y as usize) * (self.width as usize) + (x as usize)] = value;
        }
    }

    /// Invert the mask (1.0 - value).
    pub fn invert(&mut self) {
        for v in &mut self.data {
            *v = 1.0 - *v;
        }
    }

    /// Total number of pixels.
    pub fn pixel_count(&self) -> usize {
        self.data.len()
    }
}

/// Wipe pattern generator.
#[derive(Debug)]
pub struct WipeGenerator {
    /// Pattern to generate.
    pattern: WipePatternId,
    /// Generation parameters.
    params: WipeParams,
}

impl WipeGenerator {
    /// Create a new wipe generator for the given pattern.
    pub fn new(pattern: WipePatternId, params: WipeParams) -> Self {
        Self { pattern, params }
    }

    /// Generate a wipe mask at a given transition progress (0.0 to 1.0).
    ///
    /// `progress` of 0.0 means fully program, 1.0 means fully preview.
    #[allow(clippy::cast_precision_loss)]
    pub fn generate(&self, progress: f32) -> WipeMask {
        let progress = progress.clamp(0.0, 1.0);
        let w = self.params.width;
        let h = self.params.height;
        let mut mask = WipeMask::new(w, h);

        for y in 0..h {
            for x in 0..w {
                // Normalized coordinates [-1, 1]
                let nx = (x as f32 / w as f32) * 2.0 - 1.0 - self.params.center_x;
                let ny = (y as f32 / h as f32) * 2.0 - 1.0 - self.params.center_y;

                let raw = self.compute_pattern_value(nx, ny, progress);
                let value = self.apply_softness(raw, progress);
                let value = if self.params.reverse {
                    1.0 - value
                } else {
                    value
                };

                mask.set(x, y, value.clamp(0.0, 1.0));
            }
        }

        mask
    }

    /// Compute the raw pattern distance value for a normalized coordinate.
    #[allow(clippy::cast_precision_loss)]
    fn compute_pattern_value(&self, nx: f32, ny: f32, progress: f32) -> f32 {
        match self.pattern {
            WipePatternId::HorizontalBar => {
                let threshold = progress * 2.0 - 1.0;
                -(nx - threshold)
            }
            WipePatternId::VerticalBar => {
                let threshold = progress * 2.0 - 1.0;
                -(ny - threshold)
            }
            WipePatternId::DiagonalTlBr => {
                let diag = (nx + ny) / 2.0;
                let threshold = progress * 2.0 - 1.0;
                -(diag - threshold)
            }
            WipePatternId::DiagonalTrBl => {
                let diag = (-nx + ny) / 2.0;
                let threshold = progress * 2.0 - 1.0;
                -(diag - threshold)
            }
            WipePatternId::CircleIris => {
                let dist = (nx * nx + ny * ny).sqrt();
                let radius = progress * std::f32::consts::SQRT_2;
                radius - dist
            }
            WipePatternId::DiamondIris => {
                let dist = nx.abs() + ny.abs();
                let radius = progress * 2.0;
                radius - dist
            }
            WipePatternId::BoxIris => {
                let dist = nx.abs().max(ny.abs());
                let radius = progress;
                radius - dist
            }
            WipePatternId::HorizontalBlinds => {
                let segments = self.params.segments as f32;
                let phase = ((ny + 1.0) / 2.0 * segments).fract();
                progress - phase
            }
            WipePatternId::VerticalBlinds => {
                let segments = self.params.segments as f32;
                let phase = ((nx + 1.0) / 2.0 * segments).fract();
                progress - phase
            }
            WipePatternId::ClockWipe => {
                let angle = ny.atan2(nx); // -PI to PI
                let normalized_angle =
                    (angle + std::f32::consts::FRAC_PI_2 + std::f32::consts::TAU)
                        % std::f32::consts::TAU
                        / std::f32::consts::TAU;
                progress - normalized_angle
            }
            WipePatternId::HeartWipe => {
                // Simplified heart shape using distance field
                let x2 = nx;
                let y2 = ny - 0.3;
                let dist = (x2 * x2 + y2 * y2).sqrt() - 0.5 + 0.3 * ((3.0 * x2.atan2(y2)).sin());
                let radius = progress * 2.0;
                radius - dist.max(0.0)
            }
            WipePatternId::StarWipe => {
                let angle = ny.atan2(nx);
                let segments = self.params.segments as f32;
                let star_dist = 0.5 + 0.3 * (angle * segments).cos();
                let dist = (nx * nx + ny * ny).sqrt();
                let radius = progress * 2.0 * star_dist;
                radius - dist
            }
            WipePatternId::CrossWipe => {
                let dist_h = ny.abs();
                let dist_v = nx.abs();
                let cross_dist = dist_h.min(dist_v);
                let radius = progress;
                radius - cross_dist
            }
        }
    }

    /// Apply edge softness to the raw distance value.
    fn apply_softness(&self, raw: f32, _progress: f32) -> f32 {
        let softness = self.params.softness;
        if softness <= 0.0 {
            if raw >= 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            let norm_softness = softness / self.params.width.max(1) as f32 * 4.0;
            (raw / norm_softness + 0.5).clamp(0.0, 1.0)
        }
    }

    /// Get the current pattern.
    pub fn pattern(&self) -> WipePatternId {
        self.pattern
    }

    /// Set the pattern.
    pub fn set_pattern(&mut self, pattern: WipePatternId) {
        self.pattern = pattern;
    }

    /// Get the parameters.
    pub fn params(&self) -> &WipeParams {
        &self.params
    }

    /// Set the parameters.
    pub fn set_params(&mut self, params: WipeParams) {
        self.params = params;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wipe_params_default() {
        let params = WipeParams::new(1920, 1080);
        assert_eq!(params.width, 1920);
        assert_eq!(params.height, 1080);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_wipe_params_validation_zero_size() {
        let params = WipeParams::new(0, 1080);
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_wipe_mask_creation() {
        let mask = WipeMask::new(100, 100);
        assert_eq!(mask.pixel_count(), 10000);
        assert!((mask.get(0, 0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_wipe_mask_set_get() {
        let mut mask = WipeMask::new(10, 10);
        mask.set(5, 5, 0.75);
        assert!((mask.get(5, 5) - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_wipe_mask_out_of_bounds() {
        let mask = WipeMask::new(10, 10);
        assert!((mask.get(100, 100) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_wipe_mask_invert() {
        let mut mask = WipeMask::new(2, 2);
        mask.set(0, 0, 0.3);
        mask.set(1, 0, 0.7);
        mask.invert();
        assert!((mask.get(0, 0) - 0.7).abs() < 1e-5);
        assert!((mask.get(1, 0) - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_horizontal_bar_full_progress() {
        let params = WipeParams::new(100, 100);
        let gen = WipeGenerator::new(WipePatternId::HorizontalBar, params);
        let mask = gen.generate(1.0);
        // At full progress, all pixels should be fully transitioned
        for &v in &mask.data {
            assert!(v > 0.5, "Expected > 0.5, got {v}");
        }
    }

    #[test]
    fn test_horizontal_bar_zero_progress() {
        let params = WipeParams::new(100, 100);
        let gen = WipeGenerator::new(WipePatternId::HorizontalBar, params);
        let mask = gen.generate(0.0);
        // At zero progress, the vast majority of pixels should be untransitioned
        // (the single boundary pixel at nx=-1.0 may be at exactly the threshold).
        let transitioned: usize = mask.data.iter().filter(|&&v| v > 0.5).count();
        let total = mask.data.len();
        // At most 1% of pixels (boundary column) may be at the transition edge.
        assert!(
            transitioned <= total / 100,
            "Too many transitioned pixels at zero progress: {transitioned}/{total}"
        );
    }

    #[test]
    fn test_circle_iris_center() {
        let params = WipeParams::new(100, 100);
        let gen = WipeGenerator::new(WipePatternId::CircleIris, params);
        let mask = gen.generate(0.5);
        // Center should be transitioned
        let center_val = mask.get(50, 50);
        assert!(center_val > 0.0);
    }

    #[test]
    fn test_reverse_wipe() {
        let mut params = WipeParams::new(50, 50);
        params.reverse = false;
        let gen_normal = WipeGenerator::new(WipePatternId::HorizontalBar, params.clone());
        let mask_normal = gen_normal.generate(0.5);

        params.reverse = true;
        let gen_reversed = WipeGenerator::new(WipePatternId::HorizontalBar, params);
        let mask_reversed = gen_reversed.generate(0.5);

        // Reversed should be complement
        for i in 0..mask_normal.data.len() {
            let sum = mask_normal.data[i] + mask_reversed.data[i];
            assert!((sum - 1.0).abs() < 1e-4, "Expected sum ~1.0, got {sum}");
        }
    }

    #[test]
    fn test_wipe_generator_set_pattern() {
        let params = WipeParams::new(100, 100);
        let mut gen = WipeGenerator::new(WipePatternId::HorizontalBar, params);
        assert_eq!(gen.pattern(), WipePatternId::HorizontalBar);
        gen.set_pattern(WipePatternId::CircleIris);
        assert_eq!(gen.pattern(), WipePatternId::CircleIris);
    }

    #[test]
    fn test_all_patterns_produce_valid_masks() {
        let patterns = [
            WipePatternId::HorizontalBar,
            WipePatternId::VerticalBar,
            WipePatternId::DiagonalTlBr,
            WipePatternId::CircleIris,
            WipePatternId::DiamondIris,
            WipePatternId::BoxIris,
            WipePatternId::HorizontalBlinds,
            WipePatternId::ClockWipe,
            WipePatternId::StarWipe,
            WipePatternId::CrossWipe,
        ];

        for pattern in &patterns {
            let params = WipeParams::new(32, 32);
            let gen = WipeGenerator::new(*pattern, params);
            let mask = gen.generate(0.5);
            for &v in &mask.data {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "Pattern {pattern:?} produced out-of-range value {v}"
                );
            }
        }
    }

    #[test]
    fn test_progress_clamping() {
        let params = WipeParams::new(10, 10);
        let gen = WipeGenerator::new(WipePatternId::HorizontalBar, params);
        // Should not panic with out-of-range progress
        let _m1 = gen.generate(-0.5);
        let _m2 = gen.generate(1.5);
    }
}
