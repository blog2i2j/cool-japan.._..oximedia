//! 3D LUT data structure with interpolation and manipulation methods.

use super::types::{LutBlendMode, RgbColor};

/// Helper function for overlay blend mode.
pub(super) fn overlay_blend(base: f64, blend: f64) -> f64 {
    if base < 0.5 {
        2.0 * base * blend
    } else {
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
    }
}

/// 3D LUT data structure.
#[derive(Clone, Debug)]
pub struct Lut3d {
    /// LUT cube data stored as flat array \[R\]\[G\]\[B\].
    pub data: Vec<RgbColor>,
    /// Size of each dimension (cube is size x size x size).
    pub size: usize,
    /// Domain minimum values (default [0, 0, 0]).
    pub domain_min: RgbColor,
    /// Domain maximum values (default [1, 1, 1]).
    pub domain_max: RgbColor,
    /// LUT title/description.
    pub title: String,
}

impl Lut3d {
    /// Create a new 3D LUT with the given size.
    #[must_use]
    pub fn new(size: usize) -> Self {
        let total_size = size * size * size;
        Self {
            data: vec![RgbColor::default(); total_size],
            size,
            domain_min: RgbColor::new(0.0, 0.0, 0.0),
            domain_max: RgbColor::new(1.0, 1.0, 1.0),
            title: String::new(),
        }
    }

    /// Create an identity 3D LUT.
    #[must_use]
    pub fn identity(size: usize) -> Self {
        let mut lut = Self::new(size);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let color = RgbColor::new(
                        r as f64 / (size - 1) as f64,
                        g as f64 / (size - 1) as f64,
                        b as f64 / (size - 1) as f64,
                    );
                    lut.set(r, g, b, color);
                }
            }
        }
        lut
    }

    /// Get the linear index for the given RGB coordinates.
    #[must_use]
    pub(super) fn index(&self, r: usize, g: usize, b: usize) -> usize {
        r * self.size * self.size + g * self.size + b
    }

    /// Set a value in the LUT.
    pub fn set(&mut self, r: usize, g: usize, b: usize, color: RgbColor) {
        let idx = self.index(r, g, b);
        if idx < self.data.len() {
            self.data[idx] = color;
        }
    }

    /// Get a value from the LUT.
    #[must_use]
    pub fn get(&self, r: usize, g: usize, b: usize) -> RgbColor {
        let idx = self.index(r, g, b);
        self.data.get(idx).copied().unwrap_or_default()
    }

    /// Apply the LUT with nearest neighbor interpolation.
    #[must_use]
    pub fn apply_nearest(&self, color: RgbColor) -> RgbColor {
        // Normalize to domain
        let r_norm = ((color.r - self.domain_min.r) / (self.domain_max.r - self.domain_min.r))
            .clamp(0.0, 1.0);
        let g_norm = ((color.g - self.domain_min.g) / (self.domain_max.g - self.domain_min.g))
            .clamp(0.0, 1.0);
        let b_norm = ((color.b - self.domain_min.b) / (self.domain_max.b - self.domain_min.b))
            .clamp(0.0, 1.0);

        // Convert to LUT coordinates
        let max_idx = (self.size - 1) as f64;
        let r_idx = (r_norm * max_idx).round() as usize;
        let g_idx = (g_norm * max_idx).round() as usize;
        let b_idx = (b_norm * max_idx).round() as usize;

        self.get(r_idx, g_idx, b_idx)
    }

    /// Apply the LUT with trilinear interpolation.
    #[must_use]
    pub fn apply_trilinear(&self, color: RgbColor) -> RgbColor {
        // Normalize to domain
        let r_norm = ((color.r - self.domain_min.r) / (self.domain_max.r - self.domain_min.r))
            .clamp(0.0, 1.0);
        let g_norm = ((color.g - self.domain_min.g) / (self.domain_max.g - self.domain_min.g))
            .clamp(0.0, 1.0);
        let b_norm = ((color.b - self.domain_min.b) / (self.domain_max.b - self.domain_min.b))
            .clamp(0.0, 1.0);

        // Convert to LUT coordinates
        let max_idx = (self.size - 1) as f64;
        let r_pos = (r_norm * max_idx).clamp(0.0, max_idx);
        let g_pos = (g_norm * max_idx).clamp(0.0, max_idx);
        let b_pos = (b_norm * max_idx).clamp(0.0, max_idx);

        let r0 = r_pos.floor() as usize;
        let g0 = g_pos.floor() as usize;
        let b0 = b_pos.floor() as usize;

        let r1 = (r0 + 1).min(self.size - 1);
        let g1 = (g0 + 1).min(self.size - 1);
        let b1 = (b0 + 1).min(self.size - 1);

        let r_frac = r_pos - r0 as f64;
        let g_frac = g_pos - g0 as f64;
        let b_frac = b_pos - b0 as f64;

        // Get 8 corner values
        let c000 = self.get(r0, g0, b0);
        let c001 = self.get(r0, g0, b1);
        let c010 = self.get(r0, g1, b0);
        let c011 = self.get(r0, g1, b1);
        let c100 = self.get(r1, g0, b0);
        let c101 = self.get(r1, g0, b1);
        let c110 = self.get(r1, g1, b0);
        let c111 = self.get(r1, g1, b1);

        // Trilinear interpolation
        let c00 = c000.lerp(&c001, b_frac);
        let c01 = c010.lerp(&c011, b_frac);
        let c10 = c100.lerp(&c101, b_frac);
        let c11 = c110.lerp(&c111, b_frac);

        let c0 = c00.lerp(&c01, g_frac);
        let c1 = c10.lerp(&c11, g_frac);

        c0.lerp(&c1, r_frac)
    }

    /// Apply the LUT with tetrahedral interpolation.
    /// This method provides better quality than trilinear for color grading.
    #[must_use]
    pub fn apply_tetrahedral(&self, color: RgbColor) -> RgbColor {
        // Normalize to domain
        let r_norm = ((color.r - self.domain_min.r) / (self.domain_max.r - self.domain_min.r))
            .clamp(0.0, 1.0);
        let g_norm = ((color.g - self.domain_min.g) / (self.domain_max.g - self.domain_min.g))
            .clamp(0.0, 1.0);
        let b_norm = ((color.b - self.domain_min.b) / (self.domain_max.b - self.domain_min.b))
            .clamp(0.0, 1.0);

        // Convert to LUT coordinates
        let max_idx = (self.size - 1) as f64;
        let r_pos = (r_norm * max_idx).clamp(0.0, max_idx);
        let g_pos = (g_norm * max_idx).clamp(0.0, max_idx);
        let b_pos = (b_norm * max_idx).clamp(0.0, max_idx);

        let r0 = r_pos.floor() as usize;
        let g0 = g_pos.floor() as usize;
        let b0 = b_pos.floor() as usize;

        let r1 = (r0 + 1).min(self.size - 1);
        let g1 = (g0 + 1).min(self.size - 1);
        let b1 = (b0 + 1).min(self.size - 1);

        let r_frac = r_pos - r0 as f64;
        let g_frac = g_pos - g0 as f64;
        let b_frac = b_pos - b0 as f64;

        // Get 8 corner values
        let c000 = self.get(r0, g0, b0);
        let c001 = self.get(r0, g0, b1);
        let c010 = self.get(r0, g1, b0);
        let c011 = self.get(r0, g1, b1);
        let c100 = self.get(r1, g0, b0);
        let c101 = self.get(r1, g0, b1);
        let c110 = self.get(r1, g1, b0);
        let c111 = self.get(r1, g1, b1);

        // Tetrahedral interpolation
        // Divide the cube into 6 tetrahedra and determine which one contains the point
        if r_frac > g_frac {
            if g_frac > b_frac {
                // Tetrahedron 1: r > g > b
                let t1 = RgbColor::new(
                    c000.r + (c100.r - c000.r) * r_frac,
                    c000.g + (c100.g - c000.g) * r_frac,
                    c000.b + (c100.b - c000.b) * r_frac,
                );
                let t2 = RgbColor::new(
                    t1.r + (c110.r - c100.r) * g_frac,
                    t1.g + (c110.g - c100.g) * g_frac,
                    t1.b + (c110.b - c100.b) * g_frac,
                );
                RgbColor::new(
                    t2.r + (c111.r - c110.r) * b_frac,
                    t2.g + (c111.g - c110.g) * b_frac,
                    t2.b + (c111.b - c110.b) * b_frac,
                )
            } else if r_frac > b_frac {
                // Tetrahedron 2: r > b > g
                let t1 = RgbColor::new(
                    c000.r + (c100.r - c000.r) * r_frac,
                    c000.g + (c100.g - c000.g) * r_frac,
                    c000.b + (c100.b - c000.b) * r_frac,
                );
                let t2 = RgbColor::new(
                    t1.r + (c101.r - c100.r) * b_frac,
                    t1.g + (c101.g - c100.g) * b_frac,
                    t1.b + (c101.b - c100.b) * b_frac,
                );
                RgbColor::new(
                    t2.r + (c111.r - c101.r) * g_frac,
                    t2.g + (c111.g - c101.g) * g_frac,
                    t2.b + (c111.b - c101.b) * g_frac,
                )
            } else {
                // Tetrahedron 3: b > r > g
                let t1 = RgbColor::new(
                    c000.r + (c001.r - c000.r) * b_frac,
                    c000.g + (c001.g - c000.g) * b_frac,
                    c000.b + (c001.b - c000.b) * b_frac,
                );
                let t2 = RgbColor::new(
                    t1.r + (c101.r - c001.r) * r_frac,
                    t1.g + (c101.g - c001.g) * r_frac,
                    t1.b + (c101.b - c001.b) * r_frac,
                );
                RgbColor::new(
                    t2.r + (c111.r - c101.r) * g_frac,
                    t2.g + (c111.g - c101.g) * g_frac,
                    t2.b + (c111.b - c101.b) * g_frac,
                )
            }
        } else if b_frac > g_frac {
            // Tetrahedron 4: b > g > r
            let t1 = RgbColor::new(
                c000.r + (c001.r - c000.r) * b_frac,
                c000.g + (c001.g - c000.g) * b_frac,
                c000.b + (c001.b - c000.b) * b_frac,
            );
            let t2 = RgbColor::new(
                t1.r + (c011.r - c001.r) * g_frac,
                t1.g + (c011.g - c001.g) * g_frac,
                t1.b + (c011.b - c001.b) * g_frac,
            );
            RgbColor::new(
                t2.r + (c111.r - c011.r) * r_frac,
                t2.g + (c111.g - c011.g) * r_frac,
                t2.b + (c111.b - c011.b) * r_frac,
            )
        } else if b_frac > r_frac {
            // Tetrahedron 5: g > b > r
            let t1 = RgbColor::new(
                c000.r + (c010.r - c000.r) * g_frac,
                c000.g + (c010.g - c000.g) * g_frac,
                c000.b + (c010.b - c000.b) * g_frac,
            );
            let t2 = RgbColor::new(
                t1.r + (c011.r - c010.r) * b_frac,
                t1.g + (c011.g - c010.g) * b_frac,
                t1.b + (c011.b - c010.b) * b_frac,
            );
            RgbColor::new(
                t2.r + (c111.r - c011.r) * r_frac,
                t2.g + (c111.g - c011.g) * r_frac,
                t2.b + (c111.b - c011.b) * r_frac,
            )
        } else {
            // Tetrahedron 6: g > r > b
            let t1 = RgbColor::new(
                c000.r + (c010.r - c000.r) * g_frac,
                c000.g + (c010.g - c000.g) * g_frac,
                c000.b + (c010.b - c000.b) * g_frac,
            );
            let t2 = RgbColor::new(
                t1.r + (c110.r - c010.r) * r_frac,
                t1.g + (c110.g - c010.g) * r_frac,
                t1.b + (c110.b - c010.b) * r_frac,
            );
            RgbColor::new(
                t2.r + (c111.r - c110.r) * b_frac,
                t2.g + (c111.g - c110.g) * b_frac,
                t2.b + (c111.b - c110.b) * b_frac,
            )
        }
    }

    /// Validate the LUT for common issues.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // Check for NaN or infinite values
        for (idx, color) in self.data.iter().enumerate() {
            if !color.r.is_finite() || !color.g.is_finite() || !color.b.is_finite() {
                warnings.push(format!("Invalid value at index {idx}: {color:?}"));
            }
        }

        // Check domain
        if self.domain_min.r >= self.domain_max.r
            || self.domain_min.g >= self.domain_max.g
            || self.domain_min.b >= self.domain_max.b
        {
            warnings.push("Invalid domain range".to_string());
        }

        // Check size
        if self.size < 2 {
            warnings.push("LUT size too small".to_string());
        }

        if self.data.len() != self.size * self.size * self.size {
            warnings.push("LUT data size mismatch".to_string());
        }

        warnings
    }

    /// Invert the LUT (approximate).
    /// This creates a new LUT that approximately inverts the transformation.
    #[must_use]
    pub fn invert(&self) -> Self {
        let mut inverted = Self::new(self.size);
        inverted.domain_min = self.domain_min;
        inverted.domain_max = self.domain_max;

        // For each output color, find the input color that produces it
        // This is an approximation using grid sampling
        for r_out in 0..self.size {
            for g_out in 0..self.size {
                for b_out in 0..self.size {
                    let target = RgbColor::new(
                        r_out as f64 / (self.size - 1) as f64,
                        g_out as f64 / (self.size - 1) as f64,
                        b_out as f64 / (self.size - 1) as f64,
                    );

                    // Search for input that gives this output
                    let mut best_input = RgbColor::new(0.5, 0.5, 0.5);
                    let mut best_error = f64::MAX;

                    // Simple grid search
                    for r_in in 0..self.size {
                        for g_in in 0..self.size {
                            for b_in in 0..self.size {
                                let input = RgbColor::new(
                                    r_in as f64 / (self.size - 1) as f64,
                                    g_in as f64 / (self.size - 1) as f64,
                                    b_in as f64 / (self.size - 1) as f64,
                                );

                                let output = self.apply_trilinear(input);
                                let error = (output.r - target.r).powi(2)
                                    + (output.g - target.g).powi(2)
                                    + (output.b - target.b).powi(2);

                                if error < best_error {
                                    best_error = error;
                                    best_input = input;
                                }
                            }
                        }
                    }

                    inverted.set(r_out, g_out, b_out, best_input);
                }
            }
        }

        inverted
    }

    /// Compose two LUTs (apply first, then second).
    #[must_use]
    pub fn compose(&self, second: &Self) -> Self {
        let mut composed = Self::new(self.size);
        composed.domain_min = self.domain_min;
        composed.domain_max = self.domain_max;

        for r in 0..self.size {
            for g in 0..self.size {
                for b in 0..self.size {
                    let input = RgbColor::new(
                        r as f64 / (self.size - 1) as f64,
                        g as f64 / (self.size - 1) as f64,
                        b as f64 / (self.size - 1) as f64,
                    );

                    let intermediate = self.apply_trilinear(input);
                    let output = second.apply_trilinear(intermediate);

                    composed.set(r, g, b, output);
                }
            }
        }

        composed
    }

    /// Blend two LUTs with a given weight.
    /// weight = 0.0 returns self, weight = 1.0 returns other.
    #[must_use]
    pub fn blend(&self, other: &Self, weight: f64) -> Self {
        assert_eq!(self.size, other.size, "LUT sizes must match for blending");

        let mut result = Self::new(self.size);
        result.domain_min = self.domain_min.lerp(&other.domain_min, weight);
        result.domain_max = self.domain_max.lerp(&other.domain_max, weight);

        for i in 0..self.data.len() {
            result.data[i] = self.data[i].lerp(&other.data[i], weight);
        }

        result
    }

    /// Layer two LUTs (apply first, then second).
    /// This is similar to compose but preserves the original LUT sizes.
    #[must_use]
    pub fn layer(&self, top: &Self) -> Self {
        self.compose(top)
    }

    /// Mix two LUTs using different blend modes.
    #[must_use]
    pub fn mix(&self, other: &Self, mode: LutBlendMode, opacity: f64) -> Self {
        assert_eq!(self.size, other.size, "LUT sizes must match for mixing");

        let mut result = Self::new(self.size);
        result.domain_min = self.domain_min;
        result.domain_max = self.domain_max;

        let opacity = opacity.clamp(0.0, 1.0);

        for i in 0..self.data.len() {
            let base = self.data[i];
            let blend = other.data[i];

            let mixed = match mode {
                LutBlendMode::Normal => blend,
                LutBlendMode::Multiply => {
                    RgbColor::new(base.r * blend.r, base.g * blend.g, base.b * blend.b)
                }
                LutBlendMode::Screen => RgbColor::new(
                    1.0 - (1.0 - base.r) * (1.0 - blend.r),
                    1.0 - (1.0 - base.g) * (1.0 - blend.g),
                    1.0 - (1.0 - base.b) * (1.0 - blend.b),
                ),
                LutBlendMode::Overlay => RgbColor::new(
                    overlay_blend(base.r, blend.r),
                    overlay_blend(base.g, blend.g),
                    overlay_blend(base.b, blend.b),
                ),
                LutBlendMode::Add => RgbColor::new(
                    (base.r + blend.r).min(1.0),
                    (base.g + blend.g).min(1.0),
                    (base.b + blend.b).min(1.0),
                ),
                LutBlendMode::Subtract => RgbColor::new(
                    (base.r - blend.r).max(0.0),
                    (base.g - blend.g).max(0.0),
                    (base.b - blend.b).max(0.0),
                ),
            };

            result.data[i] = base.lerp(&mixed, opacity);
        }

        result
    }

    /// Adjust the strength/intensity of the LUT.
    #[must_use]
    pub fn adjust_strength(&self, strength: f64) -> Self {
        let identity = Self::identity(self.size);
        self.blend(&identity, 1.0 - strength.clamp(0.0, 1.0))
    }

    /// Resize the LUT to a different cube size.
    #[must_use]
    pub fn resize(&self, new_size: usize) -> Self {
        let mut result = Self::new(new_size);
        result.domain_min = self.domain_min;
        result.domain_max = self.domain_max;

        for r in 0..new_size {
            for g in 0..new_size {
                for b in 0..new_size {
                    let input = RgbColor::new(
                        r as f64 / (new_size - 1) as f64,
                        g as f64 / (new_size - 1) as f64,
                        b as f64 / (new_size - 1) as f64,
                    );

                    let output = self.apply_trilinear(input);
                    result.set(r, g, b, output);
                }
            }
        }

        result
    }
}
