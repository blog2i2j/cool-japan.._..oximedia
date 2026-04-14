//! Shot composition analysis.
//!
//! Analyses the compositional properties of a frame by examining how saliency
//! mass is distributed relative to classical compositional guidelines such as
//! the **rule of thirds**, symmetry axes, and leading lines.

#![allow(dead_code)]

/// Result of a composition analysis pass on a single frame.
#[derive(Debug, Clone)]
pub struct CompositionScore {
    /// Rule-of-thirds score (0.0–1.0).
    /// Higher values indicate saliency mass concentrated near thirds lines.
    pub rule_of_thirds: f32,
    /// Horizontal symmetry score (0.0–1.0).
    /// Higher values indicate a symmetric saliency distribution along the vertical midline.
    pub horizontal_symmetry: f32,
    /// Vertical symmetry score (0.0–1.0).
    pub vertical_symmetry: f32,
    /// Estimated centre-of-mass position as (x, y) in [0.0, 1.0].
    pub center_of_mass: (f32, f32),
}

/// Shot composition analyser.
pub struct CompositionAnalyzer;

impl CompositionAnalyzer {
    /// Compute the rule-of-thirds score for a saliency map.
    ///
    /// Measures how much of the total saliency mass falls within a band around
    /// each thirds line (vertical at x = 1/3 and x = 2/3; horizontal at y = 1/3
    /// and y = 2/3).  A band of ±`band_frac` (default ≈ 10% of the dimension)
    /// is used.
    ///
    /// # Arguments
    ///
    /// * `saliency` – per-pixel saliency values in row-major order
    /// * `w` – frame width in pixels
    /// * `h` – frame height in pixels
    ///
    /// # Returns
    ///
    /// Score in [0.0, 1.0] representing how much saliency falls near the thirds
    /// lines.  Returns 0.0 if the input is invalid.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn rule_of_thirds_score(saliency: &[f32], w: u32, h: u32) -> f32 {
        let width = w as usize;
        let height = h as usize;
        if saliency.len() < width * height || width == 0 || height == 0 {
            return 0.0;
        }

        // Band width = ~10% of dimension
        let band_x = ((width as f32 * 0.10).ceil() as usize).max(1);
        let band_y = ((height as f32 * 0.10).ceil() as usize).max(1);

        let total_sal: f32 = saliency[..width * height].iter().sum();
        if total_sal < f32::EPSILON {
            return 0.0;
        }

        // Thirds line positions
        let third_x1 = width / 3;
        let third_x2 = (2 * width) / 3;
        let third_y1 = height / 3;
        let third_y2 = (2 * height) / 3;

        let mut thirds_sal = 0.0f32;

        for y in 0..height {
            for x in 0..width {
                let near_vline = (x.abs_diff(third_x1) <= band_x)
                    || (x.abs_diff(third_x2) <= band_x);
                let near_hline = (y.abs_diff(third_y1) <= band_y)
                    || (y.abs_diff(third_y2) <= band_y);

                if near_vline || near_hline {
                    thirds_sal += saliency[y * width + x];
                }
            }
        }

        (thirds_sal / total_sal).clamp(0.0, 1.0)
    }

    /// Compute a full [`CompositionScore`] for a saliency map.
    ///
    /// In addition to the rule-of-thirds score, computes horizontal and vertical
    /// symmetry as well as the centre-of-mass.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(saliency: &[f32], w: u32, h: u32) -> CompositionScore {
        let width = w as usize;
        let height = h as usize;

        let zeros = CompositionScore {
            rule_of_thirds: 0.0,
            horizontal_symmetry: 1.0,
            vertical_symmetry: 1.0,
            center_of_mass: (0.5, 0.5),
        };

        if saliency.len() < width * height || width == 0 || height == 0 {
            return zeros;
        }

        let total_sal: f32 = saliency[..width * height].iter().sum();
        let rule_of_thirds = Self::rule_of_thirds_score(saliency, w, h);

        if total_sal < f32::EPSILON {
            return CompositionScore {
                rule_of_thirds,
                ..zeros
            };
        }

        // Centre of mass
        let mut cx = 0.0f64;
        let mut cy = 0.0f64;
        for y in 0..height {
            for x in 0..width {
                let v = f64::from(saliency[y * width + x]);
                cx += v * x as f64;
                cy += v * y as f64;
            }
        }
        let total_f64 = f64::from(total_sal);
        let com_x = (cx / total_f64 / width as f64) as f32;
        let com_y = (cy / total_f64 / height as f64) as f32;

        // Horizontal symmetry: compare left half saliency to right half
        let h_sym = compute_horizontal_symmetry(saliency, width, height);
        // Vertical symmetry: compare top half to bottom half
        let v_sym = compute_vertical_symmetry(saliency, width, height);

        CompositionScore {
            rule_of_thirds,
            horizontal_symmetry: h_sym,
            vertical_symmetry: v_sym,
            center_of_mass: (com_x, com_y),
        }
    }
}

/// Compute horizontal symmetry score: how similar is the left half to the mirrored right half?
#[allow(clippy::cast_precision_loss)]
fn compute_horizontal_symmetry(saliency: &[f32], width: usize, height: usize) -> f32 {
    let half = width / 2;
    if half == 0 {
        return 1.0;
    }

    let mut diff_sum = 0.0f64;
    let mut total = 0.0f64;

    for y in 0..height {
        for x in 0..half {
            let lx = x;
            let rx = width - 1 - x;
            let left = f64::from(saliency[y * width + lx]);
            let right = f64::from(saliency[y * width + rx]);
            diff_sum += (left - right).abs();
            total += left + right + f64::EPSILON;
        }
    }

    let asymmetry = (diff_sum / total) as f32;
    (1.0 - asymmetry).clamp(0.0, 1.0)
}

/// Compute vertical symmetry score.
#[allow(clippy::cast_precision_loss)]
fn compute_vertical_symmetry(saliency: &[f32], width: usize, height: usize) -> f32 {
    let half = height / 2;
    if half == 0 {
        return 1.0;
    }

    let mut diff_sum = 0.0f64;
    let mut total = 0.0f64;

    for y in 0..half {
        for x in 0..width {
            let ty = y;
            let by = height - 1 - y;
            let top = f64::from(saliency[ty * width + x]);
            let bot = f64::from(saliency[by * width + x]);
            diff_sum += (top - bot).abs();
            total += top + bot + f64::EPSILON;
        }
    }

    let asymmetry = (diff_sum / total) as f32;
    (1.0 - asymmetry).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_saliency(w: usize, h: usize) -> Vec<f32> {
        vec![1.0f32 / (w * h) as f32; w * h]
    }

    #[test]
    fn test_rule_of_thirds_zero_on_empty() {
        let score = CompositionAnalyzer::rule_of_thirds_score(&[], 0, 0);
        assert!((score - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rule_of_thirds_nonzero_for_uniform_frame() {
        // For a uniform saliency map, thirds lines cover ~40% of the area
        // so the score should be noticeably > 0.
        let w = 30u32;
        let h = 30u32;
        let sal = uniform_saliency(w as usize, h as usize);
        let score = CompositionAnalyzer::rule_of_thirds_score(&sal, w, h);
        assert!(score > 0.0, "uniform frame should have non-zero score, got {score}");
    }

    #[test]
    fn test_rule_of_thirds_concentrated_at_third() {
        // Put all saliency exactly on the 1/3 column
        let w = 30usize;
        let h = 30usize;
        let third_x = w / 3;
        let mut sal = vec![0.0f32; w * h];
        for y in 0..h {
            sal[y * w + third_x] = 1.0;
        }
        let score = CompositionAnalyzer::rule_of_thirds_score(&sal, w as u32, h as u32);
        assert!(score > 0.5, "score should be high when mass is on thirds line, got {score}");
    }

    #[test]
    fn test_analyze_uniform_centre_of_mass_near_centre() {
        let w = 32u32;
        let h = 32u32;
        let sal = uniform_saliency(w as usize, h as usize);
        let result = CompositionAnalyzer::analyze(&sal, w, h);
        let (cx, cy) = result.center_of_mass;
        assert!((cx - 0.5).abs() < 0.05, "cx={cx}");
        assert!((cy - 0.5).abs() < 0.05, "cy={cy}");
    }

    #[test]
    fn test_analyze_symmetric_frame_high_symmetry() {
        // Build a symmetric saliency map (same left and right halves)
        let w = 20usize;
        let h = 20usize;
        let mut sal = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w / 2 {
                sal[y * w + x] = 0.5;
                sal[y * w + (w - 1 - x)] = 0.5;
            }
        }
        let result = CompositionAnalyzer::analyze(&sal, w as u32, h as u32);
        assert!(
            result.horizontal_symmetry > 0.8,
            "symmetric frame should score high, got {}",
            result.horizontal_symmetry
        );
    }

    #[test]
    fn test_analyze_zero_saliency_returns_defaults() {
        let sal = vec![0.0f32; 100];
        let result = CompositionAnalyzer::analyze(&sal, 10, 10);
        assert!((result.rule_of_thirds - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_score_in_range() {
        let w = 16u32;
        let h = 16u32;
        let sal: Vec<f32> = (0..256).map(|i| (i % 17) as f32 / 16.0).collect();
        let score = CompositionAnalyzer::rule_of_thirds_score(&sal, w, h);
        assert!(score >= 0.0 && score <= 1.0, "score out of range: {score}");
    }
}
