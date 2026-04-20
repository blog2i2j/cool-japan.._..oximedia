//! Perceptual video quality metrics.
//!
//! Implements perceptually-motivated metrics including contrast sensitivity,
//! visibility thresholds, perceptual hashing, saliency-weighted PSNR, and
//! the Hasler–Süsstrunk colorfulness metric.

#![allow(dead_code)]

// ─── Contrast sensitivity function ────────────────────────────────────────────

/// Contrast sensitivity functions (CSF) for the human visual system.
pub struct ContrastSensitivity;

impl ContrastSensitivity {
    /// Spatial contrast sensitivity function (Mannos–Sakrison approximation).
    ///
    /// Returns the normalised sensitivity at `frequency_cpd` cycles per degree.
    /// Peaks around 4–6 cpd and falls off at high and low frequencies.
    #[must_use]
    pub fn spatial_csf(frequency_cpd: f32) -> f32 {
        // Mannos & Sakrison (1974) approximation:
        //   CSF(f) = 2.6 * (0.0192 + 0.114 f) * exp(-(0.114 f)^1.1)
        let f = frequency_cpd.max(0.0);
        let val = 2.6 * (0.0192 + 0.114 * f) * (-(0.114 * f).powf(1.1)).exp();
        val.max(0.0)
    }
}

// ─── Just Noticeable Difference (JND) ─────────────────────────────────────────

/// Just-noticeable difference (visibility threshold) models.
pub struct VisibilityThreshold;

impl VisibilityThreshold {
    /// Compute the JND visibility threshold.
    ///
    /// A distortion below this level is imperceptible.
    ///
    /// - `luminance` — local background luminance (0.0–1.0 normalised)
    /// - `frequency` — spatial frequency in cycles per degree
    ///
    /// Returns the minimum detectable distortion amplitude.
    #[must_use]
    pub fn compute(luminance: f32, frequency: f32) -> f32 {
        // CSF gives the maximum sensitivity; JND is its reciprocal.
        let csf = ContrastSensitivity::spatial_csf(frequency).max(1e-6);

        // Luminance masking: Weber's law — JND ∝ luminance
        let luminance_factor = (0.1 + luminance.clamp(0.0, 1.0)).powf(0.4);

        luminance_factor / csf
    }
}

// ─── Perceptual hash ──────────────────────────────────────────────────────────

/// 64-bit perceptual hash (pHash) based on the DCT of an 8×8 luma patch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PerceptualHash(pub u64);

impl PerceptualHash {
    /// Compute a pHash from a normalised 8×8 luma block.
    ///
    /// Uses the top-left 8×8 AC coefficients of the 2-D DCT.
    /// Each bit in the hash is 1 if the corresponding coefficient exceeds
    /// the median coefficient value.
    #[must_use]
    pub fn compute_dct(luma_8x8: &[f32; 64]) -> Self {
        // Compute the 2-D DCT
        let dct = dct_8x8(luma_8x8);

        // Skip DC (index 0,0); use the next 63 coefficients
        let coefficients: Vec<f32> = dct[1..].to_vec();

        let median = median_f32(&coefficients);

        let mut hash = 0u64;
        for (i, &coef) in coefficients.iter().enumerate().take(63) {
            if coef > median {
                hash |= 1u64 << i;
            }
        }

        Self(hash)
    }

    /// Compute the Hamming distance between two hashes.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        (self.0 ^ other.0).count_ones()
    }

    /// Returns `true` if the two hashes are considered visually similar
    /// (Hamming distance ≤ 10).
    #[must_use]
    pub fn is_similar_to(&self, other: &Self) -> bool {
        self.hamming_distance(other) <= 10
    }
}

// ─── Visual attention / saliency ──────────────────────────────────────────────

/// Visual attention map estimator (Itti–Koch-inspired).
pub struct VisualAttentionMap;

impl VisualAttentionMap {
    /// Compute a saliency map for a single frame.
    ///
    /// Uses a simplified centre-surround contrast model at two scales.
    ///
    /// `frame` — normalised luma samples, row-major, `width × height`.
    /// Returns a vector of the same length with values in [0, 1].
    #[must_use]
    pub fn compute_saliency(frame: &[f32], width: u32, height: u32) -> Vec<f32> {
        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        if n == 0 || frame.len() < n {
            return vec![0.0; n];
        }

        let mut saliency = vec![0.0f32; n];

        // Two-scale centre-surround: small (r=2) and medium (r=4)
        for &radius in &[2usize, 4usize] {
            for row in 0..h {
                for col in 0..w {
                    let centre = frame[row * w + col];
                    let surround = local_mean(frame, w, h, row, col, radius);
                    let response = (centre - surround).abs();
                    saliency[row * w + col] += response;
                }
            }
        }

        // Normalise to [0, 1]
        let max_val = saliency.iter().copied().fold(0.0f32, f32::max);
        if max_val > 1e-9 {
            for s in &mut saliency {
                *s /= max_val;
            }
        }

        saliency
    }
}

// ─── Saliency-weighted PSNR ───────────────────────────────────────────────────

/// PSNR metric weighted by a saliency map.
pub struct WeightedPsnr;

impl WeightedPsnr {
    /// Compute saliency-weighted PSNR.
    ///
    /// Pixels with higher saliency contribute more to the error.
    /// All slices must have the same length.
    ///
    /// Returns 100.0 if original and distorted are identical.
    #[must_use]
    pub fn compute(original: &[f32], distorted: &[f32], saliency: &[f32]) -> f32 {
        let n = original.len().min(distorted.len()).min(saliency.len());
        if n == 0 {
            return 0.0;
        }

        let mut weighted_mse = 0.0f64;
        let mut weight_sum = 0.0f64;

        for i in 0..n {
            let w = f64::from(saliency[i].max(1e-6));
            let diff = f64::from(original[i] - distorted[i]);
            weighted_mse += w * diff * diff;
            weight_sum += w;
        }

        if weight_sum < 1e-12 {
            return 0.0;
        }

        let mse = weighted_mse / weight_sum;

        if mse < 1e-12 {
            return 100.0;
        }

        // Assume normalised inputs (max = 1.0)
        10.0 * (1.0 / mse).log10() as f32
    }
}

// ─── Colorfulness metric ──────────────────────────────────────────────────────

/// Hasler–Süsstrunk colorfulness metric (2003).
pub struct ColorfulnessMetric;

impl ColorfulnessMetric {
    /// Compute the colorfulness of an image.
    ///
    /// `r`, `g`, `b` are slices of normalised (0.0–1.0) per-pixel channel
    /// values.  All three slices must have the same length.
    ///
    /// Returns a value ≥ 0; larger values indicate more colourful images:
    /// - < 15   → not colourful
    /// - 15–33  → slightly colourful
    /// - 33–45  → moderately colourful
    /// - 45–59  → quite colourful
    /// - 59–82  → highly colourful
    /// - ≥ 82   → extremely colourful
    #[must_use]
    pub fn compute(r: &[f32], g: &[f32], b: &[f32]) -> f32 {
        let n = r.len().min(g.len()).min(b.len());
        if n == 0 {
            return 0.0;
        }

        // rg = R - G, yb = 0.5*(R+G) - B
        let rg: Vec<f32> = (0..n).map(|i| r[i] - g[i]).collect();
        let yb: Vec<f32> = (0..n).map(|i| 0.5 * (r[i] + g[i]) - b[i]).collect();

        let (mean_rg, std_rg) = mean_stddev(&rg);
        let (mean_yb, std_yb) = mean_stddev(&yb);

        let std_rgyb = (std_rg * std_rg + std_yb * std_yb).sqrt();
        let mean_rgyb = (mean_rg * mean_rg + mean_yb * mean_yb).sqrt();

        // Scale from normalised [0,1] inputs to approximate 8-bit output
        (std_rgyb + 0.3 * mean_rgyb) * 255.0
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Compute the mean and standard deviation of a slice.
fn mean_stddev(values: &[f32]) -> (f32, f32) {
    let n = values.len() as f32;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// Compute the local mean of `frame` within a square radius around `(row, col)`.
fn local_mean(frame: &[f32], w: usize, h: usize, row: usize, col: usize, radius: usize) -> f32 {
    let r0 = row.saturating_sub(radius);
    let r1 = (row + radius + 1).min(h);
    let c0 = col.saturating_sub(radius);
    let c1 = (col + radius + 1).min(w);

    let mut sum = 0.0f32;
    let mut count = 0u32;

    for r in r0..r1 {
        for c in c0..c1 {
            if r == row && c == col {
                continue; // exclude centre from surround
            }
            sum += frame[r * w + c];
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

/// Compute the sorted median of a slice of f32.
fn median_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

/// Compute a 1-D DCT-II of length 8.
fn dct8(input: &[f32; 8]) -> [f32; 8] {
    let mut output = [0.0f32; 8];
    let n = 8usize;
    for k in 0..n {
        let mut sum = 0.0f32;
        for (i, &v) in input.iter().enumerate() {
            sum += v * (std::f32::consts::PI * k as f32 * (2 * i + 1) as f32 / 16.0).cos();
        }
        output[k] = sum;
    }
    output
}

/// Compute the 2-D DCT-II of an 8×8 block (row-major input).
fn dct_8x8(block: &[f32; 64]) -> [f32; 64] {
    // Apply 1-D DCT to each row
    let mut tmp = [0.0f32; 64];
    for row in 0..8 {
        let mut row_data = [0.0f32; 8];
        row_data.copy_from_slice(&block[row * 8..row * 8 + 8]);
        let row_dct = dct8(&row_data);
        tmp[row * 8..row * 8 + 8].copy_from_slice(&row_dct);
    }

    // Apply 1-D DCT to each column of `tmp`
    let mut out = [0.0f32; 64];
    for col in 0..8 {
        let mut col_data = [0.0f32; 8];
        for row in 0..8 {
            col_data[row] = tmp[row * 8 + col];
        }
        let col_dct = dct8(&col_data);
        for row in 0..8 {
            out[row * 8 + col] = col_dct[row];
        }
    }

    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ContrastSensitivity ────────────────────────────────────────────────

    #[test]
    fn test_csf_zero_frequency() {
        let s = ContrastSensitivity::spatial_csf(0.0);
        // At DC the sensitivity should be low but positive
        assert!(s >= 0.0);
    }

    #[test]
    fn test_csf_peak() {
        // Peak should be around 4–8 cpd and higher than at extremes
        let peak_region: Vec<f32> = (40..=80)
            .map(|f| ContrastSensitivity::spatial_csf(f as f32 / 10.0))
            .collect();
        let max = peak_region.iter().copied().fold(0.0f32, f32::max);
        assert!(max > 0.0, "peak CSF should be positive");
        // The peak (4–8 cpd) should be higher than at very high frequency (100 cpd)
        let high_freq = ContrastSensitivity::spatial_csf(100.0);
        assert!(
            max > high_freq,
            "peak CSF should exceed sensitivity at 100 cpd"
        );
    }

    #[test]
    fn test_csf_high_frequency_rolloff() {
        let low = ContrastSensitivity::spatial_csf(5.0);
        let high = ContrastSensitivity::spatial_csf(50.0);
        assert!(low > high, "sensitivity should drop at high frequencies");
    }

    // ── VisibilityThreshold ────────────────────────────────────────────────

    #[test]
    fn test_jnd_positive() {
        let jnd = VisibilityThreshold::compute(0.5, 4.0);
        assert!(jnd > 0.0);
    }

    #[test]
    fn test_jnd_dark_vs_bright() {
        let jnd_dark = VisibilityThreshold::compute(0.01, 4.0);
        let jnd_bright = VisibilityThreshold::compute(0.99, 4.0);
        assert!(
            jnd_bright > jnd_dark,
            "brighter backgrounds raise JND (Weber)"
        );
    }

    // ── PerceptualHash ─────────────────────────────────────────────────────

    #[test]
    fn test_phash_identical() {
        let block = [0.5f32; 64];
        let h1 = PerceptualHash::compute_dct(&block);
        let h2 = PerceptualHash::compute_dct(&block);
        assert_eq!(h1.hamming_distance(&h2), 0);
        assert!(h1.is_similar_to(&h2));
    }

    #[test]
    fn test_phash_different() {
        let mut block_a = [0.0f32; 64];
        let mut block_b = [1.0f32; 64];
        for i in 0..64 {
            block_a[i] = i as f32 / 64.0;
            block_b[i] = 1.0 - i as f32 / 64.0;
        }
        let h1 = PerceptualHash::compute_dct(&block_a);
        let h2 = PerceptualHash::compute_dct(&block_b);
        // Inverted images should have large Hamming distance
        assert!(h1.hamming_distance(&h2) > 10);
    }

    #[test]
    fn test_phash_hamming_distance_symmetry() {
        let block_a = [0.0f32; 64];
        let block_b = [1.0f32; 64];
        let h1 = PerceptualHash::compute_dct(&block_a);
        let h2 = PerceptualHash::compute_dct(&block_b);
        assert_eq!(h1.hamming_distance(&h2), h2.hamming_distance(&h1));
    }

    // ── VisualAttentionMap ─────────────────────────────────────────────────

    #[test]
    fn test_saliency_flat_frame() {
        let frame = vec![0.5f32; 16 * 16];
        let sal = VisualAttentionMap::compute_saliency(&frame, 16, 16);
        assert_eq!(sal.len(), 256);
        // Flat frame → zero contrast everywhere → all saliency = 0
        for &s in &sal {
            assert!(
                s < 1e-5,
                "expected near-zero saliency for flat frame, got {s}"
            );
        }
    }

    #[test]
    fn test_saliency_range() {
        let w = 16u32;
        let h = 16u32;
        let frame: Vec<f32> = (0..(w * h)).map(|i| (i % 2) as f32).collect();
        let sal = VisualAttentionMap::compute_saliency(&frame, w, h);
        assert_eq!(sal.len(), (w * h) as usize);
        for &s in &sal {
            assert!((0.0..=1.0).contains(&s), "saliency out of range: {s}");
        }
    }

    #[test]
    fn test_saliency_empty_frame() {
        let sal = VisualAttentionMap::compute_saliency(&[], 0, 0);
        assert!(sal.is_empty());
    }

    // ── WeightedPsnr ───────────────────────────────────────────────────────

    #[test]
    fn test_weighted_psnr_identical() {
        let sig = vec![0.5f32; 100];
        let sal = vec![1.0f32; 100];
        let psnr = WeightedPsnr::compute(&sig, &sig, &sal);
        assert!(psnr > 99.0, "identical signals should give very high PSNR");
    }

    #[test]
    fn test_weighted_psnr_positive() {
        let original = vec![1.0f32; 100];
        let distorted = vec![0.9f32; 100];
        let saliency = vec![1.0f32; 100];
        let psnr = WeightedPsnr::compute(&original, &distorted, &saliency);
        assert!(psnr > 0.0 && psnr < 100.0, "psnr={psnr}");
    }

    #[test]
    fn test_weighted_psnr_empty() {
        let psnr = WeightedPsnr::compute(&[], &[], &[]);
        assert_eq!(psnr, 0.0);
    }

    // ── ColorfulnessMetric ─────────────────────────────────────────────────

    #[test]
    fn test_colorfulness_grey() {
        // Equal R G B → no colourfulness
        let grey = vec![0.5f32; 100];
        let c = ColorfulnessMetric::compute(&grey, &grey, &grey);
        assert!(
            c < 1.0,
            "grey image should have near-zero colorfulness, got {c}"
        );
    }

    #[test]
    fn test_colorfulness_saturated_red() {
        let r = vec![1.0f32; 100];
        let g = vec![0.0f32; 100];
        let b = vec![0.0f32; 100];
        let c = ColorfulnessMetric::compute(&r, &g, &b);
        assert!(c > 10.0, "saturated red should be colourful, got {c}");
    }

    #[test]
    fn test_colorfulness_empty() {
        let c = ColorfulnessMetric::compute(&[], &[], &[]);
        assert_eq!(c, 0.0);
    }

    #[test]
    fn test_colorfulness_colourful_vs_grey() {
        let r_col = vec![1.0f32; 50]
            .into_iter()
            .chain(vec![0.0; 50])
            .collect::<Vec<_>>();
        let g_col = vec![0.0f32; 50]
            .into_iter()
            .chain(vec![1.0; 50])
            .collect::<Vec<_>>();
        let b_col = vec![0.0f32; 100];
        let grey = vec![0.5f32; 100];

        let c_col = ColorfulnessMetric::compute(&r_col, &g_col, &b_col);
        let c_grey = ColorfulnessMetric::compute(&grey, &grey, &grey);
        assert!(c_col > c_grey, "colourful image should score higher");
    }
}
