//! Texture analysis using Gray-Level Co-occurrence Matrices (GLCM),
// Haralick feature extraction, and Local Binary Patterns (LBP).
//
// # Algorithms
//
// ## GLCM (Gray-Level Co-occurrence Matrix)
// Counts how often pairs of pixel intensities co-occur at a given spatial
// offset (distance + angle).  The matrix is normalised to form a joint
// probability distribution and then summarised by Haralick features.
//
// ## Haralick Features (14 standard)
// Energy, contrast, homogeneity (IDM), correlation, entropy, dissimilarity,
// cluster shade, cluster prominence, max probability, variance, mean, sum
// average, sum variance, sum entropy — computed directly from the GLCM.
//
// ## Local Binary Patterns (LBP)
// For each pixel, compares the centre with its `P` circularly-sampled
// neighbours at radius `R`.  Each comparison contributes one bit to a binary
// code.  The module provides both the raw `P`-bit code and the
// *uniform* LBP variant (at most 2 bit transitions → spatially compact).
// A histogram over the LBP codes forms a compact texture descriptor.

// ---------------------------------------------------------------------------
// Quantisation helper
// ---------------------------------------------------------------------------

/// Quantise a float pixel value in [0, 1] to a gray level in 0..levels.
#[inline]
fn quantise(v: f64, levels: usize) -> usize {
    ((v * (levels as f64 - 1.0)).round() as usize).min(levels - 1)
}

// ---------------------------------------------------------------------------
// GLCM
// ---------------------------------------------------------------------------

/// Direction of the co-occurrence offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GlcmDirection {
    /// 0° — right neighbour (dx=1, dy=0).
    East,
    /// 45° — upper-right (dx=1, dy=-1).
    NorthEast,
    /// 90° — upper neighbour (dx=0, dy=-1).
    North,
    /// 135° — upper-left (dx=-1, dy=-1).
    NorthWest,
}

impl GlcmDirection {
    /// Return the (dx, dy) offset for this direction.
    #[must_use]
    pub const fn offset(self, distance: usize) -> (i32, i32) {
        let d = distance as i32;
        match self {
            Self::East => (d, 0),
            Self::NorthEast => (d, -d),
            Self::North => (0, -d),
            Self::NorthWest => (-d, -d),
        }
    }

    /// All four primary directions.
    pub fn all() -> [Self; 4] {
        [Self::East, Self::NorthEast, Self::North, Self::NorthWest]
    }
}

/// A Gray-Level Co-occurrence Matrix.
///
/// The matrix is square with side `levels` and holds counts of co-occurring
/// pixel intensity pairs.  Call [`GlcmMatrix::normalize`] to convert to
/// probabilities, then use [`HaralickFeatures::compute`] to extract features.
#[derive(Clone, Debug)]
pub struct GlcmMatrix {
    /// Raw counts (or probabilities after normalization).
    pub matrix: Vec<f64>,
    /// Number of gray levels (matrix is `levels × levels`).
    pub levels: usize,
}

impl GlcmMatrix {
    /// Create a new zeroed GLCM with `levels` gray levels.
    #[must_use]
    pub fn new(levels: usize) -> Self {
        Self {
            matrix: vec![0.0; levels * levels],
            levels,
        }
    }

    /// Get a mutable reference to element (i, j).
    fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut f64> {
        if i < self.levels && j < self.levels {
            Some(&mut self.matrix[i * self.levels + j])
        } else {
            None
        }
    }

    /// Get element (i, j).
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i < self.levels && j < self.levels {
            self.matrix[i * self.levels + j]
        } else {
            0.0
        }
    }

    /// Increment count at (i, j).
    fn increment(&mut self, i: usize, j: usize) {
        if let Some(v) = self.get_mut(i, j) {
            *v += 1.0;
        }
    }

    /// Normalize counts to probabilities (sum to 1).
    pub fn normalize(&mut self) {
        let total: f64 = self.matrix.iter().sum();
        if total > 0.0 {
            for v in &mut self.matrix {
                *v /= total;
            }
        }
    }

    /// Return a symmetrized copy (P\[i\]\[j\] = (count\[i\]\[j\] + count\[j\]\[i\]) / 2).
    #[must_use]
    pub fn symmetrize(&self) -> Self {
        let mut sym = Self::new(self.levels);
        for i in 0..self.levels {
            for j in 0..self.levels {
                let v = (self.get(i, j) + self.get(j, i)) / 2.0;
                sym.matrix[i * self.levels + j] = v;
            }
        }
        sym
    }
}

/// Configuration for GLCM computation.
#[derive(Clone, Debug)]
pub struct GlcmConfig {
    /// Number of gray levels to quantise to.
    pub levels: usize,
    /// Co-occurrence distance (pixels).
    pub distance: usize,
    /// Directions to include.
    pub directions: Vec<GlcmDirection>,
    /// Symmetrize the matrix after accumulation.
    pub symmetrize: bool,
}

impl Default for GlcmConfig {
    fn default() -> Self {
        Self {
            levels: 8,
            distance: 1,
            directions: GlcmDirection::all().to_vec(),
            symmetrize: true,
        }
    }
}

impl GlcmConfig {
    /// Create a new config.
    #[must_use]
    pub fn new(levels: usize) -> Self {
        Self {
            levels,
            ..Default::default()
        }
    }

    /// Set distance.
    #[must_use]
    pub fn with_distance(mut self, d: usize) -> Self {
        self.distance = d;
        self
    }

    /// Set directions.
    #[must_use]
    pub fn with_directions(mut self, dirs: Vec<GlcmDirection>) -> Self {
        self.directions = dirs;
        self
    }
}

/// Compute a GLCM from a row-major float image (values in [0, 1]).
///
/// Pixels outside the image boundary are ignored.
#[must_use]
pub fn compute_glcm(
    pixels: &[f64],
    width: usize,
    height: usize,
    config: &GlcmConfig,
) -> GlcmMatrix {
    let mut glcm = GlcmMatrix::new(config.levels);

    for &dir in &config.directions {
        let (dx, dy) = dir.offset(config.distance);
        for y in 0..height {
            for x in 0..width {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                    continue;
                }
                let i = quantise(pixels[y * width + x], config.levels);
                let j = quantise(pixels[ny as usize * width + nx as usize], config.levels);
                glcm.increment(i, j);
            }
        }
    }

    if config.symmetrize {
        glcm = glcm.symmetrize();
    }

    glcm.normalize();
    glcm
}

// ---------------------------------------------------------------------------
// Haralick features
// ---------------------------------------------------------------------------

/// The 14 Haralick texture features derived from a normalised GLCM.
#[derive(Clone, Debug, Default)]
pub struct HaralickFeatures {
    /// Angular second moment / energy: Σ p(i,j)².
    pub energy: f64,
    /// Contrast: Σ (i-j)² p(i,j).
    pub contrast: f64,
    /// Inverse difference moment / homogeneity: Σ p(i,j) / (1 + |i-j|).
    pub homogeneity: f64,
    /// Correlation: measures linear dependency of gray levels.
    pub correlation: f64,
    /// Entropy: -Σ p(i,j) log(p(i,j)+ε).
    pub entropy: f64,
    /// Dissimilarity: Σ |i-j| p(i,j).
    pub dissimilarity: f64,
    /// Cluster shade: Σ (i + j - 2μ)³ p(i,j).
    pub cluster_shade: f64,
    /// Cluster prominence: Σ (i + j - 2μ)⁴ p(i,j).
    pub cluster_prominence: f64,
    /// Maximum probability: max p(i,j).
    pub max_probability: f64,
    /// Variance: Σ (i - μ)² p(i,j).
    pub variance: f64,
    /// Mean (μx): Σ i · Σⱼ p(i,j).
    pub mean: f64,
    /// Sum average: E[p_{x+y}].
    pub sum_average: f64,
    /// Sum variance: Var[p_{x+y}].
    pub sum_variance: f64,
    /// Sum entropy: H[p_{x+y}].
    pub sum_entropy: f64,
}

impl HaralickFeatures {
    /// Compute all Haralick features from a *normalised* GLCM.
    #[must_use]
    pub fn compute(glcm: &GlcmMatrix) -> Self {
        let l = glcm.levels;
        let p = &glcm.matrix;

        // Marginal probabilities px[i] = Σⱼ p(i,j)
        let mut px = vec![0.0f64; l];
        let mut py = vec![0.0f64; l];
        for i in 0..l {
            for j in 0..l {
                let v = p[i * l + j];
                px[i] += v;
                py[j] += v;
            }
        }

        let mean_x: f64 = (0..l).map(|i| i as f64 * px[i]).sum();
        let mean_y: f64 = (0..l).map(|j| j as f64 * py[j]).sum();
        let mu = (mean_x + mean_y) / 2.0;

        let sigma_x: f64 = (0..l)
            .map(|i| (i as f64 - mean_x).powi(2) * px[i])
            .sum::<f64>()
            .sqrt();
        let sigma_y: f64 = (0..l)
            .map(|j| (j as f64 - mean_y).powi(2) * py[j])
            .sum::<f64>()
            .sqrt();

        // p_{x+y} for k = 2..=2*(l-1)
        let max_k = 2 * (l - 1);
        let mut p_sum = vec![0.0f64; max_k + 1];
        for i in 0..l {
            for j in 0..l {
                let k = i + j;
                p_sum[k] += p[i * l + j];
            }
        }

        let mut feats = Self::default();

        for i in 0..l {
            for j in 0..l {
                let v = p[i * l + j];
                let diff = (i as f64 - j as f64).abs();

                feats.energy += v * v;
                feats.contrast += diff * diff * v;
                feats.homogeneity += v / (1.0 + diff);
                feats.dissimilarity += diff * v;
                feats.entropy -= if v > 0.0 { v * v.ln() } else { 0.0 };
                feats.max_probability = feats.max_probability.max(v);
                feats.variance += (i as f64 - mu).powi(2) * v;
                feats.cluster_shade += (i as f64 + j as f64 - 2.0 * mu).powi(3) * v;
                feats.cluster_prominence += (i as f64 + j as f64 - 2.0 * mu).powi(4) * v;

                if sigma_x > 1e-10 && sigma_y > 1e-10 {
                    feats.correlation +=
                        (i as f64 - mean_x) * (j as f64 - mean_y) * v / (sigma_x * sigma_y);
                }
            }
        }

        // Sum-based features
        let sum_avg: f64 = (0..=max_k).map(|k| k as f64 * p_sum[k]).sum();
        feats.sum_average = sum_avg;

        feats.sum_variance = (0..=max_k)
            .map(|k| (k as f64 - sum_avg).powi(2) * p_sum[k])
            .sum();

        feats.sum_entropy = -(0..=max_k)
            .map(|k| {
                let v = p_sum[k];
                if v > 0.0 {
                    v * v.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        feats.mean = mean_x;

        feats
    }

    /// Return all features as a fixed-length vector (useful for ML pipelines).
    #[must_use]
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.energy,
            self.contrast,
            self.homogeneity,
            self.correlation,
            self.entropy,
            self.dissimilarity,
            self.cluster_shade,
            self.cluster_prominence,
            self.max_probability,
            self.variance,
            self.mean,
            self.sum_average,
            self.sum_variance,
            self.sum_entropy,
        ]
    }
}

// ---------------------------------------------------------------------------
// Local Binary Patterns
// ---------------------------------------------------------------------------

/// LBP variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LbpVariant {
    /// Basic circular LBP: `P`-bit binary code.
    Basic,
    /// Uniform LBP: codes with at most 2 bit transitions; others mapped to a
    /// single "non-uniform" bin.
    Uniform,
}

/// Configuration for LBP computation.
#[derive(Clone, Debug)]
pub struct LbpConfig {
    /// Number of sampling points on the circle.
    pub points: usize,
    /// Radius of the sampling circle (in pixels).
    pub radius: f64,
    /// Which LBP variant to compute.
    pub variant: LbpVariant,
}

impl Default for LbpConfig {
    fn default() -> Self {
        Self {
            points: 8,
            radius: 1.0,
            variant: LbpVariant::Uniform,
        }
    }
}

impl LbpConfig {
    /// Create a new config.
    #[must_use]
    pub fn new(points: usize, radius: f64) -> Self {
        Self {
            points,
            radius,
            ..Default::default()
        }
    }

    /// Set the variant.
    #[must_use]
    pub fn with_variant(mut self, v: LbpVariant) -> Self {
        self.variant = v;
        self
    }
}

/// Count the number of 0→1 or 1→0 bit transitions in the LBP code.
///
/// The code is treated as circular (the last bit wraps to the first).
fn count_transitions(code: u64, points: usize) -> usize {
    let mut transitions = 0;
    for i in 0..points {
        let current = (code >> i) & 1;
        let next = (code >> ((i + 1) % points)) & 1;
        if current != next {
            transitions += 1;
        }
    }
    transitions
}

/// Map an LBP code to a uniform code index.
///
/// Uniform patterns (≤ 2 transitions) each get a unique index 0..num_uniform.
/// Non-uniform patterns are all mapped to index `num_uniform`.
fn uniform_index(code: u64, points: usize) -> usize {
    // There are P*(P-1)+3 uniform bins: one per rotation class + non-uniform
    // The simple mapping: count 1-bits (popcount) for uniform patterns.
    if count_transitions(code, points) <= 2 {
        code.count_ones() as usize
    } else {
        points + 1 // non-uniform bin
    }
}

/// Bilinear interpolation into a flat row-major pixel buffer.
fn bilinear(pixels: &[f64], width: usize, height: usize, x: f64, y: f64) -> f64 {
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let safe = |px: i64, py: i64| -> f64 {
        if px >= 0 && px < width as i64 && py >= 0 && py < height as i64 {
            pixels[py as usize * width + px as usize]
        } else {
            0.0
        }
    };

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let v00 = safe(x0, y0);
    let v10 = safe(x1, y0);
    let v01 = safe(x0, y1);
    let v11 = safe(x1, y1);

    v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy + v11 * fx * fy
}

/// Compute LBP codes for every pixel in a row-major float image (values in
/// [0, 1]).  Returns a flat code vector of length `width * height`.
#[must_use]
pub fn compute_lbp(pixels: &[f64], width: usize, height: usize, config: &LbpConfig) -> Vec<u64> {
    let p = config.points;
    let r = config.radius;
    let mut codes = vec![0u64; width * height];

    // Precompute sampling angles
    let angles: Vec<f64> = (0..p)
        .map(|k| 2.0 * std::f64::consts::PI * k as f64 / p as f64)
        .collect();

    for y in 0..height {
        for x in 0..width {
            let centre = pixels[y * width + x];
            let mut code = 0u64;
            for (k, &angle) in angles.iter().enumerate() {
                let nx = x as f64 + r * angle.cos();
                let ny = y as f64 - r * angle.sin(); // image y-axis is flipped
                let neighbour = bilinear(pixels, width, height, nx, ny);
                if neighbour >= centre {
                    code |= 1 << k;
                }
            }
            codes[y * width + x] = code;
        }
    }

    codes
}

/// Compute a normalised LBP histogram (texture descriptor).
///
/// For `Basic` variant: bins 0..2^P.
/// For `Uniform` variant: bins 0..=P+1 (P+1 is the non-uniform bin).
#[must_use]
pub fn lbp_histogram(pixels: &[f64], width: usize, height: usize, config: &LbpConfig) -> Vec<f64> {
    let codes = compute_lbp(pixels, width, height, config);
    let p = config.points;

    let num_bins = match config.variant {
        LbpVariant::Basic => 1 << p,
        LbpVariant::Uniform => p + 2, // 0..=P (popcount) + non-uniform
    };

    let mut hist = vec![0.0f64; num_bins];
    for &code in &codes {
        let bin = match config.variant {
            LbpVariant::Basic => (code as usize) % num_bins,
            LbpVariant::Uniform => uniform_index(code, p).min(num_bins - 1),
        };
        hist[bin] += 1.0;
    }

    // Normalise
    let total: f64 = hist.iter().sum();
    if total > 0.0 {
        for v in &mut hist {
            *v /= total;
        }
    }

    hist
}

// ---------------------------------------------------------------------------
// High-level combined API
// ---------------------------------------------------------------------------

/// Full texture descriptor combining Haralick features and the LBP histogram.
pub struct TextureDescriptor {
    /// Haralick features (14 values).
    pub haralick: HaralickFeatures,
    /// Normalised LBP histogram.
    pub lbp_hist: Vec<f64>,
}

impl TextureDescriptor {
    /// Extract a combined texture descriptor from a grayscale float image.
    #[must_use]
    pub fn extract(
        pixels: &[f64],
        width: usize,
        height: usize,
        glcm_cfg: &GlcmConfig,
        lbp_cfg: &LbpConfig,
    ) -> Self {
        let glcm = compute_glcm(pixels, width, height, glcm_cfg);
        let haralick = HaralickFeatures::compute(&glcm);
        let lbp_hist = lbp_histogram(pixels, width, height, lbp_cfg);
        Self { haralick, lbp_hist }
    }

    /// Concatenate Haralick + LBP histogram into a single feature vector.
    #[must_use]
    pub fn feature_vector(&self) -> Vec<f64> {
        let mut v = self.haralick.to_vec();
        v.extend_from_slice(&self.lbp_hist);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard(w: usize, h: usize) -> Vec<f64> {
        (0..h)
            .flat_map(|y| (0..w).map(move |x| if (x + y) % 2 == 0 { 0.0 } else { 1.0 }))
            .collect()
    }

    fn uniform_image(w: usize, h: usize, v: f64) -> Vec<f64> {
        vec![v; w * h]
    }

    // ----- GLCM tests -----

    #[test]
    fn test_glcm_uniform_image_diagonal() {
        // A uniform image should produce a GLCM concentrated on the diagonal
        let pixels = uniform_image(8, 8, 0.5);
        let cfg = GlcmConfig::new(4).with_directions(vec![GlcmDirection::East]);
        let glcm = compute_glcm(&pixels, 8, 8, &cfg);

        // Find which diagonal bin has the concentration
        let diag_sum: f64 = (0..4).map(|i| glcm.get(i, i)).sum();
        let off_diag_sum: f64 = {
            let mut s = 0.0;
            for i in 0..4 {
                for j in 0..4 {
                    if i != j {
                        s += glcm.get(i, j);
                    }
                }
            }
            s
        };
        assert!(
            diag_sum > off_diag_sum,
            "uniform image GLCM should concentrate on diagonal; diag={diag_sum:.4} off={off_diag_sum:.4}"
        );
    }

    #[test]
    fn test_glcm_normalised_sums_to_one() {
        let pixels = checkerboard(6, 6);
        let cfg = GlcmConfig::default();
        let glcm = compute_glcm(&pixels, 6, 6, &cfg);
        let total: f64 = glcm.matrix.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "normalised GLCM should sum to 1; got {total}"
        );
    }

    #[test]
    fn test_glcm_symmetrize() {
        let pixels = checkerboard(5, 5);
        let cfg = GlcmConfig::new(4).with_directions(vec![GlcmDirection::East]);
        let glcm = compute_glcm(&pixels, 5, 5, &cfg);
        let levels = glcm.levels;
        for i in 0..levels {
            for j in 0..levels {
                let diff = (glcm.get(i, j) - glcm.get(j, i)).abs();
                assert!(
                    diff < 1e-10,
                    "symmetrized GLCM should be symmetric at ({i},{j})"
                );
            }
        }
    }

    // ----- Haralick tests -----

    #[test]
    fn test_haralick_energy_uniform() {
        // Uniform GLCM → energy = 1/levels² summed over levels² cells → energy = 1
        // (because each cell is 1/l^2, energy = sum (1/l^2)^2 * l^2 = 1/l^2)
        let pixels = uniform_image(8, 8, 0.5);
        let cfg = GlcmConfig::new(4).with_directions(vec![GlcmDirection::East]);
        let glcm = compute_glcm(&pixels, 8, 8, &cfg);
        let feats = HaralickFeatures::compute(&glcm);
        // Energy must be positive and <= 1
        assert!(feats.energy > 0.0 && feats.energy <= 1.0);
    }

    #[test]
    fn test_haralick_contrast_checkerboard_high() {
        // Checkerboard alternates between 0 and 1, so contrast should be higher than uniform
        let checker = checkerboard(8, 8);
        let unif = uniform_image(8, 8, 0.5);
        let cfg = GlcmConfig::new(8).with_directions(vec![GlcmDirection::East]);

        let g_checker = compute_glcm(&checker, 8, 8, &cfg);
        let g_unif = compute_glcm(&unif, 8, 8, &cfg);

        let f_checker = HaralickFeatures::compute(&g_checker);
        let f_unif = HaralickFeatures::compute(&g_unif);

        assert!(
            f_checker.contrast > f_unif.contrast,
            "checkerboard contrast ({:.4}) should exceed uniform ({:.4})",
            f_checker.contrast,
            f_unif.contrast
        );
    }

    #[test]
    fn test_haralick_to_vec_length() {
        let feats = HaralickFeatures::default();
        assert_eq!(feats.to_vec().len(), 14);
    }

    #[test]
    fn test_haralick_homogeneity_uniform_high() {
        // Uniform image → all pairs at same level → homogeneity high
        let pixels = uniform_image(8, 8, 0.3);
        let cfg = GlcmConfig::new(4).with_directions(vec![GlcmDirection::East]);
        let glcm = compute_glcm(&pixels, 8, 8, &cfg);
        let feats = HaralickFeatures::compute(&glcm);
        // Homogeneity max is 1 (all same level, diff=0, 1/(1+0)=1)
        assert!(
            feats.homogeneity > 0.5,
            "uniform image should have high homogeneity"
        );
    }

    // ----- LBP tests -----

    #[test]
    fn test_lbp_histogram_sums_to_one() {
        let pixels = checkerboard(10, 10);
        let cfg = LbpConfig::default();
        let hist = lbp_histogram(&pixels, 10, 10, &cfg);
        let total: f64 = hist.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "LBP histogram should sum to 1; got {total}"
        );
    }

    #[test]
    fn test_lbp_basic_bin_count() {
        let pixels = uniform_image(5, 5, 0.5);
        let cfg = LbpConfig::new(8, 1.0).with_variant(LbpVariant::Basic);
        let hist = lbp_histogram(&pixels, 5, 5, &cfg);
        assert_eq!(hist.len(), 256, "basic LBP with P=8 should have 256 bins");
    }

    #[test]
    fn test_lbp_uniform_bin_count() {
        let pixels = uniform_image(5, 5, 0.5);
        let cfg = LbpConfig::new(8, 1.0).with_variant(LbpVariant::Uniform);
        let hist = lbp_histogram(&pixels, 5, 5, &cfg);
        // P+2 = 10 bins for P=8
        assert_eq!(hist.len(), 10, "uniform LBP with P=8 should have 10 bins");
    }

    #[test]
    fn test_lbp_uniform_image_all_ones_or_zeros() {
        // Uniform image: all neighbours ≥ centre, so code is all-1s or all-0s
        // For all-0s centre, neighbours are equal → code 0xFF (all set) or 0x00
        let pixels = uniform_image(6, 6, 0.5);
        let cfg = LbpConfig::new(8, 1.0).with_variant(LbpVariant::Uniform);
        let codes = compute_lbp(&pixels, 6, 6, &cfg);
        // Either all bits set (code = 0xFF = 255) or 0, both are uniform
        for &code in &codes {
            assert!(
                count_transitions(code, 8) <= 2,
                "uniform image should yield uniform LBP codes; got code={code:#010b}"
            );
        }
    }

    // ----- Combined descriptor test -----

    #[test]
    fn test_texture_descriptor_feature_vector_length() {
        let pixels = checkerboard(8, 8);
        let glcm_cfg = GlcmConfig::new(4).with_directions(vec![GlcmDirection::East]);
        let lbp_cfg = LbpConfig::new(8, 1.0).with_variant(LbpVariant::Uniform);
        let desc = TextureDescriptor::extract(&pixels, 8, 8, &glcm_cfg, &lbp_cfg);
        let fv = desc.feature_vector();
        // 14 Haralick + (8+2) LBP uniform = 24
        assert_eq!(
            fv.len(),
            24,
            "feature vector length should be 24; got {}",
            fv.len()
        );
    }
}
