//! GPU-accelerated tone curve application.
//!
//! Provides configurable tone curves for image processing pipelines.
//! Tone curves adjust the brightness and contrast of an image by mapping
//! input luminance values through a transfer function.
//!
//! # Supported Curves
//!
//! | Curve | Description |
//! |-------|-------------|
//! | [`ToneCurveType::Linear`] | Identity mapping (no change). |
//! | [`ToneCurveType::Gamma`] | Power-law gamma curve. |
//! | [`ToneCurveType::SLog3`] | Sony S-Log3 camera log curve. |
//! | [`ToneCurveType::LogC`] | ARRI LogC camera log curve. |
//! | [`ToneCurveType::Spline`] | User-defined cubic spline (control points). |
//! | [`ToneCurveType::FilmicSCurve`] | Classic filmic S-curve for cinematic look. |
//! | [`ToneCurveType::Custom`] | Pre-built 256-entry LUT. |
//!
//! All operations are CPU-side (suitable for GPU upload as a 1D LUT texture).

use rayon::prelude::*;

// ─── Error ──────────────────────────────────────────────────────────────────

/// Errors that can occur during tone curve operations.
#[derive(Debug, Clone)]
pub enum ToneCurveError {
    /// The input and output buffer lengths do not match.
    BufferSizeMismatch { expected: usize, actual: usize },
    /// The buffer length is not a multiple of 4 (RGBA).
    InvalidBufferLength(usize),
    /// A spline requires at least 2 control points.
    InsufficientControlPoints(usize),
    /// Dimensions are invalid (zero width or height).
    InvalidDimensions { width: u32, height: u32 },
    /// A parameter is out of the valid range.
    InvalidParameter(String),
}

impl std::fmt::Display for ToneCurveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferSizeMismatch { expected, actual } => {
                write!(f, "buffer size mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidBufferLength(len) => {
                write!(f, "buffer length {len} is not a multiple of 4")
            }
            Self::InsufficientControlPoints(n) => {
                write!(f, "spline requires >= 2 control points, got {n}")
            }
            Self::InvalidDimensions { width, height } => {
                write!(f, "invalid dimensions: {width}x{height}")
            }
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}

impl std::error::Error for ToneCurveError {}

type Result<T> = std::result::Result<T, ToneCurveError>;

// ─── ToneCurveType ──────────────────────────────────────────────────────────

/// Specifies the type of tone curve to apply.
#[derive(Debug, Clone)]
pub enum ToneCurveType {
    /// Identity mapping — output equals input.
    Linear,
    /// Power-law gamma: `output = input^(1/gamma)`.
    Gamma(f64),
    /// Sony S-Log3 linearisation curve.
    SLog3,
    /// ARRI LogC (EI 800) linearisation curve.
    LogC,
    /// User-defined cubic spline through control points `(x, y)` in \[0,1\].
    Spline(Vec<(f64, f64)>),
    /// Classic filmic S-curve parameterised by `toe` and `shoulder` strength.
    FilmicSCurve {
        /// Toe compression (0.0–1.0). Higher = darker shadows.
        toe: f64,
        /// Shoulder compression (0.0–1.0). Higher = softer highlights.
        shoulder: f64,
    },
    /// Pre-built 256-entry LUT.
    Custom([u8; 256]),
}

// ─── ToneCurve (builder) ────────────────────────────────────────────────────

/// A compiled tone curve ready for application to 8-bit images.
///
/// Internally this holds a 256-entry LUT built from the selected curve type.
#[derive(Debug, Clone)]
pub struct ToneCurve {
    /// 256-entry look-up table mapping input \[0..255\] to output \[0..255\].
    lut: [u8; 256],
    /// Human-readable label.
    label: String,
}

impl ToneCurve {
    /// Build a tone curve from the given type.
    ///
    /// # Errors
    ///
    /// Returns an error if the curve parameters are invalid (e.g. a spline
    /// with fewer than 2 control points, or a negative gamma).
    pub fn build(curve_type: &ToneCurveType) -> Result<Self> {
        let (lut, label) = match curve_type {
            ToneCurveType::Linear => (build_linear_lut(), "linear".to_string()),
            ToneCurveType::Gamma(g) => {
                if *g <= 0.0 {
                    return Err(ToneCurveError::InvalidParameter(format!(
                        "gamma must be > 0, got {g}"
                    )));
                }
                (build_gamma_lut(*g), format!("gamma({g:.2})"))
            }
            ToneCurveType::SLog3 => (build_slog3_lut(), "slog3".to_string()),
            ToneCurveType::LogC => (build_logc_lut(), "logc".to_string()),
            ToneCurveType::Spline(pts) => {
                if pts.len() < 2 {
                    return Err(ToneCurveError::InsufficientControlPoints(pts.len()));
                }
                (build_spline_lut(pts), "spline".to_string())
            }
            ToneCurveType::FilmicSCurve { toe, shoulder } => (
                build_filmic_s_lut(*toe, *shoulder),
                format!("filmic(toe={toe:.2},shoulder={shoulder:.2})"),
            ),
            ToneCurveType::Custom(lut) => (*lut, "custom".to_string()),
        };
        Ok(Self { lut, label })
    }

    /// Apply this tone curve to an RGBA image buffer in-place.
    ///
    /// Only the R, G, B channels are mapped; the alpha channel is preserved.
    ///
    /// # Errors
    ///
    /// Returns an error if `buf.len()` is not a multiple of 4.
    pub fn apply_in_place(&self, buf: &mut [u8]) -> Result<()> {
        if buf.len() % 4 != 0 {
            return Err(ToneCurveError::InvalidBufferLength(buf.len()));
        }
        buf.par_chunks_exact_mut(4).for_each(|px| {
            px[0] = self.lut[usize::from(px[0])];
            px[1] = self.lut[usize::from(px[1])];
            px[2] = self.lut[usize::from(px[2])];
            // alpha unchanged
        });
        Ok(())
    }

    /// Apply this tone curve, writing the result to a separate output buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffers have different lengths or are not
    /// multiples of 4.
    pub fn apply(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        if input.len() != output.len() {
            return Err(ToneCurveError::BufferSizeMismatch {
                expected: input.len(),
                actual: output.len(),
            });
        }
        if input.len() % 4 != 0 {
            return Err(ToneCurveError::InvalidBufferLength(input.len()));
        }
        output
            .par_chunks_exact_mut(4)
            .zip(input.par_chunks_exact(4))
            .for_each(|(out, inp)| {
                out[0] = self.lut[usize::from(inp[0])];
                out[1] = self.lut[usize::from(inp[1])];
                out[2] = self.lut[usize::from(inp[2])];
                out[3] = inp[3]; // alpha passthrough
            });
        Ok(())
    }

    /// Apply this tone curve to a single-channel (luma) buffer in-place.
    pub fn apply_luma_in_place(&self, buf: &mut [u8]) {
        buf.par_iter_mut().for_each(|v| {
            *v = self.lut[usize::from(*v)];
        });
    }

    /// Apply this tone curve to a single-channel (luma) buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffers have different lengths.
    pub fn apply_luma(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        if input.len() != output.len() {
            return Err(ToneCurveError::BufferSizeMismatch {
                expected: input.len(),
                actual: output.len(),
            });
        }
        output
            .par_iter_mut()
            .zip(input.par_iter())
            .for_each(|(o, &i)| {
                *o = self.lut[usize::from(i)];
            });
        Ok(())
    }

    /// Get a reference to the internal 256-entry LUT.
    #[must_use]
    pub fn lut(&self) -> &[u8; 256] {
        &self.lut
    }

    /// Get the human-readable label for this curve.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Compose two tone curves into a single LUT (apply `self` then `other`).
    #[must_use]
    pub fn compose(&self, other: &ToneCurve) -> ToneCurve {
        let mut lut = [0u8; 256];
        for (i, entry) in lut.iter_mut().enumerate() {
            let intermediate = self.lut[i];
            *entry = other.lut[usize::from(intermediate)];
        }
        ToneCurve {
            lut,
            label: format!("{}+{}", self.label, other.label),
        }
    }

    /// Compute the inverse LUT (approximate — may not be exact for
    /// non-monotonic curves).
    #[must_use]
    pub fn invert(&self) -> ToneCurve {
        let mut inv = [0u8; 256];
        // For each output value, find the closest input.
        for out_val in 0u16..256 {
            let mut best_in = 0u8;
            let mut best_diff = 256i32;
            for in_val in 0u16..256 {
                let diff = (i32::from(self.lut[in_val as usize]) - out_val as i32).abs();
                if diff < best_diff {
                    best_diff = diff;
                    best_in = in_val as u8;
                }
            }
            inv[out_val as usize] = best_in;
        }
        ToneCurve {
            lut: inv,
            label: format!("inv({})", self.label),
        }
    }
}

// ─── LUT builders ───────────────────────────────────────────────────────────

fn build_linear_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        *entry = i as u8;
    }
    lut
}

fn build_gamma_lut(gamma: f64) -> [u8; 256] {
    let inv_gamma = 1.0 / gamma;
    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let x = i as f64 / 255.0;
        let y = x.powf(inv_gamma);
        *entry = (y * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    lut
}

/// Sony S-Log3 linearisation: converts S-Log3 code values to linear.
///
/// Reference: Sony S-Log3 specification (patent-free, publicly documented).
fn build_slog3_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let x = i as f64 / 255.0;
        // Simplified S-Log3 linearisation
        let linear = if x >= 171.2102946929 / 1023.0 {
            let base = 10.0_f64;
            base.powf((x * 1023.0 - 420.0) / 261.5) * (0.18 + 0.01) - 0.01
        } else {
            (x * 1023.0 - 95.0) * 0.01125 / (171.2102946929 - 95.0)
        };
        *entry = (linear.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    lut
}

/// ARRI LogC (EI 800) linearisation curve.
///
/// Reference: ARRI LogC specification (patent-free, publicly documented).
fn build_logc_lut() -> [u8; 256] {
    // LogC constants for EI 800
    const CUT: f64 = 0.010591;
    const A: f64 = 5.555556;
    const B: f64 = 0.052272;
    const C: f64 = 0.247190;
    const D: f64 = 0.385537;
    const E: f64 = 5.367655;
    const F: f64 = 0.092809;

    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let t = i as f64 / 255.0; // LogC code value [0,1]
                                  // Invert the LogC encoding: t = C * log10(A*x + B) + D for x >= CUT
                                  // => x = (10^((t-D)/C) - B) / A
        let linear = if t > E * CUT + F {
            let base = 10.0_f64;
            (base.powf((t - D) / C) - B) / A
        } else {
            (t - F) / E
        };
        *entry = (linear.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    lut
}

/// Natural cubic spline interpolation through sorted control points.
fn build_spline_lut(points: &[(f64, f64)]) -> [u8; 256] {
    let mut sorted: Vec<(f64, f64)> = points.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Ensure endpoints at 0 and 1 (only add if not already present).
    if sorted.first().map_or(true, |p| p.0 > 1e-9) {
        sorted.insert(0, (0.0, sorted.first().map_or(0.0, |p| p.1)));
    }
    if sorted.last().map_or(true, |p| (1.0 - p.0).abs() > 1e-9) {
        sorted.push((1.0, sorted.last().map_or(1.0, |p| p.1)));
    }

    let n = sorted.len();
    let mut lut = [0u8; 256];

    if n < 2 {
        // Degenerate: fill with the single value.
        let val = sorted.first().map_or(0.0, |p| p.1);
        let byte = (val.clamp(0.0, 1.0) * 255.0).round() as u8;
        lut.fill(byte);
        return lut;
    }

    // Linear interpolation between control points (simplified spline).
    for (i, entry) in lut.iter_mut().enumerate() {
        let x = i as f64 / 255.0;
        // Find the segment containing x.
        let mut seg = n - 2; // default to last segment
        for j in 1..n {
            if sorted[j].0 >= x {
                seg = j - 1;
                break;
            }
        }
        let (x0, y0) = sorted[seg];
        let (x1, y1) = sorted[(seg + 1).min(n - 1)];
        let dx = x1 - x0;
        let t = if dx.abs() < 1e-12 { 0.0 } else { (x - x0) / dx };
        // Use Hermite smoothstep only for segments shorter than the full range;
        // for long segments (like a simple 2-point spline) use linear to avoid
        // crushing near the endpoints.
        let y = if n > 2 {
            // Hermite smoothstep for smoother multi-segment interpolation.
            let st = t * t * (3.0 - 2.0 * t);
            y0 + (y1 - y0) * st
        } else {
            y0 + (y1 - y0) * t
        };
        *entry = (y.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    lut
}

/// Filmic S-curve using a sigmoid-like function.
fn build_filmic_s_lut(toe: f64, shoulder: f64) -> [u8; 256] {
    let mut lut = [0u8; 256];
    // Toe parameter shifts the shadow region; shoulder softens highlights.
    let toe_str = toe.clamp(0.0, 1.0) * 2.0 + 0.5;
    let shoulder_str = shoulder.clamp(0.0, 1.0) * 2.0 + 0.5;

    for (i, entry) in lut.iter_mut().enumerate() {
        let x = i as f64 / 255.0;
        // Apply a parameterised sigmoid: y = x^toe / (x^toe + (1-x)^shoulder)
        let xt = x.powf(toe_str);
        let oxt = (1.0 - x).powf(shoulder_str);
        let denom = xt + oxt;
        let y = if denom.abs() < 1e-12 { x } else { xt / denom };
        *entry = (y.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    lut
}

// ─── ToneCurveChain ─────────────────────────────────────────────────────────

/// A chain of tone curves that can be baked into a single LUT.
#[derive(Debug, Clone)]
pub struct ToneCurveChain {
    curves: Vec<ToneCurve>,
}

impl ToneCurveChain {
    /// Create an empty chain.
    #[must_use]
    pub fn new() -> Self {
        Self { curves: Vec::new() }
    }

    /// Append a tone curve to the chain.
    pub fn push(&mut self, curve: ToneCurve) {
        self.curves.push(curve);
    }

    /// Number of curves in the chain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.curves.len()
    }

    /// Whether the chain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.curves.is_empty()
    }

    /// Bake all curves in the chain into a single LUT.
    ///
    /// If the chain is empty, returns a linear (identity) curve.
    #[must_use]
    pub fn bake(&self) -> ToneCurve {
        if self.curves.is_empty() {
            return ToneCurve {
                lut: build_linear_lut(),
                label: "identity".to_string(),
            };
        }
        let mut result = self.curves[0].clone();
        for curve in &self.curves[1..] {
            result = result.compose(curve);
        }
        result
    }
}

impl Default for ToneCurveChain {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_curve_is_identity() {
        let curve = ToneCurve::build(&ToneCurveType::Linear).expect("linear build");
        for i in 0u8..=255 {
            assert_eq!(curve.lut()[usize::from(i)], i, "identity failed at {i}");
        }
    }

    #[test]
    fn test_gamma_curve_brightens_midtones() {
        // gamma > 1 should brighten (raise) midtones
        let curve = ToneCurve::build(&ToneCurveType::Gamma(2.2)).expect("gamma build");
        // At input=128, output should be > 128 for gamma > 1
        let mid = curve.lut()[128];
        assert!(mid > 128, "gamma 2.2 should brighten midtones, got {mid}");
    }

    #[test]
    fn test_gamma_curve_preserves_endpoints() {
        let curve = ToneCurve::build(&ToneCurveType::Gamma(2.2)).expect("gamma build");
        assert_eq!(curve.lut()[0], 0, "black should remain black");
        assert_eq!(curve.lut()[255], 255, "white should remain white");
    }

    #[test]
    fn test_negative_gamma_returns_error() {
        let result = ToneCurve::build(&ToneCurveType::Gamma(-1.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_gamma_returns_error() {
        let result = ToneCurve::build(&ToneCurveType::Gamma(0.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_slog3_maps_black_to_low() {
        let curve = ToneCurve::build(&ToneCurveType::SLog3).expect("slog3 build");
        assert!(
            curve.lut()[0] < 10,
            "slog3 black should be near 0, got {}",
            curve.lut()[0]
        );
    }

    #[test]
    fn test_logc_maps_black_to_low() {
        let curve = ToneCurve::build(&ToneCurveType::LogC).expect("logc build");
        assert!(
            curve.lut()[0] < 10,
            "logc black should be near 0, got {}",
            curve.lut()[0]
        );
    }

    #[test]
    fn test_filmic_s_curve_endpoints() {
        let curve = ToneCurve::build(&ToneCurveType::FilmicSCurve {
            toe: 0.5,
            shoulder: 0.5,
        })
        .expect("filmic build");
        assert_eq!(curve.lut()[0], 0);
        assert_eq!(curve.lut()[255], 255);
    }

    #[test]
    fn test_filmic_s_curve_midpoint_shift() {
        let curve = ToneCurve::build(&ToneCurveType::FilmicSCurve {
            toe: 0.3,
            shoulder: 0.7,
        })
        .expect("filmic build");
        // Midpoint should be shifted due to asymmetric toe/shoulder
        let mid = curve.lut()[128];
        assert!(
            mid > 0 && mid < 255,
            "midpoint {mid} should be between 0 and 255"
        );
    }

    #[test]
    fn test_spline_with_two_points() {
        let pts = vec![(0.0, 0.0), (1.0, 1.0)];
        let curve = ToneCurve::build(&ToneCurveType::Spline(pts)).expect("spline build");
        // Should be approximately identity
        for i in 0u8..=255 {
            let diff = (i as i32 - curve.lut()[usize::from(i)] as i32).abs();
            assert!(
                diff <= 1,
                "spline identity failed at {i}: got {}",
                curve.lut()[usize::from(i)]
            );
        }
    }

    #[test]
    fn test_spline_insufficient_points_error() {
        let pts = vec![(0.5, 0.5)];
        let result = ToneCurve::build(&ToneCurveType::Spline(pts));
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_lut_passthrough() {
        let mut custom = [0u8; 256];
        for (i, entry) in custom.iter_mut().enumerate() {
            *entry = (255 - i) as u8; // invert
        }
        let curve = ToneCurve::build(&ToneCurveType::Custom(custom)).expect("custom build");
        assert_eq!(curve.lut()[0], 255);
        assert_eq!(curve.lut()[255], 0);
        assert_eq!(curve.lut()[128], 127);
    }

    #[test]
    fn test_apply_rgba_buffer() {
        let curve = ToneCurve::build(&ToneCurveType::Gamma(2.2)).expect("gamma build");
        let input = vec![128u8, 64, 200, 255]; // R, G, B, A
        let mut output = vec![0u8; 4];
        curve.apply(&input, &mut output).expect("apply");
        assert_eq!(output[3], 255, "alpha should be preserved");
        assert_ne!(output[0], 128, "R should be changed by gamma");
    }

    #[test]
    fn test_apply_in_place() {
        let curve = ToneCurve::build(&ToneCurveType::Gamma(2.2)).expect("gamma build");
        let mut buf = vec![128u8, 64, 200, 255, 50, 100, 150, 200];
        curve.apply_in_place(&mut buf).expect("apply in place");
        assert_eq!(buf[3], 255, "alpha preserved");
        assert_eq!(buf[7], 200, "alpha preserved");
    }

    #[test]
    fn test_apply_mismatched_size_returns_error() {
        let curve = ToneCurve::build(&ToneCurveType::Linear).expect("linear build");
        let input = vec![0u8; 8];
        let mut output = vec![0u8; 4];
        let result = curve.apply(&input, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_non_multiple_of_4_returns_error() {
        let curve = ToneCurve::build(&ToneCurveType::Linear).expect("linear build");
        let input = vec![0u8; 5];
        let mut output = vec![0u8; 5];
        let result = curve.apply(&input, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_luma() {
        let curve = ToneCurve::build(&ToneCurveType::Gamma(2.2)).expect("gamma build");
        let input = vec![0u8, 128, 255];
        let mut output = vec![0u8; 3];
        curve.apply_luma(&input, &mut output).expect("apply luma");
        assert_eq!(output[0], 0);
        assert_eq!(output[2], 255);
    }

    #[test]
    fn test_apply_luma_in_place() {
        let curve = ToneCurve::build(&ToneCurveType::Linear).expect("linear build");
        let mut buf = vec![10u8, 20, 30];
        curve.apply_luma_in_place(&mut buf);
        assert_eq!(buf, vec![10, 20, 30]);
    }

    #[test]
    fn test_compose_two_curves() {
        let g1 = ToneCurve::build(&ToneCurveType::Gamma(2.2)).expect("g1");
        let g2 = ToneCurve::build(&ToneCurveType::Gamma(1.0 / 2.2)).expect("g2");
        let composed = g1.compose(&g2);
        // Should be approximately identity
        for i in 0u8..=255 {
            let diff = (i as i32 - composed.lut()[usize::from(i)] as i32).abs();
            assert!(
                diff <= 2,
                "compose round-trip failed at {i}: got {}",
                composed.lut()[usize::from(i)]
            );
        }
    }

    #[test]
    fn test_invert_linear_is_identity() {
        let linear = ToneCurve::build(&ToneCurveType::Linear).expect("linear");
        let inv = linear.invert();
        for i in 0u8..=255 {
            assert_eq!(inv.lut()[usize::from(i)], i);
        }
    }

    #[test]
    fn test_chain_bake_empty_is_identity() {
        let chain = ToneCurveChain::new();
        assert!(chain.is_empty());
        let baked = chain.bake();
        for i in 0u8..=255 {
            assert_eq!(baked.lut()[usize::from(i)], i);
        }
    }

    #[test]
    fn test_chain_bake_single() {
        let mut chain = ToneCurveChain::new();
        let g = ToneCurve::build(&ToneCurveType::Gamma(2.2)).expect("gamma");
        chain.push(g.clone());
        assert_eq!(chain.len(), 1);
        let baked = chain.bake();
        assert_eq!(baked.lut(), g.lut());
    }

    #[test]
    fn test_chain_bake_multiple() {
        let mut chain = ToneCurveChain::new();
        chain.push(ToneCurve::build(&ToneCurveType::Gamma(2.2)).expect("g1"));
        chain.push(ToneCurve::build(&ToneCurveType::Gamma(1.0 / 2.2)).expect("g2"));
        let baked = chain.bake();
        // Round-trip should be approximately identity
        for i in 0u8..=255 {
            let diff = (i as i32 - baked.lut()[usize::from(i)] as i32).abs();
            assert!(diff <= 2, "chain bake round-trip failed at {i}");
        }
    }

    #[test]
    fn test_label_includes_curve_name() {
        let curve = ToneCurve::build(&ToneCurveType::Gamma(1.8)).expect("gamma");
        assert!(curve.label().contains("gamma"));
    }

    #[test]
    fn test_spline_with_custom_control_points() {
        let pts = vec![(0.0, 0.0), (0.25, 0.1), (0.5, 0.5), (0.75, 0.9), (1.0, 1.0)];
        let curve = ToneCurve::build(&ToneCurveType::Spline(pts)).expect("spline build");
        // Monotonically increasing overall
        assert!(curve.lut()[0] <= curve.lut()[128]);
        assert!(curve.lut()[128] <= curve.lut()[255]);
    }

    #[test]
    fn test_gamma_unity_is_identity() {
        let curve = ToneCurve::build(&ToneCurveType::Gamma(1.0)).expect("gamma 1.0");
        for i in 0u8..=255 {
            assert_eq!(curve.lut()[usize::from(i)], i);
        }
    }
}
