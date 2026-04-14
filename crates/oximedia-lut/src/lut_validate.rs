//! LUT validation and integrity checking.
//!
//! Provides tools to validate 1D and 3D LUT data for correctness,
//! range compliance, monotonicity, smoothness, and structural integrity before
//! applying them in production color pipelines.
//!
//! ## Monotonicity Checks (1D LUTs)
//!
//! A 1D LUT used as a transfer function should be **monotonic** — each output
//! must be >= the previous output (increasing) or <= (decreasing).
//! [`check_1d_monotonicity`] and [`check_1d_strictly_increasing`] perform full
//! per-channel analysis and return detailed [`MonotonicityReport`]s.
//!
//! ## Smoothness Checks (3D LUTs)
//!
//! A well-behaved 3D LUT should vary smoothly across the lattice.  Large
//! first-differences between adjacent lattice neighbours indicate discontinuities.
//! [`check_3d_smoothness`] computes the distribution of inter-neighbour gradients
//! and returns a [`SmoothnessReport`] with mean, maximum, p95, and a
//! user-configurable roughness threshold.

use std::fmt;

/// Result of a single validation check.
#[derive(Clone, Debug, PartialEq)]
pub enum ValidationSeverity {
    /// Informational note that does not affect correctness.
    Info,
    /// Warning about potential quality issues.
    Warning,
    /// Error that renders the LUT unusable.
    Error,
}

impl fmt::Display for ValidationSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

/// A single validation diagnostic message.
#[derive(Clone, Debug)]
pub struct ValidationDiagnostic {
    /// Severity of this diagnostic.
    pub severity: ValidationSeverity,
    /// Human-readable message describing the issue.
    pub message: String,
    /// Optional index or coordinate where the issue was found.
    pub location: Option<String>,
}

impl ValidationDiagnostic {
    /// Create a new diagnostic.
    pub fn new(severity: ValidationSeverity, message: impl Into<String>) -> Self {
        Self {
            severity,
            message: message.into(),
            location: None,
        }
    }

    /// Create a diagnostic with a location hint.
    pub fn with_location(
        severity: ValidationSeverity,
        message: impl Into<String>,
        location: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            message: message.into(),
            location: Some(location.into()),
        }
    }

    /// Returns true if this diagnostic is an error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.severity == ValidationSeverity::Error
    }

    /// Returns true if this diagnostic is a warning.
    #[must_use]
    pub fn is_warning(&self) -> bool {
        self.severity == ValidationSeverity::Warning
    }
}

/// Result of a full LUT validation pass.
#[derive(Clone, Debug)]
pub struct ValidationReport {
    /// All diagnostics collected during validation.
    pub diagnostics: Vec<ValidationDiagnostic>,
    /// Whether the LUT passes all critical checks.
    pub passed: bool,
    /// Name or identifier of the LUT that was validated.
    pub lut_name: String,
}

impl ValidationReport {
    /// Create a new empty validation report.
    pub fn new(lut_name: impl Into<String>) -> Self {
        Self {
            diagnostics: Vec::new(),
            passed: true,
            lut_name: lut_name.into(),
        }
    }

    /// Add a diagnostic to the report.
    pub fn add(&mut self, diagnostic: ValidationDiagnostic) {
        if diagnostic.is_error() {
            self.passed = false;
        }
        self.diagnostics.push(diagnostic);
    }

    /// Count diagnostics of a given severity.
    #[must_use]
    pub fn count_severity(&self, severity: &ValidationSeverity) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| &d.severity == severity)
            .count()
    }

    /// Return all error diagnostics.
    #[must_use]
    pub fn errors(&self) -> Vec<&ValidationDiagnostic> {
        self.diagnostics.iter().filter(|d| d.is_error()).collect()
    }

    /// Return all warning diagnostics.
    #[must_use]
    pub fn warnings(&self) -> Vec<&ValidationDiagnostic> {
        self.diagnostics.iter().filter(|d| d.is_warning()).collect()
    }

    /// Total number of diagnostics.
    #[must_use]
    pub fn total_diagnostics(&self) -> usize {
        self.diagnostics.len()
    }
}

/// Configuration for the LUT validator.
#[derive(Clone, Debug)]
pub struct ValidatorConfig {
    /// Minimum allowed value in LUT entries (default: 0.0).
    pub min_value: f64,
    /// Maximum allowed value in LUT entries (default: 1.0).
    pub max_value: f64,
    /// Tolerance for out-of-range checks.
    pub range_tolerance: f64,
    /// Whether to check for monotonicity in 1D LUTs.
    pub check_monotonicity: bool,
    /// Whether to check for NaN or Infinity values.
    pub check_nan_inf: bool,
    /// Maximum allowed deviation from identity LUT (for identity checks).
    pub identity_tolerance: f64,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            min_value: 0.0,
            max_value: 1.0,
            range_tolerance: 0.001,
            check_monotonicity: true,
            check_nan_inf: true,
            identity_tolerance: 0.0001,
        }
    }
}

/// Validates LUT data for correctness and quality.
#[derive(Clone, Debug)]
pub struct LutValidator {
    /// Configuration for validation rules.
    pub config: ValidatorConfig,
}

impl LutValidator {
    /// Create a new validator with default config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ValidatorConfig::default(),
        }
    }

    /// Create a new validator with a custom config.
    #[must_use]
    pub fn with_config(config: ValidatorConfig) -> Self {
        Self { config }
    }

    /// Validate a 1D LUT channel (single channel array).
    #[must_use]
    pub fn validate_1d_channel(&self, data: &[f64], channel_name: &str) -> ValidationReport {
        let mut report = ValidationReport::new(format!("1D LUT channel: {channel_name}"));

        if data.is_empty() {
            report.add(ValidationDiagnostic::new(
                ValidationSeverity::Error,
                "LUT channel is empty",
            ));
            return report;
        }

        // Check for NaN / Infinity
        if self.config.check_nan_inf {
            for (i, &v) in data.iter().enumerate() {
                if v.is_nan() {
                    report.add(ValidationDiagnostic::with_location(
                        ValidationSeverity::Error,
                        "NaN value found",
                        format!("index {i}"),
                    ));
                } else if v.is_infinite() {
                    report.add(ValidationDiagnostic::with_location(
                        ValidationSeverity::Error,
                        "Infinite value found",
                        format!("index {i}"),
                    ));
                }
            }
        }

        // Range check
        for (i, &v) in data.iter().enumerate() {
            if v < self.config.min_value - self.config.range_tolerance {
                report.add(ValidationDiagnostic::with_location(
                    ValidationSeverity::Warning,
                    format!("Value {v:.6} below minimum {:.6}", self.config.min_value),
                    format!("index {i}"),
                ));
            }
            if v > self.config.max_value + self.config.range_tolerance {
                report.add(ValidationDiagnostic::with_location(
                    ValidationSeverity::Warning,
                    format!("Value {v:.6} above maximum {:.6}", self.config.max_value),
                    format!("index {i}"),
                ));
            }
        }

        // Monotonicity check
        if self.config.check_monotonicity && data.len() > 1 {
            let mut is_increasing = true;
            let mut is_decreasing = true;
            for w in data.windows(2) {
                if w[1] < w[0] {
                    is_increasing = false;
                }
                if w[1] > w[0] {
                    is_decreasing = false;
                }
            }
            if !is_increasing && !is_decreasing {
                report.add(ValidationDiagnostic::new(
                    ValidationSeverity::Info,
                    "Channel is neither monotonically increasing nor decreasing",
                ));
            }
        }

        report
    }

    /// Validate 3D LUT data given as a flat array of RGB triplets.
    ///
    /// `size` is the number of entries per dimension (e.g. 17 or 33).
    #[must_use]
    pub fn validate_3d_data(&self, data: &[f64], size: usize) -> ValidationReport {
        let mut report = ValidationReport::new(format!("3D LUT ({size}x{size}x{size})"));
        let expected_len = size * size * size * 3;

        if data.len() != expected_len {
            report.add(ValidationDiagnostic::new(
                ValidationSeverity::Error,
                format!(
                    "Expected {expected_len} values for {size}^3 * 3, got {}",
                    data.len()
                ),
            ));
            return report;
        }

        let mut nan_count = 0_usize;
        let mut inf_count = 0_usize;
        let mut below_count = 0_usize;
        let mut above_count = 0_usize;

        for &v in data {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            } else {
                if v < self.config.min_value - self.config.range_tolerance {
                    below_count += 1;
                }
                if v > self.config.max_value + self.config.range_tolerance {
                    above_count += 1;
                }
            }
        }

        if nan_count > 0 {
            report.add(ValidationDiagnostic::new(
                ValidationSeverity::Error,
                format!("{nan_count} NaN values found in 3D LUT data"),
            ));
        }
        if inf_count > 0 {
            report.add(ValidationDiagnostic::new(
                ValidationSeverity::Error,
                format!("{inf_count} Infinite values found in 3D LUT data"),
            ));
        }
        if below_count > 0 {
            report.add(ValidationDiagnostic::new(
                ValidationSeverity::Warning,
                format!("{below_count} values below minimum range"),
            ));
        }
        if above_count > 0 {
            report.add(ValidationDiagnostic::new(
                ValidationSeverity::Warning,
                format!("{above_count} values above maximum range"),
            ));
        }

        report
    }

    /// Check if 3D LUT data is close to an identity transform.
    ///
    /// `size` is the number of entries per dimension.
    #[must_use]
    pub fn check_identity_3d(&self, data: &[f64], size: usize) -> bool {
        let expected_len = size * size * size * 3;
        if data.len() != expected_len {
            return false;
        }

        let step = if size > 1 {
            1.0 / (size as f64 - 1.0)
        } else {
            0.0
        };

        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    let idx = (b * size * size + g * size + r) * 3;
                    let expected_r = r as f64 * step;
                    let expected_g = g as f64 * step;
                    let expected_b = b as f64 * step;

                    if (data[idx] - expected_r).abs() > self.config.identity_tolerance
                        || (data[idx + 1] - expected_g).abs() > self.config.identity_tolerance
                        || (data[idx + 2] - expected_b).abs() > self.config.identity_tolerance
                    {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Compute the maximum deviation from identity for 3D LUT data.
    ///
    /// `size` is the number of entries per dimension.
    #[must_use]
    pub fn max_identity_deviation_3d(&self, data: &[f64], size: usize) -> Option<f64> {
        let expected_len = size * size * size * 3;
        if data.len() != expected_len || size == 0 {
            return None;
        }

        let step = if size > 1 {
            1.0 / (size as f64 - 1.0)
        } else {
            0.0
        };

        let mut max_dev = 0.0_f64;

        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    let idx = (b * size * size + g * size + r) * 3;
                    let expected_r = r as f64 * step;
                    let expected_g = g as f64 * step;
                    let expected_b = b as f64 * step;

                    let dev_r = (data[idx] - expected_r).abs();
                    let dev_g = (data[idx + 1] - expected_g).abs();
                    let dev_b = (data[idx + 2] - expected_b).abs();

                    max_dev = max_dev.max(dev_r).max(dev_g).max(dev_b);
                }
            }
        }
        Some(max_dev)
    }
}

impl Default for LutValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a LUT size is a power-of-two-plus-one (common for 3D LUTs: 17, 33, 65).
#[must_use]
pub fn is_standard_lut_size(size: usize) -> bool {
    matches!(size, 17 | 33 | 65 | 129)
}

// ============================================================================
// Monotonicity checks for 1D LUTs
// ============================================================================

/// Direction of monotonicity detected in a single 1D LUT channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MonotonicDirection {
    /// All consecutive differences are non-negative.
    Increasing,
    /// All consecutive differences are non-positive.
    Decreasing,
    /// At least one pair violates the dominant direction.
    NonMonotonic,
}

/// A single monotonicity violation in a 1D LUT channel.
#[derive(Clone, Debug)]
pub struct MonotonicityViolation {
    /// Index of the first element in the violating pair.
    pub index: usize,
    /// Value at `index`.
    pub value_a: f64,
    /// Value at `index + 1`.
    pub value_b: f64,
}

/// Per-channel monotonicity result.
#[derive(Clone, Debug)]
pub struct ChannelMonotonicityResult {
    /// Channel name ("R", "G", or "B").
    pub channel: String,
    /// Detected direction or NonMonotonic.
    pub direction: MonotonicDirection,
    /// All violations (empty when monotonic).
    pub violations: Vec<MonotonicityViolation>,
}

impl ChannelMonotonicityResult {
    /// Returns `true` when the channel has no violations.
    #[must_use]
    pub fn is_monotonic(&self) -> bool {
        self.direction != MonotonicDirection::NonMonotonic
    }
}

/// Report produced by monotonicity checks.
#[derive(Clone, Debug)]
pub struct MonotonicityReport {
    /// Per-channel results (always 3 for an interleaved RGB 1D LUT).
    pub channels: Vec<ChannelMonotonicityResult>,
    /// `true` when every channel passes the check.
    pub all_monotonic: bool,
}

impl MonotonicityReport {
    /// Number of channels that are NOT monotonic.
    #[must_use]
    pub fn failing_channel_count(&self) -> usize {
        self.channels.iter().filter(|c| !c.is_monotonic()).count()
    }

    /// Total number of violations across all channels.
    #[must_use]
    pub fn total_violations(&self) -> usize {
        self.channels.iter().map(|c| c.violations.len()).sum()
    }
}

/// Check the monotonicity of a 1D LUT stored as interleaved RGB triples.
///
/// `data` must have length `size * 3` (packed as `[r0, g0, b0, r1, g1, b1, …]`).
///
/// Returns `None` when `size < 2` or `data.len() != size * 3`.
#[must_use]
pub fn check_1d_monotonicity(data: &[f64], size: usize) -> Option<MonotonicityReport> {
    if size < 2 || data.len() != size * 3 {
        return None;
    }

    const NAMES: [&str; 3] = ["R", "G", "B"];
    let mut channels = Vec::with_capacity(3);

    for ch in 0..3_usize {
        let vals: Vec<f64> = (0..size).map(|i| data[i * 3 + ch]).collect();

        // Determine dominant direction from first non-zero step.
        let dominant_increasing = vals
            .windows(2)
            .find(|w| (w[1] - w[0]).abs() > 1e-12)
            .map_or(true, |w| w[1] > w[0]);

        let mut violations = Vec::new();
        for (i, w) in vals.windows(2).enumerate() {
            let diff = w[1] - w[0];
            let is_violation = if dominant_increasing {
                diff < -1e-12
            } else {
                diff > 1e-12
            };
            if is_violation {
                violations.push(MonotonicityViolation {
                    index: i,
                    value_a: w[0],
                    value_b: w[1],
                });
            }
        }

        let direction = if violations.is_empty() {
            if dominant_increasing {
                MonotonicDirection::Increasing
            } else {
                MonotonicDirection::Decreasing
            }
        } else {
            MonotonicDirection::NonMonotonic
        };

        channels.push(ChannelMonotonicityResult {
            channel: NAMES[ch].to_string(),
            direction,
            violations,
        });
    }

    let all_monotonic = channels.iter().all(|c| c.is_monotonic());
    Some(MonotonicityReport {
        channels,
        all_monotonic,
    })
}

/// Strictly-increasing check: requires every step to be > 0 (no plateaux).
///
/// Returns `None` for invalid inputs (same conditions as [`check_1d_monotonicity`]).
#[must_use]
pub fn check_1d_strictly_increasing(data: &[f64], size: usize) -> Option<MonotonicityReport> {
    if size < 2 || data.len() != size * 3 {
        return None;
    }

    const NAMES: [&str; 3] = ["R", "G", "B"];
    let mut channels = Vec::with_capacity(3);

    for ch in 0..3_usize {
        let mut violations = Vec::new();
        for i in 0..(size - 1) {
            let a = data[i * 3 + ch];
            let b = data[(i + 1) * 3 + ch];
            if b <= a + 1e-12 {
                violations.push(MonotonicityViolation {
                    index: i,
                    value_a: a,
                    value_b: b,
                });
            }
        }
        let direction = if violations.is_empty() {
            MonotonicDirection::Increasing
        } else {
            MonotonicDirection::NonMonotonic
        };
        channels.push(ChannelMonotonicityResult {
            channel: NAMES[ch].to_string(),
            direction,
            violations,
        });
    }

    let all_monotonic = channels.iter().all(|c| c.is_monotonic());
    Some(MonotonicityReport {
        channels,
        all_monotonic,
    })
}

// ============================================================================
// Smoothness checks for 3D LUTs
// ============================================================================

/// Gradient distribution statistics.
#[derive(Clone, Debug)]
pub struct GradientStats {
    /// Mean Euclidean gradient between adjacent lattice points.
    pub mean: f64,
    /// Maximum gradient.
    pub max: f64,
    /// 95th-percentile gradient.
    pub p95: f64,
    /// Total gradient samples measured.
    pub sample_count: usize,
}

/// Smoothness report from [`check_3d_smoothness`].
#[derive(Clone, Debug)]
pub struct SmoothnessReport {
    /// Gradient statistics across all axis-aligned lattice pairs.
    pub gradient_stats: GradientStats,
    /// `true` when the `rough_pair_count` is zero.
    pub passes: bool,
    /// The threshold used.
    pub roughness_threshold: f64,
    /// Number of lattice pairs whose gradient exceeded `roughness_threshold`.
    pub rough_pair_count: usize,
}

/// Check the smoothness of a 3D LUT stored as a flat slice of RGB triplets.
///
/// For every axis-aligned pair of adjacent lattice nodes, the Euclidean distance
/// between their RGB outputs is computed.  The distribution of these gradients is
/// summarised in a [`SmoothnessReport`].
///
/// An identity LUT has gradients of `1/(size-1)` per axis step.  Much larger
/// values indicate discontinuities that may cause banding artefacts.
///
/// # Arguments
///
/// * `data`                – `size³` RGB entries as `[f64; 3]` slices.
/// * `size`                – number of lattice points per axis.
/// * `roughness_threshold` – gradient threshold; pairs above this are flagged.
///   Pass `f64::INFINITY` to disable.
///
/// Returns `None` when `data.len() != size³` or `size < 2`.
#[must_use]
pub fn check_3d_smoothness(
    data: &[[f64; 3]],
    size: usize,
    roughness_threshold: f64,
) -> Option<SmoothnessReport> {
    if size < 2 || data.len() != size * size * size {
        return None;
    }

    let mut gradients: Vec<f64> = Vec::with_capacity(3 * size * size * (size - 1));
    let idx = |r: usize, g: usize, b: usize| r * size * size + g * size + b;

    // R axis
    for g in 0..size {
        for b in 0..size {
            for r in 0..(size - 1) {
                gradients.push(euclid_rgb(&data[idx(r, g, b)], &data[idx(r + 1, g, b)]));
            }
        }
    }
    // G axis
    for r in 0..size {
        for b in 0..size {
            for g in 0..(size - 1) {
                gradients.push(euclid_rgb(&data[idx(r, g, b)], &data[idx(r, g + 1, b)]));
            }
        }
    }
    // B axis
    for r in 0..size {
        for g in 0..size {
            for b in 0..(size - 1) {
                gradients.push(euclid_rgb(&data[idx(r, g, b)], &data[idx(r, g, b + 1)]));
            }
        }
    }

    let n = gradients.len();
    if n == 0 {
        return None;
    }

    let sum: f64 = gradients.iter().sum();
    let mean = sum / n as f64;
    let max = gradients.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut sorted = gradients.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95_idx = ((n - 1) as f64 * 0.95) as usize;
    let p95 = sorted[p95_idx];

    let rough_pair_count = gradients
        .iter()
        .filter(|&&g| g > roughness_threshold)
        .count();

    Some(SmoothnessReport {
        gradient_stats: GradientStats {
            mean,
            max,
            p95,
            sample_count: n,
        },
        passes: rough_pair_count == 0,
        roughness_threshold,
        rough_pair_count,
    })
}

fn euclid_rgb(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dr = a[0] - b[0];
    let dg = a[1] - b[1];
    let db = a[2] - b[2];
    (dr * dr + dg * dg + db * db).sqrt()
}

/// Generate identity 3D LUT data for a given size.
#[must_use]
pub fn generate_identity_3d(size: usize) -> Vec<f64> {
    let step = if size > 1 {
        1.0 / (size as f64 - 1.0)
    } else {
        0.0
    };
    let mut data = Vec::with_capacity(size * size * size * 3);
    for b in 0..size {
        for g in 0..size {
            for r in 0..size {
                data.push(r as f64 * step);
                data.push(g as f64 * step);
                data.push(b as f64 * step);
            }
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_severity_display() {
        assert_eq!(ValidationSeverity::Info.to_string(), "INFO");
        assert_eq!(ValidationSeverity::Warning.to_string(), "WARNING");
        assert_eq!(ValidationSeverity::Error.to_string(), "ERROR");
    }

    #[test]
    fn test_diagnostic_is_error() {
        let d = ValidationDiagnostic::new(ValidationSeverity::Error, "bad");
        assert!(d.is_error());
        assert!(!d.is_warning());
    }

    #[test]
    fn test_diagnostic_is_warning() {
        let d = ValidationDiagnostic::new(ValidationSeverity::Warning, "warn");
        assert!(!d.is_error());
        assert!(d.is_warning());
    }

    #[test]
    fn test_diagnostic_with_location() {
        let d = ValidationDiagnostic::with_location(ValidationSeverity::Info, "note", "index 42");
        assert_eq!(d.location.as_deref(), Some("index 42"));
    }

    #[test]
    fn test_report_add_error_marks_failed() {
        let mut report = ValidationReport::new("test");
        assert!(report.passed);
        report.add(ValidationDiagnostic::new(ValidationSeverity::Error, "fail"));
        assert!(!report.passed);
    }

    #[test]
    fn test_report_count_severity() {
        let mut report = ValidationReport::new("test");
        report.add(ValidationDiagnostic::new(ValidationSeverity::Warning, "w1"));
        report.add(ValidationDiagnostic::new(ValidationSeverity::Warning, "w2"));
        report.add(ValidationDiagnostic::new(ValidationSeverity::Info, "i1"));
        assert_eq!(report.count_severity(&ValidationSeverity::Warning), 2);
        assert_eq!(report.count_severity(&ValidationSeverity::Info), 1);
        assert_eq!(report.count_severity(&ValidationSeverity::Error), 0);
    }

    #[test]
    fn test_validate_1d_empty_channel() {
        let v = LutValidator::new();
        let report = v.validate_1d_channel(&[], "R");
        assert!(!report.passed);
        assert_eq!(report.errors().len(), 1);
    }

    #[test]
    fn test_validate_1d_good_channel() {
        let v = LutValidator::new();
        let data: Vec<f64> = (0..256).map(|i| i as f64 / 255.0).collect();
        let report = v.validate_1d_channel(&data, "R");
        assert!(report.passed);
        assert_eq!(report.errors().len(), 0);
    }

    #[test]
    fn test_validate_1d_nan_detected() {
        let v = LutValidator::new();
        let data = vec![0.0, f64::NAN, 1.0];
        let report = v.validate_1d_channel(&data, "G");
        assert!(!report.passed);
        assert!(!report.errors().is_empty());
    }

    #[test]
    fn test_validate_3d_wrong_size() {
        let v = LutValidator::new();
        let data = vec![0.0; 100];
        let report = v.validate_3d_data(&data, 5);
        assert!(!report.passed);
    }

    #[test]
    fn test_validate_3d_identity_passes() {
        let v = LutValidator::new();
        let size = 5;
        let data = generate_identity_3d(size);
        let report = v.validate_3d_data(&data, size);
        assert!(report.passed);
        assert_eq!(report.warnings().len(), 0);
    }

    #[test]
    fn test_check_identity_3d_true() {
        let v = LutValidator::new();
        let size = 5;
        let data = generate_identity_3d(size);
        assert!(v.check_identity_3d(&data, size));
    }

    #[test]
    fn test_check_identity_3d_false() {
        let v = LutValidator::new();
        let size = 5;
        let mut data = generate_identity_3d(size);
        data[0] = 0.5; // perturb
        assert!(!v.check_identity_3d(&data, size));
    }

    #[test]
    fn test_max_identity_deviation() {
        let v = LutValidator::new();
        let size = 5;
        let data = generate_identity_3d(size);
        let dev = v.max_identity_deviation_3d(&data, size);
        assert!(dev.is_some());
        assert!(dev.expect("should succeed in test") < 1e-12);
    }

    #[test]
    fn test_is_standard_lut_size() {
        assert!(is_standard_lut_size(17));
        assert!(is_standard_lut_size(33));
        assert!(is_standard_lut_size(65));
        assert!(is_standard_lut_size(129));
        assert!(!is_standard_lut_size(16));
        assert!(!is_standard_lut_size(32));
    }

    #[test]
    fn test_generate_identity_3d_length() {
        let data = generate_identity_3d(5);
        assert_eq!(data.len(), 5 * 5 * 5 * 3);
    }

    #[test]
    fn test_validator_default() {
        let v = LutValidator::default();
        assert!((v.config.min_value - 0.0).abs() < f64::EPSILON);
        assert!((v.config.max_value - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validate_1d_out_of_range() {
        let v = LutValidator::new();
        let data = vec![0.0, 0.5, 1.5];
        let report = v.validate_1d_channel(&data, "B");
        assert!(report.passed); // out-of-range is a warning, not error
        assert_eq!(report.warnings().len(), 1);
    }

    // -----------------------------------------------------------------------
    // Monotonicity tests (Item 2 implementation)
    // -----------------------------------------------------------------------

    fn make_increasing_interleaved(size: usize) -> Vec<f64> {
        let scale = (size - 1) as f64;
        (0..size)
            .flat_map(|i| {
                let v = i as f64 / scale;
                [v, v, v]
            })
            .collect()
    }

    fn make_decreasing_interleaved(size: usize) -> Vec<f64> {
        let scale = (size - 1) as f64;
        (0..size)
            .flat_map(|i| {
                let v = 1.0 - i as f64 / scale;
                [v, v, v]
            })
            .collect()
    }

    fn make_identity_3d_rgb(size: usize) -> Vec<[f64; 3]> {
        let scale = (size - 1) as f64;
        let mut data = Vec::with_capacity(size * size * size);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    data.push([r as f64 / scale, g as f64 / scale, b as f64 / scale]);
                }
            }
        }
        data
    }

    #[test]
    fn test_monotonicity_increasing_passes() {
        let data = make_increasing_interleaved(17);
        let report = check_1d_monotonicity(&data, 17).expect("valid");
        assert!(report.all_monotonic);
        assert_eq!(report.total_violations(), 0);
        for ch in &report.channels {
            assert_eq!(ch.direction, MonotonicDirection::Increasing);
        }
    }

    #[test]
    fn test_monotonicity_decreasing_passes() {
        let data = make_decreasing_interleaved(9);
        let report = check_1d_monotonicity(&data, 9).expect("valid");
        assert!(report.all_monotonic);
        for ch in &report.channels {
            assert_eq!(ch.direction, MonotonicDirection::Decreasing);
        }
    }

    #[test]
    fn test_monotonicity_non_monotonic_detected() {
        let mut data = make_increasing_interleaved(11);
        let mid = 5;
        data[mid * 3] = data[(mid - 1) * 3] - 0.05; // inject reversal in R
        let report = check_1d_monotonicity(&data, 11).expect("valid");
        assert!(!report.all_monotonic);
        assert!(report.total_violations() > 0);
        assert!(report.failing_channel_count() >= 1);
    }

    #[test]
    fn test_monotonicity_violation_location_recorded() {
        let mut data = make_increasing_interleaved(11);
        let mid = 5;
        data[mid * 3] = data[(mid - 1) * 3] - 0.05;
        let report = check_1d_monotonicity(&data, 11).expect("valid");
        let red = report
            .channels
            .iter()
            .find(|c| c.channel == "R")
            .expect("R");
        assert!(!red.violations.is_empty());
        for v in &red.violations {
            assert!(v.index < 10);
        }
    }

    #[test]
    fn test_monotonicity_invalid_returns_none() {
        let data = vec![0.0f64; 15];
        assert!(check_1d_monotonicity(&data, 1).is_none()); // size < 2
        assert!(check_1d_monotonicity(&data, 10).is_none()); // len mismatch
    }

    #[test]
    fn test_strict_increasing_identity_passes() {
        let data = make_increasing_interleaved(33);
        let report = check_1d_strictly_increasing(&data, 33).expect("valid");
        assert!(report.all_monotonic);
        assert_eq!(report.total_violations(), 0);
    }

    #[test]
    fn test_strict_increasing_plateau_fails() {
        let mut data = make_increasing_interleaved(9);
        data[3 * 3 + 1] = data[4 * 3 + 1]; // plateau in G
        let report = check_1d_strictly_increasing(&data, 9).expect("valid");
        assert!(!report.all_monotonic);
        let g = report
            .channels
            .iter()
            .find(|c| c.channel == "G")
            .expect("G");
        assert!(!g.violations.is_empty());
    }

    #[test]
    fn test_strict_increasing_failing_count() {
        let mut data = make_increasing_interleaved(9);
        data[2 * 3] = data[3 * 3]; // R plateau
        let report = check_1d_strictly_increasing(&data, 9).expect("valid");
        assert!(report.failing_channel_count() >= 1);
    }

    // -----------------------------------------------------------------------
    // Smoothness tests (3D LUT)
    // -----------------------------------------------------------------------

    #[test]
    fn test_smoothness_identity_lut_passes() {
        let size = 5_usize;
        let data = make_identity_3d_rgb(size);
        let report = check_3d_smoothness(&data, size, 0.5).expect("valid");
        assert!(report.passes);
        assert_eq!(report.rough_pair_count, 0);
        assert!(report.gradient_stats.mean > 0.0);
        assert!(report.gradient_stats.max < 0.5);
    }

    #[test]
    fn test_smoothness_discontinuity_detected() {
        let size = 5_usize;
        let mut data = make_identity_3d_rgb(size);
        data[0] = [0.9, 0.9, 0.9]; // large jump
        let report = check_3d_smoothness(&data, size, 0.2).expect("valid");
        assert!(!report.passes);
        assert!(report.rough_pair_count > 0);
    }

    #[test]
    fn test_smoothness_gradient_stats_sane() {
        let size = 3_usize;
        let data = make_identity_3d_rgb(size);
        let report = check_3d_smoothness(&data, size, f64::INFINITY).expect("valid");
        assert!(report.passes);
        assert!(report.gradient_stats.mean > 0.0);
        assert!(report.gradient_stats.max >= report.gradient_stats.mean);
        assert!(report.gradient_stats.p95 <= report.gradient_stats.max);
    }

    #[test]
    fn test_smoothness_invalid_returns_none() {
        let data = vec![[0.5f64; 3]; 27];
        assert!(check_3d_smoothness(&data, 1, 1.0).is_none()); // size < 2
        assert!(check_3d_smoothness(&data, 4, 1.0).is_none()); // 4³=64 ≠ 27
    }

    #[test]
    fn test_smoothness_sample_count() {
        let size = 4_usize;
        let data = make_identity_3d_rgb(size);
        let report = check_3d_smoothness(&data, size, f64::INFINITY).expect("valid");
        let expected = 3 * size * size * (size - 1);
        assert_eq!(report.gradient_stats.sample_count, expected);
    }

    #[test]
    fn test_smoothness_constant_lut_zero_gradient() {
        let size = 3_usize;
        let data = vec![[0.5f64; 3]; size * size * size];
        let report = check_3d_smoothness(&data, size, 0.0001).expect("valid");
        assert!(report.passes);
        assert!(report.gradient_stats.max < 1e-12);
    }
}
