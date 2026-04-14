//! HDR10+ dynamic metadata generator.
//!
//! Generates per-shot trim-pass HDR10+ dynamic metadata from scene luminance
//! statistics, including Bézier curve parameters, percentile distribution
//! values, and JSON serialisation compatible with the HDR10+ toolset.
//!
//! Reference: SMPTE ST 2094-40:2020 — Dynamic Metadata for Color Volume
//! Transform — Application #4.

use crate::{HdrError, Result};

// ─── Scene luminance statistics ───────────────────────────────────────────────

/// Luminance statistics for a single shot or scene.
///
/// All luminance values are in nits (cd/m²).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SceneLuminanceStats {
    /// Frame or shot identifier.
    pub shot_id: u64,
    /// Minimum scene luminance (nits).
    pub min_nits: f32,
    /// Maximum scene luminance (nits) — MaxRGB measure.
    pub max_nits: f32,
    /// Average MaxRGB luminance (nits).
    pub avg_maxrgb_nits: f32,
    /// Luminance percentiles at \[1, 5, 10, 25, 50, 75, 90, 95, 99\] percent.
    pub percentiles: [f32; 9],
    /// Fraction of pixels whose luminance exceeds the `high_threshold_nits`
    /// (stored as a value in 0.0–1.0).
    pub fraction_bright: f32,
    /// The luminance threshold used to compute `fraction_bright` (nits).
    pub high_threshold_nits: f32,
}

impl SceneLuminanceStats {
    /// Create statistics from a flat slice of per-pixel luminance values.
    ///
    /// `pixels_nits` must be non-empty.  Percentiles are computed by sorting a
    /// copy of the input (heap allocation proportional to input length).
    ///
    /// `high_threshold_nits` defines the brightness threshold for
    /// `fraction_bright`.
    pub fn from_pixels(
        shot_id: u64,
        pixels_nits: &[f32],
        high_threshold_nits: f32,
    ) -> Result<Self> {
        if pixels_nits.is_empty() {
            return Err(HdrError::MetadataParseError(
                "cannot compute luminance stats from empty pixel buffer".to_string(),
            ));
        }

        let mut sorted: Vec<f32> = pixels_nits.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let min_nits = sorted[0];
        let max_nits = sorted[n - 1];

        // Percentile indices: 1, 5, 10, 25, 50, 75, 90, 95, 99
        let pct_targets = [1.0_f32, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0];
        let mut percentiles = [0.0f32; 9];
        for (i, &pct) in pct_targets.iter().enumerate() {
            let idx = ((pct / 100.0) * (n - 1) as f32).round() as usize;
            percentiles[i] = sorted[idx.min(n - 1)];
        }

        let sum: f32 = sorted.iter().copied().sum();
        let avg_maxrgb_nits = sum / n as f32;

        let bright_count = sorted
            .iter()
            .filter(|&&v| v > high_threshold_nits)
            .count();
        let fraction_bright = bright_count as f32 / n as f32;

        Ok(Self {
            shot_id,
            min_nits,
            max_nits,
            avg_maxrgb_nits,
            percentiles,
            fraction_bright,
            high_threshold_nits,
        })
    }

    /// Validate that all luminance values are physically meaningful.
    pub fn validate(&self) -> Result<()> {
        if self.min_nits < 0.0 {
            return Err(HdrError::InvalidLuminance(self.min_nits));
        }
        if self.max_nits < self.min_nits {
            return Err(HdrError::InvalidLuminance(self.max_nits));
        }
        if self.avg_maxrgb_nits < 0.0 {
            return Err(HdrError::InvalidLuminance(self.avg_maxrgb_nits));
        }
        if !(0.0_f32..=1.0_f32).contains(&self.fraction_bright) {
            return Err(HdrError::MetadataParseError(format!(
                "fraction_bright out of range: {}",
                self.fraction_bright
            )));
        }
        Ok(())
    }
}

// ─── Bézier curve parameters ──────────────────────────────────────────────────

/// A cubic Bézier tone-curve anchor point with normalised coordinates.
///
/// `t` is the input luminance normalised to the mastering display peak (0–1).
/// `v` is the target output luminance normalised to the target display peak (0–1).
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct BezierAnchor {
    /// Normalised input luminance (0.0–1.0).
    pub t: f32,
    /// Normalised output luminance (0.0–1.0).
    pub v: f32,
}

impl BezierAnchor {
    /// Create a new anchor point, clamping both coordinates to \[0, 1\].
    pub fn new(t: f32, v: f32) -> Self {
        Self {
            t: t.clamp(0.0, 1.0),
            v: v.clamp(0.0, 1.0),
        }
    }
}

/// A cubic Bézier tone-mapping curve defined by a sequence of anchor points.
///
/// The curve is evaluated by finding the two anchors that bracket the input `t`
/// and performing a cubic Bézier interpolation between them.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BezierToneCurve {
    /// Sorted anchor points (ascending by `t`).  Must contain at least 2 points.
    pub anchors: Vec<BezierAnchor>,
    /// Knee-point input value (normalised, 0–1).
    pub knee_point_x: f32,
    /// Knee-point output value (normalised, 0–1).
    pub knee_point_y: f32,
}

impl BezierToneCurve {
    /// Build a Bézier tone curve from scene luminance statistics.
    ///
    /// The curve maps the mastering display peak (`mastering_peak_nits`) to the
    /// target display peak (`target_peak_nits`), preserving shadow detail while
    /// applying a smooth highlight roll-off.
    ///
    /// Returns an error if the peaks are invalid or the stats fail validation.
    pub fn from_stats(
        stats: &SceneLuminanceStats,
        mastering_peak_nits: f32,
        target_peak_nits: f32,
    ) -> Result<Self> {
        stats.validate()?;
        if mastering_peak_nits <= 0.0 {
            return Err(HdrError::InvalidLuminance(mastering_peak_nits));
        }
        if target_peak_nits <= 0.0 {
            return Err(HdrError::InvalidLuminance(target_peak_nits));
        }

        // Normalise key luminance points to mastering peak.
        let safe_master = mastering_peak_nits.max(1.0);
        let ratio = (target_peak_nits / safe_master).clamp(0.001, 1.0);

        // Shadow anchor: black point maps to black point.
        let a0 = BezierAnchor::new(0.0, 0.0);

        // Shadow detail preservation — preserve the shadow region linearly.
        let shadow_nits = stats.percentiles[0].max(0.0); // 1st percentile
        let t_shadow = (shadow_nits / safe_master).clamp(0.0, 1.0);
        let a1 = BezierAnchor::new(t_shadow, t_shadow);

        // Mid-tone anchor: 50th percentile maps proportionally.
        let median_nits = stats.percentiles[4].max(0.0);
        let t_mid = (median_nits / safe_master).clamp(0.0, 1.0);
        let v_mid = (t_mid * ratio).clamp(0.0, 1.0);
        let a2 = BezierAnchor::new(t_mid, v_mid);

        // Highlight knee — 95th percentile.
        let knee_nits = stats.percentiles[7].max(0.0);
        let knee_x = (knee_nits / safe_master).clamp(0.0, 1.0);
        let knee_y = (knee_x * ratio).clamp(0.0, 1.0);
        let a3 = BezierAnchor::new(knee_x, knee_y);

        // Peak anchor: full highlights map to target ratio.
        let a4 = BezierAnchor::new(1.0, ratio);

        Ok(Self {
            anchors: vec![a0, a1, a2, a3, a4],
            knee_point_x: knee_x,
            knee_point_y: knee_y,
        })
    }

    /// Evaluate the tone curve at a normalised input `t` ∈ \[0, 1\].
    ///
    /// Performs a linear search for the enclosing segment and applies cubic
    /// Hermite interpolation between the two bounding anchors.
    pub fn evaluate(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        let anchors = &self.anchors;
        let n = anchors.len();

        if n == 0 {
            return t;
        }
        if n == 1 {
            return anchors[0].v;
        }

        // Find the enclosing segment.
        let mut lo = 0usize;
        let mut hi = n - 1;
        for i in 0..n - 1 {
            if t <= anchors[i + 1].t {
                lo = i;
                hi = i + 1;
                break;
            }
        }

        let t0 = anchors[lo].t;
        let t1 = anchors[hi].t;
        let v0 = anchors[lo].v;
        let v1 = anchors[hi].v;

        if (t1 - t0).abs() < 1e-9 {
            return v0;
        }

        // Normalise to segment space and apply smoothstep blend.
        let u = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
        let u_smooth = u * u * (3.0 - 2.0 * u);
        v0 + u_smooth * (v1 - v0)
    }
}

// ─── Trim-pass generator ──────────────────────────────────────────────────────

/// A per-shot HDR10+ trim pass descriptor.
///
/// One trim pass targets one specific display tier (e.g., 1000-nit consumer,
/// 4000-nit reference).  Multiple trim passes may be embedded in an HDR10+
/// stream to cover multiple display capabilities.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Hdr10PlusTrimPass {
    /// Shot or scene identifier.
    pub shot_id: u64,
    /// Target display peak luminance (nits) for this trim pass.
    pub target_display_peak_nits: f32,
    /// Mastering display peak luminance (nits).
    pub mastering_display_peak_nits: f32,
    /// Bézier tone curve for this trim pass.
    pub tone_curve: BezierToneCurve,
    /// Targeted system display maximum luminance (×10, stored as in ST 2094).
    pub targeted_display_max_luminance: u32,
    /// Average MaxRGB in nits.
    pub avg_maxrgb_nits: f32,
    /// Distribution (percentile) values normalised to 0–65535.
    pub distribution_values: [u16; 9],
    /// Fraction of bright pixels (0–255 per ST 2094).
    pub fraction_bright_pixels: u8,
}

impl Hdr10PlusTrimPass {
    /// Generate a trim pass from scene luminance statistics.
    ///
    /// The tone curve is derived automatically from the stats and the mastering
    /// vs target display peak luminances.
    pub fn from_stats(
        stats: &SceneLuminanceStats,
        mastering_peak_nits: f32,
        target_peak_nits: f32,
    ) -> Result<Self> {
        let tone_curve =
            BezierToneCurve::from_stats(stats, mastering_peak_nits, target_peak_nits)?;

        // Encode distribution values: normalise percentiles to 0–65535 using
        // the mastering peak as the reference.
        let mut distribution_values = [0u16; 9];
        let safe_master = mastering_peak_nits.max(1.0);
        for (i, &pct_nits) in stats.percentiles.iter().enumerate() {
            let norm = (pct_nits / safe_master).clamp(0.0, 1.0);
            distribution_values[i] = (norm * 65535.0).round() as u16;
        }

        let fraction_bright_pixels = (stats.fraction_bright * 255.0).round().clamp(0.0, 255.0) as u8;

        Ok(Self {
            shot_id: stats.shot_id,
            target_display_peak_nits: target_peak_nits,
            mastering_display_peak_nits: mastering_peak_nits,
            tone_curve,
            targeted_display_max_luminance: (target_peak_nits * 10.0).round() as u32,
            avg_maxrgb_nits: stats.avg_maxrgb_nits,
            distribution_values,
            fraction_bright_pixels,
        })
    }

    /// Serialise this trim pass to a JSON string.
    ///
    /// The JSON structure matches the informal HDR10+ metadata JSON schema used
    /// by open-source HDR10+ tools.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            HdrError::MetadataParseError(format!("JSON serialisation error: {e}"))
        })
    }

    /// Deserialise a trim pass from a JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| HdrError::MetadataParseError(format!("JSON parse error: {e}")))
    }
}

// ─── Multi-shot generator ─────────────────────────────────────────────────────

/// Batch generator: produce trim passes for an ordered sequence of shots.
///
/// A single `Hdr10PlusMetadataStream` can hold trim passes for multiple target
/// display tiers, allowing a player to select the appropriate trim pass at
/// decode time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Hdr10PlusMetadataStream {
    /// Mastering display peak luminance (nits) — same for all shots.
    pub mastering_peak_nits: f32,
    /// List of target display tiers (nits) for which trim passes are generated.
    pub target_tiers_nits: Vec<f32>,
    /// All generated trim passes, grouped by shot then by target tier.
    pub trim_passes: Vec<Hdr10PlusTrimPass>,
}

impl Hdr10PlusMetadataStream {
    /// Create a new empty metadata stream.
    pub fn new(mastering_peak_nits: f32, target_tiers_nits: Vec<f32>) -> Result<Self> {
        if mastering_peak_nits <= 0.0 {
            return Err(HdrError::InvalidLuminance(mastering_peak_nits));
        }
        if target_tiers_nits.is_empty() {
            return Err(HdrError::MetadataParseError(
                "target_tiers_nits must not be empty".to_string(),
            ));
        }
        for &t in &target_tiers_nits {
            if t <= 0.0 {
                return Err(HdrError::InvalidLuminance(t));
            }
        }
        Ok(Self {
            mastering_peak_nits,
            target_tiers_nits,
            trim_passes: Vec::new(),
        })
    }

    /// Add one shot's worth of trim passes for all configured target tiers.
    pub fn add_shot(&mut self, stats: &SceneLuminanceStats) -> Result<()> {
        stats.validate()?;
        for &tier in &self.target_tiers_nits.clone() {
            let pass = Hdr10PlusTrimPass::from_stats(stats, self.mastering_peak_nits, tier)?;
            self.trim_passes.push(pass);
        }
        Ok(())
    }

    /// Return all trim passes for a given shot ID.
    pub fn passes_for_shot(&self, shot_id: u64) -> Vec<&Hdr10PlusTrimPass> {
        self.trim_passes
            .iter()
            .filter(|p| p.shot_id == shot_id)
            .collect()
    }

    /// Return all trim passes targeting a specific display tier (nearest match).
    ///
    /// Matches by finding the tier value in `target_tiers_nits` closest to
    /// `target_nits` and collecting passes whose
    /// `target_display_peak_nits` matches that tier.
    pub fn passes_for_tier(&self, target_nits: f32) -> Vec<&Hdr10PlusTrimPass> {
        // Find closest registered tier.
        let Some(&closest_tier) = self
            .target_tiers_nits
            .iter()
            .min_by(|&&a, &&b| {
                (a - target_nits)
                    .abs()
                    .partial_cmp(&(b - target_nits).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        else {
            return Vec::new();
        };

        self.trim_passes
            .iter()
            .filter(|p| (p.target_display_peak_nits - closest_tier).abs() < 1.0)
            .collect()
    }

    /// Serialise the entire stream to a JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            HdrError::MetadataParseError(format!("JSON serialisation error: {e}"))
        })
    }

    /// Number of trim passes in this stream.
    pub fn len(&self) -> usize {
        self.trim_passes.len()
    }

    /// Return `true` if no trim passes have been added.
    pub fn is_empty(&self) -> bool {
        self.trim_passes.is_empty()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_stats(shot_id: u64) -> SceneLuminanceStats {
        let pixels: Vec<f32> = (0..1000)
            .map(|i| (i as f32 / 999.0) * 1000.0) // 0–1000 nits
            .collect();
        SceneLuminanceStats::from_pixels(shot_id, &pixels, 800.0).unwrap()
    }

    // ── SceneLuminanceStats ───────────────────────────────────────────────────

    #[test]
    fn test_stats_from_pixels_basic() {
        let stats = sample_stats(1);
        assert!(stats.min_nits >= 0.0);
        assert!(stats.max_nits > stats.min_nits);
        assert!(stats.avg_maxrgb_nits > 0.0);
    }

    #[test]
    fn test_stats_from_pixels_empty_fails() {
        let result = SceneLuminanceStats::from_pixels(1, &[], 500.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_stats_percentile_ordering() {
        let stats = sample_stats(1);
        // Percentiles should be non-decreasing.
        for i in 0..8 {
            assert!(
                stats.percentiles[i] <= stats.percentiles[i + 1],
                "percentile[{i}] > percentile[{}]: {} > {}",
                i + 1,
                stats.percentiles[i],
                stats.percentiles[i + 1]
            );
        }
    }

    #[test]
    fn test_stats_validate_ok() {
        let stats = sample_stats(1);
        assert!(stats.validate().is_ok());
    }

    // ── BezierToneCurve ───────────────────────────────────────────────────────

    #[test]
    fn test_bezier_curve_from_stats_builds_successfully() {
        let stats = sample_stats(1);
        let curve = BezierToneCurve::from_stats(&stats, 4000.0, 1000.0);
        assert!(curve.is_ok(), "curve build failed: {:?}", curve.err());
    }

    #[test]
    fn test_bezier_curve_evaluate_zero_maps_to_zero() {
        let stats = sample_stats(1);
        let curve = BezierToneCurve::from_stats(&stats, 4000.0, 1000.0).unwrap();
        let out = curve.evaluate(0.0);
        assert!(
            out.abs() < 1e-5,
            "evaluate(0) should be 0.0, got {out}"
        );
    }

    #[test]
    fn test_bezier_curve_evaluate_monotone() {
        let stats = sample_stats(1);
        let curve = BezierToneCurve::from_stats(&stats, 4000.0, 1000.0).unwrap();
        let mut prev = 0.0_f32;
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let v = curve.evaluate(t);
            assert!(v >= prev - 1e-5, "curve not monotone at t={t}: {v} < prev={prev}");
            prev = v;
        }
    }

    #[test]
    fn test_bezier_curve_evaluate_in_range() {
        let stats = sample_stats(1);
        let curve = BezierToneCurve::from_stats(&stats, 4000.0, 1000.0).unwrap();
        for i in 0..=50 {
            let t = i as f32 / 50.0;
            let v = curve.evaluate(t);
            assert!(
                (0.0..=1.0).contains(&v),
                "evaluate({t}) = {v} out of [0,1]"
            );
        }
    }

    // ── Hdr10PlusTrimPass ─────────────────────────────────────────────────────

    #[test]
    fn test_trim_pass_from_stats() {
        let stats = sample_stats(42);
        let pass = Hdr10PlusTrimPass::from_stats(&stats, 4000.0, 1000.0);
        assert!(pass.is_ok(), "trim pass failed: {:?}", pass.err());
        let pass = pass.unwrap();
        assert_eq!(pass.shot_id, 42);
        assert!((pass.target_display_peak_nits - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn test_trim_pass_json_roundtrip() {
        let stats = sample_stats(7);
        let pass = Hdr10PlusTrimPass::from_stats(&stats, 4000.0, 1000.0).unwrap();
        let json = pass.to_json().expect("serialise");
        let decoded = Hdr10PlusTrimPass::from_json(&json).expect("deserialise");
        assert_eq!(decoded.shot_id, pass.shot_id);
        assert!(
            (decoded.target_display_peak_nits - pass.target_display_peak_nits).abs() < 0.01
        );
    }

    #[test]
    fn test_trim_pass_fraction_bright_in_range() {
        let stats = sample_stats(1);
        let pass = Hdr10PlusTrimPass::from_stats(&stats, 4000.0, 1000.0).unwrap();
        // fraction_bright_pixels is u8 so always in [0, 255].
        let _ = pass.fraction_bright_pixels;
    }

    // ── Hdr10PlusMetadataStream ───────────────────────────────────────────────

    #[test]
    fn test_stream_add_multiple_shots() {
        let mut stream =
            Hdr10PlusMetadataStream::new(4000.0, vec![1000.0, 600.0]).expect("stream");
        for shot_id in 0..3 {
            let stats = sample_stats(shot_id);
            stream.add_shot(&stats).expect("add shot");
        }
        // 3 shots × 2 tiers = 6 trim passes.
        assert_eq!(stream.len(), 6);
    }

    #[test]
    fn test_stream_passes_for_shot() {
        let mut stream =
            Hdr10PlusMetadataStream::new(4000.0, vec![1000.0, 600.0]).expect("stream");
        stream.add_shot(&sample_stats(10)).expect("shot 10");
        stream.add_shot(&sample_stats(20)).expect("shot 20");
        let passes = stream.passes_for_shot(10);
        assert_eq!(passes.len(), 2, "expected 2 trim passes for shot 10");
    }

    #[test]
    fn test_stream_invalid_mastering_peak() {
        let result = Hdr10PlusMetadataStream::new(0.0, vec![1000.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_stream_json_serialises() {
        let mut stream = Hdr10PlusMetadataStream::new(4000.0, vec![1000.0]).expect("stream");
        stream.add_shot(&sample_stats(1)).expect("shot");
        let json = stream.to_json().expect("json");
        assert!(json.contains("mastering_peak_nits"));
    }
}
