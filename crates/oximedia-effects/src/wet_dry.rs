//! Wet/dry mix control for audio effects.
//!
//! Provides a standalone [`WetDryMix`] struct for blending processed (wet) and
//! unprocessed (dry) signals, with support for both linear and equal-power
//! crossfade curves.

use std::f32::consts::{FRAC_PI_2, SQRT_2};

/// Wet/dry mix descriptor for any effect that produces `f32` samples.
///
/// `wet = 1.0` → 100% effect signal (fully wet).
/// `wet = 0.0` → 100% unprocessed signal (fully dry / bypass).
///
/// # Equal-power mode
/// When `equal_power` is `true`, gains follow a constant-power curve:
/// - `wet_gain  = sin(wet * π/2)`
/// - `dry_gain  = cos(wet * π/2)`
///
/// This ensures that the perceived loudness stays constant during a
/// fade between the two signals.
///
/// # Linear mode
/// When `equal_power` is `false` (default):
/// - `wet_gain = wet`
/// - `dry_gain = 1.0 - wet`
///
/// # Examples
/// ```
/// use oximedia_effects::wet_dry::WetDryMix;
///
/// let mut mix = WetDryMix::new(0.4);
/// assert!((mix.wet_gain() - 0.4).abs() < 1e-6);
/// assert!((mix.dry_gain() - 0.6).abs() < 1e-6);
///
/// let dry = vec![0.5_f32; 4];
/// let wet = vec![1.0_f32; 4];
/// let out = mix.apply(&dry, &wet);
/// assert_eq!(out.len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct WetDryMix {
    /// Wet level in `[0.0, 1.0]`.
    pub wet: f32,
    /// Precomputed dry level (1.0 − wet for linear, cos-curve for equal-power).
    pub dry: f32,
    /// Whether to use equal-power (constant-loudness) crossfade.
    pub equal_power: bool,
}

impl WetDryMix {
    /// Create a linear wet/dry mix.
    ///
    /// `dry` is automatically set to `1.0 − wet`.
    #[must_use]
    pub fn new(wet: f32) -> Self {
        let wet = wet.clamp(0.0, 1.0);
        Self {
            wet,
            dry: 1.0 - wet,
            equal_power: false,
        }
    }

    /// Create an equal-power (constant-loudness) wet/dry mix.
    ///
    /// Uses `sin(wet * π/2)` / `cos(wet * π/2)` for gain values.
    #[must_use]
    pub fn equal_power(wet: f32) -> Self {
        let wet = wet.clamp(0.0, 1.0);
        let angle = wet * FRAC_PI_2;
        Self {
            wet,
            dry: angle.cos(),
            equal_power: true,
        }
    }

    /// Bypass: 100% dry signal (`wet = 0.0`).
    #[must_use]
    pub fn bypass() -> Self {
        Self::new(0.0)
    }

    /// Full wet: 100% effect signal (`wet = 1.0`).
    #[must_use]
    pub fn full_wet() -> Self {
        Self::new(1.0)
    }

    /// Effective gain applied to the wet (processed) signal.
    #[must_use]
    pub fn wet_gain(&self) -> f32 {
        if self.equal_power {
            (self.wet * FRAC_PI_2).sin()
        } else {
            self.wet
        }
    }

    /// Effective gain applied to the dry (unprocessed) signal.
    #[must_use]
    pub fn dry_gain(&self) -> f32 {
        if self.equal_power {
            (self.wet * FRAC_PI_2).cos()
        } else {
            self.dry
        }
    }

    /// Mix `dry` and `wet` sample slices, returning a new `Vec<f32>`.
    ///
    /// Output length equals `dry.len().min(wet.len())`.
    #[must_use]
    pub fn apply(&self, dry: &[f32], wet: &[f32]) -> Vec<f32> {
        let len = dry.len().min(wet.len());
        let wg = self.wet_gain();
        let dg = self.dry_gain();
        (0..len).map(|i| dg * dry[i] + wg * wet[i]).collect()
    }

    /// In-place mix: `wet[i] ← dry_gain × dry[i] + wet_gain × wet[i]`.
    ///
    /// Processes up to `dry.len().min(wet.len())` samples.
    pub fn apply_inplace(&self, dry: &[f32], wet: &mut [f32]) {
        let len = dry.len().min(wet.len());
        let wg = self.wet_gain();
        let dg = self.dry_gain();
        for i in 0..len {
            wet[i] = dg * dry[i] + wg * wet[i];
        }
    }

    /// Update the wet level and recompute the dry level.
    pub fn set_wet(&mut self, wet: f32) {
        self.wet = wet.clamp(0.0, 1.0);
        if self.equal_power {
            self.dry = (self.wet * FRAC_PI_2).cos();
        } else {
            self.dry = 1.0 - self.wet;
        }
    }

    /// Serialise to a compact JSON string (no external dependencies).
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"wet":{:.6},"dry":{:.6},"equal_power":{}}}"#,
            self.wet, self.dry, self.equal_power
        )
    }
}

/// Verify that equal-power gains satisfy the constant-power invariant:
/// `wet_gain² + dry_gain² ≈ 1.0`.
#[must_use]
pub fn equal_power_sum_of_squares(mix: &WetDryMix) -> f32 {
    let wg = mix.wet_gain();
    let dg = mix.dry_gain();
    wg * wg + dg * dg
}

/// Quick sanity check: at any equal-power mix position, `√2 × gain_at_45°` ≈ 1.
#[must_use]
pub fn equal_power_midpoint_gain() -> f32 {
    // At wet = 0.5, angle = π/4, sin = cos = 1/√2.
    // Returned as the gain value (should be ≈ 1/√2).
    1.0 / SQRT_2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_wet_applies_only_wet() {
        let mix = WetDryMix::full_wet();
        let dry = vec![0.0_f32; 4];
        let wet = vec![1.0_f32; 4];
        let out = mix.apply(&dry, &wet);
        for v in &out {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "full_wet should give wet=1.0, got {v}"
            );
        }
    }

    #[test]
    fn test_bypass_returns_dry() {
        let mix = WetDryMix::bypass();
        let dry = vec![0.5_f32, -0.3, 0.8, 0.1];
        let wet = vec![1.0_f32; 4];
        let out = mix.apply(&dry, &wet);
        for (i, (&d, &o)) in dry.iter().zip(out.iter()).enumerate() {
            assert!(
                (d - o).abs() < 1e-6,
                "bypass should return dry at {i}: got {o}"
            );
        }
    }

    #[test]
    fn test_equal_power_sum_of_squares_approx_one() {
        // At any wet level the equal-power gains must satisfy wg² + dg² ≈ 1.
        for i in 0..=10 {
            let wet = i as f32 / 10.0;
            let mix = WetDryMix::equal_power(wet);
            let s = equal_power_sum_of_squares(&mix);
            assert!(
                (s - 1.0).abs() < 1e-5,
                "equal_power sum-of-squares at wet={wet} = {s}, expected ≈1.0"
            );
        }
    }

    #[test]
    fn test_apply_length_correct() {
        let mix = WetDryMix::new(0.5);
        let dry = vec![0.0_f32; 7];
        let wet = vec![1.0_f32; 5];
        let out = mix.apply(&dry, &wet);
        assert_eq!(out.len(), 5, "apply should return min(dry,wet) length");
    }

    #[test]
    fn test_set_wet_updates_both_gains() {
        let mut mix = WetDryMix::new(0.2);
        mix.set_wet(0.7);
        assert!((mix.wet - 0.7).abs() < 1e-6);
        assert!((mix.dry - 0.3).abs() < 1e-6);
        assert!((mix.wet_gain() - 0.7).abs() < 1e-6);
        assert!((mix.dry_gain() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_set_wet_equal_power_updates_correctly() {
        let mut mix = WetDryMix::equal_power(0.0);
        mix.set_wet(1.0);
        // wet=1 → angle=π/2 → sin=1, cos≈0
        assert!((mix.wet_gain() - 1.0).abs() < 1e-6);
        assert!(mix.dry_gain().abs() < 1e-6);
    }

    #[test]
    fn test_apply_inplace_blends_correctly() {
        let mix = WetDryMix::new(0.5);
        let dry = vec![1.0_f32; 4];
        let mut wet = vec![0.0_f32; 4];
        mix.apply_inplace(&dry, &mut wet);
        // expected: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        for v in &wet {
            assert!(
                (v - 0.5).abs() < 1e-6,
                "apply_inplace result should be 0.5, got {v}"
            );
        }
    }

    #[test]
    fn test_linear_wet_dry_gains() {
        let mix = WetDryMix::new(0.3);
        assert!((mix.wet_gain() - 0.3).abs() < 1e-6);
        assert!((mix.dry_gain() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_to_json_contains_wet() {
        let mix = WetDryMix::new(0.5);
        let json = mix.to_json();
        assert!(json.contains("\"wet\""), "JSON should contain wet field");
        assert!(json.contains("\"dry\""), "JSON should contain dry field");
        assert!(
            json.contains("\"equal_power\""),
            "JSON should contain equal_power field"
        );
    }

    #[test]
    fn test_clamp_wet_above_one() {
        let mix = WetDryMix::new(1.5);
        assert!((mix.wet - 1.0).abs() < 1e-6, "wet should clamp to 1.0");
    }
}
