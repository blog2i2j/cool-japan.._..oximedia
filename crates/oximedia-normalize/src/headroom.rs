//! Headroom management for loudness normalisation.
//!
//! [`HeadroomManager`] ensures a configurable amount of headroom is preserved
//! above the measured true peak when applying gain.  It clamps the gain so
//! that the output peak never exceeds `0 dBFS − target_headroom_db`.
//!
//! # Headroom Calculation
//!
//! ```text
//!   allowed_peak_db = -target_headroom_db        (e.g. -1.0 dBFS for 1 dB headroom)
//!   max_safe_gain   = allowed_peak_db - measured_peak_db
//!   applied_gain    = min(requested_gain, max_safe_gain)
//! ```
//!
//! All gain values are expressed in dB; sample values are linear amplitude.
//!
//! # Example
//!
//! ```
//! use oximedia_normalize::headroom::HeadroomManager;
//!
//! let mgr = HeadroomManager::new(1.0); // 1 dB headroom below 0 dBFS
//!
//! let mut samples = vec![0.5_f32; 8];
//! // Measured peak at -6 dBFS; request +4 dB gain
//! // Max safe gain = -1.0 - (-6.0) = +5.0 dB → gain of +4 dB is safe
//! mgr.apply_gain(&mut samples, -6.0);
//! // All samples should be louder than before
//! assert!(samples[0] > 0.5);
//! ```

/// Headroom manager.
///
/// Holds a target headroom below 0 dBFS and computes/applies the maximum safe
/// gain so that the output peak does not exceed `−target_headroom_db` dBFS.
#[derive(Debug, Clone)]
pub struct HeadroomManager {
    /// Target headroom below 0 dBFS (positive value, e.g. 1.0 for −1 dBFS).
    target_headroom_db: f32,
}

impl HeadroomManager {
    /// Create a new manager with the given target headroom.
    ///
    /// `target_headroom_db` should be positive (e.g. `1.0` for −1 dBFS).
    /// Values ≤ 0 are clamped to 0 (no headroom enforcement).
    #[must_use]
    pub fn new(target_headroom_db: f32) -> Self {
        Self {
            target_headroom_db: target_headroom_db.max(0.0),
        }
    }

    /// Return the configured target headroom in dB.
    #[must_use]
    pub fn target_headroom_db(&self) -> f32 {
        self.target_headroom_db
    }

    /// Calculate the maximum gain (in dB) that keeps the peak within the
    /// allowed ceiling.
    ///
    /// `measured_peak_db` is the current true-peak level of the signal in dBFS
    /// (typically ≤ 0).  The returned gain may be negative if the signal is
    /// already above the headroom ceiling.
    ///
    /// ```
    /// use oximedia_normalize::headroom::HeadroomManager;
    ///
    /// let mgr = HeadroomManager::new(1.0); // ceiling = -1 dBFS
    /// // Current peak at -3 dBFS → can add up to 2 dB safely
    /// assert!((mgr.max_safe_gain_db(-3.0) - 2.0).abs() < 0.001);
    /// ```
    #[must_use]
    pub fn max_safe_gain_db(&self, measured_peak_db: f32) -> f32 {
        let ceiling_db = -self.target_headroom_db;
        ceiling_db - measured_peak_db
    }

    /// Apply the maximum safe gain to `samples` based on `measured_peak_db`.
    ///
    /// The applied gain is `max_safe_gain_db(measured_peak_db)` **but at most
    /// 60 dB** to guard against extreme boosts on silence or near-silence
    /// signals.  Negative gain (attenuation) is always applied without limit.
    ///
    /// Samples are multiplied by the linear equivalent of the gain in dB.
    ///
    /// # Arguments
    ///
    /// * `samples`          — mutable slice of linear amplitude values.
    /// * `measured_peak_db` — current true-peak level of the signal in dBFS.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_normalize::headroom::HeadroomManager;
    ///
    /// let mgr = HeadroomManager::new(0.0); // no headroom → gain brings peak to 0 dBFS
    /// let mut samples = vec![0.5f32; 4]; // peak = -6.02 dBFS
    /// mgr.apply_gain(&mut samples, -6.0);
    /// // Expected linear gain ≈ 2× → samples near 1.0
    /// assert!(samples[0] > 0.9 && samples[0] <= 1.0 + f32::EPSILON);
    /// ```
    pub fn apply_gain(&self, samples: &mut Vec<f32>, measured_peak_db: f32) {
        let gain_db = self.max_safe_gain_db(measured_peak_db).min(60.0);
        let linear_gain = db_to_linear(gain_db);
        for s in samples.iter_mut() {
            *s *= linear_gain;
        }
    }

    /// Apply the maximum safe gain and clamp all samples to [−1, 1].
    ///
    /// This is a safe version of [`apply_gain`](Self::apply_gain) that prevents
    /// inter-sample clipping at the cost of hard-clipping peaks.
    pub fn apply_gain_clamped(&self, samples: &mut Vec<f32>, measured_peak_db: f32) {
        self.apply_gain(samples, measured_peak_db);
        for s in samples.iter_mut() {
            *s = s.clamp(-1.0, 1.0);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a dB value to a linear amplitude multiplier.
#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_safe_gain_basic() {
        let mgr = HeadroomManager::new(1.0); // ceiling = -1 dBFS
        // peak at -6, ceiling at -1 → max gain = 5 dB
        let g = mgr.max_safe_gain_db(-6.0);
        assert!((g - 5.0).abs() < 0.001, "g={g}");
    }

    #[test]
    fn test_max_safe_gain_over_ceiling() {
        let mgr = HeadroomManager::new(3.0); // ceiling = -3 dBFS
        // peak at -1 dBFS → already above ceiling, gain = -2 dB (attenuation)
        let g = mgr.max_safe_gain_db(-1.0);
        assert!((g - (-2.0)).abs() < 0.001, "g={g}");
    }

    #[test]
    fn test_apply_gain_amplifies() {
        let mgr = HeadroomManager::new(0.0);
        let mut samples = vec![0.5f32; 4];
        mgr.apply_gain(&mut samples, -6.021); // peak ≈ -6 dBFS → gain ≈ +6 dB → ~2×
        for &s in &samples {
            assert!(s > 0.9 && s <= 1.001, "s={s}");
        }
    }

    #[test]
    fn test_apply_gain_attenuates_over_ceiling() {
        let mgr = HeadroomManager::new(6.0); // ceiling = -6 dBFS
        // peak already at 0 dBFS → should attenuate by -6 dB
        let mut samples = vec![1.0f32; 4];
        mgr.apply_gain(&mut samples, 0.0);
        let expected = db_to_linear(-6.0); // ≈ 0.501
        for &s in &samples {
            assert!((s - expected).abs() < 0.001, "s={s}");
        }
    }

    #[test]
    fn test_apply_gain_clamped_no_overflow() {
        let mgr = HeadroomManager::new(0.0);
        let mut samples = vec![0.9f32; 4];
        // Request large boost — clamped version must not exceed ±1
        mgr.apply_gain_clamped(&mut samples, -40.0);
        for &s in &samples {
            assert!(s <= 1.0 + f32::EPSILON, "s={s}");
        }
    }

    #[test]
    fn test_zero_headroom_is_valid() {
        let mgr = HeadroomManager::new(0.0);
        assert_eq!(mgr.target_headroom_db(), 0.0);
    }

    #[test]
    fn test_negative_headroom_clamped_to_zero() {
        let mgr = HeadroomManager::new(-5.0);
        assert_eq!(mgr.target_headroom_db(), 0.0);
    }

    #[test]
    fn test_db_to_linear() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-6);
        assert!((db_to_linear(20.0) - 10.0).abs() < 0.001);
        assert!((db_to_linear(-20.0) - 0.1).abs() < 0.001);
    }
}
