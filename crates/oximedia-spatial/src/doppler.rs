//! Doppler effect simulation for spatial audio.
//!
//! This module provides frequency-shift calculations based on the classic
//! Doppler formula, along with 3-D source/listener velocity helpers for
//! use in game audio and spatial rendering pipelines.
//!
//! # Formula
//!
//! ```text
//! f_observed = f_source × (c + v_listener) / (c − v_source)
//! ```
//!
//! where `c` is the speed of sound (343 m/s at 20 °C), `v_listener` is the
//! component of the listener's velocity **toward** the source, and `v_source`
//! is the component of the source's velocity **toward** the listener.  Both
//! velocities must be strictly less than `c` to be physically meaningful; this
//! implementation clamps them to `c − 0.001`.

// ─── Constants ────────────────────────────────────────────────────────────────

/// Speed of sound in dry air at 20 °C (m/s).
pub const SPEED_OF_SOUND_MS: f32 = 343.0;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for Doppler-effect calculations.
#[derive(Debug, Clone)]
pub struct DopplerConfig {
    /// Speed of sound (m/s). Default: 343.0.
    pub speed_of_sound: f32,
    /// Maximum allowed velocity magnitude (m/s). Velocities are clamped to
    /// this value before applying the Doppler formula. Default: 100.0.
    pub max_velocity: f32,
}

impl DopplerConfig {
    /// Create a configuration with default values (c = 343 m/s, max v = 100 m/s).
    pub fn new() -> Self {
        Self {
            speed_of_sound: SPEED_OF_SOUND_MS,
            max_velocity: 100.0,
        }
    }

    /// Override the speed of sound (builder-style).
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed_of_sound = speed.max(1.0); // guard against nonsensical values
        self
    }

    /// Override the maximum velocity clamp (builder-style).
    pub fn with_max_velocity(mut self, max_v: f32) -> Self {
        self.max_velocity = max_v.max(0.0);
        self
    }
}

impl Default for DopplerConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Core computation ─────────────────────────────────────────────────────────

/// Compute the observed frequency given source and listener velocities.
///
/// Both `v_source` and `v_listener` are **toward-each-other** components
/// (positive = approaching).  The result is clamped to the range
/// `[source_freq × 0.1, source_freq × 10.0]`.
///
/// # Arguments
/// * `source_freq_hz` – nominal frequency of the source (Hz).
/// * `v_source` – speed of the source toward the listener (m/s).
/// * `v_listener` – speed of the listener toward the source (m/s).
/// * `config` – Doppler configuration (speed of sound, velocity clamp).
pub fn doppler_frequency(
    source_freq_hz: f32,
    v_source: f32,
    v_listener: f32,
    config: &DopplerConfig,
) -> f32 {
    let c = config.speed_of_sound;
    let max_v = config.max_velocity.min(c - 0.001);

    // Clamp to physically valid range.
    let vs = v_source.clamp(-max_v, max_v);
    let vl = v_listener.clamp(-max_v, max_v);

    // Ensure denominators stay positive (sub-sonic sources only).
    let denominator = (c - vs).max(0.001);
    let numerator = (c + vl).max(0.001);

    let f_obs = source_freq_hz * numerator / denominator;

    // Clamp to a sensible output range.
    f_obs.clamp(source_freq_hz * 0.1, source_freq_hz * 10.0)
}

// ─── 3-D sound source ─────────────────────────────────────────────────────────

/// A sound source with a 3-D position, velocity, and a nominal frequency.
#[derive(Debug, Clone, Default)]
pub struct SoundSource {
    /// World-space position in metres (x, y, z).
    pub position: (f32, f32, f32),
    /// World-space velocity in m/s (vx, vy, vz).
    pub velocity: (f32, f32, f32),
    /// Nominal (rest-frame) frequency in Hz.
    pub frequency_hz: f32,
    /// Amplitude in the range 0.0–1.0.
    pub amplitude: f32,
}

impl SoundSource {
    /// Create a source at the given position with the given frequency.
    pub fn new(x: f32, y: f32, z: f32, freq_hz: f32) -> Self {
        Self {
            position: (x, y, z),
            velocity: (0.0, 0.0, 0.0),
            frequency_hz: freq_hz,
            amplitude: 1.0,
        }
    }

    /// Compute the component of `velocity` that is directed **toward** `listener_pos`.
    ///
    /// Returns a positive value when the source is moving toward the listener,
    /// and negative when moving away.
    pub fn velocity_toward(
        source_pos: (f32, f32, f32),
        listener_pos: (f32, f32, f32),
        velocity: (f32, f32, f32),
    ) -> f32 {
        let dx = listener_pos.0 - source_pos.0;
        let dy = listener_pos.1 - source_pos.1;
        let dz = listener_pos.2 - source_pos.2;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist < 1e-6 {
            return 0.0;
        }
        // Unit vector from source to listener.
        let ux = dx / dist;
        let uy = dy / dist;
        let uz = dz / dist;
        // Dot product: component of velocity in the source→listener direction.
        velocity.0 * ux + velocity.1 * uy + velocity.2 * uz
    }
}

// ─── 3-D Doppler ──────────────────────────────────────────────────────────────

/// Compute the observed frequency for a 3-D moving source and listener.
///
/// Extracts the velocity components along the source-listener axis and applies
/// the Doppler formula.
pub fn doppler_3d(
    source: &SoundSource,
    listener_pos: (f32, f32, f32),
    listener_velocity: (f32, f32, f32),
    config: &DopplerConfig,
) -> f32 {
    let v_source = SoundSource::velocity_toward(source.position, listener_pos, source.velocity);

    // For the listener component we want the speed toward the source.
    let v_listener =
        SoundSource::velocity_toward(listener_pos, source.position, listener_velocity);

    doppler_frequency(source.frequency_hz, v_source, v_listener, config)
}

// ─── Pitch ratio helpers ──────────────────────────────────────────────────────

/// Return the pitch-shift ratio `observed / source`.
///
/// A ratio > 1.0 indicates a pitch increase (approaching), < 1.0 a decrease
/// (receding).  Use this value with a variable-rate resampler.
pub fn doppler_pitch_ratio(source_freq: f32, observed_freq: f32) -> f32 {
    if source_freq.abs() < 1e-9 {
        return 1.0;
    }
    observed_freq / source_freq
}

/// Compute the Doppler pitch ratio for a 3-D source/listener pair.
///
/// This is a convenience wrapper that calls [`doppler_3d`] and then
/// [`doppler_pitch_ratio`].  The returned ratio is suitable for passing
/// directly to a variable-rate resampler.
pub fn compute_doppler_ratio(
    source: &SoundSource,
    listener_pos: (f32, f32, f32),
    listener_velocity: (f32, f32, f32),
    config: &DopplerConfig,
) -> f32 {
    let observed = doppler_3d(source, listener_pos, listener_velocity, config);
    doppler_pitch_ratio(source.frequency_hz, observed)
}

// ─── DopplerEffect ───────────────────────────────────────────────────────────

/// High-level Doppler effect processor.
///
/// Wraps [`DopplerConfig`] and provides a simple one-line API for computing
/// the observed frequency given source and listener approach velocities.
///
/// # Example
///
/// ```rust
/// use oximedia_spatial::doppler::DopplerEffect;
///
/// let effect = DopplerEffect::new(343.0);
/// // Stationary source and listener → observed == source.
/// let f = effect.frequency_shift(0.0, 0.0, 440.0);
/// assert!((f - 440.0).abs() < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct DopplerEffect {
    config: DopplerConfig,
}

impl DopplerEffect {
    /// Create a Doppler effect processor with the given speed of sound.
    ///
    /// # Arguments
    ///
    /// * `speed_of_sound` — Speed of sound in m/s (clamped to ≥ 1.0).
    #[must_use]
    pub fn new(speed_of_sound: f32) -> Self {
        Self {
            config: DopplerConfig::new().with_speed(speed_of_sound),
        }
    }

    /// Compute the Doppler-shifted observed frequency.
    ///
    /// # Arguments
    ///
    /// * `source_vel`   — Speed of the source **toward** the listener (m/s).
    ///   Positive = approaching, negative = receding.
    /// * `listener_vel` — Speed of the listener **toward** the source (m/s).
    ///   Positive = approaching, negative = receding.
    /// * `base_freq`    — Rest-frame frequency of the source (Hz).
    ///
    /// # Returns
    ///
    /// Observed frequency in Hz, clamped to `[base_freq × 0.1, base_freq × 10]`.
    #[must_use]
    pub fn frequency_shift(&self, source_vel: f32, listener_vel: f32, base_freq: f32) -> f32 {
        doppler_frequency(base_freq, source_vel, listener_vel, &self.config)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DopplerConfig {
        DopplerConfig::new()
    }

    // Helpers ----------------------------------------------------------------

    fn assert_near(a: f32, b: f32, tol: f32, label: &str) {
        assert!(
            (a - b).abs() < tol,
            "{label}: expected ≈ {b:.4}, got {a:.4} (tol {tol})"
        );
    }

    // ── DopplerConfig ────────────────────────────────────────────────────────

    #[test]
    fn test_config_default_speed_of_sound() {
        let c = DopplerConfig::default();
        assert_near(c.speed_of_sound, SPEED_OF_SOUND_MS, 1e-3, "default c");
    }

    #[test]
    fn test_config_with_speed_builder() {
        let c = DopplerConfig::new().with_speed(300.0);
        assert_near(c.speed_of_sound, 300.0, 1e-3, "custom speed");
    }

    // ── doppler_frequency ────────────────────────────────────────────────────

    #[test]
    fn test_doppler_static_source_returns_source_freq() {
        // Neither source nor listener moving → observed == source.
        let f = doppler_frequency(440.0, 0.0, 0.0, &cfg());
        assert_near(f, 440.0, 0.5, "stationary");
    }

    #[test]
    fn test_doppler_approaching_at_half_speed_of_sound() {
        // Source approaching at c/2 → f_obs = f * c / (c - c/2) = f * 2
        // Use a high max_velocity config so the clamp does not interfere.
        let c = SPEED_OF_SOUND_MS;
        let high_v_cfg = DopplerConfig::new().with_max_velocity(c - 1.0);
        let f = doppler_frequency(440.0, c / 2.0, 0.0, &high_v_cfg);
        assert_near(f, 880.0, 5.0, "approaching at c/2");
    }

    #[test]
    fn test_doppler_receding_reduces_frequency() {
        // Source receding (negative v_source) → observed < source.
        let f = doppler_frequency(440.0, -50.0, 0.0, &cfg());
        assert!(f < 440.0, "receding source must lower frequency, got {f}");
    }

    #[test]
    fn test_doppler_listener_approaching_raises_frequency() {
        // Listener moving toward source → observed > source.
        let f = doppler_frequency(440.0, 0.0, 50.0, &cfg());
        assert!(f > 440.0, "approaching listener must raise frequency, got {f}");
    }

    // ── SoundSource::velocity_toward ─────────────────────────────────────────

    #[test]
    fn test_velocity_toward_along_axis() {
        // Source at origin, listener at (10, 0, 0), velocity = (5, 0, 0).
        let v = SoundSource::velocity_toward((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (5.0, 0.0, 0.0));
        assert_near(v, 5.0, 1e-4, "velocity along axis toward listener");
    }

    #[test]
    fn test_velocity_toward_perpendicular_is_zero() {
        // Source at origin, listener at (10, 0, 0), velocity = (0, 5, 0) (perpendicular).
        let v = SoundSource::velocity_toward((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 5.0, 0.0));
        assert_near(v, 0.0, 1e-4, "perpendicular velocity contributes nothing");
    }

    #[test]
    fn test_velocity_toward_coincident_positions_zero() {
        // Source and listener at same position → no well-defined axis → 0.
        let v = SoundSource::velocity_toward((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (5.0, 0.0, 0.0));
        assert_near(v, 0.0, 1e-4, "coincident positions give zero");
    }

    // ── doppler_3d ───────────────────────────────────────────────────────────

    #[test]
    fn test_doppler_3d_stationary_source() {
        let src = SoundSource::new(0.0, 0.0, 0.0, 440.0);
        let listener_pos = (10.0, 0.0, 0.0);
        let f = doppler_3d(&src, listener_pos, (0.0, 0.0, 0.0), &cfg());
        assert_near(f, 440.0, 0.5, "stationary 3D source");
    }

    // ── doppler_pitch_ratio ───────────────────────────────────────────────────

    #[test]
    fn test_pitch_ratio_equal_frequencies() {
        let ratio = doppler_pitch_ratio(440.0, 440.0);
        assert_near(ratio, 1.0, 1e-4, "same freq → ratio 1");
    }

    #[test]
    fn test_pitch_ratio_double_frequency() {
        let ratio = doppler_pitch_ratio(440.0, 880.0);
        assert_near(ratio, 2.0, 1e-4, "double freq → ratio 2");
    }

    // ── clamping ─────────────────────────────────────────────────────────────

    #[test]
    fn test_doppler_clamped_at_high_velocity() {
        // Velocity >> max_velocity → result should be clamped, not NaN/inf.
        let f = doppler_frequency(440.0, 1000.0, 0.0, &cfg());
        assert!(
            f.is_finite(),
            "high velocity must not produce NaN/inf, got {f}"
        );
        assert!(f <= 440.0 * 10.0, "result must be ≤ 10x source freq");
    }

    // ── compute_doppler_ratio ─────────────────────────────────────────────────

    #[test]
    fn test_compute_doppler_ratio_stationary_is_one() {
        let src = SoundSource::new(0.0, 0.0, 0.0, 440.0);
        let ratio = compute_doppler_ratio(&src, (10.0, 0.0, 0.0), (0.0, 0.0, 0.0), &cfg());
        assert_near(ratio, 1.0, 0.01, "stationary ratio ≈ 1");
    }

    // ── DopplerEffect ────────────────────────────────────────────────────────

    #[test]
    fn test_doppler_effect_new_and_stationary() {
        let effect = DopplerEffect::new(343.0);
        let f = effect.frequency_shift(0.0, 0.0, 440.0);
        assert_near(f, 440.0, 0.5, "DopplerEffect stationary");
    }

    #[test]
    fn test_doppler_effect_source_approaching() {
        let effect = DopplerEffect::new(343.0);
        let f = effect.frequency_shift(50.0, 0.0, 440.0);
        assert!(f > 440.0, "Approaching source raises frequency, got {f}");
    }

    #[test]
    fn test_doppler_effect_source_receding() {
        let effect = DopplerEffect::new(343.0);
        let f = effect.frequency_shift(-50.0, 0.0, 440.0);
        assert!(f < 440.0, "Receding source lowers frequency, got {f}");
    }

    #[test]
    fn test_doppler_effect_listener_approaching() {
        let effect = DopplerEffect::new(343.0);
        let f = effect.frequency_shift(0.0, 50.0, 440.0);
        assert!(f > 440.0, "Approaching listener raises frequency, got {f}");
    }

    #[test]
    fn test_doppler_effect_custom_speed_of_sound() {
        let effect = DopplerEffect::new(1500.0); // underwater speed
        let f = effect.frequency_shift(0.0, 0.0, 1000.0);
        assert_near(f, 1000.0, 1.0, "Custom speed, stationary");
    }
}
