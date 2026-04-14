//! Distance attenuation models for spatial audio rendering.
//!
//! When a virtual sound source is placed at a distance from the listener,
//! its amplitude must be reduced to create the illusion of distance.  This
//! module provides several physically and perceptually motivated attenuation
//! models commonly used in spatial audio, game engines, and virtual production:
//!
//! | Model              | Description                                         |
//! |--------------------|-----------------------------------------------------|
//! | `InverseSquare`    | Point-source radiation (physically accurate)        |
//! | `Logarithmic`      | Perceptual distance curve (matches human perception)|
//! | `Linear`           | Simple proportional fade (simple fade-to-silence)   |
//! | `Exponential`      | Aggressive rolloff for enclosed spaces              |
//! | `CustomCurve`      | Arbitrary user-defined piecewise-linear curve       |
//!
//! # Reference distance
//!
//! All models define a `reference_distance` at which gain = 1.0.  Distances
//! closer than `reference_distance` can optionally be clamped or allowed to
//! boost above unity gain (controlled by the `max_gain` field).
//!
//! # Distance rolloff in practice
//!
//! Dolby Atmos and other object-audio renderers typically use a combination
//! of the inverse-square law for mid-to-far field and a proximity effect for
//! near-field sources.  The [`DistanceAttenuator`] struct applies a chosen
//! model and returns a linear gain scalar suitable for direct multiplication
//! with the audio sample buffer.
//!
//! # References
//! - OpenAL Programmer's Guide §3.4.2 — Distance Attenuation Models
//! - ITU-R BS.1116-3 — Subjective assessment of small impairments in audio
//! - Begault, D.R. (1994). *3-D Sound for Virtual Reality and Multimedia*.

use crate::SpatialError;

// ─── Attenuation model ────────────────────────────────────────────────────────

/// Piecewise-linear attenuation curve node `(distance_m, gain)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveNode {
    /// Distance from listener in metres.
    pub distance_m: f32,
    /// Linear gain at this distance.
    pub gain: f32,
}

/// Distance-based attenuation model.
#[derive(Debug, Clone)]
pub enum AttenuationModel {
    /// Physically accurate inverse-square law: `gain = (ref_dist / dist)^2`.
    ///
    /// - `reference_distance` — distance at which gain = 1.0.
    /// - `max_distance` — distances beyond this are attenuated to `rolloff_factor`
    ///   gain (not clamped to zero by default).
    /// - `rolloff_factor` — scales the rate of attenuation (1.0 = standard).
    InverseSquare {
        /// Distance at which gain is 1.0 (metres).
        reference_distance: f32,
        /// Maximum effective distance (metres).  Beyond this, gain is floored at
        /// `(reference_distance / max_distance)^2 * rolloff_factor`.
        max_distance: f32,
        /// Scales the attenuation curve.
        rolloff_factor: f32,
    },

    /// Logarithmic attenuation: `gain = ref_dist / (ref_dist + rolloff * (dist - ref_dist))`.
    ///
    /// Matches the OpenAL `AL_INVERSE_DISTANCE` model.
    Logarithmic {
        /// Distance at which gain is 1.0 (metres).
        reference_distance: f32,
        /// Rolloff steepness.
        rolloff_factor: f32,
        /// Maximum effective distance (metres).
        max_distance: f32,
    },

    /// Linear rolloff: `gain = 1 - rolloff * (dist - ref_dist) / (max_dist - ref_dist)`.
    ///
    /// Matches the OpenAL `AL_LINEAR_DISTANCE` model.
    Linear {
        /// Distance at which gain is 1.0 (metres).
        reference_distance: f32,
        /// Maximum effective distance (metres); gain = 0 at this point.
        max_distance: f32,
        /// Rolloff scale.
        rolloff_factor: f32,
    },

    /// Exponential rolloff: `gain = (dist / ref_dist)^(-rolloff)`.
    ///
    /// Matches the OpenAL `AL_EXPONENT_DISTANCE` model.
    Exponential {
        /// Distance at which gain is 1.0 (metres).
        reference_distance: f32,
        /// Exponent/rolloff factor.
        rolloff_factor: f32,
        /// Maximum effective distance (metres).
        max_distance: f32,
    },

    /// User-defined piecewise-linear curve.
    ///
    /// The curve is defined by a list of [`CurveNode`] pairs `(distance, gain)`
    /// sorted by distance.  Distances outside the defined range are clamped to
    /// the first/last node value.
    CustomCurve {
        /// Curve nodes, sorted by `distance_m` ascending.
        nodes: Vec<CurveNode>,
    },
}

impl AttenuationModel {
    /// Create a standard inverse-square model with `reference_distance = 1 m`.
    pub fn inverse_square_default() -> Self {
        Self::InverseSquare {
            reference_distance: 1.0,
            max_distance: 1000.0,
            rolloff_factor: 1.0,
        }
    }

    /// Create a logarithmic model with `reference_distance = 1 m`.
    pub fn logarithmic_default() -> Self {
        Self::Logarithmic {
            reference_distance: 1.0,
            rolloff_factor: 1.0,
            max_distance: 100.0,
        }
    }

    /// Create a linear model with `reference_distance = 1 m`, `max_distance = 50 m`.
    pub fn linear_default() -> Self {
        Self::Linear {
            reference_distance: 1.0,
            max_distance: 50.0,
            rolloff_factor: 1.0,
        }
    }

    /// Create an exponential model with `reference_distance = 1 m`, rolloff = 1.0.
    pub fn exponential_default() -> Self {
        Self::Exponential {
            reference_distance: 1.0,
            rolloff_factor: 1.0,
            max_distance: 100.0,
        }
    }

    /// Compute the linear gain for the given `distance_m`.
    ///
    /// Returns an error if the custom curve contains no nodes.
    pub fn gain_at(&self, distance_m: f32) -> Result<f32, SpatialError> {
        match self {
            Self::InverseSquare { reference_distance, max_distance, rolloff_factor } => {
                let ref_d = reference_distance.max(1e-6_f32);
                let d = distance_m.max(ref_d).min(*max_distance);
                Ok((ref_d / d).powi(2) * rolloff_factor)
            }

            Self::Logarithmic { reference_distance, rolloff_factor, max_distance } => {
                let ref_d = reference_distance.max(1e-6_f32);
                let d = distance_m.max(ref_d).min(*max_distance);
                let denom = ref_d + rolloff_factor * (d - ref_d);
                if denom.abs() < 1e-9 {
                    Ok(1.0)
                } else {
                    Ok((ref_d / denom).max(0.0))
                }
            }

            Self::Linear { reference_distance, max_distance, rolloff_factor } => {
                let ref_d = reference_distance.max(1e-6_f32);
                let max_d = max_distance.max(ref_d + 1e-6);
                let d = distance_m.clamp(ref_d, max_d);
                let gain = 1.0 - rolloff_factor * (d - ref_d) / (max_d - ref_d);
                Ok(gain.clamp(0.0, 1.0))
            }

            Self::Exponential { reference_distance, rolloff_factor, max_distance } => {
                let ref_d = reference_distance.max(1e-6_f32);
                let d = distance_m.max(ref_d).min(*max_distance);
                Ok((d / ref_d).powf(-rolloff_factor).max(0.0))
            }

            Self::CustomCurve { nodes } => {
                if nodes.is_empty() {
                    return Err(SpatialError::InvalidConfig(
                        "custom attenuation curve has no nodes".to_string(),
                    ));
                }
                if nodes.len() == 1 {
                    return Ok(nodes[0].gain);
                }
                // Find the two nodes that bracket `distance_m`.
                if distance_m <= nodes[0].distance_m {
                    return Ok(nodes[0].gain);
                }
                if distance_m >= nodes[nodes.len() - 1].distance_m {
                    return Ok(nodes[nodes.len() - 1].gain);
                }
                for i in 1..nodes.len() {
                    let a = &nodes[i - 1];
                    let b = &nodes[i];
                    if distance_m <= b.distance_m {
                        let span = b.distance_m - a.distance_m;
                        if span.abs() < 1e-9 {
                            return Ok(a.gain);
                        }
                        let t = (distance_m - a.distance_m) / span;
                        return Ok(a.gain + t * (b.gain - a.gain));
                    }
                }
                Ok(nodes[nodes.len() - 1].gain)
            }
        }
    }
}

// ─── Attenuator ───────────────────────────────────────────────────────────────

/// Configurable distance-attenuation processor.
///
/// Wraps an [`AttenuationModel`] and applies gain scaling to audio buffers.
///
/// # Example
///
/// ```rust
/// use oximedia_spatial::distance_attenuation::{DistanceAttenuator, AttenuationModel};
///
/// let attenuator = DistanceAttenuator::new(AttenuationModel::inverse_square_default())
///     .with_max_gain(1.0)
///     .with_min_gain(0.0);
///
/// let signal = vec![1.0_f32; 256];
/// let result = attenuator.apply(10.0, &signal).unwrap();
/// assert_eq!(result.len(), 256);
/// // At 10 m with reference 1 m, gain ≈ 0.01
/// assert!(result[0] < 0.02);
/// ```
#[derive(Debug, Clone)]
pub struct DistanceAttenuator {
    /// The underlying attenuation model.
    pub model: AttenuationModel,
    /// Maximum allowed gain (clamps proximity amplification).  Default: 1.0.
    pub max_gain: f32,
    /// Minimum allowed gain (floor).  Default: 0.0.
    pub min_gain: f32,
}

impl DistanceAttenuator {
    /// Create a new attenuator with the supplied model.
    pub fn new(model: AttenuationModel) -> Self {
        Self { model, max_gain: 1.0, min_gain: 0.0 }
    }

    /// Set the maximum gain (builder pattern).
    pub fn with_max_gain(mut self, max_gain: f32) -> Self {
        self.max_gain = max_gain.max(0.0);
        self
    }

    /// Set the minimum gain (builder pattern).
    pub fn with_min_gain(mut self, min_gain: f32) -> Self {
        self.min_gain = min_gain.clamp(0.0, 1.0);
        self
    }

    /// Compute the attenuation gain at the given distance.
    pub fn gain_at(&self, distance_m: f32) -> Result<f32, SpatialError> {
        let raw = self.model.gain_at(distance_m)?;
        Ok(raw.clamp(self.min_gain, self.max_gain))
    }

    /// Apply distance attenuation to a mono audio buffer in-place (by value).
    ///
    /// Returns the attenuated buffer.
    pub fn apply(&self, distance_m: f32, samples: &[f32]) -> Result<Vec<f32>, SpatialError> {
        let gain = self.gain_at(distance_m)?;
        Ok(samples.iter().map(|&s| s * gain).collect())
    }

    /// Apply attenuation in-place, mutating the supplied buffer.
    pub fn apply_inplace(&self, distance_m: f32, samples: &mut [f32]) -> Result<(), SpatialError> {
        let gain = self.gain_at(distance_m)?;
        for s in samples.iter_mut() {
            *s *= gain;
        }
        Ok(())
    }
}

// ─── Multi-source distance renderer ───────────────────────────────────────────

/// A positioned audio source for distance rendering.
#[derive(Debug, Clone)]
pub struct PositionedSource {
    /// Source identifier.
    pub id: u32,
    /// Mono audio signal.
    pub signal: Vec<f32>,
    /// World-space position `[x, y, z]` in metres.
    pub position: [f32; 3],
    /// Source-level gain (applied before distance attenuation).
    pub gain: f32,
}

impl PositionedSource {
    /// Compute the Euclidean distance from this source to the given listener position.
    pub fn distance_to(&self, listener: [f32; 3]) -> f32 {
        let dx = self.position[0] - listener[0];
        let dy = self.position[1] - listener[1];
        let dz = self.position[2] - listener[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Render multiple positioned sources to a mono mix with distance attenuation.
///
/// Each source's signal is scaled by its source gain and the distance-based
/// attenuation gain, then summed into the output buffer.
///
/// # Arguments
/// * `sources` — list of positioned sources to mix.
/// * `attenuator` — distance attenuation processor.
/// * `listener_pos` — listener world-space position `[x, y, z]`.
/// * `num_samples` — length of the output buffer.
pub fn render_mix(
    sources: &[PositionedSource],
    attenuator: &DistanceAttenuator,
    listener_pos: [f32; 3],
    num_samples: usize,
) -> Result<Vec<f32>, SpatialError> {
    let mut out = vec![0.0_f32; num_samples];
    for src in sources {
        let dist = src.distance_to(listener_pos).max(1e-9);
        let dist_gain = attenuator.gain_at(dist)?;
        let effective_gain = src.gain * dist_gain;
        let copy_len = src.signal.len().min(num_samples);
        for i in 0..copy_len {
            out[i] += src.signal[i] * effective_gain;
        }
    }
    Ok(out)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Model gain calculations ──────────────────────────────────────────────

    #[test]
    fn test_inverse_square_at_reference_distance() {
        let model = AttenuationModel::inverse_square_default();
        let gain = model.gain_at(1.0).expect("gain should compute");
        assert!((gain - 1.0).abs() < 1e-5, "gain={gain}");
    }

    #[test]
    fn test_inverse_square_doubles_halves_gain() {
        let model = AttenuationModel::InverseSquare {
            reference_distance: 1.0,
            max_distance: 1000.0,
            rolloff_factor: 1.0,
        };
        let g1 = model.gain_at(2.0).expect("ok");
        let g2 = model.gain_at(4.0).expect("ok");
        // Doubling the distance quarters the gain.
        assert!((g1 - 0.25).abs() < 1e-5, "g1={g1}");
        assert!((g2 - 0.0625).abs() < 1e-5, "g2={g2}");
    }

    #[test]
    fn test_logarithmic_at_reference_distance() {
        let model = AttenuationModel::logarithmic_default();
        let gain = model.gain_at(1.0).expect("ok");
        assert!((gain - 1.0).abs() < 1e-5, "gain={gain}");
    }

    #[test]
    fn test_logarithmic_decreases_with_distance() {
        let model = AttenuationModel::logarithmic_default();
        let g1 = model.gain_at(5.0).expect("ok");
        let g2 = model.gain_at(20.0).expect("ok");
        assert!(g1 > g2, "expected g(5)>g(20), got {g1} vs {g2}");
    }

    #[test]
    fn test_linear_at_reference_distance() {
        let model = AttenuationModel::linear_default();
        let gain = model.gain_at(1.0).expect("ok");
        assert!((gain - 1.0).abs() < 1e-5, "gain={gain}");
    }

    #[test]
    fn test_linear_at_max_distance_is_zero() {
        let model = AttenuationModel::Linear {
            reference_distance: 1.0,
            max_distance: 50.0,
            rolloff_factor: 1.0,
        };
        let gain = model.gain_at(50.0).expect("ok");
        assert!(gain < 0.01, "gain at max_distance should be ~0, got {gain}");
    }

    #[test]
    fn test_exponential_at_reference_distance() {
        let model = AttenuationModel::exponential_default();
        let gain = model.gain_at(1.0).expect("ok");
        assert!((gain - 1.0).abs() < 1e-5, "gain={gain}");
    }

    #[test]
    fn test_exponential_decreases_with_distance() {
        let model = AttenuationModel::exponential_default();
        let g1 = model.gain_at(2.0).expect("ok");
        let g2 = model.gain_at(10.0).expect("ok");
        assert!(g1 > g2, "expected g(2)>g(10), got {g1} vs {g2}");
    }

    #[test]
    fn test_custom_curve_interpolation() {
        let model = AttenuationModel::CustomCurve {
            nodes: vec![
                CurveNode { distance_m: 1.0, gain: 1.0 },
                CurveNode { distance_m: 10.0, gain: 0.0 },
            ],
        };
        let g = model.gain_at(5.5).expect("ok");
        // Linear interpolation between (1,1.0) and (10,0.0) at distance 5.5
        // t = (5.5 - 1.0) / (10.0 - 1.0) = 4.5 / 9.0 = 0.5
        assert!((g - 0.5).abs() < 1e-5, "g={g}");
    }

    #[test]
    fn test_custom_curve_empty_returns_error() {
        let model = AttenuationModel::CustomCurve { nodes: vec![] };
        assert!(model.gain_at(5.0).is_err());
    }

    #[test]
    fn test_custom_curve_below_range_clamps_to_first() {
        let model = AttenuationModel::CustomCurve {
            nodes: vec![
                CurveNode { distance_m: 2.0, gain: 0.8 },
                CurveNode { distance_m: 10.0, gain: 0.1 },
            ],
        };
        let g = model.gain_at(0.5).expect("ok");
        assert!((g - 0.8).abs() < 1e-5, "g={g}");
    }

    // ── Attenuator ──────────────────────────────────────────────────────────

    #[test]
    fn test_attenuator_apply_scales_buffer() {
        let att = DistanceAttenuator::new(AttenuationModel::inverse_square_default());
        let signal: Vec<f32> = vec![1.0; 64];
        let out = att.apply(2.0, &signal).expect("ok");
        // At 2 m, inverse-square gives 0.25.
        assert_eq!(out.len(), 64);
        assert!((out[0] - 0.25).abs() < 1e-5, "out[0]={}", out[0]);
    }

    #[test]
    fn test_attenuator_apply_inplace() {
        let att = DistanceAttenuator::new(AttenuationModel::inverse_square_default());
        let mut signal: Vec<f32> = vec![1.0; 32];
        att.apply_inplace(2.0, &mut signal).expect("ok");
        assert!((signal[0] - 0.25).abs() < 1e-5, "signal[0]={}", signal[0]);
    }

    #[test]
    fn test_attenuator_max_gain_clamp() {
        let att = DistanceAttenuator::new(AttenuationModel::inverse_square_default())
            .with_max_gain(0.5);
        // At reference distance gain would be 1.0, but max_gain=0.5 clamps it.
        let g = att.gain_at(1.0).expect("ok");
        assert!((g - 0.5).abs() < 1e-5, "g={g}");
    }

    // ── render_mix ──────────────────────────────────────────────────────────

    #[test]
    fn test_render_mix_two_sources() {
        let att = DistanceAttenuator::new(AttenuationModel::inverse_square_default());
        let sources = vec![
            PositionedSource {
                id: 0,
                signal: vec![1.0; 128],
                position: [1.0, 0.0, 0.0],
                gain: 1.0,
            },
            PositionedSource {
                id: 1,
                signal: vec![1.0; 128],
                position: [2.0, 0.0, 0.0],
                gain: 1.0,
            },
        ];
        let out = render_mix(&sources, &att, [0.0, 0.0, 0.0], 128).expect("ok");
        assert_eq!(out.len(), 128);
        // Source at 1 m: gain=1.0, source at 2 m: gain=0.25 → sum=1.25
        assert!((out[0] - 1.25).abs() < 1e-4, "out[0]={}", out[0]);
    }

    #[test]
    fn test_positioned_source_distance() {
        let src = PositionedSource {
            id: 0,
            signal: vec![],
            position: [3.0, 4.0, 0.0],
            gain: 1.0,
        };
        let d = src.distance_to([0.0, 0.0, 0.0]);
        assert!((d - 5.0).abs() < 1e-5, "d={d}");
    }
}
