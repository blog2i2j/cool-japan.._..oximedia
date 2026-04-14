//! Spatial audio metadata and rendering hints for 360° video.
//!
//! This module provides utilities for describing and processing spatial audio
//! tracks that accompany 360° video content:
//!
//! * [`AmbisonicsMetadata`] — describes an ambisonics audio track (order,
//!   channel layout, normalisation convention, head-locked gain).
//! * [`AudioSpherePoint`] — a point-source audio object with position on the
//!   sphere and gain.
//! * [`AudioSphereMap`] — a collection of audio sphere points, supporting
//!   nearest-source lookup and gain-weighted centroid computation.
//! * [`HeadRotationRenderer`] — computes head-rotation-compensated rendering
//!   hints (virtual speaker directions) for first-order ambisonics (FOA) given
//!   a listener head orientation.
//! * [`BinauralHint`] — a rendering hint for a single virtual speaker,
//!   expressing the azimuth/elevation offsets from the listener's perspective.
//!
//! ## Ambisonics channel conventions
//!
//! The module supports **ACN** (Ambisonics Channel Number) ordering and both
//! **SN3D** (Schmidt seminormalised) and **N3D** (fully normalised) normalisation
//! conventions.  Decoding to virtual loudspeaker feeds for binaural rendering is
//! implemented for first-order ambisonics (four channels: W, X, Y, Z).
//!
//! ## Coordinate convention
//!
//! Sphere positions use the same `(yaw_rad, pitch_rad)` convention as the rest
//! of the crate: yaw ∈ `[−π, +π]`, pitch ∈ `[−π/2, +π/2]`.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_360::spatial_audio_360::{
//!     AmbisonicsMetadata, AmbisonicsOrder, NormalisationConvention,
//!     AudioSphereMap, AudioSpherePoint,
//! };
//!
//! let meta = AmbisonicsMetadata {
//!     order: AmbisonicsOrder::First,
//!     normalisation: NormalisationConvention::Sn3d,
//!     channel_count: 4,
//!     head_locked_stereo_gain: 0.0,
//!     description: String::from("FOA bed"),
//! };
//! assert_eq!(meta.expected_channel_count(), 4);
//!
//! let mut sphere_map = AudioSphereMap::new();
//! sphere_map.add(AudioSpherePoint { yaw_rad: 0.0, pitch_rad: 0.0, gain_db: -6.0, label: Some("narrator".into()) });
//! let nearest = sphere_map.nearest(0.1, 0.0);
//! println!("nearest: {:?}", nearest);
//! ```

use crate::VrError;
use std::f32::consts::PI;

// ─── AmbisonicsOrder ──────────────────────────────────────────────────────────

/// Ambisonics order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmbisonicsOrder {
    /// First-order ambisonics (4 channels: W, X, Y, Z).
    First,
    /// Second-order ambisonics (9 channels).
    Second,
    /// Third-order ambisonics (16 channels).
    Third,
    /// Fourth-order ambisonics (25 channels).
    Fourth,
}

impl AmbisonicsOrder {
    /// Return the number of ACN channels for this order: `(order + 1)²`.
    #[must_use]
    pub fn channel_count(self) -> u32 {
        match self {
            Self::First => 4,
            Self::Second => 9,
            Self::Third => 16,
            Self::Fourth => 25,
        }
    }

    /// Return the integer order value.
    #[must_use]
    pub fn order_value(self) -> u32 {
        match self {
            Self::First => 1,
            Self::Second => 2,
            Self::Third => 3,
            Self::Fourth => 4,
        }
    }
}

// ─── NormalisationConvention ──────────────────────────────────────────────────

/// Ambisonics normalisation convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormalisationConvention {
    /// Schmidt seminormalised (SN3D) — used in Google Spatial Audio, TBE.
    Sn3d,
    /// Fully normalised (N3D) — used in some other ambisonics workflows.
    N3d,
    /// Furse-Malham (FuMa) — legacy normalisation, first-order only.
    FuMa,
}

// ─── AmbisonicsMetadata ───────────────────────────────────────────────────────

/// Metadata describing an ambisonics audio track.
#[derive(Debug, Clone, PartialEq)]
pub struct AmbisonicsMetadata {
    /// Ambisonics order.
    pub order: AmbisonicsOrder,
    /// Normalisation convention.
    pub normalisation: NormalisationConvention,
    /// Actual channel count as declared in the container.
    pub channel_count: u32,
    /// Head-locked stereo loudness gain in dB (0 = disabled / full ambisonics,
    /// negative values reduce the non-directional stereo component).
    pub head_locked_stereo_gain: f32,
    /// Human-readable description of the audio bed.
    pub description: String,
}

impl AmbisonicsMetadata {
    /// Return the expected channel count based on the declared order.
    #[must_use]
    pub fn expected_channel_count(&self) -> u32 {
        self.order.channel_count()
    }

    /// Validate that the declared `channel_count` matches the expected count
    /// for the stated order.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::ParseError`] if the counts differ.
    pub fn validate(&self) -> Result<(), VrError> {
        let expected = self.expected_channel_count();
        if self.channel_count != expected {
            return Err(VrError::ParseError(format!(
                "ambisonics order {:?} expects {expected} channels, but channel_count={}",
                self.order, self.channel_count
            )));
        }
        Ok(())
    }

    /// Compute the SN3D-to-N3D conversion gain for ACN channel `acn`.
    ///
    /// `gain_n3d = gain_sn3d * sqrt(2*l + 1)` where `l` is the degree (order)
    /// for ACN index `acn`, computed as `l = floor(sqrt(acn))`.
    ///
    /// Returns 1.0 for FuMa (no automatic conversion defined here).
    #[must_use]
    pub fn sn3d_to_n3d_gain(acn: u32) -> f32 {
        let l = (acn as f32).sqrt().floor() as u32;
        ((2 * l + 1) as f32).sqrt()
    }

    /// Compute the N3D-to-SN3D conversion gain for ACN channel `acn`.
    #[must_use]
    pub fn n3d_to_sn3d_gain(acn: u32) -> f32 {
        let l = (acn as f32).sqrt().floor() as u32;
        let n3d_to_sn3d = 1.0 / ((2 * l + 1) as f32).sqrt();
        n3d_to_sn3d
    }
}

// ─── AudioSpherePoint ─────────────────────────────────────────────────────────

/// A point-source audio object positioned on the sphere.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioSpherePoint {
    /// Yaw (azimuth) in radians, `[−π, +π]`.
    pub yaw_rad: f32,
    /// Pitch (elevation) in radians, `[−π/2, +π/2]`.
    pub pitch_rad: f32,
    /// Source gain in dBFS.  0 dB = reference level.
    pub gain_db: f32,
    /// Optional label for identification.
    pub label: Option<String>,
}

impl AudioSpherePoint {
    /// Return the linear gain multiplier for this source.
    ///
    /// `linear = 10^(gain_db / 20)`
    #[must_use]
    pub fn linear_gain(&self) -> f32 {
        10.0_f32.powf(self.gain_db / 20.0)
    }

    /// Convert (yaw, pitch) to a unit Cartesian direction vector `[x, y, z]`.
    ///
    /// Convention: `+x` = right (East), `+y` = up (North Pole),
    /// `+z` = forward (centre of front face).
    #[must_use]
    pub fn to_cartesian(&self) -> [f32; 3] {
        let cp = self.pitch_rad.cos();
        [
            cp * self.yaw_rad.sin(),
            self.pitch_rad.sin(),
            cp * self.yaw_rad.cos(),
        ]
    }
}

// ─── AudioSphereMap ───────────────────────────────────────────────────────────

/// A collection of audio sphere points.
///
/// Provides spatial queries: nearest source, gain-weighted centroid, and
/// rendering-priority sorting.
#[derive(Debug, Clone, Default)]
pub struct AudioSphereMap {
    points: Vec<AudioSpherePoint>,
}

impl AudioSphereMap {
    /// Create an empty sphere map.
    #[must_use]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Add a point source.
    pub fn add(&mut self, point: AudioSpherePoint) {
        self.points.push(point);
    }

    /// Return the number of point sources.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Return `true` if there are no point sources.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Return a reference to all stored points.
    pub fn points(&self) -> &[AudioSpherePoint] {
        &self.points
    }

    /// Find the nearest point source to `(yaw_rad, pitch_rad)`.
    ///
    /// Returns `None` if the map is empty.
    #[must_use]
    pub fn nearest(&self, yaw_rad: f32, pitch_rad: f32) -> Option<&AudioSpherePoint> {
        self.points.iter().min_by(|a, b| {
            let da = angular_distance(yaw_rad, pitch_rad, a.yaw_rad, a.pitch_rad);
            let db = angular_distance(yaw_rad, pitch_rad, b.yaw_rad, b.pitch_rad);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Compute the gain-weighted centroid of all sources as a `(yaw, pitch)`
    /// pair.
    ///
    /// Uses circular averaging (vector sum) weighted by linear gain.
    /// Returns `None` if the map is empty or if total linear gain is negligible.
    #[must_use]
    pub fn gain_weighted_centroid(&self) -> Option<(f32, f32)> {
        if self.points.is_empty() {
            return None;
        }
        let (mut sx, mut sy, mut sz, mut total_gain) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for p in &self.points {
            let g = p.linear_gain();
            let cart = p.to_cartesian();
            sx += g * cart[0];
            sy += g * cart[1];
            sz += g * cart[2];
            total_gain += g;
        }
        if total_gain < f32::EPSILON {
            return None;
        }
        let cx = sx / total_gain;
        let cy = sy / total_gain;
        let cz = sz / total_gain;
        let pitch = cy.clamp(-1.0, 1.0).asin();
        let yaw = cx.atan2(cz);
        Some((yaw, pitch))
    }

    /// Return the points sorted by descending linear gain (loudest first).
    #[must_use]
    pub fn sorted_by_gain(&self) -> Vec<&AudioSpherePoint> {
        let mut sorted: Vec<&AudioSpherePoint> = self.points.iter().collect();
        sorted.sort_by(|a, b| {
            b.linear_gain()
                .partial_cmp(&a.linear_gain())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }
}

// ─── BinauralHint ─────────────────────────────────────────────────────────────

/// A rendering hint for one virtual speaker in a binaural decode of FOA.
///
/// The hint expresses where a virtual speaker sits in the listener's **head-
/// relative** frame after applying a yaw/pitch head rotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinauralHint {
    /// Azimuth offset in the listener's frame, radians.  Positive = clockwise
    /// (right ear).
    pub listener_azimuth_rad: f32,
    /// Elevation offset in the listener's frame, radians.
    pub listener_elevation_rad: f32,
    /// Nominal gain for this virtual speaker channel (linear, before HRTF).
    pub gain: f32,
}

// ─── HeadRotationRenderer ─────────────────────────────────────────────────────

/// Computes head-rotation-compensated rendering hints for first-order ambisonics.
///
/// For FOA the standard virtual loudspeaker array is a symmetric 8-point layout
/// at ±30° / ±90° azimuth and ±45° elevation.  Each speaker direction is
/// rotated by the inverse of the listener's current head orientation so that
/// the sound field remains world-fixed regardless of where the listener is
/// looking.
#[derive(Debug, Clone)]
pub struct HeadRotationRenderer {
    /// Virtual loudspeaker directions (yaw, pitch) in the world frame, radians.
    speakers: Vec<(f32, f32, f32)>, // (yaw, pitch, gain)
}

impl HeadRotationRenderer {
    /// Create a renderer with the default 8-point virtual speaker layout used
    /// for FOA binaural decoding.
    ///
    /// Speakers are arranged at azimuth ±30° / ±110° and elevation ±35°,
    /// matching a common FOA binaural decode array.
    #[must_use]
    pub fn new_foa_default() -> Self {
        let az = [30.0f32, -30.0, 110.0, -110.0];
        let el = [35.0f32, -35.0];
        let mut speakers = Vec::with_capacity(8);
        for &a in &az {
            for &e in &el {
                speakers.push((a.to_radians(), e.to_radians(), 1.0));
            }
        }
        Self { speakers }
    }

    /// Create a renderer with a custom speaker layout.
    ///
    /// Each entry is `(yaw_rad, pitch_rad, gain)`.
    #[must_use]
    pub fn from_speakers(speakers: Vec<(f32, f32, f32)>) -> Self {
        Self { speakers }
    }

    /// Compute binaural rendering hints for the given listener head orientation.
    ///
    /// `head_yaw_rad` and `head_pitch_rad` describe where the listener is
    /// looking in the world frame.  The function returns the virtual speaker
    /// directions in the listener's head-relative frame.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidDimensions`] if there are no speakers.
    pub fn render_hints(
        &self,
        head_yaw_rad: f32,
        head_pitch_rad: f32,
    ) -> Result<Vec<BinauralHint>, VrError> {
        if self.speakers.is_empty() {
            return Err(VrError::InvalidDimensions("no speakers defined".into()));
        }

        // Rotation matrix for the inverse of the listener's head orientation:
        // rotate world directions by -yaw, then -pitch.
        let (sy, cy) = (-head_yaw_rad).sin_cos();
        let (sp, cp) = (-head_pitch_rad).sin_cos();
        // Inverse yaw (around Y): Ry^-1
        // Inverse pitch (around X): Rx^-1
        // Combined: Rx^-1 * Ry^-1

        let hints = self
            .speakers
            .iter()
            .map(|&(yaw, pitch, gain)| {
                // Speaker direction as unit vector
                let cp_s = pitch.cos();
                let x = cp_s * yaw.sin();
                let y = pitch.sin();
                let z = cp_s * yaw.cos();

                // Apply Ry^-1 (yaw rotation)
                let x1 = x * cy + z * sy;
                let y1 = y;
                let z1 = -x * sy + z * cy;

                // Apply Rx^-1 (pitch rotation)
                let x2 = x1;
                let y2 = y1 * cp + z1 * sp;
                let z2 = -y1 * sp + z1 * cp;

                let listener_elevation = y2.clamp(-1.0, 1.0).asin();
                let listener_azimuth = x2.atan2(z2);

                BinauralHint {
                    listener_azimuth_rad: listener_azimuth,
                    listener_elevation_rad: listener_elevation,
                    gain,
                }
            })
            .collect();

        Ok(hints)
    }

    /// Return the number of virtual speakers.
    #[must_use]
    pub fn speaker_count(&self) -> usize {
        self.speakers.len()
    }
}

// ─── Audio sphere encoding helpers ────────────────────────────────────────────

/// Encode a mono point source at `(yaw_rad, pitch_rad)` into first-order
/// ambisonics (FOA) W/X/Y/Z coefficients using ACN/SN3D.
///
/// For a unit-gain source the coefficients are:
/// * W = 1 / √2
/// * X = cos(pitch) · cos(yaw)  (forward component)
/// * Y = cos(pitch) · sin(yaw)  (left component)
/// * Z = sin(pitch)              (up component)
///
/// Returns `[W, X, Y, Z]`.
#[must_use]
pub fn encode_foa_sn3d(yaw_rad: f32, pitch_rad: f32, gain: f32) -> [f32; 4] {
    let cp = pitch_rad.cos();
    let w = gain / std::f32::consts::SQRT_2;
    let x = gain * cp * yaw_rad.cos();
    let y = gain * cp * yaw_rad.sin();
    let z = gain * pitch_rad.sin();
    [w, x, y, z]
}

/// Mix multiple ambisonics signals by summing their channel vectors.
///
/// All input slices must have the same length.  Returns the summed channel
/// vector, or an error if the lengths differ.
///
/// # Errors
///
/// Returns [`VrError::InvalidDimensions`] if any input has a different length
/// from the first, or if `signals` is empty.
pub fn mix_ambisonics(signals: &[Vec<f32>]) -> Result<Vec<f32>, VrError> {
    if signals.is_empty() {
        return Err(VrError::InvalidDimensions("no signals to mix".into()));
    }
    let ch = signals[0].len();
    for (i, s) in signals.iter().enumerate() {
        if s.len() != ch {
            return Err(VrError::InvalidDimensions(format!(
                "signal {i} has {} channels, expected {ch}",
                s.len()
            )));
        }
    }
    let mut out = vec![0.0f32; ch];
    for sig in signals {
        for (o, &v) in out.iter_mut().zip(sig.iter()) {
            *o += v;
        }
    }
    Ok(out)
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Haversine great-circle angular distance in radians.
#[inline]
pub fn angular_distance(yaw1: f32, pitch1: f32, yaw2: f32, pitch2: f32) -> f32 {
    let dpitch = pitch2 - pitch1;
    let dyaw = {
        let mut d = (yaw2 - yaw1) % (2.0 * PI);
        if d > PI {
            d -= 2.0 * PI;
        } else if d < -PI {
            d += 2.0 * PI;
        }
        d
    };
    let hav =
        (dpitch * 0.5).sin().powi(2) + pitch1.cos() * pitch2.cos() * (dyaw * 0.5).sin().powi(2);
    2.0 * hav.sqrt().clamp(0.0, 1.0).asin()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    // ── AmbisonicsOrder ───────────────────────────────────────────────────────

    #[test]
    fn ambisonics_order_channel_counts() {
        assert_eq!(AmbisonicsOrder::First.channel_count(), 4);
        assert_eq!(AmbisonicsOrder::Second.channel_count(), 9);
        assert_eq!(AmbisonicsOrder::Third.channel_count(), 16);
        assert_eq!(AmbisonicsOrder::Fourth.channel_count(), 25);
    }

    // ── AmbisonicsMetadata ────────────────────────────────────────────────────

    #[test]
    fn metadata_validate_correct() {
        let m = AmbisonicsMetadata {
            order: AmbisonicsOrder::First,
            normalisation: NormalisationConvention::Sn3d,
            channel_count: 4,
            head_locked_stereo_gain: 0.0,
            description: String::new(),
        };
        assert!(m.validate().is_ok());
    }

    #[test]
    fn metadata_validate_wrong_channel_count() {
        let m = AmbisonicsMetadata {
            order: AmbisonicsOrder::Second,
            normalisation: NormalisationConvention::N3d,
            channel_count: 4, // wrong: should be 9
            head_locked_stereo_gain: 0.0,
            description: String::new(),
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn sn3d_to_n3d_gain_acn0_is_one() {
        // ACN 0: l = 0, gain = sqrt(2*0+1) = 1
        let g = AmbisonicsMetadata::sn3d_to_n3d_gain(0);
        assert!((g - 1.0).abs() < 1e-5, "g={g}");
    }

    #[test]
    fn sn3d_n3d_round_trip() {
        for acn in 0..16u32 {
            let g_fwd = AmbisonicsMetadata::sn3d_to_n3d_gain(acn);
            let g_inv = AmbisonicsMetadata::n3d_to_sn3d_gain(acn);
            assert!(
                (g_fwd * g_inv - 1.0).abs() < 1e-5,
                "acn={acn} fwd={g_fwd} inv={g_inv}"
            );
        }
    }

    // ── AudioSpherePoint ──────────────────────────────────────────────────────

    #[test]
    fn audio_sphere_point_linear_gain_0db() {
        let p = AudioSpherePoint {
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            gain_db: 0.0,
            label: None,
        };
        assert!((p.linear_gain() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn audio_sphere_point_cartesian_forward() {
        let p = AudioSpherePoint {
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            gain_db: 0.0,
            label: None,
        };
        let cart = p.to_cartesian();
        // (yaw=0, pitch=0) should be (0, 0, 1) — directly forward
        assert!(cart[0].abs() < 1e-5, "x={}", cart[0]);
        assert!(cart[1].abs() < 1e-5, "y={}", cart[1]);
        assert!((cart[2] - 1.0).abs() < 1e-5, "z={}", cart[2]);
    }

    // ── AudioSphereMap ────────────────────────────────────────────────────────

    #[test]
    fn sphere_map_nearest_single_point() {
        let mut m = AudioSphereMap::new();
        m.add(AudioSpherePoint {
            yaw_rad: 0.5,
            pitch_rad: 0.2,
            gain_db: 0.0,
            label: None,
        });
        let n = m.nearest(0.6, 0.2).expect("non-empty");
        assert!((n.yaw_rad - 0.5).abs() < 1e-5);
    }

    #[test]
    fn sphere_map_nearest_returns_closest() {
        let mut m = AudioSphereMap::new();
        m.add(AudioSpherePoint {
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            gain_db: 0.0,
            label: None,
        });
        m.add(AudioSpherePoint {
            yaw_rad: 2.0,
            pitch_rad: 0.0,
            gain_db: 0.0,
            label: None,
        });
        // Query at yaw=0.1 — should return the yaw=0.0 source
        let n = m.nearest(0.1, 0.0).expect("non-empty");
        assert!((n.yaw_rad - 0.0).abs() < 1e-5, "yaw={}", n.yaw_rad);
    }

    #[test]
    fn sphere_map_gain_weighted_centroid_symmetric() {
        let mut m = AudioSphereMap::new();
        // Two equal-gain sources at small ±yaw and equal pitch — centroid should
        // have |yaw| near 0 (sources point in the same forward hemisphere).
        m.add(AudioSpherePoint {
            yaw_rad: 0.3,
            pitch_rad: 0.1,
            gain_db: 0.0,
            label: None,
        });
        m.add(AudioSpherePoint {
            yaw_rad: -0.3,
            pitch_rad: 0.1,
            gain_db: 0.0,
            label: None,
        });
        let (cy, _cp) = m.gain_weighted_centroid().expect("non-empty");
        // Yaw component should cancel out — centroid yaw ≈ 0
        assert!(cy.abs() < 0.05, "centroid yaw should be ~0, got {cy}");
    }

    #[test]
    fn sphere_map_sorted_by_gain_descending() {
        let mut m = AudioSphereMap::new();
        m.add(AudioSpherePoint {
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            gain_db: -12.0,
            label: None,
        });
        m.add(AudioSpherePoint {
            yaw_rad: 1.0,
            pitch_rad: 0.0,
            gain_db: 0.0,
            label: None,
        });
        m.add(AudioSpherePoint {
            yaw_rad: 2.0,
            pitch_rad: 0.0,
            gain_db: -6.0,
            label: None,
        });
        let sorted = m.sorted_by_gain();
        assert!(
            (sorted[0].gain_db - 0.0).abs() < 1e-5,
            "first={}",
            sorted[0].gain_db
        );
        assert!(
            (sorted[1].gain_db + 6.0).abs() < 1e-5,
            "second={}",
            sorted[1].gain_db
        );
        assert!(
            (sorted[2].gain_db + 12.0).abs() < 1e-5,
            "third={}",
            sorted[2].gain_db
        );
    }

    // ── HeadRotationRenderer ──────────────────────────────────────────────────

    #[test]
    fn head_rotation_renderer_foa_default_8_speakers() {
        let r = HeadRotationRenderer::new_foa_default();
        assert_eq!(r.speaker_count(), 8);
    }

    #[test]
    fn head_rotation_renderer_hints_identity_orientation() {
        let r = HeadRotationRenderer::new_foa_default();
        let hints = r.render_hints(0.0, 0.0).expect("ok");
        assert_eq!(hints.len(), 8);
        // All hints should have gain == 1.0
        for h in &hints {
            assert!((h.gain - 1.0).abs() < 1e-5, "gain={}", h.gain);
        }
    }

    #[test]
    fn head_rotation_renderer_no_speakers_error() {
        let r = HeadRotationRenderer::from_speakers(vec![]);
        assert!(r.render_hints(0.0, 0.0).is_err());
    }

    // ── encode_foa_sn3d ───────────────────────────────────────────────────────

    #[test]
    fn encode_foa_forward_source_nonzero_wx() {
        let coeff = encode_foa_sn3d(0.0, 0.0, 1.0);
        // W = 1/√2, X = 1 (forward), Y = 0, Z = 0
        assert!(
            (coeff[0] - 1.0 / std::f32::consts::SQRT_2).abs() < 1e-5,
            "W={}",
            coeff[0]
        );
        assert!((coeff[1] - 1.0).abs() < 1e-5, "X={}", coeff[1]);
        assert!(coeff[2].abs() < 1e-5, "Y={}", coeff[2]);
        assert!(coeff[3].abs() < 1e-5, "Z={}", coeff[3]);
    }

    #[test]
    fn encode_foa_top_source_z_nonzero() {
        // Source directly above (pitch = π/2)
        let coeff = encode_foa_sn3d(0.0, FRAC_PI_2, 1.0);
        // Z = sin(π/2) = 1
        assert!((coeff[3] - 1.0).abs() < 1e-4, "Z={}", coeff[3]);
    }

    // ── mix_ambisonics ────────────────────────────────────────────────────────

    #[test]
    fn mix_ambisonics_two_sources_sum_correctly() {
        let a = encode_foa_sn3d(0.0, 0.0, 1.0).to_vec();
        let b = encode_foa_sn3d(FRAC_PI_2, 0.0, 1.0).to_vec();
        let mixed = mix_ambisonics(&[a.clone(), b.clone()]).expect("ok");
        for (i, (ma, mb)) in a.iter().zip(b.iter()).enumerate() {
            assert!((mixed[i] - (ma + mb)).abs() < 1e-5, "ch={i}");
        }
    }

    #[test]
    fn mix_ambisonics_empty_input_error() {
        assert!(mix_ambisonics(&[]).is_err());
    }

    #[test]
    fn mix_ambisonics_mismatched_length_error() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 2.0];
        assert!(mix_ambisonics(&[a, b]).is_err());
    }
}
