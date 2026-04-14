//! Near-field HRTF compensation for headphone rendering.
//!
//! When a Head-Related Transfer Function (HRTF) is measured at a far-field
//! distance (typically ≥ 1 m) and then used to render a source that is
//! physically closer to the listener (e.g. 0.25 m), the rendered source
//! will sound incorrect because:
//!
//! 1. **Pressure gradient** — The sound pressure near a small source falls
//!    off faster than 1/r at close range.
//! 2. **Frequency-dependent near-field boost** — Low frequencies experience
//!    a strong proximity boost (the "proximity effect") relative to far-field.
//! 3. **Interaural level difference (ILD) asymmetry** — The ILD at near-field
//!    distances is larger than predicted by the far-field HRTF.
//!
//! This module implements a two-stage near-field compensation chain:
//!
//! ## Stage 1 — Proximity-effect shelf filter
//!
//! A first-order high-pass shelving filter that boosts low frequencies to
//! approximate the near-field pressure build-up for a point source.  The
//! shelf frequency is proportional to `c / (2π r)` where `c` is the speed of
//! sound and `r` is the source distance.
//!
//! ## Stage 2 — Distance-dependent gain compensation
//!
//! A scalar gain that compensates for the 1/r amplitude decay between the
//! HRTF measurement distance and the rendered source distance.
//!
//! # References
//! Shinn-Cunningham, B. G., Santarelli, S., & Kopčo, N. (2000).
//! "Tonal and spatial quality of near-field sounds with varying distance."
//! *Proceedings of the International Conference on Auditory Display*.
//!
//! Brungart, D. S., & Rabinowitz, W. M. (1999). "Auditory localization of
//! nearby sources. Head-related transfer functions." *Journal of the
//! Acoustical Society of America*, 106(3), 1465–1479.

use crate::SpatialError;

/// Speed of sound in air at 20 °C, in m/s.
const SPEED_OF_SOUND: f32 = 343.0;

// ─── Proximity-effect shelf filter ───────────────────────────────────────────

/// A first-order low-shelf filter implementing the proximity-effect boost.
///
/// Transfer function (bilinear-transform approximation):
/// ```text
/// H(z) = (b0 + b1*z^-1) / (1 + a1*z^-1)
/// ```
#[derive(Debug, Clone)]
struct ProximityShelf {
    b0: f32,
    b1: f32,
    a1: f32,
    /// Filter state (previous input / output).
    x_prev: f32,
    y_prev: f32,
}

impl ProximityShelf {
    /// Design a proximity-effect shelf for the given source distance.
    ///
    /// # Parameters
    /// - `distance_m`: source-to-listener distance in metres.
    /// - `hrtf_distance_m`: HRTF measurement distance in metres.
    /// - `sample_rate`: audio sample rate in Hz.
    ///
    /// The shelf boost is `hrtf_distance_m / distance_m` in the low-frequency
    /// limit, rolling off above `f_c = c / (2π * distance_m)`.
    fn new(distance_m: f32, hrtf_distance_m: f32, sample_rate: u32) -> Self {
        // Shelf gain (linear amplitude) at DC.
        let dc_gain = (hrtf_distance_m / distance_m).min(20.0); // cap at 20× (~26 dB)

        // Crossover frequency where far-field behaviour takes over.
        let f_c = SPEED_OF_SOUND / (2.0 * std::f32::consts::PI * distance_m);
        let wc = 2.0 * std::f32::consts::PI * f_c / sample_rate as f32;

        // Bilinear transform of H(s) = (s + wc * dc_gain) / (s + wc)
        // (first-order prototype with unity gain at ∞, dc_gain at DC).
        let two = 2.0_f32;
        let tan_half_wc = (wc / two).tan().clamp(1e-6, 100.0);

        // Bilinear: s → 2*fs*(1-z^-1)/(1+z^-1), but we pre-warp via tan.
        // Using the matched bilinear for a first-order shelf:
        //   b0 = 1 + dc_gain * tan_half_wc
        //   b1 = -(1 - dc_gain * tan_half_wc)
        //   a0 = 1 + tan_half_wc
        //   a1 = -(1 - tan_half_wc)   [in the denominator sign convention a1 is subtracted]
        let a0 = 1.0 + tan_half_wc;
        let b0_raw = 1.0 + dc_gain * tan_half_wc;
        let b1_raw = -(1.0 - dc_gain * tan_half_wc);
        let a1_raw = -(1.0 - tan_half_wc);

        Self {
            b0: b0_raw / a0,
            b1: b1_raw / a0,
            a1: a1_raw / a0,
            x_prev: 0.0,
            y_prev: 0.0,
        }
    }

    /// Process a single sample.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x_prev - self.a1 * self.y_prev;
        self.x_prev = x;
        self.y_prev = y;
        y
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }
}

// ─── ILD compensation ─────────────────────────────────────────────────────────

/// Per-ear near-field ILD scaling factor.
///
/// When a source is close to the listener, the interaural level difference
/// increases beyond what the far-field HRTF predicts.  This simple model
/// scales each ear independently by `r_ref / r_ear`, where `r_ear` is the
/// approximate distance from the source to each ear (estimated from the
/// head radius and source azimuth).
#[derive(Debug, Clone)]
struct IldCompensation {
    /// Gain for the left ear.
    pub gain_left: f32,
    /// Gain for the right ear.
    pub gain_right: f32,
}

impl IldCompensation {
    /// Compute ILD gains.
    ///
    /// # Parameters
    /// - `distance_m`: source distance from head centre (m).
    /// - `azimuth_deg`: source azimuth in degrees (positive = left).
    /// - `hrtf_distance_m`: HRTF reference distance (m).
    /// - `head_radius_m`: head radius model (default ≈ 0.0875 m).
    fn new(
        distance_m: f32,
        azimuth_deg: f32,
        hrtf_distance_m: f32,
        head_radius_m: f32,
    ) -> Self {
        let az = azimuth_deg.to_radians();
        // Approximate distance from source to left/right ears using law of cosines.
        // Left ear at (+head_radius, 0), right ear at (−head_radius, 0) in a 2D head model.
        let hr = head_radius_m;
        // Source in polar: x = distance * sin(az), y = distance * cos(az).
        let sx = distance_m * az.sin();
        let sy = distance_m * az.cos();

        let r_left = ((sx - hr).powi(2) + sy.powi(2)).sqrt().max(0.01);
        let r_right = ((sx + hr).powi(2) + sy.powi(2)).sqrt().max(0.01);

        // Gains: scale so that at the HRTF reference distance the gain is 1.0.
        let gain_left = (hrtf_distance_m / r_left).clamp(0.05, 20.0);
        let gain_right = (hrtf_distance_m / r_right).clamp(0.05, 20.0);

        Self {
            gain_left,
            gain_right,
        }
    }
}

// ─── NearFieldCompensator ─────────────────────────────────────────────────────

/// Near-field HRTF compensation processor.
///
/// Applies proximity-effect shelving and distance-gain compensation to
/// correct far-field HRTF rendering for near-field sources (< 1 m).
///
/// # Example
///
/// ```rust
/// use oximedia_spatial::near_field_compensation::NearFieldCompensator;
///
/// let mut comp = NearFieldCompensator::new(0.25, 1.0, 0.0, 48_000).unwrap();
/// let mono = vec![0.3_f32; 128];
/// let (left, right) = comp.process(&mono);
/// assert_eq!(left.len(), 128);
/// assert_eq!(right.len(), 128);
/// ```
#[derive(Debug, Clone)]
pub struct NearFieldCompensator {
    /// Shelf filter applied to the left channel.
    shelf_left: ProximityShelf,
    /// Shelf filter applied to the right channel.
    shelf_right: ProximityShelf,
    /// ILD gains.
    ild: IldCompensation,
    /// Amplitude gain from reference distance to source distance (1/r model).
    distance_gain: f32,
    /// Source distance in metres.
    source_distance_m: f32,
    /// HRTF reference distance in metres.
    hrtf_distance_m: f32,
    /// Head radius in metres.
    head_radius_m: f32,
    /// Source azimuth in degrees.
    azimuth_deg: f32,
    /// Audio sample rate.
    sample_rate: u32,
}

impl NearFieldCompensator {
    /// Construct a near-field compensator.
    ///
    /// # Parameters
    /// - `source_distance_m`: distance of the source to the head centre (m).
    /// - `hrtf_distance_m`: distance at which the HRTF was measured (m).
    /// - `azimuth_deg`: source azimuth (positive = left).
    /// - `sample_rate`: audio sample rate in Hz.
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidConfig`] if:
    /// - `source_distance_m` ≤ 0.
    /// - `hrtf_distance_m` ≤ 0.
    /// - `sample_rate` == 0.
    pub fn new(
        source_distance_m: f32,
        hrtf_distance_m: f32,
        azimuth_deg: f32,
        sample_rate: u32,
    ) -> Result<Self, SpatialError> {
        Self::with_head_radius(source_distance_m, hrtf_distance_m, azimuth_deg, sample_rate, 0.0875)
    }

    /// Construct with an explicit head radius model.
    ///
    /// # Parameters
    /// - `head_radius_m`: distance from head centre to each ear, in metres.
    ///   Default is 0.0875 m (ITU head model).
    ///
    /// # Errors
    /// Same as [`Self::new`] plus [`SpatialError::InvalidConfig`] if
    /// `head_radius_m` ≤ 0.
    pub fn with_head_radius(
        source_distance_m: f32,
        hrtf_distance_m: f32,
        azimuth_deg: f32,
        sample_rate: u32,
        head_radius_m: f32,
    ) -> Result<Self, SpatialError> {
        if source_distance_m <= 0.0 {
            return Err(SpatialError::InvalidConfig(format!(
                "source_distance_m must be > 0, got {source_distance_m}"
            )));
        }
        if hrtf_distance_m <= 0.0 {
            return Err(SpatialError::InvalidConfig(format!(
                "hrtf_distance_m must be > 0, got {hrtf_distance_m}"
            )));
        }
        if sample_rate == 0 {
            return Err(SpatialError::InvalidConfig(
                "sample_rate must be > 0".into(),
            ));
        }
        if head_radius_m <= 0.0 {
            return Err(SpatialError::InvalidConfig(format!(
                "head_radius_m must be > 0, got {head_radius_m}"
            )));
        }

        let shelf_left =
            ProximityShelf::new(source_distance_m, hrtf_distance_m, sample_rate);
        let shelf_right =
            ProximityShelf::new(source_distance_m, hrtf_distance_m, sample_rate);
        let ild = IldCompensation::new(
            source_distance_m,
            azimuth_deg,
            hrtf_distance_m,
            head_radius_m,
        );

        // Far-field amplitude correction: at the HRTF reference distance the far-field
        // HRTF has unity gain; at closer distances the actual amplitude is hrtf_d / source_d.
        let distance_gain = (hrtf_distance_m / source_distance_m).min(20.0);

        Ok(Self {
            shelf_left,
            shelf_right,
            ild,
            distance_gain,
            source_distance_m,
            hrtf_distance_m,
            head_radius_m,
            azimuth_deg,
            sample_rate,
        })
    }

    /// Update the source position (distance and azimuth) without reallocation.
    ///
    /// This recomputes the shelf-filter coefficients and ILD gains in place,
    /// suitable for real-time spatialisation with a moving source.
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidConfig`] if `source_distance_m` ≤ 0.
    pub fn update_position(
        &mut self,
        source_distance_m: f32,
        azimuth_deg: f32,
    ) -> Result<(), SpatialError> {
        if source_distance_m <= 0.0 {
            return Err(SpatialError::InvalidConfig(format!(
                "source_distance_m must be > 0, got {source_distance_m}"
            )));
        }
        self.source_distance_m = source_distance_m;
        self.azimuth_deg = azimuth_deg;

        self.shelf_left =
            ProximityShelf::new(source_distance_m, self.hrtf_distance_m, self.sample_rate);
        self.shelf_right =
            ProximityShelf::new(source_distance_m, self.hrtf_distance_m, self.sample_rate);
        self.ild = IldCompensation::new(
            source_distance_m,
            azimuth_deg,
            self.hrtf_distance_m,
            self.head_radius_m,
        );
        self.distance_gain = (self.hrtf_distance_m / source_distance_m).min(20.0);
        Ok(())
    }

    /// Process a mono input buffer and return compensated `(left, right)` buffers.
    ///
    /// The compensation chain is:
    /// 1. Apply proximity-effect shelf filter (independently per ear).
    /// 2. Scale by ILD gain (per ear).
    /// 3. Apply distance amplitude correction.
    pub fn process(&mut self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut left = Vec::with_capacity(input.len());
        let mut right = Vec::with_capacity(input.len());

        for &x in input {
            let l = self.shelf_left.process(x) * self.ild.gain_left * self.distance_gain;
            let r = self.shelf_right.process(x) * self.ild.gain_right * self.distance_gain;
            left.push(l);
            right.push(r);
        }

        (left, right)
    }

    /// Reset all filter states (e.g., at the start of a new segment).
    pub fn reset(&mut self) {
        self.shelf_left.reset();
        self.shelf_right.reset();
    }

    /// Return the source distance in metres.
    pub fn source_distance_m(&self) -> f32 {
        self.source_distance_m
    }

    /// Return the HRTF reference distance in metres.
    pub fn hrtf_distance_m(&self) -> f32 {
        self.hrtf_distance_m
    }

    /// Return the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Return the current ILD gain for the left ear.
    pub fn ild_gain_left(&self) -> f32 {
        self.ild.gain_left
    }

    /// Return the current ILD gain for the right ear.
    pub fn ild_gain_right(&self) -> f32 {
        self.ild.gain_right
    }

    /// Return the distance-based amplitude gain factor.
    pub fn distance_gain(&self) -> f32 {
        self.distance_gain
    }
}

// ─── ProximityEffect utility ──────────────────────────────────────────────────

/// Compute the proximity-effect crossover frequency for a point source at
/// the given distance.
///
/// Returns the frequency (Hz) below which the near-field pressure boost
/// becomes significant (typically `c / (2π r)`).
///
/// # Errors
/// Returns [`SpatialError::InvalidConfig`] if `distance_m` ≤ 0.
pub fn proximity_crossover_freq(distance_m: f32) -> Result<f32, SpatialError> {
    if distance_m <= 0.0 {
        return Err(SpatialError::InvalidConfig(format!(
            "distance_m must be > 0, got {distance_m}"
        )));
    }
    Ok(SPEED_OF_SOUND / (2.0 * std::f32::consts::PI * distance_m))
}

/// Estimate the near-field gain boost at a given frequency and source distance.
///
/// Based on the pressure near-field model:
/// `boost(f) = sqrt(1 + (f_c / f)^2)`
///
/// where `f_c` is the proximity crossover frequency.
///
/// # Errors
/// Returns [`SpatialError::InvalidConfig`] if `distance_m` ≤ 0 or `freq_hz` ≤ 0.
pub fn near_field_boost_db(distance_m: f32, freq_hz: f32) -> Result<f32, SpatialError> {
    if distance_m <= 0.0 {
        return Err(SpatialError::InvalidConfig(format!(
            "distance_m must be > 0, got {distance_m}"
        )));
    }
    if freq_hz <= 0.0 {
        return Err(SpatialError::InvalidConfig(format!(
            "freq_hz must be > 0, got {freq_hz}"
        )));
    }
    let f_c = proximity_crossover_freq(distance_m)?;
    let boost_linear = (1.0 + (f_c / freq_hz).powi(2)).sqrt();
    Ok(20.0 * boost_linear.log10())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ProximityShelf ───────────────────────────────────────────────────────

    #[test]
    fn test_shelf_boosts_dc_for_near_source() {
        // A source at 0.25 m should be boosted relative to HRTF at 1.0 m.
        let mut shelf = ProximityShelf::new(0.25, 1.0, 48_000);
        // Feed impulse, sum output energy over time.
        let mut energy = 0.0_f32;
        for i in 0..256 {
            let x = if i == 0 { 1.0 } else { 0.0 };
            energy += shelf.process(x).powi(2);
        }
        // Without shelf the impulse energy would be 1.0 (unit impulse).
        // With boost the energy should be > 1.0.
        assert!(
            energy > 1.0,
            "Near-field shelf should boost energy: {energy}"
        );
    }

    #[test]
    fn test_shelf_unity_at_far_field() {
        // When source distance == HRTF distance, gain is 1 → impulse response energy ≈ 1.
        let mut shelf = ProximityShelf::new(1.0, 1.0, 48_000);
        let mut energy = 0.0_f32;
        for i in 0..256 {
            let x = if i == 0 { 1.0 } else { 0.0 };
            energy += shelf.process(x).powi(2);
        }
        // Energy should be ≈ 1.0 (within a tolerance due to filter ringing).
        assert!(
            (energy - 1.0).abs() < 0.5,
            "Far-field source should have near-unity shelf energy: {energy}"
        );
    }

    #[test]
    fn test_shelf_reset_clears_state() {
        let mut shelf = ProximityShelf::new(0.5, 1.0, 48_000);
        for _ in 0..64 {
            shelf.process(1.0);
        }
        shelf.reset();
        let out = shelf.process(0.0);
        assert_eq!(out, 0.0, "Reset shelf should output silence for zero input");
    }

    // ── NearFieldCompensator ─────────────────────────────────────────────────

    #[test]
    fn test_compensator_construction_valid() {
        let comp = NearFieldCompensator::new(0.25, 1.0, 0.0, 48_000);
        assert!(comp.is_ok(), "Valid parameters should succeed");
    }

    #[test]
    fn test_compensator_zero_distance_rejected() {
        let comp = NearFieldCompensator::new(0.0, 1.0, 0.0, 48_000);
        assert!(comp.is_err(), "Zero source distance should be rejected");
    }

    #[test]
    fn test_compensator_zero_sample_rate_rejected() {
        let comp = NearFieldCompensator::new(0.5, 1.0, 0.0, 0);
        assert!(comp.is_err(), "Zero sample rate should be rejected");
    }

    #[test]
    fn test_compensator_output_lengths_match_input() {
        let mut comp = NearFieldCompensator::new(0.25, 1.0, 30.0, 48_000)
            .expect("compensator creation should succeed");
        let mono = vec![0.5_f32; 256];
        let (l, r) = comp.process(&mono);
        assert_eq!(l.len(), 256);
        assert_eq!(r.len(), 256);
    }

    #[test]
    fn test_compensator_near_source_boosts_amplitude() {
        let mut comp_near = NearFieldCompensator::new(0.25, 1.0, 0.0, 48_000)
            .expect("near compensator creation should succeed");
        let mut comp_far = NearFieldCompensator::new(1.0, 1.0, 0.0, 48_000)
            .expect("far compensator creation should succeed");
        let impulse: Vec<f32> = (0..256).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let (l_near, _) = comp_near.process(&impulse);
        let (l_far, _) = comp_far.process(&impulse);
        let energy_near: f32 = l_near.iter().map(|x| x * x).sum();
        let energy_far: f32 = l_far.iter().map(|x| x * x).sum();
        assert!(
            energy_near > energy_far,
            "Near-field source should have higher energy: near={energy_near}, far={energy_far}"
        );
    }

    #[test]
    fn test_compensator_ild_differs_left_right_for_off_axis_source() {
        let comp = NearFieldCompensator::new(0.5, 1.0, 60.0, 48_000)
            .expect("off-axis compensator creation should succeed");
        // Source at 60° left → left ear closer → left ILD gain should differ from right.
        let ild_left = comp.ild_gain_left();
        let ild_right = comp.ild_gain_right();
        assert!(
            (ild_left - ild_right).abs() > 0.01,
            "Off-axis source should produce asymmetric ILD: left={ild_left}, right={ild_right}"
        );
    }

    #[test]
    fn test_compensator_reset_clears_state() {
        let mut comp = NearFieldCompensator::new(0.3, 1.0, 0.0, 48_000)
            .expect("compensator reset test creation should succeed");
        let sine: Vec<f32> = (0..512)
            .map(|i| (2.0 * std::f32::consts::PI * 100.0 * i as f32 / 48_000.0).sin())
            .collect();
        comp.process(&sine);
        comp.reset();
        let silence = vec![0.0_f32; 64];
        let (l, r) = comp.process(&silence);
        assert!(
            l.iter().all(|&x| x == 0.0),
            "After reset, zero input should produce zero left output"
        );
        assert!(
            r.iter().all(|&x| x == 0.0),
            "After reset, zero input should produce zero right output"
        );
    }

    #[test]
    fn test_compensator_update_position() {
        let mut comp = NearFieldCompensator::new(0.5, 1.0, 0.0, 48_000)
            .expect("compensator for position update test should succeed");
        let result = comp.update_position(0.25, 45.0);
        assert!(result.is_ok());
        assert!((comp.source_distance_m() - 0.25).abs() < 1e-5);
    }

    // ── Utility functions ─────────────────────────────────────────────────────

    #[test]
    fn test_proximity_crossover_freq_1m() {
        let f = proximity_crossover_freq(1.0).expect("crossover freq for 1m should succeed");
        // f_c = 343 / (2π * 1) ≈ 54.6 Hz
        assert!(
            (f - 54.6).abs() < 1.0,
            "Crossover at 1 m should be ≈ 54.6 Hz, got {f}"
        );
    }

    #[test]
    fn test_proximity_crossover_freq_zero_distance_rejected() {
        let err = proximity_crossover_freq(0.0);
        assert!(err.is_err(), "Zero distance should be rejected");
    }

    #[test]
    fn test_near_field_boost_db_low_freq() {
        // At 10 Hz, a 0.25 m source should have a large boost (>> 0 dB).
        let boost = near_field_boost_db(0.25, 10.0).expect("boost calculation should succeed");
        assert!(boost > 5.0, "Low-frequency boost should be significant: {boost} dB");
    }

    #[test]
    fn test_near_field_boost_db_high_freq() {
        // At high frequency (f >> f_c), boost approaches 0 dB.
        let boost = near_field_boost_db(0.25, 10_000.0)
            .expect("high-freq boost calculation should succeed");
        assert!(
            boost < 1.0,
            "High-frequency boost should approach 0 dB: {boost} dB"
        );
    }

    #[test]
    fn test_near_field_boost_increases_as_distance_decreases() {
        let boost_far = near_field_boost_db(2.0, 50.0).expect("2m boost should succeed");
        let boost_near = near_field_boost_db(0.1, 50.0).expect("0.1m boost should succeed");
        assert!(
            boost_near > boost_far,
            "Closer source should have larger boost at 50 Hz: near={boost_near}, far={boost_far}"
        );
    }

    #[test]
    fn test_distance_gain_proportional() {
        let comp = NearFieldCompensator::new(0.5, 1.0, 0.0, 48_000)
            .expect("compensator distance gain test should succeed");
        // HRTF at 1.0 m, source at 0.5 m → gain = 1.0 / 0.5 = 2.0.
        assert!(
            (comp.distance_gain() - 2.0).abs() < 0.01,
            "Distance gain should be 2.0 for source at 0.5m with HRTF at 1.0m"
        );
    }
}
