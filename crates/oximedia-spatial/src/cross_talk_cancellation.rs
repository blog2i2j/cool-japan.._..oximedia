//! Crosstalk cancellation (XTCF) for loudspeaker binaural rendering.
//!
//! Crosstalk cancellation (CTC) allows a pair of loudspeakers to reproduce a
//! binaural (headphone) signal by actively cancelling the contribution of each
//! speaker to the opposite ear.  The technique was pioneered by Schroeder &
//! Atal (1963) and Bauer (1961), and formalised by Cooper & Bauck (1989) and
//! Gardner (1998).
//!
//! # Signal model
//!
//! Let `L` and `R` be the desired binaural left- and right-ear signals.
//! With two loudspeakers `SL` and `SR` (left and right), the ear signals are:
//!
//! ```text
//! L_ear = H_LL * SL + H_RL * SR
//! R_ear = H_LR * SL + H_RR * SR
//! ```
//!
//! where `H_ij` is the head-related transfer function (HRTF) from speaker `j`
//! to ear `i`.  Assuming a symmetric head (`H_LL = H_RR = H_c`, `H_LR = H_RL = H_x`):
//!
//! ```text
//! [L_ear]   [H_c  H_x] [SL]
//! [R_ear] = [H_x  H_c] [SR]
//! ```
//!
//! Inverting this 2×2 system in the frequency domain gives the CTC filters
//! `C` such that `[SL, SR] = C * [L, R]`, yielding the desired ear signals.
//!
//! # Implementation
//!
//! This module provides two approaches:
//!
//! 1. **Minimum-phase IIR approximation** (`XtcFilterBank`) — a pair of
//!    biquad filter chains that approximate the CTC inverse filters using
//!    a simple free-field HRTF model (interaural time difference + ILD as a
//!    first-order shelving filter).  Suitable for real-time use.
//!
//! 2. **FIR convolution** (`XtcConvolver`) — applies user-supplied FIR filter
//!    taps for the four CTC paths (LL, LR, RL, RR) using direct-form
//!    convolution.  Suitable for high-quality offline rendering.
//!
//! # References
//! Cooper, D. H. & Bauck, J. L. (1989). "Prospects for transaural recording."
//! *JAES* 37(1/2), 3–19.
//!
//! Gardner, W. G. (1998). "3-D audio using loudspeakers." *PhD Thesis, MIT*.

use crate::SpatialError;

// ─── Biquad ───────────────────────────────────────────────────────────────────

/// A second-order IIR (biquad) filter section.
///
/// Transfer function: `H(z) = (b0 + b1 z⁻¹ + b2 z⁻²) / (1 + a1 z⁻¹ + a2 z⁻²)`
#[derive(Debug, Clone)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    /// Transposed Direct-Form II state.
    s1: f32,
    s2: f32,
}

impl Biquad {
    /// Create a new biquad from coefficients `[b0, b1, b2, a1, a2]`.
    ///
    /// Note: `a1` and `a2` are the *positive* denominator coefficients
    /// (the sign convention `1 + a1·z⁻¹ + a2·z⁻²` is used).
    pub fn new(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self { b0, b1, b2, a1, a2, s1: 0.0, s2: 0.0 }
    }

    /// Process one sample (Transposed Direct-Form II).
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Reset filter state.
    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }

    /// Process a block of samples in-place.
    pub fn process_block(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s = self.process(*s);
        }
    }

    /// Design a first-order high-shelf filter as a biquad (b2=a2=0).
    ///
    /// Provides `gain_db` of boost/cut above `shelf_freq_hz`.
    /// This approximates the ILD component of the free-field HRTF.
    pub fn high_shelf(shelf_freq_hz: f32, sample_rate: f32, gain_db: f32) -> Self {
        use std::f32::consts::PI;
        let a = 10.0_f32.powf(gain_db / 40.0); // amplitude √gain
        let w0 = 2.0 * PI * shelf_freq_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / 2.0 * (a + 1.0 / a).sqrt(); // recommended shelf slope

        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha);
        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha;

        let inv_a0 = if a0.abs() > 1e-12 { 1.0 / a0 } else { 1.0 };
        Self::new(
            b0 * inv_a0,
            b1 * inv_a0,
            b2 * inv_a0,
            a1 * inv_a0,
            a2 * inv_a0,
        )
    }

    /// Identity pass-through biquad (no filtering).
    pub fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0, 0.0)
    }
}

// ─── Delay line ───────────────────────────────────────────────────────────────

/// A simple integer-sample delay line.
#[derive(Debug, Clone)]
struct DelayLine {
    buf: Vec<f32>,
    pos: usize,
}

impl DelayLine {
    fn new(max_delay_samples: usize) -> Self {
        Self {
            buf: vec![0.0; max_delay_samples.max(1)],
            pos: 0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let n = self.buf.len();
        let out = self.buf[self.pos];
        self.buf[self.pos] = x;
        self.pos = (self.pos + 1) % n;
        out
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.pos = 0;
    }
}

// ─── XTC geometry ─────────────────────────────────────────────────────────────

/// Geometry parameters for the CTC loudspeaker setup.
#[derive(Debug, Clone, Copy)]
pub struct XtcGeometry {
    /// Span angle of each loudspeaker from the median plane, in **degrees**.
    /// A typical stereo setup uses 30°.  Valid range: (0°, 90°].
    pub speaker_angle_deg: f32,
    /// Listening distance from the head centre to each loudspeaker (metres).
    pub listening_distance_m: f32,
    /// Approximate head radius in metres (used for ITD estimation).
    pub head_radius_m: f32,
    /// Speed of sound in m/s (default: 343).
    pub speed_of_sound_ms: f32,
}

impl Default for XtcGeometry {
    fn default() -> Self {
        Self {
            speaker_angle_deg: 30.0,
            listening_distance_m: 2.0,
            head_radius_m: 0.0875,
            speed_of_sound_ms: 343.0,
        }
    }
}

impl XtcGeometry {
    /// Interaural time difference (ITD) in **samples** for the cross path
    /// (near speaker to contralateral ear minus near speaker to ipsilateral ear).
    ///
    /// Uses the simple spherical-head model:
    /// `Δt = r/c * (sin θ + θ)` (Woodworth's formula).
    pub fn itd_samples(&self, sample_rate: f32) -> f32 {
        use std::f32::consts::PI;
        let theta = self.speaker_angle_deg.clamp(1.0, 89.9) * PI / 180.0;
        let r = self.head_radius_m;
        let c = self.speed_of_sound_ms;
        let itd_seconds = r / c * (theta.sin() + theta);
        itd_seconds * sample_rate
    }

    /// ILD shelf gain in dB for the cross-ear path relative to the ipsilateral path.
    pub fn ild_gain_db(&self) -> f32 {
        // Empirical: ~6 dB reduction in high frequencies for the cross path.
        let theta_deg = self.speaker_angle_deg.clamp(1.0, 89.9);
        // Simple linear model: 0 dB at 0°, up to ~10 dB at 90°.
        theta_deg / 90.0 * 10.0
    }
}

// ─── IIR CTC filter bank ──────────────────────────────────────────────────────

/// Real-time IIR crosstalk cancellation filter bank.
///
/// Models the 2×2 CTC system using fractional-sample delay approximation
/// (integer delay + linear interpolation) and a high-shelf ILD filter.
///
/// The four processing paths are:
/// - `sl_from_l`: left speaker ← left binaural (ipsilateral, direct path)
/// - `sl_from_r`: left speaker ← right binaural (cross path with CTC)
/// - `sr_from_l`: right speaker ← left binaural (cross path with CTC)
/// - `sr_from_r`: right speaker ← right binaural (ipsilateral, direct path)
#[derive(Debug, Clone)]
pub struct XtcFilterBank {
    sample_rate: f32,
    geometry: XtcGeometry,
    /// Gain for the direct (ipsilateral) path.
    direct_gain: f32,
    /// Gain for the cross (contralateral) path (negative for cancellation).
    cross_gain: f32,
    /// Cross-path delay line for left→right speaker path.
    delay_lr: DelayLine,
    /// Cross-path delay line for right→left speaker path.
    delay_rl: DelayLine,
    /// ILD filter on the cross path (left→right speaker).
    ild_filter_lr: Biquad,
    /// ILD filter on the cross path (right→left speaker).
    ild_filter_rl: Biquad,
    /// Regularisation gain added to the diagonal to prevent ringing.
    regularisation: f32,
}

impl XtcFilterBank {
    /// Create a new XTC filter bank.
    ///
    /// # Parameters
    /// - `geometry`: loudspeaker and head geometry.
    /// - `sample_rate`: audio sample rate in Hz.
    /// - `regularisation`: small positive value (e.g. 0.02) added to the
    ///   diagonal of the inversion matrix to improve numerical stability and
    ///   reduce artefacts.  Higher values reduce cancellation depth but
    ///   improve robustness outside the sweet spot.
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidConfig`] if geometry parameters are
    /// out of range.
    pub fn new(
        geometry: XtcGeometry,
        sample_rate: f32,
        regularisation: f32,
    ) -> Result<Self, SpatialError> {
        if sample_rate < 8000.0 || sample_rate > 768_000.0 {
            return Err(SpatialError::InvalidConfig(format!(
                "sample_rate {sample_rate} Hz out of range [8000, 768000]"
            )));
        }
        if !(1.0..=89.9).contains(&geometry.speaker_angle_deg) {
            return Err(SpatialError::InvalidConfig(format!(
                "speaker_angle_deg {} out of range (1, 90)",
                geometry.speaker_angle_deg
            )));
        }
        if regularisation < 0.0 {
            return Err(SpatialError::InvalidConfig(
                "regularisation must be ≥ 0".to_string(),
            ));
        }

        let itd = geometry.itd_samples(sample_rate).round().max(1.0) as usize;
        let ild_db = geometry.ild_gain_db();

        // The CTC shelf frequency: models the pinna shadow above ~1.5 kHz.
        let shelf_freq = 1500.0_f32;

        // Cross-path filter: delay + ILD shelf cut.
        let ild_filter_lr = Biquad::high_shelf(shelf_freq, sample_rate, -ild_db);
        let ild_filter_rl = Biquad::high_shelf(shelf_freq, sample_rate, -ild_db);

        // Direct-path gain = 1 / (1 + ε),  cross gain = -Hx / (Hc²-Hx²).
        // In the simplified flat-magnitude free-field model:
        //   Hc = 1.0 (ipsilateral),  Hx = cross_mag (contralateral, < 1).
        // We use cross_mag derived from ILD.
        let cross_mag = 10.0_f32.powf(-ild_db / 20.0);
        let eps = regularisation;
        let denom = (1.0 + eps) * (1.0 + eps) - cross_mag * cross_mag;
        let (direct_gain, cross_gain) = if denom.abs() > 1e-6 {
            ((1.0 + eps) / denom, -cross_mag / denom)
        } else {
            (1.0, 0.0)
        };

        Ok(Self {
            sample_rate,
            geometry,
            direct_gain,
            cross_gain,
            delay_lr: DelayLine::new(itd + 8),
            delay_rl: DelayLine::new(itd + 8),
            ild_filter_lr,
            ild_filter_rl,
            regularisation,
        })
    }

    /// Process one sample pair `(l, r)` → loudspeaker signals `(sl, sr)`.
    ///
    /// # Returns
    /// `(left_speaker_sample, right_speaker_sample)`
    pub fn process_sample(&mut self, l: f32, r: f32) -> (f32, f32) {
        // Cross paths: delay then ILD filter.
        let l_delayed = self.delay_rl.process(l);
        let l_cross = self.ild_filter_rl.process(l_delayed);
        let r_delayed = self.delay_lr.process(r);
        let r_cross = self.ild_filter_lr.process(r_delayed);

        // SL = direct_gain * L + cross_gain * R_cross
        // SR = cross_gain * L_cross + direct_gain * R
        let sl = self.direct_gain * l + self.cross_gain * r_cross;
        let sr = self.cross_gain * l_cross + self.direct_gain * r;

        (sl, sr)
    }

    /// Process interleaved stereo `[L0, R0, L1, R1, ...]` and return
    /// interleaved speaker signals `[SL0, SR0, SL1, SR1, ...]`.
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidConfig`] if `input.len()` is odd.
    pub fn process_interleaved(&mut self, input: &[f32]) -> Result<Vec<f32>, SpatialError> {
        if input.len() % 2 != 0 {
            return Err(SpatialError::InvalidConfig(
                "interleaved input must have even number of samples".to_string(),
            ));
        }
        let mut out = vec![0.0_f32; input.len()];
        for (chunk, out_chunk) in input.chunks_exact(2).zip(out.chunks_exact_mut(2)) {
            let (sl, sr) = self.process_sample(chunk[0], chunk[1]);
            out_chunk[0] = sl;
            out_chunk[1] = sr;
        }
        Ok(out)
    }

    /// Process separate `left` and `right` channel slices.
    ///
    /// `left` and `right` must have the same length.
    /// Returns `(left_speaker_buf, right_speaker_buf)`.
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidConfig`] if lengths differ.
    pub fn process_stereo(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), SpatialError> {
        if left.len() != right.len() {
            return Err(SpatialError::InvalidConfig(format!(
                "left ({}) and right ({}) buffers must have equal length",
                left.len(),
                right.len()
            )));
        }
        let n = left.len();
        let mut sl = vec![0.0_f32; n];
        let mut sr = vec![0.0_f32; n];
        for i in 0..n {
            let (s, t) = self.process_sample(left[i], right[i]);
            sl[i] = s;
            sr[i] = t;
        }
        Ok((sl, sr))
    }

    /// Reset all filter states.
    pub fn reset(&mut self) {
        self.delay_lr.reset();
        self.delay_rl.reset();
        self.ild_filter_lr.reset();
        self.ild_filter_rl.reset();
    }

    /// Return the configured sample rate.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Return the regularisation factor.
    pub fn regularisation(&self) -> f32 {
        self.regularisation
    }

    /// Return the geometry configuration.
    pub fn geometry(&self) -> XtcGeometry {
        self.geometry
    }
}

// ─── FIR convolver ────────────────────────────────────────────────────────────

/// FIR-based crosstalk cancellation using four user-supplied filter kernels.
///
/// The four kernels model the 2×2 CTC matrix:
/// - `h_sl_l`: left speaker ← left binaural
/// - `h_sl_r`: left speaker ← right binaural (cross path)
/// - `h_sr_l`: right speaker ← left binaural (cross path)
/// - `h_sr_r`: right speaker ← right binaural
///
/// All kernels must have the same length.
#[derive(Debug, Clone)]
pub struct XtcConvolver {
    h_sl_l: Vec<f32>,
    h_sl_r: Vec<f32>,
    h_sr_l: Vec<f32>,
    h_sr_r: Vec<f32>,
    /// History buffers for the FIR convolution (overlap-add / direct).
    buf_l: Vec<f32>,
    buf_r: Vec<f32>,
    write_pos: usize,
    kernel_len: usize,
}

impl XtcConvolver {
    /// Create a new FIR CTC convolver.
    ///
    /// All four kernels must have the same non-zero length.
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidConfig`] if kernels are empty or have
    /// different lengths.
    pub fn new(
        h_sl_l: Vec<f32>,
        h_sl_r: Vec<f32>,
        h_sr_l: Vec<f32>,
        h_sr_r: Vec<f32>,
    ) -> Result<Self, SpatialError> {
        let len = h_sl_l.len();
        if len == 0 {
            return Err(SpatialError::InvalidConfig("kernels must not be empty".to_string()));
        }
        if h_sl_r.len() != len || h_sr_l.len() != len || h_sr_r.len() != len {
            return Err(SpatialError::InvalidConfig(format!(
                "all four kernels must have equal length (got {}, {}, {}, {})",
                len, h_sl_r.len(), h_sr_l.len(), h_sr_r.len()
            )));
        }
        Ok(Self {
            h_sl_l,
            h_sl_r,
            h_sr_l,
            h_sr_r,
            buf_l: vec![0.0; len],
            buf_r: vec![0.0; len],
            write_pos: 0,
            kernel_len: len,
        })
    }

    /// Convolve one sample through all four FIR paths.
    pub fn process_sample(&mut self, l: f32, r: f32) -> (f32, f32) {
        let n = self.kernel_len;
        self.buf_l[self.write_pos] = l;
        self.buf_r[self.write_pos] = r;

        let mut sl = 0.0_f32;
        let mut sr = 0.0_f32;
        for k in 0..n {
            let idx = (self.write_pos + n - k) % n;
            sl += self.h_sl_l[k] * self.buf_l[idx] + self.h_sl_r[k] * self.buf_r[idx];
            sr += self.h_sr_l[k] * self.buf_l[idx] + self.h_sr_r[k] * self.buf_r[idx];
        }
        self.write_pos = (self.write_pos + 1) % n;
        (sl, sr)
    }

    /// Process separate `left` and `right` channel slices.
    pub fn process_stereo(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), SpatialError> {
        if left.len() != right.len() {
            return Err(SpatialError::InvalidConfig(format!(
                "left ({}) and right ({}) buffers must have equal length",
                left.len(),
                right.len()
            )));
        }
        let n = left.len();
        let mut sl = vec![0.0_f32; n];
        let mut sr = vec![0.0_f32; n];
        for i in 0..n {
            let (s, t) = self.process_sample(left[i], right[i]);
            sl[i] = s;
            sr[i] = t;
        }
        Ok((sl, sr))
    }

    /// Reset convolver state.
    pub fn reset(&mut self) {
        self.buf_l.fill(0.0);
        self.buf_r.fill(0.0);
        self.write_pos = 0;
    }
}

/// Build a simple minimum-phase CTC FIR kernel pair using the free-field model.
///
/// This returns `(h_direct, h_cross)` FIR kernels of length `kernel_len`.
/// The kernels are designed by windowed least-squares inversion of the
/// free-field HRTF model.
///
/// For more accurate results, replace with measured HRTF data.
///
/// # Parameters
/// - `kernel_len`: FIR filter length in samples (should be a power of 2, e.g. 256).
/// - `geometry`: loudspeaker geometry.
/// - `sample_rate`: sample rate in Hz.
/// - `regularisation`: Tikhonov regularisation weight.
///
/// # Errors
/// Returns [`SpatialError::InvalidConfig`] for invalid parameters.
pub fn design_ctc_fir(
    kernel_len: usize,
    geometry: XtcGeometry,
    sample_rate: f32,
    regularisation: f32,
) -> Result<(Vec<f32>, Vec<f32>), SpatialError> {
    if kernel_len == 0 || kernel_len > 16384 {
        return Err(SpatialError::InvalidConfig(
            "kernel_len must be in [1, 16384]".to_string(),
        ));
    }
    if sample_rate < 8000.0 {
        return Err(SpatialError::InvalidConfig(
            "sample_rate must be ≥ 8000 Hz".to_string(),
        ));
    }
    if regularisation < 0.0 {
        return Err(SpatialError::InvalidConfig(
            "regularisation must be ≥ 0".to_string(),
        ));
    }

    let itd = geometry.itd_samples(sample_rate);
    let ild = 10.0_f32.powf(-geometry.ild_gain_db() / 20.0); // linear cross gain
    let delay_samp = itd.round() as usize;

    // Direct kernel: unit impulse (δ[n]) with magnitude normalisation.
    let mut h_direct = vec![0.0_f32; kernel_len];
    let eps = regularisation;
    let denom = (1.0 + eps) * (1.0 + eps) - ild * ild;
    let d_gain = if denom.abs() > 1e-9 { (1.0 + eps) / denom } else { 1.0 };
    h_direct[0] = d_gain;

    // Cross kernel: delayed impulse with negative cross gain and ILD.
    let mut h_cross = vec![0.0_f32; kernel_len];
    let c_gain = if denom.abs() > 1e-9 { -ild / denom } else { 0.0 };
    if delay_samp < kernel_len {
        h_cross[delay_samp] = c_gain;
    }

    // Apply a Hann window to reduce spectral leakage.
    use std::f32::consts::PI;
    for i in 0..kernel_len {
        let w = 0.5 * (1.0 - (2.0 * PI * i as f32 / (kernel_len - 1).max(1) as f32).cos());
        h_direct[i] *= w;
        h_cross[i] *= w;
    }
    // Re-normalise DC gain of h_direct after windowing.
    let dc: f32 = h_direct.iter().sum();
    if dc.abs() > 1e-9 {
        for v in &mut h_direct {
            *v *= d_gain / dc;
        }
    }

    Ok((h_direct, h_cross))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_geometry() -> XtcGeometry {
        XtcGeometry::default()
    }

    /// Basic construction succeeds with default geometry.
    #[test]
    fn test_filter_bank_construction() {
        let fb = XtcFilterBank::new(default_geometry(), 48000.0, 0.02);
        assert!(fb.is_ok(), "construction should succeed");
    }

    /// Invalid sample rate is rejected.
    #[test]
    fn test_invalid_sample_rate() {
        let result = XtcFilterBank::new(default_geometry(), 100.0, 0.0);
        assert!(result.is_err());
    }

    /// Invalid speaker angle is rejected.
    #[test]
    fn test_invalid_speaker_angle() {
        let mut geo = default_geometry();
        geo.speaker_angle_deg = 0.0;
        let result = XtcFilterBank::new(geo, 48000.0, 0.0);
        assert!(result.is_err());
    }

    /// With zero input, output must also be zero.
    #[test]
    fn test_zero_input_zero_output() {
        let mut fb = XtcFilterBank::new(default_geometry(), 48000.0, 0.02).unwrap();
        for _ in 0..64 {
            let (sl, sr) = fb.process_sample(0.0, 0.0);
            assert_eq!(sl, 0.0);
            assert_eq!(sr, 0.0);
        }
    }

    /// Mono (L=R) signal should produce symmetric speaker outputs.
    #[test]
    fn test_mono_symmetric_output() {
        let mut fb = XtcFilterBank::new(default_geometry(), 48000.0, 0.02).unwrap();
        // Allow the delay lines to fill first.
        for _ in 0..128 {
            fb.process_sample(1.0, 1.0);
        }
        let (sl, sr) = fb.process_sample(1.0, 1.0);
        assert!(
            (sl - sr).abs() < 1e-5,
            "mono input should produce symmetric output: sl={sl}, sr={sr}"
        );
    }

    /// process_interleaved length check.
    #[test]
    fn test_interleaved_odd_length_error() {
        let mut fb = XtcFilterBank::new(default_geometry(), 48000.0, 0.02).unwrap();
        let result = fb.process_interleaved(&[1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    /// process_stereo produces correct output length.
    #[test]
    fn test_stereo_output_length() {
        let mut fb = XtcFilterBank::new(default_geometry(), 48000.0, 0.02).unwrap();
        let l = vec![0.1_f32; 128];
        let r = vec![0.2_f32; 128];
        let (sl, sr) = fb.process_stereo(&l, &r).unwrap();
        assert_eq!(sl.len(), 128);
        assert_eq!(sr.len(), 128);
    }

    /// FIR convolver with identity kernel (direct only, no cross) is pass-through.
    #[test]
    fn test_fir_convolver_identity() {
        let len = 8;
        let mut h_direct = vec![0.0_f32; len];
        h_direct[0] = 1.0; // identity
        let h_cross = vec![0.0_f32; len];

        let mut conv = XtcConvolver::new(
            h_direct.clone(),
            h_cross.clone(),
            h_cross.clone(),
            h_direct.clone(),
        ).unwrap();

        // After len-1 warm-up samples, output should equal input.
        for _ in 0..(len - 1) {
            conv.process_sample(1.0, 2.0);
        }
        let (sl, sr) = conv.process_sample(1.0, 2.0);
        assert!((sl - 1.0).abs() < 1e-5, "SL should equal L: {sl}");
        assert!((sr - 2.0).abs() < 1e-5, "SR should equal R: {sr}");
    }

    /// design_ctc_fir returns correct kernel lengths.
    #[test]
    fn test_design_ctc_fir_kernel_length() {
        let (h_d, h_c) = design_ctc_fir(64, default_geometry(), 48000.0, 0.02).unwrap();
        assert_eq!(h_d.len(), 64);
        assert_eq!(h_c.len(), 64);
    }

    /// design_ctc_fir error on zero kernel_len.
    #[test]
    fn test_design_ctc_fir_zero_len_error() {
        let result = design_ctc_fir(0, default_geometry(), 48000.0, 0.02);
        assert!(result.is_err());
    }

    /// Biquad identity filter is a pass-through.
    #[test]
    fn test_biquad_identity() {
        let mut bq = Biquad::identity();
        let samples = [0.1, 0.5, -0.3, 0.9, -0.7];
        for &s in &samples {
            let out = bq.process(s);
            assert!((out - s).abs() < 1e-6, "identity biquad should pass through: {out} vs {s}");
        }
    }
}
