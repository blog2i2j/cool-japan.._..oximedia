//! 5.1 / 7.1 surround panning with VBAP-style gain computation.
//!
//! This module provides:
//!
//! - [`SurroundPanPosition`] — azimuth, elevation, and divergence parameters
//!   for a virtual source in 3-D space.
//! - [`SurroundPanner`] — computes per-speaker gain coefficients for 5.1
//!   (L/R/C/LFE/Ls/Rs) and 7.1 (L/R/C/LFE/Ls/Rs/Lrs/Rrs) layouts using a
//!   VBAP-style panning law.
//! - Configurable LFE send with crossover frequency (2nd-order Butterworth
//!   low-pass).
//!
//! ## Panning Law
//!
//! For each main (non-LFE) speaker, a cosine-based weight is computed from the
//! angular distance between the virtual source azimuth and the speaker azimuth.
//! The weights are squared for sharper localisation and then normalised so that
//! the sum of squares equals 1.0 (equal-power panning).
//!
//! Divergence controls how much of the localised energy is spread uniformly
//! across all main speakers:
//!
//! ```text
//! gain_final = localised * (1 − divergence) + uniform * divergence
//! ```
//!
//! ## LFE
//!
//! The LFE channel receives a configurable send level.  When crossover is
//! enabled, a 2nd-order Butterworth LPF (default 80 Hz) extracts bass content
//! for the LFE while keeping the main speakers full-band.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// SurroundPanPosition
// ---------------------------------------------------------------------------

/// Virtual source position for surround panning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurroundPanPosition {
    /// Horizontal angle in degrees.
    /// `0.0` = front centre, positive = clockwise (right), negative = left.
    /// Range: `−180.0..=180.0`.
    pub azimuth: f32,
    /// Vertical angle in degrees.
    /// `0.0` = ear level, `90.0` = directly above, `−90.0` = directly below.
    pub elevation: f32,
    /// Spatial divergence: `0.0` = point source (max localisation), `1.0` =
    /// omnidirectional (energy spread evenly across all speakers).
    pub divergence: f32,
}

impl SurroundPanPosition {
    /// Front centre, no divergence.
    #[must_use]
    pub fn center() -> Self {
        Self {
            azimuth: 0.0,
            elevation: 0.0,
            divergence: 0.0,
        }
    }

    /// Hard left (−90°).
    #[must_use]
    pub fn left() -> Self {
        Self {
            azimuth: -90.0,
            elevation: 0.0,
            divergence: 0.0,
        }
    }

    /// Hard right (+90°).
    #[must_use]
    pub fn right() -> Self {
        Self {
            azimuth: 90.0,
            elevation: 0.0,
            divergence: 0.0,
        }
    }

    /// Rear centre (180°).
    #[must_use]
    pub fn rear() -> Self {
        Self {
            azimuth: 180.0,
            elevation: 0.0,
            divergence: 0.0,
        }
    }

    /// Construct with all parameters.
    #[must_use]
    pub fn new(azimuth: f32, elevation: f32, divergence: f32) -> Self {
        Self {
            azimuth: azimuth.clamp(-180.0, 180.0),
            elevation: elevation.clamp(-90.0, 90.0),
            divergence: divergence.clamp(0.0, 1.0),
        }
    }
}

impl Default for SurroundPanPosition {
    fn default() -> Self {
        Self::center()
    }
}

// ---------------------------------------------------------------------------
// SurroundLayout
// ---------------------------------------------------------------------------

/// Target surround speaker layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurroundLayout {
    /// 5.1: L, R, C, LFE, Ls, Rs
    Layout51,
    /// 7.1: L, R, C, LFE, Ls, Rs, Lrs, Rrs
    Layout71,
}

impl SurroundLayout {
    /// Number of output channels.
    #[must_use]
    pub fn channel_count(self) -> usize {
        match self {
            Self::Layout51 => 6,
            Self::Layout71 => 8,
        }
    }

    /// Short label for each output channel.
    #[must_use]
    pub fn channel_names(self) -> &'static [&'static str] {
        match self {
            Self::Layout51 => &["L", "R", "C", "LFE", "Ls", "Rs"],
            Self::Layout71 => &["L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs"],
        }
    }

    /// Azimuth in degrees for each *main* speaker (LFE is excluded).
    /// Indices map to output channels, skipping LFE (index 3).
    fn main_speaker_azimuths(self) -> &'static [(usize, f32)] {
        match self {
            Self::Layout51 => &[
                (0, -30.0),  // L
                (1, 30.0),   // R
                (2, 0.0),    // C
                (4, -110.0), // Ls
                (5, 110.0),  // Rs
            ],
            Self::Layout71 => &[
                (0, -30.0),  // L
                (1, 30.0),   // R
                (2, 0.0),    // C
                (4, -110.0), // Ls
                (5, 110.0),  // Rs
                (6, -150.0), // Lrs
                (7, 150.0),  // Rrs
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// BiquadLpf (LFE crossover)
// ---------------------------------------------------------------------------

/// 2nd-order Butterworth low-pass filter for LFE crossover.
#[derive(Debug, Clone)]
struct BiquadLpf {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl BiquadLpf {
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * cutoff_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * std::f32::consts::SQRT_2); // Q = 1/sqrt(2)

        let b0 = (1.0 - cos_w0) / 2.0;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn tick(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// SurroundPanner
// ---------------------------------------------------------------------------

/// VBAP-style surround panner for 5.1 and 7.1 layouts.
#[derive(Debug, Clone)]
pub struct SurroundPanner {
    layout: SurroundLayout,
    /// LFE send level (0.0..=1.0).
    lfe_send: f32,
    /// LFE crossover enabled.
    lfe_crossover_enabled: bool,
    /// LFE crossover filter.
    lfe_lpf: BiquadLpf,
    /// LFE crossover frequency in Hz.
    lfe_crossover_freq: f32,
    /// Sample rate.
    sample_rate: f32,
}

impl SurroundPanner {
    /// Create a panner for the given layout and sample rate.
    ///
    /// LFE crossover defaults to 80 Hz and is enabled.
    #[must_use]
    pub fn new(layout: SurroundLayout, sample_rate: f32) -> Self {
        let crossover = 80.0;
        Self {
            layout,
            lfe_send: 0.1,
            lfe_crossover_enabled: true,
            lfe_lpf: BiquadLpf::new(crossover, sample_rate),
            lfe_crossover_freq: crossover,
            sample_rate,
        }
    }

    /// Current layout.
    #[must_use]
    pub fn layout(&self) -> SurroundLayout {
        self.layout
    }

    /// Set the LFE send level (0.0..=1.0).
    pub fn set_lfe_send(&mut self, level: f32) {
        self.lfe_send = level.clamp(0.0, 1.0);
    }

    /// Current LFE send level.
    #[must_use]
    pub fn lfe_send(&self) -> f32 {
        self.lfe_send
    }

    /// Set LFE crossover frequency in Hz (20..=200).
    pub fn set_lfe_crossover_freq(&mut self, freq: f32) {
        self.lfe_crossover_freq = freq.clamp(20.0, 200.0);
        self.lfe_lpf = BiquadLpf::new(self.lfe_crossover_freq, self.sample_rate);
    }

    /// Current LFE crossover frequency.
    #[must_use]
    pub fn lfe_crossover_freq(&self) -> f32 {
        self.lfe_crossover_freq
    }

    /// Enable or disable LFE crossover filtering.
    pub fn set_lfe_crossover_enabled(&mut self, enabled: bool) {
        self.lfe_crossover_enabled = enabled;
        if !enabled {
            self.lfe_lpf.reset();
        }
    }

    /// Whether LFE crossover is enabled.
    #[must_use]
    pub fn lfe_crossover_enabled(&self) -> bool {
        self.lfe_crossover_enabled
    }

    /// Reset LFE crossover filter state.
    pub fn reset(&mut self) {
        self.lfe_lpf.reset();
    }

    /// Compute per-speaker gains for a given position.
    ///
    /// Returns a `Vec<f32>` with `layout.channel_count()` elements.  The LFE
    /// channel (index 3) receives `lfe_send` (crossover is not applied here —
    /// it is handled in [`Self::pan_buffer`]).
    #[must_use]
    pub fn compute_gains(&self, pos: &SurroundPanPosition) -> Vec<f32> {
        let n = self.layout.channel_count();
        let speakers = self.layout.main_speaker_azimuths();
        let mut gains = vec![0.0_f32; n];

        // Step 1: cosine weights for main speakers.
        let mut sum_sq = 0.0_f32;
        for &(out_idx, spk_az_deg) in speakers {
            let diff_deg = pos.azimuth - spk_az_deg;
            let diff_rad = diff_deg * PI / 180.0;
            let mut g = diff_rad.cos().max(0.0);

            // Elevation: attenuate surround speakers for elevated sources.
            if spk_az_deg.abs() > 60.0 && pos.elevation.abs() > 0.0 {
                let elev_factor = (1.0 - (pos.elevation.abs() / 90.0).clamp(0.0, 1.0)).max(0.0);
                g *= elev_factor;
            }
            // Elevation boost for front speakers.
            if spk_az_deg.abs() <= 60.0 && pos.elevation > 0.0 {
                g *= 1.0 + 0.3 * (pos.elevation / 90.0).clamp(0.0, 1.0);
            }

            gains[out_idx] = g;
            sum_sq += g * g;
        }

        // Step 2: equal-power normalisation.
        if sum_sq > f32::EPSILON {
            let norm = sum_sq.sqrt().recip();
            for &(out_idx, _) in speakers {
                gains[out_idx] *= norm;
            }
        }

        // Step 3: divergence — blend localised gains with uniform spread.
        let div = pos.divergence.clamp(0.0, 1.0);
        if div > f32::EPSILON {
            let main_count = speakers.len() as f32;
            let uniform = if main_count > 0.0 {
                1.0 / main_count.sqrt()
            } else {
                0.0
            };
            for &(out_idx, _) in speakers {
                gains[out_idx] = gains[out_idx] * (1.0 - div) + uniform * div;
            }
        }

        // Step 4: LFE send.
        gains[3] = self.lfe_send;

        gains
    }

    /// Pan a mono sample to surround gains (convenience for a single sample).
    #[must_use]
    pub fn pan_sample(&self, sample: f32, pos: &SurroundPanPosition) -> Vec<f32> {
        self.compute_gains(pos)
            .into_iter()
            .map(|g| sample * g)
            .collect()
    }

    /// Pan a mono input buffer to per-speaker output buffers.
    ///
    /// Returns `layout.channel_count()` buffers, each of length `input.len()`.
    /// When LFE crossover is enabled, the LFE channel receives the low-pass
    /// filtered signal scaled by `lfe_send`.
    pub fn pan_buffer(
        &mut self,
        input: &[f32],
        pos: &SurroundPanPosition,
    ) -> Vec<Vec<f32>> {
        let n = self.layout.channel_count();
        let gains = self.compute_gains(pos);
        let mut outputs = vec![vec![0.0_f32; input.len()]; n];

        for (i, &sample) in input.iter().enumerate() {
            for ch in 0..n {
                if ch == 3 {
                    // LFE: use crossover-filtered signal.
                    let lfe_sample = if self.lfe_crossover_enabled {
                        self.lfe_lpf.tick(sample)
                    } else {
                        sample
                    };
                    outputs[3][i] = lfe_sample * gains[3];
                } else {
                    outputs[ch][i] = sample * gains[ch];
                }
            }
        }

        outputs
    }

    /// Pan a mono buffer using an interleaved output format.
    ///
    /// `out` must have length `>= input.len() * channel_count()`.
    pub fn pan_buffer_interleaved(
        &mut self,
        input: &[f32],
        pos: &SurroundPanPosition,
        out: &mut [f32],
    ) {
        let n = self.layout.channel_count();
        let gains = self.compute_gains(pos);

        for (frame, &sample) in input.iter().enumerate() {
            let base = frame * n;
            if base + n > out.len() {
                break;
            }
            for ch in 0..n {
                if ch == 3 {
                    let lfe_sample = if self.lfe_crossover_enabled {
                        self.lfe_lpf.tick(sample)
                    } else {
                        sample
                    };
                    out[base + ch] += lfe_sample * gains[3];
                } else {
                    out[base + ch] += sample * gains[ch];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn gains_sum_sq_main(gains: &[f32], layout: SurroundLayout) -> f32 {
        let speakers = layout.main_speaker_azimuths();
        speakers.iter().map(|&(i, _)| gains[i] * gains[i]).sum()
    }

    // -- position constructors ----------------------------------------------

    #[test]
    fn test_position_center() {
        let pos = SurroundPanPosition::center();
        assert!((pos.azimuth).abs() < f32::EPSILON);
        assert!((pos.elevation).abs() < f32::EPSILON);
        assert!((pos.divergence).abs() < f32::EPSILON);
    }

    #[test]
    fn test_position_clamping() {
        let pos = SurroundPanPosition::new(999.0, -999.0, 5.0);
        assert!((pos.azimuth - 180.0).abs() < f32::EPSILON);
        assert!((pos.elevation - (-90.0)).abs() < f32::EPSILON);
        assert!((pos.divergence - 1.0).abs() < f32::EPSILON);
    }

    // -- gain computation ---------------------------------------------------

    #[test]
    fn test_center_loudest_on_c_channel_51() {
        let panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        let gains = panner.compute_gains(&SurroundPanPosition::center());
        let c = gains[2];
        let l = gains[0];
        let r = gains[1];
        assert!(
            c > l,
            "C ({c}) should be louder than L ({l}) for centre source"
        );
        assert!(
            c > r,
            "C ({c}) should be louder than R ({r}) for centre source"
        );
    }

    #[test]
    fn test_left_speaker_loudest_for_left_source() {
        let panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        let pos = SurroundPanPosition::new(-30.0, 0.0, 0.0); // L speaker position
        let gains = panner.compute_gains(&pos);
        let l = gains[0];
        for (i, &g) in gains.iter().enumerate() {
            if i != 0 && i != 3 {
                assert!(
                    l >= g - 1e-6,
                    "L ({l}) should be >= ch{i} ({g}) for left source"
                );
            }
        }
    }

    #[test]
    fn test_equal_power_51() {
        let panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        let pos = SurroundPanPosition::new(45.0, 0.0, 0.0);
        let gains = panner.compute_gains(&pos);
        let ss = gains_sum_sq_main(&gains, SurroundLayout::Layout51);
        assert!(
            (ss - 1.0).abs() < 0.05,
            "Sum of squares should be ~1.0, got {ss}"
        );
    }

    #[test]
    fn test_equal_power_71() {
        let panner = SurroundPanner::new(SurroundLayout::Layout71, 48000.0);
        let pos = SurroundPanPosition::new(-60.0, 10.0, 0.0);
        let gains = panner.compute_gains(&pos);
        let ss = gains_sum_sq_main(&gains, SurroundLayout::Layout71);
        assert!(
            (ss - 1.0).abs() < 0.1,
            "Sum of squares should be ~1.0, got {ss}"
        );
    }

    // -- LFE ----------------------------------------------------------------

    #[test]
    fn test_lfe_send_level() {
        let mut panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        panner.set_lfe_send(0.5);
        let gains = panner.compute_gains(&SurroundPanPosition::center());
        assert!(
            (gains[3] - 0.5).abs() < f32::EPSILON,
            "LFE should equal lfe_send, got {}",
            gains[3]
        );
    }

    #[test]
    fn test_lfe_crossover_frequency_change() {
        let mut panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        panner.set_lfe_crossover_freq(120.0);
        assert!((panner.lfe_crossover_freq() - 120.0).abs() < f32::EPSILON);
    }

    // -- divergence ---------------------------------------------------------

    #[test]
    fn test_divergence_spreads_energy() {
        let panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        let focused = panner.compute_gains(&SurroundPanPosition::new(0.0, 0.0, 0.0));
        let spread = panner.compute_gains(&SurroundPanPosition::new(0.0, 0.0, 1.0));

        // With full divergence, gains should be more uniform.
        let focused_var = variance_main(&focused, SurroundLayout::Layout51);
        let spread_var = variance_main(&spread, SurroundLayout::Layout51);
        assert!(
            spread_var < focused_var + 1e-6,
            "Full divergence should reduce gain variance: focused={focused_var}, spread={spread_var}"
        );
    }

    // -- pan_buffer ---------------------------------------------------------

    #[test]
    fn test_pan_buffer_output_shape() {
        let mut panner = SurroundPanner::new(SurroundLayout::Layout71, 48000.0);
        let input = vec![0.5_f32; 32];
        let outputs = panner.pan_buffer(&input, &SurroundPanPosition::center());
        assert_eq!(outputs.len(), 8);
        for buf in &outputs {
            assert_eq!(buf.len(), 32);
        }
    }

    // -- pan_sample ---------------------------------------------------------

    #[test]
    fn test_pan_sample_length() {
        let panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        let out = panner.pan_sample(1.0, &SurroundPanPosition::center());
        assert_eq!(out.len(), 6);
    }

    // -- rear position ------------------------------------------------------

    #[test]
    fn test_rear_position_surround_dominance() {
        let panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        let gains = panner.compute_gains(&SurroundPanPosition::rear());
        let rear_energy = gains[4] * gains[4] + gains[5] * gains[5];
        let front_energy = gains[0] * gains[0] + gains[1] * gains[1] + gains[2] * gains[2];
        assert!(
            rear_energy >= front_energy,
            "Rear Ls/Rs energy ({rear_energy}) should dominate for rear source, front={front_energy}"
        );
    }

    // -- interleaved --------------------------------------------------------

    #[test]
    fn test_pan_buffer_interleaved() {
        let mut panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
        let input = vec![1.0_f32; 4];
        let mut out = vec![0.0_f32; 4 * 6];
        panner.pan_buffer_interleaved(&input, &SurroundPanPosition::center(), &mut out);
        let has_nonzero = out.iter().any(|&v| v.abs() > f32::EPSILON);
        assert!(has_nonzero, "interleaved output should have non-zero values");
    }

    // -- helpers ------------------------------------------------------------

    fn variance_main(gains: &[f32], layout: SurroundLayout) -> f32 {
        let speakers = layout.main_speaker_azimuths();
        let n = speakers.len() as f32;
        if n < 2.0 {
            return 0.0;
        }
        let mean: f32 = speakers.iter().map(|&(i, _)| gains[i]).sum::<f32>() / n;
        let var: f32 = speakers
            .iter()
            .map(|&(i, _)| {
                let d = gains[i] - mean;
                d * d
            })
            .sum::<f32>()
            / n;
        var
    }
}
