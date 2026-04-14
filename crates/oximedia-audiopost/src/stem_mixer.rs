//! Stem-based stereo mixing.
//!
//! A **stem** is a mono audio track belonging to a named category (drums, bass,
//! melody, vocals, etc.).  A [`StemMix`] holds any number of stems and can
//! render them to a stereo output pair using per-stem gain, pan, mute, and solo
//! controls.
//!
//! # Panning law
//!
//! The module uses a **constant-power (sin/cos) pan law** so that a centred
//! signal has the same perceived loudness as a fully-panned signal:
//!
//! ```text
//! left_gain  = cos((pan + 1) * π/4)
//! right_gain = sin((pan + 1) * π/4)
//! ```
//!
//! where `pan` ranges from −1.0 (full left) through 0.0 (centre) to +1.0 (full right).

use std::fmt;
use thiserror::Error;

// ─── StemType ─────────────────────────────────────────────────────────────────

/// Category of an audio stem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StemType {
    /// Drum kit or percussion.
    Drums,
    /// Bass instruments (bass guitar, synth bass, etc.).
    Bass,
    /// Melodic instruments (keys, guitars, strings, etc.).
    Melody,
    /// Lead or backing vocals.
    Vocals,
    /// Sound effects or incidental audio.
    FX,
    /// Recorded spoken dialogue.
    Dialogue,
    /// Full music bed / pre-mix.
    Music,
    /// Environmental or atmospheric ambience.
    Ambience,
}

impl fmt::Display for StemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Drums => "Drums",
            Self::Bass => "Bass",
            Self::Melody => "Melody",
            Self::Vocals => "Vocals",
            Self::FX => "FX",
            Self::Dialogue => "Dialogue",
            Self::Music => "Music",
            Self::Ambience => "Ambience",
        };
        write!(f, "{name}")
    }
}

// ─── StemError ────────────────────────────────────────────────────────────────

/// Errors that can occur during stem mixing operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum StemError {
    /// No stem of the requested type exists in the mix.
    #[error("Stem not found: {0}")]
    StemNotFound(StemType),

    /// Two stems have different sample lengths and cannot be mixed.
    #[error("Stem length mismatch")]
    LengthMismatch,

    /// The requested gain value is outside the allowed range.
    #[error("Invalid gain: {0} dB (must be in [-96, 24])")]
    InvalidGain(f32),

    /// The requested pan value is outside the allowed range.
    #[error("Invalid pan: {0} (must be in [-1.0, 1.0])")]
    InvalidPan(f32),
}

// ─── Stem ─────────────────────────────────────────────────────────────────────

/// A single audio stem with routing controls.
#[derive(Debug, Clone)]
pub struct Stem {
    /// Category of this stem.
    pub stem_type: StemType,
    /// Mono audio samples.
    pub samples: Vec<f32>,
    /// Linear gain applied during mix (stored in dB, applied as linear).
    pub gain_db: f32,
    /// Pan position: −1.0 = full left, 0.0 = centre, +1.0 = full right.
    pub pan: f32,
    /// When `true`, this stem contributes no audio to the output mix.
    pub muted: bool,
    /// When `true`, only soloed stems are included in the output mix.
    pub solo: bool,
}

impl Stem {
    /// Create a new stem.
    ///
    /// # Arguments
    ///
    /// * `stem_type` — category label.
    /// * `samples`   — mono audio data.
    ///
    /// The stem starts at 0 dB gain, centred pan, unmuted and unsoloed.
    #[must_use]
    pub fn new(stem_type: StemType, samples: Vec<f32>) -> Self {
        Self {
            stem_type,
            samples,
            gain_db: 0.0,
            pan: 0.0,
            muted: false,
            solo: false,
        }
    }
}

// ─── StemMix ──────────────────────────────────────────────────────────────────

/// A collection of stems that can be rendered to stereo.
#[derive(Debug, Clone)]
pub struct StemMix {
    /// All stems registered in this mix.
    pub stems: Vec<Stem>,
    /// Sample rate of all stems (they must share the same rate).
    pub sample_rate: u32,
}

impl StemMix {
    /// Create an empty stem mix.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            stems: Vec::new(),
            sample_rate,
        }
    }

    /// Add a stem to the mix.
    ///
    /// If a stem of the same type already exists it is **replaced**.
    pub fn add_stem(&mut self, stem: Stem) {
        // Replace existing stem of the same type if present.
        if let Some(existing) = self
            .stems
            .iter_mut()
            .find(|s| s.stem_type == stem.stem_type)
        {
            *existing = stem;
        } else {
            self.stems.push(stem);
        }
    }

    /// Remove the stem with the given type.
    ///
    /// Does nothing if no such stem exists.
    pub fn remove_stem(&mut self, stem_type: StemType) {
        self.stems.retain(|s| s.stem_type != stem_type);
    }

    /// Set the gain (dB) for a stem.
    ///
    /// # Errors
    ///
    /// Returns [`StemError::StemNotFound`] when no stem of `stem_type` exists,
    /// or [`StemError::InvalidGain`] when `gain_db` is outside `[-96, 24]`.
    pub fn set_gain(&mut self, stem_type: StemType, gain_db: f32) -> Result<(), StemError> {
        if !(-96.0..=24.0).contains(&gain_db) {
            return Err(StemError::InvalidGain(gain_db));
        }
        let stem = self
            .stems
            .iter_mut()
            .find(|s| s.stem_type == stem_type)
            .ok_or(StemError::StemNotFound(stem_type))?;
        stem.gain_db = gain_db;
        Ok(())
    }

    /// Set the pan position for a stem.
    ///
    /// # Errors
    ///
    /// Returns [`StemError::StemNotFound`] or [`StemError::InvalidPan`].
    pub fn set_pan(&mut self, stem_type: StemType, pan: f32) -> Result<(), StemError> {
        if !(-1.0..=1.0).contains(&pan) {
            return Err(StemError::InvalidPan(pan));
        }
        let stem = self
            .stems
            .iter_mut()
            .find(|s| s.stem_type == stem_type)
            .ok_or(StemError::StemNotFound(stem_type))?;
        stem.pan = pan;
        Ok(())
    }

    /// Render all active stems to a stereo output pair `(left, right)`.
    ///
    /// **Solo semantics**: if *any* stem has `solo == true`, only soloed stems
    /// contribute to the output.  Otherwise all un-muted stems contribute.
    ///
    /// The output length equals the length of the longest stem.  Shorter stems
    /// are zero-padded implicitly.
    ///
    /// Returns `(vec![], vec![])` when there are no stems or every active stem
    /// is empty.
    #[must_use]
    pub fn render_stereo(&self) -> (Vec<f32>, Vec<f32>) {
        // Determine which stems are active.
        let any_solo = self.stems.iter().any(|s| s.solo);
        let active: Vec<&Stem> = self
            .stems
            .iter()
            .filter(|s| {
                if any_solo {
                    s.solo && !s.muted
                } else {
                    !s.muted
                }
            })
            .collect();

        if active.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let output_len = active.iter().map(|s| s.samples.len()).max().unwrap_or(0);
        if output_len == 0 {
            return (Vec::new(), Vec::new());
        }

        let mut left = vec![0.0f32; output_len];
        let mut right = vec![0.0f32; output_len];

        for stem in &active {
            let linear_gain = db_to_linear(stem.gain_db);
            // Constant-power pan law
            let angle = (stem.pan + 1.0) * std::f32::consts::FRAC_PI_4;
            let left_gain = linear_gain * angle.cos();
            let right_gain = linear_gain * angle.sin();

            for (i, &s) in stem.samples.iter().enumerate() {
                left[i] += s * left_gain;
                right[i] += s * right_gain;
            }
        }

        (left, right)
    }
}

// ─── DSP utility ──────────────────────────────────────────────────────────────

#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const FS: u32 = 48_000;

    fn dc_stem(stem_type: StemType, amplitude: f32, n: usize) -> Stem {
        Stem::new(stem_type, vec![amplitude; n])
    }

    // ── Empty mix returns empty output ─────────────────────────────────────

    #[test]
    fn empty_mix_returns_empty_output() {
        let mix = StemMix::new(FS);
        let (l, r) = mix.render_stereo();
        assert!(l.is_empty());
        assert!(r.is_empty());
    }

    // ── Single centred stem splits signal equally ──────────────────────────

    #[test]
    fn centred_stem_splits_equally() {
        let mut mix = StemMix::new(FS);
        mix.add_stem(dc_stem(StemType::Dialogue, 1.0, 100));
        let (l, r) = mix.render_stereo();
        // cos(π/4) == sin(π/4) == 1/√2 ≈ 0.7071
        for i in 0..100 {
            let diff = (l[i] - r[i]).abs();
            assert!(diff < 1e-5, "L/R should be equal at sample {i}: L={} R={}", l[i], r[i]);
        }
    }

    // ── Pan hard left → all energy in left channel ────────────────────────

    #[test]
    fn pan_hard_left_sends_signal_to_left() {
        let mut mix = StemMix::new(FS);
        let mut stem = dc_stem(StemType::Bass, 1.0, 64);
        stem.pan = -1.0;
        mix.add_stem(stem);
        let (l, r) = mix.render_stereo();
        // cos(0) = 1, sin(0) = 0
        for i in 0..64 {
            assert!(l[i].abs() > 0.99, "left should be near 1.0 at {i}");
            assert!(r[i].abs() < 1e-5, "right should be near 0.0 at {i}");
        }
    }

    // ── Pan hard right → all energy in right channel ──────────────────────

    #[test]
    fn pan_hard_right_sends_signal_to_right() {
        let mut mix = StemMix::new(FS);
        let mut stem = dc_stem(StemType::Melody, 1.0, 64);
        stem.pan = 1.0;
        mix.add_stem(stem);
        let (l, r) = mix.render_stereo();
        // cos(π/2) = 0, sin(π/2) = 1
        for i in 0..64 {
            assert!(l[i].abs() < 1e-5, "left should be near 0.0 at {i}");
            assert!(r[i].abs() > 0.99, "right should be near 1.0 at {i}");
        }
    }

    // ── Gain: -6 dB attenuates by ~half ───────────────────────────────────

    #[test]
    fn gain_minus_6db_halves_amplitude() {
        let mut mix = StemMix::new(FS);
        mix.add_stem(dc_stem(StemType::Vocals, 1.0, 64));
        mix.set_gain(StemType::Vocals, -6.0207).expect("valid gain");
        let (l, r) = mix.render_stereo();
        let expected = db_to_linear(-6.0207) * (std::f32::consts::FRAC_PI_4).cos();
        for i in 0..64 {
            let diff = (l[i] - expected).abs();
            assert!(diff < 1e-3, "sample {i}: expected ~{expected} got {}", l[i]);
            let diff_r = (r[i] - expected).abs();
            assert!(diff_r < 1e-3, "sample {i}: R expected ~{expected} got {}", r[i]);
        }
    }

    // ── Mute: muted stem contributes nothing ──────────────────────────────

    #[test]
    fn muted_stem_silent_in_output() {
        let mut mix = StemMix::new(FS);
        let mut stem = dc_stem(StemType::Drums, 1.0, 64);
        stem.muted = true;
        mix.add_stem(stem);
        let (l, r) = mix.render_stereo();
        // No active stems → empty
        assert!(l.is_empty(), "muted only mix should be empty");
        assert!(r.is_empty());
    }

    // ── Solo: only soloed stem heard ───────────────────────────────────────

    #[test]
    fn solo_stem_only_outputs_soloed_stem() {
        let mut mix = StemMix::new(FS);
        // Drums at 1.0 amplitude (soloed)
        let mut drums = dc_stem(StemType::Drums, 1.0, 64);
        drums.solo = true;
        // Bass at 0.5 amplitude (not soloed)
        let bass = dc_stem(StemType::Bass, 0.5, 64);
        mix.add_stem(drums);
        mix.add_stem(bass);
        let (l, r) = mix.render_stereo();
        // Only drums contribute; bass is silenced.
        let expected_l = 1.0_f32 * std::f32::consts::FRAC_PI_4.cos();
        for i in 0..64 {
            let diff = (l[i] - expected_l).abs();
            assert!(diff < 1e-4, "solo: sample {i} L={} expected ~{expected_l}", l[i]);
            let _ = r[i]; // just ensure no panic
        }
    }

    // ── set_gain validation ────────────────────────────────────────────────

    #[test]
    fn set_gain_out_of_range_returns_error() {
        let mut mix = StemMix::new(FS);
        mix.add_stem(dc_stem(StemType::FX, 1.0, 8));
        assert!(matches!(
            mix.set_gain(StemType::FX, 100.0),
            Err(StemError::InvalidGain(_))
        ));
    }

    // ── set_pan validation ────────────────────────────────────────────────

    #[test]
    fn set_pan_out_of_range_returns_error() {
        let mut mix = StemMix::new(FS);
        mix.add_stem(dc_stem(StemType::Ambience, 1.0, 8));
        assert!(matches!(
            mix.set_pan(StemType::Ambience, 2.0),
            Err(StemError::InvalidPan(_))
        ));
    }

    // ── Missing stem returns correct error ────────────────────────────────

    #[test]
    fn set_gain_on_missing_stem_returns_not_found() {
        let mut mix = StemMix::new(FS);
        assert!(matches!(
            mix.set_gain(StemType::Music, 0.0),
            Err(StemError::StemNotFound(StemType::Music))
        ));
    }

    // ── Remove stem ───────────────────────────────────────────────────────

    #[test]
    fn remove_stem_reduces_count() {
        let mut mix = StemMix::new(FS);
        mix.add_stem(dc_stem(StemType::Dialogue, 1.0, 8));
        mix.add_stem(dc_stem(StemType::Music, 1.0, 8));
        assert_eq!(mix.stems.len(), 2);
        mix.remove_stem(StemType::Dialogue);
        assert_eq!(mix.stems.len(), 1);
        assert_eq!(mix.stems[0].stem_type, StemType::Music);
    }
}
