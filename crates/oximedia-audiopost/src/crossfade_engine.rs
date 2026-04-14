//! Sample-accurate crossfade engine for seamless take splicing.
//!
//! Provides multiple crossfade curve shapes (linear, equal-power, S-curve / cosine,
//! logarithmic), a frame-level engine for applying the transition between two audio
//! buffers, and a splicing utility that concatenates multiple takes with crossfades.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use crate::error::{AudioPostError, AudioPostResult};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Crossfade curve shapes
// ---------------------------------------------------------------------------

/// Shape of the amplitude curve applied during a crossfade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrossfadeCurve {
    /// Linear fade-out / fade-in.
    Linear,
    /// Equal-power (constant-power) crossfade — `sqrt` of linear envelope.
    EqualPower,
    /// S-curve (cosine) crossfade — smooth first derivative at start and end.
    SCurve,
    /// Logarithmic crossfade (perceptually smooth).
    Logarithmic,
}

impl CrossfadeCurve {
    /// Evaluate the fade-in gain for a normalised position `t ∈ [0, 1]`.
    ///
    /// The corresponding fade-out gain is `1 - fade_in_gain(t)` for linear, or
    /// the mirror for the other curves.  Use [`CrossfadeCurve::gain_pair`] to
    /// obtain both simultaneously.
    #[must_use]
    pub fn fade_in_gain(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EqualPower => t.sqrt(),
            Self::SCurve => 0.5 * (1.0 - (std::f32::consts::PI * t).cos()),
            Self::Logarithmic => {
                // Map t through a log curve: gain = (10^t - 1) / 9
                (10.0_f32.powf(t) - 1.0) / 9.0
            }
        }
    }

    /// Evaluate the fade-out gain for a normalised position `t ∈ [0, 1]`.
    #[must_use]
    pub fn fade_out_gain(self, t: f32) -> f32 {
        self.fade_in_gain(1.0 - t)
    }

    /// Return `(fade_out_gain, fade_in_gain)` for position `t`.
    #[must_use]
    pub fn gain_pair(self, t: f32) -> (f32, f32) {
        (self.fade_out_gain(t), self.fade_in_gain(t))
    }

    /// A human-readable name for the curve.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::EqualPower => "Equal Power",
            Self::SCurve => "S-Curve (Cosine)",
            Self::Logarithmic => "Logarithmic",
        }
    }
}

// ---------------------------------------------------------------------------
// Crossfade region descriptor
// ---------------------------------------------------------------------------

/// Describes a single crossfade region between two takes / clips.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossfadeRegion {
    /// Duration of the crossfade in samples.
    pub length_samples: usize,
    /// Curve shape to use.
    pub curve: CrossfadeCurve,
    /// Gain applied to the outgoing clip (before the crossfade envelope).  1.0 = unity.
    pub outgoing_gain: f32,
    /// Gain applied to the incoming clip (before the crossfade envelope).  1.0 = unity.
    pub incoming_gain: f32,
}

impl CrossfadeRegion {
    /// Construct a new crossfade region.
    ///
    /// # Errors
    ///
    /// Returns an error if `length_samples` is zero or either gain is negative.
    pub fn new(
        length_samples: usize,
        curve: CrossfadeCurve,
        outgoing_gain: f32,
        incoming_gain: f32,
    ) -> AudioPostResult<Self> {
        if length_samples == 0 {
            return Err(AudioPostError::InvalidBufferSize(0));
        }
        if outgoing_gain < 0.0 {
            return Err(AudioPostError::InvalidGain(outgoing_gain));
        }
        if incoming_gain < 0.0 {
            return Err(AudioPostError::InvalidGain(incoming_gain));
        }
        Ok(Self {
            length_samples,
            curve,
            outgoing_gain,
            incoming_gain,
        })
    }

    /// Convenience constructor using unity gains.
    ///
    /// # Errors
    ///
    /// Returns an error if `length_samples` is zero.
    pub fn unity(length_samples: usize, curve: CrossfadeCurve) -> AudioPostResult<Self> {
        Self::new(length_samples, curve, 1.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Crossfade engine
// ---------------------------------------------------------------------------

/// Engine that applies crossfades between audio clips.
pub struct CrossfadeEngine {
    /// Default curve for new crossfades.
    pub default_curve: CrossfadeCurve,
}

impl CrossfadeEngine {
    /// Create a new engine with the given default curve.
    #[must_use]
    pub fn new(default_curve: CrossfadeCurve) -> Self {
        Self { default_curve }
    }

    /// Apply a crossfade between `outgoing` and `incoming` audio buffers.
    ///
    /// Both buffers must be mono and at least `region.length_samples` long.
    /// The output buffer has exactly `region.length_samples` samples representing
    /// the blended transition.
    ///
    /// # Errors
    ///
    /// Returns an error if either buffer is shorter than `region.length_samples`.
    pub fn apply(
        &self,
        outgoing: &[f32],
        incoming: &[f32],
        region: &CrossfadeRegion,
    ) -> AudioPostResult<Vec<f32>> {
        if outgoing.len() < region.length_samples {
            return Err(AudioPostError::InvalidBufferSize(outgoing.len()));
        }
        if incoming.len() < region.length_samples {
            return Err(AudioPostError::InvalidBufferSize(incoming.len()));
        }

        let n = region.length_samples;
        let mut output = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f32 / (n - 1).max(1) as f32;
            let (fade_out, fade_in) = region.curve.gain_pair(t);
            let out_sample = outgoing[i] * region.outgoing_gain * fade_out;
            let in_sample = incoming[i] * region.incoming_gain * fade_in;
            output.push(out_sample + in_sample);
        }

        Ok(output)
    }

    /// Apply a crossfade using the engine's default curve and unity gains.
    ///
    /// # Errors
    ///
    /// Returns an error if either buffer is shorter than `length_samples`.
    pub fn apply_default(
        &self,
        outgoing: &[f32],
        incoming: &[f32],
        length_samples: usize,
    ) -> AudioPostResult<Vec<f32>> {
        let region = CrossfadeRegion::unity(length_samples, self.default_curve)?;
        self.apply(outgoing, incoming, &region)
    }

    /// Splice two takes together with a crossfade, producing a single output buffer.
    ///
    /// The output consists of:
    /// 1. `outgoing[0 .. outgoing.len() - region.length_samples]`
    /// 2. The crossfade blend (region.length_samples samples)
    /// 3. `incoming[region.length_samples ..]`
    ///
    /// Both clips must be at least `region.length_samples` samples long.
    ///
    /// # Errors
    ///
    /// Returns an error if either buffer is shorter than the crossfade length.
    pub fn splice(
        &self,
        outgoing: &[f32],
        incoming: &[f32],
        region: &CrossfadeRegion,
    ) -> AudioPostResult<Vec<f32>> {
        if outgoing.len() < region.length_samples {
            return Err(AudioPostError::InvalidBufferSize(outgoing.len()));
        }
        if incoming.len() < region.length_samples {
            return Err(AudioPostError::InvalidBufferSize(incoming.len()));
        }

        let pre_len = outgoing.len() - region.length_samples;
        let xfade_out = &outgoing[pre_len..];
        let xfade_in = &incoming[..region.length_samples];
        let xfade = self.apply(xfade_out, xfade_in, region)?;

        let mut result = Vec::with_capacity(outgoing.len() + incoming.len() - region.length_samples);
        result.extend_from_slice(&outgoing[..pre_len]);
        result.extend_from_slice(&xfade);
        result.extend_from_slice(&incoming[region.length_samples..]);
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Multi-take splicer
// ---------------------------------------------------------------------------

/// A take to be spliced into the timeline.
#[derive(Debug, Clone)]
pub struct Take {
    /// Name / identifier for the take.
    pub name: String,
    /// Mono PCM samples.
    pub samples: Vec<f32>,
    /// Crossfade region to apply *after* this take (before the next one).
    /// If `None`, a hard cut is used at the join point.
    pub post_crossfade: Option<CrossfadeRegion>,
}

impl Take {
    /// Create a new take with no post-crossfade (hard cut).
    #[must_use]
    pub fn new(name: &str, samples: Vec<f32>) -> Self {
        Self {
            name: name.to_string(),
            samples,
            post_crossfade: None,
        }
    }

    /// Attach a crossfade region that will be used when transitioning to the next take.
    pub fn set_crossfade(&mut self, region: CrossfadeRegion) {
        self.post_crossfade = Some(region);
    }
}

/// Splice a sequence of takes into a single continuous buffer.
///
/// Takes are processed in order.  Where a post-crossfade is defined, the `CrossfadeEngine`
/// blends the tail of the outgoing take with the head of the incoming take.
///
/// # Errors
///
/// Returns an error if any take is too short for its crossfade, or if the take list
/// is empty.
pub fn splice_takes(
    takes: &[Take],
    engine: &CrossfadeEngine,
) -> AudioPostResult<Vec<f32>> {
    if takes.is_empty() {
        return Err(AudioPostError::Generic("take list is empty".to_string()));
    }

    let mut result: Vec<f32> = takes[0].samples.clone();

    for i in 1..takes.len() {
        let prev = &takes[i - 1];
        let curr = &takes[i];

        match &prev.post_crossfade {
            Some(region) => {
                // Splice the existing result with the new take using the crossfade.
                result = engine.splice(&result, &curr.samples, region)?;
            }
            None => {
                // Hard cut — just append.
                result.extend_from_slice(&curr.samples);
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Utility: generate a silence buffer
// ---------------------------------------------------------------------------

/// Create a silent buffer of `length_samples` samples (all zeros).
#[must_use]
pub fn silence(length_samples: usize) -> Vec<f32> {
    vec![0.0f32; length_samples]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(length: usize, freq: f32, sr: f32, amp: f32) -> Vec<f32> {
        (0..length)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect()
    }

    // --- CrossfadeCurve ---

    #[test]
    fn test_linear_gain_boundaries() {
        assert!((CrossfadeCurve::Linear.fade_in_gain(0.0)).abs() < 1e-6);
        assert!((CrossfadeCurve::Linear.fade_in_gain(1.0) - 1.0).abs() < 1e-6);
        assert!((CrossfadeCurve::Linear.fade_in_gain(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_equal_power_gain_boundaries() {
        assert!((CrossfadeCurve::EqualPower.fade_in_gain(0.0)).abs() < 1e-6);
        assert!((CrossfadeCurve::EqualPower.fade_in_gain(1.0) - 1.0).abs() < 1e-6);
        // At t=0.5, both channels should be at sqrt(0.5) ≈ 0.707
        let g = CrossfadeCurve::EqualPower.fade_in_gain(0.5);
        assert!((g - 0.5_f32.sqrt()).abs() < 1e-5, "g={g}");
    }

    #[test]
    fn test_scurve_gain_boundaries() {
        assert!((CrossfadeCurve::SCurve.fade_in_gain(0.0)).abs() < 1e-6);
        assert!((CrossfadeCurve::SCurve.fade_in_gain(1.0) - 1.0).abs() < 1e-5);
        // Midpoint of S-curve should be 0.5
        let g = CrossfadeCurve::SCurve.fade_in_gain(0.5);
        assert!((g - 0.5).abs() < 1e-5, "g={g}");
    }

    #[test]
    fn test_logarithmic_gain_boundaries() {
        assert!((CrossfadeCurve::Logarithmic.fade_in_gain(0.0)).abs() < 1e-5);
        assert!((CrossfadeCurve::Logarithmic.fade_in_gain(1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gain_pair_sums() {
        // For Linear: out + in should equal 1.0 at every point
        for &t in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let (fo, fi) = CrossfadeCurve::Linear.gain_pair(t);
            assert!((fo + fi - 1.0).abs() < 1e-5, "t={t}, fo={fo}, fi={fi}");
        }
    }

    // --- CrossfadeRegion ---

    #[test]
    fn test_crossfade_region_zero_length_error() {
        assert!(CrossfadeRegion::unity(0, CrossfadeCurve::Linear).is_err());
    }

    #[test]
    fn test_crossfade_region_negative_gain_error() {
        assert!(CrossfadeRegion::new(512, CrossfadeCurve::Linear, -1.0, 1.0).is_err());
        assert!(CrossfadeRegion::new(512, CrossfadeCurve::Linear, 1.0, -1.0).is_err());
    }

    // --- CrossfadeEngine ---

    #[test]
    fn test_apply_output_length() {
        let engine = CrossfadeEngine::new(CrossfadeCurve::Linear);
        let a = sine(1024, 440.0, 48000.0, 0.5);
        let b = sine(1024, 880.0, 48000.0, 0.5);
        let region = CrossfadeRegion::unity(512, CrossfadeCurve::Linear).expect("ok");
        let out = engine.apply(&a, &b, &region).expect("apply ok");
        assert_eq!(out.len(), 512);
    }

    #[test]
    fn test_apply_buffer_too_short_error() {
        let engine = CrossfadeEngine::new(CrossfadeCurve::EqualPower);
        let short = vec![0.0f32; 64];
        let long = sine(1024, 440.0, 48000.0, 0.5);
        let region = CrossfadeRegion::unity(512, CrossfadeCurve::EqualPower).expect("ok");
        assert!(engine.apply(&short, &long, &region).is_err());
        assert!(engine.apply(&long, &short, &region).is_err());
    }

    #[test]
    fn test_apply_crossfade_starts_at_outgoing() {
        // At t=0 the output should equal outgoing[0] (fade_in = 0).
        let engine = CrossfadeEngine::new(CrossfadeCurve::Linear);
        let out_buf = vec![1.0f32; 512];
        let in_buf = vec![0.5f32; 512];
        let region = CrossfadeRegion::unity(512, CrossfadeCurve::Linear).expect("ok");
        let result = engine.apply(&out_buf, &in_buf, &region).expect("ok");
        // First sample: fade_out=1, fade_in=0 → outgoing only
        assert!((result[0] - 1.0).abs() < 1e-5, "first={}", result[0]);
    }

    #[test]
    fn test_splice_output_length() {
        let engine = CrossfadeEngine::new(CrossfadeCurve::SCurve);
        let a = sine(2048, 440.0, 48000.0, 0.5);
        let b = sine(2048, 880.0, 48000.0, 0.5);
        let xfade_len = 512;
        let region = CrossfadeRegion::unity(xfade_len, CrossfadeCurve::SCurve).expect("ok");
        let out = engine.splice(&a, &b, &region).expect("splice ok");
        assert_eq!(out.len(), a.len() + b.len() - xfade_len);
    }

    #[test]
    fn test_splice_takes_hard_cut() {
        let engine = CrossfadeEngine::new(CrossfadeCurve::Linear);
        let t1 = Take::new("t1", vec![1.0f32; 1024]);
        let t2 = Take::new("t2", vec![0.5f32; 1024]);
        let out = splice_takes(&[t1, t2], &engine).expect("ok");
        assert_eq!(out.len(), 2048);
        // Hard cut: first 1024 samples should be 1.0
        assert!((out[1023] - 1.0).abs() < 1e-6);
        assert!((out[1024] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_splice_takes_with_crossfade() {
        let engine = CrossfadeEngine::new(CrossfadeCurve::EqualPower);
        let xfade_len = 256;
        let mut t1 = Take::new("t1", vec![1.0f32; 1024]);
        t1.set_crossfade(
            CrossfadeRegion::unity(xfade_len, CrossfadeCurve::EqualPower).expect("ok"),
        );
        let t2 = Take::new("t2", vec![0.0f32; 1024]);
        let out = splice_takes(&[t1, t2], &engine).expect("ok");
        assert_eq!(out.len(), 2048 - xfade_len);
    }

    #[test]
    fn test_splice_takes_empty_error() {
        let engine = CrossfadeEngine::new(CrossfadeCurve::Linear);
        assert!(splice_takes(&[], &engine).is_err());
    }

    #[test]
    fn test_silence_helper() {
        let buf = silence(1024);
        assert_eq!(buf.len(), 1024);
        assert!(buf.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_curve_names() {
        assert_eq!(CrossfadeCurve::Linear.name(), "Linear");
        assert_eq!(CrossfadeCurve::EqualPower.name(), "Equal Power");
        assert_eq!(CrossfadeCurve::SCurve.name(), "S-Curve (Cosine)");
        assert_eq!(CrossfadeCurve::Logarithmic.name(), "Logarithmic");
    }
}
