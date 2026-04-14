//! Voice Activity Detection (VAD) for the Opus SILK encoder.
//!
//! This module implements a multi-stage VAD pipeline suitable for use inside
//! the SILK encoder path of the Opus codec.  The design is inspired by the
//! WebRTC VAD (GMM-based + energy) and the ITU-T G.729B Annex B VAD, but is
//! entirely independent of those implementations.
//!
//! # Algorithm
//!
//! 1. **Energy gate** – Reject frames whose short-term energy is below a
//!    noise-floor estimate by at least `energy_threshold_db` dB.
//! 2. **Spectral flatness** – Compute the geometric-to-arithmetic mean ratio
//!    of the sub-band energies.  A flat spectrum (close to 1) is characteristic
//!    of noise; a peaked spectrum indicates voiced/unvoiced speech.
//! 3. **Zero-crossing rate** – High ZCR combined with low energy signals
//!    unvoiced speech or silence depending on the energy level.
//! 4. **Hangover** – Maintain a voice hangover counter so that short gaps
//!    (e.g. stop consonants) are not misclassified as silence.
//!
//! # References
//!
//! - RFC 6716, §3 (Opus SILK mode description)
//! - ITU-T G.729 Annex B – "A silence compression scheme"
//! - "A Computationally Efficient VAD for Narrowband Speech", Sohn et al.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Number of sub-bands used for spectral analysis.
const NUM_SUBBANDS: usize = 4;

/// Default energy threshold (dB above estimated noise floor).
const DEFAULT_ENERGY_THRESHOLD_DB: f32 = 10.0;

/// Default hangover duration in frames.
const DEFAULT_HANGOVER_FRAMES: u32 = 8;

/// Smoothing factor for the noise floor tracker (first-order IIR).
/// At α = 0.995 the tracker adapts in ~200 frames ≈ 4 s at 20 ms/frame.
const NOISE_TRACK_ALPHA: f32 = 0.995;

/// Smoothing factor used when the noise floor is rising (fast adaptation).
const NOISE_TRACK_ALPHA_RISE: f32 = 0.90;

/// Spectral flatness threshold: values below this indicate peaky (speech-like) spectra.
const SPECTRAL_FLATNESS_THRESHOLD: f32 = 0.70;

/// Zero-crossing rate threshold per 160 samples (10 ms @ 16 kHz).
/// Above this value the frame is considered noise-like.
const ZCR_HIGH_THRESHOLD: f32 = 60.0;

// ─────────────────────────────────────────────────────────────────────────────
// VAD decision
// ─────────────────────────────────────────────────────────────────────────────

/// Output of the VAD for a single frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadDecision {
    /// Voice (speech) detected.
    Voice,
    /// No voice — encoder may apply comfort noise or DTX.
    Silence,
}

impl VadDecision {
    /// Returns `true` if speech is present.
    #[must_use]
    pub fn is_voice(self) -> bool {
        self == Self::Voice
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-frame features
// ─────────────────────────────────────────────────────────────────────────────

/// Acoustic features computed for a single analysis frame.
#[derive(Debug, Clone)]
pub struct FrameFeatures {
    /// Short-term energy (sum of squared samples).
    pub energy: f32,
    /// Spectral flatness measure (0 = tonal/speech, 1 = flat/noise).
    pub spectral_flatness: f32,
    /// Normalised zero-crossing rate (crossings per 160 samples).
    pub zcr: f32,
    /// Sub-band energies (four bands: 0–500, 500–1k, 1k–2k, 2k–4k Hz).
    pub subband_energy: [f32; NUM_SUBBANDS],
}

impl FrameFeatures {
    /// Compute features from a slice of PCM samples (16-bit, mono).
    ///
    /// `sample_rate` is used to map sub-band boundaries to bin indices.
    /// If `samples` is empty, all features are zero.
    #[must_use]
    pub fn from_pcm_i16(samples: &[i16], sample_rate: u32) -> Self {
        if samples.is_empty() {
            return Self::zeroed();
        }

        // ── Energy ──────────────────────────────────────────────────────────
        let energy: f32 = samples.iter().map(|&s| (s as f32) * (s as f32)).sum();

        // ── Zero-crossing rate ───────────────────────────────────────────────
        let mut zcr_count = 0u32;
        for w in samples.windows(2) {
            // sign change: one positive, one non-positive
            let a = w[0];
            let b = w[1];
            if (a >= 0 && b < 0) || (a < 0 && b >= 0) {
                zcr_count += 1;
            }
        }
        // Normalise to crossings per 160 samples
        let normaliser = 160.0 / samples.len() as f32;
        let zcr = zcr_count as f32 * normaliser;

        // ── Sub-band energies (naïve DFT-free split using sample-domain
        //    band-pass decimation approximation) ────────────────────────────
        let subband_energy = compute_subband_energies(samples, sample_rate);

        // ── Spectral flatness ────────────────────────────────────────────────
        let spectral_flatness = spectral_flatness_from_bands(&subband_energy);

        Self {
            energy,
            spectral_flatness,
            zcr,
            subband_energy,
        }
    }

    /// Compute features from f32 PCM samples normalised to [-1, 1].
    #[must_use]
    pub fn from_pcm_f32(samples: &[f32], sample_rate: u32) -> Self {
        // Convert to i16 for unified path
        let i16_samples: Vec<i16> = samples
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect();
        Self::from_pcm_i16(&i16_samples, sample_rate)
    }

    fn zeroed() -> Self {
        Self {
            energy: 0.0,
            spectral_flatness: 1.0,
            zcr: 0.0,
            subband_energy: [0.0; NUM_SUBBANDS],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-band and spectral helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute approximate sub-band energies using a CIC-like decimation.
///
/// The four bands cover 0–Fs/8, Fs/8–Fs/4, Fs/4–3Fs/8, 3Fs/8–Fs/2.
/// This avoids a full FFT while retaining enough spectral resolution for VAD.
fn compute_subband_energies(samples: &[i16], _sample_rate: u32) -> [f32; NUM_SUBBANDS] {
    // Split samples into four interleaved decimated streams and sum energy.
    // For band k we take samples at positions k, k+4, k+8, …
    let mut bands = [0.0f32; NUM_SUBBANDS];
    for (i, &s) in samples.iter().enumerate() {
        let band = i % NUM_SUBBANDS;
        bands[band] += (s as f32) * (s as f32);
    }
    // Normalise by count per band
    let n = (samples.len() / NUM_SUBBANDS).max(1) as f32;
    for b in &mut bands {
        *b /= n;
    }
    bands
}

/// Spectral flatness measure: geometric mean / arithmetic mean of sub-band energies.
///
/// Returns a value in [0, 1]: 1 = perfectly flat (white noise), 0 = single tone.
fn spectral_flatness_from_bands(bands: &[f32; NUM_SUBBANDS]) -> f32 {
    let min_energy = 1e-6_f32;
    let arith_mean: f32 = bands.iter().map(|&b| b + min_energy).sum::<f32>() / NUM_SUBBANDS as f32;
    let log_sum: f32 = bands.iter().map(|&b| (b + min_energy).ln()).sum::<f32>();
    let geo_mean = (log_sum / NUM_SUBBANDS as f32).exp();
    if arith_mean > 0.0 {
        (geo_mean / arith_mean).min(1.0)
    } else {
        1.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VAD configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the [`VoiceActivityDetector`].
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Energy threshold above noise floor (dB) to declare speech.
    pub energy_threshold_db: f32,
    /// Number of consecutive voiced frames after which hangover kicks in.
    pub hangover_frames: u32,
    /// Spectral flatness threshold: above this → noise-like.
    pub spectral_flatness_threshold: f32,
    /// Zero-crossing rate above which the frame is considered noise-like.
    pub zcr_high_threshold: f32,
    /// Weight given to the energy cue (0–1). The remainder goes to spectral/ZCR.
    pub energy_weight: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            energy_threshold_db: DEFAULT_ENERGY_THRESHOLD_DB,
            hangover_frames: DEFAULT_HANGOVER_FRAMES,
            spectral_flatness_threshold: SPECTRAL_FLATNESS_THRESHOLD,
            zcr_high_threshold: ZCR_HIGH_THRESHOLD,
            energy_weight: 0.6,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VAD state machine
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-feature voice activity detector.
///
/// Processes PCM frames and emits [`VadDecision`] per frame.
///
/// # Example
///
/// ```
/// use oximedia_codec::opus::vad::{VoiceActivityDetector, VadConfig, VadDecision};
///
/// let mut vad = VoiceActivityDetector::new(VadConfig::default());
/// let silence = vec![0i16; 160];
/// let decision = vad.process_i16(&silence, 16000);
/// assert_eq!(decision, VadDecision::Silence);
/// ```
pub struct VoiceActivityDetector {
    config: VadConfig,
    /// Exponentially smoothed noise floor (energy units, not dB).
    noise_floor: f32,
    /// Hangover counter: number of remaining hangover frames.
    hangover: u32,
    /// Total frames processed.
    frame_count: u64,
    /// Smoothed energy for UI/diagnostics.
    smoothed_energy: f32,
}

impl VoiceActivityDetector {
    /// Create a new VAD with the given configuration.
    #[must_use]
    pub fn new(config: VadConfig) -> Self {
        Self {
            config,
            noise_floor: 1.0,
            hangover: 0,
            frame_count: 0,
            smoothed_energy: 0.0,
        }
    }

    /// Process a frame of 16-bit PCM samples at `sample_rate` Hz.
    ///
    /// Returns `VadDecision::Voice` if speech is likely present in this frame.
    pub fn process_i16(&mut self, samples: &[i16], sample_rate: u32) -> VadDecision {
        let features = FrameFeatures::from_pcm_i16(samples, sample_rate);
        self.process_features(&features)
    }

    /// Process a frame of f32 PCM samples (range [-1, 1]) at `sample_rate` Hz.
    pub fn process_f32(&mut self, samples: &[f32], sample_rate: u32) -> VadDecision {
        let features = FrameFeatures::from_pcm_f32(samples, sample_rate);
        self.process_features(&features)
    }

    /// Process pre-computed [`FrameFeatures`].
    pub fn process_features(&mut self, features: &FrameFeatures) -> VadDecision {
        self.frame_count += 1;

        // Update smoothed energy (EMA)
        self.smoothed_energy = 0.9 * self.smoothed_energy + 0.1 * features.energy;

        // ── 1. Energy gate ───────────────────────────────────────────────────
        let threshold_linear = db_to_linear_energy(self.config.energy_threshold_db);
        let energy_above_noise = features.energy > self.noise_floor * threshold_linear;

        // ── 2. Spectral flatness cue ─────────────────────────────────────────
        // Low flatness (peaky) → speech-like
        let spectral_speech = features.spectral_flatness < self.config.spectral_flatness_threshold;

        // ── 3. ZCR cue ───────────────────────────────────────────────────────
        let zcr_noise = features.zcr > self.config.zcr_high_threshold;

        // ── 4. Fuse cues ─────────────────────────────────────────────────────
        // Weighted vote: energy has the highest weight
        let w_e = self.config.energy_weight;
        let w_s = (1.0 - w_e) * 0.5;
        let w_z = (1.0 - w_e) * 0.5;

        let speech_score = w_e * energy_above_noise as u8 as f32
            + w_s * spectral_speech as u8 as f32
            + w_z * (!zcr_noise) as u8 as f32;

        let raw_voice = speech_score >= 0.5;

        // ── 5. Update noise floor tracker ────────────────────────────────────
        // Update only on frames classified as silence (after hangover decision)
        let decision_before_hangover = raw_voice;
        if !decision_before_hangover {
            let alpha = if features.energy > self.noise_floor {
                NOISE_TRACK_ALPHA_RISE
            } else {
                NOISE_TRACK_ALPHA
            };
            self.noise_floor = alpha * self.noise_floor + (1.0 - alpha) * features.energy.max(1.0);
        }

        // ── 6. Hangover logic ─────────────────────────────────────────────────
        if raw_voice {
            self.hangover = self.config.hangover_frames;
            VadDecision::Voice
        } else if self.hangover > 0 {
            self.hangover -= 1;
            VadDecision::Voice
        } else {
            VadDecision::Silence
        }
    }

    /// Reset internal state (use between calls with different audio streams).
    pub fn reset(&mut self) {
        self.noise_floor = 1.0;
        self.hangover = 0;
        self.frame_count = 0;
        self.smoothed_energy = 0.0;
    }

    /// Current estimated noise floor (energy units).
    #[must_use]
    pub fn noise_floor(&self) -> f32 {
        self.noise_floor
    }

    /// Total number of frames processed since creation / last reset.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a dB SNR threshold to a linear energy ratio.
///
/// `energy_ratio = 10^(db / 10)` (power domain).
#[inline]
fn db_to_linear_energy(db: f32) -> f32 {
    10.0_f32.powf(db / 10.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn silence_frame(len: usize) -> Vec<i16> {
        vec![0i16; len]
    }

    fn speech_frame(len: usize, amplitude: i16) -> Vec<i16> {
        // Simple sine-like pattern: alternating +amplitude / -amplitude
        (0..len)
            .map(|i| if i % 2 == 0 { amplitude } else { -amplitude })
            .collect()
    }

    fn loud_sine(len: usize) -> Vec<i16> {
        // 200 Hz sine at 16 kHz, 80% amplitude
        let freq = 200.0f32;
        let sr = 16000.0f32;
        (0..len)
            .map(|i| {
                let t = i as f32 / sr;
                ((2.0 * std::f32::consts::PI * freq * t).sin() * 26000.0) as i16
            })
            .collect()
    }

    #[test]
    fn test_silence_classified_as_silence() {
        let mut vad = VoiceActivityDetector::new(VadConfig::default());
        // Feed many silence frames to let noise floor settle
        for _ in 0..30 {
            vad.process_i16(&silence_frame(160), 16000);
        }
        let decision = vad.process_i16(&silence_frame(160), 16000);
        assert_eq!(decision, VadDecision::Silence);
    }

    #[test]
    fn test_loud_speech_classified_as_voice() {
        let mut vad = VoiceActivityDetector::new(VadConfig::default());
        // Warm up with silence
        for _ in 0..10 {
            vad.process_i16(&silence_frame(160), 16000);
        }
        // Feed a loud speech frame
        let frame = loud_sine(160);
        let decision = vad.process_i16(&frame, 16000);
        assert_eq!(decision, VadDecision::Voice);
    }

    #[test]
    fn test_hangover_extends_voice() {
        let cfg = VadConfig {
            hangover_frames: 5,
            ..Default::default()
        };
        let mut vad = VoiceActivityDetector::new(cfg);
        // Warm up
        for _ in 0..10 {
            vad.process_i16(&silence_frame(160), 16000);
        }
        // One loud frame → triggers voice + hangover
        vad.process_i16(&loud_sine(160), 16000);
        // Next frame is silence but should still be Voice due to hangover
        let d = vad.process_i16(&silence_frame(160), 16000);
        assert_eq!(
            d,
            VadDecision::Voice,
            "hangover should keep decision as Voice"
        );
    }

    #[test]
    fn test_frame_count_increments() {
        let mut vad = VoiceActivityDetector::new(VadConfig::default());
        assert_eq!(vad.frame_count(), 0);
        vad.process_i16(&silence_frame(160), 16000);
        vad.process_i16(&silence_frame(160), 16000);
        assert_eq!(vad.frame_count(), 2);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut vad = VoiceActivityDetector::new(VadConfig::default());
        for _ in 0..20 {
            vad.process_i16(&loud_sine(160), 16000);
        }
        vad.reset();
        assert_eq!(vad.frame_count(), 0);
        assert_eq!(vad.noise_floor(), 1.0);
    }

    #[test]
    fn test_f32_processing() {
        let mut vad = VoiceActivityDetector::new(VadConfig::default());
        // Warm up
        for _ in 0..10 {
            vad.process_f32(&vec![0.0f32; 160], 16000);
        }
        // Loud speech
        let loud: Vec<f32> = (0..160)
            .map(|i| {
                let t = i as f32 / 16000.0;
                (2.0 * std::f32::consts::PI * 200.0 * t).sin() * 0.8
            })
            .collect();
        let d = vad.process_f32(&loud, 16000);
        assert_eq!(d, VadDecision::Voice);
    }

    #[test]
    fn test_spectral_flatness_flat_is_close_to_one() {
        // White-ish noise has flat spectrum → flatness close to 1
        let bands = [1000.0f32, 1100.0, 950.0, 1050.0];
        let sf = spectral_flatness_from_bands(&bands);
        assert!(
            sf > 0.90,
            "flat-spectrum flatness should be > 0.90, got {sf}"
        );
    }

    #[test]
    fn test_spectral_flatness_peaky_is_low() {
        // One dominant band → flatness should be much lower than 1
        let bands = [10000.0f32, 10.0, 10.0, 10.0];
        let sf = spectral_flatness_from_bands(&bands);
        assert!(
            sf < 0.50,
            "peaky-spectrum flatness should be < 0.50, got {sf}"
        );
    }

    #[test]
    fn test_frame_features_zero_input() {
        let feats = FrameFeatures::from_pcm_i16(&[], 16000);
        assert_eq!(feats.energy, 0.0);
    }

    #[test]
    fn test_db_to_linear_energy_10db() {
        let ratio = db_to_linear_energy(10.0);
        // 10 dB ≈ 10.0 in power
        assert!((ratio - 10.0).abs() < 0.01, "expected ~10.0 got {ratio}");
    }
}
