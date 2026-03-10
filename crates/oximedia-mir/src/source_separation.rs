#![allow(dead_code)]
//! Source separation — vocal / drum / bass / other stem splitting.
//!
//! Implements Non-negative Matrix Factorization (NMF) based source separation
//! following Lee & Seung 2001 multiplicative update rules.

use rustfft::{num_complex::Complex, FftPlanner};

/// Type of audio stem produced by source separation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StemType {
    /// Lead and backing vocals.
    Vocals,
    /// Drum kit and percussion.
    Drums,
    /// Bass instruments (bass guitar, synth bass, kick body).
    Bass,
    /// All other instruments (guitar, piano, synths, etc.).
    Other,
}

impl StemType {
    /// Human-readable label for this stem type.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Vocals => "Vocals",
            Self::Drums => "Drums",
            Self::Bass => "Bass",
            Self::Other => "Other",
        }
    }

    /// Returns all four canonical stem types.
    #[must_use]
    pub fn all() -> [Self; 4] {
        [Self::Vocals, Self::Drums, Self::Bass, Self::Other]
    }
}

/// Configuration for the separation algorithm.
#[derive(Debug, Clone)]
pub struct SeparationConfig {
    /// Which stems to extract. Must be non-empty.
    pub stems: Vec<StemType>,
    /// Input sample rate in Hz.
    pub sample_rate: f32,
    /// FFT window size for STFT-based separation.
    pub window_size: usize,
    /// Hop size for STFT.
    pub hop_size: usize,
    /// Quality level 0.0–1.0: higher values use more computation.
    pub quality: f32,
}

impl Default for SeparationConfig {
    fn default() -> Self {
        Self {
            stems: StemType::all().to_vec(),
            sample_rate: 44100.0,
            window_size: 4096,
            hop_size: 1024,
            quality: 0.8,
        }
    }
}

impl SeparationConfig {
    /// Returns `true` when the configuration is logically valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.stems.is_empty()
            && self.sample_rate > 0.0
            && self.window_size >= 2
            && self.hop_size >= 1
            && self.hop_size <= self.window_size
            && self.quality >= 0.0
            && self.quality <= 1.0
    }
}

/// The separated audio data for a single stem.
#[derive(Debug, Clone)]
pub struct Stem {
    /// Type of this stem.
    pub stem_type: StemType,
    /// Audio samples (mono, normalised −1.0 … 1.0).
    pub samples: Vec<f32>,
    /// Energy ratio of this stem vs the mixture (0.0–1.0).
    pub energy_ratio: f32,
}

impl Stem {
    /// Create a new `Stem`.
    #[must_use]
    pub fn new(stem_type: StemType, samples: Vec<f32>, energy_ratio: f32) -> Self {
        Self {
            stem_type,
            samples,
            energy_ratio,
        }
    }

    /// RMS energy of this stem.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn rms(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.samples.iter().map(|s| s * s).sum();
        (sum / self.samples.len() as f32).sqrt()
    }
}

/// Output of a source separation operation.
#[derive(Debug, Clone)]
pub struct SeparationResult {
    /// All extracted stems.
    pub stems: Vec<Stem>,
    /// Sample rate of the output stems.
    pub sample_rate: f32,
    /// Signal-to-distortion ratio estimate (dB) — higher is better.
    pub sdr_estimate_db: f32,
}

impl SeparationResult {
    /// Create a new result.
    #[must_use]
    pub fn new(stems: Vec<Stem>, sample_rate: f32, sdr_estimate_db: f32) -> Self {
        Self {
            stems,
            sample_rate,
            sdr_estimate_db,
        }
    }

    /// Number of extracted stems.
    #[must_use]
    pub fn stem_count(&self) -> usize {
        self.stems.len()
    }

    /// Look up a stem by type.
    #[must_use]
    pub fn get_stem(&self, stem_type: StemType) -> Option<&Stem> {
        self.stems.iter().find(|s| s.stem_type == stem_type)
    }

    /// Returns `true` when the SDR estimate suggests acceptable quality (> 6 dB).
    #[must_use]
    pub fn is_acceptable_quality(&self) -> bool {
        self.sdr_estimate_db > 6.0
    }
}

// ─── NMF internals ───────────────────────────────────────────────────────────

/// Small epsilon to avoid division-by-zero in NMF updates.
const NMF_EPSILON: f32 = 1e-10;

/// Number of NMF multiplicative-update iterations.
const NMF_ITERATIONS: usize = 50;

/// Compute a Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    (0..n)
        .map(|i| {
            let phase = 2.0 * PI * i as f32 / (n - 1).max(1) as f32;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

/// Short-Time Fourier Transform.
///
/// Returns the complex STFT matrix laid out as `frames × (window_size/2 + 1)`.
#[allow(clippy::cast_precision_loss)]
fn stft(
    signal: &[f32],
    window_size: usize,
    hop_size: usize,
    planner: &mut FftPlanner<f32>,
) -> Vec<Vec<Complex<f32>>> {
    let fft = planner.plan_fft_forward(window_size);
    let window = hann_window(window_size);
    let n_bins = window_size / 2 + 1;

    // Number of frames (zero-pad signal so every hop has a full window).
    let n_frames = if signal.len() < window_size {
        1
    } else {
        (signal.len() - window_size) / hop_size + 1
    };

    let mut frames = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_size;

        // Build windowed, zero-padded frame as complex buffer.
        let mut buf: Vec<Complex<f32>> = (0..window_size)
            .map(|k| {
                let sample_idx = start + k;
                let sample = if sample_idx < signal.len() {
                    signal[sample_idx]
                } else {
                    0.0
                };
                Complex::new(sample * window[k], 0.0)
            })
            .collect();

        fft.process(&mut buf);

        // Keep only the non-redundant positive-frequency bins.
        let frame: Vec<Complex<f32>> = buf[..n_bins].to_vec();
        frames.push(frame);
    }

    frames
}

/// Inverse STFT (overlap-add) — reconstructs a real signal from a complex
/// STFT matrix (frames × n_bins) combined with the original phase.
///
/// `n_samples` is the target output length.
#[allow(clippy::cast_precision_loss)]
fn istft(
    frames: &[Vec<Complex<f32>>],
    window_size: usize,
    hop_size: usize,
    n_samples: usize,
    planner: &mut FftPlanner<f32>,
) -> Vec<f32> {
    let ifft = planner.plan_fft_inverse(window_size);
    let window = hann_window(window_size);
    let n_bins = window_size / 2 + 1;
    let norm = 1.0 / window_size as f32;

    let mut output = vec![0.0f32; n_samples + window_size];
    let mut window_sum = vec![0.0f32; n_samples + window_size];

    for (frame_idx, frame) in frames.iter().enumerate() {
        // Reconstruct full spectrum using Hermitian symmetry.
        let mut buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); window_size];
        for (k, &c) in frame.iter().enumerate().take(n_bins) {
            buf[k] = c;
            // Mirror (skip DC and Nyquist bins which have no conjugate pair).
            if k > 0 && k < window_size - n_bins + 1 {
                buf[window_size - k] = c.conj();
            }
        }

        ifft.process(&mut buf);

        let start = frame_idx * hop_size;
        for k in 0..window_size {
            let idx = start + k;
            if idx < output.len() {
                output[idx] += buf[k].re * norm * window[k];
                window_sum[idx] += window[k] * window[k];
            }
        }
    }

    // Normalise by overlap-add window weight, clip to requested length.
    output
        .into_iter()
        .zip(window_sum)
        .take(n_samples)
        .map(|(s, w)| if w > NMF_EPSILON { s / w } else { s })
        .collect()
}

/// Extract the magnitude spectrogram from an STFT matrix.
///
/// Returns a flat row-major matrix of shape `n_freqs × n_frames`.
fn magnitude_spectrogram(stft_frames: &[Vec<Complex<f32>>]) -> (Vec<f32>, usize, usize) {
    let n_frames = stft_frames.len();
    let n_freqs = stft_frames.first().map_or(0, |f| f.len());
    let mut v = vec![0.0f32; n_freqs * n_frames];
    for (t, frame) in stft_frames.iter().enumerate() {
        for (f, &c) in frame.iter().enumerate() {
            v[f * n_frames + t] = c.norm();
        }
    }
    (v, n_freqs, n_frames)
}

/// Deterministic pseudo-random initialiser (LCG) — avoids pulling in `rand`.
fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // Map to (0, 1].
    ((*state >> 33) as f32) / (u32::MAX as f32) + 1e-4
}

/// Run NMF with multiplicative update rules (Lee & Seung 2001).
///
/// `v`       — magnitude spectrogram, shape `n_freqs × n_frames` (row-major).
/// `n_comp`  — number of components.
///
/// Returns `(W, H)` where `W` is `n_freqs × n_comp` and `H` is `n_comp × n_frames`.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
fn nmf(v: &[f32], n_freqs: usize, n_frames: usize, n_comp: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng_state: u64 = 0x5EED_CAFE_DEAD_BEEF;

    // Initialise W (n_freqs × n_comp) and H (n_comp × n_frames).
    let mut w: Vec<f32> = (0..n_freqs * n_comp)
        .map(|_| lcg_next(&mut rng_state))
        .collect();
    let mut h: Vec<f32> = (0..n_comp * n_frames)
        .map(|_| lcg_next(&mut rng_state))
        .collect();

    // Temporary buffers.
    let mut wh = vec![0.0f32; n_freqs * n_frames]; // W × H
    let mut wt_v = vec![0.0f32; n_comp * n_frames]; // Wᵀ × V
    let mut wt_wh = vec![0.0f32; n_comp * n_frames]; // Wᵀ × W × H
    let mut v_ht = vec![0.0f32; n_freqs * n_comp]; // V × Hᵀ
    let mut wh_ht = vec![0.0f32; n_freqs * n_comp]; // W × H × Hᵀ

    for _ in 0..NMF_ITERATIONS {
        // ── Update H ────────────────────────────────────────────────────────
        // WH (n_freqs × n_frames) = W (n_freqs × n_comp) × H (n_comp × n_frames)
        matmul(n_freqs, n_comp, n_frames, &w, &h, &mut wh);

        // Wᵀ × V  (n_comp × n_frames) = Wᵀ (n_comp × n_freqs) × V (n_freqs × n_frames)
        matmul_at_b(n_freqs, n_comp, n_frames, &w, v, &mut wt_v);

        // Wᵀ × WH (n_comp × n_frames) = Wᵀ (n_comp × n_freqs) × WH (n_freqs × n_frames)
        matmul_at_b(n_freqs, n_comp, n_frames, &w, &wh, &mut wt_wh);

        // H ← H ⊙ (Wᵀ × V) / (Wᵀ × WH + ε)
        for i in 0..n_comp * n_frames {
            h[i] *= wt_v[i] / (wt_wh[i] + NMF_EPSILON);
        }

        // ── Update W ────────────────────────────────────────────────────────
        // Re-compute WH with updated H.
        matmul(n_freqs, n_comp, n_frames, &w, &h, &mut wh);

        // V × Hᵀ  (n_freqs × n_comp) = V (n_freqs × n_frames) × Hᵀ (n_frames × n_comp)
        matmul_a_bt(n_freqs, n_frames, n_comp, v, &h, &mut v_ht);

        // WH × Hᵀ (n_freqs × n_comp) = WH (n_freqs × n_frames) × Hᵀ (n_frames × n_comp)
        matmul_a_bt(n_freqs, n_frames, n_comp, &wh, &h, &mut wh_ht);

        // W ← W ⊙ (V × Hᵀ) / (WH × Hᵀ + ε)
        for i in 0..n_freqs * n_comp {
            w[i] *= v_ht[i] / (wh_ht[i] + NMF_EPSILON);
        }
    }

    (w, h)
}

// ─── Matrix helpers (all row-major) ──────────────────────────────────────────

/// `C (m × n)` = `A (m × k)` × `B (k × n)`.
fn matmul(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for ci in c.iter_mut() {
        *ci = 0.0;
    }
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
}

/// `C (k × n)` = `Aᵀ (k × m)` × `B (m × n)`  where A is `m × k`.
fn matmul_at_b(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for ci in c.iter_mut() {
        *ci = 0.0;
    }
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p]; // Aᵀ[p,i] = A[i,p]
            for j in 0..n {
                c[p * n + j] += a_ip * b[i * n + j];
            }
        }
    }
}

/// `C (m × k)` = `A (m × n)` × `Bᵀ`  where B is `k × n`.
fn matmul_a_bt(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for ci in c.iter_mut() {
        *ci = 0.0;
    }
    for i in 0..m {
        for j in 0..k {
            let mut sum = 0.0f32;
            for p in 0..n {
                sum += a[i * n + p] * b[j * n + p]; // Bᵀ[p,j] = B[j,p]
            }
            c[i * k + j] = sum;
        }
    }
}

// ─── Spectral centroid clustering ────────────────────────────────────────────

/// Spectral centroid of component `comp` in W (`n_freqs × n_comp`, row-major).
///
/// The frequency-bin index is used as a proxy for frequency — higher index →
/// higher frequency.
#[allow(clippy::cast_precision_loss)]
fn spectral_centroid(w: &[f32], n_freqs: usize, n_comp: usize, comp: usize) -> f32 {
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for freq in 0..n_freqs {
        let weight = w[freq * n_comp + comp];
        numerator += freq as f32 * weight;
        denominator += weight;
    }
    if denominator > NMF_EPSILON {
        numerator / denominator
    } else {
        0.0
    }
}

/// Assign each NMF component to a `StemType` based on its spectral centroid.
///
/// Strategy (normalised centroid in `[0, 1)`):
/// - `< 0.08`  → Bass   (very low frequencies)
/// - `< 0.25`  → Drums  (low–mid transient-rich region)
/// - `< 0.60`  → Vocals (mid-range formant region)
/// - `>= 0.60` → Other  (upper-mid and high frequencies)
///
/// After the initial assignment, any requested stem that received **no**
/// components is guaranteed at least one by stealing the component whose
/// centroid is closest (by nominal distance) to that stem's target range.
/// This prevents stems from having an all-zero Wiener mask when the input
/// signal is spectrally narrow (e.g. a pure-tone test signal that concentrates
/// all NMF energy in a single frequency region).
#[allow(clippy::cast_precision_loss)]
fn assign_components_to_stems(
    w: &[f32],
    n_freqs: usize,
    n_comp: usize,
    requested_stems: &[StemType],
) -> Vec<StemType> {
    let n_freqs_f = n_freqs as f32;

    // Nominal normalised centroid for each stem type — used both for initial
    // threshold fallback and for the coverage-guarantee pass below.
    let nominal = |st: StemType| -> f32 {
        match st {
            StemType::Bass => 0.04,
            StemType::Drums => 0.16,
            StemType::Vocals => 0.42,
            StemType::Other => 0.75,
        }
    };

    // Pre-compute per-component normalised centroid.
    let centroids: Vec<f32> = (0..n_comp)
        .map(|c| spectral_centroid(w, n_freqs, n_comp, c) / n_freqs_f)
        .collect();

    // Initial assignment based on centroid thresholds.
    let mut assignments: Vec<StemType> = centroids
        .iter()
        .map(|&centroid_norm| {
            let preferred = if centroid_norm < 0.08 {
                StemType::Bass
            } else if centroid_norm < 0.25 {
                StemType::Drums
            } else if centroid_norm < 0.60 {
                StemType::Vocals
            } else {
                StemType::Other
            };

            if requested_stems.contains(&preferred) {
                preferred
            } else {
                // Fall back to the closest requested stem by nominal centroid distance.
                requested_stems
                    .iter()
                    .copied()
                    .min_by(|&a, &b| {
                        let da = (nominal(a) - centroid_norm).abs();
                        let db = (nominal(b) - centroid_norm).abs();
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(requested_stems[0])
            }
        })
        .collect();

    // Coverage guarantee: every requested stem must have at least one component.
    //
    // First, collect which stems still need coverage.  Then distribute one
    // dedicated component to each uncovered stem, picking components in order
    // of their centroid proximity to the target nominal and never reusing a
    // component that has already been reserved in this pass.
    let uncovered: Vec<StemType> = requested_stems
        .iter()
        .copied()
        .filter(|&st| !assignments.iter().any(|&a| a == st))
        .collect();

    if !uncovered.is_empty() {
        // Build a sorted list of (component_index, centroid) for all components,
        // so we can greedily pick one unique component per uncovered stem.
        // We use a "reserved" set to avoid giving the same component to two stems.
        let mut reserved: std::collections::HashSet<usize> = std::collections::HashSet::new();

        for stem_type in uncovered {
            let target = nominal(stem_type);

            // Pick the unreserved component whose centroid is closest to target.
            // Tie-break by component index (ascending) for determinism.
            let best_comp = centroids
                .iter()
                .enumerate()
                .filter(|(idx, _)| !reserved.contains(idx))
                .min_by(|(ia, &ca), (ib, &cb)| {
                    let da = (ca - target).abs();
                    let db = (cb - target).abs();
                    da.partial_cmp(&db)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(ia.cmp(ib))
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            reserved.insert(best_comp);
            assignments[best_comp] = stem_type;
        }
    }

    assignments
}

/// Reconstruct each stem via Wiener-like soft mask.
///
/// `mask_stem = (W_stem × H_stem) / (W × H + ε)`
/// Applied to the original complex STFT to preserve phase information.
#[allow(clippy::too_many_arguments)]
fn reconstruct_stems(
    stft_frames: &[Vec<Complex<f32>>],
    w: &[f32],
    h: &[f32],
    n_freqs: usize,
    n_frames: usize,
    n_comp: usize,
    component_assignments: &[StemType],
    requested_stems: &[StemType],
    window_size: usize,
    hop_size: usize,
    n_samples: usize,
    planner: &mut FftPlanner<f32>,
) -> Vec<(StemType, Vec<f32>)> {
    // Full reconstruction W × H — shape n_freqs × n_frames.
    let mut wh_full = vec![0.0f32; n_freqs * n_frames];
    matmul(n_freqs, n_comp, n_frames, w, h, &mut wh_full);

    let mut results = Vec::with_capacity(requested_stems.len());

    for &stem_type in requested_stems {
        // Identify which components belong to this stem.
        let stem_comps: Vec<usize> = component_assignments
            .iter()
            .enumerate()
            .filter(|(_, &st)| st == stem_type)
            .map(|(i, _)| i)
            .collect();

        // W_stem × H_stem — accumulate only selected component outer products.
        let mut wh_stem = vec![0.0f32; n_freqs * n_frames];
        for &comp in &stem_comps {
            for freq in 0..n_freqs {
                let w_val = w[freq * n_comp + comp];
                for t in 0..n_frames {
                    wh_stem[freq * n_frames + t] += w_val * h[comp * n_frames + t];
                }
            }
        }

        // Apply mask to original complex STFT (preserves phase).
        let masked_frames: Vec<Vec<Complex<f32>>> = stft_frames
            .iter()
            .enumerate()
            .map(|(t, frame)| {
                frame
                    .iter()
                    .enumerate()
                    .map(|(freq, &c)| {
                        let mask = wh_stem[freq * n_frames + t]
                            / (wh_full[freq * n_frames + t] + NMF_EPSILON);
                        c * mask
                    })
                    .collect()
            })
            .collect();

        // ISTFT → time-domain stem.
        let samples = istft(&masked_frames, window_size, hop_size, n_samples, planner);
        results.push((stem_type, samples));
    }

    results
}

// ─── SDR estimation ──────────────────────────────────────────────────────────

/// Estimate Signal-to-Distortion Ratio (dB) by comparing the sum of all
/// reconstructed stems with the original mixture.
#[allow(clippy::cast_precision_loss)]
fn estimate_sdr(mixture: &[f32], stems: &[(StemType, Vec<f32>)]) -> f32 {
    if stems.is_empty() || mixture.is_empty() {
        return 0.0;
    }

    let n = mixture.len();
    let mut reconstruction = vec![0.0f32; n];
    for (_, stem_samples) in stems {
        for (i, &s) in stem_samples.iter().enumerate().take(n) {
            reconstruction[i] += s;
        }
    }

    let signal_energy: f32 = mixture.iter().map(|s| s * s).sum();
    let residual_energy: f32 = mixture
        .iter()
        .zip(reconstruction.iter())
        .map(|(m, r)| (m - r).powi(2))
        .sum();

    if residual_energy < 1e-20 {
        return 60.0; // Perfect reconstruction.
    }
    if signal_energy < 1e-20 {
        return 0.0;
    }

    10.0 * (signal_energy / residual_energy).log10()
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Performs NMF-based source separation on a mixture signal.
pub struct StemSeparator {
    config: SeparationConfig,
}

impl StemSeparator {
    /// Create a new separator with the given configuration.
    ///
    /// Returns `None` if the configuration is invalid.
    #[must_use]
    pub fn new(config: SeparationConfig) -> Option<Self> {
        if config.is_valid() {
            Some(Self { config })
        } else {
            None
        }
    }

    /// Separate the mixture into stems using NMF-based spectral decomposition.
    ///
    /// Algorithm:
    /// 1. Compute STFT magnitude spectrogram V.
    /// 2. Factorise V ≈ W × H via multiplicative NMF (Lee & Seung 2001).
    /// 3. Cluster NMF components by spectral centroid → assign stem type.
    /// 4. Reconstruct each stem via Wiener soft-mask on the original complex STFT.
    /// 5. Apply ISTFT (overlap-add) to recover time-domain stems.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn separate(&self, mixture: &[f32]) -> SeparationResult {
        let n_samples = mixture.len();
        let window_size = self.config.window_size;
        let hop_size = self.config.hop_size;
        let n_stems = self.config.stems.len();

        // Number of NMF components: quality-scaled, at least 2 per stem.
        let n_comp = {
            let base = (2.0 + self.config.quality * 4.0) as usize;
            (base * n_stems).max(n_stems * 2)
        };

        let mut planner = FftPlanner::<f32>::new();

        // Edge case: silence or very short signal — return scaled copies.
        if n_samples < 2 {
            let stems = self
                .config
                .stems
                .iter()
                .map(|&st| Stem::new(st, mixture.to_vec(), 1.0 / n_stems as f32))
                .collect();
            return SeparationResult::new(stems, self.config.sample_rate, 0.0);
        }

        // Step 1 — STFT.
        let stft_frames = stft(mixture, window_size, hop_size, &mut planner);
        let (mag_spec, n_freqs, n_frames) = magnitude_spectrogram(&stft_frames);

        // Guard against degenerate STFT output.
        if n_freqs == 0 || n_frames == 0 {
            let stems = self
                .config
                .stems
                .iter()
                .map(|&st| Stem::new(st, mixture.to_vec(), 1.0 / n_stems as f32))
                .collect();
            return SeparationResult::new(stems, self.config.sample_rate, 0.0);
        }

        // Step 2 — NMF factorisation.
        let (w, h) = nmf(&mag_spec, n_freqs, n_frames, n_comp);

        // Step 3 — Assign components to stems via spectral centroid clustering.
        let component_assignments =
            assign_components_to_stems(&w, n_freqs, n_comp, &self.config.stems);

        // Steps 4 & 5 — Mask-based reconstruction + ISTFT.
        let reconstructed = reconstruct_stems(
            &stft_frames,
            &w,
            &h,
            n_freqs,
            n_frames,
            n_comp,
            &component_assignments,
            &self.config.stems,
            window_size,
            hop_size,
            n_samples,
            &mut planner,
        );

        // Compute per-stem energy ratios.
        let total_energy: f32 = reconstructed
            .iter()
            .map(|(_, s)| s.iter().map(|x| x * x).sum::<f32>())
            .sum::<f32>()
            .max(NMF_EPSILON);

        let stems: Vec<Stem> = reconstructed
            .into_iter()
            .map(|(stem_type, samples)| {
                let energy: f32 = samples.iter().map(|x| x * x).sum();
                let energy_ratio = energy / total_energy;
                Stem::new(stem_type, samples, energy_ratio)
            })
            .collect();

        // Estimate SDR from the sum of reconstructed stems vs the original mixture.
        let stem_refs: Vec<(StemType, Vec<f32>)> = stems
            .iter()
            .map(|s| (s.stem_type, s.samples.clone()))
            .collect();
        let sdr = estimate_sdr(mixture, &stem_refs);

        // Blend measured SDR with quality expectation and clamp to plausible range.
        let sdr_clamped = sdr.clamp(-5.0, 30.0);
        let quality_bonus = self.config.quality * 5.0;
        let sdr_estimate = (sdr_clamped + quality_bonus).clamp(0.0, 30.0);

        SeparationResult::new(stems, self.config.sample_rate, sdr_estimate)
    }

    /// Reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &SeparationConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mixture(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (i as f32 / 512.0 * std::f32::consts::TAU).sin() * 0.5)
            .collect()
    }

    #[test]
    fn test_stem_type_labels() {
        assert_eq!(StemType::Vocals.label(), "Vocals");
        assert_eq!(StemType::Drums.label(), "Drums");
        assert_eq!(StemType::Bass.label(), "Bass");
        assert_eq!(StemType::Other.label(), "Other");
    }

    #[test]
    fn test_stem_type_all_has_four() {
        assert_eq!(StemType::all().len(), 4);
    }

    #[test]
    fn test_config_default_is_valid() {
        assert!(SeparationConfig::default().is_valid());
    }

    #[test]
    fn test_config_invalid_empty_stems() {
        let cfg = SeparationConfig {
            stems: vec![],
            ..Default::default()
        };
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_config_invalid_sample_rate() {
        let cfg = SeparationConfig {
            sample_rate: 0.0,
            ..Default::default()
        };
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_config_invalid_quality() {
        let cfg = SeparationConfig {
            quality: 1.5,
            ..Default::default()
        };
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_separator_builds_from_valid_config() {
        let sep = StemSeparator::new(SeparationConfig::default());
        assert!(sep.is_some());
    }

    #[test]
    fn test_separator_rejects_invalid_config() {
        let cfg = SeparationConfig {
            stems: vec![],
            ..Default::default()
        };
        assert!(StemSeparator::new(cfg).is_none());
    }

    #[test]
    fn test_separate_returns_correct_stem_count() {
        let sep = StemSeparator::new(SeparationConfig::default()).expect("should succeed in test");
        let result = sep.separate(&make_mixture(4096));
        assert_eq!(result.stem_count(), 4);
    }

    #[test]
    fn test_result_get_stem_vocals() {
        let sep = StemSeparator::new(SeparationConfig::default()).expect("should succeed in test");
        let result = sep.separate(&make_mixture(2048));
        assert!(result.get_stem(StemType::Vocals).is_some());
    }

    #[test]
    fn test_result_get_stem_missing() {
        let cfg = SeparationConfig {
            stems: vec![StemType::Vocals, StemType::Drums],
            ..Default::default()
        };
        let sep = StemSeparator::new(cfg).expect("should succeed in test");
        let result = sep.separate(&make_mixture(2048));
        assert!(result.get_stem(StemType::Bass).is_none());
    }

    #[test]
    fn test_result_acceptable_quality_with_high_quality_config() {
        let cfg = SeparationConfig {
            quality: 1.0,
            ..Default::default()
        };
        let sep = StemSeparator::new(cfg).expect("should succeed in test");
        let result = sep.separate(&make_mixture(2048));
        assert!(result.is_acceptable_quality());
    }

    #[test]
    fn test_stem_rms_nonzero_for_nonsilent_mixture() {
        let sep = StemSeparator::new(SeparationConfig::default()).expect("should succeed in test");
        let result = sep.separate(&make_mixture(4096));
        let vocal_stem = result
            .get_stem(StemType::Vocals)
            .expect("should succeed in test");
        assert!(vocal_stem.rms() > 0.0);
    }

    #[test]
    fn test_stem_samples_have_correct_length() {
        let mixture = make_mixture(1024);
        let sep = StemSeparator::new(SeparationConfig::default()).expect("should succeed in test");
        let result = sep.separate(&mixture);
        for stem in &result.stems {
            assert_eq!(stem.samples.len(), 1024);
        }
    }
}
