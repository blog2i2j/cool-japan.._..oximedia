//! Chroma (pitch class profile) feature computation.
//!
//! Chroma features represent the distribution of energy across the 12 pitch
//! classes of the Western chromatic scale (C, C#, D, D#, E, F, F#, G, G#,
//! A, A#, B).  They are key-invariant and robust to timbre, making them
//! widely used for chord detection, key estimation, and cover-song detection.
//!
//! The 12-bin chroma vector is computed by folding the magnitude spectrum
//! into 12 pitch-class bins using the equal-temperament frequency formula:
//!
//! ```text
//! pitch_class(freq) = round(12 * log2(freq / A4)) mod 12
//! ```
//!
//! where A4 = 440 Hz by convention.

/// Number of pitch classes in the chromatic scale.
pub const NUM_PITCH_CLASSES: usize = 12;

/// Pitch class names (starting from C).
pub const PITCH_CLASS_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// A 12-bin chroma (pitch class profile) vector.
#[derive(Debug, Clone)]
pub struct ChromaVector {
    /// 12-bin chroma values (index 0 = C, 1 = C#, ..., 11 = B).
    /// Values are non-negative and normalised to sum to 1 (if `normalised`
    /// is true) or contain raw accumulated energy.
    pub bins: [f32; NUM_PITCH_CLASSES],
    /// Whether `bins` have been L1-normalised.
    pub normalised: bool,
}

impl ChromaVector {
    /// Creates a zero-valued chroma vector.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            bins: [0.0; NUM_PITCH_CLASSES],
            normalised: false,
        }
    }

    /// Returns the dominant pitch class (index of the maximum bin).
    #[must_use]
    pub fn dominant_pitch_class(&self) -> usize {
        let mut best = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in self.bins.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        best
    }

    /// Returns the name of the dominant pitch class.
    #[must_use]
    pub fn dominant_pitch_class_name(&self) -> &'static str {
        PITCH_CLASS_NAMES[self.dominant_pitch_class()]
    }

    /// L1-normalises `bins` in place so they sum to 1.
    pub fn normalise(&mut self) {
        let sum: f32 = self.bins.iter().sum();
        if sum > 0.0 {
            for b in &mut self.bins {
                *b /= sum;
            }
            self.normalised = true;
        }
    }

    /// Returns a normalised copy.
    #[must_use]
    pub fn normalised_copy(&self) -> Self {
        let mut copy = self.clone();
        copy.normalise();
        copy
    }

    /// Computes the cosine similarity between two chroma vectors (range –1..1).
    #[must_use]
    pub fn cosine_similarity(&self, other: &ChromaVector) -> f32 {
        let dot: f32 = self
            .bins
            .iter()
            .zip(other.bins.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let norm_a: f32 = self.bins.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.bins.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm_a <= 0.0 || norm_b <= 0.0 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

impl Default for ChromaVector {
    fn default() -> Self {
        Self::zeros()
    }
}

/// Configuration for chroma feature extraction.
#[derive(Debug, Clone)]
pub struct ChromaConfig {
    /// Reference pitch for A4 in Hz (default 440.0).
    pub a4_hz: f32,
    /// Minimum frequency to consider when folding (Hz, default 27.5 = A0).
    pub min_freq: f32,
    /// Maximum frequency to consider when folding (Hz, default 4186.0 = C8).
    pub max_freq: f32,
    /// Whether to normalise the output vector.
    pub normalise: bool,
}

impl Default for ChromaConfig {
    fn default() -> Self {
        Self {
            a4_hz: 440.0,
            min_freq: 27.5,
            max_freq: 4186.0,
            normalise: true,
        }
    }
}

/// Compute a 12-bin chroma vector from a magnitude spectrum.
///
/// # Arguments
/// * `magnitude`   - Magnitude spectrum (positive frequencies, length N/2+1)
/// * `sample_rate` - Sample rate in Hz
/// * `config`      - Chroma extraction configuration
///
/// # Returns
/// A [`ChromaVector`] with 12 bins (C = 0, C# = 1, …, B = 11).
#[must_use]
pub fn chroma_vector(magnitude: &[f32], sample_rate: f32, config: &ChromaConfig) -> ChromaVector {
    let mut chroma = ChromaVector::zeros();

    if magnitude.is_empty() || sample_rate <= 0.0 {
        return chroma;
    }

    let n_bins = magnitude.len();

    for (k, &mag) in magnitude.iter().enumerate() {
        if mag <= 0.0 {
            continue;
        }
        // Frequency of bin k
        let freq = k as f32 * sample_rate / (2.0 * (n_bins - 1) as f32);
        if freq < config.min_freq || freq > config.max_freq {
            continue;
        }

        // Map frequency to pitch class using equal temperament:
        //   pitch_class = round(12 * log2(freq / A4)) mod 12
        let semitones_from_a4 = 12.0 * (freq / config.a4_hz).log2();
        // A4 is pitch class 9 (A = index 9 in C-major order)
        // pitch_class = (round(semitones_from_a4) + 9) mod 12, mapped to [0, 12)
        let rounded = semitones_from_a4.round() as i32;
        let pc = ((rounded + 9).rem_euclid(12)) as usize;

        chroma.bins[pc] += mag;
    }

    if config.normalise {
        chroma.normalise();
    }

    chroma
}

/// Compute chroma vectors from a spectrogram (frame × bins).
///
/// # Arguments
/// * `spectrogram` - Sequence of magnitude spectra (oldest first)
/// * `sample_rate` - Sample rate in Hz
/// * `config`      - Chroma extraction configuration
///
/// # Returns
/// One [`ChromaVector`] per spectrogram frame.
#[must_use]
pub fn chroma_track(
    spectrogram: &[Vec<f32>],
    sample_rate: f32,
    config: &ChromaConfig,
) -> Vec<ChromaVector> {
    spectrogram
        .iter()
        .map(|frame| chroma_vector(frame, sample_rate, config))
        .collect()
}

/// Compute the mean chroma vector over a sequence of chroma vectors.
///
/// Returns `None` if `chroma_track` is empty.
#[must_use]
pub fn mean_chroma(chroma_track_data: &[ChromaVector]) -> Option<ChromaVector> {
    if chroma_track_data.is_empty() {
        return None;
    }

    let mut accum = [0.0_f32; NUM_PITCH_CLASSES];
    for cv in chroma_track_data {
        for (a, &b) in accum.iter_mut().zip(cv.bins.iter()) {
            *a += b;
        }
    }

    let n = chroma_track_data.len() as f32;
    for a in &mut accum {
        *a /= n;
    }

    Some(ChromaVector {
        bins: accum,
        normalised: false,
    })
}

/// Estimate the most likely musical key from a chroma vector.
///
/// Uses a simple correlation with Krumhansl-Schmuckler key-profile templates
/// (approximated) for all 24 major and minor keys, returning the best-matching
/// key as (`root_pitch_class`, `is_major`).
///
/// `root_pitch_class` = 0 → C, 1 → C#, …, 11 → B.
#[must_use]
pub fn estimate_key(chroma: &ChromaVector) -> (usize, bool) {
    // Krumhansl-Schmuckler profiles (normalised, starting from C)
    // Major: based on the original 1982 experiment
    const MAJOR_PROFILE: [f32; 12] = [
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
    ];
    // Natural minor
    const MINOR_PROFILE: [f32; 12] = [
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
    ];

    let profile_mean = |p: &[f32; 12]| -> f32 { p.iter().sum::<f32>() / 12.0 };
    let major_mean = profile_mean(&MAJOR_PROFILE);
    let minor_mean = profile_mean(&MINOR_PROFILE);
    let chroma_mean: f32 = chroma.bins.iter().sum::<f32>() / 12.0;

    let profile_std = |p: &[f32; 12], m: f32| -> f32 {
        let var: f32 = p.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / 12.0;
        var.sqrt()
    };
    let chroma_std: f32 = {
        let var: f32 = chroma
            .bins
            .iter()
            .map(|&v| (v - chroma_mean) * (v - chroma_mean))
            .sum::<f32>()
            / 12.0;
        var.sqrt()
    };

    if chroma_std < f32::EPSILON {
        return (0, true); // Flat chroma — default to C major
    }

    let major_std = profile_std(&MAJOR_PROFILE, major_mean);
    let minor_std = profile_std(&MINOR_PROFILE, minor_mean);

    let mut best_key = 0usize;
    let mut best_is_major = true;
    let mut best_corr = f32::NEG_INFINITY;

    for root in 0..12 {
        // Rotate profiles to match each key
        let maj_corr = pearson_corr_profile(
            &chroma.bins,
            &rotated(MAJOR_PROFILE, root),
            chroma_mean,
            chroma_std,
            major_mean,
            major_std,
        );
        let min_corr = pearson_corr_profile(
            &chroma.bins,
            &rotated(MINOR_PROFILE, root),
            chroma_mean,
            chroma_std,
            minor_mean,
            minor_std,
        );

        if maj_corr > best_corr {
            best_corr = maj_corr;
            best_key = root;
            best_is_major = true;
        }
        if min_corr > best_corr {
            best_corr = min_corr;
            best_key = root;
            best_is_major = false;
        }
    }

    (best_key, best_is_major)
}

/// Rotate a 12-element array by `shift` positions (for key transposition).
fn rotated(arr: [f32; 12], shift: usize) -> [f32; 12] {
    let mut out = [0.0_f32; 12];
    for i in 0..12 {
        out[i] = arr[(i + 12 - shift) % 12];
    }
    out
}

/// Pearson correlation between a chroma vector and a key profile.
fn pearson_corr_profile(
    chroma: &[f32; 12],
    profile: &[f32; 12],
    chroma_mean: f32,
    chroma_std: f32,
    profile_mean: f32,
    profile_std: f32,
) -> f32 {
    if chroma_std < f32::EPSILON || profile_std < f32::EPSILON {
        return 0.0;
    }
    let cov: f32 = chroma
        .iter()
        .zip(profile.iter())
        .map(|(&c, &p)| (c - chroma_mean) * (p - profile_mean))
        .sum::<f32>()
        / 12.0;
    cov / (chroma_std * profile_std)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine_magnitude(freq_hz: f32, sample_rate: f32, fft_size: usize) -> Vec<f32> {
        let bin = ((freq_hz / sample_rate) * fft_size as f32).round() as usize;
        let bin = bin.min(fft_size / 2);
        let mut spectrum = vec![0.0_f32; fft_size / 2 + 1];
        spectrum[bin] = 1.0;
        spectrum
    }

    // ── ChromaVector ──────────────────────────────────────────────────────────

    #[test]
    fn test_chroma_vector_zeros() {
        let cv = ChromaVector::zeros();
        assert_eq!(cv.bins, [0.0_f32; 12]);
        assert!(!cv.normalised);
    }

    #[test]
    fn test_chroma_normalise() {
        let mut cv = ChromaVector::zeros();
        cv.bins[0] = 3.0;
        cv.bins[3] = 7.0;
        cv.normalise();
        assert!((cv.bins.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(cv.normalised);
    }

    #[test]
    fn test_chroma_dominant_pitch_class() {
        let mut cv = ChromaVector::zeros();
        cv.bins[7] = 5.0; // G
        assert_eq!(cv.dominant_pitch_class(), 7);
        assert_eq!(cv.dominant_pitch_class_name(), "G");
    }

    #[test]
    fn test_chroma_cosine_similarity_identical() {
        let mut cv = ChromaVector::zeros();
        cv.bins[0] = 1.0;
        cv.bins[4] = 1.0;
        let sim = cv.cosine_similarity(&cv);
        assert!((sim - 1.0).abs() < 1e-6, "Identical vectors: sim = {sim}");
    }

    #[test]
    fn test_chroma_cosine_similarity_orthogonal() {
        let mut a = ChromaVector::zeros();
        let mut b = ChromaVector::zeros();
        a.bins[0] = 1.0;
        b.bins[6] = 1.0;
        let sim = a.cosine_similarity(&b);
        assert!((sim).abs() < 1e-6, "Orthogonal vectors: sim = {sim}");
    }

    // ── chroma_vector ─────────────────────────────────────────────────────────

    #[test]
    fn test_chroma_empty_spectrum() {
        let empty: Vec<f32> = vec![];
        let config = ChromaConfig::default();
        let cv = chroma_vector(&empty, 44100.0, &config);
        assert_eq!(cv.bins, [0.0_f32; 12]);
    }

    #[test]
    fn test_chroma_a4_maps_to_a() {
        // A4 (440 Hz) should map to pitch class 9 (A).
        let sr = 44100.0;
        let fft_size = 4096;
        let spectrum = make_sine_magnitude(440.0, sr, fft_size);
        let config = ChromaConfig {
            normalise: false,
            ..Default::default()
        };
        let cv = chroma_vector(&spectrum, sr, &config);
        let dominant = cv.dominant_pitch_class();
        assert_eq!(
            dominant, 9,
            "A4 should map to pitch class 9 (A), got {dominant}"
        );
    }

    #[test]
    fn test_chroma_a5_maps_to_a() {
        // A5 (880 Hz) should also map to pitch class 9 (A) — octave invariance.
        let sr = 44100.0;
        let fft_size = 4096;
        let spectrum = make_sine_magnitude(880.0, sr, fft_size);
        let config = ChromaConfig {
            normalise: false,
            ..Default::default()
        };
        let cv = chroma_vector(&spectrum, sr, &config);
        let dominant = cv.dominant_pitch_class();
        assert_eq!(
            dominant, 9,
            "A5 should map to pitch class 9 (A), got {dominant}"
        );
    }

    #[test]
    fn test_chroma_normalised_sums_to_one() {
        let sr = 44100.0;
        let spectrum = make_sine_magnitude(440.0, sr, 2048);
        let config = ChromaConfig::default(); // normalise = true
        let cv = chroma_vector(&spectrum, sr, &config);
        let sum: f32 = cv.bins.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5 || sum == 0.0, "sum = {sum}");
    }

    #[test]
    fn test_chroma_out_of_range_freq() {
        // A frequency below min_freq should contribute no energy.
        let sr = 44100.0;
        let mut spectrum = vec![0.0_f32; 1025];
        // Bin 0 is DC (0 Hz), which is below min_freq of 27.5 Hz.
        spectrum[0] = 100.0;
        let config = ChromaConfig {
            min_freq: 27.5,
            max_freq: 4186.0,
            normalise: false,
            ..Default::default()
        };
        let cv = chroma_vector(&spectrum, sr, &config);
        let sum: f32 = cv.bins.iter().sum();
        assert_eq!(sum, 0.0, "DC bin should not contribute, sum = {sum}");
    }

    #[test]
    fn test_chroma_track_length() {
        let sr = 44100.0;
        let config = ChromaConfig::default();
        let spectrogram: Vec<Vec<f32>> = (0..8).map(|_| vec![1.0_f32; 513]).collect();
        let track = chroma_track(&spectrogram, sr, &config);
        assert_eq!(track.len(), 8);
    }

    // ── mean_chroma ────────────────────────────────────────────────────────────

    #[test]
    fn test_mean_chroma_empty() {
        assert!(mean_chroma(&[]).is_none());
    }

    #[test]
    fn test_mean_chroma_single() {
        let mut cv = ChromaVector::zeros();
        cv.bins[0] = 1.0;
        let m = mean_chroma(&[cv.clone()]).expect("unexpected None/Err");
        assert!((m.bins[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_chroma_averages_correctly() {
        let mut cv1 = ChromaVector::zeros();
        cv1.bins[0] = 2.0;
        let mut cv2 = ChromaVector::zeros();
        cv2.bins[0] = 4.0;
        let m = mean_chroma(&[cv1, cv2]).expect("mean chroma should succeed");
        assert!((m.bins[0] - 3.0).abs() < 1e-6);
    }

    // ── estimate_key ──────────────────────────────────────────────────────────

    #[test]
    fn test_estimate_key_c_major() {
        // C major chord: C (0), E (4), G (7)
        let mut cv = ChromaVector::zeros();
        cv.bins[0] = 1.0; // C
        cv.bins[4] = 0.7; // E
        cv.bins[7] = 0.8; // G
        let (key, is_major) = estimate_key(&cv);
        assert_eq!(key, 0, "Expected C (0), got {key}");
        assert!(is_major, "Expected major key");
    }

    #[test]
    fn test_estimate_key_returns_valid_range() {
        let mut cv = ChromaVector::zeros();
        for b in cv.bins.iter_mut() {
            *b = 1.0;
        }
        let (key, _) = estimate_key(&cv);
        assert!(key < 12, "Key pitch class must be in 0..12, got {key}");
    }

    #[test]
    fn test_pitch_class_names_count() {
        assert_eq!(PITCH_CLASS_NAMES.len(), NUM_PITCH_CLASSES);
    }

    #[test]
    fn test_rotated_identity() {
        let arr = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let r = rotated(arr, 0);
        assert_eq!(r, arr);
    }

    #[test]
    fn test_rotated_by_one() {
        let arr = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let r = rotated(arr, 1);
        // rotated by 1: r[i] = arr[(i + 11) % 12] = arr[i-1 mod 12]
        assert!((r[0] - 12.0).abs() < f32::EPSILON, "r[0]={}", r[0]);
        assert!((r[1] - 1.0).abs() < f32::EPSILON, "r[1]={}", r[1]);
    }
}
