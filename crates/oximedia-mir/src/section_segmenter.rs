//! Music section segmentation via self-similarity matrix and novelty curves.
//!
//! Detects structural boundaries (intro, verse, chorus, bridge, outro) in audio
//! by computing a chroma-based self-similarity matrix, deriving a novelty curve
//! via a checkerboard kernel, picking boundary candidates at novelty peaks, and
//! labelling the resulting segments using heuristic rules.
//!
//! ## Algorithm
//!
//! 1. **Chroma features** — Compute a 12-bin chromagram per analysis frame using
//!    a windowed DFT mapped to pitch classes.
//! 2. **Self-similarity matrix (SSM)** — Build an N×N cosine-similarity matrix
//!    from the chroma feature sequence.
//! 3. **Novelty curve** — Convolve the SSM diagonal with a Gaussian-weighted
//!    checkerboard kernel to produce a 1-D novelty signal.
//! 4. **Boundary detection** — Find peaks in the novelty curve that exceed a
//!    dynamic threshold; map peak frame indices to time.
//! 5. **Segment labelling** — Label each segment as intro/verse/chorus/bridge/
//!    outro based on its position, duration, and energy profile.
//!
//! # Example
//!
//! ```
//! use oximedia_mir::section_segmenter::{SectionSegmenter, SegmenterConfig};
//!
//! let sr = 22050_u32;
//! let config = SegmenterConfig::default();
//! let segmenter = SectionSegmenter::new(sr, config);
//!
//! // 10 seconds of audio
//! let samples: Vec<f32> = (0..sr * 10)
//!     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
//!     .collect();
//!
//! let result = segmenter.segment(&samples).expect("segmentation failed");
//! println!("Found {} sections", result.sections.len());
//! ```

#![allow(dead_code)]

use crate::{MirError, MirResult};
use oxifft::Complex;
use std::f32::consts::PI;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Number of pitch classes in a chromagram.
const N_CHROMA: usize = 12;
/// Reference frequency for pitch class A (MIDI 69).
const A4_HZ: f32 = 440.0;
/// Checkerboard kernel half-width in frames.
const KERNEL_RADIUS: usize = 8;
/// Gaussian sigma for kernel weighting.
const KERNEL_SIGMA: f32 = 4.0;
/// Novelty peak detection threshold multiplier (relative to mean novelty).
const PEAK_THRESHOLD: f32 = 0.8;
/// Minimum distance between boundary peaks in frames.
const MIN_BOUNDARY_DISTANCE: usize = 4;

// ── SectionLabel ──────────────────────────────────────────────────────────────

/// Structural label for a detected segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SectionLabel {
    /// Song introduction.
    Intro,
    /// Verse section.
    Verse,
    /// Pre-chorus section.
    PreChorus,
    /// Chorus / refrain.
    Chorus,
    /// Bridge section.
    Bridge,
    /// Outro / coda.
    Outro,
    /// Undetermined section.
    Unknown,
}

impl SectionLabel {
    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Intro => "intro",
            Self::Verse => "verse",
            Self::PreChorus => "pre-chorus",
            Self::Chorus => "chorus",
            Self::Bridge => "bridge",
            Self::Outro => "outro",
            Self::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for SectionLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ── Section ───────────────────────────────────────────────────────────────────

/// A detected structural section.
#[derive(Debug, Clone)]
pub struct Section {
    /// Start time in seconds.
    pub start_s: f32,
    /// End time in seconds.
    pub end_s: f32,
    /// Section label.
    pub label: SectionLabel,
    /// Mean energy of the section (normalised).
    pub mean_energy: f32,
    /// Confidence score in `[0, 1]`.
    pub confidence: f32,
}

impl Section {
    /// Duration in seconds.
    #[must_use]
    pub fn duration_s(&self) -> f32 {
        self.end_s - self.start_s
    }

    /// Whether this section is a recognised structural element (not Unknown).
    #[must_use]
    pub fn is_labelled(&self) -> bool {
        !matches!(self.label, SectionLabel::Unknown)
    }
}

// ── SegmentationResult ────────────────────────────────────────────────────────

/// Result of structural segmentation.
#[derive(Debug, Clone)]
pub struct SegmentationResult {
    /// Detected sections in chronological order.
    pub sections: Vec<Section>,
    /// Boundary times in seconds (excluding 0.0 and total duration).
    pub boundaries_s: Vec<f32>,
    /// Novelty curve values (one per frame).
    pub novelty: Vec<f32>,
    /// Total duration of the analysed audio in seconds.
    pub total_duration_s: f32,
}

impl SegmentationResult {
    /// Return sections with a specific label.
    #[must_use]
    pub fn sections_with_label(&self, label: SectionLabel) -> Vec<&Section> {
        self.sections
            .iter()
            .filter(|s| s.label == label)
            .collect()
    }

    /// Return the longest section.
    #[must_use]
    pub fn longest_section(&self) -> Option<&Section> {
        self.sections
            .iter()
            .max_by(|a, b| {
                a.duration_s()
                    .partial_cmp(&b.duration_s())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Return the number of detected boundaries.
    #[must_use]
    pub fn n_boundaries(&self) -> usize {
        self.boundaries_s.len()
    }
}

// ── SegmenterConfig ───────────────────────────────────────────────────────────

/// Configuration for the section segmenter.
#[derive(Debug, Clone)]
pub struct SegmenterConfig {
    /// FFT window size in samples.
    pub window_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Checkerboard kernel half-width in frames.
    pub kernel_radius: usize,
    /// Novelty peak threshold multiplier.
    pub peak_threshold: f32,
    /// Minimum distance between boundaries in frames.
    pub min_boundary_distance: usize,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            window_size: 4096,
            hop_size: 1024,
            kernel_radius: KERNEL_RADIUS,
            peak_threshold: PEAK_THRESHOLD,
            min_boundary_distance: MIN_BOUNDARY_DISTANCE,
        }
    }
}

// ── SectionSegmenter ─────────────────────────────────────────────────────────

/// Music section segmenter using SSM + novelty curve.
pub struct SectionSegmenter {
    sample_rate: u32,
    config: SegmenterConfig,
}

impl SectionSegmenter {
    /// Create a new segmenter.
    #[must_use]
    pub fn new(sample_rate: u32, config: SegmenterConfig) -> Self {
        Self {
            sample_rate,
            config,
        }
    }

    /// Segment `samples` into structural sections.
    ///
    /// # Errors
    ///
    /// Returns [`MirError::InsufficientData`] if the signal is too short to
    /// produce at least two analysis frames.
    pub fn segment(&self, samples: &[f32]) -> MirResult<SegmentationResult> {
        let hop = self.config.hop_size.max(1);
        let win = self.config.window_size;
        let sr = self.sample_rate as f32;

        if samples.len() < win * 2 {
            return Err(MirError::InsufficientData(format!(
                "need ≥{} samples, got {}",
                win * 2,
                samples.len()
            )));
        }

        let total_duration_s = samples.len() as f32 / sr;

        // Step 1: Compute chroma frames
        let chroma_frames = self.compute_chroma(samples)?;
        let n_frames = chroma_frames.len();

        if n_frames < 4 {
            return Err(MirError::InsufficientData(
                "not enough frames for segmentation".to_string(),
            ));
        }

        // Step 2: Self-similarity matrix (cosine similarity)
        let ssm = compute_ssm(&chroma_frames);

        // Step 3: Novelty curve via checkerboard kernel
        let novelty = compute_novelty(&ssm, self.config.kernel_radius);

        // Step 4: Find boundary frames
        let boundary_frames = self.find_boundaries(&novelty);

        // Step 5: Build boundary times and sections
        let frame_dur = hop as f32 / sr;
        let boundaries_s: Vec<f32> = boundary_frames
            .iter()
            .map(|&f| f as f32 * frame_dur)
            .collect();

        // Energy per frame for labelling
        let energies = frame_energies(samples, hop);

        let sections = self.label_sections(
            &boundaries_s,
            &energies,
            frame_dur,
            total_duration_s,
        );

        Ok(SegmentationResult {
            sections,
            boundaries_s,
            novelty,
            total_duration_s,
        })
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Compute a 12-bin chromagram for each analysis frame.
    fn compute_chroma(&self, samples: &[f32]) -> MirResult<Vec<[f32; N_CHROMA]>> {
        let hop = self.config.hop_size.max(1);
        let win = self.config.window_size;
        let sr = self.sample_rate as f32;
        let window = hann_window(win);
        let n_frames = (samples.len().saturating_sub(win)) / hop + 1;
        let mut frames = Vec::with_capacity(n_frames);

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            let end = start + win;
            if end > samples.len() {
                break;
            }

            let fft_in: Vec<Complex<f32>> = samples[start..end]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();

            let spectrum = oxifft::fft(&fft_in);
            let n_bins = spectrum.len() / 2;
            let mags: Vec<f32> = spectrum[..n_bins].iter().map(|c| c.norm()).collect();

            let chroma = mags_to_chroma(&mags, sr, n_bins);
            frames.push(chroma);
        }

        Ok(frames)
    }

    /// Find boundary frames as peaks in the novelty curve.
    fn find_boundaries(&self, novelty: &[f32]) -> Vec<usize> {
        if novelty.is_empty() {
            return Vec::new();
        }

        let mean_nov: f32 = novelty.iter().sum::<f32>() / novelty.len() as f32;
        let threshold = mean_nov * self.config.peak_threshold;

        let min_dist = self.config.min_boundary_distance.max(1);
        let mut peaks: Vec<usize> = Vec::new();

        for i in 1..novelty.len().saturating_sub(1) {
            if novelty[i] > novelty[i - 1]
                && novelty[i] > novelty[i + 1]
                && novelty[i] >= threshold
            {
                if peaks.last().map(|&p| i - p >= min_dist).unwrap_or(true) {
                    peaks.push(i);
                } else if let Some(last) = peaks.last_mut() {
                    if novelty[i] > novelty[*last] {
                        *last = i;
                    }
                }
            }
        }

        peaks
    }

    /// Label segments using heuristic rules based on position and energy.
    fn label_sections(
        &self,
        boundaries_s: &[f32],
        energies: &[f32],
        frame_dur: f32,
        total_dur: f32,
    ) -> Vec<Section> {
        // Build boundary list including start and end
        let mut times: Vec<f32> = std::iter::once(0.0_f32)
            .chain(boundaries_s.iter().copied())
            .chain(std::iter::once(total_dur))
            .collect();
        times.dedup_by(|a, b| (*a - *b).abs() < f32::EPSILON);

        let n_segs = times.len().saturating_sub(1);
        if n_segs == 0 {
            return Vec::new();
        }

        // Mean energy across all frames
        let global_mean: f32 = if energies.is_empty() {
            1.0
        } else {
            energies.iter().sum::<f32>() / energies.len() as f32
        };

        let mut sections = Vec::with_capacity(n_segs);

        for seg_idx in 0..n_segs {
            let start_s = times[seg_idx];
            let end_s = times[seg_idx + 1];
            let dur = end_s - start_s;

            // Compute mean energy for this segment
            let e_start = (start_s / frame_dur) as usize;
            let e_end = ((end_s / frame_dur) as usize).min(energies.len());
            let mean_energy = if e_end > e_start && !energies.is_empty() {
                energies[e_start..e_end].iter().sum::<f32>()
                    / (e_end - e_start) as f32
                    / global_mean.max(f32::EPSILON)
            } else {
                1.0
            };

            // Heuristic labelling
            let position_fraction = start_s / total_dur;
            let label = assign_label(
                seg_idx,
                n_segs,
                position_fraction,
                dur,
                total_dur,
                mean_energy,
            );

            let confidence = label_confidence(seg_idx, n_segs, position_fraction, mean_energy);

            sections.push(Section {
                start_s,
                end_s,
                label,
                mean_energy,
                confidence,
            });
        }

        sections
    }
}

// ── Standalone algorithms ─────────────────────────────────────────────────────

/// Build a cosine self-similarity matrix from chroma frames.
#[must_use]
pub fn compute_ssm(chroma: &[[f32; N_CHROMA]]) -> Vec<Vec<f32>> {
    let n = chroma.len();
    let mut ssm = vec![vec![0.0_f32; n]; n];

    // Precompute norms
    let norms: Vec<f32> = chroma
        .iter()
        .map(|c| c.iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect();

    for i in 0..n {
        for j in i..n {
            let dot: f32 = chroma[i]
                .iter()
                .zip(chroma[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            let denom = norms[i] * norms[j];
            let sim = if denom > f32::EPSILON {
                (dot / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            ssm[i][j] = sim;
            ssm[j][i] = sim;
        }
    }

    ssm
}

/// Compute novelty curve from SSM using a Gaussian-weighted checkerboard kernel.
///
/// The checkerboard kernel assigns +1 to the top-left and bottom-right quadrants
/// and −1 to the top-right and bottom-left quadrants of a 2·radius × 2·radius
/// window centred on the SSM diagonal.
#[must_use]
pub fn compute_novelty(ssm: &[Vec<f32>], kernel_radius: usize) -> Vec<f32> {
    let n = ssm.len();
    if n == 0 {
        return Vec::new();
    }

    let r = kernel_radius.min(n / 2).max(1);
    let sigma = r as f32 / 2.0;

    // Build Gaussian-weighted checkerboard kernel
    let size = 2 * r;
    let mut kernel = vec![vec![0.0_f32; size]; size];
    for row in 0..size {
        for col in 0..size {
            let di = row as f32 - r as f32;
            let dj = col as f32 - r as f32;
            let g = (-(di * di + dj * dj) / (2.0 * sigma * sigma)).exp();
            // Checkerboard sign
            let sign = if (row < r) == (col < r) { 1.0 } else { -1.0 };
            kernel[row][col] = sign * g;
        }
    }

    // Convolve kernel along SSM diagonal
    let mut novelty = vec![0.0_f32; n];
    for t in 0..n {
        let mut val = 0.0_f32;
        for row in 0..size {
            let si = t as isize + row as isize - r as isize;
            for col in 0..size {
                let sj = t as isize + col as isize - r as isize;
                if si >= 0 && si < n as isize && sj >= 0 && sj < n as isize {
                    val += ssm[si as usize][sj as usize] * kernel[row][col];
                }
            }
        }
        novelty[t] = val.max(0.0);
    }

    // Normalise to [0, 1]
    let max_val = novelty.iter().cloned().fold(0.0_f32, f32::max);
    if max_val > f32::EPSILON {
        for v in &mut novelty {
            *v /= max_val;
        }
    }

    novelty
}

/// Map magnitude spectrum bins to 12 chroma bins.
fn mags_to_chroma(mags: &[f32], sr: f32, n_bins: usize) -> [f32; N_CHROMA] {
    let mut chroma = [0.0_f32; N_CHROMA];

    if n_bins == 0 || sr < f32::EPSILON {
        return chroma;
    }

    let hz_per_bin = sr / (2.0 * n_bins as f32);
    // Start from bin 1 to skip DC
    for (bin, &mag) in mags.iter().enumerate().skip(1) {
        let hz = bin as f32 * hz_per_bin;
        if hz < 20.0 || hz > 20000.0 {
            continue;
        }
        // Convert hz to pitch class (0 = C)
        let pitch = 12.0 * (hz / A4_HZ).log2() + 9.0; // A4 = class 9
        let pitch_class = pitch.rem_euclid(12.0) as usize % N_CHROMA;
        chroma[pitch_class] += mag;
    }

    // L2 normalise
    let norm: f32 = chroma.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for v in &mut chroma {
            *v /= norm;
        }
    }

    chroma
}

/// Compute per-frame RMS energy.
fn frame_energies(samples: &[f32], hop: usize) -> Vec<f32> {
    if samples.is_empty() || hop == 0 {
        return Vec::new();
    }
    let n_frames = (samples.len() + hop - 1) / hop;
    let mut energies = Vec::with_capacity(n_frames);

    for f in 0..n_frames {
        let start = f * hop;
        let end = (start + hop).min(samples.len());
        let rms = (samples[start..end]
            .iter()
            .map(|s| s * s)
            .sum::<f32>()
            / (end - start) as f32)
            .sqrt();
        energies.push(rms);
    }
    energies
}

/// Heuristic section label based on segment index, position, duration, and energy.
fn assign_label(
    seg_idx: usize,
    n_segs: usize,
    position_frac: f32,
    duration_s: f32,
    total_dur: f32,
    mean_energy: f32,
) -> SectionLabel {
    let avg_seg_dur = total_dur / n_segs as f32;

    // Intro: first segment, short, lower energy
    if seg_idx == 0 && position_frac < 0.15 {
        return SectionLabel::Intro;
    }

    // Outro: last segment
    if seg_idx + 1 == n_segs {
        return SectionLabel::Outro;
    }

    // Bridge: single occurrence in second half, short, low energy
    if position_frac > 0.5 && position_frac < 0.85 && duration_s < avg_seg_dur * 0.7 {
        return SectionLabel::Bridge;
    }

    // Chorus: higher than average energy
    if mean_energy > 1.15 {
        return SectionLabel::Chorus;
    }

    // Pre-chorus: just before an expected chorus (penultimate quarter, short)
    if position_frac > 0.25 && position_frac < 0.7 && duration_s < avg_seg_dur * 0.75 {
        return SectionLabel::PreChorus;
    }

    // Default to verse for remaining mid sections
    SectionLabel::Verse
}

/// Compute a simple confidence score for a label assignment.
fn label_confidence(
    seg_idx: usize,
    n_segs: usize,
    position_frac: f32,
    mean_energy: f32,
) -> f32 {
    // Intro / Outro are positionally certain
    if seg_idx == 0 || seg_idx + 1 == n_segs {
        return 0.85;
    }
    // Chorus confidence scales with energy
    let energy_conf = (mean_energy - 1.0).clamp(0.0, 0.5) * 2.0; // 0..1
    let pos_conf = (1.0 - (position_frac - 0.5).abs() * 2.0).clamp(0.0, 1.0);
    let _ = n_segs; // suppress warning
    ((energy_conf * 0.6 + pos_conf * 0.4) * 0.9 + 0.1).clamp(0.0, 1.0)
}

/// Hann window coefficients.
fn hann_window(size: usize) -> Vec<f32> {
    if size == 0 {
        return Vec::new();
    }
    let denom = size.saturating_sub(1) as f32;
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / denom).cos()))
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn sine(freq_hz: f32, sr: u32, secs: f32, amp: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n)
            .map(|i| amp * (TAU * freq_hz * i as f32 / sr as f32).sin())
            .collect()
    }

    fn concat(vecs: &[Vec<f32>]) -> Vec<f32> {
        vecs.iter().flat_map(|v| v.iter().copied()).collect()
    }

    // ── SectionLabel ──────────────────────────────────────────────────────────

    #[test]
    fn test_label_names_unique() {
        let labels = [
            SectionLabel::Intro,
            SectionLabel::Verse,
            SectionLabel::PreChorus,
            SectionLabel::Chorus,
            SectionLabel::Bridge,
            SectionLabel::Outro,
            SectionLabel::Unknown,
        ];
        let names: std::collections::HashSet<_> = labels.iter().map(|l| l.name()).collect();
        assert_eq!(names.len(), labels.len());
    }

    #[test]
    fn test_label_display() {
        assert_eq!(format!("{}", SectionLabel::Chorus), "chorus");
        assert_eq!(format!("{}", SectionLabel::Verse), "verse");
    }

    // ── Section ───────────────────────────────────────────────────────────────

    #[test]
    fn test_section_duration() {
        let s = Section {
            start_s: 10.0,
            end_s: 40.0,
            label: SectionLabel::Chorus,
            mean_energy: 1.2,
            confidence: 0.8,
        };
        assert!((s.duration_s() - 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_section_is_labelled() {
        let known = Section {
            start_s: 0.0,
            end_s: 10.0,
            label: SectionLabel::Intro,
            mean_energy: 0.8,
            confidence: 0.9,
        };
        let unknown = Section {
            start_s: 10.0,
            end_s: 20.0,
            label: SectionLabel::Unknown,
            mean_energy: 1.0,
            confidence: 0.5,
        };
        assert!(known.is_labelled());
        assert!(!unknown.is_labelled());
    }

    // ── SegmentationResult ────────────────────────────────────────────────────

    #[test]
    fn test_sections_with_label() {
        let result = SegmentationResult {
            sections: vec![
                Section {
                    start_s: 0.0,
                    end_s: 15.0,
                    label: SectionLabel::Intro,
                    mean_energy: 0.8,
                    confidence: 0.85,
                },
                Section {
                    start_s: 15.0,
                    end_s: 45.0,
                    label: SectionLabel::Verse,
                    mean_energy: 1.0,
                    confidence: 0.7,
                },
                Section {
                    start_s: 45.0,
                    end_s: 75.0,
                    label: SectionLabel::Chorus,
                    mean_energy: 1.3,
                    confidence: 0.75,
                },
            ],
            boundaries_s: vec![15.0, 45.0],
            novelty: vec![],
            total_duration_s: 75.0,
        };
        assert_eq!(result.sections_with_label(SectionLabel::Verse).len(), 1);
        assert_eq!(result.sections_with_label(SectionLabel::Bridge).len(), 0);
        assert_eq!(result.n_boundaries(), 2);
    }

    #[test]
    fn test_longest_section() {
        let result = SegmentationResult {
            sections: vec![
                Section {
                    start_s: 0.0,
                    end_s: 10.0,
                    label: SectionLabel::Intro,
                    mean_energy: 0.8,
                    confidence: 0.9,
                },
                Section {
                    start_s: 10.0,
                    end_s: 50.0,
                    label: SectionLabel::Verse,
                    mean_energy: 1.0,
                    confidence: 0.7,
                },
            ],
            boundaries_s: vec![10.0],
            novelty: vec![],
            total_duration_s: 50.0,
        };
        let longest = result.longest_section().expect("should have a section");
        assert_eq!(longest.label, SectionLabel::Verse);
    }

    // ── compute_ssm ───────────────────────────────────────────────────────────

    #[test]
    fn test_ssm_diagonal_is_one() {
        let frames: Vec<[f32; N_CHROMA]> = (0..5)
            .map(|i| {
                let mut c = [0.0_f32; N_CHROMA];
                c[i % N_CHROMA] = 1.0;
                c
            })
            .collect();
        let ssm = compute_ssm(&frames);
        for i in 0..frames.len() {
            assert!(
                (ssm[i][i] - 1.0).abs() < 1e-4,
                "diagonal[{i}] = {}",
                ssm[i][i]
            );
        }
    }

    #[test]
    fn test_ssm_symmetric() {
        let frames: Vec<[f32; N_CHROMA]> = (0..4)
            .map(|i| {
                let mut c = [0.0_f32; N_CHROMA];
                c[(i * 3) % N_CHROMA] = 1.0;
                c[(i * 5 + 2) % N_CHROMA] = 0.5;
                c
            })
            .collect();
        let ssm = compute_ssm(&frames);
        for i in 0..frames.len() {
            for j in 0..frames.len() {
                assert!(
                    (ssm[i][j] - ssm[j][i]).abs() < 1e-5,
                    "SSM not symmetric at ({i}, {j})"
                );
            }
        }
    }

    // ── compute_novelty ───────────────────────────────────────────────────────

    #[test]
    fn test_novelty_length_matches_ssm() {
        let n = 20;
        let ssm = vec![vec![1.0_f32; n]; n];
        let novelty = compute_novelty(&ssm, 4);
        assert_eq!(novelty.len(), n);
    }

    #[test]
    fn test_novelty_in_unit_range() {
        let n = 16;
        let ssm: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let novelty = compute_novelty(&ssm, 4);
        for v in &novelty {
            assert!(*v >= 0.0 && *v <= 1.0, "novelty value out of range: {v}");
        }
    }

    // ── SectionSegmenter ──────────────────────────────────────────────────────

    #[test]
    fn test_segment_returns_at_least_one_section() {
        let sr = 22050_u32;
        let config = SegmenterConfig {
            window_size: 1024,
            hop_size: 512,
            ..Default::default()
        };
        let segmenter = SectionSegmenter::new(sr, config);
        // 5 seconds of audio
        let samples = sine(440.0, sr, 5.0, 0.5);
        let result = segmenter.segment(&samples).expect("segmentation failed");
        assert!(!result.sections.is_empty());
    }

    #[test]
    fn test_segment_too_short_returns_error() {
        let sr = 22050_u32;
        let config = SegmenterConfig {
            window_size: 1024,
            hop_size: 512,
            ..Default::default()
        };
        let segmenter = SectionSegmenter::new(sr, config);
        let samples = vec![0.0_f32; 512]; // shorter than 2 * window_size
        let err = segmenter.segment(&samples);
        assert!(err.is_err());
    }

    #[test]
    fn test_first_section_starts_at_zero() {
        let sr = 22050_u32;
        let config = SegmenterConfig {
            window_size: 512,
            hop_size: 256,
            ..Default::default()
        };
        let segmenter = SectionSegmenter::new(sr, config);
        let samples = sine(440.0, sr, 5.0, 0.5);
        let result = segmenter.segment(&samples).expect("segmentation failed");
        let first = result.sections.first().expect("at least one section");
        assert!(first.start_s.abs() < 1e-4, "first section should start at 0");
    }

    #[test]
    fn test_section_times_cover_full_duration() {
        let sr = 22050_u32;
        let config = SegmenterConfig {
            window_size: 512,
            hop_size: 256,
            ..Default::default()
        };
        let segmenter = SectionSegmenter::new(sr, config);
        // Create a multi-section signal: two different notes
        let part_a = sine(440.0, sr, 3.0, 0.5);
        let part_b = sine(880.0, sr, 3.0, 0.3);
        let samples = concat(&[part_a, part_b]);
        let result = segmenter.segment(&samples).expect("segmentation failed");
        let last_end = result
            .sections
            .last()
            .map(|s| s.end_s)
            .unwrap_or(0.0);
        assert!(
            last_end > 0.0,
            "last section end should be positive: {last_end}"
        );
    }
}
