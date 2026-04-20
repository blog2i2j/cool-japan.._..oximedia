//! Melody extraction and contour analysis.
//!
//! This module extracts the dominant melody as a sequence of discrete
//! [`MelodyNote`]s from mono PCM audio, then analyses the resulting
//! [`MelodyContour`] to compute musical descriptors (pitch range, average
//! pitch, interval sequence) and classify the overall contour shape.
//!
//! ## Algorithm overview
//!
//! 1. Divide the signal into non-overlapping hops of `hop_size` samples.
//! 2. Compute short-time energy for each hop.
//! 3. Mark a hop as *voiced* when its energy exceeds a dynamic threshold
//!    (mean energy × `VOICED_THRESHOLD_FACTOR`).
//! 4. Estimate fundamental frequency for each voiced hop using a
//!    zero-crossing-based period estimator (robust for monophonic signals).
//! 5. Group consecutive voiced hops with similar pitch into [`MelodyNote`]s.
//!
//! ## Contour shape classification
//!
//! [`analyze_shape`] inspects the linear trend and range of the note sequence
//! to assign one of six [`ContourShape`] variants.
//!
//! # Example
//!
//! ```
//! use oximedia_mir::melody_extract::{MelodyExtractorNew, analyze_shape};
//!
//! let sr = 44100_u32;
//! let hop = 512_u32;
//! // Generate 0.5 s of a 440 Hz sine wave
//! let samples: Vec<f32> = (0..sr / 2)
//!     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
//!     .collect();
//!
//! let contour = MelodyExtractorNew::extract(&samples, sr, hop);
//! assert!(!contour.notes.is_empty());
//!
//! let shape = analyze_shape(&contour);
//! println!("Contour shape: {:?}", shape);
//! ```

#![allow(dead_code)]


// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Voiced frames must have energy ≥ mean × this factor.
const VOICED_THRESHOLD_FACTOR: f32 = 0.15;

/// Maximum frequency we consider as a valid melodic pitch (Hz).
const MAX_PITCH_HZ: f32 = 2000.0;

/// Minimum frequency we consider as a valid melodic pitch (Hz).
const MIN_PITCH_HZ: f32 = 60.0;

/// Two consecutive voiced hops are merged into the same note when their pitch
/// differs by less than this many semitones.
const PITCH_MERGE_SEMITONES: f32 = 1.5;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A single detected melody note.
#[derive(Debug, Clone, PartialEq)]
pub struct MelodyNote {
    /// Estimated fundamental frequency in Hz.
    pub frequency_hz: f32,
    /// Frame index at which this note begins.
    pub onset_frame: u64,
    /// Duration of the note in frames.
    pub duration_frames: u64,
    /// Salience in \[0.0, 1.0\] (energy-normalised).
    pub salience: f32,
}

/// A sequence of melody notes with timing context.
#[derive(Debug, Clone)]
pub struct MelodyContour {
    /// Ordered list of extracted melody notes.
    pub notes: Vec<MelodyNote>,
    /// Audio sample rate (Hz).
    pub sample_rate: u32,
    /// Analysis hop size (samples).
    pub hop_size: u32,
}

impl MelodyContour {
    /// Total duration of the contour in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_secs(&self) -> f32 {
        if self.notes.is_empty() || self.sample_rate == 0 {
            return 0.0;
        }
        // notes is non-empty: checked by the early return above.
        let last = if let Some(n) = self.notes.last() {
            n
        } else {
            return 0.0;
        };
        let end_frame = last.onset_frame + last.duration_frames;
        end_frame as f32 * self.hop_size as f32 / self.sample_rate as f32
    }

    /// Pitch range `(min_hz, max_hz)` across all notes.
    ///
    /// Returns `(0.0, 0.0)` when the contour is empty.
    #[must_use]
    pub fn pitch_range_hz(&self) -> (f32, f32) {
        if self.notes.is_empty() {
            return (0.0, 0.0);
        }
        let min = self
            .notes
            .iter()
            .map(|n| n.frequency_hz)
            .fold(f32::INFINITY, f32::min);
        let max = self
            .notes
            .iter()
            .map(|n| n.frequency_hz)
            .fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    }

    /// Salience-weighted average pitch in Hz across all notes.
    ///
    /// Returns `0.0` when the contour is empty.
    #[must_use]
    pub fn average_pitch_hz(&self) -> f32 {
        if self.notes.is_empty() {
            return 0.0;
        }
        let weight_sum: f32 = self.notes.iter().map(|n| n.salience.max(0.0)).sum();
        if weight_sum < f32::EPSILON {
            return self.notes.iter().map(|n| n.frequency_hz).sum::<f32>()
                / self.notes.len() as f32;
        }
        self.notes
            .iter()
            .map(|n| n.frequency_hz * n.salience.max(0.0))
            .sum::<f32>()
            / weight_sum
    }

    /// Semitone interval sequence between consecutive notes.
    ///
    /// Positive = ascending, negative = descending.
    /// Returns an empty `Vec` when the contour contains fewer than 2 notes.
    #[must_use]
    pub fn interval_sequence(&self) -> Vec<f32> {
        if self.notes.len() < 2 {
            return Vec::new();
        }
        self.notes
            .windows(2)
            .map(|w| {
                let ratio = w[1].frequency_hz / w[0].frequency_hz.max(f32::EPSILON);
                12.0 * ratio.log2()
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Contour shape
// ─────────────────────────────────────────────────────────────────────────────

/// Qualitative shape of a melody contour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContourShape {
    /// Pitches generally rising from start to end.
    Ascending,
    /// Pitches generally falling from start to end.
    Descending,
    /// Pitches rise then fall (arc shape).
    Arch,
    /// Pitches fall then rise (valley shape).
    Valley,
    /// No significant pitch movement — stays within ±1 semitone.
    Flat,
    /// No dominant direction — mixed movements.
    Complex,
}

/// Classify the melodic shape of a [`MelodyContour`].
///
/// Uses the first/last third average pitch to detect arches and valleys, and
/// linear regression slope to discriminate ascending from descending.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn analyze_shape(contour: &MelodyContour) -> ContourShape {
    let notes = &contour.notes;

    if notes.len() < 2 {
        return ContourShape::Flat;
    }

    // Linear regression on pitch values vs. note index.
    let n = notes.len() as f32;
    let mean_x = (n - 1.0) / 2.0;
    let mean_y = notes.iter().map(|n| n.frequency_hz).sum::<f32>() / n;

    let mut sum_xy = 0.0_f32;
    let mut sum_xx = 0.0_f32;

    for (i, note) in notes.iter().enumerate() {
        let x = i as f32 - mean_x;
        let y = note.frequency_hz - mean_y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    // Pitch range in semitones
    let (min_hz, max_hz) = contour.pitch_range_hz();
    let range_semitones = if min_hz > f32::EPSILON {
        12.0 * (max_hz / min_hz).log2()
    } else {
        0.0
    };

    // Flat: less than 1.5 semitone range
    if range_semitones < 1.5 {
        return ContourShape::Flat;
    }

    let slope = if sum_xx > f32::EPSILON {
        sum_xy / sum_xx
    } else {
        return ContourShape::Flat;
    };

    // Normalise slope to mean pitch
    let normalised_slope = if mean_y > f32::EPSILON {
        slope / mean_y
    } else {
        slope
    };

    // Detect arch or valley using three-segment averages.
    if notes.len() >= 6 {
        let third = (notes.len() / 3).max(1);
        let start_avg: f32 =
            notes[..third].iter().map(|n| n.frequency_hz).sum::<f32>() / third as f32;
        let mid_avg: f32 = notes[third..notes.len() - third]
            .iter()
            .map(|n| n.frequency_hz)
            .sum::<f32>()
            / (notes.len() - 2 * third) as f32;
        let end_avg: f32 = notes[notes.len() - third..]
            .iter()
            .map(|n| n.frequency_hz)
            .sum::<f32>()
            / third as f32;

        let mid_higher = mid_avg > start_avg * 1.02 && mid_avg > end_avg * 1.02;
        let mid_lower = mid_avg < start_avg * 0.98 && mid_avg < end_avg * 0.98;

        if mid_higher {
            return ContourShape::Arch;
        }
        if mid_lower {
            return ContourShape::Valley;
        }
    }

    // Ascending / Descending / Complex based on slope magnitude.
    // Threshold: 0.002 per note relative to mean pitch
    if normalised_slope > 0.002 {
        return ContourShape::Ascending;
    }
    if normalised_slope < -0.002 {
        return ContourShape::Descending;
    }

    ContourShape::Complex
}

// ─────────────────────────────────────────────────────────────────────────────
// Melody extractor
// ─────────────────────────────────────────────────────────────────────────────

/// Extracts a melody contour from raw mono PCM audio.
#[derive(Debug, Default, Clone, Copy)]
pub struct MelodyExtractorNew;

impl MelodyExtractorNew {
    /// Extract a [`MelodyContour`] from mono PCM `samples`.
    ///
    /// # Arguments
    ///
    /// * `samples`     — mono f32 PCM samples
    /// * `sample_rate` — audio sample rate in Hz
    /// * `hop_size`    — analysis hop size in samples
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn extract(samples: &[f32], sample_rate: u32, hop_size: u32) -> MelodyContour {
        if samples.is_empty() || sample_rate == 0 || hop_size == 0 {
            return MelodyContour {
                notes: Vec::new(),
                sample_rate,
                hop_size,
            };
        }

        let hop = hop_size as usize;
        let sr = sample_rate as f32;
        let num_hops = samples.len() / hop;

        if num_hops == 0 {
            return MelodyContour {
                notes: Vec::new(),
                sample_rate,
                hop_size,
            };
        }

        // ── Per-hop energy ───────────────────────────────────────────────
        let energies: Vec<f32> = (0..num_hops)
            .map(|h| {
                let start = h * hop;
                let end = (start + hop).min(samples.len());
                let frame = &samples[start..end];
                let rms = frame.iter().map(|&s| s * s).sum::<f32>() / frame.len() as f32;
                rms.sqrt()
            })
            .collect();

        let mean_energy: f32 = energies.iter().sum::<f32>() / energies.len() as f32;
        let voiced_threshold = mean_energy * VOICED_THRESHOLD_FACTOR;

        // ── Per-hop pitch estimation (voiced hops only) ──────────────────
        // Use zero-crossing-based period detection: estimate the average distance
        // between consecutive zero-crossings (positive-going), then derive f0.
        let pitch_estimates: Vec<Option<f32>> = (0..num_hops)
            .map(|h| {
                if energies[h] < voiced_threshold {
                    return None;
                }
                let start = h * hop;
                let end = (start + hop).min(samples.len());
                let frame = &samples[start..end];
                estimate_pitch_zcr(frame, sr)
            })
            .collect();

        // ── Group consecutive voiced hops with similar pitch into notes ──
        let notes = group_into_notes(&pitch_estimates, &energies, mean_energy);

        MelodyContour {
            notes,
            sample_rate,
            hop_size,
        }
    }
}

/// Estimate fundamental frequency using zero-crossing period detection.
///
/// Finds all positive-going zero crossings, computes the mean spacing between
/// them, and converts to Hz.  Returns `None` when fewer than 2 crossings are
/// found or the implied frequency is outside `[MIN_PITCH_HZ, MAX_PITCH_HZ]`.
#[allow(clippy::cast_precision_loss)]
fn estimate_pitch_zcr(frame: &[f32], sample_rate: f32) -> Option<f32> {
    // Collect positive-going zero crossing sample indices.
    let crossings: Vec<usize> = frame
        .windows(2)
        .enumerate()
        .filter_map(|(i, w)| {
            if w[0] < 0.0 && w[1] >= 0.0 {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    if crossings.len() < 2 {
        return None;
    }

    // Mean period in samples = mean gap between consecutive positive crossings.
    let gaps: Vec<f32> = crossings
        .windows(2)
        .map(|w| (w[1] - w[0]) as f32)
        .collect();

    let mean_gap = gaps.iter().sum::<f32>() / gaps.len() as f32;

    if mean_gap < 1.0 {
        return None;
    }

    let freq = sample_rate / mean_gap;
    if (MIN_PITCH_HZ..=MAX_PITCH_HZ).contains(&freq) {
        Some(freq)
    } else {
        None
    }
}

/// Group a sequence of hop-level pitch estimates into discrete notes.
///
/// Consecutive hops are merged when:
/// - Both hops have a pitch estimate, and
/// - The pitches differ by < `PITCH_MERGE_SEMITONES`.
///
/// The resulting note's frequency is the energy-weighted mean pitch of its hops.
#[allow(clippy::cast_precision_loss)]
fn group_into_notes(
    pitches: &[Option<f32>],
    energies: &[f32],
    mean_energy: f32,
) -> Vec<MelodyNote> {
    let mut notes: Vec<MelodyNote> = Vec::new();

    // Running accumulator for the current note group.
    let mut group_start: Option<u64> = None;
    let mut group_hz_accum: f32 = 0.0;
    let mut group_energy_accum: f32 = 0.0;
    let mut group_len: u64 = 0;
    let mut group_last_hz: f32 = 0.0;

    let flush = |notes: &mut Vec<MelodyNote>,
                 start: u64,
                 hz_accum: f32,
                 energy_accum: f32,
                 len: u64,
                 mean_e: f32| {
        if len == 0 || hz_accum < f32::EPSILON {
            return;
        }
        let avg_hz = hz_accum / len as f32;
        let salience = if mean_e > f32::EPSILON {
            (energy_accum / len as f32 / mean_e).min(1.0)
        } else {
            0.0
        };
        notes.push(MelodyNote {
            frequency_hz: avg_hz,
            onset_frame: start,
            duration_frames: len,
            salience,
        });
    };

    for (i, pitch_opt) in pitches.iter().enumerate() {
        match pitch_opt {
            None => {
                // Unvoiced hop: flush the current group if any.
                if let Some(start) = group_start.take() {
                    flush(
                        &mut notes,
                        start,
                        group_hz_accum,
                        group_energy_accum,
                        group_len,
                        mean_energy,
                    );
                    group_hz_accum = 0.0;
                    group_energy_accum = 0.0;
                    group_len = 0;
                    group_last_hz = 0.0;
                }
            }
            Some(hz) => {
                let hz = *hz;
                let energy = energies.get(i).copied().unwrap_or(0.0);

                // Check if this hop merges with the running group.
                let same_note = if group_last_hz > f32::EPSILON {
                    let semitone_diff =
                        (12.0_f32 * (hz / group_last_hz.max(f32::EPSILON)).log2()).abs();
                    semitone_diff < PITCH_MERGE_SEMITONES
                } else {
                    false
                };

                if same_note {
                    group_hz_accum += hz;
                    group_energy_accum += energy;
                    group_len += 1;
                    group_last_hz = hz;
                } else {
                    // Flush previous group and start a new one.
                    if let Some(start) = group_start.take() {
                        flush(
                            &mut notes,
                            start,
                            group_hz_accum,
                            group_energy_accum,
                            group_len,
                            mean_energy,
                        );
                    }
                    group_start = Some(i as u64);
                    group_hz_accum = hz;
                    group_energy_accum = energy;
                    group_len = 1;
                    group_last_hz = hz;
                }
            }
        }
    }

    // Flush any remaining group.
    if let Some(start) = group_start {
        flush(
            &mut notes,
            start,
            group_hz_accum,
            group_energy_accum,
            group_len,
            mean_energy,
        );
    }

    notes
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    const SR: u32 = 44100;
    const HOP: u32 = 512;

    fn sine(freq_hz: f32, duration_secs: f32) -> Vec<f32> {
        let n = (SR as f32 * duration_secs) as usize;
        (0..n)
            .map(|i| (TAU * freq_hz * i as f32 / SR as f32).sin())
            .collect()
    }

    // ── Extraction tests ──

    #[test]
    fn test_silence_yields_empty_contour() {
        let silence = vec![0.0_f32; SR as usize];
        let contour = MelodyExtractorNew::extract(&silence, SR, HOP);
        assert!(
            contour.notes.is_empty(),
            "silence should yield no melody notes"
        );
    }

    #[test]
    fn test_empty_samples_empty_contour() {
        let contour = MelodyExtractorNew::extract(&[], SR, HOP);
        assert!(contour.notes.is_empty());
    }

    #[test]
    fn test_single_frequency_yields_at_least_one_note() {
        // 0.5 s of 440 Hz sine → should detect at least one note near 440 Hz
        let samples = sine(440.0, 0.5);
        let contour = MelodyExtractorNew::extract(&samples, SR, HOP);
        assert!(
            !contour.notes.is_empty(),
            "440 Hz sine should yield at least one melody note"
        );
        // All detected notes should be in a reasonable range around 440 Hz
        for note in &contour.notes {
            assert!(
                note.frequency_hz > 100.0 && note.frequency_hz < 2000.0,
                "note frequency out of range: {}",
                note.frequency_hz
            );
        }
    }

    #[test]
    fn test_duration_secs_correct() {
        let samples = sine(440.0, 1.0);
        let contour = MelodyExtractorNew::extract(&samples, SR, HOP);
        if !contour.notes.is_empty() {
            let dur = contour.duration_secs();
            assert!(
                dur > 0.0 && dur <= 1.1,
                "duration should be ≤ 1.1 s for 1 s input, got {}",
                dur
            );
        }
    }

    #[test]
    fn test_pitch_range_hz() {
        let samples = sine(440.0, 0.5);
        let contour = MelodyExtractorNew::extract(&samples, SR, HOP);
        let (min, max) = contour.pitch_range_hz();
        if contour.notes.is_empty() {
            assert!((min).abs() < f32::EPSILON && (max).abs() < f32::EPSILON);
        } else {
            assert!(min <= max, "min pitch should be ≤ max pitch");
            assert!(min > 0.0, "min pitch should be positive");
        }
    }

    #[test]
    fn test_interval_sequence_empty_for_single_note() {
        // Build a contour with exactly one note.
        let contour = MelodyContour {
            notes: vec![MelodyNote {
                frequency_hz: 440.0,
                onset_frame: 0,
                duration_frames: 10,
                salience: 0.8,
            }],
            sample_rate: SR,
            hop_size: HOP,
        };
        assert!(contour.interval_sequence().is_empty());
    }

    #[test]
    fn test_interval_sequence_octave() {
        // Two notes an octave apart → interval = +12 semitones
        let contour = MelodyContour {
            notes: vec![
                MelodyNote {
                    frequency_hz: 440.0,
                    onset_frame: 0,
                    duration_frames: 5,
                    salience: 0.9,
                },
                MelodyNote {
                    frequency_hz: 880.0,
                    onset_frame: 5,
                    duration_frames: 5,
                    salience: 0.9,
                },
            ],
            sample_rate: SR,
            hop_size: HOP,
        };
        let intervals = contour.interval_sequence();
        assert_eq!(intervals.len(), 1);
        assert!(
            (intervals[0] - 12.0).abs() < 0.1,
            "octave should be 12 semitones, got {}",
            intervals[0]
        );
    }

    // ── Contour shape tests ──

    #[test]
    fn test_flat_contour_shape() {
        let contour = MelodyContour {
            notes: (0..8)
                .map(|i| MelodyNote {
                    frequency_hz: 440.0,
                    onset_frame: i,
                    duration_frames: 1,
                    salience: 0.8,
                })
                .collect(),
            sample_rate: SR,
            hop_size: HOP,
        };
        assert_eq!(analyze_shape(&contour), ContourShape::Flat);
    }

    #[test]
    fn test_ascending_contour_shape() {
        // Pitches steadily increase from 200 Hz to 800 Hz.
        let contour = MelodyContour {
            notes: (0..12_u64)
                .map(|i| MelodyNote {
                    frequency_hz: 200.0 + i as f32 * 50.0,
                    onset_frame: i,
                    duration_frames: 1,
                    salience: 0.8,
                })
                .collect(),
            sample_rate: SR,
            hop_size: HOP,
        };
        let shape = analyze_shape(&contour);
        assert_eq!(shape, ContourShape::Ascending);
    }

    #[test]
    fn test_descending_contour_shape() {
        // Pitches steadily decrease from 800 Hz to 200 Hz.
        let contour = MelodyContour {
            notes: (0..12_u64)
                .map(|i| MelodyNote {
                    frequency_hz: 800.0 - i as f32 * 50.0,
                    onset_frame: i,
                    duration_frames: 1,
                    salience: 0.8,
                })
                .collect(),
            sample_rate: SR,
            hop_size: HOP,
        };
        let shape = analyze_shape(&contour);
        assert_eq!(shape, ContourShape::Descending);
    }

    #[test]
    fn test_arch_contour_shape() {
        // Start low, peak in middle, end low.
        let pitches = [
            200.0, 250.0, 350.0, 500.0, 650.0, 700.0, 680.0, 550.0, 400.0, 280.0, 220.0, 200.0,
        ];
        let contour = MelodyContour {
            notes: pitches
                .iter()
                .enumerate()
                .map(|(i, &hz)| MelodyNote {
                    frequency_hz: hz,
                    onset_frame: i as u64,
                    duration_frames: 1,
                    salience: 0.8,
                })
                .collect(),
            sample_rate: SR,
            hop_size: HOP,
        };
        let shape = analyze_shape(&contour);
        assert_eq!(shape, ContourShape::Arch, "pitched arch should be Arch");
    }
}
