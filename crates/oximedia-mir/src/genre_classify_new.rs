//! Fine-grained genre sub-classification from spectral audio features.
//!
//! This module provides a rule-based genre classifier that maps low-level
//! spectral features onto a hierarchical two-level taxonomy: a [`GenreFamily`]
//! (broad category) and a `sub_genre` string within that family.  Up to the
//! top-3 most likely [`GenreTag`]s are returned per classification.
//!
//! The [`FeatureExtractor`] computes all required features from raw PCM samples
//! without requiring an FFT library — it uses a brute-force DFT for small
//! frame windows in the spectral rolloff and centroid computation, keeping the
//! implementation fully self-contained.
//!
//! # Example
//!
//! ```
//! use oximedia_mir::genre_classify_new::{FeatureExtractor, GenreClassifierNew};
//!
//! let samples: Vec<f32> = (0..44100)
//!     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
//!     .collect();
//! let features = FeatureExtractor::extract(&samples, 44100);
//! let tags = GenreClassifierNew::classify(&features);
//! assert!(tags.len() <= 3);
//! ```

#![allow(dead_code)]

use std::f32::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// High-level genre family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenreFamily {
    /// Electronic, EDM, synth-based music.
    Electronic,
    /// Rock and alternative rock.
    Rock,
    /// Classical and orchestral.
    Classical,
    /// Jazz and blues.
    Jazz,
    /// Hip-hop and rap.
    HipHop,
    /// Country.
    Country,
    /// Folk and acoustic.
    Folk,
    /// Latin music.
    Latin,
    /// World and traditional music.
    World,
    /// Ambient and drone.
    Ambient,
}

/// A tagged genre prediction with sub-genre detail.
#[derive(Debug, Clone, PartialEq)]
pub struct GenreTag {
    /// Top-level genre family.
    pub family: GenreFamily,
    /// Sub-genre label within the family (e.g. "progressive rock").
    pub sub_genre: String,
    /// Confidence score in \[0.0, 1.0\].
    pub confidence: f32,
}

/// Spectral features extracted from a short audio segment.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralFeatures {
    /// Weighted mean frequency of the power spectrum (Hz).
    pub spectral_centroid: f32,
    /// Frequency below which 85 % of spectral energy is contained (Hz).
    pub spectral_rolloff: f32,
    /// Rate of sign changes per second (proxy for noisiness / percussion).
    pub zero_crossing_rate: f32,
    /// Root-mean-square energy of the signal.
    pub rms_energy: f32,
    /// Mean frame-to-frame spectral change (0 if only one frame available).
    pub spectral_flux: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature extraction
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless spectral feature extractor.
#[derive(Debug, Default, Clone, Copy)]
pub struct FeatureExtractor;

impl FeatureExtractor {
    /// Extract [`SpectralFeatures`] from raw mono PCM samples.
    ///
    /// The algorithm:
    /// 1. Compute RMS energy and zero-crossing rate over the entire signal.
    /// 2. Divide the signal into non-overlapping frames of `FRAME_SIZE` samples.
    /// 3. For each frame compute a magnitude spectrum via a brute-force DFT
    ///    (only the positive half, up to `MAX_BINS` bins for speed).
    /// 4. Accumulate per-frame spectral centroid and rolloff; average across frames.
    /// 5. Compute spectral flux as mean absolute difference between consecutive
    ///    frame magnitude spectra.
    ///
    /// Uses `sample_rate` to convert bin indices to Hz.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn extract(samples: &[f32], sample_rate: u32) -> SpectralFeatures {
        if samples.is_empty() {
            return SpectralFeatures {
                spectral_centroid: 0.0,
                spectral_rolloff: 0.0,
                zero_crossing_rate: 0.0,
                rms_energy: 0.0,
                spectral_flux: 0.0,
            };
        }

        let n = samples.len();
        let sr = sample_rate as f32;

        // ── RMS energy ───────────────────────────────────────────────────
        let rms_energy = (samples.iter().map(|&s| s * s).sum::<f32>() / n as f32).sqrt();

        // ── Zero-crossing rate (crossings per second) ────────────────────
        let crossings = samples
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
        let zero_crossing_rate = crossings as f32 / (n as f32 / sr);

        // ── Frame-based spectral analysis ────────────────────────────────
        const FRAME_SIZE: usize = 512;
        const MAX_BINS: usize = 128; // compute only the first MAX_BINS frequency bins

        let num_frames = n / FRAME_SIZE;

        if num_frames == 0 {
            // Signal shorter than one frame: return energy-only features.
            return SpectralFeatures {
                spectral_centroid: 0.0,
                spectral_rolloff: 0.0,
                zero_crossing_rate,
                rms_energy,
                spectral_flux: 0.0,
            };
        }

        let bin_count = MAX_BINS.min(FRAME_SIZE / 2);
        let bin_hz = sr / FRAME_SIZE as f32;

        // Store per-frame magnitude spectra for flux computation.
        let mut frame_mags: Vec<Vec<f32>> = Vec::with_capacity(num_frames);

        let mut centroid_sum = 0.0_f32;
        let mut rolloff_sum = 0.0_f32;

        for frame_idx in 0..num_frames {
            let start = frame_idx * FRAME_SIZE;
            let frame = &samples[start..start + FRAME_SIZE];

            // Brute-force DFT magnitude for each bin k
            let mags: Vec<f32> = (0..bin_count)
                .map(|k| {
                    let phase_step = 2.0 * PI * k as f32 / FRAME_SIZE as f32;
                    let (re, im): (f32, f32) = frame
                        .iter()
                        .enumerate()
                        .map(|(n_idx, &s)| {
                            let angle = phase_step * n_idx as f32;
                            (s * angle.cos(), -s * angle.sin())
                        })
                        .fold((0.0, 0.0), |acc, (re, im)| (acc.0 + re, acc.1 + im));
                    (re * re + im * im).sqrt()
                })
                .collect();

            let total_mag: f32 = mags.iter().sum();

            // Spectral centroid
            let centroid = if total_mag > f32::EPSILON {
                mags.iter()
                    .enumerate()
                    .map(|(k, &m)| k as f32 * bin_hz * m)
                    .sum::<f32>()
                    / total_mag
            } else {
                0.0
            };
            centroid_sum += centroid;

            // Spectral rolloff (85th percentile)
            let threshold = 0.85 * total_mag;
            let mut running = 0.0_f32;
            let rolloff_bin = mags
                .iter()
                .enumerate()
                .find_map(|(k, &m)| {
                    running += m;
                    if running >= threshold {
                        Some(k)
                    } else {
                        None
                    }
                })
                .unwrap_or(bin_count.saturating_sub(1));
            rolloff_sum += rolloff_bin as f32 * bin_hz;

            frame_mags.push(mags);
        }

        let spectral_centroid = centroid_sum / num_frames as f32;
        let spectral_rolloff = rolloff_sum / num_frames as f32;

        // ── Spectral flux ────────────────────────────────────────────────
        let spectral_flux = if frame_mags.len() < 2 {
            0.0
        } else {
            let total_diff: f32 = frame_mags
                .windows(2)
                .map(|w| {
                    w[0].iter()
                        .zip(w[1].iter())
                        .map(|(&a, &b)| (b - a).abs())
                        .sum::<f32>()
                        / bin_count as f32
                })
                .sum();
            total_diff / (frame_mags.len() - 1) as f32
        };

        SpectralFeatures {
            spectral_centroid,
            spectral_rolloff,
            zero_crossing_rate,
            rms_energy,
            spectral_flux,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Genre classifier
// ─────────────────────────────────────────────────────────────────────────────

/// Rule-based multi-label genre classifier that returns up to 3 ranked tags.
#[derive(Debug, Default, Clone, Copy)]
pub struct GenreClassifierNew;

impl GenreClassifierNew {
    /// Classify genre from spectral features, returning ≤ 3 [`GenreTag`]s sorted
    /// by descending confidence.
    ///
    /// **Rules** (each family gets a raw score; top-3 are normalised and emitted):
    ///
    /// | Signal | Family hint |
    /// |--------|-------------|
    /// | High ZCR + high flux + high energy | Rock / Electronic |
    /// | Low centroid + low energy | Ambient / Classical |
    /// | Moderate centroid + moderate energy + low ZCR | Jazz |
    /// | High centroid + moderate flux + low ZCR | Electronic |
    /// | Low centroid + low ZCR + low energy | Ambient |
    /// | Moderate centroid + high flux | HipHop |
    #[must_use]
    pub fn classify(features: &SpectralFeatures) -> Vec<GenreTag> {
        let scores = Self::raw_scores(features);

        // Sort descending by score, keep top 3.
        let mut sorted = scores;
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(3);

        // Normalise so the top-3 confidences sum to ≤ 1.0
        let total: f32 = sorted.iter().map(|s| s.1).sum();
        sorted
            .into_iter()
            .filter(|s| s.1 > 0.0)
            .map(|(family, raw_score)| {
                let confidence = if total > f32::EPSILON {
                    raw_score / total
                } else {
                    0.0
                };
                GenreTag {
                    family,
                    sub_genre: sub_genre_for(family, features),
                    confidence,
                }
            })
            .collect()
    }

    /// Compute un-normalised scores for each [`GenreFamily`].
    fn raw_scores(f: &SpectralFeatures) -> Vec<(GenreFamily, f32)> {
        // Normalise raw features to a [0,1] range for scoring
        let centroid_n = (f.spectral_centroid / 8000.0).clamp(0.0, 1.0);
        let rolloff_n = (f.spectral_rolloff / 16_000.0).clamp(0.0, 1.0);
        let zcr_n = (f.zero_crossing_rate / 500.0).clamp(0.0, 1.0);
        let rms_n = (f.rms_energy / 0.5).clamp(0.0, 1.0);
        let flux_n = (f.spectral_flux / 20.0).clamp(0.0, 1.0);

        // Each rule contributes an additive score in [0, 4].
        vec![
            (
                GenreFamily::Electronic,
                // High centroid + moderate-high rolloff + moderate ZCR + moderate energy
                centroid_n * 1.2 + rolloff_n * 0.8 + (0.5 - (zcr_n - 0.3).abs()) + rms_n * 0.5,
            ),
            (
                GenreFamily::Rock,
                // High ZCR + high energy + moderate-to-high flux
                zcr_n * 1.5 + rms_n * 1.0 + flux_n * 0.8 + centroid_n * 0.5,
            ),
            (
                GenreFamily::Classical,
                // Low centroid + low ZCR + low energy + low flux
                (1.0 - centroid_n) * 1.2
                    + (1.0 - zcr_n) * 0.8
                    + (1.0 - rms_n) * 0.5
                    + (1.0 - flux_n) * 0.5,
            ),
            (
                GenreFamily::Jazz,
                // Moderate centroid + moderate energy + low ZCR + moderate flux
                (0.5 - (centroid_n - 0.3).abs()) * 2.0
                    + (1.0 - zcr_n) * 0.8
                    + rms_n * 0.5
                    + (0.5 - (flux_n - 0.2).abs()) * 0.5,
            ),
            (
                GenreFamily::HipHop,
                // Moderate centroid + high flux + moderate energy + low ZCR
                (0.5 - (centroid_n - 0.35).abs()) * 1.5
                    + flux_n * 1.0
                    + rms_n * 0.8
                    + (1.0 - zcr_n) * 0.5,
            ),
            (
                GenreFamily::Country,
                // Moderate centroid + moderate ZCR + moderate energy
                (0.5 - (centroid_n - 0.35).abs()) * 1.5
                    + (0.5 - (zcr_n - 0.25).abs()) * 1.0
                    + rms_n * 0.5,
            ),
            (
                GenreFamily::Folk,
                // Low-moderate centroid + low ZCR + low-moderate energy
                (1.0 - centroid_n) * 1.0
                    + (1.0 - zcr_n) * 0.8
                    + (0.5 - (rms_n - 0.15).abs()) * 0.8,
            ),
            (
                GenreFamily::Latin,
                // High flux + high energy + moderate centroid
                flux_n * 1.2 + rms_n * 1.0 + centroid_n * 0.6,
            ),
            (
                GenreFamily::World,
                // Moderate rolloff + moderate flux + moderate energy
                (0.5 - (rolloff_n - 0.4).abs()) * 1.5
                    + (0.5 - (flux_n - 0.3).abs()) * 1.0
                    + rms_n * 0.5,
            ),
            (
                GenreFamily::Ambient,
                // Low centroid + low ZCR + very low energy + very low flux
                (1.0 - centroid_n) * 1.5
                    + (1.0 - zcr_n) * 1.0
                    + (1.0 - rms_n) * 0.8
                    + (1.0 - flux_n) * 0.5,
            ),
        ]
    }
}

/// Choose a sub-genre label given the family and a hint from the features.
fn sub_genre_for(family: GenreFamily, f: &SpectralFeatures) -> String {
    let centroid = f.spectral_centroid;
    let energy = f.rms_energy;
    let zcr = f.zero_crossing_rate;
    let flux = f.spectral_flux;

    match family {
        GenreFamily::Electronic => {
            if flux > 5.0 {
                "techno".to_string()
            } else if centroid > 4000.0 {
                "EDM".to_string()
            } else {
                "synthwave".to_string()
            }
        }
        GenreFamily::Rock => {
            if energy > 0.3 && zcr > 200.0 {
                "heavy metal".to_string()
            } else if centroid > 3000.0 {
                "alternative rock".to_string()
            } else {
                "classic rock".to_string()
            }
        }
        GenreFamily::Classical => {
            if centroid < 1000.0 {
                "chamber music".to_string()
            } else if energy > 0.1 {
                "orchestral".to_string()
            } else {
                "solo piano".to_string()
            }
        }
        GenreFamily::Jazz => {
            if flux > 3.0 {
                "bebop".to_string()
            } else if energy < 0.05 {
                "cool jazz".to_string()
            } else {
                "smooth jazz".to_string()
            }
        }
        GenreFamily::HipHop => {
            if energy > 0.2 {
                "trap".to_string()
            } else {
                "lo-fi hip-hop".to_string()
            }
        }
        GenreFamily::Country => {
            if zcr > 150.0 {
                "country rock".to_string()
            } else {
                "traditional country".to_string()
            }
        }
        GenreFamily::Folk => {
            if centroid < 800.0 {
                "singer-songwriter".to_string()
            } else {
                "indie folk".to_string()
            }
        }
        GenreFamily::Latin => {
            if flux > 4.0 {
                "reggaeton".to_string()
            } else {
                "salsa".to_string()
            }
        }
        GenreFamily::World => "world music".to_string(),
        GenreFamily::Ambient => {
            if energy < 0.01 {
                "dark ambient".to_string()
            } else {
                "ambient".to_string()
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    // ── Feature extraction tests ──

    #[test]
    fn test_silence_features() {
        let silence = vec![0.0_f32; 8192];
        let f = FeatureExtractor::extract(&silence, 44100);
        assert!(f.rms_energy < 1e-6, "silence should have ~0 rms energy");
        assert!(f.zero_crossing_rate < 1.0, "silence has no zero crossings");
        assert!(f.spectral_centroid < 1.0, "silence centroid should be ~0");
    }

    #[test]
    fn test_empty_input_features() {
        let f = FeatureExtractor::extract(&[], 44100);
        assert!(f.rms_energy < f32::EPSILON);
        assert!(f.zero_crossing_rate < f32::EPSILON);
        assert!(f.spectral_centroid < f32::EPSILON);
        assert!(f.spectral_rolloff < f32::EPSILON);
    }

    #[test]
    fn test_sine_wave_rms_energy() {
        let samples: Vec<f32> = (0..44100)
            .map(|i| (TAU * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let f = FeatureExtractor::extract(&samples, 44100);
        // RMS of a full-scale sine is 1/sqrt(2) ≈ 0.707
        assert!(
            (f.rms_energy - (1.0_f32 / 2.0_f32.sqrt())).abs() < 0.01,
            "sine RMS should be ~0.707, got {}",
            f.rms_energy
        );
    }

    #[test]
    fn test_sine_spectral_centroid_reasonable() {
        // 440 Hz sine: centroid should be close to 440 Hz (within a wide margin
        // due to brute-force DFT with small bin count and no windowing)
        let samples: Vec<f32> = (0..8192)
            .map(|i| (TAU * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let f = FeatureExtractor::extract(&samples, 44100);
        assert!(
            f.spectral_centroid > 100.0 && f.spectral_centroid < 4000.0,
            "centroid for 440 Hz sine should be in plausible range, got {}",
            f.spectral_centroid
        );
    }

    #[test]
    fn test_zcr_sine_low() {
        // A 440 Hz sine at 44100 Hz crosses zero ~880 times per second
        let samples: Vec<f32> = (0..44100)
            .map(|i| (TAU * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let f = FeatureExtractor::extract(&samples, 44100);
        // Should be around 880 crossings/s; allow wide tolerance
        assert!(
            f.zero_crossing_rate > 200.0 && f.zero_crossing_rate < 2000.0,
            "ZCR of 440 Hz sine should be ~880/s, got {}",
            f.zero_crossing_rate
        );
    }

    // ── Classifier tests ──

    #[test]
    fn test_classifier_returns_at_most_3_tags() {
        let f = SpectralFeatures {
            spectral_centroid: 2000.0,
            spectral_rolloff: 5000.0,
            zero_crossing_rate: 100.0,
            rms_energy: 0.2,
            spectral_flux: 2.0,
        };
        let tags = GenreClassifierNew::classify(&f);
        assert!(tags.len() <= 3, "classifier must return at most 3 tags");
        assert!(!tags.is_empty(), "classifier should return at least 1 tag");
    }

    #[test]
    fn test_silence_classification_returns_ambient_or_classical() {
        // Silence-like features → low energy, low ZCR, low centroid
        let f = SpectralFeatures {
            spectral_centroid: 100.0,
            spectral_rolloff: 200.0,
            zero_crossing_rate: 1.0,
            rms_energy: 0.001,
            spectral_flux: 0.0,
        };
        let tags = GenreClassifierNew::classify(&f);
        assert!(!tags.is_empty());
        let top_family = tags[0].family;
        assert!(
            matches!(top_family, GenreFamily::Ambient | GenreFamily::Classical),
            "silence-like features should map to Ambient or Classical, got {:?}",
            top_family
        );
    }

    #[test]
    fn test_confidence_sum_le_one() {
        let f = SpectralFeatures {
            spectral_centroid: 3000.0,
            spectral_rolloff: 8000.0,
            zero_crossing_rate: 200.0,
            rms_energy: 0.3,
            spectral_flux: 5.0,
        };
        let tags = GenreClassifierNew::classify(&f);
        let total: f32 = tags.iter().map(|t| t.confidence).sum();
        assert!(
            (total - 1.0).abs() < 0.01 || total <= 1.0 + 1e-5,
            "top-3 confidence scores should sum to ≤ 1.0, got {}",
            total
        );
    }

    #[test]
    fn test_high_zcr_high_energy_tends_to_rock() {
        // High ZCR + high energy → Rock
        let f = SpectralFeatures {
            spectral_centroid: 3000.0,
            spectral_rolloff: 7000.0,
            zero_crossing_rate: 400.0,
            rms_energy: 0.4,
            spectral_flux: 8.0,
        };
        let tags = GenreClassifierNew::classify(&f);
        assert!(!tags.is_empty());
        let top = &tags[0];
        assert!(
            matches!(top.family, GenreFamily::Rock | GenreFamily::Electronic),
            "high ZCR + energy should lean Rock or Electronic, got {:?}",
            top.family
        );
    }

    #[test]
    fn test_sub_genre_string_non_empty() {
        let f = SpectralFeatures {
            spectral_centroid: 2000.0,
            spectral_rolloff: 5000.0,
            zero_crossing_rate: 100.0,
            rms_energy: 0.2,
            spectral_flux: 2.0,
        };
        let tags = GenreClassifierNew::classify(&f);
        for tag in &tags {
            assert!(
                !tag.sub_genre.is_empty(),
                "sub_genre must not be empty for {:?}",
                tag.family
            );
        }
    }

    #[test]
    fn test_tags_sorted_descending_confidence() {
        let f = SpectralFeatures {
            spectral_centroid: 1500.0,
            spectral_rolloff: 4000.0,
            zero_crossing_rate: 30.0,
            rms_energy: 0.05,
            spectral_flux: 0.5,
        };
        let tags = GenreClassifierNew::classify(&f);
        for w in tags.windows(2) {
            assert!(
                w[0].confidence >= w[1].confidence,
                "tags should be sorted descending: {} < {}",
                w[0].confidence,
                w[1].confidence
            );
        }
    }
}
