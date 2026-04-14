//! Tempo stability analysis module.
//!
//! Measures tempo consistency from a sequence of [`TempoEvent`]s (frame-indexed
//! BPM snapshots with confidence values). The analysis computes mean BPM,
//! standard deviation, coefficient-of-variation-based stability score, tap
//! variance, and inter-event jitter in milliseconds.  A [`TempoClass`] enum
//! provides a human-readable interpretation of the result.
//!
//! # Example
//!
//! ```
//! use oximedia_mir::tempo_stability::{TempoEvent, TempoStabilityAnalyzer};
//!
//! let events: Vec<TempoEvent> = (0..16)
//!     .map(|i| TempoEvent { frame: i * 512, bpm: 120.0, confidence: 0.9 })
//!     .collect();
//! let report = TempoStabilityAnalyzer::analyze(&events);
//! assert!((report.mean_bpm - 120.0).abs() < 0.1);
//! assert!(report.stability_score > 0.99);
//! ```

#![allow(dead_code)]

/// A single BPM observation at a given audio frame with a detection confidence.
#[derive(Debug, Clone, PartialEq)]
pub struct TempoEvent {
    /// Frame index (sample offset / hop_size).
    pub frame: u64,
    /// Detected BPM at this frame.
    pub bpm: f32,
    /// Detection confidence in \[0.0, 1.0\].
    pub confidence: f32,
}

/// Full stability report derived from a sequence of [`TempoEvent`]s.
#[derive(Debug, Clone, PartialEq)]
pub struct TempoStabilityReport {
    /// Confidence-weighted mean BPM across all events.
    pub mean_bpm: f32,
    /// Population standard deviation of observed BPM values.
    pub std_dev: f32,
    /// Stability score in \[0.0, 1.0\] — 1.0 = perfectly steady, 0.0 = chaotic.
    pub stability_score: f32,
    /// Variance of inter-event BPM differences (tap variance).
    pub tap_variance: f32,
    /// Estimated jitter: root-mean-square deviation of actual interval times
    /// from ideal constant-BPM intervals, expressed in milliseconds.
    pub jitter_ms: f32,
}

/// High-level classification of a track's rhythmic feel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TempoClass {
    /// BPM is near-constant — machine-quantized or click-track locked.
    RigidlyQuantized,
    /// Small human timing variations — live performance feel.
    HumanGroove,
    /// No dominant tempo — freely flowing time.
    FreeTime,
    /// Expressive slow/fast fluctuations without a clear direction.
    Rubato,
    /// Systematic tempo increase over the event window.
    Accelerating,
    /// Systematic tempo decrease over the event window.
    Decelerating,
}

/// Stateless analyzer: all methods are pure functions operating on slices.
#[derive(Debug, Default, Clone, Copy)]
pub struct TempoStabilityAnalyzer;

impl TempoStabilityAnalyzer {
    /// Analyze a slice of [`TempoEvent`]s and return a [`TempoStabilityReport`].
    ///
    /// Returns a zeroed report when fewer than 2 events are supplied.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(events: &[TempoEvent]) -> TempoStabilityReport {
        if events.len() < 2 {
            return TempoStabilityReport {
                mean_bpm: events.first().map_or(0.0, |e| e.bpm),
                std_dev: 0.0,
                stability_score: if events.is_empty() { 0.0 } else { 1.0 },
                tap_variance: 0.0,
                jitter_ms: 0.0,
            };
        }

        // ── Confidence-weighted mean ─────────────────────────────────────
        let weight_sum: f32 = events.iter().map(|e| e.confidence.max(0.0)).sum();
        let mean_bpm = if weight_sum > f32::EPSILON {
            events
                .iter()
                .map(|e| e.bpm * e.confidence.max(0.0))
                .sum::<f32>()
                / weight_sum
        } else {
            events.iter().map(|e| e.bpm).sum::<f32>() / events.len() as f32
        };

        // ── Population standard deviation of BPM ────────────────────────
        let variance_bpm: f32 = events
            .iter()
            .map(|e| (e.bpm - mean_bpm).powi(2))
            .sum::<f32>()
            / events.len() as f32;
        let std_dev = variance_bpm.sqrt();

        // ── Coefficient of variation → stability score ───────────────────
        let cv = if mean_bpm.abs() > f32::EPSILON {
            std_dev / mean_bpm
        } else {
            0.0
        };
        let stability_score = (1.0_f32 - cv.min(1.0)).max(0.0);

        // ── Tap variance: variance of successive BPM differences ─────────
        let diffs: Vec<f32> = events
            .windows(2)
            .map(|w| (w[1].bpm - w[0].bpm).powi(2))
            .collect();
        let tap_variance = if diffs.is_empty() {
            0.0
        } else {
            diffs.iter().sum::<f32>() / diffs.len() as f32
        };

        // ── Jitter (ms): RMS deviation of actual intervals from ideal ────
        // Ideal interval at mean_bpm: 60_000 / mean_bpm milliseconds.
        // Actual interval derived from frame differences.
        // We store frame timestamps; without a sample-rate we use frame count
        // differences.  To express jitter in a physically meaningful unit we
        // convert frames to ms assuming the caller provided consistent frames
        // and that the hop-rate is implicitly 1 frame = 1 unit.
        // Instead, we derive the ideal interval in BPM-space:
        //   ideal_interval_bpm_frames = frames_per_beat = 1 beat
        //   actual_interval_bpm_change = bpm[i+1] - bpm[i]  (already captured)
        //
        // A more robust jitter estimate uses the actual frame timestamps to
        // compute expected beat positions:
        //   frame_diff[i] = frame[i+1] - frame[i]
        // and compares those to the median frame diff.
        let frame_diffs: Vec<f32> = events
            .windows(2)
            .map(|w| (w[1].frame as f32) - (w[0].frame as f32))
            .collect();

        let jitter_ms = if frame_diffs.is_empty() {
            0.0
        } else {
            // Median frame diff as "ideal"
            let mut sorted_diffs = frame_diffs.clone();
            sorted_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median_diff = if sorted_diffs.len() % 2 == 0 {
                let mid = sorted_diffs.len() / 2;
                (sorted_diffs[mid - 1] + sorted_diffs[mid]) * 0.5
            } else {
                sorted_diffs[sorted_diffs.len() / 2]
            };

            // RMS deviation from the median; scale by (60_000 / mean_bpm) to get ms
            // but since frames are dimensionless we treat 1 frame unit = 1 ms for
            // portability (the caller can scale by hop_size / sample_rate if needed).
            let rms_dev = (frame_diffs
                .iter()
                .map(|&d| (d - median_diff).powi(2))
                .sum::<f32>()
                / frame_diffs.len() as f32)
                .sqrt();

            // Convert frame-deviation to ms equivalent using mean BPM
            // rms_dev is in "frames"; 60_000 / mean_bpm ms per beat is one natural scale.
            if mean_bpm > f32::EPSILON && median_diff > f32::EPSILON {
                rms_dev / median_diff * (60_000.0 / mean_bpm)
            } else {
                rms_dev
            }
        };

        TempoStabilityReport {
            mean_bpm,
            std_dev,
            stability_score,
            tap_variance,
            jitter_ms,
        }
    }

    /// Classify the rhythmic feel from a [`TempoStabilityReport`].
    ///
    /// The classification uses the stability score and detected trends:
    ///
    /// | Condition                          | Class             |
    /// |------------------------------------|-------------------|
    /// | stability ≥ 0.97                   | `RigidlyQuantized`|
    /// | stability ≥ 0.85                   | `HumanGroove`     |
    /// | tap_variance small + accel. up     | `Accelerating`    |
    /// | tap_variance small + accel. down   | `Decelerating`    |
    /// | stability ≥ 0.50                   | `Rubato`          |
    /// | else                               | `FreeTime`        |
    #[must_use]
    pub fn classify(report: &TempoStabilityReport, events: &[TempoEvent]) -> TempoClass {
        if report.stability_score >= 0.97 {
            return TempoClass::RigidlyQuantized;
        }
        if report.stability_score >= 0.85 {
            return TempoClass::HumanGroove;
        }

        // Detect systematic trend using linear regression on BPM values.
        if events.len() >= 3 {
            if let Some(slope) = linear_slope_bpm(events) {
                let normalised = if report.mean_bpm > f32::EPSILON {
                    slope / report.mean_bpm
                } else {
                    0.0
                };
                // A clear trend: |normalised slope| > 0.005 per event
                if normalised > 0.005 {
                    return TempoClass::Accelerating;
                }
                if normalised < -0.005 {
                    return TempoClass::Decelerating;
                }
            }
        }

        if report.stability_score >= 0.50 {
            return TempoClass::Rubato;
        }

        TempoClass::FreeTime
    }
}

/// Compute the OLS linear regression slope of BPM values against event index.
///
/// Returns `None` when fewer than 2 events are available.
#[allow(clippy::cast_precision_loss)]
fn linear_slope_bpm(events: &[TempoEvent]) -> Option<f32> {
    if events.len() < 2 {
        return None;
    }
    let n = events.len() as f32;
    let mean_x = (n - 1.0) / 2.0;
    let mean_y: f32 = events.iter().map(|e| e.bpm).sum::<f32>() / n;

    let mut sum_xy = 0.0_f32;
    let mut sum_xx = 0.0_f32;

    for (i, event) in events.iter().enumerate() {
        let x = i as f32 - mean_x;
        let y = event.bpm - mean_y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    if sum_xx < f32::EPSILON {
        return None;
    }
    Some(sum_xy / sum_xx)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build N events at constant BPM with evenly-spaced frames.
    fn constant_events(bpm: f32, n: usize, hop: u64) -> Vec<TempoEvent> {
        (0..n)
            .map(|i| TempoEvent {
                frame: i as u64 * hop,
                bpm,
                confidence: 0.9,
            })
            .collect()
    }

    // ── Core metric tests ──

    #[test]
    fn test_empty_events_returns_zero_report() {
        let report = TempoStabilityAnalyzer::analyze(&[]);
        assert!((report.mean_bpm).abs() < f32::EPSILON);
        assert!((report.stability_score).abs() < f32::EPSILON);
    }

    #[test]
    fn test_single_event_returns_stability_one() {
        let events = vec![TempoEvent {
            frame: 0,
            bpm: 120.0,
            confidence: 1.0,
        }];
        let report = TempoStabilityAnalyzer::analyze(&events);
        assert!((report.mean_bpm - 120.0).abs() < 0.01);
        assert!((report.stability_score - 1.0).abs() < f32::EPSILON);
        assert!((report.std_dev).abs() < f32::EPSILON);
    }

    #[test]
    fn test_constant_bpm_perfect_stability() {
        let events = constant_events(120.0, 32, 512);
        let report = TempoStabilityAnalyzer::analyze(&events);
        assert!((report.mean_bpm - 120.0).abs() < 0.01);
        assert!((report.std_dev).abs() < 1e-4);
        assert!((report.stability_score - 1.0).abs() < 1e-4);
        assert!((report.tap_variance).abs() < 1e-6);
    }

    #[test]
    fn test_wildly_varying_bpm_low_stability() {
        let events: Vec<TempoEvent> = (0..20)
            .map(|i| TempoEvent {
                frame: i as u64 * 512,
                bpm: if i % 2 == 0 { 60.0 } else { 180.0 },
                confidence: 0.8,
            })
            .collect();
        let report = TempoStabilityAnalyzer::analyze(&events);
        assert!(
            report.stability_score < 0.5,
            "alternating 60/180 BPM should give low stability, got {}",
            report.stability_score
        );
    }

    #[test]
    fn test_std_dev_formula() {
        // Three events: [100, 120, 140] → mean=120, variance = (400+0+400)/3
        let events: Vec<TempoEvent> = vec![
            TempoEvent { frame: 0, bpm: 100.0, confidence: 1.0 },
            TempoEvent { frame: 512, bpm: 120.0, confidence: 1.0 },
            TempoEvent { frame: 1024, bpm: 140.0, confidence: 1.0 },
        ];
        let report = TempoStabilityAnalyzer::analyze(&events);
        let expected_std = ((800.0_f32 / 3.0_f32) as f32).sqrt();
        assert!(
            (report.std_dev - expected_std).abs() < 0.01,
            "std_dev mismatch: got {}, expected {}",
            report.std_dev,
            expected_std
        );
    }

    // ── Classification tests ──

    #[test]
    fn test_classify_rigid_quantization() {
        let events = constant_events(128.0, 16, 512);
        let report = TempoStabilityAnalyzer::analyze(&events);
        let class = TempoStabilityAnalyzer::classify(&report, &events);
        assert_eq!(class, TempoClass::RigidlyQuantized);
    }

    #[test]
    fn test_classify_free_time() {
        // Large BPM swings → stability near 0
        let events: Vec<TempoEvent> = (0..20)
            .map(|i| TempoEvent {
                frame: i as u64 * 512,
                bpm: 60.0 + (i as f32 * 37.0) % 120.0,
                confidence: 0.5,
            })
            .collect();
        let report = TempoStabilityAnalyzer::analyze(&events);
        let class = TempoStabilityAnalyzer::classify(&report, &events);
        // Should not be RigidlyQuantized or HumanGroove
        assert_ne!(class, TempoClass::RigidlyQuantized);
        assert_ne!(class, TempoClass::HumanGroove);
    }

    #[test]
    fn test_classify_accelerating() {
        // Monotonically increasing BPM
        let events: Vec<TempoEvent> = (0..20)
            .map(|i| TempoEvent {
                frame: i as u64 * 512,
                bpm: 80.0 + i as f32 * 5.0, // 80 → 175 BPM
                confidence: 0.9,
            })
            .collect();
        let report = TempoStabilityAnalyzer::analyze(&events);
        let class = TempoStabilityAnalyzer::classify(&report, &events);
        assert_eq!(class, TempoClass::Accelerating);
    }

    #[test]
    fn test_classify_decelerating() {
        // Monotonically decreasing BPM
        let events: Vec<TempoEvent> = (0..20)
            .map(|i| TempoEvent {
                frame: i as u64 * 512,
                bpm: 175.0 - i as f32 * 5.0, // 175 → 80 BPM
                confidence: 0.9,
            })
            .collect();
        let report = TempoStabilityAnalyzer::analyze(&events);
        let class = TempoStabilityAnalyzer::classify(&report, &events);
        assert_eq!(class, TempoClass::Decelerating);
    }

    #[test]
    fn test_jitter_zero_for_constant_bpm() {
        let events = constant_events(120.0, 16, 512);
        let report = TempoStabilityAnalyzer::analyze(&events);
        assert!(
            report.jitter_ms < 1e-3,
            "jitter should be ~0 for constant BPM, got {}",
            report.jitter_ms
        );
    }
}
