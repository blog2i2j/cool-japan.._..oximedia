//! Foley sound synchronization: event marker placement, sync point matching, and offset detection.
//!
//! This module provides tools for synchronizing recorded Foley audio to picture by detecting
//! onset events in audio waveforms and matching them to scene-level cue markers.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::error::{AudioPostError, AudioPostResult};
use crate::timecode::Timecode;
use serde::{Deserialize, Serialize};

/// A single sync point: a named moment in the picture that a Foley event must align to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncPoint {
    /// Human-readable label (e.g., "Footstep landing frame").
    pub label: String,
    /// Timecode of the sync point in the program timeline.
    pub timecode: Timecode,
    /// Tolerance window around the sync point (in seconds) within which a detected onset
    /// is considered matching.
    pub tolerance_secs: f64,
}

impl SyncPoint {
    /// Create a new sync point.
    ///
    /// # Errors
    ///
    /// Returns an error if `tolerance_secs` is negative.
    pub fn new(label: &str, timecode: Timecode, tolerance_secs: f64) -> AudioPostResult<Self> {
        if tolerance_secs < 0.0 {
            return Err(AudioPostError::Generic(format!(
                "tolerance_secs must be non-negative, got {tolerance_secs}"
            )));
        }
        Ok(Self {
            label: label.to_string(),
            timecode,
            tolerance_secs,
        })
    }
}

/// An onset event detected in a Foley recording.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnsetEvent {
    /// Position in the recording (seconds from start of the audio buffer).
    pub position_secs: f64,
    /// RMS energy level at the detected onset (linear, 0.0–1.0).
    pub energy: f32,
    /// Whether this onset has been matched to a sync point.
    pub matched: bool,
}

/// Result of matching a single sync point to a detected onset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncMatch {
    /// The sync point that was evaluated.
    pub sync_point: SyncPoint,
    /// The onset event that was chosen as the best match, if any.
    pub matched_onset: Option<OnsetEvent>,
    /// Offset in seconds from sync point to matched onset (positive = onset is late).
    pub offset_secs: Option<f64>,
    /// Whether the match falls within the tolerance window.
    pub within_tolerance: bool,
}

/// Configuration for the onset detector.
#[derive(Debug, Clone)]
pub struct OnsetDetectorConfig {
    /// Sample rate of the audio being analyzed.
    pub sample_rate: u32,
    /// Analysis window size in samples (power of two recommended).
    pub window_size: usize,
    /// Hop size between successive analysis frames in samples.
    pub hop_size: usize,
    /// RMS energy threshold below which frames are considered silence (linear, 0.0–1.0).
    pub silence_threshold: f32,
    /// Minimum energy rise (ratio) from one frame to the next to register an onset.
    pub onset_delta: f32,
    /// Minimum time between onsets in seconds (to suppress spurious detections).
    pub min_onset_gap_secs: f64,
}

impl OnsetDetectorConfig {
    /// Create a default configuration for 48 kHz audio.
    #[must_use]
    pub fn default_48k() -> Self {
        Self {
            sample_rate: 48000,
            window_size: 1024,
            hop_size: 512,
            silence_threshold: 0.001,
            onset_delta: 1.5,
            min_onset_gap_secs: 0.05,
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if any field is invalid.
    pub fn validate(&self) -> AudioPostResult<()> {
        if self.sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(self.sample_rate));
        }
        if self.window_size == 0 {
            return Err(AudioPostError::InvalidBufferSize(self.window_size));
        }
        if self.hop_size == 0 || self.hop_size > self.window_size {
            return Err(AudioPostError::InvalidBufferSize(self.hop_size));
        }
        if self.silence_threshold < 0.0 || self.silence_threshold > 1.0 {
            return Err(AudioPostError::Generic(format!(
                "silence_threshold must be 0.0–1.0, got {}",
                self.silence_threshold
            )));
        }
        if self.onset_delta <= 1.0 {
            return Err(AudioPostError::Generic(format!(
                "onset_delta must be > 1.0, got {}",
                self.onset_delta
            )));
        }
        Ok(())
    }
}

/// Onset detector operating on mono PCM `f32` samples.
pub struct OnsetDetector {
    config: OnsetDetectorConfig,
}

impl OnsetDetector {
    /// Create a new onset detector.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: OnsetDetectorConfig) -> AudioPostResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Detect onset events in a mono PCM buffer.
    ///
    /// The algorithm computes frame-level RMS energy and triggers an onset when the energy
    /// rises by at least `onset_delta` relative to the previous frame and exceeds
    /// `silence_threshold`.
    ///
    /// # Errors
    ///
    /// Returns an error if the audio buffer is empty.
    pub fn detect(&self, samples: &[f32]) -> AudioPostResult<Vec<OnsetEvent>> {
        if samples.is_empty() {
            return Err(AudioPostError::InvalidBufferSize(0));
        }

        let hop = self.config.hop_size;
        let win = self.config.window_size;
        let sr = self.config.sample_rate as f64;
        let min_gap_frames = (self.config.min_onset_gap_secs * sr / hop as f64).ceil() as usize;

        let mut events: Vec<OnsetEvent> = Vec::new();
        let mut prev_rms: f32 = 0.0;
        let mut frames_since_last_onset: usize = usize::MAX / 2;

        let num_frames = if samples.len() >= win {
            (samples.len() - win) / hop + 1
        } else {
            0
        };

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop;
            let end = (start + win).min(samples.len());
            let frame = &samples[start..end];

            // RMS energy
            let sum_sq: f32 = frame.iter().map(|&s| s * s).sum();
            let rms = (sum_sq / frame.len() as f32).sqrt();

            let is_onset = rms > self.config.silence_threshold
                && prev_rms > 0.0
                && rms / prev_rms >= self.config.onset_delta
                && frames_since_last_onset >= min_gap_frames;

            if is_onset {
                let position_secs = (frame_idx * hop) as f64 / sr;
                events.push(OnsetEvent {
                    position_secs,
                    energy: rms,
                    matched: false,
                });
                frames_since_last_onset = 0;
            } else {
                frames_since_last_onset = frames_since_last_onset.saturating_add(1);
            }

            prev_rms = rms;
        }

        Ok(events)
    }
}

/// Marker placement recommendation produced by `FoleySynchronizer`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkerPlacement {
    /// The sync point label.
    pub label: String,
    /// Recommended recording offset to apply (in seconds) so that the onset aligns to
    /// the sync point.  Positive value means the recording should be shifted earlier.
    pub recommended_offset_secs: f64,
    /// Confidence score in [0.0, 1.0] — higher is better.
    pub confidence: f32,
}

/// Synchronizer that matches detected onsets to sync points and computes timing offsets.
pub struct FoleySynchronizer {
    /// Sync points loaded into the session.
    sync_points: Vec<SyncPoint>,
    /// Reference start time of the program (seconds from session zero).
    program_start_secs: f64,
}

impl FoleySynchronizer {
    /// Create a new synchronizer.
    ///
    /// `program_start_secs` is the session-level position (in seconds) at which
    /// the program timeline begins — used to convert timecodes to absolute positions.
    #[must_use]
    pub fn new(program_start_secs: f64) -> Self {
        Self {
            sync_points: Vec::new(),
            program_start_secs,
        }
    }

    /// Add a sync point to the synchronizer.
    pub fn add_sync_point(&mut self, sp: SyncPoint) {
        self.sync_points.push(sp);
    }

    /// Return the number of loaded sync points.
    #[must_use]
    pub fn sync_point_count(&self) -> usize {
        self.sync_points.len()
    }

    /// Match detected onsets to sync points.
    ///
    /// For each sync point the closest onset within the tolerance window is selected.
    /// Returns a `SyncMatch` for every sync point (matched or unmatched).
    #[must_use]
    pub fn match_onsets(&self, onsets: &[OnsetEvent]) -> Vec<SyncMatch> {
        self.sync_points
            .iter()
            .map(|sp| {
                let sp_secs =
                    self.program_start_secs + sp.timecode.to_seconds();

                // Find the closest onset within tolerance
                let best = onsets
                    .iter()
                    .filter(|o| {
                        (o.position_secs - sp_secs).abs() <= sp.tolerance_secs
                    })
                    .min_by(|a, b| {
                        let da = (a.position_secs - sp_secs).abs();
                        let db = (b.position_secs - sp_secs).abs();
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    });

                let offset_secs = best.map(|o| o.position_secs - sp_secs);
                let within_tolerance = offset_secs
                    .map(|off| off.abs() <= sp.tolerance_secs)
                    .unwrap_or(false);

                SyncMatch {
                    sync_point: sp.clone(),
                    matched_onset: best.cloned(),
                    offset_secs,
                    within_tolerance,
                }
            })
            .collect()
    }

    /// Generate marker placement recommendations from a set of sync matches.
    ///
    /// Only matches that fall within tolerance produce recommendations.
    #[must_use]
    pub fn generate_placements(&self, matches: &[SyncMatch]) -> Vec<MarkerPlacement> {
        matches
            .iter()
            .filter(|m| m.within_tolerance)
            .map(|m| {
                let offset = m.offset_secs.unwrap_or(0.0);
                // Confidence decays linearly from 1.0 (perfect alignment) to 0.0 (at tolerance edge)
                let confidence = if m.sync_point.tolerance_secs > 0.0 {
                    let ratio = offset.abs() / m.sync_point.tolerance_secs;
                    (1.0 - ratio).clamp(0.0, 1.0) as f32
                } else {
                    1.0_f32
                };
                MarkerPlacement {
                    label: m.sync_point.label.clone(),
                    recommended_offset_secs: -offset,
                    confidence,
                }
            })
            .collect()
    }

    /// Compute the global timing offset from a collection of individual match offsets.
    ///
    /// Uses the median of all within-tolerance offsets for robustness against outliers.
    /// Returns `None` if there are no within-tolerance matches.
    #[must_use]
    pub fn compute_global_offset(matches: &[SyncMatch]) -> Option<f64> {
        let mut offsets: Vec<f64> = matches
            .iter()
            .filter(|m| m.within_tolerance)
            .filter_map(|m| m.offset_secs)
            .collect();

        if offsets.is_empty() {
            return None;
        }

        offsets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = offsets.len() / 2;
        let median = if offsets.len() % 2 == 1 {
            offsets[mid]
        } else {
            (offsets[mid - 1] + offsets[mid]) / 2.0
        };
        Some(median)
    }
}

/// Apply a timing offset to a collection of onset events (e.g., for preview / correction).
///
/// Shifts `position_secs` of every event by `offset_secs` and clamps to zero.
#[must_use]
pub fn apply_offset_to_onsets(onsets: &[OnsetEvent], offset_secs: f64) -> Vec<OnsetEvent> {
    onsets
        .iter()
        .map(|o| OnsetEvent {
            position_secs: (o.position_secs + offset_secs).max(0.0),
            energy: o.energy,
            matched: o.matched,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_timecode(secs: u8) -> Timecode {
        Timecode::from_frames(u64::from(secs) * 24, 24.0)
    }

    fn sine_burst(sr: usize, start: usize, duration_samples: usize, amplitude: f32) -> Vec<f32> {
        let mut buf = vec![0.0f32; sr * 2]; // 2-second silence
        for i in 0..duration_samples {
            let idx = start + i;
            if idx < buf.len() {
                buf[idx] = amplitude * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin();
            }
        }
        buf
    }

    #[test]
    fn test_sync_point_creation() {
        let tc = make_timecode(5);
        let sp = SyncPoint::new("Step 1", tc, 0.1).expect("valid");
        assert_eq!(sp.label, "Step 1");
        assert_eq!(sp.tolerance_secs, 0.1);
    }

    #[test]
    fn test_sync_point_negative_tolerance_error() {
        let tc = make_timecode(5);
        assert!(SyncPoint::new("bad", tc, -0.1).is_err());
    }

    #[test]
    fn test_onset_detector_config_validation() {
        let mut cfg = OnsetDetectorConfig::default_48k();
        assert!(cfg.validate().is_ok());

        cfg.sample_rate = 0;
        assert!(cfg.validate().is_err());

        cfg.sample_rate = 48000;
        cfg.hop_size = 0;
        assert!(cfg.validate().is_err());

        cfg.hop_size = 512;
        cfg.onset_delta = 0.5; // must be > 1.0
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_onset_detector_empty_buffer() {
        let cfg = OnsetDetectorConfig::default_48k();
        let detector = OnsetDetector::new(cfg).expect("valid config");
        assert!(detector.detect(&[]).is_err());
    }

    #[test]
    fn test_onset_detector_detects_burst() {
        let sr = 48000_usize;
        let cfg = OnsetDetectorConfig {
            sample_rate: sr as u32,
            window_size: 512,
            hop_size: 256,
            silence_threshold: 0.001,
            onset_delta: 2.0,
            min_onset_gap_secs: 0.02,
        };
        let detector = OnsetDetector::new(cfg).expect("valid config");
        // Create silence then a burst at ~0.5 s
        let buf = sine_burst(sr, sr / 2, 2048, 0.5);
        let onsets = detector.detect(&buf).expect("detect ok");
        assert!(!onsets.is_empty(), "should detect at least one onset");
        // The first onset should be near 0.5 s
        assert!(
            onsets[0].position_secs > 0.3 && onsets[0].position_secs < 0.7,
            "onset at {}, expected ~0.5",
            onsets[0].position_secs
        );
    }

    #[test]
    fn test_foley_synchronizer_match_onsets() {
        let mut sync = FoleySynchronizer::new(0.0);
        let tc = make_timecode(1); // 1 second
        let sp = SyncPoint::new("Step", tc, 0.2).expect("ok");
        sync.add_sync_point(sp);

        let onsets = vec![OnsetEvent {
            position_secs: 1.05, // 50 ms late
            energy: 0.5,
            matched: false,
        }];

        let matches = sync.match_onsets(&onsets);
        assert_eq!(matches.len(), 1);
        assert!(matches[0].within_tolerance);
        let offset = matches[0].offset_secs.expect("should have offset");
        assert!((offset - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_foley_synchronizer_no_match_outside_tolerance() {
        let mut sync = FoleySynchronizer::new(0.0);
        let tc = make_timecode(1);
        let sp = SyncPoint::new("Step", tc, 0.05).expect("ok");
        sync.add_sync_point(sp);

        let onsets = vec![OnsetEvent {
            position_secs: 1.3, // 300 ms late — outside 50 ms tolerance
            energy: 0.5,
            matched: false,
        }];

        let matches = sync.match_onsets(&onsets);
        assert!(!matches[0].within_tolerance);
    }

    #[test]
    fn test_compute_global_offset_median() {
        let dummy_sp = SyncPoint::new("x", make_timecode(0), 1.0).expect("ok");

        let make_match = |offset: f64| SyncMatch {
            sync_point: dummy_sp.clone(),
            matched_onset: Some(OnsetEvent {
                position_secs: offset,
                energy: 0.1,
                matched: true,
            }),
            offset_secs: Some(offset),
            within_tolerance: true,
        };

        let matches = vec![make_match(0.1), make_match(0.05), make_match(0.2)];
        let global = FoleySynchronizer::compute_global_offset(&matches);
        assert!(global.is_some());
        let v = global.expect("some");
        // Median of [0.05, 0.10, 0.20] = 0.10
        assert!((v - 0.10).abs() < 1e-9);
    }

    #[test]
    fn test_compute_global_offset_none_when_no_matches() {
        let global = FoleySynchronizer::compute_global_offset(&[]);
        assert!(global.is_none());
    }

    #[test]
    fn test_generate_placements_confidence() {
        let dummy_sp = SyncPoint::new("x", make_timecode(0), 0.2).expect("ok");
        let m = SyncMatch {
            sync_point: dummy_sp,
            matched_onset: Some(OnsetEvent {
                position_secs: 0.1,
                energy: 0.5,
                matched: true,
            }),
            offset_secs: Some(0.1), // 50% of tolerance
            within_tolerance: true,
        };
        let sync = FoleySynchronizer::new(0.0);
        let placements = sync.generate_placements(&[m]);
        assert_eq!(placements.len(), 1);
        let conf = placements[0].confidence;
        // offset/tolerance = 0.1/0.2 = 0.5 → confidence = 0.5
        assert!((conf - 0.5).abs() < 0.01, "confidence was {conf}");
        // recommended_offset should negate the positive onset offset
        assert!(placements[0].recommended_offset_secs < 0.0);
    }

    #[test]
    fn test_apply_offset_to_onsets() {
        let onsets = vec![
            OnsetEvent { position_secs: 1.0, energy: 0.5, matched: false },
            OnsetEvent { position_secs: 2.0, energy: 0.3, matched: false },
        ];
        let shifted = apply_offset_to_onsets(&onsets, -0.5);
        assert!((shifted[0].position_secs - 0.5).abs() < 1e-9);
        assert!((shifted[1].position_secs - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_apply_offset_clamps_to_zero() {
        let onsets = vec![OnsetEvent { position_secs: 0.1, energy: 0.5, matched: false }];
        let shifted = apply_offset_to_onsets(&onsets, -1.0);
        assert_eq!(shifted[0].position_secs, 0.0);
    }
}
