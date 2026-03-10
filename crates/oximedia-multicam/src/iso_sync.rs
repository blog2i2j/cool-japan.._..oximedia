//! ISO recording synchronisation for multi-camera production.
//!
//! An ISO (isolated) recording is the full, uncut feed from a single camera.
//! This module provides data structures for managing ISO tracks, correlating
//! their timecodes, and measuring / correcting clock drift between cameras.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ── IsoTrackId ────────────────────────────────────────────────────────────────

/// Opaque identifier for an ISO track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IsoTrackId(u32);

impl IsoTrackId {
    /// Wrap a raw numeric ID.
    #[must_use]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Unwrap the raw numeric value.
    #[must_use]
    pub fn value(self) -> u32 {
        self.0
    }
}

// ── IsoTrack ──────────────────────────────────────────────────────────────────

/// Metadata for a single ISO recording track.
#[derive(Debug, Clone)]
pub struct IsoTrack {
    /// Track identifier.
    pub id: IsoTrackId,
    /// Human-readable label (e.g. "CAM A ISO").
    pub label: String,
    /// Sample rate of the recording's audio, in Hz.
    pub sample_rate: u32,
    /// Frame rate of the recording's video (frames per second).
    pub frame_rate: f32,
    /// Starting SMPTE timecode string (HH:MM:SS:FF).
    pub start_timecode: String,
    /// Total duration in video frames.
    pub duration_frames: u64,
}

impl IsoTrack {
    /// Create a new `IsoTrack`.
    pub fn new(
        id: u32,
        label: impl Into<String>,
        sample_rate: u32,
        frame_rate: f32,
        start_timecode: impl Into<String>,
        duration_frames: u64,
    ) -> Self {
        Self {
            id: IsoTrackId::new(id),
            label: label.into(),
            sample_rate,
            frame_rate,
            start_timecode: start_timecode.into(),
            duration_frames,
        }
    }

    /// Duration of the track in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        if self.frame_rate <= 0.0 {
            return 0.0;
        }
        self.duration_frames as f64 / f64::from(self.frame_rate)
    }
}

// ── TimecodeCorrelation ───────────────────────────────────────────────────────

/// Correlates two ISO tracks by aligning their timecodes.
#[derive(Debug, Clone)]
pub struct TimecodeCorrelation {
    /// Reference track (considered ground-truth).
    pub reference_id: IsoTrackId,
    /// Secondary track being aligned to the reference.
    pub secondary_id: IsoTrackId,
    /// Offset in frames that must be applied to the secondary track to align it
    /// with the reference.  Positive means secondary starts *after* reference.
    pub frame_offset: i64,
    /// Confidence score for this correlation (0.0 – 1.0).
    pub confidence: f32,
}

impl TimecodeCorrelation {
    /// Create a new `TimecodeCorrelation`.
    #[must_use]
    pub fn new(reference_id: u32, secondary_id: u32, frame_offset: i64, confidence: f32) -> Self {
        Self {
            reference_id: IsoTrackId::new(reference_id),
            secondary_id: IsoTrackId::new(secondary_id),
            frame_offset,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Return `true` if the tracks appear to be in sync (small offset and high confidence).
    #[must_use]
    pub fn is_synced(&self, max_offset: i64, min_confidence: f32) -> bool {
        self.frame_offset.abs() <= max_offset && self.confidence >= min_confidence
    }

    /// Convert the frame offset to milliseconds for a given frame rate.
    #[must_use]
    pub fn offset_ms(&self, frame_rate: f32) -> f64 {
        if frame_rate <= 0.0 {
            return 0.0;
        }
        (self.frame_offset as f64 / f64::from(frame_rate)) * 1000.0
    }
}

// ── DriftSample ───────────────────────────────────────────────────────────────

/// A single clock-drift measurement between a camera and the reference clock.
#[derive(Debug, Clone, Copy)]
pub struct DriftSample {
    /// Reference time, in seconds from the start of the session.
    pub reference_time_s: f64,
    /// Measured drift at this point, in milliseconds.
    /// Positive means the camera clock is running *ahead* of reference.
    pub drift_ms: f64,
}

// ── DriftCorrectionModel ──────────────────────────────────────────────────────

/// Linear drift correction model built from a series of drift samples.
///
/// Fits a least-squares line `drift = slope * t + intercept` through the
/// samples, then uses that line to predict and compensate drift at any time.
#[derive(Debug, Clone)]
pub struct DriftCorrectionModel {
    samples: Vec<DriftSample>,
    slope: f64,
    intercept: f64,
}

impl DriftCorrectionModel {
    /// Create a model from a set of drift samples.
    ///
    /// Requires at least two distinct time points.  Returns `None` if the
    /// samples are insufficient or degenerate.
    #[must_use]
    pub fn fit(samples: Vec<DriftSample>) -> Option<Self> {
        if samples.len() < 2 {
            return None;
        }
        let n = samples.len() as f64;
        let sum_t: f64 = samples.iter().map(|s| s.reference_time_s).sum();
        let sum_d: f64 = samples.iter().map(|s| s.drift_ms).sum();
        let sum_tt: f64 = samples
            .iter()
            .map(|s| s.reference_time_s * s.reference_time_s)
            .sum();
        let sum_td: f64 = samples
            .iter()
            .map(|s| s.reference_time_s * s.drift_ms)
            .sum();
        let denom = n * sum_tt - sum_t * sum_t;
        if denom.abs() < f64::EPSILON {
            return None;
        }
        let slope = (n * sum_td - sum_t * sum_d) / denom;
        let intercept = (sum_d - slope * sum_t) / n;
        Some(Self {
            samples,
            slope,
            intercept,
        })
    }

    /// Predict the drift (ms) at `reference_time_s`.
    #[must_use]
    pub fn predict_drift_ms(&self, reference_time_s: f64) -> f64 {
        self.slope * reference_time_s + self.intercept
    }

    /// Correct a timestamp by removing the predicted drift.
    ///
    /// Returns the corrected time in seconds.
    #[must_use]
    pub fn correct_timestamp_s(&self, camera_time_s: f64) -> f64 {
        let drift_s = self.predict_drift_ms(camera_time_s) / 1000.0;
        camera_time_s - drift_s
    }

    /// Drift rate (ms per second), i.e., the fitted slope.
    #[must_use]
    pub fn drift_rate_ms_per_s(&self) -> f64 {
        self.slope
    }

    /// Number of samples used to build this model.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

// ── IsoSyncSession ────────────────────────────────────────────────────────────

/// Session-level container linking ISO tracks, correlations, and drift models.
#[derive(Debug, Default)]
pub struct IsoSyncSession {
    tracks: HashMap<IsoTrackId, IsoTrack>,
    correlations: Vec<TimecodeCorrelation>,
    drift_models: HashMap<IsoTrackId, DriftCorrectionModel>,
}

impl IsoSyncSession {
    /// Create an empty sync session.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a track to the session.
    pub fn add_track(&mut self, track: IsoTrack) {
        self.tracks.insert(track.id, track);
    }

    /// Retrieve a track by ID.
    #[must_use]
    pub fn track(&self, id: IsoTrackId) -> Option<&IsoTrack> {
        self.tracks.get(&id)
    }

    /// Record a timecode correlation.
    pub fn add_correlation(&mut self, corr: TimecodeCorrelation) {
        self.correlations.push(corr);
    }

    /// Return all correlations.
    #[must_use]
    pub fn correlations(&self) -> &[TimecodeCorrelation] {
        &self.correlations
    }

    /// Store a drift correction model for a track.
    pub fn set_drift_model(&mut self, track_id: IsoTrackId, model: DriftCorrectionModel) {
        self.drift_models.insert(track_id, model);
    }

    /// Retrieve the drift model for a track.
    #[must_use]
    pub fn drift_model(&self, track_id: IsoTrackId) -> Option<&DriftCorrectionModel> {
        self.drift_models.get(&track_id)
    }

    /// Number of tracks in the session.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_track(id: u32) -> IsoTrack {
        IsoTrack::new(id, format!("CAM {id}"), 48000, 25.0, "00:00:00:00", 2500)
    }

    #[test]
    fn test_iso_track_duration_seconds() {
        let t = make_track(1);
        // 2500 frames @ 25 fps = 100 s
        assert!((t.duration_seconds() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_iso_track_id_value() {
        let id = IsoTrackId::new(7);
        assert_eq!(id.value(), 7);
    }

    #[test]
    fn test_timecode_correlation_is_synced() {
        let corr = TimecodeCorrelation::new(0, 1, 1, 0.95);
        assert!(corr.is_synced(2, 0.9));
        assert!(!corr.is_synced(0, 0.9)); // offset 1 > max 0
    }

    #[test]
    fn test_timecode_correlation_offset_ms() {
        let corr = TimecodeCorrelation::new(0, 1, 25, 1.0);
        // 25 frames @ 25 fps = 1 second = 1000 ms
        assert!((corr.offset_ms(25.0) - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_timecode_correlation_confidence_clamped() {
        let corr = TimecodeCorrelation::new(0, 1, 0, 1.5);
        assert_eq!(corr.confidence, 1.0);
    }

    #[test]
    fn test_drift_correction_model_fit_linear() {
        let samples = vec![
            DriftSample {
                reference_time_s: 0.0,
                drift_ms: 0.0,
            },
            DriftSample {
                reference_time_s: 10.0,
                drift_ms: 5.0,
            },
            DriftSample {
                reference_time_s: 20.0,
                drift_ms: 10.0,
            },
        ];
        let model =
            DriftCorrectionModel::fit(samples).expect("multicam test operation should succeed");
        // slope should be 0.5 ms/s
        assert!((model.drift_rate_ms_per_s() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_drift_correction_model_predict() {
        let samples = vec![
            DriftSample {
                reference_time_s: 0.0,
                drift_ms: 0.0,
            },
            DriftSample {
                reference_time_s: 100.0,
                drift_ms: 10.0,
            },
        ];
        let model =
            DriftCorrectionModel::fit(samples).expect("multicam test operation should succeed");
        let predicted = model.predict_drift_ms(50.0);
        assert!((predicted - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_drift_correction_model_too_few_samples() {
        let samples = vec![DriftSample {
            reference_time_s: 0.0,
            drift_ms: 1.0,
        }];
        assert!(DriftCorrectionModel::fit(samples).is_none());
    }

    #[test]
    fn test_drift_correction_model_sample_count() {
        let samples = vec![
            DriftSample {
                reference_time_s: 0.0,
                drift_ms: 0.0,
            },
            DriftSample {
                reference_time_s: 1.0,
                drift_ms: 0.1,
            },
            DriftSample {
                reference_time_s: 2.0,
                drift_ms: 0.2,
            },
        ];
        let model =
            DriftCorrectionModel::fit(samples).expect("multicam test operation should succeed");
        assert_eq!(model.sample_count(), 3);
    }

    #[test]
    fn test_drift_correction_correct_timestamp() {
        // 1 ms/s drift: at t=10 s, drift is 10 ms = 0.01 s
        let samples = vec![
            DriftSample {
                reference_time_s: 0.0,
                drift_ms: 0.0,
            },
            DriftSample {
                reference_time_s: 10.0,
                drift_ms: 10.0,
            },
        ];
        let model =
            DriftCorrectionModel::fit(samples).expect("multicam test operation should succeed");
        let corrected = model.correct_timestamp_s(10.0);
        // 10.0 - 0.010 = 9.990
        assert!((corrected - 9.99).abs() < 1e-6);
    }

    #[test]
    fn test_iso_sync_session_track_operations() {
        let mut session = IsoSyncSession::new();
        session.add_track(make_track(1));
        session.add_track(make_track(2));
        assert_eq!(session.track_count(), 2);
        assert!(session.track(IsoTrackId::new(1)).is_some());
        assert!(session.track(IsoTrackId::new(99)).is_none());
    }

    #[test]
    fn test_iso_sync_session_correlation() {
        let mut session = IsoSyncSession::new();
        session.add_correlation(TimecodeCorrelation::new(0, 1, 2, 0.8));
        assert_eq!(session.correlations().len(), 1);
    }

    #[test]
    fn test_iso_sync_session_drift_model() {
        let mut session = IsoSyncSession::new();
        session.add_track(make_track(1));
        let samples = vec![
            DriftSample {
                reference_time_s: 0.0,
                drift_ms: 0.0,
            },
            DriftSample {
                reference_time_s: 5.0,
                drift_ms: 1.0,
            },
        ];
        let model =
            DriftCorrectionModel::fit(samples).expect("multicam test operation should succeed");
        let tid = IsoTrackId::new(1);
        session.set_drift_model(tid, model);
        assert!(session.drift_model(tid).is_some());
    }
}
