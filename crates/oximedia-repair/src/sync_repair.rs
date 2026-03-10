//! Audio/video synchronization analysis and repair.
//!
//! This module provides tools for detecting A/V sync drift, planning repairs,
//! and validating synchronization quality.

/// Describes a detected synchronization error between audio and video.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct SyncError {
    /// Video frame index where the error was detected.
    pub video_frame: u64,
    /// Audio sample index corresponding to the video frame.
    pub audio_sample: u64,
    /// Offset between expected and actual audio position in milliseconds.
    /// Positive means audio is ahead; negative means video is ahead.
    pub offset_ms: f64,
    /// Rate of drift in milliseconds per second.
    pub drift_rate: f64,
}

impl SyncError {
    /// Create a new sync error.
    #[must_use]
    pub const fn new(video_frame: u64, audio_sample: u64, offset_ms: f64, drift_rate: f64) -> Self {
        Self {
            video_frame,
            audio_sample,
            offset_ms,
            drift_rate,
        }
    }

    /// Returns `true` if the audio is ahead of the video.
    #[must_use]
    pub fn audio_is_ahead(&self) -> bool {
        self.offset_ms > 0.0
    }
}

/// Method used to repair A/V synchronization.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyncRepairMethod {
    /// Shift audio forward or backward by a fixed amount.
    AudioShift,
    /// Shift video forward or backward by a fixed amount.
    VideoShift,
    /// Resample audio to match video timing.
    Resample,
    /// Drop video frames to reduce offset.
    DropFrames,
    /// Duplicate video frames to increase offset.
    DuplicateFrames,
}

/// A planned repair action for an A/V sync problem.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct SyncRepairAction {
    /// Type of repair to apply.
    pub action_type: SyncRepairMethod,
    /// Frame index at which to apply the repair.
    pub at_frame: u64,
    /// Magnitude of the repair in milliseconds (positive = forward shift).
    pub magnitude_ms: f64,
}

impl SyncRepairAction {
    /// Create a new sync repair action.
    #[must_use]
    pub const fn new(action_type: SyncRepairMethod, at_frame: u64, magnitude_ms: f64) -> Self {
        Self {
            action_type,
            at_frame,
            magnitude_ms,
        }
    }
}

/// Analyzes timestamp pairs to detect A/V synchronization drift.
pub struct SyncAnalyzer;

impl SyncAnalyzer {
    /// Detect synchronization errors using linear regression on timestamp pairs.
    ///
    /// `video_timestamps_ms` and `audio_timestamps_ms` must have the same length.
    /// Each pair `(video_timestamps_ms[i], audio_timestamps_ms[i])` represents
    /// the expected and actual audio timestamp for a given video frame.
    ///
    /// Linear regression is performed to find the best-fit line; residuals
    /// (deviation from the fit) are reported as sync errors.
    ///
    /// Returns an empty vector if the input is empty or has fewer than 2 elements.
    #[must_use]
    pub fn detect_drift(
        video_timestamps_ms: &[f64],
        audio_timestamps_ms: &[f64],
    ) -> Vec<SyncError> {
        let n = video_timestamps_ms.len().min(audio_timestamps_ms.len());
        if n < 2 {
            return Vec::new();
        }

        // Linear regression: fit audio = slope * video + intercept
        let n_f = n as f64;
        let sum_x: f64 = video_timestamps_ms[..n].iter().sum();
        let sum_y: f64 = audio_timestamps_ms[..n].iter().sum();
        let sum_xx: f64 = video_timestamps_ms[..n].iter().map(|x| x * x).sum();
        let sum_xy: f64 = video_timestamps_ms[..n]
            .iter()
            .zip(audio_timestamps_ms[..n].iter())
            .map(|(x, y)| x * y)
            .sum();

        let denom = n_f * sum_xx - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return Vec::new();
        }

        let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n_f;

        // drift_rate is (slope - 1.0) * 1000 ms/s
        let drift_rate_ms_per_s = (slope - 1.0) * 1000.0;

        let mut errors = Vec::new();
        for i in 0..n {
            let video_ts = video_timestamps_ms[i];
            let audio_ts = audio_timestamps_ms[i];
            let predicted = slope * video_ts + intercept;
            let residual = audio_ts - predicted;

            // Only report errors where the residual is non-trivial (> 1ms)
            if residual.abs() > 1.0 {
                errors.push(SyncError::new(
                    i as u64,
                    i as u64 * 48000 / 1000, // approximate audio sample index
                    residual,
                    drift_rate_ms_per_s,
                ));
            }
        }

        errors
    }
}

/// Plans repair actions from detected synchronization errors.
pub struct SyncRepairer;

impl SyncRepairer {
    /// Plan a set of repair actions from a list of sync errors.
    ///
    /// For each error:
    /// - If `|offset_ms| <= max_shift_ms`: choose `AudioShift` if audio is ahead,
    ///   else `VideoShift`.
    /// - If the drift rate indicates growing desync: choose `Resample`.
    /// - Fallback: `DropFrames` or `DuplicateFrames` depending on sign.
    #[must_use]
    pub fn plan_repair(errors: &[SyncError], max_shift_ms: f64) -> Vec<SyncRepairAction> {
        let mut actions = Vec::new();

        for error in errors {
            let abs_offset = error.offset_ms.abs();
            let action_type = if abs_offset <= max_shift_ms {
                if error.audio_is_ahead() {
                    SyncRepairMethod::AudioShift
                } else {
                    SyncRepairMethod::VideoShift
                }
            } else if error.drift_rate.abs() > 5.0 {
                SyncRepairMethod::Resample
            } else if error.offset_ms > 0.0 {
                SyncRepairMethod::DropFrames
            } else {
                SyncRepairMethod::DuplicateFrames
            };

            actions.push(SyncRepairAction::new(
                action_type,
                error.video_frame,
                error.offset_ms,
            ));
        }

        actions
    }
}

/// Validates synchronization quality between video and audio tracks.
pub struct SyncValidator;

impl SyncValidator {
    /// Compute the expected timing drift between video and audio in milliseconds.
    ///
    /// Given `video_count` frames at `fps` frames per second, and `audio_samples`
    /// samples at `sample_rate` samples per second, compute the difference in
    /// expected duration:
    ///
    /// `drift_ms = |video_duration_ms - audio_duration_ms|`
    ///
    /// A value close to 0 indicates good synchronization.
    #[must_use]
    pub fn validate(video_count: u64, audio_samples: u64, fps: f32, sample_rate: u32) -> f64 {
        if fps <= 0.0 || sample_rate == 0 {
            return 0.0;
        }
        let video_duration_ms = video_count as f64 / fps as f64 * 1000.0;
        let audio_duration_ms = audio_samples as f64 / sample_rate as f64 * 1000.0;
        (video_duration_ms - audio_duration_ms).abs()
    }
}

// ---------------------------------------------------------------------------
// New types: SyncOffset, cross_correlate, SyncOp, SyncRepairPlan,
//            create_sync_plan, SyncAnalyzer::detect_offset_by_energy
// ---------------------------------------------------------------------------

/// Represents the current timing position of audio and video streams.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncOffset {
    /// Current audio stream position in milliseconds.
    pub audio_ms: i64,
    /// Current video stream position in milliseconds.
    pub video_ms: i64,
}

impl SyncOffset {
    /// Returns the audio-relative-to-video delay in milliseconds.
    ///
    /// Positive → audio is ahead of video.
    /// Negative → video is ahead of audio.
    #[must_use]
    pub fn av_delay_ms(&self) -> i64 {
        self.audio_ms - self.video_ms
    }

    /// Returns `true` if the absolute delay is within `tolerance_ms`.
    #[must_use]
    pub fn is_in_sync(&self, tolerance_ms: i64) -> bool {
        self.av_delay_ms().abs() <= tolerance_ms
    }
}

/// Compute the sliding dot-product cross-correlation of two equal-length slices.
///
/// Returns a vector of length `2 * a.len() - 1` where index `a.len() - 1`
/// corresponds to zero lag.  Uses O(n²) brute force — suitable for short
/// sequences.
#[must_use]
pub fn cross_correlate(a: &[f32], b: &[f32]) -> Vec<f32> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let n = a.len();
    let m = b.len();
    let out_len = n + m - 1;
    let mut out = vec![0.0f32; out_len];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            out[i + j] += av * bv;
        }
    }
    out
}

/// The corrective action to take when repairing A/V sync.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncOp {
    /// Move the audio track earlier in time (audio is late).
    ShiftAudioEarly,
    /// Move the audio track later in time (audio is early).
    ShiftAudioLate,
    /// Move the video track earlier in time (video is late).
    ShiftVideoEarly,
    /// Move the video track later in time (video is early).
    ShiftVideoLate,
    /// No action needed; streams are already in sync.
    NoAction,
}

/// A plan describing how to repair an A/V sync problem.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SyncRepairPlan {
    /// The operation to perform.
    pub operation: SyncOp,
    /// Number of frames by which to shift the target stream.
    pub offset_frames: i32,
}

impl SyncRepairPlan {
    /// Returns `true` if the planned offset is larger than `threshold` frames.
    #[must_use]
    pub fn is_significant(&self, threshold: i32) -> bool {
        self.offset_frames.abs() > threshold
    }
}

/// Create a `SyncRepairPlan` from a detected offset in milliseconds.
///
/// Positive `detected_offset_ms` means audio is ahead of video;
/// negative means video is ahead of audio.
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub fn create_sync_plan(detected_offset_ms: i64, frame_rate: f64) -> SyncRepairPlan {
    if detected_offset_ms == 0 || frame_rate <= 0.0 {
        return SyncRepairPlan {
            operation: SyncOp::NoAction,
            offset_frames: 0,
        };
    }
    let ms_per_frame = 1000.0 / frame_rate;
    #[allow(clippy::cast_precision_loss)]
    let offset_frames = (detected_offset_ms as f64 / ms_per_frame).round() as i32;

    let operation = if offset_frames > 0 {
        // Audio is ahead → shift audio late (delay it).
        SyncOp::ShiftAudioLate
    } else {
        // Video is ahead → shift audio early (advance it).
        SyncOp::ShiftAudioEarly
    };

    SyncRepairPlan {
        operation,
        offset_frames: offset_frames.abs(),
    }
}

impl SyncAnalyzer {
    /// Detect the A/V sync offset by finding the peak of the cross-correlation
    /// between RMS audio energy and per-frame video motion energy.
    ///
    /// Returns the detected offset in milliseconds (positive = audio ahead).
    /// Returns `0` if either slice is empty.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn detect_offset_by_energy(audio_rms: &[f32], video_motion: &[f32]) -> i64 {
        if audio_rms.is_empty() || video_motion.is_empty() {
            return 0;
        }
        let xcorr = cross_correlate(audio_rms, video_motion);
        // Peak index relative to zero-lag position.
        let zero_lag = audio_rms.len() - 1;
        let peak_idx = xcorr
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(zero_lag);
        // Offset in samples (positive = audio leads video).
        let lag = peak_idx as i64 - zero_lag as i64;
        lag // treat 1 sample == 1 ms for simplicity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_error_audio_is_ahead_positive() {
        let e = SyncError::new(0, 0, 50.0, 0.1);
        assert!(e.audio_is_ahead());
    }

    #[test]
    fn test_sync_error_audio_is_ahead_negative() {
        let e = SyncError::new(0, 0, -50.0, -0.1);
        assert!(!e.audio_is_ahead());
    }

    #[test]
    fn test_sync_analyzer_empty_input() {
        let errors = SyncAnalyzer::detect_drift(&[], &[]);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_sync_analyzer_single_element() {
        let errors = SyncAnalyzer::detect_drift(&[100.0], &[100.0]);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_sync_analyzer_perfect_sync() {
        // Perfect sync: audio timestamps == video timestamps
        let video: Vec<f64> = (0..10).map(|i| i as f64 * 100.0).collect();
        let audio = video.clone();
        let errors = SyncAnalyzer::detect_drift(&video, &audio);
        // Residuals should be near 0 → no significant errors
        assert!(errors.is_empty());
    }

    #[test]
    fn test_sync_analyzer_constant_offset() {
        // Constant +50ms audio offset
        let video: Vec<f64> = (0..10).map(|i| i as f64 * 100.0).collect();
        let audio: Vec<f64> = video.iter().map(|&v| v + 50.0).collect();
        let errors = SyncAnalyzer::detect_drift(&video, &audio);
        // With a constant offset, linear regression should fit perfectly → no residuals
        // (the intercept absorbs the constant offset)
        assert!(errors.is_empty());
    }

    #[test]
    fn test_sync_analyzer_with_drift() {
        // Drifting sync: audio timestamps gradually shift relative to video
        let video: Vec<f64> = (0..10).map(|i| i as f64 * 100.0).collect();
        let audio: Vec<f64> = (0..10).map(|i| i as f64 * 101.0).collect();
        // drift_rate should be non-zero; may or may not produce residual errors
        // depending on whether residuals exceed threshold
        let _errors = SyncAnalyzer::detect_drift(&video, &audio);
        // Just verify it doesn't panic
    }

    #[test]
    fn test_sync_repairer_audio_shift() {
        let errors = vec![SyncError::new(5, 240000, 20.0, 0.1)];
        let actions = SyncRepairer::plan_repair(&errors, 50.0);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].action_type, SyncRepairMethod::AudioShift);
        assert_eq!(actions[0].at_frame, 5);
    }

    #[test]
    fn test_sync_repairer_video_shift() {
        let errors = vec![SyncError::new(3, 144000, -20.0, -0.1)];
        let actions = SyncRepairer::plan_repair(&errors, 50.0);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].action_type, SyncRepairMethod::VideoShift);
    }

    #[test]
    fn test_sync_repairer_resample_high_drift() {
        let errors = vec![SyncError::new(0, 0, 200.0, 10.0)];
        let actions = SyncRepairer::plan_repair(&errors, 50.0);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].action_type, SyncRepairMethod::Resample);
    }

    #[test]
    fn test_sync_repairer_drop_frames() {
        let errors = vec![SyncError::new(0, 0, 200.0, 2.0)];
        let actions = SyncRepairer::plan_repair(&errors, 50.0);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].action_type, SyncRepairMethod::DropFrames);
    }

    #[test]
    fn test_sync_repairer_duplicate_frames() {
        let errors = vec![SyncError::new(0, 0, -200.0, -2.0)];
        let actions = SyncRepairer::plan_repair(&errors, 50.0);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].action_type, SyncRepairMethod::DuplicateFrames);
    }

    #[test]
    fn test_sync_validator_perfect() {
        // 30fps, 48000 Hz, 300 frames = 10s video, 480000 samples = 10s audio
        let drift = SyncValidator::validate(300, 480000, 30.0, 48000);
        assert!(drift < 1.0, "Expected near-zero drift, got {drift}");
    }

    #[test]
    fn test_sync_validator_drift_detected() {
        // 300 frames at 30fps = 10s, 480480 samples at 48000Hz ≈ 10.01s → 10ms drift
        let drift = SyncValidator::validate(300, 480480, 30.0, 48000);
        assert!(drift > 0.0);
    }

    #[test]
    fn test_sync_validator_zero_fps() {
        let drift = SyncValidator::validate(100, 48000, 0.0, 48000);
        assert!((drift - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sync_validator_zero_sample_rate() {
        let drift = SyncValidator::validate(100, 48000, 30.0, 0);
        assert!((drift - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sync_repair_action_new() {
        let action = SyncRepairAction::new(SyncRepairMethod::Resample, 42, -15.5);
        assert_eq!(action.action_type, SyncRepairMethod::Resample);
        assert_eq!(action.at_frame, 42);
        assert!((action.magnitude_ms - (-15.5)).abs() < f64::EPSILON);
    }

    // --- SyncOffset tests ---

    #[test]
    fn test_sync_offset_av_delay_audio_ahead() {
        let o = SyncOffset {
            audio_ms: 100,
            video_ms: 80,
        };
        assert_eq!(o.av_delay_ms(), 20);
    }

    #[test]
    fn test_sync_offset_av_delay_video_ahead() {
        let o = SyncOffset {
            audio_ms: 80,
            video_ms: 100,
        };
        assert_eq!(o.av_delay_ms(), -20);
    }

    #[test]
    fn test_sync_offset_is_in_sync_true() {
        let o = SyncOffset {
            audio_ms: 105,
            video_ms: 100,
        };
        assert!(o.is_in_sync(10));
    }

    #[test]
    fn test_sync_offset_is_in_sync_false() {
        let o = SyncOffset {
            audio_ms: 150,
            video_ms: 100,
        };
        assert!(!o.is_in_sync(10));
    }

    // --- cross_correlate tests ---

    #[test]
    fn test_cross_correlate_empty_returns_empty() {
        assert!(cross_correlate(&[], &[]).is_empty());
    }

    #[test]
    fn test_cross_correlate_unit_impulses() {
        // Correlating [1] with [1] should give [1].
        let result = cross_correlate(&[1.0], &[1.0]);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cross_correlate_peak_at_zero_lag_for_identical() {
        let sig = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let xcorr = cross_correlate(&sig, &sig);
        let zero_lag = sig.len() - 1;
        let peak_idx = xcorr
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("unexpected None/Err");
        assert_eq!(peak_idx, zero_lag);
    }

    // --- SyncOp / SyncRepairPlan tests ---

    #[test]
    fn test_create_sync_plan_no_offset() {
        let plan = create_sync_plan(0, 25.0);
        assert_eq!(plan.operation, SyncOp::NoAction);
        assert_eq!(plan.offset_frames, 0);
    }

    #[test]
    fn test_create_sync_plan_audio_ahead() {
        // 1000ms ahead at 25fps = 25 frames.
        let plan = create_sync_plan(1000, 25.0);
        assert_eq!(plan.operation, SyncOp::ShiftAudioLate);
        assert_eq!(plan.offset_frames, 25);
    }

    #[test]
    fn test_create_sync_plan_video_ahead() {
        // -400ms at 25fps = 10 frames.
        let plan = create_sync_plan(-400, 25.0);
        assert_eq!(plan.operation, SyncOp::ShiftAudioEarly);
        assert_eq!(plan.offset_frames, 10);
    }

    #[test]
    fn test_sync_repair_plan_is_significant_true() {
        let plan = SyncRepairPlan {
            operation: SyncOp::ShiftAudioLate,
            offset_frames: 15,
        };
        assert!(plan.is_significant(10));
    }

    #[test]
    fn test_sync_repair_plan_is_significant_false() {
        let plan = SyncRepairPlan {
            operation: SyncOp::ShiftAudioLate,
            offset_frames: 3,
        };
        assert!(!plan.is_significant(10));
    }

    // --- SyncAnalyzer::detect_offset_by_energy ---

    #[test]
    fn test_detect_offset_by_energy_empty_returns_zero() {
        assert_eq!(SyncAnalyzer::detect_offset_by_energy(&[], &[]), 0);
    }

    #[test]
    fn test_detect_offset_by_energy_identical_returns_zero() {
        let sig = vec![0.0f32, 0.0, 1.0, 0.0, 0.0];
        let offset = SyncAnalyzer::detect_offset_by_energy(&sig, &sig);
        assert_eq!(offset, 0);
    }
}
