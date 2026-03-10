//! Anomaly detection for media monitoring.
//!
//! Detects audio/video anomalies such as loudness drops, clipping,
//! video freeze, blackout, sync drift, and bitrate anomalies.

/// The type of anomaly detected.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum AnomalyType {
    /// Audio loudness dropped significantly.
    AudioLoudnessDrop,
    /// Audio signal is clipping.
    AudioClipping,
    /// Video frame is frozen (no motion).
    VideoFreeze,
    /// Video frame is entirely black.
    VideoBlackout,
    /// Audio/video sync has drifted beyond threshold.
    SyncDrift,
    /// Bitrate has spiked above threshold.
    BitrateSpike,
    /// Bitrate has dropped below threshold.
    BitrateDropout,
}

/// A detected anomaly with metadata.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Anomaly {
    /// The type of anomaly.
    pub anomaly_type: AnomalyType,
    /// Timestamp when anomaly was detected (milliseconds since epoch).
    pub detected_at: u64,
    /// Severity score in the range [0.0, 1.0].
    pub severity: f64,
    /// Human-readable description.
    pub description: String,
    /// Duration of the anomaly in milliseconds, if known.
    pub duration_ms: Option<u64>,
}

impl Anomaly {
    /// Create a new anomaly.
    #[must_use]
    pub fn new(
        anomaly_type: AnomalyType,
        detected_at: u64,
        severity: f64,
        description: String,
    ) -> Self {
        Self {
            anomaly_type,
            detected_at,
            severity: severity.clamp(0.0, 1.0),
            description,
            duration_ms: None,
        }
    }

    /// Create an anomaly with a known duration.
    #[must_use]
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }
}

/// Thresholds used by the anomaly detector.
#[derive(Debug, Clone)]
pub struct AnomalyThresholds {
    /// Loudness drop in dB that triggers an alert.
    pub loudness_drop_db: f64,
    /// Clip level as a normalised amplitude [0.0, 1.0].
    pub clip_level: f64,
    /// Minimum freeze duration in milliseconds before triggering.
    pub freeze_duration_ms: u64,
    /// Sync drift threshold in milliseconds.
    pub sync_drift_ms: f64,
    /// Percentage change in bitrate that triggers a spike/dropout.
    pub bitrate_change_pct: f64,
}

impl Default for AnomalyThresholds {
    fn default() -> Self {
        Self {
            loudness_drop_db: 20.0,
            clip_level: 0.98,
            freeze_duration_ms: 2000,
            sync_drift_ms: 100.0,
            bitrate_change_pct: 50.0,
        }
    }
}

/// Anomaly detector that analyses audio/video streams for problems.
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    /// History window in milliseconds used for trend analysis.
    pub history_window_ms: u64,
    /// Detection thresholds.
    pub thresholds: AnomalyThresholds,
}

impl AnomalyDetector {
    /// Create a new detector with the given thresholds.
    #[must_use]
    pub fn new(thresholds: AnomalyThresholds) -> Self {
        Self {
            history_window_ms: 5000,
            thresholds,
        }
    }

    /// Create a detector with default thresholds.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(AnomalyThresholds::default())
    }

    /// Set the history window duration.
    #[must_use]
    pub fn with_history_window(mut self, ms: u64) -> Self {
        self.history_window_ms = ms;
        self
    }

    /// Compute the RMS (Root Mean Square) amplitude of a sample buffer.
    fn rms(samples: &[f64]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = samples.iter().map(|s| s * s).sum();
        (sum_sq / samples.len() as f64).sqrt()
    }

    /// Convert linear amplitude to decibels.
    fn to_db(linear: f64) -> f64 {
        if linear <= 0.0 {
            return f64::NEG_INFINITY;
        }
        20.0 * linear.log10()
    }

    /// Check audio samples for loudness drops and clipping anomalies.
    ///
    /// `timestamp_ms` is the wall-clock timestamp of the first sample.
    #[must_use]
    pub fn check_audio(&self, samples: &[f64], timestamp_ms: u64) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();

        if samples.is_empty() {
            return anomalies;
        }

        // Split into two halves to detect loudness drop mid-block.
        let mid = samples.len() / 2;
        let first_half = &samples[..mid.max(1)];
        let second_half = &samples[mid..];

        let rms_first = Self::rms(first_half);
        let rms_second = Self::rms(second_half);

        let db_first = Self::to_db(rms_first);
        let db_second = Self::to_db(rms_second);

        // Loudness drop detection.
        if db_first.is_finite() && db_second.is_finite() {
            let drop = db_first - db_second;
            if drop >= self.thresholds.loudness_drop_db {
                let severity = (drop / (self.thresholds.loudness_drop_db * 3.0)).min(1.0);
                anomalies.push(Anomaly::new(
                    AnomalyType::AudioLoudnessDrop,
                    timestamp_ms,
                    severity,
                    format!("Audio loudness dropped by {drop:.1} dB"),
                ));
            }
        }

        // Clipping detection – count samples exceeding the clip level.
        let clip_count = samples
            .iter()
            .filter(|&&s| s.abs() >= self.thresholds.clip_level)
            .count();

        if clip_count > 0 {
            let clip_ratio = clip_count as f64 / samples.len() as f64;
            if clip_ratio > 0.001 {
                let severity = (clip_ratio * 10.0).min(1.0);
                anomalies.push(Anomaly::new(
                    AnomalyType::AudioClipping,
                    timestamp_ms,
                    severity,
                    format!(
                        "Audio clipping detected: {clip_count} samples ({:.2}%)",
                        clip_ratio * 100.0
                    ),
                ));
            }
        }

        anomalies
    }

    /// Check a sequence of frame timestamps for video freeze.
    ///
    /// `frames` is a slice of frame presentation timestamps in milliseconds.
    /// If consecutive frames have the same timestamp for longer than the
    /// threshold, a freeze anomaly is returned.
    #[must_use]
    pub fn check_video_freeze(frames: &[u64], freeze_duration_ms: u64) -> Option<Anomaly> {
        if frames.len() < 2 {
            return None;
        }

        let mut freeze_start: Option<u64> = None;
        let mut freeze_ts: Option<u64> = None;

        for window in frames.windows(2) {
            let (prev, curr) = (window[0], window[1]);
            if curr == prev {
                if freeze_start.is_none() {
                    freeze_start = Some(prev);
                    freeze_ts = Some(prev);
                }
            } else {
                if let Some(start) = freeze_start {
                    let duration = curr.saturating_sub(start);
                    if duration >= freeze_duration_ms {
                        return Some(
                            Anomaly::new(
                                AnomalyType::VideoFreeze,
                                freeze_ts.unwrap_or(start),
                                (duration as f64 / (freeze_duration_ms as f64 * 5.0)).min(1.0),
                                format!("Video freeze detected for {duration} ms"),
                            )
                            .with_duration(duration),
                        );
                    }
                }
                freeze_start = None;
                freeze_ts = None;
            }
        }

        // Check freeze that extends to end of slice.
        if let Some(start) = freeze_start {
            let last = *frames.last().unwrap_or(&start);
            let duration = last.saturating_sub(start);
            if duration >= freeze_duration_ms {
                return Some(
                    Anomaly::new(
                        AnomalyType::VideoFreeze,
                        freeze_ts.unwrap_or(start),
                        (duration as f64 / (freeze_duration_ms as f64 * 5.0)).min(1.0),
                        format!("Video freeze detected for {duration} ms"),
                    )
                    .with_duration(duration),
                );
            }
        }

        None
    }

    /// Check whether the current bitrate represents a spike relative to history.
    ///
    /// Returns `Some(Anomaly)` when a spike or dropout is detected.
    #[must_use]
    pub fn check_bitrate_spike(
        history: &[u64],
        current: u64,
        threshold_pct: f64,
    ) -> Option<Anomaly> {
        if history.is_empty() {
            return None;
        }

        let avg: f64 = history.iter().map(|&v| v as f64).sum::<f64>() / history.len() as f64;
        if avg <= 0.0 {
            return None;
        }

        let change_pct = ((current as f64 - avg) / avg) * 100.0;

        if change_pct >= threshold_pct {
            let severity = ((change_pct - threshold_pct) / threshold_pct).min(1.0);
            return Some(Anomaly::new(
                AnomalyType::BitrateSpike,
                0,
                severity,
                format!("Bitrate spike: {change_pct:.1}% above average ({avg:.0} bps)"),
            ));
        }

        if change_pct <= -threshold_pct {
            let severity = ((-change_pct - threshold_pct) / threshold_pct).min(1.0);
            return Some(Anomaly::new(
                AnomalyType::BitrateDropout,
                0,
                severity,
                format!(
                    "Bitrate dropout: {:.1}% below average ({avg:.0} bps)",
                    -change_pct
                ),
            ));
        }

        None
    }

    /// Check audio/video synchronisation drift.
    ///
    /// Returns `Some(Anomaly)` when the drift exceeds the configured threshold.
    #[must_use]
    pub fn check_sync_drift(
        &self,
        audio_pts_ms: f64,
        video_pts_ms: f64,
        timestamp_ms: u64,
    ) -> Option<Anomaly> {
        let drift = (audio_pts_ms - video_pts_ms).abs();
        if drift >= self.thresholds.sync_drift_ms {
            let severity = (drift / (self.thresholds.sync_drift_ms * 5.0)).min(1.0);
            Some(Anomaly::new(
                AnomalyType::SyncDrift,
                timestamp_ms,
                severity,
                format!("A/V sync drift of {drift:.1} ms detected"),
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detector() -> AnomalyDetector {
        AnomalyDetector::with_defaults()
    }

    // --- AnomalyType tests ---

    #[test]
    fn test_anomaly_type_equality() {
        assert_eq!(AnomalyType::AudioClipping, AnomalyType::AudioClipping);
        assert_ne!(AnomalyType::VideoFreeze, AnomalyType::BitrateSpike);
    }

    // --- Anomaly construction tests ---

    #[test]
    fn test_anomaly_severity_clamped() {
        let a = Anomaly::new(AnomalyType::AudioClipping, 0, 5.0, "test".into());
        assert!((a.severity - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_anomaly_severity_negative_clamped() {
        let a = Anomaly::new(AnomalyType::AudioClipping, 0, -1.0, "test".into());
        assert!((a.severity - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_anomaly_with_duration() {
        let a =
            Anomaly::new(AnomalyType::VideoFreeze, 1000, 0.5, "freeze".into()).with_duration(3000);
        assert_eq!(a.duration_ms, Some(3000));
    }

    #[test]
    fn test_anomaly_no_duration_by_default() {
        let a = Anomaly::new(AnomalyType::BitrateSpike, 0, 0.8, "spike".into());
        assert!(a.duration_ms.is_none());
    }

    // --- Audio check tests ---

    #[test]
    fn test_check_audio_empty() {
        let d = detector();
        assert!(d.check_audio(&[], 0).is_empty());
    }

    #[test]
    fn test_check_audio_clipping_detected() {
        let d = AnomalyDetector::new(AnomalyThresholds {
            clip_level: 0.9,
            ..AnomalyThresholds::default()
        });
        // Many clipping samples.
        let samples: Vec<f64> = (0..1000).map(|_| 0.95_f64).collect();
        let anomalies = d.check_audio(&samples, 1000);
        assert!(
            anomalies
                .iter()
                .any(|a| a.anomaly_type == AnomalyType::AudioClipping),
            "Expected clipping anomaly"
        );
    }

    #[test]
    fn test_check_audio_no_clipping_below_threshold() {
        let d = detector();
        let samples: Vec<f64> = (0..100).map(|_| 0.5_f64).collect();
        let anomalies = d.check_audio(&samples, 0);
        assert!(
            anomalies
                .iter()
                .all(|a| a.anomaly_type != AnomalyType::AudioClipping),
            "No clipping expected for 0.5 amplitude"
        );
    }

    #[test]
    fn test_check_audio_loudness_drop() {
        let d = AnomalyDetector::new(AnomalyThresholds {
            loudness_drop_db: 6.0,
            ..AnomalyThresholds::default()
        });
        // First half loud, second half silent.
        let mut samples: Vec<f64> = vec![0.9; 500];
        samples.extend(vec![0.001; 500]);
        let anomalies = d.check_audio(&samples, 0);
        assert!(
            anomalies
                .iter()
                .any(|a| a.anomaly_type == AnomalyType::AudioLoudnessDrop),
            "Expected loudness drop anomaly"
        );
    }

    // --- Video freeze tests ---

    #[test]
    fn test_check_video_freeze_no_freeze() {
        let frames: Vec<u64> = (0..10).map(|i| i * 40).collect(); // 25 fps, 40 ms apart
        assert!(AnomalyDetector::check_video_freeze(&frames, 2000).is_none());
    }

    #[test]
    fn test_check_video_freeze_detected() {
        // Frames frozen at timestamp 1000 for 3 seconds.
        let mut frames: Vec<u64> = (0..10).map(|i| i * 40).collect();
        let freeze_start = *frames.last().expect("should have last element");
        for _ in 0..100 {
            frames.push(freeze_start);
        }
        frames.push(freeze_start + 5000);
        let anomaly = AnomalyDetector::check_video_freeze(&frames, 2000);
        assert!(anomaly.is_some(), "Expected freeze anomaly");
        assert_eq!(
            anomaly.expect("anomaly should be valid").anomaly_type,
            AnomalyType::VideoFreeze
        );
    }

    #[test]
    fn test_check_video_freeze_too_short() {
        // A short freeze that doesn't exceed the threshold.
        let frames = vec![0u64, 100, 100, 200];
        assert!(AnomalyDetector::check_video_freeze(&frames, 2000).is_none());
    }

    // --- Bitrate spike tests ---

    #[test]
    fn test_check_bitrate_spike_detected() {
        let history = vec![1_000_000u64; 10];
        let current = 2_500_000u64; // 150% spike
        let result = AnomalyDetector::check_bitrate_spike(&history, current, 50.0);
        assert!(result.is_some());
        assert_eq!(
            result.expect("result should be valid").anomaly_type,
            AnomalyType::BitrateSpike
        );
    }

    #[test]
    fn test_check_bitrate_dropout_detected() {
        let history = vec![1_000_000u64; 10];
        let current = 100_000u64; // 90% drop
        let result = AnomalyDetector::check_bitrate_spike(&history, current, 50.0);
        assert!(result.is_some());
        assert_eq!(
            result.expect("result should be valid").anomaly_type,
            AnomalyType::BitrateDropout
        );
    }

    #[test]
    fn test_check_bitrate_no_anomaly() {
        let history = vec![1_000_000u64; 10];
        let current = 1_100_000u64; // 10% change — within threshold
        let result = AnomalyDetector::check_bitrate_spike(&history, current, 50.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_check_bitrate_empty_history() {
        let result = AnomalyDetector::check_bitrate_spike(&[], 1_000_000, 50.0);
        assert!(result.is_none());
    }

    // --- Sync drift tests ---

    #[test]
    fn test_sync_drift_detected() {
        let d = AnomalyDetector::new(AnomalyThresholds {
            sync_drift_ms: 50.0,
            ..AnomalyThresholds::default()
        });
        let anomaly = d.check_sync_drift(1000.0, 1200.0, 5000);
        assert!(anomaly.is_some());
        assert_eq!(
            anomaly.expect("anomaly should be valid").anomaly_type,
            AnomalyType::SyncDrift
        );
    }

    #[test]
    fn test_sync_drift_within_threshold() {
        let d = detector();
        let anomaly = d.check_sync_drift(1000.0, 1010.0, 5000);
        assert!(anomaly.is_none());
    }
}
