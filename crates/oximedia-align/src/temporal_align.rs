//! Temporal stream alignment utilities for `OxiMedia`.
//!
//! Provides [`StreamAligner`] which applies measured offsets to bring multiple
//! streams into a common time-base and reports the resulting alignment quality.

#![allow(dead_code)]

/// A signed time offset to apply to a stream's presentation timestamps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlignmentOffset {
    /// Stream identifier index.
    pub stream_index: usize,
    /// Offset in milliseconds to add to every PTS in this stream.
    pub offset_ms: f64,
    /// Confidence score `[0.0, 1.0]` from the sync detection step.
    pub confidence: f64,
}

impl AlignmentOffset {
    /// Create a new alignment offset.
    #[must_use]
    pub fn new(stream_index: usize, offset_ms: f64, confidence: f64) -> Self {
        Self {
            stream_index,
            offset_ms,
            confidence,
        }
    }

    /// Apply this offset to a raw PTS value (in ms) and return the adjusted PTS.
    #[must_use]
    pub fn apply_to_pts(&self, raw_pts_ms: f64) -> f64 {
        raw_pts_ms + self.offset_ms
    }

    /// Return `true` when the offset magnitude is within `tolerance_ms`.
    #[must_use]
    pub fn is_within_tolerance(&self, tolerance_ms: f64) -> bool {
        self.offset_ms.abs() <= tolerance_ms
    }
}

/// Represents a single stream's temporal synchronisation state.
#[derive(Debug, Clone)]
pub struct TemporalAlignment {
    /// Stream index in the session.
    pub stream_index: usize,
    /// Applied offset in milliseconds.
    pub applied_offset_ms: f64,
    /// Residual drift per second (ms/s); ideally zero.
    pub drift_rate_ms_per_s: f64,
    /// Whether alignment was applied successfully.
    pub aligned: bool,
}

impl TemporalAlignment {
    /// Create a new temporal alignment record.
    #[must_use]
    pub fn new(
        stream_index: usize,
        applied_offset_ms: f64,
        drift_rate_ms_per_s: f64,
        aligned: bool,
    ) -> Self {
        Self {
            stream_index,
            applied_offset_ms,
            drift_rate_ms_per_s,
            aligned,
        }
    }

    /// Return `true` when the stream is aligned and residual drift is negligible.
    ///
    /// "Negligible" is defined as < 0.1 ms/s.
    #[must_use]
    pub fn is_synchronized(&self) -> bool {
        self.aligned && self.drift_rate_ms_per_s.abs() < 0.1
    }

    /// Compute the predicted PTS drift after `duration_s` seconds.
    #[must_use]
    pub fn predicted_drift_ms(&self, duration_s: f64) -> f64 {
        self.drift_rate_ms_per_s * duration_s
    }
}

/// Configuration for the stream aligner.
#[derive(Debug, Clone)]
pub struct StreamAlignerConfig {
    /// Tolerance in ms; offsets beyond this are flagged.
    pub tolerance_ms: f64,
    /// Maximum allowed drift rate in ms/s before marking as unsynced.
    pub max_drift_ms_per_s: f64,
    /// Minimum confidence required to apply an offset.
    pub min_confidence: f64,
}

impl Default for StreamAlignerConfig {
    fn default() -> Self {
        Self {
            tolerance_ms: 10.0,
            max_drift_ms_per_s: 0.5,
            min_confidence: 0.60,
        }
    }
}

/// Aligns multiple media streams to a common time-base.
#[derive(Debug)]
pub struct StreamAligner {
    config: StreamAlignerConfig,
    alignments: Vec<TemporalAlignment>,
}

impl StreamAligner {
    /// Create a new aligner with the given configuration.
    #[must_use]
    pub fn new(config: StreamAlignerConfig) -> Self {
        Self {
            config,
            alignments: Vec::new(),
        }
    }

    /// Create an aligner with default configuration.
    #[must_use]
    pub fn default_aligner() -> Self {
        Self::new(StreamAlignerConfig::default())
    }

    /// Apply a set of offsets, producing [`TemporalAlignment`] records.
    ///
    /// Offsets with insufficient confidence are recorded as unaligned.
    pub fn align_streams(&mut self, offsets: &[AlignmentOffset]) -> &[TemporalAlignment] {
        self.alignments.clear();
        for off in offsets {
            let aligned = off.confidence >= self.config.min_confidence;
            let applied = if aligned { off.offset_ms } else { 0.0 };
            // Estimate drift as zero; real implementations would interpolate.
            let drift = 0.0_f64;
            self.alignments.push(TemporalAlignment::new(
                off.stream_index,
                applied,
                drift,
                aligned,
            ));
        }
        &self.alignments
    }

    /// Return the maximum absolute offset applied across all streams (ms).
    #[must_use]
    pub fn max_offset_ms(&self) -> f64 {
        self.alignments
            .iter()
            .map(|a| a.applied_offset_ms.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Count how many streams are fully synchronized.
    #[must_use]
    pub fn synchronized_count(&self) -> usize {
        self.alignments
            .iter()
            .filter(|a| a.is_synchronized())
            .count()
    }

    /// Return `true` if every stream achieved synchronisation.
    #[must_use]
    pub fn all_synchronized(&self) -> bool {
        !self.alignments.is_empty()
            && self
                .alignments
                .iter()
                .all(TemporalAlignment::is_synchronized)
    }

    /// Look up the alignment record for a given stream index.
    #[must_use]
    pub fn get_alignment(&self, stream_index: usize) -> Option<&TemporalAlignment> {
        self.alignments
            .iter()
            .find(|a| a.stream_index == stream_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── AlignmentOffset ──────────────────────────────────────────────────────

    #[test]
    fn test_apply_to_pts_positive() {
        let off = AlignmentOffset::new(0, 50.0, 0.9);
        assert!((off.apply_to_pts(1000.0) - 1050.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apply_to_pts_negative() {
        let off = AlignmentOffset::new(0, -30.0, 0.9);
        assert!((off.apply_to_pts(1000.0) - 970.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apply_to_pts_zero() {
        let off = AlignmentOffset::new(0, 0.0, 1.0);
        assert!((off.apply_to_pts(500.0) - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_within_tolerance_true() {
        let off = AlignmentOffset::new(0, 5.0, 0.9);
        assert!(off.is_within_tolerance(10.0));
    }

    #[test]
    fn test_within_tolerance_false() {
        let off = AlignmentOffset::new(0, 20.0, 0.9);
        assert!(!off.is_within_tolerance(10.0));
    }

    // ── TemporalAlignment ────────────────────────────────────────────────────

    #[test]
    fn test_is_synchronized_true() {
        let ta = TemporalAlignment::new(0, 5.0, 0.01, true);
        assert!(ta.is_synchronized());
    }

    #[test]
    fn test_is_synchronized_not_aligned() {
        let ta = TemporalAlignment::new(0, 5.0, 0.01, false);
        assert!(!ta.is_synchronized());
    }

    #[test]
    fn test_is_synchronized_high_drift() {
        let ta = TemporalAlignment::new(0, 5.0, 0.5, true);
        assert!(!ta.is_synchronized());
    }

    #[test]
    fn test_predicted_drift() {
        let ta = TemporalAlignment::new(0, 0.0, 0.05, true);
        assert!((ta.predicted_drift_ms(100.0) - 5.0).abs() < f64::EPSILON);
    }

    // ── StreamAligner ────────────────────────────────────────────────────────

    #[test]
    fn test_aligner_empty() {
        let mut aligner = StreamAligner::default_aligner();
        aligner.align_streams(&[]);
        assert!((aligner.max_offset_ms()).abs() < f64::EPSILON);
        assert!(!aligner.all_synchronized());
    }

    #[test]
    fn test_aligner_applies_confident_offset() {
        let mut aligner = StreamAligner::default_aligner();
        let offsets = [AlignmentOffset::new(0, 8.0, 0.95)];
        aligner.align_streams(&offsets);
        let al = aligner.get_alignment(0).expect("al should be valid");
        assert!((al.applied_offset_ms - 8.0).abs() < f64::EPSILON);
        assert!(al.aligned);
    }

    #[test]
    fn test_aligner_skips_low_confidence() {
        let mut aligner = StreamAligner::default_aligner();
        let offsets = [AlignmentOffset::new(0, 15.0, 0.2)];
        aligner.align_streams(&offsets);
        let al = aligner.get_alignment(0).expect("al should be valid");
        assert!((al.applied_offset_ms).abs() < f64::EPSILON);
        assert!(!al.aligned);
    }

    #[test]
    fn test_aligner_max_offset() {
        let mut aligner = StreamAligner::default_aligner();
        let offsets = [
            AlignmentOffset::new(0, 3.0, 0.9),
            AlignmentOffset::new(1, 7.0, 0.9),
            AlignmentOffset::new(2, 1.5, 0.9),
        ];
        aligner.align_streams(&offsets);
        assert!((aligner.max_offset_ms() - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_aligner_all_synchronized() {
        let mut aligner = StreamAligner::default_aligner();
        let offsets = [
            AlignmentOffset::new(0, 2.0, 0.95),
            AlignmentOffset::new(1, 4.0, 0.85),
        ];
        aligner.align_streams(&offsets);
        assert!(aligner.all_synchronized());
    }

    #[test]
    fn test_aligner_synchronized_count() {
        let mut aligner = StreamAligner::default_aligner();
        let offsets = [
            AlignmentOffset::new(0, 2.0, 0.95),  // aligned
            AlignmentOffset::new(1, 50.0, 0.10), // not aligned (low conf)
        ];
        aligner.align_streams(&offsets);
        assert_eq!(aligner.synchronized_count(), 1);
    }

    #[test]
    fn test_aligner_get_alignment_missing() {
        let aligner = StreamAligner::default_aligner();
        assert!(aligner.get_alignment(99).is_none());
    }
}
