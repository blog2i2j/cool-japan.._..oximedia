//! Multipass encoding quality comparison.
//!
//! This module provides tools to verify that multipass encoding (two-pass,
//! lookahead-based) produces measurably better quality than single-pass
//! encoding at the same average bitrate. The comparison uses:
//!
//! - Per-frame quality scores (QP-derived or PSNR-measured)
//! - Bitrate distribution analysis (variance, min/max ratio)
//! - Quality consistency metrics (standard deviation of per-frame quality)
//!
//! # Verification Strategy
//!
//! For a fair comparison between single-pass and multipass:
//! 1. Encode the same source with both methods at the same target bitrate
//! 2. Record per-frame (size, QP) tuples
//! 3. Compare:
//!    - Average quality should be equal or better for multipass
//!    - Quality variance should be lower for multipass
//!    - Bitrate distribution should be smoother for multipass
//!
//! # Usage
//!
//! ```rust
//! use oximedia_codec::multipass_quality::{
//!     PassRecorder, MultipassComparison, FrameMetric,
//! };
//!
//! let mut single = PassRecorder::new("single-pass");
//! let mut multi  = PassRecorder::new("two-pass");
//!
//! // Record frames from each encoding session
//! single.record(FrameMetric { size_bytes: 5000, qp: 28, is_keyframe: false });
//! multi.record(FrameMetric { size_bytes: 4800, qp: 27, is_keyframe: false });
//! // ... more frames ...
//!
//! let cmp = MultipassComparison::compare(&single, &multi);
//! // multipass should have equal or better average QP
//! ```

/// Per-frame metric recorded during encoding.
#[derive(Debug, Clone, Copy)]
pub struct FrameMetric {
    /// Encoded frame size in bytes.
    pub size_bytes: u32,
    /// Quantisation parameter used for this frame.
    pub qp: u8,
    /// Whether this frame was encoded as a keyframe.
    pub is_keyframe: bool,
}

/// Records per-frame metrics for one encoding pass.
#[derive(Debug, Clone)]
pub struct PassRecorder {
    /// Label for this pass (e.g. "single-pass", "two-pass").
    label: String,
    /// Recorded frame metrics in order.
    frames: Vec<FrameMetric>,
}

impl PassRecorder {
    /// Create a new pass recorder with the given label.
    #[must_use]
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            frames: Vec::new(),
        }
    }

    /// Record a single frame's metrics.
    pub fn record(&mut self, metric: FrameMetric) {
        self.frames.push(metric);
    }

    /// Get the number of recorded frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Get the label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Compute the average QP across all frames.
    #[must_use]
    pub fn average_qp(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.frames.iter().map(|f| u64::from(f.qp)).sum();
        sum as f64 / self.frames.len() as f64
    }

    /// Compute the QP standard deviation.
    #[must_use]
    pub fn qp_std_dev(&self) -> f64 {
        if self.frames.len() < 2 {
            return 0.0;
        }
        let mean = self.average_qp();
        let variance: f64 = self
            .frames
            .iter()
            .map(|f| {
                let d = f64::from(f.qp) - mean;
                d * d
            })
            .sum::<f64>()
            / (self.frames.len() - 1) as f64;
        variance.sqrt()
    }

    /// Compute the total size in bytes.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.frames.iter().map(|f| u64::from(f.size_bytes)).sum()
    }

    /// Compute the average frame size in bytes.
    #[must_use]
    pub fn average_frame_size(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }
        self.total_bytes() as f64 / self.frames.len() as f64
    }

    /// Compute the frame size standard deviation.
    #[must_use]
    pub fn size_std_dev(&self) -> f64 {
        if self.frames.len() < 2 {
            return 0.0;
        }
        let mean = self.average_frame_size();
        let variance: f64 = self
            .frames
            .iter()
            .map(|f| {
                let d = f64::from(f.size_bytes) - mean;
                d * d
            })
            .sum::<f64>()
            / (self.frames.len() - 1) as f64;
        variance.sqrt()
    }

    /// Compute min/max frame size ratio.
    #[must_use]
    pub fn size_min_max_ratio(&self) -> f64 {
        if self.frames.is_empty() {
            return 1.0;
        }
        let min = self.frames.iter().map(|f| f.size_bytes).min().unwrap_or(1);
        let max = self.frames.iter().map(|f| f.size_bytes).max().unwrap_or(1);
        if max == 0 {
            return 1.0;
        }
        f64::from(min) / f64::from(max)
    }

    /// Return a reference to all recorded frames.
    #[must_use]
    pub fn frames(&self) -> &[FrameMetric] {
        &self.frames
    }

    /// Clear all recorded data.
    pub fn reset(&mut self) {
        self.frames.clear();
    }
}

/// Result of comparing two encoding passes.
#[derive(Debug, Clone)]
pub struct MultipassComparison {
    /// Label of the reference (typically single-pass) pass.
    pub reference_label: String,
    /// Label of the candidate (typically multipass) pass.
    pub candidate_label: String,
    /// Average QP of reference pass.
    pub ref_avg_qp: f64,
    /// Average QP of candidate pass.
    pub cand_avg_qp: f64,
    /// QP std dev of reference pass.
    pub ref_qp_std_dev: f64,
    /// QP std dev of candidate pass.
    pub cand_qp_std_dev: f64,
    /// Total bytes of reference pass.
    pub ref_total_bytes: u64,
    /// Total bytes of candidate pass.
    pub cand_total_bytes: u64,
    /// Frame size std dev of reference.
    pub ref_size_std_dev: f64,
    /// Frame size std dev of candidate.
    pub cand_size_std_dev: f64,
    /// Whether the candidate has equal or better average QP (lower is better).
    pub candidate_qp_equal_or_better: bool,
    /// Whether the candidate has lower or equal QP variance.
    pub candidate_qp_more_consistent: bool,
    /// Whether the candidate has a smoother bitrate distribution.
    pub candidate_smoother_bitrate: bool,
}

impl MultipassComparison {
    /// Compare two passes: `reference` is the baseline (e.g. single-pass),
    /// `candidate` is the multipass result.
    #[must_use]
    pub fn compare(reference: &PassRecorder, candidate: &PassRecorder) -> Self {
        let ref_avg_qp = reference.average_qp();
        let cand_avg_qp = candidate.average_qp();
        let ref_qp_std = reference.qp_std_dev();
        let cand_qp_std = candidate.qp_std_dev();
        let ref_size_std = reference.size_std_dev();
        let cand_size_std = candidate.size_std_dev();

        Self {
            reference_label: reference.label().to_string(),
            candidate_label: candidate.label().to_string(),
            ref_avg_qp,
            cand_avg_qp,
            ref_qp_std_dev: ref_qp_std,
            cand_qp_std_dev: cand_qp_std,
            ref_total_bytes: reference.total_bytes(),
            cand_total_bytes: candidate.total_bytes(),
            ref_size_std_dev: ref_size_std,
            cand_size_std_dev: cand_size_std,
            candidate_qp_equal_or_better: cand_avg_qp <= ref_avg_qp + 0.5,
            candidate_qp_more_consistent: cand_qp_std <= ref_qp_std + 0.5,
            candidate_smoother_bitrate: cand_size_std <= ref_size_std * 1.1,
        }
    }

    /// Summary string for logging / assertion messages.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} vs {}: avg_qp {:.1} vs {:.1}, qp_std {:.2} vs {:.2}, \
             size_std {:.0} vs {:.0}, bytes {} vs {}",
            self.reference_label,
            self.candidate_label,
            self.ref_avg_qp,
            self.cand_avg_qp,
            self.ref_qp_std_dev,
            self.cand_qp_std_dev,
            self.ref_size_std_dev,
            self.cand_size_std_dev,
            self.ref_total_bytes,
            self.cand_total_bytes,
        )
    }
}

// =============================================================================
// Tests — Multipass encoding quality comparison
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a pass recorder with uniform frames.
    fn uniform_pass(label: &str, n: usize, size: u32, qp: u8) -> PassRecorder {
        let mut rec = PassRecorder::new(label);
        for i in 0..n {
            rec.record(FrameMetric {
                size_bytes: size,
                qp,
                is_keyframe: i == 0,
            });
        }
        rec
    }

    /// Helper: create a pass with varying complexity.
    fn varying_pass(label: &str, n: usize, base_size: u32, qp_range: (u8, u8)) -> PassRecorder {
        let mut rec = PassRecorder::new(label);
        for i in 0..n {
            let t = i as f64 / n as f64;
            // Simulate varying complexity with a sine wave
            let variation = (t * std::f64::consts::PI * 4.0).sin();
            let qp = (qp_range.0 as f64
                + (qp_range.1 as f64 - qp_range.0 as f64) * (variation + 1.0) / 2.0)
                as u8;
            let size = (base_size as f64 * (1.0 + variation * 0.3)) as u32;
            rec.record(FrameMetric {
                size_bytes: size,
                qp,
                is_keyframe: i % 30 == 0,
            });
        }
        rec
    }

    #[test]
    fn test_pass_recorder_basic() {
        let mut rec = PassRecorder::new("test");
        rec.record(FrameMetric {
            size_bytes: 1000,
            qp: 28,
            is_keyframe: true,
        });
        rec.record(FrameMetric {
            size_bytes: 500,
            qp: 30,
            is_keyframe: false,
        });
        assert_eq!(rec.frame_count(), 2);
        assert_eq!(rec.label(), "test");
        assert_eq!(rec.total_bytes(), 1500);
    }

    #[test]
    fn test_average_qp() {
        let rec = uniform_pass("test", 10, 1000, 28);
        assert!((rec.average_qp() - 28.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_qp_std_dev_uniform() {
        let rec = uniform_pass("test", 10, 1000, 28);
        assert!(
            rec.qp_std_dev() < f64::EPSILON,
            "uniform QP should have zero std dev"
        );
    }

    #[test]
    fn test_qp_std_dev_varying() {
        let mut rec = PassRecorder::new("test");
        rec.record(FrameMetric {
            size_bytes: 1000,
            qp: 20,
            is_keyframe: false,
        });
        rec.record(FrameMetric {
            size_bytes: 1000,
            qp: 40,
            is_keyframe: false,
        });
        let std_dev = rec.qp_std_dev();
        assert!(
            std_dev > 10.0,
            "QP 20 vs 40 should have large std dev, got {std_dev}"
        );
    }

    #[test]
    fn test_size_std_dev_uniform() {
        let rec = uniform_pass("test", 20, 5000, 28);
        assert!(
            rec.size_std_dev() < f64::EPSILON,
            "uniform size should have zero std dev"
        );
    }

    #[test]
    fn test_size_min_max_ratio() {
        let mut rec = PassRecorder::new("test");
        rec.record(FrameMetric {
            size_bytes: 1000,
            qp: 28,
            is_keyframe: false,
        });
        rec.record(FrameMetric {
            size_bytes: 2000,
            qp: 28,
            is_keyframe: false,
        });
        let ratio = rec.size_min_max_ratio();
        assert!(
            (ratio - 0.5).abs() < f64::EPSILON,
            "min/max ratio should be 0.5, got {ratio}"
        );
    }

    #[test]
    fn test_comparison_identical_passes() {
        let single = uniform_pass("single", 90, 5000, 28);
        let multi = uniform_pass("multi", 90, 5000, 28);
        let cmp = MultipassComparison::compare(&single, &multi);

        assert!(cmp.candidate_qp_equal_or_better);
        assert!(cmp.candidate_qp_more_consistent);
        assert!(cmp.candidate_smoother_bitrate);
    }

    #[test]
    fn test_comparison_multipass_better_qp() {
        let single = uniform_pass("single", 90, 5000, 30);
        let multi = uniform_pass("multi", 90, 5000, 26); // lower QP = better
        let cmp = MultipassComparison::compare(&single, &multi);

        assert!(
            cmp.candidate_qp_equal_or_better,
            "multipass with lower QP should be detected as better"
        );
    }

    #[test]
    fn test_comparison_multipass_more_consistent() {
        let single = varying_pass("single", 120, 5000, (20, 40));
        let multi = varying_pass("multi", 120, 5000, (26, 30)); // tighter QP range
        let cmp = MultipassComparison::compare(&single, &multi);

        assert!(
            cmp.cand_qp_std_dev < cmp.ref_qp_std_dev,
            "multipass should have lower QP variance: {} vs {}",
            cmp.cand_qp_std_dev,
            cmp.ref_qp_std_dev
        );
    }

    #[test]
    fn test_comparison_smoother_bitrate() {
        // Single-pass: large size variance
        let mut single = PassRecorder::new("single");
        for i in 0..60 {
            let size = if i % 10 == 0 { 15000 } else { 3000 };
            single.record(FrameMetric {
                size_bytes: size,
                qp: 28,
                is_keyframe: i % 10 == 0,
            });
        }

        // Multipass: smooth size distribution
        let multi = uniform_pass("multi", 60, 5000, 28);
        let cmp = MultipassComparison::compare(&single, &multi);

        assert!(
            cmp.candidate_smoother_bitrate,
            "uniform multipass should have smoother bitrate: {} vs {}",
            cmp.cand_size_std_dev, cmp.ref_size_std_dev
        );
    }

    #[test]
    fn test_comparison_summary_format() {
        let single = uniform_pass("single", 10, 5000, 28);
        let multi = uniform_pass("multi", 10, 5000, 26);
        let cmp = MultipassComparison::compare(&single, &multi);
        let summary = cmp.summary();

        assert!(summary.contains("single"));
        assert!(summary.contains("multi"));
        assert!(summary.contains("avg_qp"));
    }

    #[test]
    fn test_pass_recorder_reset() {
        let mut rec = uniform_pass("test", 10, 5000, 28);
        assert_eq!(rec.frame_count(), 10);
        rec.reset();
        assert_eq!(rec.frame_count(), 0);
        assert!(rec.total_bytes() == 0);
    }

    #[test]
    fn test_empty_recorder_defaults() {
        let rec = PassRecorder::new("empty");
        assert!(rec.average_qp() < f64::EPSILON);
        assert!(rec.qp_std_dev() < f64::EPSILON);
        assert!(rec.average_frame_size() < f64::EPSILON);
        assert!((rec.size_min_max_ratio() - 1.0).abs() < f64::EPSILON);
    }
}
