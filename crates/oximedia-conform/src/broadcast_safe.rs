//! Broadcast-safe luma/chroma range analysis for media conform in `OxiMedia`.
//!
//! Checks that pixel values in a frame conform to broadcast legal ranges
//! (e.g., EBU R103 / SMPTE 75% / 100% colour bars).  Values are normalised
//! to `[0, 255]` (8-bit); adapt by scaling for 10-bit/16-bit sources.

#![allow(dead_code)]

/// Legal luma range for broadcast delivery.
///
/// Values are in 8-bit units; broadcast legal is typically `[16, 235]` (ITU-R BT.601/709).
#[derive(Debug, Clone, Copy)]
pub struct LumaRange {
    /// Minimum legal luma value (inclusive).
    pub min: u8,
    /// Maximum legal luma value (inclusive).
    pub max: u8,
}

impl LumaRange {
    /// Standard broadcast-safe luma range: 16–235.
    #[must_use]
    pub fn broadcast() -> Self {
        Self { min: 16, max: 235 }
    }

    /// Full-swing range: 0–255.
    #[must_use]
    pub fn full_swing() -> Self {
        Self { min: 0, max: 255 }
    }

    /// Return `true` when `value` is within this luma range.
    #[must_use]
    pub fn is_broadcast_safe(&self, value: u8) -> bool {
        value >= self.min && value <= self.max
    }

    /// Return the number of out-of-range samples in a luma slice.
    #[must_use]
    pub fn count_violations(&self, samples: &[u8]) -> usize {
        samples
            .iter()
            .filter(|&&v| !self.is_broadcast_safe(v))
            .count()
    }
}

/// Legal chroma range for broadcast delivery.
///
/// Broadcast legal chroma is typically `[16, 240]`.
#[derive(Debug, Clone, Copy)]
pub struct ChromaRange {
    /// Minimum legal chroma value (inclusive).
    pub min: u8,
    /// Maximum legal chroma value (inclusive).
    pub max: u8,
}

impl ChromaRange {
    /// Standard broadcast-safe chroma range: 16–240.
    #[must_use]
    pub fn broadcast() -> Self {
        Self { min: 16, max: 240 }
    }

    /// Return `true` when `value` is within this chroma range.
    #[must_use]
    pub fn is_legal(&self, value: u8) -> bool {
        value >= self.min && value <= self.max
    }

    /// Return the number of out-of-range samples in a chroma slice.
    #[must_use]
    pub fn count_violations(&self, samples: &[u8]) -> usize {
        samples.iter().filter(|&&v| !self.is_legal(v)).count()
    }
}

/// Summary of a single frame's broadcast-safe analysis.
#[derive(Debug, Clone)]
pub struct FrameAnalysisResult {
    /// Frame PTS in milliseconds.
    pub pts_ms: i64,
    /// Number of luma violations.
    pub luma_violations: usize,
    /// Number of chroma violations.
    pub chroma_violations: usize,
    /// Total pixels checked.
    pub total_pixels: usize,
}

impl FrameAnalysisResult {
    /// Total number of violations across luma and chroma.
    #[must_use]
    pub fn total_violations(&self) -> usize {
        self.luma_violations + self.chroma_violations
    }

    /// Return `true` when the frame has no violations.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.total_violations() == 0
    }

    /// Violation rate as a fraction of total pixels `[0.0, 1.0]`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn violation_rate(&self) -> f64 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        self.total_violations() as f64 / self.total_pixels as f64
    }
}

/// Analyzes frames for broadcast-safe compliance.
#[derive(Debug)]
pub struct BroadcastSafeAnalyzer {
    luma_range: LumaRange,
    chroma_range: ChromaRange,
    results: Vec<FrameAnalysisResult>,
}

impl BroadcastSafeAnalyzer {
    /// Create a new analyzer with the given ranges.
    #[must_use]
    pub fn new(luma_range: LumaRange, chroma_range: ChromaRange) -> Self {
        Self {
            luma_range,
            chroma_range,
            results: Vec::new(),
        }
    }

    /// Create an analyzer using standard broadcast-safe ranges.
    #[must_use]
    pub fn broadcast_standard() -> Self {
        Self::new(LumaRange::broadcast(), ChromaRange::broadcast())
    }

    /// Analyze a frame represented as flat luma and chroma sample slices.
    ///
    /// `pts_ms` is the frame presentation timestamp.
    pub fn analyze_frame(
        &mut self,
        pts_ms: i64,
        luma: &[u8],
        chroma: &[u8],
    ) -> &FrameAnalysisResult {
        let lv = self.luma_range.count_violations(luma);
        let cv = self.chroma_range.count_violations(chroma);
        let total = luma.len().max(chroma.len());
        self.results.push(FrameAnalysisResult {
            pts_ms,
            luma_violations: lv,
            chroma_violations: cv,
            total_pixels: total,
        });
        // Safety: we just pushed an element so `last()` is always `Some`.
        self.results
            .last()
            .expect("results is non-empty after push")
    }

    /// Total violation count across all analyzed frames.
    #[must_use]
    pub fn violation_count(&self) -> usize {
        self.results
            .iter()
            .map(FrameAnalysisResult::total_violations)
            .sum()
    }

    /// Number of frames with at least one violation.
    #[must_use]
    pub fn dirty_frame_count(&self) -> usize {
        self.results.iter().filter(|r| !r.is_clean()).count()
    }

    /// Number of frames analyzed.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.results.len()
    }

    /// Return `true` when every analyzed frame is clean.
    #[must_use]
    pub fn all_broadcast_safe(&self) -> bool {
        !self.results.is_empty() && self.results.iter().all(FrameAnalysisResult::is_clean)
    }

    /// Borrow the per-frame results.
    #[must_use]
    pub fn results(&self) -> &[FrameAnalysisResult] {
        &self.results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── LumaRange ────────────────────────────────────────────────────────────

    #[test]
    fn test_luma_broadcast_safe_in_range() {
        let r = LumaRange::broadcast();
        assert!(r.is_broadcast_safe(16));
        assert!(r.is_broadcast_safe(235));
        assert!(r.is_broadcast_safe(128));
    }

    #[test]
    fn test_luma_broadcast_safe_below() {
        let r = LumaRange::broadcast();
        assert!(!r.is_broadcast_safe(0));
        assert!(!r.is_broadcast_safe(15));
    }

    #[test]
    fn test_luma_broadcast_safe_above() {
        let r = LumaRange::broadcast();
        assert!(!r.is_broadcast_safe(236));
        assert!(!r.is_broadcast_safe(255));
    }

    #[test]
    fn test_luma_count_violations() {
        let r = LumaRange::broadcast();
        let samples = vec![0u8, 16, 128, 235, 255];
        assert_eq!(r.count_violations(&samples), 2); // 0 and 255
    }

    #[test]
    fn test_luma_full_swing_no_violations() {
        let r = LumaRange::full_swing();
        let samples: Vec<u8> = (0..=255).collect();
        assert_eq!(r.count_violations(&samples), 0);
    }

    // ── ChromaRange ──────────────────────────────────────────────────────────

    #[test]
    fn test_chroma_legal_in_range() {
        let r = ChromaRange::broadcast();
        assert!(r.is_legal(16));
        assert!(r.is_legal(128));
        assert!(r.is_legal(240));
    }

    #[test]
    fn test_chroma_illegal_below() {
        let r = ChromaRange::broadcast();
        assert!(!r.is_legal(0));
        assert!(!r.is_legal(15));
    }

    #[test]
    fn test_chroma_illegal_above() {
        let r = ChromaRange::broadcast();
        assert!(!r.is_legal(241));
        assert!(!r.is_legal(255));
    }

    #[test]
    fn test_chroma_count_violations() {
        let r = ChromaRange::broadcast();
        let samples = vec![0u8, 16, 240, 255];
        assert_eq!(r.count_violations(&samples), 2);
    }

    // ── FrameAnalysisResult ──────────────────────────────────────────────────

    #[test]
    fn test_frame_result_clean() {
        let r = FrameAnalysisResult {
            pts_ms: 0,
            luma_violations: 0,
            chroma_violations: 0,
            total_pixels: 1920 * 1080,
        };
        assert!(r.is_clean());
        assert_eq!(r.total_violations(), 0);
    }

    #[test]
    fn test_frame_result_violation_rate() {
        let r = FrameAnalysisResult {
            pts_ms: 0,
            luma_violations: 100,
            chroma_violations: 0,
            total_pixels: 1000,
        };
        assert!((r.violation_rate() - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_result_zero_pixels() {
        let r = FrameAnalysisResult {
            pts_ms: 0,
            luma_violations: 0,
            chroma_violations: 0,
            total_pixels: 0,
        };
        assert!((r.violation_rate()).abs() < f64::EPSILON);
    }

    // ── BroadcastSafeAnalyzer ────────────────────────────────────────────────

    #[test]
    fn test_analyzer_clean_frame() {
        let mut a = BroadcastSafeAnalyzer::broadcast_standard();
        let luma: Vec<u8> = (16..=235).collect();
        let chroma: Vec<u8> = (16..=240).collect();
        a.analyze_frame(0, &luma, &chroma);
        assert_eq!(a.violation_count(), 0);
        assert!(a.all_broadcast_safe());
    }

    #[test]
    fn test_analyzer_dirty_frame() {
        let mut a = BroadcastSafeAnalyzer::broadcast_standard();
        let luma = vec![0u8, 255, 128]; // 0 and 255 are violations
        let chroma = vec![128u8];
        a.analyze_frame(1000, &luma, &chroma);
        assert_eq!(a.violation_count(), 2);
        assert!(!a.all_broadcast_safe());
    }

    #[test]
    fn test_analyzer_frame_count() {
        let mut a = BroadcastSafeAnalyzer::broadcast_standard();
        a.analyze_frame(0, &[128], &[128]);
        a.analyze_frame(33, &[128], &[128]);
        assert_eq!(a.frame_count(), 2);
    }

    #[test]
    fn test_analyzer_dirty_frame_count() {
        let mut a = BroadcastSafeAnalyzer::broadcast_standard();
        a.analyze_frame(0, &[128], &[128]); // clean
        a.analyze_frame(33, &[0, 255], &[128]); // dirty
        assert_eq!(a.dirty_frame_count(), 1);
    }
}
