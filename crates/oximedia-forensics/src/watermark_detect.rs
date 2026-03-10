#![allow(dead_code)]
//! Watermark detection for forensic analysis.
//!
//! Provides types and a detector for identifying visible, invisible, and
//! forensic watermarks within frame data.

/// Category of watermark that was (or may be) present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatermarkType {
    /// Visually apparent branding or overlay.
    Visible,
    /// Steganographically embedded, imperceptible to the eye.
    Invisible,
    /// Forensic / fingerprinting watermark embedded for provenance tracking.
    Forensic,
}

impl WatermarkType {
    /// Returns `true` for types that can be detected by simple visual inspection.
    pub fn is_detectable(&self) -> bool {
        matches!(self, WatermarkType::Visible)
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            WatermarkType::Visible => "Visible",
            WatermarkType::Invisible => "Invisible",
            WatermarkType::Forensic => "Forensic",
        }
    }
}

/// A candidate watermark signal returned by the detector.
#[derive(Debug, Clone)]
pub struct WatermarkSignal {
    /// What kind of watermark this signal represents.
    pub watermark_type: WatermarkType,
    /// Detection confidence in the range [0.0, 1.0].
    pub confidence: f64,
    /// Optional spatial location hint (x, y) within the frame.
    pub location: Option<(u32, u32)>,
    /// Additional detail string produced by the detector.
    pub detail: String,
}

impl WatermarkSignal {
    /// Construct a new signal.
    pub fn new(watermark_type: WatermarkType, confidence: f64) -> Self {
        Self {
            watermark_type,
            confidence: confidence.clamp(0.0, 1.0),
            location: None,
            detail: String::new(),
        }
    }

    /// Attach spatial location information.
    pub fn with_location(mut self, x: u32, y: u32) -> Self {
        self.location = Some((x, y));
        self
    }

    /// Attach a detail message.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }

    /// Returns `true` when confidence meets or exceeds the given threshold.
    pub fn confidence_ok(&self, threshold: f64) -> bool {
        self.confidence >= threshold.clamp(0.0, 1.0)
    }
}

/// Simulated per-frame scan result.
#[derive(Debug)]
pub struct FrameScanResult {
    /// Frame index within the video.
    pub frame_index: u64,
    /// All signals found in this frame.
    pub signals: Vec<WatermarkSignal>,
}

impl FrameScanResult {
    /// Whether any signal meets the confidence threshold.
    pub fn any_confident(&self, threshold: f64) -> bool {
        self.signals.iter().any(|s| s.confidence_ok(threshold))
    }

    /// Signals above the threshold.
    pub fn confident_signals(&self, threshold: f64) -> Vec<&WatermarkSignal> {
        self.signals
            .iter()
            .filter(|s| s.confidence_ok(threshold))
            .collect()
    }
}

/// Watermark detector that operates on raw luminance byte slices.
#[derive(Debug)]
pub struct WatermarkDetector {
    /// Minimum confidence to record a signal.
    pub confidence_threshold: f64,
    /// Accumulated scan results.
    scan_results: Vec<FrameScanResult>,
}

impl WatermarkDetector {
    /// Create a detector with the given confidence threshold.
    pub fn new(confidence_threshold: f64) -> Self {
        Self {
            confidence_threshold: confidence_threshold.clamp(0.0, 1.0),
            scan_results: Vec::new(),
        }
    }

    /// Scan a single frame represented as raw luma bytes.
    ///
    /// This implementation uses a heuristic:
    /// - High mean brightness → possible visible watermark.
    /// - Low variance → possible invisible watermark.
    /// - Bit-pattern modulation → possible forensic watermark.
    #[allow(clippy::cast_precision_loss)]
    pub fn scan_frame(&mut self, frame_index: u64, luma: &[u8]) -> &FrameScanResult {
        let mut signals = Vec::new();

        if !luma.is_empty() {
            let mean = luma.iter().map(|&v| v as f64).sum::<f64>() / luma.len() as f64;
            let variance = luma
                .iter()
                .map(|&v| {
                    let d = v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / luma.len() as f64;

            // Heuristic: bright region with low variance → visible overlay.
            if mean > 200.0 && variance < 50.0 {
                let conf = ((mean - 200.0) / 55.0).clamp(0.0, 1.0);
                let sig = WatermarkSignal::new(WatermarkType::Visible, conf)
                    .with_detail(format!("mean={mean:.1}, var={variance:.1}"));
                if sig.confidence_ok(self.confidence_threshold) {
                    signals.push(sig);
                }
            }

            // Heuristic: low mean with very low variance → invisible embed.
            if mean < 30.0 && variance < 5.0 {
                let conf = (1.0 - mean / 30.0).clamp(0.0, 1.0);
                let sig = WatermarkSignal::new(WatermarkType::Invisible, conf)
                    .with_detail(format!("mean={mean:.1}"));
                if sig.confidence_ok(self.confidence_threshold) {
                    signals.push(sig);
                }
            }

            // Heuristic: LSB modulation → forensic mark.
            let lsb_ones = luma.iter().filter(|&&v| v & 1 == 1).count();
            let lsb_ratio = lsb_ones as f64 / luma.len() as f64;
            if (lsb_ratio - 0.5).abs() < 0.05 {
                // Near-perfect 50/50 → likely structured embed.
                let conf = 1.0 - (lsb_ratio - 0.5).abs() / 0.05;
                let sig = WatermarkSignal::new(WatermarkType::Forensic, conf)
                    .with_detail(format!("lsb_ratio={lsb_ratio:.3}"));
                if sig.confidence_ok(self.confidence_threshold) {
                    signals.push(sig);
                }
            }
        }

        self.scan_results.push(FrameScanResult {
            frame_index,
            signals,
        });
        // Safety: we just pushed an element, so last() is always Some.
        self.scan_results
            .last()
            .expect("scan_results is non-empty after push")
    }

    /// Total number of detected signals across all scanned frames.
    pub fn detected_count(&self) -> usize {
        self.scan_results.iter().map(|r| r.signals.len()).sum()
    }

    /// Number of frames scanned so far.
    pub fn frames_scanned(&self) -> usize {
        self.scan_results.len()
    }

    /// Reset all accumulated results.
    pub fn reset(&mut self) {
        self.scan_results.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watermark_type_visible_is_detectable() {
        assert!(WatermarkType::Visible.is_detectable());
    }

    #[test]
    fn test_watermark_type_invisible_not_detectable() {
        assert!(!WatermarkType::Invisible.is_detectable());
    }

    #[test]
    fn test_watermark_type_forensic_not_detectable() {
        assert!(!WatermarkType::Forensic.is_detectable());
    }

    #[test]
    fn test_watermark_type_labels() {
        assert_eq!(WatermarkType::Visible.label(), "Visible");
        assert_eq!(WatermarkType::Invisible.label(), "Invisible");
        assert_eq!(WatermarkType::Forensic.label(), "Forensic");
    }

    #[test]
    fn test_signal_confidence_clamps() {
        let s = WatermarkSignal::new(WatermarkType::Visible, 2.0);
        assert_eq!(s.confidence, 1.0);
        let s2 = WatermarkSignal::new(WatermarkType::Visible, -1.0);
        assert_eq!(s2.confidence, 0.0);
    }

    #[test]
    fn test_signal_confidence_ok_true() {
        let s = WatermarkSignal::new(WatermarkType::Forensic, 0.8);
        assert!(s.confidence_ok(0.7));
    }

    #[test]
    fn test_signal_confidence_ok_false() {
        let s = WatermarkSignal::new(WatermarkType::Forensic, 0.3);
        assert!(!s.confidence_ok(0.5));
    }

    #[test]
    fn test_signal_with_location() {
        let s = WatermarkSignal::new(WatermarkType::Visible, 0.9).with_location(10, 20);
        assert_eq!(s.location, Some((10, 20)));
    }

    #[test]
    fn test_signal_with_detail() {
        let s = WatermarkSignal::new(WatermarkType::Invisible, 0.6).with_detail("embedded payload");
        assert!(s.detail.contains("embedded payload"));
    }

    #[test]
    fn test_detector_empty_frame_no_signals() {
        let mut det = WatermarkDetector::new(0.0);
        let result = det.scan_frame(0, &[]);
        assert!(result.signals.is_empty());
    }

    #[test]
    fn test_detector_detected_count_zero_initially() {
        let det = WatermarkDetector::new(0.5);
        assert_eq!(det.detected_count(), 0);
        assert_eq!(det.frames_scanned(), 0);
    }

    #[test]
    fn test_detector_frames_scanned_increases() {
        let mut det = WatermarkDetector::new(0.5);
        det.scan_frame(0, &[128u8; 100]);
        det.scan_frame(1, &[128u8; 100]);
        assert_eq!(det.frames_scanned(), 2);
    }

    #[test]
    fn test_detector_reset_clears_results() {
        let mut det = WatermarkDetector::new(0.0);
        det.scan_frame(0, &[255u8; 50]);
        assert!(det.frames_scanned() > 0);
        det.reset();
        assert_eq!(det.frames_scanned(), 0);
        assert_eq!(det.detected_count(), 0);
    }

    #[test]
    fn test_frame_scan_result_any_confident() {
        let result = FrameScanResult {
            frame_index: 0,
            signals: vec![WatermarkSignal::new(WatermarkType::Visible, 0.9)],
        };
        assert!(result.any_confident(0.8));
        assert!(!result.any_confident(0.95));
    }

    #[test]
    fn test_frame_scan_result_confident_signals() {
        let result = FrameScanResult {
            frame_index: 0,
            signals: vec![
                WatermarkSignal::new(WatermarkType::Visible, 0.9),
                WatermarkSignal::new(WatermarkType::Invisible, 0.3),
            ],
        };
        let conf = result.confident_signals(0.5);
        assert_eq!(conf.len(), 1);
    }
}
