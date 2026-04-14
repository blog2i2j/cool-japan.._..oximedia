//! Shot boundary classifier -- categorizes cuts, dissolves, wipes, and fades.
//!
//! Processes a sequence of grayscale frames and classifies transitions between
//! shots using histogram analysis, brightness tracking, and spatial quadrant
//! comparison.
//!
//! # Algorithm
//!
//! 1. Compute a 256-bin grayscale histogram for each frame.
//! 2. Measure histogram difference vs the previous frame:
//!    `diff = sum(|h1[i] - h2[i]|) / (2 * total_pixels)`.
//! 3. **Hard cut**: `diff > cut_threshold`.
//! 4. **Dissolve**: sustained `diff > gradual_threshold` over 3+ consecutive frames.
//! 5. **Fade**: average brightness drops below `fade_brightness_threshold` (near black/white).
//! 6. **Wipe**: frame quadrant analysis reveals spatially localised change.
//! 7. A `min_gap` filter prevents boundaries that are too close together.

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// Type of shot transition detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionType {
    /// Instant scene change (hard cut).
    HardCut,
    /// Gradual blend between two shots.
    Dissolve,
    /// Spatial wipe transition (horizontal/vertical/diagonal).
    Wipe,
    /// Fade to/from black or white.
    Fade,
    /// No transition detected.
    None,
}

/// A detected shot boundary.
#[derive(Debug, Clone)]
pub struct ShotBoundary {
    /// Frame index where the boundary was detected.
    pub frame_index: u32,
    /// Classification of the transition.
    pub transition_type: TransitionType,
    /// Confidence score in \[0.0, 1.0\].
    pub confidence: f32,
    /// Duration of the transition in frames (1 for hard cuts).
    pub duration_frames: u32,
}

/// Configuration for shot boundary detection.
#[derive(Debug, Clone)]
pub struct ShotBoundaryConfig {
    /// Threshold for histogram difference to detect hard cuts \[0.0, 1.0\].
    pub cut_threshold: f32,
    /// Threshold for gradual transition detection.
    pub gradual_threshold: f32,
    /// Minimum frames between detected boundaries.
    pub min_gap: u32,
    /// Fade detection: threshold for average brightness near 0 or 255.
    pub fade_brightness_threshold: f32,
}

impl Default for ShotBoundaryConfig {
    fn default() -> Self {
        Self {
            cut_threshold: 0.4,
            gradual_threshold: 0.15,
            min_gap: 5,
            fade_brightness_threshold: 20.0,
        }
    }
}

/// Streaming shot boundary detector.
///
/// Call [`process_frame`](ShotBoundaryDetector::process_frame) for each
/// consecutive grayscale frame, then retrieve results with
/// [`boundaries`](ShotBoundaryDetector::boundaries) or
/// [`finalize`](ShotBoundaryDetector::finalize).
pub struct ShotBoundaryDetector {
    config: ShotBoundaryConfig,
    prev_histogram: Option<[u32; 256]>,
    /// Recent histogram differences used for dissolve detection.
    history: Vec<f32>,
    boundaries: Vec<ShotBoundary>,
    frame_count: u32,
    last_boundary_frame: u32,
    /// Quadrant histograms from the previous frame (TL, TR, BL, BR).
    prev_quadrant_histograms: Option<[[u32; 256]; 4]>,
}

impl ShotBoundaryDetector {
    /// Create a new detector with the given configuration.
    pub fn new(config: ShotBoundaryConfig) -> Self {
        Self {
            config,
            prev_histogram: Option::None,
            history: Vec::new(),
            boundaries: Vec::new(),
            frame_count: 0,
            last_boundary_frame: 0,
            prev_quadrant_histograms: Option::None,
        }
    }

    /// Process a grayscale frame (u8 buffer of `width * height` bytes).
    ///
    /// Call this in sequence for each frame. Detected boundaries are
    /// accumulated internally.
    pub fn process_frame(&mut self, frame: &[u8], width: u32, height: u32) {
        let total_pixels = (width as usize) * (height as usize);
        let effective_len = frame.len().min(total_pixels);

        // Compute 256-bin histogram.
        let histogram = compute_histogram(&frame[..effective_len]);

        // Compute quadrant histograms for wipe detection.
        let quadrant_hists = compute_quadrant_histograms(frame, width, height);

        // Compute average brightness.
        let avg_brightness = compute_avg_brightness(&histogram, effective_len);

        if let Some(ref prev_hist) = self.prev_histogram {
            let diff = histogram_difference(prev_hist, &histogram, effective_len);

            // Check min_gap constraint.
            let gap_ok = self.frame_count.saturating_sub(self.last_boundary_frame)
                >= self.config.min_gap
                || self.boundaries.is_empty();

            if gap_ok {
                // Hard cut: large instantaneous difference.
                if diff > self.config.cut_threshold {
                    let confidence = diff.min(1.0);
                    self.boundaries.push(ShotBoundary {
                        frame_index: self.frame_count,
                        transition_type: TransitionType::HardCut,
                        confidence,
                        duration_frames: 1,
                    });
                    self.last_boundary_frame = self.frame_count;
                    self.history.clear();
                }
                // Fade: brightness near black or white.
                else if avg_brightness < self.config.fade_brightness_threshold
                    || avg_brightness > (255.0 - self.config.fade_brightness_threshold)
                {
                    if diff > self.config.gradual_threshold {
                        let confidence = diff.min(1.0);
                        self.boundaries.push(ShotBoundary {
                            frame_index: self.frame_count,
                            transition_type: TransitionType::Fade,
                            confidence,
                            duration_frames: 1,
                        });
                        self.last_boundary_frame = self.frame_count;
                        self.history.clear();
                    }
                }
                // Wipe: spatially localised change.
                else if let Some(ref prev_quads) = self.prev_quadrant_histograms {
                    if diff > self.config.gradual_threshold
                        && is_spatially_asymmetric(prev_quads, &quadrant_hists, effective_len)
                    {
                        let confidence = diff.min(1.0);
                        self.boundaries.push(ShotBoundary {
                            frame_index: self.frame_count,
                            transition_type: TransitionType::Wipe,
                            confidence,
                            duration_frames: 1,
                        });
                        self.last_boundary_frame = self.frame_count;
                        self.history.clear();
                    } else {
                        // Accumulate for dissolve detection.
                        self.history.push(diff);
                        self.try_detect_dissolve();
                    }
                } else {
                    self.history.push(diff);
                    self.try_detect_dissolve();
                }
            } else {
                // Below min_gap: still track history but do not emit.
                if diff > self.config.gradual_threshold {
                    self.history.push(diff);
                } else {
                    self.history.clear();
                }
            }
        }

        self.prev_histogram = Some(histogram);
        self.prev_quadrant_histograms = Some(quadrant_hists);
        self.frame_count += 1;
    }

    /// Get all detected boundaries so far.
    pub fn boundaries(&self) -> &[ShotBoundary] {
        &self.boundaries
    }

    /// Finalize and return all boundaries, consuming the detector.
    pub fn finalize(self) -> Vec<ShotBoundary> {
        self.boundaries
    }

    /// Reset detector state for reuse.
    pub fn reset(&mut self) {
        self.prev_histogram = Option::None;
        self.prev_quadrant_histograms = Option::None;
        self.history.clear();
        self.boundaries.clear();
        self.frame_count = 0;
        self.last_boundary_frame = 0;
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Try to detect a dissolve from accumulated history.
    fn try_detect_dissolve(&mut self) {
        // Need at least 3 consecutive frames above gradual_threshold.
        let above_count = self
            .history
            .iter()
            .rev()
            .take_while(|&&d| d > self.config.gradual_threshold)
            .count();

        if above_count >= 3 {
            let gap_ok = self.frame_count.saturating_sub(self.last_boundary_frame)
                >= self.config.min_gap
                || self.boundaries.is_empty();

            if gap_ok {
                let avg_diff: f32 =
                    self.history.iter().rev().take(above_count).sum::<f32>() / above_count as f32;

                let confidence = avg_diff.min(1.0);
                self.boundaries.push(ShotBoundary {
                    frame_index: self.frame_count.saturating_sub(above_count as u32),
                    transition_type: TransitionType::Dissolve,
                    confidence,
                    duration_frames: above_count as u32,
                });
                self.last_boundary_frame = self.frame_count;
            }
            self.history.clear();
        }
    }
}

// -----------------------------------------------------------------------
// Free functions
// -----------------------------------------------------------------------

/// Compute a 256-bin histogram from a grayscale buffer.
fn compute_histogram(frame: &[u8]) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for &pixel in frame {
        hist[pixel as usize] += 1;
    }
    hist
}

/// Compute quadrant histograms (TL, TR, BL, BR).
fn compute_quadrant_histograms(frame: &[u8], width: u32, height: u32) -> [[u32; 256]; 4] {
    let mut quads = [[0u32; 256]; 4];
    let w = width as usize;
    let h = height as usize;
    let half_w = w / 2;
    let half_h = h / 2;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if idx >= frame.len() {
                break;
            }
            let pixel = frame[idx] as usize;
            let quad_idx = match (x < half_w, y < half_h) {
                (true, true) => 0,   // TL
                (false, true) => 1,  // TR
                (true, false) => 2,  // BL
                (false, false) => 3, // BR
            };
            quads[quad_idx][pixel] += 1;
        }
    }
    quads
}

/// Compute histogram difference: `sum(|h1[i] - h2[i]|) / (2 * total_pixels)`.
fn histogram_difference(h1: &[u32; 256], h2: &[u32; 256], total_pixels: usize) -> f32 {
    if total_pixels == 0 {
        return 0.0;
    }
    let sum: u64 = h1
        .iter()
        .zip(h2.iter())
        .map(|(&a, &b)| (a as i64 - b as i64).unsigned_abs())
        .sum();
    sum as f32 / (2.0 * total_pixels as f32)
}

/// Compute average brightness from a histogram.
fn compute_avg_brightness(hist: &[u32; 256], total_pixels: usize) -> f32 {
    if total_pixels == 0 {
        return 0.0;
    }
    let weighted_sum: u64 = hist
        .iter()
        .enumerate()
        .map(|(i, &count)| i as u64 * count as u64)
        .sum();
    weighted_sum as f32 / total_pixels as f32
}

/// Check whether quadrant differences are spatially asymmetric (wipe indicator).
///
/// Returns `true` when one pair of quadrants has significantly higher difference
/// than the opposing pair, suggesting a travelling edge rather than a uniform dissolve.
fn is_spatially_asymmetric(
    prev: &[[u32; 256]; 4],
    curr: &[[u32; 256]; 4],
    total_pixels: usize,
) -> bool {
    let quad_pixels = total_pixels.max(4) / 4;

    let diffs: Vec<f32> = (0..4)
        .map(|q| {
            let sum: u64 = prev[q]
                .iter()
                .zip(curr[q].iter())
                .map(|(&a, &b)| (a as i64 - b as i64).unsigned_abs())
                .sum();
            sum as f32 / (2.0 * quad_pixels as f32)
        })
        .collect();

    let max_diff = diffs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_diff = diffs.iter().cloned().fold(f32::INFINITY, f32::min);

    // If the maximum quadrant diff is at least 3x the minimum, consider it asymmetric.
    if min_diff < 1e-9 {
        return max_diff > 0.05;
    }
    max_diff / min_diff >= 3.0
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a flat grayscale frame of the given brightness.
    fn flat_frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height) as usize]
    }

    // 1. Identical frames produce no boundaries.
    #[test]
    fn test_detector_no_boundaries_static() {
        let config = ShotBoundaryConfig::default();
        let mut det = ShotBoundaryDetector::new(config);
        let frame = flat_frame(64, 64, 128);
        for _ in 0..20 {
            det.process_frame(&frame, 64, 64);
        }
        assert!(
            det.boundaries().is_empty(),
            "expected no boundaries for static frames, got {}",
            det.boundaries().len()
        );
    }

    // 2. Sudden change detected as HardCut.
    #[test]
    fn test_detector_hard_cut() {
        let config = ShotBoundaryConfig::default();
        let mut det = ShotBoundaryDetector::new(config);
        let frame_a = flat_frame(64, 64, 30);
        let frame_b = flat_frame(64, 64, 220);

        for _ in 0..3 {
            det.process_frame(&frame_a, 64, 64);
        }
        det.process_frame(&frame_b, 64, 64);

        let boundaries = det.boundaries();
        assert!(
            !boundaries.is_empty(),
            "expected at least one boundary on hard cut"
        );
        assert_eq!(boundaries[0].transition_type, TransitionType::HardCut);
    }

    // 3. Gradual darkening detected as Fade.
    #[test]
    fn test_detector_fade_to_black() {
        let mut config = ShotBoundaryConfig::default();
        config.cut_threshold = 0.5;
        config.gradual_threshold = 0.01;
        config.fade_brightness_threshold = 30.0;
        config.min_gap = 1;

        let mut det = ShotBoundaryDetector::new(config);
        let w = 64u32;
        let h = 64u32;
        let n = (w * h) as usize;
        // Create frames with a spread of values that gradually darken.
        // Each frame has a Gaussian-like spread around a mean that decreases,
        // so consecutive frames share many pixel values and histogram diff stays small.
        let steps = 20u32;
        for step in 0..steps {
            let mean = 200.0 - (step as f64 * 10.0);
            let mut frame = vec![0u8; n];
            for (idx, pixel) in frame.iter_mut().enumerate() {
                // Spread: value = mean + noise based on position.
                let noise = ((idx % 51) as f64 - 25.0) * 2.0;
                let val = (mean + noise).clamp(0.0, 255.0);
                *pixel = val as u8;
            }
            det.process_frame(&frame, w, h);
        }

        let boundaries = det.boundaries();
        let has_fade = boundaries
            .iter()
            .any(|b| b.transition_type == TransitionType::Fade);
        assert!(has_fade, "expected a Fade boundary, got {boundaries:?}");
    }

    // 4. Boundaries respect min_gap.
    #[test]
    fn test_detector_min_gap() {
        let mut config = ShotBoundaryConfig::default();
        config.min_gap = 10;

        let mut det = ShotBoundaryDetector::new(config);
        let frame_a = flat_frame(32, 32, 30);
        let frame_b = flat_frame(32, 32, 220);

        // Alternate rapidly -- should not get a boundary every frame.
        for i in 0..20 {
            if i % 2 == 0 {
                det.process_frame(&frame_a, 32, 32);
            } else {
                det.process_frame(&frame_b, 32, 32);
            }
        }

        let boundaries = det.boundaries();
        // With min_gap=10, at most 2 boundaries should be detected in 20 frames.
        assert!(
            boundaries.len() <= 2,
            "expected at most 2 boundaries with min_gap=10, got {}",
            boundaries.len()
        );

        // Verify gap constraint: consecutive boundaries should be >= min_gap apart.
        for pair in boundaries.windows(2) {
            let gap = pair[1].frame_index.saturating_sub(pair[0].frame_index);
            assert!(gap >= 10, "boundary gap {} is less than min_gap 10", gap);
        }
    }

    // 5. Reset clears state.
    #[test]
    fn test_detector_reset() {
        let config = ShotBoundaryConfig::default();
        let mut det = ShotBoundaryDetector::new(config);
        let frame_a = flat_frame(32, 32, 30);
        let frame_b = flat_frame(32, 32, 220);

        det.process_frame(&frame_a, 32, 32);
        det.process_frame(&frame_b, 32, 32);
        assert!(!det.boundaries().is_empty());

        det.reset();
        assert!(det.boundaries().is_empty());
        assert_eq!(det.frame_count, 0);
    }

    // 6. Confidence is in [0, 1].
    #[test]
    fn test_detector_confidence_range() {
        let config = ShotBoundaryConfig::default();
        let mut det = ShotBoundaryDetector::new(config);
        let frame_a = flat_frame(32, 32, 10);
        let frame_b = flat_frame(32, 32, 250);

        det.process_frame(&frame_a, 32, 32);
        det.process_frame(&frame_b, 32, 32);

        for b in det.boundaries() {
            assert!(
                b.confidence >= 0.0 && b.confidence <= 1.0,
                "confidence {} out of [0, 1]",
                b.confidence
            );
        }
    }

    // 7. Finalize returns all boundaries.
    #[test]
    fn test_detector_finalize() {
        let config = ShotBoundaryConfig::default();
        let mut det = ShotBoundaryDetector::new(config);
        let frame_a = flat_frame(32, 32, 10);
        let frame_b = flat_frame(32, 32, 250);

        det.process_frame(&frame_a, 32, 32);
        det.process_frame(&frame_b, 32, 32);

        let count = det.boundaries().len();
        let finalized = det.finalize();
        assert_eq!(finalized.len(), count);
    }

    // 8. Defaults are reasonable.
    #[test]
    fn test_default_config() {
        let config = ShotBoundaryConfig::default();
        assert!(config.cut_threshold > 0.0 && config.cut_threshold <= 1.0);
        assert!(config.gradual_threshold > 0.0 && config.gradual_threshold <= 1.0);
        assert!(config.min_gap >= 1);
        assert!(config.fade_brightness_threshold > 0.0);
    }
}
