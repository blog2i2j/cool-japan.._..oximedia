//! Detect and report duplicate or near-duplicate frames.
//!
//! Provides a streaming [`DuplicateDetector`] that processes frames one at a
//! time and tracks which ones are duplicates of the previous frame. Three
//! comparison methods are available:
//!
//! * **MAD** (Mean Absolute Difference) -- fast, default.
//! * **Histogram** -- compares 256-bin histogram intersection.
//! * **Exact** -- byte-for-byte comparison, falls back to MAD if not identical.
//!
//! The standalone [`frame_similarity`] function compares two buffers directly.

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// Result of duplicate frame analysis for a single frame.
#[derive(Debug, Clone)]
pub struct DuplicateResult {
    /// Frame index.
    pub frame_index: u32,
    /// Whether this frame is a duplicate of the previous.
    pub is_duplicate: bool,
    /// Similarity to previous frame \[0.0, 1.0\]. 1.0 = identical.
    pub similarity: f32,
}

/// Configuration for the duplicate frame detector.
#[derive(Debug, Clone)]
pub struct DuplicateConfig {
    /// Similarity threshold above which a frame is considered duplicate \[0.0, 1.0\].
    pub threshold: f32,
    /// Method for comparing frames.
    pub method: CompareMethod,
}

/// Method used to compare two frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompareMethod {
    /// Mean Absolute Difference (fast).
    #[default]
    Mad,
    /// Histogram intersection (moderate).
    Histogram,
    /// Pixel-exact comparison (strict). Falls back to MAD similarity if not identical.
    Exact,
}

impl Default for DuplicateConfig {
    fn default() -> Self {
        Self {
            threshold: 0.98,
            method: CompareMethod::Mad,
        }
    }
}

/// Streaming duplicate frame detector.
///
/// Feed frames sequentially with [`process_frame`](DuplicateDetector::process_frame),
/// then retrieve results with [`results`](DuplicateDetector::results) or
/// [`finalize`](DuplicateDetector::finalize).
pub struct DuplicateDetector {
    config: DuplicateConfig,
    prev_frame: Option<Vec<u8>>,
    results: Vec<DuplicateResult>,
    frame_count: u32,
    dup_count: u32,
}

impl DuplicateDetector {
    /// Create a new detector with the given configuration.
    pub fn new(config: DuplicateConfig) -> Self {
        Self {
            config,
            prev_frame: Option::None,
            results: Vec::new(),
            frame_count: 0,
            dup_count: 0,
        }
    }

    /// Process a frame (grayscale or raw u8 buffer).
    ///
    /// The first frame is never flagged as duplicate. Subsequent frames are
    /// compared against the previous frame using the configured method.
    pub fn process_frame(&mut self, frame: &[u8]) {
        let similarity = if let Some(ref prev) = self.prev_frame {
            frame_similarity(prev, frame, self.config.method)
        } else {
            // First frame: no previous to compare against; similarity = 0.
            0.0
        };

        let is_duplicate = self.prev_frame.is_some() && similarity >= self.config.threshold;
        if is_duplicate {
            self.dup_count += 1;
        }

        self.results.push(DuplicateResult {
            frame_index: self.frame_count,
            is_duplicate,
            similarity,
        });

        self.prev_frame = Some(frame.to_vec());
        self.frame_count += 1;
    }

    /// Get all results so far.
    pub fn results(&self) -> &[DuplicateResult] {
        &self.results
    }

    /// Return the number of frames flagged as duplicates.
    pub fn duplicate_count(&self) -> u32 {
        self.dup_count
    }

    /// Finalize and return all results, consuming the detector.
    pub fn finalize(self) -> Vec<DuplicateResult> {
        self.results
    }

    /// Reset detector state for reuse.
    pub fn reset(&mut self) {
        self.prev_frame = Option::None;
        self.results.clear();
        self.frame_count = 0;
        self.dup_count = 0;
    }
}

// -----------------------------------------------------------------------
// Public free function
// -----------------------------------------------------------------------

/// Compare two frame buffers and return similarity in \[0.0, 1.0\].
///
/// * **Mad**: `1.0 - (sum(|a[i] - b[i]|) / (len * 255.0))`
/// * **Histogram**: histogram intersection `sum(min(h1[i], h2[i])) / sum(h1[i])`
/// * **Exact**: `1.0` if byte-identical, otherwise falls back to MAD.
pub fn frame_similarity(a: &[u8], b: &[u8], method: CompareMethod) -> f32 {
    match method {
        CompareMethod::Mad => mad_similarity(a, b),
        CompareMethod::Histogram => histogram_similarity(a, b),
        CompareMethod::Exact => {
            if a == b {
                1.0
            } else {
                mad_similarity(a, b)
            }
        }
    }
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// MAD similarity: `1.0 - mean_absolute_difference / 255.0`.
fn mad_similarity(a: &[u8], b: &[u8]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 1.0; // Two empty buffers are trivially identical.
    }
    let total_diff: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    let norm = total_diff as f64 / (len as f64 * 255.0);
    (1.0 - norm).clamp(0.0, 1.0) as f32
}

/// Histogram intersection similarity.
///
/// Computes 256-bin histograms for both buffers, then returns
/// `sum(min(h1[i], h2[i])) / sum(h1[i])`.
fn histogram_similarity(a: &[u8], b: &[u8]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let mut hist_a = [0u64; 256];
    let mut hist_b = [0u64; 256];

    for &p in a {
        hist_a[p as usize] += 1;
    }
    for &p in b {
        hist_b[p as usize] += 1;
    }

    let sum_a: u64 = hist_a.iter().sum();
    if sum_a == 0 {
        return 0.0;
    }

    let intersection: u64 = hist_a
        .iter()
        .zip(hist_b.iter())
        .map(|(&ha, &hb)| ha.min(hb))
        .sum();

    (intersection as f64 / sum_a as f64).clamp(0.0, 1.0) as f32
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(len: usize, value: u8) -> Vec<u8> {
        vec![value; len]
    }

    // 1. Identical frames detected as duplicates.
    #[test]
    fn test_identical_frames_detected() {
        let config = DuplicateConfig::default();
        let mut det = DuplicateDetector::new(config);
        let frame = flat(1024, 128);

        det.process_frame(&frame);
        det.process_frame(&frame);
        det.process_frame(&frame);

        let results = det.results();
        // Frame 0 is never a duplicate.
        assert!(!results[0].is_duplicate);
        // Frames 1 and 2 should be duplicates.
        assert!(results[1].is_duplicate);
        assert!(results[2].is_duplicate);
    }

    // 2. Different frames are not flagged.
    #[test]
    fn test_different_frames_not_duplicate() {
        let config = DuplicateConfig::default();
        let mut det = DuplicateDetector::new(config);

        det.process_frame(&flat(1024, 0));
        det.process_frame(&flat(1024, 255));

        let results = det.results();
        assert!(!results[1].is_duplicate);
    }

    // 3. Near-duplicate threshold.
    #[test]
    fn test_near_duplicate_threshold() {
        let mut config = DuplicateConfig::default();
        config.threshold = 0.99;
        let mut det = DuplicateDetector::new(config);

        let frame_a = flat(1024, 100);
        let mut frame_b = flat(1024, 100);
        // Change just one pixel slightly.
        frame_b[0] = 101;

        det.process_frame(&frame_a);
        det.process_frame(&frame_b);

        let results = det.results();
        // Should still be considered a duplicate because similarity > 0.99.
        assert!(
            results[1].is_duplicate,
            "similarity {} should be >= 0.99",
            results[1].similarity
        );
    }

    // 4. Duplicate count is correct.
    #[test]
    fn test_duplicate_count() {
        let config = DuplicateConfig::default();
        let mut det = DuplicateDetector::new(config);
        let frame = flat(256, 64);

        det.process_frame(&frame);
        det.process_frame(&frame);
        det.process_frame(&flat(256, 0)); // different
        det.process_frame(&flat(256, 0)); // dup of previous
        det.process_frame(&flat(256, 0)); // dup of previous

        assert_eq!(det.duplicate_count(), 3);
    }

    // 5. frame_similarity: identical buffers -> 1.0.
    #[test]
    fn test_frame_similarity_identical() {
        let frame = flat(512, 100);
        let sim = frame_similarity(&frame, &frame, CompareMethod::Mad);
        assert!((sim - 1.0).abs() < 1e-5, "expected 1.0, got {sim}");
    }

    // 6. frame_similarity: black vs white -> low similarity.
    #[test]
    fn test_frame_similarity_opposite() {
        let black = flat(512, 0);
        let white = flat(512, 255);
        let sim = frame_similarity(&black, &white, CompareMethod::Mad);
        assert!(sim < 0.01, "expected near 0.0, got {sim}");
    }

    // 7. Exact method: pixel-exact comparison works.
    #[test]
    fn test_exact_method() {
        let frame_a = flat(256, 100);
        let frame_b = flat(256, 100);
        let sim = frame_similarity(&frame_a, &frame_b, CompareMethod::Exact);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "expected 1.0 for identical frames, got {sim}"
        );

        // Slightly different -> falls back to MAD.
        let mut frame_c = flat(256, 100);
        frame_c[0] = 101;
        let sim2 = frame_similarity(&frame_a, &frame_c, CompareMethod::Exact);
        assert!(sim2 < 1.0, "expected < 1.0 for non-identical, got {sim2}");
        assert!(
            sim2 > 0.99,
            "expected high similarity for near-identical, got {sim2}"
        );
    }

    // 8. Reset clears previous frame.
    #[test]
    fn test_reset() {
        let config = DuplicateConfig::default();
        let mut det = DuplicateDetector::new(config);
        let frame = flat(128, 50);

        det.process_frame(&frame);
        det.process_frame(&frame);
        assert_eq!(det.duplicate_count(), 1);

        det.reset();
        assert!(det.results().is_empty());
        assert_eq!(det.duplicate_count(), 0);

        // After reset, first frame should not be flagged.
        det.process_frame(&frame);
        assert!(!det.results()[0].is_duplicate);
    }
}
