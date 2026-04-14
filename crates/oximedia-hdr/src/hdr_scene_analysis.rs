//! HDR scene brightness analysis with rolling-window peak luminance tracking
//! and scene-cut detection.
//!
//! This module provides frame-level luminance ingestion, a rolling-window
//! accumulator for detecting scene cuts via luminance delta, and per-scene
//! summary statistics including mean, median, 95th/99th percentile, and
//! dynamic range ratio.
//!
//! # Quick start
//! ```rust,ignore
//! use oximedia_hdr::hdr_scene_analysis::{SceneAnalyzer, SceneAnalyzerConfig};
//!
//! let config = SceneAnalyzerConfig::default();
//! let mut analyzer = SceneAnalyzer::new(config);
//!
//! // Feed per-frame peak nits (e.g. from FrameLuminanceStats::max_cll_nits)
//! for peak in frame_peaks {
//!     analyzer.push_frame(peak)?;
//! }
//! let scenes = analyzer.finish();
//! for scene in &scenes {
//!     println!("scene frames={} peak={:.1} nits", scene.frame_count, scene.peak_nits);
//! }
//! ```

use crate::{HdrError, Result};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for [`SceneAnalyzer`].
#[derive(Debug, Clone)]
pub struct SceneAnalyzerConfig {
    /// Rolling window size (number of frames) used to compute a smoothed
    /// baseline luminance. A scene cut is detected when the current frame's
    /// peak luminance deviates from the window mean by more than
    /// [`cut_threshold_ratio`].
    ///
    /// Default: 30 frames.
    pub window_size: usize,

    /// Fractional deviation from the rolling mean that triggers a scene cut.
    /// For example, 0.5 means a 50 % jump or drop triggers a new scene.
    ///
    /// Default: 0.5.
    pub cut_threshold_ratio: f32,

    /// Minimum number of frames a scene must contain before a new cut is
    /// accepted.  Prevents spurious single-frame flashes from creating many
    /// tiny scenes.
    ///
    /// Default: 8 frames.
    pub min_scene_frames: usize,

    /// Maximum peak luminance (nits) that the analyzer accepts.  Values above
    /// this are clamped to avoid outliers polluting statistics.
    ///
    /// Default: 10_000.0 nits (PQ ceiling).
    pub max_nits_clamp: f32,
}

impl Default for SceneAnalyzerConfig {
    fn default() -> Self {
        Self {
            window_size: 30,
            cut_threshold_ratio: 0.5,
            min_scene_frames: 8,
            max_nits_clamp: 10_000.0,
        }
    }
}

// ── Per-scene statistics ──────────────────────────────────────────────────────

/// Summary statistics for a single detected HDR scene.
///
/// All luminance values are in absolute nits (cd/m²).
#[derive(Debug, Clone)]
pub struct SceneStats {
    /// Zero-based index of the first frame in this scene.
    pub start_frame: u64,
    /// Total number of frames belonging to this scene.
    pub frame_count: u64,
    /// Maximum (peak) frame luminance observed across the scene.
    pub peak_nits: f32,
    /// Minimum frame luminance observed across the scene.
    pub min_nits: f32,
    /// Arithmetic mean of per-frame peak luminances.
    pub mean_nits: f32,
    /// Median of per-frame peak luminances (50th percentile).
    pub median_nits: f32,
    /// 95th-percentile frame peak luminance.
    pub p95_nits: f32,
    /// 99th-percentile frame peak luminance.
    pub p99_nits: f32,
    /// Dynamic range ratio: `peak_nits / (min_nits + 1.0)`.
    ///
    /// The `+ 1.0` avoids division-by-zero for near-black frames.
    pub dynamic_range_ratio: f32,
}

impl SceneStats {
    /// Build a `SceneStats` from a non-empty slice of per-frame peak nit values
    /// and the index of the first frame.
    ///
    /// # Errors
    /// Returns `HdrError::InvalidLuminance` if `samples` is empty or contains
    /// any negative value.
    pub fn from_samples(samples: &[f32], start_frame: u64) -> Result<Self> {
        if samples.is_empty() {
            return Err(HdrError::InvalidLuminance(0.0));
        }
        for &v in samples {
            if v < 0.0 {
                return Err(HdrError::InvalidLuminance(v));
            }
        }

        let frame_count = samples.len() as u64;
        let peak_nits = samples
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_nits = samples
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let mean_nits = samples.iter().copied().sum::<f32>() / samples.len() as f32;

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_nits = percentile_sorted(&sorted, 0.50);
        let p95_nits = percentile_sorted(&sorted, 0.95);
        let p99_nits = percentile_sorted(&sorted, 0.99);
        let dynamic_range_ratio = peak_nits / (min_nits + 1.0);

        Ok(Self {
            start_frame,
            frame_count,
            peak_nits,
            min_nits,
            mean_nits,
            median_nits,
            p95_nits,
            p99_nits,
            dynamic_range_ratio,
        })
    }
}

// ── Rolling-window state ──────────────────────────────────────────────────────

/// A fixed-capacity circular buffer holding the last `capacity` f32 values
/// and maintaining a running sum for O(1) mean computation.
#[derive(Debug, Clone)]
struct RollingWindow {
    buf: Vec<f32>,
    head: usize,
    count: usize,
    sum: f64,
}

impl RollingWindow {
    fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0.0_f32; capacity.max(1)],
            head: 0,
            count: 0,
            sum: 0.0,
        }
    }

    fn capacity(&self) -> usize {
        self.buf.len()
    }

    fn push(&mut self, value: f32) {
        if self.count == self.capacity() {
            // Evict the oldest element.
            self.sum -= self.buf[self.head] as f64;
        } else {
            self.count += 1;
        }
        self.buf[self.head] = value;
        self.sum += value as f64;
        self.head = (self.head + 1) % self.capacity();
    }

    fn mean(&self) -> Option<f32> {
        if self.count == 0 {
            None
        } else {
            Some((self.sum / self.count as f64) as f32)
        }
    }

    fn is_full(&self) -> bool {
        self.count == self.capacity()
    }
}

// ── SceneAnalyzer ─────────────────────────────────────────────────────────────

/// Incremental HDR scene analyser.
///
/// Feed one per-frame peak-nit value at a time via [`push_frame`].  The
/// analyser maintains a rolling window to detect scene cuts.  Call [`finish`]
/// at the end of the clip to flush the last scene.
///
/// [`push_frame`]: SceneAnalyzer::push_frame
/// [`finish`]: SceneAnalyzer::finish
#[derive(Debug)]
pub struct SceneAnalyzer {
    config: SceneAnalyzerConfig,
    window: RollingWindow,
    /// Frames accumulated for the current scene (not yet committed).
    current_scene_frames: Vec<f32>,
    /// Global frame index of the first frame in the current scene.
    current_scene_start: u64,
    /// Total frames fed so far.
    total_frames: u64,
    /// Completed scenes.
    completed_scenes: Vec<SceneStats>,
}

impl SceneAnalyzer {
    /// Create a new `SceneAnalyzer` with the given configuration.
    pub fn new(config: SceneAnalyzerConfig) -> Self {
        let window_size = config.window_size.max(1);
        Self {
            window: RollingWindow::new(window_size),
            config,
            current_scene_frames: Vec::new(),
            current_scene_start: 0,
            total_frames: 0,
            completed_scenes: Vec::new(),
        }
    }

    /// Feed a single frame's peak luminance value (in nits).
    ///
    /// The value is clamped to `[0, config.max_nits_clamp]` before processing.
    ///
    /// # Errors
    /// Returns `HdrError::InvalidLuminance` if `peak_nits` is negative (before
    /// clamping) or NaN.
    pub fn push_frame(&mut self, peak_nits: f32) -> Result<()> {
        if peak_nits.is_nan() || peak_nits < 0.0 {
            return Err(HdrError::InvalidLuminance(peak_nits));
        }
        let clamped = peak_nits.min(self.config.max_nits_clamp);
        let frame_idx = self.total_frames;
        self.total_frames += 1;

        // Detect a cut only once the rolling window is fully populated.
        let is_cut = if self.window.is_full() {
            if let Some(mean) = self.window.mean() {
                let deviation = (clamped - mean).abs();
                let threshold = mean * self.config.cut_threshold_ratio;
                deviation > threshold
                    && self.current_scene_frames.len() >= self.config.min_scene_frames
            } else {
                false
            }
        } else {
            false
        };

        if is_cut {
            self.commit_current_scene()?;
            self.current_scene_start = frame_idx;
        }

        self.window.push(clamped);
        self.current_scene_frames.push(clamped);
        Ok(())
    }

    /// Flush all remaining frames as the last scene and return all completed
    /// [`SceneStats`].
    ///
    /// After calling `finish` the analyser is reset to an empty state.
    ///
    /// # Errors
    /// Propagates any error from [`SceneStats::from_samples`].
    pub fn finish(&mut self) -> Result<Vec<SceneStats>> {
        if !self.current_scene_frames.is_empty() {
            self.commit_current_scene()?;
        }
        Ok(std::mem::take(&mut self.completed_scenes))
    }

    /// Number of frames ingested so far.
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Number of scenes committed so far (not including the current
    /// in-progress scene).
    pub fn committed_scene_count(&self) -> usize {
        self.completed_scenes.len()
    }

    // ── private helpers ───────────────────────────────────────────────────────

    fn commit_current_scene(&mut self) -> Result<()> {
        if self.current_scene_frames.is_empty() {
            return Ok(());
        }
        let samples = std::mem::take(&mut self.current_scene_frames);
        let stats = SceneStats::from_samples(&samples, self.current_scene_start)?;
        self.completed_scenes.push(stats);
        Ok(())
    }
}

// ── FramePeakSampler ──────────────────────────────────────────────────────────

/// Lightweight helper that samples peak luminance from an interleaved RGB
/// pixel buffer (linear light, nits) and returns the maximum and
/// frame-average luminance using the BT.2100 luminance coefficients.
///
/// `R`, `G`, `B` channels are in order at indices `[3i, 3i+1, 3i+2]`.
pub struct FramePeakSampler;

impl FramePeakSampler {
    /// BT.2100 luminance coefficients for Rec.2020 primaries.
    const KR: f32 = 0.2627;
    const KG: f32 = 0.6780;
    const KB: f32 = 0.0593;

    /// Compute `(peak_nits, mean_nits)` from a linear-light interleaved RGB
    /// buffer.
    ///
    /// # Errors
    /// Returns `HdrError::InvalidLuminance` if the buffer length is not a
    /// multiple of 3.
    pub fn sample_linear(pixels: &[f32]) -> Result<(f32, f32)> {
        if !pixels.len().is_multiple_of(3) {
            return Err(HdrError::InvalidLuminance(-1.0));
        }
        if pixels.is_empty() {
            return Ok((0.0, 0.0));
        }
        let n_pixels = pixels.len() / 3;
        let mut peak: f32 = 0.0;
        let mut sum: f64 = 0.0;
        for chunk in pixels.chunks_exact(3) {
            let r = chunk[0].max(0.0);
            let g = chunk[1].max(0.0);
            let b = chunk[2].max(0.0);
            let luma = Self::KR * r + Self::KG * g + Self::KB * b;
            if luma > peak {
                peak = luma;
            }
            sum += luma as f64;
        }
        let mean = (sum / n_pixels as f64) as f32;
        Ok((peak, mean))
    }
}

// ── Percentile helper ─────────────────────────────────────────────────────────

/// Return the `p`-th fractional percentile (0.0–1.0) from a **sorted** slice.
///
/// Uses linear interpolation between the two surrounding ranks.
fn percentile_sorted(sorted: &[f32], p: f32) -> f32 {
    debug_assert!(!sorted.is_empty());
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = p * (n - 1) as f32;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = rank - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a simple analyzer with a small window for unit tests.
    fn make_analyzer(window: usize, threshold: f32, min_scene: usize) -> SceneAnalyzer {
        SceneAnalyzer::new(SceneAnalyzerConfig {
            window_size: window,
            cut_threshold_ratio: threshold,
            min_scene_frames: min_scene,
            max_nits_clamp: 10_000.0,
        })
    }

    #[test]
    fn test_scene_stats_basic() {
        let samples = vec![100.0_f32, 200.0, 300.0, 400.0, 500.0];
        let stats = SceneStats::from_samples(&samples, 0).unwrap();
        assert_eq!(stats.frame_count, 5);
        assert!((stats.peak_nits - 500.0).abs() < 1e-3);
        assert!((stats.min_nits - 100.0).abs() < 1e-3);
        assert!((stats.mean_nits - 300.0).abs() < 1e-3);
        assert!((stats.median_nits - 300.0).abs() < 1e-3);
    }

    #[test]
    fn test_scene_stats_empty_returns_error() {
        let result = SceneStats::from_samples(&[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_scene_stats_negative_returns_error() {
        let result = SceneStats::from_samples(&[100.0, -1.0], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_scene_stats_percentiles() {
        // 100 evenly-spaced values [1..100].
        let samples: Vec<f32> = (1..=100).map(|x| x as f32).collect();
        let stats = SceneStats::from_samples(&samples, 0).unwrap();
        // p95 should be ~95 (within 1 nit due to interpolation).
        assert!((stats.p95_nits - 95.05).abs() < 1.0, "p95={}", stats.p95_nits);
        assert!((stats.p99_nits - 99.01).abs() < 1.0, "p99={}", stats.p99_nits);
    }

    #[test]
    fn test_dynamic_range_ratio() {
        let samples = vec![1000.0_f32, 10.0];
        let stats = SceneStats::from_samples(&samples, 0).unwrap();
        // dynamic_range_ratio = 1000 / (10 + 1) ≈ 90.9
        let expected = 1000.0_f32 / 11.0;
        assert!((stats.dynamic_range_ratio - expected).abs() < 1e-2);
    }

    #[test]
    fn test_rolling_window_mean() {
        let mut w = RollingWindow::new(4);
        w.push(10.0);
        w.push(20.0);
        w.push(30.0);
        w.push(40.0);
        let mean = w.mean().unwrap();
        assert!((mean - 25.0).abs() < 1e-3, "mean={mean}");
        // Push 50 — oldest (10) evicted; new mean = (20+30+40+50)/4 = 35.
        w.push(50.0);
        let mean2 = w.mean().unwrap();
        assert!((mean2 - 35.0).abs() < 1e-3, "mean2={mean2}");
    }

    #[test]
    fn test_analyzer_no_cut_uniform_scene() {
        let mut analyzer = make_analyzer(5, 0.5, 4);
        for _ in 0..20 {
            analyzer.push_frame(500.0).unwrap();
        }
        let scenes = analyzer.finish().unwrap();
        // All 20 frames in one scene.
        assert_eq!(scenes.len(), 1);
        assert_eq!(scenes[0].frame_count, 20);
    }

    #[test]
    fn test_analyzer_detects_scene_cut() {
        // 20 dark frames then 20 bright frames — should produce ≥2 scenes.
        let mut analyzer = make_analyzer(8, 0.5, 8);
        for _ in 0..20 {
            analyzer.push_frame(50.0).unwrap();
        }
        for _ in 0..20 {
            analyzer.push_frame(800.0).unwrap();
        }
        let scenes = analyzer.finish().unwrap();
        assert!(scenes.len() >= 2, "expected >=2 scenes, got {}", scenes.len());
    }

    #[test]
    fn test_analyzer_negative_peak_returns_error() {
        let mut analyzer = make_analyzer(5, 0.5, 4);
        let result = analyzer.push_frame(-10.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_frame_peak_sampler_uniform() {
        // All pixels at (100, 0, 0) → luma = KR * 100 = 26.27 nits.
        let pixels: Vec<f32> = std::iter::repeat([100.0_f32, 0.0, 0.0])
            .take(10)
            .flatten()
            .collect();
        let (peak, mean) = FramePeakSampler::sample_linear(&pixels).unwrap();
        let expected = 0.2627_f32 * 100.0;
        assert!((peak - expected).abs() < 1e-3, "peak={peak}");
        assert!((mean - expected).abs() < 1e-3, "mean={mean}");
    }

    #[test]
    fn test_frame_peak_sampler_invalid_length() {
        let pixels = vec![1.0_f32, 2.0]; // not multiple of 3
        let result = FramePeakSampler::sample_linear(&pixels);
        assert!(result.is_err());
    }

    #[test]
    fn test_frame_peak_sampler_empty() {
        let (peak, mean) = FramePeakSampler::sample_linear(&[]).unwrap();
        assert_eq!(peak, 0.0);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_percentile_single_element() {
        let sorted = vec![42.0_f32];
        assert_eq!(percentile_sorted(&sorted, 0.0), 42.0);
        assert_eq!(percentile_sorted(&sorted, 0.5), 42.0);
        assert_eq!(percentile_sorted(&sorted, 1.0), 42.0);
    }

    #[test]
    fn test_analyzer_clamp_applied() {
        let mut analyzer = SceneAnalyzer::new(SceneAnalyzerConfig {
            max_nits_clamp: 1000.0,
            ..Default::default()
        });
        // Feed a value above the clamp.
        analyzer.push_frame(20_000.0).unwrap();
        let scenes = analyzer.finish().unwrap();
        assert_eq!(scenes.len(), 1);
        // Peak should be clamped to 1000.
        assert!(scenes[0].peak_nits <= 1000.0 + 1e-3);
    }
}
