#![allow(dead_code)]

//! Shot density analysis for video sequences.
//!
//! This module computes how densely packed shots are across a timeline,
//! producing per-window density metrics, identifying high-activity bursts,
//! and summarising the overall editing pace.

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single shot span used as input to density analysis.
#[derive(Debug, Clone)]
pub struct ShotSpan {
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
}

impl ShotSpan {
    /// Create a new shot span.
    pub fn new(start: f64, end: f64) -> Self {
        Self { start, end }
    }

    /// Duration of the span in seconds.
    pub fn duration(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }
}

/// Density measurement for a single time window.
#[derive(Debug, Clone)]
pub struct DensityWindow {
    /// Start of the window in seconds.
    pub window_start: f64,
    /// End of the window in seconds.
    pub window_end: f64,
    /// Number of shots that overlap this window.
    pub shot_count: usize,
    /// Shots per second within this window.
    pub shots_per_second: f64,
}

/// A detected burst (region of high density).
#[derive(Debug, Clone)]
pub struct DensityBurst {
    /// Start of the burst region in seconds.
    pub start: f64,
    /// End of the burst region in seconds.
    pub end: f64,
    /// Peak shots-per-second observed in this burst.
    pub peak_density: f64,
    /// Total shots within the burst.
    pub shot_count: usize,
}

/// Summary statistics produced by density analysis.
#[derive(Debug, Clone)]
pub struct DensitySummary {
    /// Total number of shots analysed.
    pub total_shots: usize,
    /// Overall timeline duration in seconds.
    pub total_duration: f64,
    /// Mean shots per second across the timeline.
    pub mean_density: f64,
    /// Maximum shots per second in any window.
    pub max_density: f64,
    /// Minimum shots per second in any window (may be 0).
    pub min_density: f64,
    /// Standard deviation of per-window density.
    pub std_dev: f64,
    /// Detected bursts.
    pub bursts: Vec<DensityBurst>,
}

/// Analyser that computes shot density metrics.
#[derive(Debug, Clone)]
pub struct DensityAnalyzer {
    /// Window size in seconds for density computation.
    window_size: f64,
    /// Step size in seconds between consecutive windows.
    step_size: f64,
    /// Threshold (shots/second) above which a region is considered a burst.
    burst_threshold: f64,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl Default for DensityAnalyzer {
    fn default() -> Self {
        Self {
            window_size: 5.0,
            step_size: 1.0,
            burst_threshold: 2.0,
        }
    }
}

impl DensityAnalyzer {
    /// Create a new analyser with custom parameters.
    pub fn new(window_size: f64, step_size: f64, burst_threshold: f64) -> Self {
        Self {
            window_size,
            step_size,
            burst_threshold,
        }
    }

    /// Compute per-window density for the given shots.
    pub fn compute_windows(&self, shots: &[ShotSpan]) -> Vec<DensityWindow> {
        if shots.is_empty() {
            return Vec::new();
        }

        let timeline_end = shots
            .iter()
            .map(|s| s.end)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut windows = Vec::new();
        let mut t = 0.0_f64;
        while t < timeline_end {
            let w_end = t + self.window_size;
            let count = shots
                .iter()
                .filter(|s| s.start < w_end && s.end > t)
                .count();
            #[allow(clippy::cast_precision_loss)]
            let sps = count as f64 / self.window_size;
            windows.push(DensityWindow {
                window_start: t,
                window_end: w_end,
                shot_count: count,
                shots_per_second: sps,
            });
            t += self.step_size;
        }
        windows
    }

    /// Detect bursts from per-window density data.
    pub fn detect_bursts(&self, windows: &[DensityWindow]) -> Vec<DensityBurst> {
        let mut bursts: Vec<DensityBurst> = Vec::new();
        let mut in_burst = false;
        let mut burst_start = 0.0;
        let mut peak = 0.0_f64;
        let mut count = 0_usize;

        for w in windows {
            if w.shots_per_second >= self.burst_threshold {
                if !in_burst {
                    in_burst = true;
                    burst_start = w.window_start;
                    peak = w.shots_per_second;
                    count = w.shot_count;
                } else {
                    peak = peak.max(w.shots_per_second);
                    count = count.max(w.shot_count);
                }
            } else if in_burst {
                bursts.push(DensityBurst {
                    start: burst_start,
                    end: w.window_start,
                    peak_density: peak,
                    shot_count: count,
                });
                in_burst = false;
            }
        }
        // Close any open burst.
        if in_burst {
            if let Some(last) = windows.last() {
                bursts.push(DensityBurst {
                    start: burst_start,
                    end: last.window_end,
                    peak_density: peak,
                    shot_count: count,
                });
            }
        }
        bursts
    }

    /// Produce a full density summary for the given shots.
    #[allow(clippy::cast_precision_loss)]
    pub fn summarize(&self, shots: &[ShotSpan]) -> DensitySummary {
        let windows = self.compute_windows(shots);
        let bursts = self.detect_bursts(&windows);

        let total_shots = shots.len();
        let total_duration = shots.iter().map(|s| s.end).fold(0.0_f64, f64::max);

        if windows.is_empty() {
            return DensitySummary {
                total_shots,
                total_duration,
                mean_density: 0.0,
                max_density: 0.0,
                min_density: 0.0,
                std_dev: 0.0,
                bursts,
            };
        }

        let densities: Vec<f64> = windows.iter().map(|w| w.shots_per_second).collect();
        let n = densities.len() as f64;
        let mean = densities.iter().sum::<f64>() / n;
        let max = densities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = densities.iter().cloned().fold(f64::INFINITY, f64::min);
        let variance = densities.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        DensitySummary {
            total_shots,
            total_duration,
            mean_density: mean,
            max_density: max,
            min_density: min,
            std_dev,
            bursts,
        }
    }

    /// Return the configured window size.
    pub fn window_size(&self) -> f64 {
        self.window_size
    }

    /// Return the configured step size.
    pub fn step_size(&self) -> f64 {
        self.step_size
    }

    /// Return the configured burst threshold.
    pub fn burst_threshold(&self) -> f64 {
        self.burst_threshold
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_shots(ranges: &[(f64, f64)]) -> Vec<ShotSpan> {
        ranges.iter().map(|&(s, e)| ShotSpan::new(s, e)).collect()
    }

    #[test]
    fn test_shot_span_duration() {
        let s = ShotSpan::new(1.0, 3.5);
        assert!((s.duration() - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_shot_span_zero_duration() {
        let s = ShotSpan::new(5.0, 5.0);
        assert!((s.duration() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_shots_windows() {
        let a = DensityAnalyzer::default();
        let w = a.compute_windows(&[]);
        assert!(w.is_empty());
    }

    #[test]
    fn test_single_shot_windows() {
        let a = DensityAnalyzer::new(10.0, 10.0, 999.0);
        let shots = make_shots(&[(0.0, 5.0)]);
        let w = a.compute_windows(&shots);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].shot_count, 1);
    }

    #[test]
    fn test_multiple_shots_overlap() {
        let a = DensityAnalyzer::new(5.0, 5.0, 999.0);
        let shots = make_shots(&[(0.0, 3.0), (2.0, 4.0), (4.0, 6.0)]);
        let w = a.compute_windows(&shots);
        // First window [0,5): all three overlap
        assert!(w[0].shot_count >= 2);
    }

    #[test]
    fn test_detect_no_bursts() {
        let a = DensityAnalyzer::new(5.0, 1.0, 999.0);
        let shots = make_shots(&[(0.0, 10.0)]);
        let windows = a.compute_windows(&shots);
        let bursts = a.detect_bursts(&windows);
        assert!(bursts.is_empty());
    }

    #[test]
    fn test_detect_burst() {
        // Many short shots create high density.
        let a = DensityAnalyzer::new(2.0, 1.0, 1.5);
        let shots = make_shots(&[(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (10.0, 15.0)]);
        let windows = a.compute_windows(&shots);
        let bursts = a.detect_bursts(&windows);
        assert!(!bursts.is_empty());
        assert!(bursts[0].peak_density >= 1.5);
    }

    #[test]
    fn test_summarize_empty() {
        let a = DensityAnalyzer::default();
        let s = a.summarize(&[]);
        assert_eq!(s.total_shots, 0);
        assert!((s.mean_density - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_summarize_uniform() {
        let a = DensityAnalyzer::new(5.0, 5.0, 999.0);
        let shots = make_shots(&[(0.0, 5.0), (5.0, 10.0)]);
        let s = a.summarize(&shots);
        assert_eq!(s.total_shots, 2);
        assert!(s.total_duration > 0.0);
    }

    #[test]
    fn test_summarize_std_dev() {
        let a = DensityAnalyzer::new(5.0, 5.0, 999.0);
        // Uniform density → low std dev.
        let shots = make_shots(&[(0.0, 5.0), (5.0, 10.0)]);
        let s = a.summarize(&shots);
        assert!(s.std_dev < 1.0);
    }

    #[test]
    fn test_analyzer_accessors() {
        let a = DensityAnalyzer::new(3.0, 1.5, 4.0);
        assert!((a.window_size() - 3.0).abs() < f64::EPSILON);
        assert!((a.step_size() - 1.5).abs() < f64::EPSILON);
        assert!((a.burst_threshold() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_analyzer() {
        let a = DensityAnalyzer::default();
        assert!((a.window_size() - 5.0).abs() < f64::EPSILON);
        assert!((a.step_size() - 1.0).abs() < f64::EPSILON);
        assert!((a.burst_threshold() - 2.0).abs() < f64::EPSILON);
    }
}
