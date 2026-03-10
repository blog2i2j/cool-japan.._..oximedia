//! Beat-grid alignment for music and rhythmic media synchronisation.
//!
//! Detects downbeats in an audio signal and aligns it to a target beat grid,
//! producing a sample-accurate offset.

#![allow(dead_code)]

/// A regular beat grid defined by a tempo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeatGrid {
    /// Tempo in beats per minute.
    pub bpm: f64,
    /// Phase offset of the first beat in milliseconds from the stream start.
    pub phase_offset_ms: f64,
}

impl BeatGrid {
    /// Creates a new beat grid at the given BPM with zero phase offset.
    #[must_use]
    pub fn new(bpm: f64) -> Self {
        Self {
            bpm,
            phase_offset_ms: 0.0,
        }
    }

    /// Creates a beat grid with an explicit phase offset.
    #[must_use]
    pub fn with_phase(bpm: f64, phase_offset_ms: f64) -> Self {
        Self {
            bpm,
            phase_offset_ms,
        }
    }

    /// Returns the interval between consecutive beats in milliseconds.
    #[must_use]
    pub fn interval_ms(&self) -> f64 {
        if self.bpm <= 0.0 {
            f64::INFINITY
        } else {
            60_000.0 / self.bpm
        }
    }

    /// Returns the timestamp (ms from stream start) of the n-th beat (0-indexed).
    #[must_use]
    pub fn beat_time_ms(&self, beat_index: u32) -> f64 {
        self.phase_offset_ms + f64::from(beat_index) * self.interval_ms()
    }

    /// Returns the nearest beat index for a given timestamp (ms).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    #[must_use]
    pub fn nearest_beat(&self, time_ms: f64) -> u32 {
        if self.bpm <= 0.0 {
            return 0;
        }
        let offset = time_ms - self.phase_offset_ms;
        let beat_f = offset / self.interval_ms();
        beat_f.round().max(0.0) as u32
    }
}

/// Configuration for the beat-alignment algorithm.
#[derive(Debug, Clone)]
pub struct BeatAlignConfig {
    /// Target beat grid to align to.
    pub grid: BeatGrid,
    /// Maximum allowed alignment error before it is considered a non-match (ms).
    pub tolerance: f64,
    /// Sample rate of the input audio.
    pub sample_rate: u32,
}

impl BeatAlignConfig {
    /// Creates a new config.
    #[must_use]
    pub fn new(grid: BeatGrid, sample_rate: u32) -> Self {
        Self {
            grid,
            tolerance: 20.0,
            sample_rate,
        }
    }

    /// Returns the alignment tolerance in milliseconds.
    #[must_use]
    pub fn tolerance_ms(&self) -> f64 {
        self.tolerance
    }
}

/// Result of a beat-alignment operation.
#[derive(Debug, Clone, Copy)]
pub struct BeatAlignResult {
    /// Time offset that should be applied to the signal to align it (ms).
    pub offset: f64,
    /// Confidence that the downbeat was correctly detected (0.0–1.0).
    pub confidence: f64,
    /// Beat index within the target grid that the detected downbeat maps to.
    pub matched_beat_index: u32,
}

impl BeatAlignResult {
    /// Returns the offset in milliseconds.
    #[must_use]
    pub fn offset_ms(&self) -> f64 {
        self.offset
    }
}

/// Performs beat-grid alignment on an audio signal.
#[derive(Debug)]
pub struct BeatAligner {
    config: BeatAlignConfig,
}

impl BeatAligner {
    /// Creates a new aligner with the given configuration.
    #[must_use]
    pub fn new(config: BeatAlignConfig) -> Self {
        Self { config }
    }

    /// Returns a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &BeatAlignConfig {
        &self.config
    }

    /// Detects the approximate position of the first downbeat in `samples`.
    ///
    /// Uses a simple energy-onset heuristic: returns the sample index of the
    /// frame with the highest short-window RMS energy.
    ///
    /// Returns `None` when the signal is empty.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn detect_downbeat(&self, samples: &[f32]) -> Option<usize> {
        if samples.is_empty() {
            return None;
        }
        let window = (self.config.sample_rate / 100) as usize; // 10 ms window
        let window = window.max(1);
        let mut best_idx = 0usize;
        let mut best_rms = 0.0f64;

        let mut i = 0usize;
        while i + window <= samples.len() {
            let rms: f64 = samples[i..i + window]
                .iter()
                .map(|&s| f64::from(s) * f64::from(s))
                .sum::<f64>()
                / window as f64;
            if rms > best_rms {
                best_rms = rms;
                best_idx = i;
            }
            i += window;
        }
        Some(best_idx)
    }

    /// Aligns `samples` to the configured beat grid.
    ///
    /// Returns `None` when no reliable downbeat is found or the confidence is
    /// below an acceptable threshold.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn align_to_grid(&self, samples: &[f32]) -> Option<BeatAlignResult> {
        let downbeat_sample = self.detect_downbeat(samples)?;
        let downbeat_ms = (downbeat_sample as f64 / f64::from(self.config.sample_rate)) * 1000.0;

        // Find the grid beat nearest to the detected downbeat.
        let beat_idx = self.config.grid.nearest_beat(downbeat_ms);
        let grid_beat_ms = self.config.grid.beat_time_ms(beat_idx);
        let offset_ms = grid_beat_ms - downbeat_ms;

        // Simple confidence: full confidence when error is zero.
        let error = offset_ms.abs();
        let tolerance = self.config.tolerance_ms();
        let confidence = if error > tolerance {
            0.0
        } else {
            1.0 - error / tolerance
        };

        if confidence < 0.1 {
            return None;
        }

        Some(BeatAlignResult {
            offset: offset_ms,
            confidence,
            matched_beat_index: beat_idx,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(bpm: f64) -> BeatAlignConfig {
        BeatAlignConfig::new(BeatGrid::new(bpm), 48_000)
    }

    #[test]
    fn test_beat_grid_interval_120bpm() {
        let grid = BeatGrid::new(120.0);
        assert!((grid.interval_ms() - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_beat_grid_interval_60bpm() {
        let grid = BeatGrid::new(60.0);
        assert!((grid.interval_ms() - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_beat_grid_interval_zero_bpm() {
        let grid = BeatGrid::new(0.0);
        assert!(grid.interval_ms().is_infinite());
    }

    #[test]
    fn test_beat_grid_beat_time_ms() {
        let grid = BeatGrid::new(120.0); // 500 ms per beat
        assert!((grid.beat_time_ms(0) - 0.0).abs() < 1e-9);
        assert!((grid.beat_time_ms(1) - 500.0).abs() < 1e-9);
        assert!((grid.beat_time_ms(4) - 2000.0).abs() < 1e-9);
    }

    #[test]
    fn test_beat_grid_with_phase() {
        let grid = BeatGrid::with_phase(120.0, 250.0);
        assert!((grid.beat_time_ms(0) - 250.0).abs() < 1e-9);
        assert!((grid.beat_time_ms(1) - 750.0).abs() < 1e-9);
    }

    #[test]
    fn test_beat_grid_nearest_beat() {
        let grid = BeatGrid::new(120.0); // 500 ms / beat
        assert_eq!(grid.nearest_beat(0.0), 0);
        assert_eq!(grid.nearest_beat(499.0), 1);
        assert_eq!(grid.nearest_beat(1000.0), 2);
    }

    #[test]
    fn test_config_tolerance_ms() {
        let cfg = make_config(120.0);
        assert!((cfg.tolerance_ms() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_beat_align_result_offset_ms() {
        let r = BeatAlignResult {
            offset: 12.5,
            confidence: 0.9,
            matched_beat_index: 3,
        };
        assert!((r.offset_ms() - 12.5).abs() < 1e-9);
    }

    #[test]
    fn test_detect_downbeat_empty() {
        let aligner = BeatAligner::new(make_config(120.0));
        assert!(aligner.detect_downbeat(&[]).is_none());
    }

    #[test]
    fn test_detect_downbeat_finds_loudest_region() {
        let aligner = BeatAligner::new(make_config(120.0));
        // Quiet signal with a loud burst at sample 4800
        let mut samples = vec![0.01f32; 9600];
        for i in 4800..5280 {
            samples[i] = 1.0;
        }
        let idx = aligner
            .detect_downbeat(&samples)
            .expect("idx should be valid");
        // Should be somewhere near 4800
        assert!(idx >= 4320 && idx <= 5280);
    }

    #[test]
    fn test_align_to_grid_empty() {
        let aligner = BeatAligner::new(make_config(120.0));
        assert!(aligner.align_to_grid(&[]).is_none());
    }

    #[test]
    fn test_align_to_grid_returns_result() {
        let aligner = BeatAligner::new(make_config(120.0));
        // Non-trivial signal with energy at sample 0
        let mut samples = vec![0.0f32; 48_000];
        for s in &mut samples[0..480] {
            *s = 1.0;
        }
        let result = aligner.align_to_grid(&samples);
        // Should produce a result (or None if offset exceeds tolerance)
        // — we just verify it doesn't panic.
        let _ = result;
    }

    #[test]
    fn test_aligner_config_accessor() {
        let cfg = make_config(100.0);
        let aligner = BeatAligner::new(cfg);
        assert!((aligner.config().grid.bpm - 100.0).abs() < 1e-9);
    }
}
