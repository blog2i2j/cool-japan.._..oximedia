//! Rhythm-synchronized edit point detection for music-driven video cutting.
//!
//! Provides a [`BeatGrid`] derived from tempo analysis, beat-division utilities,
//! and a [`RhythmCutter`] that quantizes arbitrary cut points to the nearest beat
//! subdivision or generates evenly spaced on-beat cuts automatically.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// BeatDivision
// ---------------------------------------------------------------------------

/// A rhythmic subdivision of a single beat (quarter note = 1 beat).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BeatDivision {
    /// One bar (4 beats in 4/4 time).
    Whole,
    /// One half-note (2 beats).
    Half,
    /// One quarter-note (1 beat).
    Quarter,
    /// One eighth-note (½ beat).
    Eighth,
    /// One sixteenth-note (¼ beat).
    Sixteenth,
}

impl BeatDivision {
    /// Number of samples for this division at the given BPM and sample rate.
    ///
    /// Assumes 4/4 time where `bpm` counts quarter notes.
    #[must_use]
    pub fn samples_per_division(self, bpm: f32, sample_rate: u32) -> u64 {
        if bpm <= 0.0 || sample_rate == 0 {
            return 0;
        }
        // samples per quarter note
        let spq = (60.0 / bpm) * sample_rate as f32;
        let factor = match self {
            BeatDivision::Whole => 4.0_f32,
            BeatDivision::Half => 2.0,
            BeatDivision::Quarter => 1.0,
            BeatDivision::Eighth => 0.5,
            BeatDivision::Sixteenth => 0.25,
        };
        (spq * factor).round() as u64
    }

    /// Number of divisions per beat (quarter note).
    #[must_use]
    pub fn per_beat(self) -> u32 {
        match self {
            BeatDivision::Whole => 1,     // 1 whole note spans 4 beats
            BeatDivision::Half => 1,      // 1 half note spans 2 beats
            BeatDivision::Quarter => 1,   // exactly 1 beat
            BeatDivision::Eighth => 2,    // 2 eighths per beat
            BeatDivision::Sixteenth => 4, // 4 sixteenths per beat
        }
    }
}

// ---------------------------------------------------------------------------
// BeatGrid
// ---------------------------------------------------------------------------

/// A regular beat grid derived from a known BPM and first-beat position.
#[derive(Debug, Clone)]
pub struct BeatGrid {
    /// Tempo in beats per minute (quarter notes per minute in 4/4).
    pub bpm: f32,
    /// Sample index of the first beat.
    pub first_beat_sample: u64,
    /// Sample rate of the audio.
    pub sample_rate: u32,
}

impl BeatGrid {
    /// Create a new beat grid.
    #[must_use]
    pub fn new(bpm: f32, first_beat_sample: u64, sample_rate: u32) -> Self {
        Self {
            bpm,
            first_beat_sample,
            sample_rate,
        }
    }

    /// Samples per quarter-note beat.
    #[must_use]
    pub fn samples_per_beat(&self) -> u64 {
        if self.bpm <= 0.0 || self.sample_rate == 0 {
            return 0;
        }
        ((60.0 / self.bpm) * self.sample_rate as f32).round() as u64
    }

    /// Infinite iterator of beat positions (in samples), starting from `first_beat_sample`.
    ///
    /// Callers should use `.take(n)` or a similar combinator to limit output.
    #[must_use]
    pub fn beat_samples(&self) -> BeatSampleIter {
        BeatSampleIter {
            current: self.first_beat_sample,
            step: self.samples_per_beat(),
        }
    }

    /// Return the sample position of the nearest beat to `sample`.
    #[must_use]
    pub fn nearest_beat(&self, sample: u64) -> u64 {
        let spb = self.samples_per_beat();
        if spb == 0 {
            return self.first_beat_sample;
        }
        // Number of whole beats elapsed since first beat.
        if sample < self.first_beat_sample {
            return self.first_beat_sample;
        }
        let offset = sample - self.first_beat_sample;
        let beat_idx = offset / spb;
        let beat_a = self.first_beat_sample + beat_idx * spb;
        let beat_b = beat_a + spb;
        // Pick whichever is closer.
        let dist_a = sample - beat_a;
        let dist_b = beat_b - sample;
        if dist_a <= dist_b {
            beat_a
        } else {
            beat_b
        }
    }

    /// Quantize `sample` to the nearest grid point at `division` granularity.
    ///
    /// For example, with `division = BeatDivision::Eighth`, the grid has twice
    /// as many points as beats.
    #[must_use]
    pub fn quantize_to_grid(&self, sample: u64, division: BeatDivision) -> u64 {
        let spd = division.samples_per_division(self.bpm, self.sample_rate);
        if spd == 0 {
            return self.first_beat_sample;
        }
        if sample < self.first_beat_sample {
            return self.first_beat_sample;
        }
        let offset = sample - self.first_beat_sample;
        let div_idx = offset / spd;
        let grid_a = self.first_beat_sample + div_idx * spd;
        let grid_b = grid_a + spd;
        let dist_a = sample - grid_a;
        let dist_b = grid_b - sample;
        if dist_a <= dist_b {
            grid_a
        } else {
            grid_b
        }
    }
}

/// An iterator yielding beat positions in samples.
#[derive(Debug, Clone)]
pub struct BeatSampleIter {
    current: u64,
    step: u64,
}

impl Iterator for BeatSampleIter {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        let val = self.current;
        self.current = self.current.saturating_add(self.step);
        Some(val)
    }
}

// ---------------------------------------------------------------------------
// RhythmCutter
// ---------------------------------------------------------------------------

/// Generates and quantizes cut points onto a beat grid.
#[derive(Debug, Clone)]
pub struct RhythmCutter {
    /// The underlying beat grid.
    pub grid: BeatGrid,
}

impl RhythmCutter {
    /// Create a new rhythm cutter with the given beat grid.
    #[must_use]
    pub fn new(grid: BeatGrid) -> Self {
        Self { grid }
    }

    /// Create a rhythm cutter from BPM and sample rate, assuming the first beat
    /// is at sample 0.
    #[must_use]
    pub fn from_bpm(bpm: f32, sample_rate: u32) -> Self {
        Self::new(BeatGrid::new(bpm, 0, sample_rate))
    }

    /// Snap each cut point to the nearest beat in the grid.
    #[must_use]
    pub fn snap_cut_points(&self, cuts: &[u64]) -> Vec<u64> {
        cuts.iter().map(|&s| self.grid.nearest_beat(s)).collect()
    }

    /// Generate `cuts_per_bar` evenly spaced cut points per bar (4 beats),
    /// spanning from the first beat up to `duration_samples`.
    ///
    /// For example, `cuts_per_bar = 1` places one cut at the start of every bar;
    /// `cuts_per_bar = 2` places cuts at beats 1 and 3 of every bar.
    #[must_use]
    pub fn generate_cuts(&self, duration_samples: u64, cuts_per_bar: u8) -> Vec<u64> {
        if cuts_per_bar == 0 || duration_samples == 0 {
            return Vec::new();
        }
        let spb = self.grid.samples_per_beat();
        if spb == 0 {
            return Vec::new();
        }
        // Spacing between cuts in samples.
        // One bar = 4 beats. Cuts are distributed evenly within a bar.
        let samples_per_bar = spb * 4;
        let step = samples_per_bar / cuts_per_bar as u64;
        if step == 0 {
            return Vec::new();
        }

        let first = self.grid.first_beat_sample;
        let mut cuts = Vec::new();
        let mut pos = first;
        while pos < first + duration_samples {
            cuts.push(pos);
            pos = pos.saturating_add(step);
        }
        cuts
    }

    /// Generate cut points that fall on downbeats only (beat 1 of each bar),
    /// every `bars_per_cut` bars.
    ///
    /// For example, `bars_per_cut = 2` places a cut every 2 bars.
    #[must_use]
    pub fn downbeat_cuts(&self, duration_samples: u64, bars_per_cut: u8) -> Vec<u64> {
        if bars_per_cut == 0 || duration_samples == 0 {
            return Vec::new();
        }
        let spb = self.grid.samples_per_beat();
        if spb == 0 {
            return Vec::new();
        }
        let samples_per_bar = spb * 4;
        let step = samples_per_bar * bars_per_cut as u64;
        if step == 0 {
            return Vec::new();
        }

        let first = self.grid.first_beat_sample;
        let mut cuts = Vec::new();
        let mut pos = first;
        while pos < first + duration_samples {
            cuts.push(pos);
            pos = pos.saturating_add(step);
        }
        cuts
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ----------------------------------------------------------------

    fn grid_120bpm() -> BeatGrid {
        // 120 BPM @ 48 kHz → 24 000 samples per beat
        BeatGrid::new(120.0, 0, 48_000)
    }

    fn cutter_120bpm() -> RhythmCutter {
        RhythmCutter::new(grid_120bpm())
    }

    // BeatDivision tests -----------------------------------------------------

    #[test]
    fn test_quarter_note_samples() {
        // 120 BPM, 48 kHz → 0.5 s/beat → 24 000 samples
        let spd = BeatDivision::Quarter.samples_per_division(120.0, 48_000);
        assert_eq!(spd, 24_000);
    }

    #[test]
    fn test_eighth_note_samples() {
        let spd = BeatDivision::Eighth.samples_per_division(120.0, 48_000);
        assert_eq!(spd, 12_000);
    }

    #[test]
    fn test_whole_note_samples() {
        // Whole note = 4 beats at 120 BPM → 2 s → 96 000 samples
        let spd = BeatDivision::Whole.samples_per_division(120.0, 48_000);
        assert_eq!(spd, 96_000);
    }

    #[test]
    fn test_sixteenth_note_samples() {
        // 1/16 = 6 000 samples at 120 BPM / 48 kHz
        let spd = BeatDivision::Sixteenth.samples_per_division(120.0, 48_000);
        assert_eq!(spd, 6_000);
    }

    #[test]
    fn test_zero_bpm_returns_zero() {
        let spd = BeatDivision::Quarter.samples_per_division(0.0, 48_000);
        assert_eq!(spd, 0);
    }

    // BeatGrid tests ---------------------------------------------------------

    #[test]
    fn test_samples_per_beat_120bpm() {
        let g = grid_120bpm();
        assert_eq!(g.samples_per_beat(), 24_000);
    }

    #[test]
    fn test_beat_samples_iterator_steps() {
        let g = grid_120bpm();
        let beats: Vec<u64> = g.beat_samples().take(5).collect();
        assert_eq!(beats, vec![0, 24_000, 48_000, 72_000, 96_000]);
    }

    #[test]
    fn test_nearest_beat_on_beat() {
        let g = grid_120bpm();
        // Exactly on beat 2
        assert_eq!(g.nearest_beat(24_000), 24_000);
    }

    #[test]
    fn test_nearest_beat_between_beats_rounds_to_nearest() {
        let g = grid_120bpm();
        // 11 999 samples past beat 0 → closer to beat 0 (24 000 / 2 = 12 000 midpoint)
        assert_eq!(g.nearest_beat(11_999), 0);
        // 12 001 samples past beat 0 → closer to beat 1
        assert_eq!(g.nearest_beat(12_001), 24_000);
    }

    #[test]
    fn test_nearest_beat_before_first_beat() {
        let g = BeatGrid::new(120.0, 10_000, 48_000);
        // Sample before first beat → returns first beat
        assert_eq!(g.nearest_beat(5_000), 10_000);
    }

    #[test]
    fn test_quantize_to_eighth() {
        let g = grid_120bpm();
        // Eighth note = 12 000 samples. 7 000 < 12 000/2 = 6 000 → rounds to 12 000
        assert_eq!(g.quantize_to_grid(7_000, BeatDivision::Eighth), 12_000);
    }

    #[test]
    fn test_quantize_to_quarter_on_grid() {
        let g = grid_120bpm();
        assert_eq!(g.quantize_to_grid(24_000, BeatDivision::Quarter), 24_000);
    }

    // RhythmCutter tests -----------------------------------------------------

    #[test]
    fn test_snap_cut_points() {
        let cutter = cutter_120bpm();
        let cuts = vec![500, 23_000, 48_500];
        let snapped = cutter.snap_cut_points(&cuts);
        assert_eq!(snapped[0], 0); // 500 → beat 0
        assert_eq!(snapped[1], 24_000); // 23 000 → beat 1
        assert_eq!(snapped[2], 48_000); // 48 500 → beat 2
    }

    #[test]
    fn test_generate_cuts_one_per_bar() {
        let cutter = cutter_120bpm();
        // 4 bars = 4 * 4 * 24 000 = 384 000 samples
        let cuts = cutter.generate_cuts(384_000, 1);
        assert_eq!(cuts.len(), 4);
        // First cut at sample 0, then every bar (96 000 samples)
        assert_eq!(cuts[0], 0);
        assert_eq!(cuts[1], 96_000);
    }

    #[test]
    fn test_generate_cuts_two_per_bar() {
        let cutter = cutter_120bpm();
        // 2 bars: 2 cuts per bar → 4 cuts total
        let cuts = cutter.generate_cuts(192_000, 2);
        assert_eq!(cuts.len(), 4);
        // Step = 48 000 (half-bar at 120 BPM / 48 kHz)
        assert_eq!(cuts[1] - cuts[0], 48_000);
    }

    #[test]
    fn test_downbeat_cuts_every_bar() {
        let cutter = cutter_120bpm();
        // 3 bars of audio, 1 cut per bar → 3 downbeat cuts
        let duration = 3 * 96_000_u64;
        let cuts = cutter.downbeat_cuts(duration, 1);
        assert_eq!(cuts.len(), 3);
        assert_eq!(cuts[0], 0);
        assert_eq!(cuts[1], 96_000);
        assert_eq!(cuts[2], 192_000);
    }

    #[test]
    fn test_downbeat_cuts_every_two_bars() {
        let cutter = cutter_120bpm();
        // 4 bars of audio, cut every 2 bars → 2 downbeat cuts
        let duration = 4 * 96_000_u64;
        let cuts = cutter.downbeat_cuts(duration, 2);
        assert_eq!(cuts.len(), 2);
        assert_eq!(cuts[0], 0);
        assert_eq!(cuts[1], 192_000);
    }

    #[test]
    fn test_generate_cuts_zero_per_bar_returns_empty() {
        let cutter = cutter_120bpm();
        let cuts = cutter.generate_cuts(100_000, 0);
        assert!(cuts.is_empty());
    }

    #[test]
    fn test_downbeat_cuts_zero_bars_per_cut_returns_empty() {
        let cutter = cutter_120bpm();
        let cuts = cutter.downbeat_cuts(100_000, 0);
        assert!(cuts.is_empty());
    }

    #[test]
    fn test_snap_empty_cuts() {
        let cutter = cutter_120bpm();
        assert!(cutter.snap_cut_points(&[]).is_empty());
    }

    #[test]
    fn test_beat_grid_with_nonzero_first_beat() {
        let g = BeatGrid::new(60.0, 5_000, 48_000); // 60 BPM → 48 000 spb
        let beats: Vec<u64> = g.beat_samples().take(3).collect();
        assert_eq!(beats[0], 5_000);
        assert_eq!(beats[1], 53_000);
        assert_eq!(beats[2], 101_000);
    }
}
