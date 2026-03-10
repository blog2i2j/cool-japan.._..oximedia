//! Music-synchronized automated editing.
//!
//! Generates cut points aligned to a beat grid and paces clips to match
//! the energy of the music.

#![allow(dead_code)]

use crate::narrative::ClipInfo;

/// A grid of beat positions extracted from a music track.
#[derive(Debug, Clone)]
pub struct BeatGrid {
    /// Beats per minute of the track.
    pub bpm: f32,
    /// Timestamps (ms) of downbeats (beat 1 of each bar).
    pub downbeats_ms: Vec<u64>,
    /// Timestamps (ms) of every beat.
    pub beats_ms: Vec<u64>,
}

impl BeatGrid {
    /// Create a beat grid from BPM and downbeat positions.
    pub fn new(bpm: f32, downbeats_ms: Vec<u64>, beats_ms: Vec<u64>) -> Self {
        Self {
            bpm,
            downbeats_ms,
            beats_ms,
        }
    }

    /// Build a simple beat grid by interpolation from BPM alone.
    ///
    /// `duration_ms` is the total track length. `beats_per_bar` is usually 4.
    pub fn from_bpm(bpm: f32, duration_ms: u64, beats_per_bar: u32) -> Self {
        if bpm <= 0.0 || duration_ms == 0 {
            return Self::new(bpm, vec![], vec![]);
        }

        let beat_interval_ms = (60_000.0 / bpm) as u64;
        let mut beats_ms = Vec::new();
        let mut t = 0u64;
        while t < duration_ms {
            beats_ms.push(t);
            t += beat_interval_ms;
        }

        let downbeats_ms = beats_ms
            .iter()
            .enumerate()
            .filter(|(i, _)| i % beats_per_bar as usize == 0)
            .map(|(_, &t)| t)
            .collect();

        Self::new(bpm, downbeats_ms, beats_ms)
    }

    /// Return the timestamp of the beat nearest to `time_ms`.
    pub fn nearest_beat(&self, time_ms: u64) -> u64 {
        if self.beats_ms.is_empty() {
            return time_ms;
        }
        self.beats_ms
            .iter()
            .min_by_key(|&&b| b.abs_diff(time_ms))
            .copied()
            .unwrap_or(time_ms)
    }

    /// Return the number of bars between two timestamps.
    pub fn bars_between(&self, start_ms: u64, end_ms: u64) -> f32 {
        if self.bpm <= 0.0 {
            return 0.0;
        }
        let beat_interval_ms = 60_000.0 / self.bpm;
        let beats_per_bar = 4.0_f32;
        let duration_ms = end_ms.saturating_sub(start_ms) as f32;
        duration_ms / (beat_interval_ms * beats_per_bar)
    }
}

/// A candidate cut point aligned to the music.
#[derive(Debug, Clone)]
pub struct CutPoint {
    /// Timestamp of this cut in milliseconds.
    pub time_ms: u64,
    /// True if this cut lands on a bar downbeat.
    pub is_downbeat: bool,
    /// Energy level at this point in the music (0.0–1.0).
    pub energy: f32,
}

impl CutPoint {
    /// Create a new cut point.
    pub fn new(time_ms: u64, is_downbeat: bool, energy: f32) -> Self {
        Self {
            time_ms,
            is_downbeat,
            energy,
        }
    }
}

/// Configuration for music-synchronised editing.
#[derive(Debug, Clone)]
pub struct MusicSyncConfig {
    /// If true, snap all cuts to the nearest beat.
    pub snap_to_beat: bool,
    /// If true, prefer cuts on downbeats.
    pub cut_on_downbeat: bool,
    /// Maximum clip length expressed in bars.
    pub max_clip_duration_beats: f32,
    /// If true, match clip energy to beat energy.
    pub energy_match: bool,
}

impl Default for MusicSyncConfig {
    fn default() -> Self {
        Self {
            snap_to_beat: true,
            cut_on_downbeat: true,
            max_clip_duration_beats: 4.0,
            energy_match: true,
        }
    }
}

/// Generates music-aligned cut points from a beat grid.
pub struct MusicSyncEditor;

impl MusicSyncEditor {
    /// Generate a list of cut points for a piece of music.
    ///
    /// `cuts_per_bar` controls the cutting density: 1.0 = one cut per bar,
    /// 2.0 = one cut per two beats, 0.5 = one cut per two bars, etc.
    pub fn generate_cut_points(
        grid: &BeatGrid,
        duration_ms: u64,
        cuts_per_bar: f32,
    ) -> Vec<CutPoint> {
        if grid.beats_ms.is_empty() || cuts_per_bar <= 0.0 {
            return Vec::new();
        }

        // Determine the step size in beats (4 beats per bar)
        let step_beats = (4.0 / cuts_per_bar).max(1.0) as usize;

        let downbeat_set: std::collections::HashSet<u64> =
            grid.downbeats_ms.iter().copied().collect();

        grid.beats_ms
            .iter()
            .enumerate()
            .filter(|(i, &t)| i % step_beats == 0 && t < duration_ms)
            .map(|(_, &t)| {
                let is_downbeat = downbeat_set.contains(&t);
                // Simple ramp: energy increases towards the middle of the track
                let progress = t as f32 / duration_ms.max(1) as f32;
                let energy = 1.0 - (progress - 0.5).abs() * 2.0; // peaks at 0.5
                CutPoint::new(t, is_downbeat, energy.clamp(0.0, 1.0))
            })
            .collect()
    }
}

/// Paces clips against a list of cut points, matching energy.
pub struct ClipPacer;

impl ClipPacer {
    /// Assign clips to cut points, returning `(start_ms, clip)` pairs.
    ///
    /// High-energy cut points receive high-energy clips where possible.
    pub fn pace_clips(clips: &[ClipInfo], cut_points: &[CutPoint]) -> Vec<(u64, ClipInfo)> {
        if clips.is_empty() || cut_points.is_empty() {
            return Vec::new();
        }

        // Sort clips by energy descending so we can pair them with high-energy beats
        let mut sorted_clips: Vec<&ClipInfo> = clips.iter().collect();
        sorted_clips.sort_by(|a, b| {
            b.energy
                .partial_cmp(&a.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Sort cut points by energy descending as well
        let mut indexed_cuts: Vec<(usize, &CutPoint)> = cut_points.iter().enumerate().collect();
        indexed_cuts.sort_by(|a, b| {
            b.1.energy
                .partial_cmp(&a.1.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign clips to cut-point slots by energy rank
        let mut assignments: Vec<(usize, usize)> = Vec::new(); // (cut_idx, clip_rank)
        let n = sorted_clips.len().min(indexed_cuts.len());
        for rank in 0..n {
            assignments.push((indexed_cuts[rank].0, rank));
        }
        // Sort assignments back by cut index for chronological output
        assignments.sort_by_key(|(cut_idx, _)| *cut_idx);

        assignments
            .into_iter()
            .map(|(cut_idx, clip_rank)| {
                let start_ms = cut_points[cut_idx].time_ms;
                let clip = sorted_clips[clip_rank].clone();
                (start_ms, clip)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip(id: u64, energy: f32) -> ClipInfo {
        ClipInfo::new(id, 3.0, "action", energy, 0.9)
    }

    #[test]
    fn test_beat_grid_from_bpm() {
        let grid = BeatGrid::from_bpm(120.0, 4000, 4);
        // 120 bpm = 500ms per beat; 4 seconds → beats at 0,500,1000,1500,2000,2500,3000,3500
        assert_eq!(grid.beats_ms.len(), 8);
        assert_eq!(grid.beats_ms[0], 0);
        assert_eq!(grid.beats_ms[1], 500);
    }

    #[test]
    fn test_downbeats_extracted() {
        let grid = BeatGrid::from_bpm(120.0, 4000, 4);
        // beats every 500ms, downbeats every 4 beats = every 2000ms → 2 downbeats (0ms, 2000ms)
        assert_eq!(grid.downbeats_ms.len(), 2);
        assert!(grid.downbeats_ms.contains(&0));
        assert!(grid.downbeats_ms.contains(&2000));
    }

    #[test]
    fn test_nearest_beat() {
        let grid = BeatGrid::from_bpm(120.0, 4000, 4);
        assert_eq!(grid.nearest_beat(0), 0);
        assert_eq!(grid.nearest_beat(300), 500);
        assert_eq!(grid.nearest_beat(200), 0);
    }

    #[test]
    fn test_bars_between() {
        let grid = BeatGrid::from_bpm(120.0, 10_000, 4);
        // 120bpm: beat_interval=500ms, bar=2000ms; 4000ms = 2 bars
        let bars = grid.bars_between(0, 4000);
        assert!((bars - 2.0).abs() < 0.05, "Expected ~2 bars, got {bars}");
    }

    #[test]
    fn test_generate_cut_points_count() {
        let grid = BeatGrid::from_bpm(120.0, 8000, 4);
        // 120bpm = 500ms/beat; 8000ms → beats at 0,500,...,7500 → 16 beats
        // cuts_per_bar=1 → step=4 beats → cuts at beat indices 0,4,8,12 → 4 cuts
        let cuts = MusicSyncEditor::generate_cut_points(&grid, 8000, 1.0);
        assert_eq!(cuts.len(), 4);
    }

    #[test]
    fn test_generate_cut_points_downbeat_flag() {
        let grid = BeatGrid::from_bpm(120.0, 8000, 4);
        let cuts = MusicSyncEditor::generate_cut_points(&grid, 8000, 1.0);
        // All cuts at step=4 should be downbeats (every 4 beats = bar start)
        for cut in &cuts {
            assert!(cut.is_downbeat, "Expected downbeat at {}", cut.time_ms);
        }
    }

    #[test]
    fn test_generate_cut_points_empty_grid() {
        let grid = BeatGrid::new(120.0, vec![], vec![]);
        let cuts = MusicSyncEditor::generate_cut_points(&grid, 8000, 1.0);
        assert!(cuts.is_empty());
    }

    #[test]
    fn test_clip_pacer_basic() {
        let grid = BeatGrid::from_bpm(120.0, 8000, 4);
        let cuts = MusicSyncEditor::generate_cut_points(&grid, 8000, 2.0);
        let clips = vec![make_clip(1, 0.9), make_clip(2, 0.4)];
        let paced = ClipPacer::pace_clips(&clips, &cuts);
        assert_eq!(paced.len(), 2);
        // First result should be chronologically first cut
        assert!(paced[0].0 <= paced[1].0);
    }

    #[test]
    fn test_clip_pacer_empty_clips() {
        let cuts = vec![CutPoint::new(0, true, 0.8)];
        let paced = ClipPacer::pace_clips(&[], &cuts);
        assert!(paced.is_empty());
    }

    #[test]
    fn test_clip_pacer_empty_cuts() {
        let clips = vec![make_clip(1, 0.8)];
        let paced = ClipPacer::pace_clips(&clips, &[]);
        assert!(paced.is_empty());
    }

    #[test]
    fn test_cut_point_energy_range() {
        let grid = BeatGrid::from_bpm(120.0, 10_000, 4);
        let cuts = MusicSyncEditor::generate_cut_points(&grid, 10_000, 1.0);
        for cut in &cuts {
            assert!(cut.energy >= 0.0 && cut.energy <= 1.0);
        }
    }
}
