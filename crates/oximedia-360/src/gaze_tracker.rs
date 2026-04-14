//! Gaze and attention tracking for 360° video.
//!
//! This module provides tools for recording and analysing viewer head-pose
//! history in 360° video sessions, aggregating attention hotspots on the
//! sphere, and generating equirectangular attention heatmaps.
//!
//! ## Overview
//!
//! * [`GazeRecord`] — a single time-stamped gaze observation (yaw, pitch).
//! * [`GazeHistory`] — a bounded ring-buffer of gaze records with basic
//!   statistics (dwell time, gaze velocity, angular jerk).
//! * [`HotspotAccumulator`] — aggregates gaze hits into a discrete spherical
//!   grid and reports the top-N attention hotspots.
//! * [`AttentionHeatmap`] — renders a full equirectangular heatmap (as a
//!   `Vec<f32>` of normalised attention values) from a history of gaze records.
//! * [`GazeDwellEvent`] — emitted when gaze dwells within a region for longer
//!   than a configurable threshold.
//!
//! ## Coordinate convention
//!
//! All angles are in **radians**.  Yaw is the azimuthal angle, wrapped to
//! `[−π, +π]`.  Pitch is the elevation angle, clamped to `[−π/2, +π/2]`.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_360::gaze_tracker::{GazeRecord, GazeHistory, HotspotAccumulator};
//!
//! let mut history = GazeHistory::new(256);
//! history.push(GazeRecord { timestamp_s: 0.0, yaw_rad: 0.1, pitch_rad: 0.05 });
//! history.push(GazeRecord { timestamp_s: 0.1, yaw_rad: 0.15, pitch_rad: 0.05 });
//!
//! let mut acc = HotspotAccumulator::new(36, 18);
//! for record in history.records() {
//!     acc.record(record.yaw_rad, record.pitch_rad);
//! }
//! let hotspots = acc.top_hotspots(3);
//! println!("top hotspot: {:?}", hotspots.first());
//! ```

use crate::VrError;
use std::collections::VecDeque;
use std::f32::consts::PI;

// ─── GazeRecord ───────────────────────────────────────────────────────────────

/// A single gaze observation at a point in time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GazeRecord {
    /// Monotonically increasing timestamp, in seconds.
    pub timestamp_s: f64,
    /// Yaw (azimuth) in radians, in `[−π, +π]`.
    pub yaw_rad: f32,
    /// Pitch (elevation) in radians, in `[−π/2, +π/2]`.
    pub pitch_rad: f32,
}

// ─── GazeDwellEvent ───────────────────────────────────────────────────────────

/// Fired when the viewer's gaze stays within an angular region for at least
/// `min_dwell_s` seconds continuously.
#[derive(Debug, Clone, PartialEq)]
pub struct GazeDwellEvent {
    /// Centre yaw of the dwell region (radians).
    pub centre_yaw_rad: f32,
    /// Centre pitch of the dwell region (radians).
    pub centre_pitch_rad: f32,
    /// Total dwell duration in seconds.
    pub dwell_s: f64,
    /// Timestamp when the dwell event ended (or the latest sample time).
    pub end_time_s: f64,
}

// ─── GazeHistory ──────────────────────────────────────────────────────────────

/// A bounded ring-buffer of gaze records with derived statistics.
#[derive(Debug, Clone)]
pub struct GazeHistory {
    records: VecDeque<GazeRecord>,
    capacity: usize,
}

impl GazeHistory {
    /// Create a new gaze history with the given maximum capacity.
    ///
    /// `capacity` is clamped to at least 1.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            records: VecDeque::with_capacity(cap),
            capacity: cap,
        }
    }

    /// Append a gaze record.  If the buffer is full the oldest record is
    /// evicted.
    pub fn push(&mut self, record: GazeRecord) {
        if self.records.len() >= self.capacity {
            self.records.pop_front();
        }
        self.records.push_back(record);
    }

    /// Return an iterator over all stored records (oldest first).
    pub fn records(&self) -> impl Iterator<Item = &GazeRecord> {
        self.records.iter()
    }

    /// Number of records currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Return `true` if no records are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Clear all records.
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Compute the mean gaze position (yaw, pitch) over all stored records.
    ///
    /// Yaw is averaged using circular mean to handle the `±π` wrap-around.
    /// Returns `None` if the history is empty.
    #[must_use]
    pub fn mean_gaze(&self) -> Option<(f32, f32)> {
        if self.records.is_empty() {
            return None;
        }
        let n = self.records.len() as f32;
        // Circular mean for yaw
        let (sin_sum, cos_sum) = self.records.iter().fold((0.0f32, 0.0f32), |(s, c), r| {
            (s + r.yaw_rad.sin(), c + r.yaw_rad.cos())
        });
        let mean_yaw = sin_sum.atan2(cos_sum);
        let mean_pitch = self.records.iter().map(|r| r.pitch_rad).sum::<f32>() / n;
        Some((mean_yaw, mean_pitch))
    }

    /// Compute the instantaneous angular velocity (rad/s) between the last two
    /// records.
    ///
    /// Returns `(0.0, 0.0)` if fewer than 2 records are present or if the time
    /// delta is negligibly small.
    #[must_use]
    pub fn last_velocity(&self) -> (f32, f32) {
        if self.records.len() < 2 {
            return (0.0, 0.0);
        }
        let n = self.records.len();
        let prev = &self.records[n - 2];
        let curr = &self.records[n - 1];
        let dt = (curr.timestamp_s - prev.timestamp_s) as f32;
        if dt.abs() < f32::EPSILON {
            return (0.0, 0.0);
        }
        let dyaw = wrap_angle(curr.yaw_rad - prev.yaw_rad);
        let dpitch = curr.pitch_rad - prev.pitch_rad;
        (dyaw / dt, dpitch / dt)
    }

    /// Compute the mean angular speed (scalar, rad/s) over all inter-sample
    /// intervals in the history.
    ///
    /// Returns `0.0` if fewer than 2 records are stored.
    #[must_use]
    pub fn mean_angular_speed(&self) -> f32 {
        let n = self.records.len();
        if n < 2 {
            return 0.0;
        }
        let mut total_speed = 0.0f32;
        let mut valid = 0usize;
        for i in 1..n {
            let prev = &self.records[i - 1];
            let curr = &self.records[i];
            let dt = (curr.timestamp_s - prev.timestamp_s) as f32;
            if dt.abs() < f32::EPSILON {
                continue;
            }
            let dyaw = wrap_angle(curr.yaw_rad - prev.yaw_rad);
            let dpitch = curr.pitch_rad - prev.pitch_rad;
            let speed = (dyaw * dyaw + dpitch * dpitch).sqrt() / dt;
            total_speed += speed;
            valid += 1;
        }
        if valid == 0 {
            0.0
        } else {
            total_speed / valid as f32
        }
    }

    /// Scan the history and emit [`GazeDwellEvent`]s for contiguous intervals
    /// where the gaze stays within `radius_rad` of a central point for at
    /// least `min_dwell_s` seconds.
    ///
    /// The algorithm uses a greedy sliding window: when the gaze moves outside
    /// the radius of the *current dwell centre*, the accumulated dwell interval
    /// is evaluated and (if long enough) emitted, then a new interval begins.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidCoordinate`] if `radius_rad` ≤ 0 or
    /// `min_dwell_s` ≤ 0.
    pub fn dwell_events(
        &self,
        radius_rad: f32,
        min_dwell_s: f64,
    ) -> Result<Vec<GazeDwellEvent>, VrError> {
        if radius_rad <= 0.0 || min_dwell_s <= 0.0 {
            return Err(VrError::InvalidCoordinate);
        }
        if self.records.len() < 2 {
            return Ok(Vec::new());
        }

        let mut events = Vec::new();
        let mut window_start = 0usize;

        while window_start < self.records.len() {
            let anchor = &self.records[window_start];
            let mut window_end = window_start;

            // Extend window as long as gaze stays within radius of anchor
            for j in (window_start + 1)..self.records.len() {
                let r = &self.records[j];
                if angular_distance(anchor.yaw_rad, anchor.pitch_rad, r.yaw_rad, r.pitch_rad)
                    <= radius_rad
                {
                    window_end = j;
                } else {
                    break;
                }
            }

            if window_end > window_start {
                let t_start = self.records[window_start].timestamp_s;
                let t_end = self.records[window_end].timestamp_s;
                let dwell_s = t_end - t_start;
                if dwell_s >= min_dwell_s {
                    // Compute centroid of dwell cluster
                    let (sin_sum, cos_sum) =
                        (window_start..=window_end).fold((0.0f32, 0.0f32), |(s, c), i| {
                            (
                                s + self.records[i].yaw_rad.sin(),
                                c + self.records[i].yaw_rad.cos(),
                            )
                        });
                    let count = (window_end - window_start + 1) as f32;
                    let centre_yaw = sin_sum.atan2(cos_sum);
                    let centre_pitch = (window_start..=window_end)
                        .map(|i| self.records[i].pitch_rad)
                        .sum::<f32>()
                        / count;
                    events.push(GazeDwellEvent {
                        centre_yaw_rad: centre_yaw,
                        centre_pitch_rad: centre_pitch,
                        dwell_s,
                        end_time_s: t_end,
                    });
                }
            }

            window_start = window_end + 1;
        }

        Ok(events)
    }
}

// ─── HotspotCell ──────────────────────────────────────────────────────────────

/// A single cell in the spherical attention grid.
#[derive(Debug, Clone, PartialEq)]
pub struct HotspotCell {
    /// Centre yaw of the cell (radians).
    pub yaw_rad: f32,
    /// Centre pitch of the cell (radians).
    pub pitch_rad: f32,
    /// Total number of gaze hits accumulated in this cell.
    pub hit_count: u64,
}

// ─── HotspotAccumulator ───────────────────────────────────────────────────────

/// Accumulates gaze hits into a discrete equirectangular grid and reports
/// the cells with the highest attention density.
///
/// The sphere is discretised into `yaw_bins × pitch_bins` cells of equal
/// angular size.
#[derive(Debug, Clone)]
pub struct HotspotAccumulator {
    yaw_bins: usize,
    pitch_bins: usize,
    /// Hit counters, row-major: `counts[pitch_bin * yaw_bins + yaw_bin]`.
    counts: Vec<u64>,
    /// Total number of gaze hits recorded.
    total_hits: u64,
}

impl HotspotAccumulator {
    /// Create a new accumulator with the given grid dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidDimensions`] if either dimension is zero.
    pub fn new(yaw_bins: usize, pitch_bins: usize) -> Self {
        let yb = yaw_bins.max(1);
        let pb = pitch_bins.max(1);
        Self {
            yaw_bins: yb,
            pitch_bins: pb,
            counts: vec![0u64; yb * pb],
            total_hits: 0,
        }
    }

    /// Record a single gaze hit at the given yaw/pitch position.
    pub fn record(&mut self, yaw_rad: f32, pitch_rad: f32) {
        let (yb, pb) = self.cell_index(yaw_rad, pitch_rad);
        self.counts[pb * self.yaw_bins + yb] =
            self.counts[pb * self.yaw_bins + yb].saturating_add(1);
        self.total_hits = self.total_hits.saturating_add(1);
    }

    /// Ingest all records from a [`GazeHistory`].
    pub fn ingest_history(&mut self, history: &GazeHistory) {
        for r in history.records() {
            self.record(r.yaw_rad, r.pitch_rad);
        }
    }

    /// Return the `n` cells with the highest hit count, sorted descending.
    ///
    /// If fewer than `n` non-zero cells exist, all non-zero cells are returned.
    #[must_use]
    pub fn top_hotspots(&self, n: usize) -> Vec<HotspotCell> {
        let mut cells: Vec<HotspotCell> = self
            .counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(idx, &hit_count)| {
                let yb = idx % self.yaw_bins;
                let pb = idx / self.yaw_bins;
                HotspotCell {
                    yaw_rad: self.cell_centre_yaw(yb),
                    pitch_rad: self.cell_centre_pitch(pb),
                    hit_count,
                }
            })
            .collect();
        cells.sort_by(|a, b| b.hit_count.cmp(&a.hit_count));
        cells.truncate(n);
        cells
    }

    /// Total number of gaze hits recorded.
    #[must_use]
    pub fn total_hits(&self) -> u64 {
        self.total_hits
    }

    /// Reset all counts to zero.
    pub fn reset(&mut self) {
        self.counts.iter_mut().for_each(|c| *c = 0);
        self.total_hits = 0;
    }

    /// Return the attention fraction (hits / total_hits) for the cell containing
    /// the given yaw/pitch.  Returns `0.0` if no hits have been recorded.
    #[must_use]
    pub fn attention_fraction(&self, yaw_rad: f32, pitch_rad: f32) -> f32 {
        if self.total_hits == 0 {
            return 0.0;
        }
        let (yb, pb) = self.cell_index(yaw_rad, pitch_rad);
        self.counts[pb * self.yaw_bins + yb] as f32 / self.total_hits as f32
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    fn cell_index(&self, yaw_rad: f32, pitch_rad: f32) -> (usize, usize) {
        // Normalise yaw to [0, 1) and pitch to [0, 1)
        let yaw_norm = (wrap_angle(yaw_rad) + PI) / (2.0 * PI);
        let pitch_norm = (pitch_rad.clamp(-PI / 2.0, PI / 2.0) + PI / 2.0) / PI;

        let yb = ((yaw_norm * self.yaw_bins as f32) as usize).min(self.yaw_bins - 1);
        let pb = ((pitch_norm * self.pitch_bins as f32) as usize).min(self.pitch_bins - 1);
        (yb, pb)
    }

    fn cell_centre_yaw(&self, yb: usize) -> f32 {
        let norm = (yb as f32 + 0.5) / self.yaw_bins as f32;
        norm * 2.0 * PI - PI
    }

    fn cell_centre_pitch(&self, pb: usize) -> f32 {
        let norm = (pb as f32 + 0.5) / self.pitch_bins as f32;
        norm * PI - PI / 2.0
    }
}

// ─── AttentionHeatmap ─────────────────────────────────────────────────────────

/// Renders a per-pixel attention heatmap in equirectangular layout.
///
/// Each pixel receives an attention value proportional to the Gaussian-weighted
/// contribution of nearby gaze samples.  The output is normalised to `[0, 1]`.
#[derive(Debug, Clone)]
pub struct AttentionHeatmap {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Gaussian sigma (in radians) controlling the spatial spread of each
    /// gaze sample's contribution.
    pub sigma_rad: f32,
}

impl AttentionHeatmap {
    /// Create a new heatmap renderer.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidDimensions`] if `width` or `height` is zero.
    /// Returns [`VrError::InvalidCoordinate`] if `sigma_rad` ≤ 0.
    pub fn new(width: u32, height: u32, sigma_rad: f32) -> Result<Self, VrError> {
        if width == 0 || height == 0 {
            return Err(VrError::InvalidDimensions(
                "width and height must be > 0".into(),
            ));
        }
        if sigma_rad <= 0.0 {
            return Err(VrError::InvalidCoordinate);
        }
        Ok(Self {
            width,
            height,
            sigma_rad,
        })
    }

    /// Render the heatmap from a collection of gaze records.
    ///
    /// Returns a row-major `Vec<f32>` of length `width * height`, with values
    /// normalised to `[0, 1]`.
    ///
    /// Each pixel's value is the sum of Gaussian kernels centred at each gaze
    /// sample, evaluated at the spherical direction corresponding to that pixel.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidDimensions`] if `records` is empty.
    pub fn render<'a>(
        &self,
        records: impl Iterator<Item = &'a GazeRecord>,
    ) -> Result<Vec<f32>, VrError> {
        let samples: Vec<(f32, f32)> = records.map(|r| (r.yaw_rad, r.pitch_rad)).collect();
        if samples.is_empty() {
            return Err(VrError::InvalidDimensions(
                "no gaze records provided".into(),
            ));
        }

        let w = self.width as usize;
        let h = self.height as usize;
        let inv_sigma2 = 1.0 / (2.0 * self.sigma_rad * self.sigma_rad);

        let mut map = vec![0.0f32; w * h];
        let mut max_val = 0.0f32;

        for py in 0..h {
            let pitch = ((py as f32 + 0.5) / h as f32) * PI - PI / 2.0;
            for px in 0..w {
                let yaw = ((px as f32 + 0.5) / w as f32) * 2.0 * PI - PI;
                let mut acc = 0.0f32;
                for &(gy, gp) in &samples {
                    let dist = angular_distance(yaw, pitch, gy, gp);
                    acc += (-dist * dist * inv_sigma2).exp();
                }
                map[py * w + px] = acc;
                if acc > max_val {
                    max_val = acc;
                }
            }
        }

        // Normalise
        if max_val > f32::EPSILON {
            map.iter_mut().for_each(|v| *v /= max_val);
        }

        Ok(map)
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Wrap an angle (radians) to `[−π, +π]`.
#[inline]
fn wrap_angle(a: f32) -> f32 {
    let mut a = a % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    } else if a < -PI {
        a += 2.0 * PI;
    }
    a
}

/// Compute the great-circle angular distance (radians) between two directions
/// given as (yaw, pitch) pairs, using the haversine formula.
#[inline]
pub fn angular_distance(yaw1: f32, pitch1: f32, yaw2: f32, pitch2: f32) -> f32 {
    let dpitch = pitch2 - pitch1;
    let dyaw = wrap_angle(yaw2 - yaw1);
    let hav =
        (dpitch * 0.5).sin().powi(2) + pitch1.cos() * pitch2.cos() * (dyaw * 0.5).sin().powi(2);
    2.0 * hav.sqrt().clamp(0.0, 1.0).asin()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    fn rec(t: f64, yaw: f32, pitch: f32) -> GazeRecord {
        GazeRecord {
            timestamp_s: t,
            yaw_rad: yaw,
            pitch_rad: pitch,
        }
    }

    // ── GazeHistory ──────────────────────────────────────────────────────────

    #[test]
    fn history_capacity_eviction() {
        let mut h = GazeHistory::new(3);
        for i in 0..6 {
            h.push(rec(i as f64, 0.0, 0.0));
        }
        assert_eq!(h.len(), 3);
    }

    #[test]
    fn history_mean_gaze_empty_returns_none() {
        let h = GazeHistory::new(10);
        assert!(h.mean_gaze().is_none());
    }

    #[test]
    fn history_mean_gaze_symmetric_yaw() {
        let mut h = GazeHistory::new(10);
        // Symmetric yaw samples — circular mean should be near 0
        h.push(rec(0.0, 0.2, 0.0));
        h.push(rec(0.1, -0.2, 0.0));
        let (mean_yaw, _) = h.mean_gaze().expect("non-empty");
        assert!(mean_yaw.abs() < 0.05, "mean_yaw={mean_yaw}");
    }

    #[test]
    fn history_last_velocity_single_sample_zero() {
        let mut h = GazeHistory::new(4);
        h.push(rec(0.0, 0.0, 0.0));
        let (vy, vp) = h.last_velocity();
        assert_eq!(vy, 0.0);
        assert_eq!(vp, 0.0);
    }

    #[test]
    fn history_last_velocity_correct() {
        let mut h = GazeHistory::new(4);
        // Yaw changes by 0.3 rad over 0.1 s → velocity = 3.0 rad/s
        h.push(rec(0.0, 0.0, 0.0));
        h.push(rec(0.1, 0.3, 0.0));
        let (vy, _vp) = h.last_velocity();
        assert!((vy - 3.0).abs() < 1e-4, "vy={vy}");
    }

    #[test]
    fn history_mean_angular_speed_constant_motion() {
        let mut h = GazeHistory::new(10);
        for i in 0..5 {
            h.push(rec(i as f64 * 0.1, i as f32 * 0.1, 0.0));
        }
        let speed = h.mean_angular_speed();
        assert!(speed > 0.5, "speed={speed}");
    }

    #[test]
    fn history_dwell_events_detects_dwell() {
        let mut h = GazeHistory::new(20);
        // Gaze stays within 0.1 rad of (0, 0) for 0.5 s
        for i in 0..6 {
            h.push(rec(i as f64 * 0.1, 0.01, 0.01));
        }
        let events = h.dwell_events(0.15, 0.3).expect("valid params");
        assert!(!events.is_empty(), "expected at least one dwell event");
        assert!(events[0].dwell_s >= 0.3);
    }

    #[test]
    fn history_dwell_events_rejects_bad_params() {
        let h = GazeHistory::new(10);
        assert!(h.dwell_events(-0.1, 0.5).is_err());
        assert!(h.dwell_events(0.1, -1.0).is_err());
        assert!(h.dwell_events(0.0, 0.5).is_err());
    }

    // ── HotspotAccumulator ───────────────────────────────────────────────────

    #[test]
    fn hotspot_accumulator_top_returns_sorted() {
        let mut acc = HotspotAccumulator::new(36, 18);
        // Record many hits at (0, 0)
        for _ in 0..50 {
            acc.record(0.0, 0.0);
        }
        // Record fewer hits at (PI/2, 0)
        for _ in 0..10 {
            acc.record(FRAC_PI_2, 0.0);
        }
        let top = acc.top_hotspots(2);
        assert_eq!(top.len(), 2);
        assert!(top[0].hit_count >= top[1].hit_count);
        assert_eq!(top[0].hit_count, 50);
    }

    #[test]
    fn hotspot_accumulator_total_hits() {
        let mut acc = HotspotAccumulator::new(36, 18);
        acc.record(0.0, 0.0);
        acc.record(1.0, 0.2);
        acc.record(-1.0, -0.1);
        assert_eq!(acc.total_hits(), 3);
    }

    #[test]
    fn hotspot_accumulator_reset_clears() {
        let mut acc = HotspotAccumulator::new(36, 18);
        acc.record(0.0, 0.0);
        acc.reset();
        assert_eq!(acc.total_hits(), 0);
        assert!(acc.top_hotspots(5).is_empty());
    }

    #[test]
    fn hotspot_accumulator_ingest_history() {
        let mut h = GazeHistory::new(100);
        for i in 0..10 {
            h.push(rec(i as f64, 0.0, 0.0));
        }
        let mut acc = HotspotAccumulator::new(36, 18);
        acc.ingest_history(&h);
        assert_eq!(acc.total_hits(), 10);
    }

    // ── AttentionHeatmap ─────────────────────────────────────────────────────

    #[test]
    fn heatmap_rejects_zero_dimensions() {
        assert!(AttentionHeatmap::new(0, 32, 0.1).is_err());
        assert!(AttentionHeatmap::new(64, 0, 0.1).is_err());
    }

    #[test]
    fn heatmap_rejects_zero_sigma() {
        assert!(AttentionHeatmap::new(64, 32, 0.0).is_err());
        assert!(AttentionHeatmap::new(64, 32, -0.1).is_err());
    }

    #[test]
    fn heatmap_render_outputs_correct_size() {
        let hm = AttentionHeatmap::new(64, 32, 0.2).expect("valid");
        let records = vec![rec(0.0, 0.0, 0.0)];
        let map = hm.render(records.iter()).expect("ok");
        assert_eq!(map.len(), 64 * 32);
    }

    #[test]
    fn heatmap_render_peak_at_gaze_location() {
        let width = 64u32;
        let height = 32u32;
        let hm = AttentionHeatmap::new(width, height, 0.3).expect("valid");
        // Single gaze sample at (0, 0) → front centre
        let records = vec![rec(0.0, 0.0, 0.0)];
        let map = hm.render(records.iter()).expect("ok");

        // Peak should be at pixel corresponding to (yaw=0, pitch=0)
        // which is approximately (width/2, height/2) in equirectangular
        let peak_idx = map
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("non-empty");
        let peak_y = peak_idx / width as usize;
        let peak_x = peak_idx % width as usize;
        let centre_x = width as usize / 2;
        let centre_y = height as usize / 2;
        // Allow ±3 pixel tolerance
        assert!(
            (peak_x as i32 - centre_x as i32).abs() <= 3,
            "peak_x={peak_x}"
        );
        assert!(
            (peak_y as i32 - centre_y as i32).abs() <= 3,
            "peak_y={peak_y}"
        );
    }

    #[test]
    fn heatmap_render_normalised_to_one() {
        let hm = AttentionHeatmap::new(32, 16, 0.5).expect("valid");
        let records = vec![rec(0.0, 0.0, 0.0), rec(0.1, 1.0, 0.1)];
        let map = hm.render(records.iter()).expect("ok");
        let max_val = map.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!((max_val - 1.0).abs() < 1e-5, "max_val={max_val}");
    }

    // ── angular_distance ─────────────────────────────────────────────────────

    #[test]
    fn angular_distance_same_point_zero() {
        let d = angular_distance(0.5, 0.3, 0.5, 0.3);
        assert!(d < 1e-5, "d={d}");
    }

    #[test]
    fn angular_distance_antipodal_is_pi() {
        let d = angular_distance(0.0, 0.0, PI, 0.0);
        assert!((d - PI).abs() < 0.05, "d={d}");
    }
}
