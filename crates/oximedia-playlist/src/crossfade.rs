//! Crossfade transitions: overlap duration, fade curves, and gapless playback.
//!
//! This module provides types and logic for computing precise crossfade
//! schedules between consecutive playlist items, including curve shapes,
//! overlap windows, and gapless audio continuity.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::time::Duration;

// ── fade curve ────────────────────────────────────────────────────────────────

/// Shape of the fade curve applied during a crossfade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FadeCurve {
    /// Linear fade: gain changes at a constant rate.
    Linear,
    /// Equal-power (sine/cosine) crossfade: maintains constant perceived loudness.
    EqualPower,
    /// Logarithmic fade-out (fast at start, slow at end).
    Logarithmic,
    /// Exponential fade-in (slow at start, fast at end).
    Exponential,
    /// S-curve (smooth step) for natural-sounding transitions.
    SCurve,
}

impl FadeCurve {
    /// Compute the gain multiplier for the *outgoing* track at position `t` ∈ [0, 1].
    /// `t = 0` = start of crossfade (full volume), `t = 1` = end (silence).
    #[must_use]
    pub fn fade_out_gain(self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            FadeCurve::Linear => 1.0 - t,
            FadeCurve::EqualPower => (std::f64::consts::FRAC_PI_2 * t).cos(),
            FadeCurve::Logarithmic => if t < f64::EPSILON {
                1.0
            } else {
                (1.0 - t).ln() / (-1.0_f64).exp().ln() + 1.0
            }
            .clamp(0.0, 1.0),
            FadeCurve::Exponential => (1.0 - t).powi(2),
            FadeCurve::SCurve => {
                let v = 1.0 - t;
                v * v * (3.0 - 2.0 * v)
            }
        }
    }

    /// Compute the gain multiplier for the *incoming* track at position `t` ∈ [0, 1].
    /// `t = 0` = start of crossfade (silence), `t = 1` = end (full volume).
    #[must_use]
    pub fn fade_in_gain(self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            FadeCurve::Linear => t,
            FadeCurve::EqualPower => (std::f64::consts::FRAC_PI_2 * t).sin(),
            FadeCurve::Logarithmic => t.powi(2),
            FadeCurve::Exponential => {
                if t < f64::EPSILON {
                    0.0
                } else {
                    (t.ln() / (-1.0_f64).exp().ln() + 1.0).clamp(0.0, 1.0)
                }
            }
            FadeCurve::SCurve => t * t * (3.0 - 2.0 * t),
        }
    }

    /// Sample the combined fade pair at `n` evenly-spaced points.
    /// Returns `(fade_out, fade_in)` gain vectors.
    #[must_use]
    pub fn sample_pair(self, n: usize) -> (Vec<f64>, Vec<f64>) {
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let out: Vec<f64> = (0..n)
            .map(|i| self.fade_out_gain(i as f64 / (n - 1).max(1) as f64))
            .collect();
        let inp: Vec<f64> = (0..n)
            .map(|i| self.fade_in_gain(i as f64 / (n - 1).max(1) as f64))
            .collect();
        (out, inp)
    }
}

// ── crossfade segment ─────────────────────────────────────────────────────────

/// Describes the overlap segment between two playlist items.
#[derive(Debug, Clone)]
pub struct CrossfadeSegment {
    /// Duration of the overlap.
    pub duration: Duration,
    /// Fade curve applied.
    pub curve: FadeCurve,
    /// Offset from the end of the outgoing item where the crossfade starts.
    pub start_offset_from_end: Duration,
    /// Whether to trim silence from the end of the outgoing item.
    pub trim_trailing_silence: bool,
}

impl CrossfadeSegment {
    /// Create a simple crossfade segment.
    #[must_use]
    pub const fn new(duration: Duration, curve: FadeCurve) -> Self {
        Self {
            duration,
            curve,
            start_offset_from_end: duration,
            trim_trailing_silence: false,
        }
    }

    /// Enable trailing-silence trimming.
    #[must_use]
    pub const fn with_silence_trim(mut self) -> Self {
        self.trim_trailing_silence = true;
        self
    }

    /// The offset from the *start* of the outgoing item at which crossfade begins,
    /// given that item's total duration.
    #[must_use]
    pub fn crossfade_start(&self, item_duration: Duration) -> Duration {
        if item_duration > self.start_offset_from_end {
            item_duration
                .checked_sub(self.start_offset_from_end)
                .unwrap_or(Duration::ZERO)
        } else {
            Duration::ZERO
        }
    }
}

// ── gapless sequencer ─────────────────────────────────────────────────────────

/// Entry in a gapless playlist sequence.
#[derive(Debug, Clone)]
pub struct GaplessEntry {
    /// Track identifier.
    pub track_id: u64,
    /// Track duration.
    pub duration: Duration,
    /// Crossfade segment to apply *after* this entry (before the next).
    /// `None` = hard cut.
    pub crossfade: Option<CrossfadeSegment>,
}

impl GaplessEntry {
    /// Create a gapless entry with no crossfade.
    #[must_use]
    pub const fn new(track_id: u64, duration: Duration) -> Self {
        Self {
            track_id,
            duration,
            crossfade: None,
        }
    }

    /// Set a crossfade segment.
    #[must_use]
    pub fn with_crossfade(mut self, seg: CrossfadeSegment) -> Self {
        self.crossfade = Some(seg);
        self
    }
}

/// Compute playback schedule for a gapless sequence.
/// Returns `(track_id, start_time)` pairs where `start_time` is when
/// each track should begin (relative to sequence start).
#[must_use]
pub fn schedule_gapless(entries: &[GaplessEntry]) -> Vec<(u64, Duration)> {
    let mut schedule = Vec::with_capacity(entries.len());
    let mut cursor = Duration::ZERO;
    for (i, entry) in entries.iter().enumerate() {
        schedule.push((entry.track_id, cursor));
        // Advance cursor by this entry's duration minus the next crossfade overlap.
        let overlap = if i + 1 < entries.len() {
            entry
                .crossfade
                .as_ref()
                .map_or(Duration::ZERO, |cf| cf.duration)
        } else {
            Duration::ZERO
        };
        cursor += entry.duration.saturating_sub(overlap);
    }
    schedule
}

/// Total wall-clock duration of a gapless sequence (accounting for overlaps).
#[must_use]
pub fn sequence_duration(entries: &[GaplessEntry]) -> Duration {
    if entries.is_empty() {
        return Duration::ZERO;
    }
    let mut total = Duration::ZERO;
    for (i, entry) in entries.iter().enumerate() {
        let overlap = if i + 1 < entries.len() {
            entry
                .crossfade
                .as_ref()
                .map_or(Duration::ZERO, |cf| cf.duration)
        } else {
            Duration::ZERO
        };
        total += entry.duration.saturating_sub(overlap);
    }
    // Add final entry's full duration.
    total
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_linear_fade_out_endpoints() {
        assert!(approx_eq(FadeCurve::Linear.fade_out_gain(0.0), 1.0, 1e-9));
        assert!(approx_eq(FadeCurve::Linear.fade_out_gain(1.0), 0.0, 1e-9));
    }

    #[test]
    fn test_linear_fade_in_endpoints() {
        assert!(approx_eq(FadeCurve::Linear.fade_in_gain(0.0), 0.0, 1e-9));
        assert!(approx_eq(FadeCurve::Linear.fade_in_gain(1.0), 1.0, 1e-9));
    }

    #[test]
    fn test_equal_power_energy_constant() {
        // For equal-power, fade_out^2 + fade_in^2 ≈ 1 at all points.
        let curve = FadeCurve::EqualPower;
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let out = curve.fade_out_gain(t);
            let inp = curve.fade_in_gain(t);
            assert!(
                approx_eq(out * out + inp * inp, 1.0, 1e-9),
                "t={t}, out={out}, in={inp}"
            );
        }
    }

    #[test]
    fn test_s_curve_endpoints() {
        assert!(approx_eq(FadeCurve::SCurve.fade_out_gain(0.0), 1.0, 1e-9));
        assert!(approx_eq(FadeCurve::SCurve.fade_out_gain(1.0), 0.0, 1e-9));
        assert!(approx_eq(FadeCurve::SCurve.fade_in_gain(0.0), 0.0, 1e-9));
        assert!(approx_eq(FadeCurve::SCurve.fade_in_gain(1.0), 1.0, 1e-9));
    }

    #[test]
    fn test_exponential_endpoints() {
        assert!(approx_eq(
            FadeCurve::Exponential.fade_out_gain(0.0),
            1.0,
            1e-9
        ));
        assert!(approx_eq(
            FadeCurve::Exponential.fade_out_gain(1.0),
            0.0,
            1e-9
        ));
    }

    #[test]
    fn test_fade_clamped_out_of_range() {
        let v = FadeCurve::Linear.fade_out_gain(1.5);
        assert!((0.0..=1.0).contains(&v));
        let v2 = FadeCurve::Linear.fade_in_gain(-0.5);
        assert!((0.0..=1.0).contains(&v2));
    }

    #[test]
    fn test_sample_pair_length() {
        let (out, inp) = FadeCurve::Linear.sample_pair(5);
        assert_eq!(out.len(), 5);
        assert_eq!(inp.len(), 5);
    }

    #[test]
    fn test_sample_pair_empty() {
        let (out, inp) = FadeCurve::Linear.sample_pair(0);
        assert!(out.is_empty());
        assert!(inp.is_empty());
    }

    #[test]
    fn test_crossfade_segment_start_offset() {
        let seg = CrossfadeSegment::new(Duration::from_secs(5), FadeCurve::Linear);
        let item_dur = Duration::from_secs(30);
        assert_eq!(seg.crossfade_start(item_dur), Duration::from_secs(25));
    }

    #[test]
    fn test_crossfade_segment_start_offset_short_item() {
        let seg = CrossfadeSegment::new(Duration::from_secs(10), FadeCurve::Linear);
        let item_dur = Duration::from_secs(5);
        assert_eq!(seg.crossfade_start(item_dur), Duration::ZERO);
    }

    #[test]
    fn test_schedule_gapless_no_overlap() {
        let entries = vec![
            GaplessEntry::new(1, Duration::from_secs(60)),
            GaplessEntry::new(2, Duration::from_secs(90)),
        ];
        let sched = schedule_gapless(&entries);
        assert_eq!(sched[0], (1, Duration::ZERO));
        assert_eq!(sched[1], (2, Duration::from_secs(60)));
    }

    #[test]
    fn test_schedule_gapless_with_overlap() {
        let cf = CrossfadeSegment::new(Duration::from_secs(5), FadeCurve::EqualPower);
        let entries = vec![
            GaplessEntry::new(1, Duration::from_secs(60)).with_crossfade(cf),
            GaplessEntry::new(2, Duration::from_secs(90)),
        ];
        let sched = schedule_gapless(&entries);
        assert_eq!(sched[0].1, Duration::ZERO);
        assert_eq!(sched[1].1, Duration::from_secs(55)); // 60 - 5 overlap
    }

    #[test]
    fn test_schedule_gapless_empty() {
        let sched = schedule_gapless(&[]);
        assert!(sched.is_empty());
    }

    #[test]
    fn test_sequence_duration_no_overlap() {
        let entries = vec![
            GaplessEntry::new(1, Duration::from_secs(60)),
            GaplessEntry::new(2, Duration::from_secs(30)),
        ];
        // Without overlap: 60 + 30 = 90 s.
        assert_eq!(sequence_duration(&entries), Duration::from_secs(90));
    }

    #[test]
    fn test_sequence_duration_empty() {
        assert_eq!(sequence_duration(&[]), Duration::ZERO);
    }
}
