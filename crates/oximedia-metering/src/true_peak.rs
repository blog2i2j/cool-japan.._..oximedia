//! True-peak overshoot detection and per-channel reporting (dBTP).
#![allow(dead_code)]

/// Convert a linear true-peak amplitude to dBTP.
fn linear_to_dbtp(linear: f64) -> f64 {
    if linear <= 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * linear.log10()
    }
}

/// Overshoot severity classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OvershootSeverity {
    /// No overshoot (below allowed ceiling).
    None,
    /// Minor overshoot (0 – 0.5 dBTP above ceiling).
    Minor,
    /// Moderate overshoot (0.5 – 2.0 dBTP above ceiling).
    Moderate,
    /// Severe overshoot (> 2.0 dBTP above ceiling).
    Severe,
}

/// A single true-peak overshoot event.
#[derive(Clone, Debug)]
pub struct TruePeakOvershoot {
    /// Measured true peak in dBTP.
    pub measured_dbtp: f64,
    /// Allowed ceiling in dBTP (e.g. -1.0).
    pub ceiling_dbtp: f64,
    /// Channel index (0-based).
    pub channel: usize,
}

impl TruePeakOvershoot {
    /// Create an overshoot record.
    pub fn new(measured_dbtp: f64, ceiling_dbtp: f64, channel: usize) -> Self {
        Self {
            measured_dbtp,
            ceiling_dbtp,
            channel,
        }
    }

    /// Amount by which the ceiling is exceeded (positive = over, negative = under).
    pub fn excess_dbtp(&self) -> f64 {
        self.measured_dbtp - self.ceiling_dbtp
    }

    /// Classify the overshoot severity.
    pub fn severity(&self) -> OvershootSeverity {
        let excess = self.excess_dbtp();
        if excess <= 0.0 {
            OvershootSeverity::None
        } else if excess <= 0.5 {
            OvershootSeverity::Minor
        } else if excess <= 2.0 {
            OvershootSeverity::Moderate
        } else {
            OvershootSeverity::Severe
        }
    }
}

/// True-peak meter for a fixed number of channels.
///
/// Internally this stores only the per-channel peak-linear value; callers are
/// responsible for supplying oversampled (4x) or interpolated samples.
#[derive(Debug)]
pub struct TruePeakMeter {
    /// Per-channel running peak (linear).
    channel_peaks: Vec<f64>,
    /// Ceiling in dBTP used for overshoot detection.
    ceiling_dbtp: f64,
}

impl TruePeakMeter {
    /// Create a meter for `channels` channels with the given ceiling.
    pub fn new(channels: usize, ceiling_dbtp: f64) -> Self {
        Self {
            channel_peaks: vec![0.0; channels.max(1)],
            ceiling_dbtp,
        }
    }

    /// Process a single sample on the given channel.
    pub fn process_sample(&mut self, sample: f64, channel: usize) {
        if let Some(peak) = self.channel_peaks.get_mut(channel) {
            let abs = sample.abs();
            if abs > *peak {
                *peak = abs;
            }
        }
    }

    /// Process an interleaved frame (all channels for one sample period).
    pub fn process_frame(&mut self, frame: &[f64]) {
        for (ch, &s) in frame.iter().enumerate() {
            self.process_sample(s, ch);
        }
    }

    /// True peak in dBTP across all channels (worst case).
    pub fn true_peak_dbtp(&self) -> f64 {
        let max_linear = self.channel_peaks.iter().copied().fold(0.0_f64, f64::max);
        linear_to_dbtp(max_linear)
    }

    /// True peak in dBTP for a specific channel.
    pub fn channel_peak_dbtp(&self, channel: usize) -> Option<f64> {
        self.channel_peaks.get(channel).map(|&v| linear_to_dbtp(v))
    }

    /// Return `true` if any channel exceeds the ceiling.
    pub fn has_overshoot(&self) -> bool {
        self.true_peak_dbtp() > self.ceiling_dbtp
    }

    /// Number of channels this meter was configured for.
    pub fn num_channels(&self) -> usize {
        self.channel_peaks.len()
    }

    /// Reset all peak levels to zero.
    pub fn reset(&mut self) {
        for p in &mut self.channel_peaks {
            *p = 0.0;
        }
    }
}

/// Per-channel summary used in the final report.
#[derive(Clone, Debug)]
pub struct ChannelPeakSummary {
    /// Channel index.
    pub channel: usize,
    /// True peak for this channel in dBTP.
    pub peak_dbtp: f64,
    /// Overshoot severity for this channel.
    pub severity: OvershootSeverity,
}

/// Aggregated true-peak report across all channels.
#[derive(Clone, Debug)]
pub struct TruePeakReport {
    /// Per-channel summaries.
    pub channels: Vec<ChannelPeakSummary>,
    /// Ceiling used for evaluation.
    pub ceiling_dbtp: f64,
}

impl TruePeakReport {
    /// Build a report from a `TruePeakMeter`.
    pub fn from_meter(meter: &TruePeakMeter) -> Self {
        let ceiling = meter.ceiling_dbtp;
        let channels = (0..meter.num_channels())
            .map(|ch| {
                let peak_dbtp = meter.channel_peak_dbtp(ch).unwrap_or(f64::NEG_INFINITY);
                let overshoot = TruePeakOvershoot::new(peak_dbtp, ceiling, ch);
                ChannelPeakSummary {
                    channel: ch,
                    peak_dbtp,
                    severity: overshoot.severity(),
                }
            })
            .collect();
        Self {
            channels,
            ceiling_dbtp: ceiling,
        }
    }

    /// Return the channel index with the worst (highest) true peak.
    /// Returns `None` if there are no channels.
    pub fn worst_channel(&self) -> Option<usize> {
        self.channels
            .iter()
            .max_by(|a, b| {
                a.peak_dbtp
                    .partial_cmp(&b.peak_dbtp)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.channel)
    }

    /// `true` if any channel is in overshoot.
    pub fn has_overshoot(&self) -> bool {
        self.channels
            .iter()
            .any(|s| s.severity != OvershootSeverity::None)
    }

    /// Return the worst severity across all channels.
    pub fn worst_severity(&self) -> OvershootSeverity {
        self.channels
            .iter()
            .map(|s| s.severity)
            .max()
            .unwrap_or(OvershootSeverity::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── OvershootSeverity ────────────────────────────────────────────────────

    #[test]
    fn severity_none_when_under_ceiling() {
        let ov = TruePeakOvershoot::new(-2.0, -1.0, 0);
        assert_eq!(ov.severity(), OvershootSeverity::None);
    }

    #[test]
    fn severity_minor_small_excess() {
        let ov = TruePeakOvershoot::new(-0.8, -1.0, 0); // +0.2 dBTP excess
        assert_eq!(ov.severity(), OvershootSeverity::Minor);
    }

    #[test]
    fn severity_moderate_medium_excess() {
        let ov = TruePeakOvershoot::new(0.5, -1.0, 0); // +1.5 dBTP excess
        assert_eq!(ov.severity(), OvershootSeverity::Moderate);
    }

    #[test]
    fn severity_severe_large_excess() {
        let ov = TruePeakOvershoot::new(2.0, -1.0, 0); // +3.0 dBTP excess
        assert_eq!(ov.severity(), OvershootSeverity::Severe);
    }

    #[test]
    fn severity_ordering() {
        assert!(OvershootSeverity::None < OvershootSeverity::Minor);
        assert!(OvershootSeverity::Minor < OvershootSeverity::Moderate);
        assert!(OvershootSeverity::Moderate < OvershootSeverity::Severe);
    }

    // ── TruePeakMeter ────────────────────────────────────────────────────────

    #[test]
    fn meter_starts_at_neg_inf() {
        let m = TruePeakMeter::new(2, -1.0);
        assert!(m.true_peak_dbtp().is_infinite() && m.true_peak_dbtp() < 0.0);
    }

    #[test]
    fn meter_process_sample_tracks_channel() {
        let mut m = TruePeakMeter::new(2, -1.0);
        m.process_sample(0.9, 0);
        m.process_sample(0.5, 1);
        let ch0 = m.channel_peak_dbtp(0).expect("ch0 should be valid");
        let ch1 = m.channel_peak_dbtp(1).expect("ch1 should be valid");
        assert!(ch0 > ch1);
    }

    #[test]
    fn meter_true_peak_is_max_across_channels() {
        let mut m = TruePeakMeter::new(3, -1.0);
        m.process_sample(0.3, 0);
        m.process_sample(0.9, 1);
        m.process_sample(0.6, 2);
        assert!((m.true_peak_dbtp() - linear_to_dbtp(0.9)).abs() < 1e-9);
    }

    #[test]
    fn meter_has_overshoot_above_ceiling() {
        let mut m = TruePeakMeter::new(1, -1.0);
        m.process_sample(0.99, 0); // ≈ -0.087 dBTP > -1.0 ceiling
        assert!(m.has_overshoot());
    }

    #[test]
    fn meter_no_overshoot_below_ceiling() {
        let mut m = TruePeakMeter::new(1, -1.0);
        m.process_sample(0.5, 0); // ≈ -6 dBTP, well below -1.0 ceiling
        assert!(!m.has_overshoot());
    }

    #[test]
    fn meter_reset_clears_peaks() {
        let mut m = TruePeakMeter::new(2, -1.0);
        m.process_sample(0.9, 0);
        m.reset();
        assert!(m.true_peak_dbtp().is_infinite() && m.true_peak_dbtp() < 0.0);
    }

    #[test]
    fn meter_process_frame() {
        let mut m = TruePeakMeter::new(2, -1.0);
        m.process_frame(&[0.4, 0.8]);
        assert!(
            (m.channel_peak_dbtp(1)
                .expect("channel_peak_dbtp should succeed")
                - linear_to_dbtp(0.8))
            .abs()
                < 1e-9
        );
    }

    // ── TruePeakReport ───────────────────────────────────────────────────────

    #[test]
    fn report_worst_channel_identifies_highest_peak() {
        let mut m = TruePeakMeter::new(3, -1.0);
        m.process_sample(0.3, 0);
        m.process_sample(0.95, 1); // highest
        m.process_sample(0.5, 2);
        let report = TruePeakReport::from_meter(&m);
        assert_eq!(report.worst_channel(), Some(1));
    }

    #[test]
    fn report_has_overshoot_when_any_channel_clips() {
        let mut m = TruePeakMeter::new(2, -1.0);
        m.process_sample(0.99, 0); // overshoot
        m.process_sample(0.3, 1); // fine
        let report = TruePeakReport::from_meter(&m);
        assert!(report.has_overshoot());
    }

    #[test]
    fn report_no_overshoot_all_below_ceiling() {
        let mut m = TruePeakMeter::new(2, -1.0);
        m.process_sample(0.5, 0);
        m.process_sample(0.4, 1);
        let report = TruePeakReport::from_meter(&m);
        assert!(!report.has_overshoot());
    }

    #[test]
    fn report_worst_severity_reflects_max() {
        let mut m = TruePeakMeter::new(2, -1.0);
        m.process_sample(0.5, 0); // below ceiling
        m.process_sample(1.2, 1); // severe overshoot
        let report = TruePeakReport::from_meter(&m);
        assert_eq!(report.worst_severity(), OvershootSeverity::Severe);
    }
}
