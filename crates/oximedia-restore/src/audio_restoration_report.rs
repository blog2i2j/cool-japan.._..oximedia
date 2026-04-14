//! Audio restoration report — structured summary of restoration actions.
//!
//! This module provides a comprehensive report type that captures:
//! - Detected artifact types and counts
//! - Restoration actions applied with per-step statistics
//! - Before/after quality metrics (SNR, THD, dynamic range, RMS)
//! - Human-readable text report generation
//!
//! The report can be built incrementally by adding artifact events and
//! restoration actions, then finalised with quality measurements.

use crate::error::RestoreResult;

// ---------------------------------------------------------------------------
// Artifact types
// ---------------------------------------------------------------------------

/// Category of audio artifact detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArtifactKind {
    /// Vinyl or digital click/pop.
    Click,
    /// Electrical hum (50 Hz / 60 Hz).
    Hum,
    /// Broadband noise floor.
    Noise,
    /// Tape hiss.
    Hiss,
    /// Crackle from old recordings.
    Crackle,
    /// Clipped samples.
    Clipping,
    /// DC offset bias.
    DcOffset,
    /// Tape wow / flutter (speed variation).
    WowFlutter,
    /// Breath noise in speech/voiceover.
    Breath,
    /// Tape dropout.
    TapeDropout,
    /// Reverberant tail (room resonance).
    Reverb,
    /// Pitch deviation / vibrato.
    PitchDeviation,
    /// Phase / stereo alignment error.
    PhaseError,
    /// Unspecified / other artifact.
    Other,
}

impl ArtifactKind {
    /// Return a short display name.
    #[must_use]
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Click => "Click/Pop",
            Self::Hum => "Hum",
            Self::Noise => "Broadband Noise",
            Self::Hiss => "Hiss",
            Self::Crackle => "Crackle",
            Self::Clipping => "Clipping",
            Self::DcOffset => "DC Offset",
            Self::WowFlutter => "Wow/Flutter",
            Self::Breath => "Breath",
            Self::TapeDropout => "Tape Dropout",
            Self::Reverb => "Reverb",
            Self::PitchDeviation => "Pitch Deviation",
            Self::PhaseError => "Phase Error",
            Self::Other => "Other",
        }
    }
}

// ---------------------------------------------------------------------------
// Artifact event
// ---------------------------------------------------------------------------

/// A single detected artifact event.
#[derive(Debug, Clone)]
pub struct ArtifactEvent {
    /// Type of artifact.
    pub kind: ArtifactKind,
    /// Start sample of the artifact.
    pub start_sample: usize,
    /// Duration in samples (0 if point event).
    pub duration_samples: usize,
    /// Severity score 0.0 (minimal) – 1.0 (severe).
    pub severity: f32,
    /// Optional human-readable note.
    pub note: Option<String>,
}

impl ArtifactEvent {
    /// Create a new artifact event.
    #[must_use]
    pub fn new(
        kind: ArtifactKind,
        start_sample: usize,
        duration_samples: usize,
        severity: f32,
    ) -> Self {
        Self {
            kind,
            start_sample,
            duration_samples,
            severity: severity.clamp(0.0, 1.0),
            note: None,
        }
    }

    /// Attach a descriptive note.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Restoration action
// ---------------------------------------------------------------------------

/// A restoration action applied to the audio.
#[derive(Debug, Clone)]
pub struct RestorationAction {
    /// Step name (e.g., "Click removal", "Hum removal").
    pub step_name: String,
    /// Number of events processed.
    pub events_processed: usize,
    /// Total samples modified.
    pub samples_modified: usize,
    /// Processing latency in milliseconds.
    pub latency_ms: f32,
    /// Optional additional statistics.
    pub stats: Vec<(String, String)>,
}

impl RestorationAction {
    /// Create a new restoration action record.
    #[must_use]
    pub fn new(step_name: impl Into<String>) -> Self {
        Self {
            step_name: step_name.into(),
            events_processed: 0,
            samples_modified: 0,
            latency_ms: 0.0,
            stats: Vec::new(),
        }
    }

    /// Set the number of events processed.
    #[must_use]
    pub fn with_events(mut self, n: usize) -> Self {
        self.events_processed = n;
        self
    }

    /// Set the number of samples modified.
    #[must_use]
    pub fn with_samples_modified(mut self, n: usize) -> Self {
        self.samples_modified = n;
        self
    }

    /// Set the processing latency.
    #[must_use]
    pub fn with_latency(mut self, ms: f32) -> Self {
        self.latency_ms = ms;
        self
    }

    /// Attach a key-value statistic.
    #[must_use]
    pub fn with_stat(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.stats.push((key.into(), value.into()));
        self
    }
}

// ---------------------------------------------------------------------------
// Quality metrics
// ---------------------------------------------------------------------------

/// Audio quality metrics for before/after comparison.
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// RMS level in dBFS.
    pub rms_dbfs: f32,
    /// Peak level in dBFS.
    pub peak_dbfs: f32,
    /// Crest factor in dB.
    pub crest_factor_db: f32,
    /// Estimated SNR in dB (requires noise floor reference).
    pub snr_db: Option<f32>,
    /// Total harmonic distortion percentage (0–100).
    pub thd_pct: Option<f32>,
    /// Dynamic range in dB.
    pub dynamic_range_db: f32,
    /// DC offset as fraction of full scale (0.0–1.0).
    pub dc_offset: f32,
}

impl QualityMetrics {
    /// Measure quality metrics from a sample buffer.
    ///
    /// # Errors
    ///
    /// Returns `crate::error::RestoreError::InvalidData` for empty input.
    #[allow(clippy::cast_precision_loss)]
    pub fn measure(samples: &[f32]) -> RestoreResult<Self> {
        use crate::error::RestoreError;
        if samples.is_empty() {
            return Err(RestoreError::InvalidData(
                "cannot measure metrics on empty buffer".into(),
            ));
        }

        let n = samples.len() as f32;

        // RMS
        let mean_sq: f32 = samples.iter().map(|&s| s * s).sum::<f32>() / n;
        let rms = mean_sq.sqrt();
        let rms_dbfs = if rms > 1e-10 {
            20.0 * rms.log10()
        } else {
            -120.0
        };

        // Peak
        let peak = samples.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        let peak_dbfs = if peak > 1e-10 {
            20.0 * peak.log10()
        } else {
            -120.0
        };

        // Crest factor
        let crest_factor_db = if rms > 1e-10 && peak > 0.0 {
            20.0 * (peak / rms).log10()
        } else {
            0.0
        };

        // DC offset
        let dc: f32 = samples.iter().sum::<f32>() / n;
        let dc_offset = dc.abs();

        // Dynamic range (difference between 95th and 5th percentile of absolute values)
        let dynamic_range_db = estimate_dynamic_range(samples);

        Ok(Self {
            rms_dbfs,
            peak_dbfs,
            crest_factor_db,
            snr_db: None,
            thd_pct: None,
            dynamic_range_db,
            dc_offset,
        })
    }

    /// Return a new metrics struct with SNR filled in.
    #[must_use]
    pub fn with_snr(mut self, snr_db: f32) -> Self {
        self.snr_db = Some(snr_db);
        self
    }

    /// Return a new metrics struct with THD filled in.
    #[must_use]
    pub fn with_thd(mut self, thd_pct: f32) -> Self {
        self.thd_pct = Some(thd_pct);
        self
    }
}

/// Estimate dynamic range from sample distribution.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn estimate_dynamic_range(samples: &[f32]) -> f32 {
    let mut abs_vals: Vec<f32> = samples.iter().map(|s| s.abs()).collect();
    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = abs_vals.len();
    let lo_idx = (n as f32 * 0.05) as usize;
    let hi_idx = ((n as f32 * 0.95) as usize).min(n.saturating_sub(1));

    let lo_val = abs_vals[lo_idx].max(1e-10);
    let hi_val = abs_vals[hi_idx].max(1e-10);

    20.0 * (hi_val / lo_val).log10()
}

// ---------------------------------------------------------------------------
// The report
// ---------------------------------------------------------------------------

/// Comprehensive audio restoration report.
#[derive(Debug, Clone)]
pub struct AudioRestorationReport {
    /// Input file or source description.
    pub source_description: String,
    /// Total input duration in samples.
    pub total_samples: usize,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// List of detected artifact events.
    pub artifacts: Vec<ArtifactEvent>,
    /// Restoration actions applied (in order).
    pub actions: Vec<RestorationAction>,
    /// Quality metrics measured before restoration.
    pub before_metrics: Option<QualityMetrics>,
    /// Quality metrics measured after restoration.
    pub after_metrics: Option<QualityMetrics>,
    /// Any notes or warnings generated during restoration.
    pub notes: Vec<String>,
}

impl AudioRestorationReport {
    /// Create a new empty report.
    #[must_use]
    pub fn new(
        source_description: impl Into<String>,
        total_samples: usize,
        sample_rate: u32,
    ) -> Self {
        Self {
            source_description: source_description.into(),
            total_samples,
            sample_rate,
            artifacts: Vec::new(),
            actions: Vec::new(),
            before_metrics: None,
            after_metrics: None,
            notes: Vec::new(),
        }
    }

    /// Add an artifact event.
    pub fn add_artifact(&mut self, event: ArtifactEvent) {
        self.artifacts.push(event);
    }

    /// Add multiple artifact events at once.
    pub fn add_artifacts(&mut self, events: impl IntoIterator<Item = ArtifactEvent>) {
        self.artifacts.extend(events);
    }

    /// Record a restoration action.
    pub fn add_action(&mut self, action: RestorationAction) {
        self.actions.push(action);
    }

    /// Set pre-restoration quality metrics.
    pub fn set_before_metrics(&mut self, metrics: QualityMetrics) {
        self.before_metrics = Some(metrics);
    }

    /// Set post-restoration quality metrics.
    pub fn set_after_metrics(&mut self, metrics: QualityMetrics) {
        self.after_metrics = Some(metrics);
    }

    /// Append a note or warning.
    pub fn add_note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }

    /// Return the number of artifacts of a specific kind.
    #[must_use]
    pub fn artifact_count(&self, kind: ArtifactKind) -> usize {
        self.artifacts.iter().filter(|e| e.kind == kind).count()
    }

    /// Return the total number of artifacts detected.
    #[must_use]
    pub fn total_artifact_count(&self) -> usize {
        self.artifacts.len()
    }

    /// Return the mean severity across all artifact events, or `None` if empty.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean_artifact_severity(&self) -> Option<f32> {
        if self.artifacts.is_empty() {
            return None;
        }
        let total: f32 = self.artifacts.iter().map(|e| e.severity).sum();
        Some(total / self.artifacts.len() as f32)
    }

    /// Return the total processing time across all actions (ms).
    #[must_use]
    pub fn total_processing_ms(&self) -> f32 {
        self.actions.iter().map(|a| a.latency_ms).sum()
    }

    /// Return the total samples modified across all actions.
    #[must_use]
    pub fn total_samples_modified(&self) -> usize {
        self.actions.iter().map(|a| a.samples_modified).sum()
    }

    /// Compute the improvement in RMS (dB).
    ///
    /// Returns `None` if either metrics snapshot is unavailable.
    #[must_use]
    pub fn snr_improvement_db(&self) -> Option<f32> {
        let before = self.before_metrics.as_ref()?.snr_db?;
        let after = self.after_metrics.as_ref()?.snr_db?;
        Some(after - before)
    }

    /// Compute the change in RMS level (dB).
    ///
    /// Positive = louder after, negative = quieter after.
    #[must_use]
    pub fn rms_change_db(&self) -> Option<f32> {
        let before = self.before_metrics.as_ref()?.rms_dbfs;
        let after = self.after_metrics.as_ref()?.rms_dbfs;
        Some(after - before)
    }

    /// Generate a human-readable text report.
    #[must_use]
    pub fn to_text_report(&self) -> String {
        let mut out = String::new();
        let line = "=".repeat(60);

        out.push_str(&format!("{line}\n"));
        out.push_str("  AUDIO RESTORATION REPORT\n");
        out.push_str(&format!("{line}\n"));
        out.push_str(&format!("Source  : {}\n", self.source_description));

        let duration_s = if self.sample_rate > 0 {
            self.total_samples as f32 / self.sample_rate as f32
        } else {
            0.0
        };
        out.push_str(&format!(
            "Duration: {duration_s:.2}s  ({} samples @ {} Hz)\n",
            self.total_samples, self.sample_rate
        ));
        out.push('\n');

        // Artifacts section
        out.push_str("── DETECTED ARTIFACTS ─────────────────────────────────\n");
        if self.artifacts.is_empty() {
            out.push_str("  (none detected)\n");
        } else {
            // Group by kind
            let kinds = [
                ArtifactKind::Click,
                ArtifactKind::Crackle,
                ArtifactKind::Hum,
                ArtifactKind::Hiss,
                ArtifactKind::Noise,
                ArtifactKind::Clipping,
                ArtifactKind::DcOffset,
                ArtifactKind::WowFlutter,
                ArtifactKind::Breath,
                ArtifactKind::TapeDropout,
                ArtifactKind::Reverb,
                ArtifactKind::PitchDeviation,
                ArtifactKind::PhaseError,
                ArtifactKind::Other,
            ];
            for kind in &kinds {
                let count = self.artifact_count(*kind);
                if count > 0 {
                    let events: Vec<_> =
                        self.artifacts.iter().filter(|e| e.kind == *kind).collect();
                    let mean_sev: f32 =
                        events.iter().map(|e| e.severity).sum::<f32>() / events.len() as f32;
                    out.push_str(&format!(
                        "  {:20} {:5} events  (mean severity: {:.2})\n",
                        kind.display_name(),
                        count,
                        mean_sev
                    ));
                }
            }
        }
        out.push('\n');

        // Actions section
        out.push_str("── RESTORATION ACTIONS ─────────────────────────────────\n");
        if self.actions.is_empty() {
            out.push_str("  (none applied)\n");
        } else {
            for (i, action) in self.actions.iter().enumerate() {
                out.push_str(&format!("  {}. {}\n", i + 1, action.step_name));
                out.push_str(&format!(
                    "     Events processed: {}  Samples modified: {}  Latency: {:.1}ms\n",
                    action.events_processed, action.samples_modified, action.latency_ms
                ));
                for (k, v) in &action.stats {
                    out.push_str(&format!("     {k}: {v}\n"));
                }
            }
        }
        out.push('\n');

        // Quality metrics section
        out.push_str("── QUALITY METRICS ─────────────────────────────────────\n");
        out.push_str(&format!(
            "  {:30} {:>12} {:>12}\n",
            "Metric", "Before", "After"
        ));
        out.push_str(&format!("  {}\n", "-".repeat(56)));

        let format_metric = |val: Option<f32>| -> String {
            val.map_or_else(|| "  N/A".to_string(), |v| format!("{v:>+10.1}"))
        };

        let before_rms = self.before_metrics.as_ref().map(|m| m.rms_dbfs);
        let after_rms = self.after_metrics.as_ref().map(|m| m.rms_dbfs);
        out.push_str(&format!(
            "  {:30} {:>12} {:>12}\n",
            "RMS level (dBFS)",
            format_metric(before_rms),
            format_metric(after_rms)
        ));

        let before_peak = self.before_metrics.as_ref().map(|m| m.peak_dbfs);
        let after_peak = self.after_metrics.as_ref().map(|m| m.peak_dbfs);
        out.push_str(&format!(
            "  {:30} {:>12} {:>12}\n",
            "Peak level (dBFS)",
            format_metric(before_peak),
            format_metric(after_peak)
        ));

        let before_cf = self.before_metrics.as_ref().map(|m| m.crest_factor_db);
        let after_cf = self.after_metrics.as_ref().map(|m| m.crest_factor_db);
        out.push_str(&format!(
            "  {:30} {:>12} {:>12}\n",
            "Crest factor (dB)",
            format_metric(before_cf),
            format_metric(after_cf)
        ));

        let before_snr = self.before_metrics.as_ref().and_then(|m| m.snr_db);
        let after_snr = self.after_metrics.as_ref().and_then(|m| m.snr_db);
        out.push_str(&format!(
            "  {:30} {:>12} {:>12}\n",
            "SNR (dB)",
            format_metric(before_snr),
            format_metric(after_snr)
        ));

        let before_dr = self.before_metrics.as_ref().map(|m| m.dynamic_range_db);
        let after_dr = self.after_metrics.as_ref().map(|m| m.dynamic_range_db);
        out.push_str(&format!(
            "  {:30} {:>12} {:>12}\n",
            "Dynamic range (dB)",
            format_metric(before_dr),
            format_metric(after_dr)
        ));

        let before_dc = self.before_metrics.as_ref().map(|m| m.dc_offset);
        let after_dc = self.after_metrics.as_ref().map(|m| m.dc_offset);
        out.push_str(&format!(
            "  {:30} {:>12} {:>12}\n",
            "DC offset",
            format_metric(before_dc),
            format_metric(after_dc)
        ));

        // Summary
        out.push('\n');
        out.push_str("── SUMMARY ─────────────────────────────────────────────\n");
        out.push_str(&format!(
            "  Total artifacts : {}\n",
            self.total_artifact_count()
        ));
        if let Some(sev) = self.mean_artifact_severity() {
            out.push_str(&format!("  Mean severity   : {sev:.2}\n"));
        }
        out.push_str(&format!("  Actions applied : {}\n", self.actions.len()));
        out.push_str(&format!(
            "  Samples modified: {} ({:.1}%)\n",
            self.total_samples_modified(),
            if self.total_samples > 0 {
                self.total_samples_modified() as f32 / self.total_samples as f32 * 100.0
            } else {
                0.0
            }
        ));
        out.push_str(&format!(
            "  Total proc. time: {:.1}ms\n",
            self.total_processing_ms()
        ));

        if let Some(snr_imp) = self.snr_improvement_db() {
            out.push_str(&format!("  SNR improvement : {snr_imp:+.1} dB\n"));
        }

        if !self.notes.is_empty() {
            out.push('\n');
            out.push_str("── NOTES ────────────────────────────────────────────────\n");
            for note in &self.notes {
                out.push_str(&format!("  • {note}\n"));
            }
        }

        out.push_str(&format!("{line}\n"));
        out
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Fluent builder for `AudioRestorationReport`.
#[derive(Debug, Default)]
pub struct ReportBuilder {
    source: String,
    total_samples: usize,
    sample_rate: u32,
    artifacts: Vec<ArtifactEvent>,
    actions: Vec<RestorationAction>,
    before_metrics: Option<QualityMetrics>,
    after_metrics: Option<QualityMetrics>,
    notes: Vec<String>,
}

impl ReportBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the source description.
    #[must_use]
    pub fn source(mut self, desc: impl Into<String>) -> Self {
        self.source = desc.into();
        self
    }

    /// Set the total sample count.
    #[must_use]
    pub fn total_samples(mut self, n: usize) -> Self {
        self.total_samples = n;
        self
    }

    /// Set the sample rate.
    #[must_use]
    pub fn sample_rate(mut self, sr: u32) -> Self {
        self.sample_rate = sr;
        self
    }

    /// Add an artifact event.
    #[must_use]
    pub fn artifact(mut self, event: ArtifactEvent) -> Self {
        self.artifacts.push(event);
        self
    }

    /// Add a restoration action.
    #[must_use]
    pub fn action(mut self, action: RestorationAction) -> Self {
        self.actions.push(action);
        self
    }

    /// Set pre-restoration metrics.
    #[must_use]
    pub fn before_metrics(mut self, m: QualityMetrics) -> Self {
        self.before_metrics = Some(m);
        self
    }

    /// Set post-restoration metrics.
    #[must_use]
    pub fn after_metrics(mut self, m: QualityMetrics) -> Self {
        self.after_metrics = Some(m);
        self
    }

    /// Add a note.
    #[must_use]
    pub fn note(mut self, text: impl Into<String>) -> Self {
        self.notes.push(text.into());
        self
    }

    /// Build the final report.
    #[must_use]
    pub fn build(self) -> AudioRestorationReport {
        AudioRestorationReport {
            source_description: self.source,
            total_samples: self.total_samples,
            sample_rate: self.sample_rate,
            artifacts: self.artifacts,
            actions: self.actions,
            before_metrics: self.before_metrics,
            after_metrics: self.after_metrics,
            notes: self.notes,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SR: u32 = 44100;

    fn make_sine(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.5 * (2.0 * PI * freq * i as f32 / SR as f32).sin())
            .collect()
    }

    #[test]
    fn test_quality_metrics_measure_sine() {
        let samples = make_sine(440.0, 44100);
        let m = QualityMetrics::measure(&samples).expect("ok");
        // Sine at 0.5 amplitude: RMS = 0.5/sqrt(2) ≈ 0.354
        // 20*log10(0.354) ≈ -9.03 dBFS
        assert!(
            m.rms_dbfs > -12.0 && m.rms_dbfs < -6.0,
            "rms {}",
            m.rms_dbfs
        );
        // Peak should be ~-6 dBFS (20*log10(0.5))
        assert!(
            m.peak_dbfs > -7.0 && m.peak_dbfs < -5.0,
            "peak {}",
            m.peak_dbfs
        );
        assert!(m.crest_factor_db > 0.0);
    }

    #[test]
    fn test_quality_metrics_measure_empty_errors() {
        let result = QualityMetrics::measure(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_quality_metrics_silence() {
        let samples = vec![0.0_f32; 1024];
        let m = QualityMetrics::measure(&samples).expect("ok");
        assert!(
            m.rms_dbfs <= -110.0,
            "silence rms should be very low: {}",
            m.rms_dbfs
        );
        assert!((m.dc_offset).abs() < f32::EPSILON);
    }

    #[test]
    fn test_artifact_event_severity_clamped() {
        let e = ArtifactEvent::new(ArtifactKind::Click, 100, 10, 2.5);
        assert!(
            (e.severity - 1.0).abs() < f32::EPSILON,
            "severity should be clamped to 1.0"
        );
    }

    #[test]
    fn test_report_add_artifacts_and_count() {
        let mut report = AudioRestorationReport::new("test.wav", 44100, SR);
        report.add_artifact(ArtifactEvent::new(ArtifactKind::Click, 100, 5, 0.8));
        report.add_artifact(ArtifactEvent::new(ArtifactKind::Click, 500, 5, 0.6));
        report.add_artifact(ArtifactEvent::new(ArtifactKind::Hum, 0, 44100, 0.4));

        assert_eq!(report.artifact_count(ArtifactKind::Click), 2);
        assert_eq!(report.artifact_count(ArtifactKind::Hum), 1);
        assert_eq!(report.artifact_count(ArtifactKind::Noise), 0);
        assert_eq!(report.total_artifact_count(), 3);
    }

    #[test]
    fn test_report_mean_severity() {
        let mut report = AudioRestorationReport::new("test.wav", 44100, SR);
        report.add_artifact(ArtifactEvent::new(ArtifactKind::Click, 0, 10, 0.4));
        report.add_artifact(ArtifactEvent::new(ArtifactKind::Click, 100, 10, 0.6));
        let mean = report.mean_artifact_severity().expect("should have mean");
        assert!((mean - 0.5).abs() < 0.01, "mean should be 0.5, got {mean}");
    }

    #[test]
    fn test_report_mean_severity_empty() {
        let report = AudioRestorationReport::new("test.wav", 0, SR);
        assert!(report.mean_artifact_severity().is_none());
    }

    #[test]
    fn test_report_actions_processing_totals() {
        let mut report = AudioRestorationReport::new("test.wav", 44100, SR);
        report.add_action(
            RestorationAction::new("Click removal")
                .with_events(12)
                .with_samples_modified(600)
                .with_latency(5.3),
        );
        report.add_action(
            RestorationAction::new("Hum removal")
                .with_events(1)
                .with_samples_modified(44100)
                .with_latency(2.1),
        );

        assert_eq!(report.total_samples_modified(), 44700);
        assert!((report.total_processing_ms() - 7.4).abs() < 0.01);
    }

    #[test]
    fn test_report_builder_roundtrip() {
        let samples = make_sine(440.0, 44100);
        let before = QualityMetrics::measure(&samples).expect("ok");
        let after = QualityMetrics::measure(&samples)
            .expect("ok")
            .with_snr(45.0);

        let report = ReportBuilder::new()
            .source("vinyl_scan.wav")
            .total_samples(44100)
            .sample_rate(SR)
            .artifact(ArtifactEvent::new(ArtifactKind::Click, 1000, 10, 0.7))
            .action(
                RestorationAction::new("Click removal")
                    .with_events(1)
                    .with_samples_modified(10),
            )
            .before_metrics(before)
            .after_metrics(after)
            .note("Moderate surface noise throughout")
            .build();

        assert_eq!(report.source_description, "vinyl_scan.wav");
        assert_eq!(report.total_artifact_count(), 1);
        assert_eq!(report.actions.len(), 1);
        assert!(report.before_metrics.is_some());
        assert!(report.after_metrics.is_some());
        assert_eq!(report.notes.len(), 1);
        assert!(report.snr_improvement_db().is_none()); // before has no SNR
    }

    #[test]
    fn test_report_snr_improvement() {
        let samples = make_sine(440.0, 4096);
        let before = QualityMetrics::measure(&samples)
            .expect("ok")
            .with_snr(20.0);
        let after = QualityMetrics::measure(&samples)
            .expect("ok")
            .with_snr(35.0);

        let mut report = AudioRestorationReport::new("test.wav", 4096, SR);
        report.set_before_metrics(before);
        report.set_after_metrics(after);

        let imp = report
            .snr_improvement_db()
            .expect("should have snr improvement");
        assert!(
            (imp - 15.0).abs() < 0.01,
            "improvement should be 15 dB, got {imp}"
        );
    }

    #[test]
    fn test_report_text_report_contains_sections() {
        let mut report = AudioRestorationReport::new("test.wav", 44100, SR);
        report.add_artifact(ArtifactEvent::new(ArtifactKind::Crackle, 0, 100, 0.3));
        report.add_action(
            RestorationAction::new("Crackle removal")
                .with_events(1)
                .with_stat("Threshold", "0.3"),
        );
        report.add_note("Processed in batch mode");

        let text = report.to_text_report();
        assert!(text.contains("AUDIO RESTORATION REPORT"));
        assert!(text.contains("Crackle"));
        assert!(text.contains("Crackle removal"));
        assert!(text.contains("NOTES"));
        assert!(text.contains("Processed in batch mode"));
    }

    #[test]
    fn test_artifact_kind_display_names() {
        assert_eq!(ArtifactKind::Click.display_name(), "Click/Pop");
        assert_eq!(ArtifactKind::Hum.display_name(), "Hum");
        assert_eq!(ArtifactKind::Noise.display_name(), "Broadband Noise");
        assert_eq!(ArtifactKind::Other.display_name(), "Other");
    }

    #[test]
    fn test_restoration_action_stats() {
        let action = RestorationAction::new("Noise reduction")
            .with_stat("Noise floor", "-60 dBFS")
            .with_stat("Reduction amount", "12 dB");
        assert_eq!(action.stats.len(), 2);
        assert_eq!(action.stats[0].0, "Noise floor");
        assert_eq!(action.stats[1].1, "12 dB");
    }
}
