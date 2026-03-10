#![allow(dead_code)]
//! Loudness session management for audio post-production.
//!
//! Tracks loudness measurements across an entire mix session, enforces target
//! standards (EBU R128, ATSC A/85, etc.), and generates compliance reports.

use std::fmt;

/// Loudness standard to target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoudnessStandard {
    /// EBU R128 (-23 LUFS, tolerance +/- 1 LU).
    EbuR128,
    /// ATSC A/85 (-24 LKFS, tolerance +/- 2 LU).
    AtscA85,
    /// ITU-R BS.1770 reference.
    ItuBs1770,
    /// Streaming platform (typically -14 LUFS).
    Streaming,
    /// Custom target.
    Custom,
}

impl fmt::Display for LoudnessStandard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EbuR128 => write!(f, "EBU R128"),
            Self::AtscA85 => write!(f, "ATSC A/85"),
            Self::ItuBs1770 => write!(f, "ITU-R BS.1770"),
            Self::Streaming => write!(f, "Streaming"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// Loudness target configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct LoudnessTarget {
    /// Standard being followed.
    pub standard: LoudnessStandard,
    /// Target integrated loudness (LUFS).
    pub target_lufs: f64,
    /// Allowed tolerance (LU).
    pub tolerance_lu: f64,
    /// Maximum true-peak level (dBTP).
    pub max_true_peak_dbtp: f64,
    /// Maximum loudness range (LRA) in LU, 0 means unconstrained.
    pub max_lra_lu: f64,
}

impl LoudnessTarget {
    /// Create a target from a known standard.
    #[must_use]
    pub fn from_standard(standard: LoudnessStandard) -> Self {
        match standard {
            LoudnessStandard::EbuR128 => Self {
                standard,
                target_lufs: -23.0,
                tolerance_lu: 1.0,
                max_true_peak_dbtp: -1.0,
                max_lra_lu: 0.0,
            },
            LoudnessStandard::AtscA85 => Self {
                standard,
                target_lufs: -24.0,
                tolerance_lu: 2.0,
                max_true_peak_dbtp: -2.0,
                max_lra_lu: 0.0,
            },
            LoudnessStandard::ItuBs1770 => Self {
                standard,
                target_lufs: -24.0,
                tolerance_lu: 1.0,
                max_true_peak_dbtp: -1.0,
                max_lra_lu: 0.0,
            },
            LoudnessStandard::Streaming => Self {
                standard,
                target_lufs: -14.0,
                tolerance_lu: 1.0,
                max_true_peak_dbtp: -1.0,
                max_lra_lu: 0.0,
            },
            LoudnessStandard::Custom => Self {
                standard,
                target_lufs: -23.0,
                tolerance_lu: 1.0,
                max_true_peak_dbtp: -1.0,
                max_lra_lu: 0.0,
            },
        }
    }

    /// Create a custom loudness target.
    #[must_use]
    pub fn custom(target_lufs: f64, tolerance_lu: f64, max_true_peak_dbtp: f64) -> Self {
        Self {
            standard: LoudnessStandard::Custom,
            target_lufs,
            tolerance_lu,
            max_true_peak_dbtp,
            max_lra_lu: 0.0,
        }
    }

    /// Check if an integrated loudness value passes this target.
    #[must_use]
    pub fn is_compliant(&self, integrated_lufs: f64, true_peak_dbtp: f64) -> bool {
        let lufs_ok = (integrated_lufs - self.target_lufs).abs() <= self.tolerance_lu;
        let peak_ok = true_peak_dbtp <= self.max_true_peak_dbtp;
        lufs_ok && peak_ok
    }
}

/// A single loudness measurement snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct LoudnessMeasurement {
    /// Timestamp in seconds from session start.
    pub time_s: f64,
    /// Momentary loudness (LUFS, 400 ms window).
    pub momentary_lufs: f64,
    /// Short-term loudness (LUFS, 3 s window).
    pub short_term_lufs: f64,
    /// Running integrated loudness (LUFS).
    pub integrated_lufs: f64,
    /// True peak level (dBTP).
    pub true_peak_dbtp: f64,
}

/// Loudness session tracking an entire mix.
#[derive(Debug, Clone)]
pub struct LoudnessSession {
    /// Session name.
    pub name: String,
    /// Target to comply with.
    pub target: LoudnessTarget,
    /// Collected measurements.
    measurements: Vec<LoudnessMeasurement>,
    /// Sample rate.
    pub sample_rate: u32,
}

impl LoudnessSession {
    /// Create a new loudness session.
    #[must_use]
    pub fn new(name: &str, target: LoudnessTarget, sample_rate: u32) -> Self {
        Self {
            name: name.to_string(),
            target,
            measurements: Vec::new(),
            sample_rate,
        }
    }

    /// Add a measurement to the session.
    pub fn add_measurement(&mut self, measurement: LoudnessMeasurement) {
        self.measurements.push(measurement);
    }

    /// Number of measurements collected.
    #[must_use]
    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }

    /// Return the final integrated loudness (last measurement).
    #[must_use]
    pub fn final_integrated_lufs(&self) -> Option<f64> {
        self.measurements.last().map(|m| m.integrated_lufs)
    }

    /// Return the maximum true peak across all measurements.
    #[must_use]
    pub fn max_true_peak_dbtp(&self) -> Option<f64> {
        self.measurements
            .iter()
            .map(|m| m.true_peak_dbtp)
            .reduce(f64::max)
    }

    /// Return the loudness range (LRA) approximated as max - min short-term loudness.
    #[must_use]
    pub fn loudness_range_lu(&self) -> Option<f64> {
        if self.measurements.len() < 2 {
            return None;
        }
        let vals: Vec<f64> = self
            .measurements
            .iter()
            .map(|m| m.short_term_lufs)
            .collect();
        let min = vals.iter().copied().reduce(f64::min)?;
        let max = vals.iter().copied().reduce(f64::max)?;
        Some(max - min)
    }

    /// Generate a compliance report.
    #[must_use]
    pub fn report(&self) -> SessionReport {
        let integrated = self.final_integrated_lufs().unwrap_or(f64::NEG_INFINITY);
        let peak = self.max_true_peak_dbtp().unwrap_or(f64::NEG_INFINITY);
        let lra = self.loudness_range_lu().unwrap_or(0.0);
        let compliant = self.target.is_compliant(integrated, peak);

        SessionReport {
            session_name: self.name.clone(),
            standard: self.target.standard,
            target_lufs: self.target.target_lufs,
            measured_integrated_lufs: integrated,
            measured_true_peak_dbtp: peak,
            measured_lra_lu: lra,
            pass: compliant,
            total_measurements: self.measurements.len(),
        }
    }

    /// Compute how many dB of gain correction is needed to hit the target.
    #[must_use]
    pub fn correction_db(&self) -> Option<f64> {
        self.final_integrated_lufs()
            .map(|lufs| self.target.target_lufs - lufs)
    }

    /// Return all measurements.
    #[must_use]
    pub fn measurements(&self) -> &[LoudnessMeasurement] {
        &self.measurements
    }
}

/// Report summarising session compliance.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionReport {
    /// Session name.
    pub session_name: String,
    /// Standard applied.
    pub standard: LoudnessStandard,
    /// Target in LUFS.
    pub target_lufs: f64,
    /// Measured integrated loudness.
    pub measured_integrated_lufs: f64,
    /// Measured true peak.
    pub measured_true_peak_dbtp: f64,
    /// Measured loudness range.
    pub measured_lra_lu: f64,
    /// Overall pass / fail.
    pub pass: bool,
    /// Number of measurements in session.
    pub total_measurements: usize,
}

impl fmt::Display for SessionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Session '{}' [{standard}]: Integrated={lufs:.1} LUFS, TruePeak={peak:.1} dBTP, LRA={lra:.1} LU — {result}",
            self.session_name,
            standard = self.standard,
            lufs = self.measured_integrated_lufs,
            peak = self.measured_true_peak_dbtp,
            lra = self.measured_lra_lu,
            result = if self.pass { "PASS" } else { "FAIL" },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loudness_standard_display() {
        assert_eq!(format!("{}", LoudnessStandard::EbuR128), "EBU R128");
        assert_eq!(format!("{}", LoudnessStandard::AtscA85), "ATSC A/85");
    }

    #[test]
    fn test_target_from_standard_ebu() {
        let t = LoudnessTarget::from_standard(LoudnessStandard::EbuR128);
        assert!((t.target_lufs - (-23.0)).abs() < f64::EPSILON);
        assert!((t.tolerance_lu - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_target_from_standard_atsc() {
        let t = LoudnessTarget::from_standard(LoudnessStandard::AtscA85);
        assert!((t.target_lufs - (-24.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_custom_target() {
        let t = LoudnessTarget::custom(-16.0, 0.5, -1.0);
        assert_eq!(t.standard, LoudnessStandard::Custom);
        assert!((t.target_lufs - (-16.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compliance_pass() {
        let t = LoudnessTarget::from_standard(LoudnessStandard::EbuR128);
        assert!(t.is_compliant(-23.5, -2.0));
    }

    #[test]
    fn test_compliance_fail_loudness() {
        let t = LoudnessTarget::from_standard(LoudnessStandard::EbuR128);
        // -20.0 is 3 LU off from -23.0 => fails with 1 LU tolerance
        assert!(!t.is_compliant(-20.0, -2.0));
    }

    #[test]
    fn test_compliance_fail_peak() {
        let t = LoudnessTarget::from_standard(LoudnessStandard::EbuR128);
        // True peak above -1.0 dBTP
        assert!(!t.is_compliant(-23.0, 0.5));
    }

    #[test]
    fn test_session_creation() {
        let target = LoudnessTarget::from_standard(LoudnessStandard::EbuR128);
        let session = LoudnessSession::new("Mix01", target, 48000);
        assert_eq!(session.name, "Mix01");
        assert_eq!(session.measurement_count(), 0);
    }

    #[test]
    fn test_session_add_measurement() {
        let target = LoudnessTarget::from_standard(LoudnessStandard::EbuR128);
        let mut session = LoudnessSession::new("Mix01", target, 48000);
        session.add_measurement(LoudnessMeasurement {
            time_s: 0.0,
            momentary_lufs: -22.0,
            short_term_lufs: -22.5,
            integrated_lufs: -22.8,
            true_peak_dbtp: -3.0,
        });
        assert_eq!(session.measurement_count(), 1);
    }

    #[test]
    fn test_session_final_integrated() {
        let target = LoudnessTarget::from_standard(LoudnessStandard::EbuR128);
        let mut session = LoudnessSession::new("Mix01", target, 48000);
        session.add_measurement(LoudnessMeasurement {
            time_s: 0.0,
            momentary_lufs: -25.0,
            short_term_lufs: -24.0,
            integrated_lufs: -23.5,
            true_peak_dbtp: -4.0,
        });
        session.add_measurement(LoudnessMeasurement {
            time_s: 3.0,
            momentary_lufs: -22.0,
            short_term_lufs: -22.5,
            integrated_lufs: -23.0,
            true_peak_dbtp: -2.0,
        });
        assert!(
            (session
                .final_integrated_lufs()
                .expect("final_integrated_lufs should succeed")
                - (-23.0))
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_session_max_true_peak() {
        let target = LoudnessTarget::from_standard(LoudnessStandard::EbuR128);
        let mut session = LoudnessSession::new("Mix01", target, 48000);
        session.add_measurement(LoudnessMeasurement {
            time_s: 0.0,
            momentary_lufs: -25.0,
            short_term_lufs: -24.0,
            integrated_lufs: -23.5,
            true_peak_dbtp: -4.0,
        });
        session.add_measurement(LoudnessMeasurement {
            time_s: 3.0,
            momentary_lufs: -22.0,
            short_term_lufs: -22.5,
            integrated_lufs: -23.0,
            true_peak_dbtp: -1.5,
        });
        assert!(
            (session
                .max_true_peak_dbtp()
                .expect("max_true_peak_dbtp should succeed")
                - (-1.5))
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_session_report_pass() {
        let target = LoudnessTarget::from_standard(LoudnessStandard::EbuR128);
        let mut session = LoudnessSession::new("Mix01", target, 48000);
        session.add_measurement(LoudnessMeasurement {
            time_s: 10.0,
            momentary_lufs: -23.0,
            short_term_lufs: -23.0,
            integrated_lufs: -23.0,
            true_peak_dbtp: -2.0,
        });
        let report = session.report();
        assert!(report.pass);
    }

    #[test]
    fn test_session_correction_db() {
        let target = LoudnessTarget::from_standard(LoudnessStandard::EbuR128);
        let mut session = LoudnessSession::new("Mix01", target, 48000);
        session.add_measurement(LoudnessMeasurement {
            time_s: 10.0,
            momentary_lufs: -20.0,
            short_term_lufs: -20.0,
            integrated_lufs: -20.0,
            true_peak_dbtp: -2.0,
        });
        // Need -3 dB to go from -20 to -23
        assert!(
            (session
                .correction_db()
                .expect("correction_db should succeed")
                - (-3.0))
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_loudness_range() {
        let target = LoudnessTarget::from_standard(LoudnessStandard::EbuR128);
        let mut session = LoudnessSession::new("Mix01", target, 48000);
        session.add_measurement(LoudnessMeasurement {
            time_s: 0.0,
            momentary_lufs: -30.0,
            short_term_lufs: -28.0,
            integrated_lufs: -25.0,
            true_peak_dbtp: -3.0,
        });
        session.add_measurement(LoudnessMeasurement {
            time_s: 3.0,
            momentary_lufs: -18.0,
            short_term_lufs: -20.0,
            integrated_lufs: -23.0,
            true_peak_dbtp: -1.5,
        });
        let lra = session
            .loudness_range_lu()
            .expect("loudness_range_lu should succeed");
        assert!((lra - 8.0).abs() < f64::EPSILON); // -20 - (-28) = 8
    }

    #[test]
    fn test_session_report_display() {
        let target = LoudnessTarget::from_standard(LoudnessStandard::Streaming);
        let mut session = LoudnessSession::new("Podcast", target, 48000);
        session.add_measurement(LoudnessMeasurement {
            time_s: 60.0,
            momentary_lufs: -14.0,
            short_term_lufs: -14.0,
            integrated_lufs: -14.0,
            true_peak_dbtp: -2.0,
        });
        let report = session.report();
        let display = format!("{report}");
        assert!(display.contains("Podcast"));
        assert!(display.contains("PASS"));
    }

    #[test]
    fn test_streaming_target() {
        let t = LoudnessTarget::from_standard(LoudnessStandard::Streaming);
        assert!((t.target_lufs - (-14.0)).abs() < f64::EPSILON);
    }
}
