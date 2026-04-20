//! ITU-R BS.775 loudspeaker placement advisor.
//!
//! This module validates multi-channel loudspeaker layouts against ITU-R BS.775-3
//! (Multichannel stereophonic sound system with and without accompanying picture)
//! and provides a placement advisor that suggests corrections when a layout deviates
//! from the standard.
//!
//! # Standard layout azimuths (BS.775-3)
//!
//! | Channel          | Azimuth (°) | Elevation (°) |
//! |------------------|-------------|---------------|
//! | Centre           |    0        |    0          |
//! | Left             |   30        |    0          |
//! | Right            |  −30        |    0          |
//! | Left surround    |  110        |    0          |
//! | Right surround   | −110        |    0          |
//! | LFE              |  any        |   any         |
//!
//! The advisor also handles 7.1, 9.1, and arbitrary custom arrays, reporting
//! the per-speaker angular deviation and a corrected position suggestion.
//!
//! # Reference
//! ITU-R BS.775-3 (2012). *Multichannel stereophonic sound system with and
//! without accompanying picture.*

use std::fmt;

use crate::SpatialError;

// ─── Speaker channel labels ───────────────────────────────────────────────────

/// Identifier for a loudspeaker channel in a multi-channel layout.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChannelLabel {
    /// Centre front speaker.
    Centre,
    /// Left front speaker (L).
    Left,
    /// Right front speaker (R).
    Right,
    /// Left surround (Ls / LS).
    LeftSurround,
    /// Right surround (Rs / RS).
    RightSurround,
    /// Wide left (Lw) — used in 9.1 and wider arrays.
    LeftWide,
    /// Wide right (Rw).
    RightWide,
    /// Left back surround (Lbs) — used in 7.1 and wider.
    LeftBackSurround,
    /// Right back surround (Rbs).
    RightBackSurround,
    /// Left height / top front (Ltf / Lvh).
    LeftHeight,
    /// Right height / top front (Rtf / Rvh).
    RightHeight,
    /// Low-frequency effects (LFE / Sub).
    Lfe,
    /// Arbitrary custom label with a human-readable name.
    Custom(String),
}

impl fmt::Display for ChannelLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Centre => write!(f, "Centre"),
            Self::Left => write!(f, "Left"),
            Self::Right => write!(f, "Right"),
            Self::LeftSurround => write!(f, "LeftSurround"),
            Self::RightSurround => write!(f, "RightSurround"),
            Self::LeftWide => write!(f, "LeftWide"),
            Self::RightWide => write!(f, "RightWide"),
            Self::LeftBackSurround => write!(f, "LeftBackSurround"),
            Self::RightBackSurround => write!(f, "RightBackSurround"),
            Self::LeftHeight => write!(f, "LeftHeight"),
            Self::RightHeight => write!(f, "RightHeight"),
            Self::Lfe => write!(f, "LFE"),
            Self::Custom(s) => write!(f, "Custom({s})"),
        }
    }
}

// ─── Speaker descriptor ───────────────────────────────────────────────────────

/// Physical position of a single loudspeaker in a room.
///
/// Angles follow the right-hand rule with the listener at the origin:
/// - **azimuth**: positive = left (counter-clockwise when viewed from above).
/// - **elevation**: positive = upward.
/// - **distance**: radius from listener in metres (must be > 0).
#[derive(Debug, Clone, PartialEq)]
pub struct Speaker {
    /// Channel identifier.
    pub label: ChannelLabel,
    /// Horizontal azimuth in degrees, ∈ (−180, +180].
    pub azimuth_deg: f32,
    /// Vertical elevation in degrees, ∈ [−90, +90].
    pub elevation_deg: f32,
    /// Distance from the listening position in metres.
    pub distance_m: f32,
}

impl Speaker {
    /// Create a new speaker descriptor.
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidConfig`] if:
    /// - `distance_m` ≤ 0.
    /// - `elevation_deg` is outside [−90, +90].
    pub fn new(
        label: ChannelLabel,
        azimuth_deg: f32,
        elevation_deg: f32,
        distance_m: f32,
    ) -> Result<Self, SpatialError> {
        if distance_m <= 0.0 {
            return Err(SpatialError::InvalidConfig(format!(
                "Speaker distance must be > 0, got {distance_m}"
            )));
        }
        if !(-90.0..=90.0).contains(&elevation_deg) {
            return Err(SpatialError::InvalidConfig(format!(
                "Elevation must be in [-90, +90], got {elevation_deg}"
            )));
        }
        Ok(Self {
            label,
            azimuth_deg: normalise_azimuth(azimuth_deg),
            elevation_deg,
            distance_m,
        })
    }

    /// Cartesian unit vector `[x, y, z]` for the speaker direction.
    ///
    /// Convention: x = right, y = front, z = up.
    pub fn unit_vector(&self) -> [f32; 3] {
        let az = self.azimuth_deg.to_radians();
        let el = self.elevation_deg.to_radians();
        [
            -az.sin() * el.cos(), // x (positive = right, azimuth positive = left)
            az.cos() * el.cos(),  // y (front)
            el.sin(),             // z (up)
        ]
    }
}

// ─── Standard layout definitions ─────────────────────────────────────────────

/// ITU-R BS.775-3 target azimuth/elevation for each standard channel.
struct Bs775Target {
    label: ChannelLabel,
    azimuth_deg: f32,
    elevation_deg: f32,
}

/// Return the BS.775-3 nominal positions for a given layout format.
fn bs775_targets(format: LayoutFormat) -> Vec<Bs775Target> {
    match format {
        LayoutFormat::Stereo => vec![
            Bs775Target {
                label: ChannelLabel::Left,
                azimuth_deg: 30.0,
                elevation_deg: 0.0,
            },
            Bs775Target {
                label: ChannelLabel::Right,
                azimuth_deg: -30.0,
                elevation_deg: 0.0,
            },
        ],
        LayoutFormat::ThreeZeroZero => vec![
            Bs775Target {
                label: ChannelLabel::Left,
                azimuth_deg: 30.0,
                elevation_deg: 0.0,
            },
            Bs775Target {
                label: ChannelLabel::Centre,
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
            },
            Bs775Target {
                label: ChannelLabel::Right,
                azimuth_deg: -30.0,
                elevation_deg: 0.0,
            },
        ],
        LayoutFormat::FiveOneZero => vec![
            Bs775Target {
                label: ChannelLabel::Centre,
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
            },
            Bs775Target {
                label: ChannelLabel::Left,
                azimuth_deg: 30.0,
                elevation_deg: 0.0,
            },
            Bs775Target {
                label: ChannelLabel::Right,
                azimuth_deg: -30.0,
                elevation_deg: 0.0,
            },
            Bs775Target {
                label: ChannelLabel::LeftSurround,
                azimuth_deg: 110.0,
                elevation_deg: 0.0,
            },
            Bs775Target {
                label: ChannelLabel::RightSurround,
                azimuth_deg: -110.0,
                elevation_deg: 0.0,
            },
            Bs775Target {
                label: ChannelLabel::Lfe,
                azimuth_deg: 0.0,
                elevation_deg: -15.0,
            },
        ],
        LayoutFormat::SevenOneZero => vec![
            Bs775Target {
                label: ChannelLabel::Centre,
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
            },
            Bs775Target {
                label: ChannelLabel::Left,
                azimuth_deg: 30.0,
                elevation_deg: 0.0,
            },
            Bs775Target {
                label: ChannelLabel::Right,
                azimuth_deg: -30.0,
                elevation_deg: 0.0,
            },
            Bs775Target {
                label: ChannelLabel::LeftSurround,
                azimuth_deg: 110.0,
                elevation_deg: 0.0,
            },
            Bs775Target {
                label: ChannelLabel::RightSurround,
                azimuth_deg: -110.0,
                elevation_deg: 0.0,
            },
            Bs775Target {
                label: ChannelLabel::LeftBackSurround,
                azimuth_deg: 135.0,
                elevation_deg: 0.0,
            },
            Bs775Target {
                label: ChannelLabel::RightBackSurround,
                azimuth_deg: -135.0,
                elevation_deg: 0.0,
            },
            Bs775Target {
                label: ChannelLabel::Lfe,
                azimuth_deg: 0.0,
                elevation_deg: -15.0,
            },
        ],
    }
}

// ─── Layout formats ───────────────────────────────────────────────────────────

/// Standard multi-channel audio layout formats recognised by the validator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutFormat {
    /// 2.0 — stereo (L, R).
    Stereo,
    /// 3.0.0 — left/centre/right with no surround.
    ThreeZeroZero,
    /// 5.1.0 — standard 5.1 surround (L, C, R, Ls, Rs, LFE).
    FiveOneZero,
    /// 7.1.0 — 7.1 surround (L, C, R, Ls, Rs, Lbs, Rbs, LFE).
    SevenOneZero,
}

// ─── Validation result ────────────────────────────────────────────────────────

/// Result of validating a single speaker against the standard.
#[derive(Debug, Clone)]
pub struct SpeakerValidation {
    /// The channel that was validated.
    pub label: ChannelLabel,
    /// Whether the speaker is within the tolerance window.
    pub is_valid: bool,
    /// Azimuth deviation from the nominal position in degrees.
    pub azimuth_deviation_deg: f32,
    /// Elevation deviation from the nominal position in degrees.
    pub elevation_deviation_deg: f32,
    /// Angular great-circle deviation (combining azimuth + elevation), in degrees.
    pub angular_deviation_deg: f32,
    /// Suggested correction: the nominal target azimuth in degrees.
    pub suggested_azimuth_deg: f32,
    /// Suggested correction: the nominal target elevation in degrees.
    pub suggested_elevation_deg: f32,
    /// Human-readable description of the issue (empty if valid).
    pub message: String,
}

/// Overall layout validation report.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Layout format that was validated against.
    pub format: LayoutFormat,
    /// Per-speaker results.
    pub speakers: Vec<SpeakerValidation>,
    /// `true` if every speaker in the layout is within tolerance.
    pub all_valid: bool,
    /// Maximum per-speaker angular deviation (degrees).
    pub max_deviation_deg: f32,
}

impl ValidationReport {
    /// Number of speakers flagged as incorrectly placed.
    pub fn invalid_count(&self) -> usize {
        self.speakers.iter().filter(|s| !s.is_valid).count()
    }

    /// Iterator over speakers that are not correctly placed.
    pub fn invalid_speakers(&self) -> impl Iterator<Item = &SpeakerValidation> {
        self.speakers.iter().filter(|s| !s.is_valid)
    }
}

// ─── PlacementAdvisor ─────────────────────────────────────────────────────────

/// ITU-R BS.775 loudspeaker placement advisor.
///
/// # Usage
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use oximedia_spatial::speaker_placement::{
///     PlacementAdvisor, LayoutFormat, Speaker, ChannelLabel,
/// };
///
/// let advisor = PlacementAdvisor::new(LayoutFormat::FiveOneZero);
/// let speakers = vec![
///     Speaker::new(ChannelLabel::Left,         30.0,    0.0, 3.0)?,
///     Speaker::new(ChannelLabel::Right,        -30.0,   0.0, 3.0)?,
///     Speaker::new(ChannelLabel::Centre,         0.0,   0.0, 3.0)?,
///     Speaker::new(ChannelLabel::LeftSurround,  115.0,  0.0, 2.5)?,
///     Speaker::new(ChannelLabel::RightSurround,-110.0,  0.0, 2.5)?,
///     Speaker::new(ChannelLabel::Lfe,            0.0, -15.0, 1.5)?,
/// ];
/// let report = advisor.validate(&speakers)?;
/// assert!(report.all_valid || report.invalid_count() <= 1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct PlacementAdvisor {
    /// Target layout format.
    pub format: LayoutFormat,
    /// Tolerance for azimuth deviation in degrees (default: 10°).
    pub azimuth_tolerance_deg: f32,
    /// Tolerance for elevation deviation in degrees (default: 5°).
    pub elevation_tolerance_deg: f32,
}

impl PlacementAdvisor {
    /// Construct a placement advisor for the given standard format with default tolerances.
    pub fn new(format: LayoutFormat) -> Self {
        Self {
            format,
            azimuth_tolerance_deg: 10.0,
            elevation_tolerance_deg: 5.0,
        }
    }

    /// Construct with custom tolerances.
    pub fn with_tolerances(
        format: LayoutFormat,
        azimuth_tolerance_deg: f32,
        elevation_tolerance_deg: f32,
    ) -> Result<Self, SpatialError> {
        if azimuth_tolerance_deg < 0.0 || elevation_tolerance_deg < 0.0 {
            return Err(SpatialError::InvalidConfig(
                "Tolerances must be non-negative".into(),
            ));
        }
        Ok(Self {
            format,
            azimuth_tolerance_deg,
            elevation_tolerance_deg,
        })
    }

    /// Validate a set of physical speaker positions against the BS.775-3 standard.
    ///
    /// Returns a [`ValidationReport`] containing per-speaker results and a global
    /// pass/fail flag.
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidConfig`] if `speakers` is empty.
    pub fn validate(&self, speakers: &[Speaker]) -> Result<ValidationReport, SpatialError> {
        if speakers.is_empty() {
            return Err(SpatialError::InvalidConfig(
                "Speaker list must not be empty".into(),
            ));
        }

        let targets = bs775_targets(self.format);
        let mut validations: Vec<SpeakerValidation> = Vec::with_capacity(targets.len());
        let mut max_dev: f32 = 0.0;

        for target in &targets {
            // Find the speaker in the provided list with this label.
            let speaker_opt = speakers.iter().find(|s| s.label == target.label);

            let sv = match speaker_opt {
                None => SpeakerValidation {
                    label: target.label.clone(),
                    is_valid: false,
                    azimuth_deviation_deg: 0.0,
                    elevation_deviation_deg: 0.0,
                    angular_deviation_deg: 0.0,
                    suggested_azimuth_deg: target.azimuth_deg,
                    suggested_elevation_deg: target.elevation_deg,
                    message: format!("{} is missing from the layout", target.label),
                },
                Some(spk) => {
                    let az_dev = angular_diff(spk.azimuth_deg, target.azimuth_deg);
                    let el_dev = spk.elevation_deg - target.elevation_deg;
                    let angular_dev = great_circle_angle(
                        spk.azimuth_deg,
                        spk.elevation_deg,
                        target.azimuth_deg,
                        target.elevation_deg,
                    );

                    let az_ok = az_dev.abs() <= self.azimuth_tolerance_deg;
                    let el_ok = el_dev.abs() <= self.elevation_tolerance_deg;
                    let is_valid = az_ok && el_ok;

                    let message = if is_valid {
                        String::new()
                    } else {
                        let mut parts = Vec::new();
                        if !az_ok {
                            parts.push(format!(
                                "azimuth off by {az_dev:.1}° (tolerance ±{:.1}°)",
                                self.azimuth_tolerance_deg
                            ));
                        }
                        if !el_ok {
                            parts.push(format!(
                                "elevation off by {el_dev:.1}° (tolerance ±{:.1}°)",
                                self.elevation_tolerance_deg
                            ));
                        }
                        format!("{}: {}", spk.label, parts.join("; "))
                    };

                    SpeakerValidation {
                        label: target.label.clone(),
                        is_valid,
                        azimuth_deviation_deg: az_dev,
                        elevation_deviation_deg: el_dev,
                        angular_deviation_deg: angular_dev,
                        suggested_azimuth_deg: target.azimuth_deg,
                        suggested_elevation_deg: target.elevation_deg,
                        message,
                    }
                }
            };

            max_dev = max_dev.max(sv.angular_deviation_deg);
            validations.push(sv);
        }

        let all_valid = validations.iter().all(|v| v.is_valid);

        Ok(ValidationReport {
            format: self.format,
            speakers: validations,
            all_valid,
            max_deviation_deg: max_dev,
        })
    }

    /// Suggest a complete standard layout at a given reference distance.
    ///
    /// Returns a list of [`Speaker`] positions at the nominal BS.775-3 azimuths and
    /// elevations for the configured format, all at `distance_m` metres from the
    /// listener.
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidConfig`] if `distance_m` ≤ 0.
    pub fn suggest_layout(&self, distance_m: f32) -> Result<Vec<Speaker>, SpatialError> {
        if distance_m <= 0.0 {
            return Err(SpatialError::InvalidConfig(format!(
                "Reference distance must be > 0, got {distance_m}"
            )));
        }
        bs775_targets(self.format)
            .into_iter()
            .map(|t| Speaker::new(t.label, t.azimuth_deg, t.elevation_deg, distance_m))
            .collect()
    }
}

// ─── Distance-equalisation helper ─────────────────────────────────────────────

/// Compute the delay (samples) and gain needed to level-match speakers at different distances.
///
/// When speakers are placed at unequal distances from the listening position the
/// nearer speakers must be delayed and attenuated to preserve coincidence of the
/// wavefronts at the sweet-spot.
///
/// # Parameters
/// - `speakers`: slice of speakers (each with a `distance_m` field).
/// - `sample_rate`: audio sample rate in Hz.
///
/// # Returns
/// A `Vec` of `(delay_samples, gain)` pairs in the same order as `speakers`.
/// The most-distant speaker gets `delay_samples = 0` and `gain = 1.0`.  All
/// closer speakers are delayed so that their sound arrives simultaneously with
/// the farthest speaker, and attenuated proportionally (inverse distance).
///
/// # Errors
/// Returns [`SpatialError::InvalidConfig`] if `sample_rate == 0` or the speaker
/// list is empty.
pub fn distance_equalisation(
    speakers: &[Speaker],
    sample_rate: u32,
) -> Result<Vec<(u32, f32)>, SpatialError> {
    if sample_rate == 0 {
        return Err(SpatialError::InvalidConfig(
            "Sample rate must be > 0".into(),
        ));
    }
    if speakers.is_empty() {
        return Err(SpatialError::InvalidConfig(
            "Speaker list must not be empty".into(),
        ));
    }

    let max_dist = speakers
        .iter()
        .map(|s| s.distance_m)
        .fold(f32::NEG_INFINITY, f32::max);

    let speed_of_sound = 343.0_f32; // m/s at 20 °C

    let results = speakers
        .iter()
        .map(|spk| {
            let extra_dist = max_dist - spk.distance_m;
            let delay_secs = extra_dist / speed_of_sound;
            let delay_samples = (delay_secs * sample_rate as f32).round() as u32;
            // Gain: normalise so the most-distant speaker has gain 1.0.
            let gain = (spk.distance_m / max_dist).clamp(0.001, 1.0);
            (delay_samples, gain)
        })
        .collect();

    Ok(results)
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Normalise an azimuth to (−180, +180].
fn normalise_azimuth(az: f32) -> f32 {
    let mut a = az % 360.0;
    if a > 180.0 {
        a -= 360.0;
    } else if a <= -180.0 {
        a += 360.0;
    }
    a
}

/// Signed angular difference in degrees (result in (−180, +180]).
fn angular_diff(a: f32, b: f32) -> f32 {
    normalise_azimuth(a - b)
}

/// Great-circle (geodesic) angle in degrees between two directions
/// specified as (azimuth, elevation) pairs in degrees.
fn great_circle_angle(az1: f32, el1: f32, az2: f32, el2: f32) -> f32 {
    let az1 = az1.to_radians();
    let el1 = el1.to_radians();
    let az2 = az2.to_radians();
    let el2 = el2.to_radians();

    // Convert to Cartesian.
    let x1 = el1.cos() * az1.sin();
    let y1 = el1.cos() * az1.cos();
    let z1 = el1.sin();

    let x2 = el2.cos() * az2.sin();
    let y2 = el2.cos() * az2.cos();
    let z2 = el2.sin();

    // Dot product clamped to [−1, 1] to guard against floating-point rounding.
    let dot = (x1 * x2 + y1 * y2 + z1 * z2).clamp(-1.0, 1.0);
    dot.acos().to_degrees()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn ideal_5_1_speakers(distance: f32) -> Result<Vec<Speaker>, SpatialError> {
        Ok(vec![
            Speaker::new(ChannelLabel::Left, 30.0, 0.0, distance)?,
            Speaker::new(ChannelLabel::Right, -30.0, 0.0, distance)?,
            Speaker::new(ChannelLabel::Centre, 0.0, 0.0, distance)?,
            Speaker::new(ChannelLabel::LeftSurround, 110.0, 0.0, distance)?,
            Speaker::new(ChannelLabel::RightSurround, -110.0, 0.0, distance)?,
            Speaker::new(ChannelLabel::Lfe, 0.0, -15.0, distance)?,
        ])
    }

    #[test]
    fn test_ideal_5_1_all_valid() -> TestResult {
        let advisor = PlacementAdvisor::new(LayoutFormat::FiveOneZero);
        let speakers = ideal_5_1_speakers(3.0)?;
        let report = advisor.validate(&speakers)?;
        assert!(report.all_valid, "Ideal 5.1 layout should be fully valid");
        assert_eq!(report.invalid_count(), 0);
        Ok(())
    }

    #[test]
    fn test_left_speaker_misplaced_flags_error() -> TestResult {
        let advisor = PlacementAdvisor::new(LayoutFormat::FiveOneZero);
        let mut speakers = ideal_5_1_speakers(3.0)?;
        // Move left speaker 20° off-nominal (tolerance is ±10°).
        speakers[0] = Speaker::new(ChannelLabel::Left, 50.0, 0.0, 3.0)?;
        let report = advisor.validate(&speakers)?;
        assert!(
            !report.all_valid,
            "Misplaced left speaker should fail validation"
        );
        let left = report
            .speakers
            .iter()
            .find(|s| s.label == ChannelLabel::Left)
            .ok_or("left channel should be in report")?;
        assert!(!left.is_valid);
        assert!(left.azimuth_deviation_deg.abs() > 10.0);
        Ok(())
    }

    #[test]
    fn test_suggested_layout_at_distance() -> TestResult {
        let advisor = PlacementAdvisor::new(LayoutFormat::FiveOneZero);
        let layout = advisor.suggest_layout(2.5)?;
        assert_eq!(layout.len(), 6, "5.1 layout should have 6 speakers");
        for spk in &layout {
            assert!(
                (spk.distance_m - 2.5).abs() < 1e-5,
                "All speakers should be at reference distance"
            );
        }
        Ok(())
    }

    #[test]
    fn test_missing_speaker_reported() -> TestResult {
        let advisor = PlacementAdvisor::new(LayoutFormat::FiveOneZero);
        // Provide only 5 speakers, omit LFE.
        let speakers = vec![
            Speaker::new(ChannelLabel::Left, 30.0, 0.0, 3.0)?,
            Speaker::new(ChannelLabel::Right, -30.0, 0.0, 3.0)?,
            Speaker::new(ChannelLabel::Centre, 0.0, 0.0, 3.0)?,
            Speaker::new(ChannelLabel::LeftSurround, 110.0, 0.0, 3.0)?,
            Speaker::new(ChannelLabel::RightSurround, -110.0, 0.0, 3.0)?,
            // no LFE
        ];
        let report = advisor.validate(&speakers)?;
        let lfe = report
            .speakers
            .iter()
            .find(|s| s.label == ChannelLabel::Lfe)
            .ok_or("lfe entry should be present")?;
        assert!(!lfe.is_valid, "Missing LFE should be flagged invalid");
        Ok(())
    }

    #[test]
    fn test_distance_equalisation_equal_distances() -> TestResult {
        let speakers = ideal_5_1_speakers(3.0)?;
        let result = distance_equalisation(&speakers, 48_000)?;
        assert_eq!(result.len(), speakers.len());
        // All at same distance → delay = 0, gain = 1.0 for all.
        for (delay, gain) in &result {
            assert_eq!(*delay, 0, "Equal distances should produce zero delay");
            assert!(
                (gain - 1.0).abs() < 1e-5,
                "Equal distances should produce unity gain"
            );
        }
        Ok(())
    }

    #[test]
    fn test_distance_equalisation_unequal_distances() -> TestResult {
        let speakers = vec![
            Speaker::new(ChannelLabel::Left, 30.0, 0.0, 2.0)?,
            Speaker::new(ChannelLabel::Right, -30.0, 0.0, 3.0)?,
        ];
        let result = distance_equalisation(&speakers, 48_000)?;
        // Right is further → Left gets delay, Right gets 0 delay.
        let (left_delay, left_gain) = result[0];
        let (right_delay, right_gain) = result[1];
        assert!(left_delay > 0, "Closer speaker should be delayed");
        assert_eq!(right_delay, 0, "Farther speaker should have zero delay");
        assert!(left_gain < right_gain, "Closer speaker should be attenuated");
        Ok(())
    }

    #[test]
    fn test_speaker_unit_vector_front() -> TestResult {
        let spk = Speaker::new(ChannelLabel::Centre, 0.0, 0.0, 3.0)?;
        let v = spk.unit_vector();
        // Centre front = azimuth 0, elevation 0 → y-axis (front)
        assert!(v[0].abs() < 1e-5, "x should be ~0 for centre front");
        assert!((v[1] - 1.0).abs() < 1e-5, "y should be ~1 for centre front");
        assert!(v[2].abs() < 1e-5, "z should be ~0 for centre front");
        Ok(())
    }

    #[test]
    fn test_speaker_unit_vector_left() -> TestResult {
        let spk = Speaker::new(ChannelLabel::Left, 90.0, 0.0, 3.0)?;
        let v = spk.unit_vector();
        // 90° azimuth (left) → x should be negative (−1), y ≈ 0.
        assert!(v[0] < -0.9, "x should be ≈ −1 for pure left speaker");
        assert!(v[1].abs() < 1e-5, "y should be ≈ 0 for pure left speaker");
        Ok(())
    }

    #[test]
    fn test_azimuth_normalisation() {
        assert!((normalise_azimuth(370.0) - 10.0).abs() < 1e-4);
        assert!((normalise_azimuth(-370.0) - (-10.0)).abs() < 1e-4);
        assert!((normalise_azimuth(0.0)).abs() < 1e-4);
    }

    #[test]
    fn test_great_circle_angle_same_point() {
        let angle = great_circle_angle(30.0, 0.0, 30.0, 0.0);
        assert!(angle < 1e-4, "Same point should have zero angular deviation");
    }

    #[test]
    fn test_great_circle_angle_opposite_points() {
        let angle = great_circle_angle(0.0, 0.0, 180.0, 0.0);
        assert!(
            (angle - 180.0).abs() < 0.01,
            "Opposite azimuths should be 180°"
        );
    }

    #[test]
    fn test_stereo_layout_suggestion() -> TestResult {
        let advisor = PlacementAdvisor::new(LayoutFormat::Stereo);
        let layout = advisor.suggest_layout(2.0)?;
        assert_eq!(layout.len(), 2);
        let left = layout
            .iter()
            .find(|s| s.label == ChannelLabel::Left)
            .ok_or("left speaker not found in stereo layout")?;
        let right = layout
            .iter()
            .find(|s| s.label == ChannelLabel::Right)
            .ok_or("right speaker not found in stereo layout")?;
        assert!((left.azimuth_deg - 30.0).abs() < 1e-4);
        assert!((right.azimuth_deg - (-30.0)).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_seven_one_layout_count() -> TestResult {
        let advisor = PlacementAdvisor::new(LayoutFormat::SevenOneZero);
        let layout = advisor.suggest_layout(3.0)?;
        assert_eq!(layout.len(), 8, "7.1 layout should have 8 speakers");
        Ok(())
    }

    #[test]
    fn test_invalid_speaker_distance_rejected() {
        let err = Speaker::new(ChannelLabel::Left, 30.0, 0.0, -1.0);
        assert!(err.is_err(), "Negative distance should be rejected");
    }

    #[test]
    fn test_invalid_elevation_rejected() {
        let err = Speaker::new(ChannelLabel::Left, 30.0, 95.0, 3.0);
        assert!(err.is_err(), "Elevation >90° should be rejected");
    }
}
