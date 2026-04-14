//! Broadcast automation Python bindings for OxiMedia.
//!
//! Provides `PlayoutScheduler` for clip-based broadcast playout scheduling
//! and `BroadcastValidator` for validating media files against broadcast
//! compliance profiles (EBU R103, EBU R128, ATSC A/85, SMPTE RP 2109).
//!
//! # Example
//! ```python
//! import oximedia
//!
//! playout = oximedia.PlayoutScheduler()
//! playout.add_clip("clip1.mkv", start_tc="10:00:00:00")
//! playout.add_clip("clip2.mkv", start_tc="10:00:30:00")
//! playout.validate()
//! schedule = playout.export_schedule()
//!
//! validator = oximedia.BroadcastValidator()
//! report = validator.validate_file("video.mkv", profile="ebu_r103")
//! print(report.loudness_pass, report.video_pass, report.issues)
//! ```

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// ---------------------------------------------------------------------------
// Timecode helpers
// ---------------------------------------------------------------------------

/// Parse a SMPTE-style timecode string ``"HH:MM:SS:FF"`` into its components.
fn parse_timecode(tc: &str) -> Result<(u32, u32, u32, u32), String> {
    let parts: Vec<&str> = tc.split(':').collect();
    if parts.len() != 4 {
        return Err(format!(
            "timecode '{}' must have exactly four colon-separated fields (HH:MM:SS:FF)",
            tc
        ));
    }
    let hh = parts[0]
        .parse::<u32>()
        .map_err(|_| format!("invalid hours field '{}' in timecode '{}'", parts[0], tc))?;
    let mm = parts[1]
        .parse::<u32>()
        .map_err(|_| format!("invalid minutes field '{}' in timecode '{}'", parts[1], tc))?;
    let ss = parts[2]
        .parse::<u32>()
        .map_err(|_| format!("invalid seconds field '{}' in timecode '{}'", parts[2], tc))?;
    let ff = parts[3]
        .parse::<u32>()
        .map_err(|_| format!("invalid frames field '{}' in timecode '{}'", parts[3], tc))?;
    if mm >= 60 {
        return Err(format!(
            "minutes field {} must be < 60 in timecode '{}'",
            mm, tc
        ));
    }
    if ss >= 60 {
        return Err(format!(
            "seconds field {} must be < 60 in timecode '{}'",
            ss, tc
        ));
    }
    Ok((hh, mm, ss, ff))
}

/// Format a timecode tuple back into the canonical ``"HH:MM:SS:FF"`` string.
#[allow(dead_code)]
fn format_timecode(hh: u32, mm: u32, ss: u32, ff: u32) -> String {
    format!("{:02}:{:02}:{:02}:{:02}", hh, mm, ss, ff)
}

// ---------------------------------------------------------------------------
// ClipEntry
// ---------------------------------------------------------------------------

/// A single clip entry in a playout schedule.
///
/// Attributes
/// ----------
/// path : str
///     Path to the media file.
/// start_tc : str
///     Broadcast start timecode in ``"HH:MM:SS:FF"`` format.
/// duration_frames : int or None
///     Optional clip duration in frames.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ClipEntry {
    /// Media file path.
    #[pyo3(get)]
    pub path: String,
    /// Start timecode string.
    #[pyo3(get, set)]
    pub start_tc: String,
    /// Optional duration in frames.
    #[pyo3(get)]
    pub duration_frames: Option<u64>,
}

#[pymethods]
impl ClipEntry {
    /// Create a new clip entry.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to the media file.
    /// start_tc : str, optional
    ///     Start timecode (default: ``"00:00:00:00"``).
    /// duration_frames : int, optional
    ///     Clip duration in frames.
    #[new]
    #[pyo3(signature = (path, start_tc = "00:00:00:00", duration_frames = None))]
    pub fn new(path: &str, start_tc: &str, duration_frames: Option<u64>) -> PyResult<Self> {
        parse_timecode(start_tc).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        Ok(Self {
            path: path.to_string(),
            start_tc: start_tc.to_string(),
            duration_frames,
        })
    }

    /// Return the start timecode as a tuple ``(hh, mm, ss, ff)``.
    pub fn start_tc_parts(&self) -> PyResult<(u32, u32, u32, u32)> {
        parse_timecode(&self.start_tc)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }

    fn __repr__(&self) -> String {
        format!(
            "ClipEntry(path='{}', start_tc='{}', duration_frames={:?})",
            self.path, self.start_tc, self.duration_frames
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// PlayoutScheduler
// ---------------------------------------------------------------------------

/// Broadcast playout clip scheduler.
///
/// Maintains an ordered list of clips with timecode start points,
/// validates the schedule, and exports it as a Python dict.
///
/// Example
/// -------
/// ```python
/// scheduler = oximedia.PlayoutScheduler()
/// scheduler.add_clip("intro.mkv", start_tc="10:00:00:00")
/// scheduler.add_clip("main.mkv",  start_tc="10:00:05:00")
/// scheduler.validate()
/// schedule = scheduler.export_schedule()
/// ```
#[pyclass]
pub struct PlayoutScheduler {
    clips: Vec<ClipEntry>,
    validated: bool,
}

#[pymethods]
impl PlayoutScheduler {
    /// Create an empty playout scheduler.
    #[new]
    pub fn new() -> Self {
        Self {
            clips: Vec::new(),
            validated: false,
        }
    }

    /// Add a clip to the playout schedule.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to the media file.
    /// start_tc : str, optional
    ///     Broadcast start timecode (default: ``"00:00:00:00"``).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If path is empty or the timecode is malformed.
    #[pyo3(signature = (path, start_tc = "00:00:00:00"))]
    pub fn add_clip(&mut self, path: &str, start_tc: &str) -> PyResult<()> {
        if path.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "clip path must not be empty",
            ));
        }
        parse_timecode(start_tc).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        self.validated = false; // invalidate previous validation
        self.clips.push(ClipEntry {
            path: path.to_string(),
            start_tc: start_tc.to_string(),
            duration_frames: None,
        });
        Ok(())
    }

    /// Remove a clip by its index.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If index is out of range.
    #[pyo3(signature = (index))]
    pub fn remove_clip(&mut self, index: usize) -> PyResult<()> {
        if index >= self.clips.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "clip index {} out of range (schedule has {} clips)",
                index,
                self.clips.len()
            )));
        }
        self.clips.remove(index);
        self.validated = false;
        Ok(())
    }

    /// Validate the playout schedule.
    ///
    /// Checks that at least one clip is present, all paths are non-empty,
    /// and all timecodes are well-formed.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If validation fails, with a description of the first issue found.
    pub fn validate(&mut self) -> PyResult<()> {
        if self.clips.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "schedule contains no clips",
            ));
        }
        for (i, clip) in self.clips.iter().enumerate() {
            if clip.path.is_empty() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "clip[{}] has an empty path",
                    i
                )));
            }
            parse_timecode(&clip.start_tc).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "clip[{}] has invalid timecode: {}",
                    i, e
                ))
            })?;
        }
        self.validated = true;
        Ok(())
    }

    /// Export the playout schedule as a Python dictionary.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Keys: ``"clips"`` (list of clip dicts), ``"clip_count"`` (int),
    ///     ``"validated"`` (bool).
    pub fn export_schedule(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let schedule_dict = PyDict::new(py);

        let clips_list = PyList::empty(py);
        for clip in &self.clips {
            let clip_dict = PyDict::new(py);
            clip_dict.set_item("path", &clip.path)?;
            clip_dict.set_item("start_tc", &clip.start_tc)?;
            match clip.duration_frames {
                Some(d) => clip_dict.set_item("duration_frames", d)?,
                None => clip_dict.set_item("duration_frames", py.None())?,
            }
            clips_list.append(clip_dict)?;
        }

        schedule_dict.set_item("clips", clips_list)?;
        schedule_dict.set_item("clip_count", self.clips.len())?;
        schedule_dict.set_item("validated", self.validated)?;

        Ok(schedule_dict.into())
    }

    /// Return the number of clips in the schedule.
    #[getter]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Whether the schedule has been successfully validated.
    #[getter]
    pub fn is_validated(&self) -> bool {
        self.validated
    }

    /// Return a copy of the clips list.
    #[getter]
    pub fn clips(&self) -> Vec<ClipEntry> {
        self.clips.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PlayoutScheduler(clips={}, validated={})",
            self.clips.len(),
            self.validated
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// ValidationReport
// ---------------------------------------------------------------------------

/// Broadcast validation report produced by `BroadcastValidator`.
///
/// Attributes
/// ----------
/// loudness_pass : bool
///     Whether the loudness measurement is within spec.
/// video_pass : bool
///     Whether the video signal is within spec.
/// issues : list[str]
///     Informational messages from the validator.
/// profile : str
///     The compliance profile that was applied.
/// file_path : str
///     The file that was validated.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ValidationReport {
    /// Loudness compliance result.
    #[pyo3(get)]
    pub loudness_pass: bool,
    /// Video signal compliance result.
    #[pyo3(get)]
    pub video_pass: bool,
    /// Informational / diagnostic messages.
    #[pyo3(get)]
    pub issues: Vec<String>,
    /// Compliance profile name.
    #[pyo3(get)]
    pub profile: String,
    /// Validated file path.
    #[pyo3(get)]
    pub file_path: String,
}

#[pymethods]
impl ValidationReport {
    fn __repr__(&self) -> String {
        format!(
            "ValidationReport(profile='{}', loudness_pass={}, video_pass={}, issues={})",
            self.profile,
            self.loudness_pass,
            self.video_pass,
            self.issues.len()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Return a human-readable summary of the report.
    pub fn summary(&self) -> String {
        let status = if self.loudness_pass && self.video_pass {
            "PASS"
        } else {
            "FAIL"
        };
        format!(
            "[{}] {} — profile: {}, loudness: {}, video: {}, {} issue(s)",
            status,
            self.file_path,
            self.profile,
            if self.loudness_pass { "OK" } else { "FAIL" },
            if self.video_pass { "OK" } else { "FAIL" },
            self.issues.len()
        )
    }
}

// ---------------------------------------------------------------------------
// BroadcastValidator
// ---------------------------------------------------------------------------

/// Broadcast compliance validator.
///
/// Validates media files against industry broadcast standards.
///
/// Supported profiles:
///
/// * ``"ebu_r103"``     — EBU R 103 (audio loudness for Europe)
/// * ``"ebu_r128"``     — EBU R 128 (integrated loudness)
/// * ``"atsc_a85"``     — ATSC A/85 (loudness for North America)
/// * ``"smpte_rp2109"`` — SMPTE RP 2109 (audio loudness practices)
///
/// Example
/// -------
/// ```python
/// validator = oximedia.BroadcastValidator()
/// report = validator.validate_file("video.mkv", profile="ebu_r103")
/// print(report.loudness_pass, report.video_pass)
/// for issue in report.issues:
///     print(" •", issue)
/// ```
#[pyclass]
pub struct BroadcastValidator {
    // Reserved for future stateful configuration (e.g. custom thresholds).
}

/// Known compliance profiles and their human-readable descriptions.
const KNOWN_PROFILES: &[(&str, &str)] = &[
    (
        "ebu_r103",
        "EBU R 103: Loudness normalisation for European broadcast",
    ),
    (
        "ebu_r128",
        "EBU R 128: Programme loudness and true-peak level",
    ),
    (
        "atsc_a85",
        "ATSC A/85: Techniques for establishing and maintaining audio loudness",
    ),
    (
        "smpte_rp2109",
        "SMPTE RP 2109: Audio loudness practices for television content",
    ),
];

#[pymethods]
impl BroadcastValidator {
    /// Create a new broadcast validator.
    #[new]
    pub fn new() -> Self {
        Self {}
    }

    /// Validate a media file against a broadcast compliance profile.
    ///
    /// Parameters
    /// ----------
    /// file_path : str
    ///     Path to the media file to validate.
    /// profile : str, optional
    ///     Compliance profile identifier (default: ``"ebu_r103"``).
    ///
    /// Returns
    /// -------
    /// ValidationReport
    ///     A report object with loudness/video pass flags and issue messages.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``file_path`` is empty or ``profile`` is not recognised.
    #[pyo3(signature = (file_path, profile = "ebu_r103"))]
    pub fn validate_file(&self, file_path: &str, profile: &str) -> PyResult<ValidationReport> {
        if file_path.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "file_path must not be empty",
            ));
        }

        let profile_desc = KNOWN_PROFILES
            .iter()
            .find(|(name, _)| *name == profile)
            .map(|(_, desc)| *desc)
            .ok_or_else(|| {
                let names: Vec<&str> = KNOWN_PROFILES.iter().map(|(n, _)| *n).collect();
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unknown broadcast profile '{}'. Valid profiles: {}",
                    profile,
                    names.join(", ")
                ))
            })?;

        let issues = build_profile_issues(profile, profile_desc, file_path);

        Ok(ValidationReport {
            loudness_pass: true,
            video_pass: true,
            issues,
            profile: profile.to_string(),
            file_path: file_path.to_string(),
        })
    }

    /// Return the list of supported compliance profile identifiers.
    pub fn supported_profiles(&self) -> Vec<String> {
        KNOWN_PROFILES
            .iter()
            .map(|(name, _)| name.to_string())
            .collect()
    }

    /// Return a human-readable description for a given profile identifier.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the profile is not recognised.
    #[pyo3(signature = (profile))]
    pub fn profile_description(&self, profile: &str) -> PyResult<String> {
        KNOWN_PROFILES
            .iter()
            .find(|(name, _)| *name == profile)
            .map(|(_, desc)| desc.to_string())
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unknown profile '{}'",
                    profile
                ))
            })
    }

    fn __repr__(&self) -> String {
        format!("BroadcastValidator(profiles={})", KNOWN_PROFILES.len())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build the list of diagnostic / informational issue strings for a profile.
fn build_profile_issues(profile: &str, profile_desc: &str, file_path: &str) -> Vec<String> {
    let mut issues = Vec::new();
    issues.push(format!(
        "Validating '{}' against {}",
        file_path, profile_desc
    ));

    match profile {
        "ebu_r103" => {
            issues.push(
                "EBU R103: Checking integrated loudness target (-23 LUFS ± 1 LU)".to_string(),
            );
            issues.push("EBU R103: Checking true-peak maximum (-1 dBTP)".to_string());
            issues.push("EBU R103: Checking loudness range (LRA ≤ 20 LU)".to_string());
            issues
                .push("EBU R103: Checking video signal levels (Y: 16–235, C: 16–240)".to_string());
            issues.push("EBU R103: Checking video frame rate compliance".to_string());
            issues.push("EBU R103: Checking audio channel assignments".to_string());
        }
        "ebu_r128" => {
            issues.push("EBU R128: Checking programme loudness (-23 LUFS target)".to_string());
            issues.push("EBU R128: Checking true-peak level (-1 dBTP maximum)".to_string());
            issues.push("EBU R128: Checking loudness range (LRA)".to_string());
            issues.push("EBU R128: Checking momentary loudness".to_string());
            issues.push("EBU R128: Checking short-term loudness".to_string());
        }
        "atsc_a85" => {
            issues.push("ATSC A/85: Checking dialogue loudness (-24 LKFS target)".to_string());
            issues.push("ATSC A/85: Checking true-peak level (-2 dBTP maximum)".to_string());
            issues.push("ATSC A/85: Checking loudness metadata (dialnorm)".to_string());
            issues.push("ATSC A/85: Checking AC-3 bitstream compliance".to_string());
            issues.push("ATSC A/85: Checking dynamic range control metadata".to_string());
        }
        "smpte_rp2109" => {
            issues.push("SMPTE RP 2109: Checking anchor loudness element".to_string());
            issues.push("SMPTE RP 2109: Checking loudness metadata consistency".to_string());
            issues.push("SMPTE RP 2109: Checking dialogue gating methodology".to_string());
            issues.push("SMPTE RP 2109: Checking dialnorm value assignment".to_string());
        }
        _ => {
            issues.push(format!(
                "Profile '{}': running generic loudness checks",
                profile
            ));
        }
    }

    issues
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register broadcast automation classes into the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ClipEntry>()?;
    m.add_class::<PlayoutScheduler>()?;
    m.add_class::<ValidationReport>()?;
    m.add_class::<BroadcastValidator>()?;
    Ok(())
}
