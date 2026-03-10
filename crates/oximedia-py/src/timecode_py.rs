//! Python bindings for `oximedia-timecode`.
//!
//! Provides `PyFrameRate`, `PyTimecode`, and `PyTimecodeRange` classes along
//! with convenience functions for timecode parsing and conversion.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use oximedia_timecode::{FrameRate, Timecode, TimecodeError};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a `TimecodeError` into a Python `RuntimeError`.
fn tc_err(e: TimecodeError) -> PyErr {
    PyRuntimeError::new_err(format!("{e}"))
}

/// Resolve a floating-point fps value to the corresponding `FrameRate` enum
/// variant, optionally taking a drop-frame flag.
fn fps_to_frame_rate(fps: f64, drop_frame: Option<bool>) -> PyResult<FrameRate> {
    let df = drop_frame.unwrap_or(false);

    // Round to two decimals to tolerate minor floating-point noise.
    let rounded = (fps * 1000.0).round() / 1000.0;

    match rounded {
        v if (v - 23.976).abs() < 0.01 => Ok(FrameRate::Fps23976),
        v if (v - 24.0).abs() < 0.01 => Ok(FrameRate::Fps24),
        v if (v - 25.0).abs() < 0.01 => Ok(FrameRate::Fps25),
        v if (v - 29.97).abs() < 0.01 => {
            if df {
                Ok(FrameRate::Fps2997DF)
            } else {
                Ok(FrameRate::Fps2997NDF)
            }
        }
        v if (v - 30.0).abs() < 0.01 => Ok(FrameRate::Fps30),
        v if (v - 50.0).abs() < 0.01 => Ok(FrameRate::Fps50),
        v if (v - 59.94).abs() < 0.01 => Ok(FrameRate::Fps5994),
        v if (v - 60.0).abs() < 0.01 => Ok(FrameRate::Fps60),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported frame rate: {fps}. Supported: 23.976, 24, 25, 29.97, 30, 50, 59.94, 60"
        ))),
    }
}

/// Human-readable name for a `FrameRate`.
fn frame_rate_name(fr: FrameRate) -> String {
    match fr {
        FrameRate::Fps23976 => "23.976 fps".to_string(),
        FrameRate::Fps24 => "24 fps".to_string(),
        FrameRate::Fps25 => "25 fps (PAL)".to_string(),
        FrameRate::Fps2997DF => "29.97 fps DF (NTSC)".to_string(),
        FrameRate::Fps2997NDF => "29.97 fps NDF (NTSC)".to_string(),
        FrameRate::Fps30 => "30 fps".to_string(),
        FrameRate::Fps50 => "50 fps".to_string(),
        FrameRate::Fps5994 => "59.94 fps".to_string(),
        FrameRate::Fps60 => "60 fps".to_string(),
    }
}

/// Parse a timecode string in `HH:MM:SS:FF` or `HH:MM:SS;FF` format.
/// The `;` separator indicates drop-frame.
fn parse_tc_string(tc_str: &str, frame_rate: FrameRate) -> PyResult<Timecode> {
    let cleaned = tc_str.trim();
    if cleaned.len() < 11 {
        return Err(PyValueError::new_err(format!(
            "Invalid timecode string: '{tc_str}'. Expected HH:MM:SS:FF or HH:MM:SS;FF"
        )));
    }

    let parts: Vec<&str> = cleaned.split([':', ';']).collect();
    if parts.len() != 4 {
        return Err(PyValueError::new_err(format!(
            "Invalid timecode string: '{tc_str}'. Expected 4 parts separated by : or ;"
        )));
    }

    let hours: u8 = parts[0]
        .parse()
        .map_err(|_| PyValueError::new_err(format!("Invalid hours in timecode: '{}'", parts[0])))?;
    let minutes: u8 = parts[1].parse().map_err(|_| {
        PyValueError::new_err(format!("Invalid minutes in timecode: '{}'", parts[1]))
    })?;
    let seconds: u8 = parts[2].parse().map_err(|_| {
        PyValueError::new_err(format!("Invalid seconds in timecode: '{}'", parts[2]))
    })?;
    let frames: u8 = parts[3].parse().map_err(|_| {
        PyValueError::new_err(format!("Invalid frames in timecode: '{}'", parts[3]))
    })?;

    // If the string uses ';' as last separator and the frame rate supports DF,
    // use the DF variant.
    let effective_rate = if cleaned.contains(';') && !frame_rate.is_drop_frame() {
        // Promote to DF if the rate is 29.97-family.
        match frame_rate {
            FrameRate::Fps2997NDF => FrameRate::Fps2997DF,
            other => other,
        }
    } else {
        frame_rate
    };

    Timecode::new(hours, minutes, seconds, frames, effective_rate).map_err(tc_err)
}

/// Build a `PyFrameRate` from a `FrameRate` enum value.
fn build_py_frame_rate(fr: FrameRate) -> PyFrameRate {
    PyFrameRate {
        fps: fr.as_float(),
        is_drop_frame: fr.is_drop_frame(),
        name: frame_rate_name(fr),
        inner: fr,
    }
}

/// Build a `PyTimecode` from a `Timecode` struct and its originating `FrameRate`.
fn build_py_timecode(tc: Timecode, fr: FrameRate) -> PyTimecode {
    PyTimecode {
        hours: tc.hours,
        minutes: tc.minutes,
        seconds: tc.seconds,
        frames: tc.frames,
        is_drop_frame: tc.frame_rate.drop_frame,
        inner: tc,
        frame_rate: fr,
    }
}

// ---------------------------------------------------------------------------
// PyFrameRate
// ---------------------------------------------------------------------------

/// Frame rate representation for SMPTE timecodes.
#[pyclass]
#[derive(Clone)]
pub struct PyFrameRate {
    /// Nominal frame rate as a float (e.g. 23.976, 29.97).
    #[pyo3(get)]
    pub fps: f64,
    /// Whether this is a drop-frame rate.
    #[pyo3(get)]
    pub is_drop_frame: bool,
    /// Human-readable description.
    #[pyo3(get)]
    pub name: String,
    inner: FrameRate,
}

#[pymethods]
impl PyFrameRate {
    /// 23.976 fps (film transferred to NTSC).
    #[classmethod]
    #[allow(unused_variables)]
    fn fps_23_976(cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        build_py_frame_rate(FrameRate::Fps23976)
    }

    /// 24 fps (film).
    #[classmethod]
    #[allow(unused_variables)]
    fn fps_24(cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        build_py_frame_rate(FrameRate::Fps24)
    }

    /// 25 fps (PAL).
    #[classmethod]
    #[allow(unused_variables)]
    fn fps_25(cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        build_py_frame_rate(FrameRate::Fps25)
    }

    /// 29.97 fps drop frame (NTSC).
    #[classmethod]
    #[allow(unused_variables)]
    fn fps_29_97_df(cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        build_py_frame_rate(FrameRate::Fps2997DF)
    }

    /// 29.97 fps non-drop frame (NTSC).
    #[classmethod]
    #[allow(unused_variables)]
    fn fps_29_97_ndf(cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        build_py_frame_rate(FrameRate::Fps2997NDF)
    }

    /// 30 fps.
    #[classmethod]
    #[allow(unused_variables)]
    fn fps_30(cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        build_py_frame_rate(FrameRate::Fps30)
    }

    /// 50 fps (PAL progressive).
    #[classmethod]
    #[allow(unused_variables)]
    fn fps_50(cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        build_py_frame_rate(FrameRate::Fps50)
    }

    /// 59.94 fps (NTSC progressive).
    #[classmethod]
    #[allow(unused_variables)]
    fn fps_59_94(cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        build_py_frame_rate(FrameRate::Fps5994)
    }

    /// 60 fps.
    #[classmethod]
    #[allow(unused_variables)]
    fn fps_60(cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        build_py_frame_rate(FrameRate::Fps60)
    }

    /// Create from a floating-point value, optionally specifying drop-frame.
    #[classmethod]
    #[allow(unused_variables)]
    fn from_float(
        cls: &Bound<'_, pyo3::types::PyType>,
        fps: f64,
        drop_frame: Option<bool>,
    ) -> PyResult<Self> {
        let fr = fps_to_frame_rate(fps, drop_frame)?;
        Ok(build_py_frame_rate(fr))
    }

    /// Get the exact rational representation as (numerator, denominator).
    fn as_rational(&self) -> (u32, u32) {
        self.inner.as_rational()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyFrameRate(fps={}, drop_frame={})",
            self.fps, self.is_drop_frame
        )
    }

    fn __eq__(&self, other: &PyFrameRate) -> bool {
        self.inner == other.inner
    }
}

// ---------------------------------------------------------------------------
// PyTimecode
// ---------------------------------------------------------------------------

/// SMPTE timecode representation.
#[pyclass]
#[derive(Clone)]
pub struct PyTimecode {
    /// Hours component (0-23).
    #[pyo3(get)]
    pub hours: u8,
    /// Minutes component (0-59).
    #[pyo3(get)]
    pub minutes: u8,
    /// Seconds component (0-59).
    #[pyo3(get)]
    pub seconds: u8,
    /// Frames component (0 to fps-1).
    #[pyo3(get)]
    pub frames: u8,
    /// Whether this timecode uses drop-frame counting.
    #[pyo3(get)]
    pub is_drop_frame: bool,
    inner: Timecode,
    frame_rate: FrameRate,
}

#[pymethods]
impl PyTimecode {
    /// Create a new timecode from individual components.
    #[new]
    fn new(
        hours: u8,
        minutes: u8,
        seconds: u8,
        frames: u8,
        frame_rate: &PyFrameRate,
    ) -> PyResult<Self> {
        let tc =
            Timecode::new(hours, minutes, seconds, frames, frame_rate.inner).map_err(tc_err)?;
        Ok(build_py_timecode(tc, frame_rate.inner))
    }

    /// Parse a timecode from a string like "01:02:03:04" or "01:02:03;04".
    #[classmethod]
    #[allow(unused_variables)]
    fn from_string(
        cls: &Bound<'_, pyo3::types::PyType>,
        tc_str: &str,
        frame_rate: &PyFrameRate,
    ) -> PyResult<Self> {
        let tc = parse_tc_string(tc_str, frame_rate.inner)?;
        Ok(build_py_timecode(tc, frame_rate.inner))
    }

    /// Create a timecode from an absolute frame count.
    #[classmethod]
    #[allow(unused_variables)]
    fn from_frames(
        cls: &Bound<'_, pyo3::types::PyType>,
        frames: u64,
        frame_rate: &PyFrameRate,
    ) -> PyResult<Self> {
        let tc = Timecode::from_frames(frames, frame_rate.inner).map_err(tc_err)?;
        Ok(build_py_timecode(tc, frame_rate.inner))
    }

    /// Create a timecode from a seconds value.
    #[classmethod]
    #[allow(unused_variables)]
    fn from_seconds(
        cls: &Bound<'_, pyo3::types::PyType>,
        seconds: f64,
        frame_rate: &PyFrameRate,
    ) -> PyResult<Self> {
        let fps = frame_rate.inner.as_float();
        let total_frames = (seconds * fps).round() as u64;
        let tc = Timecode::from_frames(total_frames, frame_rate.inner).map_err(tc_err)?;
        Ok(build_py_timecode(tc, frame_rate.inner))
    }

    /// Convert to absolute frame count since 00:00:00:00.
    fn to_frames(&self) -> u64 {
        self.inner.to_frames()
    }

    /// Convert to seconds.
    fn to_seconds(&self) -> f64 {
        let frames = self.inner.to_frames();
        let (num, den) = self.frame_rate.as_rational();
        if num == 0 {
            return 0.0;
        }
        frames as f64 * den as f64 / num as f64
    }

    /// Format as a string: HH:MM:SS:FF or HH:MM:SS;FF for drop frame.
    #[pyo3(name = "to_string")]
    fn to_string_py(&self) -> String {
        format!("{}", self.inner)
    }

    /// Add another timecode (frame counts are summed).
    fn add(&self, other: &PyTimecode) -> PyResult<PyTimecode> {
        let total = self.inner.to_frames() + other.inner.to_frames();
        let tc = Timecode::from_frames(total, self.frame_rate).map_err(tc_err)?;
        Ok(build_py_timecode(tc, self.frame_rate))
    }

    /// Subtract another timecode.
    fn subtract(&self, other: &PyTimecode) -> PyResult<PyTimecode> {
        let self_frames = self.inner.to_frames();
        let other_frames = other.inner.to_frames();
        if other_frames > self_frames {
            return Err(PyValueError::new_err(
                "Cannot subtract: result would be negative",
            ));
        }
        let total = self_frames - other_frames;
        let tc = Timecode::from_frames(total, self.frame_rate).map_err(tc_err)?;
        Ok(build_py_timecode(tc, self.frame_rate))
    }

    /// Add (or subtract, if negative) a number of frames.
    fn add_frames(&self, frames: i64) -> PyResult<PyTimecode> {
        let current = self.inner.to_frames() as i64;
        let new_frames = current + frames;
        if new_frames < 0 {
            return Err(PyValueError::new_err(
                "Cannot add frames: result would be negative",
            ));
        }
        let tc = Timecode::from_frames(new_frames as u64, self.frame_rate).map_err(tc_err)?;
        Ok(build_py_timecode(tc, self.frame_rate))
    }

    /// Get the frame rate of this timecode.
    #[pyo3(name = "frame_rate")]
    fn frame_rate_py(&self) -> PyFrameRate {
        build_py_frame_rate(self.frame_rate)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyTimecode('{}', fps={}, drop_frame={})",
            self.inner,
            self.frame_rate.as_float(),
            self.is_drop_frame
        )
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __eq__(&self, other: &PyTimecode) -> bool {
        self.inner.to_frames() == other.inner.to_frames() && self.frame_rate == other.frame_rate
    }

    fn __add__(&self, other: &PyTimecode) -> PyResult<PyTimecode> {
        self.add(other)
    }

    fn __sub__(&self, other: &PyTimecode) -> PyResult<PyTimecode> {
        self.subtract(other)
    }
}

// ---------------------------------------------------------------------------
// PyTimecodeRange
// ---------------------------------------------------------------------------

/// A range between two timecodes.
#[pyclass]
#[derive(Clone)]
pub struct PyTimecodeRange {
    /// Start timecode as a formatted string.
    #[pyo3(get)]
    pub start_tc: String,
    /// End timecode as a formatted string.
    #[pyo3(get)]
    pub end_tc: String,
    /// Duration in frames.
    #[pyo3(get)]
    pub duration_frames: u64,
    start: Timecode,
    end: Timecode,
    frame_rate: FrameRate,
}

#[pymethods]
impl PyTimecodeRange {
    /// Create a new range from two PyTimecodes.
    #[new]
    fn new(start: &PyTimecode, end: &PyTimecode) -> PyResult<Self> {
        let start_frames = start.inner.to_frames();
        let end_frames = end.inner.to_frames();
        if end_frames < start_frames {
            return Err(PyValueError::new_err(
                "End timecode must be >= start timecode",
            ));
        }
        Ok(Self {
            start_tc: format!("{}", start.inner),
            end_tc: format!("{}", end.inner),
            duration_frames: end_frames - start_frames,
            start: start.inner,
            end: end.inner,
            frame_rate: start.frame_rate,
        })
    }

    /// Duration in seconds.
    fn duration_seconds(&self) -> f64 {
        let (num, den) = self.frame_rate.as_rational();
        if num == 0 {
            return 0.0;
        }
        self.duration_frames as f64 * den as f64 / num as f64
    }

    /// Check whether a timecode falls within this range (inclusive).
    fn contains(&self, tc: &PyTimecode) -> bool {
        let f = tc.inner.to_frames();
        f >= self.start.to_frames() && f <= self.end.to_frames()
    }

    /// Compute the overlap between two ranges, if any.
    fn overlap(&self, other: &PyTimecodeRange) -> PyResult<Option<PyTimecodeRange>> {
        let s1 = self.start.to_frames();
        let e1 = self.end.to_frames();
        let s2 = other.start.to_frames();
        let e2 = other.end.to_frames();

        let overlap_start = s1.max(s2);
        let overlap_end = e1.min(e2);

        if overlap_start > overlap_end {
            return Ok(None);
        }

        let start_tc = Timecode::from_frames(overlap_start, self.frame_rate).map_err(tc_err)?;
        let end_tc = Timecode::from_frames(overlap_end, self.frame_rate).map_err(tc_err)?;

        Ok(Some(PyTimecodeRange {
            start_tc: format!("{start_tc}"),
            end_tc: format!("{end_tc}"),
            duration_frames: overlap_end - overlap_start,
            start: start_tc,
            end: end_tc,
            frame_rate: self.frame_rate,
        }))
    }

    fn __repr__(&self) -> String {
        format!(
            "PyTimecodeRange('{}' - '{}', {} frames)",
            self.start_tc, self.end_tc, self.duration_frames
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Parse a timecode string at the given frame rate.
#[pyfunction]
#[pyo3(signature = (tc_str, fps, drop_frame = None))]
pub fn parse_timecode(tc_str: &str, fps: f64, drop_frame: Option<bool>) -> PyResult<PyTimecode> {
    let fr = fps_to_frame_rate(fps, drop_frame)?;
    let tc = parse_tc_string(tc_str, fr)?;
    Ok(build_py_timecode(tc, fr))
}

/// Convert an absolute frame count to a timecode.
#[pyfunction]
#[pyo3(signature = (frames, fps, drop_frame = None))]
pub fn frames_to_timecode(frames: u64, fps: f64, drop_frame: Option<bool>) -> PyResult<PyTimecode> {
    let fr = fps_to_frame_rate(fps, drop_frame)?;
    let tc = Timecode::from_frames(frames, fr).map_err(tc_err)?;
    Ok(build_py_timecode(tc, fr))
}

/// Parse a timecode string and return the equivalent time in seconds.
#[pyfunction]
#[pyo3(signature = (tc_str, fps, drop_frame = None))]
pub fn timecode_to_seconds(tc_str: &str, fps: f64, drop_frame: Option<bool>) -> PyResult<f64> {
    let fr = fps_to_frame_rate(fps, drop_frame)?;
    let tc = parse_tc_string(tc_str, fr)?;
    let total_frames = tc.to_frames();
    let (num, den) = fr.as_rational();
    if num == 0 {
        return Ok(0.0);
    }
    Ok(total_frames as f64 * den as f64 / num as f64)
}

/// Convert a seconds value to a timecode.
#[pyfunction]
#[pyo3(signature = (seconds, fps, drop_frame = None))]
pub fn seconds_to_timecode(
    seconds: f64,
    fps: f64,
    drop_frame: Option<bool>,
) -> PyResult<PyTimecode> {
    let fr = fps_to_frame_rate(fps, drop_frame)?;
    let fps_float = fr.as_float();
    let total_frames = (seconds * fps_float).round() as u64;
    let tc = Timecode::from_frames(total_frames, fr).map_err(tc_err)?;
    Ok(build_py_timecode(tc, fr))
}

/// List all supported frame rates as `PyFrameRate` objects.
#[pyfunction]
pub fn list_frame_rates() -> Vec<PyFrameRate> {
    vec![
        build_py_frame_rate(FrameRate::Fps23976),
        build_py_frame_rate(FrameRate::Fps24),
        build_py_frame_rate(FrameRate::Fps25),
        build_py_frame_rate(FrameRate::Fps2997DF),
        build_py_frame_rate(FrameRate::Fps2997NDF),
        build_py_frame_rate(FrameRate::Fps30),
        build_py_frame_rate(FrameRate::Fps50),
        build_py_frame_rate(FrameRate::Fps5994),
        build_py_frame_rate(FrameRate::Fps60),
    ]
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register timecode types and functions on the parent module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFrameRate>()?;
    m.add_class::<PyTimecode>()?;
    m.add_class::<PyTimecodeRange>()?;
    m.add_function(wrap_pyfunction!(parse_timecode, m)?)?;
    m.add_function(wrap_pyfunction!(frames_to_timecode, m)?)?;
    m.add_function(wrap_pyfunction!(timecode_to_seconds, m)?)?;
    m.add_function(wrap_pyfunction!(seconds_to_timecode, m)?)?;
    m.add_function(wrap_pyfunction!(list_frame_rates, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fps_to_frame_rate_basic() {
        let fr = fps_to_frame_rate(25.0, None);
        assert!(fr.is_ok());
        assert_eq!(fr.ok(), Some(FrameRate::Fps25));
    }

    #[test]
    fn test_fps_to_frame_rate_drop_frame() {
        let fr = fps_to_frame_rate(29.97, Some(true));
        assert!(fr.is_ok());
        assert_eq!(fr.ok(), Some(FrameRate::Fps2997DF));

        let fr_ndf = fps_to_frame_rate(29.97, Some(false));
        assert!(fr_ndf.is_ok());
        assert_eq!(fr_ndf.ok(), Some(FrameRate::Fps2997NDF));
    }

    #[test]
    fn test_fps_to_frame_rate_unsupported() {
        let fr = fps_to_frame_rate(15.0, None);
        assert!(fr.is_err());
    }

    #[test]
    fn test_parse_tc_string_ndf() {
        let tc = parse_tc_string("01:02:03:04", FrameRate::Fps25);
        assert!(tc.is_ok());
        let tc = tc.expect("parse should succeed");
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 2);
        assert_eq!(tc.seconds, 3);
        assert_eq!(tc.frames, 4);
        assert!(!tc.frame_rate.drop_frame);
    }

    #[test]
    fn test_parse_tc_string_df() {
        let tc = parse_tc_string("01:00:00;02", FrameRate::Fps2997DF);
        assert!(tc.is_ok());
        let tc = tc.expect("parse should succeed");
        assert!(tc.frame_rate.drop_frame);
    }

    #[test]
    fn test_build_py_frame_rate() {
        let pfr = build_py_frame_rate(FrameRate::Fps24);
        assert!((pfr.fps - 24.0).abs() < f64::EPSILON);
        assert!(!pfr.is_drop_frame);
    }

    #[test]
    fn test_timecode_to_frames_roundtrip() {
        let fr = FrameRate::Fps25;
        let tc = Timecode::new(1, 0, 0, 0, fr);
        assert!(tc.is_ok());
        let tc = tc.expect("timecode should be valid");
        let frames = tc.to_frames();
        // 1 hour at 25 fps = 90000 frames
        assert_eq!(frames, 90000);

        let roundtrip = Timecode::from_frames(frames, fr);
        assert!(roundtrip.is_ok());
        let rt = roundtrip.expect("from_frames should succeed");
        assert_eq!(rt.hours, 1);
        assert_eq!(rt.minutes, 0);
        assert_eq!(rt.seconds, 0);
        assert_eq!(rt.frames, 0);
    }

    #[test]
    fn test_list_frame_rates_count() {
        let rates = list_frame_rates();
        assert_eq!(rates.len(), 9);
    }

    #[test]
    fn test_frame_rate_name_variants() {
        assert_eq!(frame_rate_name(FrameRate::Fps23976), "23.976 fps");
        assert_eq!(frame_rate_name(FrameRate::Fps2997DF), "29.97 fps DF (NTSC)");
    }
}
