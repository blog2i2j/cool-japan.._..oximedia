//! Python bindings for `oximedia-edit` timeline editing.
//!
//! Provides `PyTimeline`, `PyClip`, `PyTrack`, `PyTransition`, and standalone
//! functions for creating, loading, and saving timeline projects from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyClip
// ---------------------------------------------------------------------------

/// A media clip that can be placed on a timeline track.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyClip {
    /// Source file path.
    #[pyo3(get)]
    pub source_path: String,

    /// Source in-point in seconds.
    #[pyo3(get)]
    pub in_point: f64,

    /// Source out-point in seconds.
    #[pyo3(get)]
    pub out_point: f64,

    /// Position on the timeline in seconds.
    #[pyo3(get, set)]
    pub start_time: f64,

    /// Duration on the timeline in seconds.
    #[pyo3(get)]
    pub duration: f64,

    /// Playback speed multiplier.
    #[pyo3(get)]
    pub speed: f64,

    /// Audio volume (0.0-1.0).
    #[pyo3(get)]
    pub volume: f64,

    /// Visual opacity (0.0-1.0).
    #[pyo3(get)]
    pub opacity: f64,

    /// Internal clip ID.
    #[pyo3(get)]
    pub clip_id: u64,
}

#[pymethods]
impl PyClip {
    /// Create a new clip from a source file with in/out points.
    ///
    /// Args:
    ///     source: Source file path.
    ///     in_point: Source in-point in seconds (default: 0.0).
    ///     out_point: Source out-point in seconds (default: 0.0 meaning auto).
    #[new]
    #[pyo3(signature = (source, in_point=None, out_point=None))]
    fn new(source: &str, in_point: Option<f64>, out_point: Option<f64>) -> PyResult<Self> {
        let inp = in_point.unwrap_or(0.0);
        let outp = out_point.unwrap_or(0.0);

        if inp < 0.0 {
            return Err(PyValueError::new_err(format!(
                "in_point must be >= 0.0, got {inp}"
            )));
        }
        if outp < 0.0 {
            return Err(PyValueError::new_err(format!(
                "out_point must be >= 0.0, got {outp}"
            )));
        }

        let dur = if outp > inp { outp - inp } else { 0.0 };

        Ok(Self {
            source_path: source.to_string(),
            in_point: inp,
            out_point: outp,
            start_time: 0.0,
            duration: dur,
            speed: 1.0,
            volume: 1.0,
            opacity: 1.0,
            clip_id: 0,
        })
    }

    /// Create a clip from a file, placing it at a specific timeline position.
    #[staticmethod]
    #[pyo3(signature = (path, start_time=None, duration=None))]
    fn from_file(path: &str, start_time: Option<f64>, duration: Option<f64>) -> PyResult<Self> {
        let st = start_time.unwrap_or(0.0);
        let dur = duration.unwrap_or(10.0); // default 10 seconds
        if st < 0.0 {
            return Err(PyValueError::new_err(format!(
                "start_time must be >= 0.0, got {st}"
            )));
        }
        if dur <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "duration must be > 0.0, got {dur}"
            )));
        }

        Ok(Self {
            source_path: path.to_string(),
            in_point: 0.0,
            out_point: dur,
            start_time: st,
            duration: dur,
            speed: 1.0,
            volume: 1.0,
            opacity: 1.0,
            clip_id: 0,
        })
    }

    /// Return a copy of this clip with modified speed.
    fn with_speed(&self, speed: f64) -> PyResult<Self> {
        if speed <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "speed must be > 0.0, got {speed}"
            )));
        }
        let mut clip = self.clone();
        clip.speed = speed;
        clip.duration = self.duration / speed;
        Ok(clip)
    }

    /// Return a copy of this clip with modified volume.
    fn with_volume(&self, volume: f64) -> PyResult<Self> {
        if !(0.0..=2.0).contains(&volume) {
            return Err(PyValueError::new_err(format!(
                "volume must be between 0.0 and 2.0, got {volume}"
            )));
        }
        let mut clip = self.clone();
        clip.volume = volume;
        Ok(clip)
    }

    /// Return a copy of this clip with modified opacity.
    fn with_opacity(&self, opacity: f64) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&opacity) {
            return Err(PyValueError::new_err(format!(
                "opacity must be between 0.0 and 1.0, got {opacity}"
            )));
        }
        let mut clip = self.clone();
        clip.opacity = opacity;
        Ok(clip)
    }

    /// Get the trimmed duration accounting for speed.
    fn trimmed_duration(&self) -> f64 {
        if self.speed > 0.0 {
            self.duration / self.speed
        } else {
            self.duration
        }
    }

    /// Get the end time on the timeline.
    fn end_time(&self) -> f64 {
        self.start_time + self.duration
    }

    /// Convert to a dictionary.
    fn to_dict(&self) -> HashMap<String, Py<PyAny>> {
        Python::attach(|py| {
            let mut m = HashMap::new();
            m.insert(
                "source_path".to_string(),
                self.source_path
                    .clone()
                    .into_pyobject(py)
                    .expect("str")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "in_point".to_string(),
                self.in_point
                    .into_pyobject(py)
                    .expect("f64")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "out_point".to_string(),
                self.out_point
                    .into_pyobject(py)
                    .expect("f64")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "start_time".to_string(),
                self.start_time
                    .into_pyobject(py)
                    .expect("f64")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "duration".to_string(),
                self.duration
                    .into_pyobject(py)
                    .expect("f64")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "speed".to_string(),
                self.speed
                    .into_pyobject(py)
                    .expect("f64")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "volume".to_string(),
                self.volume
                    .into_pyobject(py)
                    .expect("f64")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "opacity".to_string(),
                self.opacity
                    .into_pyobject(py)
                    .expect("f64")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "clip_id".to_string(),
                self.clip_id
                    .into_pyobject(py)
                    .expect("u64")
                    .into_any()
                    .unbind(),
            );
            m
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PyClip(source='{}', start={:.3}s, dur={:.3}s, speed={}x)",
            self.source_path, self.start_time, self.duration, self.speed
        )
    }
}

// ---------------------------------------------------------------------------
// PyTrack
// ---------------------------------------------------------------------------

/// A track in the timeline (video, audio, or subtitle).
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyTrack {
    /// Track clips.
    clips: Vec<PyClip>,

    /// Track name.
    #[pyo3(get, set)]
    pub name: String,

    /// Track type: "video", "audio", or "subtitle".
    #[pyo3(get)]
    pub track_type: String,

    /// Whether the track is muted.
    #[pyo3(get)]
    pub muted: bool,

    /// Whether the track is locked.
    #[pyo3(get)]
    pub locked: bool,

    /// Next clip ID counter.
    next_clip_id: u64,
}

#[pymethods]
impl PyTrack {
    /// Create a new track.
    ///
    /// Args:
    ///     name: Track name.
    ///     track_type: Type of track ("video", "audio", "subtitle").
    #[new]
    #[pyo3(signature = (name=None, track_type=None))]
    fn new(name: Option<&str>, track_type: Option<&str>) -> PyResult<Self> {
        let tt = track_type.unwrap_or("video");
        match tt {
            "video" | "audio" | "subtitle" => {}
            other => {
                return Err(PyValueError::new_err(format!(
                    "Invalid track type '{}'. Use: video, audio, subtitle",
                    other
                )));
            }
        }

        Ok(Self {
            clips: Vec::new(),
            name: name.unwrap_or("Track").to_string(),
            track_type: tt.to_string(),
            muted: false,
            locked: false,
            next_clip_id: 1,
        })
    }

    /// Add a clip to this track. Returns the assigned clip ID.
    fn add_clip(&mut self, mut clip: PyClip) -> PyResult<u64> {
        if self.locked {
            return Err(PyRuntimeError::new_err("Track is locked"));
        }

        clip.clip_id = self.next_clip_id;
        self.next_clip_id += 1;

        // Auto-position if start_time is 0 and there are existing clips
        if clip.start_time == 0.0 && !self.clips.is_empty() {
            clip.start_time = self.duration();
        }

        let clip_id = clip.clip_id;
        self.clips.push(clip);
        self.clips.sort_by(|a, b| {
            a.start_time
                .partial_cmp(&b.start_time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(clip_id)
    }

    /// Remove a clip by index.
    fn remove_clip(&mut self, index: usize) -> PyResult<PyClip> {
        if self.locked {
            return Err(PyRuntimeError::new_err("Track is locked"));
        }
        if index >= self.clips.len() {
            return Err(PyValueError::new_err(format!(
                "Clip index {} out of range (0..{})",
                index,
                self.clips.len()
            )));
        }
        Ok(self.clips.remove(index))
    }

    /// Get clip count.
    fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Get all clips.
    fn clips(&self) -> Vec<PyClip> {
        self.clips.clone()
    }

    /// Get a clip by index.
    fn get_clip(&self, index: usize) -> PyResult<PyClip> {
        self.clips.get(index).cloned().ok_or_else(|| {
            PyValueError::new_err(format!(
                "Clip index {} out of range (0..{})",
                index,
                self.clips.len()
            ))
        })
    }

    /// Get total track duration in seconds.
    fn duration(&self) -> f64 {
        self.clips
            .iter()
            .map(|c| c.start_time + c.duration)
            .fold(0.0_f64, f64::max)
    }

    /// Set muted state.
    fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }

    /// Set locked state.
    fn set_locked(&mut self, locked: bool) {
        self.locked = locked;
    }

    fn __repr__(&self) -> String {
        format!(
            "PyTrack(name='{}', type='{}', clips={}, dur={:.3}s)",
            self.name,
            self.track_type,
            self.clips.len(),
            self.duration(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyTransition
// ---------------------------------------------------------------------------

/// A transition between two clips.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyTransition {
    /// Transition type name.
    #[pyo3(get)]
    pub transition_type: String,

    /// Duration of the transition in seconds.
    #[pyo3(get)]
    pub duration: f64,

    /// Extra parameters.
    #[pyo3(get)]
    pub params: HashMap<String, f64>,
}

#[pymethods]
impl PyTransition {
    /// Create a cross-dissolve transition.
    #[staticmethod]
    fn dissolve(duration: f64) -> PyResult<Self> {
        if duration <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "duration must be > 0.0, got {duration}"
            )));
        }
        Ok(Self {
            transition_type: "dissolve".to_string(),
            duration,
            params: HashMap::new(),
        })
    }

    /// Create a wipe transition.
    #[staticmethod]
    #[pyo3(signature = (duration, direction=None))]
    fn wipe(duration: f64, direction: Option<&str>) -> PyResult<Self> {
        if duration <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "duration must be > 0.0, got {duration}"
            )));
        }
        let dir = direction.unwrap_or("left");
        let angle = match dir {
            "left" => 0.0,
            "right" => 180.0,
            "up" => 90.0,
            "down" => 270.0,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown wipe direction '{}'. Use: left, right, up, down",
                    dir
                )));
            }
        };
        let mut params = HashMap::new();
        params.insert("angle".to_string(), angle);
        Ok(Self {
            transition_type: "wipe".to_string(),
            duration,
            params,
        })
    }

    /// Create a fade-in from black.
    #[staticmethod]
    fn fade_in(duration: f64) -> PyResult<Self> {
        if duration <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "duration must be > 0.0, got {duration}"
            )));
        }
        Ok(Self {
            transition_type: "fade_in".to_string(),
            duration,
            params: HashMap::new(),
        })
    }

    /// Create a fade-out to black.
    #[staticmethod]
    fn fade_out(duration: f64) -> PyResult<Self> {
        if duration <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "duration must be > 0.0, got {duration}"
            )));
        }
        Ok(Self {
            transition_type: "fade_out".to_string(),
            duration,
            params: HashMap::new(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PyTransition(type='{}', duration={:.3}s)",
            self.transition_type, self.duration
        )
    }
}

// ---------------------------------------------------------------------------
// PyTimeline
// ---------------------------------------------------------------------------

/// A multi-track video/audio timeline.
#[pyclass]
pub struct PyTimeline {
    tracks: Vec<PyTrack>,

    /// Frame rate (fps).
    #[pyo3(get)]
    pub fps: f64,

    /// Video width.
    #[pyo3(get)]
    pub width: u32,

    /// Video height.
    #[pyo3(get)]
    pub height: u32,

    /// Project name.
    #[pyo3(get, set)]
    pub name: String,

    /// Next track-level clip ID.
    next_clip_id: u64,
}

#[pymethods]
impl PyTimeline {
    /// Create a new timeline.
    ///
    /// Args:
    ///     fps: Frame rate (default: 24.0).
    ///     width: Video width (default: 1920).
    ///     height: Video height (default: 1080).
    ///     name: Project name (default: "Untitled").
    #[new]
    #[pyo3(signature = (fps=None, width=None, height=None, name=None))]
    fn new(
        fps: Option<f64>,
        width: Option<u32>,
        height: Option<u32>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let f = fps.unwrap_or(24.0);
        let w = width.unwrap_or(1920);
        let h = height.unwrap_or(1080);

        if f <= 0.0 {
            return Err(PyValueError::new_err(format!("fps must be > 0.0, got {f}")));
        }
        if w == 0 || h == 0 {
            return Err(PyValueError::new_err(format!(
                "width and height must be > 0, got {w}x{h}"
            )));
        }

        Ok(Self {
            tracks: Vec::new(),
            fps: f,
            width: w,
            height: h,
            name: name.unwrap_or("Untitled").to_string(),
            next_clip_id: 1,
        })
    }

    /// Add a new track. Returns the track index.
    ///
    /// Args:
    ///     name: Track name (optional).
    ///     track_type: "video", "audio", or "subtitle" (default: "video").
    #[pyo3(signature = (name=None, track_type=None))]
    fn add_track(&mut self, name: Option<&str>, track_type: Option<&str>) -> PyResult<usize> {
        let tt = track_type.unwrap_or("video");
        let default_name = format!("{} {}", tt, self.tracks.len() + 1);
        let track_name = name.unwrap_or(&default_name);
        let track = PyTrack::new(Some(track_name), Some(tt))?;
        let index = self.tracks.len();
        self.tracks.push(track);
        Ok(index)
    }

    /// Add a clip to a specific track. Returns the assigned clip ID.
    fn add_clip(&mut self, track_index: usize, mut clip: PyClip) -> PyResult<u64> {
        if track_index >= self.tracks.len() {
            return Err(PyValueError::new_err(format!(
                "Track index {} out of range (0..{})",
                track_index,
                self.tracks.len()
            )));
        }

        clip.clip_id = self.next_clip_id;
        self.next_clip_id += 1;
        let clip_id = clip.clip_id;

        let track = &mut self.tracks[track_index];
        if track.locked {
            return Err(PyRuntimeError::new_err("Track is locked"));
        }

        // Auto-position at end if start_time is 0 and there are clips
        if clip.start_time == 0.0 && !track.clips.is_empty() {
            clip.start_time = track.duration();
        }

        track.clips.push(clip);
        track.clips.sort_by(|a, b| {
            a.start_time
                .partial_cmp(&b.start_time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(clip_id)
    }

    /// Remove a clip from a track by clip index.
    fn remove_clip(&mut self, track_index: usize, clip_index: usize) -> PyResult<PyClip> {
        if track_index >= self.tracks.len() {
            return Err(PyValueError::new_err(format!(
                "Track index {} out of range (0..{})",
                track_index,
                self.tracks.len()
            )));
        }
        self.tracks[track_index].remove_clip(clip_index)
    }

    /// Get total timeline duration in seconds.
    fn duration(&self) -> f64 {
        self.tracks
            .iter()
            .map(|t| t.duration())
            .fold(0.0_f64, f64::max)
    }

    /// Get the total number of tracks.
    fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Get the total number of clips across all tracks.
    fn clip_count(&self) -> usize {
        self.tracks.iter().map(|t| t.clips.len()).sum()
    }

    /// Get a track by index.
    fn get_track(&self, index: usize) -> PyResult<PyTrack> {
        self.tracks.get(index).cloned().ok_or_else(|| {
            PyValueError::new_err(format!(
                "Track index {} out of range (0..{})",
                index,
                self.tracks.len()
            ))
        })
    }

    /// Get all tracks.
    fn tracks(&self) -> Vec<PyTrack> {
        self.tracks.clone()
    }

    /// Render the timeline (placeholder -- returns render configuration info).
    #[pyo3(signature = (output, codec=None, quality=None))]
    fn render(&self, output: &str, codec: Option<&str>, quality: Option<u32>) -> PyResult<String> {
        let selected_codec = codec.unwrap_or("av1");
        match selected_codec {
            "av1" | "vp9" | "vp8" => {}
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported codec '{}'. Use: av1, vp9, vp8",
                    other
                )));
            }
        }

        let q = quality.unwrap_or(28);
        let info = serde_json::json!({
            "output": output,
            "codec": selected_codec,
            "quality": q,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration": self.duration(),
            "tracks": self.tracks.len(),
            "clips": self.clip_count(),
            "status": "pending_render_pipeline",
        });

        serde_json::to_string_pretty(&info)
            .map_err(|e| PyRuntimeError::new_err(format!("JSON error: {e}")))
    }

    /// Serialize the timeline to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        let data = timeline_to_json(self);
        serde_json::to_string_pretty(&data)
            .map_err(|e| PyRuntimeError::new_err(format!("JSON serialization failed: {e}")))
    }

    /// Load timeline from a JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let data: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {e}")))?;

        json_to_timeline(&data)
    }

    /// Get timeline info as a dictionary.
    fn info(&self) -> HashMap<String, Py<PyAny>> {
        Python::attach(|py| {
            let mut m = HashMap::new();
            m.insert(
                "name".to_string(),
                self.name
                    .clone()
                    .into_pyobject(py)
                    .expect("str")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "fps".to_string(),
                self.fps.into_pyobject(py).expect("f64").into_any().unbind(),
            );
            m.insert(
                "width".to_string(),
                self.width
                    .into_pyobject(py)
                    .expect("u32")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "height".to_string(),
                self.height
                    .into_pyobject(py)
                    .expect("u32")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "duration".to_string(),
                self.duration()
                    .into_pyobject(py)
                    .expect("f64")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "track_count".to_string(),
                self.track_count()
                    .into_pyobject(py)
                    .expect("usize")
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "clip_count".to_string(),
                self.clip_count()
                    .into_pyobject(py)
                    .expect("usize")
                    .into_any()
                    .unbind(),
            );
            m
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PyTimeline(name='{}', fps={}, {}x{}, tracks={}, clips={}, dur={:.3}s)",
            self.name,
            self.fps,
            self.width,
            self.height,
            self.tracks.len(),
            self.clip_count(),
            self.duration(),
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a new empty timeline.
///
/// Args:
///     fps: Frame rate (default: 24.0).
///     width: Video width (default: 1920).
///     height: Video height (default: 1080).
///     name: Project name (default: "Untitled").
#[pyfunction]
#[pyo3(signature = (fps=None, width=None, height=None, name=None))]
pub fn create_timeline(
    fps: Option<f64>,
    width: Option<u32>,
    height: Option<u32>,
    name: Option<&str>,
) -> PyResult<PyTimeline> {
    PyTimeline::new(fps, width, height, name)
}

/// Load a timeline from a JSON file.
#[pyfunction]
pub fn load_timeline(path: &str) -> PyResult<PyTimeline> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read file: {e}")))?;
    PyTimeline::from_json(&content)
}

/// Save a timeline to a JSON file.
#[pyfunction]
pub fn save_timeline(timeline: &PyTimeline, path: &str) -> PyResult<()> {
    let json = timeline.to_json()?;
    std::fs::write(path, json)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to write file: {e}")))?;
    Ok(())
}

/// List available timeline export formats.
#[pyfunction]
pub fn list_timeline_formats() -> Vec<String> {
    vec![
        "json".to_string(),
        "edl".to_string(),
        "fcpxml".to_string(),
        "otio".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// JSON serialization helpers
// ---------------------------------------------------------------------------

fn timeline_to_json(tl: &PyTimeline) -> serde_json::Value {
    serde_json::json!({
        "name": tl.name,
        "fps": tl.fps,
        "width": tl.width,
        "height": tl.height,
        "tracks": tl.tracks.iter().map(|t| {
            serde_json::json!({
                "name": t.name,
                "track_type": t.track_type,
                "muted": t.muted,
                "locked": t.locked,
                "clips": t.clips.iter().map(|c| {
                    serde_json::json!({
                        "source_path": c.source_path,
                        "in_point": c.in_point,
                        "out_point": c.out_point,
                        "start_time": c.start_time,
                        "duration": c.duration,
                        "speed": c.speed,
                        "volume": c.volume,
                        "opacity": c.opacity,
                        "clip_id": c.clip_id,
                    })
                }).collect::<Vec<_>>(),
            })
        }).collect::<Vec<_>>(),
    })
}

fn json_to_timeline(data: &serde_json::Value) -> PyResult<PyTimeline> {
    let name = data["name"].as_str().unwrap_or("Untitled");
    let fps = data["fps"].as_f64().unwrap_or(24.0);
    let width = data["width"].as_u64().unwrap_or(1920) as u32;
    let height = data["height"].as_u64().unwrap_or(1080) as u32;

    let mut tl = PyTimeline::new(Some(fps), Some(width), Some(height), Some(name))?;

    if let Some(tracks) = data["tracks"].as_array() {
        for track_data in tracks {
            let tt = track_data["track_type"].as_str().unwrap_or("video");
            let tn = track_data["name"].as_str().unwrap_or("Track");
            let idx = tl.add_track(Some(tn), Some(tt))?;

            if let Some(muted) = track_data["muted"].as_bool() {
                tl.tracks[idx].muted = muted;
            }
            if let Some(locked) = track_data["locked"].as_bool() {
                tl.tracks[idx].locked = locked;
            }

            if let Some(clips) = track_data["clips"].as_array() {
                for clip_data in clips {
                    let source = clip_data["source_path"].as_str().unwrap_or("");
                    let clip = PyClip {
                        source_path: source.to_string(),
                        in_point: clip_data["in_point"].as_f64().unwrap_or(0.0),
                        out_point: clip_data["out_point"].as_f64().unwrap_or(0.0),
                        start_time: clip_data["start_time"].as_f64().unwrap_or(0.0),
                        duration: clip_data["duration"].as_f64().unwrap_or(0.0),
                        speed: clip_data["speed"].as_f64().unwrap_or(1.0),
                        volume: clip_data["volume"].as_f64().unwrap_or(1.0),
                        opacity: clip_data["opacity"].as_f64().unwrap_or(1.0),
                        clip_id: clip_data["clip_id"].as_u64().unwrap_or(0),
                    };
                    tl.tracks[idx].clips.push(clip);
                }
            }
        }
    }

    Ok(tl)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register timeline bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimeline>()?;
    m.add_class::<PyClip>()?;
    m.add_class::<PyTrack>()?;
    m.add_class::<PyTransition>()?;
    m.add_function(wrap_pyfunction!(create_timeline, m)?)?;
    m.add_function(wrap_pyfunction!(load_timeline, m)?)?;
    m.add_function(wrap_pyfunction!(save_timeline, m)?)?;
    m.add_function(wrap_pyfunction!(list_timeline_formats, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_new() {
        let clip = PyClip::new("test.mp4", Some(1.0), Some(5.0));
        assert!(clip.is_ok());
        let clip = clip.expect("clip should be valid");
        assert_eq!(clip.source_path, "test.mp4");
        assert!((clip.in_point - 1.0).abs() < f64::EPSILON);
        assert!((clip.out_point - 5.0).abs() < f64::EPSILON);
        assert!((clip.duration - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clip_from_file() {
        let clip = PyClip::from_file("test.mp4", Some(2.0), Some(8.0));
        assert!(clip.is_ok());
        let clip = clip.expect("clip should be valid");
        assert!((clip.start_time - 2.0).abs() < f64::EPSILON);
        assert!((clip.duration - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clip_with_speed() {
        let clip = PyClip::from_file("test.mp4", None, Some(10.0)).expect("valid");
        let fast = clip.with_speed(2.0).expect("valid speed");
        assert!((fast.speed - 2.0).abs() < f64::EPSILON);
        assert!((fast.duration - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clip_invalid_speed() {
        let clip = PyClip::from_file("test.mp4", None, Some(5.0)).expect("valid");
        assert!(clip.with_speed(-1.0).is_err());
        assert!(clip.with_speed(0.0).is_err());
    }

    #[test]
    fn test_track_new() {
        let track = PyTrack::new(Some("V1"), Some("video"));
        assert!(track.is_ok());
        let track = track.expect("track should be valid");
        assert_eq!(track.name, "V1");
        assert_eq!(track.track_type, "video");
        assert_eq!(track.clip_count(), 0);
    }

    #[test]
    fn test_track_add_clip() {
        let mut track = PyTrack::new(Some("V1"), Some("video")).expect("valid");
        let clip = PyClip::from_file("test.mp4", Some(0.0), Some(5.0)).expect("valid");
        let id = track.add_clip(clip);
        assert!(id.is_ok());
        assert_eq!(track.clip_count(), 1);
    }

    #[test]
    fn test_track_locked() {
        let mut track = PyTrack::new(None, None).expect("valid");
        track.set_locked(true);
        let clip = PyClip::from_file("test.mp4", None, Some(5.0)).expect("valid");
        assert!(track.add_clip(clip).is_err());
    }

    #[test]
    fn test_timeline_new() {
        let tl = PyTimeline::new(Some(30.0), Some(1920), Some(1080), Some("Test"));
        assert!(tl.is_ok());
        let tl = tl.expect("timeline should be valid");
        assert_eq!(tl.name, "Test");
        assert!((tl.fps - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_timeline_add_track_and_clip() {
        let mut tl = PyTimeline::new(None, None, None, None).expect("valid");
        let idx = tl.add_track(Some("V1"), Some("video")).expect("valid");
        assert_eq!(idx, 0);

        let clip = PyClip::from_file("test.mp4", Some(0.0), Some(5.0)).expect("valid");
        let clip_id = tl.add_clip(0, clip).expect("valid");
        assert!(clip_id > 0);
        assert_eq!(tl.clip_count(), 1);
        assert!((tl.duration() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_timeline_json_roundtrip() {
        let mut tl =
            PyTimeline::new(Some(30.0), Some(1920), Some(1080), Some("Roundtrip")).expect("valid");
        tl.add_track(Some("V1"), Some("video")).expect("valid");
        let clip = PyClip::from_file("clip.mp4", Some(0.0), Some(5.0)).expect("valid");
        tl.add_clip(0, clip).expect("valid");

        let json = tl.to_json().expect("should serialize");
        let tl2 = PyTimeline::from_json(&json).expect("should deserialize");
        assert_eq!(tl2.name, "Roundtrip");
        assert_eq!(tl2.track_count(), 1);
        assert_eq!(tl2.clip_count(), 1);
    }

    #[test]
    fn test_transition_dissolve() {
        let t = PyTransition::dissolve(1.5);
        assert!(t.is_ok());
        let t = t.expect("valid");
        assert_eq!(t.transition_type, "dissolve");
        assert!((t.duration - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_transition_invalid_duration() {
        assert!(PyTransition::dissolve(0.0).is_err());
        assert!(PyTransition::dissolve(-1.0).is_err());
    }

    #[test]
    fn test_create_timeline_fn() {
        let tl = create_timeline(Some(60.0), Some(3840), Some(2160), Some("4K"));
        assert!(tl.is_ok());
        let tl = tl.expect("valid");
        assert!((tl.fps - 60.0).abs() < f64::EPSILON);
        assert_eq!(tl.width, 3840);
    }

    #[test]
    fn test_list_formats() {
        let formats = list_timeline_formats();
        assert!(formats.contains(&"json".to_string()));
        assert!(formats.contains(&"edl".to_string()));
    }
}
