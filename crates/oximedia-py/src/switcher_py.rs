//! Python bindings for `oximedia-switcher` live production video switcher.
//!
//! Provides `PySwitcher`, `PySwitcherSource`, `PyTransitionType`,
//! `PySwitcherConfig`, and standalone functions for live switching.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PySwitcherSource
// ---------------------------------------------------------------------------

/// A video source input for the switcher.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PySwitcherSource {
    /// Source index (slot).
    #[pyo3(get)]
    pub index: usize,

    /// Source name/label.
    #[pyo3(get, set)]
    pub name: String,

    /// Source type: sdi, ndi, file, test_pattern, media_player.
    #[pyo3(get)]
    pub source_type: String,

    /// Source URI or path.
    #[pyo3(get)]
    pub uri: Option<String>,
}

#[pymethods]
impl PySwitcherSource {
    /// Create a new switcher source.
    #[new]
    #[pyo3(signature = (name, source_type=None, uri=None))]
    fn new(name: &str, source_type: Option<&str>, uri: Option<&str>) -> PyResult<Self> {
        let st = source_type.unwrap_or("sdi");
        match st {
            "sdi" | "ndi" | "file" | "test_pattern" | "media_player" => {}
            other => {
                return Err(PyValueError::new_err(format!(
                    "Invalid source type '{}'. Use: sdi, ndi, file, test_pattern, media_player",
                    other
                )));
            }
        }
        Ok(Self {
            index: 0,
            name: name.to_string(),
            source_type: st.to_string(),
            uri: uri.map(|s| s.to_string()),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PySwitcherSource(index={}, name='{}', type='{}')",
            self.index, self.name, self.source_type,
        )
    }
}

// ---------------------------------------------------------------------------
// PyTransitionType
// ---------------------------------------------------------------------------

/// Transition type configuration.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyTransitionType {
    /// Transition name: cut, mix, wipe, dve.
    #[pyo3(get)]
    pub name: String,

    /// Duration in frames (0 for cut).
    #[pyo3(get)]
    pub duration_frames: u32,

    /// Extra parameters.
    #[pyo3(get)]
    pub params: HashMap<String, f64>,
}

#[pymethods]
impl PyTransitionType {
    /// Create a cut transition.
    #[staticmethod]
    fn cut() -> Self {
        Self {
            name: "cut".to_string(),
            duration_frames: 0,
            params: HashMap::new(),
        }
    }

    /// Create a mix/dissolve transition.
    #[staticmethod]
    #[pyo3(signature = (duration_frames=None))]
    fn mix(duration_frames: Option<u32>) -> Self {
        Self {
            name: "mix".to_string(),
            duration_frames: duration_frames.unwrap_or(30),
            params: HashMap::new(),
        }
    }

    /// Create a wipe transition.
    #[staticmethod]
    #[pyo3(signature = (duration_frames=None, pattern=None))]
    fn wipe(duration_frames: Option<u32>, pattern: Option<&str>) -> Self {
        let mut params = HashMap::new();
        let p = pattern.unwrap_or("horizontal");
        let pattern_val = match p {
            "horizontal" => 0.0,
            "vertical" => 1.0,
            "diagonal" => 2.0,
            "circle" => 3.0,
            _ => 0.0,
        };
        params.insert("pattern".to_string(), pattern_val);
        Self {
            name: "wipe".to_string(),
            duration_frames: duration_frames.unwrap_or(30),
            params,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyTransitionType(name='{}', frames={})",
            self.name, self.duration_frames,
        )
    }
}

// ---------------------------------------------------------------------------
// PySwitcherConfig
// ---------------------------------------------------------------------------

/// Configuration for a video switcher session.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PySwitcherConfig {
    /// Number of M/E rows.
    #[pyo3(get)]
    pub me_rows: usize,

    /// Number of inputs.
    #[pyo3(get)]
    pub num_inputs: usize,

    /// Number of aux outputs.
    #[pyo3(get)]
    pub num_aux: usize,

    /// Frame rate.
    #[pyo3(get)]
    pub frame_rate: f64,

    /// Preset name (if used).
    #[pyo3(get)]
    pub preset: Option<String>,
}

#[pymethods]
impl PySwitcherConfig {
    /// Create a new switcher configuration.
    #[new]
    #[pyo3(signature = (me_rows=None, num_inputs=None, num_aux=None, preset=None))]
    fn new(
        me_rows: Option<usize>,
        num_inputs: Option<usize>,
        num_aux: Option<usize>,
        preset: Option<&str>,
    ) -> PyResult<Self> {
        let (me, inp, aux) = if let Some(p) = preset {
            match p {
                "basic" => (1, 8, 2),
                "professional" => (2, 20, 6),
                "broadcast" => (4, 40, 10),
                other => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown preset '{}'. Use: basic, professional, broadcast",
                        other
                    )));
                }
            }
        } else {
            (
                me_rows.unwrap_or(1),
                num_inputs.unwrap_or(8),
                num_aux.unwrap_or(2),
            )
        };

        Ok(Self {
            me_rows: me,
            num_inputs: inp,
            num_aux: aux,
            frame_rate: 25.0,
            preset: preset.map(|s| s.to_string()),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PySwitcherConfig(me={}, inputs={}, aux={})",
            self.me_rows, self.num_inputs, self.num_aux,
        )
    }
}

// ---------------------------------------------------------------------------
// PySwitcher
// ---------------------------------------------------------------------------

/// A live production video switcher.
#[pyclass]
pub struct PySwitcher {
    config: PySwitcherConfig,
    sources: Vec<PySwitcherSource>,
    program_input: usize,
    preview_input: usize,
    me_row: usize,
    next_source_index: usize,
}

#[pymethods]
impl PySwitcher {
    /// Create a new switcher instance.
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PySwitcherConfig>) -> PyResult<Self> {
        let cfg = config
            .map(Ok)
            .unwrap_or_else(|| PySwitcherConfig::new(None, None, None, None))?;
        Ok(Self {
            config: cfg,
            sources: Vec::new(),
            program_input: 0,
            preview_input: 0,
            me_row: 0,
            next_source_index: 0,
        })
    }

    /// Add a source to the switcher. Returns the assigned input index.
    fn add_source(&mut self, mut source: PySwitcherSource) -> usize {
        source.index = self.next_source_index;
        self.next_source_index += 1;
        let idx = source.index;
        self.sources.push(source);
        idx
    }

    /// Get all sources.
    fn sources(&self) -> Vec<PySwitcherSource> {
        self.sources.clone()
    }

    /// Get source count.
    fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Get the current program input index.
    fn program_input(&self) -> usize {
        self.program_input
    }

    /// Get the current preview input index.
    fn preview_input(&self) -> usize {
        self.preview_input
    }

    /// Set the program source.
    fn set_program(&mut self, input: usize) -> PyResult<()> {
        if input >= self.next_source_index && input > 0 {
            return Err(PyValueError::new_err(format!(
                "Input index {} out of range",
                input
            )));
        }
        self.program_input = input;
        Ok(())
    }

    /// Set the preview source.
    fn set_preview(&mut self, input: usize) -> PyResult<()> {
        if input >= self.next_source_index && input > 0 {
            return Err(PyValueError::new_err(format!(
                "Input index {} out of range",
                input
            )));
        }
        self.preview_input = input;
        Ok(())
    }

    /// Perform a cut (swap program and preview).
    fn cut(&mut self) {
        std::mem::swap(&mut self.program_input, &mut self.preview_input);
    }

    /// Switch to a specific input with a transition.
    fn switch_to(&mut self, input: usize, _transition: Option<PyTransitionType>) -> PyResult<()> {
        self.preview_input = input;
        self.cut();
        Ok(())
    }

    /// Get the configuration.
    fn config(&self) -> PySwitcherConfig {
        self.config.clone()
    }

    /// Get the active M/E row.
    fn me_row(&self) -> usize {
        self.me_row
    }

    /// Set the active M/E row.
    fn set_me_row(&mut self, row: usize) -> PyResult<()> {
        if row >= self.config.me_rows {
            return Err(PyValueError::new_err(format!(
                "M/E row {} out of range (0..{})",
                row, self.config.me_rows
            )));
        }
        self.me_row = row;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "PySwitcher(me={}, sources={}, pgm={}, pvw={})",
            self.config.me_rows,
            self.sources.len(),
            self.program_input,
            self.preview_input,
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a switcher with the given sources.
#[pyfunction]
#[pyo3(signature = (sources, config=None))]
pub fn create_switcher(
    sources: Vec<PySwitcherSource>,
    config: Option<PySwitcherConfig>,
) -> PyResult<PySwitcher> {
    let mut switcher = PySwitcher::new(config)?;
    for source in sources {
        switcher.add_source(source);
    }
    Ok(switcher)
}

/// List available transition types.
#[pyfunction]
pub fn list_transition_types() -> Vec<String> {
    vec![
        "cut".to_string(),
        "mix".to_string(),
        "wipe".to_string(),
        "dve".to_string(),
    ]
}

/// List available switcher presets.
#[pyfunction]
pub fn list_switcher_presets() -> Vec<String> {
    vec![
        "basic".to_string(),
        "professional".to_string(),
        "broadcast".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register switcher bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySwitcher>()?;
    m.add_class::<PySwitcherSource>()?;
    m.add_class::<PyTransitionType>()?;
    m.add_class::<PySwitcherConfig>()?;
    m.add_function(wrap_pyfunction!(create_switcher, m)?)?;
    m.add_function(wrap_pyfunction!(list_transition_types, m)?)?;
    m.add_function(wrap_pyfunction!(list_switcher_presets, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_switcher_source_new() {
        let src = PySwitcherSource::new("Camera 1", Some("sdi"), None);
        assert!(src.is_ok());
        let src = src.expect("valid");
        assert_eq!(src.name, "Camera 1");
        assert_eq!(src.source_type, "sdi");
    }

    #[test]
    fn test_switcher_source_invalid_type() {
        let src = PySwitcherSource::new("Bad", Some("invalid"), None);
        assert!(src.is_err());
    }

    #[test]
    fn test_switcher_add_source_and_cut() {
        let mut sw = PySwitcher::new(None).expect("valid");
        let s1 = PySwitcherSource::new("Cam1", None, None).expect("valid");
        let s2 = PySwitcherSource::new("Cam2", None, None).expect("valid");
        let idx1 = sw.add_source(s1);
        let idx2 = sw.add_source(s2);

        sw.set_program(idx1).expect("valid");
        sw.set_preview(idx2).expect("valid");
        assert_eq!(sw.program_input(), idx1);
        assert_eq!(sw.preview_input(), idx2);

        sw.cut();
        assert_eq!(sw.program_input(), idx2);
        assert_eq!(sw.preview_input(), idx1);
    }

    #[test]
    fn test_transition_types() {
        let cut = PyTransitionType::cut();
        assert_eq!(cut.name, "cut");
        assert_eq!(cut.duration_frames, 0);

        let mix = PyTransitionType::mix(Some(20));
        assert_eq!(mix.name, "mix");
        assert_eq!(mix.duration_frames, 20);
    }

    #[test]
    fn test_create_switcher_fn() {
        let sources = vec![
            PySwitcherSource::new("A", None, None).expect("valid"),
            PySwitcherSource::new("B", None, None).expect("valid"),
        ];
        let sw = create_switcher(sources, None);
        assert!(sw.is_ok());
        let sw = sw.expect("valid");
        assert_eq!(sw.source_count(), 2);
    }
}
