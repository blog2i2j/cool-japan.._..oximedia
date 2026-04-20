//! Python bindings for clip management and logging.
//!
//! Provides `PyClipManager`, `PyClip`, `PyClipCollection` and standalone functions
//! for creating, listing, and exporting video clips.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_timestamp() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", dur.as_secs())
}

fn gen_id() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("clip-{:016x}", dur.as_nanos())
}

// ---------------------------------------------------------------------------
// PyClip
// ---------------------------------------------------------------------------

/// A video clip with metadata and logging information.
#[pyclass]
#[derive(Clone)]
pub struct PyClip {
    /// Clip identifier.
    #[pyo3(get)]
    pub id: String,
    /// Clip name.
    #[pyo3(get, set)]
    pub name: String,
    /// Source media file path.
    #[pyo3(get)]
    pub source_path: String,
    /// In-point timecode.
    #[pyo3(get, set)]
    pub tc_in: Option<String>,
    /// Out-point timecode.
    #[pyo3(get, set)]
    pub tc_out: Option<String>,
    /// Rating (0-5 stars).
    #[pyo3(get)]
    pub rating: u8,
    /// Keywords/tags.
    #[pyo3(get)]
    pub keywords: Vec<String>,
    /// Creation timestamp.
    #[pyo3(get)]
    pub created_at: String,
    /// Notes.
    #[pyo3(get, set)]
    pub notes: String,
}

#[pymethods]
impl PyClip {
    #[new]
    #[pyo3(signature = (source_path, name, tc_in=None, tc_out=None, rating=0))]
    fn new(
        source_path: &str,
        name: &str,
        tc_in: Option<String>,
        tc_out: Option<String>,
        rating: u8,
    ) -> PyResult<Self> {
        if rating > 5 {
            return Err(PyValueError::new_err("Rating must be 0-5"));
        }
        Ok(Self {
            id: gen_id(),
            name: name.to_string(),
            source_path: source_path.to_string(),
            tc_in,
            tc_out,
            rating,
            keywords: Vec::new(),
            created_at: now_timestamp(),
            notes: String::new(),
        })
    }

    /// Set the rating (0-5).
    fn set_rating(&mut self, rating: u8) -> PyResult<()> {
        if rating > 5 {
            return Err(PyValueError::new_err("Rating must be 0-5"));
        }
        self.rating = rating;
        Ok(())
    }

    /// Add a keyword.
    fn add_keyword(&mut self, keyword: &str) {
        let kw = keyword.to_string();
        if !self.keywords.contains(&kw) {
            self.keywords.push(kw);
        }
    }

    /// Remove a keyword.
    fn remove_keyword(&mut self, keyword: &str) {
        self.keywords.retain(|k| k != keyword);
    }

    /// Check if clip has a keyword.
    fn has_keyword(&self, keyword: &str) -> bool {
        self.keywords.iter().any(|k| k == keyword)
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("id".to_string(), self.id.clone());
        m.insert("name".to_string(), self.name.clone());
        m.insert("source_path".to_string(), self.source_path.clone());
        m.insert("tc_in".to_string(), self.tc_in.clone().unwrap_or_default());
        m.insert(
            "tc_out".to_string(),
            self.tc_out.clone().unwrap_or_default(),
        );
        m.insert("rating".to_string(), self.rating.to_string());
        m.insert("keywords".to_string(), self.keywords.join(","));
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyClip(name='{}', rating={}, keywords={})",
            self.name,
            self.rating,
            self.keywords.len()
        )
    }
}

// ---------------------------------------------------------------------------
// PyClipCollection
// ---------------------------------------------------------------------------

/// A named collection of clips.
#[pyclass]
#[derive(Clone)]
pub struct PyClipCollection {
    /// Collection name.
    #[pyo3(get)]
    pub name: String,
    /// Description.
    #[pyo3(get, set)]
    pub description: String,
    /// Number of clips.
    #[pyo3(get)]
    pub clip_count: u32,
    /// Creation timestamp.
    #[pyo3(get)]
    pub created_at: String,
}

#[pymethods]
impl PyClipCollection {
    fn __repr__(&self) -> String {
        format!(
            "PyClipCollection(name='{}', clips={})",
            self.name, self.clip_count
        )
    }
}

// ---------------------------------------------------------------------------
// PyClipManager
// ---------------------------------------------------------------------------

/// In-memory clip manager for organizing and searching clips.
#[pyclass]
pub struct PyClipManager {
    clips: Vec<PyClip>,
    collections: HashMap<String, String>, // name -> description
}

#[pymethods]
impl PyClipManager {
    #[new]
    fn new() -> Self {
        Self {
            clips: Vec::new(),
            collections: HashMap::new(),
        }
    }

    /// Add a clip to the manager.
    fn add_clip(&mut self, clip: PyClip) -> String {
        let id = clip.id.clone();
        self.clips.push(clip);
        id
    }

    /// Get a clip by ID.
    fn get_clip(&self, clip_id: &str) -> Option<PyClip> {
        self.clips.iter().find(|c| c.id == clip_id).cloned()
    }

    /// Remove a clip by ID.
    fn remove_clip(&mut self, clip_id: &str) -> PyResult<()> {
        let before = self.clips.len();
        self.clips.retain(|c| c.id != clip_id);
        if self.clips.len() == before {
            return Err(PyValueError::new_err(format!("Clip not found: {clip_id}")));
        }
        Ok(())
    }

    /// Search clips by keyword or name.
    #[pyo3(signature = (query, min_rating=None, limit=None))]
    fn search(&self, query: &str, min_rating: Option<u8>, limit: Option<u32>) -> Vec<PyClip> {
        let query_lower = query.to_lowercase();
        let max = limit.unwrap_or(100) as usize;

        self.clips
            .iter()
            .filter(|c| {
                let text_match = c.name.to_lowercase().contains(&query_lower)
                    || c.keywords
                        .iter()
                        .any(|k| k.to_lowercase().contains(&query_lower))
                    || c.source_path.to_lowercase().contains(&query_lower);
                let rating_ok = min_rating.map_or(true, |mr| c.rating >= mr);
                text_match && rating_ok
            })
            .take(max)
            .cloned()
            .collect()
    }

    /// List all clips.
    fn list_clips(&self) -> Vec<PyClip> {
        self.clips.clone()
    }

    /// Export clips as JSON string.
    fn export_json(&self) -> PyResult<String> {
        let data: Vec<HashMap<String, String>> = self.clips.iter().map(|c| c.to_dict()).collect();
        serde_json::to_string_pretty(&data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Serialize error: {e}")))
    }

    /// Create a collection.
    #[pyo3(signature = (name, description=None))]
    fn create_collection(
        &mut self,
        name: &str,
        description: Option<&str>,
    ) -> PyResult<PyClipCollection> {
        if self.collections.contains_key(name) {
            return Err(PyValueError::new_err(format!(
                "Collection already exists: {name}"
            )));
        }
        let desc = description.unwrap_or("").to_string();
        self.collections.insert(name.to_string(), desc.clone());
        Ok(PyClipCollection {
            name: name.to_string(),
            description: desc,
            clip_count: 0,
            created_at: now_timestamp(),
        })
    }

    /// Get clip count.
    fn clip_count(&self) -> usize {
        self.clips.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyClipManager(clips={}, collections={})",
            self.clips.len(),
            self.collections.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a new clip.
#[pyfunction]
#[pyo3(signature = (source_path, name, tc_in=None, tc_out=None, rating=0))]
pub fn create_clip(
    source_path: &str,
    name: &str,
    tc_in: Option<String>,
    tc_out: Option<String>,
    rating: u8,
) -> PyResult<PyClip> {
    PyClip::new(source_path, name, tc_in, tc_out, rating)
}

/// List supported export formats.
#[pyfunction]
pub fn list_clip_formats() -> Vec<String> {
    vec!["json".to_string(), "csv".to_string(), "edl".to_string()]
}

/// Export clips to JSON.
#[pyfunction]
pub fn export_clips(clips: Vec<PyClip>) -> PyResult<String> {
    let data: Vec<HashMap<String, String>> = clips.iter().map(|c| c.to_dict()).collect();
    serde_json::to_string_pretty(&data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Serialize error: {e}")))
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all clips bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClip>()?;
    m.add_class::<PyClipCollection>()?;
    m.add_class::<PyClipManager>()?;
    m.add_function(wrap_pyfunction!(create_clip, m)?)?;
    m.add_function(wrap_pyfunction!(list_clip_formats, m)?)?;
    m.add_function(wrap_pyfunction!(export_clips, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-py-clips-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_clip_creation() {
        let clip = PyClip::new(&tmp_str("video.mov"), "Take 1", None, None, 3);
        assert!(clip.is_ok());
        let c = clip.expect("should create");
        assert_eq!(c.name, "Take 1");
        assert_eq!(c.rating, 3);
    }

    #[test]
    fn test_clip_invalid_rating() {
        let clip = PyClip::new(&tmp_str("video.mov"), "Test", None, None, 10);
        assert!(clip.is_err());
    }

    #[test]
    fn test_clip_keywords() {
        let mut clip =
            PyClip::new(&tmp_str("video.mov"), "Test", None, None, 0).expect("should create");
        clip.add_keyword("interview");
        clip.add_keyword("raw");
        assert!(clip.has_keyword("interview"));
        assert!(!clip.has_keyword("final"));
        clip.remove_keyword("raw");
        assert!(!clip.has_keyword("raw"));
    }

    #[test]
    fn test_clip_manager_search() {
        let mut mgr = PyClipManager::new();
        let mut c1 = PyClip::new(&tmp_str("a.mov"), "Interview Take 1", None, None, 4)
            .expect("should create");
        c1.add_keyword("interview");
        let c2 =
            PyClip::new(&tmp_str("b.mov"), "B-Roll Sunset", None, None, 2).expect("should create");
        mgr.add_clip(c1);
        mgr.add_clip(c2);

        let results = mgr.search("interview", None, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Interview Take 1");
    }

    #[test]
    fn test_list_clip_formats() {
        let fmts = list_clip_formats();
        assert_eq!(fmts.len(), 3);
        assert!(fmts.contains(&"json".to_string()));
    }
}
