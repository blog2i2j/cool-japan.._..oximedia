//! Python bindings for Media Asset Management (MAM).
//!
//! Provides `PyAsset`, `PyCollection`, `PySearchResult`, and `PyAssetManager`
//! for cataloging, searching, tagging, and exporting media assets from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
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

fn generate_id() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("asset-{:016x}", dur.as_nanos())
}

fn compute_file_checksum(path: &str) -> PyResult<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open file: {e}")))?;
    let mut hasher_state: u64 = 0xcbf29ce484222325;
    let mut buf = [0u8; 8192];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| PyRuntimeError::new_err(format!("Read error: {e}")))?;
        if n == 0 {
            break;
        }
        for &byte in &buf[..n] {
            hasher_state ^= u64::from(byte);
            hasher_state = hasher_state.wrapping_mul(0x100000001b3);
        }
    }
    Ok(format!("{:016x}", hasher_state))
}

fn detect_format(path: &str) -> String {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown")
        .to_lowercase()
}

// ---------------------------------------------------------------------------
// PyAsset
// ---------------------------------------------------------------------------

/// A media asset in the catalog.
#[pyclass]
#[derive(Clone)]
pub struct PyAsset {
    /// Unique asset identifier.
    #[pyo3(get)]
    pub id: String,
    /// Absolute file path.
    #[pyo3(get)]
    pub path: String,
    /// File name.
    #[pyo3(get)]
    pub filename: String,
    /// File format / extension.
    #[pyo3(get)]
    pub format: String,
    /// File size in bytes.
    #[pyo3(get)]
    pub size_bytes: u64,
    /// Duration in seconds (for A/V assets).
    #[pyo3(get)]
    pub duration_secs: Option<f64>,
    /// Video width in pixels.
    #[pyo3(get)]
    pub width: Option<u32>,
    /// Video height in pixels.
    #[pyo3(get)]
    pub height: Option<u32>,
    /// Codec name.
    #[pyo3(get)]
    pub codec: Option<String>,
    /// Tags assigned to this asset.
    #[pyo3(get)]
    pub tags: Vec<String>,
    /// Collection this asset belongs to.
    #[pyo3(get)]
    pub collection: Option<String>,
    /// Timestamp when the asset was ingested.
    #[pyo3(get)]
    pub ingested_at: String,
    /// Content checksum.
    #[pyo3(get)]
    pub checksum: String,
    /// Arbitrary key-value metadata.
    metadata: HashMap<String, String>,
}

#[pymethods]
impl PyAsset {
    fn __repr__(&self) -> String {
        format!(
            "PyAsset(id='{}', filename='{}', format='{}', size={})",
            self.id, self.filename, self.format, self.size_bytes
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("id".to_string(), self.id.clone());
        m.insert("path".to_string(), self.path.clone());
        m.insert("filename".to_string(), self.filename.clone());
        m.insert("format".to_string(), self.format.clone());
        m.insert("size_bytes".to_string(), self.size_bytes.to_string());
        m.insert(
            "duration_secs".to_string(),
            self.duration_secs
                .map_or_else(String::new, |d| d.to_string()),
        );
        m.insert(
            "width".to_string(),
            self.width.map_or_else(String::new, |w| w.to_string()),
        );
        m.insert(
            "height".to_string(),
            self.height.map_or_else(String::new, |h| h.to_string()),
        );
        m.insert("codec".to_string(), self.codec.clone().unwrap_or_default());
        m.insert("tags".to_string(), self.tags.join(","));
        m.insert(
            "collection".to_string(),
            self.collection.clone().unwrap_or_default(),
        );
        m.insert("ingested_at".to_string(), self.ingested_at.clone());
        m.insert("checksum".to_string(), self.checksum.clone());
        m
    }

    /// Get a metadata value by key.
    fn get_metadata(&self, key: &str) -> Option<String> {
        self.metadata.get(key).cloned()
    }

    /// Check if the asset has a specific tag.
    fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

// ---------------------------------------------------------------------------
// PyCollection
// ---------------------------------------------------------------------------

/// A named collection of assets.
#[pyclass]
#[derive(Clone)]
pub struct PyCollection {
    /// Collection name.
    #[pyo3(get)]
    pub name: String,
    /// Description.
    #[pyo3(get)]
    pub description: String,
    /// Number of assets in this collection.
    #[pyo3(get)]
    pub asset_count: u32,
    /// Total size of assets in bytes.
    #[pyo3(get)]
    pub total_size_bytes: u64,
    /// Creation timestamp.
    #[pyo3(get)]
    pub created_at: String,
}

#[pymethods]
impl PyCollection {
    fn __repr__(&self) -> String {
        format!(
            "PyCollection(name='{}', assets={}, size={})",
            self.name, self.asset_count, self.total_size_bytes
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("name".to_string(), self.name.clone());
        m.insert("description".to_string(), self.description.clone());
        m.insert("asset_count".to_string(), self.asset_count.to_string());
        m.insert(
            "total_size_bytes".to_string(),
            self.total_size_bytes.to_string(),
        );
        m.insert("created_at".to_string(), self.created_at.clone());
        m
    }
}

// ---------------------------------------------------------------------------
// PySearchResult
// ---------------------------------------------------------------------------

/// Search result containing matching assets.
#[pyclass]
#[derive(Clone)]
pub struct PySearchResult {
    /// Total number of matches.
    #[pyo3(get)]
    pub total_count: u32,
    /// Query string used.
    #[pyo3(get)]
    pub query: String,
    /// Matching assets.
    assets: Vec<PyAsset>,
}

#[pymethods]
impl PySearchResult {
    /// Get the list of matching assets.
    fn assets(&self) -> Vec<PyAsset> {
        self.assets.clone()
    }

    /// Number of returned assets.
    fn count(&self) -> usize {
        self.assets.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PySearchResult(query='{}', total={}, returned={})",
            self.query,
            self.total_count,
            self.assets.len()
        )
    }
}

// ---------------------------------------------------------------------------
// PyAssetManager
// ---------------------------------------------------------------------------

/// In-memory media asset catalog manager.
///
/// Manages assets, collections, tags, and search within a JSON-backed catalog.
#[pyclass]
pub struct PyAssetManager {
    catalog_path: String,
    assets: HashMap<String, PyAsset>,
    collections: HashMap<String, CollectionMeta>,
}

#[derive(Clone)]
struct CollectionMeta {
    description: String,
    created_at: String,
}

#[pymethods]
impl PyAssetManager {
    /// Create a new asset manager with a catalog path.
    #[new]
    fn new(catalog_path: &str) -> PyResult<Self> {
        let mut mgr = Self {
            catalog_path: catalog_path.to_string(),
            assets: HashMap::new(),
            collections: HashMap::new(),
        };
        // Try to load existing catalog
        if std::path::Path::new(catalog_path).exists() {
            mgr.load_from_disk()?;
        }
        Ok(mgr)
    }

    /// Open an existing catalog (classmethod).
    #[classmethod]
    fn open(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        if !std::path::Path::new(path).exists() {
            return Err(PyRuntimeError::new_err(format!(
                "Catalog not found: {path}"
            )));
        }
        Self::new(path)
    }

    /// Create a new empty catalog (classmethod).
    #[classmethod]
    fn create(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let mgr = Self {
            catalog_path: path.to_string(),
            assets: HashMap::new(),
            collections: HashMap::new(),
        };
        mgr.save_to_disk()?;
        Ok(mgr)
    }

    /// Ingest a single file into the catalog.
    #[pyo3(signature = (path, tags=None, collection=None))]
    fn ingest(
        &mut self,
        path: &str,
        tags: Option<Vec<String>>,
        collection: Option<String>,
    ) -> PyResult<PyAsset> {
        let file_path = std::path::Path::new(path);
        if !file_path.exists() {
            return Err(PyValueError::new_err(format!("File not found: {path}")));
        }
        if !file_path.is_file() {
            return Err(PyValueError::new_err(format!("Not a file: {path}")));
        }

        let meta = std::fs::metadata(file_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read metadata: {e}")))?;

        let checksum = compute_file_checksum(path)?;

        // Check for duplicates
        if self.assets.values().any(|a| a.checksum == checksum) {
            return Err(PyValueError::new_err(format!(
                "Duplicate asset (checksum {checksum})"
            )));
        }

        let asset = PyAsset {
            id: generate_id(),
            path: path.to_string(),
            filename: file_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            format: detect_format(path),
            size_bytes: meta.len(),
            duration_secs: None,
            width: None,
            height: None,
            codec: None,
            tags: tags.unwrap_or_default(),
            collection: collection.clone(),
            ingested_at: now_timestamp(),
            checksum,
            metadata: HashMap::new(),
        };

        // Ensure collection exists
        if let Some(ref coll_name) = collection {
            self.collections
                .entry(coll_name.clone())
                .or_insert_with(|| CollectionMeta {
                    description: String::new(),
                    created_at: now_timestamp(),
                });
        }

        let result = asset.clone();
        self.assets.insert(asset.id.clone(), asset);
        Ok(result)
    }

    /// Ingest all files in a directory.
    #[pyo3(signature = (dir, recursive=false, tags=None))]
    fn ingest_directory(
        &mut self,
        dir: &str,
        recursive: bool,
        tags: Option<Vec<String>>,
    ) -> PyResult<Vec<PyAsset>> {
        let dir_path = std::path::Path::new(dir);
        if !dir_path.is_dir() {
            return Err(PyValueError::new_err(format!("Not a directory: {dir}")));
        }

        let mut files = Vec::new();
        collect_files_py(dir_path, recursive, &mut files)?;

        let mut results = Vec::new();
        for f in files {
            let path_str = f.to_string_lossy().to_string();
            match self.ingest(&path_str, tags.clone(), None) {
                Ok(asset) => results.push(asset),
                Err(_) => continue, // skip duplicates and errors silently
            }
        }
        Ok(results)
    }

    /// Search assets by query string.
    #[pyo3(signature = (query, tags=None, format=None, limit=None))]
    fn search(
        &self,
        query: &str,
        tags: Option<Vec<String>>,
        format: Option<String>,
        limit: Option<u32>,
    ) -> PyResult<PySearchResult> {
        let query_lower = query.to_lowercase();
        let tag_filter = tags.unwrap_or_default();
        let max_results = limit.unwrap_or(100) as usize;

        let mut matching: Vec<PyAsset> = self
            .assets
            .values()
            .filter(|a| {
                let text_match = a.filename.to_lowercase().contains(&query_lower)
                    || a.path.to_lowercase().contains(&query_lower)
                    || a.tags
                        .iter()
                        .any(|t| t.to_lowercase().contains(&query_lower))
                    || a.collection
                        .as_ref()
                        .map_or(false, |c| c.to_lowercase().contains(&query_lower))
                    || a.format.to_lowercase().contains(&query_lower);

                let tag_ok = tag_filter.is_empty()
                    || tag_filter.iter().all(|tf| a.tags.iter().any(|t| t == tf));

                let fmt_ok = format
                    .as_ref()
                    .map_or(true, |f| a.format.eq_ignore_ascii_case(f));

                text_match && tag_ok && fmt_ok
            })
            .cloned()
            .collect();

        let total = matching.len() as u32;
        matching.truncate(max_results);

        Ok(PySearchResult {
            total_count: total,
            query: query.to_string(),
            assets: matching,
        })
    }

    /// Get an asset by ID.
    fn get_asset(&self, id: &str) -> Option<PyAsset> {
        self.assets.get(id).cloned()
    }

    /// Add a tag to an asset.
    fn add_tag(&mut self, asset_id: &str, tag: &str) -> PyResult<()> {
        let asset = self
            .assets
            .get_mut(asset_id)
            .ok_or_else(|| PyValueError::new_err(format!("Asset not found: {asset_id}")))?;
        if !asset.tags.contains(&tag.to_string()) {
            asset.tags.push(tag.to_string());
        }
        Ok(())
    }

    /// Remove a tag from an asset.
    fn remove_tag(&mut self, asset_id: &str, tag: &str) -> PyResult<()> {
        let asset = self
            .assets
            .get_mut(asset_id)
            .ok_or_else(|| PyValueError::new_err(format!("Asset not found: {asset_id}")))?;
        asset.tags.retain(|t| t != tag);
        Ok(())
    }

    /// Create a new collection.
    #[pyo3(signature = (name, description=None))]
    fn create_collection(
        &mut self,
        name: &str,
        description: Option<&str>,
    ) -> PyResult<PyCollection> {
        if self.collections.contains_key(name) {
            return Err(PyValueError::new_err(format!(
                "Collection already exists: {name}"
            )));
        }
        let meta = CollectionMeta {
            description: description.unwrap_or("").to_string(),
            created_at: now_timestamp(),
        };
        self.collections.insert(name.to_string(), meta.clone());

        Ok(PyCollection {
            name: name.to_string(),
            description: meta.description,
            asset_count: 0,
            total_size_bytes: 0,
            created_at: meta.created_at,
        })
    }

    /// List all collections.
    fn list_collections(&self) -> Vec<PyCollection> {
        self.collections
            .iter()
            .map(|(name, meta)| {
                let (count, size) = self
                    .assets
                    .values()
                    .filter(|a| a.collection.as_ref() == Some(name))
                    .fold((0u32, 0u64), |(c, s), a| (c + 1, s + a.size_bytes));
                PyCollection {
                    name: name.clone(),
                    description: meta.description.clone(),
                    asset_count: count,
                    total_size_bytes: size,
                    created_at: meta.created_at.clone(),
                }
            })
            .collect()
    }

    /// Return the total number of assets.
    fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// Save the catalog to disk.
    fn save(&self) -> PyResult<()> {
        self.save_to_disk()
    }

    /// Get catalog statistics as a dictionary.
    fn catalog_stats(&self) -> HashMap<String, String> {
        let total_size: u64 = self.assets.values().map(|a| a.size_bytes).sum();
        let mut formats: HashMap<String, usize> = HashMap::new();
        for a in self.assets.values() {
            *formats.entry(a.format.clone()).or_insert(0) += 1;
        }
        let all_tags: Vec<String> = self
            .assets
            .values()
            .flat_map(|a| a.tags.iter().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let mut m = HashMap::new();
        m.insert("total_assets".to_string(), self.assets.len().to_string());
        m.insert(
            "total_collections".to_string(),
            self.collections.len().to_string(),
        );
        m.insert("total_size_bytes".to_string(), total_size.to_string());
        let fmt_strs: Vec<String> = formats.iter().map(|(k, v)| format!("{k}:{v}")).collect();
        m.insert("formats".to_string(), fmt_strs.join(","));
        m.insert("unique_tags".to_string(), all_tags.join(","));
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyAssetManager(catalog='{}', assets={}, collections={})",
            self.catalog_path,
            self.assets.len(),
            self.collections.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Internal persistence
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
struct CatalogJson {
    version: u32,
    assets: Vec<AssetJson>,
    collections: Vec<CollectionJson>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct AssetJson {
    id: String,
    path: String,
    filename: String,
    format: String,
    size_bytes: u64,
    duration_secs: Option<f64>,
    width: Option<u32>,
    height: Option<u32>,
    codec: Option<String>,
    tags: Vec<String>,
    collection: Option<String>,
    ingested_at: String,
    checksum: String,
    metadata: HashMap<String, String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CollectionJson {
    name: String,
    description: String,
    created_at: String,
}

impl PyAssetManager {
    fn load_from_disk(&mut self) -> PyResult<()> {
        let data = std::fs::read_to_string(&self.catalog_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read catalog: {e}")))?;
        let catalog: CatalogJson = serde_json::from_str(&data)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse catalog: {e}")))?;

        self.assets.clear();
        for a in catalog.assets {
            let asset = PyAsset {
                id: a.id.clone(),
                path: a.path,
                filename: a.filename,
                format: a.format,
                size_bytes: a.size_bytes,
                duration_secs: a.duration_secs,
                width: a.width,
                height: a.height,
                codec: a.codec,
                tags: a.tags,
                collection: a.collection,
                ingested_at: a.ingested_at,
                checksum: a.checksum,
                metadata: a.metadata,
            };
            self.assets.insert(a.id, asset);
        }

        self.collections.clear();
        for c in catalog.collections {
            self.collections.insert(
                c.name,
                CollectionMeta {
                    description: c.description,
                    created_at: c.created_at,
                },
            );
        }

        Ok(())
    }

    fn save_to_disk(&self) -> PyResult<()> {
        let catalog = CatalogJson {
            version: 1,
            assets: self
                .assets
                .values()
                .map(|a| AssetJson {
                    id: a.id.clone(),
                    path: a.path.clone(),
                    filename: a.filename.clone(),
                    format: a.format.clone(),
                    size_bytes: a.size_bytes,
                    duration_secs: a.duration_secs,
                    width: a.width,
                    height: a.height,
                    codec: a.codec.clone(),
                    tags: a.tags.clone(),
                    collection: a.collection.clone(),
                    ingested_at: a.ingested_at.clone(),
                    checksum: a.checksum.clone(),
                    metadata: a.metadata.clone(),
                })
                .collect(),
            collections: self
                .collections
                .iter()
                .map(|(name, meta)| CollectionJson {
                    name: name.clone(),
                    description: meta.description.clone(),
                    created_at: meta.created_at.clone(),
                })
                .collect(),
        };

        if let Some(parent) = std::path::Path::new(&self.catalog_path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create catalog directory: {e}"))
                })?;
            }
        }

        let data = serde_json::to_string_pretty(&catalog)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize catalog: {e}")))?;
        std::fs::write(&self.catalog_path, data)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write catalog: {e}")))?;
        Ok(())
    }
}

fn collect_files_py(
    dir: &std::path::Path,
    recursive: bool,
    out: &mut Vec<std::path::PathBuf>,
) -> PyResult<()> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read dir: {e}")))?;
    for entry in entries {
        let entry = entry.map_err(|e| PyRuntimeError::new_err(format!("Dir entry error: {e}")))?;
        let path = entry.path();
        if path.is_file() {
            out.push(path);
        } else if path.is_dir() && recursive {
            collect_files_py(&path, recursive, out)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Open an existing catalog, or create a new one at the given path.
#[pyfunction]
pub fn open_catalog(path: &str) -> PyResult<PyAssetManager> {
    PyAssetManager::new(path)
}

// ---------------------------------------------------------------------------
// Registration helper
// ---------------------------------------------------------------------------

/// Register all MAM bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAsset>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyAssetManager>()?;
    m.add_function(wrap_pyfunction!(open_catalog, m)?)?;
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
            .join(format!("oximedia-py-mam-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_generate_id() {
        let id = generate_id();
        assert!(id.starts_with("asset-"));
        assert!(id.len() > 10);
    }

    #[test]
    fn test_detect_format() {
        assert_eq!(detect_format(&tmp_str("video.mkv")), "mkv");
        assert_eq!(detect_format(&tmp_str("audio.FLAC")), "flac");
        assert_eq!(detect_format("noext"), "unknown");
    }

    #[test]
    fn test_asset_has_tag() {
        let asset = PyAsset {
            id: "test".to_string(),
            path: tmp_str("test.mkv"),
            filename: "test.mkv".to_string(),
            format: "mkv".to_string(),
            size_bytes: 1024,
            duration_secs: None,
            width: None,
            height: None,
            codec: None,
            tags: vec!["raw".to_string(), "dailies".to_string()],
            collection: None,
            ingested_at: "0".to_string(),
            checksum: "abc".to_string(),
            metadata: HashMap::new(),
        };
        assert!(asset.has_tag("raw"));
        assert!(!asset.has_tag("final"));
    }

    #[test]
    fn test_collection_repr() {
        let coll = PyCollection {
            name: "dailies".to_string(),
            description: "Daily footage".to_string(),
            asset_count: 10,
            total_size_bytes: 1_000_000,
            created_at: "12345".to_string(),
        };
        let repr = coll.__repr__();
        assert!(repr.contains("dailies"));
        assert!(repr.contains("10"));
    }

    #[test]
    fn test_search_result_count() {
        let result = PySearchResult {
            total_count: 5,
            query: "test".to_string(),
            assets: vec![],
        };
        assert_eq!(result.count(), 0);
        assert_eq!(result.total_count, 5);
    }
}
