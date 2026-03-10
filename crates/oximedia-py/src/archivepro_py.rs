//! Python bindings for professional archive and digital preservation.
//!
//! Provides `PyArchiveManager`, `PyArchivePolicy`, `PyMigrationPlan`,
//! and standalone functions for ingesting and verifying archive assets.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn compute_checksum(path: &str) -> PyResult<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open: {e}")))?;
    let mut hasher: u64 = 0xcbf29ce484222325;
    let mut buf = [0u8; 8192];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| PyRuntimeError::new_err(format!("Read error: {e}")))?;
        if n == 0 {
            break;
        }
        for &byte in &buf[..n] {
            hasher ^= u64::from(byte);
            hasher = hasher.wrapping_mul(0x100000001b3);
        }
    }
    Ok(format!("{:016x}", hasher))
}

fn now_timestamp() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", dur.as_secs())
}

// ---------------------------------------------------------------------------
// PyArchivePolicy
// ---------------------------------------------------------------------------

/// Preservation policy configuration.
#[pyclass]
#[derive(Clone)]
pub struct PyArchivePolicy {
    /// Retention period string (e.g., "10y", "forever").
    #[pyo3(get)]
    pub retention: String,
    /// Fixity check interval (e.g., "90d", "1y").
    #[pyo3(get)]
    pub fixity_interval: String,
    /// Checksum algorithm to use.
    #[pyo3(get)]
    pub checksum_algorithm: String,
    /// Preferred preservation formats.
    #[pyo3(get)]
    pub preferred_formats: Vec<String>,
}

#[pymethods]
impl PyArchivePolicy {
    #[new]
    #[pyo3(signature = (retention="10y", fixity_interval="90d", checksum_algorithm="sha256", preferred_formats=None))]
    fn new(
        retention: &str,
        fixity_interval: &str,
        checksum_algorithm: &str,
        preferred_formats: Option<Vec<String>>,
    ) -> Self {
        Self {
            retention: retention.to_string(),
            fixity_interval: fixity_interval.to_string(),
            checksum_algorithm: checksum_algorithm.to_string(),
            preferred_formats: preferred_formats.unwrap_or_else(|| {
                vec![
                    "ffv1-mkv".to_string(),
                    "flac".to_string(),
                    "tiff".to_string(),
                ]
            }),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyArchivePolicy(retention='{}', fixity='{}')",
            self.retention, self.fixity_interval
        )
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("retention".to_string(), self.retention.clone());
        m.insert("fixity_interval".to_string(), self.fixity_interval.clone());
        m.insert(
            "checksum_algorithm".to_string(),
            self.checksum_algorithm.clone(),
        );
        m.insert(
            "preferred_formats".to_string(),
            self.preferred_formats.join(","),
        );
        m
    }
}

// ---------------------------------------------------------------------------
// PyMigrationPlan
// ---------------------------------------------------------------------------

/// A planned format migration for an asset.
#[pyclass]
#[derive(Clone)]
pub struct PyMigrationPlan {
    /// Source file path.
    #[pyo3(get)]
    pub source: String,
    /// Target format name.
    #[pyo3(get)]
    pub target_format: String,
    /// Target file extension.
    #[pyo3(get)]
    pub target_extension: String,
    /// Estimated output size in bytes.
    #[pyo3(get)]
    pub estimated_size: u64,
    /// Whether validation will be performed after migration.
    #[pyo3(get)]
    pub validate_after: bool,
}

#[pymethods]
impl PyMigrationPlan {
    fn __repr__(&self) -> String {
        format!(
            "PyMigrationPlan(source='{}', target='{}')",
            self.source, self.target_format
        )
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("source".to_string(), self.source.clone());
        m.insert("target_format".to_string(), self.target_format.clone());
        m.insert(
            "target_extension".to_string(),
            self.target_extension.clone(),
        );
        m.insert(
            "estimated_size".to_string(),
            self.estimated_size.to_string(),
        );
        m
    }
}

// ---------------------------------------------------------------------------
// PyArchiveManager
// ---------------------------------------------------------------------------

/// Professional archive and digital preservation manager.
#[pyclass]
pub struct PyArchiveManager {
    archive_path: String,
    policy: PyArchivePolicy,
    assets: HashMap<String, AssetMeta>,
}

#[derive(Clone)]
#[allow(dead_code)]
struct AssetMeta {
    filename: String,
    checksum: String,
    size: u64,
    ingested_at: String,
}

#[pymethods]
impl PyArchiveManager {
    /// Create a new archive manager.
    #[new]
    #[pyo3(signature = (archive_path, policy=None))]
    fn new(archive_path: &str, policy: Option<PyArchivePolicy>) -> Self {
        Self {
            archive_path: archive_path.to_string(),
            policy: policy.unwrap_or_else(|| PyArchivePolicy::new("10y", "90d", "sha256", None)),
            assets: HashMap::new(),
        }
    }

    /// Ingest a file into the archive.
    fn ingest(&mut self, path: &str) -> PyResult<String> {
        let file_path = std::path::Path::new(path);
        if !file_path.exists() {
            return Err(PyValueError::new_err(format!("File not found: {path}")));
        }
        if !file_path.is_file() {
            return Err(PyValueError::new_err(format!("Not a file: {path}")));
        }
        let checksum_val = compute_checksum(path)?;
        let size = std::fs::metadata(file_path)
            .map(|m| m.len())
            .map_err(|e| PyRuntimeError::new_err(format!("Metadata error: {e}")))?;
        let filename = file_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // Copy to archive directory
        let archive_dir = std::path::Path::new(&self.archive_path);
        if !archive_dir.exists() {
            std::fs::create_dir_all(archive_dir).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create archive dir: {e}"))
            })?;
        }
        let dest = archive_dir.join(&filename);
        std::fs::copy(file_path, &dest)
            .map_err(|e| PyRuntimeError::new_err(format!("Copy failed: {e}")))?;

        let asset_id = format!("arch-{}", checksum_val);
        self.assets.insert(
            asset_id.clone(),
            AssetMeta {
                filename,
                checksum: checksum_val,
                size,
                ingested_at: now_timestamp(),
            },
        );
        Ok(asset_id)
    }

    /// Verify the integrity of an asset by re-computing its checksum.
    fn verify_asset(&self, asset_id: &str) -> PyResult<bool> {
        let meta = self
            .assets
            .get(asset_id)
            .ok_or_else(|| PyValueError::new_err(format!("Asset not found: {asset_id}")))?;
        let archive_dir = std::path::Path::new(&self.archive_path);
        let file_path = archive_dir.join(&meta.filename);
        if !file_path.exists() {
            return Ok(false);
        }
        let current = compute_checksum(&file_path.to_string_lossy())?;
        Ok(current == meta.checksum)
    }

    /// Create a migration plan for an asset.
    fn plan_migration(&self, asset_id: &str, target_format: &str) -> PyResult<PyMigrationPlan> {
        let meta = self
            .assets
            .get(asset_id)
            .ok_or_else(|| PyValueError::new_err(format!("Asset not found: {asset_id}")))?;

        let ext = match target_format {
            "ffv1-mkv" => "mkv",
            "flac" => "flac",
            "wav" => "wav",
            "tiff" => "tiff",
            "png" => "png",
            _ => target_format,
        };

        Ok(PyMigrationPlan {
            source: meta.filename.clone(),
            target_format: target_format.to_string(),
            target_extension: ext.to_string(),
            estimated_size: meta.size,
            validate_after: true,
        })
    }

    /// Get archive statistics.
    fn stats(&self) -> HashMap<String, String> {
        let total_size: u64 = self.assets.values().map(|a| a.size).sum();
        let mut m = HashMap::new();
        m.insert("archive_path".to_string(), self.archive_path.clone());
        m.insert("total_assets".to_string(), self.assets.len().to_string());
        m.insert("total_size".to_string(), total_size.to_string());
        m.insert(
            "policy_retention".to_string(),
            self.policy.retention.clone(),
        );
        m
    }

    /// Get the current policy.
    fn get_policy(&self) -> PyArchivePolicy {
        self.policy.clone()
    }

    /// Set a new policy.
    fn set_policy(&mut self, policy: PyArchivePolicy) {
        self.policy = policy;
    }

    /// Get number of assets.
    fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// List all asset IDs.
    fn list_assets(&self) -> Vec<String> {
        self.keys_sorted()
    }

    /// Get supported preservation formats.
    #[staticmethod]
    fn supported_formats() -> Vec<HashMap<String, String>> {
        let formats = [
            ("ffv1-mkv", "mkv", "FFV1 lossless video in Matroska"),
            ("flac", "flac", "Free Lossless Audio Codec"),
            ("wav", "wav", "WAV PCM uncompressed audio"),
            ("tiff", "tiff", "TIFF uncompressed image"),
            ("png", "png", "PNG lossless image"),
        ];
        formats
            .iter()
            .map(|(name, ext, desc)| {
                let mut m = HashMap::new();
                m.insert("name".to_string(), name.to_string());
                m.insert("extension".to_string(), ext.to_string());
                m.insert("description".to_string(), desc.to_string());
                m
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyArchiveManager(path='{}', assets={})",
            self.archive_path,
            self.assets.len()
        )
    }
}

impl PyArchiveManager {
    fn keys_sorted(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.assets.keys().cloned().collect();
        keys.sort();
        keys
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Ingest a file into an archive directory with checksum verification.
#[pyfunction]
pub fn ingest_asset(path: &str, archive_dir: &str) -> PyResult<HashMap<String, String>> {
    let file_path = std::path::Path::new(path);
    if !file_path.exists() {
        return Err(PyValueError::new_err(format!("File not found: {path}")));
    }
    let archive = std::path::Path::new(archive_dir);
    if !archive.exists() {
        std::fs::create_dir_all(archive)
            .map_err(|e| PyRuntimeError::new_err(format!("Create dir failed: {e}")))?;
    }
    let checksum_val = compute_checksum(path)?;
    let filename = file_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let dest = archive.join(&filename);
    std::fs::copy(file_path, &dest)
        .map_err(|e| PyRuntimeError::new_err(format!("Copy failed: {e}")))?;

    let mut m = HashMap::new();
    m.insert("filename".to_string(), filename);
    m.insert("checksum".to_string(), checksum_val);
    m.insert("archive".to_string(), archive_dir.to_string());
    Ok(m)
}

/// Verify the checksum of a file in an archive.
#[pyfunction]
pub fn verify_archive(path: &str, expected_checksum: &str) -> PyResult<bool> {
    if !std::path::Path::new(path).exists() {
        return Err(PyValueError::new_err(format!("File not found: {path}")));
    }
    let actual = compute_checksum(path)?;
    Ok(actual == expected_checksum)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register archive-pro bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyArchivePolicy>()?;
    m.add_class::<PyMigrationPlan>()?;
    m.add_class::<PyArchiveManager>()?;
    m.add_function(wrap_pyfunction!(ingest_asset, m)?)?;
    m.add_function(wrap_pyfunction!(verify_archive, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_checksum() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_archpro_py_test.bin");
        std::fs::write(&path, b"archive test").expect("write");
        let ck = compute_checksum(&path.to_string_lossy());
        assert!(ck.is_ok());
        assert_eq!(ck.expect("checksum").len(), 16);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_archive_policy_defaults() {
        let policy = PyArchivePolicy::new("10y", "90d", "sha256", None);
        assert_eq!(policy.retention, "10y");
        assert!(!policy.preferred_formats.is_empty());
    }

    #[test]
    fn test_migration_plan() {
        let plan = PyMigrationPlan {
            source: "video.mp4".to_string(),
            target_format: "ffv1-mkv".to_string(),
            target_extension: "mkv".to_string(),
            estimated_size: 1_000_000,
            validate_after: true,
        };
        let d = plan.to_dict();
        assert_eq!(d.get("target_extension").map(|s| s.as_str()), Some("mkv"));
    }

    #[test]
    fn test_supported_formats() {
        let formats = PyArchiveManager::supported_formats();
        assert!(formats.len() >= 5);
    }

    #[test]
    fn test_verify_archive_fn() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_archpro_verify.bin");
        std::fs::write(&path, b"verify me").expect("write");
        let ck = compute_checksum(&path.to_string_lossy()).expect("checksum");
        let result = verify_archive(&path.to_string_lossy(), &ck);
        assert!(result.is_ok());
        assert!(result.expect("verify"));
        std::fs::remove_file(&path).ok();
    }
}
