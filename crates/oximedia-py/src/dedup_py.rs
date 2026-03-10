//! Python bindings for media deduplication.
//!
//! Provides `PyDedupScanner`, `PyDedupReport`, `PyDuplicateGroup`,
//! and standalone functions for scanning and comparing media files.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn compute_hash(path: &str) -> PyResult<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open file: {e}")))?;
    let mut hasher: u64 = 0x6295c58d62b82175;
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
            hasher = hasher.wrapping_mul(0x517cc1b727220a95);
            hasher = hasher.rotate_left(31);
        }
    }
    Ok(format!("{:016x}", hasher))
}

fn collect_files_recursive(
    dir: &std::path::Path,
    out: &mut Vec<std::path::PathBuf>,
) -> PyResult<()> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read dir: {e}")))?;
    for entry in entries {
        let entry = entry.map_err(|e| PyRuntimeError::new_err(format!("Dir entry error: {e}")))?;
        let path = entry.path();
        if path.is_file() {
            out.push(path);
        } else if path.is_dir() {
            collect_files_recursive(&path, out)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PyDuplicateGroup
// ---------------------------------------------------------------------------

/// A group of duplicate files with similarity information.
#[pyclass]
#[derive(Clone)]
pub struct PyDuplicateGroup {
    /// Files in this duplicate group.
    #[pyo3(get)]
    pub files: Vec<String>,
    /// Content hash shared by this group.
    #[pyo3(get)]
    pub hash: String,
    /// Similarity score (1.0 for exact duplicates).
    #[pyo3(get)]
    pub similarity: f64,
    /// Detection method used.
    #[pyo3(get)]
    pub method: String,
}

#[pymethods]
impl PyDuplicateGroup {
    fn __repr__(&self) -> String {
        format!(
            "PyDuplicateGroup(files={}, similarity={:.2}, method='{}')",
            self.files.len(),
            self.similarity,
            self.method
        )
    }

    /// Number of files in the group.
    fn count(&self) -> usize {
        self.files.len()
    }

    /// Number of excess files (total - 1).
    fn excess_count(&self) -> usize {
        self.files.len().saturating_sub(1)
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("files".to_string(), self.files.join(","));
        m.insert("hash".to_string(), self.hash.clone());
        m.insert("similarity".to_string(), format!("{:.4}", self.similarity));
        m.insert("method".to_string(), self.method.clone());
        m
    }
}

// ---------------------------------------------------------------------------
// PyDedupReport
// ---------------------------------------------------------------------------

/// Report from a deduplication scan.
#[pyclass]
#[derive(Clone)]
pub struct PyDedupReport {
    /// Total files scanned.
    #[pyo3(get)]
    pub total_files: usize,
    /// Number of duplicate groups found.
    #[pyo3(get)]
    pub group_count: usize,
    /// Total number of duplicate files (excess).
    #[pyo3(get)]
    pub duplicate_count: usize,
    /// Strategy used for detection.
    #[pyo3(get)]
    pub strategy: String,
    groups: Vec<PyDuplicateGroup>,
}

#[pymethods]
impl PyDedupReport {
    /// Get all duplicate groups.
    fn groups(&self) -> Vec<PyDuplicateGroup> {
        self.groups.clone()
    }

    /// Total wasted size in bytes (estimated).
    fn estimated_wasted_bytes(&self) -> u64 {
        let mut total: u64 = 0;
        for group in &self.groups {
            for path in group.files.iter().skip(1) {
                let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                total += size;
            }
        }
        total
    }

    fn __repr__(&self) -> String {
        format!(
            "PyDedupReport(files={}, groups={}, duplicates={})",
            self.total_files, self.group_count, self.duplicate_count
        )
    }
}

// ---------------------------------------------------------------------------
// PyDedupScanner
// ---------------------------------------------------------------------------

/// Media deduplication scanner.
#[pyclass]
pub struct PyDedupScanner {
    threshold: f64,
    strategy: String,
    file_hashes: HashMap<String, Vec<String>>,
}

#[pymethods]
impl PyDedupScanner {
    /// Create a new dedup scanner.
    #[new]
    #[pyo3(signature = (strategy="fast", threshold=0.90))]
    fn new(strategy: &str, threshold: f64) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(PyValueError::new_err(
                "Threshold must be between 0.0 and 1.0",
            ));
        }
        Ok(Self {
            threshold,
            strategy: strategy.to_string(),
            file_hashes: HashMap::new(),
        })
    }

    /// Add a file to the scanner index.
    fn add_file(&mut self, path: &str) -> PyResult<String> {
        if !std::path::Path::new(path).exists() {
            return Err(PyValueError::new_err(format!("File not found: {path}")));
        }
        let hash = compute_hash(path)?;
        self.file_hashes
            .entry(hash.clone())
            .or_default()
            .push(path.to_string());
        Ok(hash)
    }

    /// Scan a directory for files.
    #[pyo3(signature = (dir, recursive=true))]
    fn scan_directory(&mut self, dir: &str, recursive: bool) -> PyResult<usize> {
        let dir_path = std::path::Path::new(dir);
        if !dir_path.is_dir() {
            return Err(PyValueError::new_err(format!("Not a directory: {dir}")));
        }
        let mut files = Vec::new();
        if recursive {
            collect_files_recursive(dir_path, &mut files)?;
        } else {
            let entries = std::fs::read_dir(dir_path)
                .map_err(|e| PyRuntimeError::new_err(format!("Read error: {e}")))?;
            for entry in entries {
                let entry =
                    entry.map_err(|e| PyRuntimeError::new_err(format!("Entry error: {e}")))?;
                if entry.path().is_file() {
                    files.push(entry.path());
                }
            }
        }
        let count = files.len();
        for file in files {
            let path_str = file.to_string_lossy().to_string();
            let _ = self.add_file(&path_str);
        }
        Ok(count)
    }

    /// Find duplicates from indexed files.
    fn find_duplicates(&self) -> PyDedupReport {
        let mut groups = Vec::new();
        let mut duplicate_count = 0usize;

        for (hash, paths) in &self.file_hashes {
            if paths.len() > 1 {
                duplicate_count += paths.len() - 1;
                groups.push(PyDuplicateGroup {
                    files: paths.clone(),
                    hash: hash.clone(),
                    similarity: 1.0,
                    method: self.strategy.clone(),
                });
            }
        }

        let total_files: usize = self.file_hashes.values().map(|v| v.len()).sum();

        PyDedupReport {
            total_files,
            group_count: groups.len(),
            duplicate_count,
            strategy: self.strategy.clone(),
            groups,
        }
    }

    /// Get total indexed files count.
    fn file_count(&self) -> usize {
        self.file_hashes.values().map(|v| v.len()).sum()
    }

    /// Clear the scanner index.
    fn clear(&mut self) {
        self.file_hashes.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "PyDedupScanner(strategy='{}', threshold={:.2}, files={})",
            self.strategy,
            self.threshold,
            self.file_count()
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Scan a list of files for duplicates and return a report.
#[pyfunction]
#[pyo3(signature = (paths, strategy="fast", threshold=0.90))]
pub fn scan_duplicates(
    paths: Vec<String>,
    strategy: &str,
    threshold: f64,
) -> PyResult<PyDedupReport> {
    let mut scanner = PyDedupScanner::new(strategy, threshold)?;
    for path in &paths {
        let _ = scanner.add_file(path);
    }
    Ok(scanner.find_duplicates())
}

/// Compare two files and return their similarity score.
#[pyfunction]
pub fn compare_files(file_a: &str, file_b: &str) -> PyResult<f64> {
    if !std::path::Path::new(file_a).exists() {
        return Err(PyValueError::new_err(format!("File not found: {file_a}")));
    }
    if !std::path::Path::new(file_b).exists() {
        return Err(PyValueError::new_err(format!("File not found: {file_b}")));
    }
    let hash_a = compute_hash(file_a)?;
    let hash_b = compute_hash(file_b)?;
    if hash_a == hash_b {
        Ok(1.0)
    } else {
        let size_a = std::fs::metadata(file_a).map(|m| m.len()).unwrap_or(0);
        let size_b = std::fs::metadata(file_b).map(|m| m.len()).unwrap_or(0);
        if size_a.max(size_b) > 0 {
            Ok(size_a.min(size_b) as f64 / size_a.max(size_b) as f64)
        } else {
            Ok(1.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register dedup bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDuplicateGroup>()?;
    m.add_class::<PyDedupReport>()?;
    m.add_class::<PyDedupScanner>()?;
    m.add_function(wrap_pyfunction!(scan_duplicates, m)?)?;
    m.add_function(wrap_pyfunction!(compare_files, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_hash() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_dedup_py_test.bin");
        std::fs::write(&path, b"test data for dedup").expect("write");
        let hash = compute_hash(&path.to_string_lossy());
        assert!(hash.is_ok());
        assert_eq!(hash.expect("hash").len(), 16);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_compute_hash_deterministic() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_dedup_py_det.bin");
        std::fs::write(&path, b"deterministic content").expect("write");
        let h1 = compute_hash(&path.to_string_lossy()).expect("hash1");
        let h2 = compute_hash(&path.to_string_lossy()).expect("hash2");
        assert_eq!(h1, h2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_duplicate_group_excess() {
        let group = PyDuplicateGroup {
            files: vec![
                "a.mp4".to_string(),
                "b.mp4".to_string(),
                "c.mp4".to_string(),
            ],
            hash: "abc123".to_string(),
            similarity: 1.0,
            method: "exact".to_string(),
        };
        assert_eq!(group.count(), 3);
        assert_eq!(group.excess_count(), 2);
    }

    #[test]
    fn test_scanner_add_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_dedup_py_scanner.bin");
        std::fs::write(&path, b"scanner test").expect("write");
        let mut scanner = PyDedupScanner::new("fast", 0.9).expect("new");
        let result = scanner.add_file(&path.to_string_lossy());
        assert!(result.is_ok());
        assert_eq!(scanner.file_count(), 1);
        std::fs::remove_file(&path).ok();
    }
}
