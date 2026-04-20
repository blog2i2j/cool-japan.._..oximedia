//! Python bindings for `oximedia-rights` digital rights and license management.
//!
//! Provides `PyRightsManager`, `PyLicense`, `PyRightsReport` for managing
//! content rights from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyLicense
// ---------------------------------------------------------------------------

/// A content license.
#[pyclass]
#[derive(Clone)]
pub struct PyLicense {
    /// Unique license identifier.
    #[pyo3(get)]
    pub license_id: String,
    /// Asset identifier.
    #[pyo3(get)]
    pub asset: String,
    /// License type: royalty-free, rights-managed, editorial, creative-commons.
    #[pyo3(get)]
    pub license_type: String,
    /// Licensee name/ID.
    #[pyo3(get)]
    pub licensee: String,
    /// Territory scope.
    #[pyo3(get)]
    pub territory: String,
    /// Start date (ISO 8601).
    #[pyo3(get)]
    pub start_date: Option<String>,
    /// End date (ISO 8601).
    #[pyo3(get)]
    pub end_date: Option<String>,
    /// Permitted uses.
    #[pyo3(get)]
    pub permitted_uses: Vec<String>,
    /// Whether license is currently active.
    #[pyo3(get)]
    pub active: bool,
    /// Royalty rate percentage.
    #[pyo3(get)]
    pub royalty_rate: Option<f64>,
}

#[pymethods]
impl PyLicense {
    fn __repr__(&self) -> String {
        format!(
            "PyLicense(id='{}', asset='{}', type='{}', licensee='{}', active={})",
            self.license_id, self.asset, self.license_type, self.licensee, self.active,
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        Python::attach(|py| -> PyResult<HashMap<String, Py<PyAny>>> {
            let mut m: HashMap<String, Py<PyAny>> = HashMap::new();
            m.insert(
                "license_id".to_string(),
                self.license_id
                    .clone()
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "asset".to_string(),
                self.asset
                    .clone()
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "license_type".to_string(),
                self.license_type
                    .clone()
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "licensee".to_string(),
                self.licensee
                    .clone()
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "territory".to_string(),
                self.territory
                    .clone()
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            m.insert(
                "active".to_string(),
                self.active
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .to_owned()
                    .into(),
            );
            m.insert(
                "permitted_uses".to_string(),
                self.permitted_uses
                    .clone()
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            Ok(m)
        })
    }
}

// ---------------------------------------------------------------------------
// PyRightsReport
// ---------------------------------------------------------------------------

/// A rights report entry.
#[pyclass]
#[derive(Clone)]
pub struct PyRightsReport {
    /// Report type.
    #[pyo3(get)]
    pub report_type: String,
    /// Total rights entries.
    #[pyo3(get)]
    pub total_entries: u32,
    /// Active licenses count.
    #[pyo3(get)]
    pub active_licenses: u32,
    /// Expiring soon count.
    #[pyo3(get)]
    pub expiring_soon: u32,
    /// Total royalty obligations.
    #[pyo3(get)]
    pub total_royalties: f64,
    /// Report generation date.
    #[pyo3(get)]
    pub generated_at: String,
}

#[pymethods]
impl PyRightsReport {
    fn __repr__(&self) -> String {
        format!(
            "PyRightsReport(type='{}', total={}, active={}, expiring={})",
            self.report_type, self.total_entries, self.active_licenses, self.expiring_soon,
        )
    }
}

// ---------------------------------------------------------------------------
// PyRightsManager
// ---------------------------------------------------------------------------

/// Digital rights and license manager.
#[pyclass]
pub struct PyRightsManager {
    rights: Vec<RightsEntry>,
    licenses: Vec<PyLicense>,
    next_rights_idx: u64,
    next_license_idx: u64,
}

/// Internal rights entry (not exposed to Python directly).
#[derive(Clone)]
#[allow(dead_code)]
struct RightsEntry {
    rights_id: String,
    asset: String,
    holder: String,
    rights_type: String,
    territory: String,
    start_date: Option<String>,
    end_date: Option<String>,
    royalty_rate: Option<f64>,
    active: bool,
}

#[pymethods]
impl PyRightsManager {
    /// Create a new rights manager.
    #[new]
    fn new() -> Self {
        Self {
            rights: Vec::new(),
            licenses: Vec::new(),
            next_rights_idx: 0,
            next_license_idx: 0,
        }
    }

    /// Register rights for an asset.
    ///
    /// Returns:
    ///     Rights ID.
    #[pyo3(signature = (asset, holder, rights_type, territory=None, start_date=None, end_date=None, royalty_rate=None))]
    fn register_rights(
        &mut self,
        asset: &str,
        holder: &str,
        rights_type: &str,
        territory: Option<&str>,
        start_date: Option<&str>,
        end_date: Option<&str>,
        royalty_rate: Option<f64>,
    ) -> PyResult<String> {
        validate_rights_type(rights_type)?;

        self.next_rights_idx += 1;
        let rights_id = format!("rights-{}", self.next_rights_idx);

        let entry = RightsEntry {
            rights_id: rights_id.clone(),
            asset: asset.to_string(),
            holder: holder.to_string(),
            rights_type: rights_type.to_string(),
            territory: territory.unwrap_or("worldwide").to_string(),
            start_date: start_date.map(|s| s.to_string()),
            end_date: end_date.map(|s| s.to_string()),
            royalty_rate,
            active: true,
        };

        self.rights.push(entry);
        Ok(rights_id)
    }

    /// Check if rights are cleared for an intended use.
    #[pyo3(signature = (asset, intended_use=None, territory=None))]
    fn check_rights(
        &self,
        asset: &str,
        intended_use: Option<&str>,
        territory: Option<&str>,
    ) -> HashMap<String, Py<PyAny>> {
        let matching_count = self
            .rights
            .iter()
            .filter(|r| r.asset == asset && r.active)
            .count();
        let cleared = matching_count > 0;
        let intended = intended_use.unwrap_or("unspecified").to_string();
        let terr = territory.unwrap_or("worldwide").to_string();
        let asset_str = asset.to_string();

        Python::attach(|py| -> PyResult<HashMap<String, Py<PyAny>>> {
            let mut result: HashMap<String, Py<PyAny>> = HashMap::new();
            result.insert(
                "asset".to_string(),
                asset_str
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            result.insert(
                "cleared".to_string(),
                cleared
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .to_owned()
                    .into(),
            );
            result.insert(
                "rights_count".to_string(),
                matching_count
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            result.insert(
                "intended_use".to_string(),
                intended
                    .into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            result.insert(
                "territory".to_string(),
                terr.into_pyobject(py)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into(),
            );
            Ok(result)
        })
        .unwrap_or_default()
    }

    /// Create a license for an asset.
    ///
    /// Returns:
    ///     License ID.
    #[pyo3(signature = (asset, license_type, licensee, territory=None, start_date=None, end_date=None, uses=None, royalty_rate=None))]
    fn create_license(
        &mut self,
        asset: &str,
        license_type: &str,
        licensee: &str,
        territory: Option<&str>,
        start_date: Option<&str>,
        end_date: Option<&str>,
        uses: Option<Vec<String>>,
        royalty_rate: Option<f64>,
    ) -> PyResult<String> {
        validate_license_type(license_type)?;

        self.next_license_idx += 1;
        let license_id = format!("lic-{}", self.next_license_idx);

        let license = PyLicense {
            license_id: license_id.clone(),
            asset: asset.to_string(),
            license_type: license_type.to_string(),
            licensee: licensee.to_string(),
            territory: territory.unwrap_or("worldwide").to_string(),
            start_date: start_date.map(|s| s.to_string()),
            end_date: end_date.map(|s| s.to_string()),
            permitted_uses: uses.unwrap_or_default(),
            active: true,
            royalty_rate,
        };

        self.licenses.push(license);
        Ok(license_id)
    }

    /// Revoke a license.
    fn revoke_license(&mut self, license_id: &str) -> PyResult<()> {
        let lic = self
            .licenses
            .iter_mut()
            .find(|l| l.license_id == license_id)
            .ok_or_else(|| PyValueError::new_err(format!("License '{}' not found", license_id)))?;
        lic.active = false;
        Ok(())
    }

    /// List all licenses, optionally filtered by asset.
    #[pyo3(signature = (asset=None, active_only=None))]
    fn list_licenses(&self, asset: Option<&str>, active_only: Option<bool>) -> Vec<PyLicense> {
        let active = active_only.unwrap_or(false);
        self.licenses
            .iter()
            .filter(|l| {
                let asset_match = asset.map_or(true, |a| l.asset == a);
                let active_match = if active { l.active } else { true };
                asset_match && active_match
            })
            .cloned()
            .collect()
    }

    /// Generate a rights summary report.
    fn generate_report(&self) -> PyRightsReport {
        let now = chrono::Utc::now().to_rfc3339();
        PyRightsReport {
            report_type: "summary".to_string(),
            total_entries: self.rights.len() as u32,
            active_licenses: self.licenses.iter().filter(|l| l.active).count() as u32,
            expiring_soon: 0,
            total_royalties: self
                .licenses
                .iter()
                .filter(|l| l.active)
                .filter_map(|l| l.royalty_rate)
                .sum(),
            generated_at: now,
        }
    }

    /// Get total rights entries count.
    fn rights_count(&self) -> usize {
        self.rights.len()
    }

    /// Get total license count.
    fn license_count(&self) -> usize {
        self.licenses.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyRightsManager(rights={}, licenses={})",
            self.rights.len(),
            self.licenses.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// List available rights types.
#[pyfunction]
pub fn list_rights_types() -> Vec<String> {
    vec![
        "master".to_string(),
        "sync".to_string(),
        "mechanical".to_string(),
        "performance".to_string(),
        "reproduction".to_string(),
        "distribution".to_string(),
    ]
}

/// List available license types.
#[pyfunction]
pub fn list_license_types() -> Vec<String> {
    vec![
        "royalty-free".to_string(),
        "rights-managed".to_string(),
        "editorial".to_string(),
        "creative-commons".to_string(),
    ]
}

/// List supported territory codes (subset).
#[pyfunction]
pub fn list_territory_codes() -> Vec<String> {
    vec![
        "worldwide".to_string(),
        "US".to_string(),
        "GB".to_string(),
        "EU".to_string(),
        "JP".to_string(),
        "CA".to_string(),
        "AU".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_rights_type(rights_type: &str) -> PyResult<()> {
    match rights_type {
        "master" | "sync" | "mechanical" | "performance" | "reproduction" | "distribution" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "Unknown rights type '{}'. Expected: master, sync, mechanical, performance, reproduction, distribution",
            other
        ))),
    }
}

fn validate_license_type(license_type: &str) -> PyResult<()> {
    match license_type {
        "royalty-free" | "rights-managed" | "editorial" | "creative-commons" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "Unknown license type '{}'. Expected: royalty-free, rights-managed, editorial, creative-commons",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all rights management bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLicense>()?;
    m.add_class::<PyRightsReport>()?;
    m.add_class::<PyRightsManager>()?;
    m.add_function(wrap_pyfunction!(list_rights_types, m)?)?;
    m.add_function(wrap_pyfunction!(list_license_types, m)?)?;
    m.add_function(wrap_pyfunction!(list_territory_codes, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_check_rights() {
        pyo3::Python::initialize();
        let mut mgr = PyRightsManager::new();
        let rid = mgr
            .register_rights(
                "song-001",
                "Label Inc",
                "master",
                None,
                None,
                None,
                Some(5.0),
            )
            .expect("register should succeed");
        assert!(rid.starts_with("rights-"));
        assert_eq!(mgr.rights_count(), 1);

        Python::attach(|py| {
            let result = mgr.check_rights("song-001", None, None);
            let cleared = result
                .get("cleared")
                .and_then(|v| v.extract::<bool>(py).ok())
                .unwrap_or(false);
            assert!(cleared);

            let result2 = mgr.check_rights("nonexistent", None, None);
            let cleared2 = result2
                .get("cleared")
                .and_then(|v| v.extract::<bool>(py).ok())
                .unwrap_or(false);
            assert!(!cleared2);
        });
    }

    #[test]
    fn test_create_and_revoke_license() {
        let mut mgr = PyRightsManager::new();
        let lid = mgr
            .create_license(
                "video-001",
                "rights-managed",
                "Streamer Co",
                None,
                None,
                None,
                Some(vec!["streaming".to_string()]),
                Some(3.0),
            )
            .expect("create license should succeed");
        assert!(lid.starts_with("lic-"));
        assert_eq!(mgr.license_count(), 1);

        let active = mgr.list_licenses(None, Some(true));
        assert_eq!(active.len(), 1);

        mgr.revoke_license(&lid).expect("revoke should succeed");
        let active = mgr.list_licenses(None, Some(true));
        assert_eq!(active.len(), 0);
    }

    #[test]
    fn test_invalid_types() {
        let mut mgr = PyRightsManager::new();
        assert!(mgr
            .register_rights("a", "h", "invalid", None, None, None, None)
            .is_err());
        assert!(mgr
            .create_license("a", "invalid", "l", None, None, None, None, None)
            .is_err());
    }

    #[test]
    fn test_generate_report() {
        let mut mgr = PyRightsManager::new();
        let _ = mgr.register_rights("a", "h", "master", None, None, None, None);
        let _ = mgr.create_license("a", "royalty-free", "l", None, None, None, None, Some(2.0));

        let report = mgr.generate_report();
        assert_eq!(report.report_type, "summary");
        assert_eq!(report.total_entries, 1);
        assert_eq!(report.active_licenses, 1);
    }

    #[test]
    fn test_standalone_functions() {
        let rights = list_rights_types();
        assert!(rights.contains(&"master".to_string()));
        let licenses = list_license_types();
        assert!(licenses.contains(&"royalty-free".to_string()));
        let territories = list_territory_codes();
        assert!(territories.contains(&"worldwide".to_string()));
    }
}
