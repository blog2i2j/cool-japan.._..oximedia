//! Python bindings for `oximedia-access` accessibility and access control.
//!
//! Provides `PyAccessManager`, `PyAccessPolicy`, `PyPermission` for managing
//! media asset access from Python.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyPermission
// ---------------------------------------------------------------------------

/// A permission grant for a media asset.
#[pyclass]
#[derive(Clone)]
pub struct PyPermission {
    /// Unique grant identifier.
    #[pyo3(get)]
    pub grant_id: String,
    /// Asset identifier.
    #[pyo3(get)]
    pub asset: String,
    /// Principal (user/group) identifier.
    #[pyo3(get)]
    pub principal: String,
    /// Permission level: read, write, admin, publish, review.
    #[pyo3(get)]
    pub permission: String,
    /// Expiration date (ISO 8601) or None for no expiry.
    #[pyo3(get)]
    pub expires: Option<String>,
    /// Territory restrictions (ISO 3166 codes).
    #[pyo3(get)]
    pub territories: Vec<String>,
    /// Whether this permission is currently active.
    #[pyo3(get)]
    pub active: bool,
}

#[pymethods]
impl PyPermission {
    fn __repr__(&self) -> String {
        format!(
            "PyPermission(grant_id='{}', asset='{}', principal='{}', perm='{}', active={})",
            self.grant_id, self.asset, self.principal, self.permission, self.active,
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self) -> HashMap<String, Py<PyAny>> {
        Python::attach(|py| {
            let mut m: HashMap<String, Py<PyAny>> = HashMap::new();
            m.insert(
                "grant_id".to_string(),
                self.grant_id.clone().into_pyobject(py).expect("str").into(),
            );
            m.insert(
                "asset".to_string(),
                self.asset.clone().into_pyobject(py).expect("str").into(),
            );
            m.insert(
                "principal".to_string(),
                self.principal
                    .clone()
                    .into_pyobject(py)
                    .expect("str")
                    .into(),
            );
            m.insert(
                "permission".to_string(),
                self.permission
                    .clone()
                    .into_pyobject(py)
                    .expect("str")
                    .into(),
            );
            m.insert(
                "active".to_string(),
                self.active
                    .into_pyobject(py)
                    .expect("bool")
                    .to_owned()
                    .into(),
            );
            m.insert(
                "territories".to_string(),
                self.territories
                    .clone()
                    .into_pyobject(py)
                    .expect("list")
                    .into(),
            );
            m
        })
    }
}

// ---------------------------------------------------------------------------
// PyAccessPolicy
// ---------------------------------------------------------------------------

/// An access control policy.
#[pyclass]
#[derive(Clone)]
pub struct PyAccessPolicy {
    /// Policy name.
    #[pyo3(get)]
    pub name: String,
    /// Default permission level for new assets.
    #[pyo3(get)]
    pub default_permission: String,
    /// Whether MFA is required.
    #[pyo3(get)]
    pub require_mfa: bool,
    /// Maximum session duration in minutes.
    #[pyo3(get)]
    pub max_session_minutes: u32,
    /// WCAG compliance level: A, AA, AAA.
    #[pyo3(get)]
    pub wcag_level: String,
    /// Allowed IP ranges (CIDR notation).
    #[pyo3(get)]
    pub ip_ranges: Vec<String>,
}

#[pymethods]
impl PyAccessPolicy {
    /// Create a new access policy.
    #[new]
    #[pyo3(signature = (name, default_permission=None, wcag_level=None))]
    fn new(name: &str, default_permission: Option<&str>, wcag_level: Option<&str>) -> Self {
        Self {
            name: name.to_string(),
            default_permission: default_permission.unwrap_or("read").to_string(),
            require_mfa: false,
            max_session_minutes: 480,
            wcag_level: wcag_level.unwrap_or("AA").to_string(),
            ip_ranges: Vec::new(),
        }
    }

    /// Set MFA requirement.
    fn set_require_mfa(&mut self, require: bool) {
        self.require_mfa = require;
    }

    /// Set maximum session duration.
    fn set_max_session_minutes(&mut self, minutes: u32) -> PyResult<()> {
        if minutes == 0 {
            return Err(PyValueError::new_err("max_session_minutes must be > 0"));
        }
        self.max_session_minutes = minutes;
        Ok(())
    }

    /// Add an IP range restriction.
    fn add_ip_range(&mut self, cidr: &str) {
        self.ip_ranges.push(cidr.to_string());
    }

    fn __repr__(&self) -> String {
        format!(
            "PyAccessPolicy(name='{}', default='{}', mfa={}, wcag='{}')",
            self.name, self.default_permission, self.require_mfa, self.wcag_level,
        )
    }
}

// ---------------------------------------------------------------------------
// PyAccessManager
// ---------------------------------------------------------------------------

/// An access control manager for media assets.
#[pyclass]
pub struct PyAccessManager {
    policies: Vec<PyAccessPolicy>,
    permissions: Vec<PyPermission>,
    next_grant_idx: u64,
}

#[pymethods]
impl PyAccessManager {
    /// Create a new access manager.
    #[new]
    fn new() -> Self {
        Self {
            policies: Vec::new(),
            permissions: Vec::new(),
            next_grant_idx: 0,
        }
    }

    /// Grant access to a principal for an asset.
    ///
    /// Returns:
    ///     Grant ID.
    #[pyo3(signature = (asset, principal, permission, expires=None, territories=None))]
    fn grant_access(
        &mut self,
        asset: &str,
        principal: &str,
        permission: &str,
        expires: Option<&str>,
        territories: Option<Vec<String>>,
    ) -> PyResult<String> {
        validate_permission_level(permission)?;

        self.next_grant_idx += 1;
        let grant_id = format!("grant-{}", self.next_grant_idx);

        let perm = PyPermission {
            grant_id: grant_id.clone(),
            asset: asset.to_string(),
            principal: principal.to_string(),
            permission: permission.to_string(),
            expires: expires.map(|e| e.to_string()),
            territories: territories.unwrap_or_default(),
            active: true,
        };

        self.permissions.push(perm);
        Ok(grant_id)
    }

    /// Revoke access from a principal.
    fn revoke_access(
        &mut self,
        asset: &str,
        principal: &str,
        permission: Option<&str>,
    ) -> PyResult<()> {
        let mut found = false;
        for perm in &mut self.permissions {
            if perm.asset == asset && perm.principal == principal && perm.active {
                if let Some(p) = permission {
                    if perm.permission == p {
                        perm.active = false;
                        found = true;
                    }
                } else {
                    perm.active = false;
                    found = true;
                }
            }
        }
        if !found {
            return Err(PyValueError::new_err(format!(
                "No active permission found for '{}' on '{}'",
                principal, asset
            )));
        }
        Ok(())
    }

    /// Check if a principal has a specific permission on an asset.
    fn check_permission(&self, asset: &str, principal: &str, permission: &str) -> bool {
        self.permissions.iter().any(|p| {
            p.asset == asset && p.principal == principal && p.permission == permission && p.active
        })
    }

    /// List permissions for an asset.
    #[pyo3(signature = (asset=None, principal=None, active_only=None))]
    fn list_permissions(
        &self,
        asset: Option<&str>,
        principal: Option<&str>,
        active_only: Option<bool>,
    ) -> Vec<PyPermission> {
        let active = active_only.unwrap_or(true);
        self.permissions
            .iter()
            .filter(|p| {
                let asset_match = asset.map_or(true, |a| p.asset == a);
                let principal_match = principal.map_or(true, |pr| p.principal == pr);
                let active_match = if active { p.active } else { true };
                asset_match && principal_match && active_match
            })
            .cloned()
            .collect()
    }

    /// Add a policy.
    fn add_policy(&mut self, policy: PyAccessPolicy) {
        self.policies.push(policy);
    }

    /// List all policies.
    fn list_policies(&self) -> Vec<PyAccessPolicy> {
        self.policies.clone()
    }

    /// Get the total number of active permissions.
    fn active_permission_count(&self) -> usize {
        self.permissions.iter().filter(|p| p.active).count()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyAccessManager(policies={}, permissions={}, active={})",
            self.policies.len(),
            self.permissions.len(),
            self.active_permission_count(),
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// List available permission levels.
#[pyfunction]
pub fn list_permission_levels() -> Vec<String> {
    vec![
        "read".to_string(),
        "write".to_string(),
        "admin".to_string(),
        "publish".to_string(),
        "review".to_string(),
    ]
}

/// List WCAG compliance levels.
#[pyfunction]
pub fn list_wcag_levels() -> Vec<String> {
    vec!["A".to_string(), "AA".to_string(), "AAA".to_string()]
}

/// List available access roles.
#[pyfunction]
pub fn list_access_roles() -> Vec<String> {
    vec![
        "viewer".to_string(),
        "editor".to_string(),
        "reviewer".to_string(),
        "publisher".to_string(),
        "administrator".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_permission_level(permission: &str) -> PyResult<()> {
    match permission {
        "read" | "write" | "admin" | "publish" | "review" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "Unknown permission '{}'. Expected: read, write, admin, publish, review",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all access control bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPermission>()?;
    m.add_class::<PyAccessPolicy>()?;
    m.add_class::<PyAccessManager>()?;
    m.add_function(wrap_pyfunction!(list_permission_levels, m)?)?;
    m.add_function(wrap_pyfunction!(list_wcag_levels, m)?)?;
    m.add_function(wrap_pyfunction!(list_access_roles, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_manager_grant_revoke() {
        let mut mgr = PyAccessManager::new();
        let grant_id = mgr
            .grant_access("video-001", "user@example.com", "read", None, None)
            .expect("grant should succeed");
        assert!(grant_id.starts_with("grant-"));
        assert_eq!(mgr.active_permission_count(), 1);
        assert!(mgr.check_permission("video-001", "user@example.com", "read"));
        assert!(!mgr.check_permission("video-001", "user@example.com", "write"));

        mgr.revoke_access("video-001", "user@example.com", Some("read"))
            .expect("revoke should succeed");
        assert_eq!(mgr.active_permission_count(), 0);
        assert!(!mgr.check_permission("video-001", "user@example.com", "read"));
    }

    #[test]
    fn test_invalid_permission() {
        let mut mgr = PyAccessManager::new();
        let result = mgr.grant_access("v1", "u1", "superadmin", None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_policy_creation() {
        let mut policy = PyAccessPolicy::new("default", None, None);
        assert_eq!(policy.default_permission, "read");
        assert_eq!(policy.wcag_level, "AA");
        assert!(!policy.require_mfa);

        policy.set_require_mfa(true);
        assert!(policy.require_mfa);
        assert!(policy.set_max_session_minutes(120).is_ok());
        assert!(policy.set_max_session_minutes(0).is_err());
    }

    #[test]
    fn test_list_permissions_filter() {
        let mut mgr = PyAccessManager::new();
        let _ = mgr.grant_access("v1", "u1", "read", None, None);
        let _ = mgr.grant_access("v1", "u2", "write", None, None);
        let _ = mgr.grant_access("v2", "u1", "admin", None, None);

        let v1_perms = mgr.list_permissions(Some("v1"), None, None);
        assert_eq!(v1_perms.len(), 2);

        let u1_perms = mgr.list_permissions(None, Some("u1"), None);
        assert_eq!(u1_perms.len(), 2);
    }

    #[test]
    fn test_standalone_functions() {
        let levels = list_permission_levels();
        assert!(levels.contains(&"read".to_string()));
        let wcag = list_wcag_levels();
        assert!(wcag.contains(&"AA".to_string()));
        let roles = list_access_roles();
        assert!(roles.contains(&"editor".to_string()));
    }
}
