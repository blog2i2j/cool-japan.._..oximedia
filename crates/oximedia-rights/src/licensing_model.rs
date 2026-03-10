//! Licensing model definitions for structured license type management.

#![allow(dead_code)]

use std::collections::HashMap;

/// Type of license grant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LicenseType {
    /// Exclusive rights — no other licensee may hold the same rights.
    Exclusive,
    /// Non-exclusive rights — multiple licensees may hold the same rights.
    NonExclusive,
    /// Sublicensable rights — the licensee may grant sub-licenses.
    Sublicensable,
    /// Work-for-hire arrangement — rights vest immediately in the hiring party.
    WorkForHire,
}

impl LicenseType {
    /// Returns `true` if rights under this type can be transferred to a third party.
    pub fn is_transferable(&self) -> bool {
        match self {
            LicenseType::Exclusive => true,
            LicenseType::NonExclusive => false,
            LicenseType::Sublicensable => true,
            LicenseType::WorkForHire => true,
        }
    }

    /// Human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            LicenseType::Exclusive => "Exclusive",
            LicenseType::NonExclusive => "Non-Exclusive",
            LicenseType::Sublicensable => "Sublicensable",
            LicenseType::WorkForHire => "Work For Hire",
        }
    }
}

/// A concrete license model describing what a holder may do with a work.
#[derive(Debug, Clone)]
pub struct LicenseModel {
    /// Unique identifier for this model.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Type classification.
    pub license_type: LicenseType,
    /// Whether modification of the work is permitted.
    pub allows_modification: bool,
    /// Whether commercial use is permitted.
    pub allows_commercial: bool,
    /// Optional maximum number of sub-licenses that may be granted.
    pub sublicense_limit: Option<u32>,
}

impl LicenseModel {
    /// Create a new `LicenseModel`.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        license_type: LicenseType,
        allows_modification: bool,
        allows_commercial: bool,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            license_type,
            allows_modification,
            allows_commercial,
            sublicense_limit: None,
        }
    }

    /// Returns `true` when this model permits granting sub-licenses.
    pub fn has_sublicense_right(&self) -> bool {
        self.license_type == LicenseType::Sublicensable
            && self.sublicense_limit.is_none_or(|lim| lim > 0)
    }

    /// Set an optional upper bound on the number of sub-licenses.
    pub fn with_sublicense_limit(mut self, limit: u32) -> Self {
        self.sublicense_limit = Some(limit);
        self
    }
}

/// Registry that stores and retrieves `LicenseModel` instances by ID.
#[derive(Debug, Default)]
pub struct LicenseModelRegistry {
    models: HashMap<String, LicenseModel>,
}

impl LicenseModelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a model to the registry, overwriting any existing entry with the same ID.
    pub fn add(&mut self, model: LicenseModel) {
        self.models.insert(model.id.clone(), model);
    }

    /// Find a model by ID, returning `None` if not present.
    pub fn find(&self, id: &str) -> Option<&LicenseModel> {
        self.models.get(id)
    }

    /// Return the total number of registered models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Returns `true` when the registry contains no models.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Remove a model by ID, returning it if present.
    pub fn remove(&mut self, id: &str) -> Option<LicenseModel> {
        self.models.remove(id)
    }

    /// Iterate over all registered models.
    pub fn iter(&self) -> impl Iterator<Item = &LicenseModel> {
        self.models.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exclusive_model() -> LicenseModel {
        LicenseModel::new(
            "excl-01",
            "Exclusive Film",
            LicenseType::Exclusive,
            true,
            true,
        )
    }

    fn nonexclusive_model() -> LicenseModel {
        LicenseModel::new(
            "nonexcl-01",
            "Stock",
            LicenseType::NonExclusive,
            false,
            true,
        )
    }

    fn sublicensable_model() -> LicenseModel {
        LicenseModel::new(
            "sub-01",
            "Platform Sub",
            LicenseType::Sublicensable,
            true,
            true,
        )
    }

    fn wfh_model() -> LicenseModel {
        LicenseModel::new(
            "wfh-01",
            "Work For Hire",
            LicenseType::WorkForHire,
            true,
            true,
        )
    }

    #[test]
    fn test_exclusive_is_transferable() {
        assert!(LicenseType::Exclusive.is_transferable());
    }

    #[test]
    fn test_non_exclusive_not_transferable() {
        assert!(!LicenseType::NonExclusive.is_transferable());
    }

    #[test]
    fn test_sublicensable_is_transferable() {
        assert!(LicenseType::Sublicensable.is_transferable());
    }

    #[test]
    fn test_wfh_is_transferable() {
        assert!(LicenseType::WorkForHire.is_transferable());
    }

    #[test]
    fn test_display_names() {
        assert_eq!(LicenseType::Exclusive.display_name(), "Exclusive");
        assert_eq!(LicenseType::NonExclusive.display_name(), "Non-Exclusive");
        assert_eq!(LicenseType::Sublicensable.display_name(), "Sublicensable");
        assert_eq!(LicenseType::WorkForHire.display_name(), "Work For Hire");
    }

    #[test]
    fn test_model_creation() {
        let m = exclusive_model();
        assert_eq!(m.id, "excl-01");
        assert!(m.allows_commercial);
        assert!(m.allows_modification);
    }

    #[test]
    fn test_sublicense_right_true_for_sublicensable() {
        let m = sublicensable_model();
        assert!(m.has_sublicense_right());
    }

    #[test]
    fn test_sublicense_right_false_for_exclusive() {
        let m = exclusive_model();
        assert!(!m.has_sublicense_right());
    }

    #[test]
    fn test_sublicense_limit_zero_disables_right() {
        let m = sublicensable_model().with_sublicense_limit(0);
        assert!(!m.has_sublicense_right());
    }

    #[test]
    fn test_sublicense_limit_nonzero_preserves_right() {
        let m = sublicensable_model().with_sublicense_limit(5);
        assert!(m.has_sublicense_right());
    }

    #[test]
    fn test_registry_add_and_find() {
        let mut reg = LicenseModelRegistry::new();
        reg.add(exclusive_model());
        assert!(reg.find("excl-01").is_some());
    }

    #[test]
    fn test_registry_find_missing_returns_none() {
        let reg = LicenseModelRegistry::new();
        assert!(reg.find("no-such-id").is_none());
    }

    #[test]
    fn test_registry_len() {
        let mut reg = LicenseModelRegistry::new();
        reg.add(exclusive_model());
        reg.add(nonexclusive_model());
        reg.add(sublicensable_model());
        reg.add(wfh_model());
        assert_eq!(reg.len(), 4);
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = LicenseModelRegistry::new();
        reg.add(exclusive_model());
        let removed = reg.remove("excl-01");
        assert!(removed.is_some());
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_overwrite() {
        let mut reg = LicenseModelRegistry::new();
        reg.add(exclusive_model());
        let updated = LicenseModel::new("excl-01", "Updated", LicenseType::Exclusive, false, false);
        reg.add(updated);
        assert_eq!(reg.len(), 1);
        assert_eq!(
            reg.find("excl-01")
                .expect("rights test operation should succeed")
                .name,
            "Updated"
        );
    }
}
