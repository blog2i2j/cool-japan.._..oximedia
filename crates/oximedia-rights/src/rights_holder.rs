//! Rights holder registry for tracking ownership of media assets.
//!
//! A rights holder is any entity — individual, company, or collective — that
//! holds a legal interest in a piece of content.  This module provides:
//!
//! - [`HolderType`]: an enum classifying the kind of entity.
//! - [`RightsHolder`]: a record describing a single rights holder.
//! - [`RightsHolderRegistry`]: a simple in-memory store that can look up
//!   holders by id or name.

#![allow(dead_code)]

use std::collections::HashMap;

/// Classification of the type of entity that holds rights.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HolderType {
    /// A natural person (e.g. an author or composer).
    Individual,
    /// A business entity such as a studio or publisher.
    Company,
    /// A collecting society or PRO representing multiple rights holders.
    CollectingSociety,
    /// A government body or public-domain custodian.
    Government,
    /// Any entity that does not fit the above categories.
    Other(String),
}

impl HolderType {
    /// Return a human-readable name for this holder type.
    #[must_use]
    pub fn display_name(&self) -> &str {
        match self {
            Self::Individual => "Individual",
            Self::Company => "Company",
            Self::CollectingSociety => "Collecting Society",
            Self::Government => "Government",
            Self::Other(s) => s.as_str(),
        }
    }
}

/// A record describing a single rights holder.
///
/// # Example
///
/// ```
/// use oximedia_rights::rights_holder::{HolderType, RightsHolder};
///
/// let holder = RightsHolder::new("rh-001", "Acme Music Ltd", HolderType::Company)
///     .with_jurisdiction("US")
///     .with_contact("licensing@acme.example");
///
/// assert_eq!(holder.name, "Acme Music Ltd");
/// assert_eq!(holder.jurisdiction.as_deref(), Some("US"));
/// ```
#[derive(Debug, Clone)]
pub struct RightsHolder {
    /// Unique identifier for this rights holder.
    pub id: String,
    /// Display name (person name or company name).
    pub name: String,
    /// Classification of the entity.
    pub holder_type: HolderType,
    /// ISO 3166-1 alpha-2 country code of the primary jurisdiction.
    pub jurisdiction: Option<String>,
    /// Contact email or URL for licensing enquiries.
    pub contact: Option<String>,
    /// Percentage share of rights (0.0 – 100.0); `None` means unspecified.
    pub share_percent: Option<f64>,
}

impl RightsHolder {
    /// Create a new `RightsHolder` with the given id, name, and type.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, holder_type: HolderType) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            holder_type,
            jurisdiction: None,
            contact: None,
            share_percent: None,
        }
    }

    /// Set the primary jurisdiction.
    #[must_use]
    pub fn with_jurisdiction(mut self, jurisdiction: impl Into<String>) -> Self {
        self.jurisdiction = Some(jurisdiction.into());
        self
    }

    /// Set the contact information.
    #[must_use]
    pub fn with_contact(mut self, contact: impl Into<String>) -> Self {
        self.contact = Some(contact.into());
        self
    }

    /// Set the rights share percentage.
    ///
    /// Values outside `[0.0, 100.0]` are clamped silently.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn with_share(mut self, percent: f64) -> Self {
        self.share_percent = Some(percent.clamp(0.0, 100.0));
        self
    }

    /// Returns `true` if this holder is an individual natural person.
    #[must_use]
    pub fn is_individual(&self) -> bool {
        matches!(self.holder_type, HolderType::Individual)
    }
}

/// An in-memory registry of [`RightsHolder`] records.
///
/// Provides fast lookup by either id or name.
///
/// # Example
///
/// ```
/// use oximedia_rights::rights_holder::{HolderType, RightsHolder, RightsHolderRegistry};
///
/// let mut reg = RightsHolderRegistry::new();
/// reg.register(RightsHolder::new("rh-1", "Alice Author", HolderType::Individual));
/// assert!(reg.find_by_id("rh-1").is_some());
/// ```
#[derive(Debug, Default)]
pub struct RightsHolderRegistry {
    holders: HashMap<String, RightsHolder>,
}

impl RightsHolderRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            holders: HashMap::new(),
        }
    }

    /// Add a rights holder to the registry.
    ///
    /// If a holder with the same id already exists it is replaced.
    pub fn register(&mut self, holder: RightsHolder) {
        self.holders.insert(holder.id.clone(), holder);
    }

    /// Remove a holder by id.  Returns the removed holder if it existed.
    pub fn remove(&mut self, id: &str) -> Option<RightsHolder> {
        self.holders.remove(id)
    }

    /// Look up a holder by its unique id.
    #[must_use]
    pub fn find_by_id(&self, id: &str) -> Option<&RightsHolder> {
        self.holders.get(id)
    }

    /// Find all holders whose name contains the given substring (case-insensitive).
    #[must_use]
    pub fn find_by_name(&self, name_fragment: &str) -> Vec<&RightsHolder> {
        let needle = name_fragment.to_lowercase();
        self.holders
            .values()
            .filter(|h| h.name.to_lowercase().contains(&needle))
            .collect()
    }

    /// Return all holders of the specified type.
    #[must_use]
    pub fn by_type(&self, holder_type: &HolderType) -> Vec<&RightsHolder> {
        self.holders
            .values()
            .filter(|h| &h.holder_type == holder_type)
            .collect()
    }

    /// Return the number of registered holders.
    #[must_use]
    pub fn len(&self) -> usize {
        self.holders.len()
    }

    /// Returns `true` if the registry contains no holders.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.holders.is_empty()
    }

    /// Return total share percentage of all holders that have one set.
    ///
    /// Useful for validating that shares sum to 100 %.
    #[must_use]
    pub fn total_share(&self) -> f64 {
        self.holders.values().filter_map(|h| h.share_percent).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_registry() -> RightsHolderRegistry {
        let mut reg = RightsHolderRegistry::new();
        reg.register(
            RightsHolder::new("rh-1", "Alice Author", HolderType::Individual)
                .with_jurisdiction("GB")
                .with_share(50.0),
        );
        reg.register(
            RightsHolder::new("rh-2", "Acme Music Ltd", HolderType::Company).with_share(30.0),
        );
        reg.register(
            RightsHolder::new("rh-3", "PRS For Music", HolderType::CollectingSociety)
                .with_share(20.0),
        );
        reg
    }

    #[test]
    fn test_registry_len() {
        assert_eq!(make_registry().len(), 3);
    }

    #[test]
    fn test_find_by_id_found() {
        let reg = make_registry();
        let h = reg
            .find_by_id("rh-1")
            .expect("rights test operation should succeed");
        assert_eq!(h.name, "Alice Author");
    }

    #[test]
    fn test_find_by_id_not_found() {
        assert!(make_registry().find_by_id("unknown").is_none());
    }

    #[test]
    fn test_find_by_name_partial() {
        let reg = make_registry();
        let hits = reg.find_by_name("alice");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "rh-1");
    }

    #[test]
    fn test_find_by_name_no_match() {
        let reg = make_registry();
        assert!(reg.find_by_name("nobody").is_empty());
    }

    #[test]
    fn test_by_type_individual() {
        let reg = make_registry();
        let individuals = reg.by_type(&HolderType::Individual);
        assert_eq!(individuals.len(), 1);
    }

    #[test]
    fn test_by_type_collecting_society() {
        let reg = make_registry();
        let cs = reg.by_type(&HolderType::CollectingSociety);
        assert_eq!(cs.len(), 1);
    }

    #[test]
    fn test_total_share() {
        let reg = make_registry();
        let total = reg.total_share();
        assert!((total - 100.0_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_remove() {
        let mut reg = make_registry();
        let removed = reg.remove("rh-2");
        assert!(removed.is_some());
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut reg = make_registry();
        assert!(reg.remove("ghost").is_none());
    }

    #[test]
    fn test_is_individual() {
        let h = RightsHolder::new("x", "Bob", HolderType::Individual);
        assert!(h.is_individual());
        let h2 = RightsHolder::new("y", "Corp", HolderType::Company);
        assert!(!h2.is_individual());
    }

    #[test]
    fn test_share_clamped_above() {
        let h = RightsHolder::new("z", "Test", HolderType::Individual).with_share(150.0);
        assert_eq!(h.share_percent, Some(100.0));
    }

    #[test]
    fn test_share_clamped_below() {
        let h = RightsHolder::new("z", "Test", HolderType::Individual).with_share(-10.0);
        assert_eq!(h.share_percent, Some(0.0));
    }

    #[test]
    fn test_holder_type_display() {
        assert_eq!(HolderType::Individual.display_name(), "Individual");
        assert_eq!(HolderType::Company.display_name(), "Company");
        assert_eq!(HolderType::Other("NGO".to_string()).display_name(), "NGO");
    }

    #[test]
    fn test_empty_registry() {
        let reg = RightsHolderRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.total_share(), 0.0);
    }
}
