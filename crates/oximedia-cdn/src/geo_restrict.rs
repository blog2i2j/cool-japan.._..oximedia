//! Geographic content restrictions.
//!
//! [`GeoRestriction`] maintains allow-lists and block-lists of ISO 3166-1
//! alpha-2 country codes and resolves whether a given country may access
//! content.
//!
//! # Resolution rules
//!
//! 1. If the country is explicitly **blocked** → deny.
//! 2. If the allow-list is **non-empty** and the country is **not** in it →
//!    deny.
//! 3. Otherwise → allow.
//!
//! An empty allow-list means "all countries are permitted (unless blocked)".

use std::collections::HashSet;

/// Geographic restriction policy.
///
/// # Example
/// ```
/// use oximedia_cdn::geo_restrict::GeoRestriction;
///
/// let mut r = GeoRestriction::new();
/// r.allow("US");
/// r.allow("CA");
/// assert!(r.is_allowed("US"));
/// assert!(!r.is_allowed("GB")); // not in allow-list
/// r.block("US");
/// assert!(!r.is_allowed("US")); // explicitly blocked overrides allow
/// ```
#[derive(Debug, Default, Clone)]
pub struct GeoRestriction {
    /// Countries that are explicitly permitted (empty = all permitted).
    allowed: HashSet<String>,
    /// Countries that are explicitly denied (takes priority over allow-list).
    blocked: HashSet<String>,
}

impl GeoRestriction {
    /// Create a new, permissive restriction policy (everything allowed).
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `country` to the allow-list.
    ///
    /// Once any country is added to the allow-list, only listed countries
    /// (that are not also blocked) are permitted.
    pub fn allow(&mut self, country: &str) {
        self.allowed.insert(country.to_uppercase());
    }

    /// Add `country` to the block-list.
    ///
    /// Blocked countries are always denied regardless of the allow-list.
    pub fn block(&mut self, country: &str) {
        self.blocked.insert(country.to_uppercase());
    }

    /// Returns `true` if `country` is permitted to access the content.
    pub fn is_allowed(&self, country: &str) -> bool {
        let country = country.to_uppercase();
        // Blocked takes priority
        if self.blocked.contains(&country) {
            return false;
        }
        // If allow-list is non-empty, country must be in it
        if !self.allowed.is_empty() && !self.allowed.contains(&country) {
            return false;
        }
        true
    }

    /// Remove `country` from the allow-list.
    pub fn remove_allow(&mut self, country: &str) {
        self.allowed.remove(&country.to_uppercase());
    }

    /// Remove `country` from the block-list.
    pub fn remove_block(&mut self, country: &str) {
        self.blocked.remove(&country.to_uppercase());
    }

    /// Returns `true` if no restrictions are configured (both lists are
    /// empty → all countries permitted).
    pub fn is_open(&self) -> bool {
        self.allowed.is_empty() && self.blocked.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_open() {
        let r = GeoRestriction::new();
        assert!(r.is_open());
        assert!(r.is_allowed("US"));
        assert!(r.is_allowed("CN"));
    }

    #[test]
    fn test_allow_list_restricts_others() {
        let mut r = GeoRestriction::new();
        r.allow("US");
        r.allow("CA");
        assert!(r.is_allowed("US"));
        assert!(r.is_allowed("CA"));
        assert!(!r.is_allowed("GB"));
    }

    #[test]
    fn test_block_overrides_allow() {
        let mut r = GeoRestriction::new();
        r.allow("US");
        r.block("US");
        assert!(!r.is_allowed("US"));
    }

    #[test]
    fn test_block_without_allow_list() {
        let mut r = GeoRestriction::new();
        r.block("KP");
        assert!(!r.is_allowed("KP"));
        assert!(r.is_allowed("US")); // not blocked
    }

    #[test]
    fn test_case_insensitive() {
        let mut r = GeoRestriction::new();
        r.allow("us");
        assert!(r.is_allowed("US"));
        assert!(r.is_allowed("us"));
    }

    #[test]
    fn test_remove_allow() {
        let mut r = GeoRestriction::new();
        r.allow("US");
        r.allow("GB");
        r.remove_allow("US");
        assert!(!r.is_allowed("US")); // removed from allow-list
        assert!(r.is_allowed("GB")); // still allowed
    }

    #[test]
    fn test_remove_block() {
        let mut r = GeoRestriction::new();
        r.block("CN");
        r.remove_block("CN");
        // After removing block and with no allow-list → permitted
        assert!(r.is_allowed("CN"));
    }
}
