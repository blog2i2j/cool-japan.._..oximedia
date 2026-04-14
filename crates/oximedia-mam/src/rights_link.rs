//! Asset–rights linking for the MAM system.
//!
//! Provides [`AssetRightsLink`], which associates a media asset with a rights
//! record (managed externally, e.g. by `oximedia-rights`), and [`RightsLinker`],
//! which manages a collection of such links and can filter to those that are
//! currently active.

#![allow(dead_code)]

// ── AssetRightsLink ───────────────────────────────────────────────────────────

/// A link between a media asset and a rights record.
///
/// An optional `expiry_ts` (Unix timestamp in seconds) controls the validity
/// window.  When `None` the link is perpetual.
#[derive(Debug, Clone, PartialEq)]
pub struct AssetRightsLink {
    /// Media asset identifier.
    pub asset_id: u64,
    /// Rights record identifier (e.g. from `oximedia-rights`).
    pub rights_id: u64,
    /// Optional expiry timestamp (Unix seconds).  `None` = never expires.
    pub expiry_ts: Option<u64>,
}

impl AssetRightsLink {
    /// Create a new `AssetRightsLink` without an expiry.
    ///
    /// # Arguments
    ///
    /// * `asset_id`  — asset being licensed.
    /// * `rights_id` — rights record that covers the asset.
    #[must_use]
    pub fn new(asset_id: u64, rights_id: u64) -> Self {
        Self {
            asset_id,
            rights_id,
            expiry_ts: None,
        }
    }

    /// Set an expiry timestamp (builder-pattern).
    #[must_use]
    pub fn with_expiry(mut self, ts: u64) -> Self {
        self.expiry_ts = Some(ts);
        self
    }

    /// Return `true` if this link is currently active at the given Unix timestamp.
    ///
    /// A link without an expiry is always active.
    #[must_use]
    pub fn is_active_at(&self, now_ts: u64) -> bool {
        match self.expiry_ts {
            None => true,
            Some(exp) => now_ts < exp,
        }
    }
}

// ── RightsLinker ──────────────────────────────────────────────────────────────

/// Manages asset-to-rights associations and queries active links.
///
/// # Example
///
/// ```rust
/// use oximedia_mam::rights_link::{AssetRightsLink, RightsLinker};
///
/// let mut linker = RightsLinker::new();
/// linker.link(AssetRightsLink::new(1, 100).with_expiry(9999999999));
/// linker.link(AssetRightsLink::new(2, 200)); // perpetual
///
/// let active = linker.active_rights(1_000_000_000);
/// assert_eq!(active.len(), 2);
/// ```
#[derive(Debug, Default)]
pub struct RightsLinker {
    links: Vec<AssetRightsLink>,
}

impl RightsLinker {
    /// Create an empty `RightsLinker`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an asset-rights link.
    pub fn link(&mut self, link: AssetRightsLink) {
        self.links.push(link);
    }

    /// Return all links that are active at `now_ts` (Unix seconds).
    #[must_use]
    pub fn active_rights(&self, now_ts: u64) -> Vec<AssetRightsLink> {
        self.links
            .iter()
            .filter(|l| l.is_active_at(now_ts))
            .cloned()
            .collect()
    }

    /// Return all links for a specific asset, active or not.
    #[must_use]
    pub fn links_for_asset(&self, asset_id: u64) -> Vec<&AssetRightsLink> {
        self.links.iter().filter(|l| l.asset_id == asset_id).collect()
    }

    /// Return all links that are active for a specific asset at `now_ts`.
    #[must_use]
    pub fn active_rights_for_asset(&self, asset_id: u64, now_ts: u64) -> Vec<AssetRightsLink> {
        self.links
            .iter()
            .filter(|l| l.asset_id == asset_id && l.is_active_at(now_ts))
            .cloned()
            .collect()
    }

    /// Remove all links for a specific asset.
    pub fn unlink_asset(&mut self, asset_id: u64) {
        self.links.retain(|l| l.asset_id != asset_id);
    }

    /// Remove all links for a specific rights record.
    pub fn unlink_rights(&mut self, rights_id: u64) {
        self.links.retain(|l| l.rights_id != rights_id);
    }

    /// Return the total number of registered links.
    #[must_use]
    pub fn len(&self) -> usize {
        self.links.len()
    }

    /// Return `true` if there are no registered links.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_link_new_no_expiry() {
        let l = AssetRightsLink::new(1, 100);
        assert_eq!(l.asset_id, 1);
        assert_eq!(l.rights_id, 100);
        assert!(l.expiry_ts.is_none());
    }

    #[test]
    fn test_link_with_expiry() {
        let l = AssetRightsLink::new(1, 100).with_expiry(2_000_000_000);
        assert_eq!(l.expiry_ts, Some(2_000_000_000));
    }

    #[test]
    fn test_is_active_perpetual() {
        let l = AssetRightsLink::new(1, 100);
        assert!(l.is_active_at(0));
        assert!(l.is_active_at(u64::MAX));
    }

    #[test]
    fn test_is_active_before_expiry() {
        let l = AssetRightsLink::new(1, 100).with_expiry(1_000);
        assert!(l.is_active_at(999));
    }

    #[test]
    fn test_is_inactive_at_expiry() {
        let l = AssetRightsLink::new(1, 100).with_expiry(1_000);
        assert!(!l.is_active_at(1_000)); // exactly at expiry = expired
        assert!(!l.is_active_at(2_000));
    }

    #[test]
    fn test_rights_linker_empty() {
        let linker = RightsLinker::new();
        assert!(linker.is_empty());
        assert_eq!(linker.len(), 0);
    }

    #[test]
    fn test_rights_linker_link_and_len() {
        let mut linker = RightsLinker::new();
        linker.link(AssetRightsLink::new(1, 100));
        linker.link(AssetRightsLink::new(2, 200));
        assert_eq!(linker.len(), 2);
        assert!(!linker.is_empty());
    }

    #[test]
    fn test_active_rights_excludes_expired() {
        let mut linker = RightsLinker::new();
        linker.link(AssetRightsLink::new(1, 100).with_expiry(500));
        linker.link(AssetRightsLink::new(2, 200)); // perpetual
        let active = linker.active_rights(1_000);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].asset_id, 2);
    }

    #[test]
    fn test_active_rights_includes_perpetual() {
        let mut linker = RightsLinker::new();
        linker.link(AssetRightsLink::new(1, 100));
        let active = linker.active_rights(9_999_999_999);
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn test_links_for_asset() {
        let mut linker = RightsLinker::new();
        linker.link(AssetRightsLink::new(1, 100));
        linker.link(AssetRightsLink::new(1, 200));
        linker.link(AssetRightsLink::new(2, 300));
        let asset1_links = linker.links_for_asset(1);
        assert_eq!(asset1_links.len(), 2);
    }

    #[test]
    fn test_active_rights_for_asset() {
        let mut linker = RightsLinker::new();
        linker.link(AssetRightsLink::new(1, 100).with_expiry(500));
        linker.link(AssetRightsLink::new(1, 200)); // perpetual
        let active = linker.active_rights_for_asset(1, 1_000);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].rights_id, 200);
    }

    #[test]
    fn test_unlink_asset() {
        let mut linker = RightsLinker::new();
        linker.link(AssetRightsLink::new(1, 100));
        linker.link(AssetRightsLink::new(2, 200));
        linker.unlink_asset(1);
        assert_eq!(linker.len(), 1);
        assert_eq!(linker.links[0].asset_id, 2);
    }

    #[test]
    fn test_unlink_rights() {
        let mut linker = RightsLinker::new();
        linker.link(AssetRightsLink::new(1, 100));
        linker.link(AssetRightsLink::new(2, 100));
        linker.link(AssetRightsLink::new(3, 200));
        linker.unlink_rights(100);
        assert_eq!(linker.len(), 1);
        assert_eq!(linker.links[0].rights_id, 200);
    }
}
