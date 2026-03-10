//! Shareable review link management.
//!
//! Provides a `ReviewLink` type for distributing view-only (or download-enabled)
//! links to assets under review, together with a `ReviewLinkRegistry` for
//! lifecycle management.

#![allow(dead_code)]

/// A shareable link that grants time-limited access to an asset for review.
#[derive(Debug, Clone)]
pub struct ReviewLink {
    /// Unique identifier for the link (e.g. a short token or UUID string).
    pub id: String,
    /// Identifier of the asset being reviewed.
    pub asset_id: String,
    /// User that created this link.
    pub creator_id: u64,
    /// Unix timestamp (seconds) when the link was created.
    pub created_at: u64,
    /// Optional Unix timestamp (seconds) after which the link is invalid.
    pub expires_at: Option<u64>,
    /// Optional password required to access the link.
    pub password: Option<String>,
    /// Whether the recipient may download the asset.
    pub allow_download: bool,
    /// Number of times the link has been viewed.
    pub view_count: u64,
}

impl ReviewLink {
    /// Create a new review link with no expiry and no password.
    #[must_use]
    pub fn new(id: &str, asset_id: &str, creator_id: u64, created_at: u64) -> Self {
        Self {
            id: id.to_owned(),
            asset_id: asset_id.to_owned(),
            creator_id,
            created_at,
            expires_at: None,
            password: None,
            allow_download: false,
            view_count: 0,
        }
    }

    /// Set an expiry timestamp (builder pattern).
    #[must_use]
    pub fn with_expiry(mut self, expires_at: u64) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Protect the link with a password (builder pattern).
    #[must_use]
    pub fn with_password(mut self, pwd: &str) -> Self {
        self.password = Some(pwd.to_owned());
        self
    }

    /// Allow the recipient to download the asset (builder pattern).
    #[must_use]
    pub fn with_download(mut self, allow: bool) -> Self {
        self.allow_download = allow;
        self
    }

    /// Return `true` if the link has passed its expiry time.
    ///
    /// Links with no expiry never expire.
    #[must_use]
    pub fn is_expired(&self, now: u64) -> bool {
        self.expires_at.map_or(false, |exp| now > exp)
    }

    /// Increment the view counter by one.
    pub fn record_view(&mut self) {
        self.view_count += 1;
    }
}

/// A registry that manages the lifecycle of all review links.
#[derive(Debug, Default)]
pub struct ReviewLinkRegistry {
    /// All links, both active and expired.
    pub links: Vec<ReviewLink>,
}

impl ReviewLinkRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a new review link.
    pub fn create(&mut self, link: ReviewLink) {
        self.links.push(link);
    }

    /// Look up a link by its id.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&ReviewLink> {
        self.links.iter().find(|l| l.id == id)
    }

    /// Look up a link by its id (mutable).
    #[must_use]
    pub fn get_mut(&mut self, id: &str) -> Option<&mut ReviewLink> {
        self.links.iter_mut().find(|l| l.id == id)
    }

    /// Return all links that have not yet expired at `now`.
    #[must_use]
    pub fn active_links(&self, now: u64) -> Vec<&ReviewLink> {
        self.links.iter().filter(|l| !l.is_expired(now)).collect()
    }

    /// Remove all expired links and return the count of removed links.
    pub fn remove_expired(&mut self, now: u64) -> usize {
        let before = self.links.len();
        self.links.retain(|l| !l.is_expired(now));
        before - self.links.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_link(id: &str, created_at: u64) -> ReviewLink {
        ReviewLink::new(id, "asset-1", 42, created_at)
    }

    #[test]
    fn test_new_link_defaults() {
        let link = make_link("abc", 1000);
        assert_eq!(link.id, "abc");
        assert_eq!(link.asset_id, "asset-1");
        assert_eq!(link.creator_id, 42);
        assert!(link.expires_at.is_none());
        assert!(link.password.is_none());
        assert!(!link.allow_download);
        assert_eq!(link.view_count, 0);
    }

    #[test]
    fn test_with_expiry() {
        let link = make_link("abc", 1000).with_expiry(2000);
        assert_eq!(link.expires_at, Some(2000));
    }

    #[test]
    fn test_with_password() {
        let link = make_link("abc", 1000).with_password("secret");
        assert_eq!(link.password.as_deref(), Some("secret"));
    }

    #[test]
    fn test_with_download() {
        let link = make_link("abc", 1000).with_download(true);
        assert!(link.allow_download);
    }

    #[test]
    fn test_is_expired_no_expiry() {
        let link = make_link("abc", 1000);
        assert!(!link.is_expired(999_999));
    }

    #[test]
    fn test_is_expired_before_expiry() {
        let link = make_link("abc", 1000).with_expiry(5000);
        assert!(!link.is_expired(4999));
    }

    #[test]
    fn test_is_expired_after_expiry() {
        let link = make_link("abc", 1000).with_expiry(5000);
        assert!(link.is_expired(5001));
    }

    #[test]
    fn test_is_expired_at_exact_boundary() {
        // Expiry is exclusive: expired only when now > expires_at.
        let link = make_link("abc", 1000).with_expiry(5000);
        assert!(!link.is_expired(5000));
    }

    #[test]
    fn test_record_view() {
        let mut link = make_link("abc", 0);
        link.record_view();
        link.record_view();
        assert_eq!(link.view_count, 2);
    }

    #[test]
    fn test_registry_create_and_get() {
        let mut reg = ReviewLinkRegistry::new();
        reg.create(make_link("tok1", 0));
        assert!(reg.get("tok1").is_some());
        assert!(reg.get("nope").is_none());
    }

    #[test]
    fn test_registry_get_mut_record_view() {
        let mut reg = ReviewLinkRegistry::new();
        reg.create(make_link("tok1", 0));
        reg.get_mut("tok1")
            .expect("collab test operation should succeed")
            .record_view();
        assert_eq!(
            reg.get("tok1")
                .expect("collab test operation should succeed")
                .view_count,
            1
        );
    }

    #[test]
    fn test_registry_active_links() {
        let mut reg = ReviewLinkRegistry::new();
        reg.create(make_link("active", 0));
        reg.create(make_link("expired", 0).with_expiry(100));
        let active = reg.active_links(200);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "active");
    }

    #[test]
    fn test_registry_remove_expired_returns_count() {
        let mut reg = ReviewLinkRegistry::new();
        reg.create(make_link("a", 0).with_expiry(100));
        reg.create(make_link("b", 0).with_expiry(200));
        reg.create(make_link("c", 0)); // no expiry
        let removed = reg.remove_expired(150);
        assert_eq!(removed, 1);
        assert_eq!(reg.links.len(), 2);
    }

    #[test]
    fn test_registry_default() {
        let reg = ReviewLinkRegistry::default();
        assert!(reg.links.is_empty());
    }
}
