//! Proxy link registry for `OxiMedia` clips.
//!
//! Associates high-resolution clips with their lower-resolution proxy files,
//! enabling offline editing workflows and bandwidth-efficient preview.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::path::PathBuf;

/// Resolution tier of a proxy file.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProxyResolution {
    /// Quarter of the original resolution.
    Quarter,
    /// Half of the original resolution.
    Half,
    /// Full original resolution proxy (e.g. uncompressed → light codec).
    Full,
    /// Custom resolution (width × height).
    Custom(u32, u32),
}

impl ProxyResolution {
    /// Returns the approximate megapixels for common tiers,
    /// assuming a 3840×2160 source.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn megapixels(&self) -> f32 {
        const SRC_W: u32 = 3840;
        const SRC_H: u32 = 2160;
        match self {
            Self::Quarter => {
                let w = SRC_W / 4;
                let h = SRC_H / 4;
                (w * h) as f32 / 1_000_000.0
            }
            Self::Half => {
                let w = SRC_W / 2;
                let h = SRC_H / 2;
                (w * h) as f32 / 1_000_000.0
            }
            Self::Full => (SRC_W * SRC_H) as f32 / 1_000_000.0,
            Self::Custom(w, h) => (*w * *h) as f32 / 1_000_000.0,
        }
    }

    /// Returns a human-readable label.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Quarter => "1/4".to_string(),
            Self::Half => "1/2".to_string(),
            Self::Full => "Full".to_string(),
            Self::Custom(w, h) => format!("{w}x{h}"),
        }
    }
}

/// A link between a source clip and one of its proxy files.
#[derive(Debug, Clone)]
pub struct ProxyLink {
    /// Source clip identifier.
    pub clip_id: u64,
    /// Filesystem path to the proxy file.
    pub proxy_path: PathBuf,
    /// Resolution tier of the proxy.
    pub resolution: ProxyResolution,
    /// Whether the proxy file has been verified to exist and is valid.
    pub verified: bool,
}

impl ProxyLink {
    /// Create a new, unverified proxy link.
    #[must_use]
    pub fn new(clip_id: u64, proxy_path: PathBuf, resolution: ProxyResolution) -> Self {
        Self {
            clip_id,
            proxy_path,
            resolution,
            verified: false,
        }
    }

    /// Mark this link as verified.
    pub fn verify(&mut self) {
        self.verified = true;
    }

    /// Returns `true` if a verified proxy is available.
    #[must_use]
    pub fn has_proxy(&self) -> bool {
        self.verified
    }

    /// Returns `true` if this link is for a quarter-resolution proxy.
    #[must_use]
    pub fn is_offline_edit_suitable(&self) -> bool {
        matches!(
            self.resolution,
            ProxyResolution::Quarter | ProxyResolution::Half
        )
    }
}

/// Registry mapping clip IDs to their proxy links (one per resolution tier).
#[derive(Debug, Default)]
pub struct ProxyLinkRegistry {
    /// clip_id → list of proxy links
    links: HashMap<u64, Vec<ProxyLink>>,
}

impl ProxyLinkRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a proxy link for a clip. If a link with the same resolution
    /// already exists for this clip, it is replaced.
    pub fn link(&mut self, proxy: ProxyLink) {
        let entry = self.links.entry(proxy.clip_id).or_default();
        // Replace existing link for same resolution
        if let Some(existing) = entry.iter_mut().find(|l| l.resolution == proxy.resolution) {
            *existing = proxy;
        } else {
            entry.push(proxy);
        }
    }

    /// Remove the proxy link for `clip_id` at the given resolution.
    /// Returns `true` if a link was removed.
    pub fn unlink(&mut self, clip_id: u64, resolution: &ProxyResolution) -> bool {
        let Some(entry) = self.links.get_mut(&clip_id) else {
            return false;
        };
        let before = entry.len();
        entry.retain(|l| &l.resolution != resolution);
        entry.len() < before
    }

    /// Find the proxy link for `clip_id` at the requested resolution.
    #[must_use]
    pub fn find_proxy(&self, clip_id: u64, resolution: &ProxyResolution) -> Option<&ProxyLink> {
        self.links
            .get(&clip_id)?
            .iter()
            .find(|l| &l.resolution == resolution)
    }

    /// Return all proxy links registered for `clip_id`.
    #[must_use]
    pub fn all_proxies(&self, clip_id: u64) -> &[ProxyLink] {
        self.links.get(&clip_id).map_or(&[], Vec::as_slice)
    }

    /// Total number of proxy links registered.
    #[must_use]
    pub fn total_link_count(&self) -> usize {
        self.links.values().map(Vec::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn test_megapixels_quarter() {
        let mp = ProxyResolution::Quarter.megapixels();
        // 960 × 540 = 518400 → 0.5184
        assert!((mp - 0.5184_f32).abs() < 1e-3);
    }

    #[test]
    fn test_megapixels_half() {
        let mp = ProxyResolution::Half.megapixels();
        // 1920 × 1080 = 2_073_600 → 2.0736
        assert!((mp - 2.0736_f32).abs() < 1e-3);
    }

    #[test]
    fn test_megapixels_full() {
        let mp = ProxyResolution::Full.megapixels();
        assert!((mp - 8.2944_f32).abs() < 1e-3);
    }

    #[test]
    fn test_megapixels_custom() {
        let mp = ProxyResolution::Custom(1280, 720).megapixels();
        assert!((mp - 0.9216_f32).abs() < 1e-3);
    }

    #[test]
    fn test_proxy_resolution_label() {
        assert_eq!(ProxyResolution::Quarter.label(), "1/4");
        assert_eq!(ProxyResolution::Custom(1920, 1080).label(), "1920x1080");
    }

    #[test]
    fn test_proxy_link_not_verified_initially() {
        let link = ProxyLink::new(1, path("/tmp/proxy.mov"), ProxyResolution::Half);
        assert!(!link.has_proxy());
    }

    #[test]
    fn test_proxy_link_verified_after_verify() {
        let mut link = ProxyLink::new(1, path("/tmp/proxy.mov"), ProxyResolution::Half);
        link.verify();
        assert!(link.has_proxy());
    }

    #[test]
    fn test_proxy_link_offline_edit_suitable_quarter() {
        let link = ProxyLink::new(1, path("/tmp/q.mov"), ProxyResolution::Quarter);
        assert!(link.is_offline_edit_suitable());
    }

    #[test]
    fn test_proxy_link_full_not_offline_edit_suitable() {
        let link = ProxyLink::new(1, path("/tmp/f.mov"), ProxyResolution::Full);
        assert!(!link.is_offline_edit_suitable());
    }

    #[test]
    fn test_registry_link_and_find() {
        let mut reg = ProxyLinkRegistry::new();
        let proxy = ProxyLink::new(42, path("/tmp/42_half.mov"), ProxyResolution::Half);
        reg.link(proxy);
        assert!(reg.find_proxy(42, &ProxyResolution::Half).is_some());
    }

    #[test]
    fn test_registry_find_missing() {
        let reg = ProxyLinkRegistry::new();
        assert!(reg.find_proxy(99, &ProxyResolution::Quarter).is_none());
    }

    #[test]
    fn test_registry_link_replaces_same_resolution() {
        let mut reg = ProxyLinkRegistry::new();
        reg.link(ProxyLink::new(1, path("/old.mov"), ProxyResolution::Half));
        reg.link(ProxyLink::new(1, path("/new.mov"), ProxyResolution::Half));
        let found = reg
            .find_proxy(1, &ProxyResolution::Half)
            .expect("find_proxy should succeed");
        assert_eq!(found.proxy_path, path("/new.mov"));
    }

    #[test]
    fn test_registry_unlink() {
        let mut reg = ProxyLinkRegistry::new();
        reg.link(ProxyLink::new(5, path("/p.mov"), ProxyResolution::Quarter));
        assert!(reg.unlink(5, &ProxyResolution::Quarter));
        assert!(reg.find_proxy(5, &ProxyResolution::Quarter).is_none());
    }

    #[test]
    fn test_registry_unlink_missing_returns_false() {
        let mut reg = ProxyLinkRegistry::new();
        assert!(!reg.unlink(999, &ProxyResolution::Half));
    }

    #[test]
    fn test_registry_total_link_count() {
        let mut reg = ProxyLinkRegistry::new();
        reg.link(ProxyLink::new(1, path("/a.mov"), ProxyResolution::Quarter));
        reg.link(ProxyLink::new(1, path("/b.mov"), ProxyResolution::Half));
        reg.link(ProxyLink::new(2, path("/c.mov"), ProxyResolution::Full));
        assert_eq!(reg.total_link_count(), 3);
    }
}
