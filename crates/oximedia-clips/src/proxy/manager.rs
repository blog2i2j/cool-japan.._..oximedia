//! Proxy management system.

use super::{ProxyLink, ProxyLinkId, ProxyQuality};
use crate::clip::ClipId;
use crate::error::{ClipError, ClipResult};
use std::collections::HashMap;

/// Manages proxy links for clips.
#[derive(Debug, Clone, Default)]
pub struct ProxyManager {
    /// Proxy links indexed by clip ID.
    links: HashMap<ClipId, Vec<ProxyLink>>,
}

impl ProxyManager {
    /// Creates a new proxy manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            links: HashMap::new(),
        }
    }

    /// Adds a proxy link.
    pub fn add_link(&mut self, link: ProxyLink) {
        self.links.entry(link.clip_id).or_default().push(link);
    }

    /// Removes a proxy link.
    ///
    /// # Errors
    ///
    /// Returns an error if the link is not found.
    pub fn remove_link(&mut self, clip_id: &ClipId, link_id: &ProxyLinkId) -> ClipResult<()> {
        if let Some(links) = self.links.get_mut(clip_id) {
            if let Some(pos) = links.iter().position(|l| &l.id == link_id) {
                links.remove(pos);
                return Ok(());
            }
        }
        Err(ClipError::ClipNotFound(clip_id.to_string()))
    }

    /// Gets all proxy links for a clip.
    #[must_use]
    pub fn get_links(&self, clip_id: &ClipId) -> Vec<&ProxyLink> {
        self.links
            .get(clip_id)
            .map_or_else(Vec::new, |links| links.iter().collect())
    }

    /// Gets a proxy link by quality.
    #[must_use]
    pub fn get_link_by_quality(
        &self,
        clip_id: &ClipId,
        quality: ProxyQuality,
    ) -> Option<&ProxyLink> {
        self.links
            .get(clip_id)
            .and_then(|links| links.iter().find(|l| l.quality == quality))
    }

    /// Gets the best available proxy (highest quality).
    #[must_use]
    pub fn get_best_proxy(&self, clip_id: &ClipId) -> Option<&ProxyLink> {
        let links = self.get_links(clip_id);
        if links.is_empty() {
            return None;
        }

        // Prioritize: High > Medium > Low > Custom
        for quality in &[
            ProxyQuality::High,
            ProxyQuality::Medium,
            ProxyQuality::Low,
            ProxyQuality::Custom,
        ] {
            if let Some(link) = links.iter().find(|l| l.quality == *quality) {
                return Some(link);
            }
        }

        None
    }

    /// Checks if a clip has any proxies.
    #[must_use]
    pub fn has_proxies(&self, clip_id: &ClipId) -> bool {
        self.links
            .get(clip_id)
            .is_some_and(|links| !links.is_empty())
    }

    /// Removes all proxies for a clip.
    pub fn clear_proxies(&mut self, clip_id: &ClipId) {
        self.links.remove(clip_id);
    }

    /// Returns the total number of proxy links.
    #[must_use]
    pub fn total_links(&self) -> usize {
        self.links.values().map(Vec::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_proxy_manager() {
        let mut manager = ProxyManager::new();
        let clip_id = ClipId::new();

        let link1 = ProxyLink::new(clip_id, PathBuf::from("/proxy/low.mov"), ProxyQuality::Low);
        let link2 = ProxyLink::new(
            clip_id,
            PathBuf::from("/proxy/high.mov"),
            ProxyQuality::High,
        );

        manager.add_link(link1);
        manager.add_link(link2);

        assert_eq!(manager.get_links(&clip_id).len(), 2);
        assert!(manager.has_proxies(&clip_id));

        let best = manager
            .get_best_proxy(&clip_id)
            .expect("get_best_proxy should succeed");
        assert_eq!(best.quality, ProxyQuality::High);
    }

    #[test]
    fn test_clear_proxies() {
        let mut manager = ProxyManager::new();
        let clip_id = ClipId::new();

        let link = ProxyLink::new(
            clip_id,
            PathBuf::from("/proxy/test.mov"),
            ProxyQuality::Medium,
        );
        manager.add_link(link);

        assert!(manager.has_proxies(&clip_id));
        manager.clear_proxies(&clip_id);
        assert!(!manager.has_proxies(&clip_id));
    }
}
