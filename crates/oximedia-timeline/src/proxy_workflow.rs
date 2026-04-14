#![allow(dead_code)]
//! Proxy workflow: offline/online editing with proxy media.
//!
//! The proxy workflow allows editors to work with lightweight proxy clips
//! (low-resolution, compressed copies of source media) while editing, then
//! automatically conform (relink) the project to full-resolution source media
//! before final export.
//!
//! # Workflow
//!
//! 1. Generate proxy clips for large source media files.
//! 2. Edit the timeline using proxy clips (fast playback, low memory).
//! 3. Call [`ProxyManager::conform`] to relink all proxy clips back to the
//!    original high-resolution sources for final render.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::clip::ClipId;

/// Resolution tier of a media file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResolutionTier {
    /// Full-resolution original media (4K+).
    Full,
    /// Half-resolution proxy (1080p-ish).
    Half,
    /// Quarter-resolution proxy (540p-ish).
    Quarter,
    /// Thumbnail or scrubbing proxy (very low resolution).
    Thumbnail,
}

/// Codec used for the proxy transcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProxyCodec {
    /// ProRes 422 Proxy — common broadcast proxy format.
    ProRes422Proxy,
    /// H.264 with fast decode settings.
    H264Fast,
    /// VP9 with fast decode settings.
    Vp9Fast,
    /// DNxHD / DNxHR — Avid proxy format.
    DnxHd,
}

/// A proxy media entry that maps a clip to its proxy and original paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyEntry {
    /// ID of the clip in the timeline.
    pub clip_id: ClipId,
    /// Path to the original (full-res) source media.
    pub source_path: String,
    /// Path to the generated proxy file.
    pub proxy_path: String,
    /// Resolution tier of the proxy.
    pub tier: ResolutionTier,
    /// Codec used for the proxy transcode.
    pub codec: ProxyCodec,
    /// Whether the proxy has been generated and is available.
    pub proxy_available: bool,
    /// Whether this clip is currently using the proxy or the original.
    pub using_proxy: bool,
}

impl ProxyEntry {
    /// Create a new proxy entry (proxy not yet generated).
    #[must_use]
    pub fn new(clip_id: ClipId, source_path: impl Into<String>, proxy_path: impl Into<String>) -> Self {
        Self {
            clip_id,
            source_path: source_path.into(),
            proxy_path: proxy_path.into(),
            tier: ResolutionTier::Half,
            proxy_available: false,
            using_proxy: false,
            codec: ProxyCodec::ProRes422Proxy,
        }
    }

    /// Mark the proxy as available and switch the clip to use it.
    pub fn mark_available(&mut self) {
        self.proxy_available = true;
        self.using_proxy = true;
    }

    /// Return the active media path based on `using_proxy`.
    #[must_use]
    pub fn active_path(&self) -> &str {
        if self.using_proxy && self.proxy_available {
            &self.proxy_path
        } else {
            &self.source_path
        }
    }
}

/// Error type for proxy workflow operations.
#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    /// Clip not registered with the proxy manager.
    #[error("Clip {0} not registered in proxy manager")]
    ClipNotRegistered(ClipId),
    /// Proxy not yet generated for the clip.
    #[error("Proxy not yet generated for clip {0}")]
    ProxyNotAvailable(ClipId),
    /// Source media not found at the expected path.
    #[error("Source media not found: {0}")]
    SourceNotFound(String),
}

/// Result type for proxy workflow operations.
pub type ProxyResult<T> = Result<T, ProxyError>;

/// Manages the proxy workflow for a timeline project.
///
/// Tracks which clips have proxies, whether proxies are active, and provides
/// methods to switch between proxy and online (full-res) modes.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ProxyManager {
    entries: HashMap<ClipId, ProxyEntry>,
}

impl ProxyManager {
    /// Create an empty proxy manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a clip for proxy workflow.
    pub fn register(&mut self, entry: ProxyEntry) {
        self.entries.insert(entry.clip_id, entry);
    }

    /// Look up the proxy entry for a clip.
    #[must_use]
    pub fn get(&self, clip_id: ClipId) -> Option<&ProxyEntry> {
        self.entries.get(&clip_id)
    }

    /// Mutable access to a clip's proxy entry.
    pub fn get_mut(&mut self, clip_id: ClipId) -> Option<&mut ProxyEntry> {
        self.entries.get_mut(&clip_id)
    }

    /// Mark a proxy as generated and available for the given clip.
    ///
    /// # Errors
    ///
    /// Returns [`ProxyError::ClipNotRegistered`] if the clip was not registered.
    pub fn mark_proxy_available(&mut self, clip_id: ClipId) -> ProxyResult<()> {
        let entry = self
            .entries
            .get_mut(&clip_id)
            .ok_or(ProxyError::ClipNotRegistered(clip_id))?;
        entry.mark_available();
        Ok(())
    }

    /// Switch a clip to use its proxy media.
    ///
    /// # Errors
    ///
    /// Returns an error if the clip is not registered or the proxy is not available.
    pub fn activate_proxy(&mut self, clip_id: ClipId) -> ProxyResult<()> {
        let entry = self
            .entries
            .get_mut(&clip_id)
            .ok_or(ProxyError::ClipNotRegistered(clip_id))?;
        if !entry.proxy_available {
            return Err(ProxyError::ProxyNotAvailable(clip_id));
        }
        entry.using_proxy = true;
        Ok(())
    }

    /// Switch a clip to use its full-resolution source media.
    ///
    /// # Errors
    ///
    /// Returns [`ProxyError::ClipNotRegistered`] if the clip was not registered.
    pub fn deactivate_proxy(&mut self, clip_id: ClipId) -> ProxyResult<()> {
        let entry = self
            .entries
            .get_mut(&clip_id)
            .ok_or(ProxyError::ClipNotRegistered(clip_id))?;
        entry.using_proxy = false;
        Ok(())
    }

    /// **Conform** the entire project: switch all registered clips back to full
    /// resolution source media for final render.  Clips whose proxies are not
    /// available are silently skipped (they are already using source media).
    pub fn conform_all(&mut self) {
        for entry in self.entries.values_mut() {
            entry.using_proxy = false;
        }
    }

    /// Switch all registered clips to proxy mode (offline edit mode).
    ///
    /// # Errors
    ///
    /// Returns a list of `(ClipId, ProxyError)` for clips whose proxies are
    /// not yet available.
    pub fn activate_all_proxies(&mut self) -> Vec<(ClipId, ProxyError)> {
        let mut errors = Vec::new();
        for (id, entry) in self.entries.iter_mut() {
            if entry.proxy_available {
                entry.using_proxy = true;
            } else {
                errors.push((*id, ProxyError::ProxyNotAvailable(*id)));
            }
        }
        errors
    }

    /// Number of clips registered with the proxy manager.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no clips are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Count of clips that currently have proxies available.
    #[must_use]
    pub fn available_count(&self) -> usize {
        self.entries.values().filter(|e| e.proxy_available).count()
    }

    /// Count of clips currently using proxy media.
    #[must_use]
    pub fn active_proxy_count(&self) -> usize {
        self.entries
            .values()
            .filter(|e| e.using_proxy && e.proxy_available)
            .count()
    }

    /// Return the active media path for a clip (proxy if available/active,
    /// otherwise source).
    ///
    /// # Errors
    ///
    /// Returns [`ProxyError::ClipNotRegistered`] if the clip has not been
    /// registered.
    pub fn active_path(&self, clip_id: ClipId) -> ProxyResult<&str> {
        let entry = self
            .entries
            .get(&clip_id)
            .ok_or(ProxyError::ClipNotRegistered(clip_id))?;
        Ok(entry.active_path())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::ClipId;

    fn new_clip_id() -> ClipId {
        ClipId::new()
    }

    #[test]
    fn test_register_and_get() {
        let mut mgr = ProxyManager::new();
        let id = new_clip_id();
        let entry = ProxyEntry::new(id, "/full/res.mov", "/proxy/res_proxy.mov");
        mgr.register(entry);
        assert!(mgr.get(id).is_some());
    }

    #[test]
    fn test_active_path_uses_source_initially() {
        let mut mgr = ProxyManager::new();
        let id = new_clip_id();
        mgr.register(ProxyEntry::new(id, "/full/res.mov", "/proxy/res.mov"));
        assert_eq!(mgr.active_path(id).unwrap(), "/full/res.mov");
    }

    #[test]
    fn test_mark_available_switches_to_proxy() {
        let mut mgr = ProxyManager::new();
        let id = new_clip_id();
        mgr.register(ProxyEntry::new(id, "/full/res.mov", "/proxy/res.mov"));
        mgr.mark_proxy_available(id).unwrap();
        assert_eq!(mgr.active_path(id).unwrap(), "/proxy/res.mov");
    }

    #[test]
    fn test_conform_all_switches_to_source() {
        let mut mgr = ProxyManager::new();
        let id = new_clip_id();
        let mut entry = ProxyEntry::new(id, "/full/res.mov", "/proxy/res.mov");
        entry.mark_available();
        mgr.register(entry);

        assert_eq!(mgr.active_proxy_count(), 1);
        mgr.conform_all();
        assert_eq!(mgr.active_proxy_count(), 0);
        assert_eq!(mgr.active_path(id).unwrap(), "/full/res.mov");
    }

    #[test]
    fn test_activate_proxy_unavailable_returns_error() {
        let mut mgr = ProxyManager::new();
        let id = new_clip_id();
        mgr.register(ProxyEntry::new(id, "/full/res.mov", "/proxy/res.mov"));
        assert!(mgr.activate_proxy(id).is_err());
    }

    #[test]
    fn test_activate_all_proxies_partial() {
        let mut mgr = ProxyManager::new();
        let id1 = new_clip_id();
        let id2 = new_clip_id();

        let mut e1 = ProxyEntry::new(id1, "/a.mov", "/a_proxy.mov");
        e1.mark_available();
        mgr.register(e1);
        mgr.register(ProxyEntry::new(id2, "/b.mov", "/b_proxy.mov")); // no proxy yet

        let errors = mgr.activate_all_proxies();
        assert_eq!(errors.len(), 1, "one clip had no proxy");
        assert_eq!(mgr.active_proxy_count(), 1);
    }

    #[test]
    fn test_deactivate_proxy() {
        let mut mgr = ProxyManager::new();
        let id = new_clip_id();
        let mut entry = ProxyEntry::new(id, "/full.mov", "/proxy.mov");
        entry.mark_available();
        mgr.register(entry);

        mgr.deactivate_proxy(id).unwrap();
        assert_eq!(mgr.active_path(id).unwrap(), "/full.mov");
    }

    #[test]
    fn test_available_count() {
        let mut mgr = ProxyManager::new();
        let id1 = new_clip_id();
        let id2 = new_clip_id();
        let mut e = ProxyEntry::new(id1, "/a.mov", "/a_p.mov");
        e.mark_available();
        mgr.register(e);
        mgr.register(ProxyEntry::new(id2, "/b.mov", "/b_p.mov"));
        assert_eq!(mgr.available_count(), 1);
    }
}
