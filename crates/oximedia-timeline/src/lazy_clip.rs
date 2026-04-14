#![allow(dead_code)]
//! Lazy clip loading that defers media probing until first access.
//!
//! When importing large projects with hundreds of clips, probing every media
//! file immediately (to determine codec, resolution, duration, etc.) can be
//! very slow.  [`LazyClipLoader`] wraps clip metadata so that probing is
//! deferred until the clip is first placed on the timeline, previewed, or
//! otherwise accessed.
//!
//! # States
//!
//! A lazy clip transitions through these states:
//!
//! ```text
//! Unprobed ──► Probing ──► Probed
//!                 │
//!                 ▼
//!              Failed
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::clip::ClipId;

/// The probing state of a lazy clip.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProbeState {
    /// Media has not been probed yet.
    Unprobed,
    /// Probing is currently in progress.
    Probing,
    /// Probing completed successfully.
    Probed,
    /// Probing failed with an error message.
    Failed(String),
}

impl ProbeState {
    /// Returns `true` if the clip has been successfully probed.
    #[must_use]
    pub fn is_probed(&self) -> bool {
        matches!(self, Self::Probed)
    }

    /// Returns `true` if the clip has not yet been probed.
    #[must_use]
    pub fn is_unprobed(&self) -> bool {
        matches!(self, Self::Unprobed)
    }

    /// Returns `true` if probing failed.
    #[must_use]
    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed(_))
    }

    /// Returns `true` if probing is in progress.
    #[must_use]
    pub fn is_probing(&self) -> bool {
        matches!(self, Self::Probing)
    }
}

/// Media information obtained from probing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeResult {
    /// Video codec name (e.g. "VP9", "AV1").
    pub video_codec: Option<String>,
    /// Audio codec name (e.g. "Opus", "FLAC").
    pub audio_codec: Option<String>,
    /// Video width in pixels.
    pub width: Option<u32>,
    /// Video height in pixels.
    pub height: Option<u32>,
    /// Duration in frames.
    pub duration_frames: Option<i64>,
    /// Frame rate numerator.
    pub fps_num: Option<u32>,
    /// Frame rate denominator.
    pub fps_den: Option<u32>,
    /// Audio sample rate.
    pub sample_rate: Option<u32>,
    /// Number of audio channels.
    pub audio_channels: Option<u16>,
    /// File size in bytes.
    pub file_size: Option<u64>,
    /// Container format (e.g. "WebM", "MKV", "MP4").
    pub container_format: Option<String>,
}

impl ProbeResult {
    /// Creates a minimal probe result (used for non-file media sources).
    #[must_use]
    pub fn synthetic(width: u32, height: u32, duration_frames: i64) -> Self {
        Self {
            video_codec: None,
            audio_codec: None,
            width: Some(width),
            height: Some(height),
            duration_frames: Some(duration_frames),
            fps_num: None,
            fps_den: None,
            sample_rate: None,
            audio_channels: None,
            file_size: None,
            container_format: None,
        }
    }

    /// Returns the video resolution as `(width, height)` if known.
    #[must_use]
    pub fn resolution(&self) -> Option<(u32, u32)> {
        match (self.width, self.height) {
            (Some(w), Some(h)) => Some((w, h)),
            _ => None,
        }
    }

    /// Returns the frame rate as a float if known.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn fps(&self) -> Option<f64> {
        match (self.fps_num, self.fps_den) {
            (Some(n), Some(d)) if d > 0 => Some(n as f64 / d as f64),
            _ => None,
        }
    }

    /// Returns `true` if this media has a video stream.
    #[must_use]
    pub fn has_video(&self) -> bool {
        self.video_codec.is_some() || self.width.is_some()
    }

    /// Returns `true` if this media has an audio stream.
    #[must_use]
    pub fn has_audio(&self) -> bool {
        self.audio_codec.is_some() || self.sample_rate.is_some()
    }
}

impl Default for ProbeResult {
    fn default() -> Self {
        Self {
            video_codec: None,
            audio_codec: None,
            width: None,
            height: None,
            duration_frames: None,
            fps_num: None,
            fps_den: None,
            sample_rate: None,
            audio_channels: None,
            file_size: None,
            container_format: None,
        }
    }
}

/// A lazily-loaded clip entry in the clip loader.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LazyClipEntry {
    /// Clip ID in the timeline.
    pub clip_id: ClipId,
    /// Path to the media file.
    pub path: String,
    /// Current probing state.
    pub state: ProbeState,
    /// Probe result (populated after successful probing).
    pub probe_result: Option<ProbeResult>,
    /// Number of times this clip has been accessed (drives priority).
    access_count: u32,
}

impl LazyClipEntry {
    /// Creates a new lazy clip entry in the `Unprobed` state.
    #[must_use]
    pub fn new(clip_id: ClipId, path: impl Into<String>) -> Self {
        Self {
            clip_id,
            path: path.into(),
            state: ProbeState::Unprobed,
            probe_result: None,
            access_count: 0,
        }
    }

    /// Record an access to this clip.
    pub fn record_access(&mut self) {
        self.access_count = self.access_count.saturating_add(1);
    }

    /// Returns the access count.
    #[must_use]
    pub fn access_count(&self) -> u32 {
        self.access_count
    }

    /// Transition to the `Probing` state.
    ///
    /// Returns `false` if the clip is not in the `Unprobed` state.
    pub fn start_probing(&mut self) -> bool {
        if self.state.is_unprobed() {
            self.state = ProbeState::Probing;
            true
        } else {
            false
        }
    }

    /// Complete probing with a successful result.
    pub fn complete_probe(&mut self, result: ProbeResult) {
        self.state = ProbeState::Probed;
        self.probe_result = Some(result);
    }

    /// Mark probing as failed.
    pub fn fail_probe(&mut self, error: impl Into<String>) {
        self.state = ProbeState::Failed(error.into());
    }

    /// Reset to `Unprobed` state (e.g., to retry probing after failure).
    pub fn reset(&mut self) {
        self.state = ProbeState::Unprobed;
        self.probe_result = None;
    }

    /// Returns `true` if probing succeeded and data is available.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.state.is_probed() && self.probe_result.is_some()
    }
}

/// Manages lazy loading of clip media for a timeline project.
///
/// Clips are registered with their file paths but not probed until
/// explicitly accessed.  The loader prioritises clips based on access
/// count so that frequently-used clips are probed first.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LazyClipLoader {
    entries: HashMap<ClipId, LazyClipEntry>,
}

impl LazyClipLoader {
    /// Creates a new empty loader.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a clip for lazy loading.
    pub fn register(&mut self, clip_id: ClipId, path: impl Into<String>) {
        self.entries
            .insert(clip_id, LazyClipEntry::new(clip_id, path));
    }

    /// Remove a clip from the loader.
    pub fn unregister(&mut self, clip_id: ClipId) -> Option<LazyClipEntry> {
        self.entries.remove(&clip_id)
    }

    /// Look up a clip entry.
    #[must_use]
    pub fn get(&self, clip_id: ClipId) -> Option<&LazyClipEntry> {
        self.entries.get(&clip_id)
    }

    /// Look up a clip entry mutably.
    pub fn get_mut(&mut self, clip_id: ClipId) -> Option<&mut LazyClipEntry> {
        self.entries.get_mut(&clip_id)
    }

    /// Access a clip, recording the access and returning whether it needs probing.
    ///
    /// Returns `true` if the clip needs probing (was unprobed and has been
    /// transitioned to `Probing`).
    pub fn access(&mut self, clip_id: ClipId) -> bool {
        if let Some(entry) = self.entries.get_mut(&clip_id) {
            entry.record_access();
            if entry.state.is_unprobed() {
                entry.start_probing();
                return true;
            }
        }
        false
    }

    /// Complete probing for a clip.
    pub fn complete_probe(&mut self, clip_id: ClipId, result: ProbeResult) {
        if let Some(entry) = self.entries.get_mut(&clip_id) {
            entry.complete_probe(result);
        }
    }

    /// Fail probing for a clip.
    pub fn fail_probe(&mut self, clip_id: ClipId, error: impl Into<String>) {
        if let Some(entry) = self.entries.get_mut(&clip_id) {
            entry.fail_probe(error);
        }
    }

    /// Returns clips that need probing, sorted by access count (most-accessed first).
    #[must_use]
    pub fn pending_probes(&self) -> Vec<ClipId> {
        let mut pending: Vec<_> = self
            .entries
            .values()
            .filter(|e| e.state.is_unprobed())
            .collect();
        pending.sort_by(|a, b| b.access_count.cmp(&a.access_count));
        pending.iter().map(|e| e.clip_id).collect()
    }

    /// Returns the number of clips that have been probed.
    #[must_use]
    pub fn probed_count(&self) -> usize {
        self.entries.values().filter(|e| e.state.is_probed()).count()
    }

    /// Returns the number of clips that are still unprobed.
    #[must_use]
    pub fn unprobed_count(&self) -> usize {
        self.entries.values().filter(|e| e.state.is_unprobed()).count()
    }

    /// Returns the number of clips that failed probing.
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.entries.values().filter(|e| e.state.is_failed()).count()
    }

    /// Total number of registered clips.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no clips are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Probe all unprobed clips immediately with the given prober function.
    ///
    /// The `prober` closure receives the file path and returns a result.
    pub fn probe_all<F>(&mut self, prober: F)
    where
        F: Fn(&str) -> Result<ProbeResult, String>,
    {
        let ids: Vec<ClipId> = self
            .entries
            .values()
            .filter(|e| e.state.is_unprobed())
            .map(|e| e.clip_id)
            .collect();

        for id in ids {
            if let Some(entry) = self.entries.get_mut(&id) {
                let path = entry.path.clone();
                entry.start_probing();
                match prober(&path) {
                    Ok(result) => entry.complete_probe(result),
                    Err(err) => entry.fail_probe(err),
                }
            }
        }
    }

    /// Reset all failed probes to `Unprobed` so they can be retried.
    pub fn retry_failed(&mut self) {
        for entry in self.entries.values_mut() {
            if entry.state.is_failed() {
                entry.reset();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cid() -> ClipId {
        ClipId::new()
    }

    #[test]
    fn test_probe_state_transitions() {
        let state = ProbeState::Unprobed;
        assert!(state.is_unprobed());
        assert!(!state.is_probed());
        assert!(!state.is_failed());
        assert!(!state.is_probing());
    }

    #[test]
    fn test_lazy_clip_entry_new() {
        let id = cid();
        let entry = LazyClipEntry::new(id, "/path/to/video.mkv");
        assert_eq!(entry.clip_id, id);
        assert_eq!(entry.path, "/path/to/video.mkv");
        assert!(entry.state.is_unprobed());
        assert!(entry.probe_result.is_none());
        assert_eq!(entry.access_count(), 0);
    }

    #[test]
    fn test_lazy_clip_start_probing() {
        let mut entry = LazyClipEntry::new(cid(), "/tmp/clip.webm");
        assert!(entry.start_probing());
        assert!(entry.state.is_probing());
        // Cannot start probing again
        assert!(!entry.start_probing());
    }

    #[test]
    fn test_lazy_clip_complete_probe() {
        let mut entry = LazyClipEntry::new(cid(), "/tmp/clip.webm");
        entry.start_probing();
        entry.complete_probe(ProbeResult::synthetic(1920, 1080, 1000));
        assert!(entry.is_ready());
        let res = entry.probe_result.as_ref().expect("should have result");
        assert_eq!(res.resolution(), Some((1920, 1080)));
    }

    #[test]
    fn test_lazy_clip_fail_probe() {
        let mut entry = LazyClipEntry::new(cid(), "/tmp/missing.webm");
        entry.start_probing();
        entry.fail_probe("file not found");
        assert!(entry.state.is_failed());
        assert!(!entry.is_ready());
    }

    #[test]
    fn test_lazy_clip_reset() {
        let mut entry = LazyClipEntry::new(cid(), "/tmp/clip.webm");
        entry.start_probing();
        entry.fail_probe("error");
        entry.reset();
        assert!(entry.state.is_unprobed());
        assert!(entry.probe_result.is_none());
    }

    #[test]
    fn test_loader_register_and_get() {
        let mut loader = LazyClipLoader::new();
        let id = cid();
        loader.register(id, "/tmp/clip.webm");
        assert_eq!(loader.len(), 1);
        assert!(loader.get(id).is_some());
    }

    #[test]
    fn test_loader_access_triggers_probing() {
        let mut loader = LazyClipLoader::new();
        let id = cid();
        loader.register(id, "/tmp/clip.webm");
        let needs_probe = loader.access(id);
        assert!(needs_probe);
        // Second access should not trigger probing again
        let needs_probe2 = loader.access(id);
        assert!(!needs_probe2);
    }

    #[test]
    fn test_loader_complete_and_fail() {
        let mut loader = LazyClipLoader::new();
        let id1 = cid();
        let id2 = cid();
        loader.register(id1, "/tmp/good.webm");
        loader.register(id2, "/tmp/bad.webm");

        loader.access(id1);
        loader.access(id2);

        loader.complete_probe(id1, ProbeResult::synthetic(1920, 1080, 500));
        loader.fail_probe(id2, "corrupt file");

        assert_eq!(loader.probed_count(), 1);
        assert_eq!(loader.failed_count(), 1);
    }

    #[test]
    fn test_loader_pending_probes_by_access_count() {
        let mut loader = LazyClipLoader::new();
        let id1 = cid();
        let id2 = cid();
        loader.register(id1, "/tmp/a.webm");
        loader.register(id2, "/tmp/b.webm");

        // Access id2 more than id1
        if let Some(e) = loader.get_mut(id2) {
            e.record_access();
            e.record_access();
            e.record_access();
        }
        if let Some(e) = loader.get_mut(id1) {
            e.record_access();
        }

        let pending = loader.pending_probes();
        assert_eq!(pending.len(), 2);
        // id2 (3 accesses) should come first
        assert_eq!(pending[0], id2);
        assert_eq!(pending[1], id1);
    }

    #[test]
    fn test_loader_unregister() {
        let mut loader = LazyClipLoader::new();
        let id = cid();
        loader.register(id, "/tmp/clip.webm");
        let removed = loader.unregister(id);
        assert!(removed.is_some());
        assert!(loader.is_empty());
    }

    #[test]
    fn test_loader_probe_all() {
        let mut loader = LazyClipLoader::new();
        let id1 = cid();
        let id2 = cid();
        loader.register(id1, "/tmp/good.webm");
        loader.register(id2, "/tmp/bad.webm");

        loader.probe_all(|path| {
            if path.contains("good") {
                Ok(ProbeResult::synthetic(1280, 720, 300))
            } else {
                Err("file not found".to_string())
            }
        });

        assert_eq!(loader.probed_count(), 1);
        assert_eq!(loader.failed_count(), 1);
        assert_eq!(loader.unprobed_count(), 0);
    }

    #[test]
    fn test_loader_retry_failed() {
        let mut loader = LazyClipLoader::new();
        let id = cid();
        loader.register(id, "/tmp/clip.webm");
        loader.access(id);
        loader.fail_probe(id, "error");
        assert_eq!(loader.failed_count(), 1);

        loader.retry_failed();
        assert_eq!(loader.failed_count(), 0);
        assert_eq!(loader.unprobed_count(), 1);
    }

    #[test]
    fn test_probe_result_synthetic() {
        let pr = ProbeResult::synthetic(3840, 2160, 2000);
        assert_eq!(pr.resolution(), Some((3840, 2160)));
        assert_eq!(pr.duration_frames, Some(2000));
        assert!(pr.has_video());
        assert!(!pr.has_audio());
    }

    #[test]
    fn test_probe_result_fps() {
        let mut pr = ProbeResult::default();
        assert!(pr.fps().is_none());
        pr.fps_num = Some(24000);
        pr.fps_den = Some(1001);
        let fps = pr.fps().expect("should have fps");
        assert!((fps - 23.976_023_976_023_976).abs() < 0.001);
    }

    #[test]
    fn test_probe_result_has_audio() {
        let mut pr = ProbeResult::default();
        assert!(!pr.has_audio());
        pr.audio_codec = Some("Opus".to_string());
        assert!(pr.has_audio());
    }

    #[test]
    fn test_access_count_increments() {
        let mut entry = LazyClipEntry::new(cid(), "/tmp/clip.webm");
        assert_eq!(entry.access_count(), 0);
        entry.record_access();
        entry.record_access();
        assert_eq!(entry.access_count(), 2);
    }
}
