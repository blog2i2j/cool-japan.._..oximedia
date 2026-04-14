#![allow(dead_code)]
//! Incremental rendering that only re-composites tracks with changed clips.
//!
//! In a multi-track timeline, re-rendering every track on every frame is
//! wasteful when only a subset of tracks have actually changed.  The
//! [`ChangeTracker`] records per-track and per-clip modifications, and
//! the [`IncrementalRenderer`] uses this information to skip unchanged
//! tracks during compositing, falling back to their cached results.
//!
//! # Architecture
//!
//! ```text
//! Timeline edit ──► ChangeTracker.mark_*(...)
//!                        │
//!                        ▼
//! IncrementalRenderer.render(frame)
//!    for each track:
//!       if tracker.is_dirty(track, frame) → re-render track layer
//!       else                              → reuse cached layer
//!    composite all layers → final frame
//! ```

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::clip::ClipId;
use crate::track::TrackId;

/// A version counter that monotonically increases on each edit.
pub type Version = u64;

/// Per-track dirty state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackDirtyState {
    /// Current version of this track (incremented on every edit).
    pub version: Version,
    /// Frame range that is dirty (inclusive start, exclusive end).
    /// `None` means the entire track is dirty.
    pub dirty_range: Option<(i64, i64)>,
    /// Set of clip IDs that were modified since last render.
    pub dirty_clips: HashSet<ClipId>,
}

impl TrackDirtyState {
    /// Creates a new dirty state (everything dirty).
    #[must_use]
    fn new() -> Self {
        Self {
            version: 1,
            dirty_range: None, // None = entire track dirty
            dirty_clips: HashSet::new(),
        }
    }

    /// Mark a specific frame range as dirty.
    fn mark_range(&mut self, start: i64, end: i64) {
        self.version += 1;
        match self.dirty_range {
            Some((existing_start, existing_end)) => {
                if existing_start == existing_end {
                    // Was clean (empty range) — set the new range directly
                    self.dirty_range = Some((start, end));
                } else {
                    // Expand the dirty range to include both
                    self.dirty_range =
                        Some((existing_start.min(start), existing_end.max(end)));
                }
            }
            None => {
                // Track was fully dirty; stays fully dirty.
            }
        }
    }

    /// Mark a specific clip as modified.
    fn mark_clip(&mut self, clip_id: ClipId) {
        self.version += 1;
        self.dirty_clips.insert(clip_id);
    }

    /// Mark the entire track as dirty.
    fn mark_all_dirty(&mut self) {
        self.version += 1;
        self.dirty_range = None;
        self.dirty_clips.clear();
    }

    /// Clear all dirty flags (after rendering).
    fn clean(&mut self) {
        self.dirty_range = Some((0, 0)); // empty range = clean
        self.dirty_clips.clear();
    }

    /// Returns `true` if the given frame falls within the dirty range.
    #[must_use]
    fn is_frame_dirty(&self, frame: i64) -> bool {
        match self.dirty_range {
            None => true, // entire track dirty
            Some((start, end)) => frame >= start && frame < end,
        }
    }
}

/// Tracks modifications across the timeline for incremental rendering.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChangeTracker {
    /// Per-track dirty state.
    tracks: HashMap<TrackId, TrackDirtyState>,
    /// Global version counter (incremented on any change).
    global_version: Version,
    /// The version at which the last full render was performed.
    last_rendered_version: Version,
}

impl ChangeTracker {
    /// Creates a new change tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current global version.
    #[must_use]
    pub fn global_version(&self) -> Version {
        self.global_version
    }

    /// Returns the version at which the last render was performed.
    #[must_use]
    pub fn last_rendered_version(&self) -> Version {
        self.last_rendered_version
    }

    /// Returns `true` if any change has occurred since the last render.
    #[must_use]
    pub fn has_changes(&self) -> bool {
        self.global_version > self.last_rendered_version
    }

    /// Register a track for change tracking.
    pub fn register_track(&mut self, track_id: TrackId) {
        self.tracks.entry(track_id).or_insert_with(TrackDirtyState::new);
    }

    /// Remove a track from change tracking.
    pub fn unregister_track(&mut self, track_id: TrackId) {
        self.tracks.remove(&track_id);
        self.global_version += 1;
    }

    /// Mark a clip as modified on a specific track.
    pub fn mark_clip_changed(&mut self, track_id: TrackId, clip_id: ClipId) {
        self.global_version += 1;
        let state = self.tracks.entry(track_id).or_insert_with(TrackDirtyState::new);
        state.mark_clip(clip_id);
    }

    /// Mark a specific frame range as dirty on a track.
    pub fn mark_range_dirty(&mut self, track_id: TrackId, start: i64, end: i64) {
        self.global_version += 1;
        let state = self.tracks.entry(track_id).or_insert_with(TrackDirtyState::new);
        state.mark_range(start, end);
    }

    /// Mark an entire track as dirty (e.g., after adding/removing a clip).
    pub fn mark_track_dirty(&mut self, track_id: TrackId) {
        self.global_version += 1;
        let state = self.tracks.entry(track_id).or_insert_with(TrackDirtyState::new);
        state.mark_all_dirty();
    }

    /// Mark all tracks as dirty (e.g., after loading a new timeline).
    pub fn mark_all_dirty(&mut self) {
        self.global_version += 1;
        for state in self.tracks.values_mut() {
            state.mark_all_dirty();
        }
    }

    /// Returns `true` if a specific track needs re-rendering at the given frame.
    #[must_use]
    pub fn is_track_dirty(&self, track_id: TrackId, frame: i64) -> bool {
        self.tracks
            .get(&track_id)
            .map_or(false, |state| state.is_frame_dirty(frame))
    }

    /// Returns `true` if a specific track has any dirty state.
    #[must_use]
    pub fn is_track_dirty_any(&self, track_id: TrackId) -> bool {
        self.tracks
            .get(&track_id)
            .map_or(false, |state| {
                state.dirty_range.is_none() || !state.dirty_clips.is_empty()
            })
    }

    /// Returns the set of dirty track IDs at the given frame.
    #[must_use]
    pub fn dirty_tracks_at(&self, frame: i64) -> Vec<TrackId> {
        self.tracks
            .iter()
            .filter(|(_, state)| state.is_frame_dirty(frame))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Returns the set of clip IDs that are dirty on a given track.
    #[must_use]
    pub fn dirty_clips(&self, track_id: TrackId) -> HashSet<ClipId> {
        self.tracks
            .get(&track_id)
            .map_or_else(HashSet::new, |state| state.dirty_clips.clone())
    }

    /// Returns the version of a specific track.
    #[must_use]
    pub fn track_version(&self, track_id: TrackId) -> Version {
        self.tracks.get(&track_id).map_or(0, |state| state.version)
    }

    /// Mark all tracks as clean (called after a full render pass).
    pub fn mark_rendered(&mut self) {
        self.last_rendered_version = self.global_version;
        for state in self.tracks.values_mut() {
            state.clean();
        }
    }

    /// Mark a single track as clean (called after rendering that track).
    pub fn mark_track_rendered(&mut self, track_id: TrackId) {
        if let Some(state) = self.tracks.get_mut(&track_id) {
            state.clean();
        }
    }

    /// Number of registered tracks.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Returns `true` if no tracks are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }
}

/// Cached layer data for a rendered track.
#[derive(Debug, Clone)]
pub struct CachedLayer {
    /// Track that produced this layer.
    pub track_id: TrackId,
    /// Version of the track when this layer was rendered.
    pub version: Version,
    /// Frame number this layer was rendered for.
    pub frame: i64,
    /// Pixel data (simplified as a Vec<u8> for RGBA).
    pub data: Vec<u8>,
    /// Width of the rendered layer.
    pub width: u32,
    /// Height of the rendered layer.
    pub height: u32,
}

/// Incremental renderer that uses [`ChangeTracker`] to skip unchanged tracks.
#[derive(Debug, Default)]
pub struct IncrementalRenderer {
    /// Change tracker.
    tracker: ChangeTracker,
    /// Cache of rendered layers indexed by (TrackId, frame).
    layer_cache: HashMap<(TrackId, i64), CachedLayer>,
    /// Maximum number of cached layers (LRU eviction).
    max_cache_entries: usize,
    /// Total renders performed.
    total_renders: u64,
    /// Renders skipped (cache hit).
    cache_hits: u64,
}

impl IncrementalRenderer {
    /// Creates a new incremental renderer.
    #[must_use]
    pub fn new(max_cache_entries: usize) -> Self {
        Self {
            tracker: ChangeTracker::new(),
            layer_cache: HashMap::new(),
            max_cache_entries,
            total_renders: 0,
            cache_hits: 0,
        }
    }

    /// Returns a reference to the change tracker.
    #[must_use]
    pub fn tracker(&self) -> &ChangeTracker {
        &self.tracker
    }

    /// Returns a mutable reference to the change tracker.
    pub fn tracker_mut(&mut self) -> &mut ChangeTracker {
        &mut self.tracker
    }

    /// Determines which tracks need re-rendering for a given frame.
    ///
    /// Returns a tuple `(dirty_tracks, cached_tracks)`.
    #[must_use]
    pub fn plan_render(&self, frame: i64, all_tracks: &[TrackId]) -> RenderPlan {
        let mut dirty = Vec::new();
        let mut cached = Vec::new();

        for &track_id in all_tracks {
            if self.tracker.is_track_dirty(track_id, frame) {
                dirty.push(track_id);
            } else if self.layer_cache.contains_key(&(track_id, frame)) {
                cached.push(track_id);
            } else {
                // Not dirty but no cache either — must render
                dirty.push(track_id);
            }
        }

        RenderPlan { dirty, cached, frame }
    }

    /// Store a rendered layer in the cache.
    pub fn cache_layer(&mut self, layer: CachedLayer) {
        let key = (layer.track_id, layer.frame);

        // Evict if at capacity
        if self.layer_cache.len() >= self.max_cache_entries && !self.layer_cache.contains_key(&key) {
            // Simple eviction: remove oldest entry
            if let Some(&oldest_key) = self.layer_cache.keys().next() {
                self.layer_cache.remove(&oldest_key);
            }
        }

        self.layer_cache.insert(key, layer);
    }

    /// Retrieve a cached layer.
    #[must_use]
    pub fn get_cached_layer(&self, track_id: TrackId, frame: i64) -> Option<&CachedLayer> {
        self.layer_cache.get(&(track_id, frame))
    }

    /// Record a render (for statistics).
    pub fn record_render(&mut self) {
        self.total_renders += 1;
    }

    /// Record a cache hit (for statistics).
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    /// Returns the cache hit ratio (0.0–1.0).
    #[must_use]
    pub fn cache_hit_ratio(&self) -> f64 {
        if self.total_renders == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / self.total_renders as f64
    }

    /// Clear the entire layer cache.
    pub fn clear_cache(&mut self) {
        self.layer_cache.clear();
        self.cache_hits = 0;
        self.total_renders = 0;
    }

    /// Invalidate cache entries for a specific track.
    pub fn invalidate_track(&mut self, track_id: TrackId) {
        self.layer_cache.retain(|&(tid, _), _| tid != track_id);
    }

    /// Number of cached layers.
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.layer_cache.len()
    }

    /// Total renders performed.
    #[must_use]
    pub fn total_renders(&self) -> u64 {
        self.total_renders
    }
}

/// A render plan produced by [`IncrementalRenderer::plan_render`].
#[derive(Debug, Clone)]
pub struct RenderPlan {
    /// Tracks that need re-rendering.
    pub dirty: Vec<TrackId>,
    /// Tracks whose cached layers can be reused.
    pub cached: Vec<TrackId>,
    /// Frame number this plan is for.
    pub frame: i64,
}

impl RenderPlan {
    /// Total number of tracks in the plan.
    #[must_use]
    pub fn total_tracks(&self) -> usize {
        self.dirty.len() + self.cached.len()
    }

    /// Fraction of tracks that can be skipped (0.0–1.0).
    #[must_use]
    pub fn skip_ratio(&self) -> f64 {
        let total = self.total_tracks();
        if total == 0 {
            return 0.0;
        }
        self.cached.len() as f64 / total as f64
    }

    /// Returns `true` if all tracks need re-rendering (no cache reuse).
    #[must_use]
    pub fn is_full_render(&self) -> bool {
        self.cached.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_change_tracker_new_is_clean() {
        let tracker = ChangeTracker::new();
        assert!(!tracker.has_changes());
        assert_eq!(tracker.global_version(), 0);
    }

    #[test]
    fn test_register_and_mark_dirty() {
        let mut tracker = ChangeTracker::new();
        let tid = TrackId::new();
        tracker.register_track(tid);
        // Newly registered tracks are fully dirty
        assert!(tracker.is_track_dirty(tid, 0));
        assert!(tracker.is_track_dirty(tid, 1000));
    }

    #[test]
    fn test_mark_rendered_cleans_tracks() {
        let mut tracker = ChangeTracker::new();
        let tid = TrackId::new();
        tracker.register_track(tid);
        tracker.mark_rendered();
        // After rendering, the track should be clean
        assert!(!tracker.is_track_dirty(tid, 50));
    }

    #[test]
    fn test_mark_range_dirty() {
        let mut tracker = ChangeTracker::new();
        let tid = TrackId::new();
        tracker.register_track(tid);
        tracker.mark_rendered();
        tracker.mark_range_dirty(tid, 10, 50);
        assert!(tracker.is_track_dirty(tid, 20));
        assert!(!tracker.is_track_dirty(tid, 5));
        assert!(!tracker.is_track_dirty(tid, 60));
    }

    #[test]
    fn test_mark_clip_changed_increments_version() {
        let mut tracker = ChangeTracker::new();
        let tid = TrackId::new();
        let cid = ClipId::new();
        tracker.register_track(tid);
        let v0 = tracker.global_version();
        tracker.mark_clip_changed(tid, cid);
        assert!(tracker.global_version() > v0);
        assert!(tracker.dirty_clips(tid).contains(&cid));
    }

    #[test]
    fn test_mark_all_dirty() {
        let mut tracker = ChangeTracker::new();
        let t1 = TrackId::new();
        let t2 = TrackId::new();
        tracker.register_track(t1);
        tracker.register_track(t2);
        tracker.mark_rendered();
        tracker.mark_all_dirty();
        assert!(tracker.is_track_dirty(t1, 0));
        assert!(tracker.is_track_dirty(t2, 0));
    }

    #[test]
    fn test_dirty_tracks_at() {
        let mut tracker = ChangeTracker::new();
        let t1 = TrackId::new();
        let t2 = TrackId::new();
        tracker.register_track(t1);
        tracker.register_track(t2);
        tracker.mark_rendered();

        tracker.mark_range_dirty(t1, 0, 100);
        let dirty = tracker.dirty_tracks_at(50);
        assert_eq!(dirty.len(), 1);
        assert_eq!(dirty[0], t1);
    }

    #[test]
    fn test_unregister_track() {
        let mut tracker = ChangeTracker::new();
        let tid = TrackId::new();
        tracker.register_track(tid);
        tracker.unregister_track(tid);
        assert!(!tracker.is_track_dirty(tid, 0));
        assert_eq!(tracker.track_count(), 0);
    }

    #[test]
    fn test_incremental_renderer_plan_all_dirty() {
        let mut renderer = IncrementalRenderer::new(100);
        let t1 = TrackId::new();
        let t2 = TrackId::new();
        renderer.tracker_mut().register_track(t1);
        renderer.tracker_mut().register_track(t2);

        let plan = renderer.plan_render(0, &[t1, t2]);
        assert!(plan.is_full_render());
        assert_eq!(plan.dirty.len(), 2);
        assert_eq!(plan.cached.len(), 0);
    }

    #[test]
    fn test_incremental_renderer_plan_with_cache() {
        let mut renderer = IncrementalRenderer::new(100);
        let t1 = TrackId::new();
        let t2 = TrackId::new();
        renderer.tracker_mut().register_track(t1);
        renderer.tracker_mut().register_track(t2);
        renderer.tracker_mut().mark_rendered();

        // Cache a layer for t1 at frame 0
        renderer.cache_layer(CachedLayer {
            track_id: t1,
            version: 1,
            frame: 0,
            data: vec![0; 4],
            width: 1,
            height: 1,
        });

        // Only t2 is dirty (mark it dirty after clean)
        renderer.tracker_mut().mark_track_dirty(t2);

        let plan = renderer.plan_render(0, &[t1, t2]);
        assert_eq!(plan.cached.len(), 1);
        assert_eq!(plan.dirty.len(), 1);
        assert!(!plan.is_full_render());
    }

    #[test]
    fn test_cache_hit_ratio() {
        let mut renderer = IncrementalRenderer::new(100);
        renderer.record_render();
        renderer.record_render();
        renderer.record_cache_hit();
        assert!((renderer.cache_hit_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_hit_ratio_zero_renders() {
        let renderer = IncrementalRenderer::new(100);
        assert!((renderer.cache_hit_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_invalidate_track_cache() {
        let mut renderer = IncrementalRenderer::new(100);
        let t1 = TrackId::new();
        let t2 = TrackId::new();
        renderer.cache_layer(CachedLayer {
            track_id: t1,
            version: 1,
            frame: 0,
            data: vec![0; 4],
            width: 1,
            height: 1,
        });
        renderer.cache_layer(CachedLayer {
            track_id: t2,
            version: 1,
            frame: 0,
            data: vec![0; 4],
            width: 1,
            height: 1,
        });
        assert_eq!(renderer.cache_size(), 2);
        renderer.invalidate_track(t1);
        assert_eq!(renderer.cache_size(), 1);
        assert!(renderer.get_cached_layer(t1, 0).is_none());
        assert!(renderer.get_cached_layer(t2, 0).is_some());
    }

    #[test]
    fn test_clear_cache() {
        let mut renderer = IncrementalRenderer::new(100);
        let tid = TrackId::new();
        renderer.cache_layer(CachedLayer {
            track_id: tid,
            version: 1,
            frame: 0,
            data: vec![0; 4],
            width: 1,
            height: 1,
        });
        renderer.record_render();
        renderer.record_cache_hit();
        renderer.clear_cache();
        assert_eq!(renderer.cache_size(), 0);
        assert_eq!(renderer.total_renders(), 0);
    }

    #[test]
    fn test_render_plan_skip_ratio() {
        let plan = RenderPlan {
            dirty: vec![TrackId::new()],
            cached: vec![TrackId::new(), TrackId::new(), TrackId::new()],
            frame: 0,
        };
        assert!((plan.skip_ratio() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_render_plan_empty() {
        let plan = RenderPlan {
            dirty: vec![],
            cached: vec![],
            frame: 0,
        };
        assert_eq!(plan.total_tracks(), 0);
        assert!((plan.skip_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_track_version() {
        let mut tracker = ChangeTracker::new();
        let tid = TrackId::new();
        tracker.register_track(tid);
        let v1 = tracker.track_version(tid);
        tracker.mark_track_dirty(tid);
        let v2 = tracker.track_version(tid);
        assert!(v2 > v1);
    }

    #[test]
    fn test_mark_track_rendered() {
        let mut tracker = ChangeTracker::new();
        let tid = TrackId::new();
        tracker.register_track(tid);
        assert!(tracker.is_track_dirty(tid, 0));
        tracker.mark_track_rendered(tid);
        assert!(!tracker.is_track_dirty(tid, 0));
    }

    #[test]
    fn test_has_changes() {
        let mut tracker = ChangeTracker::new();
        let tid = TrackId::new();
        assert!(!tracker.has_changes());
        tracker.register_track(tid);
        // register_track doesn't increment global_version directly
        tracker.mark_track_dirty(tid);
        assert!(tracker.has_changes());
        tracker.mark_rendered();
        assert!(!tracker.has_changes());
    }

    #[test]
    fn test_is_track_dirty_any() {
        let mut tracker = ChangeTracker::new();
        let tid = TrackId::new();
        tracker.register_track(tid);
        assert!(tracker.is_track_dirty_any(tid));
        tracker.mark_rendered();
        assert!(!tracker.is_track_dirty_any(tid));
        tracker.mark_clip_changed(tid, ClipId::new());
        assert!(tracker.is_track_dirty_any(tid));
    }

    #[test]
    fn test_dirty_range_expansion() {
        let mut tracker = ChangeTracker::new();
        let tid = TrackId::new();
        tracker.register_track(tid);
        tracker.mark_rendered();
        tracker.mark_range_dirty(tid, 10, 20);
        tracker.mark_range_dirty(tid, 30, 50);
        // Range should expand to cover 10..50
        assert!(tracker.is_track_dirty(tid, 15));
        assert!(tracker.is_track_dirty(tid, 35));
        // Expanded range includes gap between 20 and 30
        assert!(tracker.is_track_dirty(tid, 25));
    }

    #[test]
    fn test_cache_eviction() {
        let mut renderer = IncrementalRenderer::new(2);
        let t1 = TrackId::new();
        let t2 = TrackId::new();
        let t3 = TrackId::new();

        renderer.cache_layer(CachedLayer {
            track_id: t1, version: 1, frame: 0, data: vec![1], width: 1, height: 1,
        });
        renderer.cache_layer(CachedLayer {
            track_id: t2, version: 1, frame: 0, data: vec![2], width: 1, height: 1,
        });
        assert_eq!(renderer.cache_size(), 2);

        // Adding a third layer should evict one
        renderer.cache_layer(CachedLayer {
            track_id: t3, version: 1, frame: 0, data: vec![3], width: 1, height: 1,
        });
        assert_eq!(renderer.cache_size(), 2);
    }
}
