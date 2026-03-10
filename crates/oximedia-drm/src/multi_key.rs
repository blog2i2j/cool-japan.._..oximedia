//! Multi-key content encryption scheme.
//!
//! Supports assigning different encryption keys to different tracks (video,
//! audio, subtitle) and different time periods within a single piece of
//! content, following CENC multi-key best practices.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Identifies a track type for multi-key assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrackKind {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
    /// Subtitle / timed-text track.
    Subtitle,
    /// Data / metadata track.
    Data,
}

/// A key slot: the content key ID and the raw key material.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeySlot {
    /// Key identifier (typically 16 bytes, hex-encoded for display).
    pub key_id: Vec<u8>,
    /// Raw key material (typically 16 bytes for AES-128).
    pub key: Vec<u8>,
    /// Human-readable label for this key slot.
    pub label: String,
}

impl KeySlot {
    /// Create a new key slot.
    #[must_use]
    pub fn new(key_id: Vec<u8>, key: Vec<u8>, label: &str) -> Self {
        Self {
            key_id,
            key,
            label: label.to_string(),
        }
    }

    /// Key length in bytes.
    #[must_use]
    pub fn key_len(&self) -> usize {
        self.key.len()
    }

    /// Validate that the key ID and key are both 16 bytes (AES-128 standard).
    #[must_use]
    pub fn is_valid_aes128(&self) -> bool {
        self.key_id.len() == 16 && self.key.len() == 16
    }
}

/// Describes a time period within the content for key rotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyPeriod {
    /// Period index (0-based).
    pub index: u32,
    /// Start time offset in milliseconds from the content start.
    pub start_ms: u64,
    /// End time offset in milliseconds (exclusive). `None` means until end.
    pub end_ms: Option<u64>,
    /// Track-to-key-slot mapping for this period.
    pub track_keys: HashMap<String, usize>,
}

impl KeyPeriod {
    /// Create a new key period.
    #[must_use]
    pub fn new(index: u32, start_ms: u64, end_ms: Option<u64>) -> Self {
        Self {
            index,
            start_ms,
            end_ms,
            track_keys: HashMap::new(),
        }
    }

    /// Assign a key slot index to a track kind for this period.
    pub fn assign_key(&mut self, track: TrackKind, slot_index: usize) {
        self.track_keys.insert(format!("{track:?}"), slot_index);
    }

    /// Get the key slot index for a track kind.
    #[must_use]
    pub fn get_key_index(&self, track: TrackKind) -> Option<usize> {
        self.track_keys.get(&format!("{track:?}")).copied()
    }

    /// Duration in milliseconds (`None` if open-ended).
    #[must_use]
    pub fn duration_ms(&self) -> Option<u64> {
        self.end_ms.map(|e| e.saturating_sub(self.start_ms))
    }

    /// Check if a given timestamp falls within this period.
    #[must_use]
    pub fn contains_time(&self, ms: u64) -> bool {
        if ms < self.start_ms {
            return false;
        }
        match self.end_ms {
            Some(end) => ms < end,
            None => true,
        }
    }
}

/// Multi-key encryption scheme for a piece of content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiKeyScheme {
    /// Content identifier.
    pub content_id: String,
    /// All key slots used by this scheme.
    pub slots: Vec<KeySlot>,
    /// Key periods (sorted by start time).
    pub periods: Vec<KeyPeriod>,
    /// Default key slot index for tracks not explicitly assigned.
    pub default_slot: Option<usize>,
}

impl MultiKeyScheme {
    /// Create a new multi-key scheme.
    #[must_use]
    pub fn new(content_id: &str) -> Self {
        Self {
            content_id: content_id.to_string(),
            slots: Vec::new(),
            periods: Vec::new(),
            default_slot: None,
        }
    }

    /// Add a key slot; returns the slot index.
    pub fn add_slot(&mut self, slot: KeySlot) -> usize {
        let idx = self.slots.len();
        self.slots.push(slot);
        idx
    }

    /// Get a key slot by index.
    #[must_use]
    pub fn get_slot(&self, index: usize) -> Option<&KeySlot> {
        self.slots.get(index)
    }

    /// Set the default key slot.
    pub fn set_default_slot(&mut self, index: usize) {
        self.default_slot = Some(index);
    }

    /// Add a key period. Periods should be added in order.
    pub fn add_period(&mut self, period: KeyPeriod) {
        self.periods.push(period);
    }

    /// Number of key slots.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Number of key periods.
    #[must_use]
    pub fn period_count(&self) -> usize {
        self.periods.len()
    }

    /// Resolve the key slot for a given track kind and timestamp.
    ///
    /// Returns the slot index and a reference to the `KeySlot`.
    #[must_use]
    pub fn resolve(&self, track: TrackKind, time_ms: u64) -> Option<(usize, &KeySlot)> {
        // Find the period containing this timestamp.
        for period in &self.periods {
            if period.contains_time(time_ms) {
                if let Some(idx) = period.get_key_index(track) {
                    return self.slots.get(idx).map(|s| (idx, s));
                }
            }
        }
        // Fallback to default slot.
        self.default_slot
            .and_then(|idx| self.slots.get(idx).map(|s| (idx, s)))
    }

    /// Collect all unique key IDs used across all slots.
    #[must_use]
    pub fn all_key_ids(&self) -> Vec<&[u8]> {
        self.slots.iter().map(|s| s.key_id.as_slice()).collect()
    }

    /// Validate the entire scheme: all slot references must be valid.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        let n = self.slots.len();
        if let Some(d) = self.default_slot {
            if d >= n {
                errors.push(format!("default_slot {d} out of range (have {n} slots)"));
            }
        }
        for period in &self.periods {
            for (track, &idx) in &period.track_keys {
                if idx >= n {
                    errors.push(format!(
                        "period {} track {track} references slot {idx} but only {n} exist",
                        period.index
                    ));
                }
            }
        }
        // Check for overlapping periods.
        for i in 1..self.periods.len() {
            let prev = &self.periods[i - 1];
            let curr = &self.periods[i];
            if let Some(prev_end) = prev.end_ms {
                if curr.start_ms < prev_end {
                    errors.push(format!(
                        "period {} overlaps with period {}",
                        curr.index, prev.index
                    ));
                }
            }
        }
        errors
    }
}

/// Builder for constructing `MultiKeyScheme` step by step.
#[derive(Debug)]
pub struct MultiKeyBuilder {
    /// The scheme being built.
    scheme: MultiKeyScheme,
    /// Current period index counter.
    period_counter: u32,
}

impl MultiKeyBuilder {
    /// Start building a new scheme.
    #[must_use]
    pub fn new(content_id: &str) -> Self {
        Self {
            scheme: MultiKeyScheme::new(content_id),
            period_counter: 0,
        }
    }

    /// Add a key slot.
    #[must_use]
    pub fn slot(mut self, key_id: Vec<u8>, key: Vec<u8>, label: &str) -> Self {
        self.scheme.add_slot(KeySlot::new(key_id, key, label));
        self
    }

    /// Set the default slot index.
    #[must_use]
    pub fn default_slot(mut self, index: usize) -> Self {
        self.scheme.set_default_slot(index);
        self
    }

    /// Add a period with assignments. Returns updated builder.
    #[must_use]
    pub fn period(
        mut self,
        start_ms: u64,
        end_ms: Option<u64>,
        assignments: &[(TrackKind, usize)],
    ) -> Self {
        let mut p = KeyPeriod::new(self.period_counter, start_ms, end_ms);
        for &(track, idx) in assignments {
            p.assign_key(track, idx);
        }
        self.scheme.add_period(p);
        self.period_counter += 1;
        self
    }

    /// Finalize and return the scheme.
    #[must_use]
    pub fn build(self) -> MultiKeyScheme {
        self.scheme
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key(id: u8) -> Vec<u8> {
        vec![id; 16]
    }

    #[test]
    fn test_key_slot_new() {
        let slot = KeySlot::new(make_key(1), make_key(2), "video_key");
        assert_eq!(slot.label, "video_key");
        assert_eq!(slot.key_len(), 16);
    }

    #[test]
    fn test_key_slot_valid_aes128() {
        let good = KeySlot::new(make_key(1), make_key(2), "ok");
        assert!(good.is_valid_aes128());
        let bad = KeySlot::new(vec![1, 2], vec![3, 4], "short");
        assert!(!bad.is_valid_aes128());
    }

    #[test]
    fn test_key_period_contains() {
        let p = KeyPeriod::new(0, 1000, Some(2000));
        assert!(!p.contains_time(999));
        assert!(p.contains_time(1000));
        assert!(p.contains_time(1500));
        assert!(!p.contains_time(2000)); // exclusive end
    }

    #[test]
    fn test_key_period_open_ended() {
        let p = KeyPeriod::new(0, 500, None);
        assert!(!p.contains_time(499));
        assert!(p.contains_time(500));
        assert!(p.contains_time(u64::MAX));
    }

    #[test]
    fn test_key_period_duration() {
        let p1 = KeyPeriod::new(0, 1000, Some(3000));
        assert_eq!(p1.duration_ms(), Some(2000));
        let p2 = KeyPeriod::new(0, 1000, None);
        assert_eq!(p2.duration_ms(), None);
    }

    #[test]
    fn test_key_period_assign_get() {
        let mut p = KeyPeriod::new(0, 0, None);
        p.assign_key(TrackKind::Video, 0);
        p.assign_key(TrackKind::Audio, 1);
        assert_eq!(p.get_key_index(TrackKind::Video), Some(0));
        assert_eq!(p.get_key_index(TrackKind::Audio), Some(1));
        assert_eq!(p.get_key_index(TrackKind::Subtitle), None);
    }

    #[test]
    fn test_scheme_add_slot() {
        let mut s = MultiKeyScheme::new("content1");
        let idx = s.add_slot(KeySlot::new(make_key(1), make_key(2), "v"));
        assert_eq!(idx, 0);
        assert_eq!(s.slot_count(), 1);
    }

    #[test]
    fn test_scheme_resolve_from_period() {
        let mut s = MultiKeyScheme::new("c1");
        s.add_slot(KeySlot::new(make_key(0xAA), make_key(0xBB), "video"));
        s.add_slot(KeySlot::new(make_key(0xCC), make_key(0xDD), "audio"));
        let mut p = KeyPeriod::new(0, 0, None);
        p.assign_key(TrackKind::Video, 0);
        p.assign_key(TrackKind::Audio, 1);
        s.add_period(p);

        let (idx, slot) = s
            .resolve(TrackKind::Video, 5000)
            .expect("operation should succeed");
        assert_eq!(idx, 0);
        assert_eq!(slot.label, "video");

        let (idx, slot) = s
            .resolve(TrackKind::Audio, 5000)
            .expect("operation should succeed");
        assert_eq!(idx, 1);
        assert_eq!(slot.label, "audio");
    }

    #[test]
    fn test_scheme_resolve_fallback_default() {
        let mut s = MultiKeyScheme::new("c2");
        s.add_slot(KeySlot::new(make_key(1), make_key(2), "default"));
        s.set_default_slot(0);
        // No periods, should fall back to default.
        let (idx, _) = s
            .resolve(TrackKind::Subtitle, 0)
            .expect("operation should succeed");
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_scheme_resolve_no_match() {
        let s = MultiKeyScheme::new("c3");
        assert!(s.resolve(TrackKind::Video, 0).is_none());
    }

    #[test]
    fn test_scheme_validate_ok() {
        let scheme = MultiKeyBuilder::new("c1")
            .slot(make_key(1), make_key(2), "v")
            .slot(make_key(3), make_key(4), "a")
            .default_slot(0)
            .period(
                0,
                Some(5000),
                &[(TrackKind::Video, 0), (TrackKind::Audio, 1)],
            )
            .period(5000, None, &[(TrackKind::Video, 0), (TrackKind::Audio, 1)])
            .build();
        assert!(scheme.validate().is_empty());
    }

    #[test]
    fn test_scheme_validate_bad_slot_ref() {
        let scheme = MultiKeyBuilder::new("c2")
            .slot(make_key(1), make_key(2), "v")
            .period(0, None, &[(TrackKind::Video, 5)]) // slot 5 doesn't exist
            .build();
        let errors = scheme.validate();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_scheme_validate_overlap() {
        let scheme = MultiKeyBuilder::new("c3")
            .slot(make_key(1), make_key(2), "v")
            .period(0, Some(3000), &[(TrackKind::Video, 0)])
            .period(2000, None, &[(TrackKind::Video, 0)]) // overlaps
            .build();
        let errors = scheme.validate();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_all_key_ids() {
        let scheme = MultiKeyBuilder::new("c4")
            .slot(make_key(0xAA), make_key(0xBB), "a")
            .slot(make_key(0xCC), make_key(0xDD), "b")
            .build();
        let ids = scheme.all_key_ids();
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], &make_key(0xAA)[..]);
    }

    #[test]
    fn test_builder_full() {
        let scheme = MultiKeyBuilder::new("movie")
            .slot(make_key(1), make_key(10), "video_key")
            .slot(make_key(2), make_key(20), "audio_key")
            .default_slot(0)
            .period(
                0,
                Some(60_000),
                &[(TrackKind::Video, 0), (TrackKind::Audio, 1)],
            )
            .period(
                60_000,
                None,
                &[(TrackKind::Video, 0), (TrackKind::Audio, 1)],
            )
            .build();
        assert_eq!(scheme.content_id, "movie");
        assert_eq!(scheme.slot_count(), 2);
        assert_eq!(scheme.period_count(), 2);
        assert!(scheme.validate().is_empty());
    }
}
