//! Scope snapshot store for QC workflows.
//!
//! This module provides a time-ordered, capacity-bounded store of scope
//! renders (RGBA byte buffers), together with a diff engine for detecting
//! visual changes between consecutive frames.  It is designed for broadcast
//! quality-control pipelines where an operator needs to:
//!
//! - Record scope renders over time without unbounded memory growth.
//! - Quickly retrieve the latest render for any scope type.
//! - Compare the two most recent renders and obtain a scalar diff score.
//! - Walk the history of a particular scope type in chronological order.
//!
//! # Architecture
//!
//! `ScopeSnapshotStore` maintains a single `VecDeque` per [`ScopeType`] variant
//! through a `HashMap` keyed by a `u8` discriminant.  Each bucket holds at most
//! `capacity` entries; older snapshots are evicted when the bucket is full.
//!
//! # Diff Score
//!
//! `SnapshotDiff::compute` measures the **mean absolute difference** between
//! corresponding RGBA bytes of two snapshots and divides by 255, yielding a
//! score in `[0.0, 1.0]` where `0.0` means pixel-identical and `1.0` means
//! maximally different.  If the two snapshots have different data lengths,
//! only the overlapping prefix is compared, and the length mismatch is
//! reflected in the diff score.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::collections::{HashMap, VecDeque};

// ─── ScopeType ─────────────────────────────────────────────────────────────

/// The kind of scope whose render is being stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopeType {
    /// Luma or RGB waveform monitor.
    Waveform,
    /// YUV/IQ vectorscope.
    Vectorscope,
    /// Luma or RGB histogram.
    Histogram,
    /// Luma or RGB parade.
    Parade,
    /// CIE 1931 xy chromaticity diagram.
    CieXy,
}

/// Stable discriminant used as `HashMap` key.
impl ScopeType {
    fn key(self) -> u8 {
        match self {
            Self::Waveform => 0,
            Self::Vectorscope => 1,
            Self::Histogram => 2,
            Self::Parade => 3,
            Self::CieXy => 4,
        }
    }
}

// ─── ScopeSnapshot ─────────────────────────────────────────────────────────

/// A rendered scope image captured at a specific time with associated numeric
/// metadata (e.g. frame number, mean luma, peak value, etc.).
#[derive(Debug, Clone)]
pub struct ScopeSnapshot {
    /// Which scope produced this render.
    pub scope_type: ScopeType,
    /// Wall-clock or media time in seconds when the snapshot was taken.
    pub timestamp_secs: f64,
    /// RGBA pixel data (any dimensions; length = `width * height * 4`).
    pub data: Vec<u8>,
    /// Numeric metadata key/value pairs (e.g. `"mean_luma"`, `"peak_red"`).
    pub metadata: HashMap<String, f64>,
}

impl ScopeSnapshot {
    /// Construct a new snapshot.
    #[must_use]
    pub fn new(
        scope_type: ScopeType,
        timestamp_secs: f64,
        data: Vec<u8>,
        metadata: HashMap<String, f64>,
    ) -> Self {
        Self {
            scope_type,
            timestamp_secs,
            data,
            metadata,
        }
    }
}

// ─── SnapshotComparison ─────────────────────────────────────────────────────

/// Result of comparing two scope snapshots.
#[derive(Debug, Clone)]
pub struct SnapshotComparison {
    /// Earlier snapshot.
    pub snapshot_a: ScopeSnapshot,
    /// Later snapshot.
    pub snapshot_b: ScopeSnapshot,
    /// Normalised mean absolute pixel difference in `[0.0, 1.0]`.
    ///
    /// `0.0` = pixel-identical; `1.0` = maximally different.
    pub diff_score: f32,
    /// Number of 4×4-pixel regions (blocks) where the mean absolute difference
    /// exceeds 0.05 (i.e. roughly 13/255 grey levels).
    pub changed_regions: u32,
}

// ─── SnapshotDiff ──────────────────────────────────────────────────────────

/// Stateless diff engine.
pub struct SnapshotDiff;

impl SnapshotDiff {
    /// Compute the difference between two snapshots.
    ///
    /// # Algorithm
    ///
    /// 1. Take the overlapping byte prefix (`min(len_a, len_b)`).
    /// 2. Compute the mean absolute byte difference, normalised to `[0, 1]`.
    /// 3. Count 8×8-pixel RGBA blocks (32-byte strides) whose mean absolute
    ///    difference exceeds a threshold of `0.05` (≈13/255 grey levels).
    ///
    /// If either snapshot has zero bytes, `diff_score` is `0.0` when both are
    /// empty, or `1.0` when only one is.
    #[must_use]
    pub fn compute(a: &ScopeSnapshot, b: &ScopeSnapshot) -> SnapshotComparison {
        let len_a = a.data.len();
        let len_b = b.data.len();

        let diff_score = if len_a == 0 && len_b == 0 {
            0.0f32
        } else if len_a == 0 || len_b == 0 {
            1.0f32
        } else {
            let overlap = len_a.min(len_b);
            let total_diff: u64 = a.data[..overlap]
                .iter()
                .zip(b.data[..overlap].iter())
                .map(|(&x, &y)| u64::from(x.abs_diff(y)))
                .sum();

            // Account for length mismatch: treat missing bytes as max diff
            let length_penalty = len_a.abs_diff(len_b) as u64 * 255;
            let total_bytes = len_a.max(len_b) as u64;

            ((total_diff + length_penalty) as f32 / (total_bytes as f32 * 255.0)).clamp(0.0, 1.0)
        };

        // Count changed 8×8-pixel blocks (stride = 8*4 = 32 bytes per row-slice).
        // We treat the data as a flat byte sequence and chunk it into 256-byte
        // blocks (= 8×8 pixels × 4 bytes), measuring the mean diff per block.
        let changed_regions = Self::count_changed_regions(&a.data, &b.data);

        SnapshotComparison {
            snapshot_a: a.clone(),
            snapshot_b: b.clone(),
            diff_score,
            changed_regions,
        }
    }

    /// Count 256-byte blocks (representing 8×8-pixel RGBA tiles) where the
    /// mean absolute byte difference exceeds the threshold of ~13/255 ≈ 0.05.
    fn count_changed_regions(a: &[u8], b: &[u8]) -> u32 {
        const BLOCK: usize = 256; // 8×8×4
        const THRESHOLD_SUM: u32 = 13 * BLOCK as u32; // sum threshold for block

        let overlap = a.len().min(b.len());
        let full_blocks = overlap / BLOCK;

        let mut changed: u32 = 0;
        for i in 0..full_blocks {
            let start = i * BLOCK;
            let block_diff: u32 = a[start..start + BLOCK]
                .iter()
                .zip(b[start..start + BLOCK].iter())
                .map(|(&x, &y)| u32::from(x.abs_diff(y)))
                .sum();
            if block_diff > THRESHOLD_SUM {
                changed += 1;
            }
        }

        // Any trailing partial block or length mismatch counts as one extra
        // changed region.
        if overlap % BLOCK != 0 || a.len() != b.len() {
            changed += 1;
        }

        changed
    }
}

// ─── ScopeSnapshotStore ─────────────────────────────────────────────────────

/// A capacity-bounded, time-ordered store of scope snapshots.
///
/// Each [`ScopeType`] has its own bucket of at most `capacity` snapshots.
/// When a bucket is full, the oldest entry is evicted before the new one is
/// inserted (FIFO eviction).
pub struct ScopeSnapshotStore {
    /// Maximum number of snapshots retained **per scope type**.
    pub capacity: usize,
    buckets: HashMap<u8, VecDeque<ScopeSnapshot>>,
}

impl ScopeSnapshotStore {
    /// Create a new store with the given per-scope capacity.
    ///
    /// `capacity` must be at least 1; values of 0 are silently clamped to 1.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            buckets: HashMap::new(),
        }
    }

    /// Add a snapshot to the store.
    ///
    /// If the bucket for `snapshot.scope_type` already holds `capacity`
    /// entries, the oldest one is evicted first.
    pub fn add(&mut self, snapshot: ScopeSnapshot) {
        let key = snapshot.scope_type.key();
        let bucket = self.buckets.entry(key).or_default();
        if bucket.len() >= self.capacity {
            bucket.pop_front();
        }
        bucket.push_back(snapshot);
    }

    /// Return a reference to the most recently added snapshot for the given
    /// scope type, or `None` if no snapshots have been stored yet.
    #[must_use]
    pub fn latest(&self, scope_type: ScopeType) -> Option<&ScopeSnapshot> {
        self.buckets.get(&scope_type.key())?.back()
    }

    /// Compare the two most recent snapshots for `scope_type`.
    ///
    /// Returns `None` if fewer than two snapshots exist for that type.
    #[must_use]
    pub fn compare_latest_two(&self, scope_type: ScopeType) -> Option<SnapshotComparison> {
        let bucket = self.buckets.get(&scope_type.key())?;
        let len = bucket.len();
        if len < 2 {
            return None;
        }
        let a = &bucket[len - 2];
        let b = &bucket[len - 1];
        Some(SnapshotDiff::compute(a, b))
    }

    /// Return all snapshots for `scope_type` in chronological order (oldest
    /// first).  Returns an empty slice if none have been stored.
    #[must_use]
    pub fn history(&self, scope_type: ScopeType) -> Vec<&ScopeSnapshot> {
        match self.buckets.get(&scope_type.key()) {
            Some(bucket) => bucket.iter().collect(),
            None => Vec::new(),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(scope_type: ScopeType, ts: f64, fill: u8, len: usize) -> ScopeSnapshot {
        let mut meta = HashMap::new();
        meta.insert("fill".to_string(), f64::from(fill));
        ScopeSnapshot::new(scope_type, ts, vec![fill; len], meta)
    }

    // ── add / latest ────────────────────────────────────────────────────────

    #[test]
    fn test_add_and_latest_basic() {
        let mut store = ScopeSnapshotStore::new(4);
        store.add(make_snapshot(ScopeType::Waveform, 0.0, 100, 64));
        let latest = store.latest(ScopeType::Waveform);
        assert!(latest.is_some());
        assert_eq!(latest.expect("should exist").data[0], 100);
    }

    #[test]
    fn test_latest_returns_none_for_empty_store() {
        let store = ScopeSnapshotStore::new(4);
        assert!(store.latest(ScopeType::Histogram).is_none());
    }

    #[test]
    fn test_latest_returns_most_recent() {
        let mut store = ScopeSnapshotStore::new(4);
        store.add(make_snapshot(ScopeType::Parade, 0.0, 10, 64));
        store.add(make_snapshot(ScopeType::Parade, 1.0, 20, 64));
        store.add(make_snapshot(ScopeType::Parade, 2.0, 30, 64));
        let latest = store.latest(ScopeType::Parade).expect("should exist");
        assert_eq!(latest.data[0], 30);
        assert!((latest.timestamp_secs - 2.0).abs() < 1e-9);
    }

    // ── capacity eviction ────────────────────────────────────────────────────

    #[test]
    fn test_capacity_eviction() {
        let mut store = ScopeSnapshotStore::new(3);
        for i in 0u8..6 {
            store.add(make_snapshot(ScopeType::Waveform, f64::from(i), i, 64));
        }
        // Only the last 3 should remain
        let hist = store.history(ScopeType::Waveform);
        assert_eq!(hist.len(), 3);
        assert_eq!(hist[0].data[0], 3);
        assert_eq!(hist[2].data[0], 5);
    }

    #[test]
    fn test_capacity_one() {
        let mut store = ScopeSnapshotStore::new(1);
        store.add(make_snapshot(ScopeType::CieXy, 0.0, 7, 16));
        store.add(make_snapshot(ScopeType::CieXy, 1.0, 42, 16));
        let hist = store.history(ScopeType::CieXy);
        assert_eq!(hist.len(), 1);
        assert_eq!(hist[0].data[0], 42);
    }

    // ── diff: identical ──────────────────────────────────────────────────────

    #[test]
    fn test_diff_identical_snapshots() {
        let a = make_snapshot(ScopeType::Waveform, 0.0, 128, 256);
        let b = make_snapshot(ScopeType::Waveform, 1.0, 128, 256);
        let cmp = SnapshotDiff::compute(&a, &b);
        assert_eq!(cmp.diff_score, 0.0, "diff_score should be 0 for identical data");
    }

    // ── diff: completely different ───────────────────────────────────────────

    #[test]
    fn test_diff_maximally_different() {
        let a = make_snapshot(ScopeType::Histogram, 0.0, 0, 256);
        let b = make_snapshot(ScopeType::Histogram, 1.0, 255, 256);
        let cmp = SnapshotDiff::compute(&a, &b);
        assert!(
            cmp.diff_score > 0.9,
            "diff_score={} should be near 1.0",
            cmp.diff_score
        );
    }

    // ── diff: empty ──────────────────────────────────────────────────────────

    #[test]
    fn test_diff_both_empty() {
        let a = make_snapshot(ScopeType::Vectorscope, 0.0, 0, 0);
        let b = make_snapshot(ScopeType::Vectorscope, 1.0, 0, 0);
        let cmp = SnapshotDiff::compute(&a, &b);
        assert_eq!(cmp.diff_score, 0.0);
    }

    // ── compare_latest_two ───────────────────────────────────────────────────

    #[test]
    fn test_compare_latest_two_requires_two_snapshots() {
        let mut store = ScopeSnapshotStore::new(4);
        store.add(make_snapshot(ScopeType::Waveform, 0.0, 50, 64));
        // Only one snapshot — no comparison possible
        assert!(store.compare_latest_two(ScopeType::Waveform).is_none());
    }

    #[test]
    fn test_compare_latest_two_returns_diff() {
        let mut store = ScopeSnapshotStore::new(4);
        store.add(make_snapshot(ScopeType::Waveform, 0.0, 0, 256));
        store.add(make_snapshot(ScopeType::Waveform, 1.0, 255, 256));
        let cmp = store
            .compare_latest_two(ScopeType::Waveform)
            .expect("should exist");
        assert!(cmp.diff_score > 0.9);
    }

    // ── history order ────────────────────────────────────────────────────────

    #[test]
    fn test_history_chronological_order() {
        let mut store = ScopeSnapshotStore::new(10);
        for i in 0u8..5 {
            store.add(make_snapshot(ScopeType::CieXy, f64::from(i), i * 10, 64));
        }
        let hist = store.history(ScopeType::CieXy);
        assert_eq!(hist.len(), 5);
        for (i, snap) in hist.iter().enumerate() {
            assert_eq!(snap.data[0], i as u8 * 10, "wrong order at index {i}");
        }
    }

    #[test]
    fn test_history_empty_for_unknown_type() {
        let store = ScopeSnapshotStore::new(4);
        assert!(store.history(ScopeType::Parade).is_empty());
    }

    // ── scope type isolation ─────────────────────────────────────────────────

    #[test]
    fn test_different_scope_types_are_isolated() {
        let mut store = ScopeSnapshotStore::new(4);
        store.add(make_snapshot(ScopeType::Waveform, 0.0, 10, 16));
        store.add(make_snapshot(ScopeType::Histogram, 0.0, 20, 16));
        store.add(make_snapshot(ScopeType::Parade, 0.0, 30, 16));

        assert_eq!(store.history(ScopeType::Waveform).len(), 1);
        assert_eq!(store.history(ScopeType::Histogram).len(), 1);
        assert_eq!(store.history(ScopeType::Parade).len(), 1);
        assert!(store.history(ScopeType::Vectorscope).is_empty());
    }
}
