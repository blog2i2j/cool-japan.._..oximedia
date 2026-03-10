#![allow(dead_code)]
//! PTS/DTS timestamp utilities and repair helpers.
//!
//! Provides `PtsDts` for carrying presentation/decode timestamps,
//! `PtsQueue` for reordering packets, and `PtsDtsRepair` for fixing
//! common timestamp pathologies (negative DTS, wrong ordering).

/// Presentation timestamp (PTS) and decode timestamp (DTS) pair.
///
/// Both values are in the container's native time base (e.g. 90 kHz ticks).
/// Either value may be absent (`None`) when not signalled by the container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtsDts {
    /// Presentation timestamp.
    pub pts: Option<i64>,
    /// Decode timestamp.
    pub dts: Option<i64>,
}

impl PtsDts {
    /// Creates a `PtsDts` with both values present.
    #[must_use]
    pub const fn new(pts: i64, dts: i64) -> Self {
        Self {
            pts: Some(pts),
            dts: Some(dts),
        }
    }

    /// Creates a `PtsDts` with only a PTS (no DTS field in the stream).
    #[must_use]
    pub const fn pts_only(pts: i64) -> Self {
        Self {
            pts: Some(pts),
            dts: None,
        }
    }

    /// Creates an empty `PtsDts` (both absent).
    #[must_use]
    pub const fn none() -> Self {
        Self {
            pts: None,
            dts: None,
        }
    }

    /// Returns `true` when a PTS value is present.
    #[must_use]
    pub fn has_pts(&self) -> bool {
        self.pts.is_some()
    }

    /// Returns `true` when PTS and DTS are both present and equal.
    #[must_use]
    pub fn is_pts_dts_equal(&self) -> bool {
        match (self.pts, self.dts) {
            (Some(p), Some(d)) => p == d,
            _ => false,
        }
    }

    /// Returns the effective decode time: DTS when present, else PTS.
    #[must_use]
    pub fn effective_dts(&self) -> Option<i64> {
        self.dts.or(self.pts)
    }

    /// Returns `true` if the DTS is negative (a common pathology in some muxers).
    #[must_use]
    pub fn has_negative_dts(&self) -> bool {
        self.dts.is_some_and(|d| d < 0)
    }
}

/// A packet entry stored in the reorder queue.
#[derive(Debug, Clone)]
pub struct PtsEntry {
    /// Sequence number of the packet for stable sorting.
    pub seq: u64,
    /// Timestamp pair.
    pub ts: PtsDts,
    /// Arbitrary payload bytes (e.g. compressed frame data).
    pub data: Vec<u8>,
}

/// A small reorder queue that sorts packets by their PTS before delivery.
#[derive(Debug, Default)]
pub struct PtsQueue {
    entries: Vec<PtsEntry>,
    seq_counter: u64,
}

impl PtsQueue {
    /// Creates an empty `PtsQueue`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a packet into the queue.
    pub fn push(&mut self, ts: PtsDts, data: Vec<u8>) {
        let seq = self.seq_counter;
        self.seq_counter += 1;
        self.entries.push(PtsEntry { seq, ts, data });
    }

    /// Returns the number of entries in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns a reference to the entry with the earliest PTS (or first
    /// inserted when PTS is absent).
    #[must_use]
    pub fn earliest(&self) -> Option<&PtsEntry> {
        self.entries.iter().min_by(|a, b| {
            let pa = a.ts.pts.unwrap_or(i64::MAX);
            let pb = b.ts.pts.unwrap_or(i64::MAX);
            pa.cmp(&pb).then(a.seq.cmp(&b.seq))
        })
    }

    /// Removes and returns the entry with the earliest PTS.
    pub fn pop_earliest(&mut self) -> Option<PtsEntry> {
        if self.entries.is_empty() {
            return None;
        }
        let idx = self
            .entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let pa = a.ts.pts.unwrap_or(i64::MAX);
                let pb = b.ts.pts.unwrap_or(i64::MAX);
                pa.cmp(&pb).then(a.seq.cmp(&b.seq))
            })
            .map(|(i, _)| i)?;
        Some(self.entries.remove(idx))
    }

    /// Sorts all entries in the queue by PTS and returns them in order,
    /// draining the queue.
    pub fn reorder(&mut self) -> Vec<PtsEntry> {
        let mut out = std::mem::take(&mut self.entries);
        out.sort_by(|a, b| {
            let pa = a.ts.pts.unwrap_or(i64::MAX);
            let pb = b.ts.pts.unwrap_or(i64::MAX);
            pa.cmp(&pb).then(a.seq.cmp(&b.seq))
        });
        out
    }
}

/// Repairs common PTS/DTS pathologies in a stream of timestamps.
#[derive(Debug, Default)]
pub struct PtsDtsRepair {
    repair_count: u64,
    dts_offset: i64,
}

impl PtsDtsRepair {
    /// Creates a new `PtsDtsRepair` instance.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of timestamps that have been repaired.
    #[must_use]
    pub fn repair_count(&self) -> u64 {
        self.repair_count
    }

    /// Fixes a negative DTS by shifting it to 0 and recording the offset
    /// applied for future packets.
    ///
    /// If the DTS is already non-negative the input is returned unchanged.
    pub fn fix_negative_dts(&mut self, ts: PtsDts) -> PtsDts {
        if let Some(dts) = ts.dts {
            if dts < 0 {
                let shift = -dts;
                self.dts_offset += shift;
                self.repair_count += 1;
                return PtsDts {
                    pts: ts.pts.map(|p| p + shift),
                    dts: Some(0),
                };
            }
        }
        ts
    }

    /// Applies the accumulated DTS offset to a new timestamp pair.
    /// Use this after `fix_negative_dts` to keep subsequent packets aligned.
    #[must_use]
    pub fn apply_offset(&self, ts: PtsDts) -> PtsDts {
        if self.dts_offset == 0 {
            return ts;
        }
        PtsDts {
            pts: ts.pts.map(|p| p + self.dts_offset),
            dts: ts.dts.map(|d| d + self.dts_offset),
        }
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // 1. has_pts – Some
    #[test]
    fn test_has_pts_some() {
        let ts = PtsDts::new(100, 90);
        assert!(ts.has_pts());
    }

    // 2. has_pts – None
    #[test]
    fn test_has_pts_none() {
        let ts = PtsDts::none();
        assert!(!ts.has_pts());
    }

    // 3. is_pts_dts_equal – equal
    #[test]
    fn test_is_pts_dts_equal_true() {
        let ts = PtsDts::new(200, 200);
        assert!(ts.is_pts_dts_equal());
    }

    // 4. is_pts_dts_equal – not equal
    #[test]
    fn test_is_pts_dts_equal_false() {
        let ts = PtsDts::new(200, 180);
        assert!(!ts.is_pts_dts_equal());
    }

    // 5. is_pts_dts_equal – DTS absent
    #[test]
    fn test_is_pts_dts_equal_no_dts() {
        let ts = PtsDts::pts_only(200);
        assert!(!ts.is_pts_dts_equal());
    }

    // 6. effective_dts prefers DTS
    #[test]
    fn test_effective_dts_prefers_dts() {
        let ts = PtsDts::new(200, 180);
        assert_eq!(ts.effective_dts(), Some(180));
    }

    // 7. effective_dts falls back to PTS
    #[test]
    fn test_effective_dts_falls_back_to_pts() {
        let ts = PtsDts::pts_only(200);
        assert_eq!(ts.effective_dts(), Some(200));
    }

    // 8. has_negative_dts
    #[test]
    fn test_has_negative_dts() {
        let ts = PtsDts::new(0, -90);
        assert!(ts.has_negative_dts());
    }

    // 9. PtsQueue empty
    #[test]
    fn test_queue_empty() {
        let q = PtsQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    // 10. push / len
    #[test]
    fn test_queue_push_len() {
        let mut q = PtsQueue::new();
        q.push(PtsDts::new(100, 100), vec![1, 2, 3]);
        assert_eq!(q.len(), 1);
    }

    // 11. earliest returns smallest PTS
    #[test]
    fn test_queue_earliest() {
        let mut q = PtsQueue::new();
        q.push(PtsDts::new(300, 300), vec![]);
        q.push(PtsDts::new(100, 100), vec![]);
        q.push(PtsDts::new(200, 200), vec![]);
        assert_eq!(
            q.earliest().expect("operation should succeed").ts.pts,
            Some(100)
        );
    }

    // 12. reorder drains and sorts
    #[test]
    fn test_queue_reorder() {
        let mut q = PtsQueue::new();
        q.push(PtsDts::new(300, 300), vec![]);
        q.push(PtsDts::new(100, 100), vec![]);
        q.push(PtsDts::new(200, 200), vec![]);
        let sorted = q.reorder();
        let pts_values: Vec<i64> = sorted
            .iter()
            .map(|e| e.ts.pts.expect("operation should succeed"))
            .collect();
        assert_eq!(pts_values, vec![100, 200, 300]);
        assert!(q.is_empty());
    }

    // 13. fix_negative_dts shifts to zero
    #[test]
    fn test_fix_negative_dts() {
        let mut repair = PtsDtsRepair::new();
        let ts = PtsDts::new(0, -180);
        let fixed = repair.fix_negative_dts(ts);
        assert_eq!(fixed.dts, Some(0));
        assert_eq!(repair.repair_count(), 1);
    }

    // 14. fix_negative_dts leaves positive DTS untouched
    #[test]
    fn test_fix_negative_dts_no_op() {
        let mut repair = PtsDtsRepair::new();
        let ts = PtsDts::new(100, 90);
        let fixed = repair.fix_negative_dts(ts);
        assert_eq!(fixed.dts, Some(90));
        assert_eq!(repair.repair_count(), 0);
    }
}
