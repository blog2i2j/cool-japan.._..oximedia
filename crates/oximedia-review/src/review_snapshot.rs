//! Review snapshots: capture frame grabs with annotations baked in.
//!
//! A [`ReviewSnapshot`] associates a specific frame number with a PNG-encoded
//! image and the annotations that were active at that frame.  Snapshots can be
//! exported for sharing or archived as proof of the state of the review at a
//! particular point in time.
//!
//! Image encoding is kept dependency-free: the snapshot stores raw RGBA pixel
//! data plus dimensions; the caller is responsible for encoding to PNG if needed.

use serde::{Deserialize, Serialize};

// ─── ReviewSnapshot ───────────────────────────────────────────────────────────

/// A timestamped frame capture with optional annotation overlay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewSnapshot {
    /// Unique snapshot identifier.
    pub id: u64,
    /// Identifier of the review session this snapshot belongs to.
    pub session_id: String,
    /// Frame number within the media.
    pub frame_number: u64,
    /// Wall-clock time this snapshot was taken (Unix epoch seconds).
    pub captured_at: u64,
    /// Width of the captured frame in pixels.
    pub width: u32,
    /// Height of the captured frame in pixels.
    pub height: u32,
    /// Raw RGBA pixel data (`width * height * 4` bytes).
    /// Empty when the snapshot is a metadata-only record.
    pub pixels: Vec<u8>,
    /// Human-readable note attached to this snapshot.
    pub note: Option<String>,
    /// IDs of annotations that were active at the time of capture.
    pub annotation_ids: Vec<u64>,
}

impl ReviewSnapshot {
    /// Create a new metadata-only snapshot (no pixel data).
    #[must_use]
    pub fn new_meta(
        id: u64,
        session_id: impl Into<String>,
        frame_number: u64,
        captured_at: u64,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            id,
            session_id: session_id.into(),
            frame_number,
            captured_at,
            width,
            height,
            pixels: Vec::new(),
            note: None,
            annotation_ids: Vec::new(),
        }
    }

    /// Attach raw RGBA pixel data (builder pattern).
    #[must_use]
    pub fn with_pixels(mut self, pixels: Vec<u8>) -> Self {
        self.pixels = pixels;
        self
    }

    /// Attach a note to this snapshot (builder pattern).
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// Attach annotation IDs to this snapshot (builder pattern).
    #[must_use]
    pub fn with_annotation_ids(mut self, ids: Vec<u64>) -> Self {
        self.annotation_ids = ids;
        self
    }

    /// Returns `true` if this snapshot contains pixel data.
    #[must_use]
    pub fn has_pixels(&self) -> bool {
        let expected = self.width as usize * self.height as usize * 4;
        !self.pixels.is_empty() && self.pixels.len() >= expected
    }

    /// Sample a single RGBA pixel at `(x, y)`.
    ///
    /// Returns `[0, 0, 0, 0]` for out-of-bounds coordinates or if there is no
    /// pixel data.
    #[must_use]
    pub fn sample_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        if x >= self.width || y >= self.height || self.pixels.is_empty() {
            return [0, 0, 0, 0];
        }
        let idx = ((y * self.width + x) as usize) * 4;
        if idx + 3 >= self.pixels.len() {
            return [0, 0, 0, 0];
        }
        [
            self.pixels[idx],
            self.pixels[idx + 1],
            self.pixels[idx + 2],
            self.pixels[idx + 3],
        ]
    }
}

// ─── SnapshotCollection ───────────────────────────────────────────────────────

/// A collection of snapshots for a review session, with frame-number indexing.
#[derive(Debug, Default, Clone)]
pub struct SnapshotCollection {
    snapshots: Vec<ReviewSnapshot>,
}

impl SnapshotCollection {
    /// Create an empty collection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
        }
    }

    /// Add a snapshot.
    pub fn add(&mut self, snapshot: ReviewSnapshot) {
        self.snapshots.push(snapshot);
    }

    /// Find snapshots taken at a specific frame number.
    #[must_use]
    pub fn at_frame(&self, frame_number: u64) -> Vec<&ReviewSnapshot> {
        self.snapshots
            .iter()
            .filter(|s| s.frame_number == frame_number)
            .collect()
    }

    /// Get the snapshot with the given ID.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&ReviewSnapshot> {
        self.snapshots.iter().find(|s| s.id == id)
    }

    /// Total number of snapshots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns `true` if the collection is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Snapshots sorted by frame number (ascending).
    #[must_use]
    pub fn sorted_by_frame(&self) -> Vec<&ReviewSnapshot> {
        let mut sorted: Vec<&ReviewSnapshot> = self.snapshots.iter().collect();
        sorted.sort_by_key(|s| s.frame_number);
        sorted
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(id: u64, frame: u64) -> ReviewSnapshot {
        ReviewSnapshot::new_meta(id, "session-1", frame, 1_700_000_000 + id, 16, 16)
    }

    #[test]
    fn snapshot_no_pixels_by_default() {
        let s = make_snapshot(1, 100);
        assert!(!s.has_pixels());
    }

    #[test]
    fn snapshot_with_pixels() {
        let pixels = vec![128u8; 16 * 16 * 4];
        let s = make_snapshot(1, 100).with_pixels(pixels);
        assert!(s.has_pixels());
    }

    #[test]
    fn snapshot_sample_pixel_oob() {
        let s = make_snapshot(1, 100);
        assert_eq!(s.sample_pixel(0, 0), [0, 0, 0, 0]);
    }

    #[test]
    fn snapshot_sample_pixel_in_bounds() {
        let mut pixels = vec![0u8; 16 * 16 * 4];
        // Set pixel (2, 3) to (100, 150, 200, 255)
        let idx = (3 * 16 + 2) * 4;
        pixels[idx] = 100;
        pixels[idx + 1] = 150;
        pixels[idx + 2] = 200;
        pixels[idx + 3] = 255;
        let s = make_snapshot(1, 100).with_pixels(pixels);
        assert_eq!(s.sample_pixel(2, 3), [100, 150, 200, 255]);
    }

    #[test]
    fn snapshot_with_note_and_annotations() {
        let s = make_snapshot(1, 50)
            .with_note("colour grade issue")
            .with_annotation_ids(vec![10, 20, 30]);
        assert_eq!(s.note.as_deref(), Some("colour grade issue"));
        assert_eq!(s.annotation_ids.len(), 3);
    }

    #[test]
    fn collection_add_and_len() {
        let mut col = SnapshotCollection::new();
        assert!(col.is_empty());
        col.add(make_snapshot(1, 100));
        col.add(make_snapshot(2, 200));
        assert_eq!(col.len(), 2);
    }

    #[test]
    fn collection_at_frame() {
        let mut col = SnapshotCollection::new();
        col.add(make_snapshot(1, 100));
        col.add(make_snapshot(2, 100)); // same frame
        col.add(make_snapshot(3, 200));
        let at_100 = col.at_frame(100);
        assert_eq!(at_100.len(), 2);
        assert!(col.at_frame(999).is_empty());
    }

    #[test]
    fn collection_sorted_by_frame() {
        let mut col = SnapshotCollection::new();
        col.add(make_snapshot(1, 300));
        col.add(make_snapshot(2, 100));
        col.add(make_snapshot(3, 200));
        let sorted = col.sorted_by_frame();
        assert_eq!(sorted[0].frame_number, 100);
        assert_eq!(sorted[1].frame_number, 200);
        assert_eq!(sorted[2].frame_number, 300);
    }
}
