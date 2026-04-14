//! Dirty-region tracking for incremental overlay re-rendering.
//!
//! In broadcast graphics engines, re-rasterizing the entire frame every tick is
//! wasteful when only a small portion of the overlay has changed (e.g., a clock
//! digit flipping, a score updating). This module provides a
//! [`DirtyRegionTracker`] that accumulates axis-aligned bounding rectangles
//! marking which regions of the canvas need to be repainted.
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_graphics::dirty_region::{DirtyRegionTracker, DirtyRect};
//!
//! let mut tracker = DirtyRegionTracker::new(1920, 1080);
//! tracker.mark(DirtyRect::new(100, 50, 200, 80));  // clock area
//! tracker.mark(DirtyRect::new(800, 900, 400, 60)); // ticker strip
//!
//! for region in tracker.regions() {
//!     // re-render only this region
//!     let _ = region;
//! }
//!
//! tracker.clear(); // reset for next frame
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// DirtyRect
// ---------------------------------------------------------------------------

/// An axis-aligned rectangle describing a dirty region of the canvas.
///
/// All coordinates are in integer pixel space.  The rectangle is described by
/// its top-left corner `(x, y)` and its `width × height` extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyRect {
    /// Left edge (inclusive), in pixels.
    pub x: u32,
    /// Top edge (inclusive), in pixels.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl DirtyRect {
    /// Create a new dirty rectangle.
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Right edge (exclusive), i.e. `x + width`.
    #[must_use]
    pub fn right(&self) -> u32 {
        self.x.saturating_add(self.width)
    }

    /// Bottom edge (exclusive), i.e. `y + height`.
    #[must_use]
    pub fn bottom(&self) -> u32 {
        self.y.saturating_add(self.height)
    }

    /// Area in pixels.
    #[must_use]
    pub fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Returns `true` if this rectangle has zero area.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Returns `true` if `other` overlaps this rectangle.
    #[must_use]
    pub fn overlaps(&self, other: &DirtyRect) -> bool {
        self.x < other.right()
            && other.x < self.right()
            && self.y < other.bottom()
            && other.y < self.bottom()
    }

    /// Returns `true` if `other` is fully contained inside this rectangle.
    #[must_use]
    pub fn contains(&self, other: &DirtyRect) -> bool {
        other.x >= self.x
            && other.y >= self.y
            && other.right() <= self.right()
            && other.bottom() <= self.bottom()
    }

    /// Compute the smallest rectangle that encloses both `self` and `other`.
    ///
    /// Either operand may be empty; an empty rectangle is treated as a neutral
    /// element (so the union of an empty rect with any rect is the other rect).
    #[must_use]
    pub fn union(&self, other: &DirtyRect) -> DirtyRect {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let r = self.right().max(other.right());
        let b = self.bottom().max(other.bottom());
        DirtyRect::new(x, y, r.saturating_sub(x), b.saturating_sub(y))
    }

    /// Compute the intersection of `self` and `other`, or `None` if they do not
    /// overlap.
    #[must_use]
    pub fn intersection(&self, other: &DirtyRect) -> Option<DirtyRect> {
        if !self.overlaps(other) {
            return None;
        }
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let r = self.right().min(other.right());
        let b = self.bottom().min(other.bottom());
        if r <= x || b <= y {
            return None;
        }
        Some(DirtyRect::new(x, y, r - x, b - y))
    }

    /// Clamp this rectangle so that it lies entirely within a canvas of the
    /// given `width × height`.  Returns `None` if the result is empty.
    #[must_use]
    pub fn clamp_to_canvas(&self, canvas_width: u32, canvas_height: u32) -> Option<DirtyRect> {
        let canvas = DirtyRect::new(0, 0, canvas_width, canvas_height);
        self.intersection(&canvas)
    }
}

impl fmt::Display for DirtyRect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DirtyRect(x={}, y={}, {}x{})",
            self.x, self.y, self.width, self.height
        )
    }
}

// ---------------------------------------------------------------------------
// MergePolicy
// ---------------------------------------------------------------------------

/// Merge policy controlling when dirty rectangles are combined.
///
/// Combining many small rects into fewer large rects reduces the number of
/// re-render passes at the cost of re-rasterizing some clean pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergePolicy {
    /// Never merge; keep every dirty rect independent.
    ///
    /// Best when dirty regions are small, numerous, and non-overlapping.
    None,
    /// Merge any two rectangles that overlap or are immediately adjacent.
    ///
    /// This is a conservative strategy that prevents duplicate work.
    Overlapping,
    /// Merge any two rectangles whose union has an area no more than
    /// `wastage_limit_pct` percent larger than their combined individual areas.
    ///
    /// `wastage_limit_pct` is expressed as a value in `[0, 100]`; a value of 50
    /// means the union may waste up to 50 % extra pixels.
    Wastage(u8),
}

impl Default for MergePolicy {
    fn default() -> Self {
        Self::Overlapping
    }
}

// ---------------------------------------------------------------------------
// DirtyRegionTracker
// ---------------------------------------------------------------------------

/// Tracks which rectangular regions of a canvas need to be re-rendered.
///
/// On each animation frame:
/// 1. Call [`mark`](DirtyRegionTracker::mark) for every widget or element that
///    has changed.
/// 2. Iterate [`regions`](DirtyRegionTracker::regions) to obtain the set of
///    areas to re-rasterize.
/// 3. Call [`clear`](DirtyRegionTracker::clear) to reset for the next frame.
pub struct DirtyRegionTracker {
    canvas_width: u32,
    canvas_height: u32,
    regions: Vec<DirtyRect>,
    policy: MergePolicy,
    /// Maximum number of independent dirty rects before they are collapsed into
    /// a single full-canvas dirty rect (a safety valve for pathological cases).
    max_regions: usize,
}

impl DirtyRegionTracker {
    /// Create a tracker for a canvas of the given dimensions.
    ///
    /// Uses the default [`MergePolicy::Overlapping`] and a safety cap of 64
    /// independent dirty regions.
    #[must_use]
    pub fn new(canvas_width: u32, canvas_height: u32) -> Self {
        Self {
            canvas_width,
            canvas_height,
            regions: Vec::new(),
            policy: MergePolicy::default(),
            max_regions: 64,
        }
    }

    /// Create a tracker with an explicit merge policy and region cap.
    #[must_use]
    pub fn with_policy(
        canvas_width: u32,
        canvas_height: u32,
        policy: MergePolicy,
        max_regions: usize,
    ) -> Self {
        Self {
            canvas_width,
            canvas_height,
            regions: Vec::new(),
            policy,
            max_regions: max_regions.max(1),
        }
    }

    /// Return the canvas width.
    #[must_use]
    pub fn canvas_width(&self) -> u32 {
        self.canvas_width
    }

    /// Return the canvas height.
    #[must_use]
    pub fn canvas_height(&self) -> u32 {
        self.canvas_height
    }

    /// Return the active merge policy.
    #[must_use]
    pub fn policy(&self) -> MergePolicy {
        self.policy
    }

    /// Mark a rectangular region as dirty.
    ///
    /// The rectangle is clamped to the canvas bounds.  Empty or out-of-bounds
    /// rects are silently ignored.
    pub fn mark(&mut self, rect: DirtyRect) {
        let Some(clamped) = rect.clamp_to_canvas(self.canvas_width, self.canvas_height) else {
            return;
        };
        if clamped.is_empty() {
            return;
        }
        self.regions.push(clamped);
        self.maybe_merge();

        // Safety valve: collapse everything into a single full-canvas dirty.
        if self.regions.len() > self.max_regions {
            self.mark_all();
        }
    }

    /// Mark the entire canvas as dirty.  All previous regions are replaced by a
    /// single full-canvas rectangle.
    pub fn mark_all(&mut self) {
        self.regions.clear();
        self.regions
            .push(DirtyRect::new(0, 0, self.canvas_width, self.canvas_height));
    }

    /// Returns `true` if any region is currently dirty.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        !self.regions.is_empty()
    }

    /// Returns `true` if the entire canvas is marked dirty.
    #[must_use]
    pub fn is_full_canvas_dirty(&self) -> bool {
        if self.regions.len() != 1 {
            return false;
        }
        let r = &self.regions[0];
        r.x == 0 && r.y == 0 && r.width == self.canvas_width && r.height == self.canvas_height
    }

    /// Return a slice of the current dirty regions.
    #[must_use]
    pub fn regions(&self) -> &[DirtyRect] {
        &self.regions
    }

    /// Return the total number of distinct dirty regions.
    #[must_use]
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Compute the combined area of all dirty regions in pixels.
    ///
    /// Note: if regions overlap this may double-count some pixels.
    #[must_use]
    pub fn dirty_pixel_count(&self) -> u64 {
        self.regions.iter().map(|r| r.area()).sum()
    }

    /// Fraction of the canvas (0.0–1.0) covered by dirty regions.
    ///
    /// Computed as `dirty_pixel_count / (canvas_width * canvas_height)`.
    /// Values may exceed 1.0 when regions overlap.
    #[must_use]
    pub fn dirty_fraction(&self) -> f32 {
        let total = self.canvas_width as u64 * self.canvas_height as u64;
        if total == 0 {
            return 0.0;
        }
        self.dirty_pixel_count() as f32 / total as f32
    }

    /// Clear all dirty regions (call after each frame has been re-rendered).
    pub fn clear(&mut self) {
        self.regions.clear();
    }

    // -----------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------

    /// Apply the configured merge policy to the region list.
    fn maybe_merge(&mut self) {
        match self.policy {
            MergePolicy::None => {}
            MergePolicy::Overlapping => self.merge_overlapping(),
            MergePolicy::Wastage(limit) => self.merge_by_wastage(limit),
        }
    }

    /// Iteratively merge any pair of overlapping or adjacent rects.
    fn merge_overlapping(&mut self) {
        loop {
            let mut merged = false;
            let mut i = 0;
            while i < self.regions.len() {
                let mut j = i + 1;
                while j < self.regions.len() {
                    let a = self.regions[i];
                    let b = self.regions[j];
                    if a.overlaps(&b) || rects_adjacent(&a, &b) {
                        self.regions[i] = a.union(&b);
                        self.regions.remove(j);
                        merged = true;
                    } else {
                        j += 1;
                    }
                }
                i += 1;
            }
            if !merged {
                break;
            }
        }
    }

    /// Merge pairs whose union waste is within `limit` percent.
    fn merge_by_wastage(&mut self, limit: u8) {
        let limit_frac = limit.min(100) as f64 / 100.0;
        loop {
            let mut merged = false;
            let mut i = 0;
            while i < self.regions.len() {
                let mut j = i + 1;
                while j < self.regions.len() {
                    let a = self.regions[i];
                    let b = self.regions[j];
                    let combined = a.area() + b.area();
                    let union_area = a.union(&b).area();
                    // If union wastes <= limit% extra, merge.
                    if combined == 0
                        || (union_area as f64 / combined as f64 - 1.0) <= limit_frac
                    {
                        self.regions[i] = a.union(&b);
                        self.regions.remove(j);
                        merged = true;
                    } else {
                        j += 1;
                    }
                }
                i += 1;
            }
            if !merged {
                break;
            }
        }
    }
}

/// Returns `true` if two rects share an edge (are immediately adjacent but not
/// overlapping).  Adjacent rects can be merged without adding waste.
fn rects_adjacent(a: &DirtyRect, b: &DirtyRect) -> bool {
    // Horizontally adjacent (share a vertical edge)
    let h_adj = (a.right() == b.x || b.right() == a.x)
        && !(a.y >= b.bottom() || b.y >= a.bottom());
    // Vertically adjacent (share a horizontal edge)
    let v_adj = (a.bottom() == b.y || b.bottom() == a.y)
        && !(a.x >= b.right() || b.x >= a.right());
    h_adj || v_adj
}

// ---------------------------------------------------------------------------
// FrameDiff — convenience wrapper
// ---------------------------------------------------------------------------

/// Aggregates dirty regions for a complete frame lifecycle:
/// mark → query → clear.
///
/// Provides a lightweight wrapper that keeps a running count of total
/// frames processed and dirty-pixels saved.
pub struct FrameDiff {
    tracker: DirtyRegionTracker,
    frames_processed: u64,
    total_dirty_pixels: u64,
}

impl FrameDiff {
    /// Create a frame-diff tracker for the given canvas dimensions.
    #[must_use]
    pub fn new(canvas_width: u32, canvas_height: u32) -> Self {
        Self {
            tracker: DirtyRegionTracker::new(canvas_width, canvas_height),
            frames_processed: 0,
            total_dirty_pixels: 0,
        }
    }

    /// Mark a region as dirty (delegates to the inner tracker).
    pub fn mark(&mut self, rect: DirtyRect) {
        self.tracker.mark(rect);
    }

    /// Mark the entire canvas as dirty.
    pub fn mark_all(&mut self) {
        self.tracker.mark_all();
    }

    /// Finalize the current frame, accumulate statistics, and clear for the
    /// next frame.  Returns the set of dirty regions for this frame.
    pub fn commit_frame(&mut self) -> Vec<DirtyRect> {
        let regions = self.tracker.regions().to_vec();
        self.total_dirty_pixels += self.tracker.dirty_pixel_count();
        self.frames_processed += 1;
        self.tracker.clear();
        regions
    }

    /// Number of frames committed so far.
    #[must_use]
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }

    /// Total dirty pixels accumulated across all committed frames.
    #[must_use]
    pub fn total_dirty_pixels(&self) -> u64 {
        self.total_dirty_pixels
    }

    /// Average dirty pixels per frame.
    #[must_use]
    pub fn avg_dirty_pixels_per_frame(&self) -> f64 {
        if self.frames_processed == 0 {
            return 0.0;
        }
        self.total_dirty_pixels as f64 / self.frames_processed as f64
    }

    /// Returns `true` if any dirty regions are pending for the current frame.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.tracker.is_dirty()
    }

    /// Access the inner tracker (read-only).
    #[must_use]
    pub fn tracker(&self) -> &DirtyRegionTracker {
        &self.tracker
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirty_rect_right_and_bottom() {
        let r = DirtyRect::new(10, 20, 100, 50);
        assert_eq!(r.right(), 110);
        assert_eq!(r.bottom(), 70);
    }

    #[test]
    fn test_dirty_rect_area_and_is_empty() {
        let r = DirtyRect::new(0, 0, 100, 50);
        assert_eq!(r.area(), 5000);
        assert!(!r.is_empty());
        let empty = DirtyRect::new(0, 0, 0, 10);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_dirty_rect_overlaps() {
        let a = DirtyRect::new(0, 0, 100, 100);
        let b = DirtyRect::new(50, 50, 100, 100);
        assert!(a.overlaps(&b));
        let c = DirtyRect::new(200, 200, 50, 50);
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_dirty_rect_contains() {
        let outer = DirtyRect::new(0, 0, 200, 200);
        let inner = DirtyRect::new(10, 10, 50, 50);
        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
    }

    #[test]
    fn test_dirty_rect_union() {
        let a = DirtyRect::new(0, 0, 50, 50);
        let b = DirtyRect::new(100, 100, 50, 50);
        let u = a.union(&b);
        assert_eq!(u.x, 0);
        assert_eq!(u.y, 0);
        assert_eq!(u.right(), 150);
        assert_eq!(u.bottom(), 150);
    }

    #[test]
    fn test_dirty_rect_union_with_empty() {
        let a = DirtyRect::new(10, 20, 100, 50);
        let empty = DirtyRect::new(0, 0, 0, 0);
        assert_eq!(a.union(&empty), a);
        assert_eq!(empty.union(&a), a);
    }

    #[test]
    fn test_dirty_rect_intersection_overlap() {
        let a = DirtyRect::new(0, 0, 100, 100);
        let b = DirtyRect::new(50, 50, 100, 100);
        let i = a.intersection(&b).expect("should intersect");
        assert_eq!(i.x, 50);
        assert_eq!(i.y, 50);
        assert_eq!(i.width, 50);
        assert_eq!(i.height, 50);
    }

    #[test]
    fn test_dirty_rect_intersection_none() {
        let a = DirtyRect::new(0, 0, 50, 50);
        let b = DirtyRect::new(100, 100, 50, 50);
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_dirty_rect_clamp_to_canvas() {
        let r = DirtyRect::new(1800, 1000, 300, 300);
        let clamped = r
            .clamp_to_canvas(1920, 1080)
            .expect("should be partially visible");
        assert_eq!(clamped.right(), 1920);
        assert_eq!(clamped.bottom(), 1080);
    }

    #[test]
    fn test_dirty_rect_clamp_out_of_bounds() {
        let r = DirtyRect::new(2000, 2000, 100, 100);
        assert!(r.clamp_to_canvas(1920, 1080).is_none());
    }

    #[test]
    fn test_tracker_mark_and_regions() {
        let mut tracker = DirtyRegionTracker::new(1920, 1080);
        assert!(!tracker.is_dirty());
        tracker.mark(DirtyRect::new(100, 100, 200, 50));
        assert!(tracker.is_dirty());
        assert_eq!(tracker.region_count(), 1);
    }

    #[test]
    fn test_tracker_mark_clamped() {
        let mut tracker = DirtyRegionTracker::new(1920, 1080);
        // Mark a rect that extends beyond the canvas edge.
        tracker.mark(DirtyRect::new(1800, 0, 300, 1080));
        let r = &tracker.regions()[0];
        assert_eq!(r.right(), 1920);
    }

    #[test]
    fn test_tracker_mark_out_of_bounds_ignored() {
        let mut tracker = DirtyRegionTracker::new(1920, 1080);
        tracker.mark(DirtyRect::new(2000, 2000, 100, 100));
        assert!(!tracker.is_dirty());
    }

    #[test]
    fn test_tracker_clear() {
        let mut tracker = DirtyRegionTracker::new(1920, 1080);
        tracker.mark(DirtyRect::new(0, 0, 100, 100));
        tracker.clear();
        assert!(!tracker.is_dirty());
        assert_eq!(tracker.region_count(), 0);
    }

    #[test]
    fn test_tracker_mark_all() {
        let mut tracker = DirtyRegionTracker::new(1920, 1080);
        tracker.mark(DirtyRect::new(0, 0, 50, 50));
        tracker.mark_all();
        assert!(tracker.is_full_canvas_dirty());
        assert_eq!(tracker.region_count(), 1);
    }

    #[test]
    fn test_tracker_merge_overlapping_reduces_count() {
        let mut tracker =
            DirtyRegionTracker::with_policy(1920, 1080, MergePolicy::Overlapping, 64);
        tracker.mark(DirtyRect::new(0, 0, 100, 100));
        tracker.mark(DirtyRect::new(50, 50, 100, 100)); // overlaps
        // After merge the two rects should collapse to one.
        assert_eq!(tracker.region_count(), 1);
    }

    #[test]
    fn test_tracker_merge_none_keeps_separate() {
        let mut tracker = DirtyRegionTracker::with_policy(1920, 1080, MergePolicy::None, 64);
        tracker.mark(DirtyRect::new(0, 0, 100, 100));
        tracker.mark(DirtyRect::new(50, 50, 100, 100));
        // With MergePolicy::None, both rects are kept separate.
        assert_eq!(tracker.region_count(), 2);
    }

    #[test]
    fn test_tracker_dirty_pixel_count() {
        let mut tracker = DirtyRegionTracker::with_policy(1920, 1080, MergePolicy::None, 64);
        tracker.mark(DirtyRect::new(0, 0, 100, 100));
        tracker.mark(DirtyRect::new(200, 200, 50, 50));
        // No merging → counts are additive.
        assert_eq!(tracker.dirty_pixel_count(), 100 * 100 + 50 * 50);
    }

    #[test]
    fn test_tracker_dirty_fraction() {
        let mut tracker = DirtyRegionTracker::with_policy(100, 100, MergePolicy::None, 64);
        tracker.mark(DirtyRect::new(0, 0, 50, 100)); // half the canvas
        let frac = tracker.dirty_fraction();
        assert!((frac - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_tracker_safety_valve_collapses_to_full_canvas() {
        let mut tracker = DirtyRegionTracker::with_policy(1920, 1080, MergePolicy::None, 2);
        // Marking more rects than max_regions triggers collapse.
        tracker.mark(DirtyRect::new(0, 0, 10, 10));
        tracker.mark(DirtyRect::new(100, 0, 10, 10));
        tracker.mark(DirtyRect::new(200, 0, 10, 10)); // ← exceeds cap → collapse
        assert!(tracker.is_full_canvas_dirty());
    }

    #[test]
    fn test_frame_diff_lifecycle() {
        let mut diff = FrameDiff::new(1920, 1080);
        diff.mark(DirtyRect::new(0, 0, 100, 100));
        diff.mark(DirtyRect::new(500, 500, 200, 200));
        assert!(diff.is_dirty());
        let regions = diff.commit_frame();
        assert!(!diff.is_dirty());
        assert_eq!(diff.frames_processed(), 1);
        // At least one region was returned (merging may reduce count).
        assert!(!regions.is_empty());
    }

    #[test]
    fn test_frame_diff_avg_dirty_pixels() {
        let mut diff = FrameDiff::new(1000, 1000);
        diff.mark(DirtyRect::new(0, 0, 100, 100)); // 10_000 px
        diff.commit_frame();
        let avg = diff.avg_dirty_pixels_per_frame();
        assert!((avg - 10_000.0).abs() < 1.0);
    }

    #[test]
    fn test_merge_policy_wastage() {
        let mut tracker = DirtyRegionTracker::with_policy(1920, 1080, MergePolicy::Wastage(10), 64);
        // Two small adjacent rects — union waste is low → should merge.
        tracker.mark(DirtyRect::new(0, 0, 10, 100));
        tracker.mark(DirtyRect::new(10, 0, 10, 100)); // adjacent, no waste
        assert_eq!(tracker.region_count(), 1);
    }

    #[test]
    fn test_rects_adjacent_horizontal() {
        let a = DirtyRect::new(0, 0, 50, 100);
        let b = DirtyRect::new(50, 0, 50, 100); // immediately to the right
        assert!(rects_adjacent(&a, &b));
    }

    #[test]
    fn test_rects_adjacent_vertical() {
        let a = DirtyRect::new(0, 0, 100, 50);
        let b = DirtyRect::new(0, 50, 100, 50); // immediately below
        assert!(rects_adjacent(&a, &b));
    }

    #[test]
    fn test_rects_not_adjacent() {
        let a = DirtyRect::new(0, 0, 50, 50);
        let b = DirtyRect::new(100, 100, 50, 50); // diagonal gap
        assert!(!rects_adjacent(&a, &b));
    }

    #[test]
    fn test_dirty_rect_display() {
        let r = DirtyRect::new(10, 20, 100, 50);
        let s = format!("{r}");
        assert!(s.contains("10"));
        assert!(s.contains("20"));
    }
}
