//! Caption and subtitle collision detection and resolution.
//!
//! Identifies spatially and temporally overlapping caption boxes, then resolves
//! them using one of several configurable strategies (priority eviction, vertical
//! shift, or shrink-to-fit).

use serde::{Deserialize, Serialize};

// ── Core types ────────────────────────────────────────────────────────────────

/// An axis-aligned rectangular region occupied by a caption during a time range.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptionBox {
    /// Unique identifier for this caption element.
    pub id: String,
    /// Left edge in pixels (origin at top-left of frame).
    pub x: u32,
    /// Top edge in pixels.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Display start time in milliseconds.
    pub start_ms: u64,
    /// Display end time in milliseconds.
    pub end_ms: u64,
    /// Higher value = higher priority (survives eviction).
    pub priority: u8,
}

impl CaptionBox {
    /// Right edge (exclusive) in pixels.
    #[must_use]
    pub fn right(&self) -> u32 {
        self.x.saturating_add(self.width)
    }

    /// Bottom edge (exclusive) in pixels.
    #[must_use]
    pub fn bottom(&self) -> u32 {
        self.y.saturating_add(self.height)
    }

    /// Compute the spatial intersection area with another `CaptionBox`.
    /// Returns `0` when the boxes do not overlap spatially.
    #[must_use]
    pub fn spatial_overlap_area(&self, other: &Self) -> u32 {
        let x_overlap = overlap_1d(self.x, self.right(), other.x, other.right());
        let y_overlap = overlap_1d(self.y, self.bottom(), other.y, other.bottom());
        x_overlap.saturating_mul(y_overlap)
    }

    /// Compute temporal overlap duration in milliseconds.
    /// Returns `0` when the boxes are not displayed at the same time.
    #[must_use]
    pub fn temporal_overlap_ms(&self, other: &Self) -> u64 {
        overlap_1d_u64(self.start_ms, self.end_ms, other.start_ms, other.end_ms)
    }

    /// Returns `true` when this box collides with `other` both spatially and temporally.
    #[must_use]
    pub fn collides_with(&self, other: &Self) -> bool {
        self.spatial_overlap_area(other) > 0 && self.temporal_overlap_ms(other) > 0
    }
}

// ── Collision pair ────────────────────────────────────────────────────────────

/// A detected collision between two caption boxes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollisionPair {
    /// ID of the first colliding box.
    pub box_a: String,
    /// ID of the second colliding box.
    pub box_b: String,
    /// Pixels of spatial overlap area.
    pub overlap_area: u32,
    /// Milliseconds of temporal overlap.
    pub time_overlap_ms: u64,
}

// ── Resolution strategy ───────────────────────────────────────────────────────

/// Strategy used to resolve detected collisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolveStrategy {
    /// Remove the lower-priority box of each colliding pair.
    /// When priorities are equal, the box with the higher index is removed.
    PriorityEvict,
    /// Move the lower-priority box downward so it no longer overlaps the other.
    VerticalShift,
    /// Shrink the lower-priority box so its area fits beside / below the higher-priority one.
    ShrinkToFit,
}

// ── CollisionResolver ─────────────────────────────────────────────────────────

/// Detects and resolves collisions among a set of caption boxes.
#[derive(Debug, Clone, Default)]
pub struct CollisionResolver;

impl CollisionResolver {
    /// Create a new [`CollisionResolver`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Detect all pairs of caption boxes that collide both spatially and temporally.
    ///
    /// Runs in O(n²); suitable for typical caption counts (dozens, not millions).
    #[must_use]
    pub fn detect_collisions(&self, captions: &[CaptionBox]) -> Vec<CollisionPair> {
        let mut pairs = Vec::new();

        for i in 0..captions.len() {
            for j in (i + 1)..captions.len() {
                let a = &captions[i];
                let b = &captions[j];

                let area = a.spatial_overlap_area(b);
                let time = a.temporal_overlap_ms(b);

                if area > 0 && time > 0 {
                    pairs.push(CollisionPair {
                        box_a: a.id.clone(),
                        box_b: b.id.clone(),
                        overlap_area: area,
                        time_overlap_ms: time,
                    });
                }
            }
        }

        pairs
    }

    /// Resolve collisions in `captions` using the given `strategy`.
    ///
    /// The operation modifies `captions` in-place.  After resolution, a second
    /// pass of [`detect_collisions`](Self::detect_collisions) should return an
    /// empty list (barring extreme edge cases with `ShrinkToFit`).
    pub fn resolve(&self, captions: &mut Vec<CaptionBox>, strategy: ResolveStrategy) {
        match strategy {
            ResolveStrategy::PriorityEvict => self.resolve_priority_evict(captions),
            ResolveStrategy::VerticalShift => self.resolve_vertical_shift(captions),
            ResolveStrategy::ShrinkToFit => self.resolve_shrink_to_fit(captions),
        }
    }

    // ── Strategy implementations ──────────────────────────────────────────────

    fn resolve_priority_evict(&self, captions: &mut Vec<CaptionBox>) {
        // Collect indices to remove (lower priority loses; tie → higher index loses)
        let mut to_remove: std::collections::HashSet<usize> = std::collections::HashSet::new();

        let n = captions.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if to_remove.contains(&i) || to_remove.contains(&j) {
                    continue;
                }
                if captions[i].collides_with(&captions[j]) {
                    // Lower priority is evicted; on tie, evict j (higher index)
                    if captions[i].priority >= captions[j].priority {
                        to_remove.insert(j);
                    } else {
                        to_remove.insert(i);
                    }
                }
            }
        }

        // Remove in reverse index order to keep indices stable
        let mut remove_vec: Vec<usize> = to_remove.into_iter().collect();
        remove_vec.sort_unstable_by(|a, b| b.cmp(a));
        for idx in remove_vec {
            captions.remove(idx);
        }
    }

    fn resolve_vertical_shift(&self, captions: &mut Vec<CaptionBox>) {
        // Sort: higher priority first, then by original y (top first)
        captions.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then(a.y.cmp(&b.y))
        });

        let n = captions.len();
        for i in 0..n {
            for j in (i + 1)..n {
                // Check collision fresh after possible earlier shifts
                if captions[i].collides_with(&captions[j]) {
                    // Shift j down so its top aligns with the bottom of i
                    let new_y = captions[i].bottom();
                    captions[j].y = new_y;
                }
            }
        }
    }

    fn resolve_shrink_to_fit(&self, captions: &mut Vec<CaptionBox>) {
        // Sort: higher priority first
        captions.sort_by(|a, b| b.priority.cmp(&a.priority));

        let n = captions.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if captions[i].collides_with(&captions[j]) {
                    let available_y = captions[i].bottom();
                    // Amount of vertical overlap
                    let overlap_y = available_y.saturating_sub(captions[j].y);
                    if overlap_y > 0 && captions[j].height > overlap_y {
                        captions[j].y = available_y;
                        captions[j].height = captions[j].height.saturating_sub(overlap_y);
                    } else {
                        // Cannot shrink meaningfully — zero height signals removal
                        captions[j].height = 0;
                    }
                }
            }
        }

        // Drop zero-height boxes produced by shrinking
        captions.retain(|b| b.height > 0);
    }
}

// ── Geometry helpers ──────────────────────────────────────────────────────────

/// Length of the intersection of two integer intervals [a0, a1) and [b0, b1).
/// Returns 0 when they do not overlap.
#[inline]
#[must_use]
fn overlap_1d(a0: u32, a1: u32, b0: u32, b1: u32) -> u32 {
    let start = a0.max(b0);
    let end = a1.min(b1);
    end.saturating_sub(start)
}

/// Length of the intersection of two u64 intervals [a0, a1) and [b0, b1).
#[inline]
#[must_use]
fn overlap_1d_u64(a0: u64, a1: u64, b0: u64, b1: u64) -> u64 {
    let start = a0.max(b0);
    let end = a1.min(b1);
    end.saturating_sub(start)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_box(id: &str, x: u32, y: u32, w: u32, h: u32, t0: u64, t1: u64, prio: u8) -> CaptionBox {
        CaptionBox {
            id: id.to_string(),
            x,
            y,
            width: w,
            height: h,
            start_ms: t0,
            end_ms: t1,
            priority: prio,
        }
    }

    // 1. No captions → no collisions
    #[test]
    fn test_no_captions_no_collisions() {
        let resolver = CollisionResolver::new();
        let pairs = resolver.detect_collisions(&[]);
        assert!(pairs.is_empty());
    }

    // 2. Non-overlapping boxes → no collisions
    #[test]
    fn test_non_overlapping_no_collisions() {
        let resolver = CollisionResolver::new();
        let captions = vec![
            make_box("a", 0, 0, 100, 30, 0, 1000, 1),
            make_box("b", 200, 0, 100, 30, 0, 1000, 1),
        ];
        assert!(resolver.detect_collisions(&captions).is_empty());
    }

    // 3. Spatial overlap but no temporal overlap → not a collision
    #[test]
    fn test_spatial_overlap_no_temporal_not_collision() {
        let resolver = CollisionResolver::new();
        // Both at x=0,y=0 100×30 but non-overlapping time windows
        let captions = vec![
            make_box("a", 0, 0, 100, 30, 0, 500, 1),
            make_box("b", 0, 0, 100, 30, 500, 1000, 1),
        ];
        assert!(resolver.detect_collisions(&captions).is_empty());
    }

    // 4. Temporal overlap but no spatial overlap → not a collision
    #[test]
    fn test_temporal_overlap_no_spatial_not_collision() {
        let resolver = CollisionResolver::new();
        let captions = vec![
            make_box("a", 0, 0, 100, 30, 0, 2000, 1),
            make_box("b", 200, 100, 100, 30, 0, 2000, 1),
        ];
        assert!(resolver.detect_collisions(&captions).is_empty());
    }

    // 5. Both spatial and temporal overlap → collision detected
    #[test]
    fn test_full_collision_detected() {
        let resolver = CollisionResolver::new();
        let captions = vec![
            make_box("a", 0, 0, 200, 50, 0, 2000, 1),
            make_box("b", 100, 25, 200, 50, 1000, 3000, 2),
        ];
        let pairs = resolver.detect_collisions(&captions);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].box_a, "a");
        assert_eq!(pairs[0].box_b, "b");
        assert!(pairs[0].overlap_area > 0);
        assert!(pairs[0].time_overlap_ms > 0);
    }

    // 6. Priority eviction removes lower-priority box
    #[test]
    fn test_priority_evict_removes_lower_priority() {
        let resolver = CollisionResolver::new();
        let mut captions = vec![
            make_box("high", 0, 0, 200, 50, 0, 2000, 5),
            make_box("low", 0, 0, 200, 50, 0, 2000, 1),
        ];
        resolver.resolve(&mut captions, ResolveStrategy::PriorityEvict);
        assert_eq!(captions.len(), 1);
        assert_eq!(captions[0].id, "high");
    }

    // 7. Vertical shift moves lower box down
    #[test]
    fn test_vertical_shift_resolves_collision() {
        let resolver = CollisionResolver::new();
        let mut captions = vec![
            make_box("top", 0, 0, 400, 50, 0, 2000, 5),
            make_box("bottom", 0, 30, 400, 50, 0, 2000, 1),
        ];
        resolver.resolve(&mut captions, ResolveStrategy::VerticalShift);
        let pairs = resolver.detect_collisions(&captions);
        assert!(pairs.is_empty(), "Collisions should be resolved after vertical shift");
    }

    // 8. Shrink-to-fit reduces or eliminates overlapping box
    #[test]
    fn test_shrink_to_fit_resolves_collision() {
        let resolver = CollisionResolver::new();
        let mut captions = vec![
            make_box("top", 0, 0, 400, 60, 0, 2000, 5),
            make_box("bottom", 0, 40, 400, 80, 0, 2000, 1),
        ];
        resolver.resolve(&mut captions, ResolveStrategy::ShrinkToFit);
        let pairs = resolver.detect_collisions(&captions);
        assert!(pairs.is_empty(), "Collisions should be resolved after shrink-to-fit");
    }

    // 9. Overlap area computation
    #[test]
    fn test_spatial_overlap_area_correct() {
        let a = make_box("a", 0, 0, 100, 100, 0, 1000, 1);
        let b = make_box("b", 50, 50, 100, 100, 0, 1000, 1);
        // Overlap: x=[50,100)=50, y=[50,100)=50 → 2500
        assert_eq!(a.spatial_overlap_area(&b), 2500);
    }

    // 10. Temporal overlap computation
    #[test]
    fn test_temporal_overlap_ms_correct() {
        let a = make_box("a", 0, 0, 100, 50, 1000, 3000, 1);
        let b = make_box("b", 0, 0, 100, 50, 2000, 5000, 1);
        // Overlap: [2000,3000) = 1000 ms
        assert_eq!(a.temporal_overlap_ms(&b), 1000);
    }

    // 11. Three-way collision produces three pairs
    #[test]
    fn test_three_way_collision_three_pairs() {
        let resolver = CollisionResolver::new();
        let captions = vec![
            make_box("a", 0, 0, 200, 50, 0, 3000, 1),
            make_box("b", 0, 0, 200, 50, 0, 3000, 2),
            make_box("c", 0, 0, 200, 50, 0, 3000, 3),
        ];
        let pairs = resolver.detect_collisions(&captions);
        assert_eq!(pairs.len(), 3);
    }
}
