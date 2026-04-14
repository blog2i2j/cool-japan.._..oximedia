//! Advanced subtitle cue positioning and collision avoidance.
//!
//! Provides percentage-based anchor positions, pixel-box computation, collision
//! detection between simultaneously active cues, and a vertical-shift resolver
//! that nudges overlapping cues apart without leaving the frame.

#![allow(dead_code)]

// ============================================================================
// Types
// ============================================================================

/// Named anchor point for a subtitle cue box.
///
/// Anchors are arranged in a 3×3 grid (top/middle/bottom × left/center/right).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionAnchor {
    /// Top-left corner.
    TopLeft,
    /// Top-center.
    TopCenter,
    /// Top-right corner.
    TopRight,
    /// Middle-left.
    MiddleLeft,
    /// Middle-center (screen center).
    MiddleCenter,
    /// Middle-right.
    MiddleRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-center (standard subtitle position).
    BottomCenter,
    /// Bottom-right corner.
    BottomRight,
}

impl PositionAnchor {
    /// Returns the default `(x_pct, y_pct)` origin for this anchor as a
    /// fraction of screen dimensions in `[0.0, 100.0]`.
    ///
    /// The origin is the *attachment point* of the cue box; for example
    /// `BottomCenter` places the box origin at `(50 %, 100 %)` so the box
    /// grows upward from the bottom edge.
    #[must_use]
    pub fn default_origin_pct(&self) -> (f32, f32) {
        match self {
            Self::TopLeft => (0.0, 0.0),
            Self::TopCenter => (50.0, 0.0),
            Self::TopRight => (100.0, 0.0),
            Self::MiddleLeft => (0.0, 50.0),
            Self::MiddleCenter => (50.0, 50.0),
            Self::MiddleRight => (100.0, 50.0),
            Self::BottomLeft => (0.0, 100.0),
            Self::BottomCenter => (50.0, 100.0),
            Self::BottomRight => (100.0, 100.0),
        }
    }

    /// Returns `true` when the anchor is on the bottom half of the screen.
    #[must_use]
    pub fn is_bottom(&self) -> bool {
        matches!(self, Self::BottomLeft | Self::BottomCenter | Self::BottomRight)
    }

    /// Returns `true` when the anchor is on the top half of the screen.
    #[must_use]
    pub fn is_top(&self) -> bool {
        matches!(self, Self::TopLeft | Self::TopCenter | Self::TopRight)
    }
}

/// Position specification for a subtitle cue.
#[derive(Debug, Clone, PartialEq)]
pub struct CuePosition {
    /// Named anchor point.
    pub anchor: PositionAnchor,
    /// Horizontal margin from the anchor edge, as percentage of screen width.
    /// Positive = inward from the reference edge.
    pub margin_x_pct: f32,
    /// Vertical margin from the anchor edge, as percentage of screen height.
    /// Positive = inward from the reference edge.
    pub margin_y_pct: f32,
    /// Optional explicit line number (like WebVTT `line:N`).
    /// If `Some`, overrides the computed vertical position.
    pub line: Option<i32>,
}

impl CuePosition {
    /// Create a standard bottom-center position with default margins.
    #[must_use]
    pub fn bottom_center() -> Self {
        Self {
            anchor: PositionAnchor::BottomCenter,
            margin_x_pct: 0.0,
            margin_y_pct: 5.0,
            line: None,
        }
    }

    /// Create a top-center position with default margins.
    #[must_use]
    pub fn top_center() -> Self {
        Self {
            anchor: PositionAnchor::TopCenter,
            margin_x_pct: 0.0,
            margin_y_pct: 5.0,
            line: None,
        }
    }
}

impl Default for CuePosition {
    fn default() -> Self {
        Self::bottom_center()
    }
}

/// Bounding box for a subtitle cue, expressed as a percentage of screen
/// dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct CueBox {
    /// Anchor and margin of the cue box.
    pub position: CuePosition,
    /// Width of the cue box as a percentage of screen width `[0.0, 100.0]`.
    pub width_pct: f32,
    /// Height of the cue box as a percentage of screen height `[0.0, 100.0]`.
    pub height_pct: f32,
}

impl CueBox {
    /// Create a new cue box.
    #[must_use]
    pub fn new(position: CuePosition, width_pct: f32, height_pct: f32) -> Self {
        Self {
            position,
            width_pct: width_pct.clamp(0.0, 100.0),
            height_pct: height_pct.clamp(0.0, 100.0),
        }
    }

    /// Create a typical bottom-center cue box (80 % wide, 10 % tall).
    #[must_use]
    pub fn default_bottom() -> Self {
        Self::new(CuePosition::bottom_center(), 80.0, 10.0)
    }
}

/// A detected collision between two simultaneously active cue boxes.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionCollision {
    /// Index of the first cue in the input slice.
    pub cue_a: usize,
    /// Index of the second cue in the input slice.
    pub cue_b: usize,
    /// Fraction of the smaller box's area that is covered by the overlap,
    /// in `[0.0, 1.0]`.
    pub overlap_pct: f32,
}

// ============================================================================
// PositionResolver
// ============================================================================

/// Resolves pixel positions and collision avoidance for subtitle cue boxes.
///
/// All percentage-based coordinates are converted to pixels relative to
/// `screen_width × screen_height`.
#[derive(Debug, Clone)]
pub struct PositionResolver {
    /// Screen width in pixels.
    pub screen_width: u32,
    /// Screen height in pixels.
    pub screen_height: u32,
}

impl PositionResolver {
    /// Create a new resolver for the given screen dimensions.
    #[must_use]
    pub fn new(screen_width: u32, screen_height: u32) -> Self {
        Self {
            screen_width,
            screen_height,
        }
    }

    /// Compute the pixel bounding box `(x, y, width, height)` for a `CueBox`.
    ///
    /// The `(x, y)` returned is the **top-left** corner of the rendered box.
    /// Bottom-anchored boxes grow upward; top-anchored boxes grow downward.
    ///
    /// # Coordinate system
    ///
    /// - `x` increases to the right
    /// - `y` increases downward (0 = top of screen)
    #[must_use]
    pub fn compute_pixel_box(&self, cue: &CueBox) -> (u32, u32, u32, u32) {
        let sw = self.screen_width as f32;
        let sh = self.screen_height as f32;

        let box_w = (cue.width_pct / 100.0 * sw).round() as u32;
        let box_h = (cue.height_pct / 100.0 * sh).round() as u32;

        let (origin_x_pct, origin_y_pct) = cue.position.anchor.default_origin_pct();

        // Horizontal position: center-anchor horizontally around origin_x, then
        // apply margin (negative = move toward center).
        let anchor_x = origin_x_pct / 100.0 * sw;
        let margin_x = cue.position.margin_x_pct / 100.0 * sw;

        let x_center = match cue.position.anchor {
            PositionAnchor::TopLeft | PositionAnchor::MiddleLeft | PositionAnchor::BottomLeft => {
                anchor_x + margin_x
            }
            PositionAnchor::TopRight
            | PositionAnchor::MiddleRight
            | PositionAnchor::BottomRight => anchor_x - margin_x - box_w as f32,
            // Center anchors: center the box at the anchor.
            _ => anchor_x - box_w as f32 / 2.0,
        };

        let margin_y = cue.position.margin_y_pct / 100.0 * sh;
        let y = if let Some(line) = cue.position.line {
            // Explicit line: line * box_height from the top (or bottom for negative)
            if line >= 0 {
                line as f32 * box_h as f32
            } else {
                sh + line as f32 * box_h as f32 - box_h as f32
            }
        } else if cue.position.anchor.is_bottom() {
            // Bottom-anchored: place box so its bottom edge is at (100% - margin_y).
            let anchor_y = origin_y_pct / 100.0 * sh;
            anchor_y - margin_y - box_h as f32
        } else if cue.position.anchor.is_top() {
            let anchor_y = origin_y_pct / 100.0 * sh;
            anchor_y + margin_y
        } else {
            // Middle anchors: center vertically around origin.
            let anchor_y = origin_y_pct / 100.0 * sh;
            anchor_y - box_h as f32 / 2.0
        };

        // Clamp to screen boundaries.
        let x = (x_center.max(0.0) as u32).min(self.screen_width.saturating_sub(box_w));
        let y = (y.max(0.0) as u32).min(self.screen_height.saturating_sub(box_h));

        (x, y, box_w, box_h)
    }

    /// Detect collisions among cue boxes that are active at the *same* time.
    ///
    /// `cues` is a slice of `(start_ms, CueBox)` tuples.  Two cues are
    /// "same-time" when their active windows overlap; for simplicity this
    /// function treats each entry's `start_ms` as its only active moment and
    /// compares cues with the same `start_ms`.  (For per-frame collision
    /// detection, callers should pre-filter to cues active at the target
    /// timestamp.)
    #[must_use]
    pub fn detect_collisions(&self, cues: &[(u64, CueBox)]) -> Vec<PositionCollision> {
        let mut collisions = Vec::new();

        for i in 0..cues.len() {
            for j in (i + 1)..cues.len() {
                // Only compare cues active at the same moment (same start_ms).
                if cues[i].0 != cues[j].0 {
                    continue;
                }

                let (ax, ay, aw, ah) = self.compute_pixel_box(&cues[i].1);
                let (bx, by, bw, bh) = self.compute_pixel_box(&cues[j].1);

                if let Some(overlap) = rect_overlap_fraction(ax, ay, aw, ah, bx, by, bw, bh) {
                    collisions.push(PositionCollision {
                        cue_a: i,
                        cue_b: j,
                        overlap_pct: overlap,
                    });
                }
            }
        }

        collisions
    }

    /// Resolve vertical collisions by shifting overlapping cue boxes.
    ///
    /// Bottom-anchored cues that overlap vertically are shifted *upward*;
    /// top-anchored cues are shifted *downward*.  The algorithm iterates until
    /// no collision remains or a maximum of 20 passes is reached.
    pub fn resolve_vertically(&self, cues: &mut Vec<(u64, CueBox)>) {
        const MAX_PASSES: usize = 20;

        for _ in 0..MAX_PASSES {
            let collisions = self.detect_collisions(cues);
            if collisions.is_empty() {
                break;
            }

            for collision in &collisions {
                let (ax, ay, _aw, ah) = self.compute_pixel_box(&cues[collision.cue_a].1);
                let (bx, by, _bw, bh) = self.compute_pixel_box(&cues[collision.cue_b].1);

                // Calculate vertical overlap depth.
                let a_bottom = ay + ah;
                let b_bottom = by + bh;
                let overlap_top = ay.max(by);
                let overlap_bottom = a_bottom.min(b_bottom);

                if overlap_bottom <= overlap_top {
                    continue;
                }
                let depth = (overlap_bottom - overlap_top) as f32;
                let sh = self.screen_height as f32;

                // Shift the cue whose start_ms is later (b is the "newer" one)
                // upward if bottom-anchored, downward if top-anchored.
                let shift_pct = depth / sh * 100.0 + 1.0; // +1 % padding

                let b_cue = &mut cues[collision.cue_b].1;
                if b_cue.position.anchor.is_bottom() {
                    b_cue.position.margin_y_pct += shift_pct;
                } else {
                    b_cue.position.margin_y_pct += shift_pct;
                }

                let _ = (ax, bx); // suppress unused warnings
            }
        }
    }
}

// ============================================================================
// Geometry helpers
// ============================================================================

/// Compute the overlap fraction between two rectangles.
///
/// Returns `Some(fraction)` where `fraction` is `overlap_area / min_area`
/// if the rectangles overlap, otherwise `None`.
fn rect_overlap_fraction(
    ax: u32,
    ay: u32,
    aw: u32,
    ah: u32,
    bx: u32,
    by: u32,
    bw: u32,
    bh: u32,
) -> Option<f32> {
    let ax2 = ax + aw;
    let ay2 = ay + ah;
    let bx2 = bx + bw;
    let by2 = by + bh;

    let ox1 = ax.max(bx);
    let oy1 = ay.max(by);
    let ox2 = ax2.min(bx2);
    let oy2 = ay2.min(by2);

    if ox2 <= ox1 || oy2 <= oy1 {
        return None;
    }

    let overlap_area = ((ox2 - ox1) * (oy2 - oy1)) as f32;
    let area_a = (aw * ah) as f32;
    let area_b = (bw * bh) as f32;
    let min_area = area_a.min(area_b);

    if min_area <= 0.0 {
        return None;
    }

    Some((overlap_area / min_area).min(1.0))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn hd_resolver() -> PositionResolver {
        PositionResolver::new(1920, 1080)
    }

    // ── Anchor origin tests ───────────────────────────────────────────────────

    #[test]
    fn test_anchor_origin_bottom_center() {
        let (x, y) = PositionAnchor::BottomCenter.default_origin_pct();
        assert!((x - 50.0).abs() < 0.01);
        assert!((y - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_anchor_origin_top_left() {
        let (x, y) = PositionAnchor::TopLeft.default_origin_pct();
        assert!((x - 0.0).abs() < 0.01);
        assert!((y - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_anchor_is_bottom() {
        assert!(PositionAnchor::BottomCenter.is_bottom());
        assert!(PositionAnchor::BottomLeft.is_bottom());
        assert!(!PositionAnchor::TopCenter.is_bottom());
        assert!(!PositionAnchor::MiddleCenter.is_bottom());
    }

    #[test]
    fn test_anchor_is_top() {
        assert!(PositionAnchor::TopRight.is_top());
        assert!(!PositionAnchor::BottomRight.is_top());
    }

    // ── Pixel box computation tests ───────────────────────────────────────────

    #[test]
    fn test_pixel_box_bottom_center_no_margin() {
        let resolver = hd_resolver();
        let cue = CueBox::new(
            CuePosition {
                anchor: PositionAnchor::BottomCenter,
                margin_x_pct: 0.0,
                margin_y_pct: 0.0,
                line: None,
            },
            80.0,
            10.0,
        );
        let (x, y, w, h) = resolver.compute_pixel_box(&cue);
        // Width = 80% of 1920 = 1536; centered → x = (1920-1536)/2 = 192
        assert_eq!(w, 1536);
        assert_eq!(h, 108); // 10% of 1080
        assert_eq!(x, 192); // centered
        // Bottom-anchored, no margin → y = 1080 - 108 = 972
        assert_eq!(y, 972);
    }

    #[test]
    fn test_pixel_box_top_center() {
        let resolver = hd_resolver();
        let cue = CueBox::new(
            CuePosition {
                anchor: PositionAnchor::TopCenter,
                margin_x_pct: 0.0,
                margin_y_pct: 5.0,
                line: None,
            },
            80.0,
            10.0,
        );
        let (x, y, w, h) = resolver.compute_pixel_box(&cue);
        assert_eq!(w, 1536);
        assert_eq!(h, 108);
        // y = 0 + 5% of 1080 = 54
        assert_eq!(y, 54);
        assert_eq!(x, 192);
    }

    #[test]
    fn test_pixel_box_middle_center() {
        let resolver = hd_resolver();
        let cue = CueBox::new(
            CuePosition {
                anchor: PositionAnchor::MiddleCenter,
                margin_x_pct: 0.0,
                margin_y_pct: 0.0,
                line: None,
            },
            50.0,
            10.0,
        );
        let (x, y, w, h) = resolver.compute_pixel_box(&cue);
        assert_eq!(w, 960); // 50% of 1920
        assert_eq!(h, 108);
        assert_eq!(x, 480); // (1920 - 960) / 2
        // Middle center: y = 540 - 108/2 = 486
        assert_eq!(y, 486);
    }

    #[test]
    fn test_pixel_box_explicit_line() {
        let resolver = hd_resolver();
        let cue = CueBox::new(
            CuePosition {
                anchor: PositionAnchor::BottomCenter,
                margin_x_pct: 0.0,
                margin_y_pct: 0.0,
                line: Some(2),
            },
            80.0,
            10.0,
        );
        let (_x, y, _w, h) = resolver.compute_pixel_box(&cue);
        // line=2: y = 2 * h
        assert_eq!(y, 2 * h);
    }

    // ── Collision detection tests ─────────────────────────────────────────────

    #[test]
    fn test_no_collision_different_positions() {
        let resolver = hd_resolver();
        let top_cue = CueBox::new(
            CuePosition {
                anchor: PositionAnchor::TopCenter,
                margin_x_pct: 0.0,
                margin_y_pct: 0.0,
                line: None,
            },
            80.0,
            10.0,
        );
        let bottom_cue = CueBox::new(
            CuePosition {
                anchor: PositionAnchor::BottomCenter,
                margin_x_pct: 0.0,
                margin_y_pct: 0.0,
                line: None,
            },
            80.0,
            10.0,
        );
        // Same start time but different vertical positions.
        let cues = vec![(0u64, top_cue), (0u64, bottom_cue)];
        let collisions = resolver.detect_collisions(&cues);
        assert!(collisions.is_empty(), "top and bottom should not collide");
    }

    #[test]
    fn test_collision_detected_same_position() {
        let resolver = hd_resolver();
        // Two identical boxes at the same start_ms → full overlap.
        let c1 = CueBox::default_bottom();
        let c2 = CueBox::default_bottom();
        let cues = vec![(0u64, c1), (0u64, c2)];
        let collisions = resolver.detect_collisions(&cues);
        assert!(!collisions.is_empty(), "identical boxes must collide");
        assert!(collisions[0].overlap_pct > 0.99, "overlap should be ~1.0");
    }

    #[test]
    fn test_no_collision_different_start_times() {
        let resolver = hd_resolver();
        let c1 = CueBox::default_bottom();
        let c2 = CueBox::default_bottom();
        // Different start_ms → not compared.
        let cues = vec![(0u64, c1), (5_000u64, c2)];
        let collisions = resolver.detect_collisions(&cues);
        assert!(collisions.is_empty(), "different start times → no collision check");
    }

    #[test]
    fn test_vertical_resolution_separates_cues() {
        let resolver = hd_resolver();
        let c1 = CueBox::default_bottom();
        let c2 = CueBox::default_bottom();
        let mut cues = vec![(0u64, c1), (0u64, c2)];

        // Verify collision exists before resolution.
        assert!(!resolver.detect_collisions(&cues).is_empty());

        resolver.resolve_vertically(&mut cues);

        // After resolution, collisions should be gone.
        let remaining = resolver.detect_collisions(&cues);
        assert!(
            remaining.is_empty(),
            "after resolve_vertically, no collisions expected; remaining: {remaining:?}"
        );
    }

    #[test]
    fn test_rect_overlap_fraction_no_overlap() {
        let result = rect_overlap_fraction(0, 0, 10, 10, 20, 20, 10, 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_rect_overlap_fraction_full_overlap() {
        let result = rect_overlap_fraction(0, 0, 10, 10, 0, 0, 10, 10);
        assert!(result.is_some());
        assert!((result.unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cue_position_default() {
        let pos = CuePosition::default();
        assert_eq!(pos.anchor, PositionAnchor::BottomCenter);
    }

    #[test]
    fn test_cue_box_default_bottom() {
        let cb = CueBox::default_bottom();
        assert!((cb.width_pct - 80.0).abs() < 0.01);
        assert!((cb.height_pct - 10.0).abs() < 0.01);
    }
}
