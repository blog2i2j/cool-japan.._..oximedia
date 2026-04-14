//! Configurable safe-area margins for broadcast compliance.
//!
//! Broadcast and streaming standards define "safe zones" within which
//! graphics must be confined to guarantee visibility on all consumer
//! displays and avoid being cropped by overscan.  This module provides
//! typed margin definitions for the most common standards together with
//! helpers to compute inner rectangles and to test whether points or
//! rectangles fall within a given safe zone.
//!
//! ## Standards supported
//!
//! | Standard | Action safe | Title safe | Notes |
//! |----------|-------------|------------|-------|
//! | SMPTE RP 218-2011 | 90 % | 80 % | Classic broadcast |
//! | EBU R 95 | 88 % | 80 % | European broadcast |
//! | Custom | any | any | User-defined fractions |
//!
//! ## Example
//!
//! ```rust
//! use oximedia_graphics::layout_margins::{SafeAreaMargins, BroadcastStandard, SafeRect};
//!
//! let margins = SafeAreaMargins::for_standard(BroadcastStandard::Smpte, 1920, 1080);
//! let action = margins.action_safe_rect();
//! let title  = margins.title_safe_rect();
//! assert!(action.contains_rect(&title));
//! ```

/// A broadcast or streaming safe-area standard.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BroadcastStandard {
    /// SMPTE RP 218: action-safe 90 %, title-safe 80 %.
    Smpte,
    /// EBU R 95: action-safe 88 %, title-safe 80 %.
    EbuR95,
    /// Web / streaming: typically 95 % content area, 90 % text area.
    Streaming,
    /// Fully custom fractions supplied by the caller.
    Custom {
        /// Action-safe fraction, e.g. `0.90`.
        action_safe: f32,
        /// Title-safe fraction, e.g. `0.80`.
        title_safe: f32,
    },
}

impl BroadcastStandard {
    /// Return `(action_safe, title_safe)` fractions for this standard.
    pub fn fractions(self) -> (f32, f32) {
        match self {
            Self::Smpte => (0.90, 0.80),
            Self::EbuR95 => (0.88, 0.80),
            Self::Streaming => (0.95, 0.90),
            Self::Custom { action_safe, title_safe } => (action_safe, title_safe),
        }
    }
}

// ---------------------------------------------------------------------------
// SafeRect
// ---------------------------------------------------------------------------

/// An axis-aligned rectangle defined by pixel coordinates.
///
/// All coordinates use a top-left origin with Y increasing downward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SafeRect {
    /// Left edge (inclusive).
    pub left: u32,
    /// Top edge (inclusive).
    pub top: u32,
    /// Right edge (exclusive).
    pub right: u32,
    /// Bottom edge (exclusive).
    pub bottom: u32,
}

impl SafeRect {
    /// Create a new rectangle.
    pub fn new(left: u32, top: u32, right: u32, bottom: u32) -> Self {
        Self { left, top, right, bottom }
    }

    /// Rectangle covering the whole frame.
    pub fn full_frame(width: u32, height: u32) -> Self {
        Self { left: 0, top: 0, right: width, bottom: height }
    }

    /// Width of the rectangle in pixels.
    pub fn width(&self) -> u32 {
        self.right.saturating_sub(self.left)
    }

    /// Height of the rectangle in pixels.
    pub fn height(&self) -> u32 {
        self.bottom.saturating_sub(self.top)
    }

    /// Center `(x, y)` of the rectangle.
    pub fn center(&self) -> (f32, f32) {
        (
            (self.left + self.right) as f32 / 2.0,
            (self.top + self.bottom) as f32 / 2.0,
        )
    }

    /// Returns `true` if the pixel point `(x, y)` falls inside the rectangle.
    pub fn contains_point(&self, x: u32, y: u32) -> bool {
        x >= self.left && x < self.right && y >= self.top && y < self.bottom
    }

    /// Returns `true` if `other` is fully contained within `self`.
    pub fn contains_rect(&self, other: &Self) -> bool {
        other.left >= self.left
            && other.top >= self.top
            && other.right <= self.right
            && other.bottom <= self.bottom
    }

    /// Returns the intersection of `self` and `other`, or `None` if they
    /// do not overlap.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let left = self.left.max(other.left);
        let top = self.top.max(other.top);
        let right = self.right.min(other.right);
        let bottom = self.bottom.min(other.bottom);
        if left < right && top < bottom {
            Some(Self { left, top, right, bottom })
        } else {
            None
        }
    }

    /// Clamp an `(x, y)` point to the nearest pixel inside the rectangle.
    pub fn clamp_point(&self, x: u32, y: u32) -> (u32, u32) {
        let cx = x.clamp(self.left, self.right.saturating_sub(1));
        let cy = y.clamp(self.top, self.bottom.saturating_sub(1));
        (cx, cy)
    }
}

// ---------------------------------------------------------------------------
// SafeAreaMargins
// ---------------------------------------------------------------------------

/// Broadcast safe-area margins computed for a specific frame size.
///
/// Use [`SafeAreaMargins::for_standard`] to build margins from a known
/// [`BroadcastStandard`], or [`SafeAreaMargins::custom`] for arbitrary
/// per-edge pixel values.
#[derive(Debug, Clone)]
pub struct SafeAreaMargins {
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// Left margin in pixels (action-safe).
    pub action_left: u32,
    /// Right margin in pixels (action-safe, measured from the right edge).
    pub action_right: u32,
    /// Top margin in pixels (action-safe).
    pub action_top: u32,
    /// Bottom margin in pixels (action-safe, measured from the bottom edge).
    pub action_bottom: u32,
    /// Left margin in pixels (title-safe).
    pub title_left: u32,
    /// Right margin in pixels (title-safe, measured from the right edge).
    pub title_right: u32,
    /// Top margin in pixels (title-safe).
    pub title_top: u32,
    /// Bottom margin in pixels (title-safe, measured from the bottom edge).
    pub title_bottom: u32,
}

impl SafeAreaMargins {
    /// Build margins according to `standard` for a frame of `width × height`.
    ///
    /// Margins are calculated symmetrically: each side gets half the total
    /// cropped area.
    pub fn for_standard(standard: BroadcastStandard, width: u32, height: u32) -> Self {
        let (action_frac, title_frac) = standard.fractions();
        Self::from_fractions(action_frac, title_frac, width, height)
    }

    /// Build symmetric margins from `action_safe` and `title_safe` fractions.
    ///
    /// A fraction of `0.90` means the safe area is 90 % of the frame dimension,
    /// leaving 5 % on each edge (10 % total).
    pub fn from_fractions(action_safe: f32, title_safe: f32, width: u32, height: u32) -> Self {
        let action_safe = action_safe.clamp(0.0, 1.0);
        let title_safe = title_safe.clamp(0.0, 1.0);

        let ah = ((1.0 - action_safe) / 2.0 * height as f32).round() as u32;
        let aw = ((1.0 - action_safe) / 2.0 * width as f32).round() as u32;
        let th = ((1.0 - title_safe) / 2.0 * height as f32).round() as u32;
        let tw = ((1.0 - title_safe) / 2.0 * width as f32).round() as u32;

        Self {
            frame_width: width,
            frame_height: height,
            action_left: aw,
            action_right: aw,
            action_top: ah,
            action_bottom: ah,
            title_left: tw,
            title_right: tw,
            title_top: th,
            title_bottom: th,
        }
    }

    /// Build fully custom, asymmetric margins.
    pub fn custom(
        width: u32,
        height: u32,
        action: [u32; 4], // [left, right, top, bottom]
        title: [u32; 4],  // [left, right, top, bottom]
    ) -> Self {
        Self {
            frame_width: width,
            frame_height: height,
            action_left: action[0],
            action_right: action[1],
            action_top: action[2],
            action_bottom: action[3],
            title_left: title[0],
            title_right: title[1],
            title_top: title[2],
            title_bottom: title[3],
        }
    }

    /// The action-safe rectangle within the frame.
    pub fn action_safe_rect(&self) -> SafeRect {
        SafeRect {
            left: self.action_left,
            top: self.action_top,
            right: self.frame_width.saturating_sub(self.action_right),
            bottom: self.frame_height.saturating_sub(self.action_bottom),
        }
    }

    /// The title-safe rectangle within the frame.
    pub fn title_safe_rect(&self) -> SafeRect {
        SafeRect {
            left: self.title_left,
            top: self.title_top,
            right: self.frame_width.saturating_sub(self.title_right),
            bottom: self.frame_height.saturating_sub(self.title_bottom),
        }
    }

    /// Full-frame rectangle (for convenience).
    pub fn full_frame_rect(&self) -> SafeRect {
        SafeRect::full_frame(self.frame_width, self.frame_height)
    }

    /// Returns `true` when the graphic at `rect` is within the action-safe area.
    pub fn is_action_safe(&self, rect: &SafeRect) -> bool {
        self.action_safe_rect().contains_rect(rect)
    }

    /// Returns `true` when the graphic at `rect` is within the title-safe area.
    pub fn is_title_safe(&self, rect: &SafeRect) -> bool {
        self.title_safe_rect().contains_rect(rect)
    }

    /// Clamp `rect` to fit within the action-safe area.
    pub fn clamp_to_action_safe(&self, rect: SafeRect) -> SafeRect {
        let safe = self.action_safe_rect();
        let left = rect.left.max(safe.left);
        let top = rect.top.max(safe.top);
        let right = rect.right.min(safe.right);
        let bottom = rect.bottom.min(safe.bottom);
        SafeRect { left, top, right: right.max(left), bottom: bottom.max(top) }
    }

    /// Clamp `rect` to fit within the title-safe area.
    pub fn clamp_to_title_safe(&self, rect: SafeRect) -> SafeRect {
        let safe = self.title_safe_rect();
        let left = rect.left.max(safe.left);
        let top = rect.top.max(safe.top);
        let right = rect.right.min(safe.right);
        let bottom = rect.bottom.min(safe.bottom);
        SafeRect { left, top, right: right.max(left), bottom: bottom.max(top) }
    }

    /// Generate an overlay mask (RGBA buffer) that highlights both safe zones
    /// as semi-transparent guide lines.
    ///
    /// The returned `Vec<u8>` has length `frame_width × frame_height × 4`.
    ///
    /// - Action-safe border: drawn in `action_color` (RGBA).
    /// - Title-safe border: drawn in `title_color` (RGBA).
    pub fn render_guide_overlay(
        &self,
        action_color: [u8; 4],
        title_color: [u8; 4],
        border_thickness: u32,
    ) -> Vec<u8> {
        let w = self.frame_width;
        let h = self.frame_height;
        let mut buf = vec![0u8; (w * h * 4) as usize];

        let draw_rect_border = |buf: &mut Vec<u8>, rect: SafeRect, color: [u8; 4], t: u32| {
            for row in rect.top..rect.bottom {
                for col in rect.left..rect.right {
                    let in_top = row < rect.top + t;
                    let in_bottom = row >= rect.bottom.saturating_sub(t);
                    let in_left = col < rect.left + t;
                    let in_right = col >= rect.right.saturating_sub(t);
                    if in_top || in_bottom || in_left || in_right {
                        let idx = (row * w + col) as usize * 4;
                        if idx + 3 < buf.len() {
                            buf[idx] = color[0];
                            buf[idx + 1] = color[1];
                            buf[idx + 2] = color[2];
                            buf[idx + 3] = color[3];
                        }
                    }
                }
            }
        };

        draw_rect_border(&mut buf, self.action_safe_rect(), action_color, border_thickness.max(1));
        draw_rect_border(&mut buf, self.title_safe_rect(), title_color, border_thickness.max(1));
        buf
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 1920;
    const H: u32 = 1080;

    // 1. SMPTE action safe rect is smaller than the full frame.
    #[test]
    fn test_smpte_action_safe_smaller_than_frame() {
        let m = SafeAreaMargins::for_standard(BroadcastStandard::Smpte, W, H);
        let action = m.action_safe_rect();
        assert!(action.width() < W);
        assert!(action.height() < H);
    }

    // 2. Title safe is always contained within action safe for SMPTE.
    #[test]
    fn test_smpte_title_inside_action() {
        let m = SafeAreaMargins::for_standard(BroadcastStandard::Smpte, W, H);
        let action = m.action_safe_rect();
        let title = m.title_safe_rect();
        assert!(
            action.contains_rect(&title),
            "title safe must be inside action safe"
        );
    }

    // 3. EBU R95 fractions differ from SMPTE.
    #[test]
    fn test_ebu_r95_fractions_differ_from_smpte() {
        let (a_smpte, _) = BroadcastStandard::Smpte.fractions();
        let (a_ebu, _) = BroadcastStandard::EbuR95.fractions();
        assert!(
            (a_smpte - a_ebu).abs() > 0.001,
            "action safe fractions should differ"
        );
    }

    // 4. Custom standard returns the fractions supplied.
    #[test]
    fn test_custom_standard_fractions() {
        let std = BroadcastStandard::Custom {
            action_safe: 0.93,
            title_safe: 0.85,
        };
        let (a, t) = std.fractions();
        assert!((a - 0.93).abs() < 1e-6);
        assert!((t - 0.85).abs() < 1e-6);
    }

    // 5. SafeRect::contains_point works correctly.
    #[test]
    fn test_safe_rect_contains_point() {
        let r = SafeRect::new(10, 10, 100, 100);
        assert!(r.contains_point(10, 10));
        assert!(r.contains_point(50, 50));
        assert!(!r.contains_point(100, 50)); // right edge is exclusive
        assert!(!r.contains_point(9, 50));
    }

    // 6. SafeRect::intersect returns None for non-overlapping rects.
    #[test]
    fn test_safe_rect_no_intersection() {
        let a = SafeRect::new(0, 0, 50, 50);
        let b = SafeRect::new(100, 100, 200, 200);
        assert!(a.intersect(&b).is_none());
    }

    // 7. SafeRect::intersect returns correct overlap.
    #[test]
    fn test_safe_rect_intersection() {
        let a = SafeRect::new(0, 0, 100, 100);
        let b = SafeRect::new(50, 50, 150, 150);
        let i = a.intersect(&b).expect("should intersect");
        assert_eq!(i.left, 50);
        assert_eq!(i.top, 50);
        assert_eq!(i.right, 100);
        assert_eq!(i.bottom, 100);
    }

    // 8. is_action_safe rejects a rect that overlaps the margin.
    #[test]
    fn test_is_action_safe_rejects_overlap() {
        let m = SafeAreaMargins::for_standard(BroadcastStandard::Smpte, W, H);
        // A rect starting at (0,0) overlaps the action-safe margin.
        let bad_rect = SafeRect::new(0, 0, 500, 100);
        assert!(!m.is_action_safe(&bad_rect));
    }

    // 9. is_title_safe accepts a rect fully inside the title safe area.
    #[test]
    fn test_is_title_safe_accepts_inner_rect() {
        let m = SafeAreaMargins::for_standard(BroadcastStandard::Smpte, W, H);
        let title = m.title_safe_rect();
        // A smaller rect inside the title safe should pass.
        let inner = SafeRect::new(
            title.left + 10,
            title.top + 10,
            title.right - 10,
            title.bottom - 10,
        );
        assert!(m.is_title_safe(&inner));
    }

    // 10. clamp_to_action_safe brings an oversized rect inside the safe area.
    #[test]
    fn test_clamp_to_action_safe() {
        let m = SafeAreaMargins::for_standard(BroadcastStandard::Smpte, W, H);
        let oversized = SafeRect::new(0, 0, W, H);
        let clamped = m.clamp_to_action_safe(oversized);
        let action = m.action_safe_rect();
        assert!(action.contains_rect(&clamped),
            "clamped rect must fit inside action safe area");
    }

    // 11. render_guide_overlay returns a buffer of the correct length.
    #[test]
    fn test_render_guide_overlay_length() {
        let m = SafeAreaMargins::for_standard(BroadcastStandard::Smpte, W, H);
        let buf = m.render_guide_overlay([255, 255, 0, 128], [0, 255, 255, 128], 2);
        assert_eq!(buf.len(), (W * H * 4) as usize);
    }

    // 12. render_guide_overlay has non-zero pixels at safe-zone borders.
    #[test]
    fn test_render_guide_overlay_has_border_pixels() {
        let m = SafeAreaMargins::for_standard(BroadcastStandard::Smpte, W, H);
        let action = m.action_safe_rect();
        let buf = m.render_guide_overlay([255, 255, 0, 200], [0, 255, 255, 200], 2);
        // Check a pixel on the top border of the action safe rect.
        let col = action.left + (action.width() / 2);
        let row = action.top;
        let idx = (row * W + col) as usize * 4;
        assert!(buf[idx + 3] > 0, "border pixel alpha must be non-zero");
    }

    // 13. SafeRect::center is at the midpoint.
    #[test]
    fn test_safe_rect_center() {
        let r = SafeRect::new(0, 0, 200, 100);
        let (cx, cy) = r.center();
        assert!((cx - 100.0).abs() < 0.5);
        assert!((cy - 50.0).abs() < 0.5);
    }

    // 14. from_fractions with 100% produces a zero-margin rect.
    #[test]
    fn test_from_fractions_full() {
        let m = SafeAreaMargins::from_fractions(1.0, 1.0, W, H);
        let action = m.action_safe_rect();
        assert_eq!(action, SafeRect::new(0, 0, W, H));
    }
}
