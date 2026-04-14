//! Sign language overlay region management and picture-in-picture positioning.
//!
//! This module provides infrastructure for embedding a sign language interpreter
//! video region (picture-in-picture) alongside primary video content.  It handles:
//!
//! - **Region geometry**: defining and validating the PiP window dimensions, margins,
//!   and corner anchoring.
//! - **Temporal scheduling**: associating interpreter windows with subtitle time ranges
//!   so the overlay appears only when relevant captions are active.
//! - **Layout collision avoidance**: detecting when the interpreter region would cover
//!   speaker lower-thirds or other UI elements, and automatically repositioning to a
//!   safe quadrant.
//! - **Transition control**: configuring fade-in / fade-out durations so the window
//!   animates smoothly rather than blinking on and off.
//!
//! # Example
//!
//! ```rust
//! use oximedia_subtitle::sign_language::{
//!     SignLanguageRegion, PipAnchor, SignLanguageScheduler, OverlapPolicy,
//! };
//!
//! let region = SignLanguageRegion::new(320, 180)
//!     .with_anchor(PipAnchor::BottomRight)
//!     .with_margin(24, 24);
//!
//! let mut scheduler = SignLanguageScheduler::new(region, OverlapPolicy::Reposition);
//! scheduler.add_window(0, 5000);
//! scheduler.add_window(8000, 12000);
//!
//! let visible = scheduler.is_visible(3000);
//! assert!(visible);
//! ```

// ============================================================================
// Anchor
// ============================================================================

/// The corner of the video frame to which the PiP window is anchored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PipAnchor {
    /// Top-left corner of the video.
    TopLeft,
    /// Top-right corner of the video.
    TopRight,
    /// Bottom-left corner of the video.
    BottomLeft,
    /// Bottom-right corner of the video.
    BottomRight,
    /// Manually specified absolute position (top-left origin of the PiP box).
    Custom {
        /// X coordinate of the PiP box top-left corner in pixels.
        x: u32,
        /// Y coordinate of the PiP box top-left corner in pixels.
        y: u32,
    },
}

impl PipAnchor {
    /// Compute the top-left pixel coordinates of the PiP box given the containing
    /// frame dimensions and the PiP dimensions + margins.
    ///
    /// Returns `(x, y)` where both values are relative to the frame top-left.
    #[must_use]
    pub fn compute_position(
        self,
        frame_width: u32,
        frame_height: u32,
        pip_width: u32,
        pip_height: u32,
        margin_x: u32,
        margin_y: u32,
    ) -> (u32, u32) {
        match self {
            Self::TopLeft => (margin_x, margin_y),
            Self::TopRight => {
                let x = frame_width.saturating_sub(pip_width).saturating_sub(margin_x);
                (x, margin_y)
            }
            Self::BottomLeft => {
                let y = frame_height.saturating_sub(pip_height).saturating_sub(margin_y);
                (margin_x, y)
            }
            Self::BottomRight => {
                let x = frame_width.saturating_sub(pip_width).saturating_sub(margin_x);
                let y = frame_height.saturating_sub(pip_height).saturating_sub(margin_y);
                (x, y)
            }
            Self::Custom { x, y } => (x, y),
        }
    }

    /// Return the opposite corner (used when repositioning to avoid collisions).
    #[must_use]
    pub fn opposite(self) -> Self {
        match self {
            Self::TopLeft => Self::BottomRight,
            Self::TopRight => Self::BottomLeft,
            Self::BottomLeft => Self::TopRight,
            Self::BottomRight => Self::TopLeft,
            Self::Custom { .. } => Self::BottomRight,
        }
    }

    /// Return the next anchor in clockwise order (for exhaustive repositioning).
    #[must_use]
    pub fn next_clockwise(self) -> Self {
        match self {
            Self::TopLeft => Self::TopRight,
            Self::TopRight => Self::BottomRight,
            Self::BottomRight => Self::BottomLeft,
            Self::BottomLeft => Self::TopLeft,
            Self::Custom { .. } => Self::TopLeft,
        }
    }
}

// ============================================================================
// Region
// ============================================================================

/// An axis-aligned bounding rectangle used for overlap testing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    /// Left edge (pixels, 0-origin).
    pub x: u32,
    /// Top edge (pixels, 0-origin).
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Rect {
    /// Create a new rectangle.
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self { x, y, width, height }
    }

    /// Right edge (exclusive).
    #[must_use]
    pub fn right(&self) -> u32 {
        self.x + self.width
    }

    /// Bottom edge (exclusive).
    #[must_use]
    pub fn bottom(&self) -> u32 {
        self.y + self.height
    }

    /// Returns `true` if this rectangle overlaps with `other`.
    #[must_use]
    pub fn overlaps(&self, other: &Rect) -> bool {
        self.x < other.right()
            && self.right() > other.x
            && self.y < other.bottom()
            && self.bottom() > other.y
    }

    /// Returns the intersection of two rectangles, or `None` if they do not overlap.
    #[must_use]
    pub fn intersection(&self, other: &Rect) -> Option<Rect> {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());
        if right > x && bottom > y {
            Some(Rect::new(x, y, right - x, bottom - y))
        } else {
            None
        }
    }

    /// Area in pixels².
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// The sign language interpreter PiP region descriptor.
#[derive(Clone, Debug)]
pub struct SignLanguageRegion {
    /// Width of the interpreter window in pixels.
    pub width: u32,
    /// Height of the interpreter window in pixels.
    pub height: u32,
    /// Preferred anchor corner.
    pub anchor: PipAnchor,
    /// Horizontal margin from the frame edge (pixels).
    pub margin_x: u32,
    /// Vertical margin from the frame edge (pixels).
    pub margin_y: u32,
    /// Fade-in duration in milliseconds.
    pub fade_in_ms: u32,
    /// Fade-out duration in milliseconds.
    pub fade_out_ms: u32,
    /// Border thickness in pixels (0 = none).
    pub border_px: u32,
    /// Border colour as RGBA bytes.
    pub border_color: [u8; 4],
}

impl SignLanguageRegion {
    /// Create a new region with the given pixel dimensions.
    ///
    /// Defaults: `BottomRight` anchor, 16 px margins, 200 ms fade transitions,
    /// no border.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            anchor: PipAnchor::BottomRight,
            margin_x: 16,
            margin_y: 16,
            fade_in_ms: 200,
            fade_out_ms: 200,
            border_px: 0,
            border_color: [0, 0, 0, 255],
        }
    }

    /// Set the anchor corner.
    #[must_use]
    pub fn with_anchor(mut self, anchor: PipAnchor) -> Self {
        self.anchor = anchor;
        self
    }

    /// Set horizontal and vertical margins.
    #[must_use]
    pub fn with_margin(mut self, margin_x: u32, margin_y: u32) -> Self {
        self.margin_x = margin_x;
        self.margin_y = margin_y;
        self
    }

    /// Set fade transition durations.
    #[must_use]
    pub fn with_fades(mut self, fade_in_ms: u32, fade_out_ms: u32) -> Self {
        self.fade_in_ms = fade_in_ms;
        self.fade_out_ms = fade_out_ms;
        self
    }

    /// Set a border around the interpreter window.
    #[must_use]
    pub fn with_border(mut self, thickness_px: u32, rgba: [u8; 4]) -> Self {
        self.border_px = thickness_px;
        self.border_color = rgba;
        self
    }

    /// Compute the bounding [`Rect`] of this region within a frame of the given
    /// dimensions using the configured anchor and margins.
    #[must_use]
    pub fn bounding_rect(&self, frame_width: u32, frame_height: u32) -> Rect {
        let (x, y) = self.anchor.compute_position(
            frame_width,
            frame_height,
            self.width,
            self.height,
            self.margin_x,
            self.margin_y,
        );
        Rect::new(x, y, self.width, self.height)
    }

    /// Returns `true` if the region fits within the frame without clipping.
    #[must_use]
    pub fn fits_in_frame(&self, frame_width: u32, frame_height: u32) -> bool {
        let r = self.bounding_rect(frame_width, frame_height);
        r.right() <= frame_width && r.bottom() <= frame_height
    }

    /// Aspect ratio (width / height) as an `f32`.
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            return 0.0;
        }
        self.width as f32 / self.height as f32
    }
}

// ============================================================================
// Overlap policy
// ============================================================================

/// How the scheduler handles overlap between the PiP region and a nominated
/// exclusion rectangle (e.g. lower-third graphics).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlapPolicy {
    /// Leave the region as configured; the caller is responsible for layout.
    Ignore,
    /// Automatically try each corner in clockwise order until a non-overlapping
    /// position is found.  Falls back to the original position if all corners
    /// overlap.
    Reposition,
    /// Hide the PiP window entirely when an overlap would occur.
    Hide,
}

// ============================================================================
// Temporal window
// ============================================================================

/// A time interval during which the sign language PiP should be visible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisibilityWindow {
    /// Start of the visible window (ms, inclusive).
    pub start_ms: i64,
    /// End of the visible window (ms, exclusive).
    pub end_ms: i64,
}

impl VisibilityWindow {
    /// Create a new visibility window.
    #[must_use]
    pub fn new(start_ms: i64, end_ms: i64) -> Self {
        Self { start_ms, end_ms }
    }

    /// Duration of the window in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }

    /// Returns `true` if `timestamp_ms` is within this window.
    #[must_use]
    pub fn contains(&self, timestamp_ms: i64) -> bool {
        timestamp_ms >= self.start_ms && timestamp_ms < self.end_ms
    }
}

// ============================================================================
// Scheduler
// ============================================================================

/// Schedules sign language PiP visibility windows and manages positioning.
#[derive(Clone, Debug)]
pub struct SignLanguageScheduler {
    /// The configured PiP region.
    pub region: SignLanguageRegion,
    /// Policy for handling overlap with exclusion rects.
    pub overlap_policy: OverlapPolicy,
    /// Registered visibility windows, sorted by start time.
    windows: Vec<VisibilityWindow>,
    /// Optional exclusion rectangles (e.g. lower thirds).
    exclusions: Vec<Rect>,
}

impl SignLanguageScheduler {
    /// Create a new scheduler with the given region and overlap policy.
    #[must_use]
    pub fn new(region: SignLanguageRegion, overlap_policy: OverlapPolicy) -> Self {
        Self {
            region,
            overlap_policy,
            windows: Vec::new(),
            exclusions: Vec::new(),
        }
    }

    /// Add a visibility window.  Overlapping windows are merged at query time.
    pub fn add_window(&mut self, start_ms: i64, end_ms: i64) {
        self.windows.push(VisibilityWindow::new(start_ms, end_ms));
        self.windows.sort_by_key(|w| w.start_ms);
    }

    /// Add an exclusion rectangle (e.g. for a lower-third graphic region).
    pub fn add_exclusion(&mut self, rect: Rect) {
        self.exclusions.push(rect);
    }

    /// Remove all scheduled windows.
    pub fn clear_windows(&mut self) {
        self.windows.clear();
    }

    /// Remove all exclusion rectangles.
    pub fn clear_exclusions(&mut self) {
        self.exclusions.clear();
    }

    /// Returns `true` if the PiP should be visible at `timestamp_ms`.
    #[must_use]
    pub fn is_visible(&self, timestamp_ms: i64) -> bool {
        self.windows.iter().any(|w| w.contains(timestamp_ms))
    }

    /// Returns the number of registered windows.
    #[must_use]
    pub fn window_count(&self) -> usize {
        self.windows.len()
    }

    /// Compute the display rectangle and opacity for the given timestamp and
    /// frame dimensions.  Returns `None` if the PiP should not be shown.
    ///
    /// The `opacity` field in [`PipFrame`] accounts for fade-in / fade-out
    /// relative to the nearest window boundary.
    #[must_use]
    pub fn frame_at(
        &self,
        timestamp_ms: i64,
        frame_width: u32,
        frame_height: u32,
    ) -> Option<PipFrame> {
        // Find the active window
        let window = self.windows.iter().find(|w| w.contains(timestamp_ms))?;

        // Compute fade opacity
        let opacity = self.compute_opacity(timestamp_ms, window);

        // Determine effective anchor, respecting overlap policy
        let effective_anchor = self.resolve_anchor(frame_width, frame_height);

        let mut effective_region = self.region.clone();
        effective_region.anchor = effective_anchor;

        let rect = effective_region.bounding_rect(frame_width, frame_height);

        Some(PipFrame {
            rect,
            opacity,
            anchor: effective_anchor,
        })
    }

    /// Compute the opacity in `[0.0, 1.0]` based on distance to window edges.
    fn compute_opacity(&self, timestamp_ms: i64, window: &VisibilityWindow) -> f32 {
        let fade_in = self.region.fade_in_ms as i64;
        let fade_out = self.region.fade_out_ms as i64;

        let time_from_start = timestamp_ms - window.start_ms;
        let time_to_end = window.end_ms - timestamp_ms;

        let fade_in_alpha = if fade_in > 0 {
            (time_from_start as f32 / fade_in as f32).min(1.0).max(0.0)
        } else {
            1.0
        };

        let fade_out_alpha = if fade_out > 0 {
            (time_to_end as f32 / fade_out as f32).min(1.0).max(0.0)
        } else {
            1.0
        };

        fade_in_alpha.min(fade_out_alpha)
    }

    /// Resolve the effective anchor by trying alternatives if `Reposition` is
    /// configured and exclusion rects would cause an overlap.
    fn resolve_anchor(&self, frame_width: u32, frame_height: u32) -> PipAnchor {
        if self.exclusions.is_empty() || self.overlap_policy == OverlapPolicy::Ignore {
            return self.region.anchor;
        }

        let mut candidate = self.region.anchor;
        for _ in 0..4 {
            let rect = {
                let mut r = self.region.clone();
                r.anchor = candidate;
                r.bounding_rect(frame_width, frame_height)
            };
            let collides = self.exclusions.iter().any(|ex| rect.overlaps(ex));
            if !collides {
                return candidate;
            }
            candidate = candidate.next_clockwise();
        }
        // All corners collide — return original
        self.region.anchor
    }

    /// Merge overlapping/adjacent windows and return a compacted list.
    ///
    /// This is useful for displaying a timeline or computing total covered
    /// duration without double-counting.
    #[must_use]
    pub fn merged_windows(&self) -> Vec<VisibilityWindow> {
        let mut result: Vec<VisibilityWindow> = Vec::new();
        for w in &self.windows {
            match result.last_mut() {
                Some(last) if w.start_ms <= last.end_ms => {
                    if w.end_ms > last.end_ms {
                        last.end_ms = w.end_ms;
                    }
                }
                _ => result.push(*w),
            }
        }
        result
    }

    /// Total duration (ms) covered by all visibility windows (after merging).
    #[must_use]
    pub fn total_coverage_ms(&self) -> i64 {
        self.merged_windows()
            .iter()
            .map(|w| w.duration_ms())
            .sum()
    }
}

// ============================================================================
// PipFrame
// ============================================================================

/// The computed display parameters for the sign language PiP at a single frame.
#[derive(Clone, Debug)]
pub struct PipFrame {
    /// Pixel rectangle for the interpreter window within the output frame.
    pub rect: Rect,
    /// Composite opacity in `[0.0, 1.0]` (accounts for fade-in / fade-out).
    pub opacity: f32,
    /// Effective anchor used (may differ from configured if repositioned).
    pub anchor: PipAnchor,
}

impl PipFrame {
    /// Returns `true` if this frame is effectively invisible (opacity == 0).
    #[must_use]
    pub fn is_invisible(&self) -> bool {
        self.opacity <= 0.0
    }

    /// Returns `true` if this frame is at full opacity.
    #[must_use]
    pub fn is_fully_opaque(&self) -> bool {
        self.opacity >= 1.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── PipAnchor positioning ─────────────────────────────────────────────────

    #[test]
    fn test_anchor_top_left_position() {
        let pos = PipAnchor::TopLeft.compute_position(1920, 1080, 320, 180, 16, 16);
        assert_eq!(pos, (16, 16));
    }

    #[test]
    fn test_anchor_top_right_position() {
        let pos = PipAnchor::TopRight.compute_position(1920, 1080, 320, 180, 16, 16);
        assert_eq!(pos, (1920 - 320 - 16, 16));
    }

    #[test]
    fn test_anchor_bottom_left_position() {
        let pos = PipAnchor::BottomLeft.compute_position(1920, 1080, 320, 180, 16, 16);
        assert_eq!(pos, (16, 1080 - 180 - 16));
    }

    #[test]
    fn test_anchor_bottom_right_position() {
        let pos = PipAnchor::BottomRight.compute_position(1920, 1080, 320, 180, 16, 16);
        assert_eq!(pos, (1920 - 320 - 16, 1080 - 180 - 16));
    }

    #[test]
    fn test_anchor_custom_position() {
        let anchor = PipAnchor::Custom { x: 100, y: 200 };
        let pos = anchor.compute_position(1920, 1080, 320, 180, 0, 0);
        assert_eq!(pos, (100, 200));
    }

    #[test]
    fn test_anchor_opposite() {
        assert_eq!(PipAnchor::TopLeft.opposite(), PipAnchor::BottomRight);
        assert_eq!(PipAnchor::TopRight.opposite(), PipAnchor::BottomLeft);
        assert_eq!(PipAnchor::BottomLeft.opposite(), PipAnchor::TopRight);
        assert_eq!(PipAnchor::BottomRight.opposite(), PipAnchor::TopLeft);
    }

    #[test]
    fn test_anchor_next_clockwise() {
        assert_eq!(PipAnchor::TopLeft.next_clockwise(), PipAnchor::TopRight);
        assert_eq!(PipAnchor::TopRight.next_clockwise(), PipAnchor::BottomRight);
        assert_eq!(PipAnchor::BottomRight.next_clockwise(), PipAnchor::BottomLeft);
        assert_eq!(PipAnchor::BottomLeft.next_clockwise(), PipAnchor::TopLeft);
    }

    // ── Rect ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_rect_overlap_true() {
        let a = Rect::new(0, 0, 100, 100);
        let b = Rect::new(50, 50, 100, 100);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_rect_overlap_false_adjacent() {
        let a = Rect::new(0, 0, 100, 100);
        let b = Rect::new(100, 0, 100, 100);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_rect_intersection_some() {
        let a = Rect::new(0, 0, 100, 100);
        let b = Rect::new(50, 50, 100, 100);
        let inter = a.intersection(&b).expect("should intersect");
        assert_eq!(inter, Rect::new(50, 50, 50, 50));
    }

    #[test]
    fn test_rect_intersection_none() {
        let a = Rect::new(0, 0, 100, 100);
        let b = Rect::new(200, 0, 100, 100);
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_rect_area() {
        let r = Rect::new(0, 0, 320, 180);
        assert_eq!(r.area(), 320 * 180);
    }

    // ── SignLanguageRegion ────────────────────────────────────────────────────

    #[test]
    fn test_region_bounding_rect_bottom_right() {
        let region = SignLanguageRegion::new(320, 180)
            .with_anchor(PipAnchor::BottomRight)
            .with_margin(16, 16);
        let r = region.bounding_rect(1920, 1080);
        assert_eq!(r.x, 1920 - 320 - 16);
        assert_eq!(r.y, 1080 - 180 - 16);
        assert_eq!(r.width, 320);
        assert_eq!(r.height, 180);
    }

    #[test]
    fn test_region_fits_in_frame() {
        let region = SignLanguageRegion::new(320, 180)
            .with_anchor(PipAnchor::BottomRight)
            .with_margin(16, 16);
        assert!(region.fits_in_frame(1920, 1080));
    }

    #[test]
    fn test_region_does_not_fit_in_tiny_frame() {
        let region = SignLanguageRegion::new(320, 180).with_margin(0, 0);
        assert!(!region.fits_in_frame(200, 100));
    }

    #[test]
    fn test_region_aspect_ratio() {
        let region = SignLanguageRegion::new(320, 180);
        let ar = region.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 0.01, "ar={ar}");
    }

    #[test]
    fn test_region_with_fades() {
        let region = SignLanguageRegion::new(320, 180).with_fades(500, 300);
        assert_eq!(region.fade_in_ms, 500);
        assert_eq!(region.fade_out_ms, 300);
    }

    #[test]
    fn test_region_with_border() {
        let region = SignLanguageRegion::new(320, 180).with_border(2, [255, 255, 255, 255]);
        assert_eq!(region.border_px, 2);
        assert_eq!(region.border_color, [255, 255, 255, 255]);
    }

    // ── VisibilityWindow ──────────────────────────────────────────────────────

    #[test]
    fn test_window_contains() {
        let w = VisibilityWindow::new(1000, 5000);
        assert!(w.contains(1000));
        assert!(w.contains(3000));
        assert!(!w.contains(5000));
        assert!(!w.contains(999));
    }

    #[test]
    fn test_window_duration() {
        let w = VisibilityWindow::new(1000, 4500);
        assert_eq!(w.duration_ms(), 3500);
    }

    // ── Scheduler ────────────────────────────────────────────────────────────

    #[test]
    fn test_scheduler_is_visible_within_window() {
        let region = SignLanguageRegion::new(320, 180);
        let mut scheduler = SignLanguageScheduler::new(region, OverlapPolicy::Ignore);
        scheduler.add_window(1000, 5000);
        assert!(scheduler.is_visible(2000));
        assert!(!scheduler.is_visible(6000));
    }

    #[test]
    fn test_scheduler_window_count() {
        let region = SignLanguageRegion::new(320, 180);
        let mut scheduler = SignLanguageScheduler::new(region, OverlapPolicy::Ignore);
        scheduler.add_window(0, 1000);
        scheduler.add_window(2000, 3000);
        assert_eq!(scheduler.window_count(), 2);
    }

    #[test]
    fn test_scheduler_frame_at_returns_none_outside_window() {
        let region = SignLanguageRegion::new(320, 180);
        let mut scheduler = SignLanguageScheduler::new(region, OverlapPolicy::Ignore);
        scheduler.add_window(1000, 5000);
        assert!(scheduler.frame_at(500, 1920, 1080).is_none());
    }

    #[test]
    fn test_scheduler_frame_at_full_opacity_midpoint() {
        let region = SignLanguageRegion::new(320, 180).with_fades(200, 200);
        let mut scheduler = SignLanguageScheduler::new(region, OverlapPolicy::Ignore);
        scheduler.add_window(0, 10_000);
        let frame = scheduler.frame_at(5000, 1920, 1080).expect("should exist");
        assert!(frame.is_fully_opaque(), "opacity={}", frame.opacity);
    }

    #[test]
    fn test_scheduler_fade_in_opacity() {
        let region = SignLanguageRegion::new(320, 180).with_fades(1000, 1000);
        let mut scheduler = SignLanguageScheduler::new(region, OverlapPolicy::Ignore);
        scheduler.add_window(0, 10_000);
        let frame = scheduler.frame_at(500, 1920, 1080).expect("should exist");
        // At 500ms into a 1000ms fade-in → opacity ≈ 0.5
        assert!((frame.opacity - 0.5).abs() < 0.01, "opacity={}", frame.opacity);
    }

    #[test]
    fn test_scheduler_reposition_avoids_exclusion() {
        // Bottom-right exclusion covers the default bottom-right anchor area
        let region = SignLanguageRegion::new(320, 180)
            .with_anchor(PipAnchor::BottomRight)
            .with_margin(0, 0);
        let mut scheduler = SignLanguageScheduler::new(region, OverlapPolicy::Reposition);
        // Exclusion that covers the entire bottom-right quadrant
        scheduler.add_exclusion(Rect::new(1600, 900, 320, 180));
        scheduler.add_window(0, 5000);

        let frame = scheduler.frame_at(2500, 1920, 1080).expect("should exist");
        // Should have been repositioned away from BottomRight
        assert_ne!(frame.anchor, PipAnchor::BottomRight);
    }

    #[test]
    fn test_scheduler_merged_windows() {
        let region = SignLanguageRegion::new(320, 180);
        let mut scheduler = SignLanguageScheduler::new(region, OverlapPolicy::Ignore);
        scheduler.add_window(0, 3000);
        scheduler.add_window(2000, 5000); // overlaps with first
        scheduler.add_window(8000, 10_000);

        let merged = scheduler.merged_windows();
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].start_ms, 0);
        assert_eq!(merged[0].end_ms, 5000);
        assert_eq!(merged[1].start_ms, 8000);
    }

    #[test]
    fn test_scheduler_total_coverage_ms() {
        let region = SignLanguageRegion::new(320, 180);
        let mut scheduler = SignLanguageScheduler::new(region, OverlapPolicy::Ignore);
        scheduler.add_window(0, 3000);
        scheduler.add_window(5000, 8000);
        assert_eq!(scheduler.total_coverage_ms(), 6000);
    }

    #[test]
    fn test_scheduler_clear_windows() {
        let region = SignLanguageRegion::new(320, 180);
        let mut scheduler = SignLanguageScheduler::new(region, OverlapPolicy::Ignore);
        scheduler.add_window(0, 1000);
        scheduler.clear_windows();
        assert_eq!(scheduler.window_count(), 0);
        assert!(!scheduler.is_visible(500));
    }

    #[test]
    fn test_pip_frame_opacity_flags() {
        let frame_opaque = PipFrame {
            rect: Rect::new(0, 0, 320, 180),
            opacity: 1.0,
            anchor: PipAnchor::BottomRight,
        };
        assert!(frame_opaque.is_fully_opaque());
        assert!(!frame_opaque.is_invisible());

        let frame_invisible = PipFrame {
            rect: Rect::new(0, 0, 320, 180),
            opacity: 0.0,
            anchor: PipAnchor::BottomRight,
        };
        assert!(frame_invisible.is_invisible());
        assert!(!frame_invisible.is_fully_opaque());
    }
}
