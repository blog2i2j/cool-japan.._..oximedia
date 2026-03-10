#![allow(dead_code)]
//! Viewport and scissor-rectangle management for GPU render passes.

/// The origin convention for viewport coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ViewportOrigin {
    /// Y-axis points downward (Direct3D / Vulkan surface convention).
    #[default]
    TopLeft,
    /// Y-axis points upward (OpenGL convention).
    BottomLeft,
}

impl ViewportOrigin {
    /// Returns `true` if this is the top-left (D3D/Vulkan) convention.
    #[must_use]
    pub fn is_top_left(&self) -> bool {
        matches!(self, Self::TopLeft)
    }
}

/// A rectangular viewport region mapping NDC to window coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Viewport {
    /// X offset in pixels from the window origin.
    pub x: f32,
    /// Y offset in pixels from the window origin.
    pub y: f32,
    /// Width in pixels.
    pub width: f32,
    /// Height in pixels.
    pub height: f32,
    /// Near depth range (typically 0.0).
    pub min_depth: f32,
    /// Far depth range (typically 1.0).
    pub max_depth: f32,
    /// Coordinate origin convention.
    pub origin: ViewportOrigin,
}

impl Viewport {
    /// Create a simple full-window viewport with standard depth range.
    #[must_use]
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width,
            height,
            min_depth: 0.0,
            max_depth: 1.0,
            origin: ViewportOrigin::TopLeft,
        }
    }

    /// Create a viewport with an explicit offset and depth range.
    #[must_use]
    pub fn with_offset(
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Self {
        Self {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
            origin: ViewportOrigin::TopLeft,
        }
    }

    /// Aspect ratio (width / height). Returns `f32::INFINITY` if height is zero.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0.0 {
            return f32::INFINITY;
        }
        self.width / self.height
    }

    /// Returns `true` if the viewport is taller than it is wide.
    #[must_use]
    pub fn is_portrait(&self) -> bool {
        self.height > self.width
    }

    /// Returns `true` if the viewport is wider than it is tall.
    #[must_use]
    pub fn is_landscape(&self) -> bool {
        self.width > self.height
    }

    /// Returns `true` if the depth range is valid (min < max, both in 0..=1).
    #[must_use]
    pub fn has_valid_depth_range(&self) -> bool {
        self.min_depth >= 0.0 && self.max_depth <= 1.0 && self.min_depth < self.max_depth
    }
}

/// A scissor rectangle that clips rasterisation to a pixel-aligned region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ViewportScissor {
    /// Left edge in pixels.
    pub left: i32,
    /// Top edge in pixels.
    pub top: i32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl ViewportScissor {
    /// Create a scissor from pixel coordinates.
    #[must_use]
    pub fn new(left: i32, top: i32, width: u32, height: u32) -> Self {
        Self {
            left,
            top,
            width,
            height,
        }
    }

    /// Returns `true` if the given pixel coordinate is inside this scissor region.
    #[must_use]
    pub fn clips(&self, px: i32, py: i32) -> bool {
        px >= self.left
            && py >= self.top
            && px < self.left + self.width as i32
            && py < self.top + self.height as i32
    }

    /// Area of the scissor rectangle in pixels.
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Returns `true` if the scissor has non-zero area.
    #[must_use]
    pub fn is_non_empty(&self) -> bool {
        self.width > 0 && self.height > 0
    }
}

/// A stack of viewports for managing nested render passes.
#[derive(Debug, Default)]
pub struct ViewportStack {
    stack: Vec<Viewport>,
}

impl ViewportStack {
    /// Create an empty stack.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a viewport onto the stack.
    pub fn push(&mut self, viewport: Viewport) {
        self.stack.push(viewport);
    }

    /// Pop the top viewport from the stack. Returns `None` if the stack is empty.
    pub fn pop(&mut self) -> Option<Viewport> {
        self.stack.pop()
    }

    /// Borrow the current (top-of-stack) viewport without removing it.
    #[must_use]
    pub fn current(&self) -> Option<&Viewport> {
        self.stack.last()
    }

    /// Stack depth.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Returns `true` when there are no viewports on the stack.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewport_origin_is_top_left() {
        assert!(ViewportOrigin::TopLeft.is_top_left());
        assert!(!ViewportOrigin::BottomLeft.is_top_left());
    }

    #[test]
    fn test_viewport_aspect_ratio() {
        let vp = Viewport::new(1920.0, 1080.0);
        let ar = vp.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 1e-4);
    }

    #[test]
    fn test_viewport_aspect_ratio_zero_height() {
        let vp = Viewport::new(100.0, 0.0);
        assert_eq!(vp.aspect_ratio(), f32::INFINITY);
    }

    #[test]
    fn test_viewport_is_portrait() {
        let vp = Viewport::new(720.0, 1280.0);
        assert!(vp.is_portrait());
        assert!(!vp.is_landscape());
    }

    #[test]
    fn test_viewport_is_landscape() {
        let vp = Viewport::new(1920.0, 1080.0);
        assert!(vp.is_landscape());
        assert!(!vp.is_portrait());
    }

    #[test]
    fn test_viewport_valid_depth_range() {
        let vp = Viewport::new(100.0, 100.0);
        assert!(vp.has_valid_depth_range());
    }

    #[test]
    fn test_viewport_invalid_depth_range() {
        let mut vp = Viewport::new(100.0, 100.0);
        vp.min_depth = 0.9;
        vp.max_depth = 0.5;
        assert!(!vp.has_valid_depth_range());
    }

    #[test]
    fn test_scissor_clips_inside() {
        let s = ViewportScissor::new(10, 10, 100, 100);
        assert!(s.clips(50, 50));
    }

    #[test]
    fn test_scissor_clips_outside() {
        let s = ViewportScissor::new(10, 10, 100, 100);
        assert!(!s.clips(5, 50));
        assert!(!s.clips(50, 5));
        assert!(!s.clips(110, 50));
    }

    #[test]
    fn test_scissor_area() {
        let s = ViewportScissor::new(0, 0, 1920, 1080);
        assert_eq!(s.area(), 1920 * 1080);
    }

    #[test]
    fn test_scissor_is_non_empty() {
        let s = ViewportScissor::new(0, 0, 10, 10);
        assert!(s.is_non_empty());
        let empty = ViewportScissor::new(0, 0, 0, 0);
        assert!(!empty.is_non_empty());
    }

    #[test]
    fn test_viewport_stack_push_pop() {
        let mut stack = ViewportStack::new();
        assert!(stack.is_empty());
        stack.push(Viewport::new(800.0, 600.0));
        stack.push(Viewport::new(400.0, 300.0));
        assert_eq!(stack.depth(), 2);
        let top = stack.pop().expect("stack pop should return a value");
        assert!((top.width - 400.0).abs() < 1e-6);
        assert_eq!(stack.depth(), 1);
    }

    #[test]
    fn test_viewport_stack_current() {
        let mut stack = ViewportStack::new();
        assert!(stack.current().is_none());
        stack.push(Viewport::new(1280.0, 720.0));
        let cur = stack.current().expect("current should return a value");
        assert!((cur.width - 1280.0).abs() < 1e-6);
    }

    #[test]
    fn test_viewport_stack_pop_empty() {
        let mut stack = ViewportStack::new();
        assert!(stack.pop().is_none());
    }
}
