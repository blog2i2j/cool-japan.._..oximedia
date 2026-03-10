//! Screen capture hooks, region selection, and FPS control.
//!
//! Provides low-level integration points for hooking into screen capture
//! pipelines, selecting capture regions, and controlling capture frame rates.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::time::Duration;

/// A rectangular region of the screen to capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureRect {
    /// X offset from the screen's left edge in pixels
    pub x: i32,
    /// Y offset from the screen's top edge in pixels
    pub y: i32,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
}

impl CaptureRect {
    /// Create a new capture region.
    #[must_use]
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Create a full-screen region starting at origin.
    #[must_use]
    pub fn full_screen(width: u32, height: u32) -> Self {
        Self::new(0, 0, width, height)
    }

    /// Returns the total pixel count in this region.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Returns true if the given point (px, py) lies within this region.
    #[must_use]
    pub fn contains(&self, px: i32, py: i32) -> bool {
        px >= self.x
            && py >= self.y
            && px < self.x + self.width as i32
            && py < self.y + self.height as i32
    }

    /// Returns the intersection of two regions, or `None` if they don't overlap.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width as i32).min(other.x + other.width as i32);
        let y2 = (self.y + self.height as i32).min(other.y + other.height as i32);
        if x2 > x1 && y2 > y1 {
            Some(Self::new(x1, y1, (x2 - x1) as u32, (y2 - y1) as u32))
        } else {
            None
        }
    }

    /// Scale the region by a factor.
    #[must_use]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            x: (f64::from(self.x) * factor) as i32,
            y: (f64::from(self.y) * factor) as i32,
            width: (f64::from(self.width) * factor) as u32,
            height: (f64::from(self.height) * factor) as u32,
        }
    }
}

/// FPS control target configuration.
#[derive(Debug, Clone, Copy)]
pub struct FpsController {
    target_fps: u32,
    frame_duration: Duration,
}

impl FpsController {
    /// Create a new FPS controller with the given target.
    #[must_use]
    pub fn new(target_fps: u32) -> Self {
        let frame_duration = if target_fps == 0 {
            Duration::from_millis(16)
        } else {
            Duration::from_micros(1_000_000 / u64::from(target_fps))
        };
        Self {
            target_fps,
            frame_duration,
        }
    }

    /// Returns the target frames per second.
    #[must_use]
    pub fn target_fps(&self) -> u32 {
        self.target_fps
    }

    /// Returns the expected duration between frames.
    #[must_use]
    pub fn frame_duration(&self) -> Duration {
        self.frame_duration
    }

    /// Calculate how many frames should have elapsed in the given duration.
    #[must_use]
    pub fn frames_in_duration(&self, elapsed: Duration) -> u64 {
        if self.frame_duration.is_zero() {
            return 0;
        }
        elapsed.as_micros() as u64 / self.frame_duration.as_micros() as u64
    }

    /// Estimate the capture bitrate in bytes per second for uncompressed BGRA.
    #[must_use]
    pub fn estimated_raw_bps(&self, region: &CaptureRect) -> u64 {
        region.pixel_count() * 4 * u64::from(self.target_fps)
    }
}

/// Hook priority ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HookPriority {
    /// Low priority (runs last)
    Low = 0,
    /// Normal priority
    Normal = 50,
    /// High priority (runs first)
    High = 100,
}

/// Type of capture hook event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookEvent {
    /// Fired before a frame is captured
    PreCapture,
    /// Fired after a frame is captured
    PostCapture,
    /// Fired when the capture region changes
    RegionChanged,
    /// Fired when FPS settings change
    FpsChanged,
}

/// A registered capture hook descriptor.
#[derive(Debug, Clone)]
pub struct CaptureHook {
    /// Unique hook identifier
    pub id: String,
    /// Human-readable description
    pub description: String,
    /// Events this hook listens to
    pub events: Vec<HookEvent>,
    /// Priority for ordering
    pub priority: HookPriority,
    /// Whether the hook is currently enabled
    pub enabled: bool,
}

impl CaptureHook {
    /// Create a new capture hook.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        events: Vec<HookEvent>,
        priority: HookPriority,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            events,
            priority,
            enabled: true,
        }
    }

    /// Returns true if this hook handles the given event.
    #[must_use]
    pub fn handles(&self, event: &HookEvent) -> bool {
        self.enabled && self.events.contains(event)
    }
}

/// Registry of capture hooks ordered by priority.
#[derive(Debug, Default)]
pub struct HookRegistry {
    hooks: Vec<CaptureHook>,
}

impl HookRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a hook, maintaining priority order (highest first).
    pub fn register(&mut self, hook: CaptureHook) {
        self.hooks.push(hook);
        self.hooks.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Unregister a hook by id.
    pub fn unregister(&mut self, id: &str) {
        self.hooks.retain(|h| h.id != id);
    }

    /// Enable or disable a hook by id.
    pub fn set_enabled(&mut self, id: &str, enabled: bool) {
        if let Some(hook) = self.hooks.iter_mut().find(|h| h.id == id) {
            hook.enabled = enabled;
        }
    }

    /// Get hooks that handle a given event, in priority order.
    #[must_use]
    pub fn hooks_for_event(&self, event: &HookEvent) -> Vec<&CaptureHook> {
        self.hooks.iter().filter(|h| h.handles(event)).collect()
    }

    /// Count total registered hooks.
    #[must_use]
    pub fn count(&self) -> usize {
        self.hooks.len()
    }

    /// Count enabled hooks.
    #[must_use]
    pub fn enabled_count(&self) -> usize {
        self.hooks.iter().filter(|h| h.enabled).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capture_region_new() {
        let region = CaptureRect::new(10, 20, 800, 600);
        assert_eq!(region.x, 10);
        assert_eq!(region.y, 20);
        assert_eq!(region.width, 800);
        assert_eq!(region.height, 600);
    }

    #[test]
    fn test_capture_region_pixel_count() {
        let region = CaptureRect::full_screen(1920, 1080);
        assert_eq!(region.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_capture_region_contains() {
        let region = CaptureRect::new(100, 100, 200, 200);
        assert!(region.contains(150, 150));
        assert!(region.contains(100, 100));
        assert!(!region.contains(300, 300)); // at border (exclusive)
        assert!(!region.contains(50, 50));
    }

    #[test]
    fn test_capture_region_intersect_overlapping() {
        let a = CaptureRect::new(0, 0, 200, 200);
        let b = CaptureRect::new(100, 100, 200, 200);
        let intersection = a.intersect(&b);
        assert!(intersection.is_some());
        let r = intersection.expect("intersection should succeed");
        assert_eq!(r.x, 100);
        assert_eq!(r.y, 100);
        assert_eq!(r.width, 100);
        assert_eq!(r.height, 100);
    }

    #[test]
    fn test_capture_region_intersect_non_overlapping() {
        let a = CaptureRect::new(0, 0, 100, 100);
        let b = CaptureRect::new(200, 200, 100, 100);
        assert!(a.intersect(&b).is_none());
    }

    #[test]
    fn test_capture_region_scale() {
        let region = CaptureRect::new(0, 0, 1920, 1080);
        let scaled = region.scale(0.5);
        assert_eq!(scaled.width, 960);
        assert_eq!(scaled.height, 540);
    }

    #[test]
    fn test_fps_controller_frame_duration_60fps() {
        let ctrl = FpsController::new(60);
        assert_eq!(ctrl.target_fps(), 60);
        // 1,000,000 / 60 = 16666 µs
        assert_eq!(ctrl.frame_duration().as_micros(), 16666);
    }

    #[test]
    fn test_fps_controller_frames_in_duration() {
        let ctrl = FpsController::new(30);
        let one_second = Duration::from_secs(1);
        // 1,000,000 / 33333 = ~30
        let frames = ctrl.frames_in_duration(one_second);
        assert!(frames >= 29 && frames <= 31, "Expected ~30, got {frames}");
    }

    #[test]
    fn test_fps_controller_estimated_raw_bps() {
        let ctrl = FpsController::new(60);
        let region = CaptureRect::full_screen(1920, 1080);
        // 1920 * 1080 * 4 bytes * 60 fps
        let bps = ctrl.estimated_raw_bps(&region);
        assert_eq!(bps, 1920 * 1080 * 4 * 60);
    }

    #[test]
    fn test_hook_registry_register_and_count() {
        let mut registry = HookRegistry::new();
        let hook = CaptureHook::new(
            "h1",
            "Test hook",
            vec![HookEvent::PostCapture],
            HookPriority::Normal,
        );
        registry.register(hook);
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn test_hook_registry_priority_ordering() {
        let mut registry = HookRegistry::new();
        registry.register(CaptureHook::new(
            "low",
            "Low",
            vec![HookEvent::PreCapture],
            HookPriority::Low,
        ));
        registry.register(CaptureHook::new(
            "high",
            "High",
            vec![HookEvent::PreCapture],
            HookPriority::High,
        ));
        registry.register(CaptureHook::new(
            "normal",
            "Normal",
            vec![HookEvent::PreCapture],
            HookPriority::Normal,
        ));
        let hooks = registry.hooks_for_event(&HookEvent::PreCapture);
        assert_eq!(hooks[0].id, "high");
        assert_eq!(hooks[1].id, "normal");
        assert_eq!(hooks[2].id, "low");
    }

    #[test]
    fn test_hook_registry_disable_hook() {
        let mut registry = HookRegistry::new();
        registry.register(CaptureHook::new(
            "h1",
            "Hook",
            vec![HookEvent::PostCapture],
            HookPriority::Normal,
        ));
        registry.set_enabled("h1", false);
        assert_eq!(registry.enabled_count(), 0);
        let hooks = registry.hooks_for_event(&HookEvent::PostCapture);
        assert!(hooks.is_empty());
    }

    #[test]
    fn test_hook_registry_unregister() {
        let mut registry = HookRegistry::new();
        registry.register(CaptureHook::new(
            "h1",
            "H1",
            vec![HookEvent::PreCapture],
            HookPriority::Normal,
        ));
        registry.register(CaptureHook::new(
            "h2",
            "H2",
            vec![HookEvent::PostCapture],
            HookPriority::Normal,
        ));
        registry.unregister("h1");
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn test_hook_handles_event() {
        let hook = CaptureHook::new(
            "h",
            "desc",
            vec![HookEvent::PreCapture, HookEvent::RegionChanged],
            HookPriority::High,
        );
        assert!(hook.handles(&HookEvent::PreCapture));
        assert!(hook.handles(&HookEvent::RegionChanged));
        assert!(!hook.handles(&HookEvent::PostCapture));
    }

    #[test]
    fn test_hook_priority_ordering() {
        assert!(HookPriority::High > HookPriority::Normal);
        assert!(HookPriority::Normal > HookPriority::Low);
    }
}
