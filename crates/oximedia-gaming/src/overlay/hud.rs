//! HUD (Heads-Up Display) elements for gaming overlays.
//!
//! Provides stats display, HUD layouts, and notification banner management
//! for real-time gaming overlays.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::VecDeque;
use std::time::Duration;

/// A 2D position on the overlay canvas (normalized 0.0–1.0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormPos {
    /// Horizontal position (0.0 = left, 1.0 = right)
    pub x: f32,
    /// Vertical position (0.0 = top, 1.0 = bottom)
    pub y: f32,
}

impl NormPos {
    /// Create a new normalized position.
    #[must_use]
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x: x.clamp(0.0, 1.0),
            y: y.clamp(0.0, 1.0),
        }
    }

    /// Top-left corner.
    #[must_use]
    pub fn top_left() -> Self {
        Self::new(0.0, 0.0)
    }

    /// Top-right corner.
    #[must_use]
    pub fn top_right() -> Self {
        Self::new(1.0, 0.0)
    }

    /// Bottom-left corner.
    #[must_use]
    pub fn bottom_left() -> Self {
        Self::new(0.0, 1.0)
    }

    /// Bottom-right corner.
    #[must_use]
    pub fn bottom_right() -> Self {
        Self::new(1.0, 1.0)
    }

    /// Center.
    #[must_use]
    pub fn center() -> Self {
        Self::new(0.5, 0.5)
    }
}

/// RGBA color for HUD elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    /// Red channel
    pub r: u8,
    /// Green channel
    pub g: u8,
    /// Blue channel
    pub b: u8,
    /// Alpha channel
    pub a: u8,
}

impl Color {
    /// Create a new color.
    #[must_use]
    pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Opaque white.
    #[must_use]
    pub fn white() -> Self {
        Self::rgba(255, 255, 255, 255)
    }

    /// Opaque black.
    #[must_use]
    pub fn black() -> Self {
        Self::rgba(0, 0, 0, 255)
    }

    /// Semi-transparent black for backgrounds.
    #[must_use]
    pub fn bg() -> Self {
        Self::rgba(0, 0, 0, 160)
    }

    /// Convert to a 32-bit packed ARGB value.
    #[must_use]
    pub fn to_argb32(self) -> u32 {
        (u32::from(self.a) << 24)
            | (u32::from(self.r) << 16)
            | (u32::from(self.g) << 8)
            | u32::from(self.b)
    }
}

/// A stat entry displayed in the HUD.
#[derive(Debug, Clone)]
pub struct StatEntry {
    /// Display label
    pub label: String,
    /// Current value string
    pub value: String,
    /// Text color
    pub color: Color,
}

impl StatEntry {
    /// Create a new stat entry.
    #[must_use]
    pub fn new(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
            color: Color::white(),
        }
    }

    /// Set a custom text color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
}

/// A stats panel shown on the HUD.
#[derive(Debug)]
pub struct StatsPanel {
    /// Panel position on screen
    pub position: NormPos,
    /// Background color
    pub background: Color,
    /// Whether the panel is visible
    pub visible: bool,
    stats: Vec<StatEntry>,
}

impl StatsPanel {
    /// Create a new stats panel at the given position.
    #[must_use]
    pub fn new(position: NormPos) -> Self {
        Self {
            position,
            background: Color::bg(),
            visible: true,
            stats: Vec::new(),
        }
    }

    /// Add or update a stat entry by label.
    pub fn set_stat(&mut self, label: impl Into<String>, value: impl Into<String>) {
        let label = label.into();
        let value = value.into();
        if let Some(entry) = self.stats.iter_mut().find(|e| e.label == label) {
            entry.value = value;
        } else {
            self.stats.push(StatEntry::new(label, value));
        }
    }

    /// Get a stat value by label.
    #[must_use]
    pub fn get_stat(&self, label: &str) -> Option<&str> {
        self.stats
            .iter()
            .find(|e| e.label == label)
            .map(|e| e.value.as_str())
    }

    /// Remove a stat by label.
    pub fn remove_stat(&mut self, label: &str) {
        self.stats.retain(|e| e.label != label);
    }

    /// Count the number of stats.
    #[must_use]
    pub fn stat_count(&self) -> usize {
        self.stats.len()
    }

    /// All stat entries.
    #[must_use]
    pub fn entries(&self) -> &[StatEntry] {
        &self.stats
    }
}

/// Priority level for notification banners.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BannerPriority {
    /// Informational
    Info = 0,
    /// Warning level
    Warning = 1,
    /// Critical / urgent
    Critical = 2,
}

/// A notification banner displayed on the HUD.
#[derive(Debug, Clone)]
pub struct Banner {
    /// Unique banner id
    pub id: u64,
    /// Message text
    pub message: String,
    /// Priority level
    pub priority: BannerPriority,
    /// How long to display the banner
    pub duration: Duration,
    /// Background color
    pub color: Color,
}

impl Banner {
    /// Create a new banner.
    #[must_use]
    pub fn new(
        id: u64,
        message: impl Into<String>,
        priority: BannerPriority,
        duration: Duration,
    ) -> Self {
        let color = match priority {
            BannerPriority::Info => Color::rgba(30, 130, 200, 220),
            BannerPriority::Warning => Color::rgba(220, 160, 30, 220),
            BannerPriority::Critical => Color::rgba(200, 30, 30, 220),
        };
        Self {
            id,
            message: message.into(),
            priority,
            duration,
            color,
        }
    }
}

/// Manages a queue of notification banners.
#[derive(Debug)]
pub struct BannerQueue {
    queue: VecDeque<Banner>,
    next_id: u64,
    max_size: usize,
}

impl Default for BannerQueue {
    fn default() -> Self {
        Self::new(10)
    }
}

impl BannerQueue {
    /// Create a new banner queue.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            next_id: 1,
            max_size,
        }
    }

    /// Push a new banner onto the queue.
    ///
    /// If the queue is full, the lowest priority banner is evicted.
    pub fn push(
        &mut self,
        message: impl Into<String>,
        priority: BannerPriority,
        duration: Duration,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let banner = Banner::new(id, message, priority, duration);

        if self.queue.len() >= self.max_size {
            // Remove the oldest lowest-priority banner
            if let Some(pos) = self
                .queue
                .iter()
                .position(|b| b.priority == BannerPriority::Info)
            {
                self.queue.remove(pos);
            } else {
                self.queue.pop_front();
            }
        }
        self.queue.push_back(banner);
        id
    }

    /// Remove the next banner to display.
    pub fn pop(&mut self) -> Option<Banner> {
        // Return highest priority first
        if let Some(pos) = self
            .queue
            .iter()
            .enumerate()
            .max_by_key(|(_, b)| b.priority)
            .map(|(i, _)| i)
        {
            self.queue.remove(pos)
        } else {
            None
        }
    }

    /// Dismiss a banner by id.
    pub fn dismiss(&mut self, id: u64) {
        self.queue.retain(|b| b.id != id);
    }

    /// Number of queued banners.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns true if no banners are queued.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// The complete HUD overlay state.
#[derive(Debug)]
pub struct HudOverlay {
    /// FPS stats panel
    pub fps_panel: StatsPanel,
    /// Network stats panel
    pub network_panel: StatsPanel,
    /// Notification banners
    pub banners: BannerQueue,
    /// Whether the entire HUD is visible
    pub visible: bool,
}

impl Default for HudOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl HudOverlay {
    /// Create a new HUD overlay with default layout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fps_panel: StatsPanel::new(NormPos::top_left()),
            network_panel: StatsPanel::new(NormPos::top_right()),
            banners: BannerQueue::new(5),
            visible: true,
        }
    }

    /// Update FPS stats.
    pub fn update_fps(&mut self, fps: f64, frame_time_ms: f64) {
        self.fps_panel.set_stat("FPS", format!("{fps:.1}"));
        self.fps_panel
            .set_stat("Frame", format!("{frame_time_ms:.2}ms"));
    }

    /// Update network stats.
    pub fn update_network(&mut self, bitrate_kbps: u32, dropped_frames: u64) {
        self.network_panel
            .set_stat("Bitrate", format!("{bitrate_kbps} kbps"));
        self.network_panel
            .set_stat("Dropped", dropped_frames.to_string());
    }

    /// Toggle HUD visibility.
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_pos_clamp() {
        let pos = NormPos::new(1.5, -0.5);
        assert_eq!(pos.x, 1.0);
        assert_eq!(pos.y, 0.0);
    }

    #[test]
    fn test_norm_pos_corners() {
        assert_eq!(NormPos::top_left(), NormPos::new(0.0, 0.0));
        assert_eq!(NormPos::bottom_right(), NormPos::new(1.0, 1.0));
        assert_eq!(NormPos::center(), NormPos::new(0.5, 0.5));
    }

    #[test]
    fn test_color_to_argb32() {
        let c = Color::rgba(255, 0, 128, 200);
        let argb = c.to_argb32();
        assert_eq!((argb >> 24) & 0xFF, 200);
        assert_eq!((argb >> 16) & 0xFF, 255);
        assert_eq!((argb >> 8) & 0xFF, 0);
        assert_eq!(argb & 0xFF, 128);
    }

    #[test]
    fn test_stats_panel_set_and_get() {
        let mut panel = StatsPanel::new(NormPos::top_left());
        panel.set_stat("FPS", "60.0");
        assert_eq!(panel.get_stat("FPS"), Some("60.0"));
        assert!(panel.get_stat("Missing").is_none());
    }

    #[test]
    fn test_stats_panel_update_existing() {
        let mut panel = StatsPanel::new(NormPos::top_left());
        panel.set_stat("FPS", "30");
        panel.set_stat("FPS", "60");
        assert_eq!(panel.stat_count(), 1);
        assert_eq!(panel.get_stat("FPS"), Some("60"));
    }

    #[test]
    fn test_stats_panel_remove_stat() {
        let mut panel = StatsPanel::new(NormPos::top_left());
        panel.set_stat("FPS", "60");
        panel.set_stat("Latency", "10ms");
        panel.remove_stat("FPS");
        assert_eq!(panel.stat_count(), 1);
        assert!(panel.get_stat("FPS").is_none());
    }

    #[test]
    fn test_banner_priority_ordering() {
        assert!(BannerPriority::Critical > BannerPriority::Warning);
        assert!(BannerPriority::Warning > BannerPriority::Info);
    }

    #[test]
    fn test_banner_queue_push_and_pop() {
        let mut queue = BannerQueue::new(5);
        queue.push("info message", BannerPriority::Info, Duration::from_secs(3));
        queue.push(
            "critical!",
            BannerPriority::Critical,
            Duration::from_secs(5),
        );
        let first = queue.pop().expect("queue should have element");
        assert_eq!(first.priority, BannerPriority::Critical);
    }

    #[test]
    fn test_banner_queue_dismiss() {
        let mut queue = BannerQueue::new(5);
        let id = queue.push("msg", BannerPriority::Info, Duration::from_secs(3));
        queue.dismiss(id);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_banner_queue_max_size() {
        let mut queue = BannerQueue::new(3);
        for i in 0..5u32 {
            queue.push(
                format!("msg {i}"),
                BannerPriority::Info,
                Duration::from_secs(1),
            );
        }
        assert_eq!(queue.len(), 3);
    }

    #[test]
    fn test_hud_overlay_update_fps() {
        let mut hud = HudOverlay::new();
        hud.update_fps(59.9, 16.7);
        assert_eq!(hud.fps_panel.get_stat("FPS"), Some("59.9"));
        assert_eq!(hud.fps_panel.get_stat("Frame"), Some("16.70ms"));
    }

    #[test]
    fn test_hud_overlay_update_network() {
        let mut hud = HudOverlay::new();
        hud.update_network(6000, 2);
        assert_eq!(hud.network_panel.get_stat("Bitrate"), Some("6000 kbps"));
        assert_eq!(hud.network_panel.get_stat("Dropped"), Some("2"));
    }

    #[test]
    fn test_hud_overlay_toggle_visibility() {
        let mut hud = HudOverlay::new();
        assert!(hud.visible);
        hud.toggle();
        assert!(!hud.visible);
        hud.toggle();
        assert!(hud.visible);
    }

    #[test]
    fn test_stat_entry_with_color() {
        let entry = StatEntry::new("FPS", "60").with_color(Color::rgba(255, 100, 0, 255));
        assert_eq!(entry.color.r, 255);
        assert_eq!(entry.color.g, 100);
    }

    #[test]
    fn test_banner_colors_by_priority() {
        let info = Banner::new(1, "info", BannerPriority::Info, Duration::from_secs(1));
        let critical = Banner::new(2, "crit", BannerPriority::Critical, Duration::from_secs(1));
        // Critical should have a reddish background
        assert!(critical.color.r > critical.color.b);
        // Info should have a bluish background
        assert!(info.color.b > info.color.r);
    }
}
