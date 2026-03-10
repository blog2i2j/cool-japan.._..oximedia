//! News/data ticker graphics for broadcast overlays.
//!
//! Provides scrolling ticker rendering with priority support, configurable
//! colors and scroll speed, and a queue for managing ticker items.

use std::collections::VecDeque;

/// A single item in the news ticker.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TickerItem {
    /// Text content of the ticker item.
    pub text: String,
    /// Optional category label (e.g. "BREAKING", "SPORTS").
    pub category: Option<String>,
    /// Priority level (higher = shown sooner). Range 0–255.
    pub priority: u8,
}

impl TickerItem {
    /// Create a new ticker item.
    #[allow(dead_code)]
    pub fn new(text: impl Into<String>, category: Option<String>, priority: u8) -> Self {
        Self {
            text: text.into(),
            category,
            priority,
        }
    }

    /// Format the item as a display string including the category prefix.
    #[allow(dead_code)]
    pub fn formatted(&self, separator: &str) -> String {
        match &self.category {
            Some(cat) => format!("[{}] {}{}", cat, self.text, separator),
            None => format!("{}{}", self.text, separator),
        }
    }
}

/// Position of the ticker strip on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TickerPosition {
    /// Ticker appears at the bottom of the frame.
    Bottom,
    /// Ticker appears at the top of the frame.
    Top,
}

/// Configuration for the ticker renderer.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TickerConfig {
    /// Scroll speed in pixels per second.
    pub scroll_speed_pps: f32,
    /// Background fill color as RGBA.
    pub bg_color: [u8; 4],
    /// Text color as RGBA.
    pub text_color: [u8; 4],
    /// Separator string inserted between items.
    pub separator: String,
    /// Height of the ticker strip in pixels.
    pub height_px: u32,
    /// Screen position (top or bottom).
    pub position: TickerPosition,
}

impl Default for TickerConfig {
    fn default() -> Self {
        Self {
            scroll_speed_pps: 120.0,
            bg_color: [20, 20, 80, 230],
            text_color: [255, 255, 255, 255],
            separator: "  •  ".to_string(),
            height_px: 48,
            position: TickerPosition::Bottom,
        }
    }
}

/// Current scroll state of the ticker.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TickerState {
    /// List of items to display.
    pub items: Vec<TickerItem>,
    /// Index of the item currently being scrolled.
    pub current_item_idx: usize,
    /// Current scroll offset in pixels (positive = scrolled left).
    pub scroll_offset_px: f32,
}

impl TickerState {
    /// Create a new ticker state with the given items.
    #[allow(dead_code)]
    pub fn new(items: Vec<TickerItem>) -> Self {
        Self {
            items,
            current_item_idx: 0,
            scroll_offset_px: 0.0,
        }
    }

    /// Advance the scroll by `dt_secs` seconds at the speed specified in `config`.
    ///
    /// `item_width_px` is the rendered pixel width of the current item.
    ///
    /// Returns `true` if the ticker has advanced to a new item.
    #[allow(dead_code)]
    pub fn advance(&mut self, dt_secs: f32, item_width_px: f32, speed_pps: f32) -> bool {
        if self.items.is_empty() {
            return false;
        }

        self.scroll_offset_px += speed_pps * dt_secs;

        if self.scroll_offset_px >= item_width_px {
            self.scroll_offset_px -= item_width_px;
            self.current_item_idx = (self.current_item_idx + 1) % self.items.len();
            true
        } else {
            false
        }
    }

    /// Get the currently active ticker item, if any.
    #[allow(dead_code)]
    pub fn current_item(&self) -> Option<&TickerItem> {
        self.items.get(self.current_item_idx)
    }
}

impl Default for TickerState {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

/// Renderer for the ticker strip.
pub struct TickerRenderer;

impl TickerRenderer {
    /// Render a horizontal RGBA ticker strip.
    ///
    /// Returns a `Vec<u8>` of RGBA pixels with length `width * config.height_px * 4`.
    #[allow(dead_code)]
    pub fn render(_state: &TickerState, config: &TickerConfig, width: u32) -> Vec<u8> {
        let h = config.height_px;
        let total = (width * h * 4) as usize;
        let mut data = vec![0u8; total];

        // Fill background
        for chunk in data.chunks_exact_mut(4) {
            chunk[0] = config.bg_color[0];
            chunk[1] = config.bg_color[1];
            chunk[2] = config.bg_color[2];
            chunk[3] = config.bg_color[3];
        }

        // Draw a thin accent line at the top of the strip
        let accent_row_height = (h / 10).max(2) as usize;
        for row in 0..accent_row_height {
            for col in 0..width as usize {
                let idx = (row * width as usize + col) * 4;
                if idx + 3 < data.len() {
                    // Slightly lighter than bg as accent
                    data[idx] = config.bg_color[0].saturating_add(60);
                    data[idx + 1] = config.bg_color[1].saturating_add(60);
                    data[idx + 2] = config.bg_color[2].saturating_add(60);
                    data[idx + 3] = 255;
                }
            }
        }

        data
    }
}

/// A priority queue for ticker items.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct TickerQueue {
    items: VecDeque<TickerItem>,
}

impl TickerQueue {
    /// Create an empty ticker queue.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            items: VecDeque::new(),
        }
    }

    /// Push a standard item to the back of the queue.
    #[allow(dead_code)]
    pub fn push(&mut self, item: TickerItem) {
        self.items.push_back(item);
    }

    /// Pop the next item from the front of the queue.
    #[allow(dead_code)]
    pub fn pop(&mut self) -> Option<TickerItem> {
        self.items.pop_front()
    }

    /// Insert a breaking-news item at the front of the queue, bypassing priority ordering.
    #[allow(dead_code)]
    pub fn insert_breaking(&mut self, mut item: TickerItem) {
        item.priority = 255;
        self.items.push_front(item);
    }

    /// Returns the number of items currently in the queue.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns true if the queue is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Drain the queue into a Vec, sorted by descending priority.
    #[allow(dead_code)]
    pub fn drain_sorted(&mut self) -> Vec<TickerItem> {
        let mut items: Vec<TickerItem> = self.items.drain(..).collect();
        items.sort_by(|a, b| b.priority.cmp(&a.priority));
        items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ticker_item_formatted_with_category() {
        let item = TickerItem::new("Some news", Some("BREAKING".to_string()), 255);
        let fmt = item.formatted(" | ");
        assert!(fmt.contains("BREAKING"));
        assert!(fmt.contains("Some news"));
    }

    #[test]
    fn test_ticker_item_formatted_without_category() {
        let item = TickerItem::new("Plain text", None, 0);
        let fmt = item.formatted(" • ");
        assert!(!fmt.contains('['));
        assert!(fmt.contains("Plain text"));
    }

    #[test]
    fn test_ticker_config_default() {
        let cfg = TickerConfig::default();
        assert!(cfg.scroll_speed_pps > 0.0);
        assert!(!cfg.separator.is_empty());
        assert!(cfg.height_px > 0);
    }

    #[test]
    fn test_ticker_state_advance_scrolls() {
        let items = vec![TickerItem::new("item1", None, 0)];
        let mut state = TickerState::new(items);
        // Advance less than full item width — should not advance item
        let advanced = state.advance(0.1, 300.0, 120.0);
        assert!(!advanced);
        assert!(state.scroll_offset_px > 0.0);
    }

    #[test]
    fn test_ticker_state_advance_next_item() {
        let items = vec![
            TickerItem::new("item1", None, 0),
            TickerItem::new("item2", None, 0),
        ];
        let mut state = TickerState::new(items);
        // Advance enough to scroll past the full item width
        let advanced = state.advance(10.0, 100.0, 120.0);
        assert!(advanced);
        assert_eq!(state.current_item_idx, 1);
    }

    #[test]
    fn test_ticker_state_wraps_around() {
        let items = vec![TickerItem::new("A", None, 0), TickerItem::new("B", None, 0)];
        let mut state = TickerState::new(items);
        state.advance(10.0, 50.0, 120.0);
        state.advance(10.0, 50.0, 120.0);
        // Should have wrapped back to 0
        assert_eq!(state.current_item_idx, 0);
    }

    #[test]
    fn test_ticker_state_current_item() {
        let items = vec![TickerItem::new("hello", None, 5)];
        let state = TickerState::new(items);
        assert!(state.current_item().is_some());
        assert_eq!(
            state
                .current_item()
                .expect("current_item should succeed")
                .text,
            "hello"
        );
    }

    #[test]
    fn test_ticker_render_size() {
        let state = TickerState::default();
        let config = TickerConfig {
            height_px: 48,
            ..TickerConfig::default()
        };
        let data = TickerRenderer::render(&state, &config, 1920);
        assert_eq!(data.len(), (1920 * 48 * 4) as usize);
    }

    #[test]
    fn test_ticker_render_has_background() {
        let state = TickerState::default();
        let config = TickerConfig {
            bg_color: [50, 50, 200, 255],
            height_px: 48,
            ..TickerConfig::default()
        };
        let data = TickerRenderer::render(&state, &config, 100);
        // Last row pixels should be background color
        let row_offset = (47 * 100 * 4) as usize;
        assert_eq!(data[row_offset], 50);
        assert_eq!(data[row_offset + 2], 200);
    }

    #[test]
    fn test_ticker_queue_push_pop() {
        let mut q = TickerQueue::new();
        q.push(TickerItem::new("A", None, 0));
        q.push(TickerItem::new("B", None, 0));
        assert_eq!(q.len(), 2);
        let item = q.pop().expect("item should be valid");
        assert_eq!(item.text, "A");
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_ticker_queue_insert_breaking() {
        let mut q = TickerQueue::new();
        q.push(TickerItem::new("Normal", None, 0));
        q.insert_breaking(TickerItem::new("BREAKING", Some("BREAKING".to_string()), 0));
        // Breaking should be at front
        let first = q.pop().expect("first should be valid");
        assert_eq!(first.priority, 255);
        assert_eq!(first.text, "BREAKING");
    }

    #[test]
    fn test_ticker_queue_drain_sorted() {
        let mut q = TickerQueue::new();
        q.push(TickerItem::new("low", None, 10));
        q.push(TickerItem::new("high", None, 200));
        q.push(TickerItem::new("mid", None, 100));
        let sorted = q.drain_sorted();
        assert_eq!(sorted[0].priority, 200);
        assert_eq!(sorted[1].priority, 100);
        assert_eq!(sorted[2].priority, 10);
    }

    #[test]
    fn test_ticker_queue_is_empty() {
        let q = TickerQueue::new();
        assert!(q.is_empty());
    }
}
