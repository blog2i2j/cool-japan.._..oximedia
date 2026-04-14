//! Template-based lower-third generation for broadcast graphics overlays.
//!
//! Provides ready-made templates (news, documentary, sports) with configurable
//! styles, pixel-accurate rendering, and field validation.

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

/// Where on the frame the lower-third banner is placed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerThirdPosition {
    /// Bottom-left (most common for news)
    BottomLeft,
    /// Horizontally centred at the bottom
    BottomCenter,
    /// Bottom-right
    BottomRight,
    /// Top-left
    TopLeft,
    /// Top-right
    TopRight,
}

// ---------------------------------------------------------------------------
// Style
// ---------------------------------------------------------------------------

/// Visual appearance of a lower-third banner.
#[derive(Debug, Clone)]
pub struct LowerThirdStyle {
    /// RGBA background fill colour.
    pub background_color: [u8; 4],
    /// RGBA colour for the name / primary text line.
    pub name_color: [u8; 4],
    /// RGBA colour for the title / secondary text line.
    pub title_color: [u8; 4],
    /// Font size in pixels for the name line.
    pub font_size_name: u32,
    /// Font size in pixels for the title line.
    pub font_size_title: u32,
    /// Where the banner sits in the frame.
    pub position: LowerThirdPosition,
    /// Width as a fraction of the frame width (0.0–1.0).
    pub width_percent: f32,
    /// Height as a fraction of the frame height (0.0–1.0).
    pub height_percent: f32,
}

impl LowerThirdStyle {
    /// Style preset for a news lower-third: white text on a dark blue background.
    pub fn default_news() -> Self {
        Self {
            background_color: [15, 15, 80, 230],
            name_color: [255, 255, 255, 255],
            title_color: [200, 220, 255, 255],
            font_size_name: 48,
            font_size_title: 32,
            position: LowerThirdPosition::BottomLeft,
            width_percent: 0.55,
            height_percent: 0.12,
        }
    }

    /// Style preset for a documentary lower-third: white text on a semi-transparent black bar.
    pub fn default_documentary() -> Self {
        Self {
            background_color: [0, 0, 0, 160],
            name_color: [255, 255, 255, 255],
            title_color: [200, 200, 200, 255],
            font_size_name: 42,
            font_size_title: 28,
            position: LowerThirdPosition::BottomLeft,
            width_percent: 0.6,
            height_percent: 0.10,
        }
    }
}

// ---------------------------------------------------------------------------
// Template
// ---------------------------------------------------------------------------

/// A reusable lower-third template combining text fields and style.
#[derive(Debug, Clone)]
pub struct LowerThirdTemplate {
    /// Template identifier, e.g. `"news_standard"`.
    pub name: String,
    /// Primary text line (person's name, headline, etc.).
    pub name_field: String,
    /// Secondary text line (job title, caption, etc.).
    pub title_field: String,
    /// Optional third line (e.g. location, subtitle).
    pub subtitle_field: Option<String>,
    /// Optional path to a logo image asset.
    pub logo_path: Option<String>,
    /// Visual style for this template instance.
    pub style: LowerThirdStyle,
}

impl LowerThirdTemplate {
    // ── Preset constructors ──────────────────────────────────────────────────

    /// News-standard template: white text on a dark-blue/opaque background, bottom-left.
    pub fn news_standard() -> Self {
        Self {
            name: "news_standard".to_string(),
            name_field: "Name".to_string(),
            title_field: "Title".to_string(),
            subtitle_field: None,
            logo_path: None,
            style: LowerThirdStyle::default_news(),
        }
    }

    /// Documentary template: white text on a semi-transparent black bar, bottom-left.
    pub fn documentary() -> Self {
        Self {
            name: "documentary".to_string(),
            name_field: "Name".to_string(),
            title_field: "Location / Year".to_string(),
            subtitle_field: None,
            logo_path: None,
            style: LowerThirdStyle::default_documentary(),
        }
    }

    /// Sports template: yellow/white text on a dark background, bottom-centre.
    pub fn sports() -> Self {
        Self {
            name: "sports".to_string(),
            name_field: "Player Name".to_string(),
            title_field: "Team / Position".to_string(),
            subtitle_field: Some("Statistics".to_string()),
            logo_path: None,
            style: LowerThirdStyle {
                background_color: [20, 20, 20, 220],
                name_color: [255, 220, 0, 255],
                title_color: [255, 255, 255, 255],
                font_size_name: 52,
                font_size_title: 34,
                position: LowerThirdPosition::BottomCenter,
                width_percent: 0.50,
                height_percent: 0.13,
            },
        }
    }

    // ── Geometry ─────────────────────────────────────────────────────────────

    /// Compute the pixel rectangle `(x, y, width, height)` of this banner
    /// within a frame of the given dimensions.
    ///
    /// All values are clamped so the rectangle never extends beyond the frame.
    pub fn compute_rect(&self, frame_width: u32, frame_height: u32) -> (u32, u32, u32, u32) {
        let bw = ((frame_width as f32 * self.style.width_percent) as u32).min(frame_width);
        let bh = ((frame_height as f32 * self.style.height_percent) as u32).min(frame_height);

        // Horizontal offset
        let x = match self.style.position {
            LowerThirdPosition::BottomLeft | LowerThirdPosition::TopLeft => 0,
            LowerThirdPosition::BottomCenter => frame_width.saturating_sub(bw) / 2,
            LowerThirdPosition::BottomRight | LowerThirdPosition::TopRight => {
                frame_width.saturating_sub(bw)
            }
        };

        // Vertical offset — place at 85 % down the frame for "bottom" variants
        let y = match self.style.position {
            LowerThirdPosition::BottomLeft
            | LowerThirdPosition::BottomCenter
            | LowerThirdPosition::BottomRight => {
                let target_y = (frame_height as f32 * 0.85) as u32;
                target_y.min(frame_height.saturating_sub(bh))
            }
            LowerThirdPosition::TopLeft | LowerThirdPosition::TopRight => 0,
        };

        (x, y, bw, bh)
    }

    // ── Rendering ────────────────────────────────────────────────────────────

    /// Render this lower-third into a freshly allocated RGBA pixel buffer.
    ///
    /// Returns `(pixels, width, height)`.  Every pixel is 4 bytes (RGBA).
    /// The buffer dimensions match `compute_rect` for the given frame size.
    pub fn render(&self, frame_width: u32, frame_height: u32) -> (Vec<u8>, u32, u32) {
        let (_, _, w, h) = self.compute_rect(frame_width, frame_height);

        if w == 0 || h == 0 {
            return (Vec::new(), 0, 0);
        }

        let bg = self.style.background_color;
        let pixel_count = (w * h) as usize;
        let mut pixels = Vec::with_capacity(pixel_count * 4);

        // Fill the background
        for _ in 0..pixel_count {
            pixels.extend_from_slice(&bg);
        }

        // Paint a thin name-line placeholder bar using the name colour
        // (a real renderer would rasterise glyphs; here we use a solid stripe)
        let name_bar_h = (h / 2).max(1);
        let name_color = self.style.name_color;
        let title_color = self.style.title_color;

        // Name stripe: upper half of the banner
        for row in 4..name_bar_h.saturating_sub(2) {
            let stripe_w = ((w as f32 * 0.6) as u32).min(w);
            for col in 8..stripe_w {
                let off = ((row * w + col) * 4) as usize;
                if off + 3 < pixels.len() {
                    pixels[off] = name_color[0];
                    pixels[off + 1] = name_color[1];
                    pixels[off + 2] = name_color[2];
                    pixels[off + 3] = name_color[3];
                }
            }
        }

        // Title stripe: lower half of the banner
        for row in (name_bar_h + 2)..h.saturating_sub(4) {
            let stripe_w = ((w as f32 * 0.45) as u32).min(w);
            for col in 8..stripe_w {
                let off = ((row * w + col) * 4) as usize;
                if off + 3 < pixels.len() {
                    pixels[off] = title_color[0];
                    pixels[off + 1] = title_color[1];
                    pixels[off + 2] = title_color[2];
                    pixels[off + 3] = title_color[3];
                }
            }
        }

        (pixels, w, h)
    }

    // ── Validation ───────────────────────────────────────────────────────────

    /// Validate that all required text fields are non-empty and style values
    /// are in range.
    ///
    /// Returns `Ok(())` on success or an error message string on failure.
    pub fn validate(&self) -> Result<(), String> {
        if self.name_field.trim().is_empty() {
            return Err("name_field must not be empty".to_string());
        }
        if self.title_field.trim().is_empty() {
            return Err("title_field must not be empty".to_string());
        }
        if self.style.width_percent <= 0.0 || self.style.width_percent > 1.0 {
            return Err(format!(
                "width_percent must be in (0, 1], got {}",
                self.style.width_percent
            ));
        }
        if self.style.height_percent <= 0.0 || self.style.height_percent > 1.0 {
            return Err(format!(
                "height_percent must be in (0, 1], got {}",
                self.style.height_percent
            ));
        }
        if self.style.font_size_name == 0 {
            return Err("font_size_name must be > 0".to_string());
        }
        if self.style.font_size_title == 0 {
            return Err("font_size_title must be > 0".to_string());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LowerThird — combines a template with explicit x/y/width/duration fields
// ---------------------------------------------------------------------------

/// Style parameters exposed at the LowerThird level for quick construction.
///
/// When `bg_color` / `text_color` are provided the underlying
/// `LowerThirdStyle` fields are overridden.
#[derive(Debug, Clone)]
pub struct LowerThirdSimpleStyle {
    /// RGBA background fill colour.
    pub bg_color: [u8; 4],
    /// RGBA foreground (text) colour.
    pub text_color: [u8; 4],
    /// Font size in pixels.
    pub font_size: u32,
    /// Padding (pixels) around the text inside the banner.
    pub padding: u32,
}

impl Default for LowerThirdSimpleStyle {
    fn default() -> Self {
        Self {
            bg_color: [15, 15, 80, 230],
            text_color: [255, 255, 255, 255],
            font_size: 48,
            padding: 8,
        }
    }
}

/// A lower-third ready for on-air playout, combining a template with
/// layout overrides and a display duration.
#[derive(Debug, Clone)]
pub struct LowerThird {
    /// Underlying template (carries name/title fields + detailed style).
    pub template: LowerThirdTemplate,
    /// Simple style that can override the template's colours / font.
    pub style: LowerThirdSimpleStyle,
    /// Horizontal position as a fraction of the frame width (0.0–1.0).
    pub x: f32,
    /// Vertical position as a fraction of the frame height (0.0–1.0).
    pub y: f32,
    /// Width override as a fraction of the frame width (0.0–1.0).
    /// If zero the template's own `width_percent` is used.
    pub width: f32,
    /// How long this lower-third should remain on screen (milliseconds).
    pub duration_ms: u32,
}

impl LowerThird {
    /// Construct a lower-third from a template, overriding simple style fields.
    pub fn new(
        template: LowerThirdTemplate,
        style: LowerThirdSimpleStyle,
        x: f32,
        y: f32,
        width: f32,
        duration_ms: u32,
    ) -> Self {
        Self {
            template,
            style,
            x,
            y,
            width,
            duration_ms,
        }
    }

    /// Render the lower-third as an ASCII-art string representation.
    ///
    /// Format: a box whose top/bottom borders are drawn with `─` / `═` chars
    /// and whose body shows `[ name_field | title_field ]`.
    ///
    /// Example for name="Alice Smith" and title="Senior Reporter":
    /// ```text
    /// ╔══════════════════════════════╗
    /// ║ Alice Smith | Senior Reporter║
    /// ╚══════════════════════════════╝
    /// ```
    pub fn render_to_text(&self) -> String {
        let name = &self.template.name_field;
        let title = &self.template.title_field;
        // Inner content: " name | title "
        let inner = format!(" {} | {} ", name, title);
        let inner_len = inner.chars().count();

        // Top border
        let top_fill: String = "═".repeat(inner_len);
        let top = format!("╔{}╗", top_fill);

        // Middle row
        let middle = format!("║{}║", inner);

        // Optional subtitle row
        let subtitle_row = self.template.subtitle_field.as_ref().map(|sub| {
            let sub_inner = format!(" {} ", sub);
            let pad = if sub_inner.chars().count() < inner_len {
                " ".repeat(inner_len - sub_inner.chars().count())
            } else {
                String::new()
            };
            format!("║{}{}║", sub_inner, pad)
        });

        // Bottom border
        let bot_fill: String = "═".repeat(inner_len);
        let bot = format!("╚{}╝", bot_fill);

        let mut lines = vec![top, middle];
        if let Some(sub_line) = subtitle_row {
            lines.push(sub_line);
        }
        lines.push(bot);
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// LowerThirdQueue — scheduled display queue
// ---------------------------------------------------------------------------

/// A scheduled queue of lower-thirds, each associated with a display
/// timestamp in milliseconds.
#[derive(Debug, Default)]
pub struct LowerThirdQueue {
    /// `(lower_third, display_at_ms)` pairs in insertion order.
    pub items: std::collections::VecDeque<(LowerThird, u64)>,
}

impl LowerThirdQueue {
    /// Create a new empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a lower-third to the queue with a scheduled display time.
    pub fn add(&mut self, lt: LowerThird, display_at_ms: u64) {
        self.items.push_back((lt, display_at_ms));
    }

    /// Return a reference to the lower-third that should be on screen at
    /// `now_ms`.
    ///
    /// A lower-third is active when:
    ///   `display_at_ms <= now_ms < display_at_ms + duration_ms`
    ///
    /// If multiple items overlap (should not happen in normal usage) the
    /// earliest scheduled one is returned.
    pub fn current_item(&self, now_ms: u64) -> Option<&LowerThird> {
        for (lt, display_at_ms) in &self.items {
            let end_ms = display_at_ms.saturating_add(u64::from(lt.duration_ms));
            if *display_at_ms <= now_ms && now_ms < end_ms {
                return Some(lt);
            }
        }
        None
    }

    /// Remove all items whose display window has fully elapsed by `now_ms`.
    pub fn prune_expired(&mut self, now_ms: u64) {
        self.items.retain(|(lt, display_at_ms)| {
            let end_ms = display_at_ms.saturating_add(u64::from(lt.duration_ms));
            now_ms < end_ms
        });
    }

    /// Number of items in the queue.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_news_standard_constructor() {
        let t = LowerThirdTemplate::news_standard();
        assert_eq!(t.name, "news_standard");
        assert_eq!(t.style.position, LowerThirdPosition::BottomLeft);
        assert!(t.validate().is_ok());
    }

    #[test]
    fn test_documentary_constructor() {
        let t = LowerThirdTemplate::documentary();
        assert_eq!(t.name, "documentary");
        assert_eq!(t.style.position, LowerThirdPosition::BottomLeft);
        assert!(t.validate().is_ok());
    }

    #[test]
    fn test_sports_constructor() {
        let t = LowerThirdTemplate::sports();
        assert_eq!(t.name, "sports");
        assert_eq!(t.style.position, LowerThirdPosition::BottomCenter);
        assert!(t.subtitle_field.is_some());
        assert!(t.validate().is_ok());
    }

    #[test]
    fn test_compute_rect_within_frame_bounds() {
        let t = LowerThirdTemplate::news_standard();
        let (x, y, w, h) = t.compute_rect(1920, 1080);
        assert!(x + w <= 1920, "banner right edge must be within frame width");
        assert!(y + h <= 1080, "banner bottom edge must be within frame height");
        assert!(w > 0);
        assert!(h > 0);
    }

    #[test]
    fn test_compute_rect_sports_bottom_center() {
        let t = LowerThirdTemplate::sports();
        let (x, _y, w, _h) = t.compute_rect(1920, 1080);
        // Centre offset: (1920 - w) / 2
        let expected_x = (1920 - w) / 2;
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_render_returns_correct_dimensions() {
        let t = LowerThirdTemplate::news_standard();
        let (pixels, w, h) = t.render(1920, 1080);
        assert_eq!(pixels.len(), (w * h * 4) as usize);
        assert!(w > 0);
        assert!(h > 0);
    }

    #[test]
    fn test_render_pixel_count() {
        let t = LowerThirdTemplate::documentary();
        let (pixels, w, h) = t.render(1280, 720);
        assert_eq!(pixels.len(), (w * h * 4) as usize);
    }

    #[test]
    fn test_validate_passes_on_valid_template() {
        let t = LowerThirdTemplate::news_standard();
        assert!(t.validate().is_ok());
    }

    #[test]
    fn test_validate_fails_on_empty_name_field() {
        let mut t = LowerThirdTemplate::news_standard();
        t.name_field = "   ".to_string();
        let result = t.validate();
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("name_field"));
    }

    #[test]
    fn test_validate_fails_on_empty_title_field() {
        let mut t = LowerThirdTemplate::news_standard();
        t.title_field = String::new();
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_validate_fails_on_invalid_width_percent() {
        let mut t = LowerThirdTemplate::news_standard();
        t.style.width_percent = 1.5;
        assert!(t.validate().is_err());
    }

    // ── LowerThird render_to_text ─────────────────────────────────────────────

    fn make_lower_third(name: &str, title: &str, duration_ms: u32) -> LowerThird {
        let mut tmpl = LowerThirdTemplate::news_standard();
        tmpl.name_field = name.to_string();
        tmpl.title_field = title.to_string();
        LowerThird::new(
            tmpl,
            LowerThirdSimpleStyle::default(),
            0.0,
            0.85,
            0.55,
            duration_ms,
        )
    }

    #[test]
    fn test_render_to_text_contains_name_and_title() {
        let lt = make_lower_third("Alice Smith", "Senior Reporter", 5000);
        let text = lt.render_to_text();
        assert!(text.contains("Alice Smith"), "must contain name field");
        assert!(text.contains("Senior Reporter"), "must contain title field");
    }

    #[test]
    fn test_render_to_text_has_border_chars() {
        let lt = make_lower_third("Bob", "Director", 3000);
        let text = lt.render_to_text();
        assert!(text.contains('╔'), "top-left corner");
        assert!(text.contains('╗'), "top-right corner");
        assert!(text.contains('╚'), "bottom-left corner");
        assert!(text.contains('╝'), "bottom-right corner");
    }

    #[test]
    fn test_render_to_text_separator_pipe() {
        let lt = make_lower_third("Jane", "Producer", 4000);
        let text = lt.render_to_text();
        assert!(text.contains('|'), "pipe separator between name and title");
    }

    #[test]
    fn test_render_to_text_multiline() {
        let lt = make_lower_third("Name", "Title", 2000);
        let rendered = lt.render_to_text();
        let lines: Vec<&str> = rendered.lines().collect();
        // At minimum: top border + content + bottom border = 3 lines.
        assert!(lines.len() >= 3);
    }

    // ── LowerThirdQueue ───────────────────────────────────────────────────────

    #[test]
    fn test_queue_current_item_active() {
        let mut q = LowerThirdQueue::new();
        let lt = make_lower_third("Breaking", "News", 5000);
        q.add(lt, 1000);
        // now_ms=3000 is inside [1000, 6000)
        assert!(q.current_item(3000).is_some());
    }

    #[test]
    fn test_queue_current_item_not_yet_active() {
        let mut q = LowerThirdQueue::new();
        let lt = make_lower_third("Future", "Item", 2000);
        q.add(lt, 5000);
        // now_ms=4999 is before display_at_ms=5000
        assert!(q.current_item(4999).is_none());
    }

    #[test]
    fn test_queue_current_item_expired() {
        let mut q = LowerThirdQueue::new();
        let lt = make_lower_third("Past", "Item", 1000); // duration = 1 000 ms
        q.add(lt, 0); // active [0, 1000)
        // now_ms=1000 is no longer active (end is exclusive)
        assert!(q.current_item(1000).is_none());
    }

    #[test]
    fn test_queue_add_and_len() {
        let mut q = LowerThirdQueue::new();
        assert!(q.is_empty());
        q.add(make_lower_third("A", "B", 2000), 0);
        q.add(make_lower_third("C", "D", 2000), 10_000);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_queue_prune_expired() {
        let mut q = LowerThirdQueue::new();
        q.add(make_lower_third("Old", "Item", 500), 0); // expires at 500 ms
        q.add(make_lower_third("New", "Item", 5000), 2000); // active until 7000 ms
        q.prune_expired(1000); // 1000 >= 500 → old item pruned
        assert_eq!(q.len(), 1);
    }
}
