//! Batched draw-call aggregation for reduced per-primitive overhead.
//!
//! Issuing one draw call per primitive (rectangle, circle, text glyph) is
//! expensive at broadcast frame rates when dozens or hundreds of overlay
//! elements are composited every tick.  This module provides a
//! [`DrawBatch`] that accumulates draw commands and sorts them by layer
//! before flushing, allowing the rendering backend to process primitives in
//! a predictable order with minimal state changes.
//!
//! ## Design
//!
//! All draw commands are represented as [`DrawCommand`] variants. A batch is
//! built frame-by-frame and then consumed by a flush that returns the sorted
//! command list.  The flush is deliberately backend-agnostic — the caller
//! decides how to execute each command.
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_graphics::draw_batch::{DrawBatch, DrawCommand, FillRect, BlendMode};
//!
//! let mut batch = DrawBatch::new();
//! batch.push(DrawCommand::Rect(FillRect {
//!     x: 0.0, y: 0.0, width: 200.0, height: 50.0,
//!     color: [0, 0, 0, 200],
//!     layer: 0,
//!     blend: BlendMode::Over,
//! }));
//!
//! let commands = batch.flush();
//! assert!(!commands.is_empty());
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// BlendMode
// ---------------------------------------------------------------------------

/// Porter-Duff compositing / blend mode for a draw command.
///
/// The default mode is [`BlendMode::Over`] (standard alpha compositing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Standard *over* compositing: `src + dst*(1−src_a)`.
    Over,
    /// Multiply: `src * dst`.
    Multiply,
    /// Screen: `1 − (1−src) * (1−dst)`.
    Screen,
    /// Additive: `src + dst` (clamped).
    Add,
    /// Replace (no alpha blending): `src` overwrites `dst`.
    Replace,
}

impl Default for BlendMode {
    fn default() -> Self {
        Self::Over
    }
}

impl fmt::Display for BlendMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Over => write!(f, "over"),
            Self::Multiply => write!(f, "multiply"),
            Self::Screen => write!(f, "screen"),
            Self::Add => write!(f, "add"),
            Self::Replace => write!(f, "replace"),
        }
    }
}

// ---------------------------------------------------------------------------
// Primitive commands
// ---------------------------------------------------------------------------

/// A filled (optionally anti-aliased) rectangle.
#[derive(Debug, Clone)]
pub struct FillRect {
    /// Left edge in pixels.
    pub x: f32,
    /// Top edge in pixels.
    pub y: f32,
    /// Width in pixels.
    pub width: f32,
    /// Height in pixels.
    pub height: f32,
    /// RGBA fill colour.
    pub color: [u8; 4],
    /// Layer (z-order): lower values are drawn first (behind).
    pub layer: i32,
    /// Blend mode.
    pub blend: BlendMode,
}

impl FillRect {
    /// Create a new filled rectangle with default blend mode.
    #[must_use]
    pub fn new(x: f32, y: f32, width: f32, height: f32, color: [u8; 4], layer: i32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            color,
            layer,
            blend: BlendMode::Over,
        }
    }

    /// Area in pixels.
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }
}

/// A filled circle or ellipse.
#[derive(Debug, Clone)]
pub struct FillEllipse {
    /// Center X.
    pub cx: f32,
    /// Center Y.
    pub cy: f32,
    /// Horizontal radius.
    pub rx: f32,
    /// Vertical radius.
    pub ry: f32,
    /// RGBA fill colour.
    pub color: [u8; 4],
    /// Layer.
    pub layer: i32,
    /// Blend mode.
    pub blend: BlendMode,
}

impl FillEllipse {
    /// Create a new filled circle with equal radii.
    #[must_use]
    pub fn circle(cx: f32, cy: f32, radius: f32, color: [u8; 4], layer: i32) -> Self {
        Self {
            cx,
            cy,
            rx: radius,
            ry: radius,
            color,
            layer,
            blend: BlendMode::Over,
        }
    }
}

/// A single glyph blit from a glyph atlas.
#[derive(Debug, Clone)]
pub struct GlyphBlit {
    /// Destination X (pen position + bearing_x).
    pub x: f32,
    /// Destination Y (baseline + bearing_y).
    pub y: f32,
    /// Source UV rectangle in the atlas: `(u_min, v_min, u_max, v_max)`.
    pub uv: (f32, f32, f32, f32),
    /// Glyph width in pixels (destination size).
    pub width: f32,
    /// Glyph height in pixels.
    pub height: f32,
    /// Tint colour (typically the text colour as RGBA).
    pub color: [u8; 4],
    /// Layer.
    pub layer: i32,
    /// Blend mode.
    pub blend: BlendMode,
}

/// A textured quad blit (image / sprite).
#[derive(Debug, Clone)]
pub struct TexturedRect {
    /// Destination left in pixels.
    pub x: f32,
    /// Destination top in pixels.
    pub y: f32,
    /// Destination width.
    pub width: f32,
    /// Destination height.
    pub height: f32,
    /// Source UV rectangle `(u_min, v_min, u_max, v_max)`.
    pub uv: (f32, f32, f32, f32),
    /// Modulation colour (all 255 = no tint).
    pub tint: [u8; 4],
    /// Layer.
    pub layer: i32,
    /// Blend mode.
    pub blend: BlendMode,
}

/// A horizontal or vertical color-stop gradient fill.
#[derive(Debug, Clone)]
pub struct GradientRect {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    pub width: f32,
    /// Height.
    pub height: f32,
    /// Start color (left or top).
    pub color_start: [u8; 4],
    /// End color (right or bottom).
    pub color_end: [u8; 4],
    /// If `true`, gradient runs left→right; otherwise top→bottom.
    pub horizontal: bool,
    /// Layer.
    pub layer: i32,
    /// Blend mode.
    pub blend: BlendMode,
}

// ---------------------------------------------------------------------------
// DrawCommand
// ---------------------------------------------------------------------------

/// A single compositing command inside a [`DrawBatch`].
#[derive(Debug, Clone)]
pub enum DrawCommand {
    /// Filled rectangle.
    Rect(FillRect),
    /// Filled ellipse or circle.
    Ellipse(FillEllipse),
    /// Glyph blit from a texture atlas.
    Glyph(GlyphBlit),
    /// Textured quad / sprite blit.
    Texture(TexturedRect),
    /// Gradient-filled rectangle.
    Gradient(GradientRect),
}

impl DrawCommand {
    /// Layer of this command (used for depth sorting).
    #[must_use]
    pub fn layer(&self) -> i32 {
        match self {
            Self::Rect(r) => r.layer,
            Self::Ellipse(e) => e.layer,
            Self::Glyph(g) => g.layer,
            Self::Texture(t) => t.layer,
            Self::Gradient(g) => g.layer,
        }
    }

    /// Blend mode of this command.
    #[must_use]
    pub fn blend(&self) -> BlendMode {
        match self {
            Self::Rect(r) => r.blend,
            Self::Ellipse(e) => e.blend,
            Self::Glyph(g) => g.blend,
            Self::Texture(t) => t.blend,
            Self::Gradient(g) => g.blend,
        }
    }

    /// Axis-aligned bounding box `(x, y, w, h)` of the command in screen space.
    #[must_use]
    pub fn bounds(&self) -> (f32, f32, f32, f32) {
        match self {
            Self::Rect(r) => (r.x, r.y, r.width, r.height),
            Self::Ellipse(e) => (e.cx - e.rx, e.cy - e.ry, e.rx * 2.0, e.ry * 2.0),
            Self::Glyph(g) => (g.x, g.y, g.width, g.height),
            Self::Texture(t) => (t.x, t.y, t.width, t.height),
            Self::Gradient(g) => (g.x, g.y, g.width, g.height),
        }
    }
}

// ---------------------------------------------------------------------------
// DrawBatch
// ---------------------------------------------------------------------------

/// Accumulates draw commands for a single frame and produces a sorted flush.
///
/// Commands are sorted by layer before being returned, ensuring correct
/// depth ordering without requiring the caller to submit in the right order.
pub struct DrawBatch {
    commands: Vec<DrawCommand>,
}

impl DrawBatch {
    /// Create a new empty draw batch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    /// Create a batch with a pre-allocated capacity hint.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            commands: Vec::with_capacity(cap),
        }
    }

    /// Add a draw command to the batch.
    pub fn push(&mut self, cmd: DrawCommand) {
        self.commands.push(cmd);
    }

    /// Number of commands currently in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Returns `true` if no commands have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Clear all commands without flushing.
    pub fn clear(&mut self) {
        self.commands.clear();
    }

    /// Flush the batch: return all commands sorted by layer (ascending),
    /// preserving insertion order for commands with the same layer.
    ///
    /// The batch is cleared after flushing.
    pub fn flush(&mut self) -> Vec<DrawCommand> {
        let mut out = std::mem::take(&mut self.commands);
        out.sort_by_key(|c| c.layer());
        out
    }

    /// Return a snapshot of the current commands without clearing the batch.
    ///
    /// The returned slice is sorted by layer.
    pub fn peek_sorted(&self) -> Vec<&DrawCommand> {
        let mut refs: Vec<&DrawCommand> = self.commands.iter().collect();
        refs.sort_by_key(|c| c.layer());
        refs
    }

    /// Count how many commands belong to the given layer.
    #[must_use]
    pub fn count_by_layer(&self, layer: i32) -> usize {
        self.commands.iter().filter(|c| c.layer() == layer).count()
    }

    /// Bounding box of all commands as `(x_min, y_min, x_max, y_max)` in
    /// screen coordinates, or `None` if the batch is empty.
    #[must_use]
    pub fn total_bounds(&self) -> Option<(f32, f32, f32, f32)> {
        if self.commands.is_empty() {
            return None;
        }
        let mut x_min = f32::MAX;
        let mut y_min = f32::MAX;
        let mut x_max = f32::MIN;
        let mut y_max = f32::MIN;
        for cmd in &self.commands {
            let (x, y, w, h) = cmd.bounds();
            x_min = x_min.min(x);
            y_min = y_min.min(y);
            x_max = x_max.max(x + w);
            y_max = y_max.max(y + h);
        }
        Some((x_min, y_min, x_max, y_max))
    }
}

impl Default for DrawBatch {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BatchStats — per-flush statistics
// ---------------------------------------------------------------------------

/// Statistics produced by flushing a [`DrawBatch`].
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total number of commands flushed.
    pub total_commands: usize,
    /// Number of [`DrawCommand::Rect`] commands.
    pub rect_count: usize,
    /// Number of [`DrawCommand::Ellipse`] commands.
    pub ellipse_count: usize,
    /// Number of [`DrawCommand::Glyph`] commands.
    pub glyph_count: usize,
    /// Number of [`DrawCommand::Texture`] commands.
    pub texture_count: usize,
    /// Number of [`DrawCommand::Gradient`] commands.
    pub gradient_count: usize,
    /// Number of distinct layer values encountered.
    pub distinct_layers: usize,
}

impl BatchStats {
    /// Compute statistics for a slice of flushed commands.
    #[must_use]
    pub fn compute(commands: &[DrawCommand]) -> Self {
        let mut stats = Self {
            total_commands: commands.len(),
            ..Default::default()
        };
        let mut layers = std::collections::HashSet::new();
        for cmd in commands {
            layers.insert(cmd.layer());
            match cmd {
                DrawCommand::Rect(_) => stats.rect_count += 1,
                DrawCommand::Ellipse(_) => stats.ellipse_count += 1,
                DrawCommand::Glyph(_) => stats.glyph_count += 1,
                DrawCommand::Texture(_) => stats.texture_count += 1,
                DrawCommand::Gradient(_) => stats.gradient_count += 1,
            }
        }
        stats.distinct_layers = layers.len();
        stats
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rect(layer: i32) -> DrawCommand {
        DrawCommand::Rect(FillRect::new(0.0, 0.0, 100.0, 50.0, [255, 0, 0, 255], layer))
    }

    fn make_ellipse(layer: i32) -> DrawCommand {
        DrawCommand::Ellipse(FillEllipse::circle(50.0, 50.0, 25.0, [0, 255, 0, 255], layer))
    }

    // --- BlendMode ---

    #[test]
    fn test_blend_mode_default_is_over() {
        assert_eq!(BlendMode::default(), BlendMode::Over);
    }

    #[test]
    fn test_blend_mode_display() {
        assert_eq!(format!("{}", BlendMode::Multiply), "multiply");
        assert_eq!(format!("{}", BlendMode::Add), "add");
    }

    // --- FillRect ---

    #[test]
    fn test_fill_rect_area() {
        let r = FillRect::new(0.0, 0.0, 100.0, 50.0, [0, 0, 0, 255], 0);
        assert!((r.area() - 5000.0).abs() < f32::EPSILON);
    }

    // --- FillEllipse ---

    #[test]
    fn test_fill_ellipse_circle_constructor() {
        let e = FillEllipse::circle(100.0, 200.0, 30.0, [255, 255, 0, 255], 1);
        assert!((e.rx - 30.0).abs() < f32::EPSILON);
        assert!((e.ry - 30.0).abs() < f32::EPSILON);
    }

    // --- DrawCommand ---

    #[test]
    fn test_draw_command_layer() {
        let cmd = make_rect(5);
        assert_eq!(cmd.layer(), 5);
    }

    #[test]
    fn test_draw_command_blend() {
        let cmd = make_rect(0);
        assert_eq!(cmd.blend(), BlendMode::Over);
    }

    #[test]
    fn test_draw_command_bounds_rect() {
        let cmd = DrawCommand::Rect(FillRect::new(10.0, 20.0, 100.0, 50.0, [0, 0, 0, 0], 0));
        let (x, y, w, h) = cmd.bounds();
        assert!((x - 10.0).abs() < f32::EPSILON);
        assert!((y - 20.0).abs() < f32::EPSILON);
        assert!((w - 100.0).abs() < f32::EPSILON);
        assert!((h - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_draw_command_bounds_ellipse() {
        let cmd = make_ellipse(0);
        let (x, y, w, h) = cmd.bounds();
        // cx=50, cy=50, r=25 → x=25, y=25, w=50, h=50
        assert!((x - 25.0).abs() < f32::EPSILON);
        assert!((y - 25.0).abs() < f32::EPSILON);
        assert!((w - 50.0).abs() < f32::EPSILON);
        assert!((h - 50.0).abs() < f32::EPSILON);
    }

    // --- DrawBatch ---

    #[test]
    fn test_batch_push_and_len() {
        let mut batch = DrawBatch::new();
        assert!(batch.is_empty());
        batch.push(make_rect(0));
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_batch_flush_clears_batch() {
        let mut batch = DrawBatch::new();
        batch.push(make_rect(0));
        let _ = batch.flush();
        assert!(batch.is_empty());
    }

    #[test]
    fn test_batch_flush_sorted_by_layer() {
        let mut batch = DrawBatch::new();
        batch.push(make_rect(3));
        batch.push(make_rect(1));
        batch.push(make_ellipse(2));
        let flushed = batch.flush();
        assert_eq!(flushed[0].layer(), 1);
        assert_eq!(flushed[1].layer(), 2);
        assert_eq!(flushed[2].layer(), 3);
    }

    #[test]
    fn test_batch_flush_empty_returns_empty() {
        let mut batch = DrawBatch::new();
        let flushed = batch.flush();
        assert!(flushed.is_empty());
    }

    #[test]
    fn test_batch_clear() {
        let mut batch = DrawBatch::new();
        batch.push(make_rect(0));
        batch.clear();
        assert!(batch.is_empty());
    }

    #[test]
    fn test_batch_total_bounds_empty() {
        let batch = DrawBatch::new();
        assert!(batch.total_bounds().is_none());
    }

    #[test]
    fn test_batch_total_bounds_single() {
        let mut batch = DrawBatch::new();
        batch.push(DrawCommand::Rect(FillRect::new(
            10.0, 20.0, 100.0, 50.0, [0, 0, 0, 0], 0,
        )));
        let (x, y, xm, ym) = batch.total_bounds().expect("should have bounds");
        assert!((x - 10.0).abs() < f32::EPSILON);
        assert!((y - 20.0).abs() < f32::EPSILON);
        assert!((xm - 110.0).abs() < f32::EPSILON);
        assert!((ym - 70.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_batch_total_bounds_multi() {
        let mut batch = DrawBatch::new();
        batch.push(DrawCommand::Rect(FillRect::new(0.0, 0.0, 50.0, 50.0, [0, 0, 0, 0], 0)));
        batch.push(DrawCommand::Rect(FillRect::new(
            100.0, 100.0, 50.0, 50.0, [0, 0, 0, 0], 1,
        )));
        let (x, y, xm, ym) = batch.total_bounds().expect("should have bounds");
        assert!((x - 0.0).abs() < f32::EPSILON);
        assert!((y - 0.0).abs() < f32::EPSILON);
        assert!((xm - 150.0).abs() < f32::EPSILON);
        assert!((ym - 150.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_batch_count_by_layer() {
        let mut batch = DrawBatch::new();
        batch.push(make_rect(0));
        batch.push(make_rect(0));
        batch.push(make_rect(1));
        assert_eq!(batch.count_by_layer(0), 2);
        assert_eq!(batch.count_by_layer(1), 1);
        assert_eq!(batch.count_by_layer(99), 0);
    }

    #[test]
    fn test_batch_peek_sorted_does_not_consume() {
        let mut batch = DrawBatch::new();
        batch.push(make_rect(2));
        batch.push(make_rect(1));
        let sorted = batch.peek_sorted();
        assert_eq!(sorted[0].layer(), 1);
        assert_eq!(sorted[1].layer(), 2);
        // Batch is still intact.
        assert_eq!(batch.len(), 2);
    }

    // --- BatchStats ---

    #[test]
    fn test_batch_stats_counts() {
        let mut batch = DrawBatch::with_capacity(8);
        batch.push(make_rect(0));
        batch.push(make_rect(1));
        batch.push(make_ellipse(0));
        let cmds = batch.flush();
        let stats = BatchStats::compute(&cmds);
        assert_eq!(stats.total_commands, 3);
        assert_eq!(stats.rect_count, 2);
        assert_eq!(stats.ellipse_count, 1);
        assert_eq!(stats.glyph_count, 0);
        assert_eq!(stats.distinct_layers, 2);
    }

    #[test]
    fn test_batch_stats_empty() {
        let stats = BatchStats::compute(&[]);
        assert_eq!(stats.total_commands, 0);
        assert_eq!(stats.distinct_layers, 0);
    }

    #[test]
    fn test_batch_all_command_types() {
        let mut batch = DrawBatch::new();
        batch.push(DrawCommand::Rect(FillRect::new(0.0, 0.0, 10.0, 10.0, [0; 4], 0)));
        batch.push(DrawCommand::Ellipse(FillEllipse::circle(
            5.0, 5.0, 3.0, [0; 4], 1,
        )));
        batch.push(DrawCommand::Glyph(GlyphBlit {
            x: 0.0,
            y: 0.0,
            uv: (0.0, 0.0, 0.1, 0.1),
            width: 8.0,
            height: 12.0,
            color: [255, 255, 255, 255],
            layer: 2,
            blend: BlendMode::Over,
        }));
        batch.push(DrawCommand::Texture(TexturedRect {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            uv: (0.0, 0.0, 1.0, 1.0),
            tint: [255; 4],
            layer: 3,
            blend: BlendMode::Over,
        }));
        batch.push(DrawCommand::Gradient(GradientRect {
            x: 0.0,
            y: 0.0,
            width: 200.0,
            height: 50.0,
            color_start: [0, 0, 0, 255],
            color_end: [255, 255, 255, 255],
            horizontal: true,
            layer: 4,
            blend: BlendMode::Over,
        }));
        let cmds = batch.flush();
        let stats = BatchStats::compute(&cmds);
        assert_eq!(stats.total_commands, 5);
        assert_eq!(stats.rect_count, 1);
        assert_eq!(stats.ellipse_count, 1);
        assert_eq!(stats.glyph_count, 1);
        assert_eq!(stats.texture_count, 1);
        assert_eq!(stats.gradient_count, 1);
        assert_eq!(stats.distinct_layers, 5);
    }

    #[test]
    fn test_batch_with_capacity() {
        let batch = DrawBatch::with_capacity(64);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_batch_stable_sort_within_layer() {
        // Commands at the same layer should preserve insertion order.
        let mut batch = DrawBatch::new();
        batch.push(DrawCommand::Rect(FillRect::new(0.0, 0.0, 10.0, 10.0, [1, 0, 0, 255], 5)));
        batch.push(DrawCommand::Rect(FillRect::new(0.0, 0.0, 10.0, 10.0, [2, 0, 0, 255], 5)));
        batch.push(DrawCommand::Rect(FillRect::new(0.0, 0.0, 10.0, 10.0, [3, 0, 0, 255], 5)));
        let flushed = batch.flush();
        // All three are at layer 5; insertion order must be preserved.
        if let DrawCommand::Rect(r0) = &flushed[0] {
            assert_eq!(r0.color[0], 1);
        }
        if let DrawCommand::Rect(r1) = &flushed[1] {
            assert_eq!(r1.color[0], 2);
        }
        if let DrawCommand::Rect(r2) = &flushed[2] {
            assert_eq!(r2.color[0], 3);
        }
    }

    #[test]
    fn test_batch_default_creates_empty() {
        let batch = DrawBatch::default();
        assert!(batch.is_empty());
    }
}
