//! `TimecodeFilter` struct and frame-processing logic.

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unused_self)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::bool_to_int_with_if)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::no_effect_underscore_binding)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use std::collections::HashMap;

use fontdue::{Font, FontSettings};
use oximedia_codec::{Plane, VideoFrame};
use oximedia_core::PixelFormat;

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{InputPort, OutputPort, PortFormat, PortId, PortType, VideoPortFormat};

use super::config::{Color, FrameContext, TextStyle, TimecodeConfig};

// ── CachedGlyph ───────────────────────────────────────────────────────────────

/// Cached glyph for text rendering.
#[derive(Clone, Debug)]
pub(crate) struct CachedGlyph {
    pub bitmap: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub advance: f32,
    pub offset_x: f32,
    pub offset_y: f32,
}

// ── TimecodeFilter ────────────────────────────────────────────────────────────

/// Timecode burn-in filter.
///
/// This filter overlays timecode and metadata information onto video frames.
pub struct TimecodeFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
    config: TimecodeConfig,
    font: Font,
    frame_count: u64,
    /// Glyph cache for performance.
    glyph_cache: HashMap<(char, u32), CachedGlyph>,
}

impl TimecodeFilter {
    /// Create a new timecode filter with a provided font.
    ///
    /// # Errors
    ///
    /// Returns error if the font data is invalid.
    pub fn new(
        id: NodeId,
        name: impl Into<String>,
        config: TimecodeConfig,
        font_data: Vec<u8>,
    ) -> GraphResult<Self> {
        let font = Font::from_bytes(font_data.as_slice(), FontSettings::default())
            .map_err(|e| GraphError::ConfigurationError(format!("Invalid font data: {e}")))?;

        Ok(Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Video)
                .with_format(PortFormat::Video(VideoPortFormat::any()))],
            outputs: vec![OutputPort::new(PortId(0), "output", PortType::Video)
                .with_format(PortFormat::Video(VideoPortFormat::any()))],
            config,
            font,
            frame_count: 0,
            glyph_cache: HashMap::new(),
        })
    }

    /// Load a font from a file path and create the filter.
    ///
    /// # Errors
    ///
    /// Returns error if the font file cannot be read or is invalid.
    pub fn from_font_file(
        id: NodeId,
        name: impl Into<String>,
        config: TimecodeConfig,
        font_path: &str,
    ) -> GraphResult<Self> {
        let font_data = std::fs::read(font_path).map_err(|e| {
            GraphError::ConfigurationError(format!("Failed to read font file: {e}"))
        })?;
        Self::new(id, name, config, font_data)
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &TimecodeConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: TimecodeConfig) {
        self.config = config;
    }

    /// Update the frame context.
    pub fn set_context(&mut self, context: FrameContext) {
        self.config.context = context;
    }

    // ── Glyph helpers ─────────────────────────────────────────────────────────

    /// Get or cache a glyph.
    fn get_glyph(&mut self, c: char, font_size_int: u32) -> CachedGlyph {
        let key = (c, font_size_int);
        if let Some(cached) = self.glyph_cache.get(&key) {
            return cached.clone();
        }

        let font_size = font_size_int as f32;
        let (metrics, bitmap) = self.font.rasterize(c, font_size);

        let glyph = CachedGlyph {
            bitmap,
            width: metrics.width,
            height: metrics.height,
            advance: metrics.advance_width,
            offset_x: metrics.xmin as f32,
            offset_y: metrics.ymin as f32,
        };

        self.glyph_cache.insert(key, glyph.clone());
        glyph
    }

    /// Measure text dimensions.
    fn measure_text(&mut self, text: &str, font_size: f32) -> (u32, u32) {
        let font_size_int = font_size as u32;
        let mut width = 0.0f32;
        let mut max_height = 0.0f32;

        for c in text.chars() {
            let glyph = self.get_glyph(c, font_size_int);
            width += glyph.advance;
            max_height = max_height.max(glyph.height as f32);
        }

        (width.ceil() as u32, max_height.ceil() as u32)
    }

    // ── Rendering ─────────────────────────────────────────────────────────────

    /// Render text to an RGBA buffer.
    fn render_text(&mut self, text: &str, style: &TextStyle) -> (Vec<u8>, u32, u32) {
        let (text_width, text_height) = self.measure_text(text, style.font_size);
        let padding = style.padding;
        let buffer_width = text_width + padding * 2;
        let buffer_height = text_height + padding * 2;

        let mut buffer = vec![0u8; (buffer_width * buffer_height * 4) as usize];

        // Draw background
        if style.draw_background {
            for pixel in buffer.chunks_exact_mut(4) {
                pixel[0] = style.background.r;
                pixel[1] = style.background.g;
                pixel[2] = style.background.b;
                pixel[3] = style.background.a;
            }
        }

        // Render glyphs
        let font_size_int = style.font_size as u32;
        let mut x_pos = padding as f32;
        let baseline = (padding + text_height) as f32;

        for c in text.chars() {
            let glyph = self.get_glyph(c, font_size_int);

            let glyph_x = (x_pos + glyph.offset_x) as i32;
            let glyph_y = (baseline - glyph.height as f32 - glyph.offset_y) as i32;

            // Draw shadow
            if style.draw_shadow {
                self.draw_glyph_to_buffer(
                    &glyph.bitmap,
                    glyph.width,
                    glyph.height,
                    &mut buffer,
                    buffer_width,
                    buffer_height,
                    glyph_x + style.shadow_offset.0,
                    glyph_y + style.shadow_offset.1,
                    style.shadow_color,
                );
            }

            // Draw outline
            if style.draw_outline && style.outline_width > 0.0 {
                let outline_width = style.outline_width as i32;
                for dx in -outline_width..=outline_width {
                    for dy in -outline_width..=outline_width {
                        if dx != 0 || dy != 0 {
                            self.draw_glyph_to_buffer(
                                &glyph.bitmap,
                                glyph.width,
                                glyph.height,
                                &mut buffer,
                                buffer_width,
                                buffer_height,
                                glyph_x + dx,
                                glyph_y + dy,
                                style.outline,
                            );
                        }
                    }
                }
            }

            // Draw foreground
            self.draw_glyph_to_buffer(
                &glyph.bitmap,
                glyph.width,
                glyph.height,
                &mut buffer,
                buffer_width,
                buffer_height,
                glyph_x,
                glyph_y,
                style.foreground,
            );

            x_pos += glyph.advance;
        }

        (buffer, buffer_width, buffer_height)
    }

    /// Draw a glyph to an RGBA buffer.
    #[allow(clippy::too_many_arguments)]
    fn draw_glyph_to_buffer(
        &self,
        glyph_bitmap: &[u8],
        glyph_width: usize,
        glyph_height: usize,
        buffer: &mut [u8],
        buffer_width: u32,
        buffer_height: u32,
        x: i32,
        y: i32,
        color: Color,
    ) {
        for gy in 0..glyph_height {
            for gx in 0..glyph_width {
                let bx = x + gx as i32;
                let by = y + gy as i32;

                if bx < 0 || by < 0 || bx >= buffer_width as i32 || by >= buffer_height as i32 {
                    continue;
                }

                let glyph_alpha = glyph_bitmap[gy * glyph_width + gx];
                if glyph_alpha == 0 {
                    continue;
                }

                let buffer_idx = ((by as u32 * buffer_width + bx as u32) * 4) as usize;
                let alpha = (glyph_alpha as f32 / 255.0) * (color.a as f32 / 255.0);

                // Alpha blending
                let existing_alpha = buffer[buffer_idx + 3] as f32 / 255.0;
                let out_alpha = alpha + existing_alpha * (1.0 - alpha);

                if out_alpha > 0.0 {
                    buffer[buffer_idx] = ((color.r as f32 * alpha
                        + buffer[buffer_idx] as f32 * existing_alpha * (1.0 - alpha))
                        / out_alpha) as u8;
                    buffer[buffer_idx + 1] = ((color.g as f32 * alpha
                        + buffer[buffer_idx + 1] as f32 * existing_alpha * (1.0 - alpha))
                        / out_alpha) as u8;
                    buffer[buffer_idx + 2] = ((color.b as f32 * alpha
                        + buffer[buffer_idx + 2] as f32 * existing_alpha * (1.0 - alpha))
                        / out_alpha) as u8;
                    buffer[buffer_idx + 3] = (out_alpha * 255.0) as u8;
                }
            }
        }
    }

    // ── Compositing ───────────────────────────────────────────────────────────

    /// Composite RGBA buffer onto a video frame.
    fn composite_rgba_to_frame(
        &self,
        frame: &mut VideoFrame,
        rgba_buffer: &[u8],
        buffer_width: u32,
        buffer_height: u32,
        x: i32,
        y: i32,
    ) {
        match frame.format {
            PixelFormat::Rgb24 | PixelFormat::Rgba32 => {
                self.composite_to_rgb(frame, rgba_buffer, buffer_width, buffer_height, x, y);
            }
            PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => {
                self.composite_to_yuv(frame, rgba_buffer, buffer_width, buffer_height, x, y);
            }
            _ => {
                self.composite_to_yuv(frame, rgba_buffer, buffer_width, buffer_height, x, y);
            }
        }
    }

    /// Composite to RGB frame.
    fn composite_to_rgb(
        &self,
        frame: &mut VideoFrame,
        rgba_buffer: &[u8],
        buffer_width: u32,
        buffer_height: u32,
        x: i32,
        y: i32,
    ) {
        if frame.planes.is_empty() {
            return;
        }

        let plane = &frame.planes[0];
        let mut data = plane.data.to_vec();
        let bpp = if frame.format == PixelFormat::Rgba32 {
            4
        } else {
            3
        };

        for by in 0..buffer_height {
            for bx in 0..buffer_width {
                let fx = x + bx as i32;
                let fy = y + by as i32;

                if fx < 0 || fy < 0 || fx >= frame.width as i32 || fy >= frame.height as i32 {
                    continue;
                }

                let buffer_idx = ((by * buffer_width + bx) * 4) as usize;
                let alpha = rgba_buffer[buffer_idx + 3] as f32 / 255.0;

                if alpha < 0.01 {
                    continue;
                }

                let frame_idx = ((fy as u32 * frame.width + fx as u32) * bpp) as usize;

                for c in 0..3 {
                    let bg = data[frame_idx + c] as f32;
                    let fg = rgba_buffer[buffer_idx + c] as f32;
                    data[frame_idx + c] = (bg * (1.0 - alpha) + fg * alpha) as u8;
                }

                if bpp == 4 {
                    let bg_alpha = data[frame_idx + 3] as f32 / 255.0;
                    let out_alpha = alpha + bg_alpha * (1.0 - alpha);
                    data[frame_idx + 3] = (out_alpha * 255.0) as u8;
                }
            }
        }

        frame.planes[0] = Plane::new(data, plane.stride);
    }

    /// Composite to YUV frame.
    fn composite_to_yuv(
        &self,
        frame: &mut VideoFrame,
        rgba_buffer: &[u8],
        buffer_width: u32,
        buffer_height: u32,
        x: i32,
        y: i32,
    ) {
        let (h_sub, v_sub) = frame.format.chroma_subsampling();

        // Process Y plane
        if !frame.planes.is_empty() {
            let plane = &frame.planes[0];
            let mut y_data = plane.data.to_vec();

            for by in 0..buffer_height {
                for bx in 0..buffer_width {
                    let fx = x + bx as i32;
                    let fy = y + by as i32;

                    if fx < 0 || fy < 0 || fx >= frame.width as i32 || fy >= frame.height as i32 {
                        continue;
                    }

                    let buffer_idx = ((by * buffer_width + bx) * 4) as usize;
                    let alpha = rgba_buffer[buffer_idx + 3] as f32 / 255.0;

                    if alpha < 0.01 {
                        continue;
                    }

                    let r = rgba_buffer[buffer_idx] as f32;
                    let g = rgba_buffer[buffer_idx + 1] as f32;
                    let b = rgba_buffer[buffer_idx + 2] as f32;

                    // RGB to Y (BT.709)
                    let y_val = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                    let frame_idx = fy as usize * frame.width as usize + fx as usize;
                    let bg = y_data[frame_idx] as f32;
                    y_data[frame_idx] = (bg * (1.0 - alpha) + y_val * alpha) as u8;
                }
            }

            frame.planes[0] = Plane::new(y_data, plane.stride);
        }

        // Process U and V planes
        if frame.planes.len() >= 3 {
            for plane_idx in 1..3 {
                let plane = &frame.planes[plane_idx];
                let mut chroma_data = plane.data.to_vec();
                let chroma_width = frame.width / h_sub;
                let chroma_height = frame.height / v_sub;

                for by in 0..buffer_height {
                    for bx in 0..buffer_width {
                        let fx = x + bx as i32;
                        let fy = y + by as i32;

                        if fx < 0 || fy < 0 || fx >= frame.width as i32 || fy >= frame.height as i32
                        {
                            continue;
                        }

                        let buffer_idx = ((by * buffer_width + bx) * 4) as usize;
                        let alpha = rgba_buffer[buffer_idx + 3] as f32 / 255.0;

                        if alpha < 0.01 {
                            continue;
                        }

                        let cx = fx / h_sub as i32;
                        let cy = fy / v_sub as i32;

                        if cx < 0
                            || cy < 0
                            || cx >= chroma_width as i32
                            || cy >= chroma_height as i32
                        {
                            continue;
                        }

                        let r = rgba_buffer[buffer_idx] as f32;
                        let g = rgba_buffer[buffer_idx + 1] as f32;
                        let b = rgba_buffer[buffer_idx + 2] as f32;

                        // RGB to UV (BT.709)
                        let chroma_val = if plane_idx == 1 {
                            // U
                            -0.1146 * r - 0.3854 * g + 0.5 * b + 128.0
                        } else {
                            // V
                            0.5 * r - 0.4542 * g - 0.0458 * b + 128.0
                        };

                        let chroma_idx = cy as usize * chroma_width as usize + cx as usize;
                        let bg = chroma_data[chroma_idx] as f32;
                        chroma_data[chroma_idx] =
                            (bg * (1.0 - alpha) + chroma_val * alpha).clamp(0.0, 255.0) as u8;
                    }
                }

                frame.planes[plane_idx] = Plane::new(chroma_data, plane.stride);
            }
        }
    }

    // ── Progress bar ──────────────────────────────────────────────────────────

    /// Draw progress bar on frame.
    fn draw_progress_bar(&self, frame: &mut VideoFrame, current_frame: u64) {
        if !self.config.progress_bar.enabled || self.config.progress_bar.total_frames == 0 {
            return;
        }

        let bar = &self.config.progress_bar;
        let progress = (current_frame as f64 / bar.total_frames as f64).clamp(0.0, 1.0);
        let filled_width = (bar.width as f64 * progress) as u32;

        let (x, y) = bar.position.calculate(
            frame.width,
            frame.height,
            bar.width,
            bar.height,
            self.config.safe_margin,
        );

        let mut buffer = vec![0u8; (bar.width * bar.height * 4) as usize];

        // Fill background
        for pixel in buffer.chunks_exact_mut(4) {
            pixel[0] = bar.background.r;
            pixel[1] = bar.background.g;
            pixel[2] = bar.background.b;
            pixel[3] = bar.background.a;
        }

        // Fill progress
        for py in 0..bar.height {
            for px in 0..filled_width {
                let idx = ((py * bar.width + px) * 4) as usize;
                buffer[idx] = bar.foreground.r;
                buffer[idx + 1] = bar.foreground.g;
                buffer[idx + 2] = bar.foreground.b;
                buffer[idx + 3] = bar.foreground.a;
            }
        }

        self.composite_rgba_to_frame(frame, &buffer, bar.width, bar.height, x, y);
    }

    // ── Frame processing ──────────────────────────────────────────────────────

    /// Process a frame and add overlays.
    pub(crate) fn process_frame(&mut self, mut frame: VideoFrame) -> VideoFrame {
        // Update frame context
        self.config.context.timecode = self
            .config
            .timecode_format
            .format_timecode(self.frame_count, &self.config.context.framerate);
        self.config.context.frame_number = self.frame_count;
        self.config.context.width = frame.width;
        self.config.context.height = frame.height;

        // Draw progress bar
        self.draw_progress_bar(&mut frame, self.frame_count);

        // Draw overlay elements
        for element in &self.config.elements.clone() {
            if !element.enabled {
                continue;
            }

            let text = element.field.value(&self.config.context);
            let (rgba_buffer, buffer_width, buffer_height) =
                self.render_text(&text, &element.style);

            let (x, y) = element.position.calculate(
                frame.width,
                frame.height,
                buffer_width,
                buffer_height,
                self.config.safe_margin,
            );

            self.composite_rgba_to_frame(
                &mut frame,
                &rgba_buffer,
                buffer_width,
                buffer_height,
                x,
                y,
            );
        }

        self.frame_count += 1;
        frame
    }
}

// ── Node trait impl ───────────────────────────────────────────────────────────

impl Node for TimecodeFilter {
    fn id(&self) -> NodeId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn node_type(&self) -> NodeType {
        NodeType::Filter
    }

    fn state(&self) -> NodeState {
        self.state
    }

    fn set_state(&mut self, state: NodeState) -> GraphResult<()> {
        if !self.state.can_transition_to(state) {
            return Err(GraphError::InvalidStateTransition {
                node: self.id,
                from: self.state.to_string(),
                to: state.to_string(),
            });
        }
        self.state = state;
        Ok(())
    }

    fn inputs(&self) -> &[InputPort] {
        &self.inputs
    }

    fn outputs(&self) -> &[OutputPort] {
        &self.outputs
    }

    fn process(&mut self, input: Option<FilterFrame>) -> GraphResult<Option<FilterFrame>> {
        match input {
            Some(FilterFrame::Video(frame)) => {
                let processed = self.process_frame(frame);
                Ok(Some(FilterFrame::Video(processed)))
            }
            Some(_) => Err(GraphError::PortTypeMismatch {
                expected: "Video".to_string(),
                actual: "Audio".to_string(),
            }),
            None => Ok(None),
        }
    }

    fn reset(&mut self) -> GraphResult<()> {
        self.frame_count = 0;
        self.glyph_cache.clear();
        self.set_state(NodeState::Idle)
    }
}
