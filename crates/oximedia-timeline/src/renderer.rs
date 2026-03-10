//! Timeline renderer: renders timeline clips to a pixel buffer for preview.
//!
//! Provides a software renderer that composites video clips, transitions,
//! and effects into RGBA pixel buffers suitable for preview display.

use crate::clip::{Clip, MediaSource};
use crate::error::{TimelineError, TimelineResult};
use crate::timeline::Timeline;
use crate::transition_engine::{TransitionEngine, TransitionInput};
use crate::types::Position;

/// RGBA pixel buffer.
#[derive(Debug, Clone)]
pub struct PixelBuffer {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Raw RGBA data (4 bytes per pixel, row-major).
    pub data: Vec<u8>,
}

impl PixelBuffer {
    /// Create a new black pixel buffer.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0u8; (width * height * 4) as usize],
        }
    }

    /// Create a pixel buffer filled with a solid RGBA color.
    #[must_use]
    pub fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Self {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..width * height {
            data.extend_from_slice(&rgba);
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Get mutable reference to a pixel at (x, y) as a 4-byte slice.
    ///
    /// Returns `None` if coordinates are out of bounds.
    #[must_use]
    pub fn pixel_mut(&mut self, x: u32, y: u32) -> Option<&mut [u8]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = ((y * self.width + x) * 4) as usize;
        Some(&mut self.data[idx..idx + 4])
    }

    /// Get immutable reference to a pixel at (x, y).
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = ((y * self.width + x) * 4) as usize;
        let arr: [u8; 4] = self.data[idx..idx + 4].try_into().ok()?;
        Some(arr)
    }

    /// Composite another buffer over this one at (ox, oy) with given alpha.
    pub fn composite_over(&mut self, other: &Self, ox: i32, oy: i32, alpha: f32) {
        let alpha = alpha.clamp(0.0, 1.0);
        for sy in 0..other.height {
            let dy = oy + sy as i32;
            if dy < 0 || dy >= self.height as i32 {
                continue;
            }
            for sx in 0..other.width {
                let dx = ox + sx as i32;
                if dx < 0 || dx >= self.width as i32 {
                    continue;
                }
                let src_idx = ((sy * other.width + sx) * 4) as usize;
                let dst_idx = ((dy as u32 * self.width + dx as u32) * 4) as usize;

                let src_a = f32::from(other.data[src_idx + 3]) / 255.0 * alpha;
                let dst_a = f32::from(self.data[dst_idx + 3]) / 255.0;
                let out_a = src_a + dst_a * (1.0 - src_a);

                if out_a > 0.0 {
                    for c in 0..3 {
                        let src_c = f32::from(other.data[src_idx + c]) / 255.0;
                        let dst_c = f32::from(self.data[dst_idx + c]) / 255.0;
                        let out_c = (src_c * src_a + dst_c * dst_a * (1.0 - src_a)) / out_a;
                        self.data[dst_idx + c] = (out_c * 255.0).round() as u8;
                    }
                    self.data[dst_idx + 3] = (out_a * 255.0).round() as u8;
                }
            }
        }
    }

    /// Resize to new dimensions using nearest-neighbour.
    #[must_use]
    pub fn resize_nearest(&self, new_w: u32, new_h: u32) -> Self {
        let mut out = Self::new(new_w, new_h);
        let x_ratio = self.width as f32 / new_w as f32;
        let y_ratio = self.height as f32 / new_h as f32;
        for dy in 0..new_h {
            for dx in 0..new_w {
                let sx = (dx as f32 * x_ratio) as u32;
                let sy = (dy as f32 * y_ratio) as u32;
                let src_idx =
                    ((sy.min(self.height - 1) * self.width + sx.min(self.width - 1)) * 4) as usize;
                let dst_idx = ((dy * new_w + dx) * 4) as usize;
                out.data[dst_idx..dst_idx + 4].copy_from_slice(&self.data[src_idx..src_idx + 4]);
            }
        }
        out
    }
}

/// Render settings for a single frame.
#[derive(Debug, Clone)]
pub struct FrameRenderSettings {
    /// Target output width.
    pub width: u32,
    /// Target output height.
    pub height: u32,
    /// Background color (RGBA).
    pub background: [u8; 4],
    /// Whether to render transitions.
    pub render_transitions: bool,
}

impl Default for FrameRenderSettings {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            background: [0, 0, 0, 255],
            render_transitions: true,
        }
    }
}

/// Result of rendering a single frame.
#[derive(Debug, Clone)]
pub struct RenderedFrame {
    /// Frame position in timeline.
    pub position: Position,
    /// Rendered pixel buffer.
    pub buffer: PixelBuffer,
    /// Video track count that contributed to this frame.
    pub layers_composited: u32,
}

/// Timeline renderer: composites clips into pixel buffers.
pub struct TimelineRenderer {
    settings: FrameRenderSettings,
}

impl TimelineRenderer {
    /// Create a new renderer with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            settings: FrameRenderSettings::default(),
        }
    }

    /// Create with custom render settings.
    #[must_use]
    pub fn with_settings(settings: FrameRenderSettings) -> Self {
        Self { settings }
    }

    /// Render a single frame at the given timeline position.
    ///
    /// This produces a synthetic frame based on clip colors/metadata since
    /// actual media decoding is out of scope for this renderer.
    ///
    /// # Errors
    ///
    /// Returns an error if the position is outside timeline bounds.
    pub fn render_frame(
        &self,
        timeline: &Timeline,
        position: Position,
    ) -> TimelineResult<RenderedFrame> {
        if position.value() < 0 {
            return Err(TimelineError::InvalidPosition(
                "Position cannot be negative".to_string(),
            ));
        }

        let mut base = PixelBuffer::solid(
            self.settings.width,
            self.settings.height,
            self.settings.background,
        );

        let mut layers_composited = 0u32;

        // Composite video tracks bottom-to-top
        for track in &timeline.video_tracks {
            if track.hidden {
                continue;
            }
            let opacity = 1.0f32;
            for clip in &track.clips {
                if let Some(buf) = self.render_clip(clip, position) {
                    let resized =
                        if buf.width != self.settings.width || buf.height != self.settings.height {
                            buf.resize_nearest(self.settings.width, self.settings.height)
                        } else {
                            buf
                        };
                    base.composite_over(&resized, 0, 0, opacity);
                    layers_composited += 1;
                }
            }
        }

        // Render transitions if enabled
        if self.settings.render_transitions {
            self.apply_transitions(timeline, position, &mut base);
        }

        Ok(RenderedFrame {
            position,
            buffer: base,
            layers_composited,
        })
    }

    /// Render a range of frames.
    ///
    /// # Errors
    ///
    /// Returns error if any frame render fails.
    pub fn render_range(
        &self,
        timeline: &Timeline,
        start: Position,
        end: Position,
    ) -> TimelineResult<Vec<RenderedFrame>> {
        if start.value() > end.value() {
            return Err(TimelineError::InvalidPosition(
                "Start must be <= end".to_string(),
            ));
        }
        let mut frames = Vec::new();
        let mut pos = start;
        while pos.value() <= end.value() {
            frames.push(self.render_frame(timeline, pos)?);
            pos = Position::new(pos.value() + 1);
        }
        Ok(frames)
    }

    fn render_clip(&self, clip: &Clip, position: Position) -> Option<PixelBuffer> {
        if !clip.enabled {
            return None;
        }
        // Check if position falls within clip bounds
        let clip_start = clip.timeline_in.value();
        let clip_dur = clip.source_out.value() - clip.source_in.value();
        let clip_end = clip_start + clip_dur;
        if position.value() < clip_start || position.value() >= clip_end {
            return None;
        }

        // Synthesize a frame from the clip source
        let color = self.source_color(&clip.source);
        Some(PixelBuffer::solid(
            self.settings.width,
            self.settings.height,
            color,
        ))
    }

    fn source_color(&self, source: &MediaSource) -> [u8; 4] {
        match source {
            MediaSource::Color { rgba } => [
                (rgba[0] * 255.0) as u8,
                (rgba[1] * 255.0) as u8,
                (rgba[2] * 255.0) as u8,
                (rgba[3] * 255.0) as u8,
            ],
            MediaSource::BarsAndTone => [100, 100, 180, 255],
            MediaSource::File { path, .. } => {
                // Hash path to deterministic color for preview
                let hash = path.to_string_lossy().bytes().fold(0u32, |acc, b| {
                    acc.wrapping_mul(31).wrapping_add(u32::from(b))
                });
                [
                    ((hash >> 16) & 0xFF) as u8,
                    ((hash >> 8) & 0xFF) as u8,
                    (hash & 0xFF) as u8,
                    255,
                ]
            }
            _ => [80, 80, 80, 255],
        }
    }

    /// Apply transitions at the given position.
    ///
    /// For each video track, walk the clip list looking for adjacent clips
    /// connected by a transition. When the playhead falls within the
    /// transition zone (the overlap region at the cut point), render both
    /// the outgoing and incoming clips independently and blend them via
    /// the `TransitionEngine`.
    fn apply_transitions(&self, timeline: &Timeline, position: Position, buffer: &mut PixelBuffer) {
        let engine = TransitionEngine::new();

        for track in &timeline.video_tracks {
            if track.hidden {
                continue;
            }

            // Check each clip for a registered transition.
            for (idx, clip) in track.clips.iter().enumerate() {
                let Some(transition) = timeline.transitions.get(&clip.id) else {
                    continue;
                };
                if !transition.enabled {
                    continue;
                }
                // Skip audio-only transitions.
                if !transition.transition_type.is_video() {
                    continue;
                }

                let clip_dur = clip.source_out.value() - clip.source_in.value();
                let clip_end = clip.timeline_in.value() + clip_dur;
                let t_dur = transition.duration.0;
                if t_dur <= 0 {
                    continue;
                }

                // Determine the transition zone based on alignment.
                let (zone_start, zone_end) = match transition.alignment {
                    crate::transition::TransitionAlignment::Center => {
                        let half = t_dur / 2;
                        (clip_end - half, clip_end + (t_dur - half))
                    }
                    crate::transition::TransitionAlignment::EndAtCut => {
                        (clip_end - t_dur, clip_end)
                    }
                    crate::transition::TransitionAlignment::StartAtCut => {
                        (clip_end, clip_end + t_dur)
                    }
                };

                let pos_val = position.value();
                if pos_val < zone_start || pos_val >= zone_end {
                    continue;
                }

                // Compute normalised progress within the zone.
                let progress = if zone_end == zone_start {
                    0.0f32
                } else {
                    (pos_val - zone_start) as f32 / (zone_end - zone_start) as f32
                };

                // Render the outgoing clip frame.
                let outgoing_buf = self.render_clip(clip, position).unwrap_or_else(|| {
                    PixelBuffer::solid(
                        self.settings.width,
                        self.settings.height,
                        self.settings.background,
                    )
                });

                // Try to find the next clip as the incoming source.
                let incoming_buf = track
                    .clips
                    .get(idx + 1)
                    .and_then(|next_clip| self.render_clip(next_clip, position))
                    .unwrap_or_else(|| {
                        PixelBuffer::solid(
                            self.settings.width,
                            self.settings.height,
                            self.settings.background,
                        )
                    });

                // Resize both to output dimensions if needed.
                let out_resized = if outgoing_buf.width != self.settings.width
                    || outgoing_buf.height != self.settings.height
                {
                    outgoing_buf.resize_nearest(self.settings.width, self.settings.height)
                } else {
                    outgoing_buf
                };
                let in_resized = if incoming_buf.width != self.settings.width
                    || incoming_buf.height != self.settings.height
                {
                    incoming_buf.resize_nearest(self.settings.width, self.settings.height)
                } else {
                    incoming_buf
                };

                let input = TransitionInput {
                    outgoing: &out_resized,
                    incoming: &in_resized,
                    progress,
                };

                // Apply the transition; on error fall back to the outgoing buffer.
                let blended = engine.apply_or_fallback(transition, &input);
                *buffer = blended;
            }
        }
    }
}

impl Default for TimelineRenderer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_buffer_new() {
        let buf = PixelBuffer::new(4, 4);
        assert_eq!(buf.width, 4);
        assert_eq!(buf.height, 4);
        assert_eq!(buf.data.len(), 4 * 4 * 4);
        assert!(buf.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_pixel_buffer_solid() {
        let buf = PixelBuffer::solid(2, 2, [255, 0, 0, 255]);
        assert_eq!(buf.data[0], 255);
        assert_eq!(buf.data[1], 0);
        assert_eq!(buf.data[2], 0);
        assert_eq!(buf.data[3], 255);
    }

    #[test]
    fn test_pixel_access() {
        let buf = PixelBuffer::solid(4, 4, [100, 150, 200, 255]);
        let px = buf.pixel(2, 2).expect("should succeed in test");
        assert_eq!(px, [100, 150, 200, 255]);
        assert!(buf.pixel(10, 10).is_none());
    }

    #[test]
    fn test_pixel_buffer_composite() {
        let mut base = PixelBuffer::solid(4, 4, [0, 0, 0, 255]);
        let overlay = PixelBuffer::solid(2, 2, [255, 255, 255, 255]);
        base.composite_over(&overlay, 1, 1, 1.0);
        // Check that overlaid pixels changed
        let px = base.pixel(1, 1).expect("should succeed in test");
        assert!(px[0] > 200, "Expected overlaid pixel to be bright");
    }

    #[test]
    fn test_pixel_buffer_resize_nearest() {
        let buf = PixelBuffer::solid(4, 4, [200, 100, 50, 255]);
        let resized = buf.resize_nearest(8, 8);
        assert_eq!(resized.width, 8);
        assert_eq!(resized.height, 8);
        // All pixels should still be the same color
        let px = resized.pixel(0, 0).expect("should succeed in test");
        assert_eq!(px[0], 200);
    }

    #[test]
    fn test_renderer_default_settings() {
        let renderer = TimelineRenderer::new();
        assert_eq!(renderer.settings.width, 1920);
        assert_eq!(renderer.settings.height, 1080);
    }

    #[test]
    fn test_frame_render_settings_default() {
        let s = FrameRenderSettings::default();
        assert_eq!(s.background, [0, 0, 0, 255]);
        assert!(s.render_transitions);
    }

    #[test]
    fn test_render_empty_timeline() {
        use oximedia_core::Rational;
        let renderer = TimelineRenderer::with_settings(FrameRenderSettings {
            width: 64,
            height: 64,
            ..Default::default()
        });
        let timeline =
            Timeline::new("test", Rational::new(24, 1), 48000).expect("should succeed in test");
        let frame = renderer
            .render_frame(&timeline, Position::new(0))
            .expect("should succeed in test");
        assert_eq!(frame.buffer.width, 64);
        assert_eq!(frame.buffer.height, 64);
        assert_eq!(frame.layers_composited, 0);
    }

    #[test]
    fn test_render_negative_position_error() {
        use oximedia_core::Rational;
        let renderer = TimelineRenderer::new();
        let timeline =
            Timeline::new("test", Rational::new(24, 1), 48000).expect("should succeed in test");
        let result = renderer.render_frame(&timeline, Position::new(-1));
        assert!(result.is_err());
    }

    #[test]
    fn test_render_range() {
        use oximedia_core::Rational;
        let renderer = TimelineRenderer::with_settings(FrameRenderSettings {
            width: 16,
            height: 16,
            ..Default::default()
        });
        let timeline =
            Timeline::new("test", Rational::new(24, 1), 48000).expect("should succeed in test");
        let frames = renderer
            .render_range(&timeline, Position::new(0), Position::new(4))
            .expect("should succeed in test");
        assert_eq!(frames.len(), 5); // 0,1,2,3,4
    }

    #[test]
    fn test_render_range_invalid() {
        use oximedia_core::Rational;
        let renderer = TimelineRenderer::new();
        let timeline =
            Timeline::new("test", Rational::new(24, 1), 48000).expect("should succeed in test");
        let result = renderer.render_range(&timeline, Position::new(10), Position::new(5));
        assert!(result.is_err());
    }
}
