//! Transition preview rendering for video switchers.
//!
//! This module allows the operator to preview the result of an upcoming
//! transition *before* executing it.  A preview renderer composites the
//! program and preview sources using the currently armed transition type and
//! a configurable set of preview positions (thumbnails or full-frame scrub).
//!
//! The output of the preview renderer is a sequence of [`PreviewFrame`]s that
//! represent snapshots of the transition at different points in time.
//!
//! # Workflow
//!
//! 1. Configure the [`TransitionPreview`] with the desired transition style
//!    and preview resolution.
//! 2. Call [`TransitionPreview::generate_preview`] to compute a series of snapshots.
//! 3. Display the snapshots in the multiviewer's transition-preview pane.
//!
//! # Example
//!
//! ```rust
//! use oximedia_switcher::transition_preview::{
//!     TransitionPreview, TransitionPreviewConfig, PreviewTransitionStyle,
//! };
//!
//! let config = TransitionPreviewConfig::new(
//!     PreviewTransitionStyle::Mix,
//!     5,
//! );
//! let preview = TransitionPreview::new(config);
//!
//! let program_pixels = vec![200u8; 64];
//! let preview_pixels = vec![50u8; 64];
//!
//! let frames = preview.generate_preview(&program_pixels, &preview_pixels, 8, 8)
//!     .expect("generate ok");
//! assert_eq!(frames.len(), 5);
//! ```

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ────────────────────────────────────────────────────────────────────────────
// Errors
// ────────────────────────────────────────────────────────────────────────────

/// Errors from the transition preview subsystem.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TransitionPreviewError {
    /// The number of preview steps must be at least 2.
    #[error("Preview steps {0} is less than the minimum (2)")]
    TooFewSteps(usize),

    /// The source pixel buffers have mismatched lengths.
    #[error("Source pixel buffer length mismatch: program={0}, preview={1}")]
    PixelBufferMismatch(usize, usize),

    /// Width or height is zero.
    #[error("Frame dimensions must be non-zero (got {0}x{1})")]
    ZeroDimensions(u32, u32),

    /// The pixel buffer length does not match width * height.
    #[error("Pixel buffer length {0} does not match {1}x{2} = {3}")]
    BufferSizeMismatch(usize, u32, u32, usize),

    /// Preview generation was aborted.
    #[error("Preview generation aborted")]
    Aborted,
}

// ────────────────────────────────────────────────────────────────────────────
// Transition style (for preview purposes)
// ────────────────────────────────────────────────────────────────────────────

/// The transition style to preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreviewTransitionStyle {
    /// Cross-dissolve (linear blend).
    Mix,
    /// Horizontal wipe (left to right).
    WipeHorizontal,
    /// Vertical wipe (top to bottom).
    WipeVertical,
    /// Diagonal wipe (top-left to bottom-right).
    WipeDiagonal,
    /// Circle reveal from center.
    WipeCircle,
    /// Dip to black then reveal.
    DipToBlack,
    /// Dip to white then reveal.
    DipToWhite,
}

impl PreviewTransitionStyle {
    /// Compute the pixel blend factor for a given position and normalised
    /// pixel coordinates `(nx, ny)` in `[0.0, 1.0]`.
    ///
    /// Returns a value in `[0.0, 1.0]` where `0.0` = fully program and
    /// `1.0` = fully preview.
    pub fn blend_at(&self, position: f32, nx: f32, ny: f32) -> f32 {
        match self {
            PreviewTransitionStyle::Mix => position,
            PreviewTransitionStyle::WipeHorizontal => {
                if nx <= position {
                    1.0
                } else {
                    0.0
                }
            }
            PreviewTransitionStyle::WipeVertical => {
                if ny <= position {
                    1.0
                } else {
                    0.0
                }
            }
            PreviewTransitionStyle::WipeDiagonal => {
                let boundary = (nx + ny) / 2.0;
                if boundary <= position {
                    1.0
                } else {
                    0.0
                }
            }
            PreviewTransitionStyle::WipeCircle => {
                let dx = nx - 0.5;
                let dy = ny - 0.5;
                let dist = (dx * dx + dy * dy).sqrt() * 2.0;
                if dist <= position * 2.0 {
                    1.0
                } else {
                    0.0
                }
            }
            PreviewTransitionStyle::DipToBlack => {
                // First half: program fades to black.
                // Second half: black fades to preview.
                if position <= 0.5 {
                    // Output intensity = program * (1 - 2*position)
                    // We return "blend toward black" as a multiplier.
                    // Actually we need to return a blending factor differently.
                    // For dip: at pos 0.0 → program, at 0.5 → black, at 1.0 → preview.
                    // We encode this as a special value by returning negative to
                    // indicate "dip region".  But that is messy.
                    //
                    // Instead: treat the blend as if position < 0.5 → mix with black
                    // (controlled externally).  For simplicity in the preview
                    // renderer we encode it as: 0..0.5 → source=program dimmed,
                    // 0.5..1.0 → source=preview brightening.
                    0.0 // preview not visible yet
                } else {
                    1.0 // preview visible, brightening
                }
            }
            PreviewTransitionStyle::DipToWhite => {
                if position <= 0.5 {
                    0.0
                } else {
                    1.0
                }
            }
        }
    }

    /// For dip-style transitions, compute the brightness multiplier at the
    /// given position.  Returns 1.0 for non-dip styles.
    pub fn dip_brightness(&self, position: f32) -> f32 {
        match self {
            PreviewTransitionStyle::DipToBlack => {
                // V-shaped curve: 1.0 → 0.0 → 1.0
                if position <= 0.5 {
                    1.0 - position * 2.0
                } else {
                    (position - 0.5) * 2.0
                }
            }
            PreviewTransitionStyle::DipToWhite => {
                // Inverted V: 0.0 → 1.0 → 0.0 for white additive
                // brightness goes: normal → white → normal
                // white_amount = 1 - |2*pos - 1|
                1.0 // base brightness stays at 1.0; white is added via overlay
            }
            _ => 1.0,
        }
    }

    /// Whether this style is a dip (requires special brightness handling).
    pub fn is_dip(&self) -> bool {
        matches!(
            self,
            PreviewTransitionStyle::DipToBlack | PreviewTransitionStyle::DipToWhite
        )
    }

    /// White overlay amount for DipToWhite at the given position.
    pub fn white_overlay(&self, position: f32) -> f32 {
        match self {
            PreviewTransitionStyle::DipToWhite => {
                // Peak white at position 0.5
                let t = (1.0 - (2.0 * position - 1.0).abs()).max(0.0);
                t
            }
            _ => 0.0,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the transition preview renderer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionPreviewConfig {
    /// The transition style to preview.
    pub style: PreviewTransitionStyle,
    /// Number of preview frames (snapshots) to generate.
    pub num_steps: usize,
    /// Optional soft-edge width for wipe styles (0.0 = hard edge, 1.0 = max).
    pub soft_edge: f32,
    /// Whether to include the first frame (position = 0.0, fully program).
    pub include_start: bool,
    /// Whether to include the last frame (position = 1.0, fully preview).
    pub include_end: bool,
}

impl TransitionPreviewConfig {
    /// Create a new configuration.
    pub fn new(style: PreviewTransitionStyle, num_steps: usize) -> Self {
        Self {
            style,
            num_steps,
            soft_edge: 0.0,
            include_start: true,
            include_end: true,
        }
    }

    /// Set the soft-edge width.
    pub fn with_soft_edge(mut self, soft_edge: f32) -> Self {
        self.soft_edge = soft_edge.clamp(0.0, 1.0);
        self
    }

    /// Validate.
    pub fn validate(&self) -> Result<(), TransitionPreviewError> {
        if self.num_steps < 2 {
            return Err(TransitionPreviewError::TooFewSteps(self.num_steps));
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Preview frame
// ────────────────────────────────────────────────────────────────────────────

/// A single preview snapshot.
#[derive(Debug, Clone)]
pub struct PreviewFrame {
    /// Transition position at this snapshot (0.0 .. 1.0).
    pub position: f32,
    /// Composited pixel data (single-channel greyscale or RGBA depending on
    /// the input format).
    pub pixels: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Step index (0-based).
    pub step_index: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Preview renderer
// ────────────────────────────────────────────────────────────────────────────

/// Renders transition preview frames from two source pixel buffers.
#[derive(Debug, Clone)]
pub struct TransitionPreview {
    config: TransitionPreviewConfig,
}

impl TransitionPreview {
    /// Create a new transition preview renderer.
    pub fn new(config: TransitionPreviewConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &TransitionPreviewConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: TransitionPreviewConfig) {
        self.config = config;
    }

    /// Generate preview frames from single-channel source pixel buffers.
    ///
    /// `program_pixels` and `preview_pixels` are flat 8-bit buffers of length
    /// `width * height`.
    pub fn generate_preview(
        &self,
        program_pixels: &[u8],
        preview_pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<PreviewFrame>, TransitionPreviewError> {
        self.config.validate()?;

        if width == 0 || height == 0 {
            return Err(TransitionPreviewError::ZeroDimensions(width, height));
        }

        let expected = (width as usize) * (height as usize);
        if program_pixels.len() != expected {
            return Err(TransitionPreviewError::BufferSizeMismatch(
                program_pixels.len(),
                width,
                height,
                expected,
            ));
        }
        if preview_pixels.len() != expected {
            return Err(TransitionPreviewError::BufferSizeMismatch(
                preview_pixels.len(),
                width,
                height,
                expected,
            ));
        }

        let positions = self.compute_positions();
        let mut frames = Vec::with_capacity(positions.len());

        for (step_index, &pos) in positions.iter().enumerate() {
            let pixels = self.render_frame(program_pixels, preview_pixels, width, height, pos);
            frames.push(PreviewFrame {
                position: pos,
                pixels,
                width,
                height,
                step_index,
            });
        }

        Ok(frames)
    }

    /// Generate a single preview frame at the given position.
    pub fn generate_single(
        &self,
        program_pixels: &[u8],
        preview_pixels: &[u8],
        width: u32,
        height: u32,
        position: f32,
    ) -> Result<PreviewFrame, TransitionPreviewError> {
        if width == 0 || height == 0 {
            return Err(TransitionPreviewError::ZeroDimensions(width, height));
        }

        let expected = (width as usize) * (height as usize);
        if program_pixels.len() != expected {
            return Err(TransitionPreviewError::BufferSizeMismatch(
                program_pixels.len(),
                width,
                height,
                expected,
            ));
        }
        if preview_pixels.len() != expected {
            return Err(TransitionPreviewError::BufferSizeMismatch(
                preview_pixels.len(),
                width,
                height,
                expected,
            ));
        }

        let pos = position.clamp(0.0, 1.0);
        let pixels = self.render_frame(program_pixels, preview_pixels, width, height, pos);
        Ok(PreviewFrame {
            position: pos,
            pixels,
            width,
            height,
            step_index: 0,
        })
    }

    /// Compute the set of positions to sample.
    fn compute_positions(&self) -> Vec<f32> {
        let n = self.config.num_steps;
        let mut positions = Vec::with_capacity(n);
        for i in 0..n {
            let t = if n <= 1 {
                0.5
            } else {
                i as f32 / (n - 1) as f32
            };
            positions.push(t);
        }
        // Trim start/end if not included.
        if !self.config.include_start && !positions.is_empty() {
            if let Some(first) = positions.first() {
                if *first < f32::EPSILON {
                    positions.remove(0);
                }
            }
        }
        if !self.config.include_end && !positions.is_empty() {
            if let Some(last) = positions.last() {
                if *last > 1.0 - f32::EPSILON {
                    positions.pop();
                }
            }
        }
        positions
    }

    /// Render a single composited frame.
    fn render_frame(
        &self,
        program: &[u8],
        preview: &[u8],
        width: u32,
        height: u32,
        position: f32,
    ) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let total = w * h;
        let mut output = vec![0u8; total];
        let soft = self.config.soft_edge;
        let half_soft = soft / 2.0;
        let style = self.config.style;

        let brightness = style.dip_brightness(position);
        let white_add = style.white_overlay(position);

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if idx >= total {
                    break;
                }

                let nx = if w > 1 {
                    x as f32 / (w - 1) as f32
                } else {
                    0.5
                };
                let ny = if h > 1 {
                    y as f32 / (h - 1) as f32
                } else {
                    0.5
                };

                let raw_blend = style.blend_at(position, nx, ny);

                // Apply soft edge for wipe styles
                let blend = if half_soft > f32::EPSILON && !style.is_dip() {
                    // For spatial styles, compute distance from boundary
                    let boundary = match style {
                        PreviewTransitionStyle::WipeHorizontal => nx,
                        PreviewTransitionStyle::WipeVertical => ny,
                        PreviewTransitionStyle::WipeDiagonal => (nx + ny) / 2.0,
                        PreviewTransitionStyle::WipeCircle => {
                            let dx = nx - 0.5;
                            let dy = ny - 0.5;
                            (dx * dx + dy * dy).sqrt() * 2.0
                        }
                        _ => raw_blend,
                    };
                    if matches!(
                        style,
                        PreviewTransitionStyle::WipeHorizontal
                            | PreviewTransitionStyle::WipeVertical
                            | PreviewTransitionStyle::WipeDiagonal
                            | PreviewTransitionStyle::WipeCircle
                    ) {
                        ((position - boundary + half_soft) / (2.0 * half_soft)).clamp(0.0, 1.0)
                    } else {
                        raw_blend
                    }
                } else {
                    raw_blend
                };

                let prog_val = program[idx] as f32;
                let prev_val = preview[idx] as f32;

                let mixed = prog_val * (1.0 - blend) + prev_val * blend;
                let dimmed = mixed * brightness;
                let with_white = dimmed + white_add * 255.0;
                output[idx] = with_white.clamp(0.0, 255.0) as u8;
            }
        }

        output
    }

    /// Return the number of preview steps that will be generated.
    pub fn num_steps(&self) -> usize {
        self.config.num_steps
    }

    /// Return the current transition style.
    pub fn style(&self) -> PreviewTransitionStyle {
        self.config.style
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pixels(val: u8, count: usize) -> Vec<u8> {
        vec![val; count]
    }

    #[test]
    fn test_generate_preview_mix_correct_count() {
        let config = TransitionPreviewConfig::new(PreviewTransitionStyle::Mix, 5);
        let preview = TransitionPreview::new(config);
        let program = make_pixels(200, 16);
        let prev = make_pixels(50, 16);
        let frames = preview.generate_preview(&program, &prev, 4, 4).expect("ok");
        assert_eq!(frames.len(), 5);
    }

    #[test]
    fn test_mix_first_frame_is_program() {
        let config = TransitionPreviewConfig::new(PreviewTransitionStyle::Mix, 3);
        let preview = TransitionPreview::new(config);
        let program = make_pixels(200, 4);
        let prev = make_pixels(50, 4);
        let frames = preview.generate_preview(&program, &prev, 2, 2).expect("ok");
        // First frame at position 0.0: should be fully program
        assert!((frames[0].position - 0.0).abs() < f32::EPSILON);
        for &p in &frames[0].pixels {
            assert_eq!(p, 200);
        }
    }

    #[test]
    fn test_mix_last_frame_is_preview() {
        let config = TransitionPreviewConfig::new(PreviewTransitionStyle::Mix, 3);
        let preview = TransitionPreview::new(config);
        let program = make_pixels(200, 4);
        let prev = make_pixels(50, 4);
        let frames = preview.generate_preview(&program, &prev, 2, 2).expect("ok");
        // Last frame at position 1.0: should be fully preview
        let last = frames.last().expect("has frames");
        assert!((last.position - 1.0).abs() < f32::EPSILON);
        for &p in &last.pixels {
            assert_eq!(p, 50);
        }
    }

    #[test]
    fn test_mix_midpoint_blends() {
        let config = TransitionPreviewConfig::new(PreviewTransitionStyle::Mix, 3);
        let preview = TransitionPreview::new(config);
        let program = make_pixels(200, 4);
        let prev = make_pixels(50, 4);
        let frames = preview.generate_preview(&program, &prev, 2, 2).expect("ok");
        // Middle frame at position 0.5: blend of 200 and 50 = 125
        let mid = &frames[1];
        assert!((mid.position - 0.5).abs() < f32::EPSILON);
        for &p in &mid.pixels {
            assert_eq!(p, 125);
        }
    }

    #[test]
    fn test_wipe_horizontal_half() {
        // 4x1 image, wipe at 0.5 should reveal left half as preview
        let config = TransitionPreviewConfig::new(PreviewTransitionStyle::WipeHorizontal, 3);
        let preview = TransitionPreview::new(config);
        let program = make_pixels(200, 4);
        let prev = make_pixels(50, 4);
        let frames = preview.generate_preview(&program, &prev, 4, 1).expect("ok");
        let mid = &frames[1]; // position = 0.5
                              // Pixels where nx <= 0.5 should be preview (50), others program (200).
                              // nx values for 4 pixels: 0/3=0.0, 1/3=0.33, 2/3=0.67, 3/3=1.0
                              // So pixels 0,1 are <= 0.5 → preview(50), pixels 2,3 > 0.5 → program(200)
        assert_eq!(mid.pixels[0], 50);
        assert_eq!(mid.pixels[1], 50);
        assert_eq!(mid.pixels[2], 200);
        assert_eq!(mid.pixels[3], 200);
    }

    #[test]
    fn test_zero_dimensions_error() {
        let config = TransitionPreviewConfig::new(PreviewTransitionStyle::Mix, 3);
        let preview = TransitionPreview::new(config);
        let err = preview
            .generate_preview(&[], &[], 0, 0)
            .expect_err("zero dims");
        assert!(matches!(err, TransitionPreviewError::ZeroDimensions(0, 0)));
    }

    #[test]
    fn test_buffer_size_mismatch_error() {
        let config = TransitionPreviewConfig::new(PreviewTransitionStyle::Mix, 3);
        let preview = TransitionPreview::new(config);
        let program = make_pixels(100, 5); // wrong size for 2x2
        let prev = make_pixels(100, 4);
        let err = preview
            .generate_preview(&program, &prev, 2, 2)
            .expect_err("mismatch");
        assert!(matches!(
            err,
            TransitionPreviewError::BufferSizeMismatch(5, 2, 2, 4)
        ));
    }

    #[test]
    fn test_too_few_steps_error() {
        let config = TransitionPreviewConfig::new(PreviewTransitionStyle::Mix, 1);
        let preview = TransitionPreview::new(config);
        let err = preview
            .generate_preview(&[0], &[0], 1, 1)
            .expect_err("too few");
        assert!(matches!(err, TransitionPreviewError::TooFewSteps(1)));
    }

    #[test]
    fn test_generate_single() {
        let config = TransitionPreviewConfig::new(PreviewTransitionStyle::Mix, 5);
        let preview = TransitionPreview::new(config);
        let program = make_pixels(200, 4);
        let prev = make_pixels(50, 4);
        let frame = preview
            .generate_single(&program, &prev, 2, 2, 0.5)
            .expect("ok");
        assert!((frame.position - 0.5).abs() < f32::EPSILON);
        for &p in &frame.pixels {
            assert_eq!(p, 125);
        }
    }

    #[test]
    fn test_dip_to_black_midpoint() {
        let config = TransitionPreviewConfig::new(PreviewTransitionStyle::DipToBlack, 3);
        let preview = TransitionPreview::new(config);
        let program = make_pixels(200, 4);
        let prev = make_pixels(200, 4);
        let frames = preview.generate_preview(&program, &prev, 2, 2).expect("ok");
        // At midpoint (pos=0.5), brightness should be 0 → all black
        let mid = &frames[1];
        assert!((mid.position - 0.5).abs() < f32::EPSILON);
        for &p in &mid.pixels {
            assert_eq!(p, 0);
        }
    }

    #[test]
    fn test_soft_edge_wipe() {
        let config = TransitionPreviewConfig::new(PreviewTransitionStyle::WipeHorizontal, 3)
            .with_soft_edge(0.5);
        let preview = TransitionPreview::new(config);
        let program = make_pixels(200, 4);
        let prev = make_pixels(0, 4);
        let frames = preview.generate_preview(&program, &prev, 4, 1).expect("ok");
        // With soft edge, the transition zone should produce intermediate values
        let mid = &frames[1]; // position = 0.5
                              // Some pixels should be neither 200 nor 0
        let has_intermediate = mid.pixels.iter().any(|&p| p > 0 && p < 200);
        assert!(has_intermediate, "soft edge should produce blended pixels");
    }

    #[test]
    fn test_exclude_start_end() {
        let mut config = TransitionPreviewConfig::new(PreviewTransitionStyle::Mix, 5);
        config.include_start = false;
        config.include_end = false;
        let preview = TransitionPreview::new(config);
        let program = make_pixels(200, 4);
        let prev = make_pixels(50, 4);
        let frames = preview.generate_preview(&program, &prev, 2, 2).expect("ok");
        // Should have excluded position 0.0 and 1.0
        assert!(frames.len() < 5);
        if let Some(first) = frames.first() {
            assert!(first.position > f32::EPSILON);
        }
        if let Some(last) = frames.last() {
            assert!(last.position < 1.0 - f32::EPSILON);
        }
    }

    #[test]
    fn test_transition_style_is_dip() {
        assert!(PreviewTransitionStyle::DipToBlack.is_dip());
        assert!(PreviewTransitionStyle::DipToWhite.is_dip());
        assert!(!PreviewTransitionStyle::Mix.is_dip());
        assert!(!PreviewTransitionStyle::WipeHorizontal.is_dip());
    }

    #[test]
    fn test_white_overlay_peaks_at_midpoint() {
        let at_mid = PreviewTransitionStyle::DipToWhite.white_overlay(0.5);
        let at_start = PreviewTransitionStyle::DipToWhite.white_overlay(0.0);
        let at_end = PreviewTransitionStyle::DipToWhite.white_overlay(1.0);
        assert!(at_mid > at_start);
        assert!(at_mid > at_end);
        assert!((at_mid - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_num_steps_and_style_accessors() {
        let config = TransitionPreviewConfig::new(PreviewTransitionStyle::WipeVertical, 7);
        let preview = TransitionPreview::new(config);
        assert_eq!(preview.num_steps(), 7);
        assert_eq!(preview.style(), PreviewTransitionStyle::WipeVertical);
    }

    #[test]
    fn test_vertical_wipe_at_full() {
        // 1x4 image, wipe at 1.0 should be fully preview
        let config = TransitionPreviewConfig::new(PreviewTransitionStyle::WipeVertical, 2);
        let preview = TransitionPreview::new(config);
        let program = make_pixels(200, 4);
        let prev = make_pixels(50, 4);
        let frames = preview.generate_preview(&program, &prev, 1, 4).expect("ok");
        let last = frames.last().expect("has frames");
        for &p in &last.pixels {
            assert_eq!(p, 50);
        }
    }
}
