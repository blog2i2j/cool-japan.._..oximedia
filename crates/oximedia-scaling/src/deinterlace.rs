#![allow(dead_code)]
//! Deinterlacing configuration and processing helpers.
//!
//! Provides both a simple single-field DeinterlaceProcessor and a full
//! two-field MotionAdaptiveProcessor that selects bob vs. weave on a
//! per-pixel basis based on temporal motion detection.

/// The field order of an interlaced video signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldOrder {
    /// Top (odd) field is displayed first.
    TopFieldFirst,
    /// Bottom (even) field is displayed first.
    BottomFieldFirst,
    /// The material is already progressive.
    Progressive,
}

/// Algorithm used to convert interlaced fields to progressive frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeinterlaceMethod {
    /// Drop one field.
    FieldDrop,
    /// Blend the two fields together.
    Blend,
    /// Bob deinterlacing.
    Bob,
    /// Weave.
    Weave,
    /// Motion-adaptive deinterlacing.
    MotionAdaptive,
    /// Yadif algorithm.
    Yadif,
}

impl DeinterlaceMethod {
    /// Frame-rate multiplier relative to the input field rate.
    pub fn output_frame_rate_multiplier(&self) -> u32 {
        match self {
            DeinterlaceMethod::Bob | DeinterlaceMethod::Yadif => 2,
            _ => 1,
        }
    }
}

/// Configuration for a deinterlacing operation.
#[derive(Debug, Clone)]
pub struct DeinterlaceConfig {
    /// The field order of the input material.
    pub field_order: FieldOrder,
    /// The deinterlacing algorithm to apply.
    pub method: DeinterlaceMethod,
    /// Number of threads to use for processing (0 = auto).
    pub threads: u32,
}

impl DeinterlaceConfig {
    /// Create a new DeinterlaceConfig.
    pub fn new(field_order: FieldOrder, method: DeinterlaceMethod) -> Self {
        Self {
            field_order,
            method,
            threads: 0,
        }
    }

    /// Whether the chosen method uses temporal information.
    pub fn is_temporal(&self) -> bool {
        matches!(
            self.method,
            DeinterlaceMethod::MotionAdaptive | DeinterlaceMethod::Yadif
        )
    }

    /// Whether the input is already progressive.
    pub fn is_progressive_passthrough(&self) -> bool {
        self.field_order == FieldOrder::Progressive
    }
}

/// A single video field extracted from an interlaced frame.
#[derive(Debug, Clone)]
pub struct VideoField {
    /// Which field this is (0 = top, 1 = bottom).
    pub field_index: u8,
    /// Width of the field in pixels.
    pub width: u32,
    /// Height of the field in pixels (half the frame height).
    pub height: u32,
    /// Raw luma byte data.
    pub luma: Vec<u8>,
}

impl VideoField {
    /// Create a new VideoField with blank luma.
    pub fn blank(field_index: u8, width: u32, height: u32) -> Self {
        Self {
            field_index,
            width,
            height,
            luma: vec![0u8; (width * height) as usize],
        }
    }
}

/// Processes video fields into progressive frames (single-field interface).
#[derive(Debug)]
pub struct DeinterlaceProcessor {
    config: DeinterlaceConfig,
}

impl DeinterlaceProcessor {
    /// Create a new DeinterlaceProcessor.
    pub fn new(config: DeinterlaceConfig) -> Self {
        Self { config }
    }

    /// Access the current configuration.
    pub fn config(&self) -> &DeinterlaceConfig {
        &self.config
    }

    /// Process a single VideoField and return a progressive frame.
    pub fn process_field(&self, field: &VideoField) -> Vec<u8> {
        if self.config.is_progressive_passthrough() {
            return field.luma.clone();
        }

        match self.config.method {
            DeinterlaceMethod::FieldDrop => field.luma.clone(),
            _ => bob_deinterlace(field),
        }
    }

    /// Output frame rate given the input frame rate.
    #[allow(clippy::cast_precision_loss)]
    pub fn output_fps(&self, input_fps_num: u32, input_fps_den: u32) -> f64 {
        let multiplier = self.config.method.output_frame_rate_multiplier();
        (input_fps_num as f64 / input_fps_den as f64) * multiplier as f64
    }
}

/// Bob deinterlace: reconstruct a full-height progressive frame from a single field.
pub(crate) fn bob_deinterlace(field: &VideoField) -> Vec<u8> {
    let w = field.width as usize;
    let fh = field.height as usize;
    let full_h = fh * 2;

    let mut out = vec![0u8; w * full_h];
    let field_row_offset = (field.field_index & 1) as usize;

    for fi in 0..fh {
        let out_row = field_row_offset + fi * 2;
        let src_start = fi * w;
        let dst_start = out_row * w;
        out[dst_start..dst_start + w].copy_from_slice(&field.luma[src_start..src_start + w]);
    }

    let missing_offset = 1 - field_row_offset;
    for mi in 0..fh {
        let missing_row = missing_offset + mi * 2;
        let above_known = missing_row.checked_sub(1);
        let below_known = {
            let r = missing_row + 1;
            if r < full_h {
                Some(r)
            } else {
                None
            }
        };
        let dst_start = missing_row * w;

        match (above_known, below_known) {
            (Some(above), Some(below)) => {
                let above_start = above * w;
                let below_start = below * w;
                let averaged: Vec<u8> = (0..w)
                    .map(|x| {
                        let a = out[above_start + x] as u16;
                        let b = out[below_start + x] as u16;
                        ((a + b + 1) / 2) as u8
                    })
                    .collect();
                out[dst_start..dst_start + w].copy_from_slice(&averaged);
            }
            (None, Some(below)) => {
                let below_start = below * w;
                let (left, right) = out.split_at_mut(below_start);
                left[dst_start..dst_start + w].copy_from_slice(&right[..w]);
            }
            (Some(above), None) => {
                let above_start = above * w;
                let (left, right) = out.split_at_mut(dst_start);
                right[..w].copy_from_slice(&left[above_start..above_start + w]);
            }
            (None, None) => {}
        }
    }

    out
}

// -- Motion-Adaptive Deinterlacing ------------------------------------------

/// Configuration for the motion-adaptive deinterlace processor.
#[derive(Debug, Clone)]
pub struct MotionAdaptiveConfig {
    /// Field order of the input material.
    pub field_order: FieldOrder,
    /// Luma difference threshold above which a pixel is classified as moving.
    pub motion_threshold: u8,
}

impl MotionAdaptiveConfig {
    /// Create a new configuration with a default threshold of 16.
    pub fn new(field_order: FieldOrder) -> Self {
        Self {
            field_order,
            motion_threshold: 16,
        }
    }

    /// Set the per-pixel motion detection threshold.
    pub fn with_threshold(mut self, threshold: u8) -> Self {
        self.motion_threshold = threshold;
        self
    }
}

/// Produces full-height progressive frames using motion-adaptive bob/weave selection.
///
/// # Example
///
/// ```rust
/// use oximedia_scaling::deinterlace::{FieldOrder, MotionAdaptiveConfig, MotionAdaptiveProcessor, VideoField};
///
/// let config = MotionAdaptiveConfig::new(FieldOrder::TopFieldFirst).with_threshold(16);
/// let mut processor = MotionAdaptiveProcessor::new(config);
/// let field0 = VideoField::blank(0, 4, 4);
/// let frame0 = processor.process_field(&field0);
/// assert_eq!(frame0.len(), 4 * 8);
/// ```
#[derive(Debug)]
pub struct MotionAdaptiveProcessor {
    config: MotionAdaptiveConfig,
    prev_field_luma: Option<Vec<u8>>,
    prev_width: u32,
    prev_height: u32,
}

impl MotionAdaptiveProcessor {
    /// Create a new processor.
    pub fn new(config: MotionAdaptiveConfig) -> Self {
        Self {
            config,
            prev_field_luma: None,
            prev_width: 0,
            prev_height: 0,
        }
    }

    /// Returns the motion threshold.
    pub fn motion_threshold(&self) -> u8 {
        self.config.motion_threshold
    }

    /// Process one field and return a full-height progressive frame.
    pub fn process_field(&mut self, field: &VideoField) -> Vec<u8> {
        let w = field.width as usize;
        let fh = field.height as usize;
        let full_h = fh * 2;

        let bob_out = bob_deinterlace(field);

        let result = if let Some(ref prev) = self.prev_field_luma {
            if self.prev_width as usize == w && self.prev_height as usize == fh {
                motion_adaptive_blend(
                    field,
                    &bob_out,
                    prev,
                    w,
                    fh,
                    full_h,
                    self.config.motion_threshold,
                )
            } else {
                bob_out
            }
        } else {
            bob_out
        };

        self.prev_field_luma = Some(field.luma.clone());
        self.prev_width = field.width;
        self.prev_height = field.height;

        result
    }

    /// Reset the stored previous-field reference.
    pub fn reset(&mut self) {
        self.prev_field_luma = None;
        self.prev_width = 0;
        self.prev_height = 0;
    }
}

fn motion_adaptive_blend(
    field: &VideoField,
    bob_out: &[u8],
    prev_luma: &[u8],
    w: usize,
    fh: usize,
    full_h: usize,
    threshold: u8,
) -> Vec<u8> {
    let mut out = bob_out.to_vec();
    let field_row_offset = (field.field_index & 1) as usize;
    let missing_offset = 1 - field_row_offset;

    for mi in 0..fh {
        let missing_row = missing_offset + mi * 2;
        if missing_row >= full_h {
            continue;
        }

        let above_known = if missing_row > 0 {
            Some(missing_row - 1)
        } else {
            None
        };
        let below_known = if missing_row + 1 < full_h {
            Some(missing_row + 1)
        } else {
            None
        };
        let dst_start = missing_row * w;

        for x in 0..w {
            let prev_field_line = mi.min(prev_luma.len().saturating_div(w).saturating_sub(1));
            let prev_val = if prev_luma.len() > prev_field_line * w + x {
                prev_luma[prev_field_line * w + x]
            } else {
                0u8
            };

            let bob_val = bob_out[dst_start + x];
            let above_val = above_known.map(|r| bob_out[r * w + x]).unwrap_or(bob_val);
            let below_val = below_known.map(|r| bob_out[r * w + x]).unwrap_or(bob_val);
            let spatial_avg = (above_val as u16 + below_val as u16) / 2;
            let motion = (spatial_avg as i16 - prev_val as i16).unsigned_abs() as u8;

            out[dst_start + x] = if motion > threshold {
                bob_val
            } else {
                prev_val
            };
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_order_variants() {
        assert_ne!(FieldOrder::TopFieldFirst, FieldOrder::BottomFieldFirst);
        assert_ne!(FieldOrder::TopFieldFirst, FieldOrder::Progressive);
    }

    #[test]
    fn test_frame_rate_multiplier_bob() {
        assert_eq!(DeinterlaceMethod::Bob.output_frame_rate_multiplier(), 2);
    }

    #[test]
    fn test_frame_rate_multiplier_yadif() {
        assert_eq!(DeinterlaceMethod::Yadif.output_frame_rate_multiplier(), 2);
    }

    #[test]
    fn test_frame_rate_multiplier_blend() {
        assert_eq!(DeinterlaceMethod::Blend.output_frame_rate_multiplier(), 1);
    }

    #[test]
    fn test_frame_rate_multiplier_field_drop() {
        assert_eq!(
            DeinterlaceMethod::FieldDrop.output_frame_rate_multiplier(),
            1
        );
    }

    #[test]
    fn test_config_is_temporal_true() {
        let cfg =
            DeinterlaceConfig::new(FieldOrder::TopFieldFirst, DeinterlaceMethod::MotionAdaptive);
        assert!(cfg.is_temporal());
    }

    #[test]
    fn test_config_is_temporal_false() {
        let cfg = DeinterlaceConfig::new(FieldOrder::TopFieldFirst, DeinterlaceMethod::Blend);
        assert!(!cfg.is_temporal());
    }

    #[test]
    fn test_progressive_passthrough() {
        let cfg = DeinterlaceConfig::new(FieldOrder::Progressive, DeinterlaceMethod::FieldDrop);
        assert!(cfg.is_progressive_passthrough());
    }

    #[test]
    fn test_not_progressive_passthrough() {
        let cfg = DeinterlaceConfig::new(FieldOrder::TopFieldFirst, DeinterlaceMethod::FieldDrop);
        assert!(!cfg.is_progressive_passthrough());
    }

    #[test]
    fn test_process_field_progressive_passthrough() {
        let cfg = DeinterlaceConfig::new(FieldOrder::Progressive, DeinterlaceMethod::Bob);
        let proc = DeinterlaceProcessor::new(cfg);
        let field = VideoField::blank(0, 4, 4);
        let out = proc.process_field(&field);
        assert_eq!(out, vec![0u8; 16]);
    }

    #[test]
    fn test_process_field_field_drop() {
        let cfg = DeinterlaceConfig::new(FieldOrder::TopFieldFirst, DeinterlaceMethod::FieldDrop);
        let proc = DeinterlaceProcessor::new(cfg);
        let mut field = VideoField::blank(0, 2, 2);
        field.luma = vec![10, 20, 30, 40];
        let out = proc.process_field(&field);
        assert_eq!(out, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_process_field_blend() {
        let cfg = DeinterlaceConfig::new(FieldOrder::BottomFieldFirst, DeinterlaceMethod::Blend);
        let proc = DeinterlaceProcessor::new(cfg);
        let mut field = VideoField::blank(1, 2, 2);
        field.luma = vec![100, 200, 50, 150];
        let out = proc.process_field(&field);
        assert_eq!(out.len(), 8);
        assert_eq!(out[0..2], [100, 200]);
        assert_eq!(out[2..4], [100, 200]);
        assert_eq!(out[4..6], [75, 175]);
        assert_eq!(out[6..8], [50, 150]);
    }

    #[test]
    fn test_output_fps_bob() {
        let cfg = DeinterlaceConfig::new(FieldOrder::TopFieldFirst, DeinterlaceMethod::Bob);
        let proc = DeinterlaceProcessor::new(cfg);
        let fps = proc.output_fps(25, 1);
        assert!((fps - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_output_fps_blend() {
        let cfg = DeinterlaceConfig::new(FieldOrder::TopFieldFirst, DeinterlaceMethod::Blend);
        let proc = DeinterlaceProcessor::new(cfg);
        let fps = proc.output_fps(25, 1);
        assert!((fps - 25.0).abs() < 1e-9);
    }

    // -- MotionAdaptiveProcessor tests --------------------------------------

    #[test]
    fn test_motion_adaptive_config_default_threshold() {
        let cfg = MotionAdaptiveConfig::new(FieldOrder::TopFieldFirst);
        assert_eq!(cfg.motion_threshold, 16);
    }

    #[test]
    fn test_motion_adaptive_config_custom_threshold() {
        let cfg = MotionAdaptiveConfig::new(FieldOrder::TopFieldFirst).with_threshold(32);
        assert_eq!(cfg.motion_threshold, 32);
    }

    #[test]
    fn test_motion_adaptive_first_field_bob_fallback() {
        let cfg = MotionAdaptiveConfig::new(FieldOrder::TopFieldFirst);
        let mut proc = MotionAdaptiveProcessor::new(cfg);
        let field = VideoField::blank(0, 4, 4);
        let out = proc.process_field(&field);
        assert_eq!(out.len(), 4 * 8);
    }

    #[test]
    fn test_motion_adaptive_second_call_same_size() {
        let cfg = MotionAdaptiveConfig::new(FieldOrder::TopFieldFirst);
        let mut proc = MotionAdaptiveProcessor::new(cfg);
        let field = VideoField::blank(0, 4, 4);
        let _ = proc.process_field(&field);
        let out = proc.process_field(&field);
        assert_eq!(out.len(), 4 * 8);
    }

    #[test]
    fn test_motion_adaptive_static_content_weaves() {
        let cfg = MotionAdaptiveConfig::new(FieldOrder::TopFieldFirst).with_threshold(8);
        let mut proc = MotionAdaptiveProcessor::new(cfg);
        let mut field = VideoField::blank(0, 4, 4);
        field.luma = vec![128u8; 4 * 4];
        let _ = proc.process_field(&field);
        let out = proc.process_field(&field);
        assert_eq!(out.len(), 4 * 8);
        for &v in &out {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_motion_adaptive_reset_clears_reference() {
        let cfg = MotionAdaptiveConfig::new(FieldOrder::TopFieldFirst);
        let mut proc = MotionAdaptiveProcessor::new(cfg);
        let field = VideoField::blank(0, 4, 4);
        let _ = proc.process_field(&field);
        proc.reset();
        let out = proc.process_field(&field);
        assert_eq!(out.len(), 4 * 8);
    }

    #[test]
    fn test_motion_adaptive_motion_threshold_zero_always_bobs() {
        let cfg = MotionAdaptiveConfig::new(FieldOrder::TopFieldFirst).with_threshold(0);
        let mut proc = MotionAdaptiveProcessor::new(cfg);
        let mut field1 = VideoField::blank(0, 4, 4);
        field1.luma = vec![50u8; 16];
        let _ = proc.process_field(&field1);
        let mut field2 = VideoField::blank(0, 4, 4);
        field2.luma = vec![100u8; 16];
        let out = proc.process_field(&field2);
        assert_eq!(out.len(), 4 * 8);
        let bob = bob_deinterlace(&field2);
        assert_eq!(out, bob);
    }

    #[test]
    fn test_motion_adaptive_dimension_mismatch_falls_back_to_bob() {
        let cfg = MotionAdaptiveConfig::new(FieldOrder::TopFieldFirst);
        let mut proc = MotionAdaptiveProcessor::new(cfg);
        let field_large = VideoField::blank(0, 8, 8);
        let _ = proc.process_field(&field_large);
        let field_small = VideoField::blank(0, 4, 4);
        let out = proc.process_field(&field_small);
        let bob = bob_deinterlace(&field_small);
        assert_eq!(out, bob);
    }

    #[test]
    fn test_motion_adaptive_processor_threshold_accessor() {
        let cfg = MotionAdaptiveConfig::new(FieldOrder::TopFieldFirst).with_threshold(24);
        let proc = MotionAdaptiveProcessor::new(cfg);
        assert_eq!(proc.motion_threshold(), 24);
    }

    #[test]
    fn test_motion_adaptive_two_field_orders() {
        for &field_order in &[FieldOrder::TopFieldFirst, FieldOrder::BottomFieldFirst] {
            let cfg = MotionAdaptiveConfig::new(field_order);
            let mut proc = MotionAdaptiveProcessor::new(cfg);
            let field = VideoField::blank(0, 4, 4);
            let out = proc.process_field(&field);
            assert_eq!(out.len(), 4 * 8);
        }
    }

    #[test]
    fn test_motion_adaptive_high_motion_uses_bob() {
        let cfg = MotionAdaptiveConfig::new(FieldOrder::TopFieldFirst).with_threshold(1);
        let mut proc = MotionAdaptiveProcessor::new(cfg);
        let mut field1 = VideoField::blank(0, 4, 4);
        field1.luma = vec![0u8; 16];
        let _ = proc.process_field(&field1);
        let mut field2 = VideoField::blank(0, 4, 4);
        field2.luma = vec![200u8; 16];
        let out = proc.process_field(&field2);
        let bob = bob_deinterlace(&field2);
        assert_eq!(out, bob);
    }
}
