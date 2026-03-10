//! Super-resolution upscaling for AV1.
//!
//! AV1 supports encoding at a reduced horizontal resolution and upscaling
//! during decode. This provides coding efficiency gains at minimal quality
//! loss using Lanczos-like filtering.

#![forbid(unsafe_code)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::identity_op)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::single_match_else)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::many_single_char_names)]

use super::pipeline::FrameContext;
use super::{FrameBuffer, PlaneBuffer, ReconstructResult, ReconstructionError};

// =============================================================================
// Constants
// =============================================================================

/// Minimum super-resolution scale denominator.
pub const SUPERRES_DENOM_MIN: u8 = 9;

/// Maximum super-resolution scale denominator (no scaling).
pub const SUPERRES_DENOM_MAX: u8 = 16;

/// Number of super-resolution filter taps.
pub const SUPERRES_FILTER_TAPS: usize = 8;

/// Super-resolution filter bits.
pub const SUPERRES_FILTER_BITS: u8 = 6;

/// Super-resolution filter offset.
pub const SUPERRES_FILTER_OFFSET: i32 = 1 << (SUPERRES_FILTER_BITS - 1);

// =============================================================================
// Upscale Method
// =============================================================================

/// Upscaling method.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UpscaleMethod {
    /// Lanczos-like filter (default for AV1).
    #[default]
    Lanczos,
    /// Bilinear interpolation (simpler, lower quality).
    Bilinear,
    /// Bicubic interpolation.
    Bicubic,
    /// Nearest neighbor (fastest, lowest quality).
    Nearest,
}

impl UpscaleMethod {
    /// Get filter kernel size.
    #[must_use]
    pub const fn kernel_size(self) -> usize {
        match self {
            Self::Lanczos => 8,
            Self::Bicubic => 4,
            Self::Bilinear => 2,
            Self::Nearest => 1,
        }
    }
}

// =============================================================================
// Super-Res Configuration
// =============================================================================

/// Configuration for super-resolution.
#[derive(Clone, Debug)]
pub struct SuperResConfig {
    /// Scale denominator (9-16, 16 = no scaling).
    pub denominator: u8,
    /// Original (encoded) width.
    pub encoded_width: u32,
    /// Target (output) width.
    pub upscaled_width: u32,
    /// Frame height (unchanged).
    pub height: u32,
    /// Upscale method.
    pub method: UpscaleMethod,
}

impl Default for SuperResConfig {
    fn default() -> Self {
        Self {
            denominator: SUPERRES_DENOM_MAX,
            encoded_width: 0,
            upscaled_width: 0,
            height: 0,
            method: UpscaleMethod::Lanczos,
        }
    }
}

impl SuperResConfig {
    /// Create a new super-res configuration.
    #[must_use]
    pub fn new(encoded_width: u32, upscaled_width: u32, height: u32) -> Self {
        let denominator = if upscaled_width > 0 && encoded_width > 0 {
            let ratio = (encoded_width as f32 * 16.0 / upscaled_width as f32).round() as u8;
            ratio.clamp(SUPERRES_DENOM_MIN, SUPERRES_DENOM_MAX)
        } else {
            SUPERRES_DENOM_MAX
        };

        Self {
            denominator,
            encoded_width,
            upscaled_width,
            height,
            method: UpscaleMethod::Lanczos,
        }
    }

    /// Create from scale denominator.
    #[must_use]
    pub fn from_denominator(denominator: u8, encoded_width: u32, height: u32) -> Self {
        let denom = denominator.clamp(SUPERRES_DENOM_MIN, SUPERRES_DENOM_MAX);
        let upscaled_width = (encoded_width as u64 * 16 / u64::from(denom)) as u32;

        Self {
            denominator: denom,
            encoded_width,
            upscaled_width,
            height,
            method: UpscaleMethod::Lanczos,
        }
    }

    /// Get the scale factor.
    #[must_use]
    pub fn scale_factor(&self) -> f32 {
        16.0 / f32::from(self.denominator)
    }

    /// Check if super-res is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.denominator < SUPERRES_DENOM_MAX
    }

    /// Set upscale method.
    #[must_use]
    pub const fn with_method(mut self, method: UpscaleMethod) -> Self {
        self.method = method;
        self
    }

    /// Calculate the source x position for a target x position.
    #[must_use]
    pub fn source_x(&self, target_x: u32) -> (u32, u32) {
        // Calculate with higher precision
        let step = (u64::from(self.denominator) << 14) / 16;
        let offset = step / 2;
        let src_pos = u64::from(target_x) * step + offset;

        let integer = (src_pos >> 14) as u32;
        let fraction = ((src_pos & 0x3FFF) >> 8) as u32; // 6 bits of fraction

        (integer, fraction)
    }
}

// =============================================================================
// Lanczos Filter Kernels
// =============================================================================

/// Pre-computed Lanczos filter kernels for different phases.
const SUPERRES_FILTER_KERNELS: [[i16; SUPERRES_FILTER_TAPS]; 64] = [
    // Phase 0
    [0, 0, 0, 64, 0, 0, 0, 0],
    [0, 1, -3, 63, 4, -1, 0, 0],
    [0, 1, -5, 62, 8, -2, 0, 0],
    [-1, 2, -7, 61, 12, -3, 0, 0],
    [-1, 2, -9, 59, 17, -4, 0, 0],
    [-1, 3, -10, 57, 21, -5, -1, 0],
    [-1, 3, -11, 55, 26, -6, -1, -1],
    [-1, 3, -12, 52, 30, -6, -1, -1],
    // Phase 8
    [-1, 4, -13, 50, 34, -7, -2, -1],
    [-1, 4, -13, 47, 38, -7, -2, -2],
    [-1, 4, -13, 44, 42, -8, -2, -2],
    [-1, 4, -13, 41, 45, -8, -2, -2],
    [-1, 4, -13, 38, 48, -8, -2, -2],
    [-1, 4, -12, 35, 51, -8, -2, -3],
    [-1, 4, -12, 32, 53, -8, -2, -2],
    [-1, 4, -11, 29, 55, -7, -2, -3],
    // Phase 16
    [-1, 3, -10, 26, 57, -7, -2, -2],
    [-1, 3, -9, 23, 58, -6, -2, -2],
    [0, 3, -8, 21, 59, -5, -2, -4],
    [0, 3, -7, 18, 60, -4, -2, -4],
    [0, 2, -6, 15, 61, -3, -1, -4],
    [0, 2, -5, 13, 61, -2, -1, -4],
    [0, 2, -4, 10, 62, -1, -1, -4],
    [0, 1, -3, 8, 62, 0, 0, -4],
    // Phase 24
    [0, 1, -2, 5, 63, 1, 0, -4],
    [0, 1, -1, 3, 63, 2, 0, -4],
    [0, 0, 0, 1, 63, 3, 0, -3],
    [0, 0, 0, 0, 64, 4, 0, -4],
    [0, 0, 1, -2, 63, 5, 0, -3],
    [0, 0, 1, -3, 63, 7, 0, -4],
    [0, 0, 2, -4, 62, 9, -1, -4],
    [0, 0, 2, -6, 61, 11, -1, -3],
    // Phase 32 (center)
    [0, 0, 2, -7, 60, 13, -2, -2],
    [0, 0, 3, -8, 59, 15, -2, -3],
    [0, 0, 3, -9, 57, 18, -3, -2],
    [0, -1, 3, -10, 55, 21, -3, -1],
    [0, -1, 4, -11, 53, 24, -4, -1],
    [0, -1, 4, -11, 51, 27, -4, -2],
    [-1, -1, 4, -12, 49, 30, -4, -1],
    [-1, -1, 4, -12, 46, 33, -4, -1],
    // Phase 40
    [-1, -1, 5, -12, 44, 36, -5, -2],
    [-1, -2, 5, -12, 41, 39, -5, -1],
    [-1, -2, 5, -12, 38, 42, -5, -1],
    [-1, -2, 5, -11, 35, 44, -5, -1],
    [-1, -2, 5, -11, 33, 46, -5, -1],
    [-1, -2, 5, -10, 30, 48, -4, -2],
    [-1, -2, 5, -10, 27, 50, -4, -1],
    [-1, -2, 5, -9, 25, 51, -4, -1],
    // Phase 48
    [-1, -2, 4, -9, 22, 53, -3, 0],
    [-1, -2, 4, -8, 20, 54, -3, 0],
    [-1, -2, 4, -7, 17, 55, -2, 0],
    [-1, -2, 4, -6, 15, 56, -1, -1],
    [0, -2, 4, -5, 12, 57, 0, -2],
    [0, -2, 3, -4, 10, 58, 1, -2],
    [0, -2, 3, -3, 8, 59, 2, -3],
    [0, -1, 3, -2, 6, 59, 3, -4],
    // Phase 56
    [0, -1, 2, -1, 4, 60, 4, -4],
    [0, -1, 2, 0, 2, 60, 5, -4],
    [0, -1, 1, 0, 0, 61, 6, -3],
    [0, 0, 1, 1, -1, 60, 7, -4],
    [0, 0, 1, 1, -2, 60, 8, -4],
    [0, 0, 0, 2, -3, 59, 10, -4],
    [0, 0, 0, 2, -4, 58, 12, -4],
    [0, 0, 0, 3, -5, 57, 14, -5],
];

/// Get filter kernel for a given phase.
fn get_filter_kernel(phase: usize) -> &'static [i16; SUPERRES_FILTER_TAPS] {
    &SUPERRES_FILTER_KERNELS[phase.min(63)]
}

// =============================================================================
// Super-Res Upscaler
// =============================================================================

/// Super-resolution upscaler.
#[derive(Debug)]
pub struct SuperResUpscaler {
    /// Current configuration.
    config: Option<SuperResConfig>,
    /// Temporary row buffer.
    row_buffer: Vec<i32>,
}

impl Default for SuperResUpscaler {
    fn default() -> Self {
        Self::new()
    }
}

impl SuperResUpscaler {
    /// Create a new super-res upscaler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: None,
            row_buffer: Vec::new(),
        }
    }

    /// Create with configuration.
    #[must_use]
    pub fn with_config(config: SuperResConfig) -> Self {
        let buffer_size = config.upscaled_width as usize + SUPERRES_FILTER_TAPS;
        Self {
            config: Some(config),
            row_buffer: vec![0; buffer_size],
        }
    }

    /// Set configuration.
    pub fn set_config(&mut self, config: SuperResConfig) {
        let buffer_size = config.upscaled_width as usize + SUPERRES_FILTER_TAPS;
        self.row_buffer.resize(buffer_size, 0);
        self.config = Some(config);
    }

    /// Get current configuration.
    #[must_use]
    pub fn config(&self) -> Option<&SuperResConfig> {
        self.config.as_ref()
    }

    /// Apply super-resolution to a frame.
    ///
    /// # Errors
    ///
    /// Returns error if super-res application fails.
    pub fn apply(
        &mut self,
        frame: &mut FrameBuffer,
        context: &FrameContext,
    ) -> ReconstructResult<()> {
        // Check if super-res is needed
        if !context.needs_super_res() {
            return Ok(());
        }

        // Clone config to avoid borrow issues
        let config = self.config.clone().ok_or_else(|| {
            ReconstructionError::InvalidInput("Super-res config not set".to_string())
        })?;

        if !config.is_enabled() {
            return Ok(());
        }

        let bd = frame.bit_depth();

        // Upscale Y plane
        self.upscale_plane(frame.y_plane_mut(), &config, bd)?;

        // Upscale chroma planes (with adjusted dimensions)
        if let Some(u) = frame.u_plane_mut() {
            self.upscale_plane(u, &config, bd)?;
        }
        if let Some(v) = frame.v_plane_mut() {
            self.upscale_plane(v, &config, bd)?;
        }

        Ok(())
    }

    /// Upscale a single plane.
    fn upscale_plane(
        &mut self,
        plane: &mut PlaneBuffer,
        config: &SuperResConfig,
        bd: u8,
    ) -> ReconstructResult<()> {
        let src_width = plane.width() as usize;
        let height = plane.height() as usize;
        let target_width = config.upscaled_width as usize;

        if target_width <= src_width {
            return Ok(()); // No upscaling needed
        }

        match config.method {
            UpscaleMethod::Lanczos => {
                self.upscale_lanczos(plane, src_width, target_width, height, bd)
            }
            UpscaleMethod::Bilinear => {
                self.upscale_bilinear(plane, src_width, target_width, height, bd)
            }
            UpscaleMethod::Bicubic => {
                self.upscale_bicubic(plane, src_width, target_width, height, bd)
            }
            UpscaleMethod::Nearest => self.upscale_nearest(plane, src_width, target_width, height),
        }
    }

    /// Upscale using Lanczos-like filter.
    fn upscale_lanczos(
        &mut self,
        plane: &mut PlaneBuffer,
        src_width: usize,
        target_width: usize,
        height: usize,
        bd: u8,
    ) -> ReconstructResult<()> {
        let max_val = (1i32 << bd) - 1;
        let config = self
            .config
            .as_ref()
            .expect("config initialized before upscale_lanczos is called");

        // Process each row
        for y in 0..height {
            let row = plane.row(y as u32);

            // Upscale row
            for x in 0..target_width {
                let (src_x, phase) = config.source_x(x as u32);
                let kernel = get_filter_kernel(phase as usize);

                let mut sum: i32 = 0;
                for (i, &k) in kernel.iter().enumerate() {
                    let sx = (src_x as i32 + i as i32 - 3).clamp(0, src_width as i32 - 1) as usize;
                    sum += i32::from(row[sx]) * i32::from(k);
                }

                // Round and clamp
                let result =
                    ((sum + SUPERRES_FILTER_OFFSET) >> SUPERRES_FILTER_BITS).clamp(0, max_val);
                self.row_buffer[x] = result;
            }

            // Write back (note: this would require resizing the plane)
            // For now, just write what fits
            let dst_row = plane.row_mut(y as u32);
            let write_width = target_width.min(dst_row.len());
            for x in 0..write_width {
                dst_row[x] = self.row_buffer[x] as i16;
            }
        }

        Ok(())
    }

    /// Upscale using bilinear interpolation.
    fn upscale_bilinear(
        &mut self,
        plane: &mut PlaneBuffer,
        src_width: usize,
        target_width: usize,
        height: usize,
        bd: u8,
    ) -> ReconstructResult<()> {
        let max_val = (1i32 << bd) - 1;
        let scale = src_width as f32 / target_width as f32;

        for y in 0..height {
            let row = plane.row(y as u32);

            for x in 0..target_width {
                let src_x = x as f32 * scale;
                let x0 = src_x.floor() as usize;
                let x1 = (x0 + 1).min(src_width - 1);
                let frac = src_x.fract();

                let v0 = i32::from(row[x0]);
                let v1 = i32::from(row[x1]);
                let result = ((v0 as f32 * (1.0 - frac) + v1 as f32 * frac).round() as i32)
                    .clamp(0, max_val);
                self.row_buffer[x] = result;
            }

            let dst_row = plane.row_mut(y as u32);
            let write_width = target_width.min(dst_row.len());
            for x in 0..write_width {
                dst_row[x] = self.row_buffer[x] as i16;
            }
        }

        Ok(())
    }

    /// Upscale using bicubic interpolation.
    fn upscale_bicubic(
        &mut self,
        plane: &mut PlaneBuffer,
        src_width: usize,
        target_width: usize,
        height: usize,
        bd: u8,
    ) -> ReconstructResult<()> {
        let max_val = (1i32 << bd) - 1;
        let scale = src_width as f32 / target_width as f32;

        for y in 0..height {
            let row = plane.row(y as u32);

            for x in 0..target_width {
                let src_x = x as f32 * scale;
                let x1 = src_x.floor() as i32;
                let frac = src_x.fract();

                // Bicubic weights
                let w = bicubic_weights(frac);

                let mut sum = 0.0f32;
                for (i, &weight) in w.iter().enumerate() {
                    let sx = (x1 + i as i32 - 1).clamp(0, src_width as i32 - 1) as usize;
                    sum += f32::from(row[sx]) * weight;
                }

                let result = (sum.round() as i32).clamp(0, max_val);
                self.row_buffer[x] = result;
            }

            let dst_row = plane.row_mut(y as u32);
            let write_width = target_width.min(dst_row.len());
            for x in 0..write_width {
                dst_row[x] = self.row_buffer[x] as i16;
            }
        }

        Ok(())
    }

    /// Upscale using nearest neighbor.
    fn upscale_nearest(
        &mut self,
        plane: &mut PlaneBuffer,
        src_width: usize,
        target_width: usize,
        height: usize,
    ) -> ReconstructResult<()> {
        let scale = src_width as f32 / target_width as f32;

        for y in 0..height {
            let row = plane.row(y as u32);

            for x in 0..target_width {
                let src_x = ((x as f32 * scale).round() as usize).min(src_width - 1);
                self.row_buffer[x] = i32::from(row[src_x]);
            }

            let dst_row = plane.row_mut(y as u32);
            let write_width = target_width.min(dst_row.len());
            for x in 0..write_width {
                dst_row[x] = self.row_buffer[x] as i16;
            }
        }

        Ok(())
    }
}

/// Calculate bicubic interpolation weights.
fn bicubic_weights(t: f32) -> [f32; 4] {
    let a = -0.5f32; // Bicubic parameter

    [
        a * t * t * t - 2.0 * a * t * t + a * t,
        (a + 2.0) * t * t * t - (a + 3.0) * t * t + 1.0,
        -(a + 2.0) * t * t * t + (2.0 * a + 3.0) * t * t - a * t,
        -a * t * t * t + a * t * t,
    ]
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruct::ChromaSubsampling;

    #[test]
    fn test_upscale_method() {
        assert_eq!(UpscaleMethod::Lanczos.kernel_size(), 8);
        assert_eq!(UpscaleMethod::Bicubic.kernel_size(), 4);
        assert_eq!(UpscaleMethod::Bilinear.kernel_size(), 2);
        assert_eq!(UpscaleMethod::Nearest.kernel_size(), 1);
    }

    #[test]
    fn test_super_res_config_default() {
        let config = SuperResConfig::default();
        assert_eq!(config.denominator, SUPERRES_DENOM_MAX);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_super_res_config_new() {
        let config = SuperResConfig::new(1600, 1920, 1080);
        assert!(config.is_enabled());
        assert!(config.scale_factor() > 1.0);
    }

    #[test]
    fn test_super_res_config_from_denominator() {
        let config = SuperResConfig::from_denominator(12, 1440, 1080);
        assert_eq!(config.denominator, 12);
        assert!(config.is_enabled());
        // 1440 * 16 / 12 = 1920
        assert_eq!(config.upscaled_width, 1920);
    }

    #[test]
    fn test_super_res_config_no_scaling() {
        let config = SuperResConfig::from_denominator(16, 1920, 1080);
        assert!(!config.is_enabled());
        assert!((config.scale_factor() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_super_res_config_source_x() {
        let config = SuperResConfig::from_denominator(12, 1440, 1080);
        let (src_x, phase) = config.source_x(0);
        assert_eq!(src_x, 0);
        let _ = phase; // Phase depends on implementation details
    }

    #[test]
    fn test_super_res_upscaler_creation() {
        let upscaler = SuperResUpscaler::new();
        assert!(upscaler.config().is_none());
    }

    #[test]
    fn test_super_res_upscaler_with_config() {
        let config = SuperResConfig::from_denominator(12, 1440, 1080);
        let upscaler = SuperResUpscaler::with_config(config);
        assert!(upscaler.config().is_some());
    }

    #[test]
    fn test_super_res_upscaler_set_config() {
        let mut upscaler = SuperResUpscaler::new();
        let config = SuperResConfig::from_denominator(12, 1440, 1080);
        upscaler.set_config(config);
        assert!(upscaler.config().is_some());
    }

    #[test]
    fn test_super_res_apply_disabled() {
        let mut frame = FrameBuffer::new(64, 64, 8, ChromaSubsampling::Cs420);
        let context = FrameContext::new(64, 64); // No super-res needed

        let mut upscaler = SuperResUpscaler::new();
        let result = upscaler.apply(&mut frame, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bicubic_weights() {
        let w = bicubic_weights(0.0);
        // At t=0, weights should sum to 1 with w[1] being dominant
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        let w_half = bicubic_weights(0.5);
        let sum_half: f32 = w_half.iter().sum();
        assert!((sum_half - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_filter_kernel() {
        let kernel = get_filter_kernel(0);
        // Phase 0 should be identity-like (center tap dominant)
        let sum: i16 = kernel.iter().sum();
        assert_eq!(sum, 64); // Should sum to 64 (1.0 in fixed point)
    }

    #[test]
    fn test_constants() {
        assert_eq!(SUPERRES_DENOM_MIN, 9);
        assert_eq!(SUPERRES_DENOM_MAX, 16);
        assert_eq!(SUPERRES_FILTER_TAPS, 8);
        assert_eq!(SUPERRES_FILTER_BITS, 6);
    }
}
