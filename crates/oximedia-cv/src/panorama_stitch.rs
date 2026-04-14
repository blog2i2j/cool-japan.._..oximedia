//! Simple offset-based panorama stitching with linear alpha blending.
//!
//! This module provides a pure-Rust implementation of panorama image stitching
//! that places frames onto a shared canvas using caller-supplied offsets and
//! blends overlapping regions with a configurable linear feathering zone.
//!
//! # Algorithm
//!
//! 1. Compute the bounding rectangle of all frames from their offsets.
//! 2. Allocate an RGBA canvas covering that rectangle.
//! 3. For each frame (in order), paint its pixels onto the canvas.
//!    In the blend zone at each frame's leading/trailing edge the contribution
//!    is linearly interpolated between the existing canvas value and the new
//!    frame value.
//!
//! # Example
//!
//! ```rust
//! use oximedia_cv::panorama_stitch::{PanoramaStitcher, PanoramaConfig, StitchFrame, PanoFormat};
//!
//! let config = PanoramaConfig {
//!     blend_width_px: 8,
//!     output_format: PanoFormat::Flat,
//! };
//!
//! // Two 10x10 grayscale frames placed side-by-side with a 2-px overlap
//! let frame_a = StitchFrame {
//!     image: vec![200u8; 10 * 10],
//!     width: 10,
//!     height: 10,
//!     offset_x: 0,
//!     offset_y: 0,
//! };
//! let frame_b = StitchFrame {
//!     image: vec![100u8; 10 * 10],
//!     width: 10,
//!     height: 10,
//!     offset_x: 8,
//!     offset_y: 0,
//! };
//!
//! let stitcher = PanoramaStitcher::new(config);
//! let pano = stitcher.stitch(&[frame_a, frame_b]).expect("stitch failed");
//! assert_eq!(pano.height, 10);
//! assert!(pano.width >= 10);
//! ```

use crate::error::CvError;

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors that can arise during panorama stitching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StitchError {
    /// The `frames` slice was empty.
    NoFrames,
    /// Two frames have inconsistent channel counts (bpp mismatch).
    DimensionMismatch,
    /// A frame's offset places it entirely outside representable coordinates.
    OffsetOutOfBounds,
}

impl std::fmt::Display for StitchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoFrames => write!(f, "no frames supplied for stitching"),
            Self::DimensionMismatch => write!(f, "frames have incompatible pixel formats"),
            Self::OffsetOutOfBounds => write!(f, "frame offset is out of representable range"),
        }
    }
}

impl std::error::Error for StitchError {}

// ── Output format ─────────────────────────────────────────────────────────────

/// Output projection format for the stitched panorama.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PanoFormat {
    /// Standard flat (rectilinear) projection — the canvas is a plain 2-D grid.
    #[default]
    Flat,
    /// Equirectangular projection — the canvas spans 360° × 180°.
    ///
    /// In this implementation the canvas layout is identical to `Flat`; the
    /// distinction is preserved so that downstream consumers can attach the
    /// correct metadata (e.g. XMP spherical).
    Equirectangular,
}

// ── Input frame ───────────────────────────────────────────────────────────────

/// A single source frame to be placed on the panorama canvas.
///
/// The pixel buffer may be grayscale (1 byte/px), RGB (3 bytes/px) or
/// RGBA (4 bytes/px).  All frames in one [`PanoramaStitcher::stitch`] call
/// must use the same bytes-per-pixel count.
#[derive(Debug, Clone)]
pub struct StitchFrame {
    /// Raw pixel data in row-major order.
    pub image: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Horizontal offset (in pixels) of the frame's top-left corner on the
    /// shared canvas.  May be negative.
    pub offset_x: i64,
    /// Vertical offset (in pixels) of the frame's top-left corner on the
    /// shared canvas.  May be negative.
    pub offset_y: i64,
}

impl StitchFrame {
    /// Number of bytes per pixel inferred from the pixel buffer and dimensions.
    ///
    /// Returns `None` if the dimensions are zero or the ratio is non-integer.
    #[must_use]
    pub fn bytes_per_pixel(&self) -> Option<usize> {
        let n = (self.width as usize).checked_mul(self.height as usize)?;
        if n == 0 || self.image.len() % n != 0 {
            return None;
        }
        Some(self.image.len() / n)
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

/// The stitched panorama output.
#[derive(Debug, Clone)]
pub struct StitchedPanorama {
    /// Raw pixel data in row-major order.
    ///
    /// The channel count matches the channel count of the input frames.
    pub data: Vec<u8>,
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Bytes per pixel (same as input frames).
    pub channels: usize,
    /// Output format requested by the caller.
    pub format: PanoFormat,
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for [`PanoramaStitcher`].
#[derive(Debug, Clone)]
pub struct PanoramaConfig {
    /// Width of the linear alpha-blend zone at each frame boundary (pixels).
    ///
    /// A value of `0` disables blending (hard-edge composition).
    pub blend_width_px: u32,
    /// Projection format for the output canvas.
    pub output_format: PanoFormat,
}

impl Default for PanoramaConfig {
    fn default() -> Self {
        Self {
            blend_width_px: 16,
            output_format: PanoFormat::Flat,
        }
    }
}

// ── Stitcher ──────────────────────────────────────────────────────────────────

/// Panorama stitcher that composites frames using explicit pixel offsets.
#[derive(Debug, Clone)]
pub struct PanoramaStitcher {
    config: PanoramaConfig,
}

impl PanoramaStitcher {
    /// Create a new stitcher with the given configuration.
    #[must_use]
    pub fn new(config: PanoramaConfig) -> Self {
        Self { config }
    }

    /// Create a stitcher with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(PanoramaConfig::default())
    }

    /// Stitch `frames` onto a shared canvas.
    ///
    /// # Errors
    ///
    /// Returns [`StitchError::NoFrames`] if the slice is empty,
    /// [`StitchError::DimensionMismatch`] if frames have different bytes-per-pixel,
    /// and [`StitchError::OffsetOutOfBounds`] if offsets exceed `i32::MAX` range.
    #[allow(clippy::cast_precision_loss)]
    pub fn stitch(&self, frames: &[StitchFrame]) -> Result<StitchedPanorama, StitchError> {
        if frames.is_empty() {
            return Err(StitchError::NoFrames);
        }

        // ── Determine bytes-per-pixel (must be consistent) ────────────────────
        let bpp = frames[0]
            .bytes_per_pixel()
            .ok_or(StitchError::DimensionMismatch)?;
        for f in frames.iter().skip(1) {
            if f.bytes_per_pixel() != Some(bpp) {
                return Err(StitchError::DimensionMismatch);
            }
        }

        // ── Compute canvas bounds from all frame offsets ───────────────────────
        let mut canvas_x0 = i64::MAX;
        let mut canvas_y0 = i64::MAX;
        let mut canvas_x1 = i64::MIN;
        let mut canvas_y1 = i64::MIN;

        for f in frames {
            let fx1 = f
                .offset_x
                .checked_add(f.width as i64)
                .ok_or(StitchError::OffsetOutOfBounds)?;
            let fy1 = f
                .offset_y
                .checked_add(f.height as i64)
                .ok_or(StitchError::OffsetOutOfBounds)?;
            canvas_x0 = canvas_x0.min(f.offset_x);
            canvas_y0 = canvas_y0.min(f.offset_y);
            canvas_x1 = canvas_x1.max(fx1);
            canvas_y1 = canvas_y1.max(fy1);
        }

        let canvas_w = (canvas_x1 - canvas_x0) as u32;
        let canvas_h = (canvas_y1 - canvas_y0) as u32;

        if canvas_w == 0 || canvas_h == 0 {
            return Err(StitchError::OffsetOutOfBounds);
        }

        // ── Allocate canvas + accumulation buffer ────────────────────────────
        // We use an f32 accumulator and a weight buffer to support blending.
        let n_px = (canvas_w as usize) * (canvas_h as usize);
        let mut accum: Vec<f32> = vec![0.0; n_px * bpp];
        let mut weight: Vec<f32> = vec![0.0; n_px];

        let blend_w = self.config.blend_width_px as usize;

        for frame in frames {
            let fw = frame.width as usize;
            let fh = frame.height as usize;

            // Frame origin on the canvas (guaranteed non-negative after bounds check)
            let cx0 = (frame.offset_x - canvas_x0) as usize;
            let cy0 = (frame.offset_y - canvas_y0) as usize;

            for fy in 0..fh {
                for fx in 0..fw {
                    let frame_px_idx = fy * fw + fx;
                    let canvas_x = cx0 + fx;
                    let canvas_y = cy0 + fy;
                    let canvas_px_idx = canvas_y * (canvas_w as usize) + canvas_x;

                    // Compute blend weight based on distance to left/right edges
                    // of this frame on the canvas.
                    let dist_left = fx;
                    let dist_right = fw.saturating_sub(1).saturating_sub(fx);
                    let dist_min = dist_left.min(dist_right);

                    let alpha = if blend_w == 0 {
                        1.0f32
                    } else {
                        (dist_min as f32 / blend_w as f32).clamp(0.0, 1.0)
                    };

                    let frame_base = frame_px_idx * bpp;
                    let canvas_base = canvas_px_idx * bpp;

                    for c in 0..bpp {
                        let val = f32::from(frame.image[frame_base + c]);
                        accum[canvas_base + c] += val * alpha;
                    }
                    weight[canvas_px_idx] += alpha;
                }
            }
        }

        // ── Normalise accumulator → output bytes ─────────────────────────────
        let mut data = vec![0u8; n_px * bpp];
        for px in 0..n_px {
            let w = weight[px];
            for c in 0..bpp {
                let v = if w > 1e-6 {
                    accum[px * bpp + c] / w
                } else {
                    0.0
                };
                data[px * bpp + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }

        Ok(StitchedPanorama {
            data,
            width: canvas_w,
            height: canvas_h,
            channels: bpp,
            format: self.config.output_format,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_config(blend: u32) -> PanoramaConfig {
        PanoramaConfig {
            blend_width_px: blend,
            output_format: PanoFormat::Flat,
        }
    }

    fn solid_frame(w: u32, h: u32, ox: i64, oy: i64, val: u8) -> StitchFrame {
        StitchFrame {
            image: vec![val; (w * h) as usize],
            width: w,
            height: h,
            offset_x: ox,
            offset_y: oy,
        }
    }

    // ── Basic structural tests ────────────────────────────────────────────────

    #[test]
    fn test_no_frames_error() {
        let stitcher = PanoramaStitcher::new(flat_config(0));
        assert_eq!(stitcher.stitch(&[]).unwrap_err(), StitchError::NoFrames);
    }

    #[test]
    fn test_single_frame_passthrough() {
        let stitcher = PanoramaStitcher::new(flat_config(0));
        let frame = solid_frame(10, 8, 0, 0, 200);
        let pano = stitcher
            .stitch(&[frame])
            .expect("single frame should succeed");
        assert_eq!(pano.width, 10);
        assert_eq!(pano.height, 8);
        assert_eq!(pano.data.len(), 10 * 8);
        // All pixels should be 200
        assert!(pano.data.iter().all(|&p| p == 200), "pixel mismatch");
    }

    #[test]
    fn test_single_frame_negative_offset() {
        let stitcher = PanoramaStitcher::new(flat_config(0));
        let frame = solid_frame(5, 5, -3, -2, 128);
        let pano = stitcher
            .stitch(&[frame])
            .expect("negative offset should work");
        assert_eq!(pano.width, 5);
        assert_eq!(pano.height, 5);
    }

    #[test]
    fn test_two_frames_no_overlap() {
        let stitcher = PanoramaStitcher::new(flat_config(0));
        // Frame A: cols 0-9, Frame B: cols 10-19 (no overlap)
        let frame_a = solid_frame(10, 5, 0, 0, 100);
        let frame_b = solid_frame(10, 5, 10, 0, 200);
        let pano = stitcher
            .stitch(&[frame_a, frame_b])
            .expect("should succeed");
        assert_eq!(pano.width, 20);
        assert_eq!(pano.height, 5);
        // Left half should be 100, right half 200
        for y in 0..5usize {
            for x in 0..10usize {
                assert_eq!(pano.data[y * 20 + x], 100, "left half");
                assert_eq!(pano.data[y * 20 + x + 10], 200, "right half");
            }
        }
    }

    #[test]
    fn test_two_frames_with_overlap_blend() {
        let stitcher = PanoramaStitcher::new(flat_config(4));
        // Frame A: cols 0-9, Frame B: cols 6-15 → 4-px overlap
        let frame_a = solid_frame(10, 4, 0, 0, 100);
        let frame_b = solid_frame(10, 4, 6, 0, 200);
        let pano = stitcher
            .stitch(&[frame_a, frame_b])
            .expect("blend should succeed");
        assert_eq!(pano.width, 16);
        assert_eq!(pano.height, 4);
        // The overlap zone (cols 6-9 of pano) should be blended — not exactly 100 or 200
        let overlap_px = pano.data[0 * 16 + 7]; // middle of overlap
        assert!(
            overlap_px > 100 && overlap_px < 200,
            "expected blended value in overlap, got {overlap_px}"
        );
    }

    #[test]
    fn test_equirectangular_format_preserved() {
        let cfg = PanoramaConfig {
            blend_width_px: 0,
            output_format: PanoFormat::Equirectangular,
        };
        let stitcher = PanoramaStitcher::new(cfg);
        let frame = solid_frame(8, 4, 0, 0, 128);
        let pano = stitcher.stitch(&[frame]).expect("should succeed");
        assert_eq!(pano.format, PanoFormat::Equirectangular);
    }

    #[test]
    fn test_rgb_frames() {
        let stitcher = PanoramaStitcher::new(flat_config(0));
        let rgb_pixels = vec![255u8, 0, 0].repeat(8 * 8); // 8x8 red
        let frame = StitchFrame {
            image: rgb_pixels,
            width: 8,
            height: 8,
            offset_x: 0,
            offset_y: 0,
        };
        let pano = stitcher.stitch(&[frame]).expect("RGB should work");
        assert_eq!(pano.channels, 3);
        assert_eq!(pano.data.len(), 8 * 8 * 3);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let stitcher = PanoramaStitcher::new(flat_config(0));
        let frame_gray = solid_frame(4, 4, 0, 0, 128);
        let frame_rgb = StitchFrame {
            image: vec![255u8; 4 * 4 * 3],
            width: 4,
            height: 4,
            offset_x: 4,
            offset_y: 0,
        };
        assert_eq!(
            stitcher.stitch(&[frame_gray, frame_rgb]).unwrap_err(),
            StitchError::DimensionMismatch
        );
    }

    #[test]
    fn test_vertical_stack() {
        let stitcher = PanoramaStitcher::new(flat_config(0));
        let top = solid_frame(6, 4, 0, 0, 50);
        let bottom = solid_frame(6, 4, 0, 4, 150);
        let pano = stitcher.stitch(&[top, bottom]).expect("vertical stack");
        assert_eq!(pano.width, 6);
        assert_eq!(pano.height, 8);
        // Top half ≈ 50, bottom half ≈ 150
        for x in 0..6usize {
            assert_eq!(pano.data[0 * 6 + x], 50, "top row");
            assert_eq!(pano.data[7 * 6 + x], 150, "bottom row");
        }
    }

    #[test]
    fn test_blend_width_zero_is_hard_edge() {
        let stitcher = PanoramaStitcher::new(flat_config(0));
        let frame_a = solid_frame(8, 4, 0, 0, 80);
        let frame_b = solid_frame(8, 4, 4, 0, 160);
        let pano = stitcher.stitch(&[frame_a, frame_b]).expect("hard edge");
        // In the overlap (cols 4-7) the second frame wins since alpha=1 for both
        // and weights add to 2, so value = (80 + 160) / 2 = 120
        let overlap_px = pano.data[0 * 12 + 4];
        assert_eq!(overlap_px, 120, "expected averaged overlap pixel");
    }

    #[test]
    fn test_canvas_size_with_offset() {
        let stitcher = PanoramaStitcher::new(flat_config(0));
        let frame = StitchFrame {
            image: vec![255u8; 5 * 5],
            width: 5,
            height: 5,
            offset_x: 10,
            offset_y: 20,
        };
        // Single frame at (10,20) with size 5x5 → canvas anchored to (10,20) → 5x5
        let pano = stitcher.stitch(&[frame]).expect("offset frame");
        assert_eq!(pano.width, 5);
        assert_eq!(pano.height, 5);
    }
}
