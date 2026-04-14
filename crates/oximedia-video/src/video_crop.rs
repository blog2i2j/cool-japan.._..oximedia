//! Video cropping and region-of-interest extraction.
//!
//! Provides three cropping modes for luma (and by extension chroma) planes:
//!
//! 1. **Arbitrary rect crop** — extract any rectangular sub-region from a frame.
//! 2. **Safe-area crop** — crop to a percentage safe zone (e.g. 90% action-safe,
//!    80% title-safe), centred on the frame.
//! 3. **Content-aware centre detection** — analyse frame variance to find the
//!    most visually active region and automatically centre the crop window there.
//!
//! All functions operate on row-major `u8` luma planes.  Chroma planes should be
//! cropped with scaled rectangles according to their subsampling ratio.

// -----------------------------------------------------------------------
// Error type
// -----------------------------------------------------------------------

/// Errors that can occur during video cropping.
#[derive(Debug, thiserror::Error)]
pub enum VideoCropError {
    /// Source frame dimensions are invalid (zero width or height).
    #[error("invalid source dimensions: {width}x{height}")]
    InvalidSourceDimensions {
        /// Source frame width.
        width: u32,
        /// Source frame height.
        height: u32,
    },
    /// The requested crop rectangle extends outside the frame.
    #[error("crop rect ({x},{y})+{w}x{h} exceeds source {src_w}x{src_h}")]
    CropOutOfBounds {
        /// Crop rectangle X offset.
        x: u32,
        /// Crop rectangle Y offset.
        y: u32,
        /// Crop width.
        w: u32,
        /// Crop height.
        h: u32,
        /// Source frame width.
        src_w: u32,
        /// Source frame height.
        src_h: u32,
    },
    /// Crop dimensions are zero.
    #[error("crop dimensions must be non-zero, got {w}x{h}")]
    ZeroCropDimensions {
        /// Requested crop width.
        w: u32,
        /// Requested crop height.
        h: u32,
    },
    /// Source buffer has unexpected size.
    #[error("buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch {
        /// Expected buffer length.
        expected: usize,
        /// Actual buffer length.
        actual: usize,
    },
    /// The safe-area percentage is outside the valid range (0, 100].
    #[error("safe-area percentage {pct} is out of range (must be in (0, 100])")]
    InvalidSafeAreaPct {
        /// The invalid percentage value.
        pct: f64,
    },
    /// The requested crop is larger than the source after safe-area reduction.
    #[error("no valid crop region could be computed")]
    NoCropRegion,
}

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// An axis-aligned rectangle used to specify a crop region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropRect {
    /// Horizontal offset from the left edge of the source frame.
    pub x: u32,
    /// Vertical offset from the top edge of the source frame.
    pub y: u32,
    /// Width of the cropped region.
    pub w: u32,
    /// Height of the cropped region.
    pub h: u32,
}

impl CropRect {
    /// Create a new `CropRect`, validating that width and height are non-zero.
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Result<Self, VideoCropError> {
        if w == 0 || h == 0 {
            return Err(VideoCropError::ZeroCropDimensions { w, h });
        }
        Ok(Self { x, y, w, h })
    }

    /// Right-exclusive horizontal bound.
    #[inline]
    pub fn right(&self) -> u32 {
        self.x + self.w
    }

    /// Bottom-exclusive vertical bound.
    #[inline]
    pub fn bottom(&self) -> u32 {
        self.y + self.h
    }
}

/// Result of a content-aware crop analysis.
#[derive(Debug, Clone)]
pub struct ContentAwareCropResult {
    /// The recommended crop rectangle.
    pub rect: CropRect,
    /// Detected centre of visual activity (x, y) within the source frame.
    pub activity_centre: (u32, u32),
    /// Per-tile variance map used for centre detection.  Row-major, one
    /// `f64` value per tile.
    pub tile_variance: Vec<f64>,
    /// Number of tiles along the horizontal axis.
    pub tile_cols: u32,
    /// Number of tiles along the vertical axis.
    pub tile_rows: u32,
}

// -----------------------------------------------------------------------
// Arbitrary rect crop
// -----------------------------------------------------------------------

/// Crop a luma plane to an arbitrary [`CropRect`].
///
/// # Arguments
///
/// * `src` – source luma buffer, `src_w * src_h` bytes, row-major.
/// * `src_w` / `src_h` – source frame dimensions.
/// * `rect` – crop region (must lie within the source frame).
///
/// # Returns
///
/// A new `Vec<u8>` of `rect.w * rect.h` bytes containing the cropped luma.
///
/// # Errors
///
/// Returns [`VideoCropError`] if dimensions or bounds are invalid.
pub fn crop_rect(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    rect: CropRect,
) -> Result<Vec<u8>, VideoCropError> {
    validate_source(src, src_w, src_h)?;
    validate_bounds(src_w, src_h, rect)?;

    let sw = src_w as usize;
    let rw = rect.w as usize;
    let rh = rect.h as usize;
    let mut out = vec![0u8; rw * rh];

    for row in 0..rh {
        let src_row = (rect.y as usize) + row;
        let src_start = src_row * sw + rect.x as usize;
        let dst_start = row * rw;
        out[dst_start..dst_start + rw].copy_from_slice(&src[src_start..src_start + rw]);
    }
    Ok(out)
}

// -----------------------------------------------------------------------
// Safe-area crop
// -----------------------------------------------------------------------

/// Compute the centred safe-area [`CropRect`] for a given frame size and
/// percentage.
///
/// `safe_pct` is a value in `(0.0, 100.0]`.  For example `90.0` gives the
/// standard 90 % action-safe area.
///
/// # Errors
///
/// Returns [`VideoCropError::InvalidSafeAreaPct`] if `safe_pct` is ≤ 0 or > 100.
pub fn safe_area_rect(src_w: u32, src_h: u32, safe_pct: f64) -> Result<CropRect, VideoCropError> {
    if src_w == 0 || src_h == 0 {
        return Err(VideoCropError::InvalidSourceDimensions {
            width: src_w,
            height: src_h,
        });
    }
    if safe_pct <= 0.0 || safe_pct > 100.0 {
        return Err(VideoCropError::InvalidSafeAreaPct { pct: safe_pct });
    }
    let scale = safe_pct / 100.0;
    let cw = ((src_w as f64 * scale) as u32).max(1);
    let ch = ((src_h as f64 * scale) as u32).max(1);
    let cx = (src_w - cw) / 2;
    let cy = (src_h - ch) / 2;
    CropRect::new(cx, cy, cw, ch).map_err(|_| VideoCropError::NoCropRegion)
}

/// Crop a luma plane to the standard safe-area percentage, centred.
///
/// # Errors
///
/// Returns [`VideoCropError`] on dimension or bounds violations.
pub fn crop_safe_area(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    safe_pct: f64,
) -> Result<Vec<u8>, VideoCropError> {
    let rect = safe_area_rect(src_w, src_h, safe_pct)?;
    crop_rect(src, src_w, src_h, rect)
}

// -----------------------------------------------------------------------
// Content-aware crop
// -----------------------------------------------------------------------

/// Analyse a luma frame to find the centre of visual activity and compute
/// an auto-crop rectangle of the requested output size.
///
/// The algorithm divides the frame into `tile_size × tile_size` tiles,
/// computes the variance of each tile, then computes a variance-weighted
/// centroid to find the activity centre.  The output crop is centred on that
/// point and clamped to frame boundaries.
///
/// # Arguments
///
/// * `src` – source luma buffer.
/// * `src_w` / `src_h` – source dimensions.
/// * `out_w` / `out_h` – desired output crop size (must be ≤ source).
/// * `tile_size` – size of analysis tiles in pixels (≥ 1).
///
/// # Errors
///
/// Returns [`VideoCropError`] if dimensions are invalid or the requested crop
/// does not fit.
pub fn crop_content_aware(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    out_w: u32,
    out_h: u32,
    tile_size: u32,
) -> Result<ContentAwareCropResult, VideoCropError> {
    validate_source(src, src_w, src_h)?;
    if out_w == 0 || out_h == 0 {
        return Err(VideoCropError::ZeroCropDimensions { w: out_w, h: out_h });
    }
    if out_w > src_w || out_h > src_h {
        return Err(VideoCropError::CropOutOfBounds {
            x: 0,
            y: 0,
            w: out_w,
            h: out_h,
            src_w,
            src_h,
        });
    }
    let ts = tile_size.max(1) as usize;
    let sw = src_w as usize;
    let sh = src_h as usize;

    let tile_cols = (sw + ts - 1) / ts;
    let tile_rows = (sh + ts - 1) / ts;
    let ntiles = tile_cols * tile_rows;
    let mut variances = vec![0.0f64; ntiles];

    for tr in 0..tile_rows {
        for tc in 0..tile_cols {
            let y0 = tr * ts;
            let x0 = tc * ts;
            let y1 = (y0 + ts).min(sh);
            let x1 = (x0 + ts).min(sw);

            let mut sum = 0u64;
            let mut sum_sq = 0u64;
            let mut count = 0u64;
            for row in y0..y1 {
                for col in x0..x1 {
                    let v = src[row * sw + col] as u64;
                    sum += v;
                    sum_sq += v * v;
                    count += 1;
                }
            }
            if count > 0 {
                let mean = sum as f64 / count as f64;
                let var = sum_sq as f64 / count as f64 - mean * mean;
                variances[tr * tile_cols + tc] = var.max(0.0);
            }
        }
    }

    // Variance-weighted centroid (in pixel coordinates).
    let (cx, cy) = weighted_centroid(&variances, tile_cols, tile_rows, ts, sw, sh);

    // Centre the output crop on (cx, cy), clamped to frame.
    let x = cx.saturating_sub(out_w / 2).min(src_w - out_w);
    let y = cy.saturating_sub(out_h / 2).min(src_h - out_h);

    let rect = CropRect::new(x, y, out_w, out_h).map_err(|_| VideoCropError::NoCropRegion)?;

    Ok(ContentAwareCropResult {
        rect,
        activity_centre: (cx, cy),
        tile_variance: variances,
        tile_cols: tile_cols as u32,
        tile_rows: tile_rows as u32,
    })
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// Validate source buffer size against declared dimensions.
fn validate_source(src: &[u8], w: u32, h: u32) -> Result<(), VideoCropError> {
    if w == 0 || h == 0 {
        return Err(VideoCropError::InvalidSourceDimensions {
            width: w,
            height: h,
        });
    }
    let expected = w as usize * h as usize;
    if src.len() != expected {
        return Err(VideoCropError::BufferSizeMismatch {
            expected,
            actual: src.len(),
        });
    }
    Ok(())
}

/// Check that a crop rect fits within the source frame.
fn validate_bounds(src_w: u32, src_h: u32, r: CropRect) -> Result<(), VideoCropError> {
    if r.w == 0 || r.h == 0 {
        return Err(VideoCropError::ZeroCropDimensions { w: r.w, h: r.h });
    }
    if r.right() > src_w || r.bottom() > src_h {
        return Err(VideoCropError::CropOutOfBounds {
            x: r.x,
            y: r.y,
            w: r.w,
            h: r.h,
            src_w,
            src_h,
        });
    }
    Ok(())
}

/// Compute a variance-weighted centroid pixel position.
fn weighted_centroid(
    variances: &[f64],
    tile_cols: usize,
    tile_rows: usize,
    ts: usize,
    src_w: usize,
    src_h: usize,
) -> (u32, u32) {
    let mut wx = 0.0f64;
    let mut wy = 0.0f64;
    let mut total_w = 0.0f64;

    for tr in 0..tile_rows {
        for tc in 0..tile_cols {
            let v = variances[tr * tile_cols + tc];
            if v <= 0.0 {
                continue;
            }
            // Tile centre in pixel coordinates.
            let px = (tc * ts + ts / 2).min(src_w - 1) as f64;
            let py = (tr * ts + ts / 2).min(src_h - 1) as f64;
            wx += px * v;
            wy += py * v;
            total_w += v;
        }
    }

    if total_w < 1e-9 {
        // No variance — fall back to frame centre.
        return ((src_w / 2) as u32, (src_h / 2) as u32);
    }

    let cx = (wx / total_w).round() as u32;
    let cy = (wy / total_w).round() as u32;
    (cx.min(src_w as u32 - 1), cy.min(src_h as u32 - 1))
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_frame(w: u32, h: u32) -> Vec<u8> {
        (0..w as usize * h as usize)
            .map(|i| (i % 256) as u8)
            .collect()
    }

    fn solid_frame(w: u32, h: u32, v: u8) -> Vec<u8> {
        vec![v; (w * h) as usize]
    }

    // 1. Basic rect crop dimensions
    #[test]
    fn test_crop_rect_dimensions() {
        let src = ramp_frame(8, 8);
        let rect = CropRect::new(1, 1, 4, 4).unwrap();
        let out = crop_rect(&src, 8, 8, rect).unwrap();
        assert_eq!(out.len(), 16);
    }

    // 2. Rect crop pixel values match source
    #[test]
    fn test_crop_rect_values() {
        let mut src = vec![0u8; 9]; // 3x3
        for (i, v) in src.iter_mut().enumerate() {
            *v = i as u8;
        }
        // Crop top-left 2x2
        let rect = CropRect::new(0, 0, 2, 2).unwrap();
        let out = crop_rect(&src, 3, 3, rect).unwrap();
        assert_eq!(out, vec![0, 1, 3, 4]);
    }

    // 3. Out-of-bounds crop is rejected
    #[test]
    fn test_crop_out_of_bounds() {
        let src = solid_frame(4, 4, 0);
        let rect = CropRect::new(2, 2, 4, 4).unwrap(); // extends past edge
        assert!(matches!(
            crop_rect(&src, 4, 4, rect),
            Err(VideoCropError::CropOutOfBounds { .. })
        ));
    }

    // 4. Safe-area 90% rect is centred and correct size
    #[test]
    fn test_safe_area_90_percent() {
        let rect = safe_area_rect(100, 100, 90.0).unwrap();
        assert_eq!(rect.w, 90);
        assert_eq!(rect.h, 90);
        assert_eq!(rect.x, 5);
        assert_eq!(rect.y, 5);
    }

    // 5. Safe-area 100% == full frame
    #[test]
    fn test_safe_area_100_percent() {
        let rect = safe_area_rect(8, 6, 100.0).unwrap();
        assert_eq!(rect.x, 0);
        assert_eq!(rect.y, 0);
        assert_eq!(rect.w, 8);
        assert_eq!(rect.h, 6);
    }

    // 6. Invalid safe-area percentage rejected
    #[test]
    fn test_safe_area_invalid_pct() {
        assert!(matches!(
            safe_area_rect(100, 100, 0.0),
            Err(VideoCropError::InvalidSafeAreaPct { .. })
        ));
        assert!(matches!(
            safe_area_rect(100, 100, 110.0),
            Err(VideoCropError::InvalidSafeAreaPct { .. })
        ));
    }

    // 7. Content-aware crop on a uniform frame falls back to centre
    #[test]
    fn test_content_aware_uniform() {
        let src = solid_frame(16, 16, 128);
        let result = crop_content_aware(&src, 16, 16, 8, 8, 4).unwrap();
        // For a uniform frame the centroid falls back to the frame centre.
        let cx = result.activity_centre.0;
        let cy = result.activity_centre.1;
        assert_eq!(cx, 8);
        assert_eq!(cy, 8);
        assert_eq!(result.rect.w, 8);
        assert_eq!(result.rect.h, 8);
    }

    // 8. Content-aware crop detects activity in bright region
    #[test]
    fn test_content_aware_bright_region() {
        // 16x16 frame, mostly dark except bottom-right 4x4 quadrant (bright).
        let mut src = vec![0u8; 16 * 16];
        for row in 12..16usize {
            for col in 12..16usize {
                src[row * 16 + col] = 200 + ((row + col) % 55) as u8;
            }
        }
        let result = crop_content_aware(&src, 16, 16, 8, 8, 4).unwrap();
        // Activity centre should be in the right half.
        assert!(
            result.activity_centre.0 >= 8,
            "expected activity in right half, got x={}",
            result.activity_centre.0
        );
    }

    // 9. Content-aware crop with out_w > src_w is rejected
    #[test]
    fn test_content_aware_too_large() {
        let src = solid_frame(8, 8, 0);
        assert!(matches!(
            crop_content_aware(&src, 8, 8, 16, 8, 4),
            Err(VideoCropError::CropOutOfBounds { .. })
        ));
    }

    // 10. Buffer size mismatch is detected
    #[test]
    fn test_buffer_mismatch() {
        let rect = CropRect::new(0, 0, 2, 2).unwrap();
        let result = crop_rect(&[0u8; 5], 4, 4, rect);
        assert!(matches!(
            result,
            Err(VideoCropError::BufferSizeMismatch { .. })
        ));
    }

    // 11. Zero crop dimensions rejected via CropRect::new
    #[test]
    fn test_zero_crop_dims() {
        assert!(matches!(
            CropRect::new(0, 0, 0, 4),
            Err(VideoCropError::ZeroCropDimensions { .. })
        ));
    }

    // 12. safe_area_crop on a real frame returns correct buffer size
    #[test]
    fn test_safe_area_crop_buffer_size() {
        let src = ramp_frame(20, 20);
        let out = crop_safe_area(&src, 20, 20, 80.0).unwrap();
        // 80% of 20 = 16
        assert_eq!(out.len(), 16 * 16);
    }

    // 13. Tile variance map has correct number of entries
    #[test]
    fn test_tile_variance_count() {
        let src = ramp_frame(8, 8);
        let result = crop_content_aware(&src, 8, 8, 4, 4, 2).unwrap();
        // 8/2 = 4 tiles per axis → 16 tiles total
        assert_eq!(result.tile_variance.len(), 16);
        assert_eq!(result.tile_cols, 4);
        assert_eq!(result.tile_rows, 4);
    }
}
