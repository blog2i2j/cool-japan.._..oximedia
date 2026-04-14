//! Sub-pixel motion vector types and half-pixel refinement.
//!
//! Provides [`SubpixelMotionVector`] for representing fractional-pixel motion
//! estimates, along with [`refine_half_pixel`] which refines an integer motion
//! vector to half-pixel accuracy by evaluating bilinear-interpolated SAD at
//! all 8 half-pixel neighbours.

// ============================================================================
// Sub-pixel motion vector
// ============================================================================

/// Sub-pixel motion vector with fractional pixel displacement.
///
/// Displacement values are in pixels and may be fractional
/// (e.g. `dx = 0.5` means a half-pixel shift to the right).
#[derive(Debug, Clone)]
pub struct SubpixelMotionVector {
    /// Horizontal displacement in pixels (positive = right).
    pub dx: f32,
    /// Vertical displacement in pixels (positive = down).
    pub dy: f32,
    /// Sum of absolute differences at this position (lower = better match).
    pub sad: u32,
}

impl SubpixelMotionVector {
    /// Zero motion vector: no displacement, SAD = 0.
    pub fn zero() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            sad: 0,
        }
    }

    /// Euclidean magnitude of the displacement: `sqrt(dx² + dy²)`.
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Returns `true` if both components are effectively zero (`|dx|, |dy| < 0.01`).
    pub fn is_zero(&self) -> bool {
        self.dx.abs() < 0.01 && self.dy.abs() < 0.01
    }
}

// ============================================================================
// SAD helpers
// ============================================================================

/// Compute SAD for a block match between two frames with boundary clamping.
///
/// * `reference` / `current` – luma planes (row-major, one byte per pixel).
/// * `width` – frame width in pixels.
/// * `ref_x` / `ref_y` – top-left of the block in `reference`.
/// * `cur_x` / `cur_y` – top-left of the block in `current`.
/// * `block_size` – side length of the square block in pixels.
pub fn compute_block_sad(
    reference: &[u8],
    current: &[u8],
    width: usize,
    ref_x: usize,
    ref_y: usize,
    cur_x: usize,
    cur_y: usize,
    block_size: usize,
) -> u32 {
    let mut total = 0u32;
    let height = reference.len() / width.max(1);
    for row in 0..block_size {
        for col in 0..block_size {
            let ry = (ref_y + row).min(height.saturating_sub(1));
            let rx = (ref_x + col).min(width.saturating_sub(1));
            let cy = (cur_y + row).min(height.saturating_sub(1));
            let cx = (cur_x + col).min(width.saturating_sub(1));
            let r = reference.get(ry * width + rx).copied().unwrap_or(0);
            let c = current.get(cy * width + cx).copied().unwrap_or(0);
            total += (r as i32 - c as i32).unsigned_abs();
        }
    }
    total
}

/// Bilinear sample of a luma plane at a fractional position.
fn bilinear_sample(frame: &[u8], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let x0 = (x.floor() as usize).min(width.saturating_sub(1));
    let y0 = (y.floor() as usize).min(height.saturating_sub(1));
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let fx = x - x.floor();
    let fy = y - y.floor();
    let p00 = frame.get(y0 * width + x0).copied().unwrap_or(0) as f32;
    let p10 = frame.get(y0 * width + x1).copied().unwrap_or(0) as f32;
    let p01 = frame.get(y1 * width + x0).copied().unwrap_or(0) as f32;
    let p11 = frame.get(y1 * width + x1).copied().unwrap_or(0) as f32;
    let top = p00 * (1.0 - fx) + p10 * fx;
    let bot = p01 * (1.0 - fx) + p11 * fx;
    top * (1.0 - fy) + bot * fy
}

// ============================================================================
// Half-pixel refinement
// ============================================================================

/// Refine an integer motion vector to half-pixel accuracy.
///
/// Evaluates the integer position plus 8 half-pixel neighbours (±0.5 in each
/// axis and diagonally) using bilinear interpolation, returning the candidate
/// with the minimum SAD.
///
/// # Arguments
///
/// * `reference` / `current` – luma planes (row-major, same frame dimensions).
/// * `width` / `height` – frame dimensions in pixels.
/// * `block_x` / `block_y` – top-left corner of the block in `current`.
/// * `block_size` – side length of the square block.
/// * `initial_mv` – integer `(dx, dy)` displacement to refine.
#[allow(clippy::too_many_arguments)]
pub fn refine_half_pixel(
    reference: &[u8],
    current: &[u8],
    width: usize,
    height: usize,
    block_x: usize,
    block_y: usize,
    block_size: usize,
    initial_mv: (i32, i32),
) -> SubpixelMotionVector {
    // Integer position + 8 half-pixel neighbours
    const HALF_OFFSETS: [(f32, f32); 9] = [
        (0.0, 0.0),
        (0.5, 0.0),
        (-0.5, 0.0),
        (0.0, 0.5),
        (0.0, -0.5),
        (0.5, 0.5),
        (0.5, -0.5),
        (-0.5, 0.5),
        (-0.5, -0.5),
    ];

    let base_dx = initial_mv.0 as f32;
    let base_dy = initial_mv.1 as f32;
    let max_x = (width as f32) - 1.0;
    let max_y = (height as f32) - 1.0;

    let mut best_mv = SubpixelMotionVector {
        dx: base_dx,
        dy: base_dy,
        sad: u32::MAX,
    };

    for (fdx, fdy) in &HALF_OFFSETS {
        let cand_dx = base_dx + fdx;
        let cand_dy = base_dy + fdy;
        let mut sad_acc = 0u32;

        for row in 0..block_size {
            for col in 0..block_size {
                let ref_x = (block_x as f32 + col as f32 + cand_dx).clamp(0.0, max_x);
                let ref_y = (block_y as f32 + row as f32 + cand_dy).clamp(0.0, max_y);
                let ref_val = bilinear_sample(reference, width, height, ref_x, ref_y);

                let cur_x = (block_x + col).min(width.saturating_sub(1));
                let cur_y = (block_y + row).min(height.saturating_sub(1));
                let cur_val = current.get(cur_y * width + cur_x).copied().unwrap_or(0) as f32;

                sad_acc += (ref_val - cur_val).abs().round() as u32;
            }
        }

        if sad_acc < best_mv.sad {
            best_mv = SubpixelMotionVector {
                dx: cand_dx,
                dy: cand_dy,
                sad: sad_acc,
            };
        }
    }

    best_mv
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_frame(width: usize, height: usize, val: u8) -> Vec<u8> {
        vec![val; width * height]
    }

    // 1. SubpixelMotionVector::zero() has zero displacement and zero SAD
    #[test]
    fn test_subpixel_mv_zero() {
        let mv = SubpixelMotionVector::zero();
        assert_eq!(mv.dx, 0.0);
        assert_eq!(mv.dy, 0.0);
        assert_eq!(mv.sad, 0);
    }

    // 2. magnitude is correct (Pythagoras: 3-4-5 triangle)
    #[test]
    fn test_subpixel_mv_magnitude() {
        let mv = SubpixelMotionVector {
            dx: 3.0,
            dy: 4.0,
            sad: 0,
        };
        let mag = mv.magnitude();
        assert!((mag - 5.0).abs() < 1e-5, "expected 5.0, got {mag}");
    }

    // 3. is_zero returns true for zero vector
    #[test]
    fn test_subpixel_mv_is_zero_true() {
        assert!(SubpixelMotionVector::zero().is_zero());
    }

    // 4. is_zero returns false for non-zero displacement
    #[test]
    fn test_subpixel_mv_is_zero_false() {
        let mv = SubpixelMotionVector {
            dx: 1.0,
            dy: 0.0,
            sad: 0,
        };
        assert!(!mv.is_zero());
    }

    // 5. compute_block_sad on identical block positions returns 0
    #[test]
    fn test_compute_block_sad_identical() {
        let frame = (0u8..=255).cycle().take(16 * 16).collect::<Vec<_>>();
        let sad = compute_block_sad(&frame, &frame, 16, 2, 2, 2, 2, 4);
        assert_eq!(sad, 0, "identical block positions must have SAD=0");
    }

    // 6. compute_block_sad on different blocks returns > 0
    #[test]
    fn test_compute_block_sad_different() {
        let a = vec![0u8; 8 * 8];
        let b = vec![128u8; 8 * 8];
        let sad = compute_block_sad(&a, &b, 8, 0, 0, 0, 0, 4);
        assert!(sad > 0, "different blocks must have SAD>0, got {sad}");
    }

    // 7. refine_half_pixel on flat identical frames returns near-zero MV
    #[test]
    fn test_refine_half_pixel_flat_identical() {
        let frame = flat_frame(16, 16, 128);
        let mv = refine_half_pixel(&frame, &frame, 16, 16, 4, 4, 4, (0, 0));
        assert!(
            mv.magnitude() < 1.0,
            "flat identical frames should give near-zero MV, got ({}, {})",
            mv.dx,
            mv.dy
        );
    }

    // 8. magnitude formula: sqrt(dx²+dy²) for unit vector
    #[test]
    fn test_magnitude_unit_vector() {
        let mv = SubpixelMotionVector {
            dx: 0.0,
            dy: 1.0,
            sad: 10,
        };
        assert!((mv.magnitude() - 1.0).abs() < 1e-5);
    }
}
