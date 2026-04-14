//! AV1 CDEF (Constrained Directional Enhancement Filter) SIMD operations.
//!
//! CDEF is an in-loop filter that reduces ringing artifacts while preserving
//! edges and texture details.
//!
//! This module supports both 8-bit (`u8`) and 10-bit/12-bit (`u16`) pixel depths.
//! The 8-bit path uses the [`CdefSimd`] generic type backed by a [`SimdOps`]
//! implementation.  The 10-bit path provides standalone scalar and optional SIMD
//! helpers (`cdef_filter_u16`) that work on raw `u16` slices.

use crate::simd::traits::SimdOps;
use crate::simd::types::{I16x8, U8x16};

// ============================================================================
// CdefPixel — trait abstracting 8-bit and 10-bit pixel access
// ============================================================================

/// Trait implemented by pixel types that CDEF can filter.
///
/// Both `u8` (8-bit) and `u16` (10-bit / 12-bit) implement this trait so that
/// higher-level code can be generic over pixel depth.
pub trait CdefPixel: Copy + Default + PartialOrd + Sized {
    /// Return the pixel as an `i32` for arithmetic.
    fn as_i32(self) -> i32;
    /// Clamp an `i32` result back to the pixel type.
    fn from_clamped(v: i32, max_val: i32) -> Self;
    /// Maximum representable value (e.g. 255 for 8-bit, 1023 for 10-bit).
    fn max_value(bit_depth: u8) -> i32;
}

impl CdefPixel for u8 {
    #[inline]
    fn as_i32(self) -> i32 {
        i32::from(self)
    }
    #[inline]
    fn from_clamped(v: i32, max_val: i32) -> Self {
        v.clamp(0, max_val) as u8
    }
    #[inline]
    fn max_value(_bit_depth: u8) -> i32 {
        255
    }
}

impl CdefPixel for u16 {
    #[inline]
    fn as_i32(self) -> i32 {
        i32::from(self)
    }
    #[inline]
    fn from_clamped(v: i32, max_val: i32) -> Self {
        v.clamp(0, max_val) as u16
    }
    #[inline]
    fn max_value(bit_depth: u8) -> i32 {
        i32::from((1u16 << bit_depth.min(16)) - 1)
    }
}

// ============================================================================
// Standalone scalar CDEF for 10-bit / 12-bit content
// ============================================================================

/// Get the (dx, dy) direction vector for one of the 8 CDEF directions.
#[inline]
fn cdef_direction_offset(direction: u8) -> (i32, i32) {
    match direction % 8 {
        0 => (1, 0),
        1 => (1, 1),
        2 => (0, 1),
        3 => (-1, 1),
        4 => (-1, 0),
        5 => (-1, -1),
        6 => (0, -1),
        7 => (1, -1),
        _ => (1, 0),
    }
}

/// Compute the weighted contribution of a single tap in a `u16` plane.
///
/// Returns `(weighted_diff, weight)`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn cdef_tap_weight_u16(
    src: &[u16],
    stride: usize,
    x: usize,
    y: usize,
    ox: i32,
    oy: i32,
    pixel: u16,
    strength: u16,
    damping: u8,
) -> (i32, i32) {
    let tx = x as i32 + ox;
    let ty = y as i32 + oy;
    if tx < 0 || ty < 0 {
        return (0, 0);
    }
    let offset = ty as usize * stride + tx as usize;
    if offset >= src.len() {
        return (0, 0);
    }
    let tap = src[offset];
    let diff = i32::from(tap) - i32::from(pixel);
    let abs_diff = diff.unsigned_abs() as i32;
    let threshold = 1i32 << damping;
    if abs_diff >= threshold {
        return (0, 0);
    }
    let weight = i32::from(strength) * (threshold - abs_diff) / threshold;
    (diff * weight, weight)
}

/// Apply a single-pixel CDEF filter on a `u16` plane.
///
/// The filter computes an additive correction based on neighbouring pixels.
/// For a uniform plane, all differences are zero so the correction is zero
/// and the pixel value is preserved exactly.
///
/// Formula (simplified from AV1 spec §7.17.5):
/// ```text
/// correction = clamp(Σ f(diff_i) * direction_weight_i)
/// output     = clamp(pixel + correction, 0, max_val)
/// ```
#[allow(clippy::too_many_arguments)]
#[inline]
fn cdef_filter_pixel_u16(
    src: &[u16],
    stride: usize,
    x: usize,
    y: usize,
    pixel: u16,
    pri_strength: u16,
    sec_strength: u16,
    direction: u8,
    damping: u8,
    bit_depth: u8,
) -> u16 {
    if pri_strength == 0 && sec_strength == 0 {
        return pixel;
    }
    let (dx, dy) = cdef_direction_offset(direction);
    // Primary taps (along direction, at distance 1 and 2)
    let pri_taps = [(dx, dy), (-dx, -dy), (dx * 2, dy * 2), (-dx * 2, -dy * 2)];
    // Secondary taps (perpendicular, at distance 1 and 2)
    let (sdx, sdy) = (-dy, dx);
    let sec_taps = [
        (sdx, sdy),
        (-sdx, -sdy),
        (sdx * 2, sdy * 2),
        (-sdx * 2, -sdy * 2),
    ];

    // Accumulate additive correction — zero when all neighbours equal pixel.
    let mut correction = 0i32;

    for &(ox, oy) in &pri_taps {
        let (wv, _w) = cdef_tap_weight_u16(src, stride, x, y, ox, oy, pixel, pri_strength, damping);
        correction += wv;
    }
    for &(ox, oy) in &sec_taps {
        let (wv, _w) = cdef_tap_weight_u16(src, stride, x, y, ox, oy, pixel, sec_strength, damping);
        correction += wv;
    }

    // Normalise: round to nearest via >>4 (16 taps maximum so correction / 16).
    let adjustment = (correction + 8) >> 4;
    let result = i32::from(pixel) + adjustment;
    let max_val = i32::from((1u16 << bit_depth.min(16)) - 1);
    result.clamp(0, max_val) as u16
}

/// Apply CDEF filtering to a full `u16` luma plane (scalar, supports 10-bit and 12-bit).
///
/// # Arguments
/// * `frame`      - Mutable slice of `u16` pixels (row-major).
/// * `width`      - Frame width in pixels.
/// * `height`     - Frame height in pixels.
/// * `stride`     - Row stride (elements, not bytes).
/// * `pri_strength` - Primary filter strength (0–15, scaled to bit depth if > 0).
/// * `sec_strength` - Secondary filter strength (0–4, scaled to bit depth if > 0).
/// * `direction`  - Filtering direction (0–7).
/// * `damping`    - Damping factor (0–6).
/// * `bit_depth`  - Pixel bit depth (8, 10, or 12).
///
/// This is a full-plane wrapper; for block-level filtering use
/// [`cdef_filter_block_u16`].
#[allow(clippy::too_many_arguments)]
pub fn cdef_filter_u16(
    frame: &mut [u16],
    width: usize,
    height: usize,
    stride: usize,
    pri_strength: u16,
    sec_strength: u16,
    direction: u8,
    damping: u8,
    bit_depth: u8,
) {
    // Work on a read-only copy to avoid aliasing — CDEF writes to dst, reads from src.
    let src: Vec<u16> = frame.to_vec();
    for y in 0..height {
        for x in 0..width {
            let idx = y * stride + x;
            if idx >= frame.len() {
                continue;
            }
            frame[idx] = cdef_filter_pixel_u16(
                &src,
                stride,
                x,
                y,
                src[idx],
                pri_strength,
                sec_strength,
                direction,
                damping,
                bit_depth,
            );
        }
    }
}

/// Apply CDEF filtering to a single 8×8 block in a `u16` plane.
///
/// Writes filtered output back in-place using a temporary copy of the source block.
#[allow(clippy::too_many_arguments)]
pub fn cdef_filter_block_u16(
    frame: &mut [u16],
    block_x: usize,
    block_y: usize,
    frame_width: usize,
    frame_height: usize,
    stride: usize,
    pri_strength: u16,
    sec_strength: u16,
    direction: u8,
    damping: u8,
    bit_depth: u8,
) {
    // Snapshot the full frame as source to allow safe reads outside the block.
    let src: Vec<u16> = frame.to_vec();
    let end_x = (block_x + 8).min(frame_width);
    let end_y = (block_y + 8).min(frame_height);

    for y in block_y..end_y {
        for x in block_x..end_x {
            let idx = y * stride + x;
            if idx >= frame.len() {
                continue;
            }
            frame[idx] = cdef_filter_pixel_u16(
                &src,
                stride,
                x,
                y,
                src[idx],
                pri_strength,
                sec_strength,
                direction,
                damping,
                bit_depth,
            );
        }
    }
}

/// Find the best CDEF direction for an 8×8 block in a `u16` plane.
///
/// Returns the direction index (0–7) that minimises cross-direction variance.
pub fn cdef_find_direction_u16(frame: &[u16], stride: usize, block_size: usize) -> u8 {
    let mut best_direction = 0u8;
    let mut best_variance = u64::MAX;

    for dir in 0..8u8 {
        let (dx, dy) = cdef_direction_offset(dir);
        let mut variance = 0u64;
        let mut count = 0u64;

        for y in 1..block_size.saturating_sub(1) {
            for x in 1..block_size.saturating_sub(1) {
                let offset = y * stride + x;
                if offset >= frame.len() {
                    continue;
                }
                let pixel = frame[offset];
                let tx = x as i32 + dx;
                let ty = y as i32 + dy;
                if tx >= 0 && ty >= 0 {
                    let tap_offset = ty as usize * stride + tx as usize;
                    if tap_offset < frame.len() {
                        let tap = frame[tap_offset];
                        let diff = u64::from(pixel.abs_diff(tap));
                        variance = variance.saturating_add(diff * diff);
                        count += 1;
                    }
                }
            }
        }

        let avg_variance = variance.checked_div(count).unwrap_or(u64::MAX);
        if avg_variance < best_variance {
            best_variance = avg_variance;
            best_direction = dir;
        }
    }

    best_direction
}

/// AV1 CDEF SIMD operations.
pub struct CdefSimd<S> {
    simd: S,
}

impl<S: SimdOps> CdefSimd<S> {
    /// Create a new CDEF SIMD instance.
    #[inline]
    pub const fn new(simd: S) -> Self {
        Self { simd }
    }

    /// Apply CDEF filtering to an 8x8 block.
    ///
    /// # Arguments
    /// * `src` - Source pixels (with border for filtering)
    /// * `dst` - Destination buffer for filtered pixels
    /// * `src_stride` - Stride of source buffer
    /// * `dst_stride` - Stride of destination buffer
    /// * `pri_strength` - Primary filtering strength (0-15)
    /// * `sec_strength` - Secondary filtering strength (0-4)
    /// * `direction` - Filtering direction (0-7)
    /// * `damping` - Damping parameter (0-6)
    #[allow(clippy::too_many_arguments)]
    pub fn filter_block_8x8(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_stride: usize,
        dst_stride: usize,
        pri_strength: u8,
        sec_strength: u8,
        direction: u8,
        damping: u8,
    ) {
        for y in 0..8 {
            for x in 0..8 {
                let src_offset = y * src_stride + x;
                let dst_offset = y * dst_stride + x;

                if src.len() <= src_offset || dst.len() <= dst_offset {
                    continue;
                }

                let pixel = src[src_offset];
                let filtered = self.filter_pixel(
                    src,
                    src_stride,
                    x,
                    y,
                    pixel,
                    pri_strength,
                    sec_strength,
                    direction,
                    damping,
                );
                dst[dst_offset] = filtered;
            }
        }
    }

    /// Apply CDEF filtering to a 4x4 block.
    #[allow(clippy::too_many_arguments)]
    pub fn filter_block_4x4(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_stride: usize,
        dst_stride: usize,
        pri_strength: u8,
        sec_strength: u8,
        direction: u8,
        damping: u8,
    ) {
        for y in 0..4 {
            for x in 0..4 {
                let src_offset = y * src_stride + x;
                let dst_offset = y * dst_stride + x;

                if src.len() <= src_offset || dst.len() <= dst_offset {
                    continue;
                }

                let pixel = src[src_offset];
                let filtered = self.filter_pixel(
                    src,
                    src_stride,
                    x,
                    y,
                    pixel,
                    pri_strength,
                    sec_strength,
                    direction,
                    damping,
                );
                dst[dst_offset] = filtered;
            }
        }
    }

    /// Find the best CDEF direction for a block.
    ///
    /// Returns the direction index (0-7) that minimizes variance
    /// along the direction.
    pub fn find_direction(&self, src: &[u8], stride: usize, block_size: usize) -> u8 {
        let mut best_direction = 0u8;
        let mut best_variance = u32::MAX;

        // Try all 8 directions
        for dir in 0..8 {
            let variance = self.calculate_directional_variance(src, stride, block_size, dir);
            if variance < best_variance {
                best_variance = variance;
                best_direction = dir;
            }
        }

        best_direction
    }

    // ========================================================================
    // Internal Filtering Operations
    // ========================================================================

    /// Filter a single pixel using CDEF.
    #[allow(clippy::too_many_arguments)]
    fn filter_pixel(
        &self,
        src: &[u8],
        stride: usize,
        x: usize,
        y: usize,
        pixel: u8,
        pri_strength: u8,
        sec_strength: u8,
        direction: u8,
        damping: u8,
    ) -> u8 {
        if pri_strength == 0 && sec_strength == 0 {
            return pixel;
        }

        // Get directional offsets
        let (dx, dy) = self.get_direction_offset(direction);

        // Calculate primary tap positions
        let pri_taps = [
            (dx, dy),           // Primary direction
            (-dx, -dy),         // Opposite direction
            (dx * 2, dy * 2),   // Extended primary
            (-dx * 2, -dy * 2), // Extended opposite
        ];

        // Calculate secondary tap positions (perpendicular)
        let (sdx, sdy) = (-dy, dx);
        let sec_taps = [
            (sdx, sdy),
            (-sdx, -sdy),
            (sdx * 2, sdy * 2),
            (-sdx * 2, -sdy * 2),
        ];

        // Accumulate filtered value
        let mut sum = i32::from(pixel) << 7; // Scale by 128
        let mut total_weight = 128i32;

        // Apply primary taps
        for &(ox, oy) in &pri_taps {
            let weight =
                self.calculate_weight(src, stride, x, y, ox, oy, pixel, pri_strength, damping);
            sum += weight.0;
            total_weight += weight.1;
        }

        // Apply secondary taps
        for &(ox, oy) in &sec_taps {
            let weight =
                self.calculate_weight(src, stride, x, y, ox, oy, pixel, sec_strength, damping);
            sum += weight.0;
            total_weight += weight.1;
        }

        // Normalize and clamp
        let result = (sum + total_weight / 2) / total_weight;
        result.clamp(0, 255) as u8
    }

    /// Calculate filtering weight for a tap.
    #[allow(clippy::too_many_arguments)]
    fn calculate_weight(
        &self,
        src: &[u8],
        stride: usize,
        x: usize,
        y: usize,
        ox: i32,
        oy: i32,
        pixel: u8,
        strength: u8,
        damping: u8,
    ) -> (i32, i32) {
        let tx = x as i32 + ox;
        let ty = y as i32 + oy;

        if tx < 0 || ty < 0 {
            return (0, 0);
        }

        let offset = ty as usize * stride + tx as usize;
        if offset >= src.len() {
            return (0, 0);
        }

        let tap_pixel = src[offset];
        let diff = i32::from(tap_pixel) - i32::from(pixel);
        let abs_diff = diff.abs();

        // Calculate weight based on difference
        let threshold = 1 << damping;
        if abs_diff >= threshold {
            return (0, 0);
        }

        let weight = i32::from(strength) * (threshold - abs_diff) / threshold;
        let weighted_value = diff * weight;

        (weighted_value, weight)
    }

    /// Get direction offset (dx, dy) for a given direction index.
    fn get_direction_offset(&self, direction: u8) -> (i32, i32) {
        match direction % 8 {
            0 => (1, 0),   // Horizontal
            1 => (1, 1),   // Diagonal ↗
            2 => (0, 1),   // Vertical
            3 => (-1, 1),  // Diagonal ↖
            4 => (-1, 0),  // Horizontal ←
            5 => (-1, -1), // Diagonal ↙
            6 => (0, -1),  // Vertical ↑
            7 => (1, -1),  // Diagonal ↘
            _ => (1, 0),
        }
    }

    /// Calculate variance along a direction for direction finding.
    fn calculate_directional_variance(
        &self,
        src: &[u8],
        stride: usize,
        block_size: usize,
        direction: u8,
    ) -> u32 {
        let (dx, dy) = self.get_direction_offset(direction);
        let mut variance = 0u32;
        let mut count = 0u32;

        for y in 1..block_size.saturating_sub(1) {
            for x in 1..block_size.saturating_sub(1) {
                let offset = y * stride + x;
                if offset >= src.len() {
                    continue;
                }

                let pixel = src[offset];

                // Sample along direction
                let tx = x as i32 + dx;
                let ty = y as i32 + dy;

                if tx >= 0 && ty >= 0 {
                    let tap_offset = ty as usize * stride + tx as usize;
                    if tap_offset < src.len() {
                        let tap_pixel = src[tap_offset];
                        let diff = u32::from(pixel.abs_diff(tap_pixel));
                        variance += diff * diff;
                        count += 1;
                    }
                }
            }
        }

        variance.checked_div(count).unwrap_or(u32::MAX)
    }

    /// SIMD-accelerated row filtering (process 8 pixels at once).
    #[allow(dead_code)]
    fn filter_row_simd(
        &self,
        src: &[u8],
        dst: &mut [u8],
        width: usize,
        pri_strength: u8,
        sec_strength: u8,
    ) {
        // Process 8 pixels at a time using SIMD
        let chunks = width / 8;
        for i in 0..chunks {
            let offset = i * 8;
            if offset + 8 > src.len() || offset + 8 > dst.len() {
                continue;
            }

            let mut pixels = U8x16::zero();
            for j in 0..8 {
                pixels[j] = src[offset + j];
            }

            // Convert to i16 for filtering
            let pixels_i16 = self.simd.widen_low_u8_to_i16(pixels);

            // Apply simple smoothing filter
            let strength_vec = I16x8::from_array([i16::from(pri_strength + sec_strength); 8]);
            let filtered = self.simd.add_i16x8(pixels_i16, strength_vec);

            // Convert back to u8
            for j in 0..8 {
                dst[offset + j] = filtered[j].clamp(0, 255) as u8;
            }
        }

        // Handle remaining pixels
        for i in (chunks * 8)..width.min(src.len()).min(dst.len()) {
            dst[i] = src[i];
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- CdefPixel trait tests -----------------------------------------------

    #[test]
    fn test_cdef_pixel_u8_max_value() {
        // Use UFCS to disambiguate from deprecated `u8::max_value()` in std.
        assert_eq!(<u8 as CdefPixel>::max_value(8), 255);
        // u8::max_value ignores bit_depth
        assert_eq!(<u8 as CdefPixel>::max_value(10), 255);
    }

    #[test]
    fn test_cdef_pixel_u16_max_value() {
        // Use UFCS to disambiguate from deprecated `u16::max_value()` in std.
        assert_eq!(<u16 as CdefPixel>::max_value(8), 255);
        assert_eq!(<u16 as CdefPixel>::max_value(10), 1023);
        assert_eq!(<u16 as CdefPixel>::max_value(12), 4095);
    }

    #[test]
    fn test_cdef_pixel_u16_as_i32() {
        assert_eq!(<u16 as CdefPixel>::as_i32(1023u16), 1023);
        assert_eq!(<u16 as CdefPixel>::as_i32(4095u16), 4095);
    }

    #[test]
    fn test_cdef_pixel_u16_from_clamped() {
        // Values within range are unchanged.
        assert_eq!(<u16 as CdefPixel>::from_clamped(512, 1023), 512u16);
        // Values above max are clamped.
        assert_eq!(<u16 as CdefPixel>::from_clamped(2000, 1023), 1023u16);
        // Values below 0 clamp to 0.
        assert_eq!(<u16 as CdefPixel>::from_clamped(-5, 1023), 0u16);
    }

    // -- cdef_filter_u16 correctness tests -----------------------------------

    #[test]
    fn test_cdef_filter_u16_zero_strength_is_noop() {
        // With pri_strength = 0 and sec_strength = 0, the filter must be a no-op.
        let width = 8usize;
        let height = 8usize;
        let stride = 8usize;
        let mut frame: Vec<u16> = (0..64).map(|i| (i * 16) as u16).collect();
        let original = frame.clone();
        cdef_filter_u16(&mut frame, width, height, stride, 0, 0, 0, 4, 10);
        assert_eq!(frame, original, "zero-strength CDEF must be a no-op");
    }

    #[test]
    fn test_cdef_filter_u16_output_in_range_10bit() {
        // With any strength, output must remain in [0, 1023] for 10-bit.
        let width = 16usize;
        let height = 16usize;
        let stride = 16usize;
        let mut frame: Vec<u16> = (0..256).map(|i| (i as u16 * 4).min(1023)).collect();
        cdef_filter_u16(&mut frame, width, height, stride, 4, 2, 2, 3, 10);
        for &px in &frame {
            assert!(px <= 1023, "10-bit CDEF produced out-of-range value: {px}");
        }
    }

    #[test]
    fn test_cdef_filter_u16_output_in_range_12bit() {
        // Output must remain in [0, 4095] for 12-bit.
        let width = 16usize;
        let height = 16usize;
        let stride = 16usize;
        let mut frame: Vec<u16> = (0..256).map(|i| (i as u16 * 16).min(4095)).collect();
        cdef_filter_u16(&mut frame, width, height, stride, 8, 4, 5, 4, 12);
        for &px in &frame {
            assert!(px <= 4095, "12-bit CDEF produced out-of-range value: {px}");
        }
    }

    #[test]
    fn test_cdef_filter_u16_uniform_plane_unchanged() {
        // A uniform (flat) plane has zero gradients; the weighted filter with a
        // uniform source produces the same pixel value as input.
        let width = 8usize;
        let height = 8usize;
        let stride = 8usize;
        // All pixels are 512 (mid 10-bit).
        let mut frame = vec![512u16; 64];
        cdef_filter_u16(&mut frame, width, height, stride, 8, 4, 0, 5, 10);
        for &px in &frame {
            assert_eq!(px, 512, "uniform plane should be unchanged by CDEF");
        }
    }

    // -- cdef_filter_block_u16 tests -----------------------------------------

    #[test]
    fn test_cdef_filter_block_u16_in_range() {
        let frame_width = 16usize;
        let frame_height = 16usize;
        let stride = frame_width;
        let mut frame: Vec<u16> = (0..256).map(|i| (i as u16 * 4).min(1023)).collect();
        cdef_filter_block_u16(
            &mut frame,
            0,
            0,
            frame_width,
            frame_height,
            stride,
            4,
            2,
            0,
            4,
            10,
        );
        for &px in &frame {
            assert!(
                px <= 1023,
                "block CDEF produced out-of-range 10-bit value: {px}"
            );
        }
    }

    #[test]
    fn test_cdef_filter_block_u16_partial_frame() {
        // Filtering a block at the bottom-right corner (partial) must not panic.
        let frame_width = 10usize;
        let frame_height = 10usize;
        let stride = frame_width;
        let mut frame = vec![500u16; frame_width * frame_height];
        // Block at (6, 6) extends beyond frame — clamping must prevent OOB.
        cdef_filter_block_u16(
            &mut frame,
            6,
            6,
            frame_width,
            frame_height,
            stride,
            2,
            1,
            1,
            3,
            10,
        );
        // Verify all pixels still in range.
        for &px in &frame {
            assert!(px <= 1023);
        }
    }

    // -- cdef_find_direction_u16 tests ---------------------------------------

    #[test]
    fn test_cdef_find_direction_u16_returns_valid_direction() {
        // Any 8×8 block should produce a direction in [0, 7].
        let stride = 8usize;
        let frame: Vec<u16> = (0..64).map(|i| (i as u16) * 16).collect();
        let dir = cdef_find_direction_u16(&frame, stride, 8);
        assert!(dir < 8, "direction must be in range 0–7, got {dir}");
    }

    #[test]
    fn test_cdef_find_direction_u16_uniform_block() {
        // A uniform block has zero variance in all directions — direction 0 is
        // returned as the tie-breaking default.
        let stride = 8usize;
        let frame = vec![512u16; 64];
        let dir = cdef_find_direction_u16(&frame, stride, 8);
        // Tie: first direction (0) wins because it has the smallest variance first.
        assert!(dir < 8);
    }

    // -- Direction helper tests -----------------------------------------------

    #[test]
    fn test_cdef_direction_offset_all_directions() {
        // Each of the 8 directions must return a distinct (dx,dy) pair.
        let offsets: Vec<(i32, i32)> = (0..8).map(cdef_direction_offset).collect();
        assert_eq!(offsets.len(), 8);
        // Each direction must be non-zero (otherwise filtering would be a no-op).
        for (dx, dy) in &offsets {
            assert!(*dx != 0 || *dy != 0, "direction offset must be non-zero");
        }
        // Directions are symmetric: offset(d) == -offset(d+4 mod 8).
        for d in 0..4usize {
            let (dx, dy) = offsets[d];
            let (odx, ody) = offsets[d + 4];
            assert_eq!(
                (-dx, -dy),
                (odx, ody),
                "directions {d} and {} must be opposite",
                d + 4
            );
        }
    }
}
