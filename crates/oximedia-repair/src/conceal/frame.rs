//! Frame concealment.
//!
//! This module provides spatial and temporal error concealment for corrupt
//! video frames. Spatial concealment uses surrounding pixel data within the
//! same frame; temporal concealment leverages adjacent frames.

use crate::Result;

/// Conceal corrupt frame by copying the previous (reference) frame.
///
/// Copies as many bytes as the shorter of the two buffers allows,
/// leaving any trailing bytes in `corrupt_frame` untouched.
pub fn conceal_with_previous(corrupt_frame: &mut [u8], previous_frame: &[u8]) -> Result<()> {
    let len = corrupt_frame.len().min(previous_frame.len());
    corrupt_frame[..len].copy_from_slice(&previous_frame[..len]);
    Ok(())
}

/// Conceal corrupt frame by temporal interpolation between the previous and
/// next frames.
///
/// Each pixel is the arithmetic mean of the co-located pixels in the two
/// reference frames. Only the overlapping portion of all three buffers is
/// touched.
pub fn conceal_with_interpolation(
    corrupt_frame: &mut [u8],
    previous_frame: &[u8],
    next_frame: &[u8],
) -> Result<()> {
    let len = corrupt_frame
        .len()
        .min(previous_frame.len())
        .min(next_frame.len());

    for i in 0..len {
        corrupt_frame[i] = ((u16::from(previous_frame[i]) + u16::from(next_frame[i])) / 2) as u8;
    }

    Ok(())
}

/// Conceal a corrupt region within a frame using spatial interpolation.
///
/// The corrupt region is a rectangular block defined by `(block_x, block_y)`
/// in pixels and `(block_w, block_h)` in pixels, inside a frame of the given
/// `width` and `height`. Surrounding pixels are bilinearly averaged to fill
/// the damaged block.
pub fn conceal_spatial(
    frame: &mut [u8],
    width: usize,
    height: usize,
    block_x: usize,
    block_y: usize,
    block_w: usize,
    block_h: usize,
) -> Result<()> {
    if width == 0 || height == 0 || block_w == 0 || block_h == 0 {
        return Ok(());
    }

    // Clamp block bounds to frame dimensions
    let x_end = (block_x + block_w).min(width);
    let y_end = (block_y + block_h).min(height);
    let x_start = block_x.min(width);
    let y_start = block_y.min(height);

    if x_start >= x_end || y_start >= y_end {
        return Ok(());
    }

    // Collect border pixel averages from the surrounding ring.
    // Top border row (y_start - 1), bottom border row (y_end), left column
    // (x_start - 1), right column (x_end).
    let top_avg = if y_start > 0 {
        let row = y_start - 1;
        let sum: u32 = (x_start..x_end)
            .map(|x| u32::from(frame[row * width + x]))
            .sum();
        Some(sum as f64 / (x_end - x_start) as f64)
    } else {
        None
    };

    let bottom_avg = if y_end < height {
        let row = y_end;
        let sum: u32 = (x_start..x_end)
            .map(|x| u32::from(frame[row * width + x]))
            .sum();
        Some(sum as f64 / (x_end - x_start) as f64)
    } else {
        None
    };

    let left_avg = if x_start > 0 {
        let col = x_start - 1;
        let sum: u32 = (y_start..y_end)
            .map(|y| u32::from(frame[y * width + col]))
            .sum();
        Some(sum as f64 / (y_end - y_start) as f64)
    } else {
        None
    };

    let right_avg = if x_end < width {
        let col = x_end;
        let sum: u32 = (y_start..y_end)
            .map(|y| u32::from(frame[y * width + col]))
            .sum();
        Some(sum as f64 / (y_end - y_start) as f64)
    } else {
        None
    };

    // Bilinear interpolation from available borders
    let bw = (x_end - x_start) as f64;
    let bh = (y_end - y_start) as f64;

    for y in y_start..y_end {
        for x in x_start..x_end {
            let fy = (y - y_start) as f64 / bh; // 0..1 top-to-bottom
            let fx = (x - x_start) as f64 / bw; // 0..1 left-to-right

            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;

            if let Some(top) = top_avg {
                let w = 1.0 - fy;
                weighted_sum += top * w;
                weight_total += w;
            }
            if let Some(bottom) = bottom_avg {
                let w = fy;
                weighted_sum += bottom * w;
                weight_total += w;
            }
            if let Some(left) = left_avg {
                let w = 1.0 - fx;
                weighted_sum += left * w;
                weight_total += w;
            }
            if let Some(right) = right_avg {
                let w = fx;
                weighted_sum += right * w;
                weight_total += w;
            }

            let value = if weight_total > 0.0 {
                (weighted_sum / weight_total).clamp(0.0, 255.0) as u8
            } else {
                128 // neutral grey fallback
            };

            frame[y * width + x] = value;
        }
    }

    Ok(())
}

/// Conceal corrupt frame by weighted temporal interpolation.
///
/// Unlike simple averaging, this applies a per-pixel weight based on motion
/// estimation between the two reference frames. Regions with low motion
/// (similar in both frames) receive high confidence interpolation; regions
/// with high motion fall back towards the previous frame.
pub fn conceal_motion_compensated(
    corrupt_frame: &mut [u8],
    previous_frame: &[u8],
    next_frame: &[u8],
    width: usize,
    block_size: usize,
) -> Result<()> {
    let len = corrupt_frame
        .len()
        .min(previous_frame.len())
        .min(next_frame.len());

    if len == 0 || width == 0 || block_size == 0 {
        return Ok(());
    }

    let height = len / width;
    let blocks_x = (width + block_size - 1) / block_size;
    let blocks_y = (height + block_size - 1) / block_size;

    // Compute per-block SAD between prev and next to estimate motion
    let mut block_weights: Vec<f64> = Vec::with_capacity(blocks_x * blocks_y);

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut sad: u64 = 0;
            let mut count: u64 = 0;

            for dy in 0..block_size {
                let y = by * block_size + dy;
                if y >= height {
                    break;
                }
                for dx in 0..block_size {
                    let x = bx * block_size + dx;
                    if x >= width {
                        break;
                    }
                    let idx = y * width + x;
                    if idx < len {
                        sad += u64::from(previous_frame[idx].abs_diff(next_frame[idx]));
                        count += 1;
                    }
                }
            }

            // Low SAD = low motion = trust interpolation
            // High SAD = high motion = fall back to previous frame
            let avg_sad = if count > 0 {
                sad as f64 / count as f64
            } else {
                0.0
            };

            // Weight for next_frame contribution: 0.5 at zero motion, 0 at high motion
            let next_weight = 0.5 * (-avg_sad / 32.0).exp();
            block_weights.push(next_weight);
        }
    }

    // Apply per-pixel interpolation with block-level weights
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if idx >= len {
                break;
            }

            let bx = x / block_size;
            let by = y / block_size;
            let block_idx = by * blocks_x + bx;

            let next_w = block_weights.get(block_idx).copied().unwrap_or(0.5);
            let prev_w = 1.0 - next_w;

            let value =
                f64::from(previous_frame[idx]) * prev_w + f64::from(next_frame[idx]) * next_w;

            corrupt_frame[idx] = value.clamp(0.0, 255.0) as u8;
        }
    }

    Ok(())
}

/// Insert black frame.
pub fn insert_black_frame(frame: &mut [u8]) {
    frame.fill(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conceal_with_previous() {
        let mut corrupt = vec![0u8; 10];
        let prev = vec![42u8; 10];
        conceal_with_previous(&mut corrupt, &prev).expect("concealment should succeed");
        assert_eq!(corrupt, vec![42u8; 10]);
    }

    #[test]
    fn test_conceal_with_previous_different_lengths() {
        let mut corrupt = vec![0u8; 10];
        let prev = vec![42u8; 5];
        conceal_with_previous(&mut corrupt, &prev).expect("concealment should succeed");
        assert_eq!(&corrupt[..5], &[42, 42, 42, 42, 42]);
        assert_eq!(&corrupt[5..], &[0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_conceal_with_interpolation() {
        let mut corrupt = vec![0; 10];
        let prev = vec![100; 10];
        let next = vec![100; 10];

        conceal_with_interpolation(&mut corrupt, &prev, &next).expect("concealment should succeed");
        assert_eq!(corrupt[0], 100);
    }

    #[test]
    fn test_conceal_with_interpolation_different_values() {
        let mut corrupt = vec![0; 4];
        let prev = vec![0, 0, 0, 0];
        let next = vec![200, 200, 200, 200];

        conceal_with_interpolation(&mut corrupt, &prev, &next).expect("concealment should succeed");
        assert_eq!(corrupt[0], 100);
    }

    #[test]
    fn test_conceal_spatial_surrounded() {
        // 8x8 frame: set border pixels to 100, damage the 4x4 center
        let mut frame = vec![100u8; 64];
        // Damage center 2x2 block
        for y in 3..5 {
            for x in 3..5 {
                frame[y * 8 + x] = 0;
            }
        }
        conceal_spatial(&mut frame, 8, 8, 3, 3, 2, 2).expect("spatial concealment should succeed");
        // Center pixels should be close to 100 (interpolated from surrounding 100s)
        for y in 3..5 {
            for x in 3..5 {
                assert!(
                    frame[y * 8 + x] > 80,
                    "pixel ({x},{y}) = {}",
                    frame[y * 8 + x]
                );
            }
        }
    }

    #[test]
    fn test_conceal_spatial_empty() {
        let mut frame = vec![128u8; 64];
        conceal_spatial(&mut frame, 0, 0, 0, 0, 0, 0).expect("empty should succeed");
    }

    #[test]
    fn test_conceal_motion_compensated() {
        let mut corrupt = vec![0u8; 64]; // 8x8
        let prev = vec![100u8; 64];
        let next = vec![100u8; 64]; // Same as prev = no motion

        conceal_motion_compensated(&mut corrupt, &prev, &next, 8, 4)
            .expect("motion compensated concealment should succeed");

        // With zero motion, result should be close to 100
        for &val in &corrupt {
            assert!(val > 90, "expected close to 100, got {val}");
        }
    }

    #[test]
    fn test_insert_black_frame() {
        let mut frame = vec![255; 10];
        insert_black_frame(&mut frame);
        assert!(frame.iter().all(|&b| b == 0));
    }
}
