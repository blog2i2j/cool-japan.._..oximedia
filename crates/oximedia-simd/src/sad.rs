//! SAD (Sum of Absolute Differences) block-matching kernels.
//!
//! SAD is the most widely used distortion metric in motion estimation.  For
//! each candidate block offset the encoder computes the sum of per-pixel
//! absolute differences between the current block and the reference block.
//! The candidate with the lowest SAD wins.
//!
//! This module provides scalar implementations with tight inner loops designed
//! for compiler auto-vectorisation.  All functions accept a `stride` parameter
//! so that they can operate on a region inside a larger image buffer without
//! copying.

// ─── block_sad_16x16 ─────────────────────────────────────────────────────────

/// Compute the Sum of Absolute Differences for two 16×16 luma blocks.
///
/// Both blocks are read from flat `u8` slices with the given row stride.
///
/// # Parameters
///
/// * `a`      — first block's source slice (at least `16 * stride_a` bytes)
/// * `b`      — second block's source slice (at least `16 * stride_b` bytes)
/// * `stride` — row stride in bytes for **both** `a` and `b`.
///   Pass the full image width when the blocks are embedded in full frames.
///
/// # Returns
///
/// The total SAD value as `u32` (maximum 16 × 16 × 255 = 65 280, fits in u32).
///
/// # Panics
///
/// Does not panic; if either slice is too short the function returns `u32::MAX`
/// as a sentinel to indicate invalid input.
#[must_use]
pub fn block_sad_16x16(a: &[u8], b: &[u8], stride: usize) -> u32 {
    block_sad_generic::<16, 16>(a, b, stride, stride)
}

/// Compute SAD for two 16×16 blocks with independent strides.
///
/// This variant allows `a` and `b` to have different row strides, which is
/// useful when comparing a block in a padded frame buffer against a compact
/// motion candidate buffer.
#[must_use]
pub fn block_sad_16x16_strides(a: &[u8], b: &[u8], stride_a: usize, stride_b: usize) -> u32 {
    block_sad_generic::<16, 16>(a, b, stride_a, stride_b)
}

// ─── block_sad_8x8 ───────────────────────────────────────────────────────────

/// Compute the Sum of Absolute Differences for two 8×8 luma blocks.
///
/// Same semantics as [`block_sad_16x16`] but for 8×8 blocks.
/// Maximum result is 8 × 8 × 255 = 16 320.
#[must_use]
pub fn block_sad_8x8(a: &[u8], b: &[u8], stride: usize) -> u32 {
    block_sad_generic::<8, 8>(a, b, stride, stride)
}

/// Compute SAD for two 8×8 blocks with independent strides.
#[must_use]
pub fn block_sad_8x8_strides(a: &[u8], b: &[u8], stride_a: usize, stride_b: usize) -> u32 {
    block_sad_generic::<8, 8>(a, b, stride_a, stride_b)
}

// ─── block_sad_4x4 ───────────────────────────────────────────────────────────

/// Compute the Sum of Absolute Differences for two 4×4 luma blocks.
///
/// Maximum result is 4 × 4 × 255 = 4 080.
#[must_use]
pub fn block_sad_4x4(a: &[u8], b: &[u8], stride: usize) -> u32 {
    block_sad_generic::<4, 4>(a, b, stride, stride)
}

// ─── block_sad_8x4 ──────────────────────────────────────────────────────────

/// Compute the Sum of Absolute Differences for two 8×4 luma blocks.
///
/// Used in H.264-style half-height partitions.
/// Maximum result is 8 × 4 × 255 = 8 160.
#[must_use]
pub fn block_sad_8x4(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize) -> u32 {
    block_sad_generic::<8, 4>(a, b, a_stride, b_stride)
}

// ─── block_sad_4x8 ──────────────────────────────────────────────────────────

/// Compute the Sum of Absolute Differences for two 4×8 luma blocks.
///
/// Used in H.264-style half-width partitions.
/// Maximum result is 4 × 8 × 255 = 8 160.
#[must_use]
pub fn block_sad_4x8(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize) -> u32 {
    block_sad_generic::<4, 8>(a, b, a_stride, b_stride)
}

// ─── block_sad_32x32 ─────────────────────────────────────────────────────────

/// Compute the Sum of Absolute Differences for two 32×32 luma blocks.
///
/// Maximum result is 32 × 32 × 255 = 261 120.
#[must_use]
pub fn block_sad_32x32(a: &[u8], b: &[u8], stride: usize) -> u32 {
    block_sad_generic::<32, 32>(a, b, stride, stride)
}

// ─── Generic implementation ───────────────────────────────────────────────────

/// Generic `W × H` SAD computation.
///
/// Uses a 4× unrolled inner loop to allow the compiler to issue independent
/// load-subtract-accumulate chains.
///
/// Returns `u32::MAX` if either slice is too small.
fn block_sad_generic<const W: usize, const H: usize>(
    a: &[u8],
    b: &[u8],
    stride_a: usize,
    stride_b: usize,
) -> u32 {
    // Validate minimum buffer requirements
    if stride_a < W || stride_b < W {
        return u32::MAX;
    }
    let min_a = (H.saturating_sub(1)) * stride_a + W;
    let min_b = (H.saturating_sub(1)) * stride_b + W;
    if a.len() < min_a || b.len() < min_b {
        return u32::MAX;
    }

    let mut sad: u32 = 0;

    for row in 0..H {
        let a_row = &a[row * stride_a..row * stride_a + W];
        let b_row = &b[row * stride_b..row * stride_b + W];

        // 4× unrolled inner loop
        let mut col = 0;
        while col + 3 < W {
            sad += a_row[col].abs_diff(b_row[col]) as u32;
            sad += a_row[col + 1].abs_diff(b_row[col + 1]) as u32;
            sad += a_row[col + 2].abs_diff(b_row[col + 2]) as u32;
            sad += a_row[col + 3].abs_diff(b_row[col + 3]) as u32;
            col += 4;
        }
        while col < W {
            sad += a_row[col].abs_diff(b_row[col]) as u32;
            col += 1;
        }
    }

    sad
}

// ─── Multi-candidate SAD search ───────────────────────────────────────────────

/// Compute SAD for a list of candidate blocks and return the index and value of
/// the best (lowest) match.
///
/// # Parameters
///
/// * `reference` — reference block (flat, row-major, W×H bytes)
/// * `candidates` — slice of candidate block slices; each must be W×H bytes
/// * `w`         — block width in pixels
/// * `h`         — block height in pixels
///
/// # Returns
///
/// `Some((best_index, best_sad))` or `None` if `candidates` is empty or any
/// candidate is too small.
#[must_use]
pub fn sad_best_match<'a>(
    reference: &[u8],
    candidates: &[&'a [u8]],
    w: usize,
    h: usize,
) -> Option<(usize, u32)> {
    if candidates.is_empty() {
        return None;
    }
    let block_size = w * h;
    let mut best_idx = 0usize;
    let mut best_sad = u32::MAX;

    for (idx, &cand) in candidates.iter().enumerate() {
        if cand.len() < block_size || reference.len() < block_size {
            continue;
        }
        let mut sad = 0u32;
        let mut i = 0;
        while i + 3 < block_size {
            sad += reference[i].abs_diff(cand[i]) as u32;
            sad += reference[i + 1].abs_diff(cand[i + 1]) as u32;
            sad += reference[i + 2].abs_diff(cand[i + 2]) as u32;
            sad += reference[i + 3].abs_diff(cand[i + 3]) as u32;
            i += 4;
        }
        while i < block_size {
            sad += reference[i].abs_diff(cand[i]) as u32;
            i += 1;
        }
        if sad < best_sad {
            best_sad = sad;
            best_idx = idx;
        }
    }

    if best_sad == u32::MAX {
        None
    } else {
        Some((best_idx, best_sad))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(w: usize, h: usize, fill: u8) -> Vec<u8> {
        vec![fill; w * h]
    }

    fn make_block_ramp(w: usize, h: usize, base: u8, step: u8) -> Vec<u8> {
        (0..w * h)
            .map(|i| base.wrapping_add((i as u8).wrapping_mul(step)))
            .collect()
    }

    // ── block_sad_16x16 ───────────────────────────────────────────────────────

    #[test]
    fn sad_16x16_identical_is_zero() {
        let block = make_block(16, 16, 128);
        assert_eq!(block_sad_16x16(&block, &block, 16), 0);
    }

    #[test]
    fn sad_16x16_max_diff() {
        let a = make_block(16, 16, 0);
        let b = make_block(16, 16, 255);
        let expected = 16 * 16 * 255;
        assert_eq!(block_sad_16x16(&a, &b, 16), expected);
    }

    #[test]
    fn sad_16x16_with_stride() {
        // Embed 16×16 blocks inside a 32-wide frame
        let stride = 32usize;
        let mut a = vec![100u8; stride * 16];
        let mut b = vec![200u8; stride * 16];
        // Set the 16-pixel-wide block region
        for row in 0..16 {
            for col in 0..16 {
                a[row * stride + col] = 50;
                b[row * stride + col] = 150;
            }
        }
        let expected = 16 * 16 * 100;
        assert_eq!(block_sad_16x16(&a, &b, stride), expected);
    }

    #[test]
    fn sad_16x16_short_buffer_sentinel() {
        let a = vec![0u8; 16]; // too short
        let b = make_block(16, 16, 0);
        assert_eq!(block_sad_16x16(&a, &b, 16), u32::MAX);
    }

    // ── block_sad_8x8 ─────────────────────────────────────────────────────────

    #[test]
    fn sad_8x8_identical_is_zero() {
        let block = make_block(8, 8, 42);
        assert_eq!(block_sad_8x8(&block, &block, 8), 0);
    }

    #[test]
    fn sad_8x8_max_diff() {
        let a = make_block(8, 8, 0);
        let b = make_block(8, 8, 255);
        assert_eq!(block_sad_8x8(&a, &b, 8), 8 * 8 * 255);
    }

    #[test]
    fn sad_8x8_ramp_diff() {
        let a = make_block_ramp(8, 8, 0, 1);
        let b = make_block(8, 8, 10);
        // Each element a[i] = i, b[i] = 10; diff = |i - 10|
        let expected: u32 = (0..64u32).map(|i| (i as i32 - 10).unsigned_abs()).sum();
        assert_eq!(block_sad_8x8(&a, &b, 8), expected);
    }

    // ── block_sad_4x4 ─────────────────────────────────────────────────────────

    #[test]
    fn sad_4x4_identical_is_zero() {
        let block = make_block(4, 4, 255);
        assert_eq!(block_sad_4x4(&block, &block, 4), 0);
    }

    #[test]
    fn sad_4x4_known_diff() {
        let a = make_block(4, 4, 100);
        let b = make_block(4, 4, 150);
        assert_eq!(block_sad_4x4(&a, &b, 4), 4 * 4 * 50);
    }

    // ── block_sad_32x32 ───────────────────────────────────────────────────────

    #[test]
    fn sad_32x32_identical_is_zero() {
        let block = make_block(32, 32, 200);
        assert_eq!(block_sad_32x32(&block, &block, 32), 0);
    }

    // ── sad_best_match ────────────────────────────────────────────────────────

    #[test]
    fn best_match_returns_closest() {
        let reference = make_block(8, 8, 100);
        let cand0 = make_block(8, 8, 150); // diff = 50 per pixel
        let cand1 = make_block(8, 8, 105); // diff =  5 per pixel
        let cand2 = make_block(8, 8, 200); // diff = 100 per pixel
        let candidates: Vec<&[u8]> = vec![&cand0, &cand1, &cand2];
        let (idx, _sad) = sad_best_match(&reference, &candidates, 8, 8).expect("match found");
        assert_eq!(idx, 1, "cand1 should be the best match");
    }

    #[test]
    fn best_match_empty_returns_none() {
        let reference = make_block(8, 8, 0);
        assert!(sad_best_match(&reference, &[], 8, 8).is_none());
    }

    // ── block_sad_8x4 ─────────────────────────────────────────────────────────

    #[test]
    fn sad_8x4_identical_is_zero() {
        let block = make_block(8, 4, 77);
        assert_eq!(block_sad_8x4(&block, 8, &block, 8), 0);
    }

    #[test]
    fn sad_8x4_known_diff() {
        let a = make_block(8, 4, 100);
        let b = make_block(8, 4, 120);
        assert_eq!(block_sad_8x4(&a, 8, &b, 8), 8 * 4 * 20);
    }

    // ── block_sad_4x8 ─────────────────────────────────────────────────────────

    #[test]
    fn sad_4x8_identical_is_zero() {
        let block = make_block(4, 8, 33);
        assert_eq!(block_sad_4x8(&block, 4, &block, 4), 0);
    }

    #[test]
    fn sad_4x8_known_diff() {
        let a = make_block(4, 8, 50);
        let b = make_block(4, 8, 80);
        assert_eq!(block_sad_4x8(&a, 4, &b, 4), 4 * 8 * 30);
    }

    #[test]
    fn sad_asymmetric_blocks_work_correctly() {
        // 8x4: embed in a larger stride
        let stride = 16usize;
        let mut a = vec![100u8; stride * 4];
        let mut b = vec![200u8; stride * 4];
        for row in 0..4 {
            for col in 0..8 {
                a[row * stride + col] = 10;
                b[row * stride + col] = 60;
            }
        }
        assert_eq!(block_sad_8x4(&a, stride, &b, stride), 8 * 4 * 50);

        // 4x8: embed in a larger stride
        let mut a2 = vec![100u8; stride * 8];
        let mut b2 = vec![200u8; stride * 8];
        for row in 0..8 {
            for col in 0..4 {
                a2[row * stride + col] = 20;
                b2[row * stride + col] = 45;
            }
        }
        assert_eq!(block_sad_4x8(&a2, stride, &b2, stride), 4 * 8 * 25);
    }

    // ── sad_best_match ────────────────────────────────────────────────────────

    #[test]
    fn best_match_identical_candidate_gives_zero_sad() {
        let block = make_block(4, 4, 77);
        let candidates: Vec<&[u8]> = vec![&block];
        let (idx, sad) = sad_best_match(&block, &candidates, 4, 4).expect("match");
        assert_eq!(idx, 0);
        assert_eq!(sad, 0);
    }
}
