//! Data augmentation for image tensors.
//!
//! Provides common augmentation operations used during training:
//!
//! * Horizontal flip
//! * Random crop
//! * Brightness jitter
//! * Contrast jitter
//! * Vertical flip
//!
//! All operations work on flat `Vec<f32>` image buffers in **CHW** layout
//! (channels, height, width) with values in any range (typically `[0, 1]`).
//! No external crates are used; randomness comes from a pure-Rust LCG.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_neural::augmentation::{AugmentConfig, augment};
//!
//! let image = vec![0.5_f32; 3 * 32 * 32]; // 3×32×32 CHW
//! let (aug, _new_h, _new_w) = augment(&image, 3, 32, 32, &AugmentConfig::default(), 42);
//! assert_eq!(aug.len(), image.len());
//! ```

/// Configuration for data augmentation.
#[derive(Debug, Clone)]
pub struct AugmentConfig {
    /// Probability of applying horizontal flip (0.0-1.0).
    pub horizontal_flip_prob: f32,
    /// Probability of applying vertical flip (0.0-1.0).
    pub vertical_flip_prob: f32,
    /// Brightness jitter range `[-delta, delta]` (0 = disabled).
    pub brightness_jitter: f32,
    /// Contrast jitter multiplier range `[1-delta, 1+delta]` (0 = disabled).
    pub contrast_jitter: f32,
    /// Random crop: fraction of image to keep (0 = disabled, 1 = full image).
    /// E.g. 0.875 → crop a sub-region 87.5% the size of the original.
    pub crop_fraction: f32,
}

impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            horizontal_flip_prob: 0.5,
            vertical_flip_prob: 0.0,
            brightness_jitter: 0.1,
            contrast_jitter: 0.1,
            crop_fraction: 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LCG RNG helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Advance a 64-bit LCG state and return a value in [0, 1).
#[inline]
fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // Use upper 32 bits for better randomness
    ((*state >> 32) as f32) / (u32::MAX as f32)
}

/// Return a value in `[-1, 1)`.
#[inline]
fn lcg_signed(state: &mut u64) -> f32 {
    lcg_next(state) * 2.0 - 1.0
}

// ─────────────────────────────────────────────────────────────────────────────
// Individual augmentation operations
// ─────────────────────────────────────────────────────────────────────────────

/// Flip a CHW image horizontally in-place.
///
/// For each row in each channel, the pixel order is reversed left-to-right.
pub fn horizontal_flip(image: &mut [f32], channels: usize, height: usize, width: usize) {
    for c in 0..channels {
        for row in 0..height {
            let base = c * height * width + row * width;
            let row_slice = &mut image[base..base + width];
            row_slice.reverse();
        }
    }
}

/// Flip a CHW image vertically in-place.
///
/// For each channel, the row order is reversed top-to-bottom.
pub fn vertical_flip(image: &mut [f32], channels: usize, height: usize, width: usize) {
    for c in 0..channels {
        let chan_base = c * height * width;
        for row in 0..(height / 2) {
            let top = chan_base + row * width;
            let bot = chan_base + (height - 1 - row) * width;
            for x in 0..width {
                image.swap(top + x, bot + x);
            }
        }
    }
}

/// Add a brightness offset `delta` to every pixel (clamped to `[0, 1]`).
pub fn brightness_adjust(image: &mut [f32], delta: f32) {
    for v in image.iter_mut() {
        *v = (*v + delta).clamp(0.0, 1.0);
    }
}

/// Scale contrast around the mean: `pixel = (pixel - mean) * factor + mean`.
///
/// Values are clamped to `[0, 1]`.
pub fn contrast_adjust(image: &mut [f32], factor: f32) {
    if image.is_empty() {
        return;
    }
    let mean = image.iter().sum::<f32>() / image.len() as f32;
    for v in image.iter_mut() {
        *v = ((*v - mean) * factor + mean).clamp(0.0, 1.0);
    }
}

/// Crop a central sub-region of a CHW image.
///
/// `crop_fraction` is the fraction of the original dimensions to retain
/// (e.g. 0.875 → keep 87.5% of height and width, centred).
///
/// Returns the cropped image in CHW layout with the new height and width.
///
/// # Panics
///
/// Does not panic; if `crop_fraction` is outside `(0, 1]` the image is
/// returned unchanged.
#[must_use]
pub fn center_crop(
    image: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    crop_fraction: f32,
) -> (Vec<f32>, usize, usize) {
    if crop_fraction <= 0.0 || crop_fraction >= 1.0 {
        return (image.to_vec(), height, width);
    }

    let new_h = ((height as f32 * crop_fraction) as usize).max(1);
    let new_w = ((width as f32 * crop_fraction) as usize).max(1);
    let off_y = (height - new_h) / 2;
    let off_x = (width - new_w) / 2;

    let mut out = vec![0.0_f32; channels * new_h * new_w];
    for c in 0..channels {
        for nh in 0..new_h {
            for nw in 0..new_w {
                let src_idx = c * height * width + (off_y + nh) * width + (off_x + nw);
                let dst_idx = c * new_h * new_w + nh * new_w + nw;
                out[dst_idx] = image[src_idx];
            }
        }
    }
    (out, new_h, new_w)
}

/// Random crop: crop a sub-region of size `crop_fraction * (H, W)` at a
/// random offset determined by `rng_seed`.
///
/// Returns `(cropped_image, new_height, new_width)`.
#[must_use]
pub fn random_crop(
    image: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    crop_fraction: f32,
    rng_seed: u64,
) -> (Vec<f32>, usize, usize) {
    if crop_fraction <= 0.0 || crop_fraction >= 1.0 {
        return (image.to_vec(), height, width);
    }

    let new_h = ((height as f32 * crop_fraction) as usize).max(1);
    let new_w = ((width as f32 * crop_fraction) as usize).max(1);

    let mut state = rng_seed.wrapping_add(1);
    let max_off_y = height - new_h;
    let max_off_x = width - new_w;

    let off_y = if max_off_y > 0 {
        let r = lcg_next(&mut state);
        (r * max_off_y as f32) as usize
    } else {
        0
    };
    let off_x = if max_off_x > 0 {
        let r = lcg_next(&mut state);
        (r * max_off_x as f32) as usize
    } else {
        0
    };

    let mut out = vec![0.0_f32; channels * new_h * new_w];
    for c in 0..channels {
        for nh in 0..new_h {
            for nw in 0..new_w {
                let src_idx = c * height * width + (off_y + nh) * width + (off_x + nw);
                let dst_idx = c * new_h * new_w + nh * new_w + nw;
                out[dst_idx] = image[src_idx];
            }
        }
    }
    (out, new_h, new_w)
}

// ─────────────────────────────────────────────────────────────────────────────
// Combined augmentation
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a full augmentation pipeline to a CHW image.
///
/// Returns `(augmented_image, new_height, new_width)`.  When no spatial
/// transforms are applied, height and width are unchanged.
#[must_use]
pub fn augment(
    image: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    config: &AugmentConfig,
    seed: u64,
) -> (Vec<f32>, usize, usize) {
    let mut state = seed.wrapping_add(0xdeadbeef);
    let mut out = image.to_vec();
    let mut cur_h = height;
    let mut cur_w = width;

    // Random crop first (changes dimensions)
    if config.crop_fraction > 0.0 && config.crop_fraction < 1.0 {
        let crop_seed = {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };
        let (cropped, nh, nw) = random_crop(
            &out,
            channels,
            cur_h,
            cur_w,
            config.crop_fraction,
            crop_seed,
        );
        out = cropped;
        cur_h = nh;
        cur_w = nw;
    }

    // Horizontal flip
    if config.horizontal_flip_prob > 0.0 {
        let p = lcg_next(&mut state);
        if p < config.horizontal_flip_prob {
            horizontal_flip(&mut out, channels, cur_h, cur_w);
        }
    }

    // Vertical flip
    if config.vertical_flip_prob > 0.0 {
        let p = lcg_next(&mut state);
        if p < config.vertical_flip_prob {
            vertical_flip(&mut out, channels, cur_h, cur_w);
        }
    }

    // Brightness jitter
    if config.brightness_jitter > 0.0 {
        let delta = lcg_signed(&mut state) * config.brightness_jitter;
        brightness_adjust(&mut out, delta);
    }

    // Contrast jitter
    if config.contrast_jitter > 0.0 {
        let d = lcg_signed(&mut state) * config.contrast_jitter;
        let factor = 1.0 + d;
        contrast_adjust(&mut out, factor.max(0.0));
    }

    (out, cur_h, cur_w)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chw(channels: usize, height: usize, width: usize) -> Vec<f32> {
        let n = channels * height * width;
        (0..n).map(|i| i as f32 / n as f32).collect()
    }

    #[test]
    fn test_horizontal_flip_involution() {
        let orig = make_chw(3, 8, 8);
        let mut flipped = orig.clone();
        horizontal_flip(&mut flipped, 3, 8, 8);
        horizontal_flip(&mut flipped, 3, 8, 8);
        assert_eq!(orig, flipped, "double flip should restore original");
    }

    #[test]
    fn test_horizontal_flip_first_row() {
        // Single channel, 1×4 image: [0.1, 0.2, 0.3, 0.4] → [0.4, 0.3, 0.2, 0.1]
        let mut img = vec![0.1_f32, 0.2, 0.3, 0.4];
        horizontal_flip(&mut img, 1, 1, 4);
        assert!((img[0] - 0.4).abs() < 1e-6);
        assert!((img[3] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_vertical_flip_involution() {
        let orig = make_chw(2, 6, 6);
        let mut flipped = orig.clone();
        vertical_flip(&mut flipped, 2, 6, 6);
        vertical_flip(&mut flipped, 2, 6, 6);
        assert_eq!(orig, flipped, "double flip should restore original");
    }

    #[test]
    fn test_brightness_clamp() {
        let mut img = vec![0.8_f32, 0.5, 0.1];
        brightness_adjust(&mut img, 0.5);
        assert!((img[0] - 1.0).abs() < 1e-6); // clamped
        assert!((img[1] - 1.0).abs() < 1e-6);
        assert!((img[2] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_brightness_negative_clamp() {
        let mut img = vec![0.2_f32, 0.05];
        brightness_adjust(&mut img, -0.3);
        assert!(img[0] >= 0.0);
        assert!(img[1] >= 0.0);
    }

    #[test]
    fn test_contrast_adjust_unit_factor() {
        let orig = vec![0.3_f32, 0.5, 0.7];
        let mut img = orig.clone();
        contrast_adjust(&mut img, 1.0);
        for (a, b) in orig.iter().zip(img.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_center_crop_dimensions() {
        let img = make_chw(3, 32, 32);
        let (cropped, h, w) = center_crop(&img, 3, 32, 32, 0.5);
        assert_eq!(h, 16);
        assert_eq!(w, 16);
        assert_eq!(cropped.len(), 3 * 16 * 16);
    }

    #[test]
    fn test_center_crop_identity_on_zero() {
        let img = make_chw(1, 10, 10);
        let (cropped, h, w) = center_crop(&img, 1, 10, 10, 0.0);
        assert_eq!(h, 10);
        assert_eq!(w, 10);
        assert_eq!(cropped.len(), img.len());
    }

    #[test]
    fn test_random_crop_size() {
        let img = make_chw(1, 64, 64);
        let (cropped, h, w) = random_crop(&img, 1, 64, 64, 0.875, 42);
        let expected_dim = (64.0 * 0.875) as usize; // = 56
        assert_eq!(h, expected_dim);
        assert_eq!(w, expected_dim);
        assert_eq!(cropped.len(), expected_dim * expected_dim);
    }

    #[test]
    fn test_random_crop_different_seeds_may_differ() {
        let img = make_chw(1, 32, 32);
        let (c1, _, _) = random_crop(&img, 1, 32, 32, 0.5, 1);
        let (c2, _, _) = random_crop(&img, 1, 32, 32, 0.5, 999_999);
        // Different seeds → different crops (with overwhelming probability)
        // Just check they're the same shape
        assert_eq!(c1.len(), c2.len());
    }

    #[test]
    fn test_augment_preserves_size_without_crop() {
        let img = make_chw(3, 16, 16);
        let config = AugmentConfig {
            horizontal_flip_prob: 0.5,
            vertical_flip_prob: 0.5,
            brightness_jitter: 0.1,
            contrast_jitter: 0.1,
            crop_fraction: 0.0, // no crop
        };
        let (aug, h, w) = augment(&img, 3, 16, 16, &config, 7);
        assert_eq!(h, 16);
        assert_eq!(w, 16);
        assert_eq!(aug.len(), img.len());
    }

    #[test]
    fn test_augment_with_crop_changes_size() {
        let img = make_chw(1, 32, 32);
        let config = AugmentConfig {
            crop_fraction: 0.75,
            horizontal_flip_prob: 0.0,
            vertical_flip_prob: 0.0,
            brightness_jitter: 0.0,
            contrast_jitter: 0.0,
        };
        let (aug, h, w) = augment(&img, 1, 32, 32, &config, 0);
        let expected_dim = (32.0_f32 * 0.75) as usize; // 24
        assert_eq!(h, expected_dim);
        assert_eq!(w, expected_dim);
        assert_eq!(aug.len(), expected_dim * expected_dim);
    }

    #[test]
    fn test_augment_values_in_range() {
        let img = vec![0.5_f32; 3 * 8 * 8];
        let config = AugmentConfig::default();
        let (aug, _h, _w) = augment(&img, 3, 8, 8, &config, 1234);
        for v in &aug {
            assert!(*v >= 0.0 && *v <= 1.0, "out-of-range: {v}");
        }
    }
}
