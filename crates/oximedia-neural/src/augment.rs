//! Simple data augmentation helpers for image tensors.
//!
//! Provides standalone functions operating on flat `Vec<u8>` image buffers in
//! **HWC** layout (height × width × channels, i.e. row-major with interleaved
//! channels) with pixel values in `[0, 255]`.
//!
//! All randomness comes from the project-standard LCG; no external crates.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_neural::augment::{flip_horizontal, brightness_jitter, random_crop};
//!
//! // 4×4 RGB image, all pixels = 100
//! let img = vec![100u8; 4 * 4 * 3];
//!
//! let flipped = flip_horizontal(&img, 4, 4);
//! assert_eq!(flipped.len(), img.len());
//!
//! let jittered = brightness_jitter(&img, 1.5);
//! assert!(jittered.iter().all(|&v| v > 100));
//!
//! let cropped = random_crop(&img, 4, 4, 2, 2, 42);
//! assert_eq!(cropped.len(), 2 * 2 * 3);
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// LCG helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn lcg_step(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Flip an HWC image horizontally (mirror left↔right) and return the result.
///
/// * `img` — flat buffer of `h * w * channels` bytes.
/// * `w`   — image width in pixels.
/// * `h`   — image height in pixels.
///
/// The number of channels is inferred as `img.len() / (h * w)`.
/// Returns the original slice as-is if dimensions are inconsistent.
#[must_use]
pub fn flip_horizontal(img: &[u8], w: u32, h: u32) -> Vec<u8> {
    let w = w as usize;
    let h = h as usize;
    let total = w * h;
    if total == 0 || img.len() % total != 0 {
        return img.to_vec();
    }
    let channels = img.len() / total;
    let row_bytes = w * channels;

    let mut out = img.to_vec();
    for row in 0..h {
        let row_start = row * row_bytes;
        let row_slice = &mut out[row_start..row_start + row_bytes];
        // Reverse pixel order within the row
        for x in 0..(w / 2) {
            let left = x * channels;
            let right = (w - 1 - x) * channels;
            for c in 0..channels {
                row_slice.swap(left + c, right + c);
            }
        }
    }
    out
}

/// Randomly crop a sub-region from an HWC image using an LCG seed.
///
/// * `img`    — flat HWC byte buffer.
/// * `w`, `h` — original image dimensions.
/// * `crop_w`, `crop_h` — dimensions of the output crop.
/// * `seed`   — LCG seed.
///
/// If `crop_w >= w` or `crop_h >= h` the image is returned unchanged.
/// If dimensions are inconsistent the image is returned unchanged.
///
/// Returns a newly allocated buffer of `crop_h * crop_w * channels` bytes.
#[must_use]
pub fn random_crop(img: &[u8], w: u32, h: u32, crop_w: u32, crop_h: u32, seed: u64) -> Vec<u8> {
    let w = w as usize;
    let h = h as usize;
    let cw = crop_w as usize;
    let ch = crop_h as usize;

    if cw >= w || ch >= h || w == 0 || h == 0 {
        return img.to_vec();
    }
    let total = w * h;
    if img.len() % total != 0 {
        return img.to_vec();
    }
    let channels = img.len() / total;

    // Derive top-left corner from LCG
    let max_x = (w - cw) as u64;
    let max_y = (h - ch) as u64;

    let s1 = lcg_step(seed);
    let x_off = ((s1 >> 32) % (max_x + 1)) as usize;
    let s2 = lcg_step(s1);
    let y_off = ((s2 >> 32) % (max_y + 1)) as usize;

    let row_bytes = w * channels;
    let crop_row_bytes = cw * channels;
    let mut out = Vec::with_capacity(ch * crop_row_bytes);

    for row in 0..ch {
        let src_row = y_off + row;
        let src_col = x_off * channels;
        let src_start = src_row * row_bytes + src_col;
        out.extend_from_slice(&img[src_start..src_start + crop_row_bytes]);
    }

    out
}

/// Apply brightness jitter to an HWC image.
///
/// Each pixel channel value is multiplied by `factor` and clamped to `[0, 255]`.
/// `factor > 1.0` brightens, `factor < 1.0` darkens.
///
/// Returns a newly allocated buffer with the same dimensions.
#[must_use]
pub fn brightness_jitter(img: &[u8], factor: f32) -> Vec<u8> {
    img.iter()
        .map(|&v| (v as f32 * factor).clamp(0.0, 255.0) as u8)
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ---- flip_horizontal ----

    #[test]
    fn test_flip_horizontal_rgb() {
        // 2×2 RGB image: [R0, G0, B0, R1, G1, B1] (row 0)
        //                [R2, G2, B2, R3, G3, B3] (row 1)
        let img: Vec<u8> = vec![
            10, 20, 30, // pixel (0,0)
            40, 50, 60, // pixel (0,1)
            70, 80, 90, // pixel (1,0)
            100, 110, 120, // pixel (1,1)
        ];
        let flipped = flip_horizontal(&img, 2, 2);
        // Row 0: pixels swapped → (0,1) then (0,0)
        assert_eq!(&flipped[0..6], &[40, 50, 60, 10, 20, 30]);
        // Row 1: pixels swapped → (1,1) then (1,0)
        assert_eq!(&flipped[6..12], &[100, 110, 120, 70, 80, 90]);
    }

    #[test]
    fn test_flip_horizontal_idempotent() {
        let img = vec![1u8, 2, 3, 4, 5, 6]; // 2×1 RGB
        let once = flip_horizontal(&img, 2, 1);
        let twice = flip_horizontal(&once, 2, 1);
        assert_eq!(img, twice, "flipping twice should restore original");
    }

    #[test]
    fn test_flip_single_pixel() {
        let img = vec![42u8, 43, 44]; // 1×1 RGB
        let flipped = flip_horizontal(&img, 1, 1);
        assert_eq!(flipped, img, "single-pixel flip is no-op");
    }

    #[test]
    fn test_flip_wrong_dimensions_returns_original() {
        let img = vec![1u8; 7]; // 7 bytes, can't form h*w*ch
        let out = flip_horizontal(&img, 2, 2); // 2*2=4, not 7/4 cleanly
        assert_eq!(out, img);
    }

    // ---- random_crop ----

    #[test]
    fn test_random_crop_output_size() {
        let img = vec![0u8; 8 * 8 * 3]; // 8×8 RGB
        let crop = random_crop(&img, 8, 8, 4, 4, 42);
        assert_eq!(crop.len(), 4 * 4 * 3, "crop should have 4×4 RGB pixels");
    }

    #[test]
    fn test_random_crop_at_boundary() {
        let img: Vec<u8> = (0..255).cycle().take(10 * 6 * 1).map(|v| v as u8).collect();
        let crop = random_crop(&img, 10, 6, 8, 4, 0);
        assert_eq!(crop.len(), 8 * 4 * 1);
    }

    #[test]
    fn test_random_crop_full_size_returns_original() {
        let img = vec![7u8; 5 * 5 * 2];
        let crop = random_crop(&img, 5, 5, 5, 5, 0); // crop_w >= w
        assert_eq!(crop, img);
    }

    #[test]
    fn test_random_crop_different_seeds_may_differ() {
        let img: Vec<u8> = (0..100)
            .cycle()
            .take(16 * 16 * 1)
            .map(|v| v as u8)
            .collect();
        let c1 = random_crop(&img, 16, 16, 8, 8, 1);
        let c2 = random_crop(&img, 16, 16, 8, 8, 99999);
        // Not guaranteed to differ but very likely with different seeds
        // at least they must have the same length
        assert_eq!(c1.len(), c2.len());
    }

    // ---- brightness_jitter ----

    #[test]
    fn test_brightness_jitter_brightens() {
        let img = vec![100u8; 12];
        let bright = brightness_jitter(&img, 1.5);
        assert!(
            bright.iter().all(|&v| v >= 100),
            "factor > 1.0 should brighten"
        );
    }

    #[test]
    fn test_brightness_jitter_darkens() {
        let img = vec![200u8; 12];
        let dark = brightness_jitter(&img, 0.5);
        assert!(dark.iter().all(|&v| v <= 200), "factor < 1.0 should darken");
    }

    #[test]
    fn test_brightness_jitter_clamps_to_255() {
        let img = vec![200u8; 4];
        let bright = brightness_jitter(&img, 2.0);
        assert!(
            bright.iter().all(|&v| v == 255),
            "values should clamp to 255"
        );
    }

    #[test]
    fn test_brightness_jitter_clamps_to_zero() {
        let img = vec![100u8; 4];
        let dark = brightness_jitter(&img, 0.0);
        assert!(
            dark.iter().all(|&v| v == 0),
            "factor=0 should produce all zeros"
        );
    }

    #[test]
    fn test_brightness_jitter_identity() {
        let img = vec![128u8; 6];
        let out = brightness_jitter(&img, 1.0);
        assert_eq!(out, img, "factor=1.0 should be identity");
    }
}
