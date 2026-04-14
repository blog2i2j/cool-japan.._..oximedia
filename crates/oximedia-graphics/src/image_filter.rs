//! Image filter primitives for RGBA8 pixel buffers.
//!
//! All functions operate on RGBA8 interleaved buffers: `&mut [u8]` with
//! `width * height * 4` bytes, row-major. Each pixel is stored as
//! `[R, G, B, A]` with each channel in `[0, 255]`.
//!
//! # Performance
//! - `box_blur`: O(1) per pixel regardless of radius via sliding-window prefix sums.
//! - `gaussian_blur`: Separable two-pass convolution with rayon row parallelism.
//! - Morphological ops (`dilate_alpha`, `erode_alpha`): Separable 1D pass on alpha only.

use rayon::prelude::*;

/// Fast separable box blur using a sliding-window running sum. O(1) per pixel regardless of radius.
///
/// Operates independently on all four RGBA channels. Two passes: horizontal then vertical.
/// `radius = 0` is a no-op.
pub fn box_blur(pixels: &mut [u8], width: u32, height: u32, radius: u32) {
    if radius == 0 || width == 0 || height == 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let r = radius as usize;

    // --- Horizontal pass ---
    let mut temp = pixels.to_vec();
    for row in 0..h {
        for ch in 0..4usize {
            horizontal_box_blur_row(&pixels[row * w * 4..row * w * 4 + w * 4],
                                    &mut temp[row * w * 4..row * w * 4 + w * 4],
                                    w, ch, r);
        }
    }

    // --- Vertical pass ---
    for col in 0..w {
        for ch in 0..4usize {
            vertical_box_blur_col(&temp, pixels, w, h, col, ch, r);
        }
    }
}

/// Single-row horizontal box blur for one channel.
fn horizontal_box_blur_row(src: &[u8], dst: &mut [u8], width: usize, ch: usize, radius: usize) {
    if width == 0 {
        return;
    }
    let diameter = 2 * radius + 1;
    // Build initial running sum over [-radius, radius] clamped to [0, width)
    let mut sum: u32 = 0;
    for k in 0..=2 * radius {
        let col = k.saturating_sub(radius).min(width - 1);
        sum += src[col * 4 + ch] as u32;
    }
    for x in 0..width {
        dst[x * 4 + ch] = ((sum + diameter as u32 / 2) / diameter as u32) as u8;
        // Advance window: remove left edge, add right edge
        let remove_idx = x.saturating_sub(radius);
        let add_idx = (x + radius + 1).min(width - 1);
        sum = sum - src[remove_idx * 4 + ch] as u32 + src[add_idx * 4 + ch] as u32;
    }
}

/// Single-column vertical box blur for one channel.
fn vertical_box_blur_col(src: &[u8], dst: &mut [u8], width: usize, height: usize,
                          col: usize, ch: usize, radius: usize) {
    if height == 0 {
        return;
    }
    let diameter = 2 * radius + 1;
    let pixel_stride = width * 4;

    let mut sum: u32 = 0;
    for k in 0..=2 * radius {
        let row = k.saturating_sub(radius).min(height - 1);
        sum += src[row * pixel_stride + col * 4 + ch] as u32;
    }
    for y in 0..height {
        dst[y * pixel_stride + col * 4 + ch] = ((sum + diameter as u32 / 2) / diameter as u32) as u8;
        let remove_row = y.saturating_sub(radius);
        let add_row = (y + radius + 1).min(height - 1);
        sum = sum - src[remove_row * pixel_stride + col * 4 + ch] as u32
                  + src[add_row * pixel_stride + col * 4 + ch] as u32;
    }
}

/// Two-pass separable Gaussian blur. Kernel computed from `sigma`.
///
/// Uses rayon for parallel row processing in the horizontal pass.
/// `sigma = 0.0` is a no-op.
pub fn gaussian_blur(pixels: &mut [u8], width: u32, height: u32, sigma: f32) {
    if sigma <= 0.0 || width == 0 || height == 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let kernel = build_gaussian_kernel(sigma);

    // Pass 1: horizontal — operate row-by-row in parallel
    let mut temp: Vec<u8> = vec![0u8; w * h * 4];
    {
        let src_rows: Vec<&[u8]> = pixels.chunks_exact(w * 4).collect();
        let dst_rows: Vec<&mut [u8]> = temp.chunks_exact_mut(w * 4).collect();
        dst_rows.into_par_iter().zip(src_rows.into_par_iter()).for_each(|(dst_row, src_row)| {
            gaussian_convolve_h(src_row, dst_row, w, &kernel);
        });
    }

    // Pass 2: vertical — for each column, convolve along height
    for col in 0..w {
        for ch in 0..4usize {
            let column: Vec<f32> = (0..h)
                .map(|row| temp[row * w * 4 + col * 4 + ch] as f32)
                .collect();
            for row in 0..h {
                let val = gaussian_sample_1d(&column, row, &kernel);
                pixels[row * w * 4 + col * 4 + ch] = val.clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Build a 1D Gaussian kernel with given sigma. Half-width = ceil(3*sigma), max 50.
fn build_gaussian_kernel(sigma: f32) -> Vec<f32> {
    let half_width = ((3.0 * sigma).ceil() as usize).min(50);
    let size = 2 * half_width + 1;
    let mut kernel = Vec::with_capacity(size);
    let mut sum = 0.0f32;
    for i in 0..size {
        let x = i as f32 - half_width as f32;
        let w = (-0.5 * (x / sigma) * (x / sigma)).exp();
        kernel.push(w);
        sum += w;
    }
    if sum > 0.0 {
        for w in &mut kernel {
            *w /= sum;
        }
    }
    kernel
}

/// Apply 1D Gaussian kernel horizontally to a single row.
fn gaussian_convolve_h(src: &[u8], dst: &mut [u8], width: usize, kernel: &[f32]) {
    let half = kernel.len() / 2;
    for x in 0..width {
        for ch in 0..4usize {
            let mut acc = 0.0f32;
            for (ki, &kw) in kernel.iter().enumerate() {
                let sx = (x as isize + ki as isize - half as isize)
                    .clamp(0, (width as isize) - 1) as usize;
                acc += src[sx * 4 + ch] as f32 * kw;
            }
            dst[x * 4 + ch] = acc.clamp(0.0, 255.0) as u8;
        }
    }
}

/// Sample a 1D Gaussian-blurred value from a column signal.
fn gaussian_sample_1d(signal: &[f32], center: usize, kernel: &[f32]) -> f32 {
    let half = kernel.len() / 2;
    let len = signal.len();
    let mut acc = 0.0f32;
    for (ki, &kw) in kernel.iter().enumerate() {
        let idx = (center as isize + ki as isize - half as isize)
            .clamp(0, (len as isize) - 1) as usize;
        acc += signal[idx] * kw;
    }
    acc
}

/// Classic Unsharp Mask: `USM = original + amount * (original - blurred)`.
///
/// `threshold`: minimum absolute per-channel difference required before sharpening is applied.
/// `sigma`: standard deviation for the blur (0.0 = no-op).
/// `amount`: sharpening strength (typical range 0.5–2.0).
pub fn unsharp_mask(pixels: &mut [u8], width: u32, height: u32, sigma: f32, amount: f32, threshold: u8) {
    if sigma <= 0.0 || width == 0 || height == 0 {
        return;
    }
    let mut blurred = pixels.to_vec();
    gaussian_blur(&mut blurred, width, height, sigma);

    let n = pixels.len();
    for i in 0..n {
        let orig = pixels[i] as i32;
        let blur = blurred[i] as i32;
        let diff = orig - blur;
        if diff.unsigned_abs() as u8 > threshold {
            let sharpened = orig + (amount * diff as f32) as i32;
            pixels[i] = sharpened.clamp(0, 255) as u8;
        }
    }
}

/// Morphological dilation on the alpha channel only (channel index 3).
///
/// Expands bright alpha regions using a separable rectangular approximation:
/// horizontal max-filter then vertical max-filter. RGB channels are untouched.
/// `radius = 0` is a no-op.
pub fn dilate_alpha(pixels: &mut [u8], width: u32, height: u32, radius: u32) {
    if radius == 0 || width == 0 || height == 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let r = radius as usize;

    // Horizontal max pass on alpha into temp
    let mut temp = pixels.to_vec();
    for row in 0..h {
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            let mut max_val = 0u8;
            for k in lo..=hi {
                let v = pixels[row * w * 4 + k * 4 + 3];
                if v > max_val {
                    max_val = v;
                }
            }
            temp[row * w * 4 + x * 4 + 3] = max_val;
        }
    }

    // Vertical max pass on alpha from temp back to pixels
    for col in 0..w {
        for y in 0..h {
            let lo = y.saturating_sub(r);
            let hi = (y + r).min(h - 1);
            let mut max_val = 0u8;
            for k in lo..=hi {
                let v = temp[k * w * 4 + col * 4 + 3];
                if v > max_val {
                    max_val = v;
                }
            }
            pixels[y * w * 4 + col * 4 + 3] = max_val;
        }
    }
}

/// Morphological erosion on the alpha channel only (channel index 3).
///
/// Shrinks bright alpha regions using a separable rectangular approximation:
/// horizontal min-filter then vertical min-filter. RGB channels are untouched.
/// `radius = 0` is a no-op.
pub fn erode_alpha(pixels: &mut [u8], width: u32, height: u32, radius: u32) {
    if radius == 0 || width == 0 || height == 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let r = radius as usize;

    // Horizontal min pass on alpha into temp
    let mut temp = pixels.to_vec();
    for row in 0..h {
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            let mut min_val = 255u8;
            for k in lo..=hi {
                let v = pixels[row * w * 4 + k * 4 + 3];
                if v < min_val {
                    min_val = v;
                }
            }
            temp[row * w * 4 + x * 4 + 3] = min_val;
        }
    }

    // Vertical min pass on alpha from temp back to pixels
    for col in 0..w {
        for y in 0..h {
            let lo = y.saturating_sub(r);
            let hi = (y + r).min(h - 1);
            let mut min_val = 255u8;
            for k in lo..=hi {
                let v = temp[k * w * 4 + col * 4 + 3];
                if v < min_val {
                    min_val = v;
                }
            }
            pixels[y * w * 4 + col * 4 + 3] = min_val;
        }
    }
}

/// Fixed 3×3 emboss kernel applied to RGB channels. Alpha is unchanged.
///
/// Kernel:
/// ```text
/// [[-2, -1, 0],
///  [-1,  1, 1],
///  [ 0,  1, 2]]
/// ```
/// A 128-bias is added to center the output around mid-gray.
pub fn emboss(pixels: &mut [u8], width: u32, height: u32) {
    if width == 0 || height == 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    // Emboss kernel (row-major, 3x3)
    const KERNEL: [[i32; 3]; 3] = [
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2],
    ];

    let src = pixels.to_vec();
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3usize {
                let mut acc: i32 = 0;
                for ky in 0..3usize {
                    for kx in 0..3usize {
                        let sy = (y as isize + ky as isize - 1).clamp(0, (h as isize) - 1) as usize;
                        let sx = (x as isize + kx as isize - 1).clamp(0, (w as isize) - 1) as usize;
                        acc += KERNEL[ky][kx] * src[sy * w * 4 + sx * 4 + ch] as i32;
                    }
                }
                pixels[y * w * 4 + x * 4 + ch] = (acc + 128).clamp(0, 255) as u8;
            }
            // Alpha untouched — already in pixels from src
            // (we only modified RGB, alpha was not written above, and src was copied at start,
            //  so we need to explicitly preserve alpha from src)
            pixels[y * w * 4 + x * 4 + 3] = src[y * w * 4 + x * 4 + 3];
        }
    }
}

/// 3×3 median filter on all four channels independently.
///
/// Collects the 9-pixel neighbourhood for each channel, sorts them, and takes
/// the middle value (index 4 of 9). Border pixels use clamped (replicated) boundary.
pub fn median_filter_3x3(pixels: &mut [u8], width: u32, height: u32) {
    if width == 0 || height == 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let src = pixels.to_vec();

    for y in 0..h {
        for x in 0..w {
            for ch in 0..4usize {
                let mut neighbors = [0u8; 9];
                let mut idx = 0;
                for ky in 0..3usize {
                    for kx in 0..3usize {
                        let sy = (y as isize + ky as isize - 1).clamp(0, (h as isize) - 1) as usize;
                        let sx = (x as isize + kx as isize - 1).clamp(0, (w as isize) - 1) as usize;
                        neighbors[idx] = src[sy * w * 4 + sx * 4 + ch];
                        idx += 1;
                    }
                }
                // Sort network or simple sort for 9 elements
                neighbors.sort_unstable();
                pixels[y * w * 4 + x * 4 + ch] = neighbors[4];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a uniform RGBA8 buffer of given dimensions and color.
    fn uniform_buf(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let n = (w * h * 4) as usize;
        let mut buf = vec![0u8; n];
        for i in 0..((w * h) as usize) {
            buf[i * 4]     = r;
            buf[i * 4 + 1] = g;
            buf[i * 4 + 2] = b;
            buf[i * 4 + 3] = a;
        }
        buf
    }

    // ---- box_blur tests ----

    #[test]
    fn test_box_blur_radius_0_no_change() {
        let mut buf = uniform_buf(8, 8, 100, 150, 200, 255);
        let original = buf.clone();
        box_blur(&mut buf, 8, 8, 0);
        assert_eq!(buf, original, "radius=0 must leave buffer unchanged");
    }

    #[test]
    fn test_box_blur_uniform_no_change() {
        let mut buf = uniform_buf(16, 16, 128, 64, 32, 255);
        let original = buf.clone();
        box_blur(&mut buf, 16, 16, 3);
        // Uniform colors must survive blur (within rounding)
        for (i, (&a, &b)) in original.iter().zip(buf.iter()).enumerate() {
            assert!(
                (a as i32 - b as i32).abs() <= 1,
                "pixel {i}: expected ~{a} got {b}"
            );
        }
    }

    #[test]
    fn test_box_blur_single_pixel_spreads() {
        let w = 9u32;
        let h = 9u32;
        let mut buf = vec![0u8; (w * h * 4) as usize];
        // Place bright pixel at center
        let cx = (w / 2) as usize;
        let cy = (h / 2) as usize;
        let center_idx = cy * w as usize * 4 + cx * 4;
        buf[center_idx] = 255;
        buf[center_idx + 1] = 255;
        buf[center_idx + 2] = 255;
        buf[center_idx + 3] = 255;
        box_blur(&mut buf, w, h, 2);
        // After blur, the pixel adjacent to center should be non-zero
        let adj_idx = cy * w as usize * 4 + (cx + 1) * 4;
        assert!(buf[adj_idx] > 0, "Adjacent pixel should be non-zero after blur spread");
        // Far corner should remain dark
        assert_eq!(buf[0], 0, "Far corner should still be zero or very dim");
    }

    // ---- gaussian_blur tests ----

    #[test]
    fn test_gaussian_blur_sigma_0_no_change() {
        let mut buf = uniform_buf(8, 8, 100, 200, 50, 255);
        let original = buf.clone();
        gaussian_blur(&mut buf, 8, 8, 0.0);
        assert_eq!(buf, original, "sigma=0 must be a no-op");
    }

    #[test]
    fn test_gaussian_blur_uniform_no_change() {
        let mut buf = uniform_buf(16, 16, 77, 88, 99, 255);
        let original = buf.clone();
        gaussian_blur(&mut buf, 16, 16, 2.0);
        // Allow ±2 for boundary rounding artefacts
        for (i, (&a, &b)) in original.iter().zip(buf.iter()).enumerate() {
            assert!(
                (a as i32 - b as i32).abs() <= 2,
                "pixel {i}: expected ~{a} got {b}"
            );
        }
    }

    #[test]
    fn test_gaussian_blur_energy_conserved() {
        // Opaque image — total RGB energy should be roughly preserved
        let w = 32u32;
        let h = 32u32;
        let mut buf = vec![0u8; (w * h * 4) as usize];
        // Randomish pattern
        for i in 0..(w * h) as usize {
            buf[i * 4]     = ((i * 53 + 17) % 200) as u8;
            buf[i * 4 + 1] = ((i * 37 + 91) % 200) as u8;
            buf[i * 4 + 2] = ((i * 11 + 43) % 200) as u8;
            buf[i * 4 + 3] = 255;
        }
        let sum_before: u64 = buf.iter().copied().map(|v| v as u64).sum();
        gaussian_blur(&mut buf, w, h, 1.5);
        let sum_after: u64 = buf.iter().copied().map(|v| v as u64).sum();
        // Allow 2% deviation (boundary clamping causes slight variation)
        let tolerance = sum_before / 50;
        assert!(
            sum_after.abs_diff(sum_before) <= tolerance,
            "Energy deviation too large: before={sum_before}, after={sum_after}"
        );
    }

    #[test]
    fn test_gaussian_blur_symmetry() {
        // Blurring a horizontal line in a square image should yield a result
        // symmetric around the horizontal axis of the line.
        let w = 16u32;
        let h = 16u32;
        let mut buf = vec![0u8; (w * h * 4) as usize];
        // Draw a horizontal bright line at row h/2
        let mid_row = (h / 2) as usize;
        for x in 0..w as usize {
            buf[mid_row * w as usize * 4 + x * 4]     = 255;
            buf[mid_row * w as usize * 4 + x * 4 + 3] = 255;
        }
        gaussian_blur(&mut buf, w, h, 1.5);
        // Rows equidistant from the center line should have equal red intensity
        let above = mid_row - 2;
        let below = mid_row + 2;
        let col = (w / 2) as usize;
        let v_above = buf[above * w as usize * 4 + col * 4];
        let v_below = buf[below * w as usize * 4 + col * 4];
        assert!(
            (v_above as i32 - v_below as i32).abs() <= 2,
            "Blur symmetry violated: above={v_above}, below={v_below}"
        );
    }

    // ---- dilate_alpha tests ----

    #[test]
    fn test_dilate_alpha_expands() {
        let w = 9u32;
        let h = 9u32;
        let mut buf = vec![0u8; (w * h * 4) as usize];
        // Single alpha=255 pixel in center
        let cx = (w / 2) as usize;
        let cy = (h / 2) as usize;
        buf[cy * w as usize * 4 + cx * 4 + 3] = 255;
        dilate_alpha(&mut buf, w, h, 2);
        // Pixels within radius 2 should now have alpha=255
        let adjacent_idx = cy * w as usize * 4 + (cx + 1) * 4 + 3;
        assert_eq!(buf[adjacent_idx], 255, "Adjacent alpha should be dilated to 255");
        let far_idx = cy * w as usize * 4 + (cx + 3) * 4 + 3;
        assert_eq!(buf[far_idx], 0, "Pixels outside radius should remain 0");
    }

    // ---- erode_alpha tests ----

    #[test]
    fn test_erode_alpha_shrinks() {
        let w = 8u32;
        let h = 8u32;
        // Full alpha=255 buffer
        let mut buf = uniform_buf(w, h, 100, 100, 100, 255);
        erode_alpha(&mut buf, w, h, 1);
        // Corners should erode to 0 (they lose support from neighbors that would be out of bounds
        // only when the neighboring pixels are actually 0, but here all are 255 initially,
        // so border pixels that are now looking at "boundary" stay 255).
        // Better test: make a 4x4 center block inside an 8x8 zero frame.
        let mut buf2 = vec![0u8; (w * h * 4) as usize];
        for y in 2..6usize {
            for x in 2..6usize {
                buf2[y * w as usize * 4 + x * 4 + 3] = 255;
            }
        }
        erode_alpha(&mut buf2, w, h, 1);
        // Border of the block (row 2, row 5, col 2, col 5) should now be 0
        assert_eq!(buf2[2 * w as usize * 4 + 2 * 4 + 3], 0, "Eroded corner should be 0");
        // Interior (e.g. 3,3) should still be 255
        assert_eq!(buf2[3 * w as usize * 4 + 3 * 4 + 3], 255, "Interior should remain 255");
    }

    // ---- dilate then erode ----

    #[test]
    fn test_dilate_erode_commutative_uniform() {
        // Uniform full-alpha should survive dilate → erode
        let w = 16u32;
        let h = 16u32;
        let mut buf = uniform_buf(w, h, 50, 100, 150, 255);
        dilate_alpha(&mut buf, w, h, 2);
        erode_alpha(&mut buf, w, h, 2);
        // Alpha channels in interior should all still be 255
        for y in 3..(h - 3) as usize {
            for x in 3..(w - 3) as usize {
                let a = buf[y * w as usize * 4 + x * 4 + 3];
                assert_eq!(a, 255, "Interior alpha should be 255 after dilate+erode on uniform buffer at ({x},{y})");
            }
        }
    }

    // ---- emboss tests ----

    #[test]
    fn test_emboss_flat_gives_128() {
        // The emboss kernel [[-2,-1,0],[-1,1,1],[0,1,2]] has sum=1.
        // For a flat image at value 0, result = 0*1 + 128 bias = 128.
        let w = 8u32;
        let h = 8u32;
        let mut buf = uniform_buf(w, h, 0, 0, 0, 255);
        emboss(&mut buf, w, h);
        // Flat black image → all channels get 0*1 + 128 = 128
        for i in 0..(w * h) as usize {
            assert_eq!(buf[i * 4], 128, "Red channel should be 128 after emboss of uniform black");
            assert_eq!(buf[i * 4 + 1], 128, "Green channel should be 128");
            assert_eq!(buf[i * 4 + 2], 128, "Blue channel should be 128");
            assert_eq!(buf[i * 4 + 3], 255, "Alpha must be unchanged");
        }
    }

    // ---- median_filter_3x3 tests ----

    #[test]
    fn test_median_removes_single_outlier() {
        let w = 5u32;
        let h = 5u32;
        // All black except one bright white pixel in center
        let mut buf = vec![0u8; (w * h * 4) as usize];
        let cx = 2usize;
        let cy = 2usize;
        buf[cy * w as usize * 4 + cx * 4]     = 255;
        buf[cy * w as usize * 4 + cx * 4 + 1] = 255;
        buf[cy * w as usize * 4 + cx * 4 + 2] = 255;
        buf[cy * w as usize * 4 + cx * 4 + 3] = 255;
        median_filter_3x3(&mut buf, w, h);
        // The center outlier should have been replaced by the median (0)
        assert_eq!(buf[cy * w as usize * 4 + cx * 4], 0, "Single white pixel in black image should be removed by median");
    }

    // ---- gaussian_blur large sigma ----

    #[test]
    fn test_gaussian_blur_large_sigma() {
        let w = 64u32;
        let h = 64u32;
        let mut buf = uniform_buf(w, h, 100, 100, 100, 255);
        // Should not panic
        gaussian_blur(&mut buf, w, h, 20.0);
        // Result should still be roughly 100 everywhere
        for v in buf.chunks_exact(4).map(|p| p[0]) {
            assert!((v as i32 - 100).abs() <= 2, "Large sigma blur of uniform should stay uniform, got {v}");
        }
    }

    // ---- zero dimensions ----

    #[test]
    fn test_zero_dimensions_no_panic() {
        let mut buf = vec![0u8; 0];
        // All functions must handle zero dimensions gracefully
        box_blur(&mut buf, 0, 0, 3);
        gaussian_blur(&mut buf, 0, 0, 2.0);
        unsharp_mask(&mut buf, 0, 0, 1.5, 1.0, 10);
        dilate_alpha(&mut buf, 0, 0, 2);
        erode_alpha(&mut buf, 0, 0, 2);
        emboss(&mut buf, 0, 0);
        median_filter_3x3(&mut buf, 0, 0);
        // Width=0, height=5
        let mut buf2 = vec![0u8; 0];
        box_blur(&mut buf2, 0, 5, 1);
        // Height=0, width=5
        let mut buf3 = vec![0u8; 0];
        gaussian_blur(&mut buf3, 5, 0, 1.0);
    }

    // ---- unsharp_mask threshold ----

    #[test]
    fn test_unsharp_mask_threshold_blocks_small_diff() {
        let w = 16u32;
        let h = 16u32;
        // Slightly non-uniform image where diffs after blur will be < 255
        let mut buf = uniform_buf(w, h, 100, 100, 100, 255);
        // Add tiny variation
        buf[0] = 105;
        buf[4] = 95;
        let original = buf.clone();
        // Threshold=255 means no sharpening will ever be applied
        unsharp_mask(&mut buf, w, h, 1.0, 2.0, 255);
        assert_eq!(buf, original, "threshold=255 should block all sharpening");
    }
}
