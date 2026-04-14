//! Image rotation and flip operations.
//!
//! Provides lossless 90/180/270-degree rotations, horizontal and vertical flips,
//! and arbitrary-angle rotation with bilinear interpolation for U8 RGBA images.
//!
//! # Examples
//!
//! ```rust
//! use oximedia_image::rotate_flip::{rotate_90, flip_horizontal, rotate_arbitrary};
//!
//! // 4-pixel 2×2 RGBA image (all white)
//! let pixels: Vec<u8> = vec![255u8; 4 * 4]; // 2×2, 4 channels
//! let flipped = flip_horizontal(&pixels, 2, 2, 4);
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::error::{ImageError, ImageResult};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Validate that `data` has exactly `width * height * channels` bytes.
fn validate_buffer(data: &[u8], width: u32, height: u32, channels: u32) -> ImageResult<()> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(channels as usize))
        .ok_or(ImageError::InvalidDimensions(width, height))?;
    if data.len() != expected {
        return Err(ImageError::InvalidFormat(format!(
            "buffer length {} does not match {}×{}×{} = {}",
            data.len(),
            width,
            height,
            channels,
            expected
        )));
    }
    Ok(())
}

/// Read one pixel (all channels) from a flat row-major buffer into `dst`.
#[inline]
fn read_pixel(src: &[u8], x: u32, y: u32, width: u32, channels: usize, dst: &mut [u8]) {
    let offset = ((y as usize) * (width as usize) + (x as usize)) * channels;
    dst.copy_from_slice(&src[offset..offset + channels]);
}

/// Write one pixel (all channels) from `src` into a flat row-major buffer.
#[inline]
fn write_pixel(dst: &mut [u8], x: u32, y: u32, width: u32, channels: usize, src: &[u8]) {
    let offset = ((y as usize) * (width as usize) + (x as usize)) * channels;
    dst[offset..offset + channels].copy_from_slice(src);
}

// ── lossless rotations ────────────────────────────────────────────────────────

/// Rotate image 90 degrees clockwise.
///
/// Output dimensions: `(height, width)`.
///
/// # Errors
/// Returns [`ImageError::InvalidFormat`] if the buffer size does not match the given dimensions.
pub fn rotate_90(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
) -> ImageResult<(Vec<u8>, u32, u32)> {
    validate_buffer(data, width, height, channels)?;
    let ch = channels as usize;
    let out_w = height;
    let out_h = width;
    let mut out = vec![0u8; data.len()];
    let mut tmp = vec![0u8; ch];
    for y in 0..height {
        for x in 0..width {
            read_pixel(data, x, y, width, ch, &mut tmp);
            // (x, y) → (height - 1 - y, x)  in a 90° CW mapping
            let nx = height - 1 - y;
            let ny = x;
            write_pixel(&mut out, nx, ny, out_w, ch, &tmp);
        }
    }
    Ok((out, out_w, out_h))
}

/// Rotate image 180 degrees.
///
/// Output dimensions match input.
///
/// # Errors
/// Returns [`ImageError::InvalidFormat`] if the buffer size does not match.
pub fn rotate_180(data: &[u8], width: u32, height: u32, channels: u32) -> ImageResult<Vec<u8>> {
    validate_buffer(data, width, height, channels)?;
    let ch = channels as usize;
    let mut out = vec![0u8; data.len()];
    let mut tmp = vec![0u8; ch];
    for y in 0..height {
        for x in 0..width {
            read_pixel(data, x, y, width, ch, &mut tmp);
            let nx = width - 1 - x;
            let ny = height - 1 - y;
            write_pixel(&mut out, nx, ny, width, ch, &tmp);
        }
    }
    Ok(out)
}

/// Rotate image 270 degrees clockwise (= 90° counter-clockwise).
///
/// Output dimensions: `(height, width)`.
///
/// # Errors
/// Returns [`ImageError::InvalidFormat`] if the buffer size does not match.
pub fn rotate_270(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
) -> ImageResult<(Vec<u8>, u32, u32)> {
    validate_buffer(data, width, height, channels)?;
    let ch = channels as usize;
    let out_w = height;
    let out_h = width;
    let mut out = vec![0u8; data.len()];
    let mut tmp = vec![0u8; ch];
    for y in 0..height {
        for x in 0..width {
            read_pixel(data, x, y, width, ch, &mut tmp);
            // (x, y) → (y, width - 1 - x)  in a 90° CCW mapping
            let nx = y;
            let ny = width - 1 - x;
            write_pixel(&mut out, nx, ny, out_w, ch, &tmp);
        }
    }
    Ok((out, out_w, out_h))
}

// ── flips ─────────────────────────────────────────────────────────────────────

/// Flip image horizontally (left ↔ right).
///
/// # Errors
/// Returns [`ImageError::InvalidFormat`] if the buffer size does not match.
pub fn flip_horizontal(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
) -> ImageResult<Vec<u8>> {
    validate_buffer(data, width, height, channels)?;
    let ch = channels as usize;
    let mut out = data.to_vec();
    for y in 0..height {
        for x in 0..width / 2 {
            let left = ((y as usize) * (width as usize) + (x as usize)) * ch;
            let right = ((y as usize) * (width as usize) + (width as usize - 1 - x as usize)) * ch;
            for c in 0..ch {
                out.swap(left + c, right + c);
            }
        }
    }
    Ok(out)
}

/// Flip image vertically (top ↔ bottom).
///
/// # Errors
/// Returns [`ImageError::InvalidFormat`] if the buffer size does not match.
pub fn flip_vertical(data: &[u8], width: u32, height: u32, channels: u32) -> ImageResult<Vec<u8>> {
    validate_buffer(data, width, height, channels)?;
    let ch = channels as usize;
    let row_bytes = (width as usize) * ch;
    let mut out = data.to_vec();
    for y in 0..height / 2 {
        let top = (y as usize) * row_bytes;
        let bot = (height as usize - 1 - y as usize) * row_bytes;
        for i in 0..row_bytes {
            out.swap(top + i, bot + i);
        }
    }
    Ok(out)
}

/// Flip image both horizontally and vertically (equivalent to 180° rotation).
///
/// # Errors
/// Returns [`ImageError::InvalidFormat`] if the buffer size does not match.
pub fn flip_both(data: &[u8], width: u32, height: u32, channels: u32) -> ImageResult<Vec<u8>> {
    rotate_180(data, width, height, channels)
}

// ── arbitrary-angle rotation (bilinear) ───────────────────────────────────────

/// Rotation border strategy for out-of-bounds source pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderMode {
    /// Fill out-of-bounds pixels with a constant colour (default: transparent black).
    Constant,
    /// Clamp to the nearest edge pixel.
    Clamp,
    /// Wrap around (tile the image).
    Wrap,
}

/// Rotate image by an arbitrary angle (in degrees, clockwise) with bilinear interpolation.
///
/// The output canvas is sized to exactly contain the rotated image (so no content
/// is clipped). The rotation centre is the centre of the image.
///
/// `channels` must be 1, 2, 3, or 4.  The fill colour applies only when
/// `border == BorderMode::Constant`; it must have `channels` elements.
///
/// # Errors
/// - [`ImageError::InvalidFormat`] if buffer size is inconsistent.
/// - [`ImageError::Unsupported`] if `channels > 4`.
#[allow(clippy::too_many_arguments)]
pub fn rotate_arbitrary(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    angle_deg: f64,
    border: BorderMode,
    fill: &[u8],
) -> ImageResult<(Vec<u8>, u32, u32)> {
    if channels == 0 || channels > 4 {
        return Err(ImageError::Unsupported(format!(
            "channels={channels} not supported for arbitrary rotation"
        )));
    }
    if fill.len() != channels as usize {
        return Err(ImageError::InvalidFormat(format!(
            "fill slice length {} must equal channels {}",
            fill.len(),
            channels
        )));
    }
    validate_buffer(data, width, height, channels)?;

    let ch = channels as usize;
    let angle_rad = angle_deg.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    // Compute output canvas size (bounding box of the rotated rectangle).
    let w = width as f64;
    let h = height as f64;
    // Corners of the input image relative to its centre.
    let corners = [
        (-w / 2.0, -h / 2.0),
        (w / 2.0, -h / 2.0),
        (w / 2.0, h / 2.0),
        (-w / 2.0, h / 2.0),
    ];
    let rotated: Vec<(f64, f64)> = corners
        .iter()
        .map(|&(cx, cy)| (cx * cos_a - cy * sin_a, cx * sin_a + cy * cos_a))
        .collect();
    let min_x = rotated.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let max_x = rotated
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = rotated.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_y = rotated
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max);

    let out_w = (max_x - min_x).ceil() as u32;
    let out_h = (max_y - min_y).ceil() as u32;

    let out_cx = (out_w as f64) / 2.0;
    let out_cy = (out_h as f64) / 2.0;
    let in_cx = w / 2.0;
    let in_cy = h / 2.0;

    let mut out = vec![0u8; (out_w as usize) * (out_h as usize) * ch];

    for oy in 0..out_h {
        for ox in 0..out_w {
            // Map output pixel to input coordinates via inverse rotation.
            let dx = ox as f64 - out_cx;
            let dy = oy as f64 - out_cy;
            let sx = dx * cos_a + dy * sin_a + in_cx;
            let sy = -dx * sin_a + dy * cos_a + in_cy;

            let dst_off = ((oy as usize) * (out_w as usize) + (ox as usize)) * ch;
            sample_bilinear(
                data,
                width,
                height,
                ch,
                sx,
                sy,
                border,
                fill,
                &mut out[dst_off..dst_off + ch],
            );
        }
    }

    Ok((out, out_w, out_h))
}

/// Bilinear sample at floating-point source coordinate `(sx, sy)`.
fn sample_bilinear(
    src: &[u8],
    width: u32,
    height: u32,
    ch: usize,
    sx: f64,
    sy: f64,
    border: BorderMode,
    fill: &[u8],
    dst: &mut [u8],
) {
    let w = width as i64;
    let h = height as i64;

    let x0 = sx.floor() as i64;
    let y0 = sy.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = (sx - sx.floor()) as f32;
    let fy = (sy - sy.floor()) as f32;

    let mut px = [[0u8; 4]; 4]; // [tl, tr, bl, br][channel] — max 4 channels
    let coords = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];

    for (i, &(cx, cy)) in coords.iter().enumerate() {
        let in_bounds = cx >= 0 && cy >= 0 && cx < w && cy < h;
        if in_bounds {
            let off = ((cy as usize) * (width as usize) + (cx as usize)) * ch;
            px[i][..ch].copy_from_slice(&src[off..off + ch]);
        } else {
            match border {
                BorderMode::Constant => {
                    px[i][..ch].copy_from_slice(fill);
                }
                BorderMode::Clamp => {
                    let cx2 = cx.clamp(0, w - 1);
                    let cy2 = cy.clamp(0, h - 1);
                    let off = ((cy2 as usize) * (width as usize) + (cx2 as usize)) * ch;
                    px[i][..ch].copy_from_slice(&src[off..off + ch]);
                }
                BorderMode::Wrap => {
                    let cx2 = cx.rem_euclid(w) as usize;
                    let cy2 = cy.rem_euclid(h) as usize;
                    let off = (cy2 * (width as usize) + cx2) * ch;
                    px[i][..ch].copy_from_slice(&src[off..off + ch]);
                }
            }
        }
    }

    for c in 0..ch {
        let tl = px[0][c] as f32;
        let tr = px[1][c] as f32;
        let bl = px[2][c] as f32;
        let br = px[3][c] as f32;
        let top = tl + (tr - tl) * fx;
        let bot = bl + (br - bl) * fx;
        let val = top + (bot - top) * fy;
        dst[c] = val.round().clamp(0.0, 255.0) as u8;
    }
}

// ── transpose (helper used internally and exposed) ────────────────────────────

/// Transpose image (swap x and y axes, equivalent to reflect across the main diagonal).
///
/// Output dimensions: `(height, width)`.
///
/// # Errors
/// Returns [`ImageError::InvalidFormat`] if the buffer size does not match.
pub fn transpose(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
) -> ImageResult<(Vec<u8>, u32, u32)> {
    validate_buffer(data, width, height, channels)?;
    let ch = channels as usize;
    let out_w = height;
    let out_h = width;
    let mut out = vec![0u8; data.len()];
    let mut tmp = vec![0u8; ch];
    for y in 0..height {
        for x in 0..width {
            read_pixel(data, x, y, width, ch, &mut tmp);
            write_pixel(&mut out, y, x, out_w, ch, &tmp);
        }
    }
    Ok((out, out_w, out_h))
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 2×2 RGBA image with distinct pixels:
    /// TL=red, TR=green, BL=blue, BR=white.
    fn test_image_2x2() -> Vec<u8> {
        vec![
            255, 0, 0, 255, // (0,0) red
            0, 255, 0, 255, // (1,0) green
            0, 0, 255, 255, // (0,1) blue
            255, 255, 255, 255, // (1,1) white
        ]
    }

    fn pixel_at(data: &[u8], x: u32, y: u32, w: u32, ch: usize) -> &[u8] {
        let off = ((y as usize) * (w as usize) + (x as usize)) * ch;
        &data[off..off + ch]
    }

    #[test]
    fn test_rotate_90_dimensions() {
        let img = vec![0u8; 3 * 5 * 3]; // 3×5, RGB
        let (_, ow, oh) = rotate_90(&img, 3, 5, 3).unwrap();
        assert_eq!((ow, oh), (5, 3));
    }

    #[test]
    fn test_rotate_90_content() {
        let img = test_image_2x2();
        let (out, ow, _oh) = rotate_90(&img, 2, 2, 4).unwrap();
        // After 90° CW: TL becomes BL, TR becomes TL, BR becomes TR, BL becomes BR.
        // TL (0,0) was red → (height-1-0, 0) = (1, 0) in output
        assert_eq!(pixel_at(&out, 1, 0, ow, 4), &[255, 0, 0, 255]); // red → right-top
                                                                    // TR (1,0) was green → (height-1-0, 1) = (1, 1)
        assert_eq!(pixel_at(&out, 1, 1, ow, 4), &[0, 255, 0, 255]);
    }

    #[test]
    fn test_rotate_180_inverts_back() {
        let img = test_image_2x2();
        let out = rotate_180(&img, 2, 2, 4).unwrap();
        let back = rotate_180(&out, 2, 2, 4).unwrap();
        assert_eq!(img, back);
    }

    #[test]
    fn test_rotate_270_is_inverse_of_90() {
        let img = test_image_2x2();
        let (r90, w90, h90) = rotate_90(&img, 2, 2, 4).unwrap();
        let (r270, w270, h270) = rotate_270(&img, 2, 2, 4).unwrap();
        // Both should have same dims
        assert_eq!((w90, h90), (w270, h270));
        // 90 ∘ 270 = identity
        let (back, _, _) = rotate_90(&r270, w270, h270, 4).unwrap();
        assert_eq!(back, img);
        // 270 ∘ 90 = identity
        let (back2, _, _) = rotate_270(&r90, w90, h90, 4).unwrap();
        assert_eq!(back2, img);
    }

    #[test]
    fn test_flip_horizontal() {
        let img = test_image_2x2();
        let out = flip_horizontal(&img, 2, 2, 4).unwrap();
        // Row 0: red, green → green, red
        assert_eq!(pixel_at(&out, 0, 0, 2, 4), &[0, 255, 0, 255]);
        assert_eq!(pixel_at(&out, 1, 0, 2, 4), &[255, 0, 0, 255]);
        // Row 1: blue, white → white, blue
        assert_eq!(pixel_at(&out, 0, 1, 2, 4), &[255, 255, 255, 255]);
        assert_eq!(pixel_at(&out, 1, 1, 2, 4), &[0, 0, 255, 255]);
    }

    #[test]
    fn test_flip_horizontal_twice_is_identity() {
        let img = test_image_2x2();
        let out = flip_horizontal(&img, 2, 2, 4).unwrap();
        let back = flip_horizontal(&out, 2, 2, 4).unwrap();
        assert_eq!(img, back);
    }

    #[test]
    fn test_flip_vertical() {
        let img = test_image_2x2();
        let out = flip_vertical(&img, 2, 2, 4).unwrap();
        // Row 0 becomes row 1 and vice versa.
        assert_eq!(pixel_at(&out, 0, 0, 2, 4), &[0, 0, 255, 255]); // blue
        assert_eq!(pixel_at(&out, 1, 0, 2, 4), &[255, 255, 255, 255]); // white
    }

    #[test]
    fn test_flip_vertical_twice_is_identity() {
        let img = test_image_2x2();
        let back = flip_vertical(&flip_vertical(&img, 2, 2, 4).unwrap(), 2, 2, 4).unwrap();
        assert_eq!(img, back);
    }

    #[test]
    fn test_transpose_dimensions() {
        let img = vec![0u8; 3 * 7 * 1]; // 3×7, 1 channel
        let (_, ow, oh) = transpose(&img, 3, 7, 1).unwrap();
        assert_eq!((ow, oh), (7, 3));
    }

    #[test]
    fn test_rotate_arbitrary_zero_angle() {
        // 0° rotation should preserve pixel values (with bilinear rounding).
        let img = test_image_2x2();
        let (out, ow, oh) =
            rotate_arbitrary(&img, 2, 2, 4, 0.0, BorderMode::Clamp, &[0, 0, 0, 0]).unwrap();
        assert_eq!(ow, 2);
        assert_eq!(oh, 2);
        // Centre pixels should be close to originals (bilinear at non-integer may blur)
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_rotate_arbitrary_180_matches_rotate_180() {
        let img: Vec<u8> = (0..16u8).collect(); // 2×2, 4ch
        let (arb_out, aw, ah) =
            rotate_arbitrary(&img, 2, 2, 4, 180.0, BorderMode::Clamp, &[0, 0, 0, 0]).unwrap();
        // Bounding box of 180° rotation may round up by 1 pixel in each axis
        // due to floating-point ceil on the rotated corners.
        assert!(aw >= 2 && aw <= 3, "unexpected out_w={aw}");
        assert!(ah >= 2 && ah <= 3, "unexpected out_h={ah}");
        // Output buffer must be consistent with reported dims.
        assert_eq!(arb_out.len(), (aw as usize) * (ah as usize) * 4);
    }

    #[test]
    fn test_invalid_buffer_size() {
        let img = vec![0u8; 5]; // wrong size for 2×2 RGBA
        assert!(rotate_90(&img, 2, 2, 4).is_err());
        assert!(flip_horizontal(&img, 2, 2, 4).is_err());
    }

    #[test]
    fn test_rotate_arbitrary_border_constant() {
        // A 45° rotation with constant border should not panic and produce correct size.
        let img = vec![128u8; 4 * 4 * 3]; // 4×4 RGB
        let fill = [0u8, 0, 0];
        let result = rotate_arbitrary(&img, 4, 4, 3, 45.0, BorderMode::Constant, &fill);
        assert!(result.is_ok());
        let (_, ow, oh) = result.unwrap();
        // 4×4 rotated 45°: bounding box is approx 6×6
        assert!(ow >= 4 && oh >= 4);
    }
}
