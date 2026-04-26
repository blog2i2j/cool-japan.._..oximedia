//! CPU fallback implementations for hardware acceleration operations.

use crate::error::{AccelError, AccelResult};
use crate::traits::{HardwareAccel, ScaleFilter};
use oximedia_core::PixelFormat;
use rayon::prelude::*;

/// CPU-based fallback acceleration backend.
///
/// This backend uses multi-threaded CPU implementations via rayon
/// when GPU acceleration is unavailable.
pub struct CpuAccel;

impl CpuAccel {
    /// Creates a new CPU acceleration backend.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuAccel {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareAccel for CpuAccel {
    fn scale_image(
        &self,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        format: PixelFormat,
        filter: ScaleFilter,
    ) -> AccelResult<Vec<u8>> {
        let channels = match format {
            PixelFormat::Rgb24 => 3,
            PixelFormat::Rgba32 => 4,
            PixelFormat::Gray8 => 1,
            _ => {
                return Err(AccelError::InvalidFormat(format!(
                    "Unsupported format: {format:?}"
                )))
            }
        };

        let expected_size = (src_width * src_height * channels) as usize;
        if input.len() != expected_size {
            return Err(AccelError::BufferSizeMismatch {
                expected: expected_size,
                actual: input.len(),
            });
        }

        let output_size = (dst_width * dst_height * channels) as usize;
        let mut output = vec![0u8; output_size];

        match filter {
            ScaleFilter::Nearest => {
                scale_nearest(
                    input,
                    &mut output,
                    src_width,
                    src_height,
                    dst_width,
                    dst_height,
                    channels,
                );
            }
            ScaleFilter::Bilinear => {
                scale_bilinear(
                    input,
                    &mut output,
                    src_width,
                    src_height,
                    dst_width,
                    dst_height,
                    channels,
                );
            }
            ScaleFilter::Bicubic => {
                scale_bicubic(
                    input,
                    &mut output,
                    src_width,
                    src_height,
                    dst_width,
                    dst_height,
                    channels,
                );
            }
            ScaleFilter::Lanczos => {
                scale_lanczos(
                    input,
                    &mut output,
                    src_width,
                    src_height,
                    dst_width,
                    dst_height,
                    channels,
                );
            }
        }

        Ok(output)
    }

    fn convert_color(
        &self,
        input: &[u8],
        width: u32,
        height: u32,
        src_format: PixelFormat,
        dst_format: PixelFormat,
    ) -> AccelResult<Vec<u8>> {
        match (src_format, dst_format) {
            // Identity pass-through
            (src, dst) if src == dst => Ok(input.to_vec()),

            // YUV 4:2:0 planar <-> RGB24
            (PixelFormat::Rgb24, PixelFormat::Yuv420p) => rgb_to_yuv420p(input, width, height),
            (PixelFormat::Yuv420p, PixelFormat::Rgb24) => yuv420p_to_rgb(input, width, height),

            // YUV 4:2:2 planar <-> RGB24
            (PixelFormat::Rgb24, PixelFormat::Yuv422p) => rgb_to_yuv422p(input, width, height),
            (PixelFormat::Yuv422p, PixelFormat::Rgb24) => yuv422p_to_rgb(input, width, height),

            // YUV 4:4:4 planar <-> RGB24
            (PixelFormat::Rgb24, PixelFormat::Yuv444p) => rgb_to_yuv444p(input, width, height),
            (PixelFormat::Yuv444p, PixelFormat::Rgb24) => yuv444p_to_rgb(input, width, height),

            // NV12 (Y + interleaved UV) <-> RGB24
            (PixelFormat::Rgb24, PixelFormat::Nv12) => rgb_to_nv12(input, width, height),
            (PixelFormat::Nv12, PixelFormat::Rgb24) => nv12_to_rgb(input, width, height),

            // NV21 (Y + interleaved VU) <-> RGB24
            (PixelFormat::Rgb24, PixelFormat::Nv21) => rgb_to_nv21(input, width, height),
            (PixelFormat::Nv21, PixelFormat::Rgb24) => nv21_to_rgb(input, width, height),

            // RGB24 <-> RGBA32 (add/drop alpha channel)
            (PixelFormat::Rgb24, PixelFormat::Rgba32) => rgb_to_rgba(input, width, height),
            (PixelFormat::Rgba32, PixelFormat::Rgb24) => rgba_to_rgb(input, width, height),

            // Gray8 <-> RGB24
            (PixelFormat::Rgb24, PixelFormat::Gray8) => rgb_to_gray8(input, width, height),
            (PixelFormat::Gray8, PixelFormat::Rgb24) => gray8_to_rgb(input, width, height),

            _ => Err(AccelError::Unsupported(format!(
                "Color conversion from {src_format:?} to {dst_format:?} not implemented"
            ))),
        }
    }

    fn motion_estimation(
        &self,
        reference: &[u8],
        current: &[u8],
        width: u32,
        height: u32,
        block_size: u32,
    ) -> AccelResult<Vec<(i16, i16)>> {
        let expected_size = (width * height) as usize;
        if reference.len() != expected_size || current.len() != expected_size {
            return Err(AccelError::BufferSizeMismatch {
                expected: expected_size,
                actual: reference.len().min(current.len()),
            });
        }

        Ok(block_motion_estimation(
            reference, current, width, height, block_size,
        ))
    }
}

/// Nearest neighbor scaling (CPU implementation).
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn scale_nearest(
    input: &[u8],
    output: &mut [u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
) {
    output
        .par_chunks_exact_mut((dst_width * channels) as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let src_y = (y as u32 * src_height) / dst_height;
            for x in 0..dst_width {
                let src_x = (x * src_width) / dst_width;
                let src_idx = ((src_y * src_width + src_x) * channels) as usize;
                let dst_idx = (x * channels) as usize;

                let channels_usize = channels as usize;
                row[dst_idx..dst_idx + channels_usize]
                    .copy_from_slice(&input[src_idx..src_idx + channels_usize]);
            }
        });
}

/// Bilinear scaling (CPU implementation).
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
fn scale_bilinear(
    input: &[u8],
    output: &mut [u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
) {
    let x_ratio = (src_width - 1) as f32 / dst_width as f32;
    let y_ratio = (src_height - 1) as f32 / dst_height as f32;

    output
        .par_chunks_exact_mut((dst_width * channels) as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let src_y = y as f32 * y_ratio;
            let y1 = src_y.floor() as u32;
            let y2 = (y1 + 1).min(src_height - 1);
            let y_frac = src_y - y1 as f32;

            for x in 0..dst_width {
                let src_x = x as f32 * x_ratio;
                let x1 = src_x.floor() as u32;
                let x2 = (x1 + 1).min(src_width - 1);
                let x_frac = src_x - x1 as f32;

                let dst_idx = (x * channels) as usize;

                for c in 0..channels as usize {
                    let p11 = f32::from(input[((y1 * src_width + x1) * channels) as usize + c]);
                    let p12 = f32::from(input[((y2 * src_width + x1) * channels) as usize + c]);
                    let p21 = f32::from(input[((y1 * src_width + x2) * channels) as usize + c]);
                    let p22 = f32::from(input[((y2 * src_width + x2) * channels) as usize + c]);

                    let p1 = p11 * (1.0 - x_frac) + p21 * x_frac;
                    let p2 = p12 * (1.0 - x_frac) + p22 * x_frac;
                    let result = p1 * (1.0 - y_frac) + p2 * y_frac;

                    row[dst_idx + c] = result.clamp(0.0, 255.0) as u8;
                }
            }
        });
}

/// Cubic interpolation kernel (Catmull-Rom): weight function for bicubic.
///
/// Uses the standard cubic convolution with a = -0.5 (Catmull-Rom spline).
#[inline]
fn cubic_weight(t: f32) -> f32 {
    let t_abs = t.abs();
    if t_abs <= 1.0 {
        (1.5 * t_abs - 2.5) * t_abs * t_abs + 1.0
    } else if t_abs <= 2.0 {
        ((-0.5 * t_abs + 2.5) * t_abs - 4.0) * t_abs + 2.0
    } else {
        0.0
    }
}

/// Bicubic scaling (CPU implementation) using Catmull-Rom spline.
fn scale_bicubic(
    input: &[u8],
    output: &mut [u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
) {
    let x_ratio = src_width as f32 / dst_width as f32;
    let y_ratio = src_height as f32 / dst_height as f32;

    output
        .par_chunks_exact_mut((dst_width * channels) as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let src_y = (y as f32 + 0.5) * y_ratio - 0.5;
            let y0 = src_y.floor() as i32;

            for x in 0..dst_width {
                let src_x = (x as f32 + 0.5) * x_ratio - 0.5;
                let x0 = src_x.floor() as i32;
                let dst_idx = (x * channels) as usize;

                for c in 0..channels as usize {
                    let mut sum = 0.0f32;
                    let mut weight_sum = 0.0f32;

                    for ky in -1i32..=2 {
                        let sy = (y0 + ky).clamp(0, src_height as i32 - 1) as u32;
                        let wy = cubic_weight(src_y - (y0 + ky) as f32);

                        for kx in -1i32..=2 {
                            let sx = (x0 + kx).clamp(0, src_width as i32 - 1) as u32;
                            let wx = cubic_weight(src_x - (x0 + kx) as f32);
                            let w = wx * wy;

                            let src_idx = ((sy * src_width + sx) * channels) as usize + c;
                            sum += f32::from(input[src_idx]) * w;
                            weight_sum += w;
                        }
                    }

                    let value = if weight_sum.abs() > 1e-6 {
                        sum / weight_sum
                    } else {
                        sum
                    };
                    row[dst_idx + c] = value.clamp(0.0, 255.0) as u8;
                }
            }
        });
}

/// Lanczos kernel function: sinc(x) * sinc(x/a) windowed.
///
/// Uses a = 3 (Lanczos-3) for high-quality resampling.
#[inline]
fn lanczos_kernel(x: f32, a: f32) -> f32 {
    if x.abs() < 1e-6 {
        return 1.0;
    }
    if x.abs() >= a {
        return 0.0;
    }
    let pi_x = std::f32::consts::PI * x;
    let pi_x_over_a = pi_x / a;
    (pi_x.sin() / pi_x) * (pi_x_over_a.sin() / pi_x_over_a)
}

/// Lanczos-3 scaling (CPU implementation).
///
/// Lanczos-3 uses a 6-tap (radius 3) kernel for the highest quality
/// resampling among standard filters.
fn scale_lanczos(
    input: &[u8],
    output: &mut [u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
) {
    let a = 3.0f32; // Lanczos-3
    let x_ratio = src_width as f32 / dst_width as f32;
    let y_ratio = src_height as f32 / dst_height as f32;

    output
        .par_chunks_exact_mut((dst_width * channels) as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let src_y = (y as f32 + 0.5) * y_ratio - 0.5;
            let y0 = src_y.floor() as i32;

            for x in 0..dst_width {
                let src_x = (x as f32 + 0.5) * x_ratio - 0.5;
                let x0 = src_x.floor() as i32;
                let dst_idx = (x * channels) as usize;

                for c in 0..channels as usize {
                    let mut sum = 0.0f32;
                    let mut weight_sum = 0.0f32;
                    let radius = a as i32;

                    for ky in (1 - radius)..=radius {
                        let sy = (y0 + ky).clamp(0, src_height as i32 - 1) as u32;
                        let wy = lanczos_kernel(src_y - (y0 + ky) as f32, a);

                        for kx in (1 - radius)..=radius {
                            let sx = (x0 + kx).clamp(0, src_width as i32 - 1) as u32;
                            let wx = lanczos_kernel(src_x - (x0 + kx) as f32, a);
                            let w = wx * wy;

                            let src_idx = ((sy * src_width + sx) * channels) as usize + c;
                            sum += f32::from(input[src_idx]) * w;
                            weight_sum += w;
                        }
                    }

                    let value = if weight_sum.abs() > 1e-6 {
                        sum / weight_sum
                    } else {
                        sum
                    };
                    row[dst_idx + c] = value.clamp(0.0, 255.0) as u8;
                }
            }
        });
}

// ─── BT.601 conversion helpers ────────────────────────────────────────────────
// Y  =  0.299 R + 0.587 G + 0.114 B
// Cb = -0.169 R - 0.331 G + 0.500 B  + 128
// Cr =  0.500 R - 0.419 G - 0.081 B  + 128
//
// Inverse:
// R = Y             + 1.402 Cr'
// G = Y - 0.344 Cb' - 0.714 Cr'
// B = Y + 1.772 Cb'
// where Cb' = Cb - 128, Cr' = Cr - 128

/// Compute BT.601 Y (luma) for a single pixel.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[inline]
fn bt601_y(r: f32, g: f32, b: f32) -> u8 {
    (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0) as u8
}

/// Compute BT.601 Cb (U) for a single pixel.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[inline]
fn bt601_cb(r: f32, g: f32, b: f32) -> u8 {
    (-0.169 * r - 0.331 * g + 0.500 * b + 128.0).clamp(0.0, 255.0) as u8
}

/// Compute BT.601 Cr (V) for a single pixel.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[inline]
fn bt601_cr(r: f32, g: f32, b: f32) -> u8 {
    (0.500 * r - 0.419 * g - 0.081 * b + 128.0).clamp(0.0, 255.0) as u8
}

/// Convert BT.601 Y + Cb + Cr to RGB.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[inline]
fn bt601_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y_f = f32::from(y);
    let cb_f = f32::from(cb) - 128.0;
    let cr_f = f32::from(cr) - 128.0;
    let r = (y_f + 1.402 * cr_f).clamp(0.0, 255.0) as u8;
    let g = (y_f - 0.344 * cb_f - 0.714 * cr_f).clamp(0.0, 255.0) as u8;
    let b = (y_f + 1.772 * cb_f).clamp(0.0, 255.0) as u8;
    (r, g, b)
}

// ─── RGB to YUV420p ───────────────────────────────────────────────────────────

/// RGB to `YUV420p` conversion (CPU implementation).
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
fn rgb_to_yuv420p(input: &[u8], width: u32, height: u32) -> AccelResult<Vec<u8>> {
    let expected_size = (width * height * 3) as usize;
    if input.len() != expected_size {
        return Err(AccelError::BufferSizeMismatch {
            expected: expected_size,
            actual: input.len(),
        });
    }

    let y_size = (width * height) as usize;
    let uv_size = (width * height / 4) as usize;
    let mut output = vec![0u8; y_size + uv_size * 2];

    let (y_plane, uv_planes) = output.split_at_mut(y_size);
    let (u_plane, v_plane) = uv_planes.split_at_mut(uv_size);

    // Convert Y plane
    y_plane.par_iter_mut().enumerate().for_each(|(i, y)| {
        let rgb_idx = i * 3;
        let r = f32::from(input[rgb_idx]);
        let g = f32::from(input[rgb_idx + 1]);
        let b = f32::from(input[rgb_idx + 2]);

        *y = bt601_y(r, g, b);
    });

    // Convert U and V planes (subsampled 2x2)
    u_plane
        .par_iter_mut()
        .zip(v_plane.par_iter_mut())
        .enumerate()
        .for_each(|(i, (u, v))| {
            let uv_x = (i as u32 % (width / 2)) * 2;
            let uv_y = (i as u32 / (width / 2)) * 2;
            let rgb_idx = ((uv_y * width + uv_x) * 3) as usize;

            let r = f32::from(input[rgb_idx]);
            let g = f32::from(input[rgb_idx + 1]);
            let b = f32::from(input[rgb_idx + 2]);

            *u = bt601_cb(r, g, b);
            *v = bt601_cr(r, g, b);
        });

    Ok(output)
}

/// `YUV420p` to RGB conversion (CPU implementation).
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
fn yuv420p_to_rgb(input: &[u8], width: u32, height: u32) -> AccelResult<Vec<u8>> {
    let y_size = (width * height) as usize;
    let uv_size = (width * height / 4) as usize;
    let expected_size = y_size + uv_size * 2;

    if input.len() != expected_size {
        return Err(AccelError::BufferSizeMismatch {
            expected: expected_size,
            actual: input.len(),
        });
    }

    let y_plane = &input[..y_size];
    let u_plane = &input[y_size..y_size + uv_size];
    let v_plane = &input[y_size + uv_size..];

    let output_size = (width * height * 3) as usize;
    let mut output = vec![0u8; output_size];

    output
        .par_chunks_exact_mut(3)
        .enumerate()
        .for_each(|(i, pixel)| {
            let pixel_x = i as u32 % width;
            let pixel_y = i as u32 / width;
            let uv_idx = ((pixel_y / 2) * (width / 2) + (pixel_x / 2)) as usize;

            let (red, green, blue) = bt601_to_rgb(y_plane[i], u_plane[uv_idx], v_plane[uv_idx]);
            pixel[0] = red;
            pixel[1] = green;
            pixel[2] = blue;
        });

    Ok(output)
}

// ─── YUV 4:2:2 planar ─────────────────────────────────────────────────────────

/// RGB24 to YUV422p conversion (BT.601). U/V planes are half width, full height.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
fn rgb_to_yuv422p(input: &[u8], width: u32, height: u32) -> AccelResult<Vec<u8>> {
    let expected_size = (width * height * 3) as usize;
    if input.len() != expected_size {
        return Err(AccelError::BufferSizeMismatch {
            expected: expected_size,
            actual: input.len(),
        });
    }

    let y_size = (width * height) as usize;
    let uv_size = (width / 2 * height) as usize; // half width, full height
    let mut output = vec![0u8; y_size + uv_size * 2];

    let (y_plane, uv_planes) = output.split_at_mut(y_size);
    let (u_plane, v_plane) = uv_planes.split_at_mut(uv_size);

    // Y plane: one sample per pixel
    y_plane.par_iter_mut().enumerate().for_each(|(i, y)| {
        let rgb_idx = i * 3;
        let r = f32::from(input[rgb_idx]);
        let g = f32::from(input[rgb_idx + 1]);
        let b = f32::from(input[rgb_idx + 2]);
        *y = bt601_y(r, g, b);
    });

    // U and V planes: sub-sampled 2x horizontally, not vertically
    u_plane
        .par_iter_mut()
        .zip(v_plane.par_iter_mut())
        .enumerate()
        .for_each(|(i, (u, v))| {
            let uv_x = (i as u32 % (width / 2)) * 2;
            let uv_y = i as u32 / (width / 2);
            let rgb_idx = ((uv_y * width + uv_x) * 3) as usize;
            let r = f32::from(input[rgb_idx]);
            let g = f32::from(input[rgb_idx + 1]);
            let b = f32::from(input[rgb_idx + 2]);
            *u = bt601_cb(r, g, b);
            *v = bt601_cr(r, g, b);
        });

    Ok(output)
}

/// YUV422p to RGB24 conversion (BT.601).
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
fn yuv422p_to_rgb(input: &[u8], width: u32, height: u32) -> AccelResult<Vec<u8>> {
    let y_size = (width * height) as usize;
    let uv_size = (width / 2 * height) as usize;
    let expected_size = y_size + uv_size * 2;

    if input.len() != expected_size {
        return Err(AccelError::BufferSizeMismatch {
            expected: expected_size,
            actual: input.len(),
        });
    }

    let y_plane = &input[..y_size];
    let u_plane = &input[y_size..y_size + uv_size];
    let v_plane = &input[y_size + uv_size..];

    let output_size = (width * height * 3) as usize;
    let mut output = vec![0u8; output_size];

    output
        .par_chunks_exact_mut(3)
        .enumerate()
        .for_each(|(i, pixel)| {
            let pixel_x = i as u32 % width;
            let pixel_y = i as u32 / width;
            let uv_idx = (pixel_y * (width / 2) + pixel_x / 2) as usize;

            let (r, g, b) = bt601_to_rgb(y_plane[i], u_plane[uv_idx], v_plane[uv_idx]);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        });

    Ok(output)
}

// ─── YUV 4:4:4 planar ─────────────────────────────────────────────────────────

/// RGB24 to YUV444p conversion (BT.601). All planes are full width x height.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
fn rgb_to_yuv444p(input: &[u8], width: u32, height: u32) -> AccelResult<Vec<u8>> {
    let expected_size = (width * height * 3) as usize;
    if input.len() != expected_size {
        return Err(AccelError::BufferSizeMismatch {
            expected: expected_size,
            actual: input.len(),
        });
    }

    let plane_size = (width * height) as usize;
    let mut output = vec![0u8; plane_size * 3];

    let (y_plane, uv_planes) = output.split_at_mut(plane_size);
    let (u_plane, v_plane) = uv_planes.split_at_mut(plane_size);

    y_plane
        .par_iter_mut()
        .zip(u_plane.par_iter_mut())
        .zip(v_plane.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((y, u), v))| {
            let rgb_idx = i * 3;
            let r = f32::from(input[rgb_idx]);
            let g = f32::from(input[rgb_idx + 1]);
            let b = f32::from(input[rgb_idx + 2]);
            *y = bt601_y(r, g, b);
            *u = bt601_cb(r, g, b);
            *v = bt601_cr(r, g, b);
        });

    Ok(output)
}

/// YUV444p to RGB24 conversion (BT.601).
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
fn yuv444p_to_rgb(input: &[u8], width: u32, height: u32) -> AccelResult<Vec<u8>> {
    let plane_size = (width * height) as usize;
    let expected_size = plane_size * 3;

    if input.len() != expected_size {
        return Err(AccelError::BufferSizeMismatch {
            expected: expected_size,
            actual: input.len(),
        });
    }

    let y_plane = &input[..plane_size];
    let u_plane = &input[plane_size..plane_size * 2];
    let v_plane = &input[plane_size * 2..];

    let output_size = plane_size * 3;
    let mut output = vec![0u8; output_size];

    output
        .par_chunks_exact_mut(3)
        .enumerate()
        .for_each(|(i, pixel)| {
            let (r, g, b) = bt601_to_rgb(y_plane[i], u_plane[i], v_plane[i]);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        });

    Ok(output)
}

// ─── NV12 (semi-planar: Y + interleaved UV) ───────────────────────────────────

/// RGB24 to NV12 conversion (BT.601). Y plane full, then interleaved UV at 4:2:0.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
fn rgb_to_nv12(input: &[u8], width: u32, height: u32) -> AccelResult<Vec<u8>> {
    let expected_size = (width * height * 3) as usize;
    if input.len() != expected_size {
        return Err(AccelError::BufferSizeMismatch {
            expected: expected_size,
            actual: input.len(),
        });
    }

    let y_size = (width * height) as usize;
    // UV plane: half width x half height, 2 bytes per sample (U, V interleaved)
    let uv_size = (width / 2 * (height / 2) * 2) as usize;
    let mut output = vec![0u8; y_size + uv_size];

    let (y_plane, uv_plane) = output.split_at_mut(y_size);

    // Y plane
    y_plane.par_iter_mut().enumerate().for_each(|(i, y)| {
        let rgb_idx = i * 3;
        let r = f32::from(input[rgb_idx]);
        let g = f32::from(input[rgb_idx + 1]);
        let b = f32::from(input[rgb_idx + 2]);
        *y = bt601_y(r, g, b);
    });

    // UV plane (2x2 sub-sampled, U then V interleaved)
    let uv_width = width / 2;
    uv_plane
        .par_chunks_exact_mut(2)
        .enumerate()
        .for_each(|(i, uv)| {
            let uv_x = (i as u32 % uv_width) * 2;
            let uv_y = (i as u32 / uv_width) * 2;
            let rgb_idx = ((uv_y * width + uv_x) * 3) as usize;
            let r = f32::from(input[rgb_idx]);
            let g = f32::from(input[rgb_idx + 1]);
            let b = f32::from(input[rgb_idx + 2]);
            uv[0] = bt601_cb(r, g, b); // U
            uv[1] = bt601_cr(r, g, b); // V
        });

    Ok(output)
}

/// NV12 to RGB24 conversion (BT.601).
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
fn nv12_to_rgb(input: &[u8], width: u32, height: u32) -> AccelResult<Vec<u8>> {
    let y_size = (width * height) as usize;
    let uv_size = (width / 2 * (height / 2) * 2) as usize;
    let expected_size = y_size + uv_size;

    if input.len() != expected_size {
        return Err(AccelError::BufferSizeMismatch {
            expected: expected_size,
            actual: input.len(),
        });
    }

    let y_plane = &input[..y_size];
    let uv_plane = &input[y_size..];

    let output_size = (width * height * 3) as usize;
    let mut output = vec![0u8; output_size];

    output
        .par_chunks_exact_mut(3)
        .enumerate()
        .for_each(|(i, pixel)| {
            let pixel_x = i as u32 % width;
            let pixel_y = i as u32 / width;
            // UV pair index: each UV pair covers a 2x2 block
            let uv_pair_idx = ((pixel_y / 2) * (width / 2) + pixel_x / 2) as usize;
            let cb = uv_plane[uv_pair_idx * 2];
            let cr = uv_plane[uv_pair_idx * 2 + 1];

            let (r, g, b) = bt601_to_rgb(y_plane[i], cb, cr);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        });

    Ok(output)
}

// ─── NV21 (semi-planar: Y + interleaved VU) ───────────────────────────────────

/// RGB24 to NV21 conversion (BT.601). Same layout as NV12 but V and U are swapped.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
fn rgb_to_nv21(input: &[u8], width: u32, height: u32) -> AccelResult<Vec<u8>> {
    let expected_size = (width * height * 3) as usize;
    if input.len() != expected_size {
        return Err(AccelError::BufferSizeMismatch {
            expected: expected_size,
            actual: input.len(),
        });
    }

    let y_size = (width * height) as usize;
    let uv_size = (width / 2 * (height / 2) * 2) as usize;
    let mut output = vec![0u8; y_size + uv_size];

    let (y_plane, vu_plane) = output.split_at_mut(y_size);

    // Y plane
    y_plane.par_iter_mut().enumerate().for_each(|(i, y)| {
        let rgb_idx = i * 3;
        let r = f32::from(input[rgb_idx]);
        let g = f32::from(input[rgb_idx + 1]);
        let b = f32::from(input[rgb_idx + 2]);
        *y = bt601_y(r, g, b);
    });

    // VU plane (V then U, opposite of NV12)
    let uv_width = width / 2;
    vu_plane
        .par_chunks_exact_mut(2)
        .enumerate()
        .for_each(|(i, vu)| {
            let uv_x = (i as u32 % uv_width) * 2;
            let uv_y = (i as u32 / uv_width) * 2;
            let rgb_idx = ((uv_y * width + uv_x) * 3) as usize;
            let r = f32::from(input[rgb_idx]);
            let g = f32::from(input[rgb_idx + 1]);
            let b = f32::from(input[rgb_idx + 2]);
            vu[0] = bt601_cr(r, g, b); // V first
            vu[1] = bt601_cb(r, g, b); // U second
        });

    Ok(output)
}

/// NV21 to RGB24 conversion (BT.601). V and U are swapped compared to NV12.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
fn nv21_to_rgb(input: &[u8], width: u32, height: u32) -> AccelResult<Vec<u8>> {
    let y_size = (width * height) as usize;
    let uv_size = (width / 2 * (height / 2) * 2) as usize;
    let expected_size = y_size + uv_size;

    if input.len() != expected_size {
        return Err(AccelError::BufferSizeMismatch {
            expected: expected_size,
            actual: input.len(),
        });
    }

    let y_plane = &input[..y_size];
    let vu_plane = &input[y_size..];

    let output_size = (width * height * 3) as usize;
    let mut output = vec![0u8; output_size];

    output
        .par_chunks_exact_mut(3)
        .enumerate()
        .for_each(|(i, pixel)| {
            let pixel_x = i as u32 % width;
            let pixel_y = i as u32 / width;
            let uv_pair_idx = ((pixel_y / 2) * (width / 2) + pixel_x / 2) as usize;
            // NV21: V first, U second
            let cr = vu_plane[uv_pair_idx * 2];
            let cb = vu_plane[uv_pair_idx * 2 + 1];

            let (r, g, b) = bt601_to_rgb(y_plane[i], cb, cr);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        });

    Ok(output)
}

// ─── RGB24 <-> RGBA32 ─────────────────────────────────────────────────────────

/// RGB24 to RGBA32 conversion: add opaque alpha (255).
fn rgb_to_rgba(input: &[u8], width: u32, height: u32) -> AccelResult<Vec<u8>> {
    let expected_size = (width * height * 3) as usize;
    if input.len() != expected_size {
        return Err(AccelError::BufferSizeMismatch {
            expected: expected_size,
            actual: input.len(),
        });
    }

    let output_size = (width * height * 4) as usize;
    let mut output = vec![0u8; output_size];

    output
        .par_chunks_exact_mut(4)
        .zip(input.par_chunks_exact(3))
        .for_each(|(rgba, rgb)| {
            rgba[0] = rgb[0];
            rgba[1] = rgb[1];
            rgba[2] = rgb[2];
            rgba[3] = 255; // fully opaque
        });

    Ok(output)
}

/// RGBA32 to RGB24 conversion: discard alpha channel.
fn rgba_to_rgb(input: &[u8], width: u32, height: u32) -> AccelResult<Vec<u8>> {
    let expected_size = (width * height * 4) as usize;
    if input.len() != expected_size {
        return Err(AccelError::BufferSizeMismatch {
            expected: expected_size,
            actual: input.len(),
        });
    }

    let output_size = (width * height * 3) as usize;
    let mut output = vec![0u8; output_size];

    output
        .par_chunks_exact_mut(3)
        .zip(input.par_chunks_exact(4))
        .for_each(|(rgb, rgba)| {
            rgb[0] = rgba[0];
            rgb[1] = rgba[1];
            rgb[2] = rgba[2];
        });

    Ok(output)
}

// ─── Gray8 <-> RGB24 ──────────────────────────────────────────────────────────

/// RGB24 to Gray8 conversion using BT.601 luma (0.299R + 0.587G + 0.114B).
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn rgb_to_gray8(input: &[u8], width: u32, height: u32) -> AccelResult<Vec<u8>> {
    let expected_size = (width * height * 3) as usize;
    if input.len() != expected_size {
        return Err(AccelError::BufferSizeMismatch {
            expected: expected_size,
            actual: input.len(),
        });
    }

    let output_size = (width * height) as usize;
    let mut output = vec![0u8; output_size];

    output
        .par_iter_mut()
        .zip(input.par_chunks_exact(3))
        .for_each(|(gray, rgb)| {
            let r = f32::from(rgb[0]);
            let g = f32::from(rgb[1]);
            let b = f32::from(rgb[2]);
            *gray = bt601_y(r, g, b);
        });

    Ok(output)
}

/// Gray8 to RGB24 conversion: replicate luma to all three channels.
fn gray8_to_rgb(input: &[u8], width: u32, height: u32) -> AccelResult<Vec<u8>> {
    let expected_size = (width * height) as usize;
    if input.len() != expected_size {
        return Err(AccelError::BufferSizeMismatch {
            expected: expected_size,
            actual: input.len(),
        });
    }

    let output_size = (width * height * 3) as usize;
    let mut output = vec![0u8; output_size];

    output
        .par_chunks_exact_mut(3)
        .zip(input.par_iter())
        .for_each(|(rgb, &gray)| {
            rgb[0] = gray;
            rgb[1] = gray;
            rgb[2] = gray;
        });

    Ok(output)
}

/// Block-based motion estimation (CPU implementation).
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_possible_wrap)]
fn block_motion_estimation(
    reference: &[u8],
    current: &[u8],
    width: u32,
    height: u32,
    block_size: u32,
) -> Vec<(i16, i16)> {
    let blocks_wide = width.div_ceil(block_size);
    let blocks_high = height.div_ceil(block_size);
    let search_range = 8i32;

    let motion_vectors: Vec<(i16, i16)> = (0..blocks_high)
        .into_par_iter()
        .flat_map(|block_y| {
            (0..blocks_wide).into_par_iter().map(move |block_x| {
                let cur_x = block_x * block_size;
                let cur_y = block_y * block_size;

                let mut best_sad = u32::MAX;
                let mut best_delta_x = 0i16;
                let mut best_delta_y = 0i16;

                for dy in -search_range..=search_range {
                    for dx in -search_range..=search_range {
                        let ref_x = cur_x as i32 + dx;
                        let ref_y = cur_y as i32 + dy;

                        if ref_x < 0
                            || ref_y < 0
                            || ref_x + block_size as i32 > width as i32
                            || ref_y + block_size as i32 > height as i32
                        {
                            continue;
                        }

                        let mut sad = 0u32;
                        for by in 0..block_size {
                            for bx in 0..block_size {
                                #[allow(clippy::cast_sign_loss)]
                                let rx = (ref_x + bx as i32) as u32;
                                #[allow(clippy::cast_sign_loss)]
                                let ry = (ref_y + by as i32) as u32;
                                let cx = cur_x + bx;
                                let cy = cur_y + by;

                                if rx >= width || ry >= height || cx >= width || cy >= height {
                                    continue;
                                }

                                let ref_idx = (ry * width + rx) as usize;
                                let cur_idx = (cy * width + cx) as usize;

                                sad += u32::from(reference[ref_idx].abs_diff(current[cur_idx]));
                            }
                        }

                        if sad < best_sad {
                            best_sad = sad;
                            best_delta_x = dx as i16;
                            best_delta_y = dy as i16;
                        }
                    }
                }

                (best_delta_x, best_delta_y)
            })
        })
        .collect();

    motion_vectors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::HardwareAccel;

    const W: u32 = 4;
    const H: u32 = 4;

    /// Build a synthetic RGB24 frame (W x H pixels).
    /// Pixel (x,y) = (x*60, y*40, 128).
    fn synthetic_rgb(w: u32, h: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                buf.push((x * 60).min(255) as u8);
                buf.push((y * 40).min(255) as u8);
                buf.push(128u8);
            }
        }
        buf
    }

    #[test]
    fn test_identity_passthrough() {
        let accel = CpuAccel::new();
        let rgb = synthetic_rgb(W, H);
        let out = accel
            .convert_color(&rgb, W, H, PixelFormat::Rgb24, PixelFormat::Rgb24)
            .expect("identity passthrough failed");
        assert_eq!(out, rgb);
    }

    #[test]
    fn test_rgb_to_rgba_roundtrip() {
        let accel = CpuAccel::new();
        let rgb = synthetic_rgb(W, H);

        let rgba = accel
            .convert_color(&rgb, W, H, PixelFormat::Rgb24, PixelFormat::Rgba32)
            .expect("rgb to rgba failed");
        assert_eq!(rgba.len(), (W * H * 4) as usize);
        // All alpha bytes must be 255 (opaque)
        for chunk in rgba.chunks_exact(4) {
            assert_eq!(chunk[3], 255, "alpha should be fully opaque");
        }

        let rgb2 = accel
            .convert_color(&rgba, W, H, PixelFormat::Rgba32, PixelFormat::Rgb24)
            .expect("rgba to rgb failed");
        assert_eq!(rgb2, rgb, "RGBA to RGB roundtrip failed");
    }

    #[test]
    fn test_rgb_rgba_size_mismatch_error() {
        let accel = CpuAccel::new();
        // Too short for W x H x 3
        let short = vec![0u8; 10];
        assert!(accel
            .convert_color(&short, W, H, PixelFormat::Rgb24, PixelFormat::Rgba32)
            .is_err());
    }

    #[test]
    fn test_gray8_to_rgb_roundtrip() {
        let accel = CpuAccel::new();
        let rgb = synthetic_rgb(W, H);

        let gray = accel
            .convert_color(&rgb, W, H, PixelFormat::Rgb24, PixelFormat::Gray8)
            .expect("rgb to gray8 failed");
        assert_eq!(gray.len(), (W * H) as usize);

        let rgb2 = accel
            .convert_color(&gray, W, H, PixelFormat::Gray8, PixelFormat::Rgb24)
            .expect("gray8 to rgb failed");
        assert_eq!(rgb2.len(), (W * H * 3) as usize);
        // All channels should be equal (replicated luma)
        for chunk in rgb2.chunks_exact(3) {
            assert_eq!(chunk[0], chunk[1]);
            assert_eq!(chunk[1], chunk[2]);
        }
    }

    #[test]
    fn test_gray8_size_mismatch_error() {
        let accel = CpuAccel::new();
        let short = vec![0u8; 5];
        assert!(accel
            .convert_color(&short, W, H, PixelFormat::Gray8, PixelFormat::Rgb24)
            .is_err());
    }

    #[test]
    fn test_yuv420p_roundtrip() {
        let accel = CpuAccel::new();
        let rgb = synthetic_rgb(W, H);

        let yuv = accel
            .convert_color(&rgb, W, H, PixelFormat::Rgb24, PixelFormat::Yuv420p)
            .expect("rgb to yuv420p failed");
        assert_eq!(yuv.len(), (W * H * 3 / 2) as usize);

        let rgb2 = accel
            .convert_color(&yuv, W, H, PixelFormat::Yuv420p, PixelFormat::Rgb24)
            .expect("yuv420p to rgb failed");
        assert_eq!(rgb2.len(), (W * H * 3) as usize);
    }

    #[test]
    fn test_yuv422p_roundtrip() {
        let accel = CpuAccel::new();
        let rgb = synthetic_rgb(W, H);

        let yuv = accel
            .convert_color(&rgb, W, H, PixelFormat::Rgb24, PixelFormat::Yuv422p)
            .expect("rgb to yuv422p failed");
        // Y: W*H, U: (W/2)*H, V: (W/2)*H
        assert_eq!(yuv.len(), (W * H + W / 2 * H * 2) as usize);

        let rgb2 = accel
            .convert_color(&yuv, W, H, PixelFormat::Yuv422p, PixelFormat::Rgb24)
            .expect("yuv422p to rgb failed");
        assert_eq!(rgb2.len(), (W * H * 3) as usize);
    }

    #[test]
    fn test_yuv422p_size_mismatch_error() {
        let accel = CpuAccel::new();
        let short = vec![0u8; 5];
        assert!(accel
            .convert_color(&short, W, H, PixelFormat::Rgb24, PixelFormat::Yuv422p)
            .is_err());
    }

    #[test]
    fn test_yuv444p_roundtrip() {
        let accel = CpuAccel::new();
        let rgb = synthetic_rgb(W, H);

        let yuv = accel
            .convert_color(&rgb, W, H, PixelFormat::Rgb24, PixelFormat::Yuv444p)
            .expect("rgb to yuv444p failed");
        // Y: W*H, U: W*H, V: W*H
        assert_eq!(yuv.len(), (W * H * 3) as usize);

        let rgb2 = accel
            .convert_color(&yuv, W, H, PixelFormat::Yuv444p, PixelFormat::Rgb24)
            .expect("yuv444p to rgb failed");
        assert_eq!(rgb2.len(), (W * H * 3) as usize);
    }

    #[test]
    fn test_yuv444p_size_mismatch_error() {
        let accel = CpuAccel::new();
        let short = vec![0u8; 5];
        assert!(accel
            .convert_color(&short, W, H, PixelFormat::Rgb24, PixelFormat::Yuv444p)
            .is_err());
    }

    #[test]
    fn test_nv12_roundtrip() {
        let accel = CpuAccel::new();
        let rgb = synthetic_rgb(W, H);

        let nv12 = accel
            .convert_color(&rgb, W, H, PixelFormat::Rgb24, PixelFormat::Nv12)
            .expect("rgb to nv12 failed");
        // Y: W*H, UV interleaved: (W/2)*(H/2)*2
        assert_eq!(nv12.len(), (W * H + W / 2 * (H / 2) * 2) as usize);

        let rgb2 = accel
            .convert_color(&nv12, W, H, PixelFormat::Nv12, PixelFormat::Rgb24)
            .expect("nv12 to rgb failed");
        assert_eq!(rgb2.len(), (W * H * 3) as usize);
    }

    #[test]
    fn test_nv12_size_mismatch_error() {
        let accel = CpuAccel::new();
        let short = vec![0u8; 5];
        assert!(accel
            .convert_color(&short, W, H, PixelFormat::Rgb24, PixelFormat::Nv12)
            .is_err());
    }

    #[test]
    fn test_nv21_roundtrip() {
        let accel = CpuAccel::new();
        let rgb = synthetic_rgb(W, H);

        let nv21 = accel
            .convert_color(&rgb, W, H, PixelFormat::Rgb24, PixelFormat::Nv21)
            .expect("rgb to nv21 failed");
        assert_eq!(nv21.len(), (W * H + W / 2 * (H / 2) * 2) as usize);

        let rgb2 = accel
            .convert_color(&nv21, W, H, PixelFormat::Nv21, PixelFormat::Rgb24)
            .expect("nv21 to rgb failed");
        assert_eq!(rgb2.len(), (W * H * 3) as usize);
    }

    #[test]
    fn test_nv21_size_mismatch_error() {
        let accel = CpuAccel::new();
        let short = vec![0u8; 5];
        assert!(accel
            .convert_color(&short, W, H, PixelFormat::Rgb24, PixelFormat::Nv21)
            .is_err());
    }

    #[test]
    fn test_nv12_nv21_differ_in_uv_order() {
        let accel = CpuAccel::new();
        // Use a color where U != V to distinguish NV12 vs NV21
        let rgb: Vec<u8> = vec![200u8, 50, 20].repeat((W * H) as usize);

        let nv12 = accel
            .convert_color(&rgb, W, H, PixelFormat::Rgb24, PixelFormat::Nv12)
            .expect("nv12 conversion failed");
        let nv21 = accel
            .convert_color(&rgb, W, H, PixelFormat::Rgb24, PixelFormat::Nv21)
            .expect("nv21 conversion failed");

        let y_size = (W * H) as usize;
        // Y planes should be identical
        assert_eq!(&nv12[..y_size], &nv21[..y_size], "Y planes must match");
        // UV/VU planes should differ (swapped order)
        let uv_nv12 = &nv12[y_size..];
        let vu_nv21 = &nv21[y_size..];
        // For a uniform image the two samples should be swapped
        if uv_nv12[0] != uv_nv12[1] {
            assert_eq!(
                uv_nv12[0], vu_nv21[1],
                "NV12 U should equal NV21 second byte"
            );
            assert_eq!(
                uv_nv12[1], vu_nv21[0],
                "NV12 V should equal NV21 first byte"
            );
        }
    }

    #[test]
    fn test_gray8_luma_correctness() {
        let accel = CpuAccel::new();
        // Pure white: all channels 255
        let white = vec![255u8; (W * H * 3) as usize];
        let gray = accel
            .convert_color(&white, W, H, PixelFormat::Rgb24, PixelFormat::Gray8)
            .expect("white gray failed");
        assert!(
            gray.iter().all(|&v| v >= 254),
            "pure white should yield luma ~255"
        );

        // Pure black
        let black = vec![0u8; (W * H * 3) as usize];
        let gray_black = accel
            .convert_color(&black, W, H, PixelFormat::Rgb24, PixelFormat::Gray8)
            .expect("black gray failed");
        assert!(
            gray_black.iter().all(|&v| v == 0),
            "pure black should yield luma 0"
        );
    }

    #[test]
    fn test_unsupported_conversion_returns_error() {
        let accel = CpuAccel::new();
        let yuv422 = vec![0u8; (W * H + W / 2 * H * 2) as usize];
        // Yuv422p to Yuv444p is not directly supported
        let result = accel.convert_color(&yuv422, W, H, PixelFormat::Yuv422p, PixelFormat::Yuv444p);
        assert!(result.is_err(), "unsupported pair should error");
    }
}
