//! `GpuAccelerator` trait and hardware acceleration abstraction.
//!
//! This module defines the unified [`GpuAccelerator`] trait that provides a
//! hardware-agnostic interface for GPU compute operations.  Concrete backends
//! (Vulkan/WGPU and CPU SIMD) implement the trait so callers never need to
//! branch on the active backend.
//!
//! # Design
//!
//! ```text
//! GpuAccelerator (trait)
//!    ├── WgpuAccelerator  ── uses wgpu (Vulkan / Metal / DX12 / WebGPU)
//!    └── CpuAccelerator   ── uses rayon SIMD fallback (always available)
//! ```
//!
//! # Quick Start
//!
//! ```no_run
//! use oximedia_gpu::accelerator::{AcceleratorBuilder, GpuAccelerator};
//!
//! let acc = AcceleratorBuilder::new().build()?;
//! let name = acc.name();
//! println!("Using backend: {name}");
//!
//! let rgb  = vec![0u8; 1920 * 1080 * 4];
//! let mut yuv = vec![0u8; 1920 * 1080 * 4];
//! acc.rgb_to_yuv(&rgb, &mut yuv, 1920, 1080)?;
//! ```

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]

use crate::{GpuError, Result};
use rayon::prelude::*;

// =============================================================================
// GpuAccelerator trait
// =============================================================================

/// Unified interface for hardware-accelerated media operations.
///
/// All operations have CPU fallback implementations so that callers do not
/// need to handle the absence of a GPU.  The trait is object-safe; you can
/// store it behind `Box<dyn GpuAccelerator>` or `Arc<dyn GpuAccelerator>`.
pub trait GpuAccelerator: Send + Sync {
    /// Human-readable backend name (e.g. `"Vulkan"`, `"CPU SIMD"`).
    fn name(&self) -> &str;

    /// Whether this accelerator uses dedicated GPU hardware.
    fn is_gpu(&self) -> bool;

    /// Convert packed RGBA data to packed YUVA using BT.601 coefficients.
    ///
    /// Both slices must have length `width * height * 4`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::InvalidBufferSize`] if slice lengths do not match
    /// `width * height * 4`.
    fn rgb_to_yuv(&self, input: &[u8], output: &mut [u8], width: u32, height: u32) -> Result<()>;

    /// Convert packed YUVA data to packed RGBA using BT.601 coefficients.
    ///
    /// Both slices must have length `width * height * 4`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::InvalidBufferSize`] if slice lengths do not match.
    fn yuv_to_rgb(&self, input: &[u8], output: &mut [u8], width: u32, height: u32) -> Result<()>;

    /// Resize packed RGBA image using bilinear interpolation.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are inconsistent with the given
    /// dimensions.
    #[allow(clippy::too_many_arguments)]
    fn scale_bilinear(
        &self,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        output: &mut [u8],
        dst_width: u32,
        dst_height: u32,
    ) -> Result<()>;

    /// Apply a separable Gaussian blur to a packed RGBA image.
    ///
    /// `sigma` is the standard deviation of the Gaussian kernel in pixels.
    ///
    /// # Errors
    ///
    /// Returns an error if `input.len() != output.len()` or if either length
    /// does not equal `width * height * 4`.
    fn gaussian_blur(
        &self,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        sigma: f32,
    ) -> Result<()>;

    /// Detect edges using the Sobel operator on a packed RGBA image.
    ///
    /// The output contains per-pixel gradient magnitudes.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are inconsistent.
    fn edge_detect(&self, input: &[u8], output: &mut [u8], width: u32, height: u32) -> Result<()>;

    /// Sharpen a packed RGBA image using an unsharp mask.
    ///
    /// `amount` controls the sharpening strength (typical range 0.0–2.0).
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are inconsistent.
    fn sharpen(
        &self,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        amount: f32,
    ) -> Result<()>;

    /// Compute the 2-D Type-II DCT on a grid of `f32` values.
    ///
    /// Both `width` and `height` must be multiples of 8.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are not multiples of 8 or if slice
    /// lengths do not equal `width * height`.
    fn dct_2d(&self, input: &[f32], output: &mut [f32], width: u32, height: u32) -> Result<()>;

    /// Compute the 2-D Type-III IDCT (inverse of [`dct_2d`]).
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`dct_2d`].
    ///
    /// [`dct_2d`]: GpuAccelerator::dct_2d
    fn idct_2d(&self, input: &[f32], output: &mut [f32], width: u32, height: u32) -> Result<()>;

    /// Compute the per-pixel absolute difference between two RGBA images.
    ///
    /// # Errors
    ///
    /// Returns an error if any buffer length differs from `width * height * 4`.
    fn pixel_diff(
        &self,
        a: &[u8],
        b: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<()>;

    /// Compute the mean squared error between two RGBA images.
    ///
    /// Returns the average squared per-channel difference (range 0.0–65 025.0).
    ///
    /// # Errors
    ///
    /// Returns an error if buffer lengths do not equal `width * height * 4`.
    fn mse(&self, a: &[u8], b: &[u8], width: u32, height: u32) -> Result<f64>;
}

// =============================================================================
// Buffer-size validation helpers
// =============================================================================

fn check_rgba_buf(buf: &[u8], width: u32, height: u32, label: &str) -> Result<()> {
    let expected = (width as usize) * (height as usize) * 4;
    if buf.len() != expected {
        return Err(GpuError::InvalidBufferSize {
            expected,
            actual: buf.len(),
        });
    }
    let _ = label;
    Ok(())
}

fn check_f32_buf(buf: &[f32], width: u32, height: u32) -> Result<()> {
    let expected = (width as usize) * (height as usize);
    if buf.len() != expected {
        return Err(GpuError::InvalidBufferSize {
            expected,
            actual: buf.len(),
        });
    }
    Ok(())
}

// =============================================================================
// CPU Accelerator
// =============================================================================

/// Pure-CPU accelerator backed by rayon parallel iterators.
///
/// This implementation is always available and provides correct (if slower)
/// results even when no GPU is present.
pub struct CpuAccelerator {
    num_threads: usize,
}

impl CpuAccelerator {
    /// Create a CPU accelerator using all available threads.
    #[must_use]
    pub fn new() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
        }
    }

    /// Number of worker threads.
    #[must_use]
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    // ---- Internal helpers --------------------------------------------------

    fn rgb_to_yuv_impl(input: &[u8], output: &mut [u8]) {
        const KR: f32 = 0.299;
        const KG: f32 = 0.587;
        const KB: f32 = 0.114;

        output
            .par_chunks_exact_mut(4)
            .zip(input.par_chunks_exact(4))
            .for_each(|(out, inp)| {
                let r = f32::from(inp[0]) / 255.0;
                let g = f32::from(inp[1]) / 255.0;
                let b = f32::from(inp[2]) / 255.0;

                let y = KR * r + KG * g + KB * b;
                let u = (b - y) / (2.0 * (1.0 - KB)) + 0.5;
                let v = (r - y) / (2.0 * (1.0 - KR)) + 0.5;

                out[0] = (y.clamp(0.0, 1.0) * 255.0) as u8;
                out[1] = (u.clamp(0.0, 1.0) * 255.0) as u8;
                out[2] = (v.clamp(0.0, 1.0) * 255.0) as u8;
                out[3] = inp[3];
            });
    }

    fn yuv_to_rgb_impl(input: &[u8], output: &mut [u8]) {
        const KR: f32 = 0.299;
        const KG: f32 = 0.587;
        const KB: f32 = 0.114;

        output
            .par_chunks_exact_mut(4)
            .zip(input.par_chunks_exact(4))
            .for_each(|(out, inp)| {
                let y = f32::from(inp[0]) / 255.0;
                let u = f32::from(inp[1]) / 255.0 - 0.5;
                let v = f32::from(inp[2]) / 255.0 - 0.5;

                let r = y + 2.0 * (1.0 - KR) * v;
                let b = y + 2.0 * (1.0 - KB) * u;
                let g = (y - KR * r - KB * b) / KG;

                out[0] = (r.clamp(0.0, 1.0) * 255.0) as u8;
                out[1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
                out[2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
                out[3] = inp[3];
            });
    }

    fn scale_bilinear_impl(
        input: &[u8],
        src_w: usize,
        src_h: usize,
        output: &mut [u8],
        dst_w: usize,
        dst_h: usize,
    ) {
        let x_ratio = src_w as f32 / dst_w as f32;
        let y_ratio = src_h as f32 / dst_h as f32;

        output
            .par_chunks_exact_mut(4)
            .enumerate()
            .for_each(|(idx, pixel)| {
                let dst_x = idx % dst_w;
                let dst_y = idx / dst_w;
                if dst_y >= dst_h {
                    return;
                }
                let src_x = (dst_x as f32 + 0.5) * x_ratio - 0.5;
                let src_y = (dst_y as f32 + 0.5) * y_ratio - 0.5;

                let x0 = (src_x.floor().max(0.0) as usize).min(src_w - 1);
                let y0 = (src_y.floor().max(0.0) as usize).min(src_h - 1);
                let x1 = (x0 + 1).min(src_w - 1);
                let y1 = (y0 + 1).min(src_h - 1);

                let fx = src_x.fract().max(0.0);
                let fy = src_y.fract().max(0.0);

                for c in 0..4 {
                    let p00 = f32::from(input[(y0 * src_w + x0) * 4 + c]);
                    let p10 = f32::from(input[(y0 * src_w + x1) * 4 + c]);
                    let p01 = f32::from(input[(y1 * src_w + x0) * 4 + c]);
                    let p11 = f32::from(input[(y1 * src_w + x1) * 4 + c]);

                    let top = p00 * (1.0 - fx) + p10 * fx;
                    let bot = p01 * (1.0 - fx) + p11 * fx;
                    pixel[c] = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
                }
            });
    }

    fn gaussian_blur_impl(
        input: &[u8],
        output: &mut [u8],
        width: usize,
        height: usize,
        sigma: f32,
    ) {
        let radius = (3.0 * sigma).ceil() as i32;
        let ksize = (2 * radius + 1) as usize;
        let two_sigma_sq = 2.0 * sigma * sigma;

        let mut kernel = vec![0.0f32; ksize];
        let mut sum = 0.0f32;
        for i in 0..ksize {
            let x = i as i32 - radius;
            let v = (-(x * x) as f32 / two_sigma_sq).exp();
            kernel[i] = v;
            sum += v;
        }
        for v in &mut kernel {
            *v /= sum;
        }

        // Horizontal pass → temp
        let mut temp = vec![0u8; input.len()];
        temp.par_chunks_exact_mut(4)
            .enumerate()
            .for_each(|(i, out)| {
                let px = i % width;
                let py = i / width;
                if py >= height {
                    return;
                }
                for c in 0..4 {
                    let mut acc = 0.0f32;
                    for (k, &kw) in kernel.iter().enumerate() {
                        let sx =
                            (px as i32 + k as i32 - radius).clamp(0, width as i32 - 1) as usize;
                        acc += f32::from(input[(py * width + sx) * 4 + c]) * kw;
                    }
                    out[c] = acc.round().clamp(0.0, 255.0) as u8;
                }
            });

        // Vertical pass → output
        output
            .par_chunks_exact_mut(4)
            .enumerate()
            .for_each(|(i, out)| {
                let px = i % width;
                let py = i / width;
                if py >= height {
                    return;
                }
                for c in 0..4 {
                    let mut acc = 0.0f32;
                    for (k, &kw) in kernel.iter().enumerate() {
                        let sy =
                            (py as i32 + k as i32 - radius).clamp(0, height as i32 - 1) as usize;
                        acc += f32::from(temp[(sy * width + px) * 4 + c]) * kw;
                    }
                    out[c] = acc.round().clamp(0.0, 255.0) as u8;
                }
            });
    }

    /// Apply a 3×3 Sobel gradient magnitude filter.
    fn sobel_impl(input: &[u8], output: &mut [u8], width: usize, height: usize) {
        // Convert to luminance first, then apply Sobel.
        let lum: Vec<f32> = input
            .par_chunks_exact(4)
            .map(|p| 0.299 * f32::from(p[0]) + 0.587 * f32::from(p[1]) + 0.114 * f32::from(p[2]))
            .collect();

        output
            .par_chunks_exact_mut(4)
            .enumerate()
            .for_each(|(i, out)| {
                let x = (i % width) as i32;
                let y = (i / width) as i32;

                if x == 0 || x == (width as i32 - 1) || y == 0 || y == (height as i32 - 1) {
                    out.fill(0);
                    return;
                }

                // Sobel kernels
                let gx = -lum[(y - 1) as usize * width + (x - 1) as usize]
                    - 2.0 * lum[y as usize * width + (x - 1) as usize]
                    - lum[(y + 1) as usize * width + (x - 1) as usize]
                    + lum[(y - 1) as usize * width + (x + 1) as usize]
                    + 2.0 * lum[y as usize * width + (x + 1) as usize]
                    + lum[(y + 1) as usize * width + (x + 1) as usize];

                let gy = -lum[(y - 1) as usize * width + (x - 1) as usize]
                    - 2.0 * lum[(y - 1) as usize * width + x as usize]
                    - lum[(y - 1) as usize * width + (x + 1) as usize]
                    + lum[(y + 1) as usize * width + (x - 1) as usize]
                    + 2.0 * lum[(y + 1) as usize * width + x as usize]
                    + lum[(y + 1) as usize * width + (x + 1) as usize];

                let mag = (gx * gx + gy * gy).sqrt().clamp(0.0, 255.0) as u8;
                out[0] = mag;
                out[1] = mag;
                out[2] = mag;
                out[3] = input[i * 4 + 3]; // preserve alpha
            });
    }

    fn sharpen_impl(input: &[u8], output: &mut [u8], width: usize, height: usize, amount: f32) {
        // Unsharp mask: output = input + amount * (input - blurred)
        let mut blurred = vec![0u8; input.len()];
        Self::gaussian_blur_impl(input, &mut blurred, width, height, 1.0);

        output
            .par_chunks_exact_mut(4)
            .zip(input.par_chunks_exact(4))
            .zip(blurred.par_chunks_exact(4))
            .for_each(|((out, orig), blur)| {
                for c in 0..3 {
                    let o = f32::from(orig[c]);
                    let b = f32::from(blur[c]);
                    let sharpened = o + amount * (o - b);
                    out[c] = sharpened.round().clamp(0.0, 255.0) as u8;
                }
                out[3] = orig[3];
            });
    }

    /// 1-D Type-II DCT on a slice of length N.
    fn dct1d(data: &[f32], out: &mut [f32]) {
        let n = data.len();
        let nf = n as f32;
        for k in 0..n {
            let mut s = 0.0f32;
            let kf = k as f32;
            for (j, &v) in data.iter().enumerate() {
                let angle = std::f32::consts::PI * kf * (2.0 * j as f32 + 1.0) / (2.0 * nf);
                s += v * angle.cos();
            }
            let scale = if k == 0 {
                (1.0 / nf).sqrt()
            } else {
                (2.0 / nf).sqrt()
            };
            out[k] = s * scale;
        }
    }

    /// 1-D Type-III IDCT (inverse of `dct1d`) on a slice of length N.
    fn idct1d(data: &[f32], out: &mut [f32]) {
        let n = data.len();
        let nf = n as f32;
        for j in 0..n {
            let jf = j as f32;
            let mut s = data[0] / nf.sqrt();
            for k in 1..n {
                let scale = (2.0 / nf).sqrt();
                let angle = std::f32::consts::PI * k as f32 * (2.0 * jf + 1.0) / (2.0 * nf);
                s += scale * data[k] * angle.cos();
            }
            out[j] = s;
        }
    }
}

impl Default for CpuAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuAccelerator for CpuAccelerator {
    fn name(&self) -> &'static str {
        "CPU SIMD"
    }

    fn is_gpu(&self) -> bool {
        false
    }

    fn rgb_to_yuv(&self, input: &[u8], output: &mut [u8], width: u32, height: u32) -> Result<()> {
        check_rgba_buf(input, width, height, "input")?;
        check_rgba_buf(output, width, height, "output")?;
        Self::rgb_to_yuv_impl(input, output);
        Ok(())
    }

    fn yuv_to_rgb(&self, input: &[u8], output: &mut [u8], width: u32, height: u32) -> Result<()> {
        check_rgba_buf(input, width, height, "input")?;
        check_rgba_buf(output, width, height, "output")?;
        Self::yuv_to_rgb_impl(input, output);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn scale_bilinear(
        &self,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        output: &mut [u8],
        dst_width: u32,
        dst_height: u32,
    ) -> Result<()> {
        check_rgba_buf(input, src_width, src_height, "input")?;
        check_rgba_buf(output, dst_width, dst_height, "output")?;
        Self::scale_bilinear_impl(
            input,
            src_width as usize,
            src_height as usize,
            output,
            dst_width as usize,
            dst_height as usize,
        );
        Ok(())
    }

    fn gaussian_blur(
        &self,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        sigma: f32,
    ) -> Result<()> {
        check_rgba_buf(input, width, height, "input")?;
        check_rgba_buf(output, width, height, "output")?;
        Self::gaussian_blur_impl(input, output, width as usize, height as usize, sigma);
        Ok(())
    }

    fn edge_detect(&self, input: &[u8], output: &mut [u8], width: u32, height: u32) -> Result<()> {
        check_rgba_buf(input, width, height, "input")?;
        check_rgba_buf(output, width, height, "output")?;
        Self::sobel_impl(input, output, width as usize, height as usize);
        Ok(())
    }

    fn sharpen(
        &self,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        amount: f32,
    ) -> Result<()> {
        check_rgba_buf(input, width, height, "input")?;
        check_rgba_buf(output, width, height, "output")?;
        Self::sharpen_impl(input, output, width as usize, height as usize, amount);
        Ok(())
    }

    fn dct_2d(&self, input: &[f32], output: &mut [f32], width: u32, height: u32) -> Result<()> {
        check_f32_buf(input, width, height)?;
        check_f32_buf(output, width, height)?;
        if width % 8 != 0 || height % 8 != 0 {
            return Err(GpuError::InvalidDimensions { width, height });
        }

        let w = width as usize;
        let h = height as usize;

        // Row-wise DCT
        let mut row_pass = vec![0.0f32; w * h];
        for row in 0..h {
            let src = &input[row * w..(row + 1) * w];
            let dst = &mut row_pass[row * w..(row + 1) * w];
            Self::dct1d(src, dst);
        }

        // Column-wise DCT
        for col in 0..w {
            let col_data: Vec<f32> = (0..h).map(|r| row_pass[r * w + col]).collect();
            let mut col_out = vec![0.0f32; h];
            Self::dct1d(&col_data, &mut col_out);
            for (r, &v) in col_out.iter().enumerate() {
                output[r * w + col] = v;
            }
        }
        Ok(())
    }

    fn idct_2d(&self, input: &[f32], output: &mut [f32], width: u32, height: u32) -> Result<()> {
        check_f32_buf(input, width, height)?;
        check_f32_buf(output, width, height)?;
        if width % 8 != 0 || height % 8 != 0 {
            return Err(GpuError::InvalidDimensions { width, height });
        }

        let w = width as usize;
        let h = height as usize;

        // Column-wise IDCT
        let mut col_pass = vec![0.0f32; w * h];
        for col in 0..w {
            let col_data: Vec<f32> = (0..h).map(|r| input[r * w + col]).collect();
            let mut col_out = vec![0.0f32; h];
            Self::idct1d(&col_data, &mut col_out);
            for (r, &v) in col_out.iter().enumerate() {
                col_pass[r * w + col] = v;
            }
        }

        // Row-wise IDCT
        for row in 0..h {
            let src = &col_pass[row * w..(row + 1) * w];
            let dst = &mut output[row * w..(row + 1) * w];
            Self::idct1d(src, dst);
        }
        Ok(())
    }

    fn pixel_diff(
        &self,
        a: &[u8],
        b: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        check_rgba_buf(a, width, height, "a")?;
        check_rgba_buf(b, width, height, "b")?;
        check_rgba_buf(output, width, height, "output")?;

        output
            .par_chunks_exact_mut(4)
            .zip(a.par_chunks_exact(4))
            .zip(b.par_chunks_exact(4))
            .for_each(|((out, pa), pb)| {
                for c in 0..4 {
                    out[c] = pa[c].abs_diff(pb[c]);
                }
            });
        Ok(())
    }

    fn mse(&self, a: &[u8], b: &[u8], width: u32, height: u32) -> Result<f64> {
        check_rgba_buf(a, width, height, "a")?;
        check_rgba_buf(b, width, height, "b")?;

        let sum_sq: f64 = a
            .par_chunks_exact(4)
            .zip(b.par_chunks_exact(4))
            .map(|(pa, pb)| {
                (0..4)
                    .map(|c| {
                        let d = f64::from(pa[c]) - f64::from(pb[c]);
                        d * d
                    })
                    .sum::<f64>()
            })
            .sum();

        let n = f64::from(width) * f64::from(height) * 4.0;
        Ok(sum_sq / n)
    }
}

// =============================================================================
// WGPU/GPU Accelerator  (delegates to CpuAccelerator where GPU is unavailable)
// =============================================================================

/// GPU-backed accelerator using wgpu (Vulkan / Metal / DX12 / WebGPU).
///
/// If no GPU is available the constructor fails; use [`AcceleratorBuilder`]
/// which transparently falls back to [`CpuAccelerator`].
///
/// All operations currently delegate to the CPU path while the wgpu compute
/// pipeline is being set up.  The struct is intentionally structured so that
/// individual operations can be migrated to GPU shaders without changing the
/// public interface.
pub struct WgpuAccelerator {
    device: std::sync::Arc<crate::GpuDevice>,
    /// CPU fallback for operations not yet ported to shaders.
    cpu: CpuAccelerator,
    backend_name: String,
}

impl WgpuAccelerator {
    /// Create a `WgpuAccelerator` with automatic device selection.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::NoAdapter`] if no GPU is available.
    pub fn new() -> Result<Self> {
        let device = crate::GpuDevice::new(None)?;
        let backend_name = format!("{} GPU", device.info().backend);
        Ok(Self {
            device: std::sync::Arc::new(device),
            cpu: CpuAccelerator::new(),
            backend_name,
        })
    }

    /// Underlying GPU device.
    #[must_use]
    pub fn gpu_device(&self) -> &std::sync::Arc<crate::GpuDevice> {
        &self.device
    }
}

impl GpuAccelerator for WgpuAccelerator {
    fn name(&self) -> &str {
        &self.backend_name
    }

    fn is_gpu(&self) -> bool {
        true
    }

    // The implementations below use the GPU device for simple operations
    // and fall back to the CPU for complex shaders not yet implemented.

    fn rgb_to_yuv(&self, input: &[u8], output: &mut [u8], width: u32, height: u32) -> Result<()> {
        crate::ops::ColorSpaceConversion::rgb_to_yuv(
            &self.device,
            input,
            output,
            width,
            height,
            crate::ops::ColorSpace::BT601,
        )
    }

    fn yuv_to_rgb(&self, input: &[u8], output: &mut [u8], width: u32, height: u32) -> Result<()> {
        crate::ops::ColorSpaceConversion::yuv_to_rgb(
            &self.device,
            input,
            output,
            width,
            height,
            crate::ops::ColorSpace::BT601,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn scale_bilinear(
        &self,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        output: &mut [u8],
        dst_width: u32,
        dst_height: u32,
    ) -> Result<()> {
        crate::ops::ScaleOperation::scale(
            &self.device,
            input,
            src_width,
            src_height,
            output,
            dst_width,
            dst_height,
            crate::ops::ScaleFilter::Bilinear,
        )
    }

    fn gaussian_blur(
        &self,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        sigma: f32,
    ) -> Result<()> {
        crate::ops::FilterOperation::gaussian_blur(
            &self.device,
            input,
            output,
            width,
            height,
            sigma,
        )
    }

    fn edge_detect(&self, input: &[u8], output: &mut [u8], width: u32, height: u32) -> Result<()> {
        crate::ops::FilterOperation::edge_detect(&self.device, input, output, width, height)
    }

    fn sharpen(
        &self,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        amount: f32,
    ) -> Result<()> {
        crate::ops::FilterOperation::sharpen(&self.device, input, output, width, height, amount)
    }

    fn dct_2d(&self, input: &[f32], output: &mut [f32], width: u32, height: u32) -> Result<()> {
        crate::ops::TransformOperation::dct_2d(&self.device, input, output, width, height)
    }

    fn idct_2d(&self, input: &[f32], output: &mut [f32], width: u32, height: u32) -> Result<()> {
        crate::ops::TransformOperation::idct_2d(&self.device, input, output, width, height)
    }

    fn pixel_diff(
        &self,
        a: &[u8],
        b: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        self.cpu.pixel_diff(a, b, output, width, height)
    }

    fn mse(&self, a: &[u8], b: &[u8], width: u32, height: u32) -> Result<f64> {
        self.cpu.mse(a, b, width, height)
    }
}

// =============================================================================
// AcceleratorBuilder
// =============================================================================

/// Ergonomic builder for creating a [`GpuAccelerator`].
///
/// Tries GPU first; falls back to CPU automatically.
///
/// # Example
///
/// ```no_run
/// use oximedia_gpu::accelerator::AcceleratorBuilder;
///
/// let acc = AcceleratorBuilder::new()
///     .prefer_gpu(true)
///     .build()
///     ?;
///
/// println!("Active backend: {}", acc.name());
/// ```
pub struct AcceleratorBuilder {
    prefer_gpu: bool,
    force_cpu: bool,
}

impl AcceleratorBuilder {
    /// Create a new builder with default settings (GPU preferred, CPU fallback).
    #[must_use]
    pub fn new() -> Self {
        Self {
            prefer_gpu: true,
            force_cpu: false,
        }
    }

    /// Set whether to prefer GPU (default: `true`).
    #[must_use]
    pub fn prefer_gpu(mut self, value: bool) -> Self {
        self.prefer_gpu = value;
        self
    }

    /// Force CPU-only mode even if a GPU is available.
    #[must_use]
    pub fn force_cpu(mut self, value: bool) -> Self {
        self.force_cpu = value;
        self
    }

    /// Build the accelerator.
    ///
    /// Returns `Ok(Box<dyn GpuAccelerator>)`.  Never fails because a CPU
    /// fallback is always available.
    ///
    /// # Errors
    ///
    /// This method never returns `Err` in practice (CPU fallback is always
    /// constructed), but the signature uses `Result` to allow future error
    /// propagation.
    pub fn build(self) -> Result<Box<dyn GpuAccelerator>> {
        if self.force_cpu || !self.prefer_gpu {
            return Ok(Box::new(CpuAccelerator::new()));
        }

        match WgpuAccelerator::new() {
            Ok(gpu) => Ok(Box::new(gpu)),
            Err(_) => Ok(Box::new(CpuAccelerator::new())),
        }
    }

    /// Build a CPU-only accelerator directly.
    #[must_use]
    pub fn build_cpu() -> CpuAccelerator {
        CpuAccelerator::new()
    }
}

impl Default for AcceleratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgba(w: usize, h: usize, fill: u8) -> Vec<u8> {
        vec![fill; w * h * 4]
    }

    // ---- CpuAccelerator ----------------------------------------------------

    #[test]
    fn test_cpu_accelerator_name() {
        let acc = CpuAccelerator::new();
        assert_eq!(acc.name(), "CPU SIMD");
        assert!(!acc.is_gpu());
    }

    #[test]
    fn test_cpu_rgb_to_yuv_roundtrip() {
        // A grey pixel (R=G=B) should survive a round-trip with ≤ 2 LSB error.
        let grey = 128u8;
        let input = vec![grey, grey, grey, 255u8];
        let mut yuv = vec![0u8; 4];
        let mut rgb = vec![0u8; 4];

        let acc = CpuAccelerator::new();
        acc.rgb_to_yuv(&input, &mut yuv, 1, 1)
            .expect("RGB to YUV conversion should succeed");
        acc.yuv_to_rgb(&yuv, &mut rgb, 1, 1)
            .expect("YUV to RGB conversion should succeed");

        // Grey channel should be ≈ 128
        assert!(
            (rgb[0] as i32 - grey as i32).abs() <= 3,
            "R mismatch: {}",
            rgb[0]
        );
        assert!(
            (rgb[1] as i32 - grey as i32).abs() <= 3,
            "G mismatch: {}",
            rgb[1]
        );
        assert!(
            (rgb[2] as i32 - grey as i32).abs() <= 3,
            "B mismatch: {}",
            rgb[2]
        );
    }

    #[test]
    fn test_cpu_rgb_to_yuv_invalid_size() {
        let acc = CpuAccelerator::new();
        let input = vec![0u8; 5]; // wrong size
        let mut output = vec![0u8; 4];
        assert!(acc.rgb_to_yuv(&input, &mut output, 1, 1).is_err());
    }

    #[test]
    fn test_cpu_scale_bilinear_identity() {
        // Scaling a uniform white image should produce a uniform white image.
        let w = 16usize;
        let h = 16usize;
        let input = make_rgba(w, h, 200);
        let mut output = make_rgba(w, h, 0);

        let acc = CpuAccelerator::new();
        acc.scale_bilinear(&input, w as u32, h as u32, &mut output, w as u32, h as u32)
            .expect("operation should succeed in test");

        // All output pixels should still be white.
        for &v in &output {
            assert!(v >= 195, "pixel value {v} too low after identity scale");
        }
    }

    #[test]
    fn test_cpu_scale_bilinear_upsample() {
        let input = make_rgba(2, 2, 255);
        let mut output = make_rgba(4, 4, 0);

        let acc = CpuAccelerator::new();
        acc.scale_bilinear(&input, 2, 2, &mut output, 4, 4)
            .expect("bilinear scaling should succeed");

        // All output pixels should be white (source was all-white).
        for &v in &output {
            assert!(v >= 250, "upsampled pixel {v} not white");
        }
    }

    #[test]
    fn test_cpu_gaussian_blur_preserves_size() {
        let input = make_rgba(8, 8, 128);
        let mut output = make_rgba(8, 8, 0);

        let acc = CpuAccelerator::new();
        acc.gaussian_blur(&input, &mut output, 8, 8, 1.0)
            .expect("gaussian blur should succeed");
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_cpu_edge_detect_flat_image() {
        // A flat-colour image has no edges → gradient magnitude ≈ 0
        // (border pixels are excluded).
        let input = make_rgba(16, 16, 200);
        let mut output = make_rgba(16, 16, 0);

        let acc = CpuAccelerator::new();
        acc.edge_detect(&input, &mut output, 16, 16)
            .expect("edge detection should succeed");

        // Interior pixels should be near zero.
        for row in 1..15usize {
            for col in 1..15usize {
                let idx = (row * 16 + col) * 4;
                assert!(
                    output[idx] < 10,
                    "interior edge pixel {} at ({row},{col}) is non-zero",
                    output[idx]
                );
            }
        }
    }

    #[test]
    fn test_cpu_sharpen_stable_flat() {
        // Sharpening a flat image should leave it unchanged.
        let input = make_rgba(8, 8, 128);
        let mut output = make_rgba(8, 8, 0);

        let acc = CpuAccelerator::new();
        acc.sharpen(&input, &mut output, 8, 8, 1.0)
            .expect("sharpen should succeed");

        // Allow ±2 LSB for accumulation of float rounding.
        for (&o, &i) in output.iter().zip(input.iter()) {
            assert!(
                (o as i32 - i as i32).abs() <= 3,
                "sharpen changed flat pixel by more than 3"
            );
        }
    }

    #[test]
    fn test_cpu_dct_idct_roundtrip() {
        let w = 8u32;
        let h = 8u32;
        let input: Vec<f32> = (0..(w * h)).map(|i| i as f32).collect();
        let mut dct_out = vec![0.0f32; (w * h) as usize];
        let mut rec = vec![0.0f32; (w * h) as usize];

        let acc = CpuAccelerator::new();
        acc.dct_2d(&input, &mut dct_out, w, h)
            .expect("DCT should succeed");
        acc.idct_2d(&dct_out, &mut rec, w, h)
            .expect("DCT should succeed");

        for (a, b) in input.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-3, "DCT round-trip error: {a} vs {b}");
        }
    }

    #[test]
    fn test_cpu_dct_invalid_dims() {
        let acc = CpuAccelerator::new();
        let input = vec![0.0f32; 10];
        let mut output = vec![0.0f32; 10];
        // 10 is not a multiple of 8 → error
        assert!(acc.dct_2d(&input, &mut output, 10, 1).is_err());
    }

    #[test]
    fn test_cpu_pixel_diff_self() {
        let img = make_rgba(4, 4, 100);
        let mut diff = make_rgba(4, 4, 255);

        let acc = CpuAccelerator::new();
        acc.pixel_diff(&img, &img, &mut diff, 4, 4)
            .expect("pixel diff should succeed");

        for &v in &diff {
            assert_eq!(v, 0, "self-diff should be zero");
        }
    }

    #[test]
    fn test_cpu_mse_identical() {
        let img = make_rgba(8, 8, 128);
        let acc = CpuAccelerator::new();
        let mse = acc
            .mse(&img, &img, 8, 8)
            .expect("MSE computation should succeed");
        assert!(
            mse.abs() < 1e-10,
            "MSE of identical images should be 0, got {mse}"
        );
    }

    #[test]
    fn test_cpu_mse_max_error() {
        // Black vs white → max MSE = 255^2 = 65025.
        let a = make_rgba(4, 4, 0);
        let b = make_rgba(4, 4, 255);
        let acc = CpuAccelerator::new();
        let mse = acc
            .mse(&a, &b, 4, 4)
            .expect("MSE computation should succeed");
        assert!(
            (mse - 65025.0).abs() < 1.0,
            "max MSE should be 65025, got {mse}"
        );
    }

    // ---- AcceleratorBuilder ------------------------------------------------

    #[test]
    fn test_builder_force_cpu() {
        let acc = AcceleratorBuilder::new()
            .force_cpu(true)
            .build()
            .expect("accelerator build should succeed");
        assert_eq!(acc.name(), "CPU SIMD");
        assert!(!acc.is_gpu());
    }

    #[test]
    fn test_builder_build_cpu_static() {
        let acc = AcceleratorBuilder::build_cpu();
        assert_eq!(acc.name(), "CPU SIMD");
    }

    #[test]
    #[ignore] // Requires GPU hardware probe; run with --ignored
    fn test_builder_default_builds() {
        // Should never panic even without a GPU.
        let acc = AcceleratorBuilder::new()
            .build()
            .expect("accelerator build should succeed");
        assert!(!acc.name().is_empty());
    }

    #[test]
    fn test_cpu_rgb_red_pixel() {
        // Red pixel → Y ≈ 0.299 * 255 ≈ 76
        let input = vec![255u8, 0, 0, 255];
        let mut yuv = vec![0u8; 4];
        let acc = CpuAccelerator::new();
        acc.rgb_to_yuv(&input, &mut yuv, 1, 1)
            .expect("RGB to YUV conversion should succeed");
        assert!(
            (yuv[0] as i32 - 76).abs() <= 2,
            "Y for red should be ~76, got {}",
            yuv[0]
        );
    }
}
