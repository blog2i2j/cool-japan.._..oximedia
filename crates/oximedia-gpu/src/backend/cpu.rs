//! CPU SIMD fallback backend
//!
//! This module provides a CPU-based fallback implementation using SIMD
//! when GPU compute is not available or for small workloads where CPU
//! execution might be faster due to overhead.

use super::{Backend, BackendCapabilities, BackendType};
use crate::Result;
use rayon::prelude::*;

/// CPU backend using SIMD and multi-threading
pub struct CpuBackend {
    capabilities: BackendCapabilities,
    num_threads: usize,
}

impl CpuBackend {
    /// Create a new CPU backend
    pub fn new() -> Result<Self> {
        let num_threads = rayon::current_num_threads();

        let capabilities = BackendCapabilities {
            backend_type: BackendType::CPU,
            max_workgroup_size: (1, 1, 1), // CPU doesn't use workgroups
            max_workgroup_invocations: 1,
            max_buffer_size: usize::MAX as u64,
            compute_shaders: false,
            subgroups: false,
            push_constants: false,
        };

        Ok(Self {
            capabilities,
            num_threads,
        })
    }

    /// Get the number of CPU threads
    #[must_use]
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// RGB to YUV conversion (BT.601) using CPU SIMD
    pub fn rgb_to_yuv_bt601(input: &[u8], output: &mut [u8], width: usize, height: usize) {
        const KR: f32 = 0.299;
        const KB: f32 = 0.114;
        const KG: f32 = 0.587;

        let pixels = width * height;
        output
            .par_chunks_exact_mut(4)
            .zip(input.par_chunks_exact(4))
            .take(pixels)
            .for_each(|(out, inp)| {
                let r = f32::from(inp[0]) / 255.0;
                let g = f32::from(inp[1]) / 255.0;
                let b = f32::from(inp[2]) / 255.0;
                let a = inp[3];

                let y = KR * r + KG * g + KB * b;
                let u = (b - y) / (2.0 * (1.0 - KB)) + 0.5;
                let v = (r - y) / (2.0 * (1.0 - KR)) + 0.5;

                out[0] = (y.clamp(0.0, 1.0) * 255.0) as u8;
                out[1] = (u.clamp(0.0, 1.0) * 255.0) as u8;
                out[2] = (v.clamp(0.0, 1.0) * 255.0) as u8;
                out[3] = a;
            });
    }

    /// YUV to RGB conversion (BT.601) using CPU SIMD
    pub fn yuv_to_rgb_bt601(input: &[u8], output: &mut [u8], width: usize, height: usize) {
        const KR: f32 = 0.299;
        const KB: f32 = 0.114;
        const KG: f32 = 0.587;

        let pixels = width * height;
        output
            .par_chunks_exact_mut(4)
            .zip(input.par_chunks_exact(4))
            .take(pixels)
            .for_each(|(out, inp)| {
                let y = f32::from(inp[0]) / 255.0;
                let u = f32::from(inp[1]) / 255.0 - 0.5;
                let v = f32::from(inp[2]) / 255.0 - 0.5;
                let a = inp[3];

                let r = y + 2.0 * (1.0 - KR) * v;
                let b = y + 2.0 * (1.0 - KB) * u;
                let g = (y - KR * r - KB * b) / KG;

                out[0] = (r.clamp(0.0, 1.0) * 255.0) as u8;
                out[1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
                out[2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
                out[3] = a;
            });
    }

    /// Bilinear image resize using CPU
    #[allow(clippy::too_many_arguments)]
    pub fn resize_bilinear(
        input: &[u8],
        src_width: usize,
        src_height: usize,
        output: &mut [u8],
        dst_width: usize,
        dst_height: usize,
    ) {
        let x_ratio = src_width as f32 / dst_width as f32;
        let y_ratio = src_height as f32 / dst_height as f32;

        output
            .par_chunks_exact_mut(4)
            .enumerate()
            .for_each(|(i, pixel)| {
                let dst_x = i % dst_width;
                let dst_y = i / dst_width;

                if dst_y >= dst_height {
                    return;
                }

                let src_x = (dst_x as f32 + 0.5) * x_ratio - 0.5;
                let src_y = (dst_y as f32 + 0.5) * y_ratio - 0.5;

                let x0 = src_x.floor().max(0.0) as usize;
                let y0 = src_y.floor().max(0.0) as usize;
                let x1 = (x0 + 1).min(src_width - 1);
                let y1 = (y0 + 1).min(src_height - 1);

                let fx = src_x.fract();
                let fy = src_y.fract();

                for c in 0..4 {
                    let p00 = input[(y0 * src_width + x0) * 4 + c];
                    let p10 = input[(y0 * src_width + x1) * 4 + c];
                    let p01 = input[(y1 * src_width + x0) * 4 + c];
                    let p11 = input[(y1 * src_width + x1) * 4 + c];

                    let v0 = f32::from(p00) * (1.0 - fx) + f32::from(p10) * fx;
                    let v1 = f32::from(p01) * (1.0 - fx) + f32::from(p11) * fx;
                    let v = v0 * (1.0 - fy) + v1 * fy;

                    pixel[c] = v.round().clamp(0.0, 255.0) as u8;
                }
            });
    }

    /// Gaussian blur using CPU
    pub fn gaussian_blur(input: &[u8], output: &mut [u8], width: usize, height: usize, sigma: f32) {
        let kernel_radius = (3.0 * sigma).ceil() as i32;
        let kernel_size = (2 * kernel_radius + 1) as usize;

        // Generate Gaussian kernel
        let mut kernel = vec![0.0f32; kernel_size];
        let mut sum = 0.0f32;
        let two_sigma_sq = 2.0 * sigma * sigma;

        for i in 0..kernel_size {
            let x = i as i32 - kernel_radius;
            let value = (-(x * x) as f32 / two_sigma_sq).exp();
            kernel[i] = value;
            sum += value;
        }

        // Normalize kernel
        for value in &mut kernel {
            *value /= sum;
        }

        // Temporary buffer for horizontal pass
        let mut temp = vec![0u8; input.len()];

        // Horizontal pass
        temp.par_chunks_exact_mut(4)
            .enumerate()
            .for_each(|(i, pixel)| {
                let x = i % width;
                let y = i / width;

                if y >= height {
                    return;
                }

                for c in 0..4 {
                    let mut value = 0.0f32;

                    for k in 0..kernel_size {
                        let offset = k as i32 - kernel_radius;
                        let sample_x = (x as i32 + offset).clamp(0, width as i32 - 1) as usize;
                        let idx = (y * width + sample_x) * 4 + c;
                        value += f32::from(input[idx]) * kernel[k];
                    }

                    pixel[c] = value.round().clamp(0.0, 255.0) as u8;
                }
            });

        // Vertical pass
        output
            .par_chunks_exact_mut(4)
            .enumerate()
            .for_each(|(i, pixel)| {
                let x = i % width;
                let y = i / width;

                if y >= height {
                    return;
                }

                for c in 0..4 {
                    let mut value = 0.0f32;

                    for k in 0..kernel_size {
                        let offset = k as i32 - kernel_radius;
                        let sample_y = (y as i32 + offset).clamp(0, height as i32 - 1) as usize;
                        let idx = (sample_y * width + x) * 4 + c;
                        value += f32::from(temp[idx]) * kernel[k];
                    }

                    pixel[c] = value.round().clamp(0.0, 255.0) as u8;
                }
            });
    }

    /// Check if CPU SIMD is available
    #[must_use]
    pub fn has_simd() -> bool {
        // Check for various SIMD instruction sets
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2") || is_x86_feature_detected!("sse4.2")
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is standard on aarch64
            true
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }
}

impl Backend for CpuBackend {
    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    fn is_available() -> bool {
        // CPU backend is always available
        true
    }

    fn initialize() -> Result<Self> {
        Self::new()
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new().expect("Failed to initialize CPU backend")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_always_available() {
        assert!(CpuBackend::is_available());
    }

    #[test]
    fn test_cpu_backend_creation() {
        let backend = CpuBackend::new().expect("CPU backend creation should succeed");
        assert!(backend.num_threads() > 0);
        assert_eq!(backend.capabilities().backend_type, BackendType::CPU);
    }

    #[test]
    fn test_simd_detection() {
        let has_simd = CpuBackend::has_simd();
        println!("SIMD available: {has_simd}");
    }

    #[test]
    fn test_rgb_to_yuv_cpu() {
        let input = vec![255, 0, 0, 255]; // Red pixel
        let mut output = vec![0u8; 4];

        CpuBackend::rgb_to_yuv_bt601(&input, &mut output, 1, 1);

        // Y should be around 76 (0.299 * 255)
        assert!(output[0] > 70 && output[0] < 80);
    }

    #[test]
    fn test_resize_bilinear_cpu() {
        let input = vec![255u8; 2 * 2 * 4]; // 2x2 white image
        let mut output = vec![0u8; 4 * 4 * 4]; // 4x4 output

        CpuBackend::resize_bilinear(&input, 2, 2, &mut output, 4, 4);

        // Output should be mostly white
        assert!(output[0] > 200);
    }
}
