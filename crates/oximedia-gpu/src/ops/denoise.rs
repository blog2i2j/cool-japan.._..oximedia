//! GPU-accelerated video denoising operations.
//!
//! This module provides compute-shader-based denoise filters for video frames.
//! Three algorithms are available:
//!
//! - **Gaussian** – simple spatial blur (low latency, lower quality)
//! - **`NonLocalMeans`** – patch-based NLM denoising (higher quality, more compute)
//! - **`BilateralFilter`** – edge-preserving spatial filter (good quality/speed trade-off)
//!
//! All operations fall back to CPU SIMD code when no suitable GPU is available
//! (the `GpuDevice::new` failure path returns an error that the caller may handle
//! by switching to a software path).
//!
//! ## GPU path (bilateral filter)
//!
//! When `device.is_fallback == false` the bilateral filter is executed on the GPU
//! via a WGSL compute shader (`shaders/bilateral.wgsl`).  The NLM algorithm is
//! too expensive for a single-dispatch compute shader (O(search_area² × patch_area)
//! per pixel) and is therefore kept as a CPU fallback; GPU NLM is planned for a
//! future release using multi-pass tiled reduction.

use crate::{
    shader::{BindGroupLayoutBuilder, ShaderCompiler, ShaderSource},
    GpuDevice, GpuError, Result,
};
use bytemuck::{Pod, Zeroable};
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use wgpu::{BindGroupLayout, ComputePipeline};

// ============================================================================
// Denoise algorithm selector
// ============================================================================

/// Denoise algorithm variants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DenoiseAlgorithm {
    /// Gaussian spatial blur.
    ///
    /// Parameters:
    /// - `sigma`: standard deviation of the Gaussian kernel (e.g. 1.5).
    Gaussian {
        /// Standard deviation for the Gaussian kernel.
        sigma: f32,
    },

    /// Non-Local Means denoising.
    ///
    /// Parameters:
    /// - `h`: filter strength (higher = more smoothing, e.g. 10.0).
    /// - `patch_radius`: half-size of the comparison patch (e.g. 3).
    /// - `search_radius`: half-size of the search window (e.g. 10).
    NonLocalMeans {
        /// Filter strength (denoising parameter *h*).
        h: f32,
        /// Comparison patch half-radius.
        patch_radius: u32,
        /// Search window half-radius.
        search_radius: u32,
    },

    /// Bilateral filter (edge-preserving).
    ///
    /// Parameters:
    /// - `sigma_spatial`: spatial Gaussian standard deviation.
    /// - `sigma_range`: range Gaussian standard deviation (pixel value domain).
    BilateralFilter {
        /// Spatial Gaussian standard deviation.
        sigma_spatial: f32,
        /// Range (colour) Gaussian standard deviation.
        sigma_range: f32,
    },
}

// ============================================================================
// Parameter structs (GPU-uploadable)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GaussianDenoiseParams {
    width: u32,
    height: u32,
    kernel_radius: u32,
    _pad: u32,
    sigma: f32,
    inv_two_sigma_sq: f32,
    _pad2: [f32; 2],
}

/// GPU-side uniform layout (must match `BilateralParams` struct in bilateral.wgsl exactly).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BilateralParams {
    width: u32,
    height: u32,
    kernel_radius: u32,
    _pad: u32,
    sigma_spatial: f32,
    sigma_range: f32,
    inv_two_sigma_s_sq: f32,
    inv_two_sigma_r_sq: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NlmParams {
    width: u32,
    height: u32,
    patch_radius: u32,
    search_radius: u32,
    h_sq: f32,
    inv_patch_area: f32,
    _pad: [f32; 2],
}

// ============================================================================
// Public DenoiseOperation API
// ============================================================================

/// GPU-accelerated denoise operations.
///
/// # Note on GPU vs CPU execution
///
/// When a real GPU device is available (`!device.is_fallback`), the bilateral
/// filter runs as a wgpu compute shader (`bilateral_filter_main` entry point in
/// `shaders/bilateral.wgsl`).  If the GPU path fails at any point, execution
/// transparently falls back to the CPU SIMD implementation.
///
/// Gaussian and NLM always use the CPU path (NLM GPU path is planned for a
/// future release).
pub struct DenoiseOperation;

impl DenoiseOperation {
    /// Denoise an RGBA image using the selected algorithm.
    ///
    /// Both `input` and `output` must be `width * height * 4` bytes
    /// (packed RGBA, one byte per channel).
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid.
    pub fn denoise(
        device: &GpuDevice,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        algorithm: DenoiseAlgorithm,
    ) -> Result<()> {
        super::utils::validate_dimensions(width, height)?;
        super::utils::validate_buffer_size(input, width, height, 4)?;
        super::utils::validate_buffer_size(output, width, height, 4)?;

        match algorithm {
            DenoiseAlgorithm::Gaussian { sigma } => {
                Self::denoise_gaussian_cpu(input, output, width, height, sigma)
            }
            DenoiseAlgorithm::BilateralFilter {
                sigma_spatial,
                sigma_range,
            } => {
                // Prefer GPU path when the device is a real (non-fallback) adapter.
                if !device.is_fallback {
                    match Self::denoise_bilateral_gpu(
                        device,
                        input,
                        output,
                        width,
                        height,
                        sigma_spatial,
                        sigma_range,
                    ) {
                        Ok(()) => return Ok(()),
                        Err(e) => {
                            tracing::warn!(
                                "GPU bilateral filter failed ({e}), falling back to CPU"
                            );
                        }
                    }
                }
                // CPU fallback (also used for software adapters).
                Self::denoise_bilateral_cpu(
                    input,
                    output,
                    width,
                    height,
                    sigma_spatial,
                    sigma_range,
                )
            }
            // NLM GPU path is future work — CPU fallback only.
            DenoiseAlgorithm::NonLocalMeans {
                h,
                patch_radius,
                search_radius,
            } => {
                Self::denoise_nlm_cpu(input, output, width, height, h, patch_radius, search_radius)
            }
        }
    }

    // -----------------------------------------------------------------------
    // GPU path — bilateral filter
    // -----------------------------------------------------------------------

    /// Run the bilateral filter on the GPU via wgpu compute shader.
    #[allow(clippy::cast_possible_truncation)]
    fn denoise_bilateral_gpu(
        device: &GpuDevice,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        sigma_spatial: f32,
        sigma_range: f32,
    ) -> Result<()> {
        use super::utils::{
            calculate_dispatch_size, create_readback_buffer, create_storage_buffer,
            create_uniform_buffer,
        };

        let pipeline = Self::get_bilateral_pipeline(device)?;
        let layout = Self::get_bilateral_bind_group_layout(device)?;

        // Build uniform params buffer.
        let kernel_radius = (3.0 * sigma_spatial).ceil() as u32;
        let inv_two_ss_sq = 1.0 / (2.0 * sigma_spatial * sigma_spatial);
        let inv_two_sr_sq = 1.0 / (2.0 * sigma_range * sigma_range);

        let params = BilateralParams {
            width,
            height,
            kernel_radius,
            _pad: 0,
            sigma_spatial,
            sigma_range,
            inv_two_sigma_s_sq: inv_two_ss_sq,
            inv_two_sigma_r_sq: inv_two_sr_sq,
        };

        // Pack the u8 RGBA input as u32 words (shader reads array<u32>).
        // Each u32 packs one RGBA pixel: R<<24 | G<<16 | B<<8 | A.
        let num_pixels = (width * height) as usize;
        let mut input_u32: Vec<u32> = Vec::with_capacity(num_pixels);
        for chunk in input.chunks_exact(4) {
            let packed = ((chunk[0] as u32) << 24)
                | ((chunk[1] as u32) << 16)
                | ((chunk[2] as u32) << 8)
                | (chunk[3] as u32);
            input_u32.push(packed);
        }

        let input_bytes = bytemuck::cast_slice(&input_u32);
        let output_len = num_pixels * 4; // u32 per pixel, 4 bytes each

        let input_buffer = create_storage_buffer(device, input_bytes.len() as u64)?;
        let output_buffer = create_storage_buffer(device, output_len as u64)?;
        let params_buffer = create_uniform_buffer(device, bytemuck::bytes_of(&params))?;

        device
            .queue()
            .write_buffer(input_buffer.buffer(), 0, input_bytes);

        // Build bind group.
        let compiler = ShaderCompiler::new(device);
        let bind_group = compiler.create_bind_group(
            "Bilateral Bind Group",
            layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.buffer().as_entire_binding(),
                },
            ],
        );

        // Dispatch compute.
        {
            let mut encoder =
                device
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Bilateral Compute Encoder"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Bilateral Compute Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let (dx, dy) = calculate_dispatch_size(width, height, (16, 16));
                pass.dispatch_workgroups(dx, dy, 1);
            }
            device.queue().submit(Some(encoder.finish()));
        }

        // Copy output_buffer → readback buffer.
        let readback = create_readback_buffer(device, output_len as u64)?;
        {
            let mut encoder =
                device
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Bilateral Readback Encoder"),
                    });
            output_buffer.copy_to(&mut encoder, &readback, 0, 0, output_len as u64)?;
            device.queue().submit(Some(encoder.finish()));
        }

        device.wait();

        // Read back and unpack u32 → RGBA bytes.
        let raw = readback.read(device, 0, output_len as u64)?;
        let result_u32: &[u32] = bytemuck::cast_slice(&raw);
        for (i, &packed) in result_u32.iter().enumerate() {
            output[i * 4] = ((packed >> 24) & 0xFF) as u8;
            output[i * 4 + 1] = ((packed >> 16) & 0xFF) as u8;
            output[i * 4 + 2] = ((packed >> 8) & 0xFF) as u8;
            output[i * 4 + 3] = (packed & 0xFF) as u8;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Pipeline management (cached per process lifetime via OnceCell)
    // -----------------------------------------------------------------------

    fn get_bilateral_bind_group_layout(device: &GpuDevice) -> Result<&'static BindGroupLayout> {
        static LAYOUT: OnceCell<BindGroupLayout> = OnceCell::new();
        Ok(LAYOUT.get_or_init(|| {
            let compiler = ShaderCompiler::new(device);
            let entries = BindGroupLayoutBuilder::new()
                .add_storage_buffer_read_only(0) // input
                .add_storage_buffer(1) // output
                .add_uniform_buffer(2) // params
                .build();
            compiler.create_bind_group_layout("Bilateral Bind Group Layout", &entries)
        }))
    }

    fn get_bilateral_pipeline(device: &GpuDevice) -> Result<&'static ComputePipeline> {
        static PIPELINE: OnceCell<std::result::Result<ComputePipeline, String>> = OnceCell::new();
        PIPELINE
            .get_or_init(|| Self::init_bilateral_pipeline(device))
            .as_ref()
            .map_err(|e| GpuError::PipelineCreation(e.clone()))
    }

    fn init_bilateral_pipeline(device: &GpuDevice) -> std::result::Result<ComputePipeline, String> {
        let compiler = ShaderCompiler::new(device);
        let shader = compiler
            .compile(
                "Bilateral Shader",
                ShaderSource::Embedded(crate::shader::embedded::BILATERAL_SHADER),
            )
            .map_err(|e| format!("Failed to compile bilateral shader: {e}"))?;

        let layout = Self::get_bilateral_bind_group_layout(device)
            .map_err(|e| format!("Failed to create bilateral bind group layout: {e}"))?;

        compiler
            .create_pipeline(
                "Bilateral Pipeline",
                &shader,
                "bilateral_filter_main",
                layout,
            )
            .map_err(|e| format!("Failed to create bilateral pipeline: {e}"))
    }

    // -----------------------------------------------------------------------
    // CPU SIMD fallback implementations
    // -----------------------------------------------------------------------

    /// Gaussian denoise via separable 1D convolution.
    #[allow(clippy::cast_possible_truncation)]
    fn denoise_gaussian_cpu(
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        sigma: f32,
    ) -> Result<()> {
        let w = width as usize;
        let h = height as usize;
        let radius = (3.0 * sigma).ceil() as usize;

        // Build separable 1D Gaussian kernel.
        let kernel_size = 2 * radius + 1;
        let mut kernel = vec![0.0f32; kernel_size];
        let two_sigma_sq = 2.0 * sigma * sigma;
        let mut sum = 0.0f32;
        for (i, k) in kernel.iter_mut().enumerate() {
            let x = i as f32 - radius as f32;
            *k = (-(x * x) / two_sigma_sq).exp();
            sum += *k;
        }
        for k in &mut kernel {
            *k /= sum;
        }

        // Horizontal pass into temp buffer.
        let mut temp = vec![0u8; input.len()];
        temp.par_chunks_exact_mut(w * 4)
            .enumerate()
            .for_each(|(y, row)| {
                if y >= h {
                    return;
                }
                for x in 0..w {
                    for c in 0..4usize {
                        let mut acc = 0.0f32;
                        for (ki, &kv) in kernel.iter().enumerate() {
                            let sx = (x as i64 + ki as i64 - radius as i64).clamp(0, w as i64 - 1)
                                as usize;
                            acc += kv * f32::from(input[(y * w + sx) * 4 + c]);
                        }
                        row[x * 4 + c] = acc.round().clamp(0.0, 255.0) as u8;
                    }
                }
            });

        // Vertical pass from temp to output.
        output
            .par_chunks_exact_mut(4)
            .enumerate()
            .for_each(|(i, pixel)| {
                let x = i % w;
                let y = i / w;
                if y >= h {
                    return;
                }
                for c in 0..4usize {
                    let mut acc = 0.0f32;
                    for (ki, &kv) in kernel.iter().enumerate() {
                        let sy =
                            (y as i64 + ki as i64 - radius as i64).clamp(0, h as i64 - 1) as usize;
                        acc += kv * f32::from(temp[(sy * w + x) * 4 + c]);
                    }
                    pixel[c] = acc.round().clamp(0.0, 255.0) as u8;
                }
            });

        Ok(())
    }

    /// Bilateral filter — CPU fallback (edge-preserving denoising).
    ///
    /// Exposed as `pub(crate)` so that `FilterOperation` can delegate to this
    /// implementation without duplicating the algorithm.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn denoise_bilateral_cpu(
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        sigma_spatial: f32,
        sigma_range: f32,
    ) -> Result<()> {
        let w = width as usize;
        let h = height as usize;
        let radius = (3.0 * sigma_spatial).ceil() as usize;
        let inv_two_ss_sq = 1.0 / (2.0 * sigma_spatial * sigma_spatial);
        let inv_two_sr_sq = 1.0 / (2.0 * sigma_range * sigma_range);

        output
            .par_chunks_exact_mut(4)
            .enumerate()
            .for_each(|(i, pixel)| {
                let x = i % w;
                let y = i / w;
                if y >= h {
                    return;
                }

                let center = [
                    f32::from(input[(y * w + x) * 4]),
                    f32::from(input[(y * w + x) * 4 + 1]),
                    f32::from(input[(y * w + x) * 4 + 2]),
                    f32::from(input[(y * w + x) * 4 + 3]),
                ];

                let mut acc = [0.0f32; 4];
                let mut weight_sum = 0.0f32;

                for dy in -(radius as i64)..=(radius as i64) {
                    for dx in -(radius as i64)..=(radius as i64) {
                        let sx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                        let sy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;

                        let spatial_dist_sq = (dx * dx + dy * dy) as f32;
                        let w_spatial = (-spatial_dist_sq * inv_two_ss_sq).exp();

                        let neighbor = [
                            f32::from(input[(sy * w + sx) * 4]),
                            f32::from(input[(sy * w + sx) * 4 + 1]),
                            f32::from(input[(sy * w + sx) * 4 + 2]),
                            f32::from(input[(sy * w + sx) * 4 + 3]),
                        ];

                        let range_dist_sq = (0..3)
                            .map(|c| (center[c] - neighbor[c]).powi(2))
                            .sum::<f32>();
                        let w_range = (-range_dist_sq * inv_two_sr_sq).exp();

                        let w_total = w_spatial * w_range;
                        weight_sum += w_total;

                        for c in 0..4 {
                            acc[c] += w_total * neighbor[c];
                        }
                    }
                }

                if weight_sum > 0.0 {
                    for c in 0..4 {
                        pixel[c] = (acc[c] / weight_sum).round().clamp(0.0, 255.0) as u8;
                    }
                } else {
                    pixel.copy_from_slice(&input[i * 4..i * 4 + 4]);
                }
            });

        Ok(())
    }

    /// Non-Local Means denoising (CPU path).
    ///
    /// GPU NLM is future work — the O(search_area² × patch_area) per-pixel cost
    /// requires a multi-pass tiled reduction that does not map naturally to a
    /// single compute dispatch.
    #[allow(clippy::cast_possible_truncation)]
    fn denoise_nlm_cpu(
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        h: f32,
        patch_radius: u32,
        search_radius: u32,
    ) -> Result<()> {
        if h <= 0.0 {
            return Err(GpuError::Internal(
                "NLM filter strength h must be positive".to_string(),
            ));
        }

        let w = width as usize;
        let ht = height as usize;
        let pr = patch_radius as usize;
        let sr = search_radius as usize;
        let h_sq = h * h;
        let patch_area = ((2 * pr + 1) * (2 * pr + 1)) as f32;
        let inv_h_sq_patch = 1.0 / (h_sq * patch_area);

        output
            .par_chunks_exact_mut(4)
            .enumerate()
            .for_each(|(i, pixel)| {
                let px = i % w;
                let py = i / w;
                if py >= ht {
                    return;
                }

                let mut acc = [0.0f32; 4];
                let mut weight_sum = 0.0f32;

                // Iterate over search window.
                for qy in
                    (py as i64 - sr as i64).max(0)..=(py as i64 + sr as i64).min(ht as i64 - 1)
                {
                    for qx in
                        (px as i64 - sr as i64).max(0)..=(px as i64 + sr as i64).min(w as i64 - 1)
                    {
                        // Compute patch distance between (px,py) and (qx,qy).
                        let mut patch_dist_sq = 0.0f32;
                        for ky in -(pr as i64)..=(pr as i64) {
                            for kx in -(pr as i64)..=(pr as i64) {
                                let p_x = (px as i64 + kx).clamp(0, w as i64 - 1) as usize;
                                let p_y = (py as i64 + ky).clamp(0, ht as i64 - 1) as usize;
                                let q_x = (qx + kx).clamp(0, w as i64 - 1) as usize;
                                let q_y = (qy + ky).clamp(0, ht as i64 - 1) as usize;

                                // Use luma (channel 0) for patch comparison.
                                let diff = f32::from(input[(p_y * w + p_x) * 4])
                                    - f32::from(input[(q_y * w + q_x) * 4]);
                                patch_dist_sq += diff * diff;
                            }
                        }

                        let w_nlm = (-patch_dist_sq * inv_h_sq_patch).exp();
                        weight_sum += w_nlm;

                        for c in 0..4 {
                            acc[c] +=
                                w_nlm * f32::from(input[(qy as usize * w + qx as usize) * 4 + c]);
                        }
                    }
                }

                if weight_sum > 0.0 {
                    for c in 0..4 {
                        pixel[c] = (acc[c] / weight_sum).round().clamp(0.0, 255.0) as u8;
                    }
                } else {
                    pixel.copy_from_slice(&input[i * 4..i * 4 + 4]);
                }
            });

        Ok(())
    }

    /// Validate that `sigma > 0.0` and return a descriptive error if not.
    #[allow(dead_code)]
    fn check_sigma(sigma: f32, name: &str) -> Result<()> {
        if sigma <= 0.0 {
            Err(GpuError::Internal(format!(
                "{name} must be positive, got {sigma}"
            )))
        } else {
            Ok(())
        }
    }

    /// Convenience: Gaussian denoise with automatic sigma selection.
    ///
    /// `noise_level` is in the range \[0.0, 1.0\] where 0.0 = no noise
    /// and 1.0 = heavy noise.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid.
    pub fn auto_denoise(
        device: &GpuDevice,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        noise_level: f32,
    ) -> Result<()> {
        let sigma = noise_level.clamp(0.0, 1.0) * 3.0 + 0.5;
        Self::denoise(
            device,
            input,
            output,
            width,
            height,
            DenoiseAlgorithm::Gaussian { sigma },
        )
    }
}

// ============================================================================
// Denoise kernel wrappers (kernel module integration)
// ============================================================================

/// Denoise kernel configuration for use with the `kernels` module.
#[derive(Debug, Clone)]
pub struct DenoiseKernel {
    algorithm: DenoiseAlgorithm,
}

impl DenoiseKernel {
    /// Create a new denoise kernel with the given algorithm.
    #[must_use]
    pub fn new(algorithm: DenoiseAlgorithm) -> Self {
        Self { algorithm }
    }

    /// Create a Gaussian denoise kernel.
    #[must_use]
    pub fn gaussian(sigma: f32) -> Self {
        Self::new(DenoiseAlgorithm::Gaussian { sigma })
    }

    /// Create a bilateral filter denoise kernel.
    #[must_use]
    pub fn bilateral(sigma_spatial: f32, sigma_range: f32) -> Self {
        Self::new(DenoiseAlgorithm::BilateralFilter {
            sigma_spatial,
            sigma_range,
        })
    }

    /// Create an NLM denoise kernel.
    #[must_use]
    pub fn nlm(h: f32, patch_radius: u32, search_radius: u32) -> Self {
        Self::new(DenoiseAlgorithm::NonLocalMeans {
            h,
            patch_radius,
            search_radius,
        })
    }

    /// Apply this kernel to an RGBA image.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid or the operation fails.
    pub fn apply(
        &self,
        device: &GpuDevice,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        DenoiseOperation::denoise(device, input, output, width, height, self.algorithm)
    }

    /// Get the algorithm used by this kernel.
    #[must_use]
    pub fn algorithm(&self) -> DenoiseAlgorithm {
        self.algorithm
    }

    /// Estimate GFLOP for `width × height` frame at this algorithm.
    #[must_use]
    pub fn estimate_gflops(&self, width: u32, height: u32) -> f64 {
        let pixels = u64::from(width) * u64::from(height);
        let ops: u64 = match self.algorithm {
            DenoiseAlgorithm::Gaussian { sigma } => {
                let r = (3.0 * sigma).ceil() as u64;
                let k = 2 * r + 1;
                pixels * k * 4 * 4 // 2 passes, ~4 ops/tap, 4 channels
            }
            DenoiseAlgorithm::BilateralFilter { sigma_spatial, .. } => {
                let r = (3.0 * sigma_spatial).ceil() as u64;
                let k = (2 * r + 1).pow(2);
                pixels * k * 12 * 4 // exp + mul + add, 4 channels
            }
            DenoiseAlgorithm::NonLocalMeans {
                patch_radius,
                search_radius,
                ..
            } => {
                let pr = u64::from(2 * patch_radius + 1).pow(2);
                let sr = u64::from(2 * search_radius + 1).pow(2);
                pixels * sr * pr * 5 // patch distance + weighting
            }
        };
        ops as f64 / 1e9
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_image(w: u32, h: u32, value: u8) -> Vec<u8> {
        vec![value; (w * h * 4) as usize]
    }

    fn noisy_image(w: u32, h: u32) -> Vec<u8> {
        (0..(w * h * 4))
            .map(|i| (i as u8).wrapping_mul(37))
            .collect()
    }

    // ---- DenoiseAlgorithm --------------------------------------------------

    #[test]
    fn test_gaussian_denoise_cpu_constant_image() {
        let w = 16u32;
        let h = 16u32;
        let input = gray_image(w, h, 200);
        let mut output = vec![0u8; (w * h * 4) as usize];
        let result = DenoiseOperation::denoise_gaussian_cpu(&input, &mut output, w, h, 1.5);
        assert!(result.is_ok());
        // A constant image should pass through unchanged (all values should be 200).
        for &v in &output {
            assert_eq!(v, 200);
        }
    }

    #[test]
    fn test_gaussian_denoise_cpu_noisy() {
        let w = 32u32;
        let h = 32u32;
        let input = noisy_image(w, h);
        let mut output = vec![0u8; (w * h * 4) as usize];
        let result = DenoiseOperation::denoise_gaussian_cpu(&input, &mut output, w, h, 2.0);
        assert!(result.is_ok());
        // Output should not be all-zeros.
        assert!(output.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_bilateral_denoise_cpu_constant() {
        let w = 8u32;
        let h = 8u32;
        let input = gray_image(w, h, 100);
        let mut output = vec![0u8; (w * h * 4) as usize];
        let result = DenoiseOperation::denoise_bilateral_cpu(&input, &mut output, w, h, 1.5, 30.0);
        assert!(result.is_ok());
        for &v in &output {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn test_nlm_denoise_cpu_constant() {
        let w = 8u32;
        let h = 8u32;
        let input = gray_image(w, h, 150);
        let mut output = vec![0u8; (w * h * 4) as usize];
        let result = DenoiseOperation::denoise_nlm_cpu(&input, &mut output, w, h, 10.0, 2, 5);
        assert!(result.is_ok());
        for &v in &output {
            assert_eq!(v, 150);
        }
    }

    #[test]
    fn test_nlm_denoise_invalid_h() {
        let w = 4u32;
        let h = 4u32;
        let input = gray_image(w, h, 0);
        let mut output = vec![0u8; (w * h * 4) as usize];
        let result = DenoiseOperation::denoise_nlm_cpu(&input, &mut output, w, h, 0.0, 1, 3);
        assert!(result.is_err());
    }

    // ---- DenoiseKernel -----------------------------------------------------

    #[test]
    fn test_denoise_kernel_gaussian() {
        let k = DenoiseKernel::gaussian(1.0);
        assert_eq!(k.algorithm(), DenoiseAlgorithm::Gaussian { sigma: 1.0 });
    }

    #[test]
    fn test_denoise_kernel_bilateral() {
        let k = DenoiseKernel::bilateral(2.0, 25.0);
        assert_eq!(
            k.algorithm(),
            DenoiseAlgorithm::BilateralFilter {
                sigma_spatial: 2.0,
                sigma_range: 25.0,
            }
        );
    }

    #[test]
    fn test_denoise_kernel_nlm() {
        let k = DenoiseKernel::nlm(10.0, 3, 10);
        assert_eq!(
            k.algorithm(),
            DenoiseAlgorithm::NonLocalMeans {
                h: 10.0,
                patch_radius: 3,
                search_radius: 10,
            }
        );
    }

    #[test]
    fn test_estimate_gflops_not_zero() {
        let k = DenoiseKernel::gaussian(1.5);
        assert!(k.estimate_gflops(1920, 1080) > 0.0);

        let k2 = DenoiseKernel::nlm(10.0, 3, 10);
        assert!(k2.estimate_gflops(1920, 1080) > 0.0);
    }

    // ---- GPU bilateral path (uses fallback device in test environments) ----

    #[test]
    fn test_bilateral_denoise_via_denoise_fn_constant() {
        // Try to obtain a GPU device; fall back gracefully if unavailable.
        let device = match GpuDevice::new_fallback() {
            Ok(d) => d,
            Err(_) => return, // headless CI with no wgpu adapter — skip
        };

        let w = 8u32;
        let h = 8u32;
        let input = gray_image(w, h, 128);
        let mut output = vec![0u8; (w * h * 4) as usize];

        let result = DenoiseOperation::denoise(
            &device,
            &input,
            &mut output,
            w,
            h,
            DenoiseAlgorithm::BilateralFilter {
                sigma_spatial: 1.5,
                sigma_range: 30.0,
            },
        );

        // Constant image must stay constant regardless of GPU/CPU path.
        assert!(result.is_ok(), "denoise returned error: {:?}", result.err());
        for &v in &output {
            assert_eq!(v, 128, "constant image must be preserved");
        }
    }

    #[test]
    fn test_bilateral_denoise_params_struct_size() {
        // Ensure the CPU-side params layout is exactly 32 bytes to match the WGSL struct.
        assert_eq!(std::mem::size_of::<BilateralParams>(), 32);
    }
}
