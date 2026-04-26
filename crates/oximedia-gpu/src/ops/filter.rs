//! Convolution filter operations (blur, sharpen, edge detection)

use crate::{
    shader::{BindGroupLayoutBuilder, ShaderCompiler, ShaderSource},
    GpuDevice, GpuError, Result,
};
use bytemuck::{Pod, Zeroable};
use once_cell::sync::OnceCell;
use wgpu::{BindGroup, BindGroupLayout, ComputePipeline};

use super::utils;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FilterParams {
    width: u32,
    height: u32,
    stride: u32,
    kernel_size: u32,
    normalize: u32,
    filter_type: u32,
    padding: u32,
    sigma: f32,
}

/// Convolution filter operations
pub struct FilterOperation;

impl FilterOperation {
    /// Apply Gaussian blur
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - Input image buffer (packed RGBA format)
    /// * `output` - Output image buffer (packed RGBA format)
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `sigma` - Blur radius (standard deviation)
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid or if the GPU operation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn gaussian_blur(
        device: &GpuDevice,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        sigma: f32,
    ) -> Result<()> {
        utils::validate_dimensions(width, height)?;
        utils::validate_buffer_size(input, width, height, 4)?;
        utils::validate_buffer_size(output, width, height, 4)?;

        let kernel_size = Self::calculate_kernel_size(sigma);
        let pipeline = Self::get_gaussian_pipeline(device)?;
        let layout = Self::get_bind_group_layout(device)?;

        Self::execute_filter(
            device,
            pipeline,
            layout,
            input,
            output,
            width,
            height,
            kernel_size,
            1, // Gaussian filter type
            sigma,
        )
    }

    /// Apply sharpening filter (unsharp mask)
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - Input image buffer (packed RGBA format)
    /// * `output` - Output image buffer (packed RGBA format)
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `amount` - Sharpening strength
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid or if the GPU operation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn sharpen(
        device: &GpuDevice,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        amount: f32,
    ) -> Result<()> {
        utils::validate_dimensions(width, height)?;
        utils::validate_buffer_size(input, width, height, 4)?;
        utils::validate_buffer_size(output, width, height, 4)?;

        let pipeline = Self::get_sharpen_pipeline(device)?;
        let layout = Self::get_bind_group_layout(device)?;

        Self::execute_filter(
            device, pipeline, layout, input, output, width, height,
            5, // Kernel size for sharpening
            2, // Sharpen filter type
            amount,
        )
    }

    /// Detect edges using Sobel operator
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - Input image buffer (packed RGBA format)
    /// * `output` - Output image buffer (packed RGBA format)
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid or if the GPU operation fails.
    pub fn edge_detect(
        device: &GpuDevice,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        utils::validate_dimensions(width, height)?;
        utils::validate_buffer_size(input, width, height, 4)?;
        utils::validate_buffer_size(output, width, height, 4)?;

        let pipeline = Self::get_edge_detect_pipeline(device)?;
        let layout = Self::get_bind_group_layout(device)?;

        Self::execute_filter(
            device, pipeline, layout, input, output, width, height, 3, // 3x3 Sobel kernel
            3, // Edge detect filter type
            0.0,
        )
    }

    /// Apply custom convolution kernel
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - Input image buffer (packed RGBA format)
    /// * `output` - Output image buffer (packed RGBA format)
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `kernel` - Convolution kernel (must be square and odd-sized)
    /// * `normalize` - Whether to normalize the kernel
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid or if the GPU operation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn convolve(
        device: &GpuDevice,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        kernel: &[f32],
        normalize: bool,
    ) -> Result<()> {
        utils::validate_dimensions(width, height)?;
        utils::validate_buffer_size(input, width, height, 4)?;
        utils::validate_buffer_size(output, width, height, 4)?;

        let kernel_size = (kernel.len() as f32).sqrt() as u32;
        if kernel_size * kernel_size != kernel.len() as u32 {
            return Err(GpuError::Internal("Kernel must be square".to_string()));
        }
        if kernel_size % 2 == 0 {
            return Err(GpuError::Internal("Kernel size must be odd".to_string()));
        }

        let pipeline = Self::get_convolve_pipeline(device)?;
        let layout = Self::get_bind_group_layout_with_kernel(device)?;

        Self::execute_convolve(
            device,
            pipeline,
            layout,
            input,
            output,
            width,
            height,
            kernel,
            kernel_size,
            normalize,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_filter(
        device: &GpuDevice,
        pipeline: &ComputePipeline,
        layout: &BindGroupLayout,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        kernel_size: u32,
        filter_type: u32,
        sigma: f32,
    ) -> Result<()> {
        // Create buffers
        let input_buffer = utils::create_storage_buffer(device, input.len() as u64)?;
        let output_buffer = utils::create_storage_buffer(device, output.len() as u64)?;

        // Upload input data
        device.queue().write_buffer(input_buffer.buffer(), 0, input);

        // Create uniform buffer for parameters
        let params = FilterParams {
            width,
            height,
            stride: width,
            kernel_size,
            normalize: 1,
            filter_type,
            padding: 0,
            sigma,
        };
        let params_bytes = bytemuck::bytes_of(&params);
        let params_buffer = utils::create_uniform_buffer(device, params_bytes)?;

        // Create bind group
        let compiler = ShaderCompiler::new(device);
        let bind_group = compiler.create_bind_group(
            "Filter Bind Group",
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

        // Execute compute pass
        Self::dispatch_compute(device, pipeline, &bind_group, width, height)?;

        // Read back results
        let readback_buffer = utils::create_readback_buffer(device, output.len() as u64)?;
        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Filter Copy Encoder"),
            });

        output_buffer.copy_to(&mut encoder, &readback_buffer, 0, 0, output.len() as u64)?;

        device.queue().submit(Some(encoder.finish()));
        device.wait();

        let result = readback_buffer.read(device, 0, output.len() as u64)?;
        output.copy_from_slice(&result);

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_convolve(
        device: &GpuDevice,
        pipeline: &ComputePipeline,
        layout: &BindGroupLayout,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        kernel: &[f32],
        kernel_size: u32,
        normalize: bool,
    ) -> Result<()> {
        // Create buffers
        let input_buffer = utils::create_storage_buffer(device, input.len() as u64)?;
        let output_buffer = utils::create_storage_buffer(device, output.len() as u64)?;

        // Upload input data
        device.queue().write_buffer(input_buffer.buffer(), 0, input);

        // Create kernel buffer
        let kernel_bytes = bytemuck::cast_slice(kernel);
        let kernel_buffer = utils::create_storage_buffer(device, kernel_bytes.len() as u64)?;
        device
            .queue()
            .write_buffer(kernel_buffer.buffer(), 0, kernel_bytes);

        // Create uniform buffer for parameters
        let params = FilterParams {
            width,
            height,
            stride: width,
            kernel_size,
            normalize: u32::from(normalize),
            filter_type: 0, // Custom kernel
            padding: 0,
            sigma: 0.0,
        };
        let params_bytes = bytemuck::bytes_of(&params);
        let params_buffer = utils::create_uniform_buffer(device, params_bytes)?;

        // Create bind group
        let compiler = ShaderCompiler::new(device);
        let bind_group = compiler.create_bind_group(
            "Filter Bind Group",
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: kernel_buffer.buffer().as_entire_binding(),
                },
            ],
        );

        // Execute compute pass
        Self::dispatch_compute(device, pipeline, &bind_group, width, height)?;

        // Read back results
        let readback_buffer = utils::create_readback_buffer(device, output.len() as u64)?;
        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Filter Copy Encoder"),
            });

        output_buffer.copy_to(&mut encoder, &readback_buffer, 0, 0, output.len() as u64)?;

        device.queue().submit(Some(encoder.finish()));
        device.wait();

        let result = readback_buffer.read(device, 0, output.len() as u64)?;
        output.copy_from_slice(&result);

        Ok(())
    }

    fn dispatch_compute(
        device: &GpuDevice,
        pipeline: &ComputePipeline,
        bind_group: &BindGroup,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Filter Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Filter Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);

            let (dispatch_x, dispatch_y) = utils::calculate_dispatch_size(width, height, (16, 16));
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        device.queue().submit(Some(encoder.finish()));
        Ok(())
    }

    fn calculate_kernel_size(sigma: f32) -> u32 {
        // Use 3-sigma rule: kernel size = 2 * ceil(3 * sigma) + 1
        let radius = (3.0 * sigma).ceil() as u32;
        2 * radius + 1
    }

    fn get_bind_group_layout(device: &GpuDevice) -> Result<&'static BindGroupLayout> {
        static LAYOUT: OnceCell<BindGroupLayout> = OnceCell::new();

        Ok(LAYOUT.get_or_init(|| {
            let compiler = ShaderCompiler::new(device);
            let entries = BindGroupLayoutBuilder::new()
                .add_storage_buffer_read_only(0) // input
                .add_storage_buffer(1) // output
                .add_uniform_buffer(2) // params
                .build();

            compiler.create_bind_group_layout("Filter Bind Group Layout", &entries)
        }))
    }

    fn get_bind_group_layout_with_kernel(device: &GpuDevice) -> Result<&'static BindGroupLayout> {
        static LAYOUT: OnceCell<BindGroupLayout> = OnceCell::new();

        Ok(LAYOUT.get_or_init(|| {
            let compiler = ShaderCompiler::new(device);
            let entries = BindGroupLayoutBuilder::new()
                .add_storage_buffer_read_only(0) // input
                .add_storage_buffer(1) // output
                .add_uniform_buffer(2) // params
                .add_storage_buffer_read_only(3) // kernel
                .build();

            compiler.create_bind_group_layout("Filter Bind Group Layout (with kernel)", &entries)
        }))
    }

    fn init_pipeline(
        device: &GpuDevice,
        name: &str,
        entry_point: &str,
        layout_fn: fn(&GpuDevice) -> Result<&'static BindGroupLayout>,
    ) -> std::result::Result<ComputePipeline, String> {
        let compiler = ShaderCompiler::new(device);
        let shader = compiler
            .compile(
                "Filter Shader",
                ShaderSource::Embedded(crate::shader::embedded::FILTER_SHADER),
            )
            .map_err(|e| format!("Failed to compile filter shader: {e}"))?;

        let layout =
            layout_fn(device).map_err(|e| format!("Failed to create bind group layout: {e}"))?;

        compiler
            .create_pipeline(name, &shader, entry_point, layout)
            .map_err(|e| format!("Failed to create pipeline: {e}"))
    }

    fn get_gaussian_pipeline(device: &GpuDevice) -> Result<&'static ComputePipeline> {
        static PIPELINE: OnceCell<std::result::Result<ComputePipeline, String>> = OnceCell::new();

        PIPELINE
            .get_or_init(|| {
                FilterOperation::init_pipeline(
                    device,
                    "Gaussian Blur Pipeline",
                    "convolve_main",
                    Self::get_bind_group_layout,
                )
            })
            .as_ref()
            .map_err(|e| crate::GpuError::PipelineCreation(e.clone()))
    }

    fn get_sharpen_pipeline(device: &GpuDevice) -> Result<&'static ComputePipeline> {
        static PIPELINE: OnceCell<std::result::Result<ComputePipeline, String>> = OnceCell::new();

        PIPELINE
            .get_or_init(|| {
                FilterOperation::init_pipeline(
                    device,
                    "Sharpen Pipeline",
                    "unsharp_mask",
                    Self::get_bind_group_layout,
                )
            })
            .as_ref()
            .map_err(|e| crate::GpuError::PipelineCreation(e.clone()))
    }

    fn get_edge_detect_pipeline(device: &GpuDevice) -> Result<&'static ComputePipeline> {
        static PIPELINE: OnceCell<std::result::Result<ComputePipeline, String>> = OnceCell::new();

        PIPELINE
            .get_or_init(|| {
                FilterOperation::init_pipeline(
                    device,
                    "Edge Detect Pipeline",
                    "edge_detect",
                    Self::get_bind_group_layout,
                )
            })
            .as_ref()
            .map_err(|e| crate::GpuError::PipelineCreation(e.clone()))
    }

    fn get_convolve_pipeline(device: &GpuDevice) -> Result<&'static ComputePipeline> {
        static PIPELINE: OnceCell<std::result::Result<ComputePipeline, String>> = OnceCell::new();

        PIPELINE
            .get_or_init(|| {
                FilterOperation::init_pipeline(
                    device,
                    "Convolve Pipeline",
                    "convolve_main",
                    Self::get_bind_group_layout_with_kernel,
                )
            })
            .as_ref()
            .map_err(|e| crate::GpuError::PipelineCreation(e.clone()))
    }
}

// ---------------------------------------------------------------------------
// Separable CPU Gaussian blur (Task 9)
// ---------------------------------------------------------------------------

/// Build a normalised 1-D Gaussian kernel of radius `ceil(3σ)`.
///
/// The returned `Vec<f32>` has length `2*radius+1` and sums to 1.0.
/// If `sigma` is ≤ 0 a single-element identity kernel `[1.0]` is returned.
#[must_use]
pub fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return vec![1.0_f32];
    }
    let radius = (3.0 * sigma).ceil() as usize;
    let len = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(len);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0_f32;
    for i in 0..len {
        let x = i as f32 - radius as f32;
        let v = (-x * x / two_sigma_sq).exp();
        kernel.push(v);
        sum += v;
    }
    for k in &mut kernel {
        *k /= sum;
    }
    kernel
}

/// CPU-side separable Gaussian blur (two-pass: horizontal then vertical).
///
/// # Arguments
///
/// * `input`  — source RGBA bytes (`width × height × 4`)
/// * `output` — destination RGBA bytes (same size as input)
/// * `width`, `height` — image dimensions
/// * `sigma`  — Gaussian standard deviation in pixels (> 0)
///
/// # Errors
///
/// Returns an error if buffer sizes do not match `width × height × 4`.
pub fn gaussian_blur_separable(
    input: &[u8],
    output: &mut [u8],
    width: u32,
    height: u32,
    sigma: f32,
) -> crate::Result<()> {
    utils::validate_dimensions(width, height)?;
    utils::validate_buffer_size(input, width, height, 4)?;
    utils::validate_buffer_size(output, width, height, 4)?;

    let w = width as usize;
    let h = height as usize;
    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;

    // Horizontal pass — accumulate into f32 buffer to avoid clamping artefacts
    let mut h_pass = vec![0.0_f32; w * h * 4];
    for row in 0..h {
        for col in 0..w {
            let mut acc = [0.0_f32; 4];
            let mut wsum = 0.0_f32;
            for (ki, &kw) in kernel.iter().enumerate() {
                let sc = col as isize + ki as isize - radius as isize;
                if sc < 0 || sc >= w as isize {
                    continue;
                }
                let src = (row * w + sc as usize) * 4;
                for c in 0..4 {
                    acc[c] += kw * input[src + c] as f32;
                }
                wsum += kw;
            }
            let dst = (row * w + col) * 4;
            let inv = if wsum > 0.0 { 1.0 / wsum } else { 1.0 };
            for c in 0..4 {
                h_pass[dst + c] = acc[c] * inv;
            }
        }
    }

    // Vertical pass — read from h_pass, write to output
    for row in 0..h {
        for col in 0..w {
            let mut acc = [0.0_f32; 4];
            let mut wsum = 0.0_f32;
            for (ki, &kw) in kernel.iter().enumerate() {
                let sr = row as isize + ki as isize - radius as isize;
                if sr < 0 || sr >= h as isize {
                    continue;
                }
                let src = (sr as usize * w + col) * 4;
                for c in 0..4 {
                    acc[c] += kw * h_pass[src + c];
                }
                wsum += kw;
            }
            let dst = (row * w + col) * 4;
            let inv = if wsum > 0.0 { 1.0 / wsum } else { 1.0 };
            for c in 0..4 {
                output[dst + c] = (acc[c] * inv).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(())
}

use rayon::prelude::*;

/// CPU-side separable Gaussian blur with Rayon parallel row/column processing.
///
/// This is an optimised variant of [`gaussian_blur_separable`] that
/// parallelises both the horizontal and vertical passes using Rayon.
///
/// # Arguments
///
/// * `input`  — source RGBA bytes (`width × height × 4`)
/// * `output` — destination RGBA bytes (same size as input)
/// * `width`, `height` — image dimensions
/// * `sigma`  — Gaussian standard deviation in pixels (> 0)
///
/// # Errors
///
/// Returns an error if buffer sizes do not match `width × height × 4`.
pub fn gaussian_blur_separable_parallel(
    input: &[u8],
    output: &mut [u8],
    width: u32,
    height: u32,
    sigma: f32,
) -> crate::Result<()> {
    utils::validate_dimensions(width, height)?;
    utils::validate_buffer_size(input, width, height, 4)?;
    utils::validate_buffer_size(output, width, height, 4)?;

    let w = width as usize;
    let h = height as usize;
    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;

    // Horizontal pass (parallel over rows)
    let mut h_pass = vec![0.0_f32; w * h * 4];
    h_pass
        .par_chunks_exact_mut(w * 4)
        .enumerate()
        .for_each(|(row, row_out)| {
            for col in 0..w {
                let mut acc = [0.0_f32; 4];
                let mut wsum = 0.0_f32;
                for (ki, &kw) in kernel.iter().enumerate() {
                    let sc = col as isize + ki as isize - radius as isize;
                    if sc < 0 || sc >= w as isize {
                        continue;
                    }
                    let src = (row * w + sc as usize) * 4;
                    for c in 0..4 {
                        acc[c] += kw * input[src + c] as f32;
                    }
                    wsum += kw;
                }
                let inv = if wsum > 0.0 { 1.0 / wsum } else { 1.0 };
                let dst = col * 4;
                for c in 0..4 {
                    row_out[dst + c] = acc[c] * inv;
                }
            }
        });

    // Vertical pass (parallel over columns)
    output
        .par_chunks_exact_mut(4)
        .enumerate()
        .for_each(|(px_idx, px_out)| {
            let row = px_idx / w;
            let col = px_idx % w;
            let mut acc = [0.0_f32; 4];
            let mut wsum = 0.0_f32;
            for (ki, &kw) in kernel.iter().enumerate() {
                let sr = row as isize + ki as isize - radius as isize;
                if sr < 0 || sr >= h as isize {
                    continue;
                }
                let src = (sr as usize * w + col) * 4;
                for c in 0..4 {
                    acc[c] += kw * h_pass[src + c];
                }
                wsum += kw;
            }
            let inv = if wsum > 0.0 { 1.0 / wsum } else { 1.0 };
            for c in 0..4 {
                px_out[c] = (acc[c] * inv).round().clamp(0.0, 255.0) as u8;
            }
        });

    Ok(())
}

// ---------------------------------------------------------------------------
// CPU-side box blur (separable sliding-sum, O(w*h) per channel)
// ---------------------------------------------------------------------------

/// CPU box blur: separable two-pass sliding-sum, O(w×h) per channel.
///
/// Border pixels are handled via clamped (replicate-border) indexing.
/// The horizontal pass builds an intermediate `u32` buffer; the vertical pass
/// writes into the final `Vec<u8>`.
///
/// # Errors
///
/// Returns an error if the buffer length does not match `width × height × channels`.
pub fn box_blur(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    radius: u32,
) -> crate::Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    let ch = channels as usize;
    let expected = w * h * ch;
    if data.len() != expected {
        return Err(crate::GpuError::InvalidBufferSize {
            expected,
            actual: data.len(),
        });
    }
    if w == 0 || h == 0 {
        return Ok(data.to_vec());
    }

    let r = radius as isize;

    // --- Horizontal pass ---
    // For each (row, col, channel): average over columns [col-r .. col+r] (clamped).
    // Use a sliding-sum that tracks left/right clamped edges.
    let mut h_pass = vec![0u32; w * h * ch];
    for row in 0..h {
        for c in 0..ch {
            // Build initial window sum for col = 0.
            let right0 = r.min(w as isize - 1) as usize;
            let mut window_sum: u32 = 0;
            for kc in 0..=right0 {
                window_sum += u32::from(data[(row * w + kc) * ch + c]);
            }

            for col in 0..w {
                // Compute the actual left/right clamped boundaries for this col.
                let left = (col as isize - r).max(0) as usize;
                let right = (col as isize + r).min(w as isize - 1) as usize;

                if col > 0 {
                    // Previous column's boundaries.
                    let prev_left = ((col as isize - 1) - r).max(0) as usize;
                    let prev_right = ((col as isize - 1) + r).min(w as isize - 1) as usize;
                    // Remove pixel that dropped off the left.
                    if left > prev_left {
                        window_sum -= u32::from(data[(row * w + prev_left) * ch + c]);
                    }
                    // Add pixel that entered on the right.
                    if right > prev_right {
                        window_sum += u32::from(data[(row * w + right) * ch + c]);
                    }
                }

                let window_len = (right - left + 1) as u32;
                // Round-to-nearest division.
                h_pass[(row * w + col) * ch + c] = (window_sum + window_len / 2) / window_len;
            }
        }
    }

    // --- Vertical pass ---
    // For each (row, col, channel): average over rows [row-r .. row+r] (clamped).
    let mut output = vec![0u8; expected];
    for col in 0..w {
        for c in 0..ch {
            // Build initial window sum for row = 0.
            let bot0 = r.min(h as isize - 1) as usize;
            let mut window_sum: u32 = 0;
            for kr in 0..=bot0 {
                window_sum += h_pass[(kr * w + col) * ch + c];
            }

            for row in 0..h {
                let top = (row as isize - r).max(0) as usize;
                let bot = (row as isize + r).min(h as isize - 1) as usize;

                if row > 0 {
                    let prev_top = ((row as isize - 1) - r).max(0) as usize;
                    let prev_bot = ((row as isize - 1) + r).min(h as isize - 1) as usize;
                    if top > prev_top {
                        window_sum -= h_pass[(prev_top * w + col) * ch + c];
                    }
                    if bot > prev_bot {
                        window_sum += h_pass[(bot * w + col) * ch + c];
                    }
                }

                let window_len = (bot - top + 1) as u32;
                let avg = (window_sum + window_len / 2) / window_len;
                output[(row * w + col) * ch + c] = avg.clamp(0, 255) as u8;
            }
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// CPU-side median filter
// ---------------------------------------------------------------------------

/// CPU median filter: sorts a `(2r+1)×(2r+1)` neighbourhood per pixel/channel.
///
/// Border pixels use clamped neighbour coordinates (replicate border).
///
/// # Errors
///
/// Returns an error if the buffer length does not match `width × height × channels`.
pub fn median_filter(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    radius: u32,
) -> crate::Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    let ch = channels as usize;
    let expected = w * h * ch;
    if data.len() != expected {
        return Err(crate::GpuError::InvalidBufferSize {
            expected,
            actual: data.len(),
        });
    }
    if w == 0 || h == 0 {
        return Ok(data.to_vec());
    }

    let r = radius as isize;
    let window_len = ((2 * r + 1) * (2 * r + 1)) as usize;
    let mut output = vec![0u8; expected];

    for row in 0..h {
        for col in 0..w {
            for c in 0..ch {
                let mut window: Vec<u8> = Vec::with_capacity(window_len);
                for dy in -r..=r {
                    for dx in -r..=r {
                        let sr = (row as isize + dy).clamp(0, h as isize - 1) as usize;
                        let sc = (col as isize + dx).clamp(0, w as isize - 1) as usize;
                        window.push(data[(sr * w + sc) * ch + c]);
                    }
                }
                window.sort_unstable();
                output[(row * w + col) * ch + c] = window[window.len() / 2];
            }
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// CPU-side bilateral filter wrapper (delegates to DenoiseOperation)
// ---------------------------------------------------------------------------

/// CPU bilateral filter: edge-preserving spatial filter.
///
/// Delegates to `denoise_bilateral_cpu` so
/// there is a single canonical implementation.
///
/// # Errors
///
/// Returns an error if the buffer length does not match `width × height × channels`,
/// or if `channels != 4` (the bilateral implementation is RGBA-only).
pub fn bilateral_filter(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    sigma_spatial: f32,
    sigma_range: f32,
) -> crate::Result<Vec<u8>> {
    if channels != 4 {
        return Err(crate::GpuError::NotSupported(format!(
            "bilateral_filter requires channels == 4, got {channels}"
        )));
    }
    utils::validate_buffer_size(data, width, height, 4)?;
    let mut output = vec![0u8; data.len()];
    super::DenoiseOperation::denoise_bilateral_cpu(
        data,
        &mut output,
        width,
        height,
        sigma_spatial,
        sigma_range,
    )?;
    Ok(output)
}

/// Compare two RGBA u8 buffers and return the maximum absolute channel difference.
///
/// Useful for verifying that the separable serial and parallel implementations
/// produce bit-identical (or near-identical) results.
#[must_use]
pub fn max_channel_diff(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs())
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_sums_to_one() {
        let k = gaussian_kernel_1d(1.0);
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "kernel sum = {sum}");
    }

    #[test]
    fn test_kernel_is_symmetric() {
        let k = gaussian_kernel_1d(2.0);
        let n = k.len();
        for i in 0..n / 2 {
            assert!(
                (k[i] - k[n - 1 - i]).abs() < 1e-6,
                "asymmetric at index {i}: {} vs {}",
                k[i],
                k[n - 1 - i]
            );
        }
    }

    #[test]
    fn test_kernel_center_is_largest() {
        let k = gaussian_kernel_1d(1.5);
        let center = k[k.len() / 2];
        for &v in &k {
            assert!(center >= v, "center {center} not >= {v}");
        }
    }

    #[test]
    fn test_kernel_zero_sigma_returns_identity() {
        let k = gaussian_kernel_1d(0.0);
        assert_eq!(k.len(), 1);
        assert!((k[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_kernel_negative_sigma_returns_identity() {
        let k = gaussian_kernel_1d(-1.0);
        assert_eq!(k.len(), 1);
        assert!((k[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_blur_uniform_image_unchanged() {
        let w = 8u32;
        let h = 8u32;
        let input: Vec<u8> = (0..(w * h * 4) as usize)
            .map(|i| if i % 4 == 3 { 255 } else { 128 })
            .collect();
        let mut output = vec![0u8; (w * h * 4) as usize];
        gaussian_blur_separable(&input, &mut output, w, h, 1.5).expect("blur should succeed");
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (inp as i32 - out as i32).unsigned_abs() <= 1,
                "pixel {i}: input={inp} output={out}"
            );
        }
    }

    #[test]
    fn test_blur_reduces_contrast() {
        let w = 4u32;
        let h = 4u32;
        let mut input = vec![0u8; (w * h * 4) as usize];
        for row in 0..h as usize {
            for col in 0..w as usize {
                let v = if (row + col) % 2 == 0 { 255u8 } else { 0u8 };
                let base = (row * w as usize + col) * 4;
                input[base] = v;
                input[base + 1] = v;
                input[base + 2] = v;
                input[base + 3] = 255;
            }
        }
        let mut output = vec![0u8; (w * h * 4) as usize];
        gaussian_blur_separable(&input, &mut output, w, h, 1.0).expect("blur should succeed");
        let max_rgb = output
            .chunks(4)
            .flat_map(|px| &px[..3])
            .copied()
            .max()
            .unwrap_or(0);
        assert!(
            max_rgb < 255,
            "max_rgb after blur = {max_rgb}; expected < 255"
        );
    }

    #[test]
    fn test_blur_size_mismatch_returns_error() {
        let w = 4u32;
        let h = 4u32;
        let input = vec![0u8; (w * h * 4) as usize];
        let mut output = vec![0u8; 10];
        let result = gaussian_blur_separable(&input, &mut output, w, h, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_blur_single_pixel_passthrough() {
        let input = vec![100u8, 150u8, 200u8, 255u8];
        let mut output = vec![0u8; 4];
        gaussian_blur_separable(&input, &mut output, 1, 1, 1.0).expect("blur should succeed");
        assert_eq!(output[0], 100);
        assert_eq!(output[1], 150);
        assert_eq!(output[2], 200);
        assert_eq!(output[3], 255);
    }

    // ── Parallel blur tests ───────────────────────────────────────────────────

    #[test]
    fn test_parallel_blur_matches_serial_uniform_image() {
        let w = 16u32;
        let h = 16u32;
        let input: Vec<u8> = vec![128u8; (w * h * 4) as usize];
        let mut serial = vec![0u8; (w * h * 4) as usize];
        let mut parallel = vec![0u8; (w * h * 4) as usize];
        gaussian_blur_separable(&input, &mut serial, w, h, 1.5).expect("serial blur");
        gaussian_blur_separable_parallel(&input, &mut parallel, w, h, 1.5).expect("parallel blur");
        assert_eq!(
            max_channel_diff(&serial, &parallel),
            0,
            "serial and parallel must agree on uniform image"
        );
    }

    #[test]
    fn test_parallel_blur_matches_serial_random_image() {
        let w = 8u32;
        let h = 8u32;
        let input: Vec<u8> = (0..(w * h * 4) as usize)
            .map(|i| ((i * 37 + 13) % 256) as u8)
            .collect();
        let mut serial = vec![0u8; (w * h * 4) as usize];
        let mut parallel = vec![0u8; (w * h * 4) as usize];
        gaussian_blur_separable(&input, &mut serial, w, h, 1.0).expect("serial blur");
        gaussian_blur_separable_parallel(&input, &mut parallel, w, h, 1.0).expect("parallel blur");
        let max_diff = max_channel_diff(&serial, &parallel);
        assert_eq!(max_diff, 0, "serial and parallel outputs must be identical");
    }

    #[test]
    fn test_parallel_blur_single_pixel_passthrough() {
        let input = vec![77u8, 88, 99, 255];
        let mut output = vec![0u8; 4];
        gaussian_blur_separable_parallel(&input, &mut output, 1, 1, 2.0)
            .expect("single pixel parallel blur");
        assert_eq!(output[0], 77);
        assert_eq!(output[1], 88);
        assert_eq!(output[2], 99);
        assert_eq!(output[3], 255);
    }

    #[test]
    fn test_parallel_blur_size_mismatch_returns_error() {
        let input = vec![0u8; 4 * 4 * 4];
        let mut output = vec![0u8; 5]; // wrong
        let res = gaussian_blur_separable_parallel(&input, &mut output, 4, 4, 1.0);
        assert!(res.is_err());
    }

    #[test]
    fn test_parallel_blur_reduces_contrast() {
        let w = 8u32;
        let h = 8u32;
        let mut input = vec![0u8; (w * h * 4) as usize];
        for row in 0..h as usize {
            for col in 0..w as usize {
                let v = if (row + col) % 2 == 0 { 255u8 } else { 0u8 };
                let base = (row * w as usize + col) * 4;
                input[base] = v;
                input[base + 1] = v;
                input[base + 2] = v;
                input[base + 3] = 255;
            }
        }
        let mut output = vec![0u8; (w * h * 4) as usize];
        gaussian_blur_separable_parallel(&input, &mut output, w, h, 1.5)
            .expect("parallel contrast blur");
        let max_rgb = output
            .chunks(4)
            .flat_map(|px| &px[..3])
            .copied()
            .max()
            .unwrap_or(0);
        assert!(
            max_rgb < 255,
            "parallel blur should reduce max brightness; got {max_rgb}"
        );
    }

    #[test]
    fn test_parallel_blur_large_sigma_heavy_smoothing() {
        let w = 16u32;
        let h = 16u32;
        // Alternating black/white checkerboard
        let input: Vec<u8> = (0..(w * h) as usize)
            .flat_map(|i| {
                let row = i / w as usize;
                let col = i % w as usize;
                let v = if (row + col) % 2 == 0 { 255u8 } else { 0u8 };
                [v, v, v, 255u8]
            })
            .collect();
        let mut out_small = vec![0u8; (w * h * 4) as usize];
        let mut out_large = vec![0u8; (w * h * 4) as usize];
        gaussian_blur_separable_parallel(&input, &mut out_small, w, h, 0.5).expect("small sigma");
        gaussian_blur_separable_parallel(&input, &mut out_large, w, h, 3.0).expect("large sigma");

        let range_small: u32 = out_small
            .chunks(4)
            .map(|px| px[0] as u32)
            .max()
            .unwrap_or(0)
            - out_small
                .chunks(4)
                .map(|px| px[0] as u32)
                .min()
                .unwrap_or(0);
        let range_large: u32 = out_large
            .chunks(4)
            .map(|px| px[0] as u32)
            .max()
            .unwrap_or(0)
            - out_large
                .chunks(4)
                .map(|px| px[0] as u32)
                .min()
                .unwrap_or(0);
        assert!(
            range_large <= range_small,
            "larger sigma should produce smaller contrast range; small={range_small}, large={range_large}"
        );
    }

    #[test]
    fn test_parallel_blur_wide_image() {
        let w = 32u32;
        let h = 4u32;
        let input: Vec<u8> = (0..(w * h * 4) as usize).map(|i| (i % 256) as u8).collect();
        let mut output = vec![0u8; (w * h * 4) as usize];
        gaussian_blur_separable_parallel(&input, &mut output, w, h, 1.0)
            .expect("wide image parallel blur");
        assert_eq!(output.len(), (w * h * 4) as usize);
    }

    #[test]
    fn test_parallel_blur_tall_image() {
        let w = 4u32;
        let h = 32u32;
        let input: Vec<u8> = (0..(w * h * 4) as usize).map(|i| (i % 256) as u8).collect();
        let mut output = vec![0u8; (w * h * 4) as usize];
        gaussian_blur_separable_parallel(&input, &mut output, w, h, 1.0)
            .expect("tall image parallel blur");
        assert_eq!(output.len(), (w * h * 4) as usize);
    }

    #[test]
    fn test_max_channel_diff_identical() {
        let a = vec![128u8; 16];
        let diff = max_channel_diff(&a, &a);
        assert_eq!(diff, 0);
    }

    #[test]
    fn test_max_channel_diff_known_values() {
        let a = vec![100u8, 200, 50, 255];
        let b = vec![90u8, 210, 50, 255];
        let diff = max_channel_diff(&a, &b);
        assert_eq!(diff, 10);
    }

    // ── box_blur tests ────────────────────────────────────────────────────────

    #[test]
    fn test_box_blur_uniform() {
        // A 4×4 image filled with value 128 (3 RGB channels) should pass through
        // unchanged — the box average of identical values is the same value.
        let w = 4u32;
        let h = 4u32;
        let ch = 3u32;
        let value: u8 = 128;
        let input = vec![value; (w * h * ch) as usize];
        let output = box_blur(&input, w, h, ch, 2).expect("box_blur should succeed");
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v as i32 - value as i32).abs() <= 1,
                "pixel byte {i}: expected {value}, got {v}"
            );
        }
    }

    #[test]
    fn test_box_blur_spike() {
        // 7×7 dark image (value 0) with a single bright pixel at the centre.
        // After box blur with radius=1 the centre value should decrease and its
        // 8-neighbours should become > 0.
        let w = 7u32;
        let h = 7u32;
        let ch = 1u32;
        let mut input = vec![0u8; (w * h * ch) as usize];
        let cx = 3usize;
        let cy = 3usize;
        input[cy * w as usize + cx] = 255;

        let output = box_blur(&input, w, h, ch, 1).expect("box_blur spike should succeed");

        // Centre must be reduced.
        let centre = output[cy * w as usize + cx];
        assert!(
            centre < 255,
            "centre pixel should be reduced after box blur, got {centre}"
        );

        // At least one immediate neighbour must be > 0.
        let right = output[cy * w as usize + cx + 1];
        let below = output[(cy + 1) * w as usize + cx];
        assert!(
            right > 0 || below > 0,
            "neighbours should receive energy; right={right}, below={below}"
        );
    }

    #[test]
    fn test_box_blur_size_mismatch_returns_error() {
        // Buffer length that does not match w * h * ch should return an error.
        let result = box_blur(&[0u8; 10], 4, 4, 1, 1);
        assert!(result.is_err(), "expected error on size mismatch");
    }

    // ── median_filter tests ───────────────────────────────────────────────────

    #[test]
    fn test_median_removes_outlier() {
        // 5×5 single-channel image where all pixels are 100 except the centre,
        // which is 255. Median with radius=1 over a 3×3 window of 100-values
        // should remove the outlier (the median of [100,100,...,255] is 100).
        let w = 5u32;
        let h = 5u32;
        let ch = 1u32;
        let mut input = vec![100u8; (w * h * ch) as usize];
        let cx = 2usize;
        let cy = 2usize;
        input[cy * w as usize + cx] = 255; // outlier

        let output = median_filter(&input, w, h, ch, 1).expect("median_filter should succeed");

        let centre = output[cy * w as usize + cx];
        assert_eq!(
            centre, 100,
            "median should remove the outlier; centre={centre}"
        );
    }

    #[test]
    fn test_median_uniform_image() {
        // Uniform image must be preserved exactly.
        let w = 4u32;
        let h = 4u32;
        let ch = 4u32;
        let input = vec![77u8; (w * h * ch) as usize];
        let output = median_filter(&input, w, h, ch, 2).expect("median_filter uniform");
        assert!(output.iter().all(|&v| v == 77));
    }

    #[test]
    fn test_median_size_mismatch_returns_error() {
        let result = median_filter(&[0u8; 5], 4, 4, 1, 1);
        assert!(result.is_err(), "expected error on size mismatch");
    }

    // ── bilateral_filter tests ────────────────────────────────────────────────

    #[test]
    fn test_bilateral_edge_preserving() {
        // Left half of a 10×10 RGBA image is black (0), right half is white (255).
        // Bilateral filter with a large sigma_range should preserve the edge:
        // pixels well away from the boundary should stay close to their original value.
        let w = 10u32;
        let h = 10u32;
        let mut input = vec![0u8; (w * h * 4) as usize];
        for row in 0..h as usize {
            for col in 0..w as usize {
                let v: u8 = if col >= 5 { 255 } else { 0 };
                let base = (row * w as usize + col) * 4;
                input[base] = v;
                input[base + 1] = v;
                input[base + 2] = v;
                input[base + 3] = 255;
            }
        }

        // sigma_spatial=2 (small neighbourhood), sigma_range=10 (tight range gate
        // ⇒ edge preserved well).
        let output =
            bilateral_filter(&input, w, h, 4, 2.0, 10.0).expect("bilateral_filter should succeed");

        // Pixels in the interior of the black half should remain close to 0.
        for row in 0..h as usize {
            let col = 1usize; // well inside black half
            let base = (row * w as usize + col) * 4;
            for c in 0..3 {
                assert!(
                    output[base + c] < 64,
                    "row={row} col={col} ch={c}: expected near 0, got {}",
                    output[base + c]
                );
            }
        }

        // Pixels in the interior of the white half should remain close to 255.
        for row in 0..h as usize {
            let col = 8usize; // well inside white half
            let base = (row * w as usize + col) * 4;
            for c in 0..3 {
                assert!(
                    output[base + c] > 191,
                    "row={row} col={col} ch={c}: expected near 255, got {}",
                    output[base + c]
                );
            }
        }
    }

    #[test]
    fn test_bilateral_wrong_channels_returns_error() {
        let result = bilateral_filter(&[0u8; 9], 3, 3, 1, 2.0, 30.0);
        assert!(result.is_err(), "bilateral requires channels == 4");
    }
}
