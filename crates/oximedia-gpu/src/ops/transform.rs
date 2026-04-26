//! Transform operations (DCT, FFT) for frequency domain processing

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
struct TransformParams {
    width: u32,
    height: u32,
    block_size: u32,
    transform_type: u32,
    stride: u32,
    is_inverse: u32,
    padding1: u32,
    padding2: u32,
}

/// Transform operations for frequency domain processing
pub struct TransformOperation;

impl TransformOperation {
    /// Compute 2D DCT (Discrete Cosine Transform)
    ///
    /// Computes the forward DCT on 8x8 blocks. Input dimensions must be
    /// multiples of 8.
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - Input data (f32 values)
    /// * `output` - Output DCT coefficients
    /// * `width` - Data width (must be multiple of 8)
    /// * `height` - Data height (must be multiple of 8)
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or if the GPU operation fails.
    pub fn dct_2d(
        device: &GpuDevice,
        input: &[f32],
        output: &mut [f32],
        width: u32,
        height: u32,
    ) -> Result<()> {
        if width % 8 != 0 || height % 8 != 0 {
            return Err(GpuError::InvalidDimensions { width, height });
        }

        utils::validate_dimensions(width, height)?;

        let expected_size = (width * height) as usize;
        if input.len() < expected_size || output.len() < expected_size {
            return Err(GpuError::InvalidBufferSize {
                expected: expected_size,
                actual: input.len().min(output.len()),
            });
        }

        let pipeline = Self::get_dct_8x8_pipeline(device)?;
        let layout = Self::get_bind_group_layout(device)?;

        Self::execute_transform(
            device, pipeline, layout, input, output, width, height, 8, 0, // DCT
        )
    }

    /// Compute 2D IDCT (Inverse Discrete Cosine Transform)
    ///
    /// Computes the inverse DCT on 8x8 blocks. Input dimensions must be
    /// multiples of 8.
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - Input DCT coefficients
    /// * `output` - Output reconstructed data
    /// * `width` - Data width (must be multiple of 8)
    /// * `height` - Data height (must be multiple of 8)
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or if the GPU operation fails.
    pub fn idct_2d(
        device: &GpuDevice,
        input: &[f32],
        output: &mut [f32],
        width: u32,
        height: u32,
    ) -> Result<()> {
        if width % 8 != 0 || height % 8 != 0 {
            return Err(GpuError::InvalidDimensions { width, height });
        }

        utils::validate_dimensions(width, height)?;

        let expected_size = (width * height) as usize;
        if input.len() < expected_size || output.len() < expected_size {
            return Err(GpuError::InvalidBufferSize {
                expected: expected_size,
                actual: input.len().min(output.len()),
            });
        }

        let pipeline = Self::get_idct_8x8_pipeline(device)?;
        let layout = Self::get_bind_group_layout(device)?;

        Self::execute_transform(
            device, pipeline, layout, input, output, width, height, 8, 1, // IDCT
        )
    }

    /// Compute general 2D DCT using row-column decomposition
    ///
    /// This method works for any dimensions, not just multiples of 8.
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - Input data (f32 values)
    /// * `output` - Output DCT coefficients
    /// * `width` - Data width
    /// * `height` - Data height
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or if the GPU operation fails.
    pub fn dct_2d_general(
        device: &GpuDevice,
        input: &[f32],
        output: &mut [f32],
        width: u32,
        height: u32,
    ) -> Result<()> {
        utils::validate_dimensions(width, height)?;

        let expected_size = (width * height) as usize;
        if input.len() < expected_size || output.len() < expected_size {
            return Err(GpuError::InvalidBufferSize {
                expected: expected_size,
                actual: input.len().min(output.len()),
            });
        }

        // Two-pass DCT: row then column
        let mut temp = vec![0.0f32; expected_size];

        // Row DCT
        let row_pipeline = Self::get_dct_row_pipeline(device)?;
        let layout = Self::get_bind_group_layout(device)?;

        Self::execute_transform(
            device,
            row_pipeline,
            layout,
            input,
            &mut temp,
            width,
            height,
            width,
            0,
        )?;

        // Column DCT
        let col_pipeline = Self::get_dct_col_pipeline(device)?;

        Self::execute_transform(
            device,
            col_pipeline,
            layout,
            &temp,
            output,
            width,
            height,
            height,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_transform(
        device: &GpuDevice,
        pipeline: &ComputePipeline,
        layout: &BindGroupLayout,
        input: &[f32],
        output: &mut [f32],
        width: u32,
        height: u32,
        block_size: u32,
        transform_type: u32,
    ) -> Result<()> {
        let input_bytes = bytemuck::cast_slice(input);
        let output_size = std::mem::size_of_val(output);

        // Create buffers
        let input_buffer = utils::create_storage_buffer(device, input_bytes.len() as u64)?;
        let output_buffer = utils::create_storage_buffer(device, output_size as u64)?;

        // Upload input data
        device
            .queue()
            .write_buffer(input_buffer.buffer(), 0, input_bytes);

        // Create uniform buffer for parameters
        let params = TransformParams {
            width,
            height,
            block_size,
            transform_type,
            stride: width,
            is_inverse: 0,
            padding1: 0,
            padding2: 0,
        };
        let params_bytes = bytemuck::bytes_of(&params);
        let params_buffer = utils::create_uniform_buffer(device, params_bytes)?;

        // Create bind group
        let compiler = ShaderCompiler::new(device);
        let bind_group = compiler.create_bind_group(
            "Transform Bind Group",
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
        Self::dispatch_compute(device, pipeline, &bind_group, width, height, block_size)?;

        // Read back results
        let readback_buffer = utils::create_readback_buffer(device, output_size as u64)?;
        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Transform Copy Encoder"),
            });

        output_buffer.copy_to(&mut encoder, &readback_buffer, 0, 0, output_size as u64)?;

        device.queue().submit(Some(encoder.finish()));
        device.wait();

        let result = readback_buffer.read(device, 0, output_size as u64)?;
        let result_f32: &[f32] = bytemuck::cast_slice(&result);
        output.copy_from_slice(result_f32);

        Ok(())
    }

    fn dispatch_compute(
        device: &GpuDevice,
        pipeline: &ComputePipeline,
        bind_group: &BindGroup,
        width: u32,
        height: u32,
        block_size: u32,
    ) -> Result<()> {
        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Transform Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Transform Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);

            if block_size == 8 {
                // For 8x8 DCT, dispatch one workgroup per block
                let dispatch_x = width / 8;
                let dispatch_y = height / 8;
                compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            } else {
                // For row/column transforms
                let total_elements = width * height;
                let dispatch = total_elements.div_ceil(256);
                compute_pass.dispatch_workgroups(dispatch, 1, 1);
            }
        }

        device.queue().submit(Some(encoder.finish()));
        Ok(())
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

            compiler.create_bind_group_layout("Transform Bind Group Layout", &entries)
        }))
    }

    fn init_pipeline(
        device: &GpuDevice,
        name: &str,
        entry_point: &str,
    ) -> std::result::Result<ComputePipeline, String> {
        let compiler = ShaderCompiler::new(device);
        let shader = compiler
            .compile(
                "Transform Shader",
                ShaderSource::Embedded(crate::shader::embedded::TRANSFORM_SHADER),
            )
            .map_err(|e| format!("Failed to compile transform shader: {e}"))?;

        let layout = Self::get_bind_group_layout(device)
            .map_err(|e| format!("Failed to create bind group layout: {e}"))?;

        compiler
            .create_pipeline(name, &shader, entry_point, layout)
            .map_err(|e| format!("Failed to create pipeline: {e}"))
    }

    fn get_dct_8x8_pipeline(device: &GpuDevice) -> Result<&'static ComputePipeline> {
        static PIPELINE: OnceCell<std::result::Result<ComputePipeline, String>> = OnceCell::new();

        PIPELINE
            .get_or_init(|| {
                TransformOperation::init_pipeline(device, "DCT 8x8 Pipeline", "dct_8x8")
            })
            .as_ref()
            .map_err(|e| crate::GpuError::PipelineCreation(e.clone()))
    }

    fn get_idct_8x8_pipeline(device: &GpuDevice) -> Result<&'static ComputePipeline> {
        static PIPELINE: OnceCell<std::result::Result<ComputePipeline, String>> = OnceCell::new();

        PIPELINE
            .get_or_init(|| {
                TransformOperation::init_pipeline(device, "IDCT 8x8 Pipeline", "idct_8x8")
            })
            .as_ref()
            .map_err(|e| crate::GpuError::PipelineCreation(e.clone()))
    }

    fn get_dct_row_pipeline(device: &GpuDevice) -> Result<&'static ComputePipeline> {
        static PIPELINE: OnceCell<std::result::Result<ComputePipeline, String>> = OnceCell::new();

        PIPELINE
            .get_or_init(|| {
                TransformOperation::init_pipeline(device, "DCT Row Pipeline", "dct_row")
            })
            .as_ref()
            .map_err(|e| crate::GpuError::PipelineCreation(e.clone()))
    }

    fn get_dct_col_pipeline(device: &GpuDevice) -> Result<&'static ComputePipeline> {
        static PIPELINE: OnceCell<std::result::Result<ComputePipeline, String>> = OnceCell::new();

        PIPELINE
            .get_or_init(|| {
                TransformOperation::init_pipeline(device, "DCT Column Pipeline", "dct_col")
            })
            .as_ref()
            .map_err(|e| crate::GpuError::PipelineCreation(e.clone()))
    }
}

// =============================================================================
// CPU-side perspective transform and lens distortion correction (Task 8)
// =============================================================================

/// A 3×3 homogeneous perspective (projective) transform matrix stored in
/// row-major order.
///
/// The matrix maps homogeneous image coordinates `(x, y, 1)ᵀ` to new
/// coordinates via `(x', y', w')ᵀ = M · (x, y, 1)ᵀ`.  The Cartesian result
/// is `(x'/w', y'/w')`.
#[derive(Debug, Clone, Copy)]
pub struct PerspectiveMatrix {
    /// Row-major 3×3 elements: `[[a,b,c],[d,e,f],[g,h,i]]`.
    pub data: [[f64; 3]; 3],
}

impl PerspectiveMatrix {
    /// Create from a flat row-major array of 9 elements.
    #[must_use]
    pub fn from_array(m: [f64; 9]) -> Self {
        Self {
            data: [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]],
        }
    }

    /// Identity perspective matrix (no transform).
    #[must_use]
    pub fn identity() -> Self {
        Self::from_array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    }

    /// Apply this matrix to a point `(x, y)` and return the projected result.
    ///
    /// Returns `None` if the homogeneous weight `w` is too close to zero
    /// (the point maps to infinity).
    #[must_use]
    pub fn project(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let m = &self.data;
        let x_h = m[0][0] * x + m[0][1] * y + m[0][2];
        let y_h = m[1][0] * x + m[1][1] * y + m[1][2];
        let w = m[2][0] * x + m[2][1] * y + m[2][2];
        if w.abs() < 1e-12 {
            return None;
        }
        Some((x_h / w, y_h / w))
    }

    /// Compute the inverse of this matrix using Cramer's rule.
    ///
    /// Returns `None` if the matrix is singular.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let m = &self.data;
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if det.abs() < 1e-15 {
            return None;
        }
        let inv_det = 1.0 / det;
        let inv = [
            [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
            ],
            [
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
            ],
            [
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
            ],
        ];
        Some(Self { data: inv })
    }
}

impl Default for PerspectiveMatrix {
    fn default() -> Self {
        Self::identity()
    }
}

/// Parameters for radial + tangential lens distortion (Brown-Conrady model).
///
/// This is the same model used by OpenCV and is compatible with camera
/// calibration output from standard photogrammetry tools.
#[derive(Debug, Clone, Copy)]
pub struct LensDistortionParams {
    /// Radial distortion coefficient k₁ (typically small, e.g. -0.3 to +0.5).
    pub k1: f64,
    /// Radial distortion coefficient k₂.
    pub k2: f64,
    /// Radial distortion coefficient k₃.
    pub k3: f64,
    /// Tangential distortion coefficient p₁.
    pub p1: f64,
    /// Tangential distortion coefficient p₂.
    pub p2: f64,
    /// Focal length in pixels along the X axis.
    pub fx: f64,
    /// Focal length in pixels along the Y axis.
    pub fy: f64,
    /// Principal point X coordinate (typically `width / 2`).
    pub cx: f64,
    /// Principal point Y coordinate (typically `height / 2`).
    pub cy: f64,
}

impl LensDistortionParams {
    /// Create a default (no distortion) parameter set for an image of the
    /// given `width × height`.
    #[must_use]
    pub fn no_distortion(width: u32, height: u32) -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            fx: f64::from(width),
            fy: f64::from(height),
            cx: f64::from(width) / 2.0,
            cy: f64::from(height) / 2.0,
        }
    }
}

/// CPU-parallel perspective warp of a packed RGBA image.
///
/// Uses inverse mapping with bilinear interpolation: for each destination
/// pixel `(dx, dy)` the inverse homography maps it back to the source
/// coordinates `(sx, sy)`, which are bilinearly sampled.
///
/// Pixels that map outside the source image are filled with `fill_rgba`.
///
/// # Errors
///
/// Returns [`crate::GpuError::InvalidDimensions`] for zero dimensions or
/// [`crate::GpuError::InvalidBufferSize`] for buffer/dimension mismatches.
pub fn perspective_warp(
    input: &[u8],
    src_width: u32,
    src_height: u32,
    output: &mut [u8],
    dst_width: u32,
    dst_height: u32,
    matrix: &PerspectiveMatrix,
    fill_rgba: [u8; 4],
) -> crate::Result<()> {
    use super::utils;
    use crate::GpuError;

    if src_width == 0 || src_height == 0 {
        return Err(GpuError::InvalidDimensions {
            width: src_width,
            height: src_height,
        });
    }
    if dst_width == 0 || dst_height == 0 {
        return Err(GpuError::InvalidDimensions {
            width: dst_width,
            height: dst_height,
        });
    }
    utils::validate_buffer_size(input, src_width, src_height, 4)?;
    utils::validate_buffer_size(output, dst_width, dst_height, 4)?;

    let inv = matrix
        .inverse()
        .ok_or_else(|| GpuError::Internal("Perspective matrix is singular".to_string()))?;

    let sw = src_width as usize;
    let sh = src_height as usize;
    let dw = dst_width as usize;
    let dh = dst_height as usize;

    for dy in 0..dh {
        for dx in 0..dw {
            let dst_idx = (dy * dw + dx) * 4;
            let Some((sx_f, sy_f)) = inv.project(dx as f64, dy as f64) else {
                output[dst_idx..dst_idx + 4].copy_from_slice(&fill_rgba);
                continue;
            };

            // Bilinear interpolation
            let x0 = sx_f.floor() as isize;
            let y0 = sy_f.floor() as isize;
            let x1 = x0 + 1;
            let y1 = y0 + 1;
            let fx = sx_f - sx_f.floor();
            let fy = sy_f - sy_f.floor();

            let sample = |cx: isize, cy: isize| -> [f64; 4] {
                if cx < 0 || cy < 0 || cx >= sw as isize || cy >= sh as isize {
                    [
                        fill_rgba[0] as f64,
                        fill_rgba[1] as f64,
                        fill_rgba[2] as f64,
                        fill_rgba[3] as f64,
                    ]
                } else {
                    let idx = (cy as usize * sw + cx as usize) * 4;
                    [
                        input[idx] as f64,
                        input[idx + 1] as f64,
                        input[idx + 2] as f64,
                        input[idx + 3] as f64,
                    ]
                }
            };

            let p00 = sample(x0, y0);
            let p10 = sample(x1, y0);
            let p01 = sample(x0, y1);
            let p11 = sample(x1, y1);

            for c in 0..4 {
                let v = p00[c] * (1.0 - fx) * (1.0 - fy)
                    + p10[c] * fx * (1.0 - fy)
                    + p01[c] * (1.0 - fx) * fy
                    + p11[c] * fx * fy;
                output[dst_idx + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(())
}

/// CPU-side lens distortion correction using the Brown-Conrady model.
///
/// For each destination pixel `(x, y)` the distortion model computes the
/// corresponding distorted source coordinate and bilinearly samples the input.
/// Pixels that map outside the source image are filled with `fill_rgba`.
///
/// # Errors
///
/// Returns an error if dimensions are zero or buffers are the wrong size.
pub fn lens_undistort(
    input: &[u8],
    width: u32,
    height: u32,
    output: &mut [u8],
    params: &LensDistortionParams,
    fill_rgba: [u8; 4],
) -> crate::Result<()> {
    use super::utils;
    use crate::GpuError;

    if width == 0 || height == 0 {
        return Err(GpuError::InvalidDimensions { width, height });
    }
    utils::validate_buffer_size(input, width, height, 4)?;
    utils::validate_buffer_size(output, width, height, 4)?;

    let w = width as usize;
    let h = height as usize;
    let inv_fx = 1.0 / params.fx;
    let inv_fy = 1.0 / params.fy;

    for dy in 0..h {
        for dx in 0..w {
            // Normalised coordinates (undistorted space).
            let x_u = (dx as f64 - params.cx) * inv_fx;
            let y_u = (dy as f64 - params.cy) * inv_fy;

            // Apply Brown-Conrady radial + tangential distortion to map from
            // undistorted → distorted (where the actual sensor data lives).
            let r2 = x_u * x_u + y_u * y_u;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let radial = 1.0 + params.k1 * r2 + params.k2 * r4 + params.k3 * r6;
            let x_d =
                x_u * radial + 2.0 * params.p1 * x_u * y_u + params.p2 * (r2 + 2.0 * x_u * x_u);
            let y_d =
                y_u * radial + params.p1 * (r2 + 2.0 * y_u * y_u) + 2.0 * params.p2 * x_u * y_u;

            // Back to pixel coordinates in the distorted (source) image.
            let sx_f = x_d * params.fx + params.cx;
            let sy_f = y_d * params.fy + params.cy;

            let dst_idx = (dy * w + dx) * 4;

            let x0 = sx_f.floor() as isize;
            let y0 = sy_f.floor() as isize;
            let x1 = x0 + 1;
            let y1 = y0 + 1;
            let fx = sx_f - sx_f.floor();
            let fy = sy_f - sy_f.floor();

            let sample = |cx: isize, cy: isize| -> [f64; 4] {
                if cx < 0 || cy < 0 || cx >= w as isize || cy >= h as isize {
                    [
                        fill_rgba[0] as f64,
                        fill_rgba[1] as f64,
                        fill_rgba[2] as f64,
                        fill_rgba[3] as f64,
                    ]
                } else {
                    let idx = (cy as usize * w + cx as usize) * 4;
                    [
                        input[idx] as f64,
                        input[idx + 1] as f64,
                        input[idx + 2] as f64,
                        input[idx + 3] as f64,
                    ]
                }
            };

            let p00 = sample(x0, y0);
            let p10 = sample(x1, y0);
            let p01 = sample(x0, y1);
            let p11 = sample(x1, y1);

            for c in 0..4 {
                let v = p00[c] * (1.0 - fx) * (1.0 - fy)
                    + p10[c] * fx * (1.0 - fy)
                    + p01[c] * (1.0 - fx) * fy
                    + p11[c] * fx * fy;
                output[dst_idx + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(())
}

// =============================================================================
// CPU-side geometric (pixel-level) transforms (rotate, flip, transpose)
// =============================================================================

impl TransformOperation {
    /// Copy one pixel from `src` to `dst`, using interleaved layout.
    ///
    /// All coordinates are 0-indexed.  `src_w` is the *source* image width and
    /// `dst_w` is the *destination* image width (both in pixels).  `ch` is the
    /// number of bytes per pixel.
    #[inline]
    fn copy_pixel(
        src: &[u8],
        dst: &mut [u8],
        src_x: u32,
        src_y: u32,
        dst_x: u32,
        dst_y: u32,
        src_w: u32,
        dst_w: u32,
        ch: u32,
    ) {
        let src_off = ((src_y * src_w + src_x) * ch) as usize;
        let dst_off = ((dst_y * dst_w + dst_x) * ch) as usize;
        dst[dst_off..dst_off + ch as usize].copy_from_slice(&src[src_off..src_off + ch as usize]);
    }

    /// Rotate an interleaved pixel image 90° clockwise.
    ///
    /// Output dimensions are swapped: `out_width = height`, `out_height = width`.
    ///
    /// Pixel mapping: `output(x, y) = input(y, width_out - 1 - x)` where
    /// `width_out = height`.
    ///
    /// # Arguments
    ///
    /// * `data` – packed pixel buffer (interleaved, `channels` bytes per pixel)
    /// * `width` – source image width in pixels
    /// * `height` – source image height in pixels
    /// * `channels` – bytes per pixel (e.g. 3 for RGB, 4 for RGBA)
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `data.len() != width * height * channels`.
    #[must_use]
    pub fn rotate90(data: &[u8], width: u32, height: u32, channels: u32) -> Vec<u8> {
        // After 90° CW: out_width = in_height, out_height = in_width
        let out_width = height;
        let out_height = width;
        let mut out = vec![0u8; (out_width * out_height * channels) as usize];

        for src_y in 0..height {
            for src_x in 0..width {
                // 90° CW: dst_x = height - 1 - src_y ... wait, let's derive carefully.
                // Clockwise 90°: new_x = (in_height - 1 - src_y) is wrong.
                // The standard derivation for CW 90°:
                //   src (col=x, row=y) → dst (col=height-1-y, row=x)
                //   i.e. dst_x = src_y, dst_y = (width - 1 - src_x)
                // Verify: src(0,0) → dst_x=0, dst_y=width-1  ← top-left goes to bottom-left of output
                // That matches CW rotation where (0,0) ends at bottom-left of the output.
                // Actually let's verify with a 3x1 image rotated 90° CW:
                //   Input (width=3, height=1):  [A B C] (row 0)
                //   Output (width=1, height=3):  col 0: row0=C, row1=B, row2=A
                //   So output pixel at (x=0, y=0) = input(width-1-0, 0) = input(2,0)=C ✓
                //   using: dst_x=src_y, dst_y=(in_width-1-src_x):
                //     src(0,0): dst_x=0, dst_y=2 → output(0,2)=A ✓ (A at row2)
                //     src(1,0): dst_x=0, dst_y=1 → output(0,1)=B ✓
                //     src(2,0): dst_x=0, dst_y=0 → output(0,0)=C ✓
                let dst_x = src_y;
                let dst_y = width - 1 - src_x;
                Self::copy_pixel(
                    data, &mut out, src_x, src_y, dst_x, dst_y, width, out_width, channels,
                );
            }
        }

        out
    }

    /// Rotate an interleaved pixel image 180°.
    ///
    /// Output dimensions are the same as input.
    ///
    /// Pixel mapping: `output(x, y) = input(width-1-x, height-1-y)`.
    #[must_use]
    pub fn rotate180(data: &[u8], width: u32, height: u32, channels: u32) -> Vec<u8> {
        let mut out = vec![0u8; (width * height * channels) as usize];

        for src_y in 0..height {
            for src_x in 0..width {
                let dst_x = width - 1 - src_x;
                let dst_y = height - 1 - src_y;
                Self::copy_pixel(
                    data, &mut out, src_x, src_y, dst_x, dst_y, width, width, channels,
                );
            }
        }

        out
    }

    /// Rotate an interleaved pixel image 270° clockwise (= 90° counter-clockwise).
    ///
    /// Output dimensions are swapped: `out_width = height`, `out_height = width`.
    ///
    /// Pixel mapping: `output(x, y) = input(height-1-y, x)`.
    #[must_use]
    pub fn rotate270(data: &[u8], width: u32, height: u32, channels: u32) -> Vec<u8> {
        // After 270° CW (= 90° CCW): out_width = in_height, out_height = in_width
        let out_width = height;
        let out_height = width;
        let mut out = vec![0u8; (out_width * out_height * channels) as usize];

        for src_y in 0..height {
            for src_x in 0..width {
                // 270° CW derivation:
                //   src (col=x, row=y) → dst (col=in_height-1-src_y, row=src_x)
                //   i.e. dst_x = height-1-src_y, dst_y = src_x
                // Verify with 3x1 (width=3, height=1) rotated 270° CW:
                //   Output (width=1, height=3): row0=A, row1=B, row2=C
                //   src(0,0): dst_x=0, dst_y=0 → output(0,0)=A ✓
                //   src(1,0): dst_x=0, dst_y=1 → output(0,1)=B ✓
                //   src(2,0): dst_x=0, dst_y=2 → output(0,2)=C ✓
                let dst_x = height - 1 - src_y;
                let dst_y = src_x;
                Self::copy_pixel(
                    data, &mut out, src_x, src_y, dst_x, dst_y, width, out_width, channels,
                );
            }
        }

        out
    }

    /// Flip an interleaved pixel image horizontally (mirror left-right).
    ///
    /// Output dimensions are the same as input.
    ///
    /// Pixel mapping: `output(x, y) = input(width-1-x, y)`.
    #[must_use]
    pub fn flip_horizontal(data: &[u8], width: u32, height: u32, channels: u32) -> Vec<u8> {
        let mut out = vec![0u8; (width * height * channels) as usize];

        for src_y in 0..height {
            for src_x in 0..width {
                let dst_x = width - 1 - src_x;
                Self::copy_pixel(
                    data, &mut out, src_x, src_y, dst_x, src_y, width, width, channels,
                );
            }
        }

        out
    }

    /// Flip an interleaved pixel image vertically (mirror top-bottom).
    ///
    /// Output dimensions are the same as input.
    ///
    /// Pixel mapping: `output(x, y) = input(x, height-1-y)`.
    #[must_use]
    pub fn flip_vertical(data: &[u8], width: u32, height: u32, channels: u32) -> Vec<u8> {
        let mut out = vec![0u8; (width * height * channels) as usize];

        for src_y in 0..height {
            for src_x in 0..width {
                let dst_y = height - 1 - src_y;
                Self::copy_pixel(
                    data, &mut out, src_x, src_y, src_x, dst_y, width, width, channels,
                );
            }
        }

        out
    }

    /// Transpose an interleaved pixel image (swap x and y axes).
    ///
    /// Output dimensions are swapped: `out_width = height`, `out_height = width`.
    ///
    /// Pixel mapping: `output(x, y) = input(y, x)`.
    #[must_use]
    pub fn transpose(data: &[u8], width: u32, height: u32, channels: u32) -> Vec<u8> {
        // After transpose: out_width = in_height, out_height = in_width
        let out_width = height;
        let out_height = width;
        let mut out = vec![0u8; (out_width * out_height * channels) as usize];

        for src_y in 0..height {
            for src_x in 0..width {
                // output(x, y) = input(y, x)
                // dst_x = src_y, dst_y = src_x
                let dst_x = src_y;
                let dst_y = src_x;
                Self::copy_pixel(
                    data, &mut out, src_x, src_y, dst_x, dst_y, width, out_width, channels,
                );
            }
        }

        out
    }
}

// =============================================================================
// Tests for perspective transform and lens distortion (Task 8)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let n = (w * h * 4) as usize;
        let mut v = vec![0u8; n];
        for px in v.chunks_exact_mut(4) {
            px[0] = r;
            px[1] = g;
            px[2] = b;
            px[3] = a;
        }
        v
    }

    // ── PerspectiveMatrix ─────────────────────────────────────────────────────

    #[test]
    fn test_perspective_identity_project() {
        let m = PerspectiveMatrix::identity();
        let (x, y) = m
            .project(100.0, 200.0)
            .expect("identity must not return None");
        assert!((x - 100.0).abs() < 1e-10, "x={x}");
        assert!((y - 200.0).abs() < 1e-10, "y={y}");
    }

    #[test]
    fn test_perspective_translation() {
        // Pure translation: shift by (10, 20).
        let m = PerspectiveMatrix::from_array([1.0, 0.0, 10.0, 0.0, 1.0, 20.0, 0.0, 0.0, 1.0]);
        let (x, y) = m.project(5.0, 5.0).expect("no infinity");
        assert!((x - 15.0).abs() < 1e-10, "x={x}");
        assert!((y - 25.0).abs() < 1e-10, "y={y}");
    }

    #[test]
    fn test_perspective_inverse_is_correct() {
        let m = PerspectiveMatrix::from_array([1.0, 0.5, 10.0, -0.2, 1.0, 5.0, 0.001, 0.0, 1.0]);
        let inv = m.inverse().expect("non-singular matrix must have inverse");
        // m · inv(m) ≈ identity
        let (x_orig, y_orig) = (50.0_f64, 30.0_f64);
        let (x_proj, y_proj) = m.project(x_orig, y_orig).expect("forward project");
        let (x_back, y_back) = inv.project(x_proj, y_proj).expect("inverse project");
        assert!(
            (x_back - x_orig).abs() < 1e-6,
            "x roundtrip: {x_back} ≠ {x_orig}"
        );
        assert!(
            (y_back - y_orig).abs() < 1e-6,
            "y roundtrip: {y_back} ≠ {y_orig}"
        );
    }

    #[test]
    fn test_perspective_singular_returns_none_inverse() {
        // All-zero matrix is singular.
        let m = PerspectiveMatrix::from_array([0.0; 9]);
        assert!(m.inverse().is_none(), "singular matrix must return None");
    }

    // ── perspective_warp ──────────────────────────────────────────────────────

    #[test]
    fn test_perspective_warp_identity_preserves_image() {
        let w = 8u32;
        let h = 8u32;
        let src = solid_rgba(w, h, 100, 150, 200, 255);
        let mut dst = vec![0u8; (w * h * 4) as usize];
        perspective_warp(
            &src,
            w,
            h,
            &mut dst,
            w,
            h,
            &PerspectiveMatrix::identity(),
            [0, 0, 0, 0],
        )
        .expect("identity warp must succeed");
        // Every pixel must match the source (within bilinear rounding).
        for (s, d) in src.iter().zip(dst.iter()) {
            assert!(
                (*s as i32 - *d as i32).unsigned_abs() <= 1,
                "identity warp mismatch"
            );
        }
    }

    #[test]
    fn test_perspective_warp_out_of_bounds_uses_fill() {
        let w = 4u32;
        let h = 4u32;
        let src = solid_rgba(w, h, 255, 0, 0, 255);
        let mut dst = vec![0u8; (w * h * 4) as usize];
        // Large translation sends all destination pixels outside the source.
        let m =
            PerspectiveMatrix::from_array([1.0, 0.0, 10000.0, 0.0, 1.0, 10000.0, 0.0, 0.0, 1.0]);
        perspective_warp(&src, w, h, &mut dst, w, h, &m, [0, 255, 0, 255])
            .expect("warp must succeed");
        // All pixels should be fill colour (green).
        for i in 0..(w * h) as usize {
            assert_eq!(dst[i * 4 + 1], 255, "fill green channel mismatch");
        }
    }

    #[test]
    fn test_perspective_warp_invalid_dims_return_error() {
        let src = solid_rgba(4, 4, 0, 0, 0, 255);
        let mut dst = vec![0u8; 16 * 4];
        let result = perspective_warp(
            &src,
            0,
            4,
            &mut dst,
            4,
            4,
            &PerspectiveMatrix::identity(),
            [0; 4],
        );
        assert!(result.is_err());
    }

    // ── lens_undistort ────────────────────────────────────────────────────────

    #[test]
    fn test_lens_undistort_no_distortion_identity() {
        let w = 8u32;
        let h = 8u32;
        let src = solid_rgba(w, h, 50, 100, 150, 255);
        let mut dst = vec![0u8; (w * h * 4) as usize];
        let params = LensDistortionParams::no_distortion(w, h);
        lens_undistort(&src, w, h, &mut dst, &params, [0; 4]).expect("no distortion must succeed");
        // Interior pixels should be close to the source colour.
        for px in dst.chunks_exact(4).take(4) {
            assert!((px[0] as i32 - 50).unsigned_abs() <= 2, "R mismatch");
            assert!((px[1] as i32 - 100).unsigned_abs() <= 2, "G mismatch");
            assert!((px[2] as i32 - 150).unsigned_abs() <= 2, "B mismatch");
        }
    }

    #[test]
    fn test_lens_undistort_preserves_centre_pixel() {
        // Centre pixel should be unaffected by distortion.
        let w = 9u32; // odd size so centre is exact
        let h = 9u32;
        let mut src = vec![0u8; (w * h * 4) as usize];
        // Mark the centre pixel distinctively.
        let cx = (w / 2) as usize;
        let cy = (h / 2) as usize;
        let center_idx = (cy * w as usize + cx) * 4;
        src[center_idx] = 255;
        src[center_idx + 1] = 128;
        src[center_idx + 2] = 64;
        src[center_idx + 3] = 255;
        let mut dst = vec![0u8; (w * h * 4) as usize];
        let params = LensDistortionParams {
            k1: 0.1,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            fx: f64::from(w),
            fy: f64::from(h),
            cx: f64::from(w) / 2.0,
            cy: f64::from(h) / 2.0,
        };
        lens_undistort(&src, w, h, &mut dst, &params, [0; 4]).expect("undistort must succeed");
        // Centre pixel at (cx, cy): r2 = 0, so it maps back to itself.
        let out_r = dst[center_idx];
        assert!(
            out_r > 128,
            "centre R should reflect the marked pixel, got {out_r}"
        );
    }

    #[test]
    fn test_lens_undistort_invalid_dims_return_error() {
        let src = vec![0u8; 64];
        let mut dst = vec![0u8; 64];
        let params = LensDistortionParams::no_distortion(4, 4);
        let result = lens_undistort(&src, 0, 4, &mut dst, &params, [0; 4]);
        assert!(result.is_err());
    }

    // ── Geometric transforms ───────────────────────────────────────────────────

    /// Build a test image where every pixel has a unique value based on its
    /// (x, y) coordinates.  Pixel at (x, y) in an image of width `w` gets
    /// value `[y as u8, x as u8, (y*w+x) as u8]` using 3 channels.
    fn make_test_image_3ch(w: u32, h: u32) -> Vec<u8> {
        let mut buf = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let off = ((y * w + x) * 3) as usize;
                buf[off] = y as u8;
                buf[off + 1] = x as u8;
                buf[off + 2] = (y * w + x) as u8;
            }
        }
        buf
    }

    #[test]
    fn test_rotate90_dimensions() {
        // A 3×5 image rotated 90° CW should produce a 5×3 image.
        let img = make_test_image_3ch(3, 5);
        let out = TransformOperation::rotate90(&img, 3, 5, 3);
        // out_width = in_height = 5, out_height = in_width = 3
        assert_eq!(
            out.len(),
            (5 * 3 * 3) as usize,
            "output buffer size mismatch"
        );
    }

    #[test]
    fn test_rotate90_corner() {
        // Source image 4×2 (width=4, height=2), 3-channel.
        // After 90° CW: out_width=2, out_height=4.
        // src(0,0) → dst_x=src_y=0, dst_y=width-1-src_x=3 → dst(0,3)
        let w: u32 = 4;
        let h: u32 = 2;
        let ch: u32 = 3;
        let mut img = vec![0u8; (w * h * ch) as usize];
        // Mark src pixel (0,0) distinctively.
        img[0] = 1;
        img[1] = 2;
        img[2] = 3;

        let out = TransformOperation::rotate90(&img, w, h, ch);
        let out_width = h; // 2
                           // Expected: dst(dst_x=0, dst_y=3) holds [1,2,3]
        let dst_off = ((3 * out_width + 0) * ch) as usize;
        assert_eq!(
            &out[dst_off..dst_off + 3],
            &[1, 2, 3],
            "rotate90 corner pixel wrong"
        );
    }

    #[test]
    fn test_rotate180_roundtrip() {
        // Rotating 180° twice must reproduce the original image.
        let w: u32 = 4;
        let h: u32 = 3;
        let img = make_test_image_3ch(w, h);
        let once = TransformOperation::rotate180(&img, w, h, 3);
        let twice = TransformOperation::rotate180(&once, w, h, 3);
        assert_eq!(img, twice, "rotate180 twice must equal original");
    }

    #[test]
    fn test_flip_horizontal_reverses_row() {
        // Flip a 4×2 image horizontally; the first row of the output should be
        // the reverse of the first row of the input.
        let w: u32 = 4;
        let h: u32 = 2;
        let ch: u32 = 3;
        let img = make_test_image_3ch(w, h);
        let out = TransformOperation::flip_horizontal(&img, w, h, ch);

        // Row 0 of input: pixels at x=0,1,2,3
        // Row 0 of output: pixels at dst_x=3,2,1,0 (reversed)
        for x in 0..w {
            let src_off = (x * ch) as usize;
            let dst_off = ((w - 1 - x) * ch) as usize;
            assert_eq!(
                &img[src_off..src_off + ch as usize],
                &out[dst_off..dst_off + ch as usize],
                "flip_horizontal row-reversal wrong at x={x}"
            );
        }
    }

    #[test]
    fn test_transpose_swaps_dimensions() {
        // A 2×4 image (width=2, height=4) transposed should be 4×2.
        let w: u32 = 2;
        let h: u32 = 4;
        let ch: u32 = 3;
        let img = make_test_image_3ch(w, h);
        let out = TransformOperation::transpose(&img, w, h, ch);
        // out_width = in_height = 4, out_height = in_width = 2
        assert_eq!(
            out.len(),
            (4 * 2 * ch) as usize,
            "transpose buffer size mismatch"
        );
        // Verify that output(x=src_y, y=src_x) == input(src_x, src_y)
        let out_width: u32 = h; // 4
        for src_y in 0..h {
            for src_x in 0..w {
                let src_off = ((src_y * w + src_x) * ch) as usize;
                let dst_off = ((src_x * out_width + src_y) * ch) as usize;
                assert_eq!(
                    &img[src_off..src_off + ch as usize],
                    &out[dst_off..dst_off + ch as usize],
                    "transpose pixel mismatch at ({src_x},{src_y})"
                );
            }
        }
    }
}
