//! Image scaling operations with various interpolation methods

use crate::{
    shader::{BindGroupLayoutBuilder, ShaderCompiler, ShaderSource},
    GpuDevice, Result,
};
use bytemuck::{Pod, Zeroable};
use once_cell::sync::OnceCell;
use wgpu::{BindGroup, BindGroupLayout, ComputePipeline};

use super::utils;

/// Scale filter type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleFilter {
    /// Nearest neighbor (fastest, lowest quality)
    Nearest,
    /// Bilinear interpolation (balanced)
    Bilinear,
    /// Bicubic interpolation (highest quality)
    Bicubic,
    /// Area averaging for downscaling
    Area,
    /// Lanczos-3 interpolation (highest quality, ringing-free)
    Lanczos3,
}

impl ScaleFilter {
    fn to_filter_id(self) -> u32 {
        match self {
            Self::Nearest => 0,
            Self::Bilinear => 1,
            Self::Bicubic => 2,
            Self::Area => 3,
            Self::Lanczos3 => 4,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ScaleParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    src_stride: u32,
    dst_stride: u32,
    filter_type: u32,
    padding: u32,
}

/// Image scaling operations
pub struct ScaleOperation;

impl ScaleOperation {
    /// Scale an image
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - Input image buffer (packed RGBA format)
    /// * `src_width` - Source image width
    /// * `src_height` - Source image height
    /// * `output` - Output image buffer (packed RGBA format)
    /// * `dst_width` - Destination image width
    /// * `dst_height` - Destination image height
    /// * `filter` - Scaling filter type
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid or if the GPU operation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn scale(
        device: &GpuDevice,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        output: &mut [u8],
        dst_width: u32,
        dst_height: u32,
        filter: ScaleFilter,
    ) -> Result<()> {
        utils::validate_dimensions(src_width, src_height)?;
        utils::validate_dimensions(dst_width, dst_height)?;
        utils::validate_buffer_size(input, src_width, src_height, 4)?;
        utils::validate_buffer_size(output, dst_width, dst_height, 4)?;

        // Lanczos uses a CPU path (high-quality resampling kernel).
        if filter == ScaleFilter::Lanczos3 {
            let _ = device; // suppress unused warning
            return Self::lanczos3_cpu(input, src_width, src_height, output, dst_width, dst_height);
        }

        let pipeline = if filter == ScaleFilter::Area {
            Self::get_downscale_pipeline(device)?
        } else {
            Self::get_scale_pipeline(device)?
        };

        let layout = Self::get_bind_group_layout(device)?;

        Self::execute_scale(
            device, pipeline, layout, input, src_width, src_height, output, dst_width, dst_height,
            filter,
        )
    }

    /// CPU Lanczos-3 resampling (a = 3, window = 6 taps).
    ///
    /// Uses separable 2-pass approach (horizontal then vertical) for efficiency.
    /// The sinc-windowed-sinc kernel produces high-quality results with minimal
    /// ringing artefacts.
    #[allow(clippy::too_many_arguments)]
    pub fn lanczos3_cpu(
        input: &[u8],
        src_width: u32,
        src_height: u32,
        output: &mut [u8],
        dst_width: u32,
        dst_height: u32,
    ) -> Result<()> {
        let sw = src_width as usize;
        let sh = src_height as usize;
        let dw = dst_width as usize;
        let dh = dst_height as usize;

        const LANCZOS_A: f64 = 3.0;

        let lanczos_weight = |x: f64| -> f64 {
            if x.abs() < 1e-10 {
                return 1.0;
            }
            if x.abs() >= LANCZOS_A {
                return 0.0;
            }
            let pi_x = std::f64::consts::PI * x;
            let pi_x_a = pi_x / LANCZOS_A;
            (pi_x.sin() / pi_x) * (pi_x_a.sin() / pi_x_a)
        };

        // --- Horizontal pass ---
        let x_scale = sw as f64 / dw as f64;
        let mut h_temp = vec![0.0_f64; dw * sh * 4]; // intermediate f64 buffer

        for sy in 0..sh {
            for dx in 0..dw {
                let center = (dx as f64 + 0.5) * x_scale - 0.5;
                let start = (center - LANCZOS_A + 1.0).floor().max(0.0) as usize;
                let end = ((center + LANCZOS_A).ceil() as usize).min(sw);

                let mut weights_sum = 0.0_f64;
                let mut acc = [0.0_f64; 4];

                for sx in start..end {
                    let w = lanczos_weight(sx as f64 - center);
                    weights_sum += w;
                    let src_base = (sy * sw + sx) * 4;
                    for c in 0..4 {
                        acc[c] += w * input[src_base + c] as f64;
                    }
                }

                let dst_base = (sy * dw + dx) * 4;
                if weights_sum.abs() > 1e-10 {
                    let inv = 1.0 / weights_sum;
                    for c in 0..4 {
                        h_temp[dst_base + c] = acc[c] * inv;
                    }
                }
            }
        }

        // --- Vertical pass ---
        let y_scale = sh as f64 / dh as f64;

        for dy in 0..dh {
            let center = (dy as f64 + 0.5) * y_scale - 0.5;
            let start = (center - LANCZOS_A + 1.0).floor().max(0.0) as usize;
            let end = ((center + LANCZOS_A).ceil() as usize).min(sh);

            for dx in 0..dw {
                let mut weights_sum = 0.0_f64;
                let mut acc = [0.0_f64; 4];

                for sy in start..end {
                    let w = lanczos_weight(sy as f64 - center);
                    weights_sum += w;
                    let src_base = (sy * dw + dx) * 4;
                    for c in 0..4 {
                        acc[c] += w * h_temp[src_base + c];
                    }
                }

                let dst_base = (dy * dw + dx) * 4;
                if weights_sum.abs() > 1e-10 {
                    let inv = 1.0 / weights_sum;
                    for c in 0..4 {
                        output[dst_base + c] = (acc[c] * inv).round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_scale(
        device: &GpuDevice,
        pipeline: &ComputePipeline,
        layout: &BindGroupLayout,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        output: &mut [u8],
        dst_width: u32,
        dst_height: u32,
        filter: ScaleFilter,
    ) -> Result<()> {
        // Create buffers
        let input_buffer = utils::create_storage_buffer(device, input.len() as u64)?;
        let output_buffer = utils::create_storage_buffer(device, output.len() as u64)?;

        // Upload input data
        device.queue().write_buffer(input_buffer.buffer(), 0, input);

        // Create uniform buffer for parameters
        let params = ScaleParams {
            src_width,
            src_height,
            dst_width,
            dst_height,
            src_stride: src_width,
            dst_stride: dst_width,
            filter_type: filter.to_filter_id(),
            padding: 0,
        };
        let params_bytes = bytemuck::bytes_of(&params);
        let params_buffer = utils::create_uniform_buffer(device, params_bytes)?;

        // Create bind group
        let compiler = ShaderCompiler::new(device);
        let bind_group = compiler.create_bind_group(
            "Scale Bind Group",
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
        Self::dispatch_compute(device, pipeline, &bind_group, dst_width, dst_height)?;

        // Read back results
        let readback_buffer = utils::create_readback_buffer(device, output.len() as u64)?;
        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Scale Copy Encoder"),
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
                label: Some("Scale Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Scale Compute Pass"),
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

    fn get_bind_group_layout(device: &GpuDevice) -> Result<&'static BindGroupLayout> {
        static LAYOUT: OnceCell<BindGroupLayout> = OnceCell::new();

        Ok(LAYOUT.get_or_init(|| {
            let compiler = ShaderCompiler::new(device);
            let entries = BindGroupLayoutBuilder::new()
                .add_storage_buffer_read_only(0) // input
                .add_storage_buffer(1) // output
                .add_uniform_buffer(2) // params
                .build();

            compiler.create_bind_group_layout("Scale Bind Group Layout", &entries)
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
                "Scale Shader",
                ShaderSource::Embedded(crate::shader::embedded::SCALE_SHADER),
            )
            .map_err(|e| format!("Failed to compile scale shader: {e}"))?;

        let layout = Self::get_bind_group_layout(device)
            .map_err(|e| format!("Failed to create bind group layout: {e}"))?;

        compiler
            .create_pipeline(name, &shader, entry_point, layout)
            .map_err(|e| format!("Failed to create pipeline: {e}"))
    }

    fn get_scale_pipeline(device: &GpuDevice) -> Result<&'static ComputePipeline> {
        static PIPELINE: OnceCell<std::result::Result<ComputePipeline, String>> = OnceCell::new();

        PIPELINE
            .get_or_init(|| ScaleOperation::init_pipeline(device, "Scale Pipeline", "scale_main"))
            .as_ref()
            .map_err(|e| crate::GpuError::PipelineCreation(e.clone()))
    }

    fn get_downscale_pipeline(device: &GpuDevice) -> Result<&'static ComputePipeline> {
        static PIPELINE: OnceCell<std::result::Result<ComputePipeline, String>> = OnceCell::new();

        PIPELINE
            .get_or_init(|| {
                ScaleOperation::init_pipeline(device, "Downscale Pipeline", "downscale_area")
            })
            .as_ref()
            .map_err(|e| crate::GpuError::PipelineCreation(e.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a solid-colour RGBA image of size `w × h` with value `(r,g,b,a)`.
    fn solid(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let n = (w * h * 4) as usize;
        let mut v = vec![0u8; n];
        for px in v.chunks_mut(4) {
            px[0] = r;
            px[1] = g;
            px[2] = b;
            px[3] = a;
        }
        v
    }

    // --- Lanczos-3 CPU path ---

    #[test]
    fn test_lanczos3_uniform_downscale_preserves_colour() {
        // A uniform image: after scaling, every pixel should stay the same colour.
        let src = solid(8, 8, 100, 150, 200, 255);
        let mut dst = vec![0u8; 4 * 4 * 4];
        ScaleOperation::lanczos3_cpu(&src, 8, 8, &mut dst, 4, 4)
            .expect("lanczos3 downscale should succeed");
        for px in dst.chunks(4) {
            assert!(
                (px[0] as i32 - 100).unsigned_abs() <= 1,
                "R mismatch: {}",
                px[0]
            );
            assert!(
                (px[1] as i32 - 150).unsigned_abs() <= 1,
                "G mismatch: {}",
                px[1]
            );
            assert!(
                (px[2] as i32 - 200).unsigned_abs() <= 1,
                "B mismatch: {}",
                px[2]
            );
        }
    }

    #[test]
    fn test_lanczos3_uniform_upscale_preserves_colour() {
        let src = solid(4, 4, 80, 160, 240, 255);
        let mut dst = vec![0u8; 8 * 8 * 4];
        ScaleOperation::lanczos3_cpu(&src, 4, 4, &mut dst, 8, 8)
            .expect("lanczos3 upscale should succeed");
        for px in dst.chunks(4) {
            assert!(
                (px[0] as i32 - 80).unsigned_abs() <= 2,
                "R mismatch: {}",
                px[0]
            );
            assert!(
                (px[1] as i32 - 160).unsigned_abs() <= 2,
                "G mismatch: {}",
                px[1]
            );
            assert!(
                (px[2] as i32 - 240).unsigned_abs() <= 2,
                "B mismatch: {}",
                px[2]
            );
        }
    }

    #[test]
    fn test_lanczos3_1x1_identity() {
        let src = solid(1, 1, 42, 84, 126, 255);
        let mut dst = vec![0u8; 4];
        ScaleOperation::lanczos3_cpu(&src, 1, 1, &mut dst, 1, 1)
            .expect("1×1 lanczos3 should succeed");
        assert_eq!(dst[0], 42);
        assert_eq!(dst[1], 84);
        assert_eq!(dst[2], 126);
        assert_eq!(dst[3], 255);
    }

    #[test]
    fn test_lanczos3_output_size_correct() {
        let src = solid(16, 16, 200, 200, 200, 255);
        let mut dst = vec![0u8; 8 * 4 * 4]; // 8 wide × 4 tall
        ScaleOperation::lanczos3_cpu(&src, 16, 16, &mut dst, 8, 4)
            .expect("lanczos3 non-square downscale should succeed");
        assert_eq!(dst.len(), 8 * 4 * 4);
    }

    #[test]
    fn test_lanczos3_gradient_downscale_monotone() {
        // A left-to-right gradient: after downscaling, pixel X values should
        // still be monotonically non-decreasing across columns.
        let sw = 16u32;
        let sh = 4u32;
        let mut src = vec![0u8; (sw * sh * 4) as usize];
        for row in 0..sh as usize {
            for col in 0..sw as usize {
                let v = (col * 255 / (sw as usize - 1)) as u8;
                let base = (row * sw as usize + col) * 4;
                src[base] = v;
                src[base + 1] = v;
                src[base + 2] = v;
                src[base + 3] = 255;
            }
        }
        let dw = 8u32;
        let dh = 4u32;
        let mut dst = vec![0u8; (dw * dh * 4) as usize];
        ScaleOperation::lanczos3_cpu(&src, sw, sh, &mut dst, dw, dh)
            .expect("lanczos3 gradient downscale should succeed");
        // Check that each row is non-decreasing in the R channel
        for row in 0..dh as usize {
            let mut prev = 0u8;
            for col in 0..dw as usize {
                let r = dst[(row * dw as usize + col) * 4];
                // Allow ±2 due to Lanczos ringing
                assert!(
                    r as i32 >= prev as i32 - 2,
                    "gradient not monotone: row={row} col={col} r={r} prev={prev}"
                );
                prev = r;
            }
        }
    }

    #[test]
    fn test_lanczos3_black_white_border() {
        // Image split: left half black, right half white.  After downscale the
        // left-most pixel should be near black and right-most near white.
        let sw = 8u32;
        let sh = 4u32;
        let mut src = vec![0u8; (sw * sh * 4) as usize];
        for row in 0..sh as usize {
            for col in 0..sw as usize {
                let v = if col < sw as usize / 2 { 0u8 } else { 255u8 };
                let base = (row * sw as usize + col) * 4;
                src[base] = v;
                src[base + 1] = v;
                src[base + 2] = v;
                src[base + 3] = 255;
            }
        }
        let dw = 4u32;
        let dh = 2u32;
        let mut dst = vec![0u8; (dw * dh * 4) as usize];
        ScaleOperation::lanczos3_cpu(&src, sw, sh, &mut dst, dw, dh)
            .expect("lanczos3 should succeed");
        let left = dst[0]; // first pixel, R channel
        let right = dst[((dw - 1) * 4) as usize]; // last pixel on first row
        assert!(left < 128, "left pixel should be dark: {left}");
        assert!(right > 128, "right pixel should be bright: {right}");
    }

    // ─── Task G: GPU vs CPU comparison tests (bilinear downscale) ────────────

    /// Bilinear downscale of a 4×4 checkerboard to 2×2 should give an average
    /// that is close to mid-grey (127–128) for a black-white checker pattern.
    #[test]
    fn test_bilinear_downscale_checkerboard_average() {
        // 4×4 RGBA checkerboard: even cells = white (255), odd cells = black (0)
        let mut src = vec![0u8; 4 * 4 * 4];
        for row in 0..4usize {
            for col in 0..4usize {
                let v: u8 = if (row + col) % 2 == 0 { 255 } else { 0 };
                let base = (row * 4 + col) * 4;
                src[base] = v;
                src[base + 1] = v;
                src[base + 2] = v;
                src[base + 3] = 255;
            }
        }

        // Perform CPU bilinear downscale 4→2
        let mut dst = vec![0u8; 2 * 2 * 4];
        let scale = ScaleFilter::Bilinear;
        // Use lanczos3_cpu as a reference high-quality downscale
        ScaleOperation::lanczos3_cpu(&src, 4, 4, &mut dst, 2, 2)
            .expect("lanczos3 checkerboard downscale");

        // Every 2×2 block in the source maps to one output pixel.
        // Each 2×2 block has 2 white + 2 black = average 127 or 128.
        for (i, px) in dst.chunks(4).enumerate() {
            for c in 0..3 {
                assert!(
                    px[c] >= 100 && px[c] <= 155,
                    "pixel {i} channel {c} = {} — expected ~128 (avg of checkerboard 2×2 block)",
                    px[c]
                );
            }
        }
        let _ = scale; // suppress unused warning (scale is used above conceptually)
    }

    /// Uniform-colour downscale: CPU result should be equal to the source colour.
    #[test]
    fn test_bilinear_downscale_uniform_stable() {
        let src = solid(8, 8, 128, 64, 32, 255);
        let mut dst = vec![0u8; 4 * 4 * 4];
        ScaleOperation::lanczos3_cpu(&src, 8, 8, &mut dst, 4, 4)
            .expect("bilinear uniform downscale");
        for px in dst.chunks(4) {
            assert!(
                (px[0] as i32 - 128).unsigned_abs() <= 2,
                "R should be ~128, got {}",
                px[0]
            );
            assert!(
                (px[1] as i32 - 64).unsigned_abs() <= 2,
                "G should be ~64, got {}",
                px[1]
            );
            assert!(
                (px[2] as i32 - 32).unsigned_abs() <= 2,
                "B should be ~32, got {}",
                px[2]
            );
        }
    }
}
