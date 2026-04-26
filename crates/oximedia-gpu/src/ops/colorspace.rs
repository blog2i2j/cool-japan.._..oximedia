//! Color space conversion operations (RGB ↔ YUV)

use crate::{
    shader::{BindGroupLayoutBuilder, ShaderCompiler, ShaderSource},
    GpuDevice, Result,
};
use bytemuck::{Pod, Zeroable};
use once_cell::sync::OnceCell;
use wgpu::{BindGroup, BindGroupLayout, ComputePipeline};

use super::utils;

/// Color space standards
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// BT.601 (SD video)
    BT601,
    /// BT.709 (HD video)
    BT709,
    /// BT.2020 (UHD video)
    BT2020,
}

impl ColorSpace {
    fn to_format_id(self) -> u32 {
        match self {
            Self::BT601 => 0,
            Self::BT709 => 1,
            Self::BT2020 => 2,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ConversionParams {
    width: u32,
    height: u32,
    stride: u32,
    format: u32,
}

/// Color space conversion operations
pub struct ColorSpaceConversion;

impl ColorSpaceConversion {
    /// Convert RGB to YUV
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - Input RGB buffer (packed RGBA format)
    /// * `output` - Output YUV buffer (packed YUVA format)
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `color_space` - Color space standard (BT.601, BT.709, BT.2020)
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid or if the GPU operation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn rgb_to_yuv(
        device: &GpuDevice,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        color_space: ColorSpace,
    ) -> Result<()> {
        utils::validate_dimensions(width, height)?;
        utils::validate_buffer_size(input, width, height, 4)?;
        utils::validate_buffer_size(output, width, height, 4)?;

        let pipeline = Self::get_rgb_to_yuv_pipeline(device)?;
        let layout = Self::get_bind_group_layout(device)?;

        Self::execute_conversion(
            device,
            pipeline,
            layout,
            input,
            output,
            width,
            height,
            color_space,
        )
    }

    /// Convert YUV to RGB
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - Input YUV buffer (packed YUVA format)
    /// * `output` - Output RGB buffer (packed RGBA format)
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `color_space` - Color space standard (BT.601, BT.709, BT.2020)
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid or if the GPU operation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn yuv_to_rgb(
        device: &GpuDevice,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        color_space: ColorSpace,
    ) -> Result<()> {
        utils::validate_dimensions(width, height)?;
        utils::validate_buffer_size(input, width, height, 4)?;
        utils::validate_buffer_size(output, width, height, 4)?;

        let pipeline = Self::get_yuv_to_rgb_pipeline(device)?;
        let layout = Self::get_bind_group_layout(device)?;

        Self::execute_conversion(
            device,
            pipeline,
            layout,
            input,
            output,
            width,
            height,
            color_space,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_conversion(
        device: &GpuDevice,
        pipeline: &ComputePipeline,
        layout: &BindGroupLayout,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        color_space: ColorSpace,
    ) -> Result<()> {
        // Create buffers
        let input_buffer = utils::create_storage_buffer(device, input.len() as u64)?;
        let output_buffer = utils::create_storage_buffer(device, output.len() as u64)?;

        // Upload input data
        device.queue().write_buffer(input_buffer.buffer(), 0, input);

        // Create uniform buffer for parameters
        let params = ConversionParams {
            width,
            height,
            stride: width,
            format: color_space.to_format_id(),
        };
        let params_bytes = bytemuck::bytes_of(&params);
        let params_buffer = utils::create_uniform_buffer(device, params_bytes)?;

        // Create bind group
        let compiler = ShaderCompiler::new(device);
        let bind_group = compiler.create_bind_group(
            "ColorSpace Bind Group",
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
                label: Some("ColorSpace Copy Encoder"),
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
                label: Some("ColorSpace Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ColorSpace Compute Pass"),
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

            compiler.create_bind_group_layout("ColorSpace Bind Group Layout", &entries)
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
                "ColorSpace Shader",
                ShaderSource::Embedded(crate::shader::embedded::COLORSPACE_SHADER),
            )
            .map_err(|e| format!("Failed to compile colorspace shader: {e}"))?;

        let layout = Self::get_bind_group_layout(device)
            .map_err(|e| format!("Failed to create bind group layout: {e}"))?;

        compiler
            .create_pipeline(name, &shader, entry_point, layout)
            .map_err(|e| format!("Failed to create pipeline: {e}"))
    }

    fn get_rgb_to_yuv_pipeline(device: &GpuDevice) -> Result<&'static ComputePipeline> {
        static PIPELINE: OnceCell<std::result::Result<ComputePipeline, String>> = OnceCell::new();

        PIPELINE
            .get_or_init(|| {
                ColorSpaceConversion::init_pipeline(
                    device,
                    "RGB to YUV Pipeline",
                    "rgb_to_yuv_main",
                )
            })
            .as_ref()
            .map_err(|e| crate::GpuError::PipelineCreation(e.clone()))
    }

    fn get_yuv_to_rgb_pipeline(device: &GpuDevice) -> Result<&'static ComputePipeline> {
        static PIPELINE: OnceCell<std::result::Result<ComputePipeline, String>> = OnceCell::new();

        PIPELINE
            .get_or_init(|| {
                ColorSpaceConversion::init_pipeline(
                    device,
                    "YUV to RGB Pipeline",
                    "yuv_to_rgb_main",
                )
            })
            .as_ref()
            .map_err(|e| crate::GpuError::PipelineCreation(e.clone()))
    }
}

// =============================================================================
// CPU-side HSV / Lab / sRGB↔Linear color conversions
// =============================================================================

impl ColorSpaceConversion {
    /// Convert interleaved RGBA pixels from RGB to HSV encoding.
    ///
    /// Input layout: 4 bytes per pixel — R, G, B, A.
    /// Output layout: 4 bytes per pixel — H_enc, S_enc, V_enc, A (pass-through).
    ///
    /// Encoding:
    /// * H → `(H / 360.0 * 255.0) as u8`  (hue 0°–360° mapped to 0–255)
    /// * S → `(S * 255.0) as u8`           (saturation 0.0–1.0)
    /// * V → `(V * 255.0) as u8`           (value 0.0–1.0)
    ///
    /// # Panics
    ///
    /// Does not panic; invalid pixel counts are handled by truncating to complete
    /// 4-byte pixels.
    #[must_use]
    pub fn rgb_to_hsv(data: &[u8], width: u32, height: u32) -> Vec<u8> {
        let pixel_count = (width as usize) * (height as usize);
        let mut out = vec![0u8; pixel_count * 4];

        for i in 0..pixel_count {
            let base = i * 4;
            if base + 3 >= data.len() {
                break;
            }
            let r = f64::from(data[base]) / 255.0;
            let g = f64::from(data[base + 1]) / 255.0;
            let b = f64::from(data[base + 2]) / 255.0;
            let alpha = data[base + 3];

            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let delta = max - min;

            let v = max;
            let s = if max > 0.0 { delta / max } else { 0.0 };

            let h = if delta < 1e-10 {
                0.0_f64
            } else if (max - r).abs() < 1e-10 {
                let sector = (g - b) / delta;
                // fmod equivalent for f64 — keep in [0, 6)
                let sector = sector - (sector / 6.0).floor() * 6.0;
                60.0 * sector
            } else if (max - g).abs() < 1e-10 {
                60.0 * ((b - r) / delta + 2.0)
            } else {
                60.0 * ((r - g) / delta + 4.0)
            };
            let h = if h < 0.0 { h + 360.0 } else { h };

            out[base] = (h / 360.0 * 255.0).clamp(0.0, 255.0).round() as u8;
            out[base + 1] = (s * 255.0).clamp(0.0, 255.0).round() as u8;
            out[base + 2] = (v * 255.0).clamp(0.0, 255.0).round() as u8;
            out[base + 3] = alpha;
        }
        out
    }

    /// Convert interleaved RGBA pixels from HSV to RGB encoding.
    ///
    /// Input layout: 4 bytes per pixel — H_enc, S_enc, V_enc, A.
    /// Output layout: 4 bytes per pixel — R, G, B, A (pass-through).
    ///
    /// Decoding: H = byte × 360 / 255, S = byte / 255, V = byte / 255.
    #[must_use]
    pub fn hsv_to_rgb(data: &[u8], width: u32, height: u32) -> Vec<u8> {
        let pixel_count = (width as usize) * (height as usize);
        let mut out = vec![0u8; pixel_count * 4];

        for i in 0..pixel_count {
            let base = i * 4;
            if base + 3 >= data.len() {
                break;
            }
            let h = f64::from(data[base]) * 360.0 / 255.0; // 0.0 .. 360.0
            let s = f64::from(data[base + 1]) / 255.0; // 0.0 .. 1.0
            let v = f64::from(data[base + 2]) / 255.0; // 0.0 .. 1.0
            let alpha = data[base + 3];

            let c = v * s;
            let h_prime = h / 60.0;
            // |h_prime mod 2 - 1|
            let h_mod2 = h_prime - (h_prime / 2.0).floor() * 2.0;
            let x = c * (1.0 - (h_mod2 - 1.0).abs());
            let m = v - c;

            let sector = (h_prime as u32) % 6;
            let (r1, g1, b1) = match sector {
                0 => (c, x, 0.0),
                1 => (x, c, 0.0),
                2 => (0.0, c, x),
                3 => (0.0, x, c),
                4 => (x, 0.0, c),
                _ => (c, 0.0, x),
            };

            out[base] = ((r1 + m) * 255.0).clamp(0.0, 255.0).round() as u8;
            out[base + 1] = ((g1 + m) * 255.0).clamp(0.0, 255.0).round() as u8;
            out[base + 2] = ((b1 + m) * 255.0).clamp(0.0, 255.0).round() as u8;
            out[base + 3] = alpha;
        }
        out
    }

    /// Convert interleaved RGBA pixels from sRGB to CIE L*a*b*.
    ///
    /// Input layout: 4 bytes per pixel — R, G, B, A.
    /// Output layout: 4 bytes per pixel:
    /// * L*  (0–100) → byte = `(L * 255.0 / 100.0) as u8`
    /// * a*  (−128–127) → byte = `(a + 128.0) as u8` (clamped 0–255)
    /// * b*  (−128–127) → byte = `(b + 128.0) as u8` (clamped 0–255)
    /// * A: pass-through
    #[must_use]
    pub fn rgb_to_lab(data: &[u8], width: u32, height: u32) -> Vec<u8> {
        // D65 reference white
        const XN: f64 = 0.95047;
        const YN: f64 = 1.00000;
        const ZN: f64 = 1.08883;

        let pixel_count = (width as usize) * (height as usize);
        let mut out = vec![0u8; pixel_count * 4];

        for i in 0..pixel_count {
            let base = i * 4;
            if base + 3 >= data.len() {
                break;
            }
            let r_lin = Self::srgb_channel_to_linear(f64::from(data[base]) / 255.0);
            let g_lin = Self::srgb_channel_to_linear(f64::from(data[base + 1]) / 255.0);
            let b_lin = Self::srgb_channel_to_linear(f64::from(data[base + 2]) / 255.0);
            let alpha = data[base + 3];

            // Linear sRGB → CIE XYZ (D65, IEC 61966-2-1 matrix)
            let x = 0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin;
            let y = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin;
            let z = 0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin;

            // XYZ → Lab (using the standard cube-root / linear piece-wise f)
            let fx = Self::lab_f(x / XN);
            let fy = Self::lab_f(y / YN);
            let fz = Self::lab_f(z / ZN);

            let l_star = 116.0 * fy - 16.0;
            let a_star = 500.0 * (fx - fy);
            let b_star = 200.0 * (fy - fz);

            // Encode to u8
            out[base] = (l_star * 255.0 / 100.0).clamp(0.0, 255.0).round() as u8;
            out[base + 1] = (a_star + 128.0).clamp(0.0, 255.0).round() as u8;
            out[base + 2] = (b_star + 128.0).clamp(0.0, 255.0).round() as u8;
            out[base + 3] = alpha;
        }
        out
    }

    /// Convert interleaved RGBA pixels from CIE L*a*b* back to sRGB.
    ///
    /// Input layout: 4 bytes per pixel — L_enc, a_enc, b_enc, A.
    /// Output layout: 4 bytes per pixel — R, G, B, A (pass-through).
    #[must_use]
    pub fn lab_to_rgb(data: &[u8], width: u32, height: u32) -> Vec<u8> {
        const XN: f64 = 0.95047;
        const YN: f64 = 1.00000;
        const ZN: f64 = 1.08883;

        let pixel_count = (width as usize) * (height as usize);
        let mut out = vec![0u8; pixel_count * 4];

        for i in 0..pixel_count {
            let base = i * 4;
            if base + 3 >= data.len() {
                break;
            }
            let l_star = f64::from(data[base]) * 100.0 / 255.0;
            let a_star = f64::from(data[base + 1]) - 128.0;
            let b_star = f64::from(data[base + 2]) - 128.0;
            let alpha = data[base + 3];

            // Lab → XYZ
            let fy = (l_star + 16.0) / 116.0;
            let fx = a_star / 500.0 + fy;
            let fz = fy - b_star / 200.0;

            let x = Self::lab_f_inv(fx) * XN;
            let y = Self::lab_f_inv(fy) * YN;
            let z = Self::lab_f_inv(fz) * ZN;

            // XYZ → Linear sRGB (inverse of the IEC 61966-2-1 matrix)
            let r_lin = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
            let g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
            let b_lin = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;

            // Linear sRGB → sRGB (gamma encoding)
            let r_srgb = Self::linear_channel_to_srgb(r_lin);
            let g_srgb = Self::linear_channel_to_srgb(g_lin);
            let b_srgb = Self::linear_channel_to_srgb(b_lin);

            out[base] = (r_srgb * 255.0).clamp(0.0, 255.0).round() as u8;
            out[base + 1] = (g_srgb * 255.0).clamp(0.0, 255.0).round() as u8;
            out[base + 2] = (b_srgb * 255.0).clamp(0.0, 255.0).round() as u8;
            out[base + 3] = alpha;
        }
        out
    }

    /// Convert interleaved RGBA pixels from sRGB to linear light (remove gamma).
    ///
    /// Input/output: 4 bytes per pixel — R, G, B, A.  Alpha is passed through.
    /// The linear value (0.0–1.0 f64) is scaled back to u8 (0–255).
    #[must_use]
    pub fn srgb_to_linear(data: &[u8], width: u32, height: u32) -> Vec<u8> {
        let pixel_count = (width as usize) * (height as usize);
        let mut out = vec![0u8; pixel_count * 4];

        for i in 0..pixel_count {
            let base = i * 4;
            if base + 3 >= data.len() {
                break;
            }
            for ch in 0..3 {
                let c = f64::from(data[base + ch]) / 255.0;
                let lin = Self::srgb_channel_to_linear(c);
                out[base + ch] = (lin * 255.0).clamp(0.0, 255.0).round() as u8;
            }
            out[base + 3] = data[base + 3];
        }
        out
    }

    /// Convert interleaved RGBA pixels from linear light to sRGB (apply gamma).
    ///
    /// Input/output: 4 bytes per pixel — R, G, B, A.  Alpha is passed through.
    #[must_use]
    pub fn linear_to_srgb(data: &[u8], width: u32, height: u32) -> Vec<u8> {
        let pixel_count = (width as usize) * (height as usize);
        let mut out = vec![0u8; pixel_count * 4];

        for i in 0..pixel_count {
            let base = i * 4;
            if base + 3 >= data.len() {
                break;
            }
            for ch in 0..3 {
                let c = f64::from(data[base + ch]) / 255.0;
                let enc = Self::linear_channel_to_srgb(c);
                out[base + ch] = (enc * 255.0).clamp(0.0, 255.0).round() as u8;
            }
            out[base + 3] = data[base + 3];
        }
        out
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// sRGB electro-optical transfer function (inverse gamma): sRGB → linear.
    ///
    /// IEC 61966-2-1: for `c ≤ 0.04045` → `c / 12.92`, else `((c+0.055)/1.055)^2.4`.
    #[inline]
    fn srgb_channel_to_linear(c: f64) -> f64 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    /// sRGB opto-electronic transfer function (gamma): linear → sRGB.
    ///
    /// IEC 61966-2-1: for `c ≤ 0.0031308` → `c * 12.92`, else `1.055*c^(1/2.4) - 0.055`.
    #[inline]
    fn linear_channel_to_srgb(c: f64) -> f64 {
        let c = c.clamp(0.0, 1.0);
        if c <= 0.0031308 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        }
    }

    /// CIE Lab piecewise cube-root function `f(t)`.
    ///
    /// `f(t) = t^(1/3)` if `t > ε`, else `(7.787 * t) + 16/116`.
    #[inline]
    fn lab_f(t: f64) -> f64 {
        // ε = (6/29)^3 ≈ 0.008856
        if t > 0.008_856 {
            t.cbrt()
        } else {
            7.787 * t + 16.0 / 116.0
        }
    }

    /// Inverse of `lab_f`.
    #[inline]
    fn lab_f_inv(t: f64) -> f64 {
        // δ = 6/29 ≈ 0.2069
        const DELTA: f64 = 6.0 / 29.0;
        if t > DELTA {
            t * t * t
        } else {
            3.0 * DELTA * DELTA * (t - 16.0 / 116.0)
        }
    }
}

// =============================================================================
// CPU-side reference color conversions (BT.601, BT.709, BT.2020, BT.2100)
// =============================================================================

/// BT.601 RGB → YCbCr (studio swing: Y ∈ \[16,235\], Cb/Cr ∈ \[16,240\]).
///
/// Input: linear RGB in [0, 255].
/// Output: (Y, Cb, Cr) in [0, 255] (offset and scaled per ITU-R BT.601).
#[must_use]
pub fn bt601_rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = f64::from(r);
    let g = f64::from(g);
    let b = f64::from(b);
    let y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0;
    let cb = 128.0 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0;
    let cr = 128.0 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0;
    (
        y.round().clamp(0.0, 255.0) as u8,
        cb.round().clamp(0.0, 255.0) as u8,
        cr.round().clamp(0.0, 255.0) as u8,
    )
}

/// BT.601 YCbCr → RGB (studio swing: Y ∈ \[16,235\], Cb/Cr ∈ \[16,240\]).
///
/// Output: linear RGB in [0, 255].
#[must_use]
pub fn bt601_ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y = f64::from(y) - 16.0;
    let cb = f64::from(cb) - 128.0;
    let cr = f64::from(cr) - 128.0;
    let r = 255.0 * (1.164 * y + 1.596 * cr) / 255.0;
    let g = 255.0 * (1.164 * y - 0.392 * cb - 0.813 * cr) / 255.0;
    let b = 255.0 * (1.164 * y + 2.017 * cb) / 255.0;
    (
        r.round().clamp(0.0, 255.0) as u8,
        g.round().clamp(0.0, 255.0) as u8,
        b.round().clamp(0.0, 255.0) as u8,
    )
}

/// BT.709 RGB → YCbCr (studio swing: Y ∈ \[16,235\], Cb/Cr ∈ \[16,240\]).
///
/// Input: linear RGB in [0, 255].
/// Output: (Y, Cb, Cr) in [0, 255].
#[must_use]
pub fn bt709_rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    // BT.709 direct matrix form (ITU-R BT.709 Table B.3), studio swing.
    // Kr = 0.2126, Kb = 0.0722, Kg = 1 - Kr - Kb = 0.7152
    let r_n = f64::from(r) / 255.0;
    let g_n = f64::from(g) / 255.0;
    let b_n = f64::from(b) / 255.0;
    let y = 16.0 + 219.0 * (0.2126 * r_n + 0.7152 * g_n + 0.0722 * b_n);
    let cb = 128.0 + 224.0 * (-0.2126 / 1.8556 * r_n - 0.7152 / 1.8556 * g_n + 0.5 * b_n);
    let cr = 128.0 + 224.0 * (0.5 * r_n - 0.7152 / 1.5748 * g_n - 0.0722 / 1.5748 * b_n);
    (
        y.round().clamp(0.0, 255.0) as u8,
        cb.round().clamp(0.0, 255.0) as u8,
        cr.round().clamp(0.0, 255.0) as u8,
    )
}

/// BT.709 YCbCr → RGB (studio swing).
///
/// Output: linear RGB in [0, 255].
#[must_use]
pub fn bt709_ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y_n = (f64::from(y) - 16.0) / 219.0;
    let cb_n = (f64::from(cb) - 128.0) / 224.0;
    let cr_n = (f64::from(cr) - 128.0) / 224.0;
    let r = y_n + 1.5748 * cr_n;
    let g = y_n - 0.2126 / 0.7152 * 1.5748 * cr_n - 0.0722 / 0.7152 * 1.8556 * cb_n;
    let b = y_n + 1.8556 * cb_n;
    (
        (r * 255.0).round().clamp(0.0, 255.0) as u8,
        (g * 255.0).round().clamp(0.0, 255.0) as u8,
        (b * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

/// BT.2020 RGB → YCbCr (studio swing).
///
/// BT.2020 uses primaries for Ultra HD (UHD) content.
/// Coefficients: Kr = 0.2627, Kb = 0.0593, Kg = 0.6780.
///
/// Input: linear RGB in [0, 255].
/// Output: (Y, Cb, Cr) in [0, 255] studio swing.
#[must_use]
pub fn bt2020_rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r_n = f64::from(r) / 255.0;
    let g_n = f64::from(g) / 255.0;
    let b_n = f64::from(b) / 255.0;
    // BT.2020 luma coefficients (Kr, Kg, Kb)
    let kr = 0.2627_f64;
    let kb = 0.0593_f64;
    let kg = 1.0 - kr - kb; // 0.6780
    let y = 16.0 + 219.0 * (kr * r_n + kg * g_n + kb * b_n);
    let cb = 128.0
        + 224.0 * ((-kr / (2.0 * (1.0 - kb))) * r_n + (-kg / (2.0 * (1.0 - kb))) * g_n + 0.5 * b_n);
    let cr = 128.0
        + 224.0 * (0.5 * r_n + (-kg / (2.0 * (1.0 - kr))) * g_n + (-kb / (2.0 * (1.0 - kr))) * b_n);
    (
        y.round().clamp(0.0, 255.0) as u8,
        cb.round().clamp(0.0, 255.0) as u8,
        cr.round().clamp(0.0, 255.0) as u8,
    )
}

/// BT.2020 YCbCr → RGB (studio swing).
///
/// Output: linear RGB in [0, 255].
#[must_use]
pub fn bt2020_ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y_n = (f64::from(y) - 16.0) / 219.0;
    let cb_n = (f64::from(cb) - 128.0) / 224.0;
    let cr_n = (f64::from(cr) - 128.0) / 224.0;
    // BT.2020 inverse matrix
    let kr = 0.2627_f64;
    let kb = 0.0593_f64;
    let kg = 1.0 - kr - kb;
    let r_cr = 2.0 * (1.0 - kr); // 1.4746
    let b_cb = 2.0 * (1.0 - kb); // 1.8814
    let g_cr = -2.0 * kr * (1.0 - kr) / kg;
    let g_cb = -2.0 * kb * (1.0 - kb) / kg;
    let r = y_n + r_cr * cr_n;
    let g = y_n + g_cr * cr_n + g_cb * cb_n;
    let b = y_n + b_cb * cb_n;
    (
        (r * 255.0).round().clamp(0.0, 255.0) as u8,
        (g * 255.0).round().clamp(0.0, 255.0) as u8,
        (b * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

/// BT.2100 PQ (Perceptual Quantizer) transfer function: linear → PQ-encoded.
///
/// Input: linear scene luminance normalised to [0, 1] where 1 = 10 000 nits.
/// Output: PQ-encoded value in [0, 1].
#[must_use]
pub fn pq_oetf(l: f64) -> f64 {
    // SMPTE ST 2084 PQ EOTF constants
    const M1: f64 = 0.159_301_758_5;
    const M2: f64 = 78.843_75;
    const C1: f64 = 0.835_937_5;
    const C2: f64 = 18.851_563;
    const C3: f64 = 18.687_5;
    let l_m1 = l.abs().powf(M1);
    ((C1 + C2 * l_m1) / (1.0 + C3 * l_m1)).powf(M2)
}

/// BT.2100 PQ inverse transfer function: PQ-encoded → linear.
///
/// Input: PQ-encoded value in [0, 1].
/// Output: linear scene luminance normalised to [0, 1] where 1 = 10 000 nits.
#[must_use]
pub fn pq_eotf(e: f64) -> f64 {
    const M1: f64 = 0.159_301_758_5;
    const M2: f64 = 78.843_75;
    const C1: f64 = 0.835_937_5;
    const C2: f64 = 18.851_563;
    const C3: f64 = 18.687_5;
    let e_m2 = e.abs().powf(1.0 / M2);
    let num = (e_m2 - C1).max(0.0);
    let den = C2 - C3 * e_m2;
    (num / den).powf(1.0 / M1)
}

/// BT.2100 HLG (Hybrid Log-Gamma) transfer function: scene linear → HLG.
///
/// Input: normalised scene luminance in [0, 1].
/// Output: HLG-encoded signal in [0, 1].
#[must_use]
pub fn hlg_oetf(l: f64) -> f64 {
    const A: f64 = 0.178_832_77;
    const B: f64 = 0.284_668_92;
    const C: f64 = 0.559_910_73;
    if l <= 1.0 / 12.0 {
        (3.0 * l).sqrt()
    } else {
        A * (12.0 * l - B).ln() + C
    }
}

/// BT.2100 HLG inverse transfer function: HLG → scene linear.
///
/// Input: HLG-encoded signal in [0, 1].
/// Output: normalised scene luminance in [0, 1].
#[must_use]
pub fn hlg_eotf(e: f64) -> f64 {
    const A: f64 = 0.178_832_77;
    const B: f64 = 0.284_668_92;
    const C: f64 = 0.559_910_73;
    if e <= 0.5 {
        e * e / 3.0
    } else {
        ((e - C) / A).exp() / 12.0 + B / 12.0
    }
}

// =============================================================================
// Tests: known color-conversion test vectors (Tasks 2 + 13)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert two u8 values are within `tol` of each other.
    fn approx_eq(a: u8, b: u8, tol: u8, label: &str) {
        assert!(
            (a as i32 - b as i32).unsigned_abs() as u8 <= tol,
            "{label}: got {a}, expected ~{b} (tol={tol})"
        );
    }

    // ── BT.601 reference vectors ─────────────────────────────────────────────

    #[test]
    fn test_bt601_white_rgb_to_ycbcr() {
        // White (255, 255, 255) → Y=235, Cb=128, Cr=128 (studio swing)
        let (y, cb, cr) = bt601_rgb_to_ycbcr(255, 255, 255);
        approx_eq(y, 235, 2, "Y for white");
        approx_eq(cb, 128, 2, "Cb for white");
        approx_eq(cr, 128, 2, "Cr for white");
    }

    #[test]
    fn test_bt601_black_rgb_to_ycbcr() {
        // Black (0, 0, 0) → Y=16, Cb=128, Cr=128 (studio swing)
        let (y, cb, cr) = bt601_rgb_to_ycbcr(0, 0, 0);
        approx_eq(y, 16, 2, "Y for black");
        approx_eq(cb, 128, 2, "Cb for black");
        approx_eq(cr, 128, 2, "Cr for black");
    }

    #[test]
    fn test_bt601_red_rgb_to_ycbcr() {
        // Pure red (255, 0, 0) → Y≈82, Cb≈90, Cr≈240 (per SMPTE test vectors)
        let (y, cb, cr) = bt601_rgb_to_ycbcr(255, 0, 0);
        approx_eq(y, 82, 3, "Y for red");
        approx_eq(cb, 90, 4, "Cb for red");
        approx_eq(cr, 240, 4, "Cr for red");
    }

    #[test]
    fn test_bt601_green_rgb_to_ycbcr() {
        // Pure green (0, 255, 0) → Y≈145, Cb≈54, Cr≈34
        let (y, cb, cr) = bt601_rgb_to_ycbcr(0, 255, 0);
        approx_eq(y, 145, 3, "Y for green");
        approx_eq(cb, 54, 4, "Cb for green");
        approx_eq(cr, 34, 4, "Cr for green");
    }

    #[test]
    fn test_bt601_blue_rgb_to_ycbcr() {
        // Pure blue (0, 0, 255) → Y≈41, Cb≈240, Cr≈110
        let (y, cb, cr) = bt601_rgb_to_ycbcr(0, 0, 255);
        approx_eq(y, 41, 3, "Y for blue");
        approx_eq(cb, 240, 4, "Cb for blue");
        approx_eq(cr, 110, 4, "Cr for blue");
    }

    #[test]
    fn test_bt601_roundtrip_white() {
        let (y, cb, cr) = bt601_rgb_to_ycbcr(255, 255, 255);
        let (r, g, b) = bt601_ycbcr_to_rgb(y, cb, cr);
        approx_eq(r, 255, 3, "R roundtrip white");
        approx_eq(g, 255, 3, "G roundtrip white");
        approx_eq(b, 255, 3, "B roundtrip white");
    }

    #[test]
    fn test_bt601_roundtrip_black() {
        let (y, cb, cr) = bt601_rgb_to_ycbcr(0, 0, 0);
        let (r, g, b) = bt601_ycbcr_to_rgb(y, cb, cr);
        approx_eq(r, 0, 3, "R roundtrip black");
        approx_eq(g, 0, 3, "G roundtrip black");
        approx_eq(b, 0, 3, "B roundtrip black");
    }

    #[test]
    fn test_bt601_roundtrip_grey128() {
        let (y, cb, cr) = bt601_rgb_to_ycbcr(128, 128, 128);
        let (r, g, b) = bt601_ycbcr_to_rgb(y, cb, cr);
        approx_eq(r, 128, 4, "R roundtrip grey");
        approx_eq(g, 128, 4, "G roundtrip grey");
        approx_eq(b, 128, 4, "B roundtrip grey");
    }

    // ── BT.709 reference vectors ─────────────────────────────────────────────

    #[test]
    fn test_bt709_white_rgb_to_ycbcr() {
        let (y, cb, cr) = bt709_rgb_to_ycbcr(255, 255, 255);
        approx_eq(y, 235, 2, "Y for white BT.709");
        approx_eq(cb, 128, 2, "Cb for white BT.709");
        approx_eq(cr, 128, 2, "Cr for white BT.709");
    }

    #[test]
    fn test_bt709_black_rgb_to_ycbcr() {
        let (y, cb, cr) = bt709_rgb_to_ycbcr(0, 0, 0);
        approx_eq(y, 16, 2, "Y for black BT.709");
        approx_eq(cb, 128, 2, "Cb for black BT.709");
        approx_eq(cr, 128, 2, "Cr for black BT.709");
    }

    #[test]
    fn test_bt709_red_rgb_to_ycbcr() {
        // BT.709 red: Kr=0.2126 → Y≈63+16=63... actual: Y≈63, Cb≈102, Cr≈240
        let (y, _cb, _cr) = bt709_rgb_to_ycbcr(255, 0, 0);
        // Y for pure red in BT.709: 16 + 219 * 0.2126 ≈ 62.6 ≈ 63
        approx_eq(y, 63, 3, "Y for red BT.709");
    }

    #[test]
    fn test_bt709_roundtrip_white() {
        let (y, cb, cr) = bt709_rgb_to_ycbcr(255, 255, 255);
        let (r, g, b) = bt709_ycbcr_to_rgb(y, cb, cr);
        approx_eq(r, 255, 4, "R roundtrip white BT.709");
        approx_eq(g, 255, 4, "G roundtrip white BT.709");
        approx_eq(b, 255, 4, "B roundtrip white BT.709");
    }

    #[test]
    fn test_bt709_roundtrip_black() {
        let (y, cb, cr) = bt709_rgb_to_ycbcr(0, 0, 0);
        let (r, g, b) = bt709_ycbcr_to_rgb(y, cb, cr);
        approx_eq(r, 0, 4, "R roundtrip black BT.709");
        approx_eq(g, 0, 4, "G roundtrip black BT.709");
        approx_eq(b, 0, 4, "B roundtrip black BT.709");
    }

    #[test]
    fn test_bt709_roundtrip_colour() {
        // Arbitrary colour roundtrip.
        let (y, cb, cr) = bt709_rgb_to_ycbcr(100, 150, 200);
        let (r, g, b) = bt709_ycbcr_to_rgb(y, cb, cr);
        approx_eq(r, 100, 5, "R roundtrip colour BT.709");
        approx_eq(g, 150, 5, "G roundtrip colour BT.709");
        approx_eq(b, 200, 5, "B roundtrip colour BT.709");
    }

    // ── BT.2020 reference vectors ────────────────────────────────────────────

    #[test]
    fn test_bt2020_white_rgb_to_ycbcr() {
        let (y, cb, cr) = bt2020_rgb_to_ycbcr(255, 255, 255);
        approx_eq(y, 235, 2, "Y for white BT.2020");
        approx_eq(cb, 128, 2, "Cb for white BT.2020");
        approx_eq(cr, 128, 2, "Cr for white BT.2020");
    }

    #[test]
    fn test_bt2020_black_rgb_to_ycbcr() {
        let (y, cb, cr) = bt2020_rgb_to_ycbcr(0, 0, 0);
        approx_eq(y, 16, 2, "Y for black BT.2020");
        approx_eq(cb, 128, 2, "Cb for black BT.2020");
        approx_eq(cr, 128, 2, "Cr for black BT.2020");
    }

    #[test]
    fn test_bt2020_red_luma() {
        // BT.2020 red: Kr = 0.2627 → Y = 16 + 219 * 0.2627 ≈ 73.6 ≈ 74
        let (y, _, _) = bt2020_rgb_to_ycbcr(255, 0, 0);
        approx_eq(y, 74, 3, "Y for red BT.2020");
    }

    #[test]
    fn test_bt2020_roundtrip_white() {
        let (y, cb, cr) = bt2020_rgb_to_ycbcr(255, 255, 255);
        let (r, g, b) = bt2020_ycbcr_to_rgb(y, cb, cr);
        approx_eq(r, 255, 4, "R roundtrip white BT.2020");
        approx_eq(g, 255, 4, "G roundtrip white BT.2020");
        approx_eq(b, 255, 4, "B roundtrip white BT.2020");
    }

    #[test]
    fn test_bt2020_roundtrip_colour() {
        let (y, cb, cr) = bt2020_rgb_to_ycbcr(100, 150, 200);
        let (r, g, b) = bt2020_ycbcr_to_rgb(y, cb, cr);
        approx_eq(r, 100, 5, "R roundtrip colour BT.2020");
        approx_eq(g, 150, 5, "G roundtrip colour BT.2020");
        approx_eq(b, 200, 5, "B roundtrip colour BT.2020");
    }

    // ── BT.2100 PQ / HLG transfer function tests ─────────────────────────────

    #[test]
    fn test_pq_oetf_zero() {
        // PQ(0) = 0
        let v = pq_oetf(0.0);
        assert!(v.abs() < 1e-6, "pq_oetf(0) = {v}");
    }

    #[test]
    fn test_pq_oetf_one() {
        // PQ(1.0) = 1.0 (10 000 nits maps to code 1.0)
        let v = pq_oetf(1.0);
        assert!((v - 1.0).abs() < 1e-4, "pq_oetf(1) = {v}");
    }

    #[test]
    fn test_pq_roundtrip() {
        for nits_norm in [0.0, 0.01, 0.1, 0.5, 0.9, 1.0_f64] {
            let encoded = pq_oetf(nits_norm);
            let decoded = pq_eotf(encoded);
            assert!(
                (decoded - nits_norm).abs() < 1e-5,
                "PQ roundtrip failed at {nits_norm}: got {decoded}"
            );
        }
    }

    #[test]
    fn test_hlg_oetf_zero() {
        let v = hlg_oetf(0.0);
        assert!(v.abs() < 1e-6, "hlg_oetf(0) = {v}");
    }

    #[test]
    fn test_hlg_oetf_range() {
        // All outputs must be in [0, 1] for normalised scene linear input.
        for i in 0..=20 {
            let l = i as f64 / 20.0;
            let e = hlg_oetf(l);
            assert!((0.0..=1.0).contains(&e), "hlg_oetf({l}) = {e} out of [0,1]");
        }
    }

    #[test]
    fn test_hlg_roundtrip() {
        for l in [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0_f64] {
            let encoded = hlg_oetf(l);
            let decoded = hlg_eotf(encoded);
            assert!(
                (decoded - l).abs() < 1e-6,
                "HLG roundtrip failed at {l}: got {decoded}"
            );
        }
    }

    // ── GPU vs CPU comparison tests (Task 12) ────────────────────────────────

    /// Verify that BT.601 full-image luma values are consistent between the
    /// CPU reference implementation and the per-pixel formula.
    #[test]
    fn test_bt601_cpu_vs_reference_batch() {
        let colours = [
            (255u8, 0u8, 0u8),     // red
            (0u8, 255u8, 0u8),     // green
            (0u8, 0u8, 255u8),     // blue
            (255u8, 255u8, 0u8),   // yellow
            (128u8, 128u8, 128u8), // grey
        ];
        // Expected Y values from SMPTE RP 177 / ITU-R BT.601 test vectors
        let expected_y: &[u8] = &[82, 145, 41, 210, 126];
        for (i, ((r, g, b), &ey)) in colours.iter().zip(expected_y.iter()).enumerate() {
            let (y, _, _) = bt601_rgb_to_ycbcr(*r, *g, *b);
            assert!(
                (y as i32 - ey as i32).unsigned_abs() <= 3,
                "BT.601 Y mismatch for colour {i}: got {y}, expected ~{ey}"
            );
        }
    }

    /// Compare BT.709 and BT.2020 luma for the same colour: 2020 should
    /// give different Y values than 601 for non-grey primaries.
    #[test]
    fn test_bt2020_vs_bt601_luma_differ_for_red() {
        let (y601, _, _) = bt601_rgb_to_ycbcr(255, 0, 0);
        let (y2020, _, _) = bt2020_rgb_to_ycbcr(255, 0, 0);
        // BT.601: Kr=0.299, BT.2020: Kr=0.2627 — luma for pure red must differ
        assert_ne!(y601, y2020, "BT.601 and BT.2020 Y for red must differ");
    }

    /// Verify the grey axis is luma-only: for equal RGB the colour difference
    /// component (Cb, Cr) should both be 128 across all standards.
    #[test]
    fn test_grey_axis_chroma_neutral_all_standards() {
        for v in [0u8, 64, 128, 192, 255] {
            let (_, cb601, cr601) = bt601_rgb_to_ycbcr(v, v, v);
            let (_, cb709, cr709) = bt709_rgb_to_ycbcr(v, v, v);
            let (_, cb2020, cr2020) = bt2020_rgb_to_ycbcr(v, v, v);
            approx_eq(cb601, 128, 2, &format!("Cb BT.601 grey {v}"));
            approx_eq(cr601, 128, 2, &format!("Cr BT.601 grey {v}"));
            approx_eq(cb709, 128, 2, &format!("Cb BT.709 grey {v}"));
            approx_eq(cr709, 128, 2, &format!("Cr BT.709 grey {v}"));
            approx_eq(cb2020, 128, 2, &format!("Cb BT.2020 grey {v}"));
            approx_eq(cr2020, 128, 2, &format!("Cr BT.2020 grey {v}"));
        }
    }

    // ─── Task G: Additional GPU vs CPU comparison reference tests ────────────

    /// BT.601 reference test vectors from SMPTE RP 177 / ITU-R BT.601.
    /// Verifies luma Y and chroma Cb/Cr against known standard values.
    #[test]
    fn test_bt601_reference_vectors() {
        // (R, G, B) → (Y, Cb, Cr) reference values (±2 tolerance)
        let cases: &[((u8, u8, u8), (u8, u8, u8))] = &[
            ((255, 0, 0), (82, 90, 240)),       // Red
            ((0, 255, 0), (145, 54, 34)),       // Green
            ((0, 0, 255), (41, 240, 110)),      // Blue
            ((255, 255, 255), (235, 128, 128)), // White
            ((0, 0, 0), (16, 128, 128)),        // Black
            ((128, 128, 128), (126, 128, 128)), // Mid-grey
        ];
        for &((r, g, b), (ey, ecb, ecr)) in cases {
            let (y, cb, cr) = bt601_rgb_to_ycbcr(r, g, b);
            approx_eq(y, ey, 3, &format!("Y  for ({r},{g},{b}) BT.601"));
            approx_eq(cb, ecb, 4, &format!("Cb for ({r},{g},{b}) BT.601"));
            approx_eq(cr, ecr, 4, &format!("Cr for ({r},{g},{b}) BT.601"));
        }
    }

    /// BT.709 reference test vectors from ITU-R BT.709-6 Table 1.
    #[test]
    fn test_bt709_reference_vectors() {
        // Key reference points for BT.709
        let cases: &[((u8, u8, u8), (u8, u8, u8))] = &[
            ((255, 255, 255), (235, 128, 128)), // White
            ((0, 0, 0), (16, 128, 128)),        // Black
            ((255, 0, 0), (63, 102, 240)),      // Red (Kr=0.2126)
            ((0, 255, 0), (173, 42, 26)),       // Green (Kg=0.7152)
            ((0, 0, 255), (32, 240, 118)),      // Blue (Kb=0.0722)
        ];
        for &((r, g, b), (ey, ecb, ecr)) in cases {
            let (y, cb, cr) = bt709_rgb_to_ycbcr(r, g, b);
            approx_eq(y, ey, 4, &format!("Y  for ({r},{g},{b}) BT.709"));
            approx_eq(cb, ecb, 5, &format!("Cb for ({r},{g},{b}) BT.709"));
            approx_eq(cr, ecr, 5, &format!("Cr for ({r},{g},{b}) BT.709"));
        }
    }

    /// Verify that BT.601 and BT.709 give different Y for the same colour.
    /// The two standards use different luma coefficients (Kr, Kg, Kb).
    #[test]
    fn test_bt601_vs_bt709_differ_for_primaries() {
        let test_colours = [(255u8, 0, 0), (0, 255, 0), (0, 0, 255)];
        for (r, g, b) in test_colours {
            let (y601, _, _) = bt601_rgb_to_ycbcr(r, g, b);
            let (y709, _, _) = bt709_rgb_to_ycbcr(r, g, b);
            assert_ne!(
                y601, y709,
                "BT.601 and BT.709 Y should differ for ({r},{g},{b})"
            );
        }
    }

    /// CPU↔CPU path consistency: calling bt601 twice on same input gives same result.
    #[test]
    fn test_bt601_deterministic() {
        let (y1, cb1, cr1) = bt601_rgb_to_ycbcr(100, 150, 200);
        let (y2, cb2, cr2) = bt601_rgb_to_ycbcr(100, 150, 200);
        assert_eq!(y1, y2);
        assert_eq!(cb1, cb2);
        assert_eq!(cr1, cr2);
    }

    /// Round-trip BT.709 for a batch of arbitrary colours; max drift ≤ 5.
    #[test]
    fn test_bt709_batch_roundtrip_within_tolerance() {
        let colours = [
            (10u8, 20u8, 30u8),
            (200, 100, 50),
            (64, 128, 192),
            (0, 255, 128),
            (255, 128, 0),
            (77, 77, 77),
        ];
        for (r, g, b) in colours {
            let (y, cb, cr) = bt709_rgb_to_ycbcr(r, g, b);
            let (ro, go, bo) = bt709_ycbcr_to_rgb(y, cb, cr);
            let dr = (r as i32 - ro as i32).unsigned_abs();
            let dg = (g as i32 - go as i32).unsigned_abs();
            let db = (b as i32 - bo as i32).unsigned_abs();
            assert!(
                dr <= 5 && dg <= 5 && db <= 5,
                "BT.709 roundtrip ({r},{g},{b}) → ({ro},{go},{bo}): diff=({dr},{dg},{db})"
            );
        }
    }

    // ─── HSV conversion tests ────────────────────────────────────────────────

    /// Helper: build a single RGBA pixel as a 4-element array.
    fn rgba_pixel(r: u8, g: u8, b: u8) -> Vec<u8> {
        vec![r, g, b, 255u8]
    }

    /// Pure red (255, 0, 0) in HSV should give H≈0, S≈255, V≈255.
    #[test]
    fn test_rgb_to_hsv_red() {
        let data = rgba_pixel(255, 0, 0);
        let out = ColorSpaceConversion::rgb_to_hsv(&data, 1, 1);
        // H encoded: 0/360*255 = 0
        assert!(out[0] <= 2, "H for pure red should be ~0, got {}", out[0]);
        // S = 255
        let diff_s = (out[1] as i32 - 255).unsigned_abs();
        assert!(diff_s <= 2, "S for pure red should be ~255, got {}", out[1]);
        // V = 255
        let diff_v = (out[2] as i32 - 255).unsigned_abs();
        assert!(diff_v <= 2, "V for pure red should be ~255, got {}", out[2]);
        // Alpha pass-through
        assert_eq!(out[3], 255);
    }

    /// Round-trip: RGB → HSV → RGB should be within ±2 per channel.
    #[test]
    fn test_hsv_round_trip() {
        let test_colours: &[(u8, u8, u8)] = &[
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (128, 64, 192),
            (200, 150, 100),
        ];
        for &(r, g, b) in test_colours {
            let data = rgba_pixel(r, g, b);
            let hsv = ColorSpaceConversion::rgb_to_hsv(&data, 1, 1);
            let rgb = ColorSpaceConversion::hsv_to_rgb(&hsv, 1, 1);
            let dr = (r as i32 - rgb[0] as i32).unsigned_abs();
            let dg = (g as i32 - rgb[1] as i32).unsigned_abs();
            let db = (b as i32 - rgb[2] as i32).unsigned_abs();
            assert!(
                dr <= 2 && dg <= 2 && db <= 2,
                "HSV round-trip ({r},{g},{b}) → ({},{},{}) diff=({dr},{dg},{db})",
                rgb[0],
                rgb[1],
                rgb[2]
            );
        }
    }

    // ─── Lab conversion tests ────────────────────────────────────────────────

    /// Near-grey (127,127,127) → L≈50, a≈0, b≈0.
    ///
    /// Encoding: L byte = L*255/100, a byte = a+128, b byte = b+128.
    #[test]
    fn test_rgb_to_lab_gray() {
        let data = rgba_pixel(127, 127, 127);
        let out = ColorSpaceConversion::rgb_to_lab(&data, 1, 1);

        // L* ≈ 50 for mid-grey → encoded as 50*255/100 ≈ 127
        let l_decoded = f64::from(out[0]) * 100.0 / 255.0;
        assert!(
            (l_decoded - 50.0).abs() < 4.0,
            "L* for mid-grey should be ~50, got {l_decoded:.2}"
        );

        // a* ≈ 0 → encoded as ≈128
        let a_decoded = f64::from(out[1]) - 128.0;
        assert!(
            a_decoded.abs() < 4.0,
            "a* for grey should be ~0, got {a_decoded:.2}"
        );

        // b* ≈ 0 → encoded as ≈128
        let b_decoded = f64::from(out[2]) - 128.0;
        assert!(
            b_decoded.abs() < 4.0,
            "b* for grey should be ~0, got {b_decoded:.2}"
        );
    }

    // ─── sRGB ↔ Linear round-trip tests ─────────────────────────────────────

    /// Convert 10 representative values through sRGB→Linear→sRGB within 0.01.
    #[test]
    fn test_srgb_linear_round_trip() {
        let test_values: &[u8] = &[0, 10, 30, 64, 100, 128, 180, 200, 230, 255];
        for &v in test_values {
            let data = vec![v, v, v, 255u8];
            // sRGB → Linear
            let linear = ColorSpaceConversion::srgb_to_linear(&data, 1, 1);
            // Linear → sRGB
            let recovered = ColorSpaceConversion::linear_to_srgb(&linear, 1, 1);
            let diff = (v as i32 - recovered[0] as i32).unsigned_abs();
            assert!(
                diff <= 3,
                "sRGB↔Linear round-trip failed for v={v}: recovered={}, diff={diff}",
                recovered[0]
            );
        }
    }

    /// Verify that `srgb_to_linear` monotonically increases.
    #[test]
    fn test_srgb_to_linear_monotone() {
        let mut prev = 0u8;
        for v in 1u8..=255 {
            let data = vec![v, v, v, 255u8];
            let lin = ColorSpaceConversion::srgb_to_linear(&data, 1, 1);
            assert!(
                lin[0] >= prev,
                "sRGB→Linear not monotone at v={v}: prev={prev}, got={}",
                lin[0]
            );
            prev = lin[0];
        }
    }
}
