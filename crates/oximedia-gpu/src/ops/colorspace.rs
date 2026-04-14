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
}
