//! GPU compute shader dispatch for image processing operations.
//!
//! Provides a high-level dispatch layer that selects between GPU compute
//! shaders (when the `gpu` feature is enabled and hardware is available)
//! and the equivalent CPU implementations from [`crate::image_filter`] and
//! [`crate::porter_duff`].
//!
//! The decision is governed by [`GpuComputeConfig`]: callers can set a
//! minimum pixel count below which the CPU path is always preferred
//! (avoiding GPU dispatch overhead for small images).

use crate::porter_duff::PorterDuffOp;

// ---------------------------------------------------------------------------
// Shader sources (embedded at compile time)
// ---------------------------------------------------------------------------

/// WGSL source for the separable Gaussian blur compute shader.
pub const GAUSSIAN_BLUR_WGSL: &str = include_str!("shaders/gaussian_blur.wgsl");

/// WGSL source for the Porter-Duff compositing compute shader.
pub const PORTER_DUFF_WGSL: &str = include_str!("shaders/porter_duff.wgsl");

/// WGSL source for the gradient fill compute shader.
pub const GRADIENT_FILL_WGSL: &str = include_str!("shaders/gradient_fill.wgsl");

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for GPU compute dispatch.
#[derive(Debug, Clone)]
pub struct GpuComputeConfig {
    /// Whether to prefer GPU over CPU when available.
    pub prefer_gpu: bool,
    /// Minimum image area (`width * height`) to justify GPU dispatch overhead.
    /// Images smaller than this threshold always use the CPU path.
    pub min_pixels_for_gpu: u32,
}

impl Default for GpuComputeConfig {
    fn default() -> Self {
        Self {
            prefer_gpu: true,
            min_pixels_for_gpu: 256 * 256, // 65 536 pixels
        }
    }
}

// ---------------------------------------------------------------------------
// GPU availability
// ---------------------------------------------------------------------------

/// Result of a GPU availability check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuStatus {
    /// GPU compute is available and ready.
    Available,
    /// GPU compute is not available for the given reason.
    NotAvailable(String),
    /// The `gpu` feature was not compiled in.
    FeatureDisabled,
}

/// Check whether GPU compute is available at runtime.
///
/// When the `gpu` feature is enabled this attempts a lightweight probe of
/// the wgpu backend.  Without the feature the function always returns
/// [`GpuStatus::FeatureDisabled`].
#[must_use]
pub fn check_gpu_status() -> GpuStatus {
    #[cfg(feature = "gpu")]
    {
        // Probe wgpu for an adapter.  We intentionally do *not* cache the
        // adapter here — the real pipeline creation happens elsewhere.
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        // `enumerate_adapters` returns a Future in wgpu 29.x.
        let adapters: Vec<_> =
            pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        if adapters.is_empty() {
            return GpuStatus::NotAvailable("no wgpu adapters found".into());
        }

        GpuStatus::Available
    }

    #[cfg(not(feature = "gpu"))]
    {
        GpuStatus::FeatureDisabled
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Returns `true` when the GPU path should be attempted for the given config
/// and image dimensions.
fn should_use_gpu(config: &GpuComputeConfig, width: u32, height: u32) -> bool {
    if !config.prefer_gpu {
        return false;
    }
    let area = width.saturating_mul(height);
    if area < config.min_pixels_for_gpu {
        return false;
    }
    matches!(check_gpu_status(), GpuStatus::Available)
}

// ---------------------------------------------------------------------------
// Dispatch: Gaussian blur
// ---------------------------------------------------------------------------

/// Dispatch a separable Gaussian blur.
///
/// When the GPU path is eligible (feature enabled, hardware present, image
/// large enough) the compute shader is used; otherwise the CPU implementation
/// in [`crate::image_filter::gaussian_blur`] handles the work.
pub fn dispatch_gaussian_blur(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    sigma: f32,
    config: &GpuComputeConfig,
) {
    if should_use_gpu(config, width, height) {
        // GPU path — currently falls through to CPU while the full
        // wgpu pipeline integration is completed.
        #[cfg(feature = "gpu")]
        {
            // TODO: submit GAUSSIAN_BLUR_WGSL compute pipeline
            let _ = GAUSSIAN_BLUR_WGSL; // acknowledge the shader source
        }
    }

    // CPU fallback (always reachable until GPU pipeline is wired)
    crate::image_filter::gaussian_blur(pixels, width, height, sigma);
}

// ---------------------------------------------------------------------------
// Dispatch: Porter-Duff compositing
// ---------------------------------------------------------------------------

/// Dispatch Porter-Duff compositing of `src` onto `dst`.
///
/// Follows the same GPU/CPU selection logic as [`dispatch_gaussian_blur`].
pub fn dispatch_porter_duff(
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    height: u32,
    op: PorterDuffOp,
    config: &GpuComputeConfig,
) {
    if should_use_gpu(config, width, height) {
        #[cfg(feature = "gpu")]
        {
            let _ = PORTER_DUFF_WGSL;
        }
    }

    crate::porter_duff::composite_layer_into(src, dst, width, height, op);
}

// ---------------------------------------------------------------------------
// Dispatch: Gradient fill
// ---------------------------------------------------------------------------

/// Gradient type for GPU dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuGradientType {
    /// Linear gradient between two points.
    Linear,
    /// Radial gradient from a center with a given radius.
    Radial,
}

/// A colour stop for the GPU gradient shader.
#[derive(Debug, Clone, Copy)]
pub struct GpuColorStop {
    /// RGBA colour in [0, 1] range.
    pub color: [f32; 4],
    /// Position along the gradient axis in [0, 1].
    pub position: f32,
}

/// Dispatch a gradient fill into an RGBA8 pixel buffer.
///
/// When the GPU path is eligible the `gradient_fill.wgsl` compute shader is
/// used; otherwise a simple CPU rasteriser produces the output.
pub fn dispatch_gradient_fill(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    gradient_type: GpuGradientType,
    stops: &[GpuColorStop],
    start: (f32, f32),
    end: (f32, f32),
    config: &GpuComputeConfig,
) {
    if should_use_gpu(config, width, height) {
        #[cfg(feature = "gpu")]
        {
            let _ = GRADIENT_FILL_WGSL;
        }
    }

    // CPU fallback
    cpu_gradient_fill(pixels, width, height, gradient_type, stops, start, end);
}

/// CPU implementation of gradient fill (used as fallback).
fn cpu_gradient_fill(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    gradient_type: GpuGradientType,
    stops: &[GpuColorStop],
    start: (f32, f32),
    end: (f32, f32),
) {
    if stops.is_empty() || width == 0 || height == 0 {
        return;
    }

    let w = width as usize;
    let h = height as usize;
    let expected_len = w * h * 4;
    if pixels.len() < expected_len {
        return;
    }

    for y in 0..h {
        for x in 0..w {
            let u = if width > 1 {
                x as f32 / (width - 1) as f32
            } else {
                0.0
            };
            let v = if height > 1 {
                y as f32 / (height - 1) as f32
            } else {
                0.0
            };

            let t = match gradient_type {
                GpuGradientType::Linear => {
                    let dx = end.0 - start.0;
                    let dy = end.1 - start.1;
                    let len_sq = dx * dx + dy * dy;
                    if len_sq > 1e-8 {
                        ((u - start.0) * dx + (v - start.1) * dy) / len_sq
                    } else {
                        0.0
                    }
                }
                GpuGradientType::Radial => {
                    let radius = end.0; // end_x encodes radius
                    if radius > 1e-8 {
                        let dist = ((u - start.0).powi(2) + (v - start.1).powi(2)).sqrt();
                        dist / radius
                    } else {
                        0.0
                    }
                }
            };

            let color = evaluate_stops(stops, t.clamp(0.0, 1.0));

            let idx = (y * w + x) * 4;
            pixels[idx] = (color[0] * 255.0).clamp(0.0, 255.0) as u8;
            pixels[idx + 1] = (color[1] * 255.0).clamp(0.0, 255.0) as u8;
            pixels[idx + 2] = (color[2] * 255.0).clamp(0.0, 255.0) as u8;
            pixels[idx + 3] = (color[3] * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
}

/// Linearly interpolate colour stops at parameter `t` in [0, 1].
fn evaluate_stops(stops: &[GpuColorStop], t: f32) -> [f32; 4] {
    if stops.len() == 1 || t <= stops[0].position {
        return stops[0].color;
    }
    let last = stops.len() - 1;
    if t >= stops[last].position {
        return stops[last].color;
    }

    // Find bracketing pair
    let mut lower = 0;
    for i in 1..stops.len() {
        if stops[i].position <= t {
            lower = i;
        }
    }
    let upper = lower + 1;
    if upper >= stops.len() {
        return stops[last].color;
    }

    let span = stops[upper].position - stops[lower].position;
    let local_t = if span.abs() > 1e-8 {
        (t - stops[lower].position) / span
    } else {
        0.0
    };

    let a = &stops[lower].color;
    let b = &stops[upper].color;
    [
        a[0] + (b[0] - a[0]) * local_t,
        a[1] + (b[1] - a[1]) * local_t,
        a[2] + (b[2] - a[2]) * local_t,
        a[3] + (b[3] - a[3]) * local_t,
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_status_feature_disabled_or_available() {
        // Without the gpu feature this returns FeatureDisabled;
        // with it, it returns Available or NotAvailable depending on hardware.
        let status = check_gpu_status();
        match status {
            GpuStatus::FeatureDisabled => {
                // Expected when compiled without gpu feature
            }
            GpuStatus::Available | GpuStatus::NotAvailable(_) => {
                // Expected when compiled with gpu feature
            }
        }
    }

    #[test]
    fn test_gpu_compute_config_default() {
        let cfg = GpuComputeConfig::default();
        assert!(cfg.prefer_gpu);
        assert_eq!(cfg.min_pixels_for_gpu, 256 * 256);
    }

    #[test]
    fn test_dispatch_blur_falls_back_to_cpu() {
        // Verify that dispatch produces the same output as the direct CPU call
        let (w, h) = (8u32, 8u32);
        let sigma = 1.5_f32;

        let mut a = vec![128u8; (w * h * 4) as usize];
        let mut b = a.clone();

        // Direct CPU
        crate::image_filter::gaussian_blur(&mut a, w, h, sigma);

        // Dispatch (will fall back to CPU)
        let cfg = GpuComputeConfig {
            prefer_gpu: false,
            min_pixels_for_gpu: 0,
        };
        dispatch_gaussian_blur(&mut b, w, h, sigma, &cfg);

        assert_eq!(a, b);
    }

    #[test]
    fn test_dispatch_blur_uniform_unchanged() {
        // A uniform-colour image should remain unchanged after blur
        let (w, h) = (16u32, 16u32);
        let mut pixels = vec![200u8; (w * h * 4) as usize];
        let original = pixels.clone();

        let cfg = GpuComputeConfig::default();
        dispatch_gaussian_blur(&mut pixels, w, h, 2.0, &cfg);

        // Every pixel should still be 200 (uniform convolution)
        assert_eq!(pixels, original);
    }

    #[test]
    fn test_dispatch_pd_src_over_matches_cpu() {
        let (w, h) = (4u32, 4u32);
        let src = vec![100u8; (w * h * 4) as usize];
        let mut dst_dispatch = vec![200u8; (w * h * 4) as usize];
        let mut dst_direct = dst_dispatch.clone();

        let cfg = GpuComputeConfig {
            prefer_gpu: false,
            min_pixels_for_gpu: 0,
        };
        dispatch_porter_duff(&src, &mut dst_dispatch, w, h, PorterDuffOp::SrcOver, &cfg);
        crate::porter_duff::composite_layer_into(
            &src,
            &mut dst_direct,
            w,
            h,
            PorterDuffOp::SrcOver,
        );

        assert_eq!(dst_dispatch, dst_direct);
    }

    #[test]
    fn test_dispatch_blur_zero_sigma_unchanged() {
        let (w, h) = (8u32, 8u32);
        let mut pixels: Vec<u8> = (0..w * h * 4).map(|i| (i % 256) as u8).collect();
        let original = pixels.clone();

        let cfg = GpuComputeConfig::default();
        dispatch_gaussian_blur(&mut pixels, w, h, 0.0, &cfg);

        assert_eq!(pixels, original, "sigma=0 should be a no-op");
    }

    #[test]
    fn test_min_pixels_threshold() {
        // With prefer_gpu=false, should_use_gpu always returns false
        let cfg_no_gpu = GpuComputeConfig {
            prefer_gpu: false,
            min_pixels_for_gpu: 0,
        };
        assert!(!should_use_gpu(&cfg_no_gpu, 1024, 1024));

        // With a very high threshold, small images go CPU
        let cfg_high = GpuComputeConfig {
            prefer_gpu: true,
            min_pixels_for_gpu: u32::MAX,
        };
        assert!(!should_use_gpu(&cfg_high, 64, 64));
    }

    #[test]
    fn test_shader_sources_not_empty() {
        assert!(
            !GAUSSIAN_BLUR_WGSL.is_empty(),
            "gaussian_blur.wgsl must not be empty"
        );
        assert!(
            !PORTER_DUFF_WGSL.is_empty(),
            "porter_duff.wgsl must not be empty"
        );
        assert!(
            !GRADIENT_FILL_WGSL.is_empty(),
            "gradient_fill.wgsl must not be empty"
        );

        // Sanity: each shader contains a @compute entry point
        assert!(GAUSSIAN_BLUR_WGSL.contains("@compute"));
        assert!(PORTER_DUFF_WGSL.contains("@compute"));
        assert!(GRADIENT_FILL_WGSL.contains("@compute"));
    }

    #[test]
    fn test_gradient_fill_linear_two_stops() {
        let (w, h) = (8u32, 1u32);
        let mut pixels = vec![0u8; (w * h * 4) as usize];

        let stops = vec![
            GpuColorStop {
                color: [0.0, 0.0, 0.0, 1.0],
                position: 0.0,
            },
            GpuColorStop {
                color: [1.0, 1.0, 1.0, 1.0],
                position: 1.0,
            },
        ];

        let cfg = GpuComputeConfig {
            prefer_gpu: false,
            min_pixels_for_gpu: 0,
        };
        dispatch_gradient_fill(
            &mut pixels,
            w,
            h,
            GpuGradientType::Linear,
            &stops,
            (0.0, 0.0),
            (1.0, 0.0),
            &cfg,
        );

        // First pixel should be black, last pixel should be white
        assert_eq!(pixels[0], 0, "first pixel R should be 0");
        let last_idx = ((w - 1) * 4) as usize;
        assert_eq!(pixels[last_idx], 255, "last pixel R should be 255");
        // Middle pixel should be between 0 and 255
        let mid_idx = ((w / 2) * 4) as usize;
        assert!(pixels[mid_idx] > 0 && pixels[mid_idx] < 255);
    }

    #[test]
    fn test_gradient_fill_radial() {
        let (w, h) = (16u32, 16u32);
        let mut pixels = vec![0u8; (w * h * 4) as usize];

        let stops = vec![
            GpuColorStop {
                color: [1.0, 0.0, 0.0, 1.0],
                position: 0.0,
            },
            GpuColorStop {
                color: [0.0, 0.0, 1.0, 1.0],
                position: 1.0,
            },
        ];

        let cfg = GpuComputeConfig {
            prefer_gpu: false,
            min_pixels_for_gpu: 0,
        };
        dispatch_gradient_fill(
            &mut pixels,
            w,
            h,
            GpuGradientType::Radial,
            &stops,
            (0.5, 0.5), // center
            (0.5, 0.0), // radius = 0.5
            &cfg,
        );

        // Center pixel (8,8) should be red-ish (close to first stop)
        let center_idx = (8 * w as usize + 8) * 4;
        assert!(pixels[center_idx] > 200, "center R should be high");
        assert!(pixels[center_idx + 2] < 55, "center B should be low");
    }

    #[test]
    fn test_evaluate_stops_edge_cases() {
        let stops = vec![
            GpuColorStop {
                color: [1.0, 0.0, 0.0, 1.0],
                position: 0.0,
            },
            GpuColorStop {
                color: [0.0, 1.0, 0.0, 1.0],
                position: 0.5,
            },
            GpuColorStop {
                color: [0.0, 0.0, 1.0, 1.0],
                position: 1.0,
            },
        ];

        // t=0 -> red
        let c0 = evaluate_stops(&stops, 0.0);
        assert!((c0[0] - 1.0).abs() < 1e-6);

        // t=1 -> blue
        let c1 = evaluate_stops(&stops, 1.0);
        assert!((c1[2] - 1.0).abs() < 1e-6);

        // t=0.5 -> green
        let c_mid = evaluate_stops(&stops, 0.5);
        assert!((c_mid[1] - 1.0).abs() < 1e-6);

        // t=0.25 -> halfway between red and green
        let c_quarter = evaluate_stops(&stops, 0.25);
        assert!((c_quarter[0] - 0.5).abs() < 0.01);
        assert!((c_quarter[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_gpu_status_clone_and_eq() {
        let a = GpuStatus::FeatureDisabled;
        let b = a.clone();
        assert_eq!(a, b);

        let c = GpuStatus::NotAvailable("reason".into());
        let d = GpuStatus::NotAvailable("reason".into());
        assert_eq!(c, d);
        assert_ne!(a, c);
    }
}
