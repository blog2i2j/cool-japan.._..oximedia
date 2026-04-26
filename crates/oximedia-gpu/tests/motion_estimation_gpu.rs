//! Integration tests for GPU motion estimation.
//!
//! These tests require an actual wgpu adapter. When no adapter is available
//! (headless CI without a GPU or software renderer) the tests are skipped
//! gracefully with an eprintln! and an early return, so `cargo nextest` still
//! reports them as passed.

#[cfg(not(target_arch = "wasm32"))]
mod gpu_tests {
    use oximedia_gpu::motion_estimation::{
        BlockPartition, MotionEstimationConfig, MotionEstimator,
    };
    use oximedia_gpu::GpuDevice;

    /// Build a deterministic pseudo-random luma frame (LCG).
    fn make_noise_frame(w: u32, h: u32, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut frame = vec![0u8; (w * h) as usize];
        for pixel in frame.iter_mut() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *pixel = ((state >> 33) & 0xFF) as u8;
        }
        frame
    }

    /// Shift `src` by (dx, dy) pixels; border fills with mid-grey (128).
    fn shift_frame(src: &[u8], w: u32, h: u32, dx: i32, dy: i32) -> Vec<u8> {
        let mut dst = vec![128u8; (w * h) as usize];
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let sx = x - dx;
                let sy = y - dy;
                if sx >= 0 && sy >= 0 && sx < w as i32 && sy < h as i32 {
                    dst[(y as usize) * w as usize + x as usize] =
                        src[(sy as usize) * w as usize + sx as usize];
                }
            }
        }
        dst
    }

    /// Try to obtain a `GpuDevice`. Returns `None` when no wgpu adapter is
    /// available on the current system (CI without GPU / software renderer).
    fn try_gpu_device() -> Option<GpuDevice> {
        // Use wgpu::Instance directly to probe for any adapter without
        // panicking if none is available.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));
        if adapter.is_err() {
            return None;
        }
        GpuDevice::new(None).ok()
    }

    // ── Test 1: GPU motion estimation on a translated frame ──────────────────

    #[test]
    fn test_gpu_motion_estimation_translated_frame() {
        let device = match try_gpu_device() {
            Some(d) if !d.is_fallback => d,
            _ => {
                eprintln!(
                    "No real GPU adapter available — \
                     skipping GPU motion estimation test"
                );
                return;
            }
        };

        let w = 64u32;
        let h = 64u32;
        let shift_x = 4i32;
        let shift_y = 2i32;

        let reference = make_noise_frame(w, h, 0x1234_ABCD_EF01);
        let current = shift_frame(&reference, w, h, shift_x, shift_y);

        let estimator = MotionEstimator::new(MotionEstimationConfig {
            partition: BlockPartition::Fixed16x16,
            search_radius: 8,
            subpixel_refinement: false,
            pyramid_levels: 1,
            ..MotionEstimationConfig::default()
        });

        let result = estimator
            .estimate(&device, &reference, &current, w, h)
            .expect("GPU estimation must not fail with a real device");

        assert!(
            result.used_gpu,
            "expected GPU path to be used, got CPU fallback"
        );
        assert_eq!(result.width, w);
        assert_eq!(result.height, h);
        assert_eq!(
            result.block_mvs.len(),
            (result.blocks_x() * result.blocks_y()) as usize
        );

        // Most interior blocks (not at the border) should detect the true
        // shift within ±1 pixel tolerance.
        //
        // shift_frame(ref, w, h, shift_x, shift_y) produces:
        //   current[y][x] = reference[y - shift_y][x - shift_x]
        //
        // The block matcher finds (dx,dy) such that ref[bx+dx, by+dy] best
        // matches cur[bx, by], so the expected optimal MV is:
        //   dx = -shift_x,  dy = -shift_y
        let expected_dx = -shift_x as i16;
        let expected_dy = -shift_y as i16;

        let interior: Vec<_> = result
            .block_mvs
            .iter()
            .filter(|b| {
                b.block_x + 16 < w && b.block_y + 16 < h && b.block_x >= 8 && b.block_y >= 8
            })
            .collect();

        if interior.is_empty() {
            // Frame too small to have interior blocks at this block size; just
            // verify the result is structurally valid.
            return;
        }

        let matched = interior
            .iter()
            .filter(|b| (b.mv.dx - expected_dx).abs() <= 1 && (b.mv.dy - expected_dy).abs() <= 1)
            .count();

        assert!(
            matched >= interior.len() / 2,
            "expected at least half of interior blocks to detect shift ({shift_x},{shift_y}), \
             expected MV ({expected_dx},{expected_dy}), \
             got {matched}/{total}",
            total = interior.len()
        );
    }

    // ── Test 2: GPU result is used_gpu=true on a real device ─────────────────

    #[test]
    fn test_gpu_flag_set() {
        let device = match try_gpu_device() {
            Some(d) if !d.is_fallback => d,
            _ => {
                eprintln!("No real GPU adapter — skipping test_gpu_flag_set");
                return;
            }
        };

        let w = 32u32;
        let h = 32u32;
        let frame = vec![128u8; (w * h) as usize];

        let estimator = MotionEstimator::new(MotionEstimationConfig {
            partition: BlockPartition::Fixed16x16,
            search_radius: 4,
            subpixel_refinement: false,
            pyramid_levels: 1,
            ..MotionEstimationConfig::default()
        });

        let result = estimator
            .estimate(&device, &frame, &frame, w, h)
            .expect("estimation must succeed");

        assert!(result.used_gpu, "result.used_gpu must be true on real GPU");
    }

    // ── Test 3: fallback device produces valid CPU results ───────────────────

    #[test]
    fn test_fallback_device_cpu_path() {
        // Use the fallback (software/CPU) device; always available.
        let device = match GpuDevice::new_fallback() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("No fallback device — skipping test_fallback_device_cpu_path");
                return;
            }
        };

        let w = 32u32;
        let h = 32u32;
        let frame = vec![64u8; (w * h) as usize];

        let estimator = MotionEstimator::new(MotionEstimationConfig {
            partition: BlockPartition::Fixed16x16,
            search_radius: 2,
            subpixel_refinement: false,
            pyramid_levels: 1,
            ..MotionEstimationConfig::default()
        });

        let result = estimator
            .estimate(&device, &frame, &frame, w, h)
            .expect("CPU fallback must succeed");

        // Fallback device always uses CPU path.
        assert!(!result.used_gpu, "fallback device must use CPU path");
        assert_eq!(result.mean_mv_magnitude(), 0.0);
    }

    // ── Test 4: GPU subpixel refinement produces f32 precision MVs ───────────

    #[test]
    fn test_gpu_subpixel_refinement_enabled() {
        let device = match try_gpu_device() {
            Some(d) if !d.is_fallback => d,
            _ => {
                eprintln!("No real GPU — skipping test_gpu_subpixel_refinement_enabled");
                return;
            }
        };

        let w = 32u32;
        let h = 32u32;
        let reference = make_noise_frame(w, h, 0xDEAD_BEEF);
        let current = shift_frame(&reference, w, h, 2, 1);

        let estimator = MotionEstimator::new(MotionEstimationConfig {
            partition: BlockPartition::Fixed16x16,
            search_radius: 4,
            subpixel_refinement: true,
            pyramid_levels: 1,
            ..MotionEstimationConfig::default()
        });

        let result = estimator
            .estimate(&device, &reference, &current, w, h)
            .expect("estimation must succeed");

        if result.used_gpu {
            for b in &result.block_mvs {
                assert!(
                    b.subpixel_mv.is_some(),
                    "subpixel_mv must be populated when subpixel_refinement=true"
                );
            }
        }
    }

    // ── Test 5: GPU works with multi-level pyramid ────────────────────────────

    #[test]
    fn test_gpu_multi_level_pyramid() {
        let device = match try_gpu_device() {
            Some(d) if !d.is_fallback => d,
            _ => {
                eprintln!("No real GPU — skipping test_gpu_multi_level_pyramid");
                return;
            }
        };

        let w = 128u32;
        let h = 128u32;
        let reference = make_noise_frame(w, h, 0xCAFE_BABE);
        let current = shift_frame(&reference, w, h, 8, 4);

        let estimator = MotionEstimator::new(MotionEstimationConfig {
            partition: BlockPartition::Fixed16x16,
            search_radius: 16,
            subpixel_refinement: false,
            pyramid_levels: 3,
            ..MotionEstimationConfig::default()
        });

        let result = estimator
            .estimate(&device, &reference, &current, w, h)
            .expect("multi-level estimation must succeed");

        assert_eq!(result.width, w);
        assert_eq!(result.height, h);
        assert!(!result.block_mvs.is_empty());
    }
}
