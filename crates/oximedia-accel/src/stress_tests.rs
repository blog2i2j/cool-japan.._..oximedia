//! Stress tests and concurrent operation tests for oximedia-accel.
//!
//! Tests for:
//! - Concurrent `scale_image` + `convert_color` operations
//! - `memory_arena` allocation/deallocation patterns
//! - Device selection fallback verification

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::cpu_simd::{scale_bilinear_cpu, yuv_to_rgb_cpu};
    use crate::memory_arena::MemoryArena;

    // ── Concurrent scale + convert stress tests ───────────────────────────

    /// Stress test: 8 threads concurrently run bilinear scale + YUV→RGB.
    ///
    /// Verifies that all operations complete without panic, data races,
    /// or incorrect results when run concurrently. Each thread operates
    /// on its own independent data (no shared mutable state).
    #[test]
    fn concurrent_scale_and_convert_no_panic() {
        use std::sync::Arc;
        use std::thread;

        // Source data is read-only and shared via Arc
        let src_rgb = Arc::new(make_test_rgb(64, 64));
        let src_yuv = Arc::new(make_test_yuv420p(64, 64));

        let handles: Vec<_> = (0..8)
            .map(|_i| {
                let rgb_src = src_rgb.clone();
                let yuv_src = src_yuv.clone();

                thread::spawn(move || {
                    // Scale 64x64 → 32x32
                    let mut dst_rgb = vec![0u8; 32 * 32 * 3];
                    scale_bilinear_cpu(&rgb_src, 64, 64, &mut dst_rgb, 32, 32, 3);

                    // Convert YUV420p → RGB
                    let mut rgb_out = vec![0u8; 64 * 64 * 3];
                    yuv_to_rgb_cpu(&yuv_src, &mut rgb_out, 64, 64);

                    // Verify outputs are not trivially wrong
                    assert!(!dst_rgb.is_empty(), "scaled output must not be empty");
                    assert!(!rgb_out.is_empty(), "converted output must not be empty");

                    // The RGB output for a mid-gray YUV frame should be approximately 128
                    let mid_gray_sum: u64 = rgb_out.iter().map(|&v| u64::from(v)).sum();
                    let mean = mid_gray_sum / rgb_out.len() as u64;
                    assert!(
                        mean.abs_diff(128) < 20,
                        "mid-gray YUV→RGB mean should be ~128, got {mean}"
                    );

                    (dst_rgb, rgb_out)
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread should not panic");
        }
    }

    /// Stress test with larger 1920x1080 → 960x540 downscale across 4 threads.
    #[test]
    fn concurrent_large_scale_no_panic() {
        use std::sync::Arc;
        use std::thread;

        // Use small images to keep test fast; the concurrency is what matters
        let src = Arc::new(make_test_rgb(192, 108));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let s = src.clone();
                thread::spawn(move || {
                    let mut dst = vec![0u8; 96 * 54 * 3];
                    scale_bilinear_cpu(&s, 192, 108, &mut dst, 96, 54, 3);
                    assert!(!dst.is_empty());
                    dst
                })
            })
            .collect();

        // Verify all threads produce identical results (deterministic)
        let results: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().expect("thread join"))
            .collect();

        for i in 1..results.len() {
            assert_eq!(
                results[0], results[i],
                "concurrent scaling must produce identical results"
            );
        }
    }

    // ── Memory arena concurrent tests ─────────────────────────────────────

    /// Concurrent memory arena allocation/deallocation via Mutex.
    ///
    /// MemoryArena uses `&mut self` — concurrent access requires a Mutex wrapper.
    /// This test verifies that 4 threads each performing 100 alloc/reset cycles
    /// complete without panic and leave the arena in a valid state.
    #[test]
    fn memory_arena_concurrent_alloc_reset_no_panic() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let arena = Arc::new(Mutex::new(MemoryArena::new(1024 * 1024)));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let a = arena.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        let mut locked = a.lock().expect("mutex lock");
                        // Allocate a 1 KiB block
                        let alloc = locked.allocate(1024, 16);
                        assert!(alloc.is_some(), "1 KiB alloc should succeed in 1 MiB arena");

                        // Occasionally reset to exercise deallocation path
                        if locked.used() > 512 * 1024 {
                            locked.reset();
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread should not panic");
        }

        // Final state must be valid
        let arena_final = arena.lock().expect("final lock");
        assert_eq!(arena_final.capacity(), 1024 * 1024);
        // Used must be <= capacity
        assert!(arena_final.used() <= arena_final.capacity());
    }

    /// Sequential alloc/dealloc cycle verifying no memory leak in live_alloc_count.
    #[test]
    fn memory_arena_alloc_reset_no_leak() {
        let mut arena = MemoryArena::new(64 * 1024);

        for cycle in 0..50 {
            // Allocate several blocks
            for _ in 0..10 {
                arena
                    .allocate(512, 16)
                    .unwrap_or_else(|| panic!("alloc should succeed in cycle {cycle}"));
            }
            assert_eq!(
                arena.live_alloc_count(),
                10,
                "cycle {cycle}: should have 10 live allocs"
            );

            // Reset → no live allocs
            arena.reset();
            assert_eq!(
                arena.live_alloc_count(),
                0,
                "cycle {cycle}: after reset should be 0"
            );
            assert_eq!(
                arena.used(),
                0,
                "cycle {cycle}: after reset used should be 0"
            );
        }
    }

    /// Verifies that stats.peak_used is never exceeded even across many cycles.
    #[test]
    fn memory_arena_stats_peak_correct() {
        let mut arena = MemoryArena::new(10_000);

        // Allocate 5000 bytes
        arena.allocate(5000, 1).expect("alloc 5000");
        let peak_after_first = arena.stats().peak_used;
        assert_eq!(peak_after_first, 5000);

        arena.reset();

        // Allocate 3000 bytes → peak stays at 5000
        arena.allocate(3000, 1).expect("alloc 3000");
        assert_eq!(arena.stats().peak_used, 5000, "peak should remain at 5000");

        // Allocate enough to exceed previous peak
        arena.allocate(3000, 1).expect("alloc another 3000");
        assert_eq!(arena.stats().peak_used, 6000, "peak should update to 6000");
    }

    /// Stress test: many small allocations filling the arena, then reset.
    #[test]
    fn memory_arena_many_small_allocs() {
        let mut arena = MemoryArena::new(8192);
        let mut count = 0usize;

        loop {
            match arena.allocate(64, 8) {
                Some(_) => count += 1,
                None => break,
            }
        }

        assert!(count > 0, "should have made some allocations");
        assert_eq!(arena.live_alloc_count(), count);

        arena.reset();
        assert_eq!(arena.live_alloc_count(), 0);
        assert_eq!(arena.used(), 0);
    }

    // ── Device fallback tests ─────────────────────────────────────────────

    /// Verifies that `AccelContext::cpu_only()` returns a CPU backend.
    ///
    /// This covers the "no Vulkan device available" scenario: the CPU backend
    /// must always be functional even when GPU initialization fails.
    #[test]
    fn device_fallback_cpu_only_is_available() {
        let ctx = crate::AccelContext::cpu_only();
        assert!(
            !ctx.is_gpu_accelerated(),
            "cpu_only must not be GPU accelerated"
        );
        assert_eq!(ctx.backend_name(), "CPU");
    }

    /// Verifies that `AccelContext::new()` succeeds (GPU or CPU fallback).
    #[test]
    fn device_fallback_new_does_not_panic() {
        // new() should always succeed: GPU if available, CPU otherwise
        let ctx = crate::AccelContext::new().expect("AccelContext::new should not fail");
        // Either GPU or CPU — just verify backend_name is non-empty
        assert!(
            !ctx.backend_name().is_empty(),
            "backend name must not be empty"
        );
    }

    /// Verifies that the CPU backend can perform scaling after fallback.
    #[test]
    fn device_fallback_cpu_can_scale() {
        use crate::traits::{HardwareAccel, ScaleFilter};
        use oximedia_core::PixelFormat;

        let ctx = crate::AccelContext::cpu_only();
        let src = vec![128u8; 8 * 8 * 3];
        let result = ctx
            .scale_image(&src, 8, 8, 4, 4, PixelFormat::Rgb24, ScaleFilter::Bilinear)
            .expect("CPU scale should succeed");
        assert_eq!(result.len(), 4 * 4 * 3, "output must be 4x4x3 bytes");
    }

    /// Verifies that the CPU backend can perform color conversion after fallback.
    #[test]
    fn device_fallback_cpu_can_convert_color() {
        use crate::traits::HardwareAccel;
        use oximedia_core::PixelFormat;

        let ctx = crate::AccelContext::cpu_only();
        let yuv = make_test_yuv420p(8, 8);
        let result = ctx
            .convert_color(&yuv, 8, 8, PixelFormat::Yuv420p, PixelFormat::Rgb24)
            .expect("CPU color conversion should succeed");
        assert_eq!(result.len(), 8 * 8 * 3, "RGB output must be w*h*3 bytes");
    }

    /// Verifies that the CPU backend handles motion estimation.
    #[test]
    fn device_fallback_cpu_motion_estimation() {
        use crate::traits::HardwareAccel;

        let ctx = crate::AccelContext::cpu_only();
        let reference = vec![100u8; 16 * 16];
        let current = vec![120u8; 16 * 16];
        let mvs = ctx
            .motion_estimation(&reference, &current, 16, 16, 4)
            .expect("motion estimation should succeed");
        assert!(!mvs.is_empty(), "should return motion vectors");
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /// Build a test RGB image with a gradient pattern.
    fn make_test_rgb(w: u32, h: u32) -> Vec<u8> {
        let mut buf = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 3) as usize;
                buf[idx] = ((x * 255) / w.max(1)) as u8;
                buf[idx + 1] = ((y * 255) / h.max(1)) as u8;
                buf[idx + 2] = 128u8;
            }
        }
        buf
    }

    /// Build a mid-gray YUV420p test frame.
    fn make_test_yuv420p(w: u32, h: u32) -> Vec<u8> {
        let y_size = (w * h) as usize;
        let uv_size = y_size / 4;
        // Y=128 (mid-gray), U=128, V=128 (neutral chroma)
        vec![128u8; y_size + uv_size * 2]
    }
}
