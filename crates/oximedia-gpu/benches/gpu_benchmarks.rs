//! Performance regression benchmarks for oximedia-gpu core operations.
//!
//! These benchmarks use CPU-path functions only so they can run without a
//! GPU device.  They serve as regression guards: if a refactor accidentally
//! makes a hot path significantly slower, criterion will flag it.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oximedia_gpu::ops::colorspace::{bt601_rgb_to_ycbcr, bt601_ycbcr_to_rgb};
use oximedia_gpu::ops::filter::gaussian_blur_separable;
use oximedia_gpu::ops::scale::ScaleOperation;
use std::hint::black_box;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Generate a repeating RGBA test frame of the given dimensions.
fn make_rgba_frame(width: usize, height: usize) -> Vec<u8> {
    let len = width * height * 4;
    let mut buf = Vec::with_capacity(len);
    for i in 0..len {
        buf.push((i % 256) as u8);
    }
    buf
}

// ── BT.601 YUV ↔ RGB (CPU path) ─────────────────────────────────────────────

/// Benchmark BT.601 RGB→YCbCr conversion for a full 1920×1080 frame.
///
/// This exercises the pure-Rust scalar pixel-by-pixel path used as the CPU
/// reference implementation.
fn bench_color_convert(c: &mut Criterion) {
    let width = 1920usize;
    let height = 1080usize;
    let frame = make_rgba_frame(width, height);

    let mut group = c.benchmark_group("color_convert");

    group.bench_function(BenchmarkId::new("bt601_rgb_to_yuv", "1920x1080"), |b| {
        b.iter(|| {
            let pixels = black_box(&frame);
            let mut output = vec![(0u8, 0u8, 0u8); width * height];
            for (i, chunk) in pixels.chunks_exact(4).enumerate() {
                output[i] = bt601_rgb_to_ycbcr(chunk[0], chunk[1], chunk[2]);
            }
            output
        });
    });

    group.bench_function(BenchmarkId::new("bt601_yuv_to_rgb", "1920x1080"), |b| {
        // Prepare a YUV input (just reuse the RGBA buffer as synthetic YUV data).
        let yuv = make_rgba_frame(width, height);
        b.iter(|| {
            let pixels = black_box(&yuv);
            let mut output = vec![(0u8, 0u8, 0u8); width * height];
            for (i, chunk) in pixels.chunks_exact(4).enumerate() {
                output[i] = bt601_ycbcr_to_rgb(chunk[0], chunk[1], chunk[2]);
            }
            output
        });
    });

    group.finish();
}

// ── Bilinear downscale (CPU path via Lanczos) ────────────────────────────────

/// Benchmark bilinear downscale 1920×1080 → 960×540 using the CPU Lanczos-3
/// path exposed through `ScaleOperation::lanczos3_cpu`.
fn bench_scale(c: &mut Criterion) {
    let src_w = 1920u32;
    let src_h = 1080u32;
    let dst_w = 960u32;
    let dst_h = 540u32;

    let input = make_rgba_frame(src_w as usize, src_h as usize);
    let mut output = vec![0u8; dst_w as usize * dst_h as usize * 4];

    let mut group = c.benchmark_group("scale");

    group.bench_function(
        BenchmarkId::new("lanczos3_downscale_cpu", "1920x1080_to_960x540"),
        |b| {
            b.iter(|| {
                ScaleOperation::lanczos3_cpu(
                    black_box(&input),
                    src_w,
                    src_h,
                    black_box(&mut output),
                    dst_w,
                    dst_h,
                )
                .expect("lanczos3_cpu benchmark");
            });
        },
    );

    group.finish();
}

// ── Gaussian blur (CPU path) ─────────────────────────────────────────────────

/// Benchmark a 9×9 Gaussian blur (sigma ≈ 2.0) on a 512×512 frame.
///
/// Uses the pure-Rust separable blur that is also the CPU fallback in the GPU
/// path.
fn bench_blur(c: &mut Criterion) {
    let width = 512u32;
    let height = 512u32;
    let sigma = 2.0f32; // kernel size ≈ 9

    let input = make_rgba_frame(width as usize, height as usize);
    let mut output = vec![0u8; input.len()];

    let mut group = c.benchmark_group("blur");

    group.bench_function(
        BenchmarkId::new("gaussian_blur_separable_cpu", "512x512_sigma2"),
        |b| {
            b.iter(|| {
                gaussian_blur_separable(
                    black_box(&input),
                    black_box(&mut output),
                    width,
                    height,
                    sigma,
                )
                .expect("gaussian_blur_separable benchmark");
            });
        },
    );

    group.finish();
}

// ── Criterion entry point ────────────────────────────────────────────────────

criterion_group!(benches, bench_color_convert, bench_scale, bench_blur);
criterion_main!(benches);
