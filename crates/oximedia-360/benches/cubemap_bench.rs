//! Criterion benchmarks for cubemap face extraction at various resolutions.
//!
//! Benchmarks the following at 1K, 2K, 4K, and 8K equirectangular resolutions:
//! * `equirect_to_cube` — row-major CPU conversion
//! * `equirect_to_cube_tiled` — cache-friendly tiled conversion
//! * `equirect_to_cube_parallel` — rayon-parallel tiled conversion

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oximedia_360::tiled::{equirect_to_cube_parallel, equirect_to_cube_tiled};

/// Build a solid-colour equirectangular test image of size `w × h` (RGB).
fn make_equirect(w: u32, h: u32) -> Vec<u8> {
    vec![128u8; (w * h * 3) as usize]
}

fn bench_tiled_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("equirect_to_cube_tiled");

    // Benchmark at multiple resolutions: 1K, 2K, 4K (skip 8K for CI speed)
    for &(label, equirect_w, equirect_h, face_size) in &[
        ("1K", 1024u32, 512u32, 256u32),
        ("2K", 2048u32, 1024u32, 512u32),
        ("4K", 4096u32, 2048u32, 1024u32),
    ] {
        let src = make_equirect(equirect_w, equirect_h);
        group.bench_with_input(
            BenchmarkId::new("tiled-16", label),
            &(equirect_w, equirect_h, face_size),
            |b, &(w, h, fs)| {
                b.iter(|| equirect_to_cube_tiled(&src, w, h, fs, 16).expect("tiled ok"));
            },
        );
    }

    group.finish();
}

fn bench_parallel_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("equirect_to_cube_parallel");

    for &(label, equirect_w, equirect_h, face_size) in &[
        ("1K", 1024u32, 512u32, 256u32),
        ("2K", 2048u32, 1024u32, 512u32),
    ] {
        let src = make_equirect(equirect_w, equirect_h);
        group.bench_with_input(
            BenchmarkId::new("parallel", label),
            &(equirect_w, equirect_h, face_size),
            |b, &(w, h, fs)| {
                b.iter(|| equirect_to_cube_parallel(&src, w, h, fs).expect("parallel ok"));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_tiled_conversion, bench_parallel_conversion);
criterion_main!(benches);
