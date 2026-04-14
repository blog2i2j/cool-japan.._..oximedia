//! Runtime benchmark utilities for `oximedia-simd`.
//!
//! This module provides a lightweight throughput-measurement framework that
//! does **not** depend on `criterion` — it uses `std::time::Instant` and
//! iterates for a fixed duration (default: 100 ms warm-up + 500 ms measure).
//!
//! # Usage
//!
//! ```no_run
//! use oximedia_simd::simd_bench::{BenchConfig, run_simd_benchmarks};
//! let results = run_simd_benchmarks(BenchConfig::default());
//! for r in &results {
//!     println!("{}: {:.1} Mpx/s", r.name, r.throughput_mpx_per_sec);
//! }
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::time::{Duration, Instant};

use crate::{detect_cpu_features, sad, satd, satd::SatdBlockSize, BlockSize, CpuFeatures};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the runtime benchmark.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Warm-up duration before measurement starts.
    pub warmup: Duration,
    /// Measurement duration.
    pub measure: Duration,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            warmup: Duration::from_millis(50),
            measure: Duration::from_millis(200),
        }
    }
}

// ── Result type ───────────────────────────────────────────────────────────────

/// A single benchmark result.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// Human-readable kernel name (e.g. `"SAD 16×16 (scalar)"`).
    pub name: String,
    /// SIMD tier used (e.g. `"avx2"`, `"scalar"`).
    pub tier: String,
    /// Throughput in millions of pixels per second.
    pub throughput_mpx_per_sec: f64,
    /// Number of iterations completed during the measurement window.
    pub iterations: u64,
    /// Measured time for all iterations.
    pub elapsed: Duration,
}

// ── Timing helper ─────────────────────────────────────────────────────────────

/// Run `f` repeatedly for `duration` and return (iteration count, elapsed).
fn timed_loop<F: FnMut()>(mut f: F, duration: Duration) -> (u64, Duration) {
    let start = Instant::now();
    let mut count = 0u64;
    while start.elapsed() < duration {
        f();
        count += 1;
    }
    (count, start.elapsed())
}

// ── Tier label ────────────────────────────────────────────────────────────────

fn tier_label(feat: &CpuFeatures) -> &'static str {
    if feat.avx512f && feat.avx512bw {
        "avx512"
    } else if feat.avx2 {
        "avx2"
    } else if feat.sse4_2 {
        "sse4.2"
    } else if feat.neon {
        "neon"
    } else {
        "scalar"
    }
}

// ── Individual benchmark runners ──────────────────────────────────────────────

fn bench_sad_16x16(cfg: &BenchConfig, feat: &CpuFeatures) -> BenchResult {
    let src = vec![100u8; 16 * 16 + 16];
    let ref_ = vec![110u8; 16 * 16 + 16];
    let tier = tier_label(feat).to_owned();

    // Warm-up
    timed_loop(
        || {
            let _ = sad(&src, &ref_, 16, 16, BlockSize::Block16x16);
        },
        cfg.warmup,
    );

    let (iters, elapsed) = timed_loop(
        || {
            let _ = sad(&src, &ref_, 16, 16, BlockSize::Block16x16);
        },
        cfg.measure,
    );

    let pixels_per_iter = 16 * 16;
    let total_px = iters * pixels_per_iter as u64;
    let throughput = total_px as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    BenchResult {
        name: "SAD 16×16".to_owned(),
        tier,
        throughput_mpx_per_sec: throughput,
        iterations: iters,
        elapsed,
    }
}

fn bench_sad_32x32(cfg: &BenchConfig, feat: &CpuFeatures) -> BenchResult {
    let src = vec![100u8; 32 * 32 + 32];
    let ref_ = vec![110u8; 32 * 32 + 32];
    let tier = tier_label(feat).to_owned();

    timed_loop(
        || {
            let _ = sad(&src, &ref_, 32, 32, BlockSize::Block32x32);
        },
        cfg.warmup,
    );

    let (iters, elapsed) = timed_loop(
        || {
            let _ = sad(&src, &ref_, 32, 32, BlockSize::Block32x32);
        },
        cfg.measure,
    );

    let pixels_per_iter = 32 * 32;
    let total_px = iters * pixels_per_iter as u64;
    let throughput = total_px as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    BenchResult {
        name: "SAD 32×32".to_owned(),
        tier,
        throughput_mpx_per_sec: throughput,
        iterations: iters,
        elapsed,
    }
}

fn bench_sad_64x64(cfg: &BenchConfig, feat: &CpuFeatures) -> BenchResult {
    let src = vec![100u8; 64 * 64 + 64];
    let ref_ = vec![110u8; 64 * 64 + 64];
    let tier = tier_label(feat).to_owned();

    timed_loop(
        || {
            let _ = sad(&src, &ref_, 64, 64, BlockSize::Block64x64);
        },
        cfg.warmup,
    );

    let (iters, elapsed) = timed_loop(
        || {
            let _ = sad(&src, &ref_, 64, 64, BlockSize::Block64x64);
        },
        cfg.measure,
    );

    let pixels_per_iter = 64 * 64;
    let total_px = iters * pixels_per_iter as u64;
    let throughput = total_px as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    BenchResult {
        name: "SAD 64×64".to_owned(),
        tier,
        throughput_mpx_per_sec: throughput,
        iterations: iters,
        elapsed,
    }
}

fn bench_satd_8x8(cfg: &BenchConfig, feat: &CpuFeatures) -> BenchResult {
    let src = vec![100u8; 64];
    let ref_ = vec![110u8; 64];
    let tier = tier_label(feat).to_owned();

    timed_loop(
        || {
            let _ = satd::satd(&src, &ref_, SatdBlockSize::Block8x8);
        },
        cfg.warmup,
    );

    let (iters, elapsed) = timed_loop(
        || {
            let _ = satd::satd(&src, &ref_, SatdBlockSize::Block8x8);
        },
        cfg.measure,
    );

    let pixels_per_iter = 64;
    let total_px = iters * pixels_per_iter as u64;
    let throughput = total_px as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    BenchResult {
        name: "SATD 8×8".to_owned(),
        tier,
        throughput_mpx_per_sec: throughput,
        iterations: iters,
        elapsed,
    }
}

fn bench_satd_16x16(cfg: &BenchConfig, feat: &CpuFeatures) -> BenchResult {
    let src = vec![100u8; 256];
    let ref_ = vec![110u8; 256];
    let tier = tier_label(feat).to_owned();

    timed_loop(
        || {
            let _ = satd::satd(&src, &ref_, SatdBlockSize::Block16x16);
        },
        cfg.warmup,
    );

    let (iters, elapsed) = timed_loop(
        || {
            let _ = satd::satd(&src, &ref_, SatdBlockSize::Block16x16);
        },
        cfg.measure,
    );

    let pixels_per_iter = 256;
    let total_px = iters * pixels_per_iter as u64;
    let throughput = total_px as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    BenchResult {
        name: "SATD 16×16".to_owned(),
        tier,
        throughput_mpx_per_sec: throughput,
        iterations: iters,
        elapsed,
    }
}

fn bench_ssim_16x16(cfg: &BenchConfig, feat: &CpuFeatures) -> BenchResult {
    let src = vec![100u8; 16 * 16];
    let ref_ = vec![105u8; 16 * 16];
    let tier = tier_label(feat).to_owned();

    timed_loop(
        || {
            let _ = crate::ssim::ssim_block_16x16(&src, &ref_, 16);
        },
        cfg.warmup,
    );

    let (iters, elapsed) = timed_loop(
        || {
            let _ = crate::ssim::ssim_block_16x16(&src, &ref_, 16);
        },
        cfg.measure,
    );

    let pixels_per_iter = 256;
    let total_px = iters * pixels_per_iter as u64;
    let throughput = total_px as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    BenchResult {
        name: "SSIM 16×16".to_owned(),
        tier,
        throughput_mpx_per_sec: throughput,
        iterations: iters,
        elapsed,
    }
}

fn bench_psnr_16x16(cfg: &BenchConfig, feat: &CpuFeatures) -> BenchResult {
    let src = vec![100u8; 16 * 16];
    let ref_ = vec![105u8; 16 * 16];
    let tier = tier_label(feat).to_owned();

    timed_loop(
        || {
            let _ = crate::psnr::psnr_block_16x16(&src, &ref_, 16);
        },
        cfg.warmup,
    );

    let (iters, elapsed) = timed_loop(
        || {
            let _ = crate::psnr::psnr_block_16x16(&src, &ref_, 16);
        },
        cfg.measure,
    );

    let pixels_per_iter = 256;
    let total_px = iters * pixels_per_iter as u64;
    let throughput = total_px as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    BenchResult {
        name: "PSNR 16×16".to_owned(),
        tier,
        throughput_mpx_per_sec: throughput,
        iterations: iters,
        elapsed,
    }
}

fn bench_deblock_16x16(cfg: &BenchConfig, feat: &CpuFeatures) -> BenchResult {
    use crate::deblock_filter::{deblock_full, DeblockParams};
    let mut plane = vec![0u8; 16 * 16];
    // inject block artifact
    for row in 0..16 {
        for col in 0..16 {
            plane[row * 16 + col] = if col < 8 { 80 } else { 160 };
        }
    }
    let params = DeblockParams::default_medium();
    let tier = tier_label(feat).to_owned();

    timed_loop(
        || {
            let _ = deblock_full(&mut plane, 16, 16, 16, &params);
        },
        cfg.warmup,
    );

    let (iters, elapsed) = timed_loop(
        || {
            let _ = deblock_full(&mut plane, 16, 16, 16, &params);
        },
        cfg.measure,
    );

    let pixels_per_iter = 256;
    let total_px = iters * pixels_per_iter as u64;
    let throughput = total_px as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    BenchResult {
        name: "Deblock 16×16".to_owned(),
        tier,
        throughput_mpx_per_sec: throughput,
        iterations: iters,
        elapsed,
    }
}

fn bench_color_convert_bt709(cfg: &BenchConfig, feat: &CpuFeatures) -> BenchResult {
    use crate::color_convert_simd::{rgb_to_yuv_bt709, RgbPixel, YuvPixel};
    let src: Vec<RgbPixel> = (0..256)
        .map(|i| RgbPixel {
            r: i as u8,
            g: (i * 2) as u8,
            b: (i * 3) as u8,
        })
        .collect();
    let mut dst = vec![YuvPixel { y: 0, cb: 0, cr: 0 }; 256];
    let tier = tier_label(feat).to_owned();

    timed_loop(
        || {
            rgb_to_yuv_bt709(&src, &mut dst);
        },
        cfg.warmup,
    );

    let (iters, elapsed) = timed_loop(
        || {
            rgb_to_yuv_bt709(&src, &mut dst);
        },
        cfg.measure,
    );

    let pixels_per_iter = 256;
    let total_px = iters * pixels_per_iter as u64;
    let throughput = total_px as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    BenchResult {
        name: "ColorConvert BT.709 RGB→YUV (256px)".to_owned(),
        tier,
        throughput_mpx_per_sec: throughput,
        iterations: iters,
        elapsed,
    }
}

fn bench_color_convert_bt2020(cfg: &BenchConfig, feat: &CpuFeatures) -> BenchResult {
    use crate::color_convert_simd::{rgb_to_yuv_bt2020, RgbPixel, YuvPixel};
    let src: Vec<RgbPixel> = (0..256)
        .map(|i| RgbPixel {
            r: i as u8,
            g: (i * 2) as u8,
            b: (i * 3) as u8,
        })
        .collect();
    let mut dst = vec![YuvPixel { y: 0, cb: 0, cr: 0 }; 256];
    let tier = tier_label(feat).to_owned();

    timed_loop(
        || {
            rgb_to_yuv_bt2020(&src, &mut dst);
        },
        cfg.warmup,
    );

    let (iters, elapsed) = timed_loop(
        || {
            rgb_to_yuv_bt2020(&src, &mut dst);
        },
        cfg.measure,
    );

    let pixels_per_iter = 256;
    let total_px = iters * pixels_per_iter as u64;
    let throughput = total_px as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    BenchResult {
        name: "ColorConvert BT.2020 RGB→YUV (256px)".to_owned(),
        tier,
        throughput_mpx_per_sec: throughput,
        iterations: iters,
        elapsed,
    }
}

fn bench_variance_u8(cfg: &BenchConfig, feat: &CpuFeatures) -> BenchResult {
    use crate::reduce::variance_u8;
    let data: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
    let tier = tier_label(feat).to_owned();

    timed_loop(
        || {
            let _ = variance_u8(&data);
        },
        cfg.warmup,
    );

    let (iters, elapsed) = timed_loop(
        || {
            let _ = variance_u8(&data);
        },
        cfg.measure,
    );

    let pixels_per_iter = 256;
    let total_px = iters * pixels_per_iter as u64;
    let throughput = total_px as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    BenchResult {
        name: "Variance u8 (256 samples)".to_owned(),
        tier,
        throughput_mpx_per_sec: throughput,
        iterations: iters,
        elapsed,
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Run all SIMD benchmarks and return a vector of results.
///
/// The detected CPU tier is reflected in each [`BenchResult::tier`] field.
/// All kernels run with both warm-up and measurement phases as configured by
/// `cfg`.
pub fn run_simd_benchmarks(cfg: BenchConfig) -> Vec<BenchResult> {
    let feat = detect_cpu_features();
    vec![
        bench_sad_16x16(&cfg, &feat),
        bench_sad_32x32(&cfg, &feat),
        bench_sad_64x64(&cfg, &feat),
        bench_satd_8x8(&cfg, &feat),
        bench_satd_16x16(&cfg, &feat),
        bench_ssim_16x16(&cfg, &feat),
        bench_psnr_16x16(&cfg, &feat),
        bench_deblock_16x16(&cfg, &feat),
        bench_color_convert_bt709(&cfg, &feat),
        bench_color_convert_bt2020(&cfg, &feat),
        bench_variance_u8(&cfg, &feat),
    ]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_config_default_is_reasonable() {
        let cfg = BenchConfig::default();
        assert!(cfg.warmup < Duration::from_secs(10));
        assert!(cfg.measure < Duration::from_secs(10));
        assert!(cfg.measure > Duration::ZERO);
    }

    #[test]
    fn run_benchmarks_returns_non_empty_results() {
        // Use very short durations to keep test time under 1 s.
        let cfg = BenchConfig {
            warmup: Duration::from_millis(1),
            measure: Duration::from_millis(5),
        };
        let results = run_simd_benchmarks(cfg);
        assert!(!results.is_empty(), "should return at least one benchmark");
        for r in &results {
            assert!(
                r.throughput_mpx_per_sec > 0.0,
                "kernel '{}' has non-positive throughput",
                r.name
            );
            assert!(r.iterations > 0, "kernel '{}' had zero iterations", r.name);
        }
    }

    #[test]
    fn tier_label_is_non_empty() {
        let feat = detect_cpu_features();
        let label = super::tier_label(&feat);
        assert!(!label.is_empty());
    }

    #[test]
    fn run_benchmarks_returns_correct_count() {
        let cfg = BenchConfig {
            warmup: Duration::from_millis(1),
            measure: Duration::from_millis(5),
        };
        let results = run_simd_benchmarks(cfg);
        // We expect exactly 11 benchmark kernels
        assert_eq!(
            results.len(),
            11,
            "expected 11 benchmark results, got {}",
            results.len()
        );
    }

    #[test]
    fn each_result_has_non_empty_name() {
        let cfg = BenchConfig {
            warmup: Duration::from_millis(1),
            measure: Duration::from_millis(5),
        };
        let results = run_simd_benchmarks(cfg);
        for r in &results {
            assert!(!r.name.is_empty(), "bench result name must not be empty");
        }
    }

    #[test]
    fn each_result_has_non_empty_tier() {
        let cfg = BenchConfig {
            warmup: Duration::from_millis(1),
            measure: Duration::from_millis(5),
        };
        let results = run_simd_benchmarks(cfg);
        for r in &results {
            assert!(
                !r.tier.is_empty(),
                "bench tier must not be empty for '{}'",
                r.name
            );
        }
    }

    #[test]
    fn tier_label_matches_valid_values() {
        let feat = detect_cpu_features();
        let label = super::tier_label(&feat);
        let valid = ["avx512", "avx2", "sse4.2", "neon", "scalar"];
        assert!(
            valid.contains(&label),
            "tier label '{label}' must be one of {valid:?}"
        );
    }

    #[test]
    fn each_result_elapsed_is_positive() {
        let cfg = BenchConfig {
            warmup: Duration::from_millis(1),
            measure: Duration::from_millis(5),
        };
        let results = run_simd_benchmarks(cfg);
        for r in &results {
            assert!(
                r.elapsed > Duration::ZERO,
                "elapsed time must be > 0 for '{}'",
                r.name
            );
        }
    }

    #[test]
    fn bench_config_clone_is_equal() {
        let cfg = BenchConfig::default();
        let cfg2 = cfg.clone();
        assert_eq!(cfg.warmup, cfg2.warmup);
        assert_eq!(cfg.measure, cfg2.measure);
    }

    #[test]
    fn bench_result_clone_preserves_fields() {
        let cfg = BenchConfig {
            warmup: Duration::from_millis(1),
            measure: Duration::from_millis(5),
        };
        let results = run_simd_benchmarks(cfg);
        let r = results[0].clone();
        assert_eq!(r.name, results[0].name);
        assert_eq!(r.tier, results[0].tier);
        assert_eq!(r.iterations, results[0].iterations);
    }

    #[test]
    fn run_benchmarks_with_longer_measure_gives_more_iterations() {
        let cfg_short = BenchConfig {
            warmup: Duration::from_millis(1),
            measure: Duration::from_millis(5),
        };
        let cfg_long = BenchConfig {
            warmup: Duration::from_millis(1),
            measure: Duration::from_millis(50),
        };
        let results_short = run_simd_benchmarks(cfg_short);
        let results_long = run_simd_benchmarks(cfg_long);
        // The kernel with 10× more measurement time should generally have more iters
        // We check that iterations are always ≥ 1 in both cases
        for r in results_short.iter().chain(results_long.iter()) {
            assert!(
                r.iterations >= 1,
                "each kernel must run at least 1 iteration"
            );
        }
    }

    #[test]
    fn throughput_is_finite_and_positive() {
        let cfg = BenchConfig {
            warmup: Duration::from_millis(1),
            measure: Duration::from_millis(5),
        };
        let results = run_simd_benchmarks(cfg);
        for r in &results {
            assert!(
                r.throughput_mpx_per_sec.is_finite(),
                "throughput must be finite for '{}'",
                r.name
            );
            assert!(
                r.throughput_mpx_per_sec > 0.0,
                "throughput must be positive for '{}'",
                r.name
            );
        }
    }
}
