/// Benchmarking utilities for denoise performance.
use super::{DenoiseConfig, DenoiseMethod};
use std::time::Duration;

/// Benchmark result for a denoise operation.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Method being benchmarked.
    pub method: DenoiseMethod,
    /// Frame resolution.
    pub resolution: (u32, u32),
    /// Processing time.
    pub duration: Duration,
    /// Frames per second.
    pub fps: f32,
    /// Megapixels per second.
    pub mpixels_per_sec: f32,
}

impl BenchmarkResult {
    /// Create a benchmark result.
    #[must_use]
    pub fn new(method: DenoiseMethod, width: u32, height: u32, duration: Duration) -> Self {
        let pixels = (width * height) as f32;
        let seconds = duration.as_secs_f32();
        let fps = if seconds > 0.0 { 1.0 / seconds } else { 0.0 };
        let mpixels_per_sec = if seconds > 0.0 {
            pixels / (seconds * 1_000_000.0)
        } else {
            0.0
        };

        Self {
            method,
            resolution: (width, height),
            duration,
            fps,
            mpixels_per_sec,
        }
    }

    /// Format result as a string.
    #[must_use]
    pub fn format(&self) -> String {
        format!(
            "{:?} @ {}x{}: {:.2}ms ({:.2} fps, {:.2} MP/s)",
            self.method,
            self.resolution.0,
            self.resolution.1,
            self.duration.as_secs_f64() * 1000.0,
            self.fps,
            self.mpixels_per_sec
        )
    }
}

/// Benchmark multiple denoise methods.
#[must_use]
pub fn compare_methods(width: u32, height: u32) -> Vec<BenchmarkResult> {
    let methods = [
        DenoiseMethod::Gaussian,
        DenoiseMethod::Bilateral,
        DenoiseMethod::Median,
        DenoiseMethod::NonLocalMeans,
        DenoiseMethod::Adaptive,
    ];

    methods
        .iter()
        .map(|&method| {
            let config = DenoiseConfig::new().with_method(method);
            estimate_performance(&config, width, height)
        })
        .collect()
}

/// Estimate performance for a config.
#[must_use]
pub fn estimate_performance(config: &DenoiseConfig, width: u32, height: u32) -> BenchmarkResult {
    let pixels = (width * height) as f32;

    let base_time_per_pixel = match config.method {
        DenoiseMethod::Gaussian => 0.5,
        DenoiseMethod::Bilateral => 2.0,
        DenoiseMethod::Median => 3.0,
        DenoiseMethod::NonLocalMeans => 10.0,
        DenoiseMethod::Adaptive => 4.0,
        DenoiseMethod::Temporal => 1.5,
        DenoiseMethod::MotionCompensated => 8.0,
        DenoiseMethod::BlockMatching3D => 15.0,
        DenoiseMethod::Combined => 5.0,
    };

    let radius_factor = (config.spatial_radius as f32 / 5.0).powi(2);
    let time_ns = pixels * base_time_per_pixel * radius_factor;
    let duration = Duration::from_nanos(time_ns as u64);

    BenchmarkResult::new(config.method, width, height, duration)
}
