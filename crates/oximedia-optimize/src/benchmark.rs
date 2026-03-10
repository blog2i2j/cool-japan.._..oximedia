//! Benchmark and profiling utilities for optimization.

use crate::utils::Timer;
use std::collections::HashMap;
use std::time::Duration;

/// Benchmark configuration.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup runs.
    pub warmup_runs: usize,
    /// Number of benchmark runs.
    pub benchmark_runs: usize,
    /// Enable detailed profiling.
    pub enable_profiling: bool,
    /// Target FPS.
    pub target_fps: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_runs: 3,
            benchmark_runs: 10,
            enable_profiling: true,
            target_fps: 30.0,
        }
    }
}

/// Benchmark result for a single operation.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Operation name.
    pub name: String,
    /// Minimum time.
    pub min_time: Duration,
    /// Maximum time.
    pub max_time: Duration,
    /// Average time.
    pub avg_time: Duration,
    /// Median time.
    pub median_time: Duration,
    /// Standard deviation.
    pub std_dev: Duration,
    /// Number of iterations.
    pub iterations: usize,
}

impl BenchmarkResult {
    /// Creates a new benchmark result from timings.
    #[must_use]
    pub fn from_timings(name: String, timings: &[Duration]) -> Self {
        if timings.is_empty() {
            return Self {
                name,
                min_time: Duration::ZERO,
                max_time: Duration::ZERO,
                avg_time: Duration::ZERO,
                median_time: Duration::ZERO,
                std_dev: Duration::ZERO,
                iterations: 0,
            };
        }

        let mut sorted_timings = timings.to_vec();
        sorted_timings.sort();

        // sorted_timings is non-empty (checked above), so first/last always yield Some
        let min_time = sorted_timings[0];
        let max_time = sorted_timings[sorted_timings.len() - 1];
        let median_time = sorted_timings[sorted_timings.len() / 2];

        let total_nanos: u128 = timings.iter().map(std::time::Duration::as_nanos).sum();
        let avg_nanos = total_nanos / timings.len() as u128;
        let avg_time = Duration::from_nanos(avg_nanos as u64);

        // Calculate standard deviation
        let variance: f64 = timings
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - avg_time.as_secs_f64();
                diff * diff
            })
            .sum::<f64>()
            / timings.len() as f64;
        let std_dev = Duration::from_secs_f64(variance.sqrt());

        Self {
            name,
            min_time,
            max_time,
            avg_time,
            median_time,
            std_dev,
            iterations: timings.len(),
        }
    }

    /// Prints the result.
    pub fn print(&self) {
        println!("Benchmark: {}", self.name);
        println!("  Iterations: {}", self.iterations);
        println!("  Min:    {:?}", self.min_time);
        println!("  Max:    {:?}", self.max_time);
        println!("  Avg:    {:?}", self.avg_time);
        println!("  Median: {:?}", self.median_time);
        println!("  StdDev: {:?}", self.std_dev);
    }
}

/// Performance profiler.
#[derive(Debug, Default)]
pub struct Profiler {
    timings: HashMap<String, Vec<Duration>>,
    active_timers: HashMap<String, Timer>,
}

impl Profiler {
    /// Creates a new profiler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts profiling a section.
    pub fn start(&mut self, label: impl Into<String>) {
        let label = label.into();
        self.active_timers.insert(label.clone(), Timer::new(label));
    }

    /// Stops profiling a section.
    pub fn stop(&mut self, label: &str) {
        if let Some(timer) = self.active_timers.remove(label) {
            let duration = timer.elapsed();
            self.timings
                .entry(label.to_string())
                .or_default()
                .push(duration);
        }
    }

    /// Gets all benchmark results.
    #[must_use]
    pub fn results(&self) -> Vec<BenchmarkResult> {
        self.timings
            .iter()
            .map(|(name, timings)| BenchmarkResult::from_timings(name.clone(), timings))
            .collect()
    }

    /// Prints all results.
    pub fn print_results(&self) {
        println!("\n=== Profiling Results ===");
        for result in self.results() {
            result.print();
            println!();
        }
    }

    /// Clears all data.
    pub fn clear(&mut self) {
        self.timings.clear();
        self.active_timers.clear();
    }
}

/// Benchmark runner.
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    profiler: Profiler,
}

impl BenchmarkRunner {
    /// Creates a new benchmark runner.
    #[must_use]
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            profiler: Profiler::new(),
        }
    }

    /// Runs a benchmark.
    pub fn run<F>(&mut self, name: impl Into<String>, mut f: F) -> BenchmarkResult
    where
        F: FnMut(),
    {
        let name = name.into();

        // Warmup
        for _ in 0..self.config.warmup_runs {
            f();
        }

        // Benchmark
        let mut timings = Vec::new();
        for _ in 0..self.config.benchmark_runs {
            let timer = Timer::new(&name);
            f();
            timings.push(timer.elapsed());
        }

        BenchmarkResult::from_timings(name, &timings)
    }

    /// Runs a benchmark with profiling.
    pub fn run_with_profiling<F>(&mut self, name: impl Into<String>, mut f: F) -> BenchmarkResult
    where
        F: FnMut(&mut Profiler),
    {
        let name = name.into();

        // Warmup
        for _ in 0..self.config.warmup_runs {
            self.profiler.clear();
            f(&mut self.profiler);
        }

        // Benchmark
        self.profiler.clear();
        let mut timings = Vec::new();
        for _ in 0..self.config.benchmark_runs {
            let timer = Timer::new(&name);
            f(&mut self.profiler);
            timings.push(timer.elapsed());
        }

        BenchmarkResult::from_timings(name, &timings)
    }

    /// Gets profiler.
    #[must_use]
    pub fn profiler(&self) -> &Profiler {
        &self.profiler
    }
}

/// Comparison benchmark for different optimization levels.
pub struct ComparativeBenchmark {
    results: HashMap<String, BenchmarkResult>,
}

impl ComparativeBenchmark {
    /// Creates a new comparative benchmark.
    #[must_use]
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    /// Adds a result.
    pub fn add_result(&mut self, name: impl Into<String>, result: BenchmarkResult) {
        self.results.insert(name.into(), result);
    }

    /// Compares results and prints.
    pub fn print_comparison(&self) {
        println!("\n=== Comparative Benchmark Results ===");

        // Find baseline (usually "fast" or first entry)
        let baseline_name = self
            .results
            .keys()
            .find(|k| k.contains("fast") || k.contains("baseline"))
            .or_else(|| self.results.keys().next())
            .map(String::as_str);

        if let Some(baseline_name) = baseline_name {
            if let Some(baseline) = self.results.get(baseline_name) {
                println!("Baseline: {baseline_name}");
                println!("  Time: {:?}", baseline.avg_time);
                println!();

                for (name, result) in &self.results {
                    if name != baseline_name {
                        let speedup =
                            baseline.avg_time.as_secs_f64() / result.avg_time.as_secs_f64();
                        println!("{name}");
                        println!("  Time: {:?}", result.avg_time);
                        println!("  Speedup: {speedup:.2}x");
                        println!();
                    }
                }
            }
        }
    }
}

impl Default for ComparativeBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

/// Quality-speed tradeoff analyzer.
pub struct TradeoffAnalyzer {
    points: Vec<TradeoffPoint>,
}

#[derive(Debug, Clone)]
struct TradeoffPoint {
    label: String,
    encoding_time: Duration,
    quality: f64,
    bits: u64,
}

impl TradeoffAnalyzer {
    /// Creates a new tradeoff analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Adds a measurement point.
    pub fn add_point(
        &mut self,
        label: impl Into<String>,
        encoding_time: Duration,
        quality: f64,
        bits: u64,
    ) {
        self.points.push(TradeoffPoint {
            label: label.into(),
            encoding_time,
            quality,
            bits,
        });
    }

    /// Finds optimal configuration for target quality.
    #[must_use]
    pub fn find_optimal_for_quality(&self, target_quality: f64) -> Option<&str> {
        self.points
            .iter()
            .filter(|p| p.quality >= target_quality)
            .min_by(|a, b| a.encoding_time.cmp(&b.encoding_time))
            .map(|p| p.label.as_str())
    }

    /// Finds optimal configuration for target speed.
    #[must_use]
    pub fn find_optimal_for_speed(&self, max_time: Duration) -> Option<&str> {
        self.points
            .iter()
            .filter(|p| p.encoding_time <= max_time)
            .max_by(|a, b| {
                a.quality
                    .partial_cmp(&b.quality)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.label.as_str())
    }

    /// Calculates Pareto frontier.
    #[must_use]
    pub fn pareto_frontier(&self) -> Vec<&str> {
        let mut frontier = Vec::new();

        for point in &self.points {
            let mut dominated = false;

            for other in &self.points {
                if other.encoding_time <= point.encoding_time
                    && other.quality >= point.quality
                    && (other.encoding_time < point.encoding_time || other.quality > point.quality)
                {
                    dominated = true;
                    break;
                }
            }

            if !dominated {
                frontier.push(point.label.as_str());
            }
        }

        frontier
    }

    /// Prints analysis.
    pub fn print_analysis(&self) {
        println!("\n=== Quality-Speed Tradeoff Analysis ===");

        let frontier = self.pareto_frontier();
        println!("Pareto-optimal configurations:");
        for label in &frontier {
            if let Some(point) = self.points.iter().find(|p| &p.label == label) {
                println!(
                    "  {}: {:.2} dB, {:?}, {} bits",
                    label, point.quality, point.encoding_time, point.bits
                );
            }
        }
    }
}

impl Default for TradeoffAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_runs, 3);
        assert_eq!(config.benchmark_runs, 10);
    }

    #[test]
    fn test_benchmark_result_from_timings() {
        let timings = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(15),
        ];
        let result = BenchmarkResult::from_timings("test".to_string(), &timings);
        assert_eq!(result.iterations, 3);
        assert_eq!(result.min_time, Duration::from_millis(10));
        assert_eq!(result.max_time, Duration::from_millis(20));
    }

    #[test]
    fn test_profiler() {
        let mut profiler = Profiler::new();
        profiler.start("test1");
        std::thread::sleep(Duration::from_millis(10));
        profiler.stop("test1");

        let results = profiler.results();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "test1");
    }

    #[test]
    fn test_benchmark_runner() {
        let config = BenchmarkConfig {
            warmup_runs: 1,
            benchmark_runs: 2,
            enable_profiling: false,
            target_fps: 30.0,
        };
        let mut runner = BenchmarkRunner::new(config);

        let result = runner.run("test", || {
            std::thread::sleep(Duration::from_millis(1));
        });

        assert_eq!(result.iterations, 2);
        assert!(result.avg_time >= Duration::from_millis(1));
    }

    #[test]
    fn test_comparative_benchmark() {
        let mut comp = ComparativeBenchmark::new();

        let result1 =
            BenchmarkResult::from_timings("fast".to_string(), &[Duration::from_millis(10)]);
        let result2 =
            BenchmarkResult::from_timings("slow".to_string(), &[Duration::from_millis(20)]);

        comp.add_result("fast", result1);
        comp.add_result("slow", result2);

        assert_eq!(comp.results.len(), 2);
    }

    #[test]
    fn test_tradeoff_analyzer() {
        let mut analyzer = TradeoffAnalyzer::new();

        analyzer.add_point("fast", Duration::from_millis(10), 40.0, 1000);
        analyzer.add_point("medium", Duration::from_millis(20), 42.0, 900);
        analyzer.add_point("slow", Duration::from_millis(40), 44.0, 800);

        let optimal_quality = analyzer.find_optimal_for_quality(42.0);
        assert!(optimal_quality.is_some());

        let optimal_speed = analyzer.find_optimal_for_speed(Duration::from_millis(25));
        assert!(optimal_speed.is_some());
    }

    #[test]
    fn test_pareto_frontier() {
        let mut analyzer = TradeoffAnalyzer::new();

        analyzer.add_point("a", Duration::from_millis(10), 40.0, 1000);
        analyzer.add_point("b", Duration::from_millis(20), 42.0, 900);
        analyzer.add_point("c", Duration::from_millis(15), 41.0, 950); // Dominated
        analyzer.add_point("d", Duration::from_millis(30), 44.0, 850);

        let frontier = analyzer.pareto_frontier();
        assert!(frontier.contains(&"a"));
        assert!(frontier.contains(&"b"));
        assert!(frontier.contains(&"d"));
        // "c" should be dominated by "b" (better quality at similar or better time)
    }
}
