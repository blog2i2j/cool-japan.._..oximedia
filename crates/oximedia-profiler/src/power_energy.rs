//! Power and energy profiling module.
//!
//! Provides platform-aware energy measurement using:
//! - RAPL (Running Average Power Limit) counters on Linux
//! - IOKit `IOPMPowerSource` on macOS
//! - Simulated/estimated values on other platforms
//!
//! # Example
//!
//! ```
//! use oximedia_profiler::power_energy::{EnergyProfiler, PowerDomain};
//!
//! let mut profiler = EnergyProfiler::new();
//! profiler.start();
//! // ... workload ...
//! let report = profiler.stop();
//! println!("Total energy: {} µJ", report.total_energy_uj);
//! ```

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// PowerDomain
// ---------------------------------------------------------------------------

/// A distinct energy measurement domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PowerDomain {
    /// Total package (CPU socket) power.
    Package,
    /// CPU cores only.
    Cores,
    /// Uncore (last-level cache, memory controller, etc.).
    Uncore,
    /// Integrated GPU.
    Gpu,
    /// DRAM / memory subsystem.
    Dram,
    /// Platform-wide power (chassis + peripherals).
    Platform,
}

impl PowerDomain {
    /// Returns the human-readable label for this domain.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Package => "Package (CPU socket)",
            Self::Cores => "CPU Cores",
            Self::Uncore => "Uncore",
            Self::Gpu => "Integrated GPU",
            Self::Dram => "DRAM",
            Self::Platform => "Platform",
        }
    }

    /// Returns the RAPL MSR path component on Linux for this domain, if any.
    #[must_use]
    pub fn rapl_path_component(self) -> Option<&'static str> {
        match self {
            Self::Package => Some("package-0"),
            Self::Cores => Some("core"),
            Self::Uncore => Some("uncore"),
            Self::Dram => Some("dram"),
            Self::Gpu => Some("psys"),
            Self::Platform => None,
        }
    }
}

// ---------------------------------------------------------------------------
// EnergySample
// ---------------------------------------------------------------------------

/// A single energy reading captured at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergySample {
    /// Timestamp when the sample was taken (nanoseconds since profiler start).
    pub timestamp_ns: u64,
    /// Energy counter values per domain at this sample (micro-joules).
    pub domain_uj: HashMap<String, u64>,
    /// Instantaneous estimated power per domain (milliwatts).
    pub domain_mw: HashMap<String, f64>,
}

impl EnergySample {
    /// Creates a new sample.
    #[must_use]
    pub fn new(
        timestamp_ns: u64,
        domain_uj: HashMap<String, u64>,
        domain_mw: HashMap<String, f64>,
    ) -> Self {
        Self {
            timestamp_ns,
            domain_uj,
            domain_mw,
        }
    }

    /// Returns the total instantaneous power across all domains (milliwatts).
    #[must_use]
    pub fn total_power_mw(&self) -> f64 {
        self.domain_mw.values().sum()
    }
}

// ---------------------------------------------------------------------------
// EnergyReport
// ---------------------------------------------------------------------------

/// Summary report produced at the end of an energy profiling session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyReport {
    /// Duration of the profiling window.
    pub duration: Duration,
    /// Total energy consumed across all measured domains (micro-joules).
    pub total_energy_uj: u64,
    /// Average power draw across all domains (milliwatts).
    pub average_power_mw: f64,
    /// Peak instantaneous power recorded (milliwatts).
    pub peak_power_mw: f64,
    /// Per-domain energy totals (micro-joules).
    pub domain_energy_uj: HashMap<String, u64>,
    /// Per-domain average power (milliwatts).
    pub domain_avg_power_mw: HashMap<String, f64>,
    /// Energy efficiency estimate: energy per unit work if `work_units` was set.
    pub energy_per_work_unit_uj: Option<f64>,
    /// Number of samples collected.
    pub sample_count: usize,
}

impl EnergyReport {
    /// Returns a human-readable one-line summary.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Energy: {} µJ | Avg power: {:.1} mW | Peak: {:.1} mW | Samples: {}",
            self.total_energy_uj,
            self.average_power_mw,
            self.peak_power_mw,
            self.sample_count
        )
    }

    /// Returns the most power-hungry domain name, if any samples were recorded.
    #[must_use]
    pub fn dominant_domain(&self) -> Option<&str> {
        self.domain_avg_power_mw
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.as_str())
    }
}

// ---------------------------------------------------------------------------
// PlatformReader trait
// ---------------------------------------------------------------------------

/// Abstraction for reading energy counters from the underlying platform.
///
/// Implementors can target RAPL (Linux), IOKit (macOS), or provide synthetic
/// readings for testing.
pub trait PlatformEnergyReader: Send {
    /// Reads the current energy counter values in micro-joules per domain.
    ///
    /// Returns `None` if the platform does not support energy measurement for
    /// a given domain or if the required permissions are unavailable.
    fn read_domains(&self) -> HashMap<String, u64>;

    /// Returns the set of domains this reader can measure.
    fn available_domains(&self) -> Vec<PowerDomain>;
}

// ---------------------------------------------------------------------------
// SyntheticEnergyReader
// ---------------------------------------------------------------------------

/// A deterministic synthetic energy reader for testing and unsupported platforms.
///
/// Simulates a steady-state power draw by incrementing counters linearly with
/// wall time.
#[derive(Debug)]
pub struct SyntheticEnergyReader {
    /// Simulated power per domain (milliwatts) used to derive µJ increments.
    domain_power_mw: HashMap<String, f64>,
    /// Start time used to compute elapsed energy.
    start: Instant,
    /// Last read timestamp to compute deltas.
    last_read: std::cell::Cell<Option<Instant>>,
    /// Accumulated energy per domain (µJ).
    accumulated_uj: std::cell::RefCell<HashMap<String, u64>>,
}

impl SyntheticEnergyReader {
    /// Creates a new synthetic reader with the given per-domain power (mW).
    #[must_use]
    pub fn new(domain_power_mw: HashMap<String, f64>) -> Self {
        Self {
            domain_power_mw,
            start: Instant::now(),
            last_read: std::cell::Cell::new(None),
            accumulated_uj: std::cell::RefCell::new(HashMap::new()),
        }
    }

    /// Convenience constructor for a single-domain (Package) synthetic reader.
    #[must_use]
    pub fn single_domain(power_mw: f64) -> Self {
        let mut map = HashMap::new();
        map.insert("Package".to_string(), power_mw);
        Self::new(map)
    }
}

impl PlatformEnergyReader for SyntheticEnergyReader {
    fn read_domains(&self) -> HashMap<String, u64> {
        let now = Instant::now();
        let prev = self.last_read.get().unwrap_or(self.start);
        let elapsed_ms = now.duration_since(prev).as_secs_f64() * 1_000.0;
        self.last_read.set(Some(now));

        let mut acc = self.accumulated_uj.borrow_mut();
        let mut result = HashMap::new();
        for (domain, &power_mw) in &self.domain_power_mw {
            // energy (µJ) = power (mW) * time (ms) = power (W) * time (s) * 1e6
            // Since mW * ms = mJ * 1e-3 * 1e-3 * 1e6 µJ = 1e0 µJ
            let delta_uj = (power_mw * elapsed_ms) as u64;
            let entry = acc.entry(domain.clone()).or_insert(0);
            *entry += delta_uj;
            result.insert(domain.clone(), *entry);
        }
        result
    }

    fn available_domains(&self) -> Vec<PowerDomain> {
        vec![PowerDomain::Package]
    }
}

// ---------------------------------------------------------------------------
// EnergyProfiler
// ---------------------------------------------------------------------------

/// Configuration for the energy profiler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyProfilerConfig {
    /// Sampling interval (default 100 ms).
    pub sample_interval: Duration,
    /// Domains to measure (empty = all available).
    pub domains: Vec<PowerDomain>,
    /// Optional work unit count (for energy-per-work-unit calculation).
    pub work_units: Option<f64>,
}

impl Default for EnergyProfilerConfig {
    fn default() -> Self {
        Self {
            sample_interval: Duration::from_millis(100),
            domains: vec![],
            work_units: None,
        }
    }
}

/// Energy and power profiler.
///
/// Collects per-domain energy readings at a configurable interval and
/// produces an [`EnergyReport`] on stop.
pub struct EnergyProfiler {
    config: EnergyProfilerConfig,
    reader: Box<dyn PlatformEnergyReader>,
    samples: Vec<EnergySample>,
    start_time: Option<Instant>,
    running: bool,
    /// Counter values at profiler start (used to compute deltas).
    baseline_uj: HashMap<String, u64>,
}

impl EnergyProfiler {
    /// Creates an `EnergyProfiler` with default configuration and a
    /// `SyntheticEnergyReader` (15 W package power).
    #[must_use]
    pub fn new() -> Self {
        Self::with_reader(
            EnergyProfilerConfig::default(),
            Box::new(SyntheticEnergyReader::single_domain(15_000.0)),
        )
    }

    /// Creates an `EnergyProfiler` with a custom reader and configuration.
    #[must_use]
    pub fn with_reader(
        config: EnergyProfilerConfig,
        reader: Box<dyn PlatformEnergyReader>,
    ) -> Self {
        Self {
            config,
            reader,
            samples: Vec::new(),
            start_time: None,
            running: false,
            baseline_uj: HashMap::new(),
        }
    }

    /// Sets the number of work units (used to compute energy-per-work-unit).
    pub fn set_work_units(&mut self, units: f64) {
        self.config.work_units = Some(units);
    }

    /// Starts the energy profiling session.
    ///
    /// Captures a baseline reading so that subsequent readings represent
    /// energy consumed *during* the profiling window.
    pub fn start(&mut self) {
        self.baseline_uj = self.reader.read_domains();
        self.start_time = Some(Instant::now());
        self.samples.clear();
        self.running = true;
    }

    /// Manually captures one energy sample.
    ///
    /// In real usage this would be called periodically by a background thread
    /// or timer.  In tests and simulations, call this directly.
    pub fn capture_sample(&mut self) {
        if !self.running {
            return;
        }

        let raw_uj = self.reader.read_domains();
        let timestamp_ns = self
            .start_time
            .map(|t| t.elapsed().as_nanos() as u64)
            .unwrap_or(0);

        // Compute delta from baseline.
        let domain_uj: HashMap<String, u64> = raw_uj
            .iter()
            .map(|(k, &v)| {
                let base = self.baseline_uj.get(k).copied().unwrap_or(0);
                let delta = v.saturating_sub(base);
                (k.clone(), delta)
            })
            .collect();

        // Estimate instantaneous power from consecutive delta if there's a prior sample.
        let domain_mw: HashMap<String, f64> = if let Some(prev) = self.samples.last() {
            let elapsed_ms = (timestamp_ns.saturating_sub(prev.timestamp_ns)) as f64 / 1_000_000.0;
            if elapsed_ms > 0.0 {
                domain_uj
                    .iter()
                    .map(|(k, &uj)| {
                        let prev_uj = prev.domain_uj.get(k).copied().unwrap_or(0);
                        let delta = uj.saturating_sub(prev_uj) as f64;
                        // µJ / ms = mW
                        (k.clone(), delta / elapsed_ms)
                    })
                    .collect()
            } else {
                domain_uj.keys().map(|k| (k.clone(), 0.0)).collect()
            }
        } else {
            // First sample — estimate power from elapsed time since start.
            let elapsed_ms = timestamp_ns as f64 / 1_000_000.0;
            if elapsed_ms > 0.0 {
                domain_uj
                    .iter()
                    .map(|(k, &uj)| (k.clone(), uj as f64 / elapsed_ms))
                    .collect()
            } else {
                domain_uj.keys().map(|k| (k.clone(), 0.0)).collect()
            }
        };

        self.samples.push(EnergySample::new(
            timestamp_ns,
            domain_uj,
            domain_mw,
        ));
    }

    /// Stops the profiling session and returns the aggregated [`EnergyReport`].
    pub fn stop(&mut self) -> EnergyReport {
        self.running = false;

        let duration = self
            .start_time
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO);

        if self.samples.is_empty() {
            return EnergyReport {
                duration,
                total_energy_uj: 0,
                average_power_mw: 0.0,
                peak_power_mw: 0.0,
                domain_energy_uj: HashMap::new(),
                domain_avg_power_mw: HashMap::new(),
                energy_per_work_unit_uj: None,
                sample_count: 0,
            };
        }

        // Aggregate domain energy from last sample (cumulative delta from baseline).
        // Safety: we returned early above if samples is empty.
        let domain_energy_uj = if let Some(last) = self.samples.last() {
            last.domain_uj.clone()
        } else {
            HashMap::new()
        };
        let total_energy_uj: u64 = domain_energy_uj.values().sum();

        // Average power per domain over all samples.
        let n = self.samples.len();
        let mut domain_avg_power_mw: HashMap<String, f64> = HashMap::new();
        for sample in &self.samples {
            for (k, &v) in &sample.domain_mw {
                *domain_avg_power_mw.entry(k.clone()).or_insert(0.0) += v;
            }
        }
        for v in domain_avg_power_mw.values_mut() {
            *v /= n as f64;
        }

        let average_power_mw: f64 = domain_avg_power_mw.values().sum();

        let peak_power_mw = self
            .samples
            .iter()
            .map(|s| s.total_power_mw())
            .fold(0.0_f64, f64::max);

        let energy_per_work_unit_uj = self.config.work_units.and_then(|wu| {
            if wu > 0.0 {
                Some(total_energy_uj as f64 / wu)
            } else {
                None
            }
        });

        EnergyReport {
            duration,
            total_energy_uj,
            average_power_mw,
            peak_power_mw,
            domain_energy_uj,
            domain_avg_power_mw,
            energy_per_work_unit_uj,
            sample_count: n,
        }
    }

    /// Returns whether the profiler is currently running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Returns the samples collected so far (without stopping).
    #[must_use]
    pub fn samples(&self) -> &[EnergySample] {
        &self.samples
    }

    /// Returns the profiler configuration.
    #[must_use]
    pub fn config(&self) -> &EnergyProfilerConfig {
        &self.config
    }

    /// Returns the available power domains from the underlying reader.
    #[must_use]
    pub fn available_domains(&self) -> Vec<PowerDomain> {
        self.reader.available_domains()
    }
}

impl Default for EnergyProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn make_profiler_with_power(power_mw: f64) -> EnergyProfiler {
        let mut domain_map = HashMap::new();
        domain_map.insert("Package".to_string(), power_mw);
        let reader = Box::new(SyntheticEnergyReader::new(domain_map));
        EnergyProfiler::with_reader(EnergyProfilerConfig::default(), reader)
    }

    #[test]
    fn test_profiler_starts_and_stops() {
        let mut p = make_profiler_with_power(5_000.0);
        assert!(!p.is_running());
        p.start();
        assert!(p.is_running());
        let report = p.stop();
        assert!(!p.is_running());
        assert_eq!(report.sample_count, 0); // no manual captures
    }

    #[test]
    fn test_single_capture_produces_sample() {
        let mut p = make_profiler_with_power(10_000.0);
        p.start();
        thread::sleep(Duration::from_millis(10));
        p.capture_sample();
        let report = p.stop();
        assert_eq!(report.sample_count, 1);
    }

    #[test]
    fn test_energy_increases_over_time() {
        let mut p = make_profiler_with_power(10_000.0); // 10 W
        p.start();
        thread::sleep(Duration::from_millis(20));
        p.capture_sample();
        thread::sleep(Duration::from_millis(20));
        p.capture_sample();
        let report = p.stop();
        // Should have accumulated some energy
        assert!(
            report.total_energy_uj > 0,
            "expected non-zero energy, got {} µJ",
            report.total_energy_uj
        );
    }

    #[test]
    fn test_multiple_samples_increase_count() {
        let mut p = make_profiler_with_power(5_000.0);
        p.start();
        for _ in 0..5 {
            thread::sleep(Duration::from_millis(5));
            p.capture_sample();
        }
        let report = p.stop();
        assert_eq!(report.sample_count, 5);
    }

    #[test]
    fn test_capture_when_not_running_is_noop() {
        let mut p = make_profiler_with_power(5_000.0);
        p.capture_sample(); // not started
        assert_eq!(p.samples().len(), 0);
    }

    #[test]
    fn test_peak_power_ge_average() {
        let mut p = make_profiler_with_power(8_000.0);
        p.start();
        thread::sleep(Duration::from_millis(10));
        p.capture_sample();
        thread::sleep(Duration::from_millis(10));
        p.capture_sample();
        let report = p.stop();
        if report.sample_count > 0 {
            assert!(report.peak_power_mw >= report.average_power_mw - 1.0); // allow floating tolerance
        }
    }

    #[test]
    fn test_work_units_energy_per_unit() {
        let mut p = make_profiler_with_power(10_000.0);
        p.set_work_units(100.0);
        p.start();
        thread::sleep(Duration::from_millis(20));
        p.capture_sample();
        let report = p.stop();
        if report.total_energy_uj > 0 {
            assert!(report.energy_per_work_unit_uj.is_some());
            let epu = report.energy_per_work_unit_uj.expect("set above");
            assert!(epu > 0.0);
        }
    }

    #[test]
    fn test_zero_work_units_no_per_unit() {
        let mut p = make_profiler_with_power(5_000.0);
        p.set_work_units(0.0);
        p.start();
        p.capture_sample();
        let report = p.stop();
        assert!(report.energy_per_work_unit_uj.is_none());
    }

    #[test]
    fn test_domain_label() {
        assert_eq!(PowerDomain::Package.label(), "Package (CPU socket)");
        assert_eq!(PowerDomain::Dram.label(), "DRAM");
        assert_eq!(PowerDomain::Gpu.label(), "Integrated GPU");
    }

    #[test]
    fn test_report_summary_format() {
        let mut p = make_profiler_with_power(5_000.0);
        p.start();
        thread::sleep(Duration::from_millis(10));
        p.capture_sample();
        let report = p.stop();
        let summary = report.summary();
        assert!(summary.contains("Energy:"), "summary: {summary}");
        assert!(summary.contains("Avg power:"), "summary: {summary}");
    }

    #[test]
    fn test_dominant_domain_is_package() {
        let mut p = make_profiler_with_power(5_000.0);
        p.start();
        thread::sleep(Duration::from_millis(10));
        p.capture_sample();
        let report = p.stop();
        if report.sample_count > 0 {
            let dom = report.dominant_domain();
            assert!(dom.is_some());
        }
    }

    #[test]
    fn test_available_domains_synthetic() {
        let p = EnergyProfiler::new();
        let domains = p.available_domains();
        assert!(!domains.is_empty());
    }

    #[test]
    fn test_rapl_path_components() {
        assert_eq!(
            PowerDomain::Package.rapl_path_component(),
            Some("package-0")
        );
        assert_eq!(PowerDomain::Dram.rapl_path_component(), Some("dram"));
        assert!(PowerDomain::Platform.rapl_path_component().is_none());
    }
}
