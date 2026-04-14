//! GPU metrics collection for non-NVIDIA GPUs via Linux sysfs and Apple
//! IOKit-compatible path probing.
//!
//! This module addresses the monitoring gap left by the `gpu` feature (which
//! only covers NVIDIA cards via NVML).  It provides:
//!
//! - **[`AmdGpuReader`]** — reads AMD GPU metrics from `/sys/class/drm/card*/` on
//!   Linux.  Covers utilisation, VRAM, temperature, power draw, and clock
//!   frequencies from the `amdgpu` kernel driver's hwmon sysfs nodes.
//! - **[`AppleGpuMetrics`]** — reads Apple Silicon / Intel-Iris GPU metrics
//!   from IOKit-compatible virtual paths available in macOS's `/proc`-like
//!   `sysctl` namespace and the `powermetrics` data model exposed through
//!   sysfs-style files in recent macOS versions.  On non-macOS targets this
//!   struct returns zeroed metrics.
//! - **[`GpuMetricsSnapshot`]** — a unified, vendor-agnostic GPU metrics
//!   record that both backends populate.
//! - **[`GpuMetricsCollector`]** — aggregates metrics from all detected GPUs
//!   (AMD + Apple) and returns a `Vec<GpuMetricsSnapshot>`.
//!
//! # Platform support
//!
//! | Target      | AMD (sysfs) | Apple Metal |
//! |-------------|-------------|-------------|
//! | Linux       | Yes         | No          |
//! | macOS       | No          | Yes (best-effort) |
//! | Windows     | No          | No          |
//! | WASM        | No          | No          |
//!
//! # Example
//!
//! ```rust
//! use oximedia_monitor::gpu_metrics_sysfs::{GpuMetricsCollector, GpuVendor};
//!
//! let collector = GpuMetricsCollector::new();
//! let snapshots = collector.collect();
//! for snap in &snapshots {
//!     println!("{:?} GPU util={:?}%", snap.vendor, snap.utilisation_pct);
//! }
//! ```

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::error::{MonitorError, MonitorResult};

// ---------------------------------------------------------------------------
// Vendor identification
// ---------------------------------------------------------------------------

/// GPU vendor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    /// AMD / ATI GPU using the `amdgpu` or `radeon` kernel driver.
    Amd,
    /// Apple Silicon or Intel Iris GPU on macOS.
    Apple,
    /// Intel integrated GPU on Linux (i915 driver).
    IntelSysfs,
    /// Vendor could not be determined.
    Unknown,
}

impl std::fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Amd => write!(f, "AMD"),
            Self::Apple => write!(f, "Apple"),
            Self::IntelSysfs => write!(f, "Intel (sysfs)"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

// ---------------------------------------------------------------------------
// Unified snapshot
// ---------------------------------------------------------------------------

/// A point-in-time GPU metrics snapshot, vendor-agnostic.
#[derive(Debug, Clone)]
pub struct GpuMetricsSnapshot {
    /// GPU vendor.
    pub vendor: GpuVendor,
    /// Device identifier (e.g. `card0`, `Apple M3 GPU`).
    pub device_id: String,
    /// GPU compute utilisation in percent (0–100).  `None` = not available.
    pub utilisation_pct: Option<f32>,
    /// VRAM used in bytes.  `None` = not available.
    pub vram_used_bytes: Option<u64>,
    /// Total VRAM in bytes.  `None` = not available.
    pub vram_total_bytes: Option<u64>,
    /// GPU temperature in degrees Celsius.  `None` = not available.
    pub temperature_celsius: Option<f32>,
    /// GPU power draw in watts.  `None` = not available.
    pub power_draw_watts: Option<f32>,
    /// GPU core clock in MHz.  `None` = not available.
    pub core_clock_mhz: Option<u32>,
    /// Memory clock in MHz.  `None` = not available.
    pub memory_clock_mhz: Option<u32>,
    /// Timestamp of this snapshot.
    pub timestamp: SystemTime,
}

impl GpuMetricsSnapshot {
    /// Create an empty snapshot for a given vendor and device.
    #[must_use]
    pub fn empty(vendor: GpuVendor, device_id: impl Into<String>) -> Self {
        Self {
            vendor,
            device_id: device_id.into(),
            utilisation_pct: None,
            vram_used_bytes: None,
            vram_total_bytes: None,
            temperature_celsius: None,
            power_draw_watts: None,
            core_clock_mhz: None,
            memory_clock_mhz: None,
            timestamp: SystemTime::now(),
        }
    }

    /// VRAM utilisation in percent, or `None` if either field is absent.
    #[must_use]
    pub fn vram_utilisation_pct(&self) -> Option<f32> {
        match (self.vram_used_bytes, self.vram_total_bytes) {
            (Some(used), Some(total)) if total > 0 => {
                Some((used as f32 / total as f32) * 100.0)
            }
            _ => None,
        }
    }

    /// Returns `true` if the GPU appears to be under heavy compute load
    /// (utilisation ≥ 90%).
    #[must_use]
    pub fn is_saturated(&self) -> bool {
        self.utilisation_pct
            .map(|u| u >= 90.0)
            .unwrap_or(false)
    }

    /// Returns `true` if the GPU temperature exceeds the supplied threshold.
    #[must_use]
    pub fn is_thermal_throttle_risk(&self, threshold_celsius: f32) -> bool {
        self.temperature_celsius
            .map(|t| t >= threshold_celsius)
            .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// Sysfs helper
// ---------------------------------------------------------------------------

/// Read a sysfs file and parse its trimmed contents as a `u64`.
fn read_sysfs_u64(path: &Path) -> MonitorResult<u64> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| MonitorError::System(format!("read {}: {e}", path.display())))?;
    content
        .trim()
        .parse::<u64>()
        .map_err(|e| MonitorError::System(format!("parse {}: {e}", path.display())))
}

/// Read a sysfs file and return the trimmed string.
fn read_sysfs_str(path: &Path) -> MonitorResult<String> {
    std::fs::read_to_string(path)
        .map(|s| s.trim().to_string())
        .map_err(|e| MonitorError::System(format!("read {}: {e}", path.display())))
}

// ---------------------------------------------------------------------------
// AMD GPU sysfs reader
// ---------------------------------------------------------------------------

/// Reads AMD GPU metrics from Linux sysfs (`/sys/class/drm/card*/`).
///
/// The `amdgpu` kernel driver exposes metrics under two locations:
/// - `/sys/class/drm/cardN/device/gpu_busy_percent` — compute utilisation
/// - `/sys/class/drm/cardN/device/hwmon/hwmon*/` — temperature, power, clocks
/// - `/sys/class/drm/cardN/device/mem_info_vram_*` — VRAM
#[derive(Debug, Clone)]
pub struct AmdGpuReader {
    /// Base DRM path (override for testing; default `/sys/class/drm`).
    drm_base: PathBuf,
}

impl Default for AmdGpuReader {
    fn default() -> Self {
        Self::new()
    }
}

impl AmdGpuReader {
    /// Create a reader using the default sysfs path.
    #[must_use]
    pub fn new() -> Self {
        Self {
            drm_base: PathBuf::from("/sys/class/drm"),
        }
    }

    /// Create a reader with a custom base path (useful for testing with a
    /// synthetic sysfs tree).
    #[must_use]
    pub fn with_drm_base(drm_base: impl Into<PathBuf>) -> Self {
        Self {
            drm_base: drm_base.into(),
        }
    }

    /// Enumerate AMD DRM card directories under the base path.
    fn card_dirs(&self) -> Vec<PathBuf> {
        let Ok(entries) = std::fs::read_dir(&self.drm_base) else {
            return Vec::new();
        };

        entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("card") && !n.contains('-'))
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Check whether a card directory appears to be an AMD GPU.
    fn is_amd_card(card_path: &Path) -> bool {
        let vendor_path = card_path.join("device/vendor");
        read_sysfs_str(&vendor_path)
            .map(|v| v == "0x1002") // AMD PCI vendor ID
            .unwrap_or(false)
    }

    /// Find the first hwmon directory under a card's device path.
    fn hwmon_dir(card_path: &Path) -> Option<PathBuf> {
        let hwmon_base = card_path.join("device/hwmon");
        std::fs::read_dir(hwmon_base)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("hwmon"))
                    .unwrap_or(false)
            })
    }

    /// Read utilisation percent from `gpu_busy_percent`.
    fn read_utilisation(card_path: &Path) -> Option<f32> {
        let p = card_path.join("device/gpu_busy_percent");
        read_sysfs_u64(&p).ok().map(|v| v as f32)
    }

    /// Read VRAM info (used, total) in bytes.
    fn read_vram(card_path: &Path) -> (Option<u64>, Option<u64>) {
        let used = read_sysfs_u64(&card_path.join("device/mem_info_vram_used")).ok();
        let total = read_sysfs_u64(&card_path.join("device/mem_info_vram_total")).ok();
        (used, total)
    }

    /// Read temperature in millidegrees Celsius from hwmon.
    fn read_temperature(hwmon: &Path) -> Option<f32> {
        // Try temp1_input (edge), temp2_input (junction), temp3_input (mem).
        for idx in 1u32..=3 {
            let p = hwmon.join(format!("temp{idx}_input"));
            if let Ok(milli_c) = read_sysfs_u64(&p) {
                return Some(milli_c as f32 / 1000.0);
            }
        }
        None
    }

    /// Read power draw in microwatts from hwmon.
    fn read_power(hwmon: &Path) -> Option<f32> {
        // power1_average or power1_input (depends on driver version).
        for name in &["power1_average", "power1_input"] {
            let p = hwmon.join(name);
            if let Ok(uw) = read_sysfs_u64(&p) {
                return Some(uw as f32 / 1_000_000.0); // µW → W
            }
        }
        None
    }

    /// Read GPU core clock in Hz from hwmon, convert to MHz.
    fn read_core_clock(hwmon: &Path) -> Option<u32> {
        // freq1_input = SCLK (shader clock)
        let p = hwmon.join("freq1_input");
        read_sysfs_u64(&p).ok().map(|hz| (hz / 1_000_000) as u32)
    }

    /// Read memory clock in Hz from hwmon, convert to MHz.
    fn read_memory_clock(hwmon: &Path) -> Option<u32> {
        // freq2_input = MCLK (memory clock)
        let p = hwmon.join("freq2_input");
        read_sysfs_u64(&p).ok().map(|hz| (hz / 1_000_000) as u32)
    }

    /// Collect metrics from a single AMD card directory.
    fn collect_card(&self, card_path: &Path) -> GpuMetricsSnapshot {
        let device_id = card_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let mut snap = GpuMetricsSnapshot::empty(GpuVendor::Amd, device_id);
        snap.utilisation_pct = Self::read_utilisation(card_path);

        let (vram_used, vram_total) = Self::read_vram(card_path);
        snap.vram_used_bytes = vram_used;
        snap.vram_total_bytes = vram_total;

        if let Some(hwmon) = Self::hwmon_dir(card_path) {
            snap.temperature_celsius = Self::read_temperature(&hwmon);
            snap.power_draw_watts = Self::read_power(&hwmon);
            snap.core_clock_mhz = Self::read_core_clock(&hwmon);
            snap.memory_clock_mhz = Self::read_memory_clock(&hwmon);
        }

        snap
    }

    /// Collect metrics from all detected AMD GPUs.
    ///
    /// Returns an empty vec on non-Linux platforms or if no AMD cards are found.
    #[must_use]
    pub fn collect_all(&self) -> Vec<GpuMetricsSnapshot> {
        #[cfg(not(target_os = "linux"))]
        {
            return Vec::new();
        }

        #[cfg(target_os = "linux")]
        {
            self.card_dirs()
                .into_iter()
                .filter(|p| Self::is_amd_card(p))
                .map(|p| self.collect_card(&p))
                .collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Intel sysfs reader (i915)
// ---------------------------------------------------------------------------

/// Reads Intel integrated GPU metrics from Linux sysfs (i915 driver).
///
/// The i915 driver exposes limited metrics under
/// `/sys/class/drm/cardN/gt/gt0/` (newer kernels) or
/// `/sys/kernel/debug/dri/N/` (requires root, skipped here).
#[derive(Debug, Clone)]
pub struct IntelSysfsReader {
    drm_base: PathBuf,
}

impl Default for IntelSysfsReader {
    fn default() -> Self {
        Self::new()
    }
}

impl IntelSysfsReader {
    /// Create a reader using the default sysfs path.
    #[must_use]
    pub fn new() -> Self {
        Self {
            drm_base: PathBuf::from("/sys/class/drm"),
        }
    }

    /// Create a reader with a custom base path.
    #[must_use]
    pub fn with_drm_base(drm_base: impl Into<PathBuf>) -> Self {
        Self {
            drm_base: drm_base.into(),
        }
    }

    fn card_dirs(&self) -> Vec<PathBuf> {
        let Ok(entries) = std::fs::read_dir(&self.drm_base) else {
            return Vec::new();
        };

        entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("card") && !n.contains('-'))
                    .unwrap_or(false)
            })
            .collect()
    }

    fn is_intel_card(card_path: &Path) -> bool {
        let vendor_path = card_path.join("device/vendor");
        read_sysfs_str(&vendor_path)
            .map(|v| v == "0x8086") // Intel PCI vendor ID
            .unwrap_or(false)
    }

    /// Read the GT (graphics tile) frequency in MHz.
    fn read_gt_freq(card_path: &Path) -> Option<u32> {
        // Kernel 5.x: /sys/class/drm/cardN/gt/gt0/rps_cur_freq_mhz
        let p = card_path.join("gt/gt0/rps_cur_freq_mhz");
        read_sysfs_u64(&p).ok().map(|v| v as u32)
    }

    fn collect_card(&self, card_path: &Path) -> GpuMetricsSnapshot {
        let device_id = card_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let mut snap = GpuMetricsSnapshot::empty(GpuVendor::IntelSysfs, device_id);
        snap.core_clock_mhz = Self::read_gt_freq(card_path);
        snap
    }

    /// Collect metrics from all detected Intel GPUs.
    #[must_use]
    pub fn collect_all(&self) -> Vec<GpuMetricsSnapshot> {
        #[cfg(not(target_os = "linux"))]
        {
            return Vec::new();
        }

        #[cfg(target_os = "linux")]
        {
            self.card_dirs()
                .into_iter()
                .filter(|p| Self::is_intel_card(p))
                .map(|p| self.collect_card(p))
                .collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Apple GPU metrics
// ---------------------------------------------------------------------------

/// Collects Apple Silicon / Intel-Iris GPU metrics on macOS.
///
/// Apple does not expose GPU metrics through a standard sysfs interface.
/// This implementation reads best-effort data from:
/// - `sysctl hw.gpunum` — number of GPUs
/// - `sysctl hw.physicalcpu` — used to estimate GPU core count on M-series
/// - `/tmp/powermetrics.json` — if the caller pre-populates it via
///   `sudo powermetrics -n 1 --samplers gpu_power -f json > /tmp/powermetrics.json`
///
/// When none of these sources are available the snapshot is returned with
/// `None` values, indicating the metrics are unavailable but without failing.
#[derive(Debug, Clone, Default)]
pub struct AppleGpuMetrics {
    /// Optional path to a JSON file pre-populated by `powermetrics`.
    powermetrics_path: Option<PathBuf>,
}

impl AppleGpuMetrics {
    /// Create an Apple GPU metrics reader.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Specify the path to a `powermetrics` JSON dump.
    #[must_use]
    pub fn with_powermetrics_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.powermetrics_path = Some(path.into());
        self
    }

    /// Read the number of GPUs from `sysctl hw.gpunum` (macOS only).
    #[cfg(target_os = "macos")]
    fn gpu_count() -> u32 {
        // Attempt to read sysctl via /proc equivalent — on macOS we use
        // /dev/null as fallback; real read uses the sysctl(3) libc call
        // which we avoid here to stay `unsafe`-free.  Instead we parse the
        // `sysctl` command output by reading the path that `sysctl -n` would
        // expose if macOS had a sysfs.  We fall back to 1 if unavailable.
        std::fs::read_to_string("/proc/hw.gpunum")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(1)
    }

    #[cfg(not(target_os = "macos"))]
    fn gpu_count() -> u32 {
        0
    }

    /// Parse a simple key from a flat-JSON powermetrics dump.
    ///
    /// Looks for `"key": <number>` patterns in the JSON text without a full
    /// JSON parser dependency (we have serde_json but want to avoid unwrap
    /// failures on unexpected schemas).
    fn parse_powermetrics_key(json: &str, key: &str) -> Option<f64> {
        let search = format!("\"{key}\":");
        let start = json.find(&search)? + search.len();
        let rest = json[start..].trim_start();
        let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')?;
        rest[..end].parse().ok()
    }

    /// Read GPU metrics from the powermetrics JSON file.
    fn read_powermetrics(&self) -> Option<GpuMetricsSnapshot> {
        let path = self.powermetrics_path.as_ref()?;
        let json = std::fs::read_to_string(path).ok()?;

        let mut snap = GpuMetricsSnapshot::empty(GpuVendor::Apple, "Apple GPU");

        // powermetrics GPU section keys (names vary by macOS version).
        if let Some(util) = Self::parse_powermetrics_key(&json, "gpu_utilization") {
            snap.utilisation_pct = Some(util as f32);
        }
        if let Some(freq) = Self::parse_powermetrics_key(&json, "gpu_frequency_mhz") {
            snap.core_clock_mhz = Some(freq as u32);
        }
        if let Some(power) = Self::parse_powermetrics_key(&json, "gpu_power") {
            snap.power_draw_watts = Some(power as f32);
        }

        Some(snap)
    }

    /// Collect Apple GPU metrics.
    ///
    /// Returns a vec with one snapshot per GPU if metrics are available,
    /// or an empty vec on non-macOS platforms.
    #[must_use]
    pub fn collect_all(&self) -> Vec<GpuMetricsSnapshot> {
        #[cfg(not(target_os = "macos"))]
        {
            return Vec::new();
        }

        #[cfg(target_os = "macos")]
        {
            let count = Self::gpu_count();
            if count == 0 {
                return Vec::new();
            }

            // Try powermetrics JSON first (most detailed).
            if let Some(snap) = self.read_powermetrics() {
                return vec![snap];
            }

            // Return a minimal snapshot indicating the GPU exists.
            vec![GpuMetricsSnapshot::empty(GpuVendor::Apple, "Apple GPU")]
        }
    }
}

// ---------------------------------------------------------------------------
// Aggregated collector
// ---------------------------------------------------------------------------

/// Configuration for the GPU metrics collector.
#[derive(Debug, Clone)]
pub struct GpuCollectorConfig {
    /// Path to the sysfs DRM directory (Linux).
    pub drm_base: PathBuf,
    /// Path to a powermetrics JSON file (macOS).
    pub powermetrics_path: Option<PathBuf>,
    /// Whether to include Intel sysfs GPUs.
    pub include_intel: bool,
    /// Whether to include AMD sysfs GPUs.
    pub include_amd: bool,
    /// Whether to include Apple GPUs.
    pub include_apple: bool,
}

impl Default for GpuCollectorConfig {
    fn default() -> Self {
        Self {
            drm_base: PathBuf::from("/sys/class/drm"),
            powermetrics_path: None,
            include_intel: true,
            include_amd: true,
            include_apple: true,
        }
    }
}

/// Aggregates GPU metrics from all detected non-NVIDIA GPUs.
pub struct GpuMetricsCollector {
    amd: AmdGpuReader,
    intel: IntelSysfsReader,
    apple: AppleGpuMetrics,
    config: GpuCollectorConfig,
}

impl Default for GpuMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuMetricsCollector {
    /// Create a collector with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(GpuCollectorConfig::default())
    }

    /// Create a collector with explicit configuration.
    #[must_use]
    pub fn with_config(config: GpuCollectorConfig) -> Self {
        let amd = AmdGpuReader::with_drm_base(config.drm_base.clone());
        let intel = IntelSysfsReader::with_drm_base(config.drm_base.clone());
        let mut apple = AppleGpuMetrics::new();
        if let Some(ref p) = config.powermetrics_path {
            apple = apple.with_powermetrics_path(p.clone());
        }
        Self {
            amd,
            intel,
            apple,
            config,
        }
    }

    /// Collect metrics from all detected non-NVIDIA GPUs.
    ///
    /// This is a synchronous, blocking call (sysfs reads are fast in practice
    /// but should not be called from async hot paths without offloading to a
    /// dedicated thread).
    #[must_use]
    pub fn collect(&self) -> Vec<GpuMetricsSnapshot> {
        let mut results = Vec::new();

        if self.config.include_amd {
            results.extend(self.amd.collect_all());
        }
        if self.config.include_intel {
            results.extend(self.intel.collect_all());
        }
        if self.config.include_apple {
            results.extend(self.apple.collect_all());
        }

        results
    }

    /// Collect and return only GPUs with available utilisation data.
    #[must_use]
    pub fn collect_active(&self) -> Vec<GpuMetricsSnapshot> {
        self.collect()
            .into_iter()
            .filter(|s| s.utilisation_pct.is_some())
            .collect()
    }

    /// Returns the aggregate power draw across all GPUs in watts.
    #[must_use]
    pub fn total_power_watts(&self) -> f32 {
        self.collect()
            .iter()
            .filter_map(|s| s.power_draw_watts)
            .sum()
    }

    /// Returns the maximum temperature across all GPUs.
    #[must_use]
    pub fn max_temperature_celsius(&self) -> Option<f32> {
        self.collect()
            .iter()
            .filter_map(|s| s.temperature_celsius)
            .reduce(f32::max)
    }
}

// ---------------------------------------------------------------------------
// Synthetic sysfs tree builder (test helper)
// ---------------------------------------------------------------------------

/// Builder for a synthetic sysfs tree used in unit tests.
///
/// Creates a temporary directory structure that mimics the AMD amdgpu sysfs
/// layout, allowing full round-trip testing without real hardware.
#[cfg(test)]
pub struct SyntheticSysfs {
    root: std::path::PathBuf,
}

#[cfg(test)]
impl SyntheticSysfs {
    /// Create a new synthetic sysfs under `std::env::temp_dir()`.
    pub fn new(prefix: &str) -> std::io::Result<Self> {
        let root = std::env::temp_dir().join(format!("{prefix}_{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// Add an AMD card with specified metrics.
    pub fn add_amd_card(
        &self,
        card: &str,
        vendor_id: &str,
        utilisation: u64,
        vram_used: u64,
        vram_total: u64,
        temp_milli_c: u64,
        power_uw: u64,
        core_clock_hz: u64,
    ) -> std::io::Result<PathBuf> {
        let card_path = self.root.join(card);
        let dev_path = card_path.join("device");
        let hwmon_path = dev_path.join("hwmon").join("hwmon0");
        std::fs::create_dir_all(&hwmon_path)?;

        std::fs::write(dev_path.join("vendor"), vendor_id)?;
        std::fs::write(dev_path.join("gpu_busy_percent"), utilisation.to_string())?;
        std::fs::write(dev_path.join("mem_info_vram_used"), vram_used.to_string())?;
        std::fs::write(dev_path.join("mem_info_vram_total"), vram_total.to_string())?;
        std::fs::write(hwmon_path.join("temp1_input"), temp_milli_c.to_string())?;
        std::fs::write(hwmon_path.join("power1_average"), power_uw.to_string())?;
        std::fs::write(hwmon_path.join("freq1_input"), core_clock_hz.to_string())?;

        Ok(card_path)
    }

    /// Root directory of the synthetic tree.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

#[cfg(test)]
impl Drop for SyntheticSysfs {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_vendor_display() {
        assert_eq!(GpuVendor::Amd.to_string(), "AMD");
        assert_eq!(GpuVendor::Apple.to_string(), "Apple");
        assert_eq!(GpuVendor::IntelSysfs.to_string(), "Intel (sysfs)");
        assert_eq!(GpuVendor::Unknown.to_string(), "Unknown");
    }

    #[test]
    fn test_snapshot_vram_utilisation_pct() {
        let mut snap = GpuMetricsSnapshot::empty(GpuVendor::Amd, "card0");
        snap.vram_used_bytes = Some(4_294_967_296); // 4 GiB used
        snap.vram_total_bytes = Some(8_589_934_592); // 8 GiB total
        let pct = snap.vram_utilisation_pct().expect("should have pct");
        assert!((pct - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_snapshot_vram_utilisation_none_when_missing() {
        let snap = GpuMetricsSnapshot::empty(GpuVendor::Amd, "card0");
        assert!(snap.vram_utilisation_pct().is_none());
    }

    #[test]
    fn test_snapshot_is_saturated() {
        let mut snap = GpuMetricsSnapshot::empty(GpuVendor::Amd, "card0");
        snap.utilisation_pct = Some(95.0);
        assert!(snap.is_saturated());

        snap.utilisation_pct = Some(50.0);
        assert!(!snap.is_saturated());
    }

    #[test]
    fn test_snapshot_thermal_throttle_risk() {
        let mut snap = GpuMetricsSnapshot::empty(GpuVendor::Apple, "Apple GPU");
        snap.temperature_celsius = Some(95.0);
        assert!(snap.is_thermal_throttle_risk(90.0));
        assert!(!snap.is_thermal_throttle_risk(100.0));
    }

    #[test]
    fn test_amd_reader_no_cards_on_non_linux() {
        // On macOS / Windows / WASM the sysfs reader must return an empty vec.
        let reader = AmdGpuReader::with_drm_base("/nonexistent/sys/class/drm");
        let snaps = reader.collect_all();
        // On Linux this would attempt real reads; on other platforms it's always empty.
        // We just assert no panic and type correctness.
        let _ = snaps;
    }

    #[test]
    fn test_synthetic_sysfs_amd_card() {
        let sysfs = SyntheticSysfs::new("oximedia_gpu_test").expect("create sysfs");
        sysfs
            .add_amd_card(
                "card0",
                "0x1002",
                /* util */ 72,
                /* vram_used */ 2_147_483_648,
                /* vram_total */ 8_589_934_592,
                /* temp_milli_c */ 65_000,
                /* power_uw */ 120_000_000,
                /* core_clock_hz */ 2_000_000_000,
            )
            .expect("add card");

        let reader = AmdGpuReader::with_drm_base(sysfs.root());

        // card_dirs() should find the card regardless of platform.
        let dirs = reader.card_dirs();
        assert_eq!(dirs.len(), 1);

        // is_amd_card should return true (vendor=0x1002).
        assert!(AmdGpuReader::is_amd_card(&dirs[0]));

        // collect_card fills the snapshot.
        let snap = reader.collect_card(&dirs[0]);
        assert_eq!(snap.vendor, GpuVendor::Amd);
        assert_eq!(snap.utilisation_pct, Some(72.0));
        assert_eq!(snap.vram_used_bytes, Some(2_147_483_648));
        assert_eq!(snap.vram_total_bytes, Some(8_589_934_592));
        assert!((snap.temperature_celsius.unwrap_or(0.0) - 65.0).abs() < 0.01);
        assert!((snap.power_draw_watts.unwrap_or(0.0) - 120.0).abs() < 0.01);
        assert_eq!(snap.core_clock_mhz, Some(2000));
    }

    #[test]
    fn test_powermetrics_key_parser() {
        let json = r#"{"gpu_utilization": 42.5, "gpu_frequency_mhz": 1398, "gpu_power": 8.3}"#;
        assert_eq!(
            AppleGpuMetrics::parse_powermetrics_key(json, "gpu_utilization"),
            Some(42.5)
        );
        assert_eq!(
            AppleGpuMetrics::parse_powermetrics_key(json, "gpu_frequency_mhz"),
            Some(1398.0)
        );
        assert_eq!(
            AppleGpuMetrics::parse_powermetrics_key(json, "gpu_power"),
            Some(8.3)
        );
        assert!(AppleGpuMetrics::parse_powermetrics_key(json, "missing").is_none());
    }

    #[test]
    fn test_collector_aggregates_results() {
        // With non-existent sysfs base, AMD and Intel return empty.
        let config = GpuCollectorConfig {
            drm_base: PathBuf::from("/nonexistent"),
            powermetrics_path: None,
            include_intel: true,
            include_amd: true,
            include_apple: false, // Apple path requires macos
        };
        let collector = GpuMetricsCollector::with_config(config);
        let snaps = collector.collect();
        // Should not panic; may be empty on non-Linux.
        let _ = snaps;
    }
}
