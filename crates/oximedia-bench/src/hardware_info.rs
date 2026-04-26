#![allow(dead_code)]
//! Hardware information collection for benchmark context.
//!
//! Captures CPU capabilities, memory utilisation, and assesses whether the
//! current host is suitable for reproducible benchmarking.

/// CPU capability information.
#[derive(Debug, Clone)]
pub struct CpuInfo {
    /// Brand / model string.
    pub brand: String,
    /// Physical core count.
    pub physical_cores: usize,
    /// Logical thread count.
    pub logical_threads: usize,
    /// Base clock speed in MHz.
    pub base_freq_mhz: u32,
    /// Whether AVX2 SIMD is available.
    pub avx2: bool,
    /// Whether AVX-512 SIMD is available.
    pub avx512: bool,
    /// Whether Neon (ARM) SIMD is available.
    pub neon: bool,
}

impl CpuInfo {
    /// Construct a `CpuInfo` with sensible defaults (no SIMD).
    pub fn new(brand: impl Into<String>, physical_cores: usize, logical_threads: usize) -> Self {
        Self {
            brand: brand.into(),
            physical_cores,
            logical_threads,
            base_freq_mhz: 0,
            avx2: false,
            avx512: false,
            neon: false,
        }
    }

    /// Returns `true` when AVX2 is available.
    pub fn has_avx2(&self) -> bool {
        self.avx2
    }

    /// Returns `true` when AVX-512 is available.
    pub fn has_avx512(&self) -> bool {
        self.avx512
    }

    /// Returns `true` when Neon is available.
    pub fn has_neon(&self) -> bool {
        self.neon
    }

    /// Describe available SIMD extensions as a string list.
    pub fn simd_extensions(&self) -> Vec<&'static str> {
        let mut exts = Vec::new();
        if self.avx2 {
            exts.push("AVX2");
        }
        if self.avx512 {
            exts.push("AVX-512");
        }
        if self.neon {
            exts.push("Neon");
        }
        exts
    }
}

/// Memory utilisation snapshot.
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total physical RAM in bytes.
    pub total_bytes: u64,
    /// Currently available (free + reclaimable) RAM in bytes.
    pub available_bytes: u64,
}

impl MemoryInfo {
    /// Create a memory info record.
    pub fn new(total_bytes: u64, available_bytes: u64) -> Self {
        Self {
            total_bytes,
            available_bytes: available_bytes.min(total_bytes),
        }
    }

    /// Used bytes.
    pub fn used_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.available_bytes)
    }

    /// Utilisation as a percentage (0.0–100.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization_pct(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.used_bytes() as f64 / self.total_bytes as f64 * 100.0
    }

    /// Returns `true` when at least `required` bytes are available.
    pub fn has_available(&self, required: u64) -> bool {
        self.available_bytes >= required
    }
}

/// A point-in-time snapshot of hardware state.
#[derive(Debug, Clone)]
pub struct HardwareSnapshot {
    /// CPU information.
    pub cpu: CpuInfo,
    /// Memory information.
    pub memory: MemoryInfo,
    /// Number of available parallelism units (from the OS).
    pub parallelism: usize,
}

impl HardwareSnapshot {
    /// Assess whether the current snapshot is suitable for reproducible benchmarking.
    ///
    /// Conditions required:
    /// - Memory utilisation < 80 %
    /// - At least 2 logical threads available
    pub fn is_suitable_for_benchmark(&self) -> bool {
        self.memory.utilization_pct() < 80.0 && self.cpu.logical_threads >= 2
    }

    /// Produce a human-readable diagnostic string.
    pub fn diagnostics(&self) -> String {
        format!(
            "CPU: {} ({} cores / {} threads) | RAM: {:.1}% used | Parallelism: {} | SIMD: [{}] | Suitable: {}",
            self.cpu.brand,
            self.cpu.physical_cores,
            self.cpu.logical_threads,
            self.memory.utilization_pct(),
            self.parallelism,
            self.cpu.simd_extensions().join(", "),
            self.is_suitable_for_benchmark(),
        )
    }
}

/// Collects hardware information from the current host.
pub struct HardwareInfoCollector;

impl HardwareInfoCollector {
    /// Create a new collector.
    pub fn new() -> Self {
        Self
    }

    /// Capture a hardware snapshot.
    ///
    /// In the absence of platform-specific APIs, this uses conservative
    /// heuristics (parallelism from `std::thread`, no SIMD auto-detection).
    pub fn snapshot(&self) -> HardwareSnapshot {
        let parallelism = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let cpu = CpuInfo {
            brand: "Unknown CPU".to_string(),
            physical_cores: (parallelism / 2).max(1),
            logical_threads: parallelism,
            base_freq_mhz: 0,
            avx2: cfg!(target_feature = "avx2"),
            avx512: cfg!(target_feature = "avx512f"),
            neon: cfg!(target_feature = "neon"),
        };

        // Query real OS memory using sysinfo.
        let mut sys = sysinfo::System::new_with_specifics(
            sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
        );
        sys.refresh_memory();
        let total = sys.total_memory();
        let available = sys.available_memory();
        let memory = MemoryInfo::new(total, available);

        HardwareSnapshot {
            cpu,
            memory,
            parallelism,
        }
    }
}

impl Default for HardwareInfoCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_info_has_avx2_false_by_default() {
        let cpu = CpuInfo::new("Test CPU", 4, 8);
        assert!(!cpu.has_avx2());
    }

    #[test]
    fn test_cpu_info_has_avx2_true() {
        let mut cpu = CpuInfo::new("Test CPU", 4, 8);
        cpu.avx2 = true;
        assert!(cpu.has_avx2());
    }

    #[test]
    fn test_cpu_info_simd_extensions_empty() {
        let cpu = CpuInfo::new("Bare CPU", 2, 2);
        assert!(cpu.simd_extensions().is_empty());
    }

    #[test]
    fn test_cpu_info_simd_extensions_avx2() {
        let mut cpu = CpuInfo::new("Modern CPU", 8, 16);
        cpu.avx2 = true;
        let exts = cpu.simd_extensions();
        assert_eq!(exts, vec!["AVX2"]);
    }

    #[test]
    fn test_cpu_info_simd_extensions_multiple() {
        let mut cpu = CpuInfo::new("Server CPU", 16, 32);
        cpu.avx2 = true;
        cpu.avx512 = true;
        let exts = cpu.simd_extensions();
        assert!(exts.contains(&"AVX2"));
        assert!(exts.contains(&"AVX-512"));
    }

    #[test]
    fn test_memory_info_utilization_full() {
        let m = MemoryInfo::new(1000, 0);
        assert!((m.utilization_pct() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_memory_info_utilization_half() {
        let m = MemoryInfo::new(1000, 500);
        assert!((m.utilization_pct() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_memory_info_utilization_zero_total() {
        let m = MemoryInfo::new(0, 0);
        assert_eq!(m.utilization_pct(), 0.0);
    }

    #[test]
    fn test_memory_info_has_available_enough() {
        let m = MemoryInfo::new(1_000_000, 500_000);
        assert!(m.has_available(400_000));
    }

    #[test]
    fn test_memory_info_has_available_not_enough() {
        let m = MemoryInfo::new(1_000_000, 100_000);
        assert!(!m.has_available(200_000));
    }

    #[test]
    fn test_memory_info_clamps_available() {
        let m = MemoryInfo::new(100, 999);
        assert_eq!(m.available_bytes, 100);
    }

    #[test]
    fn test_hardware_snapshot_suitable() {
        let cpu = CpuInfo::new("Fast CPU", 8, 16);
        let memory = MemoryInfo::new(8 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024); // 50% used
        let snap = HardwareSnapshot {
            cpu,
            memory,
            parallelism: 16,
        };
        assert!(snap.is_suitable_for_benchmark());
    }

    #[test]
    fn test_hardware_snapshot_not_suitable_high_memory() {
        let cpu = CpuInfo::new("Fast CPU", 8, 16);
        // 95% used
        let total = 1_000_000u64;
        let memory = MemoryInfo::new(total, total / 20);
        let snap = HardwareSnapshot {
            cpu,
            memory,
            parallelism: 16,
        };
        assert!(!snap.is_suitable_for_benchmark());
    }

    #[test]
    fn test_hardware_snapshot_not_suitable_single_thread() {
        let cpu = CpuInfo::new("Weak CPU", 1, 1);
        let memory = MemoryInfo::new(4_000_000, 3_000_000);
        let snap = HardwareSnapshot {
            cpu,
            memory,
            parallelism: 1,
        };
        assert!(!snap.is_suitable_for_benchmark());
    }

    #[test]
    fn test_hardware_snapshot_diagnostics_nonempty() {
        let collector = HardwareInfoCollector::new();
        let snap = collector.snapshot();
        let diag = snap.diagnostics();
        assert!(!diag.is_empty());
        assert!(diag.contains("CPU:"));
        assert!(diag.contains("RAM:"));
    }

    #[test]
    fn test_collector_snapshot_positive_parallelism() {
        let collector = HardwareInfoCollector::default();
        let snap = collector.snapshot();
        assert!(snap.parallelism >= 1);
        assert!(snap.cpu.logical_threads >= 1);
    }

    #[test]
    fn test_memory_info_nonzero() {
        let collector = HardwareInfoCollector::new();
        let snap = collector.snapshot();
        // On any real host, total memory must be > 0.
        assert!(
            snap.memory.total_bytes > 0,
            "total_bytes should be > 0 (got {})",
            snap.memory.total_bytes
        );
    }
}
