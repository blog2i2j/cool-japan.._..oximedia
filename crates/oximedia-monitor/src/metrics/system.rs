//! System metrics collection (CPU, memory, disk, network, GPU, temperature).

use crate::error::MonitorResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use sysinfo::{
    Components, CpuRefreshKind, Disks, MemoryRefreshKind, Networks, RefreshKind, System,
};

/// System metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU metrics.
    pub cpu: CpuMetrics,
    /// Memory metrics.
    pub memory: MemoryMetrics,
    /// Disk I/O metrics.
    pub disk: DiskMetrics,
    /// Network I/O metrics.
    pub network: NetworkMetrics,
    /// GPU metrics (if available).
    #[cfg(feature = "gpu")]
    pub gpu: Option<GpuMetrics>,
    /// Temperature metrics.
    pub temperature: Option<TemperatureMetrics>,
    /// Timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// CPU metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    /// Total CPU usage (0.0 to 100.0).
    pub total_usage: f64,
    /// Per-core CPU usage.
    pub per_core: Vec<f64>,
    /// Number of CPUs.
    pub cpu_count: usize,
    /// Load average (1, 5, 15 minutes).
    pub load_average: (f64, f64, f64),
}

impl CpuMetrics {
    /// Get the average CPU usage.
    #[must_use]
    pub fn average(&self) -> f64 {
        self.total_usage
    }

    /// Get the maximum per-core usage.
    #[must_use]
    pub fn max_core_usage(&self) -> f64 {
        self.per_core
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(0.0)
    }
}

/// Memory metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Total memory in bytes.
    pub total: u64,
    /// Used memory in bytes.
    pub used: u64,
    /// Available memory in bytes.
    pub available: u64,
    /// Free memory in bytes.
    pub free: u64,
    /// Total swap in bytes.
    pub swap_total: u64,
    /// Used swap in bytes.
    pub swap_used: u64,
    /// Free swap in bytes.
    pub swap_free: u64,
}

impl MemoryMetrics {
    /// Get memory usage percentage.
    #[must_use]
    pub fn usage_percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.used as f64 / self.total as f64) * 100.0
        }
    }

    /// Get swap usage percentage.
    #[must_use]
    pub fn swap_usage_percent(&self) -> f64 {
        if self.swap_total == 0 {
            0.0
        } else {
            (self.swap_used as f64 / self.swap_total as f64) * 100.0
        }
    }
}

/// Disk I/O metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskMetrics {
    /// Per-disk metrics.
    pub disks: HashMap<String, DiskStats>,
    /// Total read bytes.
    pub total_read_bytes: u64,
    /// Total written bytes.
    pub total_write_bytes: u64,
    /// Total read operations.
    pub total_read_ops: u64,
    /// Total write operations.
    pub total_write_ops: u64,
}

/// Per-disk statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskStats {
    /// Mount point.
    pub mount_point: String,
    /// Total space in bytes.
    pub total_space: u64,
    /// Available space in bytes.
    pub available_space: u64,
    /// Used space in bytes.
    pub used_space: u64,
    /// File system type.
    pub fs_type: String,
    /// Is removable.
    pub is_removable: bool,
}

impl DiskStats {
    /// Get disk usage percentage.
    #[must_use]
    pub fn usage_percent(&self) -> f64 {
        if self.total_space == 0 {
            0.0
        } else {
            (self.used_space as f64 / self.total_space as f64) * 100.0
        }
    }
}

/// Network I/O metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Per-interface metrics.
    pub interfaces: HashMap<String, NetworkInterfaceStats>,
    /// Total bytes received.
    pub total_rx_bytes: u64,
    /// Total bytes transmitted.
    pub total_tx_bytes: u64,
    /// Total packets received.
    pub total_rx_packets: u64,
    /// Total packets transmitted.
    pub total_tx_packets: u64,
}

/// Per-interface network statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterfaceStats {
    /// Bytes received.
    pub rx_bytes: u64,
    /// Bytes transmitted.
    pub tx_bytes: u64,
    /// Packets received.
    pub rx_packets: u64,
    /// Packets transmitted.
    pub tx_packets: u64,
    /// Receive errors.
    pub rx_errors: u64,
    /// Transmit errors.
    pub tx_errors: u64,
}

/// GPU metrics (NVIDIA only for now).
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// Per-GPU metrics.
    pub gpus: Vec<GpuStats>,
}

/// Per-GPU statistics.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStats {
    /// GPU index.
    pub index: u32,
    /// GPU name.
    pub name: String,
    /// GPU utilization (0-100).
    pub utilization: u32,
    /// Memory total in bytes.
    pub memory_total: u64,
    /// Memory used in bytes.
    pub memory_used: u64,
    /// Memory free in bytes.
    pub memory_free: u64,
    /// Temperature in Celsius.
    pub temperature: u32,
    /// Power usage in watts.
    pub power_usage: f32,
    /// Fan speed percentage.
    pub fan_speed: u32,
}

#[cfg(feature = "gpu")]
impl GpuStats {
    /// Get memory usage percentage.
    #[must_use]
    pub fn memory_usage_percent(&self) -> f64 {
        if self.memory_total == 0 {
            0.0
        } else {
            (self.memory_used as f64 / self.memory_total as f64) * 100.0
        }
    }
}

/// Temperature metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureMetrics {
    /// Per-component temperatures.
    pub components: HashMap<String, f32>,
}

/// System metrics collector.
pub struct SystemMetricsCollector {
    sys: System,
    disks: Disks,
    networks: Networks,
    /// When `false`, disk enumeration and I/O queries are skipped in
    /// [`collect`](Self::collect).  Useful on systems with many mount points
    /// (e.g. macOS with app-wrapper volumes) where disk queries are very slow.
    enable_disk_metrics: bool,
    #[cfg(feature = "gpu")]
    nvml: Option<nvml_wrapper::Nvml>,
}

impl SystemMetricsCollector {
    /// Create a new system metrics collector with default settings.
    ///
    /// Disk and network lists are initialised lazily: they will be populated
    /// on the first call to [`collect`](Self::collect).  This avoids
    /// expensive I/O at construction time and keeps test construction fast.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn new() -> MonitorResult<Self> {
        Self::new_with_options(true)
    }

    /// Create a new system metrics collector with explicit options.
    ///
    /// `enable_disk_metrics` controls whether disk I/O and storage queries are
    /// performed during [`collect`](Self::collect).  Setting this to `false`
    /// is strongly recommended when running on macOS with many mounted volumes.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn new_with_options(enable_disk_metrics: bool) -> MonitorResult<Self> {
        let sys = System::new_with_specifics(
            RefreshKind::nothing()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        );

        #[cfg(feature = "gpu")]
        let nvml = nvml_wrapper::Nvml::init().ok();

        Ok(Self {
            sys,
            // Use lazy (empty) initialisation so that the expensive disk/network
            // scan is deferred to the first `collect()` call.
            disks: Disks::new(),
            networks: Networks::new(),
            enable_disk_metrics,
            #[cfg(feature = "gpu")]
            nvml,
        })
    }

    /// Collect current system metrics.
    ///
    /// # Errors
    ///
    /// Returns an error if collection fails.
    pub fn collect(&mut self) -> MonitorResult<SystemMetrics> {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();

        if self.enable_disk_metrics {
            self.disks.refresh(true);
            self.networks.refresh(true);
        }

        let cpu = self.collect_cpu_metrics();
        let memory = self.collect_memory_metrics();
        let disk = self.collect_disk_metrics();
        let network = self.collect_network_metrics();

        #[cfg(feature = "gpu")]
        let gpu = self.collect_gpu_metrics();

        // Temperature collection creates a new Components list on every call.
        // Skip it when disk metrics are disabled to keep collection fast.
        let temperature = if self.enable_disk_metrics {
            self.collect_temperature_metrics()
        } else {
            None
        };

        Ok(SystemMetrics {
            cpu,
            memory,
            disk,
            network,
            #[cfg(feature = "gpu")]
            gpu,
            temperature,
            timestamp: chrono::Utc::now(),
        })
    }

    fn collect_cpu_metrics(&self) -> CpuMetrics {
        let cpus = self.sys.cpus();
        let per_core: Vec<f64> = cpus.iter().map(|cpu| f64::from(cpu.cpu_usage())).collect();

        let total_usage = if per_core.is_empty() {
            0.0
        } else {
            per_core.iter().sum::<f64>() / per_core.len() as f64
        };

        let load_avg = System::load_average();

        CpuMetrics {
            total_usage,
            per_core,
            cpu_count: cpus.len(),
            load_average: (load_avg.one, load_avg.five, load_avg.fifteen),
        }
    }

    fn collect_memory_metrics(&self) -> MemoryMetrics {
        MemoryMetrics {
            total: self.sys.total_memory(),
            used: self.sys.used_memory(),
            available: self.sys.available_memory(),
            free: self.sys.free_memory(),
            swap_total: self.sys.total_swap(),
            swap_used: self.sys.used_swap(),
            swap_free: self.sys.free_swap(),
        }
    }

    fn collect_disk_metrics(&self) -> DiskMetrics {
        let mut disks = HashMap::new();
        let total_read_bytes = 0;
        let total_write_bytes = 0;

        for disk in self.disks.list() {
            let mount_point = disk.mount_point().to_string_lossy().to_string();
            let total_space = disk.total_space();
            let available_space = disk.available_space();
            let used_space = total_space.saturating_sub(available_space);

            disks.insert(
                disk.name().to_string_lossy().to_string(),
                DiskStats {
                    mount_point,
                    total_space,
                    available_space,
                    used_space,
                    fs_type: disk.file_system().to_string_lossy().to_string(),
                    is_removable: disk.is_removable(),
                },
            );
        }

        DiskMetrics {
            disks,
            total_read_bytes,
            total_write_bytes,
            total_read_ops: 0,
            total_write_ops: 0,
        }
    }

    fn collect_network_metrics(&self) -> NetworkMetrics {
        let mut interfaces = HashMap::new();
        let mut total_rx_bytes = 0;
        let mut total_tx_bytes = 0;
        let mut total_rx_packets = 0;
        let mut total_tx_packets = 0;

        for (interface_name, data) in self.networks.list() {
            let stats = NetworkInterfaceStats {
                rx_bytes: data.received(),
                tx_bytes: data.transmitted(),
                rx_packets: data.packets_received(),
                tx_packets: data.packets_transmitted(),
                rx_errors: data.errors_on_received(),
                tx_errors: data.errors_on_transmitted(),
            };

            total_rx_bytes += stats.rx_bytes;
            total_tx_bytes += stats.tx_bytes;
            total_rx_packets += stats.rx_packets;
            total_tx_packets += stats.tx_packets;

            interfaces.insert(interface_name.clone(), stats);
        }

        NetworkMetrics {
            interfaces,
            total_rx_bytes,
            total_tx_bytes,
            total_rx_packets,
            total_tx_packets,
        }
    }

    #[cfg(feature = "gpu")]
    fn collect_gpu_metrics(&self) -> Option<GpuMetrics> {
        let nvml = self.nvml.as_ref()?;

        let device_count = nvml.device_count().ok()?;
        let mut gpus = Vec::new();

        for i in 0..device_count {
            if let Ok(device) = nvml.device_by_index(i) {
                let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
                let utilization = device.utilization_rates().map(|u| u.gpu).unwrap_or(0);
                let memory = device.memory_info().ok();
                let temperature = device
                    .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                    .unwrap_or(0);
                let power_usage = device
                    .power_usage()
                    .map(|p| p as f32 / 1000.0)
                    .unwrap_or(0.0);
                let fan_speed = device.fan_speed(0).unwrap_or(0);

                let (memory_total, memory_used, memory_free) = if let Some(mem) = memory {
                    (mem.total, mem.used, mem.free)
                } else {
                    (0, 0, 0)
                };

                gpus.push(GpuStats {
                    index: i,
                    name,
                    utilization,
                    memory_total,
                    memory_used,
                    memory_free,
                    temperature,
                    power_usage,
                    fan_speed,
                });
            }
        }

        if gpus.is_empty() {
            None
        } else {
            Some(GpuMetrics { gpus })
        }
    }

    fn collect_temperature_metrics(&self) -> Option<TemperatureMetrics> {
        let components = Components::new_with_refreshed_list();
        if components.is_empty() {
            return None;
        }
        let mut component_map = HashMap::new();
        for component in &components {
            if let Some(temp) = component.temperature() {
                component_map.insert(component.label().to_string(), temp);
            }
        }
        if component_map.is_empty() {
            None
        } else {
            Some(TemperatureMetrics {
                components: component_map,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_metrics_usage_percent() {
        let metrics = MemoryMetrics {
            total: 1000,
            used: 500,
            available: 500,
            free: 500,
            swap_total: 2000,
            swap_used: 1000,
            swap_free: 1000,
        };

        assert_eq!(metrics.usage_percent(), 50.0);
        assert_eq!(metrics.swap_usage_percent(), 50.0);
    }

    #[test]
    fn test_disk_stats_usage_percent() {
        let stats = DiskStats {
            mount_point: "/".to_string(),
            total_space: 1000,
            available_space: 400,
            used_space: 600,
            fs_type: "ext4".to_string(),
            is_removable: false,
        };

        assert_eq!(stats.usage_percent(), 60.0);
    }

    #[test]
    fn test_cpu_metrics_max_core_usage() {
        let metrics = CpuMetrics {
            total_usage: 50.0,
            per_core: vec![30.0, 50.0, 70.0, 40.0],
            cpu_count: 4,
            load_average: (1.0, 1.5, 2.0),
        };

        assert_eq!(metrics.max_core_usage(), 70.0);
        assert_eq!(metrics.average(), 50.0);
    }

    #[test]
    fn test_system_metrics_collector() {
        // Disable disk I/O to keep the test fast on macOS with many mounts.
        let mut collector =
            SystemMetricsCollector::new_with_options(false).expect("operation should succeed");
        let metrics = collector.collect().expect("failed to collect");

        assert!(metrics.cpu.cpu_count > 0);
        assert!(metrics.memory.total > 0);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_stats_memory_usage_percent() {
        let stats = GpuStats {
            index: 0,
            name: "Test GPU".to_string(),
            utilization: 50,
            memory_total: 1000,
            memory_used: 750,
            memory_free: 250,
            temperature: 65,
            power_usage: 150.0,
            fan_speed: 80,
        };

        assert_eq!(stats.memory_usage_percent(), 75.0);
    }
}
