//! Render farm resource management.
//!
//! Tracks available computing resources across farm nodes and allocates
//! them to encode jobs.

/// The type of a computational resource.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceType {
    /// CPU cores / threads.
    Cpu,
    /// NVIDIA GPU unit.
    GpuNvidia,
    /// AMD GPU unit.
    GpuAmd,
    /// System memory (RAM) in gigabytes.
    Memory,
    /// Storage in gigabytes.
    Storage,
    /// Network bandwidth in Mbps.
    Network,
}

impl ResourceType {
    /// Return a human-readable label.
    #[allow(dead_code)]
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::GpuNvidia => "GPU (NVIDIA)",
            Self::GpuAmd => "GPU (AMD)",
            Self::Memory => "Memory",
            Self::Storage => "Storage",
            Self::Network => "Network",
        }
    }
}

/// A single resource on a farm node.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Resource {
    /// Unique resource identifier.
    pub id: u64,
    /// Resource type.
    pub resource_type: ResourceType,
    /// Total capacity of this resource (units depend on type).
    pub total_capacity: f64,
    /// Currently available (unallocated) capacity.
    pub available: f64,
    /// Hostname of the node this resource belongs to.
    pub hostname: String,
}

impl Resource {
    /// Create a new fully-available resource.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(id: u64, resource_type: ResourceType, total_capacity: f64, hostname: &str) -> Self {
        Self {
            id,
            resource_type,
            total_capacity,
            available: total_capacity,
            hostname: hostname.to_string(),
        }
    }

    /// Fractional utilization in [0.0, 1.0].
    #[allow(dead_code)]
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.total_capacity == 0.0 {
            return 0.0;
        }
        let used = self.total_capacity - self.available;
        (used / self.total_capacity).clamp(0.0, 1.0)
    }

    /// Attempt to allocate `amount` of this resource.
    /// Returns `true` on success and deducts from available capacity.
    #[allow(dead_code)]
    pub fn allocate(&mut self, amount: f64) -> bool {
        if amount <= 0.0 || amount > self.available {
            return false;
        }
        self.available -= amount;
        true
    }

    /// Release `amount` back to available capacity (clamped to total).
    #[allow(dead_code)]
    pub fn release(&mut self, amount: f64) {
        self.available = (self.available + amount).min(self.total_capacity);
    }

    /// Return `true` when utilization exceeds 90 %.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_overloaded(&self) -> bool {
        self.utilization() > 0.9
    }
}

/// A pool of heterogeneous resources across the farm.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct ResourcePool {
    pub resources: Vec<Resource>,
}

impl ResourcePool {
    /// Create an empty pool.
    #[allow(dead_code)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
        }
    }

    /// Add a resource to the pool.
    #[allow(dead_code)]
    pub fn add_resource(&mut self, r: Resource) {
        self.resources.push(r);
    }

    /// Find all resources of `rtype` that have at least `amount` available.
    #[allow(dead_code)]
    #[must_use]
    pub fn find_available(&self, rtype: ResourceType, amount: f64) -> Vec<&Resource> {
        self.resources
            .iter()
            .filter(|r| r.resource_type == rtype && r.available >= amount)
            .collect()
    }

    /// Average utilization across all resources (0.0 if pool is empty).
    #[allow(dead_code)]
    pub fn total_utilization(&self) -> f64 {
        if self.resources.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.resources.iter().map(Resource::utilization).sum();
        sum / self.resources.len() as f64
    }

    /// Return the resource of `rtype` with the lowest current utilization.
    #[allow(dead_code)]
    #[must_use]
    pub fn least_loaded(&self, rtype: ResourceType) -> Option<&Resource> {
        self.resources
            .iter()
            .filter(|r| r.resource_type == rtype)
            .min_by(|a, b| {
                a.utilization()
                    .partial_cmp(&b.utilization())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

/// A request to allocate resources for a specific job.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AllocationRequest {
    /// ID of the job requesting resources.
    pub job_id: u64,
    /// Number of CPU cores needed.
    pub cpu_cores: f64,
    /// Amount of RAM needed in gigabytes.
    pub memory_gb: f64,
    /// Number of GPU units needed.
    pub gpu_count: u32,
}

impl AllocationRequest {
    /// Create a new allocation request.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(job_id: u64, cpu_cores: f64, memory_gb: f64, gpu_count: u32) -> Self {
        Self {
            job_id,
            cpu_cores,
            memory_gb,
            gpu_count,
        }
    }
}

/// Allocate the resources described by `req` from `pool`.
///
/// On success, returns the IDs of the resources that were allocated.
/// On failure, returns an error string; the pool state is unchanged.
#[allow(dead_code)]
pub fn allocate_for_job(
    pool: &mut ResourcePool,
    req: &AllocationRequest,
) -> Result<Vec<u64>, String> {
    // --- validation pass (no mutation) ---
    let cpu_idx = find_resource_idx(pool, ResourceType::Cpu, req.cpu_cores).ok_or_else(|| {
        format!(
            "Insufficient CPU: needed {:.1} cores for job {}",
            req.cpu_cores, req.job_id
        )
    })?;

    let mem_idx =
        find_resource_idx(pool, ResourceType::Memory, req.memory_gb).ok_or_else(|| {
            format!(
                "Insufficient Memory: needed {:.1} GB for job {}",
                req.memory_gb, req.job_id
            )
        })?;

    let mut gpu_indices: Vec<usize> = Vec::new();
    if req.gpu_count > 0 {
        // Try NVIDIA first, then AMD
        for _ in 0..req.gpu_count {
            if let Some(idx) =
                find_resource_idx_excluding(pool, ResourceType::GpuNvidia, 1.0, &gpu_indices)
                    .or_else(|| {
                        find_resource_idx_excluding(pool, ResourceType::GpuAmd, 1.0, &gpu_indices)
                    })
            {
                gpu_indices.push(idx);
            } else {
                return Err(format!(
                    "Insufficient GPUs: needed {} for job {}",
                    req.gpu_count, req.job_id
                ));
            }
        }
    }

    // --- mutation pass ---
    let cpu_id = pool.resources[cpu_idx].id;
    pool.resources[cpu_idx].allocate(req.cpu_cores);

    let mem_id = pool.resources[mem_idx].id;
    pool.resources[mem_idx].allocate(req.memory_gb);

    let mut allocated_ids = vec![cpu_id, mem_id];
    for idx in gpu_indices {
        let gpu_id = pool.resources[idx].id;
        pool.resources[idx].allocate(1.0);
        allocated_ids.push(gpu_id);
    }

    Ok(allocated_ids)
}

fn find_resource_idx(pool: &ResourcePool, rtype: ResourceType, amount: f64) -> Option<usize> {
    pool.resources
        .iter()
        .enumerate()
        .filter(|(_, r)| r.resource_type == rtype && r.available >= amount)
        .min_by(|(_, a), (_, b)| {
            a.utilization()
                .partial_cmp(&b.utilization())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
}

fn find_resource_idx_excluding(
    pool: &ResourcePool,
    rtype: ResourceType,
    amount: f64,
    exclude_indices: &[usize],
) -> Option<usize> {
    pool.resources
        .iter()
        .enumerate()
        .filter(|(i, r)| {
            r.resource_type == rtype && r.available >= amount && !exclude_indices.contains(i)
        })
        .min_by(|(_, a), (_, b)| {
            a.utilization()
                .partial_cmp(&b.utilization())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool() -> ResourcePool {
        let mut pool = ResourcePool::new();
        pool.add_resource(Resource::new(1, ResourceType::Cpu, 32.0, "node-1"));
        pool.add_resource(Resource::new(2, ResourceType::Memory, 128.0, "node-1"));
        pool.add_resource(Resource::new(3, ResourceType::GpuNvidia, 4.0, "node-1"));
        pool.add_resource(Resource::new(4, ResourceType::Cpu, 16.0, "node-2"));
        pool.add_resource(Resource::new(5, ResourceType::Memory, 64.0, "node-2"));
        pool
    }

    #[test]
    fn test_resource_utilization_starts_at_zero() {
        let r = Resource::new(1, ResourceType::Cpu, 32.0, "node-1");
        assert_eq!(r.utilization(), 0.0);
    }

    #[test]
    fn test_resource_allocate_success() {
        let mut r = Resource::new(1, ResourceType::Memory, 64.0, "host");
        assert!(r.allocate(32.0));
        assert!((r.available - 32.0).abs() < 1e-9);
        assert!((r.utilization() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_resource_allocate_insufficient() {
        let mut r = Resource::new(1, ResourceType::Memory, 16.0, "host");
        assert!(!r.allocate(32.0));
    }

    #[test]
    fn test_resource_release_clamps_to_total() {
        let mut r = Resource::new(1, ResourceType::Cpu, 8.0, "host");
        r.allocate(4.0);
        r.release(8.0); // release more than was taken
        assert!((r.available - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_resource_is_overloaded() {
        let mut r = Resource::new(1, ResourceType::Cpu, 10.0, "host");
        r.allocate(9.5); // 95 % used
        assert!(r.is_overloaded());
    }

    #[test]
    fn test_pool_find_available() {
        let pool = make_pool();
        let avail = pool.find_available(ResourceType::Cpu, 16.0);
        // Both CPU resources (32 and 16) have >= 16 available
        assert_eq!(avail.len(), 2);
    }

    #[test]
    fn test_pool_find_available_none() {
        let pool = make_pool();
        let avail = pool.find_available(ResourceType::Cpu, 64.0);
        assert!(avail.is_empty());
    }

    #[test]
    fn test_pool_total_utilization_empty() {
        let pool = ResourcePool::new();
        assert_eq!(pool.total_utilization(), 0.0);
    }

    #[test]
    fn test_pool_least_loaded() {
        let mut pool = ResourcePool::new();
        let mut r1 = Resource::new(1, ResourceType::Cpu, 32.0, "node-1");
        r1.allocate(28.0); // high utilization
        pool.add_resource(r1);
        pool.add_resource(Resource::new(2, ResourceType::Cpu, 32.0, "node-2"));
        let ll = pool.least_loaded(ResourceType::Cpu).unwrap();
        assert_eq!(ll.id, 2);
    }

    #[test]
    fn test_allocate_for_job_success() {
        let mut pool = make_pool();
        let req = AllocationRequest::new(42, 4.0, 16.0, 0);
        let ids = allocate_for_job(&mut pool, &req).unwrap();
        assert_eq!(ids.len(), 2); // CPU + Memory
    }

    #[test]
    fn test_allocate_for_job_insufficient_cpu() {
        let mut pool = ResourcePool::new();
        pool.add_resource(Resource::new(1, ResourceType::Cpu, 4.0, "node"));
        pool.add_resource(Resource::new(2, ResourceType::Memory, 64.0, "node"));
        let req = AllocationRequest::new(1, 8.0, 16.0, 0);
        assert!(allocate_for_job(&mut pool, &req).is_err());
    }

    #[test]
    fn test_allocate_for_job_insufficient_memory() {
        let mut pool = ResourcePool::new();
        pool.add_resource(Resource::new(1, ResourceType::Cpu, 32.0, "node"));
        pool.add_resource(Resource::new(2, ResourceType::Memory, 4.0, "node"));
        let req = AllocationRequest::new(1, 4.0, 16.0, 0);
        assert!(allocate_for_job(&mut pool, &req).is_err());
    }

    #[test]
    fn test_resource_type_labels() {
        assert_eq!(ResourceType::Cpu.label(), "CPU");
        assert_eq!(ResourceType::GpuNvidia.label(), "GPU (NVIDIA)");
        assert_eq!(ResourceType::Memory.label(), "Memory");
    }
}
