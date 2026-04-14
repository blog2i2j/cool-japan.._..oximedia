//! GPU kernel scheduling simulation.
//!
//! Simulates the kernel dispatch pipeline found in modern GPU compute stacks.
//! Key concepts modelled:
//!
//! * **Kernel dependency graph** – a directed acyclic graph where edges encode
//!   "must finish before" relationships between kernels.
//! * **Launch ordering** – topological ordering of the DAG that respects all
//!   dependencies, choosing lexicographic tie-breaking for determinism.
//! * **Occupancy estimation** – computes theoretical occupancy (0.0–1.0) from
//!   active warps vs the SM warp limit.
//! * **Warp utilisation** – tracks active vs stalled warps per kernel to
//!   produce a utilisation metric.
//!
//! All structures are pure-Rust, CPU-side simulations that mirror GPU scheduler
//! semantics without requiring actual GPU hardware.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors returned by kernel scheduler operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SchedulerError {
    /// A kernel with the specified ID does not exist in the graph.
    #[error("Kernel not found: {0}")]
    KernelNotFound(u32),
    /// Adding the dependency edge would introduce a cycle.
    #[error("Dependency would create a cycle between kernel {from} and kernel {to}")]
    CyclicDependency { from: u32, to: u32 },
    /// A kernel with this ID has already been registered.
    #[error("Kernel already registered: {0}")]
    DuplicateKernel(u32),
    /// The graph contains a cycle (internal invariant violation).
    #[error("Scheduler graph contains a cycle; cannot produce valid launch order")]
    CycleDetected,
    /// Requested warp count exceeds device limit.
    #[error("Requested {requested} warps exceeds SM limit of {limit}")]
    WarpLimitExceeded { requested: u32, limit: u32 },
}

// ─── KernelSpec ───────────────────────────────────────────────────────────────

/// Specification for a single compute kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelSpec {
    /// Unique kernel identifier within the scheduler.
    pub id: u32,
    /// Human-readable name (for profiling / debug output).
    pub name: String,
    /// Number of thread groups (work groups) to dispatch.
    pub work_groups: u32,
    /// Threads per work group.
    pub threads_per_group: u32,
    /// Estimated execution time in microseconds (for scheduling heuristics).
    pub estimated_us: u64,
}

impl KernelSpec {
    /// Construct a new `KernelSpec`.
    #[must_use]
    pub fn new(
        id: u32,
        name: impl Into<String>,
        work_groups: u32,
        threads_per_group: u32,
        estimated_us: u64,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            work_groups,
            threads_per_group,
            estimated_us,
        }
    }

    /// Total number of threads this kernel launches.
    #[must_use]
    pub fn total_threads(&self) -> u64 {
        u64::from(self.work_groups) * u64::from(self.threads_per_group)
    }
}

// ─── OccupancyEstimate ────────────────────────────────────────────────────────

/// Occupancy estimate for a single kernel on a given SM configuration.
#[derive(Debug, Clone)]
pub struct OccupancyEstimate {
    /// Fraction of SM warp slots that would be active (0.0 – 1.0).
    pub theoretical_occupancy: f32,
    /// Number of warps the kernel uses.
    pub active_warps: u32,
    /// Maximum warps the SM can hold concurrently.
    pub max_warps: u32,
}

impl OccupancyEstimate {
    /// Compute occupancy for `kernel` on an SM with `sm_warp_limit` warp slots.
    ///
    /// Warp count is derived from `threads_per_group / warp_size` (rounded up),
    /// multiplied by `work_groups` (capped at `sm_warp_limit`).
    ///
    /// `warp_size` is typically 32 on NVIDIA hardware; 64 on AMD.
    #[must_use]
    pub fn compute(kernel: &KernelSpec, sm_warp_limit: u32, warp_size: u32) -> Self {
        let warp_size = warp_size.max(1);
        let warps_per_group = (kernel.threads_per_group + warp_size - 1) / warp_size;
        let active_warps = (warps_per_group * kernel.work_groups).min(sm_warp_limit);
        let max_warps = sm_warp_limit.max(1);
        let theoretical_occupancy = active_warps as f32 / max_warps as f32;
        Self {
            theoretical_occupancy: theoretical_occupancy.clamp(0.0, 1.0),
            active_warps,
            max_warps,
        }
    }
}

// ─── WarpStats ────────────────────────────────────────────────────────────────

/// Per-kernel warp utilisation statistics gathered after (simulated) execution.
#[derive(Debug, Clone)]
pub struct WarpStats {
    /// Kernel identifier this record belongs to.
    pub kernel_id: u32,
    /// Number of warps actively issuing instructions during the kernel.
    pub active_warps: u32,
    /// Number of warps stalled (waiting on memory / barriers).
    pub stalled_warps: u32,
    /// Warp utilisation: `active / (active + stalled)`.
    pub utilisation: f32,
}

impl WarpStats {
    /// Build `WarpStats` from active and stalled warp counts.
    ///
    /// `utilisation` is 0.0 when both counts are zero.
    #[must_use]
    pub fn new(kernel_id: u32, active_warps: u32, stalled_warps: u32) -> Self {
        let total = active_warps + stalled_warps;
        let utilisation = if total == 0 {
            0.0
        } else {
            active_warps as f32 / total as f32
        };
        Self {
            kernel_id,
            active_warps,
            stalled_warps,
            utilisation,
        }
    }
}

// ─── KernelScheduler ──────────────────────────────────────────────────────────

/// Kernel dependency graph and launch-order scheduler.
///
/// Kernels are registered via [`add_kernel`] and dependencies added via
/// [`add_dependency`].  Once the graph is complete, [`launch_order`] returns
/// a topological ordering that satisfies all constraints.
///
/// [`add_kernel`]: KernelScheduler::add_kernel
/// [`add_dependency`]: KernelScheduler::add_dependency
/// [`launch_order`]: KernelScheduler::launch_order
pub struct KernelScheduler {
    /// All registered kernels, keyed by their ID.
    kernels: BTreeMap<u32, KernelSpec>,
    /// Adjacency list: `deps[id]` = set of kernel IDs that `id` depends on.
    /// An edge `a → b` means "kernel `a` must wait for kernel `b`".
    deps: BTreeMap<u32, BTreeSet<u32>>,
    /// Reverse adjacency: `rdeps[b]` = kernels that depend on `b`.
    rdeps: BTreeMap<u32, BTreeSet<u32>>,
}

impl KernelScheduler {
    /// Create an empty scheduler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            kernels: BTreeMap::new(),
            deps: BTreeMap::new(),
            rdeps: BTreeMap::new(),
        }
    }

    /// Register a kernel with the scheduler.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::DuplicateKernel`] if a kernel with the same ID
    /// has already been registered.
    pub fn add_kernel(&mut self, spec: KernelSpec) -> Result<(), SchedulerError> {
        if self.kernels.contains_key(&spec.id) {
            return Err(SchedulerError::DuplicateKernel(spec.id));
        }
        let id = spec.id;
        self.kernels.insert(id, spec);
        self.deps.entry(id).or_default();
        self.rdeps.entry(id).or_default();
        Ok(())
    }

    /// Declare that kernel `dependent` must not start until kernel `dependency`
    /// has finished.
    ///
    /// # Errors
    ///
    /// * [`SchedulerError::KernelNotFound`] if either ID is not registered.
    /// * [`SchedulerError::CyclicDependency`] if the edge would introduce a cycle.
    pub fn add_dependency(
        &mut self,
        dependent: u32,
        dependency: u32,
    ) -> Result<(), SchedulerError> {
        if !self.kernels.contains_key(&dependent) {
            return Err(SchedulerError::KernelNotFound(dependent));
        }
        if !self.kernels.contains_key(&dependency) {
            return Err(SchedulerError::KernelNotFound(dependency));
        }
        // Check for cycle: would `dependency` become reachable from itself
        // through `dependent`?  i.e. is `dependency` an ancestor of `dependent`
        // already (which means adding dep→dependent creates a cycle)?
        if self.is_reachable(dependency, dependent) {
            return Err(SchedulerError::CyclicDependency {
                from: dependent,
                to: dependency,
            });
        }
        self.deps.entry(dependent).or_default().insert(dependency);
        self.rdeps.entry(dependency).or_default().insert(dependent);
        Ok(())
    }

    /// Return the IDs of all direct dependencies of `kernel_id`.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::KernelNotFound`] if the ID is not registered.
    pub fn dependencies_of(&self, kernel_id: u32) -> Result<Vec<u32>, SchedulerError> {
        if !self.kernels.contains_key(&kernel_id) {
            return Err(SchedulerError::KernelNotFound(kernel_id));
        }
        let empty = BTreeSet::new();
        let set = self.deps.get(&kernel_id).unwrap_or(&empty);
        Ok(set.iter().copied().collect())
    }

    /// Compute a valid topological launch order for all registered kernels.
    ///
    /// Uses Kahn's algorithm with a min-heap (via `BTreeSet`) for deterministic
    /// output: among ready kernels, the one with the smallest ID is picked first.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::CycleDetected`] if the graph contains a cycle
    /// (which should not happen if [`add_dependency`] correctly enforces the
    /// acyclicity invariant, but is checked defensively here).
    ///
    /// [`add_dependency`]: KernelScheduler::add_dependency
    pub fn launch_order(&self) -> Result<Vec<u32>, SchedulerError> {
        // in-degree for each kernel
        let mut in_degree: BTreeMap<u32, usize> = self
            .kernels
            .keys()
            .map(|&id| (id, self.deps[&id].len()))
            .collect();

        // Seeds: kernels with no dependencies.
        let mut ready: BTreeSet<u32> = in_degree
            .iter()
            .filter_map(|(&id, &deg)| if deg == 0 { Some(id) } else { None })
            .collect();

        let mut order = Vec::with_capacity(self.kernels.len());

        while let Some(&next) = ready.iter().next() {
            ready.remove(&next);
            order.push(next);
            // Reduce in-degree of kernels that depend on `next`.
            if let Some(dependents) = self.rdeps.get(&next) {
                for &dep in dependents {
                    let deg = in_degree.entry(dep).or_insert(0);
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        ready.insert(dep);
                    }
                }
            }
        }

        if order.len() != self.kernels.len() {
            return Err(SchedulerError::CycleDetected);
        }
        Ok(order)
    }

    /// Compute occupancy for a specific kernel.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::KernelNotFound`] if the ID is not registered.
    pub fn occupancy(
        &self,
        kernel_id: u32,
        sm_warp_limit: u32,
        warp_size: u32,
    ) -> Result<OccupancyEstimate, SchedulerError> {
        let spec = self
            .kernels
            .get(&kernel_id)
            .ok_or(SchedulerError::KernelNotFound(kernel_id))?;
        Ok(OccupancyEstimate::compute(spec, sm_warp_limit, warp_size))
    }

    /// Simulate execution and return warp statistics for each kernel in launch
    /// order.
    ///
    /// The simulation model:
    /// * Active warps = `min(warps_per_group * work_groups, sm_warp_limit)`.
    /// * Stalled warps = max(0, total_warps_launched − active_warps).
    ///
    /// # Errors
    ///
    /// Returns an error if a valid launch order cannot be produced.
    pub fn simulate_warp_stats(
        &self,
        sm_warp_limit: u32,
        warp_size: u32,
    ) -> Result<Vec<WarpStats>, SchedulerError> {
        let order = self.launch_order()?;
        let warp_size = warp_size.max(1);
        order
            .iter()
            .map(|&id| {
                let spec = self
                    .kernels
                    .get(&id)
                    .ok_or(SchedulerError::KernelNotFound(id))?;
                let warps_per_group = (spec.threads_per_group + warp_size - 1) / warp_size;
                let total_warps = warps_per_group * spec.work_groups;
                let active = total_warps.min(sm_warp_limit);
                let stalled = total_warps.saturating_sub(active);
                Ok(WarpStats::new(id, active, stalled))
            })
            .collect()
    }

    /// Number of kernels registered in the scheduler.
    #[must_use]
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }

    /// Retrieve the `KernelSpec` for a given ID, if registered.
    #[must_use]
    pub fn spec(&self, kernel_id: u32) -> Option<&KernelSpec> {
        self.kernels.get(&kernel_id)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// BFS/DFS reachability: can `target` be reached from `start` following
    /// reverse-dependency edges (i.e. following "depends on" links)?
    fn is_reachable(&self, start: u32, target: u32) -> bool {
        if start == target {
            return true;
        }
        let mut visited = BTreeSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(current) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);
            if let Some(deps) = self.deps.get(&current) {
                for &d in deps {
                    if d == target {
                        return true;
                    }
                    queue.push_back(d);
                }
            }
        }
        false
    }
}

impl Default for KernelScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spec(id: u32, work_groups: u32, threads: u32) -> KernelSpec {
        KernelSpec::new(id, format!("kernel_{id}"), work_groups, threads, 100)
    }

    // ── KernelSpec ────────────────────────────────────────────────────────────

    #[test]
    fn test_kernel_spec_total_threads() {
        let spec = make_spec(1, 4, 64);
        assert_eq!(spec.total_threads(), 256);
    }

    #[test]
    fn test_kernel_spec_zero_work_groups() {
        let spec = make_spec(2, 0, 64);
        assert_eq!(spec.total_threads(), 0);
    }

    // ── OccupancyEstimate ─────────────────────────────────────────────────────

    #[test]
    fn test_occupancy_full() {
        let spec = make_spec(1, 8, 256); // 8 warps per group (256/32), 8 groups → 64 warps
        let est = OccupancyEstimate::compute(&spec, 64, 32);
        assert_eq!(est.active_warps, 64);
        assert_eq!(est.max_warps, 64);
        assert!((est.theoretical_occupancy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_occupancy_capped_at_sm_limit() {
        let spec = make_spec(1, 100, 1024); // many warps — exceeds SM limit
        let est = OccupancyEstimate::compute(&spec, 64, 32);
        assert_eq!(est.active_warps, 64);
        assert!((est.theoretical_occupancy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_occupancy_partial() {
        let spec = make_spec(1, 2, 64); // 2 warps per group, 2 groups → 4 warps
        let est = OccupancyEstimate::compute(&spec, 32, 32);
        assert_eq!(est.active_warps, 4);
        assert!((est.theoretical_occupancy - 4.0 / 32.0).abs() < 1e-6);
    }

    // ── WarpStats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_warp_stats_utilisation_all_active() {
        let ws = WarpStats::new(1, 32, 0);
        assert!((ws.utilisation - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_warp_stats_utilisation_half() {
        let ws = WarpStats::new(2, 16, 16);
        assert!((ws.utilisation - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_warp_stats_zero_warps() {
        let ws = WarpStats::new(3, 0, 0);
        assert_eq!(ws.utilisation, 0.0);
    }

    // ── KernelScheduler – add / basic queries ─────────────────────────────────

    #[test]
    fn test_add_kernel_and_count() {
        let mut sched = KernelScheduler::new();
        sched.add_kernel(make_spec(1, 4, 64)).unwrap();
        sched.add_kernel(make_spec(2, 4, 64)).unwrap();
        assert_eq!(sched.kernel_count(), 2);
    }

    #[test]
    fn test_add_duplicate_kernel_error() {
        let mut sched = KernelScheduler::new();
        sched.add_kernel(make_spec(1, 4, 64)).unwrap();
        let err = sched.add_kernel(make_spec(1, 8, 128));
        assert!(matches!(err, Err(SchedulerError::DuplicateKernel(1))));
    }

    // ── launch_order ──────────────────────────────────────────────────────────

    #[test]
    fn test_launch_order_single_kernel() {
        let mut sched = KernelScheduler::new();
        sched.add_kernel(make_spec(7, 1, 64)).unwrap();
        let order = sched.launch_order().unwrap();
        assert_eq!(order, vec![7]);
    }

    #[test]
    fn test_launch_order_linear_chain() {
        // 1 → 2 → 3  (1 must run before 2, 2 before 3)
        let mut sched = KernelScheduler::new();
        for id in [1, 2, 3] {
            sched.add_kernel(make_spec(id, 1, 64)).unwrap();
        }
        sched.add_dependency(2, 1).unwrap(); // 2 waits for 1
        sched.add_dependency(3, 2).unwrap(); // 3 waits for 2
        let order = sched.launch_order().unwrap();
        assert_eq!(order, vec![1, 2, 3]);
    }

    #[test]
    fn test_launch_order_diamond() {
        // 1 → 2, 1 → 3, 2 → 4, 3 → 4
        let mut sched = KernelScheduler::new();
        for id in [1, 2, 3, 4] {
            sched.add_kernel(make_spec(id, 1, 64)).unwrap();
        }
        sched.add_dependency(2, 1).unwrap();
        sched.add_dependency(3, 1).unwrap();
        sched.add_dependency(4, 2).unwrap();
        sched.add_dependency(4, 3).unwrap();
        let order = sched.launch_order().unwrap();
        // 1 must be first, 4 must be last
        assert_eq!(order[0], 1);
        assert_eq!(order[3], 4);
        // 2 and 3 must appear between them
        assert!(order.contains(&2));
        assert!(order.contains(&3));
    }

    #[test]
    fn test_launch_order_independent_kernels_sorted_by_id() {
        let mut sched = KernelScheduler::new();
        for id in [5, 3, 1, 4, 2] {
            sched.add_kernel(make_spec(id, 1, 64)).unwrap();
        }
        let order = sched.launch_order().unwrap();
        assert_eq!(order, vec![1, 2, 3, 4, 5]);
    }

    // ── add_dependency errors ─────────────────────────────────────────────────

    #[test]
    fn test_add_dependency_unknown_dependent() {
        let mut sched = KernelScheduler::new();
        sched.add_kernel(make_spec(1, 1, 64)).unwrap();
        let err = sched.add_dependency(99, 1);
        assert!(matches!(err, Err(SchedulerError::KernelNotFound(99))));
    }

    #[test]
    fn test_add_dependency_unknown_dependency() {
        let mut sched = KernelScheduler::new();
        sched.add_kernel(make_spec(1, 1, 64)).unwrap();
        let err = sched.add_dependency(1, 99);
        assert!(matches!(err, Err(SchedulerError::KernelNotFound(99))));
    }

    #[test]
    fn test_add_dependency_cycle_detected() {
        let mut sched = KernelScheduler::new();
        sched.add_kernel(make_spec(1, 1, 64)).unwrap();
        sched.add_kernel(make_spec(2, 1, 64)).unwrap();
        sched.add_dependency(2, 1).unwrap(); // 2 waits for 1
                                             // Trying to make 1 wait for 2 would create a cycle.
        let err = sched.add_dependency(1, 2);
        assert!(matches!(err, Err(SchedulerError::CyclicDependency { .. })));
    }

    // ── occupancy via scheduler ───────────────────────────────────────────────

    #[test]
    fn test_scheduler_occupancy() {
        let mut sched = KernelScheduler::new();
        sched.add_kernel(make_spec(1, 4, 128)).unwrap(); // 4 warps/group, 4 groups → 16 warps
        let est = sched.occupancy(1, 64, 32).unwrap();
        assert_eq!(est.active_warps, 16);
    }

    #[test]
    fn test_scheduler_occupancy_unknown_kernel() {
        let sched = KernelScheduler::new();
        let err = sched.occupancy(42, 64, 32);
        assert!(matches!(err, Err(SchedulerError::KernelNotFound(42))));
    }

    // ── simulate_warp_stats ───────────────────────────────────────────────────

    #[test]
    fn test_simulate_warp_stats_basic() {
        let mut sched = KernelScheduler::new();
        sched.add_kernel(make_spec(1, 2, 64)).unwrap(); // 4 warps total
        sched.add_kernel(make_spec(2, 1, 64)).unwrap(); // 2 warps total
        sched.add_dependency(2, 1).unwrap();
        let stats = sched.simulate_warp_stats(32, 32).unwrap();
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].kernel_id, 1);
        assert_eq!(stats[1].kernel_id, 2);
    }

    #[test]
    fn test_simulate_warp_stats_overflow_clamps() {
        let mut sched = KernelScheduler::new();
        // 1000 work groups × 256 threads/group → 8000 warps; SM limit = 64
        sched.add_kernel(make_spec(1, 1000, 256)).unwrap();
        let stats = sched.simulate_warp_stats(64, 32).unwrap();
        assert_eq!(stats[0].active_warps, 64);
        assert!(stats[0].stalled_warps > 0);
        assert!(stats[0].utilisation < 1.0 || stats[0].stalled_warps == 0);
    }

    // ── dependencies_of ───────────────────────────────────────────────────────

    #[test]
    fn test_dependencies_of() {
        let mut sched = KernelScheduler::new();
        for id in [1, 2, 3] {
            sched.add_kernel(make_spec(id, 1, 64)).unwrap();
        }
        sched.add_dependency(3, 1).unwrap();
        sched.add_dependency(3, 2).unwrap();
        let mut deps = sched.dependencies_of(3).unwrap();
        deps.sort_unstable();
        assert_eq!(deps, vec![1, 2]);
    }

    #[test]
    fn test_dependencies_of_no_deps() {
        let mut sched = KernelScheduler::new();
        sched.add_kernel(make_spec(1, 1, 64)).unwrap();
        let deps = sched.dependencies_of(1).unwrap();
        assert!(deps.is_empty());
    }
}
