//! Distributed load balancing.
//!
//! Implements multiple load-balancing strategies for selecting worker nodes.

/// Available load-balancing strategies.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadBalanceStrategy {
    /// Cycle through workers in order.
    RoundRobin,
    /// Prefer the worker with the fewest active connections.
    LeastConnections,
    /// Round-robin weighted by each worker's `weight` field.
    WeightedRoundRobin,
    /// Choose the worker with the lowest composite load score.
    ResourceAware,
    /// Deterministically map a key to a worker (minimises reassignment on
    /// membership changes).
    ConsistentHash,
}

/// Snapshot of a worker's current load.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct WorkerLoad {
    pub worker_id: u64,
    pub connections: u32,
    pub cpu_pct: f64,
    pub memory_pct: f64,
    pub weight: u32,
}

impl WorkerLoad {
    /// Composite load score in `[0, 1]`. Higher means more loaded.
    #[must_use]
    pub fn load_score(&self) -> f64 {
        // Simple weighted average: 50 % CPU, 30 % memory, 20 % connections
        // (connections normalised to a 0-100 scale assuming 100 max).
        let conn_pct = f64::from(self.connections).min(100.0);
        (0.5 * self.cpu_pct + 0.3 * self.memory_pct + 0.2 * conn_pct) / 100.0
    }

    /// Returns `true` when the load score exceeds 0.9 (90 %).
    #[must_use]
    pub fn is_overloaded(&self) -> bool {
        self.load_score() > 0.9
    }
}

/// Routes incoming requests to registered worker nodes.
#[allow(dead_code)]
pub struct LoadBalancer {
    pub strategy: LoadBalanceStrategy,
    pub workers: Vec<WorkerLoad>,
    pub round_robin_idx: usize,
}

impl LoadBalancer {
    /// Create a load balancer with the chosen strategy and no workers.
    #[must_use]
    pub fn new(strategy: LoadBalanceStrategy) -> Self {
        Self {
            strategy,
            workers: Vec::new(),
            round_robin_idx: 0,
        }
    }

    /// Register a worker node.
    pub fn add_worker(&mut self, w: WorkerLoad) {
        self.workers.push(w);
    }

    /// Deregister a worker by ID. Returns `true` if found and removed.
    pub fn remove_worker(&mut self, id: u64) -> bool {
        let before = self.workers.len();
        self.workers.retain(|w| w.worker_id != id);
        self.workers.len() < before
    }

    /// Select a worker according to the configured strategy.
    ///
    /// Returns `None` when no workers are registered.
    pub fn select_worker(&mut self) -> Option<u64> {
        if self.workers.is_empty() {
            return None;
        }
        match &self.strategy {
            LoadBalanceStrategy::RoundRobin => {
                let idx = self.round_robin_idx % self.workers.len();
                self.round_robin_idx = self.round_robin_idx.wrapping_add(1);
                Some(self.workers[idx].worker_id)
            }
            LoadBalanceStrategy::LeastConnections => self
                .workers
                .iter()
                .min_by_key(|w| w.connections)
                .map(|w| w.worker_id),
            LoadBalanceStrategy::WeightedRoundRobin => {
                // Select based on accumulated weight; simple implementation.
                let total_weight: u32 = self.workers.iter().map(|w| w.weight).sum();
                if total_weight == 0 {
                    return Some(self.workers[0].worker_id);
                }
                let idx = self.round_robin_idx % total_weight as usize;
                self.round_robin_idx = self.round_robin_idx.wrapping_add(1);
                let mut acc = 0usize;
                for w in &self.workers {
                    acc += w.weight as usize;
                    if idx < acc {
                        return Some(w.worker_id);
                    }
                }
                self.workers.last().map(|w| w.worker_id)
            }
            LoadBalanceStrategy::ResourceAware => self
                .workers
                .iter()
                .min_by(|a, b| {
                    a.load_score()
                        .partial_cmp(&b.load_score())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|w| w.worker_id),
            LoadBalanceStrategy::ConsistentHash => {
                // Use the current round-robin index as the routing key.
                let key = self.round_robin_idx as u64;
                self.round_robin_idx = self.round_robin_idx.wrapping_add(1);
                consistent_hash(key, &self.workers)
            }
        }
    }

    /// Update the connection count and CPU usage for a worker.
    pub fn update_load(&mut self, worker_id: u64, connections: u32, cpu_pct: f64) {
        if let Some(w) = self.workers.iter_mut().find(|w| w.worker_id == worker_id) {
            w.connections = connections;
            w.cpu_pct = cpu_pct;
        }
    }
}

/// Map an arbitrary `key` to a worker using a simple consistent-hash ring.
///
/// Returns `None` when `workers` is empty.
#[must_use]
pub fn consistent_hash(key: u64, workers: &[WorkerLoad]) -> Option<u64> {
    if workers.is_empty() {
        return None;
    }
    // Hash the key with a simple mixing function, then pick a slot.
    let mut h = key;
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    let idx = (h as usize) % workers.len();
    Some(workers[idx].worker_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: u64, conns: u32, cpu: f64, mem: f64, weight: u32) -> WorkerLoad {
        WorkerLoad {
            worker_id: id,
            connections: conns,
            cpu_pct: cpu,
            memory_pct: mem,
            weight,
        }
    }

    #[test]
    fn test_load_score_zero_load() {
        let w = worker(1, 0, 0.0, 0.0, 1);
        assert_eq!(w.load_score(), 0.0);
    }

    #[test]
    fn test_load_score_full_load() {
        let w = worker(1, 100, 100.0, 100.0, 1);
        assert!((w.load_score() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_overloaded() {
        let ok = worker(1, 10, 50.0, 50.0, 1);
        assert!(!ok.is_overloaded());
        let hot = worker(2, 100, 100.0, 100.0, 1);
        assert!(hot.is_overloaded());
    }

    #[test]
    fn test_no_workers_returns_none() {
        let mut lb = LoadBalancer::new(LoadBalanceStrategy::RoundRobin);
        assert!(lb.select_worker().is_none());
    }

    #[test]
    fn test_round_robin_cycles() {
        let mut lb = LoadBalancer::new(LoadBalanceStrategy::RoundRobin);
        lb.add_worker(worker(1, 0, 0.0, 0.0, 1));
        lb.add_worker(worker(2, 0, 0.0, 0.0, 1));
        let first = lb.select_worker().expect("worker selection should succeed");
        let second = lb.select_worker().expect("worker selection should succeed");
        let third = lb.select_worker().expect("worker selection should succeed");
        assert_ne!(first, second);
        assert_eq!(first, third);
    }

    #[test]
    fn test_least_connections() {
        let mut lb = LoadBalancer::new(LoadBalanceStrategy::LeastConnections);
        lb.add_worker(worker(1, 10, 0.0, 0.0, 1));
        lb.add_worker(worker(2, 2, 0.0, 0.0, 1));
        lb.add_worker(worker(3, 7, 0.0, 0.0, 1));
        assert_eq!(
            lb.select_worker().expect("worker selection should succeed"),
            2
        );
    }

    #[test]
    fn test_resource_aware_picks_lowest_load() {
        let mut lb = LoadBalancer::new(LoadBalanceStrategy::ResourceAware);
        lb.add_worker(worker(1, 50, 80.0, 70.0, 1)); // high load
        lb.add_worker(worker(2, 0, 5.0, 5.0, 1)); // low load
        assert_eq!(
            lb.select_worker().expect("worker selection should succeed"),
            2
        );
    }

    #[test]
    fn test_remove_worker() {
        let mut lb = LoadBalancer::new(LoadBalanceStrategy::RoundRobin);
        lb.add_worker(worker(1, 0, 0.0, 0.0, 1));
        lb.add_worker(worker(2, 0, 0.0, 0.0, 1));
        assert!(lb.remove_worker(1));
        assert!(!lb.remove_worker(99)); // not found
        assert_eq!(lb.workers.len(), 1);
        assert_eq!(lb.workers[0].worker_id, 2);
    }

    #[test]
    fn test_update_load() {
        let mut lb = LoadBalancer::new(LoadBalanceStrategy::RoundRobin);
        lb.add_worker(worker(1, 0, 0.0, 0.0, 1));
        lb.update_load(1, 42, 75.0);
        assert_eq!(lb.workers[0].connections, 42);
        assert!((lb.workers[0].cpu_pct - 75.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_weighted_round_robin() {
        let mut lb = LoadBalancer::new(LoadBalanceStrategy::WeightedRoundRobin);
        lb.add_worker(worker(1, 0, 0.0, 0.0, 3));
        lb.add_worker(worker(2, 0, 0.0, 0.0, 1));
        // Over 4 selections we should see worker 1 selected 3 times.
        let results: Vec<u64> = (0..4)
            .map(|_| lb.select_worker().expect("worker selection should succeed"))
            .collect();
        let count_1 = results.iter().filter(|&&id| id == 1).count();
        let count_2 = results.iter().filter(|&&id| id == 2).count();
        assert_eq!(count_1, 3);
        assert_eq!(count_2, 1);
    }

    #[test]
    fn test_consistent_hash_stable() {
        let workers = vec![worker(10, 0, 0.0, 0.0, 1), worker(20, 0, 0.0, 0.0, 1)];
        let r1 = consistent_hash(42, &workers);
        let r2 = consistent_hash(42, &workers);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_consistent_hash_empty() {
        assert!(consistent_hash(0, &[]).is_none());
    }

    #[test]
    fn test_consistent_hash_single_worker() {
        let workers = vec![worker(99, 0, 0.0, 0.0, 1)];
        assert_eq!(consistent_hash(12345, &workers), Some(99));
    }

    #[test]
    fn test_add_multiple_workers() {
        let mut lb = LoadBalancer::new(LoadBalanceStrategy::RoundRobin);
        for i in 1..=5 {
            lb.add_worker(worker(i, 0, 0.0, 0.0, 1));
        }
        assert_eq!(lb.workers.len(), 5);
    }
}
