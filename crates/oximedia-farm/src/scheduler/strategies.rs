//! Load balancing and scheduling strategies

use super::{SchedulableTask, WorkerInfo};
use crate::{FarmError, Result};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Scheduling strategy enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least-loaded worker
    LeastLoaded,
    /// Capability-based routing
    CapabilityBased,
    /// Priority-based scheduling
    PriorityBased,
    /// Deadline-aware scheduling
    DeadlineAware,
    /// Random selection
    Random,
}

/// Load balancer for worker selection
pub struct LoadBalancer {
    strategy: SchedulingStrategy,
    round_robin_counter: AtomicUsize,
}

impl LoadBalancer {
    /// Create a new load balancer
    #[must_use]
    pub fn new(strategy: SchedulingStrategy) -> Self {
        Self {
            strategy,
            round_robin_counter: AtomicUsize::new(0),
        }
    }

    /// Select the best worker for a task
    pub fn select_worker<'a>(
        &self,
        workers: &[&'a WorkerInfo],
        task: &SchedulableTask,
    ) -> Result<&'a WorkerInfo> {
        if workers.is_empty() {
            return Err(FarmError::ResourceExhausted(
                "No workers available".to_string(),
            ));
        }

        match self.strategy {
            SchedulingStrategy::RoundRobin => self.round_robin(workers),
            SchedulingStrategy::LeastLoaded => self.least_loaded(workers),
            SchedulingStrategy::CapabilityBased => self.capability_based(workers, task),
            SchedulingStrategy::PriorityBased => self.priority_based(workers, task),
            SchedulingStrategy::DeadlineAware => self.deadline_aware(workers, task),
            SchedulingStrategy::Random => self.random(workers),
        }
    }

    /// Round-robin selection
    fn round_robin<'a>(&self, workers: &[&'a WorkerInfo]) -> Result<&'a WorkerInfo> {
        let index = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % workers.len();
        Ok(workers[index])
    }

    /// Least-loaded worker selection
    fn least_loaded<'a>(&self, workers: &[&'a WorkerInfo]) -> Result<&'a WorkerInfo> {
        workers
            .iter()
            .min_by(|a, b| {
                a.current_load
                    .score()
                    .partial_cmp(&b.current_load.score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .ok_or_else(|| FarmError::ResourceExhausted("No workers available".to_string()))
    }

    /// Capability-based selection (prefer workers with exact capabilities)
    fn capability_based<'a>(
        &self,
        workers: &[&'a WorkerInfo],
        task: &SchedulableTask,
    ) -> Result<&'a WorkerInfo> {
        // Score workers based on capability match
        let scored_workers: Vec<(f64, &WorkerInfo)> = workers
            .iter()
            .map(|w| {
                let mut score = 0.0;

                // Prefer GPU workers for GPU tasks
                if task.resource_requirements.gpu_required && w.capabilities.has_gpu {
                    score += 10.0;
                }

                // Prefer workers with exact codec support
                for codec in &task.required_capabilities {
                    if w.capabilities.supported_codecs.contains(codec) {
                        score += 5.0;
                    }
                }

                // Prefer less loaded workers
                score -= w.current_load.score() * 3.0;

                (score, *w)
            })
            .collect();

        scored_workers
            .into_iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, w)| w)
            .ok_or_else(|| FarmError::ResourceExhausted("No workers available".to_string()))
    }

    /// Priority-based selection (for high-priority tasks, select best worker)
    fn priority_based<'a>(
        &self,
        workers: &[&'a WorkerInfo],
        task: &SchedulableTask,
    ) -> Result<&'a WorkerInfo> {
        use crate::Priority;

        match task.priority {
            Priority::Critical | Priority::High => {
                // For high-priority tasks, select the least-loaded worker
                self.least_loaded(workers)
            }
            Priority::Normal => {
                // For normal tasks, use capability-based selection
                self.capability_based(workers, task)
            }
            Priority::Low => {
                // For low-priority tasks, use round-robin to spread load
                self.round_robin(workers)
            }
        }
    }

    /// Deadline-aware selection (prefer workers that can meet deadline)
    fn deadline_aware<'a>(
        &self,
        workers: &[&'a WorkerInfo],
        task: &SchedulableTask,
    ) -> Result<&'a WorkerInfo> {
        if task.deadline.is_none() {
            // No deadline, use least-loaded strategy
            return self.least_loaded(workers);
        }

        // For tasks with deadlines, prioritize least-loaded workers
        // to minimize queue time
        self.least_loaded(workers)
    }

    /// Random selection
    fn random<'a>(&self, workers: &[&'a WorkerInfo]) -> Result<&'a WorkerInfo> {
        use rand::RngExt;
        let mut rng = rand::rng();
        let index = rng.random_range(0..workers.len());
        Ok(workers[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::{ResourceRequirements, WorkerCapabilities, WorkerLoad};
    use crate::{Priority, TaskId, WorkerId, WorkerState};
    use std::collections::HashMap;

    fn create_worker(id: &str, cpu_usage: f64, has_gpu: bool) -> WorkerInfo {
        WorkerInfo {
            worker_id: WorkerId::new(id),
            state: WorkerState::Idle,
            capabilities: WorkerCapabilities {
                cpu_cores: 8,
                memory_mb: 16384,
                supported_codecs: vec!["h264".to_string(), "h265".to_string()],
                supported_formats: vec!["mp4".to_string()],
                has_gpu,
                gpu_count: if has_gpu { 1 } else { 0 },
                tags: HashMap::new(),
            },
            current_load: WorkerLoad {
                active_tasks: 0,
                cpu_usage,
                memory_usage: cpu_usage,
                disk_usage: cpu_usage,
            },
            last_heartbeat: chrono::Utc::now(),
        }
    }

    fn create_task(priority: Priority, gpu_required: bool) -> SchedulableTask {
        SchedulableTask {
            task_id: TaskId::new(),
            priority,
            required_capabilities: vec!["h264".to_string()],
            deadline: None,
            resource_requirements: ResourceRequirements {
                cpu_cores: 2,
                memory_mb: 4096,
                gpu_required,
                disk_space_mb: 1024,
            },
        }
    }

    #[test]
    fn test_round_robin() {
        let lb = LoadBalancer::new(SchedulingStrategy::RoundRobin);
        let w1 = create_worker("w1", 0.5, false);
        let w2 = create_worker("w2", 0.8, false);
        let workers = vec![&w1, &w2];

        let task = create_task(Priority::Normal, false);

        let selected1 = lb.select_worker(&workers, &task).unwrap();
        let selected2 = lb.select_worker(&workers, &task).unwrap();

        // Should alternate between workers
        assert_ne!(selected1.worker_id, selected2.worker_id);
    }

    #[test]
    fn test_least_loaded() {
        let lb = LoadBalancer::new(SchedulingStrategy::LeastLoaded);
        let w1 = create_worker("w1", 0.8, false);
        let w2 = create_worker("w2", 0.3, false);
        let workers = vec![&w1, &w2];

        let task = create_task(Priority::Normal, false);
        let selected = lb.select_worker(&workers, &task).unwrap();

        // Should select w2 (lower load)
        assert_eq!(selected.worker_id.as_str(), "w2");
    }

    #[test]
    fn test_capability_based() {
        let lb = LoadBalancer::new(SchedulingStrategy::CapabilityBased);
        let w1 = create_worker("w1", 0.5, false);
        let w2 = create_worker("w2", 0.5, true);
        let workers = vec![&w1, &w2];

        let task = create_task(Priority::Normal, true);
        let selected = lb.select_worker(&workers, &task).unwrap();

        // Should select w2 (has GPU)
        assert_eq!(selected.worker_id.as_str(), "w2");
    }

    #[test]
    fn test_priority_based_high() {
        let lb = LoadBalancer::new(SchedulingStrategy::PriorityBased);
        let w1 = create_worker("w1", 0.8, false);
        let w2 = create_worker("w2", 0.3, false);
        let workers = vec![&w1, &w2];

        let task = create_task(Priority::High, false);
        let selected = lb.select_worker(&workers, &task).unwrap();

        // High priority should select least loaded
        assert_eq!(selected.worker_id.as_str(), "w2");
    }

    #[test]
    fn test_priority_based_low() {
        let lb = LoadBalancer::new(SchedulingStrategy::PriorityBased);
        let w1 = create_worker("w1", 0.8, false);
        let w2 = create_worker("w2", 0.3, false);
        let workers = vec![&w1, &w2];

        let task = create_task(Priority::Low, false);
        let selected = lb.select_worker(&workers, &task).unwrap();

        // Low priority uses round-robin
        assert!(!selected.worker_id.as_str().is_empty());
    }

    #[test]
    fn test_empty_workers() {
        let lb = LoadBalancer::new(SchedulingStrategy::RoundRobin);
        let workers: Vec<&WorkerInfo> = vec![];

        let task = create_task(Priority::Normal, false);
        let result = lb.select_worker(&workers, &task);

        assert!(result.is_err());
    }

    #[test]
    fn test_random_selection() {
        let lb = LoadBalancer::new(SchedulingStrategy::Random);
        let w1 = create_worker("w1", 0.5, false);
        let w2 = create_worker("w2", 0.5, false);
        let workers = vec![&w1, &w2];

        let task = create_task(Priority::Normal, false);
        let selected = lb
            .select_worker(&workers, &task)
            .expect("random selection should succeed with 2 workers");

        // Should select one of the workers
        assert!(selected.worker_id.as_str() == "w1" || selected.worker_id.as_str() == "w2");
    }

    // ── LeastLoaded / PriorityBased unit tests ──────────────────────────────

    #[test]
    fn test_least_loaded_assigns_to_least_busy_worker() {
        let lb = LoadBalancer::new(SchedulingStrategy::LeastLoaded);
        let workers: Vec<WorkerInfo> = (0..10)
            .map(|i| create_worker(&format!("w{i}"), i as f64 / 10.0, false))
            .collect();
        let refs: Vec<&WorkerInfo> = workers.iter().collect();

        let task = create_task(Priority::Normal, false);
        let selected = lb
            .select_worker(&refs, &task)
            .expect("should select a worker");

        // Worker w0 has 0.0 CPU usage – the minimum
        assert_eq!(
            selected.worker_id.as_str(),
            "w0",
            "least-loaded strategy must pick the worker with minimum load score"
        );
    }

    #[test]
    fn test_priority_based_critical_picks_least_loaded() {
        let lb = LoadBalancer::new(SchedulingStrategy::PriorityBased);
        let high_load = create_worker("busy", 0.95, false);
        let low_load = create_worker("idle", 0.05, false);
        let refs = vec![&high_load, &low_load];

        let task = create_task(Priority::Critical, false);
        let selected = lb
            .select_worker(&refs, &task)
            .expect("should select a worker");
        assert_eq!(
            selected.worker_id.as_str(),
            "idle",
            "critical-priority task must go to the least-loaded worker"
        );
    }

    #[test]
    fn test_capability_based_gpu_task_selects_gpu_worker() {
        let lb = LoadBalancer::new(SchedulingStrategy::CapabilityBased);
        // Five CPU-only workers at equal load + one GPU worker at same load
        let mut workers: Vec<WorkerInfo> = (0..5)
            .map(|i| create_worker(&format!("cpu{i}"), 0.3, false))
            .collect();
        workers.push(create_worker("gpu0", 0.3, true));
        let refs: Vec<&WorkerInfo> = workers.iter().collect();

        let task = create_task(Priority::Normal, true); // GPU required
        let selected = lb
            .select_worker(&refs, &task)
            .expect("should select a worker");
        assert_eq!(
            selected.worker_id.as_str(),
            "gpu0",
            "capability-based strategy must prefer GPU worker for GPU tasks"
        );
    }

    // ── Scale tests ─────────────────────────────────────────────────────────

    /// Scale test helper: creates `n` workers with CPU loads spread from 0% to ~99%.
    fn make_workers(n: usize) -> Vec<WorkerInfo> {
        (0..n)
            .map(|i| {
                let cpu = i as f64 / n as f64;
                let has_gpu = i % 10 == 0; // every 10th worker has a GPU
                create_worker(&format!("worker-{i:04}"), cpu, has_gpu)
            })
            .collect()
    }

    /// Scale test: schedule 10 000 tasks across 100 workers using every strategy.
    ///
    /// Annotated with `#[ignore]` so it is excluded from the normal fast test
    /// suite and only runs when explicitly requested (`cargo test -- --ignored`).
    #[test]
    #[ignore = "scale test: run explicitly with `cargo test -- --ignored`"]
    fn test_all_strategies_10k_tasks_100_workers() {
        const WORKERS: usize = 100;
        const TASKS: usize = 10_000;

        let workers = make_workers(WORKERS);
        let refs: Vec<&WorkerInfo> = workers.iter().collect();

        let strategies = [
            SchedulingStrategy::RoundRobin,
            SchedulingStrategy::LeastLoaded,
            SchedulingStrategy::CapabilityBased,
            SchedulingStrategy::PriorityBased,
            SchedulingStrategy::DeadlineAware,
            SchedulingStrategy::Random,
        ];

        for strategy in strategies {
            let lb = LoadBalancer::new(strategy);
            let start = std::time::Instant::now();
            for t in 0..TASKS {
                // Cycle through priorities to exercise all strategy branches
                let priority = match t % 4 {
                    0 => Priority::Critical,
                    1 => Priority::High,
                    2 => Priority::Normal,
                    _ => Priority::Low,
                };
                let gpu_required = t % 10 == 0;
                let task = create_task(priority, gpu_required);
                lb.select_worker(&refs, &task)
                    .expect("should always select a worker with 100 workers available");
            }
            let elapsed = start.elapsed();
            assert!(
                elapsed.as_secs() < 5,
                "{strategy:?}: scheduling {TASKS} tasks across {WORKERS} workers took {elapsed:?} (> 5 s budget)"
            );
        }
    }

    /// Faster (non-ignored) smoke test: 1 000 tasks, 100 workers, all strategies,
    /// verifying correctness properties under load.
    #[test]
    fn test_all_strategies_1k_tasks_100_workers_smoke() {
        const WORKERS: usize = 100;
        const TASKS: usize = 1_000;

        let workers = make_workers(WORKERS);
        let refs: Vec<&WorkerInfo> = workers.iter().collect();

        let strategies = [
            SchedulingStrategy::RoundRobin,
            SchedulingStrategy::LeastLoaded,
            SchedulingStrategy::CapabilityBased,
            SchedulingStrategy::PriorityBased,
            SchedulingStrategy::DeadlineAware,
            SchedulingStrategy::Random,
        ];

        for strategy in strategies {
            let lb = LoadBalancer::new(strategy);
            let mut selections: Vec<String> = Vec::with_capacity(TASKS);
            for t in 0..TASKS {
                let priority = match t % 4 {
                    0 => Priority::Critical,
                    1 => Priority::High,
                    2 => Priority::Normal,
                    _ => Priority::Low,
                };
                let task = create_task(priority, t % 10 == 0);
                let w = lb
                    .select_worker(&refs, &task)
                    .expect("should always find a worker");
                selections.push(w.worker_id.to_string());
            }
            // Every selection must be a valid worker ID
            for id in &selections {
                assert!(
                    workers.iter().any(|w| w.worker_id.as_str() == id.as_str()),
                    "{strategy:?}: selected unknown worker ID '{id}'"
                );
            }
        }
    }

    /// Verify that RoundRobin distributes tasks across all workers over a full cycle.
    #[test]
    fn test_round_robin_covers_all_workers() {
        let n = 10usize;
        let lb = LoadBalancer::new(SchedulingStrategy::RoundRobin);
        let workers: Vec<WorkerInfo> = (0..n)
            .map(|i| create_worker(&format!("w{i}"), 0.5, false))
            .collect();
        let refs: Vec<&WorkerInfo> = workers.iter().collect();
        let task = create_task(Priority::Normal, false);

        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for _ in 0..n {
            let w = lb
                .select_worker(&refs, &task)
                .expect("should find a worker");
            seen.insert(w.worker_id.to_string());
        }
        assert_eq!(
            seen.len(),
            n,
            "RoundRobin should visit all {n} workers in one cycle"
        );
    }
}
