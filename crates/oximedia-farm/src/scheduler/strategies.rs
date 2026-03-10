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
        use rand::Rng;
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

        let selected1 = lb
            .select_worker(&workers, &task)
            .expect("select_worker should succeed");
        let selected2 = lb
            .select_worker(&workers, &task)
            .expect("select_worker should succeed");

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
        let selected = lb
            .select_worker(&workers, &task)
            .expect("select_worker should succeed");

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
        let selected = lb
            .select_worker(&workers, &task)
            .expect("select_worker should succeed");

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
        let selected = lb
            .select_worker(&workers, &task)
            .expect("select_worker should succeed");

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
        let selected = lb
            .select_worker(&workers, &task)
            .expect("select_worker should succeed");

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
            .expect("select_worker should succeed");

        // Should select one of the workers
        assert!(selected.worker_id.as_str() == "w1" || selected.worker_id.as_str() == "w2");
    }
}
