// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Load balancing for task distribution.

use crate::worker::{Worker, WorkerId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin
    RoundRobin,
    /// Least loaded
    LeastLoaded,
    /// Performance-based
    PerformanceBased,
    /// Geographic proximity
    Geographic,
}

/// Load balancer
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    round_robin_index: usize,
    worker_loads: HashMap<WorkerId, f64>,
}

impl LoadBalancer {
    /// Create a new load balancer
    #[must_use]
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            round_robin_index: 0,
            worker_loads: HashMap::new(),
        }
    }

    /// Select worker for task
    #[must_use]
    pub fn select_worker(&mut self, workers: &[Worker]) -> Option<WorkerId> {
        if workers.is_empty() {
            return None;
        }

        match self.strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(workers),
            LoadBalancingStrategy::LeastLoaded => self.select_least_loaded(workers),
            LoadBalancingStrategy::PerformanceBased => self.select_performance_based(workers),
            LoadBalancingStrategy::Geographic => self.select_geographic(workers),
        }
    }

    fn select_round_robin(&mut self, workers: &[Worker]) -> Option<WorkerId> {
        let worker = &workers[self.round_robin_index % workers.len()];
        self.round_robin_index += 1;
        Some(worker.id)
    }

    fn select_least_loaded(&self, workers: &[Worker]) -> Option<WorkerId> {
        workers
            .iter()
            .min_by(|a, b| {
                let load_a = self.worker_loads.get(&a.id).copied().unwrap_or(0.0);
                let load_b = self.worker_loads.get(&b.id).copied().unwrap_or(0.0);
                load_a
                    .partial_cmp(&load_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|w| w.id)
    }

    fn select_performance_based(&self, workers: &[Worker]) -> Option<WorkerId> {
        workers
            .iter()
            .max_by(|a, b| {
                a.performance_score()
                    .partial_cmp(&b.performance_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|w| w.id)
    }

    fn select_geographic(&self, workers: &[Worker]) -> Option<WorkerId> {
        // Simplified: just select first worker
        workers.first().map(|w| w.id)
    }

    /// Update worker load
    pub fn update_load(&mut self, worker_id: WorkerId, load: f64) {
        self.worker_loads.insert(worker_id, load);
    }
}

impl Default for LoadBalancer {
    fn default() -> Self {
        Self::new(LoadBalancingStrategy::LeastLoaded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::WorkerRegistration;
    use std::net::{IpAddr, Ipv4Addr};

    fn create_test_worker() -> Worker {
        let registration = WorkerRegistration {
            hostname: "worker".to_string(),
            ip_address: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)),
            port: 8080,
            capabilities: Default::default(),
            location: None,
            tags: HashMap::new(),
        };
        Worker::new(registration)
    }

    #[test]
    fn test_load_balancer_creation() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        assert_eq!(lb.strategy, LoadBalancingStrategy::RoundRobin);
    }

    #[test]
    fn test_select_worker() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        let workers = vec![create_test_worker(), create_test_worker()];

        let selected = lb.select_worker(&workers);
        assert!(selected.is_some());
    }

    #[test]
    fn test_round_robin() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        let workers = vec![create_test_worker(), create_test_worker()];

        let id1 = lb.select_worker(&workers).expect("should succeed in test");
        let id2 = lb.select_worker(&workers).expect("should succeed in test");
        assert_ne!(id1, id2);
    }
}
