// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Weighted Round-Robin (WRR) load balancing for distributed worker selection.
//!
//! Implements the smooth weighted round-robin algorithm (used by nginx) which
//! distributes requests proportionally to configured worker weights while
//! avoiding long runs of the same backend.

use std::collections::HashMap;

/// A single entry in the WRR table.
#[derive(Debug, Clone)]
struct Entry {
    weight: i64,
    current: i64,
}

/// Smooth Weighted Round-Robin scheduler.
///
/// Each worker is assigned a static weight.  On each call to [`Self::next`]
/// the scheduler picks the worker with the highest *effective* weight
/// (current weight) and temporarily reduces that worker's current weight by
/// the sum of all weights, preventing starvation.
///
/// # Example
/// ```rust
/// use oximedia_distributed::weighted_round_robin::WeightedRoundRobin;
///
/// let mut wrr = WeightedRoundRobin::new();
/// wrr.add("worker-1", 3);
/// wrr.add("worker-2", 2);
/// wrr.add("worker-3", 1);
/// let selection = wrr.next().unwrap();
/// assert!(["worker-1", "worker-2", "worker-3"].contains(&selection.as_str()));
/// ```
#[derive(Debug, Default, Clone)]
pub struct WeightedRoundRobin {
    entries: HashMap<String, Entry>,
    total_weight: i64,
}

impl WeightedRoundRobin {
    /// Create a new empty scheduler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or update a worker with the given weight.  A weight of 0 effectively
    /// removes the worker from consideration.
    pub fn add(&mut self, id: impl Into<String>, weight: u32) {
        let id = id.into();
        let w = weight as i64;

        if let Some(entry) = self.entries.get_mut(&id) {
            self.total_weight -= entry.weight;
            entry.weight = w;
            entry.current = 0;
        } else {
            self.entries.insert(
                id,
                Entry {
                    weight: w,
                    current: 0,
                },
            );
        }
        self.total_weight += w;
    }

    /// Remove a worker.  Returns `true` if it existed.
    pub fn remove(&mut self, id: &str) -> bool {
        if let Some(entry) = self.entries.remove(id) {
            self.total_weight -= entry.weight;
            true
        } else {
            false
        }
    }

    /// Select the next worker using the smooth WRR algorithm.
    ///
    /// Returns `None` if no workers are registered or all have zero weight.
    pub fn next(&mut self) -> Option<String> {
        if self.entries.is_empty() || self.total_weight == 0 {
            return None;
        }

        // Raise each worker's current weight by its static weight.
        for entry in self.entries.values_mut() {
            entry.current += entry.weight;
        }

        // Pick the worker with the highest current weight.
        let selected = self
            .entries
            .iter()
            .max_by_key(|(_, e)| e.current)
            .map(|(k, _)| k.clone())?;

        // Reduce the selected worker's current weight by the total.
        if let Some(entry) = self.entries.get_mut(&selected) {
            entry.current -= self.total_weight;
        }

        Some(selected)
    }

    /// Return the number of registered workers.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if no workers are registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the static weight of a worker, or `None` if not registered.
    pub fn weight_of(&self, id: &str) -> Option<u32> {
        self.entries.get(id).map(|e| e.weight as u32)
    }

    /// Return a sorted list of all worker IDs.
    pub fn worker_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.entries.keys().cloned().collect();
        ids.sort();
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn three_workers() -> WeightedRoundRobin {
        let mut wrr = WeightedRoundRobin::new();
        wrr.add("a", 3);
        wrr.add("b", 2);
        wrr.add("c", 1);
        wrr
    }

    #[test]
    fn test_basic_selection_returns_a_worker() {
        let mut wrr = three_workers();
        let sel = wrr.next().unwrap();
        assert!(["a", "b", "c"].contains(&sel.as_str()));
    }

    #[test]
    fn test_empty_returns_none() {
        let mut wrr = WeightedRoundRobin::new();
        assert!(wrr.next().is_none());
    }

    #[test]
    fn test_proportion_matches_weights() {
        let mut wrr = three_workers();
        let mut counts: HashMap<String, usize> = HashMap::new();
        let n = 600;
        for _ in 0..n {
            let sel = wrr.next().unwrap();
            *counts.entry(sel).or_insert(0) += 1;
        }
        // Over 600 requests with weights 3:2:1 (total 6),
        // "a" should get 300, "b" 200, "c" 100 — allow ±1.
        assert_eq!(*counts.get("a").unwrap_or(&0), 300);
        assert_eq!(*counts.get("b").unwrap_or(&0), 200);
        assert_eq!(*counts.get("c").unwrap_or(&0), 100);
    }

    #[test]
    fn test_single_worker_always_selected() {
        let mut wrr = WeightedRoundRobin::new();
        wrr.add("solo", 5);
        for _ in 0..10 {
            assert_eq!(wrr.next().unwrap(), "solo");
        }
    }

    #[test]
    fn test_remove_worker() {
        let mut wrr = three_workers();
        assert!(wrr.remove("b"));
        assert_eq!(wrr.len(), 2);
        for _ in 0..20 {
            let sel = wrr.next().unwrap();
            assert_ne!(sel, "b");
        }
    }

    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut wrr = three_workers();
        assert!(!wrr.remove("ghost"));
    }

    #[test]
    fn test_weight_of() {
        let wrr = three_workers();
        assert_eq!(wrr.weight_of("a"), Some(3));
        assert_eq!(wrr.weight_of("ghost"), None);
    }

    #[test]
    fn test_zero_weight_excluded() {
        let mut wrr = WeightedRoundRobin::new();
        wrr.add("active", 4);
        wrr.add("zero", 0);
        for _ in 0..20 {
            assert_eq!(wrr.next().unwrap(), "active");
        }
    }

    #[test]
    fn test_update_weight() {
        let mut wrr = WeightedRoundRobin::new();
        wrr.add("a", 1);
        wrr.add("b", 1);
        // Give 'a' much higher weight
        wrr.add("a", 10);
        let mut counts: HashMap<String, usize> = HashMap::new();
        for _ in 0..110 {
            *counts.entry(wrr.next().unwrap()).or_insert(0) += 1;
        }
        assert!(*counts.get("a").unwrap_or(&0) > *counts.get("b").unwrap_or(&0));
    }

    #[test]
    fn test_is_empty_and_len() {
        let mut wrr = WeightedRoundRobin::new();
        assert!(wrr.is_empty());
        wrr.add("x", 1);
        assert_eq!(wrr.len(), 1);
        assert!(!wrr.is_empty());
    }
}
