//! Counter-based metrics for monitoring discrete events.
//!
//! Provides monotonically-increasing counters suitable for tracking
//! frame counts, error counts, bytes processed, etc.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Classification of what a counter measures.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CounterType {
    /// Counts individual frames processed.
    FramesProcessed,
    /// Counts bytes transferred.
    BytesTransferred,
    /// Counts errors encountered.
    Errors,
    /// Counts jobs completed.
    JobsCompleted,
    /// Counts dropped packets or frames.
    Dropped,
    /// Custom counter with a descriptive label.
    Custom(String),
}

impl CounterType {
    /// Returns a human-readable label for the counter type.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::FramesProcessed => "frames_processed",
            Self::BytesTransferred => "bytes_transferred",
            Self::Errors => "errors",
            Self::JobsCompleted => "jobs_completed",
            Self::Dropped => "dropped",
            Self::Custom(s) => s.as_str(),
        }
    }
}

/// A single thread-safe monotonic counter.
#[derive(Debug)]
pub struct Counter {
    counter_type: CounterType,
    inner: Arc<AtomicU64>,
}

impl Counter {
    /// Create a new counter of the given type, starting at zero.
    #[must_use]
    pub fn new(counter_type: CounterType) -> Self {
        Self {
            counter_type,
            inner: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Increment the counter by 1.
    pub fn increment(&self) {
        self.inner.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the counter by an arbitrary amount.
    pub fn increment_by(&self, amount: u64) {
        self.inner.fetch_add(amount, Ordering::Relaxed);
    }

    /// Read the current counter value.
    #[must_use]
    pub fn value(&self) -> u64 {
        self.inner.load(Ordering::Relaxed)
    }

    /// The type of this counter.
    #[must_use]
    pub fn counter_type(&self) -> &CounterType {
        &self.counter_type
    }

    /// Returns a cloned handle sharing the same underlying atomic.
    #[must_use]
    pub fn clone_handle(&self) -> Self {
        Self {
            counter_type: self.counter_type.clone(),
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Clone for Counter {
    fn clone(&self) -> Self {
        self.clone_handle()
    }
}

/// A named set of counters grouped together.
#[derive(Debug, Default)]
pub struct CounterSet {
    counters: HashMap<String, Counter>,
}

impl CounterSet {
    /// Create a new, empty counter set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a counter under the given name, returning it for direct use.
    pub fn register(&mut self, name: impl Into<String>, counter_type: CounterType) -> &Counter {
        let name = name.into();
        self.counters
            .entry(name)
            .or_insert_with(|| Counter::new(counter_type))
    }

    /// Increment a named counter by 1. Creates it as a Custom counter if it
    /// does not exist yet.
    pub fn increment(&mut self, name: &str) {
        let counter = self
            .counters
            .entry(name.to_string())
            .or_insert_with(|| Counter::new(CounterType::Custom(name.to_string())));
        counter.increment();
    }

    /// Increment a named counter by `amount`.
    pub fn increment_by(&mut self, name: &str, amount: u64) {
        let counter = self
            .counters
            .entry(name.to_string())
            .or_insert_with(|| Counter::new(CounterType::Custom(name.to_string())));
        counter.increment_by(amount);
    }

    /// Get the current value of a named counter, or 0 if not registered.
    #[must_use]
    pub fn value(&self, name: &str) -> u64 {
        self.counters.get(name).map_or(0, Counter::value)
    }

    /// Sum of all counter values in this set.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.counters.values().map(Counter::value).sum()
    }

    /// Number of counters registered in this set.
    #[must_use]
    pub fn count(&self) -> usize {
        self.counters.len()
    }

    /// All counter names.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.counters
            .keys()
            .map(std::string::String::as_str)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_type_label() {
        assert_eq!(CounterType::FramesProcessed.label(), "frames_processed");
        assert_eq!(CounterType::BytesTransferred.label(), "bytes_transferred");
        assert_eq!(CounterType::Errors.label(), "errors");
        assert_eq!(CounterType::JobsCompleted.label(), "jobs_completed");
        assert_eq!(CounterType::Dropped.label(), "dropped");
        assert_eq!(
            CounterType::Custom("my_metric".to_string()).label(),
            "my_metric"
        );
    }

    #[test]
    fn test_counter_starts_at_zero() {
        let c = Counter::new(CounterType::Errors);
        assert_eq!(c.value(), 0);
    }

    #[test]
    fn test_counter_increment() {
        let c = Counter::new(CounterType::FramesProcessed);
        c.increment();
        c.increment();
        assert_eq!(c.value(), 2);
    }

    #[test]
    fn test_counter_increment_by() {
        let c = Counter::new(CounterType::BytesTransferred);
        c.increment_by(1024);
        c.increment_by(512);
        assert_eq!(c.value(), 1536);
    }

    #[test]
    fn test_counter_clone_handle_shared() {
        let c = Counter::new(CounterType::Dropped);
        let handle = c.clone_handle();
        c.increment_by(5);
        // Both share the same atomic
        assert_eq!(handle.value(), 5);
    }

    #[test]
    fn test_counter_type_accessor() {
        let c = Counter::new(CounterType::JobsCompleted);
        assert_eq!(*c.counter_type(), CounterType::JobsCompleted);
    }

    #[test]
    fn test_counter_set_new_empty() {
        let set = CounterSet::new();
        assert_eq!(set.count(), 0);
        assert_eq!(set.total(), 0);
    }

    #[test]
    fn test_counter_set_increment_creates_counter() {
        let mut set = CounterSet::new();
        set.increment("frames");
        set.increment("frames");
        assert_eq!(set.value("frames"), 2);
        assert_eq!(set.count(), 1);
    }

    #[test]
    fn test_counter_set_increment_by() {
        let mut set = CounterSet::new();
        set.increment_by("bytes", 4096);
        set.increment_by("bytes", 1024);
        assert_eq!(set.value("bytes"), 5120);
    }

    #[test]
    fn test_counter_set_value_missing() {
        let set = CounterSet::new();
        assert_eq!(set.value("nonexistent"), 0);
    }

    #[test]
    fn test_counter_set_total() {
        let mut set = CounterSet::new();
        set.increment_by("a", 10);
        set.increment_by("b", 20);
        set.increment_by("c", 30);
        assert_eq!(set.total(), 60);
    }

    #[test]
    fn test_counter_set_register() {
        let mut set = CounterSet::new();
        set.register("errors", CounterType::Errors);
        set.increment("errors");
        assert_eq!(set.value("errors"), 1);
    }

    #[test]
    fn test_counter_set_names() {
        let mut set = CounterSet::new();
        set.increment("alpha");
        set.increment("beta");
        let mut names = set.names();
        names.sort_unstable();
        assert_eq!(names, vec!["alpha", "beta"]);
    }
}
