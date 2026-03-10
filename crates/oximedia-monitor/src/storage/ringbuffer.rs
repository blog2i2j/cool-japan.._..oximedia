//! Ring buffer for in-memory time series storage.

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;

/// A data point in the time series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
    /// Value.
    pub value: f64,
}

/// Ring buffer for storing recent time series data.
#[derive(Clone)]
pub struct RingBuffer {
    capacity: usize,
    data: Arc<RwLock<VecDeque<DataPoint>>>,
}

impl RingBuffer {
    /// Create a new ring buffer with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: Arc::new(RwLock::new(VecDeque::with_capacity(capacity))),
        }
    }

    /// Push a new data point into the ring buffer.
    pub fn push(&self, point: DataPoint) {
        let mut data = self.data.write();

        if data.len() >= self.capacity {
            data.pop_front();
        }

        data.push_back(point);
    }

    /// Push a value with the current timestamp.
    pub fn push_value(&self, value: f64) {
        self.push(DataPoint {
            timestamp: Utc::now(),
            value,
        });
    }

    /// Get all data points.
    #[must_use]
    pub fn get_all(&self) -> Vec<DataPoint> {
        self.data.read().iter().cloned().collect()
    }

    /// Get data points within a time range.
    #[must_use]
    pub fn get_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<DataPoint> {
        self.data
            .read()
            .iter()
            .filter(|p| p.timestamp >= start && p.timestamp <= end)
            .cloned()
            .collect()
    }

    /// Get the last N data points.
    #[must_use]
    pub fn get_last(&self, n: usize) -> Vec<DataPoint> {
        let data = self.data.read();
        let start = data.len().saturating_sub(n);
        data.iter().skip(start).cloned().collect()
    }

    /// Get the current size of the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.read().len()
    }

    /// Check if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.read().is_empty()
    }

    /// Get the capacity of the buffer.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all data points.
    pub fn clear(&self) {
        self.data.write().clear();
    }

    /// Get the oldest data point.
    #[must_use]
    pub fn oldest(&self) -> Option<DataPoint> {
        self.data.read().front().cloned()
    }

    /// Get the newest data point.
    #[must_use]
    pub fn newest(&self) -> Option<DataPoint> {
        self.data.read().back().cloned()
    }

    /// Calculate statistics over all data points.
    #[must_use]
    pub fn stats(&self) -> BufferStats {
        let data = self.data.read();

        if data.is_empty() {
            return BufferStats::default();
        }

        let values: Vec<f64> = data.iter().map(|p| p.value).collect();

        let min = values
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(0.0);

        let max = values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(0.0);

        let sum: f64 = values.iter().sum();
        let avg = sum / values.len() as f64;

        BufferStats {
            count: values.len(),
            min,
            max,
            avg,
            sum,
        }
    }
}

/// Statistics over a ring buffer.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct BufferStats {
    /// Number of data points.
    pub count: usize,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Average value.
    pub avg: f64,
    /// Sum of all values.
    pub sum: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_ring_buffer_push() {
        let buffer = RingBuffer::new(3);

        buffer.push_value(1.0);
        buffer.push_value(2.0);
        buffer.push_value(3.0);

        assert_eq!(buffer.len(), 3);

        // Should push out the oldest value
        buffer.push_value(4.0);
        assert_eq!(buffer.len(), 3);

        let data = buffer.get_all();
        assert_eq!(data[0].value, 2.0);
        assert_eq!(data[1].value, 3.0);
        assert_eq!(data[2].value, 4.0);
    }

    #[test]
    fn test_ring_buffer_get_last() {
        let buffer = RingBuffer::new(10);

        for i in 1..=10 {
            buffer.push_value(i as f64);
        }

        let last_3 = buffer.get_last(3);
        assert_eq!(last_3.len(), 3);
        assert_eq!(last_3[0].value, 8.0);
        assert_eq!(last_3[1].value, 9.0);
        assert_eq!(last_3[2].value, 10.0);
    }

    #[test]
    fn test_ring_buffer_get_range() {
        let buffer = RingBuffer::new(10);

        let now = Utc::now();

        for i in 0..10 {
            buffer.push(DataPoint {
                timestamp: now + Duration::seconds(i),
                value: i as f64,
            });
        }

        let start = now + Duration::seconds(3);
        let end = now + Duration::seconds(7);

        let range = buffer.get_range(start, end);
        assert_eq!(range.len(), 5); // 3, 4, 5, 6, 7
    }

    #[test]
    fn test_ring_buffer_stats() {
        let buffer = RingBuffer::new(10);

        for i in 1..=5 {
            buffer.push_value(i as f64);
        }

        let stats = buffer.stats();
        assert_eq!(stats.count, 5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.avg, 3.0);
        assert_eq!(stats.sum, 15.0);
    }

    #[test]
    fn test_ring_buffer_oldest_newest() {
        let buffer = RingBuffer::new(5);

        buffer.push_value(1.0);
        buffer.push_value(2.0);
        buffer.push_value(3.0);

        assert_eq!(buffer.oldest().expect("oldest should succeed").value, 1.0);
        assert_eq!(buffer.newest().expect("newest should succeed").value, 3.0);
    }

    #[test]
    fn test_ring_buffer_clear() {
        let buffer = RingBuffer::new(10);

        for i in 1..=5 {
            buffer.push_value(i as f64);
        }

        assert_eq!(buffer.len(), 5);

        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_ring_buffer_empty_stats() {
        let buffer = RingBuffer::new(10);
        let stats = buffer.stats();

        assert_eq!(stats.count, 0);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.avg, 0.0);
    }
}
