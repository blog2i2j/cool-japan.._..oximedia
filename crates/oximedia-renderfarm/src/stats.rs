// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Statistics collection and analysis.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Statistics window
#[derive(Debug, Clone)]
pub struct StatWindow {
    values: VecDeque<f64>,
    timestamps: VecDeque<DateTime<Utc>>,
    max_size: usize,
}

impl StatWindow {
    /// Create a new statistics window
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            values: VecDeque::with_capacity(max_size),
            timestamps: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Add value
    pub fn add(&mut self, value: f64) {
        if self.values.len() >= self.max_size {
            self.values.pop_front();
            self.timestamps.pop_front();
        }
        self.values.push_back(value);
        self.timestamps.push_back(Utc::now());
    }

    /// Get mean
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Get median
    #[must_use]
    pub fn median(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.values.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    }

    /// Get standard deviation
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance =
            self.values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / self.values.len() as f64;
        variance.sqrt()
    }

    /// Get minimum
    #[must_use]
    pub fn min(&self) -> f64 {
        self.values.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Get maximum
    #[must_use]
    pub fn max(&self) -> f64 {
        self.values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get percentile
    #[must_use]
    pub fn percentile(&self, p: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.values.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() - 1) as f64 * p / 100.0) as usize;
        sorted[idx]
    }
}

/// Statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsSummary {
    /// Mean value
    pub mean: f64,
    /// Median value
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
}

impl From<&StatWindow> for StatsSummary {
    fn from(window: &StatWindow) -> Self {
        Self {
            mean: window.mean(),
            median: window.median(),
            std_dev: window.std_dev(),
            min: window.min(),
            max: window.max(),
            p95: window.percentile(95.0),
            p99: window.percentile(99.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stat_window_creation() {
        let window = StatWindow::new(100);
        assert_eq!(window.values.len(), 0);
    }

    #[test]
    fn test_stat_window_add() {
        let mut window = StatWindow::new(10);
        window.add(1.0);
        window.add(2.0);
        window.add(3.0);
        assert_eq!(window.values.len(), 3);
    }

    #[test]
    fn test_stat_window_mean() {
        let mut window = StatWindow::new(10);
        window.add(1.0);
        window.add(2.0);
        window.add(3.0);
        assert_eq!(window.mean(), 2.0);
    }

    #[test]
    fn test_stat_window_median() {
        let mut window = StatWindow::new(10);
        window.add(1.0);
        window.add(2.0);
        window.add(3.0);
        window.add(4.0);
        window.add(5.0);
        assert_eq!(window.median(), 3.0);
    }

    #[test]
    fn test_stat_window_min_max() {
        let mut window = StatWindow::new(10);
        window.add(1.0);
        window.add(5.0);
        window.add(3.0);
        assert_eq!(window.min(), 1.0);
        assert_eq!(window.max(), 5.0);
    }
}
