//! Metric correlation analysis: cross-metric correlation, anomaly correlation,
//! and causal hints for root-cause investigation.
//!
//! This module provides statistical tools for finding related metrics and
//! surfacing likely causal relationships when anomalies are detected.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ── Pearson correlation ───────────────────────────────────────────────────────

/// Compute the Pearson correlation coefficient between two equal-length slices.
/// Returns `None` if either slice is empty, has zero length, or has zero variance.
#[must_use]
pub fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.is_empty() {
        return None;
    }
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut cov = 0.0_f64;
    let mut var_x = 0.0_f64;
    let mut var_y = 0.0_f64;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    if denom < f64::EPSILON {
        return None;
    }
    Some((cov / denom).clamp(-1.0, 1.0))
}

// ── Cross-correlation with lag ────────────────────────────────────────────────

/// Result of a lagged cross-correlation search.
#[derive(Debug, Clone)]
pub struct LagResult {
    /// Lag (in samples) at which the peak correlation was found.
    /// Positive lag means `y` leads `x`.
    pub best_lag: i64,
    /// Pearson correlation at the best lag.
    pub best_r: f64,
    /// All evaluated (lag, r) pairs.
    pub all_lags: Vec<(i64, f64)>,
}

/// Compute cross-correlation of `x` vs `y` for lags in `[-max_lag, max_lag]`.
/// Returns the lag at which absolute correlation is maximised.
#[must_use]
pub fn cross_correlate(x: &[f64], y: &[f64], max_lag: usize) -> Option<LagResult> {
    if x.len() != y.len() || x.is_empty() {
        return None;
    }
    let n = x.len();
    let max_lag = max_lag.min(n - 1);
    let mut all_lags = Vec::with_capacity(2 * max_lag + 1);
    let mut best_lag = 0_i64;
    let mut best_r = 0.0_f64;

    for lag in -(max_lag as i64)..=(max_lag as i64) {
        let (xs, ys): (Vec<f64>, Vec<f64>) = if lag >= 0 {
            let l = lag as usize;
            (x[..n - l].to_vec(), y[l..].to_vec())
        } else {
            let l = (-lag) as usize;
            (x[l..].to_vec(), y[..n - l].to_vec())
        };
        if let Some(r) = pearson(&xs, &ys) {
            all_lags.push((lag, r));
            // Prefer the lag closest to zero when correlations are tied.
            if r.abs() > best_r.abs()
                || (r.abs() == best_r.abs() && lag.unsigned_abs() < best_lag.unsigned_abs())
            {
                best_r = r;
                best_lag = lag;
            }
        }
    }

    if all_lags.is_empty() {
        None
    } else {
        Some(LagResult {
            best_lag,
            best_r,
            all_lags,
        })
    }
}

// ── CorrelationMatrix ─────────────────────────────────────────────────────────

/// A symmetric matrix of pairwise Pearson correlations.
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    /// Ordered metric names (column/row labels).
    pub labels: Vec<String>,
    /// Flat row-major storage: `data[i * n + j]` = r(labels\[i\], labels\[j\]).
    pub data: Vec<f64>,
    /// Number of metrics.
    pub n: usize,
}

impl CorrelationMatrix {
    /// Compute the full pairwise correlation matrix from a map of name → time-series.
    #[must_use]
    pub fn compute(series: &HashMap<String, Vec<f64>>) -> Self {
        let labels: Vec<String> = {
            let mut v: Vec<String> = series.keys().cloned().collect();
            v.sort();
            v
        };
        let n = labels.len();
        let mut data = vec![0.0_f64; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
            for j in (i + 1)..n {
                let xi = &series[&labels[i]];
                let yj = &series[&labels[j]];
                let r = pearson(xi, yj).unwrap_or(0.0);
                data[i * n + j] = r;
                data[j * n + i] = r;
            }
        }
        Self { labels, data, n }
    }

    /// Get the correlation between two named metrics.
    #[must_use]
    pub fn get(&self, a: &str, b: &str) -> Option<f64> {
        let i = self.labels.iter().position(|l| l == a)?;
        let j = self.labels.iter().position(|l| l == b)?;
        Some(self.data[i * self.n + j])
    }

    /// Return pairs of metric names with |r| above `threshold`, sorted by |r| descending.
    #[must_use]
    pub fn strong_pairs(&self, threshold: f64) -> Vec<(String, String, f64)> {
        let mut out = Vec::new();
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let r = self.data[i * self.n + j];
                if r.abs() >= threshold {
                    out.push((self.labels[i].clone(), self.labels[j].clone(), r));
                }
            }
        }
        out.sort_by(|a, b| {
            b.2.abs()
                .partial_cmp(&a.2.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }
}

// ── Anomaly correlation ───────────────────────────────────────────────────────

/// An anomaly event: a metric exceeded a threshold at a particular sample index.
#[derive(Debug, Clone)]
pub struct AnomalyEvent {
    /// Metric name.
    pub metric: String,
    /// Sample index at which the anomaly was detected.
    pub index: usize,
    /// Observed value.
    pub value: f64,
    /// Threshold that was breached.
    pub threshold: f64,
}

impl AnomalyEvent {
    /// Create a new anomaly event.
    #[must_use]
    pub fn new(metric: impl Into<String>, index: usize, value: f64, threshold: f64) -> Self {
        Self {
            metric: metric.into(),
            index,
            value,
            threshold,
        }
    }
}

/// Group anomaly events that occur within `window` samples of each other.
/// Returns groups of co-incident anomalies, suggesting common root causes.
#[must_use]
pub fn correlate_anomalies(events: &[AnomalyEvent], window: usize) -> Vec<Vec<usize>> {
    if events.is_empty() {
        return Vec::new();
    }
    let mut visited = vec![false; events.len()];
    let mut groups: Vec<Vec<usize>> = Vec::new();

    for i in 0..events.len() {
        if visited[i] {
            continue;
        }
        let mut group = vec![i];
        visited[i] = true;
        for j in (i + 1)..events.len() {
            if visited[j] {
                continue;
            }
            let dist = events[i].index.abs_diff(events[j].index);
            if dist <= window {
                group.push(j);
                visited[j] = true;
            }
        }
        groups.push(group);
    }
    groups
}

// ── Causal hint ───────────────────────────────────────────────────────────────

/// Strength of a causal hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CausalStrength {
    /// Weak – may be coincidental.
    Weak,
    /// Moderate – likely related.
    Moderate,
    /// Strong – highly likely causal.
    Strong,
}

impl CausalStrength {
    /// Derive causal strength from an absolute Pearson r value.
    #[must_use]
    pub fn from_r(r_abs: f64) -> Self {
        if r_abs >= 0.8 {
            CausalStrength::Strong
        } else if r_abs >= 0.5 {
            CausalStrength::Moderate
        } else {
            CausalStrength::Weak
        }
    }
}

/// A human-readable causal hint linking two metrics.
#[derive(Debug, Clone)]
pub struct CausalHint {
    /// The metric that may be the cause.
    pub cause: String,
    /// The metric that may be the effect.
    pub effect: String,
    /// Pearson r at the best lag (positive = cause precedes effect).
    pub correlation: f64,
    /// Lag in samples (positive means `cause` precedes `effect`).
    pub lag_samples: i64,
    /// Derived causal strength.
    pub strength: CausalStrength,
}

impl CausalHint {
    /// Compute a causal hint from two named time-series by finding the best lag.
    #[must_use]
    pub fn compute(
        cause_name: impl Into<String>,
        effect_name: impl Into<String>,
        cause: &[f64],
        effect: &[f64],
        max_lag: usize,
    ) -> Option<Self> {
        let result = cross_correlate(cause, effect, max_lag)?;
        Some(Self {
            cause: cause_name.into(),
            effect: effect_name.into(),
            correlation: result.best_r,
            lag_samples: result.best_lag,
            strength: CausalStrength::from_r(result.best_r.abs()),
        })
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
        if n < 2 {
            return vec![start];
        }
        (0..n)
            .map(|i| start + (end - start) * i as f64 / (n - 1) as f64)
            .collect()
    }

    #[test]
    fn test_pearson_perfect_positive() {
        let x = linspace(0.0, 10.0, 11);
        let r = pearson(&x, &x).expect("operation should succeed");
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let x = linspace(0.0, 10.0, 11);
        let y: Vec<f64> = x.iter().map(|&v| -v).collect();
        let r = pearson(&x, &y).expect("operation should succeed");
        assert!((r + 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pearson_zero_variance() {
        let x = vec![5.0; 10];
        let y = linspace(0.0, 9.0, 10);
        assert!(pearson(&x, &y).is_none());
    }

    #[test]
    fn test_pearson_mismatched_lengths() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0];
        assert!(pearson(&x, &y).is_none());
    }

    #[test]
    fn test_pearson_empty() {
        assert!(pearson(&[], &[]).is_none());
    }

    #[test]
    fn test_cross_correlate_lag_zero() {
        let x = linspace(0.0, 10.0, 20);
        let result = cross_correlate(&x, &x, 5).expect("operation should succeed");
        assert_eq!(result.best_lag, 0);
        assert!((result.best_r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cross_correlate_known_lag() {
        // y is x shifted right by 2 samples.
        let base: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let x = base[0..25].to_vec();
        let y = base[2..27].to_vec();
        let result = cross_correlate(&x, &y, 5).expect("operation should succeed");
        assert!(result.best_r > 0.99);
    }

    #[test]
    fn test_correlation_matrix_self_correlation() {
        let mut series = HashMap::new();
        let v: Vec<f64> = (0..10).map(|i| i as f64).collect();
        series.insert("a".to_string(), v.clone());
        series.insert("b".to_string(), v);
        let matrix = CorrelationMatrix::compute(&series);
        assert!((matrix.get("a", "a").expect("failed to get value") - 1.0).abs() < 1e-9);
        assert!((matrix.get("b", "b").expect("failed to get value") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_correlation_matrix_strong_pairs() {
        let mut series = HashMap::new();
        let v: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let neg: Vec<f64> = v.iter().map(|&x| -x).collect();
        series.insert("a".to_string(), v);
        series.insert("b".to_string(), neg);
        let matrix = CorrelationMatrix::compute(&series);
        let pairs = matrix.strong_pairs(0.8);
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].2 + 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_correlation_matrix_unknown_metric() {
        let series: HashMap<String, Vec<f64>> = HashMap::new();
        let matrix = CorrelationMatrix::compute(&series);
        assert!(matrix.get("a", "b").is_none());
    }

    #[test]
    fn test_correlate_anomalies_grouping() {
        let events = vec![
            AnomalyEvent::new("cpu", 10, 95.0, 90.0),
            AnomalyEvent::new("mem", 11, 88.0, 85.0),
            AnomalyEvent::new("disk", 50, 99.0, 90.0),
        ];
        let groups = correlate_anomalies(&events, 3);
        assert_eq!(groups.len(), 2); // cpu+mem in one group, disk alone.
        let first_group_size = groups[0].len();
        assert_eq!(first_group_size, 2);
    }

    #[test]
    fn test_correlate_anomalies_empty() {
        assert!(correlate_anomalies(&[], 5).is_empty());
    }

    #[test]
    fn test_causal_strength_from_r() {
        assert_eq!(CausalStrength::from_r(0.9), CausalStrength::Strong);
        assert_eq!(CausalStrength::from_r(0.6), CausalStrength::Moderate);
        assert_eq!(CausalStrength::from_r(0.3), CausalStrength::Weak);
    }

    #[test]
    fn test_causal_hint_compute() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let hint =
            CausalHint::compute("cpu", "latency", &x, &x, 3).expect("operation should succeed");
        assert_eq!(hint.strength, CausalStrength::Strong);
        assert!((hint.correlation - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_causal_strength_ordering() {
        assert!(CausalStrength::Strong > CausalStrength::Moderate);
        assert!(CausalStrength::Moderate > CausalStrength::Weak);
    }
}
