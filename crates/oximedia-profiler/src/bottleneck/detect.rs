//! Bottleneck detection.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// A detected performance bottleneck.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Description of the bottleneck.
    pub description: String,

    /// Location (function/module).
    pub location: String,

    /// Time impact.
    pub time_impact: Duration,

    /// Impact percentage.
    pub impact_percentage: f64,

    /// Severity (0.0-1.0).
    pub severity: f64,

    /// Suggested optimization.
    pub suggestion: Option<String>,
}

impl Bottleneck {
    /// Create a new bottleneck.
    pub fn new(description: String, location: String, time_impact: Duration) -> Self {
        Self {
            description,
            location,
            time_impact,
            impact_percentage: 0.0,
            severity: 0.0,
            suggestion: None,
        }
    }

    /// Set the impact percentage and severity.
    pub fn with_impact(mut self, percentage: f64) -> Self {
        self.impact_percentage = percentage;
        self.severity = (percentage / 100.0).min(1.0);
        self
    }

    /// Add an optimization suggestion.
    pub fn with_suggestion(mut self, suggestion: String) -> Self {
        self.suggestion = Some(suggestion);
        self
    }

    /// Check if this is a critical bottleneck.
    pub fn is_critical(&self) -> bool {
        self.severity > 0.7
    }

    /// Check if this is a significant bottleneck.
    pub fn is_significant(&self) -> bool {
        self.severity > 0.4
    }
}

/// Bottleneck detector.
#[derive(Debug)]
pub struct BottleneckDetector {
    threshold: f64,
    bottlenecks: Vec<Bottleneck>,
}

impl BottleneckDetector {
    /// Create a new bottleneck detector.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            bottlenecks: Vec::new(),
        }
    }

    /// Add a potential bottleneck.
    pub fn add_bottleneck(&mut self, bottleneck: Bottleneck) {
        if bottleneck.impact_percentage >= self.threshold {
            self.bottlenecks.push(bottleneck);
        }
    }

    /// Detect bottlenecks from timing data.
    pub fn detect(&mut self, timings: &[(String, Duration)], total_time: Duration) {
        self.bottlenecks.clear();

        for (location, time) in timings {
            let percentage = if total_time.as_secs_f64() > 0.0 {
                (time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
            } else {
                0.0
            };

            if percentage >= self.threshold {
                let bottleneck = Bottleneck::new(
                    format!("High execution time in {}", location),
                    location.clone(),
                    *time,
                )
                .with_impact(percentage);

                self.bottlenecks.push(bottleneck);
            }
        }

        self.bottlenecks
            .sort_by(|a, b| b.severity.total_cmp(&a.severity));
    }

    /// Get detected bottlenecks.
    pub fn bottlenecks(&self) -> &[Bottleneck] {
        &self.bottlenecks
    }

    /// Get critical bottlenecks.
    pub fn critical_bottlenecks(&self) -> Vec<&Bottleneck> {
        self.bottlenecks
            .iter()
            .filter(|b| b.is_critical())
            .collect()
    }

    /// Clear all bottlenecks.
    pub fn clear(&mut self) {
        self.bottlenecks.clear();
    }
}

/// Configuration for fine-tuning bottleneck detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckTuning {
    /// Minimum percentage of total time for a function to be considered
    /// a bottleneck (default 5.0).
    pub min_impact_percent: f64,
    /// Number of standard deviations above the mean time-share a function
    /// must exceed to be flagged (default 1.5).
    pub std_dev_factor: f64,
    /// Weight given to time-share vs. call-frequency when scoring severity
    /// (0.0 = frequency only, 1.0 = time-share only; default 0.7).
    pub time_weight: f64,
    /// Maximum number of bottlenecks to report (0 = unlimited).
    pub max_results: usize,
    /// Whether to merge bottlenecks from the same module path prefix.
    pub merge_by_module: bool,
}

impl Default for BottleneckTuning {
    fn default() -> Self {
        Self {
            min_impact_percent: 5.0,
            std_dev_factor: 1.5,
            time_weight: 0.7,
            max_results: 0,
            merge_by_module: false,
        }
    }
}

impl BottleneckDetector {
    /// Detect bottlenecks using tuning parameters for finer control.
    ///
    /// In addition to the basic percentage threshold, this method uses
    /// statistical outlier detection: a function is flagged if its
    /// time-share exceeds `mean + tuning.std_dev_factor * std_dev` of the
    /// distribution of time-shares.
    pub fn detect_tuned(
        &mut self,
        timings: &[(String, Duration)],
        total_time: Duration,
        tuning: &BottleneckTuning,
    ) {
        self.bottlenecks.clear();

        if total_time.as_secs_f64() <= 0.0 || timings.is_empty() {
            return;
        }

        let total_secs = total_time.as_secs_f64();

        // Compute percentage shares
        let percentages: Vec<f64> = timings
            .iter()
            .map(|(_, t)| (t.as_secs_f64() / total_secs) * 100.0)
            .collect();

        // Statistical outlier detection
        let n = percentages.len() as f64;
        let mean_pct = percentages.iter().sum::<f64>() / n;
        let variance = percentages
            .iter()
            .map(|p| (p - mean_pct).powi(2))
            .sum::<f64>()
            / n;
        let std_dev_pct = variance.sqrt();
        let outlier_threshold = mean_pct + tuning.std_dev_factor * std_dev_pct;

        for (i, (location, time)) in timings.iter().enumerate() {
            let pct = percentages[i];

            // Must exceed either the minimum impact threshold or the
            // statistical outlier threshold.
            if pct < tuning.min_impact_percent && pct < outlier_threshold {
                continue;
            }

            // Compute severity: blend of time-share and relative frequency
            let time_severity = (pct / 100.0).min(1.0);
            let freq_severity = if std_dev_pct > 0.0 {
                ((pct - mean_pct) / (3.0 * std_dev_pct)).clamp(0.0, 1.0)
            } else {
                time_severity
            };
            let severity =
                tuning.time_weight * time_severity + (1.0 - tuning.time_weight) * freq_severity;

            let mut bottleneck = Bottleneck::new(
                format!("High execution time in {}", location),
                location.clone(),
                *time,
            );
            bottleneck.impact_percentage = pct;
            bottleneck.severity = severity.clamp(0.0, 1.0);

            self.bottlenecks.push(bottleneck);
        }

        // Merge by module if requested
        if tuning.merge_by_module {
            self.merge_by_module();
        }

        // Sort by severity descending
        self.bottlenecks
            .sort_by(|a, b| b.severity.total_cmp(&a.severity));

        // Limit results
        if tuning.max_results > 0 && self.bottlenecks.len() > tuning.max_results {
            self.bottlenecks.truncate(tuning.max_results);
        }
    }

    /// Merge bottlenecks whose location shares the same module prefix
    /// (everything before the last `::` separator).
    fn merge_by_module(&mut self) {
        use std::collections::HashMap;
        let mut modules: HashMap<String, (Duration, f64, f64)> = HashMap::new();

        for b in &self.bottlenecks {
            let module = b
                .location
                .rfind("::")
                .map_or_else(|| b.location.clone(), |pos| b.location[..pos].to_string());

            let entry = modules.entry(module).or_insert((Duration::ZERO, 0.0, 0.0));
            entry.0 += b.time_impact;
            entry.1 += b.impact_percentage;
            if b.severity > entry.2 {
                entry.2 = b.severity;
            }
        }

        self.bottlenecks = modules
            .into_iter()
            .map(|(module, (time, pct, sev))| {
                let mut b = Bottleneck::new(
                    format!("Aggregated bottleneck in module {}", module),
                    module,
                    time,
                );
                b.impact_percentage = pct;
                b.severity = sev;
                b
            })
            .collect();
    }
}

impl Default for BottleneckDetector {
    fn default() -> Self {
        Self::new(5.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bottleneck() {
        let bottleneck = Bottleneck::new(
            "Test bottleneck".to_string(),
            "test_function".to_string(),
            Duration::from_secs(1),
        )
        .with_impact(75.0);

        assert!(bottleneck.is_critical());
        assert!(bottleneck.is_significant());
        assert_eq!(bottleneck.impact_percentage, 75.0);
    }

    #[test]
    fn test_bottleneck_detector() {
        let mut detector = BottleneckDetector::new(10.0);

        let timings = vec![
            ("func1".to_string(), Duration::from_millis(500)),
            ("func2".to_string(), Duration::from_millis(300)),
            ("func3".to_string(), Duration::from_millis(200)),
        ];

        detector.detect(&timings, Duration::from_secs(1));

        assert!(!detector.bottlenecks().is_empty());
    }

    #[test]
    fn test_detect_tuned_basic() {
        let mut detector = BottleneckDetector::new(1.0);
        let tuning = BottleneckTuning::default();

        let timings = vec![
            ("func1".to_string(), Duration::from_millis(700)),
            ("func2".to_string(), Duration::from_millis(200)),
            ("func3".to_string(), Duration::from_millis(100)),
        ];

        detector.detect_tuned(&timings, Duration::from_secs(1), &tuning);
        // func1 at 70% should be detected
        assert!(!detector.bottlenecks().is_empty());
        assert_eq!(detector.bottlenecks()[0].location, "func1");
    }

    #[test]
    fn test_detect_tuned_max_results() {
        let mut detector = BottleneckDetector::new(1.0);
        let tuning = BottleneckTuning {
            min_impact_percent: 1.0,
            max_results: 1,
            ..Default::default()
        };

        let timings = vec![
            ("func1".to_string(), Duration::from_millis(500)),
            ("func2".to_string(), Duration::from_millis(300)),
            ("func3".to_string(), Duration::from_millis(200)),
        ];

        detector.detect_tuned(&timings, Duration::from_secs(1), &tuning);
        assert!(detector.bottlenecks().len() <= 1);
    }

    #[test]
    fn test_critical_bottlenecks() {
        let mut detector = BottleneckDetector::new(1.0);

        let timings = vec![
            ("func1".to_string(), Duration::from_millis(800)),
            ("func2".to_string(), Duration::from_millis(100)),
        ];

        detector.detect(&timings, Duration::from_secs(1));

        let critical = detector.critical_bottlenecks();
        assert_eq!(critical.len(), 1);
        assert_eq!(critical[0].location, "func1");
    }
}
