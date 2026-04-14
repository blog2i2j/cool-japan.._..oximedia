//! Configurable alerting rules for render farm health monitoring.
//!
//! [`AlertRule`] defines threshold conditions (queue depth, idle workers,
//! budget overrun) that are evaluated against live [`FarmMetrics`].  When a
//! condition is violated the rule emits an [`Alert`] with severity, label, and
//! the observed value.
//!
//! Rules are stateless — they carry no mutable state and can be cheaply cloned
//! and evaluated from multiple threads.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// FarmMetrics — a lightweight snapshot passed to rule evaluation
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of render farm metrics used for alert evaluation.
#[derive(Debug, Clone, Default)]
pub struct FarmMetrics {
    /// Current number of jobs waiting in the queue.
    pub queue_depth: u32,
    /// Number of workers that are currently idle (no active job assigned).
    pub idle_workers: u32,
    /// Total number of workers registered.
    pub total_workers: u32,
    /// Total cost spent so far in the billing period (USD or token units).
    pub current_cost: f64,
    /// Configured budget ceiling.
    pub budget_limit: f64,
    /// Average render time per frame in seconds over the last window.
    pub avg_frame_time_s: f64,
    /// Number of failed jobs in the last time window.
    pub recent_failures: u32,
}

impl FarmMetrics {
    /// Idle fraction in `[0.0, 1.0]`.  Returns `0.0` when no workers exist.
    #[must_use]
    pub fn idle_fraction(&self) -> f64 {
        if self.total_workers == 0 {
            return 0.0;
        }
        self.idle_workers as f64 / self.total_workers as f64
    }

    /// Budget utilisation in `[0.0, ∞)`.  Returns `0.0` when budget limit is
    /// zero.
    #[must_use]
    pub fn budget_utilisation(&self) -> f64 {
        if self.budget_limit <= 0.0 {
            return 0.0;
        }
        self.current_cost / self.budget_limit
    }
}

// ---------------------------------------------------------------------------
// AlertSeverity / Alert
// ---------------------------------------------------------------------------

/// Severity classification for a triggered alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational — no immediate action required.
    Info,
    /// Warning — potential problem building up.
    Warning,
    /// Critical — immediate operator attention required.
    Critical,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// An alert produced when a rule's condition is violated.
#[derive(Debug, Clone)]
pub struct Alert {
    /// Short machine-readable identifier (e.g. `"queue_depth_high"`).
    pub label: String,
    /// Human-readable description with threshold and observed values.
    pub message: String,
    /// Severity level.
    pub severity: AlertSeverity,
    /// The numeric value that triggered the alert.
    pub observed: f64,
    /// The configured threshold.
    pub threshold: f64,
}

// ---------------------------------------------------------------------------
// AlertCondition
// ---------------------------------------------------------------------------

/// The kind of metric checked by an [`AlertRule`].
#[derive(Debug, Clone)]
pub enum AlertCondition {
    /// Trigger when `queue_depth >= threshold`.
    QueueDepthHigh { threshold: u32 },
    /// Trigger when `idle_fraction >= threshold` (0.0–1.0).
    IdleWorkersHigh { threshold: f64 },
    /// Trigger when `budget_utilisation >= threshold` (e.g. 0.9 for 90%).
    BudgetOverrun { threshold: f64 },
    /// Trigger when `recent_failures >= threshold`.
    TooManyFailures { threshold: u32 },
    /// Trigger when `avg_frame_time_s >= threshold`.
    SlowFrameTime { threshold_s: f64 },
}

// ---------------------------------------------------------------------------
// AlertRule
// ---------------------------------------------------------------------------

/// A single configurable alert rule.
///
/// Call [`AlertRule::evaluate`] each monitoring tick; it returns `Some(Alert)`
/// when the condition is violated and `None` otherwise.
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Stable identifier for this rule.
    pub id: String,
    /// Condition that triggers the alert.
    pub condition: AlertCondition,
    /// Severity of the produced alert.
    pub severity: AlertSeverity,
}

impl AlertRule {
    /// Creates a new alert rule.
    #[must_use]
    pub fn new(id: impl Into<String>, condition: AlertCondition, severity: AlertSeverity) -> Self {
        Self {
            id: id.into(),
            condition,
            severity,
        }
    }

    /// Evaluates the rule against `metrics`.
    ///
    /// Returns `Some(Alert)` when the condition is violated, `None` otherwise.
    #[must_use]
    pub fn evaluate(&self, metrics: &FarmMetrics) -> Option<Alert> {
        match &self.condition {
            AlertCondition::QueueDepthHigh { threshold } => {
                if metrics.queue_depth >= *threshold {
                    Some(Alert {
                        label: self.id.clone(),
                        message: format!(
                            "Queue depth {} ≥ threshold {}",
                            metrics.queue_depth, threshold
                        ),
                        severity: self.severity,
                        observed: metrics.queue_depth as f64,
                        threshold: *threshold as f64,
                    })
                } else {
                    None
                }
            }

            AlertCondition::IdleWorkersHigh { threshold } => {
                let frac = metrics.idle_fraction();
                if frac >= *threshold {
                    Some(Alert {
                        label: self.id.clone(),
                        message: format!("Idle fraction {frac:.2} ≥ threshold {threshold:.2}"),
                        severity: self.severity,
                        observed: frac,
                        threshold: *threshold,
                    })
                } else {
                    None
                }
            }

            AlertCondition::BudgetOverrun { threshold } => {
                let util = metrics.budget_utilisation();
                if util >= *threshold {
                    Some(Alert {
                        label: self.id.clone(),
                        message: format!("Budget utilisation {util:.2} ≥ threshold {threshold:.2}"),
                        severity: self.severity,
                        observed: util,
                        threshold: *threshold,
                    })
                } else {
                    None
                }
            }

            AlertCondition::TooManyFailures { threshold } => {
                if metrics.recent_failures >= *threshold {
                    Some(Alert {
                        label: self.id.clone(),
                        message: format!(
                            "Recent failures {} ≥ threshold {}",
                            metrics.recent_failures, threshold
                        ),
                        severity: self.severity,
                        observed: metrics.recent_failures as f64,
                        threshold: *threshold as f64,
                    })
                } else {
                    None
                }
            }

            AlertCondition::SlowFrameTime { threshold_s } => {
                if metrics.avg_frame_time_s >= *threshold_s {
                    Some(Alert {
                        label: self.id.clone(),
                        message: format!(
                            "Avg frame time {:.2}s ≥ threshold {threshold_s:.2}s",
                            metrics.avg_frame_time_s
                        ),
                        severity: self.severity,
                        observed: metrics.avg_frame_time_s,
                        threshold: *threshold_s,
                    })
                } else {
                    None
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AlertEngine — evaluate a set of rules in one call
// ---------------------------------------------------------------------------

/// Evaluates a collection of [`AlertRule`]s against a [`FarmMetrics`] snapshot.
pub struct AlertEngine {
    rules: Vec<AlertRule>,
}

impl AlertEngine {
    /// Creates an engine with the given rules.
    #[must_use]
    pub fn new(rules: Vec<AlertRule>) -> Self {
        Self { rules }
    }

    /// Adds a rule.
    pub fn add_rule(&mut self, rule: AlertRule) {
        self.rules.push(rule);
    }

    /// Evaluates all rules and returns triggered alerts sorted by severity
    /// (Critical first).
    #[must_use]
    pub fn evaluate(&self, metrics: &FarmMetrics) -> Vec<Alert> {
        let mut alerts: Vec<Alert> = self
            .rules
            .iter()
            .filter_map(|rule| rule.evaluate(metrics))
            .collect();
        alerts.sort_by(|a, b| b.severity.cmp(&a.severity));
        alerts
    }

    /// Number of configured rules.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy_metrics() -> FarmMetrics {
        FarmMetrics {
            queue_depth: 5,
            idle_workers: 1,
            total_workers: 10,
            current_cost: 50.0,
            budget_limit: 1000.0,
            avg_frame_time_s: 2.0,
            recent_failures: 0,
        }
    }

    #[test]
    fn test_queue_depth_alert_not_triggered() {
        let rule = AlertRule::new(
            "q_high",
            AlertCondition::QueueDepthHigh { threshold: 10 },
            AlertSeverity::Warning,
        );
        assert!(rule.evaluate(&healthy_metrics()).is_none());
    }

    #[test]
    fn test_queue_depth_alert_triggered() {
        let rule = AlertRule::new(
            "q_high",
            AlertCondition::QueueDepthHigh { threshold: 3 },
            AlertSeverity::Critical,
        );
        let alert = rule.evaluate(&healthy_metrics()).expect("should trigger");
        assert_eq!(alert.severity, AlertSeverity::Critical);
        assert!((alert.observed - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_budget_overrun_triggered() {
        let rule = AlertRule::new(
            "budget",
            AlertCondition::BudgetOverrun { threshold: 0.04 },
            AlertSeverity::Warning,
        );
        // 50/1000 = 5% ≥ 4%
        let alert = rule
            .evaluate(&healthy_metrics())
            .expect("budget overrun should trigger");
        assert_eq!(alert.label, "budget");
    }

    #[test]
    fn test_idle_workers_alert() {
        let rule = AlertRule::new(
            "idle",
            AlertCondition::IdleWorkersHigh { threshold: 0.05 },
            AlertSeverity::Info,
        );
        // 1/10 = 10% ≥ 5%
        assert!(rule.evaluate(&healthy_metrics()).is_some());
    }

    #[test]
    fn test_slow_frame_time_not_triggered() {
        let rule = AlertRule::new(
            "slow",
            AlertCondition::SlowFrameTime { threshold_s: 5.0 },
            AlertSeverity::Warning,
        );
        assert!(rule.evaluate(&healthy_metrics()).is_none());
    }

    #[test]
    fn test_alert_engine_evaluates_all_rules() {
        let mut engine = AlertEngine::new(vec![
            AlertRule::new(
                "q",
                AlertCondition::QueueDepthHigh { threshold: 3 },
                AlertSeverity::Warning,
            ),
            AlertRule::new(
                "fail",
                AlertCondition::TooManyFailures { threshold: 1 },
                AlertSeverity::Critical,
            ),
        ]);
        engine.add_rule(AlertRule::new(
            "slow",
            AlertCondition::SlowFrameTime { threshold_s: 10.0 },
            AlertSeverity::Info,
        ));
        let metrics = healthy_metrics();
        let alerts = engine.evaluate(&metrics);
        // Queue depth 5 ≥ 3 triggers; failures 0 < 1 does not; frame time 2 < 10 does not
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].label, "q");
    }

    #[test]
    fn test_idle_fraction_no_workers() {
        let m = FarmMetrics {
            total_workers: 0,
            ..Default::default()
        };
        assert!((m.idle_fraction()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_budget_utilisation_no_limit() {
        let m = FarmMetrics {
            budget_limit: 0.0,
            ..Default::default()
        };
        assert!((m.budget_utilisation()).abs() < f64::EPSILON);
    }
}
