//! SLA (Service Level Agreement) monitoring.
//!
//! Tracks metric targets and records violations when values fall outside
//! agreed service levels.

/// The comparison operator for an SLA target.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum SlaOperator {
    /// Metric value must be strictly less than target.
    LessThan,
    /// Metric value must be less than or equal to target.
    LessThanOrEqual,
    /// Metric value must be strictly greater than target.
    GreaterThan,
    /// Metric value must be greater than or equal to target.
    GreaterThanOrEqual,
    /// Metric value must equal target (within floating-point tolerance).
    Equal,
}

impl SlaOperator {
    /// Evaluate whether `actual` satisfies `target` under this operator.
    #[must_use]
    pub fn evaluate(&self, actual: f64, target: f64) -> bool {
        match self {
            Self::LessThan => actual < target,
            Self::LessThanOrEqual => actual <= target,
            Self::GreaterThan => actual > target,
            Self::GreaterThanOrEqual => actual >= target,
            Self::Equal => (actual - target).abs() < 1e-9,
        }
    }

    /// Return a human-readable symbol for this operator.
    #[must_use]
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::LessThan => "<",
            Self::LessThanOrEqual => "<=",
            Self::GreaterThan => ">",
            Self::GreaterThanOrEqual => ">=",
            Self::Equal => "==",
        }
    }
}

/// A single SLA target definition.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SlaTarget {
    /// Name of the metric this target applies to.
    pub metric_name: String,
    /// The value the metric must satisfy.
    pub target_value: f64,
    /// Comparison operator.
    pub operator: SlaOperator,
    /// Measurement window in milliseconds (informational).
    pub window_ms: u64,
}

impl SlaTarget {
    /// Create a new SLA target.
    #[must_use]
    pub fn new(
        metric_name: impl Into<String>,
        target_value: f64,
        operator: SlaOperator,
        window_ms: u64,
    ) -> Self {
        Self {
            metric_name: metric_name.into(),
            target_value,
            operator,
            window_ms,
        }
    }

    /// Check whether `actual` satisfies this target.
    #[must_use]
    pub fn is_satisfied(&self, actual: f64) -> bool {
        self.operator.evaluate(actual, self.target_value)
    }

    /// Human-readable description of this target.
    #[must_use]
    pub fn description(&self) -> String {
        format!(
            "{} {} {:.4} (window: {} ms)",
            self.metric_name,
            self.operator.symbol(),
            self.target_value,
            self.window_ms,
        )
    }
}

/// A recorded SLA violation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SlaViolation {
    /// The target that was violated.
    pub target: SlaTarget,
    /// The actual measured value.
    pub actual_value: f64,
    /// Timestamp when the violation was detected (ms since epoch).
    pub violated_at: u64,
    /// How long the violation lasted in milliseconds (0 if ongoing).
    pub duration_ms: u64,
}

impl SlaViolation {
    /// Create a new SLA violation record.
    #[must_use]
    pub fn new(target: SlaTarget, actual_value: f64, violated_at: u64) -> Self {
        Self {
            target,
            actual_value,
            violated_at,
            duration_ms: 0,
        }
    }

    /// Returns `true` if the violation is marked as resolved (duration > 0).
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        self.duration_ms > 0
    }

    /// Return a human-readable description of this violation.
    #[must_use]
    pub fn description(&self) -> String {
        format!(
            "SLA violation for '{}': actual={:.4} violates {} {} (at {} ms)",
            self.target.metric_name,
            self.actual_value,
            self.target.operator.symbol(),
            self.target.target_value,
            self.violated_at,
        )
    }
}

/// Manages SLA targets and records violations.
#[derive(Debug)]
#[allow(dead_code)]
pub struct SlaMonitor {
    /// Registered SLA targets.
    pub targets: Vec<SlaTarget>,
    /// All recorded violations.
    pub violations: Vec<SlaViolation>,
}

impl SlaMonitor {
    /// Create a new, empty SLA monitor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
            violations: Vec::new(),
        }
    }

    /// Register a new SLA target.
    pub fn add_target(&mut self, target: SlaTarget) {
        self.targets.push(target);
    }

    /// Check a metric value against all matching targets.
    ///
    /// Returns `Some(SlaViolation)` for the **first** matching target that is
    /// violated.  The violation is also appended to `self.violations`.
    pub fn check(&mut self, metric: &str, value: f64, timestamp: u64) -> Option<SlaViolation> {
        for target in &self.targets {
            if target.metric_name == metric && !target.is_satisfied(value) {
                let violation = SlaViolation::new(target.clone(), value, timestamp);
                self.violations.push(violation.clone());
                return Some(violation);
            }
        }
        None
    }

    /// Calculate compliance percentage over [start, end].
    ///
    /// Compliance is defined as the percentage of time during `[start, end]`
    /// where no violation was active.  A simple approximation is used: each
    /// violation contributes its `duration_ms` (minimum 1 ms) to non-compliant
    /// time.
    #[must_use]
    pub fn compliance_pct(&self, start: u64, end: u64) -> f64 {
        if end <= start {
            return 100.0;
        }
        let window = (end - start) as f64;

        let non_compliant_ms: f64 = self
            .violations
            .iter()
            .filter(|v| v.violated_at >= start && v.violated_at < end)
            .map(|v| (v.duration_ms.max(1)) as f64)
            .sum();

        let non_compliant_clamped = non_compliant_ms.min(window);
        ((window - non_compliant_clamped) / window * 100.0).clamp(0.0, 100.0)
    }

    /// Return the total number of recorded violations.
    #[must_use]
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }

    /// Return the number of registered targets.
    #[must_use]
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Clear all recorded violations.
    pub fn clear_violations(&mut self) {
        self.violations.clear();
    }

    /// Return all violations for a specific metric.
    #[must_use]
    pub fn violations_for_metric(&self, metric: &str) -> Vec<&SlaViolation> {
        self.violations
            .iter()
            .filter(|v| v.target.metric_name == metric)
            .collect()
    }
}

impl Default for SlaMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uptime_target() -> SlaTarget {
        SlaTarget::new(
            "uptime_pct",
            99.9,
            SlaOperator::GreaterThanOrEqual,
            3_600_000,
        )
    }

    fn latency_target() -> SlaTarget {
        SlaTarget::new("latency_ms", 100.0, SlaOperator::LessThan, 60_000)
    }

    // --- SlaOperator tests ---

    #[test]
    fn test_operator_less_than() {
        let op = SlaOperator::LessThan;
        assert!(op.evaluate(50.0, 100.0));
        assert!(!op.evaluate(100.0, 100.0));
    }

    #[test]
    fn test_operator_less_than_or_equal() {
        let op = SlaOperator::LessThanOrEqual;
        assert!(op.evaluate(100.0, 100.0));
        assert!(!op.evaluate(101.0, 100.0));
    }

    #[test]
    fn test_operator_greater_than() {
        let op = SlaOperator::GreaterThan;
        assert!(op.evaluate(99.95, 99.9));
        assert!(!op.evaluate(99.9, 99.9));
    }

    #[test]
    fn test_operator_greater_than_or_equal() {
        let op = SlaOperator::GreaterThanOrEqual;
        assert!(op.evaluate(99.9, 99.9));
        assert!(!op.evaluate(99.8, 99.9));
    }

    #[test]
    fn test_operator_equal() {
        let op = SlaOperator::Equal;
        assert!(op.evaluate(1.0, 1.0));
        assert!(!op.evaluate(1.0, 2.0));
    }

    // --- SlaTarget tests ---

    #[test]
    fn test_target_satisfied() {
        let t = uptime_target();
        assert!(t.is_satisfied(99.95));
        assert!(!t.is_satisfied(99.8));
    }

    #[test]
    fn test_target_description_contains_metric() {
        let t = latency_target();
        let desc = t.description();
        assert!(desc.contains("latency_ms"));
    }

    // --- SlaMonitor tests ---

    #[test]
    fn test_monitor_add_and_count_targets() {
        let mut m = SlaMonitor::new();
        m.add_target(uptime_target());
        m.add_target(latency_target());
        assert_eq!(m.target_count(), 2);
    }

    #[test]
    fn test_monitor_check_no_violation() {
        let mut m = SlaMonitor::new();
        m.add_target(uptime_target());
        let result = m.check("uptime_pct", 99.95, 1000);
        assert!(result.is_none());
        assert_eq!(m.violation_count(), 0);
    }

    #[test]
    fn test_monitor_check_violation_recorded() {
        let mut m = SlaMonitor::new();
        m.add_target(uptime_target());
        let result = m.check("uptime_pct", 98.0, 1000);
        assert!(result.is_some());
        assert_eq!(m.violation_count(), 1);
    }

    #[test]
    fn test_monitor_unknown_metric_no_violation() {
        let mut m = SlaMonitor::new();
        m.add_target(uptime_target());
        let result = m.check("unknown_metric", 0.0, 1000);
        assert!(result.is_none());
    }

    #[test]
    fn test_monitor_compliance_no_violations() {
        let m = SlaMonitor::new();
        assert!((m.compliance_pct(0, 3_600_000) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_monitor_compliance_with_violation() {
        let mut m = SlaMonitor::new();
        m.add_target(uptime_target());
        // Simulate a violation with 60-second duration in a 1-hour window.
        let result = m.check("uptime_pct", 95.0, 0);
        let mut v = result.expect("result should be valid");
        v.duration_ms = 60_000;
        m.violations.clear();
        m.violations.push(v);
        let compliance = m.compliance_pct(0, 3_600_000);
        assert!(compliance < 100.0 && compliance > 98.0);
    }

    #[test]
    fn test_monitor_compliance_equal_timestamps() {
        let m = SlaMonitor::new();
        assert!((m.compliance_pct(1000, 1000) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_monitor_clear_violations() {
        let mut m = SlaMonitor::new();
        m.add_target(latency_target());
        m.check("latency_ms", 500.0, 0);
        assert_eq!(m.violation_count(), 1);
        m.clear_violations();
        assert_eq!(m.violation_count(), 0);
    }

    #[test]
    fn test_monitor_violations_for_metric() {
        let mut m = SlaMonitor::new();
        m.add_target(uptime_target());
        m.add_target(latency_target());
        m.check("uptime_pct", 90.0, 0);
        m.check("latency_ms", 500.0, 1);
        let uptime_violations = m.violations_for_metric("uptime_pct");
        assert_eq!(uptime_violations.len(), 1);
        let latency_violations = m.violations_for_metric("latency_ms");
        assert_eq!(latency_violations.len(), 1);
    }

    #[test]
    fn test_violation_description_contains_metric() {
        let mut m = SlaMonitor::new();
        m.add_target(latency_target());
        let v = m
            .check("latency_ms", 999.0, 5000)
            .expect("check should succeed");
        let desc = v.description();
        assert!(desc.contains("latency_ms"));
    }

    #[test]
    fn test_violation_is_resolved() {
        let t = uptime_target();
        let mut v = SlaViolation::new(t, 95.0, 0);
        assert!(!v.is_resolved());
        v.duration_ms = 1000;
        assert!(v.is_resolved());
    }
}
