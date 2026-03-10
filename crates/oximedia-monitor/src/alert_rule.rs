//! Alert rule evaluation: conditions, rules, and rule-sets.
#![allow(dead_code)]

/// Condition that determines when an alert fires.
#[derive(Clone, Debug, PartialEq)]
pub enum AlertCondition {
    /// Fire when the value exceeds the threshold.
    Above(f64),
    /// Fire when the value is below the threshold.
    Below(f64),
    /// Fire when the value is outside the inclusive range [lo, hi].
    OutsideRange(f64, f64),
    /// Fire when the value is inside the inclusive range [lo, hi].
    InsideRange(f64, f64),
    /// Always fires (unconditional).
    Always,
}

impl AlertCondition {
    /// Evaluate the condition against `value`. Returns `true` when the alert should fire.
    #[must_use]
    pub fn evaluate(&self, value: f64) -> bool {
        match self {
            Self::Above(threshold) => value > *threshold,
            Self::Below(threshold) => value < *threshold,
            Self::OutsideRange(lo, hi) => value < *lo || value > *hi,
            Self::InsideRange(lo, hi) => value >= *lo && value <= *hi,
            Self::Always => true,
        }
    }

    /// Human-readable description of the condition.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::Above(t) => format!("value > {t}"),
            Self::Below(t) => format!("value < {t}"),
            Self::OutsideRange(lo, hi) => format!("value outside [{lo}, {hi}]"),
            Self::InsideRange(lo, hi) => format!("value inside [{lo}, {hi}]"),
            Self::Always => "always".to_string(),
        }
    }
}

/// A single named alert rule that binds a metric name to a condition.
#[derive(Clone, Debug)]
pub struct AlertRule {
    /// Unique name identifying this rule.
    pub name: String,
    /// Name of the metric this rule monitors.
    pub metric: String,
    /// Condition to evaluate.
    pub condition: AlertCondition,
    /// Whether this rule is currently enabled.
    pub enabled: bool,
}

impl AlertRule {
    /// Create a new, enabled alert rule.
    pub fn new(
        name: impl Into<String>,
        metric: impl Into<String>,
        condition: AlertCondition,
    ) -> Self {
        Self {
            name: name.into(),
            metric: metric.into(),
            condition,
            enabled: true,
        }
    }

    /// Evaluate whether `value` triggers this rule.
    /// A disabled rule never matches.
    #[must_use]
    pub fn matches(&self, value: f64) -> bool {
        self.enabled && self.condition.evaluate(value)
    }

    /// Disable this rule (it will no longer fire).
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable this rule.
    pub fn enable(&mut self) {
        self.enabled = true;
    }
}

/// A fired alert produced when a rule matches.
#[derive(Clone, Debug)]
pub struct FiredAlertRecord {
    /// Name of the rule that fired.
    pub rule_name: String,
    /// The metric name.
    pub metric: String,
    /// The value that triggered the rule.
    pub value: f64,
    /// Description of the condition that was satisfied.
    pub condition_desc: String,
}

/// A collection of alert rules evaluated together.
#[derive(Debug, Default)]
pub struct AlertRuleSet {
    rules: Vec<AlertRule>,
}

impl AlertRuleSet {
    /// Create an empty rule-set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a rule to the set.
    pub fn add_rule(&mut self, rule: AlertRule) {
        self.rules.push(rule);
    }

    /// Evaluate all rules against `(metric_name, value)`.
    /// Returns a list of `FiredAlertRecord` for every rule that matches.
    #[must_use]
    pub fn evaluate_all(&self, metric: &str, value: f64) -> Vec<FiredAlertRecord> {
        self.rules
            .iter()
            .filter(|r| r.metric == metric && r.matches(value))
            .map(|r| FiredAlertRecord {
                rule_name: r.name.clone(),
                metric: r.metric.clone(),
                value,
                condition_desc: r.condition.description(),
            })
            .collect()
    }

    /// Return the names of all active (enabled) alerts for the given metric and value.
    #[must_use]
    pub fn active_alerts(&self, metric: &str, value: f64) -> Vec<String> {
        self.evaluate_all(metric, value)
            .into_iter()
            .map(|f| f.rule_name)
            .collect()
    }

    /// Number of rules currently in the set.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Number of enabled rules.
    #[must_use]
    pub fn enabled_count(&self) -> usize {
        self.rules.iter().filter(|r| r.enabled).count()
    }

    /// Mutable access to a rule by name for modification.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut AlertRule> {
        self.rules.iter_mut().find(|r| r.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── AlertCondition ───────────────────────────────────────────────────────

    #[test]
    fn condition_above_fires_when_exceeded() {
        assert!(AlertCondition::Above(80.0).evaluate(90.0));
        assert!(!AlertCondition::Above(80.0).evaluate(80.0));
        assert!(!AlertCondition::Above(80.0).evaluate(70.0));
    }

    #[test]
    fn condition_below_fires_when_under() {
        assert!(AlertCondition::Below(10.0).evaluate(5.0));
        assert!(!AlertCondition::Below(10.0).evaluate(10.0));
        assert!(!AlertCondition::Below(10.0).evaluate(15.0));
    }

    #[test]
    fn condition_outside_range_fires_outside() {
        let c = AlertCondition::OutsideRange(20.0, 80.0);
        assert!(c.evaluate(10.0));
        assert!(c.evaluate(90.0));
        assert!(!c.evaluate(50.0));
        assert!(!c.evaluate(20.0)); // boundary is within range
    }

    #[test]
    fn condition_inside_range_fires_inside() {
        let c = AlertCondition::InsideRange(20.0, 80.0);
        assert!(c.evaluate(50.0));
        assert!(c.evaluate(20.0));
        assert!(c.evaluate(80.0));
        assert!(!c.evaluate(10.0));
        assert!(!c.evaluate(90.0));
    }

    #[test]
    fn condition_always_fires() {
        assert!(AlertCondition::Always.evaluate(-1e9));
        assert!(AlertCondition::Always.evaluate(0.0));
        assert!(AlertCondition::Always.evaluate(1e9));
    }

    #[test]
    fn condition_description_above() {
        let d = AlertCondition::Above(95.0).description();
        assert!(d.contains("95"));
    }

    // ── AlertRule ────────────────────────────────────────────────────────────

    #[test]
    fn rule_matches_when_condition_met() {
        let r = AlertRule::new("high_cpu", "cpu", AlertCondition::Above(90.0));
        assert!(r.matches(95.0));
        assert!(!r.matches(85.0));
    }

    #[test]
    fn rule_disabled_never_matches() {
        let mut r = AlertRule::new("test", "cpu", AlertCondition::Always);
        r.disable();
        assert!(!r.matches(50.0));
    }

    #[test]
    fn rule_re_enabled_matches_again() {
        let mut r = AlertRule::new("test", "cpu", AlertCondition::Always);
        r.disable();
        r.enable();
        assert!(r.matches(50.0));
    }

    #[test]
    fn rule_only_matches_its_metric_via_ruleset() {
        let mut rs = AlertRuleSet::new();
        rs.add_rule(AlertRule::new("r", "cpu", AlertCondition::Above(80.0)));
        // Different metric name — should return empty
        assert!(rs.evaluate_all("memory", 95.0).is_empty());
    }

    // ── AlertRuleSet ─────────────────────────────────────────────────────────

    #[test]
    fn ruleset_evaluate_all_returns_fired_records() {
        let mut rs = AlertRuleSet::new();
        rs.add_rule(AlertRule::new(
            "high_cpu",
            "cpu",
            AlertCondition::Above(80.0),
        ));
        let fired = rs.evaluate_all("cpu", 90.0);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].rule_name, "high_cpu");
    }

    #[test]
    fn ruleset_active_alerts_returns_names() {
        let mut rs = AlertRuleSet::new();
        rs.add_rule(AlertRule::new("r1", "fps", AlertCondition::Below(25.0)));
        let names = rs.active_alerts("fps", 20.0);
        assert!(names.contains(&"r1".to_string()));
    }

    #[test]
    fn ruleset_rule_count_and_enabled_count() {
        let mut rs = AlertRuleSet::new();
        rs.add_rule(AlertRule::new("a", "x", AlertCondition::Always));
        rs.add_rule(AlertRule::new("b", "x", AlertCondition::Always));
        rs.get_mut("b").expect("get_mut should succeed").disable();
        assert_eq!(rs.rule_count(), 2);
        assert_eq!(rs.enabled_count(), 1);
    }

    #[test]
    fn ruleset_get_mut_modifies_rule() {
        let mut rs = AlertRuleSet::new();
        rs.add_rule(AlertRule::new("r", "cpu", AlertCondition::Above(50.0)));
        rs.get_mut("r").expect("get_mut should succeed").condition = AlertCondition::Above(90.0);
        assert!(!rs.evaluate_all("cpu", 60.0).is_empty() == false);
    }

    #[test]
    fn ruleset_multiple_rules_same_metric() {
        let mut rs = AlertRuleSet::new();
        rs.add_rule(AlertRule::new("warn", "cpu", AlertCondition::Above(70.0)));
        rs.add_rule(AlertRule::new("crit", "cpu", AlertCondition::Above(90.0)));
        let fired = rs.evaluate_all("cpu", 95.0);
        assert_eq!(fired.len(), 2);
    }
}
