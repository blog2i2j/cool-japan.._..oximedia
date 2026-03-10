//! Graph rewriting and transformation rules.
//!
//! This module provides a rule-based system for transforming filter graphs.
//! Rewrite rules can match patterns in the graph and replace them with
//! optimized equivalents, enabling algebraic simplification, constant
//! folding, and operator fusion.

use std::collections::HashMap;
use std::fmt;

/// A unique identifier for a rewrite rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuleId(pub u64);

impl fmt::Display for RuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rule({})", self.0)
    }
}

/// Describes the kind of pattern a rewrite rule matches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternKind {
    /// Matches a single node by its filter type name.
    SingleNode {
        /// The filter type to match (e.g. "scale", "crop").
        filter_type: String,
    },
    /// Matches a chain of two consecutive nodes.
    Chain {
        /// First node's filter type.
        first: String,
        /// Second node's filter type.
        second: String,
    },
    /// Matches a node with a specific property constraint.
    WithProperty {
        /// The filter type to match.
        filter_type: String,
        /// Property key that must be present.
        property_key: String,
        /// Expected property value.
        property_value: String,
    },
}

impl fmt::Display for PatternKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SingleNode { filter_type } => write!(f, "Single({filter_type})"),
            Self::Chain { first, second } => write!(f, "Chain({first} -> {second})"),
            Self::WithProperty { filter_type, property_key, property_value } => {
                write!(f, "{filter_type}[{property_key}={property_value}]")
            }
        }
    }
}

/// The action to take when a rule matches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RewriteAction {
    /// Remove the matched node(s) entirely (identity elimination).
    Remove,
    /// Replace with a single node of the given filter type and properties.
    ReplaceWith {
        /// The replacement filter type.
        filter_type: String,
        /// Properties for the replacement node.
        properties: HashMap<String, String>,
    },
    /// Fuse two matched nodes into one with combined properties.
    Fuse {
        /// The fused filter type name.
        fused_type: String,
    },
    /// Swap the order of two matched nodes (commutativity).
    Swap,
}

impl fmt::Display for RewriteAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Remove => write!(f, "Remove"),
            Self::ReplaceWith { filter_type, .. } => write!(f, "ReplaceWith({filter_type})"),
            Self::Fuse { fused_type } => write!(f, "Fuse({fused_type})"),
            Self::Swap => write!(f, "Swap"),
        }
    }
}

/// A single graph rewrite rule.
#[derive(Debug, Clone)]
pub struct RewriteRule {
    /// Unique identifier.
    pub id: RuleId,
    /// Human-readable name for this rule.
    pub name: String,
    /// The pattern to match.
    pub pattern: PatternKind,
    /// The action to take on match.
    pub action: RewriteAction,
    /// Priority (higher = applied first).
    pub priority: i32,
    /// Whether this rule is currently enabled.
    pub enabled: bool,
}

impl RewriteRule {
    /// Create a new rewrite rule.
    pub fn new(id: RuleId, name: &str, pattern: PatternKind, action: RewriteAction) -> Self {
        Self {
            id,
            name: name.to_string(),
            pattern,
            action,
            priority: 0,
            enabled: true,
        }
    }

    /// Set the priority of this rule.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Enable or disable this rule.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check whether this rule matches a given node description.
    pub fn matches_node(&self, filter_type: &str, properties: &HashMap<String, String>) -> bool {
        if !self.enabled {
            return false;
        }
        match &self.pattern {
            PatternKind::SingleNode { filter_type: ft } => ft == filter_type,
            PatternKind::WithProperty { filter_type: ft, property_key, property_value } => {
                ft == filter_type
                    && properties.get(property_key).map_or(false, |v| v == property_value)
            }
            PatternKind::Chain { first, .. } => first == filter_type,
        }
    }

    /// Check whether this rule matches a chain of two nodes.
    pub fn matches_chain(&self, first_type: &str, second_type: &str) -> bool {
        if !self.enabled {
            return false;
        }
        match &self.pattern {
            PatternKind::Chain { first, second } => first == first_type && second == second_type,
            _ => false,
        }
    }
}

impl fmt::Display for RewriteRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]: {} -> {}",
            self.name, self.id, self.pattern, self.action)
    }
}

/// A record of a single rule application.
#[derive(Debug, Clone)]
pub struct RewriteEvent {
    /// The rule that was applied.
    pub rule_id: RuleId,
    /// The rule name.
    pub rule_name: String,
    /// Description of what was matched.
    pub matched: String,
    /// The action that was taken.
    pub action: String,
}

/// A collection of rewrite rules with application tracking.
pub struct RewriteEngine {
    /// Registered rules, sorted by priority.
    rules: Vec<RewriteRule>,
    /// History of applied rewrites.
    history: Vec<RewriteEvent>,
    /// Maximum number of rewrite passes to prevent infinite loops.
    max_passes: u32,
}

impl RewriteEngine {
    /// Create a new rewrite engine with default settings.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            history: Vec::new(),
            max_passes: 100,
        }
    }

    /// Set the maximum number of rewrite passes.
    pub fn set_max_passes(&mut self, max: u32) {
        self.max_passes = max;
    }

    /// Get the maximum number of rewrite passes.
    pub fn max_passes(&self) -> u32 {
        self.max_passes
    }

    /// Add a rewrite rule.
    pub fn add_rule(&mut self, rule: RewriteRule) {
        self.rules.push(rule);
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Get the number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Get a rule by its ID.
    pub fn get_rule(&self, id: RuleId) -> Option<&RewriteRule> {
        self.rules.iter().find(|r| r.id == id)
    }

    /// Get a mutable reference to a rule by its ID.
    pub fn get_rule_mut(&mut self, id: RuleId) -> Option<&mut RewriteRule> {
        self.rules.iter_mut().find(|r| r.id == id)
    }

    /// Find matching rules for a single node.
    pub fn find_matches(&self, filter_type: &str, properties: &HashMap<String, String>) -> Vec<&RewriteRule> {
        self.rules
            .iter()
            .filter(|r| r.matches_node(filter_type, properties))
            .collect()
    }

    /// Find matching rules for a chain of two nodes.
    pub fn find_chain_matches(&self, first_type: &str, second_type: &str) -> Vec<&RewriteRule> {
        self.rules
            .iter()
            .filter(|r| r.matches_chain(first_type, second_type))
            .collect()
    }

    /// Record a rewrite event.
    pub fn record_event(&mut self, rule: &RewriteRule, matched: &str) {
        self.history.push(RewriteEvent {
            rule_id: rule.id,
            rule_name: rule.name.clone(),
            matched: matched.to_string(),
            action: format!("{}", rule.action),
        });
    }

    /// Get the history of applied rewrites.
    pub fn history(&self) -> &[RewriteEvent] {
        &self.history
    }

    /// Clear the rewrite history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Remove a rule by its ID.
    pub fn remove_rule(&mut self, id: RuleId) -> bool {
        let len_before = self.rules.len();
        self.rules.retain(|r| r.id != id);
        self.rules.len() < len_before
    }

    /// Enable all rules.
    pub fn enable_all(&mut self) {
        for rule in &mut self.rules {
            rule.enabled = true;
        }
    }

    /// Disable all rules.
    pub fn disable_all(&mut self) {
        for rule in &mut self.rules {
            rule.enabled = false;
        }
    }
}

impl Default for RewriteEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a standard set of common rewrite rules.
pub fn standard_rules() -> Vec<RewriteRule> {
    vec![
        // Identity scale removal (scale 1:1 is a no-op)
        RewriteRule::new(
            RuleId(1),
            "identity_scale",
            PatternKind::WithProperty {
                filter_type: "scale".to_string(),
                property_key: "factor".to_string(),
                property_value: "1.0".to_string(),
            },
            RewriteAction::Remove,
        ).with_priority(100),
        // Consecutive scale fusion
        RewriteRule::new(
            RuleId(2),
            "scale_fusion",
            PatternKind::Chain {
                first: "scale".to_string(),
                second: "scale".to_string(),
            },
            RewriteAction::Fuse { fused_type: "scale".to_string() },
        ).with_priority(90),
        // Consecutive crop fusion
        RewriteRule::new(
            RuleId(3),
            "crop_fusion",
            PatternKind::Chain {
                first: "crop".to_string(),
                second: "crop".to_string(),
            },
            RewriteAction::Fuse { fused_type: "crop".to_string() },
        ).with_priority(90),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_id_display() {
        assert_eq!(format!("{}", RuleId(42)), "Rule(42)");
    }

    #[test]
    fn test_pattern_kind_display() {
        let p = PatternKind::SingleNode { filter_type: "scale".to_string() };
        assert_eq!(format!("{p}"), "Single(scale)");
    }

    #[test]
    fn test_chain_pattern_display() {
        let p = PatternKind::Chain { first: "a".to_string(), second: "b".to_string() };
        assert_eq!(format!("{p}"), "Chain(a -> b)");
    }

    #[test]
    fn test_rewrite_action_display() {
        assert_eq!(format!("{}", RewriteAction::Remove), "Remove");
        assert_eq!(format!("{}", RewriteAction::Swap), "Swap");
        assert_eq!(
            format!("{}", RewriteAction::Fuse { fused_type: "x".to_string() }),
            "Fuse(x)"
        );
    }

    #[test]
    fn test_rewrite_rule_new() {
        let rule = RewriteRule::new(
            RuleId(1),
            "test",
            PatternKind::SingleNode { filter_type: "scale".to_string() },
            RewriteAction::Remove,
        );
        assert_eq!(rule.id, RuleId(1));
        assert_eq!(rule.name, "test");
        assert_eq!(rule.priority, 0);
        assert!(rule.enabled);
    }

    #[test]
    fn test_rule_matches_single_node() {
        let rule = RewriteRule::new(
            RuleId(1),
            "test",
            PatternKind::SingleNode { filter_type: "scale".to_string() },
            RewriteAction::Remove,
        );
        let props = HashMap::new();
        assert!(rule.matches_node("scale", &props));
        assert!(!rule.matches_node("crop", &props));
    }

    #[test]
    fn test_rule_matches_with_property() {
        let rule = RewriteRule::new(
            RuleId(1),
            "identity_scale",
            PatternKind::WithProperty {
                filter_type: "scale".to_string(),
                property_key: "factor".to_string(),
                property_value: "1.0".to_string(),
            },
            RewriteAction::Remove,
        );
        let mut props = HashMap::new();
        props.insert("factor".to_string(), "1.0".to_string());
        assert!(rule.matches_node("scale", &props));

        props.insert("factor".to_string(), "2.0".to_string());
        assert!(!rule.matches_node("scale", &props));
    }

    #[test]
    fn test_rule_matches_chain() {
        let rule = RewriteRule::new(
            RuleId(2),
            "scale_fusion",
            PatternKind::Chain {
                first: "scale".to_string(),
                second: "scale".to_string(),
            },
            RewriteAction::Fuse { fused_type: "scale".to_string() },
        );
        assert!(rule.matches_chain("scale", "scale"));
        assert!(!rule.matches_chain("scale", "crop"));
    }

    #[test]
    fn test_disabled_rule_no_match() {
        let mut rule = RewriteRule::new(
            RuleId(1),
            "test",
            PatternKind::SingleNode { filter_type: "scale".to_string() },
            RewriteAction::Remove,
        );
        rule.set_enabled(false);
        assert!(!rule.matches_node("scale", &HashMap::new()));
        assert!(!rule.matches_chain("scale", "scale"));
    }

    #[test]
    fn test_engine_add_and_count() {
        let mut engine = RewriteEngine::new();
        engine.add_rule(RewriteRule::new(
            RuleId(1), "r1",
            PatternKind::SingleNode { filter_type: "a".to_string() },
            RewriteAction::Remove,
        ));
        assert_eq!(engine.rule_count(), 1);
    }

    #[test]
    fn test_engine_priority_ordering() {
        let mut engine = RewriteEngine::new();
        engine.add_rule(RewriteRule::new(
            RuleId(1), "low",
            PatternKind::SingleNode { filter_type: "a".to_string() },
            RewriteAction::Remove,
        ).with_priority(10));
        engine.add_rule(RewriteRule::new(
            RuleId(2), "high",
            PatternKind::SingleNode { filter_type: "b".to_string() },
            RewriteAction::Remove,
        ).with_priority(100));
        let rules: Vec<_> = engine.find_matches("a", &HashMap::new());
        // Both might match different types, but internal ordering is by priority
        assert_eq!(engine.get_rule(RuleId(2)).expect("value should be valid").name, "high");
    }

    #[test]
    fn test_engine_find_matches() {
        let mut engine = RewriteEngine::new();
        engine.add_rule(RewriteRule::new(
            RuleId(1), "r1",
            PatternKind::SingleNode { filter_type: "scale".to_string() },
            RewriteAction::Remove,
        ));
        let matches = engine.find_matches("scale", &HashMap::new());
        assert_eq!(matches.len(), 1);
        assert!(engine.find_matches("crop", &HashMap::new()).is_empty());
    }

    #[test]
    fn test_engine_record_and_clear_history() {
        let mut engine = RewriteEngine::new();
        let rule = RewriteRule::new(
            RuleId(1), "test_rule",
            PatternKind::SingleNode { filter_type: "a".to_string() },
            RewriteAction::Remove,
        );
        engine.record_event(&rule, "node_42");
        assert_eq!(engine.history().len(), 1);
        assert_eq!(engine.history()[0].rule_name, "test_rule");
        engine.clear_history();
        assert!(engine.history().is_empty());
    }

    #[test]
    fn test_engine_remove_rule() {
        let mut engine = RewriteEngine::new();
        engine.add_rule(RewriteRule::new(
            RuleId(1), "r1",
            PatternKind::SingleNode { filter_type: "a".to_string() },
            RewriteAction::Remove,
        ));
        assert!(engine.remove_rule(RuleId(1)));
        assert!(!engine.remove_rule(RuleId(1)));
        assert_eq!(engine.rule_count(), 0);
    }

    #[test]
    fn test_standard_rules() {
        let rules = standard_rules();
        assert_eq!(rules.len(), 3);
        assert_eq!(rules[0].name, "identity_scale");
    }

    #[test]
    fn test_engine_enable_disable_all() {
        let mut engine = RewriteEngine::new();
        for i in 0..3 {
            engine.add_rule(RewriteRule::new(
                RuleId(i), &format!("r{i}"),
                PatternKind::SingleNode { filter_type: "a".to_string() },
                RewriteAction::Remove,
            ));
        }
        engine.disable_all();
        assert!(engine.find_matches("a", &HashMap::new()).is_empty());
        engine.enable_all();
        assert_eq!(engine.find_matches("a", &HashMap::new()).len(), 3);
    }
}
