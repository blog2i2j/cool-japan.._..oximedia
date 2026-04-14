//! Plugin priority and ordering with dependency constraints.
//!
//! When multiple plugins claim the same codec, [`PluginPriority`] determines
//! which plugin is preferred.  Lower numeric values indicate higher priority:
//! `SYSTEM` (0) beats `HIGH` (100), which beats `NORMAL` (500), and so on.
//!
//! For complex scenarios where plugins have ordering dependencies (e.g. "plugin
//! A must load before plugin B"), use [`PluginOrderEntry`] with
//! [`PluginConstraint`] and resolve the final order via [`resolve_order`].
//!
//! # Usage
//!
//! ```rust
//! use oximedia_plugin::priority::{PluginPriority, PluginOrderEntry, resolve_order};
//!
//! assert!(PluginPriority::SYSTEM < PluginPriority::HIGH);
//! assert!(PluginPriority::HIGH < PluginPriority::NORMAL);
//!
//! let entries = vec![
//!     PluginOrderEntry::new("decoder", 10).before("encoder"),
//!     PluginOrderEntry::new("encoder", 5),
//! ];
//! let order = resolve_order(&entries).expect("no cycles");
//! assert_eq!(order.is_before("decoder", "encoder"), Some(true));
//! ```

use crate::error::{PluginError, PluginResult};
use std::collections::{HashMap, HashSet};

// ── PluginPriority ────────────────────────────────────────────────────────────

/// Priority level for codec conflict resolution.
///
/// Lower values indicate higher priority.  When multiple plugins provide
/// the same codec, the one with the lowest [`PluginPriority`] wins.
/// Among plugins with equal priority, the first-registered plugin wins.
///
/// # Predefined levels
///
/// | Constant          | Value | Meaning                          |
/// |-------------------|-------|----------------------------------|
/// | `SYSTEM`          | 0     | Built-in / system-level plugins  |
/// | `HIGH`            | 100   | Highly preferred alternatives    |
/// | `NORMAL`          | 500   | Default priority                 |
/// | `LOW`             | 1000  | Fallback / last-resort plugins   |
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PluginPriority(pub u32);

impl PluginPriority {
    /// Built-in / system-level plugins.  Highest precedence.
    pub const SYSTEM: Self = Self(0);
    /// Highly preferred alternatives.
    pub const HIGH: Self = Self(100);
    /// Default priority for most plugins.
    pub const NORMAL: Self = Self(500);
    /// Fallback / last-resort plugins.  Lowest precedence.
    pub const LOW: Self = Self(1000);

    /// Convert to the internal `i32` representation used by [`PluginRegistry`].
    ///
    /// The existing registry stores priority as `i32` with *higher* values
    /// winning; `PluginPriority` uses *lower* values for higher precedence.
    /// We negate and clamp to bridge the two conventions.
    ///
    /// [`PluginRegistry`]: crate::registry::PluginRegistry
    #[must_use]
    pub fn to_registry_i32(self) -> i32 {
        // u32::MAX / 2 fits in i32 range after negation; we cap at i32::MAX.
        let raw = self.0.min(i32::MAX as u32) as i32;
        // Negate: lower PluginPriority value → higher internal i32 priority.
        raw.saturating_neg()
    }
}

impl Default for PluginPriority {
    fn default() -> Self {
        Self::NORMAL
    }
}

impl std::fmt::Display for PluginPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self.0 {
            0 => "SYSTEM",
            1..=199 => "HIGH",
            200..=799 => "NORMAL",
            _ => "LOW",
        };
        write!(f, "PluginPriority({}, {})", self.0, label)
    }
}

// ── PluginConstraint ─────────────────────────────────────────────────────────

/// A constraint on plugin ordering relative to another plugin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginConstraint {
    /// This plugin must be loaded before the named plugin.
    Before(String),
    /// This plugin must be loaded after the named plugin.
    After(String),
}

impl std::fmt::Display for PluginConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Before(name) => write!(f, "before:{name}"),
            Self::After(name) => write!(f, "after:{name}"),
        }
    }
}

// ── PluginOrderEntry ─────────────────────────────────────────────────────────

/// An entry combining a plugin name with its numeric priority and optional
/// before/after ordering constraints.
///
/// Use with [`resolve_order`] to compute a valid loading order that respects
/// all constraints while preferring higher-priority (lower numeric value)
/// plugins first.
#[derive(Debug, Clone)]
pub struct PluginOrderEntry {
    /// The plugin's unique name.
    pub name: String,
    /// Numeric priority: lower values indicate higher precedence.
    pub priority: i32,
    /// Ordering constraints relative to other plugins.
    pub constraints: Vec<PluginConstraint>,
}

impl PluginOrderEntry {
    /// Create a new order entry with no constraints.
    #[must_use]
    pub fn new(name: impl Into<String>, priority: i32) -> Self {
        Self {
            name: name.into(),
            priority,
            constraints: Vec::new(),
        }
    }

    /// Add a constraint that this plugin must be ordered before `other`.
    #[must_use]
    pub fn before(mut self, other: impl Into<String>) -> Self {
        self.constraints
            .push(PluginConstraint::Before(other.into()));
        self
    }

    /// Add a constraint that this plugin must be ordered after `other`.
    #[must_use]
    pub fn after(mut self, other: impl Into<String>) -> Self {
        self.constraints
            .push(PluginConstraint::After(other.into()));
        self
    }

    /// Check whether this entry has any ordering constraints.
    #[must_use]
    pub fn has_constraints(&self) -> bool {
        !self.constraints.is_empty()
    }
}

impl std::fmt::Display for PluginOrderEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(priority={})", self.name, self.priority)?;
        if !self.constraints.is_empty() {
            let cs: Vec<String> = self.constraints.iter().map(|c| format!("{c}")).collect();
            write!(f, " [{}]", cs.join(", "))?;
        }
        Ok(())
    }
}

// ── PluginOrder ──────────────────────────────────────────────────────────────

/// An ordered set of plugins produced by topological sort with
/// priority-based tie-breaking.
///
/// The order respects all constraints while preferring lower numeric
/// priority (higher precedence) plugins earlier in the sequence.
#[derive(Debug, Clone)]
pub struct PluginOrder {
    /// Plugins in resolved order (first = highest precedence).
    pub ordered: Vec<String>,
}

impl PluginOrder {
    /// Return the position of a plugin in the resolved order.
    ///
    /// Returns `None` if the plugin is not in the order.
    #[must_use]
    pub fn position(&self, name: &str) -> Option<usize> {
        self.ordered.iter().position(|n| n == name)
    }

    /// Check whether plugin `a` comes before plugin `b` in the order.
    ///
    /// Returns `None` if either plugin is not in the order.
    #[must_use]
    pub fn is_before(&self, a: &str, b: &str) -> Option<bool> {
        let pos_a = self.position(a)?;
        let pos_b = self.position(b)?;
        Some(pos_a < pos_b)
    }

    /// Return the number of plugins in the order.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ordered.len()
    }

    /// Check if the order is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ordered.is_empty()
    }
}

// ── resolve_order ────────────────────────────────────────────────────────────

/// Resolve a set of [`PluginOrderEntry`] descriptors into a final
/// [`PluginOrder`] using topological sort with priority-based tie-breaking.
///
/// # Algorithm
///
/// 1. Build a directed graph from constraints (`Before(B)` on A adds edge A->B,
///    `After(B)` on A adds edge B->A).
/// 2. Run Kahn's algorithm with a priority-min queue: among all nodes with
///    in-degree 0, the one with the lowest numeric priority (highest
///    precedence) is emitted first.
///
/// Unknown constraint targets (plugins mentioned in `Before`/`After` that
/// are not in the input set) are silently ignored.
///
/// # Errors
///
/// Returns [`PluginError::InitFailed`] if the constraint graph contains a
/// cycle (no valid ordering exists).
pub fn resolve_order(entries: &[PluginOrderEntry]) -> PluginResult<PluginOrder> {
    if entries.is_empty() {
        return Ok(PluginOrder {
            ordered: Vec::new(),
        });
    }

    let names: HashSet<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    let name_to_prio: HashMap<&str, i32> = entries
        .iter()
        .map(|e| (e.name.as_str(), e.priority))
        .collect();

    // Adjacency list and in-degree map.
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut in_degree: HashMap<&str, usize> = HashMap::new();

    for name in &names {
        adj.entry(name).or_default();
        in_degree.entry(name).or_insert(0);
    }

    for entry in entries {
        for constraint in &entry.constraints {
            match constraint {
                PluginConstraint::Before(target) => {
                    // entry.name must come before target -> edge entry.name -> target
                    if names.contains(target.as_str()) {
                        adj.entry(entry.name.as_str())
                            .or_default()
                            .push(target.as_str());
                        *in_degree.entry(target.as_str()).or_insert(0) += 1;
                    }
                }
                PluginConstraint::After(target) => {
                    // entry.name must come after target -> edge target -> entry.name
                    if names.contains(target.as_str()) {
                        adj.entry(target.as_str())
                            .or_default()
                            .push(entry.name.as_str());
                        *in_degree.entry(entry.name.as_str()).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    // Kahn's algorithm with priority-min tie-breaking.
    let mut ready: Vec<&str> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(&name, _)| name)
        .collect();

    let mut result: Vec<String> = Vec::with_capacity(names.len());

    while !ready.is_empty() {
        // Sort: lowest numeric priority first (highest precedence), then by name.
        ready.sort_by(|a, b| {
            let pa = name_to_prio.get(a).copied().unwrap_or(0);
            let pb = name_to_prio.get(b).copied().unwrap_or(0);
            pa.cmp(&pb).then_with(|| a.cmp(b))
        });

        let node = ready.remove(0);
        result.push(node.to_string());

        if let Some(neighbors) = adj.get(node) {
            for &neighbor in neighbors {
                if let Some(deg) = in_degree.get_mut(neighbor) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        ready.push(neighbor);
                    }
                }
            }
        }
    }

    if result.len() != names.len() {
        let resolved: HashSet<&str> = result.iter().map(|s| s.as_str()).collect();
        let stuck: Vec<String> = names
            .iter()
            .filter(|n| !resolved.contains(**n))
            .map(|n| n.to_string())
            .collect();
        return Err(PluginError::InitFailed(format!(
            "Cyclic dependency detected among plugins: {}",
            stuck.join(", ")
        )));
    }

    Ok(PluginOrder { ordered: result })
}

/// Validate that a set of constraints has no cycles without producing a
/// full ordering.
///
/// This is cheaper than [`resolve_order`] when you only need to check
/// feasibility.
pub fn validate_constraints(entries: &[PluginOrderEntry]) -> PluginResult<()> {
    resolve_order(entries).map(|_| ())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- PluginPriority tests --

    #[test]
    fn test_priority_ordering() {
        assert!(PluginPriority::SYSTEM < PluginPriority::HIGH);
        assert!(PluginPriority::HIGH < PluginPriority::NORMAL);
        assert!(PluginPriority::NORMAL < PluginPriority::LOW);
    }

    #[test]
    fn test_priority_values() {
        assert_eq!(PluginPriority::SYSTEM.0, 0);
        assert_eq!(PluginPriority::HIGH.0, 100);
        assert_eq!(PluginPriority::NORMAL.0, 500);
        assert_eq!(PluginPriority::LOW.0, 1000);
    }

    #[test]
    fn test_to_registry_i32_lower_becomes_higher() {
        let sys = PluginPriority::SYSTEM.to_registry_i32();
        let high = PluginPriority::HIGH.to_registry_i32();
        let normal = PluginPriority::NORMAL.to_registry_i32();
        let low = PluginPriority::LOW.to_registry_i32();
        assert!(sys > high, "SYSTEM should map to higher i32 than HIGH");
        assert!(high > normal);
        assert!(normal > low);
    }

    #[test]
    fn test_default_is_normal() {
        assert_eq!(PluginPriority::default(), PluginPriority::NORMAL);
    }

    #[test]
    fn test_custom_priority() {
        let p = PluginPriority(250);
        assert!(p > PluginPriority::HIGH);
        assert!(p < PluginPriority::NORMAL);
    }

    #[test]
    fn test_display() {
        let s = format!("{}", PluginPriority::SYSTEM);
        assert!(s.contains("SYSTEM"));
        let s2 = format!("{}", PluginPriority::LOW);
        assert!(s2.contains("LOW"));
    }

    // -- PluginConstraint tests --

    #[test]
    fn test_constraint_display() {
        let before = PluginConstraint::Before("foo".to_string());
        assert_eq!(format!("{before}"), "before:foo");

        let after = PluginConstraint::After("bar".to_string());
        assert_eq!(format!("{after}"), "after:bar");
    }

    // -- PluginOrderEntry tests --

    #[test]
    fn test_order_entry_display() {
        let entry = PluginOrderEntry::new("test", 42)
            .before("other")
            .after("dep");
        let s = format!("{entry}");
        assert!(s.contains("test"));
        assert!(s.contains("42"));
        assert!(s.contains("before:other"));
        assert!(s.contains("after:dep"));
    }

    #[test]
    fn test_has_constraints() {
        let no_c = PluginOrderEntry::new("plain", 0);
        assert!(!no_c.has_constraints());

        let with_c = PluginOrderEntry::new("constrained", 0).before("x");
        assert!(with_c.has_constraints());
    }

    // -- resolve_order tests --

    #[test]
    fn test_empty_resolve() {
        let order = resolve_order(&[]).expect("empty should succeed");
        assert!(order.is_empty());
        assert_eq!(order.len(), 0);
    }

    #[test]
    fn test_single_plugin() {
        let entries = vec![PluginOrderEntry::new("alpha", 10)];
        let order = resolve_order(&entries).expect("single should succeed");
        assert_eq!(order.ordered, vec!["alpha"]);
        assert_eq!(order.position("alpha"), Some(0));
    }

    #[test]
    fn test_priority_ordering_no_constraints() {
        let entries = vec![
            PluginOrderEntry::new("low", 1000),
            PluginOrderEntry::new("high", 1),
            PluginOrderEntry::new("mid", 50),
        ];
        let order = resolve_order(&entries).expect("should succeed");
        assert_eq!(order.ordered, vec!["high", "mid", "low"]);
    }

    #[test]
    fn test_before_constraint() {
        let entries = vec![
            PluginOrderEntry::new("A", 10).before("B"),
            PluginOrderEntry::new("B", 20),
        ];
        let order = resolve_order(&entries).expect("should succeed");
        assert_eq!(order.is_before("A", "B"), Some(true));
    }

    #[test]
    fn test_after_constraint() {
        let entries = vec![
            PluginOrderEntry::new("A", 1).after("B"),
            PluginOrderEntry::new("B", 100),
        ];
        let order = resolve_order(&entries).expect("should succeed");
        // A must come after B, even though A has higher precedence.
        assert_eq!(order.is_before("B", "A"), Some(true));
    }

    #[test]
    fn test_cycle_detection() {
        let entries = vec![
            PluginOrderEntry::new("X", 10).before("Y"),
            PluginOrderEntry::new("Y", 10).before("X"),
        ];
        let err = resolve_order(&entries).expect_err("cycle should fail");
        assert!(err.to_string().contains("Cyclic"));
    }

    #[test]
    fn test_unknown_constraint_target_ignored() {
        let entries = vec![
            PluginOrderEntry::new("A", 10).before("nonexistent"),
            PluginOrderEntry::new("B", 5),
        ];
        let order = resolve_order(&entries).expect("should succeed");
        assert_eq!(order.len(), 2);
    }

    #[test]
    fn test_chain_constraints() {
        // A -> B -> C via before constraints
        let entries = vec![
            PluginOrderEntry::new("C", 1),
            PluginOrderEntry::new("A", 100).before("B"),
            PluginOrderEntry::new("B", 50).before("C"),
        ];
        let order = resolve_order(&entries).expect("should succeed");
        assert_eq!(order.is_before("A", "B"), Some(true));
        assert_eq!(order.is_before("B", "C"), Some(true));
        assert_eq!(order.is_before("A", "C"), Some(true));
    }

    #[test]
    fn test_validate_constraints_ok() {
        let entries = vec![
            PluginOrderEntry::new("A", 10).before("B"),
            PluginOrderEntry::new("B", 5),
        ];
        assert!(validate_constraints(&entries).is_ok());
    }

    #[test]
    fn test_validate_constraints_cycle() {
        let entries = vec![
            PluginOrderEntry::new("A", 10).before("B"),
            PluginOrderEntry::new("B", 10).before("C"),
            PluginOrderEntry::new("C", 10).before("A"),
        ];
        assert!(validate_constraints(&entries).is_err());
    }

    #[test]
    fn test_position_not_found() {
        let order = PluginOrder {
            ordered: vec!["A".to_string()],
        };
        assert_eq!(order.position("B"), None);
    }

    #[test]
    fn test_is_before_missing_plugin() {
        let order = PluginOrder {
            ordered: vec!["A".to_string(), "B".to_string()],
        };
        assert_eq!(order.is_before("A", "C"), None);
        assert_eq!(order.is_before("C", "B"), None);
    }

    #[test]
    fn test_equal_priority_stable_by_name() {
        let entries = vec![
            PluginOrderEntry::new("beta", 0),
            PluginOrderEntry::new("alpha", 0),
            PluginOrderEntry::new("gamma", 0),
        ];
        let order = resolve_order(&entries).expect("should succeed");
        assert_eq!(order.ordered, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_complex_dag() {
        // Diamond: A -> B, A -> C, B -> D, C -> D
        let entries = vec![
            PluginOrderEntry::new("A", 10).before("B").before("C"),
            PluginOrderEntry::new("B", 5).before("D"),
            PluginOrderEntry::new("C", 3).before("D"),
            PluginOrderEntry::new("D", 1),
        ];
        let order = resolve_order(&entries).expect("should succeed");
        assert_eq!(order.is_before("A", "B"), Some(true));
        assert_eq!(order.is_before("A", "C"), Some(true));
        assert_eq!(order.is_before("B", "D"), Some(true));
        assert_eq!(order.is_before("C", "D"), Some(true));
    }
}
