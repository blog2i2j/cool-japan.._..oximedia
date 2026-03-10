//! License chaining and delegation for multi-tier DRM architectures.
//!
//! In many DRM systems a *root license* grants broad access rights, and
//! *leaf licenses* inherit from it while adding constraints (e.g. device
//! binding, time-windowed playback). This module models the license chain
//! and provides validation, serialisation, and traversal utilities.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// License node
// ---------------------------------------------------------------------------

/// The type of a license within a chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LicenseType {
    /// Root license -- the top of the chain.
    Root,
    /// Leaf license -- inherits from a parent and adds constraints.
    Leaf,
    /// Evaluation license -- for trial / preview periods.
    Evaluation,
}

impl fmt::Display for LicenseType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Root => write!(f, "root"),
            Self::Leaf => write!(f, "leaf"),
            Self::Evaluation => write!(f, "evaluation"),
        }
    }
}

/// A constraint imposed by a license.
#[derive(Debug, Clone, PartialEq)]
pub enum LicenseConstraint {
    /// The license expires at this Unix timestamp (seconds).
    ExpiresAt(u64),
    /// The license is valid only after this Unix timestamp (seconds).
    NotBefore(u64),
    /// Maximum number of plays allowed.
    MaxPlays(u32),
    /// Maximum concurrent streams allowed.
    MaxConcurrent(u32),
    /// Must be bound to a device with this fingerprint.
    DeviceBinding(String),
    /// Maximum resolution in pixels (width x height).
    MaxResolution(u32, u32),
    /// Maximum bitrate in bits/s.
    MaxBitrate(u64),
    /// Restrict to a set of territory codes (ISO 3166-1 alpha-2).
    TerritoryRestriction(Vec<String>),
}

impl LicenseConstraint {
    /// Returns a human-readable summary of this constraint.
    pub fn summary(&self) -> String {
        match self {
            Self::ExpiresAt(ts) => format!("expires at {ts}"),
            Self::NotBefore(ts) => format!("not before {ts}"),
            Self::MaxPlays(n) => format!("max {n} plays"),
            Self::MaxConcurrent(n) => format!("max {n} concurrent"),
            Self::DeviceBinding(fp) => format!("device {fp}"),
            Self::MaxResolution(w, h) => format!("max {w}x{h}"),
            Self::MaxBitrate(bps) => format!("max {bps} bps"),
            Self::TerritoryRestriction(codes) => format!("territories: {}", codes.join(",")),
        }
    }
}

/// A single license node in a chain.
#[derive(Debug, Clone)]
pub struct LicenseNode {
    /// Unique identifier for this license.
    pub id: String,
    /// The type of this license.
    pub license_type: LicenseType,
    /// Optional parent license ID. `None` for root licenses.
    pub parent_id: Option<String>,
    /// Content-ID this license applies to.
    pub content_id: String,
    /// Constraints imposed by this license.
    pub constraints: Vec<LicenseConstraint>,
    /// Unix timestamp (seconds) when this license was issued.
    pub issued_at: u64,
    /// Priority for conflict resolution (higher wins).
    pub priority: u32,
}

impl LicenseNode {
    /// Create a new root license.
    pub fn root(id: impl Into<String>, content_id: impl Into<String>, issued_at: u64) -> Self {
        Self {
            id: id.into(),
            license_type: LicenseType::Root,
            parent_id: None,
            content_id: content_id.into(),
            constraints: Vec::new(),
            issued_at,
            priority: 0,
        }
    }

    /// Create a new leaf license delegated from `parent_id`.
    pub fn leaf(
        id: impl Into<String>,
        parent_id: impl Into<String>,
        content_id: impl Into<String>,
        issued_at: u64,
    ) -> Self {
        Self {
            id: id.into(),
            license_type: LicenseType::Leaf,
            parent_id: Some(parent_id.into()),
            content_id: content_id.into(),
            constraints: Vec::new(),
            issued_at,
            priority: 0,
        }
    }

    /// Builder: add a constraint.
    pub fn with_constraint(mut self, c: LicenseConstraint) -> Self {
        self.constraints.push(c);
        self
    }

    /// Builder: set priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Returns `true` if this license has expired relative to `now` (seconds).
    pub fn is_expired(&self, now: u64) -> bool {
        self.constraints
            .iter()
            .any(|c| matches!(c, LicenseConstraint::ExpiresAt(ts) if now >= *ts))
    }

    /// Returns `true` if this license is not yet valid relative to `now`.
    pub fn is_not_yet_valid(&self, now: u64) -> bool {
        self.constraints
            .iter()
            .any(|c| matches!(c, LicenseConstraint::NotBefore(ts) if now < *ts))
    }

    /// Returns `true` if this is a root license (no parent).
    pub fn is_root(&self) -> bool {
        self.license_type == LicenseType::Root
    }

    /// Collect all constraints from this node.
    pub fn effective_constraints(&self) -> &[LicenseConstraint] {
        &self.constraints
    }
}

// ---------------------------------------------------------------------------
// License chain
// ---------------------------------------------------------------------------

/// Result of validating a license chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainValidation {
    /// The chain is valid.
    Valid,
    /// The chain has a broken link (parent not found).
    BrokenLink(String),
    /// A license in the chain has expired.
    Expired(String),
    /// A license is not yet valid.
    NotYetValid(String),
    /// The chain has a cycle.
    Cycle,
    /// The chain has no root license.
    NoRoot,
}

/// An ordered chain of licenses from root to leaf.
#[derive(Debug, Clone)]
pub struct LicenseChain {
    /// All license nodes in the chain, keyed by their ID.
    nodes: HashMap<String, LicenseNode>,
}

impl LicenseChain {
    /// Create a new, empty license chain.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Add a license node to the chain.
    pub fn add(&mut self, node: LicenseNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Remove a license node by ID.
    pub fn remove(&mut self, id: &str) -> Option<LicenseNode> {
        self.nodes.remove(id)
    }

    /// Get a license node by ID.
    pub fn get(&self, id: &str) -> Option<&LicenseNode> {
        self.nodes.get(id)
    }

    /// Return the number of nodes in the chain.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Find the root node(s) in the chain.
    pub fn roots(&self) -> Vec<&LicenseNode> {
        self.nodes.values().filter(|n| n.is_root()).collect()
    }

    /// Find all leaf nodes (nodes that are not the parent of any other node).
    pub fn leaves(&self) -> Vec<&LicenseNode> {
        let parent_ids: std::collections::HashSet<&str> = self
            .nodes
            .values()
            .filter_map(|n| n.parent_id.as_deref())
            .collect();
        self.nodes
            .values()
            .filter(|n| !parent_ids.contains(n.id.as_str()))
            .collect()
    }

    /// Walk from a leaf node up to the root, collecting all ancestor IDs.
    /// Returns `None` if a broken link is encountered.
    pub fn ancestors(&self, leaf_id: &str) -> Option<Vec<String>> {
        let mut result = Vec::new();
        let mut current_id = leaf_id.to_string();
        let mut visited = std::collections::HashSet::new();

        loop {
            if !visited.insert(current_id.clone()) {
                // Cycle detected
                return None;
            }
            let node = self.nodes.get(&current_id)?;
            result.push(current_id.clone());
            match &node.parent_id {
                Some(pid) => current_id = pid.clone(),
                None => break,
            }
        }
        Some(result)
    }

    /// Collect the effective constraints for a given license by merging all
    /// constraints from the leaf up to the root.
    pub fn effective_constraints(&self, leaf_id: &str) -> Vec<LicenseConstraint> {
        let ancestors = match self.ancestors(leaf_id) {
            Some(a) => a,
            None => return Vec::new(),
        };
        let mut constraints = Vec::new();
        // Walk root-to-leaf order (reverse of ancestors)
        for id in ancestors.iter().rev() {
            if let Some(node) = self.nodes.get(id) {
                constraints.extend(node.constraints.clone());
            }
        }
        constraints
    }

    /// Validate the entire chain structure at timestamp `now`.
    pub fn validate(&self, now: u64) -> ChainValidation {
        if self.nodes.is_empty() {
            return ChainValidation::NoRoot;
        }

        let roots = self.roots();
        if roots.is_empty() {
            return ChainValidation::NoRoot;
        }

        for node in self.nodes.values() {
            if let Some(ref parent_id) = node.parent_id {
                if !self.nodes.contains_key(parent_id) {
                    return ChainValidation::BrokenLink(node.id.clone());
                }
            }
            if node.is_expired(now) {
                return ChainValidation::Expired(node.id.clone());
            }
            if node.is_not_yet_valid(now) {
                return ChainValidation::NotYetValid(node.id.clone());
            }
        }

        // Check for cycles by walking from each node to root
        for id in self.nodes.keys() {
            if self.ancestors(id).is_none() {
                return ChainValidation::Cycle;
            }
        }

        ChainValidation::Valid
    }

    /// Return the chain depth (longest path from root to leaf).
    pub fn depth(&self) -> usize {
        let mut max_depth = 0;
        for leaf in self.leaves() {
            if let Some(path) = self.ancestors(&leaf.id) {
                max_depth = max_depth.max(path.len());
            }
        }
        max_depth
    }
}

impl Default for LicenseChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chain() -> LicenseChain {
        let mut chain = LicenseChain::new();
        let root = LicenseNode::root("root-1", "content-A", 1000);
        let leaf = LicenseNode::leaf("leaf-1", "root-1", "content-A", 1100);
        chain.add(root);
        chain.add(leaf);
        chain
    }

    #[test]
    fn test_license_type_display() {
        assert_eq!(LicenseType::Root.to_string(), "root");
        assert_eq!(LicenseType::Leaf.to_string(), "leaf");
        assert_eq!(LicenseType::Evaluation.to_string(), "evaluation");
    }

    #[test]
    fn test_license_constraint_summary() {
        let c = LicenseConstraint::MaxPlays(5);
        assert!(c.summary().contains("5"));
        let c2 = LicenseConstraint::MaxResolution(1920, 1080);
        assert!(c2.summary().contains("1920x1080"));
    }

    #[test]
    fn test_root_node_creation() {
        let root = LicenseNode::root("r1", "c1", 500);
        assert!(root.is_root());
        assert!(root.parent_id.is_none());
        assert_eq!(root.content_id, "c1");
    }

    #[test]
    fn test_leaf_node_creation() {
        let leaf = LicenseNode::leaf("l1", "r1", "c1", 600);
        assert!(!leaf.is_root());
        assert_eq!(leaf.parent_id.as_deref(), Some("r1"));
    }

    #[test]
    fn test_node_expired() {
        let node =
            LicenseNode::root("r", "c", 0).with_constraint(LicenseConstraint::ExpiresAt(1000));
        assert!(!node.is_expired(999));
        assert!(node.is_expired(1000));
    }

    #[test]
    fn test_node_not_yet_valid() {
        let node =
            LicenseNode::root("r", "c", 0).with_constraint(LicenseConstraint::NotBefore(500));
        assert!(node.is_not_yet_valid(499));
        assert!(!node.is_not_yet_valid(500));
    }

    #[test]
    fn test_chain_add_and_get() {
        let chain = sample_chain();
        assert_eq!(chain.len(), 2);
        assert!(chain.get("root-1").is_some());
        assert!(chain.get("leaf-1").is_some());
        assert!(chain.get("nonexistent").is_none());
    }

    #[test]
    fn test_chain_roots() {
        let chain = sample_chain();
        let roots = chain.roots();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].id, "root-1");
    }

    #[test]
    fn test_chain_leaves() {
        let chain = sample_chain();
        let leaves = chain.leaves();
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].id, "leaf-1");
    }

    #[test]
    fn test_chain_ancestors() {
        let chain = sample_chain();
        let ancestors = chain.ancestors("leaf-1").expect("ancestors should succeed");
        assert_eq!(ancestors, vec!["leaf-1", "root-1"]);
    }

    #[test]
    fn test_chain_validate_valid() {
        let chain = sample_chain();
        assert_eq!(chain.validate(500), ChainValidation::Valid);
    }

    #[test]
    fn test_chain_validate_broken_link() {
        let mut chain = LicenseChain::new();
        chain.add(LicenseNode::leaf("l1", "missing-parent", "c1", 100));
        let result = chain.validate(100);
        assert!(matches!(
            result,
            ChainValidation::BrokenLink(_) | ChainValidation::NoRoot
        ));
    }

    #[test]
    fn test_chain_validate_expired() {
        let mut chain = LicenseChain::new();
        chain.add(
            LicenseNode::root("r1", "c1", 0).with_constraint(LicenseConstraint::ExpiresAt(100)),
        );
        assert_eq!(
            chain.validate(200),
            ChainValidation::Expired("r1".to_string())
        );
    }

    #[test]
    fn test_chain_validate_empty() {
        let chain = LicenseChain::new();
        assert_eq!(chain.validate(0), ChainValidation::NoRoot);
    }

    #[test]
    fn test_chain_effective_constraints() {
        let mut chain = LicenseChain::new();
        chain
            .add(LicenseNode::root("r1", "c1", 0).with_constraint(LicenseConstraint::MaxPlays(10)));
        chain.add(
            LicenseNode::leaf("l1", "r1", "c1", 0)
                .with_constraint(LicenseConstraint::MaxResolution(1920, 1080)),
        );
        let constraints = chain.effective_constraints("l1");
        assert_eq!(constraints.len(), 2);
    }

    #[test]
    fn test_chain_remove() {
        let mut chain = sample_chain();
        let removed = chain.remove("leaf-1");
        assert!(removed.is_some());
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn test_chain_depth() {
        let mut chain = LicenseChain::new();
        chain.add(LicenseNode::root("r", "c", 0));
        chain.add(LicenseNode::leaf("l1", "r", "c", 0));
        chain.add(LicenseNode::leaf("l2", "l1", "c", 0));
        assert_eq!(chain.depth(), 3);
    }

    #[test]
    fn test_chain_default() {
        let chain = LicenseChain::default();
        assert!(chain.is_empty());
    }

    #[test]
    fn test_constraint_territory_summary() {
        let c = LicenseConstraint::TerritoryRestriction(vec!["US".to_string(), "GB".to_string()]);
        let s = c.summary();
        assert!(s.contains("US"));
        assert!(s.contains("GB"));
    }

    #[test]
    fn test_node_with_priority() {
        let node = LicenseNode::root("r", "c", 0).with_priority(5);
        assert_eq!(node.priority, 5);
    }
}
