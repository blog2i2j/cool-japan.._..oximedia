#![allow(dead_code)]
//! Filter graph descriptions for Python bindings.
//!
//! Defines a lightweight, pure-Rust representation of audio/video filter
//! chains that can be built and validated in Python before being sent to
//! the native processing pipeline.

use std::collections::HashMap;
use std::fmt;

/// The media domain a filter operates on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FilterDomain {
    /// Operates on video frames.
    Video,
    /// Operates on audio samples.
    Audio,
}

impl fmt::Display for FilterDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Video => write!(f, "video"),
            Self::Audio => write!(f, "audio"),
        }
    }
}

/// A single parameter value that can be passed to a filter.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    /// Integer value.
    Int(i64),
    /// Floating-point value.
    Float(f64),
    /// String value.
    Str(String),
    /// Boolean value.
    Bool(bool),
}

impl fmt::Display for ParamValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Str(v) => write!(f, "\"{v}\""),
            Self::Bool(v) => write!(f, "{v}"),
        }
    }
}

impl ParamValue {
    /// Try to extract an integer.
    pub fn as_int(&self) -> Option<i64> {
        if let Self::Int(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Try to extract a float.
    pub fn as_float(&self) -> Option<f64> {
        if let Self::Float(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Try to extract a string.
    pub fn as_str(&self) -> Option<&str> {
        if let Self::Str(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Try to extract a bool.
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }
}

/// Describes a single filter node in the filter graph.
#[derive(Debug, Clone)]
pub struct FilterNode {
    /// Unique node identifier within the graph.
    pub id: String,
    /// Filter name (e.g. `"scale"`, `"volume"`, `"crop"`).
    pub filter_name: String,
    /// Domain this filter operates on.
    pub domain: FilterDomain,
    /// Named parameters.
    pub params: HashMap<String, ParamValue>,
    /// Whether the filter is enabled (can be bypassed).
    pub enabled: bool,
}

impl FilterNode {
    /// Create a new filter node.
    pub fn new(
        id: impl Into<String>,
        filter_name: impl Into<String>,
        domain: FilterDomain,
    ) -> Self {
        Self {
            id: id.into(),
            filter_name: filter_name.into(),
            domain,
            params: HashMap::new(),
            enabled: true,
        }
    }

    /// Set a parameter.
    pub fn with_param(mut self, key: impl Into<String>, value: ParamValue) -> Self {
        self.params.insert(key.into(), value);
        self
    }

    /// Disable the filter.
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Number of parameters.
    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    /// Look up a parameter value.
    pub fn get_param(&self, key: &str) -> Option<&ParamValue> {
        self.params.get(key)
    }
}

impl fmt::Display for FilterNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.enabled { "on" } else { "off" };
        write!(
            f,
            "[{}] {} ({}) [{}]",
            self.id, self.filter_name, self.domain, status
        )
    }
}

/// Validation result for a filter graph.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the graph is valid.
    pub valid: bool,
    /// List of warning messages.
    pub warnings: Vec<String>,
    /// List of error messages.
    pub errors: Vec<String>,
}

impl ValidationResult {
    /// Create a passing result.
    pub fn ok() -> Self {
        Self {
            valid: true,
            warnings: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Create a failing result with one error.
    pub fn fail(error: impl Into<String>) -> Self {
        Self {
            valid: false,
            warnings: Vec::new(),
            errors: vec![error.into()],
        }
    }

    /// Add a warning.
    pub fn add_warning(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }

    /// Add an error and mark as invalid.
    pub fn add_error(&mut self, msg: impl Into<String>) {
        self.valid = false;
        self.errors.push(msg.into());
    }

    /// Total issue count (warnings + errors).
    pub fn issue_count(&self) -> usize {
        self.warnings.len() + self.errors.len()
    }
}

/// A linear chain of filter nodes.
#[derive(Debug, Clone, Default)]
pub struct FilterChain {
    /// Ordered list of filter nodes.
    nodes: Vec<FilterNode>,
}

impl FilterChain {
    /// Create an empty filter chain.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a filter node.
    pub fn push(&mut self, node: FilterNode) {
        self.nodes.push(node);
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get a node by index.
    pub fn get(&self, idx: usize) -> Option<&FilterNode> {
        self.nodes.get(idx)
    }

    /// Get a node by its id.
    pub fn find_by_id(&self, id: &str) -> Option<&FilterNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Remove a node by id, returning it if found.
    pub fn remove_by_id(&mut self, id: &str) -> Option<FilterNode> {
        if let Some(pos) = self.nodes.iter().position(|n| n.id == id) {
            Some(self.nodes.remove(pos))
        } else {
            None
        }
    }

    /// Count enabled nodes.
    pub fn enabled_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.enabled).count()
    }

    /// Validate the chain: check for duplicate IDs, empty names, and mixed domains.
    pub fn validate(&self) -> ValidationResult {
        let mut result = ValidationResult::ok();
        if self.nodes.is_empty() {
            result.add_warning("Filter chain is empty");
            return result;
        }

        // Check duplicate IDs
        let mut seen_ids = std::collections::HashSet::new();
        for node in &self.nodes {
            if !seen_ids.insert(&node.id) {
                result.add_error(format!("Duplicate filter ID: {}", node.id));
            }
            if node.filter_name.is_empty() {
                result.add_error(format!("Node '{}' has empty filter name", node.id));
            }
        }

        // Check for domain consistency
        let domains: std::collections::HashSet<_> = self.nodes.iter().map(|n| n.domain).collect();
        if domains.len() > 1 {
            result.add_warning("Filter chain mixes video and audio domains");
        }

        result
    }

    /// Produce a textual description of the chain.
    pub fn describe(&self) -> String {
        if self.nodes.is_empty() {
            return "(empty chain)".to_string();
        }
        self.nodes
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(" -> ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── FilterDomain ───────────────────────────────────────────────────────

    #[test]
    fn test_domain_display() {
        assert_eq!(FilterDomain::Video.to_string(), "video");
        assert_eq!(FilterDomain::Audio.to_string(), "audio");
    }

    // ── ParamValue ─────────────────────────────────────────────────────────

    #[test]
    fn test_param_value_int() {
        let v = ParamValue::Int(42);
        assert_eq!(v.as_int(), Some(42));
        assert!(v.as_float().is_none());
        assert_eq!(v.to_string(), "42");
    }

    #[test]
    fn test_param_value_float() {
        let v = ParamValue::Float(3.14);
        assert_eq!(v.as_float(), Some(3.14));
        assert!(v.as_int().is_none());
    }

    #[test]
    fn test_param_value_str() {
        let v = ParamValue::Str("hello".into());
        assert_eq!(v.as_str(), Some("hello"));
        assert!(v.as_bool().is_none());
    }

    #[test]
    fn test_param_value_bool() {
        let v = ParamValue::Bool(true);
        assert_eq!(v.as_bool(), Some(true));
        assert!(v.as_int().is_none());
    }

    // ── FilterNode ─────────────────────────────────────────────────────────

    #[test]
    fn test_filter_node_new() {
        let n = FilterNode::new("n1", "scale", FilterDomain::Video);
        assert_eq!(n.id, "n1");
        assert_eq!(n.filter_name, "scale");
        assert!(n.enabled);
        assert_eq!(n.param_count(), 0);
    }

    #[test]
    fn test_filter_node_with_params() {
        let n = FilterNode::new("n1", "scale", FilterDomain::Video)
            .with_param("width", ParamValue::Int(1920))
            .with_param("height", ParamValue::Int(1080));
        assert_eq!(n.param_count(), 2);
        assert_eq!(
            n.get_param("width")
                .expect("get_param should succeed")
                .as_int(),
            Some(1920)
        );
    }

    #[test]
    fn test_filter_node_disabled() {
        let n = FilterNode::new("n1", "volume", FilterDomain::Audio).disabled();
        assert!(!n.enabled);
    }

    #[test]
    fn test_filter_node_display() {
        let n = FilterNode::new("f1", "crop", FilterDomain::Video);
        let s = n.to_string();
        assert!(s.contains("f1"));
        assert!(s.contains("crop"));
        assert!(s.contains("on"));
    }

    // ── ValidationResult ───────────────────────────────────────────────────

    #[test]
    fn test_validation_ok() {
        let r = ValidationResult::ok();
        assert!(r.valid);
        assert_eq!(r.issue_count(), 0);
    }

    #[test]
    fn test_validation_fail() {
        let r = ValidationResult::fail("broken");
        assert!(!r.valid);
        assert_eq!(r.errors.len(), 1);
    }

    #[test]
    fn test_validation_add_error_marks_invalid() {
        let mut r = ValidationResult::ok();
        r.add_error("err1");
        assert!(!r.valid);
        assert_eq!(r.issue_count(), 1);
    }

    // ── FilterChain ────────────────────────────────────────────────────────

    #[test]
    fn test_chain_empty() {
        let c = FilterChain::new();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn test_chain_push_and_get() {
        let mut c = FilterChain::new();
        c.push(FilterNode::new("f1", "scale", FilterDomain::Video));
        assert_eq!(c.len(), 1);
        assert_eq!(c.get(0).expect("get should succeed").filter_name, "scale");
    }

    #[test]
    fn test_chain_find_by_id() {
        let mut c = FilterChain::new();
        c.push(FilterNode::new("a", "crop", FilterDomain::Video));
        c.push(FilterNode::new("b", "scale", FilterDomain::Video));
        assert!(c.find_by_id("b").is_some());
        assert!(c.find_by_id("z").is_none());
    }

    #[test]
    fn test_chain_remove_by_id() {
        let mut c = FilterChain::new();
        c.push(FilterNode::new("x", "crop", FilterDomain::Video));
        let removed = c.remove_by_id("x");
        assert!(removed.is_some());
        assert!(c.is_empty());
    }

    #[test]
    fn test_chain_enabled_count() {
        let mut c = FilterChain::new();
        c.push(FilterNode::new("a", "scale", FilterDomain::Video));
        c.push(FilterNode::new("b", "crop", FilterDomain::Video).disabled());
        assert_eq!(c.enabled_count(), 1);
    }

    #[test]
    fn test_chain_validate_empty_warns() {
        let c = FilterChain::new();
        let r = c.validate();
        assert!(r.valid);
        assert_eq!(r.warnings.len(), 1);
    }

    #[test]
    fn test_chain_validate_dup_ids() {
        let mut c = FilterChain::new();
        c.push(FilterNode::new("dup", "scale", FilterDomain::Video));
        c.push(FilterNode::new("dup", "crop", FilterDomain::Video));
        let r = c.validate();
        assert!(!r.valid);
    }

    #[test]
    fn test_chain_validate_mixed_domains_warns() {
        let mut c = FilterChain::new();
        c.push(FilterNode::new("v1", "scale", FilterDomain::Video));
        c.push(FilterNode::new("a1", "volume", FilterDomain::Audio));
        let r = c.validate();
        assert!(r.valid); // mixed domains is a warning, not an error
        assert!(!r.warnings.is_empty());
    }

    #[test]
    fn test_chain_describe_empty() {
        let c = FilterChain::new();
        assert_eq!(c.describe(), "(empty chain)");
    }

    #[test]
    fn test_chain_describe_multiple() {
        let mut c = FilterChain::new();
        c.push(FilterNode::new("a", "scale", FilterDomain::Video));
        c.push(FilterNode::new("b", "crop", FilterDomain::Video));
        let desc = c.describe();
        assert!(desc.contains("->"));
        assert!(desc.contains("scale"));
        assert!(desc.contains("crop"));
    }
}
