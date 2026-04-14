//! Hierarchical facet aggregation driven by a `FacetableDocument` trait.
//!
//! This module provides a way to build facet trees from heterogeneous document
//! types without coupling to any concrete struct. Documents expose named fields
//! through the [`FacetableDocument`] trait, and [`build_hierarchical_facet`]
//! groups them along a multi-level path (e.g., format → codec).
//!
//! # Example
//!
//! ```
//! use oximedia_search::facet::hierarchical::{
//!     FacetableDocument, HierarchicalFacet, build_hierarchical_facet,
//! };
//!
//! struct Doc { format: &'static str, codec: &'static str }
//!
//! impl FacetableDocument for Doc {
//!     fn get_field(&self, name: &str) -> Option<String> {
//!         match name {
//!             "format" => Some(self.format.to_string()),
//!             "codec"  => Some(self.codec.to_string()),
//!             _ => None,
//!         }
//!     }
//! }
//!
//! let docs: Vec<Doc> = vec![
//!     Doc { format: "video", codec: "av1" },
//!     Doc { format: "video", codec: "vp9" },
//!     Doc { format: "audio", codec: "opus" },
//! ];
//!
//! let refs: Vec<&dyn FacetableDocument> = docs.iter().map(|d| d as &dyn FacetableDocument).collect();
//! let facet = build_hierarchical_facet(&refs, &["format", "codec"]);
//!
//! assert_eq!(facet.count_at_path(&["video"]), 2);
//! assert_eq!(facet.count_at_path(&["video", "av1"]), 1);
//! assert_eq!(facet.count_at_path(&["audio", "opus"]), 1);
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────────────

/// Abstraction over any document type that exposes named string fields.
///
/// Implementors return `None` for fields that are absent, which causes the
/// document to be skipped at that path level.
pub trait FacetableDocument {
    /// Return the value of a named field, or `None` if the field is absent.
    fn get_field(&self, name: &str) -> Option<String>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tree node
// ─────────────────────────────────────────────────────────────────────────────

/// A single node in a hierarchical facet tree.
///
/// `count` reflects the number of documents that reached **at least** this
/// node (bottom-up propagation is not automatic here — each level counts
/// documents whose path passes through that node).
#[derive(Debug, Clone, Default)]
pub struct FacetNode {
    /// Label for this node (the value at this tree level).
    pub name: String,
    /// Documents counted at this node (includes all descendants).
    pub count: usize,
    /// Child nodes keyed by their label.
    pub children: HashMap<String, FacetNode>,
}

impl FacetNode {
    /// Create a new node with the given `name` and zero count.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            count: 0,
            children: HashMap::new(),
        }
    }

    /// Recursively insert a path slice into the subtree rooted at this node.
    ///
    /// Each node along the path has its count incremented. If any component is
    /// `None` (the document lacks the field), insertion stops at that depth.
    fn insert_path(&mut self, components: &[Option<String>]) {
        if components.is_empty() {
            return;
        }
        let first = match components[0].as_deref() {
            Some(v) => v,
            None => return, // field absent — stop traversal
        };

        let child = self
            .children
            .entry(first.to_string())
            .or_insert_with(|| FacetNode::new(first));

        child.count += 1;
        child.insert_path(&components[1..]);
    }

    /// Navigate from this node along `path`, returning a reference to the node
    /// at the end of the path, or `None` if any component is missing.
    #[must_use]
    pub fn navigate(&self, path: &[&str]) -> Option<&FacetNode> {
        if path.is_empty() {
            return Some(self);
        }
        let child = self.children.get(path[0])?;
        child.navigate(&path[1..])
    }

    /// Return children sorted by count descending (then name ascending for
    /// determinism).
    #[must_use]
    pub fn sorted_children(&self) -> Vec<&FacetNode> {
        let mut children: Vec<&FacetNode> = self.children.values().collect();
        children.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.name.cmp(&b.name)));
        children
    }

    /// Recursively render this node and its descendants as a JSON string.
    fn to_json_inner(&self, out: &mut String) {
        out.push('{');
        out.push_str("\"name\":");
        json_string(&self.name, out);
        out.push_str(",\"count\":");
        out.push_str(&self.count.to_string());
        if !self.children.is_empty() {
            out.push_str(",\"children\":[");
            // Render children in deterministic order (sorted by count desc, name asc).
            let mut children: Vec<&FacetNode> = self.children.values().collect();
            children.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.name.cmp(&b.name)));
            let mut first = true;
            for child in children {
                if !first {
                    out.push(',');
                }
                first = false;
                child.to_json_inner(out);
            }
            out.push(']');
        }
        out.push('}');
    }
}

/// Write a JSON-escaped string literal into `out`.
fn json_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                // Control character — use \uXXXX encoding.
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// ─────────────────────────────────────────────────────────────────────────────
// HierarchicalFacet
// ─────────────────────────────────────────────────────────────────────────────

/// A complete hierarchical facet tree built from a set of documents.
///
/// The tree root is a virtual node (unnamed, zero-count) whose children are
/// the distinct values of the **first** field in the requested path.
#[derive(Debug, Clone)]
pub struct HierarchicalFacet {
    /// Virtual root node — its children are the top-level facet values.
    pub root: FacetNode,
}

impl HierarchicalFacet {
    /// Create an empty `HierarchicalFacet` with a nameless root.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: FacetNode {
                name: String::new(),
                count: 0,
                children: HashMap::new(),
            },
        }
    }

    /// Navigate the tree along `path` and return the document count at that
    /// node, or `0` if the path does not exist.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_search::facet::hierarchical::{FacetableDocument, build_hierarchical_facet};
    ///
    /// struct Doc { format: &'static str }
    /// impl FacetableDocument for Doc {
    ///     fn get_field(&self, name: &str) -> Option<String> {
    ///         if name == "format" { Some(self.format.to_string()) } else { None }
    ///     }
    /// }
    ///
    /// let docs = vec![Doc { format: "video" }, Doc { format: "video" }];
    /// let refs: Vec<&dyn FacetableDocument> = docs.iter().map(|d| d as _).collect();
    /// let facet = build_hierarchical_facet(&refs, &["format"]);
    /// assert_eq!(facet.count_at_path(&["video"]), 2);
    /// assert_eq!(facet.count_at_path(&["audio"]), 0);
    /// ```
    #[must_use]
    pub fn count_at_path(&self, path: &[&str]) -> usize {
        self.root.navigate(path).map_or(0, |n| n.count)
    }

    /// Render the entire tree as a hand-written JSON string.
    ///
    /// The format is:
    /// ```json
    /// {"children":[{"name":"video","count":3,"children":[...]},...]}
    /// ```
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(256);
        out.push_str("{\"children\":[");
        let mut children: Vec<&FacetNode> = self.root.children.values().collect();
        children.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.name.cmp(&b.name)));
        let mut first = true;
        for child in children {
            if !first {
                out.push(',');
            }
            first = false;
            child.to_json_inner(&mut out);
        }
        out.push_str("]}");
        out
    }

    /// Return the top-level children (first-level facet values) sorted by
    /// count descending.
    #[must_use]
    pub fn top_level(&self) -> Vec<&FacetNode> {
        self.root.sorted_children()
    }
}

impl Default for HierarchicalFacet {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Builder function
// ─────────────────────────────────────────────────────────────────────────────

/// Build a [`HierarchicalFacet`] from a slice of `FacetableDocument` trait
/// objects, grouping by the sequence of field names given in `path`.
///
/// Documents that return `None` for any field in the path are counted at the
/// deepest level they can reach; they do **not** contribute to deeper nodes.
///
/// # Arguments
///
/// * `documents` — slice of document references implementing `FacetableDocument`.
/// * `path` — ordered list of field names forming the hierarchy levels.
///   e.g., `&["format", "codec"]` groups first by format, then by codec within
///   each format.
///
/// # Example
///
/// ```
/// use oximedia_search::facet::hierarchical::{FacetableDocument, build_hierarchical_facet};
///
/// struct MediaDoc { format: &'static str, codec: &'static str }
/// impl FacetableDocument for MediaDoc {
///     fn get_field(&self, name: &str) -> Option<String> {
///         match name {
///             "format" => Some(self.format.to_string()),
///             "codec"  => Some(self.codec.to_string()),
///             _        => None,
///         }
///     }
/// }
///
/// let docs = vec![
///     MediaDoc { format: "video", codec: "av1" },
///     MediaDoc { format: "video", codec: "vp9" },
///     MediaDoc { format: "audio", codec: "opus" },
/// ];
/// let refs: Vec<&dyn FacetableDocument> = docs.iter().map(|d| d as _).collect();
/// let facet = build_hierarchical_facet(&refs, &["format", "codec"]);
///
/// assert_eq!(facet.count_at_path(&["video"]), 2);
/// assert_eq!(facet.count_at_path(&["video", "av1"]), 1);
/// assert_eq!(facet.count_at_path(&["audio"]), 1);
/// ```
pub fn build_hierarchical_facet(
    documents: &[&dyn FacetableDocument],
    path: &[&str],
) -> HierarchicalFacet {
    let mut facet = HierarchicalFacet::new();

    for doc in documents {
        // Collect field values along the path for this document.
        let field_values: Vec<Option<String>> =
            path.iter().map(|field| doc.get_field(field)).collect();

        // Insert into the tree — stops at the first None.
        facet.root.insert_path(&field_values);
    }

    facet
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test document ─────────────────────────────────────────────────────

    struct TestDoc {
        format: Option<&'static str>,
        codec: Option<&'static str>,
        profile: Option<&'static str>,
    }

    impl TestDoc {
        fn new(
            format: Option<&'static str>,
            codec: Option<&'static str>,
            profile: Option<&'static str>,
        ) -> Self {
            Self {
                format,
                codec,
                profile,
            }
        }
    }

    impl FacetableDocument for TestDoc {
        fn get_field(&self, name: &str) -> Option<String> {
            match name {
                "format" => self.format.map(str::to_string),
                "codec" => self.codec.map(str::to_string),
                "profile" => self.profile.map(str::to_string),
                _ => None,
            }
        }
    }

    fn make_refs<'a>(docs: &'a [TestDoc]) -> Vec<&'a dyn FacetableDocument> {
        docs.iter().map(|d| d as &dyn FacetableDocument).collect()
    }

    // ── Core tests ────────────────────────────────────────────────────────

    #[test]
    fn test_empty_documents() {
        let docs: Vec<TestDoc> = vec![];
        let refs = make_refs(&docs);
        let facet = build_hierarchical_facet(&refs, &["format", "codec"]);
        assert_eq!(facet.count_at_path(&["video"]), 0);
        assert!(facet.top_level().is_empty());
    }

    #[test]
    fn test_single_level_grouping() {
        let docs = vec![
            TestDoc::new(Some("video"), None, None),
            TestDoc::new(Some("video"), None, None),
            TestDoc::new(Some("audio"), None, None),
        ];
        let refs = make_refs(&docs);
        let facet = build_hierarchical_facet(&refs, &["format"]);

        assert_eq!(facet.count_at_path(&["video"]), 2);
        assert_eq!(facet.count_at_path(&["audio"]), 1);
        assert_eq!(facet.count_at_path(&["image"]), 0);
    }

    #[test]
    fn test_two_level_grouping() {
        let docs = vec![
            TestDoc::new(Some("video"), Some("av1"), None),
            TestDoc::new(Some("video"), Some("vp9"), None),
            TestDoc::new(Some("video"), Some("av1"), None),
            TestDoc::new(Some("audio"), Some("opus"), None),
        ];
        let refs = make_refs(&docs);
        let facet = build_hierarchical_facet(&refs, &["format", "codec"]);

        assert_eq!(facet.count_at_path(&["video"]), 3);
        assert_eq!(facet.count_at_path(&["video", "av1"]), 2);
        assert_eq!(facet.count_at_path(&["video", "vp9"]), 1);
        assert_eq!(facet.count_at_path(&["audio"]), 1);
        assert_eq!(facet.count_at_path(&["audio", "opus"]), 1);
    }

    #[test]
    fn test_three_level_grouping() {
        let docs = vec![
            TestDoc::new(Some("video"), Some("h264"), Some("baseline")),
            TestDoc::new(Some("video"), Some("h264"), Some("main")),
            TestDoc::new(Some("video"), Some("av1"), Some("main")),
        ];
        let refs = make_refs(&docs);
        let facet = build_hierarchical_facet(&refs, &["format", "codec", "profile"]);

        assert_eq!(facet.count_at_path(&["video"]), 3);
        assert_eq!(facet.count_at_path(&["video", "h264"]), 2);
        assert_eq!(facet.count_at_path(&["video", "h264", "baseline"]), 1);
        assert_eq!(facet.count_at_path(&["video", "h264", "main"]), 1);
        assert_eq!(facet.count_at_path(&["video", "av1", "main"]), 1);
    }

    #[test]
    fn test_missing_field_stops_traversal() {
        // Docs with no codec — they increment "format" but not "codec" children.
        let docs = vec![
            TestDoc::new(Some("video"), None, None),
            TestDoc::new(Some("video"), Some("av1"), None),
        ];
        let refs = make_refs(&docs);
        let facet = build_hierarchical_facet(&refs, &["format", "codec"]);

        // "video" is incremented by both docs.
        assert_eq!(facet.count_at_path(&["video"]), 2);
        // Only the second doc contributes to "av1".
        assert_eq!(facet.count_at_path(&["video", "av1"]), 1);
    }

    #[test]
    fn test_count_at_path_nonexistent() {
        let docs = vec![TestDoc::new(Some("video"), Some("av1"), None)];
        let refs = make_refs(&docs);
        let facet = build_hierarchical_facet(&refs, &["format", "codec"]);

        assert_eq!(facet.count_at_path(&["audio"]), 0);
        assert_eq!(facet.count_at_path(&["video", "vp9"]), 0);
        assert_eq!(facet.count_at_path(&["video", "av1", "deep"]), 0);
    }

    #[test]
    fn test_to_json_basic() {
        let docs = vec![
            TestDoc::new(Some("video"), Some("av1"), None),
            TestDoc::new(Some("audio"), Some("opus"), None),
        ];
        let refs = make_refs(&docs);
        let facet = build_hierarchical_facet(&refs, &["format", "codec"]);
        let json = facet.to_json();

        assert!(json.contains("\"children\""));
        assert!(json.contains("\"video\"") || json.contains("\"audio\""));
        assert!(json.contains("\"count\""));
        // Ensure valid JSON shape
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
    }

    #[test]
    fn test_to_json_special_characters() {
        let docs: Vec<TestDoc> = vec![TestDoc::new(Some("video/mp4"), Some("h.264"), None)];
        let refs = make_refs(&docs);
        let facet = build_hierarchical_facet(&refs, &["format", "codec"]);
        let json = facet.to_json();
        // Should not crash; special chars should be present (they're not JSON-special)
        assert!(json.contains("video/mp4"));
    }

    #[test]
    fn test_top_level_order_by_count_desc() {
        let docs = vec![
            TestDoc::new(Some("audio"), None, None),
            TestDoc::new(Some("video"), None, None),
            TestDoc::new(Some("video"), None, None),
            TestDoc::new(Some("video"), None, None),
        ];
        let refs = make_refs(&docs);
        let facet = build_hierarchical_facet(&refs, &["format"]);
        let top = facet.top_level();

        assert_eq!(top.len(), 2);
        assert_eq!(top[0].name, "video");
        assert_eq!(top[0].count, 3);
        assert_eq!(top[1].name, "audio");
        assert_eq!(top[1].count, 1);
    }

    #[test]
    fn test_empty_path_field_list() {
        // Building with no path fields — all docs contribute nothing.
        let docs = vec![TestDoc::new(Some("video"), None, None)];
        let refs = make_refs(&docs);
        let facet = build_hierarchical_facet(&refs, &[]);
        assert!(facet.top_level().is_empty());
    }

    #[test]
    fn test_facet_node_default() {
        let node = FacetNode::default();
        assert!(node.name.is_empty());
        assert_eq!(node.count, 0);
        assert!(node.children.is_empty());
    }

    #[test]
    fn test_hierarchical_facet_default() {
        let facet = HierarchicalFacet::default();
        assert!(facet.top_level().is_empty());
        let json = facet.to_json();
        assert_eq!(json, "{\"children\":[]}");
    }

    #[test]
    fn test_json_escaping_in_field_value() {
        struct QuoteDoc;
        impl FacetableDocument for QuoteDoc {
            fn get_field(&self, name: &str) -> Option<String> {
                if name == "tag" {
                    Some("say \"hello\"".to_string())
                } else {
                    None
                }
            }
        }
        let doc = QuoteDoc;
        let refs: Vec<&dyn FacetableDocument> = vec![&doc];
        let facet = build_hierarchical_facet(&refs, &["tag"]);
        let json = facet.to_json();
        assert!(json.contains("\\\"hello\\\""));
    }
}
