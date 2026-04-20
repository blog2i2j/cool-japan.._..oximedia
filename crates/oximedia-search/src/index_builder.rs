//! Index builder — in-memory document index with per-field search support.
#![allow(dead_code)]

use std::collections::HashMap;

// ── IndexField ────────────────────────────────────────────────────────────────

/// Describes one field within an indexed document.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexField {
    /// Field name, e.g. `"title"`.
    pub name: String,
    /// Stored value (always text for simplicity).
    pub value: String,
    /// Whether this field is searchable (indexed).
    pub searchable: bool,
    /// Field boost factor for relevance scoring.
    pub boost: f32,
}

impl IndexField {
    /// Create a new searchable field with default boost.
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
            searchable: true,
            boost: 1.0,
        }
    }

    /// Create a stored-only (non-searchable) field.
    #[must_use]
    pub fn stored(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
            searchable: false,
            boost: 1.0,
        }
    }

    /// Return the field with a custom boost factor.
    #[must_use]
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }
}

// ── IndexDocument ─────────────────────────────────────────────────────────────

/// A document consisting of named fields to be added to the index.
#[derive(Debug, Clone, Default)]
pub struct IndexDocument {
    /// Document identifier (arbitrary string).
    pub id: String,
    /// Fields comprising this document.
    pub fields: Vec<IndexField>,
}

impl IndexDocument {
    /// Create a document with the given id.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            fields: Vec::new(),
        }
    }

    /// Add a field and return `self` for chaining.
    #[must_use]
    pub fn with_field(mut self, field: IndexField) -> Self {
        self.fields.push(field);
        self
    }
}

// ── SearchHit ─────────────────────────────────────────────────────────────────

/// A single search hit returned from `DocumentIndex::search_field`.
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// Document ID.
    pub doc_id: String,
    /// The field value that matched.
    pub field_value: String,
    /// The applied boost for this hit.
    pub boost: f32,
}

// ── DocumentIndex ─────────────────────────────────────────────────────────────

/// An in-memory document index keyed by field name.
///
/// Documents are added via [`Self::add_document`] and later searched with
/// [`Self::search_field`].
#[derive(Debug, Default)]
pub struct DocumentIndex {
    /// Maps `(field_name, term)` → list of `(doc_id, boost)`.
    inverted: HashMap<(String, String), Vec<(String, f32)>>,
    /// Number of unique documents.
    doc_count: usize,
    /// Number of unique indexed fields across all documents.
    field_count: usize,
    /// Set of known field names.
    known_fields: std::collections::HashSet<String>,
}

impl DocumentIndex {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a document to the index.
    pub fn add_document(&mut self, doc: &IndexDocument) {
        self.doc_count += 1;

        for field in &doc.fields {
            if !field.searchable {
                continue;
            }

            self.known_fields.insert(field.name.clone());

            // Simple tokenisation: split on whitespace, lowercase
            for term in field.value.split_whitespace() {
                let key = (field.name.clone(), term.to_ascii_lowercase());
                self.inverted
                    .entry(key)
                    .or_default()
                    .push((doc.id.clone(), field.boost));
            }

            self.field_count += 1;
        }
    }

    /// Total number of documents that have been indexed.
    #[must_use]
    pub fn doc_count(&self) -> usize {
        self.doc_count
    }

    /// Total number of field occurrences across all indexed documents.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.field_count
    }

    /// Search a specific field for a term.  Returns all matching hits,
    /// sorted by descending boost.
    #[must_use]
    pub fn search_field(&self, field: &str, term: &str) -> Vec<SearchHit> {
        let key = (field.to_string(), term.to_ascii_lowercase());
        let mut hits: Vec<SearchHit> = self
            .inverted
            .get(&key)
            .map(|entries| {
                entries
                    .iter()
                    .map(|(doc_id, boost)| SearchHit {
                        doc_id: doc_id.clone(),
                        field_value: term.to_string(),
                        boost: *boost,
                    })
                    .collect()
            })
            .unwrap_or_default();

        hits.sort_by(|a, b| {
            b.boost
                .partial_cmp(&a.boost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits
    }

    /// Search across ALL searchable fields for a term.
    #[must_use]
    pub fn search_all_fields(&self, term: &str) -> Vec<SearchHit> {
        let mut hits = Vec::new();
        for field in &self.known_fields {
            hits.extend(self.search_field(field, term));
        }
        hits.sort_by(|a, b| {
            b.boost
                .partial_cmp(&a.boost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits
    }

    /// Return the set of known (indexed) field names.
    #[must_use]
    pub fn known_fields(&self) -> Vec<&str> {
        let mut fields: Vec<&str> = self.known_fields.iter().map(String::as_str).collect();
        fields.sort_unstable();
        fields
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_doc(id: &str) -> IndexDocument {
        IndexDocument::new(id)
            .with_field(IndexField::new("title", "Hello World"))
            .with_field(IndexField::new("description", "A great video about nature"))
            .with_field(IndexField::stored("path", "/media/clip.mp4"))
    }

    #[test]
    fn test_index_field_new() {
        let f = IndexField::new("title", "Hello");
        assert!(f.searchable);
        assert!((f.boost - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_index_field_stored_not_searchable() {
        let path = std::env::temp_dir()
            .join("oximedia-search-indexbuilder-a.mp4")
            .to_string_lossy()
            .into_owned();
        let f = IndexField::stored("path", path);
        assert!(!f.searchable);
    }

    #[test]
    fn test_index_field_with_boost() {
        let f = IndexField::new("title", "Hi").with_boost(3.0);
        assert!((f.boost - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_document_index_doc_count() {
        let mut idx = DocumentIndex::new();
        idx.add_document(&sample_doc("1"));
        idx.add_document(&sample_doc("2"));
        assert_eq!(idx.doc_count(), 2);
    }

    #[test]
    fn test_document_index_field_count() {
        let mut idx = DocumentIndex::new();
        idx.add_document(&sample_doc("1"));
        // sample_doc has 2 searchable fields (path is stored-only)
        assert_eq!(idx.field_count(), 2);
    }

    #[test]
    fn test_document_index_search_field_found() {
        let mut idx = DocumentIndex::new();
        idx.add_document(&sample_doc("doc1"));
        let hits = idx.search_field("title", "hello");
        assert!(!hits.is_empty());
        assert_eq!(hits[0].doc_id, "doc1");
    }

    #[test]
    fn test_document_index_search_field_not_found() {
        let mut idx = DocumentIndex::new();
        idx.add_document(&sample_doc("doc1"));
        let hits = idx.search_field("title", "elephant");
        assert!(hits.is_empty());
    }

    #[test]
    fn test_document_index_search_case_insensitive() {
        let mut idx = DocumentIndex::new();
        idx.add_document(&sample_doc("doc1"));
        let hits_lower = idx.search_field("title", "world");
        let hits_upper = idx.search_field("title", "WORLD");
        assert_eq!(hits_lower.len(), hits_upper.len());
    }

    #[test]
    fn test_document_index_stored_field_not_searchable() {
        let mut idx = DocumentIndex::new();
        idx.add_document(&sample_doc("doc1"));
        // "path" is stored-only, should not be searchable
        let hits = idx.search_field("path", "media");
        assert!(hits.is_empty());
    }

    #[test]
    fn test_document_index_search_all_fields() {
        let mut idx = DocumentIndex::new();
        idx.add_document(&sample_doc("doc1"));
        let hits = idx.search_all_fields("great");
        assert!(!hits.is_empty());
        assert_eq!(hits[0].doc_id, "doc1");
    }

    #[test]
    fn test_document_index_known_fields_sorted() {
        let mut idx = DocumentIndex::new();
        idx.add_document(&sample_doc("doc1"));
        let fields = idx.known_fields();
        // Should contain "title" and "description" in sorted order
        assert!(fields.contains(&"description"));
        assert!(fields.contains(&"title"));
        assert!(fields.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_document_index_boost_ordering() {
        let mut idx = DocumentIndex::new();
        let doc =
            IndexDocument::new("d1").with_field(IndexField::new("title", "rust").with_boost(5.0));
        let doc2 =
            IndexDocument::new("d2").with_field(IndexField::new("title", "rust").with_boost(1.0));
        idx.add_document(&doc);
        idx.add_document(&doc2);
        let hits = idx.search_field("title", "rust");
        assert!(!hits.is_empty());
        assert_eq!(hits[0].doc_id, "d1"); // highest boost first
    }

    #[test]
    fn test_document_index_multiple_terms() {
        let mut idx = DocumentIndex::new();
        idx.add_document(&sample_doc("doc1"));
        let hits_a = idx.search_field("description", "great");
        let hits_b = idx.search_field("description", "video");
        assert!(!hits_a.is_empty());
        assert!(!hits_b.is_empty());
    }
}
