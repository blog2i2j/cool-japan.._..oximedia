//! Faceted search implementation.

use crate::error::SearchResult;
use serde::{Deserialize, Serialize};

/// Faceted search query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetQuery {
    /// Field to facet on
    pub field: String,
    /// Maximum number of facet values
    pub limit: usize,
}

/// Faceted search engine.
///
/// Maintains per-field value counts and produces facet aggregations
/// for filtered result sets.
pub struct FacetedSearch {
    /// Per-field value counts: field_name -> { value -> count }.
    field_values: std::collections::HashMap<String, std::collections::HashMap<String, usize>>,
}

impl FacetedSearch {
    /// Create a new faceted search engine.
    #[must_use]
    pub fn new() -> Self {
        Self {
            field_values: std::collections::HashMap::new(),
        }
    }

    /// Index a field value (increments the count for `field`:`value`).
    pub fn add_value(&mut self, field: &str, value: &str) {
        *self
            .field_values
            .entry(field.to_string())
            .or_default()
            .entry(value.to_string())
            .or_insert(0) += 1;
    }

    /// Execute a faceted search, returning the top `query.limit` values
    /// for the specified field, sorted by count descending.
    ///
    /// # Errors
    ///
    /// Returns an error if search fails.
    pub fn search(&self, query: &FacetQuery) -> SearchResult<Vec<FacetValue>> {
        let values = match self.field_values.get(&query.field) {
            Some(map) => {
                let mut sorted: Vec<_> = map
                    .iter()
                    .map(|(v, &c)| FacetValue {
                        value: v.clone(),
                        count: c,
                    })
                    .collect();
                sorted.sort_by(|a, b| b.count.cmp(&a.count));
                sorted.truncate(query.limit);
                sorted
            }
            None => Vec::new(),
        };
        Ok(values)
    }

    /// Return all indexed field names.
    #[must_use]
    pub fn fields(&self) -> Vec<String> {
        self.field_values.keys().cloned().collect()
    }
}

impl Default for FacetedSearch {
    fn default() -> Self {
        Self::new()
    }
}

/// Facet value with count
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetValue {
    /// Value
    pub value: String,
    /// Count
    pub count: usize,
}
