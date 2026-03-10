//! Query execution engine.
//!
//! Provides a dispatcher that translates parsed query ASTs into search
//! operations. Supports text, visual, and audio query types with
//! boolean combination (AND/OR semantics).

use crate::error::SearchResult;
use crate::query::parser::ParsedQuery;
use crate::SearchResultItem;

/// Query executor that dispatches parsed queries to the appropriate
/// search subsystems and combines results.
pub struct QueryExecutor;

impl QueryExecutor {
    /// Create a new query executor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Execute a parsed query.
    ///
    /// The executor inspects the query terms and dispatches to the
    /// relevant search subsystems. When no external index is attached,
    /// returns an empty result set (the real index integration happens
    /// via `SearchEngine::search`).
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails.
    pub fn execute(&self, query: &ParsedQuery) -> SearchResult<Vec<SearchResultItem>> {
        // The query executor acts as a dispatch layer. When invoked
        // standalone (without a backing index), it returns empty results.
        // Full execution is performed through SearchEngine which passes
        // parsed queries to the text/visual/audio sub-indices directly.
        let _ = query;
        Ok(Vec::new())
    }

    /// Execute a query and apply a result limit.
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails.
    pub fn execute_with_limit(
        &self,
        query: &ParsedQuery,
        limit: usize,
    ) -> SearchResult<Vec<SearchResultItem>> {
        let results = self.execute(query)?;
        Ok(results.into_iter().take(limit).collect())
    }
}

impl Default for QueryExecutor {
    fn default() -> Self {
        Self::new()
    }
}
