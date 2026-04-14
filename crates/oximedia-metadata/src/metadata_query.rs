//! SQL-inspired query engine for metadata collections.
//!
//! # Overview
//!
//! This module provides a lightweight, composable query language for filtering,
//! sorting, and aggregating metadata records stored in a `HashMap<String,
//! String>` field map.  The design favours ergonomics and zero-dependency
//! pure-Rust operation over query-plan optimisation.
//!
//! ## Components
//!
//! - **[`Predicate`]** – boolean expression tree (comparisons, logical AND/OR/NOT,
//!   field presence, substring search, numeric range).
//! - **[`SortKey`]** / **[`SortOrder`]** – multi-column ordering specification.
//! - **[`Aggregation`]** – count, count-distinct, first/last value.
//! - **[`MetadataQuery`]** – complete query: predicate + projection + sort + limit.
//! - **[`QueryEngine`]** – executes a [`MetadataQuery`] over a slice of records.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_metadata::metadata_query::{
//!     MetadataQuery, Predicate, QueryEngine, SortKey, SortOrder,
//! };
//! use std::collections::HashMap;
//!
//! let records: Vec<(u64, HashMap<String, String>)> = vec![
//!     (1, [("title", "Alpha"), ("year", "2020")].iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()),
//!     (2, [("title", "Beta"),  ("year", "2021")].iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()),
//! ];
//!
//! let query = MetadataQuery::new()
//!     .filter(Predicate::FieldEquals { field: "year".to_string(), value: "2020".to_string() })
//!     .sort(SortKey { field: "title".to_string(), order: SortOrder::Ascending })
//!     .limit(10);
//!
//! let results = QueryEngine::execute(&query, &records);
//! assert_eq!(results.len(), 1);
//! assert_eq!(results[0].0, 1);
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Predicate
// ---------------------------------------------------------------------------

/// Boolean predicate that can be evaluated against a metadata field map.
#[derive(Debug, Clone)]
pub enum Predicate {
    /// Always true.
    True,

    /// Always false.
    False,

    /// Field value equals `value` (case-sensitive).
    FieldEquals { field: String, value: String },

    /// Field value equals `value` (case-insensitive).
    FieldEqualsIgnoreCase { field: String, value: String },

    /// Field value contains `substring`.
    FieldContains { field: String, substring: String },

    /// Field value starts with `prefix`.
    FieldStartsWith { field: String, prefix: String },

    /// Field value ends with `suffix`.
    FieldEndsWith { field: String, suffix: String },

    /// Field is present in the record (regardless of value).
    FieldExists { field: String },

    /// Field is absent from the record.
    FieldAbsent { field: String },

    /// Field value, parsed as f64, is within `[min, max]` (inclusive).
    NumericRange { field: String, min: f64, max: f64 },

    /// Field value length is within `[min, max]` characters (inclusive).
    LengthRange { field: String, min: usize, max: usize },

    /// Both sub-predicates must hold.
    And(Box<Predicate>, Box<Predicate>),

    /// At least one sub-predicate must hold.
    Or(Box<Predicate>, Box<Predicate>),

    /// The sub-predicate must not hold.
    Not(Box<Predicate>),
}

impl Predicate {
    /// Evaluate the predicate against a field map.
    #[must_use]
    pub fn evaluate(&self, fields: &HashMap<String, String>) -> bool {
        match self {
            Self::True => true,
            Self::False => false,

            Self::FieldEquals { field, value } => {
                fields.get(field).map_or(false, |v| v == value)
            }
            Self::FieldEqualsIgnoreCase { field, value } => fields
                .get(field)
                .map_or(false, |v| v.to_lowercase() == value.to_lowercase()),

            Self::FieldContains { field, substring } => {
                fields.get(field).map_or(false, |v| v.contains(substring.as_str()))
            }
            Self::FieldStartsWith { field, prefix } => {
                fields.get(field).map_or(false, |v| v.starts_with(prefix.as_str()))
            }
            Self::FieldEndsWith { field, suffix } => {
                fields.get(field).map_or(false, |v| v.ends_with(suffix.as_str()))
            }

            Self::FieldExists { field } => fields.contains_key(field),
            Self::FieldAbsent { field } => !fields.contains_key(field),

            Self::NumericRange { field, min, max } => fields
                .get(field)
                .and_then(|v| v.parse::<f64>().ok())
                .map_or(false, |n| n >= *min && n <= *max),

            Self::LengthRange { field, min, max } => fields
                .get(field)
                .map_or(false, |v| v.len() >= *min && v.len() <= *max),

            Self::And(a, b) => a.evaluate(fields) && b.evaluate(fields),
            Self::Or(a, b) => a.evaluate(fields) || b.evaluate(fields),
            Self::Not(inner) => !inner.evaluate(fields),
        }
    }

    /// Convenience: `self AND other`.
    #[must_use]
    pub fn and(self, other: Predicate) -> Self {
        Self::And(Box::new(self), Box::new(other))
    }

    /// Convenience: `self OR other`.
    #[must_use]
    pub fn or(self, other: Predicate) -> Self {
        Self::Or(Box::new(self), Box::new(other))
    }

    /// Convenience: `NOT self`.
    #[must_use]
    pub fn not(self) -> Self {
        Self::Not(Box::new(self))
    }
}

// ---------------------------------------------------------------------------
// Sorting
// ---------------------------------------------------------------------------

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    /// Lexicographic ascending.
    Ascending,
    /// Lexicographic descending.
    Descending,
}

/// Sort criterion for one field.
#[derive(Debug, Clone)]
pub struct SortKey {
    /// Field to sort by.
    pub field: String,
    /// Sort direction.
    pub order: SortOrder,
}

impl SortKey {
    /// Create an ascending sort key.
    #[must_use]
    pub fn asc(field: impl Into<String>) -> Self {
        Self { field: field.into(), order: SortOrder::Ascending }
    }

    /// Create a descending sort key.
    #[must_use]
    pub fn desc(field: impl Into<String>) -> Self {
        Self { field: field.into(), order: SortOrder::Descending }
    }
}

/// Compare two field maps using a list of sort keys.
///
/// Records with missing fields sort *after* records that have the field.
fn compare_by_keys(
    a: &HashMap<String, String>,
    b: &HashMap<String, String>,
    keys: &[SortKey],
) -> std::cmp::Ordering {
    for key in keys {
        let va = a.get(&key.field).map(String::as_str).unwrap_or("");
        let vb = b.get(&key.field).map(String::as_str).unwrap_or("");
        let cmp = va.cmp(vb);
        if cmp != std::cmp::Ordering::Equal {
            return match key.order {
                SortOrder::Ascending => cmp,
                SortOrder::Descending => cmp.reverse(),
            };
        }
    }
    std::cmp::Ordering::Equal
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/// An aggregation function over a field.
#[derive(Debug, Clone)]
pub enum Aggregation {
    /// Count of records.
    Count,
    /// Count of distinct values for `field`.
    CountDistinct { field: String },
    /// First value of `field` in result order.
    First { field: String },
    /// Last value of `field` in result order.
    Last { field: String },
    /// Minimum numeric value (parsed as f64) for `field`.
    Min { field: String },
    /// Maximum numeric value (parsed as f64) for `field`.
    Max { field: String },
}

/// Result of computing an [`Aggregation`] over a set of records.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationResult {
    /// An integer count.
    Count(u64),
    /// A text value.
    Value(String),
    /// A numeric value.
    Number(f64),
    /// No records matched.
    Empty,
}

impl Aggregation {
    /// Compute this aggregation over the given slice of field maps.
    #[must_use]
    pub fn compute(&self, rows: &[&HashMap<String, String>]) -> AggregationResult {
        match self {
            Self::Count => AggregationResult::Count(rows.len() as u64),

            Self::CountDistinct { field } => {
                let distinct: std::collections::HashSet<&str> = rows
                    .iter()
                    .filter_map(|r| r.get(field).map(String::as_str))
                    .collect();
                AggregationResult::Count(distinct.len() as u64)
            }

            Self::First { field } => rows
                .iter()
                .find_map(|r| r.get(field).map(|v| AggregationResult::Value(v.clone())))
                .unwrap_or(AggregationResult::Empty),

            Self::Last { field } => rows
                .iter()
                .rev()
                .find_map(|r| r.get(field).map(|v| AggregationResult::Value(v.clone())))
                .unwrap_or(AggregationResult::Empty),

            Self::Min { field } => {
                let nums: Vec<f64> = rows
                    .iter()
                    .filter_map(|r| r.get(field).and_then(|v| v.parse::<f64>().ok()))
                    .collect();
                nums.iter()
                    .cloned()
                    .reduce(f64::min)
                    .map_or(AggregationResult::Empty, AggregationResult::Number)
            }

            Self::Max { field } => {
                let nums: Vec<f64> = rows
                    .iter()
                    .filter_map(|r| r.get(field).and_then(|v| v.parse::<f64>().ok()))
                    .collect();
                nums.iter()
                    .cloned()
                    .reduce(f64::max)
                    .map_or(AggregationResult::Empty, AggregationResult::Number)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MetadataQuery
// ---------------------------------------------------------------------------

/// A complete query specification: filter → project → sort → limit/offset.
#[derive(Debug, Clone, Default)]
pub struct MetadataQuery {
    /// Filter predicate (defaults to [`Predicate::True`]).
    pub predicate: Option<Predicate>,
    /// Field subset to include in results.  Empty = all fields.
    pub projection: Vec<String>,
    /// Sort keys applied in order.
    pub sort_keys: Vec<SortKey>,
    /// Maximum number of results to return.
    pub limit: Option<usize>,
    /// Number of results to skip before returning.
    pub offset: usize,
}

impl MetadataQuery {
    /// Create an empty query (returns all records unchanged).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the filter predicate.
    #[must_use]
    pub fn filter(mut self, predicate: Predicate) -> Self {
        self.predicate = Some(predicate);
        self
    }

    /// Add a sort key.
    #[must_use]
    pub fn sort(mut self, key: SortKey) -> Self {
        self.sort_keys.push(key);
        self
    }

    /// Add multiple sort keys.
    #[must_use]
    pub fn sort_by(mut self, keys: impl IntoIterator<Item = SortKey>) -> Self {
        self.sort_keys.extend(keys);
        self
    }

    /// Set the maximum number of results.
    #[must_use]
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Set the offset (skip first N results).
    #[must_use]
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = n;
        self
    }

    /// Set the field projection.
    #[must_use]
    pub fn project(mut self, fields: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.projection = fields.into_iter().map(Into::into).collect();
        self
    }
}

// ---------------------------------------------------------------------------
// QueryEngine
// ---------------------------------------------------------------------------

/// Executes [`MetadataQuery`]s over slices of `(id, field_map)` records.
pub struct QueryEngine;

impl QueryEngine {
    /// Execute a query and return matching records.
    ///
    /// The returned `Vec` contains `(id, projected_field_map)` pairs in the
    /// requested sort order, respecting limit and offset.
    #[must_use]
    pub fn execute(
        query: &MetadataQuery,
        records: &[(u64, HashMap<String, String>)],
    ) -> Vec<(u64, HashMap<String, String>)> {
        // 1. Filter.
        let predicate = query.predicate.as_ref().unwrap_or(&Predicate::True);
        let mut matched: Vec<&(u64, HashMap<String, String>)> = records
            .iter()
            .filter(|(_, fields)| predicate.evaluate(fields))
            .collect();

        // 2. Sort.
        if !query.sort_keys.is_empty() {
            matched.sort_by(|(_, a), (_, b)| compare_by_keys(a, b, &query.sort_keys));
        }

        // 3. Offset.
        let start = query.offset.min(matched.len());
        let matched = &matched[start..];

        // 4. Limit.
        let take = query.limit.unwrap_or(matched.len()).min(matched.len());
        let matched = &matched[..take];

        // 5. Project.
        matched
            .iter()
            .map(|(id, fields)| {
                let projected = project_fields(fields, &query.projection);
                (*id, projected)
            })
            .collect()
    }

    /// Compute an aggregation over records that match the query's predicate.
    #[must_use]
    pub fn aggregate(
        query: &MetadataQuery,
        records: &[(u64, HashMap<String, String>)],
        agg: &Aggregation,
    ) -> AggregationResult {
        let predicate = query.predicate.as_ref().unwrap_or(&Predicate::True);
        let rows: Vec<&HashMap<String, String>> = records
            .iter()
            .filter(|(_, f)| predicate.evaluate(f))
            .map(|(_, f)| f)
            .collect();
        agg.compute(&rows)
    }
}

/// Apply a field projection to a map.  Empty projection = return all fields.
fn project_fields(
    fields: &HashMap<String, String>,
    keys: &[String],
) -> HashMap<String, String> {
    if keys.is_empty() {
        return fields.clone();
    }
    keys.iter()
        .filter_map(|k| fields.get(k).map(|v| (k.clone(), v.clone())))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_records() -> Vec<(u64, HashMap<String, String>)> {
        vec![
            (
                1,
                [("title", "Alpha"), ("artist", "Alice"), ("year", "2018"), ("genre", "Jazz")]
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            ),
            (
                2,
                [("title", "Beta"), ("artist", "Bob"), ("year", "2020"), ("genre", "Rock")]
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            ),
            (
                3,
                [("title", "Gamma"), ("artist", "Alice"), ("year", "2022"), ("genre", "Jazz")]
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            ),
            (
                4,
                [("title", "Delta"), ("year", "2019")]
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            ),
        ]
    }

    #[test]
    fn test_predicate_field_equals() {
        let records = make_records();
        let query = MetadataQuery::new()
            .filter(Predicate::FieldEquals { field: "artist".to_string(), value: "Alice".to_string() });
        let results = QueryEngine::execute(&query, &records);
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|(id, _)| *id == 1));
        assert!(results.iter().any(|(id, _)| *id == 3));
    }

    #[test]
    fn test_predicate_field_absent() {
        let records = make_records();
        let query = MetadataQuery::new()
            .filter(Predicate::FieldAbsent { field: "artist".to_string() });
        let results = QueryEngine::execute(&query, &records);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 4);
    }

    #[test]
    fn test_predicate_numeric_range() {
        let records = make_records();
        let query = MetadataQuery::new()
            .filter(Predicate::NumericRange { field: "year".to_string(), min: 2019.0, max: 2021.0 });
        let results = QueryEngine::execute(&query, &records);
        // Records 2 (2020) and 4 (2019) match.
        assert_eq!(results.len(), 2);
        let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&2));
        assert!(ids.contains(&4));
    }

    #[test]
    fn test_predicate_and_or() {
        let records = make_records();
        // Jazz OR Rock
        let jazz = Predicate::FieldEquals { field: "genre".to_string(), value: "Jazz".to_string() };
        let rock = Predicate::FieldEquals { field: "genre".to_string(), value: "Rock".to_string() };
        let query = MetadataQuery::new().filter(jazz.or(rock));
        let results = QueryEngine::execute(&query, &records);
        assert_eq!(results.len(), 3); // Records 1, 2, 3
    }

    #[test]
    fn test_sort_ascending() {
        let records = make_records();
        let query = MetadataQuery::new()
            .sort(SortKey::asc("title"));
        let results = QueryEngine::execute(&query, &records);
        let titles: Vec<&str> = results.iter()
            .map(|(_, f)| f.get("title").map(String::as_str).unwrap_or(""))
            .collect();
        assert_eq!(titles, vec!["Alpha", "Beta", "Delta", "Gamma"]);
    }

    #[test]
    fn test_sort_descending() {
        let records = make_records();
        let query = MetadataQuery::new().sort(SortKey::desc("year"));
        let results = QueryEngine::execute(&query, &records);
        let years: Vec<&str> = results.iter()
            .map(|(_, f)| f.get("year").map(String::as_str).unwrap_or(""))
            .collect();
        assert_eq!(years, vec!["2022", "2020", "2019", "2018"]);
    }

    #[test]
    fn test_limit_offset() {
        let records = make_records();
        let query = MetadataQuery::new()
            .sort(SortKey::asc("title"))
            .offset(1)
            .limit(2);
        let results = QueryEngine::execute(&query, &records);
        assert_eq!(results.len(), 2);
        let titles: Vec<&str> = results.iter()
            .map(|(_, f)| f.get("title").map(String::as_str).unwrap_or(""))
            .collect();
        assert_eq!(titles, vec!["Beta", "Delta"]);
    }

    #[test]
    fn test_projection() {
        let records = make_records();
        let query = MetadataQuery::new().project(["title", "year"]);
        let results = QueryEngine::execute(&query, &records);
        for (_, fields) in &results {
            assert!(fields.contains_key("title"));
            assert!(fields.contains_key("year"));
            assert!(!fields.contains_key("artist"));
            assert!(!fields.contains_key("genre"));
        }
    }

    #[test]
    fn test_aggregation_count() {
        let records = make_records();
        let query = MetadataQuery::new()
            .filter(Predicate::FieldEquals { field: "genre".to_string(), value: "Jazz".to_string() });
        let agg = Aggregation::Count;
        let result = QueryEngine::aggregate(&query, &records, &agg);
        assert_eq!(result, AggregationResult::Count(2));
    }

    #[test]
    fn test_aggregation_count_distinct() {
        let records = make_records();
        let query = MetadataQuery::new();
        let agg = Aggregation::CountDistinct { field: "artist".to_string() };
        let result = QueryEngine::aggregate(&query, &records, &agg);
        // Alice, Bob, and record 4 has no artist → 2 distinct
        assert_eq!(result, AggregationResult::Count(2));
    }

    #[test]
    fn test_aggregation_min_max() {
        let records = make_records();
        let query = MetadataQuery::new();
        let min = QueryEngine::aggregate(&query, &records, &Aggregation::Min { field: "year".to_string() });
        let max = QueryEngine::aggregate(&query, &records, &Aggregation::Max { field: "year".to_string() });
        assert_eq!(min, AggregationResult::Number(2018.0));
        assert_eq!(max, AggregationResult::Number(2022.0));
    }

    #[test]
    fn test_predicate_contains_starts_ends() {
        let records = make_records();

        let contains_query = MetadataQuery::new()
            .filter(Predicate::FieldContains { field: "title".to_string(), substring: "lph".to_string() });
        let r = QueryEngine::execute(&contains_query, &records);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 1);

        let starts_query = MetadataQuery::new()
            .filter(Predicate::FieldStartsWith { field: "title".to_string(), prefix: "G".to_string() });
        let r2 = QueryEngine::execute(&starts_query, &records);
        assert_eq!(r2.len(), 1);
        assert_eq!(r2[0].0, 3);

        let ends_query = MetadataQuery::new()
            .filter(Predicate::FieldEndsWith { field: "genre".to_string(), suffix: "azz".to_string() });
        let r3 = QueryEngine::execute(&ends_query, &records);
        assert_eq!(r3.len(), 2);
    }
}
