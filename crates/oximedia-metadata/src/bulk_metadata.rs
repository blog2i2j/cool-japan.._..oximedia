//! Bulk metadata operations: batch read, batch write, field projection, and
//! bulk tag transformations across collections of metadata records.
//!
//! # Overview
//!
//! Working with media libraries often requires applying the same operation to
//! hundreds or thousands of files.  This module provides:
//!
//! - **[`MetadataCollection`]** – an owned store of `(id, Metadata)` pairs
//!   with O(1) lookup by id.
//! - **[`FieldProjection`]** – select a subset of fields from each record,
//!   returning lightweight `HashMap<String, String>` maps.
//! - **[`BulkOperation`]** and **[`BulkOperator`]** – apply a sequence of
//!   typed operations (set, remove, rename, copy, prefix/suffix append) to
//!   every record in a collection, collecting per-record results.
//! - **[`BatchReadResult`]** / **[`BatchWriteResult`]** – aggregated success /
//!   error reporting for bulk I/O operations.
//!
//! All functions are pure-Rust, allocation-only (no unsafe), and work in
//! O(n · m) time where n is the number of records and m is the number of
//! fields or operations.

#![allow(dead_code)]

use crate::{Error, Metadata, MetadataFormat, MetadataValue};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// MetadataCollection
// ---------------------------------------------------------------------------

/// A keyed, ordered collection of [`Metadata`] records.
///
/// Records are stored in insertion order and addressed by a `u64` id.
#[derive(Debug, Default)]
pub struct MetadataCollection {
    /// Ordered list of (id, metadata) pairs.
    records: Vec<(u64, Metadata)>,
    /// Fast id → index lookup.
    index: HashMap<u64, usize>,
}

impl MetadataCollection {
    /// Create an empty collection.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a record.  Returns `true` if the id was new.
    pub fn insert(&mut self, id: u64, metadata: Metadata) -> bool {
        if let Some(&pos) = self.index.get(&id) {
            self.records[pos].1 = metadata;
            false
        } else {
            let pos = self.records.len();
            self.records.push((id, metadata));
            self.index.insert(id, pos);
            true
        }
    }

    /// Remove a record by id.  Returns the removed metadata, if any.
    ///
    /// This performs a swap-remove internally, so insertion order is **not**
    /// preserved after a removal.
    pub fn remove(&mut self, id: u64) -> Option<Metadata> {
        let pos = *self.index.get(&id)?;
        // Swap-remove: move the last element to `pos`.
        let last_id = self.records.last().map(|(i, _)| *i)?;
        let (_, removed) = self.records.swap_remove(pos);
        self.index.remove(&id);
        if pos < self.records.len() {
            // Update the index entry for the element that was swapped in.
            self.index.insert(last_id, pos);
        }
        Some(removed)
    }

    /// Retrieve a record by id.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&Metadata> {
        let &pos = self.index.get(&id)?;
        Some(&self.records[pos].1)
    }

    /// Retrieve a mutable reference to a record by id.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut Metadata> {
        let &pos = self.index.get(&id)?;
        Some(&mut self.records[pos].1)
    }

    /// Number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// `true` if the collection contains no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Iterate over `(id, &Metadata)` pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &Metadata)> {
        self.records.iter().map(|(id, m)| (*id, m))
    }

    /// Iterate over ids in insertion order.
    pub fn ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.records.iter().map(|(id, _)| *id)
    }

    /// Collect all records as a `Vec<(u64, &Metadata)>`.
    #[must_use]
    pub fn as_slice(&self) -> Vec<(u64, &Metadata)> {
        self.records.iter().map(|(id, m)| (*id, m)).collect()
    }
}

// ---------------------------------------------------------------------------
// FieldProjection
// ---------------------------------------------------------------------------

/// Selects a fixed subset of fields from metadata records, returning only
/// those fields (as `String` text representations) for each record.
///
/// This is useful when a downstream consumer only needs certain tags, e.g.
/// title + artist + year for display in a playlist.
#[derive(Debug, Clone)]
pub struct FieldProjection {
    /// Ordered list of field keys to include.
    keys: Vec<String>,
}

impl FieldProjection {
    /// Create a new projection that selects `keys`.
    pub fn new(keys: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            keys: keys.into_iter().map(Into::into).collect(),
        }
    }

    /// Project a single `Metadata` record.
    ///
    /// Returns a map from field key → text representation.  Fields that are
    /// absent in `metadata` are omitted from the result map.
    #[must_use]
    pub fn project_one(&self, metadata: &Metadata) -> HashMap<String, String> {
        let mut out = HashMap::with_capacity(self.keys.len());
        for key in &self.keys {
            if let Some(val) = metadata.get(key) {
                out.insert(key.clone(), value_to_string(val));
            }
        }
        out
    }

    /// Project every record in a collection.
    ///
    /// Returns a `Vec<(id, projected_map)>` in the same order as
    /// [`MetadataCollection::iter`].
    #[must_use]
    pub fn project_collection(
        &self,
        collection: &MetadataCollection,
    ) -> Vec<(u64, HashMap<String, String>)> {
        collection
            .iter()
            .map(|(id, m)| (id, self.project_one(m)))
            .collect()
    }

    /// Keys selected by this projection.
    #[must_use]
    pub fn keys(&self) -> &[String] {
        &self.keys
    }
}

// ---------------------------------------------------------------------------
// BulkOperation
// ---------------------------------------------------------------------------

/// A single, typed operation that can be applied to a metadata record.
#[derive(Debug, Clone)]
pub enum BulkOperation {
    /// Set (overwrite) a field to a fixed text value.
    Set { key: String, value: String },

    /// Remove a field, if present.
    Remove { key: String },

    /// Rename a field: copy old key → new key, then remove old key.
    Rename { from: String, to: String },

    /// Copy a field value to another key without removing the original.
    Copy { from: String, to: String },

    /// Append a fixed suffix to the existing text value of a field.
    /// No-op if the field is absent.
    AppendSuffix { key: String, suffix: String },

    /// Prepend a fixed prefix to the existing text value of a field.
    /// No-op if the field is absent.
    PrependPrefix { key: String, prefix: String },

    /// Clear all fields (produces an empty metadata record with the same format).
    ClearAll,
}

/// Result of applying a single [`BulkOperation`] to a record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationOutcome {
    /// The operation changed the record.
    Applied,
    /// The operation was a no-op (e.g., field absent for `AppendSuffix`).
    NoOp,
    /// The operation failed.
    Failed(String),
}

// ---------------------------------------------------------------------------
// BulkOperator
// ---------------------------------------------------------------------------

/// Applies a pipeline of [`BulkOperation`]s to every record in a
/// [`MetadataCollection`], collecting per-record outcomes.
pub struct BulkOperator {
    operations: Vec<BulkOperation>,
}

impl BulkOperator {
    /// Create a new operator with the given ordered operation pipeline.
    #[must_use]
    pub fn new(operations: Vec<BulkOperation>) -> Self {
        Self { operations }
    }

    /// Apply all operations to every record in `collection` **in-place**.
    ///
    /// Returns a `Vec<(id, Vec<OperationOutcome>)>` — one `Vec<Outcome>`
    /// per record, in the same order as `collection`.
    pub fn apply(&self, collection: &mut MetadataCollection) -> Vec<(u64, Vec<OperationOutcome>)> {
        let ids: Vec<u64> = collection.ids().collect();
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(meta) = collection.get_mut(id) {
                let outcomes = apply_pipeline(&self.operations, meta);
                results.push((id, outcomes));
            }
        }
        results
    }

    /// Apply all operations to a single `Metadata` record, returning outcomes.
    #[must_use]
    pub fn apply_one(&self, metadata: &mut Metadata) -> Vec<OperationOutcome> {
        apply_pipeline(&self.operations, metadata)
    }
}

/// Apply an ordered list of operations to a single record.
fn apply_pipeline(ops: &[BulkOperation], meta: &mut Metadata) -> Vec<OperationOutcome> {
    ops.iter().map(|op| apply_one_op(op, meta)).collect()
}

/// Apply a single operation to a record and return the outcome.
fn apply_one_op(op: &BulkOperation, meta: &mut Metadata) -> OperationOutcome {
    match op {
        BulkOperation::Set { key, value } => {
            meta.insert(key.clone(), MetadataValue::Text(value.clone()));
            OperationOutcome::Applied
        }
        BulkOperation::Remove { key } => {
            if meta.remove(key).is_some() {
                OperationOutcome::Applied
            } else {
                OperationOutcome::NoOp
            }
        }
        BulkOperation::Rename { from, to } => {
            if let Some(val) = meta.remove(from) {
                meta.insert(to.clone(), val);
                OperationOutcome::Applied
            } else {
                OperationOutcome::NoOp
            }
        }
        BulkOperation::Copy { from, to } => {
            if let Some(val) = meta.get(from).cloned() {
                meta.insert(to.clone(), val);
                OperationOutcome::Applied
            } else {
                OperationOutcome::NoOp
            }
        }
        BulkOperation::AppendSuffix { key, suffix } => {
            if let Some(val) = meta.get(key).cloned() {
                let new_text = format!("{}{}", value_to_string(&val), suffix);
                meta.insert(key.clone(), MetadataValue::Text(new_text));
                OperationOutcome::Applied
            } else {
                OperationOutcome::NoOp
            }
        }
        BulkOperation::PrependPrefix { key, prefix } => {
            if let Some(val) = meta.get(key).cloned() {
                let new_text = format!("{}{}", prefix, value_to_string(&val));
                meta.insert(key.clone(), MetadataValue::Text(new_text));
                OperationOutcome::Applied
            } else {
                OperationOutcome::NoOp
            }
        }
        BulkOperation::ClearAll => {
            meta.clear();
            OperationOutcome::Applied
        }
    }
}

// ---------------------------------------------------------------------------
// BatchReadResult / BatchWriteResult
// ---------------------------------------------------------------------------

/// Summary of a batch read operation over multiple sources.
#[derive(Debug, Default)]
pub struct BatchReadResult {
    /// Successfully read records: `(id, Metadata)`.
    pub records: Vec<(u64, Metadata)>,
    /// Errors encountered: `(id, Error)`.
    pub errors: Vec<(u64, Error)>,
}

impl BatchReadResult {
    /// Create an empty result.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of successfully read records.
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.records.len()
    }

    /// Number of errors.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// `true` if all reads succeeded.
    #[must_use]
    pub fn all_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Convert into a [`MetadataCollection`] (errors are discarded).
    #[must_use]
    pub fn into_collection(self) -> MetadataCollection {
        let mut coll = MetadataCollection::new();
        for (id, meta) in self.records {
            coll.insert(id, meta);
        }
        coll
    }
}

/// Summary of a batch write operation.
#[derive(Debug, Default)]
pub struct BatchWriteResult {
    /// Number of records successfully written.
    pub success_count: u64,
    /// Errors encountered: `(id, Error)`.
    pub errors: Vec<(u64, Error)>,
}

impl BatchWriteResult {
    /// Create an empty result.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` if all writes succeeded.
    #[must_use]
    pub fn all_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Batch I/O helpers
// ---------------------------------------------------------------------------

/// Parse multiple byte slices in batch, each associated with an id and format.
///
/// Records that fail to parse are accumulated in [`BatchReadResult::errors`].
///
/// # Example
///
/// ```rust
/// use oximedia_metadata::bulk_metadata::batch_parse;
/// use oximedia_metadata::MetadataFormat;
///
/// let inputs: Vec<(u64, &[u8], MetadataFormat)> = vec![];
/// let result = batch_parse(&inputs);
/// assert!(result.all_ok());
/// ```
#[must_use]
pub fn batch_parse(
    inputs: &[(u64, &[u8], MetadataFormat)],
) -> BatchReadResult {
    let mut result = BatchReadResult::new();
    for &(id, data, format) in inputs {
        match Metadata::parse(data, format) {
            Ok(meta) => result.records.push((id, meta)),
            Err(e) => result.errors.push((id, e)),
        }
    }
    result
}

/// Serialize every record in a collection to bytes.
///
/// Records that fail to serialize are accumulated in
/// [`BatchWriteResult::errors`].
#[must_use]
pub fn batch_write(collection: &MetadataCollection) -> (Vec<(u64, Vec<u8>)>, BatchWriteResult) {
    let mut written = Vec::with_capacity(collection.len());
    let mut report = BatchWriteResult::new();
    for (id, meta) in collection.iter() {
        match meta.write() {
            Ok(bytes) => {
                written.push((id, bytes));
                report.success_count += 1;
            }
            Err(e) => {
                report.errors.push((id, e));
            }
        }
    }
    (written, report)
}

// ---------------------------------------------------------------------------
// BulkTagOp – high-level tag-centric operations
// ---------------------------------------------------------------------------

/// High-level tag operations that work on the key namespace rather than
/// individual records.
pub struct BulkTagOps;

impl BulkTagOps {
    /// Retain only the listed field keys in every record of the collection.
    ///
    /// Returns the total number of fields removed across all records.
    pub fn keep_only(collection: &mut MetadataCollection, keys: &[&str]) -> usize {
        let key_set: std::collections::HashSet<&str> = keys.iter().copied().collect();
        let mut removed = 0usize;
        let ids: Vec<u64> = collection.ids().collect();
        for id in ids {
            if let Some(m) = collection.get_mut(id) {
                let to_remove: Vec<String> = m
                    .keys()
                    .into_iter()
                    .filter(|k| !key_set.contains(k.as_str()))
                    .cloned()
                    .collect();
                removed += to_remove.len();
                for k in to_remove {
                    m.remove(k.as_str());
                }
            }
        }
        removed
    }

    /// Remove the listed field keys from every record of the collection.
    ///
    /// Returns the total number of fields removed across all records.
    pub fn strip_fields(collection: &mut MetadataCollection, keys: &[&str]) -> usize {
        let mut removed = 0usize;
        for id in collection.ids().collect::<Vec<_>>() {
            if let Some(meta) = collection.get_mut(id) {
                for key in keys {
                    if meta.remove(key).is_some() {
                        removed += 1;
                    }
                }
            }
        }
        removed
    }

    /// Count how many records contain a non-empty value for `field`.
    #[must_use]
    pub fn field_coverage(collection: &MetadataCollection, field: &str) -> usize {
        collection
            .iter()
            .filter(|(_, m)| m.contains(field))
            .count()
    }

    /// Collect the distinct text values for `field` across all records.
    ///
    /// Non-text values are skipped.
    #[must_use]
    pub fn distinct_values(collection: &MetadataCollection, field: &str) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for (_, meta) in collection.iter() {
            if let Some(MetadataValue::Text(v)) = meta.get(field) {
                if seen.insert(v.clone()) {
                    out.push(v.clone());
                }
            }
        }
        out.sort();
        out
    }

    /// Apply a text transformation closure to every occurrence of `field`
    /// in the collection.
    ///
    /// Only `MetadataValue::Text` values are transformed; other types are
    /// left untouched.
    pub fn transform_text<F>(collection: &mut MetadataCollection, field: &str, transform: F)
    where
        F: Fn(&str) -> String,
    {
        for id in collection.ids().collect::<Vec<_>>() {
            if let Some(meta) = collection.get_mut(id) {
                if let Some(MetadataValue::Text(v)) = meta.get(field).cloned() {
                    let new_val = transform(&v);
                    meta.insert(field.to_string(), MetadataValue::Text(new_val));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a [`MetadataValue`] to its canonical text representation.
fn value_to_string(val: &MetadataValue) -> String {
    match val {
        MetadataValue::Text(s) => s.clone(),
        MetadataValue::TextList(list) => list.join("; "),
        MetadataValue::Integer(i) => i.to_string(),
        MetadataValue::Float(f) => f.to_string(),
        MetadataValue::Boolean(b) => b.to_string(),
        MetadataValue::DateTime(dt) => dt.clone(),
        MetadataValue::Binary(b) => format!("<binary {} bytes>", b.len()),
        MetadataValue::Picture(p) => format!("<picture {}>", p.mime_type),
        MetadataValue::Pictures(ps) => format!("<{} pictures>", ps.len()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Metadata, MetadataFormat, MetadataValue};

    fn make_meta(fields: &[(&str, &str)]) -> Metadata {
        let mut m = Metadata::new(MetadataFormat::VorbisComments);
        for &(k, v) in fields {
            m.insert(k.to_string(), MetadataValue::Text(v.to_string()));
        }
        m
    }

    fn make_collection() -> MetadataCollection {
        let mut c = MetadataCollection::new();
        c.insert(1, make_meta(&[("title", "Song A"), ("artist", "Artist X"), ("year", "2020")]));
        c.insert(2, make_meta(&[("title", "Song B"), ("artist", "Artist Y"), ("year", "2021")]));
        c.insert(3, make_meta(&[("title", "Song C"), ("year", "2022")]));
        c
    }

    #[test]
    fn test_collection_insert_get_remove() {
        let mut c = MetadataCollection::new();
        let m = make_meta(&[("title", "Hello")]);
        assert!(c.insert(42, m));
        assert!(c.get(42).is_some());
        assert_eq!(c.len(), 1);
        let removed = c.remove(42);
        assert!(removed.is_some());
        assert!(c.is_empty());
    }

    #[test]
    fn test_collection_overwrite() {
        let mut c = MetadataCollection::new();
        c.insert(1, make_meta(&[("title", "Old")]));
        let is_new = c.insert(1, make_meta(&[("title", "New")]));
        assert!(!is_new);
        assert_eq!(c.len(), 1);
        let title = c.get(1)
            .and_then(|m| m.get("title"))
            .and_then(MetadataValue::as_text)
            .map(String::from);
        assert_eq!(title.as_deref(), Some("New"));
    }

    #[test]
    fn test_field_projection() {
        let c = make_collection();
        let proj = FieldProjection::new(["title", "year"]);
        let rows = proj.project_collection(&c);
        assert_eq!(rows.len(), 3);
        // All rows should have "title" and "year" but not "artist".
        for (_, map) in &rows {
            assert!(map.contains_key("title"));
            assert!(map.contains_key("year"));
            assert!(!map.contains_key("artist"));
        }
    }

    #[test]
    fn test_bulk_operation_set_remove() {
        let mut c = make_collection();
        let ops = vec![
            BulkOperation::Set { key: "genre".to_string(), value: "Jazz".to_string() },
            BulkOperation::Remove { key: "year".to_string() },
        ];
        let operator = BulkOperator::new(ops);
        let outcomes = operator.apply(&mut c);
        assert_eq!(outcomes.len(), 3);
        for (_, per_record) in &outcomes {
            assert_eq!(per_record[0], OperationOutcome::Applied); // Set genre
            assert_eq!(per_record[1], OperationOutcome::Applied); // Remove year
        }
        // Verify side effects.
        for (_, meta) in c.iter() {
            assert!(meta.contains("genre"));
            assert!(!meta.contains("year"));
        }
    }

    #[test]
    fn test_bulk_operation_rename() {
        let mut m = make_meta(&[("old_key", "value")]);
        let ops = vec![BulkOperation::Rename {
            from: "old_key".to_string(),
            to: "new_key".to_string(),
        }];
        let operator = BulkOperator::new(ops);
        let outcomes = operator.apply_one(&mut m);
        assert_eq!(outcomes[0], OperationOutcome::Applied);
        assert!(m.contains("new_key"));
        assert!(!m.contains("old_key"));
    }

    #[test]
    fn test_bulk_operation_noop_on_missing_field() {
        let mut m = make_meta(&[]);
        let ops = vec![BulkOperation::AppendSuffix {
            key: "nonexistent".to_string(),
            suffix: "_suffix".to_string(),
        }];
        let outcomes = BulkOperator::new(ops).apply_one(&mut m);
        assert_eq!(outcomes[0], OperationOutcome::NoOp);
    }

    #[test]
    fn test_bulk_operation_append_prepend() {
        let mut m = make_meta(&[("title", "Hello")]);
        let ops = vec![
            BulkOperation::PrependPrefix { key: "title".to_string(), prefix: "[Re] ".to_string() },
            BulkOperation::AppendSuffix { key: "title".to_string(), suffix: " (Live)".to_string() },
        ];
        BulkOperator::new(ops).apply_one(&mut m);
        let title = m.get("title").and_then(MetadataValue::as_text).map(String::from);
        assert_eq!(title.as_deref(), Some("[Re] Hello (Live)"));
    }

    #[test]
    fn test_bulk_tag_ops_strip_fields() {
        let mut c = make_collection();
        let removed = BulkTagOps::strip_fields(&mut c, &["year"]);
        assert_eq!(removed, 3); // 3 records each had "year"
        for (_, m) in c.iter() {
            assert!(!m.contains("year"));
        }
    }

    #[test]
    fn test_bulk_tag_ops_field_coverage() {
        let c = make_collection();
        assert_eq!(BulkTagOps::field_coverage(&c, "artist"), 2); // record 3 has no artist
        assert_eq!(BulkTagOps::field_coverage(&c, "title"), 3);
    }

    #[test]
    fn test_bulk_tag_ops_distinct_values() {
        let c = make_collection();
        let years = BulkTagOps::distinct_values(&c, "year");
        assert_eq!(years, vec!["2020".to_string(), "2021".to_string(), "2022".to_string()]);
    }

    #[test]
    fn test_bulk_tag_ops_transform_text() {
        let mut c = make_collection();
        BulkTagOps::transform_text(&mut c, "title", |s| s.to_uppercase());
        for (_, meta) in c.iter() {
            if let Some(MetadataValue::Text(t)) = meta.get("title") {
                assert_eq!(t, &t.to_uppercase());
            }
        }
    }

    #[test]
    fn test_batch_read_result_into_collection() {
        let mut r = BatchReadResult::new();
        r.records.push((1, make_meta(&[("title", "A")])));
        r.records.push((2, make_meta(&[("title", "B")])));
        let c = r.into_collection();
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn test_value_to_string_variants() {
        assert_eq!(value_to_string(&MetadataValue::Text("hi".to_string())), "hi");
        assert_eq!(value_to_string(&MetadataValue::Integer(42)), "42");
        assert_eq!(value_to_string(&MetadataValue::Boolean(true)), "true");
        assert_eq!(
            value_to_string(&MetadataValue::TextList(vec!["a".to_string(), "b".to_string()])),
            "a; b"
        );
    }
}
