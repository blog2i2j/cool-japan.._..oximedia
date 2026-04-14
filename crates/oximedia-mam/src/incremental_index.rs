//! Incremental search index updates for the MAM search layer.
//!
//! Instead of tearing down and rebuilding the entire Tantivy (or compatible)
//! index every time an asset is created, modified, or deleted, this module
//! maintains a **write-ahead change log** (WAL) of index mutations.  A
//! background-friendly [`IndexUpdateApplier`] drains the WAL in batches,
//! minimising I/O amplification.
//!
//! Key types:
//!
//! * [`IndexOperation`] – enum that represents add / update / delete of a
//!   single document.
//! * [`IndexWal`] – thread-safe WAL that accumulates [`IndexOperation`]s and
//!   tracks watermarks.
//! * [`IndexUpdateApplier`] – drains the WAL, applies batches via the
//!   pluggable [`IndexWriter`] trait, and records applied watermarks.
//! * [`InMemoryIndexWriter`] – in-memory reference implementation used in
//!   tests (and as a stand-in before Tantivy is wired up).
//! * [`IndexStats`] – counters maintained by the applier.

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::Instant;

// ---------------------------------------------------------------------------
// IndexDocument
// ---------------------------------------------------------------------------

/// A lightweight document representation suitable for the index layer.
///
/// Custom application fields live in [`IndexDocument::fields`] as plain
/// strings; richer types should be serialised by the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexDocument {
    /// Primary key — the asset UUID as a string.
    pub id: String,
    /// Ordered list of (field_name, field_value) pairs.
    pub fields: Vec<(String, String)>,
}

impl IndexDocument {
    /// Create a new document with the given id and no fields.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            fields: Vec::new(),
        }
    }

    /// Append a field.
    pub fn add_field(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.fields.push((name.into(), value.into()));
    }

    /// Return the value of the first field with `name`, if any.
    #[must_use]
    pub fn field_value(&self, name: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }
}

// ---------------------------------------------------------------------------
// IndexOperation
// ---------------------------------------------------------------------------

/// A single mutation to be applied to the search index.
#[derive(Debug, Clone)]
pub enum IndexOperation {
    /// Insert or replace a document.
    Upsert(IndexDocument),
    /// Remove the document with the given id.
    Delete { id: String },
}

impl IndexOperation {
    /// The document id this operation targets.
    #[must_use]
    pub fn document_id(&self) -> &str {
        match self {
            Self::Upsert(doc) => &doc.id,
            Self::Delete { id } => id,
        }
    }

    /// `true` for upsert operations.
    #[must_use]
    pub fn is_upsert(&self) -> bool {
        matches!(self, Self::Upsert(_))
    }

    /// `true` for delete operations.
    #[must_use]
    pub fn is_delete(&self) -> bool {
        matches!(self, Self::Delete { .. })
    }
}

// ---------------------------------------------------------------------------
// WalEntry
// ---------------------------------------------------------------------------

/// An entry in the write-ahead log.
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// The operation to apply.
    pub operation: IndexOperation,
    /// Timestamp of enqueue.
    pub enqueued_at: Instant,
}

// ---------------------------------------------------------------------------
// IndexWal
// ---------------------------------------------------------------------------

/// Thread-safe write-ahead log for index mutations.
///
/// Multiple threads may push operations concurrently; a single applier thread
/// drains entries in sequence-number order.
#[derive(Debug)]
pub struct IndexWal {
    entries: Mutex<Vec<WalEntry>>,
    next_sequence: AtomicU64,
    applied_watermark: AtomicU64,
}

impl Default for IndexWal {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexWal {
    /// Create an empty WAL.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            next_sequence: AtomicU64::new(1),
            applied_watermark: AtomicU64::new(0),
        }
    }

    /// Enqueue one operation, returning its sequence number.
    ///
    /// # Errors
    ///
    /// Returns an error string if the internal lock is poisoned.
    pub fn push(&self, operation: IndexOperation) -> Result<u64, String> {
        let seq = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let entry = WalEntry {
            sequence: seq,
            operation,
            enqueued_at: Instant::now(),
        };
        self.entries
            .lock()
            .map_err(|e| format!("WAL lock poisoned: {e}"))?
            .push(entry);
        Ok(seq)
    }

    /// Enqueue multiple operations atomically (under a single lock acquisition).
    ///
    /// Returns the sequence numbers assigned, in order.
    ///
    /// # Errors
    ///
    /// Returns an error string if the internal lock is poisoned.
    pub fn push_many(
        &self,
        operations: impl IntoIterator<Item = IndexOperation>,
    ) -> Result<Vec<u64>, String> {
        let mut guard = self
            .entries
            .lock()
            .map_err(|e| format!("WAL lock poisoned: {e}"))?;
        let mut seqs = Vec::new();
        for op in operations {
            let seq = self.next_sequence.fetch_add(1, Ordering::Relaxed);
            guard.push(WalEntry {
                sequence: seq,
                operation: op,
                enqueued_at: Instant::now(),
            });
            seqs.push(seq);
        }
        Ok(seqs)
    }

    /// Drain up to `limit` unapplied entries from the WAL and return them,
    /// sorted ascending by sequence number.
    ///
    /// The caller is responsible for calling [`IndexWal::advance_watermark`]
    /// once the entries have been successfully applied.
    ///
    /// # Errors
    ///
    /// Returns an error string if the internal lock is poisoned.
    pub fn drain(&self, limit: usize) -> Result<Vec<WalEntry>, String> {
        let watermark = self.applied_watermark.load(Ordering::Relaxed);
        let mut guard = self
            .entries
            .lock()
            .map_err(|e| format!("WAL lock poisoned: {e}"))?;

        // Collect unapplied entries up to `limit`.
        let mut pending: Vec<WalEntry> = guard
            .iter()
            .filter(|e| e.sequence > watermark)
            .take(limit)
            .cloned()
            .collect();

        // Sort ascending so the applier processes them in order.
        pending.sort_by_key(|e| e.sequence);

        // Remove drained entries from the WAL storage.
        if !pending.is_empty() {
            let max_seq = pending.last().map(|e| e.sequence).unwrap_or(0);
            guard.retain(|e| e.sequence > max_seq || e.sequence <= watermark);
        }

        Ok(pending)
    }

    /// Advance the applied watermark to `sequence`.
    ///
    /// Only call this after successfully persisting the corresponding entries.
    pub fn advance_watermark(&self, sequence: u64) {
        // Only advance; never go backwards.
        let mut current = self.applied_watermark.load(Ordering::Relaxed);
        while current < sequence {
            match self.applied_watermark.compare_exchange(
                current,
                sequence,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }

    /// Current applied watermark sequence number.
    #[must_use]
    pub fn applied_watermark(&self) -> u64 {
        self.applied_watermark.load(Ordering::Relaxed)
    }

    /// Number of entries currently in the WAL (including already-applied ones
    /// still held in memory).
    ///
    /// # Errors
    ///
    /// Returns 0 if the lock is poisoned (non-panicking).
    #[must_use]
    pub fn pending_count(&self) -> usize {
        let watermark = self.applied_watermark.load(Ordering::Relaxed);
        self.entries
            .lock()
            .map(|g| g.iter().filter(|e| e.sequence > watermark).count())
            .unwrap_or(0)
    }

    /// Clear all entries from the WAL.  Useful for tests and maintenance.
    ///
    /// # Errors
    ///
    /// Returns an error string if the internal lock is poisoned.
    pub fn clear(&self) -> Result<(), String> {
        self.entries
            .lock()
            .map_err(|e| format!("WAL lock poisoned: {e}"))?
            .clear();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// IndexWriter trait
// ---------------------------------------------------------------------------

/// Persistence back-end for the [`IndexUpdateApplier`].
///
/// Implement this to wire up Tantivy, a full-text database, or any other
/// index store.
pub trait IndexWriter {
    /// Upsert a document into the index.
    ///
    /// # Errors
    ///
    /// Returns an error description if the operation fails.
    fn upsert(&mut self, doc: &IndexDocument) -> Result<(), String>;

    /// Delete the document with the given id.
    ///
    /// # Errors
    ///
    /// Returns an error description if the document is not found or the
    /// delete fails.
    fn delete(&mut self, id: &str) -> Result<(), String>;

    /// Commit all staged changes to durable storage.
    ///
    /// # Errors
    ///
    /// Returns an error description if the commit fails.
    fn commit(&mut self) -> Result<(), String>;
}

// ---------------------------------------------------------------------------
// InMemoryIndexWriter
// ---------------------------------------------------------------------------

/// In-memory [`IndexWriter`] for testing and prototyping.
#[derive(Debug, Default)]
pub struct InMemoryIndexWriter {
    /// Current set of indexed documents keyed by document id.
    pub documents: HashMap<String, IndexDocument>,
    /// Number of commit calls issued.
    pub commit_count: u32,
    /// Number of upsert calls issued.
    pub upsert_count: u32,
    /// Number of delete calls issued.
    pub delete_count: u32,
}

impl IndexWriter for InMemoryIndexWriter {
    fn upsert(&mut self, doc: &IndexDocument) -> Result<(), String> {
        self.documents.insert(doc.id.clone(), doc.clone());
        self.upsert_count += 1;
        Ok(())
    }

    fn delete(&mut self, id: &str) -> Result<(), String> {
        self.documents.remove(id);
        self.delete_count += 1;
        Ok(())
    }

    fn commit(&mut self) -> Result<(), String> {
        self.commit_count += 1;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// IndexStats
// ---------------------------------------------------------------------------

/// Cumulative statistics for an [`IndexUpdateApplier`].
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    /// Total upsert operations applied.
    pub total_upserts: u64,
    /// Total delete operations applied.
    pub total_deletes: u64,
    /// Total commit calls issued.
    pub total_commits: u64,
    /// Total batch flushes performed.
    pub total_flushes: u64,
    /// Total errors encountered (operations that failed and were retried or skipped).
    pub total_errors: u64,
}

// ---------------------------------------------------------------------------
// IndexUpdateApplier
// ---------------------------------------------------------------------------

/// Drains the [`IndexWal`] in configurable batch sizes and applies each batch
/// to an [`IndexWriter`], committing after each successful batch.
pub struct IndexUpdateApplier<W: IndexWriter> {
    wal: Arc<IndexWal>,
    writer: W,
    batch_size: usize,
    stats: IndexStats,
}

impl<W: IndexWriter> IndexUpdateApplier<W> {
    /// Create a new applier.
    ///
    /// * `wal` – shared reference to the write-ahead log.
    /// * `writer` – index back-end.
    /// * `batch_size` – maximum number of WAL entries to process per
    ///   [`IndexUpdateApplier::apply_batch`] call.
    #[must_use]
    pub fn new(wal: Arc<IndexWal>, writer: W, batch_size: usize) -> Self {
        Self {
            wal,
            writer,
            batch_size: batch_size.max(1),
            stats: IndexStats::default(),
        }
    }

    /// Apply one batch of pending WAL entries.
    ///
    /// Returns the number of operations applied in this batch (0 means the
    /// WAL was empty).
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL drain or the commit fails.  Individual
    /// upsert/delete errors are recorded in `stats.total_errors` but do not
    /// abort the batch.
    pub fn apply_batch(&mut self) -> Result<usize, String> {
        let entries = self.wal.drain(self.batch_size)?;
        if entries.is_empty() {
            return Ok(0);
        }

        let mut applied_up_to = 0u64;
        let mut applied_count = 0usize;

        for entry in &entries {
            let op_result = match &entry.operation {
                IndexOperation::Upsert(doc) => {
                    self.writer.upsert(doc).map(|_| {
                        self.stats.total_upserts += 1;
                    })
                }
                IndexOperation::Delete { id } => {
                    self.writer.delete(id).map(|_| {
                        self.stats.total_deletes += 1;
                    })
                }
            };

            if let Err(_e) = op_result {
                self.stats.total_errors += 1;
                // Continue processing remaining entries; skip this one.
            } else {
                applied_count += 1;
                applied_up_to = entry.sequence;
            }
        }

        // Commit even if some operations had errors (we advance the watermark
        // only for entries we successfully processed).
        self.writer.commit()?;
        self.stats.total_commits += 1;
        self.stats.total_flushes += 1;

        if applied_up_to > 0 {
            self.wal.advance_watermark(applied_up_to);
        }

        Ok(applied_count)
    }

    /// Drain and apply all pending entries (in multiple batches if necessary).
    ///
    /// Returns total number of operations applied.
    ///
    /// # Errors
    ///
    /// Returns on the first batch commit error.
    pub fn apply_all(&mut self) -> Result<usize, String> {
        let mut total = 0;
        loop {
            let count = self.apply_batch()?;
            if count == 0 {
                break;
            }
            total += count;
        }
        Ok(total)
    }

    /// Current cumulative statistics.
    #[must_use]
    pub fn stats(&self) -> &IndexStats {
        &self.stats
    }

    /// Borrow the underlying writer (e.g. to inspect in tests).
    #[must_use]
    pub fn writer(&self) -> &W {
        &self.writer
    }

    /// Consume the applier, returning the underlying writer.
    pub fn into_writer(self) -> W {
        self.writer
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str, title: &str) -> IndexDocument {
        let mut doc = IndexDocument::new(id);
        doc.add_field("title", title);
        doc
    }

    fn make_wal() -> Arc<IndexWal> {
        Arc::new(IndexWal::new())
    }

    #[test]
    fn test_index_document_field_access() {
        let mut doc = IndexDocument::new("asset-1");
        doc.add_field("title", "My Film");
        doc.add_field("description", "A great film");

        assert_eq!(doc.field_value("title"), Some("My Film"));
        assert_eq!(doc.field_value("description"), Some("A great film"));
        assert_eq!(doc.field_value("missing"), None);
    }

    #[test]
    fn test_wal_push_and_pending_count() {
        let wal = IndexWal::new();
        assert_eq!(wal.pending_count(), 0);

        wal.push(IndexOperation::Upsert(make_doc("a", "Alpha")))
            .expect("push");
        wal.push(IndexOperation::Upsert(make_doc("b", "Beta")))
            .expect("push");
        assert_eq!(wal.pending_count(), 2);
    }

    #[test]
    fn test_wal_push_many() {
        let wal = IndexWal::new();
        let ops = vec![
            IndexOperation::Upsert(make_doc("x", "X")),
            IndexOperation::Delete { id: "y".into() },
        ];
        let seqs = wal.push_many(ops).expect("push_many");
        assert_eq!(seqs.len(), 2);
        assert!(seqs[0] < seqs[1]);
        assert_eq!(wal.pending_count(), 2);
    }

    #[test]
    fn test_wal_drain_advances_watermark() {
        let wal = IndexWal::new();
        wal.push(IndexOperation::Upsert(make_doc("a", "A")))
            .expect("push");
        wal.push(IndexOperation::Upsert(make_doc("b", "B")))
            .expect("push");

        let entries = wal.drain(10).expect("drain");
        assert_eq!(entries.len(), 2);

        let max_seq = entries.iter().map(|e| e.sequence).max().unwrap_or(0);
        wal.advance_watermark(max_seq);
        assert_eq!(wal.applied_watermark(), max_seq);
        assert_eq!(wal.pending_count(), 0);
    }

    #[test]
    fn test_wal_drain_respects_limit() {
        let wal = IndexWal::new();
        for i in 0..10_u32 {
            wal.push(IndexOperation::Upsert(make_doc(&i.to_string(), "x")))
                .expect("push");
        }
        let batch = wal.drain(3).expect("drain");
        assert_eq!(batch.len(), 3);
        // Remaining 7 still pending.
        assert_eq!(wal.pending_count(), 7);
    }

    #[test]
    fn test_applier_apply_batch_upsert() {
        let wal = make_wal();
        wal.push(IndexOperation::Upsert(make_doc("doc1", "Title 1")))
            .expect("push");
        wal.push(IndexOperation::Upsert(make_doc("doc2", "Title 2")))
            .expect("push");

        let writer = InMemoryIndexWriter::default();
        let mut applier = IndexUpdateApplier::new(Arc::clone(&wal), writer, 100);
        let count = applier.apply_batch().expect("apply_batch");

        assert_eq!(count, 2);
        assert_eq!(applier.stats().total_upserts, 2);
        assert_eq!(applier.stats().total_commits, 1);
        assert!(applier.writer().documents.contains_key("doc1"));
        assert!(applier.writer().documents.contains_key("doc2"));
    }

    #[test]
    fn test_applier_apply_batch_delete() {
        let wal = make_wal();
        // First upsert, then delete.
        wal.push(IndexOperation::Upsert(make_doc("doc1", "Title 1")))
            .expect("push");

        let writer = InMemoryIndexWriter::default();
        let mut applier = IndexUpdateApplier::new(Arc::clone(&wal), writer, 100);
        applier.apply_batch().expect("apply upsert");

        wal.push(IndexOperation::Delete { id: "doc1".into() })
            .expect("push delete");
        applier.apply_batch().expect("apply delete");

        assert_eq!(applier.stats().total_deletes, 1);
        assert!(!applier.writer().documents.contains_key("doc1"));
    }

    #[test]
    fn test_applier_apply_all_drains_multiple_batches() {
        let wal = make_wal();
        for i in 0..15_u32 {
            wal.push(IndexOperation::Upsert(make_doc(&i.to_string(), "x")))
                .expect("push");
        }

        let writer = InMemoryIndexWriter::default();
        // Batch size of 5 → 3 batches to drain 15 entries.
        let mut applier = IndexUpdateApplier::new(Arc::clone(&wal), writer, 5);
        let total = applier.apply_all().expect("apply_all");

        assert_eq!(total, 15);
        assert_eq!(applier.writer().documents.len(), 15);
        assert!(applier.stats().total_flushes >= 3);
    }

    #[test]
    fn test_applier_empty_wal_returns_zero() {
        let wal = make_wal();
        let writer = InMemoryIndexWriter::default();
        let mut applier = IndexUpdateApplier::new(wal, writer, 100);
        let count = applier.apply_batch().expect("apply_batch");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_wal_watermark_only_advances() {
        let wal = IndexWal::new();
        wal.advance_watermark(100);
        assert_eq!(wal.applied_watermark(), 100);
        // Attempting to go backwards should be a no-op.
        wal.advance_watermark(50);
        assert_eq!(wal.applied_watermark(), 100);
    }

    #[test]
    fn test_index_operation_predicates() {
        let upsert = IndexOperation::Upsert(make_doc("x", "X"));
        let delete = IndexOperation::Delete { id: "x".into() };

        assert!(upsert.is_upsert());
        assert!(!upsert.is_delete());
        assert!(delete.is_delete());
        assert!(!delete.is_upsert());
        assert_eq!(upsert.document_id(), "x");
        assert_eq!(delete.document_id(), "x");
    }
}
