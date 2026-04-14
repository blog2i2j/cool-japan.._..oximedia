//! Batch indexing support for high-throughput bulk document import.
//!
//! This module provides [`BatchIndexer`], which accumulates documents in an
//! in-memory buffer and flushes them in configurable batch sizes.  Compared to
//! indexing one document at a time, batching amortises serialisation overhead,
//! reduces lock contention on shared indices, and enables parallel pre-processing
//! of document features before the batch is committed.
//!
//! # Design
//!
//! ```text
//! producer(s)                 BatchIndexer
//! ──────────                  ────────────
//! push(doc)   ─────►  [ buffer ] ──► flush() ──► IndexBackend
//!                     capacity ^
//!                     auto-flush when full
//! ```
//!
//! # Example
//!
//! ```rust
//! use oximedia_search::batch_index::{BatchIndexer, BatchDocument, InMemoryBackend, IndexBackend};
//!
//! let backend = InMemoryBackend::new();
//! let mut indexer = BatchIndexer::with_capacity(backend, 3);
//!
//! for i in 0..7u32 {
//!     let doc = BatchDocument::new(format!("doc-{i}"), format!("content about item {i}"));
//!     indexer.push(doc).expect("push should succeed");
//! }
//! indexer.flush().expect("final flush should succeed");
//! assert_eq!(indexer.backend().total_indexed(), 7);
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

use crate::error::{SearchError, SearchResult};

// ─────────────────────────────────────────────────────────────────────────────
// Document type
// ─────────────────────────────────────────────────────────────────────────────

/// A document that can be submitted to a batch indexer.
#[derive(Debug, Clone)]
pub struct BatchDocument {
    /// Unique document identifier (e.g. asset UUID as string).
    pub doc_id: String,
    /// Raw text content to index (title + description + transcript, etc.).
    pub text: String,
    /// Optional arbitrary metadata key/value pairs.
    pub metadata: HashMap<String, String>,
    /// Optional binary feature vector (visual / audio embeddings).
    pub features: Option<Vec<f32>>,
    /// Optional pre-computed tags.
    pub tags: Vec<String>,
}

impl BatchDocument {
    /// Create a minimal batch document with only an ID and text body.
    #[must_use]
    pub fn new(doc_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            doc_id: doc_id.into(),
            text: text.into(),
            metadata: HashMap::new(),
            features: None,
            tags: Vec::new(),
        }
    }

    /// Attach a metadata entry.
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Attach pre-computed feature vector.
    #[must_use]
    pub fn with_features(mut self, features: Vec<f32>) -> Self {
        self.features = Some(features);
        self
    }

    /// Attach tags.
    #[must_use]
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IndexBackend trait
// ─────────────────────────────────────────────────────────────────────────────

/// Abstraction over a search index that accepts batches of documents.
///
/// Implementors are responsible for persistence, locking, and commit logic.
/// The [`InMemoryBackend`] provided in this module is useful for testing and
/// benchmarking without an on-disk index.
pub trait IndexBackend: Send {
    /// Write a batch of documents to the index.
    ///
    /// Implementations may commit or buffer internally; callers should use
    /// [`IndexBackend::commit`] to ensure all writes are durable.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] if any document in the batch cannot be indexed.
    fn write_batch(&mut self, docs: &[BatchDocument]) -> SearchResult<()>;

    /// Commit any buffered writes to make them visible to search.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] if the commit operation fails.
    fn commit(&mut self) -> SearchResult<()>;

    /// Return the total number of documents committed so far.
    fn total_indexed(&self) -> usize;
}

// ─────────────────────────────────────────────────────────────────────────────
// InMemoryBackend
// ─────────────────────────────────────────────────────────────────────────────

/// A simple in-memory [`IndexBackend`] that stores documents in a `Vec`.
///
/// Primarily intended for unit tests and benchmarks.
#[derive(Debug, Default)]
pub struct InMemoryBackend {
    /// Indexed documents.
    docs: Vec<BatchDocument>,
    /// Pending (not yet committed) documents.
    pending: Vec<BatchDocument>,
    /// Simulated failure counter — if `Some(n)`, the next `n` `write_batch`
    /// calls will return an error.  Useful for testing error paths.
    simulate_failure_count: Option<usize>,
}

impl InMemoryBackend {
    /// Create a new, empty in-memory backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the backend to fail the next `n` `write_batch` calls.
    pub fn fail_next(&mut self, n: usize) {
        self.simulate_failure_count = Some(n);
    }

    /// Return a slice of all *committed* documents.
    #[must_use]
    pub fn committed_docs(&self) -> &[BatchDocument] {
        &self.docs
    }

    /// Return the number of pending (not yet committed) documents.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

impl IndexBackend for InMemoryBackend {
    fn write_batch(&mut self, docs: &[BatchDocument]) -> SearchResult<()> {
        // Simulate failures for testing.
        if let Some(ref mut remaining) = self.simulate_failure_count {
            if *remaining > 0 {
                *remaining -= 1;
                if *remaining == 0 {
                    self.simulate_failure_count = None;
                }
                return Err(SearchError::Other(
                    "simulated write failure".to_string(),
                ));
            }
        }
        self.pending.extend_from_slice(docs);
        Ok(())
    }

    fn commit(&mut self) -> SearchResult<()> {
        self.docs.append(&mut self.pending);
        Ok(())
    }

    fn total_indexed(&self) -> usize {
        self.docs.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchIndexer
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics collected by [`BatchIndexer`] over its lifetime.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total documents pushed.
    pub docs_pushed: usize,
    /// Total automatic flushes triggered.
    pub auto_flushes: usize,
    /// Total manual flushes triggered.
    pub manual_flushes: usize,
    /// Total batches written to the backend (each flush = 1 or more batches).
    pub batches_written: usize,
    /// Total documents successfully written.
    pub docs_written: usize,
    /// Total errors encountered during flush.
    pub flush_errors: usize,
}

/// Buffered batch indexer that amortises per-document overhead.
///
/// Documents are accumulated in memory up to `capacity`.  When the buffer
/// reaches capacity an automatic flush is triggered.  A final [`Self::flush`]
/// call must be issued by the caller to drain any remaining documents.
///
/// The indexer owns the [`IndexBackend`] and exposes it via [`Self::backend`]
/// and [`Self::backend_mut`] for inspection or manual operations.
pub struct BatchIndexer<B: IndexBackend> {
    /// Underlying index backend.
    backend: B,
    /// Pending documents not yet written to the backend.
    buffer: Vec<BatchDocument>,
    /// Number of documents to accumulate before auto-flushing.
    capacity: usize,
    /// Collected statistics.
    stats: BatchStats,
    /// Whether errors during auto-flush are propagated immediately (strict) or
    /// counted and skipped (lenient).
    strict_errors: bool,
}

impl<B: IndexBackend> BatchIndexer<B> {
    /// Create a new batch indexer wrapping `backend` with the given buffer
    /// `capacity`.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    #[must_use]
    pub fn with_capacity(backend: B, capacity: usize) -> Self {
        assert!(capacity > 0, "BatchIndexer capacity must be greater than 0");
        Self {
            backend,
            buffer: Vec::with_capacity(capacity),
            capacity,
            stats: BatchStats::default(),
            strict_errors: true,
        }
    }

    /// Set lenient error mode: flush errors increment the error counter but do
    /// not abort the push operation.
    #[must_use]
    pub fn lenient(mut self) -> Self {
        self.strict_errors = false;
        self
    }

    /// Push a document into the buffer.
    ///
    /// If the buffer reaches `capacity` after this push, an automatic flush is
    /// triggered.
    ///
    /// # Errors
    ///
    /// In strict mode (default) returns an error if the auto-flush fails.
    /// In lenient mode the error is counted but not propagated.
    pub fn push(&mut self, doc: BatchDocument) -> SearchResult<()> {
        self.buffer.push(doc);
        self.stats.docs_pushed += 1;

        if self.buffer.len() >= self.capacity {
            self.stats.auto_flushes += 1;
            let result = self.do_flush();
            match result {
                Ok(()) => {}
                Err(e) => {
                    self.stats.flush_errors += 1;
                    if self.strict_errors {
                        return Err(e);
                    }
                }
            }
        }
        Ok(())
    }

    /// Flush any remaining buffered documents to the backend and commit.
    ///
    /// This is a no-op if the buffer is empty.
    ///
    /// # Errors
    ///
    /// Returns an error if the write or commit fails.
    pub fn flush(&mut self) -> SearchResult<()> {
        self.stats.manual_flushes += 1;
        self.do_flush()?;
        self.backend.commit()?;
        Ok(())
    }

    /// Return the number of documents currently buffered (not yet flushed).
    #[must_use]
    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }

    /// Return the configured buffer capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Read-only access to the underlying backend.
    #[must_use]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Mutable access to the underlying backend.
    #[must_use]
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Return a snapshot of the current statistics.
    #[must_use]
    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }

    /// Consume the indexer, returning the backend.
    ///
    /// Any unflushed documents are discarded.  Call [`Self::flush`] first if
    /// you need all documents committed.
    #[must_use]
    pub fn into_backend(self) -> B {
        self.backend
    }

    // ── internal ──────────────────────────────────────────────────────────────

    /// Write the current buffer to the backend (without committing).
    fn do_flush(&mut self) -> SearchResult<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let batch: Vec<BatchDocument> = self.buffer.drain(..).collect();
        let n = batch.len();
        self.backend.write_batch(&batch)?;
        self.stats.batches_written += 1;
        self.stats.docs_written += n;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel batch helper
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-process a slice of documents in parallel (e.g., feature extraction) and
/// return them ready for batch indexing.
///
/// The `preprocess` closure is called on each document concurrently using
/// Rayon.  Documents that fail pre-processing are dropped with a warning
/// recorded in the returned [`PreprocessStats`].
///
/// # Errors
///
/// This function itself is infallible (errors are reported via
/// [`PreprocessStats`]).  Individual document failures are counted but not
/// propagated to avoid aborting an entire batch for one bad document.
pub fn parallel_preprocess<F>(
    docs: Vec<BatchDocument>,
    preprocess: F,
) -> (Vec<BatchDocument>, PreprocessStats)
where
    F: Fn(BatchDocument) -> Result<BatchDocument, String> + Sync + Send,
{
    use rayon::prelude::*;

    let total = docs.len();
    let results: Vec<Result<BatchDocument, String>> =
        docs.into_par_iter().map(|d| preprocess(d)).collect();

    let mut processed = Vec::with_capacity(total);
    let mut failed = 0usize;
    for r in results {
        match r {
            Ok(d) => processed.push(d),
            Err(_) => failed += 1,
        }
    }

    let stats = PreprocessStats {
        total_input: total,
        succeeded: processed.len(),
        failed,
    };
    (processed, stats)
}

/// Statistics from a parallel pre-processing run.
#[derive(Debug, Clone)]
pub struct PreprocessStats {
    /// Total documents submitted for pre-processing.
    pub total_input: usize,
    /// Documents successfully pre-processed.
    pub succeeded: usize,
    /// Documents that failed pre-processing (and were dropped).
    pub failed: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str) -> BatchDocument {
        BatchDocument::new(id, format!("text content for {id}"))
    }

    // ── basic functionality ───────────────────────────────────────────────────

    #[test]
    fn test_push_below_capacity_no_auto_flush() {
        let backend = InMemoryBackend::new();
        let mut indexer = BatchIndexer::with_capacity(backend, 5);
        for i in 0..4 {
            indexer.push(make_doc(&format!("doc-{i}"))).expect("push ok");
        }
        assert_eq!(indexer.buffered_count(), 4);
        // Nothing committed yet.
        assert_eq!(indexer.backend().total_indexed(), 0);
    }

    #[test]
    fn test_auto_flush_at_capacity() {
        let backend = InMemoryBackend::new();
        let mut indexer = BatchIndexer::with_capacity(backend, 3);
        for i in 0..3 {
            indexer.push(make_doc(&format!("doc-{i}"))).expect("push ok");
        }
        // Auto-flush triggered but commit not yet called.
        assert_eq!(indexer.buffered_count(), 0);
        assert_eq!(indexer.stats().auto_flushes, 1);
        assert_eq!(indexer.stats().docs_written, 3);
    }

    #[test]
    fn test_manual_flush_commits() {
        let backend = InMemoryBackend::new();
        let mut indexer = BatchIndexer::with_capacity(backend, 10);
        for i in 0..7 {
            indexer.push(make_doc(&format!("doc-{i}"))).expect("push ok");
        }
        indexer.flush().expect("flush ok");
        assert_eq!(indexer.backend().total_indexed(), 7);
        assert_eq!(indexer.buffered_count(), 0);
    }

    #[test]
    fn test_multiple_auto_flushes_and_final_flush() {
        let backend = InMemoryBackend::new();
        let mut indexer = BatchIndexer::with_capacity(backend, 3);
        for i in 0..10u32 {
            indexer.push(make_doc(&format!("doc-{i}"))).expect("push ok");
        }
        indexer.flush().expect("final flush ok");
        assert_eq!(indexer.backend().total_indexed(), 10);
        assert_eq!(indexer.stats().auto_flushes, 3); // 9/3 = 3 auto-flushes
        assert_eq!(indexer.stats().manual_flushes, 1);
    }

    #[test]
    fn test_flush_empty_buffer_is_noop() {
        let backend = InMemoryBackend::new();
        let mut indexer = BatchIndexer::with_capacity(backend, 5);
        indexer.flush().expect("flush of empty buffer should succeed");
        assert_eq!(indexer.stats().docs_written, 0);
        assert_eq!(indexer.backend().total_indexed(), 0);
    }

    #[test]
    fn test_batch_document_builder() {
        let doc = BatchDocument::new("id-1", "hello world")
            .with_meta("codec", "h264")
            .with_features(vec![0.1, 0.2, 0.3])
            .with_tags(vec!["sport".to_string(), "outdoor".to_string()]);

        assert_eq!(doc.doc_id, "id-1");
        assert_eq!(doc.metadata.get("codec").map(String::as_str), Some("h264"));
        assert_eq!(doc.features.as_ref().map(Vec::len), Some(3));
        assert_eq!(doc.tags.len(), 2);
    }

    #[test]
    fn test_strict_error_propagated() {
        let mut backend = InMemoryBackend::new();
        backend.fail_next(1);
        let mut indexer = BatchIndexer::with_capacity(backend, 2);
        indexer.push(make_doc("a")).expect("first push ok");
        let result = indexer.push(make_doc("b")); // triggers auto-flush which fails
        assert!(result.is_err(), "error should propagate in strict mode");
        assert_eq!(indexer.stats().flush_errors, 1);
    }

    #[test]
    fn test_lenient_error_not_propagated() {
        let mut backend = InMemoryBackend::new();
        backend.fail_next(1);
        let mut indexer = BatchIndexer::with_capacity(backend, 2).lenient();
        indexer.push(make_doc("a")).expect("first push ok");
        let result = indexer.push(make_doc("b")); // triggers auto-flush which fails
        assert!(result.is_ok(), "lenient mode should not propagate error");
        assert_eq!(indexer.stats().flush_errors, 1);
    }

    #[test]
    fn test_into_backend_returns_committed_docs() {
        let backend = InMemoryBackend::new();
        let mut indexer = BatchIndexer::with_capacity(backend, 5);
        for i in 0..4 {
            indexer.push(make_doc(&format!("d{i}"))).expect("push ok");
        }
        indexer.flush().expect("flush ok");
        let backend = indexer.into_backend();
        assert_eq!(backend.total_indexed(), 4);
    }

    #[test]
    fn test_parallel_preprocess_all_succeed() {
        let docs: Vec<BatchDocument> = (0..20)
            .map(|i| BatchDocument::new(format!("doc-{i}"), format!("body {i}")))
            .collect();

        let (processed, stats) = parallel_preprocess(docs, |mut d| {
            d.tags.push("processed".to_string());
            Ok(d)
        });

        assert_eq!(stats.total_input, 20);
        assert_eq!(stats.succeeded, 20);
        assert_eq!(stats.failed, 0);
        assert!(processed.iter().all(|d| d.tags.contains(&"processed".to_string())));
    }

    #[test]
    fn test_parallel_preprocess_partial_failure() {
        let docs: Vec<BatchDocument> = (0..10)
            .map(|i| BatchDocument::new(format!("doc-{i}"), format!("body {i}")))
            .collect();

        // Fail documents with even-numbered IDs.
        let (processed, stats) = parallel_preprocess(docs, |d| {
            if d.doc_id.ends_with('0')
                || d.doc_id.ends_with('2')
                || d.doc_id.ends_with('4')
                || d.doc_id.ends_with('6')
                || d.doc_id.ends_with('8')
            {
                Err(format!("rejected: {}", d.doc_id))
            } else {
                Ok(d)
            }
        });

        assert_eq!(stats.total_input, 10);
        assert_eq!(stats.failed, 5);
        assert_eq!(stats.succeeded, 5);
        assert_eq!(processed.len(), 5);
    }

    #[test]
    fn test_stats_tracking() {
        let backend = InMemoryBackend::new();
        let mut indexer = BatchIndexer::with_capacity(backend, 4);
        for i in 0..9u32 {
            indexer.push(make_doc(&format!("d{i}"))).expect("push ok");
        }
        indexer.flush().expect("flush ok");

        let stats = indexer.stats();
        assert_eq!(stats.docs_pushed, 9);
        assert_eq!(stats.auto_flushes, 2);  // 8 docs / 4 = 2 auto-flushes
        assert_eq!(stats.manual_flushes, 1);
        assert_eq!(stats.docs_written, 9);
        assert_eq!(stats.flush_errors, 0);
    }

    #[test]
    fn test_capacity_accessor() {
        let backend = InMemoryBackend::new();
        let indexer = BatchIndexer::with_capacity(backend, 42);
        assert_eq!(indexer.capacity(), 42);
    }
}
