// cache_invalidation.rs — Edge cache invalidation patterns and tracking.
//
// Provides:
//   - `InvalidationPattern` — Exact / Prefix / Tag / Wildcard matchers
//   - `InvalidationRequest` — pattern + target regions + priority
//   - `InvalidationStatus` — lifecycle state of one request
//   - `InvalidationRecord` — one entry in the history log
//   - `InvalidationTracker` — records history; queries status; fires callbacks
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;

use crate::error::{CloudError, Result};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// InvalidationPattern
// ---------------------------------------------------------------------------

/// Describes the set of cache keys targeted by an invalidation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InvalidationPattern {
    /// Invalidate exactly one cache key.
    ///
    /// Example: `/videos/intro.mp4`
    Exact(String),

    /// Invalidate all keys that start with the given prefix.
    ///
    /// Example: `/thumbnails/` invalidates every key under that path.
    Prefix(String),

    /// Invalidate all objects tagged with the given tag value.
    ///
    /// Tags are provider-specific metadata attached to cached objects.
    Tag(String),

    /// Invalidate keys matching a glob-style wildcard pattern.
    ///
    /// Supported wildcards: `*` (any sequence, no `/`) and `**` (any sequence
    /// including path separators).
    ///
    /// Example: `/static/**.css`
    Wildcard(String),
}

impl InvalidationPattern {
    /// Returns `true` if `key` is matched by this pattern.
    pub fn matches(&self, key: &str) -> bool {
        match self {
            InvalidationPattern::Exact(target) => key == target,
            InvalidationPattern::Prefix(prefix) => key.starts_with(prefix.as_str()),
            InvalidationPattern::Tag(_) => {
                // Tag matching requires provider-side metadata lookup; the
                // in-process implementation conservatively returns `true` so
                // that callers know the object *may* need invalidation.
                true
            }
            InvalidationPattern::Wildcard(pattern) => wildcard_match(pattern, key),
        }
    }

    /// Human-readable description of the pattern type.
    pub fn kind(&self) -> &'static str {
        match self {
            InvalidationPattern::Exact(_) => "exact",
            InvalidationPattern::Prefix(_) => "prefix",
            InvalidationPattern::Tag(_) => "tag",
            InvalidationPattern::Wildcard(_) => "wildcard",
        }
    }
}

// ---------------------------------------------------------------------------
// Wildcard matching (glob-like, no regex dependency)
// ---------------------------------------------------------------------------

/// Glob-style match where `**` matches any sequence (including `/`) and `*`
/// matches any sequence that does not contain `/`.
fn wildcard_match(pattern: &str, text: &str) -> bool {
    wildcard_match_inner(pattern.as_bytes(), text.as_bytes())
}

fn wildcard_match_inner(pattern: &[u8], text: &[u8]) -> bool {
    match (pattern.first(), text.first()) {
        // Both exhausted — match.
        (None, None) => true,
        // Pattern exhausted but text is not — no match.
        (None, Some(_)) => false,
        // `**` — try skipping zero or more characters (including `/`).
        (Some(b'*'), Some(_))
            if pattern.len() >= 2 && pattern[1] == b'*' =>
        {
            let rest_pat = &pattern[2..];
            // Skip the optional `/` separator after `**`.
            let rest_pat = if rest_pat.first() == Some(&b'/') {
                &rest_pat[1..]
            } else {
                rest_pat
            };
            // Try matching rest of pattern starting at every text position.
            for start in 0..=text.len() {
                if wildcard_match_inner(rest_pat, &text[start..]) {
                    return true;
                }
            }
            false
        }
        // Single `*` — matches any character that is not `/`.
        (Some(b'*'), _) => {
            let rest_pat = &pattern[1..];
            for start in 0..=text.len() {
                // Guard: do not consume `/` for single `*`.
                if start > 0 && text[start - 1] == b'/' {
                    break;
                }
                if wildcard_match_inner(rest_pat, &text[start..]) {
                    return true;
                }
            }
            false
        }
        // Literal character match.
        (Some(pc), Some(tc)) => {
            *pc == *tc && wildcard_match_inner(&pattern[1..], &text[1..])
        }
        // Pattern has chars but text is exhausted.
        (Some(_), None) => {
            // A trailing `*` or `**` can match empty.
            pattern == b"*" || pattern == b"**"
        }
    }
}

// ---------------------------------------------------------------------------
// InvalidationRequest
// ---------------------------------------------------------------------------

/// Priority level for an invalidation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InvalidationPriority {
    /// Process when convenient (background queue).
    Low,
    /// Normal processing order.
    Normal,
    /// Elevated — process before normal requests.
    High,
    /// Emergency purge — skip queues if possible.
    Critical,
}

impl Default for InvalidationPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// A request to invalidate a set of cache keys across one or more CDN regions.
#[derive(Debug, Clone)]
pub struct InvalidationRequest {
    /// Unique request identifier (caller-assigned or generated by tracker).
    pub id: String,
    /// The pattern describing which keys to invalidate.
    pub pattern: InvalidationPattern,
    /// CDN edge regions to propagate this invalidation to.
    /// An empty list means "all regions".
    pub regions: Vec<String>,
    /// Processing priority.
    pub priority: InvalidationPriority,
    /// Unix epoch milliseconds when this request was created.
    pub created_at_ms: u64,
}

impl InvalidationRequest {
    /// Create a new request with auto-generated id and current timestamp.
    pub fn new(pattern: InvalidationPattern) -> Self {
        Self {
            id: uuid_v4_simple(),
            pattern,
            regions: Vec::new(),
            priority: InvalidationPriority::Normal,
            created_at_ms: now_ms(),
        }
    }

    /// Builder: restrict to specific regions.
    pub fn with_regions(mut self, regions: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.regions = regions.into_iter().map(Into::into).collect();
        self
    }

    /// Builder: set priority.
    pub fn with_priority(mut self, priority: InvalidationPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Returns `true` if this request targets the given region.
    pub fn targets_region(&self, region: &str) -> bool {
        self.regions.is_empty() || self.regions.iter().any(|r| r == region)
    }
}

// ---------------------------------------------------------------------------
// InvalidationStatus
// ---------------------------------------------------------------------------

/// Lifecycle state of an invalidation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InvalidationStatus {
    /// Accepted, waiting to be dispatched.
    Pending,
    /// Dispatched to the CDN; awaiting confirmation.
    InProgress,
    /// CDN confirmed complete invalidation across all target regions.
    Completed,
    /// Partially completed (some regions succeeded, some failed).
    PartialSuccess,
    /// All regions failed.
    Failed,
    /// Superseded by a newer, broader invalidation before completion.
    Superseded,
}

impl InvalidationStatus {
    /// Returns `true` if this is a terminal state (no further transitions).
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            InvalidationStatus::Completed
                | InvalidationStatus::Failed
                | InvalidationStatus::Superseded
        )
    }
}

// ---------------------------------------------------------------------------
// InvalidationRecord
// ---------------------------------------------------------------------------

/// One entry in the invalidation history log.
#[derive(Debug, Clone)]
pub struct InvalidationRecord {
    /// The original request.
    pub request: InvalidationRequest,
    /// Current lifecycle status.
    pub status: InvalidationStatus,
    /// When the status was last updated (Unix ms).
    pub updated_at_ms: u64,
    /// Regions that have confirmed completion.
    pub completed_regions: Vec<String>,
    /// Regions that reported failure.
    pub failed_regions: Vec<String>,
    /// Optional provider-side invalidation reference (e.g. AWS invalidation id).
    pub provider_ref: Option<String>,
}

impl InvalidationRecord {
    fn new(request: InvalidationRequest) -> Self {
        let now = now_ms();
        Self {
            request,
            status: InvalidationStatus::Pending,
            updated_at_ms: now,
            completed_regions: Vec::new(),
            failed_regions: Vec::new(),
            provider_ref: None,
        }
    }

    /// Transition to a new status, updating the timestamp.
    fn transition(&mut self, new_status: InvalidationStatus) {
        self.status = new_status;
        self.updated_at_ms = now_ms();
    }

    /// Mark a region as completed and recompute aggregate status.
    pub fn mark_region_complete(&mut self, region: impl Into<String>) {
        self.completed_regions.push(region.into());
        self.recompute_status();
    }

    /// Mark a region as failed and recompute aggregate status.
    pub fn mark_region_failed(&mut self, region: impl Into<String>) {
        self.failed_regions.push(region.into());
        self.recompute_status();
    }

    /// Recompute aggregate status from per-region outcomes.
    fn recompute_status(&mut self) {
        let target_count = if self.request.regions.is_empty() {
            // All-regions request: we cannot determine the total, so use what we know.
            self.completed_regions.len() + self.failed_regions.len()
        } else {
            self.request.regions.len()
        };

        let done = self.completed_regions.len() + self.failed_regions.len();

        if done < target_count && !self.request.regions.is_empty() {
            // Not all regions have reported yet.
            self.status = InvalidationStatus::InProgress;
        } else if self.failed_regions.is_empty() {
            self.status = InvalidationStatus::Completed;
        } else if self.completed_regions.is_empty() {
            self.status = InvalidationStatus::Failed;
        } else {
            self.status = InvalidationStatus::PartialSuccess;
        }
        self.updated_at_ms = now_ms();
    }
}

// ---------------------------------------------------------------------------
// InvalidationTracker
// ---------------------------------------------------------------------------

/// Callback invoked when a request's status changes.
pub type StatusCallback = Arc<dyn Fn(&InvalidationRecord) + Send + Sync>;

struct TrackerInner {
    records: HashMap<String, InvalidationRecord>,
    callbacks: Vec<StatusCallback>,
    max_history: usize,
    insertion_order: Vec<String>,
}

/// Records the history of invalidation requests and their status.
///
/// Thread-safe; clone to share across tasks.
#[derive(Clone)]
pub struct InvalidationTracker {
    inner: Arc<Mutex<TrackerInner>>,
}

impl InvalidationTracker {
    /// Create a new tracker.
    ///
    /// - `max_history` — maximum number of records retained; oldest are evicted
    ///   when exceeded.
    pub fn new(max_history: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(TrackerInner {
                records: HashMap::new(),
                callbacks: Vec::new(),
                max_history,
                insertion_order: Vec::new(),
            })),
        }
    }

    /// Register a status-change callback.
    pub fn on_status_change(&self, cb: StatusCallback) {
        self.inner.lock().callbacks.push(cb);
    }

    /// Submit a new invalidation request and return its id.
    pub fn submit(&self, request: InvalidationRequest) -> Result<String> {
        let id = request.id.clone();
        let record = InvalidationRecord::new(request);
        let callbacks: Vec<StatusCallback>;
        {
            let mut guard = self.inner.lock();
            // Enforce history limit.
            while guard.records.len() >= guard.max_history {
                if let Some(oldest) = guard.insertion_order.first().cloned() {
                    guard.records.remove(&oldest);
                    guard.insertion_order.retain(|k| k != &oldest);
                } else {
                    break;
                }
            }
            callbacks = guard.callbacks.clone();
            guard.insertion_order.push(id.clone());
            guard.records.insert(id.clone(), record.clone());
        }
        for cb in &callbacks {
            cb(&record);
        }
        Ok(id)
    }

    /// Transition a request to `InProgress`.
    pub fn mark_in_progress(&self, id: &str, provider_ref: Option<String>) -> Result<()> {
        self.update(id, |rec| {
            if rec.status.is_terminal() {
                return Err(CloudError::InvalidParameter(format!(
                    "request {} is in terminal state {:?}",
                    id, rec.status
                )));
            }
            rec.provider_ref = provider_ref;
            rec.transition(InvalidationStatus::InProgress);
            Ok(())
        })
    }

    /// Record that `region` has completed for request `id`.
    pub fn mark_region_complete(&self, id: &str, region: &str) -> Result<()> {
        self.update(id, |rec| {
            rec.mark_region_complete(region);
            Ok(())
        })
    }

    /// Record that `region` has failed for request `id`.
    pub fn mark_region_failed(&self, id: &str, region: &str) -> Result<()> {
        self.update(id, |rec| {
            rec.mark_region_failed(region);
            Ok(())
        })
    }

    /// Mark a request as superseded.
    pub fn mark_superseded(&self, id: &str) -> Result<()> {
        self.update(id, |rec| {
            rec.transition(InvalidationStatus::Superseded);
            Ok(())
        })
    }

    /// Look up the current status of a request.
    pub fn status(&self, id: &str) -> Option<InvalidationStatus> {
        self.inner.lock().records.get(id).map(|r| r.status)
    }

    /// Return a clone of the record for `id`.
    pub fn record(&self, id: &str) -> Option<InvalidationRecord> {
        self.inner.lock().records.get(id).cloned()
    }

    /// Return all records matching `pattern` (by kind or content equality).
    pub fn history_for_pattern(&self, pattern: &InvalidationPattern) -> Vec<InvalidationRecord> {
        self.inner
            .lock()
            .records
            .values()
            .filter(|r| &r.request.pattern == pattern)
            .cloned()
            .collect()
    }

    /// Return all records (cloned), sorted by creation time ascending.
    pub fn all_records(&self) -> Vec<InvalidationRecord> {
        let guard = self.inner.lock();
        let mut records: Vec<InvalidationRecord> = guard.records.values().cloned().collect();
        records.sort_by_key(|r| r.request.created_at_ms);
        records
    }

    /// Return the number of records currently stored.
    pub fn record_count(&self) -> usize {
        self.inner.lock().records.len()
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn update<F>(&self, id: &str, f: F) -> Result<()>
    where
        F: FnOnce(&mut InvalidationRecord) -> Result<()>,
    {
        let record_clone;
        let callbacks: Vec<StatusCallback>;
        {
            let mut guard = self.inner.lock();
            let rec = guard.records.get_mut(id).ok_or_else(|| {
                CloudError::NotFound(format!("invalidation request '{}' not found", id))
            })?;
            f(rec)?;
            record_clone = rec.clone();
            callbacks = guard.callbacks.clone();
        }
        for cb in &callbacks {
            cb(&record_clone);
        }
        Ok(())
    }
}

impl std::fmt::Debug for InvalidationTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = self.inner.lock();
        f.debug_struct("InvalidationTracker")
            .field("record_count", &guard.records.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// UUID helper (no dependency on uuid crate)
// ---------------------------------------------------------------------------

/// Generate a pseudo-random hex string of length 32 to use as a request id.
fn uuid_v4_simple() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut h = DefaultHasher::new();
    SystemTime::now().hash(&mut h);
    std::thread::current().id().hash(&mut h);
    let a = h.finish();
    h.write_u64(a ^ 0xDEAD_BEEF_CAFE_BABE);
    let b = h.finish();
    format!("{:016x}{:016x}", a, b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // -----------------------------------------------------------------------
    // Pattern matching

    #[test]
    fn test_exact_pattern_matches() {
        let p = InvalidationPattern::Exact("/foo/bar.mp4".into());
        assert!(p.matches("/foo/bar.mp4"));
        assert!(!p.matches("/foo/bar.mp3"));
        assert!(!p.matches("/foo/bar.mp4/extra"));
    }

    #[test]
    fn test_prefix_pattern_matches() {
        let p = InvalidationPattern::Prefix("/static/".into());
        assert!(p.matches("/static/app.js"));
        assert!(p.matches("/static/"));
        assert!(!p.matches("/dynamic/app.js"));
    }

    #[test]
    fn test_wildcard_star_matches() {
        let p = InvalidationPattern::Wildcard("/static/*.js".into());
        assert!(p.matches("/static/app.js"));
        assert!(!p.matches("/static/nested/app.js"),
            "single * should not cross /");
    }

    #[test]
    fn test_wildcard_double_star_matches() {
        let p = InvalidationPattern::Wildcard("/static/**.js".into());
        assert!(p.matches("/static/app.js"));
        assert!(p.matches("/static/nested/deep/app.js"),
            "** should cross /");
    }

    #[test]
    fn test_tag_pattern_always_matches() {
        let p = InvalidationPattern::Tag("campaign-2026".into());
        assert!(p.matches("/anything"));
        assert!(p.matches(""));
    }

    // -----------------------------------------------------------------------
    // InvalidationRequest

    #[test]
    fn test_request_targets_region_empty_means_all() {
        let req = InvalidationRequest::new(InvalidationPattern::Exact("/a".into()));
        assert!(req.targets_region("us-east-1"));
        assert!(req.targets_region("eu-west-1"));
    }

    #[test]
    fn test_request_targets_region_restricted() {
        let req = InvalidationRequest::new(InvalidationPattern::Exact("/a".into()))
            .with_regions(["us-east-1"]);
        assert!(req.targets_region("us-east-1"));
        assert!(!req.targets_region("eu-west-1"));
    }

    // -----------------------------------------------------------------------
    // InvalidationTracker

    #[test]
    fn test_tracker_submit_and_status() {
        let tracker = InvalidationTracker::new(100);
        let req = InvalidationRequest::new(InvalidationPattern::Prefix("/cdn/".into()));
        let id = tracker.submit(req).expect("submit");
        assert_eq!(tracker.status(&id), Some(InvalidationStatus::Pending));
    }

    #[test]
    fn test_tracker_mark_in_progress() {
        let tracker = InvalidationTracker::new(100);
        let id = tracker
            .submit(InvalidationRequest::new(InvalidationPattern::Exact("/a".into())))
            .expect("submit");
        tracker
            .mark_in_progress(&id, Some("cf-inv-123".into()))
            .expect("in_progress");
        assert_eq!(tracker.status(&id), Some(InvalidationStatus::InProgress));
        let rec = tracker.record(&id).expect("record");
        assert_eq!(rec.provider_ref.as_deref(), Some("cf-inv-123"));
    }

    #[test]
    fn test_tracker_region_complete_updates_status() {
        let tracker = InvalidationTracker::new(100);
        let req = InvalidationRequest::new(InvalidationPattern::Exact("/b".into()))
            .with_regions(["us-east-1", "eu-west-1"]);
        let id = tracker.submit(req).expect("submit");
        tracker.mark_region_complete(&id, "us-east-1").expect("ok");
        // Still in-progress (second region not done).
        assert_eq!(tracker.status(&id), Some(InvalidationStatus::InProgress));
        tracker.mark_region_complete(&id, "eu-west-1").expect("ok");
        assert_eq!(tracker.status(&id), Some(InvalidationStatus::Completed));
    }

    #[test]
    fn test_tracker_partial_failure() {
        let tracker = InvalidationTracker::new(100);
        let req = InvalidationRequest::new(InvalidationPattern::Exact("/c".into()))
            .with_regions(["us-east-1", "eu-west-1"]);
        let id = tracker.submit(req).expect("submit");
        tracker.mark_region_complete(&id, "us-east-1").expect("ok");
        tracker.mark_region_failed(&id, "eu-west-1").expect("ok");
        assert_eq!(tracker.status(&id), Some(InvalidationStatus::PartialSuccess));
    }

    #[test]
    fn test_tracker_superseded() {
        let tracker = InvalidationTracker::new(100);
        let id = tracker
            .submit(InvalidationRequest::new(InvalidationPattern::Tag("v1".into())))
            .expect("submit");
        tracker.mark_superseded(&id).expect("supersede");
        assert_eq!(tracker.status(&id), Some(InvalidationStatus::Superseded));
        assert!(tracker.status(&id).unwrap().is_terminal());
    }

    #[test]
    fn test_tracker_history_limit_evicts_oldest() {
        let tracker = InvalidationTracker::new(3);
        let mut ids = Vec::new();
        for i in 0..5 {
            let req =
                InvalidationRequest::new(InvalidationPattern::Exact(format!("/key/{}", i)));
            ids.push(tracker.submit(req).expect("submit"));
        }
        assert_eq!(tracker.record_count(), 3);
        // First two should have been evicted.
        assert!(tracker.status(&ids[0]).is_none());
        assert!(tracker.status(&ids[1]).is_none());
        // Last three should exist.
        assert!(tracker.status(&ids[2]).is_some());
        assert!(tracker.status(&ids[4]).is_some());
    }

    #[test]
    fn test_tracker_status_callback_called() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);
        let tracker = InvalidationTracker::new(100);
        tracker.on_status_change(Arc::new(move |_rec| {
            cc.fetch_add(1, Ordering::SeqCst);
        }));

        let id = tracker
            .submit(InvalidationRequest::new(InvalidationPattern::Exact("/x".into())))
            .expect("submit");
        tracker.mark_in_progress(&id, None).expect("in_progress");
        // submit + mark_in_progress = 2 callbacks.
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_tracker_not_found_error() {
        let tracker = InvalidationTracker::new(100);
        let err = tracker.mark_in_progress("nonexistent-id", None);
        assert!(err.is_err());
        match err.unwrap_err() {
            CloudError::NotFound(_) => {}
            other => panic!("expected NotFound, got {:?}", other),
        }
    }
}
