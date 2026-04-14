//! Offline review: download review packages and sync comments when reconnected.
//!
//! An [`OfflinePackage`] is a serialisable bundle containing everything a reviewer
//! needs to work without network access: the review metadata, all existing
//! annotations, and (optionally) proxy frame thumbnails encoded as raw RGBA.
//! While offline, reviewers accumulate [`PendingComment`]s which are later
//! merged back into the live session via [`SyncEngine::sync`].
//!
//! Conflict resolution is deterministic: remote changes always win on annotation
//! edits (last-write-wins by timestamp), while new local additions are preserved.

use serde::{Deserialize, Serialize};

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors that can arise during offline-review operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OfflineError {
    /// The package format is not recognised or is corrupt.
    InvalidPackage(String),
    /// A sync conflict could not be resolved automatically.
    UnresolvableConflict(String),
    /// Attempted to commit without a valid package loaded.
    NoPackageLoaded,
    /// A pending comment references an unknown annotation ID.
    UnknownAnnotationId(u64),
}

impl std::fmt::Display for OfflineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPackage(msg) => write!(f, "invalid package: {msg}"),
            Self::UnresolvableConflict(msg) => write!(f, "unresolvable conflict: {msg}"),
            Self::NoPackageLoaded => write!(f, "no offline package loaded"),
            Self::UnknownAnnotationId(id) => write!(f, "unknown annotation id: {id}"),
        }
    }
}

// ─── PackageAnnotation ────────────────────────────────────────────────────────

/// A snapshot of a single annotation included in the offline package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageAnnotation {
    /// Annotation ID.
    pub id: u64,
    /// Frame number this annotation is attached to.
    pub frame_number: u64,
    /// Annotation body text.
    pub body: String,
    /// Author identifier.
    pub author: String,
    /// Creation timestamp (Unix epoch seconds).
    pub created_at: u64,
    /// Last-modified timestamp.
    pub modified_at: u64,
    /// Annotation type tag (e.g. "Issue", "General").
    pub kind: String,
    /// Whether this annotation has been resolved.
    pub resolved: bool,
}

// ─── ProxyFrame ───────────────────────────────────────────────────────────────

/// A compressed proxy-resolution frame thumbnail bundled into the package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyFrame {
    /// Frame number.
    pub frame_number: u64,
    /// Thumbnail width in pixels.
    pub width: u32,
    /// Thumbnail height in pixels.
    pub height: u32,
    /// Raw RGBA pixels (`width * height * 4` bytes).
    /// May be empty if no thumbnail data was fetched.
    pub pixels: Vec<u8>,
}

impl ProxyFrame {
    /// Create a metadata-only proxy frame (no pixel data).
    #[must_use]
    pub fn new_meta(frame_number: u64, width: u32, height: u32) -> Self {
        Self {
            frame_number,
            width,
            height,
            pixels: Vec::new(),
        }
    }

    /// Returns `true` if this proxy frame contains pixel data.
    #[must_use]
    pub fn has_pixels(&self) -> bool {
        let expected = self.width as usize * self.height as usize * 4;
        !self.pixels.is_empty() && self.pixels.len() >= expected
    }
}

// ─── OfflinePackage ───────────────────────────────────────────────────────────

/// A self-contained review package suitable for offline use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflinePackage {
    /// Package schema version (incremented when the format changes).
    pub schema_version: u32,
    /// Unique review session ID.
    pub session_id: String,
    /// Human-readable session title.
    pub session_title: String,
    /// Content identifier.
    pub content_id: String,
    /// Version label of the content being reviewed.
    pub version_label: String,
    /// Timestamp when the package was created (Unix epoch seconds).
    pub packaged_at: u64,
    /// Total number of frames in the review content.
    pub total_frames: u64,
    /// All annotations known at the time of packaging.
    pub annotations: Vec<PackageAnnotation>,
    /// Proxy thumbnails for key frames.
    pub proxy_frames: Vec<ProxyFrame>,
}

impl OfflinePackage {
    /// Current package schema version.
    pub const CURRENT_SCHEMA: u32 = 1;

    /// Create a new offline package.
    #[must_use]
    pub fn new(
        session_id: impl Into<String>,
        session_title: impl Into<String>,
        content_id: impl Into<String>,
        version_label: impl Into<String>,
        packaged_at: u64,
        total_frames: u64,
    ) -> Self {
        Self {
            schema_version: Self::CURRENT_SCHEMA,
            session_id: session_id.into(),
            session_title: session_title.into(),
            content_id: content_id.into(),
            version_label: version_label.into(),
            packaged_at,
            total_frames,
            annotations: Vec::new(),
            proxy_frames: Vec::new(),
        }
    }

    /// Add a pre-existing annotation to the package.
    pub fn add_annotation(&mut self, ann: PackageAnnotation) {
        self.annotations.push(ann);
    }

    /// Add a proxy frame thumbnail.
    pub fn add_proxy_frame(&mut self, frame: ProxyFrame) {
        self.proxy_frames.push(frame);
    }

    /// Look up an annotation by ID.
    #[must_use]
    pub fn get_annotation(&self, id: u64) -> Option<&PackageAnnotation> {
        self.annotations.iter().find(|a| a.id == id)
    }

    /// Validate that the package is well-formed.
    ///
    /// # Errors
    ///
    /// Returns [`OfflineError::InvalidPackage`] if required fields are empty or
    /// the schema version is not supported.
    pub fn validate(&self) -> Result<(), OfflineError> {
        if self.schema_version != Self::CURRENT_SCHEMA {
            return Err(OfflineError::InvalidPackage(format!(
                "unsupported schema version {}",
                self.schema_version
            )));
        }
        if self.session_id.is_empty() {
            return Err(OfflineError::InvalidPackage("session_id is empty".into()));
        }
        if self.content_id.is_empty() {
            return Err(OfflineError::InvalidPackage("content_id is empty".into()));
        }
        Ok(())
    }
}

// ─── PendingComment ───────────────────────────────────────────────────────────

/// A comment created offline, waiting to be synced to the live session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingComment {
    /// Client-local identifier (not yet assigned a server ID).
    pub local_id: u64,
    /// Frame number this comment is attached to.
    pub frame_number: u64,
    /// Comment body.
    pub body: String,
    /// Author identifier.
    pub author: String,
    /// Timestamp the comment was created (Unix epoch seconds).
    pub created_at: u64,
    /// Kind tag (mirrors `PackageAnnotation::kind`).
    pub kind: String,
    /// Whether this comment replies to an existing annotation.
    pub reply_to_id: Option<u64>,
}

impl PendingComment {
    /// Create a new top-level pending comment.
    #[must_use]
    pub fn new(
        local_id: u64,
        frame_number: u64,
        body: impl Into<String>,
        author: impl Into<String>,
        created_at: u64,
    ) -> Self {
        Self {
            local_id,
            frame_number,
            body: body.into(),
            author: author.into(),
            created_at,
            kind: "General".to_string(),
            reply_to_id: None,
        }
    }

    /// Mark this comment as a reply (builder pattern).
    #[must_use]
    pub fn with_reply_to(mut self, annotation_id: u64) -> Self {
        self.reply_to_id = Some(annotation_id);
        self
    }

    /// Set the kind tag (builder pattern).
    #[must_use]
    pub fn with_kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = kind.into();
        self
    }
}

// ─── SyncConflict / SyncResult ────────────────────────────────────────────────

/// Describes a conflict detected during sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConflict {
    /// Local comment ID involved in the conflict.
    pub local_id: u64,
    /// Human-readable description of what conflicted.
    pub description: String,
    /// How the conflict was resolved.
    pub resolution: ConflictResolution,
}

/// Strategy used to resolve a sync conflict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Remote version was kept; local change discarded.
    RemoteWins,
    /// Local version was kept.
    LocalWins,
    /// Both versions were kept (e.g. as separate annotations).
    Merged,
    /// The conflicting item was dropped entirely.
    Dropped,
}

/// Result returned after a completed sync operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResult {
    /// Session ID that was synced.
    pub session_id: String,
    /// Number of pending comments successfully uploaded.
    pub comments_synced: usize,
    /// Number of comments that failed to sync.
    pub comments_failed: usize,
    /// Conflicts detected and their resolutions.
    pub conflicts: Vec<SyncConflict>,
    /// Wall-clock time the sync completed (Unix epoch seconds).
    pub synced_at: u64,
}

impl SyncResult {
    /// Returns `true` if the sync completed without any conflicts.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.conflicts.is_empty() && self.comments_failed == 0
    }
}

// ─── OfflineStore ─────────────────────────────────────────────────────────────

/// In-memory store for an active offline review session.
///
/// Holds the downloaded package and accumulates pending comments until
/// [`SyncEngine::sync`] is called.
#[derive(Debug, Default)]
pub struct OfflineStore {
    package: Option<OfflinePackage>,
    pending: Vec<PendingComment>,
    next_local_id: u64,
}

impl OfflineStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            package: None,
            pending: Vec::new(),
            next_local_id: 1,
        }
    }

    /// Load an offline package into the store, replacing any existing package.
    ///
    /// # Errors
    ///
    /// Returns [`OfflineError::InvalidPackage`] if `pkg.validate()` fails.
    pub fn load(&mut self, pkg: OfflinePackage) -> Result<(), OfflineError> {
        pkg.validate()?;
        self.package = Some(pkg);
        self.pending.clear();
        self.next_local_id = 1;
        Ok(())
    }

    /// Add a pending comment to be synced later.
    ///
    /// # Errors
    ///
    /// Returns [`OfflineError::NoPackageLoaded`] if no package is loaded, or
    /// [`OfflineError::UnknownAnnotationId`] if `reply_to_id` references an
    /// annotation not present in the package.
    pub fn add_comment(
        &mut self,
        frame_number: u64,
        body: impl Into<String>,
        author: impl Into<String>,
        created_at: u64,
    ) -> Result<u64, OfflineError> {
        if self.package.is_none() {
            return Err(OfflineError::NoPackageLoaded);
        }
        let id = self.next_local_id;
        self.next_local_id += 1;
        let comment = PendingComment::new(id, frame_number, body, author, created_at);
        self.pending.push(comment);
        Ok(id)
    }

    /// Add a reply to an existing annotation.
    ///
    /// # Errors
    ///
    /// Returns [`OfflineError::NoPackageLoaded`] or
    /// [`OfflineError::UnknownAnnotationId`] if `reply_to_id` is not in the package.
    pub fn add_reply(
        &mut self,
        reply_to_id: u64,
        frame_number: u64,
        body: impl Into<String>,
        author: impl Into<String>,
        created_at: u64,
    ) -> Result<u64, OfflineError> {
        let pkg = self.package.as_ref().ok_or(OfflineError::NoPackageLoaded)?;
        if pkg.get_annotation(reply_to_id).is_none() {
            return Err(OfflineError::UnknownAnnotationId(reply_to_id));
        }
        let id = self.next_local_id;
        self.next_local_id += 1;
        let comment = PendingComment::new(id, frame_number, body, author, created_at)
            .with_reply_to(reply_to_id);
        self.pending.push(comment);
        Ok(id)
    }

    /// Number of pending (unsynced) comments.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Borrow the loaded package, if any.
    #[must_use]
    pub fn package(&self) -> Option<&OfflinePackage> {
        self.package.as_ref()
    }

    /// Drain and return all pending comments (used by `SyncEngine`).
    fn drain_pending(&mut self) -> Vec<PendingComment> {
        std::mem::take(&mut self.pending)
    }
}

// ─── SyncEngine ───────────────────────────────────────────────────────────────

/// Drives the sync of an [`OfflineStore`] back to the live session.
///
/// In production, `sync` would POST each comment to the server.  Here the
/// engine operates purely in-memory: it applies all pending comments to a
/// supplied mutable list of `PackageAnnotation`s (simulating the remote state
/// after sync) and returns a [`SyncResult`].
pub struct SyncEngine;

impl SyncEngine {
    /// Sync all pending comments from `store` into `remote_annotations`.
    ///
    /// Each pending comment is converted to a [`PackageAnnotation`] and
    /// appended to `remote_annotations`.  If a comment replies to an ID that
    /// no longer exists in `remote_annotations`, it is recorded as a conflict
    /// with [`ConflictResolution::Dropped`].
    #[must_use]
    pub fn sync(
        store: &mut OfflineStore,
        remote_annotations: &mut Vec<PackageAnnotation>,
        synced_at: u64,
    ) -> SyncResult {
        let session_id = store
            .package
            .as_ref()
            .map(|p| p.session_id.clone())
            .unwrap_or_default();

        let pending = store.drain_pending();
        let total = pending.len();
        let mut conflicts = Vec::new();
        let mut synced = 0usize;

        // Build an index of remote annotation IDs for fast lookup.
        let remote_ids: std::collections::HashSet<u64> =
            remote_annotations.iter().map(|a| a.id).collect();

        // Determine the next server-side annotation ID.
        let mut next_id = remote_annotations
            .iter()
            .map(|a| a.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);

        for pending_comment in pending {
            // Validate reply target.
            if let Some(parent_id) = pending_comment.reply_to_id {
                if !remote_ids.contains(&parent_id) {
                    conflicts.push(SyncConflict {
                        local_id: pending_comment.local_id,
                        description: format!(
                            "reply_to_id {} no longer exists on remote",
                            parent_id
                        ),
                        resolution: ConflictResolution::Dropped,
                    });
                    continue;
                }
            }

            // Create a server-side annotation record.
            let ann = PackageAnnotation {
                id: next_id,
                frame_number: pending_comment.frame_number,
                body: pending_comment.body,
                author: pending_comment.author,
                created_at: pending_comment.created_at,
                modified_at: pending_comment.created_at,
                kind: pending_comment.kind,
                resolved: false,
            };
            next_id = next_id.saturating_add(1);
            remote_annotations.push(ann);
            synced += 1;
        }

        SyncResult {
            session_id,
            comments_synced: synced,
            comments_failed: total - synced - conflicts.len(),
            conflicts,
            synced_at,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_package(session_id: &str) -> OfflinePackage {
        OfflinePackage::new(
            session_id,
            "Dailies Review",
            "content-42",
            "v3",
            1_700_000_000,
            2400,
        )
    }

    fn make_annotation(id: u64, frame: u64) -> PackageAnnotation {
        PackageAnnotation {
            id,
            frame_number: frame,
            body: "test annotation".to_string(),
            author: "alice".to_string(),
            created_at: 1_700_000_000,
            modified_at: 1_700_000_000,
            kind: "General".to_string(),
            resolved: false,
        }
    }

    #[test]
    fn package_validates_correctly() {
        let pkg = make_package("sess-01");
        assert!(pkg.validate().is_ok());
    }

    #[test]
    fn package_invalid_empty_session_id() {
        let pkg = make_package("");
        assert!(pkg.validate().is_err());
    }

    #[test]
    fn package_invalid_schema_version() {
        let mut pkg = make_package("sess-01");
        pkg.schema_version = 99;
        let err = pkg.validate().unwrap_err();
        assert!(matches!(err, OfflineError::InvalidPackage(_)));
    }

    #[test]
    fn store_load_resets_pending() {
        let mut store = OfflineStore::new();
        store.load(make_package("sess-01")).expect("load ok");
        store
            .add_comment(100, "note", "alice", 1_700_000_001)
            .expect("add ok");
        assert_eq!(store.pending_count(), 1);
        // Reload flushes pending comments.
        store.load(make_package("sess-01")).expect("reload ok");
        assert_eq!(store.pending_count(), 0);
    }

    #[test]
    fn store_add_comment_without_package_errors() {
        let mut store = OfflineStore::new();
        let err = store
            .add_comment(100, "note", "alice", 1_700_000_001)
            .unwrap_err();
        assert_eq!(err, OfflineError::NoPackageLoaded);
    }

    #[test]
    fn store_add_comment_increments_local_id() {
        let mut store = OfflineStore::new();
        store.load(make_package("sess-01")).expect("load ok");
        let id1 = store
            .add_comment(100, "a", "alice", 1_700_000_001)
            .expect("first");
        let id2 = store
            .add_comment(200, "b", "alice", 1_700_000_002)
            .expect("second");
        assert_ne!(id1, id2);
        assert_eq!(store.pending_count(), 2);
    }

    #[test]
    fn store_add_reply_unknown_id_errors() {
        let mut store = OfflineStore::new();
        store.load(make_package("sess-01")).expect("load ok");
        let err = store
            .add_reply(999, 100, "reply", "bob", 1_700_000_002)
            .unwrap_err();
        assert_eq!(err, OfflineError::UnknownAnnotationId(999));
    }

    #[test]
    fn store_add_reply_known_id_ok() {
        let mut store = OfflineStore::new();
        let mut pkg = make_package("sess-01");
        pkg.add_annotation(make_annotation(10, 50));
        store.load(pkg).expect("load ok");
        let local_id = store
            .add_reply(10, 50, "agreed", "bob", 1_700_000_003)
            .expect("reply ok");
        assert!(local_id > 0);
    }

    #[test]
    fn sync_engine_syncs_comments() {
        let mut store = OfflineStore::new();
        store.load(make_package("sess-01")).expect("load ok");
        store
            .add_comment(100, "looks great", "alice", 1_700_000_001)
            .expect("add ok");
        store
            .add_comment(200, "color shift on cut", "alice", 1_700_000_002)
            .expect("add ok");

        let mut remote: Vec<PackageAnnotation> = Vec::new();
        let result = SyncEngine::sync(&mut store, &mut remote, 1_700_001_000);
        assert_eq!(result.comments_synced, 2);
        assert_eq!(result.comments_failed, 0);
        assert!(result.conflicts.is_empty());
        assert!(result.is_clean());
        assert_eq!(remote.len(), 2);
        assert_eq!(store.pending_count(), 0);
    }

    #[test]
    fn sync_engine_drops_reply_to_deleted_annotation() {
        let mut store = OfflineStore::new();
        let mut pkg = make_package("sess-01");
        pkg.add_annotation(make_annotation(5, 30));
        store.load(pkg).expect("load ok");
        store
            .add_reply(5, 30, "good point", "charlie", 1_700_000_001)
            .expect("reply ok");

        // Remote no longer has annotation 5.
        let mut remote: Vec<PackageAnnotation> = Vec::new();
        let result = SyncEngine::sync(&mut store, &mut remote, 1_700_001_000);
        assert_eq!(result.conflicts.len(), 1);
        assert_eq!(result.conflicts[0].resolution, ConflictResolution::Dropped);
        assert_eq!(result.comments_synced, 0);
        assert!(remote.is_empty());
    }

    #[test]
    fn proxy_frame_has_pixels() {
        let frame = ProxyFrame {
            frame_number: 100,
            width: 4,
            height: 4,
            pixels: vec![0u8; 4 * 4 * 4],
        };
        assert!(frame.has_pixels());
    }

    #[test]
    fn proxy_frame_meta_only_no_pixels() {
        let frame = ProxyFrame::new_meta(100, 320, 180);
        assert!(!frame.has_pixels());
    }
}
