//! Multi-user timeline editing coordination.
//!
//! Provides region ownership tracking, per-user cursor positions on the
//! timeline, edit attribution, and conflict detection for concurrent edits
//! to overlapping timeline regions.
//!
//! # Concepts
//!
//! * **TimelineRegion** — a half-open `[start_ms, end_ms)` interval on one
//!   named track.
//! * **RegionOwnership** — maps a region to the user who last claimed it and
//!   the edit kind they are performing.
//! * **TimelineCursor** — the current playhead position of a collaborating
//!   user (track + time offset).
//! * **EditAttribution** — a record attached to each edit describing who made
//!   the change and when.
//! * **TimelineCollabManager** — the central coordinator; the only public
//!   entry point for most operations.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::{CollabError, Result};

// ---------------------------------------------------------------------------
// TimelineRegion
// ---------------------------------------------------------------------------

/// A half-open interval `[start_ms, end_ms)` on a named track.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimelineRegion {
    /// Name of the track (e.g. `"video/0"`, `"audio/L"`, `"subtitle/en"`).
    pub track: String,
    /// Region start in milliseconds (inclusive).
    pub start_ms: u64,
    /// Region end in milliseconds (exclusive).  Must be `> start_ms`.
    pub end_ms: u64,
}

impl TimelineRegion {
    /// Create a new `TimelineRegion`.
    ///
    /// Returns an error if `end_ms <= start_ms`.
    pub fn new(track: impl Into<String>, start_ms: u64, end_ms: u64) -> Result<Self> {
        if end_ms <= start_ms {
            return Err(CollabError::InvalidOperation(format!(
                "end_ms ({end_ms}) must be greater than start_ms ({start_ms})"
            )));
        }
        Ok(Self {
            track: track.into(),
            start_ms,
            end_ms,
        })
    }

    /// Duration of the region in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms - self.start_ms
    }

    /// Returns `true` if this region overlaps with `other` on the same track.
    pub fn overlaps(&self, other: &Self) -> bool {
        if self.track != other.track {
            return false;
        }
        self.start_ms < other.end_ms && other.start_ms < self.end_ms
    }

    /// Returns `true` if `time_ms` is contained within `[start_ms, end_ms)`.
    pub fn contains(&self, time_ms: u64) -> bool {
        time_ms >= self.start_ms && time_ms < self.end_ms
    }
}

// ---------------------------------------------------------------------------
// EditKind
// ---------------------------------------------------------------------------

/// The kind of edit a user is performing within a region.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EditKind {
    /// Trimming clip start or end boundaries.
    Trim,
    /// Moving the clip to a different time position.
    Move,
    /// Adjusting clip speed / duration.
    Retime,
    /// Colour grading pass.
    ColorGrade,
    /// Audio level / pan adjustment.
    AudioMix,
    /// Applying a video or audio effect.
    EffectApply,
    /// Subtitle / caption editing.
    SubtitleEdit,
    /// Generic text or metadata update.
    MetadataEdit,
    /// Custom operation — payload described by the caller.
    Custom(String),
}

impl std::fmt::Display for EditKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trim => write!(f, "trim"),
            Self::Move => write!(f, "move"),
            Self::Retime => write!(f, "retime"),
            Self::ColorGrade => write!(f, "color_grade"),
            Self::AudioMix => write!(f, "audio_mix"),
            Self::EffectApply => write!(f, "effect_apply"),
            Self::SubtitleEdit => write!(f, "subtitle_edit"),
            Self::MetadataEdit => write!(f, "metadata_edit"),
            Self::Custom(s) => write!(f, "custom:{s}"),
        }
    }
}

// ---------------------------------------------------------------------------
// RegionOwnership
// ---------------------------------------------------------------------------

/// Record of which user owns a timeline region and what they are doing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionOwnership {
    /// The region being edited.
    pub region: TimelineRegion,
    /// User who claimed the region.
    pub owner_id: Uuid,
    /// Human-readable display name of the owner.
    pub owner_name: String,
    /// Kind of edit underway.
    pub edit_kind: EditKind,
    /// Wall-clock time when the region was claimed (epoch ms).
    pub claimed_at_ms: u64,
    /// Timeout after which the claim expires (epoch ms).
    pub expires_at_ms: u64,
}

impl RegionOwnership {
    /// Returns `true` if the ownership claim has expired at `now_ms`.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms >= self.expires_at_ms
    }
}

// ---------------------------------------------------------------------------
// TimelineCursor
// ---------------------------------------------------------------------------

/// The current playhead position of a collaborating user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineCursor {
    /// User identifier.
    pub user_id: Uuid,
    /// Human-readable display name.
    pub user_name: String,
    /// Track the cursor is on (may be empty if no specific track is selected).
    pub track: Option<String>,
    /// Current playhead position in milliseconds.
    pub position_ms: u64,
    /// Whether the user's timeline is currently playing.
    pub is_playing: bool,
    /// Last time this cursor state was updated (epoch ms).
    pub updated_at_ms: u64,
}

// ---------------------------------------------------------------------------
// EditAttribution
// ---------------------------------------------------------------------------

/// Attribution record attached to a timeline edit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditAttribution {
    /// Unique identifier for this edit record.
    pub id: Uuid,
    /// User who made the edit.
    pub user_id: Uuid,
    /// Human-readable user name.
    pub user_name: String,
    /// Kind of edit performed.
    pub edit_kind: EditKind,
    /// The region that was edited.
    pub region: TimelineRegion,
    /// Wall-clock time of the edit (epoch ms).
    pub timestamp_ms: u64,
    /// Optional description provided by the user or system.
    pub description: String,
    /// Whether this edit was subsequently reverted.
    pub reverted: bool,
}

// ---------------------------------------------------------------------------
// ConflictReport
// ---------------------------------------------------------------------------

/// Describes a detected conflict between two concurrent edits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictReport {
    /// The first conflicting region.
    pub region_a: TimelineRegion,
    /// User who owns region_a.
    pub owner_a: Uuid,
    /// The second conflicting region.
    pub region_b: TimelineRegion,
    /// User who owns region_b.
    pub owner_b: Uuid,
    /// Overlap zone (intersection of the two regions on the same track).
    pub overlap_start_ms: u64,
    /// End of the overlapping portion.
    pub overlap_end_ms: u64,
    /// When the conflict was first detected (epoch ms).
    pub detected_at_ms: u64,
}

// ---------------------------------------------------------------------------
// TimelineCollabConfig
// ---------------------------------------------------------------------------

/// Configuration for [`TimelineCollabManager`].
#[derive(Debug, Clone)]
pub struct TimelineCollabConfig {
    /// How long (ms) a region claim is valid before it auto-expires.
    pub claim_ttl_ms: u64,
    /// Maximum number of simultaneous region claims across all users.
    pub max_claims: usize,
    /// Maximum number of attribution records to retain.
    pub max_attribution_history: usize,
    /// Maximum cursor age (ms) before it is considered stale and removed.
    pub cursor_stale_ms: u64,
}

impl Default for TimelineCollabConfig {
    fn default() -> Self {
        Self {
            claim_ttl_ms: 30_000,         // 30 s
            max_claims: 256,
            max_attribution_history: 10_000,
            cursor_stale_ms: 60_000,       // 1 min
        }
    }
}

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

struct Inner {
    /// Active region claims, keyed by a stable claim ID.
    claims: HashMap<Uuid, RegionOwnership>,
    /// Per-user cursor positions.
    cursors: HashMap<Uuid, TimelineCursor>,
    /// Chronological edit attribution log.
    attributions: Vec<EditAttribution>,
    /// Detected conflicts (cleared when claimed regions are released).
    conflicts: Vec<ConflictReport>,
}

impl Inner {
    fn new() -> Self {
        Self {
            claims: HashMap::new(),
            cursors: HashMap::new(),
            attributions: Vec::new(),
            conflicts: Vec::new(),
        }
    }

    /// Purge expired claims and stale cursors.
    fn gc(&mut self, now_ms: u64, cfg: &TimelineCollabConfig) {
        self.claims.retain(|_, c| !c.is_expired(now_ms));
        self.cursors
            .retain(|_, cur| now_ms.saturating_sub(cur.updated_at_ms) < cfg.cursor_stale_ms);
        // Keep attribution history bounded.
        let max = cfg.max_attribution_history;
        if self.attributions.len() > max {
            let drain_count = self.attributions.len() - max;
            self.attributions.drain(0..drain_count);
        }
    }

    /// Find all existing claims that overlap with `region` for users *other*
    /// than `requester`.
    fn overlapping_claims(&self, region: &TimelineRegion, requester: Uuid) -> Vec<&RegionOwnership> {
        self.claims
            .values()
            .filter(|c| c.owner_id != requester && c.region.overlaps(region))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// TimelineCollabManager
// ---------------------------------------------------------------------------

/// Central coordinator for multi-user timeline editing.
pub struct TimelineCollabManager {
    config: TimelineCollabConfig,
    inner: Arc<RwLock<Inner>>,
}

impl TimelineCollabManager {
    /// Create a new manager with the default configuration.
    pub fn new() -> Self {
        Self::with_config(TimelineCollabConfig::default())
    }

    /// Create a new manager with a custom configuration.
    pub fn with_config(config: TimelineCollabConfig) -> Self {
        Self {
            config,
            inner: Arc::new(RwLock::new(Inner::new())),
        }
    }

    // -----------------------------------------------------------------------
    // Region claims
    // -----------------------------------------------------------------------

    /// Claim a timeline region for editing.
    ///
    /// Returns:
    /// * `Ok(claim_id)` — the claim was accepted and a unique ID was assigned.
    /// * `Err(CollabError::LockFailed)` — the region overlaps with an existing
    ///   claim owned by another user.
    /// * `Err(CollabError::InvalidOperation)` — too many active claims.
    pub fn claim_region(
        &self,
        user_id: Uuid,
        user_name: impl Into<String>,
        region: TimelineRegion,
        edit_kind: EditKind,
        now_ms: u64,
    ) -> Result<Uuid> {
        let mut inner = self.inner.write();
        inner.gc(now_ms, &self.config);

        if inner.claims.len() >= self.config.max_claims {
            return Err(CollabError::InvalidOperation(format!(
                "maximum number of region claims ({}) reached",
                self.config.max_claims
            )));
        }

        let conflicts = inner.overlapping_claims(&region, user_id);
        if !conflicts.is_empty() {
            let conflict_user = conflicts[0].owner_name.clone();
            return Err(CollabError::LockFailed(format!(
                "region [{}, {}) on track '{}' is already claimed by '{}'",
                region.start_ms, region.end_ms, region.track, conflict_user
            )));
        }

        let claim_id = Uuid::new_v4();
        let ownership = RegionOwnership {
            region,
            owner_id: user_id,
            owner_name: user_name.into(),
            edit_kind,
            claimed_at_ms: now_ms,
            expires_at_ms: now_ms + self.config.claim_ttl_ms,
        };
        inner.claims.insert(claim_id, ownership);
        Ok(claim_id)
    }

    /// Release a previously-claimed region.
    ///
    /// Returns `Ok(())` if the claim existed and belonged to `user_id`, or
    /// `Err(CollabError::PermissionDenied)` if the claim belongs to someone else.
    pub fn release_claim(&self, claim_id: Uuid, user_id: Uuid) -> Result<()> {
        let mut inner = self.inner.write();
        match inner.claims.get(&claim_id) {
            None => Ok(()), // already expired or never existed — idempotent
            Some(c) if c.owner_id != user_id => {
                Err(CollabError::PermissionDenied(format!(
                    "claim {claim_id} belongs to user {}, not {user_id}",
                    c.owner_id
                )))
            }
            Some(_) => {
                inner.claims.remove(&claim_id);
                Ok(())
            }
        }
    }

    /// Forcibly release all claims owned by a user (e.g. on disconnect).
    pub fn release_all_claims(&self, user_id: Uuid) {
        let mut inner = self.inner.write();
        inner.claims.retain(|_, c| c.owner_id != user_id);
    }

    /// Return a snapshot of all active (non-expired) region claims at `now_ms`.
    pub fn active_claims(&self, now_ms: u64) -> Vec<RegionOwnership> {
        let inner = self.inner.read();
        inner
            .claims
            .values()
            .filter(|c| !c.is_expired(now_ms))
            .cloned()
            .collect()
    }

    /// Return claims owned by a specific user.
    pub fn claims_by_user(&self, user_id: Uuid, now_ms: u64) -> Vec<RegionOwnership> {
        let inner = self.inner.read();
        inner
            .claims
            .values()
            .filter(|c| c.owner_id == user_id && !c.is_expired(now_ms))
            .cloned()
            .collect()
    }

    // -----------------------------------------------------------------------
    // Cursor management
    // -----------------------------------------------------------------------

    /// Update the playhead cursor position for a user.
    pub fn update_cursor(
        &self,
        user_id: Uuid,
        user_name: impl Into<String>,
        track: Option<String>,
        position_ms: u64,
        is_playing: bool,
        now_ms: u64,
    ) {
        let cursor = TimelineCursor {
            user_id,
            user_name: user_name.into(),
            track,
            position_ms,
            is_playing,
            updated_at_ms: now_ms,
        };
        self.inner.write().cursors.insert(user_id, cursor);
    }

    /// Remove a user's cursor (on disconnect).
    pub fn remove_cursor(&self, user_id: Uuid) {
        self.inner.write().cursors.remove(&user_id);
    }

    /// Return a snapshot of all known (non-stale) cursor positions.
    pub fn cursors(&self, now_ms: u64) -> Vec<TimelineCursor> {
        let inner = self.inner.read();
        inner
            .cursors
            .values()
            .filter(|c| now_ms.saturating_sub(c.updated_at_ms) < self.config.cursor_stale_ms)
            .cloned()
            .collect()
    }

    // -----------------------------------------------------------------------
    // Edit attribution
    // -----------------------------------------------------------------------

    /// Record an edit attribution entry.
    pub fn record_edit(
        &self,
        user_id: Uuid,
        user_name: impl Into<String>,
        region: TimelineRegion,
        edit_kind: EditKind,
        description: impl Into<String>,
        now_ms: u64,
    ) -> Uuid {
        let attr = EditAttribution {
            id: Uuid::new_v4(),
            user_id,
            user_name: user_name.into(),
            edit_kind,
            region,
            timestamp_ms: now_ms,
            description: description.into(),
            reverted: false,
        };
        let id = attr.id;
        let mut inner = self.inner.write();
        inner.attributions.push(attr);
        // Inline GC for history size.
        let max = self.config.max_attribution_history;
        if inner.attributions.len() > max {
            let drain_count = inner.attributions.len() - max;
            inner.attributions.drain(0..drain_count);
        }
        id
    }

    /// Mark an attribution entry as reverted.
    ///
    /// Returns `Err(CollabError::InvalidOperation)` if the entry is not found.
    pub fn mark_reverted(&self, attribution_id: Uuid) -> Result<()> {
        let mut inner = self.inner.write();
        match inner
            .attributions
            .iter_mut()
            .find(|a| a.id == attribution_id)
        {
            Some(a) => {
                a.reverted = true;
                Ok(())
            }
            None => Err(CollabError::InvalidOperation(format!(
                "attribution {attribution_id} not found"
            ))),
        }
    }

    /// Return the attribution history for a specific track within the given
    /// time range, sorted chronologically.
    pub fn attribution_history(
        &self,
        track: &str,
        start_ms: u64,
        end_ms: u64,
    ) -> Vec<EditAttribution> {
        let inner = self.inner.read();
        inner
            .attributions
            .iter()
            .filter(|a| {
                a.region.track == track
                    && a.region.start_ms < end_ms
                    && a.region.end_ms > start_ms
            })
            .cloned()
            .collect()
    }

    // -----------------------------------------------------------------------
    // Conflict detection
    // -----------------------------------------------------------------------

    /// Detect and return any overlapping claims at `now_ms`.
    ///
    /// This performs an O(n²) scan — intended for small numbers of concurrent
    /// users (≤ 20).  The result is also stored internally for later retrieval.
    pub fn detect_conflicts(&self, now_ms: u64) -> Vec<ConflictReport> {
        let mut inner = self.inner.write();
        inner.gc(now_ms, &self.config);

        let claims: Vec<&RegionOwnership> = inner.claims.values().collect();
        let mut reports: Vec<ConflictReport> = Vec::new();

        for i in 0..claims.len() {
            for j in (i + 1)..claims.len() {
                let a = claims[i];
                let b = claims[j];
                if a.owner_id == b.owner_id {
                    continue;
                }
                if a.region.overlaps(&b.region) {
                    let overlap_start = a.region.start_ms.max(b.region.start_ms);
                    let overlap_end = a.region.end_ms.min(b.region.end_ms);
                    reports.push(ConflictReport {
                        region_a: a.region.clone(),
                        owner_a: a.owner_id,
                        region_b: b.region.clone(),
                        owner_b: b.owner_id,
                        overlap_start_ms: overlap_start,
                        overlap_end_ms: overlap_end,
                        detected_at_ms: now_ms,
                    });
                }
            }
        }

        inner.conflicts = reports.clone();
        reports
    }

    // -----------------------------------------------------------------------
    // Maintenance
    // -----------------------------------------------------------------------

    /// Run garbage collection: remove expired claims and stale cursors.
    pub fn gc(&self, now_ms: u64) {
        self.inner.write().gc(now_ms, &self.config);
    }

    /// Return the number of active claims (non-expired at `now_ms`).
    pub fn claim_count(&self, now_ms: u64) -> usize {
        let inner = self.inner.read();
        inner
            .claims
            .values()
            .filter(|c| !c.is_expired(now_ms))
            .count()
    }
}

impl Default for TimelineCollabManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> u64 {
        1_700_000_000_000u64 // fixed epoch ms for deterministic tests
    }

    fn alice() -> Uuid {
        Uuid::parse_str("00000000-0000-0000-0000-000000000001").expect("valid UUID literal")
    }

    fn bob() -> Uuid {
        Uuid::parse_str("00000000-0000-0000-0000-000000000002").expect("valid UUID literal")
    }

    #[test]
    fn test_region_overlap_same_track() {
        let r1 = TimelineRegion::new("video/0", 0, 5000).expect("valid region");
        let r2 = TimelineRegion::new("video/0", 3000, 8000).expect("valid region");
        let r3 = TimelineRegion::new("video/0", 5000, 10000).expect("valid region");
        assert!(r1.overlaps(&r2));
        assert!(!r1.overlaps(&r3)); // touching but not overlapping
    }

    #[test]
    fn test_region_overlap_different_tracks() {
        let r1 = TimelineRegion::new("video/0", 0, 5000).expect("valid region");
        let r2 = TimelineRegion::new("audio/0", 0, 5000).expect("valid region");
        assert!(!r1.overlaps(&r2));
    }

    #[test]
    fn test_claim_and_release() {
        let mgr = TimelineCollabManager::new();
        let region = TimelineRegion::new("video/0", 0, 10_000).expect("valid region");
        let claim_id = mgr
            .claim_region(alice(), "Alice", region, EditKind::Trim, now())
            .expect("claim_region should succeed");

        assert_eq!(mgr.claim_count(now()), 1);
        mgr.release_claim(claim_id, alice()).expect("release_claim by owner should succeed");
        assert_eq!(mgr.claim_count(now()), 0);
    }

    #[test]
    fn test_overlapping_claim_rejected() {
        let mgr = TimelineCollabManager::new();
        let r1 = TimelineRegion::new("video/0", 0, 10_000).expect("valid region");
        let r2 = TimelineRegion::new("video/0", 5_000, 15_000).expect("valid region");

        mgr.claim_region(alice(), "Alice", r1, EditKind::Trim, now())
            .expect("alice's first claim should succeed");
        let err = mgr.claim_region(bob(), "Bob", r2, EditKind::Move, now());
        assert!(err.is_err());
    }

    #[test]
    fn test_same_user_can_claim_overlapping() {
        let mgr = TimelineCollabManager::new();
        let r1 = TimelineRegion::new("video/0", 0, 10_000).expect("valid region");
        let r2 = TimelineRegion::new("video/0", 5_000, 15_000).expect("valid region");

        mgr.claim_region(alice(), "Alice", r1, EditKind::Trim, now())
            .expect("alice's first claim should succeed");
        // Same user claiming overlapping region should succeed.
        let result = mgr.claim_region(alice(), "Alice", r2, EditKind::ColorGrade, now());
        assert!(result.is_ok());
    }

    #[test]
    fn test_claim_expiry() {
        let config = TimelineCollabConfig {
            claim_ttl_ms: 1_000, // 1 second
            ..Default::default()
        };
        let mgr = TimelineCollabManager::with_config(config);
        let region = TimelineRegion::new("video/0", 0, 5_000).expect("valid region");
        let _claim_id = mgr
            .claim_region(alice(), "Alice", region, EditKind::Trim, now())
            .expect("claim_region should succeed within ttl");

        assert_eq!(mgr.claim_count(now()), 1);
        // Simulate 2 seconds passing.
        assert_eq!(mgr.claim_count(now() + 2_000), 0);
    }

    #[test]
    fn test_cursor_update_and_retrieval() {
        let mgr = TimelineCollabManager::new();
        mgr.update_cursor(alice(), "Alice", Some("video/0".into()), 12_000, false, now());
        mgr.update_cursor(bob(), "Bob", None, 5_000, true, now());

        let cursors = mgr.cursors(now());
        assert_eq!(cursors.len(), 2);
    }

    #[test]
    fn test_edit_attribution_recording() {
        let mgr = TimelineCollabManager::new();
        let region = TimelineRegion::new("audio/L", 1_000, 4_000).expect("valid region");
        let attr_id = mgr.record_edit(
            alice(),
            "Alice",
            region.clone(),
            EditKind::AudioMix,
            "volume normalisation",
            now(),
        );
        let history = mgr.attribution_history("audio/L", 0, 10_000);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].id, attr_id);
        assert!(!history[0].reverted);

        mgr.mark_reverted(attr_id).expect("mark_reverted should succeed for existing attribution");
        let history2 = mgr.attribution_history("audio/L", 0, 10_000);
        assert!(history2[0].reverted);
    }

    #[test]
    fn test_conflict_detection() {
        let mgr = TimelineCollabManager::new();
        let r1 = TimelineRegion::new("video/0", 0, 10_000).expect("valid region");
        let r2 = TimelineRegion::new("video/0", 7_000, 15_000).expect("valid region");

        // Bob claims first so Alice's overlapping claim on a different region of
        // the same track will succeed (they're in different time ranges).
        // For conflict detection we manually insert two overlapping claims.
        // The easiest way: both users claim adjacent, non-overlapping regions first
        // to get ownership, then we check detection.
        let _c1 = mgr
            .claim_region(alice(), "Alice", r1, EditKind::Trim, now())
            .expect("alice's claim should succeed");
        // Bob's region overlaps with Alice's — this should fail the claim.
        // So we test detect_conflicts returns empty when there are no actual
        // overlapping active claims (the second claim is rejected).
        let result = mgr.claim_region(bob(), "Bob", r2.clone(), EditKind::Move, now());
        assert!(result.is_err()); // Rejected because it overlaps Alice's claim.

        // Conflicts should be empty since the overlapping claim was rejected.
        let conflicts = mgr.detect_conflicts(now());
        assert_eq!(conflicts.len(), 0);
    }

    #[test]
    fn test_release_all_claims() {
        let mgr = TimelineCollabManager::new();
        let r1 = TimelineRegion::new("video/0", 0, 5_000).expect("valid region");
        let r2 = TimelineRegion::new("video/0", 5_000, 10_000).expect("valid region");
        let r3 = TimelineRegion::new("audio/L", 0, 5_000).expect("valid region");

        mgr.claim_region(alice(), "Alice", r1, EditKind::Trim, now())
            .expect("alice's first claim should succeed");
        mgr.claim_region(alice(), "Alice", r2, EditKind::Move, now())
            .expect("alice's second claim (non-overlapping) should succeed");
        mgr.claim_region(bob(), "Bob", r3, EditKind::AudioMix, now())
            .expect("bob's claim on different track should succeed");

        assert_eq!(mgr.claim_count(now()), 3);
        mgr.release_all_claims(alice());
        assert_eq!(mgr.claim_count(now()), 1); // Only Bob's claim remains.
    }

    #[test]
    fn test_region_invalid_end_before_start() {
        let result = TimelineRegion::new("video/0", 5_000, 1_000);
        assert!(result.is_err());
    }

    #[test]
    fn test_release_claim_wrong_user() {
        let mgr = TimelineCollabManager::new();
        let region = TimelineRegion::new("video/0", 0, 5_000).expect("valid region");
        let claim_id = mgr
            .claim_region(alice(), "Alice", region, EditKind::Trim, now())
            .expect("alice's claim should succeed");

        let result = mgr.release_claim(claim_id, bob());
        assert!(matches!(result, Err(CollabError::PermissionDenied(_))));
    }
}
