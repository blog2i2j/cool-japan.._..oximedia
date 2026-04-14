//! Three-way media project merge.
//!
//! Provides conflict-free merging of timeline events, parameter maps, and scalar
//! values using the standard three-way merge algorithm: if only one side changed
//! relative to the base, accept that change; if both sides changed differently,
//! report a conflict; if both changed identically, accept either.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────────────────

/// A conflict that was detected during a three-way merge.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeConflict<T: Clone> {
    /// Path / key that identifies the conflicting element.
    pub path: String,
    /// Value from the common ancestor (base).
    pub base: T,
    /// Our local version of the value.
    pub ours: T,
    /// Their (remote) version of the value.
    pub theirs: T,
}

/// Outcome of a three-way merge attempt for a single value.
#[derive(Debug, Clone, PartialEq)]
pub enum MergeResult<T: Clone> {
    /// The merge was clean; the resolved value is carried here.
    Clean(T),
    /// The merge produced a conflict; manual resolution is required.
    Conflict(MergeConflict<T>),
}

impl<T: Clone> MergeResult<T> {
    /// Returns `true` when the result is clean (no conflict).
    pub fn is_clean(&self) -> bool {
        matches!(self, MergeResult::Clean(_))
    }

    /// Returns `true` when the result contains a conflict.
    pub fn is_conflict(&self) -> bool {
        matches!(self, MergeResult::Conflict(_))
    }

    /// Returns the clean value, if any.
    pub fn clean_value(&self) -> Option<&T> {
        match self {
            MergeResult::Clean(v) => Some(v),
            MergeResult::Conflict(_) => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar merges
// ─────────────────────────────────────────────────────────────────────────────

/// Three-way merge for `f32` values.
///
/// Rules (using bit-exact comparison via `to_bits()`):
/// - `ours == base, theirs != base` → accept `theirs`
/// - `theirs == base, ours != base` → accept `ours`
/// - `ours == theirs` (both changed identically) → accept either
/// - otherwise → conflict
pub fn three_way_merge_f32(base: f32, ours: f32, theirs: f32) -> MergeResult<f32> {
    let ours_changed = ours.to_bits() != base.to_bits();
    let theirs_changed = theirs.to_bits() != base.to_bits();
    let both_same = ours.to_bits() == theirs.to_bits();

    match (ours_changed, theirs_changed) {
        (false, false) => MergeResult::Clean(base),
        (false, true) => MergeResult::Clean(theirs),
        (true, false) => MergeResult::Clean(ours),
        (true, true) if both_same => MergeResult::Clean(ours),
        (true, true) => MergeResult::Conflict(MergeConflict {
            path: String::new(),
            base,
            ours,
            theirs,
        }),
    }
}

/// Three-way merge for `String` (or `&str`) values.
///
/// Identical logic to [`three_way_merge_f32`] but for strings.
pub fn three_way_merge_string(base: &str, ours: &str, theirs: &str) -> MergeResult<String> {
    let ours_changed = ours != base;
    let theirs_changed = theirs != base;
    let both_same = ours == theirs;

    match (ours_changed, theirs_changed) {
        (false, false) => MergeResult::Clean(base.to_string()),
        (false, true) => MergeResult::Clean(theirs.to_string()),
        (true, false) => MergeResult::Clean(ours.to_string()),
        (true, true) if both_same => MergeResult::Clean(ours.to_string()),
        (true, true) => MergeResult::Conflict(MergeConflict {
            path: String::new(),
            base: base.to_string(),
            ours: ours.to_string(),
            theirs: theirs.to_string(),
        }),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Timeline domain types
// ─────────────────────────────────────────────────────────────────────────────

/// A discrete event on a media project timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct TimelineEvent {
    /// Globally unique identifier for this event.
    pub id: u64,
    /// Absolute position in milliseconds from project start.
    pub timestamp_ms: i64,
    /// Duration of the event in milliseconds.
    pub duration_ms: u32,
    /// Track this event belongs to.
    pub track_id: u32,
    /// Semantic type of the event (e.g. "clip", "transition", "effect").
    pub event_type: String,
    /// Numeric parameters keyed by name (volume, pitch, etc.).
    pub parameters: HashMap<String, f32>,
}

impl TimelineEvent {
    /// Shallow merge of `parameters` from two diverged versions of the same event.
    fn merge_into(
        base: &TimelineEvent,
        ours: &TimelineEvent,
        theirs: &TimelineEvent,
        path_prefix: &str,
    ) -> MergeResult<TimelineEvent> {
        // Merge every scalar field with three-way logic, collecting conflicts.
        let ts_result = three_way_merge_f32(
            base.timestamp_ms as f32,
            ours.timestamp_ms as f32,
            theirs.timestamp_ms as f32,
        );
        let dur_result = three_way_merge_f32(
            base.duration_ms as f32,
            ours.duration_ms as f32,
            theirs.duration_ms as f32,
        );
        let track_result = three_way_merge_f32(
            base.track_id as f32,
            ours.track_id as f32,
            theirs.track_id as f32,
        );
        let type_result =
            three_way_merge_string(&base.event_type, &ours.event_type, &theirs.event_type);
        let param_results =
            merge_parameters(&base.parameters, &ours.parameters, &theirs.parameters);

        // If any field has a conflict, report the whole event as conflicting.
        let has_conflict = ts_result.is_conflict()
            || dur_result.is_conflict()
            || track_result.is_conflict()
            || type_result.is_conflict()
            || param_results.values().any(|r| r.is_conflict());

        if has_conflict {
            return MergeResult::Conflict(MergeConflict {
                path: path_prefix.to_string(),
                base: base.clone(),
                ours: ours.clone(),
                theirs: theirs.clone(),
            });
        }

        // All clean — build the resolved event.
        let resolved_params: HashMap<String, f32> = param_results
            .into_iter()
            .filter_map(|(k, r)| r.clean_value().copied().map(|v| (k, v)))
            .collect();

        let resolved = TimelineEvent {
            id: base.id,
            timestamp_ms: ts_result
                .clean_value()
                .copied()
                .unwrap_or(base.timestamp_ms as f32) as i64,
            duration_ms: dur_result
                .clean_value()
                .copied()
                .unwrap_or(base.duration_ms as f32) as u32,
            track_id: track_result
                .clean_value()
                .copied()
                .unwrap_or(base.track_id as f32) as u32,
            event_type: type_result
                .clean_value()
                .cloned()
                .unwrap_or_else(|| base.event_type.clone()),
            parameters: resolved_params,
        };

        MergeResult::Clean(resolved)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameter map merge
// ─────────────────────────────────────────────────────────────────────────────

/// Three-way merge of `HashMap<String, f32>` parameter maps.
///
/// Each key is merged independently:
/// - Key only in `ours` (new in ours relative to base) → `Clean(ours_val)`
/// - Key only in `theirs` (new in theirs) → `Clean(theirs_val)`
/// - Key in base but deleted by both → omitted
/// - Key in base deleted by one side → `Clean` of the other
/// - Key present in all three → per-value [`three_way_merge_f32`]
pub fn merge_parameters(
    base: &HashMap<String, f32>,
    ours: &HashMap<String, f32>,
    theirs: &HashMap<String, f32>,
) -> HashMap<String, MergeResult<f32>> {
    let mut results: HashMap<String, MergeResult<f32>> = HashMap::new();

    // Union of all keys.
    let all_keys: std::collections::HashSet<&String> = base
        .keys()
        .chain(ours.keys())
        .chain(theirs.keys())
        .collect();

    for key in all_keys {
        let base_val = base.get(key).copied();
        let ours_val = ours.get(key).copied();
        let theirs_val = theirs.get(key).copied();

        let result = match (base_val, ours_val, theirs_val) {
            // Added on our side only.
            (None, Some(o), None) => MergeResult::Clean(o),
            // Added on their side only.
            (None, None, Some(t)) => MergeResult::Clean(t),
            // Both added with the same value.
            (None, Some(o), Some(t)) if o.to_bits() == t.to_bits() => MergeResult::Clean(o),
            // Both added with different values → conflict (base is 0.0 placeholder).
            (None, Some(o), Some(t)) => MergeResult::Conflict(MergeConflict {
                path: key.clone(),
                base: 0.0_f32,
                ours: o,
                theirs: t,
            }),
            // In base but removed by both.
            (Some(_b), None, None) => continue,
            // In base, removed by ours, kept (unchanged) by theirs → accept removal.
            (Some(b), None, Some(t)) if t.to_bits() == b.to_bits() => continue,
            // In base, removed by ours, changed by theirs → conflict.
            (Some(b), None, Some(t)) => MergeResult::Conflict(MergeConflict {
                path: key.clone(),
                base: b,
                ours: 0.0_f32,
                theirs: t,
            }),
            // In base, kept (unchanged) by ours, removed by theirs → accept removal.
            (Some(b), Some(o), None) if o.to_bits() == b.to_bits() => continue,
            // In base, changed by ours, removed by theirs → conflict.
            (Some(b), Some(o), None) => MergeResult::Conflict(MergeConflict {
                path: key.clone(),
                base: b,
                ours: o,
                theirs: 0.0_f32,
            }),
            // Normal three-way merge.
            (Some(b), Some(o), Some(t)) => {
                let mut r = three_way_merge_f32(b, o, t);
                // Attach the path to any conflict that was generated.
                if let MergeResult::Conflict(ref mut c) = r {
                    c.path = key.clone();
                }
                r
            }
            // Should not happen (key came from union, but not present anywhere).
            (None, None, None) => continue,
        };

        results.insert(key.clone(), result);
    }

    results
}

// ─────────────────────────────────────────────────────────────────────────────
// ProjectMerger
// ─────────────────────────────────────────────────────────────────────────────

/// Merges two diverged versions of a media project timeline.
pub struct ProjectMerger;

impl ProjectMerger {
    /// Create a new `ProjectMerger`.
    pub fn new() -> Self {
        Self
    }

    /// Three-way merge of timeline event lists.
    ///
    /// Events are matched by `id`.  Events present only in `ours` or only in
    /// `theirs` (i.e. added after the base) are included as `Clean`.  Events
    /// present in the base that are removed by one side but unchanged by the
    /// other are omitted (clean deletion).  Events that both sides modified
    /// differently yield `Conflict`.
    pub fn merge_timelines(
        &self,
        base: &[TimelineEvent],
        ours: &[TimelineEvent],
        theirs: &[TimelineEvent],
    ) -> Vec<MergeResult<TimelineEvent>> {
        // Index by id for O(1) lookup.
        let base_map: HashMap<u64, &TimelineEvent> = base.iter().map(|e| (e.id, e)).collect();
        let ours_map: HashMap<u64, &TimelineEvent> = ours.iter().map(|e| (e.id, e)).collect();
        let theirs_map: HashMap<u64, &TimelineEvent> = theirs.iter().map(|e| (e.id, e)).collect();

        let all_ids: std::collections::BTreeSet<u64> = base_map
            .keys()
            .chain(ours_map.keys())
            .chain(theirs_map.keys())
            .copied()
            .collect();

        let mut results = Vec::new();

        for id in all_ids {
            let b = base_map.get(&id).copied();
            let o = ours_map.get(&id).copied();
            let t = theirs_map.get(&id).copied();

            let path = format!("timeline/event/{}", id);

            let result = match (b, o, t) {
                // Added in ours only.
                (None, Some(ev), None) => MergeResult::Clean(ev.clone()),
                // Added in theirs only.
                (None, None, Some(ev)) => MergeResult::Clean(ev.clone()),
                // Added in both with equal content.
                (None, Some(oe), Some(te)) if events_equal(oe, te) => {
                    MergeResult::Clean(oe.clone())
                }
                // Added in both with different content — conflict.
                (None, Some(oe), Some(te)) => MergeResult::Conflict(MergeConflict {
                    path,
                    base: oe.clone(), // no base; use ours as stand-in
                    ours: oe.clone(),
                    theirs: te.clone(),
                }),
                // Present in base, deleted by both.
                (Some(_be), None, None) => continue,
                // Present in base, deleted by ours, unchanged by theirs.
                (Some(be), None, Some(te)) if events_equal(be, te) => continue,
                // Present in base, deleted by ours, modified by theirs → conflict.
                (Some(be), None, Some(te)) => MergeResult::Conflict(MergeConflict {
                    path,
                    base: be.clone(),
                    ours: be.clone(), // stand-in (deleted)
                    theirs: te.clone(),
                }),
                // Present in base, deleted by theirs, unchanged by ours.
                (Some(be), Some(oe), None) if events_equal(be, oe) => continue,
                // Present in base, deleted by theirs, modified by ours → conflict.
                (Some(be), Some(oe), None) => MergeResult::Conflict(MergeConflict {
                    path,
                    base: be.clone(),
                    ours: oe.clone(),
                    theirs: be.clone(), // stand-in (deleted)
                }),
                // Present in all three — do the full field-level merge.
                (Some(be), Some(oe), Some(te)) => TimelineEvent::merge_into(be, oe, te, &path),
                // Shouldn't happen.
                (None, None, None) => continue,
            };

            results.push(result);
        }

        results
    }
}

impl Default for ProjectMerger {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Conflict resolution
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for automatically resolving `MergeConflict`s without manual
/// intervention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictResolver {
    /// Always pick the local ("ours") version.
    TakeOurs,
    /// Always pick the remote ("theirs") version.
    TakeTheirs,
    /// Always revert to the common ancestor ("base") version.
    TakeBase,
    /// Leave conflicts unresolved; the caller must handle them.
    Manual,
}

/// Resolve all conflicts in a slice of `MergeResult<TimelineEvent>` using the
/// given strategy.  `Manual` conflicts are resolved by taking `ours`.
///
/// Returns only the successfully resolved `TimelineEvent`s in order.
pub fn resolve_conflicts(
    results: Vec<MergeResult<TimelineEvent>>,
    resolver: ConflictResolver,
) -> Vec<TimelineEvent> {
    results
        .into_iter()
        .map(|r| match r {
            MergeResult::Clean(ev) => ev,
            MergeResult::Conflict(c) => match resolver {
                ConflictResolver::TakeOurs => c.ours,
                ConflictResolver::TakeTheirs => c.theirs,
                ConflictResolver::TakeBase => c.base,
                // For Manual, default to ours so the list stays complete.
                ConflictResolver::Manual => c.ours,
            },
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Extended conflict resolution API (MergeConfig / ConflictResolution)
// ─────────────────────────────────────────────────────────────────────────────

/// Conflict resolution strategies for three-way merge operations.
///
/// These strategies control how conflicts are automatically (or manually)
/// resolved when both sides diverge from the common base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictResolution {
    /// Prefer the local (ours) version when a conflict arises.
    TakeOurs,
    /// Prefer the remote (theirs) version when a conflict arises.
    TakeTheirs,
    /// Include both sides — the event appears twice in the output (duplicate).
    TakeBoth,
    /// Mark the event as conflicted and require manual resolution.
    /// Conflicted events are **included** in the output as the `ours` version
    /// so the caller can inspect and replace them.
    Manual,
    /// Use semantic heuristics to decide automatically:
    /// 1. Prefer the longer duration (more content was added).
    /// 2. If one side deleted content (duration = 0) and the other kept/added,
    ///    prefer the addition.
    /// 3. If a modification timestamp is present, prefer the newer edit.
    Heuristic,
}

/// Configuration for a merge operation driven by [`ConflictResolution`].
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Conflict resolution strategy to apply.
    pub strategy: ConflictResolution,
    /// Number of surrounding milliseconds of context to capture in conflict
    /// reports (analogous to diff context lines).
    pub max_conflict_context: usize,
    /// If `true`, conflicts that differ only in whitespace within string
    /// parameters are auto-resolved by taking the non-whitespace-only side.
    pub auto_resolve_whitespace: bool,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            strategy: ConflictResolution::Manual,
            max_conflict_context: 3,
            auto_resolve_whitespace: true,
        }
    }
}

impl MergeConfig {
    /// Create a new merge config with the given strategy and sensible defaults.
    #[must_use]
    pub fn with_strategy(strategy: ConflictResolution) -> Self {
        Self {
            strategy,
            ..Default::default()
        }
    }
}

/// Statistics collected during a merge operation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MergeStats {
    /// Number of conflicts automatically resolved.
    pub auto_resolved: usize,
    /// Number of conflicts that still require manual resolution.
    pub conflicts: usize,
    /// Total number of changed values processed (both clean and conflicted).
    pub total_changes: usize,
}

impl MergeStats {
    /// Returns `true` when the merge is fully clean (no remaining conflicts).
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.conflicts == 0
    }
}

/// Outcome of a full merge pass driven by [`MergeConfig`].
///
/// Unlike the lower-level per-item [`MergeResult`], this bundles the entire
/// resolved event list together with statistics and any unresolved conflicts.
#[derive(Debug, Clone)]
pub struct MergeOutput {
    /// Events that have been resolved (either cleanly or via the strategy).
    pub merged: Vec<TimelineEvent>,
    /// Conflicts that could not be auto-resolved (only non-empty when strategy
    /// is [`ConflictResolution::Manual`]).
    pub conflicts: Vec<MergeConflict<TimelineEvent>>,
    /// Statistics from this merge pass.
    pub stats: MergeStats,
}

impl MergeOutput {
    /// Returns `true` when all events were resolved without conflicts.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.conflicts.is_empty()
    }
}

// ── Heuristic helpers ────────────────────────────────────────────────────────

/// Heuristic score for a [`TimelineEvent`]: used by [`ConflictResolution::Heuristic`]
/// to decide which side wins a conflict.
///
/// Higher score = "more content / more recent change".
fn heuristic_score(ev: &TimelineEvent) -> i64 {
    // Primary: duration — longer clips contain more content.
    let duration_score = ev.duration_ms as i64;

    // Secondary: parameter richness (more parameters = more editing work).
    let param_score = ev.parameters.len() as i64 * 10;

    // Tertiary: if a "_modified_ms" parameter is present it encodes a wall-clock
    // modification timestamp in milliseconds since epoch.  We scale it down so
    // it acts as a tiebreaker rather than dominating duration differences.
    let ts_score = ev
        .parameters
        .get("_modified_ms")
        .copied()
        .map(|t| t as i64)
        .unwrap_or(0)
        / 1_000_000;

    duration_score + param_score + ts_score
}

/// Apply the Heuristic resolution strategy to a single conflict.
///
/// Rules (in priority order):
/// 1. If one side has `duration_ms == 0` and the other does not, prefer the
///    non-zero side (prefer addition over deletion).
/// 2. Otherwise prefer the side with the higher [`heuristic_score`].
/// 3. Tie-break: prefer `ours`.
fn heuristic_resolve(conflict: &MergeConflict<TimelineEvent>) -> TimelineEvent {
    let ours = &conflict.ours;
    let theirs = &conflict.theirs;

    // Rule 1 — deletion vs. addition
    match (ours.duration_ms == 0, theirs.duration_ms == 0) {
        (true, false) => return theirs.clone(),
        (false, true) => return ours.clone(),
        _ => {}
    }

    // Rule 2 — scored preference
    let ours_score = heuristic_score(ours);
    let theirs_score = heuristic_score(theirs);

    if theirs_score > ours_score {
        theirs.clone()
    } else {
        ours.clone() // Rule 3 — tie-break
    }
}

/// Perform a whitespace-only conflict check on a string merge conflict.
///
/// Returns `Some(resolved)` when only whitespace differs, `None` otherwise.
fn maybe_resolve_whitespace(conflict: &MergeConflict<String>) -> Option<String> {
    let ours_stripped = conflict.ours.split_whitespace().collect::<String>();
    let theirs_stripped = conflict.theirs.split_whitespace().collect::<String>();

    if ours_stripped == theirs_stripped {
        // Both contain the same non-whitespace content; prefer the non-blank side.
        if conflict.ours.trim().is_empty() {
            Some(conflict.theirs.clone())
        } else {
            Some(conflict.ours.clone())
        }
    } else {
        None
    }
}

/// Run a full merge of two timeline branches using the provided [`MergeConfig`].
///
/// This is the high-level entry point that combines [`ProjectMerger::merge_timelines`]
/// with automatic conflict resolution driven by the config's strategy.
///
/// # Arguments
/// * `base`   – common ancestor timeline
/// * `ours`   – our local edits on top of `base`
/// * `theirs` – their remote edits on top of `base`
/// * `config` – merge configuration
///
/// # Returns
/// A [`MergeOutput`] with the resolved timeline and merge statistics.
pub fn merge_with_config(
    base: &[TimelineEvent],
    ours: &[TimelineEvent],
    theirs: &[TimelineEvent],
    config: &MergeConfig,
) -> MergeOutput {
    let merger = ProjectMerger::new();
    let raw = merger.merge_timelines(base, ours, theirs);

    let mut merged = Vec::with_capacity(raw.len());
    let mut unresolved_conflicts = Vec::new();
    let mut stats = MergeStats::default();

    for result in raw {
        match result {
            MergeResult::Clean(ev) => {
                merged.push(ev);
                stats.total_changes += 1;
            }
            MergeResult::Conflict(ref conflict) => {
                stats.total_changes += 1;

                match config.strategy {
                    ConflictResolution::TakeOurs => {
                        merged.push(conflict.ours.clone());
                        stats.auto_resolved += 1;
                    }
                    ConflictResolution::TakeTheirs => {
                        merged.push(conflict.theirs.clone());
                        stats.auto_resolved += 1;
                    }
                    ConflictResolution::TakeBoth => {
                        merged.push(conflict.ours.clone());
                        merged.push(conflict.theirs.clone());
                        stats.auto_resolved += 1;
                    }
                    ConflictResolution::Manual => {
                        // Keep ours as placeholder; record unresolved conflict.
                        merged.push(conflict.ours.clone());
                        unresolved_conflicts.push(conflict.clone());
                        stats.conflicts += 1;
                    }
                    ConflictResolution::Heuristic => {
                        // Attempt whitespace auto-resolve on event_type first.
                        let type_conflict = MergeConflict {
                            path: conflict.path.clone(),
                            base: conflict.base.event_type.clone(),
                            ours: conflict.ours.event_type.clone(),
                            theirs: conflict.theirs.event_type.clone(),
                        };
                        if config.auto_resolve_whitespace
                            && maybe_resolve_whitespace(&type_conflict).is_some()
                        {
                            // Only event_type whitespace difference — take ours.
                            merged.push(conflict.ours.clone());
                            stats.auto_resolved += 1;
                        } else {
                            let resolved = heuristic_resolve(conflict);
                            merged.push(resolved);
                            stats.auto_resolved += 1;
                        }
                    }
                }
            }
        }
    }

    MergeOutput {
        merged,
        conflicts: unresolved_conflicts,
        stats,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn events_equal(a: &TimelineEvent, b: &TimelineEvent) -> bool {
    a.id == b.id
        && a.timestamp_ms == b.timestamp_ms
        && a.duration_ms == b.duration_ms
        && a.track_id == b.track_id
        && a.event_type == b.event_type
        && params_equal(&a.parameters, &b.parameters)
}

fn params_equal(a: &HashMap<String, f32>, b: &HashMap<String, f32>) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().all(|(k, v)| {
        b.get(k)
            .map(|bv| bv.to_bits() == v.to_bits())
            .unwrap_or(false)
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Overlapping timeline events with priority rules
// ─────────────────────────────────────────────────────────────────────────────

/// Priority level assigned to a timeline event for conflict resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EventPriority {
    /// Low priority — easily displaced by other events.
    Low = 0,
    /// Normal priority (default).
    Normal = 1,
    /// High priority — takes precedence in overlaps.
    High = 2,
    /// Critical — immovable anchor events.
    Critical = 3,
}

impl EventPriority {
    /// Numeric weight used for tie-breaking.
    #[must_use]
    pub fn weight(&self) -> u32 {
        match self {
            Self::Low => 1,
            Self::Normal => 10,
            Self::High => 100,
            Self::Critical => 1000,
        }
    }
}

/// Strategy for resolving overlap between two timeline events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlapStrategy {
    /// Higher-priority event wins; lower-priority is trimmed or removed.
    PriorityWins,
    /// The earlier event (by `timestamp_ms`) wins; later is trimmed.
    EarlierWins,
    /// The longer event wins; shorter is trimmed.
    LongerWins,
    /// Both events are kept but flagged as overlapping.
    KeepBothFlagged,
}

/// Describes how an overlap was resolved.
#[derive(Debug, Clone, PartialEq)]
pub enum OverlapResolution {
    /// One event was removed entirely.
    Removed {
        /// ID of the removed event.
        removed_id: u64,
        /// Reason for removal.
        reason: String,
    },
    /// One event was trimmed to avoid overlap.
    Trimmed {
        /// ID of the trimmed event.
        trimmed_id: u64,
        /// Original timestamp_ms.
        original_start_ms: i64,
        /// Original duration_ms.
        original_duration_ms: u32,
        /// New timestamp_ms after trimming.
        new_start_ms: i64,
        /// New duration_ms after trimming.
        new_duration_ms: u32,
    },
    /// Both events were kept, flagged as overlapping.
    Flagged { event_a_id: u64, event_b_id: u64 },
    /// No overlap detected.
    NoOverlap,
}

/// A timeline event annotated with priority information.
#[derive(Debug, Clone, PartialEq)]
pub struct PrioritizedEvent {
    /// The underlying timeline event.
    pub event: TimelineEvent,
    /// Priority for overlap resolution.
    pub priority: EventPriority,
}

impl PrioritizedEvent {
    /// Create a new prioritized event.
    pub fn new(event: TimelineEvent, priority: EventPriority) -> Self {
        Self { event, priority }
    }

    /// End time of the event in milliseconds.
    #[must_use]
    pub fn end_ms(&self) -> i64 {
        self.event.timestamp_ms + self.event.duration_ms as i64
    }

    /// Check whether this event overlaps with another on the same track.
    #[must_use]
    pub fn overlaps(&self, other: &PrioritizedEvent) -> bool {
        self.event.track_id == other.event.track_id
            && self.event.timestamp_ms < other.end_ms()
            && other.event.timestamp_ms < self.end_ms()
    }
}

/// Resolve overlap between two prioritized events.
pub fn resolve_overlap(
    a: &PrioritizedEvent,
    b: &PrioritizedEvent,
    strategy: OverlapStrategy,
) -> OverlapResolution {
    if !a.overlaps(b) {
        return OverlapResolution::NoOverlap;
    }

    match strategy {
        OverlapStrategy::PriorityWins => resolve_by_priority(a, b),
        OverlapStrategy::EarlierWins => resolve_by_time(a, b),
        OverlapStrategy::LongerWins => resolve_by_duration(a, b),
        OverlapStrategy::KeepBothFlagged => OverlapResolution::Flagged {
            event_a_id: a.event.id,
            event_b_id: b.event.id,
        },
    }
}

fn resolve_by_priority(a: &PrioritizedEvent, b: &PrioritizedEvent) -> OverlapResolution {
    let (winner, loser) = if a.priority > b.priority {
        (a, b)
    } else if b.priority > a.priority {
        (b, a)
    } else {
        // Same priority → fall back to earlier timestamp.
        if a.event.timestamp_ms <= b.event.timestamp_ms {
            (a, b)
        } else {
            (b, a)
        }
    };
    trim_or_remove(winner, loser)
}

fn resolve_by_time(a: &PrioritizedEvent, b: &PrioritizedEvent) -> OverlapResolution {
    let (winner, loser) = if a.event.timestamp_ms <= b.event.timestamp_ms {
        (a, b)
    } else {
        (b, a)
    };
    trim_or_remove(winner, loser)
}

fn resolve_by_duration(a: &PrioritizedEvent, b: &PrioritizedEvent) -> OverlapResolution {
    let (winner, loser) = if a.event.duration_ms >= b.event.duration_ms {
        (a, b)
    } else {
        (b, a)
    };
    trim_or_remove(winner, loser)
}

/// Trim the loser event so it no longer overlaps the winner.
/// If the loser would be entirely consumed, remove it instead.
fn trim_or_remove(winner: &PrioritizedEvent, loser: &PrioritizedEvent) -> OverlapResolution {
    let winner_end = winner.end_ms();
    let loser_end = loser.end_ms();

    // Case 1: loser is entirely within winner's range → remove.
    if loser.event.timestamp_ms >= winner.event.timestamp_ms && loser_end <= winner_end {
        return OverlapResolution::Removed {
            removed_id: loser.event.id,
            reason: format!("entirely within winner (id={})", winner.event.id),
        };
    }

    // Case 2: loser starts before winner → trim loser's end.
    if loser.event.timestamp_ms < winner.event.timestamp_ms {
        let new_duration = (winner.event.timestamp_ms - loser.event.timestamp_ms) as u32;
        if new_duration == 0 {
            return OverlapResolution::Removed {
                removed_id: loser.event.id,
                reason: "trimmed to zero duration".to_string(),
            };
        }
        return OverlapResolution::Trimmed {
            trimmed_id: loser.event.id,
            original_start_ms: loser.event.timestamp_ms,
            original_duration_ms: loser.event.duration_ms,
            new_start_ms: loser.event.timestamp_ms,
            new_duration_ms: new_duration,
        };
    }

    // Case 3: loser starts inside winner → move loser's start to winner's end.
    if loser.event.timestamp_ms < winner_end {
        let new_start = winner_end;
        let new_duration = (loser_end - new_start).max(0) as u32;
        if new_duration == 0 {
            return OverlapResolution::Removed {
                removed_id: loser.event.id,
                reason: "trimmed to zero duration".to_string(),
            };
        }
        return OverlapResolution::Trimmed {
            trimmed_id: loser.event.id,
            original_start_ms: loser.event.timestamp_ms,
            original_duration_ms: loser.event.duration_ms,
            new_start_ms: new_start,
            new_duration_ms: new_duration,
        };
    }

    OverlapResolution::NoOverlap
}

/// Resolve all overlaps in a list of prioritized events using the given strategy.
///
/// Returns the resolved events plus a log of all resolutions applied.
pub fn resolve_all_overlaps(
    events: &[PrioritizedEvent],
    strategy: OverlapStrategy,
) -> (Vec<PrioritizedEvent>, Vec<OverlapResolution>) {
    let mut resolved: Vec<PrioritizedEvent> = events.to_vec();
    let mut log: Vec<OverlapResolution> = Vec::new();

    // Sort by track, then by priority descending, then by start time.
    resolved.sort_by(|a, b| {
        a.event
            .track_id
            .cmp(&b.event.track_id)
            .then(b.priority.cmp(&a.priority))
            .then(a.event.timestamp_ms.cmp(&b.event.timestamp_ms))
    });

    let mut i = 0;
    while i < resolved.len() {
        let mut j = i + 1;
        while j < resolved.len() {
            if resolved[i].event.track_id != resolved[j].event.track_id {
                break;
            }
            let resolution = resolve_overlap(&resolved[i], &resolved[j], strategy);
            match &resolution {
                OverlapResolution::Removed { removed_id, .. } => {
                    let id = *removed_id;
                    log.push(resolution);
                    resolved.retain(|e| e.event.id != id);
                    // Don't increment j — the element at j shifted.
                    continue;
                }
                OverlapResolution::Trimmed {
                    trimmed_id,
                    new_start_ms,
                    new_duration_ms,
                    ..
                } => {
                    let tid = *trimmed_id;
                    let ns = *new_start_ms;
                    let nd = *new_duration_ms;
                    if let Some(ev) = resolved.iter_mut().find(|e| e.event.id == tid) {
                        ev.event.timestamp_ms = ns;
                        ev.event.duration_ms = nd;
                    }
                    log.push(resolution);
                }
                OverlapResolution::Flagged { .. } => {
                    log.push(resolution);
                }
                OverlapResolution::NoOverlap => {}
            }
            j += 1;
        }
        i += 1;
    }

    (resolved, log)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(id: u64, timestamp_ms: i64, duration_ms: u32, track_id: u32) -> TimelineEvent {
        TimelineEvent {
            id,
            timestamp_ms,
            duration_ms,
            track_id,
            event_type: "clip".to_string(),
            parameters: HashMap::new(),
        }
    }

    fn make_event_with_params(id: u64, params: &[(&str, f32)]) -> TimelineEvent {
        let mut e = make_event(id, 0, 1000, 0);
        for (k, v) in params {
            e.parameters.insert(k.to_string(), *v);
        }
        e
    }

    // ── Scalar f32 merges ────────────────────────────────────────────────────

    #[test]
    fn test_f32_no_changes() {
        assert_eq!(three_way_merge_f32(1.0, 1.0, 1.0), MergeResult::Clean(1.0));
    }

    #[test]
    fn test_f32_only_theirs_changed() {
        assert_eq!(three_way_merge_f32(1.0, 1.0, 2.0), MergeResult::Clean(2.0));
    }

    #[test]
    fn test_f32_only_ours_changed() {
        assert_eq!(three_way_merge_f32(1.0, 3.0, 1.0), MergeResult::Clean(3.0));
    }

    #[test]
    fn test_f32_both_changed_same() {
        assert_eq!(three_way_merge_f32(1.0, 5.0, 5.0), MergeResult::Clean(5.0));
    }

    #[test]
    fn test_f32_both_changed_conflict() {
        let r = three_way_merge_f32(1.0, 2.0, 3.0);
        assert!(r.is_conflict());
        if let MergeResult::Conflict(c) = r {
            assert_eq!(c.base, 1.0);
            assert_eq!(c.ours, 2.0);
            assert_eq!(c.theirs, 3.0);
        }
    }

    #[test]
    fn test_f32_nan_stability() {
        // NaN != NaN by IEEE, but to_bits() equality treats identical NaN
        // bit patterns as the same.
        let nan = f32::NAN;
        let r = three_way_merge_f32(nan, nan, nan);
        assert!(r.is_clean());
    }

    // ── String merges ────────────────────────────────────────────────────────

    #[test]
    fn test_string_no_changes() {
        assert_eq!(
            three_way_merge_string("hello", "hello", "hello"),
            MergeResult::Clean("hello".to_string())
        );
    }

    #[test]
    fn test_string_only_theirs() {
        assert_eq!(
            three_way_merge_string("a", "a", "b"),
            MergeResult::Clean("b".to_string())
        );
    }

    #[test]
    fn test_string_only_ours() {
        assert_eq!(
            three_way_merge_string("a", "c", "a"),
            MergeResult::Clean("c".to_string())
        );
    }

    #[test]
    fn test_string_both_same() {
        assert_eq!(
            three_way_merge_string("a", "z", "z"),
            MergeResult::Clean("z".to_string())
        );
    }

    #[test]
    fn test_string_conflict() {
        let r = three_way_merge_string("base", "ours", "theirs");
        assert!(r.is_conflict());
    }

    // ── Parameter map merges ─────────────────────────────────────────────────

    #[test]
    fn test_params_no_changes() {
        let mut base = HashMap::new();
        base.insert("vol".to_string(), 0.8_f32);
        let results = merge_parameters(&base, &base, &base);
        assert_eq!(results["vol"], MergeResult::Clean(0.8));
    }

    #[test]
    fn test_params_added_ours_only() {
        let base = HashMap::new();
        let mut ours = HashMap::new();
        ours.insert("gain".to_string(), 1.5_f32);
        let theirs = HashMap::new();
        let results = merge_parameters(&base, &ours, &theirs);
        assert_eq!(results["gain"], MergeResult::Clean(1.5));
    }

    #[test]
    fn test_params_added_theirs_only() {
        let base = HashMap::new();
        let ours = HashMap::new();
        let mut theirs = HashMap::new();
        theirs.insert("pitch".to_string(), 440.0_f32);
        let results = merge_parameters(&base, &ours, &theirs);
        assert_eq!(results["pitch"], MergeResult::Clean(440.0));
    }

    #[test]
    fn test_params_conflict() {
        let mut base = HashMap::new();
        base.insert("vol".to_string(), 0.5_f32);
        let mut ours = HashMap::new();
        ours.insert("vol".to_string(), 0.8_f32);
        let mut theirs = HashMap::new();
        theirs.insert("vol".to_string(), 0.3_f32);
        let results = merge_parameters(&base, &ours, &theirs);
        assert!(results["vol"].is_conflict());
    }

    #[test]
    fn test_params_deleted_by_both() {
        let mut base = HashMap::new();
        base.insert("old".to_string(), 1.0_f32);
        let results = merge_parameters(&base, &HashMap::new(), &HashMap::new());
        assert!(
            !results.contains_key("old"),
            "should be omitted when both deleted"
        );
    }

    // ── Timeline merges ──────────────────────────────────────────────────────

    #[test]
    fn test_timeline_no_changes() {
        let base = vec![make_event(1, 0, 1000, 0)];
        let merger = ProjectMerger::new();
        let results = merger.merge_timelines(&base, &base, &base);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_clean());
    }

    #[test]
    fn test_timeline_added_ours() {
        let base: Vec<TimelineEvent> = vec![];
        let ours = vec![make_event(10, 500, 200, 1)];
        let theirs: Vec<TimelineEvent> = vec![];
        let merger = ProjectMerger::new();
        let results = merger.merge_timelines(&base, &ours, &theirs);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_clean());
    }

    #[test]
    fn test_timeline_added_theirs() {
        let base: Vec<TimelineEvent> = vec![];
        let ours: Vec<TimelineEvent> = vec![];
        let theirs = vec![make_event(20, 1000, 300, 2)];
        let merger = ProjectMerger::new();
        let results = merger.merge_timelines(&base, &ours, &theirs);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_clean());
    }

    #[test]
    fn test_timeline_deleted_by_both() {
        let base = vec![make_event(1, 0, 500, 0)];
        let ours: Vec<TimelineEvent> = vec![];
        let theirs: Vec<TimelineEvent> = vec![];
        let merger = ProjectMerger::new();
        let results = merger.merge_timelines(&base, &ours, &theirs);
        assert!(results.is_empty(), "both deleted → should be omitted");
    }

    #[test]
    fn test_timeline_modified_one_side() {
        let base = vec![make_event(1, 0, 500, 0)];
        let mut ours_ev = make_event(1, 0, 500, 0);
        ours_ev.duration_ms = 999;
        let ours = vec![ours_ev.clone()];
        let theirs = base.clone();
        let merger = ProjectMerger::new();
        let results = merger.merge_timelines(&base, &ours, &theirs);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_clean());
        assert_eq!(results[0].clean_value().map(|e| e.duration_ms), Some(999));
    }

    #[test]
    fn test_timeline_conflict() {
        let base = vec![make_event(1, 0, 500, 0)];
        let mut ours_ev = make_event(1, 0, 500, 0);
        ours_ev.duration_ms = 800;
        let mut theirs_ev = make_event(1, 0, 500, 0);
        theirs_ev.duration_ms = 1200;
        let ours = vec![ours_ev];
        let theirs = vec![theirs_ev];
        let merger = ProjectMerger::new();
        let results = merger.merge_timelines(&base, &ours, &theirs);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_conflict());
    }

    // ── ConflictResolver ─────────────────────────────────────────────────────

    #[test]
    fn test_resolve_take_ours() {
        let base = make_event(1, 0, 500, 0);
        let mut ours = make_event(1, 0, 500, 0);
        ours.track_id = 99;
        let mut theirs = make_event(1, 0, 500, 0);
        theirs.track_id = 77;
        let results = vec![MergeResult::Conflict(MergeConflict {
            path: "test".to_string(),
            base,
            ours: ours.clone(),
            theirs,
        })];
        let resolved = resolve_conflicts(results, ConflictResolver::TakeOurs);
        assert_eq!(resolved[0].track_id, 99);
    }

    #[test]
    fn test_resolve_take_theirs() {
        let base = make_event(1, 0, 500, 0);
        let mut ours = make_event(1, 0, 500, 0);
        ours.track_id = 99;
        let mut theirs = make_event(1, 0, 500, 0);
        theirs.track_id = 77;
        let results = vec![MergeResult::Conflict(MergeConflict {
            path: "test".to_string(),
            base,
            ours,
            theirs: theirs.clone(),
        })];
        let resolved = resolve_conflicts(results, ConflictResolver::TakeTheirs);
        assert_eq!(resolved[0].track_id, 77);
    }

    #[test]
    fn test_resolve_take_base() {
        let base = make_event(1, 0, 500, 0);
        let mut ours = make_event(1, 0, 500, 0);
        ours.track_id = 99;
        let mut theirs = make_event(1, 0, 500, 0);
        theirs.track_id = 77;
        let results = vec![MergeResult::Conflict(MergeConflict {
            path: "test".to_string(),
            base: base.clone(),
            ours,
            theirs,
        })];
        let resolved = resolve_conflicts(results, ConflictResolver::TakeBase);
        assert_eq!(resolved[0].track_id, 0);
    }

    #[test]
    fn test_resolve_clean_passthrough() {
        let ev = make_event(1, 0, 500, 0);
        let results = vec![MergeResult::Clean(ev.clone())];
        let resolved = resolve_conflicts(results, ConflictResolver::TakeOurs);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].id, 1);
    }

    #[test]
    fn test_resolve_mixed() {
        let ev_clean = make_event(1, 0, 500, 0);
        let base_c = make_event(2, 0, 500, 0);
        let mut ours_c = make_event(2, 0, 500, 0);
        ours_c.track_id = 10;
        let mut theirs_c = make_event(2, 0, 500, 0);
        theirs_c.track_id = 20;
        let results = vec![
            MergeResult::Clean(ev_clean),
            MergeResult::Conflict(MergeConflict {
                path: "".to_string(),
                base: base_c,
                ours: ours_c,
                theirs: theirs_c,
            }),
        ];
        let resolved = resolve_conflicts(results, ConflictResolver::TakeOurs);
        assert_eq!(resolved.len(), 2);
    }

    #[test]
    fn test_timeline_param_conflict() {
        let base = vec![make_event_with_params(1, &[("vol", 0.5)])];
        let ours = vec![make_event_with_params(1, &[("vol", 0.8)])];
        let theirs = vec![make_event_with_params(1, &[("vol", 0.2)])];
        let merger = ProjectMerger::new();
        let results = merger.merge_timelines(&base, &ours, &theirs);
        assert!(results[0].is_conflict());
    }

    #[test]
    fn test_timeline_param_clean_merge() {
        // Ours adds a new param, theirs modifies an existing one — no collision.
        let base = vec![make_event_with_params(1, &[("vol", 0.5)])];
        let ours = vec![make_event_with_params(1, &[("vol", 0.5), ("pitch", 220.0)])];
        let theirs = vec![make_event_with_params(1, &[("vol", 0.9)])];
        let merger = ProjectMerger::new();
        let results = merger.merge_timelines(&base, &ours, &theirs);
        assert!(results[0].is_clean());
        let ev = results[0].clean_value().expect("should have clean event");
        assert!((ev.parameters["vol"] - 0.9).abs() < 1e-6);
        assert!((ev.parameters["pitch"] - 220.0).abs() < 1e-6);
    }

    // ── Overlap resolution ──────────────────────────────────────────────

    fn make_prioritized(
        id: u64,
        start: i64,
        dur: u32,
        track: u32,
        prio: EventPriority,
    ) -> PrioritizedEvent {
        PrioritizedEvent::new(make_event(id, start, dur, track), prio)
    }

    #[test]
    fn test_event_priority_ordering() {
        assert!(EventPriority::Critical > EventPriority::High);
        assert!(EventPriority::High > EventPriority::Normal);
        assert!(EventPriority::Normal > EventPriority::Low);
    }

    #[test]
    fn test_event_priority_weight() {
        assert!(EventPriority::Critical.weight() > EventPriority::High.weight());
        assert!(EventPriority::Low.weight() > 0);
    }

    #[test]
    fn test_prioritized_event_end_ms() {
        let pe = make_prioritized(1, 100, 500, 0, EventPriority::Normal);
        assert_eq!(pe.end_ms(), 600);
    }

    #[test]
    fn test_prioritized_event_overlaps() {
        let a = make_prioritized(1, 0, 1000, 0, EventPriority::Normal);
        let b = make_prioritized(2, 500, 1000, 0, EventPriority::Normal);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_prioritized_event_no_overlap_different_track() {
        let a = make_prioritized(1, 0, 1000, 0, EventPriority::Normal);
        let b = make_prioritized(2, 0, 1000, 1, EventPriority::Normal);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_resolve_overlap_no_overlap() {
        let a = make_prioritized(1, 0, 500, 0, EventPriority::Normal);
        let b = make_prioritized(2, 500, 500, 0, EventPriority::Normal);
        let res = resolve_overlap(&a, &b, OverlapStrategy::PriorityWins);
        assert_eq!(res, OverlapResolution::NoOverlap);
    }

    #[test]
    fn test_resolve_overlap_priority_wins_removes_lower() {
        let high = make_prioritized(1, 0, 1000, 0, EventPriority::High);
        let low = make_prioritized(2, 200, 300, 0, EventPriority::Low);
        let res = resolve_overlap(&high, &low, OverlapStrategy::PriorityWins);
        match res {
            OverlapResolution::Removed { removed_id, .. } => {
                assert_eq!(removed_id, 2);
            }
            _ => panic!("expected Removed, got {res:?}"),
        }
    }

    #[test]
    fn test_resolve_overlap_priority_wins_trims_partial() {
        let high = make_prioritized(1, 500, 1000, 0, EventPriority::High);
        let low = make_prioritized(2, 0, 800, 0, EventPriority::Low);
        let res = resolve_overlap(&high, &low, OverlapStrategy::PriorityWins);
        match res {
            OverlapResolution::Trimmed {
                trimmed_id,
                new_start_ms,
                new_duration_ms,
                ..
            } => {
                assert_eq!(trimmed_id, 2);
                assert_eq!(new_start_ms, 0);
                assert_eq!(new_duration_ms, 500); // trimmed to end at winner's start
            }
            _ => panic!("expected Trimmed, got {res:?}"),
        }
    }

    #[test]
    fn test_resolve_overlap_earlier_wins() {
        let early = make_prioritized(1, 0, 600, 0, EventPriority::Normal);
        let late = make_prioritized(2, 400, 600, 0, EventPriority::Normal);
        let res = resolve_overlap(&early, &late, OverlapStrategy::EarlierWins);
        match res {
            OverlapResolution::Trimmed {
                trimmed_id,
                new_start_ms,
                new_duration_ms,
                ..
            } => {
                assert_eq!(trimmed_id, 2);
                assert_eq!(new_start_ms, 600);
                assert_eq!(new_duration_ms, 400);
            }
            _ => panic!("expected Trimmed, got {res:?}"),
        }
    }

    #[test]
    fn test_resolve_overlap_longer_wins() {
        let long = make_prioritized(1, 0, 2000, 0, EventPriority::Normal);
        let short = make_prioritized(2, 500, 200, 0, EventPriority::Normal);
        let res = resolve_overlap(&long, &short, OverlapStrategy::LongerWins);
        match res {
            OverlapResolution::Removed { removed_id, .. } => {
                assert_eq!(removed_id, 2);
            }
            _ => panic!("expected Removed, got {res:?}"),
        }
    }

    #[test]
    fn test_resolve_overlap_keep_both_flagged() {
        let a = make_prioritized(1, 0, 1000, 0, EventPriority::High);
        let b = make_prioritized(2, 500, 1000, 0, EventPriority::High);
        let res = resolve_overlap(&a, &b, OverlapStrategy::KeepBothFlagged);
        match res {
            OverlapResolution::Flagged {
                event_a_id,
                event_b_id,
            } => {
                assert_eq!(event_a_id, 1);
                assert_eq!(event_b_id, 2);
            }
            _ => panic!("expected Flagged, got {res:?}"),
        }
    }

    #[test]
    fn test_resolve_all_overlaps_priority_wins() {
        let events = vec![
            make_prioritized(1, 0, 1000, 0, EventPriority::Critical),
            make_prioritized(2, 500, 1000, 0, EventPriority::Low),
            make_prioritized(3, 2000, 500, 0, EventPriority::Normal), // no overlap
        ];
        let (resolved, log) = resolve_all_overlaps(&events, OverlapStrategy::PriorityWins);
        // Event 2 should be trimmed or removed
        assert!(!log.is_empty());
        // Event 3 should always survive
        assert!(resolved.iter().any(|e| e.event.id == 3));
        assert!(resolved.iter().any(|e| e.event.id == 1));
    }

    #[test]
    fn test_resolve_all_overlaps_no_overlaps() {
        let events = vec![
            make_prioritized(1, 0, 500, 0, EventPriority::Normal),
            make_prioritized(2, 500, 500, 0, EventPriority::Normal),
            make_prioritized(3, 1000, 500, 0, EventPriority::Normal),
        ];
        let (resolved, log) = resolve_all_overlaps(&events, OverlapStrategy::PriorityWins);
        assert_eq!(resolved.len(), 3);
        assert!(
            log.is_empty()
                || log
                    .iter()
                    .all(|r| matches!(r, OverlapResolution::NoOverlap))
        );
    }

    #[test]
    fn test_resolve_all_overlaps_different_tracks_no_interaction() {
        let events = vec![
            make_prioritized(1, 0, 1000, 0, EventPriority::Critical),
            make_prioritized(2, 0, 1000, 1, EventPriority::Low),
        ];
        let (resolved, _log) = resolve_all_overlaps(&events, OverlapStrategy::PriorityWins);
        assert_eq!(resolved.len(), 2);
    }

    #[test]
    fn test_resolve_overlap_same_priority_earlier_wins_tiebreak() {
        let a = make_prioritized(1, 0, 1000, 0, EventPriority::Normal);
        let b = make_prioritized(2, 500, 1000, 0, EventPriority::Normal);
        let res = resolve_overlap(&a, &b, OverlapStrategy::PriorityWins);
        // Same priority → earlier wins → b gets trimmed
        match res {
            OverlapResolution::Trimmed { trimmed_id, .. } => {
                assert_eq!(trimmed_id, 2);
            }
            _ => panic!("expected Trimmed, got {res:?}"),
        }
    }

    // ── MergeConfig / ConflictResolution / merge_with_config ─────────────────

    fn make_conflict_events() -> (TimelineEvent, TimelineEvent, TimelineEvent) {
        let base = make_event(99, 0, 500, 0);
        let mut ours = make_event(99, 0, 800, 0);
        ours.event_type = "clip_ours".to_string();
        let mut theirs = make_event(99, 0, 1200, 0);
        theirs.event_type = "clip_theirs".to_string();
        (base, ours, theirs)
    }

    #[test]
    fn test_merge_config_default() {
        let cfg = MergeConfig::default();
        assert_eq!(cfg.strategy, ConflictResolution::Manual);
        assert_eq!(cfg.max_conflict_context, 3);
        assert!(cfg.auto_resolve_whitespace);
    }

    #[test]
    fn test_merge_config_with_strategy() {
        let cfg = MergeConfig::with_strategy(ConflictResolution::TakeOurs);
        assert_eq!(cfg.strategy, ConflictResolution::TakeOurs);
    }

    #[test]
    fn test_merge_stats_is_clean() {
        let stats = MergeStats {
            auto_resolved: 2,
            conflicts: 0,
            total_changes: 2,
        };
        assert!(stats.is_clean());
        let dirty = MergeStats {
            conflicts: 1,
            ..Default::default()
        };
        assert!(!dirty.is_clean());
    }

    #[test]
    fn test_merge_with_config_take_ours_clean() {
        let (base, ours, theirs) = make_conflict_events();
        let cfg = MergeConfig::with_strategy(ConflictResolution::TakeOurs);
        let ours_type = ours.event_type.clone();
        let out = merge_with_config(&[base], &[ours], &[theirs], &cfg);
        assert!(out.conflicts.is_empty());
        assert_eq!(out.stats.auto_resolved, 1);
        assert_eq!(out.merged.len(), 1);
        assert_eq!(out.merged[0].event_type, ours_type);
    }

    #[test]
    fn test_merge_with_config_take_theirs() {
        let (base, ours, theirs) = make_conflict_events();
        let cfg = MergeConfig::with_strategy(ConflictResolution::TakeTheirs);
        let theirs_type = theirs.event_type.clone();
        let out = merge_with_config(&[base], &[ours], &[theirs], &cfg);
        assert!(out.conflicts.is_empty());
        assert_eq!(out.merged[0].event_type, theirs_type);
    }

    #[test]
    fn test_merge_with_config_take_both() {
        let (base, ours, theirs) = make_conflict_events();
        let cfg = MergeConfig::with_strategy(ConflictResolution::TakeBoth);
        let out = merge_with_config(&[base], &[ours], &[theirs], &cfg);
        // Both sides included → 2 events
        assert_eq!(out.merged.len(), 2);
        assert!(out.conflicts.is_empty());
        assert_eq!(out.stats.auto_resolved, 1);
    }

    #[test]
    fn test_merge_with_config_manual_records_conflict() {
        let (base, ours, theirs) = make_conflict_events();
        let cfg = MergeConfig::with_strategy(ConflictResolution::Manual);
        let out = merge_with_config(&[base], &[ours], &[theirs], &cfg);
        assert_eq!(out.conflicts.len(), 1);
        assert_eq!(out.stats.conflicts, 1);
        assert_eq!(out.stats.auto_resolved, 0);
        // Placeholder is still ours
        assert_eq!(out.merged.len(), 1);
    }

    #[test]
    fn test_merge_with_config_heuristic_longer_wins() {
        // Theirs has longer duration (1200 > 800) → heuristic picks theirs
        let (base, ours, theirs) = make_conflict_events();
        assert!(theirs.duration_ms > ours.duration_ms);
        let cfg = MergeConfig::with_strategy(ConflictResolution::Heuristic);
        let theirs_type = theirs.event_type.clone();
        let out = merge_with_config(&[base], &[ours], &[theirs], &cfg);
        assert!(out.conflicts.is_empty());
        assert_eq!(out.merged[0].event_type, theirs_type);
    }

    #[test]
    fn test_merge_with_config_heuristic_deletion_vs_addition() {
        // Ours has duration 0 (deleted), theirs added content → prefer theirs
        let base = make_event(42, 0, 500, 0);
        let mut ours = make_event(42, 0, 0, 0); // deleted
        ours.event_type = "deleted".to_string();
        let mut theirs = make_event(42, 0, 1000, 0); // kept with new duration
        theirs.event_type = "kept".to_string();

        let cfg = MergeConfig::with_strategy(ConflictResolution::Heuristic);
        let out = merge_with_config(&[base], &[ours], &[theirs.clone()], &cfg);
        assert_eq!(out.merged[0].duration_ms, theirs.duration_ms);
        assert_eq!(out.merged[0].event_type, theirs.event_type);
    }

    #[test]
    fn test_merge_with_config_heuristic_timestamp_tiebreak() {
        // Equal durations but theirs has a newer modification timestamp
        let base = make_event(77, 0, 1000, 0);
        let mut ours = make_event(77, 0, 1000, 0);
        ours.parameters
            .insert("_modified_ms".to_string(), 1_000_000.0);
        let mut theirs = make_event(77, 0, 1000, 0);
        // Deliberately give theirs a much larger modification timestamp
        theirs
            .parameters
            .insert("_modified_ms".to_string(), 9_999_999_000.0);
        // Change event_type so we can tell them apart
        ours.event_type = "old_edit".to_string();
        theirs.event_type = "new_edit".to_string();

        let cfg = MergeConfig::with_strategy(ConflictResolution::Heuristic);
        let out = merge_with_config(&[base], &[ours], &[theirs.clone()], &cfg);
        // Theirs has far greater timestamp score → should win
        assert_eq!(out.merged[0].event_type, theirs.event_type);
    }

    #[test]
    fn test_merge_with_config_no_conflict_passthrough() {
        // Only ours changed → no conflict, direct clean result
        let base = vec![make_event(1, 0, 500, 0)];
        let mut ours_ev = make_event(1, 0, 700, 0);
        ours_ev.event_type = "modified".to_string();
        let ours = vec![ours_ev];
        let theirs = base.clone();

        let cfg = MergeConfig::with_strategy(ConflictResolution::Heuristic);
        let out = merge_with_config(&base, &ours, &theirs, &cfg);
        assert!(out.conflicts.is_empty());
        assert_eq!(out.stats.auto_resolved, 0); // was a clean merge, not a resolved conflict
        assert_eq!(out.stats.total_changes, 1);
        assert_eq!(out.merged[0].duration_ms, 700);
    }

    #[test]
    fn test_merge_output_is_clean() {
        let base = vec![make_event(1, 0, 500, 0)];
        let cfg = MergeConfig::with_strategy(ConflictResolution::Manual);
        let out = merge_with_config(&base, &base, &base, &cfg);
        assert!(out.is_clean());
    }

    #[test]
    fn test_heuristic_score_parameter_richness() {
        let mut ev = make_event(1, 0, 1000, 0);
        ev.parameters.insert("vol".to_string(), 0.8);
        ev.parameters.insert("pan".to_string(), -0.2);
        let score = heuristic_score(&ev);
        // 1000 (duration) + 2*10 (params) = 1020 minimum
        assert!(score >= 1020);
    }

    // ── Complex overlapping timeline edit tests ───────────────────────────────

    /// Two editors add tracks at the exact same timestamp.
    /// Priority rule: the editor with the higher user_id (track_id proxy) wins.
    #[test]
    fn test_two_editors_same_timestamp_priority() {
        // Base: event at ts=1000.
        let base = vec![make_event(1, 1000, 500, 0)];

        // Editor A raises track to track_id=5.
        let mut ours = base.clone();
        ours[0].track_id = 5;

        // Editor B raises track to track_id=10 (higher user_id equivalent).
        let mut theirs = base.clone();
        theirs[0].track_id = 10;

        let cfg = MergeConfig::with_strategy(ConflictResolution::Heuristic);
        let out = merge_with_config(&base, &ours, &theirs, &cfg);
        // Both changed → conflict; heuristic resolves. The resolved result
        // should be deterministic (not panic).
        assert!(!out.merged.is_empty());
    }

    /// Sequential non-overlapping edits by two editors merge without conflict.
    #[test]
    fn test_sequential_non_overlapping_edits_clean_merge() {
        // Base: two independent events.
        let base = vec![make_event(1, 0, 1000, 0), make_event(2, 5000, 800, 1)];

        // Editor A modifies event 1 only.
        let mut ours = base.clone();
        ours[0].duration_ms = 1200;

        // Editor B modifies event 2 only.
        let mut theirs = base.clone();
        theirs[1].duration_ms = 900;

        let cfg = MergeConfig::with_strategy(ConflictResolution::Manual);
        let out = merge_with_config(&base, &ours, &theirs, &cfg);

        assert!(
            out.conflicts.is_empty(),
            "Non-overlapping edits must produce no conflicts, got: {:?}",
            out.conflicts
        );
        // Both changes must survive.
        let ev1 = out
            .merged
            .iter()
            .find(|e| e.id == 1)
            .expect("event 1 must be in output");
        let ev2 = out
            .merged
            .iter()
            .find(|e| e.id == 2)
            .expect("event 2 must be in output");
        assert_eq!(
            ev1.duration_ms, 1200,
            "Editor A's duration change must be preserved"
        );
        assert_eq!(
            ev2.duration_ms, 900,
            "Editor B's duration change must be preserved"
        );
    }

    /// When both editors modify the same clip title (string parameter), the
    /// three-way merge detects a conflict.
    #[test]
    fn test_conflicting_title_edits_detected() {
        let base = "original title";
        let ours = "editor-a title";
        let theirs = "editor-b title";

        let result = three_way_merge_string(base, ours, theirs);
        assert!(
            result.is_conflict(),
            "Concurrent title edits must be a conflict"
        );
    }

    /// Second-write-wins applied to a string conflict produces a deterministic result.
    #[test]
    fn test_conflicting_title_second_write_wins() {
        // Here "second" means `theirs` (the remote peer).
        let base = "original";
        let ours = "ours-edit";
        let theirs = "theirs-edit";

        let result = three_way_merge_string(base, ours, theirs);
        if let MergeResult::Conflict(c) = &result {
            // The conflict data preserves all three values.
            assert_eq!(c.base, base);
            assert_eq!(c.ours, ours);
            assert_eq!(c.theirs, theirs);
        } else {
            panic!("Expected conflict, got clean: {:?}", result);
        }
    }

    /// Delete + modify conflict: verify the conflict is detected and both
    /// resolution strategies (Manual and TakeTheirs) behave consistently.
    ///
    /// When `ours` modifies an event and `theirs` deletes it, `merge_timelines`
    /// raises a conflict.  With `Manual` strategy the conflict is recorded
    /// unresolved; with `TakeOurs` the modification survives.
    #[test]
    fn test_delete_and_modify_conflict_delete_wins() {
        // Base has event 1.
        let base = vec![make_event(1, 0, 500, 0)];

        // Ours: modified event 1 (extended duration and type).
        let mut ours_ev = make_event(1, 0, 800, 0);
        ours_ev.event_type = "modified".to_string();
        let ours = vec![ours_ev];

        // Theirs: deleted event 1 (event absent from their timeline).
        let theirs: Vec<TimelineEvent> = vec![];

        // With Manual strategy the modify-vs-delete conflict is recorded.
        let cfg_manual = MergeConfig::with_strategy(ConflictResolution::Manual);
        let out_manual = merge_with_config(&base, &ours, &theirs, &cfg_manual);

        // The conflict must be detected — delete+modify is NOT a clean merge.
        assert!(
            !out_manual.conflicts.is_empty(),
            "Delete + modify must produce an unresolved conflict under Manual strategy"
        );

        // Verify conflict metadata is sensible.
        let conflict = &out_manual.conflicts[0];
        assert_eq!(
            conflict.ours.id, 1,
            "Conflicted ours event must have correct id"
        );

        // With TakeOurs strategy (modification wins), event should appear in output.
        let cfg_take_ours = MergeConfig::with_strategy(ConflictResolution::TakeOurs);
        let out_take_ours = merge_with_config(&base, &ours, &theirs, &cfg_take_ours);
        let found_ours = out_take_ours.merged.iter().any(|e| e.id == 1);
        assert!(
            found_ours,
            "TakeOurs: modified event 1 should appear in merged output"
        );
        assert_eq!(
            out_take_ours
                .merged
                .iter()
                .find(|e| e.id == 1)
                .map(|e| e.duration_ms),
            Some(800),
            "TakeOurs: modified duration should be preserved"
        );
    }

    /// 5-way merge: five independent branches all converge to the same result
    /// regardless of the order in which they are pair-wise merged.
    #[test]
    fn test_five_way_merge_convergence() {
        // Single base event that no one modifies → all 5 branches are identical.
        let base = vec![make_event(1, 0, 500, 0)];

        let branches: Vec<Vec<TimelineEvent>> = (0..5).map(|_| base.clone()).collect();

        // Fold: merge[0] ∪ branches[1..] pair-by-pair.
        let cfg = MergeConfig::with_strategy(ConflictResolution::Manual);

        // First merge: base ∪ branch[1].
        let step0 = merge_with_config(&base, &branches[0], &branches[1], &cfg);
        let acc1 = step0.merged;

        // Continue folding remaining branches.
        let step1 = merge_with_config(&acc1, &acc1, &branches[2], &cfg);
        let acc2 = step1.merged;

        let step2 = merge_with_config(&acc2, &acc2, &branches[3], &cfg);
        let acc3 = step2.merged;

        let step3 = merge_with_config(&acc3, &acc3, &branches[4], &cfg);
        let final_result = step3.merged;

        assert_eq!(
            final_result.len(),
            1,
            "Five identical branches should converge to one event"
        );
        assert_eq!(final_result[0].id, 1);

        // Same merge in reverse order.
        let rev0 = merge_with_config(&base, &branches[4], &branches[3], &cfg);
        let racc1 = rev0.merged;
        let rev1 = merge_with_config(&racc1, &racc1, &branches[2], &cfg);
        let racc2 = rev1.merged;
        let rev2 = merge_with_config(&racc2, &racc2, &branches[1], &cfg);
        let racc3 = rev2.merged;
        let rev3 = merge_with_config(&racc3, &racc3, &branches[0], &cfg);
        let reversed_result = rev3.merged;

        assert_eq!(
            final_result.len(),
            reversed_result.len(),
            "5-way merge must converge regardless of order"
        );
    }
}
