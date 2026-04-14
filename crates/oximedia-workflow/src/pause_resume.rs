//! Workflow pause/resume support with in-memory checkpoint serialisation.
//!
//! [`PauseCheckpoint`] captures the state of a workflow at the moment it is
//! paused (step index, completed steps, pending steps, and arbitrary key-value
//! state data). [`PauseCheckpointStore`] manages a collection of such
//! checkpoints keyed by workflow ID.
//!
//! Serialisation is implemented by hand (no `serde` derive) so there are no
//! additional dependencies.
//!
//! # Example
//!
//! ```rust
//! use oximedia_workflow::pause_resume::{PauseCheckpoint, PauseCheckpointStore};
//!
//! let cp = PauseCheckpoint::new("wf-001", 2)
//!     .with_completed("ingest")
//!     .with_completed("analyse")
//!     .with_pending("transcode")
//!     .with_pending("upload")
//!     .with_state("source_path", "/mnt/raw/clip.mxf")
//!     .with_reason("Awaiting approval");
//!
//! let json = cp.to_json();
//! let restored = PauseCheckpoint::from_json(&json).expect("round-trip");
//! assert_eq!(restored.workflow_id, "wf-001");
//! assert_eq!(restored.remaining_steps(), 2);
//! ```

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// PauseCheckpoint
// ---------------------------------------------------------------------------

/// A snapshot of workflow state recorded when the workflow is paused.
#[derive(Debug, Clone, PartialEq)]
pub struct PauseCheckpoint {
    /// Identifier of the paused workflow.
    pub workflow_id: String,
    /// Zero-based index of the step at which execution was paused.
    pub step_index: usize,
    /// Names of steps that had already completed before pausing.
    pub completed_steps: Vec<String>,
    /// Names of steps still remaining to be executed.
    pub pending_steps: Vec<String>,
    /// Arbitrary serialised outputs or metadata from completed steps.
    pub state_data: HashMap<String, String>,
    /// Unix timestamp (milliseconds) when the checkpoint was recorded.
    pub paused_at: u64,
    /// Human-readable reason the workflow was paused.
    pub pause_reason: String,
}

impl PauseCheckpoint {
    /// Create a new checkpoint for `workflow_id` paused at `step_index`.
    ///
    /// `paused_at` is set to the current wall-clock time in milliseconds.
    #[must_use]
    pub fn new(workflow_id: impl Into<String>, step_index: usize) -> Self {
        let paused_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            workflow_id: workflow_id.into(),
            step_index,
            completed_steps: Vec::new(),
            pending_steps: Vec::new(),
            state_data: HashMap::new(),
            paused_at,
            pause_reason: String::new(),
        }
    }

    /// Add a completed step name (builder pattern).
    #[must_use]
    pub fn with_completed(mut self, step: impl Into<String>) -> Self {
        self.completed_steps.push(step.into());
        self
    }

    /// Add a pending step name (builder pattern).
    #[must_use]
    pub fn with_pending(mut self, step: impl Into<String>) -> Self {
        self.pending_steps.push(step.into());
        self
    }

    /// Insert a key-value pair into the state data (builder pattern).
    #[must_use]
    pub fn with_state(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.state_data.insert(key.into(), value.into());
        self
    }

    /// Set the human-readable pause reason (builder pattern).
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.pause_reason = reason.into();
        self
    }

    /// Number of steps still pending (not yet executed).
    #[must_use]
    pub fn remaining_steps(&self) -> usize {
        self.pending_steps.len()
    }

    /// Progress as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when no steps have been completed and no steps are pending.
    #[must_use]
    pub fn progress_percent(&self) -> f32 {
        let done = self.completed_steps.len();
        let total = done + self.pending_steps.len();
        if total == 0 {
            return 0.0;
        }
        done as f32 / total as f32
    }

    // -----------------------------------------------------------------------
    // Hand-written JSON serialisation
    // -----------------------------------------------------------------------

    /// Serialise the checkpoint to a JSON string.
    ///
    /// The format is stable and suitable for long-term storage.
    #[must_use]
    pub fn to_json(&self) -> String {
        let completed = json_string_array(&self.completed_steps);
        let pending = json_string_array(&self.pending_steps);
        let state = json_string_map(&self.state_data);

        format!(
            concat!(
                "{{",
                "\"workflow_id\":{wid},",
                "\"step_index\":{si},",
                "\"completed_steps\":{completed},",
                "\"pending_steps\":{pending},",
                "\"state_data\":{state},",
                "\"paused_at\":{pa},",
                "\"pause_reason\":{reason}",
                "}}"
            ),
            wid = json_escape_string(&self.workflow_id),
            si = self.step_index,
            completed = completed,
            pending = pending,
            state = state,
            pa = self.paused_at,
            reason = json_escape_string(&self.pause_reason),
        )
    }

    /// Deserialise a checkpoint from a JSON string produced by [`to_json`].
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` when the JSON is malformed or a required field
    /// is missing.
    ///
    /// [`to_json`]: PauseCheckpoint::to_json
    pub fn from_json(json: &str) -> Result<Self, String> {
        let trimmed = json.trim();
        if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
            return Err("Not a JSON object".to_string());
        }

        let workflow_id =
            parse_json_string(trimmed, "workflow_id").ok_or("Missing field: workflow_id")?;
        let step_index =
            parse_json_u64(trimmed, "step_index").ok_or("Missing field: step_index")? as usize;
        let completed_steps =
            parse_json_string_array(trimmed, "completed_steps").unwrap_or_default();
        let pending_steps = parse_json_string_array(trimmed, "pending_steps").unwrap_or_default();
        let state_data = parse_json_string_map(trimmed, "state_data").unwrap_or_default();
        let paused_at = parse_json_u64(trimmed, "paused_at").unwrap_or(0);
        let pause_reason = parse_json_string(trimmed, "pause_reason").unwrap_or_default();

        Ok(Self {
            workflow_id,
            step_index,
            completed_steps,
            pending_steps,
            state_data,
            paused_at,
            pause_reason,
        })
    }
}

// ---------------------------------------------------------------------------
// PauseCheckpointStore
// ---------------------------------------------------------------------------

/// In-memory store for [`PauseCheckpoint`] instances.
///
/// Checkpoints are keyed by `workflow_id`.  Saving a checkpoint with an
/// existing key overwrites the previous one.
#[derive(Debug, Default)]
pub struct PauseCheckpointStore {
    checkpoints: HashMap<String, PauseCheckpoint>,
}

impl PauseCheckpointStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Save (insert or overwrite) a checkpoint.
    pub fn save(&mut self, checkpoint: PauseCheckpoint) {
        self.checkpoints
            .insert(checkpoint.workflow_id.clone(), checkpoint);
    }

    /// Load a checkpoint by workflow ID.
    #[must_use]
    pub fn load(&self, workflow_id: &str) -> Option<&PauseCheckpoint> {
        self.checkpoints.get(workflow_id)
    }

    /// Delete a checkpoint.  Returns `true` if the checkpoint existed.
    pub fn delete(&mut self, workflow_id: &str) -> bool {
        self.checkpoints.remove(workflow_id).is_some()
    }

    /// Return references to all stored checkpoints.
    #[must_use]
    pub fn list_paused(&self) -> Vec<&PauseCheckpoint> {
        self.checkpoints.values().collect()
    }

    /// Number of stored checkpoints.
    #[must_use]
    pub fn count(&self) -> usize {
        self.checkpoints.len()
    }
}

// ---------------------------------------------------------------------------
// PauseResumeController
// ---------------------------------------------------------------------------

/// High-level controller for pausing and resuming workflows by numeric ID.
///
/// Internally maps numeric workflow IDs to [`PauseCheckpoint`]s stored in a
/// [`PauseCheckpointStore`].  Numeric IDs are converted to strings for
/// compatibility with the checkpoint store's string-keyed API.
///
/// # Example
///
/// ```rust
/// use oximedia_workflow::pause_resume::PauseResumeController;
///
/// let mut ctrl = PauseResumeController::new();
/// ctrl.pause(42);
/// assert!(ctrl.is_paused(42));
/// ctrl.resume(42);
/// assert!(!ctrl.is_paused(42));
/// ```
#[derive(Debug, Default)]
pub struct PauseResumeController {
    store: PauseCheckpointStore,
}

impl PauseResumeController {
    /// Create an empty controller.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pause a workflow by its numeric ID.
    ///
    /// If the workflow is already paused this is a no-op (the checkpoint is
    /// updated with a fresh timestamp by replacing the entry).
    pub fn pause(&mut self, workflow_id: u64) {
        let key = workflow_id.to_string();
        let cp = PauseCheckpoint::new(key, 0).with_reason("Paused by PauseResumeController");
        self.store.save(cp);
    }

    /// Resume a workflow by removing its pause checkpoint.
    ///
    /// If the workflow is not currently paused this is a no-op.
    pub fn resume(&mut self, workflow_id: u64) {
        self.store.delete(&workflow_id.to_string());
    }

    /// Return `true` if a pause checkpoint exists for the given workflow ID.
    #[must_use]
    pub fn is_paused(&self, workflow_id: u64) -> bool {
        self.store.load(&workflow_id.to_string()).is_some()
    }

    /// Return the number of currently paused workflows.
    #[must_use]
    pub fn paused_count(&self) -> usize {
        self.store.count()
    }

    /// List all currently paused workflow IDs.
    #[must_use]
    pub fn paused_ids(&self) -> Vec<u64> {
        self.store
            .list_paused()
            .iter()
            .filter_map(|cp| cp.workflow_id.parse::<u64>().ok())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Hand-written JSON helpers (no serde dependency)
// ---------------------------------------------------------------------------

/// Produce a JSON-escaped quoted string: `"value"`.
fn json_escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Produce a JSON array of strings: `["a","b"]`.
fn json_string_array(items: &[String]) -> String {
    let mut out = String::from("[");
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&json_escape_string(item));
    }
    out.push(']');
    out
}

/// Produce a JSON object from a string map: `{"k":"v"}`.
fn json_string_map(map: &HashMap<String, String>) -> String {
    let mut out = String::from("{");
    let mut first = true;
    for (k, v) in map {
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(&json_escape_string(k));
        out.push(':');
        out.push_str(&json_escape_string(v));
    }
    out.push('}');
    out
}

// ---------------------------------------------------------------------------
// Minimal JSON parsing helpers
// These are intentionally simple — they support only the subset of JSON that
// `to_json` produces.
// ---------------------------------------------------------------------------

/// Find `"key":` in `json` and return the position just after the `:`.
fn find_value_start(json: &str, key: &str) -> Option<usize> {
    let search = format!("\"{}\":", key);
    json.find(&search).map(|pos| pos + search.len())
}

/// Parse a quoted string value for `key` from `json`.
fn parse_json_string(json: &str, key: &str) -> Option<String> {
    let after_colon = find_value_start(json, key)?;
    let rest = json[after_colon..].trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let inner = &rest[1..]; // skip opening quote
    let mut result = String::new();
    let mut chars = inner.chars().peekable();
    loop {
        match chars.next()? {
            '"' => break,
            '\\' => match chars.next()? {
                '"' => result.push('"'),
                '\\' => result.push('\\'),
                'n' => result.push('\n'),
                'r' => result.push('\r'),
                't' => result.push('\t'),
                c => result.push(c),
            },
            c => result.push(c),
        }
    }
    Some(result)
}

/// Parse an unsigned integer value for `key` from `json`.
fn parse_json_u64(json: &str, key: &str) -> Option<u64> {
    let after_colon = find_value_start(json, key)?;
    let rest = json[after_colon..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Parse an array of strings for `key` from `json`.
///
/// Supports the subset produced by [`json_string_array`].
fn parse_json_string_array(json: &str, key: &str) -> Option<Vec<String>> {
    let after_colon = find_value_start(json, key)?;
    let rest = json[after_colon..].trim_start();
    if !rest.starts_with('[') {
        return None;
    }
    let bracket_end = find_matching_bracket(rest, '[', ']')?;
    let inner = &rest[1..bracket_end]; // between [ and ]
    let mut result = Vec::new();
    let mut remaining = inner.trim();
    while !remaining.is_empty() {
        if !remaining.starts_with('"') {
            break;
        }
        let (s, consumed) = parse_one_json_string(remaining)?;
        result.push(s);
        remaining = remaining[consumed..].trim();
        if remaining.starts_with(',') {
            remaining = remaining[1..].trim();
        }
    }
    Some(result)
}

/// Parse a `{"k":"v", ...}` object for `key` from `json`.
///
/// Supports the subset produced by [`json_string_map`].
fn parse_json_string_map(json: &str, key: &str) -> Option<HashMap<String, String>> {
    let after_colon = find_value_start(json, key)?;
    let rest = json[after_colon..].trim_start();
    if !rest.starts_with('{') {
        return None;
    }
    let brace_end = find_matching_bracket(rest, '{', '}')?;
    let inner = &rest[1..brace_end]; // between { and }
    let mut result = HashMap::new();
    let mut remaining = inner.trim();
    while !remaining.is_empty() {
        if !remaining.starts_with('"') {
            break;
        }
        let (k, k_consumed) = parse_one_json_string(remaining)?;
        remaining = remaining[k_consumed..].trim();
        if !remaining.starts_with(':') {
            break;
        }
        remaining = remaining[1..].trim();
        let (v, v_consumed) = parse_one_json_string(remaining)?;
        result.insert(k, v);
        remaining = remaining[v_consumed..].trim();
        if remaining.starts_with(',') {
            remaining = remaining[1..].trim();
        }
    }
    Some(result)
}

/// Parse one `"..."` JSON string from the start of `s`.
///
/// Returns `(value, bytes_consumed)` including the surrounding quotes.
fn parse_one_json_string(s: &str) -> Option<(String, usize)> {
    if !s.starts_with('"') {
        return None;
    }
    let mut result = String::new();
    let mut chars = s[1..].char_indices();
    let mut consumed = 1usize; // opening quote
    loop {
        let (i, c) = chars.next()?;
        match c {
            '"' => {
                consumed += i + c.len_utf8();
                break;
            }
            '\\' => {
                let (_, next) = chars.next()?;
                consumed += 1;
                match next {
                    '"' => result.push('"'),
                    '\\' => result.push('\\'),
                    'n' => result.push('\n'),
                    'r' => result.push('\r'),
                    't' => result.push('\t'),
                    c => result.push(c),
                }
            }
            c => result.push(c),
        }
    }
    Some((result, consumed))
}

/// Find the index of the matching closing bracket/brace, handling nesting.
fn find_matching_bracket(s: &str, open: char, close: char) -> Option<usize> {
    let mut depth: usize = 0;
    let mut in_string = false;
    let mut prev_backslash = false;
    for (i, c) in s.char_indices() {
        if prev_backslash {
            prev_backslash = false;
            continue;
        }
        if in_string {
            if c == '\\' {
                prev_backslash = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
        } else if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_checkpoint_fields() {
        let cp = PauseCheckpoint::new("wf-001", 3);
        assert_eq!(cp.workflow_id, "wf-001");
        assert_eq!(cp.step_index, 3);
        assert!(cp.completed_steps.is_empty());
        assert!(cp.pending_steps.is_empty());
        assert!(cp.state_data.is_empty());
        assert!(cp.paused_at > 0);
        assert!(cp.pause_reason.is_empty());
    }

    #[test]
    fn test_progress_percent_partial() {
        let cp = PauseCheckpoint::new("wf-002", 2)
            .with_completed("step-a")
            .with_completed("step-b")
            .with_pending("step-c")
            .with_pending("step-d");
        let progress = cp.progress_percent();
        // 2 done / 4 total = 0.5
        assert!(
            (progress - 0.5).abs() < f32::EPSILON,
            "expected 0.5, got {progress}"
        );
    }

    #[test]
    fn test_progress_percent_all_done() {
        let cp = PauseCheckpoint::new("wf-003", 2)
            .with_completed("a")
            .with_completed("b");
        assert!((cp.progress_percent() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_progress_percent_none_done() {
        let cp = PauseCheckpoint::new("wf-004", 0)
            .with_pending("a")
            .with_pending("b");
        assert!((cp.progress_percent() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_to_json_from_json_round_trip() {
        let original = PauseCheckpoint::new("wf-roundtrip", 1)
            .with_completed("ingest")
            .with_pending("transcode")
            .with_pending("upload")
            .with_state("source", "/data/clip.mxf")
            .with_reason("Waiting for budget approval");

        let json = original.to_json();
        let restored = PauseCheckpoint::from_json(&json).expect("from_json should succeed");

        assert_eq!(restored.workflow_id, original.workflow_id);
        assert_eq!(restored.step_index, original.step_index);
        assert_eq!(restored.completed_steps, original.completed_steps);
        assert_eq!(restored.pending_steps, original.pending_steps);
        assert_eq!(restored.state_data, original.state_data);
        assert_eq!(restored.paused_at, original.paused_at);
        assert_eq!(restored.pause_reason, original.pause_reason);
    }

    #[test]
    fn test_save_load_delete() {
        let mut store = PauseCheckpointStore::new();
        let cp = PauseCheckpoint::new("wf-store", 0);

        store.save(cp.clone());
        assert_eq!(store.count(), 1);

        let loaded = store.load("wf-store").expect("should find it");
        assert_eq!(loaded.workflow_id, "wf-store");

        assert!(store.delete("wf-store"));
        assert_eq!(store.count(), 0);
        assert!(store.load("wf-store").is_none());
    }

    #[test]
    fn test_list_paused_returns_all() {
        let mut store = PauseCheckpointStore::new();
        store.save(PauseCheckpoint::new("wf-alpha", 0));
        store.save(PauseCheckpoint::new("wf-beta", 1));
        store.save(PauseCheckpoint::new("wf-gamma", 2));

        let list = store.list_paused();
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_remaining_steps() {
        let cp = PauseCheckpoint::new("wf-rem", 1)
            .with_pending("step-x")
            .with_pending("step-y")
            .with_pending("step-z");
        assert_eq!(cp.remaining_steps(), 3);
    }

    #[test]
    fn test_with_state() {
        let cp = PauseCheckpoint::new("wf-state", 0)
            .with_state("output_path", "/tmp/out.mp4")
            .with_state("fps", "24");
        assert_eq!(
            cp.state_data.get("output_path").map(String::as_str),
            Some("/tmp/out.mp4")
        );
        assert_eq!(cp.state_data.get("fps").map(String::as_str), Some("24"));
    }

    #[test]
    fn test_from_json_malformed_returns_err() {
        let result = PauseCheckpoint::from_json("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_nonexistent_returns_false() {
        let mut store = PauseCheckpointStore::new();
        assert!(!store.delete("ghost-workflow"));
    }
}
