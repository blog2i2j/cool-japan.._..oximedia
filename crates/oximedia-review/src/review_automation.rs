//! Review automation: auto-trigger review sessions on new version uploads.
//!
//! A [`ReviewTrigger`] watches for new content version events and automatically
//! creates review sessions according to configurable rules.  Triggers can match
//! on content-ID patterns, file size thresholds, or arbitrary metadata predicates.
//! Multiple triggers can be registered in a [`TriggerRegistry`]; each is evaluated
//! in registration order and the first matching trigger fires.
//!
//! No async runtime is required — all state is synchronous in-memory.  Callers
//! integrate with their own event bus by calling [`TriggerRegistry::on_upload`].

use serde::{Deserialize, Serialize};

// ─── VersionUploadEvent ───────────────────────────────────────────────────────

/// Event raised when a new version of content is uploaded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionUploadEvent {
    /// Unique event identifier.
    pub event_id: u64,
    /// Content identifier (e.g. asset ID or file path).
    pub content_id: String,
    /// Human-readable version label (e.g. "v3", "2024-01-15-rc1").
    pub version_label: String,
    /// Wall-clock time the upload completed (Unix epoch seconds).
    pub uploaded_at: u64,
    /// File size in bytes.
    pub file_size_bytes: u64,
    /// Arbitrary key/value metadata attached to the upload.
    pub metadata: Vec<(String, String)>,
}

impl VersionUploadEvent {
    /// Create a minimal upload event.
    #[must_use]
    pub fn new(
        event_id: u64,
        content_id: impl Into<String>,
        version_label: impl Into<String>,
        uploaded_at: u64,
    ) -> Self {
        Self {
            event_id,
            content_id: content_id.into(),
            version_label: version_label.into(),
            uploaded_at,
            file_size_bytes: 0,
            metadata: Vec::new(),
        }
    }

    /// Set the file size (builder pattern).
    #[must_use]
    pub fn with_file_size(mut self, bytes: u64) -> Self {
        self.file_size_bytes = bytes;
        self
    }

    /// Append a metadata key/value pair (builder pattern).
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.push((key.into(), value.into()));
        self
    }

    /// Look up a metadata value by key.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

// ─── TriggerCondition ─────────────────────────────────────────────────────────

/// A predicate that decides whether an upload event should trigger a review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerCondition {
    /// Always fire (matches every upload).
    Always,
    /// Fire only when the content ID starts with the given prefix.
    ContentIdPrefix(String),
    /// Fire only when the content ID contains the given substring.
    ContentIdContains(String),
    /// Fire when the file size is at least `min_bytes`.
    MinFileSize(u64),
    /// Fire when the file size is at most `max_bytes`.
    MaxFileSize(u64),
    /// Fire when the version label contains the given substring (case-insensitive).
    VersionLabelContains(String),
    /// Fire when a metadata key has the specified value.
    MetadataEquals {
        /// Metadata key to match.
        key: String,
        /// Expected metadata value.
        value: String,
    },
    /// Fire when **all** inner conditions match.
    All(Vec<TriggerCondition>),
    /// Fire when **any** inner condition matches.
    Any(Vec<TriggerCondition>),
}

impl TriggerCondition {
    /// Evaluate this condition against `event`, returning `true` if it matches.
    #[must_use]
    pub fn matches(&self, event: &VersionUploadEvent) -> bool {
        match self {
            Self::Always => true,
            Self::ContentIdPrefix(prefix) => event.content_id.starts_with(prefix.as_str()),
            Self::ContentIdContains(sub) => event.content_id.contains(sub.as_str()),
            Self::MinFileSize(min) => event.file_size_bytes >= *min,
            Self::MaxFileSize(max) => event.file_size_bytes <= *max,
            Self::VersionLabelContains(sub) => event
                .version_label
                .to_ascii_lowercase()
                .contains(sub.to_ascii_lowercase().as_str()),
            Self::MetadataEquals { key, value } => event
                .get_metadata(key)
                .map(|v| v == value.as_str())
                .unwrap_or(false),
            Self::All(conditions) => conditions.iter().all(|c| c.matches(event)),
            Self::Any(conditions) => conditions.iter().any(|c| c.matches(event)),
        }
    }
}

// ─── AutoReviewConfig ─────────────────────────────────────────────────────────

/// Configuration produced when a trigger fires — describes the review session to create.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoReviewConfig {
    /// Title template for the new review session.
    /// Use `{content_id}` and `{version}` as placeholders.
    pub title_template: String,
    /// Workflow type string (e.g. "Simple", "MultiStage").
    pub workflow_type: String,
    /// List of reviewer emails to auto-invite.
    pub reviewer_emails: Vec<String>,
    /// Optional deadline offset in seconds from the upload time.
    pub deadline_offset_s: Option<u64>,
    /// Priority (0 = lowest, 255 = highest).
    pub priority: u8,
}

impl AutoReviewConfig {
    /// Create a simple auto-review config.
    #[must_use]
    pub fn new(title_template: impl Into<String>) -> Self {
        Self {
            title_template: title_template.into(),
            workflow_type: "Simple".to_string(),
            reviewer_emails: Vec::new(),
            deadline_offset_s: None,
            priority: 128,
        }
    }

    /// Expand `{content_id}` and `{version}` placeholders in the title template.
    #[must_use]
    pub fn expand_title(&self, event: &VersionUploadEvent) -> String {
        self.title_template
            .replace("{content_id}", &event.content_id)
            .replace("{version}", &event.version_label)
    }

    /// Add a reviewer email (builder pattern).
    #[must_use]
    pub fn with_reviewer(mut self, email: impl Into<String>) -> Self {
        self.reviewer_emails.push(email.into());
        self
    }

    /// Set the deadline offset (builder pattern).
    #[must_use]
    pub fn with_deadline_offset_s(mut self, seconds: u64) -> Self {
        self.deadline_offset_s = Some(seconds);
        self
    }
}

// ─── ReviewTrigger ────────────────────────────────────────────────────────────

/// A named trigger that fires a review creation when its condition is met.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewTrigger {
    /// Unique trigger name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Condition evaluated on each upload event.
    pub condition: TriggerCondition,
    /// Review configuration to apply when the trigger fires.
    pub config: AutoReviewConfig,
    /// Whether this trigger is currently active.
    pub enabled: bool,
    /// Number of times this trigger has fired.
    pub fire_count: u64,
}

impl ReviewTrigger {
    /// Create a new enabled trigger.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        condition: TriggerCondition,
        config: AutoReviewConfig,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            condition,
            config,
            enabled: true,
            fire_count: 0,
        }
    }

    /// Disable this trigger (it will no longer match events).
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable this trigger.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Evaluate and, if `enabled` and the condition matches, record a fire and
    /// return the expanded [`AutoReviewConfig`].
    pub fn evaluate(&mut self, event: &VersionUploadEvent) -> Option<AutoReviewConfig> {
        if self.enabled && self.condition.matches(event) {
            self.fire_count += 1;
            Some(self.config.clone())
        } else {
            None
        }
    }
}

// ─── AutomationResult ─────────────────────────────────────────────────────────

/// Outcome of processing an upload event through the trigger registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationResult {
    /// Event that was processed.
    pub event_id: u64,
    /// Name of the trigger that fired, if any.
    pub triggered_by: Option<String>,
    /// Expanded review configuration, if a trigger fired.
    pub review_config: Option<AutoReviewConfig>,
    /// Deadline timestamp (= `event.uploaded_at + deadline_offset_s`), if set.
    pub deadline_ts: Option<u64>,
}

impl AutomationResult {
    /// Returns `true` if a trigger fired for this event.
    #[must_use]
    pub fn fired(&self) -> bool {
        self.triggered_by.is_some()
    }
}

// ─── TriggerRegistry ─────────────────────────────────────────────────────────

/// Registry holding a list of named [`ReviewTrigger`]s evaluated in order.
///
/// The **first** matching (enabled) trigger fires; subsequent triggers are
/// not evaluated.  Use [`TriggerRegistry::on_upload`] to process a new event.
#[derive(Debug, Default)]
pub struct TriggerRegistry {
    triggers: Vec<ReviewTrigger>,
}

impl TriggerRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            triggers: Vec::new(),
        }
    }

    /// Register a trigger at the end of the evaluation order.
    pub fn register(&mut self, trigger: ReviewTrigger) {
        self.triggers.push(trigger);
    }

    /// Remove a trigger by name.  Returns `true` if it existed.
    pub fn unregister(&mut self, name: &str) -> bool {
        let before = self.triggers.len();
        self.triggers.retain(|t| t.name != name);
        self.triggers.len() < before
    }

    /// Enable or disable a trigger by name.
    pub fn set_enabled(&mut self, name: &str, enabled: bool) {
        if let Some(t) = self.triggers.iter_mut().find(|t| t.name == name) {
            t.enabled = enabled;
        }
    }

    /// Process a [`VersionUploadEvent`], returning the first-match result.
    pub fn on_upload(&mut self, event: &VersionUploadEvent) -> AutomationResult {
        for trigger in &mut self.triggers {
            if let Some(config) = trigger.evaluate(event) {
                let deadline_ts = config
                    .deadline_offset_s
                    .map(|offset| event.uploaded_at.saturating_add(offset));
                return AutomationResult {
                    event_id: event.event_id,
                    triggered_by: Some(trigger.name.clone()),
                    review_config: Some(config),
                    deadline_ts,
                };
            }
        }
        AutomationResult {
            event_id: event.event_id,
            triggered_by: None,
            review_config: None,
            deadline_ts: None,
        }
    }

    /// Number of triggers registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.triggers.len()
    }

    /// Returns `true` if no triggers are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triggers.is_empty()
    }

    /// Borrow a trigger by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ReviewTrigger> {
        self.triggers.iter().find(|t| t.name == name)
    }

    /// Total number of fires across all triggers.
    #[must_use]
    pub fn total_fires(&self) -> u64 {
        self.triggers.iter().map(|t| t.fire_count).sum()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(id: u64, content_id: &str, version: &str) -> VersionUploadEvent {
        VersionUploadEvent::new(id, content_id, version, 1_700_000_000 + id)
    }

    fn make_trigger(name: &str, cond: TriggerCondition) -> ReviewTrigger {
        ReviewTrigger::new(
            name,
            "test trigger",
            cond,
            AutoReviewConfig::new("Review: {content_id} {version}"),
        )
    }

    #[test]
    fn condition_always_matches() {
        let evt = make_event(1, "asset-001", "v1");
        assert!(TriggerCondition::Always.matches(&evt));
    }

    #[test]
    fn condition_content_id_prefix() {
        let evt = make_event(1, "vfx/shot-010", "v2");
        assert!(TriggerCondition::ContentIdPrefix("vfx/".to_string()).matches(&evt));
        assert!(!TriggerCondition::ContentIdPrefix("audio/".to_string()).matches(&evt));
    }

    #[test]
    fn condition_content_id_contains() {
        let evt = make_event(1, "dailies/scene-03/clip-A", "v1");
        assert!(TriggerCondition::ContentIdContains("scene-03".to_string()).matches(&evt));
        assert!(!TriggerCondition::ContentIdContains("scene-99".to_string()).matches(&evt));
    }

    #[test]
    fn condition_min_file_size() {
        let evt = make_event(1, "a", "v1").with_file_size(500_000);
        assert!(TriggerCondition::MinFileSize(100_000).matches(&evt));
        assert!(!TriggerCondition::MinFileSize(1_000_000).matches(&evt));
    }

    #[test]
    fn condition_version_label_contains_case_insensitive() {
        let evt = make_event(1, "a", "v3-RC1");
        assert!(TriggerCondition::VersionLabelContains("rc1".to_string()).matches(&evt));
        assert!(!TriggerCondition::VersionLabelContains("final".to_string()).matches(&evt));
    }

    #[test]
    fn condition_metadata_equals() {
        let evt = make_event(1, "a", "v1").with_metadata("env", "prod");
        let cond = TriggerCondition::MetadataEquals {
            key: "env".to_string(),
            value: "prod".to_string(),
        };
        assert!(cond.matches(&evt));
        let cond_no = TriggerCondition::MetadataEquals {
            key: "env".to_string(),
            value: "dev".to_string(),
        };
        assert!(!cond_no.matches(&evt));
    }

    #[test]
    fn condition_all() {
        let evt = make_event(1, "vfx/clip", "v1").with_file_size(200_000);
        let cond = TriggerCondition::All(vec![
            TriggerCondition::ContentIdPrefix("vfx/".to_string()),
            TriggerCondition::MinFileSize(100_000),
        ]);
        assert!(cond.matches(&evt));
        let cond_fail = TriggerCondition::All(vec![
            TriggerCondition::ContentIdPrefix("vfx/".to_string()),
            TriggerCondition::MinFileSize(1_000_000),
        ]);
        assert!(!cond_fail.matches(&evt));
    }

    #[test]
    fn condition_any() {
        let evt = make_event(1, "audio/mix", "v1");
        let cond = TriggerCondition::Any(vec![
            TriggerCondition::ContentIdPrefix("vfx/".to_string()),
            TriggerCondition::ContentIdPrefix("audio/".to_string()),
        ]);
        assert!(cond.matches(&evt));
        let cond_fail = TriggerCondition::Any(vec![
            TriggerCondition::ContentIdPrefix("vfx/".to_string()),
            TriggerCondition::ContentIdPrefix("3d/".to_string()),
        ]);
        assert!(!cond_fail.matches(&evt));
    }

    #[test]
    fn trigger_fires_and_increments_count() {
        let mut t = make_trigger("t1", TriggerCondition::Always);
        let evt = make_event(1, "clip", "v1");
        let result = t.evaluate(&evt);
        assert!(result.is_some());
        assert_eq!(t.fire_count, 1);
    }

    #[test]
    fn trigger_disabled_does_not_fire() {
        let mut t = make_trigger("t1", TriggerCondition::Always);
        t.disable();
        let evt = make_event(1, "clip", "v1");
        assert!(t.evaluate(&evt).is_none());
        assert_eq!(t.fire_count, 0);
    }

    #[test]
    fn registry_first_match_wins() {
        let mut reg = TriggerRegistry::new();
        reg.register(make_trigger("broad", TriggerCondition::Always));
        reg.register(make_trigger("narrow", TriggerCondition::Always));
        let evt = make_event(1, "clip", "v1");
        let result = reg.on_upload(&evt);
        assert_eq!(result.triggered_by.as_deref(), Some("broad"));
        assert_eq!(reg.get("broad").map(|t| t.fire_count), Some(1));
        assert_eq!(reg.get("narrow").map(|t| t.fire_count), Some(0));
    }

    #[test]
    fn registry_no_match_returns_not_fired() {
        let mut reg = TriggerRegistry::new();
        reg.register(make_trigger(
            "vfx-only",
            TriggerCondition::ContentIdPrefix("vfx/".to_string()),
        ));
        let evt = make_event(1, "audio/mix", "v1");
        let result = reg.on_upload(&evt);
        assert!(!result.fired());
        assert_eq!(reg.total_fires(), 0);
    }

    #[test]
    fn registry_unregister() {
        let mut reg = TriggerRegistry::new();
        reg.register(make_trigger("t1", TriggerCondition::Always));
        assert_eq!(reg.len(), 1);
        assert!(reg.unregister("t1"));
        assert!(reg.is_empty());
        assert!(!reg.unregister("t1")); // already gone
    }

    #[test]
    fn expand_title_substitutes_placeholders() {
        let cfg = AutoReviewConfig::new("Review {content_id} — {version}");
        let evt = make_event(1, "vfx/shot-010", "v4");
        assert_eq!(cfg.expand_title(&evt), "Review vfx/shot-010 — v4");
    }

    #[test]
    fn deadline_ts_computed_when_offset_set() {
        let mut reg = TriggerRegistry::new();
        let config = AutoReviewConfig::new("{version}").with_deadline_offset_s(3600);
        reg.register(ReviewTrigger::new(
            "timed",
            "timed trigger",
            TriggerCondition::Always,
            config,
        ));
        let evt = make_event(42, "clip", "v1");
        let result = reg.on_upload(&evt);
        assert_eq!(result.deadline_ts, Some(1_700_000_000 + 42 + 3600));
    }
}
