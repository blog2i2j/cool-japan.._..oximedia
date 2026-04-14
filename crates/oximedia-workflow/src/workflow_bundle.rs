//! Portable workflow bundle import/export for OxiMedia.
//!
//! A [`WorkflowBundle`] is a self-contained, human-readable description of a
//! workflow that can be serialized to JSON, transmitted, stored in version
//! control, and imported into any OxiMedia installation.
//!
//! # Design
//!
//! The bundle uses string IDs and name-based dependency references instead of
//! UUIDs so that hand-editing is ergonomic.  All JSON serialization in this
//! module is **hand-written** — no `serde_json` derive is used — to remain
//! independent of the serde ecosystem for this lightweight data model.
//!
//! # Example
//!
//! ```rust
//! use oximedia_workflow::workflow_bundle::{WorkflowBundle, BundleStep};
//!
//! let mut bundle = WorkflowBundle {
//!     name: "ingest-pipeline".to_string(),
//!     version: "1.0.0".to_string(),
//!     steps: vec![
//!         BundleStep {
//!             id: "ingest".to_string(),
//!             action: "transcode".to_string(),
//!             params: vec![("input".to_string(), "/raw.mp4".to_string())],
//!             depends_on: vec![],
//!         },
//!         BundleStep {
//!             id: "upload".to_string(),
//!             action: "transfer".to_string(),
//!             params: vec![("dest".to_string(), "s3://bucket/".to_string())],
//!             depends_on: vec!["ingest".to_string()],
//!         },
//!     ],
//!     metadata: std::collections::HashMap::new(),
//! };
//!
//! let json = bundle.to_json();
//! let parsed = WorkflowBundle::from_json(&json).expect("parse");
//! assert_eq!(parsed.name, "ingest-pipeline");
//! assert_eq!(parsed.steps.len(), 2);
//!
//! bundle.validate().expect("validation should pass");
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// BundleStep
// ---------------------------------------------------------------------------

/// A single processing step within a [`WorkflowBundle`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleStep {
    /// Step identifier — must be unique within the bundle.
    pub id: String,
    /// Name of the action/handler to invoke (e.g. `"transcode"`, `"qc"`).
    pub action: String,
    /// Named parameters passed to the action.
    pub params: Vec<(String, String)>,
    /// IDs of other steps that must complete before this step can run.
    pub depends_on: Vec<String>,
}

// ---------------------------------------------------------------------------
// WorkflowBundle
// ---------------------------------------------------------------------------

/// A portable, self-contained workflow definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowBundle {
    /// Human-readable name for the workflow.
    pub name: String,
    /// Semantic version string (e.g. `"1.0.0"`).
    pub version: String,
    /// Ordered list of processing steps.
    pub steps: Vec<BundleStep>,
    /// Arbitrary key/value metadata (author, description, tags, …).
    pub metadata: HashMap<String, String>,
}

impl WorkflowBundle {
    // ------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------

    /// Serialize this bundle to a compact JSON string.
    ///
    /// The output can be fed back into [`WorkflowBundle::from_json`] for a
    /// lossless round-trip.
    #[must_use]
    pub fn to_json(&self) -> String {
        let steps_json = self
            .steps
            .iter()
            .map(bundle_step_to_json)
            .collect::<Vec<_>>()
            .join(",");

        let metadata_json = kv_map_to_json(&self.metadata);

        format!(
            r#"{{"name":"{}","version":"{}","steps":[{}],"metadata":{}}}"#,
            escape_json(&self.name),
            escape_json(&self.version),
            steps_json,
            metadata_json,
        )
    }

    /// Deserialize a bundle from a JSON string produced by [`WorkflowBundle::to_json`].
    ///
    /// # Errors
    ///
    /// Returns a descriptive error string if required fields are missing or
    /// cannot be parsed.
    pub fn from_json(s: &str) -> Result<Self, String> {
        let name = extract_string(s, "name").ok_or_else(|| "missing field: name".to_string())?;
        let version =
            extract_string(s, "version").ok_or_else(|| "missing field: version".to_string())?;

        let steps =
            extract_steps_array(s).ok_or_else(|| "missing or invalid field: steps".to_string())?;

        let metadata = extract_metadata_map(s).unwrap_or_default();

        Ok(Self {
            name,
            version,
            steps,
            metadata,
        })
    }

    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------

    /// Validate the bundle, checking for:
    ///
    /// 1. Duplicate step IDs.
    /// 2. Dependency references to unknown step IDs.
    ///
    /// Returns `Ok(())` when all checks pass, or `Err(errors)` where `errors`
    /// is a non-empty list of human-readable problem descriptions.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Build a set of known step IDs.
        let mut seen_ids: HashMap<&str, usize> = HashMap::new();
        for (i, step) in self.steps.iter().enumerate() {
            if step.id.is_empty() {
                errors.push(format!("step[{i}] has an empty id"));
                continue;
            }
            if let Some(prev) = seen_ids.insert(step.id.as_str(), i) {
                errors.push(format!(
                    "duplicate step id '{}' at indices {prev} and {i}",
                    step.id
                ));
            }
        }

        // Check dependency references.
        for step in &self.steps {
            for dep in &step.depends_on {
                if !seen_ids.contains_key(dep.as_str()) {
                    errors.push(format!(
                        "step '{}' depends_on unknown step id '{dep}'",
                        step.id
                    ));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    // ------------------------------------------------------------------
    // Convenience constructors and mutators
    // ------------------------------------------------------------------

    /// Create an empty bundle with the given name and version `"1.0.0"`.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: "1.0.0".to_string(),
            steps: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a sub-workflow by embedding its steps into this bundle.
    ///
    /// Steps from `workflow` are appended to this bundle's step list.
    /// If a step ID already exists in this bundle it is suffixed with
    /// `"_<n>"` to avoid collisions.
    pub fn add_workflow(&mut self, workflow: WorkflowBundle) {
        let existing_ids: std::collections::HashSet<String> =
            self.steps.iter().map(|s| s.id.clone()).collect();

        for mut step in workflow.steps {
            if existing_ids.contains(&step.id) {
                step.id = format!("{}_{}", step.id, self.steps.len());
            }
            self.steps.push(step);
        }
    }

    /// Serialize this bundle to JSON — alias for [`Self::to_json`].
    ///
    /// Provided so the task-specified API `export_json()` is available.
    #[must_use]
    pub fn export_json(&self) -> String {
        self.to_json()
    }
}

// ===========================================================================
// Hand-written JSON helpers
// ===========================================================================

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Escape a string for embedding in a JSON value.
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

/// Serialize a `Vec<(String, String)>` to a JSON array of `[k, v]` pairs.
fn params_to_json(params: &[(String, String)]) -> String {
    let items = params
        .iter()
        .map(|(k, v)| format!("[\"{}\",\"{}\"]", escape_json(k), escape_json(v)))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

/// Serialize a `Vec<String>` to a JSON array of strings.
fn string_vec_to_json(v: &[String]) -> String {
    let items = v
        .iter()
        .map(|s| format!("\"{}\"", escape_json(s)))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

/// Serialize a `HashMap<String, String>` to a JSON object.
fn kv_map_to_json(map: &HashMap<String, String>) -> String {
    let mut pairs: Vec<(&String, &String)> = map.iter().collect();
    // Sort for deterministic output.
    pairs.sort_by_key(|(k, _)| k.as_str());
    let items = pairs
        .iter()
        .map(|(k, v)| format!("\"{}\":\"{}\"", escape_json(k), escape_json(v)))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{items}}}")
}

/// Serialize a [`BundleStep`] to JSON.
fn bundle_step_to_json(step: &BundleStep) -> String {
    format!(
        r#"{{"id":"{}","action":"{}","params":{},"depends_on":{}}}"#,
        escape_json(&step.id),
        escape_json(&step.action),
        params_to_json(&step.params),
        string_vec_to_json(&step.depends_on),
    )
}

// ---------------------------------------------------------------------------
// Deserialization helpers
// ---------------------------------------------------------------------------

/// Extract a JSON string field from a JSON object string.
fn extract_string(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\":\"", key);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let mut value = String::new();
    let mut chars = rest.chars().peekable();
    loop {
        match chars.next()? {
            '"' => break,
            '\\' => match chars.next()? {
                '"' => value.push('"'),
                '\\' => value.push('\\'),
                'n' => value.push('\n'),
                'r' => value.push('\r'),
                't' => value.push('\t'),
                c => {
                    value.push('\\');
                    value.push(c);
                }
            },
            c => value.push(c),
        }
    }
    Some(value)
}

/// Extract the content between the first occurrence of `"key":[` and the
/// matching `]`, allowing one level of nesting.
fn extract_array_content<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{}\":[", key);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    // Find the matching `]`, tracking bracket depth.
    let mut depth = 1i32;
    // Skip inside strings (simple handling: track inside_string flag).
    let mut inside_string = false;
    let mut escape_next = false;
    for (i, c) in rest.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if inside_string {
            if c == '\\' {
                escape_next = true;
            } else if c == '"' {
                inside_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                inside_string = true;
            }
            '[' => {
                depth += 1;
            }
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&rest[..i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse the `steps` array from the bundle JSON.
fn extract_steps_array(json: &str) -> Option<Vec<BundleStep>> {
    let content = extract_array_content(json, "steps")?;
    if content.trim().is_empty() {
        return Some(Vec::new());
    }
    // Each step is a `{...}` object. We split by finding top-level `{...}` blocks.
    let mut steps = Vec::new();
    let mut remaining = content;
    loop {
        remaining = remaining.trim_start();
        if remaining.is_empty() {
            break;
        }
        if !remaining.starts_with('{') {
            break;
        }
        let (obj, rest) = extract_first_object(remaining)?;
        if let Some(step) = parse_bundle_step(obj) {
            steps.push(step);
        }
        remaining = rest.trim_start_matches(',').trim_start();
    }
    Some(steps)
}

/// Extract the first `{...}` object from `s` (handling nesting and strings).
/// Returns `(object_slice, remainder_after_object)`.
fn extract_first_object(s: &str) -> Option<(&str, &str)> {
    if !s.starts_with('{') {
        return None;
    }
    let mut depth = 0i32;
    let mut inside_string = false;
    let mut escape_next = false;
    for (i, c) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if inside_string {
            if c == '\\' {
                escape_next = true;
            } else if c == '"' {
                inside_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                inside_string = true;
            }
            '{' => {
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((&s[..=i], &s[i + 1..]));
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse a single [`BundleStep`] from its JSON object string.
fn parse_bundle_step(json: &str) -> Option<BundleStep> {
    let id = extract_string(json, "id")?;
    let action = extract_string(json, "action")?;
    let params = extract_params_array(json).unwrap_or_default();
    let depends_on = extract_string_array(json, "depends_on").unwrap_or_default();

    Some(BundleStep {
        id,
        action,
        params,
        depends_on,
    })
}

/// Parse `"params":[[k,v],...]` from a step JSON object.
fn extract_params_array(json: &str) -> Option<Vec<(String, String)>> {
    let content = extract_array_content(json, "params")?;
    if content.trim().is_empty() {
        return Some(Vec::new());
    }

    let mut pairs = Vec::new();
    let mut remaining = content;
    loop {
        remaining = remaining.trim_start();
        if remaining.is_empty() {
            break;
        }
        if !remaining.starts_with('[') {
            break;
        }
        // Find the inner `[k, v]` — two strings separated by a comma.
        let end = remaining.find(']')?;
        let inner = &remaining[1..end];
        // Parse two quoted strings.
        let k = extract_nth_string(inner, 0)?;
        let v = extract_nth_string(inner, 1)?;
        pairs.push((k, v));
        remaining = &remaining[end + 1..];
        remaining = remaining.trim_start_matches(',').trim_start();
    }
    Some(pairs)
}

/// Extract the nth quoted string (0-indexed) from a comma-separated list of
/// JSON strings like `"foo","bar"`.
fn extract_nth_string(s: &str, n: usize) -> Option<String> {
    // Collect all quoted strings in order, then return the nth one.
    let mut strings: Vec<String> = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        // Skip whitespace and commas
        while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t' || bytes[i] == b',') {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        if bytes[i] != b'"' {
            // Skip non-string characters
            i += 1;
            continue;
        }
        // Parse a quoted string starting at i
        i += 1; // skip opening "
        let mut value = String::new();
        loop {
            if i >= bytes.len() {
                break;
            }
            if bytes[i] == b'"' {
                i += 1;
                break;
            }
            if bytes[i] == b'\\' {
                i += 1;
                if i >= bytes.len() {
                    break;
                }
                match bytes[i] {
                    b'"' => {
                        value.push('"');
                        i += 1;
                    }
                    b'\\' => {
                        value.push('\\');
                        i += 1;
                    }
                    b'n' => {
                        value.push('\n');
                        i += 1;
                    }
                    b'r' => {
                        value.push('\r');
                        i += 1;
                    }
                    b't' => {
                        value.push('\t');
                        i += 1;
                    }
                    c => {
                        value.push('\\');
                        value.push(c as char);
                        i += 1;
                    }
                }
            } else {
                value.push(bytes[i] as char);
                i += 1;
            }
        }
        strings.push(value);
        if strings.len() > n {
            break;
        }
    }

    strings.into_iter().nth(n)
}

/// Extract a JSON array of strings from a JSON object.
fn extract_string_array(json: &str, key: &str) -> Option<Vec<String>> {
    let content = extract_array_content(json, key)?;
    if content.trim().is_empty() {
        return Some(Vec::new());
    }

    let mut items = Vec::new();
    let mut remaining = content;
    loop {
        remaining = remaining.trim_start();
        if remaining.is_empty() {
            break;
        }
        if !remaining.starts_with('"') {
            break;
        }
        remaining = &remaining[1..]; // skip opening "
        let mut value = String::new();
        let bytes = remaining.as_bytes();
        let mut j = 0usize;
        loop {
            if j >= bytes.len() {
                break;
            }
            if bytes[j] == b'"' {
                j += 1;
                break;
            }
            if bytes[j] == b'\\' {
                j += 1;
                if j >= bytes.len() {
                    break;
                }
                match bytes[j] {
                    b'"' => {
                        value.push('"');
                    }
                    b'\\' => {
                        value.push('\\');
                    }
                    b'n' => {
                        value.push('\n');
                    }
                    b'r' => {
                        value.push('\r');
                    }
                    b't' => {
                        value.push('\t');
                    }
                    c => {
                        value.push('\\');
                        value.push(c as char);
                    }
                }
                j += 1;
            } else {
                value.push(bytes[j] as char);
                j += 1;
            }
        }
        remaining = &remaining[j..];
        items.push(value);
        remaining = remaining.trim_start_matches(',').trim_start();
    }
    Some(items)
}

/// Extract the `metadata` object from the bundle JSON.
fn extract_metadata_map(json: &str) -> Option<HashMap<String, String>> {
    let needle = "\"metadata\":{";
    let start = json.find(needle)? + needle.len();
    let rest = &json[start..];
    let end = rest.find('}')?;
    let content = &rest[..end];

    let mut map = HashMap::new();
    if content.trim().is_empty() {
        return Some(map);
    }

    // Parse "key":"value" pairs.
    let mut remaining = content;
    loop {
        remaining = remaining.trim_start();
        if remaining.is_empty() {
            break;
        }
        if !remaining.starts_with('"') {
            break;
        }

        let k = extract_next_string(&mut remaining)?;
        remaining = remaining.trim_start();
        if !remaining.starts_with(':') {
            break;
        }
        remaining = &remaining[1..];
        remaining = remaining.trim_start();
        if !remaining.starts_with('"') {
            break;
        }
        let v = extract_next_string(&mut remaining)?;
        map.insert(k, v);
        remaining = remaining.trim_start_matches(',').trim_start();
    }
    Some(map)
}

/// Extract and consume the next quoted JSON string from `s`, advancing `s`
/// past the closing quote.
fn extract_next_string(s: &mut &str) -> Option<String> {
    let src = *s;
    if !src.starts_with('"') {
        return None;
    }
    let bytes = src.as_bytes();
    let mut value = String::new();
    let mut i = 1usize; // skip opening "
    loop {
        if i >= bytes.len() {
            return None;
        }
        if bytes[i] == b'"' {
            i += 1;
            break;
        }
        if bytes[i] == b'\\' {
            i += 1;
            if i >= bytes.len() {
                return None;
            }
            match bytes[i] {
                b'"' => {
                    value.push('"');
                }
                b'\\' => {
                    value.push('\\');
                }
                b'n' => {
                    value.push('\n');
                }
                b'r' => {
                    value.push('\r');
                }
                b't' => {
                    value.push('\t');
                }
                c => {
                    value.push('\\');
                    value.push(c as char);
                }
            }
            i += 1;
        } else {
            value.push(bytes[i] as char);
            i += 1;
        }
    }
    *s = &src[i..];
    Some(value)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ingest_bundle() -> WorkflowBundle {
        let mut meta = HashMap::new();
        meta.insert("author".to_string(), "team-media".to_string());
        meta.insert("env".to_string(), "prod".to_string());

        WorkflowBundle {
            name: "ingest-pipeline".to_string(),
            version: "2.0.0".to_string(),
            steps: vec![
                BundleStep {
                    id: "ingest".to_string(),
                    action: "transcode".to_string(),
                    params: vec![
                        ("input".to_string(), "/raw.mp4".to_string()),
                        ("preset".to_string(), "broadcast".to_string()),
                    ],
                    depends_on: vec![],
                },
                BundleStep {
                    id: "qc".to_string(),
                    action: "quality-check".to_string(),
                    params: vec![("level".to_string(), "strict".to_string())],
                    depends_on: vec!["ingest".to_string()],
                },
                BundleStep {
                    id: "upload".to_string(),
                    action: "transfer".to_string(),
                    params: vec![("dest".to_string(), "s3://bucket/out/".to_string())],
                    depends_on: vec!["ingest".to_string(), "qc".to_string()],
                },
            ],
            metadata: meta,
        }
    }

    // ------------------------------------------------------------------
    // JSON round-trip
    // ------------------------------------------------------------------

    #[test]
    fn to_json_and_back() {
        let original = ingest_bundle();
        let json = original.to_json();
        let parsed = WorkflowBundle::from_json(&json).expect("parse should succeed");
        assert_eq!(parsed.name, original.name);
        assert_eq!(parsed.version, original.version);
        assert_eq!(parsed.steps.len(), original.steps.len());
        assert_eq!(parsed.steps[0].id, "ingest");
        assert_eq!(parsed.steps[1].depends_on, vec!["ingest"]);
        assert_eq!(parsed.steps[2].depends_on, vec!["ingest", "qc"]);
    }

    #[test]
    fn round_trip_preserves_metadata() {
        let bundle = ingest_bundle();
        let json = bundle.to_json();
        let parsed = WorkflowBundle::from_json(&json).expect("parse");
        assert_eq!(
            parsed.metadata.get("author").map(|s| s.as_str()),
            Some("team-media")
        );
        assert_eq!(parsed.metadata.get("env").map(|s| s.as_str()), Some("prod"));
    }

    #[test]
    fn round_trip_preserves_params() {
        let bundle = ingest_bundle();
        let json = bundle.to_json();
        let parsed = WorkflowBundle::from_json(&json).expect("parse");
        let first_step = &parsed.steps[0];
        assert_eq!(
            first_step.params,
            vec![
                ("input".to_string(), "/raw.mp4".to_string()),
                ("preset".to_string(), "broadcast".to_string()),
            ]
        );
    }

    #[test]
    fn empty_bundle_round_trip() {
        let bundle = WorkflowBundle {
            name: "empty".to_string(),
            version: "0.1.0".to_string(),
            steps: vec![],
            metadata: HashMap::new(),
        };
        let json = bundle.to_json();
        let parsed = WorkflowBundle::from_json(&json).expect("parse");
        assert_eq!(parsed.name, "empty");
        assert!(parsed.steps.is_empty());
    }

    #[test]
    fn from_json_missing_name_returns_error() {
        let bad = r#"{"version":"1.0","steps":[],"metadata":{}}"#;
        assert!(WorkflowBundle::from_json(bad).is_err());
    }

    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------

    #[test]
    fn validate_ok_for_valid_bundle() {
        let bundle = ingest_bundle();
        assert!(bundle.validate().is_ok());
    }

    #[test]
    fn validate_detects_duplicate_ids() {
        let mut bundle = ingest_bundle();
        bundle.steps.push(BundleStep {
            id: "ingest".to_string(), // duplicate
            action: "dup-action".to_string(),
            params: vec![],
            depends_on: vec![],
        });
        let result = bundle.validate();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("duplicate")));
    }

    #[test]
    fn validate_detects_missing_dependency() {
        let mut bundle = ingest_bundle();
        bundle.steps[2].depends_on.push("ghost-step".to_string());
        let result = bundle.validate();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("ghost-step")));
    }

    #[test]
    fn validate_reports_all_errors() {
        let bundle = WorkflowBundle {
            name: "bad".to_string(),
            version: "1.0".to_string(),
            steps: vec![
                BundleStep {
                    id: "dup".to_string(),
                    action: "a".to_string(),
                    params: vec![],
                    depends_on: vec![],
                },
                BundleStep {
                    id: "dup".to_string(),
                    action: "b".to_string(),
                    params: vec![],
                    depends_on: vec!["missing".to_string()],
                },
            ],
            metadata: HashMap::new(),
        };
        let result = bundle.validate();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        // At minimum one duplicate error and one missing-dep error
        assert!(errors.len() >= 2);
    }
}
