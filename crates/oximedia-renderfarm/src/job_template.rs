// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Parameterized job templates for encoding presets.
//!
//! A [`JobTemplate`] captures a reusable render configuration whose parameters
//! may be fixed, required (caller must supply), or optional (caller may override
//! a sensible default).  Calling [`JobTemplate::instantiate`] validates supplied
//! values and produces an [`InstantiatedJob`] ready for submission to the farm.
//!
//! A [`TemplateLibrary`] ships several built-in presets and supports registration
//! of custom templates.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TemplateParam
// ---------------------------------------------------------------------------

/// Describes a single parameter in a [`JobTemplate`].
#[derive(Debug, Clone, PartialEq)]
pub enum TemplateParam {
    /// The value is baked into the template and cannot be changed by the caller.
    Fixed(String),
    /// The caller *must* supply a value at instantiation time.
    Required { description: String },
    /// The caller *may* supply a value; the given default is used otherwise.
    Optional {
        default: String,
        description: String,
    },
}

impl TemplateParam {
    /// Returns the default value, if one is available.
    ///
    /// Returns `Some` for [`TemplateParam::Fixed`] and [`TemplateParam::Optional`],
    /// `None` for [`TemplateParam::Required`].
    #[must_use]
    pub fn default_value(&self) -> Option<&str> {
        match self {
            Self::Fixed(v) => Some(v.as_str()),
            Self::Optional { default, .. } => Some(default.as_str()),
            Self::Required { .. } => None,
        }
    }

    /// Returns `true` if the caller must supply a value at instantiation time.
    #[must_use]
    pub fn is_required(&self) -> bool {
        matches!(self, Self::Required { .. })
    }

    /// Human-readable description of the parameter.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::Fixed(v) => v.as_str(),
            Self::Required { description } => description.as_str(),
            Self::Optional { description, .. } => description.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// JobTemplate
// ---------------------------------------------------------------------------

/// Reusable render job configuration with typed parameter declarations.
#[derive(Debug, Clone)]
pub struct JobTemplate {
    /// Unique identifier for this template (e.g. `"youtube-1080p"`).
    pub id: String,
    /// Display name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Broad category: `"transcode"`, `"analyze"`, `"thumbnail"`, `"archive"`.
    pub job_type: String,
    /// Parameter declarations keyed by parameter name.
    pub parameters: HashMap<String, TemplateParam>,
    /// Optional estimate of how long a job of this type typically takes.
    pub estimated_duration_secs: Option<u64>,
}

impl JobTemplate {
    /// Create a new, empty template.  Use the builder methods to populate it.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        job_type: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            job_type: job_type.into(),
            parameters: HashMap::new(),
            estimated_duration_secs: None,
        }
    }

    /// Add a parameter declaration (builder style).
    #[must_use]
    pub fn with_param(mut self, key: impl Into<String>, param: TemplateParam) -> Self {
        self.parameters.insert(key.into(), param);
        self
    }

    /// Set the human-readable description (builder style).
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set the estimated run-time in seconds (builder style).
    #[must_use]
    pub fn with_estimated_duration(mut self, secs: u64) -> Self {
        self.estimated_duration_secs = Some(secs);
        self
    }

    /// Instantiate the template using caller-supplied `values`.
    ///
    /// All `Required` parameters must be present in `values`; `Optional`
    /// parameters use their default when absent; `Fixed` parameters always
    /// use their baked-in value regardless of `values`.
    ///
    /// Returns `Err` with a message listing every missing required parameter.
    pub fn instantiate(&self, values: &HashMap<String, String>) -> Result<InstantiatedJob, String> {
        let mut resolved: HashMap<String, String> = HashMap::new();
        let mut missing: Vec<String> = Vec::new();

        for (key, param) in &self.parameters {
            match param {
                TemplateParam::Fixed(v) => {
                    resolved.insert(key.clone(), v.clone());
                }
                TemplateParam::Required { .. } => match values.get(key.as_str()) {
                    Some(v) => {
                        resolved.insert(key.clone(), v.clone());
                    }
                    None => missing.push(key.clone()),
                },
                TemplateParam::Optional { default, .. } => {
                    let v = values
                        .get(key.as_str())
                        .map(|s| s.as_str())
                        .unwrap_or(default.as_str());
                    resolved.insert(key.clone(), v.to_owned());
                }
            }
        }

        if !missing.is_empty() {
            missing.sort();
            return Err(format!(
                "missing required parameters for template '{}': {}",
                self.id,
                missing.join(", ")
            ));
        }

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Ok(InstantiatedJob {
            template_id: self.id.clone(),
            job_type: self.job_type.clone(),
            parameters: resolved,
            created_at_ms: now_ms,
        })
    }

    /// Keys of all `Required` parameters, sorted for determinism.
    #[must_use]
    pub fn required_params(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self
            .parameters
            .iter()
            .filter(|(_, p)| p.is_required())
            .map(|(k, _)| k.as_str())
            .collect();
        keys.sort_unstable();
        keys
    }

    /// `(key, default_value)` pairs for all `Optional` parameters, sorted.
    #[must_use]
    pub fn optional_params(&self) -> Vec<(&str, &str)> {
        let mut pairs: Vec<(&str, &str)> = self
            .parameters
            .iter()
            .filter_map(|(k, p)| match p {
                TemplateParam::Optional { default, .. } => Some((k.as_str(), default.as_str())),
                _ => None,
            })
            .collect();
        pairs.sort_unstable_by_key(|(k, _)| *k);
        pairs
    }
}

// ---------------------------------------------------------------------------
// InstantiatedJob
// ---------------------------------------------------------------------------

/// A fully resolved snapshot of a [`JobTemplate`] with concrete parameter values.
#[derive(Debug, Clone)]
pub struct InstantiatedJob {
    /// ID of the template this job was created from.
    pub template_id: String,
    /// Job type inherited from the template.
    pub job_type: String,
    /// Resolved parameters (fixed + caller-supplied + defaults).
    pub parameters: HashMap<String, String>,
    /// Wall-clock time of instantiation (milliseconds since UNIX epoch).
    pub created_at_ms: u64,
}

impl InstantiatedJob {
    /// Look up a resolved parameter value by key.
    #[must_use]
    pub fn get_param(&self, key: &str) -> Option<&str> {
        self.parameters.get(key).map(|s| s.as_str())
    }

    /// Serialise to a hand-written JSON string (no `serde` dependency required).
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut params_entries: Vec<String> = self
            .parameters
            .iter()
            .map(|(k, v)| format!("\"{}\":\"{}\"", escape_json(k), escape_json(v)))
            .collect();
        params_entries.sort(); // deterministic output

        format!(
            "{{\"template_id\":\"{}\",\"job_type\":\"{}\",\"created_at_ms\":{},\"parameters\":{{{}}}}}",
            escape_json(&self.template_id),
            escape_json(&self.job_type),
            self.created_at_ms,
            params_entries.join(",")
        )
    }
}

/// Minimal JSON string escaping (backslash and double-quote only).
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ---------------------------------------------------------------------------
// TemplateLibrary
// ---------------------------------------------------------------------------

/// A registry of named [`JobTemplate`]s with built-in presets.
pub struct TemplateLibrary {
    templates: HashMap<String, JobTemplate>,
}

impl TemplateLibrary {
    /// Create an empty library.
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    /// Create a library pre-loaded with the four standard presets:
    /// `youtube-1080p`, `archive-ffv1`, `thumbnail-sprite`, `analyze-quality`.
    pub fn with_defaults() -> Self {
        Self::new()
            .register(youtube_1080p_template())
            .register(archive_ffv1_template())
            .register(thumbnail_sprite_template())
            .register(analyze_quality_template())
    }

    /// Register a template.  Overwrites any template with the same id.
    #[must_use]
    pub fn register(mut self, template: JobTemplate) -> Self {
        self.templates.insert(template.id.clone(), template);
        self
    }

    /// Look up a template by id.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&JobTemplate> {
        self.templates.get(id)
    }

    /// Sorted list of registered template IDs.
    #[must_use]
    pub fn list_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.templates.keys().map(|k| k.as_str()).collect();
        ids.sort_unstable();
        ids
    }

    /// Number of registered templates.
    #[must_use]
    pub fn count(&self) -> usize {
        self.templates.len()
    }
}

impl Default for TemplateLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Built-in presets
// ---------------------------------------------------------------------------

fn youtube_1080p_template() -> JobTemplate {
    JobTemplate::new("youtube-1080p", "YouTube 1080p H.264", "transcode")
        .with_description("YouTube-optimised 1080p encode (H.264/AAC, CBR, fast-start)")
        .with_param("codec", TemplateParam::Fixed("h264".to_owned()))
        .with_param("audio_codec", TemplateParam::Fixed("aac".to_owned()))
        .with_param(
            "input_file",
            TemplateParam::Required {
                description: "Path to the source media file".to_owned(),
            },
        )
        .with_param(
            "output_file",
            TemplateParam::Required {
                description: "Destination path for the encoded file".to_owned(),
            },
        )
        .with_param(
            "bitrate",
            TemplateParam::Optional {
                default: "8000k".to_owned(),
                description: "Video bitrate (e.g. 8000k)".to_owned(),
            },
        )
        .with_param(
            "audio_bitrate",
            TemplateParam::Optional {
                default: "192k".to_owned(),
                description: "Audio bitrate".to_owned(),
            },
        )
        .with_estimated_duration(600)
}

fn archive_ffv1_template() -> JobTemplate {
    JobTemplate::new("archive-ffv1", "Lossless FFV1 Archive", "archive")
        .with_description("Lossless FFV1 video with FLAC audio in Matroska container")
        .with_param("codec", TemplateParam::Fixed("ffv1".to_owned()))
        .with_param("audio_codec", TemplateParam::Fixed("flac".to_owned()))
        .with_param("container", TemplateParam::Fixed("mkv".to_owned()))
        .with_param(
            "input_file",
            TemplateParam::Required {
                description: "Source file path".to_owned(),
            },
        )
        .with_param(
            "output_file",
            TemplateParam::Required {
                description: "Archive output path".to_owned(),
            },
        )
        .with_param(
            "level",
            TemplateParam::Optional {
                default: "3".to_owned(),
                description: "FFV1 encoding level (1, 3)".to_owned(),
            },
        )
        .with_estimated_duration(1200)
}

fn thumbnail_sprite_template() -> JobTemplate {
    JobTemplate::new("thumbnail-sprite", "Thumbnail Sprite Sheet", "thumbnail")
        .with_description("Generate a sprite sheet of thumbnails at regular intervals")
        .with_param(
            "input_file",
            TemplateParam::Required {
                description: "Source video file".to_owned(),
            },
        )
        .with_param(
            "output_dir",
            TemplateParam::Required {
                description: "Output directory for sprites".to_owned(),
            },
        )
        .with_param(
            "interval_secs",
            TemplateParam::Optional {
                default: "10".to_owned(),
                description: "Seconds between thumbnail captures".to_owned(),
            },
        )
        .with_param(
            "width",
            TemplateParam::Optional {
                default: "160".to_owned(),
                description: "Thumbnail width in pixels".to_owned(),
            },
        )
        .with_param(
            "height",
            TemplateParam::Optional {
                default: "90".to_owned(),
                description: "Thumbnail height in pixels".to_owned(),
            },
        )
        .with_estimated_duration(120)
}

fn analyze_quality_template() -> JobTemplate {
    JobTemplate::new("analyze-quality", "Quality Analysis", "analyze")
        .with_description("VMAF + PSNR + SSIM quality analysis against a reference")
        .with_param(
            "input_file",
            TemplateParam::Required {
                description: "Distorted (encoded) file to analyse".to_owned(),
            },
        )
        .with_param(
            "reference_file",
            TemplateParam::Required {
                description: "Original reference file".to_owned(),
            },
        )
        .with_param(
            "metrics",
            TemplateParam::Optional {
                default: "vmaf,psnr,ssim".to_owned(),
                description: "Comma-separated list of metrics to compute".to_owned(),
            },
        )
        .with_param(
            "output_json",
            TemplateParam::Optional {
                default: "quality_report.json".to_owned(),
                description: "Path for the JSON quality report".to_owned(),
            },
        )
        .with_estimated_duration(300)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_transcode_template() -> JobTemplate {
        JobTemplate::new("test-tmpl", "Test Template", "transcode")
            .with_param("codec", TemplateParam::Fixed("h265".to_owned()))
            .with_param(
                "input",
                TemplateParam::Required {
                    description: "Input file".to_owned(),
                },
            )
            .with_param(
                "output",
                TemplateParam::Required {
                    description: "Output file".to_owned(),
                },
            )
            .with_param(
                "crf",
                TemplateParam::Optional {
                    default: "23".to_owned(),
                    description: "Quality factor".to_owned(),
                },
            )
    }

    #[test]
    fn instantiate_with_all_params() {
        let tmpl = make_transcode_template();
        let mut values = HashMap::new();
        values.insert("input".to_owned(), "in.mov".to_owned());
        values.insert("output".to_owned(), "out.mp4".to_owned());
        values.insert("crf".to_owned(), "18".to_owned());

        let job = tmpl.instantiate(&values).expect("should succeed");
        assert_eq!(job.get_param("codec"), Some("h265"));
        assert_eq!(job.get_param("input"), Some("in.mov"));
        assert_eq!(job.get_param("crf"), Some("18"));
        assert_eq!(job.template_id, "test-tmpl");
        assert_eq!(job.job_type, "transcode");
    }

    #[test]
    fn missing_required_param_returns_err() {
        let tmpl = make_transcode_template();
        let values: HashMap<String, String> = HashMap::new();
        let err = tmpl.instantiate(&values).expect_err("should fail");
        assert!(err.contains("input"), "error should mention 'input': {err}");
        assert!(
            err.contains("output"),
            "error should mention 'output': {err}"
        );
    }

    #[test]
    fn optional_uses_default_when_not_supplied() {
        let tmpl = make_transcode_template();
        let mut values = HashMap::new();
        values.insert("input".to_owned(), "a.mov".to_owned());
        values.insert("output".to_owned(), "b.mp4".to_owned());

        let job = tmpl.instantiate(&values).expect("should succeed");
        assert_eq!(job.get_param("crf"), Some("23"), "should use default crf");
    }

    #[test]
    fn fixed_param_cannot_be_overridden() {
        let tmpl = make_transcode_template();
        let mut values = HashMap::new();
        values.insert("input".to_owned(), "a.mov".to_owned());
        values.insert("output".to_owned(), "b.mp4".to_owned());
        values.insert("codec".to_owned(), "av1".to_owned()); // attempt override

        let job = tmpl.instantiate(&values).expect("should succeed");
        assert_eq!(job.get_param("codec"), Some("h265"), "fixed must win");
    }

    #[test]
    fn required_params_returns_sorted_keys() {
        let tmpl = make_transcode_template();
        let keys = tmpl.required_params();
        assert_eq!(keys, vec!["input", "output"]);
    }

    #[test]
    fn optional_params_returns_sorted_key_default_pairs() {
        let tmpl = make_transcode_template();
        let pairs = tmpl.optional_params();
        assert_eq!(pairs, vec![("crf", "23")]);
    }

    #[test]
    fn template_library_with_defaults_has_four_templates() {
        let lib = TemplateLibrary::with_defaults();
        assert_eq!(lib.count(), 4);
        let ids = lib.list_ids();
        assert!(ids.contains(&"youtube-1080p"));
        assert!(ids.contains(&"archive-ffv1"));
        assert!(ids.contains(&"thumbnail-sprite"));
        assert!(ids.contains(&"analyze-quality"));
    }

    #[test]
    fn template_library_register_and_get() {
        let lib = TemplateLibrary::new().register(JobTemplate::new(
            "my-tmpl",
            "My Template",
            "transcode",
        ));
        assert_eq!(lib.count(), 1);
        assert!(lib.get("my-tmpl").is_some());
        assert!(lib.get("missing").is_none());
    }

    #[test]
    fn to_json_contains_template_id() {
        let tmpl = make_transcode_template();
        let mut values = HashMap::new();
        values.insert("input".to_owned(), "a.mov".to_owned());
        values.insert("output".to_owned(), "b.mp4".to_owned());
        let job = tmpl.instantiate(&values).expect("should succeed");
        let json = job.to_json();
        assert!(json.contains("test-tmpl"), "JSON must contain template_id");
        assert!(json.contains("transcode"), "JSON must contain job_type");
    }

    #[test]
    fn to_json_escapes_special_chars() {
        let tmpl = JobTemplate::new("q\"t", "Quote Test", "transcode").with_param(
            "input",
            TemplateParam::Required {
                description: "path".to_owned(),
            },
        );
        let mut values = HashMap::new();
        values.insert("input".to_owned(), "path/with\\backslash".to_owned());
        let job = tmpl.instantiate(&values).expect("should succeed");
        let json = job.to_json();
        // backslash in value should be escaped
        assert!(
            json.contains("path/with\\\\backslash"),
            "backslash escaped: {json}"
        );
    }

    #[test]
    fn template_param_is_required() {
        let r = TemplateParam::Required {
            description: "d".to_owned(),
        };
        let o = TemplateParam::Optional {
            default: "x".to_owned(),
            description: "d".to_owned(),
        };
        let f = TemplateParam::Fixed("v".to_owned());
        assert!(r.is_required());
        assert!(!o.is_required());
        assert!(!f.is_required());
    }

    #[test]
    fn template_param_default_value() {
        let r = TemplateParam::Required {
            description: "d".to_owned(),
        };
        let o = TemplateParam::Optional {
            default: "x".to_owned(),
            description: "d".to_owned(),
        };
        let f = TemplateParam::Fixed("v".to_owned());
        assert_eq!(r.default_value(), None);
        assert_eq!(o.default_value(), Some("x"));
        assert_eq!(f.default_value(), Some("v"));
    }
}
