//! Pre-built workflow templates for common media processing pipelines.
//!
//! Provides [`WorkflowTemplate`] with factory constructors for each
//! [`WorkflowTemplateKind`], parameter substitution, DOT graph generation,
//! and an `instantiate` method that returns a lightweight [`WorkflowInstance`].

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Template Kind ─────────────────────────────────────────────────────────────

/// Discriminates which pre-built media pipeline template to use.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkflowTemplateKind {
    /// ingest → validate → transcode → deliver
    IngestAndTranscode,
    /// probe → quality-gate → approve/reject
    QualityCheck,
    /// transcode proxy → archive original → catalog
    ArchiveAndProxy,
    /// normalize → transcode → package → deliver
    BroadcastDelivery,
    /// transcode multiple formats → thumbnail → upload
    SocialMediaPackage,
    /// normalize audio → encode MP3+AAC → chapters → RSS
    AudioPodcast,
    /// quick proxy → caption → publish
    NewsroomFast,
    /// clip → transcode → metadata → CDN push
    LiveEventClip,
}

impl WorkflowTemplateKind {
    /// Human-readable display name.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::IngestAndTranscode => "Ingest and Transcode",
            Self::QualityCheck => "Quality Check",
            Self::ArchiveAndProxy => "Archive and Proxy",
            Self::BroadcastDelivery => "Broadcast Delivery",
            Self::SocialMediaPackage => "Social Media Package",
            Self::AudioPodcast => "Audio Podcast",
            Self::NewsroomFast => "Newsroom Fast",
            Self::LiveEventClip => "Live Event Clip",
        }
    }
}

impl std::fmt::Display for WorkflowTemplateKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ── TemplateStep ──────────────────────────────────────────────────────────────

/// A single step definition within a [`WorkflowTemplate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateStep {
    /// Unique identifier for this step within the template.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Step type: "transcode", "normalize", "archive", "deliver", "validate", "notify", etc.
    pub step_type: String,
    /// IDs of steps that must complete before this step starts.
    pub depends_on: Vec<String>,
    /// Step-specific configuration. Values may contain `{param_name}` placeholders.
    pub config: HashMap<String, String>,
    /// Maximum wall-clock time allowed for the step (seconds).
    pub timeout_secs: u64,
    /// Number of automatic retry attempts on failure.
    pub retry_count: u32,
}

impl TemplateStep {
    /// Create a new step with no dependencies and sensible defaults.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        step_type: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            step_type: step_type.into(),
            depends_on: Vec::new(),
            config: HashMap::new(),
            timeout_secs: 3600,
            retry_count: 1,
        }
    }

    /// Set dependencies.
    #[must_use]
    pub fn with_depends_on(mut self, deps: Vec<String>) -> Self {
        self.depends_on = deps;
        self
    }

    /// Insert a config entry.
    #[must_use]
    pub fn with_config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.insert(key.into(), value.into());
        self
    }

    /// Set timeout.
    #[must_use]
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set retry count.
    #[must_use]
    pub fn with_retries(mut self, count: u32) -> Self {
        self.retry_count = count;
        self
    }
}

// ── TemplateParameter ─────────────────────────────────────────────────────────

/// A parameter that callers must or may supply when instantiating a template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateParameter {
    /// Parameter name (used as `{name}` placeholder in step config values).
    pub name: String,
    /// Human-readable description shown to the caller.
    pub description: String,
    /// Optional default value used when the caller does not supply a value.
    pub default_value: Option<String>,
    /// Whether the caller must supply this parameter.
    pub required: bool,
}

impl TemplateParameter {
    /// Create a required parameter with no default.
    #[must_use]
    pub fn required(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            default_value: None,
            required: true,
        }
    }

    /// Create an optional parameter with a default value.
    #[must_use]
    pub fn optional(
        name: impl Into<String>,
        description: impl Into<String>,
        default: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            default_value: Some(default.into()),
            required: false,
        }
    }
}

// ── WorkflowInstance ──────────────────────────────────────────────────────────

/// A concrete workflow produced by instantiating a [`WorkflowTemplate`].
///
/// Steps have had all `{param}` placeholders substituted with resolved values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowInstance {
    /// Name inherited from the template.
    pub name: String,
    /// Kind of the originating template.
    pub kind: WorkflowTemplateKind,
    /// Resolved parameters (including defaults).
    pub resolved_params: HashMap<String, String>,
    /// Instantiated steps with substituted config values.
    pub steps: Vec<TemplateStep>,
}

// ── WorkflowTemplate ─────────────────────────────────────────────────────────

/// A pre-built workflow template for a common media processing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTemplate {
    /// The kind of pipeline this template implements.
    pub kind: WorkflowTemplateKind,
    /// Human-readable name.
    pub name: String,
    /// Longer description of what the pipeline does.
    pub description: String,
    /// Ordered list of steps (order determines DOT rendering; DAG edges come from `depends_on`).
    pub steps: Vec<TemplateStep>,
    /// Parameters that can be tuned at instantiation time.
    pub parameters: Vec<TemplateParameter>,
}

impl WorkflowTemplate {
    // ── Factory ───────────────────────────────────────────────────────────────

    /// Construct a fully-configured template for the given kind.
    #[must_use]
    pub fn new(kind: WorkflowTemplateKind) -> Self {
        match kind {
            WorkflowTemplateKind::IngestAndTranscode => Self::build_ingest_and_transcode(),
            WorkflowTemplateKind::QualityCheck => Self::build_quality_check(),
            WorkflowTemplateKind::ArchiveAndProxy => Self::build_archive_and_proxy(),
            WorkflowTemplateKind::BroadcastDelivery => Self::build_broadcast_delivery(),
            WorkflowTemplateKind::SocialMediaPackage => Self::build_social_media_package(),
            WorkflowTemplateKind::AudioPodcast => Self::build_audio_podcast(),
            WorkflowTemplateKind::NewsroomFast => Self::build_newsroom_fast(),
            WorkflowTemplateKind::LiveEventClip => Self::build_live_event_clip(),
        }
    }

    /// Return one instance of each template kind.
    #[must_use]
    pub fn list_all() -> Vec<WorkflowTemplate> {
        vec![
            Self::new(WorkflowTemplateKind::IngestAndTranscode),
            Self::new(WorkflowTemplateKind::QualityCheck),
            Self::new(WorkflowTemplateKind::ArchiveAndProxy),
            Self::new(WorkflowTemplateKind::BroadcastDelivery),
            Self::new(WorkflowTemplateKind::SocialMediaPackage),
            Self::new(WorkflowTemplateKind::AudioPodcast),
            Self::new(WorkflowTemplateKind::NewsroomFast),
            Self::new(WorkflowTemplateKind::LiveEventClip),
        ]
    }

    // ── Instantiation ─────────────────────────────────────────────────────────

    /// Substitute `{param_name}` placeholders in step config values and return
    /// a [`WorkflowInstance`].
    ///
    /// # Errors
    ///
    /// Returns an error string when a required parameter is not present in
    /// `params` and has no default.
    pub fn instantiate(
        &self,
        params: &HashMap<String, String>,
    ) -> Result<WorkflowInstance, String> {
        // Build resolved parameter map (caller-supplied wins over default).
        let mut resolved: HashMap<String, String> = HashMap::new();
        for p in &self.parameters {
            if let Some(v) = params.get(&p.name) {
                resolved.insert(p.name.clone(), v.clone());
            } else if let Some(d) = &p.default_value {
                resolved.insert(p.name.clone(), d.clone());
            } else if p.required {
                return Err(format!("Required parameter '{}' was not provided", p.name));
            }
        }

        // Deep-clone steps with substituted config.
        let steps = self
            .steps
            .iter()
            .map(|s| {
                let config = s
                    .config
                    .iter()
                    .map(|(k, v)| (k.clone(), substitute(v, &resolved)))
                    .collect();
                TemplateStep {
                    id: s.id.clone(),
                    name: s.name.clone(),
                    step_type: s.step_type.clone(),
                    depends_on: s.depends_on.clone(),
                    config,
                    timeout_secs: s.timeout_secs,
                    retry_count: s.retry_count,
                }
            })
            .collect();

        Ok(WorkflowInstance {
            name: self.name.clone(),
            kind: self.kind.clone(),
            resolved_params: resolved,
            steps,
        })
    }

    // ── DOT graph generation ──────────────────────────────────────────────────

    /// Generate a Graphviz DOT representation of the step dependency graph.
    #[must_use]
    pub fn to_dot(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "digraph \"{}\" {{\n    rankdir=LR;\n    node [shape=box style=filled fillcolor=lightblue];\n",
            escape_dot(&self.name)
        ));

        // Declare nodes.
        for step in &self.steps {
            out.push_str(&format!(
                "    \"{}\" [label=\"{}\\n({})\"];\n",
                escape_dot(&step.id),
                escape_dot(&step.name),
                escape_dot(&step.step_type),
            ));
        }

        // Declare edges.
        for step in &self.steps {
            for dep in &step.depends_on {
                out.push_str(&format!(
                    "    \"{}\" -> \"{}\";\n",
                    escape_dot(dep),
                    escape_dot(&step.id),
                ));
            }
        }

        out.push('}');
        out
    }

    // ── Private builders ──────────────────────────────────────────────────────

    fn build_ingest_and_transcode() -> Self {
        let steps = vec![
            TemplateStep::new("ingest", "Ingest", "ingest")
                .with_config("source", "{source_path}")
                .with_timeout(600)
                .with_retries(2),
            TemplateStep::new("validate", "Validate", "validate")
                .with_depends_on(vec!["ingest".to_string()])
                .with_config("checks", "format,integrity,container")
                .with_timeout(300)
                .with_retries(1),
            TemplateStep::new("transcode", "Transcode", "transcode")
                .with_depends_on(vec!["validate".to_string()])
                .with_config("preset", "{transcode_preset}")
                .with_config("output", "{output_path}")
                .with_timeout(7200)
                .with_retries(1),
            TemplateStep::new("deliver", "Deliver", "deliver")
                .with_depends_on(vec!["transcode".to_string()])
                .with_config("destination", "{delivery_destination}")
                .with_config("verify_checksum", "true")
                .with_timeout(1800)
                .with_retries(3),
        ];
        let parameters = vec![
            TemplateParameter::required("source_path", "Path to the source media file"),
            TemplateParameter::required("output_path", "Path for the transcoded output"),
            TemplateParameter::required("delivery_destination", "Delivery target (path or URL)"),
            TemplateParameter::optional("transcode_preset", "Transcode quality preset", "web_h264"),
        ];
        Self {
            kind: WorkflowTemplateKind::IngestAndTranscode,
            name: "Ingest and Transcode".to_string(),
            description: "Ingest source media, validate integrity, transcode to target format, and deliver to destination.".to_string(),
            steps,
            parameters,
        }
    }

    fn build_quality_check() -> Self {
        let tmp = std::env::temp_dir();
        let steps = vec![
            TemplateStep::new("probe", "Probe Media", "probe")
                .with_config("input", "{media_path}")
                .with_config("output_json", "{probe_output}")
                .with_timeout(120)
                .with_retries(1),
            TemplateStep::new("quality_gate", "Quality Gate", "validate")
                .with_depends_on(vec!["probe".to_string()])
                .with_config("profile", "{qc_profile}")
                .with_config("threshold", "{pass_threshold}")
                .with_timeout(300)
                .with_retries(0),
            TemplateStep::new("decision", "Approve or Reject", "approve_reject")
                .with_depends_on(vec!["quality_gate".to_string()])
                .with_config("auto_approve_on_pass", "true")
                .with_timeout(86400)
                .with_retries(0),
        ];
        let parameters = vec![
            TemplateParameter::required("media_path", "Path to the media file to check"),
            TemplateParameter::optional(
                "probe_output",
                "Path for probe JSON output",
                tmp.join("oximedia-probe.json")
                    .to_string_lossy()
                    .into_owned(),
            ),
            TemplateParameter::optional(
                "qc_profile",
                "QC profile (broadcast, streaming, cinema)",
                "broadcast",
            ),
            TemplateParameter::optional(
                "pass_threshold",
                "Minimum score (0.0–1.0) to pass",
                "0.95",
            ),
        ];
        Self {
            kind: WorkflowTemplateKind::QualityCheck,
            name: "Quality Check".to_string(),
            description:
                "Probe media metadata, evaluate against a quality gate, then approve or reject."
                    .to_string(),
            steps,
            parameters,
        }
    }

    fn build_archive_and_proxy() -> Self {
        let tmp = std::env::temp_dir();
        let steps = vec![
            TemplateStep::new("proxy", "Transcode Proxy", "transcode")
                .with_config("input", "{source_path}")
                .with_config("preset", "proxy_h264")
                .with_config("output", "{proxy_output}")
                .with_timeout(3600)
                .with_retries(1),
            TemplateStep::new("archive", "Archive Original", "archive")
                .with_depends_on(vec!["proxy".to_string()])
                .with_config("source", "{source_path}")
                .with_config("destination", "{archive_path}")
                .with_config("checksum", "sha256")
                .with_timeout(7200)
                .with_retries(2),
            TemplateStep::new("catalog", "Catalog Entry", "catalog")
                .with_depends_on(vec!["archive".to_string()])
                .with_config("proxy_path", "{proxy_output}")
                .with_config("archive_path", "{archive_path}")
                .with_config("metadata_tags", "{metadata_tags}")
                .with_timeout(300)
                .with_retries(1),
        ];
        let parameters = vec![
            TemplateParameter::required("source_path", "Path to the original media file"),
            TemplateParameter::required("archive_path", "Long-term archive destination path"),
            TemplateParameter::optional(
                "proxy_output",
                "Path for the proxy file",
                tmp.join("oximedia-proxy.mp4")
                    .to_string_lossy()
                    .into_owned(),
            ),
            TemplateParameter::optional("metadata_tags", "Comma-separated metadata tags", ""),
        ];
        Self {
            kind: WorkflowTemplateKind::ArchiveAndProxy,
            name: "Archive and Proxy".to_string(),
            description: "Generate a proxy file, archive the original to long-term storage, and catalog both.".to_string(),
            steps,
            parameters,
        }
    }

    fn build_broadcast_delivery() -> Self {
        let tmp = std::env::temp_dir();
        let steps = vec![
            TemplateStep::new("normalize", "Normalize Audio/Video", "normalize")
                .with_config("input", "{source_path}")
                .with_config("loudness_target", "{loudness_lufs}")
                .with_timeout(1800)
                .with_retries(1),
            TemplateStep::new("transcode", "Transcode", "transcode")
                .with_depends_on(vec!["normalize".to_string()])
                .with_config("preset", "{broadcast_preset}")
                .with_config("output", "{work_path}")
                .with_timeout(7200)
                .with_retries(1),
            TemplateStep::new("package", "Package", "package")
                .with_depends_on(vec!["transcode".to_string()])
                .with_config("format", "{package_format}")
                .with_config("output", "{package_output}")
                .with_timeout(1800)
                .with_retries(1),
            TemplateStep::new("deliver", "Deliver", "deliver")
                .with_depends_on(vec!["package".to_string()])
                .with_config("destination", "{broadcast_destination}")
                .with_config("protocol", "ftp")
                .with_timeout(3600)
                .with_retries(3),
        ];
        let parameters = vec![
            TemplateParameter::required("source_path", "Source media path"),
            TemplateParameter::required("broadcast_destination", "FTP/delivery endpoint URL"),
            TemplateParameter::optional("loudness_lufs", "Target loudness in LUFS", "-23"),
            TemplateParameter::optional(
                "broadcast_preset",
                "Broadcast transcode preset",
                "broadcast_hd",
            ),
            TemplateParameter::optional("package_format", "Packaging format (mxf, ts, mp4)", "mxf"),
            TemplateParameter::optional(
                "package_output",
                "Packaged output path",
                tmp.join("oximedia-broadcast.mxf")
                    .to_string_lossy()
                    .into_owned(),
            ),
            TemplateParameter::optional(
                "work_path",
                "Intermediate transcode output path",
                tmp.join("oximedia-broadcast-work.mxf")
                    .to_string_lossy()
                    .into_owned(),
            ),
        ];
        Self {
            kind: WorkflowTemplateKind::BroadcastDelivery,
            name: "Broadcast Delivery".to_string(),
            description: "Normalize levels, transcode to broadcast spec, package, and deliver."
                .to_string(),
            steps,
            parameters,
        }
    }

    fn build_social_media_package() -> Self {
        let tmp = std::env::temp_dir();
        let steps = vec![
            TemplateStep::new("transcode_16_9", "Transcode 16:9", "transcode")
                .with_config("input", "{source_path}")
                .with_config("aspect", "16:9")
                .with_config("output", "{output_16_9}")
                .with_timeout(3600)
                .with_retries(1),
            TemplateStep::new("transcode_1_1", "Transcode 1:1 (Square)", "transcode")
                .with_config("input", "{source_path}")
                .with_config("aspect", "1:1")
                .with_config("output", "{output_1_1}")
                .with_timeout(3600)
                .with_retries(1),
            TemplateStep::new("transcode_9_16", "Transcode 9:16 (Vertical)", "transcode")
                .with_config("input", "{source_path}")
                .with_config("aspect", "9:16")
                .with_config("output", "{output_9_16}")
                .with_timeout(3600)
                .with_retries(1),
            TemplateStep::new("thumbnail", "Generate Thumbnail", "thumbnail")
                .with_depends_on(vec!["transcode_16_9".to_string()])
                .with_config("input", "{output_16_9}")
                .with_config("output", "{thumbnail_path}")
                .with_timeout(120)
                .with_retries(1),
            TemplateStep::new("upload", "Upload to Platform", "upload")
                .with_depends_on(vec![
                    "transcode_16_9".to_string(),
                    "transcode_1_1".to_string(),
                    "transcode_9_16".to_string(),
                    "thumbnail".to_string(),
                ])
                .with_config("platform", "{platform}")
                .with_config("api_key", "{platform_api_key}")
                .with_timeout(3600)
                .with_retries(2),
        ];
        let parameters = vec![
            TemplateParameter::required("source_path", "Source video path"),
            TemplateParameter::required("platform", "Social platform (youtube, instagram, tiktok)"),
            TemplateParameter::required("platform_api_key", "Platform API key for upload"),
            TemplateParameter::optional(
                "output_16_9",
                "16:9 output path",
                tmp.join("oximedia-social-16-9.mp4")
                    .to_string_lossy()
                    .into_owned(),
            ),
            TemplateParameter::optional(
                "output_1_1",
                "1:1 output path",
                tmp.join("oximedia-social-1-1.mp4")
                    .to_string_lossy()
                    .into_owned(),
            ),
            TemplateParameter::optional(
                "output_9_16",
                "9:16 output path",
                tmp.join("oximedia-social-9-16.mp4")
                    .to_string_lossy()
                    .into_owned(),
            ),
            TemplateParameter::optional(
                "thumbnail_path",
                "Thumbnail image path",
                tmp.join("oximedia-thumbnail.jpg")
                    .to_string_lossy()
                    .into_owned(),
            ),
        ];
        Self {
            kind: WorkflowTemplateKind::SocialMediaPackage,
            name: "Social Media Package".to_string(),
            description:
                "Transcode to 16:9, 1:1, and 9:16 aspect ratios, generate a thumbnail, then upload."
                    .to_string(),
            steps,
            parameters,
        }
    }

    fn build_audio_podcast() -> Self {
        let tmp = std::env::temp_dir();
        let steps = vec![
            TemplateStep::new("normalize_audio", "Normalize Audio", "normalize")
                .with_config("input", "{source_audio}")
                .with_config("loudness_target", "{loudness_lufs}")
                .with_timeout(600)
                .with_retries(1),
            TemplateStep::new("encode_mp3", "Encode MP3", "transcode")
                .with_depends_on(vec!["normalize_audio".to_string()])
                .with_config("codec", "mp3")
                .with_config("bitrate", "{mp3_bitrate}")
                .with_config("output", "{mp3_output}")
                .with_timeout(600)
                .with_retries(1),
            TemplateStep::new("encode_aac", "Encode AAC", "transcode")
                .with_depends_on(vec!["normalize_audio".to_string()])
                .with_config("codec", "aac")
                .with_config("bitrate", "{aac_bitrate}")
                .with_config("output", "{aac_output}")
                .with_timeout(600)
                .with_retries(1),
            TemplateStep::new("chapters", "Embed Chapters", "metadata")
                .with_depends_on(vec!["encode_mp3".to_string(), "encode_aac".to_string()])
                .with_config("chapter_file", "{chapter_file}")
                .with_timeout(120)
                .with_retries(1),
            TemplateStep::new("rss", "Publish RSS", "notify")
                .with_depends_on(vec!["chapters".to_string()])
                .with_config("feed_url", "{rss_feed_url}")
                .with_config("title", "{episode_title}")
                .with_timeout(300)
                .with_retries(2),
        ];
        let parameters = vec![
            TemplateParameter::required("source_audio", "Source audio file path"),
            TemplateParameter::required("rss_feed_url", "RSS feed endpoint URL"),
            TemplateParameter::required("episode_title", "Podcast episode title"),
            TemplateParameter::optional("loudness_lufs", "Target loudness in LUFS", "-16"),
            TemplateParameter::optional("mp3_bitrate", "MP3 bitrate (kbps)", "128k"),
            TemplateParameter::optional("aac_bitrate", "AAC bitrate (kbps)", "96k"),
            TemplateParameter::optional(
                "mp3_output",
                "MP3 output path",
                tmp.join("oximedia-episode.mp3")
                    .to_string_lossy()
                    .into_owned(),
            ),
            TemplateParameter::optional(
                "aac_output",
                "AAC output path",
                tmp.join("oximedia-episode.m4a")
                    .to_string_lossy()
                    .into_owned(),
            ),
            TemplateParameter::optional("chapter_file", "Chapter marker file path", ""),
        ];
        Self {
            kind: WorkflowTemplateKind::AudioPodcast,
            name: "Audio Podcast".to_string(),
            description:
                "Normalize audio, encode to MP3 and AAC, embed chapters, and publish RSS feed."
                    .to_string(),
            steps,
            parameters,
        }
    }

    fn build_newsroom_fast() -> Self {
        let tmp = std::env::temp_dir();
        let steps = vec![
            TemplateStep::new("proxy", "Quick Proxy", "transcode")
                .with_config("input", "{source_path}")
                .with_config("preset", "proxy_fast")
                .with_config("output", "{proxy_output}")
                .with_timeout(300)
                .with_retries(1),
            TemplateStep::new("caption", "Generate Captions", "caption")
                .with_depends_on(vec!["proxy".to_string()])
                .with_config("input", "{proxy_output}")
                .with_config("language", "{caption_language}")
                .with_config("output", "{caption_file}")
                .with_timeout(600)
                .with_retries(1),
            TemplateStep::new("publish", "Publish", "deliver")
                .with_depends_on(vec!["caption".to_string()])
                .with_config("destination", "{publish_destination}")
                .with_config("caption_file", "{caption_file}")
                .with_timeout(300)
                .with_retries(2),
        ];
        let parameters = vec![
            TemplateParameter::required("source_path", "Source media file path"),
            TemplateParameter::required("publish_destination", "Publishing endpoint URL"),
            TemplateParameter::optional(
                "proxy_output",
                "Proxy output path",
                tmp.join("oximedia-proxy-fast.mp4")
                    .to_string_lossy()
                    .into_owned(),
            ),
            TemplateParameter::optional("caption_language", "Caption language code", "en"),
            TemplateParameter::optional(
                "caption_file",
                "Caption file output path",
                tmp.join("oximedia-captions.vtt")
                    .to_string_lossy()
                    .into_owned(),
            ),
        ];
        Self {
            kind: WorkflowTemplateKind::NewsroomFast,
            name: "Newsroom Fast".to_string(),
            description: "Rapid proxy generation, automatic captioning, and immediate publish."
                .to_string(),
            steps,
            parameters,
        }
    }

    fn build_live_event_clip() -> Self {
        let tmp = std::env::temp_dir();
        let steps = vec![
            TemplateStep::new("clip", "Clip Segment", "clip")
                .with_config("input", "{source_path}")
                .with_config("start_tc", "{clip_start}")
                .with_config("end_tc", "{clip_end}")
                .with_config("output", "{clip_output}")
                .with_timeout(600)
                .with_retries(1),
            TemplateStep::new("transcode", "Transcode Clip", "transcode")
                .with_depends_on(vec!["clip".to_string()])
                .with_config("input", "{clip_output}")
                .with_config("preset", "{clip_preset}")
                .with_config("output", "{transcode_output}")
                .with_timeout(1800)
                .with_retries(1),
            TemplateStep::new("metadata", "Embed Metadata", "metadata")
                .with_depends_on(vec!["transcode".to_string()])
                .with_config("input", "{transcode_output}")
                .with_config("title", "{clip_title}")
                .with_config("event", "{event_name}")
                .with_timeout(120)
                .with_retries(1),
            TemplateStep::new("cdn_push", "CDN Push", "deliver")
                .with_depends_on(vec!["metadata".to_string()])
                .with_config("destination", "{cdn_url}")
                .with_config("protocol", "https")
                .with_timeout(1800)
                .with_retries(3),
        ];
        let parameters = vec![
            TemplateParameter::required("source_path", "Live recording or VOD source path"),
            TemplateParameter::required("cdn_url", "CDN ingest URL"),
            TemplateParameter::required("clip_start", "Clip start timecode (HH:MM:SS or frames)"),
            TemplateParameter::required("clip_end", "Clip end timecode (HH:MM:SS or frames)"),
            TemplateParameter::optional(
                "clip_output",
                "Raw clip output path",
                tmp.join("oximedia-raw-clip.mp4")
                    .to_string_lossy()
                    .into_owned(),
            ),
            TemplateParameter::optional(
                "transcode_output",
                "Transcoded clip output path",
                tmp.join("oximedia-clip.mp4").to_string_lossy().into_owned(),
            ),
            TemplateParameter::optional("clip_preset", "Transcode preset for clip", "web_h264"),
            TemplateParameter::optional("clip_title", "Clip title for metadata", ""),
            TemplateParameter::optional("event_name", "Event name for metadata", ""),
        ];
        Self {
            kind: WorkflowTemplateKind::LiveEventClip,
            name: "Live Event Clip".to_string(),
            description: "Extract a timed clip, transcode, embed metadata, and push to CDN."
                .to_string(),
            steps,
            parameters,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Replace `{param_name}` placeholders in `value` using `params`.
fn substitute(value: &str, params: &HashMap<String, String>) -> String {
    let mut out = value.to_string();
    for (k, v) in params {
        out = out.replace(&format!("{{{k}}}"), v);
    }
    out
}

/// Escape a string for use in a DOT label or node name.
fn escape_dot(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn full_params_ingest() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("source_path".into(), "/mnt/src/video.mxf".into());
        m.insert("output_path".into(), "/mnt/out/video.mp4".into());
        m.insert(
            "delivery_destination".into(),
            "ftp://server/delivery/".into(),
        );
        m
    }

    #[test]
    fn test_list_all_returns_eight_templates() {
        let all = WorkflowTemplate::list_all();
        assert_eq!(all.len(), 8);
    }

    #[test]
    fn test_ingest_and_transcode_has_four_steps() {
        let t = WorkflowTemplate::new(WorkflowTemplateKind::IngestAndTranscode);
        assert_eq!(t.steps.len(), 4);
    }

    #[test]
    fn test_quality_check_has_three_steps() {
        let t = WorkflowTemplate::new(WorkflowTemplateKind::QualityCheck);
        assert_eq!(t.steps.len(), 3);
    }

    #[test]
    fn test_archive_and_proxy_has_three_steps() {
        let t = WorkflowTemplate::new(WorkflowTemplateKind::ArchiveAndProxy);
        assert_eq!(t.steps.len(), 3);
    }

    #[test]
    fn test_broadcast_delivery_has_four_steps() {
        let t = WorkflowTemplate::new(WorkflowTemplateKind::BroadcastDelivery);
        assert_eq!(t.steps.len(), 4);
    }

    #[test]
    fn test_social_media_package_has_five_steps() {
        let t = WorkflowTemplate::new(WorkflowTemplateKind::SocialMediaPackage);
        assert_eq!(t.steps.len(), 5);
    }

    #[test]
    fn test_audio_podcast_has_five_steps() {
        let t = WorkflowTemplate::new(WorkflowTemplateKind::AudioPodcast);
        assert_eq!(t.steps.len(), 5);
    }

    #[test]
    fn test_newsroom_fast_has_three_steps() {
        let t = WorkflowTemplate::new(WorkflowTemplateKind::NewsroomFast);
        assert_eq!(t.steps.len(), 3);
    }

    #[test]
    fn test_live_event_clip_has_four_steps() {
        let t = WorkflowTemplate::new(WorkflowTemplateKind::LiveEventClip);
        assert_eq!(t.steps.len(), 4);
    }

    #[test]
    fn test_instantiate_substitutes_placeholders() {
        let t = WorkflowTemplate::new(WorkflowTemplateKind::IngestAndTranscode);
        let instance = t
            .instantiate(&full_params_ingest())
            .expect("instantiate should succeed");
        let ingest_step = instance
            .steps
            .iter()
            .find(|s| s.id == "ingest")
            .expect("ingest step must exist");
        assert_eq!(
            ingest_step.config.get("source").map(String::as_str),
            Some("/mnt/src/video.mxf")
        );
    }

    #[test]
    fn test_instantiate_missing_required_param_returns_err() {
        let t = WorkflowTemplate::new(WorkflowTemplateKind::IngestAndTranscode);
        let result = t.instantiate(&HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_instantiate_optional_defaults_applied() {
        let t = WorkflowTemplate::new(WorkflowTemplateKind::IngestAndTranscode);
        let instance = t
            .instantiate(&full_params_ingest())
            .expect("instantiate should succeed");
        assert_eq!(
            instance
                .resolved_params
                .get("transcode_preset")
                .map(String::as_str),
            Some("web_h264")
        );
    }

    #[test]
    fn test_to_dot_contains_digraph() {
        let t = WorkflowTemplate::new(WorkflowTemplateKind::IngestAndTranscode);
        let dot = t.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("rankdir=LR"));
    }

    #[test]
    fn test_to_dot_contains_step_ids() {
        let t = WorkflowTemplate::new(WorkflowTemplateKind::IngestAndTranscode);
        let dot = t.to_dot();
        assert!(dot.contains("ingest"));
        assert!(dot.contains("validate"));
        assert!(dot.contains("transcode"));
        assert!(dot.contains("deliver"));
    }

    #[test]
    fn test_to_dot_contains_edges() {
        let t = WorkflowTemplate::new(WorkflowTemplateKind::IngestAndTranscode);
        let dot = t.to_dot();
        // Deliver depends on transcode → edge transcode -> deliver
        assert!(dot.contains("->"));
    }

    #[test]
    fn test_workflow_template_kind_display() {
        assert_eq!(
            WorkflowTemplateKind::IngestAndTranscode.to_string(),
            "Ingest and Transcode"
        );
        assert_eq!(
            WorkflowTemplateKind::AudioPodcast.to_string(),
            "Audio Podcast"
        );
    }
}
