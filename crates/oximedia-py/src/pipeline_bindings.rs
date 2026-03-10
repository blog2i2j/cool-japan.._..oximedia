//! Pipeline Python binding data structures.
//!
//! Defines the plain Rust representations of pipeline stage configurations
//! and progress callbacks that bridge Python and the OxiMedia pipeline engine.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// The kind of operation a pipeline stage performs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageKind {
    /// Decode compressed data into raw frames.
    Decode,
    /// Encode raw frames into compressed data.
    Encode,
    /// Apply a filter or transform to frames.
    Filter,
    /// Mux streams into a container.
    Mux,
    /// Demux streams from a container.
    Demux,
    /// Custom user-defined stage.
    Custom(String),
}

impl StageKind {
    /// Return a human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Decode => "decode",
            Self::Encode => "encode",
            Self::Filter => "filter",
            Self::Mux => "mux",
            Self::Demux => "demux",
            Self::Custom(s) => s.as_str(),
        }
    }
}

/// Configuration for a single pipeline stage.
#[derive(Clone, Debug, PartialEq)]
pub struct StageConfig {
    /// Human-readable name for this stage.
    pub name: String,
    /// Kind of operation.
    pub kind: StageKind,
    /// Arbitrary key-value parameters forwarded to the stage.
    pub params: HashMap<String, String>,
    /// Whether this stage is enabled.
    pub enabled: bool,
}

impl StageConfig {
    /// Create a new enabled `StageConfig`.
    #[must_use]
    pub fn new(name: impl Into<String>, kind: StageKind) -> Self {
        Self {
            name: name.into(),
            kind,
            params: HashMap::new(),
            enabled: true,
        }
    }

    /// Add a parameter to the stage.
    #[must_use]
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }

    /// Disable this stage.
    #[must_use]
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Get a parameter value by key.
    #[must_use]
    pub fn get_param(&self, key: &str) -> Option<&str> {
        self.params.get(key).map(String::as_str)
    }
}

/// A sequence of [`StageConfig`]s forming a complete pipeline.
#[derive(Clone, Debug, Default)]
pub struct PipelineConfig {
    /// Ordered list of stages.
    pub stages: Vec<StageConfig>,
    /// Global pipeline metadata (e.g. input path, output path).
    pub metadata: HashMap<String, String>,
}

impl PipelineConfig {
    /// Create an empty `PipelineConfig`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a stage.
    pub fn push_stage(&mut self, stage: StageConfig) {
        self.stages.push(stage);
    }

    /// Number of enabled stages.
    #[must_use]
    pub fn enabled_count(&self) -> usize {
        self.stages.iter().filter(|s| s.enabled).count()
    }

    /// Find the first stage with the given name.
    #[must_use]
    pub fn find_stage(&self, name: &str) -> Option<&StageConfig> {
        self.stages.iter().find(|s| s.name == name)
    }

    /// Set a global metadata value.
    pub fn set_meta(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

/// A snapshot of pipeline progress sent back to Python callbacks.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgressSnapshot {
    /// Number of frames processed so far.
    pub frames_done: u64,
    /// Total frames to process (`None` if unknown).
    pub frames_total: Option<u64>,
    /// Current stage name.
    pub current_stage: String,
    /// Elapsed wall-clock time in seconds.
    pub elapsed_secs: f64,
}

impl ProgressSnapshot {
    /// Completion fraction in `[0.0, 1.0]`, or `None` if total is unknown.
    #[must_use]
    pub fn fraction(&self) -> Option<f64> {
        self.frames_total
            .filter(|&t| t > 0)
            .map(|total| (self.frames_done as f64 / total as f64).clamp(0.0, 1.0))
    }

    /// Estimated frames per second.
    #[must_use]
    pub fn fps(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.frames_done as f64 / self.elapsed_secs
        } else {
            0.0
        }
    }

    /// Estimated remaining seconds, or `None` if total is unknown.
    #[must_use]
    pub fn eta_secs(&self) -> Option<f64> {
        let fps = self.fps();
        if fps <= 0.0 {
            return None;
        }
        self.frames_total.and_then(|t| {
            let remaining = t.saturating_sub(self.frames_done);
            Some(remaining as f64 / fps)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_kind_label() {
        assert_eq!(StageKind::Decode.label(), "decode");
        assert_eq!(StageKind::Encode.label(), "encode");
        assert_eq!(StageKind::Filter.label(), "filter");
        assert_eq!(StageKind::Mux.label(), "mux");
        assert_eq!(StageKind::Demux.label(), "demux");
        assert_eq!(StageKind::Custom("flip".to_string()).label(), "flip");
    }

    #[test]
    fn test_stage_config_new_enabled() {
        let s = StageConfig::new("decode_av1", StageKind::Decode);
        assert!(s.enabled);
        assert_eq!(s.name, "decode_av1");
    }

    #[test]
    fn test_stage_config_with_param() {
        let s = StageConfig::new("enc", StageKind::Encode).with_param("crf", "28");
        assert_eq!(s.get_param("crf"), Some("28"));
    }

    #[test]
    fn test_stage_config_disabled() {
        let s = StageConfig::new("enc", StageKind::Encode).disabled();
        assert!(!s.enabled);
    }

    #[test]
    fn test_stage_config_get_param_missing() {
        let s = StageConfig::new("enc", StageKind::Encode);
        assert_eq!(s.get_param("nonexistent"), None);
    }

    #[test]
    fn test_pipeline_config_push_and_count() {
        let mut p = PipelineConfig::new();
        p.push_stage(StageConfig::new("a", StageKind::Decode));
        p.push_stage(StageConfig::new("b", StageKind::Encode).disabled());
        assert_eq!(p.stages.len(), 2);
        assert_eq!(p.enabled_count(), 1);
    }

    #[test]
    fn test_pipeline_config_find_stage() {
        let mut p = PipelineConfig::new();
        p.push_stage(StageConfig::new("scale", StageKind::Filter));
        assert!(p.find_stage("scale").is_some());
        assert!(p.find_stage("missing").is_none());
    }

    #[test]
    fn test_pipeline_config_metadata() {
        let mut p = PipelineConfig::new();
        p.set_meta("input", "/tmp/video.mkv");
        assert_eq!(
            p.metadata.get("input").map(String::as_str),
            Some("/tmp/video.mkv")
        );
    }

    #[test]
    fn test_progress_snapshot_fraction_known() {
        let snap = ProgressSnapshot {
            frames_done: 50,
            frames_total: Some(100),
            current_stage: "encode".to_string(),
            elapsed_secs: 10.0,
        };
        assert!((snap.fraction().expect("fraction should succeed") - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_progress_snapshot_fraction_unknown() {
        let snap = ProgressSnapshot {
            frames_done: 50,
            frames_total: None,
            current_stage: "encode".to_string(),
            elapsed_secs: 10.0,
        };
        assert!(snap.fraction().is_none());
    }

    #[test]
    fn test_progress_snapshot_fps() {
        let snap = ProgressSnapshot {
            frames_done: 300,
            frames_total: Some(1000),
            current_stage: "encode".to_string(),
            elapsed_secs: 10.0,
        };
        assert!((snap.fps() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_progress_snapshot_eta() {
        let snap = ProgressSnapshot {
            frames_done: 300,
            frames_total: Some(600),
            current_stage: "encode".to_string(),
            elapsed_secs: 10.0,
        };
        // remaining = 300, fps = 30 → eta = 10 s
        assert!((snap.eta_secs().expect("eta_secs should succeed") - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_progress_snapshot_eta_zero_elapsed() {
        let snap = ProgressSnapshot {
            frames_done: 0,
            frames_total: Some(600),
            current_stage: "encode".to_string(),
            elapsed_secs: 0.0,
        };
        assert!(snap.eta_secs().is_none());
    }
}
