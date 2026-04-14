//! Preset chaining and composition.
//!
//! Allows multiple presets to be composed into an ordered pipeline where
//! each step overrides or merges specific settings from the previous step.
//! This is useful for building multi-pass encoding workflows or applying
//! successive refinement layers on top of a base preset.

#![allow(dead_code)]

use std::collections::HashMap;

// ── ChainPriority ──────────────────────────────────────────────────────────

/// Determines how conflicting values are resolved when two chain links
/// specify the same parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainPriority {
    /// Later links overwrite earlier links (last-write-wins).
    LastWins,
    /// Earlier links take precedence; later links only fill gaps.
    FirstWins,
    /// Pick the numerically higher value (for bitrates, quality, etc.).
    HigherWins,
    /// Pick the numerically lower value (for latency, file-size, etc.).
    LowerWins,
}

impl Default for ChainPriority {
    fn default() -> Self {
        Self::LastWins
    }
}

// ── ChainParam ─────────────────────────────────────────────────────────────

/// A single parameter override stored in a chain link.
#[derive(Debug, Clone, PartialEq)]
pub enum ChainParam {
    /// An integer parameter (bitrate, width, height, etc.).
    Int(i64),
    /// A floating-point parameter (quality factor, CRF, etc.).
    Float(f64),
    /// A string parameter (codec name, profile, etc.).
    Text(String),
    /// A boolean flag.
    Bool(bool),
}

impl ChainParam {
    /// Return the integer value if this is an `Int` variant.
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Return the float value if this is a `Float` variant.
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Return a string reference if this is a `Text` variant.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(v) => Some(v),
            _ => None,
        }
    }

    /// Return the bool value if this is a `Bool` variant.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

// ── ChainLink ──────────────────────────────────────────────────────────────

/// A single link in a preset chain.
///
/// Each link carries a human-readable label and a set of named parameter
/// overrides that will be merged according to the chain's [`ChainPriority`].
#[derive(Debug, Clone)]
pub struct ChainLink {
    /// Human-readable label for this link (e.g. "base-1080p", "hdr-overlay").
    pub label: String,
    /// Parameter overrides keyed by canonical parameter name.
    pub params: HashMap<String, ChainParam>,
    /// Whether this link is enabled (disabled links are skipped during merge).
    pub enabled: bool,
}

impl ChainLink {
    /// Create a new, enabled chain link with the given label and no overrides.
    #[must_use]
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            params: HashMap::new(),
            enabled: true,
        }
    }

    /// Set a parameter override.
    pub fn set(&mut self, key: &str, value: ChainParam) {
        self.params.insert(key.to_string(), value);
    }

    /// Builder-style parameter setter.
    #[must_use]
    pub fn with_param(mut self, key: &str, value: ChainParam) -> Self {
        self.set(key, value);
        self
    }

    /// Disable this link so it is skipped during merge.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable this link.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Get a parameter by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&ChainParam> {
        self.params.get(key)
    }

    /// Number of parameter overrides in this link.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.params.len()
    }
}

// ── PresetChain ────────────────────────────────────────────────────────────

/// An ordered sequence of [`ChainLink`]s merged using a [`ChainPriority`].
///
/// Call [`PresetChain::resolve`] to flatten the chain into a single
/// parameter map.
#[derive(Debug, Clone)]
pub struct PresetChain {
    /// Human-readable name for this chain.
    pub name: String,
    /// Ordered links (index 0 is applied first).
    links: Vec<ChainLink>,
    /// Conflict-resolution strategy.
    priority: ChainPriority,
}

impl PresetChain {
    /// Create a new, empty preset chain.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            links: Vec::new(),
            priority: ChainPriority::default(),
        }
    }

    /// Create a chain with a specific priority strategy.
    #[must_use]
    pub fn with_priority(mut self, priority: ChainPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Append a link to the end of the chain.
    pub fn push(&mut self, link: ChainLink) {
        self.links.push(link);
    }

    /// Insert a link at the given position.
    ///
    /// # Panics
    /// Panics if `index > self.len()`.
    pub fn insert(&mut self, index: usize, link: ChainLink) {
        self.links.insert(index, link);
    }

    /// Remove and return the link at the given position.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    pub fn remove(&mut self, index: usize) -> ChainLink {
        self.links.remove(index)
    }

    /// Number of links (including disabled ones).
    #[must_use]
    pub fn len(&self) -> usize {
        self.links.len()
    }

    /// Whether the chain contains no links.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// Number of *enabled* links.
    #[must_use]
    pub fn enabled_count(&self) -> usize {
        self.links.iter().filter(|l| l.enabled).count()
    }

    /// Immutable access to a link by index.
    #[must_use]
    pub fn get_link(&self, index: usize) -> Option<&ChainLink> {
        self.links.get(index)
    }

    /// Mutable access to a link by index.
    pub fn get_link_mut(&mut self, index: usize) -> Option<&mut ChainLink> {
        self.links.get_mut(index)
    }

    /// Resolve (flatten) the chain into a single parameter map.
    ///
    /// Disabled links are skipped. Conflicts are resolved according to the
    /// chain's [`ChainPriority`].
    #[must_use]
    pub fn resolve(&self) -> HashMap<String, ChainParam> {
        let mut result: HashMap<String, ChainParam> = HashMap::new();

        for link in self.links.iter().filter(|l| l.enabled) {
            for (key, value) in &link.params {
                match self.priority {
                    ChainPriority::LastWins => {
                        result.insert(key.clone(), value.clone());
                    }
                    ChainPriority::FirstWins => {
                        result.entry(key.clone()).or_insert_with(|| value.clone());
                    }
                    ChainPriority::HigherWins => {
                        let insert = match result.get(key) {
                            Some(ChainParam::Int(existing)) => {
                                value.as_int().map_or(false, |v| v > *existing)
                            }
                            Some(ChainParam::Float(existing)) => {
                                value.as_float().map_or(false, |v| v > *existing)
                            }
                            None => true,
                            _ => false,
                        };
                        if insert {
                            result.insert(key.clone(), value.clone());
                        }
                    }
                    ChainPriority::LowerWins => {
                        let insert = match result.get(key) {
                            Some(ChainParam::Int(existing)) => {
                                value.as_int().map_or(false, |v| v < *existing)
                            }
                            Some(ChainParam::Float(existing)) => {
                                value.as_float().map_or(false, |v| v < *existing)
                            }
                            None => true,
                            _ => false,
                        };
                        if insert {
                            result.insert(key.clone(), value.clone());
                        }
                    }
                }
            }
        }

        result
    }

    /// Return the labels of all enabled links, in order.
    #[must_use]
    pub fn enabled_labels(&self) -> Vec<&str> {
        self.links
            .iter()
            .filter(|l| l.enabled)
            .map(|l| l.label.as_str())
            .collect()
    }

    /// Return all unique parameter keys mentioned across all enabled links.
    #[must_use]
    pub fn all_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .links
            .iter()
            .filter(|l| l.enabled)
            .flat_map(|l| l.params.keys().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        keys.sort();
        keys
    }
}

// ── ContainerFormat ────────────────────────────────────────────────────────

/// A container / wrapper format for encoded media.
///
/// Used by [`ChainedPreset`] to declare which formats a step can accept as
/// input and which it produces as output.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ContainerFormat {
    /// MPEG-4 container (H.264, AAC, MP4A-LATM …)
    Mp4,
    /// Matroska / WebM container (VP9, AV1, Opus …)
    Mkv,
    /// WebM container.
    WebM,
    /// Raw transport stream (MPEG-TS).
    Ts,
    /// Fragmented MP4 (DASH / CMAF).
    FragmentedMp4,
    /// HTTP Live Streaming segment.
    Hls,
    /// FLAC audio container.
    Flac,
    /// Opus audio in an Ogg container.
    Ogg,
    /// PCM / WAV audio.
    Wav,
    /// A user-defined format token.
    Custom(String),
}

impl ContainerFormat {
    /// Return a human-readable name for the format.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Mp4 => "MP4",
            Self::Mkv => "MKV",
            Self::WebM => "WebM",
            Self::Ts => "TS",
            Self::FragmentedMp4 => "fMP4",
            Self::Hls => "HLS",
            Self::Flac => "FLAC",
            Self::Ogg => "OGG",
            Self::Wav => "WAV",
            Self::Custom(s) => s.as_str(),
        }
    }
}

// ── ChainedPreset ──────────────────────────────────────────────────────────

/// A preset in a pipeline chain, carrying its format contract.
#[derive(Debug, Clone)]
pub struct ChainedPreset {
    /// Human-readable identifier (e.g. "YouTube 1080p").
    pub name: String,
    /// Formats this preset can accept as input (empty = accepts any).
    pub input_formats: Vec<ContainerFormat>,
    /// Format this preset produces as output.
    pub output_format: ContainerFormat,
    /// Codec used by this preset step (informational).
    pub codec: String,
}

impl ChainedPreset {
    /// Create a new chained preset.
    pub fn new(
        name: impl Into<String>,
        codec: impl Into<String>,
        input_formats: Vec<ContainerFormat>,
        output_format: ContainerFormat,
    ) -> Self {
        Self {
            name: name.into(),
            codec: codec.into(),
            input_formats,
            output_format,
        }
    }
}

// ── CompatibilityError ─────────────────────────────────────────────────────

/// Describes a format mismatch between two consecutive chain steps.
#[derive(Debug, Clone, PartialEq)]
pub struct CompatibilityError {
    /// Zero-based index of the step where the mismatch occurs (the *first*
    /// of the pair; the next step is `step + 1`).
    pub step: usize,
    /// Name of the producing preset at `step`.
    pub from_preset: String,
    /// Name of the consuming preset at `step + 1`.
    pub to_preset: String,
    /// Human-readable description of why the pair is incompatible.
    pub reason: String,
}

// ── ChainCompatibilityValidator ────────────────────────────────────────────

/// Validates that the output format of each step is accepted by the next step.
pub struct ChainCompatibilityValidator;

impl ChainCompatibilityValidator {
    /// Check that consecutive preset pairs in `chain` are format-compatible.
    ///
    /// Returns `Ok(())` when the chain is empty, has a single step, or all
    /// consecutive pairs are compatible.  Returns `Err(Vec<CompatibilityError>)`
    /// when one or more mismatches are detected; the vec is never empty in the
    /// `Err` case.
    ///
    /// # Rules
    ///
    /// 1. For each pair `(N, N+1)`: the output format of step N must appear in
    ///    `step[N+1].input_formats` (or `step[N+1].input_formats` must be empty,
    ///    meaning "accepts any format").
    /// 2. Audio-only codecs (FLAC, Ogg/Opus) may not chain into video-output
    ///    presets.
    pub fn validate(chain: &[ChainedPreset]) -> std::result::Result<(), Vec<CompatibilityError>> {
        if chain.len() <= 1 {
            return Ok(());
        }

        let mut errors: Vec<CompatibilityError> = Vec::new();

        for i in 0..(chain.len() - 1) {
            let from = &chain[i];
            let to = &chain[i + 1];

            // Rule 1: format must be accepted.
            let format_ok =
                to.input_formats.is_empty() || to.input_formats.contains(&from.output_format);

            if !format_ok {
                errors.push(CompatibilityError {
                    step: i,
                    from_preset: from.name.clone(),
                    to_preset: to.name.clone(),
                    reason: format!(
                        "output format {} from '{}' is not accepted by '{}' (accepts: {})",
                        from.output_format.as_str(),
                        from.name,
                        to.name,
                        if to.input_formats.is_empty() {
                            "any".to_string()
                        } else {
                            to.input_formats
                                .iter()
                                .map(|f| f.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        }
                    ),
                });
            }

            // Rule 2: audio-only codec cannot feed a video preset.
            let from_audio_only = matches!(
                from.output_format,
                ContainerFormat::Flac | ContainerFormat::Ogg | ContainerFormat::Wav
            );
            let to_has_video = matches!(
                to.output_format,
                ContainerFormat::Mp4
                    | ContainerFormat::Mkv
                    | ContainerFormat::WebM
                    | ContainerFormat::Ts
                    | ContainerFormat::FragmentedMp4
                    | ContainerFormat::Hls
            );
            if format_ok && from_audio_only && to_has_video {
                errors.push(CompatibilityError {
                    step: i,
                    from_preset: from.name.clone(),
                    to_preset: to.name.clone(),
                    reason: format!(
                        "audio-only output ({}) from '{}' cannot feed video preset '{}'",
                        from.output_format.as_str(),
                        from.name,
                        to.name,
                    ),
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn base_link() -> ChainLink {
        ChainLink::new("base")
            .with_param("bitrate", ChainParam::Int(5_000_000))
            .with_param("codec", ChainParam::Text("h264".into()))
            .with_param("crf", ChainParam::Float(23.0))
    }

    fn overlay_link() -> ChainLink {
        ChainLink::new("overlay")
            .with_param("bitrate", ChainParam::Int(8_000_000))
            .with_param("hdr", ChainParam::Bool(true))
    }

    // ── ChainLink ──

    #[test]
    fn test_chain_link_creation() {
        let link = ChainLink::new("test");
        assert_eq!(link.label, "test");
        assert!(link.enabled);
        assert_eq!(link.param_count(), 0);
    }

    #[test]
    fn test_chain_link_with_param() {
        let link = base_link();
        assert_eq!(link.param_count(), 3);
        assert_eq!(
            link.get("bitrate").expect("get should succeed").as_int(),
            Some(5_000_000)
        );
    }

    #[test]
    fn test_chain_link_disable_enable() {
        let mut link = ChainLink::new("x");
        assert!(link.enabled);
        link.disable();
        assert!(!link.enabled);
        link.enable();
        assert!(link.enabled);
    }

    #[test]
    fn test_chain_param_as_text() {
        let p = ChainParam::Text("hevc".into());
        assert_eq!(p.as_text(), Some("hevc"));
        assert_eq!(p.as_int(), None);
    }

    #[test]
    fn test_chain_param_as_bool() {
        let p = ChainParam::Bool(true);
        assert_eq!(p.as_bool(), Some(true));
        assert_eq!(p.as_float(), None);
    }

    // ── PresetChain ──

    #[test]
    fn test_chain_empty() {
        let chain = PresetChain::new("empty");
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert!(chain.resolve().is_empty());
    }

    #[test]
    fn test_chain_last_wins_default() {
        let mut chain = PresetChain::new("test");
        chain.push(base_link());
        chain.push(overlay_link());
        let resolved = chain.resolve();
        // overlay's bitrate (8M) should overwrite base's (5M)
        assert_eq!(
            resolved
                .get("bitrate")
                .expect("get should succeed")
                .as_int(),
            Some(8_000_000)
        );
        // codec comes from base only
        assert_eq!(
            resolved.get("codec").expect("get should succeed").as_text(),
            Some("h264")
        );
        // hdr comes from overlay
        assert_eq!(
            resolved.get("hdr").expect("get should succeed").as_bool(),
            Some(true)
        );
    }

    #[test]
    fn test_chain_first_wins() {
        let mut chain = PresetChain::new("fw").with_priority(ChainPriority::FirstWins);
        chain.push(base_link());
        chain.push(overlay_link());
        let resolved = chain.resolve();
        // base's bitrate (5M) wins
        assert_eq!(
            resolved
                .get("bitrate")
                .expect("get should succeed")
                .as_int(),
            Some(5_000_000)
        );
    }

    #[test]
    fn test_chain_higher_wins() {
        let mut chain = PresetChain::new("hw").with_priority(ChainPriority::HigherWins);
        chain.push(base_link());
        chain.push(overlay_link());
        let resolved = chain.resolve();
        assert_eq!(
            resolved
                .get("bitrate")
                .expect("get should succeed")
                .as_int(),
            Some(8_000_000)
        );
    }

    #[test]
    fn test_chain_lower_wins() {
        let mut chain = PresetChain::new("lw").with_priority(ChainPriority::LowerWins);
        chain.push(base_link());
        chain.push(overlay_link());
        let resolved = chain.resolve();
        assert_eq!(
            resolved
                .get("bitrate")
                .expect("get should succeed")
                .as_int(),
            Some(5_000_000)
        );
    }

    #[test]
    fn test_chain_disabled_link_skipped() {
        let mut chain = PresetChain::new("skip");
        chain.push(base_link());
        let mut disabled = overlay_link();
        disabled.disable();
        chain.push(disabled);
        let resolved = chain.resolve();
        assert_eq!(
            resolved
                .get("bitrate")
                .expect("get should succeed")
                .as_int(),
            Some(5_000_000)
        );
        assert!(!resolved.contains_key("hdr"));
    }

    #[test]
    fn test_chain_enabled_count() {
        let mut chain = PresetChain::new("ec");
        chain.push(base_link());
        let mut d = overlay_link();
        d.disable();
        chain.push(d);
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.enabled_count(), 1);
    }

    #[test]
    fn test_chain_enabled_labels() {
        let mut chain = PresetChain::new("labels");
        chain.push(base_link());
        chain.push(overlay_link());
        assert_eq!(chain.enabled_labels(), vec!["base", "overlay"]);
    }

    #[test]
    fn test_chain_all_keys() {
        let mut chain = PresetChain::new("keys");
        chain.push(base_link());
        chain.push(overlay_link());
        let keys = chain.all_keys();
        assert!(keys.contains(&"bitrate".to_string()));
        assert!(keys.contains(&"codec".to_string()));
        assert!(keys.contains(&"hdr".to_string()));
        assert!(keys.contains(&"crf".to_string()));
    }

    #[test]
    fn test_chain_insert_and_remove() {
        let mut chain = PresetChain::new("ir");
        chain.push(base_link());
        chain.push(overlay_link());
        let mid = ChainLink::new("mid");
        chain.insert(1, mid);
        assert_eq!(chain.len(), 3);
        assert_eq!(
            chain.get_link(1).expect("get_link should succeed").label,
            "mid"
        );
        let removed = chain.remove(1);
        assert_eq!(removed.label, "mid");
        assert_eq!(chain.len(), 2);
    }

    // ── ChainCompatibilityValidator ──────────────────────────────────────────

    fn mp4_preset(name: &str) -> ChainedPreset {
        ChainedPreset::new(
            name,
            "h264",
            vec![ContainerFormat::Mp4],
            ContainerFormat::Mp4,
        )
    }

    fn mkv_preset(name: &str) -> ChainedPreset {
        ChainedPreset::new(
            name,
            "vp9",
            vec![ContainerFormat::Mkv],
            ContainerFormat::Mkv,
        )
    }

    fn any_input_preset(name: &str) -> ChainedPreset {
        ChainedPreset::new(
            name,
            "h264",
            vec![], // empty = accepts any
            ContainerFormat::Mp4,
        )
    }

    fn flac_preset(name: &str) -> ChainedPreset {
        ChainedPreset::new(
            name,
            "flac",
            vec![ContainerFormat::Flac],
            ContainerFormat::Flac,
        )
    }

    #[test]
    fn test_single_preset_always_compatible() {
        let chain = vec![mp4_preset("step1")];
        assert!(ChainCompatibilityValidator::validate(&chain).is_ok());
    }

    #[test]
    fn test_empty_chain_compatible() {
        let chain: Vec<ChainedPreset> = vec![];
        assert!(ChainCompatibilityValidator::validate(&chain).is_ok());
    }

    #[test]
    fn test_compatible_chain_mp4_to_mp4() {
        let chain = vec![mp4_preset("ingest"), mp4_preset("deliver")];
        assert!(ChainCompatibilityValidator::validate(&chain).is_ok());
    }

    #[test]
    fn test_compatible_chain_any_input() {
        // "any_input" accepts everything, so mp4 output feeds it fine.
        let chain = vec![mp4_preset("ingest"), any_input_preset("deliver")];
        assert!(ChainCompatibilityValidator::validate(&chain).is_ok());
    }

    #[test]
    fn test_incompatible_chain_mp4_to_mkv_input() {
        let chain = vec![mp4_preset("step1"), mkv_preset("step2")];
        let result = ChainCompatibilityValidator::validate(&chain);
        assert!(result.is_err(), "MP4 output should not feed MKV-only input");
        let errors = result.expect_err("should have errors");
        assert_eq!(errors[0].step, 0);
        assert_eq!(errors[0].from_preset, "step1");
        assert_eq!(errors[0].to_preset, "step2");
    }

    #[test]
    fn test_incompatible_chain_reports_all_errors() {
        // step0→step1 ok (both mp4); step1→step2 fail (mp4→mkv-only).
        let chain = vec![mp4_preset("s0"), mp4_preset("s1"), mkv_preset("s2")];
        let errors = ChainCompatibilityValidator::validate(&chain).expect_err("should have errors");
        assert_eq!(errors.len(), 1, "Only one mismatch at step 1→2");
        assert_eq!(errors[0].step, 1);
    }

    #[test]
    fn test_audio_only_cannot_feed_video_preset() {
        // FLAC → MP4 video preset: should be caught by codec rule.
        let chain = vec![
            flac_preset("audio-encode"),
            any_input_preset("mp4-mux"), // accepts any, outputs MP4
        ];
        let errors = ChainCompatibilityValidator::validate(&chain)
            .expect_err("audio-only feeding video should be an error");
        assert!(!errors.is_empty());
        assert!(
            errors[0].reason.contains("audio-only"),
            "Reason should mention audio-only"
        );
    }

    #[test]
    fn test_compatibility_error_fields() {
        let chain = vec![mp4_preset("A"), mkv_preset("B")];
        let errors = ChainCompatibilityValidator::validate(&chain).expect_err("should have errors");
        let err = &errors[0];
        assert_eq!(err.step, 0);
        assert_eq!(err.from_preset, "A");
        assert_eq!(err.to_preset, "B");
        assert!(!err.reason.is_empty());
    }

    #[test]
    fn test_three_step_all_compatible() {
        let chain = vec![
            mp4_preset("ingest"),
            mp4_preset("transcode"),
            mp4_preset("deliver"),
        ];
        assert!(ChainCompatibilityValidator::validate(&chain).is_ok());
    }

    #[test]
    fn test_container_format_as_str() {
        assert_eq!(ContainerFormat::Mp4.as_str(), "MP4");
        assert_eq!(ContainerFormat::Flac.as_str(), "FLAC");
        assert_eq!(ContainerFormat::Custom("xyz".to_string()).as_str(), "xyz");
    }

    #[test]
    fn test_chained_preset_construction() {
        let p = ChainedPreset::new(
            "test",
            "av1",
            vec![ContainerFormat::WebM],
            ContainerFormat::Mkv,
        );
        assert_eq!(p.name, "test");
        assert_eq!(p.codec, "av1");
        assert_eq!(p.output_format, ContainerFormat::Mkv);
        assert_eq!(p.input_formats.len(), 1);
    }
}
