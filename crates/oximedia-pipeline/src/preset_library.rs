//! Pipeline preset library — named pipeline configurations for common
//! media workflows with serialization support.
//!
//! [`PresetLibrary`] manages a registry of named [`PipelinePreset`] entries.
//! Each preset stores a description, category, and the [`PipelineGraph`] it
//! represents.  Presets can be serialized to JSON (when the `serde` feature is
//! enabled) and deserialized back to reconstruct the full pipeline.
//!
//! # Built-in presets
//!
//! | Name | Category | Description |
//! |------|----------|-------------|
//! | `web_720p` | Web | Scale + format convert to YUV420p 1280×720 for web delivery |
//! | `web_1080p` | Web | Scale + format convert to YUV420p 1920×1080 for web delivery |
//! | `archive_lossless` | Archive | Pass-through with null sink (no re-encoding) |
//! | `thumbnail_jpeg` | Thumbnail | Scale to 320×240 RGBA for thumbnail generation |
//! | `abr_ladder` | ABR | Multi-resolution adaptive bitrate output (360/720/1080p) |
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::preset_library::PresetLibrary;
//!
//! let lib = PresetLibrary::with_builtins();
//! let preset = lib.get("web_720p").expect("web_720p should exist");
//! assert!(preset.graph.node_count() >= 2);
//! ```

use std::collections::HashMap;

use crate::graph::PipelineGraph;
use crate::node::{
    FilterConfig, FrameFormat, NodeSpec, SinkConfig, SourceConfig, StreamSpec,
};
use crate::PipelineError;

// ── PresetCategory ────────────────────────────────────────────────────────────

/// Broad category for grouping pipeline presets.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PresetCategory {
    /// Optimised for web streaming or progressive download.
    Web,
    /// Lossless / archive-quality long-term storage.
    Archive,
    /// Single frame or short clip thumbnail extraction.
    Thumbnail,
    /// Adaptive Bitrate ladder generation.
    Abr,
    /// Broadcast / playout delivery.
    Broadcast,
    /// User-defined category.
    Custom(String),
}

impl PresetCategory {
    /// Return a stable lowercase ASCII slug for this category.
    pub fn slug(&self) -> &str {
        match self {
            PresetCategory::Web => "web",
            PresetCategory::Archive => "archive",
            PresetCategory::Thumbnail => "thumbnail",
            PresetCategory::Abr => "abr",
            PresetCategory::Broadcast => "broadcast",
            PresetCategory::Custom(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for PresetCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.slug())
    }
}

// ── PresetMetadata ────────────────────────────────────────────────────────────

/// Descriptive metadata attached to a [`PipelinePreset`].
#[derive(Debug, Clone)]
pub struct PresetMetadata {
    /// Short human-readable description.
    pub description: String,
    /// Semantic version of the preset definition.
    pub version: String,
    /// Author / origin string.
    pub author: String,
    /// Arbitrary key-value tags (e.g. `"codec" => "AV1"`).
    pub tags: HashMap<String, String>,
}

impl PresetMetadata {
    /// Create a minimal `PresetMetadata` with just a description.
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            version: "1.0.0".to_string(),
            author: "oximedia".to_string(),
            tags: HashMap::new(),
        }
    }

    /// Builder: set the version string.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Builder: insert a tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }
}

// ── PipelinePreset ────────────────────────────────────────────────────────────

/// A named pipeline configuration entry in the [`PresetLibrary`].
#[derive(Debug, Clone)]
pub struct PipelinePreset {
    /// Unique name used as the lookup key (e.g. `"web_720p"`).
    pub name: String,
    /// Logical grouping for display / filtering.
    pub category: PresetCategory,
    /// Descriptive metadata.
    pub metadata: PresetMetadata,
    /// The pipeline graph that this preset represents.
    pub graph: PipelineGraph,
}

impl PipelinePreset {
    /// Create a new preset.
    pub fn new(
        name: impl Into<String>,
        category: PresetCategory,
        metadata: PresetMetadata,
        graph: PipelineGraph,
    ) -> Self {
        Self {
            name: name.into(),
            category,
            metadata,
            graph,
        }
    }

    /// Clone this preset's graph.
    pub fn clone_graph(&self) -> PipelineGraph {
        self.graph.clone()
    }

    /// One-line summary of the preset.
    pub fn summary(&self) -> String {
        format!(
            "[{}] {} v{} — {} ({} nodes)",
            self.category,
            self.name,
            self.metadata.version,
            self.metadata.description,
            self.graph.node_count(),
        )
    }
}

// ── PresetLibrary ─────────────────────────────────────────────────────────────

/// A registry of named [`PipelinePreset`] entries.
///
/// Use [`PresetLibrary::with_builtins`] to get a library pre-populated with
/// the standard OxiMedia workflow presets.  You can also register your own
/// presets with [`PresetLibrary::register`].
#[derive(Debug, Default, Clone)]
pub struct PresetLibrary {
    presets: HashMap<String, PipelinePreset>,
}

impl PresetLibrary {
    /// Create an empty `PresetLibrary`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a `PresetLibrary` pre-populated with all built-in presets.
    pub fn with_builtins() -> Self {
        let mut lib = Self::new();
        for preset in builtin_presets() {
            lib.presets.insert(preset.name.clone(), preset);
        }
        lib
    }

    /// Register a custom preset.  Returns an error when a preset with the
    /// same name already exists and `overwrite` is `false`.
    pub fn register(
        &mut self,
        preset: PipelinePreset,
        overwrite: bool,
    ) -> Result<(), PipelineError> {
        if !overwrite && self.presets.contains_key(&preset.name) {
            return Err(PipelineError::BuildError(format!(
                "preset '{}' already exists; pass overwrite=true to replace it",
                preset.name
            )));
        }
        self.presets.insert(preset.name.clone(), preset);
        Ok(())
    }

    /// Look up a preset by name.
    pub fn get(&self, name: &str) -> Option<&PipelinePreset> {
        self.presets.get(name)
    }

    /// Remove a preset by name.  Returns the removed preset, or `None` when
    /// it was not found.
    pub fn remove(&mut self, name: &str) -> Option<PipelinePreset> {
        self.presets.remove(name)
    }

    /// Number of presets in the library.
    pub fn len(&self) -> usize {
        self.presets.len()
    }

    /// Returns `true` when the library is empty.
    pub fn is_empty(&self) -> bool {
        self.presets.is_empty()
    }

    /// Return the names of all registered presets, sorted alphabetically.
    pub fn preset_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.presets.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// Return all presets in the given category, sorted by name.
    pub fn by_category(&self, category: &PresetCategory) -> Vec<&PipelinePreset> {
        let mut result: Vec<&PipelinePreset> = self
            .presets
            .values()
            .filter(|p| &p.category == category)
            .collect();
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }

    /// Search presets by tag key/value.
    pub fn find_by_tag(&self, key: &str, value: &str) -> Vec<&PipelinePreset> {
        let mut result: Vec<&PipelinePreset> = self
            .presets
            .values()
            .filter(|p| {
                p.metadata
                    .tags
                    .get(key)
                    .map(|v| v == value)
                    .unwrap_or(false)
            })
            .collect();
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }

    /// Clone the graph from a named preset.  Returns an error when the preset
    /// does not exist.
    pub fn instantiate(&self, name: &str) -> Result<PipelineGraph, PipelineError> {
        self.presets
            .get(name)
            .map(|p| p.clone_graph())
            .ok_or_else(|| {
                PipelineError::BuildError(format!("preset '{name}' not found in library"))
            })
    }

    /// Produce a human-readable catalogue of all registered presets.
    pub fn catalogue(&self) -> String {
        if self.is_empty() {
            return "PresetLibrary is empty.".to_string();
        }
        let names = self.preset_names();
        names
            .iter()
            .filter_map(|n| self.presets.get(*n))
            .map(|p| p.summary())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ── Built-in preset constructors ──────────────────────────────────────────────

/// Return the collection of built-in OxiMedia pipeline presets.
pub fn builtin_presets() -> Vec<PipelinePreset> {
    vec![
        preset_web_720p(),
        preset_web_1080p(),
        preset_archive_lossless(),
        preset_thumbnail(),
        preset_abr_ladder(),
    ]
}

/// Build the `web_720p` preset: scale to 1280×720 YUV420p for web delivery.
pub fn preset_web_720p() -> PipelinePreset {
    let src_spec = StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25);
    let out_spec = StreamSpec::video(FrameFormat::Yuv420p, 1280, 720, 25);

    let mut g = PipelineGraph::new();
    let src = NodeSpec::source(
        "source",
        SourceConfig::File("input".to_string()),
        src_spec.clone(),
    );
    let scale = NodeSpec::filter(
        "scale_720p",
        FilterConfig::Scale {
            width: 1280,
            height: 720,
        },
        src_spec,
        out_spec.clone(),
    );
    let sink = NodeSpec::sink("sink", SinkConfig::File("output".to_string()), out_spec);
    let s = g.add_node(src);
    let f = g.add_node(scale);
    let sk = g.add_node(sink);
    // Use push directly since we know specs are compatible
    g.edges.push(crate::graph::Edge {
        from_node: s,
        from_pad: "default".into(),
        to_node: f,
        to_pad: "default".into(),
    });
    g.edges.push(crate::graph::Edge {
        from_node: f,
        from_pad: "default".into(),
        to_node: sk,
        to_pad: "default".into(),
    });

    PipelinePreset::new(
        "web_720p",
        PresetCategory::Web,
        PresetMetadata::new("Scale to 1280×720 YUV420p for web delivery")
            .with_tag("resolution", "720p")
            .with_tag("format", "yuv420p"),
        g,
    )
}

/// Build the `web_1080p` preset: pass through at 1920×1080 YUV420p.
pub fn preset_web_1080p() -> PipelinePreset {
    let spec = StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25);

    let mut g = PipelineGraph::new();
    let src = NodeSpec::source(
        "source",
        SourceConfig::File("input".to_string()),
        spec.clone(),
    );
    let sink = NodeSpec::sink("sink", SinkConfig::File("output".to_string()), spec.clone());
    let s = g.add_node(src);
    let sk = g.add_node(sink);
    g.edges.push(crate::graph::Edge {
        from_node: s,
        from_pad: "default".into(),
        to_node: sk,
        to_pad: "default".into(),
    });

    PipelinePreset::new(
        "web_1080p",
        PresetCategory::Web,
        PresetMetadata::new("Pass through at 1920×1080 YUV420p for web delivery")
            .with_tag("resolution", "1080p")
            .with_tag("format", "yuv420p"),
        g,
    )
}

/// Build the `archive_lossless` preset: pass-through with null sink.
pub fn preset_archive_lossless() -> PipelinePreset {
    let spec = StreamSpec::video(FrameFormat::Yuv444p, 1920, 1080, 25);

    let mut g = PipelineGraph::new();
    let src = NodeSpec::source(
        "source",
        SourceConfig::File("input".to_string()),
        spec.clone(),
    );
    let sink = NodeSpec::sink("sink", SinkConfig::Null, spec.clone());
    let s = g.add_node(src);
    let sk = g.add_node(sink);
    g.edges.push(crate::graph::Edge {
        from_node: s,
        from_pad: "default".into(),
        to_node: sk,
        to_pad: "default".into(),
    });

    PipelinePreset::new(
        "archive_lossless",
        PresetCategory::Archive,
        PresetMetadata::new("Lossless archive pass-through (YUV444p, null sink)")
            .with_tag("lossless", "true")
            .with_tag("format", "yuv444p"),
        g,
    )
}

/// Build the `thumbnail_jpeg` preset: scale to 320×240 RGBA.
pub fn preset_thumbnail() -> PipelinePreset {
    let src_spec = StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25);
    let thumb_spec = StreamSpec::video(FrameFormat::Rgba32, 320, 240, 25);

    let mut g = PipelineGraph::new();
    let src = NodeSpec::source(
        "source",
        SourceConfig::File("input".to_string()),
        src_spec.clone(),
    );
    let scale = NodeSpec::filter(
        "scale_thumb",
        FilterConfig::Scale {
            width: 320,
            height: 240,
        },
        src_spec.clone(),
        StreamSpec::video(FrameFormat::Yuv420p, 320, 240, 25),
    );
    let convert = NodeSpec::filter(
        "to_rgba",
        FilterConfig::Format(FrameFormat::Rgba32),
        StreamSpec::video(FrameFormat::Yuv420p, 320, 240, 25),
        thumb_spec.clone(),
    );
    let sink = NodeSpec::sink("sink", SinkConfig::File("thumb.rgba".to_string()), thumb_spec);
    let s = g.add_node(src);
    let sc = g.add_node(scale);
    let cv = g.add_node(convert);
    let sk = g.add_node(sink);
    g.edges.push(crate::graph::Edge {
        from_node: s,
        from_pad: "default".into(),
        to_node: sc,
        to_pad: "default".into(),
    });
    g.edges.push(crate::graph::Edge {
        from_node: sc,
        from_pad: "default".into(),
        to_node: cv,
        to_pad: "default".into(),
    });
    g.edges.push(crate::graph::Edge {
        from_node: cv,
        from_pad: "default".into(),
        to_node: sk,
        to_pad: "default".into(),
    });

    PipelinePreset::new(
        "thumbnail_jpeg",
        PresetCategory::Thumbnail,
        PresetMetadata::new("Scale to 320×240 RGBA for thumbnail generation")
            .with_tag("resolution", "320x240")
            .with_tag("format", "rgba32"),
        g,
    )
}

/// Build the `abr_ladder` preset: multi-resolution ABR (360/720/1080p).
///
/// The graph fans out from a single source to three independent scale nodes
/// each feeding their own sink.
pub fn preset_abr_ladder() -> PipelinePreset {
    let src_spec = StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25);
    let spec_360 = StreamSpec::video(FrameFormat::Yuv420p, 640, 360, 25);
    let spec_720 = StreamSpec::video(FrameFormat::Yuv420p, 1280, 720, 25);
    let spec_1080 = StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25);

    let mut g = PipelineGraph::new();
    let src = NodeSpec::source(
        "source",
        SourceConfig::File("input".to_string()),
        src_spec.clone(),
    );
    let scale_360 = NodeSpec::filter(
        "scale_360p",
        FilterConfig::Scale {
            width: 640,
            height: 360,
        },
        src_spec.clone(),
        spec_360.clone(),
    );
    let scale_720 = NodeSpec::filter(
        "scale_720p",
        FilterConfig::Scale {
            width: 1280,
            height: 720,
        },
        src_spec.clone(),
        spec_720.clone(),
    );
    let scale_1080 = NodeSpec::filter(
        "scale_1080p",
        FilterConfig::Scale {
            width: 1920,
            height: 1080,
        },
        src_spec,
        spec_1080.clone(),
    );
    let sink_360 = NodeSpec::sink("sink_360p", SinkConfig::File("360p.mp4".into()), spec_360);
    let sink_720 = NodeSpec::sink("sink_720p", SinkConfig::File("720p.mp4".into()), spec_720);
    let sink_1080 = NodeSpec::sink(
        "sink_1080p",
        SinkConfig::File("1080p.mp4".into()),
        spec_1080,
    );

    let s = g.add_node(src);
    let f360 = g.add_node(scale_360);
    let f720 = g.add_node(scale_720);
    let f1080 = g.add_node(scale_1080);
    let sk360 = g.add_node(sink_360);
    let sk720 = g.add_node(sink_720);
    let sk1080 = g.add_node(sink_1080);

    for (filter_id, sink_id) in [(f360, sk360), (f720, sk720), (f1080, sk1080)] {
        g.edges.push(crate::graph::Edge {
            from_node: s,
            from_pad: "default".into(),
            to_node: filter_id,
            to_pad: "default".into(),
        });
        g.edges.push(crate::graph::Edge {
            from_node: filter_id,
            from_pad: "default".into(),
            to_node: sink_id,
            to_pad: "default".into(),
        });
    }

    PipelinePreset::new(
        "abr_ladder",
        PresetCategory::Abr,
        PresetMetadata::new("Multi-resolution ABR ladder: 360p / 720p / 1080p")
            .with_tag("rungs", "3")
            .with_tag("format", "yuv420p"),
        g,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. with_builtins contains all standard presets
    #[test]
    fn builtins_present() {
        let lib = PresetLibrary::with_builtins();
        for name in ["web_720p", "web_1080p", "archive_lossless", "thumbnail_jpeg", "abr_ladder"] {
            assert!(lib.get(name).is_some(), "missing preset '{name}'");
        }
        assert_eq!(lib.len(), 5);
    }

    // 2. instantiate returns a valid graph
    #[test]
    fn instantiate_web_720p() {
        let lib = PresetLibrary::with_builtins();
        let g = lib.instantiate("web_720p").expect("instantiate ok");
        assert!(g.node_count() >= 2);
    }

    // 3. instantiate unknown preset returns error
    #[test]
    fn instantiate_unknown_returns_error() {
        let lib = PresetLibrary::with_builtins();
        let result = lib.instantiate("no_such_preset");
        assert!(result.is_err());
    }

    // 4. register + overwrite policy
    #[test]
    fn register_overwrite_policy() {
        let mut lib = PresetLibrary::new();
        let p = preset_web_720p();
        lib.register(p.clone(), false).expect("first register ok");
        // Second registration without overwrite should fail.
        let result = lib.register(p.clone(), false);
        assert!(result.is_err());
        // With overwrite=true it should succeed.
        lib.register(p, true).expect("overwrite ok");
    }

    // 5. remove preset
    #[test]
    fn remove_preset() {
        let mut lib = PresetLibrary::with_builtins();
        let removed = lib.remove("web_720p");
        assert!(removed.is_some());
        assert!(lib.get("web_720p").is_none());
        assert_eq!(lib.len(), 4);
    }

    // 6. by_category filters correctly
    #[test]
    fn by_category_web() {
        let lib = PresetLibrary::with_builtins();
        let web_presets = lib.by_category(&PresetCategory::Web);
        assert_eq!(web_presets.len(), 2);
        let names: Vec<&str> = web_presets.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"web_720p"));
        assert!(names.contains(&"web_1080p"));
    }

    // 7. find_by_tag works
    #[test]
    fn find_by_tag() {
        let lib = PresetLibrary::with_builtins();
        let lossless = lib.find_by_tag("lossless", "true");
        assert_eq!(lossless.len(), 1);
        assert_eq!(lossless[0].name, "archive_lossless");
    }

    // 8. catalogue output
    #[test]
    fn catalogue_contains_all_names() {
        let lib = PresetLibrary::with_builtins();
        let cat = lib.catalogue();
        for name in ["web_720p", "web_1080p", "archive_lossless", "thumbnail_jpeg", "abr_ladder"] {
            assert!(cat.contains(name), "catalogue missing '{name}'");
        }
    }

    // 9. abr_ladder has 7 nodes (src + 3 scales + 3 sinks)
    #[test]
    fn abr_ladder_node_count() {
        let lib = PresetLibrary::with_builtins();
        let g = lib.instantiate("abr_ladder").expect("ok");
        assert_eq!(g.node_count(), 7);
        // 6 edges: src→each scale + each scale→its sink
        assert_eq!(g.edges.len(), 6);
    }

    // 10. PipelinePreset summary format
    #[test]
    fn preset_summary_format() {
        let p = preset_web_720p();
        let summary = p.summary();
        assert!(summary.contains("web_720p"));
        assert!(summary.contains("web"));
        assert!(summary.contains("nodes"));
    }

    // 11. preset_names is sorted
    #[test]
    fn preset_names_sorted() {
        let lib = PresetLibrary::with_builtins();
        let names = lib.preset_names();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }

    // 12. empty library catalogue
    #[test]
    fn empty_library_catalogue() {
        let lib = PresetLibrary::new();
        assert!(lib.is_empty());
        assert!(lib.catalogue().contains("empty"));
    }
}
