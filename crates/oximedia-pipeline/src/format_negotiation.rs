//! Pipeline format negotiation — auto-insert format conversion nodes, negotiate
//! compatible pixel formats between connected stages.
//!
//! [`FormatNegotiator`] walks the edges of a [`PipelineGraph`] and inserts
//! [`FilterConfig::Format`] conversion nodes wherever the output format of one
//! node does not match the expected input format of its successor.  The result
//! is a new [`PipelineGraph`] that is format-consistent end-to-end.
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::graph::PipelineGraph;
//! use oximedia_pipeline::node::{NodeSpec, SourceConfig, SinkConfig, StreamSpec, FrameFormat};
//! use oximedia_pipeline::format_negotiation::FormatNegotiator;
//!
//! let mut g = PipelineGraph::new();
//! let src_spec = StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25);
//! let sink_spec = StreamSpec::video(FrameFormat::Rgb24, 1920, 1080, 25);
//!
//! let src = NodeSpec::source("src", SourceConfig::File("in.mkv".into()), src_spec.clone());
//! let sink = NodeSpec::sink("sink", SinkConfig::Null, sink_spec.clone());
//! let s = g.add_node(src);
//! let sk = g.add_node(sink);
//! // force-add a cross-format edge without the normal guard
//! g.edges.push(oximedia_pipeline::graph::Edge {
//!     from_node: s, from_pad: "default".into(),
//!     to_node: sk, to_pad: "default".into(),
//! });
//!
//! let negotiator = FormatNegotiator::new();
//! let (result, _report) = negotiator.negotiate(g).expect("negotiation ok");
//! // conversion node should have been injected
//! assert!(result.node_count() >= 3);
//! ```

use std::collections::HashMap;

use crate::graph::{Edge, PipelineGraph};
use crate::node::{FilterConfig, FrameFormat, NodeId, NodeSpec, StreamSpec};
use crate::PipelineError;

// ── FormatConversionRule ──────────────────────────────────────────────────────

/// Describes an auto-negotiated format conversion that was inserted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatConversionRule {
    /// Human-readable label of the upstream node.
    pub from_node_name: String,
    /// Pad name on the upstream node.
    pub from_pad: String,
    /// Human-readable label of the downstream node.
    pub to_node_name: String,
    /// Pad name on the downstream node.
    pub to_pad: String,
    /// The source format before conversion.
    pub source_format: FrameFormat,
    /// The target format required by the downstream node.
    pub target_format: FrameFormat,
    /// The `NodeId` of the injected conversion node.
    pub converter_node_id: NodeId,
}

// ── NegotiationReport ─────────────────────────────────────────────────────────

/// A summary produced by [`FormatNegotiator::negotiate`].
#[derive(Debug, Clone, Default)]
pub struct NegotiationReport {
    /// All conversion nodes that were automatically inserted.
    pub inserted_conversions: Vec<FormatConversionRule>,
    /// Edges that were left unchanged because both sides already agreed.
    pub unchanged_edges: usize,
    /// Edges that could not be negotiated (e.g. audio→video mismatch).
    pub incompatible_edges: Vec<String>,
}

impl NegotiationReport {
    /// Returns `true` when no incompatible edges were found.
    pub fn is_compatible(&self) -> bool {
        self.incompatible_edges.is_empty()
    }

    /// Total number of conversions inserted during negotiation.
    pub fn conversion_count(&self) -> usize {
        self.inserted_conversions.len()
    }

    /// One-line human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "negotiation: {} conversions inserted, {} unchanged, {} incompatible",
            self.inserted_conversions.len(),
            self.unchanged_edges,
            self.incompatible_edges.len(),
        )
    }
}

// ── FormatCompatibility ───────────────────────────────────────────────────────

/// Result of checking whether two formats need a conversion node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatCompatibility {
    /// Both sides already use the same format — no conversion needed.
    Compatible,
    /// Formats differ but are within the same stream kind — a format filter
    /// can bridge them.
    NeedsConversion {
        /// The downstream pad's required format.
        target: FrameFormat,
    },
    /// Formats belong to different stream kinds (audio ↔ video) — cannot be
    /// bridged by a simple format filter.
    Incompatible,
}

impl FormatCompatibility {
    /// Evaluate compatibility between an upstream output format `from` and a
    /// downstream input format `to`.
    pub fn check(from: FrameFormat, to: FrameFormat) -> Self {
        if from == to {
            return FormatCompatibility::Compatible;
        }
        // Same domain (both video or both audio)?
        if from.is_video() == to.is_video() {
            FormatCompatibility::NeedsConversion { target: to }
        } else {
            FormatCompatibility::Incompatible
        }
    }
}

// ── FormatNegotiatorConfig ────────────────────────────────────────────────────

/// Configuration for the format negotiator.
#[derive(Debug, Clone)]
pub struct FormatNegotiatorConfig {
    /// When `true`, edges where the upstream pad is unknown (e.g. the pad name
    /// does not match any registered output) are silently skipped instead of
    /// causing an error.
    pub skip_unknown_pads: bool,
    /// When `true`, incompatible edges (audio↔video) produce an error instead
    /// of being silently noted in the report.
    pub error_on_incompatible: bool,
}

impl Default for FormatNegotiatorConfig {
    fn default() -> Self {
        Self {
            skip_unknown_pads: true,
            error_on_incompatible: false,
        }
    }
}

// ── FormatNegotiator ──────────────────────────────────────────────────────────

/// Walks a [`PipelineGraph`] and auto-inserts format conversion filter nodes
/// wherever an edge crosses a format boundary.
///
/// The negotiator returns a **new** graph with the additional nodes; the
/// original graph is consumed.
#[derive(Debug, Clone, Default)]
pub struct FormatNegotiator {
    /// Negotiation configuration.
    pub config: FormatNegotiatorConfig,
}

impl FormatNegotiator {
    /// Create a new `FormatNegotiator` with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a `FormatNegotiator` with custom configuration.
    pub fn with_config(config: FormatNegotiatorConfig) -> Self {
        Self { config }
    }

    /// Negotiate formats for every edge in `graph`.
    ///
    /// Returns the modified graph (with conversion nodes injected) and a
    /// [`NegotiationReport`] detailing all changes made.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::IncompatibleStreams`] when
    /// `config.error_on_incompatible` is `true` and an audio↔video edge is
    /// encountered.
    pub fn negotiate(
        &self,
        graph: PipelineGraph,
    ) -> Result<(PipelineGraph, NegotiationReport), PipelineError> {
        let mut result = PipelineGraph::new();
        // Copy all nodes into the result graph first (maintaining their ids).
        for (_, spec) in &graph.nodes {
            result.nodes.insert(spec.id, spec.clone());
        }

        let mut report = NegotiationReport::default();
        // We will rebuild the edge list, replacing cross-format edges with a
        // conversion-node-plus-two-edges construction.
        let mut new_edges: Vec<Edge> = Vec::new();

        for edge in &graph.edges {
            let from_spec = graph.nodes.get(&edge.from_node);
            let to_spec = graph.nodes.get(&edge.to_node);

            let (from_node_spec, to_node_spec) = match (from_spec, to_spec) {
                (Some(f), Some(t)) => (f, t),
                _ => {
                    // One of the nodes is missing; pass the edge through unchanged.
                    new_edges.push(edge.clone());
                    continue;
                }
            };

            // Resolve pad formats.
            let from_fmt = from_node_spec
                .output_pads
                .iter()
                .find(|(n, _)| n == &edge.from_pad)
                .map(|(_, s)| s.format);

            let to_fmt = to_node_spec
                .input_pads
                .iter()
                .find(|(n, _)| n == &edge.to_pad)
                .map(|(_, s)| s.format);

            match (from_fmt, to_fmt) {
                (None, _) | (_, None) => {
                    // Unknown pad.
                    if self.config.skip_unknown_pads {
                        new_edges.push(edge.clone());
                    }
                    report.unchanged_edges += 1;
                }
                (Some(ff), Some(tf)) => {
                    match FormatCompatibility::check(ff, tf) {
                        FormatCompatibility::Compatible => {
                            new_edges.push(edge.clone());
                            report.unchanged_edges += 1;
                        }
                        FormatCompatibility::NeedsConversion { target } => {
                            // Build a mid-stream StreamSpec by cloning the upstream
                            // output pad spec but overriding the format.
                            let base_spec = from_node_spec
                                .output_pads
                                .iter()
                                .find(|(n, _)| n == &edge.from_pad)
                                .map(|(_, s)| s.clone())
                                .unwrap_or_default();

                            let conv_spec = StreamSpec {
                                format: target,
                                ..base_spec.clone()
                            };

                            // Create a format-conversion filter node.
                            let conv_node = NodeSpec::filter(
                                format!(
                                    "__fmt_conv_{}_{}",
                                    from_node_spec.name, to_node_spec.name
                                ),
                                FilterConfig::Format(target),
                                base_spec,
                                conv_spec,
                            );
                            let conv_id = conv_node.id;
                            result.nodes.insert(conv_id, conv_node);

                            // upstream → converter
                            new_edges.push(Edge {
                                from_node: edge.from_node,
                                from_pad: edge.from_pad.clone(),
                                to_node: conv_id,
                                to_pad: "default".to_string(),
                            });
                            // converter → downstream
                            new_edges.push(Edge {
                                from_node: conv_id,
                                from_pad: "default".to_string(),
                                to_node: edge.to_node,
                                to_pad: edge.to_pad.clone(),
                            });

                            report.inserted_conversions.push(FormatConversionRule {
                                from_node_name: from_node_spec.name.clone(),
                                from_pad: edge.from_pad.clone(),
                                to_node_name: to_node_spec.name.clone(),
                                to_pad: edge.to_pad.clone(),
                                source_format: ff,
                                target_format: tf,
                                converter_node_id: conv_id,
                            });
                        }
                        FormatCompatibility::Incompatible => {
                            let desc = format!(
                                "{}.{} → {}.{}: {:?} is incompatible with {:?}",
                                from_node_spec.name,
                                edge.from_pad,
                                to_node_spec.name,
                                edge.to_pad,
                                ff,
                                tf,
                            );
                            if self.config.error_on_incompatible {
                                return Err(PipelineError::IncompatibleStreams);
                            }
                            report.incompatible_edges.push(desc);
                            // Pass the original edge through unchanged so the
                            // graph remains structurally complete.
                            new_edges.push(edge.clone());
                        }
                    }
                }
            }
        }

        result.edges = new_edges;
        Ok((result, report))
    }

    /// Convenience wrapper that returns only the negotiated graph (discards the
    /// report).
    pub fn negotiate_graph(
        &self,
        graph: PipelineGraph,
    ) -> Result<PipelineGraph, PipelineError> {
        self.negotiate(graph).map(|(g, _)| g)
    }

    /// Return a map from every `FrameFormat` variant to the set of formats it
    /// can be directly converted to (same stream-kind only).
    pub fn conversion_matrix() -> HashMap<FrameFormat, Vec<FrameFormat>> {
        let video = vec![
            FrameFormat::Yuv420p,
            FrameFormat::Yuv422p,
            FrameFormat::Yuv444p,
            FrameFormat::Rgb24,
            FrameFormat::Rgba32,
            FrameFormat::Nv12,
        ];
        let audio = vec![
            FrameFormat::Float32Planar,
            FrameFormat::S16Interleaved,
            FrameFormat::F32Interleaved,
        ];

        let mut map = HashMap::new();
        for &fmt in &video {
            let targets: Vec<FrameFormat> = video.iter().filter(|&&t| t != fmt).copied().collect();
            map.insert(fmt, targets);
        }
        for &fmt in &audio {
            let targets: Vec<FrameFormat> = audio.iter().filter(|&&t| t != fmt).copied().collect();
            map.insert(fmt, targets);
        }
        map
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Edge as GEdge;
    use crate::node::{SinkConfig, SourceConfig, StreamSpec};

    fn video_spec(fmt: FrameFormat) -> StreamSpec {
        StreamSpec::video(fmt, 1280, 720, 25)
    }

    fn audio_spec(fmt: FrameFormat) -> StreamSpec {
        StreamSpec::audio(fmt, 48000, 2)
    }

    fn build_graph_with_formats(
        src_fmt: FrameFormat,
        sink_fmt: FrameFormat,
    ) -> (PipelineGraph, NodeId, NodeId) {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mkv".into()), video_spec(src_fmt));
        let sink = NodeSpec::sink("sink", SinkConfig::Null, video_spec(sink_fmt));
        let s = g.add_node(src);
        let sk = g.add_node(sink);
        // Force-add the edge bypassing format checks in graph.connect()
        g.edges.push(GEdge {
            from_node: s,
            from_pad: "default".into(),
            to_node: sk,
            to_pad: "default".into(),
        });
        (g, s, sk)
    }

    // 1. Compatible formats: no conversion inserted
    #[test]
    fn compatible_formats_no_conversion() {
        let (g, _, _) = build_graph_with_formats(FrameFormat::Yuv420p, FrameFormat::Yuv420p);
        let neg = FormatNegotiator::new();
        let (out, report) = neg.negotiate(g).expect("negotiate ok");
        assert_eq!(report.conversion_count(), 0);
        assert_eq!(report.unchanged_edges, 1);
        // Only src + sink, no extra nodes
        assert_eq!(out.node_count(), 2);
    }

    // 2. Differing video formats: conversion node inserted
    #[test]
    fn different_video_formats_inserts_conversion() {
        let (g, _, _) = build_graph_with_formats(FrameFormat::Yuv420p, FrameFormat::Rgb24);
        let neg = FormatNegotiator::new();
        let (out, report) = neg.negotiate(g).expect("negotiate ok");
        assert_eq!(report.conversion_count(), 1);
        // src + sink + 1 converter
        assert_eq!(out.node_count(), 3);
        // The converter edge list should have 2 edges (src→conv + conv→sink)
        assert_eq!(out.edges.len(), 2);
    }

    // 3. Incompatible audio→video edge: recorded, not error (default config)
    #[test]
    fn audio_video_mismatch_non_fatal_by_default() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source(
            "src",
            SourceConfig::File("in.mp4".into()),
            video_spec(FrameFormat::Yuv420p),
        );
        let sink = NodeSpec::sink("sink", SinkConfig::Null, audio_spec(FrameFormat::S16Interleaved));
        let s = g.add_node(src);
        let sk = g.add_node(sink);
        g.edges.push(GEdge {
            from_node: s,
            from_pad: "default".into(),
            to_node: sk,
            to_pad: "default".into(),
        });
        let neg = FormatNegotiator::new();
        let (_, report) = neg.negotiate(g).expect("should not fail with default config");
        assert!(!report.is_compatible());
        assert_eq!(report.incompatible_edges.len(), 1);
    }

    // 4. error_on_incompatible flag causes error
    #[test]
    fn error_on_incompatible_flag() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source(
            "src",
            SourceConfig::File("in.mp4".into()),
            video_spec(FrameFormat::Yuv420p),
        );
        let sink = NodeSpec::sink(
            "sink",
            SinkConfig::Null,
            audio_spec(FrameFormat::Float32Planar),
        );
        let s = g.add_node(src);
        let sk = g.add_node(sink);
        g.edges.push(GEdge {
            from_node: s,
            from_pad: "default".into(),
            to_node: sk,
            to_pad: "default".into(),
        });
        let mut cfg = FormatNegotiatorConfig::default();
        cfg.error_on_incompatible = true;
        let neg = FormatNegotiator::with_config(cfg);
        let result = neg.negotiate(g);
        assert!(
            result.is_err(),
            "expected error on incompatible audio↔video edge"
        );
    }

    // 5. FormatCompatibility::check basics
    #[test]
    fn format_compatibility_check() {
        use FormatCompatibility::*;
        assert_eq!(
            FormatCompatibility::check(FrameFormat::Yuv420p, FrameFormat::Yuv420p),
            Compatible
        );
        assert_eq!(
            FormatCompatibility::check(FrameFormat::Yuv420p, FrameFormat::Rgb24),
            NeedsConversion { target: FrameFormat::Rgb24 }
        );
        assert_eq!(
            FormatCompatibility::check(FrameFormat::Yuv420p, FrameFormat::S16Interleaved),
            Incompatible
        );
    }

    // 6. Multiple edges: only mismatched ones get converters
    #[test]
    fn multiple_edges_selective_conversion() {
        let mut g = PipelineGraph::new();
        let src1 = NodeSpec::source(
            "src1",
            SourceConfig::File("a.mp4".into()),
            video_spec(FrameFormat::Yuv420p),
        );
        let src2 = NodeSpec::source(
            "src2",
            SourceConfig::File("b.mp4".into()),
            video_spec(FrameFormat::Rgb24),
        );
        let sink1 = NodeSpec::sink("sink1", SinkConfig::Null, video_spec(FrameFormat::Yuv420p));
        let sink2 = NodeSpec::sink("sink2", SinkConfig::Null, video_spec(FrameFormat::Yuv420p));
        let s1 = g.add_node(src1);
        let s2 = g.add_node(src2);
        let sk1 = g.add_node(sink1);
        let sk2 = g.add_node(sink2);
        // s1→sk1 is compatible (yuv420p→yuv420p)
        g.edges.push(GEdge {
            from_node: s1,
            from_pad: "default".into(),
            to_node: sk1,
            to_pad: "default".into(),
        });
        // s2→sk2 needs conversion (rgb24→yuv420p)
        g.edges.push(GEdge {
            from_node: s2,
            from_pad: "default".into(),
            to_node: sk2,
            to_pad: "default".into(),
        });
        let neg = FormatNegotiator::new();
        let (out, report) = neg.negotiate(g).expect("negotiate ok");
        assert_eq!(report.conversion_count(), 1);
        assert_eq!(report.unchanged_edges, 1);
        // 4 original + 1 converter = 5 nodes
        assert_eq!(out.node_count(), 5);
    }

    // 7. NegotiationReport helpers
    #[test]
    fn report_summary_contains_counts() {
        let mut report = NegotiationReport::default();
        report.unchanged_edges = 3;
        report.incompatible_edges.push("a→b".to_string());
        let summary = report.summary();
        assert!(summary.contains("0 conversions"));
        assert!(summary.contains("3 unchanged"));
        assert!(summary.contains("1 incompatible"));
        assert!(!report.is_compatible());
    }

    // 8. conversion_matrix covers all FrameFormat variants
    #[test]
    fn conversion_matrix_completeness() {
        let matrix = FormatNegotiator::conversion_matrix();
        // All video formats are covered
        for fmt in [
            FrameFormat::Yuv420p,
            FrameFormat::Rgb24,
            FrameFormat::Rgba32,
            FrameFormat::Nv12,
        ] {
            assert!(matrix.contains_key(&fmt), "missing {fmt:?} in matrix");
            // Each video format can convert to at least 1 other video format
            let targets = &matrix[&fmt];
            assert!(!targets.is_empty(), "{fmt:?} has no conversion targets");
        }
        // Audio formats
        for fmt in [
            FrameFormat::S16Interleaved,
            FrameFormat::Float32Planar,
            FrameFormat::F32Interleaved,
        ] {
            assert!(matrix.contains_key(&fmt));
        }
    }

    // 9. Empty graph negotiation is a no-op
    #[test]
    fn empty_graph_negotiation() {
        let g = PipelineGraph::new();
        let neg = FormatNegotiator::new();
        let (out, report) = neg.negotiate(g).expect("ok");
        assert_eq!(out.node_count(), 0);
        assert_eq!(report.conversion_count(), 0);
        assert!(report.is_compatible());
    }

    // 10. negotiate_graph convenience wrapper
    #[test]
    fn negotiate_graph_convenience() {
        let (g, _, _) = build_graph_with_formats(FrameFormat::Yuv420p, FrameFormat::Rgba32);
        let neg = FormatNegotiator::new();
        let out = neg.negotiate_graph(g).expect("ok");
        assert!(out.node_count() >= 3);
    }
}
