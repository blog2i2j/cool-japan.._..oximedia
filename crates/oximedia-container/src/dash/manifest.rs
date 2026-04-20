//! MPEG-DASH MPD manifest emitter.
//!
//! Implements a compliant MPEG-DASH MPD (Media Presentation Description) XML
//! emitter targeting the DASH-IF Interoperability Points (IOP) *on-demand*
//! profile with `SegmentTemplate`.
//!
//! # Supported profile
//!
//! - `urn:mpeg:dash:profile:isoff-on-demand:2011`
//!
//! # Example
//!
//! ```
//! use oximedia_container::dash::{DashManifestConfig, DashRepresentation, emit_mpd};
//!
//! let config = DashManifestConfig {
//!     media_presentation_duration: "PT10S".to_string(),
//!     min_buffer_time: "PT2S".to_string(),
//!     base_url: Some("https://cdn.example.com/content/".to_string()),
//!     adaptation_sets: vec![
//!         oximedia_container::dash::DashAdaptationSet {
//!             id: 1,
//!             content_type: "video".to_string(),
//!             mime_type: "video/mp4".to_string(),
//!             codecs: "av01.0.04M.08".to_string(),
//!             representations: vec![
//!                 DashRepresentation {
//!                     id: "video_1080p".to_string(),
//!                     bandwidth: 4_000_000,
//!                     width: Some(1920),
//!                     height: Some(1080),
//!                     frame_rate: Some("30000/1001".to_string()),
//!                     audio_sampling_rate: None,
//!                     segment_template: oximedia_container::dash::DashSegmentTemplate {
//!                         timescale: 90000,
//!                         duration: Some(270000),
//!                         initialization: "video_1080p/init.mp4".to_string(),
//!                         media: "video_1080p/seg$Number$.m4s".to_string(),
//!                         start_number: 1,
//!                         segment_timeline: None,
//!                     },
//!                 },
//!             ],
//!         },
//!     ],
//! };
//!
//! let mpd = emit_mpd(&config);
//! assert!(mpd.contains("urn:mpeg:dash:profile:isoff-on-demand:2011"));
//! ```

use std::fmt::Write as FmtWrite;

// ============================================================================
// Public types
// ============================================================================

/// Configuration for a DASH `SegmentTimeline` S-element.
///
/// Each entry corresponds to an `<S>` element inside a `<SegmentTimeline>`.
#[derive(Clone, Debug)]
pub struct DashSegmentTimelineEntry {
    /// Presentation time (`t` attribute). `None` means "continue from previous".
    pub t: Option<u64>,
    /// Duration of each segment (`d` attribute).
    pub d: u64,
    /// Repeat count (`r` attribute). 0 means the segment is not repeated.
    pub r: u32,
}

/// Optional `SegmentTimeline` block inside a `SegmentTemplate`.
///
/// When present this replaces the `duration` attribute on the parent
/// `SegmentTemplate` and allows irregular segment durations.
#[derive(Clone, Debug)]
pub struct DashSegmentTimeline {
    /// Ordered list of timeline entries.
    pub entries: Vec<DashSegmentTimelineEntry>,
}

/// DASH `SegmentTemplate` description for a single `Representation`.
#[derive(Clone, Debug)]
pub struct DashSegmentTemplate {
    /// Number of time units per second (e.g. `90000` for video, `48000` for
    /// audio).
    pub timescale: u32,
    /// Uniform segment duration in timescale units.  Must be `Some` when
    /// `segment_timeline` is `None`, and is ignored when a `SegmentTimeline`
    /// is provided.
    pub duration: Option<u64>,
    /// URL template for the initialisation segment (e.g.
    /// `"init-$RepresentationID$.mp4"`).
    pub initialization: String,
    /// URL template for media segments.  Must contain at least one of
    /// `$Number$` or `$Time$` (e.g. `"seg-$Number$-$RepresentationID$.m4s"`).
    pub media: String,
    /// First segment number (`startNumber` attribute, default `1`).
    pub start_number: u32,
    /// Optional `SegmentTimeline`.  When `Some`, `duration` is ignored.
    pub segment_timeline: Option<DashSegmentTimeline>,
}

/// A single DASH `Representation`.
///
/// Contains all attributes of the `<Representation>` XML element together
/// with its own `SegmentTemplate`.
#[derive(Clone, Debug)]
pub struct DashRepresentation {
    /// Representation ID string (used in `$RepresentationID$` substitution).
    pub id: String,
    /// Bandwidth in bits per second.
    pub bandwidth: u64,
    /// Display width in pixels (video only).
    pub width: Option<u32>,
    /// Display height in pixels (video only).
    pub height: Option<u32>,
    /// Frame rate string (e.g. `"30000/1001"` or `"25"`). Video only.
    pub frame_rate: Option<String>,
    /// Audio sampling rate in Hz. Audio only.
    pub audio_sampling_rate: Option<u32>,
    /// Per-representation `SegmentTemplate`.
    pub segment_template: DashSegmentTemplate,
}

/// A single DASH `AdaptationSet`.
///
/// Groups representations that share the same codec, content type and MIME
/// type.
#[derive(Clone, Debug)]
pub struct DashAdaptationSet {
    /// Adaptation set ID (integer, used as the `id` XML attribute).
    pub id: u32,
    /// Content type string (e.g. `"video"` or `"audio"`).
    pub content_type: String,
    /// MIME type (e.g. `"video/mp4"` or `"audio/mp4"`).
    pub mime_type: String,
    /// Codec string shared by all representations in this set (e.g.
    /// `"av01.0.04M.08"`).
    pub codecs: String,
    /// List of representations belonging to this adaptation set.
    pub representations: Vec<DashRepresentation>,
}

/// Top-level configuration for a complete DASH MPD manifest.
///
/// Passed to [`emit_mpd`] to produce the XML string.
#[derive(Clone, Debug)]
pub struct DashManifestConfig {
    /// ISO 8601 duration string for the total presentation duration
    /// (e.g. `"PT1M30S"`).
    pub media_presentation_duration: String,
    /// ISO 8601 duration string for the minimum buffer time
    /// (e.g. `"PT2S"`).
    pub min_buffer_time: String,
    /// Optional base URL prepended to all segment URLs.
    pub base_url: Option<String>,
    /// Ordered list of adaptation sets.
    pub adaptation_sets: Vec<DashAdaptationSet>,
}

// ============================================================================
// Emitter
// ============================================================================

/// Emits a complete MPEG-DASH MPD XML document from `config`.
///
/// Uses the DASH-IF IOP on-demand profile:
/// `urn:mpeg:dash:profile:isoff-on-demand:2011`.
///
/// The output is a valid, indented XML string.  No external XML library is
/// used; the document is built with `format!`/`write!` macros so there are no
/// additional dependencies.
#[must_use]
pub fn emit_mpd(config: &DashManifestConfig) -> String {
    let mut out = String::with_capacity(4096);

    // XML declaration
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");

    // MPD root element
    writeln!(
        &mut out,
        "<MPD\n  xmlns=\"urn:mpeg:dash:schema:mpd:2011\"\n  \
         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n  \
         xsi:schemaLocation=\"urn:mpeg:dash:schema:mpd:2011 \
         http://standards.iso.org/ittf/PubliclyAvailableStandards/\
         MPEG-DASH_schema_files/DASH-MPD.xsd\"\n  \
         profiles=\"urn:mpeg:dash:profile:isoff-on-demand:2011\"\n  \
         type=\"static\"\n  \
         mediaPresentationDuration=\"{mpd}\"\n  \
         minBufferTime=\"{mbt}\">",
        mpd = xml_escape(&config.media_presentation_duration),
        mbt = xml_escape(&config.min_buffer_time),
    )
    .unwrap_or_default();

    // Optional BaseURL
    if let Some(ref base) = config.base_url {
        writeln!(&mut out, "  <BaseURL>{}</BaseURL>", xml_escape(base)).unwrap_or_default();
    }

    // Period
    out.push_str("  <Period id=\"1\" start=\"PT0S\">\n");

    for adapt in &config.adaptation_sets {
        emit_adaptation_set(&mut out, adapt);
    }

    out.push_str("  </Period>\n");
    out.push_str("</MPD>\n");
    out
}

// ============================================================================
// Private helpers
// ============================================================================

/// Writes an `<AdaptationSet>` element (and its children) into `out`.
fn emit_adaptation_set(out: &mut String, adapt: &DashAdaptationSet) {
    writeln!(
        out,
        "    <AdaptationSet\n      id=\"{id}\"\n      \
         contentType=\"{ct}\"\n      \
         mimeType=\"{mt}\"\n      \
         codecs=\"{co}\"\n      \
         segmentAlignment=\"true\"\n      \
         subsegmentAlignment=\"true\">",
        id = adapt.id,
        ct = xml_escape(&adapt.content_type),
        mt = xml_escape(&adapt.mime_type),
        co = xml_escape(&adapt.codecs),
    )
    .unwrap_or_default();

    for repr in &adapt.representations {
        emit_representation(out, repr);
    }

    out.push_str("    </AdaptationSet>\n");
}

/// Writes a `<Representation>` element (and its `SegmentTemplate`) into `out`.
fn emit_representation(out: &mut String, repr: &DashRepresentation) {
    let mut attrs = format!(
        "      id=\"{}\" bandwidth=\"{}\"",
        xml_escape(&repr.id),
        repr.bandwidth
    );

    if let Some(w) = repr.width {
        attrs.push_str(&format!(" width=\"{w}\""));
    }
    if let Some(h) = repr.height {
        attrs.push_str(&format!(" height=\"{h}\""));
    }
    if let Some(ref fr) = repr.frame_rate {
        attrs.push_str(&format!(" frameRate=\"{}\"", xml_escape(fr)));
    }
    if let Some(asr) = repr.audio_sampling_rate {
        attrs.push_str(&format!(" audioSamplingRate=\"{asr}\""));
    }

    writeln!(out, "      <Representation\n{attrs}>").unwrap_or_default();

    emit_segment_template(out, &repr.segment_template);

    out.push_str("      </Representation>\n");
}

/// Writes a `<SegmentTemplate>` element into `out`.
fn emit_segment_template(out: &mut String, tmpl: &DashSegmentTemplate) {
    let mut attrs = format!(
        "        timescale=\"{}\" startNumber=\"{}\"",
        tmpl.timescale, tmpl.start_number,
    );

    // duration is omitted when a SegmentTimeline is present
    if tmpl.segment_timeline.is_none() {
        if let Some(dur) = tmpl.duration {
            attrs.push_str(&format!(" duration=\"{dur}\""));
        }
    }

    attrs.push_str(&format!(
        " initialization=\"{}\" media=\"{}\"",
        xml_escape(&tmpl.initialization),
        xml_escape(&tmpl.media),
    ));

    if let Some(ref timeline) = tmpl.segment_timeline {
        writeln!(out, "        <SegmentTemplate\n{attrs}>").unwrap_or_default();
        out.push_str("          <SegmentTimeline>\n");
        for entry in &timeline.entries {
            let mut s_attrs = format!(" d=\"{}\"", entry.d);
            if let Some(t) = entry.t {
                s_attrs = format!(" t=\"{t}\"{s_attrs}");
            }
            if entry.r > 0 {
                s_attrs.push_str(&format!(" r=\"{}\"", entry.r));
            }
            writeln!(out, "            <S{s_attrs}/>").unwrap_or_default();
        }
        out.push_str("          </SegmentTimeline>\n");
        out.push_str("        </SegmentTemplate>\n");
    } else {
        writeln!(out, "        <SegmentTemplate\n{attrs}/>").unwrap_or_default();
    }
}

/// Escapes XML special characters in attribute values and text content.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            other => out.push(other),
        }
    }
    out
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_minimal_config() -> DashManifestConfig {
        DashManifestConfig {
            media_presentation_duration: "PT10S".to_string(),
            min_buffer_time: "PT2S".to_string(),
            base_url: None,
            adaptation_sets: vec![DashAdaptationSet {
                id: 1,
                content_type: "video".to_string(),
                mime_type: "video/mp4".to_string(),
                codecs: "av01.0.04M.08".to_string(),
                representations: vec![DashRepresentation {
                    id: "v1".to_string(),
                    bandwidth: 1_000_000,
                    width: Some(1280),
                    height: Some(720),
                    frame_rate: Some("25".to_string()),
                    audio_sampling_rate: None,
                    segment_template: DashSegmentTemplate {
                        timescale: 90000,
                        duration: Some(270000),
                        initialization: "v1/init.mp4".to_string(),
                        media: "v1/seg$Number$.m4s".to_string(),
                        start_number: 1,
                        segment_timeline: None,
                    },
                }],
            }],
        }
    }

    #[test]
    fn test_emit_mpd_contains_profile() {
        let config = make_minimal_config();
        let mpd = emit_mpd(&config);
        assert!(
            mpd.contains("urn:mpeg:dash:profile:isoff-on-demand:2011"),
            "MPD should contain the DASH on-demand profile URI"
        );
    }

    #[test]
    fn test_emit_mpd_contains_segment_template() {
        let config = make_minimal_config();
        let mpd = emit_mpd(&config);
        assert!(
            mpd.contains("SegmentTemplate"),
            "MPD should contain a SegmentTemplate element"
        );
        assert!(
            mpd.contains("$Number$"),
            "Media template should contain $Number$ placeholder"
        );
    }

    #[test]
    fn test_emit_mpd_representation_ids() {
        let config = DashManifestConfig {
            media_presentation_duration: "PT30S".to_string(),
            min_buffer_time: "PT4S".to_string(),
            base_url: None,
            adaptation_sets: vec![DashAdaptationSet {
                id: 1,
                content_type: "video".to_string(),
                mime_type: "video/mp4".to_string(),
                codecs: "av01.0.08M.10".to_string(),
                representations: vec![
                    DashRepresentation {
                        id: "video_1080p".to_string(),
                        bandwidth: 4_000_000,
                        width: Some(1920),
                        height: Some(1080),
                        frame_rate: Some("30000/1001".to_string()),
                        audio_sampling_rate: None,
                        segment_template: DashSegmentTemplate {
                            timescale: 90000,
                            duration: Some(270000),
                            initialization: "1080p/init.mp4".to_string(),
                            media: "1080p/$Number$.m4s".to_string(),
                            start_number: 1,
                            segment_timeline: None,
                        },
                    },
                    DashRepresentation {
                        id: "video_720p".to_string(),
                        bandwidth: 2_000_000,
                        width: Some(1280),
                        height: Some(720),
                        frame_rate: Some("30000/1001".to_string()),
                        audio_sampling_rate: None,
                        segment_template: DashSegmentTemplate {
                            timescale: 90000,
                            duration: Some(270000),
                            initialization: "720p/init.mp4".to_string(),
                            media: "720p/$Number$.m4s".to_string(),
                            start_number: 1,
                            segment_timeline: None,
                        },
                    },
                ],
            }],
        };
        let mpd = emit_mpd(&config);
        assert!(
            mpd.contains("video_1080p"),
            "MPD should contain 1080p representation ID"
        );
        assert!(
            mpd.contains("video_720p"),
            "MPD should contain 720p representation ID"
        );
    }

    #[test]
    fn test_emit_mpd_with_base_url() {
        let mut config = make_minimal_config();
        config.base_url = Some("https://cdn.example.com/".to_string());
        let mpd = emit_mpd(&config);
        assert!(mpd.contains("<BaseURL>https://cdn.example.com/</BaseURL>"));
    }

    #[test]
    fn test_emit_mpd_audio_representation() {
        let config = DashManifestConfig {
            media_presentation_duration: "PT10S".to_string(),
            min_buffer_time: "PT2S".to_string(),
            base_url: None,
            adaptation_sets: vec![DashAdaptationSet {
                id: 2,
                content_type: "audio".to_string(),
                mime_type: "audio/mp4".to_string(),
                codecs: "opus".to_string(),
                representations: vec![DashRepresentation {
                    id: "audio_opus_48k".to_string(),
                    bandwidth: 128_000,
                    width: None,
                    height: None,
                    frame_rate: None,
                    audio_sampling_rate: Some(48000),
                    segment_template: DashSegmentTemplate {
                        timescale: 48000,
                        duration: Some(96000),
                        initialization: "audio/init.mp4".to_string(),
                        media: "audio/$Number$.m4s".to_string(),
                        start_number: 1,
                        segment_timeline: None,
                    },
                }],
            }],
        };
        let mpd = emit_mpd(&config);
        assert!(mpd.contains("audio_opus_48k"));
        assert!(mpd.contains("audioSamplingRate=\"48000\""));
        assert!(mpd.contains("contentType=\"audio\""));
    }

    #[test]
    fn test_emit_mpd_segment_timeline() {
        let config = DashManifestConfig {
            media_presentation_duration: "PT10S".to_string(),
            min_buffer_time: "PT2S".to_string(),
            base_url: None,
            adaptation_sets: vec![DashAdaptationSet {
                id: 1,
                content_type: "video".to_string(),
                mime_type: "video/mp4".to_string(),
                codecs: "av01.0.04M.08".to_string(),
                representations: vec![DashRepresentation {
                    id: "v1".to_string(),
                    bandwidth: 1_000_000,
                    width: Some(1280),
                    height: Some(720),
                    frame_rate: None,
                    audio_sampling_rate: None,
                    segment_template: DashSegmentTemplate {
                        timescale: 90000,
                        duration: None,
                        initialization: "v1/init.mp4".to_string(),
                        media: "v1/seg$Time$.m4s".to_string(),
                        start_number: 1,
                        segment_timeline: Some(DashSegmentTimeline {
                            entries: vec![
                                DashSegmentTimelineEntry {
                                    t: Some(0),
                                    d: 270000,
                                    r: 2,
                                },
                                DashSegmentTimelineEntry {
                                    t: None,
                                    d: 180000,
                                    r: 0,
                                },
                            ],
                        }),
                    },
                }],
            }],
        };
        let mpd = emit_mpd(&config);
        assert!(mpd.contains("SegmentTimeline"));
        assert!(mpd.contains("<S"));
        assert!(mpd.contains("d=\"270000\""));
    }

    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("a&b"), "a&amp;b");
        assert_eq!(xml_escape("<tag>"), "&lt;tag&gt;");
        assert_eq!(xml_escape("\"hello\""), "&quot;hello&quot;");
        assert_eq!(xml_escape("it's"), "it&apos;s");
        assert_eq!(xml_escape("normal"), "normal");
    }
}
