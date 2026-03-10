//! Export timeline to various formats.

use std::io::Write as IoWrite;
use std::path::Path;

use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;

use crate::clip::MediaSource;
use crate::error::{TimelineError, TimelineResult};
use crate::timeline::Timeline;

/// Export format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportFormat {
    /// EDL (Edit Decision List) - CMX 3600.
    Edl,
    /// FCPXML (Final Cut Pro XML).
    Fcpxml,
    /// Adobe Premiere XML.
    PremiereXml,
    /// AAF (Advanced Authoring Format).
    Aaf,
    /// `DaVinci` Resolve XML.
    ResolveXml,
    /// JSON (`OxiMedia` native format).
    Json,
}

impl ExportFormat {
    /// Gets file extension for format.
    #[must_use]
    pub const fn file_extension(self) -> &'static str {
        match self {
            Self::Edl => "edl",
            Self::Fcpxml => "fcpxml",
            Self::PremiereXml => "xml",
            Self::Aaf => "aaf",
            Self::ResolveXml => "xml",
            Self::Json => "json",
        }
    }
}

/// Export options.
#[derive(Clone, Debug)]
pub struct ExportOptions {
    /// Whether to export audio tracks.
    pub export_audio: bool,
    /// Whether to export video tracks.
    pub export_video: bool,
    /// Whether to export markers.
    pub export_markers: bool,
    /// Whether to export effects.
    pub export_effects: bool,
    /// Whether to export transitions.
    pub export_transitions: bool,
    /// Whether to flatten nested sequences.
    pub flatten_nested: bool,
    /// Whether to export with relative paths.
    pub use_relative_paths: bool,
    /// Start frame number (for EDL).
    pub start_frame: Option<i64>,
}

impl ExportOptions {
    /// Creates default export options (export everything).
    #[must_use]
    pub fn default_all() -> Self {
        Self {
            export_audio: true,
            export_video: true,
            export_markers: true,
            export_effects: true,
            export_transitions: true,
            flatten_nested: false,
            use_relative_paths: false,
            start_frame: None,
        }
    }

    /// Creates options for EDL export.
    #[must_use]
    pub fn for_edl() -> Self {
        Self {
            export_audio: false, // EDL typically video only
            export_video: true,
            export_markers: true,
            export_effects: false, // EDL doesn't support effects
            export_transitions: true,
            flatten_nested: true,
            use_relative_paths: false,
            start_frame: Some(0),
        }
    }

    /// Creates options for XML export.
    #[must_use]
    pub fn for_xml() -> Self {
        Self {
            export_audio: true,
            export_video: true,
            export_markers: true,
            export_effects: true,
            export_transitions: true,
            flatten_nested: false,
            use_relative_paths: false,
            start_frame: None,
        }
    }
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self::default_all()
    }
}

/// Export statistics.
#[derive(Clone, Debug, Default)]
pub struct ExportStats {
    /// Number of video tracks exported.
    pub video_tracks: usize,
    /// Number of audio tracks exported.
    pub audio_tracks: usize,
    /// Number of clips exported.
    pub clips: usize,
    /// Number of transitions exported.
    pub transitions: usize,
    /// Number of effects exported.
    pub effects: usize,
    /// Number of markers exported.
    pub markers: usize,
    /// File size in bytes.
    pub file_size: usize,
}

/// Timeline exporter.
#[allow(dead_code)]
pub struct TimelineExporter {
    options: ExportOptions,
}

// ---------------------------------------------------------------------------
// FCPXML generation helpers
// ---------------------------------------------------------------------------

/// Format a number of frames as an FCPXML rational-time string "num/denom s".
/// For example 24 frames at 24fps → "2400/2400s" (1 second expressed with
/// the standard 2400 sub-frame denominator).
fn frames_to_fcpxml_time(frames: i64, fps_num: i64, fps_den: i64) -> String {
    // FCPXML convention: time = frames * (fps_den / fps_num) seconds
    // expressed as a rational "N/Ds".
    // We use 2400 as the common denominator sub-frame scale (industry standard).
    // time_in_seconds = frames * fps_den / fps_num
    // With sub-frame scale S = 2400:
    //   num = frames * fps_den * S / fps_num
    //   den = S
    let scale: i64 = 2400;
    let num = frames * fps_den * scale / fps_num;
    format!("{num}/{scale}s")
}

/// Get the file path string from a `MediaSource`, or an empty string for
/// generated sources.
fn media_source_to_url(source: &MediaSource) -> String {
    match source {
        MediaSource::File { path, .. } => {
            format!("file://{}", path.display())
        }
        _ => String::new(),
    }
}

/// Build the full FCPXML string for the given timeline.
fn build_fcpxml(timeline: &Timeline, stats: &mut ExportStats) -> TimelineResult<String> {
    let mut buf: Vec<u8> = Vec::new();
    let mut w = Writer::new_with_indent(&mut buf, b' ', 2);

    // XML declaration
    w.write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;

    // Derive fps_num / fps_den from frame_rate (stored as Rational num/den
    // where the rational represents fps, e.g. num=24, den=1).
    let fps_num = timeline.frame_rate.num;
    let fps_den = timeline.frame_rate.den;

    // <fcpxml version="1.9">
    let mut root = BytesStart::new("fcpxml");
    root.push_attribute(("version", "1.9"));
    w.write_event(Event::Start(root))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;

    // ---- <resources> -------------------------------------------------------
    w.write_event(Event::Start(BytesStart::new("resources")))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;

    // Write <format> element describing the sequence frame rate.
    {
        // frameDuration is time-of-one-frame: fps_den / fps_num seconds
        // expressed as "fps_den*scale / (fps_num*scale) s" simplified.
        // Standard: frameDuration = "100/2400s" for 24fps.
        let fd_num: i64 = 2400 / fps_num.max(1) * fps_den;
        let fd_den: i64 = 2400;
        let frame_duration = format!("{fd_num}/{fd_den}s");

        let mut fmt = BytesStart::new("format");
        fmt.push_attribute(("id", "r1"));
        fmt.push_attribute(("frameDuration", frame_duration.as_str()));
        fmt.push_attribute((
            "width",
            timeline.video_tracks.first().map_or("1920", |_| "1920"),
        ));
        fmt.push_attribute((
            "height",
            timeline.video_tracks.first().map_or("1080", |_| "1080"),
        ));
        w.write_event(Event::Empty(fmt))
            .map_err(|e| TimelineError::ExportError(e.to_string()))?;
    }

    // Write <asset> elements for every file-based clip.
    let mut asset_id = 2_usize;
    // We collect (clip_id → asset_ref_id) for use in the spine.
    let mut clip_to_asset: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    for track in &timeline.video_tracks {
        for clip in &track.clips {
            let url = media_source_to_url(&clip.source);
            if url.is_empty() {
                continue;
            }
            let aid = format!("r{asset_id}");
            asset_id += 1;
            clip_to_asset.insert(clip.id.to_string(), aid.clone());

            let clip_dur = frames_to_fcpxml_time(clip.source_duration().value(), fps_num, fps_den);
            let mut asset = BytesStart::new("asset");
            asset.push_attribute(("id", aid.as_str()));
            asset.push_attribute(("name", clip.name.as_str()));
            asset.push_attribute(("src", url.as_str()));
            asset.push_attribute(("duration", clip_dur.as_str()));
            w.write_event(Event::Empty(asset))
                .map_err(|e| TimelineError::ExportError(e.to_string()))?;
        }
    }

    w.write_event(Event::End(BytesEnd::new("resources")))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;

    // ---- <library> → <event> → <project> → <sequence> → <spine> -----------
    w.write_event(Event::Start(BytesStart::new("library")))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;

    let mut event_elem = BytesStart::new("event");
    event_elem.push_attribute(("name", timeline.name.as_str()));
    w.write_event(Event::Start(event_elem))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;

    let mut project = BytesStart::new("project");
    project.push_attribute(("name", timeline.name.as_str()));
    w.write_event(Event::Start(project))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;

    // Sequence duration = timeline duration.
    let seq_dur = frames_to_fcpxml_time(timeline.duration.value(), fps_num, fps_den);
    let mut sequence = BytesStart::new("sequence");
    sequence.push_attribute(("duration", seq_dur.as_str()));
    sequence.push_attribute(("tcFormat", "NDF"));
    sequence.push_attribute(("tcStart", "0s"));
    sequence.push_attribute(("format", "r1"));
    w.write_event(Event::Start(sequence))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;

    w.write_event(Event::Start(BytesStart::new("spine")))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;

    // Write clips from each video track into the spine.
    for track in &timeline.video_tracks {
        stats.video_tracks += 1;
        for clip in &track.clips {
            let asset_ref = clip_to_asset
                .get(&clip.id.to_string())
                .map_or("r2", std::string::String::as_str);

            let offset = frames_to_fcpxml_time(clip.timeline_in.value(), fps_num, fps_den);
            let dur = frames_to_fcpxml_time(clip.source_duration().value(), fps_num, fps_den);
            let start = frames_to_fcpxml_time(clip.source_in.value(), fps_num, fps_den);

            let mut ac = BytesStart::new("asset-clip");
            ac.push_attribute(("ref", asset_ref));
            ac.push_attribute(("name", clip.name.as_str()));
            ac.push_attribute(("offset", offset.as_str()));
            ac.push_attribute(("duration", dur.as_str()));
            ac.push_attribute(("start", start.as_str()));
            w.write_event(Event::Empty(ac))
                .map_err(|e| TimelineError::ExportError(e.to_string()))?;

            stats.clips += 1;
        }
    }

    w.write_event(Event::End(BytesEnd::new("spine")))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;
    w.write_event(Event::End(BytesEnd::new("sequence")))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;
    w.write_event(Event::End(BytesEnd::new("project")))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;
    w.write_event(Event::End(BytesEnd::new("event")))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;
    w.write_event(Event::End(BytesEnd::new("library")))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;
    w.write_event(Event::End(BytesEnd::new("fcpxml")))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;

    String::from_utf8(buf).map_err(|e| TimelineError::ExportError(e.to_string()))
}

// ---------------------------------------------------------------------------
// xmeml (Premiere / Resolve) generation helpers
// ---------------------------------------------------------------------------

/// Write a simple XML element containing a text value.
fn write_text_element<W: IoWrite>(w: &mut Writer<W>, tag: &str, value: &str) -> TimelineResult<()> {
    w.write_event(Event::Start(BytesStart::new(tag)))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;
    w.write_event(Event::Text(BytesText::new(value)))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;
    w.write_event(Event::End(BytesEnd::new(tag)))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;
    Ok(())
}

/// Build an xmeml XML string for the given timeline.
/// `version` should be `"4"` for Premiere and `"5"` for Resolve.
fn build_xmeml(
    timeline: &Timeline,
    version: &str,
    stats: &mut ExportStats,
    options: &ExportOptions,
) -> TimelineResult<String> {
    let mut buf: Vec<u8> = Vec::new();
    let mut w = Writer::new_with_indent(&mut buf, b' ', 2);

    w.write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;

    let fps_num = timeline.frame_rate.num;
    let fps_den = timeline.frame_rate.den;
    // xmeml uses integer timebase (fps as integer).
    let timebase = (fps_num / fps_den.max(1)).to_string();

    // <xmeml version="4">
    let mut root = BytesStart::new("xmeml");
    root.push_attribute(("version", version));
    w.write_event(Event::Start(root))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;

    w.write_event(Event::Start(BytesStart::new("sequence")))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;

    write_text_element(&mut w, "name", &timeline.name)?;
    write_text_element(&mut w, "timebase", &timebase)?;

    // <media>
    w.write_event(Event::Start(BytesStart::new("media")))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;

    // ------ <video> -------
    if options.export_video {
        w.write_event(Event::Start(BytesStart::new("video")))
            .map_err(|e| TimelineError::ExportError(e.to_string()))?;

        for track in &timeline.video_tracks {
            stats.video_tracks += 1;
            w.write_event(Event::Start(BytesStart::new("track")))
                .map_err(|e| TimelineError::ExportError(e.to_string()))?;

            for clip in &track.clips {
                w.write_event(Event::Start(BytesStart::new("clipitem")))
                    .map_err(|e| TimelineError::ExportError(e.to_string()))?;

                write_text_element(&mut w, "name", &clip.name)?;

                // <file> with <pathurl>
                let url = media_source_to_url(&clip.source);
                if !url.is_empty() {
                    w.write_event(Event::Start(BytesStart::new("file")))
                        .map_err(|e| TimelineError::ExportError(e.to_string()))?;
                    write_text_element(&mut w, "pathurl", &url)?;
                    w.write_event(Event::End(BytesEnd::new("file")))
                        .map_err(|e| TimelineError::ExportError(e.to_string()))?;
                }

                write_text_element(&mut w, "in", &clip.source_in.value().to_string())?;
                write_text_element(&mut w, "out", &clip.source_out.value().to_string())?;
                write_text_element(&mut w, "start", &clip.timeline_in.value().to_string())?;
                write_text_element(&mut w, "end", &clip.timeline_out().value().to_string())?;

                w.write_event(Event::End(BytesEnd::new("clipitem")))
                    .map_err(|e| TimelineError::ExportError(e.to_string()))?;

                stats.clips += 1;
            }

            w.write_event(Event::End(BytesEnd::new("track")))
                .map_err(|e| TimelineError::ExportError(e.to_string()))?;
        }

        w.write_event(Event::End(BytesEnd::new("video")))
            .map_err(|e| TimelineError::ExportError(e.to_string()))?;
    }

    // ------ <audio> -------
    if options.export_audio {
        w.write_event(Event::Start(BytesStart::new("audio")))
            .map_err(|e| TimelineError::ExportError(e.to_string()))?;

        for track in &timeline.audio_tracks {
            stats.audio_tracks += 1;
            w.write_event(Event::Start(BytesStart::new("track")))
                .map_err(|e| TimelineError::ExportError(e.to_string()))?;

            for clip in &track.clips {
                w.write_event(Event::Start(BytesStart::new("clipitem")))
                    .map_err(|e| TimelineError::ExportError(e.to_string()))?;

                write_text_element(&mut w, "name", &clip.name)?;

                let url = media_source_to_url(&clip.source);
                if !url.is_empty() {
                    w.write_event(Event::Start(BytesStart::new("file")))
                        .map_err(|e| TimelineError::ExportError(e.to_string()))?;
                    write_text_element(&mut w, "pathurl", &url)?;
                    w.write_event(Event::End(BytesEnd::new("file")))
                        .map_err(|e| TimelineError::ExportError(e.to_string()))?;
                }

                write_text_element(&mut w, "in", &clip.source_in.value().to_string())?;
                write_text_element(&mut w, "out", &clip.source_out.value().to_string())?;
                write_text_element(&mut w, "start", &clip.timeline_in.value().to_string())?;
                write_text_element(&mut w, "end", &clip.timeline_out().value().to_string())?;

                w.write_event(Event::End(BytesEnd::new("clipitem")))
                    .map_err(|e| TimelineError::ExportError(e.to_string()))?;

                stats.clips += 1;
            }

            w.write_event(Event::End(BytesEnd::new("track")))
                .map_err(|e| TimelineError::ExportError(e.to_string()))?;
        }

        w.write_event(Event::End(BytesEnd::new("audio")))
            .map_err(|e| TimelineError::ExportError(e.to_string()))?;
    }

    w.write_event(Event::End(BytesEnd::new("media")))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;
    w.write_event(Event::End(BytesEnd::new("sequence")))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;
    w.write_event(Event::End(BytesEnd::new("xmeml")))
        .map_err(|e| TimelineError::ExportError(e.to_string()))?;

    String::from_utf8(buf).map_err(|e| TimelineError::ExportError(e.to_string()))
}

// ---------------------------------------------------------------------------
// TimelineExporter implementation
// ---------------------------------------------------------------------------

impl TimelineExporter {
    /// Creates a new exporter with options.
    #[must_use]
    pub fn new(options: ExportOptions) -> Self {
        Self { options }
    }

    /// Creates a new exporter with default options.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ExportOptions::default())
    }

    /// Exports a timeline to a file.
    ///
    /// # Errors
    ///
    /// Returns error if export fails.
    pub fn export_file(
        &self,
        timeline: &Timeline,
        path: &Path,
        format: ExportFormat,
    ) -> TimelineResult<ExportStats> {
        match format {
            ExportFormat::Edl => self.export_edl(timeline, path),
            ExportFormat::Fcpxml => self.export_fcpxml(timeline, path),
            ExportFormat::PremiereXml => self.export_premiere_xml(timeline, path),
            ExportFormat::Aaf => self.export_aaf(timeline, path),
            ExportFormat::ResolveXml => self.export_resolve_xml(timeline, path),
            ExportFormat::Json => self.export_json(timeline, path),
        }
    }

    /// Exports to EDL format (CMX 3600) using the oximedia-edl crate.
    ///
    /// Only video clips are exported (EDL is primarily a video edit format).
    /// Each clip becomes one EDL event with:
    ///   - source timecodes from `clip.source_in` / `clip.source_out`
    ///   - record timecodes from `clip.timeline_in` / `clip.timeline_out()`
    fn export_edl(&self, timeline: &Timeline, path: &Path) -> TimelineResult<ExportStats> {
        use oximedia_edl::event::{EditType, EdlEvent, TrackType as EdlTrackType};
        use oximedia_edl::timecode::{EdlFrameRate, EdlTimecode};
        use oximedia_edl::{Edl, EdlFormat, EdlGenerator};

        // Choose the closest EDL frame rate.
        let fps_num = timeline.frame_rate.num;
        let fps_den = timeline.frame_rate.den;
        let fps_approx = if fps_den == 0 {
            fps_num
        } else {
            fps_num / fps_den
        };

        let edl_fps = match fps_approx {
            24 => EdlFrameRate::Fps24,
            25 => EdlFrameRate::Fps25,
            30 => EdlFrameRate::Fps30,
            50 => EdlFrameRate::Fps50,
            60 => EdlFrameRate::Fps60,
            _ => EdlFrameRate::Fps25,
        };

        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_title(timeline.name.clone());
        edl.set_frame_rate(edl_fps);

        let mut stats = ExportStats::default();
        let mut event_number: u32 = 1;

        /// Helper: convert a frame count to an `EdlTimecode`.
        fn frames_to_tc(frames: i64, rate: EdlFrameRate) -> EdlTimecode {
            let f = frames.max(0) as u64;
            EdlTimecode::from_frames(f, rate).unwrap_or_else(|_| {
                EdlTimecode::new(0, 0, 0, 0, rate).expect("hardcoded value is valid")
            })
        }

        for track in &timeline.video_tracks {
            stats.video_tracks += 1;
            for clip in &track.clips {
                if !clip.enabled {
                    continue;
                }

                let src_in = frames_to_tc(clip.source_in.value(), edl_fps);
                let src_out = frames_to_tc(clip.source_out.value(), edl_fps);
                let rec_in = frames_to_tc(clip.timeline_in.value(), edl_fps);
                let rec_out = frames_to_tc(clip.timeline_out().value(), edl_fps);

                // Reel name: derive from clip name (truncated to 8 chars, uppercased).
                let reel: String = clip
                    .name
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .take(8)
                    .collect::<String>()
                    .to_uppercase();
                let reel = if reel.is_empty() {
                    format!("R{event_number:04}")
                } else {
                    reel
                };

                let mut event = EdlEvent::new(
                    event_number,
                    reel,
                    EdlTrackType::Video,
                    EditType::Cut,
                    src_in,
                    src_out,
                    rec_in,
                    rec_out,
                );
                event.set_clip_name(clip.name.clone());

                edl.add_event(event)
                    .map_err(|e| TimelineError::ExportError(format!("EDL event error: {e}")))?;

                stats.clips += 1;
                event_number += 1;
            }
        }

        let generator = EdlGenerator::new();
        let content = generator
            .generate(&edl)
            .map_err(|e| TimelineError::ExportError(format!("EDL generate error: {e}")))?;

        std::fs::write(path, &content)?;
        stats.file_size = content.len();
        stats.markers = timeline.markers.markers().len();

        Ok(stats)
    }

    /// Exports to FCPXML format.
    ///
    /// Generates Final Cut Pro XML (version 1.9) from the timeline.
    fn export_fcpxml(&self, timeline: &Timeline, path: &Path) -> TimelineResult<ExportStats> {
        let mut stats = ExportStats::default();
        let xml = build_fcpxml(timeline, &mut stats)?;
        std::fs::write(path, &xml)?;
        stats.file_size = xml.len();
        stats.markers = timeline.markers.markers().len();
        Ok(stats)
    }

    /// Exports to Premiere XML format (xmeml v4).
    fn export_premiere_xml(&self, timeline: &Timeline, path: &Path) -> TimelineResult<ExportStats> {
        let mut stats = ExportStats::default();
        let xml = build_xmeml(timeline, "4", &mut stats, &self.options)?;
        std::fs::write(path, &xml)?;
        stats.file_size = xml.len();
        stats.markers = timeline.markers.markers().len();
        Ok(stats)
    }

    /// Exports to AAF format using the `oximedia-aaf` crate.
    ///
    /// Builds a minimal SMPTE ST 377-1 compliant AAF container from the
    /// [`Timeline`].  Each enabled video clip is added as a source-clip
    /// reference inside a composition mob track; audio tracks are
    /// added as separate tracks when [`ExportOptions::export_audio`] is set.
    ///
    /// The output file is written via [`oximedia_aaf::writer::AafBuilder`]
    /// which serialises the object graph into a Microsoft Structured Storage
    /// byte stream.
    fn export_aaf(&self, timeline: &Timeline, path: &Path) -> TimelineResult<ExportStats> {
        use oximedia_aaf::composition::{
            CompositionMob, Sequence, SequenceComponent, SourceClip, Track, TrackType, UsageCode,
        };
        use oximedia_aaf::dictionary::Auid;
        use oximedia_aaf::timeline::{EditRate, Position as AafPosition};
        use oximedia_aaf::writer::AafBuilder;

        let fps_num = timeline.frame_rate.num as i32;
        let fps_den = (timeline.frame_rate.den.max(1)) as i32;
        let edit_rate = EditRate::new(fps_num, fps_den);

        let mut comp_mob = CompositionMob::new(uuid::Uuid::new_v4(), &timeline.name);
        comp_mob.usage_code = Some(UsageCode::TopLevel);

        let mut stats = ExportStats::default();
        let mut slot_id: u32 = 1;

        // ---- Video tracks ------------------------------------------------
        if self.options.export_video {
            for track in &timeline.video_tracks {
                stats.video_tracks += 1;
                let mut sequence = Sequence::new(Auid::PICTURE);

                for clip in &track.clips {
                    if !clip.enabled {
                        continue;
                    }
                    let src_in = clip.source_in.value().max(0);
                    let length = clip.source_duration().value().max(0);

                    // Each clip references its own source mob (created on demand).
                    // Use the clip's UUID as the source mob ID.
                    let source_clip = SourceClip::new(
                        length,
                        AafPosition::new(src_in),
                        *clip.id.as_uuid(),
                        1, // source mob slot 1
                    );
                    sequence.add_component(SequenceComponent::SourceClip(source_clip));
                    stats.clips += 1;
                }

                // Derive track name from slot index.
                let track_name = format!("V{slot_id}");
                let mut aaf_track = Track::new(slot_id, track_name, edit_rate, TrackType::Picture);
                aaf_track.set_sequence(sequence);
                comp_mob.add_track(aaf_track);
                slot_id += 1;
            }
        }

        // ---- Audio tracks ------------------------------------------------
        if self.options.export_audio {
            for track in &timeline.audio_tracks {
                stats.audio_tracks += 1;
                let mut sequence = Sequence::new(Auid::SOUND);

                for clip in &track.clips {
                    if !clip.enabled {
                        continue;
                    }
                    let src_in = clip.source_in.value().max(0);
                    let length = clip.source_duration().value().max(0);

                    let source_clip =
                        SourceClip::new(length, AafPosition::new(src_in), *clip.id.as_uuid(), 1);
                    sequence.add_component(SequenceComponent::SourceClip(source_clip));
                    stats.clips += 1;
                }

                let track_name = format!("A{slot_id}");
                let mut aaf_track = Track::new(slot_id, track_name, edit_rate, TrackType::Sound);
                aaf_track.set_sequence(sequence);
                comp_mob.add_track(aaf_track);
                slot_id += 1;
            }
        }

        AafBuilder::new()
            .add_composition_mob(comp_mob)
            .write_to_file(path)
            .map_err(|e| TimelineError::ExportError(format!("AAF write error: {e}")))?;

        let file_size = std::fs::metadata(path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        stats.file_size = file_size;
        stats.markers = timeline.markers.markers().len();

        Ok(stats)
    }

    /// Exports to `DaVinci` Resolve XML format (xmeml v5).
    fn export_resolve_xml(&self, timeline: &Timeline, path: &Path) -> TimelineResult<ExportStats> {
        let mut stats = ExportStats::default();
        let xml = build_xmeml(timeline, "5", &mut stats, &self.options)?;
        std::fs::write(path, &xml)?;
        stats.file_size = xml.len();
        stats.markers = timeline.markers.markers().len();
        Ok(stats)
    }

    /// Exports to JSON format.
    fn export_json(&self, timeline: &Timeline, path: &Path) -> TimelineResult<ExportStats> {
        let json = serde_json::to_string_pretty(timeline)?;
        std::fs::write(path, &json)?;

        // Count all effects across every clip on every track, plus track-level effects.
        let effect_count: usize = timeline
            .video_tracks
            .iter()
            .chain(&timeline.audio_tracks)
            .chain(&timeline.subtitle_tracks)
            .map(|track| {
                // Track-level effects.
                let track_effects = track.effects.effects().len();
                // Clip-level effects.
                let clip_effects: usize =
                    track.clips.iter().map(|c| c.effects.effects().len()).sum();
                track_effects + clip_effects
            })
            .sum();

        Ok(ExportStats {
            video_tracks: timeline.video_tracks.len(),
            audio_tracks: timeline.audio_tracks.len(),
            clips: timeline.clip_count(),
            transitions: timeline.transitions.len(),
            effects: effect_count,
            markers: timeline.markers.markers().len(),
            file_size: json.len(),
        })
    }

    /// Exports to string (for in-memory export).
    ///
    /// # Errors
    ///
    /// Returns error if export fails.
    pub fn export_to_string(
        &self,
        timeline: &Timeline,
        format: ExportFormat,
    ) -> TimelineResult<String> {
        match format {
            ExportFormat::Json => Ok(serde_json::to_string_pretty(timeline)?),
            ExportFormat::Fcpxml => {
                let mut stats = ExportStats::default();
                build_fcpxml(timeline, &mut stats)
            }
            ExportFormat::PremiereXml => {
                let mut stats = ExportStats::default();
                build_xmeml(timeline, "4", &mut stats, &self.options)
            }
            ExportFormat::ResolveXml => {
                let mut stats = ExportStats::default();
                build_xmeml(timeline, "5", &mut stats, &self.options)
            }
            _ => Err(TimelineError::ExportError(format!(
                "String export not supported for {format:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Clip, MediaSource};
    use crate::types::Position;
    use oximedia_core::Rational;
    use std::path::PathBuf;

    fn create_test_timeline() -> Timeline {
        Timeline::new("Test Timeline", Rational::new(24, 1), 48000).expect("should succeed in test")
    }

    fn create_timeline_with_clips() -> Timeline {
        let mut tl = Timeline::new("My Project", Rational::new(24, 1), 48000)
            .expect("should succeed in test");
        let vid_id = tl.add_video_track("V1").expect("should succeed in test");
        let clip = Clip::new(
            "Shot 01".to_string(),
            MediaSource::file(PathBuf::from("/media/shot01.mov")),
            Position::new(0),
            Position::new(48),
            Position::new(0),
        )
        .expect("should succeed in test");
        tl.add_clip(vid_id, clip).expect("should succeed in test");
        tl
    }

    #[test]
    fn test_export_format_extension() {
        assert_eq!(ExportFormat::Edl.file_extension(), "edl");
        assert_eq!(ExportFormat::Fcpxml.file_extension(), "fcpxml");
        assert_eq!(ExportFormat::Aaf.file_extension(), "aaf");
        assert_eq!(ExportFormat::Json.file_extension(), "json");
    }

    #[test]
    fn test_export_options_default() {
        let opts = ExportOptions::default();
        assert!(opts.export_audio);
        assert!(opts.export_video);
        assert!(opts.export_markers);
    }

    #[test]
    fn test_export_options_for_edl() {
        let opts = ExportOptions::for_edl();
        assert!(!opts.export_audio);
        assert!(opts.export_video);
        assert!(opts.flatten_nested);
    }

    #[test]
    fn test_export_options_for_xml() {
        let opts = ExportOptions::for_xml();
        assert!(opts.export_audio);
        assert!(opts.export_video);
        assert!(opts.export_effects);
    }

    #[test]
    fn test_export_stats() {
        let stats = ExportStats::default();
        assert_eq!(stats.video_tracks, 0);
        assert_eq!(stats.audio_tracks, 0);
        assert_eq!(stats.clips, 0);
    }

    #[test]
    fn test_timeline_exporter_creation() {
        let exporter = TimelineExporter::with_defaults();
        assert!(exporter.options.export_audio);
        assert!(exporter.options.export_video);
    }

    #[test]
    fn test_export_to_string_json() {
        let timeline = create_test_timeline();
        let exporter = TimelineExporter::with_defaults();
        let result = exporter.export_to_string(&timeline, ExportFormat::Json);
        assert!(result.is_ok());
        assert!(result
            .expect("should succeed in test")
            .contains("Test Timeline"));
    }

    #[test]
    fn test_export_to_string_fcpxml() {
        let timeline = create_timeline_with_clips();
        let exporter = TimelineExporter::with_defaults();
        let result = exporter.export_to_string(&timeline, ExportFormat::Fcpxml);
        assert!(result.is_ok(), "fcpxml export failed: {result:?}");
        let xml = result.expect("should succeed in test");
        assert!(xml.contains("fcpxml"));
        assert!(xml.contains("My Project"));
        assert!(xml.contains("asset-clip"));
        assert!(xml.contains("Shot 01"));
    }

    #[test]
    fn test_export_to_string_premiere_xml() {
        let timeline = create_timeline_with_clips();
        let exporter = TimelineExporter::with_defaults();
        let result = exporter.export_to_string(&timeline, ExportFormat::PremiereXml);
        assert!(result.is_ok(), "premiere xml export failed: {result:?}");
        let xml = result.expect("should succeed in test");
        assert!(xml.contains("xmeml"));
        assert!(xml.contains("version=\"4\""));
        assert!(xml.contains("My Project"));
        assert!(xml.contains("Shot 01"));
        assert!(xml.contains("clipitem"));
    }

    #[test]
    fn test_export_to_string_resolve_xml() {
        let timeline = create_timeline_with_clips();
        let exporter = TimelineExporter::with_defaults();
        let result = exporter.export_to_string(&timeline, ExportFormat::ResolveXml);
        assert!(result.is_ok(), "resolve xml export failed: {result:?}");
        let xml = result.expect("should succeed in test");
        assert!(xml.contains("xmeml"));
        assert!(xml.contains("version=\"5\""));
        assert!(xml.contains("My Project"));
    }

    #[test]
    fn test_export_fcpxml_to_file() {
        let timeline = create_timeline_with_clips();
        let exporter = TimelineExporter::with_defaults();
        let tmp = std::env::temp_dir().join("test_export.fcpxml");
        let result = exporter.export_fcpxml(&timeline, &tmp);
        assert!(result.is_ok(), "fcpxml file export failed: {result:?}");
        let stats = result.expect("should succeed in test");
        assert!(stats.clips >= 1);
        assert!(stats.file_size > 0);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_export_premiere_to_file() {
        let timeline = create_timeline_with_clips();
        let exporter = TimelineExporter::with_defaults();
        let tmp = std::env::temp_dir().join("test_export_premiere.xml");
        let result = exporter.export_premiere_xml(&timeline, &tmp);
        assert!(result.is_ok(), "premiere file export failed: {result:?}");
        let stats = result.expect("should succeed in test");
        assert!(stats.clips >= 1);
        assert!(stats.file_size > 0);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_export_resolve_to_file() {
        let timeline = create_timeline_with_clips();
        let exporter = TimelineExporter::with_defaults();
        let tmp = std::env::temp_dir().join("test_export_resolve.xml");
        let result = exporter.export_resolve_xml(&timeline, &tmp);
        assert!(result.is_ok(), "resolve file export failed: {result:?}");
        let stats = result.expect("should succeed in test");
        assert!(stats.clips >= 1);
        assert!(stats.file_size > 0);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_frames_to_fcpxml_time() {
        // 24 frames at 24fps → "2400/2400s" (1 second)
        let s = frames_to_fcpxml_time(24, 24, 1);
        assert_eq!(s, "2400/2400s");
        // 0 frames → "0/2400s"
        let s0 = frames_to_fcpxml_time(0, 24, 1);
        assert_eq!(s0, "0/2400s");
    }

    #[test]
    fn test_export_edl_to_file() {
        let timeline = create_timeline_with_clips();
        let exporter = TimelineExporter::with_defaults();
        let tmp = std::env::temp_dir().join("test_export.edl");
        let result = exporter.export_edl(&timeline, &tmp);
        assert!(result.is_ok(), "edl file export failed: {result:?}");
        let stats = result.expect("should succeed in test");
        assert!(stats.clips >= 1, "Expected at least one clip in EDL");
        assert!(stats.file_size > 0, "EDL file should be non-empty");

        // Verify the file content.
        let content = std::fs::read_to_string(&tmp).expect("should succeed in test");
        assert!(content.contains("TITLE:"), "EDL should have a TITLE line");
        assert!(content.contains("FCM:"), "EDL should have an FCM line");
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_export_json_effect_count() {
        use crate::effects::{Effect, EffectType};

        let mut timeline = create_test_timeline();
        let v_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        let mut clip = Clip::new(
            "clip1".to_string(),
            MediaSource::file(PathBuf::from("/tmp/c.mov")),
            Position::new(0),
            Position::new(24),
            Position::new(0),
        )
        .expect("should succeed in test");
        // Add two effects to the clip.
        clip.effects
            .add_effect(Effect::new("Blur".to_string(), EffectType::Blur));
        clip.effects
            .add_effect(Effect::new("CC".to_string(), EffectType::ColorCorrection));
        timeline
            .add_clip(v_id, clip)
            .expect("should succeed in test");

        let exporter = TimelineExporter::with_defaults();
        let tmp = std::env::temp_dir().join("test_export_effects.json");
        let result = exporter.export_json(&timeline, &tmp);
        assert!(result.is_ok(), "json export failed: {result:?}");
        let stats = result.expect("should succeed in test");
        assert_eq!(stats.effects, 2, "Should count 2 clip effects");
        std::fs::remove_file(&tmp).ok();
    }
}
