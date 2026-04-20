//! FCPXML export — export timeline to Final Cut Pro XML format.
//!
//! Generates FCPXML v1.11 compatible XML documents containing resources,
//! clips, transitions, and markers from the timeline.

use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;
use std::io::Cursor;

use crate::clip::MediaSource;
use crate::error::{TimelineError, TimelineResult};
use crate::marker::MarkerType;
use crate::timeline::Timeline;
use crate::transition::TransitionType;

/// FCPXML export options.
#[derive(Clone, Debug)]
pub struct FcpxmlExportOptions {
    /// FCPXML version (default "1.11").
    pub version: String,
    /// Whether to include markers in the export.
    pub include_markers: bool,
    /// Whether to include transitions.
    pub include_transitions: bool,
    /// Whether to include effects metadata.
    pub include_effects: bool,
}

impl Default for FcpxmlExportOptions {
    fn default() -> Self {
        Self {
            version: "1.11".to_string(),
            include_markers: true,
            include_transitions: true,
            include_effects: true,
        }
    }
}

/// FCPXML exporter.
pub struct FcpxmlExporter;

impl FcpxmlExporter {
    /// Exports the timeline to FCPXML format.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn export(timeline: &Timeline, options: &FcpxmlExportOptions) -> TimelineResult<String> {
        let mut writer = Writer::new_with_indent(Cursor::new(Vec::new()), b' ', 2);

        // XML declaration
        writer
            .write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        // DOCTYPE
        writer
            .write_event(Event::Text(BytesText::new("\n<!DOCTYPE fcpxml>\n")))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        // <fcpxml version="...">
        let mut fcpxml = BytesStart::new("fcpxml");
        fcpxml.push_attribute(("version", options.version.as_str()));
        writer
            .write_event(Event::Start(fcpxml))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        // <resources>
        Self::write_resources(&mut writer, timeline)?;

        // <library>
        Self::write_library(&mut writer, timeline, options)?;

        // </fcpxml>
        writer
            .write_event(Event::End(BytesEnd::new("fcpxml")))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        let result = writer.into_inner().into_inner();
        String::from_utf8(result)
            .map_err(|e| TimelineError::ExportError(format!("UTF-8 error: {e}")))
    }

    /// Writes the <resources> section with media references.
    fn write_resources(
        writer: &mut Writer<Cursor<Vec<u8>>>,
        timeline: &Timeline,
    ) -> TimelineResult<()> {
        writer
            .write_event(Event::Start(BytesStart::new("resources")))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        // Format resource
        let mut format_elem = BytesStart::new("format");
        format_elem.push_attribute(("id", "r0"));
        let fr_str = format!("{}/{}s", timeline.frame_rate.den, timeline.frame_rate.num);
        format_elem.push_attribute(("frameDuration", fr_str.as_str()));
        format_elem.push_attribute(("name", "FFVideoFormat1080p"));
        writer
            .write_event(Event::Empty(format_elem))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        // Collect unique media sources
        let mut resource_id = 1u32;
        let all_tracks = timeline
            .video_tracks
            .iter()
            .chain(&timeline.audio_tracks)
            .chain(&timeline.subtitle_tracks);

        for track in all_tracks {
            for clip in &track.clips {
                if let MediaSource::File { ref path, .. } = clip.source {
                    let mut asset = BytesStart::new("asset");
                    let id_str = format!("r{resource_id}");
                    asset.push_attribute(("id", id_str.as_str()));
                    asset.push_attribute(("name", clip.name.as_str()));
                    let path_str = path.to_string_lossy();
                    asset.push_attribute(("src", path_str.as_ref()));
                    writer
                        .write_event(Event::Empty(asset))
                        .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;
                    resource_id += 1;
                }
            }
        }

        writer
            .write_event(Event::End(BytesEnd::new("resources")))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        Ok(())
    }

    /// Writes the <library> section with events, projects, and sequences.
    fn write_library(
        writer: &mut Writer<Cursor<Vec<u8>>>,
        timeline: &Timeline,
        options: &FcpxmlExportOptions,
    ) -> TimelineResult<()> {
        writer
            .write_event(Event::Start(BytesStart::new("library")))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        // <event name="...">
        let mut event_elem = BytesStart::new("event");
        event_elem.push_attribute(("name", timeline.name.as_str()));
        writer
            .write_event(Event::Start(event_elem))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        // <project name="...">
        let mut project_elem = BytesStart::new("project");
        project_elem.push_attribute(("name", timeline.name.as_str()));
        writer
            .write_event(Event::Start(project_elem))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        // <sequence>
        let mut seq_elem = BytesStart::new("sequence");
        let dur_str = format!(
            "{}/{}s",
            timeline.duration.value() * timeline.frame_rate.den,
            timeline.frame_rate.num
        );
        seq_elem.push_attribute(("duration", dur_str.as_str()));
        seq_elem.push_attribute(("format", "r0"));
        writer
            .write_event(Event::Start(seq_elem))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        // <spine>
        writer
            .write_event(Event::Start(BytesStart::new("spine")))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        // Write clips from video tracks
        let mut resource_id = 1u32;
        for track in &timeline.video_tracks {
            for clip in &track.clips {
                Self::write_clip(writer, clip, &mut resource_id, timeline, options)?;
            }
        }

        // </spine>
        writer
            .write_event(Event::End(BytesEnd::new("spine")))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        // Write markers
        if options.include_markers {
            for marker in timeline.markers.markers() {
                Self::write_marker(writer, marker, timeline)?;
            }
        }

        // </sequence>
        writer
            .write_event(Event::End(BytesEnd::new("sequence")))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        // </project>
        writer
            .write_event(Event::End(BytesEnd::new("project")))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        // </event>
        writer
            .write_event(Event::End(BytesEnd::new("event")))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        // </library>
        writer
            .write_event(Event::End(BytesEnd::new("library")))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        Ok(())
    }

    /// Writes a single clip element.
    #[allow(clippy::cast_precision_loss)]
    fn write_clip(
        writer: &mut Writer<Cursor<Vec<u8>>>,
        clip: &crate::clip::Clip,
        resource_id: &mut u32,
        timeline: &Timeline,
        options: &FcpxmlExportOptions,
    ) -> TimelineResult<()> {
        let den = timeline.frame_rate.den;
        let num = timeline.frame_rate.num;

        let mut clip_elem = BytesStart::new("clip");
        clip_elem.push_attribute(("name", clip.name.as_str()));

        let offset_str = format!("{}/{}s", clip.timeline_in.value() * den, num);
        clip_elem.push_attribute(("offset", offset_str.as_str()));

        let dur_str = format!("{}/{}s", clip.timeline_duration().value() * den, num);
        clip_elem.push_attribute(("duration", dur_str.as_str()));

        let start_str = format!("{}/{}s", clip.source_in.value() * den, num);
        clip_elem.push_attribute(("start", start_str.as_str()));

        if let MediaSource::File { .. } = clip.source {
            let ref_str = format!("r{resource_id}");
            clip_elem.push_attribute(("ref", ref_str.as_str()));
            *resource_id += 1;
        }

        // Check if we need transitions
        let has_transition =
            options.include_transitions && timeline.transitions.contains_key(&clip.id);

        if has_transition || (options.include_effects && !clip.effects.is_empty()) {
            writer
                .write_event(Event::Start(clip_elem))
                .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

            // Write transition if present
            if let Some(transition) = timeline.transitions.get(&clip.id) {
                if options.include_transitions {
                    let mut trans_elem = BytesStart::new("transition");
                    let trans_dur = format!("{}/{}s", transition.duration.value() * den, num);
                    trans_elem.push_attribute(("duration", trans_dur.as_str()));
                    let trans_name = match transition.transition_type {
                        TransitionType::Dissolve => "Cross Dissolve",
                        TransitionType::DipToBlack => "Fade to Black",
                        TransitionType::DipToWhite => "Fade to White",
                        TransitionType::Wipe => "Wipe",
                        TransitionType::Push => "Push",
                        TransitionType::Slide => "Slide",
                        _ => "Cross Dissolve",
                    };
                    trans_elem.push_attribute(("name", trans_name));
                    writer
                        .write_event(Event::Empty(trans_elem))
                        .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;
                }
            }

            writer
                .write_event(Event::End(BytesEnd::new("clip")))
                .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;
        } else {
            writer
                .write_event(Event::Empty(clip_elem))
                .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;
        }

        Ok(())
    }

    /// Writes a single marker element.
    fn write_marker(
        writer: &mut Writer<Cursor<Vec<u8>>>,
        marker: &crate::marker::Marker,
        timeline: &Timeline,
    ) -> TimelineResult<()> {
        let den = timeline.frame_rate.den;
        let num = timeline.frame_rate.num;

        let tag = match marker.marker_type {
            MarkerType::Chapter => "chapter-marker",
            _ => "marker",
        };

        let mut marker_elem = BytesStart::new(tag);
        let offset_str = format!("{}/{}s", marker.position.value() * den, num);
        marker_elem.push_attribute(("start", offset_str.as_str()));
        marker_elem.push_attribute(("value", marker.name.as_str()));

        if let Some(dur) = marker.duration {
            let dur_str = format!("{}/{}s", dur.value() * den, num);
            marker_elem.push_attribute(("duration", dur_str.as_str()));
        }

        writer
            .write_event(Event::Empty(marker_elem))
            .map_err(|e| TimelineError::ExportError(format!("XML write error: {e}")))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Clip, MediaSource};
    use crate::marker::Marker;
    use crate::transition::Transition;
    use crate::types::{Duration, Position};
    use oximedia_core::Rational;
    use std::path::PathBuf;

    fn create_test_timeline() -> Timeline {
        Timeline::new("Test Project", Rational::new(24, 1), 48000).expect("should succeed in test")
    }

    #[test]
    fn test_fcpxml_export_empty_timeline() {
        let timeline = create_test_timeline();
        let options = FcpxmlExportOptions::default();
        let result = FcpxmlExporter::export(&timeline, &options);
        assert!(result.is_ok());
        let xml = result.expect("should succeed in test");
        assert!(xml.contains("<fcpxml"));
        assert!(xml.contains("</fcpxml>"));
        assert!(xml.contains("<resources"));
        assert!(xml.contains("<library"));
    }

    #[test]
    fn test_fcpxml_export_with_clips() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        let clip = Clip::new(
            "Clip1".to_string(),
            MediaSource::file(std::env::temp_dir().join("oximedia-timeline-fcpxml-test.mov")),
            Position::new(0),
            Position::new(100),
            Position::new(0),
        )
        .expect("should succeed in test");
        timeline
            .add_clip(track_id, clip)
            .expect("should succeed in test");

        let options = FcpxmlExportOptions::default();
        let xml = FcpxmlExporter::export(&timeline, &options).expect("should succeed in test");
        assert!(xml.contains("Clip1"));
        assert!(xml.contains("<clip"));
    }

    #[test]
    fn test_fcpxml_export_with_markers() {
        let mut timeline = create_test_timeline();
        timeline
            .markers
            .add_marker(Marker::new(Position::new(50), "Marker1".to_string()));
        timeline
            .markers
            .add_marker(Marker::chapter(Position::new(100), "Chapter1".to_string()));

        let options = FcpxmlExportOptions::default();
        let xml = FcpxmlExporter::export(&timeline, &options).expect("should succeed in test");
        assert!(xml.contains("Marker1"));
        assert!(xml.contains("Chapter1"));
        assert!(xml.contains("chapter-marker"));
    }

    #[test]
    fn test_fcpxml_export_with_transitions() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        let clip = Clip::new(
            "Clip1".to_string(),
            MediaSource::black(),
            Position::new(0),
            Position::new(100),
            Position::new(0),
        )
        .expect("should succeed in test");
        let clip_id = clip.id;
        timeline
            .add_clip(track_id, clip)
            .expect("should succeed in test");
        timeline
            .add_transition(clip_id, Transition::dissolve(Duration::new(24)))
            .expect("should succeed in test");

        let options = FcpxmlExportOptions::default();
        let xml = FcpxmlExporter::export(&timeline, &options).expect("should succeed in test");
        assert!(xml.contains("transition"));
        assert!(xml.contains("Cross Dissolve"));
    }

    #[test]
    fn test_fcpxml_export_no_markers_option() {
        let mut timeline = create_test_timeline();
        timeline
            .markers
            .add_marker(Marker::new(Position::new(50), "Marker1".to_string()));

        let options = FcpxmlExportOptions {
            include_markers: false,
            ..Default::default()
        };
        let xml = FcpxmlExporter::export(&timeline, &options).expect("should succeed in test");
        assert!(!xml.contains("Marker1"));
    }

    #[test]
    fn test_fcpxml_export_version() {
        let timeline = create_test_timeline();
        let options = FcpxmlExportOptions {
            version: "1.10".to_string(),
            ..Default::default()
        };
        let xml = FcpxmlExporter::export(&timeline, &options).expect("should succeed in test");
        assert!(xml.contains("version=\"1.10\""));
    }

    #[test]
    fn test_fcpxml_export_options_default() {
        let opts = FcpxmlExportOptions::default();
        assert_eq!(opts.version, "1.11");
        assert!(opts.include_markers);
        assert!(opts.include_transitions);
        assert!(opts.include_effects);
    }

    #[test]
    fn test_fcpxml_export_contains_project_name() {
        let timeline = create_test_timeline();
        let options = FcpxmlExportOptions::default();
        let xml = FcpxmlExporter::export(&timeline, &options).expect("should succeed in test");
        assert!(xml.contains("Test Project"));
    }

    #[test]
    fn test_fcpxml_export_format_resource() {
        let timeline = create_test_timeline();
        let options = FcpxmlExportOptions::default();
        let xml = FcpxmlExporter::export(&timeline, &options).expect("should succeed in test");
        assert!(xml.contains("<format"));
        assert!(xml.contains("frameDuration"));
    }

    #[test]
    fn test_fcpxml_export_multiple_clips() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");

        for i in 0..3 {
            let clip = Clip::new(
                format!("Clip{i}"),
                MediaSource::black(),
                Position::new(0),
                Position::new(50),
                Position::new(i * 50),
            )
            .expect("should succeed in test");
            timeline
                .add_clip(track_id, clip)
                .expect("should succeed in test");
        }

        let options = FcpxmlExportOptions::default();
        let xml = FcpxmlExporter::export(&timeline, &options).expect("should succeed in test");
        assert!(xml.contains("Clip0"));
        assert!(xml.contains("Clip1"));
        assert!(xml.contains("Clip2"));
    }
}
