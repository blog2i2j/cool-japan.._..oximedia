//! EDL importer for conforming.

use crate::error::{ConformError, ConformResult};
use crate::importers::TimelineImporter;
use crate::types::{ClipReference, FrameRate, Timecode, TrackType};
use oximedia_edl::event::EdlEvent;
use oximedia_edl::Edl;
use std::path::Path;

/// EDL importer.
pub struct EdlImporter;

impl EdlImporter {
    /// Create a new EDL importer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Convert EDL frame rate to conform frame rate.
    fn convert_frame_rate(edl_fps: oximedia_edl::timecode::EdlFrameRate) -> FrameRate {
        match edl_fps {
            oximedia_edl::timecode::EdlFrameRate::Fps24 => FrameRate::Fps24,
            oximedia_edl::timecode::EdlFrameRate::Fps25 => FrameRate::Fps25,
            oximedia_edl::timecode::EdlFrameRate::Fps2997DF => FrameRate::Fps2997DF,
            oximedia_edl::timecode::EdlFrameRate::Fps2997NDF => FrameRate::Fps2997NDF,
            oximedia_edl::timecode::EdlFrameRate::Fps30 => FrameRate::Fps30,
            _ => FrameRate::Fps25, // Default fallback
        }
    }

    /// Convert EDL timecode to conform timecode.
    fn convert_timecode(edl_tc: &oximedia_edl::timecode::EdlTimecode) -> Timecode {
        Timecode::new(
            edl_tc.hours(),
            edl_tc.minutes(),
            edl_tc.seconds(),
            edl_tc.frames(),
        )
    }

    /// Convert EDL track type to conform track type.
    fn convert_track_type(edl_track: &oximedia_edl::event::TrackType) -> TrackType {
        if edl_track.has_video() && edl_track.has_audio() {
            TrackType::AudioVideo
        } else if edl_track.has_video() {
            TrackType::Video
        } else {
            TrackType::Audio
        }
    }

    /// Convert an EDL event to a clip reference.
    fn event_to_clip(event: &EdlEvent, fps: FrameRate) -> ClipReference {
        let mut metadata: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        metadata.insert("reel".to_string(), event.reel.clone());
        metadata.insert("event_number".to_string(), event.number.to_string());
        metadata.insert("edit_type".to_string(), event.edit_type.to_string());

        if let Some(clip_name) = &event.clip_name {
            metadata.insert("clip_name".to_string(), clip_name.clone());
        }

        ClipReference {
            id: format!("event_{}", event.number),
            source_file: event.clip_name.clone(),
            source_in: Self::convert_timecode(&event.source_in),
            source_out: Self::convert_timecode(&event.source_out),
            record_in: Self::convert_timecode(&event.record_in),
            record_out: Self::convert_timecode(&event.record_out),
            track: Self::convert_track_type(&event.track),
            fps,
            metadata,
        }
    }

    /// Import from EDL object.
    pub fn import_from_edl(edl: &Edl) -> ConformResult<Vec<ClipReference>> {
        let fps = Self::convert_frame_rate(edl.frame_rate);
        let clips: Vec<ClipReference> = edl
            .events
            .iter()
            .map(|event| Self::event_to_clip(event, fps))
            .collect();
        Ok(clips)
    }
}

impl Default for EdlImporter {
    fn default() -> Self {
        Self::new()
    }
}

impl TimelineImporter for EdlImporter {
    fn import<P: AsRef<Path>>(&self, path: P) -> ConformResult<Vec<ClipReference>> {
        let edl = Edl::from_file(path.as_ref()).map_err(|e| ConformError::Edl(e.to_string()))?;
        Self::import_from_edl(&edl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_edl::event::{EditType, TrackType as EdlTrackType};
    use oximedia_edl::timecode::{EdlFrameRate, EdlTimecode};

    #[test]
    fn test_convert_frame_rate() {
        assert_eq!(
            EdlImporter::convert_frame_rate(EdlFrameRate::Fps25),
            FrameRate::Fps25
        );
        assert_eq!(
            EdlImporter::convert_frame_rate(EdlFrameRate::Fps2997DF),
            FrameRate::Fps2997DF
        );
    }

    #[test]
    fn test_convert_timecode() {
        let edl_tc =
            EdlTimecode::new(1, 23, 45, 12, EdlFrameRate::Fps25).expect("edl_tc should be valid");
        let tc = EdlImporter::convert_timecode(&edl_tc);
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 23);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 12);
    }

    #[test]
    fn test_event_to_clip() {
        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("tc1 should be valid");
        let tc2 = EdlTimecode::new(1, 0, 10, 0, EdlFrameRate::Fps25).expect("tc2 should be valid");

        let event = EdlEvent::new(
            1,
            "A001".to_string(),
            EdlTrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        let clip = EdlImporter::event_to_clip(&event, FrameRate::Fps25);
        assert_eq!(clip.id, "event_1");
        assert_eq!(clip.track, TrackType::Video);
    }
}
