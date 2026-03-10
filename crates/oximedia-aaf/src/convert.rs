//! Conversion utilities
//!
//! This module provides conversion between AAF and other formats:
//! - EDL (Edit Decision List) export
//! - XML export
//! - Timeline conversion
//! - Media reference conversion

use crate::composition::{CompositionMob, SequenceComponent, SourceClip, Track};
use crate::metadata::Timecode;
use crate::timeline::{EditRate, Position};
use crate::{AafError, AafFile, Result};
use std::fmt::Write as FmtWrite;

/// EDL export functionality
pub struct EdlExporter {
    /// Title
    title: String,
    /// Frame rate
    frame_rate: EditRate,
    /// Drop frame
    drop_frame: bool,
}

impl EdlExporter {
    /// Create a new EDL exporter
    pub fn new(title: impl Into<String>, frame_rate: EditRate) -> Self {
        Self {
            title: title.into(),
            frame_rate,
            drop_frame: frame_rate.is_ntsc(),
        }
    }

    /// Export AAF to EDL
    pub fn export(&self, aaf: &AafFile) -> Result<String> {
        let mut edl = String::new();

        // Write title
        writeln!(&mut edl, "TITLE: {}", self.title)
            .map_err(|e| AafError::ConversionError(e.to_string()))?;

        // Write FCM (Frame Code Mode)
        if self.drop_frame {
            writeln!(&mut edl, "FCM: DROP FRAME")
                .map_err(|e| AafError::ConversionError(e.to_string()))?;
        } else {
            writeln!(&mut edl, "FCM: NON-DROP FRAME")
                .map_err(|e| AafError::ConversionError(e.to_string()))?;
        }

        writeln!(&mut edl).map_err(|e| AafError::ConversionError(e.to_string()))?;

        // Get composition mobs
        let comp_mobs = aaf.composition_mobs();

        if let Some(comp_mob) = comp_mobs.first() {
            self.export_composition(comp_mob, &mut edl)?;
        }

        Ok(edl)
    }

    /// Export a composition mob to EDL
    fn export_composition(&self, comp_mob: &CompositionMob, edl: &mut String) -> Result<()> {
        let mut event_number = 1;

        // Process each track
        for track in comp_mob.tracks() {
            if track.is_picture() || track.is_sound() {
                event_number = self.export_track(&track, edl, event_number)?;
            }
        }

        Ok(())
    }

    /// Export a track to EDL
    fn export_track(&self, track: &Track, edl: &mut String, mut event_number: u32) -> Result<u32> {
        if let Some(ref sequence) = track.sequence {
            let mut timeline_pos = Position::zero();

            for component in &sequence.components {
                if let SequenceComponent::SourceClip(clip) = component {
                    self.export_edit(edl, event_number, track, clip, timeline_pos)?;
                    event_number += 1;
                    timeline_pos = Position(timeline_pos.0 + clip.length);
                }
            }
        }

        Ok(event_number)
    }

    /// Export a single edit to EDL
    fn export_edit(
        &self,
        edl: &mut String,
        event_number: u32,
        track: &Track,
        clip: &SourceClip,
        timeline_pos: Position,
    ) -> Result<()> {
        // Format: EVENT# REEL TRACK EDIT_TYPE
        let track_type = if track.is_picture() { "V" } else { "A" };
        let reel = format!("{}", clip.source_mob_id)
            .chars()
            .take(8)
            .collect::<String>();

        writeln!(edl, "{event_number:03} {reel} {track_type} C")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;

        // Source in/out
        let source_in = self.position_to_timecode(clip.start_time);
        let source_out = self.position_to_timecode(Position(clip.start_time.0 + clip.length));

        // Record in/out
        let record_in = self.position_to_timecode(timeline_pos);
        let record_out = self.position_to_timecode(Position(timeline_pos.0 + clip.length));

        writeln!(edl, "* FROM CLIP NAME: SOURCE_{event_number}")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;

        writeln!(edl, "{source_in} {source_out} {record_in} {record_out}")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;

        writeln!(edl).map_err(|e| AafError::ConversionError(e.to_string()))?;

        Ok(())
    }

    /// Convert position to timecode string
    fn position_to_timecode(&self, position: Position) -> String {
        let _fps = self.frame_rate.to_float().round() as u8;
        let tc = Timecode::from_position(position, self.frame_rate);
        tc.to_string()
    }
}

/// XML export functionality
pub struct XmlExporter {
    /// Include metadata
    include_metadata: bool,
}

impl XmlExporter {
    /// Create a new XML exporter
    #[must_use]
    pub fn new() -> Self {
        Self {
            include_metadata: true,
        }
    }

    /// Set whether to include metadata
    #[must_use]
    pub fn with_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Export AAF to XML
    pub fn export(&self, aaf: &AafFile) -> Result<String> {
        let mut xml = String::new();

        writeln!(&mut xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(&mut xml, "<aaf>").map_err(|e| AafError::ConversionError(e.to_string()))?;

        // Export header
        self.export_header(aaf.header(), &mut xml)?;

        // Export composition mobs
        for comp_mob in aaf.composition_mobs() {
            self.export_composition_mob(comp_mob, &mut xml)?;
        }

        writeln!(&mut xml, "</aaf>").map_err(|e| AafError::ConversionError(e.to_string()))?;

        Ok(xml)
    }

    /// Export header to XML
    fn export_header(&self, header: &crate::object_model::Header, xml: &mut String) -> Result<()> {
        writeln!(xml, "  <header>").map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "    <version>{}</version>", header.version_string())
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "  </header>").map_err(|e| AafError::ConversionError(e.to_string()))?;

        Ok(())
    }

    /// Export composition mob to XML
    fn export_composition_mob(&self, comp_mob: &CompositionMob, xml: &mut String) -> Result<()> {
        writeln!(
            xml,
            "  <composition id=\"{}\" name=\"{}\">",
            comp_mob.mob_id(),
            comp_mob.name()
        )
        .map_err(|e| AafError::ConversionError(e.to_string()))?;

        for track in comp_mob.tracks() {
            self.export_track(&track, xml)?;
        }

        writeln!(xml, "  </composition>").map_err(|e| AafError::ConversionError(e.to_string()))?;

        Ok(())
    }

    /// Export track to XML
    fn export_track(&self, track: &Track, xml: &mut String) -> Result<()> {
        let track_type = if track.is_picture() {
            "picture"
        } else if track.is_sound() {
            "sound"
        } else {
            "unknown"
        };

        writeln!(
            xml,
            "    <track id=\"{}\" name=\"{}\" type=\"{}\" edit_rate=\"{}\">",
            track.track_id, track.name, track_type, track.edit_rate
        )
        .map_err(|e| AafError::ConversionError(e.to_string()))?;

        if let Some(ref sequence) = track.sequence {
            for component in &sequence.components {
                self.export_component(component, xml)?;
            }
        }

        writeln!(xml, "    </track>").map_err(|e| AafError::ConversionError(e.to_string()))?;

        Ok(())
    }

    /// Export sequence component to XML
    fn export_component(&self, component: &SequenceComponent, xml: &mut String) -> Result<()> {
        match component {
            SequenceComponent::SourceClip(clip) => {
                writeln!(
                    xml,
                    "      <source_clip length=\"{}\" start=\"{}\" mob_id=\"{}\" slot_id=\"{}\" />",
                    clip.length, clip.start_time, clip.source_mob_id, clip.source_mob_slot_id
                )
                .map_err(|e| AafError::ConversionError(e.to_string()))?;
            }
            SequenceComponent::Filler(filler) => {
                writeln!(xml, "      <filler length=\"{}\" />", filler.length)
                    .map_err(|e| AafError::ConversionError(e.to_string()))?;
            }
            SequenceComponent::Transition(trans) => {
                writeln!(
                    xml,
                    "      <transition length=\"{}\" cut_point=\"{}\" />",
                    trans.length, trans.cut_point
                )
                .map_err(|e| AafError::ConversionError(e.to_string()))?;
            }
            SequenceComponent::Effect(effect) => {
                writeln!(
                    xml,
                    "      <effect operation_id=\"{}\" />",
                    effect.operation_id
                )
                .map_err(|e| AafError::ConversionError(e.to_string()))?;
            }
        }

        Ok(())
    }
}

impl Default for XmlExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Timeline converter - converts AAF timeline to a simplified representation
pub struct TimelineConverter;

impl TimelineConverter {
    /// Convert AAF to timeline representation
    pub fn convert(aaf: &AafFile) -> Result<Timeline> {
        let mut timeline = Timeline {
            name: String::new(),
            edit_rate: None,
            duration: None,
            tracks: Vec::new(),
        };

        if let Some(comp_mob) = aaf.composition_mobs().first() {
            timeline.name = comp_mob.name().to_string();
            timeline.edit_rate = comp_mob.edit_rate();
            timeline.duration = comp_mob.duration();

            for track in comp_mob.tracks() {
                timeline.tracks.push(Self::convert_track(&track)?);
            }
        }

        Ok(timeline)
    }

    /// Convert a track
    fn convert_track(track: &Track) -> Result<TimelineTrack> {
        let mut clips = Vec::new();

        if let Some(ref sequence) = track.sequence {
            let mut position = Position::zero();

            for component in &sequence.components {
                if let SequenceComponent::SourceClip(clip) = component {
                    clips.push(TimelineClip {
                        start: position,
                        duration: clip.length,
                        source_start: clip.start_time,
                        source_id: clip.source_mob_id.to_string(),
                    });
                    position = Position(position.0 + clip.length);
                }
            }
        }

        Ok(TimelineTrack {
            name: track.name.clone(),
            track_type: if track.is_picture() {
                "picture"
            } else if track.is_sound() {
                "sound"
            } else {
                "unknown"
            }
            .to_string(),
            clips,
        })
    }
}

/// Simplified timeline representation
#[derive(Debug, Clone)]
pub struct Timeline {
    /// Timeline name
    pub name: String,
    /// Edit rate
    pub edit_rate: Option<EditRate>,
    /// Duration
    pub duration: Option<i64>,
    /// Tracks
    pub tracks: Vec<TimelineTrack>,
}

/// Timeline track
#[derive(Debug, Clone)]
pub struct TimelineTrack {
    /// Track name
    pub name: String,
    /// Track type
    pub track_type: String,
    /// Clips
    pub clips: Vec<TimelineClip>,
}

/// Timeline clip
#[derive(Debug, Clone)]
pub struct TimelineClip {
    /// Start position in timeline
    pub start: Position,
    /// Duration
    pub duration: i64,
    /// Start in source
    pub source_start: Position,
    /// Source ID
    pub source_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::{SequenceComponent, TrackType};
    use crate::timeline::EditRate;
    use crate::writer::{CompositionBuilder, SequenceBuilder, TrackBuilder};
    use crate::AafFile;
    use uuid::Uuid;

    fn create_test_aaf() -> AafFile {
        let clip1 = SourceClip::new(100, Position::zero(), Uuid::new_v4(), 1);
        let clip2 = SourceClip::new(50, Position::new(100), Uuid::new_v4(), 1);

        let sequence = SequenceBuilder::picture()
            .add_component(SequenceComponent::SourceClip(clip1))
            .add_component(SequenceComponent::SourceClip(clip2))
            .build();

        let track = TrackBuilder::new(1, "Video", EditRate::PAL_25, TrackType::Picture)
            .with_sequence(sequence)
            .build();

        let comp = CompositionBuilder::new("Test Composition")
            .add_track(track)
            .build();

        let mut aaf = AafFile::new();
        aaf.content_storage.add_composition_mob(comp);
        aaf
    }

    #[test]
    fn test_edl_export() {
        let aaf = create_test_aaf();
        let exporter = EdlExporter::new("Test EDL", EditRate::PAL_25);
        let edl = exporter.export(&aaf).expect("edl should be valid");

        assert!(edl.contains("TITLE: Test EDL"));
        assert!(edl.contains("FCM:"));
    }

    #[test]
    fn test_xml_export() {
        let aaf = create_test_aaf();
        let exporter = XmlExporter::new();
        let xml = exporter.export(&aaf).expect("xml should be valid");

        assert!(xml.contains("<?xml"));
        assert!(xml.contains("<aaf>"));
        assert!(xml.contains("<composition"));
        assert!(xml.contains("</aaf>"));
    }

    #[test]
    fn test_timeline_conversion() {
        let aaf = create_test_aaf();
        let timeline = TimelineConverter::convert(&aaf).expect("timeline should be valid");

        assert_eq!(timeline.name, "Test Composition");
        assert_eq!(timeline.tracks.len(), 1);
        assert_eq!(timeline.tracks[0].clips.len(), 2);
    }

    #[test]
    fn test_timeline_track_conversion() {
        let aaf = create_test_aaf();
        let timeline = TimelineConverter::convert(&aaf).expect("timeline should be valid");

        let track = &timeline.tracks[0];
        assert_eq!(track.name, "Video");
        assert_eq!(track.track_type, "picture");
    }

    #[test]
    fn test_timeline_clip() {
        let aaf = create_test_aaf();
        let timeline = TimelineConverter::convert(&aaf).expect("timeline should be valid");

        let clip = &timeline.tracks[0].clips[0];
        assert_eq!(clip.start.0, 0);
        assert_eq!(clip.duration, 100);
    }
}
