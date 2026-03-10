//! EDL format conversion between CMX 3400, CMX 3600, GVG, and other formats.
//!
//! This module provides functionality to convert between different EDL
//! formats while preserving as much information as possible.

use crate::error::{EdlError, EdlResult};
use crate::event::{EditType, EdlEvent};
use crate::{Edl, EdlFormat};

/// EDL format converter.
#[derive(Debug)]
pub struct EdlConverter {
    /// Preserve comments during conversion.
    pub preserve_comments: bool,
    /// Preserve motion effects during conversion.
    pub preserve_motion_effects: bool,
    /// Preserve wipe patterns during conversion.
    pub preserve_wipe_patterns: bool,
    /// Allow lossy conversion (may drop unsupported features).
    pub allow_lossy: bool,
}

impl EdlConverter {
    /// Create a new EDL converter with default settings.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            preserve_comments: true,
            preserve_motion_effects: true,
            preserve_wipe_patterns: true,
            allow_lossy: false,
        }
    }

    /// Create a lossless EDL converter (fails if features would be lost).
    #[must_use]
    pub const fn lossless() -> Self {
        Self {
            preserve_comments: true,
            preserve_motion_effects: true,
            preserve_wipe_patterns: true,
            allow_lossy: false,
        }
    }

    /// Create a lossy EDL converter (drops unsupported features).
    #[must_use]
    pub const fn lossy() -> Self {
        Self {
            preserve_comments: false,
            preserve_motion_effects: false,
            preserve_wipe_patterns: false,
            allow_lossy: true,
        }
    }

    /// Convert an EDL to a different format.
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion fails or would lose data in lossless mode.
    pub fn convert(&self, edl: &Edl, target_format: EdlFormat) -> EdlResult<Edl> {
        // If formats are the same, just clone
        if edl.format == target_format {
            return Ok(edl.clone());
        }

        match (edl.format, target_format) {
            (EdlFormat::Cmx3600, EdlFormat::Cmx3400) => self.cmx3600_to_cmx3400(edl),
            (EdlFormat::Cmx3400, EdlFormat::Cmx3600) => self.cmx3400_to_cmx3600(edl),
            (EdlFormat::Cmx3600, EdlFormat::Gvg) => self.cmx3600_to_gvg(edl),
            (EdlFormat::Gvg, EdlFormat::Cmx3600) => self.gvg_to_cmx3600(edl),
            (EdlFormat::Cmx3600, EdlFormat::SonyBve9000) => self.cmx3600_to_sony_bve(edl),
            (EdlFormat::SonyBve9000, EdlFormat::Cmx3600) => self.sony_bve_to_cmx3600(edl),
            _ => Err(EdlError::ConversionError(format!(
                "Unsupported conversion: {:?} to {:?}",
                edl.format, target_format
            ))),
        }
    }

    /// Convert CMX 3600 to CMX 3400.
    fn cmx3600_to_cmx3400(&self, edl: &Edl) -> EdlResult<Edl> {
        let mut new_edl = Edl::new(EdlFormat::Cmx3400);
        new_edl.title = edl.title.clone();
        new_edl.frame_rate = edl.frame_rate;

        for event in &edl.events {
            // CMX 3400 has limited support for some features
            if !self.allow_lossy {
                // Check for unsupported features
                if event.wipe_pattern.is_some() {
                    return Err(EdlError::ConversionError(
                        "CMX 3400 does not support wipe patterns".to_string(),
                    ));
                }

                if event.key_type.is_some() {
                    return Err(EdlError::ConversionError(
                        "CMX 3400 does not support key types".to_string(),
                    ));
                }
            }

            let mut new_event = event.clone();

            // Drop unsupported features if lossy conversion is allowed
            if !self.preserve_wipe_patterns {
                new_event.wipe_pattern = None;
            }

            if !self.preserve_comments {
                new_event.comments.clear();
            }

            new_edl.events.push(new_event);
        }

        Ok(new_edl)
    }

    /// Convert CMX 3400 to CMX 3600.
    fn cmx3400_to_cmx3600(&self, edl: &Edl) -> EdlResult<Edl> {
        let mut new_edl = Edl::new(EdlFormat::Cmx3600);
        new_edl.title = edl.title.clone();
        new_edl.frame_rate = edl.frame_rate;

        // CMX 3600 is a superset of CMX 3400, so this is lossless
        for event in &edl.events {
            new_edl.events.push(event.clone());
        }

        Ok(new_edl)
    }

    /// Convert CMX 3600 to GVG (Grass Valley Group).
    fn cmx3600_to_gvg(&self, edl: &Edl) -> EdlResult<Edl> {
        let mut new_edl = Edl::new(EdlFormat::Gvg);
        new_edl.title = edl.title.clone();
        new_edl.frame_rate = edl.frame_rate;

        for event in &edl.events {
            // GVG format differences
            let new_event = event.clone();

            // GVG uses different wipe numbering
            if let Some(_wipe) = &event.wipe_pattern {
                if !self.allow_lossy && !self.preserve_wipe_patterns {
                    // Could add wipe pattern conversion here
                }
            }

            new_edl.events.push(new_event);
        }

        Ok(new_edl)
    }

    /// Convert GVG to CMX 3600.
    fn gvg_to_cmx3600(&self, edl: &Edl) -> EdlResult<Edl> {
        let mut new_edl = Edl::new(EdlFormat::Cmx3600);
        new_edl.title = edl.title.clone();
        new_edl.frame_rate = edl.frame_rate;

        for event in &edl.events {
            new_edl.events.push(event.clone());
        }

        Ok(new_edl)
    }

    /// Convert CMX 3600 to Sony BVE-9000.
    fn cmx3600_to_sony_bve(&self, edl: &Edl) -> EdlResult<Edl> {
        let mut new_edl = Edl::new(EdlFormat::SonyBve9000);
        new_edl.title = edl.title.clone();
        new_edl.frame_rate = edl.frame_rate;

        for event in &edl.events {
            let mut new_event = event.clone();

            // Sony BVE has different motion effect syntax
            if event.motion_effect.is_some() && !self.preserve_motion_effects {
                if !self.allow_lossy {
                    return Err(EdlError::ConversionError(
                        "Sony BVE motion effects require manual conversion".to_string(),
                    ));
                }
                new_event.motion_effect = None;
            }

            new_edl.events.push(new_event);
        }

        Ok(new_edl)
    }

    /// Convert Sony BVE-9000 to CMX 3600.
    fn sony_bve_to_cmx3600(&self, edl: &Edl) -> EdlResult<Edl> {
        let mut new_edl = Edl::new(EdlFormat::Cmx3600);
        new_edl.title = edl.title.clone();
        new_edl.frame_rate = edl.frame_rate;

        for event in &edl.events {
            new_edl.events.push(event.clone());
        }

        Ok(new_edl)
    }

    /// Simplify an EDL by converting complex edits to simpler ones.
    ///
    /// # Errors
    ///
    /// Returns an error if simplification fails.
    pub fn simplify(&self, edl: &Edl) -> EdlResult<Edl> {
        let mut new_edl = edl.clone();

        for event in &mut new_edl.events {
            // Convert dissolves with very short durations to cuts
            if event.edit_type == EditType::Dissolve {
                if let Some(duration) = event.transition_duration {
                    if duration <= 2 {
                        event.edit_type = EditType::Cut;
                        event.transition_duration = None;
                    }
                }
            }

            // Convert wipes to dissolves
            if event.edit_type == EditType::Wipe && self.allow_lossy {
                event.edit_type = EditType::Dissolve;
                event.wipe_pattern = None;
            }
        }

        Ok(new_edl)
    }

    /// Optimize an EDL by removing redundant events and merging adjacent cuts.
    ///
    /// # Errors
    ///
    /// Returns an error if optimization fails.
    pub fn optimize(&self, edl: &Edl) -> EdlResult<Edl> {
        let mut new_edl = Edl::new(edl.format);
        new_edl.title = edl.title.clone();
        new_edl.frame_rate = edl.frame_rate;

        let mut event_num = 1;
        let mut prev_event: Option<&EdlEvent> = None;

        for event in &edl.events {
            // Check if this event can be merged with the previous one
            let mut should_add = true;

            if let Some(prev) = prev_event {
                // Can merge if same reel, track, and edit type is cut
                if prev.reel == event.reel
                    && prev.track == event.track
                    && prev.edit_type == EditType::Cut
                    && event.edit_type == EditType::Cut
                    && prev.record_out == event.record_in
                    && prev.source_out == event.source_in
                {
                    // Extend the previous event instead of adding a new one
                    if let Some(last_event) = new_edl.events.last_mut() {
                        last_event.record_out = event.record_out;
                        last_event.source_out = event.source_out;
                        should_add = false;
                    }
                }
            }

            if should_add {
                let mut new_event = event.clone();
                new_event.number = event_num;
                new_edl.events.push(new_event);
                event_num += 1;
            }

            prev_event = Some(event);
        }

        Ok(new_edl)
    }
}

impl Default for EdlConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::TrackType;
    use crate::timecode::{EdlFrameRate, EdlTimecode};

    #[test]
    fn test_same_format_conversion() {
        let edl = Edl::new(EdlFormat::Cmx3600);
        let converter = EdlConverter::new();
        let result = converter
            .convert(&edl, EdlFormat::Cmx3600)
            .expect("conversion should succeed");
        assert_eq!(result.format, EdlFormat::Cmx3600);
    }

    #[test]
    fn test_cmx3600_to_cmx3400() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");

        let event = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        edl.add_event(event).expect("add_event should succeed");

        let converter = EdlConverter::new();
        let result = converter
            .convert(&edl, EdlFormat::Cmx3400)
            .expect("conversion should succeed");
        assert_eq!(result.format, EdlFormat::Cmx3400);
        assert_eq!(result.events.len(), 1);
    }

    #[test]
    fn test_lossy_conversion() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");

        let mut event = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Wipe,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        event.set_transition_duration(30);
        event.set_wipe_pattern(crate::event::WipePattern::Horizontal);

        edl.add_event(event).expect("add_event should succeed");

        // Lossless conversion should fail
        let converter = EdlConverter::lossless();
        assert!(converter.convert(&edl, EdlFormat::Cmx3400).is_err());

        // Lossy conversion should succeed
        let converter = EdlConverter::lossy();
        let result = converter
            .convert(&edl, EdlFormat::Cmx3400)
            .expect("conversion should succeed");
        assert_eq!(result.format, EdlFormat::Cmx3400);
    }

    #[test]
    fn test_simplify() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");

        let mut event = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Dissolve,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        event.set_transition_duration(1); // Very short dissolve

        edl.add_event(event).expect("add_event should succeed");

        let converter = EdlConverter::lossy();
        let result = converter.simplify(&edl).expect("simplify should succeed");

        assert_eq!(result.events[0].edit_type, EditType::Cut);
        assert_eq!(result.events[0].transition_duration, None);
    }

    #[test]
    fn test_optimize() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        // Two adjacent cuts from the same reel
        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc3 = EdlTimecode::new(1, 0, 10, 0, EdlFrameRate::Fps25).expect("failed to create");

        let event1 = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        let event2 = EdlEvent::new(
            2,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc2,
            tc3,
            tc2,
            tc3,
        );

        edl.add_event(event1).expect("add_event should succeed");
        edl.add_event(event2).expect("add_event should succeed");

        let converter = EdlConverter::new();
        let result = converter.optimize(&edl).expect("optimize should succeed");

        // Should merge into a single event
        assert_eq!(result.events.len(), 1);
        assert_eq!(result.events[0].record_in, tc1);
        assert_eq!(result.events[0].record_out, tc3);
    }
}
