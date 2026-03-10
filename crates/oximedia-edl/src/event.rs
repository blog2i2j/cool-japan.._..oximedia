//! EDL event types and related structures.
//!
//! This module defines the core event structures used in EDL files,
//! including cuts, dissolves, wipes, and keys.

use crate::audio::AudioChannel;
use crate::error::{EdlError, EdlResult};
use crate::motion::MotionEffect;
use crate::timecode::EdlTimecode;
use std::fmt;
use std::str::FromStr;

/// EDL event representing a single edit in the timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct EdlEvent {
    /// Event number (sequential, starting from 1).
    pub number: u32,

    /// Reel name or source identifier.
    pub reel: String,

    /// Track type (video, audio, or combined).
    pub track: TrackType,

    /// Edit type (cut, dissolve, wipe, key).
    pub edit_type: EditType,

    /// Source in timecode.
    pub source_in: EdlTimecode,

    /// Source out timecode.
    pub source_out: EdlTimecode,

    /// Record in timecode.
    pub record_in: EdlTimecode,

    /// Record out timecode.
    pub record_out: EdlTimecode,

    /// Optional transition duration (in frames).
    pub transition_duration: Option<u32>,

    /// Optional motion effect.
    pub motion_effect: Option<MotionEffect>,

    /// Optional clip name or comment.
    pub clip_name: Option<String>,

    /// Additional comments.
    pub comments: Vec<String>,

    /// Wipe pattern (if edit_type is Wipe).
    pub wipe_pattern: Option<WipePattern>,

    /// Key type (if edit_type is Key).
    pub key_type: Option<KeyType>,
}

impl EdlEvent {
    /// Create a new EDL event.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        number: u32,
        reel: String,
        track: TrackType,
        edit_type: EditType,
        source_in: EdlTimecode,
        source_out: EdlTimecode,
        record_in: EdlTimecode,
        record_out: EdlTimecode,
    ) -> Self {
        Self {
            number,
            reel,
            track,
            edit_type,
            source_in,
            source_out,
            record_in,
            record_out,
            transition_duration: None,
            motion_effect: None,
            clip_name: None,
            comments: Vec::new(),
            wipe_pattern: None,
            key_type: None,
        }
    }

    /// Validate the event for consistency.
    ///
    /// # Errors
    ///
    /// Returns an error if the event has invalid timecode ranges.
    pub fn validate(&self) -> EdlResult<()> {
        // Validate source range
        if self.source_in >= self.source_out {
            return Err(EdlError::InvalidSourceRange);
        }

        // Validate record range
        if self.record_in >= self.record_out {
            return Err(EdlError::InvalidRecordRange);
        }

        // Validate transition duration for dissolves and wipes
        if matches!(self.edit_type, EditType::Dissolve | EditType::Wipe)
            && self.transition_duration.is_none()
        {
            return Err(EdlError::MissingField("transition_duration".to_string()));
        }

        // Validate wipe pattern for wipes
        if self.edit_type == EditType::Wipe && self.wipe_pattern.is_none() {
            return Err(EdlError::MissingField("wipe_pattern".to_string()));
        }

        // Validate key type for keys
        if self.edit_type == EditType::Key && self.key_type.is_none() {
            return Err(EdlError::MissingField("key_type".to_string()));
        }

        Ok(())
    }

    /// Get the duration of the event in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.record_out.to_frames() - self.record_in.to_frames()
    }

    /// Check if this event overlaps with another event.
    #[must_use]
    pub fn overlaps_with(&self, other: &Self) -> bool {
        // Check if the tracks are compatible
        if !self.track.overlaps_with(&other.track) {
            return false;
        }

        // Check for timecode overlap
        !(self.record_out <= other.record_in || self.record_in >= other.record_out)
    }

    /// Add a comment to the event.
    pub fn add_comment(&mut self, comment: String) {
        self.comments.push(comment);
    }

    /// Set the clip name.
    pub fn set_clip_name(&mut self, name: String) {
        self.clip_name = Some(name);
    }

    /// Set the motion effect.
    pub fn set_motion_effect(&mut self, effect: MotionEffect) {
        self.motion_effect = Some(effect);
    }

    /// Set the transition duration.
    pub fn set_transition_duration(&mut self, duration: u32) {
        self.transition_duration = Some(duration);
    }

    /// Set the wipe pattern.
    pub fn set_wipe_pattern(&mut self, pattern: WipePattern) {
        self.wipe_pattern = Some(pattern);
    }

    /// Set the key type.
    pub fn set_key_type(&mut self, key_type: KeyType) {
        self.key_type = Some(key_type);
    }
}

/// Track type for EDL events.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TrackType {
    /// Video track.
    Video,
    /// Single audio track (A, A2, A3, A4, etc.).
    Audio(AudioChannel),
    /// Audio pair (AA, AA/V).
    AudioPair,
    /// Audio with video (A/V).
    AudioWithVideo,
    /// Audio pair with video (AA/V).
    AudioPairWithVideo,
    /// Multiple audio channels.
    AudioMulti(Vec<AudioChannel>),
    /// Video with multiple audio channels.
    VideoWithAudioMulti(Vec<AudioChannel>),
}

impl TrackType {
    /// Check if this track type includes video.
    #[must_use]
    pub const fn has_video(&self) -> bool {
        matches!(
            self,
            Self::Video
                | Self::AudioWithVideo
                | Self::AudioPairWithVideo
                | Self::VideoWithAudioMulti(_)
        )
    }

    /// Check if this track type includes audio.
    #[must_use]
    pub const fn has_audio(&self) -> bool {
        matches!(
            self,
            Self::Audio(_)
                | Self::AudioPair
                | Self::AudioWithVideo
                | Self::AudioPairWithVideo
                | Self::AudioMulti(_)
                | Self::VideoWithAudioMulti(_)
        )
    }

    /// Check if this track type overlaps with another track type.
    #[must_use]
    pub fn overlaps_with(&self, other: &Self) -> bool {
        match (self, other) {
            // Video tracks overlap with other video tracks
            (Self::Video, Self::Video) => true,
            // Audio tracks overlap if they share channels
            (Self::Audio(ch1), Self::Audio(ch2)) => ch1 == ch2,
            // AudioPair overlaps with AudioPair
            (Self::AudioPair, Self::AudioPair) => true,
            // AudioWithVideo overlaps with video or audio
            (Self::AudioWithVideo, Self::Video) | (Self::Video, Self::AudioWithVideo) => true,
            (Self::AudioWithVideo, Self::Audio(_)) | (Self::Audio(_), Self::AudioWithVideo) => true,
            // AudioPairWithVideo overlaps with video or audio
            (Self::AudioPairWithVideo, Self::Video) | (Self::Video, Self::AudioPairWithVideo) => {
                true
            }
            (Self::AudioPairWithVideo, Self::AudioPair)
            | (Self::AudioPair, Self::AudioPairWithVideo) => true,
            // Multi-channel overlaps require detailed checking
            _ => false,
        }
    }
}

impl FromStr for TrackType {
    type Err = EdlError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_uppercase().as_str() {
            "V" => Ok(Self::Video),
            "A" => Ok(Self::Audio(AudioChannel::A1)),
            "A2" => Ok(Self::Audio(AudioChannel::A2)),
            "A3" => Ok(Self::Audio(AudioChannel::A3)),
            "A4" => Ok(Self::Audio(AudioChannel::A4)),
            "AA" => Ok(Self::AudioPair),
            "A/V" => Ok(Self::AudioWithVideo),
            "AA/V" => Ok(Self::AudioPairWithVideo),
            _ => Err(EdlError::InvalidTrackType(s.to_string())),
        }
    }
}

impl fmt::Display for TrackType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Video => "V",
            Self::Audio(AudioChannel::A1) => "A",
            Self::Audio(AudioChannel::A2) => "A2",
            Self::Audio(AudioChannel::A3) => "A3",
            Self::Audio(AudioChannel::A4) => "A4",
            Self::Audio(_) => "A",
            Self::AudioPair => "AA",
            Self::AudioWithVideo => "A/V",
            Self::AudioPairWithVideo => "AA/V",
            Self::AudioMulti(_) => "A",
            Self::VideoWithAudioMulti(_) => "V",
        };
        write!(f, "{s}")
    }
}

/// Edit type for EDL events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EditType {
    /// Cut (instantaneous transition).
    Cut,
    /// Dissolve (cross-fade).
    Dissolve,
    /// Wipe (geometric transition).
    Wipe,
    /// Key (compositing).
    Key,
}

impl FromStr for EditType {
    type Err = EdlError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_uppercase().as_str() {
            "C" => Ok(Self::Cut),
            "D" => Ok(Self::Dissolve),
            "W" => Ok(Self::Wipe),
            "K" => Ok(Self::Key),
            _ => Err(EdlError::InvalidEditType(s.to_string())),
        }
    }
}

impl fmt::Display for EditType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Cut => "C",
            Self::Dissolve => "D",
            Self::Wipe => "W",
            Self::Key => "K",
        };
        write!(f, "{s}")
    }
}

/// Wipe pattern for wipe transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WipePattern {
    /// Horizontal wipe (left to right).
    Horizontal,
    /// Vertical wipe (top to bottom).
    Vertical,
    /// Diagonal wipe (top-left to bottom-right).
    Diagonal,
    /// Circle wipe (expanding or contracting).
    Circle,
    /// Custom wipe pattern with numeric code.
    Custom(u16),
}

impl FromStr for WipePattern {
    type Err = EdlError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_uppercase().as_str() {
            "HORIZONTAL" | "H" | "001" => Ok(Self::Horizontal),
            "VERTICAL" | "V" | "002" => Ok(Self::Vertical),
            "DIAGONAL" | "D" | "003" => Ok(Self::Diagonal),
            "CIRCLE" | "C" | "004" => Ok(Self::Circle),
            _ => {
                // Try to parse as numeric code
                if let Ok(code) = s.trim().parse::<u16>() {
                    Ok(Self::Custom(code))
                } else {
                    Err(EdlError::InvalidWipePattern(s.to_string()))
                }
            }
        }
    }
}

impl fmt::Display for WipePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Horizontal => "001",
            Self::Vertical => "002",
            Self::Diagonal => "003",
            Self::Circle => "004",
            Self::Custom(code) => return write!(f, "{code:03}"),
        };
        write!(f, "{s}")
    }
}

/// Key type for key/compositing events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KeyType {
    /// Luminance key.
    Luminance,
    /// Chroma key.
    Chroma,
    /// Alpha key.
    Alpha,
    /// Custom key type.
    Custom(u16),
}

impl FromStr for KeyType {
    type Err = EdlError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_uppercase().as_str() {
            "LUMINANCE" | "L" => Ok(Self::Luminance),
            "CHROMA" | "C" => Ok(Self::Chroma),
            "ALPHA" | "A" => Ok(Self::Alpha),
            _ => {
                // Try to parse as numeric code
                if let Ok(code) = s.trim().parse::<u16>() {
                    Ok(Self::Custom(code))
                } else {
                    Err(EdlError::InvalidKeyType(s.to_string()))
                }
            }
        }
    }
}

impl fmt::Display for KeyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Luminance => "Luminance",
            Self::Chroma => "Chroma",
            Self::Alpha => "Alpha",
            Self::Custom(code) => return write!(f, "{code}"),
        };
        write!(f, "{s}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timecode::EdlFrameRate;

    #[test]
    fn test_track_type_parsing() {
        assert_eq!(
            "V".parse::<TrackType>().expect("operation should succeed"),
            TrackType::Video
        );
        assert_eq!(
            "A".parse::<TrackType>().expect("operation should succeed"),
            TrackType::Audio(AudioChannel::A1)
        );
        assert_eq!(
            "AA".parse::<TrackType>().expect("operation should succeed"),
            TrackType::AudioPair
        );
        assert_eq!(
            "A/V"
                .parse::<TrackType>()
                .expect("operation should succeed"),
            TrackType::AudioWithVideo
        );
    }

    #[test]
    fn test_edit_type_parsing() {
        assert_eq!(
            "C".parse::<EditType>().expect("operation should succeed"),
            EditType::Cut
        );
        assert_eq!(
            "D".parse::<EditType>().expect("operation should succeed"),
            EditType::Dissolve
        );
        assert_eq!(
            "W".parse::<EditType>().expect("operation should succeed"),
            EditType::Wipe
        );
        assert_eq!(
            "K".parse::<EditType>().expect("operation should succeed"),
            EditType::Key
        );
    }

    #[test]
    fn test_wipe_pattern_parsing() {
        assert_eq!(
            "001"
                .parse::<WipePattern>()
                .expect("operation should succeed"),
            WipePattern::Horizontal
        );
        assert_eq!(
            "100"
                .parse::<WipePattern>()
                .expect("operation should succeed"),
            WipePattern::Custom(100)
        );
    }

    #[test]
    fn test_event_validation() {
        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 10, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc3 = EdlTimecode::new(1, 0, 20, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc4 = EdlTimecode::new(1, 0, 30, 0, EdlFrameRate::Fps25).expect("failed to create");

        let event = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc3,
            tc4,
        );

        assert!(event.validate().is_ok());
    }

    #[test]
    fn test_event_overlap_detection() {
        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 10, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc3 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc4 = EdlTimecode::new(1, 0, 15, 0, EdlFrameRate::Fps25).expect("failed to create");

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
            "A002".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc3,
            tc4,
            tc3,
            tc4,
        );

        assert!(event1.overlaps_with(&event2));
    }

    #[test]
    fn test_track_has_video() {
        assert!(TrackType::Video.has_video());
        assert!(!TrackType::Audio(AudioChannel::A1).has_video());
        assert!(TrackType::AudioWithVideo.has_video());
    }

    #[test]
    fn test_track_has_audio() {
        assert!(!TrackType::Video.has_audio());
        assert!(TrackType::Audio(AudioChannel::A1).has_audio());
        assert!(TrackType::AudioWithVideo.has_audio());
    }
}
