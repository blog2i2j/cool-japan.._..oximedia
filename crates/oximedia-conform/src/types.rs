//! Common types for oximedia-conform.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Frame rate representation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FrameRate {
    /// 23.976 fps (24000/1001).
    Fps23976,
    /// 24 fps.
    Fps24,
    /// 25 fps (PAL).
    Fps25,
    /// 29.97 fps (30000/1001) drop-frame.
    Fps2997DF,
    /// 29.97 fps (30000/1001) non-drop-frame.
    Fps2997NDF,
    /// 30 fps.
    Fps30,
    /// 50 fps.
    Fps50,
    /// 59.94 fps (60000/1001).
    Fps5994,
    /// 60 fps.
    Fps60,
    /// Custom frame rate.
    Custom(f64),
}

impl FrameRate {
    /// Get the frame rate as a floating-point number.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        match self {
            Self::Fps23976 => 24000.0 / 1001.0,
            Self::Fps24 => 24.0,
            Self::Fps25 => 25.0,
            Self::Fps2997DF | Self::Fps2997NDF => 30000.0 / 1001.0,
            Self::Fps30 => 30.0,
            Self::Fps50 => 50.0,
            Self::Fps5994 => 60000.0 / 1001.0,
            Self::Fps60 => 60.0,
            Self::Custom(fps) => fps,
        }
    }

    /// Get the frame rate as a rational (numerator, denominator).
    #[must_use]
    pub const fn as_rational(self) -> (u32, u32) {
        match self {
            Self::Fps23976 => (24000, 1001),
            Self::Fps24 => (24, 1),
            Self::Fps25 => (25, 1),
            Self::Fps2997DF | Self::Fps2997NDF => (30000, 1001),
            Self::Fps30 => (30, 1),
            Self::Fps50 => (50, 1),
            Self::Fps5994 => (60000, 1001),
            Self::Fps60 => (60, 1),
            Self::Custom(_) => (0, 0), // Not representable
        }
    }

    /// Check if this is a drop-frame rate.
    #[must_use]
    pub const fn is_drop_frame(self) -> bool {
        matches!(self, Self::Fps2997DF)
    }
}

/// Timecode representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Timecode {
    /// Hours (0-23).
    pub hours: u8,
    /// Minutes (0-59).
    pub minutes: u8,
    /// Seconds (0-59).
    pub seconds: u8,
    /// Frames (0-fps-1).
    pub frames: u8,
}

impl Timecode {
    /// Create a new timecode.
    #[must_use]
    pub const fn new(hours: u8, minutes: u8, seconds: u8, frames: u8) -> Self {
        Self {
            hours,
            minutes,
            seconds,
            frames,
        }
    }

    /// Convert to total frames.
    #[must_use]
    pub fn to_frames(self, fps: FrameRate) -> u64 {
        let fps_int = fps.as_f64() as u64;
        let hours_frames = u64::from(self.hours) * 3600 * fps_int;
        let minutes_frames = u64::from(self.minutes) * 60 * fps_int;
        let seconds_frames = u64::from(self.seconds) * fps_int;
        hours_frames + minutes_frames + seconds_frames + u64::from(self.frames)
    }

    /// Create from total frames.
    #[must_use]
    pub fn from_frames(frames: u64, fps: FrameRate) -> Self {
        let fps_int = fps.as_f64() as u64;
        let hours = (frames / (3600 * fps_int)) as u8;
        let remaining = frames % (3600 * fps_int);
        let minutes = (remaining / (60 * fps_int)) as u8;
        let remaining = remaining % (60 * fps_int);
        let seconds = (remaining / fps_int) as u8;
        let frames = (remaining % fps_int) as u8;
        Self::new(hours, minutes, seconds, frames)
    }

    /// Parse from string in format "HH:MM:SS:FF".
    ///
    /// # Errors
    ///
    /// Returns an error if the format is invalid.
    pub fn parse(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 4 {
            return Err(format!("Invalid timecode format: {s}"));
        }

        let hours = parts[0]
            .parse::<u8>()
            .map_err(|_| format!("Invalid hours: {}", parts[0]))?;
        let minutes = parts[1]
            .parse::<u8>()
            .map_err(|_| format!("Invalid minutes: {}", parts[1]))?;
        let seconds = parts[2]
            .parse::<u8>()
            .map_err(|_| format!("Invalid seconds: {}", parts[2]))?;
        let frames = parts[3]
            .parse::<u8>()
            .map_err(|_| format!("Invalid frames: {}", parts[3]))?;

        Ok(Self::new(hours, minutes, seconds, frames))
    }
}

impl std::fmt::Display for Timecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:02}:{:02}:{:02}:{:02}",
            self.hours, self.minutes, self.seconds, self.frames
        )
    }
}

/// Media file reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaFile {
    /// Unique identifier.
    pub id: uuid::Uuid,
    /// File path.
    pub path: PathBuf,
    /// File name.
    pub filename: String,
    /// Duration in seconds.
    pub duration: Option<f64>,
    /// Start timecode.
    pub timecode_start: Option<Timecode>,
    /// Width in pixels.
    pub width: Option<u32>,
    /// Height in pixels.
    pub height: Option<u32>,
    /// Frame rate.
    pub fps: Option<FrameRate>,
    /// File size in bytes.
    pub size: Option<u64>,
    /// MD5 checksum.
    pub md5: Option<String>,
    /// `XXHash` checksum.
    pub xxhash: Option<String>,
    /// Additional metadata (JSON).
    pub metadata: Option<String>,
    /// When the file was cataloged.
    pub cataloged_at: chrono::DateTime<chrono::Utc>,
}

impl MediaFile {
    /// Create a new media file reference.
    #[must_use]
    pub fn new(path: PathBuf) -> Self {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        Self {
            id: uuid::Uuid::new_v4(),
            path,
            filename,
            duration: None,
            timecode_start: None,
            width: None,
            height: None,
            fps: None,
            size: None,
            md5: None,
            xxhash: None,
            metadata: None,
            cataloged_at: chrono::Utc::now(),
        }
    }
}

/// Clip reference from EDL/XML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipReference {
    /// Clip identifier (reel name, clip name, etc.).
    pub id: String,
    /// Source file name or pattern.
    pub source_file: Option<String>,
    /// Source in timecode.
    pub source_in: Timecode,
    /// Source out timecode.
    pub source_out: Timecode,
    /// Record in timecode.
    pub record_in: Timecode,
    /// Record out timecode.
    pub record_out: Timecode,
    /// Track type (video, audio).
    pub track: TrackType,
    /// Frame rate.
    pub fps: FrameRate,
    /// Additional metadata.
    pub metadata: std::collections::HashMap<String, String>,
}

/// Track type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackType {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
    /// Both audio and video.
    AudioVideo,
}

/// Clip match result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipMatch {
    /// The clip reference being matched.
    pub clip: ClipReference,
    /// The matched media file.
    pub media: MediaFile,
    /// Match score (0.0 - 1.0).
    pub score: f64,
    /// Match method used.
    pub method: MatchMethod,
    /// Additional match details.
    pub details: String,
}

/// Match method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchMethod {
    /// Exact filename match.
    ExactFilename,
    /// Fuzzy filename match.
    FuzzyFilename,
    /// Timecode match.
    Timecode,
    /// Content hash match.
    ContentHash,
    /// Duration match.
    Duration,
    /// Combined match (multiple methods).
    Combined,
}

impl std::fmt::Display for MatchMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::ExactFilename => "Exact Filename",
            Self::FuzzyFilename => "Fuzzy Filename",
            Self::Timecode => "Timecode",
            Self::ContentHash => "Content Hash",
            Self::Duration => "Duration",
            Self::Combined => "Combined",
        };
        write!(f, "{s}")
    }
}

/// Output format for conformed sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Concatenated MP4 file.
    Mp4,
    /// Matroska container.
    Matroska,
    /// Frame sequence (DPX).
    FrameSequenceDpx,
    /// Frame sequence (TIFF).
    FrameSequenceTiff,
    /// Frame sequence (PNG).
    FrameSequencePng,
    /// EDL with updated paths.
    Edl,
    /// Final Cut Pro XML.
    FcpXml,
    /// Adobe Premiere XML.
    PremiereXml,
    /// AAF file.
    Aaf,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Mp4 => "MP4",
            Self::Matroska => "Matroska",
            Self::FrameSequenceDpx => "DPX Sequence",
            Self::FrameSequenceTiff => "TIFF Sequence",
            Self::FrameSequencePng => "PNG Sequence",
            Self::Edl => "EDL",
            Self::FcpXml => "FCP XML",
            Self::PremiereXml => "Premiere XML",
            Self::Aaf => "AAF",
        };
        write!(f, "{s}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_rate_conversion() {
        assert!((FrameRate::Fps25.as_f64() - 25.0).abs() < f64::EPSILON);
        assert!((FrameRate::Fps2997NDF.as_f64() - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_timecode_frames_conversion() {
        let tc = Timecode::new(1, 0, 0, 0);
        let frames = tc.to_frames(FrameRate::Fps25);
        assert_eq!(frames, 90000); // 1 hour * 3600 seconds * 25 fps

        let tc2 = Timecode::from_frames(frames, FrameRate::Fps25);
        assert_eq!(tc, tc2);
    }

    #[test]
    fn test_timecode_parsing() {
        let tc = Timecode::parse("01:23:45:12").expect("tc should be valid");
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 23);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 12);
    }

    #[test]
    fn test_timecode_display() {
        let tc = Timecode::new(1, 23, 45, 12);
        assert_eq!(tc.to_string(), "01:23:45:12");
    }

    #[test]
    fn test_timecode_comparison() {
        let tc1 = Timecode::new(1, 0, 0, 0);
        let tc2 = Timecode::new(1, 0, 0, 1);
        assert!(tc1 < tc2);
    }
}
