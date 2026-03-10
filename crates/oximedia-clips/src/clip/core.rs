//! Core clip data structures.

use crate::logging::Rating;
use crate::marker::Marker;
use chrono::{DateTime, Utc};
use oximedia_core::types::Rational;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// Unique identifier for a clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClipId(Uuid);

impl ClipId {
    /// Creates a new random clip ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Creates a clip ID from a UUID.
    #[must_use]
    pub const fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Returns the inner UUID.
    #[must_use]
    pub const fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for ClipId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ClipId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for ClipId {
    type Err = uuid::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(Uuid::parse_str(s)?))
    }
}

/// A video clip with metadata and logging information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Clip {
    /// Unique identifier.
    pub id: ClipId,

    /// File path to the media file.
    pub file_path: PathBuf,

    /// Display name.
    pub name: String,

    /// Optional description.
    pub description: Option<String>,

    /// Duration in frames.
    pub duration: Option<i64>,

    /// Frame rate.
    #[serde(skip)]
    pub frame_rate: Option<Rational>,

    /// In point (frame number).
    pub in_point: Option<i64>,

    /// Out point (frame number).
    pub out_point: Option<i64>,

    /// Star rating.
    pub rating: Rating,

    /// Is this clip marked as favorite?
    pub is_favorite: bool,

    /// Is this clip rejected?
    pub is_rejected: bool,

    /// Keywords associated with this clip.
    pub keywords: Vec<String>,

    /// Markers within this clip.
    pub markers: Vec<Marker>,

    /// Creation timestamp.
    pub created_at: DateTime<Utc>,

    /// Last modified timestamp.
    pub modified_at: DateTime<Utc>,

    /// Custom metadata as JSON.
    pub custom_metadata: Option<String>,
}

impl Clip {
    /// Creates a new clip from a file path.
    #[must_use]
    pub fn new(file_path: PathBuf) -> Self {
        let now = Utc::now();
        let name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("Untitled")
            .to_string();

        Self {
            id: ClipId::new(),
            file_path,
            name,
            description: None,
            duration: None,
            frame_rate: None,
            in_point: None,
            out_point: None,
            rating: Rating::Unrated,
            is_favorite: false,
            is_rejected: false,
            keywords: Vec::new(),
            markers: Vec::new(),
            created_at: now,
            modified_at: now,
            custom_metadata: None,
        }
    }

    /// Sets the clip name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
        self.modified_at = Utc::now();
    }

    /// Sets the clip description.
    pub fn set_description(&mut self, description: impl Into<String>) {
        self.description = Some(description.into());
        self.modified_at = Utc::now();
    }

    /// Sets the duration.
    pub fn set_duration(&mut self, duration: i64) {
        self.duration = Some(duration);
        self.modified_at = Utc::now();
    }

    /// Sets the frame rate.
    pub fn set_frame_rate(&mut self, frame_rate: Rational) {
        self.frame_rate = Some(frame_rate);
        self.modified_at = Utc::now();
    }

    /// Sets the in point.
    pub fn set_in_point(&mut self, in_point: i64) {
        self.in_point = Some(in_point);
        self.modified_at = Utc::now();
    }

    /// Sets the out point.
    pub fn set_out_point(&mut self, out_point: i64) {
        self.out_point = Some(out_point);
        self.modified_at = Utc::now();
    }

    /// Sets the rating.
    pub fn set_rating(&mut self, rating: Rating) {
        self.rating = rating;
        self.modified_at = Utc::now();
    }

    /// Marks the clip as favorite.
    pub fn set_favorite(&mut self, is_favorite: bool) {
        self.is_favorite = is_favorite;
        self.modified_at = Utc::now();
    }

    /// Marks the clip as rejected.
    pub fn set_rejected(&mut self, is_rejected: bool) {
        self.is_rejected = is_rejected;
        self.modified_at = Utc::now();
    }

    /// Adds a keyword.
    pub fn add_keyword(&mut self, keyword: impl Into<String>) {
        let keyword = keyword.into();
        if !self.keywords.contains(&keyword) {
            self.keywords.push(keyword);
            self.modified_at = Utc::now();
        }
    }

    /// Removes a keyword.
    pub fn remove_keyword(&mut self, keyword: &str) {
        if let Some(pos) = self.keywords.iter().position(|k| k == keyword) {
            self.keywords.remove(pos);
            self.modified_at = Utc::now();
        }
    }

    /// Adds a marker.
    pub fn add_marker(&mut self, marker: Marker) {
        self.markers.push(marker);
        self.modified_at = Utc::now();
    }

    /// Removes a marker by ID.
    pub fn remove_marker(&mut self, marker_id: &crate::marker::MarkerId) {
        if let Some(pos) = self.markers.iter().position(|m| &m.id == marker_id) {
            self.markers.remove(pos);
            self.modified_at = Utc::now();
        }
    }

    /// Returns the effective duration (considering in/out points).
    #[must_use]
    pub fn effective_duration(&self) -> Option<i64> {
        match (self.in_point, self.out_point, self.duration) {
            (Some(in_p), Some(out), _) => Some(out - in_p),
            (Some(in_p), None, Some(dur)) => Some(dur - in_p),
            (None, Some(out), _) => Some(out),
            (None, None, Some(dur)) => Some(dur),
            _ => None,
        }
    }

    /// Checks if the clip has valid in/out points.
    #[must_use]
    pub fn has_valid_range(&self) -> bool {
        match (self.in_point, self.out_point) {
            (Some(in_p), Some(out)) => in_p < out,
            _ => true,
        }
    }

    /// Returns whether the clip is fully logged (has rating and keywords).
    #[must_use]
    pub fn is_logged(&self) -> bool {
        self.rating != Rating::Unrated || !self.keywords.is_empty()
    }

    /// Returns whether the clip file exists.
    #[must_use]
    pub fn file_exists(&self) -> bool {
        self.file_path.exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_creation() {
        let clip = Clip::new(PathBuf::from("/test/video.mov"));
        assert_eq!(clip.name, "video.mov");
        assert_eq!(clip.rating, Rating::Unrated);
        assert!(!clip.is_favorite);
        assert!(!clip.is_rejected);
    }

    #[test]
    fn test_clip_keywords() {
        let mut clip = Clip::new(PathBuf::from("/test/video.mov"));
        clip.add_keyword("interview");
        clip.add_keyword("john-doe");
        assert_eq!(clip.keywords.len(), 2);

        clip.add_keyword("interview"); // Duplicate
        assert_eq!(clip.keywords.len(), 2);

        clip.remove_keyword("john-doe");
        assert_eq!(clip.keywords.len(), 1);
    }

    #[test]
    fn test_effective_duration() {
        let mut clip = Clip::new(PathBuf::from("/test/video.mov"));
        assert_eq!(clip.effective_duration(), None);

        clip.set_duration(1000);
        assert_eq!(clip.effective_duration(), Some(1000));

        clip.set_in_point(100);
        clip.set_out_point(500);
        assert_eq!(clip.effective_duration(), Some(400));
    }

    #[test]
    fn test_clip_id() {
        let id1 = ClipId::new();
        let id2 = ClipId::new();
        assert_ne!(id1, id2);

        let id_str = id1.to_string();
        let id_parsed: ClipId = id_str.parse().expect("parse should succeed");
        assert_eq!(id1, id_parsed);
    }
}
