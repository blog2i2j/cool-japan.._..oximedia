//! Professional logging interface for clips.

use crate::clip::{Clip, ClipId};
use crate::logging::Rating;
use crate::marker::Marker;
use crate::note::Note;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Professional logging session for clips.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logger {
    /// Logging session ID.
    pub session_id: String,

    /// Session start time.
    pub started_at: DateTime<Utc>,

    /// User performing the logging.
    pub user: Option<String>,

    /// Logged clips.
    logs: HashMap<ClipId, LogEntry>,
}

/// A single log entry for a clip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// When this clip was logged.
    pub logged_at: DateTime<Utc>,

    /// Rating assigned.
    pub rating: Rating,

    /// Keywords assigned.
    pub keywords: Vec<String>,

    /// Markers added.
    pub markers: Vec<Marker>,

    /// Notes added.
    pub notes: Vec<Note>,

    /// Whether the clip was marked as favorite.
    pub is_favorite: bool,

    /// Whether the clip was rejected.
    pub is_rejected: bool,
}

impl Logger {
    /// Creates a new logging session.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            started_at: Utc::now(),
            user: None,
            logs: HashMap::new(),
        }
    }

    /// Sets the user for this logging session.
    pub fn set_user(&mut self, user: impl Into<String>) {
        self.user = Some(user.into());
    }

    /// Logs a clip with rating and keywords.
    #[allow(clippy::too_many_arguments)]
    pub fn log_clip(
        &mut self,
        clip_id: ClipId,
        rating: Rating,
        keywords: Vec<String>,
        markers: Vec<Marker>,
        notes: Vec<Note>,
        is_favorite: bool,
        is_rejected: bool,
    ) {
        let entry = LogEntry {
            logged_at: Utc::now(),
            rating,
            keywords,
            markers,
            notes,
            is_favorite,
            is_rejected,
        };

        self.logs.insert(clip_id, entry);
    }

    /// Adds a rating to a clip.
    pub fn add_rating(&mut self, clip_id: ClipId, rating: Rating) {
        self.logs
            .entry(clip_id)
            .or_insert_with(|| LogEntry {
                logged_at: Utc::now(),
                rating: Rating::Unrated,
                keywords: Vec::new(),
                markers: Vec::new(),
                notes: Vec::new(),
                is_favorite: false,
                is_rejected: false,
            })
            .rating = rating;
    }

    /// Adds a keyword to a clip.
    pub fn add_keyword(&mut self, clip_id: ClipId, keyword: impl Into<String>) {
        let entry = self.logs.entry(clip_id).or_insert_with(|| LogEntry {
            logged_at: Utc::now(),
            rating: Rating::Unrated,
            keywords: Vec::new(),
            markers: Vec::new(),
            notes: Vec::new(),
            is_favorite: false,
            is_rejected: false,
        });

        let keyword = keyword.into();
        if !entry.keywords.contains(&keyword) {
            entry.keywords.push(keyword);
        }
    }

    /// Adds a marker to a clip.
    pub fn add_marker(&mut self, clip_id: ClipId, marker: Marker) {
        self.logs
            .entry(clip_id)
            .or_insert_with(|| LogEntry {
                logged_at: Utc::now(),
                rating: Rating::Unrated,
                keywords: Vec::new(),
                markers: Vec::new(),
                notes: Vec::new(),
                is_favorite: false,
                is_rejected: false,
            })
            .markers
            .push(marker);
    }

    /// Gets the log entry for a clip.
    #[must_use]
    pub fn get_log(&self, clip_id: &ClipId) -> Option<&LogEntry> {
        self.logs.get(clip_id)
    }

    /// Returns all logged clips.
    #[must_use]
    pub fn logged_clips(&self) -> Vec<ClipId> {
        self.logs.keys().copied().collect()
    }

    /// Returns the number of logged clips.
    #[must_use]
    pub fn count(&self) -> usize {
        self.logs.len()
    }

    /// Applies this logging session to clips.
    pub fn apply_to_clips(&self, clips: &mut HashMap<ClipId, Clip>) {
        for (clip_id, entry) in &self.logs {
            if let Some(clip) = clips.get_mut(clip_id) {
                clip.set_rating(entry.rating);
                clip.set_favorite(entry.is_favorite);
                clip.set_rejected(entry.is_rejected);

                for keyword in &entry.keywords {
                    clip.add_keyword(keyword.clone());
                }

                for marker in &entry.markers {
                    clip.add_marker(marker.clone());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logger_creation() {
        let logger = Logger::new("session-001");
        assert_eq!(logger.session_id, "session-001");
        assert_eq!(logger.count(), 0);
    }

    #[test]
    fn test_log_clip() {
        let mut logger = Logger::new("session-001");
        let clip_id = ClipId::new();

        logger.add_rating(clip_id, Rating::FourStars);
        logger.add_keyword(clip_id, "interview");
        logger.add_keyword(clip_id, "john-doe");

        let entry = logger.get_log(&clip_id).expect("get_log should succeed");
        assert_eq!(entry.rating, Rating::FourStars);
        assert_eq!(entry.keywords.len(), 2);
        assert_eq!(logger.count(), 1);
    }
}
