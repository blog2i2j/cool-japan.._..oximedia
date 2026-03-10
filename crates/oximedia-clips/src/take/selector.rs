//! Take types and selection.

use crate::clip::ClipId;
use crate::logging::Rating;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TakeId(Uuid);

impl TakeId {
    /// Creates a new random take ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Creates a take ID from a UUID.
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

impl Default for TakeId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TakeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A take represents one version of a shot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Take {
    /// Unique identifier.
    pub id: TakeId,

    /// Associated clip ID.
    pub clip_id: ClipId,

    /// Take number.
    pub take_number: u32,

    /// Scene/shot identifier.
    pub scene: String,

    /// Take name.
    pub name: String,

    /// Rating.
    pub rating: Rating,

    /// Is this the selected/best take?
    pub is_selected: bool,

    /// Is this take good (circled)?
    pub is_good: bool,

    /// Notes about this take.
    pub notes: Option<String>,

    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
}

impl Take {
    /// Creates a new take.
    #[must_use]
    pub fn new(clip_id: ClipId, scene: impl Into<String>, take_number: u32) -> Self {
        Self {
            id: TakeId::new(),
            clip_id,
            take_number,
            scene: scene.into(),
            name: format!("Take {take_number}"),
            rating: Rating::Unrated,
            is_selected: false,
            is_good: false,
            notes: None,
            created_at: Utc::now(),
        }
    }

    /// Sets the take as selected.
    pub fn set_selected(&mut self, selected: bool) {
        self.is_selected = selected;
    }

    /// Sets the take as good (circled).
    pub fn set_good(&mut self, good: bool) {
        self.is_good = good;
    }

    /// Sets the rating.
    pub fn set_rating(&mut self, rating: Rating) {
        self.rating = rating;
    }

    /// Sets notes.
    pub fn set_notes(&mut self, notes: impl Into<String>) {
        self.notes = Some(notes.into());
    }

    /// Sets the name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }
}

/// Selector for choosing the best take.
#[derive(Debug, Clone, Copy)]
pub enum TakeSelector {
    /// Highest rated take.
    HighestRated,
    /// Most recent take.
    MostRecent,
    /// First good take.
    FirstGood,
    /// Last good take.
    LastGood,
    /// Manually selected take.
    ManuallySelected,
}

impl TakeSelector {
    /// Selects the best take from a list.
    #[must_use]
    pub fn select<'a>(&self, takes: &'a [Take]) -> Option<&'a Take> {
        if takes.is_empty() {
            return None;
        }

        match self {
            Self::HighestRated => takes.iter().max_by_key(|t| t.rating),

            Self::MostRecent => takes.iter().max_by_key(|t| t.created_at),

            Self::FirstGood => takes.iter().find(|t| t.is_good),

            Self::LastGood => takes.iter().rfind(|t| t.is_good),

            Self::ManuallySelected => takes.iter().find(|t| t.is_selected),
        }
    }

    /// Selects all good takes.
    #[must_use]
    pub fn select_good(takes: &[Take]) -> Vec<&Take> {
        takes.iter().filter(|t| t.is_good).collect()
    }

    /// Selects takes by minimum rating.
    #[must_use]
    pub fn select_by_rating(takes: &[Take], min_rating: Rating) -> Vec<&Take> {
        takes.iter().filter(|t| t.rating >= min_rating).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_take_creation() {
        let clip_id = ClipId::new();
        let take = Take::new(clip_id, "Scene 1", 1);
        assert_eq!(take.take_number, 1);
        assert_eq!(take.scene, "Scene 1");
        assert!(!take.is_selected);
        assert!(!take.is_good);
    }

    #[test]
    fn test_take_selector() {
        let clip_id = ClipId::new();
        let mut take1 = Take::new(clip_id, "Scene 1", 1);
        let mut take2 = Take::new(clip_id, "Scene 1", 2);
        let mut take3 = Take::new(clip_id, "Scene 1", 3);

        take1.set_rating(Rating::ThreeStars);
        take2.set_rating(Rating::FiveStars);
        take3.set_rating(Rating::FourStars);
        take2.set_good(true);

        let takes = vec![take1.clone(), take2.clone(), take3.clone()];

        let best = TakeSelector::HighestRated
            .select(&takes)
            .expect("select should succeed");
        assert_eq!(best.take_number, 2);

        let first_good = TakeSelector::FirstGood
            .select(&takes)
            .expect("select should succeed");
        assert_eq!(first_good.take_number, 2);

        let good_takes = TakeSelector::select_good(&takes);
        assert_eq!(good_takes.len(), 1);
    }
}
