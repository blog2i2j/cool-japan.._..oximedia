//! Foley recording, editing, and library management.

use crate::error::{AudioPostError, AudioPostResult};
use crate::timecode::Timecode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Foley session for recording and managing foley sounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoleySession {
    /// Session ID
    pub id: Uuid,
    /// Session name
    pub name: String,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of recording tracks
    pub track_count: usize,
    /// Cue markers
    cues: Vec<FoleyCue>,
    /// Recorded takes
    takes: HashMap<usize, FoleyTake>,
    /// Next take ID
    next_take_id: usize,
}

impl FoleySession {
    /// Create a new foley session
    ///
    /// # Errors
    ///
    /// Returns an error if track count is invalid
    pub fn new(name: &str, sample_rate: u32, track_count: usize) -> AudioPostResult<Self> {
        if track_count == 0 || track_count > 8 {
            return Err(AudioPostError::InvalidChannelCount(track_count));
        }

        Ok(Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            sample_rate,
            track_count,
            cues: Vec::new(),
            takes: HashMap::new(),
            next_take_id: 1,
        })
    }

    /// Add a cue marker
    pub fn add_cue(&mut self, cue: FoleyCue) {
        self.cues.push(cue);
        self.cues.sort_by(|a, b| {
            a.timecode
                .partial_cmp(&b.timecode)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get all cues
    #[must_use]
    pub fn get_cues(&self) -> &[FoleyCue] {
        &self.cues
    }

    /// Add a take
    pub fn add_take(&mut self, take: FoleyTake) -> usize {
        let id = self.next_take_id;
        self.takes.insert(id, take);
        self.next_take_id += 1;
        id
    }

    /// Get a take
    ///
    /// # Errors
    ///
    /// Returns an error if the take is not found
    pub fn get_take(&self, id: usize) -> AudioPostResult<&FoleyTake> {
        self.takes.get(&id).ok_or(AudioPostError::TakeNotFound(id))
    }
}

/// Foley cue marker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoleyCue {
    /// Timecode
    pub timecode: Timecode,
    /// Description
    pub description: String,
    /// Category
    pub category: FoleyCategory,
}

impl FoleyCue {
    /// Create a new foley cue
    #[must_use]
    pub fn new(timecode: Timecode, description: &str, category: FoleyCategory) -> Self {
        Self {
            timecode,
            description: description.to_string(),
            category,
        }
    }
}

/// Foley category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FoleyCategory {
    /// Footsteps
    Footsteps,
    /// Cloth movement
    Cloth,
    /// Props handling
    Props,
    /// Body movement
    Body,
    /// Custom category
    Custom,
}

/// Foley take
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoleyTake {
    /// Take number
    pub number: u32,
    /// Audio file paths (one per track)
    pub audio_paths: Vec<String>,
    /// Category
    pub category: FoleyCategory,
    /// Performance notes
    pub notes: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl FoleyTake {
    /// Create a new foley take
    #[must_use]
    pub fn new(number: u32, audio_paths: Vec<String>, category: FoleyCategory) -> Self {
        Self {
            number,
            audio_paths,
            category,
            notes: String::new(),
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Foley library for managing sound effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoleyLibrary {
    /// Library name
    pub name: String,
    /// Sound effects in the library
    effects: HashMap<Uuid, FoleyEffect>,
    /// Category index
    category_index: HashMap<FoleyCategory, Vec<Uuid>>,
}

impl FoleyLibrary {
    /// Create a new foley library
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            effects: HashMap::new(),
            category_index: HashMap::new(),
        }
    }

    /// Add an effect to the library
    pub fn add_effect(&mut self, effect: FoleyEffect) -> Uuid {
        let id = effect.id;
        let category = effect.category;

        self.effects.insert(id, effect);

        self.category_index.entry(category).or_default().push(id);

        id
    }

    /// Get an effect by ID
    #[must_use]
    pub fn get_effect(&self, id: &Uuid) -> Option<&FoleyEffect> {
        self.effects.get(id)
    }

    /// Search effects by category
    #[must_use]
    pub fn search_by_category(&self, category: FoleyCategory) -> Vec<&FoleyEffect> {
        self.category_index
            .get(&category)
            .map_or_else(Vec::new, |ids| {
                ids.iter().filter_map(|id| self.effects.get(id)).collect()
            })
    }

    /// Search effects by tag
    #[must_use]
    pub fn search_by_tag(&self, tag: &str) -> Vec<&FoleyEffect> {
        self.effects
            .values()
            .filter(|effect| effect.tags.contains(&tag.to_string()))
            .collect()
    }

    /// Get all effects
    #[must_use]
    pub fn get_all_effects(&self) -> Vec<&FoleyEffect> {
        self.effects.values().collect()
    }
}

/// Foley sound effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoleyEffect {
    /// Effect ID
    pub id: Uuid,
    /// Effect name
    pub name: String,
    /// Category
    pub category: FoleyCategory,
    /// Audio file path
    pub audio_path: String,
    /// Duration in seconds
    pub duration: f64,
    /// Tags for searching
    pub tags: Vec<String>,
    /// Metadata
    pub metadata: FoleyMetadata,
}

impl FoleyEffect {
    /// Create a new foley effect
    #[must_use]
    pub fn new(name: &str, category: FoleyCategory, audio_path: &str, duration: f64) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            category,
            audio_path: audio_path.to_string(),
            duration,
            tags: Vec::new(),
            metadata: FoleyMetadata::default(),
        }
    }

    /// Add a tag
    pub fn add_tag(&mut self, tag: &str) {
        if !self.tags.contains(&tag.to_string()) {
            self.tags.push(tag.to_string());
        }
    }
}

/// Foley effect metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FoleyMetadata {
    /// Surface type (for footsteps)
    pub surface: Option<String>,
    /// Intensity (1-10)
    pub intensity: Option<u8>,
    /// Recording date
    pub date: Option<chrono::DateTime<chrono::Utc>>,
    /// Microphone used
    pub microphone: Option<String>,
    /// Additional notes
    pub notes: String,
}

/// Footstep surface types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FootstepSurface {
    /// Concrete
    Concrete,
    /// Wood
    Wood,
    /// Gravel
    Gravel,
    /// Grass
    Grass,
    /// Metal
    Metal,
    /// Carpet
    Carpet,
    /// Tile
    Tile,
    /// Snow
    Snow,
    /// Mud
    Mud,
    /// Water
    Water,
}

impl FootstepSurface {
    /// Get all surface types
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::Concrete,
            Self::Wood,
            Self::Gravel,
            Self::Grass,
            Self::Metal,
            Self::Carpet,
            Self::Tile,
            Self::Snow,
            Self::Mud,
            Self::Water,
        ]
    }

    /// Get surface name as string
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Concrete => "Concrete",
            Self::Wood => "Wood",
            Self::Gravel => "Gravel",
            Self::Grass => "Grass",
            Self::Metal => "Metal",
            Self::Carpet => "Carpet",
            Self::Tile => "Tile",
            Self::Snow => "Snow",
            Self::Mud => "Mud",
            Self::Water => "Water",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foley_session_creation() {
        let session = FoleySession::new("Scene 1", 48000, 4).expect("failed to create");
        assert_eq!(session.name, "Scene 1");
        assert_eq!(session.sample_rate, 48000);
        assert_eq!(session.track_count, 4);
    }

    #[test]
    fn test_invalid_track_count() {
        assert!(FoleySession::new("Scene 1", 48000, 0).is_err());
        assert!(FoleySession::new("Scene 1", 48000, 9).is_err());
    }

    #[test]
    fn test_add_cue() {
        let mut session = FoleySession::new("Scene 1", 48000, 4).expect("failed to create");
        let cue = FoleyCue::new(
            Timecode::from_frames(1000, 24.0),
            "Footstep",
            FoleyCategory::Footsteps,
        );
        session.add_cue(cue);
        assert_eq!(session.get_cues().len(), 1);
    }

    #[test]
    fn test_add_take() {
        let mut session = FoleySession::new("Scene 1", 48000, 4).expect("failed to create");
        let take = FoleyTake::new(
            1,
            vec!["track1.wav".to_string(), "track2.wav".to_string()],
            FoleyCategory::Footsteps,
        );
        let id = session.add_take(take);
        assert_eq!(id, 1);
        assert!(session.get_take(id).is_ok());
    }

    #[test]
    fn test_foley_library() {
        let mut library = FoleyLibrary::new("My Library");
        let effect = FoleyEffect::new(
            "Footstep Wood",
            FoleyCategory::Footsteps,
            "footstep.wav",
            0.5,
        );
        let id = library.add_effect(effect);
        assert!(library.get_effect(&id).is_some());
    }

    #[test]
    fn test_search_by_category() {
        let mut library = FoleyLibrary::new("My Library");
        let effect1 = FoleyEffect::new(
            "Footstep Wood",
            FoleyCategory::Footsteps,
            "footstep.wav",
            0.5,
        );
        let effect2 = FoleyEffect::new("Door Close", FoleyCategory::Props, "door.wav", 1.0);
        library.add_effect(effect1);
        library.add_effect(effect2);

        let footsteps = library.search_by_category(FoleyCategory::Footsteps);
        assert_eq!(footsteps.len(), 1);
    }

    #[test]
    fn test_search_by_tag() {
        let mut library = FoleyLibrary::new("My Library");
        let mut effect = FoleyEffect::new(
            "Footstep Wood",
            FoleyCategory::Footsteps,
            "footstep.wav",
            0.5,
        );
        effect.add_tag("wood");
        effect.add_tag("heavy");
        library.add_effect(effect);

        let results = library.search_by_tag("wood");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_footstep_surfaces() {
        let surfaces = FootstepSurface::all();
        assert_eq!(surfaces.len(), 10);
        assert_eq!(FootstepSurface::Wood.as_str(), "Wood");
    }

    #[test]
    fn test_foley_effect_tags() {
        let mut effect = FoleyEffect::new(
            "Footstep Wood",
            FoleyCategory::Footsteps,
            "footstep.wav",
            0.5,
        );
        effect.add_tag("wood");
        effect.add_tag("heavy");
        effect.add_tag("wood"); // Duplicate should be ignored
        assert_eq!(effect.tags.len(), 2);
    }

    #[test]
    fn test_cue_sorting() {
        let mut session = FoleySession::new("Scene 1", 48000, 4).expect("failed to create");

        let cue1 = FoleyCue::new(
            Timecode::from_frames(2000, 24.0),
            "Cue 2",
            FoleyCategory::Footsteps,
        );
        let cue2 = FoleyCue::new(
            Timecode::from_frames(1000, 24.0),
            "Cue 1",
            FoleyCategory::Footsteps,
        );

        session.add_cue(cue1);
        session.add_cue(cue2);

        let cues = session.get_cues();
        assert_eq!(cues[0].description, "Cue 1");
        assert_eq!(cues[1].description, "Cue 2");
    }
}
