//! Mixer session management.
//!
//! Provides session save/load, templates, scenes, and undo/redo functionality.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

use crate::{
    automation::AutomationSnapshot, bus::BusGroup, AutomationData, Bus, BusId, Channel, ChannelId,
};

/// Session version for compatibility.
pub const SESSION_VERSION: u32 = 1;

/// Maximum undo history size.
pub const MAX_UNDO_HISTORY: usize = 100;

/// Mixer session data (serializable state).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    /// Session version.
    pub version: u32,

    /// Session name.
    pub name: String,

    /// Session ID.
    pub id: Uuid,

    /// Creation timestamp (Unix epoch).
    pub created_at: u64,

    /// Last modified timestamp (Unix epoch).
    pub modified_at: u64,

    /// Sample rate in Hz.
    pub sample_rate: u32,

    /// Buffer size in samples.
    pub buffer_size: usize,

    /// Channels (serialized).
    pub channels: HashMap<ChannelId, Channel>,

    /// Buses (serialized).
    pub buses: HashMap<BusId, Bus>,

    /// Master bus.
    pub master_bus: Bus,

    /// Automation data.
    pub automation: AutomationData,

    /// Channel templates.
    pub channel_templates: Vec<ChannelTemplate>,

    /// Scenes (automation snapshots).
    pub scenes: Vec<Scene>,

    /// Bus groups.
    pub bus_groups: Vec<BusGroup>,

    /// Session metadata.
    pub metadata: HashMap<String, String>,
}

impl SessionData {
    /// Create new session data.
    #[must_use]
    pub fn new(name: String, sample_rate: u32, buffer_size: usize) -> Self {
        use oximedia_audio::ChannelLayout;
        let master_bus = Bus::new(
            "Master".to_string(),
            crate::bus::BusType::Master,
            ChannelLayout::Stereo,
            sample_rate,
            buffer_size,
        );

        Self {
            version: SESSION_VERSION,
            name,
            id: Uuid::new_v4(),
            created_at: current_timestamp(),
            modified_at: current_timestamp(),
            sample_rate,
            buffer_size,
            channels: HashMap::new(),
            buses: HashMap::new(),
            master_bus,
            automation: AutomationData::new(),
            channel_templates: Vec::new(),
            scenes: Vec::new(),
            bus_groups: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Update modification timestamp.
    pub fn touch(&mut self) {
        self.modified_at = current_timestamp();
    }

    /// Serialize to JSON.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON.
    ///
    /// # Errors
    ///
    /// Returns error if deserialization fails.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Validate session data.
    #[must_use]
    pub fn validate(&self) -> bool {
        // Basic validation
        if self.version != SESSION_VERSION {
            return false;
        }
        if self.sample_rate == 0 || self.buffer_size == 0 {
            return false;
        }
        true
    }
}

/// Channel template for quick setup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelTemplate {
    /// Template ID.
    pub id: Uuid,

    /// Template name.
    pub name: String,

    /// Template description.
    pub description: String,

    /// Channel configuration.
    pub channel: Channel,

    /// Category (e.g., "Vocals", "Drums", "Synth").
    pub category: String,

    /// Tags for searching.
    pub tags: Vec<String>,
}

impl ChannelTemplate {
    /// Create new channel template.
    #[must_use]
    pub fn new(name: String, channel: Channel, category: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            description: String::new(),
            channel,
            category,
            tags: Vec::new(),
        }
    }

    /// Create channel from template.
    #[must_use]
    pub fn instantiate(&self, name: String) -> Channel {
        let mut channel = self.channel.clone();
        channel.set_name(name);
        channel
    }
}

/// Scene (mixer snapshot).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    /// Scene ID.
    pub id: Uuid,

    /// Scene name.
    pub name: String,

    /// Scene description.
    pub description: String,

    /// Automation snapshot.
    pub snapshot: AutomationSnapshot,

    /// Scene color.
    pub color: Option<String>,

    /// Scene number (for ordering).
    pub number: u32,
}

impl Scene {
    /// Create new scene.
    #[must_use]
    pub fn new(name: String, snapshot: AutomationSnapshot) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            description: String::new(),
            snapshot,
            color: None,
            number: 0,
        }
    }

    /// Apply scene to automation.
    pub fn apply(&self, automation: &mut AutomationData, time_samples: u64) {
        self.snapshot.apply_to(automation, time_samples);
    }
}

/// Undo/redo action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UndoAction {
    /// Channel added.
    ChannelAdded {
        /// Channel ID.
        id: ChannelId,
        /// Channel data.
        channel: Channel,
    },
    /// Channel removed.
    ChannelRemoved {
        /// Channel ID.
        id: ChannelId,
        /// Channel data.
        channel: Channel,
    },
    /// Channel modified.
    ChannelModified {
        /// Channel ID.
        id: ChannelId,
        /// State before modification.
        before: Channel,
        /// State after modification.
        after: Channel,
    },
    /// Bus added.
    BusAdded {
        /// Bus ID.
        id: BusId,
        /// Bus data.
        bus: Bus,
    },
    /// Bus removed.
    BusRemoved {
        /// Bus ID.
        id: BusId,
        /// Bus data.
        bus: Bus,
    },
    /// Bus modified.
    BusModified {
        /// Bus ID.
        id: BusId,
        /// State before modification.
        before: Bus,
        /// State after modification.
        after: Bus,
    },
    /// Automation point added.
    AutomationPointAdded {
        /// Automated parameter.
        parameter: crate::automation::AutomationParameter,
        /// Automation point.
        point: crate::automation::AutomationPoint,
    },
    /// Automation point removed.
    AutomationPointRemoved {
        /// Automated parameter.
        parameter: crate::automation::AutomationParameter,
        /// Automation point.
        point: crate::automation::AutomationPoint,
    },
    /// Scene added.
    SceneAdded {
        /// Scene data.
        scene: Scene,
    },
    /// Scene removed.
    SceneRemoved {
        /// Scene data.
        scene: Scene,
    },
}

/// Undo/redo manager.
#[derive(Debug, Clone)]
pub struct UndoManager {
    /// Undo history.
    undo_stack: VecDeque<UndoAction>,

    /// Redo history.
    redo_stack: VecDeque<UndoAction>,

    /// Maximum history size.
    max_history: usize,
}

impl UndoManager {
    /// Create new undo manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            undo_stack: VecDeque::new(),
            redo_stack: VecDeque::new(),
            max_history: MAX_UNDO_HISTORY,
        }
    }

    /// Push action to undo stack.
    pub fn push(&mut self, action: UndoAction) {
        self.undo_stack.push_back(action);
        if self.undo_stack.len() > self.max_history {
            self.undo_stack.pop_front();
        }
        // Clear redo stack on new action
        self.redo_stack.clear();
    }

    /// Undo last action.
    pub fn undo(&mut self) -> Option<UndoAction> {
        if let Some(action) = self.undo_stack.pop_back() {
            self.redo_stack.push_back(action.clone());
            Some(action)
        } else {
            None
        }
    }

    /// Redo last undone action.
    pub fn redo(&mut self) -> Option<UndoAction> {
        if let Some(action) = self.redo_stack.pop_back() {
            self.undo_stack.push_back(action.clone());
            Some(action)
        } else {
            None
        }
    }

    /// Check if undo is available.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Check if redo is available.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    /// Get undo stack size.
    #[must_use]
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Get redo stack size.
    #[must_use]
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }
}

impl Default for UndoManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Mixer session manager.
#[derive(Debug, Clone)]
pub struct MixerSession {
    /// Session data.
    data: SessionData,

    /// Undo/redo manager.
    undo_manager: UndoManager,

    /// Session is modified.
    modified: bool,

    /// Session file path.
    file_path: Option<String>,
}

impl MixerSession {
    /// Create new session.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: SessionData::new("Untitled".to_string(), 48000, 512),
            undo_manager: UndoManager::new(),
            modified: false,
            file_path: None,
        }
    }

    /// Create session with configuration.
    #[must_use]
    pub fn with_config(name: String, sample_rate: u32, buffer_size: usize) -> Self {
        Self {
            data: SessionData::new(name, sample_rate, buffer_size),
            undo_manager: UndoManager::new(),
            modified: false,
            file_path: None,
        }
    }

    /// Get session data.
    #[must_use]
    pub fn data(&self) -> &SessionData {
        &self.data
    }

    /// Get mutable session data.
    #[must_use]
    pub fn data_mut(&mut self) -> &mut SessionData {
        self.modified = true;
        &mut self.data
    }

    /// Check if session is modified.
    #[must_use]
    pub fn is_modified(&self) -> bool {
        self.modified
    }

    /// Mark session as saved.
    pub fn mark_saved(&mut self) {
        self.modified = false;
    }

    /// Get file path.
    #[must_use]
    pub fn file_path(&self) -> Option<&str> {
        self.file_path.as_deref()
    }

    /// Set file path.
    pub fn set_file_path(&mut self, path: Option<String>) {
        self.file_path = path;
    }

    /// Get undo manager.
    #[must_use]
    pub fn undo_manager(&self) -> &UndoManager {
        &self.undo_manager
    }

    /// Get mutable undo manager.
    #[must_use]
    pub fn undo_manager_mut(&mut self) -> &mut UndoManager {
        &mut self.undo_manager
    }

    /// Save session to JSON.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn save(&mut self) -> Result<String, serde_json::Error> {
        self.data.touch();
        self.mark_saved();
        self.data.to_json()
    }

    /// Load session from JSON.
    ///
    /// # Errors
    ///
    /// Returns error if deserialization fails.
    pub fn load(json: &str) -> Result<Self, serde_json::Error> {
        let data = SessionData::from_json(json)?;
        Ok(Self {
            data,
            undo_manager: UndoManager::new(),
            modified: false,
            file_path: None,
        })
    }

    /// Create snapshot of current state.
    #[must_use]
    pub fn create_snapshot(&self) -> SessionData {
        self.data.clone()
    }

    /// Restore from snapshot.
    pub fn restore_snapshot(&mut self, snapshot: SessionData) {
        self.data = snapshot;
        self.modified = true;
    }

    /// Add channel template.
    pub fn add_template(&mut self, template: ChannelTemplate) {
        self.data.channel_templates.push(template);
        self.modified = true;
    }

    /// Get channel template by ID.
    #[must_use]
    pub fn get_template(&self, id: Uuid) -> Option<&ChannelTemplate> {
        self.data.channel_templates.iter().find(|t| t.id == id)
    }

    /// Remove channel template.
    pub fn remove_template(&mut self, id: Uuid) {
        self.data.channel_templates.retain(|t| t.id != id);
        self.modified = true;
    }

    /// Add scene.
    pub fn add_scene(&mut self, scene: Scene) {
        self.data.scenes.push(scene);
        self.modified = true;
    }

    /// Get scene by ID.
    #[must_use]
    pub fn get_scene(&self, id: Uuid) -> Option<&Scene> {
        self.data.scenes.iter().find(|s| s.id == id)
    }

    /// Remove scene.
    pub fn remove_scene(&mut self, id: Uuid) {
        self.data.scenes.retain(|s| s.id != id);
        self.modified = true;
    }

    /// Recall scene.
    pub fn recall_scene(&mut self, id: Uuid, time_samples: u64) {
        if let Some(scene) = self.get_scene(id).cloned() {
            scene.apply(&mut self.data.automation, time_samples);
            self.modified = true;
        }
    }
}

impl Default for MixerSession {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current Unix timestamp.
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_audio::ChannelLayout;

    #[test]
    fn test_session_creation() {
        let session = MixerSession::new();
        assert_eq!(session.data().version, SESSION_VERSION);
        assert!(!session.is_modified());
    }

    #[test]
    fn test_session_modification() {
        let mut session = MixerSession::new();
        assert!(!session.is_modified());

        session.data_mut().name = "Modified".to_string();
        assert!(session.is_modified());

        session.mark_saved();
        assert!(!session.is_modified());
    }

    #[test]
    fn test_session_serialization() {
        let session = MixerSession::with_config("Test Session".to_string(), 48000, 512);
        let json = session.data().to_json().expect("json should be valid");

        let loaded = SessionData::from_json(&json).expect("loaded should be valid");
        assert_eq!(loaded.name, "Test Session");
        assert_eq!(loaded.sample_rate, 48000);
    }

    #[test]
    fn test_channel_template() {
        use crate::ChannelType;

        let channel = Channel::new(
            "Template".to_string(),
            ChannelType::Stereo,
            ChannelLayout::Stereo,
            48000,
            512,
        );

        let template =
            ChannelTemplate::new("Vocal Template".to_string(), channel, "Vocals".to_string());

        let instance = template.instantiate("Lead Vocal".to_string());
        assert_eq!(instance.name(), "Lead Vocal");
    }

    #[test]
    fn test_scene() {
        use crate::automation::{AutomationParameter, AutomationSnapshot};

        let mut snapshot = AutomationSnapshot::new("Scene 1".to_string());
        snapshot.set_value(AutomationParameter::MasterGain, 0.8);

        let scene = Scene::new("Main Mix".to_string(), snapshot);
        assert_eq!(scene.name, "Main Mix");
    }

    #[test]
    fn test_undo_manager() {
        let mut manager = UndoManager::new();

        let channel = Channel::new(
            "Test".to_string(),
            crate::ChannelType::Stereo,
            ChannelLayout::Stereo,
            48000,
            512,
        );

        let action = UndoAction::ChannelAdded {
            id: ChannelId(Uuid::new_v4()),
            channel,
        };

        assert!(!manager.can_undo());
        manager.push(action);
        assert!(manager.can_undo());

        let undone = manager.undo();
        assert!(undone.is_some());
        assert!(manager.can_redo());

        let redone = manager.redo();
        assert!(redone.is_some());
        assert!(manager.can_undo());
    }

    #[test]
    fn test_undo_history_limit() {
        let mut manager = UndoManager::new();
        manager.max_history = 5;

        for i in 0..10 {
            let channel = Channel::new(
                format!("Channel {i}"),
                crate::ChannelType::Stereo,
                ChannelLayout::Stereo,
                48000,
                512,
            );
            let action = UndoAction::ChannelAdded {
                id: ChannelId(Uuid::new_v4()),
                channel,
            };
            manager.push(action);
        }

        assert_eq!(manager.undo_count(), 5);
    }

    #[test]
    fn test_session_templates() {
        let mut session = MixerSession::new();

        let channel = Channel::new(
            "Template".to_string(),
            crate::ChannelType::Stereo,
            ChannelLayout::Stereo,
            48000,
            512,
        );

        let template =
            ChannelTemplate::new("Test Template".to_string(), channel, "Test".to_string());
        let template_id = template.id;

        session.add_template(template);
        assert!(session.get_template(template_id).is_some());

        session.remove_template(template_id);
        assert!(session.get_template(template_id).is_none());
    }

    #[test]
    fn test_session_scenes() {
        let mut session = MixerSession::new();

        let snapshot = crate::automation::AutomationSnapshot::new("Scene 1".to_string());
        let scene = Scene::new("Test Scene".to_string(), snapshot);
        let scene_id = scene.id;

        session.add_scene(scene);
        assert!(session.get_scene(scene_id).is_some());

        session.remove_scene(scene_id);
        assert!(session.get_scene(scene_id).is_none());
    }

    #[test]
    fn test_session_validation() {
        let data = SessionData::new("Test".to_string(), 48000, 512);
        assert!(data.validate());

        let mut invalid_data = data;
        invalid_data.sample_rate = 0;
        assert!(!invalid_data.validate());
    }

    #[test]
    fn test_session_snapshot() {
        let session = MixerSession::with_config("Test".to_string(), 48000, 512);
        let snapshot = session.create_snapshot();

        assert_eq!(snapshot.name, "Test");
        assert_eq!(snapshot.sample_rate, 48000);
    }
}
