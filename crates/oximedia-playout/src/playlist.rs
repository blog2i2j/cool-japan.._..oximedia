//! Playlist management and playback sequencing
//!
//! Supports multiple playlist formats (SMIL, XML, JSON), item sequencing,
//! transitions, looping, dynamic insertion, and ad markers.

use crate::scheduler::{CuePoint, Transition};
use crate::{PlayoutError, Result};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

/// Playlist format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaylistFormat {
    /// Synchronized Multimedia Integration Language
    SMIL,
    /// Custom XML format
    XML,
    /// JSON format
    JSON,
    /// M3U8 format
    M3U8,
}

/// Playlist item representing a single piece of content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaylistItem {
    /// Unique item ID
    pub id: Uuid,

    /// Display name
    pub name: String,

    /// Content file path
    pub path: PathBuf,

    /// Duration in frames (None for live sources)
    pub duration_frames: Option<u64>,

    /// In point (frame offset from start)
    pub in_point: u64,

    /// Out point (frame offset from start, None for end of file)
    pub out_point: Option<u64>,

    /// Transition in
    pub transition_in: Transition,

    /// Transition out
    pub transition_out: Transition,

    /// Audio level adjustment (0.0 - 2.0, 1.0 = unity)
    pub audio_level: f32,

    /// Video opacity (0.0 - 1.0)
    pub video_opacity: f32,

    /// Cue points
    pub cue_points: Vec<CuePoint>,

    /// Ad insertion markers
    pub ad_markers: Vec<AdMarker>,

    /// Metadata tags
    pub metadata: HashMap<String, String>,

    /// Enabled flag
    pub enabled: bool,

    /// Loop count (0 = no loop, -1 = infinite)
    pub loop_count: i32,

    /// Current loop iteration
    pub current_loop: i32,
}

impl PlaylistItem {
    /// Create a new playlist item
    pub fn new(name: String, path: PathBuf) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            path,
            duration_frames: None,
            in_point: 0,
            out_point: None,
            transition_in: Transition::Cut,
            transition_out: Transition::Cut,
            audio_level: 1.0,
            video_opacity: 1.0,
            cue_points: Vec::new(),
            ad_markers: Vec::new(),
            metadata: HashMap::new(),
            enabled: true,
            loop_count: 0,
            current_loop: 0,
        }
    }

    /// Get effective duration considering in/out points
    pub fn effective_duration(&self) -> Option<u64> {
        match (self.duration_frames, self.out_point) {
            (Some(duration), Some(out)) => {
                let start = self.in_point;
                let end = out.min(duration);
                if end > start {
                    Some(end - start)
                } else {
                    None
                }
            }
            (Some(duration), None) => {
                if duration > self.in_point {
                    Some(duration - self.in_point)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if item should loop
    pub fn should_loop(&self) -> bool {
        self.loop_count == -1 || self.current_loop < self.loop_count
    }

    /// Increment loop counter
    pub fn increment_loop(&mut self) {
        if self.loop_count != 0 {
            self.current_loop += 1;
        }
    }

    /// Reset loop counter
    pub fn reset_loop(&mut self) {
        self.current_loop = 0;
    }
}

/// Ad insertion marker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdMarker {
    /// Unique marker ID
    pub id: String,

    /// Frame position in content
    pub frame_position: u64,

    /// Ad duration in frames
    pub duration_frames: u64,

    /// Ad break type
    pub break_type: AdBreakType,

    /// Maximum number of ads
    pub max_ads: u32,

    /// SCTE-35 event ID (if applicable)
    pub scte35_event_id: Option<u32>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Ad break type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdBreakType {
    /// Pre-roll (before content)
    PreRoll,
    /// Mid-roll (during content)
    MidRoll,
    /// Post-roll (after content)
    PostRoll,
}

/// Playlist playback mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaybackMode {
    /// Play once and stop
    Once,
    /// Loop entire playlist
    Loop,
    /// Shuffle items
    Shuffle,
    /// Sequential with random fill
    RandomFill,
}

/// Playlist metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaylistMetadata {
    /// Playlist name
    pub name: String,

    /// Description
    pub description: String,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,

    /// Author
    pub author: String,

    /// Version
    pub version: String,

    /// Custom tags
    pub tags: Vec<String>,
}

impl Default for PlaylistMetadata {
    fn default() -> Self {
        Self {
            name: "Untitled Playlist".to_string(),
            description: String::new(),
            created_at: Utc::now(),
            modified_at: Utc::now(),
            author: String::new(),
            version: "1.0".to_string(),
            tags: Vec::new(),
        }
    }
}

/// Complete playlist
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Playlist {
    /// Playlist ID
    pub id: Uuid,

    /// Metadata
    pub metadata: PlaylistMetadata,

    /// Playlist items in order
    pub items: Vec<PlaylistItem>,

    /// Playback mode
    pub mode: PlaybackMode,

    /// Fill content for gaps
    pub fill_content: Option<PathBuf>,

    /// Default transition
    pub default_transition: Transition,

    /// Enabled flag
    pub enabled: bool,
}

impl Playlist {
    /// Create a new empty playlist
    pub fn new(name: String) -> Self {
        let metadata = PlaylistMetadata {
            name,
            ..Default::default()
        };

        Self {
            id: Uuid::new_v4(),
            metadata,
            items: Vec::new(),
            mode: PlaybackMode::Once,
            fill_content: None,
            default_transition: Transition::Cut,
            enabled: true,
        }
    }

    /// Add an item to the playlist
    pub fn add_item(&mut self, item: PlaylistItem) {
        self.items.push(item);
        self.metadata.modified_at = Utc::now();
    }

    /// Insert an item at a specific position
    pub fn insert_item(&mut self, index: usize, item: PlaylistItem) {
        if index <= self.items.len() {
            self.items.insert(index, item);
            self.metadata.modified_at = Utc::now();
        }
    }

    /// Remove an item by ID
    pub fn remove_item(&mut self, item_id: Uuid) -> Result<()> {
        let original_len = self.items.len();
        self.items.retain(|item| item.id != item_id);

        if self.items.len() < original_len {
            self.metadata.modified_at = Utc::now();
            Ok(())
        } else {
            Err(PlayoutError::Playlist(format!("Item not found: {item_id}")))
        }
    }

    /// Move an item to a new position
    pub fn move_item(&mut self, item_id: Uuid, new_index: usize) -> Result<()> {
        let old_index = self
            .items
            .iter()
            .position(|item| item.id == item_id)
            .ok_or_else(|| PlayoutError::Playlist(format!("Item not found: {item_id}")))?;

        if new_index >= self.items.len() {
            return Err(PlayoutError::Playlist("Invalid index".to_string()));
        }

        let item = self.items.remove(old_index);
        self.items.insert(new_index, item);
        self.metadata.modified_at = Utc::now();

        Ok(())
    }

    /// Get item by ID
    pub fn get_item(&self, item_id: Uuid) -> Option<&PlaylistItem> {
        self.items.iter().find(|item| item.id == item_id)
    }

    /// Get mutable item by ID
    pub fn get_item_mut(&mut self, item_id: Uuid) -> Option<&mut PlaylistItem> {
        self.items.iter_mut().find(|item| item.id == item_id)
    }

    /// Get total duration in frames
    pub fn total_duration(&self) -> Option<u64> {
        let mut total = 0u64;
        for item in &self.items {
            if let Some(duration) = item.effective_duration() {
                total += duration;
            } else {
                return None; // Can't calculate total if any item has unknown duration
            }
        }
        Some(total)
    }

    /// Clear all items
    pub fn clear(&mut self) {
        self.items.clear();
        self.metadata.modified_at = Utc::now();
    }

    /// Shuffle items (if mode is Shuffle)
    pub fn shuffle(&mut self) {
        if self.mode == PlaybackMode::Shuffle {
            // Simple shuffle using indices
            use std::collections::hash_map::RandomState;
            use std::hash::BuildHasher;

            let mut indices: Vec<usize> = (0..self.items.len()).collect();
            let state = RandomState::new();

            // Fisher-Yates shuffle
            for i in (1..indices.len()).rev() {
                let j = (state.hash_one(i) as usize) % (i + 1);
                indices.swap(i, j);
            }

            let mut new_items = Vec::with_capacity(self.items.len());
            for idx in indices {
                new_items.push(self.items[idx].clone());
            }
            self.items = new_items;
        }
    }

    /// Validate playlist
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.items.is_empty() {
            errors.push("Playlist is empty".to_string());
        }

        for (idx, item) in self.items.iter().enumerate() {
            if !item.path.exists() {
                errors.push(format!(
                    "Item {} ({}): File not found: {:?}",
                    idx, item.name, item.path
                ));
            }

            if let Some(out_pt) = item.out_point {
                if out_pt <= item.in_point {
                    errors.push(format!(
                        "Item {} ({}): Out point must be after in point",
                        idx, item.name
                    ));
                }
            }

            if item.audio_level < 0.0 || item.audio_level > 2.0 {
                errors.push(format!(
                    "Item {} ({}): Invalid audio level: {}",
                    idx, item.name, item.audio_level
                ));
            }

            if item.video_opacity < 0.0 || item.video_opacity > 1.0 {
                errors.push(format!(
                    "Item {} ({}): Invalid video opacity: {}",
                    idx, item.name, item.video_opacity
                ));
            }
        }

        errors
    }
}

// ---------------------------------------------------------------------------
// XML / SMIL helper functions (module-private)
// ---------------------------------------------------------------------------

/// Escape special XML characters in a string value.
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Extract the value of an XML attribute written as `attr="value"` from a
/// single line of text.  Returns `None` when the attribute is not present.
fn extract_attr(line: &str, attr: &str) -> Option<String> {
    let needle = format!("{attr}=\"");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

// ---------------------------------------------------------------------------

/// Playlist manager
pub struct PlaylistManager {
    playlists: Arc<RwLock<HashMap<Uuid, Playlist>>>,
    active_playlist: Arc<RwLock<Option<Uuid>>>,
    current_item_index: Arc<RwLock<usize>>,
}

impl PlaylistManager {
    /// Create a new playlist manager
    pub fn new() -> Self {
        Self {
            playlists: Arc::new(RwLock::new(HashMap::new())),
            active_playlist: Arc::new(RwLock::new(None)),
            current_item_index: Arc::new(RwLock::new(0)),
        }
    }

    /// Add a playlist
    pub fn add_playlist(&self, playlist: Playlist) -> Uuid {
        let id = playlist.id;
        self.playlists.write().insert(id, playlist);
        id
    }

    /// Remove a playlist
    pub fn remove_playlist(&self, playlist_id: Uuid) -> Result<()> {
        let mut playlists = self.playlists.write();
        playlists
            .remove(&playlist_id)
            .ok_or_else(|| PlayoutError::Playlist(format!("Playlist not found: {playlist_id}")))?;
        Ok(())
    }

    /// Get a playlist
    pub fn get_playlist(&self, playlist_id: Uuid) -> Option<Playlist> {
        self.playlists.read().get(&playlist_id).cloned()
    }

    /// Set active playlist
    pub fn set_active(&self, playlist_id: Uuid) -> Result<()> {
        if !self.playlists.read().contains_key(&playlist_id) {
            return Err(PlayoutError::Playlist(format!(
                "Playlist not found: {playlist_id}"
            )));
        }

        *self.active_playlist.write() = Some(playlist_id);
        *self.current_item_index.write() = 0;

        Ok(())
    }

    /// Get active playlist
    pub fn get_active(&self) -> Option<Playlist> {
        self.active_playlist
            .read()
            .and_then(|id| self.get_playlist(id))
    }

    /// Get current item in active playlist
    pub fn get_current_item(&self) -> Option<PlaylistItem> {
        let active_id = (*self.active_playlist.read())?;
        let playlist = self.get_playlist(active_id)?;
        let index = *self.current_item_index.read();

        if index < playlist.items.len() {
            Some(playlist.items[index].clone())
        } else {
            None
        }
    }

    /// Advance to next item
    pub fn next_item(&self) -> Option<PlaylistItem> {
        let active_id = (*self.active_playlist.read())?;
        let playlist = self.get_playlist(active_id)?;
        let mut index = self.current_item_index.write();

        *index += 1;

        if *index >= playlist.items.len() {
            match playlist.mode {
                PlaybackMode::Once => {
                    return None;
                }
                PlaybackMode::Loop => {
                    *index = 0;
                }
                PlaybackMode::Shuffle => {
                    // Re-shuffle and start over
                    *index = 0;
                }
                PlaybackMode::RandomFill => {
                    // Use fill content or wrap around
                    *index = 0;
                }
            }
        }

        if *index < playlist.items.len() {
            Some(playlist.items[*index].clone())
        } else {
            None
        }
    }

    /// Get previous item
    pub fn previous_item(&self) -> Option<PlaylistItem> {
        let active_id = (*self.active_playlist.read())?;
        let playlist = self.get_playlist(active_id)?;
        let mut index = self.current_item_index.write();

        if *index > 0 {
            *index -= 1;
            Some(playlist.items[*index].clone())
        } else {
            None
        }
    }

    /// Jump to specific item
    pub fn jump_to_item(&self, item_id: Uuid) -> Result<PlaylistItem> {
        let active_id = self
            .active_playlist
            .read()
            .ok_or_else(|| PlayoutError::Playlist("No active playlist".to_string()))?;

        let playlist = self
            .get_playlist(active_id)
            .ok_or_else(|| PlayoutError::Playlist(format!("Playlist not found: {active_id}")))?;

        let index = playlist
            .items
            .iter()
            .position(|item| item.id == item_id)
            .ok_or_else(|| PlayoutError::Playlist(format!("Item not found: {item_id}")))?;

        *self.current_item_index.write() = index;
        Ok(playlist.items[index].clone())
    }

    /// Load playlist from file
    pub fn load_from_file(&self, path: &Path, format: PlaylistFormat) -> Result<Uuid> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| PlayoutError::Playlist(format!("Failed to read file: {e}")))?;

        let playlist = match format {
            PlaylistFormat::JSON => self.parse_json(&content)?,
            PlaylistFormat::XML => self.parse_xml(&content)?,
            PlaylistFormat::SMIL => self.parse_smil(&content)?,
            PlaylistFormat::M3U8 => self.parse_m3u8(&content)?,
        };

        Ok(self.add_playlist(playlist))
    }

    /// Save playlist to file
    pub fn save_to_file(
        &self,
        playlist_id: Uuid,
        path: &Path,
        format: PlaylistFormat,
    ) -> Result<()> {
        let playlist = self
            .get_playlist(playlist_id)
            .ok_or_else(|| PlayoutError::Playlist(format!("Playlist not found: {playlist_id}")))?;

        let content = match format {
            PlaylistFormat::JSON => self.serialize_json(&playlist)?,
            PlaylistFormat::XML => self.serialize_xml(&playlist)?,
            PlaylistFormat::SMIL => self.serialize_smil(&playlist)?,
            PlaylistFormat::M3U8 => self.serialize_m3u8(&playlist)?,
        };

        std::fs::write(path, content)
            .map_err(|e| PlayoutError::Playlist(format!("Failed to write file: {e}")))?;

        Ok(())
    }

    /// Parse JSON format
    fn parse_json(&self, content: &str) -> Result<Playlist> {
        serde_json::from_str(content)
            .map_err(|e| PlayoutError::Playlist(format!("JSON parse error: {e}")))
    }

    /// Serialize to JSON
    fn serialize_json(&self, playlist: &Playlist) -> Result<String> {
        serde_json::to_string_pretty(playlist)
            .map_err(|e| PlayoutError::Playlist(format!("JSON serialize error: {e}")))
    }

    /// Parse XML format (simplified)
    fn parse_xml(&self, content: &str) -> Result<Playlist> {
        // Extract playlist name from <playlist name="..." ...> tag
        let playlist_name = content
            .lines()
            .find(|line| line.contains("<playlist"))
            .and_then(|line| extract_attr(line, "name"))
            .unwrap_or_else(|| "Imported XML Playlist".to_string());

        let mut playlist = Playlist::new(playlist_name);

        for line in content.lines() {
            let trimmed = line.trim();
            if !trimmed.starts_with("<item") {
                continue;
            }

            let path_str = extract_attr(trimmed, "path").unwrap_or_default();
            if path_str.is_empty() {
                continue;
            }

            let name = extract_attr(trimmed, "name").unwrap_or_else(|| path_str.clone());

            let mut item = PlaylistItem::new(name, PathBuf::from(&path_str));

            // Try "duration_frames" first, fall back to "duration" (seconds * 25)
            if let Some(frames_str) = extract_attr(trimmed, "duration_frames") {
                if let Ok(frames) = frames_str.parse::<u64>() {
                    item.duration_frames = Some(frames);
                }
            } else if let Some(dur_str) = extract_attr(trimmed, "duration") {
                if let Ok(secs) = dur_str.parse::<u64>() {
                    item.duration_frames = Some(secs * 25);
                }
            }

            playlist.add_item(item);
        }

        Ok(playlist)
    }

    /// Serialize to XML
    fn serialize_xml(&self, playlist: &Playlist) -> Result<String> {
        let mut out = String::new();
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str(&format!(
            "<playlist name=\"{}\" version=\"1.0\">\n",
            escape_xml(&playlist.metadata.name)
        ));

        for item in &playlist.items {
            out.push_str(&format!(
                "  <item name=\"{}\" path=\"{}\"",
                escape_xml(&item.name),
                escape_xml(&item.path.display().to_string())
            ));
            if let Some(frames) = item.duration_frames {
                out.push_str(&format!(" duration_frames=\"{frames}\""));
            }
            out.push_str(" />\n");
        }

        out.push_str("</playlist>\n");
        Ok(out)
    }

    /// Parse SMIL format
    fn parse_smil(&self, content: &str) -> Result<Playlist> {
        // Extract title from <meta name="title" content="..."/>
        let playlist_name = content
            .lines()
            .find(|line| line.contains("<meta") && line.contains("name=\"title\""))
            .and_then(|line| extract_attr(line, "content"))
            .unwrap_or_else(|| "Imported SMIL Playlist".to_string());

        let mut playlist = Playlist::new(playlist_name);

        for line in content.lines() {
            let trimmed = line.trim();

            // Accept <video, <audio, <ref elements
            let is_media = trimmed.starts_with("<video")
                || trimmed.starts_with("<audio")
                || trimmed.starts_with("<ref");

            if !is_media {
                continue;
            }

            let src = extract_attr(trimmed, "src").unwrap_or_default();
            if src.is_empty() {
                continue;
            }

            let name = extract_attr(trimmed, "title").unwrap_or_else(|| src.clone());

            let mut item = PlaylistItem::new(name, PathBuf::from(&src));

            // Parse dur attribute: "300s" or "300" → seconds → frames at 25 fps
            if let Some(dur_str) = extract_attr(trimmed, "dur") {
                let dur_stripped = dur_str.trim_end_matches('s');
                if let Ok(secs) = dur_stripped.parse::<f64>() {
                    item.duration_frames = Some((secs * 25.0).round() as u64);
                }
            }

            playlist.add_item(item);
        }

        Ok(playlist)
    }

    /// Serialize to SMIL
    fn serialize_smil(&self, playlist: &Playlist) -> Result<String> {
        let mut out = String::new();
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str("<smil>\n");
        out.push_str("  <head>\n");
        out.push_str(&format!(
            "    <meta name=\"title\" content=\"{}\"/>\n",
            escape_xml(&playlist.metadata.name)
        ));
        out.push_str("  </head>\n");
        out.push_str("  <body>\n");
        out.push_str("    <seq>\n");

        for item in &playlist.items {
            let src = escape_xml(&item.path.display().to_string());
            let dur_secs = item.duration_frames.map_or(0, |f| f / 25);
            out.push_str(&format!(
                "      <video src=\"{src}\" dur=\"{dur_secs}s\"/>\n"
            ));
        }

        out.push_str("    </seq>\n");
        out.push_str("  </body>\n");
        out.push_str("</smil>\n");
        Ok(out)
    }

    /// Parse M3U8 format
    fn parse_m3u8(&self, content: &str) -> Result<Playlist> {
        let mut playlist = Playlist::new("Imported M3U8".to_string());

        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() {
                continue; // Skip comments and empty lines
            }

            let item = PlaylistItem::new(line.to_string(), PathBuf::from(line));
            playlist.add_item(item);
        }

        Ok(playlist)
    }

    /// Serialize to M3U8
    fn serialize_m3u8(&self, playlist: &Playlist) -> Result<String> {
        let mut content = String::from("#EXTM3U\n");
        content.push_str(&format!("#EXTINF:-1,{}\n", playlist.metadata.name));

        for item in &playlist.items {
            if let Some(duration) = item.effective_duration() {
                content.push_str(&format!("#EXTINF:{},{}\n", duration, item.name));
            }
            content.push_str(&format!("{}\n", item.path.display()));
        }

        Ok(content)
    }

    /// Get all playlists
    pub fn list_playlists(&self) -> Vec<(Uuid, String)> {
        self.playlists
            .read()
            .iter()
            .map(|(id, pl)| (*id, pl.metadata.name.clone()))
            .collect()
    }

    /// Clear all playlists
    pub fn clear_all(&self) {
        self.playlists.write().clear();
        *self.active_playlist.write() = None;
        *self.current_item_index.write() = 0;
    }
}

impl Default for PlaylistManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_playlist_creation() {
        let playlist = Playlist::new("Test Playlist".to_string());
        assert_eq!(playlist.metadata.name, "Test Playlist");
        assert_eq!(playlist.items.len(), 0);
    }

    #[test]
    fn test_add_remove_items() {
        let mut playlist = Playlist::new("Test".to_string());
        let item = PlaylistItem::new("Item 1".to_string(), PathBuf::from("/test.mxf"));
        let item_id = item.id;

        playlist.add_item(item);
        assert_eq!(playlist.items.len(), 1);

        playlist
            .remove_item(item_id)
            .expect("should succeed in test");
        assert_eq!(playlist.items.len(), 0);
    }

    #[test]
    fn test_move_item() {
        let mut playlist = Playlist::new("Test".to_string());
        let item1 = PlaylistItem::new("Item 1".to_string(), PathBuf::from("/test1.mxf"));
        let item2 = PlaylistItem::new("Item 2".to_string(), PathBuf::from("/test2.mxf"));
        let item1_id = item1.id;

        playlist.add_item(item1);
        playlist.add_item(item2);

        playlist
            .move_item(item1_id, 1)
            .expect("should succeed in test");
        assert_eq!(playlist.items[1].id, item1_id);
    }

    #[test]
    fn test_effective_duration() {
        let mut item = PlaylistItem::new("Test".to_string(), PathBuf::from("/test.mxf"));
        item.duration_frames = Some(1000);
        item.in_point = 100;
        item.out_point = Some(500);

        assert_eq!(item.effective_duration(), Some(400));
    }

    #[test]
    fn test_playlist_manager() {
        let manager = PlaylistManager::new();
        let playlist = Playlist::new("Test".to_string());
        let playlist_id = manager.add_playlist(playlist);

        manager
            .set_active(playlist_id)
            .expect("should succeed in test");
        let active = manager.get_active().expect("should succeed in test");
        assert_eq!(active.id, playlist_id);
    }

    #[test]
    fn test_next_item_loop() {
        let manager = PlaylistManager::new();
        let mut playlist = Playlist::new("Test".to_string());
        playlist.mode = PlaybackMode::Loop;

        let item1 = PlaylistItem::new("Item 1".to_string(), PathBuf::from("/test1.mxf"));
        let item2 = PlaylistItem::new("Item 2".to_string(), PathBuf::from("/test2.mxf"));

        playlist.add_item(item1);
        playlist.add_item(item2);

        let playlist_id = manager.add_playlist(playlist);
        manager
            .set_active(playlist_id)
            .expect("should succeed in test");

        // Get first item
        let _first = manager.get_current_item();

        // Advance to second
        let _second = manager.next_item();

        // Should loop back to first
        let looped = manager.next_item();
        assert!(looped.is_some());
    }

    #[test]
    fn test_ad_marker() {
        let marker = AdMarker {
            id: "ad1".to_string(),
            frame_position: 1000,
            duration_frames: 600,
            break_type: AdBreakType::MidRoll,
            max_ads: 3,
            scte35_event_id: Some(123),
            metadata: HashMap::new(),
        };

        assert_eq!(marker.break_type, AdBreakType::MidRoll);
        assert_eq!(marker.duration_frames, 600);
    }

    #[test]
    fn test_m3u8_parsing() {
        let manager = PlaylistManager::new();
        let m3u8_content = r#"#EXTM3U
#EXTINF:-1,Test Playlist
#EXTINF:100,Video 1
/path/to/video1.mp4
#EXTINF:200,Video 2
/path/to/video2.mp4
"#;

        let playlist = manager
            .parse_m3u8(m3u8_content)
            .expect("should succeed in test");
        assert_eq!(playlist.items.len(), 2);
    }
}
