//! Marker management system.

use super::{Marker, MarkerId, MarkerType};
use crate::clip::ClipId;
use crate::error::{ClipError, ClipResult};
use std::collections::HashMap;

/// Manages markers for clips.
#[derive(Debug, Clone, Default)]
pub struct MarkerManager {
    /// Markers indexed by clip ID.
    markers: HashMap<ClipId, Vec<Marker>>,
}

impl MarkerManager {
    /// Creates a new marker manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            markers: HashMap::new(),
        }
    }

    /// Adds a marker to a clip.
    pub fn add_marker(&mut self, clip_id: ClipId, marker: Marker) {
        self.markers.entry(clip_id).or_default().push(marker);
    }

    /// Removes a marker from a clip.
    ///
    /// # Errors
    ///
    /// Returns an error if the marker is not found.
    pub fn remove_marker(&mut self, clip_id: &ClipId, marker_id: &MarkerId) -> ClipResult<()> {
        if let Some(markers) = self.markers.get_mut(clip_id) {
            if let Some(pos) = markers.iter().position(|m| &m.id == marker_id) {
                markers.remove(pos);
                return Ok(());
            }
        }
        Err(ClipError::MarkerNotFound(marker_id.to_string()))
    }

    /// Gets all markers for a clip.
    #[must_use]
    pub fn get_markers(&self, clip_id: &ClipId) -> Vec<&Marker> {
        self.markers
            .get(clip_id)
            .map_or_else(Vec::new, |markers| markers.iter().collect())
    }

    /// Gets markers of a specific type for a clip.
    #[must_use]
    pub fn get_markers_by_type(&self, clip_id: &ClipId, marker_type: MarkerType) -> Vec<&Marker> {
        self.markers.get(clip_id).map_or_else(Vec::new, |markers| {
            markers
                .iter()
                .filter(|m| m.marker_type == marker_type)
                .collect()
        })
    }

    /// Gets a specific marker.
    #[must_use]
    pub fn get_marker(&self, clip_id: &ClipId, marker_id: &MarkerId) -> Option<&Marker> {
        self.markers
            .get(clip_id)
            .and_then(|markers| markers.iter().find(|m| &m.id == marker_id))
    }

    /// Updates a marker.
    ///
    /// # Errors
    ///
    /// Returns an error if the marker is not found.
    pub fn update_marker(
        &mut self,
        clip_id: &ClipId,
        marker_id: &MarkerId,
        updated: Marker,
    ) -> ClipResult<()> {
        if let Some(markers) = self.markers.get_mut(clip_id) {
            if let Some(marker) = markers.iter_mut().find(|m| &m.id == marker_id) {
                *marker = updated;
                return Ok(());
            }
        }
        Err(ClipError::MarkerNotFound(marker_id.to_string()))
    }

    /// Gets all chapter markers for a clip, sorted by frame.
    #[must_use]
    pub fn get_chapters(&self, clip_id: &ClipId) -> Vec<&Marker> {
        let mut chapters = self.get_markers_by_type(clip_id, MarkerType::Chapter);
        chapters.sort_by_key(|m| m.frame);
        chapters
    }

    /// Gets all to-do markers for a clip.
    #[must_use]
    pub fn get_todos(&self, clip_id: &ClipId) -> Vec<&Marker> {
        self.get_markers_by_type(clip_id, MarkerType::ToDo)
    }

    /// Gets incomplete to-do markers for a clip.
    #[must_use]
    pub fn get_incomplete_todos(&self, clip_id: &ClipId) -> Vec<&Marker> {
        self.get_markers_by_type(clip_id, MarkerType::ToDo)
            .into_iter()
            .filter(|m| !m.is_completed)
            .collect()
    }

    /// Clears all markers for a clip.
    pub fn clear_markers(&mut self, clip_id: &ClipId) {
        self.markers.remove(clip_id);
    }

    /// Returns the total number of markers across all clips.
    #[must_use]
    pub fn total_markers(&self) -> usize {
        self.markers.values().map(Vec::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_manager() {
        let mut manager = MarkerManager::new();
        let clip_id = ClipId::new();

        let marker1 = Marker::chapter(100, "Chapter 1");
        let marker2 = Marker::chapter(200, "Chapter 2");
        let marker3 = Marker::todo(150, "Fix color");

        manager.add_marker(clip_id, marker1.clone());
        manager.add_marker(clip_id, marker2.clone());
        manager.add_marker(clip_id, marker3);

        assert_eq!(manager.get_markers(&clip_id).len(), 3);
        assert_eq!(manager.get_chapters(&clip_id).len(), 2);
        assert_eq!(manager.get_todos(&clip_id).len(), 1);
    }

    #[test]
    fn test_remove_marker() {
        let mut manager = MarkerManager::new();
        let clip_id = ClipId::new();
        let marker = Marker::chapter(100, "Chapter 1");
        let marker_id = marker.id;

        manager.add_marker(clip_id, marker);
        assert_eq!(manager.get_markers(&clip_id).len(), 1);

        manager
            .remove_marker(&clip_id, &marker_id)
            .expect("remove_marker should succeed");
        assert_eq!(manager.get_markers(&clip_id).len(), 0);
    }
}
