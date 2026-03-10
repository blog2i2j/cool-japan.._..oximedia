//! Take management system.

use super::{Take, TakeId, TakeSelector};
use crate::clip::ClipId;
use crate::error::{ClipError, ClipResult};
use std::collections::HashMap;

/// Manages takes for clips.
#[derive(Debug, Clone, Default)]
pub struct TakeManager {
    /// Takes indexed by scene.
    takes_by_scene: HashMap<String, Vec<Take>>,

    /// Takes indexed by clip ID.
    takes_by_clip: HashMap<ClipId, Vec<TakeId>>,
}

impl TakeManager {
    /// Creates a new take manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            takes_by_scene: HashMap::new(),
            takes_by_clip: HashMap::new(),
        }
    }

    /// Adds a take.
    pub fn add_take(&mut self, take: Take) {
        self.takes_by_clip
            .entry(take.clip_id)
            .or_default()
            .push(take.id);

        self.takes_by_scene
            .entry(take.scene.clone())
            .or_default()
            .push(take);
    }

    /// Gets all takes for a scene.
    #[must_use]
    pub fn get_scene_takes(&self, scene: &str) -> Vec<&Take> {
        self.takes_by_scene
            .get(scene)
            .map_or_else(Vec::new, |takes| takes.iter().collect())
    }

    /// Gets all takes for a clip.
    #[must_use]
    pub fn get_clip_takes(&self, clip_id: &ClipId) -> Vec<&Take> {
        let take_ids = self.takes_by_clip.get(clip_id);
        if let Some(take_ids) = take_ids {
            self.takes_by_scene
                .values()
                .flat_map(|takes| takes.iter())
                .filter(|take| take_ids.contains(&take.id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Gets a specific take.
    #[must_use]
    pub fn get_take(&self, take_id: &TakeId) -> Option<&Take> {
        self.takes_by_scene
            .values()
            .flat_map(|takes| takes.iter())
            .find(|take| &take.id == take_id)
    }

    /// Updates a take.
    ///
    /// # Errors
    ///
    /// Returns an error if the take is not found.
    pub fn update_take(&mut self, take_id: &TakeId, updated: Take) -> ClipResult<()> {
        for takes in self.takes_by_scene.values_mut() {
            if let Some(take) = takes.iter_mut().find(|t| &t.id == take_id) {
                *take = updated;
                return Ok(());
            }
        }
        Err(ClipError::TakeNotFound(take_id.to_string()))
    }

    /// Removes a take.
    ///
    /// # Errors
    ///
    /// Returns an error if the take is not found.
    pub fn remove_take(&mut self, take_id: &TakeId) -> ClipResult<()> {
        for takes in self.takes_by_scene.values_mut() {
            if let Some(pos) = takes.iter().position(|t| &t.id == take_id) {
                let take = takes.remove(pos);
                if let Some(clip_takes) = self.takes_by_clip.get_mut(&take.clip_id) {
                    clip_takes.retain(|id| id != take_id);
                }
                return Ok(());
            }
        }
        Err(ClipError::TakeNotFound(take_id.to_string()))
    }

    /// Selects the best take for a scene using a selector.
    #[must_use]
    pub fn select_best_take(&self, scene: &str, selector: TakeSelector) -> Option<Take> {
        let takes = self.get_scene_takes(scene);
        if takes.is_empty() {
            return None;
        }

        let takes_vec: Vec<Take> = takes.into_iter().cloned().collect();
        selector.select(&takes_vec).cloned()
    }

    /// Gets the number of takes for a scene.
    #[must_use]
    pub fn scene_take_count(&self, scene: &str) -> usize {
        self.takes_by_scene.get(scene).map_or(0, Vec::len)
    }

    /// Gets all scene names.
    #[must_use]
    pub fn all_scenes(&self) -> Vec<String> {
        self.takes_by_scene.keys().cloned().collect()
    }

    /// Returns the total number of takes.
    #[must_use]
    pub fn total_takes(&self) -> usize {
        self.takes_by_scene.values().map(Vec::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_take_manager() {
        let mut manager = TakeManager::new();
        let clip_id = ClipId::new();

        let take1 = Take::new(clip_id, "Scene 1", 1);
        let take2 = Take::new(clip_id, "Scene 1", 2);

        manager.add_take(take1);
        manager.add_take(take2);

        assert_eq!(manager.scene_take_count("Scene 1"), 2);
        assert_eq!(manager.total_takes(), 2);
    }

    #[test]
    fn test_remove_take() {
        let mut manager = TakeManager::new();
        let clip_id = ClipId::new();
        let take = Take::new(clip_id, "Scene 1", 1);
        let take_id = take.id;

        manager.add_take(take);
        assert_eq!(manager.total_takes(), 1);

        manager
            .remove_take(&take_id)
            .expect("remove_take should succeed");
        assert_eq!(manager.total_takes(), 0);
    }
}
