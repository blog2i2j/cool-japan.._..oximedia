//! Scene management.

use crate::{GamingError, GamingResult};

/// Scene manager for switching between different scenes.
pub struct SceneManager {
    scenes: Vec<Scene>,
    active_scene: Option<String>,
}

/// Scene definition.
#[derive(Debug, Clone)]
pub struct Scene {
    /// Scene name
    pub name: String,
    /// Scene description
    pub description: String,
}

impl SceneManager {
    /// Create a new scene manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            scenes: Vec::new(),
            active_scene: None,
        }
    }

    /// Add a scene.
    pub fn add_scene(&mut self, scene: Scene) {
        self.scenes.push(scene);
    }

    /// Remove a scene.
    pub fn remove_scene(&mut self, name: &str) -> GamingResult<()> {
        if self.active_scene.as_deref() == Some(name) {
            return Err(GamingError::SceneNotFound(
                "Cannot remove active scene".to_string(),
            ));
        }
        self.scenes.retain(|s| s.name != name);
        Ok(())
    }

    /// Switch to a scene.
    pub fn switch_to(&mut self, name: &str) -> GamingResult<()> {
        if !self.scenes.iter().any(|s| s.name == name) {
            return Err(GamingError::SceneNotFound(name.to_string()));
        }
        self.active_scene = Some(name.to_string());
        Ok(())
    }

    /// Get active scene.
    #[must_use]
    pub fn active_scene(&self) -> Option<&str> {
        self.active_scene.as_deref()
    }

    /// Get scene count.
    #[must_use]
    pub fn scene_count(&self) -> usize {
        self.scenes.len()
    }
}

impl Default for SceneManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_manager_creation() {
        let manager = SceneManager::new();
        assert_eq!(manager.scene_count(), 0);
    }

    #[test]
    fn test_add_scene() {
        let mut manager = SceneManager::new();
        manager.add_scene(Scene {
            name: "Gameplay".to_string(),
            description: "Main gameplay scene".to_string(),
        });
        assert_eq!(manager.scene_count(), 1);
    }

    #[test]
    fn test_switch_scene() {
        let mut manager = SceneManager::new();
        manager.add_scene(Scene {
            name: "Gameplay".to_string(),
            description: "Main gameplay scene".to_string(),
        });
        manager
            .switch_to("Gameplay")
            .expect("switch should succeed");
        assert_eq!(manager.active_scene(), Some("Gameplay"));
    }
}
