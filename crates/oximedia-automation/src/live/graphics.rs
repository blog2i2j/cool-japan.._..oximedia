//! Graphics automation for live production.

use crate::{AutomationError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Graphics layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphicsLayer {
    /// Layer ID
    pub id: usize,
    /// Layer name
    pub name: String,
    /// Template name
    pub template: String,
    /// Is visible
    pub visible: bool,
    /// Layer data
    pub data: HashMap<String, String>,
}

/// Graphics automation.
pub struct GraphicsAutomation {
    layers: HashMap<usize, GraphicsLayer>,
    next_layer_id: usize,
}

impl GraphicsAutomation {
    /// Create a new graphics automation.
    pub fn new() -> Self {
        info!("Creating graphics automation");

        Self {
            layers: HashMap::new(),
            next_layer_id: 1,
        }
    }

    /// Create a new graphics layer.
    pub fn create_layer(&mut self, name: String, template: String) -> Result<usize> {
        let id = self.next_layer_id;
        self.next_layer_id += 1;

        debug!("Creating graphics layer {}: {}", id, name);

        let layer = GraphicsLayer {
            id,
            name,
            template,
            visible: false,
            data: HashMap::new(),
        };

        self.layers.insert(id, layer);

        Ok(id)
    }

    /// Update layer data.
    pub fn update_layer(&mut self, id: usize, data: HashMap<String, String>) -> Result<()> {
        let layer = self
            .layers
            .get_mut(&id)
            .ok_or_else(|| AutomationError::NotFound(format!("Layer {id}")))?;

        debug!("Updating layer {}: {:?}", id, data);
        layer.data = data;

        Ok(())
    }

    /// Show a layer.
    pub fn show_layer(&mut self, id: usize) -> Result<()> {
        let layer = self
            .layers
            .get_mut(&id)
            .ok_or_else(|| AutomationError::NotFound(format!("Layer {id}")))?;

        info!("Showing graphics layer: {}", layer.name);
        layer.visible = true;

        Ok(())
    }

    /// Hide a layer.
    pub fn hide_layer(&mut self, id: usize) -> Result<()> {
        let layer = self
            .layers
            .get_mut(&id)
            .ok_or_else(|| AutomationError::NotFound(format!("Layer {id}")))?;

        info!("Hiding graphics layer: {}", layer.name);
        layer.visible = false;

        Ok(())
    }

    /// Remove a layer.
    pub fn remove_layer(&mut self, id: usize) -> Result<()> {
        self.layers
            .remove(&id)
            .ok_or_else(|| AutomationError::NotFound(format!("Layer {id}")))?;

        info!("Removed graphics layer: {}", id);

        Ok(())
    }

    /// Get all visible layers.
    pub fn visible_layers(&self) -> Vec<&GraphicsLayer> {
        self.layers.values().filter(|layer| layer.visible).collect()
    }

    /// Hide all layers.
    pub fn hide_all(&mut self) {
        info!("Hiding all graphics layers");

        for layer in self.layers.values_mut() {
            layer.visible = false;
        }
    }
}

impl Default for GraphicsAutomation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graphics_creation() {
        let graphics = GraphicsAutomation::new();
        assert_eq!(graphics.visible_layers().len(), 0);
    }

    #[test]
    fn test_create_layer() {
        let mut graphics = GraphicsAutomation::new();
        let id = graphics
            .create_layer("Lower Third".to_string(), "lower_third.xml".to_string())
            .expect("operation should succeed");
        assert_eq!(id, 1);
    }

    #[test]
    fn test_show_hide_layer() {
        let mut graphics = GraphicsAutomation::new();
        let id = graphics
            .create_layer("Lower Third".to_string(), "lower_third.xml".to_string())
            .expect("operation should succeed");

        assert_eq!(graphics.visible_layers().len(), 0);

        graphics.show_layer(id).expect("show_layer should succeed");
        assert_eq!(graphics.visible_layers().len(), 1);

        graphics.hide_layer(id).expect("hide_layer should succeed");
        assert_eq!(graphics.visible_layers().len(), 0);
    }

    #[test]
    fn test_update_layer() {
        let mut graphics = GraphicsAutomation::new();
        let id = graphics
            .create_layer("Lower Third".to_string(), "lower_third.xml".to_string())
            .expect("operation should succeed");

        let mut data = HashMap::new();
        data.insert("name".to_string(), "John Doe".to_string());
        data.insert("title".to_string(), "Reporter".to_string());

        graphics
            .update_layer(id, data)
            .expect("update_layer should succeed");
    }
}
