//! Template system for graphics

use crate::color::{Color, Gradient};
use crate::error::Result;
use crate::primitives::{Circle, Fill, Point, Rect, Stroke};
use crate::text::TextStyle;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Template definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Template {
    /// Template name
    pub name: String,
    /// Template version
    pub version: String,
    /// Description
    pub description: Option<String>,
    /// Resolution (width, height)
    pub resolution: (u32, u32),
    /// Layers
    pub layers: Vec<Layer>,
    /// Variables
    pub variables: HashMap<String, VariableDefinition>,
}

impl Template {
    /// Create a new template
    #[must_use]
    pub fn new(name: String, resolution: (u32, u32)) -> Self {
        Self {
            name,
            version: "1.0".to_string(),
            description: None,
            resolution,
            layers: Vec::new(),
            variables: HashMap::new(),
        }
    }

    /// Add a layer
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    /// Add a variable
    pub fn add_variable(&mut self, name: String, var: VariableDefinition) {
        self.variables.insert(name, var);
    }

    /// Render with data
    pub fn render(&self, data: &HashMap<String, String>) -> Result<RenderedTemplate> {
        let mut rendered_layers = Vec::new();

        for layer in &self.layers {
            if let Some(rendered) = layer.render(data)? {
                rendered_layers.push(rendered);
            }
        }

        Ok(RenderedTemplate {
            resolution: self.resolution,
            layers: rendered_layers,
        })
    }

    /// Load from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(Into::into)
    }

    /// Save to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Into::into)
    }
}

/// Variable definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableDefinition {
    /// Variable type
    pub var_type: VariableType,
    /// Default value
    pub default: Option<String>,
    /// Description
    pub description: Option<String>,
}

/// Variable type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VariableType {
    /// Text string
    Text,
    /// Number
    Number,
    /// Color
    Color,
    /// Image path
    Image,
}

/// Template layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    /// Layer ID
    pub id: String,
    /// Layer type
    pub layer_type: LayerType,
    /// Visible
    pub visible: bool,
    /// Opacity (0.0 to 1.0)
    pub opacity: f32,
    /// Blend mode
    pub blend_mode: BlendMode,
    /// Conditional visibility
    pub visible_if: Option<String>,
}

impl Layer {
    /// Create a new layer
    #[must_use]
    pub fn new(id: String, layer_type: LayerType) -> Self {
        Self {
            id,
            layer_type,
            visible: true,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            visible_if: None,
        }
    }

    /// Render layer with data
    pub fn render(&self, data: &HashMap<String, String>) -> Result<Option<RenderedLayer>> {
        // Check conditional visibility
        if let Some(ref condition) = self.visible_if {
            if !self.evaluate_condition(condition, data) {
                return Ok(None);
            }
        }

        if !self.visible {
            return Ok(None);
        }

        let rendered_type = self.layer_type.render(data)?;

        Ok(Some(RenderedLayer {
            id: self.id.clone(),
            layer_type: rendered_type,
            opacity: self.opacity,
            blend_mode: self.blend_mode,
        }))
    }

    fn evaluate_condition(&self, condition: &str, data: &HashMap<String, String>) -> bool {
        // Simple condition evaluation (var exists and is not empty)
        if let Some(value) = data.get(condition) {
            !value.is_empty()
        } else {
            false
        }
    }
}

/// Layer type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LayerType {
    /// Rectangle layer
    Rectangle {
        /// Position
        position: [f32; 2],
        /// Size
        size: [f32; 2],
        /// Fill
        fill: TemplateFill,
        /// Stroke
        stroke: Option<TemplateStroke>,
        /// Corner radius
        corner_radius: f32,
    },
    /// Circle layer
    Circle {
        /// Center
        center: [f32; 2],
        /// Radius
        radius: f32,
        /// Fill
        fill: TemplateFill,
        /// Stroke
        stroke: Option<TemplateStroke>,
    },
    /// Text layer
    Text {
        /// Content (can use {{variable}})
        content: String,
        /// Position
        position: [f32; 2],
        /// Font family
        font_family: String,
        /// Font size
        font_size: f32,
        /// Color (can use {{variable}})
        color: String,
    },
    /// Image layer
    Image {
        /// Position
        position: [f32; 2],
        /// Size
        size: [f32; 2],
        /// Image path (can use {{variable}})
        path: String,
    },
}

impl LayerType {
    /// Render layer type with data
    pub fn render(&self, data: &HashMap<String, String>) -> Result<RenderedLayerType> {
        match self {
            Self::Rectangle {
                position,
                size,
                fill,
                stroke,
                corner_radius,
            } => Ok(RenderedLayerType::Rectangle {
                rect: Rect::new(position[0], position[1], size[0], size[1]),
                fill: fill.render(data)?,
                stroke: stroke.as_ref().map(|s| s.render(data)).transpose()?,
                corner_radius: *corner_radius,
            }),
            Self::Circle {
                center,
                radius,
                fill,
                stroke,
            } => Ok(RenderedLayerType::Circle {
                circle: Circle::new(center[0], center[1], *radius),
                fill: fill.render(data)?,
                stroke: stroke.as_ref().map(|s| s.render(data)).transpose()?,
            }),
            Self::Text {
                content,
                position,
                font_family,
                font_size,
                color,
            } => {
                let rendered_content = substitute_variables(content, data);
                let rendered_color = substitute_variables(color, data);
                let text_color = Color::from_hex(&rendered_color)?;

                Ok(RenderedLayerType::Text {
                    content: rendered_content,
                    position: Point::new(position[0], position[1]),
                    style: TextStyle::new(font_family.clone(), *font_size, text_color),
                })
            }
            Self::Image {
                position,
                size,
                path,
            } => {
                let rendered_path = substitute_variables(path, data);
                Ok(RenderedLayerType::Image {
                    rect: Rect::new(position[0], position[1], size[0], size[1]),
                    path: rendered_path,
                })
            }
        }
    }
}

/// Template fill
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TemplateFill {
    /// Solid color (can use {{variable}})
    Solid {
        /// Color value
        color: String,
    },
    /// Gradient
    Gradient {
        /// Gradient type
        gradient: TemplateGradient,
    },
}

impl TemplateFill {
    fn render(&self, data: &HashMap<String, String>) -> Result<Fill> {
        match self {
            Self::Solid { color } => {
                let rendered = substitute_variables(color, data);
                let c = Color::from_hex(&rendered)?;
                Ok(Fill::Solid(c))
            }
            Self::Gradient { gradient } => {
                let g = gradient.render(data)?;
                Ok(Fill::Gradient(g))
            }
        }
    }
}

/// Template gradient
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "gradient_type")]
pub enum TemplateGradient {
    /// Linear gradient
    Linear {
        /// Start
        start: [f32; 2],
        /// End
        end: [f32; 2],
        /// Stops
        stops: Vec<(f32, String)>,
    },
    /// Radial gradient
    Radial {
        /// Center
        center: [f32; 2],
        /// Radius
        radius: f32,
        /// Stops
        stops: Vec<(f32, String)>,
    },
}

impl TemplateGradient {
    fn render(&self, data: &HashMap<String, String>) -> Result<Gradient> {
        match self {
            Self::Linear { start, end, stops } => {
                let mut color_stops = Vec::new();
                for (pos, color) in stops {
                    let rendered = substitute_variables(color, data);
                    let c = Color::from_hex(&rendered)?;
                    color_stops.push((*pos, c));
                }
                Ok(Gradient::linear(
                    (start[0], start[1]),
                    (end[0], end[1]),
                    color_stops,
                ))
            }
            Self::Radial {
                center,
                radius,
                stops,
            } => {
                let mut color_stops = Vec::new();
                for (pos, color) in stops {
                    let rendered = substitute_variables(color, data);
                    let c = Color::from_hex(&rendered)?;
                    color_stops.push((*pos, c));
                }
                Ok(Gradient::radial(
                    (center[0], center[1]),
                    *radius,
                    color_stops,
                ))
            }
        }
    }
}

/// Template stroke
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateStroke {
    /// Color (can use {{variable}})
    pub color: String,
    /// Width
    pub width: f32,
}

impl TemplateStroke {
    fn render(&self, data: &HashMap<String, String>) -> Result<Stroke> {
        let rendered = substitute_variables(&self.color, data);
        let color = Color::from_hex(&rendered)?;
        Ok(Stroke::new(color, self.width))
    }
}

/// Blend mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlendMode {
    /// Normal
    Normal,
    /// Multiply
    Multiply,
    /// Screen
    Screen,
    /// Overlay
    Overlay,
    /// Add
    Add,
}

/// Rendered template
#[derive(Debug, Clone)]
pub struct RenderedTemplate {
    /// Resolution
    pub resolution: (u32, u32),
    /// Rendered layers
    pub layers: Vec<RenderedLayer>,
}

/// Rendered layer
#[derive(Debug, Clone)]
pub struct RenderedLayer {
    /// Layer ID
    pub id: String,
    /// Layer type
    pub layer_type: RenderedLayerType,
    /// Opacity
    pub opacity: f32,
    /// Blend mode
    pub blend_mode: BlendMode,
}

/// Rendered layer type
#[derive(Debug, Clone)]
pub enum RenderedLayerType {
    /// Rectangle
    Rectangle {
        /// Rectangle
        rect: Rect,
        /// Fill
        fill: Fill,
        /// Stroke
        stroke: Option<Stroke>,
        /// Corner radius
        corner_radius: f32,
    },
    /// Circle
    Circle {
        /// Circle
        circle: Circle,
        /// Fill
        fill: Fill,
        /// Stroke
        stroke: Option<Stroke>,
    },
    /// Text
    Text {
        /// Content
        content: String,
        /// Position
        position: Point,
        /// Style
        style: TextStyle,
    },
    /// Image
    Image {
        /// Rectangle
        rect: Rect,
        /// Image path
        path: String,
    },
}

/// Substitute template variables
fn substitute_variables(template: &str, data: &HashMap<String, String>) -> String {
    let mut result = template.to_string();
    for (key, value) in data {
        let pattern = format!("{{{{{key}}}}}");
        result = result.replace(&pattern, value);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_creation() {
        let template = Template::new("test".to_string(), (1920, 1080));
        assert_eq!(template.name, "test");
        assert_eq!(template.resolution, (1920, 1080));
    }

    #[test]
    fn test_template_add_layer() {
        let mut template = Template::new("test".to_string(), (1920, 1080));
        let layer = Layer::new(
            "bg".to_string(),
            LayerType::Rectangle {
                position: [0.0, 0.0],
                size: [1920.0, 1080.0],
                fill: TemplateFill::Solid {
                    color: "#000000".to_string(),
                },
                stroke: None,
                corner_radius: 0.0,
            },
        );
        template.add_layer(layer);
        assert_eq!(template.layers.len(), 1);
    }

    #[test]
    fn test_variable_substitution() {
        let template = "Hello {{name}}!";
        let mut data = HashMap::new();
        data.insert("name".to_string(), "World".to_string());

        let result = substitute_variables(template, &data);
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_template_render() {
        let mut template = Template::new("test".to_string(), (1920, 1080));
        let layer = Layer::new(
            "text".to_string(),
            LayerType::Text {
                content: "{{title}}".to_string(),
                position: [100.0, 100.0],
                font_family: "Arial".to_string(),
                font_size: 32.0,
                color: "#FFFFFF".to_string(),
            },
        );
        template.add_layer(layer);

        let mut data = HashMap::new();
        data.insert("title".to_string(), "Test Title".to_string());

        let rendered = template.render(&data).expect("rendered should be valid");
        assert_eq!(rendered.layers.len(), 1);
    }

    #[test]
    fn test_template_json() {
        let template = Template::new("test".to_string(), (1920, 1080));
        let json = template.to_json().expect("json should be valid");
        let loaded = Template::from_json(&json).expect("loaded should be valid");
        assert_eq!(loaded.name, template.name);
        assert_eq!(loaded.resolution, template.resolution);
    }

    #[test]
    fn test_layer_visibility() {
        let layer = Layer::new(
            "test".to_string(),
            LayerType::Rectangle {
                position: [0.0, 0.0],
                size: [100.0, 100.0],
                fill: TemplateFill::Solid {
                    color: "#000000".to_string(),
                },
                stroke: None,
                corner_radius: 0.0,
            },
        );

        let data = HashMap::new();
        let rendered = layer.render(&data).expect("rendered should be valid");
        assert!(rendered.is_some());
    }

    #[test]
    fn test_layer_conditional_visibility() {
        let mut layer = Layer::new(
            "test".to_string(),
            LayerType::Rectangle {
                position: [0.0, 0.0],
                size: [100.0, 100.0],
                fill: TemplateFill::Solid {
                    color: "#000000".to_string(),
                },
                stroke: None,
                corner_radius: 0.0,
            },
        );
        layer.visible_if = Some("show_bg".to_string());

        let data = HashMap::new();
        let rendered = layer.render(&data).expect("rendered should be valid");
        assert!(rendered.is_none());

        let mut data = HashMap::new();
        data.insert("show_bg".to_string(), "true".to_string());
        let rendered = layer.render(&data).expect("rendered should be valid");
        assert!(rendered.is_some());
    }

    #[test]
    fn test_blend_modes() {
        assert_eq!(BlendMode::Normal, BlendMode::Normal);
        assert_ne!(BlendMode::Normal, BlendMode::Multiply);
    }
}
