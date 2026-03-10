//! Caption style templates and presets

use crate::types::{Alignment, CaptionStyle, Color, FontStyle, FontWeight, TextDecoration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Caption style template
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Template {
    /// Template name
    pub name: String,
    /// Description
    pub description: String,
    /// Style
    pub style: CaptionStyle,
    /// Broadcaster-specific (e.g., "Netflix", "BBC")
    pub broadcaster: Option<String>,
}

/// Template library
pub struct TemplateLibrary {
    templates: HashMap<String, Template>,
}

impl TemplateLibrary {
    /// Create a new template library with default templates
    #[must_use]
    pub fn new() -> Self {
        let mut library = Self {
            templates: HashMap::new(),
        };

        library.add_default_templates();
        library
    }

    /// Add a template
    pub fn add_template(&mut self, template: Template) {
        self.templates.insert(template.name.clone(), template);
    }

    /// Get a template by name
    #[must_use]
    pub fn get_template(&self, name: &str) -> Option<&Template> {
        self.templates.get(name)
    }

    /// List all template names
    #[must_use]
    pub fn list_templates(&self) -> Vec<String> {
        self.templates.keys().cloned().collect()
    }

    /// Add default templates
    fn add_default_templates(&mut self) {
        // Standard white text on black background
        self.add_template(Template {
            name: "Standard".to_string(),
            description: "Standard white text on black background".to_string(),
            style: CaptionStyle {
                font_family: "Arial".to_string(),
                font_size: 32,
                font_weight: FontWeight::Normal,
                font_style: FontStyle::Normal,
                text_decoration: TextDecoration::default(),
                color: Color::white(),
                background_color: Some(Color::new(0, 0, 0, 180)),
                outline_color: Some(Color::black()),
                outline_width: 1,
                shadow_color: Some(Color::new(0, 0, 0, 128)),
                shadow_offset: (2, 2),
                alignment: Alignment::Center,
            },
            broadcaster: None,
        });

        // Netflix style
        self.add_template(Template {
            name: "Netflix".to_string(),
            description: "Netflix subtitle style".to_string(),
            style: CaptionStyle {
                font_family: "Arial".to_string(),
                font_size: 36,
                font_weight: FontWeight::Normal,
                font_style: FontStyle::Normal,
                text_decoration: TextDecoration::default(),
                color: Color::white(),
                background_color: None,
                outline_color: Some(Color::black()),
                outline_width: 2,
                shadow_color: Some(Color::new(0, 0, 0, 200)),
                shadow_offset: (2, 2),
                alignment: Alignment::Center,
            },
            broadcaster: Some("Netflix".to_string()),
        });

        // BBC iPlayer style
        self.add_template(Template {
            name: "BBC".to_string(),
            description: "BBC iPlayer subtitle style".to_string(),
            style: CaptionStyle {
                font_family: "Arial".to_string(),
                font_size: 34,
                font_weight: FontWeight::Normal,
                font_style: FontStyle::Normal,
                text_decoration: TextDecoration::default(),
                color: Color::white(),
                background_color: Some(Color::new(0, 0, 0, 200)),
                outline_color: None,
                outline_width: 0,
                shadow_color: None,
                shadow_offset: (0, 0),
                alignment: Alignment::Center,
            },
            broadcaster: Some("BBC".to_string()),
        });

        // YouTube style
        self.add_template(Template {
            name: "YouTube".to_string(),
            description: "YouTube subtitle style".to_string(),
            style: CaptionStyle {
                font_family: "Roboto".to_string(),
                font_size: 32,
                font_weight: FontWeight::Normal,
                font_style: FontStyle::Normal,
                text_decoration: TextDecoration::default(),
                color: Color::white(),
                background_color: Some(Color::new(0, 0, 0, 150)),
                outline_color: Some(Color::black()),
                outline_width: 1,
                shadow_color: None,
                shadow_offset: (0, 0),
                alignment: Alignment::Center,
            },
            broadcaster: Some("YouTube".to_string()),
        });

        // High contrast (for accessibility)
        self.add_template(Template {
            name: "High Contrast".to_string(),
            description: "High contrast style for accessibility".to_string(),
            style: CaptionStyle {
                font_family: "Arial".to_string(),
                font_size: 38,
                font_weight: FontWeight::Bold,
                font_style: FontStyle::Normal,
                text_decoration: TextDecoration::default(),
                color: Color::white(),
                background_color: Some(Color::black()),
                outline_color: None,
                outline_width: 0,
                shadow_color: None,
                shadow_offset: (0, 0),
                alignment: Alignment::Center,
            },
            broadcaster: None,
        });

        // Yellow on black (for hard of hearing)
        self.add_template(Template {
            name: "Yellow on Black".to_string(),
            description: "Yellow text on black background for hard of hearing".to_string(),
            style: CaptionStyle {
                font_family: "Arial".to_string(),
                font_size: 34,
                font_weight: FontWeight::Bold,
                font_style: FontStyle::Normal,
                text_decoration: TextDecoration::default(),
                color: Color::rgb(255, 255, 0),
                background_color: Some(Color::black()),
                outline_color: None,
                outline_width: 0,
                shadow_color: None,
                shadow_offset: (0, 0),
                alignment: Alignment::Center,
            },
            broadcaster: None,
        });

        // Transparent background
        self.add_template(Template {
            name: "Transparent".to_string(),
            description: "White text with no background".to_string(),
            style: CaptionStyle {
                font_family: "Arial".to_string(),
                font_size: 32,
                font_weight: FontWeight::Bold,
                font_style: FontStyle::Normal,
                text_decoration: TextDecoration::default(),
                color: Color::white(),
                background_color: None,
                outline_color: Some(Color::black()),
                outline_width: 3,
                shadow_color: Some(Color::new(0, 0, 0, 200)),
                shadow_offset: (3, 3),
                alignment: Alignment::Center,
            },
            broadcaster: None,
        });
    }

    /// Save templates to JSON
    pub fn save_to_json(&self) -> serde_json::Result<String> {
        let templates: Vec<&Template> = self.templates.values().collect();
        serde_json::to_string_pretty(&templates)
    }

    /// Load templates from JSON
    pub fn load_from_json(&mut self, json: &str) -> serde_json::Result<()> {
        let templates: Vec<Template> = serde_json::from_str(json)?;
        for template in templates {
            self.add_template(template);
        }
        Ok(())
    }
}

impl Default for TemplateLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_library() {
        let library = TemplateLibrary::new();
        assert!(library.get_template("Standard").is_some());
        assert!(library.get_template("Netflix").is_some());
        assert!(library.get_template("BBC").is_some());
    }

    #[test]
    fn test_custom_template() {
        let mut library = TemplateLibrary::new();

        let custom = Template {
            name: "Custom".to_string(),
            description: "Custom style".to_string(),
            style: CaptionStyle::default(),
            broadcaster: None,
        };

        library.add_template(custom);
        assert!(library.get_template("Custom").is_some());
    }

    #[test]
    fn test_json_serialization() {
        let library = TemplateLibrary::new();
        let json = library
            .save_to_json()
            .expect("JSON serialization should succeed");
        assert!(!json.is_empty());

        let mut new_library = TemplateLibrary {
            templates: HashMap::new(),
        };
        new_library
            .load_from_json(&json)
            .expect("JSON loading should succeed");
        assert!(!new_library.templates.is_empty());
    }
}
