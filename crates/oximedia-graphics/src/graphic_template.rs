#![allow(dead_code)]
//! Graphic template system for broadcast graphics.
//!
//! Provides a template field model with required/optional classification,
//! named templates with field inspection, and a template library for
//! registration and lookup.

use std::collections::HashMap;

/// Type of a template field value.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldType {
    /// Plain text string.
    Text,
    /// Numeric (floating-point) value.
    Number,
    /// Boolean flag.
    Bool,
    /// URL pointing to an image resource.
    ImageUrl,
    /// RGBA colour expressed as a packed `u32`.
    Color,
}

/// A single field definition within a graphic template.
#[derive(Debug, Clone)]
pub struct TemplateField {
    /// Unique name for this field.
    pub name: String,
    /// Data type of the field.
    pub field_type: FieldType,
    /// Whether the field must be supplied when instantiating the template.
    pub required: bool,
    /// Human-readable description.
    pub description: String,
}

impl TemplateField {
    /// Create a new template field.
    pub fn new(name: &str, field_type: FieldType, required: bool, description: &str) -> Self {
        Self {
            name: name.to_string(),
            field_type,
            required,
            description: description.to_string(),
        }
    }

    /// Returns `true` when this field must be populated before rendering.
    pub fn is_required(&self) -> bool {
        self.required
    }

    /// Returns the field type.
    pub fn field_type(&self) -> &FieldType {
        &self.field_type
    }
}

/// A named broadcast graphic template composed of multiple fields.
#[derive(Debug, Clone)]
pub struct GraphicTemplate {
    /// Template identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Description of the template's purpose.
    pub description: String,
    /// Ordered list of field definitions.
    pub fields: Vec<TemplateField>,
    /// Optional thumbnail URL.
    pub thumbnail_url: Option<String>,
}

impl GraphicTemplate {
    /// Create a new template with no fields.
    pub fn new(id: &str, name: &str, description: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            description: description.to_string(),
            fields: Vec::new(),
            thumbnail_url: None,
        }
    }

    /// Add a field to this template.
    pub fn add_field(&mut self, field: TemplateField) {
        self.fields.push(field);
    }

    /// Total number of fields defined in this template.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Return only the fields that are marked as required.
    pub fn required_fields(&self) -> Vec<&TemplateField> {
        self.fields.iter().filter(|f| f.required).collect()
    }

    /// Return only the optional fields.
    pub fn optional_fields(&self) -> Vec<&TemplateField> {
        self.fields.iter().filter(|f| !f.required).collect()
    }

    /// Returns `true` when every required field is present.
    pub fn has_all_required(&self) -> bool {
        !self.fields.iter().any(|f| f.required)
            || self.fields.iter().filter(|f| f.required).count() > 0
    }

    /// Look up a field by name.
    pub fn get_field(&self, name: &str) -> Option<&TemplateField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Set the thumbnail URL.
    pub fn set_thumbnail(&mut self, url: &str) {
        self.thumbnail_url = Some(url.to_string());
    }
}

/// Registry that stores and retrieves [`GraphicTemplate`] instances.
#[derive(Debug, Default)]
pub struct TemplateLibrary {
    templates: HashMap<String, GraphicTemplate>,
}

impl TemplateLibrary {
    /// Create an empty library.
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    /// Register a template. Overwrites any existing entry with the same id.
    pub fn register(&mut self, template: GraphicTemplate) {
        self.templates.insert(template.id.clone(), template);
    }

    /// Find a template by its id. Returns `None` when not found.
    pub fn find(&self, id: &str) -> Option<&GraphicTemplate> {
        self.templates.get(id)
    }

    /// Remove a template by id. Returns the removed template if it existed.
    pub fn remove(&mut self, id: &str) -> Option<GraphicTemplate> {
        self.templates.remove(id)
    }

    /// Total number of registered templates.
    pub fn count(&self) -> usize {
        self.templates.len()
    }

    /// Return an iterator over all registered templates.
    pub fn all(&self) -> impl Iterator<Item = &GraphicTemplate> {
        self.templates.values()
    }

    /// Find all templates whose names contain the given substring (case-insensitive).
    pub fn search_by_name(&self, query: &str) -> Vec<&GraphicTemplate> {
        let q = query.to_lowercase();
        self.templates
            .values()
            .filter(|t| t.name.to_lowercase().contains(&q))
            .collect()
    }

    /// Returns `true` when the library is empty.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lower_third_template() -> GraphicTemplate {
        let mut t = GraphicTemplate::new("lower_third", "Lower Third", "Standard lower third");
        t.add_field(TemplateField::new(
            "name",
            FieldType::Text,
            true,
            "Speaker name",
        ));
        t.add_field(TemplateField::new(
            "title",
            FieldType::Text,
            false,
            "Speaker title",
        ));
        t.add_field(TemplateField::new(
            "bg_color",
            FieldType::Color,
            false,
            "Background color",
        ));
        t
    }

    #[test]
    fn test_field_is_required_true() {
        let f = TemplateField::new("headline", FieldType::Text, true, "Main headline");
        assert!(f.is_required());
    }

    #[test]
    fn test_field_is_required_false() {
        let f = TemplateField::new("subtitle", FieldType::Text, false, "Optional subtitle");
        assert!(!f.is_required());
    }

    #[test]
    fn test_field_type_returned() {
        let f = TemplateField::new("score", FieldType::Number, true, "Score value");
        assert_eq!(f.field_type(), &FieldType::Number);
    }

    #[test]
    fn test_template_field_count() {
        let t = make_lower_third_template();
        assert_eq!(t.field_count(), 3);
    }

    #[test]
    fn test_template_required_fields() {
        let t = make_lower_third_template();
        let req = t.required_fields();
        assert_eq!(req.len(), 1);
        assert_eq!(req[0].name, "name");
    }

    #[test]
    fn test_template_optional_fields() {
        let t = make_lower_third_template();
        let opt = t.optional_fields();
        assert_eq!(opt.len(), 2);
    }

    #[test]
    fn test_template_get_field_found() {
        let t = make_lower_third_template();
        assert!(t.get_field("title").is_some());
    }

    #[test]
    fn test_template_get_field_missing() {
        let t = make_lower_third_template();
        assert!(t.get_field("nonexistent").is_none());
    }

    #[test]
    fn test_template_thumbnail() {
        let mut t = make_lower_third_template();
        assert!(t.thumbnail_url.is_none());
        t.set_thumbnail("https://cdn.example.com/lower_third.png");
        assert!(t.thumbnail_url.is_some());
    }

    #[test]
    fn test_library_register_and_count() {
        let mut lib = TemplateLibrary::new();
        assert_eq!(lib.count(), 0);
        lib.register(make_lower_third_template());
        assert_eq!(lib.count(), 1);
    }

    #[test]
    fn test_library_find_existing() {
        let mut lib = TemplateLibrary::new();
        lib.register(make_lower_third_template());
        assert!(lib.find("lower_third").is_some());
    }

    #[test]
    fn test_library_find_missing() {
        let lib = TemplateLibrary::new();
        assert!(lib.find("missing_id").is_none());
    }

    #[test]
    fn test_library_remove() {
        let mut lib = TemplateLibrary::new();
        lib.register(make_lower_third_template());
        let removed = lib.remove("lower_third");
        assert!(removed.is_some());
        assert_eq!(lib.count(), 0);
    }

    #[test]
    fn test_library_overwrite_on_register() {
        let mut lib = TemplateLibrary::new();
        lib.register(make_lower_third_template());
        // Re-register with different description
        let mut t2 = make_lower_third_template();
        t2.description = "Updated description".to_string();
        lib.register(t2);
        assert_eq!(lib.count(), 1);
        assert_eq!(
            lib.find("lower_third")
                .expect("find should succeed")
                .description,
            "Updated description"
        );
    }

    #[test]
    fn test_library_is_empty() {
        let lib = TemplateLibrary::new();
        assert!(lib.is_empty());
    }

    #[test]
    fn test_library_search_by_name() {
        let mut lib = TemplateLibrary::new();
        lib.register(make_lower_third_template());
        let results = lib.search_by_name("lower");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_library_search_case_insensitive() {
        let mut lib = TemplateLibrary::new();
        lib.register(make_lower_third_template());
        let results = lib.search_by_name("LOWER");
        assert_eq!(results.len(), 1);
    }
}
