//! Flexible metadata schema registry for media assets.
//!
//! Operators can define named schemas that describe the structure and validation
//! rules of metadata attached to assets, collections, or any domain object.
//! A schema is a versioned set of `FieldSchema` definitions; when metadata is
//! validated against a schema each field's value is checked for type conformance
//! and constraint satisfaction.
//!
//! # Key types
//!
//! * `FieldDataType` – the data type of a schema field (String, Integer, Float, …).
//! * `FieldConstraint` – a single validation constraint (required, min/max, regex, …).
//! * `FieldSchema` – definition of a single field including its constraints.
//! * `MetadataSchema` – a versioned collection of `FieldSchema`s.
//! * `SchemaVersion` – tracks the evolution of a schema definition.
//! * `SchemaRegistry` – stores and looks up schemas by name.
//! * `ValidationResult` – the outcome of validating a metadata map against a schema.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// FieldDataType
// ---------------------------------------------------------------------------

/// The data type of a metadata field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldDataType {
    /// UTF-8 text.
    String,
    /// 64-bit signed integer represented as a string in metadata maps.
    Integer,
    /// 64-bit IEEE-754 float represented as a string.
    Float,
    /// Boolean: accepts `"true"` / `"false"` (case-insensitive).
    Boolean,
    /// ISO-8601 date (YYYY-MM-DD).
    Date,
    /// ISO-8601 datetime.
    DateTime,
    /// Absolute URI.
    Uri,
    /// RFC 5321 email address.
    Email,
    /// JSON-encoded object or array.
    Json,
    /// One value from an enumerated list.
    Enum(Vec<String>),
    /// Comma-separated list of values from an enumerated list.
    MultiEnum(Vec<String>),
}

impl FieldDataType {
    /// Human-readable name for the type.
    #[must_use]
    pub fn type_name(&self) -> &str {
        match self {
            Self::String => "string",
            Self::Integer => "integer",
            Self::Float => "float",
            Self::Boolean => "boolean",
            Self::Date => "date",
            Self::DateTime => "datetime",
            Self::Uri => "uri",
            Self::Email => "email",
            Self::Json => "json",
            Self::Enum(_) => "enum",
            Self::MultiEnum(_) => "multi_enum",
        }
    }

    /// Validate that a raw string value conforms to this data type.
    #[must_use]
    pub fn validate_value(&self, value: &str) -> bool {
        match self {
            Self::String | Self::Json => true,
            Self::Integer => value.trim().parse::<i64>().is_ok(),
            Self::Float => value.trim().parse::<f64>().is_ok(),
            Self::Boolean => matches!(value.to_lowercase().as_str(), "true" | "false"),
            Self::Date => {
                // Require YYYY-MM-DD
                let parts: Vec<&str> = value.splitn(3, '-').collect();
                parts.len() == 3
                    && parts[0].len() == 4
                    && parts[1].len() == 2
                    && parts[2].len() == 2
                    && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit()))
            }
            Self::DateTime => {
                // Require at minimum the date portion
                let date_part = value.split('T').next().unwrap_or("");
                let parts: Vec<&str> = date_part.splitn(3, '-').collect();
                parts.len() == 3
                    && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit()))
            }
            Self::Uri => value.contains("://") && !value.is_empty(),
            Self::Email => value.contains('@') && value.contains('.'),
            Self::Enum(allowed) => allowed.iter().any(|a| a == value),
            Self::MultiEnum(allowed) => value
                .split(',')
                .map(str::trim)
                .all(|v| allowed.iter().any(|a| a == v)),
        }
    }
}

// ---------------------------------------------------------------------------
// FieldConstraint
// ---------------------------------------------------------------------------

/// A single validation rule applied to a field value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldConstraint {
    /// The field must be present and non-empty.
    Required,
    /// Minimum character length (String / Json fields).
    MinLength(usize),
    /// Maximum character length.
    MaxLength(usize),
    /// Minimum numeric value (applies to Integer / Float fields).
    MinValue(f64),
    /// Maximum numeric value.
    MaxValue(f64),
    /// The value must match this regular-expression pattern.
    /// Stored as a string; validated via a simple prefix/suffix/contains match
    /// to avoid pulling in a regex crate.
    Pattern(String),
    /// The value must not be blank (for optional fields).
    NotBlank,
    /// The value must be unique across the metadata store
    /// (informational — actual uniqueness check is the caller's responsibility).
    Unique,
}

impl FieldConstraint {
    /// Validate a raw string value against this constraint.
    ///
    /// Returns `Ok(())` if the constraint is satisfied, or `Err` with a
    /// human-readable message.
    ///
    /// # Errors
    ///
    /// Returns a `String` describing the violated constraint.
    pub fn check(&self, field_name: &str, value: Option<&str>) -> Result<(), String> {
        match self {
            Self::Required => {
                match value {
                    None | Some("") => Err(format!("Field '{field_name}' is required")),
                    _ => Ok(()),
                }
            }
            Self::NotBlank => {
                match value {
                    Some(v) if v.trim().is_empty() => {
                        Err(format!("Field '{field_name}' must not be blank"))
                    }
                    _ => Ok(()),
                }
            }
            Self::MinLength(min) => {
                if let Some(v) = value {
                    if v.len() < *min {
                        return Err(format!(
                            "Field '{field_name}' must be at least {min} characters long (got {})",
                            v.len()
                        ));
                    }
                }
                Ok(())
            }
            Self::MaxLength(max) => {
                if let Some(v) = value {
                    if v.len() > *max {
                        return Err(format!(
                            "Field '{field_name}' must be at most {max} characters long (got {})",
                            v.len()
                        ));
                    }
                }
                Ok(())
            }
            Self::MinValue(min) => {
                if let Some(v) = value {
                    if let Ok(n) = v.trim().parse::<f64>() {
                        if n < *min {
                            return Err(format!(
                                "Field '{field_name}' must be ≥ {min} (got {n})"
                            ));
                        }
                    }
                }
                Ok(())
            }
            Self::MaxValue(max) => {
                if let Some(v) = value {
                    if let Ok(n) = v.trim().parse::<f64>() {
                        if n > *max {
                            return Err(format!(
                                "Field '{field_name}' must be ≤ {max} (got {n})"
                            ));
                        }
                    }
                }
                Ok(())
            }
            Self::Pattern(pattern) => {
                if let Some(v) = value {
                    // Simple pattern matching: treat as substring search
                    if !v.contains(pattern.as_str()) {
                        return Err(format!(
                            "Field '{field_name}' must match pattern '{pattern}'"
                        ));
                    }
                }
                Ok(())
            }
            // Unique is informational only — always passes at this layer
            Self::Unique => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// FieldSchema
// ---------------------------------------------------------------------------

/// Definition of a single metadata field including its type and constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    /// Machine-readable field name (snake_case).
    pub name: String,
    /// Human-readable label.
    pub label: String,
    /// Optional description.
    pub description: Option<String>,
    /// Data type of the field.
    pub data_type: FieldDataType,
    /// Ordered list of constraints applied to values of this field.
    pub constraints: Vec<FieldConstraint>,
    /// Default value if the field is omitted.
    pub default_value: Option<String>,
    /// Whether this field is shown in compact/summary views.
    pub is_summary_field: bool,
    /// Display order (lower = earlier).
    pub display_order: u32,
}

impl FieldSchema {
    /// Create a new field schema.
    #[must_use]
    pub fn new(name: impl Into<String>, label: impl Into<String>, data_type: FieldDataType) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            description: None,
            data_type,
            constraints: vec![],
            default_value: None,
            is_summary_field: false,
            display_order: 0,
        }
    }

    /// Add a constraint to this field.
    #[must_use]
    pub fn with_constraint(mut self, constraint: FieldConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Mark the field as required (shortcut for `FieldConstraint::Required`).
    #[must_use]
    pub fn required(self) -> Self {
        self.with_constraint(FieldConstraint::Required)
    }

    /// Set the display order.
    #[must_use]
    pub fn with_order(mut self, order: u32) -> Self {
        self.display_order = order;
        self
    }

    /// Set a default value.
    #[must_use]
    pub fn with_default(mut self, default: impl Into<String>) -> Self {
        self.default_value = Some(default.into());
        self
    }

    /// Validate a raw string value (or absence) against this field's data type
    /// and constraints, returning all errors found.
    #[must_use]
    pub fn validate(&self, value: Option<&str>) -> Vec<String> {
        let mut errors = Vec::new();

        // Type check (only when a value is present)
        if let Some(v) = value {
            if !v.is_empty() && !self.data_type.validate_value(v) {
                errors.push(format!(
                    "Field '{}' has invalid value '{}' for type {}",
                    self.name,
                    v,
                    self.data_type.type_name()
                ));
            }
        }

        // Constraint checks
        for constraint in &self.constraints {
            if let Err(msg) = constraint.check(&self.name, value) {
                errors.push(msg);
            }
        }

        errors
    }
}

// ---------------------------------------------------------------------------
// SchemaVersion
// ---------------------------------------------------------------------------

/// A record of a schema version change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaVersion {
    /// Sequential version number (starts at 1).
    pub version: u32,
    /// Who made the change.
    pub changed_by: Uuid,
    /// Summary of what changed in this version.
    pub change_summary: String,
    /// When the change was recorded.
    pub changed_at: DateTime<Utc>,
}

impl SchemaVersion {
    /// Create a new schema version entry.
    #[must_use]
    pub fn new(version: u32, changed_by: Uuid, summary: impl Into<String>) -> Self {
        Self {
            version,
            changed_by,
            change_summary: summary.into(),
            changed_at: Utc::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// MetadataSchema
// ---------------------------------------------------------------------------

/// A versioned set of field definitions describing the structure of asset metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataSchema {
    /// Unique schema ID.
    pub id: Uuid,
    /// Unique machine-readable name (e.g. `"broadcast_asset"`, `"music_track"`).
    pub name: String,
    /// Human-readable label.
    pub label: String,
    /// Optional description.
    pub description: Option<String>,
    /// Current version number (incremented on each `update_field` / `remove_field`).
    pub current_version: u32,
    /// Ordered field definitions keyed by field name.
    fields: HashMap<String, FieldSchema>,
    /// History of version changes.
    version_history: Vec<SchemaVersion>,
    /// Who created this schema.
    pub created_by: Uuid,
    /// When the schema was first created.
    pub created_at: DateTime<Utc>,
}

impl MetadataSchema {
    /// Create a new empty schema.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        label: impl Into<String>,
        created_by: Uuid,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            label: label.into(),
            description: None,
            current_version: 1,
            fields: HashMap::new(),
            version_history: vec![SchemaVersion::new(1, created_by, "Initial schema")],
            created_by,
            created_at: now,
        }
    }

    /// Add or replace a field definition.  Bumps the schema version.
    pub fn upsert_field(&mut self, field: FieldSchema, changed_by: Uuid, summary: impl Into<String>) {
        self.fields.insert(field.name.clone(), field);
        self.current_version += 1;
        self.version_history.push(SchemaVersion::new(
            self.current_version,
            changed_by,
            summary,
        ));
    }

    /// Remove a field by name.  Returns `true` if the field existed.
    pub fn remove_field(&mut self, field_name: &str, changed_by: Uuid, summary: impl Into<String>) -> bool {
        let removed = self.fields.remove(field_name).is_some();
        if removed {
            self.current_version += 1;
            self.version_history.push(SchemaVersion::new(
                self.current_version,
                changed_by,
                summary,
            ));
        }
        removed
    }

    /// Retrieve a field by name.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&FieldSchema> {
        self.fields.get(name)
    }

    /// All field definitions sorted by `display_order`.
    #[must_use]
    pub fn fields_ordered(&self) -> Vec<&FieldSchema> {
        let mut fields: Vec<&FieldSchema> = self.fields.values().collect();
        fields.sort_by_key(|f| (f.display_order, f.name.as_str()));
        fields
    }

    /// Number of defined fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Version history (oldest first).
    #[must_use]
    pub fn version_history(&self) -> &[SchemaVersion] {
        &self.version_history
    }

    /// Validate a metadata map against this schema.
    ///
    /// Returns a [`ValidationResult`] containing any errors found.
    #[must_use]
    pub fn validate(&self, metadata: &HashMap<String, String>) -> ValidationResult {
        let mut errors: HashMap<String, Vec<String>> = HashMap::new();

        for field in self.fields.values() {
            let value = metadata.get(&field.name).map(String::as_str);
            let field_errors = field.validate(value);
            if !field_errors.is_empty() {
                errors.insert(field.name.clone(), field_errors);
            }
        }

        // Detect unknown fields
        let mut unknown: Vec<String> = Vec::new();
        for key in metadata.keys() {
            if !self.fields.contains_key(key.as_str()) {
                unknown.push(key.clone());
            }
        }

        ValidationResult { errors, unknown_fields: unknown }
    }
}

// ---------------------------------------------------------------------------
// ValidationResult
// ---------------------------------------------------------------------------

/// The outcome of validating a metadata map against a schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Per-field validation errors.  Empty map = all fields valid.
    pub errors: HashMap<String, Vec<String>>,
    /// Fields present in the metadata but not defined in the schema.
    pub unknown_fields: Vec<String>,
}

impl ValidationResult {
    /// Whether validation passed (no errors).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Total number of validation errors across all fields.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.values().map(Vec::len).sum()
    }
}

// ---------------------------------------------------------------------------
// SchemaRegistry
// ---------------------------------------------------------------------------

/// Error type for schema registry operations.
#[derive(Debug, thiserror::Error)]
pub enum SchemaRegistryError {
    /// A schema with this name already exists.
    #[error("Schema already exists: {0}")]
    AlreadyExists(String),
    /// No schema found for the given name.
    #[error("Schema not found: {0}")]
    NotFound(String),
}

/// In-memory registry mapping schema names to `MetadataSchema` instances.
#[derive(Default)]
pub struct SchemaRegistry {
    schemas: HashMap<String, MetadataSchema>,
}

impl SchemaRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new schema.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaRegistryError::AlreadyExists`] if a schema with the same
    /// name is already registered.
    pub fn register(&mut self, schema: MetadataSchema) -> Result<(), SchemaRegistryError> {
        if self.schemas.contains_key(&schema.name) {
            return Err(SchemaRegistryError::AlreadyExists(schema.name.clone()));
        }
        self.schemas.insert(schema.name.clone(), schema);
        Ok(())
    }

    /// Replace an existing schema (e.g. after adding fields).
    ///
    /// # Errors
    ///
    /// Returns [`SchemaRegistryError::NotFound`] if the schema does not exist.
    pub fn update(&mut self, schema: MetadataSchema) -> Result<(), SchemaRegistryError> {
        if !self.schemas.contains_key(&schema.name) {
            return Err(SchemaRegistryError::NotFound(schema.name.clone()));
        }
        self.schemas.insert(schema.name.clone(), schema);
        Ok(())
    }

    /// Get a schema by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&MetadataSchema> {
        self.schemas.get(name)
    }

    /// Get a mutable schema by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut MetadataSchema> {
        self.schemas.get_mut(name)
    }

    /// Remove a schema by name.  Returns `true` if the schema existed.
    pub fn deregister(&mut self, name: &str) -> bool {
        self.schemas.remove(name).is_some()
    }

    /// All registered schema names.
    #[must_use]
    pub fn schema_names(&self) -> Vec<&str> {
        self.schemas.keys().map(String::as_str).collect()
    }

    /// Number of registered schemas.
    #[must_use]
    pub fn count(&self) -> usize {
        self.schemas.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_user() -> Uuid {
        Uuid::new_v4()
    }

    #[test]
    fn field_data_type_validate_integer() {
        let t = FieldDataType::Integer;
        assert!(t.validate_value("42"));
        assert!(t.validate_value("-7"));
        assert!(!t.validate_value("not_an_int"));
        assert!(!t.validate_value("3.14"));
    }

    #[test]
    fn field_data_type_validate_boolean() {
        let t = FieldDataType::Boolean;
        assert!(t.validate_value("true"));
        assert!(t.validate_value("FALSE"));
        assert!(!t.validate_value("yes"));
    }

    #[test]
    fn field_data_type_validate_enum() {
        let t = FieldDataType::Enum(vec!["draft".into(), "published".into()]);
        assert!(t.validate_value("draft"));
        assert!(!t.validate_value("archived"));
    }

    #[test]
    fn field_constraint_required() {
        let c = FieldConstraint::Required;
        assert!(c.check("title", None).is_err());
        assert!(c.check("title", Some("")).is_err());
        assert!(c.check("title", Some("Hello")).is_ok());
    }

    #[test]
    fn field_constraint_min_max_length() {
        let min = FieldConstraint::MinLength(3);
        let max = FieldConstraint::MaxLength(10);

        assert!(min.check("tag", Some("ab")).is_err());
        assert!(min.check("tag", Some("abc")).is_ok());
        assert!(max.check("tag", Some("this is too long indeed")).is_err());
        assert!(max.check("tag", Some("short")).is_ok());
    }

    #[test]
    fn field_constraint_numeric_range() {
        let min = FieldConstraint::MinValue(0.0);
        let max = FieldConstraint::MaxValue(100.0);

        assert!(min.check("score", Some("-1")).is_err());
        assert!(min.check("score", Some("0")).is_ok());
        assert!(max.check("score", Some("101")).is_err());
        assert!(max.check("score", Some("100")).is_ok());
    }

    #[test]
    fn schema_validate_missing_required_field() {
        let user = make_user();
        let mut schema = MetadataSchema::new("test_schema", "Test Schema", user);

        let field = FieldSchema::new("title", "Title", FieldDataType::String).required();
        schema.upsert_field(field, user, "add title field");

        let metadata: HashMap<String, String> = HashMap::new();
        let result = schema.validate(&metadata);

        assert!(!result.is_valid());
        assert!(result.errors.contains_key("title"));
    }

    #[test]
    fn schema_validate_valid_metadata() {
        let user = make_user();
        let mut schema = MetadataSchema::new("track_schema", "Music Track", user);

        let title_field = FieldSchema::new("title", "Title", FieldDataType::String).required();
        let year_field = FieldSchema::new("year", "Year", FieldDataType::Integer)
            .with_constraint(FieldConstraint::MinValue(1900.0))
            .with_constraint(FieldConstraint::MaxValue(2100.0));

        schema.upsert_field(title_field, user, "add title");
        schema.upsert_field(year_field, user, "add year");

        let mut metadata = HashMap::new();
        metadata.insert("title".into(), "Suite in D minor".into());
        metadata.insert("year".into(), "1985".into());

        let result = schema.validate(&metadata);
        assert!(result.is_valid(), "errors: {:?}", result.errors);
    }

    #[test]
    fn schema_detects_unknown_fields() {
        let user = make_user();
        let schema = MetadataSchema::new("empty", "Empty Schema", user);

        let mut metadata = HashMap::new();
        metadata.insert("surprise_field".into(), "value".into());

        let result = schema.validate(&metadata);
        assert!(result.unknown_fields.contains(&"surprise_field".to_string()));
    }

    #[test]
    fn schema_version_bumps_on_upsert() {
        let user = make_user();
        let mut schema = MetadataSchema::new("v_test", "Version Test", user);
        assert_eq!(schema.current_version, 1);

        let f = FieldSchema::new("foo", "Foo", FieldDataType::String);
        schema.upsert_field(f, user, "add foo");
        assert_eq!(schema.current_version, 2);

        schema.remove_field("foo", user, "remove foo");
        assert_eq!(schema.current_version, 3);
        assert_eq!(schema.version_history().len(), 3);
    }

    #[test]
    fn registry_register_and_retrieve() {
        let user = make_user();
        let mut registry = SchemaRegistry::new();

        let schema = MetadataSchema::new("asset_schema", "Asset Schema", user);
        registry.register(schema).expect("register should succeed");

        assert_eq!(registry.count(), 1);
        assert!(registry.get("asset_schema").is_some());

        // Duplicate registration should fail
        let dup = MetadataSchema::new("asset_schema", "Duplicate", user);
        assert!(registry.register(dup).is_err());
    }

    #[test]
    fn registry_deregister() {
        let user = make_user();
        let mut registry = SchemaRegistry::new();

        let schema = MetadataSchema::new("tmp_schema", "Temp", user);
        registry.register(schema).unwrap();
        assert!(registry.deregister("tmp_schema"));
        assert!(!registry.deregister("tmp_schema")); // already gone
    }
}
