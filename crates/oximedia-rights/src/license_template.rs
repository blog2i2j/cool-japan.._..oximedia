//! License-agreement template engine.
//!
//! Provides parameterised templates for common license types (royalty-free,
//! rights-managed, editorial-only, etc.) that can be instantiated with
//! specific party names, territories, dates, and financial terms.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ── LicenseKind ────────────────────────────────────────────────────────────

/// High-level license type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LicenseKind {
    /// Royalty-free: one-time fee, unlimited use within scope.
    RoyaltyFree,
    /// Rights-managed: fee depends on specific usage.
    RightsManaged,
    /// Editorial use only.
    EditorialOnly,
    /// Creative Commons (variant specified in template fields).
    CreativeCommons,
    /// Exclusive license: licensee is the sole user.
    Exclusive,
    /// Non-exclusive: multiple licensees permitted.
    NonExclusive,
}

impl LicenseKind {
    /// Whether this license kind typically requires per-use fees.
    #[must_use]
    pub fn requires_per_use_fee(&self) -> bool {
        matches!(self, Self::RightsManaged)
    }

    /// Whether this license restricts usage context.
    #[must_use]
    pub fn has_usage_restriction(&self) -> bool {
        matches!(self, Self::EditorialOnly | Self::CreativeCommons)
    }
}

// ── TemplateField ──────────────────────────────────────────────────────────

/// A variable field inside a license template.
#[derive(Debug, Clone)]
pub struct TemplateField {
    /// Machine-readable field key (e.g. `"licensor_name"`).
    pub key: String,
    /// Human-readable label.
    pub label: String,
    /// Whether the field is mandatory.
    pub required: bool,
    /// Optional default value.
    pub default: Option<String>,
}

impl TemplateField {
    /// Create a new required field with no default.
    #[must_use]
    pub fn required(key: &str, label: &str) -> Self {
        Self {
            key: key.to_string(),
            label: label.to_string(),
            required: true,
            default: None,
        }
    }

    /// Create a new optional field with a default.
    #[must_use]
    pub fn optional(key: &str, label: &str, default: &str) -> Self {
        Self {
            key: key.to_string(),
            label: label.to_string(),
            required: false,
            default: Some(default.to_string()),
        }
    }

    /// Resolve the field value from a provided map, falling back to the
    /// default. Returns `None` only if the field is missing and has no default.
    #[must_use]
    pub fn resolve(&self, values: &HashMap<String, String>) -> Option<String> {
        values
            .get(&self.key)
            .cloned()
            .or_else(|| self.default.clone())
    }
}

// ── LicenseTemplate ────────────────────────────────────────────────────────

/// A parameterised license template.
///
/// The template body is a string containing `{{field_key}}` placeholders
/// that are replaced when the template is instantiated.
#[derive(Debug, Clone)]
pub struct LicenseTemplate {
    /// Template identifier.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// Kind of license.
    pub kind: LicenseKind,
    /// Defined fields.
    pub fields: Vec<TemplateField>,
    /// Template body with `{{key}}` placeholders.
    pub body: String,
}

impl LicenseTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(id: &str, title: &str, kind: LicenseKind, body: &str) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            kind,
            fields: Vec::new(),
            body: body.to_string(),
        }
    }

    /// Add a field to the template.
    #[must_use]
    pub fn with_field(mut self, field: TemplateField) -> Self {
        self.fields.push(field);
        self
    }

    /// Instantiate the template with the given values.
    ///
    /// Returns `Err` if a required field is missing and has no default.
    pub fn instantiate(
        &self,
        values: &HashMap<String, String>,
    ) -> std::result::Result<String, String> {
        let mut output = self.body.clone();
        for field in &self.fields {
            let resolved = field.resolve(values).ok_or_else(|| {
                format!("Missing required field: {} ({})", field.key, field.label)
            })?;
            let placeholder = format!("{{{{{}}}}}", field.key);
            output = output.replace(&placeholder, &resolved);
        }
        Ok(output)
    }

    /// List the keys of all required fields that are not satisfied by the
    /// given value map.
    #[must_use]
    pub fn missing_fields(&self, values: &HashMap<String, String>) -> Vec<String> {
        self.fields
            .iter()
            .filter(|f| f.required && f.resolve(values).is_none())
            .map(|f| f.key.clone())
            .collect()
    }

    /// Number of defined fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Number of required fields.
    #[must_use]
    pub fn required_field_count(&self) -> usize {
        self.fields.iter().filter(|f| f.required).count()
    }
}

// ── TemplateRegistry ───────────────────────────────────────────────────────

/// A collection of reusable [`LicenseTemplate`]s.
#[derive(Debug, Clone, Default)]
pub struct TemplateRegistry {
    templates: HashMap<String, LicenseTemplate>,
}

impl TemplateRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a template.
    pub fn register(&mut self, template: LicenseTemplate) {
        self.templates.insert(template.id.clone(), template);
    }

    /// Look up a template by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&LicenseTemplate> {
        self.templates.get(id)
    }

    /// List all template IDs, sorted.
    #[must_use]
    pub fn list_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.templates.keys().map(String::as_str).collect();
        ids.sort_unstable();
        ids
    }

    /// Number of templates in the registry.
    #[must_use]
    pub fn count(&self) -> usize {
        self.templates.len()
    }

    /// Find templates by license kind.
    #[must_use]
    pub fn find_by_kind(&self, kind: LicenseKind) -> Vec<&LicenseTemplate> {
        self.templates.values().filter(|t| t.kind == kind).collect()
    }

    /// Create a registry pre-loaded with standard templates.
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        reg.register(Self::royalty_free_template());
        reg.register(Self::rights_managed_template());
        reg.register(Self::editorial_only_template());
        reg
    }

    /// Standard royalty-free template.
    #[must_use]
    fn royalty_free_template() -> LicenseTemplate {
        LicenseTemplate::new(
            "rf-standard",
            "Standard Royalty-Free License",
            LicenseKind::RoyaltyFree,
            "LICENSE AGREEMENT\n\
             Licensor: {{licensor}}\n\
             Licensee: {{licensee}}\n\
             Asset: {{asset_id}}\n\
             Territory: {{territory}}\n\
             This royalty-free license grants unlimited usage within the specified territory.",
        )
        .with_field(TemplateField::required("licensor", "Licensor Name"))
        .with_field(TemplateField::required("licensee", "Licensee Name"))
        .with_field(TemplateField::required("asset_id", "Asset ID"))
        .with_field(TemplateField::optional(
            "territory",
            "Territory",
            "Worldwide",
        ))
    }

    /// Standard rights-managed template.
    #[must_use]
    fn rights_managed_template() -> LicenseTemplate {
        LicenseTemplate::new(
            "rm-standard",
            "Standard Rights-Managed License",
            LicenseKind::RightsManaged,
            "RIGHTS-MANAGED LICENSE\n\
             Licensor: {{licensor}}\n\
             Licensee: {{licensee}}\n\
             Asset: {{asset_id}}\n\
             Usage: {{usage}}\n\
             Fee: {{fee}}\n\
             This rights-managed license authorises the specified usage only.",
        )
        .with_field(TemplateField::required("licensor", "Licensor Name"))
        .with_field(TemplateField::required("licensee", "Licensee Name"))
        .with_field(TemplateField::required("asset_id", "Asset ID"))
        .with_field(TemplateField::required("usage", "Usage Description"))
        .with_field(TemplateField::required("fee", "License Fee"))
    }

    /// Standard editorial-only template.
    #[must_use]
    fn editorial_only_template() -> LicenseTemplate {
        LicenseTemplate::new(
            "editorial-standard",
            "Editorial Use Only License",
            LicenseKind::EditorialOnly,
            "EDITORIAL LICENSE\n\
             Licensor: {{licensor}}\n\
             Licensee: {{licensee}}\n\
             Asset: {{asset_id}}\n\
             This license permits editorial use only. Commercial use is prohibited.",
        )
        .with_field(TemplateField::required("licensor", "Licensor Name"))
        .with_field(TemplateField::required("licensee", "Licensee Name"))
        .with_field(TemplateField::required("asset_id", "Asset ID"))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_values() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("licensor".into(), "Acme Corp".into());
        m.insert("licensee".into(), "Widget Inc".into());
        m.insert("asset_id".into(), "VID-12345".into());
        m
    }

    // ── LicenseKind ──

    #[test]
    fn test_rights_managed_per_use() {
        assert!(LicenseKind::RightsManaged.requires_per_use_fee());
        assert!(!LicenseKind::RoyaltyFree.requires_per_use_fee());
    }

    #[test]
    fn test_editorial_has_restriction() {
        assert!(LicenseKind::EditorialOnly.has_usage_restriction());
        assert!(!LicenseKind::Exclusive.has_usage_restriction());
    }

    // ── TemplateField ──

    #[test]
    fn test_required_field_no_default() {
        let f = TemplateField::required("name", "Name");
        assert!(f.required);
        assert!(f.default.is_none());
    }

    #[test]
    fn test_optional_field_default() {
        let f = TemplateField::optional("territory", "Territory", "Worldwide");
        assert!(!f.required);
        assert_eq!(f.default.as_deref(), Some("Worldwide"));
    }

    #[test]
    fn test_field_resolve_from_values() {
        let f = TemplateField::required("licensor", "Licensor");
        let vals = sample_values();
        assert_eq!(f.resolve(&vals), Some("Acme Corp".into()));
    }

    #[test]
    fn test_field_resolve_fallback_default() {
        let f = TemplateField::optional("territory", "Territory", "Worldwide");
        let vals: HashMap<String, String> = HashMap::new();
        assert_eq!(f.resolve(&vals), Some("Worldwide".into()));
    }

    #[test]
    fn test_field_resolve_missing_required() {
        let f = TemplateField::required("missing_key", "Missing");
        let vals: HashMap<String, String> = HashMap::new();
        assert!(f.resolve(&vals).is_none());
    }

    // ── LicenseTemplate ──

    #[test]
    fn test_template_instantiate_success() {
        let reg = TemplateRegistry::with_defaults();
        let tpl = reg
            .get("rf-standard")
            .expect("rights test operation should succeed");
        let vals = sample_values();
        let result = tpl.instantiate(&vals);
        assert!(result.is_ok());
        let text = result.expect("rights test operation should succeed");
        assert!(text.contains("Acme Corp"));
        assert!(text.contains("Widget Inc"));
        assert!(text.contains("VID-12345"));
        // Territory should fall back to default "Worldwide"
        assert!(text.contains("Worldwide"));
    }

    #[test]
    fn test_template_instantiate_missing_required() {
        let reg = TemplateRegistry::with_defaults();
        let tpl = reg
            .get("rm-standard")
            .expect("rights test operation should succeed");
        // Missing "usage" and "fee"
        let vals = sample_values();
        let result = tpl.instantiate(&vals);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("usage"));
    }

    #[test]
    fn test_template_missing_fields() {
        let reg = TemplateRegistry::with_defaults();
        let tpl = reg
            .get("rm-standard")
            .expect("rights test operation should succeed");
        let vals = sample_values();
        let missing = tpl.missing_fields(&vals);
        assert!(missing.contains(&"usage".to_string()));
        assert!(missing.contains(&"fee".to_string()));
    }

    #[test]
    fn test_template_field_count() {
        let reg = TemplateRegistry::with_defaults();
        let tpl = reg
            .get("rf-standard")
            .expect("rights test operation should succeed");
        assert_eq!(tpl.field_count(), 4);
        assert_eq!(tpl.required_field_count(), 3);
    }

    // ── TemplateRegistry ──

    #[test]
    fn test_registry_defaults() {
        let reg = TemplateRegistry::with_defaults();
        assert_eq!(reg.count(), 3);
    }

    #[test]
    fn test_registry_list_ids_sorted() {
        let reg = TemplateRegistry::with_defaults();
        let ids = reg.list_ids();
        for pair in ids.windows(2) {
            assert!(pair[0] <= pair[1]);
        }
    }

    #[test]
    fn test_registry_find_by_kind() {
        let reg = TemplateRegistry::with_defaults();
        let rf = reg.find_by_kind(LicenseKind::RoyaltyFree);
        assert_eq!(rf.len(), 1);
        assert_eq!(rf[0].id, "rf-standard");
    }

    #[test]
    fn test_registry_empty() {
        let reg = TemplateRegistry::new();
        assert_eq!(reg.count(), 0);
        assert!(reg.get("nonexistent").is_none());
    }
}
