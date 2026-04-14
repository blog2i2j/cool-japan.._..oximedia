#![allow(dead_code)]
//! Reusable job templates for common encoding farm tasks.
//!
//! Provides a template system for defining recurring job configurations:
//! - Predefined templates for common encoding profiles (H.264, HEVC, AV1)
//! - Template parameterization with variable substitution
//! - Template validation before job instantiation
//! - Template registry for managing named templates
//! - Template versioning support

use std::collections::HashMap;

/// A parameter definition for a template variable.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TemplateParam {
    /// Name of the parameter.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Default value if not supplied.
    pub default_value: Option<String>,
    /// Whether this parameter is required.
    pub required: bool,
    /// Allowed values (empty means any value is accepted).
    pub allowed_values: Vec<String>,
}

impl TemplateParam {
    /// Create a required parameter with no default.
    pub fn required(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            default_value: None,
            required: true,
            allowed_values: Vec::new(),
        }
    }

    /// Create an optional parameter with a default value.
    pub fn optional(
        name: impl Into<String>,
        description: impl Into<String>,
        default: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            default_value: Some(default.into()),
            required: false,
            allowed_values: Vec::new(),
        }
    }

    /// Add allowed values to constrain the parameter.
    #[must_use]
    pub fn with_allowed_values(mut self, values: Vec<String>) -> Self {
        self.allowed_values = values;
        self
    }
}

/// Error type for template operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateError {
    /// A required parameter is missing.
    MissingParameter(String),
    /// A parameter value is not in the allowed set.
    InvalidParameterValue {
        /// Parameter name.
        name: String,
        /// The invalid value that was supplied.
        value: String,
    },
    /// Template not found in registry.
    TemplateNotFound(String),
    /// Template already exists in registry.
    TemplateAlreadyExists(String),
    /// The template body contains an unknown variable reference.
    UnknownVariable(String),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingParameter(name) => write!(f, "missing required parameter: {name}"),
            Self::InvalidParameterValue { name, value } => {
                write!(f, "invalid value '{value}' for parameter '{name}'")
            }
            Self::TemplateNotFound(name) => write!(f, "template not found: {name}"),
            Self::TemplateAlreadyExists(name) => write!(f, "template already exists: {name}"),
            Self::UnknownVariable(var) => write!(f, "unknown variable in template body: {var}"),
        }
    }
}

impl std::error::Error for TemplateError {}

/// Result type for template operations.
pub type Result<T> = std::result::Result<T, TemplateError>;

/// A job template that can be instantiated with parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JobTemplate {
    /// Unique template name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Template version string.
    pub version: String,
    /// Parameter definitions.
    pub params: Vec<TemplateParam>,
    /// Codec identifier (e.g., "h264", "hevc", "av1").
    pub codec: String,
    /// Container format (e.g., "mp4", "mkv").
    pub container: String,
    /// Bitrate string template (may contain `{{variable}}` references).
    pub bitrate: String,
    /// Resolution string template.
    pub resolution: String,
    /// Additional key-value settings with possible variable references.
    pub settings: HashMap<String, String>,
}

impl JobTemplate {
    /// Create a new job template builder.
    pub fn builder(name: impl Into<String>) -> JobTemplateBuilder {
        JobTemplateBuilder {
            name: name.into(),
            description: String::new(),
            version: "1.0".to_string(),
            params: Vec::new(),
            codec: String::new(),
            container: String::new(),
            bitrate: String::new(),
            resolution: String::new(),
            settings: HashMap::new(),
        }
    }

    /// Validate that the provided parameters satisfy this template's requirements.
    ///
    /// # Errors
    ///
    /// Returns `TemplateError::MissingParameter` if a required parameter is absent,
    /// or `TemplateError::InvalidParameterValue` if a value is not in the allowed set.
    pub fn validate_params(&self, params: &HashMap<String, String>) -> Result<()> {
        for p in &self.params {
            match params.get(&p.name) {
                Some(val) => {
                    if !p.allowed_values.is_empty() && !p.allowed_values.contains(val) {
                        return Err(TemplateError::InvalidParameterValue {
                            name: p.name.clone(),
                            value: val.clone(),
                        });
                    }
                }
                None => {
                    if p.required && p.default_value.is_none() {
                        return Err(TemplateError::MissingParameter(p.name.clone()));
                    }
                }
            }
        }
        Ok(())
    }

    /// Resolve a template string by substituting `{{variable}}` references.
    ///
    /// Uses the provided `params` map and falls back to default values.
    fn resolve_string(&self, template: &str, params: &HashMap<String, String>) -> String {
        let mut result = template.to_string();
        for p in &self.params {
            let placeholder = format!("{{{{{}}}}}", p.name);
            if let Some(val) = params.get(&p.name) {
                result = result.replace(&placeholder, val);
            } else if let Some(ref default) = p.default_value {
                result = result.replace(&placeholder, default);
            }
        }
        result
    }

    /// Instantiate the template with the given parameters, producing a resolved `ResolvedJob`.
    ///
    /// # Errors
    ///
    /// Returns a `TemplateError` if validation fails.
    pub fn instantiate(&self, params: &HashMap<String, String>) -> Result<ResolvedJob> {
        self.validate_params(params)?;

        let resolved_settings: HashMap<String, String> = self
            .settings
            .iter()
            .map(|(k, v)| (k.clone(), self.resolve_string(v, params)))
            .collect();

        Ok(ResolvedJob {
            template_name: self.name.clone(),
            template_version: self.version.clone(),
            codec: self.resolve_string(&self.codec, params),
            container: self.resolve_string(&self.container, params),
            bitrate: self.resolve_string(&self.bitrate, params),
            resolution: self.resolve_string(&self.resolution, params),
            settings: resolved_settings,
        })
    }
}

/// A fully resolved job configuration produced from a template instantiation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResolvedJob {
    /// Name of the template used.
    pub template_name: String,
    /// Version of the template used.
    pub template_version: String,
    /// Resolved codec.
    pub codec: String,
    /// Resolved container.
    pub container: String,
    /// Resolved bitrate string.
    pub bitrate: String,
    /// Resolved resolution string.
    pub resolution: String,
    /// Resolved settings.
    pub settings: HashMap<String, String>,
}

/// Builder for creating `JobTemplate` instances.
#[derive(Debug)]
pub struct JobTemplateBuilder {
    /// Template name.
    name: String,
    /// Template description.
    description: String,
    /// Template version.
    version: String,
    /// Parameter definitions.
    params: Vec<TemplateParam>,
    /// Codec string.
    codec: String,
    /// Container string.
    container: String,
    /// Bitrate string.
    bitrate: String,
    /// Resolution string.
    resolution: String,
    /// Settings map.
    settings: HashMap<String, String>,
}

impl JobTemplateBuilder {
    /// Set the template description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set the template version.
    pub fn version(mut self, ver: impl Into<String>) -> Self {
        self.version = ver.into();
        self
    }

    /// Add a parameter definition.
    #[must_use]
    pub fn param(mut self, param: TemplateParam) -> Self {
        self.params.push(param);
        self
    }

    /// Set the codec.
    pub fn codec(mut self, codec: impl Into<String>) -> Self {
        self.codec = codec.into();
        self
    }

    /// Set the container format.
    pub fn container(mut self, container: impl Into<String>) -> Self {
        self.container = container.into();
        self
    }

    /// Set the bitrate template string.
    pub fn bitrate(mut self, bitrate: impl Into<String>) -> Self {
        self.bitrate = bitrate.into();
        self
    }

    /// Set the resolution template string.
    pub fn resolution(mut self, resolution: impl Into<String>) -> Self {
        self.resolution = resolution.into();
        self
    }

    /// Add a setting key-value pair.
    pub fn setting(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.settings.insert(key.into(), value.into());
        self
    }

    /// Build the `JobTemplate`.
    #[must_use]
    pub fn build(self) -> JobTemplate {
        JobTemplate {
            name: self.name,
            description: self.description,
            version: self.version,
            params: self.params,
            codec: self.codec,
            container: self.container,
            bitrate: self.bitrate,
            resolution: self.resolution,
            settings: self.settings,
        }
    }
}

/// A registry of named job templates.
#[derive(Debug, Default)]
pub struct TemplateRegistry {
    /// Map of template name to template.
    templates: HashMap<String, JobTemplate>,
}

impl TemplateRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new template.
    ///
    /// # Errors
    ///
    /// Returns `TemplateError::TemplateAlreadyExists` if a template with the same name exists.
    pub fn register(&mut self, template: JobTemplate) -> Result<()> {
        if self.templates.contains_key(&template.name) {
            return Err(TemplateError::TemplateAlreadyExists(template.name.clone()));
        }
        self.templates.insert(template.name.clone(), template);
        Ok(())
    }

    /// Get a template by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&JobTemplate> {
        self.templates.get(name)
    }

    /// Remove a template by name.
    pub fn remove(&mut self, name: &str) -> Option<JobTemplate> {
        self.templates.remove(name)
    }

    /// List all registered template names.
    pub fn list_names(&self) -> Vec<&str> {
        self.templates.keys().map(String::as_str).collect()
    }

    /// Get the number of registered templates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Check if the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h264_template() -> JobTemplate {
        JobTemplate::builder("h264-web")
            .description("H.264 web streaming template")
            .codec("h264")
            .container("mp4")
            .bitrate("{{bitrate}}")
            .resolution("{{resolution}}")
            .param(TemplateParam::required("bitrate", "Target bitrate"))
            .param(TemplateParam::optional(
                "resolution",
                "Output resolution",
                "1920x1080",
            ))
            .setting("profile", "high")
            .setting("preset", "{{preset}}")
            .param(TemplateParam::optional(
                "preset",
                "Encoding preset",
                "medium",
            ))
            .build()
    }

    #[test]
    fn test_template_creation() {
        let t = h264_template();
        assert_eq!(t.name, "h264-web");
        assert_eq!(t.codec, "h264");
        assert_eq!(t.container, "mp4");
        assert_eq!(t.params.len(), 3);
    }

    #[test]
    fn test_validate_params_ok() {
        let t = h264_template();
        let mut params = HashMap::new();
        params.insert("bitrate".to_string(), "5000k".to_string());
        assert!(t.validate_params(&params).is_ok());
    }

    #[test]
    fn test_validate_params_missing_required() {
        let t = h264_template();
        let params = HashMap::new();
        let err = t.validate_params(&params).unwrap_err();
        assert_eq!(err, TemplateError::MissingParameter("bitrate".to_string()));
    }

    #[test]
    fn test_validate_params_invalid_value() {
        let t = JobTemplate::builder("test")
            .param(
                TemplateParam::required("codec", "Codec choice")
                    .with_allowed_values(vec!["h264".to_string(), "hevc".to_string()]),
            )
            .build();
        let mut params = HashMap::new();
        params.insert("codec".to_string(), "vp9".to_string());
        let err = t.validate_params(&params).unwrap_err();
        assert_eq!(
            err,
            TemplateError::InvalidParameterValue {
                name: "codec".to_string(),
                value: "vp9".to_string(),
            }
        );
    }

    #[test]
    fn test_instantiate_template() {
        let t = h264_template();
        let mut params = HashMap::new();
        params.insert("bitrate".to_string(), "8000k".to_string());
        params.insert("resolution".to_string(), "3840x2160".to_string());
        let resolved = t.instantiate(&params).unwrap();
        assert_eq!(resolved.codec, "h264");
        assert_eq!(resolved.bitrate, "8000k");
        assert_eq!(resolved.resolution, "3840x2160");
    }

    #[test]
    fn test_instantiate_defaults() {
        let t = h264_template();
        let mut params = HashMap::new();
        params.insert("bitrate".to_string(), "5000k".to_string());
        let resolved = t.instantiate(&params).unwrap();
        assert_eq!(resolved.resolution, "1920x1080"); // default
        assert_eq!(resolved.settings.get("preset").unwrap(), "medium"); // default
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = TemplateRegistry::new();
        reg.register(h264_template()).unwrap();
        assert_eq!(reg.len(), 1);
        assert!(reg.get("h264-web").is_some());
    }

    #[test]
    fn test_registry_duplicate() {
        let mut reg = TemplateRegistry::new();
        reg.register(h264_template()).unwrap();
        let err = reg.register(h264_template()).unwrap_err();
        assert_eq!(
            err,
            TemplateError::TemplateAlreadyExists("h264-web".to_string())
        );
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = TemplateRegistry::new();
        reg.register(h264_template()).unwrap();
        let removed = reg.remove("h264-web");
        assert!(removed.is_some());
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_list_names() {
        let mut reg = TemplateRegistry::new();
        reg.register(h264_template()).unwrap();
        let names = reg.list_names();
        assert_eq!(names.len(), 1);
        assert!(names.contains(&"h264-web"));
    }

    #[test]
    fn test_template_error_display() {
        let err = TemplateError::MissingParameter("foo".to_string());
        assert_eq!(err.to_string(), "missing required parameter: foo");
    }

    #[test]
    fn test_resolved_job_template_info() {
        let t = h264_template();
        let mut params = HashMap::new();
        params.insert("bitrate".to_string(), "2000k".to_string());
        let resolved = t.instantiate(&params).unwrap();
        assert_eq!(resolved.template_name, "h264-web");
        assert_eq!(resolved.template_version, "1.0");
    }
}
