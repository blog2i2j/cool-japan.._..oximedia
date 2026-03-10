#![allow(dead_code)]
//! Job templates for reusable, parameterised job definitions.
//!
//! Templates allow defining common job shapes once and instantiating them
//! with different parameters at runtime, reducing boilerplate and ensuring
//! consistency across similar jobs.

use std::collections::HashMap;
use std::fmt;

/// Priority level for template-instantiated jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemplatePriority {
    /// Critical priority — executed first.
    Critical,
    /// High priority.
    High,
    /// Normal / default priority.
    Normal,
    /// Low priority — background work.
    Low,
}

impl fmt::Display for TemplatePriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemplatePriority::Critical => write!(f, "critical"),
            TemplatePriority::High => write!(f, "high"),
            TemplatePriority::Normal => write!(f, "normal"),
            TemplatePriority::Low => write!(f, "low"),
        }
    }
}

/// A parameter slot in a job template.
#[derive(Debug, Clone, PartialEq)]
pub struct TemplateParam {
    /// Parameter name (used as a placeholder key).
    pub name: String,
    /// Human-readable description of the parameter.
    pub description: String,
    /// Whether this parameter must be provided at instantiation.
    pub required: bool,
    /// Default value if not provided and not required.
    pub default_value: Option<String>,
    /// Optional validation regex pattern.
    pub validation_pattern: Option<String>,
}

impl TemplateParam {
    /// Create a new required parameter.
    pub fn required(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: true,
            default_value: None,
            validation_pattern: None,
        }
    }

    /// Create a new optional parameter with a default.
    pub fn optional(
        name: impl Into<String>,
        description: impl Into<String>,
        default: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: false,
            default_value: Some(default.into()),
            validation_pattern: None,
        }
    }

    /// Add a validation pattern.
    pub fn with_validation(mut self, pattern: impl Into<String>) -> Self {
        self.validation_pattern = Some(pattern.into());
        self
    }

    /// Resolve the effective value for this parameter given user-supplied values.
    pub fn resolve(&self, supplied: Option<&String>) -> Result<String, TemplateError> {
        match (supplied, &self.default_value) {
            (Some(val), _) => Ok(val.clone()),
            (None, Some(def)) => Ok(def.clone()),
            (None, None) if self.required => {
                Err(TemplateError::MissingParameter(self.name.clone()))
            }
            (None, None) => Ok(String::new()),
        }
    }
}

/// Errors that can occur when working with job templates.
#[derive(Debug, Clone, PartialEq)]
pub enum TemplateError {
    /// A required parameter was not supplied.
    MissingParameter(String),
    /// A parameter value failed validation.
    ValidationFailed {
        /// The parameter that failed.
        param: String,
        /// The value that was provided.
        value: String,
        /// The pattern it was validated against.
        pattern: String,
    },
    /// The template was not found.
    NotFound(String),
    /// Duplicate template name.
    Duplicate(String),
    /// Template body contains an undefined placeholder.
    UndefinedPlaceholder(String),
}

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemplateError::MissingParameter(p) => write!(f, "Missing required parameter: {p}"),
            TemplateError::ValidationFailed {
                param,
                value,
                pattern,
            } => {
                write!(
                    f,
                    "Validation failed for {param}={value} (pattern: {pattern})"
                )
            }
            TemplateError::NotFound(name) => write!(f, "Template not found: {name}"),
            TemplateError::Duplicate(name) => write!(f, "Duplicate template: {name}"),
            TemplateError::UndefinedPlaceholder(ph) => {
                write!(f, "Undefined placeholder in template body: {ph}")
            }
        }
    }
}

/// A reusable job template.
#[derive(Debug, Clone)]
pub struct JobTemplate {
    /// Unique template identifier.
    pub name: String,
    /// Human-readable description of the template.
    pub description: String,
    /// Version string for the template.
    pub version: String,
    /// Default priority for jobs created from this template.
    pub default_priority: TemplatePriority,
    /// Template parameters.
    pub params: Vec<TemplateParam>,
    /// Body text with `{{param_name}}` placeholders.
    pub body: String,
    /// Tags to apply to instantiated jobs.
    pub tags: Vec<String>,
    /// Maximum retries for instantiated jobs.
    pub max_retries: u32,
    /// Timeout in seconds for instantiated jobs.
    pub timeout_secs: u64,
}

impl JobTemplate {
    /// Create a new job template.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        body: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            version: "1.0.0".to_string(),
            default_priority: TemplatePriority::Normal,
            params: Vec::new(),
            body: body.into(),
            tags: Vec::new(),
            max_retries: 3,
            timeout_secs: 3600,
        }
    }

    /// Add a parameter to the template.
    pub fn with_param(mut self, param: TemplateParam) -> Self {
        self.params.push(param);
        self
    }

    /// Set the default priority.
    pub fn with_priority(mut self, priority: TemplatePriority) -> Self {
        self.default_priority = priority;
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set max retries.
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set the timeout.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set the version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Instantiate the template with the given parameter values, producing a resolved job spec.
    pub fn instantiate(
        &self,
        values: &HashMap<String, String>,
    ) -> Result<JobInstance, TemplateError> {
        // Resolve all parameters
        let mut resolved = HashMap::new();
        for param in &self.params {
            let value = param.resolve(values.get(&param.name))?;
            resolved.insert(param.name.clone(), value);
        }
        // Substitute placeholders in body
        let mut body = self.body.clone();
        for (key, val) in &resolved {
            let placeholder = format!("{{{{{key}}}}}");
            body = body.replace(&placeholder, val);
        }
        Ok(JobInstance {
            template_name: self.name.clone(),
            template_version: self.version.clone(),
            priority: self.default_priority,
            resolved_body: body,
            resolved_params: resolved,
            tags: self.tags.clone(),
            max_retries: self.max_retries,
            timeout_secs: self.timeout_secs,
        })
    }

    /// List the names of all required parameters.
    pub fn required_params(&self) -> Vec<&str> {
        self.params
            .iter()
            .filter(|p| p.required)
            .map(|p| p.name.as_str())
            .collect()
    }

    /// List the names of all optional parameters.
    pub fn optional_params(&self) -> Vec<&str> {
        self.params
            .iter()
            .filter(|p| !p.required)
            .map(|p| p.name.as_str())
            .collect()
    }
}

/// A concrete job instance produced from a template.
#[derive(Debug, Clone)]
pub struct JobInstance {
    /// Name of the source template.
    pub template_name: String,
    /// Version of the source template.
    pub template_version: String,
    /// Priority of this job.
    pub priority: TemplatePriority,
    /// Resolved body with all placeholders substituted.
    pub resolved_body: String,
    /// Map of resolved parameter values.
    pub resolved_params: HashMap<String, String>,
    /// Tags inherited from the template.
    pub tags: Vec<String>,
    /// Max retries.
    pub max_retries: u32,
    /// Timeout in seconds.
    pub timeout_secs: u64,
}

/// Registry for managing multiple job templates.
#[derive(Debug, Default)]
pub struct TemplateRegistry {
    /// Storage for templates keyed by name.
    templates: HashMap<String, JobTemplate>,
}

impl TemplateRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a template. Returns an error if a template with the same name already exists.
    pub fn register(&mut self, template: JobTemplate) -> Result<(), TemplateError> {
        if self.templates.contains_key(&template.name) {
            return Err(TemplateError::Duplicate(template.name.clone()));
        }
        self.templates.insert(template.name.clone(), template);
        Ok(())
    }

    /// Remove a template by name.
    pub fn unregister(&mut self, name: &str) -> Result<JobTemplate, TemplateError> {
        self.templates
            .remove(name)
            .ok_or_else(|| TemplateError::NotFound(name.to_string()))
    }

    /// Look up a template by name.
    pub fn get(&self, name: &str) -> Result<&JobTemplate, TemplateError> {
        self.templates
            .get(name)
            .ok_or_else(|| TemplateError::NotFound(name.to_string()))
    }

    /// Instantiate a template by name with the given values.
    pub fn instantiate(
        &self,
        name: &str,
        values: &HashMap<String, String>,
    ) -> Result<JobInstance, TemplateError> {
        let template = self.get(name)?;
        template.instantiate(values)
    }

    /// List all registered template names.
    pub fn list(&self) -> Vec<&str> {
        self.templates.keys().map(|k| k.as_str()).collect()
    }

    /// Return the number of registered templates.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Check whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_template() -> JobTemplate {
        JobTemplate::new(
            "transcode-hd",
            "Transcode video to HD",
            "transcode {{input}} -> {{output}} at {{bitrate}}",
        )
        .with_param(TemplateParam::required("input", "Input file path"))
        .with_param(TemplateParam::required("output", "Output file path"))
        .with_param(TemplateParam::optional(
            "bitrate",
            "Target bitrate",
            "5000000",
        ))
        .with_tag("video")
        .with_tag("transcode")
        .with_priority(TemplatePriority::High)
    }

    #[test]
    fn test_template_creation() {
        let t = sample_template();
        assert_eq!(t.name, "transcode-hd");
        assert_eq!(t.params.len(), 3);
        assert_eq!(t.default_priority, TemplatePriority::High);
    }

    #[test]
    fn test_instantiate_success() {
        let t = sample_template();
        let mut vals = HashMap::new();
        vals.insert("input".to_string(), "video.mp4".to_string());
        vals.insert("output".to_string(), "video_hd.mp4".to_string());
        let instance = t.instantiate(&vals).expect("instance should be valid");
        assert_eq!(
            instance.resolved_body,
            "transcode video.mp4 -> video_hd.mp4 at 5000000"
        );
        assert_eq!(instance.tags.len(), 2);
    }

    #[test]
    fn test_instantiate_with_override() {
        let t = sample_template();
        let mut vals = HashMap::new();
        vals.insert("input".to_string(), "a.mp4".to_string());
        vals.insert("output".to_string(), "b.mp4".to_string());
        vals.insert("bitrate".to_string(), "8000000".to_string());
        let instance = t.instantiate(&vals).expect("instance should be valid");
        assert!(instance.resolved_body.contains("8000000"));
    }

    #[test]
    fn test_instantiate_missing_required() {
        let t = sample_template();
        let vals = HashMap::new();
        let result = t.instantiate(&vals);
        assert!(result.is_err());
        if let Err(TemplateError::MissingParameter(p)) = result {
            assert_eq!(p, "input");
        }
    }

    #[test]
    fn test_required_params_list() {
        let t = sample_template();
        let req = t.required_params();
        assert_eq!(req.len(), 2);
        assert!(req.contains(&"input"));
        assert!(req.contains(&"output"));
    }

    #[test]
    fn test_optional_params_list() {
        let t = sample_template();
        let opt = t.optional_params();
        assert_eq!(opt.len(), 1);
        assert!(opt.contains(&"bitrate"));
    }

    #[test]
    fn test_template_priority_display() {
        assert_eq!(TemplatePriority::Critical.to_string(), "critical");
        assert_eq!(TemplatePriority::High.to_string(), "high");
        assert_eq!(TemplatePriority::Normal.to_string(), "normal");
        assert_eq!(TemplatePriority::Low.to_string(), "low");
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = TemplateRegistry::new();
        reg.register(sample_template())
            .expect("test expectation failed");
        assert_eq!(reg.len(), 1);
        assert!(reg.get("transcode-hd").is_ok());
    }

    #[test]
    fn test_registry_duplicate() {
        let mut reg = TemplateRegistry::new();
        reg.register(sample_template())
            .expect("test expectation failed");
        let result = reg.register(sample_template());
        assert!(matches!(result, Err(TemplateError::Duplicate(_))));
    }

    #[test]
    fn test_registry_unregister() {
        let mut reg = TemplateRegistry::new();
        reg.register(sample_template())
            .expect("test expectation failed");
        let removed = reg
            .unregister("transcode-hd")
            .expect("removed should be valid");
        assert_eq!(removed.name, "transcode-hd");
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_not_found() {
        let reg = TemplateRegistry::new();
        assert!(matches!(
            reg.get("nonexistent"),
            Err(TemplateError::NotFound(_))
        ));
    }

    #[test]
    fn test_registry_instantiate() {
        let mut reg = TemplateRegistry::new();
        reg.register(sample_template())
            .expect("test expectation failed");
        let mut vals = HashMap::new();
        vals.insert("input".to_string(), "x.mp4".to_string());
        vals.insert("output".to_string(), "y.mp4".to_string());
        let instance = reg
            .instantiate("transcode-hd", &vals)
            .expect("instance should be valid");
        assert!(instance.resolved_body.contains("x.mp4"));
    }

    #[test]
    fn test_template_with_version_and_timeout() {
        let t = JobTemplate::new("test", "desc", "body")
            .with_version("2.0.0")
            .with_timeout(7200)
            .with_max_retries(5);
        assert_eq!(t.version, "2.0.0");
        assert_eq!(t.timeout_secs, 7200);
        assert_eq!(t.max_retries, 5);
    }

    #[test]
    fn test_template_error_display() {
        let e = TemplateError::MissingParameter("input".to_string());
        assert_eq!(e.to_string(), "Missing required parameter: input");
        let e2 = TemplateError::Duplicate("dup".to_string());
        assert_eq!(e2.to_string(), "Duplicate template: dup");
    }
}
