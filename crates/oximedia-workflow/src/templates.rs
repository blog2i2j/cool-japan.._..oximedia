//! Workflow template library.
//!
//! Provides pre-built workflow templates for common media processing pipelines
//! such as ingest-and-transcode, archive, and QC review workflows.

#![allow(dead_code)]

use std::collections::HashMap;

/// Category of workflow template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateCategory {
    /// Ingest workflows (file watch, validation, ingestion).
    Ingest,
    /// Transcode workflows.
    Transcode,
    /// Delivery workflows (packaging, distribution).
    Delivery,
    /// Archive workflows.
    Archive,
    /// Quality control workflows.
    QC,
    /// Distribution workflows (CDN upload, broadcast).
    Distribution,
}

/// Parameter type for template parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamType {
    /// String value.
    String,
    /// Integer value.
    Integer,
    /// Floating-point value.
    Float,
    /// Boolean value.
    Bool,
    /// File path value.
    FilePath,
    /// Enumeration (one of the provided options).
    Enum(Vec<std::string::String>),
}

/// Template parameter definition.
#[derive(Debug, Clone)]
pub struct TemplateParam {
    /// Parameter name.
    pub name: std::string::String,
    /// Parameter type.
    pub param_type: ParamType,
    /// Default value (as string representation).
    pub default_value: Option<std::string::String>,
    /// Whether this parameter is required.
    pub required: bool,
}

impl TemplateParam {
    /// Create a required parameter.
    #[must_use]
    pub fn required(name: impl Into<std::string::String>, param_type: ParamType) -> Self {
        Self {
            name: name.into(),
            param_type,
            default_value: None,
            required: true,
        }
    }

    /// Create an optional parameter with a default.
    #[must_use]
    pub fn optional(
        name: impl Into<std::string::String>,
        param_type: ParamType,
        default_value: impl Into<std::string::String>,
    ) -> Self {
        Self {
            name: name.into(),
            param_type,
            default_value: Some(default_value.into()),
            required: false,
        }
    }
}

/// Action to take when a step fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorAction {
    /// Fail the entire workflow.
    Fail,
    /// Retry the step up to N times.
    Retry(u32),
    /// Skip this step and continue.
    Skip,
    /// Send a notification and continue.
    Notify,
}

/// A single step in a workflow template.
#[derive(Debug, Clone)]
pub struct TemplateStep {
    /// Step type identifier (e.g. "validate", "transcode", "deliver").
    pub step_type: std::string::String,
    /// Parameters for this step.
    pub params: HashMap<std::string::String, std::string::String>,
    /// Action to take on error.
    pub on_error: ErrorAction,
}

impl TemplateStep {
    /// Create a new template step.
    #[must_use]
    pub fn new(step_type: impl Into<std::string::String>, on_error: ErrorAction) -> Self {
        Self {
            step_type: step_type.into(),
            params: HashMap::new(),
            on_error,
        }
    }

    /// Add a parameter to the step.
    #[must_use]
    pub fn with_param(
        mut self,
        key: impl Into<std::string::String>,
        value: impl Into<std::string::String>,
    ) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }
}

/// A workflow template.
#[derive(Debug, Clone)]
pub struct WorkflowTemplate {
    /// Template name.
    pub name: std::string::String,
    /// Template description.
    pub description: std::string::String,
    /// Template category.
    pub category: TemplateCategory,
    /// Template parameters.
    pub parameters: Vec<TemplateParam>,
    /// Template steps.
    pub steps: Vec<TemplateStep>,
}

impl WorkflowTemplate {
    /// Create a new workflow template.
    #[must_use]
    pub fn new(
        name: impl Into<std::string::String>,
        description: impl Into<std::string::String>,
        category: TemplateCategory,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            category,
            parameters: Vec::new(),
            steps: Vec::new(),
        }
    }

    /// Add a parameter definition.
    #[must_use]
    pub fn with_parameter(mut self, param: TemplateParam) -> Self {
        self.parameters.push(param);
        self
    }

    /// Add a step.
    #[must_use]
    pub fn with_step(mut self, step: TemplateStep) -> Self {
        self.steps.push(step);
        self
    }
}

/// Library of pre-built workflow templates.
pub struct TemplateLibrary;

impl TemplateLibrary {
    /// Ingest and transcode template.
    ///
    /// Steps: watch folder → validate → transcode → deliver.
    #[must_use]
    pub fn ingest_and_transcode() -> WorkflowTemplate {
        WorkflowTemplate::new(
            "ingest-and-transcode",
            "Watch folder for incoming media, validate, transcode, and deliver",
            TemplateCategory::Ingest,
        )
        .with_parameter(TemplateParam::required("watch_folder", ParamType::FilePath))
        .with_parameter(TemplateParam::required(
            "output_folder",
            ParamType::FilePath,
        ))
        .with_parameter(TemplateParam::optional(
            "transcode_preset",
            ParamType::Enum(vec![
                "broadcast_hd".to_string(),
                "web_h264".to_string(),
                "archive_prores".to_string(),
            ]),
            "web_h264",
        ))
        .with_parameter(TemplateParam::optional(
            "validate_on_ingest",
            ParamType::Bool,
            "true",
        ))
        .with_step(
            TemplateStep::new("watch_folder", ErrorAction::Fail)
                .with_param("path", "{{watch_folder}}")
                .with_param("pattern", "*.{mxf,mp4,mov}"),
        )
        .with_step(
            TemplateStep::new("validate", ErrorAction::Retry(2))
                .with_param("enabled", "{{validate_on_ingest}}")
                .with_param("checks", "format,integrity,container"),
        )
        .with_step(
            TemplateStep::new("transcode", ErrorAction::Retry(1))
                .with_param("preset", "{{transcode_preset}}")
                .with_param("output", "{{output_folder}}"),
        )
        .with_step(
            TemplateStep::new("deliver", ErrorAction::Notify)
                .with_param("destination", "{{output_folder}}")
                .with_param("verify_checksum", "true"),
        )
    }

    /// Archive workflow template.
    ///
    /// Steps: validate → checksum → archive → notify.
    #[must_use]
    pub fn archive_workflow() -> WorkflowTemplate {
        WorkflowTemplate::new(
            "archive-workflow",
            "Validate media, compute checksums, archive to long-term storage, and notify",
            TemplateCategory::Archive,
        )
        .with_parameter(TemplateParam::required("source_path", ParamType::FilePath))
        .with_parameter(TemplateParam::required(
            "archive_destination",
            ParamType::FilePath,
        ))
        .with_parameter(TemplateParam::optional(
            "checksum_algorithm",
            ParamType::Enum(vec![
                "md5".to_string(),
                "sha256".to_string(),
                "xxhash".to_string(),
            ]),
            "sha256",
        ))
        .with_parameter(TemplateParam::optional(
            "notify_email",
            ParamType::String,
            "",
        ))
        .with_step(
            TemplateStep::new("validate", ErrorAction::Fail)
                .with_param("source", "{{source_path}}")
                .with_param("strict", "true"),
        )
        .with_step(
            TemplateStep::new("checksum", ErrorAction::Fail)
                .with_param("algorithm", "{{checksum_algorithm}}")
                .with_param("output_manifest", "true"),
        )
        .with_step(
            TemplateStep::new("archive", ErrorAction::Retry(3))
                .with_param("destination", "{{archive_destination}}")
                .with_param("verify_after_copy", "true"),
        )
        .with_step(
            TemplateStep::new("notify", ErrorAction::Skip)
                .with_param("channel", "email")
                .with_param("to", "{{notify_email}}")
                .with_param("message", "Archive complete"),
        )
    }

    /// QC review workflow template.
    ///
    /// Steps: analyze → report → approve/reject.
    #[must_use]
    pub fn qc_review() -> WorkflowTemplate {
        let tmp = std::env::temp_dir();
        WorkflowTemplate::new(
            "qc-review",
            "Automated QC analysis, report generation, and approve/reject decision",
            TemplateCategory::QC,
        )
        .with_parameter(TemplateParam::required("media_path", ParamType::FilePath))
        .with_parameter(TemplateParam::optional(
            "qc_profile",
            ParamType::Enum(vec![
                "broadcast".to_string(),
                "streaming".to_string(),
                "cinema".to_string(),
            ]),
            "broadcast",
        ))
        .with_parameter(TemplateParam::optional(
            "report_output",
            ParamType::FilePath,
            tmp.join("oximedia-qc-report.json")
                .to_string_lossy()
                .into_owned(),
        ))
        .with_parameter(TemplateParam::optional(
            "auto_approve_threshold",
            ParamType::Float,
            "0.95",
        ))
        .with_step(
            TemplateStep::new("analyze", ErrorAction::Fail)
                .with_param("input", "{{media_path}}")
                .with_param("profile", "{{qc_profile}}")
                .with_param("checks", "loudness,black_frames,freeze,color_bars"),
        )
        .with_step(
            TemplateStep::new("generate_report", ErrorAction::Notify)
                .with_param("output", "{{report_output}}")
                .with_param("format", "json"),
        )
        .with_step(
            TemplateStep::new("approve_or_reject", ErrorAction::Fail)
                .with_param("auto_threshold", "{{auto_approve_threshold}}")
                .with_param("manual_review_on_fail", "true"),
        )
    }

    /// Get all built-in templates.
    #[must_use]
    pub fn all() -> Vec<WorkflowTemplate> {
        vec![
            Self::ingest_and_transcode(),
            Self::archive_workflow(),
            Self::qc_review(),
        ]
    }

    /// Find a template by name.
    #[must_use]
    pub fn find_by_name(name: &str) -> Option<WorkflowTemplate> {
        Self::all().into_iter().find(|t| t.name == name)
    }

    /// Find templates by category.
    #[must_use]
    pub fn find_by_category(category: &TemplateCategory) -> Vec<WorkflowTemplate> {
        Self::all()
            .into_iter()
            .filter(|t| &t.category == category)
            .collect()
    }
}

/// Instantiates a workflow template with provided parameters.
pub struct TemplateInstantiator;

impl TemplateInstantiator {
    /// Instantiate a template with provided parameters, returning a workflow JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error string if required parameters are missing.
    pub fn instantiate(
        template: &WorkflowTemplate,
        params: HashMap<std::string::String, std::string::String>,
    ) -> Result<std::string::String, std::string::String> {
        // Validate required parameters
        for param in &template.parameters {
            if param.required && !params.contains_key(&param.name) {
                return Err(format!("Required parameter '{}' is missing", param.name));
            }
        }

        // Collect all params with defaults applied
        let mut resolved_params: HashMap<std::string::String, std::string::String> = HashMap::new();
        for param in &template.parameters {
            if let Some(value) = params.get(&param.name) {
                resolved_params.insert(param.name.clone(), value.clone());
            } else if let Some(default) = &param.default_value {
                resolved_params.insert(param.name.clone(), default.clone());
            }
        }

        // Build JSON representation
        let steps_json: Vec<std::string::String> = template
            .steps
            .iter()
            .map(|step| {
                let params_json: Vec<std::string::String> = step
                    .params
                    .iter()
                    .map(|(k, v)| {
                        let resolved = substitute_params(v, &resolved_params);
                        format!("\"{k}\":\"{resolved}\"")
                    })
                    .collect();

                let on_error_str = match &step.on_error {
                    ErrorAction::Fail => "\"fail\"",
                    ErrorAction::Skip => "\"skip\"",
                    ErrorAction::Notify => "\"notify\"",
                    ErrorAction::Retry(n) => {
                        // We'll inline the retry value
                        return format!(
                            "{{\"step_type\":\"{}\",\"on_error\":{{\"retry\":{}}},\"params\":{{{}}}}}",
                            step.step_type,
                            n,
                            params_json.join(",")
                        );
                    }
                };

                format!(
                    "{{\"step_type\":\"{}\",\"on_error\":{},\"params\":{{{}}}}}",
                    step.step_type,
                    on_error_str,
                    params_json.join(",")
                )
            })
            .collect();

        let params_json: Vec<std::string::String> = resolved_params
            .iter()
            .map(|(k, v)| format!("\"{k}\":\"{v}\""))
            .collect();

        Ok(format!(
            "{{\"name\":\"{}\",\"category\":\"{:?}\",\"params\":{{{}}},\"steps\":[{}]}}",
            template.name,
            template.category,
            params_json.join(","),
            steps_json.join(",")
        ))
    }
}

/// Substitute `{{param_name}}` placeholders in a value string.
fn substitute_params(
    value: &str,
    params: &HashMap<std::string::String, std::string::String>,
) -> std::string::String {
    let mut result = value.to_string();
    for (key, val) in params {
        let placeholder = format!("{{{{{key}}}}}");
        result = result.replace(&placeholder, val);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingest_and_transcode_template() {
        let template = TemplateLibrary::ingest_and_transcode();
        assert_eq!(template.name, "ingest-and-transcode");
        assert_eq!(template.category, TemplateCategory::Ingest);
        assert!(!template.steps.is_empty());
        assert!(!template.parameters.is_empty());
    }

    #[test]
    fn test_archive_workflow_template() {
        let template = TemplateLibrary::archive_workflow();
        assert_eq!(template.name, "archive-workflow");
        assert_eq!(template.category, TemplateCategory::Archive);
        assert_eq!(template.steps.len(), 4);
    }

    #[test]
    fn test_qc_review_template() {
        let template = TemplateLibrary::qc_review();
        assert_eq!(template.name, "qc-review");
        assert_eq!(template.category, TemplateCategory::QC);
        assert_eq!(template.steps.len(), 3);
    }

    #[test]
    fn test_template_library_all() {
        let all = TemplateLibrary::all();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_find_by_name() {
        let template = TemplateLibrary::find_by_name("archive-workflow");
        assert!(template.is_some());
        assert_eq!(
            template.expect("should succeed in test").name,
            "archive-workflow"
        );
    }

    #[test]
    fn test_find_by_name_not_found() {
        let template = TemplateLibrary::find_by_name("nonexistent");
        assert!(template.is_none());
    }

    #[test]
    fn test_find_by_category() {
        let templates = TemplateLibrary::find_by_category(&TemplateCategory::Ingest);
        assert!(!templates.is_empty());
        for t in &templates {
            assert_eq!(t.category, TemplateCategory::Ingest);
        }
    }

    #[test]
    fn test_instantiate_with_required_params() {
        let template = TemplateLibrary::ingest_and_transcode();
        let mut params = HashMap::new();
        params.insert("watch_folder".to_string(), "/mnt/ingest".to_string());
        params.insert("output_folder".to_string(), "/mnt/output".to_string());

        let result = TemplateInstantiator::instantiate(&template, params);
        assert!(result.is_ok());
        let json = result.expect("should succeed in test");
        assert!(json.contains("ingest-and-transcode"));
        assert!(json.contains("/mnt/ingest"));
    }

    #[test]
    fn test_instantiate_missing_required_param() {
        let template = TemplateLibrary::ingest_and_transcode();
        let params = HashMap::new(); // Missing required params

        let result = TemplateInstantiator::instantiate(&template, params);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Required parameter"));
    }

    #[test]
    fn test_instantiate_with_defaults() {
        let template = TemplateLibrary::archive_workflow();
        let mut params = HashMap::new();
        params.insert("source_path".to_string(), "/src".to_string());
        params.insert("archive_destination".to_string(), "/archive".to_string());
        // Not providing optional params - defaults should be used

        let result = TemplateInstantiator::instantiate(&template, params);
        assert!(result.is_ok());
        let json = result.expect("should succeed in test");
        assert!(json.contains("sha256")); // default checksum algorithm
    }

    #[test]
    fn test_param_type_enum() {
        let param = TemplateParam::optional(
            "preset",
            ParamType::Enum(vec!["a".to_string(), "b".to_string()]),
            "a",
        );
        assert_eq!(param.name, "preset");
        assert!(!param.required);
        assert_eq!(param.default_value, Some("a".to_string()));
    }

    #[test]
    fn test_error_action_retry() {
        let step = TemplateStep::new("transcode", ErrorAction::Retry(3));
        assert_eq!(step.on_error, ErrorAction::Retry(3));
    }

    #[test]
    fn test_substitute_params() {
        let mut params = HashMap::new();
        params.insert("folder".to_string(), "/mnt/data".to_string());
        let result = substitute_params("input={{folder}}/file.mp4", &params);
        assert_eq!(result, "input=/mnt/data/file.mp4");
    }
}
