//! Job definition and configuration

use crate::operations::{AnalysisType, FileOperation, OutputFormat};
use crate::types::{JobContext, JobId, Priority, ResourceRequirements, RetryPolicy, Schedule};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Batch job definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchJob {
    /// Unique job identifier
    pub id: JobId,
    /// Human-readable job name
    pub name: String,
    /// Job operation
    pub operation: BatchOperation,
    /// Input file specifications
    pub inputs: Vec<InputSpec>,
    /// Output specifications
    pub outputs: Vec<OutputSpec>,
    /// Job priority
    pub priority: Priority,
    /// Retry policy
    pub retry: RetryPolicy,
    /// Job dependencies (must complete before this job)
    pub dependencies: Vec<JobId>,
    /// Schedule configuration
    pub schedule: Schedule,
    /// Resource requirements
    pub resources: ResourceRequirements,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Job context (runtime state)
    #[serde(skip)]
    pub context: Option<JobContext>,
}

impl BatchJob {
    /// Create a new batch job
    #[must_use]
    pub fn new(name: String, operation: BatchOperation) -> Self {
        let id = JobId::new();
        Self {
            id: id.clone(),
            name: name.clone(),
            operation,
            inputs: Vec::new(),
            outputs: Vec::new(),
            priority: Priority::default(),
            retry: RetryPolicy::default(),
            dependencies: Vec::new(),
            schedule: Schedule::default(),
            resources: ResourceRequirements::default(),
            metadata: HashMap::new(),
            context: Some(JobContext::new(id, name)),
        }
    }

    /// Add an input specification
    pub fn add_input(&mut self, input: InputSpec) {
        self.inputs.push(input);
    }

    /// Add an output specification
    pub fn add_output(&mut self, output: OutputSpec) {
        self.outputs.push(output);
    }

    /// Add a dependency
    pub fn add_dependency(&mut self, job_id: JobId) {
        self.dependencies.push(job_id);
    }

    /// Set priority
    pub fn set_priority(&mut self, priority: Priority) {
        self.priority = priority;
    }

    /// Set retry policy
    pub fn set_retry_policy(&mut self, retry: RetryPolicy) {
        self.retry = retry;
    }

    /// Set schedule
    pub fn set_schedule(&mut self, schedule: Schedule) {
        self.schedule = schedule;
    }

    /// Set resource requirements
    pub fn set_resources(&mut self, resources: ResourceRequirements) {
        self.resources = resources;
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get metadata value
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

/// Batch operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchOperation {
    /// Transcode operation
    Transcode {
        /// Preset name
        preset: String,
    },
    /// Quality check operation
    QualityCheck {
        /// QC profile name
        profile: String,
    },
    /// Analysis operation
    Analyze {
        /// Type of analysis
        analysis_type: AnalysisType,
    },
    /// File operation
    FileOp {
        /// File operation type
        operation: FileOperation,
    },
    /// Custom script operation
    Custom {
        /// Path to script file
        script: PathBuf,
    },
    /// Pipeline (multi-step)
    Pipeline {
        /// Ordered list of operations
        steps: Vec<PipelineStep>,
    },
}

/// Pipeline step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStep {
    /// Step name
    pub name: String,
    /// Step operation
    pub operation: BatchOperation,
    /// Continue on error
    pub continue_on_error: bool,
    /// Condition for execution (Lua expression)
    pub condition: Option<String>,
}

/// Input file specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSpec {
    /// File pattern (glob or regex)
    pub pattern: String,
    /// Search recursively
    pub recursive: bool,
    /// File filters
    pub filters: Vec<FileFilter>,
    /// Base directory
    pub base_dir: Option<PathBuf>,
}

impl InputSpec {
    /// Create a new input specification
    #[must_use]
    pub fn new(pattern: String) -> Self {
        Self {
            pattern,
            recursive: false,
            filters: Vec::new(),
            base_dir: None,
        }
    }

    /// Set recursive search
    #[must_use]
    pub fn recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    /// Add a filter
    #[must_use]
    pub fn with_filter(mut self, filter: FileFilter) -> Self {
        self.filters.push(filter);
        self
    }

    /// Set base directory
    #[must_use]
    pub fn with_base_dir(mut self, base_dir: PathBuf) -> Self {
        self.base_dir = Some(base_dir);
        self
    }
}

/// File filter criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileFilter {
    /// File extension filter
    Extension(String),
    /// Minimum file size in bytes
    MinSize(u64),
    /// Maximum file size in bytes
    MaxSize(u64),
    /// Modified after date (ISO 8601)
    ModifiedAfter(String),
    /// Modified before date (ISO 8601)
    ModifiedBefore(String),
    /// Custom Lua filter
    Custom(String),
}

/// Output file specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSpec {
    /// Output path template
    pub template: String,
    /// Output format
    pub format: OutputFormat,
    /// Overwrite existing files
    pub overwrite: bool,
    /// Create directories
    pub create_dirs: bool,
    /// Additional options
    pub options: HashMap<String, String>,
}

impl OutputSpec {
    /// Create a new output specification
    #[must_use]
    pub fn new(template: String, format: OutputFormat) -> Self {
        Self {
            template,
            format,
            overwrite: false,
            create_dirs: true,
            options: HashMap::new(),
        }
    }

    /// Set overwrite flag
    #[must_use]
    pub fn overwrite(mut self, overwrite: bool) -> Self {
        self.overwrite = overwrite;
        self
    }

    /// Set create directories flag
    #[must_use]
    pub fn create_dirs(mut self, create_dirs: bool) -> Self {
        self.create_dirs = create_dirs;
        self
    }

    /// Add an option
    #[must_use]
    pub fn with_option(mut self, key: String, value: String) -> Self {
        self.options.insert(key, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_job_creation() {
        let job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        assert_eq!(job.name, "test-job");
        assert_eq!(job.priority, Priority::Normal);
        assert!(job.inputs.is_empty());
        assert!(job.outputs.is_empty());
    }

    #[test]
    fn test_add_input() {
        let mut job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        job.add_input(InputSpec::new("*.mp4".to_string()));
        assert_eq!(job.inputs.len(), 1);
    }

    #[test]
    fn test_add_output() {
        let mut job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        job.add_output(OutputSpec::new("output.mp4".to_string(), OutputFormat::Mp4));
        assert_eq!(job.outputs.len(), 1);
    }

    #[test]
    fn test_add_dependency() {
        let mut job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        job.add_dependency(JobId::new());
        assert_eq!(job.dependencies.len(), 1);
    }

    #[test]
    fn test_set_priority() {
        let mut job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        job.set_priority(Priority::High);
        assert_eq!(job.priority, Priority::High);
    }

    #[test]
    fn test_metadata() {
        let mut job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        job.add_metadata("project".to_string(), "test-project".to_string());
        assert_eq!(
            job.get_metadata("project"),
            Some(&"test-project".to_string())
        );
    }

    #[test]
    fn test_input_spec_builder() {
        let tmp = std::env::temp_dir();
        let input = InputSpec::new("*.mp4".to_string())
            .recursive(true)
            .with_filter(FileFilter::Extension("mp4".to_string()))
            .with_base_dir(tmp.clone());

        assert!(input.recursive);
        assert_eq!(input.filters.len(), 1);
        assert_eq!(input.base_dir, Some(tmp));
    }

    #[test]
    fn test_output_spec_builder() {
        let output = OutputSpec::new("output.mp4".to_string(), OutputFormat::Mp4)
            .overwrite(true)
            .create_dirs(false)
            .with_option("bitrate".to_string(), "5000k".to_string());

        assert!(output.overwrite);
        assert!(!output.create_dirs);
        assert_eq!(output.options.get("bitrate"), Some(&"5000k".to_string()));
    }
}
