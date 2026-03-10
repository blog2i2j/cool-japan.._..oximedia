//! Watch folder implementation for automatic job submission

use crate::error::{BatchError, Result};
use crate::job::BatchJob;
use crate::BatchEngine;
use notify::{Event, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Watch folder configuration
#[derive(Debug, Clone)]
pub struct WatchConfig {
    /// Folder to watch
    pub folder: PathBuf,
    /// File pattern to match
    pub pattern: Option<String>,
    /// Job template to use
    pub template: String,
    /// Watch recursively
    pub recursive: bool,
    /// Debounce delay in milliseconds
    pub debounce_ms: u64,
}

impl WatchConfig {
    /// Create a new watch configuration
    #[must_use]
    pub fn new(folder: PathBuf) -> Self {
        Self {
            folder,
            pattern: None,
            template: String::new(),
            recursive: false,
            debounce_ms: 1000,
        }
    }

    /// Set file pattern
    #[must_use]
    pub fn with_pattern(mut self, pattern: String) -> Self {
        self.pattern = Some(pattern);
        self
    }

    /// Set template
    #[must_use]
    pub fn with_template(mut self, template: String) -> Self {
        self.template = template;
        self
    }

    /// Set recursive
    #[must_use]
    pub const fn recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }
}

/// Watch folder service
pub struct WatchFolder {
    config: WatchConfig,
    engine: Arc<BatchEngine>,
}

impl WatchFolder {
    /// Create a new watch folder service
    #[must_use]
    pub fn new(config: WatchConfig, engine: Arc<BatchEngine>) -> Self {
        Self { config, engine }
    }

    /// Start watching
    ///
    /// # Errors
    ///
    /// Returns an error if watching fails
    pub async fn start(&self) -> Result<()> {
        let (tx, mut rx) = mpsc::channel(100);

        // Create watcher
        let mut watcher =
            notify::recommended_watcher(move |res: std::result::Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    let _ = tx.blocking_send(event);
                }
            })
            .map_err(|e| BatchError::WatchError(e.to_string()))?;

        // Start watching
        let mode = if self.config.recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        watcher
            .watch(&self.config.folder, mode)
            .map_err(|e| BatchError::WatchError(e.to_string()))?;

        tracing::info!("Watching folder: {}", self.config.folder.display());

        // Process events
        while let Some(event) = rx.recv().await {
            if let Err(e) = self.handle_event(event).await {
                tracing::error!("Error handling event: {}", e);
            }
        }

        Ok(())
    }

    async fn handle_event(&self, event: Event) -> Result<()> {
        match event.kind {
            notify::EventKind::Create(_) | notify::EventKind::Modify(_) => {
                for path in event.paths {
                    if path.is_file() && self.matches_pattern(&path) {
                        self.submit_job_for_file(&path).await?;
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn matches_pattern(&self, path: &Path) -> bool {
        if let Some(pattern) = &self.config.pattern {
            if let Some(filename) = path.file_name() {
                let filename_str = filename.to_string_lossy();
                return glob::Pattern::new(pattern)
                    .ok()
                    .is_some_and(|p| p.matches(&filename_str));
            }
        }

        true
    }

    async fn submit_job_for_file(&self, path: &Path) -> Result<()> {
        tracing::info!("New file detected: {}", path.display());

        // Determine the operation based on the template name
        let operation = match self.config.template.as_str() {
            "transcode" | "transcode-web" | "transcode-mobile" | "transcode-broadcast" => {
                crate::job::BatchOperation::Transcode {
                    preset: self.config.template.clone(),
                }
            }
            "qc" | "qc-full" | "qc-quick" => crate::job::BatchOperation::QualityCheck {
                profile: self.config.template.clone(),
            },
            "analyze" | "analyze-video" | "analyze-audio" => crate::job::BatchOperation::Analyze {
                analysis_type: crate::operations::AnalysisType::VideoQuality,
            },
            _ => crate::job::BatchOperation::FileOp {
                operation: crate::operations::FileOperation::Copy { overwrite: false },
            },
        };

        let job_name = format!("Auto[{}]: {}", self.config.template, path.display());
        let mut job = BatchJob::new(job_name, operation);

        // Add the detected file as input
        job.add_input(crate::job::InputSpec::new(
            path.to_string_lossy().to_string(),
        ));

        let job_id = self.engine.submit_job(job).await?;
        tracing::info!("Job submitted: {}", job_id);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_watch_config_creation() {
        let config = WatchConfig::new(PathBuf::from("/tmp"));
        assert_eq!(config.folder, PathBuf::from("/tmp"));
        assert!(!config.recursive);
    }

    #[test]
    fn test_watch_config_builder() {
        let config = WatchConfig::new(PathBuf::from("/tmp"))
            .with_pattern("*.mp4".to_string())
            .with_template("transcode".to_string())
            .recursive(true);

        assert_eq!(config.pattern, Some("*.mp4".to_string()));
        assert_eq!(config.template, "transcode");
        assert!(config.recursive);
    }

    #[tokio::test]
    async fn test_watch_folder_creation() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let db_path = temp_dir.path().join("test.db");
        let engine = Arc::new(
            BatchEngine::new(db_path.to_str().expect("path should be valid UTF-8"), 2)
                .expect("path should be valid UTF-8"),
        );

        let config = WatchConfig::new(temp_dir.path().to_path_buf());
        let watcher = WatchFolder::new(config, engine);

        assert!(std::mem::size_of_val(&watcher) > 0);
    }
}
