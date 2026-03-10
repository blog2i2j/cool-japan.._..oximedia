//! Migration execution

use super::{MigrationPlan, MigrationStrategy};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Migration execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationResult {
    /// Source file
    pub source: PathBuf,
    /// Output file(s)
    pub outputs: Vec<PathBuf>,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Execution time in milliseconds
    pub duration_ms: u64,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Migration executor
pub struct MigrationExecutor {
    dry_run: bool,
    preserve_original: bool,
}

impl Default for MigrationExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl MigrationExecutor {
    /// Create a new migration executor
    #[must_use]
    pub fn new() -> Self {
        Self {
            dry_run: false,
            preserve_original: true,
        }
    }

    /// Enable dry-run mode (no actual changes)
    #[must_use]
    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Set whether to preserve original files
    #[must_use]
    pub fn with_preserve_original(mut self, preserve: bool) -> Self {
        self.preserve_original = preserve;
        self
    }

    /// Execute a migration plan
    ///
    /// # Errors
    ///
    /// Returns an error if migration fails
    pub fn execute(&self, plan: &MigrationPlan) -> Result<MigrationResult> {
        let start = std::time::Instant::now();

        if self.dry_run {
            return Ok(MigrationResult {
                source: plan.source.clone(),
                outputs: vec![self.get_output_path(&plan.source, &plan.target_format)],
                success: true,
                error: None,
                duration_ms: 0,
                timestamp: chrono::Utc::now(),
            });
        }

        let result = match &plan.strategy {
            MigrationStrategy::Direct => self.execute_direct(plan),
            MigrationStrategy::TwoStep { intermediate } => {
                self.execute_two_step(plan, intermediate)
            }
            MigrationStrategy::KeepBoth => self.execute_keep_both(plan),
            MigrationStrategy::Replace => self.execute_replace(plan),
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(outputs) => Ok(MigrationResult {
                source: plan.source.clone(),
                outputs,
                success: true,
                error: None,
                duration_ms,
                timestamp: chrono::Utc::now(),
            }),
            Err(e) => Ok(MigrationResult {
                source: plan.source.clone(),
                outputs: Vec::new(),
                success: false,
                error: Some(e.to_string()),
                duration_ms,
                timestamp: chrono::Utc::now(),
            }),
        }
    }

    fn execute_direct(&self, plan: &MigrationPlan) -> Result<Vec<PathBuf>> {
        // Placeholder for actual transcoding
        // In a real implementation, this would call FFmpeg or similar
        let output = self.get_output_path(&plan.source, &plan.target_format);

        // For now, just copy the file as a placeholder
        fs::copy(&plan.source, &output)?;

        Ok(vec![output])
    }

    fn execute_two_step(&self, plan: &MigrationPlan, _intermediate: &str) -> Result<Vec<PathBuf>> {
        // Placeholder for two-step migration
        self.execute_direct(plan)
    }

    fn execute_keep_both(&self, plan: &MigrationPlan) -> Result<Vec<PathBuf>> {
        let output = self.get_output_path(&plan.source, &plan.target_format);
        fs::copy(&plan.source, &output)?;
        Ok(vec![plan.source.clone(), output])
    }

    fn execute_replace(&self, plan: &MigrationPlan) -> Result<Vec<PathBuf>> {
        let output = self.get_output_path(&plan.source, &plan.target_format);
        fs::copy(&plan.source, &output)?;

        if !self.preserve_original {
            fs::remove_file(&plan.source)?;
        }

        Ok(vec![output])
    }

    fn get_output_path(&self, source: &PathBuf, target: &crate::PreservationFormat) -> PathBuf {
        let mut output = source.clone();
        output.set_extension(target.extension());
        output
    }

    /// Execute multiple migrations in batch
    ///
    /// # Errors
    ///
    /// Returns an error if batch execution fails
    pub fn execute_batch(&self, plans: &[MigrationPlan]) -> Result<Vec<MigrationResult>> {
        plans.iter().map(|plan| self.execute(plan)).collect()
    }

    /// Execute migrations in parallel using rayon
    ///
    /// # Errors
    ///
    /// Returns an error if parallel execution fails
    pub fn execute_parallel(&self, plans: &[MigrationPlan]) -> Result<Vec<MigrationResult>> {
        use rayon::prelude::*;

        Ok(plans
            .par_iter()
            .map(|plan| {
                self.execute(plan).unwrap_or_else(|e| MigrationResult {
                    source: plan.source.clone(),
                    outputs: Vec::new(),
                    success: false,
                    error: Some(e.to_string()),
                    duration_ms: 0,
                    timestamp: chrono::Utc::now(),
                })
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PreservationFormat;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_dry_run_migration() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Test content")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let plan = MigrationPlan {
            source: file.path().to_path_buf(),
            source_format: "mp4".to_string(),
            target_format: PreservationFormat::VideoFfv1Mkv,
            strategy: MigrationStrategy::Direct,
            priority: super::super::MigrationPriority::Medium,
            risk_score: 0.5,
            recommended_date: None,
        };

        let executor = MigrationExecutor::new().with_dry_run(true);
        let result = executor.execute(&plan).expect("operation should succeed");

        assert!(result.success);
        assert_eq!(result.outputs.len(), 1);
    }

    #[test]
    fn test_keep_both_strategy() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Test content")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let plan = MigrationPlan {
            source: file.path().to_path_buf(),
            source_format: "mp4".to_string(),
            target_format: PreservationFormat::VideoFfv1Mkv,
            strategy: MigrationStrategy::KeepBoth,
            priority: super::super::MigrationPriority::Medium,
            risk_score: 0.5,
            recommended_date: None,
        };

        let executor = MigrationExecutor::new();
        let result = executor.execute(&plan).expect("operation should succeed");

        assert!(result.success);
        // KeepBoth should return both source and output
        assert_eq!(result.outputs.len(), 2);
    }
}
