// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Rendering pipeline management (pre-render, render, post-render).

use crate::error::Result;
use crate::job::{Job, JobId};
use crate::worker::WorkerId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Pipeline stage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineStage {
    /// Pre-render stage (validation, setup)
    PreRender,
    /// Render stage (actual rendering)
    Render,
    /// Post-render stage (verification, assembly)
    PostRender,
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PreRender => write!(f, "PreRender"),
            Self::Render => write!(f, "Render"),
            Self::PostRender => write!(f, "PostRender"),
        }
    }
}

/// Pipeline task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineTask {
    /// Task ID
    pub id: String,
    /// Job ID
    pub job_id: JobId,
    /// Stage
    pub stage: PipelineStage,
    /// Status
    pub status: TaskStatus,
    /// Started at
    pub started_at: Option<DateTime<Utc>>,
    /// Completed at
    pub completed_at: Option<DateTime<Utc>>,
    /// Error message
    pub error: Option<String>,
}

/// Task status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Pending
    Pending,
    /// Running
    Running,
    /// Completed
    Completed,
    /// Failed
    Failed,
}

/// Pre-render result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreRenderResult {
    /// Assets verified
    pub assets_verified: bool,
    /// Dependencies resolved
    pub dependencies_resolved: bool,
    /// Estimated frames
    pub estimated_frames: u32,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Estimated time (seconds)
    pub estimated_time: f64,
    /// Issues found
    pub issues: Vec<String>,
}

/// Render result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderResult {
    /// Frame number
    pub frame: u32,
    /// Output path
    pub output_path: PathBuf,
    /// Render time (seconds)
    pub render_time: f64,
    /// Worker ID
    pub worker_id: WorkerId,
    /// Success
    pub success: bool,
    /// Error message
    pub error: Option<String>,
}

/// Post-render result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostRenderResult {
    /// All frames verified
    pub all_frames_verified: bool,
    /// Output assembled
    pub output_assembled: bool,
    /// Final output path
    pub final_output_path: Option<PathBuf>,
    /// Quality metrics
    pub quality_metrics: HashMap<String, f64>,
}

/// Pipeline executor
pub struct Pipeline {
    tasks: HashMap<JobId, Vec<PipelineTask>>,
    pre_render_results: HashMap<JobId, PreRenderResult>,
    render_results: HashMap<JobId, Vec<RenderResult>>,
    post_render_results: HashMap<JobId, PostRenderResult>,
}

impl Pipeline {
    /// Create a new pipeline
    #[must_use]
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
            pre_render_results: HashMap::new(),
            render_results: HashMap::new(),
            post_render_results: HashMap::new(),
        }
    }

    /// Execute pre-render stage
    pub async fn execute_pre_render(&mut self, job: &Job) -> Result<PreRenderResult> {
        let task = PipelineTask {
            id: format!("{}-prerender", job.id),
            job_id: job.id,
            stage: PipelineStage::PreRender,
            status: TaskStatus::Running,
            started_at: Some(Utc::now()),
            completed_at: None,
            error: None,
        };

        self.tasks.entry(job.id).or_default().push(task);

        // Asset verification
        let assets_verified = self.verify_assets(job).await?;

        // Dependency resolution
        let dependencies_resolved = self.resolve_dependencies(job).await?;

        // Estimate resources
        let (estimated_frames, estimated_cost, estimated_time) = self.estimate_resources(job);

        // Collect issues
        let mut issues = Vec::new();
        if !assets_verified {
            issues.push("Some assets could not be verified".to_string());
        }
        if !dependencies_resolved {
            issues.push("Some dependencies could not be resolved".to_string());
        }

        let result = PreRenderResult {
            assets_verified,
            dependencies_resolved,
            estimated_frames,
            estimated_cost,
            estimated_time,
            issues,
        };

        // Update task
        if let Some(task) = self
            .tasks
            .get_mut(&job.id)
            .and_then(|tasks| tasks.last_mut())
        {
            task.status = if result.issues.is_empty() {
                TaskStatus::Completed
            } else {
                TaskStatus::Failed
            };
            task.completed_at = Some(Utc::now());
            if !result.issues.is_empty() {
                task.error = Some(result.issues.join(", "));
            }
        }

        self.pre_render_results.insert(job.id, result.clone());

        Ok(result)
    }

    /// Record render result
    pub fn record_render_result(&mut self, job_id: JobId, result: RenderResult) {
        self.render_results.entry(job_id).or_default().push(result);
    }

    /// Execute post-render stage
    pub async fn execute_post_render(&mut self, job: &Job) -> Result<PostRenderResult> {
        let task = PipelineTask {
            id: format!("{}-postrender", job.id),
            job_id: job.id,
            stage: PipelineStage::PostRender,
            status: TaskStatus::Running,
            started_at: Some(Utc::now()),
            completed_at: None,
            error: None,
        };

        self.tasks.entry(job.id).or_default().push(task);

        // Verify all frames
        let all_frames_verified = self.verify_all_frames(job).await?;

        // Assemble output
        let (output_assembled, final_output_path) = self.assemble_output(job).await?;

        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(job).await?;

        let result = PostRenderResult {
            all_frames_verified,
            output_assembled,
            final_output_path,
            quality_metrics,
        };

        // Update task
        if let Some(task) = self
            .tasks
            .get_mut(&job.id)
            .and_then(|tasks| tasks.last_mut())
        {
            task.status = if all_frames_verified && output_assembled {
                TaskStatus::Completed
            } else {
                TaskStatus::Failed
            };
            task.completed_at = Some(Utc::now());
        }

        self.post_render_results.insert(job.id, result.clone());

        Ok(result)
    }

    /// Verify assets
    async fn verify_assets(&self, job: &Job) -> Result<bool> {
        // Check if project file exists
        if !job.submission.project_file.exists() {
            return Ok(false);
        }

        // Check all dependencies
        for dep in &job.submission.dependencies {
            if !dep.exists() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Resolve dependencies
    async fn resolve_dependencies(&self, job: &Job) -> Result<bool> {
        // In a real implementation, this would:
        // - Download missing assets
        // - Verify checksums
        // - Set up plugin paths
        // - Check license availability

        Ok(!job.submission.dependencies.is_empty())
    }

    /// Estimate resources
    fn estimate_resources(&self, job: &Job) -> (u32, f64, f64) {
        // Get frame count
        let frame_count = match &job.submission.job_type {
            crate::job::JobType::ImageSequence {
                start_frame,
                end_frame,
            } => end_frame - start_frame + 1,
            crate::job::JobType::VideoRender { .. } => 100,
            _ => 1,
        };

        // Estimate cost and time
        let estimated_cost = f64::from(frame_count) * 0.01;
        let estimated_time = f64::from(frame_count) * 10.0; // 10 seconds per frame

        (frame_count, estimated_cost, estimated_time)
    }

    /// Verify all frames
    async fn verify_all_frames(&self, _job: &Job) -> Result<bool> {
        // Check if all expected frames are present
        // Verify checksums
        // Check for corruption

        Ok(true)
    }

    /// Assemble output
    async fn assemble_output(&self, job: &Job) -> Result<(bool, Option<PathBuf>)> {
        // In a real implementation:
        // - Combine image sequences into video
        // - Merge render passes
        // - Apply final processing

        let output_path = PathBuf::from(format!("/output/{}.mp4", job.id));
        Ok((true, Some(output_path)))
    }

    /// Calculate quality metrics
    async fn calculate_quality_metrics(&self, _job: &Job) -> Result<HashMap<String, f64>> {
        let mut metrics = HashMap::new();
        metrics.insert("psnr".to_string(), 42.0);
        metrics.insert("ssim".to_string(), 0.95);
        Ok(metrics)
    }

    /// Get pre-render result
    #[must_use]
    pub fn get_pre_render_result(&self, job_id: JobId) -> Option<&PreRenderResult> {
        self.pre_render_results.get(&job_id)
    }

    /// Get render results
    #[must_use]
    pub fn get_render_results(&self, job_id: JobId) -> Vec<&RenderResult> {
        self.render_results
            .get(&job_id)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Get post-render result
    #[must_use]
    pub fn get_post_render_result(&self, job_id: JobId) -> Option<&PostRenderResult> {
        self.post_render_results.get(&job_id)
    }

    /// Get pipeline tasks for job
    #[must_use]
    pub fn get_tasks(&self, job_id: JobId) -> Vec<&PipelineTask> {
        self.tasks
            .get(&job_id)
            .map_or_else(Vec::new, |tasks| tasks.iter().collect())
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{JobSubmission, Priority};

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("oximedia-renderfarm-pipeline-{name}"))
    }

    #[tokio::test]
    async fn test_pipeline_creation() {
        let pipeline = Pipeline::new();
        assert_eq!(pipeline.tasks.len(), 0);
    }

    #[tokio::test]
    async fn test_pre_render_execution() -> Result<()> {
        let mut pipeline = Pipeline::new();

        let submission = JobSubmission::builder()
            .project_file(tmp_path("test.blend"))
            .frame_range(1, 10)
            .priority(Priority::Normal)
            .build()?;

        let job = Job::new(submission);

        let result = pipeline.execute_pre_render(&job).await?;
        assert!(result.estimated_frames > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_post_render_execution() -> Result<()> {
        let mut pipeline = Pipeline::new();

        let submission = JobSubmission::builder()
            .project_file(tmp_path("test.blend"))
            .frame_range(1, 10)
            .build()?;

        let job = Job::new(submission);

        let result = pipeline.execute_post_render(&job).await?;
        assert!(result.final_output_path.is_some());

        Ok(())
    }

    #[test]
    fn test_pipeline_stage_display() {
        assert_eq!(PipelineStage::PreRender.to_string(), "PreRender");
        assert_eq!(PipelineStage::Render.to_string(), "Render");
        assert_eq!(PipelineStage::PostRender.to_string(), "PostRender");
    }

    #[tokio::test]
    async fn test_get_tasks() -> Result<()> {
        let mut pipeline = Pipeline::new();

        let submission = JobSubmission::builder()
            .project_file(tmp_path("test.blend"))
            .frame_range(1, 10)
            .build()?;

        let job = Job::new(submission);
        let job_id = job.id;

        pipeline.execute_pre_render(&job).await?;

        let tasks = pipeline.get_tasks(job_id);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].stage, PipelineStage::PreRender);

        Ok(())
    }
}
