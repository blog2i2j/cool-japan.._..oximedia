//! Comprehensive examples of batch processing usage

#![allow(dead_code)]

use crate::job::{BatchJob, BatchOperation, InputSpec, OutputSpec, PipelineStep};
use crate::operations::{AnalysisType, FileOperation, OutputFormat};
use crate::types::{Priority, ResourceRequirements, RetryPolicy, Schedule};
use crate::{BatchEngine, Result};
use std::path::PathBuf;
use std::sync::Arc;

/// Example: Basic file copy job
///
/// This example demonstrates how to create a simple job that copies files
/// from one location to another.
///
/// # Errors
///
/// Returns an error if job submission fails
pub async fn example_basic_copy(engine: &Arc<BatchEngine>) -> Result<()> {
    let mut job = BatchJob::new(
        "Copy Media Files".to_string(),
        BatchOperation::FileOp {
            operation: FileOperation::Copy { overwrite: false },
        },
    );

    // Add input specification
    job.add_input(InputSpec::new("*.mp4".to_string()).recursive(false));

    // Add output specification
    job.add_output(OutputSpec::new(
        "/output/{filename}".to_string(),
        OutputFormat::Mp4,
    ));

    // Set priority
    job.set_priority(Priority::Normal);

    // Submit the job
    let job_id = engine.submit_job(job).await?;
    tracing::info!("Submitted copy job: {}", job_id);

    Ok(())
}

/// Example: Transcoding with multiple outputs
///
/// This example shows how to transcode a video file to multiple formats
/// and qualities simultaneously.
///
/// # Errors
///
/// Returns an error if job submission fails
pub async fn example_multi_output_transcode(engine: &Arc<BatchEngine>) -> Result<()> {
    let mut job = BatchJob::new(
        "Multi-Output Transcode".to_string(),
        BatchOperation::Transcode {
            preset: "web".to_string(),
        },
    );

    // Input
    job.add_input(InputSpec::new("source.mp4".to_string()));

    // Multiple outputs
    job.add_output(OutputSpec::new(
        "output_1080p.mp4".to_string(),
        OutputFormat::Mp4,
    ));
    job.add_output(OutputSpec::new(
        "output_720p.mp4".to_string(),
        OutputFormat::Mp4,
    ));
    job.add_output(OutputSpec::new(
        "output_480p.mp4".to_string(),
        OutputFormat::Mp4,
    ));

    // Set high priority for important transcode jobs
    job.set_priority(Priority::High);

    // Set retry policy for reliability
    job.set_retry_policy(RetryPolicy::new(3, 60, true));

    // Set resource requirements
    job.set_resources(ResourceRequirements {
        cpu_cores: Some(4),
        memory_mb: Some(4096),
        gpu: true,
        disk_space_mb: Some(10240),
    });

    let job_id = engine.submit_job(job).await?;
    tracing::info!("Submitted multi-output transcode job: {}", job_id);

    Ok(())
}

/// Example: Quality control check
///
/// This example demonstrates running automated quality control checks
/// on media files.
///
/// # Errors
///
/// Returns an error if job submission fails
pub async fn example_quality_control(engine: &Arc<BatchEngine>) -> Result<()> {
    let mut job = BatchJob::new(
        "QC Check".to_string(),
        BatchOperation::QualityCheck {
            profile: "full".to_string(),
        },
    );

    // Input pattern
    job.add_input(
        InputSpec::new("**/*.mp4".to_string())
            .recursive(true)
            .with_base_dir(PathBuf::from("/incoming")),
    );

    // Metadata
    job.add_metadata("project".to_string(), "production-2024".to_string());
    job.add_metadata("client".to_string(), "ACME Corp".to_string());

    let job_id = engine.submit_job(job).await?;
    tracing::info!("Submitted QC job: {}", job_id);

    Ok(())
}

/// Example: Scheduled batch job
///
/// This example shows how to schedule a job to run at a specific time.
///
/// # Errors
///
/// Returns an error if job submission fails
///
/// # Panics
///
/// Panics if the scheduled time cannot be constructed (invalid date arithmetic).
pub async fn example_scheduled_job(engine: &Arc<BatchEngine>) -> Result<()> {
    let mut job = BatchJob::new(
        "Nightly Archive".to_string(),
        BatchOperation::FileOp {
            operation: FileOperation::Archive {
                format: crate::operations::file_ops::ArchiveFormat::TarGz,
                compression: 9,
            },
        },
    );

    job.add_input(InputSpec::new("daily_files/*".to_string()));
    job.add_output(OutputSpec::new(
        "archive_{date}.tar.gz".to_string(),
        OutputFormat::Custom("tar.gz".to_string()),
    ));

    // Schedule for tomorrow at 2 AM
    let tomorrow = chrono::Utc::now() + chrono::Duration::days(1);
    let scheduled_time = tomorrow
        .date_naive()
        .and_hms_opt(2, 0, 0)
        .ok_or_else(|| crate::error::BatchError::InvalidJobConfig("invalid time 02:00:00".into()))?
        .and_utc();

    job.set_schedule(Schedule::At(scheduled_time));

    let job_id = engine.submit_job(job).await?;
    tracing::info!("Submitted scheduled job: {}", job_id);

    Ok(())
}

/// Example: Recurring batch job
///
/// This example demonstrates creating a recurring job that runs on a schedule.
///
/// # Errors
///
/// Returns an error if job submission fails
pub async fn example_recurring_job(engine: &Arc<BatchEngine>) -> Result<()> {
    let mut job = BatchJob::new(
        "Hourly Cleanup".to_string(),
        BatchOperation::FileOp {
            operation: FileOperation::Delete { confirm: true },
        },
    );

    job.add_input(InputSpec::new("temp/*.tmp".to_string()));

    // Run every hour
    job.set_schedule(Schedule::Recurring {
        expression: "0 * * * *".to_string(), // Cron expression
    });

    let job_id = engine.submit_job(job).await?;
    tracing::info!("Submitted recurring job: {}", job_id);

    Ok(())
}

/// Example: Job with dependencies
///
/// This example shows how to create jobs that depend on other jobs completing first.
///
/// # Errors
///
/// Returns an error if job submission fails
pub async fn example_job_dependencies(engine: &Arc<BatchEngine>) -> Result<()> {
    // Job 1: Download files
    let mut job1 = BatchJob::new(
        "Download Files".to_string(),
        BatchOperation::FileOp {
            operation: FileOperation::Copy { overwrite: false },
        },
    );
    job1.add_input(InputSpec::new("source/*.mp4".to_string()));
    let job1_id = engine.submit_job(job1).await?;

    // Job 2: Transcode (depends on job1)
    let mut job2 = BatchJob::new(
        "Transcode".to_string(),
        BatchOperation::Transcode {
            preset: "web".to_string(),
        },
    );
    job2.add_dependency(job1_id.clone());
    let job2_id = engine.submit_job(job2).await?;

    // Job 3: QC Check (depends on job2)
    let mut job3 = BatchJob::new(
        "QC Check".to_string(),
        BatchOperation::QualityCheck {
            profile: "quick".to_string(),
        },
    );
    job3.add_dependency(job2_id.clone());
    let job3_id = engine.submit_job(job3).await?;

    tracing::info!(
        "Submitted job chain: {} -> {} -> {}",
        job1_id,
        job2_id,
        job3_id
    );

    Ok(())
}

/// Example: Multi-step pipeline
///
/// This example demonstrates creating a complex pipeline with multiple steps.
///
/// # Errors
///
/// Returns an error if job submission fails
pub async fn example_pipeline(engine: &Arc<BatchEngine>) -> Result<()> {
    let pipeline_steps = vec![
        PipelineStep {
            name: "Copy Input".to_string(),
            operation: BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
            continue_on_error: false,
            condition: None,
        },
        PipelineStep {
            name: "Transcode".to_string(),
            operation: BatchOperation::Transcode {
                preset: "web".to_string(),
            },
            continue_on_error: false,
            condition: Some("file_size > 1048576".to_string()),
        },
        PipelineStep {
            name: "QC Check".to_string(),
            operation: BatchOperation::QualityCheck {
                profile: "full".to_string(),
            },
            continue_on_error: true,
            condition: None,
        },
        PipelineStep {
            name: "Create Archive".to_string(),
            operation: BatchOperation::FileOp {
                operation: FileOperation::Archive {
                    format: crate::operations::file_ops::ArchiveFormat::Zip,
                    compression: 6,
                },
            },
            continue_on_error: false,
            condition: None,
        },
    ];

    let mut job = BatchJob::new(
        "Complete Pipeline".to_string(),
        BatchOperation::Pipeline {
            steps: pipeline_steps,
        },
    );

    job.add_input(InputSpec::new("*.mp4".to_string()));
    job.set_priority(Priority::High);

    let job_id = engine.submit_job(job).await?;
    tracing::info!("Submitted pipeline job: {}", job_id);

    Ok(())
}

/// Example: Batch analysis jobs
///
/// This example shows how to run various analysis operations on media files.
///
/// # Errors
///
/// Returns an error if job submission fails
pub async fn example_batch_analysis(engine: &Arc<BatchEngine>) -> Result<()> {
    // Scene detection
    let mut scene_job = BatchJob::new(
        "Scene Detection".to_string(),
        BatchOperation::Analyze {
            analysis_type: AnalysisType::SceneDetection,
        },
    );
    scene_job.add_input(InputSpec::new("*.mp4".to_string()));
    let scene_id = engine.submit_job(scene_job).await?;

    // Black frame detection
    let mut black_job = BatchJob::new(
        "Black Frame Detection".to_string(),
        BatchOperation::Analyze {
            analysis_type: AnalysisType::BlackFrameDetection,
        },
    );
    black_job.add_input(InputSpec::new("*.mp4".to_string()));
    let black_id = engine.submit_job(black_job).await?;

    // Audio loudness measurement
    let mut loudness_job = BatchJob::new(
        "Loudness Measurement".to_string(),
        BatchOperation::Analyze {
            analysis_type: AnalysisType::LoudnessMeasurement,
        },
    );
    loudness_job.add_input(InputSpec::new("*.mp4".to_string()));
    let loudness_id = engine.submit_job(loudness_job).await?;

    tracing::info!(
        "Submitted analysis jobs: {}, {}, {}",
        scene_id,
        black_id,
        loudness_id
    );

    Ok(())
}

/// Example: Archive creation with multiple files
///
/// This example shows how to create archives from multiple source files.
///
/// # Errors
///
/// Returns an error if job submission fails
pub async fn example_archive_creation(engine: &Arc<BatchEngine>) -> Result<()> {
    let mut job = BatchJob::new(
        "Create Project Archive".to_string(),
        BatchOperation::FileOp {
            operation: FileOperation::Archive {
                format: crate::operations::file_ops::ArchiveFormat::Zip,
                compression: 9,
            },
        },
    );

    // Include multiple file patterns
    job.add_input(InputSpec::new("project/**/*.mp4".to_string()).recursive(true));
    job.add_input(InputSpec::new("project/**/*.xml".to_string()).recursive(true));
    job.add_input(InputSpec::new("project/**/*.aaf".to_string()).recursive(true));

    job.add_output(OutputSpec::new(
        "project_{date}.zip".to_string(),
        OutputFormat::Custom("zip".to_string()),
    ));

    let job_id = engine.submit_job(job).await?;
    tracing::info!("Submitted archive job: {}", job_id);

    Ok(())
}

/// Example: File verification with checksums
///
/// This example demonstrates calculating and verifying file checksums.
///
/// # Errors
///
/// Returns an error if job submission fails
pub async fn example_checksum_verification(engine: &Arc<BatchEngine>) -> Result<()> {
    let mut job = BatchJob::new(
        "Calculate Checksums".to_string(),
        BatchOperation::FileOp {
            operation: FileOperation::Checksum {
                algorithm: crate::operations::file_ops::HashAlgorithm::Sha256,
            },
        },
    );

    job.add_input(InputSpec::new("deliverables/*.mp4".to_string()));

    // Output checksum files
    job.add_output(OutputSpec::new(
        "{filename}.sha256".to_string(),
        OutputFormat::Custom("txt".to_string()),
    ));

    let job_id = engine.submit_job(job).await?;
    tracing::info!("Submitted checksum job: {}", job_id);

    Ok(())
}

/// Example: Broadcast delivery workflow
///
/// This example shows a complete workflow for broadcast delivery.
///
/// # Errors
///
/// Returns an error if job submission fails
pub async fn example_broadcast_workflow(engine: &Arc<BatchEngine>) -> Result<()> {
    // Step 1: QC check
    let mut qc_job = BatchJob::new(
        "Broadcast QC".to_string(),
        BatchOperation::QualityCheck {
            profile: "broadcast".to_string(),
        },
    );
    qc_job.add_input(InputSpec::new("master.mxf".to_string()));
    qc_job.set_priority(Priority::High);
    let qc_id = engine.submit_job(qc_job).await?;

    // Step 2: Transcode to broadcast format
    let mut transcode_job = BatchJob::new(
        "Broadcast Transcode".to_string(),
        BatchOperation::Transcode {
            preset: "broadcast".to_string(),
        },
    );
    transcode_job.add_dependency(qc_id.clone());
    transcode_job.add_output(OutputSpec::new(
        "broadcast_{date}.mxf".to_string(),
        OutputFormat::Mxf,
    ));
    transcode_job.set_priority(Priority::High);
    let transcode_id = engine.submit_job(transcode_job).await?;

    // Step 3: Generate checksums
    let mut checksum_job = BatchJob::new(
        "Generate Checksums".to_string(),
        BatchOperation::FileOp {
            operation: FileOperation::Checksum {
                algorithm: crate::operations::file_ops::HashAlgorithm::Sha256,
            },
        },
    );
    checksum_job.add_dependency(transcode_id.clone());
    let checksum_id = engine.submit_job(checksum_job).await?;

    tracing::info!(
        "Submitted broadcast workflow: QC {} -> Transcode {} -> Checksum {}",
        qc_id,
        transcode_id,
        checksum_id
    );

    Ok(())
}

/// Example: Web delivery workflow
///
/// This example shows a workflow for web platform delivery.
///
/// # Errors
///
/// Returns an error if job submission fails
pub async fn example_web_delivery(engine: &Arc<BatchEngine>) -> Result<()> {
    let mut job = BatchJob::new(
        "Web Delivery".to_string(),
        BatchOperation::Transcode {
            preset: "web".to_string(),
        },
    );

    job.add_input(InputSpec::new("source.mp4".to_string()));

    // Multiple web formats
    job.add_output(
        OutputSpec::new("web_1080p.mp4".to_string(), OutputFormat::Mp4)
            .with_option("resolution".to_string(), "1920x1080".to_string())
            .with_option("bitrate".to_string(), "5000k".to_string()),
    );

    job.add_output(
        OutputSpec::new("web_720p.mp4".to_string(), OutputFormat::Mp4)
            .with_option("resolution".to_string(), "1280x720".to_string())
            .with_option("bitrate".to_string(), "2500k".to_string()),
    );

    job.add_output(
        OutputSpec::new("web_480p.mp4".to_string(), OutputFormat::Mp4)
            .with_option("resolution".to_string(), "854x480".to_string())
            .with_option("bitrate".to_string(), "1000k".to_string()),
    );

    job.set_priority(Priority::Normal);
    job.add_metadata("platform".to_string(), "web".to_string());

    let job_id = engine.submit_job(job).await?;
    tracing::info!("Submitted web delivery job: {}", job_id);

    Ok(())
}

/// Example: Mobile delivery workflow
///
/// This example shows a workflow optimized for mobile delivery.
///
/// # Errors
///
/// Returns an error if job submission fails
pub async fn example_mobile_delivery(engine: &Arc<BatchEngine>) -> Result<()> {
    let mut job = BatchJob::new(
        "Mobile Delivery".to_string(),
        BatchOperation::Transcode {
            preset: "mobile".to_string(),
        },
    );

    job.add_input(InputSpec::new("source.mp4".to_string()));

    // Mobile-optimized output
    job.add_output(
        OutputSpec::new("mobile.mp4".to_string(), OutputFormat::Mp4)
            .with_option("resolution".to_string(), "640x360".to_string())
            .with_option("bitrate".to_string(), "500k".to_string())
            .with_option("profile".to_string(), "baseline".to_string()),
    );

    job.set_priority(Priority::Normal);
    job.add_metadata("platform".to_string(), "mobile".to_string());

    let job_id = engine.submit_job(job).await?;
    tracing::info!("Submitted mobile delivery job: {}", job_id);

    Ok(())
}

/// Example: Social media workflow
///
/// This example shows workflows for various social media platforms.
///
/// # Errors
///
/// Returns an error if job submission fails
pub async fn example_social_media(engine: &Arc<BatchEngine>) -> Result<()> {
    // YouTube
    let mut youtube_job = BatchJob::new(
        "YouTube Upload".to_string(),
        BatchOperation::Transcode {
            preset: "youtube".to_string(),
        },
    );
    youtube_job.add_input(InputSpec::new("source.mp4".to_string()));
    youtube_job.add_output(
        OutputSpec::new("youtube.mp4".to_string(), OutputFormat::Mp4)
            .with_option("resolution".to_string(), "1920x1080".to_string())
            .with_option("bitrate".to_string(), "8000k".to_string()),
    );
    let youtube_id = engine.submit_job(youtube_job).await?;

    // Instagram
    let mut instagram_job = BatchJob::new(
        "Instagram Post".to_string(),
        BatchOperation::Transcode {
            preset: "instagram".to_string(),
        },
    );
    instagram_job.add_input(InputSpec::new("source.mp4".to_string()));
    instagram_job.add_output(
        OutputSpec::new("instagram.mp4".to_string(), OutputFormat::Mp4)
            .with_option("resolution".to_string(), "1080x1080".to_string())
            .with_option("bitrate".to_string(), "3500k".to_string()),
    );
    let instagram_id = engine.submit_job(instagram_job).await?;

    // TikTok
    let mut tiktok_job = BatchJob::new(
        "TikTok Video".to_string(),
        BatchOperation::Transcode {
            preset: "tiktok".to_string(),
        },
    );
    tiktok_job.add_input(InputSpec::new("source.mp4".to_string()));
    tiktok_job.add_output(
        OutputSpec::new("tiktok.mp4".to_string(), OutputFormat::Mp4)
            .with_option("resolution".to_string(), "1080x1920".to_string())
            .with_option("bitrate".to_string(), "2000k".to_string()),
    );
    let tiktok_id = engine.submit_job(tiktok_job).await?;

    tracing::info!(
        "Submitted social media jobs: YouTube {}, Instagram {}, TikTok {}",
        youtube_id,
        instagram_id,
        tiktok_id
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_engine() -> (Arc<BatchEngine>, TempDir) {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let db_path = temp_dir.path().join("test.db");
        let db_path_str = db_path
            .to_str()
            .expect("path should be valid UTF-8")
            .to_string();
        (
            Arc::new(BatchEngine::new(&db_path_str, 2).expect("failed to create")),
            temp_dir,
        )
    }

    #[tokio::test]
    async fn test_basic_copy_example() {
        let (engine, _dir) = create_test_engine().await;
        let result = example_basic_copy(&engine).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_quality_control_example() {
        let (engine, _dir) = create_test_engine().await;
        let result = example_quality_control(&engine).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_scheduled_job_example() {
        let (engine, _dir) = create_test_engine().await;
        let result = example_scheduled_job(&engine).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_pipeline_example() {
        let (engine, _dir) = create_test_engine().await;
        let result = example_pipeline(&engine).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_batch_analysis_example() {
        let (engine, _dir) = create_test_engine().await;
        let result = example_batch_analysis(&engine).await;
        assert!(result.is_ok());
    }
}
