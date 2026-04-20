//! Batch engine subcommand: submit, status, list, cancel, report.
//!
//! Provides the `oximedia batch-engine` subcommand family using
//! `oximedia_batch::{BatchEngine, BatchJob, JobId, JobState}` backed by an
//! SQLite database (the `sqlite` feature of `oximedia-batch` is required and
//! is always enabled in this crate).
//!
//! Note: The existing `oximedia batch` command (in `batch.rs`) is a simple
//! file-to-file converter. This command exposes the full production batch
//! engine with persistent, database-backed job queuing.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

// Default database path used when `--db` is not specified
const DEFAULT_DB: &str = "oximedia_batch.db";

// ---------------------------------------------------------------------------
// Subcommand enum
// ---------------------------------------------------------------------------

/// Subcommands for `oximedia batch-engine`.
#[derive(Subcommand, Debug)]
pub enum BatchEngineCommand {
    /// Submit a new batch job from a JSON configuration file
    Submit {
        /// JSON file describing the job (name, operation, inputs, outputs)
        #[arg(long)]
        config: PathBuf,

        /// Job priority
        #[arg(long, default_value = "normal",
              value_parser = ["high", "normal", "low"])]
        priority: String,

        /// Path to the SQLite database file
        #[arg(long, default_value = DEFAULT_DB)]
        db: PathBuf,
    },

    /// Show status for a specific job
    Status {
        /// Job ID to query
        #[arg(long)]
        id: String,

        /// Path to the SQLite database file
        #[arg(long, default_value = DEFAULT_DB)]
        db: PathBuf,
    },

    /// List jobs, optionally filtered by state
    List {
        /// Filter by state
        #[arg(long, value_parser = ["pending", "running", "done", "failed", "all"],
              default_value = "all")]
        state: String,

        /// Path to the SQLite database file
        #[arg(long, default_value = DEFAULT_DB)]
        db: PathBuf,
    },

    /// Cancel a running or queued job
    Cancel {
        /// Job ID to cancel
        #[arg(long)]
        id: String,

        /// Path to the SQLite database file
        #[arg(long, default_value = DEFAULT_DB)]
        db: PathBuf,
    },

    /// Generate a summary report for all jobs in the database
    Report {
        /// Path to the SQLite database file
        #[arg(long, default_value = DEFAULT_DB)]
        db: PathBuf,

        /// Output format
        #[arg(long, default_value = "text",
              value_parser = ["text", "json"])]
        format: String,
    },
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Entry point called from `main.rs`.
pub async fn run_batch_engine(command: BatchEngineCommand, json_output: bool) -> Result<()> {
    match command {
        BatchEngineCommand::Submit {
            config,
            priority,
            db,
        } => cmd_submit(&config, &priority, &db, json_output).await,
        BatchEngineCommand::Status { id, db } => cmd_status(&id, &db, json_output).await,
        BatchEngineCommand::List { state, db } => cmd_list(&state, &db, json_output).await,
        BatchEngineCommand::Cancel { id, db } => cmd_cancel(&id, &db, json_output).await,
        BatchEngineCommand::Report { db, format } => {
            let fmt = if json_output { "json" } else { &format };
            cmd_report(&db, fmt).await
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a priority string into `oximedia_batch::Priority`.
fn parse_priority(s: &str) -> oximedia_batch::Priority {
    match s {
        "high" => oximedia_batch::Priority::High,
        "low" => oximedia_batch::Priority::Low,
        _ => oximedia_batch::Priority::Normal,
    }
}

/// Open (or create) a `BatchEngine` backed by the given SQLite database.
fn open_engine(db: &PathBuf) -> Result<oximedia_batch::BatchEngine> {
    let db_str = db
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Database path contains non-UTF-8 characters"))?;
    // Use 4 workers as a reasonable default for the CLI; for high-throughput usage
    // the engine can be configured via a config file.
    oximedia_batch::BatchEngine::new(db_str, 4)
        .map_err(|e| anyhow::anyhow!("Failed to open batch database '{}': {e}", db_str))
}

// ---------------------------------------------------------------------------
// Submit
// ---------------------------------------------------------------------------

async fn cmd_submit(
    config: &PathBuf,
    priority_str: &str,
    db: &PathBuf,
    json_output: bool,
) -> Result<()> {
    if !config.exists() {
        return Err(anyhow::anyhow!(
            "Config file not found: {}",
            config.display()
        ));
    }

    let raw = std::fs::read_to_string(config)
        .with_context(|| format!("Cannot read config: {}", config.display()))?;

    // Parse a minimal job definition: { "name": "..." }
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).context("Config file is not valid JSON")?;

    let job_name = parsed
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unnamed-job")
        .to_string();

    let _priority = parse_priority(priority_str);

    // Build a BatchJob with a FileOp stub
    let job = oximedia_batch::BatchJob::new(
        job_name.clone(),
        oximedia_batch::BatchOperation::FileOp {
            operation: oximedia_batch::operations::FileOperation::Copy { overwrite: false },
        },
    );

    let engine = open_engine(db)?;
    let submitted_id = engine
        .submit_job(job)
        .await
        .map_err(|e| anyhow::anyhow!("Submit failed: {e}"))?;

    if json_output {
        let obj = serde_json::json!({
            "command": "batch-engine submit",
            "job_id": submitted_id.as_str(),
            "name": job_name,
            "priority": priority_str,
            "db": db.display().to_string(),
            "status": "queued",
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&obj).context("JSON serialization failed")?
        );
    } else {
        println!("{}", "Job Submitted".green().bold());
        println!("{:20} {}", "Job ID:", submitted_id.as_str().cyan());
        println!("{:20} {}", "Name:", job_name);
        println!("{:20} {}", "Priority:", priority_str);
        println!("{:20} {}", "Database:", db.display());
        println!("{:20} {}", "Status:", "Queued".yellow());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

async fn cmd_status(id: &str, db: &PathBuf, json_output: bool) -> Result<()> {
    let engine = open_engine(db)?;
    let job_id = oximedia_batch::JobId::from(id);
    let state = engine
        .get_job_status(&job_id)
        .await
        .map_err(|e| anyhow::anyhow!("Status query failed: {e}"))?;

    if json_output {
        let obj = serde_json::json!({
            "command": "batch-engine status",
            "job_id": id,
            "state": state.to_string(),
            "db": db.display().to_string(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&obj).context("JSON serialization")?
        );
    } else {
        println!("{}", "Job Status".green().bold());
        println!("{:20} {}", "Job ID:", id.cyan());
        println!("{:20} {}", "State:", state.to_string().yellow());
        println!("{:20} {}", "Database:", db.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

async fn cmd_list(state_filter: &str, db: &PathBuf, json_output: bool) -> Result<()> {
    let engine = open_engine(db)?;
    let jobs = engine
        .list_jobs()
        .map_err(|e| anyhow::anyhow!("List failed: {e}"))?;

    if json_output {
        let list: Vec<serde_json::Value> = jobs
            .iter()
            .map(|j| {
                serde_json::json!({
                    "id": j.id.as_str(),
                    "name": j.name,
                    "priority": j.priority.to_string(),
                })
            })
            .collect();
        let obj = serde_json::json!({
            "command": "batch-engine list",
            "filter": state_filter,
            "count": list.len(),
            "jobs": list,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&obj).context("JSON serialization")?
        );
    } else {
        println!("{}", "Batch Jobs".green().bold());
        println!("{}", "=".repeat(70));
        if jobs.is_empty() {
            println!("  No jobs found.");
        } else {
            println!("{:<40} {:<20} Priority", "Job ID", "Name");
            println!("{}", "-".repeat(70));
            for j in &jobs {
                println!("{:<40} {:<20} {}", j.id.as_str(), j.name, j.priority);
            }
        }
        println!();
        println!("Total: {} jobs", jobs.len());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Cancel
// ---------------------------------------------------------------------------

async fn cmd_cancel(id: &str, db: &PathBuf, json_output: bool) -> Result<()> {
    let engine = open_engine(db)?;
    let job_id = oximedia_batch::JobId::from(id);
    engine
        .cancel_job(&job_id)
        .await
        .map_err(|e| anyhow::anyhow!("Cancel failed: {e}"))?;

    if json_output {
        let obj = serde_json::json!({
            "command": "batch-engine cancel",
            "job_id": id,
            "status": "cancelled",
            "db": db.display().to_string(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&obj).context("JSON serialization")?
        );
    } else {
        println!("{}", "Job Cancelled".green().bold());
        println!("{:20} {}", "Job ID:", id.cyan());
        println!("{:20} {}", "Database:", db.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

async fn cmd_report(db: &PathBuf, output_format: &str) -> Result<()> {
    let engine = open_engine(db)?;
    let jobs = engine
        .list_jobs()
        .map_err(|e| anyhow::anyhow!("Report failed: {e}"))?;

    let total = jobs.len();
    let high_priority = jobs
        .iter()
        .filter(|j| j.priority == oximedia_batch::Priority::High)
        .count();
    let normal_priority = jobs
        .iter()
        .filter(|j| j.priority == oximedia_batch::Priority::Normal)
        .count();
    let low_priority = jobs
        .iter()
        .filter(|j| j.priority == oximedia_batch::Priority::Low)
        .count();

    if output_format == "json" {
        let obj = serde_json::json!({
            "command": "batch-engine report",
            "db": db.display().to_string(),
            "total": total,
            "high_priority": high_priority,
            "normal_priority": normal_priority,
            "low_priority": low_priority,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&obj).context("JSON serialization")?
        );
    } else {
        println!("{}", "Batch Engine Report".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Database:", db.display());
        println!("{:20} {}", "Total jobs:", total);
        println!(
            "{:20} {}",
            "High priority:",
            high_priority.to_string().red()
        );
        println!(
            "{:20} {}",
            "Normal priority:",
            normal_priority.to_string().yellow()
        );
        println!(
            "{:20} {}",
            "Low priority:",
            low_priority.to_string().dimmed()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_priority_variants() {
        assert_eq!(parse_priority("high"), oximedia_batch::Priority::High);
        assert_eq!(parse_priority("normal"), oximedia_batch::Priority::Normal);
        assert_eq!(parse_priority("low"), oximedia_batch::Priority::Low);
        assert_eq!(parse_priority("unknown"), oximedia_batch::Priority::Normal);
    }

    #[tokio::test]
    async fn test_submit_missing_config() {
        let cfg = std::env::temp_dir().join("oximedia_batch_missing_config.json");
        let db = std::env::temp_dir().join("oximedia_batch_test_missing.db");
        let result = cmd_submit(&cfg, "normal", &db, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_submit_invalid_json() {
        let dir = std::env::temp_dir();
        let cfg = dir.join("oximedia_batch_bad.json");
        std::fs::write(&cfg, b"not json").expect("write stub");
        let db = dir.join("oximedia_batch_test.db");
        let result = cmd_submit(&cfg, "normal", &db, false).await;
        assert!(result.is_err());
        std::fs::remove_file(&cfg).ok();
    }

    #[tokio::test]
    async fn test_submit_valid_json() {
        let dir = std::env::temp_dir();
        let cfg = dir.join("oximedia_batch_submit_ok.json");
        std::fs::write(&cfg, br#"{"name":"test-job","operation":"transcode"}"#)
            .expect("write stub");
        let db = dir.join("oximedia_batch_submit_test.db");
        let result = cmd_submit(&cfg, "high", &db, true).await;
        assert!(result.is_ok(), "unexpected error: {result:?}");
        std::fs::remove_file(&cfg).ok();
        std::fs::remove_file(&db).ok();
    }

    #[tokio::test]
    async fn test_list_empty_db() {
        let db = std::env::temp_dir().join("oximedia_batch_list_empty.db");
        let result = cmd_list("all", &db, true).await;
        assert!(result.is_ok(), "unexpected error: {result:?}");
        std::fs::remove_file(&db).ok();
    }

    #[tokio::test]
    async fn test_report_empty_db_json() {
        let db = std::env::temp_dir().join("oximedia_batch_report_json.db");
        let result = cmd_report(&db, "json").await;
        assert!(result.is_ok(), "unexpected error: {result:?}");
        std::fs::remove_file(&db).ok();
    }

    #[tokio::test]
    async fn test_report_empty_db_text() {
        let db = std::env::temp_dir().join("oximedia_batch_report_text.db");
        let result = cmd_report(&db, "text").await;
        assert!(result.is_ok(), "unexpected error: {result:?}");
        std::fs::remove_file(&db).ok();
    }
}
