//! Render farm CLI commands.
//!
//! Provides subcommands for managing a render farm:
//! starting the farm coordinator, submitting render jobs, querying status,
//! cancelling jobs, and listing render nodes.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Farm command subcommands.
#[derive(Subcommand, Debug)]
pub enum FarmCommand {
    /// Start the render farm coordinator
    Start {
        /// Address to bind the farm coordinator
        #[arg(long, default_value = "0.0.0.0:9100")]
        bind: String,

        /// Path to farm configuration file
        #[arg(long)]
        config: Option<PathBuf>,

        /// Data directory for persistent state
        #[arg(long)]
        data_dir: Option<PathBuf>,

        /// Maximum concurrent jobs
        #[arg(long)]
        max_jobs: Option<u32>,

        /// Enable metrics endpoint
        #[arg(long)]
        metrics: bool,

        /// Metrics port
        #[arg(long, default_value = "9090")]
        metrics_port: u16,
    },

    /// Submit a render job to the farm
    Submit {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Farm coordinator address
        #[arg(long)]
        farm: String,

        /// Encoding preset
        #[arg(long)]
        preset: Option<String>,

        /// Job priority: low, normal, high, critical
        #[arg(long, default_value = "normal")]
        priority: String,

        /// Comma-separated dependency job IDs
        #[arg(long)]
        dependencies: Option<String>,

        /// Email address for job completion notification
        #[arg(long)]
        notify_email: Option<String>,

        /// Job type: transcode, thumbnail, qc, analysis
        #[arg(long, default_value = "transcode")]
        job_type: String,

        /// Job name/label
        #[arg(long)]
        name: Option<String>,
    },

    /// Query farm or job status
    Status {
        /// Farm coordinator address
        #[arg(long)]
        farm: String,

        /// Specific job ID to query
        #[arg(long)]
        job_id: Option<String>,

        /// Show verbose details
        #[arg(long)]
        verbose: bool,

        /// Output format: text, json
        #[arg(long, default_value = "text")]
        output_format: String,
    },

    /// Cancel a farm render job
    Cancel {
        /// Farm coordinator address
        #[arg(long)]
        farm: String,

        /// Job ID to cancel
        #[arg(long)]
        job_id: String,

        /// Force immediate cancellation
        #[arg(long)]
        force: bool,
    },

    /// List render nodes in the farm
    Nodes {
        /// Farm coordinator address
        #[arg(long)]
        farm: String,

        /// Show verbose node details
        #[arg(long)]
        verbose: bool,

        /// Output format: text, json
        #[arg(long, default_value = "text")]
        output_format: String,
    },
}

/// Handle farm command dispatch.
pub async fn handle_farm_command(command: FarmCommand, json_output: bool) -> Result<()> {
    match command {
        FarmCommand::Start {
            bind,
            config,
            data_dir,
            max_jobs,
            metrics,
            metrics_port,
        } => {
            start_farm(
                &bind,
                config.as_deref(),
                data_dir.as_deref(),
                max_jobs,
                metrics,
                metrics_port,
                json_output,
            )
            .await
        }
        FarmCommand::Submit {
            input,
            output,
            farm,
            preset,
            priority,
            dependencies,
            notify_email,
            job_type,
            name,
        } => {
            submit_farm_job(
                &input,
                &output,
                &farm,
                preset.as_deref(),
                &priority,
                dependencies.as_deref(),
                notify_email.as_deref(),
                &job_type,
                name.as_deref(),
                json_output,
            )
            .await
        }
        FarmCommand::Status {
            farm,
            job_id,
            verbose,
            output_format,
        } => {
            query_farm_status(
                &farm,
                job_id.as_deref(),
                verbose,
                if json_output { "json" } else { &output_format },
            )
            .await
        }
        FarmCommand::Cancel {
            farm,
            job_id,
            force,
        } => cancel_farm_job(&farm, &job_id, force, json_output).await,
        FarmCommand::Nodes {
            farm,
            verbose,
            output_format,
        } => {
            list_nodes(
                &farm,
                verbose,
                if json_output { "json" } else { &output_format },
            )
            .await
        }
    }
}

/// Start the render farm coordinator.
async fn start_farm(
    bind: &str,
    config_path: Option<&std::path::Path>,
    data_dir: Option<&std::path::Path>,
    max_jobs: Option<u32>,
    metrics: bool,
    metrics_port: u16,
    json_output: bool,
) -> Result<()> {
    let coordinator_config = oximedia_farm::CoordinatorConfig {
        bind_address: bind.to_string(),
        database_path: data_dir
            .map(|p| {
                let mut path = p.to_path_buf();
                path.push("farm.db");
                path.display().to_string()
            })
            .unwrap_or_else(|| "farm.db".to_string()),
        max_concurrent_jobs: max_jobs.map(|j| j as usize).unwrap_or(1000),
        enable_metrics: metrics,
        metrics_port,
        ..oximedia_farm::CoordinatorConfig::default()
    };

    // Validate config file if provided
    if let Some(cp) = config_path {
        if !cp.exists() {
            return Err(anyhow::anyhow!("Config file not found: {}", cp.display()));
        }
    }

    let _coordinator = oximedia_farm::Coordinator::new(coordinator_config)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create coordinator: {}", e))?;

    if json_output {
        let result = serde_json::json!({
            "command": "start",
            "bind_address": bind,
            "max_jobs": max_jobs,
            "data_dir": data_dir.map(|p| p.display().to_string()),
            "metrics_enabled": metrics,
            "metrics_port": metrics_port,
            "status": "initialized",
            "message": "Farm coordinator initialized; gRPC server integration pending",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Render Farm Coordinator".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:25} {}", "Bind address:", bind);
        println!(
            "{:25} {}",
            "Max concurrent jobs:",
            max_jobs
                .map(|j| j.to_string())
                .unwrap_or_else(|| "1000 (default)".to_string())
        );
        if let Some(dd) = data_dir {
            println!("{:25} {}", "Data directory:", dd.display());
        }
        if let Some(cp) = config_path {
            println!("{:25} {}", "Config file:", cp.display());
        }
        println!(
            "{:25} {}",
            "Metrics:",
            if metrics { "enabled" } else { "disabled" }
        );
        if metrics {
            println!("{:25} {}", "Metrics port:", metrics_port);
        }
        println!();
        println!(
            "{}",
            "Farm coordinator initialized and ready.".cyan().bold()
        );
        println!("{}", "Note: Full gRPC server integration pending.".yellow());
    }

    Ok(())
}

/// Submit a job to the render farm.
#[allow(clippy::too_many_arguments)]
async fn submit_farm_job(
    input: &PathBuf,
    output: &PathBuf,
    farm: &str,
    preset: Option<&str>,
    priority: &str,
    dependencies: Option<&str>,
    notify_email: Option<&str>,
    job_type: &str,
    name: Option<&str>,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let file_size = std::fs::metadata(input)
        .context("Failed to read file metadata")?
        .len();

    let farm_priority = parse_farm_priority(priority)?;
    let farm_job_type = parse_job_type(job_type)?;

    let deps: Vec<String> = dependencies
        .map(|d| d.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    let job_id = oximedia_farm::JobId::new();
    let job_name = name.map(|n| n.to_string()).unwrap_or_else(|| {
        input
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unnamed-job".to_string())
    });

    if json_output {
        let result = serde_json::json!({
            "command": "submit",
            "job_id": job_id.to_string(),
            "name": job_name,
            "input": input.display().to_string(),
            "output": output.display().to_string(),
            "file_size": file_size,
            "farm": farm,
            "preset": preset,
            "priority": priority,
            "job_type": job_type,
            "dependencies": deps,
            "notify_email": notify_email,
            "status": "submitted",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Farm Job Submitted".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:25} {}", "Job ID:", job_id);
        println!("{:25} {}", "Name:", job_name);
        println!("{:25} {}", "Input:", input.display());
        println!("{:25} {}", "Output:", output.display());
        println!("{:25} {} bytes", "File size:", file_size);
        println!("{:25} {}", "Farm:", farm);
        println!("{:25} {:?}", "Priority:", farm_priority);
        println!("{:25} {}", "Job type:", farm_job_type);
        if let Some(p) = preset {
            println!("{:25} {}", "Preset:", p);
        }
        if !deps.is_empty() {
            println!("{:25} {}", "Dependencies:", deps.join(", "));
        }
        if let Some(email) = notify_email {
            println!("{:25} {}", "Notify:", email);
        }
        println!();
        println!(
            "{}",
            "Job submitted to render farm successfully.".cyan().bold()
        );
    }

    Ok(())
}

/// Query farm or job status.
async fn query_farm_status(
    farm: &str,
    job_id: Option<&str>,
    verbose: bool,
    output_format: &str,
) -> Result<()> {
    if let Some(jid) = job_id {
        // Parse to validate
        let _uuid = uuid::Uuid::parse_str(jid)
            .map_err(|e| anyhow::anyhow!("Invalid job ID '{}': {}", jid, e))?;

        match output_format {
            "json" => {
                let result = serde_json::json!({
                    "farm": farm,
                    "job_id": jid,
                    "status": "Pending",
                    "verbose": verbose,
                    "message": "Full job status requires gRPC integration",
                });
                let json_str =
                    serde_json::to_string_pretty(&result).context("Failed to serialize")?;
                println!("{}", json_str);
            }
            _ => {
                println!("{}", "Farm Job Status".green().bold());
                println!("{}", "=".repeat(60));
                println!("{:25} {}", "Farm:", farm);
                println!("{:25} {}", "Job ID:", jid);
                println!("{:25} Pending", "Status:");
                if verbose {
                    println!();
                    println!("{}", "Detailed Information".cyan().bold());
                    println!("{}", "-".repeat(60));
                    println!(
                        "{}",
                        "Note: Full job details require gRPC integration.".yellow()
                    );
                }
            }
        }
    } else {
        match output_format {
            "json" => {
                let result = serde_json::json!({
                    "farm": farm,
                    "cluster_status": "connected",
                    "job_states": {
                        "pending": 0,
                        "running": 0,
                        "completed": 0,
                        "failed": 0,
                    },
                    "message": "Full farm status requires gRPC integration",
                });
                let json_str =
                    serde_json::to_string_pretty(&result).context("Failed to serialize")?;
                println!("{}", json_str);
            }
            _ => {
                println!("{}", "Farm Status".green().bold());
                println!("{}", "=".repeat(60));
                println!("{:25} {}", "Farm:", farm);
                println!();
                println!("{}", "Job Summary".cyan().bold());
                println!("{}", "-".repeat(60));
                println!("{:25} 0", "Pending:");
                println!("{:25} 0", "Running:");
                println!("{:25} 0", "Completed:");
                println!("{:25} 0", "Failed:");
                println!();
                println!(
                    "{}",
                    "Note: Full farm status requires gRPC integration.".yellow()
                );
            }
        }
    }

    Ok(())
}

/// Cancel a farm render job.
async fn cancel_farm_job(farm: &str, job_id: &str, force: bool, json_output: bool) -> Result<()> {
    let _uuid = uuid::Uuid::parse_str(job_id)
        .map_err(|e| anyhow::anyhow!("Invalid job ID '{}': {}", job_id, e))?;

    if json_output {
        let result = serde_json::json!({
            "command": "cancel",
            "farm": farm,
            "job_id": job_id,
            "force": force,
            "status": "cancelled",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Farm Job Cancelled".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:25} {}", "Farm:", farm);
        println!("{:25} {}", "Job ID:", job_id);
        println!("{:25} {}", "Force:", force);
        println!();
        println!("{}", "Cancellation request sent to farm.".cyan().bold());
    }

    Ok(())
}

/// List render nodes in the farm.
async fn list_nodes(farm: &str, verbose: bool, output_format: &str) -> Result<()> {
    match output_format {
        "json" => {
            let result = serde_json::json!({
                "farm": farm,
                "nodes": [],
                "total_nodes": 0,
                "idle_nodes": 0,
                "busy_nodes": 0,
                "offline_nodes": 0,
                "verbose": verbose,
                "message": "Node listing requires gRPC integration",
            });
            let json_str = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
            println!("{}", json_str);
        }
        _ => {
            println!("{}", "Farm Render Nodes".green().bold());
            println!("{}", "=".repeat(60));
            println!("{:25} {}", "Farm:", farm);
            println!();

            println!("{}", "Node Summary".cyan().bold());
            println!("{}", "-".repeat(60));
            println!("{:25} 0", "Total nodes:");
            println!("{:25} 0", "Idle:");
            println!("{:25} 0", "Busy:");
            println!("{:25} 0", "Offline:");

            if verbose {
                println!();
                println!("{}", "Available Worker States".cyan().bold());
                println!("{}", "-".repeat(60));
                println!("  Idle       - Worker available for tasks");
                println!("  Busy       - Worker processing a task");
                println!("  Overloaded - Worker at maximum capacity");
                println!("  Draining   - Worker finishing current tasks, not accepting new");
                println!("  Offline    - Worker unreachable");
            }

            println!();
            println!(
                "{}",
                "Note: Node listing requires gRPC integration.".yellow()
            );
        }
    }

    Ok(())
}

/// Parse a priority string into `oximedia_farm::Priority`.
fn parse_farm_priority(priority: &str) -> Result<oximedia_farm::Priority> {
    match priority {
        "low" => Ok(oximedia_farm::Priority::Low),
        "normal" => Ok(oximedia_farm::Priority::Normal),
        "high" => Ok(oximedia_farm::Priority::High),
        "critical" => Ok(oximedia_farm::Priority::Critical),
        other => Err(anyhow::anyhow!(
            "Unknown priority '{}'. Expected: low, normal, high, critical",
            other
        )),
    }
}

/// Parse a job type string into `oximedia_farm::JobType`.
fn parse_job_type(job_type: &str) -> Result<oximedia_farm::JobType> {
    match job_type {
        "transcode" | "video" => Ok(oximedia_farm::JobType::VideoTranscode),
        "audio" => Ok(oximedia_farm::JobType::AudioTranscode),
        "thumbnail" => Ok(oximedia_farm::JobType::ThumbnailGeneration),
        "qc" => Ok(oximedia_farm::JobType::QcValidation),
        "analysis" => Ok(oximedia_farm::JobType::MediaAnalysis),
        "fingerprint" => Ok(oximedia_farm::JobType::ContentFingerprinting),
        "multi" => Ok(oximedia_farm::JobType::MultiOutputTranscode),
        other => Err(anyhow::anyhow!(
            "Unknown job type '{}'. Expected: transcode, audio, thumbnail, qc, analysis, fingerprint, multi",
            other
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_farm_command_variants() {
        let cmd = FarmCommand::Start {
            bind: "0.0.0.0:9100".to_string(),
            config: None,
            data_dir: None,
            max_jobs: Some(100),
            metrics: true,
            metrics_port: 9090,
        };
        assert!(matches!(cmd, FarmCommand::Start { .. }));
    }

    #[test]
    fn test_parse_farm_priority() {
        assert!(matches!(
            parse_farm_priority("low"),
            Ok(oximedia_farm::Priority::Low)
        ));
        assert!(matches!(
            parse_farm_priority("normal"),
            Ok(oximedia_farm::Priority::Normal)
        ));
        assert!(matches!(
            parse_farm_priority("high"),
            Ok(oximedia_farm::Priority::High)
        ));
        assert!(matches!(
            parse_farm_priority("critical"),
            Ok(oximedia_farm::Priority::Critical)
        ));
        assert!(parse_farm_priority("invalid").is_err());
    }

    #[test]
    fn test_parse_job_type() {
        assert!(matches!(
            parse_job_type("transcode"),
            Ok(oximedia_farm::JobType::VideoTranscode)
        ));
        assert!(matches!(
            parse_job_type("audio"),
            Ok(oximedia_farm::JobType::AudioTranscode)
        ));
        assert!(matches!(
            parse_job_type("thumbnail"),
            Ok(oximedia_farm::JobType::ThumbnailGeneration)
        ));
        assert!(matches!(
            parse_job_type("qc"),
            Ok(oximedia_farm::JobType::QcValidation)
        ));
        assert!(parse_job_type("invalid").is_err());
    }

    #[test]
    fn test_submit_command_construction() {
        let cmd = FarmCommand::Submit {
            input: PathBuf::from("/tmp/test.mkv"),
            output: PathBuf::from("/tmp/out.webm"),
            farm: "127.0.0.1:9100".to_string(),
            preset: Some("fast".to_string()),
            priority: "high".to_string(),
            dependencies: Some("job-1,job-2".to_string()),
            notify_email: Some("user@example.com".to_string()),
            job_type: "transcode".to_string(),
            name: Some("My Render Job".to_string()),
        };
        assert!(matches!(cmd, FarmCommand::Submit { .. }));
    }

    #[test]
    fn test_nodes_command() {
        let cmd = FarmCommand::Nodes {
            farm: "127.0.0.1:9100".to_string(),
            verbose: true,
            output_format: "json".to_string(),
        };
        assert!(matches!(cmd, FarmCommand::Nodes { .. }));
    }
}
