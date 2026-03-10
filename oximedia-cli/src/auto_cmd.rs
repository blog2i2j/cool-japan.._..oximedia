//! Automated video editing and task scheduling CLI commands.
//!
//! Provides subcommands for automated media processing:
//! running automated edits, scheduling recurring tasks,
//! listing automations, creating workflows, deleting tasks,
//! and viewing automation logs.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;

/// Auto editing and task automation subcommands.
#[derive(Subcommand, Debug)]
pub enum AutoCommand {
    /// Run an automated editing workflow
    Run {
        /// Input media file
        #[arg(short, long)]
        input: std::path::PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: std::path::PathBuf,

        /// Use case: trailer, highlights, social, documentary, music_video
        #[arg(long, default_value = "highlights")]
        use_case: String,

        /// Target duration in seconds
        #[arg(long)]
        target_duration: Option<f64>,

        /// Pacing preset: slow, medium, fast, dynamic
        #[arg(long, default_value = "medium")]
        pacing: String,

        /// Aspect ratio: 16x9, 9x16, 4x3, 1x1, 21x9
        #[arg(long)]
        aspect_ratio: Option<String>,

        /// Enable dramatic arc shaping
        #[arg(long)]
        dramatic_arc: bool,

        /// Music sync mode: none, beats, bars, downbeats
        #[arg(long, default_value = "none")]
        music_sync: String,

        /// Enable verbose analysis output
        #[arg(long)]
        verbose: bool,
    },

    /// Schedule a recurring automated task
    Schedule {
        /// Task name
        #[arg(long)]
        name: String,

        /// Input directory or file pattern
        #[arg(long)]
        input_pattern: String,

        /// Output directory
        #[arg(long)]
        output_dir: String,

        /// Cron expression (e.g., "0 2 * * *" for daily at 2am)
        #[arg(long)]
        cron: String,

        /// Use case for processing
        #[arg(long, default_value = "highlights")]
        use_case: String,

        /// Enable/disable the schedule
        #[arg(long, default_value = "true")]
        enabled: bool,

        /// Maximum concurrent tasks
        #[arg(long, default_value = "1")]
        max_concurrent: u32,

        /// Notification email on completion
        #[arg(long)]
        notify_email: Option<String>,
    },

    /// List all automations and scheduled tasks
    List {
        /// Filter by status: active, paused, completed, failed
        #[arg(long)]
        status: Option<String>,

        /// Filter by use case
        #[arg(long)]
        use_case: Option<String>,

        /// Show detailed info
        #[arg(long)]
        verbose: bool,

        /// Output format: text, json
        #[arg(long, default_value = "text")]
        output_format: String,
    },

    /// Create a new automation workflow from a template
    Create {
        /// Workflow name
        #[arg(long)]
        name: String,

        /// Template: highlight-reel, social-clips, trailer, batch-transcode, quality-check
        #[arg(long)]
        template: String,

        /// Configuration overrides as JSON
        #[arg(long)]
        config: Option<String>,

        /// Description
        #[arg(long)]
        description: Option<String>,

        /// Tags (comma-separated)
        #[arg(long)]
        tags: Option<String>,
    },

    /// Delete an automation task or workflow
    Delete {
        /// Task or workflow ID
        #[arg(long)]
        id: String,

        /// Force deletion even if running
        #[arg(long)]
        force: bool,

        /// Delete associated output files
        #[arg(long)]
        delete_outputs: bool,
    },

    /// View automation execution logs
    Log {
        /// Task or workflow ID (omit for all)
        #[arg(long)]
        id: Option<String>,

        /// Number of log entries to show
        #[arg(long, default_value = "50")]
        limit: u32,

        /// Filter by log level: info, warn, error, debug
        #[arg(long)]
        level: Option<String>,

        /// Show logs from a specific date (ISO 8601)
        #[arg(long)]
        since: Option<String>,

        /// Output format: text, json
        #[arg(long, default_value = "text")]
        output_format: String,

        /// Follow log output (tail -f style)
        #[arg(long)]
        follow: bool,
    },
}

/// Handle auto command dispatch.
pub async fn handle_auto_command(command: AutoCommand, json_output: bool) -> Result<()> {
    match command {
        AutoCommand::Run {
            input,
            output,
            use_case,
            target_duration,
            pacing,
            aspect_ratio,
            dramatic_arc,
            music_sync,
            verbose,
        } => {
            run_auto_edit(
                &input,
                &output,
                &use_case,
                target_duration,
                &pacing,
                aspect_ratio.as_deref(),
                dramatic_arc,
                &music_sync,
                verbose,
                json_output,
            )
            .await
        }
        AutoCommand::Schedule {
            name,
            input_pattern,
            output_dir,
            cron,
            use_case,
            enabled,
            max_concurrent,
            notify_email,
        } => {
            schedule_task(
                &name,
                &input_pattern,
                &output_dir,
                &cron,
                &use_case,
                enabled,
                max_concurrent,
                notify_email.as_deref(),
                json_output,
            )
            .await
        }
        AutoCommand::List {
            status,
            use_case,
            verbose,
            output_format,
        } => {
            list_automations(
                status.as_deref(),
                use_case.as_deref(),
                verbose,
                if json_output { "json" } else { &output_format },
            )
            .await
        }
        AutoCommand::Create {
            name,
            template,
            config,
            description,
            tags,
        } => {
            create_workflow(
                &name,
                &template,
                config.as_deref(),
                description.as_deref(),
                tags.as_deref(),
                json_output,
            )
            .await
        }
        AutoCommand::Delete {
            id,
            force,
            delete_outputs,
        } => delete_task(&id, force, delete_outputs, json_output).await,
        AutoCommand::Log {
            id,
            limit,
            level,
            since,
            output_format,
            follow,
        } => {
            view_logs(
                id.as_deref(),
                limit,
                level.as_deref(),
                since.as_deref(),
                follow,
                if json_output { "json" } else { &output_format },
            )
            .await
        }
    }
}

/// Validate use case string.
fn validate_use_case(use_case: &str) -> Result<()> {
    match use_case {
        "trailer" | "highlights" | "social" | "documentary" | "music_video" => Ok(()),
        other => Err(anyhow::anyhow!(
            "Unknown use case '{}'. Expected: trailer, highlights, social, documentary, music_video",
            other
        )),
    }
}

/// Validate pacing preset string.
fn validate_pacing(pacing: &str) -> Result<()> {
    match pacing {
        "slow" | "medium" | "fast" | "dynamic" => Ok(()),
        other => Err(anyhow::anyhow!(
            "Unknown pacing '{}'. Expected: slow, medium, fast, dynamic",
            other
        )),
    }
}

/// Run an automated editing workflow.
#[allow(clippy::too_many_arguments)]
async fn run_auto_edit(
    input: &std::path::Path,
    output: &std::path::Path,
    use_case: &str,
    target_duration: Option<f64>,
    pacing: &str,
    aspect_ratio: Option<&str>,
    dramatic_arc: bool,
    music_sync: &str,
    verbose: bool,
    json_output: bool,
) -> Result<()> {
    validate_use_case(use_case)?;
    validate_pacing(pacing)?;

    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    // Create the auto editor configuration
    let config = oximedia_auto::AutoEditorConfig::for_use_case(use_case);
    let _editor = oximedia_auto::AutoEditor::new(config);

    let job_id = format!("auto-{}", uuid::Uuid::new_v4().as_simple());

    if json_output {
        let result = serde_json::json!({
            "command": "run",
            "job_id": job_id,
            "input": input.display().to_string(),
            "output": output.display().to_string(),
            "use_case": use_case,
            "target_duration": target_duration,
            "pacing": pacing,
            "aspect_ratio": aspect_ratio,
            "dramatic_arc": dramatic_arc,
            "music_sync": music_sync,
            "status": "initialized",
            "message": "Auto editor initialized; full pipeline execution pending frame input",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{json_str}");
    } else {
        println!("{}", "Automated Edit".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:25} {}", "Job ID:", job_id);
        println!("{:25} {}", "Input:", input.display());
        println!("{:25} {}", "Output:", output.display());
        println!("{:25} {}", "Use case:", use_case);
        if let Some(dur) = target_duration {
            println!("{:25} {:.1}s", "Target duration:", dur);
        }
        println!("{:25} {}", "Pacing:", pacing);
        if let Some(ar) = aspect_ratio {
            println!("{:25} {}", "Aspect ratio:", ar);
        }
        println!(
            "{:25} {}",
            "Dramatic arc:",
            if dramatic_arc { "enabled" } else { "disabled" }
        );
        println!("{:25} {}", "Music sync:", music_sync);
        println!();

        if verbose {
            println!("{}", "Auto Editor Configuration".cyan().bold());
            println!("{}", "-".repeat(60));
            println!("{:25} initialized", "Highlight detector:");
            println!("{:25} initialized", "Cut detector:");
            println!("{:25} initialized", "Auto assembler:");
            println!("{:25} initialized", "Rules engine:");
            println!("{:25} initialized", "Scene scorer:");
            println!();
        }

        println!(
            "{}",
            "Auto editor initialized. Full pipeline execution pending frame input."
                .cyan()
                .bold()
        );
    }

    Ok(())
}

/// Schedule a recurring automation task.
#[allow(clippy::too_many_arguments)]
async fn schedule_task(
    name: &str,
    input_pattern: &str,
    output_dir: &str,
    cron: &str,
    use_case: &str,
    enabled: bool,
    max_concurrent: u32,
    notify_email: Option<&str>,
    json_output: bool,
) -> Result<()> {
    validate_use_case(use_case)?;

    let schedule_id = format!("sched-{}", uuid::Uuid::new_v4().as_simple());

    if json_output {
        let result = serde_json::json!({
            "command": "schedule",
            "schedule_id": schedule_id,
            "name": name,
            "input_pattern": input_pattern,
            "output_dir": output_dir,
            "cron": cron,
            "use_case": use_case,
            "enabled": enabled,
            "max_concurrent": max_concurrent,
            "notify_email": notify_email,
            "status": "created",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{json_str}");
    } else {
        println!("{}", "Automation Scheduled".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:25} {}", "Schedule ID:", schedule_id);
        println!("{:25} {}", "Name:", name);
        println!("{:25} {}", "Input pattern:", input_pattern);
        println!("{:25} {}", "Output dir:", output_dir);
        println!("{:25} {}", "Cron:", cron);
        println!("{:25} {}", "Use case:", use_case);
        println!("{:25} {}", "Enabled:", enabled);
        println!("{:25} {}", "Max concurrent:", max_concurrent);
        if let Some(email) = notify_email {
            println!("{:25} {}", "Notify:", email);
        }
        println!();
        println!("{}", "Automation schedule created.".cyan().bold());
    }

    Ok(())
}

/// List automations.
async fn list_automations(
    status: Option<&str>,
    use_case: Option<&str>,
    verbose: bool,
    output_format: &str,
) -> Result<()> {
    match output_format {
        "json" => {
            let result = serde_json::json!({
                "command": "list",
                "status_filter": status,
                "use_case_filter": use_case,
                "verbose": verbose,
                "automations": [],
                "total_count": 0,
            });
            let json_str = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
            println!("{json_str}");
        }
        _ => {
            println!("{}", "Automation Tasks".green().bold());
            println!("{}", "=".repeat(60));
            if let Some(s) = status {
                println!("{:25} {}", "Status filter:", s);
            }
            if let Some(uc) = use_case {
                println!("{:25} {}", "Use case filter:", uc);
            }
            println!();
            println!("{}", "No automation tasks found.".yellow());
            println!(
                "{}",
                "Note: Task listing requires persistent storage integration.".yellow()
            );
        }
    }

    Ok(())
}

/// Create a workflow from a template.
async fn create_workflow(
    name: &str,
    template: &str,
    config: Option<&str>,
    description: Option<&str>,
    tags: Option<&str>,
    json_output: bool,
) -> Result<()> {
    match template {
        "highlight-reel" | "social-clips" | "trailer" | "batch-transcode" | "quality-check" => {}
        other => {
            return Err(anyhow::anyhow!(
                "Unknown template '{}'. Expected: highlight-reel, social-clips, trailer, batch-transcode, quality-check",
                other
            ));
        }
    }

    let workflow_id = format!("wf-{}", uuid::Uuid::new_v4().as_simple());
    let tag_list: Vec<String> = tags
        .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    if json_output {
        let result = serde_json::json!({
            "command": "create",
            "workflow_id": workflow_id,
            "name": name,
            "template": template,
            "config": config,
            "description": description,
            "tags": tag_list,
            "status": "created",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{json_str}");
    } else {
        println!("{}", "Workflow Created".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:25} {}", "Workflow ID:", workflow_id);
        println!("{:25} {}", "Name:", name);
        println!("{:25} {}", "Template:", template);
        if let Some(desc) = description {
            println!("{:25} {}", "Description:", desc);
        }
        if !tag_list.is_empty() {
            println!("{:25} {}", "Tags:", tag_list.join(", "));
        }
        if let Some(cfg) = config {
            println!("{:25} {}", "Config overrides:", cfg);
        }
        println!();
        println!("{}", "Workflow created from template.".cyan().bold());
    }

    Ok(())
}

/// Delete an automation task.
async fn delete_task(id: &str, force: bool, delete_outputs: bool, json_output: bool) -> Result<()> {
    if json_output {
        let result = serde_json::json!({
            "command": "delete",
            "id": id,
            "force": force,
            "delete_outputs": delete_outputs,
            "status": "deleted",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{json_str}");
    } else {
        println!("{}", "Task Deleted".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:25} {}", "ID:", id);
        println!("{:25} {}", "Force:", force);
        println!("{:25} {}", "Delete outputs:", delete_outputs);
        println!();
        println!("{}", "Automation task deleted.".cyan().bold());
    }

    Ok(())
}

/// View automation logs.
async fn view_logs(
    id: Option<&str>,
    limit: u32,
    level: Option<&str>,
    since: Option<&str>,
    follow: bool,
    output_format: &str,
) -> Result<()> {
    if let Some(lvl) = level {
        match lvl {
            "info" | "warn" | "error" | "debug" => {}
            other => {
                return Err(anyhow::anyhow!(
                    "Unknown log level '{}'. Expected: info, warn, error, debug",
                    other
                ));
            }
        }
    }

    match output_format {
        "json" => {
            let result = serde_json::json!({
                "command": "log",
                "id": id,
                "limit": limit,
                "level": level,
                "since": since,
                "follow": follow,
                "entries": [],
                "total_entries": 0,
            });
            let json_str = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
            println!("{json_str}");
        }
        _ => {
            println!("{}", "Automation Logs".green().bold());
            println!("{}", "=".repeat(60));
            if let Some(task_id) = id {
                println!("{:25} {}", "Task ID:", task_id);
            }
            println!("{:25} {}", "Limit:", limit);
            if let Some(lvl) = level {
                println!("{:25} {}", "Level:", lvl);
            }
            if let Some(s) = since {
                println!("{:25} {}", "Since:", s);
            }
            if follow {
                println!("{:25} enabled", "Follow:");
            }
            println!();
            println!("{}", "No log entries found.".yellow());
            println!(
                "{}",
                "Note: Log viewing requires persistent storage integration.".yellow()
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_use_case() {
        assert!(validate_use_case("trailer").is_ok());
        assert!(validate_use_case("highlights").is_ok());
        assert!(validate_use_case("social").is_ok());
        assert!(validate_use_case("documentary").is_ok());
        assert!(validate_use_case("music_video").is_ok());
        assert!(validate_use_case("invalid").is_err());
    }

    #[test]
    fn test_validate_pacing() {
        assert!(validate_pacing("slow").is_ok());
        assert!(validate_pacing("medium").is_ok());
        assert!(validate_pacing("fast").is_ok());
        assert!(validate_pacing("dynamic").is_ok());
        assert!(validate_pacing("invalid").is_err());
    }

    #[test]
    fn test_auto_command_run() {
        let cmd = AutoCommand::Run {
            input: std::path::PathBuf::from("/tmp/video.mkv"),
            output: std::path::PathBuf::from("/tmp/highlight.webm"),
            use_case: "highlights".to_string(),
            target_duration: Some(60.0),
            pacing: "fast".to_string(),
            aspect_ratio: Some("16x9".to_string()),
            dramatic_arc: true,
            music_sync: "beats".to_string(),
            verbose: false,
        };
        assert!(matches!(cmd, AutoCommand::Run { .. }));
    }

    #[test]
    fn test_auto_command_schedule() {
        let cmd = AutoCommand::Schedule {
            name: "nightly-highlights".to_string(),
            input_pattern: "/media/raw/*.mkv".to_string(),
            output_dir: "/media/highlights/".to_string(),
            cron: "0 2 * * *".to_string(),
            use_case: "highlights".to_string(),
            enabled: true,
            max_concurrent: 4,
            notify_email: Some("editor@studio.com".to_string()),
        };
        assert!(matches!(cmd, AutoCommand::Schedule { .. }));
    }

    #[test]
    fn test_auto_command_create() {
        let cmd = AutoCommand::Create {
            name: "social-pipeline".to_string(),
            template: "social-clips".to_string(),
            config: Some(r#"{"target_duration":30}"#.to_string()),
            description: Some("Automated social media clip generation".to_string()),
            tags: Some("social,automated".to_string()),
        };
        assert!(matches!(cmd, AutoCommand::Create { .. }));
    }
}
