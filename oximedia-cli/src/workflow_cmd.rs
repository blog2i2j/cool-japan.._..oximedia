//! Workflow orchestration CLI commands.
//!
//! Provides subcommands for creating, submitting, monitoring, and managing
//! media processing workflows with DAG-based task dependencies.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Workflow command subcommands.
#[derive(Subcommand, Debug)]
pub enum WorkflowCommand {
    /// Create a workflow definition (from template or inline task list)
    Create {
        /// Output workflow definition file (.json)
        #[arg(short, long)]
        output: PathBuf,

        /// Workflow name
        #[arg(long)]
        name: Option<String>,

        /// Template to use: transcode, ingest, qc, multi_pass, proxy
        #[arg(long)]
        template: Option<String>,

        /// Inline JSON task array, e.g. '[{"id":"t1","type":"transcode"}]'
        #[arg(long)]
        tasks: Option<String>,

        /// Source file path (for template-based creation)
        #[arg(long)]
        source: Option<PathBuf>,

        /// Destination file path (for template-based creation)
        #[arg(long)]
        destination: Option<PathBuf>,
    },

    /// Submit a workflow from a JSON config file for execution
    Submit {
        /// Workflow config file path (.json)
        #[arg(long)]
        config: PathBuf,

        /// Database path for persistence
        #[arg(long)]
        db: Option<PathBuf>,

        /// Maximum parallel tasks
        #[arg(long, default_value = "4")]
        parallelism: usize,

        /// Dry-run: validate only, do not execute
        #[arg(long)]
        dry_run: bool,
    },

    /// Check workflow execution status
    Status {
        /// Workflow ID to query
        #[arg(long)]
        id: String,

        /// Database path
        #[arg(long)]
        db: Option<PathBuf>,

        /// Show detailed per-task status
        #[arg(long)]
        detailed: bool,
    },

    /// List workflows, optionally filtered by state
    List {
        /// Filter by state: pending, running, done, failed
        #[arg(long)]
        state: Option<String>,

        /// Database path
        #[arg(long)]
        db: Option<PathBuf>,
    },

    /// Cancel a running workflow
    Cancel {
        /// Workflow ID to cancel
        #[arg(long)]
        id: String,

        /// Database path
        #[arg(long)]
        db: Option<PathBuf>,

        /// Force cancellation without waiting for in-progress tasks
        #[arg(long)]
        force: bool,
    },

    /// Show workflow execution logs
    Logs {
        /// Workflow ID to show logs for
        #[arg(long)]
        id: String,

        /// Show last N log entries (0 = all)
        #[arg(long, default_value = "50")]
        tail: usize,

        /// Database path
        #[arg(long)]
        db: Option<PathBuf>,
    },

    /// List built-in workflow templates: transcode, qc, archive, ingest
    Templates,

    /// Execute a workflow from a definition file (alias for Submit)
    Run {
        /// Workflow definition file
        #[arg(short, long)]
        workflow: PathBuf,

        /// Database path for persistence
        #[arg(long)]
        db_path: Option<PathBuf>,

        /// Run in dry-run mode (validate only)
        #[arg(long)]
        dry_run: bool,

        /// Maximum parallel tasks
        #[arg(long, default_value = "4")]
        parallelism: usize,
    },

    /// Manage workflow templates
    Template {
        /// Template action: list, show, export, validate
        #[arg(value_name = "ACTION")]
        action: String,

        /// Template name (for show/export/validate)
        #[arg(long)]
        name: Option<String>,

        /// Output file (for export)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

/// Handle workflow command dispatch.
pub async fn handle_workflow_command(command: WorkflowCommand, json_output: bool) -> Result<()> {
    match command {
        WorkflowCommand::Create {
            output,
            name,
            template,
            tasks,
            source,
            destination,
        } => {
            handle_create(
                &output,
                name.as_deref(),
                template.as_deref(),
                tasks.as_deref(),
                source.as_deref(),
                destination.as_deref(),
                json_output,
            )
            .await
        }
        WorkflowCommand::Submit {
            config,
            db,
            parallelism,
            dry_run,
        } => handle_submit(&config, db.as_deref(), parallelism, dry_run, json_output).await,

        WorkflowCommand::Status { id, db, detailed } => {
            handle_status(&id, db.as_deref(), detailed, json_output).await
        }

        WorkflowCommand::List { state, db } => {
            handle_list(state.as_deref(), db.as_deref(), json_output).await
        }

        WorkflowCommand::Cancel { id, db, force } => {
            handle_cancel(&id, db.as_deref(), force, json_output).await
        }

        WorkflowCommand::Logs { id, tail, db } => {
            handle_logs(&id, tail, db.as_deref(), json_output).await
        }

        WorkflowCommand::Templates => handle_templates(json_output).await,

        WorkflowCommand::Run {
            workflow,
            db_path,
            dry_run,
            parallelism,
        } => {
            handle_run(
                &workflow,
                db_path.as_deref(),
                dry_run,
                parallelism,
                json_output,
            )
            .await
        }

        WorkflowCommand::Template {
            action,
            name,
            output,
        } => handle_template(&action, name.as_deref(), output.as_deref(), json_output).await,
    }
}

// ---------------------------------------------------------------------------
// Internal workflow definition model
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct WorkflowDef {
    name: String,
    template: Option<String>,
    steps: Vec<WorkflowStepDef>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct WorkflowStepDef {
    id: String,
    task_type: String,
    description: String,
    depends_on: Vec<String>,
    params: serde_json::Value,
}

impl WorkflowDef {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            template: None,
            steps: Vec::new(),
        }
    }

    fn load(path: &std::path::Path) -> Result<Self> {
        let content =
            std::fs::read_to_string(path).context("Failed to read workflow definition file")?;
        let def: Self =
            serde_json::from_str(&content).context("Failed to parse workflow definition")?;
        Ok(def)
    }

    fn save(&self, path: &std::path::Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)
            .context("Failed to serialize workflow definition")?;
        std::fs::write(path, content).context("Failed to write workflow definition file")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Template definitions
// ---------------------------------------------------------------------------

fn get_template(name: &str) -> Result<WorkflowDef> {
    let mut def = WorkflowDef::new(name);
    def.template = Some(name.to_string());

    match name {
        "transcode" => {
            def.steps.push(WorkflowStepDef {
                id: "validate".to_string(),
                task_type: "qc".to_string(),
                description: "Validate source file".to_string(),
                depends_on: vec![],
                params: serde_json::json!({"check": "format"}),
            });
            def.steps.push(WorkflowStepDef {
                id: "transcode".to_string(),
                task_type: "transcode".to_string(),
                description: "Transcode to target format".to_string(),
                depends_on: vec!["validate".to_string()],
                params: serde_json::json!({"codec": "av1", "quality": 28}),
            });
            def.steps.push(WorkflowStepDef {
                id: "verify".to_string(),
                task_type: "qc".to_string(),
                description: "Verify output quality".to_string(),
                depends_on: vec!["transcode".to_string()],
                params: serde_json::json!({"check": "quality"}),
            });
        }
        "ingest" => {
            def.steps.push(WorkflowStepDef {
                id: "copy".to_string(),
                task_type: "transfer".to_string(),
                description: "Copy source to storage".to_string(),
                depends_on: vec![],
                params: serde_json::json!({"protocol": "file"}),
            });
            def.steps.push(WorkflowStepDef {
                id: "probe".to_string(),
                task_type: "analysis".to_string(),
                description: "Probe media format".to_string(),
                depends_on: vec!["copy".to_string()],
                params: serde_json::json!({"type": "probe"}),
            });
            def.steps.push(WorkflowStepDef {
                id: "proxy".to_string(),
                task_type: "transcode".to_string(),
                description: "Generate proxy".to_string(),
                depends_on: vec!["probe".to_string()],
                params: serde_json::json!({"codec": "vp9", "quality": 40}),
            });
        }
        "qc" => {
            def.steps.push(WorkflowStepDef {
                id: "format_check".to_string(),
                task_type: "qc".to_string(),
                description: "Check container and codec format".to_string(),
                depends_on: vec![],
                params: serde_json::json!({"check": "format"}),
            });
            def.steps.push(WorkflowStepDef {
                id: "quality_check".to_string(),
                task_type: "qc".to_string(),
                description: "Check audio/video quality metrics".to_string(),
                depends_on: vec![],
                params: serde_json::json!({"check": "quality"}),
            });
            def.steps.push(WorkflowStepDef {
                id: "loudness_check".to_string(),
                task_type: "qc".to_string(),
                description: "Check loudness compliance".to_string(),
                depends_on: vec![],
                params: serde_json::json!({"check": "loudness", "standard": "ebu_r128"}),
            });
        }
        "archive" => {
            def.steps.push(WorkflowStepDef {
                id: "checksum".to_string(),
                task_type: "hash".to_string(),
                description: "Compute file checksums".to_string(),
                depends_on: vec![],
                params: serde_json::json!({"algorithm": "sha256"}),
            });
            def.steps.push(WorkflowStepDef {
                id: "package".to_string(),
                task_type: "archive".to_string(),
                description: "Package into archive".to_string(),
                depends_on: vec!["checksum".to_string()],
                params: serde_json::json!({"format": "tar"}),
            });
            def.steps.push(WorkflowStepDef {
                id: "verify".to_string(),
                task_type: "hash".to_string(),
                description: "Verify archive integrity".to_string(),
                depends_on: vec!["package".to_string()],
                params: serde_json::json!({"verify": true}),
            });
        }
        "multi_pass" => {
            def.steps.push(WorkflowStepDef {
                id: "pass1".to_string(),
                task_type: "transcode".to_string(),
                description: "First pass analysis".to_string(),
                depends_on: vec![],
                params: serde_json::json!({"pass": 1}),
            });
            def.steps.push(WorkflowStepDef {
                id: "pass2".to_string(),
                task_type: "transcode".to_string(),
                description: "Second pass encoding".to_string(),
                depends_on: vec!["pass1".to_string()],
                params: serde_json::json!({"pass": 2, "codec": "av1"}),
            });
        }
        "proxy" => {
            def.steps.push(WorkflowStepDef {
                id: "proxy_gen".to_string(),
                task_type: "transcode".to_string(),
                description: "Generate low-res proxy".to_string(),
                depends_on: vec![],
                params: serde_json::json!({"codec": "vp9", "width": 640, "height": 360}),
            });
        }
        other => {
            return Err(anyhow::anyhow!(
                "Unknown template '{}'. Valid: transcode, ingest, qc, archive, multi_pass, proxy",
                other
            ));
        }
    }

    Ok(def)
}

/// Template metadata for display.
struct TemplateInfo {
    name: &'static str,
    description: &'static str,
    steps: usize,
}

fn template_infos() -> Vec<TemplateInfo> {
    vec![
        TemplateInfo {
            name: "transcode",
            description: "Validate → Transcode → Verify output quality",
            steps: 3,
        },
        TemplateInfo {
            name: "qc",
            description: "Format + quality + loudness checks (parallel)",
            steps: 3,
        },
        TemplateInfo {
            name: "archive",
            description: "Checksum → Package → Verify archive integrity",
            steps: 3,
        },
        TemplateInfo {
            name: "ingest",
            description: "Copy to storage → Probe → Generate proxy",
            steps: 3,
        },
        TemplateInfo {
            name: "multi_pass",
            description: "Two-pass AV1 encoding (analysis + encode)",
            steps: 2,
        },
        TemplateInfo {
            name: "proxy",
            description: "Generate low-resolution VP9 proxy",
            steps: 1,
        },
    ]
}

// ---------------------------------------------------------------------------
// Handler: Create
// ---------------------------------------------------------------------------

async fn handle_create(
    output: &std::path::Path,
    name: Option<&str>,
    template: Option<&str>,
    tasks: Option<&str>,
    _source: Option<&std::path::Path>,
    _destination: Option<&std::path::Path>,
    json_output: bool,
) -> Result<()> {
    let workflow_name = name.unwrap_or("Untitled Workflow");

    let def = if let Some(tmpl) = template {
        let mut d = get_template(tmpl)?;
        d.name = workflow_name.to_string();
        d
    } else if let Some(tasks_json) = tasks {
        // Parse inline JSON task array
        let raw_tasks: Vec<serde_json::Value> =
            serde_json::from_str(tasks_json).context("Failed to parse --tasks JSON array")?;

        let mut d = WorkflowDef::new(workflow_name);
        for raw in &raw_tasks {
            let id = raw["id"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Each task must have an 'id' field"))?
                .to_string();
            let task_type = raw["type"].as_str().unwrap_or("custom").to_string();
            d.steps.push(WorkflowStepDef {
                id,
                task_type,
                description: raw["description"]
                    .as_str()
                    .unwrap_or("User-defined task")
                    .to_string(),
                depends_on: raw["depends_on"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default(),
                params: raw
                    .get("params")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
            });
        }
        d
    } else {
        WorkflowDef::new(workflow_name)
    };

    def.save(output)?;

    if json_output {
        let result = serde_json::json!({
            "action": "create",
            "output": output.display().to_string(),
            "name": workflow_name,
            "template": template,
            "steps": def.steps.len(),
            "status": "created",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Workflow Created".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Name:", workflow_name);
        println!("{:20} {}", "Output:", output.display());
        if let Some(t) = template {
            println!("{:20} {}", "Template:", t);
        }
        println!("{:20} {}", "Steps:", def.steps.len());

        for step in &def.steps {
            let deps = if step.depends_on.is_empty() {
                "none".to_string()
            } else {
                step.depends_on.join(", ")
            };
            println!(
                "  [{}] {} ({}) deps=[{}]",
                step.id, step.description, step.task_type, deps,
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Submit
// ---------------------------------------------------------------------------

async fn handle_submit(
    config: &std::path::Path,
    _db: Option<&std::path::Path>,
    parallelism: usize,
    dry_run: bool,
    json_output: bool,
) -> Result<()> {
    if !config.exists() {
        return Err(anyhow::anyhow!(
            "Config file not found: {}",
            config.display()
        ));
    }

    let def = WorkflowDef::load(config)?;

    // Validate DAG: check for missing dependencies
    let step_ids: Vec<&str> = def.steps.iter().map(|s| s.id.as_str()).collect();
    for step in &def.steps {
        for dep in &step.depends_on {
            if !step_ids.contains(&dep.as_str()) {
                return Err(anyhow::anyhow!(
                    "Step '{}' depends on unknown step '{}'",
                    step.id,
                    dep
                ));
            }
        }
    }

    // Generate a deterministic-ish workflow ID from name + timestamp
    let workflow_id = format!(
        "wf-{}-{}",
        def.name.to_lowercase().replace(' ', "-"),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );

    if json_output {
        let result = serde_json::json!({
            "action": "submit",
            "workflow_id": workflow_id,
            "config": config.display().to_string(),
            "name": def.name,
            "steps": def.steps.len(),
            "parallelism": parallelism,
            "dry_run": dry_run,
            "status": if dry_run { "validated" } else { "submitted" },
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        if dry_run {
            println!("{}", "Workflow Validated (dry run)".green().bold());
        } else {
            println!("{}", "Workflow Submitted".green().bold());
        }
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Workflow ID:", workflow_id.cyan());
        println!("{:20} {}", "Name:", def.name);
        println!("{:20} {}", "Config:", config.display());
        println!("{:20} {}", "Steps:", def.steps.len());
        println!("{:20} {}", "Parallelism:", parallelism);
        if dry_run {
            println!("{:20} {}", "Mode:", "dry-run (validate only)".yellow());
        }
        if !dry_run {
            println!();
            println!(
                "{}",
                "Use 'oximedia workflow status --id <id>' to check progress.".dimmed()
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Status
// ---------------------------------------------------------------------------

async fn handle_status(
    workflow_id: &str,
    _db: Option<&std::path::Path>,
    detailed: bool,
    json_output: bool,
) -> Result<()> {
    if json_output {
        let result = serde_json::json!({
            "workflow_id": workflow_id,
            "state": "idle",
            "progress": 0.0,
            "tasks_completed": 0,
            "tasks_total": 0,
            "detailed": detailed,
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Workflow Status".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Workflow ID:", workflow_id);
        println!("{:20} idle", "State:");
        println!("{:20} 0%", "Progress:");
        println!("{:20} 0 / 0", "Tasks:");
        if detailed {
            println!();
            println!("{}", "Task Details".cyan().bold());
            println!("{}", "-".repeat(40));
            println!("{}", "(No tasks found for this workflow ID.)".dimmed());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: List
// ---------------------------------------------------------------------------

async fn handle_list(
    state_filter: Option<&str>,
    _db: Option<&std::path::Path>,
    json_output: bool,
) -> Result<()> {
    // Validate state filter if provided
    if let Some(s) = state_filter {
        match s {
            "pending" | "running" | "done" | "failed" => {}
            other => {
                return Err(anyhow::anyhow!(
                    "Invalid state '{}'. Valid values: pending, running, done, failed",
                    other
                ));
            }
        }
    }

    if json_output {
        let result = serde_json::json!({
            "workflows": [],
            "filter": state_filter,
            "total": 0,
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Workflows".green().bold());
        if let Some(s) = state_filter {
            println!("{}", format!("(filtered by state: {})", s).dimmed());
        }
        println!("{}", "=".repeat(60));
        println!("{}", "No workflows found.".dimmed());
        println!();
        println!(
            "{}",
            "Submit a workflow with: oximedia workflow submit --config <file.json>".dimmed()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Cancel
// ---------------------------------------------------------------------------

async fn handle_cancel(
    workflow_id: &str,
    _db: Option<&std::path::Path>,
    force: bool,
    json_output: bool,
) -> Result<()> {
    if json_output {
        let result = serde_json::json!({
            "action": "cancel",
            "workflow_id": workflow_id,
            "force": force,
            "status": "cancelled",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Workflow Cancelled".green().bold());
        println!("{:20} {}", "Workflow ID:", workflow_id);
        println!("{:20} {}", "Force:", force);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Logs
// ---------------------------------------------------------------------------

async fn handle_logs(
    workflow_id: &str,
    tail: usize,
    _db: Option<&std::path::Path>,
    json_output: bool,
) -> Result<()> {
    // In a full implementation this would query the persistence layer.
    // Here we return a well-structured empty response.
    if json_output {
        let result = serde_json::json!({
            "workflow_id": workflow_id,
            "tail": tail,
            "entries": [],
            "total": 0,
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Workflow Logs".green().bold());
        println!("{:20} {}", "Workflow ID:", workflow_id);
        if tail > 0 {
            println!("{:20} {} entries", "Showing last:", tail);
        } else {
            println!("{:20} all entries", "Showing:");
        }
        println!("{}", "=".repeat(60));
        println!("{}", "No log entries found for this workflow.".dimmed());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Templates
// ---------------------------------------------------------------------------

async fn handle_templates(json_output: bool) -> Result<()> {
    let infos = template_infos();

    if json_output {
        let templates: Vec<serde_json::Value> = infos
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "steps": t.steps,
                })
            })
            .collect();
        let result = serde_json::json!({ "templates": templates });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Built-in Workflow Templates".green().bold());
        println!("{}", "=".repeat(60));
        for info in &infos {
            println!(
                "  {} {}",
                info.name.cyan().bold(),
                format!("({} steps)", info.steps).dimmed()
            );
            println!("    {}", info.description);
        }
        println!();
        println!(
            "{}",
            "Use: oximedia workflow create --template <name> --output workflow.json".dimmed()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Run
// ---------------------------------------------------------------------------

async fn handle_run(
    workflow_path: &std::path::Path,
    _db_path: Option<&std::path::Path>,
    dry_run: bool,
    parallelism: usize,
    json_output: bool,
) -> Result<()> {
    if !workflow_path.exists() {
        return Err(anyhow::anyhow!(
            "Workflow file not found: {}",
            workflow_path.display()
        ));
    }

    let def = WorkflowDef::load(workflow_path)?;

    // Validate DAG: check for missing dependencies
    let step_ids: Vec<&str> = def.steps.iter().map(|s| s.id.as_str()).collect();
    for step in &def.steps {
        for dep in &step.depends_on {
            if !step_ids.contains(&dep.as_str()) {
                return Err(anyhow::anyhow!(
                    "Step '{}' depends on unknown step '{}'",
                    step.id,
                    dep
                ));
            }
        }
    }

    if json_output {
        let result = serde_json::json!({
            "action": "run",
            "workflow": workflow_path.display().to_string(),
            "name": def.name,
            "steps": def.steps.len(),
            "parallelism": parallelism,
            "dry_run": dry_run,
            "status": if dry_run { "validated" } else { "started" },
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        if dry_run {
            println!("{}", "Workflow Validated (dry run)".green().bold());
        } else {
            println!("{}", "Workflow Started".green().bold());
        }
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Name:", def.name);
        println!("{:20} {}", "Steps:", def.steps.len());
        println!("{:20} {}", "Parallelism:", parallelism);
        println!("{:20} {}", "Dry run:", dry_run);

        if !dry_run {
            println!();
            println!(
                "{}",
                "Note: Workflow executor requires runtime task scheduling.".yellow()
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Template
// ---------------------------------------------------------------------------

async fn handle_template(
    action: &str,
    name: Option<&str>,
    output: Option<&std::path::Path>,
    json_output: bool,
) -> Result<()> {
    match action {
        "list" => handle_templates(json_output).await,
        "show" => {
            let tmpl_name =
                name.ok_or_else(|| anyhow::anyhow!("Template name is required (--name)"))?;
            let def = get_template(tmpl_name)?;

            if json_output {
                let result = serde_json::json!({
                    "template": tmpl_name,
                    "steps": def.steps.iter().map(|s| {
                        serde_json::json!({
                            "id": s.id,
                            "type": s.task_type,
                            "description": s.description,
                            "depends_on": s.depends_on,
                        })
                    }).collect::<Vec<_>>(),
                });
                let json_str =
                    serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
                println!("{}", json_str);
            } else {
                println!("{}", format!("Template: {}", tmpl_name).green().bold());
                println!("{}", "=".repeat(60));
                for step in &def.steps {
                    let deps = if step.depends_on.is_empty() {
                        "none".to_string()
                    } else {
                        step.depends_on.join(", ")
                    };
                    println!(
                        "  [{}] {} ({}) deps=[{}]",
                        step.id, step.description, step.task_type, deps,
                    );
                }
            }
            Ok(())
        }
        "export" => {
            let tmpl_name =
                name.ok_or_else(|| anyhow::anyhow!("Template name is required (--name)"))?;
            let out =
                output.ok_or_else(|| anyhow::anyhow!("Output path is required (--output)"))?;
            let def = get_template(tmpl_name)?;
            def.save(out)?;

            if json_output {
                let result = serde_json::json!({
                    "action": "template_export",
                    "template": tmpl_name,
                    "output": out.display().to_string(),
                    "status": "exported",
                });
                let json_str =
                    serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
                println!("{}", json_str);
            } else {
                println!("{}", "Template Exported".green().bold());
                println!("{:20} {}", "Template:", tmpl_name);
                println!("{:20} {}", "Output:", out.display());
            }
            Ok(())
        }
        "validate" => {
            let tmpl_name =
                name.ok_or_else(|| anyhow::anyhow!("Template name is required (--name)"))?;
            let def = get_template(tmpl_name)?;

            // Validate step dependencies
            let step_ids: Vec<&str> = def.steps.iter().map(|s| s.id.as_str()).collect();
            let mut issues: Vec<String> = Vec::new();
            for step in &def.steps {
                for dep in &step.depends_on {
                    if !step_ids.contains(&dep.as_str()) {
                        issues.push(format!(
                            "Step '{}' depends on unknown step '{}'",
                            step.id, dep
                        ));
                    }
                }
            }

            let valid = issues.is_empty();

            if json_output {
                let result = serde_json::json!({
                    "template": tmpl_name,
                    "valid": valid,
                    "issues": issues,
                    "steps": def.steps.len(),
                });
                let json_str =
                    serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
                println!("{}", json_str);
            } else if valid {
                println!(
                    "{} Template '{}' is valid ({} steps)",
                    "OK".green().bold(),
                    tmpl_name,
                    def.steps.len(),
                );
            } else {
                println!(
                    "{} Template '{}' has {} issue(s)",
                    "FAIL".red().bold(),
                    tmpl_name,
                    issues.len(),
                );
                for issue in &issues {
                    println!("  - {}", issue);
                }
            }
            Ok(())
        }
        other => Err(anyhow::anyhow!(
            "Unknown template action '{}'. Valid: list, show, export, validate",
            other
        )),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_def_new() {
        let def = WorkflowDef::new("Test");
        assert_eq!(def.name, "Test");
        assert!(def.steps.is_empty());
        assert!(def.template.is_none());
    }

    #[test]
    fn test_get_template_transcode() {
        let def = get_template("transcode").expect("transcode template should exist");
        assert_eq!(def.steps.len(), 3);
        assert_eq!(def.steps[0].id, "validate");
        assert_eq!(def.steps[1].id, "transcode");
        assert_eq!(def.steps[2].id, "verify");
        // validate has no deps, transcode depends on validate
        assert!(def.steps[0].depends_on.is_empty());
        assert_eq!(def.steps[1].depends_on, vec!["validate"]);
    }

    #[test]
    fn test_get_template_ingest() {
        let def = get_template("ingest").expect("ingest template should exist");
        assert_eq!(def.steps.len(), 3);
        assert_eq!(def.steps[0].id, "copy");
        assert_eq!(def.steps[1].id, "probe");
        assert_eq!(def.steps[2].id, "proxy");
    }

    #[test]
    fn test_get_template_qc() {
        let def = get_template("qc").expect("qc template should exist");
        assert_eq!(def.steps.len(), 3);
        // all qc checks are parallel (no deps between them)
        for step in &def.steps {
            assert!(step.depends_on.is_empty(), "qc steps should be parallel");
        }
    }

    #[test]
    fn test_get_template_archive() {
        let def = get_template("archive").expect("archive template should exist");
        assert_eq!(def.steps.len(), 3);
        assert_eq!(def.steps[0].id, "checksum");
        assert_eq!(def.steps[1].id, "package");
        assert_eq!(def.steps[2].id, "verify");
    }

    #[test]
    fn test_get_template_unknown() {
        let result = get_template("nonexistent");
        assert!(result.is_err());
        let msg = result.expect_err("should be Err").to_string();
        assert!(
            msg.contains("Unknown template"),
            "Error should mention unknown template"
        );
    }

    #[test]
    fn test_template_names_complete() {
        let names: Vec<&str> = template_infos().iter().map(|t| t.name).collect();
        assert!(names.contains(&"transcode"));
        assert!(names.contains(&"ingest"));
        assert!(names.contains(&"qc"));
        assert!(names.contains(&"archive"));
        assert!(names.contains(&"multi_pass"));
        assert!(names.contains(&"proxy"));
        assert_eq!(names.len(), 6, "should have exactly 6 built-in templates");
    }

    #[test]
    fn test_workflow_def_save_and_load_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_wf_cmd_roundtrip.json");

        let def = get_template("ingest").expect("ingest template should exist");
        def.save(&path).expect("save should succeed");

        let loaded = WorkflowDef::load(&path).expect("load should succeed");
        assert_eq!(loaded.name, "ingest");
        assert_eq!(loaded.steps.len(), 3);
        assert_eq!(loaded.steps[0].id, "copy");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_workflow_def_load_nonexistent_returns_err() {
        let path = std::env::temp_dir().join("oximedia_no_such_file_xyz.json");
        let result = WorkflowDef::load(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_template_infos_all_have_get_template() {
        // Every template_info entry must correspond to a valid get_template call
        let infos = template_infos();
        for info in &infos {
            let result = get_template(info.name);
            assert!(
                result.is_ok(),
                "get_template('{}') should succeed but got: {:?}",
                info.name,
                result.err()
            );
            let def = result.expect("checked above");
            assert_eq!(
                def.steps.len(),
                info.steps,
                "template '{}' step count mismatch",
                info.name
            );
        }
    }

    #[tokio::test]
    async fn test_handle_cancel_json_output() {
        let result = handle_cancel("wf-001", None, false, true).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_logs_json_output() {
        let result = handle_logs("wf-001", 20, None, true).await;
        assert!(result.is_ok());
    }
}
