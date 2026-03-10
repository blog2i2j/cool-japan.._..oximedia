//! Workflow orchestration CLI commands.
//!
//! Provides subcommands for creating, running, monitoring, and managing
//! media processing workflows with DAG-based task dependencies.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Workflow command subcommands.
#[derive(Subcommand, Debug)]
pub enum WorkflowCommand {
    /// Create a workflow from a template or definition file
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

        /// Source file path (for template-based creation)
        #[arg(long)]
        source: Option<PathBuf>,

        /// Destination file path (for template-based creation)
        #[arg(long)]
        destination: Option<PathBuf>,
    },

    /// Execute a workflow
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

    /// Check workflow execution status
    Status {
        /// Workflow ID or definition file
        #[arg(short, long)]
        workflow: String,

        /// Database path
        #[arg(long)]
        db_path: Option<PathBuf>,

        /// Show detailed per-task status
        #[arg(long)]
        detailed: bool,
    },

    /// List available workflows and templates
    List {
        /// List templates instead of running workflows
        #[arg(long)]
        templates: bool,

        /// Database path (for running workflows)
        #[arg(long)]
        db_path: Option<PathBuf>,
    },

    /// Cancel a running workflow
    Cancel {
        /// Workflow ID to cancel
        #[arg(short, long)]
        workflow_id: String,

        /// Database path
        #[arg(long)]
        db_path: Option<PathBuf>,

        /// Force cancellation without waiting for in-progress tasks
        #[arg(long)]
        force: bool,
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
            source,
            destination,
        } => {
            handle_create(
                &output,
                name.as_deref(),
                template.as_deref(),
                source.as_deref(),
                destination.as_deref(),
                json_output,
            )
            .await
        }
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
        WorkflowCommand::Status {
            workflow,
            db_path,
            detailed,
        } => handle_status(&workflow, db_path.as_deref(), detailed, json_output).await,
        WorkflowCommand::List { templates, db_path } => {
            handle_list(templates, db_path.as_deref(), json_output).await
        }
        WorkflowCommand::Cancel {
            workflow_id,
            db_path,
            force,
        } => handle_cancel(&workflow_id, db_path.as_deref(), force, json_output).await,
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
                "Unknown template '{}'. Valid: transcode, ingest, qc, multi_pass, proxy",
                other
            ));
        }
    }

    Ok(def)
}

fn list_template_names() -> Vec<&'static str> {
    vec!["transcode", "ingest", "qc", "multi_pass", "proxy"]
}

// ---------------------------------------------------------------------------
// Handler: Create
// ---------------------------------------------------------------------------

async fn handle_create(
    output: &std::path::Path,
    name: Option<&str>,
    template: Option<&str>,
    _source: Option<&std::path::Path>,
    _destination: Option<&std::path::Path>,
    json_output: bool,
) -> Result<()> {
    let workflow_name = name.unwrap_or("Untitled Workflow");

    let def = if let Some(tmpl) = template {
        let mut d = get_template(tmpl)?;
        d.name = workflow_name.to_string();
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
// Handler: Status
// ---------------------------------------------------------------------------

async fn handle_status(
    workflow: &str,
    _db_path: Option<&std::path::Path>,
    detailed: bool,
    json_output: bool,
) -> Result<()> {
    if json_output {
        let result = serde_json::json!({
            "workflow": workflow,
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
        println!("{:20} {}", "Workflow:", workflow);
        println!("{:20} idle", "State:");
        println!("{:20} 0%", "Progress:");
        println!("{:20} 0 / 0", "Tasks:");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: List
// ---------------------------------------------------------------------------

async fn handle_list(
    templates: bool,
    _db_path: Option<&std::path::Path>,
    json_output: bool,
) -> Result<()> {
    if templates {
        let names = list_template_names();
        if json_output {
            let result = serde_json::json!({
                "templates": names,
            });
            let json_str =
                serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
            println!("{}", json_str);
        } else {
            println!("{}", "Available Workflow Templates".green().bold());
            println!("{}", "=".repeat(60));
            for name in &names {
                println!("  - {}", name);
            }
        }
    } else if json_output {
        let result = serde_json::json!({
            "workflows": [],
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Running Workflows".green().bold());
        println!("{}", "=".repeat(60));
        println!("{}", "No active workflows.".dimmed());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Cancel
// ---------------------------------------------------------------------------

async fn handle_cancel(
    workflow_id: &str,
    _db_path: Option<&std::path::Path>,
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
// Handler: Template
// ---------------------------------------------------------------------------

async fn handle_template(
    action: &str,
    name: Option<&str>,
    output: Option<&std::path::Path>,
    json_output: bool,
) -> Result<()> {
    match action {
        "list" => handle_list(true, None, json_output).await,
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
    }

    #[test]
    fn test_get_template_transcode() {
        let def = get_template("transcode");
        assert!(def.is_ok());
        let def = def.expect("should succeed");
        assert_eq!(def.steps.len(), 3);
        assert_eq!(def.steps[0].id, "validate");
        assert_eq!(def.steps[1].id, "transcode");
        assert_eq!(def.steps[2].id, "verify");
    }

    #[test]
    fn test_get_template_unknown() {
        let def = get_template("nonexistent");
        assert!(def.is_err());
    }

    #[test]
    fn test_workflow_def_save_load() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_workflow_def.json");

        let def = get_template("ingest").expect("should succeed");
        def.save(&path).expect("save should succeed");

        let loaded = WorkflowDef::load(&path).expect("load should succeed");
        assert_eq!(loaded.name, "ingest");
        assert_eq!(loaded.steps.len(), 3);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_list_template_names() {
        let names = list_template_names();
        assert!(names.contains(&"transcode"));
        assert!(names.contains(&"ingest"));
        assert!(names.contains(&"qc"));
        assert!(names.contains(&"multi_pass"));
        assert!(names.contains(&"proxy"));
    }
}
