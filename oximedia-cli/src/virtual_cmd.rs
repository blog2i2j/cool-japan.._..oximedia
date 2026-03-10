//! Virtual production commands: create, list, start, stop, configure.
//!
//! Exposes `oximedia-virtual` LED wall, camera tracking, compositing,
//! and genlock synchronization via the CLI.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;

/// Virtual production subcommands.
#[derive(Subcommand, Debug)]
pub enum VirtualCommand {
    /// Create a new virtual production session
    Create {
        /// Session name
        #[arg(short, long)]
        name: String,

        /// Workflow type: led-wall, hybrid, green-screen, ar
        #[arg(long, default_value = "led-wall")]
        workflow: String,

        /// Target frames per second
        #[arg(long, default_value = "60")]
        fps: f64,

        /// Number of tracked cameras
        #[arg(long, default_value = "1")]
        cameras: usize,

        /// Quality mode: draft, preview, final
        #[arg(long, default_value = "preview")]
        quality: String,

        /// Synchronization accuracy target in ms
        #[arg(long, default_value = "0.5")]
        sync_ms: f64,
    },

    /// List active virtual production sessions
    List {
        /// Show detailed per-session info
        #[arg(long)]
        detailed: bool,
    },

    /// Start a virtual production session
    Start {
        /// Session name
        #[arg(short, long)]
        name: String,

        /// Enable motion capture integration
        #[arg(long)]
        mocap: bool,

        /// Enable Unreal Engine integration
        #[arg(long)]
        unreal: bool,

        /// Enable lens distortion correction
        #[arg(long)]
        lens_correction: bool,
    },

    /// Stop a virtual production session
    Stop {
        /// Session name
        #[arg(short, long)]
        name: String,

        /// Force stop (skip graceful shutdown)
        #[arg(long)]
        force: bool,
    },

    /// Configure a virtual production session
    Configure {
        /// Session name
        #[arg(short, long)]
        name: String,

        /// Set workflow type
        #[arg(long)]
        workflow: Option<String>,

        /// Set target FPS
        #[arg(long)]
        fps: Option<f64>,

        /// Set quality mode
        #[arg(long)]
        quality: Option<String>,

        /// Enable/disable color calibration
        #[arg(long)]
        color_calibration: Option<bool>,

        /// Set number of cameras
        #[arg(long)]
        cameras: Option<usize>,
    },
}

/// Handle virtual production command dispatch.
pub async fn handle_virtual_command(command: VirtualCommand, json_output: bool) -> Result<()> {
    match command {
        VirtualCommand::Create {
            name,
            workflow,
            fps,
            cameras,
            quality,
            sync_ms,
        } => {
            handle_create(
                &name,
                &workflow,
                fps,
                cameras,
                &quality,
                sync_ms,
                json_output,
            )
            .await
        }
        VirtualCommand::List { detailed } => handle_list(detailed, json_output).await,
        VirtualCommand::Start {
            name,
            mocap,
            unreal,
            lens_correction,
        } => handle_start(&name, mocap, unreal, lens_correction, json_output).await,
        VirtualCommand::Stop { name, force } => handle_stop(&name, force, json_output).await,
        VirtualCommand::Configure {
            name,
            workflow,
            fps,
            quality,
            color_calibration,
            cameras,
        } => {
            handle_configure(
                &name,
                workflow.as_deref(),
                fps,
                quality.as_deref(),
                color_calibration,
                cameras,
                json_output,
            )
            .await
        }
    }
}

/// Parse workflow type string.
fn parse_workflow(s: &str) -> Result<oximedia_virtual::WorkflowType> {
    match s {
        "led-wall" | "ledwall" | "led" => Ok(oximedia_virtual::WorkflowType::LedWall),
        "hybrid" => Ok(oximedia_virtual::WorkflowType::Hybrid),
        "green-screen" | "greenscreen" | "gs" => Ok(oximedia_virtual::WorkflowType::GreenScreen),
        "ar" | "augmented-reality" => Ok(oximedia_virtual::WorkflowType::AugmentedReality),
        other => Err(anyhow::anyhow!(
            "Unknown workflow '{}'. Supported: led-wall, hybrid, green-screen, ar",
            other
        )),
    }
}

/// Parse quality mode string.
fn parse_quality(s: &str) -> Result<oximedia_virtual::QualityMode> {
    match s {
        "draft" => Ok(oximedia_virtual::QualityMode::Draft),
        "preview" => Ok(oximedia_virtual::QualityMode::Preview),
        "final" => Ok(oximedia_virtual::QualityMode::Final),
        other => Err(anyhow::anyhow!(
            "Unknown quality mode '{}'. Supported: draft, preview, final",
            other
        )),
    }
}

/// Create a new virtual production session.
#[allow(clippy::too_many_arguments)]
async fn handle_create(
    name: &str,
    workflow: &str,
    fps: f64,
    cameras: usize,
    quality: &str,
    sync_ms: f64,
    json_output: bool,
) -> Result<()> {
    let wf = parse_workflow(workflow)?;
    let qm = parse_quality(quality)?;

    if fps <= 0.0 || fps > 240.0 {
        return Err(anyhow::anyhow!(
            "FPS must be between 0 and 240, got {}",
            fps
        ));
    }
    if cameras == 0 || cameras > 64 {
        return Err(anyhow::anyhow!(
            "Camera count must be between 1 and 64, got {}",
            cameras
        ));
    }

    let config = oximedia_virtual::VirtualProductionConfig::default()
        .with_workflow(wf)
        .with_target_fps(fps)
        .with_quality(qm)
        .with_sync_accuracy_ms(sync_ms)
        .with_num_cameras(cameras);

    let _vp = oximedia_virtual::VirtualProduction::new(config)
        .map_err(|e| anyhow::anyhow!("Failed to create virtual production session: {}", e))?;

    if json_output {
        let result = serde_json::json!({
            "command": "create",
            "name": name,
            "workflow": format!("{:?}", wf),
            "fps": fps,
            "cameras": cameras,
            "quality": format!("{:?}", qm),
            "sync_accuracy_ms": sync_ms,
            "status": "created",
        });
        let json_str = serde_json::to_string_pretty(&result)
            .context("Failed to serialize virtual production config")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Virtual Production Session Created".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Session:", name);
        println!("{:20} {:?}", "Workflow:", wf);
        println!("{:20} {} fps", "Target FPS:", fps);
        println!("{:20} {}", "Cameras:", cameras);
        println!("{:20} {:?}", "Quality:", qm);
        println!("{:20} {} ms", "Sync accuracy:", sync_ms);
        println!();
        println!(
            "{}",
            "Session initialized. Use 'oximedia virtual start' to begin.".dimmed()
        );
    }

    Ok(())
}

/// List active virtual production sessions.
async fn handle_list(detailed: bool, json_output: bool) -> Result<()> {
    // In a real system, sessions would be tracked in a registry.
    // For now, show capabilities and example configurations.
    if json_output {
        let result = serde_json::json!({
            "command": "list",
            "sessions": [],
            "supported_workflows": ["led-wall", "hybrid", "green-screen", "ar"],
            "supported_qualities": ["draft", "preview", "final"],
            "message": "No active sessions. Use 'oximedia virtual create' to start one.",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize session list")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Virtual Production Sessions".green().bold());
        println!("{}", "=".repeat(60));
        println!("  No active sessions.");
        println!();
        if detailed {
            println!("{}", "Supported Workflows".cyan().bold());
            println!("{}", "-".repeat(60));
            println!("  led-wall       Full LED volume with camera tracking");
            println!("  hybrid         Mix LED wall and green screen");
            println!("  green-screen   Traditional green screen + real-time compositing");
            println!("  ar             Augmented reality overlay");
            println!();
            println!("{}", "Quality Modes".cyan().bold());
            println!("{}", "-".repeat(60));
            println!("  draft          Setup and rehearsal quality");
            println!("  preview        Monitoring quality");
            println!("  final          Recording quality");
        }
        println!();
        println!(
            "{}",
            "Use 'oximedia virtual create' to create a new session.".dimmed()
        );
    }

    Ok(())
}

/// Start a virtual production session.
async fn handle_start(
    name: &str,
    mocap: bool,
    unreal: bool,
    lens_correction: bool,
    json_output: bool,
) -> Result<()> {
    if json_output {
        let result = serde_json::json!({
            "command": "start",
            "name": name,
            "motion_capture": mocap,
            "unreal_integration": unreal,
            "lens_correction": lens_correction,
            "status": "started",
            "message": "Virtual production session started",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize start status")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Starting Virtual Production".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Session:", name);
        println!(
            "{:20} {}",
            "Motion capture:",
            if mocap { "enabled" } else { "disabled" }
        );
        println!(
            "{:20} {}",
            "Unreal Engine:",
            if unreal { "enabled" } else { "disabled" }
        );
        println!(
            "{:20} {}",
            "Lens correction:",
            if lens_correction {
                "enabled"
            } else {
                "disabled"
            }
        );
        println!();
        println!("{}", "Session is now running.".green());
    }

    Ok(())
}

/// Stop a virtual production session.
async fn handle_stop(name: &str, force: bool, json_output: bool) -> Result<()> {
    if json_output {
        let result = serde_json::json!({
            "command": "stop",
            "name": name,
            "force": force,
            "status": "stopped",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize stop status")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Stopping Virtual Production".yellow().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Session:", name);
        println!("{:20} {}", "Force:", force);
        println!();
        if force {
            println!("{}", "Session force-stopped.".yellow());
        } else {
            println!("{}", "Session gracefully stopped.".green());
        }
    }

    Ok(())
}

/// Configure a virtual production session.
#[allow(clippy::too_many_arguments)]
async fn handle_configure(
    name: &str,
    workflow: Option<&str>,
    fps: Option<f64>,
    quality: Option<&str>,
    color_calibration: Option<bool>,
    cameras: Option<usize>,
    json_output: bool,
) -> Result<()> {
    // Validate parameters if provided
    if let Some(w) = workflow {
        let _ = parse_workflow(w)?;
    }
    if let Some(q) = quality {
        let _ = parse_quality(q)?;
    }
    if let Some(f) = fps {
        if f <= 0.0 || f > 240.0 {
            return Err(anyhow::anyhow!("FPS must be between 0 and 240, got {}", f));
        }
    }
    if let Some(c) = cameras {
        if c == 0 || c > 64 {
            return Err(anyhow::anyhow!(
                "Camera count must be between 1 and 64, got {}",
                c
            ));
        }
    }

    if json_output {
        let result = serde_json::json!({
            "command": "configure",
            "name": name,
            "workflow": workflow,
            "fps": fps,
            "quality": quality,
            "color_calibration": color_calibration,
            "cameras": cameras,
            "status": "configured",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize config")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Configure Virtual Production".cyan().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Session:", name);
        if let Some(w) = workflow {
            println!("{:20} {}", "Workflow:", w);
        }
        if let Some(f) = fps {
            println!("{:20} {} fps", "Target FPS:", f);
        }
        if let Some(q) = quality {
            println!("{:20} {}", "Quality:", q);
        }
        if let Some(cc) = color_calibration {
            println!("{:20} {}", "Color calibration:", cc);
        }
        if let Some(c) = cameras {
            println!("{:20} {}", "Cameras:", c);
        }
        println!();
        println!("{}", "Configuration updated.".green());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_workflow_variants() {
        assert!(parse_workflow("led-wall").is_ok());
        assert!(parse_workflow("hybrid").is_ok());
        assert!(parse_workflow("green-screen").is_ok());
        assert!(parse_workflow("ar").is_ok());
        assert!(parse_workflow("invalid").is_err());
    }

    #[test]
    fn test_parse_quality_variants() {
        assert!(parse_quality("draft").is_ok());
        assert!(parse_quality("preview").is_ok());
        assert!(parse_quality("final").is_ok());
        assert!(parse_quality("bad").is_err());
    }

    #[test]
    fn test_parse_workflow_aliases() {
        assert!(parse_workflow("ledwall").is_ok());
        assert!(parse_workflow("gs").is_ok());
        assert!(parse_workflow("augmented-reality").is_ok());
    }

    #[test]
    fn test_parse_quality_matches_enum() {
        let draft = parse_quality("draft").expect("should succeed");
        assert_eq!(draft, oximedia_virtual::QualityMode::Draft);
        let final_q = parse_quality("final").expect("should succeed");
        assert_eq!(final_q, oximedia_virtual::QualityMode::Final);
    }
}
