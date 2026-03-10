//! Multi-camera CLI commands for OxiMedia.
//!
//! Provides multi-camera synchronization, switching, compositing,
//! color matching, and export commands.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Multi-camera command subcommands.
#[derive(Subcommand, Debug)]
pub enum MulticamCommand {
    /// Synchronize multiple camera angles
    Sync {
        /// Input camera files
        #[arg(short, long, required = true, num_args = 2..)]
        inputs: Vec<PathBuf>,

        /// Output synchronized timeline file (JSON)
        #[arg(short, long)]
        output: PathBuf,

        /// Sync method: audio, timecode, marker
        #[arg(long, default_value = "audio")]
        method: String,

        /// Drift tolerance in frames
        #[arg(long, default_value = "2")]
        drift_tolerance: u32,
    },

    /// Switch between camera angles at specified points
    Switch {
        /// Input camera files
        #[arg(short, long, required = true, num_args = 2..)]
        inputs: Vec<PathBuf>,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// JSON switch points: [{"time": 1.0, "camera": 0}, ...]
        #[arg(long)]
        switch_points: Option<String>,

        /// Enable automatic switching based on content analysis
        #[arg(long)]
        auto_switch: bool,

        /// Minimum shot duration in seconds for auto-switch
        #[arg(long, default_value = "2.0")]
        min_duration: f64,
    },

    /// Composite multiple cameras into a single frame layout
    Composite {
        /// Input camera files
        #[arg(short, long, required = true, num_args = 2..)]
        inputs: Vec<PathBuf>,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Layout type: grid, pip, side_by_side, stack
        #[arg(long, default_value = "grid")]
        layout: String,

        /// Output width in pixels
        #[arg(long)]
        width: Option<u32>,

        /// Output height in pixels
        #[arg(long)]
        height: Option<u32>,

        /// Grid spacing in pixels
        #[arg(long, default_value = "4")]
        spacing: u32,
    },

    /// Match colors across camera angles
    ColorMatch {
        /// Reference camera file
        #[arg(long)]
        reference: PathBuf,

        /// Input camera files to match
        #[arg(short, long, required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Output directory for matched files
        #[arg(short, long)]
        output_dir: PathBuf,
    },

    /// Export multi-camera timeline in various formats
    Export {
        /// Input timeline file (JSON)
        #[arg(short, long)]
        timeline: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Export format: multicam_edl, xml, json
        #[arg(long, default_value = "multicam_edl")]
        format: String,
    },

    /// Show information about multi-camera layouts
    Layouts {},
}

/// Handle multicam command dispatch.
pub async fn handle_multicam_command(command: MulticamCommand, json_output: bool) -> Result<()> {
    match command {
        MulticamCommand::Sync {
            inputs,
            output,
            method,
            drift_tolerance,
        } => sync_cameras(&inputs, &output, &method, drift_tolerance, json_output).await,
        MulticamCommand::Switch {
            inputs,
            output,
            switch_points,
            auto_switch,
            min_duration,
        } => {
            switch_cameras(
                &inputs,
                &output,
                switch_points.as_deref(),
                auto_switch,
                min_duration,
                json_output,
            )
            .await
        }
        MulticamCommand::Composite {
            inputs,
            output,
            layout,
            width,
            height,
            spacing,
        } => {
            composite_cameras(
                &inputs,
                &output,
                &layout,
                width,
                height,
                spacing,
                json_output,
            )
            .await
        }
        MulticamCommand::ColorMatch {
            reference,
            inputs,
            output_dir,
        } => color_match(&reference, &inputs, &output_dir, json_output).await,
        MulticamCommand::Export {
            timeline,
            output,
            format,
        } => export_timeline(&timeline, &output, &format, json_output).await,
        MulticamCommand::Layouts {} => list_layouts(json_output).await,
    }
}

/// Validate sync method.
fn validate_sync_method(method: &str) -> Result<()> {
    match method {
        "audio" | "timecode" | "marker" => Ok(()),
        other => Err(anyhow::anyhow!(
            "Unknown sync method '{}'. Expected: audio, timecode, marker",
            other
        )),
    }
}

/// Validate layout type and return description.
fn layout_description(layout: &str) -> Result<&'static str> {
    match layout {
        "grid" => Ok("Grid layout (auto-sized)"),
        "pip" => Ok("Picture-in-picture (main + inset)"),
        "side_by_side" => Ok("Side-by-side (horizontal split)"),
        "stack" => Ok("Vertical stack layout"),
        other => Err(anyhow::anyhow!(
            "Unknown layout '{}'. Expected: grid, pip, side_by_side, stack",
            other
        )),
    }
}

/// Synchronize multiple camera angles.
async fn sync_cameras(
    inputs: &[PathBuf],
    output: &PathBuf,
    method: &str,
    drift_tolerance: u32,
    json_output: bool,
) -> Result<()> {
    validate_sync_method(method)?;

    for input in inputs {
        if !input.exists() {
            return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
        }
    }

    // Build a multicam config
    let config = oximedia_multicam::MultiCamConfig {
        angle_count: inputs.len(),
        enable_audio_sync: method == "audio",
        enable_timecode_sync: method == "timecode",
        enable_visual_sync: method == "marker",
        drift_tolerance,
        ..oximedia_multicam::MultiCamConfig::default()
    };

    // Generate sync result data
    let sync_result = serde_json::json!({
        "cameras": inputs.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "method": method,
        "angle_count": config.angle_count,
        "drift_tolerance": drift_tolerance,
        "frame_rate": config.frame_rate,
        "status": "sync_ready",
        "offsets": inputs.iter().enumerate().map(|(i, _)| {
            serde_json::json!({ "camera": i, "offset_frames": 0, "confidence": 1.0 })
        }).collect::<Vec<_>>(),
        "message": "Cameras configured; audio/timecode sync requires frame decoding pipeline",
    });

    let json_str =
        serde_json::to_string_pretty(&sync_result).context("Failed to serialize sync result")?;

    tokio::fs::write(output, json_str.as_bytes())
        .await
        .context("Failed to write output file")?;

    if json_output {
        println!("{}", json_str);
    } else {
        println!("{}", "Multi-Camera Sync".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Cameras:", inputs.len());
        println!("{:20} {}", "Method:", method);
        println!("{:20} {} frames", "Drift tolerance:", drift_tolerance);
        println!("{:20} {}", "Output:", output.display());
        println!();
        for (i, input) in inputs.iter().enumerate() {
            println!("  Camera {}: {}", i, input.display());
        }
        println!();
        println!(
            "{}",
            "Sync configuration written. Frame decoding pipeline needed for actual sync.".yellow()
        );
    }

    Ok(())
}

/// Switch between camera angles.
async fn switch_cameras(
    inputs: &[PathBuf],
    output: &PathBuf,
    switch_points_json: Option<&str>,
    auto_switch: bool,
    min_duration: f64,
    json_output: bool,
) -> Result<()> {
    for input in inputs {
        if !input.exists() {
            return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
        }
    }

    let switch_points: Vec<serde_json::Value> = if let Some(json) = switch_points_json {
        serde_json::from_str(json).context("Failed to parse switch points JSON")?
    } else {
        Vec::new()
    };

    let switch_result = serde_json::json!({
        "cameras": inputs.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "auto_switch": auto_switch,
        "min_shot_duration_secs": min_duration,
        "switch_points": switch_points,
        "output": output.display().to_string(),
        "status": "switch_ready",
        "message": "Switch list configured. Frame decoding pipeline needed for rendering.",
    });

    let json_str = serde_json::to_string_pretty(&switch_result)
        .context("Failed to serialize switch result")?;

    tokio::fs::write(output, json_str.as_bytes())
        .await
        .context("Failed to write output file")?;

    if json_output {
        println!("{}", json_str);
    } else {
        println!("{}", "Multi-Camera Switch".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Cameras:", inputs.len());
        println!("{:20} {}", "Auto-switch:", auto_switch);
        println!("{:20} {:.1}s", "Min duration:", min_duration);
        println!("{:20} {}", "Switch points:", switch_points.len());
        println!("{:20} {}", "Output:", output.display());
        println!();
        if !switch_points.is_empty() {
            println!("{}", "Switch Points".cyan().bold());
            println!("{}", "-".repeat(40));
            for sp in &switch_points {
                let time = sp.get("time").and_then(|t| t.as_f64()).unwrap_or(0.0);
                let cam = sp.get("camera").and_then(|c| c.as_u64()).unwrap_or(0);
                println!("  {:.2}s -> Camera {}", time, cam);
            }
        }
    }

    Ok(())
}

/// Composite multiple cameras into a single frame layout.
async fn composite_cameras(
    inputs: &[PathBuf],
    output: &PathBuf,
    layout: &str,
    width: Option<u32>,
    height: Option<u32>,
    spacing: u32,
    json_output: bool,
) -> Result<()> {
    let layout_desc = layout_description(layout)?;

    for input in inputs {
        if !input.exists() {
            return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
        }
    }

    let out_w = width.unwrap_or(1920);
    let out_h = height.unwrap_or(1080);

    // Use grid compositor to calculate layout
    let (grid_rows, grid_cols) =
        oximedia_multicam::composite::grid::GridCompositor::optimal_grid_for_angles(inputs.len());

    let mut grid = oximedia_multicam::composite::grid::GridCompositor::new(out_w, out_h);
    grid.set_spacing(spacing);
    let cells = grid.calculate_grid(grid_rows, grid_cols);

    let composite_result = serde_json::json!({
        "cameras": inputs.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "layout": layout,
        "layout_description": layout_desc,
        "output_width": out_w,
        "output_height": out_h,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "spacing": spacing,
        "cells": cells.iter().map(|(x, y, w, h)| {
            serde_json::json!({ "x": x, "y": y, "width": w, "height": h })
        }).collect::<Vec<_>>(),
        "output": output.display().to_string(),
        "status": "composite_ready",
        "message": "Layout computed. Frame decoding pipeline needed for rendering.",
    });

    let json_str = serde_json::to_string_pretty(&composite_result)
        .context("Failed to serialize composite result")?;

    tokio::fs::write(output, json_str.as_bytes())
        .await
        .context("Failed to write output file")?;

    if json_output {
        println!("{}", json_str);
    } else {
        println!("{}", "Multi-Camera Composite".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Cameras:", inputs.len());
        println!("{:20} {} ({})", "Layout:", layout, layout_desc);
        println!("{:20} {}x{}", "Output size:", out_w, out_h);
        println!("{:20} {}x{}", "Grid:", grid_rows, grid_cols);
        println!("{:20} {}px", "Spacing:", spacing);
        println!("{:20} {}", "Output:", output.display());
        println!();
        println!("{}", "Cell Layout".cyan().bold());
        println!("{}", "-".repeat(40));
        for (i, (cx, cy, cw, ch)) in cells.iter().enumerate() {
            if i < inputs.len() {
                println!("  Camera {}: {}x{} at ({}, {})", i, cw, ch, cx, cy);
            }
        }
    }

    Ok(())
}

/// Match colors across camera angles.
async fn color_match(
    reference: &PathBuf,
    inputs: &[PathBuf],
    output_dir: &PathBuf,
    json_output: bool,
) -> Result<()> {
    if !reference.exists() {
        return Err(anyhow::anyhow!(
            "Reference file not found: {}",
            reference.display()
        ));
    }
    for input in inputs {
        if !input.exists() {
            return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
        }
    }

    tokio::fs::create_dir_all(output_dir)
        .await
        .context("Failed to create output directory")?;

    let match_result = serde_json::json!({
        "reference": reference.display().to_string(),
        "cameras": inputs.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "output_dir": output_dir.display().to_string(),
        "status": "color_match_ready",
        "adjustments": inputs.iter().enumerate().map(|(i, _)| {
            serde_json::json!({
                "camera": i,
                "white_balance_shift": [0.0, 0.0, 0.0],
                "exposure_offset": 0.0,
                "saturation_factor": 1.0,
            })
        }).collect::<Vec<_>>(),
        "message": "Color matching configured. Frame decoding pipeline needed for analysis.",
    });

    let json_str = serde_json::to_string_pretty(&match_result)
        .context("Failed to serialize color match result")?;

    if json_output {
        println!("{}", json_str);
    } else {
        println!("{}", "Multi-Camera Color Match".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Reference:", reference.display());
        println!("{:20} {}", "Cameras:", inputs.len());
        println!("{:20} {}", "Output dir:", output_dir.display());
        println!();
        for (i, input) in inputs.iter().enumerate() {
            println!("  Camera {}: {}", i, input.display());
        }
        println!();
        println!(
            "{}",
            "Color matching requires frame decoding pipeline for analysis.".yellow()
        );
    }

    Ok(())
}

/// Export multi-camera timeline.
async fn export_timeline(
    timeline: &PathBuf,
    output: &PathBuf,
    format: &str,
    json_output: bool,
) -> Result<()> {
    if !timeline.exists() {
        return Err(anyhow::anyhow!(
            "Timeline file not found: {}",
            timeline.display()
        ));
    }

    match format {
        "multicam_edl" | "xml" | "json" => {}
        other => {
            return Err(anyhow::anyhow!(
                "Unknown export format '{}'. Expected: multicam_edl, xml, json",
                other
            ));
        }
    }

    let timeline_data = tokio::fs::read_to_string(timeline)
        .await
        .context("Failed to read timeline file")?;

    // For now, write through or convert format
    let export_data = match format {
        "json" => timeline_data,
        "multicam_edl" => {
            format!(
                "TITLE: OxiMedia Multi-Camera Export\nFCM: NON-DROP FRAME\n\n{}\n",
                "* Exported from OxiMedia multicam timeline"
            )
        }
        "xml" => {
            format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<multicam>\n  <source>OxiMedia</source>\n  <timeline>{}</timeline>\n</multicam>\n",
                timeline.display()
            )
        }
        _ => timeline_data,
    };

    tokio::fs::write(output, export_data.as_bytes())
        .await
        .context("Failed to write export file")?;

    if json_output {
        let result = serde_json::json!({
            "timeline": timeline.display().to_string(),
            "output": output.display().to_string(),
            "format": format,
            "status": "exported",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize export result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Timeline Export".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Timeline:", timeline.display());
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {}", "Format:", format);
    }

    Ok(())
}

/// List available multi-camera layouts.
async fn list_layouts(json_output: bool) -> Result<()> {
    let layouts = vec![
        ("grid", "Auto-sized grid layout (2x2, 3x3, etc.)"),
        ("pip", "Picture-in-picture with main view and corner inset"),
        ("side_by_side", "Horizontal split between two cameras"),
        ("stack", "Vertical stack of camera views"),
    ];

    if json_output {
        let items: Vec<serde_json::Value> = layouts
            .iter()
            .map(|(name, desc)| serde_json::json!({ "name": name, "description": desc }))
            .collect();
        let json_str =
            serde_json::to_string_pretty(&items).context("Failed to serialize layouts")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Available Multi-Camera Layouts".green().bold());
        println!("{}", "=".repeat(60));
        for (name, desc) in &layouts {
            println!("  {:20} {}", name.cyan(), desc);
        }
        println!();
        println!(
            "{}",
            "Use 'oximedia multicam composite --layout <name>' to apply a layout.".dimmed()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_sync_method() {
        assert!(validate_sync_method("audio").is_ok());
        assert!(validate_sync_method("timecode").is_ok());
        assert!(validate_sync_method("marker").is_ok());
        assert!(validate_sync_method("invalid").is_err());
    }

    #[test]
    fn test_layout_description() {
        assert!(layout_description("grid").is_ok());
        assert!(layout_description("pip").is_ok());
        assert!(layout_description("side_by_side").is_ok());
        assert!(layout_description("stack").is_ok());
        assert!(layout_description("unknown").is_err());
    }

    #[test]
    fn test_layout_description_values() {
        let desc = layout_description("grid").expect("valid layout");
        assert!(desc.contains("Grid"));
    }

    #[test]
    fn test_validate_sync_method_error_message() {
        let err = validate_sync_method("xyz").expect_err("should fail");
        let msg = format!("{}", err);
        assert!(msg.contains("xyz"));
    }

    #[test]
    fn test_layout_description_pip() {
        let desc = layout_description("pip").expect("valid layout");
        assert!(desc.contains("Picture"));
    }
}
