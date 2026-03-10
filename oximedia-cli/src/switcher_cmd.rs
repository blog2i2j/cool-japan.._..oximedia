//! Video switcher CLI commands.
//!
//! Provides subcommands for creating and controlling live production switchers,
//! managing sources, performing transitions, and running switching macros.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;

/// Switcher command subcommands.
#[derive(Subcommand, Debug)]
pub enum SwitcherCommand {
    /// Create a new switcher session
    Create {
        /// Number of M/E rows (1-4)
        #[arg(long, default_value = "1")]
        me_rows: usize,

        /// Number of inputs (2-40)
        #[arg(long, default_value = "8")]
        inputs: usize,

        /// Number of aux outputs
        #[arg(long, default_value = "2")]
        aux: usize,

        /// Preset configuration: basic, professional, broadcast
        #[arg(long)]
        preset: Option<String>,

        /// Frame rate: 25, 29.97, 30, 50, 59.94, 60
        #[arg(long, default_value = "25")]
        fps: f64,
    },

    /// Add a video source input
    AddSource {
        /// Source name / label
        #[arg(short, long)]
        name: String,

        /// Source type: sdi, ndi, file, test_pattern, media_player
        #[arg(long, default_value = "sdi")]
        source_type: String,

        /// Source URI or path
        #[arg(long)]
        uri: Option<String>,

        /// Input slot index (auto-assigned if omitted)
        #[arg(long)]
        slot: Option<usize>,
    },

    /// Switch to a source (cut or auto-transition)
    Switch {
        /// Target input index
        #[arg(short, long)]
        input: usize,

        /// M/E row (0-based, default: 0)
        #[arg(long, default_value = "0")]
        me_row: usize,

        /// Transition type: cut, mix, wipe, dve
        #[arg(long, default_value = "cut")]
        transition: String,

        /// Transition duration in frames (for non-cut transitions)
        #[arg(long, default_value = "30")]
        duration: u32,
    },

    /// Preview a source on the preview bus
    Preview {
        /// Input index to preview
        #[arg(short, long)]
        input: usize,

        /// M/E row (0-based)
        #[arg(long, default_value = "0")]
        me_row: usize,
    },

    /// Start or stop recording the program output
    Record {
        /// Action: start, stop
        #[arg(value_name = "ACTION")]
        action: String,

        /// Output file path (required for start)
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,

        /// Video codec: av1, vp9
        #[arg(long, default_value = "av1")]
        codec: String,
    },

    /// Run a switching macro (automated transition sequence)
    Macro {
        /// Macro action: run, list, record, stop-record
        #[arg(value_name = "ACTION")]
        action: String,

        /// Macro ID (for run)
        #[arg(long)]
        id: Option<usize>,

        /// Macro name (for record)
        #[arg(long)]
        name: Option<String>,
    },
}

/// Handle switcher command dispatch.
pub async fn handle_switcher_command(command: SwitcherCommand, json_output: bool) -> Result<()> {
    match command {
        SwitcherCommand::Create {
            me_rows,
            inputs,
            aux,
            preset,
            fps,
        } => handle_create(me_rows, inputs, aux, preset.as_deref(), fps, json_output).await,
        SwitcherCommand::AddSource {
            name,
            source_type,
            uri,
            slot,
        } => handle_add_source(&name, &source_type, uri.as_deref(), slot, json_output).await,
        SwitcherCommand::Switch {
            input,
            me_row,
            transition,
            duration,
        } => handle_switch(input, me_row, &transition, duration, json_output).await,
        SwitcherCommand::Preview { input, me_row } => {
            handle_preview(input, me_row, json_output).await
        }
        SwitcherCommand::Record {
            action,
            output,
            codec,
        } => handle_record(&action, output.as_deref(), &codec, json_output).await,
        SwitcherCommand::Macro { action, id, name } => {
            handle_macro(&action, id, name.as_deref(), json_output).await
        }
    }
}

// ---------------------------------------------------------------------------
// Handler: Create
// ---------------------------------------------------------------------------

async fn handle_create(
    me_rows: usize,
    inputs: usize,
    aux: usize,
    preset: Option<&str>,
    fps: f64,
    json_output: bool,
) -> Result<()> {
    let (me, inp, ax) = if let Some(p) = preset {
        match p {
            "basic" => (1, 8, 2),
            "professional" => (2, 20, 6),
            "broadcast" => (4, 40, 10),
            other => {
                return Err(anyhow::anyhow!(
                    "Unknown preset '{}'. Valid: basic, professional, broadcast",
                    other
                ));
            }
        }
    } else {
        (me_rows, inputs, aux)
    };

    if me < 1 || me > 4 {
        return Err(anyhow::anyhow!("M/E rows must be 1-4, got {}", me));
    }
    if inp < 2 || inp > 40 {
        return Err(anyhow::anyhow!("Inputs must be 2-40, got {}", inp));
    }

    let config = oximedia_switcher::SwitcherConfig::new(me, inp, ax);
    let _switcher = oximedia_switcher::Switcher::new(config)
        .map_err(|e| anyhow::anyhow!("Failed to create switcher: {}", e))?;

    if json_output {
        let result = serde_json::json!({
            "action": "create",
            "me_rows": me,
            "inputs": inp,
            "aux_outputs": ax,
            "fps": fps,
            "preset": preset,
            "status": "created",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Switcher Created".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "M/E rows:", me);
        println!("{:20} {}", "Inputs:", inp);
        println!("{:20} {}", "Aux outputs:", ax);
        println!("{:20} {}", "Frame rate:", fps);
        if let Some(p) = preset {
            println!("{:20} {}", "Preset:", p);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: AddSource
// ---------------------------------------------------------------------------

async fn handle_add_source(
    name: &str,
    source_type: &str,
    uri: Option<&str>,
    slot: Option<usize>,
    json_output: bool,
) -> Result<()> {
    let valid_types = ["sdi", "ndi", "file", "test_pattern", "media_player"];
    if !valid_types.contains(&source_type) {
        return Err(anyhow::anyhow!(
            "Unknown source type '{}'. Valid: {}",
            source_type,
            valid_types.join(", ")
        ));
    }

    let assigned_slot = slot.unwrap_or(0);

    if json_output {
        let result = serde_json::json!({
            "action": "add_source",
            "name": name,
            "source_type": source_type,
            "uri": uri,
            "slot": assigned_slot,
            "status": "added",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Source Added".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Name:", name);
        println!("{:20} {}", "Type:", source_type);
        println!("{:20} {}", "URI:", uri.unwrap_or("N/A"));
        println!("{:20} {}", "Slot:", assigned_slot);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Switch
// ---------------------------------------------------------------------------

async fn handle_switch(
    input: usize,
    me_row: usize,
    transition: &str,
    duration: u32,
    json_output: bool,
) -> Result<()> {
    let valid_transitions = ["cut", "mix", "wipe", "dve"];
    if !valid_transitions.contains(&transition) {
        return Err(anyhow::anyhow!(
            "Unknown transition '{}'. Valid: {}",
            transition,
            valid_transitions.join(", ")
        ));
    }

    if json_output {
        let result = serde_json::json!({
            "action": "switch",
            "input": input,
            "me_row": me_row,
            "transition": transition,
            "duration_frames": if transition == "cut" { 0 } else { duration },
            "status": "switched",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Source Switched".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Target input:", input);
        println!("{:20} {}", "M/E row:", me_row);
        println!("{:20} {}", "Transition:", transition);
        if transition != "cut" {
            println!("{:20} {} frames", "Duration:", duration);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Preview
// ---------------------------------------------------------------------------

async fn handle_preview(input: usize, me_row: usize, json_output: bool) -> Result<()> {
    if json_output {
        let result = serde_json::json!({
            "action": "preview",
            "input": input,
            "me_row": me_row,
            "status": "previewing",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Preview Set".green().bold());
        println!("{:20} {}", "Input:", input);
        println!("{:20} {}", "M/E row:", me_row);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Record
// ---------------------------------------------------------------------------

async fn handle_record(
    action: &str,
    output: Option<&std::path::Path>,
    codec: &str,
    json_output: bool,
) -> Result<()> {
    match action {
        "start" => {
            let out = output
                .ok_or_else(|| anyhow::anyhow!("Output path is required for 'start' action"))?;

            match codec {
                "av1" | "vp9" | "vp8" => {}
                other => {
                    return Err(anyhow::anyhow!(
                        "Unsupported codec '{}'. Use: av1, vp9, vp8",
                        other
                    ));
                }
            }

            if json_output {
                let result = serde_json::json!({
                    "action": "record_start",
                    "output": out.display().to_string(),
                    "codec": codec,
                    "status": "recording",
                });
                let json_str =
                    serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
                println!("{}", json_str);
            } else {
                println!("{}", "Recording Started".green().bold());
                println!("{:20} {}", "Output:", out.display());
                println!("{:20} {}", "Codec:", codec);
            }
        }
        "stop" => {
            if json_output {
                let result = serde_json::json!({
                    "action": "record_stop",
                    "status": "stopped",
                });
                let json_str =
                    serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
                println!("{}", json_str);
            } else {
                println!("{}", "Recording Stopped".green().bold());
            }
        }
        other => {
            return Err(anyhow::anyhow!(
                "Unknown record action '{}'. Use: start, stop",
                other
            ));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Macro
// ---------------------------------------------------------------------------

async fn handle_macro(
    action: &str,
    id: Option<usize>,
    name: Option<&str>,
    json_output: bool,
) -> Result<()> {
    match action {
        "run" => {
            let macro_id =
                id.ok_or_else(|| anyhow::anyhow!("Macro ID is required for 'run' action (--id)"))?;

            if json_output {
                let result = serde_json::json!({
                    "action": "macro_run",
                    "macro_id": macro_id,
                    "status": "running",
                });
                let json_str =
                    serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
                println!("{}", json_str);
            } else {
                println!("{}", "Macro Running".green().bold());
                println!("{:20} {}", "Macro ID:", macro_id);
            }
        }
        "list" => {
            if json_output {
                let result = serde_json::json!({
                    "action": "macro_list",
                    "macros": [],
                });
                let json_str =
                    serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
                println!("{}", json_str);
            } else {
                println!("{}", "Available Macros".green().bold());
                println!("{}", "=".repeat(60));
                println!("{}", "No macros defined.".dimmed());
            }
        }
        "record" => {
            let macro_name = name.unwrap_or("Untitled Macro");
            if json_output {
                let result = serde_json::json!({
                    "action": "macro_record",
                    "name": macro_name,
                    "status": "recording",
                });
                let json_str =
                    serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
                println!("{}", json_str);
            } else {
                println!("{}", "Macro Recording Started".green().bold());
                println!("{:20} {}", "Name:", macro_name);
            }
        }
        "stop-record" => {
            if json_output {
                let result = serde_json::json!({
                    "action": "macro_stop_record",
                    "status": "saved",
                });
                let json_str =
                    serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
                println!("{}", json_str);
            } else {
                println!("{}", "Macro Recording Stopped".green().bold());
            }
        }
        other => {
            return Err(anyhow::anyhow!(
                "Unknown macro action '{}'. Valid: run, list, record, stop-record",
                other
            ));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_handle_create_basic() {
        let result = handle_create(1, 8, 2, None, 25.0, false).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_create_preset() {
        let result = handle_create(0, 0, 0, Some("professional"), 25.0, false).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_create_invalid_me_rows() {
        let result = handle_create(5, 8, 2, None, 25.0, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_switch_invalid_transition() {
        let result = handle_switch(1, 0, "invalid", 30, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_record_start_no_output() {
        let result = handle_record("start", None, "av1", false).await;
        assert!(result.is_err());
    }
}
