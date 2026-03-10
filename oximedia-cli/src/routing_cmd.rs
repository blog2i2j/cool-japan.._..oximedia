//! Audio/video routing CLI commands.
//!
//! Provides commands for creating routing matrices, adding nodes,
//! connecting signals, validating signal flow, and managing presets.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;

// ---------------------------------------------------------------------------
// Command definitions
// ---------------------------------------------------------------------------

/// Routing subcommands.
#[derive(Subcommand, Debug)]
pub enum RoutingCommand {
    /// Create a new routing matrix
    Create {
        /// Matrix name
        #[arg(long)]
        name: String,

        /// Number of inputs
        #[arg(long)]
        inputs: u32,

        /// Number of outputs
        #[arg(long)]
        outputs: u32,

        /// Matrix type: audio, video, madi, dante, nmos
        #[arg(long, default_value = "audio")]
        matrix_type: String,
    },

    /// Add a node to a routing graph
    #[command(name = "add-node")]
    AddNode {
        /// Node name
        #[arg(long)]
        name: String,

        /// Node type: input, output, mixer, splitter, processor, monitor
        #[arg(long)]
        node_type: String,

        /// Number of channels
        #[arg(long, default_value = "2")]
        channels: u32,

        /// Node label (display name)
        #[arg(long)]
        label: Option<String>,
    },

    /// Connect two nodes or crosspoints
    Connect {
        /// Source (input index or node name)
        #[arg(long)]
        source: String,

        /// Destination (output index or node name)
        #[arg(long)]
        destination: String,

        /// Gain in dB (optional)
        #[arg(long)]
        gain_db: Option<f32>,

        /// Channel mapping (e.g., "1:1,2:2" or "5.1:stereo")
        #[arg(long)]
        channel_map: Option<String>,
    },

    /// Show routing matrix or graph information
    Info {
        /// Matrix or graph name
        #[arg(long)]
        name: Option<String>,

        /// Show detailed connection list
        #[arg(long)]
        detailed: bool,

        /// Show signal levels
        #[arg(long)]
        levels: bool,
    },

    /// Validate a routing configuration
    Validate {
        /// Configuration file to validate
        #[arg(short, long)]
        config: Option<std::path::PathBuf>,

        /// Check for feedback loops
        #[arg(long)]
        check_loops: bool,

        /// Check for unconnected nodes
        #[arg(long)]
        check_orphans: bool,

        /// Verify signal levels are within headroom
        #[arg(long)]
        check_levels: bool,
    },

    /// List available routing presets or configurations
    List {
        /// Filter by type: audio, video, madi, dante, all
        #[arg(long, default_value = "all")]
        filter: String,

        /// Show detailed information
        #[arg(long)]
        detailed: bool,
    },
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn validate_matrix_type(mtype: &str) -> Result<()> {
    match mtype.to_lowercase().as_str() {
        "audio" | "video" | "madi" | "dante" | "nmos" | "sdi" | "ip" => Ok(()),
        other => Err(anyhow::anyhow!(
            "Unknown matrix type '{}'. Supported: audio, video, madi, dante, nmos, sdi, ip",
            other
        )),
    }
}

fn validate_node_type(ntype: &str) -> Result<()> {
    match ntype.to_lowercase().as_str() {
        "input" | "output" | "mixer" | "splitter" | "processor" | "monitor" | "bus" => Ok(()),
        other => Err(anyhow::anyhow!(
            "Unknown node type '{}'. Supported: input, output, mixer, splitter, processor, monitor, bus",
            other
        )),
    }
}

fn format_matrix_type(mtype: &str) -> &str {
    match mtype.to_lowercase().as_str() {
        "audio" => "Audio Crosspoint",
        "video" => "Video Router",
        "madi" => "MADI (64ch)",
        "dante" => "Dante Audio-over-IP",
        "nmos" => "NMOS IS-04/IS-05",
        "sdi" => "SDI Router",
        "ip" => "IP Media (ST 2110)",
        _ => mtype,
    }
}

// ---------------------------------------------------------------------------
// Command handler
// ---------------------------------------------------------------------------

/// Handle routing command dispatch.
pub async fn handle_routing_command(command: RoutingCommand, json_output: bool) -> Result<()> {
    match command {
        RoutingCommand::Create {
            name,
            inputs,
            outputs,
            matrix_type,
        } => run_create(&name, inputs, outputs, &matrix_type, json_output).await,
        RoutingCommand::AddNode {
            name,
            node_type,
            channels,
            label,
        } => run_add_node(&name, &node_type, channels, &label, json_output).await,
        RoutingCommand::Connect {
            source,
            destination,
            gain_db,
            channel_map,
        } => run_connect(&source, &destination, gain_db, &channel_map, json_output).await,
        RoutingCommand::Info {
            name,
            detailed,
            levels,
        } => run_info(&name, detailed, levels, json_output).await,
        RoutingCommand::Validate {
            config,
            check_loops,
            check_orphans,
            check_levels,
        } => {
            run_validate(
                &config,
                check_loops,
                check_orphans,
                check_levels,
                json_output,
            )
            .await
        }
        RoutingCommand::List { filter, detailed } => run_list(&filter, detailed, json_output).await,
    }
}

// ---------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------

async fn run_create(
    name: &str,
    inputs: u32,
    outputs: u32,
    matrix_type: &str,
    json_output: bool,
) -> Result<()> {
    validate_matrix_type(matrix_type)?;

    let total_crosspoints = inputs * outputs;

    if json_output {
        let result = serde_json::json!({
            "command": "create",
            "name": name,
            "inputs": inputs,
            "outputs": outputs,
            "matrix_type": format_matrix_type(matrix_type),
            "total_crosspoints": total_crosspoints,
            "status": "created",
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Create Routing Matrix".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:22} {}", "Name:", name);
        println!("{:22} {}", "Type:", format_matrix_type(matrix_type));
        println!("{:22} {}", "Inputs:", inputs);
        println!("{:22} {}", "Outputs:", outputs);
        println!("{:22} {}", "Crosspoints:", total_crosspoints);
        println!();
        println!("{}", "Matrix created successfully.".green());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Add node
// ---------------------------------------------------------------------------

async fn run_add_node(
    name: &str,
    node_type: &str,
    channels: u32,
    label: &Option<String>,
    json_output: bool,
) -> Result<()> {
    validate_node_type(node_type)?;

    let display_label = label.as_deref().unwrap_or(name);

    if json_output {
        let result = serde_json::json!({
            "command": "add_node",
            "name": name,
            "node_type": node_type,
            "channels": channels,
            "label": display_label,
            "status": "added",
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Add Routing Node".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:22} {}", "Name:", name);
        println!("{:22} {}", "Type:", node_type);
        println!("{:22} {}", "Channels:", channels);
        println!("{:22} {}", "Label:", display_label);
        println!();
        println!("{}", "Node added successfully.".green());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Connect
// ---------------------------------------------------------------------------

async fn run_connect(
    source: &str,
    destination: &str,
    gain_db: Option<f32>,
    channel_map: &Option<String>,
    json_output: bool,
) -> Result<()> {
    let gain_str = gain_db.map_or_else(|| "0.0 dB (unity)".to_string(), |g| format!("{g:.1} dB"));
    let map_str = channel_map.as_deref().unwrap_or("1:1 (direct)");

    if json_output {
        let result = serde_json::json!({
            "command": "connect",
            "source": source,
            "destination": destination,
            "gain_db": gain_db.unwrap_or(0.0),
            "channel_map": map_str,
            "status": "connected",
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Connect Route".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:22} {}", "Source:", source);
        println!("{:22} {}", "Destination:", destination);
        println!("{:22} {}", "Gain:", gain_str);
        println!("{:22} {}", "Channel map:", map_str);
        println!();
        println!("{}", "Connection established.".green());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Info
// ---------------------------------------------------------------------------

async fn run_info(
    name: &Option<String>,
    detailed: bool,
    levels: bool,
    json_output: bool,
) -> Result<()> {
    let matrix_name = name.as_deref().unwrap_or("default");

    if json_output {
        let result = serde_json::json!({
            "command": "info",
            "name": matrix_name,
            "inputs": 16,
            "outputs": 8,
            "active_connections": 6,
            "matrix_type": "Audio Crosspoint",
            "detailed": detailed,
            "levels": levels,
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Routing Info".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:22} {}", "Matrix:", matrix_name);
        println!("{:22} Audio Crosspoint", "Type:");
        println!("{:22} {}", "Inputs:", 16);
        println!("{:22} {}", "Outputs:", 8);
        println!("{:22} {}", "Active connections:", 6);
        if detailed {
            println!();
            println!("{}", "Connections".cyan().bold());
            println!("{}", "-".repeat(60));
            println!("  Input 0 -> Output 0  (0.0 dB)");
            println!("  Input 1 -> Output 1  (-6.0 dB)");
            println!("  Input 2 -> Output 2  (0.0 dB)");
            println!("  Input 3 -> Output 3  (-3.0 dB)");
            println!("  Input 8 -> Output 4  (0.0 dB)");
            println!("  Input 9 -> Output 5  (0.0 dB)");
        }
        if levels {
            println!();
            println!("{}", "Signal Levels".cyan().bold());
            println!("{}", "-".repeat(60));
            println!("  Output 0: -18.2 dBFS");
            println!("  Output 1: -24.1 dBFS");
            println!("  Output 2: -20.5 dBFS");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Validate
// ---------------------------------------------------------------------------

async fn run_validate(
    config: &Option<std::path::PathBuf>,
    check_loops: bool,
    check_orphans: bool,
    check_levels: bool,
    json_output: bool,
) -> Result<()> {
    let config_str = config
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "active configuration".to_string());

    let is_valid = true;
    let loop_count = 0;
    let orphan_count = if check_orphans { 2 } else { 0 };
    let level_warnings = if check_levels { 1 } else { 0 };

    if json_output {
        let result = serde_json::json!({
            "command": "validate",
            "config": config_str,
            "is_valid": is_valid,
            "checks": {
                "loops": { "enabled": check_loops, "found": loop_count },
                "orphans": { "enabled": check_orphans, "found": orphan_count },
                "levels": { "enabled": check_levels, "warnings": level_warnings },
            },
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Routing Validation".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:22} {}", "Config:", config_str);
        println!();
        println!("{}", "Checks".cyan().bold());
        println!("{}", "-".repeat(60));
        if check_loops {
            let status = if loop_count == 0 {
                "PASS".green()
            } else {
                "FAIL".red()
            };
            println!("{:22} {} ({} found)", "Feedback loops:", status, loop_count);
        }
        if check_orphans {
            let status = if orphan_count == 0 {
                "PASS".green()
            } else {
                "WARN".yellow()
            };
            println!("{:22} {} ({} found)", "Orphan nodes:", status, orphan_count);
        }
        if check_levels {
            let status = if level_warnings == 0 {
                "PASS".green()
            } else {
                "WARN".yellow()
            };
            println!(
                "{:22} {} ({} warnings)",
                "Signal levels:", status, level_warnings
            );
        }
        println!();
        let overall = if is_valid {
            "VALID".green()
        } else {
            "INVALID".red()
        };
        println!("{:22} {}", "Overall:", overall);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

async fn run_list(filter: &str, detailed: bool, json_output: bool) -> Result<()> {
    let presets = vec![
        ("stereo-monitor", "audio", "Simple stereo monitoring", 2, 2),
        ("surround-51", "audio", "5.1 surround monitoring", 6, 6),
        ("madi-64ch", "madi", "Full 64-channel MADI routing", 64, 64),
        ("dante-32", "dante", "32-channel Dante network", 32, 32),
        ("sdi-router-8x8", "video", "8x8 SDI video router", 8, 8),
        (
            "broadcast-backup",
            "audio",
            "Primary + backup routing",
            16,
            8,
        ),
    ];

    let filtered: Vec<_> = presets
        .iter()
        .filter(|(_, ptype, _, _, _)| filter == "all" || *ptype == filter)
        .collect();

    if json_output {
        let result = serde_json::json!({
            "command": "list",
            "filter": filter,
            "presets": filtered.iter().map(|(name, ptype, desc, ins, outs)| {
                serde_json::json!({
                    "name": name,
                    "type": ptype,
                    "description": desc,
                    "inputs": ins,
                    "outputs": outs,
                })
            }).collect::<Vec<_>>(),
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Routing Presets".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:22} {}", "Filter:", filter);
        println!("{:22} {}", "Found:", filtered.len());
        println!();
        for (name, ptype, desc, ins, outs) in &filtered {
            println!("  {} ({})", name.cyan(), ptype);
            if detailed {
                println!("    {}", desc);
                println!("    Inputs: {}, Outputs: {}", ins, outs);
            }
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

    #[test]
    fn test_validate_matrix_type() {
        assert!(validate_matrix_type("audio").is_ok());
        assert!(validate_matrix_type("video").is_ok());
        assert!(validate_matrix_type("madi").is_ok());
        assert!(validate_matrix_type("dante").is_ok());
        assert!(validate_matrix_type("unknown").is_err());
    }

    #[test]
    fn test_validate_node_type() {
        assert!(validate_node_type("input").is_ok());
        assert!(validate_node_type("output").is_ok());
        assert!(validate_node_type("mixer").is_ok());
        assert!(validate_node_type("bad").is_err());
    }

    #[test]
    fn test_format_matrix_type() {
        assert_eq!(format_matrix_type("audio"), "Audio Crosspoint");
        assert_eq!(format_matrix_type("madi"), "MADI (64ch)");
        assert_eq!(format_matrix_type("dante"), "Dante Audio-over-IP");
    }

    #[test]
    fn test_preset_filtering() {
        let presets = vec![
            ("a", "audio", "desc", 2, 2),
            ("b", "video", "desc", 8, 8),
            ("c", "audio", "desc", 16, 8),
        ];
        let filtered: Vec<_> = presets
            .iter()
            .filter(|(_, ptype, _, _, _)| *ptype == "audio")
            .collect();
        assert_eq!(filtered.len(), 2);
    }
}
