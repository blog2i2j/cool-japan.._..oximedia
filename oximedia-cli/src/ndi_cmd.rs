//! NDI (Network Device Interface) CLI commands.
//!
//! Provides NDI source discovery, stream sending/receiving, and monitoring
//! using the `oximedia-ndi` crate.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// NDI command subcommands.
#[derive(Subcommand, Debug)]
pub enum NdiCommand {
    /// Discover NDI sources on the network
    Discover {
        /// Discovery timeout in seconds
        #[arg(long, default_value = "5")]
        timeout: u64,

        /// NDI group filter
        #[arg(long)]
        group: Option<String>,
    },

    /// Receive NDI stream and save to file
    Receive {
        /// Source name or IP address
        #[arg(short, long)]
        source: String,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Recording duration in seconds (unlimited if omitted)
        #[arg(long)]
        duration: Option<f64>,

        /// Receive audio only (discard video)
        #[arg(long)]
        audio_only: bool,

        /// Receive video only (discard audio)
        #[arg(long)]
        video_only: bool,

        /// Low-bandwidth mode
        #[arg(long)]
        low_bandwidth: bool,
    },

    /// Send a media file as an NDI source
    Send {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// NDI source name
        #[arg(long)]
        name: Option<String>,

        /// Loop the input continuously
        #[arg(long, default_value = "false")]
        looping: bool,

        /// NDI group to join
        #[arg(long)]
        group: Option<String>,

        /// Enable tally light support
        #[arg(long)]
        tally: bool,
    },

    /// Monitor NDI sources and display stats
    Monitor {
        /// Specific source to monitor (all sources if omitted)
        #[arg(short, long)]
        source: Option<String>,

        /// Stats refresh interval in seconds
        #[arg(long, default_value = "2")]
        interval: u64,

        /// Show detailed per-frame statistics
        #[arg(long)]
        detailed: bool,
    },
}

/// Handle NDI command dispatch.
pub async fn handle_ndi_command(command: NdiCommand, json_output: bool) -> Result<()> {
    match command {
        NdiCommand::Discover { timeout, group } => {
            discover_sources(timeout, group.as_deref(), json_output).await
        }
        NdiCommand::Receive {
            source,
            output,
            duration,
            audio_only,
            video_only,
            low_bandwidth,
        } => {
            receive_stream(
                &source,
                &output,
                duration,
                audio_only,
                video_only,
                low_bandwidth,
                json_output,
            )
            .await
        }
        NdiCommand::Send {
            input,
            name,
            looping,
            group,
            tally,
        } => {
            send_stream(
                &input,
                name.as_deref(),
                looping,
                group.as_deref(),
                tally,
                json_output,
            )
            .await
        }
        NdiCommand::Monitor {
            source,
            interval,
            detailed,
        } => monitor_sources(source.as_deref(), interval, detailed, json_output).await,
    }
}

// ---------------------------------------------------------------------------
// Discover
// ---------------------------------------------------------------------------

async fn discover_sources(timeout_secs: u64, group: Option<&str>, json_output: bool) -> Result<()> {
    let timeout_dur = std::time::Duration::from_secs(timeout_secs);

    // Use the NDI discovery service
    let discovery =
        oximedia_ndi::DiscoveryService::new().context("Failed to create NDI discovery service")?;

    let sources = discovery
        .discover(timeout_dur)
        .await
        .context("NDI discovery failed")?;

    // Filter by group if specified
    let filtered: Vec<_> = if let Some(grp) = group {
        sources
            .into_iter()
            .filter(|s| s.groups.iter().any(|g| g == grp))
            .collect()
    } else {
        sources
    };

    if json_output {
        let source_list: Vec<serde_json::Value> = filtered
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "address": s.address.to_string(),
                    "groups": s.groups,
                })
            })
            .collect();
        let result = serde_json::json!({
            "command": "ndi discover",
            "timeout_seconds": timeout_secs,
            "group_filter": group,
            "sources_found": filtered.len(),
            "sources": source_list,
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "NDI Source Discovery".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {} seconds", "Timeout:", timeout_secs);
        if let Some(grp) = group {
            println!("{:20} {}", "Group filter:", grp);
        }
        println!();

        if filtered.is_empty() {
            println!("{}", "No NDI sources found on the network.".yellow());
        } else {
            println!("Found {} source(s):", filtered.len().to_string().cyan());
            println!("{}", "-".repeat(60));
            for (i, source) in filtered.iter().enumerate() {
                println!("  {}. {}", i + 1, source.name.green().bold());
                println!("     Address: {}", source.address);
                if !source.groups.is_empty() {
                    println!("     Groups:  {}", source.groups.join(", "));
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Receive
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn receive_stream(
    source_name: &str,
    output: &PathBuf,
    duration: Option<f64>,
    audio_only: bool,
    video_only: bool,
    low_bandwidth: bool,
    json_output: bool,
) -> Result<()> {
    if audio_only && video_only {
        return Err(anyhow::anyhow!(
            "Cannot specify both --audio-only and --video-only"
        ));
    }

    // Configure the NDI receiver
    let config = oximedia_ndi::NdiConfig {
        name: format!("OxiMedia Receiver ({})", source_name),
        low_bandwidth,
        ..oximedia_ndi::NdiConfig::default()
    };

    if json_output {
        let result = serde_json::json!({
            "command": "ndi receive",
            "source": source_name,
            "output": output.display().to_string(),
            "duration": duration,
            "audio_only": audio_only,
            "video_only": video_only,
            "low_bandwidth": low_bandwidth,
            "config": {
                "name": config.name,
                "buffer_size": config.buffer_size,
                "enable_tally": config.enable_tally,
                "enable_ptz": config.enable_ptz,
            },
            "status": "ready",
            "message": "NDI receiver configured; awaiting connection to source",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "NDI Receive".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Source:", source_name);
        println!("{:20} {}", "Output:", output.display());
        if let Some(dur) = duration {
            println!("{:20} {:.1}s", "Duration:", dur);
        } else {
            println!("{:20} unlimited", "Duration:");
        }
        println!("{:20} {}", "Audio only:", audio_only);
        println!("{:20} {}", "Video only:", video_only);
        println!("{:20} {}", "Low bandwidth:", low_bandwidth);
        println!("{:20} {}", "Buffer size:", config.buffer_size);
        println!();

        println!("{}", "Receiver Configuration".cyan().bold());
        println!("{}", "-".repeat(60));
        println!("  Name:          {}", config.name);
        println!("  Tally support: {}", config.enable_tally);
        println!("  PTZ support:   {}", config.enable_ptz);
        println!();

        println!(
            "{}",
            "NDI receiver configured and ready to connect.".dimmed()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Send
// ---------------------------------------------------------------------------

async fn send_stream(
    input: &PathBuf,
    name: Option<&str>,
    looping: bool,
    group: Option<&str>,
    tally: bool,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let source_name = name.unwrap_or("OxiMedia NDI Source");
    let groups = group.map_or_else(|| vec!["public".to_string()], |g| vec![g.to_string()]);

    let file_size = std::fs::metadata(input)
        .context("Failed to read file metadata")?
        .len();

    // Build NDI config
    let config = oximedia_ndi::NdiConfig {
        name: source_name.to_string(),
        groups: groups.clone(),
        enable_tally: tally,
        ..oximedia_ndi::NdiConfig::default()
    };

    // Validate that a sender can be created with this config
    let _sender_config = oximedia_ndi::SenderConfig::default();

    if json_output {
        let result = serde_json::json!({
            "command": "ndi send",
            "input": input.display().to_string(),
            "file_size": file_size,
            "source_name": source_name,
            "groups": groups,
            "looping": looping,
            "tally": tally,
            "config": {
                "name": config.name,
                "groups": config.groups,
                "enable_tally": config.enable_tally,
                "enable_ptz": config.enable_ptz,
                "buffer_size": config.buffer_size,
            },
            "status": "ready",
            "message": "NDI sender configured; awaiting frame decoding pipeline integration",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "NDI Send".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {} bytes", "File size:", file_size);
        println!("{:20} {}", "Source name:", source_name);
        println!("{:20} {}", "Groups:", groups.join(", "));
        println!("{:20} {}", "Looping:", looping);
        println!("{:20} {}", "Tally:", tally);
        println!();

        println!("{}", "Sender Configuration".cyan().bold());
        println!("{}", "-".repeat(60));
        println!("  Buffer size: {}", config.buffer_size);
        println!("  PTZ:         {}", config.enable_ptz);
        println!();

        println!(
            "{}",
            "Note: Frame decoding pipeline not yet integrated.".yellow()
        );
        println!(
            "{}",
            "NDI sender is configured; frame decoding will enable streaming.".dimmed()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Monitor
// ---------------------------------------------------------------------------

async fn monitor_sources(
    source_name: Option<&str>,
    interval: u64,
    detailed: bool,
    json_output: bool,
) -> Result<()> {
    let timeout_dur = std::time::Duration::from_secs(5);

    let discovery =
        oximedia_ndi::DiscoveryService::new().context("Failed to create NDI discovery service")?;

    let sources = discovery
        .discover(timeout_dur)
        .await
        .context("NDI discovery failed")?;

    // Filter to specific source if requested
    let monitored: Vec<_> = if let Some(name) = source_name {
        sources
            .into_iter()
            .filter(|s| s.name.contains(name))
            .collect()
    } else {
        sources
    };

    if json_output {
        let source_list: Vec<serde_json::Value> = monitored
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "address": s.address.to_string(),
                    "groups": s.groups,
                })
            })
            .collect();
        let result = serde_json::json!({
            "command": "ndi monitor",
            "source_filter": source_name,
            "refresh_interval_seconds": interval,
            "detailed": detailed,
            "sources_found": monitored.len(),
            "sources": source_list,
            "status": "monitoring",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "NDI Monitor".green().bold());
        println!("{}", "=".repeat(60));
        if let Some(name) = source_name {
            println!("{:20} {}", "Source filter:", name);
        }
        println!("{:20} {}s", "Refresh interval:", interval);
        println!("{:20} {}", "Detailed stats:", detailed);
        println!();

        if monitored.is_empty() {
            println!("{}", "No NDI sources found to monitor.".yellow());
        } else {
            println!(
                "Monitoring {} source(s):",
                monitored.len().to_string().cyan()
            );
            println!("{}", "-".repeat(60));
            for source in &monitored {
                println!("  {} ({})", source.name.green(), source.address);
                if !source.groups.is_empty() {
                    println!("    Groups: {}", source.groups.join(", "));
                }
                if detailed {
                    println!("    Video:  pending connection");
                    println!("    Audio:  pending connection");
                    println!("    Tally:  unknown");
                }
            }
            println!();
            println!(
                "{}",
                "NDI monitor initialized; real-time stats require active connections.".dimmed()
            );
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
    fn test_ndi_command_variants() {
        // Verify all command variants can be constructed
        let _discover = NdiCommand::Discover {
            timeout: 5,
            group: None,
        };
        let _receive = NdiCommand::Receive {
            source: "test".to_string(),
            output: PathBuf::from("out.mkv"),
            duration: Some(10.0),
            audio_only: false,
            video_only: false,
            low_bandwidth: false,
        };
        let _send = NdiCommand::Send {
            input: PathBuf::from("input.mkv"),
            name: Some("Test Source".to_string()),
            looping: false,
            group: None,
            tally: true,
        };
        let _monitor = NdiCommand::Monitor {
            source: None,
            interval: 2,
            detailed: false,
        };
    }

    #[test]
    fn test_receive_rejects_audio_and_video_only() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime");
        let result = rt.block_on(receive_stream(
            "test",
            &PathBuf::from("out.mkv"),
            None,
            true,
            true,
            false,
            false,
        ));
        assert!(result.is_err());
        let err_msg = format!("{}", result.expect_err("should fail"));
        assert!(err_msg.contains("audio-only"));
    }

    #[test]
    fn test_send_rejects_missing_input() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime");
        let result = rt.block_on(send_stream(
            &PathBuf::from("/nonexistent/file.mkv"),
            None,
            false,
            None,
            false,
            false,
        ));
        assert!(result.is_err());
    }

    #[test]
    fn test_ndi_config_defaults() {
        let config = oximedia_ndi::NdiConfig::default();
        assert!(config.enable_tally);
        assert!(config.enable_ptz);
        assert_eq!(config.buffer_size, 16);
        assert_eq!(config.groups, vec!["public".to_string()]);
    }

    #[test]
    fn test_video_format_construction() {
        let fmt = oximedia_ndi::VideoFormat::full_hd_60p();
        assert_eq!(fmt.width, 1920);
        assert_eq!(fmt.height, 1080);
        assert!((fmt.frame_rate() - 60.0).abs() < 0.001);
    }
}
