//! Video-over-IP CLI commands.
//!
//! Provides commands for sending, receiving, discovering, and monitoring
//! professional video-over-IP streams using the `oximedia-videoip` crate.
//! Supports RTP, SRT, and RIST protocols.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Video-over-IP command subcommands.
#[derive(Subcommand, Debug)]
pub enum VideoIpCommand {
    /// Send a media file as a video-over-IP stream
    Send {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// Destination address (IP:port)
        #[arg(long)]
        address: String,

        /// Transport protocol: rtp, srt, rist
        #[arg(long, default_value = "rtp")]
        protocol: String,

        /// Stream source name
        #[arg(long)]
        name: Option<String>,

        /// Loop the input continuously
        #[arg(long, default_value = "false")]
        looping: bool,

        /// Target bitrate in kbps
        #[arg(long)]
        bitrate: Option<u32>,

        /// Enable FEC (Forward Error Correction)
        #[arg(long)]
        fec: bool,

        /// Video resolution (e.g., "1920x1080")
        #[arg(long)]
        resolution: Option<String>,

        /// Frame rate (e.g., 30, 59.94)
        #[arg(long)]
        fps: Option<f64>,
    },

    /// Receive a video-over-IP stream and save to file
    Receive {
        /// Listen address (IP:port)
        #[arg(long)]
        address: String,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Transport protocol: rtp, srt, rist
        #[arg(long, default_value = "rtp")]
        protocol: String,

        /// Recording duration in seconds (unlimited if omitted)
        #[arg(long)]
        duration: Option<f64>,

        /// Jitter buffer size in milliseconds
        #[arg(long, default_value = "50")]
        jitter_buffer_ms: u32,
    },

    /// Discover available video-over-IP streams on the network
    Discover {
        /// Discovery method: sap, mdns
        #[arg(long, default_value = "mdns")]
        method: String,

        /// Discovery timeout in seconds
        #[arg(long, default_value = "5")]
        timeout: u64,
    },

    /// Monitor a video-over-IP stream and display statistics
    Monitor {
        /// Stream address (IP:port)
        #[arg(long)]
        address: String,

        /// Transport protocol: rtp, srt, rist
        #[arg(long, default_value = "rtp")]
        protocol: String,

        /// Stats refresh interval in seconds
        #[arg(long, default_value = "2")]
        interval: u64,

        /// Show detailed per-packet statistics
        #[arg(long)]
        detailed: bool,
    },
}

/// Handle video-over-IP command dispatch.
pub async fn handle_videoip_command(command: VideoIpCommand, json_output: bool) -> Result<()> {
    match command {
        VideoIpCommand::Send {
            input,
            address,
            protocol,
            name,
            looping,
            bitrate,
            fec,
            resolution,
            fps,
        } => {
            send_stream(
                &input,
                &address,
                &protocol,
                name.as_deref(),
                looping,
                bitrate,
                fec,
                resolution.as_deref(),
                fps,
                json_output,
            )
            .await
        }
        VideoIpCommand::Receive {
            address,
            output,
            protocol,
            duration,
            jitter_buffer_ms,
        } => {
            receive_stream(
                &address,
                &output,
                &protocol,
                duration,
                jitter_buffer_ms,
                json_output,
            )
            .await
        }
        VideoIpCommand::Discover { method, timeout } => {
            discover_streams(&method, timeout, json_output).await
        }
        VideoIpCommand::Monitor {
            address,
            protocol,
            interval,
            detailed,
        } => monitor_stream(&address, &protocol, interval, detailed, json_output).await,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_protocol(protocol: &str) -> Result<()> {
    match protocol {
        "rtp" | "srt" | "rist" => Ok(()),
        other => Err(anyhow::anyhow!(
            "Unsupported protocol '{}'. Supported: rtp, srt, rist",
            other
        )),
    }
}

fn parse_resolution(res: &str) -> Result<(u32, u32)> {
    let parts: Vec<&str> = res.split('x').collect();
    if parts.len() != 2 {
        return Err(anyhow::anyhow!(
            "Invalid resolution '{}'. Expected format: WIDTHxHEIGHT (e.g., 1920x1080)",
            res
        ));
    }
    let width: u32 = parts[0]
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid width in resolution: {}", parts[0]))?;
    let height: u32 = parts[1]
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid height in resolution: {}", parts[1]))?;
    Ok((width, height))
}

// ---------------------------------------------------------------------------
// Send
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn send_stream(
    input: &PathBuf,
    address: &str,
    protocol: &str,
    name: Option<&str>,
    looping: bool,
    bitrate: Option<u32>,
    fec: bool,
    resolution: Option<&str>,
    fps: Option<f64>,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }
    validate_protocol(protocol)?;

    let file_size = std::fs::metadata(input)
        .context("Failed to read file metadata")?
        .len();

    let (width, height) = if let Some(res) = resolution {
        parse_resolution(res)?
    } else {
        (1920, 1080) // Default to Full HD
    };

    let frame_rate = fps.unwrap_or(30.0);

    // Build VideoIP config
    let video_config = oximedia_videoip::VideoConfig::new(width, height, frame_rate)
        .map_err(|e| anyhow::anyhow!("Invalid video config: {}", e))?;
    let audio_config = oximedia_videoip::AudioConfig::new(48000, 2)
        .map_err(|e| anyhow::anyhow!("Invalid audio config: {}", e))?;

    let source_name = name.unwrap_or("OxiMedia VideoIP Source");

    if json_output {
        let result = serde_json::json!({
            "command": "videoip send",
            "input": input.display().to_string(),
            "file_size": file_size,
            "address": address,
            "protocol": protocol,
            "source_name": source_name,
            "looping": looping,
            "bitrate_kbps": bitrate,
            "fec_enabled": fec,
            "video": {
                "width": width,
                "height": height,
                "fps": frame_rate,
            },
            "audio": {
                "sample_rate": 48000,
                "channels": 2,
            },
            "status": "ready",
            "message": "VideoIP sender configured; awaiting frame decoding pipeline integration",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "VideoIP Send".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:22} {}", "Input:", input.display());
        println!("{:22} {} bytes", "File size:", file_size);
        println!("{:22} {}", "Destination:", address);
        println!("{:22} {}", "Protocol:", protocol.to_uppercase());
        println!("{:22} {}", "Source name:", source_name);
        println!("{:22} {}", "Looping:", looping);
        if let Some(br) = bitrate {
            println!("{:22} {} kbps", "Bitrate:", br);
        }
        println!("{:22} {}", "FEC:", fec);
        println!();

        println!("{}", "Video Configuration".cyan().bold());
        println!("{}", "-".repeat(60));
        println!(
            "  Resolution:   {}x{}",
            video_config.format.resolution.width, video_config.format.resolution.height
        );
        println!("  Frame rate:   {:.3} fps", frame_rate);
        println!();

        println!("{}", "Audio Configuration".cyan().bold());
        println!("{}", "-".repeat(60));
        println!("  Sample rate:  {} Hz", audio_config.format.sample_rate);
        println!("  Channels:     {}", audio_config.format.channels);
        println!();

        println!(
            "{}",
            "Note: Frame decoding pipeline not yet integrated.".yellow()
        );
        println!(
            "{}",
            "VideoIP sender is configured; frame decoding will enable streaming.".dimmed()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Receive
// ---------------------------------------------------------------------------

async fn receive_stream(
    address: &str,
    output: &PathBuf,
    protocol: &str,
    duration: Option<f64>,
    jitter_buffer_ms: u32,
    json_output: bool,
) -> Result<()> {
    validate_protocol(protocol)?;

    if json_output {
        let result = serde_json::json!({
            "command": "videoip receive",
            "address": address,
            "output": output.display().to_string(),
            "protocol": protocol,
            "duration": duration,
            "jitter_buffer_ms": jitter_buffer_ms,
            "status": "ready",
            "message": "VideoIP receiver configured; awaiting connection",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "VideoIP Receive".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:22} {}", "Listen address:", address);
        println!("{:22} {}", "Output:", output.display());
        println!("{:22} {}", "Protocol:", protocol.to_uppercase());
        if let Some(dur) = duration {
            println!("{:22} {:.1}s", "Duration:", dur);
        } else {
            println!("{:22} unlimited", "Duration:");
        }
        println!("{:22} {} ms", "Jitter buffer:", jitter_buffer_ms);
        println!();

        println!("{}", "Protocol Details".cyan().bold());
        println!("{}", "-".repeat(60));
        match protocol {
            "rtp" => {
                println!("  Type:          Real-time Transport Protocol");
                println!("  Reliability:   Best-effort (UDP)");
                println!("  Use case:      Low-latency, local network");
            }
            "srt" => {
                println!("  Type:          Secure Reliable Transport");
                println!("  Reliability:   ARQ-based retransmission");
                println!("  Use case:      Internet streaming, cloud ingest");
            }
            "rist" => {
                println!("  Type:          Reliable Internet Stream Transport");
                println!("  Reliability:   ARQ with bonding support");
                println!("  Use case:      Broadcast contribution links");
            }
            _ => {}
        }
        println!();

        println!(
            "{}",
            "VideoIP receiver configured and ready to connect.".dimmed()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Discover
// ---------------------------------------------------------------------------

async fn discover_streams(method: &str, timeout_secs: u64, json_output: bool) -> Result<()> {
    match method {
        "sap" | "mdns" => {}
        other => {
            return Err(anyhow::anyhow!(
                "Unsupported discovery method '{}'. Supported: sap, mdns",
                other
            ));
        }
    }

    // Use VideoIP discovery client
    let client = oximedia_videoip::discovery::DiscoveryClient::new()
        .map_err(|e| anyhow::anyhow!("Failed to create discovery client: {}", e))?;

    let sources = client
        .discover_all(timeout_secs)
        .await
        .map_err(|e| anyhow::anyhow!("Discovery failed: {}", e))?;

    if json_output {
        let source_list: Vec<serde_json::Value> = sources
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "address": s.address.to_string(),
                    "port": s.port,
                })
            })
            .collect();
        let result = serde_json::json!({
            "command": "videoip discover",
            "method": method,
            "timeout_seconds": timeout_secs,
            "sources_found": sources.len(),
            "sources": source_list,
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "VideoIP Stream Discovery".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Method:", method.to_uppercase());
        println!("{:20} {} seconds", "Timeout:", timeout_secs);
        println!();

        if sources.is_empty() {
            println!(
                "{}",
                "No video-over-IP streams found on the network.".yellow()
            );
        } else {
            println!("Found {} stream(s):", sources.len().to_string().cyan());
            println!("{}", "-".repeat(60));
            for (i, source) in sources.iter().enumerate() {
                println!(
                    "  {}. {} ({}:{})",
                    i + 1,
                    source.name.green().bold(),
                    source.address,
                    source.port,
                );
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Monitor
// ---------------------------------------------------------------------------

async fn monitor_stream(
    address: &str,
    protocol: &str,
    interval: u64,
    detailed: bool,
    json_output: bool,
) -> Result<()> {
    validate_protocol(protocol)?;

    // Create a stats tracker to show monitoring capabilities
    let stats = oximedia_videoip::stats::NetworkStats::new();

    if json_output {
        let result = serde_json::json!({
            "command": "videoip monitor",
            "address": address,
            "protocol": protocol,
            "refresh_interval_seconds": interval,
            "detailed": detailed,
            "initial_stats": {
                "packets_sent": stats.packets_sent,
                "packets_received": stats.packets_received,
                "bytes_sent": stats.bytes_sent,
                "bytes_received": stats.bytes_received,
                "packets_lost": stats.packets_lost,
                "packets_recovered": stats.packets_recovered,
                "loss_rate": stats.loss_rate,
                "jitter_us": stats.jitter_us,
                "avg_rtt_us": stats.avg_rtt_us,
            },
            "status": "monitoring",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "VideoIP Stream Monitor".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:22} {}", "Address:", address);
        println!("{:22} {}", "Protocol:", protocol.to_uppercase());
        println!("{:22} {}s", "Refresh interval:", interval);
        println!("{:22} {}", "Detailed stats:", detailed);
        println!();

        println!("{}", "Network Statistics".cyan().bold());
        println!("{}", "-".repeat(60));
        println!("  Packets sent:      {}", stats.packets_sent);
        println!("  Packets received:  {}", stats.packets_received);
        println!("  Bytes sent:        {}", stats.bytes_sent);
        println!("  Bytes received:    {}", stats.bytes_received);
        println!("  Packets lost:      {}", stats.packets_lost);
        println!("  Packets recovered: {}", stats.packets_recovered);
        println!("  Loss rate:         {:.4}%", stats.loss_rate * 100.0);
        println!("  Jitter:            {} us", stats.jitter_us);
        println!("  RTT:               {} us", stats.avg_rtt_us);

        if detailed {
            println!();
            println!("{}", "Extended Statistics".cyan().bold());
            println!("{}", "-".repeat(60));
            println!("  Out of order:      {}", stats.packets_out_of_order);
            println!("  Duplicates:        {}", stats.packets_duplicate);
            println!("  Current bitrate:   {} bps", stats.current_bitrate);
        }

        println!();
        println!(
            "{}",
            "Stream monitor initialized; active stream required for real-time stats.".dimmed()
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
    fn test_validate_protocol_valid() {
        assert!(validate_protocol("rtp").is_ok());
        assert!(validate_protocol("srt").is_ok());
        assert!(validate_protocol("rist").is_ok());
    }

    #[test]
    fn test_validate_protocol_invalid() {
        let result = validate_protocol("http");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_resolution_valid() {
        let (w, h) = parse_resolution("1920x1080").expect("valid resolution");
        assert_eq!(w, 1920);
        assert_eq!(h, 1080);

        let (w, h) = parse_resolution("3840x2160").expect("valid resolution");
        assert_eq!(w, 3840);
        assert_eq!(h, 2160);
    }

    #[test]
    fn test_parse_resolution_invalid() {
        assert!(parse_resolution("invalid").is_err());
        assert!(parse_resolution("1920:1080").is_err());
        assert!(parse_resolution("abcxdef").is_err());
    }

    #[test]
    fn test_send_rejects_missing_input() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime");
        let result = rt.block_on(send_stream(
            &PathBuf::from("/nonexistent/file.mkv"),
            "192.168.1.100:5004",
            "rtp",
            None,
            false,
            None,
            false,
            None,
            None,
            false,
        ));
        assert!(result.is_err());
    }
}
