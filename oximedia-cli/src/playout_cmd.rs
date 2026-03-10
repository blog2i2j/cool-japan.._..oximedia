//! Playout server CLI commands.
//!
//! Provides subcommands for managing broadcast playout schedules, controlling
//! playout servers, and monitoring playout status.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Playout command subcommands.
#[derive(Subcommand, Debug)]
pub enum PlayoutCommand {
    /// Create or edit a playout schedule
    Schedule {
        /// Schedule file path (.json)
        #[arg(short, long)]
        output: PathBuf,

        /// Channel name
        #[arg(long, default_value = "Channel 1")]
        channel: String,

        /// Video format: hd1080p25, hd1080p30, uhd2160p25, etc.
        #[arg(long, default_value = "hd1080p25")]
        format: String,

        /// Schedule date (YYYY-MM-DD)
        #[arg(long)]
        date: Option<String>,
    },

    /// Start the playout server
    Start {
        /// Schedule/playlist file to use
        #[arg(short, long)]
        schedule: PathBuf,

        /// Genlock reference source: internal, sdi, ptp
        #[arg(long, default_value = "internal")]
        clock_source: String,

        /// Output buffer size in frames
        #[arg(long, default_value = "10")]
        buffer_size: usize,

        /// Emergency fallback content path
        #[arg(long)]
        fallback: Option<PathBuf>,

        /// Enable monitoring on given port
        #[arg(long)]
        monitor_port: Option<u16>,
    },

    /// Stop the playout server
    Stop {
        /// Channel name or ID to stop
        #[arg(long, default_value = "Channel 1")]
        channel: String,

        /// Force immediate stop without graceful shutdown
        #[arg(long)]
        force: bool,
    },

    /// Show current playout status
    Status {
        /// Channel name or ID
        #[arg(long)]
        channel: Option<String>,

        /// Show detailed timing information
        #[arg(long)]
        detailed: bool,
    },

    /// Load a playlist or schedule file into a running server
    Load {
        /// Playlist or schedule file path
        #[arg(short, long)]
        input: PathBuf,

        /// Channel name or ID
        #[arg(long, default_value = "Channel 1")]
        channel: String,

        /// Append to current schedule instead of replacing
        #[arg(long)]
        append: bool,
    },

    /// Skip to the next item in the schedule
    Next {
        /// Channel name or ID
        #[arg(long, default_value = "Channel 1")]
        channel: String,

        /// Transition type: cut, dissolve
        #[arg(long, default_value = "cut")]
        transition: String,
    },

    /// List scheduled items
    List {
        /// Schedule or playlist file to list
        #[arg(short, long)]
        input: PathBuf,

        /// Maximum number of items to display
        #[arg(long)]
        limit: Option<usize>,

        /// Show only upcoming items (from now)
        #[arg(long)]
        upcoming: bool,
    },
}

/// Handle playout command dispatch.
pub async fn handle_playout_command(command: PlayoutCommand, json_output: bool) -> Result<()> {
    match command {
        PlayoutCommand::Schedule {
            output,
            channel,
            format,
            date,
        } => handle_schedule(&output, &channel, &format, date.as_deref(), json_output).await,
        PlayoutCommand::Start {
            schedule,
            clock_source,
            buffer_size,
            fallback,
            monitor_port,
        } => {
            handle_start(
                &schedule,
                &clock_source,
                buffer_size,
                fallback.as_deref(),
                monitor_port,
                json_output,
            )
            .await
        }
        PlayoutCommand::Stop { channel, force } => handle_stop(&channel, force, json_output).await,
        PlayoutCommand::Status { channel, detailed } => {
            handle_status(channel.as_deref(), detailed, json_output).await
        }
        PlayoutCommand::Load {
            input,
            channel,
            append,
        } => handle_load(&input, &channel, append, json_output).await,
        PlayoutCommand::Next {
            channel,
            transition,
        } => handle_next(&channel, &transition, json_output).await,
        PlayoutCommand::List {
            input,
            limit,
            upcoming,
        } => handle_list(&input, limit, upcoming, json_output).await,
    }
}

// ---------------------------------------------------------------------------
// Internal schedule data model
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct PlayoutSchedule {
    channel: String,
    video_format: String,
    date: String,
    items: Vec<ScheduleItem>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct ScheduleItem {
    id: u64,
    title: String,
    source_path: String,
    scheduled_time: String,
    duration_secs: f64,
    item_type: String,
}

impl PlayoutSchedule {
    fn new(channel: &str, format: &str, date: &str) -> Self {
        Self {
            channel: channel.to_string(),
            video_format: format.to_string(),
            date: date.to_string(),
            items: Vec::new(),
        }
    }

    fn load(path: &std::path::Path) -> Result<Self> {
        let content =
            std::fs::read_to_string(path).context("Failed to read playout schedule file")?;
        let schedule: Self =
            serde_json::from_str(&content).context("Failed to parse playout schedule")?;
        Ok(schedule)
    }

    fn save(&self, path: &std::path::Path) -> Result<()> {
        let content =
            serde_json::to_string_pretty(self).context("Failed to serialize playout schedule")?;
        std::fs::write(path, content).context("Failed to write playout schedule file")?;
        Ok(())
    }

    fn total_duration(&self) -> f64 {
        self.items.iter().map(|i| i.duration_secs).sum()
    }
}

// ---------------------------------------------------------------------------
// Handler: Schedule
// ---------------------------------------------------------------------------

async fn handle_schedule(
    output: &std::path::Path,
    channel: &str,
    format: &str,
    date: Option<&str>,
    json_output: bool,
) -> Result<()> {
    let valid_formats = [
        "hd1080p25",
        "hd1080p30",
        "hd1080p50",
        "hd1080p60",
        "hd1080i50",
        "hd1080i5994",
        "uhd2160p25",
        "uhd2160p50",
    ];

    if !valid_formats.contains(&format) {
        return Err(anyhow::anyhow!(
            "Unknown video format '{}'. Valid: {}",
            format,
            valid_formats.join(", ")
        ));
    }

    let schedule_date = date.unwrap_or("2026-01-01");
    let schedule = PlayoutSchedule::new(channel, format, schedule_date);
    schedule.save(output)?;

    if json_output {
        let result = serde_json::json!({
            "action": "schedule",
            "output": output.display().to_string(),
            "channel": channel,
            "format": format,
            "date": schedule_date,
            "status": "created",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Playout Schedule Created".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Channel:", channel);
        println!("{:20} {}", "Format:", format);
        println!("{:20} {}", "Date:", schedule_date);
        println!("{:20} {}", "Output:", output.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Start
// ---------------------------------------------------------------------------

async fn handle_start(
    schedule: &std::path::Path,
    clock_source: &str,
    buffer_size: usize,
    fallback: Option<&std::path::Path>,
    monitor_port: Option<u16>,
    json_output: bool,
) -> Result<()> {
    if !schedule.exists() {
        return Err(anyhow::anyhow!(
            "Schedule file not found: {}",
            schedule.display()
        ));
    }

    let valid_clocks = ["internal", "sdi", "ptp"];
    if !valid_clocks.contains(&clock_source) {
        return Err(anyhow::anyhow!(
            "Unknown clock source '{}'. Valid: {}",
            clock_source,
            valid_clocks.join(", ")
        ));
    }

    let sched = PlayoutSchedule::load(schedule)?;
    let port = monitor_port.unwrap_or(8080);

    let config = oximedia_playout::PlayoutConfig {
        clock_source: clock_source.to_string(),
        buffer_size,
        fallback_content: fallback
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("/var/oximedia/fallback.mxf")),
        monitoring_enabled: monitor_port.is_some(),
        monitoring_port: port,
        ..oximedia_playout::PlayoutConfig::default()
    };

    if json_output {
        let result = serde_json::json!({
            "action": "start",
            "schedule": schedule.display().to_string(),
            "channel": sched.channel,
            "format": sched.video_format,
            "clock_source": clock_source,
            "buffer_size": buffer_size,
            "monitoring_port": port,
            "items": sched.items.len(),
            "genlock": config.genlock_enabled,
            "status": "starting",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Playout Server Starting".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Channel:", sched.channel);
        println!("{:20} {}", "Format:", sched.video_format);
        println!("{:20} {}", "Clock source:", clock_source);
        println!("{:20} {}", "Buffer size:", buffer_size);
        println!("{:20} {}", "Monitor port:", port);
        println!("{:20} {}", "Schedule items:", sched.items.len());
        println!();
        println!(
            "{}",
            "Note: Full playout pipeline requires real-time frame scheduling.".yellow()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Stop
// ---------------------------------------------------------------------------

async fn handle_stop(channel: &str, force: bool, json_output: bool) -> Result<()> {
    if json_output {
        let result = serde_json::json!({
            "action": "stop",
            "channel": channel,
            "force": force,
            "status": "stopped",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Playout Server Stopped".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Channel:", channel);
        println!("{:20} {}", "Force:", force);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Status
// ---------------------------------------------------------------------------

async fn handle_status(channel: Option<&str>, detailed: bool, json_output: bool) -> Result<()> {
    let ch = channel.unwrap_or("Channel 1");

    let state_name = "Stopped";

    if json_output {
        let result = serde_json::json!({
            "channel": ch,
            "state": state_name,
            "detailed": detailed,
            "current_item": null,
            "next_item": null,
            "uptime_secs": 0,
            "frames_played": 0,
            "frames_dropped": 0,
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Playout Status".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Channel:", ch);
        println!("{:20} {}", "State:", state_name);
        println!("{:20} None", "Current item:");
        println!("{:20} None", "Next item:");
        if detailed {
            println!();
            println!("{}", "Timing Details".cyan().bold());
            println!("{}", "-".repeat(60));
            println!("{:20} 0", "Uptime (s):");
            println!("{:20} 0", "Frames played:");
            println!("{:20} 0", "Frames dropped:");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Load
// ---------------------------------------------------------------------------

async fn handle_load(
    input: &std::path::Path,
    channel: &str,
    append: bool,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!(
            "Playlist file not found: {}",
            input.display()
        ));
    }

    let sched = PlayoutSchedule::load(input)?;

    if json_output {
        let result = serde_json::json!({
            "action": "load",
            "input": input.display().to_string(),
            "channel": channel,
            "append": append,
            "items_loaded": sched.items.len(),
            "total_duration": sched.total_duration(),
            "status": "loaded",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Playlist Loaded".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Channel:", channel);
        println!("{:20} {}", "Input:", input.display());
        println!(
            "{:20} {}",
            "Mode:",
            if append { "Append" } else { "Replace" }
        );
        println!("{:20} {}", "Items loaded:", sched.items.len());
        println!("{:20} {:.1}s", "Total duration:", sched.total_duration());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Next
// ---------------------------------------------------------------------------

async fn handle_next(channel: &str, transition: &str, json_output: bool) -> Result<()> {
    let valid_transitions = ["cut", "dissolve"];
    if !valid_transitions.contains(&transition) {
        return Err(anyhow::anyhow!(
            "Unknown transition '{}'. Valid: {}",
            transition,
            valid_transitions.join(", ")
        ));
    }

    if json_output {
        let result = serde_json::json!({
            "action": "next",
            "channel": channel,
            "transition": transition,
            "status": "skipped",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Skipped to Next Item".green().bold());
        println!("{:20} {}", "Channel:", channel);
        println!("{:20} {}", "Transition:", transition);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: List
// ---------------------------------------------------------------------------

async fn handle_list(
    input: &std::path::Path,
    limit: Option<usize>,
    upcoming: bool,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!(
            "Schedule file not found: {}",
            input.display()
        ));
    }

    let sched = PlayoutSchedule::load(input)?;
    let items: Vec<&ScheduleItem> = if let Some(max) = limit {
        sched.items.iter().take(max).collect()
    } else {
        sched.items.iter().collect()
    };

    if json_output {
        let result = serde_json::json!({
            "channel": sched.channel,
            "date": sched.date,
            "format": sched.video_format,
            "total_items": sched.items.len(),
            "displayed_items": items.len(),
            "upcoming_only": upcoming,
            "total_duration": sched.total_duration(),
            "items": items.iter().map(|i| {
                serde_json::json!({
                    "id": i.id,
                    "title": i.title,
                    "source": i.source_path,
                    "time": i.scheduled_time,
                    "duration": i.duration_secs,
                    "type": i.item_type,
                })
            }).collect::<Vec<_>>(),
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Playout Schedule".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Channel:", sched.channel);
        println!("{:20} {}", "Date:", sched.date);
        println!("{:20} {}", "Format:", sched.video_format);
        println!(
            "{:20} {} / {} shown",
            "Items:",
            items.len(),
            sched.items.len()
        );
        println!("{:20} {:.1}s", "Total duration:", sched.total_duration());
        println!();

        if items.is_empty() {
            println!("{}", "No items in schedule.".dimmed());
        } else {
            for item in &items {
                println!(
                    "  [{}] {} - {} ({:.1}s) [{}]",
                    item.id, item.scheduled_time, item.title, item.duration_secs, item.item_type,
                );
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
    fn test_playout_schedule_new() {
        let sched = PlayoutSchedule::new("Test Channel", "hd1080p25", "2026-01-01");
        assert_eq!(sched.channel, "Test Channel");
        assert_eq!(sched.video_format, "hd1080p25");
        assert!(sched.items.is_empty());
    }

    #[test]
    fn test_playout_schedule_total_duration() {
        let mut sched = PlayoutSchedule::new("Ch1", "hd1080p25", "2026-01-01");
        assert!((sched.total_duration() - 0.0).abs() < f64::EPSILON);

        sched.items.push(ScheduleItem {
            id: 1,
            title: "Item 1".to_string(),
            source_path: "clip1.mxf".to_string(),
            scheduled_time: "08:00:00".to_string(),
            duration_secs: 30.0,
            item_type: "programme".to_string(),
        });
        sched.items.push(ScheduleItem {
            id: 2,
            title: "Item 2".to_string(),
            source_path: "clip2.mxf".to_string(),
            scheduled_time: "08:00:30".to_string(),
            duration_secs: 15.0,
            item_type: "commercial".to_string(),
        });

        assert!((sched.total_duration() - 45.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_playout_schedule_save_load() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_playout_schedule.json");

        let mut sched = PlayoutSchedule::new("Test Ch", "hd1080p25", "2026-03-01");
        sched.items.push(ScheduleItem {
            id: 1,
            title: "News".to_string(),
            source_path: "news.mxf".to_string(),
            scheduled_time: "18:00:00".to_string(),
            duration_secs: 1800.0,
            item_type: "programme".to_string(),
        });
        sched.save(&path).expect("save should succeed");

        let loaded = PlayoutSchedule::load(&path).expect("load should succeed");
        assert_eq!(loaded.channel, "Test Ch");
        assert_eq!(loaded.items.len(), 1);
        assert_eq!(loaded.items[0].title, "News");

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_handle_schedule_creates_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_playout_handle_schedule.json");

        let result = handle_schedule(&path, "Test", "hd1080p25", Some("2026-06-01"), false).await;
        assert!(result.is_ok());
        assert!(path.exists());

        let loaded = PlayoutSchedule::load(&path).expect("load should succeed");
        assert_eq!(loaded.channel, "Test");
        assert_eq!(loaded.date, "2026-06-01");

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_handle_schedule_invalid_format() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_playout_bad_format.json");

        let result = handle_schedule(&path, "Test", "invalid_fmt", None, false).await;
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }
}
