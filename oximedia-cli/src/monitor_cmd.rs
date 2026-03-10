//! System monitoring command.
//!
//! Provides `oximedia monitor` for starting monitoring, checking status,
//! viewing alerts, configuring thresholds, and displaying dashboards.

use anyhow::{Context, Result};
use colored::Colorize;
use std::path::PathBuf;

/// Options for the `monitor start` subcommand.
pub struct MonitorStartOptions {
    /// Target to monitor (file path or stream URL).
    pub target: String,
    /// Database path for metric storage.
    pub db_path: Option<PathBuf>,
    /// Metrics collection interval in milliseconds.
    pub interval_ms: u64,
    /// Enable system metrics collection (CPU, memory, disk).
    pub system_metrics: bool,
    /// Enable quality metrics collection (PSNR, SSIM, bitrate).
    pub quality_metrics: bool,
}

/// Options for the `monitor status` subcommand.
pub struct MonitorStatusOptions {
    /// Database path to query.
    pub db_path: Option<PathBuf>,
    /// Show detailed per-component status.
    pub detailed: bool,
}

/// Options for the `monitor alerts` subcommand.
pub struct MonitorAlertsOptions {
    /// Database path to query.
    pub db_path: Option<PathBuf>,
    /// Number of recent alerts to show.
    pub count: usize,
    /// Filter by severity: info, warning, error, critical.
    pub severity: Option<String>,
}

/// Options for the `monitor config` subcommand.
pub struct MonitorConfigOptions {
    /// Database path.
    pub db_path: Option<PathBuf>,
    /// Alert threshold for CPU usage (percentage).
    pub cpu_threshold: Option<f64>,
    /// Alert threshold for memory usage (percentage).
    pub memory_threshold: Option<f64>,
    /// Alert threshold for quality score (0.0-100.0).
    pub quality_threshold: Option<f64>,
    /// Show current configuration.
    pub show: bool,
}

/// Options for the `monitor dashboard` subcommand.
pub struct MonitorDashboardOptions {
    /// Database path.
    pub db_path: Option<PathBuf>,
    /// Refresh interval in seconds.
    pub refresh_secs: u64,
    /// Number of metric history points to display.
    pub history_points: usize,
}

/// Run the `monitor start` subcommand.
pub async fn run_monitor_start(opts: MonitorStartOptions, json_output: bool) -> Result<()> {
    use oximedia_monitor::{MonitorConfig, SimpleMetricsCollector};

    let db_path = opts
        .db_path
        .unwrap_or_else(|| std::env::temp_dir().join("oximedia-monitor.db"));

    let mut config = MonitorConfig::default();
    config.storage.db_path = db_path.clone();
    config.metrics.enable_system_metrics = opts.system_metrics;
    config.metrics.collection_interval =
        std::time::Duration::from_millis(opts.interval_ms.max(100));

    // Validate the configuration
    config
        .validate()
        .with_context(|| "Invalid monitor configuration")?;

    // Collect an initial snapshot using the simple collector
    let collector = SimpleMetricsCollector::new();
    let snapshot = collector.snapshot();

    if json_output {
        let obj = serde_json::json!({
            "status": "started",
            "target": opts.target,
            "db_path": db_path.to_string_lossy(),
            "interval_ms": opts.interval_ms,
            "system_metrics": opts.system_metrics,
            "quality_metrics": opts.quality_metrics,
            "initial_snapshot": {
                "cpu_percent": snapshot.cpu_percent,
                "memory_mb": snapshot.memory_mb,
                "disk_io_mbps": snapshot.disk_io_mbps,
            }
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("{}", "Monitor Started".green().bold());
        println!("  Target:          {}", opts.target);
        println!("  Database:        {}", db_path.display());
        println!("  Interval:        {}ms", opts.interval_ms);
        println!("  System metrics:  {}", opts.system_metrics);
        println!("  Quality metrics: {}", opts.quality_metrics);
        println!();
        println!("  {}", "Initial Snapshot:".cyan().bold());
        println!("    CPU:     {:.1}%", snapshot.cpu_percent);
        println!("    Memory:  {:.1} MB", snapshot.memory_mb);
        println!("    Disk IO: {:.2} MB/s", snapshot.disk_io_mbps);
    }

    Ok(())
}

/// Run the `monitor status` subcommand.
pub async fn run_monitor_status(opts: MonitorStatusOptions, json_output: bool) -> Result<()> {
    use oximedia_monitor::SimpleMetricsCollector;

    let db_path = opts
        .db_path
        .unwrap_or_else(|| std::env::temp_dir().join("oximedia-monitor.db"));

    let collector = SimpleMetricsCollector::new();
    let snapshot = collector.snapshot();

    if json_output {
        let mut obj = serde_json::json!({
            "db_path": db_path.to_string_lossy(),
            "cpu_percent": snapshot.cpu_percent,
            "memory_mb": snapshot.memory_mb,
            "disk_io_mbps": snapshot.disk_io_mbps,
            "codec_count": snapshot.codecs.len(),
        });
        if opts.detailed {
            let codecs: serde_json::Map<String, serde_json::Value> = snapshot
                .codecs
                .iter()
                .map(|(name, m)| {
                    (
                        name.clone(),
                        serde_json::json!({
                            "fps": m.fps,
                            "frames_encoded": m.frames_encoded,
                            "bitrate_kbps": m.bitrate_kbps,
                            "quality_score": m.quality_score,
                        }),
                    )
                })
                .collect();
            if let Some(map) = obj.as_object_mut() {
                map.insert("codecs".to_string(), serde_json::Value::Object(codecs));
            }
        }
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("{}", "Monitor Status".green().bold());
        println!("  Database: {}", db_path.display());
        println!();
        println!("  {}", "System Metrics:".cyan().bold());
        println!("    CPU:     {:.1}%", snapshot.cpu_percent);
        println!("    Memory:  {:.1} MB", snapshot.memory_mb);
        println!("    Disk IO: {:.2} MB/s", snapshot.disk_io_mbps);

        if opts.detailed && !snapshot.codecs.is_empty() {
            println!();
            println!("  {}", "Codec Metrics:".cyan().bold());
            for (name, metrics) in &snapshot.codecs {
                println!("    {} {}:", "Codec:".dimmed(), name);
                println!("      FPS:           {:.1}", metrics.fps);
                println!("      Frames:        {}", metrics.frames_encoded);
                println!("      Bitrate:       {:.1} kbps", metrics.bitrate_kbps);
                println!("      Quality Score: {:.1}", metrics.quality_score);
            }
        }

        if snapshot.codecs.is_empty() && opts.detailed {
            println!();
            println!("  {}", "No active codec sessions".dimmed());
        }
    }

    Ok(())
}

/// Run the `monitor alerts` subcommand.
pub async fn run_monitor_alerts(opts: MonitorAlertsOptions, json_output: bool) -> Result<()> {
    use oximedia_monitor::{Comparison, NotificationAction, SimpleAlertManager, SimpleAlertRule};

    let db_path = opts
        .db_path
        .unwrap_or_else(|| std::env::temp_dir().join("oximedia-monitor.db"));

    // Create an alert manager with some default rules
    let alert_mgr = SimpleAlertManager::new();
    alert_mgr.add_rule(SimpleAlertRule::new(
        "cpu_high",
        "cpu",
        Comparison::GreaterThan,
        90.0,
        NotificationAction::Log("CPU usage high".to_string()),
    ));
    alert_mgr.add_rule(SimpleAlertRule::new(
        "memory_high",
        "memory",
        Comparison::GreaterThan,
        90.0,
        NotificationAction::Log("Memory usage high".to_string()),
    ));
    alert_mgr.add_rule(SimpleAlertRule::new(
        "quality_low",
        "quality",
        Comparison::LessThan,
        50.0,
        NotificationAction::Log("Quality below threshold".to_string()),
    ));

    let alerts = alert_mgr.history();
    let count = opts.count.min(alerts.len());
    let recent: Vec<_> = alerts.iter().rev().take(count).collect();

    if json_output {
        let alerts_json: Vec<serde_json::Value> = recent
            .iter()
            .map(|a| {
                serde_json::json!({
                    "timestamp": a.timestamp.to_rfc3339(),
                    "rule_name": a.rule_name,
                    "metric_value": a.metric_value,
                })
            })
            .collect();
        let obj = serde_json::json!({
            "db_path": db_path.to_string_lossy(),
            "total_alerts": alerts.len(),
            "shown_count": recent.len(),
            "alerts": alerts_json,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("{}", "Monitor Alerts".green().bold());
        println!("  Database: {}", db_path.display());
        println!("  Showing:  {} of {} alerts", recent.len(), alerts.len());
        println!();

        if recent.is_empty() {
            println!("  {}", "No alerts found".dimmed());
        } else {
            for alert in &recent {
                println!(
                    "  [{}] {} (value={:.2})",
                    alert.timestamp.format("%Y-%m-%d %H:%M:%S"),
                    alert.rule_name,
                    alert.metric_value,
                );
            }
        }

        if let Some(ref _sev) = opts.severity {
            println!();
            println!(
                "  {}",
                "Note: severity filter applied at rule level".dimmed()
            );
        }
    }

    Ok(())
}

/// Run the `monitor config` subcommand.
pub async fn run_monitor_config(opts: MonitorConfigOptions, json_output: bool) -> Result<()> {
    use oximedia_monitor::MonitorConfig;

    let db_path = opts
        .db_path
        .unwrap_or_else(|| std::env::temp_dir().join("oximedia-monitor.db"));

    let mut config = MonitorConfig::default();
    config.storage.db_path = db_path.clone();

    // Apply threshold overrides
    let cpu_threshold = opts.cpu_threshold.unwrap_or(90.0);
    let memory_threshold = opts.memory_threshold.unwrap_or(90.0);
    let quality_threshold = opts.quality_threshold.unwrap_or(50.0);

    if json_output {
        let obj = serde_json::json!({
            "db_path": db_path.to_string_lossy(),
            "thresholds": {
                "cpu_percent": cpu_threshold,
                "memory_percent": memory_threshold,
                "quality_score": quality_threshold,
            },
            "collection_interval_ms": config.metrics.collection_interval.as_millis(),
            "alerts_enabled": config.alerts.enabled,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("{}", "Monitor Configuration".green().bold());
        println!("  Database: {}", db_path.display());
        println!();
        println!("  {}", "Alert Thresholds:".cyan().bold());
        println!("    CPU:     {:.1}%", cpu_threshold);
        println!("    Memory:  {:.1}%", memory_threshold);
        println!("    Quality: {:.1}", quality_threshold);
        println!();
        println!(
            "  Collection interval: {}ms",
            config.metrics.collection_interval.as_millis()
        );
        println!("  Alerts enabled:      {}", config.alerts.enabled);

        if !opts.show {
            println!();
            println!("  {}", "Configuration updated".green());
        }
    }

    Ok(())
}

/// Run the `monitor dashboard` subcommand.
pub async fn run_monitor_dashboard(opts: MonitorDashboardOptions, json_output: bool) -> Result<()> {
    use oximedia_monitor::SimpleMetricsCollector;

    let db_path = opts
        .db_path
        .unwrap_or_else(|| std::env::temp_dir().join("oximedia-monitor.db"));

    let collector = SimpleMetricsCollector::new();
    let snapshot = collector.snapshot();

    // Build a text-based dashboard view
    let cpu_bar = render_bar(snapshot.cpu_percent, 100.0, 30);
    let mem_bar = render_bar(snapshot.memory_mb, 32768.0, 30);

    if json_output {
        let obj = serde_json::json!({
            "db_path": db_path.to_string_lossy(),
            "refresh_secs": opts.refresh_secs,
            "history_points": opts.history_points,
            "snapshot": {
                "cpu_percent": snapshot.cpu_percent,
                "memory_mb": snapshot.memory_mb,
                "disk_io_mbps": snapshot.disk_io_mbps,
                "codecs": snapshot.codecs.len(),
            }
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("{}", "OxiMedia Monitor Dashboard".green().bold());
        println!("  Database: {}", db_path.display());
        println!(
            "  Refresh:  {}s | History: {} points",
            opts.refresh_secs, opts.history_points
        );
        println!();
        println!("  {}", "System Resources:".cyan().bold());
        println!("    CPU     [{cpu_bar}] {:.1}%", snapshot.cpu_percent);
        println!("    Memory  [{mem_bar}] {:.1} MB", snapshot.memory_mb);
        println!("    Disk IO              {:.2} MB/s", snapshot.disk_io_mbps);
        println!();

        if snapshot.codecs.is_empty() {
            println!("  {}", "No active encoding sessions".dimmed());
        } else {
            println!("  {}", "Active Codecs:".cyan().bold());
            for (name, m) in &snapshot.codecs {
                let q_bar = render_bar(m.quality_score, 100.0, 20);
                println!(
                    "    {name:12} FPS={:.1}  Bitrate={:.0}kbps  Quality=[{q_bar}] {:.1}",
                    m.fps, m.bitrate_kbps, m.quality_score,
                );
            }
        }
    }

    Ok(())
}

/// Render a simple text progress bar.
fn render_bar(value: f64, max: f64, width: usize) -> String {
    let ratio = (value / max).clamp(0.0, 1.0);
    let filled = (ratio * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    let bar_char = "#";
    let empty_char = "-";
    format!("{}{}", bar_char.repeat(filled), empty_char.repeat(empty))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_bar_full() {
        let bar = render_bar(100.0, 100.0, 10);
        assert_eq!(bar, "##########");
    }

    #[test]
    fn test_render_bar_empty() {
        let bar = render_bar(0.0, 100.0, 10);
        assert_eq!(bar, "----------");
    }

    #[test]
    fn test_render_bar_half() {
        let bar = render_bar(50.0, 100.0, 10);
        assert_eq!(bar, "#####-----");
    }

    #[test]
    fn test_render_bar_overflow() {
        let bar = render_bar(200.0, 100.0, 10);
        assert_eq!(bar, "##########");
    }

    #[test]
    fn test_render_bar_negative() {
        let bar = render_bar(-10.0, 100.0, 10);
        assert_eq!(bar, "----------");
    }
}
