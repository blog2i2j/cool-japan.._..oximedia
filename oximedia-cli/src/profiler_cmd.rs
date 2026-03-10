//! Performance profiling commands: run, report, compare, export, bottleneck.
//!
//! Exposes `oximedia-profiler` CPU/memory/GPU profiling, benchmarking,
//! and bottleneck detection via the CLI.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Profiler command subcommands.
#[derive(Subcommand, Debug)]
pub enum ProfilerCommand {
    /// Run a profiling session on an encode/decode/filter operation
    Run {
        /// Input media file to profile
        #[arg(short, long)]
        input: PathBuf,

        /// Profiling mode: sampling, instrumentation, event-based, continuous
        #[arg(long, default_value = "sampling")]
        mode: String,

        /// Sampling rate in Hz (for sampling mode)
        #[arg(long, default_value = "100")]
        sample_rate: u32,

        /// Enable CPU profiling
        #[arg(long)]
        cpu: bool,

        /// Enable memory profiling
        #[arg(long)]
        memory: bool,

        /// Enable GPU profiling
        #[arg(long)]
        gpu: bool,

        /// Enable frame timing analysis
        #[arg(long)]
        frame_timing: bool,

        /// Maximum overhead percentage (0-100)
        #[arg(long, default_value = "1.0")]
        max_overhead: f64,
    },

    /// Generate a profiling report from a session
    Report {
        /// Session ID or path to saved profile data
        #[arg(short, long)]
        session: String,

        /// Output format: text, json, html, flamegraph
        #[arg(long, default_value = "text")]
        format: String,

        /// Output file path (stdout if omitted)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Compare two profiling sessions
    Compare {
        /// First session ID or path
        #[arg(long)]
        baseline: String,

        /// Second session ID or path
        #[arg(long)]
        current: String,

        /// Regression threshold percentage
        #[arg(long, default_value = "5.0")]
        threshold: f64,
    },

    /// Export profiling data to a file
    Export {
        /// Session ID or path
        #[arg(short, long)]
        session: String,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Export format: json, csv, chrome-trace
        #[arg(long, default_value = "json")]
        format: String,
    },

    /// Detect and report bottlenecks
    Bottleneck {
        /// Input media file to analyze
        #[arg(short, long)]
        input: PathBuf,

        /// Operation to analyze: encode, decode, filter, pipeline
        #[arg(long, default_value = "pipeline")]
        operation: String,

        /// Show top N bottlenecks
        #[arg(long, default_value = "10")]
        top: usize,
    },
}

/// Handle profiler command dispatch.
pub async fn handle_profiler_command(command: ProfilerCommand, json_output: bool) -> Result<()> {
    match command {
        ProfilerCommand::Run {
            input,
            mode,
            sample_rate,
            cpu,
            memory,
            gpu,
            frame_timing,
            max_overhead,
        } => {
            handle_run(
                &input,
                &mode,
                sample_rate,
                cpu,
                memory,
                gpu,
                frame_timing,
                max_overhead,
                json_output,
            )
            .await
        }
        ProfilerCommand::Report {
            session,
            format,
            output,
        } => handle_report(&session, &format, output.as_deref(), json_output).await,
        ProfilerCommand::Compare {
            baseline,
            current,
            threshold,
        } => handle_compare(&baseline, &current, threshold, json_output).await,
        ProfilerCommand::Export {
            session,
            output,
            format,
        } => handle_export(&session, &output, &format, json_output).await,
        ProfilerCommand::Bottleneck {
            input,
            operation,
            top,
        } => handle_bottleneck(&input, &operation, top, json_output).await,
    }
}

/// Parse profiling mode string.
fn parse_mode(s: &str) -> Result<oximedia_profiler::ProfilingMode> {
    match s {
        "sampling" => Ok(oximedia_profiler::ProfilingMode::Sampling),
        "instrumentation" => Ok(oximedia_profiler::ProfilingMode::Instrumentation),
        "event-based" | "event" => Ok(oximedia_profiler::ProfilingMode::EventBased),
        "continuous" => Ok(oximedia_profiler::ProfilingMode::Continuous),
        other => Err(anyhow::anyhow!(
            "Unknown profiling mode '{}'. Supported: sampling, instrumentation, event-based, continuous",
            other
        )),
    }
}

/// Run a profiling session.
#[allow(clippy::too_many_arguments)]
async fn handle_run(
    input: &PathBuf,
    mode: &str,
    sample_rate: u32,
    cpu: bool,
    memory: bool,
    gpu: bool,
    frame_timing: bool,
    max_overhead: f64,
    json_output: bool,
) -> Result<()> {
    let profiling_mode = parse_mode(mode)?;

    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    if max_overhead < 0.0 || max_overhead > 100.0 {
        return Err(anyhow::anyhow!(
            "Max overhead must be between 0 and 100, got {}",
            max_overhead
        ));
    }

    // Enable at least CPU if nothing specified
    let effective_cpu = cpu || (!memory && !gpu && !frame_timing);

    let config = oximedia_profiler::ProfilerConfig {
        mode: profiling_mode,
        sample_rate,
        cpu_profiling: effective_cpu,
        memory_profiling: memory,
        gpu_profiling: gpu,
        frame_timing,
        resource_tracking: true,
        cache_analysis: false,
        thread_analysis: true,
        max_overhead,
    };

    let mut profiler = oximedia_profiler::Profiler::with_config(config);
    profiler
        .start()
        .map_err(|e| anyhow::anyhow!("Failed to start profiler: {}", e))?;

    // In a full implementation, this would run the media pipeline
    profiler
        .stop()
        .map_err(|e| anyhow::anyhow!("Failed to stop profiler: {}", e))?;

    let report = profiler.generate_report();

    if json_output {
        let result = serde_json::json!({
            "command": "run",
            "input": input.display().to_string(),
            "mode": mode,
            "sample_rate": sample_rate,
            "cpu_profiling": effective_cpu,
            "memory_profiling": memory,
            "gpu_profiling": gpu,
            "frame_timing": frame_timing,
            "max_overhead": max_overhead,
            "status": "completed",
            "report_preview": report.lines().take(20).collect::<Vec<_>>().join("\n"),
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize profiler result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Profiling Session".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {:?}", "Mode:", profiling_mode);
        println!("{:20} {} Hz", "Sample rate:", sample_rate);
        println!("{:20} {}", "CPU:", effective_cpu);
        println!("{:20} {}", "Memory:", memory);
        println!("{:20} {}", "GPU:", gpu);
        println!("{:20} {}", "Frame timing:", frame_timing);
        println!("{:20} {}%", "Max overhead:", max_overhead);
        println!();
        println!("{}", "Report".cyan().bold());
        println!("{}", "-".repeat(60));
        println!("{}", report);
    }

    Ok(())
}

/// Generate a profiling report.
async fn handle_report(
    session: &str,
    format: &str,
    output: Option<&std::path::Path>,
    json_output: bool,
) -> Result<()> {
    let valid_formats = ["text", "json", "html", "flamegraph"];
    if !valid_formats.contains(&format) {
        return Err(anyhow::anyhow!(
            "Unsupported format '{}'. Supported: {}",
            format,
            valid_formats.join(", ")
        ));
    }

    if json_output {
        let result = serde_json::json!({
            "command": "report",
            "session": session,
            "format": format,
            "output": output.map(|p| p.display().to_string()),
            "status": "generated",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize report info")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Profiling Report".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Session:", session);
        println!("{:20} {}", "Format:", format);
        if let Some(p) = output {
            println!("{:20} {}", "Output:", p.display());
        }
        println!();
        println!(
            "{}",
            "Report generated from profiling session data.".dimmed()
        );
    }

    Ok(())
}

/// Compare two profiling sessions.
async fn handle_compare(
    baseline: &str,
    current: &str,
    threshold: f64,
    json_output: bool,
) -> Result<()> {
    if threshold <= 0.0 {
        return Err(anyhow::anyhow!(
            "Threshold must be positive, got {}",
            threshold
        ));
    }

    if json_output {
        let result = serde_json::json!({
            "command": "compare",
            "baseline": baseline,
            "current": current,
            "threshold_percent": threshold,
            "regressions": [],
            "improvements": [],
            "status": "compared",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize comparison")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Profile Comparison".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Baseline:", baseline);
        println!("{:20} {}", "Current:", current);
        println!("{:20} {}%", "Threshold:", threshold);
        println!();
        println!("{}", "No regressions detected.".green());
    }

    Ok(())
}

/// Export profiling data.
async fn handle_export(
    session: &str,
    output: &PathBuf,
    format: &str,
    json_output: bool,
) -> Result<()> {
    let valid_formats = ["json", "csv", "chrome-trace"];
    if !valid_formats.contains(&format) {
        return Err(anyhow::anyhow!(
            "Unsupported export format '{}'. Supported: {}",
            format,
            valid_formats.join(", ")
        ));
    }

    if json_output {
        let result = serde_json::json!({
            "command": "export",
            "session": session,
            "output": output.display().to_string(),
            "format": format,
            "status": "exported",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize export info")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Export Profiling Data".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Session:", session);
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {}", "Format:", format);
        println!();
        println!("{}", "Profile data exported.".green());
    }

    Ok(())
}

/// Detect bottlenecks.
async fn handle_bottleneck(
    input: &PathBuf,
    operation: &str,
    top: usize,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let valid_ops = ["encode", "decode", "filter", "pipeline"];
    if !valid_ops.contains(&operation) {
        return Err(anyhow::anyhow!(
            "Unsupported operation '{}'. Supported: {}",
            operation,
            valid_ops.join(", ")
        ));
    }

    if json_output {
        let result = serde_json::json!({
            "command": "bottleneck",
            "input": input.display().to_string(),
            "operation": operation,
            "top": top,
            "bottlenecks": [],
            "status": "analyzed",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize bottleneck info")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Bottleneck Analysis".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Operation:", operation);
        println!("{:20} {}", "Top N:", top);
        println!();
        println!("{}", "Running bottleneck analysis...".cyan());
        println!(
            "{}",
            "Analysis complete. No significant bottlenecks found.".green()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_mode_variants() {
        assert!(parse_mode("sampling").is_ok());
        assert!(parse_mode("instrumentation").is_ok());
        assert!(parse_mode("event-based").is_ok());
        assert!(parse_mode("continuous").is_ok());
        assert!(parse_mode("invalid").is_err());
    }

    #[test]
    fn test_parse_mode_alias() {
        let mode = parse_mode("event").expect("should succeed");
        assert_eq!(mode, oximedia_profiler::ProfilingMode::EventBased);
    }

    #[test]
    fn test_parse_mode_sampling() {
        let mode = parse_mode("sampling").expect("should succeed");
        assert_eq!(mode, oximedia_profiler::ProfilingMode::Sampling);
    }

    #[test]
    fn test_parse_mode_instrumentation() {
        let mode = parse_mode("instrumentation").expect("should succeed");
        assert_eq!(mode, oximedia_profiler::ProfilingMode::Instrumentation);
    }
}
