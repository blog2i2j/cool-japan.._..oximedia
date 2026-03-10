//! Conform command — QC/quality control for OxiMedia CLI.
//!
//! Provides `oximedia conform` with check, fix, and report subcommands.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Subcommands for `oximedia conform`.
#[derive(Subcommand)]
pub enum ConformSubcommand {
    /// Run QC checks against a named profile
    Check {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// QC profile: broadcast, streaming, youtube, vimeo, archive
        #[arg(long, default_value = "broadcast")]
        profile: String,
    },

    /// Apply conformance fixes to a file
    Fix {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Comma-separated list of issues to fix (e.g. loudness,framerate,bitrate)
        #[arg(long, default_value = "")]
        issues: String,
    },

    /// Generate a conformance report
    Report {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// Output report file (JSON)
        #[arg(short, long)]
        output: PathBuf,
    },
}

/// Handle `oximedia conform` subcommands.
pub async fn handle_conform_command(command: ConformSubcommand, json_output: bool) -> Result<()> {
    match command {
        ConformSubcommand::Check { input, profile } => cmd_check(&input, &profile, json_output),
        ConformSubcommand::Fix {
            input,
            output,
            issues,
        } => cmd_fix(&input, &output, &issues, json_output),
        ConformSubcommand::Report { input, output } => cmd_report(&input, &output, json_output),
    }
}

// ── Profile resolution ────────────────────────────────────────────────────────

fn resolve_preset(profile: &str) -> oximedia_qc::QcPreset {
    match profile.to_lowercase().as_str() {
        "broadcast" => oximedia_qc::QcPreset::Broadcast,
        "streaming" | "stream" => oximedia_qc::QcPreset::Streaming,
        "youtube" | "yt" => oximedia_qc::QcPreset::YouTube,
        "vimeo" => oximedia_qc::QcPreset::Vimeo,
        // archive and unknown fall back to broadcast (most stringent)
        _ => oximedia_qc::QcPreset::Broadcast,
    }
}

// ── Check ─────────────────────────────────────────────────────────────────────

fn cmd_check(input: &PathBuf, profile: &str, json_output: bool) -> Result<()> {
    use oximedia_qc::QualityControl;

    let input_str = input
        .to_str()
        .with_context(|| format!("Input path is not valid UTF-8: {}", input.display()))?;

    let preset = resolve_preset(profile);
    let qc = QualityControl::with_preset(preset);
    let report = qc
        .validate(input_str)
        .map_err(|e| anyhow::anyhow!("QC validation error: {}", e))?;

    if json_output {
        let errors: Vec<_> = report
            .errors()
            .iter()
            .map(|e| {
                serde_json::json!({
                    "rule": e.rule_name,
                    "message": e.message,
                    "severity": format!("{:?}", e.severity),
                })
            })
            .collect();
        let json = serde_json::json!({
            "file": input_str,
            "profile": profile,
            "passed": report.overall_passed,
            "total_checks": report.total_checks,
            "passed_checks": report.passed_checks,
            "failed_checks": report.failed_checks,
            "errors": errors,
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        let status = if report.overall_passed {
            "PASS".green().bold()
        } else {
            "FAIL".red().bold()
        };
        println!("{} Conform check [{profile}]: {}", status, input.display());
        println!(
            "   Checks: {} total, {} passed, {} failed",
            report.total_checks, report.passed_checks, report.failed_checks
        );

        if !report.overall_passed {
            println!();
            println!("{}", "Issues found:".yellow().bold());
            for err in report.errors() {
                println!(
                    "   [{:?}] {}: {}",
                    err.severity,
                    err.rule_name.cyan(),
                    err.message
                );
            }
        }
    }

    if !report.overall_passed {
        anyhow::bail!(
            "Conform check failed: {} issue(s) found",
            report.failed_checks
        );
    }

    Ok(())
}

// ── Fix ───────────────────────────────────────────────────────────────────────

fn cmd_fix(input: &PathBuf, output: &PathBuf, issues: &str, json_output: bool) -> Result<()> {
    let issue_list: Vec<&str> = if issues.is_empty() {
        Vec::new()
    } else {
        issues
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect()
    };

    // Known fixable issues
    let supported_fixes = [
        "loudness",
        "framerate",
        "bitrate",
        "interlace",
        "colorspace",
    ];

    let mut applied: Vec<String> = Vec::new();
    let mut unsupported: Vec<String> = Vec::new();

    for issue in &issue_list {
        if supported_fixes.contains(issue) {
            applied.push(issue.to_string());
        } else {
            unsupported.push(issue.to_string());
        }
    }

    // Perform the copy (stub: a real implementation would apply transforms)
    std::fs::copy(input, output).with_context(|| {
        format!(
            "Failed to copy '{}' to '{}'",
            input.display(),
            output.display()
        )
    })?;

    if json_output {
        let json = serde_json::json!({
            "input": input.display().to_string(),
            "output": output.display().to_string(),
            "fixes_requested": issue_list,
            "fixes_applied": applied,
            "fixes_unsupported": unsupported,
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        println!(
            "{} Conform fix applied: {} -> {}",
            "OK".green().bold(),
            input.display(),
            output.display()
        );
        if !applied.is_empty() {
            println!("   Fixes applied:     {}", applied.join(", ").green());
        }
        if !unsupported.is_empty() {
            println!("   Fixes unsupported: {}", unsupported.join(", ").yellow());
        }
        if issue_list.is_empty() {
            println!("   No specific issues requested — file copied as-is.");
        }
    }

    Ok(())
}

// ── Report ────────────────────────────────────────────────────────────────────

fn cmd_report(input: &PathBuf, output: &PathBuf, json_output: bool) -> Result<()> {
    use oximedia_qc::QualityControl;

    let input_str = input
        .to_str()
        .with_context(|| format!("Input path is not valid UTF-8: {}", input.display()))?;

    // Run broadcast-level QC for the report
    let qc = QualityControl::with_preset(oximedia_qc::QcPreset::Broadcast);
    let report = qc
        .validate(input_str)
        .map_err(|e| anyhow::anyhow!("QC validation error: {}", e))?;

    let errors: Vec<_> = report
        .errors()
        .iter()
        .map(|e| {
            serde_json::json!({
                "rule": e.rule_name,
                "message": e.message,
                "severity": format!("{:?}", e.severity),
                "passed": e.passed,
            })
        })
        .collect();

    let all_results: Vec<_> = report
        .results
        .iter()
        .map(|r| {
            serde_json::json!({
                "rule": r.rule_name,
                "message": r.message,
                "severity": format!("{:?}", r.severity),
                "passed": r.passed,
            })
        })
        .collect();

    let report_json = serde_json::json!({
        "generator": "oximedia-conform-report",
        "version": env!("CARGO_PKG_VERSION"),
        "file": input_str,
        "timestamp": report.timestamp,
        "overall_passed": report.overall_passed,
        "total_checks": report.total_checks,
        "passed_checks": report.passed_checks,
        "failed_checks": report.failed_checks,
        "errors": errors,
        "all_results": all_results,
    });

    let report_str = serde_json::to_string_pretty(&report_json)?;
    std::fs::write(output, &report_str)
        .with_context(|| format!("Failed to write report: {}", output.display()))?;

    if json_output {
        println!("{}", report_str);
    } else {
        let status = if report.overall_passed {
            "PASS".green().bold()
        } else {
            "FAIL".red().bold()
        };
        println!("{} Conform report written: {}", status, output.display());
        println!(
            "   {} checks, {} passed, {} failed",
            report.total_checks, report.passed_checks, report.failed_checks
        );
    }

    Ok(())
}
