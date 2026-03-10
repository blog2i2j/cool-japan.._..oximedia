//! Media forensics command.
//!
//! Provides `oximedia forensics` for detecting tampering, splicing, compression
//! artefacts, and other authenticity indicators using `oximedia-forensics`.

use anyhow::{Context, Result};
use colored::Colorize;
use std::path::PathBuf;

/// Options for the `forensics` command.
pub struct ForensicsOptions {
    pub input: PathBuf,
    pub all: bool,
    pub tests: String,
    pub output_format: String,
    pub report: Option<PathBuf>,
}

/// Entry point called from `main.rs`.
pub async fn run_forensics(opts: ForensicsOptions, json_output: bool) -> Result<()> {
    use oximedia_forensics::ForensicsAnalyzer;

    let tests = resolve_tests(&opts);
    let config = build_config(&tests);

    // Read the input file into memory for analysis
    let data = std::fs::read(&opts.input)
        .with_context(|| format!("Failed to read input: {}", opts.input.display()))?;

    let analyzer = ForensicsAnalyzer::with_config(config);
    let report = analyzer
        .analyze(&data)
        .with_context(|| "Forensic analysis failed")?;

    let use_json = json_output || opts.output_format.to_lowercase() == "json";

    // Optionally persist the report
    if let Some(ref report_path) = opts.report {
        let content = render_report_text(&report, &opts.input);
        std::fs::write(report_path, &content)
            .with_context(|| format!("Failed to write report: {}", report_path.display()))?;
    }

    if use_json {
        output_json(&report, &opts)?;
    } else {
        output_text(&report, &opts)?;
    }

    Ok(())
}

/// Determine which tests to run based on `--all` and `--tests`.
fn resolve_tests(opts: &ForensicsOptions) -> Vec<String> {
    let all_tests = vec![
        "ela",
        "noise",
        "compression",
        "splicing",
        "metadata",
        "tampering",
        "geometric",
        "lighting",
    ];

    if opts.all || opts.tests.is_empty() {
        return all_tests.iter().map(|s| s.to_string()).collect();
    }

    opts.tests
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Build a `ForensicsConfig` enabling the requested tests.
fn build_config(tests: &[String]) -> oximedia_forensics::ForensicsConfig {
    use oximedia_forensics::ForensicsConfig;

    let enable_ela = tests.iter().any(|t| t == "ela");
    let enable_noise = tests.iter().any(|t| t == "noise");
    let enable_compression = tests.iter().any(|t| t == "compression");
    let enable_metadata = tests.iter().any(|t| t == "metadata");
    let enable_geometric = tests.iter().any(|t| t == "geometric");
    let enable_lighting = tests.iter().any(|t| t == "lighting");

    ForensicsConfig {
        enable_ela,
        enable_noise_analysis: enable_noise,
        enable_compression_analysis: enable_compression,
        enable_metadata_analysis: enable_metadata,
        enable_geometric_analysis: enable_geometric,
        enable_lighting_analysis: enable_lighting,
        ..Default::default()
    }
}

/// Render a plain-text report for writing to file.
fn render_report_text(report: &oximedia_forensics::TamperingReport, input: &PathBuf) -> String {
    let mut buf = String::new();
    buf.push_str(&format!("Forensic Analysis Report\n"));
    buf.push_str(&format!("File: {}\n\n", input.display()));
    buf.push_str(&format!(
        "Tampering detected: {}\n",
        report.tampering_detected
    ));
    buf.push_str(&format!(
        "Overall confidence: {:.2}%\n",
        report.overall_confidence * 100.0
    ));
    buf.push_str(&format!("Summary: {}\n\n", report.summary));
    buf.push_str("Tests:\n");
    for (name, test) in &report.tests {
        buf.push_str(&format!(
            "  {}: detected={} confidence={:.2}%\n",
            name,
            test.tampering_detected,
            test.confidence * 100.0
        ));
        for finding in &test.findings {
            buf.push_str(&format!("    - {}\n", finding));
        }
    }
    if !report.recommendations.is_empty() {
        buf.push_str("\nRecommendations:\n");
        for rec in &report.recommendations {
            buf.push_str(&format!("  * {}\n", rec));
        }
    }
    buf
}

/// Output results as JSON.
fn output_json(
    report: &oximedia_forensics::TamperingReport,
    opts: &ForensicsOptions,
) -> Result<()> {
    let tests_json: serde_json::Map<String, serde_json::Value> = report
        .tests
        .iter()
        .map(|(name, test)| {
            (
                name.clone(),
                serde_json::json!({
                    "tampering_detected": test.tampering_detected,
                    "confidence": test.confidence,
                    "confidence_level": format!("{:?}", test.confidence_level()),
                    "findings": test.findings,
                }),
            )
        })
        .collect();

    let obj = serde_json::json!({
        "file": opts.input.to_string_lossy(),
        "tampering_detected": report.tampering_detected,
        "overall_confidence": report.overall_confidence,
        "summary": report.summary,
        "recommendations": report.recommendations,
        "tests": tests_json,
        "report_file": opts.report.as_ref().map(|p| p.to_string_lossy().into_owned()),
    });

    println!("{}", serde_json::to_string_pretty(&obj)?);
    Ok(())
}

/// Output results as human-readable text.
fn output_text(
    report: &oximedia_forensics::TamperingReport,
    opts: &ForensicsOptions,
) -> Result<()> {
    use oximedia_forensics::ConfidenceLevel;

    println!("{}", "Media Forensics Analysis".green().bold());
    println!("  File: {}", opts.input.display());

    let confidence_pct = report.overall_confidence * 100.0;
    let level = ConfidenceLevel::from_score(report.overall_confidence);

    if report.tampering_detected {
        println!(
            "  Verdict:    {} ({:.1}% confidence - {:?})",
            "TAMPERING DETECTED".red().bold(),
            confidence_pct,
            level
        );
    } else {
        println!(
            "  Verdict:    {} ({:.1}% confidence - {:?})",
            "NO TAMPERING FOUND".green().bold(),
            confidence_pct,
            level
        );
    }

    println!("  Summary:    {}", report.summary);

    if !report.tests.is_empty() {
        println!("\n  {}", "Test Results:".cyan().bold());
        let mut test_names: Vec<&String> = report.tests.keys().collect();
        test_names.sort();
        for name in test_names {
            let test = &report.tests[name];
            let status = if test.tampering_detected {
                "FLAGGED".red().to_string()
            } else {
                "clean ".green().to_string()
            };
            println!(
                "    {} {:20} conf={:.1}%",
                status,
                name,
                test.confidence * 100.0
            );
            for finding in &test.findings {
                println!("              {}", finding.dimmed());
            }
        }
    }

    if !report.recommendations.is_empty() {
        println!("\n  {}", "Recommendations:".yellow().bold());
        for rec in &report.recommendations {
            println!("    * {}", rec);
        }
    }

    if let Some(ref rp) = opts.report {
        println!("\n{} Report saved: {}", "✓".green(), rp.display());
    }

    Ok(())
}
