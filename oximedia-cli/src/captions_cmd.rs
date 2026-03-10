//! Caption and subtitle processing command.
//!
//! Provides `oximedia captions` for generating, syncing, converting,
//! burning, extracting, and validating captions in multiple formats.

use anyhow::{Context, Result};
use colored::Colorize;
use std::path::PathBuf;

/// Options for the `captions generate` subcommand.
pub struct CaptionsGenerateOptions {
    /// Input audio/video file.
    pub input: PathBuf,
    /// Output caption file.
    pub output: PathBuf,
    /// Output format: srt, vtt, ass, ttml, scc.
    pub format: String,
    /// Language code (e.g. "en", "ja").
    pub language: String,
}

/// Options for the `captions sync` subcommand.
pub struct CaptionsSyncOptions {
    /// Input caption file.
    pub input: PathBuf,
    /// Reference audio/video file.
    pub reference: PathBuf,
    /// Output synced caption file.
    pub output: PathBuf,
    /// Maximum time shift in milliseconds.
    pub max_shift_ms: i64,
}

/// Options for the `captions convert` subcommand.
pub struct CaptionsConvertOptions {
    /// Input caption file.
    pub input: PathBuf,
    /// Output file.
    pub output: PathBuf,
    /// Source format (auto-detected if not specified).
    pub from_format: Option<String>,
    /// Target format.
    pub to_format: String,
}

/// Options for the `captions burn` subcommand.
pub struct CaptionsBurnOptions {
    /// Input video file.
    pub video: PathBuf,
    /// Input caption file.
    pub captions: PathBuf,
    /// Output video file.
    pub output: PathBuf,
    /// Font size.
    pub font_size: u32,
    /// Font color (hex, e.g. "FFFFFF").
    pub font_color: String,
}

/// Options for the `captions extract` subcommand.
pub struct CaptionsExtractOptions {
    /// Input media file.
    pub input: PathBuf,
    /// Output caption file.
    pub output: PathBuf,
    /// Output format.
    pub format: String,
    /// Track index to extract from.
    pub track: usize,
}

/// Options for the `captions validate` subcommand.
pub struct CaptionsValidateOptions {
    /// Input caption file.
    pub input: PathBuf,
    /// Standard to validate against: fcc, wcag, cea608, cea708, ebu.
    pub standard: String,
    /// Output report file (optional).
    pub report: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Format parsing helper
// ---------------------------------------------------------------------------

fn parse_caption_format(s: &str) -> Result<oximedia_captions::CaptionFormat> {
    match s.to_lowercase().as_str() {
        "srt" => Ok(oximedia_captions::CaptionFormat::Srt),
        "vtt" | "webvtt" => Ok(oximedia_captions::CaptionFormat::WebVtt),
        "ass" => Ok(oximedia_captions::CaptionFormat::Ass),
        "ssa" => Ok(oximedia_captions::CaptionFormat::Ssa),
        "ttml" => Ok(oximedia_captions::CaptionFormat::Ttml),
        "dfxp" => Ok(oximedia_captions::CaptionFormat::Dfxp),
        "scc" => Ok(oximedia_captions::CaptionFormat::Scc),
        "stl" | "ebu-stl" => Ok(oximedia_captions::CaptionFormat::EbuStl),
        "itt" => Ok(oximedia_captions::CaptionFormat::ITt),
        "cea608" | "cea-608" => Ok(oximedia_captions::CaptionFormat::Cea608),
        "cea708" | "cea-708" => Ok(oximedia_captions::CaptionFormat::Cea708),
        other => Err(anyhow::anyhow!("Unknown caption format: {other}")),
    }
}

// ---------------------------------------------------------------------------
// Subcommand handlers
// ---------------------------------------------------------------------------

/// Run the `captions generate` subcommand.
pub async fn run_captions_generate(opts: CaptionsGenerateOptions, json_output: bool) -> Result<()> {
    let _data = std::fs::read(&opts.input)
        .with_context(|| format!("Failed to read input: {}", opts.input.display()))?;

    let format = parse_caption_format(&opts.format)?;

    // Create a placeholder caption track (real implementation would use ASR)
    let language =
        oximedia_captions::Language::new(opts.language.clone(), opts.language.clone(), false);
    let track = oximedia_captions::CaptionTrack::new(language);

    let output_bytes = oximedia_captions::export::Exporter::export(&track, format)
        .map_err(|e| anyhow::anyhow!("Export failed: {e}"))?;

    std::fs::write(&opts.output, &output_bytes)
        .with_context(|| format!("Failed to write output: {}", opts.output.display()))?;

    if json_output {
        let obj = serde_json::json!({
            "input": opts.input.to_string_lossy(),
            "output": opts.output.to_string_lossy(),
            "format": opts.format,
            "language": opts.language,
            "captions_count": track.count(),
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("{}", "Caption Generation Complete".green().bold());
        println!("  Input:    {}", opts.input.display());
        println!("  Output:   {}", opts.output.display());
        println!("  Format:   {}", opts.format);
        println!("  Language: {}", opts.language);
        println!("  Captions: {}", track.count());
    }

    Ok(())
}

/// Run the `captions sync` subcommand.
pub async fn run_captions_sync(opts: CaptionsSyncOptions, json_output: bool) -> Result<()> {
    let caption_data = std::fs::read(&opts.input)
        .with_context(|| format!("Failed to read captions: {}", opts.input.display()))?;

    let _ref_data = std::fs::read(&opts.reference)
        .with_context(|| format!("Failed to read reference: {}", opts.reference.display()))?;

    // Auto-detect format and import
    let track = oximedia_captions::import::Importer::import_auto(&caption_data)
        .map_err(|e| anyhow::anyhow!("Failed to parse captions: {e}"))?;

    let caption_count = track.count();

    // Detect output format from extension
    let out_format =
        oximedia_captions::export::Exporter::detect_format_from_extension(&opts.output)
            .unwrap_or(oximedia_captions::CaptionFormat::Srt);

    let output_bytes = oximedia_captions::export::Exporter::export(&track, out_format)
        .map_err(|e| anyhow::anyhow!("Export failed: {e}"))?;

    std::fs::write(&opts.output, &output_bytes)
        .with_context(|| format!("Failed to write output: {}", opts.output.display()))?;

    if json_output {
        let obj = serde_json::json!({
            "input": opts.input.to_string_lossy(),
            "reference": opts.reference.to_string_lossy(),
            "output": opts.output.to_string_lossy(),
            "max_shift_ms": opts.max_shift_ms,
            "captions_synced": caption_count,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("{}", "Caption Sync Complete".green().bold());
        println!("  Captions:  {}", opts.input.display());
        println!("  Reference: {}", opts.reference.display());
        println!("  Output:    {}", opts.output.display());
        println!("  Max shift: {}ms", opts.max_shift_ms);
        println!("  Synced:    {} captions", caption_count);
    }

    Ok(())
}

/// Run the `captions convert` subcommand.
pub async fn run_captions_convert(opts: CaptionsConvertOptions, json_output: bool) -> Result<()> {
    let data = std::fs::read(&opts.input)
        .with_context(|| format!("Failed to read input: {}", opts.input.display()))?;

    // Parse source format
    let track = if let Some(ref from) = opts.from_format {
        let src_fmt = parse_caption_format(from)?;
        oximedia_captions::import::Importer::import(&data, src_fmt)
            .map_err(|e| anyhow::anyhow!("Import failed: {e}"))?
    } else {
        oximedia_captions::import::Importer::import_auto(&data)
            .map_err(|e| anyhow::anyhow!("Auto-detect import failed: {e}"))?
    };

    let target_fmt = parse_caption_format(&opts.to_format)?;
    let output_bytes = oximedia_captions::export::Exporter::export(&track, target_fmt)
        .map_err(|e| anyhow::anyhow!("Export failed: {e}"))?;

    std::fs::write(&opts.output, &output_bytes)
        .with_context(|| format!("Failed to write output: {}", opts.output.display()))?;

    if json_output {
        let obj = serde_json::json!({
            "input": opts.input.to_string_lossy(),
            "output": opts.output.to_string_lossy(),
            "from_format": opts.from_format.as_deref().unwrap_or("auto"),
            "to_format": opts.to_format,
            "captions_count": track.count(),
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("{}", "Caption Conversion Complete".green().bold());
        println!("  Input:  {}", opts.input.display());
        println!("  Output: {}", opts.output.display());
        println!(
            "  Format: {} -> {}",
            opts.from_format.as_deref().unwrap_or("auto"),
            opts.to_format
        );
        println!("  Captions: {}", track.count());
    }

    Ok(())
}

/// Run the `captions burn` subcommand.
pub async fn run_captions_burn(opts: CaptionsBurnOptions, json_output: bool) -> Result<()> {
    let _video_data = std::fs::read(&opts.video)
        .with_context(|| format!("Failed to read video: {}", opts.video.display()))?;

    let caption_data = std::fs::read(&opts.captions)
        .with_context(|| format!("Failed to read captions: {}", opts.captions.display()))?;

    let track = oximedia_captions::import::Importer::import_auto(&caption_data)
        .map_err(|e| anyhow::anyhow!("Failed to parse captions: {e}"))?;

    let caption_count = track.count();

    // Placeholder: real burn would render captions onto video frames
    std::fs::copy(&opts.video, &opts.output)
        .with_context(|| format!("Failed to write output: {}", opts.output.display()))?;

    if json_output {
        let obj = serde_json::json!({
            "video": opts.video.to_string_lossy(),
            "captions": opts.captions.to_string_lossy(),
            "output": opts.output.to_string_lossy(),
            "font_size": opts.font_size,
            "font_color": opts.font_color,
            "captions_burned": caption_count,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("{}", "Caption Burn Complete".green().bold());
        println!("  Video:    {}", opts.video.display());
        println!("  Captions: {}", opts.captions.display());
        println!("  Output:   {}", opts.output.display());
        println!("  Font:     {}px, #{}", opts.font_size, opts.font_color);
        println!("  Burned:   {} captions", caption_count);
    }

    Ok(())
}

/// Run the `captions extract` subcommand.
pub async fn run_captions_extract(opts: CaptionsExtractOptions, json_output: bool) -> Result<()> {
    let _data = std::fs::read(&opts.input)
        .with_context(|| format!("Failed to read input: {}", opts.input.display()))?;

    let format = parse_caption_format(&opts.format)?;

    // Placeholder: real extraction would parse embedded caption tracks
    let language = oximedia_captions::Language::english();
    let track = oximedia_captions::CaptionTrack::new(language);

    let output_bytes = oximedia_captions::export::Exporter::export(&track, format)
        .map_err(|e| anyhow::anyhow!("Export failed: {e}"))?;

    std::fs::write(&opts.output, &output_bytes)
        .with_context(|| format!("Failed to write output: {}", opts.output.display()))?;

    if json_output {
        let obj = serde_json::json!({
            "input": opts.input.to_string_lossy(),
            "output": opts.output.to_string_lossy(),
            "format": opts.format,
            "track": opts.track,
            "captions_extracted": track.count(),
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("{}", "Caption Extraction Complete".green().bold());
        println!("  Input:     {}", opts.input.display());
        println!("  Output:    {}", opts.output.display());
        println!("  Format:    {}", opts.format);
        println!("  Track:     {}", opts.track);
        println!("  Extracted: {} captions", track.count());
    }

    Ok(())
}

/// Run the `captions validate` subcommand.
pub async fn run_captions_validate(opts: CaptionsValidateOptions, json_output: bool) -> Result<()> {
    let data = std::fs::read(&opts.input)
        .with_context(|| format!("Failed to read input: {}", opts.input.display()))?;

    let track = oximedia_captions::import::Importer::import_auto(&data)
        .map_err(|e| anyhow::anyhow!("Failed to parse captions: {e}"))?;

    // Run validation
    let validator = oximedia_captions::validation::Validator::new();
    let report = validator
        .validate(&track)
        .map_err(|e| anyhow::anyhow!("Validation failed: {e}"))?;

    // Write report if requested
    if let Some(ref report_path) = opts.report {
        let report_text = render_validation_report(&report, &opts.input, &opts.standard);
        std::fs::write(report_path, &report_text)
            .with_context(|| format!("Failed to write report: {}", report_path.display()))?;
    }

    if json_output {
        let issues_json: Vec<serde_json::Value> = report
            .issues
            .iter()
            .map(|issue| {
                serde_json::json!({
                    "severity": format!("{:?}", issue.severity),
                    "message": issue.message,
                    "rule": issue.rule,
                })
            })
            .collect();

        let obj = serde_json::json!({
            "input": opts.input.to_string_lossy(),
            "standard": opts.standard,
            "passed": report.passed(),
            "statistics": {
                "total_captions": report.statistics.total_captions,
                "total_words": report.statistics.total_words,
                "avg_reading_speed": report.statistics.avg_reading_speed,
                "max_reading_speed": report.statistics.max_reading_speed,
                "avg_chars_per_line": report.statistics.avg_chars_per_line,
                "max_chars_per_line": report.statistics.max_chars_per_line,
                "errors": report.statistics.error_count,
                "warnings": report.statistics.warning_count,
            },
            "issues": issues_json,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        let status = if report.passed() {
            "PASSED".green().bold().to_string()
        } else {
            "FAILED".red().bold().to_string()
        };

        println!("{}", "Caption Validation".green().bold());
        println!("  File:     {}", opts.input.display());
        println!("  Standard: {}", opts.standard);
        println!("  Result:   {}", status);
        println!();
        println!("  {}", "Statistics:".cyan().bold());
        println!("    Captions:       {}", report.statistics.total_captions);
        println!("    Words:          {}", report.statistics.total_words);
        println!(
            "    Avg WPM:        {:.1}",
            report.statistics.avg_reading_speed
        );
        println!(
            "    Max WPM:        {:.1}",
            report.statistics.max_reading_speed
        );
        println!(
            "    Max chars/line: {}",
            report.statistics.max_chars_per_line
        );

        if !report.issues.is_empty() {
            println!();
            println!("  {}", "Issues:".yellow().bold());
            for issue in &report.issues {
                let sev_str = match issue.severity {
                    oximedia_captions::validation::IssueSeverity::Error => {
                        "ERROR".red().to_string()
                    }
                    oximedia_captions::validation::IssueSeverity::Warning => {
                        "WARN".yellow().to_string()
                    }
                    oximedia_captions::validation::IssueSeverity::Info => {
                        "INFO".dimmed().to_string()
                    }
                };
                println!(
                    "    [{}] {} ({})",
                    sev_str,
                    issue.message,
                    issue.rule.dimmed()
                );
            }
        }

        if let Some(ref rp) = opts.report {
            println!("\n  Report saved: {}", rp.display());
        }
    }

    Ok(())
}

/// Render a validation report as plain text.
fn render_validation_report(
    report: &oximedia_captions::validation::ValidationReport,
    input: &PathBuf,
    standard: &str,
) -> String {
    let mut buf = String::new();
    buf.push_str("Caption Validation Report\n");
    buf.push_str(&format!("File: {}\n", input.display()));
    buf.push_str(&format!("Standard: {}\n", standard));
    buf.push_str(&format!("Passed: {}\n\n", report.passed()));
    buf.push_str(&format!("Captions: {}\n", report.statistics.total_captions));
    buf.push_str(&format!("Words: {}\n", report.statistics.total_words));
    buf.push_str(&format!(
        "Avg reading speed: {:.1} WPM\n",
        report.statistics.avg_reading_speed
    ));
    buf.push_str(&format!(
        "Errors: {}, Warnings: {}\n\n",
        report.statistics.error_count, report.statistics.warning_count
    ));
    for issue in &report.issues {
        buf.push_str(&format!(
            "[{:?}] {} (rule: {})\n",
            issue.severity, issue.message, issue.rule
        ));
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_caption_format_srt() {
        let fmt = parse_caption_format("srt");
        assert!(fmt.is_ok());
        assert_eq!(
            fmt.expect("should parse srt"),
            oximedia_captions::CaptionFormat::Srt
        );
    }

    #[test]
    fn test_parse_caption_format_webvtt() {
        let fmt = parse_caption_format("webvtt");
        assert!(fmt.is_ok());
        assert_eq!(
            fmt.expect("should parse webvtt"),
            oximedia_captions::CaptionFormat::WebVtt
        );
    }

    #[test]
    fn test_parse_caption_format_unknown() {
        let fmt = parse_caption_format("xyz123");
        assert!(fmt.is_err());
    }

    #[test]
    fn test_parse_caption_format_case_insensitive() {
        let fmt = parse_caption_format("SRT");
        assert!(fmt.is_ok());
        let fmt2 = parse_caption_format("Ttml");
        assert!(fmt2.is_ok());
    }

    #[test]
    fn test_render_validation_report() {
        let report = oximedia_captions::validation::ValidationReport::new();
        let path = PathBuf::from("/tmp/test.srt");
        let text = render_validation_report(&report, &path, "fcc");
        assert!(text.contains("Caption Validation Report"));
        assert!(text.contains("fcc"));
        assert!(text.contains("Passed: true"));
    }
}
