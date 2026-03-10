//! EDL (Edit Decision List) parse, validate, export, and conform commands.
//!
//! Uses `oximedia-edl` for CMX3600 and other EDL format handling.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// EDL subcommands.
#[derive(Subcommand, Debug)]
pub enum EdlCommand {
    /// Parse an EDL file and display its events
    Parse {
        /// EDL input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// EDL format hint: cmx3600, cmx3400, cmx340, gvg, sony-bve9000
        #[arg(long, default_value = "cmx3600")]
        format: String,

        /// Strict parsing (reject non-standard lines)
        #[arg(long)]
        strict: bool,
    },

    /// Validate an EDL file and report issues
    Validate {
        /// EDL input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Validation level: strict, standard, lenient
        #[arg(long, default_value = "standard")]
        level: String,
    },

    /// Re-export an EDL in a different format
    Export {
        /// EDL input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Output EDL file
        #[arg(short, long)]
        output: PathBuf,

        /// Target format
        #[arg(long, default_value = "cmx3600")]
        format: String,

        /// Include comment lines in output
        #[arg(long)]
        comments: bool,

        /// Include clip names in output
        #[arg(long)]
        clip_names: bool,
    },

    /// Show conform report for an EDL against a media directory
    Conform {
        /// EDL input file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Media directory to search for source clips
        #[arg(long)]
        media_dir: Option<PathBuf>,
    },
}

/// Handle edl subcommand dispatch.
pub async fn handle_edl_command(cmd: EdlCommand, json_output: bool) -> Result<()> {
    match cmd {
        EdlCommand::Parse {
            file,
            format,
            strict,
        } => parse_edl_file(&file, &format, strict, json_output).await,
        EdlCommand::Validate { file, level } => validate_edl_file(&file, &level, json_output).await,
        EdlCommand::Export {
            file,
            output,
            format,
            comments,
            clip_names,
        } => export_edl(&file, &output, &format, comments, clip_names, json_output).await,
        EdlCommand::Conform { file, media_dir } => {
            conform_edl(&file, media_dir.as_deref(), json_output).await
        }
    }
}

/// Parse and display an EDL.
async fn parse_edl_file(
    path: &PathBuf,
    _format: &str,
    strict: bool,
    json_output: bool,
) -> Result<()> {
    use oximedia_edl::EdlParser;

    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read EDL file: {}", path.display()))?;

    let mut parser = EdlParser::new();
    parser.set_strict_mode(strict);

    let edl = parser
        .parse(&text)
        .with_context(|| format!("Failed to parse EDL: {}", path.display()))?;

    if json_output {
        let events: Vec<serde_json::Value> = edl
            .events
            .iter()
            .map(|e| {
                serde_json::json!({
                    "number": e.number,
                    "reel": e.reel,
                    "clip_name": e.clip_name,
                    "comments": e.comments,
                })
            })
            .collect();
        let obj = serde_json::json!({
            "file": path.to_string_lossy(),
            "format": format!("{:?}", edl.format),
            "title": edl.title,
            "event_count": edl.event_count(),
            "duration_seconds": edl.total_duration_seconds(),
            "events": events,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "EDL Parse".green().bold());
    println!("  File:     {}", path.display());
    println!("  Format:   {:?}", edl.format);
    if let Some(ref title) = edl.title {
        println!("  Title:    {}", title);
    }
    println!("  Events:   {}", edl.event_count());
    println!("  Duration: {:.2}s", edl.total_duration_seconds());

    if edl.events.is_empty() {
        println!("  {}", "(no events found)".dimmed());
    } else {
        println!("\n  {}", "Events:".cyan().bold());
        let max_display = 20_usize;
        for event in edl.events.iter().take(max_display) {
            let clip = event.clip_name.as_deref().unwrap_or(&event.reel);
            println!("    {:>4}  {}", event.number.to_string().yellow(), clip);
            if !event.comments.is_empty() {
                for c in &event.comments {
                    println!("          {}", c.dimmed());
                }
            }
        }
        if edl.event_count() > max_display {
            println!(
                "    {} ... and {} more",
                "".dimmed(),
                edl.event_count() - max_display
            );
        }
    }

    Ok(())
}

/// Validate an EDL file.
async fn validate_edl_file(path: &PathBuf, level: &str, json_output: bool) -> Result<()> {
    use oximedia_edl::{EdlParser, EdlValidator};

    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read EDL file: {}", path.display()))?;

    let edl = EdlParser::new()
        .parse(&text)
        .with_context(|| format!("Failed to parse EDL: {}", path.display()))?;

    let validator = match level.to_lowercase().as_str() {
        "strict" => EdlValidator::strict(),
        "lenient" => EdlValidator::lenient(),
        _ => EdlValidator::standard(),
    };

    let report = validator
        .validate(&edl)
        .with_context(|| "EDL validation failed with an internal error")?;

    if json_output {
        let obj = serde_json::json!({
            "file": path.to_string_lossy(),
            "level": level,
            "valid": !report.has_errors(),
            "error_count": report.error_count(),
            "warning_count": report.warning_count(),
            "errors": report.errors,
            "warnings": report.warnings,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "EDL Validate".green().bold());
    println!("  File:     {}", path.display());
    println!("  Level:    {}", level.cyan());

    if report.has_errors() {
        println!(
            "  Status:   {} ({} error(s))",
            "INVALID".red().bold(),
            report.error_count()
        );
        for err in &report.errors {
            println!("    {} {}", "ERROR:".red(), err);
        }
    } else {
        println!("  Status:   {}", "VALID".green().bold());
    }

    if report.has_warnings() {
        println!("  Warnings: {}", report.warning_count());
        for warn in &report.warnings {
            println!("    {} {}", "WARN:".yellow(), warn);
        }
    }

    Ok(())
}

/// Re-export an EDL in a (possibly different) format.
async fn export_edl(
    path: &PathBuf,
    output: &PathBuf,
    _format: &str,
    comments: bool,
    clip_names: bool,
    json_output: bool,
) -> Result<()> {
    use oximedia_edl::{EdlGenerator, EdlParser};

    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read EDL file: {}", path.display()))?;

    let edl = EdlParser::new()
        .parse(&text)
        .with_context(|| format!("Failed to parse EDL: {}", path.display()))?;

    let mut generator = EdlGenerator::new();
    generator.set_include_comments(comments);
    generator.set_include_clip_names(clip_names);

    let out_text = generator
        .generate(&edl)
        .with_context(|| "Failed to generate EDL output")?;

    std::fs::write(output, &out_text)
        .with_context(|| format!("Failed to write EDL: {}", output.display()))?;

    if json_output {
        let obj = serde_json::json!({
            "operation": "edl_export",
            "input": path.to_string_lossy(),
            "output": output.to_string_lossy(),
            "events": edl.event_count(),
            "bytes_written": out_text.len(),
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "EDL Export".green().bold());
    println!("  Input:   {}", path.display());
    println!("  Output:  {}", output.display());
    println!("  Events:  {}", edl.event_count());
    println!("  Bytes:   {}", out_text.len());
    println!("{} Written: {}", "✓".green(), output.display());

    Ok(())
}

/// Show a conform report for an EDL.
async fn conform_edl(
    path: &PathBuf,
    media_dir: Option<&std::path::Path>,
    json_output: bool,
) -> Result<()> {
    use oximedia_edl::EdlParser;

    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read EDL file: {}", path.display()))?;

    let edl = EdlParser::new()
        .parse(&text)
        .with_context(|| format!("Failed to parse EDL: {}", path.display()))?;

    // Collect unique reels
    let mut reels: Vec<String> = edl.events.iter().map(|e| e.reel.clone()).collect();
    reels.sort();
    reels.dedup();

    if json_output {
        let reel_statuses: Vec<serde_json::Value> = reels
            .iter()
            .map(|r| {
                let found = media_dir.map_or(false, |d| {
                    d.join(r).exists()
                        || d.join(format!("{}.mkv", r)).exists()
                        || d.join(format!("{}.mov", r)).exists()
                });
                serde_json::json!({
                    "reel": r,
                    "status": if found { "online" } else { "offline" },
                })
            })
            .collect();
        let obj = serde_json::json!({
            "file": path.to_string_lossy(),
            "media_dir": media_dir.map(|d| d.to_string_lossy().into_owned()),
            "event_count": edl.event_count(),
            "unique_reels": reels.len(),
            "reels": reel_statuses,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "EDL Conform Report".green().bold());
    println!("  EDL:       {}", path.display());
    if let Some(dir) = media_dir {
        println!("  Media dir: {}", dir.display());
    }
    println!("  Events:    {}", edl.event_count());
    println!("  Reels:     {}", reels.len());

    let mut online = 0_usize;
    let mut offline = 0_usize;

    println!("\n  {}", "Reel Status:".cyan().bold());
    for reel in &reels {
        let found = media_dir.map_or(false, |d| {
            d.join(reel).exists()
                || d.join(format!("{}.mkv", reel)).exists()
                || d.join(format!("{}.mov", reel)).exists()
        });
        if found {
            online += 1;
            println!("    {} {}", "ONLINE ".green(), reel);
        } else {
            offline += 1;
            let marker = if media_dir.is_some() {
                "OFFLINE".red()
            } else {
                "UNKNOWN".yellow()
            };
            println!("    {} {}", marker, reel);
        }
    }

    println!(
        "\n  Online: {}  Offline/Unknown: {}",
        online.to_string().green(),
        offline.to_string().red()
    );

    Ok(())
}
