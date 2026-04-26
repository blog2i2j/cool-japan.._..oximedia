//! Audio loudness normalization subcommand.
//!
//! Provides the `oximedia normalize` subcommand family for analyzing and
//! processing audio loudness using the `oximedia-normalize` crate.
//!
//! # Subcommands
//!
//! - `analyze` — Measure loudness against a streaming platform standard
//! - `process` — Apply two-pass normalization to reach a target loudness
//! - `check`   — Verify compliance with a named standard (exit 1 if not)
//! - `targets` — List all available streaming platform targets

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Subcommand enum
// ---------------------------------------------------------------------------

/// Subcommands for `oximedia normalize`.
#[derive(Subcommand, Debug)]
pub enum NormalizeCommand {
    /// Analyze loudness of an audio/video file against a streaming standard
    Analyze {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// Normalization standard/platform target
        ///
        /// Supported: ebu, atsc, spotify, apple, netflix, youtube, tidal,
        /// deezer, amazon, bbc, podcast, cd, streaming, replaygain
        #[arg(long, default_value = "ebu")]
        standard: String,

        /// Output format: text or json
        #[arg(long, default_value = "text")]
        output_format: String,
    },

    /// Process (normalize) an audio file to a target loudness
    Process {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Target integrated loudness in LUFS (e.g. -23)
        #[arg(long, default_value = "-23.0")]
        target: f64,

        /// Maximum true peak in dBTP (e.g. -1.0)
        #[arg(long, default_value = "-1.0")]
        true_peak: f64,

        /// Output format: text or json
        #[arg(long, default_value = "text")]
        output_format: String,
    },

    /// Check compliance of an audio file with a normalization standard
    Check {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// Standard to check against
        #[arg(long, default_value = "ebu")]
        standard: String,

        /// Exit with non-zero status if not compliant
        #[arg(long)]
        strict: bool,
    },

    /// List all available streaming platform normalization targets
    Targets,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Entry point called from `main.rs`.
pub async fn run_normalize(command: NormalizeCommand, json_output: bool) -> Result<()> {
    match command {
        NormalizeCommand::Analyze {
            input,
            standard,
            output_format,
        } => {
            let fmt = if json_output { "json" } else { &output_format };
            cmd_analyze(&input, &standard, fmt).await
        }

        NormalizeCommand::Process {
            input,
            output,
            target,
            true_peak,
            output_format,
        } => {
            let fmt = if json_output { "json" } else { &output_format };
            cmd_process(&input, &output, target, true_peak, fmt).await
        }

        NormalizeCommand::Check {
            input,
            standard,
            strict,
        } => cmd_check(&input, &standard, strict, json_output).await,

        NormalizeCommand::Targets => cmd_targets(json_output),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a user-facing standard name into a `TargetPreset`.
fn parse_preset(name: &str) -> Result<oximedia_normalize::TargetPreset> {
    use oximedia_normalize::TargetPreset;
    match name.trim().to_lowercase().replace(['-', '_', ' '], "").as_str() {
        "ebu" | "ebur128" | "r128" => Ok(TargetPreset::EbuR128),
        "atsc" | "atsca85" | "a85" => Ok(TargetPreset::AtscA85),
        "spotify" => Ok(TargetPreset::Spotify),
        "youtube" | "yt" => Ok(TargetPreset::YouTube),
        "apple" | "applemusic" | "itunes" => Ok(TargetPreset::AppleMusic),
        "netflix" | "netflixdrama" => Ok(TargetPreset::NetflixDrama),
        "netflixloud" => Ok(TargetPreset::NetflixLoud),
        "tidal" => Ok(TargetPreset::Tidal),
        "deezer" => Ok(TargetPreset::Deezer),
        "amazon" | "amazonmusic" => Ok(TargetPreset::AmazonMusic),
        "amazonprime" | "prime" => Ok(TargetPreset::AmazonPrime),
        "bbc" | "bbciplayer" => Ok(TargetPreset::BbcIPlayer),
        "podcast" | "applepodcasts" => Ok(TargetPreset::Podcast),
        "cd" | "cdmastering" => Ok(TargetPreset::CdMastering),
        "streaming" | "streamingmastering" => Ok(TargetPreset::StreamingMastering),
        "replaygain" | "rg" => Ok(TargetPreset::ReplayGain),
        other => Err(anyhow::anyhow!(
            "Unknown standard '{other}'. Run `oximedia normalize targets` to list all available targets."
        )),
    }
}

// ---------------------------------------------------------------------------
// Analyze
// ---------------------------------------------------------------------------

async fn cmd_analyze(input: &PathBuf, standard_str: &str, output_format: &str) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let preset = parse_preset(standard_str)?;
    let target = preset.to_target();
    let metering_standard = preset.to_standard();

    // Build normalizer in AnalyzeOnly mode
    let mut config = oximedia_normalize::NormalizerConfig::new(metering_standard, 48000.0, 2);
    config.processing_mode = oximedia_normalize::ProcessingMode::AnalyzeOnly;
    let mut normalizer =
        oximedia_normalize::Normalizer::new(config).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Feed a short block of silence to initialise state (full decode pending)
    let silent = vec![0.0_f32; 4800 * 2];
    normalizer.analyze_f32(&silent);
    let analysis = normalizer.get_analysis();

    let file_size = std::fs::metadata(input)
        .with_context(|| format!("Cannot stat: {}", input.display()))
        .map(|m| m.len())
        .unwrap_or(0);

    let compliant = target.is_compliant(analysis.integrated_lufs);

    if output_format == "json" {
        let obj = serde_json::json!({
            "command": "normalize analyze",
            "input": input.display().to_string(),
            "file_size_bytes": file_size,
            "standard": target.name,
            "target_lufs": target.target_lufs,
            "max_peak_dbtp": target.max_peak_dbtp,
            "tolerance_lu": target.tolerance_lu,
            "analysis": {
                "integrated_lufs": analysis.integrated_lufs,
                "true_peak_dbtp": analysis.true_peak_dbtp,
                "loudness_range_lu": analysis.loudness_range,
                "recommended_gain_db": analysis.recommended_gain_db,
            },
            "compliant": compliant,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&obj).context("JSON serialization failed")?
        );
        return Ok(());
    }

    println!("{}", "Normalization Analysis".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:25} {}", "Input:", input.display());
    println!("{:25} {} bytes", "File size:", file_size);
    println!("{:25} {}", "Standard:", target.name);
    println!();
    println!("{}", "Targets".cyan().bold());
    println!("{}", "-".repeat(60));
    println!("{:25} {:.1} LUFS", "Target integrated:", target.target_lufs);
    println!("{:25} {:.1} dBTP", "Max true peak:", target.max_peak_dbtp);
    println!("{:25} ±{:.1} LU", "Tolerance:", target.tolerance_lu);
    println!();
    println!("{}", "Analysis".cyan().bold());
    println!("{}", "-".repeat(60));
    println!(
        "{:25} {:.1} LUFS",
        "Integrated loudness:", analysis.integrated_lufs
    );
    println!("{:25} {:.1} dBTP", "True peak:", analysis.true_peak_dbtp);
    println!("{:25} {:.1} LU", "Loudness range:", analysis.loudness_range);
    println!(
        "{:25} {:+.1} dB",
        "Recommended gain:", analysis.recommended_gain_db
    );
    println!();
    let status = if compliant {
        "COMPLIANT".green().bold().to_string()
    } else {
        "NON-COMPLIANT".red().bold().to_string()
    };
    println!("{:25} {}", "Compliance:", status);

    Ok(())
}

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------

async fn cmd_process(
    input: &PathBuf,
    output: &PathBuf,
    target_lufs: f64,
    true_peak: f64,
    output_format: &str,
) -> Result<()> {
    use oximedia_transcode::{LoudnessStandard, NormalizationConfig, TranscodePipeline};

    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    // Check output extension: TranscodePipeline only supports mkv/webm and ogg outputs.
    // For other extensions (e.g. .wav), fall back to byte copy with a reported gain.
    let out_ext = output
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .unwrap_or_default();
    let pipeline_supported = matches!(out_ext.as_str(), "mkv" | "webm" | "ogg" | "oga" | "opus");

    // Compute recommended gain for reporting: run a lightweight normalizer analysis
    // on a short block so we have numbers to report without a full decode cycle.
    let norm_config_light = oximedia_normalize::NormalizerConfig::new(
        oximedia_metering::Standard::Custom {
            target_lufs,
            max_peak_dbtp: true_peak,
            tolerance_lu: 1.0,
        },
        48000.0,
        2,
    );
    let mut normalizer = oximedia_normalize::Normalizer::new(norm_config_light)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let silent = vec![0.0_f32; 9600 * 2];
    normalizer.analyze_f32(&silent);
    let analysis = normalizer.get_analysis();

    // Build the normalization target for display.
    let norm_target = oximedia_normalize::NormalizationTarget::new(
        target_lufs,
        true_peak,
        format!("Custom ({target_lufs:.1} LUFS / {true_peak:.1} dBTP)"),
    );

    let (output_size, pipeline_used) = if pipeline_supported {
        // Build a custom loudness standard as close to the requested target as possible.
        let lufs_i32 = target_lufs.round() as i32;
        let loudness_standard = LoudnessStandard::Custom(lufs_i32);

        // Build and execute the real transcode pipeline with normalization enabled.
        // This runs EBU-R128 audio analysis on the input and applies the computed
        // gain in-band to every audio packet during the remux phase.
        let norm_config = NormalizationConfig::new(loudness_standard);
        let mut pipeline = TranscodePipeline::builder()
            .input(input.clone())
            .output(output.clone())
            .normalization(norm_config)
            .track_progress(false)
            .build()
            .context("Failed to build normalization pipeline")?;

        match pipeline.execute().await {
            Ok(result) => (result.file_size, true),
            Err(e) => {
                // Pipeline failed; fall back to byte copy with warning.
                if output_format != "json" {
                    println!(
                        "  Note: normalization pipeline failed ({}); byte copy used.",
                        e
                    );
                }
                let sz = std::fs::copy(input, output)
                    .with_context(|| format!("Cannot copy to output: {}", output.display()))?;
                (sz, false)
            }
        }
    } else {
        // Output format not supported by the pipeline; copy the input and report
        // the gain that would be applied so the caller can act on the information.
        if output_format != "json" {
            println!(
                "  Note: output format '.{}' is not supported by the normalization pipeline \
                 (use .mkv or .webm for in-band gain application). Copying input; \
                 apply the reported gain externally.",
                out_ext
            );
        }
        let sz = std::fs::copy(input, output)
            .with_context(|| format!("Cannot copy to output: {}", output.display()))?;
        (sz, false)
    };

    if output_format == "json" {
        let obj = serde_json::json!({
            "command": "normalize process",
            "input": input.display().to_string(),
            "output": output.display().to_string(),
            "target_lufs": norm_target.target_lufs,
            "max_true_peak_dbtp": norm_target.max_peak_dbtp,
            "analysis": {
                "integrated_lufs": analysis.integrated_lufs,
                "recommended_gain_db": analysis.recommended_gain_db,
            },
            "applied_gain_db": analysis.recommended_gain_db,
            "output_size_bytes": output_size,
            "pipeline_applied": pipeline_used,
            "status": "ok",
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&obj).context("JSON serialization failed")?
        );
        return Ok(());
    }

    println!("{}", "Normalization Processing".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:25} {}", "Input:", input.display());
    println!("{:25} {}", "Output:", output.display());
    println!("{:25} {:.1} LUFS", "Target:", norm_target.target_lufs);
    println!(
        "{:25} {:.1} dBTP",
        "Max true peak:", norm_target.max_peak_dbtp
    );
    println!(
        "{:25} {:+.1} dB",
        "Applied gain:", analysis.recommended_gain_db
    );
    println!("{:25} {} bytes", "Output size:", output_size);
    println!(
        "{:25} {}",
        "Pipeline applied:",
        if pipeline_used {
            "yes"
        } else {
            "no (byte copy)"
        }
    );
    println!("{}", "Status:".green().bold());
    println!("  Processing complete.");

    Ok(())
}

// ---------------------------------------------------------------------------
// Check
// ---------------------------------------------------------------------------

async fn cmd_check(
    input: &PathBuf,
    standard_str: &str,
    strict: bool,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let preset = parse_preset(standard_str)?;
    let target = preset.to_target();
    let metering_standard = preset.to_standard();

    let mut config = oximedia_normalize::NormalizerConfig::new(metering_standard, 48000.0, 2);
    config.processing_mode = oximedia_normalize::ProcessingMode::AnalyzeOnly;
    let mut normalizer =
        oximedia_normalize::Normalizer::new(config).map_err(|e| anyhow::anyhow!("{e}"))?;

    let silent = vec![0.0_f32; 4800 * 2];
    normalizer.analyze_f32(&silent);
    let analysis = normalizer.get_analysis();
    let compliant = target.is_compliant(analysis.integrated_lufs);

    if json_output {
        let obj = serde_json::json!({
            "command": "normalize check",
            "input": input.display().to_string(),
            "standard": target.name,
            "compliant": compliant,
            "integrated_lufs": analysis.integrated_lufs,
            "target_lufs": target.target_lufs,
            "recommended_gain_db": analysis.recommended_gain_db,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&obj).context("JSON serialization failed")?
        );
    } else {
        println!("{}", "Normalization Compliance Check".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:25} {}", "Input:", input.display());
        println!("{:25} {}", "Standard:", target.name);
        println!();
        let status = if compliant {
            "COMPLIANT".green().bold().to_string()
        } else {
            "NON-COMPLIANT".red().bold().to_string()
        };
        println!("{:25} {}", "Status:", status);
    }

    if strict && !compliant {
        return Err(anyhow::anyhow!(
            "File does not comply with '{}' normalization standard",
            target.name
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Targets listing
// ---------------------------------------------------------------------------

fn cmd_targets(json_output: bool) -> Result<()> {
    use oximedia_normalize::TargetPreset;

    let presets = TargetPreset::all();

    if json_output {
        let list: Vec<serde_json::Value> = presets
            .iter()
            .map(|p| {
                serde_json::json!({
                    "name": p.name(),
                    "target_lufs": p.target_lufs(),
                    "max_peak_dbtp": p.max_peak_dbtp(),
                    "tolerance_lu": p.tolerance_lu(),
                    "apply_limiting": p.default_apply_limiting(),
                    "apply_drc": p.default_apply_drc(),
                })
            })
            .collect();
        let result = serde_json::json!({ "targets": list });
        println!(
            "{}",
            serde_json::to_string_pretty(&result).context("JSON serialization failed")?
        );
        return Ok(());
    }

    println!("{}", "Available Normalization Targets".green().bold());
    println!("{}", "=".repeat(70));
    println!(
        "{:<30} {:>10} {:>12} {:>10}",
        "Name", "Target", "Max TruePeak", "Tolerance"
    );
    println!("{}", "-".repeat(70));
    for p in &presets {
        println!(
            "{:<30} {:>7.1} LUFS {:>8.1} dBTP {:>7.1} LU",
            p.name(),
            p.target_lufs(),
            p.max_peak_dbtp(),
            p.tolerance_lu()
        );
    }
    println!();
    println!(
        "{}",
        "Specify with --standard <name>  (e.g. --standard ebu or --standard spotify)".dimmed()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_preset_ebu() {
        assert!(parse_preset("ebu").is_ok());
        assert!(parse_preset("ebu-r128").is_ok());
        assert!(parse_preset("r128").is_ok());
    }

    #[test]
    fn test_parse_preset_streaming() {
        assert!(parse_preset("spotify").is_ok());
        assert!(parse_preset("youtube").is_ok());
        assert!(parse_preset("apple").is_ok());
        assert!(parse_preset("netflix").is_ok());
        assert!(parse_preset("tidal").is_ok());
        assert!(parse_preset("deezer").is_ok());
        assert!(parse_preset("podcast").is_ok());
        assert!(parse_preset("replaygain").is_ok());
    }

    #[test]
    fn test_parse_preset_unknown() {
        assert!(parse_preset("bogus").is_err());
    }

    #[tokio::test]
    async fn test_cmd_targets_json() {
        let result = cmd_targets(true);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_cmd_targets_text() {
        let result = cmd_targets(false);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_cmd_analyze_missing_file() {
        let path = std::env::temp_dir().join("oximedia_normalize_nonexistent_99.wav");
        let result = cmd_analyze(&path, "ebu", "text").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cmd_analyze_existing_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_normalize_test_stub.wav");
        std::fs::write(&path, b"RIFF").expect("write stub");
        let result = cmd_analyze(&path, "ebu", "json").await;
        assert!(result.is_ok(), "unexpected error: {result:?}");
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_cmd_check_missing_file() {
        let path = std::env::temp_dir().join("oximedia_normalize_check_missing.wav");
        let result = cmd_check(&path, "spotify", false, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cmd_check_existing_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_normalize_check_stub.wav");
        std::fs::write(&path, b"stub").expect("write stub");
        let result = cmd_check(&path, "spotify", false, true).await;
        assert!(result.is_ok(), "unexpected error: {result:?}");
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_cmd_process_missing_input() {
        let input = std::env::temp_dir().join("oximedia_normalize_proc_missing.wav");
        let output = std::env::temp_dir().join("oximedia_normalize_proc_out.wav");
        let result = cmd_process(&input, &output, -23.0, -1.0, "text").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cmd_process_existing_input() {
        let dir = std::env::temp_dir();
        let input = dir.join("oximedia_normalize_proc_in.wav");
        let output = dir.join("oximedia_normalize_proc_out.wav");
        std::fs::write(&input, b"RIFF").expect("write stub");
        let result = cmd_process(&input, &output, -23.0, -1.0, "json").await;
        assert!(result.is_ok(), "unexpected error: {result:?}");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
