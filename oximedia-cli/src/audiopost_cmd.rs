//! Audio post-production CLI commands.
//!
//! Provides audiopost commands: adr, mix, stems, delivery, restore.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Audio post-production subcommands.
#[derive(Subcommand, Debug)]
pub enum AudiopostCommand {
    /// Manage ADR (Automated Dialogue Replacement) sessions
    Adr {
        /// Input video/audio file for reference
        #[arg(short, long)]
        input: PathBuf,

        /// Output ADR session file
        #[arg(short, long)]
        output: PathBuf,

        /// Cue list file (CSV or JSON)
        #[arg(long)]
        cue_list: Option<PathBuf>,

        /// Starting timecode (e.g. "01:00:00:00")
        #[arg(long)]
        timecode_start: Option<String>,

        /// Pre-roll duration in seconds
        #[arg(long, default_value = "3.0")]
        pre_roll: f64,

        /// Post-roll duration in seconds
        #[arg(long, default_value = "2.0")]
        post_roll: f64,

        /// Sample rate
        #[arg(long, default_value = "48000")]
        sample_rate: u32,
    },

    /// Mix multiple audio tracks together
    Mix {
        /// Input audio files (multiple)
        #[arg(short, long, required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Comma-separated gain levels in dB for each input (e.g. "0,-3,-6")
        #[arg(long)]
        levels: Option<String>,

        /// Normalize output loudness
        #[arg(long)]
        normalize: bool,

        /// Target loudness in LUFS
        #[arg(long, default_value = "-14.0")]
        target_lufs: f64,

        /// Output format: wav, flac, ogg
        #[arg(long)]
        format: Option<String>,

        /// Bit depth: 16, 24, 32
        #[arg(long, default_value = "24")]
        bit_depth: u32,
    },

    /// Create or export audio stems
    Stems {
        /// Input multi-track or mixed audio file
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for stem files
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Comma-separated stem names (e.g. "dialogue,music,effects,foley,ambience")
        #[arg(long)]
        stem_names: Option<String>,

        /// Output format: wav, flac
        #[arg(long, default_value = "wav")]
        format: String,

        /// Sample rate for exported stems
        #[arg(long)]
        sample_rate: Option<u32>,

        /// Bit depth: 16, 24, 32
        #[arg(long, default_value = "24")]
        bit_depth: u32,
    },

    /// Check audio against delivery specifications
    Delivery {
        /// Input audio file
        #[arg(short, long)]
        input: PathBuf,

        /// Delivery spec: broadcast, cinema, streaming, podcast
        #[arg(long, default_value = "broadcast")]
        spec: String,

        /// Attempt to fix non-compliant audio
        #[arg(long)]
        fix: bool,

        /// Output file for fixed audio (required with --fix)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output format: text, json
        #[arg(long, default_value = "text")]
        output_format: String,
    },

    /// Restore degraded audio (declip, dehum, decrackle, denoise)
    Restore {
        /// Input audio file
        #[arg(short, long)]
        input: PathBuf,

        /// Output audio file
        #[arg(short, long)]
        output: PathBuf,

        /// Apply declipping
        #[arg(long)]
        declip: bool,

        /// Apply hum removal (50/60 Hz and harmonics)
        #[arg(long)]
        dehum: bool,

        /// Apply crackle removal
        #[arg(long)]
        decrackle: bool,

        /// Apply noise reduction
        #[arg(long)]
        denoise: bool,

        /// Apply all restoration steps
        #[arg(long)]
        all: bool,

        /// Restoration strength (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        strength: f64,
    },
}

/// Handle audiopost command dispatch.
pub async fn handle_audiopost_command(command: AudiopostCommand, json_output: bool) -> Result<()> {
    match command {
        AudiopostCommand::Adr {
            input,
            output,
            cue_list,
            timecode_start,
            pre_roll,
            post_roll,
            sample_rate,
        } => {
            handle_adr(
                &input,
                &output,
                cue_list.as_deref(),
                timecode_start.as_deref(),
                pre_roll,
                post_roll,
                sample_rate,
                json_output,
            )
            .await
        }
        AudiopostCommand::Mix {
            inputs,
            output,
            levels,
            normalize,
            target_lufs,
            format,
            bit_depth,
        } => {
            handle_mix(
                &inputs,
                &output,
                levels.as_deref(),
                normalize,
                target_lufs,
                format.as_deref(),
                bit_depth,
                json_output,
            )
            .await
        }
        AudiopostCommand::Stems {
            input,
            output_dir,
            stem_names,
            format,
            sample_rate,
            bit_depth,
        } => {
            handle_stems(
                &input,
                &output_dir,
                stem_names.as_deref(),
                &format,
                sample_rate,
                bit_depth,
                json_output,
            )
            .await
        }
        AudiopostCommand::Delivery {
            input,
            spec,
            fix,
            output,
            output_format,
        } => {
            handle_delivery(
                &input,
                &spec,
                fix,
                output.as_deref(),
                if json_output { "json" } else { &output_format },
            )
            .await
        }
        AudiopostCommand::Restore {
            input,
            output,
            declip,
            dehum,
            decrackle,
            denoise,
            all,
            strength,
        } => {
            handle_restore(
                &input,
                &output,
                declip || all,
                dehum || all,
                decrackle || all,
                denoise || all,
                strength,
                json_output,
            )
            .await
        }
    }
}

/// Parse delivery target string.
fn parse_delivery_target(spec: &str) -> Result<oximedia_audiopost::delivery_spec::DeliveryTarget> {
    use oximedia_audiopost::delivery_spec::DeliveryTarget;
    match spec {
        "broadcast" => Ok(DeliveryTarget::Broadcast),
        "cinema" => Ok(DeliveryTarget::Cinema),
        "streaming" => Ok(DeliveryTarget::Streaming),
        "podcast" => Ok(DeliveryTarget::Podcast),
        other => Err(anyhow::anyhow!(
            "Unknown delivery spec '{}'. Expected: broadcast, cinema, streaming, podcast",
            other
        )),
    }
}

/// Handle ADR session creation.
async fn handle_adr(
    input: &PathBuf,
    output: &PathBuf,
    cue_list: Option<&std::path::Path>,
    timecode_start: Option<&str>,
    pre_roll: f64,
    post_roll: f64,
    sample_rate: u32,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let tc_start = timecode_start.unwrap_or("01:00:00:00");
    let session = oximedia_audiopost::adr::AdrSession::new("ADR Session", sample_rate);

    if json_output {
        let result = serde_json::json!({
            "status": "adr_session_created",
            "input": input.display().to_string(),
            "output": output.display().to_string(),
            "timecode_start": tc_start,
            "pre_roll": pre_roll,
            "post_roll": post_roll,
            "sample_rate": sample_rate,
            "cue_list": cue_list.map(|p| p.display().to_string()),
            "cue_count": session.cue_count(),
        });
        let result_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", result_str);
    } else {
        println!("{}", "ADR Session".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {}", "Timecode start:", tc_start);
        println!("{:20} {:.1}s", "Pre-roll:", pre_roll);
        println!("{:20} {:.1}s", "Post-roll:", post_roll);
        println!("{:20} {} Hz", "Sample rate:", sample_rate);

        if let Some(cue_path) = cue_list {
            println!("{:20} {}", "Cue list:", cue_path.display());
        }

        println!();
        println!(
            "{}",
            "ADR session initialized. Use cue list import for batch setup.".dimmed()
        );
    }

    Ok(())
}

/// Handle audio mixing.
async fn handle_mix(
    inputs: &[PathBuf],
    output: &PathBuf,
    levels: Option<&str>,
    normalize: bool,
    target_lufs: f64,
    format: Option<&str>,
    bit_depth: u32,
    json_output: bool,
) -> Result<()> {
    // Validate inputs exist
    for (i, input) in inputs.iter().enumerate() {
        if !input.exists() {
            return Err(anyhow::anyhow!(
                "Input file {} not found: {}",
                i + 1,
                input.display()
            ));
        }
    }

    // Parse levels
    let parsed_levels: Vec<f64> = if let Some(lvl_str) = levels {
        lvl_str
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<f64>()
                    .map_err(|e| anyhow::anyhow!("Invalid level value '{}': {}", s, e))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![0.0; inputs.len()]
    };

    let output_format =
        format.unwrap_or_else(|| output.extension().and_then(|e| e.to_str()).unwrap_or("wav"));

    // Create mixing console (validates parameters)
    let _console = oximedia_audiopost::mixing::MixingConsole::new(48000, 512)
        .map_err(|e| anyhow::anyhow!("Failed to create mixing console: {}", e))?;

    if json_output {
        let result = serde_json::json!({
            "status": "mix_pending",
            "inputs": inputs.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
            "output": output.display().to_string(),
            "levels_db": parsed_levels,
            "normalize": normalize,
            "target_lufs": if normalize { Some(target_lufs) } else { None },
            "format": output_format,
            "bit_depth": bit_depth,
            "input_count": inputs.len(),
            "message": "Mixing console configured; full render pending audio pipeline integration",
        });
        let result_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", result_str);
    } else {
        println!("{}", "Audio Mix".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {}", "Format:", output_format);
        println!("{:20} {}-bit", "Bit depth:", bit_depth);
        if normalize {
            println!("{:20} {} LUFS", "Target loudness:", target_lufs);
        }
        println!();

        println!("{}", "Input Tracks".cyan().bold());
        println!("{}", "-".repeat(60));
        for (i, input) in inputs.iter().enumerate() {
            let level = parsed_levels.get(i).copied().unwrap_or(0.0);
            println!("  [{:>2}] {:30} {:+.1} dB", i + 1, input.display(), level);
        }
        println!();

        println!(
            "{}",
            "Note: Full mixing pipeline pending audio decoding integration.".yellow()
        );
        println!(
            "{}",
            "Mixing console is ready; audio decoding will enable end-to-end render.".dimmed()
        );
    }

    Ok(())
}

/// Handle stem creation/export.
async fn handle_stems(
    input: &PathBuf,
    output_dir: &PathBuf,
    stem_names: Option<&str>,
    format: &str,
    sample_rate: Option<u32>,
    bit_depth: u32,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let stems: Vec<String> = if let Some(names) = stem_names {
        names.split(',').map(|s| s.trim().to_string()).collect()
    } else {
        oximedia_audiopost::stems::StemType::standard_types()
            .iter()
            .map(|t| t.as_str().to_string())
            .collect()
    };

    if json_output {
        let result = serde_json::json!({
            "status": "stems_pending",
            "input": input.display().to_string(),
            "output_dir": output_dir.display().to_string(),
            "stems": stems,
            "format": format,
            "sample_rate": sample_rate,
            "bit_depth": bit_depth,
            "message": "Stem exporter configured; full render pending audio pipeline integration",
        });
        let result_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", result_str);
    } else {
        println!("{}", "Stem Export".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Output dir:", output_dir.display());
        println!("{:20} {}", "Format:", format);
        println!("{:20} {}-bit", "Bit depth:", bit_depth);
        if let Some(sr) = sample_rate {
            println!("{:20} {} Hz", "Sample rate:", sr);
        }
        println!();

        println!("{}", "Stems".cyan().bold());
        println!("{}", "-".repeat(60));
        for (i, stem) in stems.iter().enumerate() {
            println!("  [{:>2}] {}", i + 1, stem);
        }
        println!();

        println!(
            "{}",
            "Note: Stem export pending audio decoding pipeline integration.".yellow()
        );
    }

    Ok(())
}

/// Handle delivery spec checking.
async fn handle_delivery(
    input: &PathBuf,
    spec: &str,
    fix: bool,
    output: Option<&std::path::Path>,
    output_format: &str,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    if fix && output.is_none() {
        return Err(anyhow::anyhow!("--output is required when using --fix"));
    }

    let target = parse_delivery_target(spec)?;
    let delivery_spec =
        oximedia_audiopost::delivery_spec::AudioDeliverySpec::from_target(target, 2, 48000);

    let file_size = std::fs::metadata(input)
        .context("Failed to read file metadata")?
        .len();

    match output_format {
        "json" => {
            let result = serde_json::json!({
                "input": input.display().to_string(),
                "spec": spec,
                "target_loudness_lkfs": delivery_spec.max_loudness_lkfs,
                "max_true_peak_dbtp": delivery_spec.max_true_peak_dbtp,
                "channels": delivery_spec.channels,
                "sample_rate_hz": delivery_spec.sample_rate_hz,
                "bit_depth": delivery_spec.bit_depth,
                "file_size": file_size,
                "fix": fix,
                "status": "analysis_pending",
                "message": "Delivery spec defined; audio analysis pending pipeline integration",
            });
            let result_str =
                serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
            println!("{}", result_str);
        }
        _ => {
            println!("{}", "Delivery Specification Check".green().bold());
            println!("{}", "=".repeat(60));
            println!("{:25} {}", "Input:", input.display());
            println!("{:25} {} bytes", "File size:", file_size);
            println!("{:25} {}", "Spec:", spec);
            println!();

            println!("{}", "Requirements".cyan().bold());
            println!("{}", "-".repeat(60));
            println!(
                "{:25} {} LKFS",
                "Max loudness:", delivery_spec.max_loudness_lkfs
            );
            println!(
                "{:25} {} dBTP",
                "Max true peak:", delivery_spec.max_true_peak_dbtp
            );
            println!("{:25} {}", "Channels:", delivery_spec.channels);
            println!("{:25} {} Hz", "Sample rate:", delivery_spec.sample_rate_hz);
            println!("{:25} {}-bit", "Bit depth:", delivery_spec.bit_depth);

            if fix {
                println!();
                println!(
                    "{:25} {}",
                    "Fix mode:",
                    "enabled (will attempt to fix non-compliant audio)".yellow()
                );
                if let Some(out_path) = output {
                    println!("{:25} {}", "Fixed output:", out_path.display());
                }
            }

            println!();
            println!(
                "{}",
                "Note: Audio analysis pending pipeline integration.".yellow()
            );
        }
    }

    Ok(())
}

/// Handle audio restoration.
async fn handle_restore(
    input: &PathBuf,
    output: &PathBuf,
    declip: bool,
    dehum: bool,
    decrackle: bool,
    denoise: bool,
    strength: f64,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }
    if !(0.0..=1.0).contains(&strength) {
        return Err(anyhow::anyhow!(
            "Strength must be between 0.0 and 1.0, got {}",
            strength
        ));
    }

    if !declip && !dehum && !decrackle && !denoise {
        return Err(anyhow::anyhow!(
            "At least one restoration step must be enabled (--declip, --dehum, --decrackle, --denoise, or --all)"
        ));
    }

    let steps: Vec<&str> = [
        if declip { Some("declip") } else { None },
        if dehum { Some("dehum") } else { None },
        if decrackle { Some("decrackle") } else { None },
        if denoise { Some("denoise") } else { None },
    ]
    .iter()
    .filter_map(|s| *s)
    .collect();

    if json_output {
        let result = serde_json::json!({
            "status": "restore_pending",
            "input": input.display().to_string(),
            "output": output.display().to_string(),
            "steps": steps,
            "strength": strength,
            "message": "Restoration pipeline configured; full processing pending audio decode integration",
        });
        let result_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", result_str);
    } else {
        println!("{}", "Audio Restoration".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {:.2}", "Strength:", strength);
        println!();

        println!("{}", "Restoration Steps".cyan().bold());
        println!("{}", "-".repeat(60));
        println!(
            "  Declip:     {}",
            if declip {
                "enabled".green().to_string()
            } else {
                "disabled".dimmed().to_string()
            }
        );
        println!(
            "  Dehum:      {}",
            if dehum {
                "enabled".green().to_string()
            } else {
                "disabled".dimmed().to_string()
            }
        );
        println!(
            "  Decrackle:  {}",
            if decrackle {
                "enabled".green().to_string()
            } else {
                "disabled".dimmed().to_string()
            }
        );
        println!(
            "  Denoise:    {}",
            if denoise {
                "enabled".green().to_string()
            } else {
                "disabled".dimmed().to_string()
            }
        );
        println!();

        println!(
            "{}",
            "Note: Audio restoration pending audio pipeline integration.".yellow()
        );
        println!(
            "{}",
            "Spectral noise reducer and restoration modules are ready.".dimmed()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_delivery_target_broadcast() {
        let target = parse_delivery_target("broadcast");
        assert!(target.is_ok());
    }

    #[test]
    fn test_parse_delivery_target_all() {
        for spec in &["broadcast", "cinema", "streaming", "podcast"] {
            assert!(parse_delivery_target(spec).is_ok());
        }
    }

    #[test]
    fn test_parse_delivery_target_invalid() {
        assert!(parse_delivery_target("unknown").is_err());
    }

    #[test]
    fn test_restore_validates_strength() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime");
        let tmp_in = std::env::temp_dir().join("test_restore_input.wav");
        let tmp_out = std::env::temp_dir().join("test_restore_output.wav");
        let result = rt.block_on(handle_restore(
            &tmp_in, &tmp_out, true, false, false, false, 2.0, false,
        ));
        assert!(result.is_err());
    }

    #[test]
    fn test_restore_requires_at_least_one_step() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime");
        let tmp_in = std::env::temp_dir().join("test_restore_no_steps.wav");
        let tmp_out = std::env::temp_dir().join("test_restore_no_steps_out.wav");
        // Create dummy input file
        std::fs::write(&tmp_in, b"dummy").ok();
        let result = rt.block_on(handle_restore(
            &tmp_in, &tmp_out, false, false, false, false, 0.5, false,
        ));
        assert!(result.is_err());
        std::fs::remove_file(&tmp_in).ok();
    }
}
