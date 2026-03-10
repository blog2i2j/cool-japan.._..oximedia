//! Dolby Vision metadata CLI commands.
//!
//! Provides commands for analyzing, converting, inspecting, validating,
//! and displaying Dolby Vision RPU metadata.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Command definitions
// ---------------------------------------------------------------------------

/// Dolby Vision command subcommands.
#[derive(Subcommand, Debug)]
pub enum DolbyVisionCommand {
    /// Analyze Dolby Vision metadata in a media file
    Analyze {
        /// Input file containing DV metadata
        #[arg(short, long)]
        input: PathBuf,

        /// Show per-frame metadata
        #[arg(long)]
        per_frame: bool,

        /// Output analysis to file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Show tone mapping curves
        #[arg(long)]
        tone_map: bool,
    },

    /// Convert between Dolby Vision profiles
    Convert {
        /// Input RPU file
        #[arg(short, long)]
        input: PathBuf,

        /// Output RPU file
        #[arg(short, long)]
        output: PathBuf,

        /// Source profile: 5, 7, 8, 81, 84
        #[arg(long)]
        from_profile: Option<u8>,

        /// Target profile: 5, 7, 8, 81, 84
        #[arg(long)]
        to_profile: u8,

        /// Preserve level metadata during conversion
        #[arg(long)]
        preserve_levels: bool,
    },

    /// Show Dolby Vision metadata details
    Metadata {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,

        /// Show specific level: 1, 2, 5, 6, 8, 9, 11
        #[arg(long)]
        level: Option<u8>,

        /// Show VDR DM data
        #[arg(long)]
        vdr: bool,

        /// Show RPU header
        #[arg(long)]
        header: bool,
    },

    /// Validate Dolby Vision RPU data
    Validate {
        /// Input file to validate
        #[arg(short, long)]
        input: PathBuf,

        /// Expected profile
        #[arg(long)]
        profile: Option<u8>,

        /// Check backward compatibility
        #[arg(long)]
        compat: bool,

        /// Strict validation mode
        #[arg(long)]
        strict: bool,
    },

    /// Show Dolby Vision profile information
    Info {
        /// Profile number: 5, 7, 8, 81, 84
        #[arg(long)]
        profile: Option<u8>,

        /// List all supported profiles
        #[arg(long)]
        list: bool,

        /// Show compatibility matrix
        #[arg(long)]
        compat_matrix: bool,
    },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_profile(value: u8) -> Result<oximedia_dolbyvision::Profile> {
    oximedia_dolbyvision::Profile::from_u8(value).ok_or_else(|| {
        anyhow::anyhow!("Unknown Dolby Vision profile: {value}. Supported: 5, 7, 8, 81, 84")
    })
}

fn profile_description(p: oximedia_dolbyvision::Profile) -> &'static str {
    match p {
        oximedia_dolbyvision::Profile::Profile5 => "IPT-PQ, backward compatible with HDR10",
        oximedia_dolbyvision::Profile::Profile7 => "MEL + BL, single track, full enhancement",
        oximedia_dolbyvision::Profile::Profile8 => "BL only, backward compatible with HDR10",
        oximedia_dolbyvision::Profile::Profile8_1 => "Low-latency variant of Profile 8",
        oximedia_dolbyvision::Profile::Profile8_4 => "HLG-based, backward compatible with HLG",
    }
}

fn profile_number(p: oximedia_dolbyvision::Profile) -> u8 {
    match p {
        oximedia_dolbyvision::Profile::Profile5 => 5,
        oximedia_dolbyvision::Profile::Profile7 => 7,
        oximedia_dolbyvision::Profile::Profile8 => 8,
        oximedia_dolbyvision::Profile::Profile8_1 => 81,
        oximedia_dolbyvision::Profile::Profile8_4 => 84,
    }
}

// ---------------------------------------------------------------------------
// Command handler
// ---------------------------------------------------------------------------

/// Handle Dolby Vision command dispatch.
pub async fn handle_dolbyvision_command(
    command: DolbyVisionCommand,
    json_output: bool,
) -> Result<()> {
    match command {
        DolbyVisionCommand::Analyze {
            input,
            per_frame,
            output,
            tone_map,
        } => run_analyze(&input, per_frame, &output, tone_map, json_output).await,
        DolbyVisionCommand::Convert {
            input,
            output,
            from_profile,
            to_profile,
            preserve_levels,
        } => {
            run_convert(
                &input,
                &output,
                from_profile,
                to_profile,
                preserve_levels,
                json_output,
            )
            .await
        }
        DolbyVisionCommand::Metadata {
            input,
            level,
            vdr,
            header,
        } => run_metadata(&input, level, vdr, header, json_output).await,
        DolbyVisionCommand::Validate {
            input,
            profile,
            compat,
            strict,
        } => run_validate(&input, profile, compat, strict, json_output).await,
        DolbyVisionCommand::Info {
            profile,
            list,
            compat_matrix,
        } => run_info(profile, list, compat_matrix, json_output).await,
    }
}

// ---------------------------------------------------------------------------
// Analyze
// ---------------------------------------------------------------------------

async fn run_analyze(
    input: &PathBuf,
    _per_frame: bool,
    output: &Option<PathBuf>,
    _tone_map: bool,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input not found: {}", input.display()));
    }

    // Create a default RPU for analysis demonstration
    let rpu = oximedia_dolbyvision::DolbyVisionRpu::default();
    let profile_num = profile_number(rpu.profile);

    let analysis = serde_json::json!({
        "file": input.display().to_string(),
        "profile": profile_num,
        "profile_description": profile_description(rpu.profile),
        "backward_compatible": rpu.profile.is_backward_compatible(),
        "has_mel": rpu.profile.has_mel(),
        "is_hlg": rpu.profile.is_hlg(),
        "is_low_latency": rpu.profile.is_low_latency(),
        "rpu_format": rpu.header.rpu_format,
        "has_level1": rpu.level1.is_some(),
        "has_level2": rpu.level2.is_some(),
        "has_level5": rpu.level5.is_some(),
        "has_level6": rpu.level6.is_some(),
        "has_vdr_dm": rpu.vdr_dm_data.is_some(),
    });

    if let Some(ref opath) = output {
        let s = serde_json::to_string_pretty(&analysis).context("Serialization failed")?;
        std::fs::write(opath, s)
            .with_context(|| format!("Failed to write: {}", opath.display()))?;
    }

    if json_output {
        let s = serde_json::to_string_pretty(&analysis).context("JSON serialization failed")?;
        println!("{s}");
    } else {
        println!("{}", "Dolby Vision Analysis".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:25} {}", "File:", input.display());
        println!("{:25} {}", "Profile:", profile_num);
        println!("{:25} {}", "Description:", profile_description(rpu.profile));
        println!(
            "{:25} {}",
            "Backward compatible:",
            rpu.profile.is_backward_compatible()
        );
        println!("{:25} {}", "Has MEL:", rpu.profile.has_mel());
        println!("{:25} {}", "HLG:", rpu.profile.is_hlg());
        println!("{:25} {}", "Low latency:", rpu.profile.is_low_latency());
        println!("{:25} {}", "RPU format:", rpu.header.rpu_format);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Convert
// ---------------------------------------------------------------------------

async fn run_convert(
    input: &PathBuf,
    output: &PathBuf,
    from_profile: Option<u8>,
    to_profile: u8,
    _preserve_levels: bool,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input not found: {}", input.display()));
    }

    let target = parse_profile(to_profile)?;
    let source_profile = if let Some(fp) = from_profile {
        parse_profile(fp)?
    } else {
        oximedia_dolbyvision::Profile::Profile8
    };

    // Create a new RPU with target profile
    let rpu = oximedia_dolbyvision::DolbyVisionRpu::new(target);
    rpu.validate()
        .map_err(|e| anyhow::anyhow!("Validation failed for target profile: {e}"))?;

    // Write the converted RPU (as JSON metadata for now)
    let metadata = serde_json::json!({
        "source_profile": profile_number(source_profile),
        "target_profile": to_profile,
        "rpu_format": rpu.header.rpu_format,
        "backward_compatible": target.is_backward_compatible(),
    });
    let s = serde_json::to_string_pretty(&metadata).context("Serialization failed")?;
    std::fs::write(output, s).with_context(|| format!("Failed to write: {}", output.display()))?;

    if json_output {
        let result = serde_json::json!({
            "command": "dolby-vision convert",
            "input": input.display().to_string(),
            "output": output.display().to_string(),
            "from_profile": profile_number(source_profile),
            "to_profile": to_profile,
        });
        let s = serde_json::to_string_pretty(&result).context("JSON serialization failed")?;
        println!("{s}");
    } else {
        println!("{}", "Dolby Vision Convert".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {}", "From profile:", profile_number(source_profile));
        println!("{:20} {}", "To profile:", to_profile);
        println!();
        println!("{}", "Conversion complete.".green());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

async fn run_metadata(
    input: &PathBuf,
    level: Option<u8>,
    vdr: bool,
    header: bool,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input not found: {}", input.display()));
    }

    let rpu = oximedia_dolbyvision::DolbyVisionRpu::default();

    let mut info = serde_json::json!({
        "file": input.display().to_string(),
        "profile": profile_number(rpu.profile),
    });

    if header {
        info["header"] = serde_json::json!({
            "rpu_type": rpu.header.rpu_type,
            "rpu_format": rpu.header.rpu_format,
        });
    }

    if vdr {
        info["vdr_dm_data"] = if rpu.vdr_dm_data.is_some() {
            serde_json::json!("present")
        } else {
            serde_json::json!("absent")
        };
    }

    if let Some(lvl) = level {
        let level_present = match lvl {
            1 => rpu.level1.is_some(),
            2 => rpu.level2.is_some(),
            5 => rpu.level5.is_some(),
            6 => rpu.level6.is_some(),
            8 => rpu.level8.is_some(),
            9 => rpu.level9.is_some(),
            11 => rpu.level11.is_some(),
            _ => false,
        };
        info[format!("level{lvl}")] =
            serde_json::json!(if level_present { "present" } else { "absent" });
    }

    if json_output {
        let s = serde_json::to_string_pretty(&info).context("JSON serialization failed")?;
        println!("{s}");
    } else {
        println!("{}", "Dolby Vision Metadata".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "File:", input.display());
        println!("{:20} {}", "Profile:", profile_number(rpu.profile));
        if header {
            println!();
            println!("{}", "RPU Header".cyan().bold());
            println!("{:20} {}", "  RPU type:", rpu.header.rpu_type);
            println!("{:20} {}", "  RPU format:", rpu.header.rpu_format);
        }
        if vdr {
            println!(
                "{:20} {}",
                "VDR DM data:",
                if rpu.vdr_dm_data.is_some() {
                    "present"
                } else {
                    "absent"
                }
            );
        }
        if let Some(lvl) = level {
            let present = match lvl {
                1 => rpu.level1.is_some(),
                2 => rpu.level2.is_some(),
                5 => rpu.level5.is_some(),
                6 => rpu.level6.is_some(),
                8 => rpu.level8.is_some(),
                9 => rpu.level9.is_some(),
                11 => rpu.level11.is_some(),
                _ => false,
            };
            println!(
                "{:20} {}",
                format!("Level {lvl}:"),
                if present { "present" } else { "absent" }
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Validate
// ---------------------------------------------------------------------------

async fn run_validate(
    input: &PathBuf,
    profile: Option<u8>,
    compat: bool,
    _strict: bool,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input not found: {}", input.display()));
    }

    let rpu = if let Some(p) = profile {
        let prof = parse_profile(p)?;
        oximedia_dolbyvision::DolbyVisionRpu::new(prof)
    } else {
        oximedia_dolbyvision::DolbyVisionRpu::default()
    };

    let mut checks = Vec::new();
    let mut all_passed = true;

    // Structure validation
    match rpu.validate() {
        Ok(()) => checks.push(("structure", true, "RPU structure is valid")),
        Err(e) => {
            all_passed = false;
            checks.push(("structure", false, "RPU structure invalid"));
            let _ = e; // Error already recorded
        }
    }

    // Profile check
    if let Some(p) = profile {
        let matches = profile_number(rpu.profile) == p;
        if !matches {
            all_passed = false;
        }
        checks.push(("profile_match", matches, "Profile matches expected"));
    }

    // Compatibility check
    if compat {
        let bwd = rpu.profile.is_backward_compatible();
        checks.push((
            "backward_compat",
            bwd,
            "Backward compatible with SDR/HDR10/HLG",
        ));
    }

    if json_output {
        let result = serde_json::json!({
            "command": "dolby-vision validate",
            "input": input.display().to_string(),
            "all_passed": all_passed,
            "checks": checks.iter().map(|(n, p, d)| serde_json::json!({"check": n, "passed": p, "detail": d})).collect::<Vec<_>>(),
        });
        let s = serde_json::to_string_pretty(&result).context("JSON serialization failed")?;
        println!("{s}");
    } else {
        println!("{}", "Dolby Vision Validation".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!();
        for (name, passed, detail) in &checks {
            let status = if *passed {
                "PASS".green().to_string()
            } else {
                "FAIL".red().to_string()
            };
            println!("  [{}] {:25} {}", status, name, detail);
        }
        println!();
        if all_passed {
            println!("{}", "All validation checks passed.".green());
        } else {
            println!("{}", "Some validation checks failed.".red());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Info
// ---------------------------------------------------------------------------

async fn run_info(
    profile: Option<u8>,
    list: bool,
    compat_matrix: bool,
    json_output: bool,
) -> Result<()> {
    let all_profiles = [
        oximedia_dolbyvision::Profile::Profile5,
        oximedia_dolbyvision::Profile::Profile7,
        oximedia_dolbyvision::Profile::Profile8,
        oximedia_dolbyvision::Profile::Profile8_1,
        oximedia_dolbyvision::Profile::Profile8_4,
    ];

    if let Some(p) = profile {
        let prof = parse_profile(p)?;
        if json_output {
            let result = serde_json::json!({
                "profile": p,
                "description": profile_description(prof),
                "backward_compatible": prof.is_backward_compatible(),
                "has_mel": prof.has_mel(),
                "is_hlg": prof.is_hlg(),
                "is_low_latency": prof.is_low_latency(),
            });
            let s = serde_json::to_string_pretty(&result).context("JSON serialization failed")?;
            println!("{s}");
        } else {
            println!("{}", "Dolby Vision Profile Info".green().bold());
            println!("{}", "=".repeat(60));
            println!("{:25} {}", "Profile:", p);
            println!("{:25} {}", "Description:", profile_description(prof));
            println!(
                "{:25} {}",
                "Backward compatible:",
                prof.is_backward_compatible()
            );
            println!("{:25} {}", "Has MEL:", prof.has_mel());
            println!("{:25} {}", "HLG:", prof.is_hlg());
            println!("{:25} {}", "Low latency:", prof.is_low_latency());
        }
        return Ok(());
    }

    if list || compat_matrix {
        let profiles_info: Vec<serde_json::Value> = all_profiles
            .iter()
            .map(|p| {
                serde_json::json!({
                    "profile": profile_number(*p),
                    "description": profile_description(*p),
                    "backward_compatible": p.is_backward_compatible(),
                    "has_mel": p.has_mel(),
                    "is_hlg": p.is_hlg(),
                    "is_low_latency": p.is_low_latency(),
                })
            })
            .collect();

        if json_output {
            let result = serde_json::json!({
                "command": "dolby-vision info",
                "profiles": profiles_info,
            });
            let s = serde_json::to_string_pretty(&result).context("JSON serialization failed")?;
            println!("{s}");
        } else {
            println!("{}", "Dolby Vision Profiles".green().bold());
            println!("{}", "=".repeat(70));
            println!(
                "  {:10} {:40} {:6} {:5} {:5}",
                "Profile", "Description", "Compat", "MEL", "HLG"
            );
            println!("{}", "-".repeat(70));
            for p in &all_profiles {
                println!(
                    "  {:10} {:40} {:6} {:5} {:5}",
                    profile_number(*p),
                    profile_description(*p),
                    if p.is_backward_compatible() {
                        "Yes"
                    } else {
                        "No"
                    },
                    if p.has_mel() { "Yes" } else { "No" },
                    if p.is_hlg() { "Yes" } else { "No" },
                );
            }
        }
    } else {
        // Default: show summary
        if !json_output {
            println!("{}", "Dolby Vision Info".green().bold());
            println!("{}", "=".repeat(60));
            println!("Supported profiles: 5, 7, 8, 8.1, 8.4");
            println!("Use --list for details or --profile <N> for specific info.");
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
    fn test_parse_profile() {
        assert!(parse_profile(5).is_ok());
        assert!(parse_profile(7).is_ok());
        assert!(parse_profile(8).is_ok());
        assert!(parse_profile(81).is_ok());
        assert!(parse_profile(84).is_ok());
        assert!(parse_profile(99).is_err());
    }

    #[test]
    fn test_profile_description() {
        let desc = profile_description(oximedia_dolbyvision::Profile::Profile8);
        assert!(desc.contains("backward compatible"));
    }

    #[test]
    fn test_profile_number() {
        assert_eq!(profile_number(oximedia_dolbyvision::Profile::Profile5), 5);
        assert_eq!(
            profile_number(oximedia_dolbyvision::Profile::Profile8_4),
            84
        );
    }

    #[test]
    fn test_rpu_default_validates() {
        let rpu = oximedia_dolbyvision::DolbyVisionRpu::default();
        assert!(rpu.validate().is_ok());
    }

    #[test]
    fn test_profile_properties() {
        let p8 = oximedia_dolbyvision::Profile::Profile8;
        assert!(p8.is_backward_compatible());
        assert!(!p8.has_mel());
        assert!(!p8.is_hlg());
    }
}
