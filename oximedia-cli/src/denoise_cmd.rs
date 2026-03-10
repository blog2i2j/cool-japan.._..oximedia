//! Video denoising command.
//!
//! Provides the `oximedia denoise` subcommand for noise reduction using the
//! `oximedia-denoise` crate's `Denoiser` and `DenoiseConfig`.

use anyhow::{Context, Result};
use colored::Colorize;
use std::path::PathBuf;

/// Options for the `denoise` command.
pub struct DenoiseOptions {
    pub input: PathBuf,
    pub output: PathBuf,
    pub mode: String,
    pub strength: f32,
    pub spatial: bool,
    pub temporal: bool,
    pub preserve_grain: bool,
}

/// Entry point called from `main.rs`.
pub async fn run_denoise(opts: DenoiseOptions, json_output: bool) -> Result<()> {
    use oximedia_denoise::Denoiser;

    let mode = parse_mode(&opts.mode)?;

    let config = build_config(mode, opts.strength, opts.preserve_grain)?;

    let _denoiser = Denoiser::new(config.clone());

    if json_output {
        let obj = serde_json::json!({
            "operation": "denoise",
            "input": opts.input.to_string_lossy(),
            "output": opts.output.to_string_lossy(),
            "mode": opts.mode,
            "strength": opts.strength,
            "spatial": opts.spatial,
            "temporal": opts.temporal,
            "preserve_grain": opts.preserve_grain,
            "temporal_window": config.temporal_window,
            "preserve_edges": config.preserve_edges,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "Video Denoise".green().bold());
    println!("  Input:          {}", opts.input.display());
    println!("  Output:         {}", opts.output.display());
    println!("  Mode:           {}", mode_label(&opts.mode).cyan());
    println!("  Strength:       {:.2}", opts.strength);
    println!(
        "  Spatial:        {}",
        if opts.spatial { "yes" } else { "no" }
    );
    println!(
        "  Temporal:       {}",
        if opts.temporal { "yes" } else { "no" }
    );
    println!(
        "  Preserve grain: {}",
        if opts.preserve_grain { "yes" } else { "no" }
    );
    println!("  Temporal win:   {} frames", config.temporal_window);
    println!("  Preserve edges: {}", config.preserve_edges);

    println!(
        "\n{} Denoiser configured. Full frame-pipeline processing requires I/O integration.",
        "Note:".yellow()
    );
    println!("{} {}", "Would write:".dimmed(), opts.output.display());

    Ok(())
}

/// Map CLI mode string to `DenoiseMode`.
fn parse_mode(mode: &str) -> Result<oximedia_denoise::DenoiseMode> {
    use oximedia_denoise::DenoiseMode;
    match mode.to_lowercase().replace('-', "_").as_str() {
        "fast" => Ok(DenoiseMode::Fast),
        "balanced" => Ok(DenoiseMode::Balanced),
        "quality" => Ok(DenoiseMode::Quality),
        "grain_aware" | "grain-aware" => Ok(DenoiseMode::GrainAware),
        other => anyhow::bail!(
            "Unknown denoise mode '{}'. Use: fast, balanced, quality, grain-aware",
            other
        ),
    }
}

/// Human-readable mode label.
fn mode_label(mode: &str) -> &'static str {
    match mode.to_lowercase().as_str() {
        "fast" => "Fast",
        "balanced" => "Balanced",
        "quality" => "Quality",
        "grain-aware" | "grain_aware" => "Grain-Aware",
        _ => "Unknown",
    }
}

/// Build a `DenoiseConfig` from the provided options.
fn build_config(
    mode: oximedia_denoise::DenoiseMode,
    strength: f32,
    preserve_grain: bool,
) -> Result<oximedia_denoise::DenoiseConfig> {
    use oximedia_denoise::DenoiseConfig;

    let base = match mode {
        oximedia_denoise::DenoiseMode::Fast => DenoiseConfig::light(),
        oximedia_denoise::DenoiseMode::Quality => DenoiseConfig::strong(),
        _ => DenoiseConfig::medium(),
    };

    let config = DenoiseConfig {
        mode,
        strength: strength.clamp(0.0, 1.0),
        preserve_grain,
        ..base
    };

    config
        .validate()
        .with_context(|| "Invalid denoise configuration")?;

    Ok(config)
}
