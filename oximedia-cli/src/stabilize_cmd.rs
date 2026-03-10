//! Video stabilisation command.
//!
//! Provides `oximedia stabilize` using `oximedia-stabilize` to remove unwanted
//! camera shake via configurable motion models and quality presets.

use anyhow::{Context, Result};
use colored::Colorize;
use std::path::PathBuf;

/// Options for the `stabilize` command.
pub struct StabilizeOptions {
    pub input: PathBuf,
    pub output: PathBuf,
    pub mode: String,
    pub quality: String,
    pub smoothing: u32,
    pub zoom: bool,
}

/// Entry point called from `main.rs`.
pub async fn run_stabilize(opts: StabilizeOptions, json_output: bool) -> Result<()> {
    use oximedia_stabilize::{StabilizeConfig, Stabilizer};

    let stab_mode = parse_mode(&opts.mode)?;
    let quality = parse_quality(&opts.quality)?;

    // Smoothing strength as normalised 0.0-1.0 from the frame-count window
    let smoothing_strength = (opts.smoothing as f64 / 100.0).clamp(0.01, 1.0);

    let config = StabilizeConfig::new()
        .with_mode(stab_mode)
        .with_quality(quality)
        .with_smoothing_strength(smoothing_strength)
        .with_zoom_optimization(opts.zoom);

    config
        .validate()
        .with_context(|| "Invalid stabilisation configuration")?;

    let _stabilizer =
        Stabilizer::new(config.clone()).with_context(|| "Failed to initialise Stabilizer")?;

    if json_output {
        let obj = serde_json::json!({
            "operation": "stabilize",
            "input": opts.input.to_string_lossy(),
            "output": opts.output.to_string_lossy(),
            "mode": opts.mode,
            "quality": opts.quality,
            "smoothing_frames": opts.smoothing,
            "smoothing_strength": smoothing_strength,
            "zoom": opts.zoom,
            "enable_multipass": config.enable_multipass,
            "feature_count": config.feature_count,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "Video Stabilise".green().bold());
    println!("  Input:            {}", opts.input.display());
    println!("  Output:           {}", opts.output.display());
    println!("  Mode:             {}", mode_label(&opts.mode).cyan());
    println!(
        "  Quality:          {}",
        quality_label(&opts.quality).cyan()
    );
    println!("  Smoothing window: {} frames", opts.smoothing);
    println!("  Smoothing str.:   {:.3}", smoothing_strength);
    println!(
        "  Auto-zoom:        {}",
        if opts.zoom { "yes" } else { "no" }
    );
    println!("  Multi-pass:       {}", config.enable_multipass);
    println!("  Feature count:    {}", config.feature_count);

    println!(
        "\n{} Stabiliser configured. Full frame pipeline requires I/O integration.",
        "Note:".yellow()
    );
    println!("{} {}", "Would write:".dimmed(), opts.output.display());

    Ok(())
}

/// Map CLI mode string to `StabilizationMode`.
fn parse_mode(mode: &str) -> Result<oximedia_stabilize::StabilizationMode> {
    use oximedia_stabilize::StabilizationMode;
    match mode.to_lowercase().as_str() {
        "translation" => Ok(StabilizationMode::Translation),
        "affine" => Ok(StabilizationMode::Affine),
        "perspective" => Ok(StabilizationMode::Perspective),
        "3d" | "threed" => Ok(StabilizationMode::ThreeD),
        other => anyhow::bail!(
            "Unknown stabilisation mode '{}'. Use: translation, affine, perspective, 3d",
            other
        ),
    }
}

/// Map CLI quality string to `QualityPreset`.
fn parse_quality(quality: &str) -> Result<oximedia_stabilize::QualityPreset> {
    use oximedia_stabilize::QualityPreset;
    match quality.to_lowercase().as_str() {
        "fast" => Ok(QualityPreset::Fast),
        "balanced" => Ok(QualityPreset::Balanced),
        "maximum" | "max" => Ok(QualityPreset::Maximum),
        other => anyhow::bail!(
            "Unknown quality preset '{}'. Use: fast, balanced, maximum",
            other
        ),
    }
}

fn mode_label(mode: &str) -> &'static str {
    match mode.to_lowercase().as_str() {
        "translation" => "Translation",
        "affine" => "Affine",
        "perspective" => "Perspective",
        "3d" => "3D",
        _ => "Unknown",
    }
}

fn quality_label(quality: &str) -> &'static str {
    match quality.to_lowercase().as_str() {
        "fast" => "Fast",
        "balanced" => "Balanced",
        "maximum" | "max" => "Maximum",
        _ => "Unknown",
    }
}
