//! Video scopes command.
//!
//! Provides the `oximedia scopes` subcommand family for generating broadcast-quality
//! video scopes (waveform, vectorscope, histogram, parade, false color) using the
//! `oximedia-scopes` crate.

use anyhow::{bail, Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Subcommands for the `scopes` command.
#[derive(Subcommand, Debug)]
pub enum ScopesCommand {
    /// Generate waveform display
    Waveform {
        /// Input video file
        #[arg(short, long)]
        input: PathBuf,

        /// Output scope image path
        #[arg(short, long)]
        output: PathBuf,

        /// Waveform mode: luma, rgb_parade, rgb_overlay, ycbcr
        #[arg(long, default_value = "luma")]
        mode: String,

        /// Scope display width in pixels
        #[arg(long, default_value = "512")]
        width: u32,

        /// Scope display height in pixels
        #[arg(long, default_value = "512")]
        height: u32,

        /// Specific frame number (default: first frame)
        #[arg(long)]
        frame: Option<u64>,

        /// Show graticule overlay
        #[arg(long)]
        graticule: bool,
    },

    /// Generate vectorscope display
    Vectorscope {
        /// Input video file
        #[arg(short, long)]
        input: PathBuf,

        /// Output scope image path
        #[arg(short, long)]
        output: PathBuf,

        /// Display mode: circular, rectangular
        #[arg(long, default_value = "circular")]
        mode: String,

        /// Scope display size in pixels (square)
        #[arg(long, default_value = "512")]
        size: u32,

        /// Specific frame number (default: first frame)
        #[arg(long)]
        frame: Option<u64>,

        /// Show SMPTE color bar targets
        #[arg(long)]
        targets: bool,

        /// Vectorscope gain / zoom
        #[arg(long, default_value = "1.0")]
        gain: f64,
    },

    /// Generate histogram
    Histogram {
        /// Input video file
        #[arg(short, long)]
        input: PathBuf,

        /// Output scope image path
        #[arg(short, long)]
        output: PathBuf,

        /// Histogram mode: rgb, luma, overlay, stacked, logarithmic
        #[arg(long, default_value = "rgb")]
        mode: String,

        /// Scope display width in pixels
        #[arg(long, default_value = "512")]
        width: u32,

        /// Scope display height in pixels
        #[arg(long, default_value = "256")]
        height: u32,

        /// Specific frame number (default: first frame)
        #[arg(long)]
        frame: Option<u64>,
    },

    /// Generate RGB or YCbCr parade display
    Parade {
        /// Input video file
        #[arg(short, long)]
        input: PathBuf,

        /// Output scope image path
        #[arg(short, long)]
        output: PathBuf,

        /// Parade mode: rgb, ycbcr
        #[arg(long, default_value = "rgb")]
        mode: String,

        /// Scope display width in pixels
        #[arg(long, default_value = "768")]
        width: u32,

        /// Scope display height in pixels
        #[arg(long, default_value = "256")]
        height: u32,

        /// Specific frame number (default: first frame)
        #[arg(long)]
        frame: Option<u64>,
    },

    /// Generate false color exposure display
    FalseColor {
        /// Input video file
        #[arg(short, long)]
        input: PathBuf,

        /// Output scope image path
        #[arg(short, long)]
        output: PathBuf,

        /// Specific frame number (default: first frame)
        #[arg(long)]
        frame: Option<u64>,

        /// Show color scale legend alongside the image
        #[arg(long)]
        scale: bool,
    },
}

/// Entry point for `oximedia scopes <subcommand>`.
pub async fn handle_scopes_command(command: ScopesCommand, json_output: bool) -> Result<()> {
    match command {
        ScopesCommand::Waveform {
            input,
            output,
            mode,
            width,
            height,
            frame,
            graticule,
        } => {
            run_waveform(
                &input,
                &output,
                &mode,
                width,
                height,
                frame,
                graticule,
                json_output,
            )
            .await
        }
        ScopesCommand::Vectorscope {
            input,
            output,
            mode,
            size,
            frame,
            targets,
            gain,
        } => {
            run_vectorscope(
                &input,
                &output,
                &mode,
                size,
                frame,
                targets,
                gain,
                json_output,
            )
            .await
        }
        ScopesCommand::Histogram {
            input,
            output,
            mode,
            width,
            height,
            frame,
        } => run_histogram(&input, &output, &mode, width, height, frame, json_output).await,
        ScopesCommand::Parade {
            input,
            output,
            mode,
            width,
            height,
            frame,
        } => run_parade(&input, &output, &mode, width, height, frame, json_output).await,
        ScopesCommand::FalseColor {
            input,
            output,
            frame,
            scale,
        } => run_false_color(&input, &output, frame, scale, json_output).await,
    }
}

// ---------------------------------------------------------------------------
// Frame extraction helper
// ---------------------------------------------------------------------------

/// Extract a single frame as RGB24 data from a video file.
///
/// For now this creates a placeholder gradient frame when the full demux/decode
/// pipeline is not yet wired, while still exercising the scopes engine.  When
/// OxiMedia I/O integration is complete, this function should open the container,
/// seek to `_frame_num`, decode, and return raw RGB24 bytes plus dimensions.
fn extract_frame_rgb(input: &std::path::Path, _frame_num: u64) -> Result<(Vec<u8>, u32, u32)> {
    // Verify the input path exists so the user gets a clear error early.
    if !input.exists() {
        bail!("Input file not found: {}", input.display());
    }

    // Placeholder: create a 256x256 colour-ramp frame that gives interesting
    // scope output (diagonal gradient with varying R/G/B channels).
    let w: u32 = 256;
    let h: u32 = 256;
    let mut data = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            data[idx] = (x & 0xFF) as u8; // R
            data[idx + 1] = (y & 0xFF) as u8; // G
            data[idx + 2] = (((x + y) / 2) & 0xFF) as u8; // B
        }
    }

    Ok((data, w, h))
}

// ---------------------------------------------------------------------------
// Write scope RGBA data to output path (raw RGBA or described via JSON)
// ---------------------------------------------------------------------------

fn write_scope_output(
    output: &std::path::Path,
    scope: &oximedia_scopes::ScopeData,
    json_output: bool,
    scope_label: &str,
) -> Result<()> {
    if json_output {
        let obj = serde_json::json!({
            "scope": scope_label,
            "width": scope.width,
            "height": scope.height,
            "format": "RGBA",
            "bytes": scope.data.len(),
            "output": output.to_string_lossy(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&obj).context("JSON serialisation")?
        );
        return Ok(());
    }

    // Write raw RGBA file (consumer can load as width*height*4 RGBA bytes)
    std::fs::write(output, &scope.data)
        .with_context(|| format!("Failed to write scope image to {}", output.display()))?;

    println!("{}", format!("{scope_label} Scope").green().bold());
    println!("  Output:     {}", output.display());
    println!("  Dimensions: {}x{}", scope.width, scope.height);
    println!("  Format:     RGBA ({} bytes)", scope.data.len());

    Ok(())
}

// ---------------------------------------------------------------------------
// Waveform
// ---------------------------------------------------------------------------

async fn run_waveform(
    input: &std::path::Path,
    output: &std::path::Path,
    mode: &str,
    width: u32,
    height: u32,
    frame_num: Option<u64>,
    graticule: bool,
    json_output: bool,
) -> Result<()> {
    use oximedia_scopes::{
        HistogramMode, ScopeConfig, ScopeType, VectorscopeMode, VideoScopes, WaveformMode,
    };

    let scope_type = match mode.to_lowercase().as_str() {
        "luma" => ScopeType::WaveformLuma,
        "rgb_parade" | "rgb-parade" => ScopeType::WaveformRgbParade,
        "rgb_overlay" | "rgb-overlay" => ScopeType::WaveformRgbOverlay,
        "ycbcr" => ScopeType::WaveformYcbcr,
        other => bail!(
            "Unknown waveform mode '{}'. Use: luma, rgb_parade, rgb_overlay, ycbcr",
            other
        ),
    };

    let config = ScopeConfig {
        width,
        height,
        show_graticule: graticule,
        show_labels: graticule,
        anti_alias: true,
        waveform_mode: WaveformMode::Overlay,
        vectorscope_mode: VectorscopeMode::Circular,
        histogram_mode: HistogramMode::Overlay,
        vectorscope_gain: 1.0,
        highlight_gamut: false,
        gamut_colorspace: oximedia_scopes::GamutColorspace::Rec709,
    };

    let (frame_data, fw, fh) = extract_frame_rgb(input, frame_num.unwrap_or(0))?;
    let scopes = VideoScopes::new(config);
    let scope_data = scopes
        .analyze(&frame_data, fw, fh, scope_type)
        .map_err(|e| anyhow::anyhow!("Waveform analysis failed: {e}"))?;

    write_scope_output(output, &scope_data, json_output, "Waveform")
}

// ---------------------------------------------------------------------------
// Vectorscope
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn run_vectorscope(
    input: &std::path::Path,
    output: &std::path::Path,
    mode: &str,
    size: u32,
    frame_num: Option<u64>,
    targets: bool,
    gain: f64,
    json_output: bool,
) -> Result<()> {
    use oximedia_scopes::{
        HistogramMode, ScopeConfig, ScopeType, VectorscopeMode, VideoScopes, WaveformMode,
    };

    let vectorscope_mode = match mode.to_lowercase().as_str() {
        "circular" => VectorscopeMode::Circular,
        "rectangular" => VectorscopeMode::Rectangular,
        other => bail!(
            "Unknown vectorscope mode '{}'. Use: circular, rectangular",
            other
        ),
    };

    let config = ScopeConfig {
        width: size,
        height: size,
        show_graticule: targets,
        show_labels: targets,
        anti_alias: true,
        waveform_mode: WaveformMode::Overlay,
        vectorscope_mode,
        histogram_mode: HistogramMode::Overlay,
        vectorscope_gain: gain as f32,
        highlight_gamut: false,
        gamut_colorspace: oximedia_scopes::GamutColorspace::Rec709,
    };

    let (frame_data, fw, fh) = extract_frame_rgb(input, frame_num.unwrap_or(0))?;
    let scopes = VideoScopes::new(config);
    let scope_data = scopes
        .analyze(&frame_data, fw, fh, ScopeType::Vectorscope)
        .map_err(|e| anyhow::anyhow!("Vectorscope analysis failed: {e}"))?;

    write_scope_output(output, &scope_data, json_output, "Vectorscope")
}

// ---------------------------------------------------------------------------
// Histogram
// ---------------------------------------------------------------------------

async fn run_histogram(
    input: &std::path::Path,
    output: &std::path::Path,
    mode: &str,
    width: u32,
    height: u32,
    frame_num: Option<u64>,
    json_output: bool,
) -> Result<()> {
    use oximedia_scopes::{
        HistogramMode, ScopeConfig, ScopeType, VectorscopeMode, VideoScopes, WaveformMode,
    };

    let (scope_type, histogram_mode) = match mode.to_lowercase().as_str() {
        "rgb" => (ScopeType::HistogramRgb, HistogramMode::Overlay),
        "luma" => (ScopeType::HistogramLuma, HistogramMode::Overlay),
        "overlay" => (ScopeType::HistogramRgb, HistogramMode::Overlay),
        "stacked" => (ScopeType::HistogramRgb, HistogramMode::Stacked),
        "logarithmic" | "log" => (ScopeType::HistogramRgb, HistogramMode::Logarithmic),
        other => bail!(
            "Unknown histogram mode '{}'. Use: rgb, luma, overlay, stacked, logarithmic",
            other
        ),
    };

    let config = ScopeConfig {
        width,
        height,
        show_graticule: true,
        show_labels: true,
        anti_alias: true,
        waveform_mode: WaveformMode::Overlay,
        vectorscope_mode: VectorscopeMode::Circular,
        histogram_mode,
        vectorscope_gain: 1.0,
        highlight_gamut: false,
        gamut_colorspace: oximedia_scopes::GamutColorspace::Rec709,
    };

    let (frame_data, fw, fh) = extract_frame_rgb(input, frame_num.unwrap_or(0))?;
    let scopes = VideoScopes::new(config);
    let scope_data = scopes
        .analyze(&frame_data, fw, fh, scope_type)
        .map_err(|e| anyhow::anyhow!("Histogram analysis failed: {e}"))?;

    write_scope_output(output, &scope_data, json_output, "Histogram")
}

// ---------------------------------------------------------------------------
// Parade
// ---------------------------------------------------------------------------

async fn run_parade(
    input: &std::path::Path,
    output: &std::path::Path,
    mode: &str,
    width: u32,
    height: u32,
    frame_num: Option<u64>,
    json_output: bool,
) -> Result<()> {
    use oximedia_scopes::{
        HistogramMode, ScopeConfig, ScopeType, VectorscopeMode, VideoScopes, WaveformMode,
    };

    let scope_type = match mode.to_lowercase().as_str() {
        "rgb" => ScopeType::ParadeRgb,
        "ycbcr" => ScopeType::ParadeYcbcr,
        other => bail!("Unknown parade mode '{}'. Use: rgb, ycbcr", other),
    };

    let config = ScopeConfig {
        width,
        height,
        show_graticule: true,
        show_labels: true,
        anti_alias: true,
        waveform_mode: WaveformMode::Overlay,
        vectorscope_mode: VectorscopeMode::Circular,
        histogram_mode: HistogramMode::Overlay,
        vectorscope_gain: 1.0,
        highlight_gamut: false,
        gamut_colorspace: oximedia_scopes::GamutColorspace::Rec709,
    };

    let (frame_data, fw, fh) = extract_frame_rgb(input, frame_num.unwrap_or(0))?;
    let scopes = VideoScopes::new(config);
    let scope_data = scopes
        .analyze(&frame_data, fw, fh, scope_type)
        .map_err(|e| anyhow::anyhow!("Parade analysis failed: {e}"))?;

    write_scope_output(output, &scope_data, json_output, "Parade")
}

// ---------------------------------------------------------------------------
// False Color
// ---------------------------------------------------------------------------

async fn run_false_color(
    input: &std::path::Path,
    output: &std::path::Path,
    frame_num: Option<u64>,
    show_scale: bool,
    json_output: bool,
) -> Result<()> {
    use oximedia_scopes::{ScopeConfig, ScopeType, VideoScopes};

    let config = ScopeConfig::default();
    let (frame_data, fw, fh) = extract_frame_rgb(input, frame_num.unwrap_or(0))?;

    let scopes = VideoScopes::new(config);
    let scope_data = scopes
        .analyze(&frame_data, fw, fh, ScopeType::FalseColor)
        .map_err(|e| anyhow::anyhow!("False color analysis failed: {e}"))?;

    if json_output {
        let stats = oximedia_scopes::false_color::compute_false_color_stats(&frame_data, fw, fh);
        let zone_map: serde_json::Map<String, serde_json::Value> = stats
            .zone_distribution
            .iter()
            .map(|(name, pct)| {
                (
                    name.clone(),
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(f64::from(*pct))
                            .unwrap_or_else(|| serde_json::Number::from(0)),
                    ),
                )
            })
            .collect();

        let obj = serde_json::json!({
            "scope": "FalseColor",
            "width": scope_data.width,
            "height": scope_data.height,
            "format": "RGBA",
            "bytes": scope_data.data.len(),
            "output": output.to_string_lossy(),
            "show_scale": show_scale,
            "stats": {
                "highlight_clip_pct": stats.highlight_clip_percent,
                "shadow_clip_pct": stats.shadow_clip_percent,
                "good_exposure_pct": stats.good_exposure_percent,
                "zone_distribution": zone_map,
            },
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&obj).context("JSON serialisation")?
        );
        return Ok(());
    }

    // If scale legend requested, append it beneath the false color image
    let final_data = if show_scale {
        let legend_height = 20u32;
        let scale = oximedia_scopes::false_color::FalseColorScale::default();
        let legend =
            oximedia_scopes::false_color::generate_false_color_legend(fw, legend_height, &scale);
        let mut combined = scope_data.data.clone();
        combined.extend_from_slice(&legend);
        combined
    } else {
        scope_data.data.clone()
    };

    std::fs::write(output, &final_data)
        .with_context(|| format!("Failed to write false color image to {}", output.display()))?;

    let stats = oximedia_scopes::false_color::compute_false_color_stats(&frame_data, fw, fh);

    println!("{}", "False Color Scope".green().bold());
    println!("  Output:         {}", output.display());
    println!(
        "  Dimensions:     {}x{}",
        scope_data.width, scope_data.height
    );
    println!("  Format:         RGBA ({} bytes)", final_data.len());
    println!("  Good exposure:  {:.1}%", stats.good_exposure_percent);
    println!("  Shadow clip:    {:.1}%", stats.shadow_clip_percent);
    println!("  Highlight clip: {:.1}%", stats.highlight_clip_percent);
    if show_scale {
        println!("  Legend:          appended ({}px tall)", 20);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_input() -> PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_scopes_test_input.bin");
        if let Ok(mut f) = std::fs::File::create(&path) {
            let _ = f.write_all(b"dummy");
        }
        path
    }

    #[tokio::test]
    async fn test_waveform_luma() {
        let input = temp_input();
        let output = std::env::temp_dir().join("test_waveform.rgba");
        let result = run_waveform(&input, &output, "luma", 64, 64, None, false, false).await;
        assert!(result.is_ok());
        assert!(output.exists());
        let _ = std::fs::remove_file(&output);
    }

    #[tokio::test]
    async fn test_waveform_rgb_parade() {
        let input = temp_input();
        let output = std::env::temp_dir().join("test_wf_rgbparade.rgba");
        let result = run_waveform(&input, &output, "rgb_parade", 64, 64, None, true, false).await;
        assert!(result.is_ok());
        let _ = std::fs::remove_file(&output);
    }

    #[tokio::test]
    async fn test_vectorscope_circular() {
        let input = temp_input();
        let output = std::env::temp_dir().join("test_vectorscope.rgba");
        let result = run_vectorscope(&input, &output, "circular", 64, None, true, 1.0, false).await;
        assert!(result.is_ok());
        let _ = std::fs::remove_file(&output);
    }

    #[tokio::test]
    async fn test_histogram_rgb() {
        let input = temp_input();
        let output = std::env::temp_dir().join("test_histogram.rgba");
        let result = run_histogram(&input, &output, "rgb", 64, 64, None, false).await;
        assert!(result.is_ok());
        let _ = std::fs::remove_file(&output);
    }

    #[tokio::test]
    async fn test_parade_rgb() {
        let input = temp_input();
        let output = std::env::temp_dir().join("test_parade.rgba");
        let result = run_parade(&input, &output, "rgb", 96, 64, None, false).await;
        assert!(result.is_ok());
        let _ = std::fs::remove_file(&output);
    }

    #[tokio::test]
    async fn test_false_color() {
        let input = temp_input();
        let output = std::env::temp_dir().join("test_false_color.rgba");
        let result = run_false_color(&input, &output, None, false, false).await;
        assert!(result.is_ok());
        let _ = std::fs::remove_file(&output);
    }

    #[tokio::test]
    async fn test_false_color_with_scale() {
        let input = temp_input();
        let output = std::env::temp_dir().join("test_false_color_scale.rgba");
        let result = run_false_color(&input, &output, None, true, false).await;
        assert!(result.is_ok());
        let _ = std::fs::remove_file(&output);
    }

    #[tokio::test]
    async fn test_json_output() {
        let input = temp_input();
        let output = std::env::temp_dir().join("test_wf_json.rgba");
        let result = run_waveform(&input, &output, "luma", 64, 64, None, false, true).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_bad_waveform_mode() {
        let input = temp_input();
        let output = std::env::temp_dir().join("test_bad_wf.rgba");
        let result = run_waveform(&input, &output, "invalid", 64, 64, None, false, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_bad_vectorscope_mode() {
        let input = temp_input();
        let output = std::env::temp_dir().join("test_bad_vs.rgba");
        let result = run_vectorscope(&input, &output, "invalid", 64, None, false, 1.0, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_missing_input() {
        let output = std::env::temp_dir().join("test_missing.rgba");
        let result = run_waveform(
            std::path::Path::new("/nonexistent/video.mkv"),
            &output,
            "luma",
            64,
            64,
            None,
            false,
            false,
        )
        .await;
        assert!(result.is_err());
    }
}
