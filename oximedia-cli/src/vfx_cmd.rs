//! Visual effects and compositing CLI commands.
//!
//! Provides subcommands for applying effects, chroma keying, transitions,
//! and generating test patterns.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// VFX command subcommands.
#[derive(Subcommand, Debug)]
pub enum VfxCommand {
    /// Apply a visual effect to an image or video frame
    Apply {
        /// Input image/frame file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Effect name: blur, sharpen, glow, vintage, sepia, negative, posterize, pixelate
        #[arg(short, long)]
        effect: String,

        /// Effect strength (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        strength: f64,

        /// Extra parameters (key=value pairs, comma-separated)
        #[arg(long)]
        params: Option<String>,
    },

    /// List available effects
    List,

    /// Apply chroma keying (green/blue screen)
    Keying {
        /// Input image/frame file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Key color: "green", "blue", or hex "#RRGGBB"
        #[arg(long, default_value = "green")]
        key_color: String,

        /// Color tolerance (0.0-1.0)
        #[arg(long, default_value = "0.3")]
        tolerance: f64,

        /// Edge softness (0.0-1.0)
        #[arg(long, default_value = "0.1")]
        softness: f64,

        /// Background image to composite
        #[arg(long)]
        background: Option<PathBuf>,

        /// Enable spill suppression
        #[arg(long)]
        spill_suppress: bool,
    },

    /// Apply a transition between two images/frames
    Transition {
        /// First input image/frame
        #[arg(long)]
        input1: PathBuf,

        /// Second input image/frame
        #[arg(long)]
        input2: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Transition type: dissolve, wipe_left, wipe_right, wipe_up, wipe_down, fade
        #[arg(long, default_value = "dissolve")]
        transition_type: String,

        /// Transition progress (0.0-1.0, where 0.0 = input1, 1.0 = input2)
        #[arg(long, default_value = "0.5")]
        progress: f64,
    },

    /// Generate a test pattern image
    Generate {
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Pattern type: color_bars, gradient, noise, checkerboard, solid
        #[arg(short, long)]
        pattern: String,

        /// Image width in pixels
        #[arg(long, default_value = "1920")]
        width: u32,

        /// Image height in pixels
        #[arg(long, default_value = "1080")]
        height: u32,

        /// Primary color for patterns (hex "#RRGGBB")
        #[arg(long)]
        color1: Option<String>,

        /// Secondary color for patterns (hex "#RRGGBB")
        #[arg(long)]
        color2: Option<String>,

        /// Block size for checkerboard pattern
        #[arg(long, default_value = "64")]
        block_size: u32,
    },

    /// Apply color grading with blend modes
    Blend {
        /// Base image
        #[arg(long)]
        base: PathBuf,

        /// Overlay image
        #[arg(long)]
        overlay: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Blend mode: multiply, screen, overlay, hard_light, soft_light, difference
        #[arg(long, default_value = "overlay")]
        mode: String,

        /// Mix amount (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        mix: f64,
    },
}

/// Handle VFX command dispatch.
pub async fn handle_vfx_command(command: VfxCommand, json_output: bool) -> Result<()> {
    match command {
        VfxCommand::Apply {
            input,
            output,
            effect,
            strength,
            params,
        } => {
            handle_apply(
                &input,
                &output,
                &effect,
                strength,
                params.as_deref(),
                json_output,
            )
            .await
        }
        VfxCommand::List => handle_list(json_output).await,
        VfxCommand::Keying {
            input,
            output,
            key_color,
            tolerance,
            softness,
            background,
            spill_suppress,
        } => {
            handle_keying(
                &input,
                &output,
                &key_color,
                tolerance,
                softness,
                background.as_ref(),
                spill_suppress,
                json_output,
            )
            .await
        }
        VfxCommand::Transition {
            input1,
            input2,
            output,
            transition_type,
            progress,
        } => {
            handle_transition(
                &input1,
                &input2,
                &output,
                &transition_type,
                progress,
                json_output,
            )
            .await
        }
        VfxCommand::Generate {
            output,
            pattern,
            width,
            height,
            color1,
            color2,
            block_size,
        } => {
            handle_generate(
                &output,
                &pattern,
                width,
                height,
                color1.as_deref(),
                color2.as_deref(),
                block_size,
                json_output,
            )
            .await
        }
        VfxCommand::Blend {
            base,
            overlay,
            output,
            mode,
            mix,
        } => handle_blend(&base, &overlay, &output, &mode, mix, json_output).await,
    }
}

// ---------------------------------------------------------------------------
// Available effects and transitions
// ---------------------------------------------------------------------------

const EFFECTS: &[(&str, &str)] = &[
    ("blur", "Gaussian blur with configurable radius"),
    ("sharpen", "Unsharp mask sharpening"),
    ("glow", "Soft glow effect (bloom)"),
    ("vintage", "Vintage film look with color shift"),
    ("sepia", "Sepia tone color grading"),
    ("negative", "Invert all colors"),
    ("posterize", "Reduce color levels for poster effect"),
    ("pixelate", "Mosaic/pixelation effect"),
    ("vignette", "Darkened edges vignette"),
    ("contrast", "Contrast enhancement"),
    ("brightness", "Brightness adjustment"),
    ("saturation", "Color saturation adjustment"),
    ("hue_shift", "Hue rotation"),
    ("noise", "Film grain noise overlay"),
    ("emboss", "Emboss/relief effect"),
    ("edge_detect", "Edge detection (Sobel)"),
];

const TRANSITIONS: &[(&str, &str)] = &[
    ("dissolve", "Cross-dissolve blend between frames"),
    ("wipe_left", "Horizontal wipe from right to left"),
    ("wipe_right", "Horizontal wipe from left to right"),
    ("wipe_up", "Vertical wipe from bottom to top"),
    ("wipe_down", "Vertical wipe from top to bottom"),
    ("fade", "Fade through black"),
];

const BLEND_MODES: &[(&str, &str)] = &[
    ("multiply", "Darken by multiplying pixel values"),
    ("screen", "Lighten using screen formula"),
    ("overlay", "Overlay: multiply dark, screen light"),
    ("hard_light", "Hard light blending"),
    ("soft_light", "Soft light blending"),
    ("difference", "Absolute difference between layers"),
    ("exclusion", "Exclusion blending"),
    ("linear_dodge", "Add (linear dodge)"),
    ("linear_burn", "Subtract (linear burn)"),
];

const PATTERNS: &[(&str, &str)] = &[
    ("color_bars", "SMPTE/EBU color bars"),
    ("gradient", "Horizontal gradient between two colors"),
    ("noise", "Random noise pattern"),
    ("checkerboard", "Checkerboard pattern"),
    ("solid", "Solid color fill"),
];

// ---------------------------------------------------------------------------
// Handler: Apply
// ---------------------------------------------------------------------------

async fn handle_apply(
    input: &PathBuf,
    output: &PathBuf,
    effect: &str,
    strength: f64,
    params: Option<&str>,
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

    // Validate effect name
    let valid_effect = EFFECTS.iter().any(|(name, _)| *name == effect);
    if !valid_effect {
        let available: Vec<&str> = EFFECTS.iter().map(|(n, _)| *n).collect();
        return Err(anyhow::anyhow!(
            "Unknown effect '{}'. Available: {}",
            effect,
            available.join(", ")
        ));
    }

    // Parse extra parameters
    let param_map = parse_params(params);

    if json_output {
        let result = serde_json::json!({
            "action": "apply_effect",
            "input": input.display().to_string(),
            "output": output.display().to_string(),
            "effect": effect,
            "strength": strength,
            "params": param_map,
            "status": "pending_frame_pipeline",
            "message": "Effect configured; awaiting frame decoding pipeline integration",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Apply Effect".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {}", "Effect:", effect);
        println!("{:20} {:.2}", "Strength:", strength);

        if !param_map.is_empty() {
            println!("{:20}", "Parameters:");
            for (k, v) in &param_map {
                println!("  {:18} {}", format!("{}:", k), v);
            }
        }

        println!();
        println!(
            "{}",
            "Note: Effect processing requires frame decoding pipeline.".yellow()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: List
// ---------------------------------------------------------------------------

async fn handle_list(json_output: bool) -> Result<()> {
    if json_output {
        let result = serde_json::json!({
            "effects": EFFECTS.iter().map(|(n, d)| {
                serde_json::json!({"name": n, "description": d})
            }).collect::<Vec<_>>(),
            "transitions": TRANSITIONS.iter().map(|(n, d)| {
                serde_json::json!({"name": n, "description": d})
            }).collect::<Vec<_>>(),
            "blend_modes": BLEND_MODES.iter().map(|(n, d)| {
                serde_json::json!({"name": n, "description": d})
            }).collect::<Vec<_>>(),
            "patterns": PATTERNS.iter().map(|(n, d)| {
                serde_json::json!({"name": n, "description": d})
            }).collect::<Vec<_>>(),
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Available Effects".green().bold());
        println!("{}", "=".repeat(60));
        for (name, desc) in EFFECTS {
            println!("  {:16} {}", name.cyan(), desc);
        }
        println!();

        println!("{}", "Available Transitions".green().bold());
        println!("{}", "=".repeat(60));
        for (name, desc) in TRANSITIONS {
            println!("  {:16} {}", name.cyan(), desc);
        }
        println!();

        println!("{}", "Blend Modes".green().bold());
        println!("{}", "=".repeat(60));
        for (name, desc) in BLEND_MODES {
            println!("  {:16} {}", name.cyan(), desc);
        }
        println!();

        println!("{}", "Test Patterns".green().bold());
        println!("{}", "=".repeat(60));
        for (name, desc) in PATTERNS {
            println!("  {:16} {}", name.cyan(), desc);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Keying
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn handle_keying(
    input: &PathBuf,
    output: &PathBuf,
    key_color: &str,
    tolerance: f64,
    softness: f64,
    background: Option<&PathBuf>,
    spill_suppress: bool,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    if !(0.0..=1.0).contains(&tolerance) {
        return Err(anyhow::anyhow!(
            "Tolerance must be between 0.0 and 1.0, got {}",
            tolerance
        ));
    }
    if !(0.0..=1.0).contains(&softness) {
        return Err(anyhow::anyhow!(
            "Softness must be between 0.0 and 1.0, got {}",
            softness
        ));
    }

    if let Some(bg) = background {
        if !bg.exists() {
            return Err(anyhow::anyhow!(
                "Background file not found: {}",
                bg.display()
            ));
        }
    }

    let (key_r, key_g, key_b) = parse_key_color(key_color)?;

    if json_output {
        let result = serde_json::json!({
            "action": "chroma_key",
            "input": input.display().to_string(),
            "output": output.display().to_string(),
            "key_color": {
                "name": key_color,
                "r": key_r,
                "g": key_g,
                "b": key_b,
            },
            "tolerance": tolerance,
            "softness": softness,
            "background": background.map(|b| b.display().to_string()),
            "spill_suppress": spill_suppress,
            "status": "pending_frame_pipeline",
            "message": "Chroma key configured; awaiting frame decoding pipeline integration",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Chroma Key".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Output:", output.display());
        println!(
            "{:20} {} (RGB: {},{},{})",
            "Key color:", key_color, key_r, key_g, key_b
        );
        println!("{:20} {:.2}", "Tolerance:", tolerance);
        println!("{:20} {:.2}", "Softness:", softness);
        println!(
            "{:20} {}",
            "Background:",
            background.map_or("none".to_string(), |b| b.display().to_string())
        );
        println!("{:20} {}", "Spill suppress:", spill_suppress);
        println!();
        println!(
            "{}",
            "Note: Chroma key requires frame decoding pipeline.".yellow()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Transition
// ---------------------------------------------------------------------------

async fn handle_transition(
    input1: &PathBuf,
    input2: &PathBuf,
    output: &PathBuf,
    transition_type: &str,
    progress: f64,
    json_output: bool,
) -> Result<()> {
    if !input1.exists() {
        return Err(anyhow::anyhow!(
            "Input1 file not found: {}",
            input1.display()
        ));
    }
    if !input2.exists() {
        return Err(anyhow::anyhow!(
            "Input2 file not found: {}",
            input2.display()
        ));
    }
    if !(0.0..=1.0).contains(&progress) {
        return Err(anyhow::anyhow!(
            "Progress must be between 0.0 and 1.0, got {}",
            progress
        ));
    }

    let valid_transition = TRANSITIONS.iter().any(|(name, _)| *name == transition_type);
    if !valid_transition {
        let available: Vec<&str> = TRANSITIONS.iter().map(|(n, _)| *n).collect();
        return Err(anyhow::anyhow!(
            "Unknown transition '{}'. Available: {}",
            transition_type,
            available.join(", ")
        ));
    }

    if json_output {
        let result = serde_json::json!({
            "action": "transition",
            "input1": input1.display().to_string(),
            "input2": input2.display().to_string(),
            "output": output.display().to_string(),
            "transition_type": transition_type,
            "progress": progress,
            "status": "pending_frame_pipeline",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Transition".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input 1:", input1.display());
        println!("{:20} {}", "Input 2:", input2.display());
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {}", "Type:", transition_type);
        println!("{:20} {:.2}", "Progress:", progress);
        println!();
        println!(
            "{}",
            "Note: Transition requires frame decoding pipeline.".yellow()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Generate
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn handle_generate(
    output: &PathBuf,
    pattern: &str,
    width: u32,
    height: u32,
    color1: Option<&str>,
    color2: Option<&str>,
    block_size: u32,
    json_output: bool,
) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(anyhow::anyhow!(
            "Width and height must be > 0, got {}x{}",
            width,
            height
        ));
    }

    let valid_pattern = PATTERNS.iter().any(|(name, _)| *name == pattern);
    if !valid_pattern {
        let available: Vec<&str> = PATTERNS.iter().map(|(n, _)| *n).collect();
        return Err(anyhow::anyhow!(
            "Unknown pattern '{}'. Available: {}",
            pattern,
            available.join(", ")
        ));
    }

    // Generate the pattern data
    let data = generate_pattern_data(pattern, width, height, color1, color2, block_size)?;

    // Write as raw PPM (simple lossless format)
    let ppm_header = format!("P6\n{} {}\n255\n", width, height);
    let mut file_data = ppm_header.into_bytes();
    file_data.extend_from_slice(&data);
    std::fs::write(output, &file_data).context("Failed to write output file")?;

    if json_output {
        let result = serde_json::json!({
            "action": "generate",
            "output": output.display().to_string(),
            "pattern": pattern,
            "width": width,
            "height": height,
            "status": "generated",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Pattern Generated".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {}", "Pattern:", pattern);
        println!("{:20} {}x{}", "Resolution:", width, height);
        println!("{:20} {} bytes", "Size:", file_data.len());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handler: Blend
// ---------------------------------------------------------------------------

async fn handle_blend(
    base: &PathBuf,
    overlay: &PathBuf,
    output: &PathBuf,
    mode: &str,
    mix: f64,
    json_output: bool,
) -> Result<()> {
    if !base.exists() {
        return Err(anyhow::anyhow!("Base image not found: {}", base.display()));
    }
    if !overlay.exists() {
        return Err(anyhow::anyhow!(
            "Overlay image not found: {}",
            overlay.display()
        ));
    }
    if !(0.0..=1.0).contains(&mix) {
        return Err(anyhow::anyhow!(
            "Mix must be between 0.0 and 1.0, got {}",
            mix
        ));
    }

    let valid_mode = BLEND_MODES.iter().any(|(name, _)| *name == mode);
    if !valid_mode {
        let available: Vec<&str> = BLEND_MODES.iter().map(|(n, _)| *n).collect();
        return Err(anyhow::anyhow!(
            "Unknown blend mode '{}'. Available: {}",
            mode,
            available.join(", ")
        ));
    }

    if json_output {
        let result = serde_json::json!({
            "action": "blend",
            "base": base.display().to_string(),
            "overlay": overlay.display().to_string(),
            "output": output.display().to_string(),
            "mode": mode,
            "mix": mix,
            "status": "pending_frame_pipeline",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Blend".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Base:", base.display());
        println!("{:20} {}", "Overlay:", overlay.display());
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {}", "Mode:", mode);
        println!("{:20} {:.2}", "Mix:", mix);
        println!();
        println!(
            "{}",
            "Note: Blend requires frame decoding pipeline.".yellow()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Pattern generation
// ---------------------------------------------------------------------------

fn generate_pattern_data(
    pattern: &str,
    width: u32,
    height: u32,
    color1: Option<&str>,
    color2: Option<&str>,
    block_size: u32,
) -> Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    let mut data = vec![0u8; w * h * 3];

    match pattern {
        "color_bars" => generate_color_bars(&mut data, w, h),
        "gradient" => {
            let c1 = parse_color_hex(color1.unwrap_or("#000000"))?;
            let c2 = parse_color_hex(color2.unwrap_or("#FFFFFF"))?;
            generate_gradient(&mut data, w, h, c1, c2);
        }
        "noise" => generate_noise(&mut data, w, h),
        "checkerboard" => {
            let c1 = parse_color_hex(color1.unwrap_or("#FFFFFF"))?;
            let c2 = parse_color_hex(color2.unwrap_or("#000000"))?;
            generate_checkerboard(&mut data, w, h, block_size as usize, c1, c2);
        }
        "solid" => {
            let c1 = parse_color_hex(color1.unwrap_or("#808080"))?;
            generate_solid(&mut data, w, h, c1);
        }
        _ => {
            return Err(anyhow::anyhow!("Unknown pattern: {}", pattern));
        }
    }

    Ok(data)
}

fn generate_color_bars(data: &mut [u8], width: usize, height: usize) {
    // SMPTE color bars: white, yellow, cyan, green, magenta, red, blue
    let bars: [(u8, u8, u8); 7] = [
        (235, 235, 235), // white
        (235, 235, 16),  // yellow
        (16, 235, 235),  // cyan
        (16, 235, 16),   // green
        (235, 16, 235),  // magenta
        (235, 16, 16),   // red
        (16, 16, 235),   // blue
    ];

    let bar_width = width / 7;
    for y in 0..height {
        for x in 0..width {
            let bar_idx = (x / bar_width.max(1)).min(6);
            let (r, g, b) = bars[bar_idx];
            let idx = (y * width + x) * 3;
            if idx + 2 < data.len() {
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
            }
        }
    }
}

fn generate_gradient(
    data: &mut [u8],
    width: usize,
    height: usize,
    c1: (u8, u8, u8),
    c2: (u8, u8, u8),
) {
    for y in 0..height {
        for x in 0..width {
            let t = if width > 1 {
                x as f64 / (width - 1) as f64
            } else {
                0.0
            };
            let r = (c1.0 as f64 * (1.0 - t) + c2.0 as f64 * t).round() as u8;
            let g = (c1.1 as f64 * (1.0 - t) + c2.1 as f64 * t).round() as u8;
            let b = (c1.2 as f64 * (1.0 - t) + c2.2 as f64 * t).round() as u8;
            let idx = (y * width + x) * 3;
            if idx + 2 < data.len() {
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
            }
        }
    }
}

fn generate_noise(data: &mut [u8], width: usize, height: usize) {
    // Simple deterministic pseudo-noise using a linear congruential generator
    let mut state: u64 = 0x5DEE_CE66_D_u64;
    for y in 0..height {
        for x in 0..width {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let val = ((state >> 33) & 0xFF) as u8;
            let idx = (y * width + x) * 3;
            if idx + 2 < data.len() {
                data[idx] = val;
                data[idx + 1] = val;
                data[idx + 2] = val;
            }
        }
    }
}

fn generate_checkerboard(
    data: &mut [u8],
    width: usize,
    height: usize,
    block_size: usize,
    c1: (u8, u8, u8),
    c2: (u8, u8, u8),
) {
    let bs = block_size.max(1);
    for y in 0..height {
        for x in 0..width {
            let checker = ((x / bs) + (y / bs)) % 2 == 0;
            let (r, g, b) = if checker { c1 } else { c2 };
            let idx = (y * width + x) * 3;
            if idx + 2 < data.len() {
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
            }
        }
    }
}

fn generate_solid(data: &mut [u8], width: usize, height: usize, color: (u8, u8, u8)) {
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            if idx + 2 < data.len() {
                data[idx] = color.0;
                data[idx + 1] = color.1;
                data[idx + 2] = color.2;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing utilities
// ---------------------------------------------------------------------------

fn parse_key_color(color: &str) -> Result<(u8, u8, u8)> {
    match color.to_lowercase().as_str() {
        "green" => Ok((0, 177, 64)),
        "blue" => Ok((0, 71, 187)),
        s if s.starts_with('#') && s.len() == 7 => {
            let r = u8::from_str_radix(&s[1..3], 16)
                .map_err(|_| anyhow::anyhow!("Invalid hex color: {}", color))?;
            let g = u8::from_str_radix(&s[3..5], 16)
                .map_err(|_| anyhow::anyhow!("Invalid hex color: {}", color))?;
            let b = u8::from_str_radix(&s[5..7], 16)
                .map_err(|_| anyhow::anyhow!("Invalid hex color: {}", color))?;
            Ok((r, g, b))
        }
        _ => Err(anyhow::anyhow!(
            "Invalid key color '{}'. Use 'green', 'blue', or '#RRGGBB'",
            color
        )),
    }
}

fn parse_color_hex(hex: &str) -> Result<(u8, u8, u8)> {
    let s = hex.trim();
    if s.starts_with('#') && s.len() == 7 {
        let r = u8::from_str_radix(&s[1..3], 16)
            .map_err(|_| anyhow::anyhow!("Invalid hex color: {}", hex))?;
        let g = u8::from_str_radix(&s[3..5], 16)
            .map_err(|_| anyhow::anyhow!("Invalid hex color: {}", hex))?;
        let b = u8::from_str_radix(&s[5..7], 16)
            .map_err(|_| anyhow::anyhow!("Invalid hex color: {}", hex))?;
        Ok((r, g, b))
    } else {
        Err(anyhow::anyhow!(
            "Invalid hex color '{}'. Expected format: #RRGGBB",
            hex
        ))
    }
}

fn parse_params(params: Option<&str>) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    if let Some(p) = params {
        for pair in p.split(',') {
            let parts: Vec<&str> = pair.splitn(2, '=').collect();
            if parts.len() == 2 {
                map.insert(parts[0].trim().to_string(), parts[1].trim().to_string());
            }
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_key_color_green() {
        let (r, g, b) = parse_key_color("green").expect("should parse green");
        assert_eq!(r, 0);
        assert_eq!(g, 177);
        assert_eq!(b, 64);
    }

    #[test]
    fn test_parse_key_color_hex() {
        let (r, g, b) = parse_key_color("#FF8000").expect("should parse hex");
        assert_eq!(r, 255);
        assert_eq!(g, 128);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_parse_key_color_invalid() {
        assert!(parse_key_color("purple").is_err());
        assert!(parse_key_color("#GG0000").is_err());
        assert!(parse_key_color("#FFF").is_err());
    }

    #[test]
    fn test_parse_color_hex() {
        let (r, g, b) = parse_color_hex("#000000").expect("should parse black");
        assert_eq!((r, g, b), (0, 0, 0));

        let (r, g, b) = parse_color_hex("#FFFFFF").expect("should parse white");
        assert_eq!((r, g, b), (255, 255, 255));
    }

    #[test]
    fn test_parse_params() {
        let map = parse_params(Some("radius=5,intensity=0.8"));
        assert_eq!(map.get("radius").map(|s| s.as_str()), Some("5"));
        assert_eq!(map.get("intensity").map(|s| s.as_str()), Some("0.8"));
    }

    #[test]
    fn test_parse_params_empty() {
        let map = parse_params(None);
        assert!(map.is_empty());
    }

    #[test]
    fn test_generate_color_bars() {
        let mut data = vec![0u8; 70 * 10 * 3];
        generate_color_bars(&mut data, 70, 10);
        // First bar should be white-ish
        assert!(data[0] > 200);
        assert!(data[1] > 200);
        assert!(data[2] > 200);
    }

    #[test]
    fn test_generate_solid() {
        let mut data = vec![0u8; 4 * 4 * 3];
        generate_solid(&mut data, 4, 4, (128, 64, 32));
        assert_eq!(data[0], 128);
        assert_eq!(data[1], 64);
        assert_eq!(data[2], 32);
    }

    #[test]
    fn test_generate_checkerboard() {
        let mut data = vec![0u8; 4 * 4 * 3];
        generate_checkerboard(&mut data, 4, 4, 2, (255, 255, 255), (0, 0, 0));
        // (0,0) should be white, (2,0) should be black
        assert_eq!(data[0], 255); // (0,0) R
        let idx_2_0 = 2 * 3;
        assert_eq!(data[idx_2_0], 0); // (2,0) R
    }

    #[test]
    fn test_generate_gradient() {
        let mut data = vec![0u8; 3 * 1 * 3]; // 3 pixels wide, 1 pixel tall
        generate_gradient(&mut data, 3, 1, (0, 0, 0), (255, 255, 255));
        assert_eq!(data[0], 0); // first pixel R
        assert_eq!(data[3], 128); // middle pixel R (approximately)
        assert_eq!(data[6], 255); // last pixel R
    }

    #[test]
    fn test_generate_pattern_data_solid() {
        let data = generate_pattern_data("solid", 8, 8, Some("#FF0000"), None, 1)
            .expect("should generate solid");
        assert_eq!(data.len(), 8 * 8 * 3);
        assert_eq!(data[0], 255);
        assert_eq!(data[1], 0);
        assert_eq!(data[2], 0);
    }
}
