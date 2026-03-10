//! Color management command for `oximedia color`.
//!
//! Provides convert, info, matrix, and delta-e subcommands via `oximedia-colormgmt`.

use anyhow::Result;
use clap::Subcommand;
use colored::Colorize;
use oximedia_colormgmt::{ColorSpaceId, ColorTransformUtil};

/// Subcommands for `oximedia color`.
#[derive(Subcommand)]
pub enum ColorCommand {
    /// Convert an RGB triplet between two color spaces
    Convert {
        /// Red channel value (0.0 – 1.0)
        #[arg(long)]
        r: f64,

        /// Green channel value (0.0 – 1.0)
        #[arg(long)]
        g: f64,

        /// Blue channel value (0.0 – 1.0)
        #[arg(long)]
        b: f64,

        /// Source color space (srgb, rec709, rec2020, p3-dci, p3-d65, aces-ap0, aces-ap1, linear)
        #[arg(long)]
        from: String,

        /// Destination color space
        #[arg(long)]
        to: String,
    },

    /// Show information about a color space
    Info {
        /// Color space name (srgb, rec709, rec2020, p3-dci, p3-d65, aces-ap0, aces-ap1, linear)
        #[arg(value_name = "SPACE")]
        space: String,
    },

    /// Show the 3×3 conversion matrix between two color spaces
    Matrix {
        /// Source color space
        #[arg(long)]
        from: String,

        /// Destination color space
        #[arg(long)]
        to: String,
    },

    /// Compute ΔE (color difference) between two RGB colors
    DeltaE {
        /// First color: red channel
        #[arg(long)]
        r1: f64,

        /// First color: green channel
        #[arg(long)]
        g1: f64,

        /// First color: blue channel
        #[arg(long)]
        b1: f64,

        /// Second color: red channel
        #[arg(long)]
        r2: f64,

        /// Second color: green channel
        #[arg(long)]
        g2: f64,

        /// Second color: blue channel
        #[arg(long)]
        b2: f64,

        /// Color space for both inputs (default: srgb)
        #[arg(long, default_value = "srgb")]
        space: String,
    },
}

/// Entry point called from `main.rs`.
pub async fn run_color(command: ColorCommand, json_output: bool) -> Result<()> {
    match command {
        ColorCommand::Convert { r, g, b, from, to } => {
            cmd_convert(r, g, b, &from, &to, json_output)
        }
        ColorCommand::Info { space } => cmd_info(&space, json_output),
        ColorCommand::Matrix { from, to } => cmd_matrix(&from, &to, json_output),
        ColorCommand::DeltaE {
            r1,
            g1,
            b1,
            r2,
            g2,
            b2,
            space,
        } => cmd_delta_e(r1, g1, b1, r2, g2, b2, &space, json_output),
    }
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

fn cmd_convert(
    r: f64,
    g: f64,
    b: f64,
    from_str: &str,
    to_str: &str,
    json_output: bool,
) -> Result<()> {
    let from = parse_color_space_id(from_str)?;
    let to = parse_color_space_id(to_str)?;

    let input = [[r, g, b]];
    let output = ColorTransformUtil::convert(&input, from, to)
        .map_err(|e| anyhow::anyhow!("Color conversion failed: {}", e))?;

    let [ro, go, bo] = output[0];

    if json_output {
        let obj = serde_json::json!({
            "from": from.name(),
            "to": to.name(),
            "input": { "r": r, "g": g, "b": b },
            "output": { "r": ro, "g": go, "b": bo },
            "input_8bit": {
                "r": (r.clamp(0.0, 1.0) * 255.0).round() as u8,
                "g": (g.clamp(0.0, 1.0) * 255.0).round() as u8,
                "b": (b.clamp(0.0, 1.0) * 255.0).round() as u8,
            },
            "output_8bit": {
                "r": (ro.clamp(0.0, 1.0) * 255.0).round() as u8,
                "g": (go.clamp(0.0, 1.0) * 255.0).round() as u8,
                "b": (bo.clamp(0.0, 1.0) * 255.0).round() as u8,
            },
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "Color Space Conversion".green().bold());
    println!("  From:    {}", from.name().cyan());
    println!("  To:      {}", to.name().cyan());
    println!();
    println!("  {} Input:  ({:.6}, {:.6}, {:.6})", "→".blue(), r, g, b);
    println!(
        "          8-bit: rgb({}, {}, {})",
        (r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (b.clamp(0.0, 1.0) * 255.0).round() as u8
    );
    println!();
    println!(
        "  {} Output: ({:.6}, {:.6}, {:.6})",
        "✓".green(),
        ro,
        go,
        bo
    );
    println!(
        "          8-bit: rgb({}, {}, {})",
        (ro.clamp(0.0, 1.0) * 255.0).round() as u8,
        (go.clamp(0.0, 1.0) * 255.0).round() as u8,
        (bo.clamp(0.0, 1.0) * 255.0).round() as u8
    );

    Ok(())
}

fn cmd_info(space_str: &str, json_output: bool) -> Result<()> {
    use oximedia_colormgmt::colorspaces::ColorSpace;

    let id = parse_color_space_id(space_str)?;
    let cs: ColorSpace = id
        .to_color_space()
        .map_err(|e| anyhow::anyhow!("Color space definition error: {}", e))?;

    // Extract primaries
    let r_xy = cs.primaries.red;
    let g_xy = cs.primaries.green;
    let b_xy = cs.primaries.blue;

    if json_output {
        let obj = serde_json::json!({
            "id": format!("{:?}", id),
            "name": cs.name,
            "white_point": format!("{:?}", cs.white_point),
            "transfer_characteristic": format!("{:?}", cs.transfer),
            "primaries": {
                "red_xy": [r_xy.0, r_xy.1],
                "green_xy": [g_xy.0, g_xy.1],
                "blue_xy": [b_xy.0, b_xy.1],
            },
            "rgb_to_xyz_matrix": cs.rgb_to_xyz,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "Color Space Information".green().bold());
    println!("  Name:          {}", cs.name.yellow().bold());
    println!("  ID:            {:?}", id);
    println!("  White point:   {:?}", cs.white_point);
    println!("  Transfer fn:   {:?}", cs.transfer);
    println!();
    println!("  {}", "Primaries (CIE xy chromaticity):".cyan().bold());
    println!("    Red:   ({:.6}, {:.6})", r_xy.0, r_xy.1);
    println!("    Green: ({:.6}, {:.6})", g_xy.0, g_xy.1);
    println!("    Blue:  ({:.6}, {:.6})", b_xy.0, b_xy.1);
    println!();
    println!("  {}", "RGB → XYZ matrix:".cyan().bold());
    let m = cs.rgb_to_xyz;
    println!("    [ {:+.6}  {:+.6}  {:+.6} ]", m[0][0], m[0][1], m[0][2]);
    println!("    [ {:+.6}  {:+.6}  {:+.6} ]", m[1][0], m[1][1], m[1][2]);
    println!("    [ {:+.6}  {:+.6}  {:+.6} ]", m[2][0], m[2][1], m[2][2]);

    Ok(())
}

fn cmd_matrix(from_str: &str, to_str: &str, json_output: bool) -> Result<()> {
    use oximedia_colormgmt::transforms::create_rgb_to_rgb_matrix;

    let from_id = parse_color_space_id(from_str)?;
    let to_id = parse_color_space_id(to_str)?;

    let from_cs = from_id
        .to_color_space()
        .map_err(|e| anyhow::anyhow!("Source color space error: {}", e))?;
    let to_cs = to_id
        .to_color_space()
        .map_err(|e| anyhow::anyhow!("Destination color space error: {}", e))?;

    let matrix = create_rgb_to_rgb_matrix(&from_cs, &to_cs)
        .map_err(|e| anyhow::anyhow!("Matrix computation failed: {}", e))?;

    if json_output {
        let obj = serde_json::json!({
            "from": from_id.name(),
            "to": to_id.name(),
            "matrix_3x3": [
                [matrix[0][0], matrix[0][1], matrix[0][2]],
                [matrix[1][0], matrix[1][1], matrix[1][2]],
                [matrix[2][0], matrix[2][1], matrix[2][2]],
            ],
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "Color Conversion Matrix".green().bold());
    println!("  From:  {}", from_id.name().cyan());
    println!("  To:    {}", to_id.name().cyan());
    println!();
    println!(
        "  {} (3×3 linearized RGB matrix):",
        "Matrix".yellow().bold()
    );
    println!(
        "    [ {:+.8}  {:+.8}  {:+.8} ]",
        matrix[0][0], matrix[0][1], matrix[0][2]
    );
    println!(
        "    [ {:+.8}  {:+.8}  {:+.8} ]",
        matrix[1][0], matrix[1][1], matrix[1][2]
    );
    println!(
        "    [ {:+.8}  {:+.8}  {:+.8} ]",
        matrix[2][0], matrix[2][1], matrix[2][2]
    );
    println!();
    println!("  Note: apply EOTF before and OETF after this matrix.");

    Ok(())
}

fn cmd_delta_e(
    r1: f64,
    g1: f64,
    b1: f64,
    r2: f64,
    g2: f64,
    b2: f64,
    space_str: &str,
    json_output: bool,
) -> Result<()> {
    use oximedia_colormgmt::delta_e::{delta_e_1976, delta_e_2000};
    use oximedia_colormgmt::xyz::{Lab, Xyz};

    let cs_id = parse_color_space_id(space_str)?;
    let cs = cs_id
        .to_color_space()
        .map_err(|e| anyhow::anyhow!("Color space error: {}", e))?;

    // Convert both RGB values to XYZ
    let xyz1: Xyz = cs.rgb_to_xyz([r1, g1, b1]);
    let xyz2: Xyz = cs.rgb_to_xyz([r2, g2, b2]);

    // D65 white point for Lab conversion
    let d65 = Xyz::new(0.950_489, 1.000_000, 1.088_840);

    let lab1 = Lab::from_xyz(&xyz1, &d65);
    let lab2 = Lab::from_xyz(&xyz2, &d65);

    let de76 = delta_e_1976(&lab1, &lab2);
    let de2000 = delta_e_2000(&lab1, &lab2);

    let perception_76 = interpret_delta_e(de76);
    let perception_2000 = interpret_delta_e(de2000);

    if json_output {
        let obj = serde_json::json!({
            "color_space": cs_id.name(),
            "color1": { "r": r1, "g": g1, "b": b1 },
            "color2": { "r": r2, "g": g2, "b": b2 },
            "lab1": { "L": lab1.l, "a": lab1.a, "b": lab1.b },
            "lab2": { "L": lab2.l, "a": lab2.a, "b": lab2.b },
            "delta_e_1976": de76,
            "delta_e_2000": de2000,
            "perception_1976": perception_76,
            "perception_2000": perception_2000,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "ΔE Color Difference".green().bold());
    println!("  Space:    {}", cs_id.name().cyan());
    println!();
    println!(
        "  {} Color 1: ({:.4}, {:.4}, {:.4})",
        "●".blue(),
        r1,
        g1,
        b1
    );
    println!(
        "          Lab: L={:.2}  a={:.2}  b={:.2}",
        lab1.l, lab1.a, lab1.b
    );
    println!();
    println!(
        "  {} Color 2: ({:.4}, {:.4}, {:.4})",
        "●".yellow(),
        r2,
        g2,
        b2
    );
    println!(
        "          Lab: L={:.2}  a={:.2}  b={:.2}",
        lab2.l, lab2.a, lab2.b
    );
    println!();
    println!("  {}", "Results:".cyan().bold());
    println!("    ΔE 1976  (CIE76):   {:7.4}  — {}", de76, perception_76);
    println!(
        "    ΔE 2000 (CIEDE2000): {:7.4}  — {}",
        de2000, perception_2000
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a color space string into a `ColorSpaceId` variant.
fn parse_color_space_id(s: &str) -> Result<ColorSpaceId> {
    match s.to_lowercase().replace(['-', '_', '.'], "").as_str() {
        "srgb" | "srgb1" => Ok(ColorSpaceId::SRGB),
        "rec709" | "bt709" | "itu709" | "hd" => Ok(ColorSpaceId::Rec709),
        "rec2020" | "bt2020" | "itu2020" | "uhd" => Ok(ColorSpaceId::Rec2020),
        "p3dci" | "dcip3" | "dci" => Ok(ColorSpaceId::P3DCI),
        "p3d65" | "displayp3" | "p3" => Ok(ColorSpaceId::P3D65),
        "acesap0" | "aces20651" | "aces" => Ok(ColorSpaceId::AcesAP0),
        "acesap1" | "acescg" | "ap1" => Ok(ColorSpaceId::AcesAP1),
        "linear" | "scenelinear" | "linearrec709" => Ok(ColorSpaceId::Linear),
        other => anyhow::bail!(
            "Unknown color space '{}'. Supported: srgb, rec709, rec2020, p3-dci, p3-d65, aces-ap0, aces-ap1, linear",
            other
        ),
    }
}

/// Human-readable interpretation of a Delta E value.
fn interpret_delta_e(de: f64) -> &'static str {
    if de < 1.0 {
        "imperceptible difference"
    } else if de < 2.0 {
        "perceptible only to trained eye"
    } else if de < 3.5 {
        "perceptible to average observer"
    } else if de < 5.0 {
        "clearly perceptible"
    } else if de < 10.0 {
        "significant difference"
    } else {
        "very large difference"
    }
}
