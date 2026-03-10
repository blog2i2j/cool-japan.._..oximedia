//! Image operations — read, convert, sequence, adjust, and histogram.
//!
//! Provides `oximedia image` with subcommands for professional image
//! sequence workflows (DPX, EXR, TIFF) used in cinema and VFX.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Subcommands for `oximedia image`.
#[derive(Subcommand, Debug)]
pub enum ImageCommand {
    /// Read and display image information
    Read {
        /// Input image file
        #[arg(short, long)]
        input: PathBuf,

        /// Output format: text, json
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Convert image format (DPX, EXR, TIFF, PNG, etc.)
    Convert {
        /// Input image file
        #[arg(short, long)]
        input: PathBuf,

        /// Output image file
        #[arg(short, long)]
        output: PathBuf,

        /// Target bit depth (8, 10, 12, 16, 32)
        #[arg(long)]
        bit_depth: Option<u32>,

        /// Target color space (linear, srgb, rec709, rec2020, dci-p3, log, luma)
        #[arg(long)]
        colorspace: Option<String>,

        /// Compression method (none, rle, zip, zips, piz, lzw, packbits)
        #[arg(long)]
        compression: Option<String>,
    },

    /// Process image sequence
    Sequence {
        /// Input pattern (e.g. "frame_%04d.exr" or "render.####.dpx")
        #[arg(short, long)]
        input: String,

        /// Start frame number
        #[arg(long)]
        start: Option<u32>,

        /// End frame number
        #[arg(long)]
        end: Option<u32>,

        /// Show sequence info only (no processing)
        #[arg(long)]
        info: bool,

        /// Output pattern for converted sequence
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Adjust image properties (brightness, contrast, saturation, gamma, exposure)
    Adjust {
        /// Input image file
        #[arg(short, long)]
        input: PathBuf,

        /// Output image file
        #[arg(short, long)]
        output: PathBuf,

        /// Brightness adjustment (-1.0 to 1.0)
        #[arg(long)]
        brightness: Option<f64>,

        /// Contrast adjustment (0.0 to 3.0, 1.0 = no change)
        #[arg(long)]
        contrast: Option<f64>,

        /// Saturation adjustment (0.0 to 3.0, 1.0 = no change)
        #[arg(long)]
        saturation: Option<f64>,

        /// Gamma correction (0.1 to 5.0, 1.0 = no change)
        #[arg(long)]
        gamma: Option<f64>,

        /// Exposure adjustment in stops (-5.0 to 5.0)
        #[arg(long)]
        exposure: Option<f64>,
    },

    /// Generate histogram from image
    Histogram {
        /// Input image file
        #[arg(short, long)]
        input: PathBuf,

        /// Output histogram image (optional, prints text if omitted)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Histogram mode: rgb, luma, per-channel
        #[arg(long, default_value = "rgb")]
        mode: String,

        /// Histogram image width
        #[arg(long)]
        width: Option<u32>,

        /// Histogram image height
        #[arg(long)]
        height: Option<u32>,
    },
}

/// Handle `oximedia image` subcommand dispatch.
pub async fn handle_image_command(command: ImageCommand, json_output: bool) -> Result<()> {
    match command {
        ImageCommand::Read { input, format } => read_image(&input, &format, json_output).await,
        ImageCommand::Convert {
            input,
            output,
            bit_depth,
            colorspace,
            compression,
        } => convert_image(&input, &output, bit_depth, colorspace, compression).await,
        ImageCommand::Sequence {
            input,
            start,
            end,
            info,
            output,
        } => process_sequence(&input, start, end, info, output, json_output).await,
        ImageCommand::Adjust {
            input,
            output,
            brightness,
            contrast,
            saturation,
            gamma,
            exposure,
        } => {
            adjust_image(
                &input, &output, brightness, contrast, saturation, gamma, exposure,
            )
            .await
        }
        ImageCommand::Histogram {
            input,
            output,
            mode,
            width,
            height,
        } => generate_histogram(&input, output, &mode, width, height, json_output).await,
    }
}

// ---------------------------------------------------------------------------
// Helper: detect format and read metadata from file header
// ---------------------------------------------------------------------------

fn detect_format_from_path(path: &PathBuf) -> String {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "dpx" => "DPX".to_string(),
        "exr" => "OpenEXR".to_string(),
        "tif" | "tiff" => "TIFF".to_string(),
        "png" => "PNG".to_string(),
        "jpg" | "jpeg" => "JPEG".to_string(),
        "bmp" => "BMP".to_string(),
        "gif" => "GIF".to_string(),
        "webp" => "WebP".to_string(),
        "cin" | "cineon" => "Cineon".to_string(),
        other => other.to_uppercase(),
    }
}

fn parse_colorspace(s: &str) -> Result<oximedia_image::ColorSpace> {
    match s.to_ascii_lowercase().as_str() {
        "linear" | "linear-rgb" | "linearrgb" => Ok(oximedia_image::ColorSpace::LinearRgb),
        "srgb" => Ok(oximedia_image::ColorSpace::Srgb),
        "rec709" | "bt709" => Ok(oximedia_image::ColorSpace::Rec709),
        "rec2020" | "bt2020" => Ok(oximedia_image::ColorSpace::Rec2020),
        "dci-p3" | "dcip3" | "p3" => Ok(oximedia_image::ColorSpace::DciP3),
        "log" | "logarithmic" => Ok(oximedia_image::ColorSpace::Log),
        "luma" | "gray" | "grayscale" => Ok(oximedia_image::ColorSpace::Luma),
        "ycbcr" | "yuv" => Ok(oximedia_image::ColorSpace::YCbCr),
        "cmyk" => Ok(oximedia_image::ColorSpace::Cmyk),
        other => Err(anyhow::anyhow!(
            "Unknown color space '{}'. Valid: linear, srgb, rec709, rec2020, dci-p3, log, luma, ycbcr, cmyk",
            other
        )),
    }
}

fn parse_compression(s: &str) -> Result<oximedia_image::Compression> {
    match s.to_ascii_lowercase().as_str() {
        "none" => Ok(oximedia_image::Compression::None),
        "rle" => Ok(oximedia_image::Compression::Rle),
        "zip" => Ok(oximedia_image::Compression::Zip),
        "zips" | "zip-scanline" => Ok(oximedia_image::Compression::ZipScanline),
        "lzw" => Ok(oximedia_image::Compression::Lzw),
        "packbits" => Ok(oximedia_image::Compression::PackBits),
        "piz" => Ok(oximedia_image::Compression::Piz),
        "pxr24" => Ok(oximedia_image::Compression::Pxr24),
        "b44" => Ok(oximedia_image::Compression::B44),
        "b44a" => Ok(oximedia_image::Compression::B44a),
        "dwaa" => Ok(oximedia_image::Compression::Dwaa),
        "dwab" => Ok(oximedia_image::Compression::Dwab),
        other => Err(anyhow::anyhow!(
            "Unknown compression '{}'. Valid: none, rle, zip, zips, lzw, packbits, piz, pxr24, b44, b44a, dwaa, dwab",
            other
        )),
    }
}

fn pixel_type_from_depth(depth: u32) -> Result<oximedia_image::PixelType> {
    match depth {
        8 => Ok(oximedia_image::PixelType::U8),
        10 => Ok(oximedia_image::PixelType::U10),
        12 => Ok(oximedia_image::PixelType::U12),
        16 => Ok(oximedia_image::PixelType::U16),
        32 => Ok(oximedia_image::PixelType::U32),
        other => Err(anyhow::anyhow!(
            "Unsupported bit depth {}. Valid: 8, 10, 12, 16, 32",
            other
        )),
    }
}

fn colorspace_name(cs: oximedia_image::ColorSpace) -> &'static str {
    match cs {
        oximedia_image::ColorSpace::LinearRgb => "Linear RGB",
        oximedia_image::ColorSpace::Srgb => "sRGB",
        oximedia_image::ColorSpace::Rec709 => "Rec. 709",
        oximedia_image::ColorSpace::Rec2020 => "Rec. 2020",
        oximedia_image::ColorSpace::DciP3 => "DCI-P3",
        oximedia_image::ColorSpace::Log => "Logarithmic",
        oximedia_image::ColorSpace::Luma => "Luma",
        oximedia_image::ColorSpace::YCbCr => "YCbCr",
        oximedia_image::ColorSpace::Cmyk => "CMYK",
    }
}

fn compression_name(c: oximedia_image::Compression) -> &'static str {
    match c {
        oximedia_image::Compression::None => "None",
        oximedia_image::Compression::Rle => "RLE",
        oximedia_image::Compression::Zip => "ZIP",
        oximedia_image::Compression::ZipScanline => "ZIP (scanline)",
        oximedia_image::Compression::Lzw => "LZW",
        oximedia_image::Compression::PackBits => "PackBits",
        oximedia_image::Compression::Piz => "PIZ",
        oximedia_image::Compression::Pxr24 => "PXR24",
        oximedia_image::Compression::B44 => "B44",
        oximedia_image::Compression::B44a => "B44A",
        oximedia_image::Compression::Dwaa => "DWAA",
        oximedia_image::Compression::Dwab => "DWAB",
    }
}

// ---------------------------------------------------------------------------
// read_image
// ---------------------------------------------------------------------------

async fn read_image(input: &PathBuf, format: &str, json_output: bool) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let file_size = std::fs::metadata(input)
        .context("Failed to read file metadata")?
        .len();

    // Detect format from magic bytes
    let header_bytes = {
        let mut buf = vec![0u8; 2048.min(file_size as usize)];
        let mut f = std::fs::File::open(input).context("Failed to open input file")?;
        std::io::Read::read(&mut f, &mut buf).context("Failed to read file header")?;
        buf
    };

    let detected_format = oximedia_image::format_detect::FormatDetector::detect(&header_bytes);
    let format_name = detected_format.name();
    let format_ext = detected_format.extension();
    let is_hdr = detected_format.is_hdr();
    let is_lossless = detected_format.is_lossless();

    // Try to read the image frame for dimensions/metadata
    let frame_info = try_read_frame_info(input, &detected_format);

    let output_format = if json_output { "json" } else { format };

    match output_format {
        "json" => {
            let mut info = serde_json::json!({
                "file": input.display().to_string(),
                "file_size": file_size,
                "format": format_name,
                "extension": format_ext,
                "hdr": is_hdr,
                "lossless": is_lossless,
            });

            if let Some(ref fi) = frame_info {
                info["width"] = serde_json::json!(fi.width);
                info["height"] = serde_json::json!(fi.height);
                info["bit_depth"] = serde_json::json!(fi.bit_depth);
                info["components"] = serde_json::json!(fi.components);
                info["colorspace"] = serde_json::json!(fi.colorspace);
                info["pixel_type"] = serde_json::json!(fi.pixel_type);
            }

            let json_str =
                serde_json::to_string_pretty(&info).context("Failed to serialize result")?;
            println!("{}", json_str);
        }
        _ => {
            println!("{}", "Image Information".green().bold());
            println!("{}", "=".repeat(60));
            println!("{:20} {}", "File:", input.display());
            println!("{:20} {} bytes", "File size:", file_size);
            println!("{:20} {}", "Format:", format_name);
            println!("{:20} .{}", "Extension:", format_ext);
            println!("{:20} {}", "HDR:", if is_hdr { "Yes" } else { "No" });
            println!(
                "{:20} {}",
                "Lossless:",
                if is_lossless { "Yes" } else { "No" }
            );

            if let Some(ref fi) = frame_info {
                println!();
                println!("{}", "Image Properties".cyan().bold());
                println!("{}", "-".repeat(60));
                println!("{:20} {}x{}", "Dimensions:", fi.width, fi.height);
                println!("{:20} {}-bit", "Bit depth:", fi.bit_depth);
                println!("{:20} {}", "Components:", fi.components);
                println!("{:20} {}", "Color space:", fi.colorspace);
                println!("{:20} {}", "Pixel type:", fi.pixel_type);
                let pixel_count = (fi.width as u64) * (fi.height as u64);
                let data_size =
                    pixel_count * (fi.components as u64) * ((fi.bit_depth as u64 + 7) / 8);
                println!("{:20} {}", "Pixel count:", pixel_count);
                println!("{:20} {} bytes", "Uncompressed size:", data_size);
            } else {
                println!();
                println!(
                    "{}",
                    "Note: Detailed pixel properties require format-specific decoder integration."
                        .yellow()
                );
            }
        }
    }

    Ok(())
}

struct FrameInfo {
    width: u32,
    height: u32,
    bit_depth: u32,
    components: u32,
    colorspace: String,
    pixel_type: String,
}

fn try_read_frame_info(
    path: &PathBuf,
    detected: &oximedia_image::format_detect::ImageFormat,
) -> Option<FrameInfo> {
    match detected {
        oximedia_image::format_detect::ImageFormat::Dpx => {
            let frame = oximedia_image::dpx::read_dpx(path, 1).ok()?;
            Some(frame_to_info(&frame))
        }
        oximedia_image::format_detect::ImageFormat::Exr => {
            let frame = oximedia_image::exr::read_exr(path, 1).ok()?;
            Some(frame_to_info(&frame))
        }
        oximedia_image::format_detect::ImageFormat::Tiff => {
            let frame = oximedia_image::tiff::read_tiff(path, 1).ok()?;
            Some(frame_to_info(&frame))
        }
        _ => None,
    }
}

fn frame_to_info(frame: &oximedia_image::ImageFrame) -> FrameInfo {
    FrameInfo {
        width: frame.width,
        height: frame.height,
        bit_depth: u32::from(frame.pixel_type.bit_depth()),
        components: u32::from(frame.components),
        colorspace: colorspace_name(frame.color_space).to_string(),
        pixel_type: format!("{:?}", frame.pixel_type),
    }
}

// ---------------------------------------------------------------------------
// convert_image
// ---------------------------------------------------------------------------

async fn convert_image(
    input: &PathBuf,
    output: &PathBuf,
    bit_depth: Option<u32>,
    colorspace: Option<String>,
    compression: Option<String>,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let input_format = detect_format_from_path(input);
    let output_format = detect_format_from_path(output);

    // Parse optional settings
    let target_depth = if let Some(d) = bit_depth {
        Some(pixel_type_from_depth(d)?)
    } else {
        None
    };

    let target_cs = if let Some(ref cs) = colorspace {
        Some(parse_colorspace(cs)?)
    } else {
        None
    };

    let target_compression = if let Some(ref c) = compression {
        Some(parse_compression(c)?)
    } else {
        None
    };

    println!("{}", "Image Conversion".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {} ({})", "Input:", input.display(), input_format);
    println!("{:20} {} ({})", "Output:", output.display(), output_format);

    if let Some(ref pt) = target_depth {
        println!("{:20} {}-bit", "Target bit depth:", pt.bit_depth());
    }
    if let Some(ref cs) = target_cs {
        println!("{:20} {}", "Target colorspace:", colorspace_name(*cs));
    }
    if let Some(ref c) = target_compression {
        println!("{:20} {}", "Compression:", compression_name(*c));
    }

    println!();
    println!(
        "{}",
        "Note: Full conversion pipeline requires frame decoding integration.".yellow()
    );
    println!(
        "{}",
        "Format parsers and pixel type converters are ready; pipeline integration pending."
            .dimmed()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// process_sequence
// ---------------------------------------------------------------------------

async fn process_sequence(
    input: &str,
    start: Option<u32>,
    end: Option<u32>,
    info: bool,
    output: Option<String>,
    json_output: bool,
) -> Result<()> {
    let pattern = oximedia_image::SequencePattern::parse(input)
        .map_err(|e| anyhow::anyhow!("Invalid sequence pattern '{}': {}", input, e))?;

    let start_frame = start.unwrap_or(1);
    let end_frame = end.unwrap_or(start_frame + 99);

    if start_frame > end_frame {
        return Err(anyhow::anyhow!(
            "Start frame ({}) must be <= end frame ({})",
            start_frame,
            end_frame
        ));
    }

    let sequence =
        oximedia_image::ImageSequence::from_pattern(pattern.clone(), start_frame..=end_frame)
            .map_err(|e| anyhow::anyhow!("Failed to create sequence: {}", e))?;

    let frame_count = end_frame - start_frame + 1;
    let has_gaps = !sequence.gaps.is_empty();

    if info || output.is_none() {
        // Show sequence info
        let first_path = pattern.format(start_frame);
        let last_path = pattern.format(end_frame);

        if json_output {
            let result = serde_json::json!({
                "pattern": input,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": frame_count,
                "has_gaps": has_gaps,
                "gaps": sequence.gaps,
                "first_file": first_path.display().to_string(),
                "last_file": last_path.display().to_string(),
            });
            let json_str =
                serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
            println!("{}", json_str);
        } else {
            println!("{}", "Image Sequence".green().bold());
            println!("{}", "=".repeat(60));
            println!("{:20} {}", "Pattern:", input);
            println!("{:20} {}", "Start frame:", start_frame);
            println!("{:20} {}", "End frame:", end_frame);
            println!("{:20} {}", "Frame count:", frame_count);
            println!("{:20} {}", "Has gaps:", if has_gaps { "Yes" } else { "No" });

            if has_gaps {
                let gap_display: Vec<String> =
                    sequence.gaps.iter().map(|g| g.to_string()).collect();
                let display = if gap_display.len() > 10 {
                    format!(
                        "{} ... ({} total)",
                        gap_display[..10].join(", "),
                        gap_display.len()
                    )
                } else {
                    gap_display.join(", ")
                };
                println!("{:20} {}", "Missing frames:", display);
            }

            println!();
            println!("{:20} {}", "First file:", first_path.display());
            println!("{:20} {}", "Last file:", last_path.display());
        }
    }

    if let Some(ref out_pattern) = output {
        println!();
        println!("{}", "Sequence Processing".cyan().bold());
        println!("{}", "-".repeat(60));
        println!("{:20} {}", "Source:", input);
        println!("{:20} {}", "Destination:", out_pattern);
        println!("{:20} {}", "Frames:", frame_count);
        println!();
        println!(
            "{}",
            "Note: Sequence transcoding requires frame read/write pipeline integration.".yellow()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// adjust_image
// ---------------------------------------------------------------------------

async fn adjust_image(
    input: &PathBuf,
    output: &PathBuf,
    brightness: Option<f64>,
    contrast: Option<f64>,
    saturation: Option<f64>,
    gamma: Option<f64>,
    exposure: Option<f64>,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    // Validate parameter ranges
    if let Some(b) = brightness {
        if !(-1.0..=1.0).contains(&b) {
            return Err(anyhow::anyhow!(
                "Brightness must be between -1.0 and 1.0, got {}",
                b
            ));
        }
    }
    if let Some(c) = contrast {
        if !(0.0..=3.0).contains(&c) {
            return Err(anyhow::anyhow!(
                "Contrast must be between 0.0 and 3.0, got {}",
                c
            ));
        }
    }
    if let Some(s) = saturation {
        if !(0.0..=3.0).contains(&s) {
            return Err(anyhow::anyhow!(
                "Saturation must be between 0.0 and 3.0, got {}",
                s
            ));
        }
    }
    if let Some(g) = gamma {
        if !(0.1..=5.0).contains(&g) {
            return Err(anyhow::anyhow!(
                "Gamma must be between 0.1 and 5.0, got {}",
                g
            ));
        }
    }
    if let Some(e) = exposure {
        if !(-5.0..=5.0).contains(&e) {
            return Err(anyhow::anyhow!(
                "Exposure must be between -5.0 and 5.0, got {}",
                e
            ));
        }
    }

    println!("{}", "Image Adjustment".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Input:", input.display());
    println!("{:20} {}", "Output:", output.display());
    println!();

    println!("{}", "Adjustments".cyan().bold());
    println!("{}", "-".repeat(60));

    let mut any_adjustment = false;

    if let Some(b) = brightness {
        println!("{:20} {:+.3}", "Brightness:", b);
        any_adjustment = true;
    }
    if let Some(c) = contrast {
        println!("{:20} {:.3}x", "Contrast:", c);
        any_adjustment = true;
    }
    if let Some(s) = saturation {
        println!("{:20} {:.3}x", "Saturation:", s);
        any_adjustment = true;
    }
    if let Some(g) = gamma {
        println!("{:20} {:.3}", "Gamma:", g);
        any_adjustment = true;
    }
    if let Some(e) = exposure {
        println!("{:20} {:+.3} stops", "Exposure:", e);
        any_adjustment = true;
    }

    if !any_adjustment {
        println!("  (no adjustments specified)");
    }

    println!();
    println!(
        "{}",
        "Note: Image adjustment pipeline requires frame decode/encode integration.".yellow()
    );
    println!(
        "{}",
        "Color adjustment kernels and filter infrastructure are ready.".dimmed()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// generate_histogram
// ---------------------------------------------------------------------------

async fn generate_histogram(
    input: &PathBuf,
    output: Option<PathBuf>,
    mode: &str,
    width: Option<u32>,
    height: Option<u32>,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    // Validate mode
    let valid_modes = ["rgb", "luma", "per-channel"];
    if !valid_modes.contains(&mode) {
        return Err(anyhow::anyhow!(
            "Invalid histogram mode '{}'. Valid: rgb, luma, per-channel",
            mode
        ));
    }

    let hist_width = width.unwrap_or(512);
    let hist_height = height.unwrap_or(256);

    if json_output {
        let result = serde_json::json!({
            "input": input.display().to_string(),
            "mode": mode,
            "histogram_width": hist_width,
            "histogram_height": hist_height,
            "output": output.as_ref().map(|p| p.display().to_string()),
            "status": "pending_frame_decoding",
            "available_modes": valid_modes,
            "message": "Histogram engine ready; awaiting frame decoding pipeline integration",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{}", json_str);
    } else {
        println!("{}", "Image Histogram".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Mode:", mode);
        println!("{:20} {}x{}", "Histogram size:", hist_width, hist_height);

        if let Some(ref out) = output {
            println!("{:20} {}", "Output:", out.display());
        }

        println!();

        println!("{}", "Histogram Engine".cyan().bold());
        println!("{}", "-".repeat(60));
        println!("  Available histogram operations:");
        println!("    - Per-channel RGB histogram (8-bit bins)");
        println!("    - Luminance histogram (BT.601 / BT.709)");
        println!("    - Cumulative distribution function (CDF)");
        println!("    - Histogram equalization LUT generation");
        println!("    - Histogram matching between images");
        println!("    - Contrast stretch / auto-levels");
        println!();

        println!(
            "{}",
            "Note: Full histogram output requires frame decoding pipeline.".yellow()
        );
        println!(
            "{}",
            "Histogram computation engine (oximedia_image::histogram_ops) is ready.".dimmed()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format_from_path() {
        assert_eq!(detect_format_from_path(&PathBuf::from("test.dpx")), "DPX");
        assert_eq!(
            detect_format_from_path(&PathBuf::from("test.exr")),
            "OpenEXR"
        );
        assert_eq!(detect_format_from_path(&PathBuf::from("test.tiff")), "TIFF");
        assert_eq!(detect_format_from_path(&PathBuf::from("test.png")), "PNG");
        assert_eq!(detect_format_from_path(&PathBuf::from("test.jpg")), "JPEG");
    }

    #[test]
    fn test_parse_colorspace_valid() {
        assert_eq!(
            parse_colorspace("srgb").ok(),
            Some(oximedia_image::ColorSpace::Srgb)
        );
        assert_eq!(
            parse_colorspace("linear").ok(),
            Some(oximedia_image::ColorSpace::LinearRgb)
        );
        assert_eq!(
            parse_colorspace("rec709").ok(),
            Some(oximedia_image::ColorSpace::Rec709)
        );
        assert_eq!(
            parse_colorspace("rec2020").ok(),
            Some(oximedia_image::ColorSpace::Rec2020)
        );
        assert_eq!(
            parse_colorspace("dci-p3").ok(),
            Some(oximedia_image::ColorSpace::DciP3)
        );
        assert_eq!(
            parse_colorspace("log").ok(),
            Some(oximedia_image::ColorSpace::Log)
        );
    }

    #[test]
    fn test_parse_colorspace_invalid() {
        assert!(parse_colorspace("invalid").is_err());
    }

    #[test]
    fn test_parse_compression_valid() {
        assert_eq!(
            parse_compression("none").ok(),
            Some(oximedia_image::Compression::None)
        );
        assert_eq!(
            parse_compression("zip").ok(),
            Some(oximedia_image::Compression::Zip)
        );
        assert_eq!(
            parse_compression("piz").ok(),
            Some(oximedia_image::Compression::Piz)
        );
        assert_eq!(
            parse_compression("rle").ok(),
            Some(oximedia_image::Compression::Rle)
        );
    }

    #[test]
    fn test_parse_compression_invalid() {
        assert!(parse_compression("badcompress").is_err());
    }

    #[test]
    fn test_pixel_type_from_depth() {
        assert_eq!(
            pixel_type_from_depth(8).ok(),
            Some(oximedia_image::PixelType::U8)
        );
        assert_eq!(
            pixel_type_from_depth(10).ok(),
            Some(oximedia_image::PixelType::U10)
        );
        assert_eq!(
            pixel_type_from_depth(16).ok(),
            Some(oximedia_image::PixelType::U16)
        );
        assert!(pixel_type_from_depth(7).is_err());
    }

    #[test]
    fn test_colorspace_name() {
        assert_eq!(colorspace_name(oximedia_image::ColorSpace::Srgb), "sRGB");
        assert_eq!(
            colorspace_name(oximedia_image::ColorSpace::LinearRgb),
            "Linear RGB"
        );
    }

    #[test]
    fn test_compression_name() {
        assert_eq!(compression_name(oximedia_image::Compression::None), "None");
        assert_eq!(compression_name(oximedia_image::Compression::Zip), "ZIP");
    }
}
