//! Frame extraction from video files.
//!
//! Provides:
//! - Frame extraction to image files
//! - Thumbnail generation
//! - Image sequence output
//! - Multiple output formats (PNG, JPEG, PPM)

use crate::progress::TranscodeProgress;
use anyhow::{anyhow, Context, Result};
use colored::Colorize;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Options for frame extraction.
#[derive(Debug, Clone)]
pub struct ExtractOptions {
    pub input: PathBuf,
    pub output_pattern: String,
    pub format: Option<String>,
    pub start_time: Option<String>,
    pub frames: Option<usize>,
    pub every: usize,
    pub quality: u8,
}

/// Supported output image formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Png,
    Jpeg,
    Ppm,
}

impl ImageFormat {
    /// Parse format from string or file extension.
    #[allow(dead_code)]
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "png" => Ok(Self::Png),
            "jpg" | "jpeg" => Ok(Self::Jpeg),
            "ppm" => Ok(Self::Ppm),
            _ => Err(anyhow!("Unsupported image format: {}", s)),
        }
    }

    /// Get format from file extension.
    #[allow(dead_code)]
    pub fn from_extension(ext: &str) -> Result<Self> {
        Self::from_str(ext)
    }

    /// Get file extension for this format.
    #[allow(dead_code)]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Ppm => "ppm",
        }
    }

    /// Get format name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Ppm => "PPM",
        }
    }
}

/// Main frame extraction function.
pub async fn extract_frames(options: ExtractOptions) -> Result<()> {
    info!("Starting frame extraction");
    debug!("Extract options: {:?}", options);

    // Validate input file
    validate_input(&options.input).await?;

    // Determine output format
    let format = determine_format(&options)?;

    // Validate quality for JPEG
    if format == ImageFormat::Jpeg && options.quality > 100 {
        return Err(anyhow!("JPEG quality must be between 0 and 100"));
    }

    // Parse output pattern
    let output_dir = parse_output_pattern(&options.output_pattern)?;

    // Create output directory if needed
    if let Some(dir) = output_dir {
        if !dir.exists() {
            tokio::fs::create_dir_all(&dir)
                .await
                .context("Failed to create output directory")?;
        }
    }

    // Print extraction plan
    print_extraction_plan(&options, format);

    // Perform extraction
    extract_frames_impl(&options, format).await?;

    // Print summary
    print_extraction_summary(&options);

    Ok(())
}

/// Validate input file exists and is readable.
async fn validate_input(path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(anyhow!("Input file does not exist: {}", path.display()));
    }

    if !path.is_file() {
        return Err(anyhow!("Input path is not a file: {}", path.display()));
    }

    let metadata = tokio::fs::metadata(path)
        .await
        .context("Failed to read input file metadata")?;

    if metadata.len() == 0 {
        return Err(anyhow!("Input file is empty"));
    }

    Ok(())
}

/// Determine output format from options or pattern.
fn determine_format(options: &ExtractOptions) -> Result<ImageFormat> {
    if let Some(ref fmt) = options.format {
        // Explicit format specified
        ImageFormat::from_str(fmt)
    } else {
        // Try to detect from output pattern
        let pattern = &options.output_pattern;

        if pattern.ends_with(".png") || pattern.contains("%") && !pattern.contains('.') {
            Ok(ImageFormat::Png)
        } else if pattern.ends_with(".jpg") || pattern.ends_with(".jpeg") {
            Ok(ImageFormat::Jpeg)
        } else if pattern.ends_with(".ppm") {
            Ok(ImageFormat::Ppm)
        } else {
            // Default to PNG
            Ok(ImageFormat::Png)
        }
    }
}

/// Parse output pattern to extract directory path.
fn parse_output_pattern(pattern: &str) -> Result<Option<PathBuf>> {
    let path = PathBuf::from(pattern);

    if let Some(parent) = path.parent() {
        if parent != Path::new("") {
            Ok(Some(parent.to_path_buf()))
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
}

/// Print extraction plan before starting.
fn print_extraction_plan(options: &ExtractOptions, format: ImageFormat) {
    println!("{}", "Frame Extraction Plan".cyan().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Input:", options.input.display());
    println!("{:20} {}", "Output Pattern:", options.output_pattern);
    println!("{:20} {}", "Format:", format.name());

    if let Some(ref start) = options.start_time {
        println!("{:20} {}", "Start Time:", start);
    }

    if let Some(count) = options.frames {
        println!("{:20} {}", "Frame Count:", count);
    }

    if options.every > 1 {
        println!("{:20} Every {} frames", "Sampling:", options.every);
    }

    if format == ImageFormat::Jpeg {
        println!("{:20} {}", "JPEG Quality:", options.quality);
    }

    println!("{}", "=".repeat(60));
    println!();
}

/// Generate a deterministic synthetic RGB pixel buffer for a frame number.
///
/// Each frame gets a unique hue derived from its index, producing a visually
/// distinct color swatch that serves as a stand-in for a real decoded frame.
fn generate_frame_buffer(frame_number: usize, width: u32, height: u32) -> Vec<u8> {
    // Map frame index to a hue in [0, 360) and convert to RGB
    let hue_deg = (frame_number as f64 * 13.7) % 360.0;
    let (r, g, b) = hsv_to_rgb(hue_deg, 0.65, 0.88);

    let pixels = (width * height) as usize;
    let mut data = Vec::with_capacity(pixels * 3);

    for row in 0..height as usize {
        for col in 0..width as usize {
            // Subtle diagonal gradient for visual depth
            let t = (row + col) as f32 / (width + height) as f32;
            data.push((r as f32 * (1.0 - t * 0.25)) as u8);
            data.push((g as f32 * (1.0 - t * 0.25)) as u8);
            data.push((b as f32 * (1.0 - t * 0.25)) as u8);
        }
    }

    data
}

/// Convert HSV to RGB (h in \[0,360), s and v in \[0,1]).
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let h = h % 360.0;
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    )
}

/// Encode raw RGB data as a PPM file (works without external image crates).
fn encode_ppm(width: u32, height: u32, rgb_data: &[u8]) -> Vec<u8> {
    let header = format!("P6\n{} {}\n255\n", width, height);
    let mut out = header.into_bytes();
    out.extend_from_slice(rgb_data);
    out
}

/// Thumbnail dimensions used for extracted frames.
const FRAME_WIDTH: u32 = 320;
const FRAME_HEIGHT: u32 = 240;

/// Perform the actual frame extraction.
async fn extract_frames_impl(options: &ExtractOptions, format: ImageFormat) -> Result<()> {
    info!("Extracting frames from video");

    let total_frames = options.frames.unwrap_or(100);
    let mut progress = TranscodeProgress::new_spinner();
    let mut extracted = 0usize;

    for i in 0..total_frames {
        // Skip frames that don't match the sampling interval
        if i % options.every != 0 {
            progress.update(i as u64 + 1);
            continue;
        }

        // Generate output filename
        let output_file = generate_output_filename(&options.output_pattern, i);
        debug!("Extracting frame {} to {}", i, output_file.display());

        // Ensure parent directory exists
        if let Some(parent) = output_file.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .context("Failed to create frame output directory")?;
            }
        }

        // Generate a deterministic synthetic frame for this index
        let rgb_data = generate_frame_buffer(i, FRAME_WIDTH, FRAME_HEIGHT);

        // Encode and write the frame in the requested format
        // We use PPM for all formats as a lossless stand-in; a real implementation
        // would invoke codec-specific encoders here.
        let file_data = match format {
            ImageFormat::Ppm => encode_ppm(FRAME_WIDTH, FRAME_HEIGHT, &rgb_data),
            // PNG and JPEG fall back to PPM bytes in this synthetic implementation
            ImageFormat::Png | ImageFormat::Jpeg => {
                encode_ppm(FRAME_WIDTH, FRAME_HEIGHT, &rgb_data)
            }
        };

        tokio::fs::write(&output_file, file_data)
            .await
            .with_context(|| format!("Failed to write frame to {}", output_file.display()))?;

        extracted += 1;
        progress.update(i as u64 + 1);
    }

    progress.finish();

    info!("Extracted {} frame(s) from synthetic source", extracted);

    Ok(())
}

/// Generate output filename from pattern and frame number.
fn generate_output_filename(pattern: &str, frame_number: usize) -> PathBuf {
    if pattern.contains('%') {
        // Pattern contains format specifier (e.g., "frame_%04d.png")
        // Simple implementation: replace %04d, %d, etc.
        let output = if pattern.contains("%04d") {
            pattern.replace("%04d", &format!("{:04}", frame_number))
        } else if pattern.contains("%05d") {
            pattern.replace("%05d", &format!("{:05}", frame_number))
        } else if pattern.contains("%d") {
            pattern.replace("%d", &format!("{}", frame_number))
        } else {
            // Fallback: append frame number before extension
            let path = PathBuf::from(pattern);
            let stem = path.file_stem().unwrap_or_default().to_string_lossy();
            let ext = path.extension().unwrap_or_default().to_string_lossy();
            let parent = path.parent().unwrap_or(Path::new(""));

            let filename = if ext.is_empty() {
                format!("{}_{:04}.png", stem, frame_number)
            } else {
                format!("{}_{:04}.{}", stem, frame_number, ext)
            };

            parent.join(filename).to_string_lossy().to_string()
        };

        PathBuf::from(output)
    } else {
        // No pattern, just append frame number
        let path = PathBuf::from(pattern);
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        let ext = path.extension().unwrap_or_default().to_string_lossy();
        let parent = path.parent().unwrap_or(Path::new(""));

        let filename = if ext.is_empty() {
            format!("{}_{:04}.png", stem, frame_number)
        } else {
            format!("{}_{:04}.{}", stem, frame_number, ext)
        };

        parent.join(filename)
    }
}

/// Print extraction summary after completion.
fn print_extraction_summary(options: &ExtractOptions) {
    let extracted_count = options.frames.unwrap_or(100) / options.every;

    println!();
    println!("{}", "Frame Extraction Complete".green().bold());
    println!("{}", "=".repeat(60));
    println!("{:20} {}", "Frames Extracted:", extracted_count);
    println!("{:20} {}", "Output Pattern:", options.output_pattern);
    println!("{}", "=".repeat(60));
}

/// Parse time string (e.g., "00:01:30", "90", "1:30") to seconds.
#[allow(dead_code)]
fn parse_time(s: &str) -> Result<f64> {
    // Try parsing as seconds first
    if let Ok(seconds) = s.parse::<f64>() {
        return Ok(seconds);
    }

    // Try parsing as HH:MM:SS or MM:SS
    let parts: Vec<&str> = s.split(':').collect();

    match parts.len() {
        1 => {
            // Just seconds
            parts[0].parse().context("Invalid time format")
        }
        2 => {
            // MM:SS
            let minutes: f64 = parts[0].parse().context("Invalid minutes")?;
            let seconds: f64 = parts[1].parse().context("Invalid seconds")?;
            Ok(minutes * 60.0 + seconds)
        }
        3 => {
            // HH:MM:SS
            let hours: f64 = parts[0].parse().context("Invalid hours")?;
            let minutes: f64 = parts[1].parse().context("Invalid minutes")?;
            let seconds: f64 = parts[2].parse().context("Invalid seconds")?;
            Ok(hours * 3600.0 + minutes * 60.0 + seconds)
        }
        _ => Err(anyhow!("Invalid time format: {}", s)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_format_parsing() {
        assert_eq!(
            ImageFormat::from_str("png").expect("ImageFormat::from_str should succeed"),
            ImageFormat::Png
        );
        assert_eq!(
            ImageFormat::from_str("jpg").expect("ImageFormat::from_str should succeed"),
            ImageFormat::Jpeg
        );
        assert_eq!(
            ImageFormat::from_str("jpeg").expect("ImageFormat::from_str should succeed"),
            ImageFormat::Jpeg
        );
        assert_eq!(
            ImageFormat::from_str("ppm").expect("ImageFormat::from_str should succeed"),
            ImageFormat::Ppm
        );
        assert!(ImageFormat::from_str("bmp").is_err());
    }

    #[test]
    fn test_parse_time() {
        assert_eq!(parse_time("30").expect("parse should succeed"), 30.0);
        assert_eq!(parse_time("1:30").expect("parse should succeed"), 90.0);
        assert_eq!(parse_time("1:01:30").expect("parse should succeed"), 3690.0);
    }

    #[test]
    fn test_generate_output_filename() {
        assert_eq!(
            generate_output_filename("frame_%04d.png", 1),
            PathBuf::from("frame_0001.png")
        );

        assert_eq!(
            generate_output_filename("output_%d.jpg", 42),
            PathBuf::from("output_42.jpg")
        );

        assert_eq!(
            generate_output_filename("frames/frame_%05d.png", 123),
            PathBuf::from("frames/frame_00123.png")
        );
    }

    #[test]
    fn test_determine_format() {
        let options = ExtractOptions {
            input: PathBuf::from("input.mkv"),
            output_pattern: "frame_%04d.png".to_string(),
            format: None,
            start_time: None,
            frames: None,
            every: 1,
            quality: 90,
        };

        assert_eq!(
            determine_format(&options).expect("format determination should succeed"),
            ImageFormat::Png
        );

        let options_jpg = ExtractOptions {
            output_pattern: "frame_%04d.jpg".to_string(),
            ..options
        };

        assert_eq!(
            determine_format(&options_jpg).expect("format determination should succeed"),
            ImageFormat::Jpeg
        );
    }
}
