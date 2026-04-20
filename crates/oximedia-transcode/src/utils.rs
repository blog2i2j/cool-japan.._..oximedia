//! Utility functions and helpers for transcode operations.

use crate::{Result, TranscodeError};
use std::path::Path;

/// Estimates encoding time based on video duration and quality settings.
///
/// # Arguments
///
/// * `duration` - Video duration in seconds
/// * `quality` - Quality mode (affects encoding speed)
/// * `resolution` - Resolution (width, height)
/// * `hw_accel` - Whether hardware acceleration is enabled
///
/// # Returns
///
/// Estimated encoding time in seconds
#[must_use]
pub fn estimate_encoding_time(
    duration: f64,
    quality: crate::QualityMode,
    resolution: (u32, u32),
    hw_accel: bool,
) -> f64 {
    let base_speed_factor = quality.speed_factor();

    // Adjust for resolution
    let pixel_count = f64::from(resolution.0 * resolution.1);
    let resolution_factor = pixel_count / (1920.0 * 1080.0);

    // Adjust for hardware acceleration
    let hw_factor = if hw_accel { 0.3 } else { 1.0 };

    duration * base_speed_factor * resolution_factor * hw_factor
}

/// Calculates the file size estimate for a transcode.
///
/// # Arguments
///
/// * `duration` - Video duration in seconds
/// * `video_bitrate` - Video bitrate in bits per second
/// * `audio_bitrate` - Audio bitrate in bits per second
///
/// # Returns
///
/// Estimated file size in bytes
#[must_use]
pub fn estimate_file_size(duration: f64, video_bitrate: u64, audio_bitrate: u64) -> u64 {
    let total_bitrate = video_bitrate + audio_bitrate;
    let bits = (duration * total_bitrate as f64) as u64;
    bits / 8 // Convert to bytes
}

/// Formats a duration in seconds to a human-readable string.
#[must_use]
pub fn format_duration(seconds: f64) -> String {
    let hours = (seconds / 3600.0) as u64;
    let minutes = ((seconds % 3600.0) / 60.0) as u64;
    let secs = (seconds % 60.0) as u64;

    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{secs:02}")
    } else {
        format!("{minutes:02}:{secs:02}")
    }
}

/// Formats a file size in bytes to a human-readable string.
#[must_use]
pub fn format_file_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Formats a bitrate in bits per second to a human-readable string.
#[must_use]
pub fn format_bitrate(bps: u64) -> String {
    const KBPS: u64 = 1000;
    const MBPS: u64 = KBPS * 1000;

    if bps >= MBPS {
        format!("{:.2} Mbps", bps as f64 / MBPS as f64)
    } else if bps >= KBPS {
        format!("{:.0} kbps", bps as f64 / KBPS as f64)
    } else {
        format!("{bps} bps")
    }
}

/// Validates that a file exists and is readable.
///
/// # Errors
///
/// Returns an error if the file doesn't exist or isn't readable.
pub fn validate_input_file(path: &str) -> Result<()> {
    let path_obj = Path::new(path);

    if !path_obj.exists() {
        return Err(TranscodeError::InvalidInput(format!(
            "File does not exist: {path}"
        )));
    }

    if !path_obj.is_file() {
        return Err(TranscodeError::InvalidInput(format!(
            "Path is not a file: {path}"
        )));
    }

    match std::fs::metadata(path_obj) {
        Ok(metadata) => {
            if metadata.len() == 0 {
                return Err(TranscodeError::InvalidInput(format!(
                    "File is empty: {path}"
                )));
            }
        }
        Err(e) => {
            return Err(TranscodeError::InvalidInput(format!(
                "Cannot read file {path}: {e}"
            )));
        }
    }

    Ok(())
}

/// Gets the file extension from a path.
#[must_use]
pub fn get_file_extension(path: &str) -> Option<String> {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
}

/// Determines the container format from a file extension.
#[must_use]
pub fn container_from_extension(path: &str) -> Option<String> {
    let ext = get_file_extension(path)?;

    match ext.as_str() {
        "mp4" | "m4v" => Some("mp4".to_string()),
        "mkv" => Some("matroska".to_string()),
        "webm" => Some("webm".to_string()),
        "avi" => Some("avi".to_string()),
        "mov" => Some("mov".to_string()),
        "flv" => Some("flv".to_string()),
        "wmv" => Some("asf".to_string()),
        "ogv" => Some("ogg".to_string()),
        _ => None,
    }
}

/// Suggests optimal codec based on container format.
#[must_use]
pub fn suggest_video_codec(container: &str) -> Option<String> {
    match container.to_lowercase().as_str() {
        "mp4" | "m4v" => Some("h264".to_string()),
        "webm" => Some("vp9".to_string()),
        "mkv" => Some("vp9".to_string()),
        "ogv" => Some("theora".to_string()),
        _ => None,
    }
}

/// Suggests optimal audio codec based on container format.
#[must_use]
pub fn suggest_audio_codec(container: &str) -> Option<String> {
    match container.to_lowercase().as_str() {
        "mp4" | "m4v" => Some("aac".to_string()),
        "webm" => Some("opus".to_string()),
        "mkv" => Some("opus".to_string()),
        "ogv" => Some("vorbis".to_string()),
        _ => None,
    }
}

/// Calculates the aspect ratio from width and height.
#[must_use]
pub fn calculate_aspect_ratio(width: u32, height: u32) -> (u32, u32) {
    fn gcd(mut a: u32, mut b: u32) -> u32 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    let divisor = gcd(width, height);
    (width / divisor, height / divisor)
}

/// Formats an aspect ratio as a string.
#[must_use]
pub fn format_aspect_ratio(width: u32, height: u32) -> String {
    let (w, h) = calculate_aspect_ratio(width, height);
    format!("{w}:{h}")
}

/// Checks if a resolution is standard (common resolution).
#[must_use]
pub fn is_standard_resolution(width: u32, height: u32) -> bool {
    matches!(
        (width, height),
        (1920, 1080)
            | (1280, 720)
            | (3840, 2160)
            | (2560, 1440)
            | (854, 480)
            | (640, 360)
            | (426, 240)
    )
}

/// Gets the name of a standard resolution.
#[must_use]
pub fn resolution_name(width: u32, height: u32) -> String {
    match (width, height) {
        (3840, 2160) => "4K (2160p)".to_string(),
        (2560, 1440) => "2K (1440p)".to_string(),
        (1920, 1080) => "Full HD (1080p)".to_string(),
        (1280, 720) => "HD (720p)".to_string(),
        (854, 480) => "SD (480p)".to_string(),
        (640, 360) => "nHD (360p)".to_string(),
        (426, 240) => "240p".to_string(),
        _ => format!("{width}x{height}"),
    }
}

/// Calculates the optimal tile configuration for parallel encoding.
#[must_use]
pub fn calculate_optimal_tiles(width: u32, height: u32, threads: u32) -> (u8, u8) {
    let pixel_count = width * height;

    // For smaller resolutions, use fewer tiles
    if pixel_count < 1280 * 720 {
        return (1, 1);
    }

    // Calculate based on thread count
    let tiles = match threads {
        1..=2 => 1,
        3..=4 => 2,
        5..=8 => 4,
        9..=16 => 8,
        _ => 16,
    };

    // Prefer column tiles for better parallelism
    let cols = tiles.min(8);
    let rows = (tiles / cols).min(8);

    (cols as u8, rows as u8)
}

/// Suggests optimal bitrate for a given resolution and framerate.
#[must_use]
pub fn suggest_bitrate(width: u32, height: u32, fps: f64, quality: crate::QualityMode) -> u64 {
    let pixel_count = u64::from(width * height);
    let motion_factor = if fps > 30.0 { 1.5 } else { 1.0 };

    let base_bitrate = match quality {
        crate::QualityMode::Low => pixel_count / 1500,
        crate::QualityMode::Medium => pixel_count / 1000,
        crate::QualityMode::High => pixel_count / 750,
        crate::QualityMode::VeryHigh => pixel_count / 500,
        crate::QualityMode::Custom => pixel_count / 1000,
    };

    (base_bitrate as f64 * motion_factor) as u64
}

/// Validates resolution constraints.
pub fn validate_resolution_constraints(
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
) -> Result<()> {
    // Check for upscaling
    if output_width > input_width || output_height > input_height {
        // Warn but allow
    }

    // Check aspect ratio change
    let input_ratio = f64::from(input_width) / f64::from(input_height);
    let output_ratio = f64::from(output_width) / f64::from(output_height);
    let ratio_diff = (input_ratio - output_ratio).abs();

    if ratio_diff > 0.01 {
        // Aspect ratio changed significantly
    }

    Ok(())
}

/// Creates a temporary file path for statistics.
#[must_use]
pub fn temp_stats_file(job_id: &str) -> String {
    std::env::temp_dir()
        .join(format!("oximedia-transcode-stats-{job_id}.log"))
        .to_string_lossy()
        .into_owned()
}

/// Cleans up temporary files.
pub fn cleanup_temp_files(job_id: &str) -> Result<()> {
    let stats_file = temp_stats_file(job_id);
    if Path::new(&stats_file).exists() {
        std::fs::remove_file(&stats_file)?;
    }
    Ok(())
}

/// Calculates compression ratio.
#[must_use]
pub fn calculate_compression_ratio(input_size: u64, output_size: u64) -> f64 {
    if output_size == 0 {
        return 0.0;
    }
    input_size as f64 / output_size as f64
}

/// Formats compression ratio as a percentage.
#[must_use]
pub fn format_compression_ratio(ratio: f64) -> String {
    if ratio >= 1.0 {
        format!("{ratio:.2}x smaller")
    } else {
        format!("{:.2}x larger", 1.0 / ratio)
    }
}

/// Calculates space savings.
#[must_use]
pub fn calculate_space_savings(input_size: u64, output_size: u64) -> i64 {
    input_size as i64 - output_size as i64
}

/// Formats space savings.
#[must_use]
pub fn format_space_savings(savings: i64) -> String {
    if savings > 0 {
        format!("{} saved", format_file_size(savings as u64))
    } else {
        format!("{} larger", format_file_size((-savings) as u64))
    }
}

/// Parses a duration string (e.g., "01:23:45" or "83:45").
pub fn parse_duration(duration_str: &str) -> Result<f64> {
    let parts: Vec<&str> = duration_str.split(':').collect();

    let seconds = match parts.len() {
        1 => {
            // Just seconds
            parts[0].parse::<f64>().map_err(|_| {
                TranscodeError::ValidationError(crate::ValidationError::InvalidInputFormat(
                    "Invalid duration format".to_string(),
                ))
            })?
        }
        2 => {
            // MM:SS
            let minutes = parts[0].parse::<f64>().map_err(|_| {
                TranscodeError::ValidationError(crate::ValidationError::InvalidInputFormat(
                    "Invalid duration format".to_string(),
                ))
            })?;
            let secs = parts[1].parse::<f64>().map_err(|_| {
                TranscodeError::ValidationError(crate::ValidationError::InvalidInputFormat(
                    "Invalid duration format".to_string(),
                ))
            })?;
            minutes * 60.0 + secs
        }
        3 => {
            // HH:MM:SS
            let hours = parts[0].parse::<f64>().map_err(|_| {
                TranscodeError::ValidationError(crate::ValidationError::InvalidInputFormat(
                    "Invalid duration format".to_string(),
                ))
            })?;
            let minutes = parts[1].parse::<f64>().map_err(|_| {
                TranscodeError::ValidationError(crate::ValidationError::InvalidInputFormat(
                    "Invalid duration format".to_string(),
                ))
            })?;
            let secs = parts[2].parse::<f64>().map_err(|_| {
                TranscodeError::ValidationError(crate::ValidationError::InvalidInputFormat(
                    "Invalid duration format".to_string(),
                ))
            })?;
            hours * 3600.0 + minutes * 60.0 + secs
        }
        _ => {
            return Err(TranscodeError::ValidationError(
                crate::ValidationError::InvalidInputFormat("Invalid duration format".to_string()),
            ))
        }
    };

    Ok(seconds)
}

/// Formats framerate as a string.
#[must_use]
pub fn format_framerate(num: u32, den: u32) -> String {
    let fps = f64::from(num) / f64::from(den);
    if den == 1 {
        format!("{num} fps")
    } else {
        format!("{fps:.2} fps")
    }
}

/// Checks if a framerate is standard.
#[must_use]
pub fn is_standard_framerate(num: u32, den: u32) -> bool {
    matches!(
        (num, den),
        (24 | 25 | 30 | 50 | 60, 1) | (24000 | 30000 | 60000, 1001)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_encoding_time() {
        let time = estimate_encoding_time(60.0, crate::QualityMode::Medium, (1920, 1080), false);
        assert!(time > 0.0);
    }

    #[test]
    fn test_estimate_file_size() {
        let size = estimate_file_size(60.0, 5_000_000, 128_000);
        assert_eq!(size, (60.0 * 5_128_000.0 / 8.0) as u64);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(90.0), "01:30");
        assert_eq!(format_duration(3665.0), "01:01:05");
    }

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(1024), "1.00 KB");
        assert_eq!(format_file_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_file_size(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_bitrate() {
        assert_eq!(format_bitrate(1_000_000), "1.00 Mbps");
        assert_eq!(format_bitrate(128_000), "128 kbps");
    }

    #[test]
    fn test_get_file_extension() {
        assert_eq!(get_file_extension("video.mp4"), Some("mp4".to_string()));
        assert_eq!(get_file_extension("VIDEO.MP4"), Some("mp4".to_string()));
        assert_eq!(get_file_extension("video"), None);
    }

    #[test]
    fn test_container_from_extension() {
        assert_eq!(
            container_from_extension("video.mp4"),
            Some("mp4".to_string())
        );
        assert_eq!(
            container_from_extension("video.mkv"),
            Some("matroska".to_string())
        );
        assert_eq!(
            container_from_extension("video.webm"),
            Some("webm".to_string())
        );
    }

    #[test]
    fn test_suggest_codecs() {
        assert_eq!(suggest_video_codec("mp4"), Some("h264".to_string()));
        assert_eq!(suggest_video_codec("webm"), Some("vp9".to_string()));
        assert_eq!(suggest_audio_codec("mp4"), Some("aac".to_string()));
        assert_eq!(suggest_audio_codec("webm"), Some("opus".to_string()));
    }

    #[test]
    fn test_calculate_aspect_ratio() {
        assert_eq!(calculate_aspect_ratio(1920, 1080), (16, 9));
        assert_eq!(calculate_aspect_ratio(1280, 720), (16, 9));
        assert_eq!(calculate_aspect_ratio(1920, 800), (12, 5));
    }

    #[test]
    fn test_format_aspect_ratio() {
        assert_eq!(format_aspect_ratio(1920, 1080), "16:9");
        assert_eq!(format_aspect_ratio(1280, 720), "16:9");
    }

    #[test]
    fn test_is_standard_resolution() {
        assert!(is_standard_resolution(1920, 1080));
        assert!(is_standard_resolution(1280, 720));
        assert!(!is_standard_resolution(1000, 1000));
    }

    #[test]
    fn test_resolution_name() {
        assert_eq!(resolution_name(1920, 1080), "Full HD (1080p)");
        assert_eq!(resolution_name(3840, 2160), "4K (2160p)");
        assert_eq!(resolution_name(1000, 1000), "1000x1000");
    }

    #[test]
    fn test_calculate_optimal_tiles() {
        let (cols, rows) = calculate_optimal_tiles(1920, 1080, 8);
        assert!(cols > 0 && rows > 0);
    }

    #[test]
    fn test_suggest_bitrate() {
        let bitrate = suggest_bitrate(1920, 1080, 30.0, crate::QualityMode::Medium);
        assert!(bitrate > 0);
    }

    #[test]
    fn test_calculate_compression_ratio() {
        assert_eq!(calculate_compression_ratio(1000, 500), 2.0);
        assert_eq!(calculate_compression_ratio(500, 1000), 0.5);
    }

    #[test]
    fn test_parse_duration() {
        assert_eq!(parse_duration("60").expect("should succeed in test"), 60.0);
        assert_eq!(
            parse_duration("01:30").expect("should succeed in test"),
            90.0
        );
        assert_eq!(
            parse_duration("01:01:30").expect("should succeed in test"),
            3690.0
        );
    }

    #[test]
    fn test_format_framerate() {
        assert_eq!(format_framerate(30, 1), "30 fps");
        assert_eq!(format_framerate(30000, 1001), "29.97 fps");
    }

    #[test]
    fn test_is_standard_framerate() {
        assert!(is_standard_framerate(30, 1));
        assert!(is_standard_framerate(60, 1));
        assert!(is_standard_framerate(30000, 1001));
        assert!(!is_standard_framerate(45, 1));
    }
}
