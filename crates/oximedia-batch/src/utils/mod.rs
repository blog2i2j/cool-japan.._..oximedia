//! Utility functions and helpers for batch processing

pub mod converters;
pub mod helpers;
pub mod validators;

use crate::error::{BatchError, Result};
use std::path::{Path, PathBuf};

/// File size units
#[derive(Debug, Clone, Copy)]
pub enum SizeUnit {
    /// Bytes
    Bytes,
    /// Kilobytes
    Kilobytes,
    /// Megabytes
    Megabytes,
    /// Gigabytes
    Gigabytes,
    /// Terabytes
    Terabytes,
}

/// Format file size in human-readable format
///
/// # Arguments
///
/// * `size_bytes` - Size in bytes
///
/// # Examples
///
/// ```
/// use oximedia_batch::utils::format_file_size;
///
/// let formatted = format_file_size(1024);
/// assert_eq!(formatted, "1.00 KB");
/// ```
#[must_use]
pub fn format_file_size(size_bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    #[allow(clippy::cast_precision_loss)]
    if size_bytes >= TB {
        format!("{:.2} TB", size_bytes as f64 / TB as f64)
    } else if size_bytes >= GB {
        format!("{:.2} GB", size_bytes as f64 / GB as f64)
    } else if size_bytes >= MB {
        format!("{:.2} MB", size_bytes as f64 / MB as f64)
    } else if size_bytes >= KB {
        format!("{:.2} KB", size_bytes as f64 / KB as f64)
    } else {
        format!("{size_bytes} B")
    }
}

/// Parse file size string to bytes
///
/// # Arguments
///
/// * `size_str` - Size string (e.g., "10MB", "1.5GB")
///
/// # Errors
///
/// Returns an error if the string cannot be parsed
///
/// # Examples
///
/// ```
/// use oximedia_batch::utils::parse_file_size;
///
/// let bytes = parse_file_size("10MB")?;
/// assert_eq!(bytes, 10485760);
/// ```
pub fn parse_file_size(size_str: &str) -> Result<u64> {
    let size_str = size_str.trim().to_uppercase();

    let (num_str, unit) = if size_str.ends_with("TB") {
        (size_str.trim_end_matches("TB"), SizeUnit::Terabytes)
    } else if size_str.ends_with("GB") {
        (size_str.trim_end_matches("GB"), SizeUnit::Gigabytes)
    } else if size_str.ends_with("MB") {
        (size_str.trim_end_matches("MB"), SizeUnit::Megabytes)
    } else if size_str.ends_with("KB") {
        (size_str.trim_end_matches("KB"), SizeUnit::Kilobytes)
    } else if size_str.ends_with('B') {
        (size_str.trim_end_matches('B'), SizeUnit::Bytes)
    } else {
        (size_str.as_str(), SizeUnit::Bytes)
    };

    let value: f64 = num_str
        .trim()
        .parse()
        .map_err(|e| BatchError::ValidationError(format!("Invalid size value: {e}")))?;

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let bytes = match unit {
        SizeUnit::Bytes => value as u64,
        SizeUnit::Kilobytes => (value * 1024.0) as u64,
        SizeUnit::Megabytes => (value * 1_048_576.0) as u64,
        SizeUnit::Gigabytes => (value * 1_073_741_824.0) as u64,
        SizeUnit::Terabytes => (value * 1_099_511_627_776.0) as u64,
    };

    Ok(bytes)
}

/// Format duration in human-readable format
///
/// # Arguments
///
/// * `seconds` - Duration in seconds
///
/// # Examples
///
/// ```
/// use oximedia_batch::utils::format_duration;
///
/// let formatted = format_duration(3665);
/// assert_eq!(formatted, "1h 1m 5s");
/// ```
#[must_use]
pub fn format_duration(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;

    if hours > 0 {
        format!("{hours}h {minutes}m {secs}s")
    } else if minutes > 0 {
        format!("{minutes}m {secs}s")
    } else {
        format!("{secs}s")
    }
}

/// Parse duration string to seconds
///
/// # Arguments
///
/// * `duration_str` - Duration string (e.g., "1h30m", "90s")
///
/// # Errors
///
/// Returns an error if the string cannot be parsed
pub fn parse_duration(duration_str: &str) -> Result<u64> {
    let mut total_seconds = 0u64;
    let mut current_num = String::new();

    for c in duration_str.chars() {
        if c.is_ascii_digit() {
            current_num.push(c);
        } else if c == 'h' || c == 'H' {
            let hours: u64 = current_num
                .parse()
                .map_err(|e| BatchError::ValidationError(format!("Invalid hours: {e}")))?;
            total_seconds += hours * 3600;
            current_num.clear();
        } else if c == 'm' || c == 'M' {
            let minutes: u64 = current_num
                .parse()
                .map_err(|e| BatchError::ValidationError(format!("Invalid minutes: {e}")))?;
            total_seconds += minutes * 60;
            current_num.clear();
        } else if c == 's' || c == 'S' {
            let seconds: u64 = current_num
                .parse()
                .map_err(|e| BatchError::ValidationError(format!("Invalid seconds: {e}")))?;
            total_seconds += seconds;
            current_num.clear();
        }
    }

    // Handle trailing number without unit (assume seconds)
    if !current_num.is_empty() {
        let seconds: u64 = current_num
            .parse()
            .map_err(|e| BatchError::ValidationError(format!("Invalid duration: {e}")))?;
        total_seconds += seconds;
    }

    Ok(total_seconds)
}

/// Sanitize filename for safe filesystem usage
///
/// # Arguments
///
/// * `filename` - Original filename
///
/// # Examples
///
/// ```
/// use oximedia_batch::utils::sanitize_filename;
///
/// let safe = sanitize_filename("file:name?.txt");
/// assert_eq!(safe, "file_name_.txt");
/// ```
#[must_use]
pub fn sanitize_filename(filename: &str) -> String {
    filename
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Ensure directory exists, create if not
///
/// # Arguments
///
/// * `path` - Directory path
///
/// # Errors
///
/// Returns an error if directory creation fails
pub fn ensure_directory(path: &Path) -> Result<()> {
    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }
    Ok(())
}

/// Get file extension
///
/// # Arguments
///
/// * `path` - File path
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use oximedia_batch::utils::get_extension;
///
/// let ext = get_extension(Path::new("file.mp4"));
/// assert_eq!(ext, Some("mp4".to_string()));
/// ```
#[must_use]
pub fn get_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_lowercase)
}

/// Check if file is a video file
///
/// # Arguments
///
/// * `path` - File path
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use oximedia_batch::utils::is_video_file;
///
/// assert!(is_video_file(Path::new("video.mp4")));
/// assert!(!is_video_file(Path::new("audio.mp3")));
/// ```
#[must_use]
pub fn is_video_file(path: &Path) -> bool {
    const VIDEO_EXTENSIONS: &[&str] = &[
        "mp4", "mov", "avi", "mkv", "mxf", "ts", "m2ts", "webm", "flv", "wmv", "mpg", "mpeg",
    ];

    get_extension(path).is_some_and(|ext| VIDEO_EXTENSIONS.contains(&ext.as_str()))
}

/// Check if file is an audio file
///
/// # Arguments
///
/// * `path` - File path
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use oximedia_batch::utils::is_audio_file;
///
/// assert!(is_audio_file(Path::new("audio.mp3")));
/// assert!(!is_audio_file(Path::new("video.mp4")));
/// ```
#[must_use]
pub fn is_audio_file(path: &Path) -> bool {
    const AUDIO_EXTENSIONS: &[&str] = &[
        "mp3", "wav", "flac", "aac", "m4a", "ogg", "wma", "aiff", "opus",
    ];

    get_extension(path).is_some_and(|ext| AUDIO_EXTENSIONS.contains(&ext.as_str()))
}

/// Check if file is an image file
///
/// # Arguments
///
/// * `path` - File path
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use oximedia_batch::utils::is_image_file;
///
/// assert!(is_image_file(Path::new("photo.jpg")));
/// assert!(!is_image_file(Path::new("video.mp4")));
/// ```
#[must_use]
pub fn is_image_file(path: &Path) -> bool {
    const IMAGE_EXTENSIONS: &[&str] = &[
        "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp", "svg", "dpx", "exr",
    ];

    get_extension(path).is_some_and(|ext| IMAGE_EXTENSIONS.contains(&ext.as_str()))
}

/// Generate unique filename by appending number if file exists
///
/// # Arguments
///
/// * `path` - Original file path
///
/// # Examples
///
/// ```
/// use std::path::PathBuf;
/// use oximedia_batch::utils::make_unique_filename;
///
/// let unique = make_unique_filename(&PathBuf::from("/tmp/file.txt"));
/// // If /tmp/file.txt exists, returns /tmp/file_1.txt, etc.
/// ```
#[must_use]
pub fn make_unique_filename(path: &Path) -> PathBuf {
    if !path.exists() {
        return path.to_path_buf();
    }

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    let parent = path.parent().unwrap_or_else(|| Path::new(""));

    for i in 1..1000 {
        let new_name = if extension.is_empty() {
            format!("{stem}_{i}")
        } else {
            format!("{stem}_{i}.{extension}")
        };

        let new_path = parent.join(new_name);
        if !new_path.exists() {
            return new_path;
        }
    }

    path.to_path_buf()
}

/// Calculate file hash
///
/// # Arguments
///
/// * `path` - File path
///
/// # Errors
///
/// Returns an error if file reading fails
pub fn calculate_file_hash(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 8192];

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(1024), "1.00 KB");
        assert_eq!(format_file_size(1048576), "1.00 MB");
        assert_eq!(format_file_size(1073741824), "1.00 GB");
    }

    #[test]
    fn test_parse_file_size() {
        assert_eq!(
            parse_file_size("10MB").expect("operation should succeed"),
            10485760
        );
        assert_eq!(
            parse_file_size("1GB").expect("operation should succeed"),
            1073741824
        );
        assert_eq!(
            parse_file_size("1024").expect("operation should succeed"),
            1024
        );
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(3665), "1h 1m 5s");
        assert_eq!(format_duration(90), "1m 30s");
        assert_eq!(format_duration(45), "45s");
    }

    #[test]
    fn test_parse_duration() {
        assert_eq!(
            parse_duration("1h30m").expect("operation should succeed"),
            5400
        );
        assert_eq!(parse_duration("90s").expect("operation should succeed"), 90);
        assert_eq!(
            parse_duration("1h").expect("operation should succeed"),
            3600
        );
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("file:name?.txt"), "file_name_.txt");
        assert_eq!(sanitize_filename("normal.txt"), "normal.txt");
        assert_eq!(sanitize_filename("my-file_123.mp4"), "my-file_123.mp4");
    }

    #[test]
    fn test_is_video_file() {
        assert!(is_video_file(Path::new("video.mp4")));
        assert!(is_video_file(Path::new("movie.mkv")));
        assert!(!is_video_file(Path::new("audio.mp3")));
    }

    #[test]
    fn test_is_audio_file() {
        assert!(is_audio_file(Path::new("audio.mp3")));
        assert!(is_audio_file(Path::new("song.wav")));
        assert!(!is_audio_file(Path::new("video.mp4")));
    }

    #[test]
    fn test_is_image_file() {
        assert!(is_image_file(Path::new("photo.jpg")));
        assert!(is_image_file(Path::new("image.png")));
        assert!(!is_image_file(Path::new("video.mp4")));
    }

    #[test]
    fn test_get_extension() {
        assert_eq!(
            get_extension(Path::new("file.mp4")),
            Some("mp4".to_string())
        );
        assert_eq!(
            get_extension(Path::new("FILE.MP4")),
            Some("mp4".to_string())
        );
        assert_eq!(get_extension(Path::new("noext")), None);
    }
}
