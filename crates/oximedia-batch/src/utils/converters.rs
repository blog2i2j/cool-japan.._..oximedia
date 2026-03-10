//! Conversion utilities for batch processing

use crate::error::{BatchError, Result};
use crate::types::{JobState, Priority};
use std::path::{Path, PathBuf};

/// Convert job state to string
#[must_use]
pub fn job_state_to_string(state: JobState) -> String {
    match state {
        JobState::Queued => "queued".to_string(),
        JobState::Running => "running".to_string(),
        JobState::Completed => "completed".to_string(),
        JobState::Failed => "failed".to_string(),
        JobState::Cancelled => "cancelled".to_string(),
        JobState::Pending => "pending".to_string(),
    }
}

/// Convert string to job state
///
/// # Arguments
///
/// * `s` - String representation
///
/// # Errors
///
/// Returns an error if string is invalid
pub fn string_to_job_state(s: &str) -> Result<JobState> {
    match s.to_lowercase().as_str() {
        "queued" => Ok(JobState::Queued),
        "running" => Ok(JobState::Running),
        "completed" => Ok(JobState::Completed),
        "failed" => Ok(JobState::Failed),
        "cancelled" => Ok(JobState::Cancelled),
        "pending" => Ok(JobState::Pending),
        _ => Err(BatchError::ValidationError(format!(
            "Invalid job state: {s}"
        ))),
    }
}

/// Convert priority to string
#[must_use]
pub fn priority_to_string(priority: Priority) -> String {
    match priority {
        Priority::Low => "low".to_string(),
        Priority::Normal => "normal".to_string(),
        Priority::High => "high".to_string(),
    }
}

/// Convert string to priority
///
/// # Arguments
///
/// * `s` - String representation
///
/// # Errors
///
/// Returns an error if string is invalid
pub fn string_to_priority(s: &str) -> Result<Priority> {
    match s.to_lowercase().as_str() {
        "low" => Ok(Priority::Low),
        "normal" => Ok(Priority::Normal),
        "high" => Ok(Priority::High),
        _ => Err(BatchError::ValidationError(format!(
            "Invalid priority: {s}"
        ))),
    }
}

/// Convert absolute path to relative path
///
/// # Arguments
///
/// * `path` - Absolute path
/// * `base` - Base directory
///
/// # Errors
///
/// Returns an error if conversion fails
pub fn to_relative_path(path: &Path, base: &Path) -> Result<PathBuf> {
    path.strip_prefix(base)
        .map(std::path::Path::to_path_buf)
        .map_err(|e| BatchError::FileOperationError(e.to_string()))
}

/// Convert relative path to absolute path
///
/// # Arguments
///
/// * `path` - Relative path
/// * `base` - Base directory
#[must_use]
pub fn to_absolute_path(path: &Path, base: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

/// Convert bitrate string to bps
///
/// # Arguments
///
/// * `bitrate_str` - Bitrate string (e.g., "5000k", "2M")
///
/// # Errors
///
/// Returns an error if parsing fails
pub fn parse_bitrate(bitrate_str: &str) -> Result<u64> {
    let bitrate_str = bitrate_str.trim().to_uppercase();

    let (num_str, multiplier) = if bitrate_str.ends_with('M') {
        (bitrate_str.trim_end_matches('M'), 1_000_000)
    } else if bitrate_str.ends_with('K') {
        (bitrate_str.trim_end_matches('K'), 1_000)
    } else {
        (bitrate_str.as_str(), 1)
    };

    let value: u64 = num_str
        .parse()
        .map_err(|e| BatchError::ValidationError(format!("Invalid bitrate: {e}")))?;

    Ok(value * multiplier)
}

/// Format bitrate to human-readable string
///
/// # Arguments
///
/// * `bps` - Bitrate in bits per second
#[must_use]
pub fn format_bitrate(bps: u64) -> String {
    #[allow(clippy::cast_precision_loss)]
    if bps >= 1_000_000 {
        format!("{:.1}M", bps as f64 / 1_000_000.0)
    } else if bps >= 1_000 {
        format!("{}k", bps / 1_000)
    } else {
        format!("{bps}")
    }
}

/// Convert resolution string to dimensions
///
/// # Arguments
///
/// * `resolution_str` - Resolution string (e.g., "1920x1080", "1280x720")
///
/// # Errors
///
/// Returns an error if parsing fails
pub fn parse_resolution(resolution_str: &str) -> Result<(u32, u32)> {
    let parts: Vec<&str> = resolution_str.split('x').collect();

    if parts.len() != 2 {
        return Err(BatchError::ValidationError(format!(
            "Invalid resolution format: {resolution_str}"
        )));
    }

    let width: u32 = parts[0]
        .trim()
        .parse()
        .map_err(|e| BatchError::ValidationError(format!("Invalid width: {e}")))?;

    let height: u32 = parts[1]
        .trim()
        .parse()
        .map_err(|e| BatchError::ValidationError(format!("Invalid height: {e}")))?;

    Ok((width, height))
}

/// Format resolution as string
///
/// # Arguments
///
/// * `width` - Width in pixels
/// * `height` - Height in pixels
#[must_use]
pub fn format_resolution(width: u32, height: u32) -> String {
    format!("{width}x{height}")
}

/// Convert framerate string to fps
///
/// # Arguments
///
/// * `framerate_str` - Framerate string (e.g., "30", "29.97", "23.976")
///
/// # Errors
///
/// Returns an error if parsing fails
pub fn parse_framerate(framerate_str: &str) -> Result<f64> {
    framerate_str
        .trim()
        .parse()
        .map_err(|e| BatchError::ValidationError(format!("Invalid framerate: {e}")))
}

/// Format framerate as string
///
/// # Arguments
///
/// * `fps` - Frames per second
#[must_use]
pub fn format_framerate(fps: f64) -> String {
    if (fps - fps.round()).abs() < 0.001 {
        format!("{fps:.0}")
    } else {
        format!("{fps:.3}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_state_conversion() {
        let state = JobState::Running;
        let state_str = job_state_to_string(state);
        assert_eq!(state_str, "running");

        let parsed = string_to_job_state(&state_str).expect("operation should succeed");
        assert_eq!(parsed, state);
    }

    #[test]
    fn test_priority_conversion() {
        let priority = Priority::High;
        let priority_str = priority_to_string(priority);
        assert_eq!(priority_str, "high");

        let parsed = string_to_priority(&priority_str).expect("operation should succeed");
        assert_eq!(parsed, priority);
    }

    #[test]
    fn test_parse_bitrate() {
        assert_eq!(
            parse_bitrate("5000k").expect("operation should succeed"),
            5_000_000
        );
        assert_eq!(
            parse_bitrate("2M").expect("operation should succeed"),
            2_000_000
        );
        assert_eq!(
            parse_bitrate("1000").expect("operation should succeed"),
            1000
        );
    }

    #[test]
    fn test_format_bitrate() {
        assert_eq!(format_bitrate(5_000_000), "5.0M");
        assert_eq!(format_bitrate(2_000), "2k");
        assert_eq!(format_bitrate(500), "500");
    }

    #[test]
    fn test_parse_resolution() {
        let (width, height) = parse_resolution("1920x1080").expect("operation should succeed");
        assert_eq!(width, 1920);
        assert_eq!(height, 1080);
    }

    #[test]
    fn test_format_resolution() {
        assert_eq!(format_resolution(1920, 1080), "1920x1080");
    }

    #[test]
    fn test_parse_framerate() {
        assert_eq!(
            parse_framerate("30").expect("operation should succeed"),
            30.0
        );
        assert_eq!(
            parse_framerate("29.97").expect("operation should succeed"),
            29.97
        );
    }

    #[test]
    fn test_format_framerate() {
        assert_eq!(format_framerate(30.0), "30");
        assert_eq!(format_framerate(29.97), "29.970");
    }
}
