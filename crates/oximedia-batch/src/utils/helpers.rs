//! Helper functions for batch processing

use crate::error::Result;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Get current timestamp in seconds.
///
/// Returns 0 if the system clock is set before the Unix epoch (an anomalous
/// condition that cannot occur on any normal operating system configuration).
#[must_use]
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Get current timestamp in milliseconds.
///
/// Returns 0 if the system clock is set before the Unix epoch (an anomalous
/// condition that cannot occur on any normal operating system configuration).
#[must_use]
pub fn current_timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

/// Generate a random job ID
#[must_use]
pub fn generate_job_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Calculate percentage
#[must_use]
pub fn calculate_percentage(current: u64, total: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let result = (current as f64 / total as f64) * 100.0;
    result
}

/// Clamp value between min and max
#[must_use]
pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Join paths safely
#[must_use]
pub fn safe_path_join(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

/// Get file size
///
/// # Arguments
///
/// * `path` - File path
///
/// # Errors
///
/// Returns an error if file metadata cannot be read
pub fn get_file_size(path: &Path) -> Result<u64> {
    Ok(std::fs::metadata(path)?.len())
}

/// Check if directory is empty
///
/// # Arguments
///
/// * `path` - Directory path
///
/// # Errors
///
/// Returns an error if directory cannot be read
pub fn is_directory_empty(path: &Path) -> Result<bool> {
    Ok(std::fs::read_dir(path)?.next().is_none())
}

/// Count files in directory
///
/// # Arguments
///
/// * `path` - Directory path
/// * `recursive` - Whether to count recursively
///
/// # Errors
///
/// Returns an error if directory cannot be read
pub fn count_files(path: &Path, recursive: bool) -> Result<usize> {
    let mut count = 0;

    if recursive {
        for entry in walkdir::WalkDir::new(path) {
            let entry =
                entry.map_err(|e| crate::error::BatchError::FileOperationError(e.to_string()))?;
            if entry.path().is_file() {
                count += 1;
            }
        }
    } else {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            if entry.path().is_file() {
                count += 1;
            }
        }
    }

    Ok(count)
}

/// Calculate total size of directory
///
/// # Arguments
///
/// * `path` - Directory path
///
/// # Errors
///
/// Returns an error if directory cannot be read
pub fn directory_size(path: &Path) -> Result<u64> {
    let mut total_size = 0;

    for entry in walkdir::WalkDir::new(path) {
        let entry =
            entry.map_err(|e| crate::error::BatchError::FileOperationError(e.to_string()))?;
        if entry.path().is_file() {
            total_size += entry.metadata()?.len();
        }
    }

    Ok(total_size)
}

/// Create temporary directory
///
/// # Errors
///
/// Returns an error if directory creation fails
pub fn create_temp_dir() -> Result<PathBuf> {
    let temp_dir = std::env::temp_dir();
    let unique_name = format!("oximedia_batch_{}", generate_job_id());
    let path = temp_dir.join(unique_name);
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

/// Clean up temporary files
///
/// # Arguments
///
/// * `path` - Directory to clean
///
/// # Errors
///
/// Returns an error if cleanup fails
pub fn cleanup_temp_files(path: &Path) -> Result<()> {
    if path.exists() {
        std::fs::remove_dir_all(path)?;
    }
    Ok(())
}

/// Round to decimal places
#[must_use]
pub fn round_to_decimals(value: f64, decimals: u32) -> f64 {
    #[allow(clippy::cast_possible_wrap)]
    let multiplier = 10_f64.powi(decimals as i32);
    (value * multiplier).round() / multiplier
}

/// Format percentage
#[must_use]
pub fn format_percentage(value: f64) -> String {
    format!("{value:.1}%")
}

/// Format timestamp as ISO 8601
#[must_use]
pub fn format_timestamp_iso8601(timestamp: u64) -> String {
    #[allow(clippy::cast_possible_wrap)]
    let datetime =
        chrono::DateTime::from_timestamp(timestamp as i64, 0).unwrap_or_else(chrono::Utc::now);
    datetime.to_rfc3339()
}

/// Retry with exponential backoff
///
/// # Arguments
///
/// * `max_attempts` - Maximum number of attempts
/// * `initial_delay_ms` - Initial delay in milliseconds
/// * `f` - Function to retry
///
/// # Errors
///
/// Returns the last error if all attempts fail
pub async fn retry_with_backoff<F, Fut, T, E>(
    max_attempts: u32,
    initial_delay_ms: u64,
    mut f: F,
) -> std::result::Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = std::result::Result<T, E>>,
{
    let mut attempt = 0;
    let mut delay = initial_delay_ms;

    loop {
        match f().await {
            Ok(result) => return Ok(result),
            Err(err) => {
                attempt += 1;
                if attempt >= max_attempts {
                    return Err(err);
                }

                tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
                delay *= 2; // Exponential backoff
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_current_timestamp() {
        let ts = current_timestamp();
        assert!(ts > 0);
    }

    #[test]
    fn test_current_timestamp_millis() {
        let ts = current_timestamp_millis();
        assert!(ts > 0);
    }

    #[test]
    fn test_generate_job_id() {
        let id1 = generate_job_id();
        let id2 = generate_job_id();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_calculate_percentage() {
        assert_eq!(calculate_percentage(50, 100), 50.0);
        assert_eq!(calculate_percentage(0, 100), 0.0);
        assert_eq!(calculate_percentage(100, 100), 100.0);
        assert_eq!(calculate_percentage(0, 0), 0.0);
    }

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5, 0, 10), 5);
        assert_eq!(clamp(-5, 0, 10), 0);
        assert_eq!(clamp(15, 0, 10), 10);
    }

    #[test]
    fn test_round_to_decimals() {
        assert_eq!(round_to_decimals(3.14159, 2), 3.14);
        assert_eq!(round_to_decimals(3.14159, 3), 3.142);
    }

    #[test]
    fn test_format_percentage() {
        assert_eq!(format_percentage(50.5), "50.5%");
    }

    #[test]
    fn test_count_files() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");

        // Create some test files
        std::fs::write(temp_dir.path().join("file1.txt"), b"test").expect("failed to join");
        std::fs::write(temp_dir.path().join("file2.txt"), b"test").expect("failed to join");

        let count = count_files(temp_dir.path(), false).expect("operation should succeed");
        assert_eq!(count, 2);
    }

    #[test]
    fn test_is_directory_empty() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        assert!(is_directory_empty(temp_dir.path()).expect("operation should succeed"));

        std::fs::write(temp_dir.path().join("file.txt"), b"test").expect("failed to join");
        assert!(!is_directory_empty(temp_dir.path()).expect("operation should succeed"));
    }

    #[test]
    fn test_directory_size() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");

        std::fs::write(temp_dir.path().join("file1.txt"), b"test").expect("failed to join");
        std::fs::write(temp_dir.path().join("file2.txt"), b"test").expect("failed to join");

        let size = directory_size(temp_dir.path()).expect("operation should succeed");
        assert_eq!(size, 8);
    }

    #[test]
    fn test_create_temp_dir() {
        let temp_dir = create_temp_dir().expect("operation should succeed");
        assert!(temp_dir.exists());
        cleanup_temp_files(&temp_dir).expect("operation should succeed");
        assert!(!temp_dir.exists());
    }
}
