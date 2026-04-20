//! Validation utilities for batch processing

use crate::error::{BatchError, Result};
use crate::job::BatchJob;
use std::path::Path;

/// Validate job configuration
///
/// # Arguments
///
/// * `job` - Job to validate
///
/// # Errors
///
/// Returns an error if validation fails
pub fn validate_job(job: &BatchJob) -> Result<()> {
    // Validate job name
    if job.name.is_empty() {
        return Err(BatchError::ValidationError(
            "Job name cannot be empty".to_string(),
        ));
    }

    // Validate inputs
    if job.inputs.is_empty() {
        return Err(BatchError::ValidationError(
            "Job must have at least one input".to_string(),
        ));
    }

    // Validate input patterns
    for input in &job.inputs {
        validate_pattern(&input.pattern)?;
    }

    // Validate outputs
    for output in &job.outputs {
        validate_output_template(&output.template)?;
    }

    // Validate dependencies don't form cycles
    validate_dependencies(job)?;

    Ok(())
}

/// Validate file pattern
///
/// # Arguments
///
/// * `pattern` - File pattern to validate
///
/// # Errors
///
/// Returns an error if pattern is invalid
pub fn validate_pattern(pattern: &str) -> Result<()> {
    if pattern.is_empty() {
        return Err(BatchError::ValidationError(
            "Pattern cannot be empty".to_string(),
        ));
    }

    // Check for valid glob pattern
    glob::Pattern::new(pattern)?;

    Ok(())
}

/// Validate output template
///
/// # Arguments
///
/// * `template` - Output template to validate
///
/// # Errors
///
/// Returns an error if template is invalid
pub fn validate_output_template(template: &str) -> Result<()> {
    if template.is_empty() {
        return Err(BatchError::ValidationError(
            "Output template cannot be empty".to_string(),
        ));
    }

    // Check for balanced braces in template variables
    let mut brace_count = 0;
    for c in template.chars() {
        match c {
            '{' => brace_count += 1,
            '}' => {
                brace_count -= 1;
                if brace_count < 0 {
                    return Err(BatchError::ValidationError(
                        "Unbalanced braces in template".to_string(),
                    ));
                }
            }
            _ => {}
        }
    }

    if brace_count != 0 {
        return Err(BatchError::ValidationError(
            "Unbalanced braces in template".to_string(),
        ));
    }

    Ok(())
}

/// Validate job dependencies
///
/// # Arguments
///
/// * `job` - Job to validate
///
/// # Errors
///
/// Returns an error if dependencies form a cycle
fn validate_dependencies(job: &BatchJob) -> Result<()> {
    // Check that the job does not depend on itself
    if job.dependencies.contains(&job.id) {
        return Err(BatchError::ValidationError(format!(
            "Job '{}' depends on itself",
            job.id
        )));
    }

    // Check for duplicate dependency entries
    let mut seen = std::collections::HashSet::new();
    for dep in &job.dependencies {
        if !seen.insert(dep) {
            return Err(BatchError::ValidationError(format!(
                "Duplicate dependency '{}' in job '{}'",
                dep, job.id
            )));
        }
    }

    Ok(())
}

/// Validate file path
///
/// # Arguments
///
/// * `path` - Path to validate
///
/// # Errors
///
/// Returns an error if path is invalid
pub fn validate_path(path: &Path) -> Result<()> {
    // Check for null bytes
    let path_str = path
        .to_str()
        .ok_or_else(|| BatchError::ValidationError("Invalid path encoding".to_string()))?;

    if path_str.contains('\0') {
        return Err(BatchError::ValidationError(
            "Path contains null byte".to_string(),
        ));
    }

    // Check for relative path traversal
    if path_str.contains("..") {
        return Err(BatchError::ValidationError(
            "Path contains relative traversal".to_string(),
        ));
    }

    Ok(())
}

/// Validate bitrate value
///
/// # Arguments
///
/// * `bitrate` - Bitrate in bps
///
/// # Errors
///
/// Returns an error if bitrate is out of range
pub fn validate_bitrate(bitrate: u64) -> Result<()> {
    const MIN_BITRATE: u64 = 64_000; // 64 kbps
    const MAX_BITRATE: u64 = 500_000_000; // 500 Mbps

    if bitrate < MIN_BITRATE {
        return Err(BatchError::ValidationError(format!(
            "Bitrate too low: {bitrate} (minimum: {MIN_BITRATE})"
        )));
    }

    if bitrate > MAX_BITRATE {
        return Err(BatchError::ValidationError(format!(
            "Bitrate too high: {bitrate} (maximum: {MAX_BITRATE})"
        )));
    }

    Ok(())
}

/// Validate resolution
///
/// # Arguments
///
/// * `width` - Width in pixels
/// * `height` - Height in pixels
///
/// # Errors
///
/// Returns an error if resolution is invalid
pub fn validate_resolution(width: u32, height: u32) -> Result<()> {
    const MIN_DIMENSION: u32 = 16;
    const MAX_DIMENSION: u32 = 8192;

    if width < MIN_DIMENSION || height < MIN_DIMENSION {
        return Err(BatchError::ValidationError(format!(
            "Resolution too small: {width}x{height} (minimum: {MIN_DIMENSION}x{MIN_DIMENSION})"
        )));
    }

    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(BatchError::ValidationError(format!(
            "Resolution too large: {width}x{height} (maximum: {MAX_DIMENSION}x{MAX_DIMENSION})"
        )));
    }

    // Check if dimensions are even (required for many codecs)
    if width % 2 != 0 || height % 2 != 0 {
        return Err(BatchError::ValidationError(format!(
            "Resolution dimensions must be even: {width}x{height}"
        )));
    }

    Ok(())
}

/// Validate framerate
///
/// # Arguments
///
/// * `fps` - Frames per second
///
/// # Errors
///
/// Returns an error if framerate is invalid
pub fn validate_framerate(fps: f64) -> Result<()> {
    const MIN_FPS: f64 = 1.0;
    const MAX_FPS: f64 = 240.0;

    if fps < MIN_FPS {
        return Err(BatchError::ValidationError(format!(
            "Framerate too low: {fps} (minimum: {MIN_FPS})"
        )));
    }

    if fps > MAX_FPS {
        return Err(BatchError::ValidationError(format!(
            "Framerate too high: {fps} (maximum: {MAX_FPS})"
        )));
    }

    Ok(())
}

/// Validate email address
///
/// # Arguments
///
/// * `email` - Email address to validate
///
/// # Errors
///
/// Returns an error if email is invalid
pub fn validate_email(email: &str) -> Result<()> {
    if !email.contains('@') {
        return Err(BatchError::ValidationError(
            "Invalid email format".to_string(),
        ));
    }

    let parts: Vec<&str> = email.split('@').collect();
    if parts.len() != 2 {
        return Err(BatchError::ValidationError(
            "Invalid email format".to_string(),
        ));
    }

    if parts[0].is_empty() || parts[1].is_empty() {
        return Err(BatchError::ValidationError(
            "Invalid email format".to_string(),
        ));
    }

    Ok(())
}

/// Validate URL
///
/// # Arguments
///
/// * `url` - URL to validate
///
/// # Errors
///
/// Returns an error if URL is invalid
pub fn validate_url(url: &str) -> Result<()> {
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(BatchError::ValidationError(
            "URL must start with http:// or https://".to_string(),
        ));
    }

    if url.len() < 10 {
        return Err(BatchError::ValidationError("URL too short".to_string()));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::FileOperation;

    #[test]
    fn test_validate_pattern() {
        assert!(validate_pattern("*.mp4").is_ok());
        assert!(validate_pattern("**/*.mp4").is_ok());
        assert!(validate_pattern("").is_err());
    }

    #[test]
    fn test_validate_output_template() {
        assert!(validate_output_template("{filename}.mp4").is_ok());
        assert!(validate_output_template("output_{date}.mp4").is_ok());
        assert!(validate_output_template("{unbalanced").is_err());
        assert!(validate_output_template("unbalanced}").is_err());
    }

    #[test]
    fn test_validate_path() {
        assert!(validate_path(&std::env::temp_dir().join("oximedia-batch-file.mp4")).is_ok());
        assert!(validate_path(Path::new("/tmp/../etc/passwd")).is_err());
    }

    #[test]
    fn test_validate_bitrate() {
        assert!(validate_bitrate(5_000_000).is_ok());
        assert!(validate_bitrate(100).is_err());
        assert!(validate_bitrate(1_000_000_000).is_err());
    }

    #[test]
    fn test_validate_resolution() {
        assert!(validate_resolution(1920, 1080).is_ok());
        assert!(validate_resolution(1280, 720).is_ok());
        assert!(validate_resolution(10, 10).is_err());
        assert!(validate_resolution(10000, 10000).is_err());
        assert!(validate_resolution(1921, 1080).is_err()); // Odd width
    }

    #[test]
    fn test_validate_framerate() {
        assert!(validate_framerate(30.0).is_ok());
        assert!(validate_framerate(29.97).is_ok());
        assert!(validate_framerate(0.5).is_err());
        assert!(validate_framerate(300.0).is_err());
    }

    #[test]
    fn test_validate_email() {
        assert!(validate_email("user@example.com").is_ok());
        assert!(validate_email("invalid").is_err());
        assert!(validate_email("@example.com").is_err());
        assert!(validate_email("user@").is_err());
    }

    #[test]
    fn test_validate_url() {
        assert!(validate_url("https://example.com").is_ok());
        assert!(validate_url("http://example.com").is_ok());
        assert!(validate_url("example.com").is_err());
        assert!(validate_url("ftp://example.com").is_err());
    }

    #[test]
    fn test_validate_job() {
        let mut job = BatchJob::new(
            "test".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        // No inputs - should fail
        assert!(validate_job(&job).is_err());

        // Add input - should pass
        job.add_input(crate::job::InputSpec::new("*.mp4".to_string()));
        assert!(validate_job(&job).is_ok());
    }
}
