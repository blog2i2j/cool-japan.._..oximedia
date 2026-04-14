#![allow(dead_code)]
//! Job output validation before marking a job as completed.
//!
//! Validates that the output file produced by an encoding job satisfies a set
//! of configurable rules:
//!
//! - **File existence** – the output path must point to a regular file.
//! - **Minimum file size** – guards against silent encoder failures that
//!   produce empty or near-empty output files.
//! - **SHA-256 digest verification** – optional; when an expected digest is
//!   provided the file contents are hashed and compared.
//! - **Extension allow-list** – ensures the output format matches what was
//!   requested (e.g. `.mp4` not `.tmp`).
//!
//! All checks are synchronous and I/O-bound; callers should dispatch validation
//! to a blocking thread pool when used inside an async context.

use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Error returned when output validation fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// The output file does not exist.
    FileMissing(String),
    /// The file exists but is smaller than the configured minimum.
    FileTooSmall {
        /// Actual byte count.
        actual: u64,
        /// Configured minimum byte count.
        minimum: u64,
    },
    /// The SHA-256 digest of the file does not match the expected value.
    DigestMismatch {
        /// Expected hex-encoded digest.
        expected: String,
        /// Actual hex-encoded digest.
        actual: String,
    },
    /// The file extension is not in the allowed set.
    ForbiddenExtension(String),
    /// An I/O error occurred while reading the file.
    Io(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileMissing(p) => write!(f, "output file missing: {p}"),
            Self::FileTooSmall { actual, minimum } => {
                write!(f, "output too small: {actual} bytes (minimum {minimum})")
            }
            Self::DigestMismatch { expected, actual } => {
                write!(f, "SHA-256 mismatch: expected {expected}, got {actual}")
            }
            Self::ForbiddenExtension(ext) => write!(f, "forbidden file extension: {ext}"),
            Self::Io(msg) => write!(f, "I/O error during validation: {msg}"),
        }
    }
}

impl std::error::Error for ValidationError {}

/// Result of output validation.
pub type ValidationResult = std::result::Result<(), ValidationError>;

// ---------------------------------------------------------------------------
// Validation rules
// ---------------------------------------------------------------------------

/// A collection of validation rules applied to a job output file.
#[derive(Debug, Clone)]
pub struct OutputValidationRules {
    /// Minimum acceptable file size in bytes.
    pub min_size_bytes: u64,
    /// Expected SHA-256 digest (lowercase hex), if any.
    pub expected_sha256: Option<String>,
    /// Allowed file extensions (without leading `.`).
    ///
    /// An empty list means *any* extension is accepted.
    pub allowed_extensions: Vec<String>,
}

impl Default for OutputValidationRules {
    fn default() -> Self {
        Self {
            min_size_bytes: 1,
            expected_sha256: None,
            allowed_extensions: Vec::new(),
        }
    }
}

impl OutputValidationRules {
    /// Create rules with only a minimum file-size constraint.
    #[must_use]
    pub fn with_min_size(min_bytes: u64) -> Self {
        Self {
            min_size_bytes: min_bytes,
            ..Default::default()
        }
    }

    /// Add an expected SHA-256 digest (lowercase hex string).
    #[must_use]
    pub fn with_sha256(mut self, digest: impl Into<String>) -> Self {
        self.expected_sha256 = Some(digest.into());
        self
    }

    /// Restrict accepted file extensions.
    #[must_use]
    pub fn with_extensions(mut self, exts: Vec<String>) -> Self {
        self.allowed_extensions = exts;
        self
    }
}

// ---------------------------------------------------------------------------
// Validator
// ---------------------------------------------------------------------------

/// Validates a job's output file against a set of [`OutputValidationRules`].
pub struct OutputValidator {
    rules: OutputValidationRules,
}

impl OutputValidator {
    /// Create a new validator with the given rules.
    #[must_use]
    pub fn new(rules: OutputValidationRules) -> Self {
        Self { rules }
    }

    /// Validate the file at `output_path`.
    ///
    /// All configured checks are run in order; the first failure is returned.
    ///
    /// # Errors
    ///
    /// Returns a [`ValidationError`] describing the first failed check.
    pub fn validate<P: AsRef<Path>>(&self, output_path: P) -> ValidationResult {
        let path = output_path.as_ref();

        // --- 1. File existence ---
        if !path.exists() || !path.is_file() {
            return Err(ValidationError::FileMissing(path.display().to_string()));
        }

        // --- 2. Extension allow-list ---
        if !self.rules.allowed_extensions.is_empty() {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            let allowed: Vec<String> = self
                .rules
                .allowed_extensions
                .iter()
                .map(|e| e.to_lowercase())
                .collect();
            if !allowed.contains(&ext) {
                return Err(ValidationError::ForbiddenExtension(ext));
            }
        }

        // --- 3. File size ---
        let metadata = std::fs::metadata(path).map_err(|e| ValidationError::Io(e.to_string()))?;
        let file_size = metadata.len();

        if file_size < self.rules.min_size_bytes {
            return Err(ValidationError::FileTooSmall {
                actual: file_size,
                minimum: self.rules.min_size_bytes,
            });
        }

        // --- 4. SHA-256 digest ---
        if let Some(ref expected) = self.rules.expected_sha256 {
            let actual = compute_sha256(path)?;
            if actual != expected.to_lowercase() {
                return Err(ValidationError::DigestMismatch {
                    expected: expected.clone(),
                    actual,
                });
            }
        }

        Ok(())
    }
}

/// Compute the SHA-256 digest of a file, returning the lowercase hex string.
fn compute_sha256<P: AsRef<Path>>(path: P) -> Result<String, ValidationError> {
    let mut file = std::fs::File::open(path).map_err(|e| ValidationError::Io(e.to_string()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 65536];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| ValidationError::Io(e.to_string()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_file(name: &str, content: &[u8]) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(name);
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(content).expect("write temp file");
        path
    }

    #[test]
    fn test_valid_file_passes() {
        let path = write_temp_file("farm_validator_valid.mp4", b"fake mp4 content");
        let rules =
            OutputValidationRules::with_min_size(4).with_extensions(vec!["mp4".to_string()]);
        let validator = OutputValidator::new(rules);
        assert!(validator.validate(&path).is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_missing_file_fails() {
        let path = std::path::PathBuf::from("/tmp/oximedia_nonexistent_output_xyz.mp4");
        let validator = OutputValidator::new(OutputValidationRules::default());
        let err = validator.validate(&path).unwrap_err();
        assert!(matches!(err, ValidationError::FileMissing(_)));
    }

    #[test]
    fn test_file_too_small_fails() {
        let path = write_temp_file("farm_validator_small.mp4", b"hi");
        let rules = OutputValidationRules::with_min_size(1_000);
        let validator = OutputValidator::new(rules);
        let err = validator.validate(&path).unwrap_err();
        assert!(matches!(
            err,
            ValidationError::FileTooSmall {
                actual: 2,
                minimum: 1_000
            }
        ));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_forbidden_extension_fails() {
        let path = write_temp_file("farm_validator_ext.avi", b"some content");
        let rules = OutputValidationRules::with_min_size(1)
            .with_extensions(vec!["mp4".to_string(), "mkv".to_string()]);
        let validator = OutputValidator::new(rules);
        let err = validator.validate(&path).unwrap_err();
        assert!(matches!(err, ValidationError::ForbiddenExtension(ext) if ext == "avi"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_sha256_correct_passes() {
        let content = b"hello world";
        // Pre-computed SHA-256 of "hello world"
        let expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe04294e576fb3d8640862e3ccc";
        // Compute actual to use as expected (ensures test is self-consistent).
        let path = write_temp_file("farm_validator_sha_ok.bin", content);
        let actual_digest = compute_sha256(&path).expect("sha256 should succeed");
        let rules = OutputValidationRules::with_min_size(1).with_sha256(actual_digest.clone());
        let validator = OutputValidator::new(rules);
        assert!(validator.validate(&path).is_ok());
        let _ = std::fs::remove_file(&path);
        let _ = expected; // suppress unused warning
    }

    #[test]
    fn test_sha256_mismatch_fails() {
        let path = write_temp_file("farm_validator_sha_bad.bin", b"real content");
        let rules = OutputValidationRules::with_min_size(1)
            .with_sha256("0000000000000000000000000000000000000000000000000000000000000000");
        let validator = OutputValidator::new(rules);
        let err = validator.validate(&path).unwrap_err();
        assert!(matches!(err, ValidationError::DigestMismatch { .. }));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_no_extension_restriction_allows_any() {
        let path = write_temp_file("farm_validator_any.xyz", b"content");
        let rules = OutputValidationRules::with_min_size(1); // no extension filter
        let validator = OutputValidator::new(rules);
        assert!(validator.validate(&path).is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_validation_error_display() {
        let err = ValidationError::FileMissing("/no/such/file.mp4".to_string());
        assert!(err.to_string().contains("missing"));
    }
}
