//! File integrity checking.
//!
//! Provides CRC32-based and size-based integrity verification for media files,
//! along with a batch checker for processing multiple files at once.

/// Metadata required to verify a single file's integrity.
#[derive(Debug, Clone)]
pub struct IntegrityCheck {
    /// Path to the file being checked.
    pub file_path: String,
    /// Expected file size in bytes, if known.
    pub expected_size: Option<u64>,
    /// Expected CRC32 checksum, if known.
    pub expected_crc32: Option<u32>,
    /// Number of files processed so far (counter used externally).
    pub file_count: usize,
}

/// The result of an integrity check on a single file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrityResult {
    /// File passed all checks.
    Ok,
    /// File size does not match.
    SizeMismatch {
        /// Expected size in bytes.
        expected: u64,
        /// Actual size in bytes.
        actual: u64,
    },
    /// CRC32 checksum does not match.
    ChecksumMismatch {
        /// Expected CRC32 value.
        expected: u32,
        /// Actual CRC32 value.
        actual: u32,
    },
    /// File does not exist.
    Missing,
    /// File is shorter than expected (a special case of size mismatch).
    Truncated {
        /// Expected size in bytes.
        expected: u64,
        /// Actual size in bytes.
        actual: u64,
    },
}

impl IntegrityResult {
    /// Return `true` when the file passed all checks.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Return `true` when automated repair may be possible.
    ///
    /// `Truncated` files can potentially be recovered; `Missing` and
    /// `ChecksumMismatch` files generally cannot be repaired without the
    /// original data.
    #[must_use]
    pub fn is_repairable(&self) -> bool {
        matches!(self, Self::Truncated { .. } | Self::SizeMismatch { .. })
    }
}

/// Compute a simple CRC32 checksum of `data` using the standard polynomial.
///
/// This is a pure-Rust implementation using the standard IEEE 802.3 polynomial
/// (0xEDB88320) with the standard initial value (0xFFFF_FFFF).
#[must_use]
pub fn crc32_simple(data: &[u8]) -> u32 {
    const POLY: u32 = 0xEDB8_8320;
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFF_FFFF
}

/// Verify a single `IntegrityCheck` given a callback that returns the actual
/// file size for a path.
///
/// CRC32 verification is skipped in this function because computing a checksum
/// requires reading the actual file bytes, which is the caller's responsibility.
/// Use `BatchIntegrityChecker::verify_all` for a higher-level API.
#[must_use]
pub fn verify_check(
    check: &IntegrityCheck,
    actual_size: u64,
    file_exists: bool,
) -> IntegrityResult {
    if !file_exists {
        return IntegrityResult::Missing;
    }

    if let Some(expected) = check.expected_size {
        if actual_size < expected {
            return IntegrityResult::Truncated {
                expected,
                actual: actual_size,
            };
        }
        if actual_size != expected {
            return IntegrityResult::SizeMismatch {
                expected,
                actual: actual_size,
            };
        }
    }

    IntegrityResult::Ok
}

/// Performs integrity checks on a collection of files.
#[derive(Debug, Default)]
pub struct BatchIntegrityChecker {
    checks: Vec<IntegrityCheck>,
}

impl BatchIntegrityChecker {
    /// Create a new, empty `BatchIntegrityChecker`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a check to the batch.
    pub fn add(&mut self, check: IntegrityCheck) {
        self.checks.push(check);
    }

    /// Run all checks, using `get_size` to obtain the actual file size for each
    /// path.  Returns a `Vec` of `(file_path, IntegrityResult)` pairs.
    ///
    /// A return value of `0` from `get_size` is treated as "file missing" when
    /// `expected_size` is `Some` and > 0, or as size 0 otherwise.
    #[must_use]
    pub fn verify_all(&self, get_size: &dyn Fn(&str) -> u64) -> Vec<(String, IntegrityResult)> {
        self.checks
            .iter()
            .map(|check| {
                let actual_size = get_size(&check.file_path);
                // Treat size 0 as missing when the expected size is > 0
                let file_exists = match check.expected_size {
                    Some(expected) if expected > 0 => actual_size > 0,
                    _ => true, // unknown expected size: assume exists
                };
                let result = verify_check(check, actual_size, file_exists);
                (check.file_path.clone(), result)
            })
            .collect()
    }

    /// Count the number of failed checks in a result set.
    #[must_use]
    pub fn failure_count(results: &[(String, IntegrityResult)]) -> usize {
        results.iter().filter(|(_, r)| !r.is_ok()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_check(path: &str, expected_size: Option<u64>) -> IntegrityCheck {
        IntegrityCheck {
            file_path: path.to_string(),
            expected_size,
            expected_crc32: None,
            file_count: 0,
        }
    }

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-repair-integrity-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_integrity_result_is_ok() {
        assert!(IntegrityResult::Ok.is_ok());
        assert!(!IntegrityResult::Missing.is_ok());
    }

    #[test]
    fn test_integrity_result_is_repairable_truncated() {
        let r = IntegrityResult::Truncated {
            expected: 1000,
            actual: 500,
        };
        assert!(r.is_repairable());
    }

    #[test]
    fn test_integrity_result_is_repairable_size_mismatch() {
        let r = IntegrityResult::SizeMismatch {
            expected: 1000,
            actual: 1200,
        };
        assert!(r.is_repairable());
    }

    #[test]
    fn test_integrity_result_not_repairable_missing() {
        assert!(!IntegrityResult::Missing.is_repairable());
    }

    #[test]
    fn test_integrity_result_not_repairable_checksum() {
        let r = IntegrityResult::ChecksumMismatch {
            expected: 0xDEAD_BEEF,
            actual: 0x1234_5678,
        };
        assert!(!r.is_repairable());
    }

    #[test]
    fn test_crc32_empty() {
        // CRC32 of empty data is a well-known value: 0x00000000
        let crc = crc32_simple(&[]);
        assert_eq!(crc, 0x0000_0000);
    }

    #[test]
    fn test_crc32_known_value() {
        // CRC32 of b"123456789" is the standard check value 0xCBF43926
        let crc = crc32_simple(b"123456789");
        assert_eq!(crc, 0xCBF4_3926);
    }

    #[test]
    fn test_crc32_deterministic() {
        let data = b"hello world";
        assert_eq!(crc32_simple(data), crc32_simple(data));
    }

    #[test]
    fn test_crc32_different_for_different_data() {
        assert_ne!(crc32_simple(b"hello"), crc32_simple(b"world"));
    }

    #[test]
    fn test_verify_check_ok() {
        let check = make_check(&tmp_str("video.mp4"), Some(1000));
        let result = verify_check(&check, 1000, true);
        assert_eq!(result, IntegrityResult::Ok);
    }

    #[test]
    fn test_verify_check_missing() {
        let check = make_check(&tmp_str("missing.mp4"), Some(1000));
        let result = verify_check(&check, 0, false);
        assert_eq!(result, IntegrityResult::Missing);
    }

    #[test]
    fn test_verify_check_truncated() {
        let check = make_check(&tmp_str("truncated.mp4"), Some(1000));
        let result = verify_check(&check, 500, true);
        assert_eq!(
            result,
            IntegrityResult::Truncated {
                expected: 1000,
                actual: 500
            }
        );
    }

    #[test]
    fn test_verify_check_size_mismatch_larger() {
        let check = make_check(&tmp_str("large.mp4"), Some(1000));
        let result = verify_check(&check, 1200, true);
        assert_eq!(
            result,
            IntegrityResult::SizeMismatch {
                expected: 1000,
                actual: 1200
            }
        );
    }

    #[test]
    fn test_batch_checker_empty() {
        let checker = BatchIntegrityChecker::new();
        let results = checker.verify_all(&|_| 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_checker_all_ok() {
        let mut checker = BatchIntegrityChecker::new();
        checker.add(make_check("a.mp4", Some(100)));
        checker.add(make_check("b.mp4", Some(200)));
        let results = checker.verify_all(&|path| if path == "a.mp4" { 100 } else { 200 });
        assert_eq!(BatchIntegrityChecker::failure_count(&results), 0);
    }

    #[test]
    fn test_batch_checker_failure_count() {
        let mut checker = BatchIntegrityChecker::new();
        checker.add(make_check("good.mp4", Some(100)));
        checker.add(make_check("bad.mp4", Some(500)));
        let results = checker.verify_all(&|path| if path == "good.mp4" { 100 } else { 50 });
        assert_eq!(BatchIntegrityChecker::failure_count(&results), 1);
    }
}
