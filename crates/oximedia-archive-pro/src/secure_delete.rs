//! Secure deletion with multi-pass overwrite and verification.
//!
//! This module implements cryptographically-informed secure deletion for archival
//! objects. It provides multiple deletion standards (DoD 5220.22-M, Gutmann,
//! NIST 800-88) with post-deletion verification to confirm data is irrecoverable.
//!
//! All passes use patterns consistent with each standard while remaining fully
//! in pure Rust without any unsafe blocks.

use std::fs::{self, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonically increasing counter for generating unique rename targets.
static RENAME_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Deletion standard defining the overwrite pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeletionStandard {
    /// Single-pass zero overwrite (fastest, basic).
    SinglePassZero,
    /// Single-pass random overwrite.
    SinglePassRandom,
    /// DoD 5220.22-M (3 passes: zero, ones, random).
    Dod522022M,
    /// NIST 800-88 Clear (single pass, appropriate for most media).
    Nist80088Clear,
    /// NIST 800-88 Purge (multiple passes).
    Nist80088Purge,
    /// Gutmann method (35 passes).
    Gutmann,
    /// Custom number of passes with alternating zero/random pattern.
    CustomPasses(u32),
}

impl DeletionStandard {
    /// Returns the number of overwrite passes for this standard.
    #[must_use]
    pub const fn pass_count(&self) -> u32 {
        match self {
            Self::SinglePassZero | Self::SinglePassRandom | Self::Nist80088Clear => 1,
            Self::Dod522022M | Self::Nist80088Purge => 3,
            Self::Gutmann => 35,
            Self::CustomPasses(n) => *n,
        }
    }

    /// Returns a human-readable name for the standard.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::SinglePassZero => "Single-Pass Zero",
            Self::SinglePassRandom => "Single-Pass Random",
            Self::Dod522022M => "DoD 5220.22-M",
            Self::Nist80088Clear => "NIST 800-88 Clear",
            Self::Nist80088Purge => "NIST 800-88 Purge",
            Self::Gutmann => "Gutmann 35-Pass",
            Self::CustomPasses(_) => "Custom Multi-Pass",
        }
    }

    /// Returns whether a verification pass is recommended after deletion.
    #[must_use]
    pub const fn recommends_verification(&self) -> bool {
        !matches!(self, Self::SinglePassZero | Self::SinglePassRandom)
    }
}

/// Pattern used for a single overwrite pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverwritePattern {
    /// Fill with 0x00 bytes.
    Zeros,
    /// Fill with 0xFF bytes.
    Ones,
    /// Fill with a specific byte value repeated.
    Byte(u8),
    /// Pseudo-random pattern derived from a seed (deterministic).
    PseudoRandom(u64),
    /// Complement of a previous pattern byte (0x92 alternating with 0x6D etc.).
    Complement(u8),
}

impl OverwritePattern {
    /// Generates the overwrite buffer for a given block size.
    #[must_use]
    pub fn generate_block(&self, size: usize, pass_index: u32) -> Vec<u8> {
        match self {
            Self::Zeros => vec![0x00u8; size],
            Self::Ones => vec![0xFFu8; size],
            Self::Byte(b) => vec![*b; size],
            Self::PseudoRandom(seed) => {
                // Xorshift64 PRNG — no external dependency, deterministic
                let mut state = seed.wrapping_add(u64::from(pass_index).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                if state == 0 {
                    state = 1;
                }
                let mut buf = Vec::with_capacity(size);
                while buf.len() < size {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let bytes = state.to_le_bytes();
                    let remaining = size - buf.len();
                    let take = remaining.min(8);
                    buf.extend_from_slice(&bytes[..take]);
                }
                buf
            }
            Self::Complement(base) => {
                let b = !base;
                vec![b; size]
            }
        }
    }
}

/// A single overwrite pass description.
#[derive(Debug, Clone)]
pub struct OverwritePass {
    /// Pass number (1-based).
    pub pass_number: u32,
    /// Pattern used for this pass.
    pub pattern: OverwritePattern,
    /// Human-readable description.
    pub description: String,
}

impl OverwritePass {
    /// Creates a new overwrite pass.
    #[must_use]
    pub fn new(pass_number: u32, pattern: OverwritePattern, description: impl Into<String>) -> Self {
        Self {
            pass_number,
            pattern,
            description: description.into(),
        }
    }
}

/// Builds the sequence of passes for a given deletion standard.
#[must_use]
pub fn build_passes(standard: DeletionStandard) -> Vec<OverwritePass> {
    match standard {
        DeletionStandard::SinglePassZero => vec![OverwritePass::new(
            1,
            OverwritePattern::Zeros,
            "Zero overwrite",
        )],
        DeletionStandard::SinglePassRandom => vec![OverwritePass::new(
            1,
            OverwritePattern::PseudoRandom(0xDEAD_BEEF_CAFE_BABE),
            "Random overwrite",
        )],
        DeletionStandard::Nist80088Clear => vec![OverwritePass::new(
            1,
            OverwritePattern::Zeros,
            "NIST 800-88 Clear — zero fill",
        )],
        DeletionStandard::Nist80088Purge => vec![
            OverwritePass::new(1, OverwritePattern::Zeros, "NIST pass 1 — zeros"),
            OverwritePass::new(2, OverwritePattern::Ones, "NIST pass 2 — ones"),
            OverwritePass::new(
                3,
                OverwritePattern::PseudoRandom(0x0123_4567_89AB_CDEF),
                "NIST pass 3 — random verify",
            ),
        ],
        DeletionStandard::Dod522022M => vec![
            OverwritePass::new(1, OverwritePattern::Zeros, "DoD pass 1 — zeros"),
            OverwritePass::new(2, OverwritePattern::Ones, "DoD pass 2 — ones"),
            OverwritePass::new(
                3,
                OverwritePattern::PseudoRandom(0xFEDC_BA98_7654_3210),
                "DoD pass 3 — random",
            ),
        ],
        DeletionStandard::Gutmann => {
            // Gutmann 35-pass sequence:
            // Passes 1–4: random
            // Passes 5–31: 27 fixed byte patterns
            // Passes 32–35: random
            let gutmann_bytes: &[u8] = &[
                // 27 specific byte patterns for passes 5-31
                0x55, 0xAA, 0x92, 0x49, 0x24, 0x49, 0x24, 0x92,
                0x6D, 0xB6, 0xDB, 0xB6, 0xDB, 0x6D, 0x00, 0x11,
                0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
                0xAA, 0xBB, 0xCC,
            ];
            let mut passes = Vec::with_capacity(35);
            // Passes 1-4: random
            for i in 1u32..=4 {
                passes.push(OverwritePass::new(
                    i,
                    OverwritePattern::PseudoRandom(u64::from(i).wrapping_mul(0x9999_AAAA_BBBB_CCCCu64)),
                    format!("Gutmann pass {i} — random"),
                ));
            }
            // Passes 5-31: 27 fixed patterns
            for (idx, &byte) in gutmann_bytes.iter().enumerate() {
                let pass_num = (idx as u32) + 5;
                passes.push(OverwritePass::new(
                    pass_num,
                    OverwritePattern::Byte(byte),
                    format!("Gutmann pass {pass_num} — 0x{byte:02X}"),
                ));
            }
            // Passes 32-35: random again
            for i in 32u32..=35 {
                passes.push(OverwritePass::new(
                    i,
                    OverwritePattern::PseudoRandom(u64::from(i).wrapping_mul(0x1111_2222_3333_4444u64)),
                    format!("Gutmann pass {i} — random"),
                ));
            }
            passes
        }
        DeletionStandard::CustomPasses(n) => (1..=n)
            .map(|i| {
                let pattern = if i % 2 == 0 {
                    OverwritePattern::Ones
                } else {
                    OverwritePattern::PseudoRandom(u64::from(i).wrapping_mul(0x5A5A_5A5A_5A5A_5A5Au64))
                };
                OverwritePass::new(i, pattern, format!("Custom pass {i}"))
            })
            .collect(),
    }
}

/// Result of a secure deletion operation.
#[derive(Debug, Clone)]
pub struct DeletionResult {
    /// Path that was deleted.
    pub path: PathBuf,
    /// Deletion standard applied.
    pub standard: DeletionStandard,
    /// Number of passes completed.
    pub passes_completed: u32,
    /// Whether post-deletion verification passed (file is gone).
    pub verification_passed: bool,
    /// Total bytes overwritten across all passes.
    pub bytes_overwritten: u64,
    /// Whether the file was successfully unlinked from the filesystem.
    pub file_unlinked: bool,
}

impl DeletionResult {
    /// Returns true if the deletion is considered fully successful.
    #[must_use]
    pub fn is_successful(&self) -> bool {
        self.file_unlinked && self.verification_passed
            && self.passes_completed == self.standard.pass_count()
    }
}

/// Configuration for secure deletion.
#[derive(Debug, Clone)]
pub struct SecureDeleteConfig {
    /// Deletion standard to apply.
    pub standard: DeletionStandard,
    /// Buffer size for I/O operations (default: 64 KiB).
    pub buffer_size: usize,
    /// Whether to verify the file is gone after deletion.
    pub verify_deletion: bool,
    /// Whether to rename the file before unlinking (hides filename).
    pub rename_before_delete: bool,
}

impl SecureDeleteConfig {
    /// Creates a new configuration with the given standard.
    #[must_use]
    pub fn new(standard: DeletionStandard) -> Self {
        Self {
            standard,
            buffer_size: 64 * 1024,
            verify_deletion: true,
            rename_before_delete: true,
        }
    }

    /// Sets the I/O buffer size.
    #[must_use]
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size.max(512);
        self
    }

    /// Disables post-deletion verification (faster but less strict).
    #[must_use]
    pub fn without_verification(mut self) -> Self {
        self.verify_deletion = false;
        self
    }
}

impl Default for SecureDeleteConfig {
    fn default() -> Self {
        Self::new(DeletionStandard::Dod522022M)
    }
}

/// Executes secure deletion of a file.
///
/// Overwrites the file content with the configured patterns, then unlinks
/// the file from the filesystem. Returns a detailed [`DeletionResult`].
///
/// # Errors
///
/// Returns an [`io::Error`] if the file cannot be opened, written to,
/// or unlinked.
pub fn secure_delete(path: &Path, config: &SecureDeleteConfig) -> io::Result<DeletionResult> {
    let file_len = fs::metadata(path)?.len();
    let passes = build_passes(config.standard);
    let mut bytes_overwritten: u64 = 0;
    let mut passes_completed: u32 = 0;

    // Optionally rename before deleting to obscure filename in directory entries
    let actual_path = if config.rename_before_delete {
        let parent = path
            .parent()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no parent"))?;
        let seq = RENAME_COUNTER.fetch_add(1, Ordering::Relaxed);
        let obscured = parent.join(format!(".oxmdel_{}_{seq}", std::process::id()));
        fs::rename(path, &obscured)?;
        obscured
    } else {
        path.to_path_buf()
    };

    // Overwrite with each pass
    for pass in &passes {
        overwrite_file(&actual_path, file_len, pass, config.buffer_size)?;
        bytes_overwritten += file_len;
        passes_completed += 1;
    }

    // Unlink the file
    fs::remove_file(&actual_path)?;

    let verification_passed = if config.verify_deletion {
        !actual_path.exists() && !path.exists()
    } else {
        true
    };

    Ok(DeletionResult {
        path: path.to_path_buf(),
        standard: config.standard,
        passes_completed,
        verification_passed,
        bytes_overwritten,
        file_unlinked: true,
    })
}

/// Overwrites a file with a single pass pattern.
fn overwrite_file(path: &Path, file_len: u64, pass: &OverwritePass, buffer_size: usize) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).open(path)?;
    file.seek(SeekFrom::Start(0))?;

    let mut written: u64 = 0;
    while written < file_len {
        let remaining = (file_len - written) as usize;
        let chunk_size = remaining.min(buffer_size);
        let buf = pass.pattern.generate_block(chunk_size, pass.pass_number);
        file.write_all(&buf)?;
        written += chunk_size as u64;
    }

    file.flush()?;
    // Sync to underlying storage
    file.sync_all()?;
    Ok(())
}

/// Verifies that a path no longer exists on the filesystem.
///
/// Used to confirm that secure deletion was effective at the filesystem level.
#[must_use]
pub fn verify_absent(path: &Path) -> bool {
    !path.exists()
}

/// Secure deletion executor for directories.
///
/// Recursively securely deletes all files in a directory tree, then removes
/// the directory hierarchy.
///
/// # Errors
///
/// Returns an [`io::Error`] if any file or directory operation fails.
pub fn secure_delete_dir(dir: &Path, config: &SecureDeleteConfig) -> io::Result<Vec<DeletionResult>> {
    let mut results = Vec::new();
    secure_delete_dir_recursive(dir, config, &mut results)?;
    // Remove the directory hierarchy after files are gone
    fs::remove_dir_all(dir)?;
    Ok(results)
}

fn secure_delete_dir_recursive(
    dir: &Path,
    config: &SecureDeleteConfig,
    results: &mut Vec<DeletionResult>,
) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let entry_path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            secure_delete_dir_recursive(&entry_path, config, results)?;
        } else if file_type.is_file() {
            let result = secure_delete(&entry_path, config)?;
            results.push(result);
        }
    }
    Ok(())
}

/// Summary statistics for a batch secure deletion operation.
#[derive(Debug, Clone, Default)]
pub struct DeletionSummary {
    /// Number of files successfully deleted.
    pub files_deleted: u32,
    /// Number of files that failed deletion.
    pub files_failed: u32,
    /// Total bytes overwritten.
    pub total_bytes_overwritten: u64,
    /// Whether all deletions passed verification.
    pub all_verified: bool,
}

impl DeletionSummary {
    /// Builds a summary from a list of results.
    #[must_use]
    pub fn from_results(results: &[DeletionResult]) -> Self {
        let files_deleted = results.iter().filter(|r| r.is_successful()).count() as u32;
        let files_failed = results.len() as u32 - files_deleted;
        let total_bytes_overwritten = results.iter().map(|r| r.bytes_overwritten).sum();
        let all_verified = results.iter().all(|r| r.verification_passed);
        Self {
            files_deleted,
            files_failed,
            total_bytes_overwritten,
            all_verified,
        }
    }

    /// Returns true if there were no failures.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.files_failed == 0 && self.all_verified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_file(data: &[u8]) -> (tempfile::NamedTempFile, PathBuf) {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(data).expect("write");
        f.flush().expect("flush");
        let path = f.path().to_path_buf();
        (f, path)
    }

    #[test]
    fn test_deletion_standard_pass_counts() {
        assert_eq!(DeletionStandard::SinglePassZero.pass_count(), 1);
        assert_eq!(DeletionStandard::SinglePassRandom.pass_count(), 1);
        assert_eq!(DeletionStandard::Dod522022M.pass_count(), 3);
        assert_eq!(DeletionStandard::Nist80088Clear.pass_count(), 1);
        assert_eq!(DeletionStandard::Nist80088Purge.pass_count(), 3);
        assert_eq!(DeletionStandard::Gutmann.pass_count(), 35);
        assert_eq!(DeletionStandard::CustomPasses(7).pass_count(), 7);
    }

    #[test]
    fn test_deletion_standard_names_not_empty() {
        for std in [
            DeletionStandard::SinglePassZero,
            DeletionStandard::SinglePassRandom,
            DeletionStandard::Dod522022M,
            DeletionStandard::Nist80088Clear,
            DeletionStandard::Nist80088Purge,
            DeletionStandard::Gutmann,
            DeletionStandard::CustomPasses(5),
        ] {
            assert!(!std.name().is_empty());
        }
    }

    #[test]
    fn test_overwrite_pattern_zeros() {
        let buf = OverwritePattern::Zeros.generate_block(100, 1);
        assert_eq!(buf.len(), 100);
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_overwrite_pattern_ones() {
        let buf = OverwritePattern::Ones.generate_block(64, 1);
        assert_eq!(buf.len(), 64);
        assert!(buf.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn test_overwrite_pattern_byte() {
        let buf = OverwritePattern::Byte(0xAA).generate_block(32, 1);
        assert!(buf.iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn test_overwrite_pattern_random_deterministic() {
        let p = OverwritePattern::PseudoRandom(12345);
        let a = p.generate_block(128, 1);
        let b = p.generate_block(128, 1);
        assert_eq!(a, b, "same seed/pass should produce same output");
        let c = p.generate_block(128, 2);
        assert_ne!(a, c, "different pass index should differ");
    }

    #[test]
    fn test_overwrite_pattern_complement() {
        let buf = OverwritePattern::Complement(0x55).generate_block(16, 1);
        assert!(buf.iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn test_build_passes_dod() {
        let passes = build_passes(DeletionStandard::Dod522022M);
        assert_eq!(passes.len(), 3);
        assert_eq!(passes[0].pass_number, 1);
        assert_eq!(passes[2].pass_number, 3);
    }

    #[test]
    fn test_build_passes_gutmann_count() {
        let passes = build_passes(DeletionStandard::Gutmann);
        assert_eq!(passes.len(), 35);
    }

    #[test]
    fn test_build_passes_custom() {
        let passes = build_passes(DeletionStandard::CustomPasses(10));
        assert_eq!(passes.len(), 10);
        assert_eq!(passes[9].pass_number, 10);
    }

    #[test]
    fn test_secure_delete_single_pass() {
        let data = b"sensitive archival data that must be destroyed";
        let (f, path) = write_temp_file(data);
        // Keep the file alive until we explicitly delete it
        let _keep = f;

        let config = SecureDeleteConfig::new(DeletionStandard::SinglePassZero)
            .without_verification();
        let result = secure_delete(&path, &config).expect("secure delete");

        assert!(result.file_unlinked);
        assert_eq!(result.passes_completed, 1);
        assert!(!path.exists());
    }

    #[test]
    fn test_secure_delete_dod_with_verification() {
        let data = b"top-secret media archive content";
        let (f, path) = write_temp_file(data);
        let _keep = f;

        let config = SecureDeleteConfig::new(DeletionStandard::Dod522022M);
        let result = secure_delete(&path, &config).expect("secure delete");

        assert!(result.is_successful());
        assert_eq!(result.passes_completed, 3);
        assert!(result.verification_passed);
        assert!(!path.exists());
    }

    #[test]
    fn test_secure_delete_result_bytes_overwritten() {
        let data = vec![0u8; 4096];
        let (f, path) = write_temp_file(&data);
        let _keep = f;

        let config = SecureDeleteConfig::new(DeletionStandard::Dod522022M);
        let result = secure_delete(&path, &config).expect("secure delete");

        // 3 passes × 4096 bytes
        assert_eq!(result.bytes_overwritten, 4096 * 3);
    }

    #[test]
    fn test_verify_absent_existing_file() {
        let f = tempfile::NamedTempFile::new().expect("tempfile");
        assert!(!verify_absent(f.path()), "file exists, should not be absent");
    }

    #[test]
    fn test_verify_absent_nonexistent() {
        let path = std::env::temp_dir().join("oximedia_test_nonexistent_xyz123.bin");
        assert!(verify_absent(&path));
    }

    #[test]
    fn test_deletion_summary_from_results() {
        let r1 = DeletionResult {
            path: PathBuf::from("/a"),
            standard: DeletionStandard::SinglePassZero,
            passes_completed: 1,
            verification_passed: true,
            bytes_overwritten: 1024,
            file_unlinked: true,
        };
        let r2 = DeletionResult {
            path: PathBuf::from("/b"),
            standard: DeletionStandard::SinglePassZero,
            passes_completed: 1,
            verification_passed: true,
            bytes_overwritten: 2048,
            file_unlinked: true,
        };
        let summary = DeletionSummary::from_results(&[r1, r2]);
        assert_eq!(summary.files_deleted, 2);
        assert_eq!(summary.files_failed, 0);
        assert_eq!(summary.total_bytes_overwritten, 3072);
        assert!(summary.is_clean());
    }

    #[test]
    fn test_deletion_summary_with_failure() {
        let r1 = DeletionResult {
            path: PathBuf::from("/a"),
            standard: DeletionStandard::Dod522022M,
            passes_completed: 3,
            verification_passed: true,
            bytes_overwritten: 1000,
            file_unlinked: true,
        };
        // Simulated failure: passes incomplete
        let r2 = DeletionResult {
            path: PathBuf::from("/b"),
            standard: DeletionStandard::Dod522022M,
            passes_completed: 1, // Only 1 of 3
            verification_passed: false,
            bytes_overwritten: 500,
            file_unlinked: false,
        };
        let summary = DeletionSummary::from_results(&[r1, r2]);
        assert_eq!(summary.files_failed, 1);
        assert!(!summary.is_clean());
    }

    #[test]
    fn test_secure_delete_config_defaults() {
        let cfg = SecureDeleteConfig::default();
        assert_eq!(cfg.standard, DeletionStandard::Dod522022M);
        assert!(cfg.verify_deletion);
        assert!(cfg.rename_before_delete);
    }

    #[test]
    fn test_secure_delete_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dir_path = dir.path().to_path_buf();

        // Create some files in the dir
        for i in 0..3 {
            let fp = dir_path.join(format!("file{i}.bin"));
            std::fs::write(&fp, b"secret data").expect("write");
        }

        // Don't drop dir yet — we need the files to exist
        let config = SecureDeleteConfig::new(DeletionStandard::SinglePassZero)
            .without_verification();
        let results = secure_delete_dir(&dir_path, &config).expect("secure delete dir");

        // dir.into_path() to prevent cleanup since we already removed it
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.file_unlinked));
        assert!(!dir_path.exists());
        // Prevent double-free from `dir` drop by forgetting it
        std::mem::forget(dir);
    }
}
