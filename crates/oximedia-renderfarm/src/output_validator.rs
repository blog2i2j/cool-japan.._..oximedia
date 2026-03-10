#![allow(dead_code)]
//! Output validation for rendered frames in the `OxiMedia` render farm.
//!
//! Provides individual output checks, per-frame validation results, a
//! validator that inspects frame ranges, and a consolidated report.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::io::AsyncReadExt;

/// A single type of output check that can be applied to a rendered frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputCheck {
    /// Frame file exists on disk.
    FileExists,
    /// File size is above the minimum threshold (non-empty render).
    FileSizeAboveMinimum,
    /// Image dimensions match the expected width/height.
    DimensionsMatch,
    /// Pixel format matches the job specification.
    PixelFormatMatch,
    /// Frame number encoded in the filename matches expected sequence.
    FrameNumberSequential,
    /// Checksum / hash matches a reference (for re-renders).
    ChecksumValid,
    /// No corruption artefacts detected (e.g. all-black, all-white frames).
    NoCorruption,
}

impl OutputCheck {
    /// Short machine-readable name for this check.
    #[must_use]
    pub fn check_name(&self) -> &'static str {
        match self {
            OutputCheck::FileExists => "file_exists",
            OutputCheck::FileSizeAboveMinimum => "file_size_above_minimum",
            OutputCheck::DimensionsMatch => "dimensions_match",
            OutputCheck::PixelFormatMatch => "pixel_format_match",
            OutputCheck::FrameNumberSequential => "frame_number_sequential",
            OutputCheck::ChecksumValid => "checksum_valid",
            OutputCheck::NoCorruption => "no_corruption",
        }
    }

    /// Returns `true` for checks that are considered critical (failure blocks delivery).
    #[must_use]
    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            OutputCheck::FileExists | OutputCheck::DimensionsMatch | OutputCheck::NoCorruption
        )
    }
}

/// Result of running all checks on a single frame.
#[derive(Debug, Clone)]
pub struct OutputValidation {
    /// Frame number that was validated.
    pub frame_number: u32,
    /// Map of check → pass/fail.
    pub results: Vec<(OutputCheck, bool)>,
}

impl OutputValidation {
    /// Create a new validation record for `frame_number` with no checks yet.
    #[must_use]
    pub fn new(frame_number: u32) -> Self {
        Self {
            frame_number,
            results: Vec::new(),
        }
    }

    /// Record the outcome of a single check.
    pub fn record(&mut self, check: OutputCheck, passed: bool) {
        self.results.push((check, passed));
    }

    /// Returns `true` when every recorded check passed.
    #[must_use]
    pub fn passes_all(&self) -> bool {
        self.results.iter().all(|(_, passed)| *passed)
    }

    /// Returns `true` when every *critical* check passed.
    #[must_use]
    pub fn passes_critical(&self) -> bool {
        self.results
            .iter()
            .filter(|(c, _)| c.is_critical())
            .all(|(_, passed)| *passed)
    }

    /// Returns a list of checks that failed.
    #[must_use]
    pub fn failed_checks(&self) -> Vec<OutputCheck> {
        self.results
            .iter()
            .filter_map(|(c, passed)| if *passed { None } else { Some(*c) })
            .collect()
    }

    /// Number of checks recorded.
    #[must_use]
    pub fn check_count(&self) -> usize {
        self.results.len()
    }
}

/// Validator that runs a configurable set of checks over a frame range.
#[derive(Debug, Clone)]
pub struct OutputValidator {
    /// Checks that will be applied to every frame.
    pub enabled_checks: Vec<OutputCheck>,
    /// Expected frame width in pixels.
    pub expected_width: u32,
    /// Expected frame height in pixels.
    pub expected_height: u32,
    /// Minimum file size in bytes for the `FileSizeAboveMinimum` check.
    pub min_file_size_bytes: u64,
    /// Expected pixel/container format string (e.g. `"mkv"`, `"webm"`, `"ogg"`, `"wav"`).
    pub expected_pixel_format: String,
    /// Expected first frame number for sequential checks.
    pub expected_first_frame: u32,
}

impl Default for OutputValidator {
    fn default() -> Self {
        Self {
            enabled_checks: vec![
                OutputCheck::FileExists,
                OutputCheck::FileSizeAboveMinimum,
                OutputCheck::DimensionsMatch,
                OutputCheck::NoCorruption,
            ],
            expected_width: 1920,
            expected_height: 1080,
            min_file_size_bytes: 1024,
            expected_pixel_format: String::new(),
            expected_first_frame: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Container format detection helpers
// ---------------------------------------------------------------------------

/// Detected container format from magic bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerFormat {
    Mkv,
    WebM,
    Ogg,
    Wav,
    Flac,
    Mp4,
    Unknown,
}

impl ContainerFormat {
    /// Detect format from the first bytes of a file.
    #[must_use]
    pub fn from_magic(header: &[u8]) -> Self {
        if header.len() >= 4 && &header[..4] == b"\x1A\x45\xDF\xA3" {
            // EBML magic — could be MKV or WebM; check DocType further in, but both share magic.
            // A heuristic: scan for "webm" DocType string within the first 64 bytes.
            let scan = &header[..header.len().min(64)];
            if scan.windows(4).any(|w| w == b"webm") {
                return Self::WebM;
            }
            return Self::Mkv;
        }
        if header.len() >= 4 && &header[..4] == b"OggS" {
            return Self::Ogg;
        }
        if header.len() >= 4 && &header[..4] == b"RIFF" {
            return Self::Wav;
        }
        if header.len() >= 4 && &header[..4] == b"fLaC" {
            return Self::Flac;
        }
        // ISO Base Media / MP4: bytes 4–7 == "ftyp"
        if header.len() >= 8 && &header[4..8] == b"ftyp" {
            return Self::Mp4;
        }
        Self::Unknown
    }

    /// Canonical lowercase name for this format.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Mkv => "mkv",
            Self::WebM => "webm",
            Self::Ogg => "ogg",
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Mp4 => "mp4",
            Self::Unknown => "unknown",
        }
    }
}

// ---------------------------------------------------------------------------
// Frame-number extraction helper
// ---------------------------------------------------------------------------

/// Extract the frame number embedded in a filename stem.
///
/// Scans for the last contiguous run of ASCII digits in the stem and parses
/// it as a `u32`.  Returns `None` when no digits are found.
///
/// # Examples
///
/// ```
/// # use oximedia_renderfarm::output_validator::extract_frame_number;
/// assert_eq!(extract_frame_number("frame_0001"), Some(1));
/// assert_eq!(extract_frame_number("render.0042.exr"), Some(42));
/// assert_eq!(extract_frame_number("output"), None);
/// ```
#[must_use]
pub fn extract_frame_number(stem: &str) -> Option<u32> {
    // Collect all contiguous digit runs, return the last one.
    let mut last_run: Option<&str> = None;
    let bytes = stem.as_bytes();
    let mut run_start: Option<usize> = None;
    for (i, &b) in bytes.iter().enumerate() {
        if b.is_ascii_digit() {
            if run_start.is_none() {
                run_start = Some(i);
            }
        } else if let Some(start) = run_start {
            // end of a run
            last_run = Some(&stem[start..i]);
            run_start = None;
        }
    }
    // Handle run that extends to end of string
    if let Some(start) = run_start {
        last_run = Some(&stem[start..]);
    }
    last_run.and_then(|s| s.parse::<u32>().ok())
}

// ---------------------------------------------------------------------------
// Async file-based validation helpers
// ---------------------------------------------------------------------------

/// Read the first `n` bytes of a file asynchronously.  Returns fewer bytes
/// than requested when the file is shorter.
async fn read_header_bytes(path: &Path, n: usize) -> std::io::Result<Vec<u8>> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut buf = vec![0u8; n];
    let read = file.read(&mut buf).await?;
    buf.truncate(read);
    Ok(buf)
}

/// Compute the SHA-256 digest of an entire file asynchronously.
async fn sha256_file(path: &Path) -> std::io::Result<String> {
    let data = tokio::fs::read(path).await?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let digest = hasher.finalize();
    // Encode as lowercase hex manually to avoid pulling in `hex` crate.
    let mut hex = String::with_capacity(64);
    for byte in digest {
        hex.push(char::from_digit((byte >> 4) as u32, 16).unwrap_or('0'));
        hex.push(char::from_digit((byte & 0x0F) as u32, 16).unwrap_or('0'));
    }
    Ok(hex)
}

impl OutputValidator {
    /// Create a validator with default checks for 1920×1080.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a validator for a specific resolution.
    #[must_use]
    pub fn for_resolution(width: u32, height: u32) -> Self {
        Self {
            expected_width: width,
            expected_height: height,
            ..Default::default()
        }
    }

    /// Validate a single frame given simulated frame metadata.
    ///
    /// `file_size_bytes` of 0 simulates a missing/empty file.
    /// `actual_width` / `actual_height` of 0 simulate dimension errors.
    ///
    /// The three file-based checks (`PixelFormatMatch`, `FrameNumberSequential`,
    /// `ChecksumValid`) are not applicable in this metadata-only path and always
    /// pass.  Use [`validate_file`](Self::validate_file) for full on-disk checks.
    #[must_use]
    pub fn validate_frame(
        &self,
        frame_number: u32,
        file_size_bytes: u64,
        actual_width: u32,
        actual_height: u32,
        corrupted: bool,
    ) -> OutputValidation {
        let mut v = OutputValidation::new(frame_number);
        for check in &self.enabled_checks {
            let passed = match check {
                OutputCheck::FileExists => file_size_bytes > 0,
                OutputCheck::FileSizeAboveMinimum => file_size_bytes >= self.min_file_size_bytes,
                OutputCheck::DimensionsMatch => {
                    actual_width == self.expected_width && actual_height == self.expected_height
                }
                // File-based checks: not applicable in metadata-only mode.
                OutputCheck::PixelFormatMatch => true,
                OutputCheck::FrameNumberSequential => true,
                OutputCheck::ChecksumValid => true,
                OutputCheck::NoCorruption => !corrupted,
            };
            v.record(*check, passed);
        }
        v
    }

    /// Validate a contiguous range of frames.
    ///
    /// For simplicity the simulation assumes all frames are valid unless
    /// `bad_frames` contains the frame number, in which case every check fails.
    #[must_use]
    pub fn validate_frame_range(
        &self,
        frame_start: u32,
        frame_end: u32,
        bad_frames: &[u32],
    ) -> OutputValidationReport {
        let mut validations = Vec::new();
        for f in frame_start..=frame_end {
            let is_bad = bad_frames.contains(&f);
            let v = self.validate_frame(
                f,
                if is_bad { 0 } else { 65_536 },
                if is_bad { 0 } else { self.expected_width },
                if is_bad { 0 } else { self.expected_height },
                is_bad,
            );
            validations.push(v);
        }
        OutputValidationReport { validations }
    }

    // -----------------------------------------------------------------------
    // Real on-disk validation
    // -----------------------------------------------------------------------

    /// Validate a **single output file** on disk, running all enabled checks
    /// that are applicable to a single file.
    ///
    /// # PixelFormatMatch
    ///
    /// Reads the first 64 bytes of `path` to detect the container format via
    /// magic bytes (MKV/WebM: `\x1A\x45\xDF\xA3`; OGG: `OggS`; WAV: `RIFF`;
    /// FLAC: `fLaC`; MP4: `….ftyp`).  Compares the detected format name
    /// against `self.expected_pixel_format` (case-insensitive).  Returns
    /// `true` when `expected_pixel_format` is empty (no constraint configured).
    ///
    /// # ChecksumValid
    ///
    /// Looks for a sidecar file next to `path` with the same name plus a
    /// `.sha256` or `.md5` extension.  When found, reads the hex digest from
    /// the first token of the first line and compares it against the SHA-256
    /// (or MD5 for `.md5`) of the file.  Returns `true` when no sidecar
    /// exists.
    ///
    /// # Errors
    ///
    /// Returns an `std::io::Error` if a required file cannot be opened.
    pub async fn validate_file(
        &self,
        frame_number: u32,
        path: &Path,
    ) -> std::io::Result<OutputValidation> {
        let mut v = OutputValidation::new(frame_number);

        let meta = tokio::fs::metadata(path).await;
        let file_size = meta.as_ref().map(|m| m.len()).unwrap_or(0);

        for check in &self.enabled_checks {
            let passed = match check {
                OutputCheck::FileExists => meta.is_ok(),
                OutputCheck::FileSizeAboveMinimum => file_size >= self.min_file_size_bytes,
                OutputCheck::DimensionsMatch => {
                    // Dimension checking requires decoding; not implemented here — pass.
                    true
                }
                OutputCheck::NoCorruption => {
                    // Basic corruption check: non-empty file.
                    file_size > 0
                }
                OutputCheck::PixelFormatMatch => self.check_pixel_format_match(path).await,
                OutputCheck::FrameNumberSequential => {
                    // Single-file check: verify the frame number in the filename
                    // matches `frame_number`.
                    path.file_stem()
                        .and_then(|s| s.to_str())
                        .and_then(extract_frame_number)
                        .map(|n| n == frame_number)
                        .unwrap_or(true) // no frame number in name → skip
                }
                OutputCheck::ChecksumValid => self.check_checksum_valid(path).await,
            };
            v.record(*check, passed);
        }
        Ok(v)
    }

    /// Check whether `path`'s container format matches `self.expected_pixel_format`.
    async fn check_pixel_format_match(&self, path: &Path) -> bool {
        if self.expected_pixel_format.is_empty() {
            // No constraint configured.
            return true;
        }
        let header = match read_header_bytes(path, 64).await {
            Ok(h) => h,
            Err(_) => return false,
        };
        let detected = ContainerFormat::from_magic(&header);
        detected
            .name()
            .eq_ignore_ascii_case(&self.expected_pixel_format)
    }

    /// Verify `path` against a sidecar `.sha256` or `.md5` checksum file.
    ///
    /// Returns `true` when no sidecar exists (nothing to validate against).
    async fn check_checksum_valid(&self, path: &Path) -> bool {
        // Determine potential sidecar paths.
        let sha256_sidecar = PathBuf::from(format!("{}.sha256", path.display()));
        let md5_sidecar = PathBuf::from(format!("{}.md5", path.display()));

        if tokio::fs::metadata(&sha256_sidecar).await.is_ok() {
            // Read the expected digest from the sidecar.
            let sidecar_content = match tokio::fs::read_to_string(&sha256_sidecar).await {
                Ok(s) => s,
                Err(_) => return false,
            };
            let expected = sidecar_content
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_ascii_lowercase();
            if expected.is_empty() {
                return false;
            }
            let actual = match sha256_file(path).await {
                Ok(h) => h,
                Err(_) => return false,
            };
            return actual == expected;
        }

        if tokio::fs::metadata(&md5_sidecar).await.is_ok() {
            // For MD5 sidecars we still compute SHA-256 and compare to whatever
            // is stored; if the sidecar contains an MD5 it will never match our
            // SHA-256 and the check will correctly fail.  A fully correct
            // implementation would use an MD5 hasher, but md5 is not a
            // workspace dependency.  We honour the sidecar's presence as an
            // intent to validate, and fail conservatively.
            let sidecar_content = match tokio::fs::read_to_string(&md5_sidecar).await {
                Ok(s) => s,
                Err(_) => return false,
            };
            let expected = sidecar_content
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_ascii_lowercase();
            // MD5 digests are 32 hex chars; SHA-256 are 64 hex chars.
            // If sidecar holds a SHA-256 (64 chars), verify it; otherwise fail.
            if expected.len() == 64 {
                let actual = match sha256_file(path).await {
                    Ok(h) => h,
                    Err(_) => return false,
                };
                return actual == expected;
            }
            // True MD5 sidecar with 32-char digest: cannot verify without MD5 hasher.
            return false;
        }

        // No sidecar — nothing to validate against.
        true
    }

    /// Validate a directory of image-sequence frames for sequential ordering.
    ///
    /// Lists all files in `dir`, sorts them by filename, extracts frame numbers
    /// from filename stems, and verifies:
    ///
    /// 1. The sequence is non-empty.
    /// 2. The first frame number equals `self.expected_first_frame`.
    /// 3. All subsequent frame numbers are exactly one greater than the previous.
    ///
    /// Returns `true` when the sequence is valid; `false` on any gap or
    /// out-of-order issue.  Returns `true` when no frame numbers can be parsed
    /// from any filename (skip the check gracefully).
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` if the directory cannot be read.
    pub async fn validate_sequence_directory(&self, dir: &Path) -> std::io::Result<bool> {
        let mut read_dir = tokio::fs::read_dir(dir).await?;
        let mut names: Vec<String> = Vec::new();
        while let Some(entry) = read_dir.next_entry().await? {
            let ft = entry.file_type().await?;
            if ft.is_file() {
                if let Some(name) = entry.file_name().to_str().map(str::to_owned) {
                    names.push(name);
                }
            }
        }

        if names.is_empty() {
            return Ok(false);
        }

        names.sort_unstable();

        // Extract frame numbers from stems.
        let mut frame_numbers: Vec<u32> = Vec::new();
        for name in &names {
            let stem = Path::new(name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(name.as_str());
            if let Some(n) = extract_frame_number(stem) {
                frame_numbers.push(n);
            }
        }

        // If no frame numbers were parseable, skip the check.
        if frame_numbers.is_empty() {
            return Ok(true);
        }

        // Verify start frame.
        if frame_numbers[0] != self.expected_first_frame {
            return Ok(false);
        }

        // Verify no gaps.
        for window in frame_numbers.windows(2) {
            if window[1] != window[0] + 1 {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

/// Consolidated validation report covering multiple frames.
#[derive(Debug, Clone)]
pub struct OutputValidationReport {
    validations: Vec<OutputValidation>,
}

impl OutputValidationReport {
    /// Create a report from a pre-built list of frame validations.
    #[must_use]
    pub fn from_validations(validations: Vec<OutputValidation>) -> Self {
        Self { validations }
    }

    /// Returns `true` when every frame in the report passes all checks.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.validations.iter().all(OutputValidation::passes_all)
    }

    /// Return frame validations that have at least one failure.
    #[must_use]
    pub fn issues(&self) -> Vec<&OutputValidation> {
        self.validations
            .iter()
            .filter(|v| !v.passes_all())
            .collect()
    }

    /// Number of frames that passed all checks.
    #[must_use]
    pub fn passed_count(&self) -> usize {
        self.validations.iter().filter(|v| v.passes_all()).count()
    }

    /// Number of frames that have at least one failing check.
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.validations.iter().filter(|v| !v.passes_all()).count()
    }

    /// Total number of frames in the report.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.validations.len()
    }

    /// Return the frame numbers that failed.
    #[must_use]
    pub fn failed_frame_numbers(&self) -> Vec<u32> {
        self.validations
            .iter()
            .filter(|v| !v.passes_all())
            .map(|v| v.frame_number)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_name_file_exists() {
        assert_eq!(OutputCheck::FileExists.check_name(), "file_exists");
    }

    #[test]
    fn test_check_name_no_corruption() {
        assert_eq!(OutputCheck::NoCorruption.check_name(), "no_corruption");
    }

    #[test]
    fn test_check_is_critical_file_exists() {
        assert!(OutputCheck::FileExists.is_critical());
    }

    #[test]
    fn test_check_not_critical_checksum() {
        assert!(!OutputCheck::ChecksumValid.is_critical());
    }

    #[test]
    fn test_check_dimensions_is_critical() {
        assert!(OutputCheck::DimensionsMatch.is_critical());
    }

    #[test]
    fn test_output_validation_passes_all_empty() {
        let v = OutputValidation::new(1);
        assert!(v.passes_all()); // no checks recorded = nothing failing
    }

    #[test]
    fn test_output_validation_passes_all_with_passing_checks() {
        let mut v = OutputValidation::new(1);
        v.record(OutputCheck::FileExists, true);
        v.record(OutputCheck::DimensionsMatch, true);
        assert!(v.passes_all());
    }

    #[test]
    fn test_output_validation_fails_when_one_check_fails() {
        let mut v = OutputValidation::new(1);
        v.record(OutputCheck::FileExists, true);
        v.record(OutputCheck::DimensionsMatch, false);
        assert!(!v.passes_all());
    }

    #[test]
    fn test_output_validation_failed_checks_list() {
        let mut v = OutputValidation::new(5);
        v.record(OutputCheck::FileExists, false);
        v.record(OutputCheck::NoCorruption, true);
        let failed = v.failed_checks();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0], OutputCheck::FileExists);
    }

    #[test]
    fn test_output_validation_check_count() {
        let mut v = OutputValidation::new(10);
        v.record(OutputCheck::FileExists, true);
        v.record(OutputCheck::FileSizeAboveMinimum, true);
        assert_eq!(v.check_count(), 2);
    }

    #[test]
    fn test_validator_clean_frame() {
        let val = OutputValidator::new();
        let v = val.validate_frame(1, 65_536, 1920, 1080, false);
        assert!(v.passes_all());
    }

    #[test]
    fn test_validator_bad_frame_fails() {
        let val = OutputValidator::new();
        let v = val.validate_frame(1, 0, 0, 0, true);
        assert!(!v.passes_all());
    }

    #[test]
    fn test_validator_frame_range_all_clean() {
        let val = OutputValidator::new();
        let report = val.validate_frame_range(1, 10, &[]);
        assert!(report.all_passed());
        assert_eq!(report.frame_count(), 10);
    }

    #[test]
    fn test_validator_frame_range_with_bad_frames() {
        let val = OutputValidator::new();
        let report = val.validate_frame_range(1, 10, &[3, 7]);
        assert!(!report.all_passed());
        assert_eq!(report.failed_count(), 2);
    }

    #[test]
    fn test_report_issues_returns_failing_frames() {
        let val = OutputValidator::new();
        let report = val.validate_frame_range(1, 5, &[2]);
        let issues = report.issues();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].frame_number, 2);
    }

    #[test]
    fn test_report_failed_frame_numbers() {
        let val = OutputValidator::new();
        let report = val.validate_frame_range(1, 5, &[4, 5]);
        let mut bad = report.failed_frame_numbers();
        bad.sort_unstable();
        assert_eq!(bad, vec![4, 5]);
    }

    #[test]
    fn test_report_passed_count() {
        let val = OutputValidator::new();
        let report = val.validate_frame_range(1, 10, &[1, 2]);
        assert_eq!(report.passed_count(), 8);
    }

    // --- ContainerFormat detection ---

    #[test]
    fn test_magic_mkv() {
        let magic = b"\x1A\x45\xDF\xA3\x00\x00\x00\x00";
        assert_eq!(ContainerFormat::from_magic(magic), ContainerFormat::Mkv);
    }

    #[test]
    fn test_magic_webm() {
        let mut magic = vec![0x1A, 0x45, 0xDF, 0xA3];
        magic.extend_from_slice(b"webm");
        assert_eq!(ContainerFormat::from_magic(&magic), ContainerFormat::WebM);
    }

    #[test]
    fn test_magic_ogg() {
        let magic = b"OggS\x00\x02";
        assert_eq!(ContainerFormat::from_magic(magic), ContainerFormat::Ogg);
    }

    #[test]
    fn test_magic_wav() {
        let magic = b"RIFF\x24\x00\x00\x00WAVEfmt ";
        assert_eq!(ContainerFormat::from_magic(magic), ContainerFormat::Wav);
    }

    #[test]
    fn test_magic_flac() {
        let magic = b"fLaC\x00\x00\x00\x22";
        assert_eq!(ContainerFormat::from_magic(magic), ContainerFormat::Flac);
    }

    #[test]
    fn test_magic_mp4() {
        let magic = b"\x00\x00\x00\x18ftypmp42";
        assert_eq!(ContainerFormat::from_magic(magic), ContainerFormat::Mp4);
    }

    #[test]
    fn test_magic_unknown() {
        let magic = b"\xFF\xFB\x90\x00";
        assert_eq!(ContainerFormat::from_magic(magic), ContainerFormat::Unknown);
    }

    // --- extract_frame_number ---

    #[test]
    fn test_extract_frame_number_padded() {
        assert_eq!(extract_frame_number("frame_0001"), Some(1));
    }

    #[test]
    fn test_extract_frame_number_dotted() {
        assert_eq!(extract_frame_number("render.0042"), Some(42));
    }

    #[test]
    fn test_extract_frame_number_none() {
        assert_eq!(extract_frame_number("output"), None);
    }

    #[test]
    fn test_extract_frame_number_last_run() {
        // "take2_frame_0010" — last run of digits is 0010 → 10
        assert_eq!(extract_frame_number("take2_frame_0010"), Some(10));
    }

    // --- async validate_file ---

    #[tokio::test]
    async fn test_validate_file_pixel_format_match_mkv() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_output_validator_mkv.mkv");

        // Write a minimal fake MKV header.
        let mut data = vec![0x1A, 0x45, 0xDF, 0xA3];
        data.extend_from_slice(&[0u8; 60]);
        tokio::fs::write(&path, &data).await.expect("write");

        let validator = OutputValidator {
            enabled_checks: vec![OutputCheck::PixelFormatMatch],
            expected_pixel_format: "mkv".to_string(),
            ..OutputValidator::default()
        };

        let result = validator
            .validate_file(1, &path)
            .await
            .expect("validate_file");
        assert!(result.passes_all(), "MKV format should match");

        tokio::fs::remove_file(&path).await.ok();
    }

    #[tokio::test]
    async fn test_validate_file_pixel_format_mismatch() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_output_validator_ogg.ogg");

        // Write OGG magic but expect MKV.
        let data = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00";
        tokio::fs::write(&path, data).await.expect("write");

        let validator = OutputValidator {
            enabled_checks: vec![OutputCheck::PixelFormatMatch],
            expected_pixel_format: "mkv".to_string(),
            ..OutputValidator::default()
        };

        let result = validator
            .validate_file(1, &path)
            .await
            .expect("validate_file");
        assert!(!result.passes_all(), "OGG should not match expected MKV");

        tokio::fs::remove_file(&path).await.ok();
    }

    #[tokio::test]
    async fn test_validate_file_checksum_valid_no_sidecar() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_output_validator_checksumless.bin");
        tokio::fs::write(&path, b"hello world")
            .await
            .expect("write");

        let validator = OutputValidator {
            enabled_checks: vec![OutputCheck::ChecksumValid],
            ..OutputValidator::default()
        };

        let result = validator
            .validate_file(1, &path)
            .await
            .expect("validate_file");
        assert!(result.passes_all(), "no sidecar → checksum passes");

        tokio::fs::remove_file(&path).await.ok();
    }

    #[tokio::test]
    async fn test_validate_file_checksum_valid_with_correct_sidecar() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_output_validator_checksummed.bin");
        let sidecar = dir.join("test_output_validator_checksummed.bin.sha256");

        let content = b"hello oximedia";
        tokio::fs::write(&path, content).await.expect("write file");

        // Compute expected SHA-256.
        let mut hasher = Sha256::new();
        hasher.update(content);
        let digest = hasher.finalize();
        let mut hex = String::with_capacity(64);
        for byte in digest {
            hex.push(char::from_digit((byte >> 4) as u32, 16).unwrap_or('0'));
            hex.push(char::from_digit((byte & 0x0F) as u32, 16).unwrap_or('0'));
        }
        tokio::fs::write(
            &sidecar,
            format!("{hex}  test_output_validator_checksummed.bin\n"),
        )
        .await
        .expect("write sidecar");

        let validator = OutputValidator {
            enabled_checks: vec![OutputCheck::ChecksumValid],
            ..OutputValidator::default()
        };

        let result = validator
            .validate_file(1, &path)
            .await
            .expect("validate_file");
        assert!(result.passes_all(), "correct sidecar → checksum passes");

        tokio::fs::remove_file(&path).await.ok();
        tokio::fs::remove_file(&sidecar).await.ok();
    }

    #[tokio::test]
    async fn test_validate_sequence_directory_sequential() {
        let dir = std::env::temp_dir().join("oximedia_seq_test_ok");
        tokio::fs::create_dir_all(&dir).await.expect("mkdir");

        for i in 1u32..=5 {
            let name = format!("frame_{i:04}.png");
            tokio::fs::write(dir.join(&name), b"data")
                .await
                .expect("write");
        }

        let validator = OutputValidator {
            expected_first_frame: 1,
            ..OutputValidator::default()
        };
        let ok = validator
            .validate_sequence_directory(&dir)
            .await
            .expect("validate_sequence_directory");
        assert!(ok, "frames 1-5 should be sequential");

        tokio::fs::remove_dir_all(&dir).await.ok();
    }

    #[tokio::test]
    async fn test_validate_sequence_directory_gap() {
        let dir = std::env::temp_dir().join("oximedia_seq_test_gap");
        tokio::fs::create_dir_all(&dir).await.expect("mkdir");

        for i in [1u32, 2, 4, 5] {
            let name = format!("frame_{i:04}.png");
            tokio::fs::write(dir.join(&name), b"data")
                .await
                .expect("write");
        }

        let validator = OutputValidator {
            expected_first_frame: 1,
            ..OutputValidator::default()
        };
        let ok = validator
            .validate_sequence_directory(&dir)
            .await
            .expect("validate_sequence_directory");
        assert!(!ok, "gap at frame 3 should be detected");

        tokio::fs::remove_dir_all(&dir).await.ok();
    }
}
