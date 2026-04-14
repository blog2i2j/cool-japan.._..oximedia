//! Sidecar file generation with human-readable checksum manifests.
//!
//! Generates `.sha256`, `.md5`, `.blake3`, and combined `.checksums` sidecar
//! files alongside archived media, following conventions used by media archives,
//! BagIt, and digital preservation workflows.

use crate::{ArchiveError, ArchiveResult};
use serde::{Deserialize, Serialize};
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// SidecarFormat
// ---------------------------------------------------------------------------

/// Output format for a sidecar checksum file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidecarFormat {
    /// Plain `sha256sum`-compatible format: `<hex>  <filename>`.
    Sha256Sum,
    /// Plain `md5sum`-compatible format.
    Md5Sum,
    /// BLAKE3 bsum-compatible format.
    Blake3Sum,
    /// Combined multi-algorithm JSON manifest.
    JsonManifest,
    /// Combined multi-algorithm human-readable text manifest.
    TextManifest,
}

impl SidecarFormat {
    /// File extension (without leading dot) for this sidecar format.
    #[must_use]
    pub fn file_extension(&self) -> &str {
        match self {
            Self::Sha256Sum => "sha256",
            Self::Md5Sum => "md5",
            Self::Blake3Sum => "blake3",
            Self::JsonManifest => "checksums.json",
            Self::TextManifest => "checksums.txt",
        }
    }
}

// ---------------------------------------------------------------------------
// ChecksumEntry — one file's checksums
// ---------------------------------------------------------------------------

/// All available checksums for a single file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChecksumEntry {
    /// Relative path of the file (relative to the archive root).
    pub path: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// SHA-256 hex digest (if computed).
    pub sha256: Option<String>,
    /// MD5 hex digest (if computed).
    pub md5: Option<String>,
    /// BLAKE3 hex digest (if computed).
    pub blake3: Option<String>,
    /// CRC32 hex digest (if computed).
    pub crc32: Option<String>,
    /// ISO-8601 timestamp of when the checksum was recorded.
    pub recorded_at: String,
}

impl ChecksumEntry {
    /// Create a new entry with the given path and size.
    #[must_use]
    pub fn new(path: impl Into<String>, size_bytes: u64) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            path: path.into(),
            size_bytes,
            sha256: None,
            md5: None,
            blake3: None,
            crc32: None,
            recorded_at: now,
        }
    }

    /// Render a single line in `sha256sum` format.
    ///
    /// Returns `None` if no SHA-256 digest is available.
    #[must_use]
    pub fn sha256sum_line(&self) -> Option<String> {
        self.sha256
            .as_ref()
            .map(|h| format!("{}  {}", h, self.path))
    }

    /// Render a single line in `md5sum` format.
    #[must_use]
    pub fn md5sum_line(&self) -> Option<String> {
        self.md5.as_ref().map(|h| format!("{}  {}", h, self.path))
    }

    /// Render a single line in BLAKE3 bsum format.
    #[must_use]
    pub fn blake3sum_line(&self) -> Option<String> {
        self.blake3
            .as_ref()
            .map(|h| format!("{}  {}", h, self.path))
    }

    /// Render a human-readable multi-algorithm text block for this entry.
    #[must_use]
    pub fn to_text_block(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "File:    {}", self.path);
        let _ = writeln!(out, "Size:    {} bytes", self.size_bytes);
        if let Some(ref h) = self.sha256 {
            let _ = writeln!(out, "SHA-256: {h}");
        }
        if let Some(ref h) = self.blake3 {
            let _ = writeln!(out, "BLAKE3:  {h}");
        }
        if let Some(ref h) = self.md5 {
            let _ = writeln!(out, "MD5:     {h}");
        }
        if let Some(ref h) = self.crc32 {
            let _ = writeln!(out, "CRC32:   {h}");
        }
        let _ = writeln!(out, "Recorded:{}", self.recorded_at);
        out
    }
}

// ---------------------------------------------------------------------------
// SidecarManifest — a collection of checksum entries
// ---------------------------------------------------------------------------

/// A manifest of checksum entries for an archive or directory.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SidecarManifest {
    /// Archive or directory name (informational).
    pub archive_name: String,
    /// ISO-8601 timestamp of manifest generation.
    pub generated_at: String,
    /// Tool version string.
    pub generator: String,
    /// All entries.
    pub entries: Vec<ChecksumEntry>,
}

impl SidecarManifest {
    /// Create a new empty manifest.
    #[must_use]
    pub fn new(archive_name: impl Into<String>) -> Self {
        Self {
            archive_name: archive_name.into(),
            generated_at: chrono::Utc::now().to_rfc3339(),
            generator: "oximedia-archive".to_string(),
            entries: Vec::new(),
        }
    }

    /// Add an entry.
    pub fn add(&mut self, entry: ChecksumEntry) {
        self.entries.push(entry);
    }

    /// Number of entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Total bytes across all entries.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.size_bytes).sum()
    }

    /// Render the manifest in the given format.
    ///
    /// Returns the sidecar file content as a `String`.
    pub fn render(&self, format: SidecarFormat) -> ArchiveResult<String> {
        match format {
            SidecarFormat::Sha256Sum => Ok(self.render_sha256sum()),
            SidecarFormat::Md5Sum => Ok(self.render_md5sum()),
            SidecarFormat::Blake3Sum => Ok(self.render_blake3sum()),
            SidecarFormat::JsonManifest => self.render_json(),
            SidecarFormat::TextManifest => Ok(self.render_text()),
        }
    }

    fn render_sha256sum(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# SHA-256 checksums generated by oximedia-archive");
        let _ = writeln!(out, "# Generated: {}", self.generated_at);
        let _ = writeln!(out, "# Archive:   {}", self.archive_name);
        out.push('\n');
        for entry in &self.entries {
            if let Some(line) = entry.sha256sum_line() {
                let _ = writeln!(out, "{line}");
            }
        }
        out
    }

    fn render_md5sum(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# MD5 checksums generated by oximedia-archive");
        let _ = writeln!(out, "# Generated: {}", self.generated_at);
        let _ = writeln!(out, "# Archive:   {}", self.archive_name);
        out.push('\n');
        for entry in &self.entries {
            if let Some(line) = entry.md5sum_line() {
                let _ = writeln!(out, "{line}");
            }
        }
        out
    }

    fn render_blake3sum(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# BLAKE3 checksums generated by oximedia-archive");
        let _ = writeln!(out, "# Generated: {}", self.generated_at);
        let _ = writeln!(out, "# Archive:   {}", self.archive_name);
        out.push('\n');
        for entry in &self.entries {
            if let Some(line) = entry.blake3sum_line() {
                let _ = writeln!(out, "{line}");
            }
        }
        out
    }

    fn render_json(&self) -> ArchiveResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            ArchiveError::Validation(format!("sidecar JSON serialization failed: {e}"))
        })
    }

    fn render_text(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "========================================");
        let _ = writeln!(out, " CHECKSUM MANIFEST");
        let _ = writeln!(out, " Archive:   {}", self.archive_name);
        let _ = writeln!(out, " Generated: {}", self.generated_at);
        let _ = writeln!(out, " Tool:      {}", self.generator);
        let _ = writeln!(
            out,
            " Files:     {} ({} bytes total)",
            self.entries.len(),
            self.total_bytes()
        );
        let _ = writeln!(out, "========================================");
        out.push('\n');
        for entry in &self.entries {
            out.push_str(&entry.to_text_block());
            out.push('\n');
        }
        out
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> ArchiveResult<Self> {
        serde_json::from_str(json).map_err(|e| {
            ArchiveError::Validation(format!("sidecar JSON deserialization failed: {e}"))
        })
    }
}

// ---------------------------------------------------------------------------
// SidecarGenerator — writes sidecar files to disk
// ---------------------------------------------------------------------------

/// Configuration for sidecar generation.
#[derive(Debug, Clone)]
pub struct SidecarConfig {
    /// Generate SHA-256 `.sha256` sidecar.
    pub generate_sha256: bool,
    /// Generate MD5 `.md5` sidecar.
    pub generate_md5: bool,
    /// Generate BLAKE3 `.blake3` sidecar.
    pub generate_blake3: bool,
    /// Generate combined JSON manifest.
    pub generate_json: bool,
    /// Generate combined text manifest.
    pub generate_text: bool,
    /// Place sidecars next to the source file (instead of in a dedicated dir).
    pub inline: bool,
    /// Directory to place sidecars in when `inline` is false.
    pub sidecar_dir: Option<PathBuf>,
}

impl Default for SidecarConfig {
    fn default() -> Self {
        Self {
            generate_sha256: true,
            generate_md5: false,
            generate_blake3: true,
            generate_json: true,
            generate_text: true,
            inline: false,
            sidecar_dir: None,
        }
    }
}

/// Writes sidecar files for a `SidecarManifest`.
pub struct SidecarGenerator {
    config: SidecarConfig,
}

impl SidecarGenerator {
    /// Create a new generator with the given config.
    #[must_use]
    pub fn new(config: SidecarConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(SidecarConfig::default())
    }

    /// Write all configured sidecar formats for `manifest` to disk.
    ///
    /// `base_path` is the directory containing the archived files.
    /// Returns paths of all written sidecar files.
    pub fn write_sidecars(
        &self,
        manifest: &SidecarManifest,
        base_path: &Path,
    ) -> ArchiveResult<Vec<PathBuf>> {
        let sidecar_dir = if self.config.inline {
            base_path.to_path_buf()
        } else {
            match &self.config.sidecar_dir {
                Some(d) => d.clone(),
                None => base_path.join("sidecars"),
            }
        };

        std::fs::create_dir_all(&sidecar_dir)?;

        let archive_stem = sanitize_filename(&manifest.archive_name);
        let mut written = Vec::new();

        if self.config.generate_sha256 {
            let content = manifest.render(SidecarFormat::Sha256Sum)?;
            let path = sidecar_dir.join(format!("{archive_stem}.sha256"));
            std::fs::write(&path, content.as_bytes())?;
            written.push(path);
        }

        if self.config.generate_md5 {
            let content = manifest.render(SidecarFormat::Md5Sum)?;
            let path = sidecar_dir.join(format!("{archive_stem}.md5"));
            std::fs::write(&path, content.as_bytes())?;
            written.push(path);
        }

        if self.config.generate_blake3 {
            let content = manifest.render(SidecarFormat::Blake3Sum)?;
            let path = sidecar_dir.join(format!("{archive_stem}.blake3"));
            std::fs::write(&path, content.as_bytes())?;
            written.push(path);
        }

        if self.config.generate_json {
            let content = manifest.render(SidecarFormat::JsonManifest)?;
            let path = sidecar_dir.join(format!("{archive_stem}.checksums.json"));
            std::fs::write(&path, content.as_bytes())?;
            written.push(path);
        }

        if self.config.generate_text {
            let content = manifest.render(SidecarFormat::TextManifest)?;
            let path = sidecar_dir.join(format!("{archive_stem}.checksums.txt"));
            std::fs::write(&path, content.as_bytes())?;
            written.push(path);
        }

        Ok(written)
    }

    /// Compute checksums for a file's byte content and return a `ChecksumEntry`.
    ///
    /// Computes whichever algorithms are enabled in the config.
    pub fn compute_entry(&self, path: &str, data: &[u8]) -> ChecksumEntry {
        let mut entry = ChecksumEntry::new(path, data.len() as u64);

        if self.config.generate_sha256 {
            use sha2::Digest as _;
            let mut hasher = sha2::Sha256::new();
            hasher.update(data);
            entry.sha256 = Some(hex::encode(hasher.finalize()));
        }

        if self.config.generate_md5 {
            use md5::Digest as _;
            let mut hasher = md5::Md5::new();
            hasher.update(data);
            entry.md5 = Some(hex::encode(hasher.finalize()));
        }

        if self.config.generate_blake3 {
            entry.blake3 = Some(blake3::hash(data).to_hex().to_string());
        }

        entry
    }
}

// ---------------------------------------------------------------------------
// Parse sha256sum-format sidecar files
// ---------------------------------------------------------------------------

/// Parse a sha256sum-format file into `(hex_digest, filename)` pairs.
///
/// Lines starting with `#` are treated as comments and skipped. Empty lines
/// are ignored.
pub fn parse_sha256sum_file(content: &str) -> Vec<(String, String)> {
    content
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .filter_map(|line| {
            // Format: `<hex>  <filename>` (two spaces) or `<hex> <filename>` (one space)
            let trimmed = line.trim();
            let split_pos = trimmed.find("  ").or_else(|| trimmed.find(' '))?;
            let hex = trimmed[..split_pos].to_string();
            let filename = trimmed[split_pos..].trim().to_string();
            if hex.is_empty() || filename.is_empty() {
                None
            } else {
                Some((hex, filename))
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c => c,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(path: &str, size: u64, sha256: &str) -> ChecksumEntry {
        ChecksumEntry {
            path: path.to_string(),
            size_bytes: size,
            sha256: Some(sha256.to_string()),
            md5: Some("abc123".to_string()),
            blake3: Some("def456".to_string()),
            crc32: Some("00aabbcc".to_string()),
            recorded_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    // --- SidecarFormat ---

    #[test]
    fn test_sidecar_format_extensions() {
        assert_eq!(SidecarFormat::Sha256Sum.file_extension(), "sha256");
        assert_eq!(SidecarFormat::Md5Sum.file_extension(), "md5");
        assert_eq!(SidecarFormat::Blake3Sum.file_extension(), "blake3");
        assert_eq!(
            SidecarFormat::JsonManifest.file_extension(),
            "checksums.json"
        );
        assert_eq!(
            SidecarFormat::TextManifest.file_extension(),
            "checksums.txt"
        );
    }

    // --- ChecksumEntry ---

    #[test]
    fn test_entry_sha256sum_line() {
        let e = make_entry("clip.mxf", 1024, "abcdef01234567");
        let line = e.sha256sum_line().expect("sha256sum line");
        assert_eq!(line, "abcdef01234567  clip.mxf");
    }

    #[test]
    fn test_entry_md5sum_line() {
        let e = make_entry("clip.mxf", 1024, "aaa");
        let line = e.md5sum_line().expect("md5sum line");
        assert!(line.contains("abc123"));
        assert!(line.contains("clip.mxf"));
    }

    #[test]
    fn test_entry_blake3sum_line() {
        let e = make_entry("clip.mxf", 1024, "aaa");
        let line = e.blake3sum_line().expect("blake3sum line");
        assert!(line.contains("def456"));
    }

    #[test]
    fn test_entry_to_text_block() {
        let e = make_entry("video.mxf", 4096, "sha256hex");
        let text = e.to_text_block();
        assert!(text.contains("video.mxf"));
        assert!(text.contains("4096 bytes"));
        assert!(text.contains("sha256hex"));
    }

    #[test]
    fn test_entry_no_checksums() {
        let e = ChecksumEntry::new("empty.bin", 0);
        assert!(e.sha256sum_line().is_none());
        assert!(e.md5sum_line().is_none());
        assert!(e.blake3sum_line().is_none());
    }

    // --- SidecarManifest ---

    #[test]
    fn test_manifest_new_empty() {
        let m = SidecarManifest::new("my-archive");
        assert_eq!(m.entry_count(), 0);
        assert_eq!(m.total_bytes(), 0);
        assert_eq!(m.archive_name, "my-archive");
    }

    #[test]
    fn test_manifest_add_and_count() {
        let mut m = SidecarManifest::new("archive");
        m.add(make_entry("a.mxf", 1000, "aaa"));
        m.add(make_entry("b.mxf", 2000, "bbb"));
        assert_eq!(m.entry_count(), 2);
        assert_eq!(m.total_bytes(), 3000);
    }

    #[test]
    fn test_manifest_render_sha256sum() {
        let mut m = SidecarManifest::new("test-arc");
        m.add(make_entry("clip.mxf", 1024, "deadbeef"));
        let content = m.render(SidecarFormat::Sha256Sum).expect("render sha256");
        assert!(content.contains("deadbeef  clip.mxf"));
        assert!(content.contains("# SHA-256"));
    }

    #[test]
    fn test_manifest_render_md5sum() {
        let mut m = SidecarManifest::new("test-arc");
        m.add(make_entry("a.mxf", 100, "sha"));
        let content = m.render(SidecarFormat::Md5Sum).expect("render md5");
        assert!(content.contains("abc123  a.mxf"));
    }

    #[test]
    fn test_manifest_render_blake3sum() {
        let mut m = SidecarManifest::new("test-arc");
        m.add(make_entry("a.mxf", 100, "sha"));
        let content = m.render(SidecarFormat::Blake3Sum).expect("render blake3");
        assert!(content.contains("def456  a.mxf"));
    }

    #[test]
    fn test_manifest_render_json_roundtrip() {
        let mut m = SidecarManifest::new("json-arc");
        m.add(make_entry("video.mkv", 8192, "sha256abc"));
        let json = m.render(SidecarFormat::JsonManifest).expect("render json");

        let restored = SidecarManifest::from_json(&json).expect("from json");
        assert_eq!(restored.archive_name, "json-arc");
        assert_eq!(restored.entry_count(), 1);
        assert_eq!(restored.entries[0].path, "video.mkv");
        assert_eq!(restored.entries[0].sha256.as_deref(), Some("sha256abc"));
    }

    #[test]
    fn test_manifest_render_text_contains_header() {
        let m = SidecarManifest::new("text-arc");
        let content = m.render(SidecarFormat::TextManifest).expect("render text");
        assert!(content.contains("CHECKSUM MANIFEST"));
        assert!(content.contains("text-arc"));
    }

    #[test]
    fn test_manifest_render_text_contains_entries() {
        let mut m = SidecarManifest::new("arc");
        m.add(make_entry("film.mkv", 2048, "film_sha"));
        let content = m.render(SidecarFormat::TextManifest).expect("render text");
        assert!(content.contains("film.mkv"));
        assert!(content.contains("film_sha"));
    }

    // --- SidecarGenerator ---

    #[test]
    fn test_generator_compute_entry_sha256() {
        let cfg = SidecarConfig {
            generate_sha256: true,
            generate_md5: false,
            generate_blake3: false,
            generate_json: false,
            generate_text: false,
            inline: true,
            sidecar_dir: None,
        };
        let gen = SidecarGenerator::new(cfg);
        let data = b"hello sidecar world";
        let entry = gen.compute_entry("test.bin", data);
        assert_eq!(entry.size_bytes, data.len() as u64);
        assert!(entry.sha256.is_some());
        assert!(entry.md5.is_none());
        assert!(entry.blake3.is_none());
    }

    #[test]
    fn test_generator_compute_entry_all_algorithms() {
        let gen = SidecarGenerator::with_defaults();
        let data = b"multi-algorithm checksum data";
        let entry = gen.compute_entry("multi.bin", data);
        assert!(entry.sha256.is_some());
        assert!(entry.blake3.is_some());
        // default config has md5 disabled
        assert!(entry.md5.is_none());
    }

    #[test]
    fn test_generator_sha256_known_vector() {
        let cfg = SidecarConfig {
            generate_sha256: true,
            generate_md5: false,
            generate_blake3: false,
            generate_json: false,
            generate_text: false,
            inline: true,
            sidecar_dir: None,
        };
        let gen = SidecarGenerator::new(cfg);
        let entry = gen.compute_entry("abc.bin", b"abc");
        assert_eq!(
            entry.sha256.as_deref(),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
    }

    #[test]
    fn test_generator_write_sidecars() {
        let dir = std::env::temp_dir().join("oximedia_sidecar_write_test");
        std::fs::create_dir_all(&dir).ok();

        let cfg = SidecarConfig {
            generate_sha256: true,
            generate_md5: false,
            generate_blake3: true,
            generate_json: true,
            generate_text: true,
            inline: false,
            sidecar_dir: Some(dir.join("sidecars")),
        };
        let gen = SidecarGenerator::new(cfg);

        let mut manifest = SidecarManifest::new("my-archive");
        manifest.add(make_entry("clip.mxf", 1024, "aaabbbccc"));

        let written = gen.write_sidecars(&manifest, &dir).expect("write sidecars");
        assert_eq!(written.len(), 4); // sha256, blake3, json, text

        for path in &written {
            assert!(path.exists(), "sidecar not written: {}", path.display());
            let content = std::fs::read_to_string(path).expect("read sidecar");
            assert!(!content.is_empty());
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_generator_write_inline_sidecars() {
        let dir = std::env::temp_dir().join("oximedia_sidecar_inline_test");
        std::fs::create_dir_all(&dir).ok();

        let cfg = SidecarConfig {
            generate_sha256: true,
            generate_md5: false,
            generate_blake3: false,
            generate_json: false,
            generate_text: false,
            inline: true,
            sidecar_dir: None,
        };
        let gen = SidecarGenerator::new(cfg);

        let mut manifest = SidecarManifest::new("inline-arc");
        manifest.add(make_entry("file.bin", 512, "sha"));

        let written = gen.write_sidecars(&manifest, &dir).expect("write inline");
        assert_eq!(written.len(), 1);
        assert!(written[0].starts_with(&dir));

        std::fs::remove_dir_all(&dir).ok();
    }

    // --- parse_sha256sum_file ---

    #[test]
    fn test_parse_sha256sum_basic() {
        let content = "abcdef1234567890  file.mxf\n1234567890abcdef  dir/clip.mkv\n";
        let entries = parse_sha256sum_file(content);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "abcdef1234567890");
        assert_eq!(entries[0].1, "file.mxf");
        assert_eq!(entries[1].1, "dir/clip.mkv");
    }

    #[test]
    fn test_parse_sha256sum_skips_comments() {
        let content = "# This is a comment\nabc123  file.bin\n# Another comment\n";
        let entries = parse_sha256sum_file(content);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].1, "file.bin");
    }

    #[test]
    fn test_parse_sha256sum_skips_empty_lines() {
        let content = "abc  a.bin\n\n\ndef  b.bin\n";
        let entries = parse_sha256sum_file(content);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_parse_sha256sum_empty_input() {
        let entries = parse_sha256sum_file("");
        assert!(entries.is_empty());
    }

    #[test]
    fn test_sanitize_filename_replaces_special_chars() {
        let name = "archive:v1/test*file";
        let sanitized = sanitize_filename(name);
        assert!(!sanitized.contains(':'));
        assert!(!sanitized.contains('/'));
        assert!(!sanitized.contains('*'));
    }

    // --- Round-trip: generate entry from data, then verify via sha256sum parse ---

    #[test]
    fn test_sidecar_roundtrip_via_sha256sum() {
        let gen = SidecarGenerator::new(SidecarConfig {
            generate_sha256: true,
            generate_md5: false,
            generate_blake3: false,
            generate_json: false,
            generate_text: false,
            inline: true,
            sidecar_dir: None,
        });
        let data = b"round-trip test content for sidecar";
        let entry = gen.compute_entry("round.bin", data);

        let mut manifest = SidecarManifest::new("rt-arc");
        manifest.add(entry.clone());

        let sha256content = manifest.render(SidecarFormat::Sha256Sum).expect("render");
        let parsed = parse_sha256sum_file(&sha256content);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].0, entry.sha256.as_deref().expect("sha256 set"));
        assert_eq!(parsed[0].1, "round.bin");
    }

    // --- New tests for sidecar generation (implementation items) ---

    #[test]
    fn test_manifest_generator_field() {
        let m = SidecarManifest::new("test-arc");
        assert_eq!(m.generator, "oximedia-archive");
    }

    #[test]
    fn test_manifest_generated_at_is_rfc3339() {
        let m = SidecarManifest::new("test-arc");
        // Should be parseable as RFC-3339
        let parsed = chrono::DateTime::parse_from_rfc3339(&m.generated_at);
        assert!(
            parsed.is_ok(),
            "generated_at should be valid RFC-3339: {}",
            m.generated_at
        );
    }

    #[test]
    fn test_entry_recorded_at_is_rfc3339() {
        let e = ChecksumEntry::new("file.bin", 100);
        let parsed = chrono::DateTime::parse_from_rfc3339(&e.recorded_at);
        assert!(
            parsed.is_ok(),
            "recorded_at should be valid RFC-3339: {}",
            e.recorded_at
        );
    }

    #[test]
    fn test_manifest_total_bytes_multiple_entries() {
        let mut m = SidecarManifest::new("arc");
        m.add(ChecksumEntry::new("a.bin", 1000));
        m.add(ChecksumEntry::new("b.bin", 2000));
        m.add(ChecksumEntry::new("c.bin", 3000));
        assert_eq!(m.total_bytes(), 6000);
    }

    #[test]
    fn test_render_text_includes_size_total() {
        let mut m = SidecarManifest::new("size-arc");
        m.add(ChecksumEntry::new("a.bin", 512));
        m.add(ChecksumEntry::new("b.bin", 1024));
        let content = m.render(SidecarFormat::TextManifest).expect("render text");
        // Total bytes (512 + 1024 = 1536) should appear in the header
        assert!(
            content.contains("1536 bytes"),
            "expected total bytes in header: {content}"
        );
    }

    #[test]
    fn test_sha256sum_comment_lines_skipped_in_parse() {
        let content = "# Auto-generated\n# Generator: oximedia\nabc  file.bin\n";
        let entries = parse_sha256sum_file(content);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].1, "file.bin");
    }

    #[test]
    fn test_generator_compute_entry_blake3_known_vector() {
        let cfg = SidecarConfig {
            generate_sha256: false,
            generate_md5: false,
            generate_blake3: true,
            generate_json: false,
            generate_text: false,
            inline: true,
            sidecar_dir: None,
        };
        let gen = SidecarGenerator::new(cfg);
        let data = b"hello";
        let entry = gen.compute_entry("hello.bin", data);
        let expected = blake3::hash(data).to_hex().to_string();
        assert_eq!(entry.blake3.as_deref(), Some(expected.as_str()));
    }

    #[test]
    fn test_manifest_json_has_all_fields() {
        let mut m = SidecarManifest::new("json-fields");
        m.add(make_entry("test.mxf", 4096, "abcdef"));
        let json = m.render(SidecarFormat::JsonManifest).expect("render json");
        // Verify key fields appear in JSON
        assert!(json.contains("\"archive_name\""), "missing archive_name");
        assert!(json.contains("\"generator\""), "missing generator");
        assert!(json.contains("\"generated_at\""), "missing generated_at");
        assert!(json.contains("\"entries\""), "missing entries");
    }

    #[test]
    fn test_sidecar_config_default_disables_md5() {
        let cfg = SidecarConfig::default();
        assert!(!cfg.generate_md5, "MD5 should be disabled by default");
        assert!(cfg.generate_sha256, "SHA-256 should be enabled by default");
        assert!(cfg.generate_blake3, "BLAKE3 should be enabled by default");
    }

    #[test]
    fn test_parse_sha256sum_single_space_separator() {
        // Some tools emit one space instead of two
        let content = "abc123 file.bin\n";
        let entries = parse_sha256sum_file(content);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "abc123");
    }

    #[test]
    fn test_sidecar_write_directory_created() {
        let dir = std::env::temp_dir().join("oximedia_sidecar_dir_create_test");
        // Ensure dir does NOT exist beforehand
        let _ = std::fs::remove_dir_all(&dir);

        let sidecar_subdir = dir.join("sidecars");
        let cfg = SidecarConfig {
            generate_sha256: true,
            generate_md5: false,
            generate_blake3: false,
            generate_json: false,
            generate_text: false,
            inline: false,
            sidecar_dir: Some(sidecar_subdir.clone()),
        };
        let gen = SidecarGenerator::new(cfg);
        let manifest = SidecarManifest::new("dir-create-arc");
        gen.write_sidecars(&manifest, &dir)
            .expect("write should create dir");
        assert!(sidecar_subdir.exists(), "sidecar dir should be created");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
