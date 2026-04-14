//! LOCKSS/CLOCKSS-compatible preservation package generation.
//!
//! LOCKSS (Lots of Copies Keep Stuff Safe) and CLOCKSS (Controlled LOCKSS)
//! are distributed preservation networks that validate and replicate digital
//! content using a majority-vote integrity mechanism.
//!
//! This module generates packages in the format expected by LOCKSS and CLOCKSS
//! ingest systems: a content directory with a manifest (`lockss.xml`) and
//! checksum files conforming to the LOCKSS WARC/AU (Archival Unit) model.
//!
//! References:
//! - LOCKSS Technical Overview: <https://www.lockss.org/technology/>
//! - CLOCKSS: <https://clockss.org/>

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Identifies the LOCKSS network variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LockssNetwork {
    /// Public LOCKSS network (open-access content).
    Lockss,
    /// Controlled LOCKSS network (triggered access, embargoed content).
    Clockss,
}

impl LockssNetwork {
    /// Returns the conventional prefix string for this network.
    #[must_use]
    pub const fn prefix(&self) -> &'static str {
        match self {
            Self::Lockss => "LOCKSS",
            Self::Clockss => "CLOCKSS",
        }
    }
}

/// Metadata for a LOCKSS Archival Unit (AU).
///
/// An AU is the smallest unit of content tracked by LOCKSS.  All files within
/// one AU are stored, validated, and replicated together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockssAuMetadata {
    /// Globally unique AU identifier (typically a URN or DOI-derived string).
    pub au_id: String,
    /// Human-readable title of the AU.
    pub title: String,
    /// Publisher name.
    pub publisher: String,
    /// Publication year.
    pub year: Option<u32>,
    /// ISSN or ISSNL of the journal/series (if applicable).
    pub issn: Option<String>,
    /// Volume number (if applicable).
    pub volume: Option<String>,
    /// Base URL for content retrieval (required by LOCKSS daemon).
    pub base_url: String,
    /// Additional parameters stored as key-value pairs.
    pub params: HashMap<String, String>,
    /// LOCKSS network variant.
    pub network: LockssNetwork,
}

impl LockssAuMetadata {
    /// Creates a new AU metadata record.
    #[must_use]
    pub fn new(
        au_id: impl Into<String>,
        title: impl Into<String>,
        publisher: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            au_id: au_id.into(),
            title: title.into(),
            publisher: publisher.into(),
            year: None,
            issn: None,
            volume: None,
            base_url: base_url.into(),
            params: HashMap::new(),
            network: LockssNetwork::Lockss,
        }
    }

    /// Sets the publication year.
    #[must_use]
    pub const fn with_year(mut self, year: u32) -> Self {
        self.year = Some(year);
        self
    }

    /// Sets the ISSN.
    #[must_use]
    pub fn with_issn(mut self, issn: impl Into<String>) -> Self {
        self.issn = Some(issn.into());
        self
    }

    /// Sets the volume.
    #[must_use]
    pub fn with_volume(mut self, volume: impl Into<String>) -> Self {
        self.volume = Some(volume.into());
        self
    }

    /// Sets the network variant (LOCKSS or CLOCKSS).
    #[must_use]
    pub const fn with_network(mut self, network: LockssNetwork) -> Self {
        self.network = network;
        self
    }

    /// Adds an arbitrary parameter.
    #[must_use]
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }
}

/// A content file entry within a LOCKSS Archival Unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockssFile {
    /// Path within the AU (relative to the AU root).
    pub path: PathBuf,
    /// SHA-256 checksum hex string.
    pub sha256: String,
    /// File size in bytes.
    pub size: u64,
    /// Last-modified timestamp (Unix epoch seconds).
    pub mtime: i64,
}

/// A generated LOCKSS/CLOCKSS Archival Unit package on disk.
#[derive(Debug)]
pub struct LockssPackage {
    /// Root directory of the generated package.
    pub root: PathBuf,
    /// AU metadata.
    pub metadata: LockssAuMetadata,
    /// Content files included in the package.
    pub files: Vec<LockssFile>,
}

impl LockssPackage {
    /// Returns the path to the `lockss.xml` manifest within this package.
    #[must_use]
    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("lockss.xml")
    }

    /// Returns the path to the SHA-256 checksum manifest.
    #[must_use]
    pub fn checksum_manifest_path(&self) -> PathBuf {
        self.root.join("sha256manifest.txt")
    }

    /// Returns the total size of all content files in bytes.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

/// Builds a LOCKSS/CLOCKSS-compatible Archival Unit package.
pub struct LockssPackageBuilder {
    output_root: PathBuf,
    metadata: LockssAuMetadata,
    source_files: Vec<(PathBuf, PathBuf)>, // (source_path, relative_dest_path)
}

impl LockssPackageBuilder {
    /// Creates a new package builder.
    ///
    /// * `output_root` – directory where the AU package will be written.
    /// * `metadata`    – AU-level metadata.
    #[must_use]
    pub fn new(output_root: impl Into<PathBuf>, metadata: LockssAuMetadata) -> Self {
        Self {
            output_root: output_root.into(),
            metadata,
            source_files: Vec::new(),
        }
    }

    /// Adds a file to the package.
    ///
    /// * `source`       – filesystem path to the source file.
    /// * `dest_path`    – relative path within the AU (e.g. `"content/video.mkv"`).
    ///
    /// # Errors
    ///
    /// Returns an error if the source file cannot be accessed.
    pub fn add_file(
        mut self,
        source: &Path,
        dest_path: &Path,
    ) -> Result<Self, crate::Error> {
        if !source.exists() {
            return Err(crate::Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Source file not found: {}", source.display()),
            )));
        }
        self.source_files.push((source.to_owned(), dest_path.to_owned()));
        Ok(self)
    }

    /// Builds and writes the LOCKSS AU package to disk.
    ///
    /// Creates the output directory, copies content files, computes SHA-256
    /// checksums, writes `lockss.xml` and `sha256manifest.txt`.
    ///
    /// # Errors
    ///
    /// Returns an error if any I/O operation fails.
    pub fn build(self) -> Result<LockssPackage, crate::Error> {
        use sha2::{Digest, Sha256};
        use std::io::Read;

        std::fs::create_dir_all(&self.output_root)?;
        let content_dir = self.output_root.join("content");
        std::fs::create_dir_all(&content_dir)?;

        let mut lockss_files = Vec::new();

        for (source, dest_rel) in &self.source_files {
            let dest_full = content_dir.join(dest_rel);
            if let Some(parent) = dest_full.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(source, &dest_full)?;

            // Compute SHA-256 checksum
            let mut file = std::fs::File::open(&dest_full)?;
            let mut hasher = Sha256::new();
            let mut buf = vec![0u8; 64 * 1024];
            loop {
                let n = file.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
            }
            let hash = hex::encode(hasher.finalize());

            let meta = std::fs::metadata(&dest_full)?;
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);

            lockss_files.push(LockssFile {
                path: Path::new("content").join(dest_rel),
                sha256: hash,
                size: meta.len(),
                mtime,
            });
        }

        // Write lockss.xml manifest
        let manifest_xml = Self::generate_manifest_xml(&self.metadata, &lockss_files);
        std::fs::write(self.output_root.join("lockss.xml"), &manifest_xml)?;

        // Write SHA-256 checksum manifest
        let checksum_txt = Self::generate_checksum_manifest(&lockss_files);
        std::fs::write(self.output_root.join("sha256manifest.txt"), &checksum_txt)?;

        Ok(LockssPackage {
            root: self.output_root,
            metadata: self.metadata,
            files: lockss_files,
        })
    }

    fn generate_manifest_xml(meta: &LockssAuMetadata, files: &[LockssFile]) -> String {
        let mut xml = String::new();
        xml.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
        xml.push('\n');
        xml.push_str("<lockss-au>\n");
        xml.push_str(&format!("  <au_id>{}</au_id>\n", Self::xml_escape(&meta.au_id)));
        xml.push_str(&format!("  <title>{}</title>\n", Self::xml_escape(&meta.title)));
        xml.push_str(&format!("  <publisher>{}</publisher>\n", Self::xml_escape(&meta.publisher)));
        xml.push_str(&format!("  <base_url>{}</base_url>\n", Self::xml_escape(&meta.base_url)));
        xml.push_str(&format!("  <network>{}</network>\n", meta.network.prefix()));
        if let Some(y) = meta.year {
            xml.push_str(&format!("  <year>{y}</year>\n"));
        }
        if let Some(ref issn) = meta.issn {
            xml.push_str(&format!("  <issn>{}</issn>\n", Self::xml_escape(issn)));
        }
        if let Some(ref vol) = meta.volume {
            xml.push_str(&format!("  <volume>{}</volume>\n", Self::xml_escape(vol)));
        }
        for (k, v) in &meta.params {
            xml.push_str(&format!(
                r#"  <param key="{}" value="{}"/>"#,
                Self::xml_escape(k),
                Self::xml_escape(v)
            ));
            xml.push('\n');
        }
        xml.push_str("  <files>\n");
        for f in files {
            xml.push_str(&format!(
                r#"    <file path="{}" sha256="{}" size="{}"/>"#,
                Self::xml_escape(&f.path.to_string_lossy()),
                f.sha256,
                f.size
            ));
            xml.push('\n');
        }
        xml.push_str("  </files>\n");
        xml.push_str("</lockss-au>\n");
        xml
    }

    fn generate_checksum_manifest(files: &[LockssFile]) -> String {
        let mut lines = Vec::new();
        lines.push("# SHA-256 checksum manifest (LOCKSS/CLOCKSS AU)".to_string());
        for f in files {
            lines.push(format!("{}  {}", f.sha256, f.path.to_string_lossy()));
        }
        lines.join("\n") + "\n"
    }

    fn xml_escape(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn make_au_meta(id: &str) -> LockssAuMetadata {
        LockssAuMetadata::new(
            id,
            "Test Archival Unit",
            "Test Publisher",
            "https://example.org/lockss/au/",
        )
        .with_year(2024)
        .with_issn("1234-5678")
        .with_network(LockssNetwork::Clockss)
    }

    #[test]
    fn test_lockss_package_created() {
        let tmp = TempDir::new().expect("temp dir");
        let pkg_dir = tmp.path().join("au1");

        let mut content = tempfile::NamedTempFile::new().expect("tmp");
        content.write_all(b"media content").expect("write");
        content.flush().expect("flush");

        let meta = make_au_meta("urn:test:au-001");
        let pkg = LockssPackageBuilder::new(&pkg_dir, meta)
            .add_file(content.path(), Path::new("video.mkv"))
            .expect("add_file")
            .build()
            .expect("build");

        assert!(pkg.root.exists());
        assert!(pkg.manifest_path().exists(), "lockss.xml must exist");
        assert!(pkg.checksum_manifest_path().exists(), "sha256manifest.txt must exist");
        assert_eq!(pkg.files.len(), 1);
        assert!(pkg.total_bytes() > 0);
    }

    #[test]
    fn test_lockss_xml_contains_au_id() {
        let tmp = TempDir::new().expect("temp dir");
        let pkg_dir = tmp.path().join("au2");

        let mut f = tempfile::NamedTempFile::new().expect("tmp");
        f.write_all(b"data").expect("write");
        f.flush().expect("flush");

        let meta = make_au_meta("urn:test:au-002");
        let pkg = LockssPackageBuilder::new(&pkg_dir, meta)
            .add_file(f.path(), Path::new("data.bin"))
            .expect("add")
            .build()
            .expect("build");

        let xml = std::fs::read_to_string(pkg.manifest_path()).expect("read");
        assert!(xml.contains("urn:test:au-002"), "AU ID must be in manifest");
        assert!(xml.contains("CLOCKSS"), "network must be recorded");
    }

    #[test]
    fn test_checksum_manifest_contains_sha256() {
        let tmp = TempDir::new().expect("temp dir");
        let pkg_dir = tmp.path().join("au3");

        let mut f = tempfile::NamedTempFile::new().expect("tmp");
        f.write_all(b"checksum test").expect("write");
        f.flush().expect("flush");

        let meta = make_au_meta("urn:test:au-003");
        let pkg = LockssPackageBuilder::new(&pkg_dir, meta)
            .add_file(f.path(), Path::new("check.bin"))
            .expect("add")
            .build()
            .expect("build");

        let txt = std::fs::read_to_string(pkg.checksum_manifest_path()).expect("read");
        assert!(txt.contains("check.bin"), "filename must appear in checksum manifest");
        // SHA-256 hex is 64 characters
        let sha_line = txt.lines().find(|l| l.contains("check.bin")).expect("line");
        let hex_part = sha_line.split_whitespace().next().expect("hex");
        assert_eq!(hex_part.len(), 64, "SHA-256 hex must be 64 chars");
    }

    #[test]
    fn test_lockss_network_prefix() {
        assert_eq!(LockssNetwork::Lockss.prefix(), "LOCKSS");
        assert_eq!(LockssNetwork::Clockss.prefix(), "CLOCKSS");
    }

    #[test]
    fn test_missing_source_file_returns_error() {
        let tmp = TempDir::new().expect("temp dir");
        let meta = make_au_meta("urn:test:err");
        let result = LockssPackageBuilder::new(tmp.path().join("pkg"), meta)
            .add_file(Path::new("/nonexistent/file.mkv"), Path::new("file.mkv"));
        assert!(result.is_err());
    }

    #[test]
    fn test_lockss_metadata_year_and_issn_in_xml() {
        let tmp = TempDir::new().expect("temp dir");
        let pkg_dir = tmp.path().join("au4");

        let mut f = tempfile::NamedTempFile::new().expect("tmp");
        f.write_all(b"data for issn test").expect("write");
        f.flush().expect("flush");

        let meta = LockssAuMetadata::new(
            "urn:test:au-004",
            "Annual Report 2024",
            "Test Publisher",
            "https://example.org/lockss/",
        )
        .with_year(2024)
        .with_issn("9876-5432")
        .with_volume("7");

        let pkg = LockssPackageBuilder::new(&pkg_dir, meta)
            .add_file(f.path(), Path::new("report.pdf"))
            .expect("add")
            .build()
            .expect("build");

        let xml = std::fs::read_to_string(pkg.manifest_path()).expect("read");
        assert!(xml.contains("<year>2024</year>"), "year must appear in XML");
        assert!(xml.contains("<issn>9876-5432</issn>"), "ISSN must appear in XML");
        assert!(xml.contains("<volume>7</volume>"), "volume must appear in XML");
    }

    #[test]
    fn test_lockss_multi_file_total_bytes() {
        let tmp = TempDir::new().expect("temp dir");
        let pkg_dir = tmp.path().join("au5");

        let mut f1 = tempfile::NamedTempFile::new().expect("tmp");
        f1.write_all(b"file one content").expect("write");
        f1.flush().expect("flush");

        let mut f2 = tempfile::NamedTempFile::new().expect("tmp");
        f2.write_all(b"file two content longer").expect("write");
        f2.flush().expect("flush");

        let meta = make_au_meta("urn:test:au-005");
        let pkg = LockssPackageBuilder::new(&pkg_dir, meta)
            .add_file(f1.path(), Path::new("a.mkv"))
            .expect("add f1")
            .add_file(f2.path(), Path::new("b.mkv"))
            .expect("add f2")
            .build()
            .expect("build");

        assert_eq!(pkg.files.len(), 2, "should have 2 file entries");
        assert!(pkg.total_bytes() > 0, "total bytes must be positive");
        // Each file should have a distinct SHA-256 (different content)
        let hashes: std::collections::HashSet<_> = pkg.files.iter().map(|f| &f.sha256).collect();
        assert_eq!(hashes.len(), 2, "SHA-256 hashes must differ for different content");
    }

    #[test]
    fn test_lockss_au_metadata_with_param() {
        let meta = LockssAuMetadata::new(
            "urn:test:param",
            "Param Test",
            "Pub",
            "https://example.org/",
        )
        .with_param("journal_id", "test_j")
        .with_param("volume", "42");

        assert_eq!(meta.params.get("journal_id").map(String::as_str), Some("test_j"));
        assert_eq!(meta.params.len(), 2);
    }
}
