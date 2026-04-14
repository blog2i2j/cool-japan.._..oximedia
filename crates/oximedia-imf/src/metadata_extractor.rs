//! Extract all metadata from an IMF package to JSON or XML format.
//!
//! [`MetadataExtractor`] scans a package directory and serialises asset
//! information to JSON or XML without requiring a full CPL parse.
//!
//! # Example
//! ```no_run
//! use oximedia_imf::metadata_extractor::{MetadataExtractor, OutputFormat};
//!
//! let json = MetadataExtractor::extract("/path/to/imp", OutputFormat::Json)
//!     .expect("extraction failed");
//! println!("{json}");
//! ```

#![allow(dead_code, missing_docs)]

use crate::{ImfError, ImfResult};
use std::path::Path;

/// Target serialisation format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Json,
    Xml,
}

/// A single discovered asset.
#[derive(Debug, Clone)]
pub struct AssetMeta {
    pub filename: String,
    pub size_bytes: u64,
    pub extension: String,
}

/// Top-level package metadata.
#[derive(Debug, Clone, Default)]
pub struct PackageMetadata {
    pub package_path: String,
    pub package_id: Option<String>,
    pub creator: Option<String>,
    pub issue_date: Option<String>,
    pub assets: Vec<AssetMeta>,
    pub cpl_count: usize,
    pub pkl_count: usize,
    pub mxf_count: usize,
    pub total_size_bytes: u64,
}

impl PackageMetadata {
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut s = String::from("{\n");
        s.push_str(&format!("  \"packagePath\": {},\n", jstr(&self.package_path)));
        if let Some(ref id) = self.package_id {
            s.push_str(&format!("  \"packageId\": {},\n", jstr(id)));
        }
        if let Some(ref c) = self.creator {
            s.push_str(&format!("  \"creator\": {},\n", jstr(c)));
        }
        if let Some(ref d) = self.issue_date {
            s.push_str(&format!("  \"issueDate\": {},\n", jstr(d)));
        }
        s.push_str(&format!("  \"cplCount\": {},\n", self.cpl_count));
        s.push_str(&format!("  \"pklCount\": {},\n", self.pkl_count));
        s.push_str(&format!("  \"mxfCount\": {},\n", self.mxf_count));
        s.push_str(&format!("  \"totalSizeBytes\": {},\n", self.total_size_bytes));
        s.push_str("  \"assets\": [\n");
        for (i, a) in self.assets.iter().enumerate() {
            let comma = if i + 1 < self.assets.len() { "," } else { "" };
            s.push_str(&format!(
                "    {{\"filename\": {}, \"extension\": {}, \"sizeBytes\": {}}}{}\n",
                jstr(&a.filename), jstr(&a.extension), a.size_bytes, comma
            ));
        }
        s.push_str("  ]\n}\n");
        s
    }

    #[must_use]
    pub fn to_xml(&self) -> String {
        let mut s = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<PackageMetadata>\n");
        s.push_str(&format!("  <PackagePath>{}</PackagePath>\n", xe(&self.package_path)));
        if let Some(ref id) = self.package_id {
            s.push_str(&format!("  <PackageId>{}</PackageId>\n", xe(id)));
        }
        if let Some(ref c) = self.creator {
            s.push_str(&format!("  <Creator>{}</Creator>\n", xe(c)));
        }
        if let Some(ref d) = self.issue_date {
            s.push_str(&format!("  <IssueDate>{}</IssueDate>\n", xe(d)));
        }
        s.push_str(&format!("  <CplCount>{}</CplCount>\n", self.cpl_count));
        s.push_str(&format!("  <PklCount>{}</PklCount>\n", self.pkl_count));
        s.push_str(&format!("  <MxfCount>{}</MxfCount>\n", self.mxf_count));
        s.push_str(&format!("  <TotalSizeBytes>{}</TotalSizeBytes>\n", self.total_size_bytes));
        s.push_str("  <Assets>\n");
        for a in &self.assets {
            s.push_str(&format!(
                "    <Asset filename=\"{}\" extension=\"{}\" sizeBytes=\"{}\"/>\n",
                xa(&a.filename), xa(&a.extension), a.size_bytes
            ));
        }
        s.push_str("  </Assets>\n</PackageMetadata>\n");
        s
    }
}

/// Extracts IMF package metadata and serialises it.
pub struct MetadataExtractor;

impl MetadataExtractor {
    /// Extract metadata from `pkg_path` and return serialised output.
    ///
    /// # Errors
    ///
    /// Returns `ImfError::InvalidPackage` if the path is not a valid directory.
    pub fn extract(pkg_path: impl AsRef<Path>, format: OutputFormat) -> ImfResult<String> {
        let meta = Self::collect(pkg_path)?;
        Ok(match format {
            OutputFormat::Json => meta.to_json(),
            OutputFormat::Xml => meta.to_xml(),
        })
    }

    /// Collect `PackageMetadata` from a package directory.
    ///
    /// # Errors
    ///
    /// Returns `ImfError::InvalidPackage` if path doesn't exist.
    pub fn collect(pkg_path: impl AsRef<Path>) -> ImfResult<PackageMetadata> {
        let pkg_path = pkg_path.as_ref();
        if !pkg_path.exists() || !pkg_path.is_dir() {
            return Err(ImfError::InvalidPackage(format!(
                "Not a valid directory: {}", pkg_path.display()
            )));
        }

        let mut meta = PackageMetadata {
            package_path: pkg_path.to_string_lossy().to_string(),
            ..Default::default()
        };

        let am = pkg_path.join("ASSETMAP.xml");
        if am.exists() {
            if let Ok(content) = std::fs::read_to_string(&am) {
                meta.package_id = extract_attr(&content, "Id");
                meta.creator = extract_attr(&content, "Creator");
                meta.issue_date = extract_attr(&content, "IssueDate");
            }
        }

        let entries = std::fs::read_dir(pkg_path).map_err(|e| ImfError::Other(e.to_string()))?;
        for entry in entries.flatten() {
            let fp = entry.path();
            if !fp.is_file() { continue; }
            let filename = fp.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            let extension = fp.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
            let size_bytes = fp.metadata().map(|m| m.len()).unwrap_or(0);

            match extension.as_str() {
                "xml" => {
                    let upper = filename.to_uppercase();
                    if upper.starts_with("CPL") { meta.cpl_count += 1; }
                    else if upper.starts_with("PKL") { meta.pkl_count += 1; }
                }
                "mxf" => { meta.mxf_count += 1; }
                _ => {}
            }

            meta.total_size_bytes += size_bytes;
            meta.assets.push(AssetMeta { filename, size_bytes, extension });
        }

        meta.assets.sort_by(|a, b| a.filename.cmp(&b.filename));
        Ok(meta)
    }
}

fn jstr(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}
fn xe(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}
fn xa(s: &str) -> String { xe(s).replace('"', "&quot;") }

fn extract_attr(xml: &str, key: &str) -> Option<String> {
    let pat = format!("{key}=\"");
    let start = xml.find(&pat)? + pat.len();
    let end = xml[start..].find('"')? + start;
    Some(xml[start..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pkg(suffix: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("oximedia_me_{suffix}"));
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("ASSETMAP.xml"),
            r#"<AssetMap Id="urn:uuid:abc123" Creator="Suite" IssueDate="2024-01-01"/>"#).ok();
        std::fs::write(dir.join("PKL_001.xml"), b"<PackingList/>").ok();
        std::fs::write(dir.join("CPL_001.xml"), b"<CompositionPlaylist/>").ok();
        std::fs::write(dir.join("video.mxf"), b"fake mxf").ok();
        dir
    }

    #[test]
    fn test_collect_counts() {
        let dir = make_pkg("counts");
        let meta = MetadataExtractor::collect(&dir).expect("ok");
        assert_eq!(meta.cpl_count, 1);
        assert_eq!(meta.pkl_count, 1);
        assert_eq!(meta.mxf_count, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_collect_assetmap_attrs() {
        let dir = make_pkg("attrs");
        let meta = MetadataExtractor::collect(&dir).expect("ok");
        assert_eq!(meta.package_id.as_deref(), Some("urn:uuid:abc123"));
        assert_eq!(meta.creator.as_deref(), Some("Suite"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_extract_json() {
        let dir = make_pkg("json");
        let json = MetadataExtractor::extract(&dir, OutputFormat::Json).expect("ok");
        assert!(json.contains("\"cplCount\": 1"));
        assert!(json.contains("\"mxfCount\": 1"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_extract_xml() {
        let dir = make_pkg("xml");
        let xml = MetadataExtractor::extract(&dir, OutputFormat::Xml).expect("ok");
        assert!(xml.contains("<CplCount>1</CplCount>"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_nonexistent() {
        assert!(MetadataExtractor::collect("/nonexistent/xyz").is_err());
    }
}
