#![allow(dead_code)]

//! Submission Information Package (SIP) builder for OAIS-compliant archival workflows.
//!
//! This module provides a fluent builder for constructing SIPs that conform to the
//! Open Archival Information System (OAIS) reference model. A SIP is the package
//! submitted by a producer to an archive for ingest.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Represents the type of content within a SIP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentCategory {
    /// Video media content.
    Video,
    /// Audio media content.
    Audio,
    /// Image media content.
    Image,
    /// Document content.
    Document,
    /// Mixed or other content.
    Mixed,
}

impl ContentCategory {
    /// Returns a human-readable label for the category.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Video => "video",
            Self::Audio => "audio",
            Self::Image => "image",
            Self::Document => "document",
            Self::Mixed => "mixed",
        }
    }
}

/// Describes a single file entry within a SIP.
#[derive(Debug, Clone)]
pub struct SipFileEntry {
    /// Relative path within the package.
    pub relative_path: PathBuf,
    /// Original source path on disk.
    pub source_path: PathBuf,
    /// Size in bytes.
    pub size_bytes: u64,
    /// SHA-256 checksum hex string.
    pub checksum_sha256: String,
    /// MIME type of the file.
    pub mime_type: String,
    /// Content category classification.
    pub category: ContentCategory,
}

/// Describes producer information for the SIP.
#[derive(Debug, Clone)]
pub struct ProducerInfo {
    /// Name of the producing organization or individual.
    pub name: String,
    /// Contact email address.
    pub contact_email: String,
    /// Identifier for the producer within the archive system.
    pub producer_id: String,
}

impl ProducerInfo {
    /// Creates a new `ProducerInfo` with the given name.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            contact_email: String::new(),
            producer_id: String::new(),
        }
    }

    /// Sets the contact email.
    #[must_use]
    pub fn with_email(mut self, email: &str) -> Self {
        self.contact_email = email.to_string();
        self
    }

    /// Sets the producer identifier.
    #[must_use]
    pub fn with_id(mut self, id: &str) -> Self {
        self.producer_id = id.to_string();
        self
    }

    /// Returns whether the producer info is considered complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        !self.name.is_empty() && !self.contact_email.is_empty() && !self.producer_id.is_empty()
    }
}

/// Transfer agreement status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferAgreementStatus {
    /// No agreement exists yet.
    None,
    /// Agreement is pending review.
    Pending,
    /// Agreement has been approved.
    Approved,
    /// Agreement was rejected.
    Rejected,
}

/// Builder for constructing Submission Information Packages.
#[derive(Debug, Clone)]
pub struct SipBuilder {
    /// Unique package identifier.
    package_id: String,
    /// Title of the package.
    title: String,
    /// Producer information.
    producer: Option<ProducerInfo>,
    /// Files included in the SIP.
    files: Vec<SipFileEntry>,
    /// Descriptive metadata key-value pairs.
    descriptive_metadata: HashMap<String, String>,
    /// Transfer agreement status.
    agreement_status: TransferAgreementStatus,
    /// Submission timestamp.
    submission_time: Option<SystemTime>,
    /// Maximum allowed package size in bytes (0 = unlimited).
    max_size_bytes: u64,
}

impl SipBuilder {
    /// Creates a new `SipBuilder` with the given package identifier.
    #[must_use]
    pub fn new(package_id: &str) -> Self {
        Self {
            package_id: package_id.to_string(),
            title: String::new(),
            producer: None,
            files: Vec::new(),
            descriptive_metadata: HashMap::new(),
            agreement_status: TransferAgreementStatus::None,
            submission_time: None,
            max_size_bytes: 0,
        }
    }

    /// Sets the package title.
    #[must_use]
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    /// Sets the producer information.
    #[must_use]
    pub fn with_producer(mut self, producer: ProducerInfo) -> Self {
        self.producer = Some(producer);
        self
    }

    /// Sets the transfer agreement status.
    #[must_use]
    pub fn with_agreement(mut self, status: TransferAgreementStatus) -> Self {
        self.agreement_status = status;
        self
    }

    /// Sets the maximum allowed package size in bytes.
    #[must_use]
    pub fn with_max_size(mut self, max_bytes: u64) -> Self {
        self.max_size_bytes = max_bytes;
        self
    }

    /// Adds a descriptive metadata field.
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.descriptive_metadata
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Adds a file entry to the SIP.
    pub fn add_file(&mut self, entry: SipFileEntry) -> &mut Self {
        self.files.push(entry);
        self
    }

    /// Returns the total size of all files in bytes.
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size_bytes).sum()
    }

    /// Returns the number of files in the package.
    #[must_use]
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Checks whether the SIP exceeds the maximum size limit.
    #[must_use]
    pub fn exceeds_size_limit(&self) -> bool {
        self.max_size_bytes > 0 && self.total_size_bytes() > self.max_size_bytes
    }

    /// Validates the SIP builder state and returns a list of validation issues.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.package_id.is_empty() {
            issues.push("Package ID is required".to_string());
        }
        if self.title.is_empty() {
            issues.push("Package title is required".to_string());
        }
        if self.producer.is_none() {
            issues.push("Producer information is required".to_string());
        }
        if self.files.is_empty() {
            issues.push("At least one file is required".to_string());
        }
        if self.agreement_status != TransferAgreementStatus::Approved {
            issues.push("Transfer agreement must be approved".to_string());
        }
        if self.exceeds_size_limit() {
            issues.push(format!(
                "Package size {} exceeds limit {}",
                self.total_size_bytes(),
                self.max_size_bytes
            ));
        }

        // Check for duplicate relative paths
        let mut seen_paths = std::collections::HashSet::new();
        for file in &self.files {
            if !seen_paths.insert(&file.relative_path) {
                issues.push(format!(
                    "Duplicate relative path: {}",
                    file.relative_path.display()
                ));
            }
        }

        issues
    }

    /// Builds the finalized SIP descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error string if validation fails.
    pub fn build(mut self) -> Result<SipDescriptor, String> {
        let issues = self.validate();
        if !issues.is_empty() {
            return Err(issues.join("; "));
        }

        self.submission_time = Some(SystemTime::now());

        Ok(SipDescriptor {
            package_id: self.package_id,
            title: self.title,
            producer: self.producer.expect("producer validated: is_none check passed above"),
            files: self.files,
            descriptive_metadata: self.descriptive_metadata,
            agreement_status: self.agreement_status,
            submission_time: self.submission_time.expect("submission_time set on line above"),
            total_size_bytes: self.total_size_bytes(),
        })
    }
}

/// A finalized Submission Information Package descriptor.
#[derive(Debug, Clone)]
pub struct SipDescriptor {
    /// Unique package identifier.
    pub package_id: String,
    /// Title of the package.
    pub title: String,
    /// Producer information.
    pub producer: ProducerInfo,
    /// Files included in the SIP.
    pub files: Vec<SipFileEntry>,
    /// Descriptive metadata key-value pairs.
    pub descriptive_metadata: HashMap<String, String>,
    /// Transfer agreement status.
    pub agreement_status: TransferAgreementStatus,
    /// When the SIP was submitted.
    pub submission_time: SystemTime,
    /// Total size of all files in bytes.
    pub total_size_bytes: u64,
}

impl SipDescriptor {
    /// Returns the content categories present in the package.
    #[must_use]
    pub fn content_categories(&self) -> Vec<ContentCategory> {
        let mut categories: Vec<ContentCategory> = self
            .files
            .iter()
            .map(|f| f.category)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        categories.sort_by_key(|c| c.label());
        categories
    }

    /// Looks up a file by its relative path.
    #[must_use]
    pub fn find_file(&self, relative_path: &Path) -> Option<&SipFileEntry> {
        self.files.iter().find(|f| f.relative_path == relative_path)
    }
}

impl std::hash::Hash for ContentCategory {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
    }
}

impl std::cmp::Eq for SipFileEntry {}

impl std::cmp::PartialEq for SipFileEntry {
    fn eq(&self, other: &Self) -> bool {
        self.relative_path == other.relative_path
    }
}

/// Implement `Eq` for `ContentCategory` is already derived above via the derive macro,
/// but we need `Hash` for `HashSet` usage — done above.

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_file(name: &str, size: u64, cat: ContentCategory) -> SipFileEntry {
        SipFileEntry {
            relative_path: PathBuf::from(name),
            source_path: PathBuf::from(format!("/source/{name}")),
            size_bytes: size,
            checksum_sha256: "abcdef1234567890".to_string(),
            mime_type: "application/octet-stream".to_string(),
            category: cat,
        }
    }

    fn valid_builder() -> SipBuilder {
        let mut builder = SipBuilder::new("SIP-001")
            .with_title("Test Package")
            .with_producer(
                ProducerInfo::new("Test Org")
                    .with_email("test@example.com")
                    .with_id("PROD-1"),
            )
            .with_agreement(TransferAgreementStatus::Approved);
        builder.add_file(sample_file("video.mkv", 1_000_000, ContentCategory::Video));
        builder
    }

    #[test]
    fn test_content_category_label() {
        assert_eq!(ContentCategory::Video.label(), "video");
        assert_eq!(ContentCategory::Audio.label(), "audio");
        assert_eq!(ContentCategory::Image.label(), "image");
        assert_eq!(ContentCategory::Document.label(), "document");
        assert_eq!(ContentCategory::Mixed.label(), "mixed");
    }

    #[test]
    fn test_producer_info_new() {
        let p = ProducerInfo::new("Org");
        assert_eq!(p.name, "Org");
        assert!(p.contact_email.is_empty());
        assert!(!p.is_complete());
    }

    #[test]
    fn test_producer_info_complete() {
        let p = ProducerInfo::new("Org")
            .with_email("a@b.com")
            .with_id("ID-1");
        assert!(p.is_complete());
    }

    #[test]
    fn test_sip_builder_total_size() {
        let mut builder = SipBuilder::new("SIP-001");
        builder.add_file(sample_file("a.mkv", 500, ContentCategory::Video));
        builder.add_file(sample_file("b.wav", 300, ContentCategory::Audio));
        assert_eq!(builder.total_size_bytes(), 800);
        assert_eq!(builder.file_count(), 2);
    }

    #[test]
    fn test_sip_builder_size_limit() {
        let mut builder = SipBuilder::new("SIP-001").with_max_size(100);
        builder.add_file(sample_file("big.mkv", 200, ContentCategory::Video));
        assert!(builder.exceeds_size_limit());
    }

    #[test]
    fn test_sip_builder_no_size_limit() {
        let mut builder = SipBuilder::new("SIP-001").with_max_size(0);
        builder.add_file(sample_file("big.mkv", 999_999_999, ContentCategory::Video));
        assert!(!builder.exceeds_size_limit());
    }

    #[test]
    fn test_validate_empty_builder() {
        let builder = SipBuilder::new("");
        let issues = builder.validate();
        assert!(issues.iter().any(|i| i.contains("Package ID")));
        assert!(issues.iter().any(|i| i.contains("title")));
        assert!(issues.iter().any(|i| i.contains("Producer")));
        assert!(issues.iter().any(|i| i.contains("file")));
        assert!(issues.iter().any(|i| i.contains("agreement")));
    }

    #[test]
    fn test_validate_valid_builder() {
        let builder = valid_builder();
        let issues = builder.validate();
        assert!(issues.is_empty(), "unexpected issues: {issues:?}");
    }

    #[test]
    fn test_build_success() {
        let builder = valid_builder();
        let sip = builder.build().expect("build should succeed");
        assert_eq!(sip.package_id, "SIP-001");
        assert_eq!(sip.title, "Test Package");
        assert_eq!(sip.files.len(), 1);
        assert_eq!(sip.total_size_bytes, 1_000_000);
    }

    #[test]
    fn test_build_failure_missing_title() {
        let mut builder = SipBuilder::new("SIP-001")
            .with_producer(
                ProducerInfo::new("Org")
                    .with_email("a@b.com")
                    .with_id("ID-1"),
            )
            .with_agreement(TransferAgreementStatus::Approved);
        builder.add_file(sample_file("a.mkv", 100, ContentCategory::Video));
        assert!(builder.build().is_err());
    }

    #[test]
    fn test_sip_descriptor_content_categories() {
        let mut builder = valid_builder();
        builder.add_file(sample_file("audio.wav", 500, ContentCategory::Audio));
        let sip = builder.build().expect("operation should succeed");
        let cats = sip.content_categories();
        assert_eq!(cats.len(), 2);
    }

    #[test]
    fn test_sip_descriptor_find_file() {
        let sip = valid_builder().build().expect("operation should succeed");
        assert!(sip.find_file(Path::new("video.mkv")).is_some());
        assert!(sip.find_file(Path::new("nonexistent.txt")).is_none());
    }

    #[test]
    fn test_metadata_in_builder() {
        let builder = SipBuilder::new("SIP-001")
            .with_metadata("dc:title", "My Film")
            .with_metadata("dc:creator", "Author");
        assert_eq!(builder.descriptive_metadata.len(), 2);
    }
}
