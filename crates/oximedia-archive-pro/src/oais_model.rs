//! OAIS reference model implementation.
//!
//! Implements key concepts from ISO 14721 (Open Archival Information System),
//! including Submission, Archival and Dissemination Information Packages (SIP/AIP/DIP)
//! and a simple in-memory repository.

use serde::{Deserialize, Serialize};

/// The type of an OAIS information package.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OaisPackageType {
    /// Submission Information Package – submitted by the producer
    Sip,
    /// Archival Information Package – stored by the repository
    Aip,
    /// Dissemination Information Package – delivered to the consumer
    Dip,
}

/// Preservation Description Information (PDI) associated with a package.
///
/// Based on the five PDI categories defined in OAIS (ISO 14721:2012 §2.2.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreservationDescriptionInfo {
    /// Reference information (identifiers for the content)
    pub reference: String,
    /// Provenance information (history and chain of custody)
    pub provenance: String,
    /// Context information (relationships to other packages)
    pub context: String,
    /// Fixity information (checksum or other integrity mechanism)
    pub fixity: String,
    /// Access rights information
    pub access_rights: String,
}

impl Default for PreservationDescriptionInfo {
    fn default() -> Self {
        Self {
            reference: String::new(),
            provenance: String::new(),
            context: String::new(),
            fixity: String::new(),
            access_rights: "unrestricted".to_owned(),
        }
    }
}

impl PreservationDescriptionInfo {
    /// Returns `true` if all mandatory PDI fields are non-empty.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        !self.reference.is_empty() && !self.provenance.is_empty() && !self.fixity.is_empty()
    }
}

/// An OAIS information package (SIP, AIP, or DIP).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaisPackage {
    /// Unique package identifier
    pub id: String,
    /// Package type
    pub package_type: OaisPackageType,
    /// List of content object identifiers or paths
    pub content_info: Vec<String>,
    /// Preservation Description Information
    pub pdi: PreservationDescriptionInfo,
    /// Unix timestamp (seconds) when the package was created
    pub created_at: u64,
}

impl OaisPackage {
    /// Create a new OAIS package with default PDI.
    #[must_use]
    pub fn new(id: &str, ptype: OaisPackageType, now: u64) -> Self {
        Self {
            id: id.to_owned(),
            package_type: ptype,
            content_info: Vec::new(),
            pdi: PreservationDescriptionInfo::default(),
            created_at: now,
        }
    }

    /// Derive a DIP from this package (typically an AIP), copying all content.
    ///
    /// The DIP gets a derived identifier and its type is set to `Dip`.
    #[must_use]
    pub fn to_dip(&self) -> OaisPackage {
        OaisPackage {
            id: format!("{}-dip", self.id),
            package_type: OaisPackageType::Dip,
            content_info: self.content_info.clone(),
            pdi: self.pdi.clone(),
            created_at: self.created_at,
        }
    }

    /// Returns `true` if the package has content and complete PDI.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        !self.content_info.is_empty() && self.pdi.is_complete()
    }

    /// Add a content object to this package.
    pub fn add_content(&mut self, item: impl Into<String>) {
        self.content_info.push(item.into());
    }
}

/// A simple in-memory OAIS repository.
#[derive(Debug, Default)]
pub struct OaisRepository {
    /// All ingested packages
    pub packages: Vec<OaisPackage>,
}

impl OaisRepository {
    /// Create a new empty repository.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest a package into the repository.
    pub fn ingest(&mut self, pkg: OaisPackage) {
        self.packages.push(pkg);
    }

    /// Find a package by its ID.
    #[must_use]
    pub fn find(&self, id: &str) -> Option<&OaisPackage> {
        self.packages.iter().find(|p| p.id == id)
    }

    /// Count all Archival Information Packages (AIPs) in the repository.
    #[must_use]
    pub fn aip_count(&self) -> usize {
        self.packages
            .iter()
            .filter(|p| p.package_type == OaisPackageType::Aip)
            .count()
    }

    /// Count all Submission Information Packages (SIPs) in the repository.
    #[must_use]
    pub fn sip_count(&self) -> usize {
        self.packages
            .iter()
            .filter(|p| p.package_type == OaisPackageType::Sip)
            .count()
    }

    /// Return all packages of a specific type.
    #[must_use]
    pub fn packages_of_type(&self, ptype: OaisPackageType) -> Vec<&OaisPackage> {
        self.packages
            .iter()
            .filter(|p| p.package_type == ptype)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sip(id: &str) -> OaisPackage {
        OaisPackage::new(id, OaisPackageType::Sip, 1_700_000_000)
    }

    fn make_complete_aip(id: &str) -> OaisPackage {
        let mut pkg = OaisPackage::new(id, OaisPackageType::Aip, 1_700_000_000);
        pkg.add_content("file1.mkv");
        pkg.pdi.reference = "REF-001".to_owned();
        pkg.pdi.provenance = "Producer A".to_owned();
        pkg.pdi.fixity = "sha256:abc123".to_owned();
        pkg
    }

    #[test]
    fn test_package_new() {
        let pkg = make_sip("sip-001");
        assert_eq!(pkg.id, "sip-001");
        assert_eq!(pkg.package_type, OaisPackageType::Sip);
        assert!(pkg.content_info.is_empty());
    }

    #[test]
    fn test_package_to_dip() {
        let aip = make_complete_aip("aip-001");
        let dip = aip.to_dip();
        assert_eq!(dip.id, "aip-001-dip");
        assert_eq!(dip.package_type, OaisPackageType::Dip);
        assert_eq!(dip.content_info, aip.content_info);
    }

    #[test]
    fn test_package_is_complete_true() {
        let aip = make_complete_aip("aip-complete");
        assert!(aip.is_complete());
    }

    #[test]
    fn test_package_is_complete_false_no_content() {
        let mut pkg = OaisPackage::new("aip-empty", OaisPackageType::Aip, 0);
        pkg.pdi.reference = "R".to_owned();
        pkg.pdi.provenance = "P".to_owned();
        pkg.pdi.fixity = "F".to_owned();
        assert!(!pkg.is_complete());
    }

    #[test]
    fn test_package_is_complete_false_missing_pdi() {
        let mut pkg = OaisPackage::new("aip-no-pdi", OaisPackageType::Aip, 0);
        pkg.add_content("file.mkv");
        // PDI is incomplete by default (reference and provenance are empty)
        assert!(!pkg.is_complete());
    }

    #[test]
    fn test_pdi_is_complete() {
        let mut pdi = PreservationDescriptionInfo::default();
        pdi.reference = "R".to_owned();
        pdi.provenance = "P".to_owned();
        pdi.fixity = "F".to_owned();
        assert!(pdi.is_complete());
    }

    #[test]
    fn test_pdi_is_complete_missing_field() {
        let mut pdi = PreservationDescriptionInfo::default();
        pdi.reference = "R".to_owned();
        // provenance and fixity are empty
        assert!(!pdi.is_complete());
    }

    #[test]
    fn test_repository_ingest_and_find() {
        let mut repo = OaisRepository::new();
        repo.ingest(make_sip("sip-001"));
        let found = repo.find("sip-001");
        assert!(found.is_some());
        assert_eq!(found.expect("operation should succeed").id, "sip-001");
    }

    #[test]
    fn test_repository_find_missing() {
        let repo = OaisRepository::new();
        assert!(repo.find("does-not-exist").is_none());
    }

    #[test]
    fn test_repository_aip_count() {
        let mut repo = OaisRepository::new();
        repo.ingest(make_sip("sip-1"));
        repo.ingest(make_complete_aip("aip-1"));
        repo.ingest(make_complete_aip("aip-2"));
        assert_eq!(repo.aip_count(), 2);
    }

    #[test]
    fn test_repository_sip_count() {
        let mut repo = OaisRepository::new();
        repo.ingest(make_sip("sip-1"));
        repo.ingest(make_sip("sip-2"));
        repo.ingest(make_complete_aip("aip-1"));
        assert_eq!(repo.sip_count(), 2);
    }

    #[test]
    fn test_repository_packages_of_type_dip() {
        let mut repo = OaisRepository::new();
        let aip = make_complete_aip("aip-1");
        let dip = aip.to_dip();
        repo.ingest(aip);
        repo.ingest(dip);
        let dips = repo.packages_of_type(OaisPackageType::Dip);
        assert_eq!(dips.len(), 1);
    }

    #[test]
    fn test_repository_empty_counts() {
        let repo = OaisRepository::new();
        assert_eq!(repo.aip_count(), 0);
        assert_eq!(repo.sip_count(), 0);
    }
}
