//! Archival IMF package support with long-term preservation metadata (OAIS).
//!
//! [`ImfArchive`] wraps an IMF package with OAIS-compliant provenance, fixity,
//! and preservation metadata for long-term digital preservation (LTDP).
//!
//! # Example
//! ```no_run
//! use oximedia_imf::imf_archive::{ImfArchive, OaisPackageType};
//!
//! let archive = ImfArchive::new("/path/to/imp", OaisPackageType::Aip)
//!     .with_originator("Production Studio")
//!     .with_rights("CC-BY-4.0");
//! archive.write_manifest("/path/to/imp/ARCHIVE_MANIFEST.xml").expect("write ok");
//! ```

#![allow(dead_code, missing_docs)]

use crate::{ImfError, ImfResult};
use std::path::Path;

/// OAIS information package type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OaisPackageType {
    Sip,
    Aip,
    Dip,
}

impl std::fmt::Display for OaisPackageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sip => write!(f, "SIP"),
            Self::Aip => write!(f, "AIP"),
            Self::Dip => write!(f, "DIP"),
        }
    }
}

/// A preservation event in the OAIS history.
#[derive(Debug, Clone)]
pub struct PreservationEvent {
    pub date_time: String,
    pub agent: String,
    pub event_type: String,
    pub outcome: String,
    pub note: Option<String>,
}

impl PreservationEvent {
    #[must_use]
    pub fn new(
        date_time: impl Into<String>,
        agent: impl Into<String>,
        event_type: impl Into<String>,
        outcome: impl Into<String>,
    ) -> Self {
        Self {
            date_time: date_time.into(),
            agent: agent.into(),
            event_type: event_type.into(),
            outcome: outcome.into(),
            note: None,
        }
    }

    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }
}

/// An IMF archival package with long-term preservation metadata.
pub struct ImfArchive {
    pub package_path: String,
    pub package_type: OaisPackageType,
    pub originator: Option<String>,
    pub rights: Option<String>,
    pub content_identifier: Option<String>,
    pub languages: Vec<String>,
    pub preservation_events: Vec<PreservationEvent>,
    pub custom_metadata: Vec<(String, String)>,
}

impl ImfArchive {
    #[must_use]
    pub fn new(path: impl Into<String>, package_type: OaisPackageType) -> Self {
        Self {
            package_path: path.into(),
            package_type,
            originator: None,
            rights: None,
            content_identifier: None,
            languages: Vec::new(),
            preservation_events: Vec::new(),
            custom_metadata: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_originator(mut self, o: impl Into<String>) -> Self {
        self.originator = Some(o.into());
        self
    }

    #[must_use]
    pub fn with_rights(mut self, r: impl Into<String>) -> Self {
        self.rights = Some(r.into());
        self
    }

    #[must_use]
    pub fn with_content_identifier(mut self, id: impl Into<String>) -> Self {
        self.content_identifier = Some(id.into());
        self
    }

    #[must_use]
    pub fn add_language(mut self, lang: impl Into<String>) -> Self {
        self.languages.push(lang.into());
        self
    }

    pub fn record_event(&mut self, event: PreservationEvent) {
        self.preservation_events.push(event);
    }

    pub fn add_custom(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.custom_metadata.push((key.into(), value.into()));
    }

    #[must_use]
    pub fn to_xml(&self) -> String {
        let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<ArchiveManifest>\n");
        xml.push_str(&format!("  <PackagePath>{}</PackagePath>\n", xe(&self.package_path)));
        xml.push_str(&format!("  <OaisType>{}</OaisType>\n", self.package_type));
        if let Some(ref o) = self.originator {
            xml.push_str(&format!("  <Originator>{}</Originator>\n", xe(o)));
        }
        if let Some(ref r) = self.rights {
            xml.push_str(&format!("  <Rights>{}</Rights>\n", xe(r)));
        }
        if let Some(ref id) = self.content_identifier {
            xml.push_str(&format!("  <ContentIdentifier>{}</ContentIdentifier>\n", xe(id)));
        }
        if !self.languages.is_empty() {
            xml.push_str("  <Languages>\n");
            for lang in &self.languages {
                xml.push_str(&format!("    <Language>{}</Language>\n", xe(lang)));
            }
            xml.push_str("  </Languages>\n");
        }
        if !self.preservation_events.is_empty() {
            xml.push_str("  <PreservationHistory>\n");
            for ev in &self.preservation_events {
                xml.push_str(&format!(
                    "    <Event><DateTime>{}</DateTime><Agent>{}</Agent>\
                     <Type>{}</Type><Outcome>{}</Outcome>{}</Event>\n",
                    xe(&ev.date_time), xe(&ev.agent), xe(&ev.event_type), xe(&ev.outcome),
                    ev.note.as_ref().map(|n| format!("<Note>{}</Note>", xe(n))).unwrap_or_default()
                ));
            }
            xml.push_str("  </PreservationHistory>\n");
        }
        if !self.custom_metadata.is_empty() {
            xml.push_str("  <CustomMetadata>\n");
            for (k, v) in &self.custom_metadata {
                xml.push_str(&format!("    <Entry key=\"{}\">{}</Entry>\n", xa(k), xe(v)));
            }
            xml.push_str("  </CustomMetadata>\n");
        }
        xml.push_str("</ArchiveManifest>\n");
        xml
    }

    /// Write the archive manifest XML to a file.
    ///
    /// # Errors
    /// Returns `ImfError::Io` on write failure.
    pub fn write_manifest(&self, dest: impl AsRef<Path>) -> ImfResult<()> {
        std::fs::write(dest, self.to_xml()).map_err(|e| ImfError::Other(e.to_string()))
    }

    /// Read an `ARCHIVE_MANIFEST.xml` from `path`.
    ///
    /// # Errors
    /// Returns error if file cannot be read or is not a manifest.
    pub fn read_manifest(path: impl AsRef<Path>) -> ImfResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| ImfError::Other(e.to_string()))?;
        if !content.contains("<ArchiveManifest") {
            return Err(ImfError::InvalidPackage("Not an ArchiveManifest document".to_string()));
        }
        let package_path = extract_tag(&content, "PackagePath").unwrap_or_default();
        let pt_str = extract_tag(&content, "OaisType").unwrap_or_default();
        let package_type = match pt_str.as_str() {
            "SIP" => OaisPackageType::Sip,
            "DIP" => OaisPackageType::Dip,
            _ => OaisPackageType::Aip,
        };
        Ok(Self {
            package_path,
            package_type,
            originator: extract_tag(&content, "Originator"),
            rights: extract_tag(&content, "Rights"),
            content_identifier: extract_tag(&content, "ContentIdentifier"),
            languages: Vec::new(),
            preservation_events: Vec::new(),
            custom_metadata: Vec::new(),
        })
    }
}

fn xe(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}
fn xa(s: &str) -> String { xe(s).replace('"', "&quot;") }
fn extract_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)? + start;
    Some(xml[start..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xml_roundtrip() {
        let dir = std::env::temp_dir().join("oximedia_archive_rt");
        std::fs::create_dir_all(&dir).ok();
        let p = dir.join("ARCHIVE_MANIFEST.xml");
        let archive = ImfArchive::new("/pkg", OaisPackageType::Sip)
            .with_originator("TestOrg");
        archive.write_manifest(&p).expect("write ok");
        let loaded = ImfArchive::read_manifest(&p).expect("read ok");
        assert_eq!(loaded.package_path, "/pkg");
        assert_eq!(loaded.package_type, OaisPackageType::Sip);
        assert_eq!(loaded.originator.as_deref(), Some("TestOrg"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_preservation_event() {
        let mut archive = ImfArchive::new("/pkg", OaisPackageType::Aip);
        archive.record_event(PreservationEvent::new("2024-01-01", "Bot", "Ingest", "OK"));
        let xml = archive.to_xml();
        assert!(xml.contains("<Agent>Bot</Agent>"));
    }

    #[test]
    fn test_custom_metadata() {
        let mut archive = ImfArchive::new("/pkg", OaisPackageType::Dip);
        archive.add_custom("Genre", "Drama");
        let xml = archive.to_xml();
        assert!(xml.contains("Drama"));
    }

    #[test]
    fn test_languages() {
        let archive = ImfArchive::new("/pkg", OaisPackageType::Aip)
            .add_language("en").add_language("fr");
        let xml = archive.to_xml();
        assert!(xml.contains("<Language>en</Language>"));
        assert!(xml.contains("<Language>fr</Language>"));
    }

    #[test]
    fn test_oais_display() {
        assert_eq!(format!("{}", OaisPackageType::Sip), "SIP");
        assert_eq!(format!("{}", OaisPackageType::Aip), "AIP");
        assert_eq!(format!("{}", OaisPackageType::Dip), "DIP");
    }

    #[test]
    fn test_content_identifier() {
        let archive = ImfArchive::new("/pkg", OaisPackageType::Aip)
            .with_content_identifier("urn:uuid:test-1234");
        let xml = archive.to_xml();
        assert!(xml.contains("<ContentIdentifier>urn:uuid:test-1234</ContentIdentifier>"));
    }

    #[test]
    fn test_xml_escaping_in_paths() {
        let archive = ImfArchive::new("/pkg & path <test>", OaisPackageType::Aip);
        let xml = archive.to_xml();
        assert!(xml.contains("&amp;"));
        assert!(xml.contains("&lt;"));
        assert!(!xml.contains("<test>"));
    }

    #[test]
    fn test_read_manifest_invalid_content() {
        let dir = std::env::temp_dir().join("oximedia_archive_inv");
        std::fs::create_dir_all(&dir).ok();
        let p = dir.join("NOT_MANIFEST.xml");
        std::fs::write(&p, b"<SomethingElse/>").expect("write ok");
        let result = ImfArchive::read_manifest(&p);
        assert!(result.is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_preservation_event_with_note() {
        let mut archive = ImfArchive::new("/pkg", OaisPackageType::Aip);
        archive.record_event(
            PreservationEvent::new("2024-06-01", "Validator", "QualityCheck", "Pass")
                .with_note("All checksums verified"),
        );
        let xml = archive.to_xml();
        assert!(xml.contains("<Note>All checksums verified</Note>"));
    }

    #[test]
    fn test_multiple_events_ordering() {
        let mut archive = ImfArchive::new("/pkg", OaisPackageType::Sip);
        archive.record_event(PreservationEvent::new("2024-01-01", "A", "Ingest", "OK"));
        archive.record_event(PreservationEvent::new("2024-06-15", "B", "Validate", "OK"));
        let xml = archive.to_xml();
        let pos_a = xml.find("<Agent>A</Agent>").expect("agent A present");
        let pos_b = xml.find("<Agent>B</Agent>").expect("agent B present");
        assert!(pos_a < pos_b, "events should appear in insertion order");
    }
}
