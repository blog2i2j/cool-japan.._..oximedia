//! PREMIS (Preservation Metadata Implementation Strategies) support
//!
//! PREMIS is a standard for preservation metadata in digital archives.
//! See: <https://www.loc.gov/standards/premis/>

use crate::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// PREMIS object types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectType {
    /// File object
    File,
    /// Bitstream
    Bitstream,
    /// Representation
    Representation,
    /// Intellectual entity
    IntellectualEntity,
}

/// PREMIS event types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// Capture/creation
    Capture,
    /// Ingestion into repository
    Ingestion,
    /// Format migration
    Migration,
    /// Validation
    Validation,
    /// Fixity check
    FixityCheck,
    /// Replication
    Replication,
    /// Other event
    Other(String),
}

/// PREMIS object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PremisObject {
    /// Object identifier
    pub identifier: String,
    /// Object type
    pub object_type: ObjectType,
    /// Original name
    pub original_name: Option<String>,
    /// File size in bytes
    pub size: Option<u64>,
    /// Format
    pub format: Option<String>,
    /// Creation date
    pub creation_date: chrono::DateTime<chrono::Utc>,
    /// Checksums
    pub checksums: Vec<(String, String)>, // (algorithm, value)
}

impl PremisObject {
    /// Create a PREMIS object from a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn from_file(path: &Path, identifier: String) -> Result<Self> {
        let metadata = fs::metadata(path)?;
        let size = metadata.len();
        let original_name = path.file_name().and_then(|n| n.to_str()).map(String::from);

        // Detect format from extension
        let format = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| format!("fmt/{}", e.to_uppercase()));

        Ok(Self {
            identifier,
            object_type: ObjectType::File,
            original_name,
            size: Some(size),
            format,
            creation_date: chrono::Utc::now(),
            checksums: Vec::new(),
        })
    }

    /// Add a checksum
    #[must_use]
    pub fn with_checksum(mut self, algorithm: &str, value: &str) -> Self {
        self.checksums
            .push((algorithm.to_string(), value.to_string()));
        self
    }

    /// Convert to XML
    ///
    /// # Errors
    ///
    /// Returns an error if XML serialization fails
    pub fn to_xml(&self) -> Result<String> {
        let mut xml = String::new();
        xml.push_str("  <object>\n");
        xml.push_str(&format!(
            "    <objectIdentifier>{}</objectIdentifier>\n",
            escape_xml(&self.identifier)
        ));
        xml.push_str(&format!(
            "    <objectCategory>{:?}</objectCategory>\n",
            self.object_type
        ));

        if let Some(ref name) = self.original_name {
            xml.push_str(&format!(
                "    <originalName>{}</originalName>\n",
                escape_xml(name)
            ));
        }

        if let Some(size) = self.size {
            xml.push_str(&format!("    <size>{size}</size>\n"));
        }

        if let Some(ref format) = self.format {
            xml.push_str(&format!("    <format>{}</format>\n", escape_xml(format)));
        }

        for (algo, value) in &self.checksums {
            xml.push_str("    <fixity>\n");
            xml.push_str(&format!(
                "      <messageDigestAlgorithm>{}</messageDigestAlgorithm>\n",
                escape_xml(algo)
            ));
            xml.push_str(&format!(
                "      <messageDigest>{}</messageDigest>\n",
                escape_xml(value)
            ));
            xml.push_str("    </fixity>\n");
        }

        xml.push_str("  </object>\n");
        Ok(xml)
    }
}

/// PREMIS event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PremisEvent {
    /// Event identifier
    pub identifier: String,
    /// Event type
    pub event_type: EventType,
    /// Event date/time
    pub date_time: chrono::DateTime<chrono::Utc>,
    /// Event detail
    pub detail: Option<String>,
    /// Outcome
    pub outcome: Option<String>,
    /// Linking objects
    pub linking_objects: Vec<String>,
}

impl PremisEvent {
    /// Create a new PREMIS event
    #[must_use]
    pub fn new(identifier: String, event_type: EventType) -> Self {
        Self {
            identifier,
            event_type,
            date_time: chrono::Utc::now(),
            detail: None,
            outcome: None,
            linking_objects: Vec::new(),
        }
    }

    /// Set event detail
    #[must_use]
    pub fn with_detail(mut self, detail: &str) -> Self {
        self.detail = Some(detail.to_string());
        self
    }

    /// Set outcome
    #[must_use]
    pub fn with_outcome(mut self, outcome: &str) -> Self {
        self.outcome = Some(outcome.to_string());
        self
    }

    /// Add linking object
    #[must_use]
    pub fn with_linking_object(mut self, object_id: &str) -> Self {
        self.linking_objects.push(object_id.to_string());
        self
    }

    /// Convert to XML
    ///
    /// # Errors
    ///
    /// Returns an error if XML serialization fails
    pub fn to_xml(&self) -> Result<String> {
        let mut xml = String::new();
        xml.push_str("  <event>\n");
        xml.push_str(&format!(
            "    <eventIdentifier>{}</eventIdentifier>\n",
            escape_xml(&self.identifier)
        ));
        xml.push_str(&format!(
            "    <eventType>{}</eventType>\n",
            event_type_to_string(&self.event_type)
        ));
        xml.push_str(&format!(
            "    <eventDateTime>{}</eventDateTime>\n",
            self.date_time.to_rfc3339()
        ));

        if let Some(ref detail) = self.detail {
            xml.push_str(&format!(
                "    <eventDetail>{}</eventDetail>\n",
                escape_xml(detail)
            ));
        }

        if let Some(ref outcome) = self.outcome {
            xml.push_str(&"    <eventOutcomeInformation>\n".to_string());
            xml.push_str(&format!(
                "      <eventOutcome>{}</eventOutcome>\n",
                escape_xml(outcome)
            ));
            xml.push_str("    </eventOutcomeInformation>\n");
        }

        for obj_id in &self.linking_objects {
            xml.push_str(&format!(
                "    <linkingObjectIdentifier>{}</linkingObjectIdentifier>\n",
                escape_xml(obj_id)
            ));
        }

        xml.push_str("  </event>\n");
        Ok(xml)
    }
}

fn event_type_to_string(event_type: &EventType) -> String {
    match event_type {
        EventType::Capture => "capture".to_string(),
        EventType::Ingestion => "ingestion".to_string(),
        EventType::Migration => "migration".to_string(),
        EventType::Validation => "validation".to_string(),
        EventType::FixityCheck => "fixity check".to_string(),
        EventType::Replication => "replication".to_string(),
        EventType::Other(s) => s.clone(),
    }
}

/// PREMIS metadata document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PremisMetadata {
    /// Objects
    pub objects: Vec<PremisObject>,
    /// Events
    pub events: Vec<PremisEvent>,
}

impl Default for PremisMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl PremisMetadata {
    /// Create a new PREMIS metadata document
    #[must_use]
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Create PREMIS metadata for a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn for_file(path: &Path) -> Result<Self> {
        let identifier = format!("obj-{}", chrono::Utc::now().timestamp());
        let object = PremisObject::from_file(path, identifier.clone())?;

        let event = PremisEvent::new(
            format!("evt-{}", chrono::Utc::now().timestamp()),
            EventType::Capture,
        )
        .with_detail("File captured for preservation")
        .with_outcome("success")
        .with_linking_object(&identifier);

        Ok(Self {
            objects: vec![object],
            events: vec![event],
        })
    }

    /// Add an object
    #[must_use]
    pub fn with_object(mut self, object: PremisObject) -> Self {
        self.objects.push(object);
        self
    }

    /// Add an event
    #[must_use]
    pub fn with_event(mut self, event: PremisEvent) -> Self {
        self.events.push(event);
        self
    }

    /// Convert to XML
    ///
    /// # Errors
    ///
    /// Returns an error if XML serialization fails
    pub fn to_xml(&self) -> Result<String> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<premis xmlns=\"http://www.loc.gov/premis/v3\" version=\"3.0\">\n");

        for object in &self.objects {
            xml.push_str(&object.to_xml()?);
        }

        for event in &self.events {
            xml.push_str(&event.to_xml()?);
        }

        xml.push_str("</premis>\n");
        Ok(xml)
    }

    /// Save to file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written
    pub fn save(&self, path: &Path) -> Result<()> {
        let xml = self.to_xml()?;
        fs::write(path, xml)?;
        Ok(())
    }
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_premis_object_creation() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Test content")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let object = PremisObject::from_file(file.path(), "obj-001".to_string())
            .expect("operation should succeed")
            .with_checksum("SHA-256", "abc123");

        assert_eq!(object.identifier, "obj-001");
        assert_eq!(object.object_type, ObjectType::File);
        assert_eq!(object.checksums.len(), 1);
    }

    #[test]
    fn test_premis_event_creation() {
        let event = PremisEvent::new("evt-001".to_string(), EventType::Ingestion)
            .with_detail("Ingested into archive")
            .with_outcome("success")
            .with_linking_object("obj-001");

        assert_eq!(event.identifier, "evt-001");
        assert_eq!(event.event_type, EventType::Ingestion);
        assert_eq!(event.linking_objects.len(), 1);
    }

    #[test]
    fn test_premis_metadata_xml() {
        let metadata = PremisMetadata::new().with_object(PremisObject {
            identifier: "obj-001".to_string(),
            object_type: ObjectType::File,
            original_name: Some("test.mkv".to_string()),
            size: Some(1024),
            format: Some("video/x-matroska".to_string()),
            creation_date: chrono::Utc::now(),
            checksums: vec![("SHA-256".to_string(), "abc123".to_string())],
        });

        let xml = metadata.to_xml().expect("operation should succeed");
        assert!(xml.contains("<premis"));
        assert!(xml.contains("obj-001"));
        assert!(xml.contains("SHA-256"));
    }

    #[test]
    fn test_for_file() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Preservation test")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let metadata = PremisMetadata::for_file(file.path()).expect("operation should succeed");
        assert_eq!(metadata.objects.len(), 1);
        assert_eq!(metadata.events.len(), 1);
    }
}
