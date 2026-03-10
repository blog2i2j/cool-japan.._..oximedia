//! Preservation metadata generation and management
//!
//! Supports multiple metadata standards:
//! - PREMIS: Preservation Metadata Implementation Strategies
//! - METS: Metadata Encoding and Transmission Standard
//! - Dublin Core: Basic descriptive metadata

pub mod embed;
pub mod extract;
pub mod mets;
pub mod premis;

pub use embed::MetadataEmbedder;
pub use extract::MetadataExtractor;
pub use mets::{MetsBuilder, MetsDocument};
pub use premis::{PremisEvent, PremisMetadata, PremisObject};

use serde::{Deserialize, Serialize};

/// Dublin Core metadata elements
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DublinCore {
    /// Title
    pub title: Option<String>,
    /// Creator
    pub creator: Option<String>,
    /// Subject
    pub subject: Vec<String>,
    /// Description
    pub description: Option<String>,
    /// Publisher
    pub publisher: Option<String>,
    /// Contributor
    pub contributor: Vec<String>,
    /// Date
    pub date: Option<String>,
    /// Type
    pub type_: Option<String>,
    /// Format
    pub format: Option<String>,
    /// Identifier
    pub identifier: Option<String>,
    /// Source
    pub source: Option<String>,
    /// Language
    pub language: Option<String>,
    /// Relation
    pub relation: Option<String>,
    /// Coverage
    pub coverage: Option<String>,
    /// Rights
    pub rights: Option<String>,
}

impl DublinCore {
    /// Create a new Dublin Core metadata record
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the title
    #[must_use]
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = Some(title.to_string());
        self
    }

    /// Set the creator
    #[must_use]
    pub fn with_creator(mut self, creator: &str) -> Self {
        self.creator = Some(creator.to_string());
        self
    }

    /// Add a subject
    #[must_use]
    pub fn with_subject(mut self, subject: &str) -> Self {
        self.subject.push(subject.to_string());
        self
    }

    /// Set the description
    #[must_use]
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// Set the format
    #[must_use]
    pub fn with_format(mut self, format: &str) -> Self {
        self.format = Some(format.to_string());
        self
    }

    /// Set the rights
    #[must_use]
    pub fn with_rights(mut self, rights: &str) -> Self {
        self.rights = Some(rights.to_string());
        self
    }

    /// Convert to XML
    ///
    /// # Errors
    ///
    /// Returns an error if XML serialization fails
    pub fn to_xml(&self) -> crate::Result<String> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<metadata xmlns:dc=\"http://purl.org/dc/elements/1.1/\">\n");

        if let Some(ref title) = self.title {
            xml.push_str(&format!("  <dc:title>{}</dc:title>\n", escape_xml(title)));
        }
        if let Some(ref creator) = self.creator {
            xml.push_str(&format!(
                "  <dc:creator>{}</dc:creator>\n",
                escape_xml(creator)
            ));
        }
        for subject in &self.subject {
            xml.push_str(&format!(
                "  <dc:subject>{}</dc:subject>\n",
                escape_xml(subject)
            ));
        }
        if let Some(ref description) = self.description {
            xml.push_str(&format!(
                "  <dc:description>{}</dc:description>\n",
                escape_xml(description)
            ));
        }
        if let Some(ref format) = self.format {
            xml.push_str(&format!(
                "  <dc:format>{}</dc:format>\n",
                escape_xml(format)
            ));
        }
        if let Some(ref rights) = self.rights {
            xml.push_str(&format!(
                "  <dc:rights>{}</dc:rights>\n",
                escape_xml(rights)
            ));
        }

        xml.push_str("</metadata>\n");
        Ok(xml)
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

    #[test]
    fn test_dublin_core_creation() {
        let dc = DublinCore::new()
            .with_title("Test Media")
            .with_creator("Test Creator")
            .with_format("video/x-matroska");

        assert_eq!(dc.title, Some("Test Media".to_string()));
        assert_eq!(dc.creator, Some("Test Creator".to_string()));
        assert_eq!(dc.format, Some("video/x-matroska".to_string()));
    }

    #[test]
    fn test_dublin_core_xml() {
        let dc = DublinCore::new().with_title("Test").with_creator("Author");

        let xml = dc.to_xml().expect("operation should succeed");
        assert!(xml.contains("<dc:title>Test</dc:title>"));
        assert!(xml.contains("<dc:creator>Author</dc:creator>"));
    }

    #[test]
    fn test_xml_escaping() {
        assert_eq!(escape_xml("a & b"), "a &amp; b");
        assert_eq!(escape_xml("<tag>"), "&lt;tag&gt;");
    }
}
