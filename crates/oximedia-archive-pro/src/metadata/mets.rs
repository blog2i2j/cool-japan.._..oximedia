//! METS (Metadata Encoding and Transmission Standard) support
//!
//! METS is a standard for encoding descriptive, administrative, and structural metadata
//! for digital library objects.

use crate::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// METS file section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetsFile {
    /// File ID
    pub id: String,
    /// File location
    pub location: PathBuf,
    /// MIME type
    pub mime_type: Option<String>,
    /// File size
    pub size: Option<u64>,
    /// Checksums
    pub checksums: Vec<(String, String)>,
}

/// METS structural map division
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetsDiv {
    /// Division ID
    pub id: String,
    /// Label
    pub label: Option<String>,
    /// Type
    pub type_: Option<String>,
    /// File pointers
    pub file_pointers: Vec<String>,
    /// Sub-divisions
    pub subdivisions: Vec<MetsDiv>,
}

/// METS document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetsDocument {
    /// Object ID
    pub object_id: String,
    /// Title
    pub title: Option<String>,
    /// Files
    pub files: Vec<MetsFile>,
    /// Structural map
    pub struct_map: Option<MetsDiv>,
}

impl Default for MetsDocument {
    fn default() -> Self {
        Self::new("default-object")
    }
}

impl MetsDocument {
    /// Create a new METS document
    #[must_use]
    pub fn new(object_id: &str) -> Self {
        Self {
            object_id: object_id.to_string(),
            title: None,
            files: Vec::new(),
            struct_map: None,
        }
    }

    /// Set the title
    #[must_use]
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = Some(title.to_string());
        self
    }

    /// Add a file
    #[must_use]
    pub fn with_file(mut self, file: MetsFile) -> Self {
        self.files.push(file);
        self
    }

    /// Set structural map
    #[must_use]
    pub fn with_struct_map(mut self, struct_map: MetsDiv) -> Self {
        self.struct_map = Some(struct_map);
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
        xml.push_str("<mets xmlns=\"http://www.loc.gov/METS/\" ");
        xml.push_str("xmlns:xlink=\"http://www.w3.org/1999/xlink\" ");
        xml.push_str(&format!("OBJID=\"{}\">\n", escape_xml(&self.object_id)));

        // Header
        xml.push_str("  <metsHdr>\n");
        if let Some(ref title) = self.title {
            xml.push_str(&format!(
                "    <altRecordID>{}</altRecordID>\n",
                escape_xml(title)
            ));
        }
        xml.push_str("  </metsHdr>\n");

        // File section
        if !self.files.is_empty() {
            xml.push_str("  <fileSec>\n");
            xml.push_str("    <fileGrp>\n");
            for file in &self.files {
                xml.push_str(&format!("      <file ID=\"{}\"", escape_xml(&file.id)));
                if let Some(ref mime) = file.mime_type {
                    xml.push_str(&format!(" MIMETYPE=\"{}\"", escape_xml(mime)));
                }
                if let Some(size) = file.size {
                    xml.push_str(&format!(" SIZE=\"{size}\""));
                }
                xml.push_str(">\n");

                for (algo, value) in &file.checksums {
                    xml.push_str(&format!(
                        "        <checksum CHECKSUMTYPE=\"{}\">{}</checksum>\n",
                        escape_xml(algo),
                        escape_xml(value)
                    ));
                }

                xml.push_str(&format!(
                    "        <FLocat LOCTYPE=\"URL\" xlink:href=\"{}\"/>\n",
                    escape_xml(&file.location.to_string_lossy())
                ));
                xml.push_str("      </file>\n");
            }
            xml.push_str("    </fileGrp>\n");
            xml.push_str("  </fileSec>\n");
        }

        // Structural map
        if let Some(ref div) = self.struct_map {
            xml.push_str("  <structMap>\n");
            xml.push_str(&render_div(div, 2)?);
            xml.push_str("  </structMap>\n");
        }

        xml.push_str("</mets>\n");
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

fn render_div(div: &MetsDiv, indent: usize) -> Result<String> {
    let mut xml = String::new();
    let spaces = " ".repeat(indent);

    xml.push_str(&format!("{}<div ID=\"{}\"", spaces, escape_xml(&div.id)));
    if let Some(ref label) = div.label {
        xml.push_str(&format!(" LABEL=\"{}\"", escape_xml(label)));
    }
    if let Some(ref type_) = div.type_ {
        xml.push_str(&format!(" TYPE=\"{}\"", escape_xml(type_)));
    }
    xml.push_str(">\n");

    for fptr in &div.file_pointers {
        xml.push_str(&format!(
            "{}  <fptr FILEID=\"{}\"/>\n",
            spaces,
            escape_xml(fptr)
        ));
    }

    for subdiv in &div.subdivisions {
        xml.push_str(&render_div(subdiv, indent + 2)?);
    }

    xml.push_str(&format!("{spaces}</div>\n"));
    Ok(xml)
}

/// METS builder
pub struct MetsBuilder {
    document: MetsDocument,
}

impl MetsBuilder {
    /// Create a new METS builder
    #[must_use]
    pub fn new(object_id: &str) -> Self {
        Self {
            document: MetsDocument::new(object_id),
        }
    }

    /// Set the title
    #[must_use]
    pub fn with_title(mut self, title: &str) -> Self {
        self.document = self.document.with_title(title);
        self
    }

    /// Add a file
    #[must_use]
    pub fn add_file(mut self, id: &str, location: PathBuf, mime_type: Option<String>) -> Self {
        self.document = self.document.with_file(MetsFile {
            id: id.to_string(),
            location,
            mime_type,
            size: None,
            checksums: Vec::new(),
        });
        self
    }

    /// Build the METS document
    #[must_use]
    pub fn build(self) -> MetsDocument {
        self.document
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
    fn test_mets_document_creation() {
        let doc = MetsDocument::new("obj-001").with_title("Test Object");

        assert_eq!(doc.object_id, "obj-001");
        assert_eq!(doc.title, Some("Test Object".to_string()));
    }

    #[test]
    fn test_mets_with_files() {
        let doc = MetsBuilder::new("obj-001")
            .add_file(
                "file-1",
                PathBuf::from("video.mkv"),
                Some("video/x-matroska".to_string()),
            )
            .build();

        assert_eq!(doc.files.len(), 1);
        assert_eq!(doc.files[0].id, "file-1");
    }

    #[test]
    fn test_mets_xml() {
        let doc = MetsDocument::new("obj-001").with_title("Test");

        let xml = doc.to_xml().expect("operation should succeed");
        assert!(xml.contains("<mets"));
        assert!(xml.contains("obj-001"));
    }
}
