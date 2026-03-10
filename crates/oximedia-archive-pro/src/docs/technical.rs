//! Technical documentation generation

use crate::Result;
use std::fs;
use std::path::Path;

/// Technical documentation generator
pub struct TechnicalDocGenerator;

impl Default for TechnicalDocGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalDocGenerator {
    /// Create a new technical documentation generator
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Generate technical documentation for a file
    ///
    /// # Errors
    ///
    /// Returns an error if documentation cannot be generated
    pub fn generate(&self, file_path: &Path) -> Result<String> {
        let metadata = fs::metadata(file_path)?;
        let mut doc = String::from("# Technical Specification\n\n");

        doc.push_str("## File Information\n\n");
        doc.push_str(&format!("- **Path**: `{}`\n", file_path.display()));
        doc.push_str(&format!(
            "- **Size**: {} bytes ({:.2} MB)\n",
            metadata.len(),
            metadata.len() as f64 / 1_048_576.0
        ));

        if let Some(ext) = file_path.extension() {
            doc.push_str(&format!("- **Extension**: `.{}`\n", ext.to_string_lossy()));
        }

        doc.push_str("\n## Format Specifications\n\n");
        doc.push_str("Format-specific technical details would be included here.\n\n");

        doc.push_str("## Codec Information\n\n");
        doc.push_str("Codec details would be extracted and documented here.\n\n");

        doc.push_str("## Preservation Notes\n\n");
        doc.push_str("- Long-term preservation considerations\n");
        doc.push_str("- Format viability assessment\n");
        doc.push_str("- Migration recommendations\n");

        Ok(doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_generate_technical_doc() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Test content for technical doc")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let generator = TechnicalDocGenerator::new();
        let doc = generator
            .generate(file.path())
            .expect("operation should succeed");

        assert!(doc.contains("Technical Specification"));
        assert!(doc.contains("File Information"));
    }
}
